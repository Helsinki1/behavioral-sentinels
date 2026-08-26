"""Coordinator for frozen BFCL experiment-12 runs.

This module deliberately keeps BFCL semantics in :mod:`bfcl_runner12` and run
identity/budget semantics in :mod:`runner12`.  It adds only the workflow around
those two layers: freeze an official public selection, initialize a declared
run, validate the selection again at execution time, and schedule trajectories
and clean-shadow quizzes without permitting an unsafe partial retry.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

from experiments12.bfcl_runner12 import (
    BFCLBridge,
    BFCLRunnerConfig,
    BFCLTransport,
    freeze_bfcl_public_task_manifest,
    generate_bfcl_passive_quiz,
    run_bfcl_task,
)
from experiments12.core.artifacts import atomic_write_jsonl, read_json, sha256_file
from experiments12.core.budget import BudgetError, BudgetLedger
from experiments12.core.transport import Transport
from experiments12.cli12 import _confirmatory_analysis_lock, _environment
from experiments12.domains.bfcl import (
    BFCLAdapter,
    BFCLTaskRecord,
    LICENSE_IDENTIFIER as BFCL_LICENSE_IDENTIFIER,
    PINNED_COMMIT as BFCL_PINNED_COMMIT,
    V4_FUNCTION_DOC_FILES,
    V4_MULTI_TURN_CATEGORIES,
    V4_OFFICIAL_SOURCE_FILES,
)
from experiments12.domains.base import InputArtifact
from experiments12.execution_sharding12 import (
    ExecutionShard,
    add_execution_shard_arguments,
)
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.models12 import TARGET_MODEL_NAMES
from experiments12.passive_spec12 import (
    assert_passive_runtime_overrides,
    passive_monitor_spec_from_manifest,
)
from experiments12.pairing12 import JobCell, TaskRef, make_pair_manifest
from experiments12.paths12 import REPOSITORY_ROOT, RUNS_ROOT
from experiments12.planning_lock12 import (
    ScientificLaunchBinding,
    assert_scientific_launch,
)
from experiments12.runner12 import (
    RunSummary,
    _job_state,
    _stage_ledger,
    _validate_run_inputs,
    load_task_manifest,
    pair_task_id,
)
from experiments12.shadow12 import score_clean_trajectory
from experiments12.source_registry12 import SourceAllocationBinding
from experiments12.spec12 import ARMS, OPERATIONAL_PROVIDER_USD, Stage


BFCL_RUN_COORDINATOR_VERSION = 2
DEFAULT_ARTIFACTS_ROOT = RUNS_ROOT
DEFAULT_BRIDGE_SCRIPT = Path(__file__).with_name("bfcl_bridge12.py")
DEFAULT_ENV_FILE = REPOSITORY_ROOT / ".env"

BFCL_EXPECTED_INPUT_ROLES = frozenset(
    {
        f"bfcl:license:{BFCL_LICENSE_IDENTIFIER}",
        "bfcl:bridge",
        *(f"bfcl:v4:{category}" for category in V4_MULTI_TURN_CATEGORIES),
        *(
            f"bfcl:v4:possible_answer:{category}"
            for category in V4_MULTI_TURN_CATEGORIES
        ),
        *(
            f"bfcl:v4:function_doc:{class_name}"
            for class_name, _path in V4_FUNCTION_DOC_FILES
        ),
        *(
            f"bfcl:v4:official_source:{name}"
            for name, _path in V4_OFFICIAL_SOURCE_FILES
        ),
    }
)


class LoadableBFCLBridge(BFCLBridge, Protocol):
    """The official bridge surface used by this coordinator."""

    def load_tasks(
        self,
        *,
        categories: Sequence[str] = V4_MULTI_TURN_CATEGORIES,
        task_ids: Sequence[str] = (),
    ) -> tuple[BFCLTaskRecord, ...]: ...


def bfcl_provenance_receipts(
    artifacts: Sequence[InputArtifact],
    *,
    repository_root: Path | None = None,
) -> tuple[ArtifactReceipt, ...]:
    """Freeze every BFCL datum, evaluator source, and bridge file by content."""

    root = (repository_root or REPOSITORY_ROOT).resolve()
    by_role = {artifact.role: artifact for artifact in artifacts}
    if len(by_role) != len(artifacts) or set(by_role) != BFCL_EXPECTED_INPUT_ROLES:
        missing = sorted(BFCL_EXPECTED_INPUT_ROLES.difference(by_role))
        extra = sorted(set(by_role).difference(BFCL_EXPECTED_INPUT_ROLES))
        raise ValueError(f"BFCL provenance input mismatch: missing={missing}, extra={extra}")
    receipts: list[ArtifactReceipt] = []
    for role in sorted(by_role):
        artifact = by_role[role]
        receipt = ArtifactReceipt.from_file(
            name=role,
            path=artifact.path,
            workspace=root,
            upstream_commit=BFCL_PINNED_COMMIT,
            license_id=BFCL_LICENSE_IDENTIFIER,
        )
        if receipt.sha256 != artifact.sha256:
            raise ValueError(f"BFCL input changed after adapter audit: {role}")
        receipts.append(receipt)
    return tuple(receipts)


def _validate_bfcl_receipts(receipts: Sequence[ArtifactReceipt]) -> None:
    names = [receipt.name for receipt in receipts]
    if len(names) != len(set(names)) or set(names) != BFCL_EXPECTED_INPUT_ROLES:
        missing = sorted(BFCL_EXPECTED_INPUT_ROLES.difference(names))
        extra = sorted(set(names).difference(BFCL_EXPECTED_INPUT_ROLES))
        raise ValueError(f"BFCL provenance receipt mismatch: missing={missing}, extra={extra}")
    for receipt in receipts:
        if (
            receipt.upstream_commit != BFCL_PINNED_COMMIT
            or receipt.license_id != BFCL_LICENSE_IDENTIFIER
        ):
            raise ValueError(f"BFCL provenance metadata mismatch: {receipt.name}")


def _manifest_bfcl_receipts(manifest: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for item in manifest.get("benchmark_receipts", ()):
        if not isinstance(item, Mapping) or item.get("name") not in BFCL_EXPECTED_INPUT_ROLES:
            continue
        name = str(item["name"])
        if name in result:
            raise ValueError(f"duplicate BFCL manifest receipt: {name}")
        result[name] = (
            item.get("sha256"),
            item.get("upstream_commit"),
            item.get("license_id"),
        )
    return result


def _assert_runtime_bfcl_receipts(
    manifest: Mapping[str, Any], receipts: Sequence[ArtifactReceipt]
) -> None:
    _validate_bfcl_receipts(receipts)
    expected = {
        receipt.name: (
            receipt.sha256,
            receipt.upstream_commit,
            receipt.license_id,
        )
        for receipt in receipts
    }
    if _manifest_bfcl_receipts(manifest) != expected:
        raise ValueError("BFCL runtime inputs differ from the frozen run manifest")


TransportFactory = Callable[
    [BudgetLedger, Path, Mapping[str, str] | None, int], BFCLTransport
]


@dataclass(frozen=True)
class FrozenBFCLSelection:
    output_path: Path
    sha256: str
    task_ids: tuple[str, ...]
    categories: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "output_path": str(self.output_path),
            "sha256": self.sha256,
            "task_ids": list(self.task_ids),
            "categories": list(self.categories),
            "n_tasks": len(self.task_ids),
        }


@dataclass(frozen=True)
class ValidatedBFCLRun:
    manifest: Mapping[str, Any]
    cells: tuple[JobCell, ...]
    records_by_pair_task_id: Mapping[str, BFCLTaskRecord]


def freeze_selected_official_tasks(
    *,
    bridge: LoadableBFCLBridge,
    output_path: Path,
    categories: Sequence[str] = V4_MULTI_TURN_CATEGORIES,
    task_ids: Sequence[str] | None = None,
    one_task_smoke: bool = False,
) -> FrozenBFCLSelection:
    """Freeze a deterministic official BFCL selection in runner12 format."""

    requested_ids = tuple(str(value) for value in (task_ids or ()))
    if one_task_smoke and len(requested_ids) > 1:
        raise ValueError("one-task smoke accepts at most one explicit BFCL task id")

    loaded = tuple(bridge.load_tasks(categories=tuple(categories), task_ids=requested_ids))
    if not loaded:
        raise ValueError("BFCL bridge returned no tasks for the requested selection")

    selected = tuple(sorted(loaded, key=lambda task: (task.task_id, task.task_sha256)))
    if one_task_smoke:
        selected = selected[:1]

    digest = freeze_bfcl_public_task_manifest(
        output_path,
        selected,
        categories=tuple(sorted({task.category for task in selected})),
        task_ids=tuple(task.task_id for task in selected),
    )
    return FrozenBFCLSelection(
        output_path=output_path,
        sha256=digest,
        task_ids=tuple(task.task_id for task in selected),
        categories=tuple(sorted({task.category for task in selected})),
    )


def _task_refs(task_manifest_path: Path) -> tuple[TaskRef, ...]:
    rows = load_task_manifest(task_manifest_path)
    if not rows:
        raise ValueError("BFCL task manifest is empty")
    refs: list[TaskRef] = []
    for row in rows:
        if row.get("benchmark") != "bfcl_multi_turn":
            raise ValueError("BFCL run task manifest contains a non-BFCL task")
        if row.get("condition") != "official_native_tools":
            raise ValueError("BFCL run requires condition=official_native_tools")
        refs.append(
            TaskRef(
                benchmark=str(row["benchmark"]),
                task_id=str(row["task_id"]),
                task_sha256=str(row["task_sha256"]),
            )
        )
    return tuple(refs)


def initialize_bfcl_run(
    *,
    run_id: str,
    stage: Stage,
    task_manifest_path: Path,
    models: Sequence[str],
    arms: Sequence[str],
    replicas: int = 1,
    seed: int = 1201,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    repository_root: Path | None = None,
    benchmark_receipts: Sequence[ArtifactReceipt] = (),
    source_registry_path: Path | None = None,
    baseline_profile_path: Path | None = None,
    planning_lock_path: Path | None = None,
    threshold_path: Path | None = None,
    smoke_wave: str | None = None,
    realized_allocation_path: Path | None = None,
) -> RunLayout:
    """Create a frozen BFCL run using the common pair/manifest machinery."""

    repository_root = repository_root or REPOSITORY_ROOT
    task_rows = load_task_manifest(task_manifest_path)
    refs = _task_refs(task_manifest_path)
    source_ids = {str(row["source_task_id"]) for row in task_rows}
    if stage is Stage.SMOKE and len(source_ids) not in {1, 3, 5}:
        raise ValueError("BFCL smoke runs require 1, 3, or 5 frozen public tasks")
    if not models:
        raise ValueError("at least one target model is required")
    unknown_models = sorted(set(models).difference(TARGET_MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"unknown target models: {unknown_models}")
    if not arms:
        raise ValueError("at least one observation arm is required")
    arm_names = {item.name for item in ARMS}
    unknown_arms = sorted(set(arms).difference(arm_names))
    if unknown_arms:
        raise ValueError(f"unknown observation arms: {unknown_arms}")
    if replicas < 1:
        raise ValueError("replicas must be positive")
    _validate_bfcl_receipts(benchmark_receipts)
    analysis_lock = _confirmatory_analysis_lock(
        stage=stage,
        task_rows=task_rows,
        thresholds_path=threshold_path,
    )

    layout = RunLayout.for_run(artifacts_root, run_id)
    launch_binding = assert_scientific_launch(
        task_rows=task_rows,
        stage=stage,
        models=tuple(models),
        arms=tuple(arms),
        operators=("none",),
        replicates=replicas,
        ledger_path=layout.ledger,
        registry_path=source_registry_path,
        projection_lock_path=planning_lock_path,
        baseline_profile_path=baseline_profile_path,
        smoke_wave=smoke_wave,
        realized_allocation_path=realized_allocation_path,
    )
    if layout.manifest.exists() or layout.pairs.exists():
        raise FileExistsError(f"run {run_id!r} is already initialized")

    cells = make_pair_manifest(
        tasks=refs,
        models=tuple(models),
        arms=tuple(arms),
        operators=("none",),
        replicates=replicas,
        randomization_seed=seed,
    )
    layout.create()
    atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])

    task_receipt = ArtifactReceipt.from_file(
        name="task_manifest",
        path=task_manifest_path,
        workspace=repository_root,
        license_id="Apache-2.0",
    )
    if any(receipt.name == "task_manifest" for receipt in benchmark_receipts):
        raise ValueError("benchmark_receipts must not replace the task_manifest receipt")
    launch_receipts: list[ArtifactReceipt] = []
    for name, path in (
        ("source_allocation_registry", source_registry_path),
        ("measured_baseline_resource_profile", baseline_profile_path),
        ("cost_sample_size_projection_lock", planning_lock_path),
        ("realized_source_allocation", realized_allocation_path),
    ):
        if path is not None:
            launch_receipts.append(
                ArtifactReceipt.from_file(
                    name=name,
                    path=path,
                    workspace=repository_root,
                )
            )
    extra_config: dict[str, Any] = {
        "benchmark": "bfcl_multi_turn",
        "bfcl_condition": "official_native_tools",
        "bfcl_run_coordinator_version": BFCL_RUN_COORDINATOR_VERSION,
        "n_public_tasks": len(source_ids),
        "n_tasks": len(refs),
        "n_cells": len(cells),
        "smoke_wave": (
            None
            if stage is not Stage.SMOKE
            else "micro"
            if len(source_ids) == 1
            else "single_model"
            if len(source_ids) == 3
            else "all_models"
        ),
        "replicas": replicas,
    }
    if isinstance(launch_binding, ScientificLaunchBinding):
        extra_config["scientific_launch_lock"] = launch_binding.as_dict()
    elif isinstance(launch_binding, SourceAllocationBinding):
        extra_config["source_allocation"] = launch_binding.as_dict()
    if analysis_lock is not None:
        extra_config["analysis_lock"] = analysis_lock
    manifest = build_manifest(
        run_id=run_id,
        stage=stage,
        repository_root=repository_root,
        pair_manifest_sha256=sha256_file(layout.pairs),
        models=tuple(models),
        arms=tuple(arms),
        operators=("none",),
        randomization_seed=seed,
        benchmark_receipts=(
            task_receipt,
            *tuple(benchmark_receipts),
            *launch_receipts,
        ),
        extra_config=extra_config,
    )
    write_manifest_once(layout.manifest, manifest)
    BudgetLedger(
        layout.ledger,
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    return layout


def validate_bfcl_run(
    *,
    layout: RunLayout,
    task_manifest_path: Path,
    task_records: Sequence[BFCLTaskRecord],
    benchmark_receipts: Sequence[ArtifactReceipt],
    repository_root: Path | None = None,
) -> ValidatedBFCLRun:
    """Validate frozen receipts, declared pairs, and reloaded official tasks."""

    del repository_root  # the common validator uses the repository's frozen root
    records = tuple(task_records)
    if not records:
        raise ValueError("no BFCL task records supplied for validation")

    manifest, cells, _ = _validate_run_inputs(
        layout=layout,
        task_manifest_path=task_manifest_path,
        tasks=tuple(task.as_domain_task() for task in records),
    )
    _assert_runtime_bfcl_receipts(manifest, benchmark_receipts)
    if any(cell.pair_key.domain != "bfcl_multi_turn" for cell in cells):
        raise ValueError("declared pair manifest contains a non-BFCL cell")
    if any(cell.operator != "none" for cell in cells):
        raise ValueError("BFCL cells must use operator=none")

    record_by_identity = {
        (task.task_id, task.task_sha256): task
        for task in records
    }
    records_by_pair: dict[str, BFCLTaskRecord] = {}
    for task in records:
        domain_task = task.as_domain_task()
        records_by_pair[pair_task_id(domain_task)] = record_by_identity[
            (task.task_id, task.task_sha256)
        ]
    declared_pair_ids = {cell.pair_key.task_id for cell in cells}
    if declared_pair_ids != set(records_by_pair):
        missing = sorted(declared_pair_ids.difference(records_by_pair))
        extra = sorted(set(records_by_pair).difference(declared_pair_ids))
        raise ValueError(f"BFCL runtime selection mismatch: missing={missing}, extra={extra}")

    frozen_rows = load_task_manifest(task_manifest_path)
    frozen_source_ids = {str(row["source_task_id"]) for row in frozen_rows}
    runtime_source_ids = {task.task_id for task in records}
    if frozen_source_ids != runtime_source_ids:
        raise ValueError("BFCL bridge did not reload the exact frozen source-task set")
    if Stage(str(manifest["stage"])) is Stage.SMOKE and len(frozen_source_ids) not in {1, 3, 5}:
        raise ValueError("BFCL smoke runs require 1, 3, or 5 frozen public tasks")
    return ValidatedBFCLRun(
        manifest=manifest,
        cells=cells,
        records_by_pair_task_id=records_by_pair,
    )


def _default_transport_factory(
    ledger: BudgetLedger,
    event_path: Path,
    environ: Mapping[str, str] | None,
    max_attempts: int,
) -> BFCLTransport:
    return cast(
        BFCLTransport,
        Transport(
            ledger=ledger,
            event_log_path=event_path,
            environ=environ,
            max_attempts=max_attempts,
        ),
    )


def _read_job_state(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("state")
    if state not in {"complete", "failed"}:
        raise ValueError(f"invalid job-state file: {path}")
    return str(state)


def _refuse_unsafe_retry(*, job_path: Path, event_path: Path, output_path: Path) -> None:
    state = _read_job_state(job_path)
    if state == "failed":
        raise FileExistsError(
            f"failed BFCL cell requires audit; automatic retry refused: {job_path.name}"
        )
    if state == "complete" and not output_path.exists():
        raise FileExistsError(
            f"BFCL job is marked complete but output is missing: {output_path}"
        )
    if event_path.exists() != output_path.exists():
        raise FileExistsError(
            "partial BFCL artifacts require audit; automatic retry refused: "
            f"event={event_path.exists()}, output={output_path.exists()}"
        )


async def execute_bfcl_run(
    *,
    run_id: str,
    task_manifest_path: Path,
    bridge: LoadableBFCLBridge,
    benchmark_receipts: Sequence[ArtifactReceipt],
    yes_spend: bool,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    repository_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    phase: str = "both",
    run_judge: bool | None = None,
    judge_model: str | None = None,
    max_new_cells: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    config: BFCLRunnerConfig = BFCLRunnerConfig(),
    transport_factory: TransportFactory = _default_transport_factory,
) -> RunSummary:
    """Execute declared BFCL trajectory/shadow cells under the global ledger."""

    if yes_spend is not True:
        raise PermissionError("paid BFCL execution requires explicit --yes-spend")
    if phase == "shadows":
        phase = "shadow"
    if phase not in {"trajectories", "shadow", "both"}:
        raise ValueError("phase must be trajectories, shadow, or both")
    if max_new_cells is not None and max_new_cells < 1:
        raise ValueError("max_new_cells must be positive")
    shard = ExecutionShard(count=shard_count, index=shard_index)

    layout = RunLayout.for_run(artifacts_root, run_id)
    frozen_rows = load_task_manifest(task_manifest_path)
    source_ids = tuple(sorted({str(row["source_task_id"]) for row in frozen_rows}))
    records = tuple(
        bridge.load_tasks(
            categories=V4_MULTI_TURN_CATEGORIES,
            task_ids=source_ids,
        )
    )
    validated = validate_bfcl_run(
        layout=layout,
        task_manifest_path=task_manifest_path,
        task_records=records,
        benchmark_receipts=benchmark_receipts,
        repository_root=repository_root,
    )
    stage = Stage(str(validated.manifest["stage"]))
    if stage is Stage.OFFLINE:
        raise ValueError("offline stage is forbidden from making provider calls")
    passive_spec = passive_monitor_spec_from_manifest(validated.manifest)
    assert_passive_runtime_overrides(
        passive_spec,
        run_judge=run_judge,
        judge_model=judge_model,
    )
    if config.probe_max_output_tokens != passive_spec["frozen_probe"]["max_output_tokens"]:
        raise ValueError("BFCL probe output limit differs from the frozen passive spec")
    if config.temperature != passive_spec["determinism"]["temperature"]:
        raise ValueError("BFCL temperature differs from the frozen passive spec")
    shard_cells = shard.select(validated.cells)
    ledger = _stage_ledger(layout, run_id=run_id, stage=stage)
    transport = transport_factory(
        ledger,
        layout.events / "call_attempts.jsonl",
        environ,
        3 if stage is Stage.SMOKE else 6,
    )

    completed = failed = skipped = visited = 0
    new_cells = 0
    job_root = layout.results / "jobs"
    shadow_job_root = layout.results / "shadow_jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    shadow_job_root.mkdir(parents=True, exist_ok=True)

    if phase in {"trajectories", "both"}:
        for cell in shard_cells:
            output_path = layout.trajectories / f"{cell.cell_id}.json"
            event_path = layout.events / f"trajectory-{cell.cell_id}.jsonl"
            job_path = job_root / f"{cell.cell_id}.json"
            _refuse_unsafe_retry(
                job_path=job_path,
                event_path=event_path,
                output_path=output_path,
            )
            existed = output_path.exists()
            if not existed and max_new_cells is not None and new_cells >= max_new_cells:
                skipped += 1
                continue
            if not existed:
                visited += 1
                new_cells += 1
            task = validated.records_by_pair_task_id[cell.pair_key.task_id]
            try:
                await run_bfcl_task(
                    run_id=run_id,
                    cell_id=cell.cell_id,
                    model=cell.pair_key.model,
                    task=task,
                    arm_name=cell.arm,
                    bridge=bridge,
                    transport=transport,
                    event_path=event_path,
                    output_path=output_path,
                    config=config,
                )
                _job_state(
                    job_path,
                    cell=cell,
                    state="complete",
                    detail={
                        "phase": "trajectory",
                        "trajectory_sha256": sha256_file(output_path),
                    },
                )
                if existed:
                    completed += 1
                    skipped += 1
                else:
                    completed += 1
            except (BudgetError, FileExistsError):
                raise
            except Exception as exc:  # preserve evidence and prohibit an automatic retry
                _job_state(
                    job_path,
                    cell=cell,
                    state="failed",
                    detail={"phase": "trajectory", "error_type": type(exc).__name__},
                )
                failed += 1

    if phase in {"shadow", "both"}:
        for cell in shard_cells:
            if cell.arm != "clean":
                continue
            trajectory_path = layout.trajectories / f"{cell.cell_id}.json"
            if not trajectory_path.exists():
                skipped += 1
                continue
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            output_path = layout.shadow / f"{cell.cell_id}.json"
            event_path = layout.events / f"shadow-{cell.cell_id}.jsonl"
            job_path = shadow_job_root / f"{cell.cell_id}.json"
            _refuse_unsafe_retry(
                job_path=job_path,
                event_path=event_path,
                output_path=output_path,
            )
            existed = output_path.exists()
            if not existed and max_new_cells is not None and new_cells >= max_new_cells:
                skipped += 1
                continue
            if not existed:
                visited += 1
                new_cells += 1

            checkpoint_turns = tuple(
                int(value) for value in trajectory.get("checkpoint_turns", [])
            )
            task_records = trajectory.get("task_records")
            if not isinstance(task_records, list):
                raise ValueError("BFCL trajectory lacks public task_records")
            quiz_by_checkpoint = {
                turn: generate_bfcl_passive_quiz(task_records, after_turn=turn)
                for turn in checkpoint_turns
            }
            try:
                await score_clean_trajectory(
                    run_id=run_id,
                    trajectory=trajectory,
                    transport=transport,
                    event_path=event_path,
                    output_path=output_path,
                    passive_monitor_spec=passive_spec,
                    quiz_by_checkpoint=quiz_by_checkpoint,
                    run_judge=run_judge,
                    judge_model=judge_model,
                )
                _job_state(
                    job_path,
                    cell=cell,
                    state="complete",
                    detail={
                        "phase": "shadow",
                        "shadow_sha256": sha256_file(output_path),
                    },
                )
                if existed:
                    skipped += 1
                else:
                    completed += 1
            except (BudgetError, FileExistsError):
                raise
            except Exception as exc:  # preserve evidence and prohibit an automatic retry
                _job_state(
                    job_path,
                    cell=cell,
                    state="failed",
                    detail={"phase": "shadow", "error_type": type(exc).__name__},
                )
                failed += 1

    return RunSummary(
        declared_cells=len(validated.cells),
        visited_cells=visited,
        completed_cells=completed,
        failed_cells=failed,
        skipped_cells=skipped,
        phase=phase,
        shard_count=shard.count,
        shard_index=shard.index,
        shard_cells=len(shard_cells),
    )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="freeze an official public BFCL selection")
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE_SCRIPT)
    freeze.add_argument("--categories", default=",".join(V4_MULTI_TURN_CATEGORIES))
    freeze.add_argument("--task-ids", default="")
    freeze.add_argument("--one-task-smoke", action="store_true")
    freeze.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))

    init = sub.add_parser("init", help="initialize a frozen BFCL run")
    init.add_argument("--run-id", required=True)
    init.add_argument("--stage", required=True, choices=[stage.value for stage in Stage])
    init.add_argument("--tasks", required=True, type=Path)
    init.add_argument("--models", required=True)
    init.add_argument("--arms", required=True)
    init.add_argument("--replicas", type=int, default=1)
    init.add_argument("--seed", type=int, default=1201)
    init.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE_SCRIPT)
    init.add_argument("--source-registry", type=Path)
    init.add_argument("--baseline-profile", type=Path)
    init.add_argument("--planning-lock", type=Path)
    init.add_argument(
        "--thresholds",
        type=Path,
        help="locked calibration artifact; required for confirmatory init",
    )
    init.add_argument(
        "--smoke-wave", choices=("single_model", "all_models"), default=None
    )
    init.add_argument("--realized-allocation", type=Path)
    init.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    init.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)

    run = sub.add_parser("run", help="execute declared BFCL cells")
    run.add_argument("--run-id", required=True)
    run.add_argument("--tasks", required=True, type=Path)
    run.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE_SCRIPT)
    run.add_argument("--phase", choices=("trajectories", "shadow", "both"), default="both")
    run.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="optional assertion; must match the judge setting frozen in the manifest",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        help="optional assertion; must match the judge model frozen in the manifest",
    )
    run.add_argument("--max-new-cells", type=int)
    add_execution_shard_arguments(run)
    run.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    run.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    run.add_argument("--yes-spend", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "freeze":
            environment = _environment(args.env_file)
            adapter = BFCLAdapter(environment=environment)
            with adapter.bridge_client(args.bridge, base_environment=environment) as bridge:
                frozen = freeze_selected_official_tasks(
                    bridge=bridge,
                    output_path=args.output,
                    categories=_csv(args.categories),
                    task_ids=_csv(args.task_ids),
                    one_task_smoke=args.one_task_smoke,
                )
            print(json.dumps(frozen.as_dict(), sort_keys=True))
            return 0
        if args.command == "init":
            environment = _environment(args.env_file)
            adapter = BFCLAdapter(environment=environment)
            bridge = adapter.bridge_client(args.bridge, base_environment=environment)
            provenance = bfcl_provenance_receipts(bridge.input_artifacts)
            layout = initialize_bfcl_run(
                run_id=args.run_id,
                stage=Stage(args.stage),
                task_manifest_path=args.tasks,
                models=_csv(args.models),
                arms=_csv(args.arms),
                replicas=args.replicas,
                seed=args.seed,
                artifacts_root=args.artifacts_root,
                benchmark_receipts=provenance,
                source_registry_path=args.source_registry,
                baseline_profile_path=args.baseline_profile,
                planning_lock_path=args.planning_lock,
                threshold_path=args.thresholds,
                smoke_wave=args.smoke_wave,
                realized_allocation_path=args.realized_allocation,
            )
            print(json.dumps({"run_id": args.run_id, "manifest": str(layout.manifest)}))
            return 0
        if not args.yes_spend:
            raise PermissionError("paid BFCL execution requires explicit --yes-spend")
        ExecutionShard(count=args.shard_count, index=args.shard_index)
        environment = _environment(args.env_file)
        adapter = BFCLAdapter(environment=environment)
        with adapter.bridge_client(args.bridge, base_environment=environment) as bridge:
            provenance = bfcl_provenance_receipts(bridge.input_artifacts)
            summary = asyncio.run(
                execute_bfcl_run(
                    run_id=args.run_id,
                    task_manifest_path=args.tasks,
                    bridge=bridge,
                    benchmark_receipts=provenance,
                    yes_spend=True,
                    artifacts_root=args.artifacts_root,
                    environ=environment,
                    phase=args.phase,
                    run_judge=args.judge,
                    judge_model=args.judge_model,
                    max_new_cells=args.max_new_cells,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                )
            )
        print(
            json.dumps(
                {
                    "declared_cells": summary.declared_cells,
                    "visited_cells": summary.visited_cells,
                    "completed_cells": summary.completed_cells,
                    "failed_cells": summary.failed_cells,
                    "skipped_cells": summary.skipped_cells,
                    "phase": summary.phase,
                    "shard_count": summary.shard_count,
                    "shard_index": summary.shard_index,
                    "shard_cells": summary.shard_cells,
                },
                sort_keys=True,
            )
        )
        return 0 if summary.failed_cells == 0 else 2
    except (BudgetError, FileExistsError, FileNotFoundError, PermissionError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
