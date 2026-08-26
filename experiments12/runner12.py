"""Freeze and execute declared Experiment 12 scripted benchmark cells.

This module is intentionally narrow: Evolving Intent and other scripted text
tasks run here; BFCL uses its isolated official-environment bridge.  A paid run
must already have an immutable manifest and pair table, and every request key is
namespaced by run so the shared experiment ledger cannot collide across runs.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments12.cli12 import DEFAULT_ARTIFACTS, REPOSITORY_ROOT, _environment
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
)
from experiments12.core.budget import BudgetError, BudgetLedger
from experiments12.core.transport import Transport, TransportError
from experiments12.domains.base import DomainTask
from experiments12.domains.evolving_intent import EvolvingIntentAdapter
from experiments12.execution_sharding12 import (
    ExecutionShard,
    add_execution_shard_arguments,
)
from experiments12.harness12 import HarnessConfig, run_scripted_task
from experiments12.manifest12 import RunLayout, validate_manifest_files
from experiments12.passive_spec12 import (
    assert_passive_runtime_overrides,
    passive_monitor_spec_from_manifest,
)
from experiments12.pairing12 import JobCell
from experiments12.shadow12 import score_clean_trajectory
from experiments12.spec12 import (
    ARMS,
    ObservationKind,
    OPERATIONAL_PROVIDER_USD,
    STAGE_PROVIDER_USD,
    Stage,
)


TASK_MANIFEST_VERSION = 1
RUNNER_VERSION = 1


def _assert_frozen_receipt_file(
    manifest: Mapping[str, Any], name: str, path: str | Path
) -> None:
    digest = sha256_file(path)
    matches = [
        receipt
        for receipt in manifest.get("benchmark_receipts", ())
        if isinstance(receipt, Mapping) and receipt.get("name") == name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise ValueError(f"runtime {name} differs from the frozen run manifest")


def pair_task_id(task: DomainTask) -> str:
    """Stable, condition-aware task identity used by paired cell manifests."""

    if "::" in task.task_id:
        raise ValueError("source task_id may not contain the Experiment 12 separator '::'")
    return f"{task.task_id}::{task.condition}"


def task_manifest_rows(tasks: Sequence[DomainTask]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.domain, pair_task_id(task))
        if key in seen:
            raise ValueError(f"duplicate domain task in task manifest: {key}")
        seen.add(key)
        rows.append(
            {
                "task_manifest_version": TASK_MANIFEST_VERSION,
                "benchmark": task.domain,
                "task_id": key[1],
                "source_task_id": task.task_id,
                "condition": task.condition,
                "num_turns": len(task.turns),
                "source_sha256": task.source_sha256,
                "task_sha256": task.task_sha256,
            }
        )
    if not rows:
        raise ValueError("cannot freeze an empty task manifest")
    return rows


def freeze_task_manifest(path: str | Path, tasks: Sequence[DomainTask]) -> str:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("task manifest is write-once")
    return atomic_write_jsonl(destination, task_manifest_rows(tasks))


def load_task_manifest(path: str | Path) -> tuple[dict[str, Any], ...]:
    expected = {
        "task_manifest_version",
        "benchmark",
        "task_id",
        "source_task_id",
        "condition",
        "num_turns",
        "source_sha256",
        "task_sha256",
    }
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("task manifest is empty")
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(f"task manifest row {index} has missing or unexpected fields")
        if row["task_manifest_version"] != TASK_MANIFEST_VERSION:
            raise ValueError(f"task manifest row {index} has an unsupported version")
        if any(
            not isinstance(row[key], str) or not row[key]
            for key in (
                "benchmark",
                "task_id",
                "source_task_id",
                "condition",
                "source_sha256",
                "task_sha256",
            )
        ):
            raise ValueError(f"task manifest row {index} contains an invalid string")
        if row["task_id"] != f"{row['source_task_id']}::{row['condition']}":
            raise ValueError(f"task manifest row {index} has a noncanonical task_id")
        if (
            isinstance(row["num_turns"], bool)
            or not isinstance(row["num_turns"], int)
            or row["num_turns"] < 1
        ):
            raise ValueError(f"task manifest row {index} has invalid num_turns")
        for digest_key in ("source_sha256", "task_sha256"):
            digest = row[digest_key]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"task manifest row {index} has invalid {digest_key}")
        identity = (row["benchmark"], row["task_id"], row["task_sha256"])
        if identity in identities:
            raise ValueError("task manifest contains a duplicate identity")
        identities.add(identity)
        result.append(row)
    return tuple(result)


def load_pair_cells(path: str | Path) -> tuple[JobCell, ...]:
    cells = tuple(JobCell.from_dict(row) for row in read_jsonl(path))
    if not cells:
        raise ValueError("pair manifest is empty")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("pair manifest contains duplicate cells")
    return cells


def resolve_declared_tasks(
    tasks: Sequence[DomainTask],
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], DomainTask]:
    available = {
        (task.domain, pair_task_id(task), task.task_sha256): task for task in tasks
    }
    if len(available) != len(tasks):
        raise ValueError("adapter returned duplicate task identities")
    resolved: dict[tuple[str, str, str], DomainTask] = {}
    for row in rows:
        key = (str(row["benchmark"]), str(row["task_id"]), str(row["task_sha256"]))
        try:
            task = available[key]
        except KeyError as exc:
            raise ValueError(f"frozen task is absent or changed: {key[0]}/{key[1]}") from exc
        if (
            task.task_id != row["source_task_id"]
            or task.condition != row["condition"]
            or len(task.turns) != row["num_turns"]
            or task.source_sha256 != row["source_sha256"]
        ):
            raise ValueError(f"frozen task metadata changed: {key[0]}/{key[1]}")
        resolved[key] = task
    return resolved


def _validate_run_inputs(
    *,
    layout: RunLayout,
    task_manifest_path: Path,
    tasks: Sequence[DomainTask],
) -> tuple[dict[str, Any], tuple[JobCell, ...], dict[tuple[str, str, str], DomainTask]]:
    if not layout.manifest.is_file() or not layout.pairs.is_file():
        raise FileNotFoundError("run must be initialized before dispatch")
    manifest = read_json(layout.manifest)
    errors = validate_manifest_files(
        manifest,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_path=layout.pairs,
    )
    if errors:
        raise ValueError("frozen run validation failed: " + "; ".join(errors))
    task_digest = sha256_file(task_manifest_path)
    receipts = manifest.get("benchmark_receipts", ())
    if not any(
        receipt.get("name") == "task_manifest" and receipt.get("sha256") == task_digest
        for receipt in receipts
        if isinstance(receipt, Mapping)
    ):
        raise ValueError("supplied task manifest is not the one frozen into this run")
    rows = load_task_manifest(task_manifest_path)
    resolved = resolve_declared_tasks(tasks, rows)
    rows_by_identity = {
        (str(row["benchmark"]), str(row["task_id"]), str(row["task_sha256"])): row
        for row in rows
    }
    cells = load_pair_cells(layout.pairs)
    active_arm_names = {
        item.name for item in ARMS if item.observation is ObservationKind.ACTIVE
    }
    for cell in cells:
        key = (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        if key not in resolved:
            raise ValueError(f"pair cell refers to an undeclared task: {cell.cell_id}")
        if cell.pair_key.model not in manifest.get("models", ()):
            raise ValueError(f"pair cell model is outside the frozen manifest: {cell.cell_id}")
        if cell.arm not in manifest.get("arms", ()) or cell.operator not in manifest.get(
            "operators", ()
        ):
            raise ValueError(f"pair cell treatment changed: {cell.cell_id}")
        condition = str(rows_by_identity[key]["condition"])
        if condition == "t1" and cell.arm in active_arm_names:
            raise ValueError("active observation arms are forbidden for t1 tasks")
    if len(cells) != int(manifest.get("extra_config", {}).get("n_cells", -1)):
        raise ValueError("pair cell count differs from frozen manifest")
    return manifest, cells, resolved


def _stage_ledger(layout: RunLayout, run_id: str, stage: Stage) -> BudgetLedger:
    return BudgetLedger(
        layout.ledger,
        operational_caps_usd={
            provider: Decimal(str(cap))
            for provider, cap in OPERATIONAL_PROVIDER_USD.items()
        },
        request_scope=run_id,
        scope_caps_usd={
            provider: Decimal(str(cap))
            for provider, cap in STAGE_PROVIDER_USD[stage].items()
        },
    )


def _job_state(path: Path, *, cell: JobCell, state: str, detail: Mapping[str, Any]) -> None:
    if state not in {"complete", "failed"}:
        raise ValueError("invalid job state")
    atomic_write_json(
        path,
        {
            "runner_version": RUNNER_VERSION,
            "cell_id": cell.cell_id,
            "state": state,
            **dict(detail),
        },
    )


@dataclass(frozen=True, slots=True)
class RunSummary:
    declared_cells: int
    visited_cells: int
    completed_cells: int
    failed_cells: int
    skipped_cells: int
    phase: str
    shard_count: int = 1
    shard_index: int = 0
    shard_cells: int | None = None


async def execute_scripted_run(
    *,
    run_id: str,
    task_manifest_path: str | Path,
    tasks: Sequence[DomainTask],
    artifacts_root: str | Path = DEFAULT_ARTIFACTS,
    environ: Mapping[str, str] | None = None,
    phase: str = "both",
    run_judge: bool | None = None,
    judge_model: str | None = None,
    max_new_cells: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    config: HarnessConfig = HarnessConfig(),
    evolving_dataset_path: str | Path | None = None,
    evolving_build_receipt_path: str | Path | None = None,
) -> RunSummary:
    if phase not in {"trajectories", "shadow", "both"}:
        raise ValueError("phase must be trajectories, shadow, or both")
    if max_new_cells is not None and max_new_cells < 1:
        raise ValueError("max_new_cells must be positive or None")
    shard = ExecutionShard(count=shard_count, index=shard_index)
    layout = RunLayout.for_run(artifacts_root, run_id)
    manifest, cells, task_index = _validate_run_inputs(
        layout=layout,
        task_manifest_path=Path(task_manifest_path),
        tasks=tasks,
    )
    if any(cell.pair_key.domain == "evolving_intent_gsm8k" for cell in cells):
        if evolving_dataset_path is None or evolving_build_receipt_path is None:
            raise ValueError(
                "Evolving runtime requires its frozen dataset and build receipt"
            )
        _assert_frozen_receipt_file(
            manifest, "evolving_rendered_dataset", evolving_dataset_path
        )
        _assert_frozen_receipt_file(
            manifest, "evolving_build_receipt", evolving_build_receipt_path
        )
        build_receipt = read_json(evolving_build_receipt_path)
        frozen_dataset = build_receipt.get("frozen_dataset")
        if (
            not isinstance(frozen_dataset, Mapping)
            or frozen_dataset.get("sha256") != sha256_file(evolving_dataset_path)
        ):
            raise ValueError("Evolving build receipt no longer attests the runtime dataset")
    stage = Stage(manifest["stage"])
    if stage is Stage.OFFLINE:
        raise ValueError("offline stage is forbidden from making provider calls")
    passive_spec = passive_monitor_spec_from_manifest(manifest)
    assert_passive_runtime_overrides(
        passive_spec,
        run_judge=run_judge,
        judge_model=judge_model,
    )
    if config.checkpoint_every != passive_spec["checkpoint"]["every"]:
        raise ValueError("runtime checkpoint cadence differs from the frozen passive spec")
    if config.probe_max_output_tokens != passive_spec["frozen_probe"]["max_output_tokens"]:
        raise ValueError("runtime probe output limit differs from the frozen passive spec")
    if config.temperature != passive_spec["determinism"]["temperature"]:
        raise ValueError("runtime temperature differs from the frozen passive spec")
    if any(cell.operator != "none" for cell in cells):
        raise ValueError(
            "scripted observer-effect runner only accepts operator=none; "
            "deployment uses the intervention runner"
        )
    shard_cells = shard.select(cells)
    ledger = _stage_ledger(layout, run_id, stage)
    transport = Transport(
        ledger,
        layout.events / "call_attempts.jsonl",
        environ=environ,
        max_attempts=3 if stage is Stage.SMOKE else 6,
    )
    job_root = layout.results / "jobs"
    shadow_job_root = layout.results / "shadow_jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    shadow_job_root.mkdir(parents=True, exist_ok=True)
    completed = failed = skipped = visited = new_cells = 0

    if phase in {"trajectories", "both"}:
        for cell in shard_cells:
            output = layout.trajectories / f"{cell.cell_id}.json"
            job = job_root / f"{cell.cell_id}.json"
            if output.exists():
                completed += 1
                skipped += 1
                continue
            if job.exists() and read_json(job).get("state") == "failed":
                failed += 1
                skipped += 1
                continue
            if max_new_cells is not None and new_cells >= max_new_cells:
                skipped += 1
                continue
            visited += 1
            new_cells += 1
            key = (
                cell.pair_key.domain,
                cell.pair_key.task_id,
                str(cell.pair_key.task_sha256),
            )
            try:
                trajectory = await run_scripted_task(
                    run_id=run_id,
                    cell_id=cell.cell_id,
                    model=cell.pair_key.model,
                    task=task_index[key],
                    arm_name=cell.arm,
                    transport=transport,
                    event_path=layout.events / f"trajectory-{cell.cell_id}.jsonl",
                    output_path=output,
                    config=config,
                )
            except Exception as exc:
                detail: dict[str, Any] = {"error_type": type(exc).__name__}
                if isinstance(exc, TransportError):
                    detail.update(
                        {
                            "transport_category": exc.category,
                            "http_status": exc.http_status,
                            "attempts": len(exc.attempts),
                        }
                    )
                _job_state(job, cell=cell, state="failed", detail=detail)
                failed += 1
                if isinstance(exc, BudgetError):
                    raise
                continue
            _job_state(
                job,
                cell=cell,
                state="complete",
                detail={"trajectory_sha256": sha256_file(output), "success": trajectory["evaluation"]["success"]},
            )
            completed += 1

    if phase in {"shadow", "both"}:
        for cell in shard_cells:
            if cell.arm != "clean" or cell.operator != "none":
                continue
            trajectory_path = layout.trajectories / f"{cell.cell_id}.json"
            output = layout.shadow / f"{cell.cell_id}.json"
            job = shadow_job_root / f"{cell.cell_id}.json"
            existed = output.exists()
            if not trajectory_path.exists():
                skipped += 1
                continue
            if job.exists() and read_json(job).get("state") == "failed":
                skipped += 1
                continue
            if not existed and max_new_cells is not None and new_cells >= max_new_cells:
                skipped += 1
                continue
            if not existed:
                visited += 1
                new_cells += 1
            try:
                result = await score_clean_trajectory(
                    run_id=run_id,
                    trajectory=read_json(trajectory_path),
                    transport=transport,
                    event_path=layout.events / f"shadow-{cell.cell_id}.jsonl",
                    output_path=output,
                    passive_monitor_spec=passive_spec,
                    run_judge=run_judge,
                    judge_model=judge_model,
                )
            except Exception as exc:
                detail = {"error_type": type(exc).__name__}
                if isinstance(exc, TransportError):
                    detail.update(
                        {
                            "transport_category": exc.category,
                            "http_status": exc.http_status,
                            "attempts": len(exc.attempts),
                        }
                    )
                _job_state(job, cell=cell, state="failed", detail=detail)
                failed += 1
                if isinstance(exc, BudgetError):
                    raise
                continue
            if existed:
                skipped += 1
            else:
                _job_state(
                    job,
                    cell=cell,
                    state="complete",
                    detail={
                        "shadow_sha256": sha256_file(output),
                        "monitor_methods": result["monitor_methods"],
                        "passive_monitor_spec_sha256": result[
                            "passive_monitor_spec_sha256"
                        ],
                    },
                )
                completed += 1

    return RunSummary(
        declared_cells=len(cells),
        visited_cells=visited,
        completed_cells=completed,
        failed_cells=failed,
        skipped_cells=skipped,
        phase=phase,
        shard_count=shard.count,
        shard_index=shard.index,
        shard_cells=len(shard_cells),
    )


def _split(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("comma-separated values must be nonempty and unique")
    return result


def _freeze_evolving(args: argparse.Namespace) -> int:
    adapter = EvolvingIntentAdapter(args.dataset, expected_sha256=args.dataset_sha256)
    tasks = list(adapter.load_tasks())
    conditions = set(_split(args.conditions))
    if conditions - {"t1", "t7"}:
        raise ValueError("Evolving Intent conditions must be t1 and/or t7")
    tasks = [task for task in tasks if task.condition in conditions]
    if args.task_ids:
        selected = set(_split(args.task_ids))
        available = {task.task_id for task in tasks}
        missing = selected - available
        if missing:
            raise ValueError(f"unknown source task IDs: {sorted(missing)}")
        tasks = [task for task in tasks if task.task_id in selected]
    if args.limit is not None:
        source_order = list(dict.fromkeys(task.task_id for task in tasks))[: args.limit]
        selected = set(source_order)
        tasks = [task for task in tasks if task.task_id in selected]
    digest = freeze_task_manifest(args.output, tasks)
    print(f"frozen {len(tasks)} condition-specific tasks: {args.output}")
    print(f"task_manifest_sha256={digest}")
    print(f"source_dataset_sha256={adapter.source_sha256}")
    return 0


async def _run_evolving(args: argparse.Namespace) -> int:
    if not args.yes_spend:
        raise ValueError("provider dispatch requires --yes-spend")
    ExecutionShard(count=args.shard_count, index=args.shard_index)
    adapter = EvolvingIntentAdapter(args.dataset, expected_sha256=args.dataset_sha256)
    summary = await execute_scripted_run(
        run_id=args.run_id,
        task_manifest_path=args.tasks,
        tasks=adapter.load_tasks(),
        artifacts_root=args.artifacts,
        environ=_environment(args.env_file),
        phase=args.phase,
        run_judge=args.judge,
        judge_model=args.judge_model,
        max_new_cells=args.max_new_cells,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        evolving_dataset_path=args.dataset,
        evolving_build_receipt_path=args.build_receipt,
    )
    print(
        f"run={args.run_id} phase={summary.phase} visited={summary.visited_cells} "
        f"completed={summary.completed_cells} failed={summary.failed_cells} "
        f"skipped={summary.skipped_cells} shard={summary.shard_index}/"
        f"{summary.shard_count} shard_cells={summary.shard_cells} "
        f"declared={summary.declared_cells}"
    )
    return 0 if summary.failed_cells == 0 else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-evolving")
    freeze.add_argument("--dataset", required=True)
    freeze.add_argument("--dataset-sha256", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--conditions", default="t1,t7")
    freeze.add_argument("--task-ids", default=None)
    freeze.add_argument("--limit", type=int, default=None)
    freeze.set_defaults(func=_freeze_evolving)

    run = commands.add_parser("run-evolving")
    run.add_argument("--yes-spend", action="store_true")
    run.add_argument("--run-id", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--dataset-sha256", required=True)
    run.add_argument("--build-receipt", required=True)
    run.add_argument("--tasks", required=True)
    run.add_argument("--phase", choices=("trajectories", "shadow", "both"), default="both")
    run.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="optional assertion; must match the judge setting frozen in the run manifest",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        help="optional assertion; must match the judge model frozen in the run manifest",
    )
    run.add_argument("--max-new-cells", type=int, default=None)
    add_execution_shard_arguments(run)
    run.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    run.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    run.set_defaults(func=_run_evolving)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.func(args)
        return asyncio.run(result) if asyncio.iscoroutine(result) else int(result)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
