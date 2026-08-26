"""Genuinely online adaptive deployment for Evolving Intent.

This is the primary ecological deployment runner.  At every non-final task
turn it observes the *current* target history, immediately compares the score
with a calibration-locked threshold, and (subject to a frozen per-task cap)
applies the declared operator before the next benchmark turn.  Any intervention
therefore changes the prefix seen by every later observer.

The older :mod:`experiments12.deployment12` runner remains a controlled
two-pass replay sensitivity.  It deliberately freezes actions from a separate
trajectory and must not be described as online adaptation.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments12.cli12 import (
    DEFAULT_ARTIFACTS,
    REPOSITORY_ROOT,
    _environment,
    _evolving_provenance_receipts,
)
from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetError, BudgetLedger
from experiments12.core.transport import (
    CompletionResult,
    JsonSchemaOutput,
    Transport,
    TransportError,
)
from experiments12.deployment12 import (
    THRESHOLD_LOCK_RECEIPT,
    LockedMethodThreshold,
    ThresholdLockArtifact,
    _validate_evolving_runtime_provenance,
    freeze_threshold_lock,
    load_threshold_lock,
)
from experiments12.domains.base import DomainTask
from experiments12.domains.evolving_intent import EvolvingIntentAdapter
from experiments12.execution_sharding12 import (
    ExecutionShard,
    add_execution_shard_arguments,
)
from experiments12.harness12 import (
    ARM_TO_PROBE,
    DEFAULT_REASONING_EFFORT,
    HarnessConfig,
    conservative_input_token_bound,
    grade_final_numeric,
)
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.models12 import CATALOG
from experiments12.monitors.frozen_probe import build_frozen_probe_fork
from experiments12.monitors.frozen_quiz import build_quiz_fork, grade_quiz
from experiments12.monitors.judge import (
    JUDGE_RESPONSE_SCHEMA,
    build_judge_request,
    parse_judge_output,
)
from experiments12.monitors.trace_rules import score_trace_rules
from experiments12.operators12 import (
    CheckpointSchedule,
    CompactionConfig,
    InterventionType,
    ScheduleMode,
    ScheduledMember,
    SignalReference,
    apply_intervention,
    freeze_initial_instructions,
    freeze_public_state,
    freeze_visible_prefix,
    make_feedback_note,
)
from experiments12.pairing12 import (
    CompletenessReport,
    JobCell,
    TaskRef,
    check_completeness,
    make_pair_manifest,
)
from experiments12.passive_quizzes12 import generate_evolving_passive_quiz
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    canonical_passive_monitor_spec,
    effective_passive_method_names,
    passive_monitor_spec_from_manifest,
    quiz_generator_spec,
    validate_passive_monitor_spec,
)
from experiments12.planning_lock12 import (
    ScientificLaunchBinding,
    assert_scientific_launch,
)
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.prepare_deployment12 import (
    CALIBRATION_EXTRACT_RECEIPT,
    CALIBRATION_MANIFEST_RECEIPT,
    CALIBRATION_THRESHOLDS_RECEIPT,
    deployment_threshold_lock_from_analysis,
    verify_analysis_threshold_derivation,
    verify_calibration_extract_against_run,
)
from experiments12.runner12 import (
    RunSummary,
    _stage_ledger,
    _validate_run_inputs,
    load_task_manifest,
    pair_task_id,
    resolve_declared_tasks,
)
from experiments12.spec12 import (
    Benchmark,
    OPERATIONAL_PROVIDER_USD,
    Operator,
    Stage,
)


ADAPTIVE_RUNNER_VERSION = 1
ADAPTIVE_SCHEMA_VERSION = 1
ADAPTIVE_DEPLOYMENT_MODE = "online_adaptive"
ADAPTIVE_POLICY = "natural_threshold_per_task_cap"
PRIMARY_MAX_ACTIONS_PER_TASK = 1
PRIMARY_REPLICATES = 1
ADAPTIVE_RESULT_SUBDIR = "adaptive_deployment"
ADAPTIVE_JOB_SUBDIR = "adaptive_deployment_jobs"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_OPERATORS = {
    Operator.NONE.value: InterventionType.NONE,
    Operator.COMPACT.value: InterventionType.COMPACT,
    Operator.REGROUND.value: InterventionType.REGROUND,
    Operator.FEEDBACK.value: InterventionType.FEEDBACK,
}
_PASSIVE_METHODS = frozenset(effective_passive_method_names())


class AdaptiveDeploymentError(ValueError):
    """An online design, artifact, prefix, or receipt failed closed."""


@dataclass(frozen=True, slots=True)
class AdaptivePreparationResult:
    run_id: str
    manifest_path: Path
    manifest_sha256: str
    pair_manifest_path: Path
    pair_manifest_sha256: str
    threshold_lock_path: Path
    threshold_lock_sha256: str
    declared_cells: int


def _digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdaptiveDeploymentError(f"{name} must be a lowercase SHA256 digest")
    return value


def _request_key(run_id: str, cell_id: str, kind: str, checkpoint: int) -> str:
    if not _RUN_ID_RE.fullmatch(run_id) or not _CELL_ID_RE.fullmatch(cell_id):
        raise AdaptiveDeploymentError("run/cell identifiers are unsafe")
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint < 1:
        raise AdaptiveDeploymentError("request checkpoint must be positive")
    return f"{run_id}/{cell_id}/adaptive-{kind}-{checkpoint}"


def _method_kind(method: str) -> tuple[str, str | None]:
    if method in ARM_TO_PROBE:
        return "active_carry", ARM_TO_PROBE[method]
    if method.startswith("active_"):
        raise AdaptiveDeploymentError(f"unknown active method: {method!r}")
    if method not in _PASSIVE_METHODS:
        raise AdaptiveDeploymentError(f"unknown adaptive observation method: {method!r}")
    if method in {"turn_clock", "context_use"}:
        return "baseline", None
    return "passive_zero_carry", None


def _call_record(result: CompletionResult) -> dict[str, Any]:
    return {
        "call_event_ids": [attempt.event_id for attempt in result.attempts],
        "resolved_model_id": result.model_id,
        "response_id": result.response_id,
        "request_id": result.request_id,
        "finish_reason": result.finish_reason,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
        },
        "accounted_cost_usd": str(result.cost_usd),
        "elapsed_ms": sum(attempt.elapsed_ms or 0 for attempt in result.attempts),
    }


def _accounting(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "elapsed_ms",
    )

    def empty() -> dict[str, Any]:
        return {**{field: 0 for field in fields}, "accounted_cost_usd": Decimal("0")}

    categories = {"agent": empty(), "observer": empty()}
    event_ids: set[str] = set()
    resolved_model_ids: set[str] = set()
    for event in events:
        if event.get("event") == "task_turn":
            category = "agent"
        elif event.get("event") == "signal_observed" and event.get("call") is not None:
            category = "observer"
        else:
            continue
        call = event.get("call")
        if not isinstance(call, Mapping) or not isinstance(call.get("usage"), Mapping):
            raise AdaptiveDeploymentError("paid event lacks an exact call receipt")
        ids = call.get("call_event_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or event_ids.intersection(ids)
        ):
            raise AdaptiveDeploymentError("call event IDs are empty or duplicated")
        event_ids.update(ids)
        bucket = categories[category]
        bucket["calls"] += 1
        for field in fields[1:-1]:
            value = call["usage"].get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AdaptiveDeploymentError("call usage is invalid")
            bucket[field] += value
        elapsed = call.get("elapsed_ms")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise AdaptiveDeploymentError("call elapsed time is invalid")
        bucket["elapsed_ms"] += elapsed
        try:
            cost = Decimal(str(call["accounted_cost_usd"]))
        except Exception as exc:
            raise AdaptiveDeploymentError("call cost is invalid") from exc
        if not cost.is_finite() or cost < 0:
            raise AdaptiveDeploymentError("call cost is negative or non-finite")
        bucket["accounted_cost_usd"] += cost
        model_id = call.get("resolved_model_id")
        if not isinstance(model_id, str) or not model_id:
            raise AdaptiveDeploymentError("call lacks a resolved model identity")
        resolved_model_ids.add(model_id)

    total = empty()
    for bucket in categories.values():
        for field in fields:
            total[field] += bucket[field]
        total["accounted_cost_usd"] += bucket["accounted_cost_usd"]
        bucket["accounted_cost_usd"] = str(bucket["accounted_cost_usd"])
    total["accounted_cost_usd"] = str(total["accounted_cost_usd"])
    return {
        "by_category": categories,
        "total": total,
        "call_event_ids": sorted(event_ids),
        "resolved_model_ids": sorted(resolved_model_ids),
    }


def _public_state(task: DomainTask, checkpoint: int) -> dict[str, Any]:
    return {
        "completed_task_turns": [
            {"turn": turn.index, "user_message": turn.user_message}
            for turn in task.turns[:checkpoint]
        ],
        "public_metadata": dict(task.public_metadata),
    }


def _record_hash(record: Mapping[str, Any], *, field: str) -> str:
    if field in record:
        raise AdaptiveDeploymentError(f"{field} cannot be self-hashed")
    return sha256_json(dict(record))


def _validate_hashed_record(record: Mapping[str, Any], *, field: str) -> None:
    if not isinstance(record, Mapping):
        raise AdaptiveDeploymentError("hashed record must be an object")
    digest = _digest(field, record.get(field))
    core = {key: value for key, value in record.items() if key != field}
    if sha256_json(core) != digest:
        raise AdaptiveDeploymentError(f"{field} does not bind its record")


def _feedback_note(prefix: Any) -> Any:
    visible = [
        message.get("content")
        for message in reversed(prefix.messages)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    for content in visible:
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for length in range(min(40, len(stripped)), 0, -1):
                quote = stripped[:length]
                try:
                    return make_feedback_note(prefix, watch=(quote,))
                except ValueError:
                    continue
    raise AdaptiveDeploymentError("current prefix has no bounded feedback evidence")


def _online_schedule(cell: JobCell, checkpoint: int, horizon: int) -> CheckpointSchedule:
    audit_member = f"{cell.cell_id}:online-audit"
    eligible = tuple(range(1, horizon))
    members = tuple(
        sorted(
            (
                ScheduledMember(cell.cell_id, eligible, (checkpoint,)),
                ScheduledMember(audit_member, eligible, (checkpoint,)),
            ),
            key=lambda row: row.member_id,
        )
    )
    return CheckpointSchedule(
        group_id=sha256_json(
            {
                "mode": ADAPTIVE_DEPLOYMENT_MODE,
                "cell_id": cell.cell_id,
                "checkpoint": checkpoint,
            }
        )[:24],
        mode=ScheduleMode.MATCHED,
        members=members,
        seed=cell.seed,
    )


def _validate_cell_task_threshold(
    cell: JobCell,
    task: DomainTask,
    threshold: LockedMethodThreshold,
) -> None:
    if task.domain != Benchmark.EVOLVING_GSM8K.value or len(task.turns) < 2:
        raise AdaptiveDeploymentError("adaptive deployment requires multi-turn Evolving tasks")
    if cell.operator not in _OPERATORS:
        raise AdaptiveDeploymentError("adaptive cell has an unsupported operator")
    _method_kind(cell.arm)
    identity = (
        cell.pair_key.model,
        cell.pair_key.domain,
        cell.pair_key.task_id,
        str(cell.pair_key.task_sha256),
    )
    expected = (
        cell.pair_key.model,
        task.domain,
        pair_task_id(task),
        task.task_sha256,
    )
    if identity != expected:
        raise AdaptiveDeploymentError("adaptive cell and task identities differ")
    if (threshold.model, threshold.benchmark, threshold.method) != (
        cell.pair_key.model,
        cell.pair_key.domain,
        cell.arm,
    ):
        raise AdaptiveDeploymentError("adaptive cell uses another method's threshold")


def validate_adaptive_design(
    *,
    cells: Sequence[JobCell],
    task_index: Mapping[tuple[str, str, str], DomainTask],
    threshold_lock: ThresholdLockArtifact,
) -> None:
    """Validate exact slices and paired none/operator observation burden."""

    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise AdaptiveDeploymentError("adaptive cells are empty or duplicated")
    expected_thresholds = {
        (cell.pair_key.model, cell.pair_key.domain, cell.arm) for cell in cells
    }
    observed_thresholds = {
        (row.model, row.benchmark, row.method) for row in threshold_lock.methods
    }
    if observed_thresholds != expected_thresholds:
        raise AdaptiveDeploymentError(
            "threshold lock does not exactly cover adaptive method slices"
        )
    blocks: dict[str, list[JobCell]] = {}
    for cell in cells:
        blocks.setdefault(cell.block_id, []).append(cell)
        key = (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        try:
            task = task_index[key]
            threshold = threshold_lock.threshold_for(
                cell.pair_key.model, cell.pair_key.domain, cell.arm
            )
        except (KeyError, ValueError) as exc:
            raise AdaptiveDeploymentError("adaptive cell lacks its task or threshold") from exc
        _validate_cell_task_threshold(cell, task, threshold)

    all_methods = {cell.arm for cell in cells}
    all_operators = {cell.operator for cell in cells}
    block_identities: set[tuple[str, str, str, int, str]] = set()
    for block_id, block in blocks.items():
        identities = {
            (
                cell.pair_key.model,
                cell.pair_key.domain,
                cell.pair_key.task_id,
                cell.pair_key.replicate_id,
                str(cell.pair_key.task_sha256),
            )
            for cell in block
        }
        if len(identities) != 1:
            raise AdaptiveDeploymentError(f"adaptive block {block_id} mixes tasks")
        identity = next(iter(identities))
        if identity in block_identities:
            raise AdaptiveDeploymentError(
                "adaptive model/task/replicate identity is split across blocks"
            )
        block_identities.add(identity)
        by_method: dict[str, set[str]] = {}
        for cell in block:
            operators = by_method.setdefault(cell.arm, set())
            if cell.operator in operators:
                raise AdaptiveDeploymentError("adaptive method block duplicates an operator")
            operators.add(cell.operator)
        operator_sets = set(tuple(sorted(value)) for value in by_method.values())
        if (
            set(by_method) != all_methods
            or len(operator_sets) != 1
            or next(iter(operator_sets), ()) != tuple(sorted(all_operators))
        ):
            raise AdaptiveDeploymentError(
                "every adaptive block must contain the exact method/operator product"
            )
        for operators in by_method.values():
            if Operator.NONE.value not in operators or len(operators) < 2:
                raise AdaptiveDeploymentError(
                    "each adaptive method needs operator=none and at least one operator cell"
                )


def _manifest_mode(manifest: Mapping[str, Any]) -> None:
    extra = manifest.get("extra_config")
    if not isinstance(extra, Mapping):
        raise AdaptiveDeploymentError("adaptive manifest has no extra_config")
    if (
        extra.get("deployment_mode") != ADAPTIVE_DEPLOYMENT_MODE
        or extra.get("deployment_policy") != ADAPTIVE_POLICY
        or manifest.get("stage") != Stage.CONFIRMATORY.value
    ):
        raise AdaptiveDeploymentError(
            "manifest is not frozen for confirmatory online adaptive deployment"
        )
    launch = extra.get("scientific_launch_lock")
    if not isinstance(launch, Mapping):
        raise AdaptiveDeploymentError("adaptive manifest lacks its scientific launch lock")
    allocation = launch.get("source_allocation")
    projected = launch.get("projected_provider_usd")
    source_ids = allocation.get("source_ids") if isinstance(allocation, Mapping) else None
    if (
        not isinstance(allocation, Mapping)
        or allocation.get("benchmark") != Benchmark.EVOLVING_GSM8K.value
        or allocation.get("stage") != "deployment"
        or allocation.get("wave") is not None
        or not isinstance(source_ids, list)
        or not source_ids
        or len(source_ids) != len(set(source_ids))
        or any(not isinstance(item, str) or not item for item in source_ids)
        or extra.get("n_tasks") != len(source_ids)
        or not isinstance(projected, Mapping)
        or set(projected) != {"openai", "fireworks"}
        or isinstance(launch.get("required_n_tasks"), bool)
        or not isinstance(launch.get("required_n_tasks"), int)
        or not 1 <= launch["required_n_tasks"] <= len(source_ids)
    ):
        raise AdaptiveDeploymentError("adaptive scientific launch binding is invalid")
    for provider in ("openai", "fireworks"):
        try:
            amount = Decimal(str(projected[provider]))
        except Exception as exc:
            raise AdaptiveDeploymentError(
                "adaptive scientific cost projection is invalid"
            ) from exc
        if not amount.is_finite() or amount < 0:
            raise AdaptiveDeploymentError(
                "adaptive scientific cost projection is invalid"
            )

    receipts = manifest.get("benchmark_receipts")
    if not isinstance(receipts, list):
        raise AdaptiveDeploymentError("adaptive manifest receipts are invalid")

    def receipt_digest(name: str) -> str:
        matches = [
            row for row in receipts
            if isinstance(row, Mapping) and row.get("name") == name
        ]
        if len(matches) != 1:
            raise AdaptiveDeploymentError(f"adaptive manifest lacks exact {name}")
        return _digest(name, matches[0].get("sha256"))

    if receipt_digest("source_allocation_registry") != allocation.get(
        "registry_sha256"
    ) or receipt_digest("cost_sample_size_projection_lock") != launch.get(
        "projection_lock_sha256"
    ):
        raise AdaptiveDeploymentError("adaptive scientific receipt binding changed")
    receipt_digest("measured_baseline_resource_profile")
    realized = allocation.get("realized_allocation_sha256")
    if realized is not None and receipt_digest("realized_source_allocation") != _digest(
        "realized_allocation_sha256", realized
    ):
        raise AdaptiveDeploymentError("adaptive realized allocation receipt changed")


def _validate_manifest_matrix(
    *,
    manifest: Mapping[str, Any],
    cells: Sequence[JobCell],
    task_index: Mapping[tuple[str, str, str], DomainTask],
) -> None:
    extra = manifest["extra_config"]
    models = _unique_names(tuple(manifest.get("models", ())), "manifest models")
    methods = _unique_names(tuple(manifest.get("arms", ())), "manifest methods")
    operators = _unique_names(
        tuple(manifest.get("operators", ())), "manifest operators"
    )
    replicates = extra.get("replicates")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates != PRIMARY_REPLICATES
        or extra.get("n_tasks") != len(task_index)
    ):
        raise AdaptiveDeploymentError("adaptive manifest task/replicate lock is invalid")
    expected = {
        (model, domain, task_id, task_sha256, replicate, method, operator)
        for model in models
        for domain, task_id, task_sha256 in task_index
        for replicate in range(replicates)
        for method in methods
        for operator in operators
    }
    actual = {
        (
            cell.pair_key.model,
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
            cell.pair_key.replicate_id,
            cell.arm,
            cell.operator,
        )
        for cell in cells
    }
    if (
        actual != expected
        or len(actual) != len(cells)
        or extra.get("n_cells") != len(expected)
    ):
        raise AdaptiveDeploymentError(
            "adaptive cells do not equal the frozen task/model/method/operator product"
        )


def _unique_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise AdaptiveDeploymentError(f"{name} must be a sequence")
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value for value in result)
    ):
        raise AdaptiveDeploymentError(f"{name} must contain unique nonempty names")
    return result


def _runtime_config(
    config: HarnessConfig, compaction_config: CompactionConfig
) -> dict[str, Any]:
    if not isinstance(config, HarnessConfig) or not isinstance(
        compaction_config, CompactionConfig
    ):
        raise AdaptiveDeploymentError(
            "adaptive runtime needs typed harness/compaction configuration"
        )
    return {
        "checkpoint_every": config.checkpoint_every,
        "task_max_output_tokens": config.task_max_output_tokens,
        "probe_max_output_tokens": config.probe_max_output_tokens,
        "temperature": config.temperature,
        "compaction": {
            "keep_last_messages": compaction_config.keep_last_messages,
            "max_excerpt_bytes": compaction_config.max_excerpt_bytes,
            "max_summary_bytes": compaction_config.max_summary_bytes,
            "config_sha256": compaction_config.config_sha256,
        },
    }


def prepare_adaptive_run(
    *,
    deployment_run_id: str,
    task_manifest_path: str | Path,
    calibration_threshold_path: str | Path,
    calibration_extract_path: str | Path,
    calibration_manifest_path: str | Path,
    source_registry_path: str | Path,
    baseline_profile_path: str | Path,
    planning_lock_path: str | Path,
    realized_allocation_path: str | Path | None = None,
    tasks: Sequence[DomainTask],
    models: Sequence[str],
    methods: Sequence[str],
    operators: Sequence[str],
    natural_max_actions_per_task: int,
    randomization_seed: int,
    replicates: int = 1,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS,
    evolving_dataset_path: str | Path | None = None,
    evolving_build_receipt_path: str | Path | None = None,
    config: HarnessConfig = HarnessConfig(),
    compaction_config: CompactionConfig = CompactionConfig(),
) -> AdaptivePreparationResult:
    """Freeze an online run from a reproducible calibration derivation.

    This function is provider-free.  It recomputes the analysis thresholds,
    rejects global calibration/deployment task overlap through the shared
    preparation helper, and creates no pass-one trajectory or replay schedule.
    """

    model_names = _unique_names(models, "models")
    method_names = _unique_names(methods, "methods")
    operator_names = _unique_names(operators, "operators")
    if any(model not in DEFAULT_REASONING_EFFORT for model in model_names):
        raise AdaptiveDeploymentError("adaptive models are outside the frozen target slate")
    for method in method_names:
        _method_kind(method)
    if set(operator_names) - set(_OPERATORS):
        raise AdaptiveDeploymentError("adaptive operators contain an unsupported value")
    if Operator.NONE.value not in operator_names or len(operator_names) < 2:
        raise AdaptiveDeploymentError(
            "adaptive preparation needs operator=none and at least one operator"
        )
    if natural_max_actions_per_task != PRIMARY_MAX_ACTIONS_PER_TASK:
        raise AdaptiveDeploymentError(
            "primary online deployment requires exactly one intervention per task"
        )
    # Production adaptive deployment is a distinct source allocation within
    # the confirmatory provider budget; it is not a smoke-stage convenience.
    deployment_stage = Stage.CONFIRMATORY
    if (
        isinstance(randomization_seed, bool)
        or not isinstance(randomization_seed, int)
        or randomization_seed < 0
    ):
        raise AdaptiveDeploymentError("adaptive seed is invalid")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates != PRIMARY_REPLICATES
    ):
        raise AdaptiveDeploymentError(
            "primary online deployment requires exactly one replicate per source task"
        )
    runtime_config = _runtime_config(config, compaction_config)
    canonical_spec = validate_passive_monitor_spec(canonical_passive_monitor_spec())
    if (
        config.checkpoint_every != 1
        or config.temperature != canonical_spec["determinism"]["temperature"]
        or config.probe_max_output_tokens
        != canonical_spec["frozen_probe"]["max_output_tokens"]
    ):
        raise AdaptiveDeploymentError(
            "adaptive runtime differs from canonical checkpoint/probe determinism"
        )

    raw_files = (
        (Path(task_manifest_path), "task manifest"),
        (Path(calibration_threshold_path), "calibration thresholds"),
        (Path(calibration_extract_path), "calibration extract"),
        (Path(calibration_manifest_path), "calibration manifest"),
        (Path(source_registry_path), "source allocation registry"),
        (Path(baseline_profile_path), "measured baseline resource profile"),
        (Path(planning_lock_path), "cost/sample-size projection lock"),
    )
    for path, label in raw_files:
        if path.is_symlink() or not path.resolve().is_file():
            raise FileNotFoundError(f"{label} must be an existing non-symlink file")
    (
        task_file,
        threshold_file,
        extract_file,
        calibration_manifest_file,
        registry_file,
        baseline_profile_file,
        planning_lock_file,
    ) = (path.resolve() for path, _label in raw_files)
    realized_allocation_file: Path | None = None
    if realized_allocation_path is not None:
        candidate = Path(realized_allocation_path)
        if candidate.is_symlink() or not candidate.resolve().is_file():
            raise FileNotFoundError(
                "realized source allocation must be an existing non-symlink file"
            )
        realized_allocation_file = candidate.resolve()
    for path, label in (
        (task_file, "task manifest"),
        (threshold_file, "calibration thresholds"),
        (extract_file, "calibration extract"),
        (calibration_manifest_file, "calibration manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} must be an existing non-symlink file")
    task_rows = load_task_manifest(task_file)
    task_index = resolve_declared_tasks(tasks, task_rows)
    if any(
        row["benchmark"] != Benchmark.EVOLVING_GSM8K.value
        or row["condition"] != "t7"
        or row["num_turns"] < 2
        for row in task_rows
    ):
        raise AdaptiveDeploymentError(
            "online adaptive preparation requires Evolving Intent t7 task rows only"
        )
    threshold_payload, extract_sha256 = verify_analysis_threshold_derivation(
        threshold_file, extract_file
    )
    calibration_manifest_sha256 = sha256_file(calibration_manifest_file)
    if threshold_payload.get("source_manifest_sha256") != calibration_manifest_sha256:
        raise AdaptiveDeploymentError(
            "supplied calibration manifest differs from threshold provenance"
        )
    calibration_layout = RunLayout.for_run(
        artifacts_root, str(threshold_payload["source_run_id"])
    )
    if calibration_layout.manifest.resolve() != calibration_manifest_file:
        raise AdaptiveDeploymentError(
            "calibration manifest path is outside its declared run layout"
        )
    verify_calibration_extract_against_run(
        calibration_layout,
        extract_file,
        expected_manifest_sha256=calibration_manifest_sha256,
    )
    threshold_lock = deployment_threshold_lock_from_analysis(
        threshold_payload,
        deployment_task_rows=task_rows,
        models=model_names,
        methods=method_names,
        natural_max_actions_per_task=natural_max_actions_per_task,
        # Unused online, but the shared strict lock schema requires a positive
        # sensitivity budget and a declared anchor.
        matched_actions_per_method=1,
        yoke_anchor_method=method_names[0],
    )
    task_refs = tuple(
        TaskRef(row["benchmark"], row["task_id"], row["task_sha256"])
        for row in task_rows
    )
    cells = make_pair_manifest(
        tasks=task_refs,
        models=model_names,
        arms=method_names,
        operators=operator_names,
        replicates=replicates,
        randomization_seed=randomization_seed,
    )
    validate_adaptive_design(
        cells=cells, task_index=task_index, threshold_lock=threshold_lock
    )
    evolving_receipts = _evolving_provenance_receipts(
        list(task_rows),
        dataset_path=(
            None if evolving_dataset_path is None else str(evolving_dataset_path)
        ),
        build_receipt_path=(
            None
            if evolving_build_receipt_path is None
            else str(evolving_build_receipt_path)
        ),
    )
    destination = RunLayout.for_run(artifacts_root, deployment_run_id)
    if destination.root.exists():
        raise FileExistsError("adaptive run already exists; preparation is write-once")
    try:
        launch_binding = assert_scientific_launch(
            task_rows=task_rows,
            stage=deployment_stage,
            allocation_stage="deployment",
            design_family="deployment",
            models=model_names,
            arms=method_names,
            operators=operator_names,
            replicates=replicates,
            ledger_path=destination.ledger,
            registry_path=registry_file,
            projection_lock_path=planning_lock_file,
            baseline_profile_path=baseline_profile_file,
            realized_allocation_path=realized_allocation_file,
        )
    except ValueError as exc:
        raise AdaptiveDeploymentError(
            f"scientific adaptive launch gate failed: {exc}"
        ) from exc
    if not isinstance(launch_binding, ScientificLaunchBinding):
        raise AdaptiveDeploymentError(
            "scientific adaptive launch gate returned no projection binding"
        )

    # All external validation above occurs before the destination namespace is
    # created.  The remaining writes are local immutable preparation artifacts.
    destination.create()
    atomic_write_jsonl(destination.pairs, [cell.as_dict() for cell in cells])
    pair_sha256 = sha256_file(destination.pairs)
    threshold_lock_path = destination.results / "deployment_threshold_lock.json"
    threshold_lock_sha256 = freeze_threshold_lock(
        threshold_lock_path, threshold_lock
    )
    receipts = (
        ArtifactReceipt.from_file(
            "task_manifest", task_file, workspace=REPOSITORY_ROOT
        ),
        *evolving_receipts,
        ArtifactReceipt.from_file(
            CALIBRATION_MANIFEST_RECEIPT,
            calibration_manifest_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            CALIBRATION_EXTRACT_RECEIPT,
            extract_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            CALIBRATION_THRESHOLDS_RECEIPT,
            threshold_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            THRESHOLD_LOCK_RECEIPT,
            threshold_lock_path,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "source_allocation_registry",
            registry_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "measured_baseline_resource_profile",
            baseline_profile_file,
            workspace=REPOSITORY_ROOT,
        ),
        ArtifactReceipt.from_file(
            "cost_sample_size_projection_lock",
            planning_lock_file,
            workspace=REPOSITORY_ROOT,
        ),
        *(
            ()
            if realized_allocation_file is None
            else (
                ArtifactReceipt.from_file(
                    "realized_source_allocation",
                    realized_allocation_file,
                    workspace=REPOSITORY_ROOT,
                ),
            )
        ),
    )
    extra_config: dict[str, Any] = {
        "n_tasks": len(task_rows),
        "n_cells": len(cells),
        "replicates": replicates,
        "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
        "deployment_policy": ADAPTIVE_POLICY,
        "natural_max_actions_per_task": threshold_lock.natural_max_actions_per_task,
        "threshold_lock_sha256": threshold_lock_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "calibration_extract_sha256": extract_sha256,
        "adaptive_runtime": runtime_config,
        "scientific_launch_lock": launch_binding.as_dict(),
    }
    extra_config["analysis_lock"] = {
        "threshold_artifact_sha256": sha256_file(threshold_file),
        "calibration_manifest_sha256": calibration_manifest_sha256,
    }
    manifest = build_manifest(
        run_id=deployment_run_id,
        stage=deployment_stage,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_sha256=pair_sha256,
        models=model_names,
        arms=method_names,
        operators=operator_names,
        randomization_seed=randomization_seed,
        benchmark_receipts=receipts,
        extra_config=extra_config,
    )
    manifest_sha256 = write_manifest_once(destination.manifest, manifest)
    BudgetLedger(
        destination.ledger,
        operational_caps_usd={
            provider: Decimal(str(amount))
            for provider, amount in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    return AdaptivePreparationResult(
        run_id=deployment_run_id,
        manifest_path=destination.manifest,
        manifest_sha256=manifest_sha256,
        pair_manifest_path=destination.pairs,
        pair_manifest_sha256=pair_sha256,
        threshold_lock_path=threshold_lock_path,
        threshold_lock_sha256=threshold_lock_sha256,
        declared_cells=len(cells),
    )


async def _observe_current_prefix(
    *,
    run_id: str,
    cell: JobCell,
    task: DomainTask,
    checkpoint: int,
    checkpoint_index: int,
    messages: list[dict[str, Any]],
    task_records: Sequence[Mapping[str, Any]],
    transport: Transport,
    config: HarnessConfig,
    passive_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one exact current-prefix signal record.

    Active calls mutate ``messages`` by carrying the probe exchange.  Every
    passive call receives a fresh fork and the equality check at the end makes
    zero carry an enforced invariant rather than a comment.
    """

    kind, active_variant = _method_kind(cell.arm)
    before = sha256_json(messages)
    score: float
    call: dict[str, Any] | None = None
    raw_output: str | None = None
    observer_request_sha256: str | None = None
    monitor_spec_sha256: str
    details: dict[str, Any]
    model = cell.pair_key.model
    if model not in DEFAULT_REASONING_EFFORT:
        raise AdaptiveDeploymentError("target model has no frozen runtime settings")
    reasoning_effort = passive_spec["determinism"]["reasoning_effort_by_target"][model]
    temperature = passive_spec["determinism"]["temperature"]

    if active_variant is not None:
        instance = generate_probe_instance(
            active_variant, task.instance_id, checkpoint_index
        )
        probe_user = {"role": "user", "content": render_probe_prompt(instance)}
        messages.append(probe_user)
        observer_request_sha256 = sha256_json(messages)
        result = await transport.complete(
            model,
            messages,
            purpose="adaptive_active_probe",
            request_key=_request_key(run_id, cell.cell_id, "active-probe", checkpoint),
            input_token_estimate=conservative_input_token_bound(messages),
            max_output_tokens=config.probe_max_output_tokens,
            temperature=config.temperature,
            reasoning_effort=DEFAULT_REASONING_EFFORT[model],
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("active adaptive probe returned tool calls")
        probe_assistant = {"role": "assistant", "content": result.text}
        messages.append(probe_assistant)
        grade = grade_probe_response(instance, result.text)
        score = 0.0 if grade.passed else 1.0
        raw_output = result.text
        call = _call_record(result)
        monitor_spec_sha256 = sha256_json(
            {
                "method": cell.arm,
                "variant": active_variant,
                "checkpoint_index": checkpoint_index,
                "prompt": probe_user["content"],
                "expected_sha256": sha256_json(instance.expected_answer),
            }
        )
        details = {
            "variant": active_variant,
            "probe_user": probe_user,
            "probe_assistant": probe_assistant,
            "passed": grade.passed,
            "value_correct": grade.value_correct,
            "exact_format": grade.exact_format,
            "grade_error": grade.error,
            "expected_sha256": sha256_json(instance.expected_answer),
        }
    elif cell.arm.startswith("frozen_probe:"):
        variant = cell.arm.split(":", 1)[1]
        if variant not in tuple(passive_spec["frozen_probe"]["variants"]):
            raise AdaptiveDeploymentError("frozen probe variant changed")
        instance = generate_probe_instance(variant, task.instance_id, checkpoint_index)
        fork = build_frozen_probe_fork(messages, instance)
        observer_request_sha256 = sha256_json(fork.messages)
        result = await transport.complete(
            model,
            list(fork.messages),
            purpose="adaptive_frozen_probe",
            request_key=_request_key(run_id, cell.cell_id, "frozen-probe", checkpoint),
            input_token_estimate=conservative_input_token_bound(fork.messages),
            max_output_tokens=passive_spec["frozen_probe"]["max_output_tokens"],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("frozen adaptive probe returned tool calls")
        grade = grade_probe_response(instance, result.text)
        score = 0.0 if grade.passed else 1.0
        raw_output = result.text
        call = _call_record(result)
        monitor_spec_sha256 = fork.spec_sha256
        details = {
            "variant": variant,
            "passed": grade.passed,
            "value_correct": grade.value_correct,
            "exact_format": grade.exact_format,
            "grade_error": grade.error,
            "expected_sha256": sha256_json(instance.expected_answer),
        }
    elif cell.arm == "frozen_quiz":
        questions = generate_evolving_passive_quiz(task_records, checkpoint)
        fork = build_quiz_fork(messages, questions, checkpoint)
        observer_request_sha256 = sha256_json(fork)
        result = await transport.complete(
            model,
            fork,
            purpose="adaptive_frozen_quiz",
            request_key=_request_key(run_id, cell.cell_id, "frozen-quiz", checkpoint),
            input_token_estimate=conservative_input_token_bound(fork),
            max_output_tokens=passive_spec["frozen_quiz"]["max_output_tokens"],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("adaptive frozen quiz returned tool calls")
        grade = grade_quiz(
            questions,
            result.text,
            fire_at_wrong=passive_spec["frozen_quiz"]["fire_at_wrong"],
        )
        score = grade.risk
        raw_output = result.text
        call = _call_record(result)
        monitor_spec_sha256 = grade.spec_sha256
        details = {
            "n_wrong": grade.n_wrong,
            "question_ids": [question.question_id for question in questions],
            "quiz_generator": quiz_generator_spec(passive_spec, task.domain),
        }
    elif cell.arm == "trace_judge":
        judge = passive_spec["trace_judge"]
        if judge["enabled"] is not True:
            raise AdaptiveDeploymentError("canonical trace judge is disabled")
        request = build_judge_request(messages, checkpoint, benchmark=task.domain)
        observer_request_sha256 = sha256_json(request)
        schema = JsonSchemaOutput.from_schema("trace_risk", JUDGE_RESPONSE_SCHEMA)
        result = await transport.complete(
            judge["model"],
            request,
            purpose="adaptive_trace_judge",
            request_key=_request_key(run_id, cell.cell_id, "trace-judge", checkpoint),
            input_token_estimate=conservative_input_token_bound(
                request, extra_bytes=len(str(JUDGE_RESPONSE_SCHEMA).encode("utf-8"))
            ),
            max_output_tokens=judge["max_output_tokens"],
            temperature=temperature,
            reasoning_effort=judge["reasoning_effort"],
            output_schema=schema,
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("adaptive trace judge returned tool calls")
        verdict = parse_judge_output(result.text)
        score = verdict.risk
        raw_output = result.text
        call = _call_record(result)
        monitor_spec_sha256 = verdict.spec_sha256
        details = {
            "concerns": list(verdict.concerns),
            "evidence": list(verdict.evidence),
        }
    elif cell.arm == "trace_rules":
        copied_prefix = [dict(message) for message in messages]
        observer_request_sha256 = sha256_json(copied_prefix)
        result = score_trace_rules(
            copied_prefix,
            event_flags={},
            fire_threshold=passive_spec["trace_rules"]["fire_threshold"],
        )
        score = result.risk
        monitor_spec_sha256 = result.spec_sha256
        details = {"reasons": list(result.reasons), "natural_rule_fired": result.fired}
    elif cell.arm == "turn_clock":
        score = checkpoint / len(task.turns)
        monitor_spec_sha256 = sha256_json(passive_spec["baselines"]["turn_clock"])
        details = {"completed_turns": checkpoint, "task_horizon": len(task.turns)}
    elif cell.arm == "context_use":
        latest = task_records[-1].get("call")
        if not isinstance(latest, Mapping) or not isinstance(latest.get("usage"), Mapping):
            raise AdaptiveDeploymentError("context baseline lacks latest task-call usage")
        input_tokens = latest["usage"].get("input_tokens")
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
            raise AdaptiveDeploymentError("context baseline has invalid input tokens")
        context_window = CATALOG.models[model].context_window_tokens
        score = min(1.0, input_tokens / context_window)
        monitor_spec_sha256 = sha256_json(passive_spec["baselines"]["context_use"])
        details = {
            "raw_input_tokens": input_tokens,
            "context_window_tokens": context_window,
        }
    else:  # pragma: no cover - guarded by _method_kind.
        raise AssertionError("unreachable adaptive method")

    after = sha256_json(messages)
    carried = kind == "active_carry"
    if carried != (before != after):
        raise AdaptiveDeploymentError(
            "active/passive carry boundary disagrees with the target history"
        )
    core = {
        "event": "signal_observed",
        "method": cell.arm,
        "observation_kind": kind,
        "checkpoint": checkpoint,
        "checkpoint_index": checkpoint_index,
        "actionable_before_turn": checkpoint + 1,
        "score": float(score),
        "carried_into_target": carried,
        "source_prefix_before_observation_sha256": before,
        "source_prefix_sha256": after,
        "observer_request_sha256": observer_request_sha256,
        "observer_response_sha256": (
            None if raw_output is None else sha256_json(raw_output)
        ),
        "raw_output": raw_output,
        "monitor_spec_sha256": monitor_spec_sha256,
        "passive_monitor_spec_sha256": (
            None if carried else PASSIVE_MONITOR_SPEC_SHA256
        ),
        "details": details,
        "call": call,
    }
    return {**core, "signal_record_sha256": _record_hash(core, field="signal_record_sha256")}


def _decision_record(
    *,
    signal: Mapping[str, Any],
    threshold: LockedMethodThreshold,
    cap: int,
    actions_before: int,
) -> dict[str, Any]:
    _validate_hashed_record(signal, field="signal_record_sha256")
    score = signal.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
        or signal.get("method") != threshold.method
    ):
        raise AdaptiveDeploymentError("adaptive signal score/method is invalid")
    threshold_fired = float(score) >= threshold.threshold
    selected = threshold_fired and actions_before < cap
    core = {
        "event": "adaptive_decision",
        "policy": ADAPTIVE_POLICY,
        "method": threshold.method,
        "checkpoint": signal["checkpoint"],
        "actionable_before_turn": signal["actionable_before_turn"],
        "signal_record_sha256": signal["signal_record_sha256"],
        "source_prefix_sha256": signal["source_prefix_sha256"],
        "score": float(score),
        "locked_threshold": threshold.threshold,
        "threshold_record_sha256": threshold.lock_sha256,
        "threshold_fired": threshold_fired,
        "per_task_action_cap": cap,
        "actions_before": actions_before,
        "action_selected": selected,
        "actions_after": actions_before + int(selected),
    }
    return {**core, "decision_sha256": _record_hash(core, field="decision_sha256")}


def _apply_online_action(
    *,
    cell: JobCell,
    task: DomainTask,
    messages: Sequence[Mapping[str, Any]],
    signal: Mapping[str, Any],
    decision: Mapping[str, Any],
    instructions: Any,
    compaction_config: CompactionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_hashed_record(decision, field="decision_sha256")
    if decision.get("action_selected") is not True:
        raise AdaptiveDeploymentError("unselected adaptive action cannot be applied")
    checkpoint = int(decision["checkpoint"])
    prefix = freeze_visible_prefix(
        domain=task.domain,
        task_id=task.task_id,
        after_turn=checkpoint,
        messages=messages,
    )
    if prefix.prefix_sha256 != signal.get("source_prefix_sha256"):
        raise AdaptiveDeploymentError("adaptive signal and action prefixes differ")
    schedule = _online_schedule(cell, checkpoint, len(task.turns))
    reference = SignalReference(
        method=cell.arm,
        checkpoint=checkpoint,
        source_prefix_sha256=prefix.prefix_sha256,
        signal_record_sha256=signal["signal_record_sha256"],
        frozen_two_pass=False,
    )
    kind = _OPERATORS[cell.operator]
    public_state = None
    feedback = None
    compact = None
    if kind is InterventionType.REGROUND:
        public_state = freeze_public_state(
            domain=task.domain,
            task_id=task.task_id,
            after_turn=checkpoint,
            state=_public_state(task, checkpoint),
        )
    elif kind is InterventionType.FEEDBACK:
        feedback = _feedback_note(prefix)
    elif kind is InterventionType.COMPACT:
        compact = compaction_config
    application = apply_intervention(
        intervention_type=kind,
        prefix=prefix,
        schedule=schedule,
        member_id=cell.cell_id,
        checkpoint=checkpoint,
        instructions=(
            instructions
            if kind in {InterventionType.COMPACT, InterventionType.REGROUND}
            else None
        ),
        public_state=public_state,
        feedback=feedback,
        compaction_config=compact,
        signal=reference,
    )
    event = {
        **application.as_event(),
        "adaptive_policy": ADAPTIVE_POLICY,
        "decision_sha256": decision["decision_sha256"],
        "score": decision["score"],
        "locked_threshold": decision["locked_threshold"],
        "threshold_record_sha256": decision["threshold_record_sha256"],
        "per_task_action_cap": decision["per_task_action_cap"],
        "declared_operator": cell.operator,
        "observation_method": cell.arm,
        "feedback_generation": (
            "deterministic_current_prefix_quote_only"
            if kind is InterventionType.FEEDBACK
            else None
        ),
    }
    return application.continued_history, event


def _design(
    *,
    run_id: str,
    cell: JobCell,
    task: DomainTask,
    threshold: LockedMethodThreshold,
    threshold_lock: ThresholdLockArtifact,
    threshold_lock_sha256: str,
    manifest_sha256: str,
    pair_manifest_sha256: str,
    passive_spec: Mapping[str, Any],
    config: HarnessConfig,
    compaction_config: CompactionConfig,
) -> dict[str, Any]:
    return {
        "adaptive_runner_version": ADAPTIVE_RUNNER_VERSION,
        "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
        "deployment_policy": ADAPTIVE_POLICY,
        "run_id": run_id,
        "cell": cell.as_dict(),
        "task": task.manifest_record(),
        "manifest_sha256": _digest("manifest_sha256", manifest_sha256),
        "pair_manifest_sha256": _digest(
            "pair_manifest_sha256", pair_manifest_sha256
        ),
        "threshold_lock_sha256": _digest(
            "threshold_lock_sha256", threshold_lock_sha256
        ),
        "calibration_run_id": threshold_lock.calibration_run_id,
        "calibration_manifest_sha256": threshold_lock.calibration_manifest_sha256,
        "threshold": threshold.as_dict(),
        "threshold_record_sha256": threshold.lock_sha256,
        "per_task_action_cap": threshold_lock.natural_max_actions_per_task,
        "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
        "passive_monitor_spec": dict(passive_spec),
        "checkpoint_turns": list(range(1, len(task.turns))),
        "runtime_config": _runtime_config(config, compaction_config),
    }


def _replay_signal_record(
    *,
    cell: JobCell,
    task: DomainTask,
    checkpoint: int,
    messages: Sequence[Mapping[str, Any]],
    task_records: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
    passive_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute one stored signal without making an observer call."""

    kind, active_variant = _method_kind(cell.arm)
    carried_messages = [dict(message) for message in messages]
    before = sha256_json(carried_messages)
    raw_output = record.get("raw_output")
    call = record.get("call")
    score: float
    observer_request_sha256: str | None = None
    monitor_spec_sha256: str
    details: dict[str, Any]

    paid_method = active_variant is not None or cell.arm in {
        "frozen_quiz",
        "trace_judge",
    } or cell.arm.startswith("frozen_probe:")
    if paid_method:
        if not isinstance(raw_output, str) or not isinstance(call, Mapping):
            raise AdaptiveDeploymentError(
                "paid adaptive signal lacks its output/call during history replay"
            )
    elif raw_output is not None or call is not None:
        raise AdaptiveDeploymentError(
            "deterministic adaptive signal contains paid-output fields"
        )

    try:
        if active_variant is not None:
            instance = generate_probe_instance(
                active_variant, task.instance_id, checkpoint
            )
            probe_user = {
                "role": "user",
                "content": render_probe_prompt(instance),
            }
            carried_messages.append(probe_user)
            observer_request_sha256 = sha256_json(carried_messages)
            probe_assistant = {"role": "assistant", "content": raw_output}
            carried_messages.append(probe_assistant)
            grade = grade_probe_response(instance, raw_output)
            score = 0.0 if grade.passed else 1.0
            monitor_spec_sha256 = sha256_json(
                {
                    "method": cell.arm,
                    "variant": active_variant,
                    "checkpoint_index": checkpoint,
                    "prompt": probe_user["content"],
                    "expected_sha256": sha256_json(instance.expected_answer),
                }
            )
            details = {
                "variant": active_variant,
                "probe_user": probe_user,
                "probe_assistant": probe_assistant,
                "passed": grade.passed,
                "value_correct": grade.value_correct,
                "exact_format": grade.exact_format,
                "grade_error": grade.error,
                "expected_sha256": sha256_json(instance.expected_answer),
            }
        elif cell.arm.startswith("frozen_probe:"):
            variant = cell.arm.split(":", 1)[1]
            if variant not in tuple(passive_spec["frozen_probe"]["variants"]):
                raise AdaptiveDeploymentError("frozen probe variant changed")
            instance = generate_probe_instance(variant, task.instance_id, checkpoint)
            fork = build_frozen_probe_fork(carried_messages, instance)
            observer_request_sha256 = sha256_json(fork.messages)
            grade = grade_probe_response(instance, raw_output)
            score = 0.0 if grade.passed else 1.0
            monitor_spec_sha256 = fork.spec_sha256
            details = {
                "variant": variant,
                "passed": grade.passed,
                "value_correct": grade.value_correct,
                "exact_format": grade.exact_format,
                "grade_error": grade.error,
                "expected_sha256": sha256_json(instance.expected_answer),
            }
        elif cell.arm == "frozen_quiz":
            questions = generate_evolving_passive_quiz(task_records, checkpoint)
            fork = build_quiz_fork(carried_messages, questions, checkpoint)
            observer_request_sha256 = sha256_json(fork)
            grade = grade_quiz(
                questions,
                raw_output,
                fire_at_wrong=passive_spec["frozen_quiz"]["fire_at_wrong"],
            )
            score = grade.risk
            monitor_spec_sha256 = grade.spec_sha256
            details = {
                "n_wrong": grade.n_wrong,
                "question_ids": [question.question_id for question in questions],
                "quiz_generator": quiz_generator_spec(passive_spec, task.domain),
            }
        elif cell.arm == "trace_judge":
            judge = passive_spec["trace_judge"]
            if judge["enabled"] is not True:
                raise AdaptiveDeploymentError("canonical trace judge is disabled")
            request = build_judge_request(
                carried_messages, checkpoint, benchmark=task.domain
            )
            observer_request_sha256 = sha256_json(request)
            verdict = parse_judge_output(raw_output)
            score = verdict.risk
            monitor_spec_sha256 = verdict.spec_sha256
            details = {
                "concerns": list(verdict.concerns),
                "evidence": list(verdict.evidence),
            }
        elif cell.arm == "trace_rules":
            observer_request_sha256 = sha256_json(carried_messages)
            result = score_trace_rules(
                carried_messages,
                event_flags={},
                fire_threshold=passive_spec["trace_rules"]["fire_threshold"],
            )
            score = result.risk
            monitor_spec_sha256 = result.spec_sha256
            details = {
                "reasons": list(result.reasons),
                "natural_rule_fired": result.fired,
            }
        elif cell.arm == "turn_clock":
            score = checkpoint / len(task.turns)
            monitor_spec_sha256 = sha256_json(
                passive_spec["baselines"]["turn_clock"]
            )
            details = {
                "completed_turns": checkpoint,
                "task_horizon": len(task.turns),
            }
        elif cell.arm == "context_use":
            latest = task_records[-1].get("call")
            usage = latest.get("usage") if isinstance(latest, Mapping) else None
            input_tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
            if (
                isinstance(input_tokens, bool)
                or not isinstance(input_tokens, int)
                or input_tokens < 0
            ):
                raise AdaptiveDeploymentError(
                    "context baseline has invalid input tokens during history replay"
                )
            context_window = CATALOG.models[cell.pair_key.model].context_window_tokens
            score = min(1.0, input_tokens / context_window)
            monitor_spec_sha256 = sha256_json(
                passive_spec["baselines"]["context_use"]
            )
            details = {
                "raw_input_tokens": input_tokens,
                "context_window_tokens": context_window,
            }
        else:  # pragma: no cover - guarded by _method_kind.
            raise AssertionError("unreachable adaptive replay method")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AdaptiveDeploymentError):
            raise
        raise AdaptiveDeploymentError(
            "adaptive signal cannot be regenerated during history replay"
        ) from exc

    after = sha256_json(carried_messages)
    carried = kind == "active_carry"
    core = {
        "event": "signal_observed",
        "method": cell.arm,
        "observation_kind": kind,
        "checkpoint": checkpoint,
        "checkpoint_index": checkpoint,
        "actionable_before_turn": checkpoint + 1,
        "score": float(score),
        "carried_into_target": carried,
        "source_prefix_before_observation_sha256": before,
        "source_prefix_sha256": after,
        "observer_request_sha256": observer_request_sha256,
        "observer_response_sha256": (
            None if raw_output is None else sha256_json(raw_output)
        ),
        "raw_output": raw_output,
        "monitor_spec_sha256": monitor_spec_sha256,
        "passive_monitor_spec_sha256": (
            None if carried else PASSIVE_MONITOR_SPEC_SHA256
        ),
        "details": details,
        "call": call,
    }
    expected = {
        **core,
        "signal_record_sha256": _record_hash(core, field="signal_record_sha256"),
    }
    return expected, carried_messages


def replay_adaptive_history(
    *,
    cell: JobCell,
    task: DomainTask,
    start: Mapping[str, Any],
    output: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    compaction_config: CompactionConfig,
) -> None:
    """Rebuild the exact online target history and every deterministic action."""

    if not isinstance(task, DomainTask) or not isinstance(
        compaction_config, CompactionConfig
    ):
        raise AdaptiveDeploymentError(
            "adaptive history replay needs the frozen task/runtime configuration"
        )
    if start.get("task") != task.manifest_record():
        raise AdaptiveDeploymentError(
            "adaptive history replay task differs from the frozen dataset"
        )
    try:
        passive_spec = validate_passive_monitor_spec(
            start["passive_monitor_spec"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdaptiveDeploymentError(
            "adaptive history replay lacks the canonical passive monitor spec"
        ) from exc
    checkpoints = tuple(range(1, len(task.turns)))
    if (
        start.get("checkpoint_turns") != list(checkpoints)
        or start.get("passive_monitor_spec_sha256")
        != PASSIVE_MONITOR_SPEC_SHA256
        or output.get("checkpoint_turns") != list(checkpoints)
    ):
        raise AdaptiveDeploymentError(
            "adaptive history replay checkpoint/spec lock changed"
        )
    kind, active_variant = _method_kind(cell.arm)
    if (
        output.get("observation_method") != cell.arm
        or output.get("observation_kind") != kind
        or output.get("active_probe_variant") != active_variant
        or output.get("operator") != cell.operator
    ):
        raise AdaptiveDeploymentError("adaptive history replay identity changed")
    task_records = output.get("task_records")
    signal_records = output.get("signal_records")
    decisions = output.get("decision_records")
    interventions = output.get("intervention_records")
    if (
        not isinstance(task_records, list)
        or not isinstance(signal_records, list)
        or not isinstance(decisions, list)
        or not isinstance(interventions, list)
        or len(task_records) != len(task.turns)
        or len(signal_records) != len(checkpoints)
        or len(decisions) != len(checkpoints)
    ):
        raise AdaptiveDeploymentError(
            "adaptive history replay record coverage is incomplete"
        )
    try:
        threshold = LockedMethodThreshold.from_dict(
            start["threshold"], where="adaptive replay threshold"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdaptiveDeploymentError(
            "adaptive history replay threshold is invalid"
        ) from exc

    setup = (
        None
        if active_variant is None
        else render_initial_instruction(
            active_variant, task.instance_id, checkpoints
        )
    )
    first_content = task.turns[0].user_message
    if setup:
        first_content = setup + "\n\n--- BENCHMARK MESSAGE ---\n" + first_content
    instructions = freeze_initial_instructions(
        domain=task.domain,
        task_id=task.task_id,
        messages=({"role": "user", "content": first_content},),
    )
    replayed_events: list[dict[str, Any]] = [dict(start)]
    replayed_task_records: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    assistant_messages: list[str] = []
    intervention_index = 0
    actions_before = 0
    for turn_number, turn in enumerate(task.turns, 1):
        record = task_records[turn_number - 1]
        if not isinstance(record, Mapping):
            raise AdaptiveDeploymentError(
                "adaptive history replay task record is not an object"
            )
        content = first_content if turn_number == 1 else turn.user_message
        user = {"role": "user", "content": content}
        messages.append(user)
        assistant = record.get("assistant_message")
        if (
            not isinstance(assistant, Mapping)
            or set(assistant) != {"role", "content"}
            or assistant.get("role") != "assistant"
            or not isinstance(assistant.get("content"), str)
        ):
            raise AdaptiveDeploymentError(
                "adaptive history replay assistant message is invalid"
            )
        expected_task = {
            "event": "task_turn",
            "task_turn": turn_number,
            "user_message": user,
            "assistant_message": dict(assistant),
            "request_prefix_sha256": sha256_json(messages),
            "call": record.get("call"),
            "continued_history_sha256": "",
        }
        messages.append(dict(assistant))
        expected_task["continued_history_sha256"] = sha256_json(messages)
        if dict(record) != expected_task:
            raise AdaptiveDeploymentError(
                "adaptive task request/history does not replay exactly"
            )
        replayed_task_records.append(expected_task)
        replayed_events.append(expected_task)
        assistant_messages.append(str(assistant["content"]))
        if turn_number == len(task.turns):
            continue

        signal = signal_records[turn_number - 1]
        if not isinstance(signal, Mapping):
            raise AdaptiveDeploymentError(
                "adaptive history replay signal is not an object"
            )
        expected_signal, messages = _replay_signal_record(
            cell=cell,
            task=task,
            checkpoint=turn_number,
            messages=messages,
            task_records=replayed_task_records,
            record=signal,
            passive_spec=passive_spec,
        )
        if dict(signal) != expected_signal:
            raise AdaptiveDeploymentError(
                "adaptive signal prompt/grade/history does not replay exactly"
            )
        replayed_events.append(expected_signal)
        decision = decisions[turn_number - 1]
        expected_decision = _decision_record(
            signal=expected_signal,
            threshold=threshold,
            cap=int(start["per_task_action_cap"]),
            actions_before=actions_before,
        )
        if not isinstance(decision, Mapping) or dict(decision) != expected_decision:
            raise AdaptiveDeploymentError(
                "adaptive decision does not replay exactly"
            )
        replayed_events.append(expected_decision)
        actions_before = int(expected_decision["actions_after"])
        if expected_decision["action_selected"] is not True:
            continue
        if intervention_index >= len(interventions):
            raise AdaptiveDeploymentError(
                "adaptive selected action lacks its replayable intervention"
            )
        try:
            continued, expected_intervention = _apply_online_action(
                cell=cell,
                task=task,
                messages=messages,
                signal=expected_signal,
                decision=expected_decision,
                instructions=instructions,
                compaction_config=compaction_config,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdaptiveDeploymentError(
                "adaptive operator history cannot be replayed"
            ) from exc
        intervention = interventions[intervention_index]
        if (
            not isinstance(intervention, Mapping)
            or dict(intervention) != expected_intervention
        ):
            raise AdaptiveDeploymentError(
                "adaptive operator continued history does not replay exactly"
            )
        intervention_index += 1
        replayed_events.append(expected_intervention)
        messages = continued

    if intervention_index != len(interventions):
        raise AdaptiveDeploymentError(
            "adaptive history replay has undeclared interventions"
        )
    if list(events[:-1]) != replayed_events:
        raise AdaptiveDeploymentError(
            "adaptive chronological event history does not replay exactly"
        )
    if output.get("probe_records") != [
        record
        for record in signal_records
        if record.get("observation_kind") == "active_carry"
    ]:
        raise AdaptiveDeploymentError("adaptive carried-probe rollup changed")
    if output.get("task_assistant_messages") != assistant_messages:
        raise AdaptiveDeploymentError("adaptive assistant-message rollup changed")
    if (
        output.get("messages") != messages
        or output.get("transcript_sha256") != sha256_json(messages)
    ):
        raise AdaptiveDeploymentError("adaptive final carried history changed")
    prediction, success = grade_final_numeric(
        assistant_messages[-1], task.evaluation_label
    )
    expected_evaluation = {
        "prediction": prediction,
        "evaluation_label_sha256": (
            None
            if task.evaluation_label is None
            else sha256_json(task.evaluation_label)
        ),
        "success": success,
    }
    if output.get("evaluation") != expected_evaluation:
        raise AdaptiveDeploymentError(
            "adaptive final outcome does not reproduce from frozen task history"
        )


def _validate_existing(
    *,
    output_file: Path,
    event_file: Path,
    start: Mapping[str, Any],
    task: DomainTask,
    compaction_config: CompactionConfig,
) -> dict[str, Any]:
    output = read_json(output_file)
    if (
        output.get("schema_version") != ADAPTIVE_SCHEMA_VERSION
        or output.get("adaptive_runner_version") != ADAPTIVE_RUNNER_VERSION
        or output.get("complete") is not True
        or output.get("design_sha256") != start.get("design_sha256")
        or output.get("transcript_sha256") != sha256_json(output.get("messages"))
    ):
        raise AdaptiveDeploymentError("existing adaptive output is torn or changed")
    frozen_cell = start.get("cell")
    frozen_task = start.get("task")
    if (
        not isinstance(frozen_cell, Mapping)
        or not isinstance(frozen_task, Mapping)
        or output.get("run_id") != start.get("run_id")
        or output.get("cell_id") != frozen_cell.get("cell_id")
        or output.get("model") != frozen_cell.get("pair_key", {}).get("model")
        or output.get("domain") != frozen_task.get("domain")
        or output.get("task_id") != frozen_task.get("task_id")
        or output.get("task_sha256") != frozen_task.get("task_sha256")
        or output.get("observation_method") != frozen_cell.get("arm")
        or output.get("operator") != frozen_cell.get("operator")
        or output.get("manifest_sha256") != start.get("manifest_sha256")
        or output.get("pair_manifest_sha256") != start.get("pair_manifest_sha256")
        or output.get("threshold_lock_sha256")
        != start.get("threshold_lock_sha256")
        or output.get("threshold_record_sha256")
        != start.get("threshold_record_sha256")
    ):
        raise AdaptiveDeploymentError("existing adaptive output identity changed")
    if not event_file.is_file():
        raise AdaptiveDeploymentError("adaptive output lacks its event log")
    events = read_jsonl(event_file)
    if len(events) < 2 or events[0] != dict(start) or events[-1].get("event") != "complete":
        raise AdaptiveDeploymentError("adaptive event log is partial or belongs elsewhere")
    complete = events[-1]
    expected_complete = {
        "event": "complete",
        "design_sha256": start.get("design_sha256"),
        "task_turns": len(output.get("task_records", ())),
        "signals": len(output.get("signal_records", ())),
        "selected_actions": sum(
            bool(row.get("action_selected"))
            for row in output.get("decision_records", ())
            if isinstance(row, Mapping)
        ),
        "transcript_sha256": output.get("transcript_sha256"),
        "output_sha256": sha256_file(output_file),
        "prediction": output.get("evaluation", {}).get("prediction"),
        "success": output.get("evaluation", {}).get("success"),
    }
    if dict(complete) != expected_complete:
        raise AdaptiveDeploymentError("adaptive completion receipt changed")
    if output.get("event_log_prefix_sha256") != sha256_json(events[:-1]):
        raise AdaptiveDeploymentError("adaptive output does not bind its event prefix")
    task_records = output.get("task_records")
    signal_records = output.get("signal_records")
    decisions = output.get("decision_records")
    interventions = output.get("intervention_records")
    if any(
        not isinstance(value, list)
        for value in (task_records, signal_records, decisions, interventions)
    ):
        raise AdaptiveDeploymentError("adaptive output record arrays are invalid")
    logged = events[:-1]
    if task_records != [row for row in logged if row.get("event") == "task_turn"]:
        raise AdaptiveDeploymentError("adaptive task records differ from the event log")
    if signal_records != [
        row for row in logged if row.get("event") == "signal_observed"
    ]:
        raise AdaptiveDeploymentError("adaptive signal records differ from the event log")
    if decisions != [
        row for row in logged if row.get("event") == "adaptive_decision"
    ]:
        raise AdaptiveDeploymentError("adaptive decisions differ from the event log")
    if interventions != [
        row for row in logged if row.get("event") == "intervention_applied"
    ]:
        raise AdaptiveDeploymentError(
            "adaptive interventions differ from the event log"
        )
    replay_adaptive_history(
        cell=JobCell.from_dict(start["cell"]),
        task=task,
        start=start,
        output=output,
        events=events,
        compaction_config=compaction_config,
    )
    for record in signal_records:
        _validate_hashed_record(record, field="signal_record_sha256")
    for record in decisions:
        _validate_hashed_record(record, field="decision_sha256")
    if [row.get("checkpoint") for row in signal_records] != output.get(
        "checkpoint_turns"
    ):
        raise AdaptiveDeploymentError("adaptive output has incomplete signal coverage")
    if [row.get("signal_record_sha256") for row in decisions] != [
        row.get("signal_record_sha256") for row in signal_records
    ]:
        raise AdaptiveDeploymentError("adaptive decisions do not exactly cover signals")
    if [row.get("continued_history_sha256") for row in task_records[:-1]] != [
        row.get("source_prefix_before_observation_sha256")
        for row in signal_records
    ]:
        raise AdaptiveDeploymentError(
            "adaptive signals do not observe the exact post-task prefixes"
        )
    expected_kind, _variant = _method_kind(str(output.get("observation_method")))
    for signal in signal_records:
        carried = signal.get("carried_into_target")
        if (
            signal.get("observation_kind") != expected_kind
            or carried != (expected_kind == "active_carry")
            or (
                carried is False
                and signal.get("source_prefix_before_observation_sha256")
                != signal.get("source_prefix_sha256")
            )
            or (
                carried is True
                and signal.get("source_prefix_before_observation_sha256")
                == signal.get("source_prefix_sha256")
            )
        ):
            raise AdaptiveDeploymentError("adaptive carry/zero-carry receipt changed")
    try:
        locked = LockedMethodThreshold.from_dict(
            start["threshold"], where="adaptive start threshold"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdaptiveDeploymentError("adaptive start threshold is invalid") from exc
    actions_before = 0
    for signal, decision in zip(signal_records, decisions):
        expected = _decision_record(
            signal=signal,
            threshold=locked,
            cap=int(start["per_task_action_cap"]),
            actions_before=actions_before,
        )
        if decision != expected:
            raise AdaptiveDeploymentError(
                "adaptive decision does not reproduce from signal and lock"
            )
        actions_before = int(decision["actions_after"])
    if len(interventions) != sum(bool(row.get("action_selected")) for row in decisions):
        raise AdaptiveDeploymentError("adaptive selected actions do not match interventions")
    selected_decisions = [row for row in decisions if row.get("action_selected") is True]
    expected_intervention = _OPERATORS[str(output["operator"])].value
    for decision, intervention in zip(selected_decisions, interventions):
        if (
            intervention.get("decision_sha256") != decision.get("decision_sha256")
            or intervention.get("checkpoint") != decision.get("checkpoint")
            or intervention.get("source_prefix_sha256")
            != decision.get("source_prefix_sha256")
            or intervention.get("signal_source_prefix_sha256")
            != decision.get("source_prefix_sha256")
            or intervention.get("signal_record_sha256")
            != decision.get("signal_record_sha256")
            or intervention.get("signal_method") != output.get("observation_method")
            or intervention.get("signal_frozen_two_pass") is not False
            or intervention.get("declared_operator") != output.get("operator")
            or intervention.get("intervention_type") != expected_intervention
        ):
            raise AdaptiveDeploymentError(
                "adaptive intervention does not bind its same-pass decision"
            )
    calculated = _accounting([*task_records, *signal_records])
    if output.get("accounting") != calculated:
        raise AdaptiveDeploymentError("adaptive accounting does not reconcile")
    if output.get("observation_burden") != {
        "checkpoints": len(signal_records),
        "paid_observer_calls": calculated["by_category"]["observer"]["calls"],
        "carried_probe_calls": sum(
            row.get("observation_kind") == "active_carry" for row in signal_records
        ),
    }:
        raise AdaptiveDeploymentError("adaptive observation burden changed")
    return output


async def run_adaptive_task(
    *,
    run_id: str,
    cell: JobCell,
    task: DomainTask,
    threshold: LockedMethodThreshold,
    threshold_lock: ThresholdLockArtifact,
    threshold_lock_sha256: str,
    manifest_sha256: str,
    pair_manifest_sha256: str,
    passive_monitor_spec: Mapping[str, Any],
    transport: Transport,
    event_path: str | Path,
    output_path: str | Path,
    yes_spend: bool = False,
    config: HarnessConfig = HarnessConfig(),
    compaction_config: CompactionConfig = CompactionConfig(),
) -> dict[str, Any]:
    """Execute one current-prefix adaptive Evolving cell exactly once."""

    if not yes_spend:
        raise AdaptiveDeploymentError("provider dispatch requires explicit yes_spend=True")
    if not isinstance(config, HarnessConfig) or config.checkpoint_every != 1:
        raise AdaptiveDeploymentError(
            "adaptive deployment observes every non-final turn; checkpoint_every must be 1"
        )
    if not isinstance(compaction_config, CompactionConfig):
        raise AdaptiveDeploymentError("compaction_config must be CompactionConfig")
    if not isinstance(threshold_lock, ThresholdLockArtifact):
        raise AdaptiveDeploymentError("threshold_lock must be a frozen deployment lock")
    _validate_cell_task_threshold(cell, task, threshold)
    passive_spec = validate_passive_monitor_spec(passive_monitor_spec)
    if (
        config.temperature != passive_spec["determinism"]["temperature"]
        or config.probe_max_output_tokens
        != passive_spec["frozen_probe"]["max_output_tokens"]
    ):
        raise AdaptiveDeploymentError(
            "adaptive probe settings differ from the canonical monitor contract"
        )
    if cell.pair_key.model not in CATALOG.models:
        raise AdaptiveDeploymentError("adaptive target model is outside the frozen catalog")
    design = _design(
        run_id=run_id,
        cell=cell,
        task=task,
        threshold=threshold,
        threshold_lock=threshold_lock,
        threshold_lock_sha256=threshold_lock_sha256,
        manifest_sha256=manifest_sha256,
        pair_manifest_sha256=pair_manifest_sha256,
        passive_spec=passive_spec,
        config=config,
        compaction_config=compaction_config,
    )
    design_sha256 = sha256_json(design)
    start = {"event": "start", "design_sha256": design_sha256, **design}
    output_file = Path(output_path)
    event_file = Path(event_path)
    if output_file.exists():
        return _validate_existing(
            output_file=output_file,
            event_file=event_file,
            start=start,
            task=task,
            compaction_config=compaction_config,
        )
    if event_file.exists():
        events = read_jsonl(event_file)
        if not events or events[0] != start:
            raise AdaptiveDeploymentError("adaptive partial belongs to another design")
        raise AdaptiveDeploymentError(
            "partial adaptive event log cannot be resumed without duplicate-call risk"
        )
    event_file.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(event_file, start)

    checkpoints = tuple(range(1, len(task.turns)))
    _kind, active_variant = _method_kind(cell.arm)
    setup = (
        None
        if active_variant is None
        else render_initial_instruction(active_variant, task.instance_id, checkpoints)
    )
    first_content = task.turns[0].user_message
    if setup:
        first_content = setup + "\n\n--- BENCHMARK MESSAGE ---\n" + first_content
    instructions = freeze_initial_instructions(
        domain=task.domain,
        task_id=task.task_id,
        messages=({"role": "user", "content": first_content},),
    )

    messages: list[dict[str, Any]] = []
    assistant_task: list[str] = []
    events: list[dict[str, Any]] = [start]
    task_records: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    actions = 0

    for turn_number, turn in enumerate(task.turns, 1):
        content = first_content if turn_number == 1 else turn.user_message
        user = {"role": "user", "content": content}
        messages.append(user)
        request_prefix_sha256 = sha256_json(messages)
        result = await transport.complete(
            cell.pair_key.model,
            messages,
            purpose="adaptive_agent_turn",
            request_key=_request_key(run_id, cell.cell_id, "task", turn_number),
            input_token_estimate=conservative_input_token_bound(messages),
            max_output_tokens=config.task_max_output_tokens,
            temperature=config.temperature,
            reasoning_effort=DEFAULT_REASONING_EFFORT[cell.pair_key.model],
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("adaptive Evolving task returned tool calls")
        assistant = {"role": "assistant", "content": result.text}
        messages.append(assistant)
        assistant_task.append(result.text)
        task_event = {
            "event": "task_turn",
            "task_turn": turn_number,
            "user_message": user,
            "assistant_message": assistant,
            "request_prefix_sha256": request_prefix_sha256,
            "call": _call_record(result),
            "continued_history_sha256": sha256_json(messages),
        }
        append_jsonl(event_file, task_event)
        events.append(task_event)
        task_records.append(task_event)
        if turn_number == len(task.turns):
            continue

        signal = await _observe_current_prefix(
            run_id=run_id,
            cell=cell,
            task=task,
            checkpoint=turn_number,
            checkpoint_index=turn_number,
            messages=messages,
            task_records=task_records,
            transport=transport,
            config=config,
            passive_spec=passive_spec,
        )
        if (
            signal["source_prefix_before_observation_sha256"]
            != task_event["continued_history_sha256"]
        ):
            raise AssertionError("adaptive observer did not receive the current task prefix")
        append_jsonl(event_file, signal)
        events.append(signal)
        signal_records.append(signal)
        decision = _decision_record(
            signal=signal,
            threshold=threshold,
            cap=threshold_lock.natural_max_actions_per_task,
            actions_before=actions,
        )
        append_jsonl(event_file, decision)
        events.append(decision)
        decision_records.append(decision)
        if not decision["action_selected"]:
            continue
        messages, intervention = _apply_online_action(
            cell=cell,
            task=task,
            messages=messages,
            signal=signal,
            decision=decision,
            instructions=instructions,
            compaction_config=compaction_config,
        )
        actions += 1
        if actions != decision["actions_after"]:
            raise AssertionError("adaptive action count drifted")
        append_jsonl(event_file, intervention)
        events.append(intervention)
        intervention_records.append(intervention)

    if len(signal_records) != len(checkpoints) or len(decision_records) != len(checkpoints):
        raise AssertionError("adaptive sensing did not cover every non-final checkpoint")
    prediction, success = grade_final_numeric(
        assistant_task[-1], task.evaluation_label
    )
    transcript_sha256 = sha256_json(messages)
    accounting = _accounting([*task_records, *signal_records])
    materialized = {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "adaptive_runner_version": ADAPTIVE_RUNNER_VERSION,
        "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
        "deployment_policy": ADAPTIVE_POLICY,
        "run_id": run_id,
        "cell_id": cell.cell_id,
        "design_sha256": design_sha256,
        "manifest_sha256": manifest_sha256,
        "pair_manifest_sha256": pair_manifest_sha256,
        "threshold_lock_sha256": threshold_lock_sha256,
        "threshold_record_sha256": threshold.lock_sha256,
        "calibration_run_id": threshold_lock.calibration_run_id,
        "calibration_manifest_sha256": threshold_lock.calibration_manifest_sha256,
        "model": cell.pair_key.model,
        "domain": task.domain,
        "task_id": task.task_id,
        "condition": task.condition,
        "task_sha256": task.task_sha256,
        "observation_method": cell.arm,
        "observation_kind": _method_kind(cell.arm)[0],
        "active_probe_variant": active_variant,
        "operator": cell.operator,
        "per_task_action_cap": threshold_lock.natural_max_actions_per_task,
        "checkpoint_turns": list(checkpoints),
        "messages": messages,
        "task_assistant_messages": assistant_task,
        "task_records": task_records,
        "signal_records": signal_records,
        "decision_records": decision_records,
        "intervention_records": intervention_records,
        "probe_records": [
            record
            for record in signal_records
            if record["observation_kind"] == "active_carry"
        ],
        "evaluation": {
            "prediction": prediction,
            "evaluation_label_sha256": (
                None
                if task.evaluation_label is None
                else sha256_json(task.evaluation_label)
            ),
            "success": success,
        },
        "transcript_sha256": transcript_sha256,
        "accounting": accounting,
        "observation_burden": {
            "checkpoints": len(signal_records),
            "paid_observer_calls": accounting["by_category"]["observer"]["calls"],
            "carried_probe_calls": sum(
                record["observation_kind"] == "active_carry"
                for record in signal_records
            ),
        },
        "event_log_prefix_sha256": sha256_json(events),
        "complete": True,
    }
    atomic_write_json(output_file, materialized)
    append_jsonl(
        event_file,
        {
            "event": "complete",
            "design_sha256": design_sha256,
            "task_turns": len(task_records),
            "signals": len(signal_records),
            "selected_actions": actions,
            "transcript_sha256": transcript_sha256,
            "output_sha256": sha256_file(output_file),
            "prediction": prediction,
            "success": success,
        },
    )
    return materialized


def _require_receipt(
    manifest: Mapping[str, Any], *, name: str, path: str | Path
) -> str:
    digest = sha256_file(path)
    matches = [
        receipt
        for receipt in manifest.get("benchmark_receipts", ())
        if isinstance(receipt, Mapping) and receipt.get("name") == name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise AdaptiveDeploymentError(f"runtime {name} differs from the frozen manifest")
    return digest


def _job_state(
    path: Path, *, cell: JobCell, state: str, detail: Mapping[str, Any]
) -> None:
    if state not in {"complete", "failed"}:
        raise AdaptiveDeploymentError("invalid adaptive job state")
    atomic_write_json(
        path,
        {
            "adaptive_runner_version": ADAPTIVE_RUNNER_VERSION,
            "cell_id": cell.cell_id,
            "state": state,
            **dict(detail),
        },
    )


async def execute_adaptive_run(
    *,
    run_id: str,
    task_manifest_path: str | Path,
    threshold_lock_path: str | Path,
    tasks: Sequence[DomainTask],
    yes_spend: bool = False,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS,
    environ: Mapping[str, str] | None = None,
    max_new_cells: int | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    config: HarnessConfig = HarnessConfig(),
    compaction_config: CompactionConfig = CompactionConfig(),
    transport: Transport | None = None,
    evolving_dataset_path: str | Path | None = None,
    evolving_build_receipt_path: str | Path | None = None,
) -> RunSummary:
    """Validate the prepared online run, then dispatch declared cells."""

    # This gate intentionally precedes every artifact read and directory write.
    if not yes_spend:
        raise AdaptiveDeploymentError("provider dispatch requires explicit yes_spend=True")
    if max_new_cells is not None and (
        isinstance(max_new_cells, bool)
        or not isinstance(max_new_cells, int)
        or max_new_cells < 1
    ):
        raise AdaptiveDeploymentError("max_new_cells must be positive or null")
    try:
        shard = ExecutionShard(count=shard_count, index=shard_index)
    except ValueError as exc:
        raise AdaptiveDeploymentError(str(exc)) from exc
    layout = RunLayout.for_run(artifacts_root, run_id)
    manifest, cells, task_index = _validate_run_inputs(
        layout=layout,
        task_manifest_path=Path(task_manifest_path),
        tasks=tasks,
    )
    _manifest_mode(manifest)
    _validate_manifest_matrix(
        manifest=manifest, cells=cells, task_index=task_index
    )
    try:
        passive_spec = passive_monitor_spec_from_manifest(manifest)
    except ValueError as exc:
        raise AdaptiveDeploymentError(str(exc)) from exc
    _validate_evolving_runtime_provenance(
        manifest=manifest,
        cells=cells,
        task_index=task_index,
        dataset_path=evolving_dataset_path,
        build_receipt_path=evolving_build_receipt_path,
    )
    threshold_path = Path(threshold_lock_path)
    threshold_digest = _require_receipt(
        manifest, name=THRESHOLD_LOCK_RECEIPT, path=threshold_path
    )
    threshold_lock = load_threshold_lock(threshold_path)
    extra = manifest["extra_config"]
    if (
        extra.get("threshold_lock_sha256") != threshold_digest
        or extra.get("natural_max_actions_per_task")
        != threshold_lock.natural_max_actions_per_task
        or extra.get("calibration_manifest_sha256")
        != threshold_lock.calibration_manifest_sha256
        or extra.get("adaptive_runtime")
        != _runtime_config(config, compaction_config)
    ):
        raise AdaptiveDeploymentError("adaptive manifest threshold provenance changed")
    validate_adaptive_design(
        cells=cells, task_index=task_index, threshold_lock=threshold_lock
    )
    stage = Stage(manifest["stage"])
    if stage is Stage.OFFLINE:
        raise AdaptiveDeploymentError("offline stage cannot dispatch model calls")
    shard_cells = shard.select(cells)
    if transport is None:
        ledger = _stage_ledger(layout, run_id, stage)
        transport = Transport(
            ledger,
            layout.events / "call_attempts.jsonl",
            environ=environ,
            max_attempts=3 if stage is Stage.SMOKE else 6,
        )

    output_root = layout.results / ADAPTIVE_RESULT_SUBDIR
    job_root = layout.results / ADAPTIVE_JOB_SUBDIR
    output_root.mkdir(parents=True, exist_ok=True)
    job_root.mkdir(parents=True, exist_ok=True)
    manifest_digest = sha256_file(layout.manifest)
    pair_digest = sha256_file(layout.pairs)
    completed = failed = skipped = visited = new_cells = 0
    for cell in shard_cells:
        output = output_root / f"{cell.cell_id}.json"
        job = job_root / f"{cell.cell_id}.json"
        key = (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        threshold = threshold_lock.threshold_for(
            cell.pair_key.model, cell.pair_key.domain, cell.arm
        )
        kwargs = dict(
            run_id=run_id,
            cell=cell,
            task=task_index[key],
            threshold=threshold,
            threshold_lock=threshold_lock,
            threshold_lock_sha256=threshold_digest,
            manifest_sha256=manifest_digest,
            pair_manifest_sha256=pair_digest,
            passive_monitor_spec=passive_spec,
            transport=transport,
            event_path=layout.events / f"adaptive-{cell.cell_id}.jsonl",
            output_path=output,
            yes_spend=True,
            config=config,
            compaction_config=compaction_config,
        )
        if output.exists():
            existing = await run_adaptive_task(**kwargs)
            if job.exists():
                receipt = read_json(job)
                if (
                    receipt.get("state") != "complete"
                    or receipt.get("output_sha256") != sha256_file(output)
                ):
                    raise AdaptiveDeploymentError(
                        "adaptive output conflicts with its job receipt"
                    )
            else:
                _job_state(
                    job,
                    cell=cell,
                    state="complete",
                    detail={
                        "output_sha256": sha256_file(output),
                        "success": existing["evaluation"]["success"],
                        "accounting_sha256": sha256_json(existing["accounting"]),
                    },
                )
            completed += 1
            skipped += 1
            continue
        if job.exists():
            receipt = read_json(job)
            if receipt.get("state") == "failed":
                failed += 1
                skipped += 1
                continue
            raise AdaptiveDeploymentError("adaptive job receipt exists without output")
        if max_new_cells is not None and new_cells >= max_new_cells:
            skipped += 1
            continue
        visited += 1
        new_cells += 1
        try:
            result = await run_adaptive_task(**kwargs)
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
            detail={
                "output_sha256": sha256_file(output),
                "success": result["evaluation"]["success"],
                "accounting_sha256": sha256_json(result["accounting"]),
            },
        )
        completed += 1
    return RunSummary(
        declared_cells=len(cells),
        visited_cells=visited,
        completed_cells=completed,
        failed_cells=failed,
        skipped_cells=skipped,
        phase="adaptive_deployment",
        shard_count=shard.count,
        shard_index=shard.index,
        shard_cells=len(shard_cells),
    )


def adaptive_completeness(
    layout: RunLayout, cells: Sequence[JobCell]
) -> CompletenessReport:
    states: list[tuple[str, str]] = []
    for cell in cells:
        output = layout.results / ADAPTIVE_RESULT_SUBDIR / f"{cell.cell_id}.json"
        job = layout.results / ADAPTIVE_JOB_SUBDIR / f"{cell.cell_id}.json"
        if output.is_file() and job.is_file() and read_json(job).get("state") == "complete":
            states.append((cell.cell_id, "complete"))
        elif job.is_file() and read_json(job).get("state") == "failed":
            states.append((cell.cell_id, "failed"))
        else:
            states.append((cell.cell_id, "missing"))
    return check_completeness(cells, states)


def extract_adaptive_outcomes(
    cells: Sequence[JobCell],
    outputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Extract every declared task cell without intersecting present outputs."""

    expected = {cell.cell_id for cell in cells}
    if set(outputs) != expected:
        raise AdaptiveDeploymentError(
            "adaptive outputs do not exactly cover declared cells"
        )
    rows: list[dict[str, Any]] = []
    for cell in cells:
        output = outputs[cell.cell_id]
        success = output.get("evaluation", {}).get("success")
        if (
            output.get("complete") is not True
            or output.get("cell_id") != cell.cell_id
            or output.get("observation_method") != cell.arm
            or output.get("operator") != cell.operator
            or output.get("deployment_mode") != ADAPTIVE_DEPLOYMENT_MODE
            or not isinstance(success, bool)
        ):
            raise AdaptiveDeploymentError("adaptive output identity/outcome is invalid")
        signals = output.get("signal_records")
        decisions = output.get("decision_records")
        interventions = output.get("intervention_records")
        if any(not isinstance(value, list) for value in (signals, decisions, interventions)):
            raise AdaptiveDeploymentError("adaptive output records are invalid")
        rows.append(
            {
                "cell_id": cell.cell_id,
                "model": cell.pair_key.model,
                "benchmark": cell.pair_key.domain,
                "task_id": cell.pair_key.task_id,
                "replicate_id": cell.pair_key.replicate_id,
                "method": cell.arm,
                "operator": cell.operator,
                "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
                "success": success,
                "observations": len(signals),
                "threshold_firings": sum(
                    bool(row.get("threshold_fired")) for row in decisions
                ),
                "selected_actions": sum(
                    bool(row.get("action_selected")) for row in decisions
                ),
                "applied_interventions": len(interventions),
                "accounting": output.get("accounting"),
            }
        )
    return tuple(rows)


def _split_csv(value: str) -> tuple[str, ...]:
    return _unique_names(
        tuple(item.strip() for item in value.split(",") if item.strip()),
        "comma-separated values",
    )


def _prepare_evolving(args: argparse.Namespace) -> int:
    adapter = EvolvingIntentAdapter(
        args.dataset, expected_sha256=args.dataset_sha256
    )
    result = prepare_adaptive_run(
        deployment_run_id=args.run_id,
        task_manifest_path=args.tasks,
        calibration_threshold_path=args.calibration_thresholds,
        calibration_extract_path=args.calibration_extract,
        calibration_manifest_path=args.calibration_manifest,
        source_registry_path=args.source_registry,
        baseline_profile_path=args.baseline_profile,
        planning_lock_path=args.planning_lock,
        realized_allocation_path=args.realized_allocation,
        tasks=adapter.load_tasks(),
        models=_split_csv(args.models),
        methods=_split_csv(args.methods),
        operators=_split_csv(args.operators),
        natural_max_actions_per_task=args.max_actions_per_task,
        randomization_seed=args.seed,
        replicates=args.replicates,
        artifacts_root=args.artifacts,
        evolving_dataset_path=args.dataset,
        evolving_build_receipt_path=args.build_receipt,
    )
    print(
        f"prepared online-adaptive run={result.run_id} "
        f"cells={result.declared_cells} manifest={result.manifest_sha256}"
    )
    print(f"threshold_lock={result.threshold_lock_path}")
    return 0


async def _run_evolving(args: argparse.Namespace) -> int:
    # Keep the explicit authorization check before even hashing the dataset.
    if not args.yes_spend:
        raise AdaptiveDeploymentError("provider dispatch requires --yes-spend")
    try:
        ExecutionShard(count=args.shard_count, index=args.shard_index)
    except ValueError as exc:
        raise AdaptiveDeploymentError(str(exc)) from exc
    adapter = EvolvingIntentAdapter(
        args.dataset, expected_sha256=args.dataset_sha256
    )
    summary = await execute_adaptive_run(
        run_id=args.run_id,
        task_manifest_path=args.tasks,
        threshold_lock_path=args.thresholds,
        tasks=adapter.load_tasks(),
        yes_spend=True,
        artifacts_root=args.artifacts,
        environ=_environment(args.env_file),
        max_new_cells=args.max_new_cells,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        evolving_dataset_path=args.dataset,
        evolving_build_receipt_path=args.build_receipt,
    )
    print(
        f"run={args.run_id} phase=adaptive_deployment "
        f"visited={summary.visited_cells} completed={summary.completed_cells} "
        f"failed={summary.failed_cells} skipped={summary.skipped_cells} "
        f"shard={summary.shard_index}/{summary.shard_count} "
        f"shard_cells={summary.shard_cells} declared={summary.declared_cells}"
    )
    return 0 if summary.failed_cells == 0 else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-evolving", help="freeze a provider-free online adaptive run"
    )
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--dataset-sha256", required=True)
    prepare.add_argument("--build-receipt", required=True)
    prepare.add_argument("--tasks", required=True)
    prepare.add_argument("--calibration-thresholds", required=True)
    prepare.add_argument("--calibration-extract", required=True)
    prepare.add_argument("--calibration-manifest", required=True)
    prepare.add_argument("--source-registry", required=True)
    prepare.add_argument("--realized-allocation", default=None)
    prepare.add_argument("--baseline-profile", required=True)
    prepare.add_argument("--planning-lock", required=True)
    prepare.add_argument("--models", required=True)
    prepare.add_argument(
        "--methods",
        default=(
            "active_recompute,frozen_probe:recompute,frozen_quiz,"
            "trace_judge,trace_rules,turn_clock,context_use"
        ),
    )
    prepare.add_argument(
        "--operators",
        default=(
            f"{Operator.NONE.value},{Operator.COMPACT.value},"
            f"{Operator.REGROUND.value}"
        ),
    )
    prepare.add_argument(
        "--max-actions-per-task",
        type=int,
        default=PRIMARY_MAX_ACTIONS_PER_TASK,
        help="primary policy cap; frozen to exactly one intervention per task",
    )
    prepare.add_argument("--replicates", type=int, default=PRIMARY_REPLICATES)
    prepare.add_argument("--seed", type=int, default=120120)
    prepare.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    prepare.set_defaults(func=_prepare_evolving)

    run = commands.add_parser(
        "run-evolving", help="execute the frozen current-prefix online run"
    )
    run.add_argument("--yes-spend", action="store_true")
    run.add_argument("--run-id", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--dataset-sha256", required=True)
    run.add_argument("--build-receipt", required=True)
    run.add_argument("--tasks", required=True)
    run.add_argument("--thresholds", required=True)
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


__all__ = [
    "ADAPTIVE_DEPLOYMENT_MODE",
    "ADAPTIVE_JOB_SUBDIR",
    "ADAPTIVE_POLICY",
    "PRIMARY_MAX_ACTIONS_PER_TASK",
    "PRIMARY_REPLICATES",
    "ADAPTIVE_RESULT_SUBDIR",
    "ADAPTIVE_RUNNER_VERSION",
    "ADAPTIVE_SCHEMA_VERSION",
    "AdaptiveDeploymentError",
    "AdaptivePreparationResult",
    "adaptive_completeness",
    "execute_adaptive_run",
    "extract_adaptive_outcomes",
    "main",
    "parser",
    "prepare_adaptive_run",
    "run_adaptive_task",
    "validate_adaptive_design",
]
