"""Strict, provider-free analysis of Experiment 12 two-pass deployments.

The declared pair table is always the denominator.  Before using any outcome,
this module verifies the immutable manifest and schedule chain, exact output,
job, and event coverage, every call attempt and its reconciled ledger row, and
the final complete-event hash.  Evolving Intent outcomes and the exact carried
history are recomputed from the frozen rendered dataset.  A benchmark without
an exact provider-free transcript replay is refused rather than partially
validated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from statistics import fmean
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from experiments12.adaptive_analysis12 import observation_class
from experiments12.analysis12 import (
    AnalysisInputError,
    _AttemptResource,
    _load_attempt_resources,
)
from experiments12.cli12 import DEFAULT_ARTIFACTS, REPOSITORY_ROOT
from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.schemas import CallAttemptRecord, CallStatus, record_to_dict
from experiments12.deployment12 import (
    DEPLOYMENT_RUNNER_VERSION,
    DEPLOYMENT_SCHEDULE_RECEIPT,
    PASS_ONE_RECEIPT,
    THRESHOLD_LOCK_RECEIPT,
    TWO_PASS_DEPLOYMENT_MODE,
    DeploymentArtifactError,
    DeploymentScheduleArtifact,
    DeploymentScheduleGroup,
    _OPERATOR_TYPES,
    build_deployment_schedule,
    _accounting,
    deployment_runtime_config,
    load_deployment_schedule,
    load_pass_one_observations,
    load_threshold_lock,
)
from experiments12.domains.base import DomainTask
from experiments12.domains.evolving_intent import EvolvingIntentAdapter
from experiments12.figures12 import DeploymentBar, write_deployment_grouped_bars
from experiments12.harness12 import ARM_TO_PROBE, grade_final_numeric
from experiments12.harness12 import HarnessConfig
from experiments12.manifest12 import RunLayout, validate_manifest_files
from experiments12.models12 import CATALOG
from experiments12.operators12 import (
    CompactionConfig,
    InterventionType,
    SignalReference,
    apply_intervention,
    freeze_initial_instructions,
    freeze_public_state,
    freeze_visible_prefix,
    make_feedback_note,
)
from experiments12.pairing12 import JobCell, TaskRef, make_pair_manifest
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.runner12 import load_task_manifest, pair_task_id
from experiments12.spec12 import Stage


TWO_PASS_ANALYSIS_VERSION = 1
TWO_PASS_ANALYSIS_TYPE = "two_pass_deployment_analysis"
TWO_PASS_VALIDATION_TYPE = "two_pass_deployment_validation"
DEPLOYMENT_OUTPUT_SUBDIR = "deployment"
DEPLOYMENT_JOB_SUBDIR = "deployment_jobs"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_CONTROL_OPERATOR = "none"
_CONFIDENCE = 0.95
_BOOTSTRAP_ITERATIONS = 2_000
_BOOTSTRAP_SEED = 12_012

_METRICS: tuple[tuple[str, str, str], ...] = (
    ("success", "proportion", "higher"),
    ("action_rate", "proportion_of_observed_checkpoints", "descriptive"),
    ("acted_on_task", "proportion", "descriptive"),
    ("scheduled_actions", "count", "descriptive"),
    ("applied_interventions", "count", "descriptive"),
    ("task_tokens", "tokens", "lower"),
    ("observer_tokens", "tokens", "lower"),
    ("total_tokens", "tokens", "lower"),
    ("latency_ms", "milliseconds", "lower"),
    ("actual_cost_usd", "USD", "lower"),
    ("reported_cost_usd", "USD", "lower"),
    ("estimated_cost_usd", "USD", "lower"),
    ("upper_bound_cost_usd", "USD", "lower"),
    ("failed_retry_attempts", "count", "lower"),
)

_FIGURE_METRICS: tuple[tuple[str, str], ...] = (
    ("success", "Task success rate"),
    ("action_rate", "Actions / observed checkpoints"),
    ("total_tokens", "Pass-two tokens per task"),
    ("latency_ms", "Pass-two latency per task (ms)"),
    ("actual_cost_usd", "Pass-two cost per task (USD)"),
)

_OUTPUT_FIELDS = {
    "schema_version",
    "deployment_runner_version",
    "run_id",
    "cell_id",
    "design_sha256",
    "model",
    "domain",
    "task_id",
    "condition",
    "task_sha256",
    "arm",
    "operator",
    "estimand",
    "schedule",
    "observation_method",
    "active_probe_variant",
    "messages",
    "task_assistant_messages",
    "task_records",
    "probe_records",
    "intervention_records",
    "evaluation",
    "transcript_sha256",
    "accounting",
    "complete",
}

_JOB_FIELDS = {
    "deployment_runner_version",
    "cell_id",
    "state",
    "output_sha256",
    "success",
    "accounting_sha256",
}


@dataclass(frozen=True, slots=True)
class TwoPassValidationReport:
    source_run_id: str
    source_manifest_sha256: str
    source_pair_manifest_sha256: str
    source_schedule_sha256: str
    expected_cells: int
    valid_outputs: int
    valid_jobs: int
    valid_event_logs: int
    call_attempts: int
    ledger_reservations: int
    canonical_regraded_cells: int
    cached_official_cells: int
    primary_ready: bool = True
    artifact_type: str = TWO_PASS_VALIDATION_TYPE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TwoPassMetricSummary:
    model: str
    benchmark: str
    observation_class: str
    method: str
    operator: str
    estimand: str
    metric: str
    unit: str
    favorable_direction: str
    n_tasks: int
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_iterations: int
    bootstrap_seed: int
    bootstrap_unit: str = "source_task"


@dataclass(frozen=True, slots=True)
class TwoPassOperatorEffect:
    model: str
    benchmark: str
    observation_class: str
    method: str
    operator: str
    control_operator: str
    estimand: str
    metric: str
    unit: str
    favorable_direction: str
    n_tasks: int
    control_mean: float
    operator_mean: float
    effect: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_iterations: int
    bootstrap_seed: int
    effect_definition: str = "operator_minus_none"
    bootstrap_unit: str = "paired_source_task"


@dataclass(frozen=True, slots=True)
class TwoPassMethodEffect:
    model: str
    benchmark: str
    reference_observation_class: str
    reference_method: str
    comparison_observation_class: str
    comparison_method: str
    operator: str
    estimand: str
    metric: str
    unit: str
    favorable_direction: str
    n_tasks: int
    reference_mean: float
    comparison_mean: float
    effect: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_iterations: int
    bootstrap_seed: int
    effect_definition: str = "comparison_method_minus_reference_method"
    bootstrap_unit: str = "paired_source_task"


@dataclass(frozen=True, slots=True)
class _ValidatedRun:
    manifest: Mapping[str, Any]
    cells: tuple[JobCell, ...]
    schedule: DeploymentScheduleArtifact
    outputs: Mapping[str, Mapping[str, Any]]
    rows: tuple[dict[str, Any], ...]
    report: TwoPassValidationReport


def _digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AnalysisInputError(f"{context} must be a lowercase SHA256 digest")
    return value


def _nonnegative_number(value: Any, *, context: str, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisInputError(f"{context} must be finite and non-negative")
    if integer and not isinstance(value, int):
        raise AnalysisInputError(f"{context} must be a non-negative integer")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AnalysisInputError(f"{context} must be finite and non-negative")
    return number


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisInputError("cannot take a bootstrap quantile of no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _bootstrap_interval(
    values: Sequence[float],
    *,
    identity: Sequence[str],
    iterations: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    if not values:
        raise AnalysisInputError("bootstrap values are empty")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise AnalysisInputError("bootstrap_iterations must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AnalysisInputError("bootstrap_seed must be an integer")
    if not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise AnalysisInputError("confidence must lie strictly between zero and one")
    prefix = "\0".join(("exp12/two-pass-task-bootstrap/v1", str(seed), *identity))
    population = len(values)
    sampled_means: list[float] = []
    for iteration in range(iterations):
        sample = []
        for draw in range(population):
            material = f"{prefix}\0{iteration}\0{draw}".encode("utf-8")
            import hashlib

            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % population
            sample.append(values[index])
        sampled_means.append(fmean(sample))
    sampled_means.sort()
    tail = (1 - float(confidence)) / 2
    return _quantile(sampled_means, tail), _quantile(sampled_means, 1 - tail)


def _manifest_receipts(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("benchmark_receipts")
    if not isinstance(raw, list) or not raw:
        raise AnalysisInputError("two-pass manifest has no artifact receipts")
    result: dict[str, Mapping[str, Any]] = {}
    for index, receipt in enumerate(raw):
        if not isinstance(receipt, Mapping):
            raise AnalysisInputError(f"manifest receipt {index} is not an object")
        name = receipt.get("name")
        path = receipt.get("path")
        digest = receipt.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(path, str)
            or not path
        ):
            raise AnalysisInputError(f"manifest receipt {index} has invalid identity")
        _digest(digest, context=f"manifest receipt {name} SHA256")
        if name in result:
            raise AnalysisInputError(f"manifest duplicates receipt {name!r}")
        result[name] = receipt
    return result


def _receipt_path(receipt: Mapping[str, Any], *, required_local: bool) -> Path | None:
    raw = str(receipt["path"])
    if raw.startswith("external:"):
        if required_local:
            raise AnalysisInputError(
                f"receipt {receipt['name']!r} is external and cannot be revalidated"
            )
        return None
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisInputError(f"receipt {receipt['name']!r} has an unsafe path")
    root = REPOSITORY_ROOT.resolve()
    candidate = root / relative
    current = candidate
    while current != root:
        if current.is_symlink():
            raise AnalysisInputError(f"receipt {receipt['name']!r} is linked")
        current = current.parent
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AnalysisInputError(f"receipt {receipt['name']!r} escapes the repository") from exc
    if path.is_symlink() or not path.is_file():
        raise AnalysisInputError(f"receipt {receipt['name']!r} is missing or linked")
    if sha256_file(path) != receipt["sha256"]:
        raise AnalysisInputError(f"receipt {receipt['name']!r} changed")
    return path


def _required_receipt(
    receipts: Mapping[str, Mapping[str, Any]], name: str, path: Path
) -> str:
    try:
        receipt = receipts[name]
    except KeyError as exc:
        raise AnalysisInputError(f"two-pass manifest lacks receipt {name!r}") from exc
    digest = sha256_file(path)
    if receipt.get("sha256") != digest:
        raise AnalysisInputError(f"two-pass {name} differs from its manifest receipt")
    return digest


def _exact_directory(root: Path, expected: set[str], *, context: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise AnalysisInputError(f"two-pass {context} directory is missing or linked")
    entries = tuple(root.iterdir())
    if (
        {path.name for path in entries} != expected
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise AnalysisInputError(f"two-pass {context} files do not exactly cover declared cells")


def _load_cells(
    layout: RunLayout, manifest: Mapping[str, Any]
) -> tuple[JobCell, ...]:
    try:
        cells = tuple(JobCell.from_dict(row) for row in read_jsonl(layout.pairs))
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisInputError(f"two-pass pair table is invalid: {exc}") from exc
    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise AnalysisInputError("two-pass pair table is empty or duplicates cells")
    models = manifest.get("models")
    methods = manifest.get("arms")
    operators = manifest.get("operators")
    for name, values in (("models", models), ("methods", methods), ("operators", operators)):
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise AnalysisInputError(f"two-pass manifest {name} are invalid")
    expected_treatments = {(method, operator) for method in methods for operator in operators}
    blocks: dict[str, list[JobCell]] = {}
    for cell in cells:
        if (
            cell.pair_key.model not in models
            or cell.arm not in methods
            or cell.operator not in operators
        ):
            raise AnalysisInputError("two-pass cell falls outside its manifest")
        blocks.setdefault(cell.block_id, []).append(cell)
    for block_id, block in blocks.items():
        treatments = [(cell.arm, cell.operator) for cell in block]
        identities = {
            (
                cell.pair_key.model,
                cell.pair_key.domain,
                cell.pair_key.task_id,
                cell.pair_key.replicate_id,
                cell.pair_key.task_sha256,
            )
            for cell in block
        }
        if (
            set(treatments) != expected_treatments
            or len(treatments) != len(expected_treatments)
            or len(identities) != 1
            or sorted(cell.block_position for cell in block) != list(range(len(block)))
        ):
            raise AnalysisInputError(f"two-pass block is incomplete or mixed: {block_id}")
    extra = manifest.get("extra_config")
    if (
        not isinstance(extra, Mapping)
        or extra.get("deployment_mode") != TWO_PASS_DEPLOYMENT_MODE
        or extra.get("n_cells") != len(cells)
        or extra.get("replicates") != 1
        or extra.get("deployment_estimand") not in {
            "natural_threshold",
            "matched_rate_top_k",
            "yoked_anchor",
        }
    ):
        raise AnalysisInputError("manifest is not an exact two-pass deployment design")
    if any(cell.pair_key.replicate_id != 0 for cell in cells):
        raise AnalysisInputError("primary two-pass analysis requires replicate zero only")
    return cells


def _runtime_configs(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], HarnessConfig, CompactionConfig]:
    extra = manifest.get("extra_config")
    raw = extra.get("deployment_runtime") if isinstance(extra, Mapping) else None
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "harness",
        "harness_config_sha256",
        "compaction",
        "compaction_config_sha256",
    }:
        raise AnalysisInputError("manifest deployment runtime lock has an invalid schema")
    harness = raw.get("harness")
    compaction = raw.get("compaction")
    if not isinstance(harness, Mapping) or set(harness) != {
        "checkpoint_every",
        "task_max_output_tokens",
        "probe_max_output_tokens",
        "temperature",
    }:
        raise AnalysisInputError("manifest harness runtime lock has an invalid schema")
    if not isinstance(compaction, Mapping) or set(compaction) != {
        "keep_last_messages",
        "max_excerpt_bytes",
        "max_summary_bytes",
    }:
        raise AnalysisInputError("manifest compaction runtime lock has an invalid schema")
    try:
        harness_config = HarnessConfig(**dict(harness))
        compaction_config = CompactionConfig(**dict(compaction))
        canonical = deployment_runtime_config(harness_config, compaction_config)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"manifest deployment runtime lock is invalid: {exc}") from exc
    if dict(raw) != canonical:
        raise AnalysisInputError("manifest deployment runtime values/hashes do not reproduce")
    return raw, harness_config, compaction_config


def _validate_task_manifest(
    receipts: Mapping[str, Mapping[str, Any]], cells: Sequence[JobCell]
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    try:
        receipt = receipts["task_manifest"]
    except KeyError as exc:
        raise AnalysisInputError("two-pass manifest lacks the task manifest receipt") from exc
    path = _receipt_path(receipt, required_local=True)
    assert path is not None
    try:
        rows = load_task_manifest(path)
    except ValueError as exc:
        raise AnalysisInputError(f"two-pass task manifest is invalid: {exc}") from exc
    declared = {
        (cell.pair_key.domain, cell.pair_key.task_id, str(cell.pair_key.task_sha256))
        for cell in cells
    }
    recorded = {
        (str(row["benchmark"]), str(row["task_id"]), str(row["task_sha256"]))
        for row in rows
    }
    if declared != recorded:
        raise AnalysisInputError("task manifest does not exactly cover the two-pass pair table")
    return path, rows


def _load_regrade_tasks(
    receipts: Mapping[str, Mapping[str, Any]], cells: Sequence[JobCell]
) -> dict[tuple[str, str, str], DomainTask]:
    domains = {cell.pair_key.domain for cell in cells}
    if domains != {"evolving_intent_gsm8k"}:
        return {}
    try:
        receipt = receipts["evolving_rendered_dataset"]
    except KeyError as exc:
        raise AnalysisInputError(
            "Evolving two-pass analysis lacks its rendered dataset receipt"
        ) from exc
    dataset = _receipt_path(receipt, required_local=True)
    assert dataset is not None
    try:
        tasks = EvolvingIntentAdapter(
            dataset, expected_sha256=str(receipt["sha256"])
        ).load_tasks()
    except ValueError as exc:
        raise AnalysisInputError(f"frozen Evolving dataset is invalid: {exc}") from exc
    index = {
        (task.domain, pair_task_id(task), task.task_sha256): task for task in tasks
    }
    declared = {
        (cell.pair_key.domain, cell.pair_key.task_id, str(cell.pair_key.task_sha256))
        for cell in cells
    }
    if not declared.issubset(index):
        raise AnalysisInputError("frozen Evolving dataset does not cover every declared task")
    return {key: index[key] for key in declared}


def _rebuild_schedule(
    schedule: DeploymentScheduleArtifact,
    cells: Sequence[JobCell],
    *,
    pair_sha256: str,
    pass_one_path: Path,
    threshold_path: Path,
) -> None:
    pass_one = load_pass_one_observations(pass_one_path)
    threshold = load_threshold_lock(threshold_path)
    feedback = {
        (row.member_id, row.checkpoint): row.evidence
        for group in schedule.groups
        for row in group.feedback
    }
    rebuilt = build_deployment_schedule(
        estimand=schedule.estimand,
        cells=cells,
        pair_manifest_sha256=pair_sha256,
        pass_one=pass_one,
        pass_one_artifact_sha256=sha256_file(pass_one_path),
        threshold_lock=threshold,
        threshold_lock_sha256=sha256_file(threshold_path),
        feedback_plans=feedback,
    )
    if rebuilt != schedule:
        raise AnalysisInputError(
            "two-pass schedule does not reproduce from pair/pass-one/threshold locks"
        )


def _group_index(
    schedule: DeploymentScheduleArtifact, cells: Sequence[JobCell]
) -> dict[tuple[str, str], DeploymentScheduleGroup]:
    groups = {(group.block_id, group.observation_method): group for group in schedule.groups}
    expected = {(cell.block_id, cell.arm) for cell in cells}
    if set(groups) != expected:
        raise AnalysisInputError("two-pass schedule groups do not exactly cover cells")
    for key, group in groups.items():
        members = {
            cell.cell_id
            for cell in cells
            if (cell.block_id, cell.arm) == key
        }
        recorded_members = {member.member_id for member in group.schedule.members}
        if members != recorded_members:
            raise AnalysisInputError("two-pass schedule member coverage changed")
        method_cells = [cell for cell in cells if (cell.block_id, cell.arm) == key]
        first = method_cells[0]
        if (
            group.model != first.pair_key.model
            or group.benchmark != first.pair_key.domain
            or group.task_id != first.pair_key.task_id
            or group.task_sha256 != first.pair_key.task_sha256
            or group.replicate_id != first.pair_key.replicate_id
            or group.active_variant != ARM_TO_PROBE.get(first.arm)
        ):
            raise AnalysisInputError("two-pass schedule group identity changed")
    return groups


def _attempt_totals(
    call: Any,
    *,
    expected_purpose: str,
    expected_request_key: str,
    expected_model: str,
    attempts: Mapping[str, _AttemptResource],
    seen: set[str],
    context: str,
) -> dict[str, Any]:
    expected_call_fields = {
        "call_event_ids",
        "resolved_model_id",
        "response_id",
        "request_id",
        "finish_reason",
        "usage",
        "accounted_cost_usd",
        "elapsed_ms",
    }
    if not isinstance(call, Mapping) or set(call) != expected_call_fields:
        raise AnalysisInputError(f"{context} has invalid call accounting")
    event_ids = call.get("call_event_ids")
    if (
        not isinstance(event_ids, list)
        or not event_ids
        or any(not isinstance(item, str) or not item for item in event_ids)
        or len(event_ids) != len(set(event_ids))
    ):
        raise AnalysisInputError(f"{context} has invalid call event IDs")
    if seen.intersection(event_ids):
        raise AnalysisInputError(f"{context} reuses a call event ID")
    try:
        selected = [attempts[event_id] for event_id in event_ids]
    except KeyError as exc:
        raise AnalysisInputError(f"{context} references an absent call attempt") from exc
    if any(item.purpose != expected_purpose for item in selected):
        raise AnalysisInputError(f"{context} has the wrong call purpose")
    if (
        selected[-1].status is not CallStatus.SUCCEEDED
        or any(item.status is not CallStatus.FAILED for item in selected[:-1])
    ):
        raise AnalysisInputError(f"{context} lacks failed-retry*/successful-final ordering")
    spec = CATALOG.models.get(expected_model)
    if spec is None:
        raise AnalysisInputError(f"{context} uses an unknown target model")
    for attempt_number, attempt in enumerate(selected, 1):
        if (
            attempt.attempt_number != attempt_number
            or attempt.request_key != f"{expected_request_key}/attempt-{attempt_number}"
            or attempt.provider != spec.provider
            or attempt.model != spec.model
        ):
            raise AnalysisInputError(f"{context} attempt identity changed")
    usage = call.get("usage")
    final = selected[-1]
    if (
        not isinstance(usage, Mapping)
        or set(usage)
        != {"input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"}
        or (
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("cached_input_tokens"),
            usage.get("reasoning_tokens"),
        )
        != (
            final.input_tokens,
            final.output_tokens,
            final.cached_input_tokens,
            final.reasoning_tokens,
        )
    ):
        raise AnalysisInputError(f"{context} usage disagrees with its successful attempt")
    if call.get("elapsed_ms") != sum(item.elapsed_ms for item in selected):
        raise AnalysisInputError(f"{context} latency disagrees with its attempts")
    try:
        recorded_cost = Decimal(str(call.get("accounted_cost_usd")))
    except Exception as exc:
        raise AnalysisInputError(f"{context} has invalid recorded cost") from exc
    if not recorded_cost.is_finite() or recorded_cost < 0 or recorded_cost != final.actual_cost_usd:
        raise AnalysisInputError(f"{context} cost disagrees with its successful attempt")
    seen.update(event_ids)
    quality_costs = {
        quality: sum(
            (item.actual_cost_usd for item in selected if item.cost_quality == quality),
            Decimal("0"),
        )
        for quality in ("reported", "estimated", "upper_bound")
    }
    return {
        "tokens": sum(item.input_tokens + item.output_tokens for item in selected),
        "latency_ms": sum(item.elapsed_ms for item in selected),
        "actual_cost_usd": sum(
            (item.actual_cost_usd for item in selected), Decimal("0")
        ),
        "failed_retries": len(selected) - 1,
        "quality_costs": quality_costs,
    }


def _expected_middle_events(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks = output["task_records"]
    probes = output["probe_records"]
    interventions = output["intervention_records"]
    result: list[Mapping[str, Any]] = []
    for turn, task in enumerate(tasks, 1):
        if not isinstance(task, Mapping) or task.get("event") != "task_turn" or task.get("task_turn") != turn:
            raise AnalysisInputError("two-pass task records are not contiguous")
        result.append(task)
        turn_probes = [row for row in probes if isinstance(row, Mapping) and row.get("after_task_turn") == turn]
        turn_interventions = [
            row for row in interventions if isinstance(row, Mapping) and row.get("checkpoint") == turn
        ]
        if len(turn_probes) > 1 or len(turn_interventions) > 1:
            raise AnalysisInputError("two-pass output duplicates checkpoint events")
        result.extend(turn_probes)
        result.extend(turn_interventions)
    if len(result) != len(tasks) + len(probes) + len(interventions):
        raise AnalysisInputError("two-pass output contains off-timeline checkpoint events")
    return result


def _validate_interventions(
    records: Any,
    *,
    cell: JobCell,
    group: DeploymentScheduleGroup,
) -> None:
    if not isinstance(records, list) or len(records) != len(group.actions):
        raise AnalysisInputError(f"two-pass interventions do not match schedule: {cell.cell_id}")
    for record, trigger in zip(records, group.actions, strict=True):
        if not isinstance(record, Mapping):
            raise AnalysisInputError(f"two-pass intervention is invalid: {cell.cell_id}")
        if (
            record.get("event") != "intervention_applied"
            or record.get("intervention_type") != _OPERATOR_TYPES[cell.operator].value
            or record.get("declared_operator") != cell.operator
            or record.get("member_id") != cell.cell_id
            or record.get("checkpoint") != trigger.checkpoint
            or record.get("schedule_group_id") != group.group_id
            or record.get("schedule_sha256") != group.schedule.schedule_sha256
            or record.get("observation_method") != group.observation_method
            or record.get("selection_policy") != trigger.selection_policy.value
            or record.get("trigger_score") != trigger.score
            or record.get("locked_threshold") != trigger.locked_threshold
            or record.get("natural_threshold_fired") != trigger.natural_threshold_fired
            or record.get("threshold_record_sha256") != trigger.threshold_record_sha256
            or record.get("signal_method") != trigger.trigger_method
            or record.get("signal_source_prefix_sha256") != trigger.source_prefix_sha256
            or record.get("signal_record_sha256") != trigger.signal_record_sha256
            or record.get("signal_frozen_two_pass") is not True
        ):
            raise AnalysisInputError(f"two-pass intervention binding changed: {cell.cell_id}")
        provenance = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "provenance_sha256",
                "declared_operator",
                "observation_method",
                "selection_policy",
                "trigger_score",
                "locked_threshold",
                "natural_threshold_fired",
                "threshold_record_sha256",
            }
        }
        if sha256_json(provenance) != record.get("provenance_sha256"):
            raise AnalysisInputError(f"two-pass intervention provenance changed: {cell.cell_id}")


def _public_state(task: DomainTask, checkpoint: int) -> dict[str, Any]:
    return {
        "completed_task_turns": [
            {"turn": turn.index, "user_message": turn.user_message}
            for turn in task.turns[:checkpoint]
        ],
        "public_metadata": dict(task.public_metadata),
    }


def _replay_visible_history(
    *,
    cell: JobCell,
    group: DeploymentScheduleGroup,
    task: DomainTask,
    output: Mapping[str, Any],
    compaction_config: CompactionConfig,
) -> None:
    """Recreate the exact carried target history from frozen public inputs."""

    task_records = output["task_records"]
    probe_records = output["probe_records"]
    intervention_records = output["intervention_records"]
    active_variant = group.active_variant
    probe_ordinals = {
        checkpoint: index
        for index, checkpoint in enumerate(group.observation_checkpoints, 1)
    }
    setup = (
        None
        if active_variant is None
        else render_initial_instruction(
            active_variant,
            task.instance_id,
            tuple(range(1, len(group.observation_checkpoints) + 1)),
        )
    )
    first_content = task.turns[0].user_message
    if setup:
        first_content = setup + "\n\n--- BENCHMARK MESSAGE ---\n" + first_content
    initial = freeze_initial_instructions(
        domain=task.domain,
        task_id=task.task_id,
        messages=({"role": "user", "content": first_content},),
    )
    probe_by_turn = {
        record.get("after_task_turn"): record for record in probe_records
    }
    intervention_by_turn = {
        record.get("checkpoint"): record for record in intervention_records
    }
    if len(probe_by_turn) != len(probe_records) or len(intervention_by_turn) != len(
        intervention_records
    ):
        raise AnalysisInputError(f"two-pass replay events duplicate turns: {cell.cell_id}")
    messages: list[dict[str, Any]] = []
    assistant_messages: list[str] = []
    for turn_number, task_turn in enumerate(task.turns, 1):
        record = task_records[turn_number - 1]
        if not isinstance(record, Mapping) or set(record) != {
            "event",
            "task_turn",
            "user_message",
            "assistant_message",
            "call",
            "continued_history_sha256",
        }:
            raise AnalysisInputError(f"two-pass task event schema changed: {cell.cell_id}")
        user = {
            "role": "user",
            "content": first_content if turn_number == 1 else task_turn.user_message,
        }
        assistant = record.get("assistant_message")
        if (
            record.get("event") != "task_turn"
            or record.get("task_turn") != turn_number
            or record.get("user_message") != user
            or not isinstance(assistant, Mapping)
            or set(assistant) != {"role", "content"}
            or assistant.get("role") != "assistant"
            or not isinstance(assistant.get("content"), str)
        ):
            raise AnalysisInputError(f"two-pass task event cannot be replayed: {cell.cell_id}")
        messages.extend((user, dict(assistant)))
        assistant_messages.append(str(assistant["content"]))
        if record.get("continued_history_sha256") != sha256_json(messages):
            raise AnalysisInputError(
                f"two-pass task continued-history hash changed: {cell.cell_id}"
            )

        probe_index = probe_ordinals.get(turn_number)
        if probe_index is not None and active_variant is not None:
            probe = probe_by_turn.get(turn_number)
            if not isinstance(probe, Mapping):
                raise AnalysisInputError(f"two-pass carried probe is missing: {cell.cell_id}")
            assistant_probe = probe.get("assistant_message")
            response = (
                assistant_probe.get("content")
                if isinstance(assistant_probe, Mapping)
                else None
            )
            if (
                not isinstance(response, str)
                or not isinstance(assistant_probe, Mapping)
                or set(assistant_probe) != {"role", "content"}
                or assistant_probe.get("role") != "assistant"
            ):
                raise AnalysisInputError(f"two-pass probe response is invalid: {cell.cell_id}")
            instance = generate_probe_instance(
                active_variant, task.instance_id, probe_index
            )
            grade = grade_probe_response(instance, response)
            expected_probe = {
                "event": "active_probe",
                "after_task_turn": turn_number,
                "checkpoint_index": probe_index,
                "variant": active_variant,
                "user_message": {
                    "role": "user",
                    "content": render_probe_prompt(instance),
                },
                "assistant_message": dict(assistant_probe),
                "grade": {
                    "passed": grade.passed,
                    "value_correct": grade.value_correct,
                    "exact_format": grade.exact_format,
                    "error": grade.error,
                    "expected_sha256": sha256_json(instance.expected_answer),
                },
                "call": probe.get("call"),
                "changes_frozen_timing": False,
            }
            if dict(probe) != expected_probe:
                raise AnalysisInputError(
                    f"two-pass carried probe prompt/grade changed: {cell.cell_id}"
                )
            messages.extend(
                (dict(expected_probe["user_message"]), dict(assistant_probe))
            )
        elif turn_number in probe_by_turn:
            raise AnalysisInputError(f"two-pass has an undeclared carried probe: {cell.cell_id}")

        if turn_number not in intervention_by_turn:
            continue
        try:
            trigger = group.action_for(turn_number)
            prefix = freeze_visible_prefix(
                domain=task.domain,
                task_id=task.task_id,
                after_turn=turn_number,
                messages=messages,
            )
            signal = SignalReference(
                method=trigger.trigger_method,
                checkpoint=turn_number,
                source_prefix_sha256=trigger.source_prefix_sha256,
                signal_record_sha256=trigger.signal_record_sha256,
                schedule_sha256=group.schedule.schedule_sha256,
                frozen_two_pass=True,
            )
            kind = _OPERATOR_TYPES[cell.operator]
            feedback = None
            public_state = None
            compact = None
            if kind is InterventionType.FEEDBACK:
                evidence = group.feedback_for(cell.cell_id, turn_number)
                if evidence is None:
                    raise AnalysisInputError("frozen feedback evidence is missing")
                feedback = make_feedback_note(
                    prefix,
                    good=evidence.good,
                    bad=evidence.bad,
                    watch=evidence.watch,
                )
            elif kind is InterventionType.REGROUND:
                public_state = freeze_public_state(
                    domain=task.domain,
                    task_id=task.task_id,
                    after_turn=turn_number,
                    state=_public_state(task, turn_number),
                )
            elif kind is InterventionType.COMPACT:
                compact = compaction_config
            application = apply_intervention(
                intervention_type=kind,
                prefix=prefix,
                schedule=group.schedule,
                member_id=cell.cell_id,
                checkpoint=turn_number,
                instructions=(
                    initial
                    if kind in {InterventionType.COMPACT, InterventionType.REGROUND}
                    else None
                ),
                public_state=public_state,
                feedback=feedback,
                compaction_config=compact,
                signal=signal,
            )
        except (DeploymentArtifactError, ValueError) as exc:
            raise AnalysisInputError(
                f"two-pass operator history cannot be replayed: {cell.cell_id}: {exc}"
            ) from exc
        expected_intervention = {
            **application.as_event(),
            "declared_operator": cell.operator,
            "observation_method": group.observation_method,
            "selection_policy": trigger.selection_policy.value,
            "trigger_score": trigger.score,
            "locked_threshold": trigger.locked_threshold,
            "natural_threshold_fired": trigger.natural_threshold_fired,
            "threshold_record_sha256": trigger.threshold_record_sha256,
        }
        if dict(intervention_by_turn[turn_number]) != expected_intervention:
            raise AnalysisInputError(
                f"two-pass operator continued history changed: {cell.cell_id}"
            )
        messages = application.continued_history
    if output["task_assistant_messages"] != assistant_messages:
        raise AnalysisInputError(f"two-pass assistant-message replay changed: {cell.cell_id}")
    if output["messages"] != messages or output["transcript_sha256"] != sha256_json(messages):
        raise AnalysisInputError(f"two-pass final carried history changed: {cell.cell_id}")


def _validate_output(
    *,
    cell: JobCell,
    group: DeploymentScheduleGroup,
    output: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    output_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    pair_sha256: str,
    schedule_sha256: str,
    pass_one_sha256: str,
    threshold_sha256: str,
    runtime_config: Mapping[str, Any],
    compaction_config: CompactionConfig,
    regrade_task: DomainTask | None,
    attempts: Mapping[str, _AttemptResource],
    seen_attempts: set[str],
) -> dict[str, Any]:
    if (
        set(output) != _OUTPUT_FIELDS
        or output.get("schema_version") != 2
        or output.get("deployment_runner_version") != DEPLOYMENT_RUNNER_VERSION
        or output.get("complete") is not True
    ):
        raise AnalysisInputError(f"two-pass output schema is invalid: {cell.cell_id}")
    condition = cell.pair_key.task_id.rsplit("::", 1)[-1]
    if (
        output.get("run_id") != manifest.get("run_id")
        or output.get("cell_id") != cell.cell_id
        or output.get("model") != cell.pair_key.model
        or output.get("domain") != cell.pair_key.domain
        or output.get("task_id") != cell.pair_key.task_id.rsplit("::", 1)[0]
        or output.get("condition") != condition
        or output.get("task_sha256") != cell.pair_key.task_sha256
        or output.get("arm") != cell.arm
        or output.get("operator") != cell.operator
        or output.get("observation_method") != cell.arm
        or output.get("active_probe_variant") != group.active_variant
    ):
        raise AnalysisInputError(f"two-pass output identity changed: {cell.cell_id}")
    # The conditional expression above is deliberately followed by an exact
    # unconditional check so zero-action and action-bearing groups share one
    # unambiguous estimand contract.
    if output.get("estimand") != manifest["extra_config"]["deployment_estimand"]:
        raise AnalysisInputError(f"two-pass output estimand changed: {cell.cell_id}")
    schedule = output.get("schedule")
    if not isinstance(schedule, Mapping) or set(schedule) != {
        "artifact_sha256",
        "group_id",
        "schedule_sha256",
        "pass_one_artifact_sha256",
        "threshold_lock_sha256",
        "observation_checkpoints",
        "action_checkpoints",
    }:
        raise AnalysisInputError(f"two-pass output schedule is invalid: {cell.cell_id}")
    if dict(schedule) != {
        "artifact_sha256": schedule_sha256,
        "group_id": group.group_id,
        "schedule_sha256": group.schedule.schedule_sha256,
        "pass_one_artifact_sha256": pass_one_sha256,
        "threshold_lock_sha256": threshold_sha256,
        "observation_checkpoints": list(group.observation_checkpoints),
        "action_checkpoints": [row.checkpoint for row in group.actions],
    }:
        raise AnalysisInputError(f"two-pass output schedule binding changed: {cell.cell_id}")
    if (
        not isinstance(events, Sequence)
        or len(events) < 2
        or not isinstance(events[0], Mapping)
        or not isinstance(events[-1], Mapping)
    ):
        raise AnalysisInputError(f"two-pass event log is invalid: {cell.cell_id}")
    start = events[0]
    design = {key: value for key, value in start.items() if key not in {"event", "design_sha256"}}
    if (
        set(start)
        != {
            "event",
            "design_sha256",
            "deployment_runner_version",
            "run_id",
            "cell",
            "task",
            "estimand",
            "schedule_artifact_sha256",
            "pass_one_artifact_sha256",
            "threshold_lock_sha256",
            "group",
            "runtime_config",
        }
        or start.get("event") != "start"
        or start.get("deployment_runner_version") != DEPLOYMENT_RUNNER_VERSION
        or sha256_json(design) != start.get("design_sha256")
        or output.get("design_sha256") != start.get("design_sha256")
        or start.get("run_id") != manifest.get("run_id")
        or start.get("cell") != cell.as_dict()
        or start.get("group") != group.as_dict()
        or start.get("estimand") != output.get("estimand")
        or start.get("schedule_artifact_sha256") != schedule_sha256
        or start.get("pass_one_artifact_sha256") != pass_one_sha256
        or start.get("threshold_lock_sha256") != threshold_sha256
        or start.get("runtime_config") != runtime_config
    ):
        raise AnalysisInputError(f"two-pass start/design receipt changed: {cell.cell_id}")
    task_design = start.get("task")
    if not isinstance(task_design, Mapping):
        raise AnalysisInputError(f"two-pass start task is invalid: {cell.cell_id}")
    if regrade_task is not None and dict(task_design) != regrade_task.manifest_record():
        raise AnalysisInputError(f"two-pass start task differs from frozen dataset: {cell.cell_id}")
    if (
        task_design.get("domain") != cell.pair_key.domain
        or task_design.get("task_id") != output.get("task_id")
        or task_design.get("condition") != output.get("condition")
        or task_design.get("task_sha256") != cell.pair_key.task_sha256
    ):
        raise AnalysisInputError(f"two-pass task design identity changed: {cell.cell_id}")
    middle = _expected_middle_events(output)
    if list(events[1:-1]) != middle:
        raise AnalysisInputError(f"two-pass event/output records differ: {cell.cell_id}")
    complete = events[-1]
    expected_complete = {
        "event": "complete",
        "design_sha256": output["design_sha256"],
        "task_turns": len(output["task_records"]),
        "active_probe_calls": len(output["probe_records"]),
        "interventions": len(output["intervention_records"]),
        "transcript_sha256": output["transcript_sha256"],
        "output_sha256": output_sha256,
        "prediction": output["evaluation"]["prediction"],
        "success": output["evaluation"]["success"],
    }
    if dict(complete) != expected_complete:
        raise AnalysisInputError(f"two-pass complete receipt changed: {cell.cell_id}")
    if (
        not isinstance(output["messages"], list)
        or sha256_json(output["messages"]) != output["transcript_sha256"]
        or not isinstance(output["task_records"], list)
        or not isinstance(output["probe_records"], list)
        or not isinstance(output["intervention_records"], list)
    ):
        raise AnalysisInputError(f"two-pass transcript is invalid: {cell.cell_id}")
    if regrade_task is None:
        raise AnalysisInputError(
            "two-pass benchmark has no exact provider-free transcript replay"
        )
    _replay_visible_history(
        cell=cell,
        group=group,
        task=regrade_task,
        output=output,
        compaction_config=compaction_config,
    )
    expected_probes = len(group.observation_checkpoints) if group.active_variant else 0
    if len(output["probe_records"]) != expected_probes:
        raise AnalysisInputError(f"two-pass active probe coverage changed: {cell.cell_id}")
    if len(output["task_records"]) != task_design.get("num_turns"):
        raise AnalysisInputError(f"two-pass task-turn coverage changed: {cell.cell_id}")
    assistant_messages: list[str] = []
    task_tokens = observer_tokens = latency_ms = failed_retries = 0
    actual_cost = Decimal("0")
    quality_costs = {
        "reported": Decimal("0"),
        "estimated": Decimal("0"),
        "upper_bound": Decimal("0"),
    }
    for turn, record in enumerate(output["task_records"], 1):
        assistant = record.get("assistant_message") if isinstance(record, Mapping) else None
        text = assistant.get("content") if isinstance(assistant, Mapping) else None
        if not isinstance(text, str):
            raise AnalysisInputError(f"two-pass assistant message is invalid: {cell.cell_id}")
        assistant_messages.append(text)
        resource = _attempt_totals(
            record.get("call"),
            expected_purpose="deployment_agent_turn",
            expected_request_key=f"{manifest['run_id']}/{cell.cell_id}/deployment-task-{turn}",
            expected_model=cell.pair_key.model,
            attempts=attempts,
            seen=seen_attempts,
            context=f"{cell.cell_id} task {turn}",
        )
        task_tokens += resource["tokens"]
        latency_ms += resource["latency_ms"]
        actual_cost += resource["actual_cost_usd"]
        for quality in quality_costs:
            quality_costs[quality] += resource["quality_costs"][quality]
        failed_retries += resource["failed_retries"]
    if output["task_assistant_messages"] != assistant_messages:
        raise AnalysisInputError(f"two-pass assistant-message rollup changed: {cell.cell_id}")
    for index, record in enumerate(output["probe_records"], 1):
        if (
            not isinstance(record, Mapping)
            or record.get("event") != "active_probe"
            or record.get("checkpoint_index") != index
            or record.get("after_task_turn") != group.observation_checkpoints[index - 1]
            or record.get("variant") != group.active_variant
        ):
            raise AnalysisInputError(f"two-pass active probe identity changed: {cell.cell_id}")
        resource = _attempt_totals(
            record.get("call"),
            expected_purpose="deployment_active_probe",
            expected_request_key=f"{manifest['run_id']}/{cell.cell_id}/deployment-probe-{index}",
            expected_model=cell.pair_key.model,
            attempts=attempts,
            seen=seen_attempts,
            context=f"{cell.cell_id} probe {index}",
        )
        observer_tokens += resource["tokens"]
        latency_ms += resource["latency_ms"]
        actual_cost += resource["actual_cost_usd"]
        for quality in quality_costs:
            quality_costs[quality] += resource["quality_costs"][quality]
        failed_retries += resource["failed_retries"]
    evaluation = output.get("evaluation")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "prediction",
        "evaluation_label_sha256",
        "success",
    }:
        raise AnalysisInputError(f"two-pass evaluation is invalid: {cell.cell_id}")
    if regrade_task is not None:
        prediction, success = grade_final_numeric(
            assistant_messages[-1], regrade_task.evaluation_label
        )
        expected_label_sha = (
            None
            if regrade_task.evaluation_label is None
            else sha256_json(regrade_task.evaluation_label)
        )
        if (
            evaluation.get("prediction") != prediction
            or evaluation.get("success") != success
            or evaluation.get("evaluation_label_sha256") != expected_label_sha
            or not isinstance(success, bool)
        ):
            raise AnalysisInputError(f"canonical final-outcome regrade changed: {cell.cell_id}")
        outcome_source = "canonical_regrade"
    else:
        success = evaluation.get("success")
        if not isinstance(success, bool):
            raise AnalysisInputError(f"two-pass output lacks binary official success: {cell.cell_id}")
        outcome_source = "cached_official_not_provider_free_regradable"
    accounting = output.get("accounting")
    if (
        not isinstance(accounting, Mapping)
        or accounting
        != _accounting([*output["task_records"], *output["probe_records"]])
    ):
        raise AnalysisInputError(f"two-pass accounting is invalid: {cell.cell_id}")
    observations = len(group.observation_checkpoints)
    scheduled = len(group.actions)
    unit_id = f"{cell.pair_key.task_id}/r{cell.pair_key.replicate_id}"
    return {
        "cell_id": cell.cell_id,
        "model": cell.pair_key.model,
        "benchmark": cell.pair_key.domain,
        "task_id": cell.pair_key.task_id,
        "replicate_id": cell.pair_key.replicate_id,
        "unit_id": unit_id,
        "observation_class": observation_class(cell.arm),
        "method": cell.arm,
        "operator": cell.operator,
        "deployment_mode": TWO_PASS_DEPLOYMENT_MODE,
        "estimand": output["estimand"],
        "success": success,
        "outcome_source": outcome_source,
        "observations": observations,
        "scheduled_actions": scheduled,
        "action_rate": scheduled / observations,
        "acted_on_task": int(scheduled > 0),
        "applied_interventions": len(output["intervention_records"]),
        "task_tokens": task_tokens,
        "observer_tokens": observer_tokens,
        "total_tokens": task_tokens + observer_tokens,
        "latency_ms": latency_ms,
        "actual_cost_usd": float(actual_cost),
        "reported_cost_usd": float(quality_costs["reported"]),
        "estimated_cost_usd": float(quality_costs["estimated"]),
        "upper_bound_cost_usd": float(quality_costs["upper_bound"]),
        "failed_retry_attempts": failed_retries,
    }


def _run_ledger_reservations(layout: RunLayout, run_id: str) -> set[str]:
    try:
        uri = layout.ledger.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                "SELECT reservation_id, request_key FROM reservations"
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisInputError(
            f"could not audit run-scoped ledger reservations: {type(exc).__name__}"
        ) from exc
    result = {
        str(reservation_id)
        for reservation_id, request_key in rows
        if isinstance(request_key, str) and request_key.startswith(run_id + "/")
    }
    if not result:
        raise AnalysisInputError("two-pass run has no run-scoped ledger reservations")
    return result


def _strict_call_attempt_ids(path: Path) -> set[str]:
    """Parse every append-only attempt row; no ignored metadata rows are allowed."""

    rows = read_jsonl(path)
    if not rows:
        raise AnalysisInputError("two-pass call-attempt log is empty")
    event_ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise AnalysisInputError(f"call-attempt row {index} is not an object")
        try:
            attempt = CallAttemptRecord.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisInputError(f"call-attempt row {index} is invalid: {exc}") from exc
        if record_to_dict(attempt) != dict(raw):
            raise AnalysisInputError(
                f"call-attempt row {index} has missing, extra, or noncanonical fields"
            )
        if attempt.event_id in event_ids:
            raise AnalysisInputError(f"call-attempt row {index} duplicates an event ID")
        event_ids.add(attempt.event_id)
    return event_ids


def _validated_run(
    layout: RunLayout, *, expected_manifest_sha256: str
) -> _ValidatedRun:
    expected = _digest(expected_manifest_sha256, context="expected manifest SHA256")
    if layout.manifest.is_symlink() or layout.pairs.is_symlink():
        raise AnalysisInputError("two-pass manifest and pair table must not be symlinks")
    manifest_sha = sha256_file(layout.manifest)
    if manifest_sha != expected:
        raise AnalysisInputError("two-pass manifest differs from its external SHA256")
    manifest = read_json(layout.manifest)
    if not isinstance(manifest, Mapping) or manifest.get("run_id") != layout.root.name:
        raise AnalysisInputError("two-pass manifest identity is invalid")
    if manifest.get("stage") != Stage.CONFIRMATORY.value:
        raise AnalysisInputError("primary two-pass analysis requires a confirmatory run")
    integrity = validate_manifest_files(
        manifest, repository_root=REPOSITORY_ROOT, pair_manifest_path=layout.pairs
    )
    if integrity:
        raise AnalysisInputError("two-pass manifest failed integrity: " + ", ".join(integrity))
    cells = _load_cells(layout, manifest)
    runtime_config, _harness_config, compaction_config = _runtime_configs(manifest)
    pair_sha = sha256_file(layout.pairs)
    if pair_sha != manifest.get("pair_manifest_sha256"):
        raise AnalysisInputError("two-pass pair table differs from its manifest")
    receipts = _manifest_receipts(manifest)
    # Every repository-local receipt remains immutable.  Required deployment
    # sources receive stronger fixed-location checks below.
    for receipt in receipts.values():
        _receipt_path(receipt, required_local=False)
    _task_manifest_path, task_rows = _validate_task_manifest(receipts, cells)
    try:
        regenerated_cells = make_pair_manifest(
            tasks=tuple(
                TaskRef(
                    benchmark=str(row["benchmark"]),
                    task_id=str(row["task_id"]),
                    task_sha256=str(row["task_sha256"]),
                )
                for row in task_rows
            ),
            models=tuple(manifest["models"]),
            arms=tuple(manifest["arms"]),
            operators=tuple(manifest["operators"]),
            replicates=1,
            randomization_seed=manifest.get("randomization_seed"),
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"two-pass pair design cannot be regenerated: {exc}") from exc
    if tuple(cells) != tuple(regenerated_cells):
        raise AnalysisInputError(
            "two-pass pair table is not the exact frozen model/task/method/operator product"
        )
    pass_one_path = layout.results / "deployment_pass_one.json"
    threshold_path = layout.results / "deployment_threshold_lock.json"
    schedule_path = layout.results / "deployment_schedule.json"
    for path, label in (
        (pass_one_path, "pass-one"),
        (threshold_path, "threshold"),
        (schedule_path, "schedule"),
    ):
        if path.is_symlink() or not path.is_file():
            raise AnalysisInputError(f"two-pass {label} artifact is missing or linked")
    pass_one_sha = _required_receipt(receipts, PASS_ONE_RECEIPT, pass_one_path)
    threshold_sha = _required_receipt(receipts, THRESHOLD_LOCK_RECEIPT, threshold_path)
    schedule_sha = _required_receipt(receipts, DEPLOYMENT_SCHEDULE_RECEIPT, schedule_path)
    try:
        schedule = load_deployment_schedule(schedule_path)
        _rebuild_schedule(
            schedule,
            cells,
            pair_sha256=pair_sha,
            pass_one_path=pass_one_path,
            threshold_path=threshold_path,
        )
    except (DeploymentArtifactError, ValueError) as exc:
        raise AnalysisInputError(f"two-pass schedule chain is invalid: {exc}") from exc
    if (
        schedule.pair_manifest_sha256 != pair_sha
        or schedule.pass_one_artifact_sha256 != pass_one_sha
        or schedule.threshold_lock_sha256 != threshold_sha
        or schedule.estimand.value != manifest["extra_config"]["deployment_estimand"]
    ):
        raise AnalysisInputError("two-pass schedule source/design binding changed")
    groups = _group_index(schedule, cells)
    regrade_tasks = _load_regrade_tasks(receipts, cells)

    output_root = layout.results / DEPLOYMENT_OUTPUT_SUBDIR
    job_root = layout.results / DEPLOYMENT_JOB_SUBDIR
    expected_names = {f"{cell.cell_id}.json" for cell in cells}
    _exact_directory(output_root, expected_names, context="output")
    _exact_directory(job_root, expected_names, context="job")
    event_names = {f"deployment-{cell.cell_id}.jsonl" for cell in cells}
    _exact_directory(
        layout.events,
        event_names | {"call_attempts.jsonl"},
        context="event",
    )
    strict_attempt_ids = _strict_call_attempt_ids(layout.events / "call_attempts.jsonl")
    attempts = _load_attempt_resources(layout)
    if strict_attempt_ids != set(attempts):
        raise AnalysisInputError("strict call-attempt parsing disagrees with accounting join")
    seen_attempts: set[str] = set()
    outputs: dict[str, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        output_path = output_root / f"{cell.cell_id}.json"
        job_path = job_root / f"{cell.cell_id}.json"
        event_path = layout.events / f"deployment-{cell.cell_id}.jsonl"
        output = read_json(output_path)
        job = read_json(job_path)
        events = read_jsonl(event_path)
        if not isinstance(output, Mapping) or not isinstance(job, Mapping):
            raise AnalysisInputError(f"two-pass output/job is invalid: {cell.cell_id}")
        output_sha = sha256_file(output_path)
        row = _validate_output(
            cell=cell,
            group=groups[(cell.block_id, cell.arm)],
            output=output,
            events=events,
            output_sha256=output_sha,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            pair_sha256=pair_sha,
            schedule_sha256=schedule_sha,
            pass_one_sha256=pass_one_sha,
            threshold_sha256=threshold_sha,
            runtime_config=runtime_config,
            compaction_config=compaction_config,
            regrade_task=regrade_tasks.get(
                (
                    cell.pair_key.domain,
                    cell.pair_key.task_id,
                    str(cell.pair_key.task_sha256),
                )
            ),
            attempts=attempts,
            seen_attempts=seen_attempts,
        )
        if set(job) != _JOB_FIELDS or dict(job) != {
            "deployment_runner_version": DEPLOYMENT_RUNNER_VERSION,
            "cell_id": cell.cell_id,
            "state": "complete",
            "output_sha256": output_sha,
            "success": output["evaluation"]["success"],
            "accounting_sha256": sha256_json(output["accounting"]),
        }:
            raise AnalysisInputError(f"two-pass job receipt changed: {cell.cell_id}")
        outputs[cell.cell_id] = output
        rows.append(row)
    if seen_attempts != set(attempts):
        raise AnalysisInputError(
            "two-pass call attempts are missing from outputs or belong to undeclared work"
        )
    ledger_reservations = _run_ledger_reservations(layout, str(manifest["run_id"]))
    attempt_reservations = {
        str(attempt.reservation_id) for attempt in attempts.values()
    }
    if ledger_reservations != attempt_reservations:
        raise AnalysisInputError(
            "two-pass ledger reservations do not exactly cover recorded call attempts"
        )
    canonical_count = sum(row["outcome_source"] == "canonical_regrade" for row in rows)
    report = TwoPassValidationReport(
        source_run_id=str(manifest["run_id"]),
        source_manifest_sha256=manifest_sha,
        source_pair_manifest_sha256=pair_sha,
        source_schedule_sha256=schedule_sha,
        expected_cells=len(cells),
        valid_outputs=len(outputs),
        valid_jobs=len(outputs),
        valid_event_logs=len(outputs),
        call_attempts=len(attempts),
        ledger_reservations=len(ledger_reservations),
        canonical_regraded_cells=canonical_count,
        cached_official_cells=len(rows) - canonical_count,
    )
    return _ValidatedRun(
        manifest=manifest,
        cells=cells,
        schedule=schedule,
        outputs=outputs,
        rows=tuple(rows),
        report=report,
    )


def validate_two_pass_run(
    layout: RunLayout, *, expected_manifest_sha256: str
) -> TwoPassValidationReport:
    """Validate one complete two-pass run without writing or provider access."""

    return _validated_run(
        layout, expected_manifest_sha256=expected_manifest_sha256
    ).report


def _validated_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    materialized = tuple(dict(row) for row in rows)
    if not materialized:
        raise AnalysisInputError("two-pass outcome rows are empty")
    required = {
        "cell_id",
        "model",
        "benchmark",
        "task_id",
        "replicate_id",
        "unit_id",
        "observation_class",
        "method",
        "operator",
        "deployment_mode",
        "estimand",
        "success",
        "outcome_source",
        "observations",
        "scheduled_actions",
        "action_rate",
        "acted_on_task",
        "applied_interventions",
        "task_tokens",
        "observer_tokens",
        "total_tokens",
        "latency_ms",
        "actual_cost_usd",
        "reported_cost_usd",
        "estimated_cost_usd",
        "upper_bound_cost_usd",
        "failed_retry_attempts",
    }
    seen: set[str] = set()
    for index, row in enumerate(materialized):
        if set(row) != required:
            raise AnalysisInputError(f"two-pass row {index} has an invalid schema")
        for key in (
            "cell_id",
            "model",
            "benchmark",
            "task_id",
            "unit_id",
            "observation_class",
            "method",
            "operator",
            "estimand",
            "outcome_source",
        ):
            if not isinstance(row[key], str) or not row[key]:
                raise AnalysisInputError(f"two-pass row {index} has invalid {key}")
        if row["cell_id"] in seen:
            raise AnalysisInputError(f"two-pass rows duplicate cell {row['cell_id']}")
        seen.add(row["cell_id"])
        if (
            row["deployment_mode"] != TWO_PASS_DEPLOYMENT_MODE
            or row["observation_class"] != observation_class(row["method"])
            or not isinstance(row["success"], bool)
        ):
            raise AnalysisInputError(f"two-pass row {index} has invalid design/outcome")
        if (
            isinstance(row["replicate_id"], bool)
            or not isinstance(row["replicate_id"], int)
            or row["replicate_id"] != 0
            or row["unit_id"] != f"{row['task_id']}/r0"
            or row["outcome_source"]
            not in {
                "canonical_regrade",
                "cached_official_not_provider_free_regradable",
            }
        ):
            raise AnalysisInputError(
                f"two-pass row {index} has invalid task-unit/outcome provenance"
            )
        for key in (
            "observations",
            "scheduled_actions",
            "acted_on_task",
            "applied_interventions",
            "task_tokens",
            "observer_tokens",
            "total_tokens",
            "latency_ms",
            "failed_retry_attempts",
        ):
            _nonnegative_number(row[key], context=f"two-pass row {index} {key}", integer=True)
        _nonnegative_number(row["action_rate"], context=f"two-pass row {index} action_rate")
        _nonnegative_number(row["actual_cost_usd"], context=f"two-pass row {index} cost")
        for key in (
            "reported_cost_usd",
            "estimated_cost_usd",
            "upper_bound_cost_usd",
        ):
            _nonnegative_number(row[key], context=f"two-pass row {index} {key}")
        if (
            row["observations"] < 1
            or row["scheduled_actions"] > row["observations"]
            or row["action_rate"] != row["scheduled_actions"] / row["observations"]
            or row["acted_on_task"] != int(row["scheduled_actions"] > 0)
            or row["applied_interventions"] != row["scheduled_actions"]
            or row["total_tokens"] != row["task_tokens"] + row["observer_tokens"]
            or not math.isclose(
                row["actual_cost_usd"],
                row["reported_cost_usd"]
                + row["estimated_cost_usd"]
                + row["upper_bound_cost_usd"],
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise AnalysisInputError(f"two-pass row {index} has inconsistent rates/resources")
    return materialized


def summarize_two_pass_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = _BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
    confidence: float = _CONFIDENCE,
) -> tuple[
    tuple[TwoPassMetricSummary, ...],
    tuple[TwoPassOperatorEffect, ...],
    tuple[TwoPassMethodEffect, ...],
]:
    """Return absolute summaries plus task-paired operator and method effects."""

    materialized = _validated_rows(rows)
    slices: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, dict[str, Any]]]] = {}
    for row in materialized:
        slice_key = (row["model"], row["benchmark"], row["estimand"])
        treatment = (row["method"], row["operator"])
        units = slices.setdefault(slice_key, {}).setdefault(treatment, {})
        if row["unit_id"] in units:
            raise AnalysisInputError("two-pass treatment duplicates one source-task unit")
        units[row["unit_id"]] = row
    summaries: list[TwoPassMetricSummary] = []
    operator_effects: list[TwoPassOperatorEffect] = []
    method_effects: list[TwoPassMethodEffect] = []
    for (model, benchmark, estimand), treatments in sorted(slices.items()):
        methods = sorted({method for method, _ in treatments})
        operators = sorted({operator for _, operator in treatments})
        expected = {(method, operator) for method in methods for operator in operators}
        if set(treatments) != expected or _CONTROL_OPERATOR not in operators or len(operators) < 2:
            raise AnalysisInputError(
                f"two-pass slice {model}/{benchmark}/{estimand} is not the exact method/operator product"
            )
        units = set(next(iter(treatments.values())))
        if not units or any(set(rows_by_unit) != units for rows_by_unit in treatments.values()):
            raise AnalysisInputError("two-pass treatment denominators are not task-paired")
        ordered_units = sorted(units)
        for (method, operator), rows_by_unit in sorted(treatments.items()):
            for metric, unit, favorable in _METRICS:
                values = [float(rows_by_unit[unit_id][metric]) for unit_id in ordered_units]
                low, high = _bootstrap_interval(
                    values,
                    identity=(model, benchmark, estimand, method, operator, metric, "absolute"),
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
                summaries.append(
                    TwoPassMetricSummary(
                        model=model,
                        benchmark=benchmark,
                        observation_class=observation_class(method),
                        method=method,
                        operator=operator,
                        estimand=estimand,
                        metric=metric,
                        unit=unit,
                        favorable_direction=favorable,
                        n_tasks=len(values),
                        mean=fmean(values),
                        ci_low=low,
                        ci_high=high,
                        confidence=float(confidence),
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
        for method in methods:
            control = treatments[(method, _CONTROL_OPERATOR)]
            for operator in operators:
                if operator == _CONTROL_OPERATOR:
                    continue
                treated = treatments[(method, operator)]
                for metric, unit, favorable in _METRICS:
                    control_values = [float(control[key][metric]) for key in ordered_units]
                    treated_values = [float(treated[key][metric]) for key in ordered_units]
                    differences = [b - a for a, b in zip(control_values, treated_values, strict=True)]
                    low, high = _bootstrap_interval(
                        differences,
                        identity=(model, benchmark, estimand, method, operator, metric, "operator"),
                        iterations=bootstrap_iterations,
                        seed=bootstrap_seed,
                        confidence=confidence,
                    )
                    operator_effects.append(
                        TwoPassOperatorEffect(
                            model=model,
                            benchmark=benchmark,
                            observation_class=observation_class(method),
                            method=method,
                            operator=operator,
                            control_operator=_CONTROL_OPERATOR,
                            estimand=estimand,
                            metric=metric,
                            unit=unit,
                            favorable_direction=favorable,
                            n_tasks=len(differences),
                            control_mean=fmean(control_values),
                            operator_mean=fmean(treated_values),
                            effect=fmean(differences),
                            ci_low=low,
                            ci_high=high,
                            confidence=float(confidence),
                            bootstrap_iterations=bootstrap_iterations,
                            bootstrap_seed=bootstrap_seed,
                        )
                    )
        for operator in operators:
            for left_index, reference_method in enumerate(methods):
                for comparison_method in methods[left_index + 1 :]:
                    reference = treatments[(reference_method, operator)]
                    comparison = treatments[(comparison_method, operator)]
                    for metric, unit, favorable in _METRICS:
                        reference_values = [float(reference[key][metric]) for key in ordered_units]
                        comparison_values = [float(comparison[key][metric]) for key in ordered_units]
                        differences = [
                            b - a
                            for a, b in zip(reference_values, comparison_values, strict=True)
                        ]
                        low, high = _bootstrap_interval(
                            differences,
                            identity=(
                                model,
                                benchmark,
                                estimand,
                                operator,
                                reference_method,
                                comparison_method,
                                metric,
                                "method",
                            ),
                            iterations=bootstrap_iterations,
                            seed=bootstrap_seed,
                            confidence=confidence,
                        )
                        method_effects.append(
                            TwoPassMethodEffect(
                                model=model,
                                benchmark=benchmark,
                                reference_observation_class=observation_class(reference_method),
                                reference_method=reference_method,
                                comparison_observation_class=observation_class(comparison_method),
                                comparison_method=comparison_method,
                                operator=operator,
                                estimand=estimand,
                                metric=metric,
                                unit=unit,
                                favorable_direction=favorable,
                                n_tasks=len(differences),
                                reference_mean=fmean(reference_values),
                                comparison_mean=fmean(comparison_values),
                                effect=fmean(differences),
                                ci_low=low,
                                ci_high=high,
                                confidence=float(confidence),
                                bootstrap_iterations=bootstrap_iterations,
                                bootstrap_seed=bootstrap_seed,
                            )
                        )
    return tuple(summaries), tuple(operator_effects), tuple(method_effects)


def extract_two_pass_run(
    layout: RunLayout,
    *,
    expected_manifest_sha256: str,
    bootstrap_iterations: int = _BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate and analyze one complete two-pass deployment, read-only."""

    validated = _validated_run(
        layout, expected_manifest_sha256=expected_manifest_sha256
    )
    summaries, operator_effects, method_effects = summarize_two_pass_outcomes(
        validated.rows,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "two_pass_analysis_version": TWO_PASS_ANALYSIS_VERSION,
        "artifact_type": TWO_PASS_ANALYSIS_TYPE,
        "source_run_id": validated.report.source_run_id,
        "source_manifest_sha256": validated.report.source_manifest_sha256,
        "source_pair_manifest_sha256": validated.report.source_pair_manifest_sha256,
        "source_schedule_sha256": validated.report.source_schedule_sha256,
        "deployment_mode": TWO_PASS_DEPLOYMENT_MODE,
        "estimand": validated.schedule.estimand.value,
        "statistical_unit": "source_task",
        "comparison_semantics": (
            "absolute outcomes use every declared treatment cell; operator effects are "
            "paired operator minus none; method effects are paired comparison minus "
            "reference on identical source tasks"
        ),
        "resource_semantics": (
            "pass-two tokens, latency, and dollars include every failed retry and the "
            "successful final task/active-probe attempt; frozen pass-one observer cost "
            "is not duplicated into each replay cell"
        ),
        "validation": validated.report.as_dict(),
        "rows": list(validated.rows),
        "metric_summaries": [asdict(row) for row in summaries],
        "operator_effects": [asdict(row) for row in operator_effects],
        "method_effects": [asdict(row) for row in method_effects],
    }


def _safe_stem(value: str) -> str:
    return _SAFE_STEM.sub("-", value).strip("-") or "slice"


def _atomic_write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise AnalysisInputError(f"cannot write empty paper table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise AnalysisInputError(f"paper table {path.name} has inconsistent columns")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return _atomic_write_text(path, buffer.getvalue())


def write_two_pass_tables(
    analysis: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, ...]:
    """Write compact, machine-readable paper tables from validated analysis."""

    if analysis.get("artifact_type") != TWO_PASS_ANALYSIS_TYPE:
        raise AnalysisInputError("two-pass table input has the wrong artifact type")
    destination = Path(output_dir)
    tables = (
        ("task_rows.csv", analysis.get("rows")),
        ("metric_summaries.csv", analysis.get("metric_summaries")),
        ("operator_effects.csv", analysis.get("operator_effects")),
        ("method_effects.csv", analysis.get("method_effects")),
    )
    written = []
    for name, rows in tables:
        if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
            raise AnalysisInputError(f"two-pass table input lacks {name}")
        written.append(_write_csv(destination / name, rows))
    return tuple(written)


def write_two_pass_figures(
    analysis: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, ...]:
    """Write success, action-rate, token, latency, and cost figures per slice."""

    if analysis.get("artifact_type") != TWO_PASS_ANALYSIS_TYPE:
        raise AnalysisInputError("two-pass figure input has the wrong artifact type")
    summaries = analysis.get("metric_summaries")
    if not isinstance(summaries, list) or not summaries:
        raise AnalysisInputError("two-pass figure input lacks metric summaries")
    destination = Path(output_dir)
    written: list[Path] = []
    for metric, label in _FIGURE_METRICS:
        grouped: dict[tuple[str, str, str], list[DeploymentBar]] = {}
        for row in summaries:
            if not isinstance(row, Mapping) or row.get("metric") != metric:
                continue
            grouped.setdefault(
                (str(row["model"]), str(row["benchmark"]), str(row["estimand"])), []
            ).append(
                DeploymentBar(
                    observation_class=str(row["observation_class"]),
                    operator=str(row["operator"]),
                    method=str(row["method"]),
                    value=float(row["mean"]),
                    n_tasks=int(row["n_tasks"]),
                    ci_low=min(float(row["mean"]), float(row["ci_low"])),
                    ci_high=max(float(row["mean"]), float(row["ci_high"])),
                )
            )
        for (model, benchmark, estimand), bars in sorted(grouped.items()):
            stem = "-".join(map(_safe_stem, (metric, benchmark, model, estimand)))
            artifact = write_deployment_grouped_bars(
                bars,
                destination / f"two-pass-{stem}.svg",
                title=f"Two-pass {metric}: {model} on {benchmark} ({estimand})",
                y_label=label,
            )
            written.extend((artifact.svg_path, artifact.data_path))
    if not written:
        raise AnalysisInputError("two-pass figure input has no plottable metrics")
    return tuple(written)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="audit a complete two-pass deployment")
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    extract = commands.add_parser("extract", help="audit and analyze a two-pass deployment")
    extract.add_argument("--run-id", required=True)
    extract.add_argument("--manifest-sha256", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--tables")
    extract.add_argument("--figures")
    extract.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    extract.add_argument("--bootstrap-iterations", type=int, default=_BOOTSTRAP_ITERATIONS)
    extract.add_argument("--bootstrap-seed", type=int, default=_BOOTSTRAP_SEED)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        layout = RunLayout.for_run(args.artifacts, args.run_id)
        if args.command == "validate":
            report = validate_two_pass_run(
                layout, expected_manifest_sha256=args.manifest_sha256
            )
            atomic_write_json(args.output, report.as_dict())
        else:
            analysis = extract_two_pass_run(
                layout,
                expected_manifest_sha256=args.manifest_sha256,
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )
            atomic_write_json(args.output, analysis)
            if args.tables:
                write_two_pass_tables(analysis, args.tables)
            if args.figures:
                write_two_pass_figures(analysis, args.figures)
        return 0
    except (
        AnalysisInputError,
        DeploymentArtifactError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TWO_PASS_ANALYSIS_TYPE",
    "TWO_PASS_ANALYSIS_VERSION",
    "TWO_PASS_VALIDATION_TYPE",
    "TwoPassMetricSummary",
    "TwoPassMethodEffect",
    "TwoPassOperatorEffect",
    "TwoPassValidationReport",
    "extract_two_pass_run",
    "main",
    "parser",
    "summarize_two_pass_outcomes",
    "validate_two_pass_run",
    "write_two_pass_figures",
    "write_two_pass_tables",
]
