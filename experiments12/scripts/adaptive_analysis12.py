"""Fail-closed paper analysis for Experiment 12 online deployment runs.

The declared pair table is always the denominator.  This module validates every
online output and its append-only event receipt, reconciles every paid attempt
to the run ledger, then reports task-paired operator effects and absolute
task-level deployment performance.  It never dispatches provider calls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from experiments12.adaptive_deployment12 import (
    ADAPTIVE_DEPLOYMENT_MODE,
    ADAPTIVE_JOB_SUBDIR,
    ADAPTIVE_POLICY,
    ADAPTIVE_RESULT_SUBDIR,
    PRIMARY_MAX_ACTIONS_PER_TASK,
    AdaptiveDeploymentError,
    _manifest_mode,
    _runtime_config,
    _validate_existing,
    extract_adaptive_outcomes,
)
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
from experiments12.core.schemas import CallStatus
from experiments12.deployment12 import THRESHOLD_LOCK_RECEIPT, load_threshold_lock
from experiments12.domains.base import DomainTask
from experiments12.domains.evolving_intent import EvolvingIntentAdapter
from experiments12.figures12 import DeploymentBar, write_deployment_grouped_bars
from experiments12.harness12 import HarnessConfig
from experiments12.manifest12 import RunLayout, validate_manifest_files
from experiments12.models12 import CATALOG
from experiments12.monitors.judge import JUDGE_MODEL_NAME
from experiments12.operators12 import CompactionConfig
from experiments12.pairing12 import JobCell
from experiments12.runner12 import pair_task_id


ADAPTIVE_ANALYSIS_VERSION = 1
ADAPTIVE_ANALYSIS_TYPE = "online_adaptive_deployment_analysis"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_CONTROL_OPERATOR = "none"
_CONFIDENCE = 0.95
_BOOTSTRAP_ITERATIONS = 2_000
_BOOTSTRAP_SEED = 12_012

_METRICS: tuple[tuple[str, str, str], ...] = (
    ("success", "proportion", "higher"),
    ("threshold_firings", "count", "descriptive"),
    ("selected_actions", "count", "descriptive"),
    ("task_tokens", "tokens", "lower"),
    ("observer_tokens", "tokens", "lower"),
    ("total_tokens", "tokens", "lower"),
    ("latency_ms", "milliseconds", "lower"),
    ("actual_cost_usd", "USD", "lower"),
)


@dataclass(frozen=True, slots=True)
class AdaptiveMetricSummary:
    model: str
    benchmark: str
    observation_class: str
    method: str
    operator: str
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
class AdaptiveOperatorEffect:
    model: str
    benchmark: str
    observation_class: str
    method: str
    operator: str
    control_operator: str
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


def observation_class(method: str) -> str:
    """Return the frozen paper grouping for one observation method."""

    if not isinstance(method, str) or not method:
        raise AnalysisInputError("adaptive method must be a non-empty string")
    if method.startswith("active_"):
        return "active"
    if method in {"turn_clock", "context_use"}:
        return "baseline"
    if method.startswith("frozen_probe:") or method == "frozen_quiz":
        return "passive-behavioral"
    if method in {"trace_judge", "trace_rules"}:
        return "passive-observational"
    raise AnalysisInputError(f"adaptive method has no frozen paper class: {method!r}")


def _digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AnalysisInputError(f"{context} must be a lowercase SHA256 digest")
    return value


def _number(value: Any, *, context: str, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisInputError(f"{context} must be a finite non-negative number")
    if integer and not isinstance(value, int):
        raise AnalysisInputError(f"{context} must be a non-negative integer")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise AnalysisInputError(f"{context} must be a finite non-negative number")
    return result


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
    prefix = "\0".join(("exp12/adaptive-task-bootstrap/v1", str(seed), *identity))
    population = len(values)
    bootstrap: list[float] = []
    for iteration in range(iterations):
        sampled: list[float] = []
        for draw in range(population):
            material = f"{prefix}\0{iteration}\0{draw}".encode("utf-8")
            index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % population
            sampled.append(values[index])
        bootstrap.append(fmean(sampled))
    bootstrap.sort()
    tail = (1 - float(confidence)) / 2
    return _quantile(bootstrap, tail), _quantile(bootstrap, 1 - tail)


def _validated_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    materialized: list[dict[str, Any]] = []
    cell_ids: set[str] = set()
    required = {
        "cell_id",
        "model",
        "benchmark",
        "task_id",
        "replicate_id",
        "unit_id",
        "method",
        "observation_class",
        "operator",
        "deployment_mode",
        "success",
        "observations",
        "threshold_firings",
        "selected_actions",
        "applied_interventions",
        "task_tokens",
        "observer_tokens",
        "total_tokens",
        "latency_ms",
        "actual_cost_usd",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise AnalysisInputError(f"adaptive row {index} has an invalid schema")
        row = dict(raw)
        for key in (
            "cell_id",
            "model",
            "benchmark",
            "task_id",
            "unit_id",
            "method",
            "observation_class",
            "operator",
        ):
            if not isinstance(row[key], str) or not row[key]:
                raise AnalysisInputError(f"adaptive row {index} has invalid {key}")
        if row["cell_id"] in cell_ids:
            raise AnalysisInputError(f"adaptive row duplicates cell {row['cell_id']}")
        cell_ids.add(row["cell_id"])
        if row["deployment_mode"] != ADAPTIVE_DEPLOYMENT_MODE:
            raise AnalysisInputError("adaptive row is not from online deployment")
        if row["observation_class"] != observation_class(row["method"]):
            raise AnalysisInputError("adaptive row observation class changed")
        replicate = row["replicate_id"]
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise AnalysisInputError("adaptive replicate_id must be non-negative")
        if not isinstance(row["success"], bool):
            raise AnalysisInputError("adaptive success must be boolean")
        for key in (
            "observations",
            "threshold_firings",
            "selected_actions",
            "applied_interventions",
            "task_tokens",
            "observer_tokens",
            "total_tokens",
            "latency_ms",
        ):
            _number(row[key], context=f"adaptive row {index} {key}", integer=True)
        _number(row["actual_cost_usd"], context=f"adaptive row {index} cost")
        if row["total_tokens"] != row["task_tokens"] + row["observer_tokens"]:
            raise AnalysisInputError("adaptive total_tokens does not equal its categories")
        if row["selected_actions"] > PRIMARY_MAX_ACTIONS_PER_TASK:
            raise AnalysisInputError("adaptive row exceeds the primary one-action cap")
        if row["applied_interventions"] != row["selected_actions"]:
            raise AnalysisInputError("adaptive selected actions and interventions differ")
        materialized.append(row)
    if not materialized:
        raise AnalysisInputError("adaptive outcome rows are empty")
    return tuple(materialized)


def summarize_adaptive_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = _BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
    confidence: float = _CONFIDENCE,
) -> tuple[tuple[AdaptiveMetricSummary, ...], tuple[AdaptiveOperatorEffect, ...]]:
    """Summarize exact treatment products and paired operator-minus-none effects."""

    materialized = _validated_rows(rows)
    slices: dict[tuple[str, str], dict[tuple[str, str], dict[str, dict[str, Any]]]] = {}
    for row in materialized:
        slice_key = (row["model"], row["benchmark"])
        treatment = (row["method"], row["operator"])
        units = slices.setdefault(slice_key, {}).setdefault(treatment, {})
        if row["unit_id"] in units:
            raise AnalysisInputError(
                f"adaptive treatment duplicates unit {slice_key}/{treatment}/{row['unit_id']}"
            )
        units[row["unit_id"]] = row

    summaries: list[AdaptiveMetricSummary] = []
    effects: list[AdaptiveOperatorEffect] = []
    for (model, benchmark), treatments in sorted(slices.items()):
        methods = sorted({method for method, _operator in treatments})
        operators = sorted({operator for _method, operator in treatments})
        expected = {(method, operator) for method in methods for operator in operators}
        if set(treatments) != expected or _CONTROL_OPERATOR not in operators or len(operators) < 2:
            raise AnalysisInputError(
                f"adaptive slice {model}/{benchmark} is not the exact method/operator product"
            )
        reference_units = set(next(iter(treatments.values())))
        if not reference_units or any(set(units) != reference_units for units in treatments.values()):
            raise AnalysisInputError(
                f"adaptive slice {model}/{benchmark} has unpaired treatment denominators"
            )
        ordered_units = sorted(reference_units)
        for (method, operator), units in sorted(treatments.items()):
            observation = observation_class(method)
            for metric, unit, favorable in _METRICS:
                values = [
                    float(units[unit_id][metric]) for unit_id in ordered_units
                ]
                ci_low, ci_high = _bootstrap_interval(
                    values,
                    identity=(model, benchmark, method, operator, metric, "absolute"),
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                    confidence=confidence,
                )
                summaries.append(
                    AdaptiveMetricSummary(
                        model=model,
                        benchmark=benchmark,
                        observation_class=observation,
                        method=method,
                        operator=operator,
                        metric=metric,
                        unit=unit,
                        favorable_direction=favorable,
                        n_tasks=len(values),
                        mean=fmean(values),
                        ci_low=ci_low,
                        ci_high=ci_high,
                        confidence=float(confidence),
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
        for method in methods:
            control = treatments[(method, _CONTROL_OPERATOR)]
            observation = observation_class(method)
            for operator in operators:
                if operator == _CONTROL_OPERATOR:
                    continue
                treated = treatments[(method, operator)]
                for metric, unit, favorable in _METRICS:
                    control_values = [
                        float(control[unit_id][metric]) for unit_id in ordered_units
                    ]
                    treated_values = [
                        float(treated[unit_id][metric]) for unit_id in ordered_units
                    ]
                    differences = [
                        value - baseline
                        for baseline, value in zip(control_values, treated_values)
                    ]
                    ci_low, ci_high = _bootstrap_interval(
                        differences,
                        identity=(model, benchmark, method, operator, metric, "paired"),
                        iterations=bootstrap_iterations,
                        seed=bootstrap_seed,
                        confidence=confidence,
                    )
                    effects.append(
                        AdaptiveOperatorEffect(
                            model=model,
                            benchmark=benchmark,
                            observation_class=observation,
                            method=method,
                            operator=operator,
                            control_operator=_CONTROL_OPERATOR,
                            metric=metric,
                            unit=unit,
                            favorable_direction=favorable,
                            n_tasks=len(differences),
                            control_mean=fmean(control_values),
                            operator_mean=fmean(treated_values),
                            effect=fmean(differences),
                            ci_low=ci_low,
                            ci_high=ci_high,
                            confidence=float(confidence),
                            bootstrap_iterations=bootstrap_iterations,
                            bootstrap_seed=bootstrap_seed,
                        )
                    )
    return tuple(summaries), tuple(effects)


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
        raise AnalysisInputError(
            f"{context} lacks failed-retry*/successful-final ordering"
        )
    spec = CATALOG.models.get(expected_model)
    if spec is None:
        raise AnalysisInputError(f"{context} uses an unknown model")
    for attempt_number, attempt in enumerate(selected, 1):
        if (
            attempt.attempt_number != attempt_number
            or attempt.request_key
            != f"{expected_request_key}/attempt-{attempt_number}"
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
            usage.get("cached_input_tokens", 0),
            usage.get("reasoning_tokens", 0),
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
    except (InvalidOperation, ValueError) as exc:
        raise AnalysisInputError(f"{context} has invalid recorded cost") from exc
    if (
        not recorded_cost.is_finite()
        or recorded_cost < 0
        or recorded_cost != final.actual_cost_usd
    ):
        raise AnalysisInputError(f"{context} cost disagrees with its successful attempt")
    seen.update(event_ids)
    return {
        "tokens": sum(item.input_tokens + item.output_tokens for item in selected),
        "latency_ms": sum(item.elapsed_ms for item in selected),
        "actual_cost_usd": sum(
            (item.actual_cost_usd for item in selected), Decimal("0")
        ),
    }


def _observer_purpose(method: str) -> str | None:
    if method.startswith("active_"):
        return "adaptive_active_probe"
    if method.startswith("frozen_probe:"):
        return "adaptive_frozen_probe"
    if method == "frozen_quiz":
        return "adaptive_frozen_quiz"
    if method == "trace_judge":
        return "adaptive_trace_judge"
    if method in {"trace_rules", "turn_clock", "context_use"}:
        return None
    raise AnalysisInputError(f"adaptive method has no call-purpose contract: {method!r}")


def _observer_request_kind(method: str) -> str | None:
    if method.startswith("active_"):
        return "active-probe"
    if method.startswith("frozen_probe:"):
        return "frozen-probe"
    if method == "frozen_quiz":
        return "frozen-quiz"
    if method == "trace_judge":
        return "trace-judge"
    if method in {"trace_rules", "turn_clock", "context_use"}:
        return None
    raise AnalysisInputError(f"adaptive method has no request-key contract: {method!r}")


def _require_start_runtime_binding(
    start: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    cell_id: str,
) -> None:
    """Require the executed cell to use the manifest-frozen runtime exactly."""

    extra = manifest.get("extra_config")
    frozen = extra.get("adaptive_runtime") if isinstance(extra, Mapping) else None
    if not isinstance(frozen, Mapping) or start.get("runtime_config") != frozen:
        raise AnalysisInputError(
            f"adaptive start runtime differs from its manifest: {cell_id}"
        )


def _resource_row(
    cell: JobCell,
    output: Mapping[str, Any],
    base: Mapping[str, Any],
    attempts: Mapping[str, _AttemptResource],
    seen: set[str],
) -> dict[str, Any]:
    task_tokens = observer_tokens = latency_ms = 0
    actual_cost = Decimal("0")
    task_records = output.get("task_records")
    signal_records = output.get("signal_records")
    if not isinstance(task_records, list) or not isinstance(signal_records, list):
        raise AnalysisInputError(f"adaptive output records are invalid: {cell.cell_id}")
    run_id = output.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise AnalysisInputError(f"adaptive output lacks its run ID: {cell.cell_id}")
    for index, record in enumerate(task_records, 1):
        if not isinstance(record, Mapping):
            raise AnalysisInputError(f"adaptive task record is invalid: {cell.cell_id}")
        resource = _attempt_totals(
            record.get("call"),
            expected_purpose="adaptive_agent_turn",
            expected_request_key=(
                f"{run_id}/{cell.cell_id}/adaptive-task-{index}"
            ),
            expected_model=cell.pair_key.model,
            attempts=attempts,
            seen=seen,
            context=f"{cell.cell_id} task {index}",
        )
        task_tokens += resource["tokens"]
        latency_ms += resource["latency_ms"]
        actual_cost += resource["actual_cost_usd"]
    purpose = _observer_purpose(cell.arm)
    request_kind = _observer_request_kind(cell.arm)
    for index, record in enumerate(signal_records, 1):
        if not isinstance(record, Mapping):
            raise AnalysisInputError(f"adaptive signal record is invalid: {cell.cell_id}")
        call = record.get("call")
        if purpose is None:
            if call is not None:
                raise AnalysisInputError(
                    f"deterministic adaptive observer contains a paid call: {cell.cell_id}"
                )
            continue
        resource = _attempt_totals(
            call,
            expected_purpose=purpose,
            expected_request_key=(
                f"{run_id}/{cell.cell_id}/adaptive-{request_kind}-"
                f"{record.get('checkpoint')}"
            ),
            expected_model=(
                JUDGE_MODEL_NAME if cell.arm == "trace_judge" else cell.pair_key.model
            ),
            attempts=attempts,
            seen=seen,
            context=f"{cell.cell_id} observer {index}",
        )
        observer_tokens += resource["tokens"]
        latency_ms += resource["latency_ms"]
        actual_cost += resource["actual_cost_usd"]
    unit_id = f"{cell.pair_key.task_id}/r{cell.pair_key.replicate_id}"
    return {
        "cell_id": cell.cell_id,
        "model": base["model"],
        "benchmark": base["benchmark"],
        "task_id": cell.pair_key.task_id,
        "replicate_id": cell.pair_key.replicate_id,
        "unit_id": unit_id,
        "method": base["method"],
        "observation_class": observation_class(base["method"]),
        "operator": base["operator"],
        "deployment_mode": base["deployment_mode"],
        "success": base["success"],
        "observations": base["observations"],
        "threshold_firings": base["threshold_firings"],
        "selected_actions": base["selected_actions"],
        "applied_interventions": base["applied_interventions"],
        "task_tokens": task_tokens,
        "observer_tokens": observer_tokens,
        "total_tokens": task_tokens + observer_tokens,
        "latency_ms": latency_ms,
        "actual_cost_usd": float(actual_cost),
    }


def _manifest_local_receipt(
    manifest: Mapping[str, Any], name: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        row
        for row in manifest.get("benchmark_receipts", ())
        if isinstance(row, Mapping) and row.get("name") == name
    ]
    if len(matches) != 1:
        raise AnalysisInputError(f"adaptive manifest lacks one exact {name} receipt")
    receipt = matches[0]
    raw = receipt.get("path")
    digest = receipt.get("sha256")
    if (
        not isinstance(raw, str)
        or not raw
        or raw.startswith("external:")
        or not isinstance(digest, str)
    ):
        raise AnalysisInputError(f"adaptive {name} receipt is not locally replayable")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisInputError(f"adaptive {name} receipt path is unsafe")
    repository = REPOSITORY_ROOT.resolve()
    candidate = repository / relative
    current = candidate
    while current != repository:
        if current.is_symlink():
            raise AnalysisInputError(f"adaptive {name} receipt is linked")
        current = current.parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise AnalysisInputError(f"adaptive {name} receipt escapes the repository") from exc
    if resolved.is_symlink() or not resolved.is_file() or sha256_file(resolved) != digest:
        raise AnalysisInputError(f"adaptive {name} receipt changed")
    return receipt, resolved


def _adaptive_replay_inputs(
    manifest: Mapping[str, Any], cells: Sequence[JobCell]
) -> tuple[dict[tuple[str, str, str], DomainTask], CompactionConfig]:
    """Load the canonical task texts and reproduce the manifest runtime lock."""

    extra = manifest.get("extra_config")
    runtime = extra.get("adaptive_runtime") if isinstance(extra, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise AnalysisInputError("adaptive manifest lacks its runtime replay lock")
    compaction = runtime.get("compaction")
    if not isinstance(compaction, Mapping):
        raise AnalysisInputError("adaptive compaction replay lock is invalid")
    try:
        harness_config = HarnessConfig(
            checkpoint_every=runtime["checkpoint_every"],
            task_max_output_tokens=runtime["task_max_output_tokens"],
            probe_max_output_tokens=runtime["probe_max_output_tokens"],
            temperature=runtime["temperature"],
        )
        compaction_config = CompactionConfig(
            keep_last_messages=compaction["keep_last_messages"],
            max_excerpt_bytes=compaction["max_excerpt_bytes"],
            max_summary_bytes=compaction["max_summary_bytes"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisInputError("adaptive runtime replay lock is invalid") from exc
    try:
        canonical_runtime = _runtime_config(harness_config, compaction_config)
    except AdaptiveDeploymentError as exc:
        raise AnalysisInputError(str(exc)) from exc
    if dict(runtime) != canonical_runtime:
        raise AnalysisInputError("adaptive runtime replay values/hashes do not reproduce")

    receipt, dataset = _manifest_local_receipt(
        manifest, "evolving_rendered_dataset"
    )
    try:
        tasks = EvolvingIntentAdapter(
            dataset, expected_sha256=str(receipt["sha256"])
        ).load_tasks()
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisInputError("adaptive frozen Evolving dataset is invalid") from exc
    available = {
        (task.domain, pair_task_id(task), task.task_sha256): task for task in tasks
    }
    if len(available) != len(tasks):
        raise AnalysisInputError(
            "adaptive frozen Evolving dataset duplicates a canonical task identity"
        )
    declared = {
        (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        for cell in cells
    }
    if not declared.issubset(available):
        raise AnalysisInputError(
            "adaptive frozen Evolving dataset does not cover every declared task"
        )
    return {key: available[key] for key in declared}, compaction_config


def _load_declared_run(
    layout: RunLayout, *, expected_manifest_sha256: str
) -> tuple[Mapping[str, Any], tuple[JobCell, ...], str, str]:
    expected = _digest(expected_manifest_sha256, context="expected manifest SHA256")
    if layout.manifest.is_symlink() or layout.pairs.is_symlink():
        raise AnalysisInputError("adaptive manifest and pair table must not be symlinks")
    manifest_sha = sha256_file(layout.manifest)
    if manifest_sha != expected:
        raise AnalysisInputError("adaptive manifest differs from its external SHA256")
    manifest = read_json(layout.manifest)
    if not isinstance(manifest, Mapping) or manifest.get("run_id") != layout.root.name:
        raise AnalysisInputError("adaptive manifest identity is invalid")
    integrity = validate_manifest_files(
        manifest,
        repository_root=REPOSITORY_ROOT,
        pair_manifest_path=layout.pairs,
    )
    if integrity:
        raise AnalysisInputError("adaptive manifest failed integrity: " + ", ".join(integrity))
    try:
        _manifest_mode(manifest)
    except AdaptiveDeploymentError as exc:
        raise AnalysisInputError(str(exc)) from exc
    pair_sha = sha256_file(layout.pairs)
    if pair_sha != manifest.get("pair_manifest_sha256"):
        raise AnalysisInputError("adaptive pair table differs from its manifest")
    try:
        cells = tuple(JobCell.from_dict(row) for row in read_jsonl(layout.pairs))
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisInputError(f"adaptive pair table is invalid: {exc}") from exc
    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise AnalysisInputError("adaptive pair table is empty or duplicates cells")
    models = manifest.get("models")
    methods = manifest.get("arms")
    operators = manifest.get("operators")
    if any(
        not isinstance(values, list)
        or not values
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
        for values in (models, methods, operators)
    ):
        raise AnalysisInputError("adaptive manifest treatment names are invalid")
    expected_treatments = {(method, operator) for method in methods for operator in operators}
    blocks: dict[str, list[JobCell]] = {}
    for cell in cells:
        if (
            cell.pair_key.model not in models
            or cell.arm not in methods
            or cell.operator not in operators
        ):
            raise AnalysisInputError("adaptive cell falls outside its manifest")
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
            raise AnalysisInputError(f"adaptive block is incomplete or mixed: {block_id}")
    extra = manifest.get("extra_config")
    if (
        not isinstance(extra, Mapping)
        or extra.get("n_cells") != len(cells)
        or extra.get("natural_max_actions_per_task") != PRIMARY_MAX_ACTIONS_PER_TASK
    ):
        raise AnalysisInputError("adaptive manifest count/action cap is invalid")
    threshold_path = layout.results / "deployment_threshold_lock.json"
    receipt = [
        row
        for row in manifest.get("benchmark_receipts", ())
        if isinstance(row, Mapping) and row.get("name") == THRESHOLD_LOCK_RECEIPT
    ]
    threshold_sha = sha256_file(threshold_path)
    if (
        len(receipt) != 1
        or receipt[0].get("sha256") != threshold_sha
        or extra.get("threshold_lock_sha256") != threshold_sha
    ):
        raise AnalysisInputError("adaptive threshold lock receipt is invalid")
    threshold_lock = load_threshold_lock(threshold_path)
    if threshold_lock.natural_max_actions_per_task != PRIMARY_MAX_ACTIONS_PER_TASK:
        raise AnalysisInputError("adaptive threshold lock changed the one-action cap")
    expected_slices = {
        (cell.pair_key.model, cell.pair_key.domain, cell.arm) for cell in cells
    }
    threshold_slices = {
        (row.model, row.benchmark, row.method) for row in threshold_lock.methods
    }
    if threshold_slices != expected_slices:
        raise AnalysisInputError("adaptive threshold lock does not cover exact run slices")
    return manifest, cells, manifest_sha, pair_sha


def extract_adaptive_run(
    layout: RunLayout,
    *,
    expected_manifest_sha256: str,
    bootstrap_iterations: int = _BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate and analyze one complete online deployment run, read-only."""

    manifest, cells, manifest_sha, pair_sha = _load_declared_run(
        layout, expected_manifest_sha256=expected_manifest_sha256
    )
    replay_tasks, compaction_config = _adaptive_replay_inputs(manifest, cells)
    output_root = layout.results / ADAPTIVE_RESULT_SUBDIR
    job_root = layout.results / ADAPTIVE_JOB_SUBDIR
    expected_names = {f"{cell.cell_id}.json" for cell in cells}
    for root, label in ((output_root, "output"), (job_root, "job")):
        if not root.is_dir() or root.is_symlink():
            raise AnalysisInputError(f"adaptive {label} directory is missing or linked")
        entries = tuple(root.iterdir())
        actual = {path.name for path in entries if path.is_file()}
        if (
            actual != expected_names
            or any(path.is_symlink() or not path.is_file() for path in entries)
        ):
            raise AnalysisInputError(
                f"adaptive {label} files do not exactly cover declared cells"
            )
    expected_event_names = {
        f"adaptive-{cell.cell_id}.jsonl" for cell in cells
    }
    actual_event_names = {
        path.name
        for path in layout.events.glob("adaptive-*.jsonl")
        if path.is_file()
    }
    if actual_event_names != expected_event_names:
        raise AnalysisInputError(
            "adaptive event logs do not exactly cover declared cells"
        )
    outputs: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        output_path = output_root / f"{cell.cell_id}.json"
        event_path = layout.events / f"adaptive-{cell.cell_id}.jsonl"
        job_path = job_root / f"{cell.cell_id}.json"
        if output_path.is_symlink() or event_path.is_symlink() or job_path.is_symlink():
            raise AnalysisInputError("adaptive run artifacts must not be symlinks")
        events = read_jsonl(event_path)
        if not events or not isinstance(events[0], Mapping):
            raise AnalysisInputError(f"adaptive event log is invalid: {cell.cell_id}")
        start = events[0]
        design = {
            key: value
            for key, value in start.items()
            if key not in {"event", "design_sha256"}
        }
        if (
            start.get("event") != "start"
            or sha256_json(design) != start.get("design_sha256")
            or start.get("run_id") != manifest.get("run_id")
            or start.get("manifest_sha256") != manifest_sha
            or start.get("pair_manifest_sha256") != pair_sha
            or start.get("deployment_mode") != ADAPTIVE_DEPLOYMENT_MODE
            or start.get("deployment_policy") != ADAPTIVE_POLICY
            or not isinstance(start.get("cell"), Mapping)
            or dict(start["cell"]) != cell.as_dict()
        ):
            raise AnalysisInputError(f"adaptive start receipt changed: {cell.cell_id}")
        _require_start_runtime_binding(start, manifest, cell_id=cell.cell_id)
        try:
            output = _validate_existing(
                output_file=output_path,
                event_file=event_path,
                start=start,
                task=replay_tasks[
                    (
                        cell.pair_key.domain,
                        cell.pair_key.task_id,
                        str(cell.pair_key.task_sha256),
                    )
                ],
                compaction_config=compaction_config,
            )
        except AdaptiveDeploymentError as exc:
            raise AnalysisInputError(f"adaptive output failed validation: {exc}") from exc
        job = read_json(job_path)
        if (
            not isinstance(job, Mapping)
            or job.get("cell_id") != cell.cell_id
            or job.get("state") != "complete"
            or job.get("output_sha256") != sha256_file(output_path)
            or job.get("success") != output.get("evaluation", {}).get("success")
            or job.get("accounting_sha256") != sha256_json(output.get("accounting"))
        ):
            raise AnalysisInputError(f"adaptive job receipt changed: {cell.cell_id}")
        outputs[cell.cell_id] = output

    try:
        base_rows = extract_adaptive_outcomes(cells, outputs)
    except AdaptiveDeploymentError as exc:
        raise AnalysisInputError(str(exc)) from exc
    attempts = _load_attempt_resources(layout)
    seen_attempts: set[str] = set()
    enriched = tuple(
        _resource_row(cell, outputs[cell.cell_id], base, attempts, seen_attempts)
        for cell, base in zip(cells, base_rows)
    )
    if seen_attempts != set(attempts):
        raise AnalysisInputError(
            "adaptive call attempts are missing from outputs or belong to undeclared work"
        )
    summaries, effects = summarize_adaptive_outcomes(
        enriched,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "adaptive_analysis_version": ADAPTIVE_ANALYSIS_VERSION,
        "artifact_type": ADAPTIVE_ANALYSIS_TYPE,
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": manifest_sha,
        "source_pair_manifest_sha256": pair_sha,
        "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
        "deployment_policy": ADAPTIVE_POLICY,
        "per_task_action_cap": PRIMARY_MAX_ACTIONS_PER_TASK,
        "statistical_unit": "source_task",
        "comparison_semantics": (
            "absolute outcomes use every declared treatment cell; operator effects are "
            "paired operator minus none on identical source tasks; each treatment cell "
            "is a separately generated trajectory"
        ),
        "resource_semantics": (
            "tokens, latency, and dollars include reconciled failed retries and the "
            "successful final task/observer attempt"
        ),
        "rows": enriched,
        "metric_summaries": [asdict(row) for row in summaries],
        "operator_effects": [asdict(row) for row in effects],
    }


def _safe_stem(value: str) -> str:
    return _SAFE_STEM.sub("-", value).strip("-") or "slice"


def write_adaptive_figures(
    analysis: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, ...]:
    """Write one success figure per model/benchmark from a validated analysis."""

    if (
        not isinstance(analysis, Mapping)
        or analysis.get("artifact_type") != ADAPTIVE_ANALYSIS_TYPE
        or analysis.get("deployment_mode") != ADAPTIVE_DEPLOYMENT_MODE
    ):
        raise AnalysisInputError("adaptive figure input has the wrong artifact type")
    rows = analysis.get("metric_summaries")
    if not isinstance(rows, list) or not rows:
        raise AnalysisInputError("adaptive figure input lacks metric summaries")
    grouped: dict[tuple[str, str], list[DeploymentBar]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("metric") != "success":
            continue
        try:
            bar = DeploymentBar(
                observation_class=str(row["observation_class"]),
                operator=str(row["operator"]),
                method=str(row["method"]),
                value=float(row["mean"]),
                n_tasks=int(row["n_tasks"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
            )
            key = (str(row["model"]), str(row["benchmark"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisInputError(f"adaptive success summary is invalid: {exc}") from exc
        grouped.setdefault(key, []).append(bar)
    if not grouped:
        raise AnalysisInputError("adaptive figure input has no success summaries")
    destination = Path(output_dir)
    written: list[Path] = []
    for (model, benchmark), bars in sorted(grouped.items()):
        path = destination / f"deployment-{_safe_stem(benchmark)}-{_safe_stem(model)}.svg"
        artifact = write_deployment_grouped_bars(
            bars,
            path,
            title=f"Online deployment: {model} on {benchmark}",
            y_label="Task success rate",
        )
        written.extend((artifact.svg_path, artifact.data_path))
    return tuple(written)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        "extract", help="validate and analyze a complete online deployment run"
    )
    extract.add_argument("--run-id", required=True)
    extract.add_argument("--manifest-sha256", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--figures")
    extract.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    extract.add_argument(
        "--bootstrap-iterations", type=int, default=_BOOTSTRAP_ITERATIONS
    )
    extract.add_argument("--bootstrap-seed", type=int, default=_BOOTSTRAP_SEED)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        analysis = extract_adaptive_run(
            RunLayout.for_run(args.artifacts, args.run_id),
            expected_manifest_sha256=args.manifest_sha256,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        atomic_write_json(args.output, analysis)
        if args.figures:
            write_adaptive_figures(analysis, args.figures)
        return 0
    except (
        AnalysisInputError,
        AdaptiveDeploymentError,
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
    "ADAPTIVE_ANALYSIS_TYPE",
    "ADAPTIVE_ANALYSIS_VERSION",
    "AdaptiveMetricSummary",
    "AdaptiveOperatorEffect",
    "extract_adaptive_run",
    "main",
    "observation_class",
    "parser",
    "summarize_adaptive_outcomes",
    "write_adaptive_figures",
]
