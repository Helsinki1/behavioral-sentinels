"""Fail-closed extraction and paper analysis for Experiment 12 runs.

The paired task trajectory is the statistical unit.  Active signals are scored
on the trajectories in which they were actually carried; passive signals and
non-adaptive baselines are scored on immutable clean trajectories.  This is an
intentional ecological comparison, not a claim that the two signal classes saw
the same counterfactual trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from experiments12.cli12 import DEFAULT_ARTIFACTS, REPOSITORY_ROOT
from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
)
from experiments12.core.schemas import CallAttemptRecord, CallStatus
from experiments12.figures12 import (
    write_observer_effect_forest,
    write_observer_metric_effect_forest,
    write_pr_curves,
)
from experiments12.manifest12 import RunLayout
from experiments12.metrics12 import (
    CheckpointScore,
    MetricInputError,
    ObservationTrace,
    PairedEffect,
    PairedMetricEffect,
    PredictionMetrics,
    TaskArmMeasurement,
    TaskOutcome,
    ThresholdSelection,
    grouped_prediction_metrics,
    paired_active_effects,
    paired_observer_effects,
    select_fixed_firing_rate_threshold,
)
from experiments12.pairing12 import JobCell
from experiments12.source_registry12 import SourceRegistryError, normalize_source_id
from experiments12.spec12 import Benchmark
from experiments12.passive_spec12 import (
    effective_passive_method_names,
    passive_monitor_spec_from_manifest,
)
from experiments12.validate12 import validate_run


ANALYSIS_VERSION = 2
THRESHOLD_ARTIFACT_VERSION = 1
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")


class AnalysisInputError(ValueError):
    """A frozen run is incomplete, mixed, or unsafe to analyze."""


@dataclass(frozen=True, slots=True)
class _AttemptResource:
    event_id: str
    purpose: str
    status: CallStatus
    input_tokens: int
    output_tokens: int
    elapsed_ms: int
    actual_cost_usd: Decimal
    cost_quality: str
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    reservation_id: str | None = None
    provider: str | None = None
    model: str | None = None
    attempt_number: int | None = None
    request_key: str | None = None


def _digest(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnalysisInputError(f"{context} must be a lowercase SHA256 digest")
    return value


def _manifest_analysis_config(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], Mapping[str, str] | None]:
    stage = manifest.get("stage")
    extra = manifest.get("extra_config")
    if not isinstance(extra, Mapping):
        raise AnalysisInputError("manifest extra_config must be an object")
    try:
        methods = effective_passive_method_names(
            passive_monitor_spec_from_manifest(manifest)
        )
    except ValueError as exc:
        raise AnalysisInputError(f"manifest passive monitor lock is invalid: {exc}") from exc

    lock_value = extra.get("analysis_lock")
    lock: Mapping[str, str] | None = None
    if lock_value is not None:
        if not isinstance(lock_value, Mapping) or set(lock_value) != {
            "threshold_artifact_sha256",
            "calibration_manifest_sha256",
        }:
            raise AnalysisInputError("manifest analysis_lock has an invalid schema")
        lock = {
            "threshold_artifact_sha256": _digest(
                lock_value.get("threshold_artifact_sha256"),
                context="analysis_lock threshold_artifact_sha256",
            ),
            "calibration_manifest_sha256": _digest(
                lock_value.get("calibration_manifest_sha256"),
                context="analysis_lock calibration_manifest_sha256",
            ),
        }
    if stage == "confirmatory" and lock is None:
        raise AnalysisInputError("confirmatory manifest lacks a frozen analysis_lock")
    if stage != "confirmatory" and lock is not None:
        raise AnalysisInputError("analysis_lock is permitted only for confirmatory runs")
    return methods, lock


def _identity(cell: JobCell) -> str:
    return f"{cell.pair_key.task_id}/r{cell.pair_key.replicate_id}"


def _source_task_id(cell: JobCell) -> str:
    pair_task_id = cell.pair_key.task_id
    if pair_task_id.count("::") != 1:
        raise AnalysisInputError(
            f"pair task ID is not canonical source::condition: {pair_task_id!r}"
        )
    source, condition = pair_task_id.split("::", 1)
    if not source or not condition:
        raise AnalysisInputError(
            f"pair task ID is not canonical source::condition: {pair_task_id!r}"
        )
    return source


def _load_cells(layout: RunLayout) -> tuple[JobCell, ...]:
    cells = tuple(JobCell.from_dict(row) for row in read_jsonl(layout.pairs))
    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise AnalysisInputError("pair manifest is empty or duplicates cells")
    return cells


def _validated_materialization(
    layout: RunLayout,
    *,
    expected_manifest_sha256: str,
) -> tuple[Mapping[str, Any], tuple[JobCell, ...], dict[str, Mapping[str, Any]]]:
    if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64:
        raise AnalysisInputError("analysis requires an externally recorded manifest SHA256")
    report = validate_run(
        layout,
        repository_root=REPOSITORY_ROOT,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if not report.primary_ready:
        codes = sorted({issue.code for issue in report.errors})
        raise AnalysisInputError("run failed validation: " + ", ".join(codes))
    manifest = read_json(layout.manifest)
    cells = _load_cells(layout)
    trajectories: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        path = layout.trajectories / f"{cell.cell_id}.json"
        value = read_json(path)
        if value.get("complete") is not True:
            raise AnalysisInputError(f"trajectory is incomplete: {cell.cell_id}")
        trajectories[cell.cell_id] = value
    return manifest, cells, trajectories


def _load_attempt_resources(layout: RunLayout) -> dict[str, _AttemptResource]:
    """Join append-only call attempts to reconciled ledger rows, read-only."""

    attempts: dict[str, CallAttemptRecord] = {}
    for path in sorted(layout.events.rglob("*.jsonl")):
        for value in read_jsonl(path):
            if not isinstance(value, Mapping) or not {
                "event_id",
                "reservation_id",
            }.issubset(value):
                continue
            try:
                attempt = CallAttemptRecord.from_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisInputError(f"invalid call attempt in {path.name}: {exc}") from exc
            if attempt.event_id in attempts:
                raise AnalysisInputError(f"duplicate call attempt event: {attempt.event_id}")
            attempts[attempt.event_id] = attempt
    if not attempts:
        raise AnalysisInputError("run contains no append-only call attempts")

    try:
        uri = layout.ledger.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = {
                str(row["reservation_id"]): row
                for row in connection.execute(
                    """
                    SELECT reservation_id, provider, purpose, request_key, state,
                           actual_micro_usd,
                           cost_quality, request_status, input_tokens, output_tokens,
                           cached_input_tokens, reasoning_tokens
                    FROM reservations
                    """
                ).fetchall()
            }
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisInputError(
            f"could not read reconciled call ledger: {type(exc).__name__}"
        ) from exc

    result: dict[str, _AttemptResource] = {}
    for event_id, attempt in attempts.items():
        row = rows.get(attempt.reservation_id)
        if row is None:
            raise AnalysisInputError(f"call attempt lacks ledger row: {event_id}")
        if (
            row["state"] != "reconciled"
            or row["provider"] != attempt.provider
            or row["purpose"] != attempt.purpose
            or row["request_status"] != attempt.status.value
        ):
            raise AnalysisInputError(f"call attempt disagrees with ledger: {event_id}")
        ledger_usage = (
            row["input_tokens"],
            row["output_tokens"],
            row["cached_input_tokens"],
            row["reasoning_tokens"],
        )
        attempt_usage = (
            attempt.usage.input_tokens,
            attempt.usage.output_tokens,
            attempt.usage.cached_input_tokens,
            attempt.usage.reasoning_tokens,
        )
        if ledger_usage != attempt_usage:
            raise AnalysisInputError(f"call token usage disagrees with ledger: {event_id}")
        actual_micro = row["actual_micro_usd"]
        quality = row["cost_quality"]
        if (
            isinstance(actual_micro, bool)
            or not isinstance(actual_micro, int)
            or actual_micro < 0
            or quality not in {"reported", "estimated", "upper_bound"}
            or attempt.elapsed_ms is None
            or attempt.estimated_cost_usd is None
        ):
            raise AnalysisInputError(f"call accounting is incomplete: {event_id}")
        expected_micro = int(
            (attempt.estimated_cost_usd * Decimal("1000000")).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        if expected_micro != actual_micro:
            raise AnalysisInputError(f"call cost disagrees with ledger: {event_id}")
        result[event_id] = _AttemptResource(
            event_id=event_id,
            purpose=attempt.purpose,
            status=attempt.status,
            input_tokens=attempt.usage.input_tokens,
            output_tokens=attempt.usage.output_tokens,
            elapsed_ms=attempt.elapsed_ms,
            actual_cost_usd=Decimal(actual_micro) / Decimal("1000000"),
            cost_quality=str(quality),
            cached_input_tokens=attempt.usage.cached_input_tokens,
            reasoning_tokens=attempt.usage.reasoning_tokens,
            reservation_id=attempt.reservation_id,
            provider=attempt.provider,
            model=attempt.model,
            attempt_number=attempt.attempt_number,
            request_key=row["request_key"],
        )
    return result


def task_outcomes(
    cells: Iterable[JobCell],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    operator: str = "none",
) -> tuple[TaskOutcome, ...]:
    """Extract one official end-task success value for every declared arm."""

    rows: list[TaskOutcome] = []
    for cell in cells:
        if cell.operator != operator:
            continue
        trajectory = trajectories[cell.cell_id]
        success = trajectory.get("evaluation", {}).get("success")
        if not isinstance(success, bool):
            raise AnalysisInputError(f"trajectory lacks binary official success: {cell.cell_id}")
        rows.append(
            TaskOutcome(
                model=cell.pair_key.model,
                benchmark=cell.pair_key.domain,
                task_id=_identity(cell),
                arm=cell.arm,
                outcome=float(success),
            )
        )
    if not rows:
        raise AnalysisInputError(f"run contains no operator={operator!r} outcomes")
    return tuple(rows)


_ACCOUNTING_KEYS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "elapsed_ms",
    "accounted_cost_usd",
)


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisInputError(f"{context} must be a non-negative integer")
    return value


def _nonnegative_decimal(value: Any, *, context: str) -> Decimal:
    if isinstance(value, bool):
        raise AnalysisInputError(f"{context} must be a finite non-negative decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AnalysisInputError(
            f"{context} must be a finite non-negative decimal"
        ) from exc
    if not result.is_finite() or result < 0:
        raise AnalysisInputError(f"{context} must be a finite non-negative decimal")
    return result


def _record_accounting(
    records: Any,
    *,
    context: str,
    seen_event_ids: set[str],
) -> tuple[dict[str, Any] | None, set[str]]:
    if not isinstance(records, list):
        raise AnalysisInputError(f"{context} records must be a list")
    if not records:
        return None, set()
    bucket: dict[str, Any] = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "elapsed_ms": 0,
        "accounted_cost_usd": Decimal("0"),
    }
    resolved_models: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise AnalysisInputError(f"{context} record {index} is not an object")
        call = record.get("call")
        if not isinstance(call, Mapping):
            raise AnalysisInputError(f"{context} record {index} lacks call accounting")
        event_ids = call.get("call_event_ids")
        if (
            not isinstance(event_ids, list)
            or not event_ids
            or any(not isinstance(item, str) or not item for item in event_ids)
            or len(event_ids) != len(set(event_ids))
        ):
            raise AnalysisInputError(f"{context} record {index} has invalid call IDs")
        reused = seen_event_ids.intersection(event_ids)
        if reused:
            raise AnalysisInputError(
                f"call event is reused within one task trajectory: {sorted(reused)[0]}"
            )
        seen_event_ids.update(event_ids)
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            raise AnalysisInputError(f"{context} record {index} lacks token usage")
        for key in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            bucket[key] += _nonnegative_int(
                usage.get(key), context=f"{context} record {index} {key}"
            )
        bucket["elapsed_ms"] += _nonnegative_int(
            call.get("elapsed_ms"), context=f"{context} record {index} elapsed_ms"
        )
        bucket["accounted_cost_usd"] += _nonnegative_decimal(
            call.get("accounted_cost_usd"),
            context=f"{context} record {index} accounted_cost_usd",
        )
        resolved = call.get("resolved_model_id")
        if not isinstance(resolved, str) or not resolved:
            raise AnalysisInputError(
                f"{context} record {index} lacks a resolved model identifier"
            )
        resolved_models.add(resolved)
        bucket["calls"] += 1
    return bucket, resolved_models


def _validate_rollup_bucket(
    value: Any,
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_ACCOUNTING_KEYS):
        raise AnalysisInputError(f"{context} accounting bucket has an invalid schema")
    for key in _ACCOUNTING_KEYS[:-1]:
        actual = _nonnegative_int(value.get(key), context=f"{context} {key}")
        if actual != expected[key]:
            raise AnalysisInputError(f"{context} {key} disagrees with call records")
    actual_cost = _nonnegative_decimal(
        value.get("accounted_cost_usd"), context=f"{context} accounted_cost_usd"
    )
    if actual_cost != expected["accounted_cost_usd"]:
        raise AnalysisInputError(
            f"{context} accounted_cost_usd disagrees with call records"
        )


def _attempt_category(
    records: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, _AttemptResource],
    *,
    expected_purpose: str,
    context: str,
) -> dict[str, Any]:
    result = {
        "tokens": 0,
        "elapsed_ms": 0,
        "actual_cost_usd": Decimal("0"),
    }
    for index, record in enumerate(records):
        aggregate = record["call"]
        nested = record.get("calls")
        if nested is None:
            materialized_calls = (aggregate,)
        else:
            if (
                not isinstance(nested, list)
                or not nested
                or any(not isinstance(item, Mapping) for item in nested)
            ):
                raise AnalysisInputError(
                    f"{context} record {index} has invalid nested calls"
                )
            materialized_calls = tuple(nested)
            flattened_ids = [
                event_id
                for call in materialized_calls
                for event_id in call.get("call_event_ids", ())
            ]
            if flattened_ids != aggregate.get("call_event_ids"):
                raise AnalysisInputError(
                    f"{context} record {index} nested call IDs disagree with aggregate"
                )
            aggregate_usage = aggregate.get("usage")
            if not isinstance(aggregate_usage, Mapping):
                raise AnalysisInputError(
                    f"{context} record {index} aggregate usage is invalid"
                )
            for key in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
            ):
                if sum(call.get("usage", {}).get(key, -1) for call in materialized_calls) != aggregate_usage.get(key):
                    raise AnalysisInputError(
                        f"{context} record {index} nested {key} disagrees with aggregate"
                    )
            if sum(call.get("elapsed_ms", -1) for call in materialized_calls) != aggregate.get(
                "elapsed_ms"
            ):
                raise AnalysisInputError(
                    f"{context} record {index} nested latency disagrees with aggregate"
                )
            try:
                nested_cost = sum(
                    (_nonnegative_decimal(call.get("accounted_cost_usd"), context=f"{context} record {index} nested cost") for call in materialized_calls),
                    Decimal("0"),
                )
                aggregate_cost = _nonnegative_decimal(
                    aggregate.get("accounted_cost_usd"),
                    context=f"{context} record {index} aggregate cost",
                )
            except (TypeError, ValueError) as exc:
                raise AnalysisInputError(
                    f"{context} record {index} nested cost is invalid"
                ) from exc
            if nested_cost != aggregate_cost:
                raise AnalysisInputError(
                    f"{context} record {index} nested cost disagrees with aggregate"
                )

        for call_index, call in enumerate(materialized_calls):
            event_ids = call["call_event_ids"]
            call_attempts: list[_AttemptResource] = []
            for event_id in event_ids:
                try:
                    attempt = attempts[event_id]
                except KeyError as exc:
                    raise AnalysisInputError(
                        f"{context} call attempt is absent: {event_id}"
                    ) from exc
                if attempt.purpose != expected_purpose:
                    raise AnalysisInputError(
                        f"{context} call attempt has wrong purpose: {event_id}"
                    )
                call_attempts.append(attempt)
            if (
                call_attempts[-1].status is not CallStatus.SUCCEEDED
                or any(item.status is not CallStatus.FAILED for item in call_attempts[:-1])
            ):
                raise AnalysisInputError(
                    f"{context} call {index}.{call_index} lacks "
                    "failed-retry*/successful-final ordering"
                )
            final = call_attempts[-1]
            usage = call["usage"]
            if (
                usage["input_tokens"],
                usage["output_tokens"],
            ) != (final.input_tokens, final.output_tokens):
                raise AnalysisInputError(
                    f"{context} successful-call usage disagrees with attempt log"
                )
            if call["elapsed_ms"] != sum(item.elapsed_ms for item in call_attempts):
                raise AnalysisInputError(f"{context} latency disagrees with attempt log")
            for attempt in call_attempts:
                result["tokens"] += attempt.input_tokens + attempt.output_tokens
                result["elapsed_ms"] += attempt.elapsed_ms
                result["actual_cost_usd"] += attempt.actual_cost_usd
    return result


def _trajectory_resources(
    trajectory: Mapping[str, Any],
    *,
    cell_id: str,
    attempt_resources: Mapping[str, _AttemptResource] | None = None,
) -> dict[str, Any]:
    seen_event_ids: set[str] = set()
    task, task_models = _record_accounting(
        trajectory.get("task_records"),
        context=f"{cell_id} task",
        seen_event_ids=seen_event_ids,
    )
    if task is None:
        raise AnalysisInputError(f"trajectory has no task calls: {cell_id}")
    observer, observer_models = _record_accounting(
        trajectory.get("probe_records"),
        context=f"{cell_id} observer",
        seen_event_ids=seen_event_ids,
    )
    expected_categories = {"agent": task}
    if observer is not None:
        expected_categories["active_monitor"] = observer
    accounting = trajectory.get("accounting")
    if not isinstance(accounting, Mapping) or set(accounting) != {
        "by_category",
        "resolved_model_ids",
    }:
        raise AnalysisInputError(f"trajectory has invalid accounting schema: {cell_id}")
    categories = accounting.get("by_category")
    if not isinstance(categories, Mapping) or set(categories) != set(expected_categories):
        raise AnalysisInputError(f"trajectory accounting categories disagree: {cell_id}")
    for name, expected in expected_categories.items():
        _validate_rollup_bucket(
            categories[name], expected, context=f"{cell_id} {name}"
        )
    expected_models = sorted(task_models | observer_models)
    if accounting.get("resolved_model_ids") != expected_models:
        raise AnalysisInputError(
            f"trajectory resolved model IDs disagree with call records: {cell_id}"
        )
    observer = observer or {
        "input_tokens": 0,
        "output_tokens": 0,
        "elapsed_ms": 0,
        "accounted_cost_usd": Decimal("0"),
    }
    task_tokens = task["input_tokens"] + task["output_tokens"]
    observer_tokens = observer["input_tokens"] + observer["output_tokens"]
    latency_ms = task["elapsed_ms"] + observer["elapsed_ms"]
    actual_cost = task["accounted_cost_usd"] + observer["accounted_cost_usd"]
    if attempt_resources is not None:
        exact_task = _attempt_category(
            trajectory["task_records"],
            attempt_resources,
            expected_purpose="agent_turn",
            context=f"{cell_id} task",
        )
        exact_observer = (
            _attempt_category(
                trajectory["probe_records"],
                attempt_resources,
                expected_purpose="active_probe",
                context=f"{cell_id} observer",
            )
            if trajectory["probe_records"]
            else {
                "tokens": 0,
                "elapsed_ms": 0,
                "actual_cost_usd": Decimal("0"),
            }
        )
        task_tokens = exact_task["tokens"]
        observer_tokens = exact_observer["tokens"]
        latency_ms = exact_task["elapsed_ms"] + exact_observer["elapsed_ms"]
        actual_cost = (
            exact_task["actual_cost_usd"] + exact_observer["actual_cost_usd"]
        )
    return {
        "task_tokens": task_tokens,
        "observer_tokens": observer_tokens,
        "total_tokens": task_tokens + observer_tokens,
        "latency_ms": latency_ms,
        "actual_cost_usd": float(actual_cost),
    }


def task_measurements(
    cells: Iterable[JobCell],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    operator: str = "none",
    attempt_resources: Mapping[str, _AttemptResource] | None = None,
) -> tuple[TaskArmMeasurement, ...]:
    """Extract one validated success/resource row per declared task arm."""

    rows: list[TaskArmMeasurement] = []
    for cell in cells:
        if cell.operator != operator:
            continue
        try:
            trajectory = trajectories[cell.cell_id]
        except KeyError as exc:
            raise AnalysisInputError(f"missing trajectory: {cell.cell_id}") from exc
        if trajectory.get("complete") is not True or trajectory.get("arm") != cell.arm:
            raise AnalysisInputError(f"trajectory identity/complete flag differs: {cell.cell_id}")
        success = trajectory.get("evaluation", {}).get("success")
        if not isinstance(success, bool):
            raise AnalysisInputError(f"trajectory lacks binary official success: {cell.cell_id}")
        resources = _trajectory_resources(
            trajectory,
            cell_id=cell.cell_id,
            attempt_resources=attempt_resources,
        )
        if cell.arm == "clean" and resources["observer_tokens"] != 0:
            raise AnalysisInputError(f"clean trajectory contains observer usage: {cell.cell_id}")
        rows.append(
            TaskArmMeasurement(
                model=cell.pair_key.model,
                benchmark=cell.pair_key.domain,
                task_id=_identity(cell),
                arm=cell.arm,
                success=float(success),
                **resources,
            )
        )
    if not rows:
        raise AnalysisInputError(f"run contains no operator={operator!r} measurements")
    return tuple(rows)


def observer_metric_effects(
    measurements: Sequence[TaskArmMeasurement],
    *,
    bootstrap_iterations: int = 2_000,
    seed: int = 12_012,
) -> dict[str, tuple[PairedMetricEffect, ...]]:
    """Return six task-paired active-minus-clean effects for every active arm."""

    arms = sorted({row.arm for row in measurements} - {"clean"})
    if not arms:
        raise AnalysisInputError("observer-effect run has no active arms")
    result: dict[str, tuple[PairedMetricEffect, ...]] = {}
    for active_arm in arms:
        relevant = tuple(
            row for row in measurements if row.arm in {"clean", active_arm}
        )
        try:
            result[active_arm] = paired_observer_effects(
                relevant,
                active_arm=active_arm,
                clean_arm="clean",
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        except MetricInputError as exc:
            raise AnalysisInputError(
                f"unpaired observer-effect arm {active_arm}: {exc}"
            ) from exc
    return result


def observer_effects(
    outcomes: Sequence[TaskOutcome],
    *,
    bootstrap_iterations: int = 2_000,
    seed: int = 12_012,
) -> dict[str, tuple[PairedEffect, ...]]:
    """Return separately paired active-minus-clean effects for every active arm."""

    arms = sorted({row.arm for row in outcomes} - {"clean"})
    if not arms:
        raise AnalysisInputError("observer-effect run has no active arms")
    result: dict[str, tuple[PairedEffect, ...]] = {}
    for active_arm in arms:
        relevant = tuple(row for row in outcomes if row.arm in {"clean", active_arm})
        try:
            result[active_arm] = paired_active_effects(
                relevant,
                active_arm=active_arm,
                clean_arm="clean",
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
        except MetricInputError as exc:
            raise AnalysisInputError(f"unpaired observer-effect arm {active_arm}: {exc}") from exc
    return result


def _event_checkpoint(trajectory: Mapping[str, Any]) -> int | None:
    success = trajectory.get("evaluation", {}).get("success")
    if not isinstance(success, bool):
        raise AnalysisInputError("trajectory lacks binary official success")
    records = trajectory.get("task_records")
    if not isinstance(records, list):
        raise AnalysisInputError("trajectory task_records must be a list")
    horizon = len(records)
    if horizon < 1:
        raise AnalysisInputError("trajectory has no task records")
    if trajectory.get("domain") == "bfcl_multi_turn":
        failure_turns: list[int] = []
        expected_keys = {
            "invalid_call_observed",
            "execution_failure_observed",
            "state_check_failure_observed",
            "state_check_available",
        }
        for expected_turn, record in enumerate(records, 1):
            if not isinstance(record, Mapping) or record.get("task_turn") != expected_turn:
                raise AnalysisInputError("BFCL task records are not contiguous")
            indicators = record.get("failure_indicators")
            if not isinstance(indicators, Mapping) or set(indicators) != expected_keys:
                raise AnalysisInputError("BFCL turn lacks exact official failure indicators")
            if any(not isinstance(indicators[key], bool) for key in expected_keys):
                raise AnalysisInputError("BFCL failure indicators must be boolean")
            if (
                indicators["state_check_failure_observed"]
                and not indicators["state_check_available"]
            ):
                raise AnalysisInputError(
                    "BFCL state-check failure is marked without an available check"
                )
            if any(
                indicators[key]
                for key in (
                    "invalid_call_observed",
                    "execution_failure_observed",
                    "state_check_failure_observed",
                )
            ):
                failure_turns.append(expected_turn)
        if failure_turns:
            return min(failure_turns)
    # Evolving Intent has only a verified final answer. BFCL tasks with no
    # earlier official action failure retain final official episode failure at
    # the horizon rather than inventing a turn-level cause.
    return None if success else horizon


def _active_trace(
    cell: JobCell,
    trajectory: Mapping[str, Any],
    *,
    split: str,
) -> ObservationTrace | None:
    records = trajectory.get("probe_records")
    if cell.arm == "clean":
        if records not in ([], None):
            raise AnalysisInputError("clean trajectory contains active probe records")
        return None
    if not isinstance(records, list) or not records:
        # A one-turn baseline condition has no actionable checkpoint and does
        # not contribute to signal-quality estimates.
        if len(trajectory.get("task_records", ())) == 1:
            return None
        raise AnalysisInputError(f"active trajectory lacks probe records: {cell.cell_id}")
    checkpoints: list[CheckpointScore] = []
    horizon = len(trajectory["task_records"])
    for record in records:
        grade = record.get("grade")
        turn = record.get("after_task_turn")
        if not isinstance(grade, Mapping) or not isinstance(grade.get("passed"), bool):
            raise AnalysisInputError(f"active probe has no deterministic grade: {cell.cell_id}")
        if isinstance(turn, bool) or not isinstance(turn, int):
            raise AnalysisInputError(f"active probe has invalid checkpoint: {cell.cell_id}")
        checkpoints.append(
            CheckpointScore(
                checkpoint=turn,
                score=0.0 if grade["passed"] else 1.0,
                actionable=turn < horizon,
            )
        )
    return ObservationTrace(
        model=cell.pair_key.model,
        benchmark=cell.pair_key.domain,
        method=cell.arm,
        task_id=_identity(cell),
        split=split,
        checkpoints=tuple(checkpoints),
        event_checkpoint=_event_checkpoint(trajectory),
        source_task_id=_source_task_id(cell),
    )


def _passive_method(record: Mapping[str, Any]) -> str:
    method = record.get("method")
    if not isinstance(method, str) or not method:
        raise AnalysisInputError("shadow record has no method")
    variant = record.get("variant")
    if variant is None:
        return method
    if not isinstance(variant, str) or not variant:
        raise AnalysisInputError("shadow record has an invalid variant")
    return f"{method}:{variant}"


def _shadow_traces(
    cell: JobCell,
    trajectory: Mapping[str, Any],
    shadow: Mapping[str, Any],
    *,
    split: str,
) -> tuple[ObservationTrace, ...]:
    if cell.arm != "clean":
        raise AnalysisInputError("passive traces must be sourced from clean cells")
    records = shadow.get("records")
    if not isinstance(records, list) or not records:
        raise AnalysisInputError(f"clean cell lacks passive records: {cell.cell_id}")
    declared_checkpoints = trajectory.get("checkpoint_turns")
    if (
        not isinstance(declared_checkpoints, list)
        or any(
            isinstance(turn, bool) or not isinstance(turn, int) or turn < 1
            for turn in declared_checkpoints
        )
        or declared_checkpoints != sorted(set(declared_checkpoints))
    ):
        raise AnalysisInputError(f"clean cell has invalid checkpoints: {cell.cell_id}")
    grouped: dict[str, list[CheckpointScore]] = {}
    base_methods: set[str] = set()
    seen_method_turns: set[tuple[str, int]] = set()
    horizon = len(trajectory["task_records"])
    for record in records:
        method = _passive_method(record)
        base_method = record.get("method")
        assert isinstance(base_method, str)
        base_methods.add(base_method)
        turn = record.get("checkpoint_turn")
        score = record.get("score")
        actionable_before = record.get("actionable_before_turn")
        if (
            isinstance(turn, bool)
            or not isinstance(turn, int)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or isinstance(actionable_before, bool)
            or not isinstance(actionable_before, int)
            or actionable_before != turn + 1
        ):
            raise AnalysisInputError(f"invalid passive checkpoint in {cell.cell_id}/{method}")
        identity = (method, turn)
        if identity in seen_method_turns:
            raise AnalysisInputError(
                f"duplicate passive checkpoint in {cell.cell_id}/{method}/t{turn}"
            )
        seen_method_turns.add(identity)
        grouped.setdefault(method, []).append(
            CheckpointScore(
                checkpoint=turn,
                score=float(score),
                actionable=actionable_before <= horizon,
            )
        )
    monitor_methods = shadow.get("monitor_methods")
    if (
        not isinstance(monitor_methods, list)
        or any(not isinstance(method, str) or not method for method in monitor_methods)
        or monitor_methods != sorted(base_methods)
    ):
        raise AnalysisInputError(f"shadow monitor_methods disagree: {cell.cell_id}")
    expected_turns = set(declared_checkpoints)
    for method, checkpoints in grouped.items():
        actual_turns = {item.checkpoint for item in checkpoints}
        if actual_turns != expected_turns:
            missing = sorted(expected_turns - actual_turns)
            extra = sorted(actual_turns - expected_turns)
            raise AnalysisInputError(
                f"passive checkpoint coverage differs in {cell.cell_id}/{method}; "
                f"missing={missing}, extra={extra}"
            )
    event = _event_checkpoint(trajectory)
    return tuple(
        ObservationTrace(
            model=cell.pair_key.model,
            benchmark=cell.pair_key.domain,
            method=method,
            task_id=_identity(cell),
            split=split,
            checkpoints=tuple(sorted(checkpoints, key=lambda item: item.checkpoint)),
            event_checkpoint=event,
            source_task_id=_source_task_id(cell),
        )
        for method, checkpoints in sorted(grouped.items())
    )


def signal_traces(
    layout: RunLayout,
    cells: Sequence[JobCell],
    trajectories: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    require_passive: bool = True,
    required_passive_methods: Sequence[str] | None = None,
) -> tuple[ObservationTrace, ...]:
    """Extract ecological active and passive task traces without pooling turns."""

    if not isinstance(split, str) or not split:
        raise AnalysisInputError("split must be a non-empty string")
    required: set[str] | None = None
    if required_passive_methods is not None:
        if (
            isinstance(required_passive_methods, (str, bytes))
            or not isinstance(required_passive_methods, Sequence)
            or not required_passive_methods
            or any(
                not isinstance(method, str) or not method
                for method in required_passive_methods
            )
            or len(required_passive_methods) != len(set(required_passive_methods))
        ):
            raise AnalysisInputError(
                "required_passive_methods must be unique non-empty method names"
            )
        required = set(required_passive_methods)
    traces: list[ObservationTrace] = []
    inferred_required: set[str] | None = None
    for cell in cells:
        if cell.operator != "none":
            continue
        trajectory = trajectories[cell.cell_id]
        active = _active_trace(cell, trajectory, split=split)
        if active is not None:
            traces.append(active)
        if cell.arm != "clean" or len(trajectory.get("task_records", ())) == 1:
            continue
        shadow_path = layout.shadow / f"{cell.cell_id}.json"
        if not shadow_path.is_file():
            if require_passive:
                raise AnalysisInputError(f"missing clean shadow output: {cell.cell_id}")
            continue
        passive = _shadow_traces(
            cell, trajectory, read_json(shadow_path), split=split
        )
        observed = {trace.method for trace in passive}
        expected = required if required is not None else inferred_required
        if expected is None:
            inferred_required = observed
        elif observed != expected:
            raise AnalysisInputError(
                f"passive method set differs in {cell.cell_id}; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        traces.extend(passive)
    if not traces:
        raise AnalysisInputError("run contains no actionable signal traces")
    if require_passive and inferred_required is None and required is None:
        raise AnalysisInputError("run contains no passive method set")
    return tuple(traces)


def extract_run(
    layout: RunLayout,
    *,
    expected_manifest_sha256: str,
    split: str,
    require_passive: bool = True,
) -> dict[str, Any]:
    manifest, cells, trajectories = _validated_materialization(
        layout, expected_manifest_sha256=expected_manifest_sha256
    )
    stage = manifest.get("stage")
    locked_splits = {"calibration": "calibration", "confirmatory": "confirmatory"}
    if stage in locked_splits and split != locked_splits[stage]:
        raise AnalysisInputError(f"stage={stage} must be extracted as split={stage}")
    if stage not in locked_splits and split in locked_splits.values():
        raise AnalysisInputError(
            f"stage={stage!r} cannot be mislabeled as a locked {split} split"
        )
    required_passive_methods, analysis_lock = _manifest_analysis_config(manifest)
    outcomes = task_outcomes(cells, trajectories)
    effects = observer_effects(outcomes)
    attempt_resources = _load_attempt_resources(layout)
    measurements = task_measurements(
        cells,
        trajectories,
        attempt_resources=attempt_resources,
    )
    metric_effects = observer_metric_effects(measurements)
    traces = signal_traces(
        layout,
        cells,
        trajectories,
        split=split,
        require_passive=require_passive,
        required_passive_methods=required_passive_methods,
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "run_id": manifest["run_id"],
        "stage": manifest["stage"],
        "manifest_sha256": expected_manifest_sha256,
        "split": split,
        "analysis_lock": analysis_lock,
        "required_passive_methods": (
            None
            if required_passive_methods is None
            else list(required_passive_methods)
        ),
        "statistical_unit": "task_trajectory",
        "event_semantics": (
            "BFCL uses the earliest recorded invalid-call, execution-failure, or "
            "state-check-failure turn; otherwise official episode failure is placed "
            "at the final horizon; Evolving Intent is final-only"
        ),
        "comparison_semantics": (
            "active signals use carried active trajectories; passive and baseline "
            "signals use zero-carry clean trajectories"
        ),
        "observer_effect_semantics": (
            "all effects are active minus clean on identical task trajectories; "
            "confidence intervals use a paired task bootstrap"
        ),
        "resource_semantics": (
            "task and observer tokens are reconciled attempt-level input plus output "
            "tokens; latency and recorded cost include failed retries and successful "
            "task/active-probe attempts; cost quality remains auditable in the ledger"
        ),
        "outcomes": [asdict(row) for row in outcomes],
        "task_measurements": [asdict(row) for row in measurements],
        "observer_effects": {
            arm: [asdict(row) for row in rows] for arm, rows in effects.items()
        },
        "observer_effect_table": [
            asdict(row)
            for arm in sorted(metric_effects)
            for row in metric_effects[arm]
        ],
        "signal_traces": [asdict(row) for row in traces],
    }


def _trace_from_dict(value: Mapping[str, Any]) -> ObservationTrace:
    return ObservationTrace(
        model=str(value["model"]),
        benchmark=str(value["benchmark"]),
        method=str(value["method"]),
        task_id=str(value["task_id"]),
        split=str(value["split"]),
        checkpoints=tuple(
            CheckpointScore(
                checkpoint=int(item["checkpoint"]),
                score=float(item["score"]),
                actionable=bool(item["actionable"]),
            )
            for item in value["checkpoints"]
        ),
        event_checkpoint=(
            None if value.get("event_checkpoint") is None else int(value["event_checkpoint"])
        ),
        complete=bool(value.get("complete", True)),
        source_task_id=(
            None
            if value.get("source_task_id") is None
            else str(value["source_task_id"])
        ),
    )


def calibrate_thresholds(
    traces: Iterable[ObservationTrace],
    *,
    target_firing_rate: float,
) -> tuple[ThresholdSelection, ...]:
    groups: dict[tuple[str, str, str], list[ObservationTrace]] = {}
    task_ids: set[tuple[str, str, str, str]] = set()
    for trace in traces:
        if trace.split != "calibration":
            raise AnalysisInputError("thresholds may be fit only on calibration traces")
        identity = (trace.model, trace.benchmark, trace.method, trace.task_id)
        if identity in task_ids:
            raise AnalysisInputError(f"duplicate calibration trace: {identity}")
        task_ids.add(identity)
        groups.setdefault(identity[:3], []).append(trace)
    if not groups:
        raise AnalysisInputError("calibration traces are empty")
    return tuple(
        select_fixed_firing_rate_threshold(group, target_firing_rate=target_firing_rate)
        for _key, group in sorted(groups.items())
    )


def _source_task_keys(
    traces: Sequence[ObservationTrace], *, context: str
) -> tuple[tuple[str, str], ...]:
    keys: set[tuple[str, str]] = set()
    for trace in traces:
        if trace.source_task_id is None:
            raise AnalysisInputError(
                f"{context} trace lacks canonical source_task_id: "
                f"{trace.model}/{trace.benchmark}/{trace.task_id}"
            )
        raw_source = trace.source_task_id.split("::", 1)[0]
        try:
            source = normalize_source_id(trace.benchmark, trace.source_task_id)
        except SourceRegistryError as exc:
            if (
                trace.benchmark == Benchmark.EVOLVING_GSM8K.value
                and (
                    raw_source.isdecimal()
                    or raw_source.startswith("extracted-gsm8k-test-")
                )
            ):
                raise AnalysisInputError(
                    f"{context} trace has invalid Evolving source_task_id"
                ) from exc
            source = raw_source
        keys.add((trace.benchmark, source))
    if not keys:
        raise AnalysisInputError(f"{context} source-task set is empty")
    return tuple(sorted(keys))


def require_source_task_disjointness(
    confirmatory_traces: Sequence[ObservationTrace],
    calibration_source_tasks: Sequence[tuple[str, str]],
) -> None:
    """Reject reuse of a source task across models, conditions, or replicates."""

    normalized_calibration: list[tuple[str, str]] = []
    for benchmark, source_task_id in calibration_source_tasks:
        raw_source = source_task_id.split("::", 1)[0]
        try:
            source = normalize_source_id(benchmark, source_task_id)
        except SourceRegistryError as exc:
            if (
                benchmark == Benchmark.EVOLVING_GSM8K.value
                and (
                    raw_source.isdecimal()
                    or raw_source.startswith("extracted-gsm8k-test-")
                )
            ):
                raise AnalysisInputError(
                    "calibration lock has an invalid Evolving source_task_id"
                ) from exc
            source = raw_source
        normalized_calibration.append((benchmark, source))
    calibration = set(normalized_calibration)
    if len(calibration) != len(normalized_calibration):
        raise AnalysisInputError("calibration source-task lock contains duplicates")
    overlap = set(
        _source_task_keys(confirmatory_traces, context="confirmatory")
    ).intersection(calibration)
    if overlap:
        raise AnalysisInputError(
            "calibration and confirmatory source tasks overlap globally: "
            + ", ".join(
                f"{benchmark}/{task}" for benchmark, task in sorted(overlap)[:5]
            )
        )


def _method_slices(
    traces: Sequence[ObservationTrace],
) -> tuple[tuple[str, str, str], ...]:
    result = tuple(
        sorted({(trace.model, trace.benchmark, trace.method) for trace in traces})
    )
    if not result:
        raise AnalysisInputError("method slice set is empty")
    return result


def make_threshold_artifact(
    source: Mapping[str, Any],
    traces: Sequence[ObservationTrace],
    thresholds: Sequence[ThresholdSelection],
    *,
    target_firing_rate: float,
    source_extract_sha256: str,
) -> dict[str, Any]:
    """Build the immutable calibration receipt later bound into a test manifest."""

    if source.get("stage") != "calibration" or source.get("split") != "calibration":
        raise AnalysisInputError("threshold artifact source must be a calibration extract")
    _digest(source.get("manifest_sha256"), context="calibration manifest SHA256")
    _digest(source_extract_sha256, context="calibration extract SHA256")
    required_passive = source.get("required_passive_methods")
    if (
        not isinstance(required_passive, list)
        or not required_passive
        or required_passive != sorted(set(required_passive))
        or any(not isinstance(item, str) or not item for item in required_passive)
    ):
        raise AnalysisInputError(
            "calibration extract lacks exact required_passive_methods"
        )
    materialized = tuple(traces)
    required_slices = _method_slices(materialized)
    threshold_keys = tuple(
        sorted((row.model, row.benchmark, row.method) for row in thresholds)
    )
    if threshold_keys != required_slices:
        raise AnalysisInputError("calibration thresholds do not exactly cover method slices")
    target = float(target_firing_rate)
    if any(row.target_firing_rate != target for row in thresholds):
        raise AnalysisInputError("calibration thresholds do not share the declared target rate")
    return {
        "threshold_artifact_version": THRESHOLD_ARTIFACT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "artifact_type": "locked_fixed_rate_thresholds",
        "source_run_id": source["run_id"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_extract_sha256": source_extract_sha256,
        "target_firing_rate": target,
        "calibration_source_tasks": [
            {"benchmark": benchmark, "source_task_id": source_task_id}
            for benchmark, source_task_id in _source_task_keys(
                materialized, context="calibration"
            )
        ],
        "required_passive_methods": list(required_passive),
        "required_method_slices": [
            {"model": model, "benchmark": benchmark, "method": method}
            for model, benchmark, method in required_slices
        ],
        "thresholds": [asdict(row) for row in thresholds],
    }


def load_threshold_artifact(
    value: Mapping[str, Any],
) -> tuple[
    tuple[ThresholdSelection, ...],
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    """Strictly parse a locked calibration artifact; unknown fields fail."""

    expected = {
        "threshold_artifact_version",
        "analysis_version",
        "artifact_type",
        "source_run_id",
        "source_manifest_sha256",
        "source_extract_sha256",
        "target_firing_rate",
        "calibration_source_tasks",
        "required_passive_methods",
        "required_method_slices",
        "thresholds",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AnalysisInputError("threshold artifact has missing or unexpected fields")
    if (
        value.get("threshold_artifact_version") != THRESHOLD_ARTIFACT_VERSION
        or value.get("analysis_version") != ANALYSIS_VERSION
        or value.get("artifact_type") != "locked_fixed_rate_thresholds"
    ):
        raise AnalysisInputError("threshold artifact version/type is unsupported")
    if not isinstance(value.get("source_run_id"), str) or not value["source_run_id"]:
        raise AnalysisInputError("threshold artifact source_run_id is invalid")
    _digest(value.get("source_manifest_sha256"), context="threshold source manifest")
    _digest(value.get("source_extract_sha256"), context="threshold source extract")
    target = value.get("target_firing_rate")
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not 0 <= float(target) <= 1
    ):
        raise AnalysisInputError("threshold target_firing_rate is invalid")

    passive = value.get("required_passive_methods")
    if (
        not isinstance(passive, list)
        or not passive
        or passive != sorted(set(passive))
        or any(not isinstance(item, str) or not item for item in passive)
    ):
        raise AnalysisInputError("threshold required_passive_methods is invalid")

    slice_rows = value.get("required_method_slices")
    if not isinstance(slice_rows, list) or not slice_rows:
        raise AnalysisInputError("threshold required_method_slices is empty")
    slices: list[tuple[str, str, str]] = []
    for row in slice_rows:
        if not isinstance(row, Mapping) or set(row) != {"model", "benchmark", "method"}:
            raise AnalysisInputError("threshold method slice has an invalid schema")
        item = (row["model"], row["benchmark"], row["method"])
        if any(not isinstance(part, str) or not part for part in item):
            raise AnalysisInputError("threshold method slice contains an invalid name")
        slices.append(item)
    if slices != sorted(set(slices)):
        raise AnalysisInputError("threshold method slices must be sorted and unique")

    source_rows = value.get("calibration_source_tasks")
    if not isinstance(source_rows, list) or not source_rows:
        raise AnalysisInputError("threshold calibration_source_tasks is empty")
    source_tasks: list[tuple[str, str]] = []
    for row in source_rows:
        if not isinstance(row, Mapping) or set(row) != {"benchmark", "source_task_id"}:
            raise AnalysisInputError("calibration source task has an invalid schema")
        item = (row["benchmark"], row["source_task_id"])
        if any(not isinstance(part, str) or not part for part in item):
            raise AnalysisInputError("calibration source task contains an invalid name")
        source_tasks.append(item)
    if source_tasks != sorted(set(source_tasks)):
        raise AnalysisInputError("calibration source tasks must be sorted and unique")

    rows = value.get("thresholds")
    if not isinstance(rows, list) or not rows:
        raise AnalysisInputError("threshold list is empty")
    try:
        thresholds = tuple(ThresholdSelection(**row) for row in rows)
    except (MetricInputError, TypeError, KeyError) as exc:
        raise AnalysisInputError(f"threshold selection is invalid: {exc}") from exc
    threshold_keys = tuple(
        sorted((row.model, row.benchmark, row.method) for row in thresholds)
    )
    if threshold_keys != tuple(slices):
        raise AnalysisInputError("threshold rows do not exactly match required method slices")
    if any(row.target_firing_rate != float(target) for row in thresholds):
        raise AnalysisInputError("threshold rows disagree on target_firing_rate")
    return thresholds, tuple(slices), tuple(source_tasks), tuple(passive)


def verify_threshold_binding(
    confirmatory_extract: Mapping[str, Any],
    threshold_artifact: Mapping[str, Any],
    *,
    threshold_artifact_sha256: str,
) -> tuple[
    tuple[ThresholdSelection, ...],
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """Verify both manifest-bound hashes and the frozen passive method set."""

    thresholds, required_slices, calibration_tasks, required_passive = (
        load_threshold_artifact(threshold_artifact)
    )
    lock = confirmatory_extract.get("analysis_lock")
    if not isinstance(lock, Mapping) or set(lock) != {
        "threshold_artifact_sha256",
        "calibration_manifest_sha256",
    }:
        raise AnalysisInputError("confirmatory extract lacks its manifest analysis_lock")
    if _digest(
        threshold_artifact_sha256, context="threshold artifact SHA256"
    ) != lock.get("threshold_artifact_sha256"):
        raise AnalysisInputError(
            "threshold artifact SHA256 differs from the confirmatory manifest lock"
        )
    if threshold_artifact["source_manifest_sha256"] != lock.get(
        "calibration_manifest_sha256"
    ):
        raise AnalysisInputError(
            "calibration manifest SHA256 differs from the confirmatory manifest lock"
        )
    if confirmatory_extract.get("required_passive_methods") != list(required_passive):
        raise AnalysisInputError(
            "confirmatory passive method set differs from the calibration lock"
        )
    return thresholds, required_slices, calibration_tasks


def score_locked(
    traces: Iterable[ObservationTrace],
    thresholds: Sequence[ThresholdSelection],
    *,
    required_method_slices: Sequence[tuple[str, str, str]] | None = None,
) -> tuple[PredictionMetrics, ...]:
    materialized = tuple(traces)
    if not materialized or any(trace.split != "confirmatory" for trace in materialized):
        raise AnalysisInputError("locked reporting requires confirmatory traces only")
    mapping = {(row.model, row.benchmark, row.method): row for row in thresholds}
    if len(mapping) != len(thresholds):
        raise AnalysisInputError("threshold file contains duplicate method slices")
    observed = {
        (trace.model, trace.benchmark, trace.method) for trace in materialized
    }
    required = observed if required_method_slices is None else set(required_method_slices)
    if required_method_slices is not None and len(required) != len(required_method_slices):
        raise AnalysisInputError("required method slice set contains duplicates")
    if observed != required:
        raise AnalysisInputError(
            "confirmatory method slices differ from the frozen required set; "
            f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
        )
    if set(mapping) != required:
        raise AnalysisInputError(
            "threshold method slices differ from the frozen required set; "
            f"missing={sorted(required - set(mapping))}, extra={sorted(set(mapping) - required)}"
        )
    try:
        return grouped_prediction_metrics(materialized, locked_thresholds=mapping)
    except MetricInputError as exc:
        raise AnalysisInputError(str(exc)) from exc


def _effects_from_extract(value: Mapping[str, Any]) -> dict[str, tuple[PairedEffect, ...]]:
    result: dict[str, tuple[PairedEffect, ...]] = {}
    for arm, rows in value.get("observer_effects", {}).items():
        result[str(arm)] = tuple(PairedEffect(**row) for row in rows)
    return result


def _metric_effects_from_extract(
    value: Mapping[str, Any],
) -> tuple[PairedMetricEffect, ...]:
    rows = value.get("observer_effect_table")
    if not isinstance(rows, list) or not rows:
        raise AnalysisInputError("extract lacks observer_effect_table")
    try:
        effects = tuple(PairedMetricEffect(**row) for row in rows)
    except (MetricInputError, TypeError) as exc:
        raise AnalysisInputError(f"observer_effect_table is invalid: {exc}") from exc
    identities = {
        (row.active_arm, row.model, row.benchmark, row.metric) for row in effects
    }
    if len(identities) != len(effects):
        raise AnalysisInputError("observer_effect_table contains duplicate slices")
    return effects


def _safe_stem(value: str) -> str:
    return _SAFE_STEM.sub("-", value).strip("-") or "slice"


def write_observer_figures(extracted: Mapping[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    for arm, effects in _effects_from_extract(extracted).items():
        write_observer_effect_forest(
            effects,
            destination / f"observer-effect-{_safe_stem(arm)}.svg",
            title=f"Observer effect: {arm}",
        )
    resource_groups: dict[tuple[str, str], list[PairedMetricEffect]] = {}
    for effect in _metric_effects_from_extract(extracted):
        if effect.metric == "success":
            continue
        resource_groups.setdefault((effect.active_arm, effect.metric), []).append(effect)
    for (arm, metric), effects in sorted(resource_groups.items()):
        write_observer_metric_effect_forest(
            effects,
            destination
            / f"observer-effect-{_safe_stem(arm)}-{_safe_stem(metric)}.svg",
            title=f"Observer resource effect: {arm} / {metric.replace('_', ' ')}",
        )


def write_signal_figures(
    summaries: Sequence[PredictionMetrics], output_dir: str | Path
) -> None:
    destination = Path(output_dir)
    groups: dict[tuple[str, str], list[PredictionMetrics]] = {}
    for summary in summaries:
        groups.setdefault((summary.model, summary.benchmark), []).append(summary)
    for (model, benchmark), rows in sorted(groups.items()):
        stem = _safe_stem(f"{benchmark}-{model}")
        write_pr_curves(
            rows,
            destination / f"signal-pr-{stem}.svg",
            title=f"Signal quality: {model} on {benchmark}",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--run-id", required=True)
    extract.add_argument("--manifest-sha256", required=True)
    extract.add_argument("--split", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--figures")
    extract.add_argument("--allow-missing-passive", action="store_true")
    extract.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--target-firing-rate", required=True, type=float)
    calibrate.add_argument("--output", required=True)

    score = commands.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--thresholds", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--figures")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "extract":
            value = extract_run(
                RunLayout.for_run(args.artifacts, args.run_id),
                expected_manifest_sha256=args.manifest_sha256,
                split=args.split,
                require_passive=not args.allow_missing_passive,
            )
            atomic_write_json(args.output, value)
            if args.figures:
                write_observer_figures(value, args.figures)
            return 0
        source = read_json(args.input)
        traces = tuple(_trace_from_dict(row) for row in source["signal_traces"])
        if args.command == "calibrate":
            thresholds = calibrate_thresholds(
                traces, target_firing_rate=args.target_firing_rate
            )
            artifact = make_threshold_artifact(
                source,
                traces,
                thresholds,
                target_firing_rate=args.target_firing_rate,
                source_extract_sha256=sha256_file(args.input),
            )
            atomic_write_json(args.output, artifact)
            return 0
        locked = read_json(args.thresholds)
        thresholds, required_slices, calibration_tasks = verify_threshold_binding(
            source,
            locked,
            threshold_artifact_sha256=sha256_file(args.thresholds),
        )
        require_source_task_disjointness(traces, calibration_tasks)
        summaries = score_locked(
            traces,
            thresholds,
            required_method_slices=required_slices,
        )
        atomic_write_json(
            args.output,
            {
                "analysis_version": ANALYSIS_VERSION,
                "source_run_id": source["run_id"],
                "source_manifest_sha256": source["manifest_sha256"],
                "threshold_source_run_id": locked["source_run_id"],
                "threshold_source_manifest_sha256": locked["source_manifest_sha256"],
                "metrics": [asdict(row) for row in summaries],
            },
        )
        if args.figures:
            write_signal_figures(summaries, args.figures)
        return 0
    except (AnalysisInputError, FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalysisInputError",
    "calibrate_thresholds",
    "extract_run",
    "load_threshold_artifact",
    "make_threshold_artifact",
    "observer_metric_effects",
    "observer_effects",
    "require_source_task_disjointness",
    "score_locked",
    "signal_traces",
    "task_measurements",
    "task_outcomes",
    "verify_threshold_binding",
    "write_observer_figures",
    "write_signal_figures",
]
