"""Task-clustered metrics for Experiment 12.

The statistical unit is always a task trajectory.  Checkpoints are retained
inside :class:`ObservationTrace` only to determine whether a task ever fires
and, if so, whether the first firing is early enough to act on.  They are never
flattened into independent rows for precision, recall, calibration, or
bootstrapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from statistics import fmean, median
from typing import Iterable, Mapping, Sequence


class MetricInputError(ValueError):
    """Inputs are incomplete, duplicated, mixed, or otherwise unsafe to score."""


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MetricInputError(f"{name} must be a non-empty string")


def _unit_interval(name: str, value: float | int) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetricInputError(f"{name} must be finite and in [0, 1]")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise MetricInputError(f"{name} must be finite and in [0, 1]")
    return number


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MetricInputError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """One complete task-level outcome for one experimental arm."""

    model: str
    benchmark: str
    task_id: str
    arm: str
    outcome: float
    complete: bool = True

    def __post_init__(self) -> None:
        for name in ("model", "benchmark", "task_id", "arm"):
            _nonempty(name, getattr(self, name))
        object.__setattr__(self, "outcome", _unit_interval("outcome", self.outcome))
        if not isinstance(self.complete, bool):
            raise MetricInputError("complete must be boolean")


@dataclass(frozen=True, slots=True)
class TaskArmMeasurement:
    """Success and recorded resource use for one complete task/arm trajectory.

    ``task_tokens`` and ``observer_tokens`` count reconciled attempt-level input
    plus output tokens, including retries. Cached-input and reasoning-token
    fields are provider subtotals and therefore are not added again.
    ``actual_cost_usd`` sums the ledger's reconciled amounts, not the pre-run
    reservation; each reservation's reported/estimated/upper-bound quality
    remains available in the ledger.
    """

    model: str
    benchmark: str
    task_id: str
    arm: str
    success: float
    task_tokens: int
    observer_tokens: int
    total_tokens: int
    latency_ms: int
    actual_cost_usd: float
    complete: bool = True

    def __post_init__(self) -> None:
        for name in ("model", "benchmark", "task_id", "arm"):
            _nonempty(name, getattr(self, name))
        object.__setattr__(self, "success", _unit_interval("success", self.success))
        for name in ("task_tokens", "observer_tokens", "total_tokens", "latency_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetricInputError(f"{name} must be a non-negative integer")
        if self.total_tokens != self.task_tokens + self.observer_tokens:
            raise MetricInputError("total_tokens must equal task_tokens + observer_tokens")
        if (
            isinstance(self.actual_cost_usd, bool)
            or not isinstance(self.actual_cost_usd, (int, float))
            or not math.isfinite(self.actual_cost_usd)
            or self.actual_cost_usd < 0
        ):
            raise MetricInputError("actual_cost_usd must be finite and non-negative")
        object.__setattr__(self, "actual_cost_usd", float(self.actual_cost_usd))
        if not isinstance(self.complete, bool):
            raise MetricInputError("complete must be boolean")


@dataclass(frozen=True, slots=True)
class PairedEffect:
    model: str
    benchmark: str
    active_arm: str
    clean_arm: str
    n_tasks: int
    clean_mean: float
    active_mean: float
    effect: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_iterations: int
    bootstrap_seed: int
    bootstrap_unit: str = "task"


@dataclass(frozen=True, slots=True)
class PairedMetricEffect:
    """One table-ready active-minus-clean contrast on paired task trajectories."""

    model: str
    benchmark: str
    active_arm: str
    clean_arm: str
    metric: str
    unit: str
    favorable_direction: str
    n_tasks: int
    clean_mean: float
    active_mean: float
    effect: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_iterations: int
    bootstrap_seed: int
    effect_definition: str = "active_minus_clean"
    bootstrap_unit: str = "task"

    def __post_init__(self) -> None:
        for name in (
            "model",
            "benchmark",
            "active_arm",
            "clean_arm",
            "metric",
            "unit",
        ):
            _nonempty(name, getattr(self, name))
        if self.active_arm == self.clean_arm:
            raise MetricInputError("active_arm and clean_arm must differ")
        if self.favorable_direction not in {"higher", "lower"}:
            raise MetricInputError("favorable_direction must be 'higher' or 'lower'")
        expected_metadata = {
            metric: (unit, direction)
            for metric, unit, direction in _OBSERVER_METRICS
        }
        if expected_metadata.get(self.metric) != (
            self.unit,
            self.favorable_direction,
        ):
            raise MetricInputError("metric unit/direction differs from the frozen schema")
        _positive_int("n_tasks", self.n_tasks)
        _positive_int("bootstrap_iterations", self.bootstrap_iterations)
        for name in ("clean_mean", "active_mean", "effect", "ci_low", "ci_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise MetricInputError(f"{name} must be finite")
        if self.ci_low > self.ci_high:
            raise MetricInputError("ci_low must not exceed ci_high")
        if self.metric == "success":
            _unit_interval("clean_mean", self.clean_mean)
            _unit_interval("active_mean", self.active_mean)
        elif self.clean_mean < 0 or self.active_mean < 0:
            raise MetricInputError("resource means must be non-negative")
        if not math.isclose(
            self.active_mean - self.clean_mean,
            self.effect,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise MetricInputError("effect must equal active_mean - clean_mean")
        if not isinstance(self.confidence, (int, float)) or not 0 < self.confidence < 1:
            raise MetricInputError("confidence must be strictly between 0 and 1")
        if isinstance(self.bootstrap_seed, bool) or not isinstance(self.bootstrap_seed, int):
            raise MetricInputError("bootstrap_seed must be an integer")
        if self.effect_definition != "active_minus_clean":
            raise MetricInputError("effect_definition must be active_minus_clean")
        if self.bootstrap_unit != "task":
            raise MetricInputError("bootstrap_unit must be task")


_OBSERVER_METRICS: tuple[tuple[str, str, str], ...] = (
    ("success", "proportion", "higher"),
    ("task_tokens", "tokens", "lower"),
    ("observer_tokens", "tokens", "lower"),
    ("total_tokens", "tokens", "lower"),
    ("latency_ms", "milliseconds", "lower"),
    ("actual_cost_usd", "USD", "lower"),
)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise MetricInputError("cannot take a quantile of no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_index(
    seed: int,
    model: str,
    benchmark: str,
    iteration: int,
    draw: int,
    population: int,
) -> int:
    material = f"exp12/task-bootstrap/v1\0{seed}\0{model}\0{benchmark}\0{iteration}\0{draw}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % population


def paired_active_effects(
    outcomes: Iterable[TaskOutcome],
    *,
    active_arm: str,
    clean_arm: str,
    bootstrap_iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 12_012,
) -> tuple[PairedEffect, ...]:
    """Compute active-minus-clean effects with a paired *task* bootstrap.

    Pairing is fail-closed: an incomplete record, duplicate arm record, missing
    partner, or unexpected arm raises instead of silently changing the sample.
    Each bootstrap draw samples paired task differences, never turns.
    """

    _nonempty("active_arm", active_arm)
    _nonempty("clean_arm", clean_arm)
    if active_arm == clean_arm:
        raise MetricInputError("active_arm and clean_arm must differ")
    _positive_int("bootstrap_iterations", bootstrap_iterations)
    if not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise MetricInputError("confidence must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MetricInputError("seed must be an integer")

    groups: dict[tuple[str, str], dict[str, dict[str, TaskOutcome]]] = {}
    count = 0
    for record in outcomes:
        count += 1
        if not isinstance(record, TaskOutcome):
            raise MetricInputError("all outcomes must be TaskOutcome records")
        if not record.complete:
            raise MetricInputError(
                f"incomplete paired outcome: {record.model}/{record.benchmark}/{record.task_id}"
            )
        if record.arm not in {active_arm, clean_arm}:
            raise MetricInputError(f"unexpected arm in paired input: {record.arm!r}")
        task = groups.setdefault((record.model, record.benchmark), {}).setdefault(
            record.task_id, {}
        )
        if record.arm in task:
            raise MetricInputError(
                f"duplicate paired outcome: {record.model}/{record.benchmark}/"
                f"{record.task_id}/{record.arm}"
            )
        task[record.arm] = record
    if count == 0:
        raise MetricInputError("paired outcomes are empty")

    results: list[PairedEffect] = []
    for (model, benchmark), tasks in sorted(groups.items()):
        differences: list[float] = []
        clean_values: list[float] = []
        active_values: list[float] = []
        for task_id in sorted(tasks):
            arms = tasks[task_id]
            missing = {active_arm, clean_arm}.difference(arms)
            if missing:
                raise MetricInputError(
                    f"unpaired task {model}/{benchmark}/{task_id}; missing {sorted(missing)!r}"
                )
            clean = arms[clean_arm].outcome
            active = arms[active_arm].outcome
            clean_values.append(clean)
            active_values.append(active)
            differences.append(active - clean)

        bootstrap: list[float] = []
        n_tasks = len(differences)
        for iteration in range(bootstrap_iterations):
            sampled = [
                differences[
                    _bootstrap_index(
                        seed, model, benchmark, iteration, draw, n_tasks
                    )
                ]
                for draw in range(n_tasks)
            ]
            bootstrap.append(fmean(sampled))
        bootstrap.sort()
        tail = (1.0 - float(confidence)) / 2.0
        results.append(
            PairedEffect(
                model=model,
                benchmark=benchmark,
                active_arm=active_arm,
                clean_arm=clean_arm,
                n_tasks=n_tasks,
                clean_mean=fmean(clean_values),
                active_mean=fmean(active_values),
                effect=fmean(differences),
                ci_low=_quantile(bootstrap, tail),
                ci_high=_quantile(bootstrap, 1.0 - tail),
                confidence=float(confidence),
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=seed,
            )
        )
    return tuple(results)


def paired_observer_effects(
    measurements: Iterable[TaskArmMeasurement],
    *,
    active_arm: str,
    clean_arm: str = "clean",
    bootstrap_iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 12_012,
) -> tuple[PairedMetricEffect, ...]:
    """Report success and resource effects with one fail-closed paired denominator.

    The exact same task pairs are used for all six metrics. The bootstrap draws
    paired task trajectories and applies those shared draws to each metric.
    """

    _nonempty("active_arm", active_arm)
    _nonempty("clean_arm", clean_arm)
    if active_arm == clean_arm:
        raise MetricInputError("active_arm and clean_arm must differ")
    _positive_int("bootstrap_iterations", bootstrap_iterations)
    if not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise MetricInputError("confidence must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MetricInputError("seed must be an integer")

    groups: dict[tuple[str, str], dict[str, dict[str, TaskArmMeasurement]]] = {}
    count = 0
    for record in measurements:
        count += 1
        if not isinstance(record, TaskArmMeasurement):
            raise MetricInputError("all measurements must be TaskArmMeasurement records")
        if not record.complete:
            raise MetricInputError(
                f"incomplete paired measurement: {record.model}/{record.benchmark}/"
                f"{record.task_id}"
            )
        if record.arm not in {active_arm, clean_arm}:
            raise MetricInputError(f"unexpected arm in paired input: {record.arm!r}")
        if record.arm == clean_arm and record.observer_tokens != 0:
            raise MetricInputError(
                f"clean task has observer tokens: {record.model}/{record.benchmark}/"
                f"{record.task_id}"
            )
        task = groups.setdefault((record.model, record.benchmark), {}).setdefault(
            record.task_id, {}
        )
        if record.arm in task:
            raise MetricInputError(
                f"duplicate paired measurement: {record.model}/{record.benchmark}/"
                f"{record.task_id}/{record.arm}"
            )
        task[record.arm] = record
    if count == 0:
        raise MetricInputError("paired measurements are empty")

    results: list[PairedMetricEffect] = []
    for (model, benchmark), tasks in sorted(groups.items()):
        paired: list[tuple[TaskArmMeasurement, TaskArmMeasurement]] = []
        for task_id in sorted(tasks):
            arms = tasks[task_id]
            missing = {active_arm, clean_arm}.difference(arms)
            if missing:
                raise MetricInputError(
                    f"unpaired task {model}/{benchmark}/{task_id}; missing {sorted(missing)!r}"
                )
            paired.append((arms[clean_arm], arms[active_arm]))

        n_tasks = len(paired)
        # Use one deterministic task-index matrix for every metric in this
        # model/benchmark slice. This keeps each interval task-paired and makes
        # cross-metric comparisons share the same bootstrap resamples.
        draws = tuple(
            tuple(
                _bootstrap_index(seed, model, benchmark, iteration, draw, n_tasks)
                for draw in range(n_tasks)
            )
            for iteration in range(bootstrap_iterations)
        )
        for metric, unit, favorable_direction in _OBSERVER_METRICS:
            clean_values = [float(getattr(clean, metric)) for clean, _active in paired]
            active_values = [float(getattr(active, metric)) for _clean, active in paired]
            differences = [
                active - clean for clean, active in zip(clean_values, active_values)
            ]
            bootstrap = sorted(
                fmean(differences[index] for index in sampled_indices)
                for sampled_indices in draws
            )
            tail = (1.0 - float(confidence)) / 2.0
            results.append(
                PairedMetricEffect(
                    model=model,
                    benchmark=benchmark,
                    active_arm=active_arm,
                    clean_arm=clean_arm,
                    metric=metric,
                    unit=unit,
                    favorable_direction=favorable_direction,
                    n_tasks=n_tasks,
                    clean_mean=fmean(clean_values),
                    active_mean=fmean(active_values),
                    effect=fmean(differences),
                    ci_low=_quantile(bootstrap, tail),
                    ci_high=_quantile(bootstrap, 1.0 - tail),
                    confidence=float(confidence),
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=seed,
                )
            )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    """One score nested within a task; never an independent metric row."""

    checkpoint: int
    score: float
    actionable: bool = True

    def __post_init__(self) -> None:
        _positive_int("checkpoint", self.checkpoint)
        object.__setattr__(self, "score", _unit_interval("score", self.score))
        if not isinstance(self.actionable, bool):
            raise MetricInputError("actionable must be boolean")


@dataclass(frozen=True, slots=True)
class ObservationTrace:
    """All observation checkpoints and eventual degradation for one task."""

    model: str
    benchmark: str
    method: str
    task_id: str
    split: str
    checkpoints: tuple[CheckpointScore, ...]
    event_checkpoint: int | None
    complete: bool = True
    source_task_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("model", "benchmark", "method", "task_id", "split"):
            _nonempty(name, getattr(self, name))
        if not isinstance(self.checkpoints, tuple) or not self.checkpoints:
            raise MetricInputError("checkpoints must be a non-empty tuple")
        if any(not isinstance(item, CheckpointScore) for item in self.checkpoints):
            raise MetricInputError("checkpoints must contain CheckpointScore records")
        indices = [item.checkpoint for item in self.checkpoints]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise MetricInputError("checkpoint indices must be unique and strictly increasing")
        if self.event_checkpoint is not None:
            _positive_int("event_checkpoint", self.event_checkpoint)
        if not isinstance(self.complete, bool):
            raise MetricInputError("complete must be boolean")
        if self.source_task_id is not None:
            _nonempty("source_task_id", self.source_task_id)


@dataclass(frozen=True, slots=True)
class TaskPrediction:
    model: str
    benchmark: str
    method: str
    task_id: str
    split: str
    label: int
    score: float
    has_actionable_checkpoint: bool


@dataclass(frozen=True, slots=True)
class PRPoint:
    threshold: float | None
    precision: float
    recall: float
    fired_tasks: int


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_prediction: float | None
    observed_rate: float | None


@dataclass(frozen=True, slots=True)
class LeadTimeSummary:
    n_event_tasks: int
    n_detected_actionably: int
    n_without_actionable_checkpoint: int
    actionable_recall: float
    mean_checkpoints: float | None
    median_checkpoints: float | None
    lead_times: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    model: str
    benchmark: str
    method: str
    split: str
    threshold: float
    target_firing_rate: float
    achieved_firing_rate: float
    n_tasks: int
    calibration_digest: str
    selection_rule: str
    tie_break_seed: int
    target_fire_count: int

    def __post_init__(self) -> None:
        for name in ("model", "benchmark", "method", "split"):
            _nonempty(name, getattr(self, name))
        if self.split != "calibration":
            raise MetricInputError("ThresholdSelection split must be calibration")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise MetricInputError("threshold must be finite and non-negative")
        target = _unit_interval("target_firing_rate", self.target_firing_rate)
        achieved = _unit_interval("achieved_firing_rate", self.achieved_firing_rate)
        object.__setattr__(self, "target_firing_rate", target)
        object.__setattr__(self, "achieved_firing_rate", achieved)
        if achieved > target:
            raise MetricInputError("achieved_firing_rate may not exceed target_firing_rate")
        _positive_int("n_tasks", self.n_tasks)
        if self.selection_rule != "task_score_rank_hash_ties":
            raise MetricInputError(
                "selection_rule must be task_score_rank_hash_ties"
            )
        if isinstance(self.tie_break_seed, bool) or not isinstance(self.tie_break_seed, int):
            raise MetricInputError("tie_break_seed must be an integer")
        if (
            isinstance(self.target_fire_count, bool)
            or not isinstance(self.target_fire_count, int)
            or not 0 <= self.target_fire_count <= self.n_tasks
        ):
            raise MetricInputError("target_fire_count must lie between zero and n_tasks")
        if self.target_fire_count / self.n_tasks != achieved:
            raise MetricInputError(
                "achieved_firing_rate must equal target_fire_count / n_tasks"
            )
        if target - achieved >= 1 / self.n_tasks + 1e-15:
            raise MetricInputError(
                "fixed-rate selection must be within one task of its target"
            )
        if (
            not isinstance(self.calibration_digest, str)
            or len(self.calibration_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.calibration_digest)
        ):
            raise MetricInputError("calibration_digest must be a lowercase SHA256 digest")


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    model: str
    benchmark: str
    method: str
    split: str
    n_tasks: int
    n_positive_tasks: int
    locked_threshold: float
    threshold_source: str
    selection_rule: str
    target_firing_rate: float | None
    realized_score_boundary: float
    precision: float | None
    recall: float
    true_positive_tasks: int
    false_positive_tasks: int
    firing_rate: float
    auprc: float
    brier: float
    expected_calibration_error: float
    calibration_bins: tuple[CalibrationBin, ...]
    pr_curve: tuple[PRPoint, ...]
    lead_time: LeadTimeSummary
    statistical_unit: str = "task"


def _trace_key(trace: ObservationTrace) -> tuple[str, str, str, str]:
    return trace.model, trace.benchmark, trace.method, trace.task_id


def _validate_traces(traces: Iterable[ObservationTrace]) -> tuple[ObservationTrace, ...]:
    materialized = tuple(traces)
    if not materialized:
        raise MetricInputError("observation traces are empty")
    seen: set[tuple[str, str, str, str]] = set()
    for trace in materialized:
        if not isinstance(trace, ObservationTrace):
            raise MetricInputError("all traces must be ObservationTrace records")
        if not trace.complete:
            raise MetricInputError(f"incomplete observation trace: {_trace_key(trace)!r}")
        key = _trace_key(trace)
        if key in seen:
            raise MetricInputError(f"duplicate task trace: {key!r}")
        seen.add(key)
    return materialized


def _homogeneous(
    traces: Sequence[ObservationTrace],
) -> tuple[str, str, str, str]:
    values = {(t.model, t.benchmark, t.method, t.split) for t in traces}
    if len(values) != 1:
        raise MetricInputError(
            "prediction metrics require one model/benchmark/method/split slice"
        )
    return next(iter(values))


def _actionable_checkpoints(trace: ObservationTrace) -> tuple[CheckpointScore, ...]:
    return tuple(
        checkpoint
        for checkpoint in trace.checkpoints
        if checkpoint.actionable
        and (
            trace.event_checkpoint is None
            or checkpoint.checkpoint < trace.event_checkpoint
        )
    )


def collapse_task_predictions(
    traces: Iterable[ObservationTrace],
) -> tuple[TaskPrediction, ...]:
    """Collapse all checkpoints to one maximum actionable score per task."""

    materialized = _validate_traces(traces)
    predictions: list[TaskPrediction] = []
    for trace in sorted(materialized, key=_trace_key):
        actionable = _actionable_checkpoints(trace)
        predictions.append(
            TaskPrediction(
                model=trace.model,
                benchmark=trace.benchmark,
                method=trace.method,
                task_id=trace.task_id,
                split=trace.split,
                label=int(trace.event_checkpoint is not None),
                score=max((item.score for item in actionable), default=0.0),
                has_actionable_checkpoint=bool(actionable),
            )
        )
    return tuple(predictions)


def _pr_curve(predictions: Sequence[TaskPrediction]) -> tuple[tuple[PRPoint, ...], float]:
    positives = sum(item.label for item in predictions)
    if positives == 0:
        raise MetricInputError("PR metrics require at least one positive task")
    points = [PRPoint(threshold=None, precision=1.0, recall=0.0, fired_tasks=0)]
    thresholds = sorted(
        {item.score for item in predictions if item.has_actionable_checkpoint},
        reverse=True,
    )
    prior_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        fired = [
            item
            for item in predictions
            if item.has_actionable_checkpoint and item.score >= threshold
        ]
        true_positives = sum(item.label for item in fired)
        precision = true_positives / len(fired)
        recall = true_positives / positives
        points.append(
            PRPoint(
                threshold=threshold,
                precision=precision,
                recall=recall,
                fired_tasks=len(fired),
            )
        )
        if recall > prior_recall:
            area += (recall - prior_recall) * precision
            prior_recall = recall
    return tuple(points), area


def _calibration(
    predictions: Sequence[TaskPrediction], bins: int
) -> tuple[tuple[CalibrationBin, ...], float, float]:
    _positive_int("calibration_bins", bins)
    buckets: list[list[TaskPrediction]] = [[] for _ in range(bins)]
    for item in predictions:
        index = min(int(item.score * bins), bins - 1)
        buckets[index].append(item)
    result: list[CalibrationBin] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        if bucket:
            mean_prediction = fmean(item.score for item in bucket)
            observed = fmean(item.label for item in bucket)
            ece += len(bucket) / len(predictions) * abs(mean_prediction - observed)
        else:
            mean_prediction = None
            observed = None
        result.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_prediction=mean_prediction,
                observed_rate=observed,
            )
        )
    brier = fmean((item.score - item.label) ** 2 for item in predictions)
    return tuple(result), brier, ece


def _threshold_value(value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricInputError("locked threshold must be a finite non-negative number")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise MetricInputError("locked threshold must be a finite non-negative number")
    return threshold


def _tie_break_value(seed: int, *identity: str) -> int:
    material = "\0".join(("exp12/fixed-rate-tie/v1", str(seed), *identity))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest(), "big")


def _first_firing(trace: ObservationTrace, threshold: float) -> int | None:
    for checkpoint in _actionable_checkpoints(trace):
        if checkpoint.score >= threshold:
            return checkpoint.checkpoint
    return None


def _lead_time(
    traces: Sequence[ObservationTrace],
    threshold: float,
    *,
    fired_task_ids: set[str] | None = None,
) -> LeadTimeSummary:
    event_traces = [trace for trace in traces if trace.event_checkpoint is not None]
    lead_times: list[int] = []
    without_actionable = 0
    for trace in event_traces:
        actionable = _actionable_checkpoints(trace)
        if not actionable:
            without_actionable += 1
        first = (
            _first_firing(trace, threshold)
            if fired_task_ids is None or trace.task_id in fired_task_ids
            else None
        )
        if first is not None:
            # _actionable_checkpoints guarantees the difference is positive.
            lead_times.append(int(trace.event_checkpoint) - first)
    return LeadTimeSummary(
        n_event_tasks=len(event_traces),
        n_detected_actionably=len(lead_times),
        n_without_actionable_checkpoint=without_actionable,
        actionable_recall=len(lead_times) / len(event_traces),
        mean_checkpoints=fmean(lead_times) if lead_times else None,
        median_checkpoints=float(median(lead_times)) if lead_times else None,
        lead_times=tuple(sorted(lead_times)),
    )


def prediction_metrics(
    traces: Iterable[ObservationTrace],
    *,
    locked_threshold: float | ThresholdSelection,
    calibration_bins: int = 10,
) -> PredictionMetrics:
    """Score a pre-locked threshold and task-level ranking/calibration metrics."""

    materialized = _validate_traces(traces)
    model, benchmark, method, split = _homogeneous(materialized)
    if isinstance(locked_threshold, ThresholdSelection):
        if (
            locked_threshold.model,
            locked_threshold.benchmark,
            locked_threshold.method,
        ) != (model, benchmark, method):
            raise MetricInputError("locked threshold was calibrated for a different slice")
        threshold = _threshold_value(locked_threshold.threshold)
        threshold_source = "calibration_locked_fixed_rate"
        selection_rule = locked_threshold.selection_rule
        target_firing_rate: float | None = locked_threshold.target_firing_rate
    else:
        threshold = _threshold_value(locked_threshold)
        threshold_source = "provided_locked"
        selection_rule = "fixed_threshold"
        target_firing_rate = None

    predictions = collapse_task_predictions(materialized)
    pr_curve, auprc = _pr_curve(predictions)
    calibration, brier, ece = _calibration(predictions, calibration_bins)
    if isinstance(locked_threshold, ThresholdSelection):
        if any(not item.has_actionable_checkpoint for item in predictions):
            raise MetricInputError(
                "fixed-rate ranking requires an actionable checkpoint for every task"
            )
        target_count = math.floor(
            locked_threshold.target_firing_rate * len(predictions) + 1e-12
        )
        ranked = sorted(
            predictions,
            key=lambda item: (
                -item.score,
                _tie_break_value(
                    locked_threshold.tie_break_seed,
                    item.model,
                    item.benchmark,
                    item.method,
                    item.task_id,
                ),
                item.task_id,
            ),
        )
        fired = ranked[:target_count]
        realized_boundary = (
            math.nextafter(max(item.score for item in predictions), math.inf)
            if not fired
            else fired[-1].score
        )
        realized_rate = len(fired) / len(predictions)
        if not (
            realized_rate <= locked_threshold.target_firing_rate + 1e-15
            and locked_threshold.target_firing_rate - realized_rate
            < 1 / len(predictions) + 1e-15
        ):
            raise MetricInputError("fixed-rate ranking missed its declared tolerance")
    else:
        fired = [
            item
            for item in predictions
            if item.has_actionable_checkpoint and item.score >= threshold
        ]
        realized_boundary = threshold
    positives = sum(item.label for item in predictions)
    true_positives = sum(item.label for item in fired)
    false_positives = len(fired) - true_positives
    precision = true_positives / len(fired) if fired else None
    recall = true_positives / positives
    return PredictionMetrics(
        model=model,
        benchmark=benchmark,
        method=method,
        split=split,
        n_tasks=len(predictions),
        n_positive_tasks=positives,
        locked_threshold=threshold,
        threshold_source=threshold_source,
        selection_rule=selection_rule,
        target_firing_rate=target_firing_rate,
        realized_score_boundary=realized_boundary,
        precision=precision,
        recall=recall,
        true_positive_tasks=true_positives,
        false_positive_tasks=false_positives,
        firing_rate=len(fired) / len(predictions),
        auprc=auprc,
        brier=brier,
        expected_calibration_error=ece,
        calibration_bins=calibration,
        pr_curve=pr_curve,
        lead_time=_lead_time(
            materialized,
            realized_boundary,
            fired_task_ids={item.task_id for item in fired},
        ),
    )


def select_fixed_firing_rate_threshold(
    calibration_traces: Iterable[ObservationTrace],
    *,
    target_firing_rate: float,
    tie_break_seed: int = 12_012,
) -> ThresholdSelection:
    """Lock answer-blind score ranking with deterministic hash tie-breaking.

    The selected count is ``floor(target * n)``. Thus the firing rate never
    exceeds the target and differs by strictly less than one task. Hashes use
    only the declared seed and task/method identity; labels never break ties.
    """

    target = _unit_interval("target_firing_rate", target_firing_rate)
    if isinstance(tie_break_seed, bool) or not isinstance(tie_break_seed, int):
        raise MetricInputError("tie_break_seed must be an integer")
    traces = _validate_traces(calibration_traces)
    model, benchmark, method, split = _homogeneous(traces)
    if split != "calibration":
        raise MetricInputError("threshold selection is permitted on calibration split only")
    predictions = collapse_task_predictions(traces)
    if any(not item.has_actionable_checkpoint for item in predictions):
        raise MetricInputError(
            "fixed-rate ranking requires an actionable checkpoint for every task"
        )
    target_count = math.floor(target * len(predictions) + 1e-12)
    ranked = sorted(
        predictions,
        key=lambda item: (
            -item.score,
            _tie_break_value(
                tie_break_seed,
                item.model,
                item.benchmark,
                item.method,
                item.task_id,
            ),
            item.task_id,
        ),
    )
    selected = ranked[:target_count]
    threshold = (
        math.nextafter(max(item.score for item in predictions), math.inf)
        if not selected
        else selected[-1].score
    )
    achieved = target_count / len(predictions)

    selected_ids = {item.task_id for item in selected}
    digest_material = (
        f"selection_rule=task_score_rank_hash_ties\nseed={tie_break_seed}\n"
        + "\n".join(
            f"{item.task_id}\t{item.score:.17g}\t"
            f"{int(item.has_actionable_checkpoint)}\t{int(item.task_id in selected_ids)}"
            for item in sorted(predictions, key=lambda item: item.task_id)
        )
    )
    calibration_digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()
    return ThresholdSelection(
        model=model,
        benchmark=benchmark,
        method=method,
        split="calibration",
        threshold=threshold,
        target_firing_rate=target,
        achieved_firing_rate=achieved,
        n_tasks=len(predictions),
        calibration_digest=calibration_digest,
        selection_rule="task_score_rank_hash_ties",
        tie_break_seed=tie_break_seed,
        target_fire_count=target_count,
    )


def grouped_prediction_metrics(
    traces: Iterable[ObservationTrace],
    *,
    locked_thresholds: Mapping[tuple[str, str, str], float | ThresholdSelection],
    calibration_bins: int = 10,
) -> tuple[PredictionMetrics, ...]:
    """Score model/benchmark/method slices without ever pooling checkpoints."""

    materialized = _validate_traces(traces)
    groups: dict[tuple[str, str, str, str], list[ObservationTrace]] = {}
    for trace in materialized:
        groups.setdefault(
            (trace.model, trace.benchmark, trace.method, trace.split), []
        ).append(trace)
    summaries: list[PredictionMetrics] = []
    for (model, benchmark, method, _split), group in sorted(groups.items()):
        key = (model, benchmark, method)
        if key not in locked_thresholds:
            raise MetricInputError(f"missing locked threshold for {key!r}")
        summaries.append(
            prediction_metrics(
                group,
                locked_threshold=locked_thresholds[key],
                calibration_bins=calibration_bins,
            )
        )
    return tuple(summaries)


__all__ = [
    "CalibrationBin",
    "CheckpointScore",
    "LeadTimeSummary",
    "MetricInputError",
    "ObservationTrace",
    "PRPoint",
    "PairedEffect",
    "PairedMetricEffect",
    "PredictionMetrics",
    "TaskArmMeasurement",
    "TaskOutcome",
    "TaskPrediction",
    "ThresholdSelection",
    "collapse_task_predictions",
    "grouped_prediction_metrics",
    "paired_active_effects",
    "paired_observer_effects",
    "prediction_metrics",
    "select_fixed_firing_rate_threshold",
]
