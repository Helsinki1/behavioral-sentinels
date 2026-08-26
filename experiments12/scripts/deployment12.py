"""Answer-blind two-pass deployment for Experiment 12.

Pass one contains only method scores and content hashes from completed public
prefixes.  Calibration thresholds and action-count budgets are locked in a
separate artifact.  This module combines those two immutable inputs into a
method-specific schedule *before* any pass-two outcome exists, then reruns each
declared cell.  Active methods carry their probe exchanges at every observation
checkpoint; passive and baseline methods make no pass-two observer calls.

The primary estimand uses each method's natural locked-threshold firings.  The
matched-rate sensitivity selects exactly calibration-fixed ``k`` highest-score
checkpoints across each model x benchmark x method slice, with seeded hash
tie-breaking that never reads outcomes.  The yoking sensitivity copies a
preregistered anchor method's natural schedule.  In every estimand,
``operator=none`` is the no-intervention control under the same observation
burden and frozen action schedule.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from experiments12.cli12 import DEFAULT_ARTIFACTS, REPOSITORY_ROOT, _environment
from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetError
from experiments12.core.transport import CompletionResult, Transport, TransportError
from experiments12.domains.base import DomainTask, validate_sha256
from experiments12.domains.evolving_intent import (
    PINNED_COMMIT as EVOLVING_PINNED_COMMIT,
    EvolvingIntentAdapter,
)
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
from experiments12.manifest12 import RunLayout
from experiments12.metrics12 import ThresholdSelection
from experiments12.operators12 import (
    CheckpointSchedule,
    CompactionConfig,
    FeedbackNote,
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
from experiments12.pairing12 import CompletenessReport, JobCell, check_completeness
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.runner12 import (
    RunSummary,
    _assert_frozen_receipt_file,
    _stage_ledger,
    _validate_run_inputs,
)
from experiments12.spec12 import Benchmark, Operator, Stage


DEPLOYMENT_RUNNER_VERSION = 3
DEPLOYMENT_SCHEDULE_VERSION = 2
PASS_ONE_VERSION = 1
THRESHOLD_LOCK_VERSION = 2
DEPLOYMENT_RUNTIME_CONFIG_VERSION = 1

DEPLOYMENT_SCHEDULE_KIND = "experiment12_answer_blind_deployment_schedule"
PASS_ONE_KIND = "experiment12_deployment_pass_one_observations"
THRESHOLD_LOCK_KIND = "experiment12_deployment_threshold_lock"

DEPLOYMENT_SCHEDULE_RECEIPT = "deployment_observation_schedule"
PASS_ONE_RECEIPT = "deployment_pass_one_observations"
THRESHOLD_LOCK_RECEIPT = "deployment_threshold_lock"
TWO_PASS_DEPLOYMENT_MODE = "two_pass_frozen"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_OPERATOR_TYPES = {
    Operator.NONE.value: InterventionType.NONE,
    Operator.COMPACT.value: InterventionType.COMPACT,
    Operator.REGROUND.value: InterventionType.REGROUND,
    Operator.FEEDBACK.value: InterventionType.FEEDBACK,
}


class DeploymentArtifactError(ValueError):
    """A deployment artifact is incomplete, contaminated, or inconsistent."""


class DeploymentEstimand(str, Enum):
    NATURAL_THRESHOLD = "natural_threshold"
    MATCHED_RATE_TOP_K = "matched_rate_top_k"
    YOKED_ANCHOR = "yoked_anchor"


def deployment_runtime_config(
    config: HarnessConfig = HarnessConfig(),
    compaction_config: CompactionConfig = CompactionConfig(),
) -> dict[str, Any]:
    """Return the exact pass-two runtime settings frozen before dispatch."""

    if not isinstance(config, HarnessConfig) or not isinstance(
        compaction_config, CompactionConfig
    ):
        raise DeploymentArtifactError(
            "deployment runtime needs typed harness/compaction configuration"
        )
    harness = {
        "checkpoint_every": config.checkpoint_every,
        "task_max_output_tokens": config.task_max_output_tokens,
        "probe_max_output_tokens": config.probe_max_output_tokens,
        "temperature": config.temperature,
    }
    compaction = {
        "keep_last_messages": compaction_config.keep_last_messages,
        "max_excerpt_bytes": compaction_config.max_excerpt_bytes,
        "max_summary_bytes": compaction_config.max_summary_bytes,
    }
    return {
        "schema_version": DEPLOYMENT_RUNTIME_CONFIG_VERSION,
        "harness": harness,
        "harness_config_sha256": sha256_json(harness),
        "compaction": compaction,
        "compaction_config_sha256": compaction_config.config_sha256,
    }


def _exact_object(name: str, value: Any, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DeploymentArtifactError(f"{name} must be an object")
    if set(value) != fields:
        raise DeploymentArtifactError(
            f"{name} has missing or unexpected fields; "
            f"missing={sorted(fields - set(value))}, unknown={sorted(set(value) - fields)}"
        )
    return value


def _identifier(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise DeploymentArtifactError(f"{name} must be bounded single-line text")
    return value


def _digest(name: str, value: Any) -> str:
    try:
        return validate_sha256(name, value)
    except ValueError as exc:
        raise DeploymentArtifactError(str(exc)) from exc


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeploymentArtifactError(f"{name} must be an integer >= {minimum}")
    return value


def _number(name: str, value: Any, *, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeploymentArtifactError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (unit and result > 1):
        bound = " in [0,1]" if unit else " non-negative and finite"
        raise DeploymentArtifactError(f"{name} must be{bound}")
    return result


def _write_once(path: str | Path, value: Mapping[str, Any], label: str) -> str:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"{label} is write-once")
    return atomic_write_json(destination, dict(value))


@dataclass(frozen=True, slots=True)
class LockedMethodThreshold:
    model: str
    benchmark: str
    method: str
    threshold: float
    target_firing_rate: float
    achieved_firing_rate: float
    calibration_n_tasks: int
    calibration_digest: str
    selection_rule: str
    tie_break_seed: int
    calibration_target_fire_count: int

    def __post_init__(self) -> None:
        for name in ("model", "benchmark", "method"):
            _identifier(name, getattr(self, name))
        object.__setattr__(self, "threshold", _number("threshold", self.threshold))
        object.__setattr__(
            self,
            "target_firing_rate",
            _number("target_firing_rate", self.target_firing_rate, unit=True),
        )
        object.__setattr__(
            self,
            "achieved_firing_rate",
            _number("achieved_firing_rate", self.achieved_firing_rate, unit=True),
        )
        _integer("calibration_n_tasks", self.calibration_n_tasks, minimum=1)
        _digest("calibration_digest", self.calibration_digest)
        if self.selection_rule != "task_score_rank_hash_ties":
            raise DeploymentArtifactError("unsupported calibration threshold selection rule")
        if isinstance(self.tie_break_seed, bool) or not isinstance(self.tie_break_seed, int):
            raise DeploymentArtifactError("tie_break_seed must be an integer")
        _integer("calibration_target_fire_count", self.calibration_target_fire_count)
        if self.calibration_target_fire_count > self.calibration_n_tasks:
            raise DeploymentArtifactError("calibration target fire count exceeds task count")
        if (
            self.calibration_target_fire_count / self.calibration_n_tasks
            != self.achieved_firing_rate
        ):
            raise DeploymentArtifactError(
                "calibration fire count does not match achieved firing rate"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "method": self.method,
            "threshold": self.threshold,
            "target_firing_rate": self.target_firing_rate,
            "achieved_firing_rate": self.achieved_firing_rate,
            "calibration_n_tasks": self.calibration_n_tasks,
            "calibration_digest": self.calibration_digest,
            "selection_rule": self.selection_rule,
            "tie_break_seed": self.tie_break_seed,
            "calibration_target_fire_count": self.calibration_target_fire_count,
        }

    @property
    def lock_sha256(self) -> str:
        return sha256_json(self.as_dict())

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "LockedMethodThreshold":
        item = _exact_object(
            where,
            value,
            {
                "model",
                "benchmark",
                "method",
                "threshold",
                "target_firing_rate",
                "achieved_firing_rate",
                "calibration_n_tasks",
                "calibration_digest",
                "selection_rule",
                "tie_break_seed",
                "calibration_target_fire_count",
            },
        )
        result = cls(
            model=item["model"],
            benchmark=item["benchmark"],
            method=item["method"],
            threshold=item["threshold"],
            target_firing_rate=item["target_firing_rate"],
            achieved_firing_rate=item["achieved_firing_rate"],
            calibration_n_tasks=item["calibration_n_tasks"],
            calibration_digest=item["calibration_digest"],
            selection_rule=item["selection_rule"],
            tie_break_seed=item["tie_break_seed"],
            calibration_target_fire_count=item["calibration_target_fire_count"],
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ThresholdLockArtifact:
    calibration_run_id: str
    calibration_manifest_sha256: str
    natural_max_actions_per_task: int
    matched_actions_per_method: int
    yoke_anchor_method: str
    methods: tuple[LockedMethodThreshold, ...]
    schema_version: int = THRESHOLD_LOCK_VERSION
    kind: str = THRESHOLD_LOCK_KIND
    calibration_locked: bool = True
    confirmatory_outcomes_seen: bool = False

    def __post_init__(self) -> None:
        _identifier("calibration_run_id", self.calibration_run_id)
        _digest("calibration_manifest_sha256", self.calibration_manifest_sha256)
        _integer(
            "natural_max_actions_per_task",
            self.natural_max_actions_per_task,
            minimum=1,
        )
        _integer(
            "matched_actions_per_method",
            self.matched_actions_per_method,
            minimum=1,
        )
        _identifier("yoke_anchor_method", self.yoke_anchor_method)
        if self.schema_version != THRESHOLD_LOCK_VERSION or self.kind != THRESHOLD_LOCK_KIND:
            raise DeploymentArtifactError("unsupported threshold-lock artifact")
        if self.calibration_locked is not True or self.confirmatory_outcomes_seen is not False:
            raise DeploymentArtifactError(
                "thresholds must be calibration-locked before confirmatory outcomes"
            )
        keys = tuple((row.model, row.benchmark, row.method) for row in self.methods)
        if not self.methods or keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise DeploymentArtifactError("locked method thresholds must be unique and sorted")
        if self.yoke_anchor_method not in {row.method for row in self.methods}:
            raise DeploymentArtifactError("yoke anchor is absent from the threshold lock")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "calibration_run_id": self.calibration_run_id,
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "calibration_locked": self.calibration_locked,
            "confirmatory_outcomes_seen": self.confirmatory_outcomes_seen,
            "natural_max_actions_per_task": self.natural_max_actions_per_task,
            "matched_actions_per_method": self.matched_actions_per_method,
            "yoke_anchor_method": self.yoke_anchor_method,
            "methods": [row.as_dict() for row in self.methods],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ThresholdLockArtifact":
        item = _exact_object(
            "threshold lock",
            value,
            {
                "schema_version",
                "kind",
                "calibration_run_id",
                "calibration_manifest_sha256",
                "calibration_locked",
                "confirmatory_outcomes_seen",
                "natural_max_actions_per_task",
                "matched_actions_per_method",
                "yoke_anchor_method",
                "methods",
            },
        )
        if not isinstance(item["methods"], list):
            raise DeploymentArtifactError("threshold lock methods must be an array")
        result = cls(
            schema_version=item["schema_version"],
            kind=item["kind"],
            calibration_run_id=item["calibration_run_id"],
            calibration_manifest_sha256=item["calibration_manifest_sha256"],
            calibration_locked=item["calibration_locked"],
            confirmatory_outcomes_seen=item["confirmatory_outcomes_seen"],
            natural_max_actions_per_task=item["natural_max_actions_per_task"],
            matched_actions_per_method=item["matched_actions_per_method"],
            yoke_anchor_method=item["yoke_anchor_method"],
            methods=tuple(
                LockedMethodThreshold.from_dict(row, where=f"methods[{index}]")
                for index, row in enumerate(item["methods"])
            ),
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError("threshold lock is not canonical")
        return result

    def threshold_for(self, model: str, benchmark: str, method: str) -> LockedMethodThreshold:
        matches = [
            row
            for row in self.methods
            if (row.model, row.benchmark, row.method) == (model, benchmark, method)
        ]
        if len(matches) != 1:
            raise DeploymentArtifactError(
                f"missing locked threshold for {(model, benchmark, method)!r}"
            )
        return matches[0]


def load_threshold_lock(path: str | Path) -> ThresholdLockArtifact:
    return ThresholdLockArtifact.from_dict(read_json(path))


def freeze_threshold_lock(path: str | Path, artifact: ThresholdLockArtifact) -> str:
    if not isinstance(artifact, ThresholdLockArtifact):
        raise TypeError("artifact must be ThresholdLockArtifact")
    return _write_once(path, artifact.as_dict(), "deployment threshold lock")


def threshold_lock_from_calibration(
    *,
    calibration_run_id: str,
    calibration_manifest_sha256: str,
    selections: Sequence[ThresholdSelection],
    natural_max_actions_per_task: int,
    matched_actions_per_method: int,
    yoke_anchor_method: str,
) -> ThresholdLockArtifact:
    """Convert only calibration-stage selections into the deployment lock."""

    if not selections or any(not isinstance(row, ThresholdSelection) for row in selections):
        raise DeploymentArtifactError("selections must contain calibration thresholds")
    if any(row.split != "calibration" for row in selections):
        raise DeploymentArtifactError("deployment thresholds must come from calibration")
    methods = tuple(
        sorted(
            (
                LockedMethodThreshold(
                    model=row.model,
                    benchmark=row.benchmark,
                    method=row.method,
                    threshold=row.threshold,
                    target_firing_rate=row.target_firing_rate,
                    achieved_firing_rate=row.achieved_firing_rate,
                    calibration_n_tasks=row.n_tasks,
                    calibration_digest=row.calibration_digest,
                    selection_rule=row.selection_rule,
                    tie_break_seed=row.tie_break_seed,
                    calibration_target_fire_count=row.target_fire_count,
                )
                for row in selections
            ),
            key=lambda row: (row.model, row.benchmark, row.method),
        )
    )
    return ThresholdLockArtifact(
        calibration_run_id=calibration_run_id,
        calibration_manifest_sha256=calibration_manifest_sha256,
        natural_max_actions_per_task=natural_max_actions_per_task,
        matched_actions_per_method=matched_actions_per_method,
        yoke_anchor_method=yoke_anchor_method,
        methods=methods,
    )


@dataclass(frozen=True, slots=True)
class PassOneCheckpoint:
    checkpoint: int
    score: float
    source_prefix_sha256: str
    signal_record_sha256: str

    def __post_init__(self) -> None:
        _integer("pass-one checkpoint", self.checkpoint, minimum=1)
        object.__setattr__(self, "score", _number("pass-one score", self.score, unit=True))
        _digest("source_prefix_sha256", self.source_prefix_sha256)
        _digest("signal_record_sha256", self.signal_record_sha256)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "score": self.score,
            "source_prefix_sha256": self.source_prefix_sha256,
            "signal_record_sha256": self.signal_record_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "PassOneCheckpoint":
        item = _exact_object(
            where,
            value,
            {"checkpoint", "score", "source_prefix_sha256", "signal_record_sha256"},
        )
        result = cls(**item)
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PassOneMethodTrace:
    model: str
    benchmark: str
    task_id: str
    task_sha256: str
    replicate_id: int
    method: str
    active_variant: str | None
    source_trajectory_sha256: str
    task_horizon: int
    checkpoints: tuple[PassOneCheckpoint, ...]

    def __post_init__(self) -> None:
        for name in ("model", "benchmark", "task_id", "method"):
            _identifier(name, getattr(self, name))
        _digest("task_sha256", self.task_sha256)
        _digest("source_trajectory_sha256", self.source_trajectory_sha256)
        _integer("replicate_id", self.replicate_id)
        _integer("task_horizon", self.task_horizon, minimum=2)
        if self.active_variant is not None:
            _identifier("active_variant", self.active_variant)
        indices = tuple(row.checkpoint for row in self.checkpoints)
        if (
            not self.checkpoints
            or indices != tuple(sorted(indices))
            or len(indices) != len(set(indices))
            or any(index >= self.task_horizon for index in indices)
        ):
            raise DeploymentArtifactError(
                "pass-one checkpoints must be unique, increasing, and pre-final"
            )

    @property
    def identity(self) -> tuple[str, str, str, str, int, str]:
        return (
            self.model,
            self.benchmark,
            self.task_id,
            self.task_sha256,
            self.replicate_id,
            self.method,
        )

    @property
    def trace_sha256(self) -> str:
        return sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "task_sha256": self.task_sha256,
            "replicate_id": self.replicate_id,
            "method": self.method,
            "active_variant": self.active_variant,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "task_horizon": self.task_horizon,
            "checkpoints": [row.as_dict() for row in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "PassOneMethodTrace":
        item = _exact_object(
            where,
            value,
            {
                "model",
                "benchmark",
                "task_id",
                "task_sha256",
                "replicate_id",
                "method",
                "active_variant",
                "source_trajectory_sha256",
                "task_horizon",
                "checkpoints",
            },
        )
        if not isinstance(item["checkpoints"], list):
            raise DeploymentArtifactError(f"{where}.checkpoints must be an array")
        result = cls(
            model=item["model"],
            benchmark=item["benchmark"],
            task_id=item["task_id"],
            task_sha256=item["task_sha256"],
            replicate_id=item["replicate_id"],
            method=item["method"],
            active_variant=item["active_variant"],
            source_trajectory_sha256=item["source_trajectory_sha256"],
            task_horizon=item["task_horizon"],
            checkpoints=tuple(
                PassOneCheckpoint.from_dict(row, where=f"{where}.checkpoints[{index}]")
                for index, row in enumerate(item["checkpoints"])
            ),
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PassOneObservationArtifact:
    source_run_id: str
    source_manifest_sha256: str
    traces: tuple[PassOneMethodTrace, ...]
    schema_version: int = PASS_ONE_VERSION
    kind: str = PASS_ONE_KIND
    observation_prefixes_only: bool = True
    outcome_fields_present: bool = False
    complete: bool = True

    def __post_init__(self) -> None:
        _identifier("source_run_id", self.source_run_id)
        _digest("source_manifest_sha256", self.source_manifest_sha256)
        if self.schema_version != PASS_ONE_VERSION or self.kind != PASS_ONE_KIND:
            raise DeploymentArtifactError("unsupported pass-one observation artifact")
        if (
            self.observation_prefixes_only is not True
            or self.outcome_fields_present is not False
            or self.complete is not True
        ):
            raise DeploymentArtifactError("pass one must be complete, prefix-only, and outcome-free")
        identities = tuple(trace.identity for trace in self.traces)
        if not self.traces or identities != tuple(sorted(identities)) or len(
            identities
        ) != len(set(identities)):
            raise DeploymentArtifactError("pass-one traces must be unique and sorted")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_run_id": self.source_run_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "observation_prefixes_only": self.observation_prefixes_only,
            "outcome_fields_present": self.outcome_fields_present,
            "complete": self.complete,
            "traces": [trace.as_dict() for trace in self.traces],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PassOneObservationArtifact":
        item = _exact_object(
            "pass-one observations",
            value,
            {
                "schema_version",
                "kind",
                "source_run_id",
                "source_manifest_sha256",
                "observation_prefixes_only",
                "outcome_fields_present",
                "complete",
                "traces",
            },
        )
        if not isinstance(item["traces"], list):
            raise DeploymentArtifactError("pass-one traces must be an array")
        result = cls(
            schema_version=item["schema_version"],
            kind=item["kind"],
            source_run_id=item["source_run_id"],
            source_manifest_sha256=item["source_manifest_sha256"],
            observation_prefixes_only=item["observation_prefixes_only"],
            outcome_fields_present=item["outcome_fields_present"],
            complete=item["complete"],
            traces=tuple(
                PassOneMethodTrace.from_dict(row, where=f"traces[{index}]")
                for index, row in enumerate(item["traces"])
            ),
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError("pass-one observations are not canonical")
        return result


def load_pass_one_observations(path: str | Path) -> PassOneObservationArtifact:
    return PassOneObservationArtifact.from_dict(read_json(path))


def freeze_pass_one_observations(
    path: str | Path,
    artifact: PassOneObservationArtifact,
) -> str:
    if not isinstance(artifact, PassOneObservationArtifact):
        raise TypeError("artifact must be PassOneObservationArtifact")
    return _write_once(path, artifact.as_dict(), "pass-one observation artifact")


_OUTCOME_KEYS = frozenset(
    {
        "answer",
        "correctness",
        "evaluation",
        "evaluation_label",
        "event_checkpoint",
        "final_label",
        "gold",
        "gold_answer",
        "ground_truth",
        "label",
        "official_score",
        "official_success",
        "outcome",
        "success",
        "target_answer",
    }
)


def _assert_outcome_free(value: Any, path: str = "observer_record") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeploymentArtifactError(f"{path} has a non-string key")
            normalized = "_".join(
                "".join(character.lower() if character.isalnum() else " " for character in key).split()
            )
            if normalized in _OUTCOME_KEYS:
                raise DeploymentArtifactError(
                    f"pass-one observer record contains forbidden outcome field {key!r}"
                )
            _assert_outcome_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_outcome_free(item, f"{path}[{index}]")


def pass_one_trace_from_records(
    *,
    model: str,
    benchmark: str,
    task_id: str,
    task_sha256: str,
    replicate_id: int,
    method: str,
    source_trajectory_sha256: str,
    task_horizon: int,
    records: Sequence[Mapping[str, Any]],
) -> PassOneMethodTrace:
    """Reduce raw observer records to score/hash-only, outcome-free checkpoints."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise DeploymentArtifactError("pass-one observer records must be a nonempty sequence")
    active_variant = ARM_TO_PROBE.get(method)
    if method.startswith("active_") and active_variant is None:
        raise DeploymentArtifactError("unknown active pass-one method")
    raw_method, separator, passive_variant = method.partition(":")
    checkpoints: list[PassOneCheckpoint] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DeploymentArtifactError("pass-one observer record must be an object")
        _assert_outcome_free(record, f"records[{index}]")
        prefix_sha = record.get("source_prefix_sha256")
        _digest("source_prefix_sha256", prefix_sha)
        if record.get("source_trajectory_sha256") not in (
            None,
            source_trajectory_sha256,
        ):
            raise DeploymentArtifactError("observer record belongs to another trajectory")
        if active_variant is not None:
            if (
                record.get("event") != "active_probe"
                or record.get("variant") != active_variant
                or not isinstance(record.get("grade"), Mapping)
                or not isinstance(record["grade"].get("passed"), bool)
            ):
                raise DeploymentArtifactError("active pass-one record is invalid")
            checkpoint = record.get("after_task_turn")
            score = 0.0 if record["grade"]["passed"] else 1.0
        else:
            if record.get("method") != raw_method:
                raise DeploymentArtifactError("passive pass-one method changed")
            if separator and record.get("variant") != passive_variant:
                raise DeploymentArtifactError("passive pass-one variant changed")
            if not separator and record.get("variant") is not None:
                raise DeploymentArtifactError("unqualified passive method has a variant")
            checkpoint = record.get("checkpoint_turn")
            score = record.get("score")
            actionable_before = record.get("actionable_before_turn")
            if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
                raise DeploymentArtifactError("passive observer checkpoint is invalid")
            if actionable_before is not None and actionable_before != checkpoint + 1:
                raise DeploymentArtifactError("passive observer action timing is invalid")
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
            raise DeploymentArtifactError("observer checkpoint is invalid")
        checkpoints.append(
            PassOneCheckpoint(
                checkpoint=checkpoint,
                score=score,
                source_prefix_sha256=prefix_sha,
                signal_record_sha256=sha256_json(dict(record)),
            )
        )
    return PassOneMethodTrace(
        model=model,
        benchmark=benchmark,
        task_id=task_id,
        task_sha256=task_sha256,
        replicate_id=replicate_id,
        method=method,
        active_variant=active_variant,
        source_trajectory_sha256=source_trajectory_sha256,
        task_horizon=task_horizon,
        checkpoints=tuple(checkpoints),
    )


def build_pass_one_observation_artifact(
    *,
    source_run_id: str,
    source_manifest_sha256: str,
    traces: Iterable[PassOneMethodTrace],
) -> PassOneObservationArtifact:
    """Seal already-scored public-prefix traces without accepting outcomes."""

    materialized = tuple(traces)
    if any(not isinstance(trace, PassOneMethodTrace) for trace in materialized):
        raise DeploymentArtifactError("traces must contain PassOneMethodTrace records")
    return PassOneObservationArtifact(
        source_run_id=source_run_id,
        source_manifest_sha256=source_manifest_sha256,
        traces=tuple(sorted(materialized, key=lambda row: row.identity)),
    )


@dataclass(frozen=True, slots=True)
class FeedbackEvidence:
    good: tuple[str, ...] = ()
    bad: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            FeedbackNote(
                source_prefix_sha256="0" * 64,
                good=self.good,
                bad=self.bad,
                watch=self.watch,
            )
        except ValueError as exc:
            raise DeploymentArtifactError(str(exc)) from exc

    def as_dict(self) -> dict[str, list[str]]:
        return {"good": list(self.good), "bad": list(self.bad), "watch": list(self.watch)}

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "FeedbackEvidence":
        item = _exact_object(where, value, {"good", "bad", "watch"})
        if any(not isinstance(item[key], list) for key in ("good", "bad", "watch")):
            raise DeploymentArtifactError(f"{where} sections must be arrays")
        result = cls(
            good=tuple(item["good"]), bad=tuple(item["bad"]), watch=tuple(item["watch"])
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ScheduledTrigger:
    checkpoint: int
    trigger_method: str
    score: float
    locked_threshold: float
    natural_threshold_fired: bool
    selection_policy: DeploymentEstimand
    source_prefix_sha256: str
    signal_record_sha256: str
    threshold_record_sha256: str

    def __post_init__(self) -> None:
        _integer("trigger checkpoint", self.checkpoint, minimum=1)
        _identifier("trigger_method", self.trigger_method)
        object.__setattr__(self, "score", _number("trigger score", self.score, unit=True))
        object.__setattr__(
            self, "locked_threshold", _number("locked threshold", self.locked_threshold)
        )
        if not isinstance(self.natural_threshold_fired, bool):
            raise DeploymentArtifactError("natural_threshold_fired must be boolean")
        if not isinstance(self.selection_policy, DeploymentEstimand):
            raise DeploymentArtifactError("selection_policy must be DeploymentEstimand")
        for name in (
            "source_prefix_sha256",
            "signal_record_sha256",
            "threshold_record_sha256",
        ):
            _digest(name, getattr(self, name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "trigger_method": self.trigger_method,
            "score": self.score,
            "locked_threshold": self.locked_threshold,
            "natural_threshold_fired": self.natural_threshold_fired,
            "selection_policy": self.selection_policy.value,
            "source_prefix_sha256": self.source_prefix_sha256,
            "signal_record_sha256": self.signal_record_sha256,
            "threshold_record_sha256": self.threshold_record_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "ScheduledTrigger":
        item = _exact_object(
            where,
            value,
            {
                "checkpoint",
                "trigger_method",
                "score",
                "locked_threshold",
                "natural_threshold_fired",
                "selection_policy",
                "source_prefix_sha256",
                "signal_record_sha256",
                "threshold_record_sha256",
            },
        )
        try:
            policy = DeploymentEstimand(item["selection_policy"])
        except (TypeError, ValueError) as exc:
            raise DeploymentArtifactError(f"{where} has invalid selection_policy") from exc
        result = cls(
            checkpoint=item["checkpoint"],
            trigger_method=item["trigger_method"],
            score=item["score"],
            locked_threshold=item["locked_threshold"],
            natural_threshold_fired=item["natural_threshold_fired"],
            selection_policy=policy,
            source_prefix_sha256=item["source_prefix_sha256"],
            signal_record_sha256=item["signal_record_sha256"],
            threshold_record_sha256=item["threshold_record_sha256"],
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ScheduledFeedback:
    member_id: str
    checkpoint: int
    evidence: FeedbackEvidence

    def __post_init__(self) -> None:
        _identifier("feedback member_id", self.member_id)
        _integer("feedback checkpoint", self.checkpoint, minimum=1)
        if not isinstance(self.evidence, FeedbackEvidence):
            raise DeploymentArtifactError("scheduled feedback needs FeedbackEvidence")

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "checkpoint": self.checkpoint,
            "evidence": self.evidence.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "ScheduledFeedback":
        item = _exact_object(where, value, {"member_id", "checkpoint", "evidence"})
        result = cls(
            member_id=item["member_id"],
            checkpoint=item["checkpoint"],
            evidence=FeedbackEvidence.from_dict(item["evidence"], where=f"{where}.evidence"),
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result


def _schedule_from_dict(value: Any, *, where: str) -> CheckpointSchedule:
    item = _exact_object(
        where,
        value,
        {"schema_version", "group_id", "mode", "seed", "yoke_anchor_member_id", "members"},
    )
    if not isinstance(item["members"], list):
        raise DeploymentArtifactError(f"{where}.members must be an array")
    members = []
    for index, raw in enumerate(item["members"]):
        member = _exact_object(
            f"{where}.members[{index}]",
            raw,
            {"member_id", "eligible_checkpoints", "action_checkpoints"},
        )
        if not isinstance(member["eligible_checkpoints"], list) or not isinstance(
            member["action_checkpoints"], list
        ):
            raise DeploymentArtifactError("schedule checkpoints must be arrays")
        members.append(
            ScheduledMember(
                member_id=member["member_id"],
                eligible_checkpoints=tuple(member["eligible_checkpoints"]),
                action_checkpoints=tuple(member["action_checkpoints"]),
            )
        )
    try:
        mode = ScheduleMode(item["mode"])
    except (TypeError, ValueError) as exc:
        raise DeploymentArtifactError(f"{where}.mode is invalid") from exc
    result = CheckpointSchedule(
        group_id=item["group_id"],
        mode=mode,
        members=tuple(members),
        seed=item["seed"],
        yoke_anchor_member_id=item["yoke_anchor_member_id"],
        schema_version=item["schema_version"],
    )
    if result.as_dict() != dict(item):
        raise DeploymentArtifactError(f"{where} is not canonical")
    return result


@dataclass(frozen=True, slots=True)
class DeploymentScheduleGroup:
    group_id: str
    block_id: str
    model: str
    benchmark: str
    task_id: str
    task_sha256: str
    replicate_id: int
    observation_method: str
    active_variant: str | None
    source_trajectory_sha256: str
    pass_one_trace_sha256: str
    observation_checkpoints: tuple[int, ...]
    actions: tuple[ScheduledTrigger, ...]
    schedule: CheckpointSchedule
    feedback: tuple[ScheduledFeedback, ...]

    def __post_init__(self) -> None:
        for name in (
            "group_id",
            "block_id",
            "model",
            "benchmark",
            "task_id",
            "observation_method",
        ):
            _identifier(name, getattr(self, name))
        for name in ("task_sha256", "source_trajectory_sha256", "pass_one_trace_sha256"):
            _digest(name, getattr(self, name))
        _integer("replicate_id", self.replicate_id)
        if self.active_variant is not None:
            _identifier("active_variant", self.active_variant)
        if (
            not self.observation_checkpoints
            or self.observation_checkpoints != tuple(sorted(set(self.observation_checkpoints)))
        ):
            raise DeploymentArtifactError("observation checkpoints must be unique and increasing")
        action_checkpoints = tuple(action.checkpoint for action in self.actions)
        if action_checkpoints != tuple(sorted(set(action_checkpoints))):
            raise DeploymentArtifactError("scheduled triggers must be unique and increasing")
        if not set(action_checkpoints).issubset(self.observation_checkpoints):
            raise DeploymentArtifactError("actions must use observed checkpoints")
        if self.schedule.group_id != self.group_id:
            raise DeploymentArtifactError("operator schedule has another group_id")
        if self.schedule.action_checkpoints != action_checkpoints:
            raise DeploymentArtifactError("operator schedule and trigger checkpoints differ")
        feedback_keys = tuple((row.member_id, row.checkpoint) for row in self.feedback)
        if feedback_keys != tuple(sorted(feedback_keys)) or len(feedback_keys) != len(
            set(feedback_keys)
        ):
            raise DeploymentArtifactError("scheduled feedback must be unique and sorted")

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "block_id": self.block_id,
            "model": self.model,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "task_sha256": self.task_sha256,
            "replicate_id": self.replicate_id,
            "observation_method": self.observation_method,
            "active_variant": self.active_variant,
            "source_trajectory_sha256": self.source_trajectory_sha256,
            "pass_one_trace_sha256": self.pass_one_trace_sha256,
            "observation_checkpoints": list(self.observation_checkpoints),
            "actions": [action.as_dict() for action in self.actions],
            "schedule": self.schedule.as_dict(),
            "feedback": [row.as_dict() for row in self.feedback],
        }

    @classmethod
    def from_dict(cls, value: Any, *, where: str) -> "DeploymentScheduleGroup":
        item = _exact_object(
            where,
            value,
            {
                "group_id",
                "block_id",
                "model",
                "benchmark",
                "task_id",
                "task_sha256",
                "replicate_id",
                "observation_method",
                "active_variant",
                "source_trajectory_sha256",
                "pass_one_trace_sha256",
                "observation_checkpoints",
                "actions",
                "schedule",
                "feedback",
            },
        )
        if any(not isinstance(item[key], list) for key in ("observation_checkpoints", "actions", "feedback")):
            raise DeploymentArtifactError(f"{where} checkpoint/action/feedback fields must be arrays")
        result = cls(
            group_id=item["group_id"],
            block_id=item["block_id"],
            model=item["model"],
            benchmark=item["benchmark"],
            task_id=item["task_id"],
            task_sha256=item["task_sha256"],
            replicate_id=item["replicate_id"],
            observation_method=item["observation_method"],
            active_variant=item["active_variant"],
            source_trajectory_sha256=item["source_trajectory_sha256"],
            pass_one_trace_sha256=item["pass_one_trace_sha256"],
            observation_checkpoints=tuple(item["observation_checkpoints"]),
            actions=tuple(
                ScheduledTrigger.from_dict(row, where=f"{where}.actions[{index}]")
                for index, row in enumerate(item["actions"])
            ),
            schedule=_schedule_from_dict(item["schedule"], where=f"{where}.schedule"),
            feedback=tuple(
                ScheduledFeedback.from_dict(row, where=f"{where}.feedback[{index}]")
                for index, row in enumerate(item["feedback"])
            ),
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError(f"{where} is not canonical")
        return result

    def action_for(self, checkpoint: int) -> ScheduledTrigger:
        matches = [row for row in self.actions if row.checkpoint == checkpoint]
        if len(matches) != 1:
            raise DeploymentArtifactError("checkpoint has no unique frozen trigger")
        return matches[0]

    def feedback_for(self, member_id: str, checkpoint: int) -> FeedbackEvidence | None:
        matches = [
            row.evidence
            for row in self.feedback
            if row.member_id == member_id and row.checkpoint == checkpoint
        ]
        if len(matches) > 1:
            raise DeploymentArtifactError("duplicate feedback plan")
        return matches[0] if matches else None


@dataclass(frozen=True, slots=True)
class DeploymentScheduleArtifact:
    estimand: DeploymentEstimand
    pair_manifest_sha256: str
    pass_one_artifact_sha256: str
    threshold_lock_sha256: str
    pass_one_source_run_id: str
    calibration_run_id: str
    calibration_manifest_sha256: str
    groups: tuple[DeploymentScheduleGroup, ...]
    schema_version: int = DEPLOYMENT_SCHEDULE_VERSION
    kind: str = DEPLOYMENT_SCHEDULE_KIND
    answer_blind: bool = True
    outcomes_absent_at_freeze: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.estimand, DeploymentEstimand):
            raise DeploymentArtifactError("estimand must be DeploymentEstimand")
        for name in (
            "pair_manifest_sha256",
            "pass_one_artifact_sha256",
            "threshold_lock_sha256",
            "calibration_manifest_sha256",
        ):
            _digest(name, getattr(self, name))
        _identifier("pass_one_source_run_id", self.pass_one_source_run_id)
        _identifier("calibration_run_id", self.calibration_run_id)
        if self.schema_version != DEPLOYMENT_SCHEDULE_VERSION or self.kind != DEPLOYMENT_SCHEDULE_KIND:
            raise DeploymentArtifactError("unsupported deployment schedule artifact")
        if self.answer_blind is not True or self.outcomes_absent_at_freeze is not True:
            raise DeploymentArtifactError("deployment schedule must be answer-blind and pre-outcome")
        keys = tuple((row.block_id, row.observation_method) for row in self.groups)
        if not self.groups or keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise DeploymentArtifactError("deployment groups must be unique and sorted")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "estimand": self.estimand.value,
            "pair_manifest_sha256": self.pair_manifest_sha256,
            "pass_one_artifact_sha256": self.pass_one_artifact_sha256,
            "threshold_lock_sha256": self.threshold_lock_sha256,
            "pass_one_source_run_id": self.pass_one_source_run_id,
            "calibration_run_id": self.calibration_run_id,
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "answer_blind": self.answer_blind,
            "outcomes_absent_at_freeze": self.outcomes_absent_at_freeze,
            "groups": [group.as_dict() for group in self.groups],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DeploymentScheduleArtifact":
        item = _exact_object(
            "deployment schedule",
            value,
            {
                "schema_version",
                "kind",
                "estimand",
                "pair_manifest_sha256",
                "pass_one_artifact_sha256",
                "threshold_lock_sha256",
                "pass_one_source_run_id",
                "calibration_run_id",
                "calibration_manifest_sha256",
                "answer_blind",
                "outcomes_absent_at_freeze",
                "groups",
            },
        )
        if not isinstance(item["groups"], list):
            raise DeploymentArtifactError("deployment schedule groups must be an array")
        try:
            estimand = DeploymentEstimand(item["estimand"])
        except (TypeError, ValueError) as exc:
            raise DeploymentArtifactError("deployment schedule has invalid estimand") from exc
        result = cls(
            schema_version=item["schema_version"],
            kind=item["kind"],
            estimand=estimand,
            pair_manifest_sha256=item["pair_manifest_sha256"],
            pass_one_artifact_sha256=item["pass_one_artifact_sha256"],
            threshold_lock_sha256=item["threshold_lock_sha256"],
            pass_one_source_run_id=item["pass_one_source_run_id"],
            calibration_run_id=item["calibration_run_id"],
            calibration_manifest_sha256=item["calibration_manifest_sha256"],
            answer_blind=item["answer_blind"],
            outcomes_absent_at_freeze=item["outcomes_absent_at_freeze"],
            groups=tuple(
                DeploymentScheduleGroup.from_dict(row, where=f"groups[{index}]")
                for index, row in enumerate(item["groups"])
            ),
        )
        if result.as_dict() != dict(item):
            raise DeploymentArtifactError("deployment schedule is not canonical")
        return result


def load_deployment_schedule(path: str | Path) -> DeploymentScheduleArtifact:
    return DeploymentScheduleArtifact.from_dict(read_json(path))


def _trace_identity(cell: JobCell) -> tuple[str, str, str, str, int, str]:
    return (
        cell.pair_key.model,
        cell.pair_key.domain,
        cell.pair_key.task_id,
        str(cell.pair_key.task_sha256),
        cell.pair_key.replicate_id,
        cell.arm,
    )


def _natural_records(
    trace: PassOneMethodTrace,
    threshold: LockedMethodThreshold,
    cap: int,
) -> tuple[PassOneCheckpoint, ...]:
    fired = [row for row in trace.checkpoints if row.score >= threshold.threshold]
    return tuple(fired[:cap])


def _matched_top_k_by_trace(
    traces: Sequence[PassOneMethodTrace],
    threshold_lock: ThresholdLockArtifact,
) -> dict[tuple[str, str, str, str, int, str], tuple[PassOneCheckpoint, ...]]:
    """Spend one fixed action budget across each method's deployment slice."""

    grouped: dict[
        tuple[str, str, str],
        list[tuple[PassOneMethodTrace, PassOneCheckpoint]],
    ] = {}
    selected_by_trace: dict[
        tuple[str, str, str, str, int, str],
        list[PassOneCheckpoint],
    ] = {trace.identity: [] for trace in traces}
    for trace in traces:
        key = (trace.model, trace.benchmark, trace.method)
        grouped.setdefault(key, []).extend(
            (trace, checkpoint) for checkpoint in trace.checkpoints
        )

    budget = threshold_lock.matched_actions_per_method
    for key, candidates in sorted(grouped.items()):
        locked = threshold_lock.threshold_for(*key)
        if budget > len(candidates):
            raise DeploymentArtifactError(
                f"matched budget k={budget} exceeds {len(candidates)} "
                f"observed checkpoints for {key!r}"
            )

        def rank(
            candidate: tuple[PassOneMethodTrace, PassOneCheckpoint],
        ) -> tuple[Any, ...]:
            trace, checkpoint = candidate
            tie_hash = sha256_json(
                {
                    "selection": "matched_rate_top_k",
                    "tie_break_seed": locked.tie_break_seed,
                    "trace_identity": trace.identity,
                    "checkpoint": checkpoint.checkpoint,
                    "source_prefix_sha256": checkpoint.source_prefix_sha256,
                    "signal_record_sha256": checkpoint.signal_record_sha256,
                }
            )
            return (-checkpoint.score, tie_hash, trace.identity, checkpoint.checkpoint)

        for trace, checkpoint in sorted(candidates, key=rank)[:budget]:
            selected_by_trace[trace.identity].append(checkpoint)

    result = {
        identity: tuple(sorted(checkpoints, key=lambda row: row.checkpoint))
        for identity, checkpoints in selected_by_trace.items()
    }
    selected_counts: dict[tuple[str, str, str], int] = {}
    for trace in traces:
        key = (trace.model, trace.benchmark, trace.method)
        selected_counts[key] = selected_counts.get(key, 0) + len(result[trace.identity])
    if set(selected_counts) != set(grouped) or any(
        count != budget for count in selected_counts.values()
    ):
        raise DeploymentArtifactError("matched-rate selection did not spend its exact budget")
    return result


def _group_id(block_id: str, method: str, estimand: DeploymentEstimand) -> str:
    return sha256_json(
        {"block_id": block_id, "method": method, "estimand": estimand.value}
    )[:24]


def build_deployment_schedule(
    *,
    estimand: DeploymentEstimand | str,
    cells: Sequence[JobCell],
    pair_manifest_sha256: str,
    pass_one: PassOneObservationArtifact,
    pass_one_artifact_sha256: str,
    threshold_lock: ThresholdLockArtifact,
    threshold_lock_sha256: str,
    feedback_plans: Mapping[tuple[str, int], FeedbackEvidence] | None = None,
) -> DeploymentScheduleArtifact:
    """Build method-specific actions without accepting any task outcome."""

    try:
        policy = estimand if isinstance(estimand, DeploymentEstimand) else DeploymentEstimand(estimand)
    except (TypeError, ValueError) as exc:
        raise DeploymentArtifactError("unknown deployment estimand") from exc
    for name, digest in (
        ("pair_manifest_sha256", pair_manifest_sha256),
        ("pass_one_artifact_sha256", pass_one_artifact_sha256),
        ("threshold_lock_sha256", threshold_lock_sha256),
    ):
        _digest(name, digest)
    if not cells or len({cell.cell_id for cell in cells}) != len(cells):
        raise DeploymentArtifactError("pair cells are empty or duplicated")
    if any(cell.operator not in _OPERATOR_TYPES for cell in cells):
        raise DeploymentArtifactError("pair manifest contains an unsupported operator")
    expected_trace_keys = {_trace_identity(cell) for cell in cells}
    trace_map = {trace.identity: trace for trace in pass_one.traces}
    if set(trace_map) != expected_trace_keys:
        missing = sorted(expected_trace_keys - set(trace_map))
        extra = sorted(set(trace_map) - expected_trace_keys)
        raise DeploymentArtifactError(
            f"pass one does not exactly cover declared methods; missing={missing}, extra={extra}"
        )
    expected_threshold_keys = {
        (cell.pair_key.model, cell.pair_key.domain, cell.arm) for cell in cells
    }
    actual_threshold_keys = {
        (row.model, row.benchmark, row.method) for row in threshold_lock.methods
    }
    if actual_threshold_keys != expected_threshold_keys:
        raise DeploymentArtifactError("threshold lock does not exactly cover declared methods")
    for trace in pass_one.traces:
        if trace.active_variant != ARM_TO_PROBE.get(trace.method):
            raise DeploymentArtifactError("pass-one active variant differs from its method")
        if trace.method.startswith("active_") and trace.active_variant is None:
            raise DeploymentArtifactError("pass one contains an unknown active method")

    matched_by_trace = (
        _matched_top_k_by_trace(pass_one.traces, threshold_lock)
        if policy is DeploymentEstimand.MATCHED_RATE_TOP_K
        else {}
    )

    blocks: dict[str, list[JobCell]] = {}
    for cell in cells:
        blocks.setdefault(cell.block_id, []).append(cell)
    by_block_method: dict[tuple[str, str], list[JobCell]] = {}
    for block_id, block_cells in blocks.items():
        identities = {
            (
                cell.pair_key.model,
                cell.pair_key.domain,
                cell.pair_key.task_id,
                cell.pair_key.task_sha256,
                cell.pair_key.replicate_id,
            )
            for cell in block_cells
        }
        if len(identities) != 1:
            raise DeploymentArtifactError("pair block mixes model/task identities")
        methods = {cell.arm for cell in block_cells}
        if policy is DeploymentEstimand.YOKED_ANCHOR and threshold_lock.yoke_anchor_method not in methods:
            raise DeploymentArtifactError("every yoked block must contain the locked anchor method")
        for cell in block_cells:
            by_block_method.setdefault((block_id, cell.arm), []).append(cell)

    feedback_values = dict(feedback_plans or {})
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or not isinstance(value, FeedbackEvidence)
        for key, value in feedback_values.items()
    ):
        raise DeploymentArtifactError("feedback_plans must map (cell_id, checkpoint) to evidence")
    expected_feedback: set[tuple[str, int]] = set()
    groups: list[DeploymentScheduleGroup] = []
    for (block_id, method), method_cells in sorted(by_block_method.items()):
        method_cells = sorted(method_cells, key=lambda cell: cell.cell_id)
        operators = tuple(cell.operator for cell in method_cells)
        if len(operators) != len(set(operators)):
            raise DeploymentArtifactError("a method block contains duplicate operator cells")
        if Operator.NONE.value not in operators:
            raise DeploymentArtifactError(
                "every method block requires an operator=none no-intervention control"
            )
        first = method_cells[0]
        trace = trace_map[_trace_identity(first)]
        own_threshold = threshold_lock.threshold_for(
            first.pair_key.model, first.pair_key.domain, method
        )
        trigger_method = method
        trigger_trace = trace
        trigger_threshold = own_threshold
        if policy is DeploymentEstimand.NATURAL_THRESHOLD:
            selected = _natural_records(
                trace, own_threshold, threshold_lock.natural_max_actions_per_task
            )
        elif policy is DeploymentEstimand.MATCHED_RATE_TOP_K:
            selected = matched_by_trace[trace.identity]
        else:
            anchor_cell = next(
                cell
                for cell in blocks[block_id]
                if cell.arm == threshold_lock.yoke_anchor_method
            )
            trigger_method = threshold_lock.yoke_anchor_method
            trigger_trace = trace_map[_trace_identity(anchor_cell)]
            trigger_threshold = threshold_lock.threshold_for(
                anchor_cell.pair_key.model,
                anchor_cell.pair_key.domain,
                trigger_method,
            )
            selected = _natural_records(
                trigger_trace,
                trigger_threshold,
                threshold_lock.natural_max_actions_per_task,
            )
            own_indices = {row.checkpoint for row in trace.checkpoints}
            if any(row.checkpoint not in own_indices for row in selected):
                raise DeploymentArtifactError(
                    "yoked anchor selected a checkpoint unavailable to a compared method"
                )
        actions = tuple(
            ScheduledTrigger(
                checkpoint=row.checkpoint,
                trigger_method=trigger_method,
                score=row.score,
                locked_threshold=trigger_threshold.threshold,
                natural_threshold_fired=row.score >= trigger_threshold.threshold,
                selection_policy=policy,
                source_prefix_sha256=row.source_prefix_sha256,
                signal_record_sha256=row.signal_record_sha256,
                threshold_record_sha256=trigger_threshold.lock_sha256,
            )
            for row in selected
        )
        observations = tuple(row.checkpoint for row in trace.checkpoints)
        action_checkpoints = tuple(row.checkpoint for row in actions)
        group_id = _group_id(block_id, method, policy)
        member_ids = tuple(sorted(cell.cell_id for cell in method_cells))
        if len(member_ids) < 2:
            raise DeploymentArtifactError(
                "each method needs at least two operator cells including no-intervention"
            )
        schedule = CheckpointSchedule(
            group_id=group_id,
            mode=ScheduleMode.MATCHED,
            members=tuple(
                ScheduledMember(member_id, observations, action_checkpoints)
                for member_id in member_ids
            ),
            # Actions are score-selected.  This deterministic audit seed never
            # participates in checkpoint selection.
            seed=int(group_id[:16], 16),
        )
        scheduled_feedback: list[ScheduledFeedback] = []
        for cell in method_cells:
            for checkpoint in action_checkpoints:
                key = (cell.cell_id, checkpoint)
                if cell.operator == Operator.FEEDBACK.value:
                    expected_feedback.add(key)
                    if key not in feedback_values:
                        raise DeploymentArtifactError(
                            f"missing frozen feedback evidence for {key!r}"
                        )
                    scheduled_feedback.append(
                        ScheduledFeedback(cell.cell_id, checkpoint, feedback_values[key])
                    )
                elif key in feedback_values:
                    raise DeploymentArtifactError("non-feedback cell has feedback evidence")
        groups.append(
            DeploymentScheduleGroup(
                group_id=group_id,
                block_id=block_id,
                model=first.pair_key.model,
                benchmark=first.pair_key.domain,
                task_id=first.pair_key.task_id,
                task_sha256=str(first.pair_key.task_sha256),
                replicate_id=first.pair_key.replicate_id,
                observation_method=method,
                active_variant=trace.active_variant,
                source_trajectory_sha256=trace.source_trajectory_sha256,
                pass_one_trace_sha256=trace.trace_sha256,
                observation_checkpoints=observations,
                actions=actions,
                schedule=schedule,
                feedback=tuple(
                    sorted(scheduled_feedback, key=lambda row: (row.member_id, row.checkpoint))
                ),
            )
        )
    if set(feedback_values) != expected_feedback:
        raise DeploymentArtifactError("feedback plan has missing or undeclared entries")
    if policy is DeploymentEstimand.MATCHED_RATE_TOP_K:
        scheduled_counts: dict[tuple[str, str, str], int] = {}
        for group in groups:
            key = (group.model, group.benchmark, group.observation_method)
            scheduled_counts[key] = scheduled_counts.get(key, 0) + len(group.actions)
        if set(scheduled_counts) != actual_threshold_keys or any(
            count != threshold_lock.matched_actions_per_method
            for count in scheduled_counts.values()
        ):
            raise DeploymentArtifactError(
                "matched-rate schedule did not preserve the locked method budget"
            )
    return DeploymentScheduleArtifact(
        estimand=policy,
        pair_manifest_sha256=pair_manifest_sha256,
        pass_one_artifact_sha256=pass_one_artifact_sha256,
        threshold_lock_sha256=threshold_lock_sha256,
        pass_one_source_run_id=pass_one.source_run_id,
        calibration_run_id=threshold_lock.calibration_run_id,
        calibration_manifest_sha256=threshold_lock.calibration_manifest_sha256,
        groups=tuple(sorted(groups, key=lambda row: (row.block_id, row.observation_method))),
    )


def freeze_deployment_schedule(
    path: str | Path,
    artifact: DeploymentScheduleArtifact,
    *,
    outcome_artifacts_root: str | Path,
) -> str:
    """Freeze a schedule only while the pass-two outcome namespace is empty."""

    if not isinstance(artifact, DeploymentScheduleArtifact):
        raise TypeError("artifact must be DeploymentScheduleArtifact")
    destination = Path(path).resolve()
    outcome_root = Path(outcome_artifacts_root).resolve()
    if destination == outcome_root or outcome_root in destination.parents:
        raise DeploymentArtifactError("schedule cannot be written inside the outcome namespace")
    if outcome_root.is_file() or (
        outcome_root.exists()
        and any(item.is_file() for item in outcome_root.rglob("*"))
    ):
        raise DeploymentArtifactError(
            "deployment outcomes already exist; post-outcome schedule freezing is forbidden"
        )
    return _write_once(destination, artifact.as_dict(), "deployment observation schedule")


def _feedback_plan_from_artifact(
    artifact: DeploymentScheduleArtifact,
) -> dict[tuple[str, int], FeedbackEvidence]:
    return {
        (row.member_id, row.checkpoint): row.evidence
        for group in artifact.groups
        for row in group.feedback
    }


def validate_deployment_schedule(
    artifact: DeploymentScheduleArtifact,
    *,
    cells: Sequence[JobCell],
    task_index: Mapping[tuple[str, str, str], DomainTask],
    pass_one: PassOneObservationArtifact,
    threshold_lock: ThresholdLockArtifact,
) -> dict[tuple[str, str], DeploymentScheduleGroup]:
    """Recompute the schedule from its locked sources and bind every task."""

    try:
        rebuilt = build_deployment_schedule(
            estimand=artifact.estimand,
            cells=cells,
            pair_manifest_sha256=artifact.pair_manifest_sha256,
            pass_one=pass_one,
            pass_one_artifact_sha256=artifact.pass_one_artifact_sha256,
            threshold_lock=threshold_lock,
            threshold_lock_sha256=artifact.threshold_lock_sha256,
            feedback_plans=_feedback_plan_from_artifact(artifact),
        )
    except DeploymentArtifactError as exc:
        raise DeploymentArtifactError(
            f"deployment schedule does not reproduce from locked sources: {exc}"
        ) from exc
    if rebuilt != artifact:
        raise DeploymentArtifactError(
            "deployment schedule does not reproduce from pass one and locked thresholds"
        )
    groups = {(group.block_id, group.observation_method): group for group in artifact.groups}
    for cell in cells:
        key = (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        try:
            task = task_index[key]
            group = groups[(cell.block_id, cell.arm)]
        except KeyError as exc:
            raise DeploymentArtifactError("schedule refers to an undeclared task/method") from exc
        _validate_cell_contract(cell=cell, task=task, group=group)
    return groups


def _public_feedback_fields(
    task: DomainTask,
    checkpoint: int,
    active_variant: str | None,
    probe_index: int | None,
) -> tuple[str, ...]:
    # Restrict to checkpoint-local material that remains visible even after an
    # earlier compaction/reground.  This prevents frozen notes from relying on
    # stochastic pass-one assistant text or future benchmark turns.
    fields = [task.turns[checkpoint - 1].user_message]
    if active_variant is not None and probe_index is not None:
        fields.append(
            render_probe_prompt(
                generate_probe_instance(active_variant, task.instance_id, probe_index)
            )
        )
    return tuple(fields)


def _validate_cell_contract(
    *,
    cell: JobCell,
    task: DomainTask,
    group: DeploymentScheduleGroup,
) -> None:
    if cell.operator not in _OPERATOR_TYPES:
        raise DeploymentArtifactError("unsupported deployment operator")
    if task.domain != Benchmark.EVOLVING_GSM8K.value:
        raise DeploymentArtifactError("deployment12 only accepts Evolving Intent tasks")
    if (
        group.block_id != cell.block_id
        or group.model != cell.pair_key.model
        or group.benchmark != cell.pair_key.domain
        or group.task_id != cell.pair_key.task_id
        or group.task_sha256 != task.task_sha256
        or group.replicate_id != cell.pair_key.replicate_id
        or group.observation_method != cell.arm
        or group.active_variant != ARM_TO_PROBE.get(cell.arm)
        or len(task.turns) <= max(group.observation_checkpoints)
    ):
        raise DeploymentArtifactError("cell, task, method, and schedule identities differ")
    if cell.arm.startswith("active_") and group.active_variant is None:
        raise DeploymentArtifactError("unknown active deployment method")
    member_ids = {member.member_id for member in group.schedule.members}
    if cell.cell_id not in member_ids:
        raise DeploymentArtifactError("cell is absent from its method-specific schedule")
    action_checkpoints = {row.checkpoint for row in group.actions}
    planned = {
        (row.member_id, row.checkpoint): row.evidence for row in group.feedback
    }
    expected = (
        {(cell.cell_id, checkpoint) for checkpoint in action_checkpoints}
        if cell.operator == Operator.FEEDBACK.value
        else set()
    )
    actual = {key for key in planned if key[0] == cell.cell_id}
    if actual != expected:
        raise DeploymentArtifactError("feedback evidence does not exactly cover feedback actions")
    probe_ordinals = {
        checkpoint: index
        for index, checkpoint in enumerate(group.observation_checkpoints, 1)
    }
    for member_id, checkpoint in actual:
        evidence = planned[(member_id, checkpoint)]
        fields = _public_feedback_fields(
            task,
            checkpoint,
            group.active_variant,
            probe_ordinals.get(checkpoint),
        )
        for quote in (*evidence.good, *evidence.bad, *evidence.watch):
            if not any(quote in field for field in fields):
                raise DeploymentArtifactError(
                    "feedback quote is not checkpoint-local public material"
                )


def _request_key(run_id: str, cell_id: str, kind: str, index: int) -> str:
    if not _RUN_ID_RE.fullmatch(run_id) or not _CELL_ID_RE.fullmatch(cell_id):
        raise DeploymentArtifactError("run/cell identifiers are unsafe")
    return f"{run_id}/{cell_id}/deployment-{kind}-{index}"


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
    empty = {key: 0 for key in fields}
    empty["accounted_cost_usd"] = Decimal("0")
    categories = {"agent": dict(empty), "active_monitor": dict(empty)}
    resolved_ids: set[str] = set()
    for event in events:
        if event.get("event") not in {"task_turn", "active_probe"}:
            continue
        bucket = categories[
            "agent" if event["event"] == "task_turn" else "active_monitor"
        ]
        call = event["call"]
        bucket["calls"] += 1
        for key in fields[1:-1]:
            bucket[key] += int(call["usage"][key])
        bucket["elapsed_ms"] += int(call["elapsed_ms"])
        bucket["accounted_cost_usd"] += Decimal(call["accounted_cost_usd"])
        resolved_ids.add(call["resolved_model_id"])
    total = dict(empty)
    for bucket in categories.values():
        for key in fields:
            total[key] += bucket[key]
        total["accounted_cost_usd"] += bucket["accounted_cost_usd"]
        bucket["accounted_cost_usd"] = str(bucket["accounted_cost_usd"])
    total["accounted_cost_usd"] = str(total["accounted_cost_usd"])
    return {
        "by_category": categories,
        "total": total,
        "resolved_model_ids": sorted(resolved_ids),
    }


def _public_state(task: DomainTask, checkpoint: int) -> dict[str, Any]:
    return {
        "completed_task_turns": [
            {"turn": turn.index, "user_message": turn.user_message}
            for turn in task.turns[:checkpoint]
        ],
        "public_metadata": dict(task.public_metadata),
    }


def _design(
    *,
    run_id: str,
    cell: JobCell,
    task: DomainTask,
    group: DeploymentScheduleGroup,
    artifact: DeploymentScheduleArtifact,
    schedule_artifact_sha256: str,
    config: HarnessConfig,
    compaction_config: CompactionConfig,
) -> dict[str, Any]:
    return {
        "deployment_runner_version": DEPLOYMENT_RUNNER_VERSION,
        "run_id": run_id,
        "cell": cell.as_dict(),
        "task": task.manifest_record(),
        "estimand": artifact.estimand.value,
        "schedule_artifact_sha256": schedule_artifact_sha256,
        "pass_one_artifact_sha256": artifact.pass_one_artifact_sha256,
        "threshold_lock_sha256": artifact.threshold_lock_sha256,
        "group": group.as_dict(),
        "runtime_config": deployment_runtime_config(config, compaction_config),
    }


async def run_deployment_task(
    *,
    run_id: str,
    cell: JobCell,
    task: DomainTask,
    group: DeploymentScheduleGroup,
    schedule_artifact: DeploymentScheduleArtifact,
    schedule_artifact_sha256: str,
    transport: Transport,
    event_path: str | Path,
    output_path: str | Path,
    yes_spend: bool = False,
    config: HarnessConfig = HarnessConfig(),
    compaction_config: CompactionConfig = CompactionConfig(),
) -> dict[str, Any]:
    """Run one pass-two cell; pass-two probe grades never alter actions."""

    if not yes_spend:
        raise DeploymentArtifactError("provider dispatch requires explicit yes_spend=True")
    if not isinstance(compaction_config, CompactionConfig):
        raise DeploymentArtifactError("compaction_config must be CompactionConfig")
    if cell.pair_key.model not in DEFAULT_REASONING_EFFORT:
        raise DeploymentArtifactError("model runtime settings are not frozen")
    _validate_cell_contract(cell=cell, task=task, group=group)
    if group not in schedule_artifact.groups:
        raise DeploymentArtifactError("method group is absent from the frozen artifact")
    schedule_digest = _digest("schedule_artifact_sha256", schedule_artifact_sha256)
    event_file, output_file = Path(event_path), Path(output_path)
    design = _design(
        run_id=run_id,
        cell=cell,
        task=task,
        group=group,
        artifact=schedule_artifact,
        schedule_artifact_sha256=schedule_digest,
        config=config,
        compaction_config=compaction_config,
    )
    design_sha256 = sha256_json(design)
    start = {"event": "start", "design_sha256": design_sha256, **design}
    if output_file.exists():
        existing = read_json(output_file)
        if (
            existing.get("complete") is not True
            or existing.get("design_sha256") != design_sha256
            or sha256_json(existing.get("messages")) != existing.get("transcript_sha256")
        ):
            raise DeploymentArtifactError("existing deployment output is torn or changed")
        if not event_file.exists() or read_jsonl(event_file)[0] != start:
            raise DeploymentArtifactError("materialized output lacks its exact event log")
        expected_accounting = _accounting(
            [*existing.get("task_records", ()), *existing.get("probe_records", ())]
        )
        if existing.get("accounting") != expected_accounting:
            raise DeploymentArtifactError("materialized accounting does not reconcile")
        return existing
    if event_file.exists():
        events = read_jsonl(event_file)
        if not events or events[0] != start:
            raise DeploymentArtifactError("existing event log belongs to another design")
        raise DeploymentArtifactError(
            "partial deployment event log cannot be replayed without duplicate-call risk"
        )
    event_file.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(event_file, start)

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

    messages: list[dict[str, Any]] = []
    assistant_task: list[str] = []
    events: list[dict[str, Any]] = [start]
    probe_records: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    action_checkpoints = {row.checkpoint for row in group.actions}

    for turn_number in range(1, len(task.turns) + 1):
        turn = task.next_turn(len(assistant_task))
        if turn is None or turn.index != turn_number:
            raise AssertionError("task did not expose one contiguous next turn")
        user = {
            "role": "user",
            "content": first_content if turn_number == 1 else turn.user_message,
        }
        messages.append(user)
        result = await transport.complete(
            cell.pair_key.model,
            messages,
            purpose="deployment_agent_turn",
            request_key=_request_key(run_id, cell.cell_id, "task", turn_number),
            input_token_estimate=conservative_input_token_bound(messages),
            max_output_tokens=config.task_max_output_tokens,
            temperature=config.temperature,
            reasoning_effort=DEFAULT_REASONING_EFFORT[cell.pair_key.model],
        )
        if result.tool_calls:
            raise DeploymentArtifactError("scripted Evolving task returned tool calls")
        assistant = {"role": "assistant", "content": result.text}
        messages.append(assistant)
        assistant_task.append(result.text)
        task_event = {
            "event": "task_turn",
            "task_turn": turn_number,
            "user_message": user,
            "assistant_message": assistant,
            "call": _call_record(result),
            "continued_history_sha256": sha256_json(messages),
        }
        append_jsonl(event_file, task_event)
        events.append(task_event)

        if turn_number in probe_ordinals and active_variant is not None:
            probe_index = probe_ordinals[turn_number]
            instance = generate_probe_instance(active_variant, task.instance_id, probe_index)
            probe_user = {"role": "user", "content": render_probe_prompt(instance)}
            messages.append(probe_user)
            result = await transport.complete(
                cell.pair_key.model,
                messages,
                purpose="deployment_active_probe",
                request_key=_request_key(run_id, cell.cell_id, "probe", probe_index),
                input_token_estimate=conservative_input_token_bound(messages),
                max_output_tokens=config.probe_max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=DEFAULT_REASONING_EFFORT[cell.pair_key.model],
            )
            if result.tool_calls:
                raise DeploymentArtifactError("active deployment probe returned tool calls")
            probe_assistant = {"role": "assistant", "content": result.text}
            messages.append(probe_assistant)
            grade = grade_probe_response(instance, result.text)
            probe_event = {
                "event": "active_probe",
                "after_task_turn": turn_number,
                "checkpoint_index": probe_index,
                "variant": active_variant,
                "user_message": probe_user,
                "assistant_message": probe_assistant,
                "grade": {
                    "passed": grade.passed,
                    "value_correct": grade.value_correct,
                    "exact_format": grade.exact_format,
                    "error": grade.error,
                    "expected_sha256": sha256_json(instance.expected_answer),
                },
                "call": _call_record(result),
                "changes_frozen_timing": False,
            }
            append_jsonl(event_file, probe_event)
            events.append(probe_event)
            probe_records.append(probe_event)

        if turn_number not in action_checkpoints:
            continue
        prefix = freeze_visible_prefix(
            domain=task.domain,
            task_id=task.task_id,
            after_turn=turn_number,
            messages=messages,
        )
        trigger = group.action_for(turn_number)
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
                raise DeploymentArtifactError("feedback action lacks frozen evidence")
            feedback = make_feedback_note(
                prefix, good=evidence.good, bad=evidence.bad, watch=evidence.watch
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
        messages = application.continued_history
        intervention_event = {
            **application.as_event(),
            "declared_operator": cell.operator,
            "observation_method": group.observation_method,
            "selection_policy": trigger.selection_policy.value,
            "trigger_score": trigger.score,
            "locked_threshold": trigger.locked_threshold,
            "natural_threshold_fired": trigger.natural_threshold_fired,
            "threshold_record_sha256": trigger.threshold_record_sha256,
        }
        append_jsonl(event_file, intervention_event)
        events.append(intervention_event)
        intervention_records.append(intervention_event)

    if len(intervention_records) != len(group.actions):
        raise AssertionError("not every frozen trigger was applied")
    prediction, success = grade_final_numeric(assistant_task[-1], task.evaluation_label)
    transcript_sha256 = sha256_json(messages)
    materialized = {
        "schema_version": 2,
        "deployment_runner_version": DEPLOYMENT_RUNNER_VERSION,
        "run_id": run_id,
        "cell_id": cell.cell_id,
        "design_sha256": design_sha256,
        "model": cell.pair_key.model,
        "domain": task.domain,
        "task_id": task.task_id,
        "condition": task.condition,
        "task_sha256": task.task_sha256,
        "arm": cell.arm,
        "operator": cell.operator,
        "estimand": schedule_artifact.estimand.value,
        "schedule": {
            "artifact_sha256": schedule_digest,
            "group_id": group.group_id,
            "schedule_sha256": group.schedule.schedule_sha256,
            "pass_one_artifact_sha256": schedule_artifact.pass_one_artifact_sha256,
            "threshold_lock_sha256": schedule_artifact.threshold_lock_sha256,
            "observation_checkpoints": list(group.observation_checkpoints),
            "action_checkpoints": [row.checkpoint for row in group.actions],
        },
        "observation_method": group.observation_method,
        "active_probe_variant": active_variant,
        "messages": messages,
        "task_assistant_messages": assistant_task,
        "task_records": [event for event in events if event.get("event") == "task_turn"],
        "probe_records": probe_records,
        "intervention_records": intervention_records,
        "evaluation": {
            "prediction": prediction,
            "evaluation_label_sha256": (
                None if task.evaluation_label is None else sha256_json(task.evaluation_label)
            ),
            "success": success,
        },
        "transcript_sha256": transcript_sha256,
        "accounting": _accounting(events),
        "complete": True,
    }
    atomic_write_json(output_file, materialized)
    append_jsonl(
        event_file,
        {
            "event": "complete",
            "design_sha256": design_sha256,
            "task_turns": len(assistant_task),
            "active_probe_calls": len(probe_records),
            "interventions": len(intervention_records),
            "transcript_sha256": transcript_sha256,
            "output_sha256": sha256_file(output_file),
            "prediction": prediction,
            "success": success,
        },
    )
    return materialized


def _job_state(path: Path, *, cell: JobCell, state: str, detail: Mapping[str, Any]) -> None:
    if state not in {"complete", "failed"}:
        raise DeploymentArtifactError("invalid deployment job state")
    atomic_write_json(
        path,
        {
            "deployment_runner_version": DEPLOYMENT_RUNNER_VERSION,
            "cell_id": cell.cell_id,
            "state": state,
            **dict(detail),
        },
    )


def _require_receipt(
    manifest: Mapping[str, Any],
    *,
    name: str,
    path: Path,
) -> str:
    digest = sha256_file(path)
    receipts = manifest.get("benchmark_receipts", ())
    if not any(
        isinstance(receipt, Mapping)
        and receipt.get("name") == name
        and receipt.get("sha256") == digest
        for receipt in receipts
    ):
        raise DeploymentArtifactError(f"{name} is not frozen into the run manifest")
    return digest


def _validate_evolving_runtime_provenance(
    *,
    manifest: Mapping[str, Any],
    cells: Sequence[JobCell],
    task_index: Mapping[tuple[str, str, str], DomainTask],
    dataset_path: str | Path | None,
    build_receipt_path: str | Path | None,
) -> None:
    """Bind deployment to the same rendered Evolving artifact as pass one."""

    evolving_cells = tuple(
        cell
        for cell in cells
        if cell.pair_key.domain == Benchmark.EVOLVING_GSM8K.value
    )
    if not evolving_cells:
        if dataset_path is not None or build_receipt_path is not None:
            raise DeploymentArtifactError(
                "Evolving provenance was supplied for a non-Evolving deployment"
            )
        return
    if dataset_path is None or build_receipt_path is None:
        raise DeploymentArtifactError(
            "Evolving deployment requires its frozen dataset and build receipt"
        )
    dataset = Path(dataset_path)
    build_receipt = Path(build_receipt_path)
    try:
        _assert_frozen_receipt_file(manifest, "evolving_rendered_dataset", dataset)
        _assert_frozen_receipt_file(
            manifest, "evolving_build_receipt", build_receipt
        )
    except ValueError as exc:
        raise DeploymentArtifactError(str(exc)) from exc
    dataset_sha256 = sha256_file(dataset)
    payload = read_json(build_receipt)
    frozen_dataset = payload.get("frozen_dataset")
    if (
        payload.get("benchmark") != Benchmark.EVOLVING_GSM8K.value
        or payload.get("upstream_commit") != EVOLVING_PINNED_COMMIT
        or payload.get("shared_across_target_arms_and_models") is not True
        or not isinstance(frozen_dataset, Mapping)
        or frozen_dataset.get("sha256") != dataset_sha256
    ):
        raise DeploymentArtifactError(
            "Evolving build receipt does not attest the runtime dataset"
        )
    declared_tasks = {
        (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        for cell in evolving_cells
    }
    if {
        task_index[key].source_sha256 for key in declared_tasks
    } != {dataset_sha256}:
        raise DeploymentArtifactError(
            "Evolving deployment tasks are not derived from the runtime dataset"
        )


async def execute_deployment_run(
    *,
    run_id: str,
    task_manifest_path: str | Path,
    pass_one_path: str | Path,
    threshold_lock_path: str | Path,
    schedule_path: str | Path,
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
    if not yes_spend:
        raise DeploymentArtifactError("provider dispatch requires explicit yes_spend=True")
    if max_new_cells is not None and max_new_cells < 1:
        raise DeploymentArtifactError("max_new_cells must be positive or null")
    try:
        shard = ExecutionShard(count=shard_count, index=shard_index)
    except ValueError as exc:
        raise DeploymentArtifactError(str(exc)) from exc
    layout = RunLayout.for_run(artifacts_root, run_id)
    manifest, cells, task_index = _validate_run_inputs(
        layout=layout,
        task_manifest_path=Path(task_manifest_path),
        tasks=tasks,
    )
    extra_config = manifest.get("extra_config")
    if (
        not isinstance(extra_config, Mapping)
        or extra_config.get("deployment_mode") != TWO_PASS_DEPLOYMENT_MODE
    ):
        raise DeploymentArtifactError(
            "manifest is not frozen for controlled two-pass deployment"
        )
    runtime_config = deployment_runtime_config(config, compaction_config)
    if extra_config.get("deployment_runtime") != runtime_config:
        raise DeploymentArtifactError(
            "deployment runtime configuration differs from the frozen manifest"
        )
    _validate_evolving_runtime_provenance(
        manifest=manifest,
        cells=cells,
        task_index=task_index,
        dataset_path=evolving_dataset_path,
        build_receipt_path=evolving_build_receipt_path,
    )
    pair_digest = sha256_file(layout.pairs)
    pass_one_file = Path(pass_one_path)
    threshold_file = Path(threshold_lock_path)
    schedule_file = Path(schedule_path)
    pass_one_digest = _require_receipt(
        manifest, name=PASS_ONE_RECEIPT, path=pass_one_file
    )
    threshold_digest = _require_receipt(
        manifest, name=THRESHOLD_LOCK_RECEIPT, path=threshold_file
    )
    schedule_digest = _require_receipt(
        manifest, name=DEPLOYMENT_SCHEDULE_RECEIPT, path=schedule_file
    )
    pass_one = load_pass_one_observations(pass_one_file)
    threshold_lock = load_threshold_lock(threshold_file)
    schedule_artifact = load_deployment_schedule(schedule_file)
    if (
        schedule_artifact.pair_manifest_sha256 != pair_digest
        or schedule_artifact.pass_one_artifact_sha256 != pass_one_digest
        or schedule_artifact.threshold_lock_sha256 != threshold_digest
    ):
        raise DeploymentArtifactError("schedule source receipts do not match supplied artifacts")
    groups = validate_deployment_schedule(
        schedule_artifact,
        cells=cells,
        task_index=task_index,
        pass_one=pass_one,
        threshold_lock=threshold_lock,
    )
    declared_estimand = extra_config.get("deployment_estimand")
    if declared_estimand != schedule_artifact.estimand.value:
        raise DeploymentArtifactError("manifest does not bind the deployment estimand")
    stage = Stage(manifest["stage"])
    if stage is Stage.OFFLINE:
        raise DeploymentArtifactError("offline stage cannot dispatch model calls")
    shard_cells = shard.select(cells)
    if transport is None:
        ledger = _stage_ledger(layout, run_id, stage)
        transport = Transport(
            ledger,
            layout.events / "call_attempts.jsonl",
            environ=environ,
            max_attempts=3 if stage is Stage.SMOKE else 6,
        )

    output_root = layout.results / "deployment"
    job_root = layout.results / "deployment_jobs"
    output_root.mkdir(parents=True, exist_ok=True)
    job_root.mkdir(parents=True, exist_ok=True)
    completed = failed = skipped = visited = new_cells = 0
    for cell in shard_cells:
        output = output_root / f"{cell.cell_id}.json"
        job = job_root / f"{cell.cell_id}.json"
        key = (
            cell.pair_key.domain,
            cell.pair_key.task_id,
            str(cell.pair_key.task_sha256),
        )
        kwargs = dict(
            run_id=run_id,
            cell=cell,
            task=task_index[key],
            group=groups[(cell.block_id, cell.arm)],
            schedule_artifact=schedule_artifact,
            schedule_artifact_sha256=schedule_digest,
            transport=transport,
            event_path=layout.events / f"deployment-{cell.cell_id}.jsonl",
            output_path=output,
            yes_spend=True,
            config=config,
            compaction_config=compaction_config,
        )
        if output.exists():
            existing = await run_deployment_task(**kwargs)
            if job.exists():
                receipt = read_json(job)
                if (
                    receipt.get("state") != "complete"
                    or receipt.get("output_sha256") != sha256_file(output)
                ):
                    raise DeploymentArtifactError("output conflicts with its job receipt")
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
            raise DeploymentArtifactError("job receipt exists without its output")
        if max_new_cells is not None and new_cells >= max_new_cells:
            skipped += 1
            continue
        visited += 1
        new_cells += 1
        try:
            result = await run_deployment_task(**kwargs)
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
        phase="deployment",
        shard_count=shard.count,
        shard_index=shard.index,
        shard_cells=len(shard_cells),
    )


def deployment_completeness(layout: RunLayout, cells: Sequence[JobCell]) -> CompletenessReport:
    states: list[tuple[str, str]] = []
    for cell in cells:
        output = layout.results / "deployment" / f"{cell.cell_id}.json"
        job = layout.results / "deployment_jobs" / f"{cell.cell_id}.json"
        if output.is_file() and job.is_file() and read_json(job).get("state") == "complete":
            states.append((cell.cell_id, "complete"))
        elif job.is_file() and read_json(job).get("state") == "failed":
            states.append((cell.cell_id, "failed"))
        else:
            states.append((cell.cell_id, "missing"))
    return check_completeness(cells, states)


def extract_deployment_outcomes(
    cells: Sequence[JobCell],
    outputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Extract one task-level row per declared cell; never intersect present files."""

    expected = {cell.cell_id for cell in cells}
    if set(outputs) != expected:
        raise DeploymentArtifactError("deployment outputs do not exactly cover declared cells")
    rows = []
    for cell in cells:
        output = outputs[cell.cell_id]
        success = output.get("evaluation", {}).get("success")
        if (
            output.get("complete") is not True
            or output.get("cell_id") != cell.cell_id
            or output.get("arm") != cell.arm
            or output.get("operator") != cell.operator
            or not isinstance(success, bool)
        ):
            raise DeploymentArtifactError("deployment output identity/outcome is invalid")
        schedule = output.get("schedule")
        if not isinstance(schedule, Mapping) or not isinstance(
            schedule.get("action_checkpoints"), list
        ):
            raise DeploymentArtifactError("deployment output lacks its action schedule")
        rows.append(
            {
                "cell_id": cell.cell_id,
                "model": cell.pair_key.model,
                "benchmark": cell.pair_key.domain,
                "task_id": cell.pair_key.task_id,
                "replicate_id": cell.pair_key.replicate_id,
                "method": cell.arm,
                "operator": cell.operator,
                "estimand": output.get("estimand"),
                "success": success,
                "scheduled_actions": len(schedule["action_checkpoints"]),
                "applied_interventions": len(output.get("intervention_records", ())),
                "accounting": output.get("accounting"),
            }
        )
    return tuple(rows)


async def _run_evolving(args: argparse.Namespace) -> int:
    if not args.yes_spend:
        raise DeploymentArtifactError("provider dispatch requires --yes-spend")
    try:
        ExecutionShard(count=args.shard_count, index=args.shard_index)
    except ValueError as exc:
        raise DeploymentArtifactError(str(exc)) from exc
    adapter = EvolvingIntentAdapter(args.dataset, expected_sha256=args.dataset_sha256)
    summary = await execute_deployment_run(
        run_id=args.run_id,
        task_manifest_path=args.tasks,
        pass_one_path=args.pass_one,
        threshold_lock_path=args.thresholds,
        schedule_path=args.schedule,
        tasks=adapter.load_tasks(),
        yes_spend=args.yes_spend,
        artifacts_root=args.artifacts,
        environ=_environment(args.env_file),
        max_new_cells=args.max_new_cells,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        evolving_dataset_path=args.dataset,
        evolving_build_receipt_path=args.build_receipt,
    )
    print(
        f"run={args.run_id} phase=deployment visited={summary.visited_cells} "
        f"completed={summary.completed_cells} failed={summary.failed_cells} "
        f"skipped={summary.skipped_cells} shard={summary.shard_index}/"
        f"{summary.shard_count} shard_cells={summary.shard_cells} "
        f"declared={summary.declared_cells}"
    )
    return 0 if summary.failed_cells == 0 else 3


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-evolving")
    run.add_argument("--yes-spend", action="store_true")
    run.add_argument("--run-id", required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--dataset-sha256", required=True)
    run.add_argument("--build-receipt", required=True)
    run.add_argument("--tasks", required=True)
    run.add_argument("--pass-one", required=True)
    run.add_argument("--thresholds", required=True)
    run.add_argument("--schedule", required=True)
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
    "DEPLOYMENT_RUNNER_VERSION",
    "DEPLOYMENT_RUNTIME_CONFIG_VERSION",
    "DEPLOYMENT_SCHEDULE_VERSION",
    "PASS_ONE_VERSION",
    "THRESHOLD_LOCK_VERSION",
    "DEPLOYMENT_SCHEDULE_KIND",
    "PASS_ONE_KIND",
    "THRESHOLD_LOCK_KIND",
    "DEPLOYMENT_SCHEDULE_RECEIPT",
    "PASS_ONE_RECEIPT",
    "THRESHOLD_LOCK_RECEIPT",
    "DeploymentArtifactError",
    "DeploymentEstimand",
    "deployment_runtime_config",
    "LockedMethodThreshold",
    "ThresholdLockArtifact",
    "PassOneCheckpoint",
    "PassOneMethodTrace",
    "PassOneObservationArtifact",
    "FeedbackEvidence",
    "ScheduledTrigger",
    "ScheduledFeedback",
    "DeploymentScheduleGroup",
    "DeploymentScheduleArtifact",
    "load_threshold_lock",
    "freeze_threshold_lock",
    "threshold_lock_from_calibration",
    "load_pass_one_observations",
    "freeze_pass_one_observations",
    "pass_one_trace_from_records",
    "build_pass_one_observation_artifact",
    "build_deployment_schedule",
    "load_deployment_schedule",
    "freeze_deployment_schedule",
    "validate_deployment_schedule",
    "run_deployment_task",
    "execute_deployment_run",
    "deployment_completeness",
    "extract_deployment_outcomes",
    "parser",
    "main",
]
