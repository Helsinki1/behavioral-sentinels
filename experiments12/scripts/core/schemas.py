"""Immutable, versioned records shared by Experiment 12 infrastructure.

The records deliberately contain no API keys, authorization headers, or request
bodies.  Call accounting can therefore be persisted without risking secrets.
Exact experiment transcripts belong in :class:`TrajectoryRecord`; call metadata
contains only identifiers, usage, status, and cost.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import math
import re
from typing import Any, Mapping, TypeAlias


SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class CallStatus(str, Enum):
    """Outcome of one provider request attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TrajectoryStatus(str, Enum):
    """Whether a materialized trajectory may be used in analysis."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _nonnegative_int(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")


def _sha256(name: str, value: str | None) -> None:
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")


def _tuple_of_strings(name: str, value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a tuple of strings")


def _decimal_or_none(name: str, value: Decimal | None) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a finite non-negative Decimal or None")


@dataclass(frozen=True, slots=True)
class PairKey:
    """The indivisible unit on which Experiment 12 arms are paired."""

    model: str
    domain: str
    task_id: str
    replicate_id: int = 0
    task_sha256: str | None = None

    def __post_init__(self) -> None:
        _nonempty("model", self.model)
        _nonempty("domain", self.domain)
        _nonempty("task_id", self.task_id)
        _nonnegative_int("replicate_id", self.replicate_id)
        _sha256("task_sha256", self.task_sha256)

    @property
    def stable_id(self) -> str:
        return f"{self.model}/{self.domain}/{self.task_id}/r{self.replicate_id}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairKey":
        return cls(
            model=str(value["model"]),
            domain=str(value["domain"]),
            task_id=str(value["task_id"]),
            replicate_id=int(value.get("replicate_id", 0)),
            task_sha256=value.get("task_sha256"),
        )


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Raw token counters returned by a provider.

    Cached and reasoning tokens are retained separately and are not added again
    by :attr:`total_tokens`, because providers commonly include them in input or
    output totals already.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    provider_reported_total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "provider_reported_total_tokens",
        ):
            _nonnegative_int(name, getattr(self, name))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "TokenUsage":
        value = value or {}
        return cls(
            input_tokens=int(value.get("input_tokens", 0) or 0),
            output_tokens=int(value.get("output_tokens", 0) or 0),
            cached_input_tokens=int(value.get("cached_input_tokens", 0) or 0),
            reasoning_tokens=int(value.get("reasoning_tokens", 0) or 0),
            provider_reported_total_tokens=(
                None
                if value.get("provider_reported_total_tokens") is None
                else int(value["provider_reported_total_tokens"])
            ),
        )


@dataclass(frozen=True, slots=True)
class CallAttemptRecord:
    """Audit record for exactly one network attempt, including retries."""

    event_id: str
    reservation_id: str
    provider: str
    model: str
    purpose: str
    attempt_number: int
    status: CallStatus
    started_at: str
    finished_at: str
    usage: TokenUsage = TokenUsage()
    estimated_cost_usd: Decimal | None = None
    elapsed_ms: int | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "reservation_id",
            "provider",
            "model",
            "purpose",
            "started_at",
            "finished_at",
        ):
            _nonempty(name, getattr(self, name))
        if isinstance(self.attempt_number, bool) or self.attempt_number < 1:
            raise ValueError("attempt_number must be at least 1")
        if not isinstance(self.status, CallStatus):
            raise ValueError("status must be a CallStatus")
        if not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be TokenUsage")
        _decimal_or_none("estimated_cost_usd", self.estimated_cost_usd)
        _nonnegative_int("elapsed_ms", self.elapsed_ms)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CallAttemptRecord":
        cost = value.get("estimated_cost_usd")
        try:
            parsed_cost = None if cost is None else Decimal(str(cost))
        except InvalidOperation as exc:
            raise ValueError("estimated_cost_usd is not decimal") from exc
        return cls(
            event_id=str(value["event_id"]),
            reservation_id=str(value["reservation_id"]),
            provider=str(value["provider"]),
            model=str(value["model"]),
            purpose=str(value["purpose"]),
            attempt_number=int(value["attempt_number"]),
            status=CallStatus(value["status"]),
            started_at=str(value["started_at"]),
            finished_at=str(value["finished_at"]),
            usage=TokenUsage.from_dict(value.get("usage")),
            estimated_cost_usd=parsed_cost,
            elapsed_ms=value.get("elapsed_ms"),
            provider_request_id=value.get("provider_request_id"),
            finish_reason=value.get("finish_reason"),
            error_type=value.get("error_type"),
            error_message=value.get("error_message"),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """Exact messages and task-visible observations for one agent turn."""

    turn: int
    user_message: str
    assistant_message: str
    call_event_ids: tuple[str, ...]
    hallucination: bool | None = None
    error_codes: tuple[str, ...] = ()
    reset_before_turn: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.turn, bool) or self.turn < 1:
            raise ValueError("turn must be at least 1")
        if not isinstance(self.user_message, str) or not isinstance(self.assistant_message, str):
            raise ValueError("turn messages must be strings")
        _tuple_of_strings("call_event_ids", self.call_event_ids)
        _tuple_of_strings("error_codes", self.error_codes)
        if self.hallucination is not None and not isinstance(self.hallucination, bool):
            raise ValueError("hallucination must be bool or None")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TurnRecord":
        return cls(
            turn=int(value["turn"]),
            user_message=str(value["user_message"]),
            assistant_message=str(value["assistant_message"]),
            call_event_ids=tuple(value.get("call_event_ids", ())),
            hallucination=value.get("hallucination"),
            error_codes=tuple(value.get("error_codes", ())),
            reset_before_turn=bool(value.get("reset_before_turn", False)),
        )


@dataclass(frozen=True, slots=True)
class MonitorRecord:
    """A passive monitor decision tied to one immutable source prefix.

    Outcome labels are intentionally absent.  Analysis joins those later, which
    prevents the monitoring worker from receiving future or ground-truth data.
    """

    monitor_event_id: str
    source_trajectory_sha256: str
    monitor_spec_sha256: str
    checkpoint_turn: int
    observable_after_turn: int
    actionable_before_turn: int
    fired: bool
    call_event_ids: tuple[str, ...] = ()
    score: float | None = None
    raw_output: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty("monitor_event_id", self.monitor_event_id)
        _nonempty("source_trajectory_sha256", self.source_trajectory_sha256)
        _nonempty("monitor_spec_sha256", self.monitor_spec_sha256)
        _sha256("source_trajectory_sha256", self.source_trajectory_sha256)
        _sha256("monitor_spec_sha256", self.monitor_spec_sha256)
        for name in ("checkpoint_turn", "observable_after_turn", "actionable_before_turn"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.observable_after_turn != self.checkpoint_turn:
            raise ValueError("a checkpoint must become observable after its checkpoint turn")
        if self.actionable_before_turn <= self.observable_after_turn:
            raise ValueError("a monitor decision must act strictly after it is observed")
        if not isinstance(self.fired, bool):
            raise ValueError("fired must be bool")
        _tuple_of_strings("call_event_ids", self.call_event_ids)
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
        ):
            raise ValueError("score must be finite numeric data or None")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MonitorRecord":
        return cls(
            monitor_event_id=str(value["monitor_event_id"]),
            source_trajectory_sha256=str(value["source_trajectory_sha256"]),
            monitor_spec_sha256=str(value["monitor_spec_sha256"]),
            checkpoint_turn=int(value["checkpoint_turn"]),
            observable_after_turn=int(value["observable_after_turn"]),
            actionable_before_turn=int(value["actionable_before_turn"]),
            fired=bool(value["fired"]),
            call_event_ids=tuple(value.get("call_event_ids", ())),
            score=value.get("score"),
            raw_output=value.get("raw_output"),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """Versioned materialization of a trajectory's append-only events."""

    run_id: str
    experiment_id: str
    pair_key: PairKey
    arm: str
    system_message: str
    turns: tuple[TurnRecord, ...]
    status: TrajectoryStatus
    started_at: str
    finished_at: str | None = None
    reset_turns: tuple[int, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "experiment_id", "arm", "started_at"):
            _nonempty(name, getattr(self, name))
        if not isinstance(self.pair_key, PairKey):
            raise ValueError("pair_key must be PairKey")
        if not isinstance(self.system_message, str):
            raise ValueError("system_message must be a string")
        if not isinstance(self.turns, tuple) or any(not isinstance(t, TurnRecord) for t in self.turns):
            raise ValueError("turns must be a tuple of TurnRecord values")
        if not isinstance(self.status, TrajectoryStatus):
            raise ValueError("status must be TrajectoryStatus")
        if self.status is TrajectoryStatus.COMPLETE and not self.finished_at:
            raise ValueError("a complete trajectory requires finished_at")
        if self.finished_at is not None and not isinstance(self.finished_at, str):
            raise ValueError("finished_at must be a string or None")
        if not isinstance(self.reset_turns, tuple) or any(
            isinstance(t, bool) or not isinstance(t, int) or t < 1 for t in self.reset_turns
        ):
            raise ValueError("reset_turns must be a tuple of positive integers")
        if tuple(sorted(set(self.reset_turns))) != self.reset_turns:
            raise ValueError("reset_turns must be sorted and unique")
        turn_numbers = tuple(t.turn for t in self.turns)
        if tuple(sorted(set(turn_numbers))) != turn_numbers:
            raise ValueError("turn records must be sorted with unique turn numbers")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        return cls(
            run_id=str(value["run_id"]),
            experiment_id=str(value["experiment_id"]),
            pair_key=PairKey.from_dict(value["pair_key"]),
            arm=str(value["arm"]),
            system_message=str(value["system_message"]),
            turns=tuple(TurnRecord.from_dict(t) for t in value.get("turns", ())),
            status=TrajectoryStatus(value["status"]),
            started_at=str(value["started_at"]),
            finished_at=value.get("finished_at"),
            reset_turns=tuple(int(t) for t in value.get("reset_turns", ())),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


def record_to_dict(value: Any) -> JSONValue:
    """Convert records to plain JSON data without weakening immutability."""

    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: record_to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return record_to_dict(value.value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        return {str(k): record_to_dict(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [record_to_dict(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON-serializable record data")


__all__ = [
    "SCHEMA_VERSION",
    "CallStatus",
    "TrajectoryStatus",
    "PairKey",
    "TokenUsage",
    "CallAttemptRecord",
    "TurnRecord",
    "MonitorRecord",
    "TrajectoryRecord",
    "record_to_dict",
]
