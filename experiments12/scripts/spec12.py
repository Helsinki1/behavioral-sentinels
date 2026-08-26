"""Immutable scientific specification for Experiment 12.

Runtime tuning belongs in a run manifest.  The names and primary estimands here
are deliberately boring constants so analysis cannot infer a new design from
whichever artifact files happen to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class ObservationKind(str, Enum):
    NONE = "none"
    ACTIVE = "active_carry"
    PASSIVE = "passive_zero_carry"
    BASELINE = "nonadaptive_baseline"


class Stage(str, Enum):
    OFFLINE = "offline"
    SMOKE = "smoke"
    BASELINE_GATE = "baseline_gate"
    CALIBRATION = "calibration"
    CONFIRMATORY = "confirmatory"


class Benchmark(str, Enum):
    EVOLVING_GSM8K = "evolving_intent_gsm8k"
    BFCL_MULTI_TURN = "bfcl_multi_turn"
    TURNBENCH_CLASSIC = "turnbench_ms_classic"


class Operator(str, Enum):
    NONE = "none"
    COMPACT = "lossy_compaction"
    REGROUND = "public_state_reground"
    FEEDBACK = "good_bad_watch_feedback"


@dataclass(frozen=True)
class Arm:
    name: str
    observation: ObservationKind
    probe: str | None
    confirmatory_core: bool


@dataclass(frozen=True)
class Monitor:
    name: str
    observation: ObservationKind
    sees_gold: bool
    enters_target_history: bool


ARMS: Final[tuple[Arm, ...]] = (
    Arm("clean", ObservationKind.NONE, None, True),
    Arm("active_name_copy", ObservationKind.ACTIVE, "name_copy", False),
    Arm("active_name_recall", ObservationKind.ACTIVE, "name_recall", False),
    Arm("active_counter", ObservationKind.ACTIVE, "counter", False),
    Arm("active_recompute", ObservationKind.ACTIVE, "recompute", True),
)

MONITORS: Final[tuple[Monitor, ...]] = (
    Monitor("turn_clock", ObservationKind.BASELINE, False, False),
    Monitor("context_use", ObservationKind.BASELINE, False, False),
    Monitor("frozen_probe", ObservationKind.PASSIVE, False, False),
    Monitor("frozen_quiz", ObservationKind.PASSIVE, False, False),
    Monitor("trace_judge", ObservationKind.PASSIVE, False, False),
    Monitor("trace_rules", ObservationKind.PASSIVE, False, False),
)

PRIMARY_OPERATORS: Final[tuple[Operator, ...]] = (
    Operator.COMPACT,
    Operator.REGROUND,
)

SECONDARY_OPERATORS: Final[tuple[Operator, ...]] = (Operator.FEEDBACK,)

TARGET_MODELS: Final[tuple[str, ...]] = (
    "gpt-oss-120b",
    "deepseek-v4-flash-0731",
    "qwen3p7-plus",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
)

CORE_BENCHMARKS: Final[tuple[Benchmark, ...]] = (
    Benchmark.EVOLVING_GSM8K,
    Benchmark.BFCL_MULTI_TURN,
)

PERMISSION_GATED_BENCHMARKS: Final[tuple[Benchmark, ...]] = (
    Benchmark.TURNBENCH_CLASSIC,
)

# A checkpoint produced after turn t cannot be an early warning for failure t.
PRIMARY_EVENT_OFFSET: Final[int] = 1
FEEDBACK_MAX_TOKENS: Final[int] = 80

HARD_PROVIDER_USD: Final[dict[str, float]] = {
    "openai": 500.0,
    "fireworks": 30.0,
}
OPERATIONAL_PROVIDER_USD: Final[dict[str, float]] = {
    "openai": 400.0,
    "fireworks": 24.0,
}

STAGE_PROVIDER_USD: Final[dict[Stage, dict[str, float]]] = {
    Stage.OFFLINE: {"openai": 0.0, "fireworks": 0.0},
    Stage.SMOKE: {"openai": 25.0, "fireworks": 5.0},
    Stage.BASELINE_GATE: {"openai": 50.0, "fireworks": 5.0},
    Stage.CALIBRATION: {"openai": 125.0, "fireworks": 7.0},
    Stage.CONFIRMATORY: {"openai": 200.0, "fireworks": 7.0},
}


def arm(name: str) -> Arm:
    matches = [item for item in ARMS if item.name == name]
    if len(matches) != 1:
        raise KeyError(f"unknown Experiment 12 arm: {name}")
    return matches[0]


def validate_spec() -> None:
    """Fail early if a future edit corrupts the frozen taxonomy."""

    arm_names = [item.name for item in ARMS]
    monitor_names = [item.name for item in MONITORS]
    if len(arm_names) != len(set(arm_names)):
        raise ValueError("duplicate arm name")
    if len(monitor_names) != len(set(monitor_names)):
        raise ValueError("duplicate monitor name")
    if any(item.observation is ObservationKind.PASSIVE for item in ARMS):
        raise ValueError("passive monitors must not be trajectory arms")
    if any(item.enters_target_history for item in MONITORS):
        raise ValueError("a zero-carry monitor cannot enter target history")
    if any(item.sees_gold for item in MONITORS):
        raise ValueError("observers cannot receive benchmark gold")
    if set(STAGE_PROVIDER_USD[Stage.OFFLINE].values()) != {0.0}:
        raise ValueError("offline stage must make no paid calls")
    for stage, limits in STAGE_PROVIDER_USD.items():
        for provider, amount in limits.items():
            if amount > OPERATIONAL_PROVIDER_USD[provider]:
                raise ValueError(f"{stage}: stage limit exceeds operational stop")
    for provider, operational in OPERATIONAL_PROVIDER_USD.items():
        if operational >= HARD_PROVIDER_USD[provider]:
            raise ValueError("operational stop must leave a hard-cap reserve")


validate_spec()
