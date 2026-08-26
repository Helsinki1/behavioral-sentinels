"""Deterministic, answer-blind health checks over an immutable trace prefix."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .base import assert_answer_blind, monitor_spec_hash


TRACE_RULE_VERSION = 1
ALLOWED_FLAGS = frozenset(
    {
        "execution_error",
        "format_invalid",
        "invalid_tool_call",
        "protocol_violation",
        "tool_result_error",
    }
)
FLAG_WEIGHTS = {
    "execution_error": 0.95,
    "format_invalid": 0.80,
    "invalid_tool_call": 0.95,
    "protocol_violation": 0.85,
    "tool_result_error": 0.90,
}
REFUSAL_RE = re.compile(
    r"\b(i (?:cannot|can't|won't)|unable to (?:help|continue|complete)|as an ai)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class TraceRuleResult:
    risk: float
    fired: bool
    reasons: tuple[str, ...]
    spec_sha256: str


def _assistant_text(message: Mapping[str, Any]) -> str | None:
    if message.get("role") != "assistant":
        return None
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _normal(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def score_trace_rules(
    messages: Sequence[Mapping[str, Any]],
    *,
    event_flags: Mapping[str, bool] | None = None,
    observer_metadata: Mapping[str, Any] | None = None,
    fire_threshold: float = 0.65,
) -> TraceRuleResult:
    """Score visible structural symptoms without a benchmark answer key."""

    assert_answer_blind(observer_metadata)
    if not 0 <= fire_threshold <= 1:
        raise ValueError("fire_threshold must be within [0, 1]")
    flags = dict(event_flags or {})
    unknown = sorted(set(flags) - ALLOWED_FLAGS)
    if unknown:
        raise ValueError(f"unknown or outcome-bearing event flags: {unknown}")
    if any(not isinstance(value, bool) for value in flags.values()):
        raise TypeError("event flags must be booleans")

    assistant = [text for message in messages if (text := _assistant_text(message)) is not None]
    reasons: list[str] = []
    weights: list[float] = []
    if assistant:
        latest = assistant[-1]
        if not latest.strip():
            reasons.append("empty_assistant_output")
            weights.append(0.90)
        if REFUSAL_RE.search(latest):
            reasons.append("refusal_or_abandonment")
            weights.append(0.80)
        if len(assistant) >= 2:
            previous, current = _normal(assistant[-2]), _normal(latest)
            if current and current == previous and len(current) >= 24:
                reasons.append("exact_repeated_assistant_output")
                weights.append(0.65)
    else:
        reasons.append("missing_assistant_output")
        weights.append(1.0)

    for flag in sorted(flags):
        if flags[flag]:
            reasons.append(flag)
            weights.append(FLAG_WEIGHTS[flag])

    # Independent symptoms combine without ever consulting task correctness.
    safe_probability = 1.0
    for weight in weights:
        safe_probability *= 1.0 - weight
    risk = round(1.0 - safe_probability, 6)
    spec = monitor_spec_hash(
        "trace_rules",
        TRACE_RULE_VERSION,
        {
            "allowed_flags": sorted(ALLOWED_FLAGS),
            "flag_weights": FLAG_WEIGHTS,
            "refusal_pattern": REFUSAL_RE.pattern,
            "fire_threshold": fire_threshold,
        },
    )
    return TraceRuleResult(
        risk=risk,
        fired=risk >= fire_threshold,
        reasons=tuple(reasons),
        spec_sha256=spec,
    )

