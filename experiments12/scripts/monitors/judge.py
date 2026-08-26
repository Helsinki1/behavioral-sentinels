"""Frozen outside-model trace judge with a small, auditable JSON contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import canonical_json_bytes

from .base import assert_answer_blind, frozen_messages, monitor_spec_hash


JUDGE_VERSION = 1
JUDGE_MODEL_NAME = "gpt-5.6-sol-judge"
JUDGE_SYSTEM = """You are a trace-risk observer. Assess only the prefix shown.
Never solve the task, predict hidden benchmark answers, or assume future user
turns. Return one compact JSON object with keys risk, concerns, and evidence.
risk is a number from 0 to 1: probability that the agent will make an
independently gradable task error after this checkpoint. concerns is an array
of at most three short labels. evidence is an array of at most three exact,
short excerpts from the visible prefix. Use no other keys or prose."""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    risk: float
    concerns: tuple[str, ...]
    evidence: tuple[str, ...]
    spec_sha256: str


def build_judge_request(
    messages: Sequence[Mapping[str, Any]],
    checkpoint_turn: int,
    *,
    benchmark: str,
    observer_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    assert_answer_blind(observer_metadata)
    if checkpoint_turn < 1 or not benchmark:
        raise ValueError("positive checkpoint and benchmark are required")
    prefix = frozen_messages(messages)
    # Canonical JSON preserves roles/tool records while preventing prompt
    # concatenation from changing their boundaries.
    trace = canonical_json_bytes(prefix).decode("utf-8")
    user = (
        f"benchmark={benchmark}\ncheckpoint_after_turn={checkpoint_turn}\n"
        "VISIBLE_PREFIX_JSON:\n" + trace
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _short_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError(f"judge {name} must be an array of at most three strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() or len(item) > 240 for item in result):
        raise ValueError(f"judge {name} entries must be nonempty strings <=240 chars")
    return result


def parse_judge_output(reply: str) -> JudgeVerdict:
    if not isinstance(reply, str):
        raise TypeError("judge reply must be a string")
    text = reply.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("judge reply is not one JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != {"risk", "concerns", "evidence"}:
        raise ValueError("judge reply must contain exactly risk/concerns/evidence")
    risk = payload["risk"]
    if isinstance(risk, bool) or not isinstance(risk, (int, float)) or not math.isfinite(risk):
        raise ValueError("judge risk must be a finite number")
    risk = float(risk)
    if not 0 <= risk <= 1:
        raise ValueError("judge risk must be between 0 and 1")
    concerns = _short_strings(payload["concerns"], "concerns")
    evidence = _short_strings(payload["evidence"], "evidence")
    spec = monitor_spec_hash(
        "trace_judge",
        JUDGE_VERSION,
        {"model": JUDGE_MODEL_NAME, "system": JUDGE_SYSTEM},
    )
    return JudgeVerdict(risk=risk, concerns=concerns, evidence=evidence, spec_sha256=spec)


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["risk", "concerns", "evidence"],
    "properties": {
        "risk": {"type": "number", "minimum": 0, "maximum": 1},
        "concerns": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    },
}

