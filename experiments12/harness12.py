"""Budgeted, resumable harness for scripted multi-turn Experiment 12 tasks.

This runner covers Evolving Intent and other scripted text domains. Interactive
tool environments (BFCL) use their official bridge but emit the same trajectory
event vocabulary. Active probes are separate target-model calls whose prompt and
actual response stay in history; clean/passive calls never do.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_json,
)
from experiments12.core.transport import CompletionResult, Transport
from experiments12.domains.base import DomainTask
from experiments12.probes12 import (
    CURRENT_COPY,
    INITIAL_RECALL,
    RECOMPUTE,
    STATEFUL_COUNTER,
    append_carried_probe_exchange,
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.spec12 import ObservationKind, arm as get_arm


HARNESS_VERSION = 2
ARM_TO_PROBE = {
    "active_name_copy": CURRENT_COPY,
    "active_name_recall": INITIAL_RECALL,
    "active_counter": STATEFUL_COUNTER,
    "active_recompute": RECOMPUTE,
}
DEFAULT_REASONING_EFFORT = {
    "gpt-oss-120b": "low",
    "deepseek-v4-flash-0731": "none",
    "qwen3p7-plus": "none",
    "gpt-5.6-luna": "medium",
    "gpt-5.6-terra": "medium",
}


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    checkpoint_every: int = 1
    task_max_output_tokens: int = 1800
    probe_max_output_tokens: int = 192
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.task_max_output_tokens < 1 or self.probe_max_output_tokens < 1:
            raise ValueError("output token ceilings must be positive")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must lie within [0, 2]")


def conservative_input_token_bound(
    messages: Sequence[Mapping[str, Any]],
    *,
    extra_bytes: int = 0,
) -> int:
    """Reserve at least one token per UTF-8 byte plus structural overhead."""

    if extra_bytes < 0:
        raise ValueError("extra_bytes cannot be negative")
    total = 256 + extra_bytes
    for message in messages:
        total += len(str(message.get("role", "")).encode("utf-8"))
        total += len(str(message.get("content", "")).encode("utf-8"))
        total += len(json.dumps(message.get("tool_calls", []), sort_keys=True).encode("utf-8"))
    return total


def _request_key(run_id: str, cell_id: str, kind: str, index: int) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("run_id must be a short, non-secret identifier")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", cell_id):
        raise ValueError("cell_id must be a short, non-secret identifier")
    return f"{run_id}/{cell_id}/{kind}-{index}"


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


def _checkpoint_turns(task: DomainTask, every: int) -> tuple[int, ...]:
    # No post-final probe: it cannot affect or warn about a future task turn.
    return tuple(turn for turn in range(every, len(task.turns), every))


def _extract_numeric_answer(response: str) -> str | None:
    if not response:
        return None
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", response)
    if boxed:
        return boxed[-1].strip()
    patterns = (
        r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$",
        r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return str(matches[-1]).strip().rstrip(".")
    numbers = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", response)
    return numbers[-1] if numbers else None


def _normalized_number(value: str | None) -> Decimal | None:
    if value is None:
        return None
    text = value.strip().replace(",", "").replace("$", "").replace("%", "")
    if "####" in text:
        text = text.rsplit("####", 1)[-1].strip()
    latex_fraction = re.fullmatch(
        r"\s*\\frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}\s*",
        text,
    )
    fraction = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*", text)
    try:
        matched_fraction = latex_fraction or fraction
        if matched_fraction:
            denominator = Decimal(matched_fraction.group(2))
            return (
                None
                if denominator == 0
                else Decimal(matched_fraction.group(1)) / denominator
            )
        return Decimal(text)
    except InvalidOperation:
        # Standard GSM8K evaluation uses the last numeric value in a declared
        # final answer. Models often wrap that value in prose, currency, or
        # LaTeX (for example ``\\boxed{8\\text{ years}}``), so exact-string
        # comparison would turn correct answers into false negatives.
        # Parse embedded fractions before scalar tokens so ``\\frac{1}{2}``
        # is one value, not the misleading pair 1 and 2.  A declared payload
        # containing conflicting values is ambiguous and is not credited.
        candidates: list[Decimal] = []
        fraction_patterns = (
            re.compile(
                r"(?P<sign>[-+]?)\\frac\{\s*(?P<num>\d+(?:\.\d+)?)\s*\}"
                r"\{\s*(?P<den>\d+(?:\.\d+)?)\s*\}"
            ),
            re.compile(
                r"(?<![\d.])(?P<num>[-+]?\d+(?:\.\d+)?)\s*/\s*"
                r"(?P<den>\d+(?:\.\d+)?)(?![\d.])"
            ),
        )
        remainder = text
        for pattern in fraction_patterns:
            def replace_fraction(match: re.Match[str]) -> str:
                denominator = Decimal(match.group("den"))
                if denominator == 0:
                    return " INVALID_FRACTION "
                numerator = Decimal(match.group("num"))
                sign = match.groupdict().get("sign")
                if sign == "-":
                    numerator = -numerator
                candidates.append(numerator / denominator)
                return " "

            remainder = pattern.sub(replace_fraction, remainder)
        numbers = re.findall(r"[-+]?[0-9]+(?:\.[0-9]+)?", remainder)
        candidates.extend(Decimal(number) for number in numbers)
        if not candidates or "INVALID_FRACTION" in remainder:
            return None
        return candidates[0] if all(item == candidates[0] for item in candidates) else None


def grade_final_numeric(response: str, label: str | None) -> tuple[str | None, bool | None]:
    """GSM8K-compatible final grading; None label means no evaluator claim."""

    prediction = _extract_numeric_answer(response)
    if label is None:
        return prediction, None
    predicted_number = _normalized_number(prediction)
    label_number = _normalized_number(label)
    if predicted_number is not None and label_number is not None:
        return prediction, predicted_number == label_number
    return prediction, (prediction or "").strip().lower() == label.strip().lower()


def _rollup(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, Any]] = {}
    resolved_ids: set[str] = set()
    for event in events:
        if event.get("event") not in {"task_turn", "active_probe"}:
            continue
        category = "agent" if event["event"] == "task_turn" else "active_monitor"
        bucket = categories.setdefault(
            category,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "elapsed_ms": 0,
                "accounted_cost_usd": Decimal("0"),
            },
        )
        call = event["call"]
        usage = call["usage"]
        bucket["calls"] += 1
        for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
            bucket[key] += int(usage[key])
        bucket["elapsed_ms"] += int(call["elapsed_ms"])
        bucket["accounted_cost_usd"] += Decimal(call["accounted_cost_usd"])
        resolved_ids.add(call["resolved_model_id"])
    for bucket in categories.values():
        bucket["accounted_cost_usd"] = str(bucket["accounted_cost_usd"])
    return {"by_category": categories, "resolved_model_ids": sorted(resolved_ids)}


def _start_event(
    run_id: str,
    cell_id: str,
    model: str,
    task: DomainTask,
    arm_name: str,
    config: HarnessConfig,
    checkpoint_turns: tuple[int, ...],
) -> dict[str, Any]:
    design = {
        "run_id": run_id,
        "cell_id": cell_id,
        "model": model,
        "task": task.manifest_record(),
        "arm": arm_name,
        "config": {
            "checkpoint_every": config.checkpoint_every,
            "task_max_output_tokens": config.task_max_output_tokens,
            "probe_max_output_tokens": config.probe_max_output_tokens,
            "temperature": config.temperature,
        },
        "checkpoint_turns": list(checkpoint_turns),
        "harness_version": HARNESS_VERSION,
    }
    return {"event": "start", "design_sha256": sha256_json(design), **design}


def _restore(
    event_path: Path,
    expected_start: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Restore exact history from committed events; fail on an orphaned design."""

    events = read_jsonl(event_path)
    if not events or events[0] != expected_start:
        raise ValueError("existing trajectory event log belongs to a different frozen design")
    messages: list[dict[str, Any]] = []
    assistant_task: list[str] = []
    for event in events[1:]:
        if event.get("event") == "task_turn":
            messages.extend((event["user_message"], event["assistant_message"]))
            assistant_task.append(event["assistant_message"]["content"])
        elif event.get("event") == "active_probe":
            messages.extend((event["user_message"], event["assistant_message"]))
        elif event.get("event") == "complete":
            continue
        else:
            raise ValueError("trajectory event log contains an unknown/torn event")
    return events, messages, assistant_task


async def run_scripted_task(
    *,
    run_id: str,
    cell_id: str,
    model: str,
    task: DomainTask,
    arm_name: str,
    transport: Transport,
    event_path: str | Path,
    output_path: str | Path,
    config: HarnessConfig = HarnessConfig(),
) -> dict[str, Any]:
    """Run/resume one declared scripted cell and atomically materialize it."""

    arm = get_arm(arm_name)
    if task.condition == "t1" and arm.observation is ObservationKind.ACTIVE:
        raise ValueError("active observation arms are forbidden for t1 tasks")
    probe_variant = ARM_TO_PROBE.get(arm_name)
    if (arm.probe is None) != (probe_variant is None):
        raise ValueError("arm/probe taxonomy mismatch")
    if model not in DEFAULT_REASONING_EFFORT:
        raise ValueError(f"runtime settings are not frozen for model {model}")
    event_file, output_file = Path(event_path), Path(output_path)
    checkpoint_turns = _checkpoint_turns(task, config.checkpoint_every)
    start = _start_event(run_id, cell_id, model, task, arm_name, config, checkpoint_turns)

    if output_file.exists():
        existing = read_json(output_file)
        if existing.get("design_sha256") != start["design_sha256"]:
            raise ValueError("materialized trajectory belongs to a different design")
        return existing
    if event_file.exists():
        events, messages, assistant_task = _restore(event_file, start)
        if any(event.get("event") == "complete" for event in events):
            raise ValueError("complete event exists without materialized trajectory")
    else:
        event_file.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(event_file, start)
        events, messages, assistant_task = [start], [], []

    setup = None
    if probe_variant is not None and checkpoint_turns:
        setup = render_initial_instruction(probe_variant, task.instance_id, checkpoint_turns)

    completed_turns = len(assistant_task)
    existing_probe_turns = {
        int(event["after_task_turn"])
        for event in events
        if event.get("event") == "active_probe"
    }
    # If a restored task turn was committed immediately before its active probe,
    # finish that probe before advancing to the next benchmark turn.
    for turn_number in range(1, len(task.turns) + 1):
        if turn_number > completed_turns:
            turn = task.turns[turn_number - 1]
            content = turn.user_message
            if turn_number == 1 and setup:
                content = setup + "\n\n--- BENCHMARK MESSAGE ---\n" + content
            user_message = {"role": "user", "content": content}
            messages.append(user_message)
            result = await transport.complete(
                model,
                messages,
                purpose="agent_turn",
                request_key=_request_key(run_id, cell_id, "task", turn_number),
                input_token_estimate=conservative_input_token_bound(messages),
                max_output_tokens=config.task_max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=DEFAULT_REASONING_EFFORT[model],
            )
            if result.tool_calls:
                raise ValueError("scripted text task unexpectedly returned tool calls")
            assistant_message = {"role": "assistant", "content": result.text}
            messages.append(assistant_message)
            assistant_task.append(result.text)
            event = {
                "event": "task_turn",
                "task_turn": turn_number,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "call": _call_record(result),
            }
            append_jsonl(event_file, event)
            events.append(event)
            completed_turns += 1

        if (
            probe_variant is not None
            and turn_number in checkpoint_turns
            and turn_number not in existing_probe_turns
        ):
            checkpoint_index = checkpoint_turns.index(turn_number) + 1
            instance = generate_probe_instance(probe_variant, task.instance_id, checkpoint_index)
            prompt = render_probe_prompt(instance)
            probe_user = {"role": "user", "content": prompt}
            messages.append(probe_user)
            result = await transport.complete(
                model,
                messages,
                purpose="active_probe",
                request_key=_request_key(run_id, cell_id, "probe", checkpoint_index),
                input_token_estimate=conservative_input_token_bound(messages),
                max_output_tokens=config.probe_max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=DEFAULT_REASONING_EFFORT[model],
            )
            if result.tool_calls:
                raise ValueError("active text probe unexpectedly returned tool calls")
            probe_assistant = {"role": "assistant", "content": result.text}
            messages.append(probe_assistant)
            grade = grade_probe_response(instance, result.text)
            event = {
                "event": "active_probe",
                "after_task_turn": turn_number,
                "checkpoint_index": checkpoint_index,
                "variant": probe_variant,
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
                # Exact target-visible history through the carried probe
                # response. Deployment pass one consumes this content address,
                # never task gold or the final task outcome.
                "source_prefix_sha256": sha256_json(messages),
            }
            append_jsonl(event_file, event)
            events.append(event)
            existing_probe_turns.add(turn_number)

    final_response = assistant_task[-1]
    prediction, success = grade_final_numeric(final_response, task.evaluation_label)
    transcript_sha256 = sha256_json(messages)
    complete_event = {
        "event": "complete",
        "task_turns": len(assistant_task),
        "transcript_sha256": transcript_sha256,
        "prediction": prediction,
        "success": success,
    }
    events.append(complete_event)
    materialized = {
        "schema_version": 1,
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "cell_id": cell_id,
        "design_sha256": start["design_sha256"],
        "model": model,
        "domain": task.domain,
        "task_id": task.task_id,
        "condition": task.condition,
        "task_sha256": task.task_sha256,
        "arm": arm_name,
        "active_probe_variant": probe_variant,
        "checkpoint_turns": list(checkpoint_turns),
        "messages": messages,
        "task_assistant_messages": assistant_task,
        "task_records": [event for event in events if event.get("event") == "task_turn"],
        "probe_records": [event for event in events if event.get("event") == "active_probe"],
        "evaluation": {
            "prediction": prediction,
            "evaluation_label_sha256": (
                None if task.evaluation_label is None else sha256_json(task.evaluation_label)
            ),
            "success": success,
        },
        "transcript_sha256": transcript_sha256,
        "accounting": _rollup(events),
        "complete": True,
    }
    atomic_write_json(output_file, materialized)
    # Materialize first: a crash between these writes leaves a usable, hashed
    # trajectory rather than a misleading "complete" event with no artifact.
    append_jsonl(event_file, complete_event)
    return materialized
