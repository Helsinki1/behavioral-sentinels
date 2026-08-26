"""Native-tool runner for pinned BFCL V4 multi-turn episodes.

The official bridge owns tool execution and final correctness.  This module
only carries public user, assistant, and tool-result messages through the
target model.  It deliberately refuses to resume a partial cell: a paid call
may already have reached a provider even when the trajectory turn was not yet
materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_json,
)
from experiments12.core.transport import CompletionResult, JsonSchemaTool, ToolCall
from experiments12.domains.bfcl import (
    DOMAIN,
    V4_MULTI_TURN_CATEGORIES,
    BFCLOfficialEpisodeEvaluation,
    BFCLPublicState,
    BFCLStartedEpisode,
    BFCLTaskRecord,
    BFCLTurnExecution,
    StateCheckStatus,
)
from experiments12.harness12 import (
    ARM_TO_PROBE,
    DEFAULT_REASONING_EFFORT,
    _call_record,
    _rollup,
    conservative_input_token_bound,
)
from experiments12.passive_quizzes12 import generate_bfcl_passive_quiz
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.runner12 import freeze_task_manifest
from experiments12.spec12 import arm as get_arm


BFCL_RUNNER_VERSION = 3
BFCL_CONDITION = "official_native_tools"
MODEL_FINAL_TERMINATION = "model_final_response"
TOOL_BATCH_CAP_TERMINATION = "tool_batch_limit_exhausted"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SAFE_CELL_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")


class BFCLBridge(Protocol):
    """Small bridge surface consumed by the runner and synthetic tests."""

    def begin_episode(self, episode_id: str, task_id: str) -> BFCLStartedEpisode: ...

    def execute_tools(
        self,
        episode_id: str,
        task_id: str,
        turn_index: int,
        tool_calls: Sequence[ToolCall],
    ) -> BFCLTurnExecution: ...

    def materialize_public_state(
        self,
        episode_id: str,
        task_id: str,
        after_turn: int,
    ) -> BFCLPublicState: ...

    def evaluate_episode(
        self,
        episode_id: str,
        task_id: str,
    ) -> BFCLOfficialEpisodeEvaluation: ...


class BFCLTransport(Protocol):
    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult: ...


@dataclass(frozen=True, slots=True)
class BFCLRunnerConfig:
    max_tool_batches_per_turn: int = 12
    task_max_output_tokens: int = 1800
    probe_max_output_tokens: int = 192
    temperature: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_tool_batches_per_turn",
            "task_max_output_tokens",
            "probe_max_output_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must lie within [0, 2]")


def _request_key(run_id: str, cell_id: str, kind: str, *indices: int) -> str:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be a short safe identifier")
    if _SAFE_CELL_ID.fullmatch(cell_id) is None:
        raise ValueError("cell_id must be a short safe identifier")
    if not kind or any(isinstance(index, bool) or index < 1 for index in indices):
        raise ValueError("request-key kind/indices are invalid")
    suffix = "-".join(str(index) for index in indices)
    return f"{run_id}/{cell_id}/{kind}-{suffix}"


def _episode_id(run_id: str, cell_id: str) -> str:
    _request_key(run_id, cell_id, "episode", 1)
    return f"{run_id}:{cell_id}"


def _tool_call_message(result: CompletionResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "name": call.name,
                "arguments": call.arguments_json,
            }
            for call in result.tool_calls
        ]
    return message


def _tool_schema_bytes(tools: Sequence[JsonSchemaTool]) -> int:
    return len(
        canonical_json_bytes(
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.schema,
                    "strict": tool.strict,
                }
                for tool in tools
            ]
        )
    )


def _aggregate_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not calls:
        raise ValueError("cannot aggregate an empty call list")
    event_ids: list[str] = []
    resolved_ids: set[str] = set()
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    elapsed_ms = 0
    cost = Decimal("0")
    for call in calls:
        ids = call.get("call_event_ids")
        call_usage = call.get("usage")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or not isinstance(call_usage, Mapping)
        ):
            raise ValueError("per-call accounting record is incomplete")
        event_ids.extend(ids)
        resolved = call.get("resolved_model_id")
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("per-call accounting lacks resolved_model_id")
        resolved_ids.add(resolved)
        for key in usage:
            value = call_usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"per-call {key} is invalid")
            usage[key] += value
        elapsed = call.get("elapsed_ms")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise ValueError("per-call elapsed_ms is invalid")
        elapsed_ms += elapsed
        cost += Decimal(str(call.get("accounted_cost_usd")))
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("call event IDs repeat within one task turn")
    final = calls[-1]
    return {
        "call_event_ids": event_ids,
        "resolved_model_id": (
            next(iter(resolved_ids)) if len(resolved_ids) == 1 else "+".join(sorted(resolved_ids))
        ),
        "response_id": final.get("response_id"),
        "request_id": final.get("request_id"),
        "finish_reason": final.get("finish_reason"),
        "usage": usage,
        "accounted_cost_usd": str(cost),
        "elapsed_ms": elapsed_ms,
    }


def _execution_record(
    execution: BFCLTurnExecution,
    tool_calls: Sequence[ToolCall],
    *,
    batch_index: int | None,
    finalization: bool,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "finalization": finalization,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments_json": call.arguments_json,
            }
            for call in tool_calls
        ],
        "results": [
            {
                "call_id": result.call_id,
                "name": result.name,
                "status": result.status.value,
                "output_json": result.output_json,
            }
            for result in execution.results
        ],
        "state_check": execution.state_check.value,
    }


def validate_bfcl_task_turn_record(
    record: Mapping[str, Any],
    *,
    max_tool_batches_per_turn: int,
) -> None:
    """Validate the exact normal-or-capped BFCL turn materialization."""

    if (
        isinstance(max_tool_batches_per_turn, bool)
        or not isinstance(max_tool_batches_per_turn, int)
        or max_tool_batches_per_turn < 1
    ):
        raise ValueError("max_tool_batches_per_turn must be a positive integer")
    if record.get("event") != "task_turn":
        raise ValueError("BFCL turn record must be a task_turn event")
    calls = record.get("calls")
    executions = record.get("tool_executions")
    messages = record.get("messages")
    assistant = record.get("assistant_message")
    capped = record.get("capped")
    if (
        not isinstance(calls, list)
        or not calls
        or len(calls) > max_tool_batches_per_turn
        or not isinstance(executions, list)
        or not executions
        or not isinstance(messages, list)
        or not messages
        or not isinstance(assistant, Mapping)
        or assistant.get("role") != "assistant"
        or not isinstance(assistant.get("content"), str)
        or not isinstance(capped, bool)
        or record.get("tool_batch_limit") != max_tool_batches_per_turn
    ):
        raise ValueError("BFCL turn record has an invalid bounded-agent schema")

    termination_reason = record.get("termination_reason")
    expected_reason = (
        TOOL_BATCH_CAP_TERMINATION if capped else MODEL_FINAL_TERMINATION
    )
    if termination_reason != expected_reason:
        raise ValueError("BFCL turn termination reason disagrees with capped status")
    actual_assistant_positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if (
        len(actual_assistant_positions) != len(calls)
        or messages[actual_assistant_positions[-1]] != assistant
    ):
        raise ValueError("BFCL assistant_message must be the last actual model message")

    finalization = executions[-1]
    if (
        not isinstance(finalization, Mapping)
        or finalization.get("finalization") is not True
        or finalization.get("tool_calls") != []
        or finalization.get("results") != []
        or not isinstance(finalization.get("state_check"), str)
    ):
        raise ValueError("BFCL turn lacks an explicit empty bridge finalization")

    tool_executions = executions[:-1]
    if any(
        not isinstance(execution, Mapping)
        or execution.get("batch_index") != index
        or execution.get("finalization") is not False
        or not isinstance(execution.get("tool_calls"), list)
        or not execution.get("tool_calls")
        or not isinstance(execution.get("results"), list)
        or not isinstance(execution.get("state_check"), str)
        for index, execution in enumerate(tool_executions, 1)
    ):
        raise ValueError("BFCL tool-batch execution sequence is invalid")
    for execution in tool_executions:
        execution_calls = execution["tool_calls"]
        execution_results = execution["results"]
        if any(
            not isinstance(call, Mapping)
            or not isinstance(call.get("call_id"), str)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments_json"), str)
            for call in execution_calls
        ) or any(
            not isinstance(result, Mapping)
            or not isinstance(result.get("call_id"), str)
            or not isinstance(result.get("name"), str)
            or not isinstance(result.get("output_json"), str)
            for result in execution_results
        ):
            raise ValueError("BFCL tool-batch result materialization is invalid")
        if [
            (call.get("call_id"), call.get("name")) for call in execution_calls
        ] != [
            (result.get("call_id"), result.get("name"))
            for result in execution_results
        ]:
            raise ValueError("BFCL tool calls and results do not correspond")

    if capped:
        last_tool_execution = tool_executions[-1] if tool_executions else None
        assistant_tool_calls = assistant.get("tool_calls")
        normalized_assistant_calls = (
            [
                {
                    "call_id": call.get("id"),
                    "name": call.get("name"),
                    "arguments_json": call.get("arguments"),
                }
                for call in assistant_tool_calls
            ]
            if isinstance(assistant_tool_calls, list)
            and all(isinstance(call, Mapping) for call in assistant_tool_calls)
            else None
        )
        expected_trailing_tool_messages = (
            [
                {
                    "role": "tool",
                    "tool_call_id": result.get("call_id"),
                    "content": result.get("output_json"),
                }
                for result in last_tool_execution.get("results", ())
            ]
            if isinstance(last_tool_execution, Mapping)
            else []
        )
        if (
            len(calls) != max_tool_batches_per_turn
            or len(tool_executions) != max_tool_batches_per_turn
            or finalization.get("batch_index") is not None
            or not isinstance(assistant_tool_calls, list)
            or not assistant_tool_calls
            or normalized_assistant_calls != last_tool_execution.get("tool_calls")
            or messages[actual_assistant_positions[-1] + 1 :]
            != expected_trailing_tool_messages
        ):
            raise ValueError("BFCL capped turn is not an exact tool-batch exhaustion")
    elif (
        len(tool_executions) != len(calls) - 1
        or finalization.get("batch_index") != len(calls)
        or "tool_calls" in assistant
        or messages[-1] != assistant
    ):
        raise ValueError("BFCL normally completed turn lacks a final model response")


def _failure_indicators(executions: Sequence[BFCLTurnExecution]) -> dict[str, bool]:
    indicators = [execution.failure_indicators for execution in executions]
    return {
        "invalid_call_observed": any(item.invalid_call_observed for item in indicators),
        "execution_failure_observed": any(
            item.execution_failure_observed for item in indicators
        ),
        "state_check_failure_observed": any(
            item.state_check_failure_observed for item in indicators
        ),
        "state_check_available": any(item.state_check_available for item in indicators),
    }


def _task_provenance(task: BFCLTaskRecord) -> dict[str, Any]:
    return {
        "domain": DOMAIN,
        "task_id": task.task_id,
        "condition": BFCL_CONDITION,
        "category": task.category,
        "num_turns": len(task.turns),
        "source_sha256": task.source_sha256,
        "task_sha256": task.task_sha256,
        "evaluation_label_sha256": None,
        "public_metadata": {
            "bfcl_category": task.category,
            "tool_interface": "native",
        },
    }


def _start_event(
    *,
    run_id: str,
    cell_id: str,
    model: str,
    task: BFCLTaskRecord,
    arm_name: str,
    checkpoint_turns: Sequence[int],
    config: BFCLRunnerConfig,
) -> dict[str, Any]:
    design = {
        "run_id": run_id,
        "cell_id": cell_id,
        "model": model,
        "task": _task_provenance(task),
        "arm": arm_name,
        "config": {
            "max_tool_batches_per_turn": config.max_tool_batches_per_turn,
            "task_max_output_tokens": config.task_max_output_tokens,
            "probe_max_output_tokens": config.probe_max_output_tokens,
            "temperature": config.temperature,
        },
        "checkpoint_turns": list(checkpoint_turns),
        "bfcl_runner_version": BFCL_RUNNER_VERSION,
    }
    return {"event": "start", "design_sha256": sha256_json(design), **design}


def _completed_output(
    output_path: Path,
    event_path: Path,
    expected_start: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if not output_path.exists() and not event_path.exists():
        return None
    if not output_path.is_file() or not event_path.is_file():
        raise FileExistsError("BFCL cell is partial; audit it before any retry")
    output = read_json(output_path)
    events = read_jsonl(event_path)
    valid_event_sequence = all(
        isinstance(event, Mapping)
        and event.get("event") in {"start", "task_turn", "active_probe", "complete"}
        for event in events
    )
    task_events = [
        event for event in events if isinstance(event, Mapping) and event.get("event") == "task_turn"
    ]
    probe_events = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "active_probe"
    ]
    max_batches = expected_start.get("config", {}).get("max_tool_batches_per_turn")
    task_count = expected_start.get("task", {}).get("num_turns")
    try:
        if (
            isinstance(max_batches, bool)
            or not isinstance(max_batches, int)
            or max_batches < 1
        ):
            raise ValueError("start event lacks the bounded-agent limit")
        for task_event in task_events:
            validate_bfcl_task_turn_record(
                task_event,
                max_tool_batches_per_turn=max_batches,
            )
    except ValueError as exc:
        raise FileExistsError(
            "BFCL completed cell has an invalid bounded-agent turn record"
        ) from exc
    materialized_messages: list[Any] = []
    for event in events[1:-1]:
        if not isinstance(event, Mapping):
            continue
        if event.get("event") == "task_turn":
            materialized_messages.extend(event.get("messages", ()))
        elif event.get("event") == "active_probe":
            materialized_messages.extend(
                (event.get("user_message"), event.get("assistant_message"))
            )
    expected_assistant_messages = [
        event["assistant_message"]["content"] for event in task_events
    ]
    if (
        not isinstance(output, Mapping)
        or output.get("complete") is not True
        or output.get("bfcl_runner_version") != BFCL_RUNNER_VERSION
        or output.get("design_sha256") != expected_start.get("design_sha256")
        or not events
        or not valid_event_sequence
        or events[0] != expected_start
        or sum(event.get("event") == "start" for event in events) != 1
        or sum(event.get("event") == "complete" for event in events) != 1
        or not isinstance(events[-1], Mapping)
        or events[-1].get("event") != "complete"
        or events[-1].get("transcript_sha256") != output.get("transcript_sha256")
        or events[-1].get("evaluation_sha256") != sha256_json(output.get("evaluation"))
        or len(task_events) != task_count
        or task_events != output.get("task_records")
        or probe_events != output.get("probe_records")
        or materialized_messages != output.get("messages")
        or output.get("task_assistant_messages") != expected_assistant_messages
        or sha256_json(output.get("messages")) != output.get("transcript_sha256")
    ):
        raise FileExistsError("BFCL cell artifacts are partial or differ from the frozen design")
    return output


async def run_bfcl_task(
    *,
    run_id: str,
    cell_id: str,
    model: str,
    task: BFCLTaskRecord,
    arm_name: str,
    bridge: BFCLBridge,
    transport: BFCLTransport,
    event_path: str | Path,
    output_path: str | Path,
    config: BFCLRunnerConfig = BFCLRunnerConfig(),
) -> dict[str, Any]:
    """Run one declared BFCL cell, refusing all implicit partial resumes."""

    if not isinstance(task, BFCLTaskRecord):
        raise TypeError("task must be a BFCLTaskRecord")
    arm = get_arm(arm_name)
    probe_variant = ARM_TO_PROBE.get(arm_name)
    if (arm.probe is None) != (probe_variant is None):
        raise ValueError("arm/probe taxonomy mismatch")
    if model not in DEFAULT_REASONING_EFFORT:
        raise ValueError(f"runtime settings are not frozen for model {model}")
    checkpoints = tuple(range(1, len(task.turns)))
    start = _start_event(
        run_id=run_id,
        cell_id=cell_id,
        model=model,
        task=task,
        arm_name=arm_name,
        checkpoint_turns=checkpoints,
        config=config,
    )
    event_file, output_file = Path(event_path), Path(output_path)
    existing = _completed_output(output_file, event_file, start)
    if existing is not None:
        return dict(existing)
    event_file.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(event_file, start)

    episode_id = _episode_id(run_id, cell_id)
    started = bridge.begin_episode(episode_id, task.task_id)
    if started.episode_id != episode_id or started.task_id != task.task_id:
        raise ValueError("official bridge began a different BFCL episode")

    task_instance_id = f"{DOMAIN}/{task.task_id}/{BFCL_CONDITION}"
    setup = (
        None
        if probe_variant is None or not checkpoints
        else render_initial_instruction(
            probe_variant,
            task_instance_id,
            tuple(range(1, len(checkpoints) + 1)),
        )
    )
    history: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    final_assistant_messages: list[str] = []

    for turn in task.turns:
        turn_start = len(history)
        user_content = turn.user_message
        if turn.index == 1 and setup is not None:
            user_content = setup + "\n\n--- BFCL USER MESSAGE ---\n" + user_content
        user_message = {"role": "user", "content": user_content}
        history.append(user_message)
        calls: list[dict[str, Any]] = []
        executions: list[BFCLTurnExecution] = []
        execution_records: list[dict[str, Any]] = []
        seen_tool_call_ids: set[str] = set()
        final_assistant: dict[str, Any] | None = None
        capped = False

        for batch_index in range(1, config.max_tool_batches_per_turn + 1):
            result = await transport.complete(
                model,
                history,
                purpose="agent_turn",
                request_key=_request_key(
                    run_id, cell_id, "bfcl-task", turn.index, batch_index
                ),
                input_token_estimate=conservative_input_token_bound(
                    history,
                    extra_bytes=_tool_schema_bytes(turn.tools),
                ),
                max_output_tokens=config.task_max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=DEFAULT_REASONING_EFFORT[model],
                tools=turn.tools,
                tool_choice="auto" if turn.tools else None,
            )
            call = _call_record(result)
            calls.append(call)
            assistant_message = _tool_call_message(result)
            history.append(assistant_message)
            if result.tool_calls:
                current_ids = {item.call_id for item in result.tool_calls}
                if len(current_ids) != len(result.tool_calls) or seen_tool_call_ids & current_ids:
                    raise ValueError("BFCL tool-call IDs must be unique across one user turn")
                seen_tool_call_ids.update(current_ids)
                execution = bridge.execute_tools(
                    episode_id,
                    task.task_id,
                    turn.index,
                    result.tool_calls,
                )
                executions.append(execution)
                execution_records.append(
                    _execution_record(
                        execution,
                        result.tool_calls,
                        batch_index=batch_index,
                        finalization=False,
                    )
                )
                history.extend(dict(message) for message in execution.tool_messages())
                continue

            final_assistant = assistant_message
            break
        else:
            # The last actual target response is a tool-call message.  Preserve
            # it exactly, close the official bridge turn with no invented model
            # text, and treat the bounded-agent cap as an auditable outcome.
            final_assistant = assistant_message
            capped = True

        if final_assistant is None:
            raise RuntimeError("BFCL turn ended without a final assistant response")
        finalization = bridge.execute_tools(
            episode_id,
            task.task_id,
            turn.index,
            (),
        )
        if finalization.results:
            raise ValueError("empty BFCL turn-finalization call returned tool results")
        executions.append(finalization)
        execution_records.append(
            _execution_record(
                finalization,
                (),
                batch_index=None if capped else batch_index,
                finalization=True,
            )
        )
        public_state = bridge.materialize_public_state(
            episode_id,
            task.task_id,
            turn.index,
        )
        turn_messages = [dict(message) for message in history[turn_start:]]
        tool_results = [
            result
            for execution in execution_records
            for result in execution["results"]
        ]
        record = {
            "event": "task_turn",
            "task_turn": turn.index,
            "user_message": dict(user_message),
            "assistant_message": dict(final_assistant),
            "capped": capped,
            "termination_reason": (
                TOOL_BATCH_CAP_TERMINATION if capped else MODEL_FINAL_TERMINATION
            ),
            "tool_batch_limit": config.max_tool_batches_per_turn,
            "messages": turn_messages,
            "calls": calls,
            "call": _aggregate_calls(calls),
            "tool_executions": execution_records,
            "tool_results": tool_results,
            "failure_indicators": _failure_indicators(executions),
            "public_state_json": public_state.state_json,
            "public_state_sha256": public_state.state_sha256,
        }
        validate_bfcl_task_turn_record(
            record,
            max_tool_batches_per_turn=config.max_tool_batches_per_turn,
        )
        append_jsonl(event_file, record)
        task_records.append(record)
        final_assistant_messages.append(str(final_assistant.get("content", "")))

        if probe_variant is not None and turn.index in checkpoints:
            checkpoint_index = checkpoints.index(turn.index) + 1
            instance = generate_probe_instance(
                probe_variant,
                task_instance_id,
                checkpoint_index,
            )
            probe_prompt = {
                "role": "user",
                "content": render_probe_prompt(instance),
            }
            history.append(probe_prompt)
            result = await transport.complete(
                model,
                history,
                purpose="active_probe",
                request_key=_request_key(
                    run_id, cell_id, "bfcl-probe", checkpoint_index
                ),
                input_token_estimate=conservative_input_token_bound(history),
                max_output_tokens=config.probe_max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=DEFAULT_REASONING_EFFORT[model],
                tools=(),
            )
            if result.tool_calls:
                raise ValueError("active BFCL probe unexpectedly returned tool calls")
            grade = grade_probe_response(instance, result.text)
            probe_assistant = {"role": "assistant", "content": result.text}
            history.append(probe_assistant)
            probe_record = {
                "event": "active_probe",
                "after_task_turn": turn.index,
                "checkpoint_index": checkpoint_index,
                "variant": probe_variant,
                "user_message": probe_prompt,
                "assistant_message": probe_assistant,
                "grade": {
                    "passed": grade.passed,
                    "value_correct": grade.value_correct,
                    "exact_format": grade.exact_format,
                    "error": grade.error,
                    "expected_sha256": sha256_json(instance.expected_answer),
                },
                "call": _call_record(result),
                # Canonical target-visible history through the carried probe
                # response.  This is the active pass-one evidence address.
                "source_prefix_sha256": sha256_json(history),
            }
            append_jsonl(event_file, probe_record)
            probe_records.append(probe_record)

    evaluation = bridge.evaluate_episode(episode_id, task.task_id)
    transcript_sha256 = sha256_json(history)
    all_events: list[Mapping[str, Any]] = [*task_records, *probe_records]
    evaluation_record = {
        "prediction": None,
        "evaluation_label_sha256": None,
        "success": evaluation.official_success,
        "official_success": evaluation.official_success,
        "official_score": str(evaluation.official_score),
        "official_result": evaluation.official_result,
    }
    complete_event = {
        "event": "complete",
        "task_turns": len(task_records),
        "transcript_sha256": transcript_sha256,
        "evaluation_sha256": sha256_json(evaluation_record),
        "prediction": None,
        "success": evaluation.official_success,
    }
    materialized = {
        "schema_version": 1,
        "bfcl_runner_version": BFCL_RUNNER_VERSION,
        "run_id": run_id,
        "cell_id": cell_id,
        "design_sha256": start["design_sha256"],
        "model": model,
        "domain": DOMAIN,
        "task_id": task.task_id,
        "condition": BFCL_CONDITION,
        "task_sha256": task.task_sha256,
        "arm": arm_name,
        "active_probe_variant": probe_variant,
        "checkpoint_turns": list(checkpoints),
        "messages": history,
        "task_assistant_messages": final_assistant_messages,
        "task_records": task_records,
        "probe_records": probe_records,
        "evaluation": evaluation_record,
        "transcript_sha256": transcript_sha256,
        "accounting": _rollup(all_events),
        "complete": True,
    }
    atomic_write_json(output_file, materialized)
    append_jsonl(event_file, complete_event)
    return materialized


def freeze_bfcl_public_task_manifest(
    path: str | Path,
    tasks: Sequence[BFCLTaskRecord],
    *,
    categories: Sequence[str] | None = None,
    task_ids: Sequence[str] = (),
) -> str:
    """Freeze a deterministic answer-blind selection using runner12's format."""

    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise ValueError("tasks must be a sequence of BFCLTaskRecord values")
    task_tuple = tuple(tasks)
    if not task_tuple or any(not isinstance(task, BFCLTaskRecord) for task in task_tuple):
        raise ValueError("tasks must contain BFCLTaskRecord values")
    selected_categories = (
        tuple(sorted({task.category for task in task_tuple}))
        if categories is None
        else tuple(categories)
    )
    if (
        not selected_categories
        or len(selected_categories) != len(set(selected_categories))
        or any(category not in V4_MULTI_TURN_CATEGORIES for category in selected_categories)
    ):
        raise ValueError("categories must be unique BFCL V4 multi-turn categories")
    if isinstance(task_ids, (str, bytes)) or not isinstance(task_ids, Sequence):
        raise ValueError("task_ids must be a sequence")
    selected_ids = tuple(task_ids)
    if any(not isinstance(item, str) or not item for item in selected_ids) or len(
        selected_ids
    ) != len(set(selected_ids)):
        raise ValueError("task_ids must be unique non-empty strings")
    selected = [
        task
        for task in task_tuple
        if task.category in selected_categories
        and (not selected_ids or task.task_id in selected_ids)
    ]
    if selected_ids and {task.task_id for task in selected} != set(selected_ids):
        raise ValueError("selected task_ids are not exactly available in the requested categories")
    if not selected_ids and {task.category for task in selected} != set(selected_categories):
        raise ValueError("BFCL task selection does not cover every requested category")
    if not selected:
        raise ValueError("BFCL task selection is empty")
    identities = [(task.category, task.task_id, task.task_sha256) for task in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("BFCL task selection contains duplicate identities")
    domain_tasks = [
        task.as_domain_task()
        for task in sorted(selected, key=lambda item: (item.category, item.task_id))
    ]
    return freeze_task_manifest(path, domain_tasks)


__all__ = [
    "BFCL_CONDITION",
    "BFCL_RUNNER_VERSION",
    "BFCLBridge",
    "BFCLRunnerConfig",
    "BFCLTransport",
    "freeze_bfcl_public_task_manifest",
    "generate_bfcl_passive_quiz",
    "run_bfcl_task",
    "validate_bfcl_task_turn_record",
]
