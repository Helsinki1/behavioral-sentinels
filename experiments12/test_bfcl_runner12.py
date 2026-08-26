from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile
import unittest

from experiments12.bfcl_runner12 import (
    BFCL_CONDITION,
    BFCLRunnerConfig,
    freeze_bfcl_public_task_manifest,
    generate_bfcl_passive_quiz,
    run_bfcl_task,
)
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_json,
)
from experiments12.core.schemas import CallAttemptRecord, CallStatus, TokenUsage
from experiments12.core.transport import CompletionResult, JsonSchemaTool, ToolCall
from experiments12.domains.base import canonical_json_sha256
from experiments12.domains.bfcl import (
    BFCLOfficialEpisodeEvaluation,
    BFCLPublicState,
    BFCLStartedEpisode,
    BFCLTaskRecord,
    BFCLTaskTurn,
    BFCLToolExecutionResult,
    BFCLTurnExecution,
    StateCheckStatus,
    ToolExecutionStatus,
)
from experiments12.deployment12 import DeploymentArtifactError
from experiments12.prepare_deployment12 import _active_records
from experiments12.runner12 import load_task_manifest
from experiments12.shadow12 import score_clean_trajectory


def make_task(
    task_id: str = "bfcl-task-1",
    *,
    category: str = "multi_turn_base",
    turns: int = 2,
) -> BFCLTaskRecord:
    tool = JsonSchemaTool.from_schema(
        "lookup_contact",
        "Look up one public contact",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )
    task_turns = tuple(
        BFCLTaskTurn(index, f"Official user turn {index}", (tool,))
        for index in range(1, turns + 1)
    )
    core = {
        "task_id": task_id,
        "category": category,
        "turns": [turn.bridge_payload() for turn in task_turns],
    }
    return BFCLTaskRecord(
        task_id=task_id,
        category=category,
        turns=task_turns,
        source_sha256="a" * 64,
        task_sha256=canonical_json_sha256(core),
    )


class FakeTransport:
    def __init__(self, *, always_tools: bool = False) -> None:
        self.always_tools = always_tools
        self.calls: list[dict[str, object]] = []
        self.counter = 0

    async def complete(self, model, messages, **kwargs):
        self.counter += 1
        copied_messages = json.loads(json.dumps(messages))
        self.calls.append({"model": model, "messages": copied_messages, **kwargs})
        purpose = kwargs["purpose"]
        tool_calls: tuple[ToolCall, ...] = ()
        if purpose in {"active_probe", "frozen_probe"}:
            text = "PROBE: 00000000"
        elif purpose == "frozen_quiz":
            text = "A1: 1\nA2: 2\nA3: lookup_contact\nA4: execution_failure"
        elif purpose == "trace_judge":
            text = '{"risk":0.25,"concerns":[],"evidence":[]}'
        else:
            match = re.search(r"/bfcl-task-(\d+)-(\d+)$", kwargs["request_key"])
            if match is None:
                raise AssertionError("unexpected synthetic request key")
            turn, batch = (int(value) for value in match.groups())
            should_call = self.always_tools or (turn == 1 and batch in {1, 2})
            if should_call:
                tool_calls = (
                    ToolCall(
                        f"tool-{turn}-{batch}",
                        "lookup_contact",
                        json.dumps(
                            {"name": f"Person {batch}"},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
                text = ""
            else:
                text = f"Completed public turn {turn}."
        attempt = CallAttemptRecord(
            event_id=f"event-{self.counter}",
            reservation_id=f"reservation-{self.counter}",
            provider="openai",
            model=model,
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            usage=TokenUsage(input_tokens=100 + self.counter, output_tokens=5),
            estimated_cost_usd=Decimal("0.001"),
            elapsed_ms=10,
        )
        return CompletionResult(
            text=text,
            tool_calls=tool_calls,
            usage=attempt.usage,
            response_id=f"response-{self.counter}",
            request_id=f"request-{self.counter}",
            model_id=model,
            finish_reason="tool_calls" if tool_calls else "stop",
            cost_usd=Decimal("0.001"),
            attempts=(attempt,),
        )


class FakeBridge:
    def __init__(self) -> None:
        self.begin_calls: list[tuple[str, str]] = []
        self.execute_calls: list[tuple[int, tuple[ToolCall, ...]]] = []
        self.state_calls: list[int] = []
        self.evaluate_calls = 0

    def begin_episode(self, episode_id, task_id):
        self.begin_calls.append((episode_id, task_id))
        return BFCLStartedEpisode(episode_id, task_id)

    def execute_tools(self, episode_id, task_id, turn_index, tool_calls):
        calls = tuple(tool_calls)
        self.execute_calls.append((turn_index, calls))
        results = tuple(
            BFCLToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                status=(
                    ToolExecutionStatus.EXECUTION_FAILURE
                    if call.call_id.endswith("-2")
                    else ToolExecutionStatus.SUCCEEDED
                ),
                output_json=(
                    '{"error":"synthetic failure"}'
                    if call.call_id.endswith("-2")
                    else '{"email":"public@example.test"}'
                ),
            )
            for call in calls
        )
        return BFCLTurnExecution(
            episode_id=episode_id,
            task_id=task_id,
            turn_index=turn_index,
            results=results,
            state_check=(StateCheckStatus.NOT_RUN if calls else StateCheckStatus.PASSED),
        )

    def materialize_public_state(self, episode_id, task_id, after_turn):
        self.state_calls.append(after_turn)
        state = {"completed_turns": after_turn, "revision": after_turn}
        return BFCLPublicState(
            episode_id=episode_id,
            task_id=task_id,
            after_turn=after_turn,
            state_json=json.dumps(state, sort_keys=True, separators=(",", ":")),
            state_sha256=canonical_json_sha256(state),
        )

    def evaluate_episode(self, episode_id, task_id):
        self.evaluate_calls += 1
        result = {"checker": "official", "public_summary": "synthetic"}
        return BFCLOfficialEpisodeEvaluation(
            episode_id=episode_id,
            task_id=task_id,
            official_score=Decimal("0.75"),
            official_success=True,
            official_result_json=json.dumps(
                result, sort_keys=True, separators=(",", ":")
            ),
        )


class NoCallTransport:
    async def complete(self, *_args, **_kwargs):
        raise AssertionError("this passive configuration must make no model calls")


class BFCLRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_bfcl_runner_")
        self.root = Path(self.temp.name)
        self.event_path = self.root / "cell.events.jsonl"
        self.output_path = self.root / "cell.json"
        self.task = make_task()

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def run_cell(self, arm: str, *, bridge=None, transport=None, config=None):
        bridge = bridge or FakeBridge()
        transport = transport or FakeTransport()
        output = await run_bfcl_task(
            run_id="bfcl-run",
            cell_id="cell-1",
            model="gpt-5.6-luna",
            task=self.task,
            arm_name=arm,
            bridge=bridge,
            transport=transport,
            event_path=self.event_path,
            output_path=self.output_path,
            config=config or BFCLRunnerConfig(task_max_output_tokens=80),
        )
        return output, bridge, transport

    async def test_clean_repeated_native_tool_batches_finalize_and_shadow(self) -> None:
        output, bridge, transport = await self.run_cell("clean")
        self.assertTrue(output["complete"])
        self.assertEqual(
            [(turn, len(calls)) for turn, calls in bridge.execute_calls],
            [(1, 1), (1, 1), (1, 0), (2, 0)],
        )
        self.assertEqual(bridge.state_calls, [1, 2])
        self.assertEqual(bridge.evaluate_calls, 1)
        self.assertEqual(len(output["task_records"][0]["calls"]), 3)
        self.assertFalse(output["task_records"][0]["capped"])
        self.assertEqual(
            output["task_records"][0]["termination_reason"],
            "model_final_response",
        )
        self.assertEqual(output["task_records"][0]["tool_batch_limit"], 12)
        aggregate_ids = output["task_records"][0]["call"]["call_event_ids"]
        self.assertEqual(aggregate_ids, ["event-1", "event-2", "event-3"])
        self.assertTrue(
            output["task_records"][0]["failure_indicators"][
                "execution_failure_observed"
            ]
        )
        self.assertEqual(output["task_records"][0]["public_state_sha256"], canonical_json_sha256({"completed_turns": 1, "revision": 1}))
        self.assertIn("messages", output["task_records"][0])
        self.assertIn("tool", [message["role"] for message in output["messages"]])
        self.assertNotIn("ACTIVE CARRIED PROBE", str(output["messages"]))
        self.assertEqual(output["probe_records"], [])
        self.assertEqual(output["transcript_sha256"], sha256_json(output["messages"]))
        self.assertTrue(
            all(
                str(call["request_key"]).startswith("bfcl-run/cell-1/")
                for call in transport.calls
            )
        )
        events = read_jsonl(self.event_path)
        self.assertEqual(
            [event["event"] for event in events],
            ["start", "task_turn", "task_turn", "complete"],
        )

        # Existing shadow12 can consume the common task-record fields; the
        # complete native trace remains available in each record's `messages`.
        passive_transport = FakeTransport()
        shadow = await score_clean_trajectory(
            run_id="bfcl-run",
            trajectory=output,
            transport=passive_transport,
            event_path=self.root / "shadow.events.jsonl",
            output_path=self.root / "shadow.json",
            quiz_by_checkpoint={
                turn: generate_bfcl_passive_quiz(output["task_records"], turn)
                for turn in output["checkpoint_turns"]
            },
        )
        self.assertTrue(shadow["complete"])
        self.assertEqual(shadow["domain"], "bfcl_multi_turn")
        context = next(
            record
            for record in shadow["records"]
            if record["method"] == "context_use" and record["checkpoint_turn"] == 1
        )
        self.assertEqual(context["raw_input_tokens"], 103)
        rules = next(
            record
            for record in shadow["records"]
            if record["method"] == "trace_rules" and record["checkpoint_turn"] == 1
        )
        self.assertIn("execution_error", rules["reasons"])
        self.assertTrue(rules["observed_event_flags"]["tool_result_error"])

        changed_quiz = list(generate_bfcl_passive_quiz(output["task_records"], 1))
        changed_quiz[0] = replace(changed_quiz[0], expected=99)
        with self.assertRaisesRegex(ValueError, "canonical generator"):
            await score_clean_trajectory(
                run_id="bfcl-run",
                trajectory=output,
                transport=NoCallTransport(),
                event_path=self.root / "changed-shadow.events.jsonl",
                output_path=self.root / "changed-shadow.json",
                quiz_by_checkpoint={1: tuple(changed_quiz)},
            )

    async def test_active_setup_and_probe_are_carried_into_later_tool_turn(self) -> None:
        output, bridge, transport = await self.run_cell("active_counter")
        self.assertEqual(len(output["probe_records"]), 1)
        probe = output["probe_records"][0]
        probe_index = output["messages"].index(probe["assistant_message"])
        self.assertEqual(
            probe["source_prefix_sha256"],
            sha256_json(output["messages"][: probe_index + 1]),
        )
        self.assertEqual(len(_active_records(output, "active_counter")), 1)
        forged = json.loads(json.dumps(output))
        forged["probe_records"][0]["source_prefix_sha256"] = "f" * 64
        with self.assertRaisesRegex(DeploymentArtifactError, "carried-prefix hash changed"):
            _active_records(forged, "active_counter")
        self.assertIn("increment this number", output["messages"][0]["content"])
        probe_calls = [call for call in transport.calls if call["purpose"] == "active_probe"]
        self.assertEqual(len(probe_calls), 1)
        self.assertEqual(probe_calls[0]["tools"], ())
        second_turn_first_call = next(
            call
            for call in transport.calls
            if str(call["request_key"]).endswith("/bfcl-task-2-1")
        )
        second_history = second_turn_first_call["messages"]
        self.assertTrue(any(message["role"] == "tool" for message in second_history))
        self.assertTrue(
            any("ACTIVE CARRIED PROBE" in message.get("content", "") for message in second_history)
        )
        self.assertEqual(bridge.state_calls, [1, 2])
        events = read_jsonl(self.event_path)
        self.assertEqual(
            [event["event"] for event in events],
            ["start", "task_turn", "active_probe", "task_turn", "complete"],
        )

    async def test_completed_cell_is_idempotent_but_partial_cell_never_resumes(self) -> None:
        output, bridge, transport = await self.run_cell("clean")
        call_count = len(transport.calls)
        execute_count = len(bridge.execute_calls)
        second, _bridge, _transport = await self.run_cell(
            "clean", bridge=bridge, transport=transport
        )
        self.assertEqual(second, output)
        self.assertEqual(len(transport.calls), call_count)
        self.assertEqual(len(bridge.execute_calls), execute_count)

        partial_event = self.root / "partial.events.jsonl"
        partial_output = self.root / "partial.json"
        atomic_write_jsonl(partial_event, [{"event": "start", "partial": True}])
        untouched_bridge, untouched_transport = FakeBridge(), FakeTransport()
        with self.assertRaisesRegex(FileExistsError, "partial"):
            await run_bfcl_task(
                run_id="bfcl-run",
                cell_id="cell-2",
                model="gpt-5.6-luna",
                task=self.task,
                arm_name="clean",
                bridge=untouched_bridge,
                transport=untouched_transport,
                event_path=partial_event,
                output_path=partial_output,
            )
        self.assertEqual(untouched_bridge.begin_calls, [])
        self.assertEqual(untouched_transport.calls, [])

    async def test_batch_limit_is_auditable_completed_agent_outcome(self) -> None:
        bridge = FakeBridge()
        transport = FakeTransport(always_tools=True)
        config = BFCLRunnerConfig(max_tool_batches_per_turn=2, task_max_output_tokens=80)
        output, _bridge, _transport = await self.run_cell(
            "clean", bridge=bridge, transport=transport, config=config
        )
        self.assertTrue(self.event_path.exists())
        self.assertTrue(self.output_path.exists())
        self.assertTrue(output["complete"])
        self.assertTrue(output["evaluation"]["official_success"])
        self.assertEqual(len(output["task_records"]), 2)
        self.assertEqual(output["task_assistant_messages"], ["", ""])
        self.assertEqual(
            [(turn, len(calls)) for turn, calls in bridge.execute_calls],
            [(1, 1), (1, 1), (1, 0), (2, 1), (2, 1), (2, 0)],
        )
        self.assertEqual(bridge.state_calls, [1, 2])
        self.assertEqual(bridge.evaluate_calls, 1)
        for turn, record in enumerate(output["task_records"], 1):
            self.assertTrue(record["capped"])
            self.assertEqual(record["termination_reason"], "tool_batch_limit_exhausted")
            self.assertEqual(record["tool_batch_limit"], 2)
            self.assertEqual(len(record["calls"]), 2)
            self.assertTrue(record["assistant_message"]["tool_calls"])
            self.assertEqual(record["assistant_message"], record["messages"][-2])
            self.assertEqual(record["messages"][-1]["role"], "tool")
            self.assertEqual(record["tool_executions"][-1]["batch_index"], None)
            self.assertTrue(record["tool_executions"][-1]["finalization"])
            self.assertEqual(record["tool_executions"][-1]["tool_calls"], [])
            self.assertEqual(record["tool_executions"][-1]["results"], [])
            if turn == 1:
                next_turn = next(
                    call
                    for call in transport.calls
                    if str(call["request_key"]).endswith("/bfcl-task-2-1")
                )
                self.assertEqual(next_turn["messages"][-1]["role"], "user")
                self.assertTrue(
                    any(
                        message.get("tool_call_id") == "tool-1-2"
                        for message in next_turn["messages"]
                        if isinstance(message, dict)
                    )
                )
        self.assertEqual(
            [event["event"] for event in read_jsonl(self.event_path)],
            ["start", "task_turn", "task_turn", "complete"],
        )

        # A coherent-looking completed artifact is not reusable if it omits
        # the explicit bounded-agent termination evidence.
        spent_calls = len(transport.calls)
        forged_output = read_json(self.output_path)
        forged_events = read_jsonl(self.event_path)
        del forged_output["task_records"][0]["termination_reason"]
        del forged_events[1]["termination_reason"]
        atomic_write_json(self.output_path, forged_output)
        atomic_write_jsonl(self.event_path, forged_events)
        with self.assertRaisesRegex(FileExistsError, "bounded-agent turn record"):
            await self.run_cell("clean", bridge=bridge, transport=transport, config=config)
        self.assertEqual(len(transport.calls), spent_calls)


class BFCLHelpersTests(unittest.TestCase):
    def test_freeze_selected_public_manifest_uses_runner_format(self) -> None:
        tasks = (
            make_task("z-task", category="multi_turn_miss_param"),
            make_task("a-task", category="multi_turn_base"),
        )
        with tempfile.TemporaryDirectory(prefix="experiment12_bfcl_freeze_") as tmp:
            path = Path(tmp) / "tasks.jsonl"
            digest = freeze_bfcl_public_task_manifest(
                path,
                tasks,
                categories=("multi_turn_base",),
                task_ids=("a-task",),
            )
            self.assertEqual(len(digest), 64)
            rows = load_task_manifest(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task_id"], f"a-task::{BFCL_CONDITION}")
            serialized = json.dumps(rows)
            self.assertNotIn("tools", rows[0])
            self.assertNotIn("gold", serialized.lower())
            with self.assertRaises(FileExistsError):
                freeze_bfcl_public_task_manifest(path, tasks)

    def test_passive_quiz_uses_only_observed_tool_names_statuses_and_counts(self) -> None:
        records = [
            {
                "event": "task_turn",
                "task_turn": 1,
                "tool_results": [
                    {
                        "name": "lookup_contact",
                        "status": "succeeded",
                        "output_json": '{"private_like_value":"never quiz this"}',
                    }
                ],
                "private_state": {"gold": "DO_NOT_USE"},
            },
            {
                "event": "task_turn",
                "task_turn": 2,
                "tool_results": [],
            },
        ]
        first = generate_bfcl_passive_quiz(records, 1)
        changed_private = json.loads(json.dumps(records))
        changed_private[0]["private_state"] = {"gold": "CHANGED_SECRET"}
        changed_private[0]["tool_results"][0]["output_json"] = '{"another":"secret"}'
        second = generate_bfcl_passive_quiz(changed_private, 1)
        self.assertEqual(first, second)
        expected = {question.question_id: question.expected for question in first}
        self.assertEqual(expected["bfcl_completed_turns_1"], 1)
        self.assertEqual(expected["bfcl_tool_results_1"], 1)
        self.assertEqual(expected["bfcl_latest_status_1"], "succeeded")
        serialized = json.dumps([question.text for question in first])
        self.assertNotIn("DO_NOT_USE", serialized)
        self.assertNotIn("never quiz this", serialized)
        with self.assertRaises(ValueError):
            generate_bfcl_passive_quiz(records, 3)


if __name__ == "__main__":
    unittest.main()
