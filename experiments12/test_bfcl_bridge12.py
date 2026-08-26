"""Offline integration tests for the real pinned BFCL V4 bridge."""

from __future__ import annotations

import ast
from decimal import Decimal
import json
from pathlib import Path
import unittest

from experiments12.core.transport import ToolCall
from experiments12.domains.bfcl import (
    BFCLAdapter,
    BFCLBridgeError,
    StateCheckStatus,
    ToolExecutionStatus,
)


HERE = Path(__file__).resolve().parent
BFCL_ROOT = HERE / "external" / "gorilla-pinned"
BRIDGE_SCRIPT = HERE / "bfcl_bridge12.py"
PINNED_CHECKOUT_AVAILABLE = (
    BFCL_ROOT.is_dir()
    and (BFCL_ROOT / ".git").exists()
    and (
        BFCL_ROOT
        / "berkeley-function-call-leaderboard"
        / "bfcl_eval"
        / "data"
        / "BFCL_v4_multi_turn_base.json"
    ).is_file()
)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        call_id,
        name,
        json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested_key
            for item in value.values()
            for nested_key in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested_key for item in value for nested_key in _all_keys(item)}
    return set()


@unittest.skipUnless(PINNED_CHECKOUT_AVAILABLE, "pinned Gorilla checkout is unavailable")
class RealBFCLBridgeTests(unittest.TestCase):
    def adapter(self) -> BFCLAdapter:
        return BFCLAdapter(environment={"BFCL_ROOT": str(BFCL_ROOT)})

    def test_selected_v4_task_uses_exact_holdout_timing_and_native_schemas(self) -> None:
        with self.adapter().bridge_client(BRIDGE_SCRIPT) as bridge:
            tasks = bridge.load_tasks(
                categories=("multi_turn_miss_func",),
                task_ids=("multi_turn_miss_func_0",),
            )

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_id, "multi_turn_miss_func_0")
        self.assertEqual(len(task.turns), 5)
        names_by_turn = [{tool.name for tool in turn.tools} for turn in task.turns]
        self.assertNotIn("sort", names_by_turn[0])
        self.assertNotIn("sort", names_by_turn[2])
        self.assertIn("sort", names_by_turn[3])
        self.assertIn("sort", names_by_turn[4])
        self.assertEqual(
            task.turns[3].user_message,
            "I have updated some more functions you can choose from. What about now?",
        )
        self.assertTrue(
            all(tool.schema["type"] == "object" for turn in task.turns for tool in turn.tools)
        )
        self.assertTrue(
            all(tool.strict is False for turn in task.turns for tool in turn.tools)
        )
        public_keys = _all_keys(task.bridge_core())
        self.assertNotIn("initial_config", public_keys)
        self.assertNotIn("ground_truth", public_keys)
        self.assertNotIn("possible_answer", public_keys)

    def test_repeated_batches_empty_turn_close_and_redacted_state(self) -> None:
        with self.adapter().bridge_client(BRIDGE_SCRIPT) as bridge:
            task = bridge.load_tasks(
                categories=("multi_turn_base",),
                task_ids=("multi_turn_base_0",),
            )[0]
            bridge.begin_episode("batch-and-state", task.task_id)

            first = bridge.execute_tools(
                "batch-and-state",
                task.task_id,
                1,
                (_call("call-1", "cd", {"folder": "document"}),),
            )
            second = bridge.execute_tools(
                "batch-and-state",
                task.task_id,
                1,
                (
                    _call("call-2", "mkdir", {"dir_name": "temp"}),
                    _call(
                        "call-3",
                        "mv",
                        {"destination": "temp", "source": "final_report.pdf"},
                    ),
                ),
            )
            ended = bridge.execute_tools(
                "batch-and-state", task.task_id, 1, ()
            )
            state = bridge.materialize_public_state(
                "batch-and-state", task.task_id, 1
            )

            self.assertEqual(
                tuple(result.status for result in first.results + second.results),
                (
                    ToolExecutionStatus.SUCCEEDED,
                    ToolExecutionStatus.SUCCEEDED,
                    ToolExecutionStatus.SUCCEEDED,
                ),
            )
            self.assertEqual(ended.results, ())
            self.assertIs(ended.state_check, StateCheckStatus.NOT_RUN)
            self.assertIn("GorillaFileSystem", state.state["classes"])
            state_keys = {key.casefold() for key in _all_keys(state.state)}
            self.assertNotIn("password", state_keys)
            self.assertNotIn("initial_config", state_keys)
            self.assertNotIn("ground_truth", state_keys)
            serialized_state = json.dumps(state.state, sort_keys=True)
            self.assertNotIn("Kj8#mP9$vL2", serialized_state)

            with self.assertRaises(BFCLBridgeError) as premature:
                bridge.evaluate_episode("batch-and-state", task.task_id)
            self.assertEqual(premature.exception.code, "episode_not_complete")
            with self.assertRaises(BFCLBridgeError) as ended_turn:
                bridge.execute_tools(
                    "batch-and-state", task.task_id, 1, ()
                )
            self.assertEqual(ended_turn.exception.code, "turn_order")

    def test_invalid_and_execution_failure_statuses_are_objective(self) -> None:
        with self.adapter().bridge_client(BRIDGE_SCRIPT) as bridge:
            task = bridge.load_tasks(
                categories=("multi_turn_base",),
                task_ids=("multi_turn_base_1",),
            )[0]
            bridge.begin_episode("failure-status", task.task_id)
            execution = bridge.execute_tools(
                "failure-status",
                task.task_id,
                1,
                (
                    _call("unknown", "not_exposed", {}),
                    _call("missing", "cd", {}),
                    _call("failure", "pwd", {"unexpected": 1}),
                ),
            )

        self.assertEqual(
            tuple(result.status for result in execution.results),
            (
                ToolExecutionStatus.INVALID_CALL,
                ToolExecutionStatus.INVALID_CALL,
                ToolExecutionStatus.EXECUTION_FAILURE,
            ),
        )
        self.assertIs(execution.state_check, StateCheckStatus.NOT_RUN)
        self.assertTrue(execution.failure_indicators.invalid_call_observed)
        self.assertTrue(execution.failure_indicators.execution_failure_observed)
        self.assertFalse(execution.failure_indicators.state_check_available)

    def test_ground_truth_pilot_scores_only_at_final_episode_evaluation(self) -> None:
        batches = (
            (
                (_call("g-1", "cd", {"folder": "document"}),),
                (
                    _call("g-2", "mkdir", {"dir_name": "temp"}),
                    _call(
                        "g-3",
                        "mv",
                        {"destination": "temp", "source": "final_report.pdf"},
                    ),
                ),
            ),
            (
                (
                    _call("g-4", "cd", {"folder": "temp"}),
                    _call(
                        "g-5",
                        "grep",
                        {"file_name": "final_report.pdf", "pattern": "budget analysis"},
                    ),
                ),
            ),
            ((_call("g-6", "sort", {"file_name": "final_report.pdf"}),),),
            (
                (
                    _call("g-7", "cd", {"folder": ".."}),
                    _call(
                        "g-8",
                        "mv",
                        {"destination": "temp", "source": "previous_report.pdf"},
                    ),
                ),
                (
                    _call("g-9", "cd", {"folder": "temp"}),
                    _call(
                        "g-10",
                        "diff",
                        {
                            "file_name1": "final_report.pdf",
                            "file_name2": "previous_report.pdf",
                        },
                    ),
                ),
            ),
        )

        with self.adapter().bridge_client(BRIDGE_SCRIPT) as bridge:
            task = bridge.load_tasks(
                categories=("multi_turn_base",),
                task_ids=("multi_turn_base_0",),
            )[0]
            bridge.begin_episode("official-gold-pilot", task.task_id)
            for turn_index, turn_batches in enumerate(batches, 1):
                for batch in turn_batches:
                    execution = bridge.execute_tools(
                        "official-gold-pilot", task.task_id, turn_index, batch
                    )
                    self.assertTrue(
                        all(
                            result.status is ToolExecutionStatus.SUCCEEDED
                            for result in execution.results
                        )
                    )
                    self.assertIs(execution.state_check, StateCheckStatus.NOT_RUN)
                bridge.execute_tools(
                    "official-gold-pilot", task.task_id, turn_index, ()
                )
            evaluation = bridge.evaluate_episode(
                "official-gold-pilot", task.task_id
            )

        self.assertEqual(evaluation.official_score, Decimal("1"))
        self.assertTrue(evaluation.official_success)
        self.assertTrue(evaluation.official_result["valid"])
        self.assertEqual(
            evaluation.official_result["checker"], "BFCL_v4_multi_turn_checker"
        )
        result_keys = _all_keys(evaluation.official_result)
        self.assertNotIn("ground_truth", result_keys)
        self.assertNotIn("possible_answer", result_keys)
        self.assertNotIn("execution_result", result_keys)


class BFCLBridgeIsolationTests(unittest.TestCase):
    def test_bridge_has_no_model_or_network_imports(self) -> None:
        tree = ast.parse(BRIDGE_SCRIPT.read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module)
        forbidden_prefixes = (
            "openai",
            "anthropic",
            "requests",
            "urllib",
            "http.client",
            "socket",
            "bfcl_eval.model_handler",
        )
        self.assertFalse(
            sorted(
                name
                for name in imports
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in forbidden_prefixes
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
