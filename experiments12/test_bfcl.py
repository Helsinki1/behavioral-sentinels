"""No-network tests for the pinned BFCL subprocess contract."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from experiments12.core.transport import JsonSchemaTool, ToolCall
from experiments12.domains.base import (
    ArtifactIntegrityError,
    DomainUnavailableError,
    DomainValidationError,
)
from experiments12.domains.bfcl import (
    BFCLAdapter,
    BRIDGE_CAPABILITIES,
    BRIDGE_PROTOCOL,
    LICENSE_IDENTIFIER,
    PINNED_COMMIT,
    ROOT_ENVIRONMENT_VARIABLE,
    StateCheckStatus,
    ToolExecutionStatus,
    V4_FUNCTION_DOC_FILES,
    V4_MULTI_TURN_FILES,
    V4_OFFICIAL_SOURCE_FILES,
    V4_POSSIBLE_ANSWER_FILES,
    bridge_event_schemas,
)


APACHE_LICENSE_FIXTURE = """
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

9. Accepting Warranty or Additional Liability.

END OF TERMS AND CONDITIONS
""".strip()


FAKE_BRIDGE_SOURCE = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import sys

COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
CAPABILITIES = [
    "begin_episode",
    "evaluate_episode",
    "execute_tools",
    "load_tasks",
    "materialize_public_state",
]


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def respond(request, payload):
    event = {
        "schema_version": 1,
        "kind": "response",
        "operation": request["operation"],
        "request_id": request["request_id"],
        "ok": True,
        "payload": payload,
    }
    if os.environ.get("FAKE_BRIDGE_EXTRA") == "1" and request["operation"] == "hello":
        event["unexpected"] = "strict schemas must reject me"
    sys.stdout.write(canonical(event) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    operation = request["operation"]
    payload = request["payload"]
    if operation == "hello":
        respond(
            request,
            {
                "protocol": "bfcl-v4-jsonl",
                "bfcl_commit": COMMIT,
                "license": "Apache-2.0",
                "capabilities": CAPABILITIES,
            },
        )
    elif operation == "load_tasks":
        category = payload["categories"][0]
        task_id = payload["task_ids"][0] if payload["task_ids"] else category + "_0"
        tool = {
            "name": "lookup_contact",
            "description": "Look up one contact",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        task = {
            "task_id": task_id,
            "category": category,
            "turns": [
                {"index": 1, "user_message": "Find Ada", "tools": [tool]},
                {"index": 2, "user_message": "Now update the record", "tools": [tool]},
            ],
        }
        task["task_sha256"] = digest(task)
        if os.environ.get("FAKE_BAD_TASK_HASH") == "1":
            task["task_sha256"] = "0" * 64
        respond(request, {"tasks": [task]})
    elif operation == "begin_episode":
        respond(
            request,
            {
                "episode_id": payload["episode_id"],
                "task_id": payload["task_id"],
                "started": True,
            },
        )
    elif operation == "execute_tools":
        results = []
        for call in payload["tool_calls"]:
            if call["name"] == "missing_tool":
                status = "invalid_call"
                output = {"error": "unknown tool"}
            elif call["name"] == "explode":
                status = "execution_failure"
                output = {"error": "execution failed"}
            else:
                status = "succeeded"
                output = {"email": "ada@example.test"}
            results.append(
                {
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "status": status,
                    "output": output,
                }
            )
        respond(
            request,
            {
                "episode_id": payload["episode_id"],
                "task_id": payload["task_id"],
                "turn_index": payload["turn_index"],
                "results": results,
                "state_check": "failed",
            },
        )
    elif operation == "materialize_public_state":
        state = {"contacts": {"Ada": {"email": "ada@example.test"}}, "revision": 2}
        respond(
            request,
            {
                "episode_id": payload["episode_id"],
                "task_id": payload["task_id"],
                "after_turn": payload["after_turn"],
                "state": state,
                "state_sha256": digest(state),
            },
        )
    elif operation == "evaluate_episode":
        respond(
            request,
            {
                "episode_id": payload["episode_id"],
                "task_id": payload["task_id"],
                "official_score": 0.25,
                "official_success": False,
                "official_result": {"checker": "official", "passed": False},
            },
        )
'''


class BFCLTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_bfcl_")
        self.root = Path(self.temp.name) / "gorilla"
        self.bridge_script = Path(self.temp.name) / "fake_bfcl_bridge.py"
        self.bridge_script.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_checkout(
        self,
        *,
        commit: str = PINNED_COMMIT,
        license_text: str = APACHE_LICENSE_FIXTURE,
        omit_category: str | None = None,
    ) -> Path:
        git = self.root / ".git"
        git.mkdir(parents=True)
        (git / "HEAD").write_text(commit + "\n", encoding="utf-8")
        (self.root / "LICENSE").write_text(license_text, encoding="utf-8")
        data = self.root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
        data.mkdir(parents=True)
        for category, filename in V4_MULTI_TURN_FILES:
            if category == omit_category:
                continue
            (data / filename).write_text(
                json.dumps({"category": category, "synthetic": True}) + "\n",
                encoding="utf-8",
            )
        possible_answers = data / "possible_answer"
        possible_answers.mkdir()
        for category, filename in V4_POSSIBLE_ANSWER_FILES:
            (possible_answers / filename).write_text(
                json.dumps(
                    {"id": f"{category}_0", "ground_truth": []},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        function_docs = data / "multi_turn_func_doc"
        function_docs.mkdir()
        for class_name, filename in V4_FUNCTION_DOC_FILES:
            (function_docs / filename).write_text(
                json.dumps(
                    {
                        "name": f"fixture_{class_name}",
                        "description": "fixture",
                        "parameters": {
                            "type": "dict",
                            "properties": {},
                            "required": [],
                        },
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        package = self.root / "berkeley-function-call-leaderboard"
        for _label, relative in V4_OFFICIAL_SOURCE_FILES:
            source = package / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("# pinned source fixture\n", encoding="utf-8")
        return self.root

    def adapter(self) -> BFCLAdapter:
        return BFCLAdapter(environment={ROOT_ENVIRONMENT_VARIABLE: str(self.root)})

    def test_requires_explicit_root_exact_commit_license_and_complete_v4_set(self) -> None:
        with self.assertRaisesRegex(DomainUnavailableError, "BFCL_ROOT"):
            BFCLAdapter(environment={})

        self.make_checkout(commit="1" * 40)
        with self.assertRaisesRegex(ArtifactIntegrityError, PINNED_COMMIT):
            self.adapter()

        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_bfcl_")
        self.root = Path(self.temp.name) / "gorilla"
        self.bridge_script = Path(self.temp.name) / "fake_bfcl_bridge.py"
        self.bridge_script.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        self.make_checkout(license_text="not a license")
        with self.assertRaisesRegex(ArtifactIntegrityError, "Apache-2.0"):
            self.adapter()

        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_bfcl_")
        self.root = Path(self.temp.name) / "gorilla"
        self.bridge_script = Path(self.temp.name) / "fake_bfcl_bridge.py"
        self.bridge_script.write_text(FAKE_BRIDGE_SOURCE, encoding="utf-8")
        self.make_checkout(omit_category="multi_turn_miss_param")
        with self.assertRaisesRegex(DomainUnavailableError, "complete V4"):
            self.adapter()

    def test_readiness_artifacts_boundary_and_no_static_semantics(self) -> None:
        self.make_checkout()
        adapter = self.adapter()
        self.assertTrue(adapter.readiness.ready_for_external_bridge)
        self.assertEqual(adapter.readiness.checkout_commit, PINNED_COMMIT)
        self.assertEqual(adapter.readiness.license_identifier, LICENSE_IDENTIFIER)
        self.assertEqual(len(adapter.readiness.v4_data_artifacts), 4)
        self.assertEqual(len(adapter.readiness.v4_possible_answer_artifacts), 4)
        self.assertEqual(len(adapter.readiness.v4_function_doc_artifacts), 8)
        self.assertEqual(
            len(adapter.readiness.v4_official_source_artifacts),
            len(V4_OFFICIAL_SOURCE_FILES),
        )
        self.assertEqual(
            len(adapter.input_artifacts),
            1
            + len(V4_MULTI_TURN_FILES)
            + len(V4_POSSIBLE_ANSWER_FILES)
            + len(V4_FUNCTION_DOC_FILES)
            + len(V4_OFFICIAL_SOURCE_FILES),
        )
        self.assertEqual(adapter.loader_boundary().external_root, str(self.root.resolve()))
        with self.assertRaisesRegex(DomainUnavailableError, "interactive"):
            adapter.load_tasks()

        first_data = Path(adapter.readiness.v4_data_artifacts[0].path)
        first_data.write_text("changed after validation\n", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactIntegrityError, "SHA256 mismatch"):
            adapter.loader_boundary()

    def test_answer_docs_and_official_sources_are_rechecked_fail_closed(self) -> None:
        self.make_checkout()
        adapter = self.adapter()
        representatives = (
            adapter.readiness.v4_possible_answer_artifacts[0],
            adapter.readiness.v4_function_doc_artifacts[0],
            adapter.readiness.v4_official_source_artifacts[-1],
        )
        for artifact in representatives:
            with self.subTest(role=artifact.role):
                path = Path(artifact.path)
                original = path.read_bytes()
                path.write_bytes(original + b"dirty mutation\n")
                with self.assertRaisesRegex(ArtifactIntegrityError, "SHA256 mismatch"):
                    adapter.loader_boundary()
                path.write_bytes(original)
                self.assertEqual(
                    adapter.loader_boundary().pinned_commit,
                    PINNED_COMMIT,
                )

    def test_fake_bridge_full_episode_contract_and_objective_turn_indicators(self) -> None:
        self.make_checkout()
        adapter = self.adapter()
        with adapter.bridge_client(self.bridge_script) as bridge:
            tasks = bridge.load_tasks(
                categories=("multi_turn_base",),
                task_ids=("multi_turn_base_7",),
            )
            self.assertEqual(len(tasks), 1)
            task = tasks[0]
            self.assertEqual(task.task_id, "multi_turn_base_7")
            self.assertEqual(len(task.turns), 2)
            self.assertIsInstance(task.turns[0].tools[0], JsonSchemaTool)
            common = task.as_domain_task()
            self.assertIsNone(common.evaluation_label)
            self.assertEqual(common.turns[0].user_message, "Find Ada")

            started = bridge.begin_episode("episode-7", task.task_id)
            self.assertEqual(started.episode_id, "episode-7")
            execution = bridge.execute_tools(
                "episode-7",
                task.task_id,
                1,
                (
                    ToolCall("call-1", "missing_tool", '{"name":"Ada"}'),
                    ToolCall("call-2", "explode", '{"name":"Ada"}'),
                ),
            )
            self.assertEqual(
                tuple(result.status for result in execution.results),
                (ToolExecutionStatus.INVALID_CALL, ToolExecutionStatus.EXECUTION_FAILURE),
            )
            self.assertIs(execution.state_check, StateCheckStatus.FAILED)
            indicators = execution.failure_indicators
            self.assertTrue(indicators.invalid_call_observed)
            self.assertTrue(indicators.execution_failure_observed)
            self.assertTrue(indicators.state_check_failure_observed)
            self.assertTrue(indicators.state_check_available)
            self.assertTrue(indicators.any_observed_failure)
            self.assertEqual(execution.tool_messages()[0]["role"], "tool")

            state = bridge.materialize_public_state("episode-7", task.task_id, 1)
            self.assertEqual(state.state["revision"], 2)
            evaluation = bridge.evaluate_episode("episode-7", task.task_id)
            self.assertEqual(evaluation.official_score, Decimal("0.25"))
            self.assertFalse(evaluation.official_success)
            self.assertEqual(evaluation.official_result["checker"], "official")
        self.assertFalse(bridge.running)

    def test_bridge_rejects_unknown_fields_and_bad_task_digest(self) -> None:
        self.make_checkout()
        adapter = self.adapter()
        extra = adapter.bridge_client(
            self.bridge_script,
            base_environment={"FAKE_BRIDGE_EXTRA": "1"},
        )
        with self.assertRaisesRegex(DomainValidationError, "unknown"):
            extra.start()
        self.assertFalse(extra.running)

        with adapter.bridge_client(
            self.bridge_script,
            base_environment={"FAKE_BAD_TASK_HASH": "1"},
        ) as bridge:
            with self.assertRaisesRegex(ArtifactIntegrityError, "task_sha256"):
                bridge.load_tasks(categories=("multi_turn_base",))

    def test_contract_has_strict_envelopes_and_no_per_turn_correctness_field(self) -> None:
        schema = bridge_event_schemas()
        self.assertEqual(schema["protocol"], BRIDGE_PROTOCOL)
        self.assertEqual(set(schema["operations"]), BRIDGE_CAPABILITIES | {"hello"})
        self.assertFalse(schema["request_envelope"]["additional_properties"])
        self.assertFalse(schema["success_envelope"]["additional_properties"])
        serialized = json.dumps(schema).lower()
        self.assertNotIn("per_turn_correct", serialized)
        self.assertNotIn("ground_truth", serialized)


if __name__ == "__main__":
    unittest.main()
