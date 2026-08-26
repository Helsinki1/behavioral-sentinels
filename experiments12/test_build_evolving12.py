"""Offline tests for the resumable Evolving Intent reproduction builder."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import venv

from experiments12.build_evolving12 import (
    BRIDGE_CONTRACT_SHA256,
    BRIDGE_COMPATIBILITY_PATCHES,
    BRIDGE_PROTOCOL,
    BRIDGE_RUNTIME_DEPENDENCIES,
    BridgeRuntimeAttestation,
    BuildSettings,
    BridgeProtocolError,
    EvolvingReproductionBuilder,
    PINNED_COMMIT,
    ReadinessBlocked,
    ROOT_ENVIRONMENT_VARIABLE,
    STAGES,
    ResumeBlocked,
    SubprocessUpstreamBridge,
    _validate_capabilities,
    audit_reproduction,
    parse_ids,
    select_source_tasks,
)
from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
)
from experiments12.core.schemas import TokenUsage
from experiments12.domains.base import InputArtifact
from experiments12.domains.evolving_intent import EvolvingIntentAdapter


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkout(root: Path, *, with_bridge: bool = True) -> Path | None:
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(PINNED_COMMIT + "\n", encoding="utf-8")
    (root / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted, free of charge, to any person...\n",
        encoding="utf-8",
    )
    _write_json(
        root / "intent_construction/eval_indices/gsm8k_eval_ids.json",
        {"task-12": 12, "task-14": 14, "task-16": 16},
    )
    for stage_dir in ("intent_extraction", "counterfactual", "predecessor"):
        stage_root = (
            root / "intent_construction" / stage_dir
            if stage_dir == "intent_extraction"
            else root / "intent_construction" / "retrospective_expansion" / stage_dir
        )
        stage_root.mkdir(parents=True)
        (stage_root / "prompts.py").write_text(
            f"PROMPT = {stage_dir!r}\n", encoding="utf-8"
        )
        (stage_root / "validation.py").write_text("# validator\n", encoding="utf-8")
    simulator = root / "situated_simulation/user_simulation.py"
    simulator.parent.mkdir(parents=True)
    simulator.write_text("# rule simulator fixture\n", encoding="utf-8")
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root / ".venv")
    if not with_bridge:
        return None
    bridge = root / "experiment12_bridge.py"
    bridge.write_text("# checkout-local bridge fixture\n", encoding="utf-8")
    return bridge


def _gsm8k(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(17):
            handle.write(
                json.dumps({"question": f"Question {index}?", "answer": str(index * 2)})
                + "\n"
            )


class FakeBridge:
    def __init__(
        self,
        root: Path,
        bridge_path: Path,
        runtime: BridgeRuntimeAttestation,
    ) -> None:
        self.root = root
        self._runtime = runtime
        self._artifact = InputArtifact(
            "upstream_transport_bridge", str(bridge_path), sha256_file(bridge_path)
        )
        self.exchanges: list[dict[str, object]] = []

    @property
    def artifact(self) -> InputArtifact:
        return self._artifact

    @property
    def runtime(self) -> BridgeRuntimeAttestation:
        return self._runtime

    def exchange(self, request):
        self.exchanges.append(dict(request))
        operation = request["operation"]
        if operation == "capabilities":
            return {
                "protocol": BRIDGE_PROTOCOL,
                "contract_sha256": BRIDGE_CONTRACT_SHA256,
                "upstream_commit": PINNED_COMMIT,
                "transport_mode": "emit_requests_only",
                "stages": list(STAGES),
                "renderer": {"kind": "rule_based"},
                "runtime": {
                    "python": self.runtime.python_version,
                    "math_verifier_mode": "math_verify",
                    "dependencies": {
                        name: {"available": True, "version": version}
                        for name, version in BRIDGE_RUNTIME_DEPENDENCIES.items()
                    },
                },
                "compatibility_patches": list(BRIDGE_COMPATIBILITY_PATCHES),
            }
        if operation == "advance_stage":
            if request["model_result"] is None:
                stage = request["stage"]
                role = "judge" if stage == "predecessor_generation" else "generator"
                prompt_dir = {
                    "intent_extraction": "intent_extraction",
                    "counterfactual_generation": "retrospective_expansion/counterfactual",
                    "predecessor_generation": "retrospective_expansion/predecessor",
                }[stage]
                return {
                    "status": "needs_model_call",
                    "state": {"asked": True},
                    "call": {
                        "call_key": f"{stage}-call",
                        "role": role,
                        "messages": [
                            {"role": "system", "content": f"Run {stage}."},
                            {"role": "user", "content": request["source_task"]["question"]},
                        ],
                        "temperature": 0.7,
                        "max_output_tokens": 128,
                        "output_schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                        "prompt_files": [
                            f"intent_construction/{prompt_dir}/prompts.py"
                        ],
                    },
                }
            return {
                "status": "complete",
                "artifact": {
                    "stage": request["stage"],
                    "validated": True,
                    "model_output_sha256": request["model_result"]["output_sha256"],
                },
            }
        if operation == "render_pair":
            source = request["source_task"]
            return {
                "status": "rendered",
                "simulator": {"kind": "rule_based"},
                "records": [
                    {
                        "task_id": source["task_id"],
                        "condition": "t1",
                        "turns": [source["question"]],
                        "label": source["answer"],
                        "gold": "must be dropped",
                    },
                    {
                        "task_id": source["task_id"],
                        "condition": "t7",
                        "turns": [f"Public revision {index}" for index in range(1, 8)],
                        "label": str(int(source["answer"]) + 1),
                        "change_plan": ["must be dropped"],
                    },
                ],
            }
        raise AssertionError(operation)


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete(self, model_name, messages, **kwargs):
        self.calls.append({"model": model_name, "messages": messages, **kwargs})
        index = len(self.calls)
        return SimpleNamespace(
            text='{"ok":true}',
            tool_calls=(),
            usage=TokenUsage(input_tokens=10, output_tokens=3),
            response_id=f"response-{index}",
            request_id=f"request-{index}",
            model_id=f"resolved-{model_name}",
            finish_reason="stop",
            cost_usd=0,
            attempts=(SimpleNamespace(event_id=f"event-{index}"),),
        )


class UtilityTests(unittest.TestCase):
    def test_parse_tiny_ids(self) -> None:
        self.assertEqual(parse_ids("12,14,16"), (12, 14, 16))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_ids("12,12")

    def test_bridge_preflight_rejects_math_verifier_fallback(self) -> None:
        dependencies = {
            name: {"available": True, "version": version}
            for name, version in BRIDGE_RUNTIME_DEPENDENCIES.items()
        }
        dependencies["math-verify"] = {"available": False, "version": None}
        with self.assertRaisesRegex(BridgeProtocolError, "runtime does not match"):
            _validate_capabilities(
                {
                    "protocol": BRIDGE_PROTOCOL,
                    "contract_sha256": BRIDGE_CONTRACT_SHA256,
                    "upstream_commit": PINNED_COMMIT,
                    "transport_mode": "emit_requests_only",
                    "stages": list(STAGES),
                    "renderer": {"kind": "rule_based"},
                    "runtime": {
                        "python": platform.python_version(),
                        "math_verifier_mode": "string_normalization_fallback",
                        "dependencies": dependencies,
                    },
                    "compatibility_patches": list(BRIDGE_COMPATIBILITY_PATCHES),
                },
                expected_python_version=platform.python_version(),
            )

    def test_readiness_reports_missing_bridge_without_executing_checkout(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_audit_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            _checkout(root, with_bridge=False)
            gsm8k = Path(temp) / "test.jsonl"
            _gsm8k(gsm8k)
            report = audit_reproduction(
                gsm8k_test_path=gsm8k,
                bridge_path=None,
                environment={ROOT_ENVIRONMENT_VARIABLE: str(root)},
            )
            self.assertFalse(report.ready)
            self.assertIn(
                "external_bridge_missing",
                {issue.code for issue in report.issues if issue.blocking},
            )
            self.assertIn(
                "upstream_generated_dataset_unreleased",
                {issue.code for issue in report.issues if not issue.blocking},
            )

    def test_readiness_defaults_to_and_attests_checkout_venv_python(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_runtime_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            bridge_path = _checkout(root)
            assert bridge_path is not None
            gsm8k = Path(temp) / "test.jsonl"
            _gsm8k(gsm8k)
            report = audit_reproduction(
                gsm8k_test_path=gsm8k,
                bridge_path=bridge_path,
                environment={ROOT_ENVIRONMENT_VARIABLE: str(root)},
            )
            self.assertTrue(report.ready)
            runtime = report.bridge_runtime
            assert runtime is not None
            self.assertEqual(runtime.python_path, str(root / ".venv/bin/python"))
            self.assertEqual(runtime.python_version, platform.python_version())
            self.assertEqual(runtime.python_version_info[:2], (3, 12))
            self.assertEqual(runtime.implementation, "cpython")
            self.assertEqual(
                runtime.dependency_lock_sha256,
                sha256_file(runtime.dependency_lock_path),
            )
            self.assertEqual(len(runtime.attestation_sha256), 64)
            roles = {artifact.role for artifact in report.input_artifacts}
            self.assertIn("bridge_python_executable", roles)
            self.assertIn("bridge_python_venv_config", roles)
            serialized = report.as_dict()["bridge_runtime"]
            self.assertEqual(serialized["attestation_sha256"], runtime.attestation_sha256)

    def test_readiness_rejects_missing_or_outside_interpreter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_runtime_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            bridge_path = _checkout(root)
            assert bridge_path is not None
            gsm8k = Path(temp) / "test.jsonl"
            _gsm8k(gsm8k)
            common = {
                "gsm8k_test_path": gsm8k,
                "bridge_path": bridge_path,
                "environment": {ROOT_ENVIRONMENT_VARIABLE: str(root)},
            }
            missing = audit_reproduction(
                **common,
                bridge_python_path=root / ".venv/bin/not-python",
            )
            self.assertFalse(missing.ready)
            self.assertIn(
                "bridge_python_unavailable",
                {issue.code for issue in missing.issues if issue.blocking},
            )
            outside = audit_reproduction(
                **common,
                bridge_python_path=sys.executable,
            )
            self.assertFalse(outside.ready)
            self.assertIn(
                "bridge_python_outside_checkout_venv",
                {issue.code for issue in outside.issues if issue.blocking},
            )

    def test_subprocess_bridge_fails_closed_after_runtime_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_runtime_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            bridge_path = _checkout(root)
            assert bridge_path is not None
            python_path = root / ".venv/bin/python"
            bridge = SubprocessUpstreamBridge(root, bridge_path, python_path)
            config = root / ".venv/pyvenv.cfg"
            config.write_text(
                config.read_text(encoding="utf-8") + "\n# mutated after audit\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReadinessBlocked, "changed after readiness"):
                bridge.exchange({"operation": "must-not-run"})

    def test_subprocess_bridge_fails_closed_after_interpreter_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_runtime_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            bridge_path = _checkout(root)
            assert bridge_path is not None
            python_path = root / ".venv/bin/python"
            bridge = SubprocessUpstreamBridge(root, bridge_path, python_path)
            python_path.unlink()
            python_path.symlink_to("/bin/false")
            with self.assertRaisesRegex(ReadinessBlocked, "interpreter failed its probe"):
                bridge.exchange({"operation": "must-not-run"})

    def test_capabilities_reject_runtime_version_mismatch(self) -> None:
        dependencies = {
            name: {"available": True, "version": version}
            for name, version in BRIDGE_RUNTIME_DEPENDENCIES.items()
        }
        with self.assertRaisesRegex(BridgeProtocolError, "Python version"):
            _validate_capabilities(
                {
                    "protocol": BRIDGE_PROTOCOL,
                    "contract_sha256": BRIDGE_CONTRACT_SHA256,
                    "upstream_commit": PINNED_COMMIT,
                    "transport_mode": "emit_requests_only",
                    "stages": list(STAGES),
                    "renderer": {"kind": "rule_based"},
                    "runtime": {
                        "python": "0.0.0",
                        "math_verifier_mode": "math_verify",
                        "dependencies": dependencies,
                    },
                    "compatibility_patches": list(BRIDGE_COMPATIBILITY_PATCHES),
                },
                expected_python_version=platform.python_version(),
            )

    def test_source_selection_uses_zero_based_official_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_source_") as temp:
            root = Path(temp)
            gsm8k = root / "test.jsonl"
            _gsm8k(gsm8k)
            ids = root / "ids.json"
            _write_json(ids, {"task-12": 12, "task-14": 14, "task-16": 16})
            tasks, inputs = select_source_tasks(gsm8k, ids, (12, 14, 16))
            self.assertEqual([task.task_id for task in tasks], ["task-12", "task-14", "task-16"])
            self.assertEqual(tasks[0].question, "Question 12?")
            self.assertEqual(len(inputs), 2)

    def test_source_selection_accepts_released_manifest_wrapper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_source_") as temp:
            root = Path(temp)
            gsm8k = root / "test.jsonl"
            _gsm8k(gsm8k)
            ids = root / "ids.json"
            _write_json(
                ids,
                {
                    "dataset": "gsm8k",
                    "split": "test",
                    "samples": [
                        {"task_id": "task-12", "original_id": 12},
                        {"task_id": "task-14", "original_id": 14},
                        {"task_id": "task-16", "original_id": 16},
                    ],
                },
            )
            tasks, _ = select_source_tasks(gsm8k, ids, (12, 14, 16))
            self.assertEqual([task.task_id for task in tasks], ["task-12", "task-14", "task-16"])

    def test_subprocess_bridge_receives_json_without_provider_credentials(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_bridge_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            bridge_path = _checkout(root)
            assert bridge_path is not None
            bridge_path.write_text(
                "import json, os, sys\n"
                "request = json.load(sys.stdin)\n"
                "json.dump({'operation': request['operation'], "
                "'has_openai_key': 'OPENAI_API_KEY' in os.environ}, sys.stdout)\n",
                encoding="utf-8",
            )
            bridge = SubprocessUpstreamBridge(
                root, bridge_path, root / ".venv/bin/python"
            )
            response = bridge.exchange({"operation": "fixture"})
            self.assertEqual(response, {"operation": "fixture", "has_openai_key": False})

    def test_subprocess_bridge_honors_frozen_hash_seed_across_processes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="evolving_bridge_seed_") as temp:
            root = Path(temp) / "upstream"
            root.mkdir()
            bridge_path = _checkout(root)
            assert bridge_path is not None
            bridge_path.write_text(
                "import json, os, sys\n"
                "request = json.load(sys.stdin)\n"
                "items = list(set(request['items']))\n"
                "json.dump({'items': items, 'seed': os.environ.get('PYTHONHASHSEED'), "
                "'no_user_site': os.environ.get('PYTHONNOUSERSITE')}, sys.stdout)\n",
                encoding="utf-8",
            )
            bridge = SubprocessUpstreamBridge(
                root, bridge_path, root / ".venv/bin/python"
            )
            responses = [
                bridge.exchange({"items": ["gamma", "alpha", "beta", "delta"]})
                for _ in range(12)
            ]
            self.assertTrue(all(response == responses[0] for response in responses))
            self.assertEqual(responses[0]["seed"], "42")
            self.assertEqual(responses[0]["no_user_site"], "1")


class BuilderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="evolving_builder_")
        self.base = Path(self.temp.name)
        self.root = self.base / "upstream"
        self.root.mkdir()
        bridge_path = _checkout(self.root)
        assert bridge_path is not None
        self.bridge_path = bridge_path
        self.gsm8k = self.base / "test.jsonl"
        _gsm8k(self.gsm8k)
        self.output = self.base / "build"
        self.readiness = audit_reproduction(
            gsm8k_test_path=self.gsm8k,
            bridge_path=bridge_path,
            environment={ROOT_ENVIRONMENT_VARIABLE: str(self.root)},
        )
        self.assertTrue(self.readiness.ready)
        assert self.readiness.bridge_runtime is not None
        self.runtime = self.readiness.bridge_runtime

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    def _builder(self, transport: FakeTransport, bridge: FakeBridge) -> EvolvingReproductionBuilder:
        return EvolvingReproductionBuilder(
            readiness=self.readiness,
            gsm8k_test_path=self.gsm8k,
            source_ids=(12, 14, 16),
            output_dir=self.output,
            settings=BuildSettings(
                generator_model="generator-model",
                judge_model="judge-model",
                generator_reasoning_effort="medium",
                judge_reasoning_effort="low",
            ),
            transport=transport,
            bridge=bridge,
        )

    async def test_build_freezes_strict_pairs_hashes_calls_and_resumes(self) -> None:
        transport = FakeTransport()
        bridge = FakeBridge(self.root, self.bridge_path, self.runtime)
        builder = self._builder(transport, bridge)
        result = await builder.build(authorize_model_calls=True)

        self.assertEqual(result.num_source_tasks, 3)
        self.assertEqual(result.num_condition_records, 6)
        self.assertEqual(len(transport.calls), 9)
        self.assertEqual(
            [call["model"] for call in transport.calls[:3]],
            ["generator-model", "generator-model", "judge-model"],
        )
        frozen = read_json(result.dataset_path)
        serialized = json.dumps(frozen, sort_keys=True).lower()
        self.assertTrue(frozen["shared_across_target_arms_and_models"])
        self.assertNotIn("change_plan", serialized)
        self.assertNotIn('"gold"', serialized)
        tasks = EvolvingIntentAdapter(result.dataset_path).load_tasks()
        self.assertEqual([(task.task_id, task.condition) for task in tasks[:2]], [("task-12", "t1"), ("task-12", "t7")])

        receipt = read_json(result.receipt_path)
        self.assertEqual(len(receipt["calls"]), 9)
        self.assertEqual(receipt["calls"][0]["attempt_event_ids"], ["event-1"])
        self.assertIn("prompt_sha256", receipt["calls"][0])
        self.assertEqual(
            sha256_file(receipt["calls"][0]["request_artifact"]["path"]),
            receipt["calls"][0]["request_artifact"]["sha256"],
        )
        self.assertEqual(
            sha256_file(receipt["calls"][0]["response_artifact"]["path"]),
            receipt["calls"][0]["response_artifact"]["sha256"],
        )
        self.assertEqual(receipt["frozen_dataset"]["sha256"], result.dataset_sha256)
        self.assertEqual(receipt["bridge_runtime"], self.runtime.as_dict())
        config = read_json(self.output / "build_config.json")
        self.assertEqual(config["bridge"]["runtime"], self.runtime.as_dict())

        calls_before = len(transport.calls)
        resumed = await self._builder(transport, bridge).build(authorize_model_calls=True)
        self.assertEqual(len(transport.calls), calls_before)
        self.assertEqual(resumed.dataset_sha256, result.dataset_sha256)

    async def test_unresolved_call_blocks_resume_instead_of_double_spending(self) -> None:
        transport = FakeTransport()
        bridge = FakeBridge(self.root, self.bridge_path, self.runtime)
        builder = self._builder(transport, bridge)
        self.output.mkdir(parents=True)
        atomic_write_json(self.output / "build_config.json", builder._build_config)
        task_dir = self.output / "tasks/12"
        task_dir.mkdir(parents=True)
        task = builder.tasks[0]
        atomic_write_json(
            task_dir / "source.json",
            {
                "schema_version": 1,
                "build_sha256": builder.build_sha256,
                "source": task.private_dict(),
                "source_sha256": sha256_json(task.private_dict()),
            },
        )
        input_hash = sha256_json(
            {
                "build_sha256": builder.build_sha256,
                "stage": STAGES[0],
                "source": task.private_dict(),
                "prior_artifacts": {},
            }
        )
        atomic_write_json(
            task_dir / f"{STAGES[0]}.work.json",
            {
                "schema_version": 1,
                "input_sha256": input_hash,
                "status": "call_pending",
                "calls": [],
            },
        )
        with self.assertRaisesRegex(ResumeBlocked, "unresolved provider call"):
            await builder.build(authorize_model_calls=True)
        self.assertEqual(transport.calls, [])

    async def test_saved_response_recovers_after_crash_without_duplicate_call(self) -> None:
        transport = FakeTransport()
        bridge = FakeBridge(self.root, self.bridge_path, self.runtime)
        builder = self._builder(transport, bridge)
        real_atomic_write = atomic_write_json
        crashed = False

        def fail_once(path, value, **kwargs):
            nonlocal crashed
            if (
                not crashed
                and str(path).endswith(".work.json")
                and isinstance(value, dict)
                and value.get("status") == "result_ready"
            ):
                crashed = True
                raise OSError("synthetic crash after response persistence")
            return real_atomic_write(path, value, **kwargs)

        with patch("experiments12.build_evolving12.atomic_write_json", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "synthetic crash"):
                await builder.build(authorize_model_calls=True)
        self.assertEqual(len(transport.calls), 1)

        result = await self._builder(transport, bridge).build(authorize_model_calls=True)
        self.assertEqual(len(transport.calls), 9)
        self.assertEqual(result.num_condition_records, 6)


if __name__ == "__main__":
    unittest.main()
