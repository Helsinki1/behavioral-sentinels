from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments12.analysis12 import _manifest_analysis_config
from experiments12.bfcl_run12 import (
    BFCL_EXPECTED_INPUT_ROLES,
    _default_transport_factory,
    execute_bfcl_run,
    freeze_selected_official_tasks,
    initialize_bfcl_run,
    validate_bfcl_run,
)
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
)
from experiments12.core.budget import BudgetLedger
from experiments12.core.schemas import CallAttemptRecord, CallStatus, TokenUsage
from experiments12.core.transport import CompletionResult
from experiments12.domains.bfcl import LICENSE_IDENTIFIER, PINNED_COMMIT
from experiments12.manifest12 import ArtifactReceipt
from experiments12.passive_spec12 import effective_passive_method_names
from experiments12.runner12 import load_pair_cells, load_task_manifest
from experiments12.source_registry12 import load_source_registry
from experiments12.spec12 import Stage
from experiments12.scripts.tests.test_bfcl_runner12 import (
    FakeBridge,
    FakeTransport,
    make_task,
)


class LoadableFakeBridge(FakeBridge):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tuple(tasks)
        self.load_calls: list[tuple[tuple[str, ...], tuple[str, ...] | None]] = []

    def load_tasks(self, *, categories, task_ids=None):
        ids = None if task_ids is None else tuple(task_ids)
        self.load_calls.append((tuple(categories), ids))
        return tuple(
            task
            for task in self.tasks
            if task.category in categories and (not ids or task.task_id in ids)
        )


class ShadowCapableTransport(FakeTransport):
    async def complete(self, model, messages, **kwargs):
        purpose = kwargs["purpose"]
        if purpose not in {"frozen_probe", "frozen_quiz", "trace_judge"}:
            return await super().complete(model, messages, **kwargs)
        self.counter += 1
        copied_messages = json.loads(json.dumps(messages))
        self.calls.append({"model": model, "messages": copied_messages, **kwargs})
        if purpose == "frozen_probe":
            text = "PROBE: 00000000"
        elif purpose == "trace_judge":
            text = '{"risk":0.25,"concerns":[],"evidence":[]}'
        else:
            text = "A1: 1\nA2: 0\nA3: none"
        usage = TokenUsage(input_tokens=120, output_tokens=8)
        attempt = CallAttemptRecord(
            event_id=f"monitor-event-{self.counter}",
            reservation_id=f"monitor-reservation-{self.counter}",
            provider="openai",
            model=model,
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            usage=usage,
            estimated_cost_usd=Decimal("0.001"),
            elapsed_ms=10,
        )
        return CompletionResult(
            text=text,
            tool_calls=(),
            usage=usage,
            response_id=f"monitor-response-{self.counter}",
            request_id=f"monitor-request-{self.counter}",
            model_id=model,
            finish_reason="stop",
            cost_usd=Decimal("0.001"),
            attempts=(attempt,),
        )


class BFCLCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_bfcl_run_")
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.task_path = self.root / "bfcl_tasks.jsonl"
        self.provenance = tuple(
            ArtifactReceipt(
                name=name,
                path=f"external:{name.rsplit(':', 1)[-1]}",
                sha256=hashlib.sha256(name.encode("utf-8")).hexdigest(),
                upstream_commit=PINNED_COMMIT,
                license_id=LICENSE_IDENTIFIER,
            )
            for name in sorted(BFCL_EXPECTED_INPUT_ROLES)
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_default_transport_factory_binds_append_only_event_log(self) -> None:
        ledger = BudgetLedger(self.root / "factory-ledger.sqlite3")
        event_log = self.root / "factory-events.jsonl"
        transport = _default_transport_factory(ledger, event_log, {}, 3)
        self.assertIs(transport.ledger, ledger)
        self.assertEqual(transport.event_log_path, event_log)
        self.assertEqual(transport.max_attempts, 3)

    def freeze(self, tasks=None, *, one_task_smoke=True):
        tasks = tuple(tasks or (make_task(),))
        bridge = LoadableFakeBridge(tasks)
        selection = freeze_selected_official_tasks(
            bridge=bridge,
            output_path=self.task_path,
            categories=tuple(sorted({task.category for task in tasks})),
            one_task_smoke=one_task_smoke,
        )
        return selection, bridge

    def initialize(self, *, arms=("clean",), stage=Stage.SMOKE):
        return initialize_bfcl_run(
            run_id="bfcl-smoke",
            stage=stage,
            task_manifest_path=self.task_path,
            models=("gpt-5.6-luna",),
            arms=arms,
            artifacts_root=self.artifacts,
            seed=17,
            benchmark_receipts=self.provenance,
        )

    def test_freeze_one_task_and_initialize_with_common_manifest(self) -> None:
        tasks = (
            make_task("z-task", category="multi_turn_miss_param"),
            make_task("a-task", category="multi_turn_base"),
        )
        selection, _bridge = self.freeze(tasks)
        self.assertEqual(selection.task_ids, ("a-task",))
        rows = load_task_manifest(self.task_path)
        self.assertEqual(rows[0]["task_id"], "a-task::official_native_tools")
        self.assertNotIn("tools", rows[0])

        layout = self.initialize(arms=("clean", "active_counter"))
        manifest = read_json(layout.manifest)
        self.assertEqual(manifest["stage"], "smoke")
        self.assertEqual(manifest["extra_config"]["n_cells"], 2)
        self.assertEqual(layout.ledger, self.artifacts / "_global_budget.sqlite3")
        self.assertTrue(layout.ledger.exists())
        cells = load_pair_cells(layout.pairs)
        self.assertEqual({cell.operator for cell in cells}, {"none"})

        runtime_bridge = LoadableFakeBridge((tasks[1],))
        validated = validate_bfcl_run(
            layout=layout,
            task_manifest_path=self.task_path,
            task_records=runtime_bridge.tasks,
            benchmark_receipts=self.provenance,
        )
        self.assertEqual(len(validated.cells), 2)

    def test_three_task_smoke_is_representable_and_two_tasks_fail_closed(self) -> None:
        tasks = tuple(make_task(f"task-{index}") for index in range(3))
        self.freeze(tasks, one_task_smoke=False)
        layout = self.initialize()
        manifest = read_json(layout.manifest)
        self.assertEqual(manifest["extra_config"]["smoke_wave"], "single_model")
        self.assertEqual(manifest["extra_config"]["n_public_tasks"], 3)

        other = self.root / "two.jsonl"
        freeze_selected_official_tasks(
            bridge=LoadableFakeBridge(tasks[:2]),
            output_path=other,
            categories=("multi_turn_base",),
        )
        with self.assertRaisesRegex(ValueError, "1, 3, or 5"):
            initialize_bfcl_run(
                run_id="invalid-two-task-smoke",
                stage=Stage.SMOKE,
                task_manifest_path=other,
                models=("gpt-5.6-luna",),
                arms=("clean",),
                artifacts_root=self.artifacts,
                benchmark_receipts=self.provenance,
            )

    def test_runtime_provenance_must_exactly_match_frozen_receipts(self) -> None:
        self.freeze()
        layout = self.initialize()
        changed = list(self.provenance)
        changed[0] = replace(changed[0], sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "runtime inputs differ"):
            validate_bfcl_run(
                layout=layout,
                task_manifest_path=self.task_path,
                task_records=(make_task(),),
                benchmark_receipts=tuple(changed),
            )

    def test_scientific_gate_failure_writes_no_bfcl_run_or_ledger(self) -> None:
        sources = load_source_registry()["benchmarks"]["bfcl_multi_turn"][
            "allocations"
        ]["baseline_gate"]["source_ids"]
        tasks = tuple(
            make_task(source, category=source.rsplit("_", 1)[0])
            for source in sources
        )
        self.freeze(tasks, one_task_smoke=False)
        with self.assertRaisesRegex(ValueError, "source allocation registry"):
            self.initialize(stage=Stage.BASELINE_GATE)
        self.assertFalse((self.artifacts / "bfcl-smoke").exists())
        self.assertFalse((self.artifacts / "_global_budget.sqlite3").exists())

    def test_confirmatory_init_requires_and_binds_threshold_artifact(self) -> None:
        self.freeze()
        threshold = self.root / "bfcl-thresholds.json"
        atomic_write_json(
            threshold,
            {
                "artifact_type": "locked_fixed_rate_thresholds",
                "source_manifest_sha256": "a" * 64,
                "required_passive_methods": list(effective_passive_method_names()),
                "calibration_source_tasks": [
                    {
                        "benchmark": "bfcl_multi_turn",
                        "source_task_id": "multi_turn_base_20",
                    }
                ],
            },
        )
        with patch("experiments12.bfcl_run12.assert_scientific_launch", return_value=None):
            with self.assertRaisesRegex(ValueError, "requires --thresholds"):
                initialize_bfcl_run(
                    run_id="bfcl-confirmatory-missing-threshold",
                    stage=Stage.CONFIRMATORY,
                    task_manifest_path=self.task_path,
                    models=("gpt-5.6-luna",),
                    arms=("clean", "active_recompute"),
                    artifacts_root=self.artifacts,
                    benchmark_receipts=self.provenance,
                )
            layout = initialize_bfcl_run(
                run_id="bfcl-confirmatory",
                stage=Stage.CONFIRMATORY,
                task_manifest_path=self.task_path,
                models=("gpt-5.6-luna",),
                arms=("clean", "active_recompute"),
                artifacts_root=self.artifacts,
                benchmark_receipts=self.provenance,
                threshold_path=threshold,
            )
        self.assertFalse(
            (self.artifacts / "bfcl-confirmatory-missing-threshold").exists()
        )
        manifest = read_json(layout.manifest)
        expected = {
            "threshold_artifact_sha256": sha256_file(threshold),
            "calibration_manifest_sha256": "a" * 64,
        }
        self.assertEqual(manifest["extra_config"]["analysis_lock"], expected)
        _required_passive, analysis_lock = _manifest_analysis_config(manifest)
        self.assertEqual(analysis_lock, expected)

    async def test_spend_gate_precedes_bridge_and_transport(self) -> None:
        bridge = LoadableFakeBridge((make_task(),))
        made_transport = False

        def factory(*_args):
            nonlocal made_transport
            made_transport = True
            return FakeTransport()

        with self.assertRaisesRegex(PermissionError, "--yes-spend"):
            await execute_bfcl_run(
                run_id="not-initialized",
                task_manifest_path=self.root / "missing.jsonl",
                bridge=bridge,
                benchmark_receipts=self.provenance,
                yes_spend=False,
                artifacts_root=self.artifacts,
                transport_factory=factory,
            )
        self.assertEqual(bridge.load_calls, [])
        self.assertFalse(made_transport)

    async def test_runtime_monitor_flags_cannot_override_frozen_manifest(self) -> None:
        self.freeze()
        self.initialize()
        bridge = LoadableFakeBridge((make_task(),))
        made_transport = False

        def factory(*_args):
            nonlocal made_transport
            made_transport = True
            return FakeTransport()

        with self.assertRaisesRegex(ValueError, "judge setting"):
            await execute_bfcl_run(
                run_id="bfcl-smoke",
                task_manifest_path=self.task_path,
                bridge=bridge,
                benchmark_receipts=self.provenance,
                yes_spend=True,
                artifacts_root=self.artifacts,
                run_judge=False,
                transport_factory=factory,
            )
        self.assertFalse(made_transport)
        with self.assertRaisesRegex(ValueError, "judge model"):
            await execute_bfcl_run(
                run_id="bfcl-smoke",
                task_manifest_path=self.task_path,
                bridge=bridge,
                benchmark_receipts=self.provenance,
                yes_spend=True,
                artifacts_root=self.artifacts,
                judge_model="gpt-5.6-luna",
                transport_factory=factory,
            )
        self.assertFalse(made_transport)

    async def test_max_new_cells_uses_shared_run_scoped_stage_ledger(self) -> None:
        selection, _ = self.freeze()
        self.assertEqual(selection.task_ids, ("bfcl-task-1",))
        layout = self.initialize(arms=("clean", "active_counter"))
        bridge = LoadableFakeBridge((make_task(),))
        transports: list[FakeTransport] = []
        ledgers = []
        max_attempts: list[int] = []

        def factory(ledger, event_path, environ, attempts):
            del event_path, environ
            ledgers.append(ledger)
            max_attempts.append(attempts)
            transport = FakeTransport()
            transports.append(transport)
            return transport

        first = await execute_bfcl_run(
            run_id="bfcl-smoke",
            task_manifest_path=self.task_path,
            bridge=bridge,
            benchmark_receipts=self.provenance,
            yes_spend=True,
            artifacts_root=self.artifacts,
            phase="trajectories",
            max_new_cells=1,
            transport_factory=factory,
        )
        self.assertEqual(first.completed_cells, 1)
        self.assertEqual(first.skipped_cells, 1)
        self.assertEqual(len(tuple(layout.trajectories.glob("*.json"))), 1)
        self.assertEqual(ledgers[0].path, layout.ledger)
        self.assertEqual(ledgers[0].request_scope, "bfcl-smoke")
        self.assertEqual(max_attempts, [3])
        self.assertTrue(
            all(
                str(call["request_key"]).startswith("bfcl-smoke/")
                for call in transports[0].calls
            )
        )

        second = await execute_bfcl_run(
            run_id="bfcl-smoke",
            task_manifest_path=self.task_path,
            bridge=bridge,
            benchmark_receipts=self.provenance,
            yes_spend=True,
            artifacts_root=self.artifacts,
            phase="trajectories",
            max_new_cells=1,
            transport_factory=factory,
        )
        self.assertEqual(second.completed_cells, 2)
        self.assertEqual(second.skipped_cells, 1)
        self.assertEqual(len(tuple(layout.trajectories.glob("*.json"))), 2)

    async def test_two_shards_issue_no_duplicate_provider_requests(self) -> None:
        self.freeze()
        layout = self.initialize(arms=("clean", "active_counter"))
        bridge = LoadableFakeBridge((make_task(),))
        transports: list[FakeTransport] = []

        def factory(*_args):
            transport = FakeTransport()
            transports.append(transport)
            return transport

        summaries = []
        for shard_index in range(2):
            summaries.append(
                await execute_bfcl_run(
                    run_id="bfcl-smoke",
                    task_manifest_path=self.task_path,
                    bridge=bridge,
                    benchmark_receipts=self.provenance,
                    yes_spend=True,
                    artifacts_root=self.artifacts,
                    phase="trajectories",
                    max_new_cells=1,
                    shard_count=2,
                    shard_index=shard_index,
                    transport_factory=factory,
                )
            )
        self.assertEqual(len(tuple(layout.trajectories.glob("*.json"))), 2)
        self.assertEqual([summary.declared_cells for summary in summaries], [2, 2])
        self.assertEqual([summary.shard_cells for summary in summaries], [1, 1])
        self.assertEqual([summary.visited_cells for summary in summaries], [1, 1])
        request_sets = [
            {str(call["request_key"]) for call in transport.calls}
            for transport in transports
        ]
        self.assertTrue(request_sets[0])
        self.assertTrue(request_sets[1])
        self.assertTrue(request_sets[0].isdisjoint(request_sets[1]))

        replay = await execute_bfcl_run(
            run_id="bfcl-smoke",
            task_manifest_path=self.task_path,
            bridge=bridge,
            benchmark_receipts=self.provenance,
            yes_spend=True,
            artifacts_root=self.artifacts,
            phase="trajectories",
            shard_count=2,
            shard_index=0,
            transport_factory=factory,
        )
        self.assertEqual(replay.visited_cells, 0)
        self.assertEqual(transports[-1].calls, [])

    async def test_partial_trajectory_is_never_retried(self) -> None:
        self.freeze()
        layout = self.initialize()
        cell = load_pair_cells(layout.pairs)[0]
        partial = layout.events / f"trajectory-{cell.cell_id}.jsonl"
        atomic_write_jsonl(partial, [{"event": "start", "partial": True}])
        bridge = LoadableFakeBridge((make_task(),))
        transport = FakeTransport()

        with self.assertRaisesRegex(FileExistsError, "partial BFCL artifacts"):
            await execute_bfcl_run(
                run_id="bfcl-smoke",
                task_manifest_path=self.task_path,
                bridge=bridge,
                benchmark_receipts=self.provenance,
                yes_spend=True,
                artifacts_root=self.artifacts,
                phase="trajectories",
                transport_factory=lambda *_args: transport,
            )
        self.assertEqual(transport.calls, [])
        self.assertEqual(bridge.begin_calls, [])

    async def test_clean_shadow_gets_checkpoint_specific_bfcl_quiz(self) -> None:
        self.freeze()
        layout = self.initialize()
        bridge = LoadableFakeBridge((make_task(),))
        transport = ShadowCapableTransport()
        summary = await execute_bfcl_run(
            run_id="bfcl-smoke",
            task_manifest_path=self.task_path,
            bridge=bridge,
            benchmark_receipts=self.provenance,
            yes_spend=True,
            artifacts_root=self.artifacts,
            phase="both",
            transport_factory=lambda *_args: transport,
        )
        self.assertEqual(summary.completed_cells, 2)
        shadow_paths = tuple(layout.shadow.glob("*.json"))
        self.assertEqual(len(shadow_paths), 1)
        shadow = read_json(shadow_paths[0])
        quizzes = [record for record in shadow["records"] if record["method"] == "frozen_quiz"]
        self.assertEqual([record["checkpoint_turn"] for record in quizzes], [1])
        quiz_call = next(call for call in transport.calls if call["purpose"] == "frozen_quiz")
        quiz_prompt = quiz_call["messages"][-1]["content"]
        self.assertIn("How many BFCL user turns", quiz_prompt)
        self.assertIn("How many official tool results", quiz_prompt)
        self.assertTrue(str(quiz_call["request_key"]).startswith("bfcl-smoke/shadow/"))


if __name__ == "__main__":
    unittest.main()
