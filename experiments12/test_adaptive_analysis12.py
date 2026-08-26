from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from experiments12.adaptive_analysis12 import (
    ADAPTIVE_ANALYSIS_TYPE,
    _attempt_totals,
    _require_start_runtime_binding,
    extract_adaptive_run,
    observation_class,
    summarize_adaptive_outcomes,
    write_adaptive_figures,
)
from experiments12.adaptive_deployment12 import (
    ADAPTIVE_DEPLOYMENT_MODE,
    ADAPTIVE_POLICY,
    _runtime_config,
    execute_adaptive_run,
)
from experiments12.analysis12 import AnalysisInputError, _AttemptResource
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetLedger
from experiments12.core.schemas import CallStatus
from experiments12.core.transport import Transport
from experiments12.deployment12 import (
    THRESHOLD_LOCK_RECEIPT,
    LockedMethodThreshold,
    ThresholdLockArtifact,
    freeze_threshold_lock,
)
from experiments12.domains.evolving_intent import EvolvingIntentAdapter, PINNED_COMMIT
from experiments12.harness12 import HarnessConfig
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.operators12 import CompactionConfig
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.planning_lock12 import ScientificLaunchBinding
from experiments12.runner12 import freeze_task_manifest, pair_task_id
from experiments12.source_registry12 import SourceAllocationBinding
from experiments12.spec12 import OPERATIONAL_PROVIDER_USD, Operator, Stage


ROOT = Path(__file__).resolve().parent.parent
MODEL = "gpt-5.6-luna"


class _MockResponse:
    def __init__(self, payload: object, request_id: str) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.status = 200
        self.headers = {"X-Request-ID": request_id}

    def read(self, limit: int = -1) -> bytes:
        return self.body if limit < 0 else self.body[:limit]

    def close(self) -> None:
        return None


class _MockOpenAI:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: object, *, timeout: float) -> _MockResponse:
        del timeout
        body = json.loads(request.data)
        self.calls += 1
        serialized = json.dumps(body.get("input", []), sort_keys=True)
        answer = "PROBE: 00000000" if "ACTIVE CARRIED PROBE" in serialized else "Answer: 6"
        return _MockResponse(
            {
                "id": f"response-{self.calls}",
                "model": body["model"],
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": answer}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
            f"request-{self.calls}",
        )


def rows() -> list[dict]:
    result: list[dict] = []
    outcomes = {
        ("active_recompute", "none"): (False, True),
        ("active_recompute", "compact"): (True, True),
        ("frozen_probe:recompute", "none"): (False, True),
        ("frozen_probe:recompute", "compact"): (False, False),
    }
    for method, operator in outcomes:
        for task_index, success in enumerate(outcomes[(method, operator)]):
            task_tokens = 100 + task_index
            observer_tokens = 10 if method.startswith("active_") else 8
            result.append(
                {
                    "cell_id": f"{method}-{operator}-{task_index}",
                    "model": "gpt-5.6-luna",
                    "benchmark": "evolving_intent_gsm8k",
                    "task_id": f"task-{task_index}::t7",
                    "replicate_id": 0,
                    "unit_id": f"task-{task_index}::t7/r0",
                    "method": method,
                    "observation_class": observation_class(method),
                    "operator": operator,
                    "deployment_mode": "online_adaptive",
                    "success": success,
                    "observations": 6,
                    "threshold_firings": 1,
                    "selected_actions": 1,
                    "applied_interventions": 1,
                    "task_tokens": task_tokens,
                    "observer_tokens": observer_tokens,
                    "total_tokens": task_tokens + observer_tokens,
                    "latency_ms": 50,
                    "actual_cost_usd": 0.01,
                }
            )
    return result


class AdaptiveAnalysisTests(unittest.TestCase):
    def test_start_runtime_must_equal_manifest_lock(self) -> None:
        frozen = {
            "checkpoint_every": 1,
            "task_max_output_tokens": 1800,
            "probe_max_output_tokens": 192,
            "temperature": None,
            "compaction": {
                "keep_last_messages": 4,
                "max_excerpt_bytes": 240,
                "max_summary_bytes": 1600,
                "config_sha256": "a" * 64,
            },
        }
        manifest = {"extra_config": {"adaptive_runtime": frozen}}
        _require_start_runtime_binding(
            {"runtime_config": frozen}, manifest, cell_id="cell"
        )
        changed = {**frozen, "task_max_output_tokens": 1799}
        with self.assertRaisesRegex(AnalysisInputError, "differs from its manifest"):
            _require_start_runtime_binding(
                {"runtime_config": changed}, manifest, cell_id="cell"
            )

    def test_exact_products_produce_absolute_and_paired_metrics(self) -> None:
        summaries, effects = summarize_adaptive_outcomes(
            rows(), bootstrap_iterations=40
        )
        self.assertEqual(len(summaries), 4 * 8)
        self.assertEqual(len(effects), 2 * 8)
        success = {
            (row.method, row.operator): row
            for row in effects
            if row.metric == "success"
        }
        self.assertEqual(success[("active_recompute", "compact")].effect, 0.5)
        self.assertEqual(
            success[("frozen_probe:recompute", "compact")].effect, -0.5
        )
        self.assertTrue(
            all(row.bootstrap_unit == "paired_source_task" for row in effects)
        )
        self.assertTrue(all(row.bootstrap_unit == "source_task" for row in summaries))

    def test_missing_cell_and_action_cap_fail_closed(self) -> None:
        incomplete = rows()[:-1]
        with self.assertRaisesRegex(AnalysisInputError, "unpaired treatment"):
            summarize_adaptive_outcomes(incomplete, bootstrap_iterations=2)
        changed = rows()
        changed[0]["selected_actions"] = 2
        changed[0]["applied_interventions"] = 2
        with self.assertRaisesRegex(AnalysisInputError, "one-action"):
            summarize_adaptive_outcomes(changed, bootstrap_iterations=2)

    def test_retry_resources_include_failed_attempts(self) -> None:
        attempts = {
            "failed": _AttemptResource(
                event_id="failed",
                purpose="adaptive_agent_turn",
                status=CallStatus.FAILED,
                input_tokens=9,
                output_tokens=0,
                elapsed_ms=7,
                actual_cost_usd=Decimal("0.002"),
                cost_quality="estimated",
                provider="openai",
                model=MODEL,
                attempt_number=1,
                request_key="run/cell/adaptive-task-1/attempt-1",
            ),
            "success": _AttemptResource(
                event_id="success",
                purpose="adaptive_agent_turn",
                status=CallStatus.SUCCEEDED,
                input_tokens=10,
                output_tokens=2,
                elapsed_ms=3,
                actual_cost_usd=Decimal("0.003"),
                cost_quality="reported",
                provider="openai",
                model=MODEL,
                attempt_number=2,
                request_key="run/cell/adaptive-task-1/attempt-2",
            ),
        }
        seen: set[str] = set()
        total = _attempt_totals(
            {
                "call_event_ids": ["failed", "success"],
                "resolved_model_id": MODEL,
                "response_id": "response",
                "request_id": "request",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                },
                "elapsed_ms": 10,
                "accounted_cost_usd": "0.003",
            },
            expected_purpose="adaptive_agent_turn",
            expected_request_key="run/cell/adaptive-task-1",
            expected_model=MODEL,
            attempts=attempts,
            seen=seen,
            context="test",
        )
        self.assertEqual(total["tokens"], 21)
        self.assertEqual(total["latency_ms"], 10)
        self.assertEqual(total["actual_cost_usd"], Decimal("0.005"))
        self.assertEqual(seen, set(attempts))
        changed = {
            **attempts,
            "success": replace(
                attempts["success"], request_key="run/cell/wrong/attempt-2"
            ),
        }
        with self.assertRaisesRegex(AnalysisInputError, "attempt identity"):
            _attempt_totals(
                {
                    "call_event_ids": ["failed", "success"],
                    "resolved_model_id": MODEL,
                    "response_id": "response",
                    "request_id": "request",
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                    "elapsed_ms": 10,
                    "accounted_cost_usd": "0.003",
                },
                expected_purpose="adaptive_agent_turn",
                expected_request_key="run/cell/adaptive-task-1",
                expected_model=MODEL,
                attempts=changed,
                seen=set(),
                context="test",
            )

    def test_success_figures_are_written_from_analysis_artifact(self) -> None:
        summaries, effects = summarize_adaptive_outcomes(
            rows(), bootstrap_iterations=5
        )
        analysis = {
            "artifact_type": ADAPTIVE_ANALYSIS_TYPE,
            "deployment_mode": "online_adaptive",
            "metric_summaries": [asdict(row) for row in summaries],
            "operator_effects": [asdict(row) for row in effects],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_adaptive_figures(analysis, Path(tmp))
            self.assertEqual(len(paths), 2)
            sidecar = next(path for path in paths if path.name.endswith(".data.json"))
            data = read_json(sidecar)
        self.assertEqual(data["figure_type"], "deployment_grouped_bars")
        self.assertEqual(len(data["rows"]), 4)


class AdaptiveHistoryExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_replays_canonical_task_runtime_and_rejects_coherent_omission(
        self,
    ) -> None:
        artifact_parent = ROOT / "experiments12" / "artifacts"
        artifact_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_parent) as temporary:
            fixture = Path(temporary)
            artifacts = fixture / "runs"
            dataset = fixture / "evolving.json"
            build_receipt = fixture / "build-receipt.json"
            task_manifest = fixture / "tasks.jsonl"
            registry = fixture / "source-registry.json"
            baseline = fixture / "baseline-profile.json"
            planning = fixture / "planning-lock.json"
            atomic_write_json(
                dataset,
                {
                    "tasks": [
                        {
                            "task_id": "adaptive-paper-case",
                            "condition": "t1",
                            "turns": ["What is twice three?"],
                            "label": "6",
                        },
                        {
                            "task_id": "adaptive-paper-case",
                            "condition": "t7",
                            "turns": [
                                "Start with three apples.",
                                "Keep that amount.",
                                "Remember the amount.",
                                "Do not change it.",
                                "Prepare to double it.",
                                "Use the remembered amount.",
                                "What is twice that amount?",
                            ],
                            "label": "6",
                        },
                    ]
                },
            )
            atomic_write_json(
                build_receipt,
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "upstream_commit": PINNED_COMMIT,
                    "shared_across_target_arms_and_models": True,
                    "frozen_dataset": {"sha256": sha256_file(dataset)},
                },
            )
            tasks = EvolvingIntentAdapter(
                dataset, expected_sha256=sha256_file(dataset)
            ).load_tasks()
            task = next(item for item in tasks if item.condition == "t7")
            freeze_task_manifest(task_manifest, (task,))
            atomic_write_json(registry, {"fixture": "adaptive-history"})
            atomic_write_json(baseline, {"fixture": "adaptive-history"})
            atomic_write_json(planning, {"fixture": "adaptive-history"})

            run_id = "adaptive-history-analysis-test"
            layout = RunLayout.for_run(artifacts, run_id)
            layout.create()
            methods = ("active_recompute",)
            operators = (Operator.NONE.value, Operator.COMPACT.value)
            cells = make_pair_manifest(
                tasks=(
                    TaskRef(
                        benchmark=task.domain,
                        task_id=pair_task_id(task),
                        task_sha256=task.task_sha256,
                    ),
                ),
                models=(MODEL,),
                arms=methods,
                operators=operators,
                randomization_seed=12012,
            )
            atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])
            pair_sha = sha256_file(layout.pairs)
            threshold = LockedMethodThreshold(
                model=MODEL,
                benchmark=task.domain,
                method=methods[0],
                threshold=0.0,
                target_firing_rate=1.0,
                achieved_firing_rate=1.0,
                calibration_n_tasks=1,
                calibration_digest=sha256_json({"fixture": "threshold"}),
                selection_rule="task_score_rank_hash_ties",
                tie_break_seed=12012,
                calibration_target_fire_count=1,
            )
            threshold_lock = ThresholdLockArtifact(
                calibration_run_id="adaptive-calibration-test",
                calibration_manifest_sha256="b" * 64,
                natural_max_actions_per_task=1,
                matched_actions_per_method=1,
                yoke_anchor_method=methods[0],
                methods=(threshold,),
            )
            threshold_path = layout.results / "deployment_threshold_lock.json"
            threshold_sha = freeze_threshold_lock(threshold_path, threshold_lock)
            launch = ScientificLaunchBinding(
                allocation=SourceAllocationBinding(
                    registry_sha256=sha256_file(registry),
                    benchmark=task.domain,
                    stage="deployment",
                    wave=None,
                    source_ids=(task.task_id,),
                ),
                projection_lock_sha256=sha256_file(planning),
                projected_provider_usd={"openai": "1", "fireworks": "1"},
                required_n_tasks=1,
            )
            harness = HarnessConfig(task_max_output_tokens=30)
            compaction = CompactionConfig()
            receipts = (
                ArtifactReceipt.from_file(
                    "task_manifest", task_manifest, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    "evolving_rendered_dataset", dataset, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    "evolving_build_receipt", build_receipt, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    THRESHOLD_LOCK_RECEIPT, threshold_path, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    "source_allocation_registry", registry, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    "measured_baseline_resource_profile", baseline, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    "cost_sample_size_projection_lock", planning, workspace=ROOT
                ),
            )
            manifest = build_manifest(
                run_id=run_id,
                stage=Stage.CONFIRMATORY,
                repository_root=ROOT,
                pair_manifest_sha256=pair_sha,
                models=(MODEL,),
                arms=methods,
                operators=operators,
                randomization_seed=12012,
                benchmark_receipts=receipts,
                extra_config={
                    "n_tasks": 1,
                    "n_cells": len(cells),
                    "replicates": 1,
                    "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
                    "deployment_policy": ADAPTIVE_POLICY,
                    "threshold_lock_sha256": threshold_sha,
                    "natural_max_actions_per_task": 1,
                    "calibration_manifest_sha256": (
                        threshold_lock.calibration_manifest_sha256
                    ),
                    "adaptive_runtime": _runtime_config(harness, compaction),
                    "scientific_launch_lock": launch.as_dict(),
                },
            )
            manifest_sha = write_manifest_once(layout.manifest, manifest)
            ledger = BudgetLedger(
                layout.ledger,
                operational_caps_usd={
                    provider: Decimal(str(value))
                    for provider, value in OPERATIONAL_PROVIDER_USD.items()
                },
            )
            transport = Transport(
                ledger,
                layout.events / "call_attempts.jsonl",
                environ={"OPENAI_API_KEY": "mock-only"},
                urlopen=_MockOpenAI(),
            )
            summary = await execute_adaptive_run(
                run_id=run_id,
                task_manifest_path=task_manifest,
                threshold_lock_path=threshold_path,
                tasks=(task,),
                yes_spend=True,
                artifacts_root=artifacts,
                config=harness,
                compaction_config=compaction,
                transport=transport,
                evolving_dataset_path=dataset,
                evolving_build_receipt_path=build_receipt,
            )
            self.assertEqual(summary.completed_cells, len(cells))
            analysis = extract_adaptive_run(
                layout,
                expected_manifest_sha256=manifest_sha,
                bootstrap_iterations=2,
            )
            self.assertEqual(len(analysis["rows"]), len(cells))

            compact_cell = next(
                cell for cell in cells if cell.operator == Operator.COMPACT.value
            )
            output_path = (
                layout.results / "adaptive_deployment" / f"{compact_cell.cell_id}.json"
            )
            event_path = layout.events / f"adaptive-{compact_cell.cell_id}.jsonl"
            job_path = (
                layout.results
                / "adaptive_deployment_jobs"
                / f"{compact_cell.cell_id}.json"
            )
            output = read_json(output_path)
            events = read_jsonl(event_path)
            second_task = next(
                row
                for row in events
                if row.get("event") == "task_turn" and row.get("task_turn") == 2
            )
            second_task["request_prefix_sha256"] = sha256_json(
                [second_task["user_message"]]
            )
            output["task_records"] = [
                row for row in events[:-1] if row.get("event") == "task_turn"
            ]
            output["messages"] = []
            output["transcript_sha256"] = sha256_json([])
            output["event_log_prefix_sha256"] = sha256_json(events[:-1])
            atomic_write_json(output_path, output)
            events[-1]["transcript_sha256"] = output["transcript_sha256"]
            events[-1]["output_sha256"] = sha256_file(output_path)
            atomic_write_jsonl(event_path, events)
            job = read_json(job_path)
            job["output_sha256"] = sha256_file(output_path)
            atomic_write_json(job_path, job)
            with self.assertRaisesRegex(
                AnalysisInputError, "request/history|final carried history"
            ):
                extract_adaptive_run(
                    layout,
                    expected_manifest_sha256=manifest_sha,
                    bootstrap_iterations=2,
                )


if __name__ == "__main__":
    unittest.main()
