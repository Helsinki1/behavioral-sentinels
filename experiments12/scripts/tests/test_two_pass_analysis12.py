from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

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
    DEPLOYMENT_SCHEDULE_RECEIPT,
    PASS_ONE_RECEIPT,
    THRESHOLD_LOCK_RECEIPT,
    DeploymentEstimand,
    LockedMethodThreshold,
    PassOneCheckpoint,
    PassOneMethodTrace,
    PassOneObservationArtifact,
    ThresholdLockArtifact,
    build_deployment_schedule,
    deployment_runtime_config,
    execute_deployment_run,
    freeze_deployment_schedule,
    freeze_pass_one_observations,
    freeze_threshold_lock,
)
from experiments12.domains.evolving_intent import EvolvingIntentAdapter, PINNED_COMMIT
from experiments12.harness12 import ARM_TO_PROBE, HarnessConfig
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.runner12 import freeze_task_manifest, pair_task_id
from experiments12.spec12 import OPERATIONAL_PROVIDER_USD, Operator, Stage
from experiments12.two_pass_analysis12 import (
    TWO_PASS_ANALYSIS_TYPE,
    _attempt_totals,
    extract_two_pass_run,
    summarize_two_pass_outcomes,
    validate_two_pass_run,
    write_two_pass_figures,
    write_two_pass_tables,
)


ROOT = Path(__file__).resolve().parents[3]
MODEL = "gpt-5.6-luna"
METHODS = ("active_recompute", "trace_rules")
OPERATORS = (Operator.NONE.value, Operator.REGROUND.value)


def _rows() -> list[dict]:
    outcomes = {
        ("active_recompute", "none"): (False, True),
        ("active_recompute", "reground"): (True, True),
        ("frozen_probe:recompute", "none"): (False, True),
        ("frozen_probe:recompute", "reground"): (False, False),
    }
    rows: list[dict] = []
    for (method, operator), successes in outcomes.items():
        for task_index, success in enumerate(successes):
            task_tokens = 100 + task_index
            observer_tokens = 10 if method.startswith("active_") else 0
            scheduled = task_index
            observations = 2
            rows.append(
                {
                    "cell_id": f"{method}-{operator}-{task_index}",
                    "model": MODEL,
                    "benchmark": "evolving_intent_gsm8k",
                    "task_id": f"task-{task_index}::t7",
                    "replicate_id": 0,
                    "unit_id": f"task-{task_index}::t7/r0",
                    "observation_class": (
                        "active" if method.startswith("active_") else "passive-behavioral"
                    ),
                    "method": method,
                    "operator": operator,
                    "deployment_mode": "two_pass_frozen",
                    "estimand": "natural_threshold",
                    "success": success,
                    "outcome_source": "canonical_regrade",
                    "observations": observations,
                    "scheduled_actions": scheduled,
                    "action_rate": scheduled / observations,
                    "acted_on_task": int(scheduled > 0),
                    "applied_interventions": scheduled,
                    "task_tokens": task_tokens,
                    "observer_tokens": observer_tokens,
                    "total_tokens": task_tokens + observer_tokens,
                    "latency_ms": 50,
                    "actual_cost_usd": 0.01,
                    "reported_cost_usd": 0.0,
                    "estimated_cost_usd": 0.01,
                    "upper_bound_cost_usd": 0.0,
                    "failed_retry_attempts": 0,
                }
            )
    return rows


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


class TwoPassSummaryTests(unittest.TestCase):
    def test_retry_costs_are_attributed_by_quality_and_call_class(self) -> None:
        attempts = {
            "failed": _AttemptResource(
                event_id="failed",
                purpose="deployment_agent_turn",
                status=CallStatus.FAILED,
                input_tokens=9,
                output_tokens=0,
                elapsed_ms=7,
                actual_cost_usd=Decimal("0.002"),
                cost_quality="upper_bound",
                provider="openai",
                model="gpt-5.6-luna",
                attempt_number=1,
                request_key="run/cell/deployment-task-1/attempt-1",
            ),
            "success": _AttemptResource(
                event_id="success",
                purpose="deployment_agent_turn",
                status=CallStatus.SUCCEEDED,
                input_tokens=10,
                output_tokens=2,
                elapsed_ms=3,
                actual_cost_usd=Decimal("0.003"),
                cost_quality="estimated",
                provider="openai",
                model="gpt-5.6-luna",
                attempt_number=2,
                request_key="run/cell/deployment-task-1/attempt-2",
            ),
        }
        total = _attempt_totals(
            {
                "call_event_ids": ["failed", "success"],
                "resolved_model_id": "gpt-5.6-luna",
                "response_id": "response",
                "request_id": "request",
                "finish_reason": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                },
                "accounted_cost_usd": "0.003",
                "elapsed_ms": 10,
            },
            expected_purpose="deployment_agent_turn",
            expected_request_key="run/cell/deployment-task-1",
            expected_model=MODEL,
            attempts=attempts,
            seen=set(),
            context="test",
        )
        self.assertEqual(total["tokens"], 21)
        self.assertEqual(total["actual_cost_usd"], Decimal("0.005"))
        self.assertEqual(total["quality_costs"]["upper_bound"], Decimal("0.002"))
        self.assertEqual(total["quality_costs"]["estimated"], Decimal("0.003"))
        self.assertEqual(total["failed_retries"], 1)

    def test_absolute_operator_and_method_paired_summaries(self) -> None:
        summaries, operator_effects, method_effects = summarize_two_pass_outcomes(
            _rows(), bootstrap_iterations=30
        )
        self.assertEqual(len(summaries), 4 * 14)
        self.assertEqual(len(operator_effects), 2 * 14)
        self.assertEqual(len(method_effects), 2 * 14)
        active = next(
            row
            for row in operator_effects
            if row.method == "active_recompute"
            and row.operator == "reground"
            and row.metric == "success"
        )
        self.assertEqual(active.effect, 0.5)
        passive = next(
            row
            for row in operator_effects
            if row.method == "frozen_probe:recompute"
            and row.operator == "reground"
            and row.metric == "success"
        )
        self.assertEqual(passive.effect, -0.5)
        self.assertTrue(all(row.bootstrap_unit == "paired_source_task" for row in method_effects))

    def test_missing_treatment_fails_and_tables_figures_write(self) -> None:
        with self.assertRaisesRegex(AnalysisInputError, "paired"):
            summarize_two_pass_outcomes(_rows()[:-1], bootstrap_iterations=2)
        replicated = _rows()
        replicated[0]["replicate_id"] = 1
        replicated[0]["unit_id"] = replicated[0]["task_id"] + "/r1"
        with self.assertRaisesRegex(AnalysisInputError, "task-unit"):
            summarize_two_pass_outcomes(replicated, bootstrap_iterations=2)
        summaries, operator_effects, method_effects = summarize_two_pass_outcomes(
            _rows(), bootstrap_iterations=4
        )
        analysis = {
            "artifact_type": TWO_PASS_ANALYSIS_TYPE,
            "deployment_mode": "two_pass_frozen",
            "rows": _rows(),
            "metric_summaries": [asdict(row) for row in summaries],
            "operator_effects": [asdict(row) for row in operator_effects],
            "method_effects": [asdict(row) for row in method_effects],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = write_two_pass_tables(analysis, root / "tables")
            figures = write_two_pass_figures(analysis, root / "figures")
            self.assertEqual(len(tables), 4)
            self.assertEqual(len(figures), 10)
            self.assertIn("cell_id,model,benchmark", tables[0].read_text(encoding="utf-8"))


class TwoPassMaterializationTests(unittest.IsolatedAsyncioTestCase):
    async def _build_run(self, root: Path) -> tuple[RunLayout, str]:
        dataset = root / "evolving.json"
        build_receipt = root / "build-receipt.json"
        task_manifest = root / "tasks.jsonl"
        atomic_write_json(
            dataset,
            {
                "tasks": [
                    {
                        "task_id": "paper-case",
                        "condition": "t1",
                        "turns": ["What is twice three?"],
                        "label": "6",
                    },
                    {
                        "task_id": "paper-case",
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
        tasks = EvolvingIntentAdapter(dataset, expected_sha256=sha256_file(dataset)).load_tasks()
        task = next(item for item in tasks if item.condition == "t7")
        freeze_task_manifest(task_manifest, (task,))

        run_id = "two-pass-analysis-test"
        layout = RunLayout.for_run(root / "artifacts", run_id)
        layout.create()
        cells = make_pair_manifest(
            tasks=(TaskRef(task.domain, pair_task_id(task), task.task_sha256),),
            models=(MODEL,),
            arms=METHODS,
            operators=OPERATORS,
            randomization_seed=12012,
        )
        atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])
        pair_sha = sha256_file(layout.pairs)
        traces = []
        for method in METHODS:
            checkpoint = PassOneCheckpoint(
                checkpoint=1,
                score=0.9,
                source_prefix_sha256=sha256_json({"method": method, "kind": "prefix"}),
                signal_record_sha256=sha256_json({"method": method, "kind": "signal"}),
            )
            traces.append(
                PassOneMethodTrace(
                    model=MODEL,
                    benchmark=task.domain,
                    task_id=pair_task_id(task),
                    task_sha256=task.task_sha256,
                    replicate_id=0,
                    method=method,
                    active_variant=ARM_TO_PROBE.get(method),
                    source_trajectory_sha256=sha256_json({"method": method, "source": "pass-one"}),
                    task_horizon=7,
                    checkpoints=(checkpoint,),
                )
            )
        pass_one = PassOneObservationArtifact(
            source_run_id="pass-one-test",
            source_manifest_sha256="a" * 64,
            traces=tuple(sorted(traces, key=lambda row: row.identity)),
        )
        thresholds = ThresholdLockArtifact(
            calibration_run_id="calibration-test",
            calibration_manifest_sha256="b" * 64,
            natural_max_actions_per_task=1,
            matched_actions_per_method=1,
            yoke_anchor_method="trace_rules",
            methods=tuple(
                sorted(
                    (
                        LockedMethodThreshold(
                            model=MODEL,
                            benchmark=task.domain,
                            method=method,
                            threshold=0.5,
                            target_firing_rate=0.2,
                            achieved_firing_rate=0.2,
                            calibration_n_tasks=20,
                            calibration_digest=sha256_json({"method": method}),
                            selection_rule="task_score_rank_hash_ties",
                            tie_break_seed=12012,
                            calibration_target_fire_count=4,
                        )
                        for method in METHODS
                    ),
                    key=lambda row: (row.model, row.benchmark, row.method),
                )
            ),
        )
        pass_path = layout.results / "deployment_pass_one.json"
        threshold_path = layout.results / "deployment_threshold_lock.json"
        schedule_path = layout.results / "deployment_schedule.json"
        pass_sha = freeze_pass_one_observations(pass_path, pass_one)
        threshold_sha = freeze_threshold_lock(threshold_path, thresholds)
        schedule = build_deployment_schedule(
            estimand=DeploymentEstimand.NATURAL_THRESHOLD,
            cells=cells,
            pair_manifest_sha256=pair_sha,
            pass_one=pass_one,
            pass_one_artifact_sha256=pass_sha,
            threshold_lock=thresholds,
            threshold_lock_sha256=threshold_sha,
            feedback_plans={},
        )
        freeze_deployment_schedule(
            schedule_path,
            schedule,
            outcome_artifacts_root=layout.results / "deployment",
        )
        manifest = build_manifest(
            run_id=run_id,
            stage=Stage.CONFIRMATORY,
            repository_root=ROOT,
            pair_manifest_sha256=pair_sha,
            models=(MODEL,),
            arms=METHODS,
            operators=OPERATORS,
            randomization_seed=12012,
            benchmark_receipts=(
                ArtifactReceipt.from_file("task_manifest", task_manifest, workspace=ROOT),
                ArtifactReceipt.from_file(
                    "evolving_rendered_dataset", dataset, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    "evolving_build_receipt", build_receipt, workspace=ROOT
                ),
                ArtifactReceipt.from_file(PASS_ONE_RECEIPT, pass_path, workspace=ROOT),
                ArtifactReceipt.from_file(
                    THRESHOLD_LOCK_RECEIPT, threshold_path, workspace=ROOT
                ),
                ArtifactReceipt.from_file(
                    DEPLOYMENT_SCHEDULE_RECEIPT, schedule_path, workspace=ROOT
                ),
            ),
            extra_config={
                "n_cells": len(cells),
                "replicates": 1,
                "deployment_mode": "two_pass_frozen",
                "deployment_estimand": "natural_threshold",
                "deployment_runtime": deployment_runtime_config(),
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
        opener = _MockOpenAI()
        transport = Transport(
            ledger,
            layout.events / "call_attempts.jsonl",
            environ={"OPENAI_API_KEY": "mock-only"},
            urlopen=opener,
        )
        summary = await execute_deployment_run(
            run_id=run_id,
            task_manifest_path=task_manifest,
            pass_one_path=pass_path,
            threshold_lock_path=threshold_path,
            schedule_path=schedule_path,
            tasks=(task,),
            yes_spend=True,
            artifacts_root=layout.root.parent,
            transport=transport,
            evolving_dataset_path=dataset,
            evolving_build_receipt_path=build_receipt,
        )
        self.assertEqual(summary.completed_cells, len(cells))
        return layout, manifest_sha

    async def test_strict_end_to_end_validation_regrading_and_tamper_rejection(self) -> None:
        artifact_parent = ROOT / "experiments12" / "data_results" / "runs"
        artifact_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_parent) as temporary:
            layout, manifest_sha = await self._build_run(Path(temporary))
            report = validate_two_pass_run(
                layout, expected_manifest_sha256=manifest_sha
            )
            self.assertTrue(report.primary_ready)
            self.assertEqual(report.expected_cells, 4)
            self.assertEqual(report.canonical_regraded_cells, 4)
            analysis = extract_two_pass_run(
                layout,
                expected_manifest_sha256=manifest_sha,
                bootstrap_iterations=5,
            )
            self.assertTrue(analysis["validation"]["primary_ready"])
            self.assertEqual({row["outcome_source"] for row in analysis["rows"]}, {"canonical_regrade"})
            active_rows = [row for row in analysis["rows"] if row["method"].startswith("active_")]
            passive_rows = [row for row in analysis["rows"] if not row["method"].startswith("active_")]
            self.assertTrue(all(row["observer_tokens"] > 0 for row in active_rows))
            self.assertTrue(all(row["observer_tokens"] == 0 for row in passive_rows))
            self.assertTrue(
                all(
                    row["actual_cost_usd"]
                    == row["reported_cost_usd"]
                    + row["estimated_cost_usd"]
                    + row["upper_bound_cost_usd"]
                    for row in analysis["rows"]
                )
            )
            with self.assertRaisesRegex(AnalysisInputError, "external SHA256"):
                validate_two_pass_run(layout, expected_manifest_sha256="0" * 64)

            cell_id = analysis["rows"][0]["cell_id"]
            output_path = layout.results / "deployment" / f"{cell_id}.json"
            event_path = layout.events / f"deployment-{cell_id}.jsonl"
            job_path = layout.results / "deployment_jobs" / f"{cell_id}.json"
            original_output = read_json(output_path)
            original_events = read_jsonl(event_path)
            original_job = read_json(job_path)

            # A coherent rewrite of the claimed final transcript must still
            # fail exact carried-history replay.
            changed_output = read_json(output_path)
            changed_output["messages"] = []
            changed_output["transcript_sha256"] = sha256_json([])
            atomic_write_json(output_path, changed_output)
            changed_events = read_jsonl(event_path)
            changed_events[-1]["transcript_sha256"] = sha256_json([])
            changed_events[-1]["output_sha256"] = sha256_file(output_path)
            atomic_write_jsonl(event_path, changed_events)
            changed_job = read_json(job_path)
            changed_job["output_sha256"] = sha256_file(output_path)
            atomic_write_json(job_path, changed_job)
            with self.assertRaisesRegex(AnalysisInputError, "final carried history"):
                validate_two_pass_run(layout, expected_manifest_sha256=manifest_sha)
            atomic_write_json(output_path, original_output)
            atomic_write_jsonl(event_path, original_events)
            atomic_write_json(job_path, original_job)

            # Even a self-consistent start/output rewrite cannot substitute a
            # different runtime configuration for the manifest-frozen one.
            changed_output = read_json(output_path)
            changed_events = read_jsonl(event_path)
            changed_job = read_json(job_path)
            changed_events[0]["runtime_config"] = deployment_runtime_config(
                HarnessConfig(task_max_output_tokens=1799)
            )
            changed_design = {
                key: value
                for key, value in changed_events[0].items()
                if key not in {"event", "design_sha256"}
            }
            changed_design_sha = sha256_json(changed_design)
            changed_events[0]["design_sha256"] = changed_design_sha
            changed_output["design_sha256"] = changed_design_sha
            atomic_write_json(output_path, changed_output)
            changed_events[-1]["design_sha256"] = changed_design_sha
            changed_events[-1]["output_sha256"] = sha256_file(output_path)
            atomic_write_jsonl(event_path, changed_events)
            changed_job["output_sha256"] = sha256_file(output_path)
            atomic_write_json(job_path, changed_job)
            with self.assertRaisesRegex(AnalysisInputError, "start/design"):
                validate_two_pass_run(layout, expected_manifest_sha256=manifest_sha)
            atomic_write_json(output_path, original_output)
            atomic_write_jsonl(event_path, original_events)
            atomic_write_json(job_path, original_job)

            # Exact event coverage means even an otherwise harmless-looking
            # extra row in the attempt log is forbidden.
            attempt_path = layout.events / "call_attempts.jsonl"
            original_attempts = read_jsonl(attempt_path)
            atomic_write_jsonl(attempt_path, [*original_attempts, {"junk": "undeclared"}])
            with self.assertRaisesRegex(AnalysisInputError, "call-attempt row"):
                validate_two_pass_run(layout, expected_manifest_sha256=manifest_sha)
            atomic_write_jsonl(attempt_path, original_attempts)

            output = read_json(output_path)
            changed_success = not output["evaluation"]["success"]
            output["evaluation"]["success"] = changed_success
            atomic_write_json(output_path, output)
            events = read_jsonl(event_path)
            events[-1]["success"] = changed_success
            events[-1]["output_sha256"] = sha256_file(output_path)
            atomic_write_jsonl(event_path, events)
            job = read_json(job_path)
            job["success"] = changed_success
            job["output_sha256"] = sha256_file(output_path)
            atomic_write_json(job_path, job)
            with self.assertRaisesRegex(AnalysisInputError, "canonical final-outcome regrade"):
                extract_two_pass_run(
                    layout,
                    expected_manifest_sha256=manifest_sha,
                    bootstrap_iterations=2,
                )


if __name__ == "__main__":
    unittest.main()
