from __future__ import annotations

import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import unittest

from experiments12.analysis12 import (
    AnalysisInputError,
    _AttemptResource,
    _load_attempt_resources,
    calibrate_thresholds,
    load_threshold_artifact,
    make_threshold_artifact,
    observer_metric_effects,
    observer_effects,
    require_source_task_disjointness,
    score_locked,
    signal_traces,
    task_measurements,
    task_outcomes,
    verify_threshold_binding,
)
from experiments12.core.artifacts import atomic_write_jsonl, sha256_json
from experiments12.core.budget import BudgetLedger
from experiments12.core.schemas import (
    CallAttemptRecord,
    CallStatus,
    PairKey,
    TokenUsage,
    record_to_dict,
)
from experiments12.manifest12 import RunLayout
from experiments12.metrics12 import CheckpointScore, ObservationTrace
from experiments12.pairing12 import JobCell


def cell(cell_id: str, task_id: str, arm: str) -> JobCell:
    return JobCell(
        cell_id=cell_id,
        block_id=f"block-{task_id}",
        block_position=0 if arm == "clean" else 1,
        pair_key=PairKey(
            model="gpt-5.6-luna",
            domain="evolving_intent_gsm8k",
            task_id=task_id,
            task_sha256="a" * 64,
        ),
        arm=arm,
        operator="none",
        seed=12,
    )


def trajectory(*, success: bool, arm: str, probe_failed: bool = False):
    turns = [
        {"event": "task_turn", "task_turn": index}
        for index in range(1, 4)
    ]
    probes = []
    if arm != "clean":
        probes = [
            {
                "event": "active_probe",
                "after_task_turn": 2,
                "grade": {"passed": not probe_failed},
            }
        ]
    return {
        "arm": arm,
        "domain": "evolving_intent_gsm8k",
        "checkpoint_turns": [2],
        "task_records": turns,
        "probe_records": probes,
        "evaluation": {"success": success},
    }


def _call(
    event_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    elapsed_ms: int,
    cost: str,
) -> dict[str, object]:
    return {
        "call_event_ids": [event_id],
        "resolved_model_id": "gpt-5.6-luna",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        },
        "accounted_cost_usd": cost,
        "elapsed_ms": elapsed_ms,
    }


def accounted_trajectory(
    *,
    success: bool,
    arm: str,
    task_usage: tuple[int, int, int, str],
    observer_usage: tuple[int, int, int, str] | None = None,
) -> dict[str, object]:
    task_call = _call(
        f"{arm}-task",
        input_tokens=task_usage[0],
        output_tokens=task_usage[1],
        elapsed_ms=task_usage[2],
        cost=task_usage[3],
    )
    task_record = {"event": "task_turn", "task_turn": 1, "call": task_call}
    probe_records = []
    if observer_usage is not None:
        probe_call = _call(
            f"{arm}-probe",
            input_tokens=observer_usage[0],
            output_tokens=observer_usage[1],
            elapsed_ms=observer_usage[2],
            cost=observer_usage[3],
        )
        probe_records.append({"event": "active_probe", "call": probe_call})

    def bucket(call):
        usage = call["usage"]
        return {
            "calls": 1,
            **usage,
            "elapsed_ms": call["elapsed_ms"],
            "accounted_cost_usd": call["accounted_cost_usd"],
        }

    categories = {"agent": bucket(task_call)}
    if probe_records:
        categories["active_monitor"] = bucket(probe_records[0]["call"])
    return {
        "complete": True,
        "arm": arm,
        "task_records": [task_record],
        "probe_records": probe_records,
        "evaluation": {"success": success},
        "accounting": {
            "by_category": categories,
            "resolved_model_ids": ["gpt-5.6-luna"],
        },
    }


class AnalysisTests(unittest.TestCase):
    def test_attempt_loader_joins_append_only_events_to_reconciled_ledger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="analysis12_attempts_") as raw:
            layout = RunLayout.for_run(raw, "run-calls")
            layout.create()
            ledger = BudgetLedger(layout.ledger)
            reservation = ledger.reserve(
                "openai", "0.01", purpose="agent_turn", request_key="run-calls/call-1"
            )
            usage = TokenUsage(input_tokens=20, output_tokens=5)
            ledger.reconcile(
                reservation.reservation_id,
                "0.001",
                usage=usage,
                request_status=CallStatus.SUCCEEDED,
                cost_quality="estimated",
            )
            attempt = CallAttemptRecord(
                event_id="event-1",
                reservation_id=reservation.reservation_id,
                provider="openai",
                model="gpt-5.6-luna",
                purpose="agent_turn",
                attempt_number=1,
                status=CallStatus.SUCCEEDED,
                started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:00:01Z",
                usage=usage,
                estimated_cost_usd=Decimal("0.001"),
                elapsed_ms=25,
            )
            atomic_write_jsonl(
                layout.events / "calls.jsonl", [record_to_dict(attempt)]
            )
            loaded = _load_attempt_resources(layout)
            self.assertEqual(loaded["event-1"].input_tokens, 20)
            self.assertEqual(loaded["event-1"].actual_cost_usd, Decimal("0.001"))
            self.assertEqual(loaded["event-1"].cost_quality, "estimated")

    def test_observer_effect_pairs_each_active_arm_against_clean(self) -> None:
        cells = (
            cell("c1", "task-1::t7", "clean"),
            cell("a1", "task-1::t7", "active_recompute"),
            cell("c2", "task-2::t7", "clean"),
            cell("a2", "task-2::t7", "active_recompute"),
        )
        trajectories = {
            "c1": trajectory(success=True, arm="clean"),
            "a1": trajectory(success=False, arm="active_recompute"),
            "c2": trajectory(success=True, arm="clean"),
            "a2": trajectory(success=True, arm="active_recompute"),
        }
        outcomes = task_outcomes(cells, trajectories)
        effects = observer_effects(outcomes, bootstrap_iterations=20)
        self.assertEqual(set(effects), {"active_recompute"})
        self.assertEqual(effects["active_recompute"][0].effect, -0.5)
        self.assertEqual(effects["active_recompute"][0].n_tasks, 2)

    def test_signals_keep_active_on_active_and_passive_on_clean(self) -> None:
        with tempfile.TemporaryDirectory(prefix="analysis12_") as raw:
            layout = RunLayout.for_run(raw, "run-a")
            layout.create()
            clean = cell("clean-cell", "task-1::t7", "clean")
            active = cell("active-cell", "task-1::t7", "active_recompute")
            source = trajectory(success=False, arm="clean")
            treated = trajectory(
                success=True, arm="active_recompute", probe_failed=True
            )
            from experiments12.core.artifacts import atomic_write_json

            atomic_write_json(
                layout.shadow / "clean-cell.json",
                {
                    "records": [
                        {
                            "method": "turn_clock",
                            "checkpoint_turn": 2,
                            "actionable_before_turn": 3,
                            "score": 0.5,
                        },
                        {
                            "method": "frozen_probe",
                            "variant": "recompute",
                            "checkpoint_turn": 2,
                            "actionable_before_turn": 3,
                            "score": 0.0,
                        },
                    ],
                    "monitor_methods": ["frozen_probe", "turn_clock"],
                },
            )
            traces = signal_traces(
                layout,
                (clean, active),
                {"clean-cell": source, "active-cell": treated},
                split="exploratory",
            )
            by_method = {trace.method: trace for trace in traces}
            self.assertEqual(
                set(by_method),
                {"active_recompute", "turn_clock", "frozen_probe:recompute"},
            )
            self.assertIsNone(by_method["active_recompute"].event_checkpoint)
            self.assertEqual(by_method["active_recompute"].checkpoints[0].score, 1.0)
            self.assertEqual(by_method["turn_clock"].event_checkpoint, 3)
            with self.assertRaises(AnalysisInputError):
                signal_traces(
                    layout,
                    (clean, active),
                    {"clean-cell": source, "active-cell": treated},
                    split="exploratory",
                    required_passive_methods=(
                        "turn_clock",
                        "frozen_probe:recompute",
                        "trace_judge",
                    ),
                )

    def test_task_resource_extraction_and_effect_table_are_paired(self) -> None:
        cells = (
            cell("c1", "task-1::t7", "clean"),
            cell("a1", "task-1::t7", "active_recompute"),
            cell("c2", "task-2::t7", "clean"),
            cell("a2", "task-2::t7", "active_recompute"),
        )
        trajectories = {
            "c1": accounted_trajectory(
                success=True, arm="clean", task_usage=(90, 10, 100, "0.010")
            ),
            "a1": accounted_trajectory(
                success=False,
                arm="active_recompute",
                task_usage=(105, 15, 120, "0.012"),
                observer_usage=(25, 5, 30, "0.004"),
            ),
            "c2": accounted_trajectory(
                success=True, arm="clean", task_usage=(180, 20, 200, "0.020")
            ),
            "a2": accounted_trajectory(
                success=True,
                arm="active_recompute",
                task_usage=(210, 20, 240, "0.024"),
                observer_usage=(35, 5, 40, "0.005"),
            ),
        }
        measurements = task_measurements(cells, trajectories)
        effects = observer_metric_effects(measurements, bootstrap_iterations=30)
        by_metric = {row.metric: row for row in effects["active_recompute"]}
        self.assertEqual(by_metric["success"].effect, -0.5)
        self.assertEqual(by_metric["task_tokens"].effect, 25.0)
        self.assertEqual(by_metric["observer_tokens"].effect, 35.0)
        self.assertEqual(by_metric["total_tokens"].effect, 60.0)
        self.assertEqual(by_metric["latency_ms"].effect, 65.0)
        self.assertAlmostEqual(by_metric["actual_cost_usd"].effect, 0.0075)

        trajectories["a1"]["accounting"]["by_category"]["agent"]["input_tokens"] += 1
        with self.assertRaises(AnalysisInputError):
            task_measurements(cells, trajectories)

    def test_attempt_level_resources_include_failed_retries(self) -> None:
        clean = cell("clean", "task-1::t7", "clean")
        active = cell("active", "task-1::t7", "active_recompute")
        clean_trajectory = accounted_trajectory(
            success=True, arm="clean", task_usage=(90, 10, 100, "0.010")
        )
        active_trajectory = accounted_trajectory(
            success=True,
            arm="active_recompute",
            task_usage=(105, 15, 120, "0.012"),
            observer_usage=(25, 5, 30, "0.004"),
        )
        active_task_call = active_trajectory["task_records"][0]["call"]
        active_task_call["call_event_ids"] = ["active-retry", "active-task"]
        active_task_call["elapsed_ms"] = 170
        active_trajectory["accounting"]["by_category"]["agent"]["elapsed_ms"] = 170
        attempts = {
            "clean-task": _AttemptResource(
                "clean-task", "agent_turn", CallStatus.SUCCEEDED,
                90, 10, 100, Decimal("0.010"), "estimated",
            ),
            "active-retry": _AttemptResource(
                "active-retry", "agent_turn", CallStatus.FAILED,
                50, 10, 50, Decimal("0.006"), "upper_bound",
            ),
            "active-task": _AttemptResource(
                "active-task", "agent_turn", CallStatus.SUCCEEDED,
                105, 15, 120, Decimal("0.012"), "estimated",
            ),
            "active_recompute-probe": _AttemptResource(
                "active_recompute-probe", "active_probe", CallStatus.SUCCEEDED,
                25, 5, 30, Decimal("0.004"), "estimated",
            ),
        }
        rows = task_measurements(
            (clean, active),
            {"clean": clean_trajectory, "active": active_trajectory},
            attempt_resources=attempts,
        )
        by_arm = {row.arm: row for row in rows}
        self.assertEqual(by_arm["active_recompute"].task_tokens, 180)
        self.assertEqual(by_arm["active_recompute"].latency_ms, 200)
        self.assertAlmostEqual(by_arm["active_recompute"].actual_cost_usd, 0.022)

    def test_attempt_resources_flatten_bfcl_multi_call_turns(self) -> None:
        clean = cell("clean", "task-1::native", "clean")
        source = accounted_trajectory(
            success=True, arm="clean", task_usage=(30, 3, 30, "0.003")
        )
        first = _call(
            "bfcl-first", input_tokens=10, output_tokens=1, elapsed_ms=10, cost="0.001"
        )
        second = _call(
            "bfcl-second", input_tokens=20, output_tokens=2, elapsed_ms=20, cost="0.002"
        )
        source["task_records"][0]["call"]["call_event_ids"] = [
            "bfcl-first",
            "bfcl-second",
        ]
        source["task_records"][0]["calls"] = [first, second]
        attempts = {
            "bfcl-first": _AttemptResource(
                "bfcl-first", "agent_turn", CallStatus.SUCCEEDED,
                10, 1, 10, Decimal("0.001"), "estimated",
            ),
            "bfcl-second": _AttemptResource(
                "bfcl-second", "agent_turn", CallStatus.SUCCEEDED,
                20, 2, 20, Decimal("0.002"), "estimated",
            ),
        }
        rows = task_measurements(
            (clean,), {"clean": source}, attempt_resources=attempts
        )
        self.assertEqual(rows[0].task_tokens, 33)
        self.assertEqual(rows[0].latency_ms, 30)
        self.assertEqual(rows[0].actual_cost_usd, 0.003)

        source["task_records"][0]["calls"][1]["call_event_ids"] = ["wrong"]
        with self.assertRaisesRegex(AnalysisInputError, "nested call IDs"):
            task_measurements((clean,), {"clean": source}, attempt_resources=attempts)

    def test_bfcl_uses_earliest_official_turn_failure_but_evolving_is_final_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="analysis12_bfcl_") as raw:
            layout = RunLayout.for_run(raw, "run-b")
            layout.create()
            clean = cell("clean-cell", "task-1::native", "clean")
            active = cell("active-cell", "task-1::native", "active_recompute")
            clean = JobCell(
                clean.cell_id,
                clean.block_id,
                clean.block_position,
                PairKey(
                    clean.pair_key.model,
                    "bfcl_multi_turn",
                    clean.pair_key.task_id,
                    task_sha256="a" * 64,
                ),
                clean.arm,
                clean.operator,
                clean.seed,
            )
            active = JobCell(
                active.cell_id,
                active.block_id,
                active.block_position,
                PairKey(
                    active.pair_key.model,
                    "bfcl_multi_turn",
                    active.pair_key.task_id,
                    task_sha256="a" * 64,
                ),
                active.arm,
                active.operator,
                active.seed,
            )
            failure_keys = {
                "invalid_call_observed": False,
                "execution_failure_observed": False,
                "state_check_failure_observed": False,
                "state_check_available": True,
            }
            source = trajectory(success=True, arm="clean")
            source["domain"] = "bfcl_multi_turn"
            for index, record in enumerate(source["task_records"], 1):
                record["failure_indicators"] = dict(failure_keys)
                if index == 2:
                    record["failure_indicators"]["execution_failure_observed"] = True
            treated = trajectory(success=True, arm="active_recompute", probe_failed=True)
            treated["domain"] = "bfcl_multi_turn"
            for index, record in enumerate(treated["task_records"], 1):
                record["failure_indicators"] = dict(failure_keys)
                if index == 2:
                    record["failure_indicators"]["execution_failure_observed"] = True
            from experiments12.core.artifacts import atomic_write_json

            atomic_write_json(
                layout.shadow / "clean-cell.json",
                {
                    "records": [
                        {
                            "method": "turn_clock",
                            "checkpoint_turn": 2,
                            "actionable_before_turn": 3,
                            "score": 0.5,
                        }
                    ],
                    "monitor_methods": ["turn_clock"],
                },
            )
            traces = signal_traces(
                layout,
                (clean, active),
                {"clean-cell": source, "active-cell": treated},
                split="exploratory",
            )
            self.assertEqual({row.event_checkpoint for row in traces}, {2})

    def test_calibration_locks_then_confirmatory_scores(self) -> None:
        calibration = (
            ObservationTrace(
                "m", "b", "method", "cal-1", "calibration",
                (CheckpointScore(1, 0.9),), 2,
            ),
            ObservationTrace(
                "m", "b", "method", "cal-2", "calibration",
                (CheckpointScore(1, 0.2),), None,
            ),
        )
        thresholds = calibrate_thresholds(calibration, target_firing_rate=0.5)
        confirmatory = (
            ObservationTrace(
                "m", "b", "method", "test-1", "confirmatory",
                (CheckpointScore(1, 0.95),), 2,
            ),
            ObservationTrace(
                "m", "b", "method", "test-2", "confirmatory",
                (CheckpointScore(1, 0.1),), None,
            ),
        )
        scored = score_locked(confirmatory, thresholds)
        self.assertEqual(scored[0].precision, 1.0)
        self.assertEqual(scored[0].recall, 1.0)
        with self.assertRaises(AnalysisInputError):
            score_locked(calibration, thresholds)

    def test_threshold_artifact_binds_methods_hashes_and_global_source_ids(self) -> None:
        calibration = (
            ObservationTrace(
                "m", "b", "method", "source-1::t7/r0", "calibration",
                (CheckpointScore(1, 0.9),), 2, source_task_id="source-1",
            ),
            ObservationTrace(
                "m", "b", "method", "source-2::t7/r0", "calibration",
                (CheckpointScore(1, 0.1),), None, source_task_id="source-2",
            ),
        )
        thresholds = calibrate_thresholds(calibration, target_firing_rate=0.5)
        source = {
            "run_id": "cal-run",
            "stage": "calibration",
            "split": "calibration",
            "manifest_sha256": "a" * 64,
            "required_passive_methods": ["method"],
        }
        artifact = make_threshold_artifact(
            source,
            calibration,
            thresholds,
            target_firing_rate=0.5,
            source_extract_sha256="b" * 64,
        )
        parsed, slices, source_tasks, passive = load_threshold_artifact(artifact)
        self.assertEqual(parsed, thresholds)
        self.assertEqual(slices, (("m", "b", "method"),))
        self.assertEqual(passive, ("method",))

        artifact_sha = sha256_json(artifact)
        confirm_extract = {
            "analysis_lock": {
                "threshold_artifact_sha256": artifact_sha,
                "calibration_manifest_sha256": "a" * 64,
            },
            "required_passive_methods": ["method"],
        }
        verify_threshold_binding(
            confirm_extract,
            artifact,
            threshold_artifact_sha256=artifact_sha,
        )
        with self.assertRaises(AnalysisInputError):
            verify_threshold_binding(
                confirm_extract,
                artifact,
                threshold_artifact_sha256="c" * 64,
            )

        reused_different_condition_and_model = (
            ObservationTrace(
                "other-model", "b", "method", "source-1::t1/r9", "confirmatory",
                (CheckpointScore(1, 0.8),), 2, source_task_id="source-1",
            ),
        )
        with self.assertRaises(AnalysisInputError):
            require_source_task_disjointness(
                reused_different_condition_and_model, source_tasks
            )
        aliased_evolving_source = (
            ObservationTrace(
                "m",
                "evolving_intent_gsm8k",
                "method",
                "extracted-gsm8k-test-770::t7/r0",
                "confirmatory",
                (CheckpointScore(1, 0.8),),
                2,
                source_task_id="extracted-gsm8k-test-770",
            ),
        )
        with self.assertRaisesRegex(AnalysisInputError, "overlap globally"):
            require_source_task_disjointness(
                aliased_evolving_source,
                (("evolving_intent_gsm8k", "770"),),
            )

        extra = ObservationTrace(
            "m", "b", "extra", "source-3::t7/r0", "confirmatory",
            (CheckpointScore(1, 0.7),), 2, source_task_id="source-3",
        )
        valid = ObservationTrace(
            "m", "b", "method", "source-3::t7/r0", "confirmatory",
            (CheckpointScore(1, 0.7),), 2, source_task_id="source-3",
        )
        with self.assertRaises(AnalysisInputError):
            score_locked((valid, extra), thresholds, required_method_slices=slices)
        with self.assertRaises(AnalysisInputError):
            score_locked(
                (valid,),
                (*thresholds, replace(thresholds[0], method="extra")),
            )


if __name__ == "__main__":
    unittest.main()
