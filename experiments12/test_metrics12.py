"""Pure synthetic tests for task-clustered Experiment 12 metrics."""

from __future__ import annotations

import math
from dataclasses import replace
import unittest

from experiments12.metrics12 import (
    CheckpointScore,
    MetricInputError,
    ObservationTrace,
    TaskArmMeasurement,
    TaskOutcome,
    collapse_task_predictions,
    paired_active_effects,
    paired_observer_effects,
    prediction_metrics,
    select_fixed_firing_rate_threshold,
)


def _trace(
    task_id: str,
    scores: tuple[tuple[int, float, bool], ...],
    event: int | None,
    *,
    split: str = "confirmatory",
) -> ObservationTrace:
    return ObservationTrace(
        model="model-a",
        benchmark="bench-a",
        method="passive-judge",
        task_id=task_id,
        split=split,
        checkpoints=tuple(
            CheckpointScore(checkpoint=index, score=score, actionable=actionable)
            for index, score, actionable in scores
        ),
        event_checkpoint=event,
    )


class PairedEffectTests(unittest.TestCase):
    def _outcomes(self):
        rows = []
        clean = [1, 1, 0, 1, 0, 1]
        active = [1, 0, 0, 0, 0, 1]
        for index, (clean_value, active_value) in enumerate(zip(clean, active)):
            task_id = f"task-{index}"
            rows.extend(
                [
                    TaskOutcome("model-a", "bench-a", task_id, "clean", clean_value),
                    TaskOutcome("model-a", "bench-a", task_id, "active", active_value),
                ]
            )
        return rows

    def test_effect_is_paired_and_bootstrap_is_deterministic(self) -> None:
        forward = paired_active_effects(
            self._outcomes(),
            active_arm="active",
            clean_arm="clean",
            bootstrap_iterations=300,
            seed=91,
        )
        reverse = paired_active_effects(
            reversed(self._outcomes()),
            active_arm="active",
            clean_arm="clean",
            bootstrap_iterations=300,
            seed=91,
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(forward[0].n_tasks, 6)
        self.assertAlmostEqual(forward[0].effect, -2 / 6)
        self.assertEqual(forward[0].bootstrap_unit, "task")
        self.assertLessEqual(forward[0].ci_low, forward[0].ci_high)

    def test_pairing_fails_closed(self) -> None:
        rows = self._outcomes()
        with self.assertRaises(MetricInputError):
            paired_active_effects(
                rows[:-1], active_arm="active", clean_arm="clean", bootstrap_iterations=10
            )
        with self.assertRaises(MetricInputError):
            paired_active_effects(
                rows + [rows[0]],
                active_arm="active",
                clean_arm="clean",
                bootstrap_iterations=10,
            )
        incomplete = list(rows)
        incomplete[0] = TaskOutcome(
            "model-a", "bench-a", "task-0", "clean", 1, complete=False
        )
        with self.assertRaises(MetricInputError):
            paired_active_effects(
                incomplete,
                active_arm="active",
                clean_arm="clean",
                bootstrap_iterations=10,
            )


class PairedObserverResourceTests(unittest.TestCase):
    def _measurements(self):
        return (
            TaskArmMeasurement("m", "b", "t1", "clean", 1, 100, 0, 100, 200, 0.01),
            TaskArmMeasurement("m", "b", "t1", "active", 0, 120, 30, 150, 260, 0.016),
            TaskArmMeasurement("m", "b", "t2", "clean", 1, 200, 0, 200, 300, 0.02),
            TaskArmMeasurement("m", "b", "t2", "active", 1, 230, 40, 270, 380, 0.029),
        )

    def test_all_six_metrics_share_one_paired_task_denominator(self) -> None:
        rows = paired_observer_effects(
            self._measurements(),
            active_arm="active",
            bootstrap_iterations=40,
            seed=7,
        )
        by_metric = {row.metric: row for row in rows}
        self.assertEqual(
            set(by_metric),
            {
                "success",
                "task_tokens",
                "observer_tokens",
                "total_tokens",
                "latency_ms",
                "actual_cost_usd",
            },
        )
        self.assertEqual({row.n_tasks for row in rows}, {2})
        self.assertEqual({row.bootstrap_unit for row in rows}, {"task"})
        self.assertEqual(by_metric["success"].effect, -0.5)
        self.assertEqual(by_metric["task_tokens"].effect, 25.0)
        self.assertEqual(by_metric["observer_tokens"].effect, 35.0)
        self.assertEqual(by_metric["total_tokens"].effect, 60.0)
        self.assertAlmostEqual(by_metric["actual_cost_usd"].effect, 0.0075)

    def test_resource_pairing_and_clean_zero_carry_fail_closed(self) -> None:
        rows = self._measurements()
        with self.assertRaises(MetricInputError):
            paired_observer_effects(
                rows[:-1], active_arm="active", bootstrap_iterations=10
            )
        bad_clean = replace(rows[0], observer_tokens=1, total_tokens=101)
        with self.assertRaises(MetricInputError):
            paired_observer_effects(
                (bad_clean, *rows[1:]),
                active_arm="active",
                bootstrap_iterations=10,
            )


class PredictionMetricTests(unittest.TestCase):
    def _confirmatory(self):
        return (
            _trace("positive-high", ((1, 0.2, True), (2, 0.9, True)), 5),
            _trace("negative-high", ((1, 0.8, True), (2, 0.1, True)), None),
            _trace("positive-mid", ((1, 0.6, True),), 4),
            _trace("negative-low", ((1, 0.1, True), (2, 0.05, True)), None),
        )

    def test_pr_auprc_calibration_locked_metrics_and_lead_time(self) -> None:
        summary = prediction_metrics(
            self._confirmatory(), locked_threshold=0.7, calibration_bins=5
        )
        self.assertEqual(summary.n_tasks, 4)
        self.assertEqual(summary.n_positive_tasks, 2)
        self.assertAlmostEqual(summary.auprc, 5 / 6)
        self.assertAlmostEqual(summary.precision or 0.0, 0.5)
        self.assertAlmostEqual(summary.recall, 0.5)
        self.assertAlmostEqual(summary.firing_rate, 0.5)
        self.assertAlmostEqual(summary.brier, (0.01 + 0.64 + 0.16 + 0.01) / 4)
        self.assertEqual(summary.statistical_unit, "task")
        self.assertEqual(summary.lead_time.lead_times, (3,))
        self.assertEqual(sum(item.count for item in summary.calibration_bins), 4)

    def test_checkpoints_are_nested_not_independent_rows(self) -> None:
        predictions = collapse_task_predictions(self._confirmatory())
        self.assertEqual(len(predictions), 4)
        self.assertEqual({item.task_id for item in predictions}, {
            "positive-high", "negative-high", "positive-mid", "negative-low"
        })

        # A post-event or explicitly non-actionable high score cannot fire.
        traces = (
            _trace("event", ((1, 0.2, True), (4, 1.0, True)), 4),
            _trace("control", ((1, 0.9, False), (2, 0.1, True)), None),
        )
        collapsed = collapse_task_predictions(traces)
        self.assertEqual([item.score for item in collapsed], [0.1, 0.2])

    def test_duplicate_task_trace_and_no_positive_fail_closed(self) -> None:
        one = self._confirmatory()[0]
        with self.assertRaises(MetricInputError):
            collapse_task_predictions((one, one))
        negatives = (
            _trace("n1", ((1, 0.2, True),), None),
            _trace("n2", ((1, 0.8, True),), None),
        )
        with self.assertRaises(MetricInputError):
            prediction_metrics(negatives, locked_threshold=0.5)


class ThresholdSelectionTests(unittest.TestCase):
    def _calibration(self):
        return tuple(
            _trace(
                f"cal-{index}",
                ((1, score, True),),
                3 if index % 2 else None,
                split="calibration",
            )
            for index, score in enumerate((0.9, 0.8, 0.6, 0.1), 1)
        )

    def test_fixed_rate_selection_is_calibration_only_and_non_exceeding(self) -> None:
        selection = select_fixed_firing_rate_threshold(
            self._calibration(), target_firing_rate=0.5
        )
        self.assertEqual(selection.threshold, 0.8)
        self.assertEqual(selection.achieved_firing_rate, 0.5)
        self.assertEqual(selection.split, "calibration")
        self.assertEqual(len(selection.calibration_digest), 64)

        summary = prediction_metrics(
            self._confirmatory_for_selection(), locked_threshold=selection
        )
        self.assertEqual(summary.threshold_source, "calibration_locked_fixed_rate")
        self.assertEqual(summary.locked_threshold, 0.8)
        self.assertEqual(summary.selection_rule, "task_score_rank_hash_ties")
        self.assertEqual(summary.firing_rate, 0.5)

    def _confirmatory_for_selection(self):
        return (
            _trace("c1", ((1, 0.95, True),), 3),
            _trace("c2", ((1, 0.70, True),), None),
        )

    def test_selection_rejects_confirmatory_and_handles_zero_target(self) -> None:
        with self.assertRaises(MetricInputError):
            select_fixed_firing_rate_threshold(
                self._confirmatory_for_selection(), target_firing_rate=0.5
            )
        selection = select_fixed_firing_rate_threshold(
            self._calibration(), target_firing_rate=0.0
        )
        self.assertEqual(selection.achieved_firing_rate, 0.0)
        self.assertGreater(selection.threshold, 0.9)
        self.assertTrue(math.isfinite(selection.threshold))

    def test_binary_ties_use_deterministic_answer_blind_fixed_rate_ranking(self) -> None:
        calibration = tuple(
            _trace(
                f"binary-{index}",
                ((1, float(index < 8), True),),
                3 if index % 3 == 0 else None,
                split="calibration",
            )
            for index in range(10)
        )
        selection = select_fixed_firing_rate_threshold(
            calibration,
            target_firing_rate=0.35,
            tie_break_seed=99,
        )
        reversed_selection = select_fixed_firing_rate_threshold(
            reversed(calibration),
            target_firing_rate=0.35,
            tie_break_seed=99,
        )
        self.assertEqual(selection, reversed_selection)
        self.assertEqual(selection.target_fire_count, 3)
        self.assertEqual(selection.achieved_firing_rate, 0.3)
        self.assertLess(0.35 - selection.achieved_firing_rate, 1 / 10)

        confirmatory = tuple(
            _trace(
                f"test-binary-{index}",
                ((1, float(index < 9), True),),
                3 if index % 2 == 0 else None,
            )
            for index in range(10)
        )
        summary = prediction_metrics(confirmatory, locked_threshold=selection)
        reverse = prediction_metrics(
            reversed(confirmatory), locked_threshold=selection
        )
        self.assertEqual(summary, reverse)
        self.assertEqual(summary.firing_rate, 0.3)

    def test_threshold_selection_schema_is_strict(self) -> None:
        selection = select_fixed_firing_rate_threshold(
            self._calibration(), target_firing_rate=0.5
        )
        with self.assertRaises(MetricInputError):
            replace(selection, split="confirmatory")
        with self.assertRaises(MetricInputError):
            replace(selection, calibration_digest="bad")
        with self.assertRaises(MetricInputError):
            replace(selection, target_fire_count=selection.n_tasks + 1)


if __name__ == "__main__":
    unittest.main()
