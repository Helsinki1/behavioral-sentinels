"""Pure synthetic SVG/sidecar tests; no plotting dependency or old results."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from experiments12.figures12 import (
    AdvantageCell,
    DeploymentBar,
    FigureInputError,
    write_deployment_grouped_bars,
    write_method_advantage_heatmap,
    write_observer_effect_forest,
    write_observer_metric_effect_forest,
    write_pr_curves,
)
from experiments12.metrics12 import (
    CheckpointScore,
    ObservationTrace,
    PairedEffect,
    PairedMetricEffect,
    prediction_metrics,
)


def _parse(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    ET.fromstring(text)
    return text


class FigureTests(unittest.TestCase):
    def test_observer_effect_forest_is_zero_centered_with_sidecar(self) -> None:
        effects = (
            PairedEffect("model-a", "TurnBench", "active", "clean", 50, 0.80, 0.74, -0.06, -0.10, -0.02, 0.95, 500, 7),
            PairedEffect("model-b", "TurnBench", "active", "clean", 50, 0.76, 0.78, 0.02, -0.01, 0.05, 0.95, 500, 7),
            PairedEffect("model-a", "BFCL", "active", "clean", 40, 0.70, 0.62, -0.08, -0.13, -0.03, 0.95, 500, 7),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forest.svg"
            artifact = write_observer_effect_forest(effects, path)
            svg = _parse(artifact.svg_path)
            data = json.loads(artifact.data_path.read_text(encoding="utf-8"))
            self.assertIn('class="zero-line"', svg)
            self.assertIn("negative means degradation", svg)
            self.assertEqual(data["figure_type"], "observer_effect_forest")
            self.assertEqual(data["statistical_unit"], "task")
            self.assertEqual(data["axis"]["minimum"], -data["axis"]["maximum"])
            self.assertEqual(len(data["rows"]), 3)

    def _summary(self, method: str, offset: float = 0.0):
        traces = (
            ObservationTrace(
                "model-a", "TurnBench", method, "p1", "confirmatory",
                (CheckpointScore(1, 0.9 - offset),), 4,
            ),
            ObservationTrace(
                "model-a", "TurnBench", method, "p2", "confirmatory",
                (CheckpointScore(1, 0.6 - offset),), 3,
            ),
            ObservationTrace(
                "model-a", "TurnBench", method, "n1", "confirmatory",
                (CheckpointScore(1, 0.7 - offset),), None,
            ),
            ObservationTrace(
                "model-a", "TurnBench", method, "n2", "confirmatory",
                (CheckpointScore(1, 0.1),), None,
            ),
        )
        return prediction_metrics(traces, locked_threshold=0.65)

    def test_pr_curves_have_fixed_honest_axes_and_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pr.svg"
            artifact = write_pr_curves(
                (self._summary("quiz"), self._summary("judge", 0.05)), path
            )
            svg = _parse(artifact.svg_path)
            data = json.loads(artifact.data_path.read_text(encoding="utf-8"))
            self.assertIn('data-axis-min="0" data-axis-max="1"', svg)
            self.assertIn("locked operating points are diamonds", svg)
            self.assertIn("#0072B2", svg)
            self.assertEqual(data["axes"]["recall"], {"minimum": 0.0, "maximum": 1.0})
            self.assertEqual(len(data["series"]), 2)

    def test_resource_effect_forest_keeps_metric_units_in_sidecar(self) -> None:
        effects = (
            PairedMetricEffect(
                "model-a", "BFCL", "active", "clean", "total_tokens", "tokens",
                "lower", 20, 1000, 1250, 250, 100, 400, 0.95, 200, 12,
            ),
            PairedMetricEffect(
                "model-b", "BFCL", "active", "clean", "total_tokens", "tokens",
                "lower", 20, 900, 1100, 200, 50, 350, 0.95, 200, 12,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource.svg"
            artifact = write_observer_metric_effect_forest(effects, path)
            svg = _parse(artifact.svg_path)
            data = json.loads(artifact.data_path.read_text(encoding="utf-8"))
            self.assertIn("positive is extra burden", svg)
            self.assertEqual(data["figure_type"], "observer_metric_effect_forest")
            self.assertEqual(data["metric"], "total_tokens")
            self.assertEqual(data["unit"], "tokens")

    def test_grouped_bars_include_zero_and_operator_data(self) -> None:
        bars = (
            DeploymentBar("baseline", "none", "clock", 0.71, 80, 0.66, 0.76),
            DeploymentBar("active", "reground", "recompute", 0.65, 80, 0.60, 0.70),
            DeploymentBar("passive-observational", "reground", "judge", 0.77, 80, 0.72, 0.82),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.svg"
            artifact = write_deployment_grouped_bars(bars, path)
            svg = _parse(artifact.svg_path)
            data = json.loads(artifact.data_path.read_text(encoding="utf-8"))
            self.assertIn('class="zero-line"', svg)
            self.assertTrue(data["axis"]["zero_included"])
            self.assertLessEqual(data["axis"]["minimum"], 0)
            self.assertGreaterEqual(data["axis"]["maximum"], 0)
            self.assertEqual({row["operator"] for row in data["rows"]}, {"none", "reground"})

    def test_heatmap_is_symmetric_and_records_three_conditions(self) -> None:
        cells = tuple(
            AdvantageCell(
                method,
                trace,
                context,
                difficulty,
                advantage,
                30,
            )
            for method, advantage in (("quiz-minus-clock", -0.04), ("judge-minus-clock", 0.06))
            for trace in ("short", "long")
            for context in ("full", "lossy")
            for difficulty in ("easy", "hard")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.svg"
            artifact = write_method_advantage_heatmap(cells, path)
            svg = _parse(artifact.svg_path)
            data = json.loads(artifact.data_path.read_text(encoding="utf-8"))
            self.assertIn('data-zero-centered="true"', svg)
            self.assertIn("#0072B2", svg)
            self.assertIn("#D55E00", svg)
            self.assertEqual(
                data["color_axis"]["minimum"], -data["color_axis"]["maximum"]
            )
            self.assertEqual(data["color_axis"]["center"], 0.0)
            self.assertEqual(len(data["dimensions"]["trace_lengths"]), 2)
            self.assertEqual(len(data["dimensions"]["context_difficulty_rows"]), 4)

    def test_empty_and_duplicate_inputs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.svg"
            with self.assertRaises(FigureInputError):
                write_observer_effect_forest((), path)
            duplicate = AdvantageCell("m", "short", "full", "easy", 0.1, 2)
            with self.assertRaises(FigureInputError):
                write_method_advantage_heatmap((duplicate, duplicate), path)


if __name__ == "__main__":
    unittest.main()
