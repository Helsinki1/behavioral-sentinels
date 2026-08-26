#!/usr/bin/env python3
"""Provider-free paper layer for the final Experiment 12 deployments.

The full command must be run only after adaptive-analysis.json and
two-pass-analysis.json have been produced by their fail-closed source
analyzers.  This script makes no provider calls.  It verifies immutable run
provenance, reconstructs every statistic from row-level source-task units,
and emits deterministic JSON, CSV, SVG, and SVG-data sidecars.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import io
import json
import math
import os
from pathlib import Path
import tempfile
from statistics import fmean
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments12.manifest12 import code_tree_hash  # noqa: E402


VERSION = 1
ARTIFACT_TYPE = "experiment12_deployment_paper_post_analysis"
SEED = 12_012
CONFIDENCE = 0.95
DEFAULT_ITERATIONS = 2_000
EXPECTED_CODE_HASH = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)

ONLINE_RUN = "e12-deploy-online-evolving-luna-40-v1"
YOKED_RUN = "e12-deploy-twopass-yoked-evolving-luna-40-v1"
ONLINE_MANIFEST_SHA256 = (
    "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
)
ONLINE_PAIRS_SHA256 = (
    "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
)
YOKED_MANIFEST_SHA256 = (
    "8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250"
)
YOKED_PAIRS_SHA256 = (
    "e45838cb64c522100fb2f0c3f212a00736ab5e1dfb9c501d22f8710c4b6a006e"
)
YOKED_SCHEDULE_SHA256 = (
    "fa6ebd579a58369d13343c22870d3772fa8c4f4ddc1b07e2e3120f23a92f635f"
)

ARTIFACTS = ROOT / "experiments12" / "data_results" / "runs"
DEFAULT_ONLINE = ARTIFACTS / ONLINE_RUN / "results" / "adaptive-analysis.json"
DEFAULT_YOKED = ARTIFACTS / YOKED_RUN / "results" / "two-pass-analysis.json"
DEFAULT_OUTPUT = ROOT / "experiments12" / "data_results" / "derived" / "deployment-paper-post-analysis-v1"

ONLINE_METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
    "context_use",
)
YOKED_METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "turn_clock",
    "context_use",
)
ONLINE_OPERATORS = (
    "none",
    "lossy_compaction",
    "public_state_reground",
    "good_bad_watch_feedback",
)
YOKED_OPERATORS = ("none", "lossy_compaction", "public_state_reground")
PLOT_METHODS = (
    "turn_clock",
    "context_use",
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
)
METHOD_LABELS = {
    "active_recompute": "Active recompute",
    "frozen_probe:recompute": "Frozen recompute",
    "frozen_quiz": "Frozen quiz",
    "trace_judge": "Trace judge",
    "trace_rules": "Trace rules",
    "turn_clock": "Turn clock",
    "context_use": "Context use",
}
OPERATOR_LABELS = {
    "none": "No state action",
    "lossy_compaction": "Lossy compaction",
    "public_state_reground": "Public-state reground",
    "good_bad_watch_feedback": "GOOD/BAD/WATCH",
}
CLASS_COLORS = {"baseline": "#64748b", "active": "#dc2626", "passive": "#2563eb"}
OPERATOR_COLORS = {
    "none": "#64748b",
    "lossy_compaction": "#d97706",
    "public_state_reground": "#0f766e",
    "good_bad_watch_feedback": "#7c3aed",
}

ONLINE_ROW_FIELDS = {
    "cell_id", "model", "benchmark", "task_id", "replicate_id", "unit_id",
    "method", "observation_class", "operator", "deployment_mode", "success",
    "observations", "threshold_firings", "selected_actions", "applied_interventions",
    "task_tokens", "observer_tokens", "total_tokens", "latency_ms", "actual_cost_usd",
}
YOKED_ROW_FIELDS = {
    "cell_id", "model", "benchmark", "task_id", "replicate_id", "unit_id",
    "observation_class", "method", "operator", "deployment_mode", "estimand",
    "success", "outcome_source", "observations", "scheduled_actions", "action_rate",
    "acted_on_task", "applied_interventions", "task_tokens", "observer_tokens",
    "total_tokens", "latency_ms", "actual_cost_usd", "reported_cost_usd",
    "estimated_cost_usd", "upper_bound_cost_usd", "failed_retry_attempts",
}


class PostAnalysisError(RuntimeError):
    """A source artifact is partial, mismatched, or scientifically ambiguous."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostAnalysisError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSONL: {path}")
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), f"JSONL row is not an object: {path}:{line_number}")
        result.append(value)
    return result


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path.name}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"CSV columns differ: {path.name}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def paper_class(method: str) -> str:
    if method in {"turn_clock", "context_use"}:
        return "baseline"
    if method == "active_recompute":
        return "active"
    if method in {"frozen_probe:recompute", "frozen_quiz", "trace_judge", "trace_rules"}:
        return "passive"
    raise PostAnalysisError(f"unknown paper method: {method}")


def number(value: Any, *, context: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"non-numeric {context}")
    result = float(value)
    require(math.isfinite(result) and result >= 0, f"invalid numeric {context}")
    return result


def quantile(values: Sequence[float], probability: float) -> float:
    require(bool(values), "cannot take quantile of no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class BootstrapPlan:
    """One deterministic source-task resampling plan shared by every estimate."""

    def __init__(self, n_units: int, iterations: int, seed: int = SEED) -> None:
        require(n_units > 0, "bootstrap requires source-task units")
        require(iterations > 0, "bootstrap iterations must be positive")
        require(seed == SEED, f"bootstrap seed is frozen to {SEED}")
        self.n_units = n_units
        self.iterations = iterations
        self.seed = seed
        samples: list[tuple[int, ...]] = []
        prefix = f"exp12/deployment-paper-bootstrap/v1\0{seed}\0{n_units}".encode()
        for iteration in range(iterations):
            row = []
            for draw in range(n_units):
                material = prefix + f"\0{iteration}\0{draw}".encode()
                row.append(int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % n_units)
            samples.append(tuple(row))
        self.samples = tuple(samples)

    def interval(self, values: Sequence[float], confidence: float = CONFIDENCE) -> tuple[float, float]:
        require(len(values) == self.n_units, "bootstrap vector has wrong unit count")
        means = [fmean(values[index] for index in sample) for sample in self.samples]
        tail = (1 - confidence) / 2
        return quantile(means, tail), quantile(means, 1 - tail)


def metric_value(row: Mapping[str, Any], metric: str, *, source: str) -> float:
    if metric == "success":
        require(isinstance(row.get("success"), bool), f"{source} success is not boolean")
        return float(row["success"])
    if metric == "firing_incidence":
        require(source == "online", "firing incidence exists only for online deployment")
        return float(number(row["threshold_firings"], context="threshold_firings") > 0)
    if metric == "action_incidence":
        field = "selected_actions" if source == "online" else "acted_on_task"
        return float(number(row[field], context=field) > 0)
    if metric in {"firing_rate", "action_rate"} and source == "online":
        numerator = "threshold_firings" if metric == "firing_rate" else "selected_actions"
        observations = number(row["observations"], context="observations")
        require(observations > 0, "online task has no observations")
        return number(row[numerator], context=numerator) / observations
    return number(row[metric], context=f"{source}/{metric}")


def method_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    operators: Sequence[str],
    row_fields: set[str],
    mode: str,
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    expected_methods, expected_operators = set(methods), set(operators)
    seen_cells: set[str] = set()
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        require(set(row) == row_fields, f"{mode} row {index} schema changed")
        require(row.get("method") in expected_methods, f"{mode} row has undeclared method")
        require(row.get("operator") in expected_operators, f"{mode} row has undeclared operator")
        require(row.get("model") == "gpt-5.6-luna", f"{mode} row model changed")
        require(row.get("benchmark") == "evolving_intent_gsm8k", f"{mode} benchmark changed")
        require(row.get("replicate_id") == 0, f"{mode} replicate changed")
        require(isinstance(row.get("unit_id"), str) and row["unit_id"], f"{mode} unit ID invalid")
        require(isinstance(row.get("cell_id"), str) and row["cell_id"], f"{mode} cell invalid")
        require(row["cell_id"] not in seen_cells, f"{mode} duplicate cell")
        seen_cells.add(str(row["cell_id"]))
        key = (str(row["method"]), str(row["operator"]))
        units = grouped.setdefault(key, {})
        require(row["unit_id"] not in units, f"{mode} duplicate treatment unit")
        units[str(row["unit_id"])] = row
    expected_product = {(method, operator) for method in methods for operator in operators}
    require(set(grouped) == expected_product, f"{mode} is not the exact method/operator product")
    reference_units = set(next(iter(grouped.values())))
    require(len(reference_units) == 40, f"{mode} requires exactly 40 source tasks")
    require(all(set(units) == reference_units for units in grouped.values()), f"{mode} denominators are not paired")
    return grouped


def verify_source_success_summaries(
    payload: Mapping[str, Any], grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]]
) -> None:
    rows = payload.get("metric_summaries")
    require(isinstance(rows, list), "source analysis lacks metric_summaries")
    source = {
        (str(row["method"]), str(row["operator"])): row
        for row in rows
        if isinstance(row, Mapping) and row.get("metric") == "success"
    }
    require(set(source) == set(grouped), "source success summaries do not cover treatments")
    for key, units in grouped.items():
        expected = fmean(float(row["success"]) for row in units.values())
        observed = float(source[key]["mean"])
        require(int(source[key]["n_tasks"]) == len(units), "source summary n differs")
        require(math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12), "source success mean differs from rows")


def verify_pair_cells(run_id: str, rows: Sequence[Mapping[str, Any]], expected_sha: str) -> None:
    pairs = ARTIFACTS / run_id / "pairs.jsonl"
    require(sha256_file(pairs) == expected_sha, f"{run_id} pair manifest changed")
    declared = {str(row["cell_id"]) for row in read_jsonl(pairs)}
    observed = {str(row["cell_id"]) for row in rows}
    require(declared == observed, f"{run_id} analysis rows do not exactly cover declared cells")


def verify_run_file(run_id: str, name: str, expected_sha: str) -> Path:
    path = ARTIFACTS / run_id / name
    require(path.is_file() and not path.is_symlink(), f"{run_id}/{name} is missing or linked")
    require(sha256_file(path) == expected_sha, f"{run_id}/{name} changed")
    return path


def verify_online(path: Path) -> tuple[Mapping[str, Any], dict[tuple[str, str], dict[str, Mapping[str, Any]]]]:
    payload = read_json(path)
    require(isinstance(payload, Mapping), "online analysis is not an object")
    require(payload.get("artifact_type") == "online_adaptive_deployment_analysis", "online artifact type changed")
    require(payload.get("source_run_id") == ONLINE_RUN, "online source run changed")
    require(payload.get("source_manifest_sha256") == ONLINE_MANIFEST_SHA256, "online manifest provenance changed")
    require(payload.get("source_pair_manifest_sha256") == ONLINE_PAIRS_SHA256, "online pair provenance changed")
    require(payload.get("deployment_mode") == "online_adaptive", "online deployment mode changed")
    require(payload.get("deployment_policy") == "natural_threshold_per_task_cap", "online policy changed")
    require(payload.get("per_task_action_cap") == 1, "online action cap changed")
    require(payload.get("statistical_unit") == "source_task", "online statistical unit changed")
    verify_run_file(ONLINE_RUN, "manifest.json", ONLINE_MANIFEST_SHA256)
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 1120, "online analysis is partial")
    grouped = method_index(
        rows,
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        row_fields=ONLINE_ROW_FIELDS,
        mode="online",
    )
    verify_pair_cells(ONLINE_RUN, rows, ONLINE_PAIRS_SHA256)
    verify_source_success_summaries(payload, grouped)
    return payload, grouped


def verify_yoked(path: Path) -> tuple[Mapping[str, Any], dict[tuple[str, str], dict[str, Mapping[str, Any]]]]:
    payload = read_json(path)
    require(isinstance(payload, Mapping), "yoked analysis is not an object")
    require(payload.get("artifact_type") == "two_pass_deployment_analysis", "yoked artifact type changed")
    require(payload.get("source_run_id") == YOKED_RUN, "yoked source run changed")
    require(payload.get("source_manifest_sha256") == YOKED_MANIFEST_SHA256, "yoked manifest provenance changed")
    require(payload.get("source_pair_manifest_sha256") == YOKED_PAIRS_SHA256, "yoked pair provenance changed")
    require(payload.get("source_schedule_sha256") == YOKED_SCHEDULE_SHA256, "yoked schedule provenance changed")
    require(payload.get("deployment_mode") == "two_pass_frozen", "yoked deployment mode changed")
    require(payload.get("estimand") == "yoked_anchor", "yoked estimand changed")
    require(payload.get("statistical_unit") == "source_task", "yoked statistical unit changed")
    validation = payload.get("validation")
    require(
        isinstance(validation, Mapping)
        and validation.get("primary_ready") is True
        and validation.get("expected_cells") == 480
        and validation.get("valid_outputs") == 480,
        "yoked source validation is not complete",
    )
    verify_run_file(YOKED_RUN, "manifest.json", YOKED_MANIFEST_SHA256)
    schedule_path = verify_run_file(YOKED_RUN, "results/deployment_schedule.json", YOKED_SCHEDULE_SHA256)
    schedule = read_json(schedule_path)
    groups = schedule.get("groups")
    require(isinstance(groups, list) and len(groups) == 160, "yoked schedule group count changed")
    for group in groups:
        actions = group.get("actions") if isinstance(group, Mapping) else None
        require(
            isinstance(actions, list)
            and len(actions) == 1
            and actions[0].get("checkpoint") == 1
            and actions[0].get("trigger_method") == "active_recompute",
            "yoked schedule is not the frozen checkpoint-1 active-anchor sensitivity",
        )
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 480, "yoked analysis is partial")
    grouped = method_index(
        rows,
        methods=YOKED_METHODS,
        operators=YOKED_OPERATORS,
        row_fields=YOKED_ROW_FIELDS,
        mode="yoked",
    )
    verify_pair_cells(YOKED_RUN, rows, YOKED_PAIRS_SHA256)
    verify_source_success_summaries(payload, grouped)
    return payload, grouped


def unit_order(grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]]) -> tuple[str, ...]:
    return tuple(sorted(next(iter(grouped.values()))))


def unit_name(metric: str) -> str:
    return {
        "success": "proportion",
        "firing_incidence": "proportion",
        "action_incidence": "proportion",
        "firing_rate": "proportion_of_observed_checkpoints",
        "action_rate": "proportion_of_observed_checkpoints",
        "threshold_firings": "count",
        "selected_actions": "count",
        "acted_on_task": "proportion",
        "observer_tokens": "tokens",
        "task_tokens": "tokens",
        "total_tokens": "tokens",
        "latency_ms": "milliseconds",
        "actual_cost_usd": "USD",
    }[metric]


def summarize(
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    *,
    methods: Sequence[str],
    operators: Sequence[str],
    metrics: Sequence[str],
    source: str,
    plan: BootstrapPlan,
    role: str,
) -> list[dict[str, Any]]:
    units = unit_order(grouped)
    result: list[dict[str, Any]] = []
    for method in methods:
        for operator in operators:
            rows = grouped[(method, operator)]
            for metric in metrics:
                values = [metric_value(rows[unit], metric, source=source) for unit in units]
                low, high = plan.interval(values)
                result.append(
                    {
                        "analysis_role": role,
                        "paper_class": paper_class(method),
                        "method": method,
                        "operator": operator,
                        "metric": metric,
                        "unit": unit_name(metric),
                        "n_tasks": len(values),
                        "mean": fmean(values),
                        "ci_low": low,
                        "ci_high": high,
                        "confidence": CONFIDENCE,
                        "bootstrap_iterations": plan.iterations,
                        "bootstrap_seed": plan.seed,
                        "bootstrap_unit": "paired_source_task",
                    }
                )
    return result


def paired_effects(
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    *,
    methods: Sequence[str],
    operators: Sequence[str],
    source: str,
    plan: BootstrapPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Success-only paired operator, method, and method×operator effects."""

    units = unit_order(grouped)

    def vector(method: str, operator: str) -> list[float]:
        rows = grouped[(method, operator)]
        return [metric_value(rows[unit], "success", source=source) for unit in units]

    operator_effects: list[dict[str, Any]] = []
    for method in methods:
        control = vector(method, "none")
        for operator in operators:
            if operator == "none":
                continue
            treated = vector(method, operator)
            differences = [b - a for a, b in zip(control, treated, strict=True)]
            low, high = plan.interval(differences)
            operator_effects.append(
                {
                    "paper_class": paper_class(method),
                    "method": method,
                    "operator": operator,
                    "control_operator": "none",
                    "metric": "success",
                    "unit": "proportion_points",
                    "n_tasks": len(units),
                    "control_mean": fmean(control),
                    "operator_mean": fmean(treated),
                    "effect": fmean(differences),
                    "ci_low": low,
                    "ci_high": high,
                    "effect_definition": "operator_minus_none",
                    "confidence": CONFIDENCE,
                    "bootstrap_iterations": plan.iterations,
                    "bootstrap_seed": plan.seed,
                    "bootstrap_unit": "paired_source_task",
                }
            )

    method_effects: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for reference_index, reference_method in enumerate(methods):
        for comparison_method in methods[reference_index + 1 :]:
            reference_none = vector(reference_method, "none")
            comparison_none = vector(comparison_method, "none")
            base_gap = [b - a for a, b in zip(reference_none, comparison_none, strict=True)]
            for operator in operators:
                reference = vector(reference_method, operator)
                comparison = vector(comparison_method, operator)
                gap = [b - a for a, b in zip(reference, comparison, strict=True)]
                low, high = plan.interval(gap)
                method_effects.append(
                    {
                        "reference_class": paper_class(reference_method),
                        "reference_method": reference_method,
                        "comparison_class": paper_class(comparison_method),
                        "comparison_method": comparison_method,
                        "operator": operator,
                        "metric": "success",
                        "unit": "proportion_points",
                        "n_tasks": len(units),
                        "reference_mean": fmean(reference),
                        "comparison_mean": fmean(comparison),
                        "effect": fmean(gap),
                        "ci_low": low,
                        "ci_high": high,
                        "effect_definition": "comparison_method_minus_reference_method",
                        "confidence": CONFIDENCE,
                        "bootstrap_iterations": plan.iterations,
                        "bootstrap_seed": plan.seed,
                        "bootstrap_unit": "paired_source_task",
                    }
                )
                if operator == "none":
                    continue
                interaction = [g - b for b, g in zip(base_gap, gap, strict=True)]
                interaction_low, interaction_high = plan.interval(interaction)
                interactions.append(
                    {
                        "reference_class": paper_class(reference_method),
                        "reference_method": reference_method,
                        "comparison_class": paper_class(comparison_method),
                        "comparison_method": comparison_method,
                        "operator": operator,
                        "control_operator": "none",
                        "metric": "success",
                        "unit": "proportion_points",
                        "n_tasks": len(units),
                        "effect": fmean(interaction),
                        "ci_low": interaction_low,
                        "ci_high": interaction_high,
                        "effect_definition": "(comparison-reference)_operator_minus_(comparison-reference)_none",
                        "confidence": CONFIDENCE,
                        "bootstrap_iterations": plan.iterations,
                        "bootstrap_seed": plan.seed,
                        "bootstrap_unit": "paired_source_task",
                    }
                )
    return operator_effects, method_effects, interactions


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>",
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".title{font-size:24px;font-weight:720;letter-spacing:-.3px}.subtitle{font-size:12.5px;fill:#5b6679}",
        ".label{font-size:12px;font-weight:620}.small{font-size:10.5px;fill:#657188}.tick{font-size:10px;fill:#738096}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="28" y="34">{escape(title)}</text>',
        f'<text class="subtitle" x="28" y="55">{escape(subtitle)}</text>',
        f'<line x1="28" y1="70" x2="{width - 28}" y2="70" stroke="#e5e9ef"/>',
    ]


def summary_map(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["method"]), str(row["operator"])): row for row in rows if row["metric"] == metric}


def online_performance_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    values = summary_map(rows, "success")
    width, height = 1480, 650
    lines = svg_header(
        width,
        height,
        "Online deployment performance",
        "Natural scalar-cutoff policies · 40 paired source tasks per method/operator · 95% task bootstrap intervals",
    )
    panel_left, panel_top, panel_width, panel_height = 70.0, 108.0, 330.0, 390.0
    gap = 25.0
    for panel_index, operator in enumerate(ONLINE_OPERATORS):
        x0 = panel_left + panel_index * (panel_width + gap)
        lines.append(f'<rect x="{x0:.1f}" y="88" width="{panel_width:.1f}" height="450" rx="8" fill="#fafbfc" stroke="#e8ebf0"/>')
        lines.append(f'<text class="label" x="{x0 + panel_width / 2:.1f}" y="105" text-anchor="middle">{escape(OPERATOR_LABELS[operator])}</text>')
        for tick in range(0, 6):
            value = tick / 5
            y = panel_top + panel_height * (1 - value)
            lines.append(f'<line x1="{x0 + 34:.1f}" y1="{y:.1f}" x2="{x0 + panel_width - 12:.1f}" y2="{y:.1f}" stroke="#e7eaf0"/>')
            if panel_index == 0:
                lines.append(f'<text class="tick" x="{x0 + 28:.1f}" y="{y + 3:.1f}" text-anchor="end">{int(value * 100)}%</text>')
        usable = panel_width - 58
        step = usable / len(PLOT_METHODS)
        for method_index, method in enumerate(PLOT_METHODS):
            row = values[(method, operator)]
            center = x0 + 42 + step * (method_index + 0.5)
            mean, low, high = float(row["mean"]), float(row["ci_low"]), float(row["ci_high"])
            y_mean = panel_top + panel_height * (1 - mean)
            y_low = panel_top + panel_height * (1 - low)
            y_high = panel_top + panel_height * (1 - high)
            color = CLASS_COLORS[paper_class(method)]
            lines.extend(
                [
                    f'<rect x="{center - 10:.1f}" y="{y_mean:.1f}" width="20" height="{panel_top + panel_height - y_mean:.1f}" rx="2" fill="{color}" opacity=".88"/>',
                    f'<line x1="{center:.1f}" y1="{y_high:.1f}" x2="{center:.1f}" y2="{y_low:.1f}" stroke="#172033"/>',
                    f'<line x1="{center - 4:.1f}" y1="{y_high:.1f}" x2="{center + 4:.1f}" y2="{y_high:.1f}" stroke="#172033"/>',
                    f'<line x1="{center - 4:.1f}" y1="{y_low:.1f}" x2="{center + 4:.1f}" y2="{y_low:.1f}" stroke="#172033"/>',
                    f'<text class="tick" x="{center:.1f}" y="520" text-anchor="end" transform="rotate(-48 {center:.1f} 520)">{escape(METHOD_LABELS[method])}</text>',
                ]
            )
    legend_x = 490
    for index, klass in enumerate(("baseline", "active", "passive")):
        x = legend_x + index * 170
        lines.append(f'<rect x="{x}" y="591" width="12" height="12" rx="2" fill="{CLASS_COLORS[klass]}"/>')
        lines.append(f'<text class="small" x="{x + 18}" y="601">{klass.title()}</text>')
    lines.append('<text class="small" x="28" y="630">Bars show exact method-level means; “passive” merges behavioral and observational subclasses only for the paper grouping.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def online_firing_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    firing = summary_map(rows, "firing_incidence")
    action = summary_map(rows, "action_incidence")
    width, height = 1480, 625
    lines = svg_header(
        width,
        height,
        "Actual online firing and action incidence",
        "Share of 40 tasks with ≥1 scalar-threshold firing or selected state action; this is not a fixed 20% rank policy",
    )
    panel_left, panel_width, gap = 65.0, 335.0, 22.0
    top, row_gap = 125.0, 56.0
    for panel_index, operator in enumerate(ONLINE_OPERATORS):
        x0 = panel_left + panel_index * (panel_width + gap)
        plot_left, plot_right = x0 + 102, x0 + panel_width - 16
        lines.append(f'<rect x="{x0:.1f}" y="88" width="{panel_width:.1f}" height="450" rx="8" fill="#fafbfc" stroke="#e8ebf0"/>')
        lines.append(f'<text class="label" x="{x0 + panel_width/2:.1f}" y="106" text-anchor="middle">{escape(OPERATOR_LABELS[operator])}</text>')
        for tick in range(0, 5):
            rate = tick / 4
            x = plot_left + (plot_right - plot_left) * rate
            lines.append(f'<line x1="{x:.1f}" y1="115" x2="{x:.1f}" y2="512" stroke="#e5e9ef"/>')
            lines.append(f'<text class="tick" x="{x:.1f}" y="531" text-anchor="middle">{int(rate*100)}%</text>')
        for method_index, method in enumerate(PLOT_METHODS):
            y = top + method_index * row_gap
            if panel_index == 0:
                lines.append(f'<text class="small" x="{x0 + 96:.1f}" y="{y + 4:.1f}" text-anchor="end">{escape(METHOD_LABELS[method])}</text>')
            fire_rate = float(firing[(method, operator)]["mean"])
            action_rate = float(action[(method, operator)]["mean"])
            x_fire = plot_left + (plot_right - plot_left) * fire_rate
            x_action = plot_left + (plot_right - plot_left) * action_rate
            lines.extend(
                [
                    f'<line x1="{min(x_fire,x_action):.1f}" y1="{y:.1f}" x2="{max(x_fire,x_action):.1f}" y2="{y:.1f}" stroke="#b8c0cc" stroke-width="2"/>',
                    f'<circle cx="{x_fire:.1f}" cy="{y:.1f}" r="5" fill="#d97706"/>',
                    f'<rect x="{x_action-5:.1f}" y="{y-5:.1f}" width="10" height="10" rx="2" fill="#2563eb"/>',
                ]
            )
    lines.extend(
        [
            '<circle cx="590" cy="579" r="5" fill="#d97706"/><text class="small" x="602" y="583">Any threshold firing</text>',
            '<rect x="760" y="574" width="10" height="10" rx="2" fill="#2563eb"/><text class="small" x="778" y="583">Any selected action</text>',
            '<text class="small" x="28" y="608">One-action cap applies per task. Exact counts, mean firing counts, and intervals remain in the JSON/CSV sidecars.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def online_resource_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    metrics = (
        ("observer_tokens", "Observer tokens / task"),
        ("total_tokens", "Total tokens / task"),
        ("actual_cost_usd", "Actual USD / task"),
    )
    width, height = 1480, 620
    lines = svg_header(
        width,
        height,
        "Online deployment resources",
        "End-to-end task plus observer accounting; points are method/operator means with 95% source-task bootstrap intervals",
    )
    left, panel_width, gap = 66.0, 430.0, 40.0
    top, row_gap = 135.0, 52.0
    for panel_index, (metric, label) in enumerate(metrics):
        mapped = summary_map(rows, metric)
        x0 = left + panel_index * (panel_width + gap)
        plot_left, plot_right = x0 + 130, x0 + panel_width - 16
        maximum = max(float(row["ci_high"]) for row in mapped.values())
        maximum = maximum * 1.08 if maximum > 0 else 1.0
        lines.append(f'<rect x="{x0:.1f}" y="88" width="{panel_width:.1f}" height="455" rx="8" fill="#fafbfc" stroke="#e8ebf0"/>')
        lines.append(f'<text class="label" x="{x0 + panel_width/2:.1f}" y="107" text-anchor="middle">{escape(label)}</text>')
        for tick in range(5):
            value = maximum * tick / 4
            x = plot_left + (plot_right - plot_left) * tick / 4
            lines.append(f'<line x1="{x:.1f}" y1="120" x2="{x:.1f}" y2="505" stroke="#e5e9ef"/>')
            display = f"{value:.3f}" if metric == "actual_cost_usd" else f"{value:,.0f}"
            lines.append(f'<text class="tick" x="{x:.1f}" y="526" text-anchor="middle">{display}</text>')
        for method_index, method in enumerate(PLOT_METHODS):
            y = top + method_index * row_gap
            lines.append(f'<text class="small" x="{plot_left-8:.1f}" y="{y+4:.1f}" text-anchor="end">{escape(METHOD_LABELS[method])}</text>')
            for operator_index, operator in enumerate(ONLINE_OPERATORS):
                row = mapped[(method, operator)]
                mean, low, high = map(float, (row["mean"], row["ci_low"], row["ci_high"]))
                offset = (operator_index - 1.5) * 5
                y_point = y + offset
                x_mean = plot_left + (plot_right - plot_left) * mean / maximum
                x_low = plot_left + (plot_right - plot_left) * low / maximum
                x_high = plot_left + (plot_right - plot_left) * high / maximum
                color = OPERATOR_COLORS[operator]
                lines.append(f'<line x1="{x_low:.1f}" y1="{y_point:.1f}" x2="{x_high:.1f}" y2="{y_point:.1f}" stroke="{color}" stroke-width="1.5"/>')
                lines.append(f'<circle cx="{x_mean:.1f}" cy="{y_point:.1f}" r="3.6" fill="{color}"/>')
    legend_y = 578
    for index, operator in enumerate(ONLINE_OPERATORS):
        x = 250 + index * 270
        lines.append(f'<circle cx="{x}" cy="{legend_y}" r="4" fill="{OPERATOR_COLORS[operator]}"/>')
        lines.append(f'<text class="small" x="{x+10}" y="{legend_y+4}">{escape(OPERATOR_LABELS[operator])}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def forest_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    subtitle: str,
    label: Callable[[Mapping[str, Any]], str],
    color: Callable[[Mapping[str, Any]], str],
    width: int = 1160,
) -> str:
    height = max(330, 118 + 31 * len(rows))
    lines = svg_header(width, height, title, subtitle)
    left, right, top = 360.0, width - 55.0, 102.0
    low = min(0.0, *(float(row["ci_low"]) for row in rows))
    high = max(0.0, *(float(row["ci_high"]) for row in rows))
    span = high - low
    padding = max(0.015, span * 0.08)
    low, high = low - padding, high + padding

    def x(value: float) -> float:
        return left + (right - left) * (value - low) / (high - low)

    zero = x(0.0)
    lines.append(f'<line x1="{zero:.1f}" y1="84" x2="{zero:.1f}" y2="{height-45}" stroke="#8993a4" stroke-width="1.4"/>')
    for tick in range(5):
        value = low + (high - low) * tick / 4
        xt = x(value)
        lines.append(f'<line x1="{xt:.1f}" y1="84" x2="{xt:.1f}" y2="{height-45}" stroke="#e7eaf0"/>')
        lines.append(f'<text class="tick" x="{xt:.1f}" y="{height-25}" text-anchor="middle">{value*100:+.1f} pp</text>')
    for index, row in enumerate(rows):
        y = top + index * 31
        effect, ci_low, ci_high = map(float, (row["effect"], row["ci_low"], row["ci_high"]))
        lines.append(f'<text class="small" x="{left-12:.1f}" y="{y+4:.1f}" text-anchor="end">{escape(label(row))}</text>')
        lines.append(f'<line x1="{x(ci_low):.1f}" y1="{y:.1f}" x2="{x(ci_high):.1f}" y2="{y:.1f}" stroke="{color(row)}" stroke-width="2"/>')
        lines.append(f'<circle cx="{x(effect):.1f}" cy="{y:.1f}" r="4.5" fill="{color(row)}"/>')
    lines.append(f'<text class="small" x="28" y="{height - 9}">Effects are paired by source task; intervals are descriptive percentile bootstrap intervals without multiplicity correction.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def yoked_sensitivity_svg(method_effects: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]]) -> str:
    effects = [row for row in method_effects if row["reference_method"] == "active_recompute"]
    changes = [row for row in interactions if row["reference_method"] == "active_recompute"]
    width, height = 1480, 600
    lines = svg_header(
        width,
        height,
        "Yoked controlled sensitivity",
        "Same 40 tasks; one active-anchored action at checkpoint 1 in every cell · isolates carry/operator sensitivity, not natural trigger timing",
    )

    def panel(rows: Sequence[Mapping[str, Any]], x0: float, panel_width: float, heading: str, row_label: Callable[[Mapping[str, Any]], str]) -> None:
        low = min(0.0, *(float(row["ci_low"]) for row in rows))
        high = max(0.0, *(float(row["ci_high"]) for row in rows))
        padding = max(0.02, (high - low) * 0.08)
        low, high = low - padding, high + padding
        plot_left, plot_right = x0 + 235, x0 + panel_width - 25

        def x(value: float) -> float:
            return plot_left + (plot_right - plot_left) * (value - low) / (high - low)

        lines.append(f'<rect x="{x0}" y="87" width="{panel_width}" height="455" rx="8" fill="#fafbfc" stroke="#e8ebf0"/>')
        lines.append(f'<text class="label" x="{x0+panel_width/2:.1f}" y="108" text-anchor="middle">{escape(heading)}</text>')
        for tick in range(5):
            value = low + (high - low) * tick / 4
            xt = x(value)
            lines.append(f'<line x1="{xt:.1f}" y1="122" x2="{xt:.1f}" y2="505" stroke="{("#8993a4" if abs(value)<1e-12 else "#e7eaf0")}"/>')
            lines.append(f'<text class="tick" x="{xt:.1f}" y="526" text-anchor="middle">{value*100:+.1f} pp</text>')
        zero = x(0.0)
        lines.append(f'<line x1="{zero:.1f}" y1="122" x2="{zero:.1f}" y2="505" stroke="#8993a4"/>')
        row_gap = 37 if len(rows) <= 9 else 27
        for index, row in enumerate(rows):
            y = 145 + index * row_gap
            effect, ci_low, ci_high = map(float, (row["effect"], row["ci_low"], row["ci_high"]))
            color_value = OPERATOR_COLORS[str(row["operator"])]
            lines.append(f'<text class="small" x="{plot_left-10:.1f}" y="{y+4:.1f}" text-anchor="end">{escape(row_label(row))}</text>')
            lines.append(f'<line x1="{x(ci_low):.1f}" y1="{y:.1f}" x2="{x(ci_high):.1f}" y2="{y:.1f}" stroke="{color_value}" stroke-width="2"/>')
            lines.append(f'<circle cx="{x(effect):.1f}" cy="{y:.1f}" r="4.5" fill="{color_value}"/>')

    panel(effects, 28, 690, "Method − active recompute, within operator", lambda row: f"{METHOD_LABELS[str(row['comparison_method'])]} · {OPERATOR_LABELS[str(row['operator'])]}")
    panel(changes, 750, 700, "Change in method gap, operator versus none", lambda row: f"{METHOD_LABELS[str(row['comparison_method'])]} · {OPERATOR_LABELS[str(row['operator'])]}")
    lines.append('<text class="small" x="28" y="579">Positive values favor the comparison method. This is a controlled schedule sensitivity on the same source-task sample, not an independent replication.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_figure(output_dir: Path, stem: str, svg: str, data: Mapping[str, Any]) -> tuple[str, str]:
    svg_path = output_dir / f"{stem}.svg"
    data_path = output_dir / f"{stem}.data.json"
    atomic_text(svg_path, svg)
    atomic_json(data_path, data)
    return svg_path.name, data_path.name


def full_analysis(online_path: Path, yoked_path: Path, output_dir: Path, iterations: int) -> dict[str, Any]:
    online_path = online_path.resolve()
    yoked_path = yoked_path.resolve()
    output_dir = output_dir.resolve()
    code_hash = code_tree_hash(ROOT / "experiments12")
    require(code_hash == EXPECTED_CODE_HASH, f"frozen code tree changed: {code_hash}")
    online_payload, online_grouped = verify_online(online_path)
    yoked_payload, yoked_grouped = verify_yoked(yoked_path)
    plan = BootstrapPlan(40, iterations, SEED)

    online_performance = summarize(
        online_grouped,
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        metrics=("success",),
        source="online",
        plan=plan,
        role="primary_online_natural_policy",
    )
    online_operating = summarize(
        online_grouped,
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        metrics=(
            "firing_incidence", "action_incidence", "firing_rate", "action_rate",
            "threshold_firings", "selected_actions",
            "task_tokens", "observer_tokens", "total_tokens", "latency_ms", "actual_cost_usd",
        ),
        source="online",
        plan=plan,
        role="primary_online_natural_policy",
    )
    online_operator, online_method, online_interactions = paired_effects(
        online_grouped,
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        source="online",
        plan=plan,
    )
    yoked_summaries = summarize(
        yoked_grouped,
        methods=YOKED_METHODS,
        operators=YOKED_OPERATORS,
        metrics=("success", "action_incidence", "action_rate", "task_tokens", "observer_tokens", "total_tokens", "latency_ms", "actual_cost_usd"),
        source="yoked",
        plan=plan,
        role="controlled_yoked_checkpoint1_sensitivity",
    )
    yoked_operator, yoked_method, yoked_interactions = paired_effects(
        yoked_grouped,
        methods=YOKED_METHODS,
        operators=YOKED_OPERATORS,
        source="yoked",
        plan=plan,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_payloads = {
        "online-performance.csv": online_performance,
        "online-operating-resources.csv": online_operating,
        "online-operator-effects.csv": online_operator,
        "online-method-effects.csv": online_method,
        "online-method-operator-interactions.csv": online_interactions,
        "yoked-controlled-summaries.csv": yoked_summaries,
        "yoked-operator-effects.csv": yoked_operator,
        "yoked-method-effects.csv": yoked_method,
        "yoked-method-operator-interactions.csv": yoked_interactions,
    }
    for name, rows in csv_payloads.items():
        write_csv(output_dir / name, rows)

    figures: list[str] = []
    sidecars: list[str] = []
    figure_specs = (
        (
            "online-performance",
            online_performance_svg(online_performance),
            {"figure_type": "grouped_online_success", "analysis_role": "primary_online_natural_policy", "rows": online_performance},
        ),
        (
            "online-firing-actions",
            online_firing_svg(online_operating),
            {"figure_type": "online_firing_action_incidence", "policy_warning": "scalar cutoff; not fixed-count rank", "rows": [row for row in online_operating if row["metric"] in {"firing_incidence", "action_incidence", "firing_rate", "action_rate", "threshold_firings", "selected_actions"}]},
        ),
        (
            "online-resources",
            online_resource_svg(online_operating),
            {"figure_type": "online_end_to_end_resources", "rows": [row for row in online_operating if row["metric"] in {"observer_tokens", "total_tokens", "actual_cost_usd"}]},
        ),
        (
            "online-success-interactions",
            forest_svg(
                [row for row in online_interactions if row["reference_method"] == "active_recompute"],
                title="Online method × operator interactions",
                subtitle="Change in each zero-carry method’s success gap versus active recompute, operator relative to no state action",
                label=lambda row: f"{METHOD_LABELS[str(row['comparison_method'])]} · {OPERATOR_LABELS[str(row['operator'])]}",
                color=lambda row: OPERATOR_COLORS[str(row["operator"])],
            ),
            {"figure_type": "online_success_method_operator_interactions", "effect_definition": "(comparison-active)_operator - (comparison-active)_none", "rows": [row for row in online_interactions if row["reference_method"] == "active_recompute"]},
        ),
        (
            "yoked-controlled-sensitivity",
            yoked_sensitivity_svg(yoked_method, yoked_interactions),
            {"figure_type": "yoked_checkpoint1_controlled_sensitivity", "schedule": "one active-anchored action at checkpoint 1 in every method/operator cell", "not_an_independent_replication": True, "method_effects": [row for row in yoked_method if row["reference_method"] == "active_recompute"], "interactions": [row for row in yoked_interactions if row["reference_method"] == "active_recompute"]},
        ),
    )
    for stem, svg, data in figure_specs:
        figure, sidecar = write_figure(output_dir, stem, svg, {"schema_version": 1, **data})
        figures.append(figure)
        sidecars.append(sidecar)

    result = {
        "post_analysis_version": VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "provider_calls_made": 0,
        "statistical_unit": "source_task",
        "bootstrap": {
            "method": "shared deterministic source-task percentile bootstrap",
            "confidence": CONFIDENCE,
            "iterations": iterations,
            "seed": SEED,
            "namespace": "exp12/deployment-paper-bootstrap/v1",
            "multiplicity_adjustment": None,
        },
        "paper_grouping": {
            "baseline": ["turn_clock", "context_use"],
            "active": ["active_recompute"],
            "passive": ["frozen_probe:recompute", "frozen_quiz", "trace_judge", "trace_rules"],
            "note": "passive-behavioral and passive-observational remain distinct methods but are merged only at the visual class level",
        },
        "provenance": {
            "code_tree_sha256": code_hash,
            "online_analysis_path": str(online_path.relative_to(ROOT)),
            "online_analysis_sha256": sha256_file(online_path),
            "online_source_run_id": ONLINE_RUN,
            "online_source_manifest_sha256": online_payload["source_manifest_sha256"],
            "online_source_pair_manifest_sha256": online_payload["source_pair_manifest_sha256"],
            "yoked_analysis_path": str(yoked_path.relative_to(ROOT)),
            "yoked_analysis_sha256": sha256_file(yoked_path),
            "yoked_source_run_id": YOKED_RUN,
            "yoked_source_manifest_sha256": yoked_payload["source_manifest_sha256"],
            "yoked_source_pair_manifest_sha256": yoked_payload["source_pair_manifest_sha256"],
            "yoked_source_schedule_sha256": yoked_payload["source_schedule_sha256"],
        },
        "semantics": {
            "online": "primary ecological deployment: separately generated trajectories under natural scalar-cutoff policies with a one-action task cap",
            "yoked": "controlled sensitivity on the same 40 source tasks: one active-anchored action at checkpoint 1 in every method/operator cell; not natural timing and not independent replication",
            "method_effect": "comparison method minus reference method on identical source tasks",
            "operator_effect": "operator minus none within method on identical source tasks",
            "interaction": "(comparison-reference) under operator minus the same method gap under none",
            "resource_warning": "online resources are end-to-end; yoked resources are pass-two sensitivity and do not duplicate frozen pass-one passive observer cost",
        },
        "online_primary": {
            "performance_summaries": online_performance,
            "operating_resource_summaries": online_operating,
            "operator_effects": online_operator,
            "method_effects": online_method,
            "method_operator_interactions": online_interactions,
        },
        "yoked_controlled_sensitivity": {
            "schedule": {"source_tasks": 40, "actions_per_cell": 1, "action_checkpoint": 1, "anchor_method": "active_recompute"},
            "summaries": yoked_summaries,
            "operator_effects": yoked_operator,
            "method_effects": yoked_method,
            "method_operator_interactions": yoked_interactions,
        },
        "csv_files": sorted(csv_payloads),
        "figure_files": figures,
        "figure_data_files": sidecars,
    }
    atomic_json(output_dir / "deployment-paper-post-analysis.json", result)
    require(code_tree_hash(ROOT / "experiments12") == EXPECTED_CODE_HASH, "code tree changed during post-analysis")
    return result


def expected_command() -> str:
    return (
        "python3 experiments12/scripts/posthoc/deployment_post_analysis12.py "
        f"--online {DEFAULT_ONLINE.relative_to(ROOT)} "
        f"--yoked {DEFAULT_YOKED.relative_to(ROOT)} "
        f"--output-dir {DEFAULT_OUTPUT.relative_to(ROOT)} "
        f"--bootstrap-iterations {DEFAULT_ITERATIONS}"
    )


def dry_check() -> dict[str, Any]:
    code_hash = code_tree_hash(ROOT / "experiments12")
    require(code_hash == EXPECTED_CODE_HASH, f"frozen code tree changed: {code_hash}")
    verify_run_file(ONLINE_RUN, "manifest.json", ONLINE_MANIFEST_SHA256)
    verify_run_file(ONLINE_RUN, "pairs.jsonl", ONLINE_PAIRS_SHA256)
    verify_run_file(YOKED_RUN, "manifest.json", YOKED_MANIFEST_SHA256)
    verify_run_file(YOKED_RUN, "pairs.jsonl", YOKED_PAIRS_SHA256)
    verify_run_file(YOKED_RUN, "results/deployment_schedule.json", YOKED_SCHEDULE_SHA256)
    return {
        "dry_check": "passed",
        "provider_calls_made": 0,
        "code_tree_sha256": code_hash,
        "online_analysis_ready": DEFAULT_ONLINE.is_file(),
        "yoked_analysis_ready": DEFAULT_YOKED.is_file(),
        "full_command_after_both_inputs_exist": expected_command(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Create provider-free paper tables and figures from complete Experiment 12 "
            "online and yoked deployment analyses."
        )
    )
    result.add_argument("--online", type=Path, default=DEFAULT_ONLINE)
    result.add_argument("--yoked", type=Path, default=DEFAULT_YOKED)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_ITERATIONS)
    result.add_argument(
        "--dry-check",
        action="store_true",
        help="verify frozen static provenance and print the future command without loading analyses",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.dry_check:
            print(json.dumps(dry_check(), indent=2, sort_keys=True))
            return 0
        require(args.bootstrap_iterations > 0, "bootstrap iterations must be positive")
        result = full_analysis(args.online, args.yoked, args.output_dir, args.bootstrap_iterations)
        print(
            json.dumps(
                {
                    "output": str((args.output_dir / "deployment-paper-post-analysis.json").resolve()),
                    "csv_files": len(result["csv_files"]),
                    "figure_files": len(result["figure_files"]),
                    "provider_calls_made": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    except (PostAnalysisError, FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
