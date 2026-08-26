#!/usr/bin/env python3
"""Independent raw-row audit for deployment-interaction-confirmatory-v1.

This audit does not import the generated figure builder or either source
analyzer.  It reconstructs all displayed n=40 online and yoked success effects
from raw analysis rows, independently repeats each analyzer's frozen paired
bootstrap, independently recomputes the cumulative n=38 leave-two-unit effects,
and verifies JSON, CSV, SVG sidecar, SVG semantics, provenance, and receipt
hashes.  It makes no provider calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments12.manifest12 import code_tree_hash  # noqa: E402


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
YOKED_ANALYSIS_SHA256 = (
    "3b8344d1888e4f42e61d5fa6c67acc7ad2a81fcb53c6fc1bfb4434006f02eeb0"
)

ARTIFACTS = ROOT / "experiments12" / "data_results" / "runs"
GENERATED = ROOT / "experiments12" / "data_results" / "derived"
ONLINE_ANALYSIS = (
    GENERATED
    / "adaptive-analysis-staging-v1"
    / "analysis"
    / "adaptive-analysis.json"
)
ONLINE_SENSITIVITY = ONLINE_ANALYSIS.with_name(
    "adaptive-analysis-leave-two-units.json"
)
YOKED_ANALYSIS = (
    ARTIFACTS
    / YOKED_RUN
    / "results"
    / "two-pass-analysis.json"
)
STEM = GENERATED / "deployment-interaction-confirmatory-v1"
BUILDER = GENERATED / "build_deployment_interaction12.py"

MODEL = "gpt-5.6-luna"
BENCHMARK = "evolving_intent_gsm8k"
CONTROL = "none"
CONFIDENCE = 0.95
ITERATIONS = 2_000
SEED = 12_012
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
PLOT_METHODS = (
    "turn_clock",
    "context_use",
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
)
ONLINE_OPERATORS = (
    "none",
    "lossy_compaction",
    "public_state_reground",
    "good_bad_watch_feedback",
)
YOKED_OPERATORS = (
    "none",
    "lossy_compaction",
    "public_state_reground",
)
ONLINE_ROW_FIELDS = {
    "cell_id", "model", "benchmark", "task_id", "replicate_id", "unit_id",
    "method", "observation_class", "operator", "deployment_mode", "success",
    "observations", "threshold_firings", "selected_actions",
    "applied_interventions", "task_tokens", "observer_tokens", "total_tokens",
    "latency_ms", "actual_cost_usd",
}
YOKED_ROW_FIELDS = {
    "cell_id", "model", "benchmark", "task_id", "replicate_id", "unit_id",
    "observation_class", "method", "operator", "deployment_mode", "estimand",
    "success", "outcome_source", "observations", "scheduled_actions",
    "action_rate", "acted_on_task", "applied_interventions", "task_tokens",
    "observer_tokens", "total_tokens", "latency_ms", "actual_cost_usd",
    "reported_cost_usd", "estimated_cost_usd", "upper_bound_cost_usd",
    "failed_retry_attempts",
}
ONLINE_TOP_FIELDS = {
    "adaptive_analysis_version", "artifact_type", "source_run_id",
    "source_manifest_sha256", "source_pair_manifest_sha256", "deployment_mode",
    "deployment_policy", "per_task_action_cap", "statistical_unit",
    "comparison_semantics", "resource_semantics", "rows", "metric_summaries",
    "operator_effects",
}
YOKED_TOP_FIELDS = {
    "two_pass_analysis_version", "artifact_type", "source_run_id",
    "source_manifest_sha256", "source_pair_manifest_sha256",
    "source_schedule_sha256", "deployment_mode", "estimand", "statistical_unit",
    "comparison_semantics", "resource_semantics", "rows", "metric_summaries",
    "operator_effects", "method_effects", "validation",
}
SENSITIVITY_TOP_FIELDS = {
    "artifact_type", "sensitivity_version", "source_run_id",
    "source_analysis_path", "source_analysis_sha256", "exclusion_reason",
    "excluded_source_units", "exclusion_scope", "treatments",
    "excluded_rows_per_treatment", "excluded_rows", "remaining_rows",
    "remaining_source_tasks_per_treatment",
    "balanced_paired_design_after_exclusion", "excluded_cell_ids", "rows",
    "metric_summaries", "operator_effects",
}
EXPECTED_AFFECTED_UNITS = {
    "extracted-gsm8k-test-814::t7/r0",
    "extracted-gsm8k-test-989::t7/r0",
}
FLOAT_FIELDS = {
    "control_success", "operator_success", "effect", "ci_low", "ci_high",
    "confidence", "natural_action_task_rate", "natural_action_checkpoint_rate",
    "sensitivity_n38_effect", "sensitivity_n38_ci_low", "sensitivity_n38_ci_high",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_TOLERANCE = 1e-12


class AuditFailure(AssertionError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def checked_digest(value: str, *, context: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value), f"invalid {context}")
    return value


def read_json(path: Path) -> Mapping[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"JSON is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSONL: {path}")
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), f"JSONL row invalid: {path}:{index}")
        result.append(value)
    return result


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def observation_class(method: str) -> str:
    if method == "active_recompute":
        return "active"
    if method in {"turn_clock", "context_use"}:
        return "baseline"
    if method in {"frozen_probe:recompute", "frozen_quiz"}:
        return "passive-behavioral"
    if method in {"trace_judge", "trace_rules"}:
        return "passive-observational"
    raise AuditFailure(f"unknown method: {method}")


def paper_class(method: str) -> str:
    value = observation_class(method)
    return "passive" if value.startswith("passive-") else value


def numeric(value: Any, *, context: str, integer: bool = False) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"non-numeric {context}")
    number = float(value)
    require(math.isfinite(number) and number >= 0, f"invalid {context}")
    if integer:
        require(isinstance(value, int), f"non-integer {context}")
    return number


def same_number(left: Any, right: Any, *, context: str) -> None:
    require(
        isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12),
        f"numeric mismatch: {context}",
    )


def verify_hash(path: Path, expected: str, *, context: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing or linked {context}")
    require(sha_file(path) == expected, f"{context} hash changed")


def grouped_rows(
    payload: Mapping[str, Any],
    *,
    source: str,
    methods: Sequence[str],
    operators: Sequence[str],
    fields: set[str],
    expected_rows: int,
    expected_tasks: int,
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == expected_rows, f"{source} row dimension")
    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    cells: set[str] = set()
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping) and set(row) == fields, f"{source} row {index} schema")
        method, operator = row.get("method"), row.get("operator")
        require(method in methods and operator in operators, f"{source} treatment changed")
        require(row.get("model") == MODEL and row.get("benchmark") == BENCHMARK, f"{source} slice changed")
        require(row.get("replicate_id") == 0 and not isinstance(row.get("replicate_id"), bool), f"{source} replicate changed")
        require(row.get("unit_id") == f"{row.get('task_id')}/r0", f"{source} unit identity")
        require(row.get("observation_class") == observation_class(str(method)), f"{source} observation class")
        require(isinstance(row.get("success"), bool), f"{source} success type")
        require(isinstance(row.get("cell_id"), str) and row["cell_id"] not in cells, f"{source} duplicate cell")
        cells.add(str(row["cell_id"]))
        observations = numeric(row.get("observations"), context=f"{source} observations", integer=True)
        require(observations > 0, f"{source} no observations")
        for field in ("task_tokens", "observer_tokens", "total_tokens", "latency_ms"):
            numeric(row.get(field), context=f"{source} {field}", integer=True)
        numeric(row.get("actual_cost_usd"), context=f"{source} cost")
        require(row["total_tokens"] == row["task_tokens"] + row["observer_tokens"], f"{source} token accounting")
        if source == "online":
            require(row.get("deployment_mode") == "online_adaptive", "online mode")
            for field in ("threshold_firings", "selected_actions", "applied_interventions"):
                numeric(row.get(field), context=f"online {field}", integer=True)
            require(row["selected_actions"] <= 1 and row["applied_interventions"] == row["selected_actions"], "online action accounting")
        else:
            require(row.get("deployment_mode") == "two_pass_frozen" and row.get("estimand") == "yoked_anchor", "yoked design")
            for field in ("scheduled_actions", "acted_on_task", "applied_interventions", "failed_retry_attempts"):
                numeric(row.get(field), context=f"yoked {field}", integer=True)
            same_number(row.get("action_rate"), row["scheduled_actions"] / row["observations"], context="yoked action rate")
            require(row["acted_on_task"] == int(row["scheduled_actions"] > 0), "yoked acted flag")
            require(row["applied_interventions"] == row["scheduled_actions"], "yoked applied count")
        treatment = groups.setdefault((str(method), str(operator)), {})
        require(row["unit_id"] not in treatment, f"{source} duplicate treatment unit")
        treatment[str(row["unit_id"])] = row
    expected_product = {(method, operator) for method in methods for operator in operators}
    require(set(groups) == expected_product, f"{source} treatment product")
    unit_set = set(next(iter(groups.values())))
    require(len(unit_set) == expected_tasks, f"{source} task denominator")
    require(all(set(rows_by_unit) == unit_set for rows_by_unit in groups.values()), f"{source} pairing")
    return groups


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    require(bool(sorted_values), "bootstrap values empty")
    position = probability * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_interval(
    values: Sequence[float], *, source: str, method: str, operator: str
) -> tuple[float, float]:
    require(bool(values), "bootstrap values empty")
    if source == "online":
        namespace = "exp12/adaptive-task-bootstrap/v1"
        identity = (MODEL, BENCHMARK, method, operator, "success", "paired")
    else:
        namespace = "exp12/two-pass-task-bootstrap/v1"
        identity = (MODEL, BENCHMARK, "yoked_anchor", method, operator, "success", "operator")
    prefix = "\0".join((namespace, str(SEED), *identity))
    population = len(values)
    means: list[float] = []
    for iteration in range(ITERATIONS):
        sample = []
        for draw in range(population):
            material = f"{prefix}\0{iteration}\0{draw}".encode("utf-8")
            selected = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % population
            sample.append(values[selected])
        means.append(fmean(sample))
    means.sort()
    tail = (1 - CONFIDENCE) / 2
    return quantile(means, tail), quantile(means, 1 - tail)


def classify_sign(value: float) -> str:
    if value > ZERO_TOLERANCE:
        return "positive"
    if value < -ZERO_TOLERANCE:
        return "negative"
    return "zero"


def classify_interval(low: float, high: float) -> str:
    if low > ZERO_TOLERANCE:
        return "positive_excludes_zero"
    if high < -ZERO_TOLERANCE:
        return "negative_excludes_zero"
    return "includes_zero"


def recompute_effect(
    groups: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    *,
    source: str,
    method: str,
    operator: str,
    excluded_units: set[str] | None = None,
) -> dict[str, float | int]:
    control = groups[(method, CONTROL)]
    treated = groups[(method, operator)]
    units = sorted(set(control) - (excluded_units or set()))
    require(set(units) == set(treated) - (excluded_units or set()), "effect pairing changed")
    control_values = [float(control[unit]["success"]) for unit in units]
    treated_values = [float(treated[unit]["success"]) for unit in units]
    differences = [value - baseline for baseline, value in zip(control_values, treated_values, strict=True)]
    low, high = bootstrap_interval(differences, source=source, method=method, operator=operator)
    return {
        "n_tasks": len(units),
        "control_mean": fmean(control_values),
        "operator_mean": fmean(treated_values),
        "effect": fmean(differences),
        "ci_low": low,
        "ci_high": high,
    }


def source_effect_map(
    payload: Mapping[str, Any],
    *,
    expected_count: int,
    expected_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = payload.get("operator_effects")
    require(isinstance(rows, list) and len(rows) == expected_count, "source operator-effect dimension")
    success_rows = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("metric") == "success"
    ]
    result = {
        (str(row["method"]), str(row["operator"])): row
        for row in success_rows
    }
    require(
        len(success_rows) == len(expected_keys) and set(result) == expected_keys,
        "source success-effect coverage",
    )
    return result


def verify_source_effect(recomputed: Mapping[str, Any], source: Mapping[str, Any], *, context: str) -> None:
    require(
        source.get("control_operator") == CONTROL
        and source.get("effect_definition") == "operator_minus_none"
        and source.get("confidence") == CONFIDENCE
        and source.get("bootstrap_iterations") == ITERATIONS
        and source.get("bootstrap_seed") == SEED
        and source.get("bootstrap_unit") == "paired_source_task",
        f"source effect contract: {context}",
    )
    for field in ("n_tasks", "control_mean", "operator_mean", "effect", "ci_low", "ci_high"):
        if field == "n_tasks":
            require(source.get(field) == recomputed[field], f"source n mismatch: {context}")
        else:
            same_number(source.get(field), recomputed[field], context=f"{context}/{field}")


def sensitivity_map(
    payload: Mapping[str, Any],
    *,
    online_sha256: str,
    online_payload: Mapping[str, Any],
    online_groups: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
) -> tuple[
    dict[tuple[str, str], dict[str, Mapping[str, Any]]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    require(
        set(payload) == SENSITIVITY_TOP_FIELDS
        and payload.get("sensitivity_version") == 1
        and payload.get("artifact_type")
        == "experiment12_online_adaptive_leave_two_source_units_sensitivity"
        and payload.get("source_run_id") == ONLINE_RUN
        and payload.get("source_analysis_path") == relative(ONLINE_ANALYSIS)
        and payload.get("source_analysis_sha256") == online_sha256
        and payload.get("exclusion_reason")
        == "three cells required semantic judge-attempt recovery"
        and payload.get("exclusion_scope")
        == "both_source_units_from_every_method_operator_treatment"
        and set(payload.get("excluded_source_units", ())) == EXPECTED_AFFECTED_UNITS
        and payload.get("treatments") == 28
        and payload.get("excluded_rows_per_treatment") == 2
        and payload.get("excluded_rows") == 56
        and payload.get("remaining_rows") == 1064
        and payload.get("remaining_source_tasks_per_treatment") == 38
        and payload.get("balanced_paired_design_after_exclusion") is True,
        "sensitivity contract changed",
    )
    sensitivity_groups = grouped_rows(
        payload,
        source="online",
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        fields=ONLINE_ROW_FIELDS,
        expected_rows=1064,
        expected_tasks=38,
    )
    full_rows, sensitivity_rows = online_payload.get("rows"), payload.get("rows")
    require(isinstance(full_rows, list) and isinstance(sensitivity_rows, list), "sensitivity rows missing")
    full_by_cell = {str(row["cell_id"]): row for row in full_rows}
    sensitivity_by_cell = {str(row["cell_id"]): row for row in sensitivity_rows}
    excluded_by_cell = {
        str(row["cell_id"]): row
        for row in full_rows
        if str(row["unit_id"]) in EXPECTED_AFFECTED_UNITS
    }
    excluded_cell_ids = payload.get("excluded_cell_ids")
    require(
        len(full_by_cell) == 1120
        and len(sensitivity_by_cell) == 1064
        and len(excluded_by_cell) == 56
        and isinstance(excluded_cell_ids, list)
        and excluded_cell_ids == sorted(excluded_by_cell)
        and set(sensitivity_by_cell) == set(full_by_cell) - set(excluded_by_cell)
        and all(
            sensitivity_by_cell[cell] == full_by_cell[cell]
            for cell in sensitivity_by_cell
        ),
        "sensitivity rows are not the exact leave-two subset",
    )
    full_units = set(next(iter(online_groups.values())))
    sensitivity_units = set(next(iter(sensitivity_groups.values())))
    require(
        EXPECTED_AFFECTED_UNITS <= full_units
        and sensitivity_units == full_units - EXPECTED_AFFECTED_UNITS,
        "sensitivity source-task set changed",
    )
    summaries = payload.get("metric_summaries")
    require(isinstance(summaries, list) and len(summaries) == 224, "sensitivity summary dimension")
    expected_keys = {
        (method, operator)
        for method in ONLINE_METHODS
        for operator in ONLINE_OPERATORS[1:]
    }
    effects = source_effect_map(
        payload,
        expected_count=168,
        expected_keys=expected_keys,
    )
    return sensitivity_groups, effects


def expected_rows(
    online_groups: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    online_source_effects: Mapping[tuple[str, str], Mapping[str, Any]],
    sensitivity_groups: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    sensitivity_effects: Mapping[tuple[str, str], Mapping[str, Any]],
    yoked_groups: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    yoked_source_effects: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for panel, source, methods, operators, groups, source_effects in (
        ("online_natural", "online", PLOT_METHODS, ONLINE_OPERATORS[1:], online_groups, online_source_effects),
        ("yoked_checkpoint1", "yoked", tuple(method for method in PLOT_METHODS if method in YOKED_METHODS), YOKED_OPERATORS[1:], yoked_groups, yoked_source_effects),
    ):
        for method in methods:
            for operator in operators:
                n40 = recompute_effect(groups, source=source, method=method, operator=operator)
                verify_source_effect(n40, source_effects[(method, operator)], context=f"{source}/{method}/{operator}")
                if source == "online":
                    n38 = recompute_effect(
                        sensitivity_groups,
                        source="online",
                        method=method,
                        operator=operator,
                    )
                    verify_source_effect(
                        n38,
                        sensitivity_effects[(method, operator)],
                        context=f"sensitivity/{method}/{operator}",
                    )
                    sign_changed = classify_sign(float(n40["effect"])) != classify_sign(float(n38["effect"]))
                    ci_changed = classify_interval(float(n40["ci_low"]), float(n40["ci_high"])) != classify_interval(float(n38["ci_low"]), float(n38["ci_high"]))
                    treated = groups[(method, operator)]
                    task_rate = fmean(float(row["selected_actions"] > 0) for row in treated.values())
                    checkpoint_rate = fmean(float(row["selected_actions"]) / float(row["observations"]) for row in treated.values())
                    sensitivity_effect = float(n38["effect"])
                    sensitivity_low = float(n38["ci_low"])
                    sensitivity_high = float(n38["ci_high"])
                else:
                    task_rate = checkpoint_rate = None
                    sign_changed = ci_changed = False
                    sensitivity_effect = sensitivity_low = sensitivity_high = None
                result.append(
                    {
                        "panel": panel,
                        "analysis_role": (
                            "primary_online_natural_policy"
                            if source == "online"
                            else "controlled_aggressive_checkpoint1_yoked_sensitivity"
                        ),
                        "paper_class": paper_class(method),
                        "method": method,
                        "operator": operator,
                        "control_operator": CONTROL,
                        "n_tasks": int(n40["n_tasks"]),
                        "control_success": float(n40["control_mean"]),
                        "operator_success": float(n40["operator_mean"]),
                        "effect": float(n40["effect"]),
                        "ci_low": float(n40["ci_low"]),
                        "ci_high": float(n40["ci_high"]),
                        "confidence": CONFIDENCE,
                        "bootstrap_iterations": ITERATIONS,
                        "bootstrap_seed": SEED,
                        "bootstrap_unit": "paired_source_task",
                        "natural_action_task_rate": task_rate,
                        "natural_action_checkpoint_rate": checkpoint_rate,
                        "n38_point_sign_changed": sign_changed,
                        "n38_ci_relation_changed": ci_changed,
                        "n38_qualitative_changed": sign_changed or ci_changed,
                        "sensitivity_n38_effect": sensitivity_effect,
                        "sensitivity_n38_ci_low": sensitivity_low,
                        "sensitivity_n38_ci_high": sensitivity_high,
                    }
                )
    require(len(result) == 29, "recomputed figure row dimension")
    return result


def compare_row(observed: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    require(set(observed) == set(expected), f"row schema mismatch: {context}")
    for field, value in expected.items():
        if field in FLOAT_FIELDS and value is not None:
            same_number(observed.get(field), value, context=f"{context}/{field}")
        else:
            require(observed.get(field) == value, f"row mismatch: {context}/{field}")


def axis_half(rows: Sequence[Mapping[str, Any]]) -> float:
    maximum = max(abs(float(row[field])) for row in rows for field in ("effect", "ci_low", "ci_high"))
    return min(1.0, max(0.10, math.ceil(maximum * 20 - 1e-12) / 20))


def audit(online_sha256: str, sensitivity_sha256: str) -> dict[str, Any]:
    online_sha256 = checked_digest(online_sha256, context="online SHA256")
    sensitivity_sha256 = checked_digest(sensitivity_sha256, context="sensitivity SHA256")
    require(code_tree_hash(ROOT / "experiments12") == EXPECTED_CODE_HASH, "frozen code hash changed")
    verify_hash(ONLINE_ANALYSIS, online_sha256, context="staged online analysis")
    verify_hash(ONLINE_SENSITIVITY, sensitivity_sha256, context="leave-two sensitivity")
    verify_hash(YOKED_ANALYSIS, YOKED_ANALYSIS_SHA256, context="yoked analysis")
    for path, expected, context in (
        (ARTIFACTS / ONLINE_RUN / "manifest.json", ONLINE_MANIFEST_SHA256, "online manifest"),
        (ARTIFACTS / ONLINE_RUN / "pairs.jsonl", ONLINE_PAIRS_SHA256, "online pairs"),
        (ARTIFACTS / YOKED_RUN / "manifest.json", YOKED_MANIFEST_SHA256, "yoked manifest"),
        (ARTIFACTS / YOKED_RUN / "pairs.jsonl", YOKED_PAIRS_SHA256, "yoked pairs"),
        (ARTIFACTS / YOKED_RUN / "results" / "deployment_schedule.json", YOKED_SCHEDULE_SHA256, "yoked schedule"),
    ):
        verify_hash(path, expected, context=context)

    online_payload, yoked_payload = read_json(ONLINE_ANALYSIS), read_json(YOKED_ANALYSIS)
    require(
        set(online_payload) == ONLINE_TOP_FIELDS
        and online_payload.get("adaptive_analysis_version") == 1
        and online_payload.get("artifact_type") == "online_adaptive_deployment_analysis"
        and online_payload.get("source_run_id") == ONLINE_RUN
        and online_payload.get("source_manifest_sha256") == ONLINE_MANIFEST_SHA256
        and online_payload.get("source_pair_manifest_sha256") == ONLINE_PAIRS_SHA256
        and online_payload.get("deployment_policy") == "natural_threshold_per_task_cap"
        and online_payload.get("per_task_action_cap") == 1,
        "online analysis contract",
    )
    require(
        set(yoked_payload) == YOKED_TOP_FIELDS
        and yoked_payload.get("two_pass_analysis_version") == 1
        and yoked_payload.get("artifact_type") == "two_pass_deployment_analysis"
        and yoked_payload.get("source_run_id") == YOKED_RUN
        and yoked_payload.get("source_manifest_sha256") == YOKED_MANIFEST_SHA256
        and yoked_payload.get("source_pair_manifest_sha256") == YOKED_PAIRS_SHA256
        and yoked_payload.get("source_schedule_sha256") == YOKED_SCHEDULE_SHA256
        and yoked_payload.get("estimand") == "yoked_anchor",
        "yoked analysis contract",
    )
    online_groups = grouped_rows(
        online_payload,
        source="online",
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        fields=ONLINE_ROW_FIELDS,
        expected_rows=1120,
        expected_tasks=40,
    )
    yoked_groups = grouped_rows(
        yoked_payload,
        source="yoked",
        methods=YOKED_METHODS,
        operators=YOKED_OPERATORS,
        fields=YOKED_ROW_FIELDS,
        expected_rows=480,
        expected_tasks=40,
    )
    require(set(next(iter(online_groups.values()))) == set(next(iter(yoked_groups.values()))), "panel source-task sets differ")
    online_keys = {
        (method, operator)
        for method in ONLINE_METHODS
        for operator in ONLINE_OPERATORS[1:]
    }
    yoked_keys = {
        (method, operator)
        for method in YOKED_METHODS
        for operator in YOKED_OPERATORS[1:]
    }
    online_effects = source_effect_map(
        online_payload, expected_count=168, expected_keys=online_keys
    )
    yoked_effects = source_effect_map(
        yoked_payload, expected_count=112, expected_keys=yoked_keys
    )
    sensitivity_groups, sensitivity_effects = sensitivity_map(
        read_json(ONLINE_SENSITIVITY),
        online_sha256=online_sha256,
        online_payload=online_payload,
        online_groups=online_groups,
    )
    expected = expected_rows(
        online_groups,
        online_effects,
        sensitivity_groups,
        sensitivity_effects,
        yoked_groups,
        yoked_effects,
    )

    payload_path = STEM.with_suffix(".json")
    csv_path = STEM.with_suffix(".csv")
    svg_path = STEM.with_suffix(".svg")
    sidecar_path = STEM.with_suffix(".svg.data.json")
    receipt_path = STEM.with_suffix(".receipt.json")
    payload, sidecar, receipt = read_json(payload_path), read_json(sidecar_path), read_json(receipt_path)
    observed_rows = payload.get("rows")
    require(isinstance(observed_rows, list) and len(observed_rows) == len(expected), "payload row dimension")
    for index, (observed, wanted) in enumerate(zip(observed_rows, expected, strict=True)):
        require(isinstance(observed, Mapping), f"payload row {index} invalid")
        compare_row(observed, wanted, context=f"payload row {index}")
    changed = [row for row in expected if row["panel"] == "online_natural" and row["n38_qualitative_changed"]]
    leave_audit = payload.get("leave_two_unit_qualitative_audit")
    require(
        payload.get("artifact_type") == "experiment12_deployment_interaction_figure"
        and payload.get("provider_calls_made") == 0
        and payload.get("statistical_unit") == "source_task"
        and payload.get("designs_never_pooled") is True
        and isinstance(leave_audit, Mapping)
        and leave_audit.get("n40_effects_displayed") == 21
        and leave_audit.get("n38_effects_compared") == 21
        and leave_audit.get("changed_displayed_effects") == len(changed),
        "payload scientific contract",
    )
    axis = payload.get("axis")
    half = axis_half(expected)
    require(isinstance(axis, Mapping), "axis missing")
    same_number(axis.get("minimum"), -half, context="axis minimum")
    same_number(axis.get("maximum"), half, context="axis maximum")

    require(sidecar.get("width") == 1480 and sidecar.get("height") == 1525, "sidecar dimensions")
    require(sidecar.get("designs_never_pooled") is True and sidecar.get("rows") == observed_rows, "sidecar data mismatch")
    require(sidecar.get("source_hashes") == {
        "online_analysis_sha256": online_sha256,
        "online_sensitivity_sha256": sensitivity_sha256,
        "yoked_analysis_sha256": YOKED_ANALYSIS_SHA256,
    }, "sidecar source hashes")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == len(expected), "CSV row dimension")
    for index, (csv_row, json_row) in enumerate(zip(csv_rows, observed_rows, strict=True)):
        require(set(csv_row) == set(json_row), f"CSV columns {index}")
        for field, value in json_row.items():
            expected_text = "" if value is None else str(value)
            require(csv_row[field] == expected_text, f"CSV value {index}/{field}")

    svg_markup = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(svg_markup)
    require(root.attrib.get("width") == "1480" and root.attrib.get("height") == "1525", "SVG dimensions")
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    effects = [node for node in root.findall(".//svg:circle", namespace) if node.attrib.get("class") == "effect"]
    ci_lines = [node for node in root.findall(".//svg:line", namespace) if node.attrib.get("class") == "ci"]
    require(len(effects) == 29 and len(ci_lines) == 87, "SVG plotted mark count")
    font_sizes = [float(value) for value in re.findall(r"font-size:\s*([0-9.]+)px", svg_markup)]
    require(font_sizes and min(font_sizes) * 7 * 72 / 1480 >= 7, "SVG text below 7 pt at 7-inch width")
    text_content = " ".join(text.strip() for text in root.itertext() if text.strip())
    for phrase in (
        "Deployment effects depend on observation × state action",
        "Natural online deployment (primary ecological result)",
        "Aggressive checkpoint-1 yoked sensitivity",
        "not a replication",
        "estimates are never pooled with Panel A",
        "tasks 814 and 989",
        "adaptive-analysis-leave-two-units.json",
        "no multiplicity correction",
    ):
        require(phrase in text_content, f"SVG phrase missing: {phrase}")
    if changed:
        require(f"changes {len(changed)} displayed sign/CI conclusion" in text_content, "SVG sensitivity change count")
    else:
        require("changes no displayed effect sign or CI-excludes-zero conclusion" in text_content, "SVG stability caption")

    expected_outputs = {
        relative(payload_path): sha_file(payload_path),
        relative(csv_path): sha_file(csv_path),
        relative(svg_path): sha_file(svg_path),
        relative(sidecar_path): sha_file(sidecar_path),
    }
    require(
        receipt.get("artifact_type") == "experiment12_deployment_interaction_receipt"
        and receipt.get("provider_calls_made") == 0
        and receipt.get("builder_path") == relative(BUILDER)
        and receipt.get("builder_sha256") == sha_file(BUILDER)
        and receipt.get("code_tree_sha256") == EXPECTED_CODE_HASH
        and receipt.get("designs_never_pooled") is True
        and receipt.get("outputs") == expected_outputs
        and receipt.get("leave_two_unit_qualitative_audit") == leave_audit,
        "receipt contract",
    )
    inputs = receipt.get("inputs")
    require(isinstance(inputs, Mapping), "receipt inputs missing")
    for path, expected_hash in (
        (ONLINE_ANALYSIS, online_sha256),
        (ONLINE_SENSITIVITY, sensitivity_sha256),
        (YOKED_ANALYSIS, YOKED_ANALYSIS_SHA256),
    ):
        require(inputs.get(relative(path)) == expected_hash, f"receipt input hash: {path.name}")

    return {
        "status": "pass",
        "provider_calls_made": 0,
        "code_tree_sha256": EXPECTED_CODE_HASH,
        "online_raw_rows_reconstructed": 1120,
        "yoked_raw_rows_reconstructed": 480,
        "n40_effects_recomputed": 29,
        "n38_effects_recomputed": 21,
        "n38_qualitative_changes": len(changed),
        "figure_marks_verified": 29,
        "minimum_print_text_pt_at_7in": round(min(font_sizes) * 7 * 72 / 1480, 3),
        "receipt_sha256": sha_file(receipt_path),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--online-analysis-sha256", required=True)
    result.add_argument("--online-sensitivity-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        print(
            json.dumps(
                audit(args.online_analysis_sha256, args.online_sensitivity_sha256),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        AuditFailure,
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
