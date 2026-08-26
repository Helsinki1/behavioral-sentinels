#!/usr/bin/env python3
"""Build one provider-free paper figure for deployment operator effects.

The primary panel uses the complete strict online adaptive analysis.  The
second panel is a separate, aggressive checkpoint-1 yoked sensitivity.  Both
show task-success change relative to the same method's monitored/no-state-
action control.  The cumulative n=38 affected-source-unit sensitivity is used
only to qualify the n=40 online panel; it never replaces or pools its rows.

This generated builder is intentionally fail-closed.  It requires externally
recorded SHA256 digests for the staged online analysis and its leave-two-unit
sensitivity, verifies frozen run provenance and exact treatment dimensions,
and refuses to write anything until all inputs validate.  It makes no provider
calls and does not import either source analyzer.
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
import re
import sys
import tempfile
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


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
OUTPUT_STEM = GENERATED / "deployment-interaction-confirmatory-v1"

MODEL = "gpt-5.6-luna"
BENCHMARK = "evolving_intent_gsm8k"
CONTROL_OPERATOR = "none"
CONFIDENCE = 0.95
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 12_012
ONLINE_TASKS = 40
SENSITIVITY_TASKS = 38
YOKED_TASKS = 40

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
PLOT_OPERATORS = {
    "online_natural": ONLINE_OPERATORS[1:],
    "yoked_checkpoint1": YOKED_OPERATORS[1:],
}

METHOD_LABELS = {
    "turn_clock": "Turn clock",
    "context_use": "Context use",
    "active_recompute": "Active recompute",
    "frozen_probe:recompute": "Passive recompute",
    "frozen_quiz": "Passive quiz",
    "trace_judge": "Trace judge",
    "trace_rules": "Trace rules",
}
OPERATOR_LABELS = {
    "lossy_compaction": "Compact",
    "public_state_reground": "Reground",
    "good_bad_watch_feedback": "Feedback",
}
OPERATOR_COLORS = {
    "lossy_compaction": "#C55A11",
    "public_state_reground": "#007A78",
    "good_bad_watch_feedback": "#7B2CBF",
}
CLASS_COLORS = {
    "baseline": "#64748B",
    "active": "#D55E00",
    "passive": "#0072B2",
}

ONLINE_TOP_FIELDS = {
    "adaptive_analysis_version",
    "artifact_type",
    "source_run_id",
    "source_manifest_sha256",
    "source_pair_manifest_sha256",
    "deployment_mode",
    "deployment_policy",
    "per_task_action_cap",
    "statistical_unit",
    "comparison_semantics",
    "resource_semantics",
    "rows",
    "metric_summaries",
    "operator_effects",
}
YOKED_TOP_FIELDS = {
    "two_pass_analysis_version",
    "artifact_type",
    "source_run_id",
    "source_manifest_sha256",
    "source_pair_manifest_sha256",
    "source_schedule_sha256",
    "deployment_mode",
    "estimand",
    "statistical_unit",
    "comparison_semantics",
    "resource_semantics",
    "rows",
    "metric_summaries",
    "operator_effects",
    "method_effects",
    "validation",
}
SENSITIVITY_TOP_FIELDS = {
    "artifact_type",
    "sensitivity_version",
    "source_run_id",
    "source_analysis_path",
    "source_analysis_sha256",
    "exclusion_reason",
    "excluded_source_units",
    "exclusion_scope",
    "treatments",
    "excluded_rows_per_treatment",
    "excluded_rows",
    "remaining_rows",
    "remaining_source_tasks_per_treatment",
    "balanced_paired_design_after_exclusion",
    "excluded_cell_ids",
    "rows",
    "metric_summaries",
    "operator_effects",
}
ONLINE_ROW_FIELDS = {
    "cell_id",
    "model",
    "benchmark",
    "task_id",
    "replicate_id",
    "unit_id",
    "method",
    "observation_class",
    "operator",
    "deployment_mode",
    "success",
    "observations",
    "threshold_firings",
    "selected_actions",
    "applied_interventions",
    "task_tokens",
    "observer_tokens",
    "total_tokens",
    "latency_ms",
    "actual_cost_usd",
}
YOKED_ROW_FIELDS = {
    "cell_id",
    "model",
    "benchmark",
    "task_id",
    "replicate_id",
    "unit_id",
    "observation_class",
    "method",
    "operator",
    "deployment_mode",
    "estimand",
    "success",
    "outcome_source",
    "observations",
    "scheduled_actions",
    "action_rate",
    "acted_on_task",
    "applied_interventions",
    "task_tokens",
    "observer_tokens",
    "total_tokens",
    "latency_ms",
    "actual_cost_usd",
    "reported_cost_usd",
    "estimated_cost_usd",
    "upper_bound_cost_usd",
    "failed_retry_attempts",
}
EXPECTED_AFFECTED_UNITS = (
    "extracted-gsm8k-test-814::t7/r0",
    "extracted-gsm8k-test-989::t7/r0",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_TOLERANCE = 1e-12


class InteractionBuildError(RuntimeError):
    """A required source, dimension, or scientific invariant changed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise InteractionBuildError(message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: str, *, context: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value), f"{context} is not SHA256")
    return value


def read_json(path: Path) -> Mapping[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"JSON is not an object: {path}")
    return value


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSONL: {path}")
    result: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, Mapping), f"JSONL row is not an object: {path}:{line_number}")
        result.append(value)
    return result


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "refusing to write empty interaction CSV")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), "interaction CSV columns differ")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def observation_class(method: str) -> str:
    if method == "active_recompute":
        return "active"
    if method in {"turn_clock", "context_use"}:
        return "baseline"
    if method in {"frozen_probe:recompute", "frozen_quiz"}:
        return "passive-behavioral"
    if method in {"trace_judge", "trace_rules"}:
        return "passive-observational"
    raise InteractionBuildError(f"unknown method: {method}")


def paper_class(method: str) -> str:
    observed = observation_class(method)
    if observed.startswith("passive-"):
        return "passive"
    return observed


def nonnegative(value: Any, *, context: str, integer: bool = False) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"non-numeric {context}")
    number = float(value)
    require(math.isfinite(number) and number >= 0, f"invalid {context}")
    if integer:
        require(isinstance(value, int), f"non-integer {context}")
    return number


def close(left: Any, right: Any, *, context: str) -> None:
    require(
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12),
        f"numeric mismatch: {context}",
    )


def verify_static_file(path: Path, expected_sha256: str, *, context: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing or linked {context}")
    require(sha_file(path) == expected_sha256, f"{context} hash changed")


def validate_row(
    row: Mapping[str, Any], *, source: str, fields: set[str], index: int
) -> None:
    require(set(row) == fields, f"{source} row {index} schema changed")
    for key in (
        "cell_id",
        "model",
        "benchmark",
        "task_id",
        "unit_id",
        "method",
        "observation_class",
        "operator",
        "deployment_mode",
    ):
        require(isinstance(row[key], str) and row[key], f"{source} row {index} invalid {key}")
    require(row["model"] == MODEL and row["benchmark"] == BENCHMARK, f"{source} slice changed")
    require(row["replicate_id"] == 0 and not isinstance(row["replicate_id"], bool), f"{source} replicate changed")
    require(row["unit_id"] == f"{row['task_id']}/r0", f"{source} unit identity changed")
    require(row["observation_class"] == observation_class(str(row["method"])), f"{source} class changed")
    require(isinstance(row["success"], bool), f"{source} success is not boolean")
    require(nonnegative(row["observations"], context=f"{source} observations", integer=True) > 0, f"{source} has no observations")
    for key in ("task_tokens", "observer_tokens", "total_tokens", "latency_ms"):
        nonnegative(row[key], context=f"{source} {key}", integer=True)
    nonnegative(row["actual_cost_usd"], context=f"{source} cost")
    require(row["total_tokens"] == row["task_tokens"] + row["observer_tokens"], f"{source} token accounting changed")
    if source == "online":
        require(row["deployment_mode"] == "online_adaptive", "online deployment mode changed")
        for key in ("threshold_firings", "selected_actions", "applied_interventions"):
            nonnegative(row[key], context=f"online {key}", integer=True)
        require(row["selected_actions"] <= 1, "online action cap exceeded")
        require(row["applied_interventions"] == row["selected_actions"], "online action accounting changed")
    else:
        require(row["deployment_mode"] == "two_pass_frozen", "yoked deployment mode changed")
        require(row["estimand"] == "yoked_anchor", "yoked estimand changed")
        for key in (
            "scheduled_actions",
            "acted_on_task",
            "applied_interventions",
            "failed_retry_attempts",
        ):
            nonnegative(row[key], context=f"yoked {key}", integer=True)
        nonnegative(row["action_rate"], context="yoked action rate")
        for key in ("reported_cost_usd", "estimated_cost_usd", "upper_bound_cost_usd"):
            nonnegative(row[key], context=f"yoked {key}")
        require(row["scheduled_actions"] <= row["observations"], "yoked schedule exceeds observations")
        close(row["action_rate"], row["scheduled_actions"] / row["observations"], context="yoked action rate")
        require(row["acted_on_task"] == int(row["scheduled_actions"] > 0), "yoked acted flag changed")
        require(row["applied_interventions"] == row["scheduled_actions"], "yoked intervention count changed")


def group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    methods: Sequence[str],
    operators: Sequence[str],
    fields: set[str],
    tasks: int,
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    expected_methods, expected_operators = set(methods), set(operators)
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    cells: set[str] = set()
    for index, row in enumerate(rows):
        validate_row(row, source=source, fields=fields, index=index)
        method, operator = str(row["method"]), str(row["operator"])
        require(method in expected_methods and operator in expected_operators, f"{source} treatment changed")
        require(row["cell_id"] not in cells, f"{source} duplicate cell")
        cells.add(str(row["cell_id"]))
        units = grouped.setdefault((method, operator), {})
        require(row["unit_id"] not in units, f"{source} duplicate treatment unit")
        units[str(row["unit_id"])] = row
    expected_product = {(method, operator) for method in methods for operator in operators}
    require(set(grouped) == expected_product, f"{source} is not the exact method/operator product")
    reference = set(next(iter(grouped.values())))
    require(len(reference) == tasks, f"{source} source-task denominator changed")
    require(all(set(units) == reference for units in grouped.values()), f"{source} treatments are not paired")
    return grouped


def verify_pair_coverage(
    *, run_id: str, rows: Sequence[Mapping[str, Any]], expected_sha256: str
) -> None:
    pairs_path = ARTIFACTS / run_id / "pairs.jsonl"
    verify_static_file(pairs_path, expected_sha256, context=f"{run_id} pairs")
    pairs = read_jsonl(pairs_path)
    require(len(pairs) == len(rows), f"{run_id} pair dimension changed")
    declared = {str(row.get("cell_id")) for row in pairs}
    observed = {str(row["cell_id"]) for row in rows}
    require(declared == observed, f"{run_id} analysis does not cover declared cells")


def success_source_maps(
    payload: Mapping[str, Any],
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    *,
    source: str,
    expected_summary_count: int,
    expected_effect_count: int,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    summaries = payload.get("metric_summaries")
    effects = payload.get("operator_effects")
    require(isinstance(summaries, list) and len(summaries) == expected_summary_count, f"{source} summary dimension changed")
    require(isinstance(effects, list) and len(effects) == expected_effect_count, f"{source} effect dimension changed")
    success_summary_rows = [
        row
        for row in summaries
        if isinstance(row, Mapping) and row.get("metric") == "success"
    ]
    success_effect_rows = [
        row
        for row in effects
        if isinstance(row, Mapping) and row.get("metric") == "success"
    ]
    success_summaries = {
        (str(row["method"]), str(row["operator"])): row
        for row in success_summary_rows
    }
    success_effects = {
        (str(row["method"]), str(row["operator"])): row
        for row in success_effect_rows
    }
    require(
        len(success_summary_rows) == len(grouped)
        and set(success_summaries) == set(grouped),
        f"{source} success summary coverage changed",
    )
    expected_effect_keys = {
        (method, operator)
        for method, operator in grouped
        if operator != CONTROL_OPERATOR
    }
    require(
        len(success_effect_rows) == len(expected_effect_keys)
        and set(success_effects) == expected_effect_keys,
        f"{source} success effect coverage changed",
    )
    for key, units in grouped.items():
        row = success_summaries[key]
        values = [float(unit["success"]) for unit in units.values()]
        require(row.get("n_tasks") == len(values), f"{source} success summary n changed")
        close(row.get("mean"), fmean(values), context=f"{source} summary {key}")
    for key, row in success_effects.items():
        method, operator = key
        control = grouped[(method, CONTROL_OPERATOR)]
        treated = grouped[(method, operator)]
        units = sorted(control)
        control_values = [float(control[unit]["success"]) for unit in units]
        treated_values = [float(treated[unit]["success"]) for unit in units]
        differences = [value - baseline for baseline, value in zip(control_values, treated_values, strict=True)]
        require(
            row.get("control_operator") == CONTROL_OPERATOR
            and row.get("effect_definition") == "operator_minus_none"
            and row.get("bootstrap_unit") == "paired_source_task"
            and row.get("confidence") == CONFIDENCE
            and row.get("bootstrap_iterations") == BOOTSTRAP_ITERATIONS
            and row.get("bootstrap_seed") == BOOTSTRAP_SEED
            and row.get("n_tasks") == len(units),
            f"{source} effect contract changed: {key}",
        )
        close(row.get("control_mean"), fmean(control_values), context=f"{source} control {key}")
        close(row.get("operator_mean"), fmean(treated_values), context=f"{source} treated {key}")
        close(row.get("effect"), fmean(differences), context=f"{source} effect {key}")
        low, high = row.get("ci_low"), row.get("ci_high")
        require(
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and not isinstance(low, bool)
            and not isinstance(high, bool)
            and -1 <= float(low) <= float(high) <= 1,
            f"{source} effect interval invalid: {key}",
        )
    return success_summaries, success_effects


def verify_online(
    path: Path, expected_sha256: str
) -> tuple[
    Mapping[str, Any],
    dict[tuple[str, str], dict[str, Mapping[str, Any]]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    require(sha_file(path) == expected_sha256, "staged online analysis hash mismatch")
    payload = read_json(path)
    require(set(payload) == ONLINE_TOP_FIELDS, "online top-level schema changed")
    require(
        payload.get("adaptive_analysis_version") == 1
        and payload.get("artifact_type") == "online_adaptive_deployment_analysis"
        and payload.get("source_run_id") == ONLINE_RUN
        and payload.get("source_manifest_sha256") == ONLINE_MANIFEST_SHA256
        and payload.get("source_pair_manifest_sha256") == ONLINE_PAIRS_SHA256
        and payload.get("deployment_mode") == "online_adaptive"
        and payload.get("deployment_policy") == "natural_threshold_per_task_cap"
        and payload.get("per_task_action_cap") == 1
        and payload.get("statistical_unit") == "source_task",
        "online analysis contract changed",
    )
    verify_static_file(
        ARTIFACTS / ONLINE_RUN / "manifest.json",
        ONLINE_MANIFEST_SHA256,
        context="online manifest",
    )
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 1_120, "online analysis must contain exactly 1120 rows")
    grouped = group_rows(
        rows,
        source="online",
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        fields=ONLINE_ROW_FIELDS,
        tasks=ONLINE_TASKS,
    )
    verify_pair_coverage(run_id=ONLINE_RUN, rows=rows, expected_sha256=ONLINE_PAIRS_SHA256)
    _summaries, effects = success_source_maps(
        payload,
        grouped,
        source="online",
        expected_summary_count=224,
        expected_effect_count=168,
    )
    return payload, grouped, effects


def verify_yoked(
    path: Path,
) -> tuple[
    Mapping[str, Any],
    dict[tuple[str, str], dict[str, Mapping[str, Any]]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    require(sha_file(path) == YOKED_ANALYSIS_SHA256, "yoked analysis hash mismatch")
    payload = read_json(path)
    require(set(payload) == YOKED_TOP_FIELDS, "yoked top-level schema changed")
    validation = payload.get("validation")
    require(
        payload.get("two_pass_analysis_version") == 1
        and payload.get("artifact_type") == "two_pass_deployment_analysis"
        and payload.get("source_run_id") == YOKED_RUN
        and payload.get("source_manifest_sha256") == YOKED_MANIFEST_SHA256
        and payload.get("source_pair_manifest_sha256") == YOKED_PAIRS_SHA256
        and payload.get("source_schedule_sha256") == YOKED_SCHEDULE_SHA256
        and payload.get("deployment_mode") == "two_pass_frozen"
        and payload.get("estimand") == "yoked_anchor"
        and payload.get("statistical_unit") == "source_task"
        and isinstance(validation, Mapping)
        and validation.get("primary_ready") is True
        and validation.get("expected_cells") == 480
        and validation.get("valid_outputs") == 480
        and validation.get("valid_jobs") == 480
        and validation.get("valid_event_logs") == 480,
        "yoked analysis contract changed",
    )
    verify_static_file(
        ARTIFACTS / YOKED_RUN / "manifest.json",
        YOKED_MANIFEST_SHA256,
        context="yoked manifest",
    )
    schedule_path = ARTIFACTS / YOKED_RUN / "results" / "deployment_schedule.json"
    verify_static_file(schedule_path, YOKED_SCHEDULE_SHA256, context="yoked schedule")
    schedule = read_json(schedule_path)
    groups = schedule.get("groups")
    require(isinstance(groups, list) and len(groups) == 160, "yoked schedule dimension changed")
    for group in groups:
        actions = group.get("actions") if isinstance(group, Mapping) else None
        require(
            isinstance(actions, list)
            and len(actions) == 1
            and actions[0].get("checkpoint") == 1
            and actions[0].get("trigger_method") == "active_recompute",
            "yoked schedule is not checkpoint-1 active-anchored",
        )
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 480, "yoked analysis must contain exactly 480 rows")
    grouped = group_rows(
        rows,
        source="yoked",
        methods=YOKED_METHODS,
        operators=YOKED_OPERATORS,
        fields=YOKED_ROW_FIELDS,
        tasks=YOKED_TASKS,
    )
    verify_pair_coverage(run_id=YOKED_RUN, rows=rows, expected_sha256=YOKED_PAIRS_SHA256)
    method_effects = payload.get("method_effects")
    require(isinstance(method_effects, list) and len(method_effects) == 252, "yoked method-effect dimension changed")
    _summaries, effects = success_source_maps(
        payload,
        grouped,
        source="yoked",
        expected_summary_count=168,
        expected_effect_count=112,
    )
    return payload, grouped, effects


def sign_label(value: float) -> str:
    if value > ZERO_TOLERANCE:
        return "positive"
    if value < -ZERO_TOLERANCE:
        return "negative"
    return "zero"


def interval_label(low: float, high: float) -> str:
    if low > ZERO_TOLERANCE:
        return "positive_excludes_zero"
    if high < -ZERO_TOLERANCE:
        return "negative_excludes_zero"
    return "includes_zero"


def verify_sensitivity(
    path: Path,
    *,
    expected_sha256: str,
    online_sha256: str,
    online_payload: Mapping[str, Any],
    online_grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    online_effects: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    require(sha_file(path) == expected_sha256, "leave-two-unit sensitivity hash mismatch")
    payload = read_json(path)
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
        and tuple(payload.get("excluded_source_units", ()))
        == EXPECTED_AFFECTED_UNITS
        and payload.get("treatments") == 28
        and payload.get("excluded_rows_per_treatment") == 2
        and payload.get("excluded_rows") == 56
        and payload.get("remaining_rows") == 1_064
        and payload.get("remaining_source_tasks_per_treatment")
        == SENSITIVITY_TASKS
        and payload.get("balanced_paired_design_after_exclusion") is True,
        "leave-two-unit sensitivity contract changed",
    )
    rows = payload.get("rows")
    require(
        isinstance(rows, list) and len(rows) == 1_064,
        "sensitivity must contain exactly 1064 retained rows",
    )
    grouped = group_rows(
        rows,
        source="online",
        methods=ONLINE_METHODS,
        operators=ONLINE_OPERATORS,
        fields=ONLINE_ROW_FIELDS,
        tasks=SENSITIVITY_TASKS,
    )
    full_rows = online_payload.get("rows")
    require(isinstance(full_rows, list), "online source rows missing")
    full_by_cell = {str(row["cell_id"]): row for row in full_rows}
    retained_by_cell = {str(row["cell_id"]): row for row in rows}
    excluded_by_cell = {
        str(row["cell_id"]): row
        for row in full_rows
        if str(row["unit_id"]) in EXPECTED_AFFECTED_UNITS
    }
    excluded_cell_ids = payload.get("excluded_cell_ids")
    require(
        len(full_by_cell) == 1_120
        and len(retained_by_cell) == 1_064
        and len(excluded_by_cell) == 56
        and isinstance(excluded_cell_ids, list)
        and excluded_cell_ids == sorted(excluded_by_cell)
        and set(retained_by_cell) == set(full_by_cell) - set(excluded_by_cell)
        and all(retained_by_cell[cell] == full_by_cell[cell] for cell in retained_by_cell),
        "sensitivity rows are not the exact leave-two subset of the n40 analysis",
    )
    full_units = set(next(iter(online_grouped.values())))
    retained_units = set(next(iter(grouped.values())))
    require(
        set(EXPECTED_AFFECTED_UNITS) <= full_units
        and retained_units == full_units - set(EXPECTED_AFFECTED_UNITS),
        "sensitivity paired source-task set changed",
    )
    _summaries, success = success_source_maps(
        payload,
        grouped,
        source="sensitivity",
        expected_summary_count=224,
        expected_effect_count=168,
    )
    require(set(success) == set(online_effects), "n40/n38 success-effect identities differ")
    require(
        all(row.get("n_tasks") == SENSITIVITY_TASKS for row in success.values()),
        "sensitivity success-effect denominator changed",
    )
    return success


def effect_rows(
    *,
    online_grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    online_effects: Mapping[tuple[str, str], Mapping[str, Any]],
    sensitivity: Mapping[tuple[str, str], Mapping[str, Any]],
    yoked_effects: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for panel, methods, operators, effects in (
        ("online_natural", PLOT_METHODS, ONLINE_OPERATORS[1:], online_effects),
        ("yoked_checkpoint1", tuple(method for method in PLOT_METHODS if method in YOKED_METHODS), YOKED_OPERATORS[1:], yoked_effects),
    ):
        for method in methods:
            for operator in operators:
                source = effects[(method, operator)]
                sensitivity_row = sensitivity[(method, operator)] if panel == "online_natural" else None
                if panel == "online_natural":
                    treated = online_grouped[(method, operator)]
                    action_task_rate = fmean(float(row["selected_actions"] > 0) for row in treated.values())
                    action_checkpoint_rate = fmean(
                        float(row["selected_actions"]) / float(row["observations"])
                        for row in treated.values()
                    )
                    sign_changed = sign_label(float(source["effect"])) != sign_label(
                        float(sensitivity_row["effect"])
                    )
                    inference_changed = interval_label(
                        float(source["ci_low"]), float(source["ci_high"])
                    ) != interval_label(
                        float(sensitivity_row["ci_low"]),
                        float(sensitivity_row["ci_high"]),
                    )
                    sensitivity_n38_effect = float(sensitivity_row["effect"])
                    sensitivity_n38_ci_low = float(sensitivity_row["ci_low"])
                    sensitivity_n38_ci_high = float(sensitivity_row["ci_high"])
                else:
                    action_task_rate = None
                    action_checkpoint_rate = None
                    sign_changed = False
                    inference_changed = False
                    sensitivity_n38_effect = None
                    sensitivity_n38_ci_low = None
                    sensitivity_n38_ci_high = None
                result.append(
                    {
                        "panel": panel,
                        "analysis_role": (
                            "primary_online_natural_policy"
                            if panel == "online_natural"
                            else "controlled_aggressive_checkpoint1_yoked_sensitivity"
                        ),
                        "paper_class": paper_class(method),
                        "method": method,
                        "operator": operator,
                        "control_operator": CONTROL_OPERATOR,
                        "n_tasks": int(source["n_tasks"]),
                        "control_success": float(source["control_mean"]),
                        "operator_success": float(source["operator_mean"]),
                        "effect": float(source["effect"]),
                        "ci_low": float(source["ci_low"]),
                        "ci_high": float(source["ci_high"]),
                        "confidence": float(source["confidence"]),
                        "bootstrap_iterations": int(source["bootstrap_iterations"]),
                        "bootstrap_seed": int(source["bootstrap_seed"]),
                        "bootstrap_unit": str(source["bootstrap_unit"]),
                        "natural_action_task_rate": action_task_rate,
                        "natural_action_checkpoint_rate": action_checkpoint_rate,
                        "n38_point_sign_changed": sign_changed,
                        "n38_ci_relation_changed": inference_changed,
                        "n38_qualitative_changed": sign_changed or inference_changed,
                        "sensitivity_n38_effect": sensitivity_n38_effect,
                        "sensitivity_n38_ci_low": sensitivity_n38_ci_low,
                        "sensitivity_n38_ci_high": sensitivity_n38_ci_high,
                    }
                )
    require(len(result) == 29, "displayed interaction dimension changed")
    return result


def axis_half_span(rows: Sequence[Mapping[str, Any]]) -> float:
    maximum = max(
        abs(float(row[field]))
        for row in rows
        for field in ("effect", "ci_low", "ci_high")
    )
    require(maximum <= 1 + 1e-12, "success effect exceeds logical bounds")
    return min(1.0, max(0.10, math.ceil(maximum * 20 - 1e-12) / 20))


def percent(value: float) -> str:
    return f"{value * 100:.1f}".rstrip("0").rstrip(".")


def tick_label(value: float) -> str:
    number = value * 100
    return f"{number:+.1f}".replace(".0", "")


def build_svg(rows: Sequence[Mapping[str, Any]], *, half_span: float) -> str:
    width, height = 1480, 1525
    online = [row for row in rows if row["panel"] == "online_natural"]
    yoked = [row for row in rows if row["panel"] == "yoked_checkpoint1"]
    require(len(online) == 21 and len(yoked) == 8, "SVG panel row counts changed")
    changed = [row for row in online if row["n38_qualitative_changed"]]
    plot_left, plot_right = 430.0, 1070.0

    def x(value: float) -> float:
        return plot_left + (value + half_span) / (2 * half_span) * (plot_right - plot_left)

    body: list[str] = ['<rect width="100%" height="100%" fill="#FFFFFF"/>']
    body.append('<text x="30" y="46" class="title">Deployment effects depend on observation × state action</text>')
    body.append('<text x="30" y="78" class="subtitle">Success change versus each method’s monitored/no-state-action control; positive means the state action helped</text>')
    legend_x = 30
    for operator in ONLINE_OPERATORS[1:]:
        body.append(f'<circle cx="{legend_x + 8}" cy="108" r="7" fill="{OPERATOR_COLORS[operator]}"/>')
        body.append(f'<text x="{legend_x + 24}" y="115" class="legend">{escape(OPERATOR_LABELS[operator])}</text>')
        legend_x += 155
    class_x = 560
    for class_name, label in (("baseline", "baseline"), ("active", "active carry"), ("passive", "passive zero-carry")):
        body.append(f'<rect x="{class_x}" y="100" width="15" height="15" rx="2" fill="{CLASS_COLORS[class_name]}"/>')
        body.append(f'<text x="{class_x + 22}" y="115" class="legend">{label}</text>')
        class_x += 205

    def grid(top: float, bottom: float, tick_y: float, axis_y: float) -> None:
        for index in range(5):
            value = -half_span + index * half_span / 2
            xpos = x(value)
            css = "zero" if index == 2 else "grid"
            body.append(f'<line x1="{xpos:.1f}" y1="{top:.1f}" x2="{xpos:.1f}" y2="{bottom:.1f}" class="{css}"/>')
            body.append(f'<text x="{xpos:.1f}" y="{tick_y:.1f}" text-anchor="middle" class="tick">{tick_label(value)}</text>')
        body.append(f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{axis_y:.1f}" text-anchor="middle" class="axis">change in task success (percentage points)</text>')

    def panel_rows(
        panel_rows_value: Sequence[Mapping[str, Any]],
        *,
        y_start: float,
        row_h: float,
        operator_count: int,
        action_annotations: bool,
    ) -> None:
        row_lookup = {(str(row["method"]), str(row["operator"])): row for row in panel_rows_value}
        methods = []
        for row in panel_rows_value:
            if row["method"] not in methods:
                methods.append(str(row["method"]))
        operators = tuple(str(row["operator"]) for row in panel_rows_value[:operator_count])
        row_index = 0
        for method_index, method in enumerate(methods):
            group_y = y_start + row_index * row_h
            group_height = operator_count * row_h - 4
            if method_index % 2 == 0:
                body.append(f'<rect x="30" y="{group_y - 13:.1f}" width="1418" height="{group_height:.1f}" rx="5" fill="#F7F9FC"/>')
            center_y = group_y + (operator_count - 1) * row_h / 2
            class_name = paper_class(method)
            body.append(f'<rect x="30" y="{center_y - 8:.1f}" width="15" height="15" rx="2" fill="{CLASS_COLORS[class_name]}"/>')
            body.append(f'<text x="54" y="{center_y + 7:.1f}" class="method">{escape(METHOD_LABELS[method])}</text>')
            for operator in operators:
                row = row_lookup[(method, operator)]
                y = y_start + row_index * row_h
                effect, low, high = (float(row["effect"]), float(row["ci_low"]), float(row["ci_high"]))
                color = OPERATOR_COLORS[operator]
                flag = " †" if row["n38_qualitative_changed"] else ""
                tooltip = (
                    f"{METHOD_LABELS[method]} · {OPERATOR_LABELS[operator]} · "
                    f"Δ={effect * 100:+.1f} pp, 95% CI [{low * 100:+.1f}, {high * 100:+.1f}], n={row['n_tasks']}"
                )
                body.append(f'<text x="410" y="{y + 7:.1f}" text-anchor="end" class="operator">{escape(OPERATOR_LABELS[operator] + flag)}</text>')
                body.append(f'<line x1="{x(low):.1f}" y1="{y:.1f}" x2="{x(high):.1f}" y2="{y:.1f}" class="ci" style="stroke:{color}"/>')
                body.append(f'<line x1="{x(low):.1f}" y1="{y - 5:.1f}" x2="{x(low):.1f}" y2="{y + 5:.1f}" class="ci" style="stroke:{color}"/>')
                body.append(f'<line x1="{x(high):.1f}" y1="{y - 5:.1f}" x2="{x(high):.1f}" y2="{y + 5:.1f}" class="ci" style="stroke:{color}"/>')
                body.append(f'<circle class="effect" cx="{x(effect):.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#FFFFFF" stroke-width="1.4"><title>{escape(tooltip)}</title></circle>')
                if action_annotations:
                    body.append(
                        f'<text x="1095" y="{y + 7:.1f}" class="action">{percent(float(row["natural_action_task_rate"]))}% tasks · {percent(float(row["natural_action_checkpoint_rate"]))}% checks</text>'
                    )
                row_index += 1

    body.append('<text x="30" y="158" class="panel">A · Natural online deployment (primary ecological result)</text>')
    body.append('<text x="30" y="188" class="note">40 paired tasks per row; effect = state action − the same method’s monitored/no-state-action control.</text>')
    body.append('<text x="30" y="216" class="note">Right labels: tasks acted on / mean observed-checkpoint action rate under each separately deployed natural policy.</text>')
    body.append('<text x="54" y="244" class="header">method</text>')
    body.append('<text x="410" y="244" text-anchor="end" class="header">state action</text>')
    body.append('<text x="1095" y="244" class="header">natural actions</text>')
    grid(251, 892, 920, 947)
    panel_rows(online, y_start=272, row_h=30, operator_count=3, action_annotations=True)

    body.append('<text x="30" y="995" class="panel">B · Aggressive checkpoint-1 yoked sensitivity (controlled; not a replication)</text>')
    body.append('<text x="30" y="1025" class="note">Same 40 source tasks; every cell follows the active anchor’s checkpoint-1 schedule.</text>')
    body.append('<text x="30" y="1053" class="note">Tests operator sensitivity—not natural trigger timing—and has no feedback arm; estimates are never pooled with Panel A.</text>')
    body.append('<text x="54" y="1082" class="header">method</text>')
    body.append('<text x="410" y="1082" text-anchor="end" class="header">state action</text>')
    grid(1091, 1349, 1377, 1404)
    panel_rows(yoked, y_start=1112, row_h=31, operator_count=2, action_annotations=False)

    marker = "† " if changed else ""
    body.append(
        f'<text x="30" y="1439" class="sensitivity">{marker}Cumulative n=38 leave-two-unit sensitivity: tasks 814 and 989; see adaptive-analysis-leave-two-units.json.</text>'
    )
    if changed:
        body.append(f'<text x="30" y="1467" class="sensitivity">It changes {len(changed)} displayed sign/CI conclusion(s); flagged rows are marked.</text>')
    else:
        body.append('<text x="30" y="1467" class="sensitivity">It changes no displayed effect sign or CI-excludes-zero conclusion.</text>')
    body.append('<text x="30" y="1497" class="footnote">95% paired source-task bootstrap CIs; no multiplicity correction. Panels are separate designs and are never pooled.</text>')

    style = """
      text { font-family: 'Liberation Sans'; fill: #243B53; }
      .title { font-size: 36px; font-weight: 750; fill: #1F2933; }
      .subtitle, .legend, .note, .tick, .axis, .operator, .action, .footnote, .sensitivity { font-size: 21px; }
      .subtitle, .note, .tick, .axis, .footnote { fill: #52606D; }
      .panel { font-size: 27px; font-weight: 750; fill: #102A43; }
      .header { font-size: 21px; font-weight: 700; fill: #334E68; }
      .method { font-size: 22px; font-weight: 700; fill: #334E68; }
      .operator { font-weight: 650; }
      .action { fill: #486581; }
      .sensitivity { font-weight: 650; fill: #334E68; }
      .grid { stroke: #D9E2EC; stroke-width: 1; }
      .zero { stroke: #52606D; stroke-width: 2; }
      .ci { stroke-width: 2.4; stroke-linecap: round; }
    """
    description = (
        "Two separate forest-plot panels show paired task-success changes for state "
        "actions relative to the same observation method's monitored no-state-action "
        "control: natural online deployment and an aggressive checkpoint-1 yoked sensitivity."
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        '<title id="title">Deployment effects depend on observation × state action</title>\n'
        f'<desc id="desc">{escape(description)}</desc>\n'
        f'<defs><style>{style}</style></defs>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def build(online_sha256: str, sensitivity_sha256: str) -> dict[str, Any]:
    code_hash = code_tree_hash(ROOT / "experiments12")
    require(code_hash == EXPECTED_CODE_HASH, f"frozen code tree changed: {code_hash}")
    online_sha256 = digest(online_sha256, context="online analysis hash")
    sensitivity_sha256 = digest(sensitivity_sha256, context="online sensitivity hash")
    online_payload, online_grouped, online_effects = verify_online(
        ONLINE_ANALYSIS, online_sha256
    )
    yoked_payload, yoked_grouped, yoked_effects = verify_yoked(YOKED_ANALYSIS)
    require(
        set(next(iter(online_grouped.values())))
        == set(next(iter(yoked_grouped.values()))),
        "online and yoked source-task sets differ",
    )
    sensitivity = verify_sensitivity(
        ONLINE_SENSITIVITY,
        expected_sha256=sensitivity_sha256,
        online_sha256=online_sha256,
        online_payload=online_payload,
        online_grouped=online_grouped,
        online_effects=online_effects,
    )
    rows = effect_rows(
        online_grouped=online_grouped,
        online_effects=online_effects,
        sensitivity=sensitivity,
        yoked_effects=yoked_effects,
    )
    half_span = axis_half_span(rows)
    changed = [row for row in rows if row["panel"] == "online_natural" and row["n38_qualitative_changed"]]
    payload = {
        "schema_version": 1,
        "artifact_type": "experiment12_deployment_interaction_figure",
        "provider_calls_made": 0,
        "statistical_unit": "source_task",
        "effect_definition": "operator success minus the same method's none-operator success on identical source tasks",
        "designs_never_pooled": True,
        "bootstrap": {
            "confidence": CONFIDENCE,
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired_source_task",
            "multiplicity_adjustment": None,
            "source": "strict source analyzers; independently audited from raw analysis rows",
        },
        "sources": {
            "online_analysis": {
                "path": relative(ONLINE_ANALYSIS),
                "sha256": online_sha256,
                "source_run_id": ONLINE_RUN,
                "source_manifest_sha256": online_payload["source_manifest_sha256"],
                "source_pair_manifest_sha256": online_payload["source_pair_manifest_sha256"],
            },
            "online_leave_two_unit_sensitivity": {
                "path": relative(ONLINE_SENSITIVITY),
                "sha256": sensitivity_sha256,
                "excluded_unit_ids": list(EXPECTED_AFFECTED_UNITS),
                "n_tasks": SENSITIVITY_TASKS,
            },
            "yoked_analysis": {
                "path": relative(YOKED_ANALYSIS),
                "sha256": YOKED_ANALYSIS_SHA256,
                "source_run_id": YOKED_RUN,
                "source_manifest_sha256": yoked_payload["source_manifest_sha256"],
                "source_pair_manifest_sha256": yoked_payload["source_pair_manifest_sha256"],
                "source_schedule_sha256": yoked_payload["source_schedule_sha256"],
            },
        },
        "panels": {
            "online_natural": {
                "analysis_role": "primary_online_natural_policy",
                "n_tasks": ONLINE_TASKS,
                "methods": list(PLOT_METHODS),
                "operators": list(ONLINE_OPERATORS[1:]),
                "policy": "natural scalar thresholds; separately deployed trajectories; one-action task cap",
                "action_annotation": "proportion of tasks acted on and mean per-task selected-actions/observations",
                "rows": [row for row in rows if row["panel"] == "online_natural"],
            },
            "yoked_checkpoint1": {
                "analysis_role": "controlled_aggressive_checkpoint1_yoked_sensitivity",
                "n_tasks": YOKED_TASKS,
                "methods": [method for method in PLOT_METHODS if method in YOKED_METHODS],
                "operators": list(YOKED_OPERATORS[1:]),
                "schedule": "one active-anchored checkpoint-1 schedule per source task in every cell",
                "not_natural_trigger_timing": True,
                "not_independent_replication": True,
                "rows": [row for row in rows if row["panel"] == "yoked_checkpoint1"],
            },
        },
        "leave_two_unit_qualitative_audit": {
            "n40_effects_displayed": 21,
            "n38_effects_compared": 21,
            "changed_displayed_effects": len(changed),
            "any_point_sign_changed": any(row["n38_point_sign_changed"] for row in changed),
            "any_ci_relation_changed": any(row["n38_ci_relation_changed"] for row in changed),
            "changed_rows": [
                {
                    "method": row["method"],
                    "operator": row["operator"],
                    "point_sign_changed": row["n38_point_sign_changed"],
                    "ci_relation_changed": row["n38_ci_relation_changed"],
                }
                for row in changed
            ],
        },
        "axis": {
            "minimum": -half_span,
            "maximum": half_span,
            "unit": "task-success proportion",
            "shared_across_panels_for_visual_comparison_only": True,
        },
        "rows": rows,
    }
    svg = build_svg(rows, half_span=half_span)
    sidecar = {
        "schema_version": 1,
        "figure_type": "separate_online_and_yoked_operator_effect_forests",
        "title": "Deployment effects depend on observation × state action",
        "width": 1480,
        "height": 1525,
        "recommended_paper_placement": "full two-column width",
        "target_print_width_inches": 7.0,
        "minimum_text_points_at_target_width": 21 * 7 * 72 / 1480,
        "designs_never_pooled": True,
        "axis": payload["axis"],
        "source_hashes": {
            "online_analysis_sha256": online_sha256,
            "online_sensitivity_sha256": sensitivity_sha256,
            "yoked_analysis_sha256": YOKED_ANALYSIS_SHA256,
        },
        "leave_two_unit_qualitative_audit": payload["leave_two_unit_qualitative_audit"],
        "rows": rows,
    }

    json_path = OUTPUT_STEM.with_suffix(".json")
    csv_path = OUTPUT_STEM.with_suffix(".csv")
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    sidecar_path = OUTPUT_STEM.with_suffix(".svg.data.json")
    receipt_path = OUTPUT_STEM.with_suffix(".receipt.json")
    atomic_json(json_path, payload)
    atomic_csv(csv_path, rows)
    atomic_text(svg_path, svg)
    atomic_json(sidecar_path, sidecar)
    outputs = {
        relative(json_path): sha_file(json_path),
        relative(csv_path): sha_file(csv_path),
        relative(svg_path): sha_file(svg_path),
        relative(sidecar_path): sha_file(sidecar_path),
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "experiment12_deployment_interaction_receipt",
        "provider_calls_made": 0,
        "builder_path": relative(Path(__file__)),
        "builder_sha256": sha_file(Path(__file__)),
        "code_tree_sha256": code_hash,
        "inputs": {
            relative(ONLINE_ANALYSIS): online_sha256,
            relative(ONLINE_SENSITIVITY): sensitivity_sha256,
            relative(YOKED_ANALYSIS): YOKED_ANALYSIS_SHA256,
            relative(ARTIFACTS / ONLINE_RUN / "manifest.json"): ONLINE_MANIFEST_SHA256,
            relative(ARTIFACTS / ONLINE_RUN / "pairs.jsonl"): ONLINE_PAIRS_SHA256,
            relative(ARTIFACTS / YOKED_RUN / "manifest.json"): YOKED_MANIFEST_SHA256,
            relative(ARTIFACTS / YOKED_RUN / "pairs.jsonl"): YOKED_PAIRS_SHA256,
            relative(ARTIFACTS / YOKED_RUN / "results" / "deployment_schedule.json"): YOKED_SCHEDULE_SHA256,
        },
        "dimensions": {
            "online_rows": 1_120,
            "online_tasks": ONLINE_TASKS,
            "online_displayed_effects": 21,
            "sensitivity_tasks": SENSITIVITY_TASKS,
            "sensitivity_displayed_effects_compared": 21,
            "yoked_rows": 480,
            "yoked_tasks": YOKED_TASKS,
            "yoked_displayed_effects": 8,
        },
        "designs_never_pooled": True,
        "leave_two_unit_qualitative_audit": payload["leave_two_unit_qualitative_audit"],
        "outputs": outputs,
    }
    atomic_json(receipt_path, receipt)
    require(code_tree_hash(ROOT / "experiments12") == EXPECTED_CODE_HASH, "frozen code changed during build")
    return {
        "status": "built",
        "provider_calls_made": 0,
        "output_stem": relative(OUTPUT_STEM),
        "receipt_sha256": sha_file(receipt_path),
        "displayed_effects": len(rows),
        "n38_qualitative_changes": len(changed),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--online-analysis-sha256",
        required=True,
        help="externally recorded SHA256 of the complete staged n=40 analysis",
    )
    result.add_argument(
        "--online-sensitivity-sha256",
        required=True,
        help="externally recorded SHA256 of the cumulative n=38 sensitivity artifact",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = build(args.online_analysis_sha256, args.online_sensitivity_sha256)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        InteractionBuildError,
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
