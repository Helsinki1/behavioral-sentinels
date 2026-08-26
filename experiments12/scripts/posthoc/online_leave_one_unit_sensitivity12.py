"""Provider-free cumulative affected-source-unit sensitivity for Experiment 12.

Run this only after ``adaptive_analysis12 extract`` has produced the complete,
strictly validated online analysis.  The two affected source units are derived
from the complete frozen recovery-cell set; no outcome is used to select them.
The script removes both units from every method/operator treatment, recomputes
the frozen task-bootstrap summaries, and writes hash-bound JSON and Markdown
receipts.  It never dispatches provider calls or edits production artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from experiments12.adaptive_analysis12 import (
    ADAPTIVE_ANALYSIS_TYPE,
    ADAPTIVE_ANALYSIS_VERSION,
    summarize_adaptive_outcomes,
)
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    sha256_json,
)


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "experiment12_online_cumulative_affected_unit_sensitivity"
NORMALIZATION_CASE_INDEX_TYPE = "experiment12_adaptive_normalization_case_index"
AFFECTED_CELL_ID = "d52046b6eb74a76ecdc3debc"
RECOVERY_CELL_TREATMENTS = {
    "d52046b6eb74a76ecdc3debc": ("trace_judge", "lossy_compaction"),
    "89df41e0daa1262a43fa5e55": ("trace_judge", "public_state_reground"),
    "786d95760ccdb86713c26936": ("trace_judge", "public_state_reground"),
}
EXPECTED_RECOVERY_TASKS = {
    "d52046b6eb74a76ecdc3debc": "extracted-gsm8k-test-814::t7",
    "89df41e0daa1262a43fa5e55": "extracted-gsm8k-test-814::t7",
    "786d95760ccdb86713c26936": "extracted-gsm8k-test-989::t7",
}
EXPECTED_AFFECTED_UNIT_IDS = tuple(
    sorted({f"{task_id}/r0" for task_id in EXPECTED_RECOVERY_TASKS.values()})
)
EXPECTED_AFFECTED_UNITS = 2
EXPECTED_RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
EXPECTED_MANIFEST_SHA256 = (
    "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
)
EXPECTED_PAIR_MANIFEST_SHA256 = (
    "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
)
EXPECTED_MODEL = "gpt-5.6-luna"
EXPECTED_BENCHMARK = "evolving_intent_gsm8k"
EXPECTED_METHOD_NAMES = (
    "active_recompute",
    "context_use",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
)
EXPECTED_OPERATOR_NAMES = (
    "good_bad_watch_feedback",
    "lossy_compaction",
    "none",
    "public_state_reground",
)
EXPECTED_ROWS = 1_120
EXPECTED_TASKS = 40
EXPECTED_METHODS = 7
EXPECTED_OPERATORS = 4
EXPECTED_TREATMENTS = EXPECTED_METHODS * EXPECTED_OPERATORS
EXPECTED_REMOVED_ROWS = EXPECTED_TREATMENTS * EXPECTED_AFFECTED_UNITS
EXPECTED_FILTERED_ROWS = EXPECTED_ROWS - EXPECTED_REMOVED_ROWS
FILTERED_TASKS = EXPECTED_TASKS - EXPECTED_AFFECTED_UNITS
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_SEED = 12_012
CONFIDENCE = 0.95

SUCCESS_METRIC = "success"
THRESHOLD_METRIC = "threshold_firings"
ACTION_RATE_METRIC = "selected_actions"
SCIENTIFIC_OUTCOME_METRICS = (SUCCESS_METRIC,)
ACTION_POLICY_METRICS = (THRESHOLD_METRIC, ACTION_RATE_METRIC)
KEY_RESOURCE_METRICS = (
    "task_tokens",
    "observer_tokens",
    "total_tokens",
    "latency_ms",
    "actual_cost_usd",
)
COMPARISON_METRICS = (
    *SCIENTIFIC_OUTCOME_METRICS,
    *ACTION_POLICY_METRICS,
    *KEY_RESOURCE_METRICS,
)

EXPECTED_OUTPUT_NAME = "adaptive-analysis-leave-two-units.json"
EXPECTED_MARKDOWN_NAME = "adaptive-analysis-leave-two-units.md"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ZERO_TOLERANCE = 1e-12
_BOUNDED_RATE_ABSOLUTE_SHIFT = 1 / EXPECTED_TASKS
_DESCRIPTIVE_RELATIVE_SHIFT = 0.05

_SUMMARY_KEY = (
    "model",
    "benchmark",
    "observation_class",
    "method",
    "operator",
    "metric",
)
_EFFECT_KEY = (
    "model",
    "benchmark",
    "observation_class",
    "method",
    "operator",
    "control_operator",
    "metric",
)


class SensitivityError(ValueError):
    """Raised when the sensitivity input or design fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _digest(value: str, *, context: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise SensitivityError(f"{context} must be a lowercase SHA256 digest")
    return value


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensitivityError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SensitivityError(f"{context} must be finite")
    return result


def _index(
    rows: Iterable[Mapping[str, Any]],
    key_fields: Sequence[str],
    *,
    context: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SensitivityError(f"{context} row {position} is not an object")
        try:
            key = tuple(row[field] for field in key_fields)
        except KeyError as exc:
            raise SensitivityError(f"{context} row lacks {exc.args[0]}") from exc
        if key in result:
            raise SensitivityError(f"{context} rows duplicate key {key!r}")
        result[key] = row
    if not result:
        raise SensitivityError(f"{context} rows are empty")
    return result


def _sign(value: float) -> str:
    if value > _ZERO_TOLERANCE:
        return "positive"
    if value < -_ZERO_TOLERANCE:
        return "negative"
    return "zero"


def _interval_relation(low: float, high: float) -> str:
    if low > _ZERO_TOLERANCE:
        return "positive_excludes_zero"
    if high < -_ZERO_TOLERANCE:
        return "negative_excludes_zero"
    return "includes_zero"


def _relative_delta(delta: float, reference: float) -> float | None:
    if abs(reference) <= _ZERO_TOLERANCE:
        return None
    return delta / abs(reference)


def _metric_group(metric: str) -> str:
    if metric in SCIENTIFIC_OUTCOME_METRICS:
        return "scientific_outcome"
    if metric in ACTION_POLICY_METRICS:
        return "action_policy"
    if metric in KEY_RESOURCE_METRICS:
        return "resource"
    raise SensitivityError(f"comparison metric has no reporting group: {metric!r}")


def _summary_comparison(
    original: Mapping[str, Any], filtered: Mapping[str, Any]
) -> dict[str, Any]:
    metric = str(original["metric"])
    if metric not in COMPARISON_METRICS or filtered.get("metric") != metric:
        raise SensitivityError("summary comparison metric changed")
    original_mean = _finite(original.get("mean"), context="n40 summary mean")
    filtered_mean = _finite(filtered.get("mean"), context="n38 summary mean")
    delta = filtered_mean - original_mean
    relative = _relative_delta(delta, original_mean)
    zero_transition = (_sign(original_mean) == "zero") != (
        _sign(filtered_mean) == "zero"
    )
    if metric in {SUCCESS_METRIC, ACTION_RATE_METRIC}:
        material = (
            abs(delta) >= _BOUNDED_RATE_ABSOLUTE_SHIFT - _ZERO_TOLERANCE
            or zero_transition
        )
        rule = (
            "absolute mean shift >= 1/40 (0.025), or zero-to-nonzero change"
        )
    else:
        material = (
            zero_transition
            or (
                relative is not None
                and abs(relative) >= _DESCRIPTIVE_RELATIVE_SHIFT
            )
        )
        rule = "absolute relative mean shift >= 5%; zero-to-nonzero also flags"
    identity = {field: original[field] for field in _SUMMARY_KEY}
    return {
        **identity,
        "metric_group": _metric_group(metric),
        "n40": {
            "n_tasks": original["n_tasks"],
            "mean": original_mean,
            "ci_low": _finite(original.get("ci_low"), context="n40 summary CI low"),
            "ci_high": _finite(original.get("ci_high"), context="n40 summary CI high"),
        },
        "n38": {
            "n_tasks": filtered["n_tasks"],
            "mean": filtered_mean,
            "ci_low": _finite(filtered.get("ci_low"), context="n38 summary CI low"),
            "ci_high": _finite(filtered.get("ci_high"), context="n38 summary CI high"),
        },
        "mean_delta_n38_minus_n40": delta,
        "relative_mean_delta": relative,
        "zero_nonzero_changed": zero_transition,
        "material_absolute_shift": material,
        "materiality_rule": rule,
    }


def _effect_comparison(
    original: Mapping[str, Any], filtered: Mapping[str, Any]
) -> dict[str, Any]:
    metric = str(original["metric"])
    if metric not in COMPARISON_METRICS or filtered.get("metric") != metric:
        raise SensitivityError("effect comparison metric changed")
    original_effect = _finite(original.get("effect"), context="n40 effect")
    filtered_effect = _finite(filtered.get("effect"), context="n38 effect")
    original_low = _finite(original.get("ci_low"), context="n40 effect CI low")
    original_high = _finite(original.get("ci_high"), context="n40 effect CI high")
    filtered_low = _finite(filtered.get("ci_low"), context="n38 effect CI low")
    filtered_high = _finite(filtered.get("ci_high"), context="n38 effect CI high")
    original_sign = _sign(original_effect)
    filtered_sign = _sign(filtered_effect)
    original_inference = _interval_relation(original_low, original_high)
    filtered_inference = _interval_relation(filtered_low, filtered_high)
    sign_changed = original_sign != filtered_sign
    inference_changed = original_inference != filtered_inference
    reasons: list[str] = []
    if sign_changed:
        reasons.append("point_effect_sign_changed")
    if inference_changed:
        reasons.append("confidence_interval_relation_to_zero_changed")
    identity = {field: original[field] for field in _EFFECT_KEY}
    return {
        **identity,
        "metric_group": _metric_group(metric),
        "n40": {
            "n_tasks": original["n_tasks"],
            "control_mean": _finite(
                original.get("control_mean"), context="n40 control mean"
            ),
            "operator_mean": _finite(
                original.get("operator_mean"), context="n40 operator mean"
            ),
            "effect": original_effect,
            "ci_low": original_low,
            "ci_high": original_high,
            "point_sign": original_sign,
            "inference": original_inference,
        },
        "n38": {
            "n_tasks": filtered["n_tasks"],
            "control_mean": _finite(
                filtered.get("control_mean"), context="n38 control mean"
            ),
            "operator_mean": _finite(
                filtered.get("operator_mean"), context="n38 operator mean"
            ),
            "effect": filtered_effect,
            "ci_low": filtered_low,
            "ci_high": filtered_high,
            "point_sign": filtered_sign,
            "inference": filtered_inference,
        },
        "effect_delta_n38_minus_n40": filtered_effect - original_effect,
        "point_sign_changed": sign_changed,
        "inference_changed": inference_changed,
        "material_change": bool(reasons),
        "material_reasons": reasons,
        "materiality_rule": (
            "flag any change in point-effect sign or in whether the paired 95% "
            "interval is negative-excluding-zero, includes-zero, or "
            "positive-excluding-zero"
        ),
    }


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_unlinked_components(path: Path, *, context: str) -> None:
    """Reject a symlink at the target or in any existing ancestor."""

    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise SensitivityError(f"{context} must not traverse symlinks")


def _validated_output_paths(
    input_path: Path, output_path: Path, markdown_path: Path
) -> tuple[Path, Path]:
    """Confine writes to the two declared analysis products beside the input."""

    _require_unlinked_components(input_path, context="input analysis")
    _require_unlinked_components(output_path, context="JSON receipt output")
    _require_unlinked_components(markdown_path, context="Markdown receipt output")
    input_parent = input_path.resolve().parent
    if output_path.name != EXPECTED_OUTPUT_NAME:
        raise SensitivityError(
            f"JSON receipt output must be named {EXPECTED_OUTPUT_NAME}"
        )
    if markdown_path.name != EXPECTED_MARKDOWN_NAME:
        raise SensitivityError(
            f"Markdown receipt output must be named {EXPECTED_MARKDOWN_NAME}"
        )
    if (
        output_path.parent.resolve() != input_parent
        or markdown_path.parent.resolve() != input_parent
    ):
        raise SensitivityError(
            "receipt outputs must be written beside the validated source analysis"
        )
    for path, context in (
        (output_path, "JSON receipt output"),
        (markdown_path, "Markdown receipt output"),
    ):
        if path.exists() and not path.is_file():
            raise SensitivityError(f"{context} must be a regular file if it exists")
    if output_path.resolve() == input_path.resolve():
        raise SensitivityError("output must not overwrite the source analysis")
    if markdown_path.resolve() in {input_path.resolve(), output_path.resolve()}:
        raise SensitivityError("Markdown must have a distinct path")
    return output_path, markdown_path


def _require_unchanged_input(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise SensitivityError("input analysis changed during sensitivity generation")


def _validated_normalization_cases(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    _require_unlinked_components(path, context="normalization case index")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise SensitivityError("normalization case index differs from its external SHA256")
    value = read_json(path)
    if (
        not isinstance(value, Mapping)
        or value.get("artifact_type") != NORMALIZATION_CASE_INDEX_TYPE
        or value.get("run_id") != EXPECTED_RUN_ID
        or value.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or value.get("source_pair_manifest_sha256") != EXPECTED_PAIR_MANIFEST_SHA256
    ):
        raise SensitivityError("normalization case index identity changed")
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or any(
        not isinstance(row, Mapping) for row in raw_cases
    ):
        raise SensitivityError("normalization case index has invalid cases")
    cases = tuple(dict(row) for row in raw_cases)
    by_cell = {str(row.get("cell_id")): row for row in cases}
    if len(by_cell) != len(cases) or set(by_cell) != set(RECOVERY_CELL_TREATMENTS):
        raise SensitivityError("normalization case set is not the exact recovered-cell set")
    for cell_id, expected_treatment in RECOVERY_CELL_TREATMENTS.items():
        row = by_cell[cell_id]
        groups = row.get("groups")
        if (
            (row.get("method"), row.get("operator")) != expected_treatment
            or row.get("source_task_id") != EXPECTED_RECOVERY_TASKS[cell_id]
            or row.get("replicate_id") != 0
            or row.get("unit_id") != f"{EXPECTED_RECOVERY_TASKS[cell_id]}/r0"
            or row.get("normalization_required") is not True
            or not isinstance(groups, list)
            or not groups
            or not isinstance(row.get("attempt_chain_sha256"), str)
        ):
            raise SensitivityError(f"normalization case lock changed: {cell_id}")
    expected_group_shape = {
        "d52046b6eb74a76ecdc3debc": [(5, 1, 2)],
        "89df41e0daa1262a43fa5e55": [(6, 4, 5)],
        "786d95760ccdb86713c26936": [(5, 1, 2), (6, 0, 1)],
    }
    for cell_id, shapes in expected_group_shape.items():
        groups = by_cell[cell_id]["groups"]
        observed = [
            (
                row.get("checkpoint"),
                row.get("semantic_failed_attempts"),
                row.get("logical_attempts"),
            )
            for row in groups
            if isinstance(row, Mapping)
        ]
        if observed != shapes:
            raise SensitivityError(f"normalization attempt-chain shape changed: {cell_id}")
    if value.get("affected_units") != list(EXPECTED_AFFECTED_UNIT_IDS):
        raise SensitivityError("normalization case index affected-unit set changed")
    return value, cases


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return str(value)


def _markdown(receipt: Mapping[str, Any], *, json_sha256: str) -> str:
    audit = receipt["audit_result"]
    selection = receipt["affected_units_selection"]
    material_effects = receipt["comparisons"]["material_operator_effect_changes"]
    material_summaries = receipt["comparisons"]["material_absolute_summary_shifts"]
    lines = [
        "# Online cumulative leave-two-source-units sensitivity",
        "",
        "This provider-free sensitivity derives two omitted source units from the "
        "complete documented recovery-cell set and removes both units from every "
        "method/operator treatment. No outcome field is used for selection.",
        "",
        f"- Source analysis SHA256: `{receipt['source']['analysis_sha256']}`",
        f"- Receipt JSON SHA256: `{json_sha256}`",
        f"- Script SHA256: `{receipt['generator']['script_sha256']}`",
        "- Documented recovery cells: "
        + ", ".join(
            f"`{row['cell_id']}`" for row in selection["documented_recovery_cells"]
        ),
        "- Derived source units: "
        + ", ".join(f"`{unit_id}`" for unit_id in selection["affected_unit_ids"]),
        f"- Removed rows: {selection['removed_rows']} (two from each of 28 treatments)",
        f"- Denominator: {receipt['design']['source_tasks']} -> {receipt['design']['filtered_tasks']} paired source tasks",
        f"- Scientific outcome changes: {audit['scientific_outcome_change_count']}",
        f"- Action-policy changes: {audit['action_policy_change_count']}",
        f"- Resource-sensitivity flags: {audit['resource_sensitivity_count']}",
        f"- Overall assessment: **{audit['assessment']}**",
        "",
        "## Frozen rules",
        "",
        "- Bootstrap: 2,000 paired source-task resamples, seed 12012, 95% intervals.",
        "- An operator effect is flagged if its point sign changes or its 95% interval "
        "changes among negative/excludes-zero, includes-zero, and positive/excludes-zero.",
        "- Absolute success and selected-action-rate means flag at 1/40 (0.025), "
        "or for zero/nonzero changes.",
        "- Threshold-firing and resource means flag at a 5% relative shift, or for "
        "zero/nonzero changes.",
        "",
        "## Material operator-effect changes",
        "",
    ]
    if material_effects:
        lines.extend(
            [
                "| group | method | operator | metric | n=40 effect / inference | n=38 effect / inference | reason |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in material_effects:
            lines.append(
                "| {group} | {method} | {operator} | {metric} | {old} / {old_i} | "
                "{new} / {new_i} | {reason} |".format(
                    group=row["metric_group"],
                    method=row["method"],
                    operator=row["operator"],
                    metric=row["metric"],
                    old=_format_number(row["n40"]["effect"]),
                    old_i=row["n40"]["inference"],
                    new=_format_number(row["n38"]["effect"]),
                    new_i=row["n38"]["inference"],
                    reason=", ".join(row["material_reasons"]),
                )
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Material absolute-summary shifts", ""])
    if material_summaries:
        lines.extend(
            [
                "| group | method | operator | metric | n=40 mean | n=38 mean | delta |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in material_summaries:
            lines.append(
                "| {group} | {method} | {operator} | {metric} | {old} | {new} | {delta} |".format(
                    group=row["metric_group"],
                    method=row["method"],
                    operator=row["operator"],
                    metric=row["metric"],
                    old=_format_number(row["n40"]["mean"]),
                    new=_format_number(row["n38"]["mean"]),
                    delta=_format_number(row["mean_delta_n38_minus_n40"]),
                )
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "The JSON receipt contains every n=40 versus n=38 outcome, action, and resource "
            "comparison and the complete recomputed n=38 summaries/effects.",
            "",
        ]
    )
    return "\n".join(lines)


def build_receipt(
    analysis: Mapping[str, Any],
    *,
    input_path: Path,
    input_sha256: str,
    normalization_cases_path: Path,
    normalization_cases_sha256: str,
    normalization_cases: Sequence[Mapping[str, Any]],
    affected_cell_id: str,
) -> dict[str, Any]:
    if (
        analysis.get("artifact_type") != ADAPTIVE_ANALYSIS_TYPE
        or analysis.get("adaptive_analysis_version") != ADAPTIVE_ANALYSIS_VERSION
    ):
        raise SensitivityError("input is not an adaptive deployment analysis")
    if analysis.get("source_run_id") != EXPECTED_RUN_ID:
        raise SensitivityError("input source run is not the frozen online run")
    if analysis.get("source_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise SensitivityError("input source manifest differs from the frozen run")
    if analysis.get("source_pair_manifest_sha256") != EXPECTED_PAIR_MANIFEST_SHA256:
        raise SensitivityError("input source pair manifest differs from the frozen run")
    if analysis.get("deployment_mode") != "online_adaptive":
        raise SensitivityError("input is not the online adaptive estimand")
    if analysis.get("statistical_unit") != "source_task":
        raise SensitivityError("input statistical unit is not source_task")
    if affected_cell_id != AFFECTED_CELL_ID:
        raise SensitivityError("primary affected cell ID differs from the frozen design")
    raw_rows = analysis.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_ROWS:
        raise SensitivityError(f"input must contain exactly {EXPECTED_ROWS} rows")
    rows = tuple(dict(row) for row in raw_rows if isinstance(row, Mapping))
    if len(rows) != len(raw_rows):
        raise SensitivityError("input rows must all be objects")

    methods = sorted({str(row.get("method")) for row in rows})
    operators = sorted({str(row.get("operator")) for row in rows})
    units = sorted({str(row.get("unit_id")) for row in rows})
    models = sorted({str(row.get("model")) for row in rows})
    benchmarks = sorted({str(row.get("benchmark")) for row in rows})
    task_ids = sorted({str(row.get("task_id")) for row in rows})
    replicate_ids = {row.get("replicate_id") for row in rows}
    if (
        methods != list(EXPECTED_METHOD_NAMES)
        or operators != list(EXPECTED_OPERATOR_NAMES)
        or len(units) != EXPECTED_TASKS
        or len(task_ids) != EXPECTED_TASKS
        or models != [EXPECTED_MODEL]
        or benchmarks != [EXPECTED_BENCHMARK]
        or replicate_ids != {0}
        or len({str(row.get("cell_id")) for row in rows}) != EXPECTED_ROWS
        or any(
            row.get("unit_id")
            != f"{row.get('task_id')}/r{row.get('replicate_id')}"
            for row in rows
        )
    ):
        raise SensitivityError("input treatment/task slice differs from the frozen design")
    treatment_counts: dict[tuple[str, str], int] = {}
    treatment_units: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        treatment = (str(row["method"]), str(row["operator"]))
        treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
        treatment_units.setdefault(treatment, set()).add(str(row["unit_id"]))
    expected_treatments = {(method, operator) for method in methods for operator in operators}
    expected_units = set(units)
    if (
        set(treatment_counts) != expected_treatments
        or set(treatment_counts.values()) != {EXPECTED_TASKS}
        or any(value != expected_units for value in treatment_units.values())
    ):
        raise SensitivityError("input is not the exact 7 x 4 x 40 treatment product")

    recovery_rows: list[dict[str, Any]] = []
    declared_case_index = {
        str(row["cell_id"]): row for row in normalization_cases
    }
    for recovery_cell_id, expected_treatment in RECOVERY_CELL_TREATMENTS.items():
        matches = [row for row in rows if row.get("cell_id") == recovery_cell_id]
        if len(matches) != 1:
            raise SensitivityError(
                f"recovery cell {recovery_cell_id} must map to exactly one analysis row"
            )
        recovery_row = matches[0]
        declared = declared_case_index[recovery_cell_id]
        if (
            recovery_row.get("method"),
            recovery_row.get("operator"),
        ) != expected_treatment:
            raise SensitivityError(
                f"recovery cell {recovery_cell_id} treatment identity changed"
            )
        expected_task_id = EXPECTED_RECOVERY_TASKS[recovery_cell_id]
        if (
            recovery_row.get("task_id") != expected_task_id
            or recovery_row.get("replicate_id") != 0
            or recovery_row.get("unit_id") != f"{expected_task_id}/r0"
            or declared.get("source_task_id") != recovery_row.get("task_id")
            or declared.get("unit_id") != recovery_row.get("unit_id")
            or declared.get("method") != recovery_row.get("method")
            or declared.get("operator") != recovery_row.get("operator")
        ):
            raise SensitivityError(
                f"recovery cell {recovery_cell_id} source-task identity changed"
            )
        recovery_rows.append(recovery_row)
    affected_matches = [
        row for row in recovery_rows if row.get("cell_id") == affected_cell_id
    ]
    if len(affected_matches) != 1:
        raise SensitivityError("primary affected cell is absent from recovery cells")
    recovery_units = {str(row.get("unit_id")) for row in recovery_rows}
    expected_recovery_units = set(EXPECTED_AFFECTED_UNIT_IDS)
    if recovery_units != expected_recovery_units:
        raise SensitivityError(
            "documented recovery cells do not map to the exact two frozen source units"
        )
    removed = tuple(row for row in rows if str(row.get("unit_id")) in recovery_units)
    filtered_rows = tuple(
        row for row in rows if str(row.get("unit_id")) not in recovery_units
    )
    removed_treatment_counts: dict[tuple[str, str], int] = {}
    for row in removed:
        treatment = (str(row.get("method")), str(row.get("operator")))
        removed_treatment_counts[treatment] = removed_treatment_counts.get(treatment, 0) + 1
    if (
        len(removed) != EXPECTED_REMOVED_ROWS
        or set(removed_treatment_counts) != expected_treatments
        or set(removed_treatment_counts.values()) != {EXPECTED_AFFECTED_UNITS}
        or len(filtered_rows) != EXPECTED_FILTERED_ROWS
        or len({str(row.get("unit_id")) for row in filtered_rows}) != FILTERED_TASKS
    ):
        raise SensitivityError(
            "cumulative removal did not remove two rows from every treatment"
        )
    filtered_counts: dict[tuple[str, str], int] = {}
    filtered_treatment_units: dict[tuple[str, str], set[str]] = {}
    for row in filtered_rows:
        treatment = (str(row["method"]), str(row["operator"]))
        filtered_counts[treatment] = filtered_counts.get(treatment, 0) + 1
        filtered_treatment_units.setdefault(treatment, set()).add(str(row["unit_id"]))
    expected_filtered_units = expected_units - recovery_units
    if (
        set(filtered_counts) != expected_treatments
        or set(filtered_counts.values()) != {FILTERED_TASKS}
        or any(
            value != expected_filtered_units
            for value in filtered_treatment_units.values()
        )
    ):
        raise SensitivityError(
            "filtered treatments do not share the identical 38 source units"
        )

    full_summaries, full_effects = summarize_adaptive_outcomes(
        rows,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence=CONFIDENCE,
    )
    canonical_full_summaries = [asdict(row) for row in full_summaries]
    canonical_full_effects = [asdict(row) for row in full_effects]
    if (
        len(canonical_full_summaries) != EXPECTED_TREATMENTS * len(COMPARISON_METRICS)
        or len(canonical_full_effects)
        != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1) * len(COMPARISON_METRICS)
        or {row["n_tasks"] for row in canonical_full_summaries} != {EXPECTED_TASKS}
        or {row["n_tasks"] for row in canonical_full_effects} != {EXPECTED_TASKS}
        or analysis.get("metric_summaries") != canonical_full_summaries
        or analysis.get("operator_effects") != canonical_full_effects
    ):
        raise SensitivityError(
            "input summaries do not exactly reproduce from source rows and frozen bootstrap"
        )

    filtered_summaries, filtered_effects = summarize_adaptive_outcomes(
        filtered_rows,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence=CONFIDENCE,
    )
    n38_summaries = [asdict(row) for row in filtered_summaries]
    n38_effects = [asdict(row) for row in filtered_effects]
    if (
        len(n38_summaries) != EXPECTED_TREATMENTS * len(COMPARISON_METRICS)
        or len(n38_effects)
        != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1) * len(COMPARISON_METRICS)
        or {row["n_tasks"] for row in n38_summaries} != {FILTERED_TASKS}
        or {row["n_tasks"] for row in n38_effects} != {FILTERED_TASKS}
    ):
        raise SensitivityError("recomputed summaries do not use exactly 38 tasks")

    full_summary_index = _index(
        canonical_full_summaries, _SUMMARY_KEY, context="n40 summaries"
    )
    filtered_summary_index = _index(
        n38_summaries, _SUMMARY_KEY, context="n38 summaries"
    )
    full_effect_index = _index(canonical_full_effects, _EFFECT_KEY, context="n40 effects")
    filtered_effect_index = _index(n38_effects, _EFFECT_KEY, context="n38 effects")
    if set(full_summary_index) != set(filtered_summary_index):
        raise SensitivityError("n=40 and n=38 summary identities differ")
    if set(full_effect_index) != set(filtered_effect_index):
        raise SensitivityError("n=40 and n=38 effect identities differ")

    selected_summary_keys = sorted(
        key for key in full_summary_index if str(key[-1]) in COMPARISON_METRICS
    )
    selected_effect_keys = sorted(
        key for key in full_effect_index if str(key[-1]) in COMPARISON_METRICS
    )
    summary_comparisons = [
        _summary_comparison(full_summary_index[key], filtered_summary_index[key])
        for key in selected_summary_keys
    ]
    effect_comparisons = [
        _effect_comparison(full_effect_index[key], filtered_effect_index[key])
        for key in selected_effect_keys
    ]
    expected_summary_comparisons = EXPECTED_TREATMENTS * len(COMPARISON_METRICS)
    expected_effect_comparisons = (
        EXPECTED_METHODS * (EXPECTED_OPERATORS - 1) * len(COMPARISON_METRICS)
    )
    success_effects = [
        row for row in effect_comparisons if row["metric"] == SUCCESS_METRIC
    ]
    action_effects = [
        row for row in effect_comparisons if row["metric"] in ACTION_POLICY_METRICS
    ]
    threshold_effects = [
        row for row in effect_comparisons if row["metric"] == THRESHOLD_METRIC
    ]
    action_rate_effects = [
        row for row in effect_comparisons if row["metric"] == ACTION_RATE_METRIC
    ]
    resource_effects = [
        row for row in effect_comparisons if row["metric"] in KEY_RESOURCE_METRICS
    ]
    if (
        len(summary_comparisons) != expected_summary_comparisons
        or len(effect_comparisons) != expected_effect_comparisons
        or len(success_effects) != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1)
        or len(action_effects)
        != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1) * len(ACTION_POLICY_METRICS)
        or len(threshold_effects) != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1)
        or len(action_rate_effects) != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1)
        or len(resource_effects)
        != EXPECTED_METHODS * (EXPECTED_OPERATORS - 1) * len(KEY_RESOURCE_METRICS)
    ):
        raise SensitivityError("comparison coverage is incomplete")

    material_effects = [row for row in effect_comparisons if row["material_change"]]
    material_summaries = [
        row for row in summary_comparisons if row["material_absolute_shift"]
    ]
    scientific_effect_changes = [
        row
        for row in material_effects
        if row["metric_group"] == "scientific_outcome"
    ]
    action_effect_changes = [
        row for row in material_effects if row["metric_group"] == "action_policy"
    ]
    resource_effect_changes = [
        row for row in material_effects if row["metric_group"] == "resource"
    ]
    scientific_summary_changes = [
        row
        for row in material_summaries
        if row["metric_group"] == "scientific_outcome"
    ]
    action_summary_changes = [
        row for row in material_summaries if row["metric_group"] == "action_policy"
    ]
    resource_summary_changes = [
        row for row in material_summaries if row["metric_group"] == "resource"
    ]
    scientific_changed = bool(scientific_effect_changes or scientific_summary_changes)
    action_policy_changed = bool(action_effect_changes or action_summary_changes)
    resource_changed = bool(resource_effect_changes or resource_summary_changes)
    if scientific_changed:
        assessment = "scientific_outcome_conclusion_changed"
    elif action_policy_changed:
        assessment = "outcome_stable_action_policy_qualification_changed"
    elif resource_changed:
        assessment = "outcome_and_action_stable_resource_sensitivity_detected"
    else:
        assessment = "stable_under_paired_cumulative_affected_units_out"
    affected_identities = [
        {
            key: row[key]
            for key in (
                "cell_id",
                "model",
                "benchmark",
                "task_id",
                "replicate_id",
                "unit_id",
                "method",
                "operator",
            )
        }
        for row in sorted(recovery_rows, key=lambda item: str(item["cell_id"]))
    ]
    script_path = Path(__file__).resolve()
    analyzer_path = script_path.parents[1] / "adaptive_analysis12.py"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "created_at_utc": _utc_now(),
        "provider_calls_made": 0,
        "analysis_only": True,
        "source": {
            "analysis_path": _path_label(input_path),
            "analysis_size_bytes": input_path.stat().st_size,
            "analysis_sha256": input_sha256,
            "analysis_sha256_verified_after_writes": True,
            "source_run_id": analysis["source_run_id"],
            "source_manifest_sha256": analysis["source_manifest_sha256"],
            "source_pair_manifest_sha256": analysis["source_pair_manifest_sha256"],
            "deployment_mode": analysis["deployment_mode"],
            "statistical_unit": analysis["statistical_unit"],
            "n40_metric_summaries_sha256": sha256_json(canonical_full_summaries),
            "n40_operator_effects_sha256": sha256_json(canonical_full_effects),
            "normalization_case_index_path": _path_label(normalization_cases_path),
            "normalization_case_index_size_bytes": normalization_cases_path.stat().st_size,
            "normalization_case_index_sha256": normalization_cases_sha256,
            "normalization_case_index_verified_after_writes": True,
        },
        "generator": {
            "script_path": _path_label(script_path),
            "script_sha256": sha256_file(script_path),
            "adaptive_analysis_path": _path_label(analyzer_path),
            "adaptive_analysis_sha256": sha256_file(analyzer_path),
            "frozen_function": (
                "experiments12.adaptive_analysis12.summarize_adaptive_outcomes"
            ),
        },
        "design": {
            "model": models[0],
            "benchmark": benchmarks[0],
            "source_rows": len(rows),
            "filtered_rows": len(filtered_rows),
            "source_tasks": len(units),
            "affected_source_units": len(recovery_units),
            "filtered_tasks": len({row["unit_id"] for row in filtered_rows}),
            "recovery_cells": len(recovery_rows),
            "removed_rows": len(removed),
            "methods": methods,
            "operators": operators,
            "treatments": len(expected_treatments),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "bootstrap_unit": "paired_source_task",
            "n40_common_unit_set_sha256": sha256_json(sorted(expected_units)),
            "n38_common_unit_set_sha256": sha256_json(
                sorted(expected_filtered_units)
            ),
        },
        "affected_units_selection": {
            "primary_affected_cell_id": affected_cell_id,
            "identification_rule": (
                "find the unique source row for every cell in the complete documented "
                "recovery-cell set; derive the distinct unit_id set; omit every row "
                "whose unit_id belongs to that set"
            ),
            "selection_fields_used": ["cell_id", "unit_id"],
            "outcome_fields_used_for_selection": [],
            "affected_row_identities": affected_identities,
            "affected_unit_ids": sorted(recovery_units),
            "distinct_affected_units": len(recovery_units),
            "documented_recovery_cells": [
                {
                    key: row[key]
                    for key in (
                        "cell_id",
                        "method",
                        "operator",
                        "task_id",
                        "replicate_id",
                        "unit_id",
                    )
                }
                for row in sorted(recovery_rows, key=lambda item: str(item["cell_id"]))
            ],
            "normalization_attempt_chain_sha256_by_cell": {
                cell_id: declared_case_index[cell_id]["attempt_chain_sha256"]
                for cell_id in sorted(declared_case_index)
            },
            "recovery_cells_map_to_exact_predeclared_unit_set": True,
            "new_recovery_unit_policy": (
                "fail; extend the predeclared recovery-cell inventory and rerun a "
                "cumulative exclusion across every distinct affected source unit"
            ),
            "removed_rows": len(removed),
            "removed_treatments": len(removed_treatment_counts),
            "removed_rows_per_treatment": EXPECTED_AFFECTED_UNITS,
            "removed_cell_ids_sha256": sha256_json(
                sorted(str(row["cell_id"]) for row in removed)
            ),
            "removed_rows_sha256": sha256_json(
                sorted(removed, key=lambda row: str(row["cell_id"]))
            ),
            "filtered_rows_sha256": sha256_json(
                sorted(filtered_rows, key=lambda row: str(row["cell_id"]))
            ),
        },
        "materiality_rules": {
            "effect_point_sign": {
                "zero_tolerance": _ZERO_TOLERANCE,
                "flag": "n40 and n38 point-effect sign labels differ",
            },
            "effect_inference": {
                "confidence": CONFIDENCE,
                "classes": [
                    "negative_excludes_zero",
                    "includes_zero",
                    "positive_excludes_zero",
                ],
                "flag": "n40 and n38 confidence-interval classes differ",
            },
            "absolute_success": {
                "threshold": _BOUNDED_RATE_ABSOLUTE_SHIFT,
                "flag": "absolute mean shift is at least 1/40, or zero/nonzero changes",
            },
            "absolute_selected_actions_action_rate": {
                "threshold": _BOUNDED_RATE_ABSOLUTE_SHIFT,
                "flag": "absolute mean shift is at least 1/40, or zero/nonzero changes",
            },
            "absolute_threshold_firings": {
                "threshold_relative_to_n40_mean": _DESCRIPTIVE_RELATIVE_SHIFT,
                "flag": "absolute relative mean shift is at least 5%; zero/nonzero also flags",
            },
            "absolute_resource": {
                "threshold_relative_to_n40_mean": _DESCRIPTIVE_RELATIVE_SHIFT,
                "flag": "absolute relative mean shift is at least 5%; zero-to-nonzero also flags",
            },
            "operator_effect_material_change": (
                "point-effect sign change OR confidence-interval class change"
            ),
        },
        "comparisons": {
            "metrics": list(COMPARISON_METRICS),
            "absolute_summaries": summary_comparisons,
            "operator_effects": effect_comparisons,
            "success_operator_effects_compared": len(success_effects),
            "threshold_firing_operator_effects_compared": len(threshold_effects),
            "selected_action_rate_operator_effects_compared": len(
                action_rate_effects
            ),
            "key_resource_operator_effects_compared": len(resource_effects),
            "material_operator_effect_changes": material_effects,
            "material_absolute_summary_shifts": material_summaries,
            "scientific_outcome_operator_effect_changes": scientific_effect_changes,
            "action_policy_operator_effect_changes": action_effect_changes,
            "resource_operator_effect_changes": resource_effect_changes,
            "scientific_outcome_absolute_summary_shifts": scientific_summary_changes,
            "action_policy_absolute_summary_shifts": action_summary_changes,
            "resource_absolute_summary_shifts": resource_summary_changes,
        },
        "recomputed_n38": {
            "metric_summaries": n38_summaries,
            "operator_effects": n38_effects,
            "metric_summaries_sha256": sha256_json(n38_summaries),
            "operator_effects_sha256": sha256_json(n38_effects),
        },
        "audit_result": {
            "source_summaries_exactly_reproduced": True,
            "all_affected_units_removed_from_every_treatment": True,
            "two_units_removed_from_every_treatment": True,
            "all_treatments_share_exact_n38_unit_set": True,
            "all_recovery_cells_mapped_to_predeclared_units": True,
            "normalization_case_index_hash_verified": True,
            "selection_was_outcome_blind": True,
            "all_metric_summaries_compared": len(summary_comparisons),
            "all_operator_effects_compared": len(effect_comparisons),
            "success_operator_effects_compared": len(success_effects),
            "threshold_firing_operator_effects_compared": len(threshold_effects),
            "selected_action_rate_operator_effects_compared": len(
                action_rate_effects
            ),
            "key_resource_operator_effects_compared": len(resource_effects),
            "material_operator_effect_changes": len(material_effects),
            "material_absolute_summary_shifts": len(material_summaries),
            "scientific_outcome_change_count": len(scientific_effect_changes)
            + len(scientific_summary_changes),
            "action_policy_change_count": len(action_effect_changes)
            + len(action_summary_changes),
            "resource_sensitivity_count": len(resource_effect_changes)
            + len(resource_summary_changes),
            "scientific_outcome_assessment": (
                "changed" if scientific_changed else "stable"
            ),
            "action_policy_assessment": (
                "changed" if action_policy_changed else "stable"
            ),
            "resource_assessment": (
                "sensitive" if resource_changed else "stable"
            ),
            "assessment": assessment,
        },
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--input", required=True, help="validated adaptive-analysis JSON")
    root.add_argument(
        "--expected-input-sha256",
        required=True,
        help="externally recorded SHA256 of the complete adaptive analysis",
    )
    root.add_argument(
        "--normalization-cases",
        required=True,
        help="hash-bound staging normalization case index",
    )
    root.add_argument(
        "--expected-normalization-cases-sha256",
        required=True,
        help="externally recorded SHA256 of the normalization case index",
    )
    root.add_argument("--output", required=True, help="JSON sensitivity receipt")
    root.add_argument("--markdown", required=True, help="Markdown sensitivity receipt")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        expected = _digest(
            args.expected_input_sha256, context="expected input analysis SHA256"
        )
        expected_cases = _digest(
            args.expected_normalization_cases_sha256,
            context="expected normalization case index SHA256",
        )
        input_path = Path(args.input)
        normalization_cases_path = Path(args.normalization_cases)
        output_path, markdown_path = _validated_output_paths(
            input_path, Path(args.output), Path(args.markdown)
        )
        if input_path.is_symlink() or not input_path.is_file():
            raise SensitivityError("input analysis must be a regular non-symlink file")
        actual = sha256_file(input_path)
        if actual != expected:
            raise SensitivityError("input analysis differs from its external SHA256")
        analysis = read_json(input_path)
        if not isinstance(analysis, Mapping):
            raise SensitivityError("input analysis must be a JSON object")
        _case_index, normalization_cases = _validated_normalization_cases(
            normalization_cases_path, expected_cases
        )
        receipt = build_receipt(
            analysis,
            input_path=input_path,
            input_sha256=actual,
            normalization_cases_path=normalization_cases_path,
            normalization_cases_sha256=expected_cases,
            normalization_cases=normalization_cases,
            affected_cell_id=AFFECTED_CELL_ID,
        )
        _require_unchanged_input(input_path, actual)
        _validated_normalization_cases(normalization_cases_path, expected_cases)
        atomic_write_json(output_path, receipt)
        _require_unchanged_input(input_path, actual)
        _validated_normalization_cases(normalization_cases_path, expected_cases)
        receipt_sha = sha256_file(output_path)
        atomic_write_text(markdown_path, _markdown(receipt, json_sha256=receipt_sha))
        _require_unchanged_input(input_path, actual)
        _validated_normalization_cases(normalization_cases_path, expected_cases)
        return 0
    except (SensitivityError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AFFECTED_CELL_ID",
    "ARTIFACT_TYPE",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "KEY_RESOURCE_METRICS",
    "RECOVERY_CELL_TREATMENTS",
    "SensitivityError",
    "build_receipt",
    "main",
    "parser",
]
