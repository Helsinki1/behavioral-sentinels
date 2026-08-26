#!/usr/bin/env python3
"""Provider-free diagnostic for rank-selected versus scalar firing policies.

This is deliberately a generated, post-hoc audit artifact.  It reads only the
frozen Luna calibration extract/thresholds and the completed 40-task deployment
pass-one trajectories and shadows.  It makes no provider calls and does not
modify the Experiment 12 source tree.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments12.analysis12 import (  # noqa: E402
    _validated_materialization,
    signal_traces,
)
from experiments12.core.artifacts import sha256_file  # noqa: E402
from experiments12.manifest12 import RunLayout, code_tree_hash  # noqa: E402
from experiments12.metrics12 import (  # noqa: E402
    ObservationTrace,
    collapse_task_predictions,
)


MODEL = "gpt-5.6-luna"
BENCHMARK = "evolving_intent_gsm8k"
TARGET_RATE = 0.20
TIE_SEED = 12_012
EXPECTED_CODE_TREE_SHA256 = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)
PASS_ONE_RUN_ID = "e12-deploy-twopass-pass1-evolving-luna-40-v1"
EXPECTED_PASS_ONE_MANIFEST_SHA256 = (
    "b5f4d46f8deb8b899c8d9cb35ae3758f858077e6a94704a30f9c3cabe5d2aa8f"
)
METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
    "context_use",
)
TIMING_METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "trace_judge",
)
METHOD_LABELS = {
    "active_recompute": "Active recompute (carry)",
    "frozen_probe:recompute": "Frozen recompute",
    "frozen_quiz": "Frozen quiz",
    "trace_judge": "Trace judge",
    "trace_rules": "Trace rules",
    "turn_clock": "Turn clock",
    "context_use": "Context use",
}

GENERATED = REPOSITORY_ROOT / "experiments12" / "data_results" / "derived"
ARTIFACTS = REPOSITORY_ROOT / "experiments12" / "data_results" / "runs"
CALIBRATION_EXTRACT = (
    ARTIFACTS
    / "e12-calibration-evolving-core-v2"
    / "results"
    / "extract-calibration.json"
)
CALIBRATION_MANIFEST = (
    ARTIFACTS / "e12-calibration-evolving-core-v2" / "manifest.json"
)
THRESHOLDS = GENERATED / "thresholds-calibration-evolving-core-v2.json"
PASS_ONE_VALIDATION = (
    ARTIFACTS / PASS_ONE_RUN_ID / "results" / "validation-pass-one.json"
)
OUTPUT_JSON = GENERATED / "luna-threshold-diagnostic12.json"
OUTPUT_RATES_SVG = GENERATED / "luna-threshold-rates12.svg"
OUTPUT_TIMING_SVG = GENERATED / "luna-top8-timing12.svg"


class DiagnosticError(RuntimeError):
    """Raised when a frozen input or diagnostic invariant is violated."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _actionable_dict_checkpoints(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    event = trace.get("event_checkpoint")
    checkpoints = trace.get("checkpoints")
    _require(isinstance(checkpoints, list) and checkpoints, "calibration trace is empty")
    actionable = [
        row
        for row in checkpoints
        if row.get("actionable") is True
        and (event is None or int(row["checkpoint"]) < int(event))
    ]
    _require(bool(actionable), "calibration trace has no actionable checkpoint")
    return actionable


def _calibration_scores(
    extract: Mapping[str, Any], method: str
) -> list[dict[str, Any]]:
    traces = [
        row
        for row in extract["signal_traces"]
        if row.get("model") == MODEL
        and row.get("benchmark") == BENCHMARK
        and row.get("method") == method
    ]
    rows: list[dict[str, Any]] = []
    for trace in traces:
        actionable = _actionable_dict_checkpoints(trace)
        maximum = max(float(row["score"]) for row in actionable)
        earliest = min(
            int(row["checkpoint"])
            for row in actionable
            if float(row["score"]) == maximum
        )
        rows.append(
            {
                "task_id": str(trace["task_id"]),
                "source_task_id": str(trace["source_task_id"]),
                "score": maximum,
                "earliest_max_checkpoint": earliest,
            }
        )
    rows.sort(key=lambda row: row["task_id"])
    return rows


def _deployment_rows(
    traces: Iterable[ObservationTrace], method: str
) -> list[dict[str, Any]]:
    selected = tuple(
        trace
        for trace in traces
        if trace.model == MODEL
        and trace.benchmark == BENCHMARK
        and trace.method == method
    )
    predictions = collapse_task_predictions(selected)
    trace_by_task = {trace.task_id: trace for trace in selected}
    _require(
        len(trace_by_task) == len(selected),
        f"duplicate deployment traces for {method}",
    )
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        _require(
            prediction.has_actionable_checkpoint,
            f"deployment trace has no actionable checkpoint: {method}/{prediction.task_id}",
        )
        trace = trace_by_task[prediction.task_id]
        actionable = [
            checkpoint
            for checkpoint in trace.checkpoints
            if checkpoint.actionable
            and (
                trace.event_checkpoint is None
                or checkpoint.checkpoint < trace.event_checkpoint
            )
        ]
        earliest = min(
            checkpoint.checkpoint
            for checkpoint in actionable
            if checkpoint.score == prediction.score
        )
        rows.append(
            {
                "task_id": prediction.task_id,
                "source_task_id": trace.source_task_id,
                "score": prediction.score,
                "earliest_max_checkpoint": earliest,
                "event_checkpoint": trace.event_checkpoint,
                "signal_path": (
                    "carried_active_trajectory"
                    if method == "active_recompute"
                    else "zero_carry_clean_shadow"
                ),
            }
        )
    rows.sort(key=lambda row: row["task_id"])
    return rows


def _common_tie_sha256(source_task_id: str) -> str:
    material = "\0".join(
        (
            "exp12/common-task-rank-tie/v1",
            str(TIE_SEED),
            MODEL,
            BENCHMARK,
            source_task_id,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _rank_common(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            _common_tie_sha256(str(row["source_task_id"])),
            str(row["source_task_id"]),
        ),
    )
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:count], start=1):
        result.append(
            {
                **dict(row),
                "rank": rank,
                "common_tie_sha256": _common_tie_sha256(
                    str(row["source_task_id"])
                ),
            }
        )
    return result


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _score_label(value: float) -> str:
    if value == 0.0 or value == 1.0:
        return str(int(value))
    return f"{value:.3g}"


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>",
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172033}",
        ".title{font-size:25px;font-weight:700;letter-spacing:-.3px}",
        ".subtitle{font-size:13px;fill:#596579}",
        ".axis{font-size:12px;fill:#657188}",
        ".label{font-size:13px;font-weight:600}",
        ".value{font-size:11px;font-weight:650}",
        ".note{font-size:11px;fill:#657188}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _write_rates_svg(rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1180, 650
    left, right = 292.0, 1110.0
    plot_width = right - left
    row_start, row_gap = 128.0, 64.0
    axis_y = 584.0
    lines = _svg_header(width, height, "Rank and scalar-implied firing rates")
    lines.extend(
        [
            '<text class="title" x="34" y="38">Rank target versus scalar-threshold firing</text>',
            '<text class="subtitle" x="34" y="61">Luna · Evolving-Intent GSM8K · task-level maximum over actionable checkpoints</text>',
            '<line x1="34" y1="79" x2="1146" y2="79" stroke="#e6eaf0"/>',
            '<circle cx="365" cy="96" r="5" fill="#2563eb"/>',
            '<text class="axis" x="376" y="100">Intended rank policy</text>',
            '<circle cx="525" cy="96" r="5" fill="#d97706"/>',
            '<text class="axis" x="536" y="100">Calibration, scalar ≥ cutoff</text>',
            '<rect x="735" y="91" width="10" height="10" rx="2" fill="#dc2626"/>',
            '<text class="axis" x="752" y="100">Deployment pass one, scalar ≥ cutoff</text>',
        ]
    )
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100.0
        color = "#9db7ea" if tick == 20 else "#e8ebf1"
        stroke_width = "1.6" if tick == 20 else "1"
        lines.append(
            f'<line x1="{x:.1f}" y1="112" x2="{x:.1f}" y2="{axis_y}" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text class="axis" x="{x:.1f}" y="{axis_y + 21:.1f}" text-anchor="middle">{tick}%</text>'
        )
    for index, row in enumerate(rows):
        y = row_start + index * row_gap
        if index % 2 == 0:
            lines.append(
                f'<rect x="28" y="{y - 27:.1f}" width="1118" height="54" rx="7" fill="#f8fafc"/>'
            )
        lines.append(
            f'<text class="label" x="42" y="{y + 5:.1f}">{escape(METHOD_LABELS[str(row["method"])])}</text>'
        )
        cal = float(row["scalar_implied"]["calibration"]["firing_rate"])
        dep = float(row["scalar_implied"]["deployment_pass_one"]["firing_rate"])
        target = float(row["intended_rank_policy"]["deployment_firing_rate"])
        x_cal = left + plot_width * cal
        x_dep = left + plot_width * dep
        x_target = left + plot_width * target
        lines.append(
            f'<line x1="{min(x_cal, x_dep):.1f}" y1="{y:.1f}" x2="{max(x_cal, x_dep):.1f}" y2="{y:.1f}" stroke="#cbd2dd" stroke-width="2"/>'
        )
        lines.append(
            f'<circle cx="{x_target:.1f}" cy="{y - 14:.1f}" r="5.5" fill="#2563eb"/>'
        )
        lines.append(
            f'<text class="value" x="{x_target + 9:.1f}" y="{y - 10:.1f}" fill="#1d4ed8">{_pct(target)}</text>'
        )
        lines.append(
            f'<circle cx="{x_cal:.1f}" cy="{y:.1f}" r="5.5" fill="#d97706"/>'
        )
        cal_anchor = "end" if cal > 0.92 else "start"
        cal_dx = -9 if cal > 0.92 else 9
        lines.append(
            f'<text class="value" x="{x_cal + cal_dx:.1f}" y="{y + 4:.1f}" text-anchor="{cal_anchor}" fill="#a85e05">{_pct(cal)}</text>'
        )
        lines.append(
            f'<rect x="{x_dep - 5.5:.1f}" y="{y + 8.5:.1f}" width="11" height="11" rx="2" fill="#dc2626"/>'
        )
        dep_anchor = "end" if dep > 0.92 else "start"
        dep_dx = -9 if dep > 0.92 else 9
        lines.append(
            f'<text class="value" x="{x_dep + dep_dx:.1f}" y="{y + 18:.1f}" text-anchor="{dep_anchor}" fill="#b91c1c">{_pct(dep)}</text>'
        )
    lines.extend(
        [
            f'<line x1="{left:.1f}" y1="{axis_y:.1f}" x2="{right:.1f}" y2="{axis_y:.1f}" stroke="#8993a4"/>',
            '<text class="note" x="34" y="632">Rank selection fixes the count (4/20 calibration; 8/40 deployment). Scalar implication counts every task whose maximum score is ≥ the stored numeric cutoff.</text>',
            "</svg>",
        ]
    )
    _atomic_text(OUTPUT_RATES_SVG, "\n".join(lines) + "\n")


def _task_suffix(source_task_id: str) -> str:
    return source_task_id.rsplit("-", 1)[-1]


def _write_timing_svg(timing: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1200, 710
    left, column_gap = 255.0, 151.0
    band_tops = (114.0, 300.0, 486.0)
    band_height = 164.0
    colors = ("#2563eb", "#7c3aed", "#0f766e")
    lines = _svg_header(width, height, "Task-level timing of top-eight signals")
    lines.extend(
        [
            '<text class="title" x="34" y="38">When the top-eight task signals peak</text>',
            '<text class="subtitle" x="34" y="61">Deployment pass one · earliest checkpoint attaining each task’s maximum score</text>',
            '<line x1="34" y1="79" x2="1166" y2="79" stroke="#e6eaf0"/>',
        ]
    )
    for checkpoint in range(1, 7):
        x = left + (checkpoint - 1) * column_gap
        lines.append(
            f'<text class="axis" x="{x:.1f}" y="99" text-anchor="middle">after turn {checkpoint}</text>'
        )
        lines.append(
            f'<line x1="{x:.1f}" y1="108" x2="{x:.1f}" y2="652" stroke="#e5e9ef"/>'
        )
    for method_index, block in enumerate(timing):
        method = str(block["method"])
        top = band_tops[method_index]
        color = colors[method_index]
        lines.append(
            f'<rect x="26" y="{top:.1f}" width="1140" height="{band_height:.1f}" rx="10" fill="{("#f8fafc" if method_index % 2 == 0 else "#fbfbfe")}" stroke="#edf0f4"/>'
        )
        lines.append(
            f'<text class="label" x="42" y="{top + 29:.1f}">{escape(METHOD_LABELS[method])}</text>'
        )
        mean_value = float(block["timing_summary"]["mean_checkpoint"])
        lines.append(
            f'<text class="note" x="42" y="{top + 49:.1f}">mean {mean_value:.2f}</text>'
        )
        by_checkpoint: dict[int, list[Mapping[str, Any]]] = {
            checkpoint: [] for checkpoint in range(1, 7)
        }
        for row in block["selected_top8"]:
            by_checkpoint[int(row["earliest_max_checkpoint"])].append(row)
        for checkpoint, task_rows in by_checkpoint.items():
            x = left + (checkpoint - 1) * column_gap
            for stack_index, row in enumerate(
                sorted(task_rows, key=lambda item: int(item["rank"]))
            ):
                y = top + 25.0 + stack_index * 17.0
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
                )
                task = _task_suffix(str(row["source_task_id"]))
                score = _score_label(float(row["score"]))
                lines.append(
                    f'<text class="value" x="{x + 9:.1f}" y="{y + 4:.1f}" fill="{color}">#{int(row["rank"])} · {escape(task)} ({score})</text>'
                )
    lines.extend(
        [
            '<text class="note" x="34" y="676">Labels are rank · source-task ID (maximum score). Ties use one SHA-256 ordering over task identity only—method and outcome are excluded.</text>',
            '<text class="note" x="34" y="695">Active recompute is measured on its carried trajectory; frozen recompute and trace judge are measured on clean zero-carry shadows.</text>',
            "</svg>",
        ]
    )
    _atomic_text(OUTPUT_TIMING_SVG, "\n".join(lines) + "\n")


def main() -> None:
    code_hash_before = code_tree_hash(REPOSITORY_ROOT / "experiments12")
    _require(
        code_hash_before == EXPECTED_CODE_TREE_SHA256,
        f"code tree changed before diagnostic: {code_hash_before}",
    )

    threshold_payload = _read_json(THRESHOLDS)
    calibration_extract = _read_json(CALIBRATION_EXTRACT)
    calibration_manifest_sha256 = sha256_file(CALIBRATION_MANIFEST)
    calibration_extract_sha256 = sha256_file(CALIBRATION_EXTRACT)
    _require(
        threshold_payload.get("source_extract_sha256") == calibration_extract_sha256,
        "calibration extract differs from frozen threshold provenance",
    )
    _require(
        threshold_payload.get("source_manifest_sha256")
        == calibration_manifest_sha256,
        "calibration manifest differs from frozen threshold provenance",
    )
    _require(
        float(threshold_payload.get("target_firing_rate")) == TARGET_RATE,
        "threshold target rate is not 20%",
    )
    threshold_rows = {
        (row["model"], row["benchmark"], row["method"]): row
        for row in threshold_payload["thresholds"]
    }

    pass_one_layout = RunLayout.for_run(ARTIFACTS, PASS_ONE_RUN_ID)
    pass_one_manifest_sha256 = sha256_file(pass_one_layout.manifest)
    _require(
        pass_one_manifest_sha256 == EXPECTED_PASS_ONE_MANIFEST_SHA256,
        "pass-one manifest differs from the frozen completed run",
    )
    _manifest, cells, trajectories = _validated_materialization(
        pass_one_layout,
        expected_manifest_sha256=EXPECTED_PASS_ONE_MANIFEST_SHA256,
    )
    deployment_traces = signal_traces(
        pass_one_layout,
        cells,
        trajectories,
        split="deployment_pass_one_diagnostic",
    )

    rate_rows: list[dict[str, Any]] = []
    deployment_by_method: dict[str, list[dict[str, Any]]] = {}
    common_sources: set[str] | None = None
    for method in METHODS:
        threshold = threshold_rows.get((MODEL, BENCHMARK, method))
        _require(threshold is not None, f"missing frozen threshold for {method}")
        _require(
            threshold.get("selection_rule") == "task_score_rank_hash_ties",
            f"unexpected calibration selection rule for {method}",
        )
        calibration = _calibration_scores(calibration_extract, method)
        deployment = _deployment_rows(deployment_traces, method)
        _require(len(calibration) == 20, f"expected 20 calibration tasks for {method}")
        _require(len(deployment) == 40, f"expected 40 deployment tasks for {method}")
        sources = {str(row["source_task_id"]) for row in deployment}
        _require(len(sources) == 40, f"deployment source IDs are not unique for {method}")
        if common_sources is None:
            common_sources = sources
        else:
            _require(
                sources == common_sources,
                f"deployment source-task set differs for {method}",
            )
        deployment_by_method[method] = deployment
        scalar = float(threshold["threshold"])
        calibration_fired = sum(float(row["score"]) >= scalar for row in calibration)
        deployment_fired = sum(float(row["score"]) >= scalar for row in deployment)
        calibration_equal = sum(float(row["score"]) == scalar for row in calibration)
        deployment_equal = sum(float(row["score"]) == scalar for row in deployment)
        calibration_rank_count = int(threshold["target_fire_count"])
        deployment_rank_count = math.floor(TARGET_RATE * len(deployment) + 1e-12)
        _require(calibration_rank_count == 4, f"calibration rank count differs for {method}")
        _require(deployment_rank_count == 8, f"deployment rank count differs for {method}")
        rate_rows.append(
            {
                "method": method,
                "stored_scalar_cutoff": scalar,
                "intended_rank_policy": {
                    "calibration_fired_tasks": calibration_rank_count,
                    "calibration_n_tasks": len(calibration),
                    "calibration_firing_rate": calibration_rank_count / len(calibration),
                    "deployment_fired_tasks": deployment_rank_count,
                    "deployment_n_tasks": len(deployment),
                    "deployment_firing_rate": deployment_rank_count / len(deployment),
                    "rule": "top floor(0.20*n) task maxima; hash-break score ties",
                },
                "scalar_implied": {
                    "rule": "task maximum score >= stored scalar cutoff",
                    "calibration": {
                        "fired_tasks": calibration_fired,
                        "n_tasks": len(calibration),
                        "firing_rate": calibration_fired / len(calibration),
                        "tasks_exactly_at_cutoff": calibration_equal,
                    },
                    "deployment_pass_one": {
                        "fired_tasks": deployment_fired,
                        "n_tasks": len(deployment),
                        "firing_rate": deployment_fired / len(deployment),
                        "tasks_exactly_at_cutoff": deployment_equal,
                    },
                },
            }
        )

    timing_rows: list[dict[str, Any]] = []
    for method in TIMING_METHODS:
        selected = _rank_common(deployment_by_method[method], count=8)
        counts = Counter(int(row["earliest_max_checkpoint"]) for row in selected)
        checkpoint_values = [
            int(row["earliest_max_checkpoint"]) for row in selected
        ]
        timing_rows.append(
            {
                "method": method,
                "selection_rule": "top 8 task maxima; common method-independent SHA-256 ties",
                "selected_top8": selected,
                "timing_summary": {
                    "checkpoint_counts": {
                        str(checkpoint): counts.get(checkpoint, 0)
                        for checkpoint in range(1, 7)
                    },
                    "mean_checkpoint": statistics.fmean(checkpoint_values),
                    "median_checkpoint": statistics.median(checkpoint_values),
                },
            }
        )

    validation_payload = _read_json(PASS_ONE_VALIDATION)
    _require(
        validation_payload.get("primary_ready") is True
        and validation_payload.get("trajectory_outputs") == 80
        and validation_payload.get("shadow_outputs") == 40,
        "pass-one validation receipt is not complete",
    )
    payload: dict[str, Any] = {
        "artifact_type": "experiment12_provider_free_threshold_policy_diagnostic",
        "schema_version": 1,
        "analysis_status": "post_hoc_diagnostic_not_confirmatory_outcome_estimate",
        "model": MODEL,
        "benchmark": BENCHMARK,
        "target_firing_rate": TARGET_RATE,
        "definitions": {
            "task_score": "maximum score over checkpoints actionable before the task event",
            "scalar_implication": "fire when task_score >= the stored numeric cutoff",
            "rank_policy": "select exactly floor(target_rate * n_tasks) highest task scores",
            "timing": "earliest actionable checkpoint attaining the task maximum",
            "ecological_paths": "active on carried trajectories; passive on clean zero-carry shadows",
        },
        "common_tie_rule": {
            "namespace": "exp12/common-task-rank-tie/v1",
            "seed": TIE_SEED,
            "identity_fields": ["model", "benchmark", "source_task_id"],
            "excluded_fields": ["method", "outcome", "event_checkpoint"],
            "digest": "SHA-256 over NUL-delimited namespace, seed, and identity fields",
        },
        "provenance": {
            "code_tree_sha256": code_hash_before,
            "threshold_artifact": str(THRESHOLDS.relative_to(REPOSITORY_ROOT)),
            "threshold_artifact_sha256": sha256_file(THRESHOLDS),
            "calibration_extract": str(CALIBRATION_EXTRACT.relative_to(REPOSITORY_ROOT)),
            "calibration_extract_sha256": calibration_extract_sha256,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "deployment_pass_one_run_id": PASS_ONE_RUN_ID,
            "deployment_pass_one_manifest_sha256": pass_one_manifest_sha256,
            "deployment_pass_one_pairs_sha256": sha256_file(pass_one_layout.pairs),
            "deployment_pass_one_validation": str(
                PASS_ONE_VALIDATION.relative_to(REPOSITORY_ROOT)
            ),
            "deployment_pass_one_validation_sha256": sha256_file(PASS_ONE_VALIDATION),
            "provider_calls_made_by_diagnostic": 0,
        },
        "sample_sizes": {
            "calibration_tasks_per_method": 20,
            "deployment_pass_one_tasks_per_method": 40,
            "deployment_trajectories": int(validation_payload["trajectory_outputs"]),
            "deployment_clean_shadows": int(validation_payload["shadow_outputs"]),
        },
        "firing_rate_comparison": rate_rows,
        "task_level_top8_timing": timing_rows,
        "figure_files": [
            str(OUTPUT_RATES_SVG.relative_to(REPOSITORY_ROOT)),
            str(OUTPUT_TIMING_SVG.relative_to(REPOSITORY_ROOT)),
        ],
    }
    _atomic_json(OUTPUT_JSON, payload)
    _write_rates_svg(rate_rows)
    _write_timing_svg(timing_rows)

    code_hash_after = code_tree_hash(REPOSITORY_ROOT / "experiments12")
    _require(
        code_hash_after == EXPECTED_CODE_TREE_SHA256,
        f"code tree changed while generating diagnostic: {code_hash_after}",
    )
    print(
        json.dumps(
            {
                "code_tree_sha256": code_hash_after,
                "json": str(OUTPUT_JSON.relative_to(REPOSITORY_ROOT)),
                "rates_svg": str(OUTPUT_RATES_SVG.relative_to(REPOSITORY_ROOT)),
                "timing_svg": str(OUTPUT_TIMING_SVG.relative_to(REPOSITORY_ROOT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
