"""Build the exploratory active-probe mechanism summary for Experiment 12.

Inputs are the two completed baseline-gate extracts.  The builder independently
reconstructs every active-minus-clean success contrast and its paired task
bootstrap interval from task outcomes before emitting paper artifacts.  The
four arms are treated as distinct mechanisms, not as an ordinal complexity
scale.  This script is provider-free.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import csv
import html
import io
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from experiments12.core.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    sha256_file,
)
from experiments12.manifest12 import RunLayout, code_tree_hash
from experiments12.metrics12 import TaskOutcome, paired_active_effects
from experiments12.probes12 import PROBE_DEFINITIONS
from experiments12.validate12 import validate_run


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments12"
ARTIFACTS = PACKAGE / "data_results" / "runs"
OUTPUT_STEM = PACKAGE / "data_results" / "derived" / "active-probe-mechanism-exploratory-v1"
EXPECTED_CODE_TREE_SHA256 = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)

RUNS = (
    {
        "run_id": "e12-baseline-evolving-allarms-allmodels-v1",
        "benchmark": "evolving_intent_gsm8k",
        "manifest_sha256": (
            "f538166a9b1e657429be547e617822b9160df1b37e496526518b4424a6d3b852"
        ),
        "pairs_sha256": (
            "0266e83f0b134e91135036cc64ea068dd9209223beddf25ee7ec7aa9c6eea9a6"
        ),
        "extract_sha256": (
            "75840685649e87da837dd01a66ad8197a178cb3a8fa19c5e88cd67c638801f51"
        ),
        "validation_sha256": (
            "bd338bc2c6287a94776788df06731c57db2c4633e4dd24250f75aa4db619c214"
        ),
        "models": (
            "gpt-oss-120b",
            "deepseek-v4-flash-0731",
            "qwen3p7-plus",
            "gpt-5.6-luna",
            "gpt-5.6-terra",
        ),
        "n_outcomes": 500,
    },
    {
        "run_id": "e12-baseline-bfcl-allarms-fourmodels-v2",
        "benchmark": "bfcl_multi_turn",
        "manifest_sha256": (
            "eca11658fb6167e0877f4180517c197fbefacc2dffd2d43f6f6530e09e962407"
        ),
        "pairs_sha256": (
            "ece586629a907630c24923df7ca55f0a24f1f8bdd4d4f867f83e9ea7038f855b"
        ),
        "extract_sha256": (
            "6161557d8ecb18c4564f415ef2046be473f1378898ab9611c3eb93e0c9d0e42b"
        ),
        "validation_sha256": (
            "2381423273eca79082cbde12103a3285e6223f6fa0843f387c966114f8f6bd15"
        ),
        "models": (
            "gpt-oss-120b",
            "qwen3p7-plus",
            "gpt-5.6-luna",
            "gpt-5.6-terra",
        ),
        "n_outcomes": 400,
    },
)

ARM_SPECS = (
    {
        "active_arm": "active_name_copy",
        "variant": "current_copy",
        "short_label": "name-copy",
        "description": "copy the current-turn name/code",
        "color": "#0072B2",
    },
    {
        "active_arm": "active_name_recall",
        "variant": "initial_recall",
        "short_label": "name-recall",
        "description": "recall the initial-only name/code",
        "color": "#009E73",
    },
    {
        "active_arm": "active_counter",
        "variant": "stateful_counter",
        "short_label": "counter",
        "description": "update a counter carried across checkpoints",
        "color": "#CC79A7",
    },
    {
        "active_arm": "active_recompute",
        "variant": "recompute",
        "short_label": "recompute",
        "description": "solve fresh current-turn arithmetic",
        "color": "#D55E00",
    },
)
ARM_ORDER = {row["active_arm"]: index for index, row in enumerate(ARM_SPECS)}
ARM_BY_NAME = {row["active_arm"]: row for row in ARM_SPECS}


class MechanismInputError(ValueError):
    """A baseline-gate source is incomplete, changed, or internally inconsistent."""


def _close(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    )


def _source_paths(layout: RunLayout) -> tuple[Path, Path]:
    return (
        layout.results / "extract-baseline-exploratory.json",
        layout.results / "validation-baseline-exploratory.json",
    )


def _validate_source(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_id = str(spec["run_id"])
    layout = RunLayout.for_run(ARTIFACTS, run_id)
    extract_path, validation_path = _source_paths(layout)
    expected_hashes = (
        (layout.manifest, spec["manifest_sha256"], "manifest"),
        (layout.pairs, spec["pairs_sha256"], "pairs"),
        (extract_path, spec["extract_sha256"], "extract"),
        (validation_path, spec["validation_sha256"], "validation"),
    )
    for path, expected, label in expected_hashes:
        if sha256_file(path) != expected:
            raise MechanismInputError(f"{run_id} {label} hash changed")

    report = validate_run(
        layout,
        repository_root=ROOT,
        expected_manifest_sha256=str(spec["manifest_sha256"]),
    )
    if not report.primary_ready or report.errors or report.warnings:
        raise MechanismInputError(f"{run_id} fails strict run validation")
    recorded_validation = read_json(validation_path)
    if (
        recorded_validation.get("primary_ready") is not True
        or recorded_validation.get("errors")
        or recorded_validation.get("warnings")
        or recorded_validation.get("manifest_sha256") != spec["manifest_sha256"]
    ):
        raise MechanismInputError(f"{run_id} recorded validation is not clean")

    manifest = read_json(layout.manifest)
    expected_arms = ["clean", *(row["active_arm"] for row in ARM_SPECS)]
    if (
        manifest.get("stage") != "baseline_gate"
        or manifest.get("operators") != ["none"]
        or manifest.get("arms") != expected_arms
        or tuple(manifest.get("models", ())) != tuple(spec["models"])
        or manifest.get("repository", {}).get("code_tree_sha256")
        != EXPECTED_CODE_TREE_SHA256
    ):
        raise MechanismInputError(f"{run_id} manifest design changed")

    extract = read_json(extract_path)
    if (
        extract.get("run_id") != run_id
        or extract.get("manifest_sha256") != spec["manifest_sha256"]
        or extract.get("stage") != "baseline_gate"
        or extract.get("split") != "baseline_exploratory"
        or extract.get("observer_effect_semantics")
        != "all effects are active minus clean on identical task trajectories; confidence intervals use a paired task bootstrap"
    ):
        raise MechanismInputError(f"{run_id} extract identity/semantics changed")

    raw_outcomes = extract.get("outcomes")
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) != spec["n_outcomes"]:
        raise MechanismInputError(f"{run_id} outcome count changed")
    outcomes = [TaskOutcome(**row) for row in raw_outcomes]
    if any(
        outcome.benchmark != spec["benchmark"] or not outcome.complete
        for outcome in outcomes
    ):
        raise MechanismInputError(f"{run_id} contains invalid outcomes")
    expected_outcome_keys = {
        (outcome.model, outcome.task_id, outcome.arm) for outcome in outcomes
    }
    if len(expected_outcome_keys) != len(outcomes):
        raise MechanismInputError(f"{run_id} duplicates outcome cells")

    raw_table = extract.get("observer_effect_table")
    if not isinstance(raw_table, list):
        raise MechanismInputError(f"{run_id} observer-effect table is invalid")
    source_success_rows = [row for row in raw_table if row.get("metric") == "success"]
    expected_effect_count = len(spec["models"]) * len(ARM_SPECS)
    if len(source_success_rows) != expected_effect_count:
        raise MechanismInputError(f"{run_id} success-effect row count changed")
    source_lookup = {
        (row.get("model"), row.get("benchmark"), row.get("active_arm")): row
        for row in source_success_rows
    }
    if len(source_lookup) != len(source_success_rows):
        raise MechanismInputError(f"{run_id} duplicates success-effect rows")

    effect_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for arm_spec in ARM_SPECS:
        arm = arm_spec["active_arm"]
        selected = [outcome for outcome in outcomes if outcome.arm in {"clean", arm}]
        reconstructed = paired_active_effects(
            selected,
            active_arm=arm,
            clean_arm="clean",
            bootstrap_iterations=2_000,
            confidence=0.95,
            seed=12_012,
        )
        if len(reconstructed) != len(spec["models"]):
            raise MechanismInputError(f"{run_id}/{arm} model coverage changed")
        for effect in reconstructed:
            source = source_lookup.get((effect.model, effect.benchmark, arm))
            if source is None:
                raise MechanismInputError(f"{run_id}/{effect.model}/{arm} lacks source row")
            expected = asdict(effect)
            for field, value in expected.items():
                source_value = source.get(field)
                equal = _close(value, source_value) if isinstance(value, float) else value == source_value
                if not equal:
                    raise MechanismInputError(
                        f"{run_id}/{effect.model}/{arm} reconstructed {field} disagrees"
                    )
            if (
                source.get("effect_definition") != "active_minus_clean"
                or source.get("unit") != "proportion"
                or source.get("favorable_direction") != "higher"
            ):
                raise MechanismInputError(f"{run_id}/{effect.model}/{arm} metadata changed")
            relation = (
                "below_zero"
                if effect.ci_high < 0
                else "above_zero"
                if effect.ci_low > 0
                else "includes_zero"
            )
            effect_rows.append(
                {
                    "analysis_status": "exploratory_baseline_gate",
                    "run_id": run_id,
                    "benchmark": effect.benchmark,
                    "model": effect.model,
                    "mechanism": arm_spec["short_label"],
                    "active_arm": arm,
                    "probe_variant": arm_spec["variant"],
                    "clean_arm": effect.clean_arm,
                    "n_tasks": effect.n_tasks,
                    "clean_success": effect.clean_mean,
                    "active_success": effect.active_mean,
                    "paired_success_effect": effect.effect,
                    "effect_definition": "active_minus_clean",
                    "ci_low": effect.ci_low,
                    "ci_high": effect.ci_high,
                    "confidence": effect.confidence,
                    "bootstrap_iterations": effect.bootstrap_iterations,
                    "bootstrap_seed": effect.bootstrap_seed,
                    "bootstrap_unit": effect.bootstrap_unit,
                    "ci_relation_to_zero": relation,
                }
            )

        by_cell: dict[tuple[str, str], dict[str, TaskOutcome]] = defaultdict(dict)
        for outcome in selected:
            cell = by_cell[(outcome.model, outcome.task_id)]
            if outcome.arm in cell:
                raise MechanismInputError(f"{run_id}/{arm} duplicates a paired task")
            cell[outcome.arm] = outcome
        for (model, task_id), arm_values in sorted(by_cell.items()):
            if set(arm_values) != {"clean", arm}:
                raise MechanismInputError(f"{run_id}/{model}/{task_id}/{arm} is unpaired")
            clean_value = arm_values["clean"].outcome
            active_value = arm_values[arm].outcome
            paired_rows.append(
                {
                    "run_id": run_id,
                    "benchmark": str(spec["benchmark"]),
                    "model": model,
                    "task_id": task_id,
                    "mechanism": arm_spec["short_label"],
                    "active_arm": arm,
                    "clean_success": clean_value,
                    "active_success": active_value,
                    "paired_difference": active_value - clean_value,
                }
            )

    effect_rows.sort(
        key=lambda row: (
            tuple(spec["models"]).index(row["model"]),
            ARM_ORDER[row["active_arm"]],
        )
    )
    paired_rows.sort(
        key=lambda row: (
            tuple(spec["models"]).index(row["model"]),
            ARM_ORDER[row["active_arm"]],
            row["task_id"],
        )
    )
    source_receipt = {
        "run_id": run_id,
        "benchmark": spec["benchmark"],
        "models": list(spec["models"]),
        "manifest_sha256": spec["manifest_sha256"],
        "pairs_sha256": spec["pairs_sha256"],
        "extract_sha256": spec["extract_sha256"],
        "validation_sha256": spec["validation_sha256"],
        "outcomes": len(outcomes),
        "paired_effect_rows": len(effect_rows),
        "strict_validation": {"primary_ready": True, "errors": 0, "warnings": 0},
    }
    return effect_rows, paired_rows, source_receipt


def _mechanism_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARM_SPECS:
        definition = PROBE_DEFINITIONS[arm["variant"]]
        rows.append(
            {
                "active_arm": arm["active_arm"],
                "mechanism": arm["short_label"],
                "probe_variant": arm["variant"],
                "description": arm["description"],
                "protocol_label": definition.label,
                "fixed_output_characters": definition.components.output_length,
                "component_profile": {
                    "copyability": definition.components.copyability,
                    "memory_load": definition.components.memory_load,
                    "reasoning_load": definition.components.reasoning_load,
                    "copy_source": definition.components.copy_source,
                },
                "interpretation": (
                    "multi-dimensional mechanism descriptor; not an ordinal complexity rank"
                ),
            }
        )
    return rows


def _descriptive_summaries(effect_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in effect_rows:
        groups[(row["benchmark"], row["active_arm"])].append(row)
        groups[("all_benchmarks", row["active_arm"])].append(row)
    result: list[dict[str, Any]] = []
    for (benchmark, arm), rows in sorted(
        groups.items(),
        key=lambda item: (
            {"evolving_intent_gsm8k": 0, "bfcl_multi_turn": 1, "all_benchmarks": 2}[
                item[0][0]
            ],
            ARM_ORDER[item[0][1]],
        ),
    ):
        effects = [float(row["paired_success_effect"]) for row in rows]
        result.append(
            {
                "benchmark": benchmark,
                "mechanism": ARM_BY_NAME[arm]["short_label"],
                "active_arm": arm,
                "n_model_strata": len(rows),
                "median_effect": median(effects),
                "minimum_effect": min(effects),
                "maximum_effect": max(effects),
                "negative_point_estimates": sum(value < 0 for value in effects),
                "zero_point_estimates": sum(value == 0 for value in effects),
                "positive_point_estimates": sum(value > 0 for value in effects),
                "intervals_below_zero": sum(
                    row["ci_relation_to_zero"] == "below_zero" for row in rows
                ),
                "intervals_including_zero": sum(
                    row["ci_relation_to_zero"] == "includes_zero" for row in rows
                ),
                "intervals_above_zero": sum(
                    row["ci_relation_to_zero"] == "above_zero" for row in rows
                ),
                "scope": "descriptive aggregation across model strata; not a pooled estimate",
            }
        )
    return result


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _model_label(model: str) -> str:
    return {
        "gpt-oss-120b": "GPT-OSS 120B",
        "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
        "qwen3p7-plus": "Qwen3.7 Plus",
        "gpt-5.6-luna": "GPT-5.6 Luna",
        "gpt-5.6-terra": "GPT-5.6 Terra",
    }.get(model, model)


def _effect_text(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.2f}"


def _figure(
    effect_rows: Sequence[Mapping[str, Any]],
    *,
    source_json_sha256: str,
) -> tuple[str, dict[str, Any]]:
    width = 1500
    height = 790
    margin = 30
    panel_gap = 14
    panel_height = 224
    mechanism_y = (74, 108, 142, 176)
    model_order = {
        "gpt-oss-120b": 0,
        "deepseek-v4-flash-0731": 1,
        "qwen3p7-plus": 2,
        "gpt-5.6-luna": 3,
        "gpt-5.6-terra": 4,
    }
    rows_by_benchmark = {
        benchmark: sorted(
            [row for row in effect_rows if row["benchmark"] == benchmark],
            key=lambda row: (model_order[row["model"]], ARM_ORDER[row["active_arm"]]),
        )
        for benchmark in ("evolving_intent_gsm8k", "bfcl_multi_turn")
    }
    body: list[str] = ['<rect width="100%" height="100%" fill="#FFFFFF"/>']
    body.append(
        '<text x="30" y="37" class="title">Active-probe mechanism screen</text>'
    )
    body.append(
        '<rect x="1160" y="18" width="309" height="28" rx="14" fill="#FFF0F0" stroke="#D64545"/>'
        '<text x="1314.5" y="37" text-anchor="middle" class="badge">EXPLORATORY · BASELINE GATE</text>'
    )
    body.append(
        '<text x="30" y="62" class="subtitle">Paired task-success effect of carried active probes (active − clean); points with paired-task bootstrap 95% CIs; n = 20 tasks per point</text>'
    )
    body.append(
        '<text x="30" y="84" class="guardrail">Mechanisms are multi-dimensional and are not ordered as a complexity scale. Left is worse task success; right is better.</text>'
    )
    legend_x = 30
    for arm in ARM_SPECS:
        body.append(
            f'<circle cx="{legend_x + 6}" cy="111" r="5" fill="{arm["color"]}"/>'
            f'<text x="{legend_x + 17}" y="115" class="legend">{_esc(arm["short_label"])}</text>'
        )
        legend_x += 128
    body.append(
        '<text x="560" y="115" class="small">All probes use the same exact 15-character response envelope.</text>'
    )

    plotted: list[dict[str, Any]] = []
    sections = (
        ("evolving_intent_gsm8k", "Evolving-Intent GSM8K", 148),
        ("bfcl_multi_turn", "BFCL multi-turn", 438),
    )
    for benchmark, benchmark_label, section_y in sections:
        benchmark_rows = rows_by_benchmark[benchmark]
        models = sorted({row["model"] for row in benchmark_rows}, key=model_order.get)
        n_panels = len(models)
        panel_width = (width - 2 * margin - (n_panels - 1) * panel_gap) / n_panels
        if n_panels == 4:
            panel_width = 288
            total_width = n_panels * panel_width + (n_panels - 1) * panel_gap
            start_x = (width - total_width) / 2
        else:
            start_x = margin
        body.append(
            f'<text x="{start_x:.1f}" y="{section_y - 10}" class="benchmark">{_esc(benchmark_label)}</text>'
        )
        for model_index, model in enumerate(models):
            x = start_x + model_index * (panel_width + panel_gap)
            body.append(
                f'<rect x="{x:.1f}" y="{section_y:.1f}" width="{panel_width:.1f}" height="{panel_height}" rx="7" fill="#FAFBFC" stroke="#D9E2EC"/>'
            )
            body.append(
                f'<text x="{x + panel_width / 2:.1f}" y="{section_y + 26}" text-anchor="middle" class="model">{_esc(_model_label(model))}</text>'
            )
            plot_left = x + 82
            plot_right = x + panel_width - 17
            zero_x = (plot_left + plot_right) / 2
            body.append(
                f'<rect x="{plot_left:.1f}" y="{section_y + 47}" width="{zero_x - plot_left:.1f}" height="143" fill="#FFF7F5"/>'
                f'<rect x="{zero_x:.1f}" y="{section_y + 47}" width="{plot_right - zero_x:.1f}" height="143" fill="#F3FAF7"/>'
                f'<line x1="{zero_x:.1f}" y1="{section_y + 43}" x2="{zero_x:.1f}" y2="{section_y + 194}" class="zero"/>'
            )
            model_rows = {
                row["active_arm"]: row
                for row in benchmark_rows
                if row["model"] == model
            }
            if set(model_rows) != set(ARM_ORDER):
                raise MechanismInputError(f"figure lacks {benchmark}/{model} mechanisms")
            for arm_index, arm in enumerate(ARM_SPECS):
                row = model_rows[arm["active_arm"]]
                y = section_y + mechanism_y[arm_index]

                def xpos(value: float) -> float:
                    if not -1.0 <= value <= 1.0:
                        raise MechanismInputError("success effect/CI falls outside [-1, 1]")
                    return plot_left + (value + 1.0) * (plot_right - plot_left) / 2.0

                low_x = xpos(float(row["ci_low"]))
                high_x = xpos(float(row["ci_high"]))
                point_x = xpos(float(row["paired_success_effect"]))
                body.append(
                    f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" class="arm" style="fill:{arm["color"]}">{_esc(arm["short_label"])}</text>'
                    f'<line x1="{low_x:.1f}" y1="{y:.1f}" x2="{high_x:.1f}" y2="{y:.1f}" stroke="{arm["color"]}" stroke-width="3" stroke-linecap="round"/>'
                    f'<line x1="{low_x:.1f}" y1="{y - 4:.1f}" x2="{low_x:.1f}" y2="{y + 4:.1f}" stroke="{arm["color"]}"/>'
                    f'<line x1="{high_x:.1f}" y1="{y - 4:.1f}" x2="{high_x:.1f}" y2="{y + 4:.1f}" stroke="{arm["color"]}"/>'
                    f'<circle cx="{point_x:.1f}" cy="{y:.1f}" r="6" fill="{arm["color"]}" stroke="#FFFFFF" stroke-width="1.5"><title>{_esc(model)} · {_esc(arm["short_label"])}: {_effect_text(float(row["paired_success_effect"]))} [{float(row["ci_low"]):+.3f}, {float(row["ci_high"]):+.3f}]</title></circle>'
                )
                text_anchor = "end" if float(row["paired_success_effect"]) < 0 else "start"
                text_x = point_x - 9 if text_anchor == "end" else point_x + 9
                body.append(
                    f'<text x="{text_x:.1f}" y="{y - 8:.1f}" text-anchor="{text_anchor}" class="effect">{_effect_text(float(row["paired_success_effect"]))}</text>'
                )
                plotted.append(dict(row))
            axis_y = section_y + 207
            body.append(
                f'<line x1="{plot_left:.1f}" y1="{axis_y:.1f}" x2="{plot_right:.1f}" y2="{axis_y:.1f}" class="axis"/>'
            )
            for value, label in ((-1.0, "−1"), (0.0, "0"), (1.0, "+1")):
                tick_x = plot_left + (value + 1.0) * (plot_right - plot_left) / 2.0
                body.append(
                    f'<line x1="{tick_x:.1f}" y1="{axis_y:.1f}" x2="{tick_x:.1f}" y2="{axis_y + 4:.1f}" class="axis"/>'
                    f'<text x="{tick_x:.1f}" y="{axis_y + 15:.1f}" text-anchor="middle" class="tick">{label}</text>'
                )

    body.append(
        '<text x="30" y="744" class="footnote">Intervals are unadjusted exploratory 95% bootstrap CIs (2,000 paired-task draws; seed 12,012). No multiplicity correction or monotonic trend test was prespecified.</text>'
    )
    body.append(
        '<text x="30" y="764" class="footnote">DeepSeek appears only in Evolving-Intent; the completed BFCL replacement screen contains four models. Use this figure for mechanism generation, not confirmation.</text>'
    )
    style = """
      text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill: #1F2933; }
      .title { font-size: 24px; font-weight: 760; letter-spacing: -0.25px; }
      .subtitle { font-size: 13px; fill: #52606D; }
      .guardrail { font-size: 12px; font-weight: 650; fill: #7B341E; }
      .badge { font-size: 11px; font-weight: 750; fill: #B42318; letter-spacing: 0.6px; }
      .legend { font-size: 11px; font-weight: 650; fill: #334E68; }
      .small, .footnote { font-size: 10.5px; fill: #616E7C; }
      .benchmark { font-size: 15px; font-weight: 760; fill: #102A43; }
      .model { font-size: 12px; font-weight: 700; fill: #243B53; }
      .arm { font-size: 9.5px; font-weight: 700; }
      .effect { font-size: 9px; font-weight: 700; fill: #243B53; }
      .tick { font-size: 9px; fill: #7B8794; }
      .axis { stroke: #829AB1; stroke-width: 1; }
      .zero { stroke: #334E68; stroke-width: 1.4; stroke-dasharray: 4 3; }
    """
    description = (
        "Exploratory small-multiple forest plot of paired active-minus-clean task "
        "success for four carried probe mechanisms across Evolving-Intent GSM8K "
        "and BFCL models, with 95 percent paired-task bootstrap intervals."
    )
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        '<title id="title">Active-probe mechanism screen</title>\n'
        f'<desc id="desc">{_esc(description)}</desc>\n'
        f'<defs><style>{style}</style></defs>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    sidecar = {
        "schema_version": 1,
        "figure_type": "exploratory_active_probe_mechanism_forest",
        "title": "Active-probe mechanism screen",
        "description": description,
        "source_json_sha256": source_json_sha256,
        "analysis_status": "exploratory_baseline_gate",
        "effect_definition": "active_minus_clean task success",
        "x_axis": {"minimum": -1.0, "maximum": 1.0, "unit": "proportion"},
        "interval": {
            "confidence": 0.95,
            "bootstrap_unit": "paired task",
            "bootstrap_iterations": 2_000,
            "bootstrap_seed": 12_012,
            "multiplicity_adjusted": False,
        },
        "interpretation_guardrail": (
            "mechanisms are multi-dimensional and are not an ordinal complexity scale"
        ),
        "rows": plotted,
        "width": width,
        "height": height,
    }
    return svg, sidecar


def main() -> None:
    live_hash_before = code_tree_hash(PACKAGE)
    if live_hash_before != EXPECTED_CODE_TREE_SHA256:
        raise MechanismInputError(
            f"frozen code tree changed before mechanism extraction: {live_hash_before}"
        )

    effect_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for spec in RUNS:
        run_effects, run_pairs, source_receipt = _validate_source(spec)
        effect_rows.extend(run_effects)
        paired_rows.extend(run_pairs)
        source_receipts.append(source_receipt)
    benchmark_order = {"evolving_intent_gsm8k": 0, "bfcl_multi_turn": 1}
    model_order = {
        "gpt-oss-120b": 0,
        "deepseek-v4-flash-0731": 1,
        "qwen3p7-plus": 2,
        "gpt-5.6-luna": 3,
        "gpt-5.6-terra": 4,
    }
    effect_rows.sort(
        key=lambda row: (
            benchmark_order[row["benchmark"]],
            model_order[row["model"]],
            ARM_ORDER[row["active_arm"]],
        )
    )
    paired_rows.sort(
        key=lambda row: (
            benchmark_order[row["benchmark"]],
            model_order[row["model"]],
            ARM_ORDER[row["active_arm"]],
            row["task_id"],
        )
    )
    if len(effect_rows) != 36 or len(paired_rows) != 720:
        raise MechanismInputError("unexpected combined effect/paired-task row count")
    if any(row["n_tasks"] != 20 for row in effect_rows):
        raise MechanismInputError("every mechanism contrast must use 20 paired tasks")

    summaries = _descriptive_summaries(effect_rows)
    payload = {
        "schema_version": 1,
        "artifact": "active_probe_mechanism_exploratory_summary",
        "analysis_status": "exploratory_baseline_gate",
        "code_tree_sha256": live_hash_before,
        "estimand": (
            "paired difference in official end-task success: carried active probe arm "
            "minus clean arm on identical task IDs"
        ),
        "inference": {
            "confidence_intervals": (
                "unadjusted 95% paired-task bootstrap intervals, 2,000 draws, seed 12,012"
            ),
            "multiplicity_adjustment": None,
            "scope": "hypothesis-generating mechanism screen; not confirmatory evidence",
            "guardrail": (
                "the four mechanisms are multi-dimensional; their display order is the "
                "protocol catalog order, not a validated complexity ranking; no monotonic "
                "trend is imposed or claimed"
            ),
        },
        "coverage_caveat": (
            "DeepSeek is available only in the completed Evolving-Intent screen; the "
            "completed BFCL replacement screen contains four models"
        ),
        "mechanisms": _mechanism_catalog(),
        "source_runs": source_receipts,
        "counts": {
            "benchmarks": 2,
            "benchmark_model_strata": 9,
            "mechanisms": 4,
            "paired_effect_rows": len(effect_rows),
            "paired_task_rows": len(paired_rows),
            "tasks_per_effect": 20,
        },
        "descriptive_summaries": summaries,
        "paired_effect_rows": effect_rows,
        "paired_task_rows": paired_rows,
    }

    json_path = OUTPUT_STEM.with_suffix(".json")
    effect_csv_path = OUTPUT_STEM.with_suffix(".csv")
    paired_csv_path = OUTPUT_STEM.with_suffix(".paired.csv")
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    svg_data_path = OUTPUT_STEM.with_suffix(".svg.data.json")
    receipt_path = OUTPUT_STEM.with_suffix(".receipt.json")

    json_sha = atomic_write_json(json_path, payload)
    effect_csv_sha = atomic_write_bytes(
        effect_csv_path, _csv_bytes(effect_rows, tuple(effect_rows[0].keys()))
    )
    paired_csv_sha = atomic_write_bytes(
        paired_csv_path, _csv_bytes(paired_rows, tuple(paired_rows[0].keys()))
    )
    svg, sidecar = _figure(effect_rows, source_json_sha256=json_sha)
    svg_sha = atomic_write_bytes(svg_path, svg.encode("utf-8"))
    svg_data_sha = atomic_write_json(svg_data_path, sidecar)

    live_hash_after = code_tree_hash(PACKAGE)
    if live_hash_after != live_hash_before:
        raise MechanismInputError(
            f"frozen code tree changed during mechanism extraction: {live_hash_after}"
        )
    receipt = {
        "schema_version": 1,
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256_file(__file__),
        "provider_calls_made": 0,
        "code_tree_sha256_before": live_hash_before,
        "code_tree_sha256_after": live_hash_after,
        "outputs": {
            str(json_path.relative_to(ROOT)): json_sha,
            str(effect_csv_path.relative_to(ROOT)): effect_csv_sha,
            str(paired_csv_path.relative_to(ROOT)): paired_csv_sha,
            str(svg_path.relative_to(ROOT)): svg_sha,
            str(svg_data_path.relative_to(ROOT)): svg_data_sha,
        },
    }
    receipt_sha = atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "code_tree_sha256": live_hash_after,
                "paired_effect_rows": len(effect_rows),
                "paired_task_rows": len(paired_rows),
                "receipt": str(receipt_path.relative_to(ROOT)),
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
