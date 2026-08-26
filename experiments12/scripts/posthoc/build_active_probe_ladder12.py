"""Build a provider-free active-probe observer-effect synthesis.

The exploratory n=20 mechanism screens and powered n=56 recompute contrasts
remain visibly and analytically separate.  Sign counts are descriptive counts
of model/benchmark strata; no effects are pooled across shared tasks or models.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.manifest12 import RunLayout, code_tree_hash
from experiments12.validate12 import validate_run


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments12"
ARTIFACTS = PACKAGE / "data_results" / "runs"
OUTPUT_STEM = PACKAGE / "data_results" / "derived" / "active-probe-ladder-confirmatory-v1"
EXPECTED_CODE_HASH = "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"

PROBES = (
    {
        "arm": "active_name_copy",
        "short_label": "Copy current\ncode",
        "probe": "current copy",
        "burden_axis": "direct copy",
        "order": 0,
    },
    {
        "arm": "active_name_recall",
        "short_label": "Recall initial\ncode",
        "probe": "initial recall",
        "burden_axis": "memory",
        "order": 1,
    },
    {
        "arm": "active_counter",
        "short_label": "Update carried\ncounter",
        "probe": "stateful counter",
        "burden_axis": "memory + update",
        "order": 2,
    },
    {
        "arm": "active_recompute",
        "short_label": "Recompute\narithmetic",
        "probe": "recompute",
        "burden_axis": "reasoning",
        "order": 3,
    },
)
PROBE_BY_ARM = {row["arm"]: row for row in PROBES}
PROBE_ORDER = tuple(row["arm"] for row in PROBES)

SOURCES = (
    {
        "study": "exploratory_mechanism",
        "run_id": "e12-baseline-evolving-allarms-allmodels-v1",
        "benchmark": "evolving_intent_gsm8k",
        "manifest_sha256": "f538166a9b1e657429be547e617822b9160df1b37e496526518b4424a6d3b852",
        "pairs_sha256": "0266e83f0b134e91135036cc64ea068dd9209223beddf25ee7ec7aa9c6eea9a6",
        "extract_name": "extract-baseline-exploratory.json",
        "extract_sha256": "75840685649e87da837dd01a66ad8197a178cb3a8fa19c5e88cd67c638801f51",
        "n_models": 5,
        "n_tasks": 20,
        "arms": ("clean", *PROBE_ORDER),
        "manifest_stage": "baseline_gate",
        "extract_stage": "baseline_gate",
        "split": "baseline_exploratory",
    },
    {
        "study": "exploratory_mechanism",
        "run_id": "e12-baseline-bfcl-allarms-fourmodels-v2",
        "benchmark": "bfcl_multi_turn",
        "manifest_sha256": "eca11658fb6167e0877f4180517c197fbefacc2dffd2d43f6f6530e09e962407",
        "pairs_sha256": "ece586629a907630c24923df7ca55f0a24f1f8bdd4d4f867f83e9ea7038f855b",
        "extract_name": "extract-baseline-exploratory.json",
        "extract_sha256": "6161557d8ecb18c4564f415ef2046be473f1378898ab9611c3eb93e0c9d0e42b",
        "n_models": 4,
        "n_tasks": 20,
        "arms": ("clean", *PROBE_ORDER),
        "manifest_stage": "baseline_gate",
        "extract_stage": "baseline_gate",
        "split": "baseline_exploratory",
    },
    {
        "study": "confirmatory_powered",
        "run_id": "e12-confirmatory-evolving-core-v2",
        "benchmark": "evolving_intent_gsm8k",
        "manifest_sha256": "b0be97ed6b01c9c3b003a5900b8e596772fffe3c68c4b9cdd47f46ff329c7056",
        "pairs_sha256": "ccb98c678dc0d9ff9caee539ccd9859aa406abffc16b4bb9eaaf0abfd0bb6a6c",
        "extract_name": "extract-confirmatory.json",
        "extract_sha256": "26e1a7ff96cad026f1cabf35375053032c0e57e133f5834772a480754d1c23db",
        "n_models": 4,
        "n_tasks": 56,
        "arms": ("clean", "active_recompute"),
        "manifest_stage": "confirmatory",
        "extract_stage": "confirmatory",
        "split": "confirmatory",
    },
    {
        "study": "confirmatory_powered",
        "run_id": "e12-confirmatory-bfcl-core-v3",
        "benchmark": "bfcl_multi_turn",
        "manifest_sha256": "551a4574a1fe502c9304feff46c956ffb6b19a96af84d54d3e18d3a60fd919c3",
        "pairs_sha256": "3f2802f6f7471a65f758b0a8c60fc60a5a0334e906732c1f1560c1b31e990be4",
        "extract_name": "extract-confirmatory.json",
        "extract_sha256": "48398ece77a0d2800975f13fa1f11db723ce055e8c1a1dd0098ebe802a14a927",
        "n_models": 3,
        "n_tasks": 56,
        "arms": ("clean", "active_recompute"),
        "manifest_stage": "confirmatory",
        "extract_stage": "confirmatory",
        "split": "confirmatory",
    },
)


class LadderInputError(ValueError):
    """A frozen source or statistical reconstruction is inconsistent."""


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise LadderInputError("cannot take a quantile of no values")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_index(
    seed: int,
    model: str,
    benchmark: str,
    iteration: int,
    draw: int,
    population: int,
) -> int:
    material = f"exp12/task-bootstrap/v1\0{seed}\0{model}\0{benchmark}\0{iteration}\0{draw}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % population


def _recompute_effect(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    model: str,
    benchmark: str,
    active_arm: str,
    n_tasks: int,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    tasks: dict[str, dict[str, float]] = defaultdict(dict)
    for row in outcomes:
        if row.get("model") != model or row.get("benchmark") != benchmark:
            continue
        arm = row.get("arm")
        if arm not in {"clean", active_arm}:
            continue
        if row.get("complete") is not True or row.get("outcome") not in {0.0, 1.0}:
            raise LadderInputError(f"incomplete/non-binary outcome: {model}/{benchmark}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise LadderInputError("outcome has invalid task ID")
        if arm in tasks[task_id]:
            raise LadderInputError(f"duplicate outcome: {model}/{task_id}/{arm}")
        tasks[task_id][str(arm)] = float(row["outcome"])
    if len(tasks) != n_tasks or any(set(arms) != {"clean", active_arm} for arms in tasks.values()):
        raise LadderInputError(f"unpaired outcome slice: {model}/{benchmark}/{active_arm}")
    ordered = [tasks[task_id] for task_id in sorted(tasks)]
    clean = [row["clean"] for row in ordered]
    active = [row[active_arm] for row in ordered]
    differences = [active_value - clean_value for clean_value, active_value in zip(clean, active)]
    bootstrap = sorted(
        fmean(
            differences[
                _bootstrap_index(seed, model, benchmark, iteration, draw, n_tasks)
            ]
            for draw in range(n_tasks)
        )
        for iteration in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "n_tasks": n_tasks,
        "clean_mean": fmean(clean),
        "active_mean": fmean(active),
        "effect": fmean(differences),
        "ci_low": _quantile(bootstrap, tail),
        "ci_high": _quantile(bootstrap, 1.0 - tail),
    }


def _source_outcomes_from_raw(
    layout: RunLayout,
    *,
    extract_outcomes: Sequence[Mapping[str, Any]],
    expected_cells: int,
) -> set[str]:
    cells = read_jsonl(layout.pairs)
    if len(cells) != expected_cells:
        raise LadderInputError(f"{layout.root.name} raw cell count changed")
    raw: dict[tuple[str, str, str, str], float] = {}
    task_identities: set[str] = set()
    for cell in cells:
        pair = cell.get("pair_key")
        if not isinstance(pair, Mapping):
            raise LadderInputError("pair manifest row lacks pair_key")
        task_id = f"{pair['task_id']}/r{pair['replicate_id']}"
        task_identities.add(task_id)
        trajectory = read_json(layout.trajectories / f"{cell['cell_id']}.json")
        success = trajectory.get("evaluation", {}).get("success")
        if not isinstance(success, bool) or trajectory.get("complete") is not True:
            raise LadderInputError(f"raw trajectory lacks success: {cell['cell_id']}")
        key = (str(pair["model"]), str(pair["domain"]), task_id, str(cell["arm"]))
        if key in raw:
            raise LadderInputError(f"duplicate raw outcome: {key}")
        raw[key] = float(success)
    extracted: dict[tuple[str, str, str, str], float] = {}
    for row in extract_outcomes:
        key = (str(row["model"]), str(row["benchmark"]), str(row["task_id"]), str(row["arm"]))
        if key in extracted or row.get("complete") is not True:
            raise LadderInputError(f"invalid extracted outcome: {key}")
        extracted[key] = float(row["outcome"])
    if extracted != raw:
        raise LadderInputError(f"{layout.root.name} extract outcomes disagree with raw trajectories")
    return task_identities


def _effect_sign(value: float) -> str:
    if value < -1e-12:
        return "negative"
    if value > 1e-12:
        return "positive"
    return "zero"


def _ci_sign(low: float, high: float) -> str:
    if high < 0.0:
        return "negative"
    if low > 0.0:
        return "positive"
    return "includes_zero"


def _extract_source(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    run_id = str(spec["run_id"])
    layout = RunLayout.for_run(ARTIFACTS, run_id)
    extract_path = layout.results / str(spec["extract_name"])
    if sha256_file(layout.manifest) != spec["manifest_sha256"]:
        raise LadderInputError(f"{run_id} manifest hash changed")
    if sha256_file(layout.pairs) != spec["pairs_sha256"]:
        raise LadderInputError(f"{run_id} pairs hash changed")
    if sha256_file(extract_path) != spec["extract_sha256"]:
        raise LadderInputError(f"{run_id} extract hash changed")
    validation = validate_run(
        layout,
        repository_root=ROOT,
        expected_manifest_sha256=str(spec["manifest_sha256"]),
    )
    if not validation.primary_ready or validation.errors or validation.warnings:
        raise LadderInputError(f"{run_id} does not pass strict validation")
    manifest = read_json(layout.manifest)
    extract = read_json(extract_path)
    if (
        manifest.get("stage") != spec["manifest_stage"]
        or manifest.get("arms") != list(spec["arms"])
        or manifest.get("repository", {}).get("code_tree_sha256") != EXPECTED_CODE_HASH
        or extract.get("run_id") != run_id
        or extract.get("stage") != spec["extract_stage"]
        or extract.get("split") != spec["split"]
        or extract.get("manifest_sha256") != spec["manifest_sha256"]
        or extract.get("observer_effect_semantics")
        != "all effects are active minus clean on identical task trajectories; confidence intervals use a paired task bootstrap"
    ):
        raise LadderInputError(f"{run_id} frozen study contract changed")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != spec["n_models"]:
        raise LadderInputError(f"{run_id} model count changed")
    outcomes = extract.get("outcomes")
    if not isinstance(outcomes, list):
        raise LadderInputError(f"{run_id} outcomes are invalid")
    expected_cells = int(spec["n_models"]) * int(spec["n_tasks"]) * len(spec["arms"])
    if len(outcomes) != expected_cells:
        raise LadderInputError(f"{run_id} extracted outcome count changed")
    task_identities = _source_outcomes_from_raw(
        layout,
        extract_outcomes=outcomes,
        expected_cells=expected_cells,
    )
    if len(task_identities) != spec["n_tasks"]:
        raise LadderInputError(f"{run_id} unique task count changed")

    table = extract.get("observer_effect_table")
    if not isinstance(table, list):
        raise LadderInputError(f"{run_id} effect table is invalid")
    success_rows = [row for row in table if row.get("metric") == "success"]
    active_arms = set(spec["arms"]) - {"clean"}
    if len(success_rows) != len(models) * len(active_arms):
        raise LadderInputError(f"{run_id} success effect count changed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in success_rows:
        model = source.get("model")
        arm = source.get("active_arm")
        if model not in models or arm not in active_arms or source.get("benchmark") != spec["benchmark"]:
            raise LadderInputError(f"{run_id} unexpected effect slice")
        key = (str(model), str(arm))
        if key in seen:
            raise LadderInputError(f"{run_id} duplicate effect slice: {key}")
        seen.add(key)
        for field, expected in (
            ("clean_arm", "clean"),
            ("effect_definition", "active_minus_clean"),
            ("unit", "proportion"),
            ("favorable_direction", "higher"),
            ("bootstrap_unit", "task"),
            ("n_tasks", spec["n_tasks"]),
            ("bootstrap_iterations", 2_000),
            ("bootstrap_seed", 12_012),
            ("confidence", 0.95),
        ):
            if source.get(field) != expected:
                raise LadderInputError(f"{run_id}/{model}/{arm} changed {field}")
        recomputed = _recompute_effect(
            outcomes,
            model=str(model),
            benchmark=str(spec["benchmark"]),
            active_arm=str(arm),
            n_tasks=int(spec["n_tasks"]),
            iterations=2_000,
            seed=12_012,
            confidence=0.95,
        )
        if any(not _close(recomputed[field], source.get(field)) for field in recomputed):
            raise LadderInputError(f"{run_id}/{model}/{arm} effect reconstruction failed")
        probe = PROBE_BY_ARM[str(arm)]
        effect = float(source["effect"])
        ci_low = float(source["ci_low"])
        ci_high = float(source["ci_high"])
        rows.append(
            {
                "study": spec["study"],
                "inference_status": (
                    "exploratory_mechanism_screen"
                    if spec["study"] == "exploratory_mechanism"
                    else "powered_prespecified_recompute"
                ),
                "run_id": run_id,
                "benchmark": spec["benchmark"],
                "model": model,
                "active_arm": arm,
                "probe": probe["probe"],
                "burden_axis": probe["burden_axis"],
                "display_order": probe["order"],
                "n_tasks": source["n_tasks"],
                "clean_success": source["clean_mean"],
                "active_success": source["active_mean"],
                "effect": effect,
                "effect_percentage_points": effect * 100.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_percentage_points": ci_low * 100.0,
                "ci_high_percentage_points": ci_high * 100.0,
                "point_sign": _effect_sign(effect),
                "strict_ci_sign": _ci_sign(ci_low, ci_high),
                "ci_excludes_zero": _ci_sign(ci_low, ci_high) != "includes_zero",
                "confidence": source["confidence"],
                "bootstrap_iterations": source["bootstrap_iterations"],
                "bootstrap_seed": source["bootstrap_seed"],
                "bootstrap_unit": source["bootstrap_unit"],
            }
        )
    rows.sort(key=lambda row: (row["benchmark"], row["model"], row["display_order"]))
    inventory = {
        "study": spec["study"],
        "run_id": run_id,
        "benchmark": spec["benchmark"],
        "manifest_sha256": spec["manifest_sha256"],
        "pairs_sha256": spec["pairs_sha256"],
        "extract_path": str(extract_path.relative_to(ROOT)),
        "extract_sha256": spec["extract_sha256"],
        "models": models,
        "n_tasks_per_model_arm": spec["n_tasks"],
        "task_identity_count": len(task_identities),
        "task_identities_sha256": sha256_json(sorted(task_identities)),
        "raw_outcomes_verified": len(outcomes),
        "strict_validation": {"primary_ready": True, "errors": 0, "warnings": 0},
    }
    return rows, inventory, task_identities


def _sign_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    point = Counter(str(row["point_sign"]) for row in rows)
    interval = Counter(str(row["strict_ci_sign"]) for row in rows)
    return {
        "n_strata": len(rows),
        "point_sign_counts": {
            "negative": point["negative"],
            "zero": point["zero"],
            "positive": point["positive"],
        },
        "strict_ci_sign_counts": {
            "negative": interval["negative"],
            "includes_zero": interval["includes_zero"],
            "positive": interval["positive"],
        },
        "semantics": "descriptive counts of model/benchmark strata; no effect pooling or independence claim",
    }


def _monotonicity(exploratory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in exploratory:
        grouped[(str(row["benchmark"]), str(row["model"]))][str(row["active_arm"])] = row
    strata = []
    step_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    worst_counts: Counter[str] = Counter()
    for (benchmark, model), methods in sorted(grouped.items()):
        if set(methods) != set(PROBE_ORDER):
            raise LadderInputError(f"incomplete exploratory ladder: {benchmark}/{model}")
        effects = [float(methods[arm]["effect"]) for arm in PROBE_ORDER]
        steps = [effects[index + 1] - effects[index] for index in range(len(effects) - 1)]
        step_labels = []
        for step in steps:
            if step < -1e-12:
                label = "worsening"
            elif step > 1e-12:
                label = "improving"
            else:
                label = "tie"
            step_labels.append(label)
            step_counts[label] += 1
        if all(label == "tie" for label in step_labels):
            pattern = "flat"
        elif all(label in {"worsening", "tie"} for label in step_labels):
            pattern = "monotone_worsening"
        elif all(label in {"improving", "tie"} for label in step_labels):
            pattern = "monotone_improving"
        else:
            pattern = "mixed_direction"
        pattern_counts[pattern] += 1
        minimum = min(effects)
        worst = [arm for arm, effect in zip(PROBE_ORDER, effects) if _close(effect, minimum)]
        for arm in worst:
            worst_counts[arm] += 1
        strata.append(
            {
                "benchmark": benchmark,
                "model": model,
                "ordered_arms": list(PROBE_ORDER),
                "effects": effects,
                "adjacent_changes": steps,
                "adjacent_labels": step_labels,
                "pattern": pattern,
                "matches_nonflat_monotone_worsening": pattern == "monotone_worsening",
                "worst_effect": minimum,
                "worst_arms_with_ties": worst,
            }
        )
    return {
        "display_order": list(PROBE_ORDER),
        "order_semantics": (
            "conceptual burden order (copy, memory, memory+update, reasoning), not a "
            "validated scalar dose; all formats have fixed matched output length"
        ),
        "hypothesized_pattern": "success effect becomes non-increasing (more harmful) along the display order",
        "n_model_benchmark_strata": len(strata),
        "pattern_counts": {
            "monotone_worsening": pattern_counts["monotone_worsening"],
            "mixed_direction": pattern_counts["mixed_direction"],
            "monotone_improving": pattern_counts["monotone_improving"],
            "flat": pattern_counts["flat"],
        },
        "adjacent_step_counts": {
            "worsening": step_counts["worsening"],
            "improving": step_counts["improving"],
            "tie": step_counts["tie"],
        },
        "worst_probe_counts_with_ties": dict(sorted(worst_counts.items())),
        "strata": strata,
        "conclusion": (
            "Only one of nine exploratory strata shows a non-flat monotone worsening "
            "pattern; the mechanism screen does not support a general complexity-dose rule."
        ),
    }


def _cross_stage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exploratory = {
        (str(row["benchmark"]), str(row["model"])): row
        for row in rows
        if row["study"] == "exploratory_mechanism" and row["active_arm"] == "active_recompute"
    }
    confirmatory = {
        (str(row["benchmark"]), str(row["model"])): row
        for row in rows
        if row["study"] == "confirmatory_powered"
    }
    shared = sorted(set(exploratory) & set(confirmatory))
    comparisons = []
    counts: Counter[str] = Counter()
    for key in shared:
        first = exploratory[key]
        second = confirmatory[key]
        first_sign = str(first["point_sign"])
        second_sign = str(second["point_sign"])
        if "zero" in {first_sign, second_sign}:
            status = "zero_in_either_stage"
        elif first_sign == second_sign:
            status = "same_point_sign"
        else:
            status = "point_sign_flip"
        counts[status] += 1
        comparisons.append(
            {
                "benchmark": key[0],
                "model": key[1],
                "exploratory_n": first["n_tasks"],
                "exploratory_effect": first["effect"],
                "confirmatory_n": second["n_tasks"],
                "confirmatory_effect": second["effect"],
                "status": status,
            }
        )
    return {
        "shared_model_benchmark_strata": len(shared),
        "counts": {
            "same_point_sign": counts["same_point_sign"],
            "point_sign_flip": counts["point_sign_flip"],
            "zero_in_either_stage": counts["zero_in_either_stage"],
        },
        "comparisons": comparisons,
        "caution": (
            "The n=20 and n=56 task sets are disjoint; this is a stability diagnostic, "
            "not a pooled estimate or same-task replication."
        ),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pretty_model(model: str) -> str:
    return {
        "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
        "gpt-oss-120b": "GPT-OSS 120B",
        "qwen3p7-plus": "Qwen3P7 Plus",
        "gpt-5.6-luna": "GPT-5.6 Luna",
        "gpt-5.6-terra": "GPT-5.6 Terra",
    }.get(model, model)


def _pretty_benchmark(benchmark: str) -> str:
    return {
        "evolving_intent_gsm8k": "Evolving-Intent GSM8K",
        "bfcl_multi_turn": "BFCL multi-turn",
    }.get(benchmark, benchmark)


def _figure_label(benchmark: str, model: str) -> str:
    benchmark_label = {
        "evolving_intent_gsm8k": "Evolving",
        "bfcl_multi_turn": "BFCL",
    }.get(benchmark, benchmark)
    model_label = {
        "deepseek-v4-flash-0731": "DeepSeek V4",
        "gpt-oss-120b": "GPT-OSS 120B",
        "qwen3p7-plus": "Qwen3P7+",
        "gpt-5.6-luna": "GPT-5.6 Luna",
        "gpt-5.6-terra": "GPT-5.6 Terra",
    }.get(model, model)
    return f"{benchmark_label} · {model_label}"


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int]:
    benchmark = {"evolving_intent_gsm8k": 0, "bfcl_multi_turn": 1}[str(row["benchmark"])]
    model = {
        "deepseek-v4-flash-0731": 0,
        "gpt-oss-120b": 1,
        "qwen3p7-plus": 2,
        "gpt-5.6-luna": 3,
        "gpt-5.6-terra": 4,
    }[str(row["model"])]
    return benchmark, model


def _effect_fill(effect: float, *, maximum: float = 0.65) -> tuple[str, str]:
    strength = min(abs(effect) / maximum, 1.0)
    base = (249, 250, 251)
    target = (213, 94, 0) if effect < 0 else (0, 114, 178) if effect > 0 else (180, 180, 180)
    mix = 0.12 + 0.76 * strength if effect else 0.08
    rgb = tuple(round(a * (1.0 - mix) + b * mix) for a, b in zip(base, target))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", "#FFFFFF" if strength > 0.62 else "#172B4D"


def _figure(
    rows: Sequence[Mapping[str, Any]],
    monotonicity: Mapping[str, Any],
    *,
    source_json_sha256: str,
) -> tuple[str, dict[str, Any]]:
    exploratory = sorted(
        [row for row in rows if row["study"] == "exploratory_mechanism"],
        key=lambda row: (*_row_sort_key(row), int(row["display_order"])),
    )
    confirmatory = sorted(
        [row for row in rows if row["study"] == "confirmatory_powered"],
        key=_row_sort_key,
    )
    exploratory_strata = sorted(
        {(str(row["benchmark"]), str(row["model"])) for row in exploratory},
        key=lambda item: _row_sort_key({"benchmark": item[0], "model": item[1]}),
    )
    lookup = {(row["benchmark"], row["model"], row["active_arm"]): row for row in exploratory}
    pattern_lookup = {
        (row["benchmark"], row["model"]): row["pattern"]
        for row in monotonicity["strata"]
    }
    pattern_label = {
        "monotone_worsening": "worsens",
        "mixed_direction": "mixed",
        "monotone_improving": "opposite",
        "flat": "flat",
    }

    width, height = 1480, 1305
    body: list[str] = ['<rect width="100%" height="100%" fill="#FFFFFF"/>']
    body.append('<text x="34" y="46" class="title">Observer effect of carried active probes</text>')
    body.append('<text x="34" y="78" class="subtitle">Paired change in task success (active − clean); negative values mean the carried probe harmed success</text>')
    body.append('<rect x="34" y="94" width="18" height="18" rx="2" fill="#D55E00"/><text x="61" y="110" class="legend">harm</text>')
    body.append('<rect x="125" y="94" width="18" height="18" rx="2" fill="#0072B2"/><text x="152" y="110" class="legend">benefit</text>')
    body.append('<rect x="246" y="93" width="20" height="20" rx="2" fill="#FFFFFF" stroke="#172B4D" stroke-width="2.5"/><text x="276" y="110" class="legend">95% paired-bootstrap CI excludes zero</text>')
    body.append('<text x="34" y="151" class="panel">A · Exploratory mechanism screen — n = 20 paired tasks per cell</text>')
    body.append('<text x="34" y="180" class="note">Display order is conceptual burden (copy → memory → update → reasoning), not a validated dose; output length is fixed.</text>')

    left, cell_w, row_h, top = 348.0, 194.0, 43.0, 242.0
    for index, probe in enumerate(PROBES):
        x = left + (index + 0.5) * cell_w
        lines = str(probe["short_label"]).split("\n")
        spans = "".join(
            f'<tspan x="{x:.1f}" dy="{0 if line_index == 0 else 22}">{_esc(line)}</tspan>'
            for line_index, line in enumerate(lines)
        )
        body.append(f'<text x="{x:.1f}" y="204" text-anchor="middle" class="column">{spans}</text>')
    body.append(f'<text x="{left + 4 * cell_w + 76:.1f}" y="217" text-anchor="middle" class="column">pattern</text>')

    plotted_exploratory = []
    prior_benchmark = None
    for row_index, (benchmark, model) in enumerate(exploratory_strata):
        y = top + row_index * row_h
        if prior_benchmark is not None and benchmark != prior_benchmark:
            body.append(f'<line x1="34" y1="{y - 5:.1f}" x2="1445" y2="{y - 5:.1f}" class="benchmark-rule"/>')
        label = _figure_label(benchmark, model)
        body.append(f'<text x="{left - 14:.1f}" y="{y + 28:.1f}" text-anchor="end" class="row-label">{_esc(label)}</text>')
        for probe_index, arm in enumerate(PROBE_ORDER):
            row = lookup[(benchmark, model, arm)]
            effect = float(row["effect"])
            fill, text_color = _effect_fill(effect)
            x = left + probe_index * cell_w + 4
            w = cell_w - 8
            strong = bool(row["ci_excludes_zero"])
            display = f"{effect * 100:+.0f}"
            tooltip = (
                f"{label} · {row['probe']} · clean={float(row['clean_success']):.3f}, "
                f"active={float(row['active_success']):.3f}, Δ={effect:+.3f}, "
                f"95% CI [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}], n=20"
            )
            body.append(
                f'<rect x="{x:.1f}" y="{y + 4:.1f}" width="{w:.1f}" height="{row_h - 8:.1f}" rx="5" fill="{fill}" stroke="#172B4D" stroke-width="{2.6 if strong else 0.7}"><title>{_esc(tooltip)}</title></rect>'
            )
            body.append(f'<text x="{x + w / 2:.1f}" y="{y + 28:.1f}" text-anchor="middle" class="cell" style="fill:{text_color}">{_esc(display)}</text>')
            plotted_exploratory.append(dict(row))
        pattern = pattern_label[pattern_lookup[(benchmark, model)]]
        body.append(f'<text x="{left + 4 * cell_w + 76:.1f}" y="{y + 28:.1f}" text-anchor="middle" class="pattern">{_esc(pattern)}</text>')
        prior_benchmark = benchmark

    body.append('<text x="34" y="658" class="note">Cells: percentage-point effects. “Worsens” = non-increasing; “opposite” = non-decreasing; “mixed” = changes direction.</text>')
    body.append('<text x="34" y="704" class="panel">B · Powered, prespecified recompute contrast — n = 56 paired tasks per cell</text>')
    body.append('<text x="34" y="733" class="note">Separate tasks and inference tier. Points: active − clean; lines: 95% paired task-bootstrap intervals.</text>')

    axis_left, axis_right, axis_top = 348.0, 1165.0, 798.0
    axis_min, axis_max = -0.45, 0.45
    def xscale(value: float) -> float:
        return axis_left + (value - axis_min) / (axis_max - axis_min) * (axis_right - axis_left)
    for tick in (-0.4, -0.2, 0.0, 0.2, 0.4):
        x = xscale(tick)
        tick_class = "zero" if tick == 0 else "grid"
        body.append(f'<line x1="{x:.1f}" y1="765" x2="{x:.1f}" y2="1074" class="{tick_class}"/>')
        body.append(f'<text x="{x:.1f}" y="758" text-anchor="middle" class="tick">{tick * 100:+.0f}</text>')
    body.append(f'<text x="{(axis_left + axis_right) / 2:.1f}" y="1107" text-anchor="middle" class="axis-label">change in success (percentage points)</text>')
    plotted_confirmatory = []
    prior_benchmark = None
    for index, row in enumerate(confirmatory):
        y = axis_top + index * 43
        benchmark, model = str(row["benchmark"]), str(row["model"])
        if prior_benchmark is not None and benchmark != prior_benchmark:
            body.append(f'<line x1="34" y1="{y - 21:.1f}" x2="1415" y2="{y - 21:.1f}" class="benchmark-rule"/>')
        label = _figure_label(benchmark, model)
        body.append(f'<text x="{axis_left - 14:.1f}" y="{y + 7:.1f}" text-anchor="end" class="row-label">{_esc(label)}</text>')
        low, effect, high = float(row["ci_low"]), float(row["effect"]), float(row["ci_high"])
        body.append(f'<line x1="{xscale(low):.1f}" y1="{y:.1f}" x2="{xscale(high):.1f}" y2="{y:.1f}" class="ci"/>')
        body.append(f'<line x1="{xscale(low):.1f}" y1="{y - 5:.1f}" x2="{xscale(low):.1f}" y2="{y + 5:.1f}" class="ci"/>')
        body.append(f'<line x1="{xscale(high):.1f}" y1="{y - 5:.1f}" x2="{xscale(high):.1f}" y2="{y + 5:.1f}" class="ci"/>')
        color = "#D55E00" if effect < 0 else "#0072B2"
        tooltip = f"{label} · recompute · Δ={effect:+.3f}, 95% CI [{low:+.3f}, {high:+.3f}], n=56"
        body.append(f'<circle cx="{xscale(effect):.1f}" cy="{y:.1f}" r="6.3" fill="{color}" stroke="#FFFFFF" stroke-width="1.4"><title>{_esc(tooltip)}</title></circle>')
        body.append(f'<text x="1192" y="{y + 7:.1f}" class="effect-label">{effect * 100:+.1f} [{low * 100:+.1f}, {high * 100:+.1f}]</text>')
        plotted_confirmatory.append(dict(row))
        prior_benchmark = benchmark

    body.append('<rect x="34" y="1125" width="1411" height="124" rx="8" fill="#F7F9FC" stroke="#BCCCDC"/>')
    body.append('<text x="52" y="1154" class="summary">Descriptive pattern, not a pooled effect</text>')
    body.append('<text x="52" y="1182" class="summary-text">Exploratory: 18 negative / 8 zero / 10 positive; only 1 of 9 non-flat strata shows monotone worsening.</text>')
    body.append('<text x="52" y="1210" class="summary-text">Adjacent exploratory steps: 9 worsen / 10 improve / 8 tie. Powered recompute: 6/7 negative; 3/7 CIs strictly below zero.</text>')
    body.append('<text x="52" y="1238" class="summary-text">Conclusion: active-probe harm is a cross-model trend for recompute, not a universal rule or a simple complexity dose-response.</text>')
    body.append('<text x="34" y="1285" class="footnote">Cells are model × benchmark contrasts. Tasks repeat across models; sign counts are descriptive, not independent observations.</text>')

    style = """
      text { font-family: 'Liberation Sans'; fill: #1F2933; }
      .title { font-size: 36px; font-weight: 760; letter-spacing: -0.3px; }
      .subtitle { font-size: 21px; fill: #52606D; }
      .legend, .note, .footnote { font-size: 21px; fill: #52606D; }
      .panel { font-size: 26px; font-weight: 760; fill: #102A43; }
      .column { font-size: 21px; font-weight: 700; fill: #334E68; }
      .row-label { font-size: 21px; fill: #334E68; }
      .cell { font-size: 21px; font-weight: 760; }
      .pattern { font-size: 21px; font-weight: 700; fill: #52606D; }
      .benchmark-rule { stroke: #829AB1; stroke-width: 1.2; stroke-dasharray: 4 4; }
      .grid { stroke: #D9E2EC; stroke-width: 1; }
      .zero { stroke: #52606D; stroke-width: 1.8; }
      .tick, .axis-label { font-size: 21px; fill: #52606D; }
      .ci { stroke: #334E68; stroke-width: 2.3; stroke-linecap: round; }
      .effect-label { font-size: 21px; font-weight: 650; fill: #334E68; }
      .summary { font-size: 22px; font-weight: 750; fill: #102A43; }
      .summary-text { font-size: 21px; fill: #334E68; }
    """
    description = (
        "Exploratory heatmap and separate powered forest plot of active-minus-clean "
        "task-success effects for four carried probes across reasoning and action benchmarks."
    )
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        '<title id="title">Observer effect of carried active probes</title>\n'
        f'<desc id="desc">{_esc(description)}</desc>\n'
        f'<defs><style>{style}</style></defs>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    sidecar = {
        "schema_version": 1,
        "figure_type": "exploratory_ladder_heatmap_plus_confirmatory_recompute_forest",
        "title": "Observer effect of carried active probes",
        "description": description,
        "source_json_sha256": source_json_sha256,
        "effect_definition": "task success proportion in active arm minus clean arm",
        "exploratory_panel": {
            "n_tasks_per_cell": 20,
            "rows": plotted_exploratory,
            "encoding": "cell label is percentage-point effect; thick outline means 95% CI excludes zero",
        },
        "confirmatory_panel": {
            "n_tasks_per_cell": 56,
            "rows": plotted_confirmatory,
            "encoding": "point estimate and 95% paired task-bootstrap interval, percentage-point axis",
            "axis_min": axis_min,
            "axis_max": axis_max,
        },
        "width": width,
        "height": height,
    }
    return svg, sidecar


def main() -> None:
    live_hash_before = code_tree_hash(PACKAGE)
    if live_hash_before != EXPECTED_CODE_HASH:
        raise LadderInputError(f"frozen code tree changed before synthesis: {live_hash_before}")
    all_rows: list[dict[str, Any]] = []
    inventory = []
    task_sets: dict[tuple[str, str], set[str]] = {}
    for spec in SOURCES:
        rows, source, task_identities = _extract_source(spec)
        all_rows.extend(rows)
        inventory.append(source)
        task_sets[(str(spec["study"]), str(spec["benchmark"]))] = task_identities
    if len(all_rows) != 43:
        raise LadderInputError("expected 36 exploratory and 7 confirmatory effects")
    all_rows.sort(
        key=lambda row: (
            0 if row["study"] == "exploratory_mechanism" else 1,
            *_row_sort_key(row),
            int(row["display_order"]),
        )
    )
    exploratory = [row for row in all_rows if row["study"] == "exploratory_mechanism"]
    confirmatory = [row for row in all_rows if row["study"] == "confirmatory_powered"]
    if len(exploratory) != 36 or len(confirmatory) != 7:
        raise LadderInputError("study-tier effect counts changed")
    monotonicity = _monotonicity(exploratory)
    task_set_separation = []
    for benchmark in ("evolving_intent_gsm8k", "bfcl_multi_turn"):
        exploratory_tasks = task_sets[("exploratory_mechanism", benchmark)]
        confirmatory_tasks = task_sets[("confirmatory_powered", benchmark)]
        overlap = exploratory_tasks & confirmatory_tasks
        if overlap:
            raise LadderInputError(f"{benchmark} exploratory/confirmatory task sets overlap")
        task_set_separation.append(
            {
                "benchmark": benchmark,
                "exploratory_task_identities": len(exploratory_tasks),
                "confirmatory_task_identities": len(confirmatory_tasks),
                "overlap": 0,
                "exploratory_task_identities_sha256": sha256_json(sorted(exploratory_tasks)),
                "confirmatory_task_identities_sha256": sha256_json(sorted(confirmatory_tasks)),
            }
        )
    payload = {
        "schema_version": 1,
        "artifact": "active_probe_observer_effect_ladder",
        "code_tree_sha256": live_hash_before,
        "effect_definition": "paired task success proportion: active probe minus clean",
        "interpretation": {
            "negative": "carried active probe reduced task success",
            "positive": "carried active probe increased task success",
            "zero": "no point-estimate change",
        },
        "inference_separation": {
            "exploratory_mechanism": "n=20 per model/benchmark/probe; hypothesis-generating",
            "confirmatory_powered": "n=56 per model/benchmark; prespecified active_recompute contrast",
            "prohibition": "no pooling across tiers, models, benchmarks, or repeated task identities",
        },
        "probe_display_order": [
            {
                "arm": row["arm"],
                "probe": row["probe"],
                "burden_axis": row["burden_axis"],
                "order": row["order"],
            }
            for row in PROBES
        ],
        "source_runs": inventory,
        "task_set_separation": task_set_separation,
        "counts": {
            "effect_rows": len(all_rows),
            "exploratory_effect_rows": len(exploratory),
            "confirmatory_effect_rows": len(confirmatory),
            "exploratory_model_benchmark_strata": 9,
            "confirmatory_model_benchmark_strata": 7,
        },
        "descriptive_sign_counts": {
            "exploratory": _sign_summary(exploratory),
            "confirmatory_recompute": _sign_summary(confirmatory),
            "by_exploratory_probe": {
                arm: _sign_summary([row for row in exploratory if row["active_arm"] == arm])
                for arm in PROBE_ORDER
            },
            "by_benchmark_and_tier": {
                f"{study}/{benchmark}": _sign_summary(
                    [row for row in all_rows if row["study"] == study and row["benchmark"] == benchmark]
                )
                for study in ("exploratory_mechanism", "confirmatory_powered")
                for benchmark in ("evolving_intent_gsm8k", "bfcl_multi_turn")
            },
        },
        "exploratory_monotonicity": monotonicity,
        "recompute_cross_stage_stability": _cross_stage(all_rows),
        "rows": all_rows,
        "headline": (
            "The powered recompute effect is negative in six of seven model/benchmark "
            "strata, but exploratory probe burden is non-monotone and one powered "
            "stratum is positive: this is a trend, not a rule."
        ),
    }

    json_path = OUTPUT_STEM.with_suffix(".json")
    csv_path = OUTPUT_STEM.with_suffix(".csv")
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    sidecar_path = OUTPUT_STEM.with_suffix(".svg.data.json")
    receipt_path = OUTPUT_STEM.with_suffix(".receipt.json")
    json_sha = atomic_write_json(json_path, payload)
    csv_fields = (
        "study",
        "inference_status",
        "run_id",
        "benchmark",
        "model",
        "active_arm",
        "probe",
        "burden_axis",
        "display_order",
        "n_tasks",
        "clean_success",
        "active_success",
        "effect",
        "effect_percentage_points",
        "ci_low",
        "ci_high",
        "ci_low_percentage_points",
        "ci_high_percentage_points",
        "point_sign",
        "strict_ci_sign",
        "ci_excludes_zero",
        "confidence",
        "bootstrap_iterations",
        "bootstrap_seed",
        "bootstrap_unit",
    )
    csv_sha = atomic_write_bytes(csv_path, _csv_bytes(all_rows, csv_fields))
    svg, sidecar = _figure(all_rows, monotonicity, source_json_sha256=json_sha)
    svg_sha = atomic_write_bytes(svg_path, svg.encode("utf-8"))
    sidecar_sha = atomic_write_json(sidecar_path, sidecar)
    live_hash_after = code_tree_hash(PACKAGE)
    if live_hash_after != live_hash_before:
        raise LadderInputError(f"frozen code tree changed during synthesis: {live_hash_after}")
    receipt = {
        "schema_version": 1,
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256_file(__file__),
        "provider_calls_made": 0,
        "code_tree_sha256_before": live_hash_before,
        "code_tree_sha256_after": live_hash_after,
        "source_extracts": {
            source["extract_path"]: source["extract_sha256"] for source in inventory
        },
        "outputs": {
            str(json_path.relative_to(ROOT)): json_sha,
            str(csv_path.relative_to(ROOT)): csv_sha,
            str(svg_path.relative_to(ROOT)): svg_sha,
            str(sidecar_path.relative_to(ROOT)): sidecar_sha,
        },
    }
    receipt_sha = atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "code_tree_sha256": live_hash_after,
                "effect_rows": len(all_rows),
                "exploratory_rows": len(exploratory),
                "confirmatory_rows": len(confirmatory),
                "receipt": str(receipt_path.relative_to(ROOT)),
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
