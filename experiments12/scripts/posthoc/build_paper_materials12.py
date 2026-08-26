#!/usr/bin/env python3
"""Build the final, provider-free Experiment 12 paper-material inventory.

The normal command is intentionally fail-closed.  It accepts only the frozen
confirmatory and deployment designs, verifies the final analyses and audit
receipts, derives every numerical claim from their machine-readable rows, and
writes ``PAPER_MATERIALS12.json`` plus ``PAPER_MATERIALS12.md``.  ``--dry-check``
checks the frozen static evidence and reports which final deployment products
are still missing without writing anything.

This script never imports a provider client, dispatches a model call, or edits
source/config/raw artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_CODE_TREE_SHA256 = (
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

G = ROOT / "experiments12" / "data_results" / "derived"
A = ROOT / "experiments12" / "data_results" / "runs"
EVOLVING_RUN = A / "e12-confirmatory-evolving-core-v2"
BFCL_RUN = A / "e12-confirmatory-bfcl-core-v3"
ONLINE_ROOT = A / ONLINE_RUN
YOKED_ROOT = A / YOKED_RUN
PASS1_ROOT = A / "e12-deploy-twopass-pass1-evolving-luna-40-v1"

DEFAULTS = {
    "readme": ROOT / "README.md",
    "evolving_score": EVOLVING_RUN / "results" / "score-confirmatory.json",
    "bfcl_score": BFCL_RUN / "results" / "score-confirmatory-no-opportunity-v1.json",
    "bfcl_sensitivity": BFCL_RUN / "results" / "score-confirmatory-complete-case-sensitivity.json",
    "evolving_validation": EVOLVING_RUN / "results" / "validation-confirmatory.json",
    "bfcl_validation": BFCL_RUN / "results" / "validation-confirmatory.json",
    "ladder": G / "active-probe-ladder-confirmatory-v1.json",
    "mechanism": G / "active-probe-mechanism-exploratory-v1.json",
    "overhead": G / "observer-overhead-confirmatory-v1.json",
    "online": ONLINE_ROOT / "results" / "adaptive-analysis.json",
    "yoked": YOKED_ROOT / "results" / "two-pass-analysis.json",
    "yoked_validation": YOKED_ROOT / "results" / "validation-two-pass.json",
    "post": G / "deployment-paper-post-analysis-v1" / "deployment-paper-post-analysis.json",
    "sensitivity": ONLINE_ROOT / "results" / "adaptive-analysis-leave-two-units.json",
    "staging_receipt": G / "adaptive-analysis-attested-v1" / "staging-receipt.json",
    "analysis_receipt": G / "adaptive-analysis-attested-v1" / "analysis-receipt.json",
    "output_json": G / "PAPER_MATERIALS12.json",
    "output_md": G / "PAPER_MATERIALS12.md",
}

METHODS = (
    "active_recompute",
    "context_use",
    "frozen_probe:current_copy",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
)
ONLINE_METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
    "context_use",
)
ONLINE_OPERATORS = (
    "none",
    "lossy_compaction",
    "public_state_reground",
    "good_bad_watch_feedback",
)
YOKED_METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "turn_clock",
    "context_use",
)
YOKED_OPERATORS = ("none", "lossy_compaction", "public_state_reground")
MODEL_LABELS = {
    "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "gpt-oss-120b": "GPT-OSS-120B",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
}
METHOD_LABELS = {
    "active_recompute": "Active recompute",
    "frozen_probe:current_copy": "Frozen current-copy",
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
    "good_bad_watch_feedback": "quote-only WATCH reminder",
}


class MaterialsError(RuntimeError):
    """Raised when an input is missing, partial, or scientifically mismatched."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterialsError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def regular(path: Path, *, context: str) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing or linked {context}: {path}")
    return path


def read_json(path: Path, *, context: str) -> Mapping[str, Any]:
    regular(path, context=context)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, Mapping), f"{context} is not a JSON object")
    return value


def file_record(path: Path, *, role: str, artifact_type: str | None = None) -> dict[str, Any]:
    regular(path, context=role)
    record: dict[str, Any] = {
        "role": role,
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if artifact_type is not None:
        record["artifact_type"] = artifact_type
    return record


def finite(value: Any, *, context: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"non-numeric {context}")
    result = float(value)
    require(math.isfinite(result), f"non-finite {context}")
    return result


def exact_grid(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    expected: set[tuple[Any, ...]],
    *,
    context: str,
) -> None:
    observed = [tuple(row.get(field) for field in fields) for row in rows]
    require(len(observed) == len(set(observed)), f"{context} has duplicate grid rows")
    require(set(observed) == expected, f"{context} treatment grid changed")


def validate_code_tree() -> str:
    from experiments12.manifest12 import code_tree_hash

    result = code_tree_hash(ROOT / "experiments12")
    require(result == EXPECTED_CODE_TREE_SHA256, f"frozen code tree changed: {result}")
    return result


def paper_section(readme: Path) -> dict[str, Any]:
    text = regular(readme, context="README").read_text(encoding="utf-8")
    require(text.startswith("# Paper Contents\n"), "README no longer begins with Paper Contents")
    end = text.find("\n---\n", 1)
    require(end > 0, "Paper Contents terminator is missing")
    section = text[:end].strip() + "\n"
    return {
        "path": relative(readme),
        "file_sha256": sha256_file(readme),
        "section_sha256": hashlib.sha256(section.encode("utf-8")).hexdigest(),
        "title": "Active and passive test-time observation methods for detecting long-horizon performance degradation",
    }


def validate_confirmation(
    validation_path: Path,
    score_path: Path,
    *,
    run_id: str,
    manifest_sha: str,
    pair_sha: str,
    benchmark: str,
    models: Sequence[str],
    score_n: Mapping[str, int],
    expected_cells: int,
    expected_shadows: int,
) -> tuple[Mapping[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    validation = read_json(validation_path, context=f"{run_id} validation")
    require(validation.get("run_id") == run_id, f"{run_id} validation identity changed")
    require(validation.get("manifest_sha256") == manifest_sha, f"{run_id} manifest changed")
    require(validation.get("pair_manifest_sha256") == pair_sha, f"{run_id} pairs changed")
    require(validation.get("primary_ready") is True, f"{run_id} is not primary-ready")
    require(validation.get("errors") == [] and validation.get("warnings") == [], f"{run_id} validation is not clean")
    require(validation.get("expected_cells") == expected_cells, f"{run_id} cell count changed")
    require(validation.get("trajectory_outputs") == expected_cells, f"{run_id} trajectories incomplete")
    require(validation.get("valid_trajectories") == expected_cells, f"{run_id} invalid trajectories")
    require(validation.get("shadow_outputs") == expected_shadows, f"{run_id} shadows incomplete")

    score = read_json(score_path, context=f"{run_id} signal score")
    require(score.get("source_run_id") == run_id, f"{run_id} score identity changed")
    require(score.get("source_manifest_sha256") == manifest_sha, f"{run_id} score provenance changed")
    metrics = score.get("metrics")
    require(isinstance(metrics, list) and all(isinstance(row, Mapping) for row in metrics), f"{run_id} score rows invalid")
    expected = {(model, method) for model in models for method in METHODS}
    exact_grid(metrics, ("model", "method"), expected, context=f"{run_id} signal score")
    require({row.get("benchmark") for row in metrics} == {benchmark}, f"{run_id} benchmark changed")
    require(all(row.get("n_tasks") == score_n[str(row["model"])] for row in metrics), f"{run_id} signal denominators changed")
    for row in metrics:
        for field in ("precision", "recall", "auprc", "firing_rate"):
            value = finite(row.get(field), context=f"{run_id}/{row['model']}/{row['method']}/{field}")
            require(0 <= value <= 1, f"{run_id} invalid {field}")
    winners: list[dict[str, Any]] = []
    for model in models:
        candidates = [row for row in metrics if row["model"] == model]
        best = max(float(row["auprc"]) for row in candidates)
        tied = sorted(str(row["method"]) for row in candidates if math.isclose(float(row["auprc"]), best, abs_tol=1e-15))
        winners.append(
            {
                "benchmark": benchmark,
                "model": model,
                "n_tasks": score_n[model],
                "highest_auprc": best,
                "winning_methods": tied,
                "winning_classes": sorted({paper_class(method) for method in tied}),
                "tie": len(tied) > 1,
            }
        )
    selected_fields = (
        "benchmark", "model", "method", "n_tasks", "n_positive_tasks", "locked_threshold",
        "selection_rule", "firing_rate", "precision", "recall", "auprc", "lead_time",
    )
    clean_metrics = [{field: row.get(field) for field in selected_fields} for row in metrics]
    return validation, clean_metrics, winners


def paper_class(method: str) -> str:
    if method == "active_recompute":
        return "active"
    if method in {"turn_clock", "context_use"}:
        return "baseline"
    if method in set(METHODS) | set(ONLINE_METHODS):
        return "passive"
    raise MaterialsError(f"unknown method: {method}")


def validate_ladder(path: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]], dict[str, Any]]:
    value = read_json(path, context="active-probe ladder")
    require(value.get("artifact") == "active_probe_observer_effect_ladder", "ladder artifact changed")
    require(value.get("code_tree_sha256") == EXPECTED_CODE_TREE_SHA256, "ladder code provenance changed")
    rows = value.get("rows")
    require(isinstance(rows, list), "ladder lacks rows")
    powered = [dict(row) for row in rows if row.get("inference_status") == "powered_prespecified_recompute"]
    require(len(powered) == 7, "ladder must contain seven powered strata")
    expected_strata = {
        *(('evolving_intent_gsm8k', model) for model in ('deepseek-v4-flash-0731', 'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-oss-120b')),
        *(('bfcl_multi_turn', model) for model in ('gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-oss-120b')),
    }
    exact_grid(powered, ("benchmark", "model"), expected_strata, context="powered observer effects")
    require(all(row.get("active_arm") == "active_recompute" and row.get("n_tasks") == 56 for row in powered), "powered ladder treatment or n changed")
    for row in powered:
        finite(row.get("effect"), context="powered success effect")
        low = finite(row.get("ci_low"), context="powered CI low")
        high = finite(row.get("ci_high"), context="powered CI high")
        require(low <= float(row["effect"]) <= high, "powered effect is outside its CI")
    negative = sum(float(row["effect"]) < 0 for row in powered)
    positive = sum(float(row["effect"]) > 0 for row in powered)
    strict_negative = sum(float(row["ci_high"]) < 0 for row in powered)
    strict_positive = sum(float(row["ci_low"]) > 0 for row in powered)
    require((negative, positive, strict_negative, strict_positive) == (6, 1, 3, 0), "powered observer-effect headline changed; review claims")
    headline = {
        "strata": 7,
        "negative_point_estimates": negative,
        "positive_point_estimates": positive,
        "negative_intervals_excluding_zero": strict_negative,
        "positive_intervals_excluding_zero": strict_positive,
        "wording": "Active recomputation usually reduced success, but not universally: 6/7 point estimates were negative, 3 paired 95% intervals excluded zero below, and one stratum was positive.",
    }
    return value, sorted(powered, key=lambda row: (row["benchmark"], row["model"])), headline


def validate_mechanism(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    value = read_json(path, context="exploratory active mechanism")
    require(value.get("artifact") == "active_probe_mechanism_exploratory_summary", "mechanism artifact changed")
    require(value.get("analysis_status") == "exploratory_baseline_gate", "mechanism status changed")
    require(value.get("code_tree_sha256") == EXPECTED_CODE_TREE_SHA256, "mechanism code provenance changed")
    counts = value.get("counts")
    require(isinstance(counts, Mapping), "mechanism counts missing")
    require(counts.get("paired_effect_rows") == 36 and counts.get("paired_task_rows") == 720, "mechanism sample changed")
    require(counts.get("tasks_per_effect") == 20 and counts.get("mechanisms") == 4, "mechanism design changed")
    summaries = value.get("descriptive_summaries")
    require(isinstance(summaries, list) and len(summaries) == 12, "mechanism summaries changed")
    all_benchmarks = [row for row in summaries if row.get("benchmark") == "all_benchmarks"]
    require(len(all_benchmarks) == 4, "mechanism all-benchmark summaries changed")
    medians = {str(row["mechanism"]): finite(row["median_effect"], context="mechanism median") for row in all_benchmarks}
    effect_rows = value.get("paired_effect_rows")
    require(isinstance(effect_rows, list) and len(effect_rows) == 36, "mechanism effect rows changed")
    signs = {
        "negative": sum(float(row["paired_success_effect"]) < 0 for row in effect_rows),
        "positive": sum(float(row["paired_success_effect"]) > 0 for row in effect_rows),
        "zero": sum(float(row["paired_success_effect"]) == 0 for row in effect_rows),
    }
    require(signs == {"negative": 18, "positive": 10, "zero": 8}, "exploratory mechanism heterogeneity changed; review")
    return value, {
        "status": "exploratory_only",
        "model_benchmark_strata": 9,
        "tasks_per_arm_stratum": 20,
        "probe_median_success_effects": medians,
        "stratum_point_sign_counts": signs,
        "conclusion": "Copy, recall, counter, and recomputation effects are heterogeneous; the data do not support a monotonic chore-complexity claim.",
    }


def decimal_value(value: Any, *, context: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise MaterialsError(f"invalid decimal {context}") from exc
    require(result.is_finite(), f"invalid decimal {context}")
    return result


def validate_overhead(path: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]], dict[str, Any]]:
    value = read_json(path, context="observer overhead")
    require(value.get("artifact") == "confirmatory_observer_overhead", "overhead artifact changed")
    require(value.get("code_tree_sha256") == EXPECTED_CODE_TREE_SHA256, "overhead code provenance changed")
    counts = value.get("counts")
    require(isinstance(counts, Mapping), "overhead counts missing")
    require(counts == {"benchmark_model_strata": 7, "benchmarks": 2, "methods": 8, "model_tasks": 392, "task_method_rows": 3136}, "overhead dimensions changed")
    rows = value.get("summaries")
    require(isinstance(rows, list) and len(rows) == 56, "overhead summaries changed")
    grouped: dict[str, dict[str, Decimal]] = {
        method: {"tasks": Decimal(0), "calls": Decimal(0), "tokens": Decimal(0), "latency_ms": Decimal(0), "cost_usd": Decimal(0)}
        for method in METHODS
    }
    for row in rows:
        method = str(row.get("method"))
        require(method in grouped, "overhead method changed")
        grouped[method]["tasks"] += decimal_value(row.get("n_tasks"), context="overhead n")
        grouped[method]["calls"] += decimal_value(row.get("sum_provider_call_count"), context="overhead calls")
        grouped[method]["tokens"] += decimal_value(row.get("sum_total_tokens"), context="overhead tokens")
        grouped[method]["latency_ms"] += decimal_value(row.get("sum_latency_ms"), context="overhead latency")
        grouped[method]["cost_usd"] += decimal_value(row.get("sum_cost_usd"), context="overhead cost")
    aggregates: list[dict[str, Any]] = []
    for method in METHODS:
        row = grouped[method]
        require(row["tasks"] == 392, f"overhead denominator changed for {method}")
        aggregates.append(
            {
                "method": method,
                "observation_class": paper_class(method),
                "model_tasks": int(row["tasks"]),
                "provider_calls": int(row["calls"]),
                "total_tokens": int(row["tokens"]),
                "summed_provider_elapsed_ms": int(row["latency_ms"]),
                "cost_usd": f"{row['cost_usd']:.6f}",
            }
        )
    by_method = {row["method"]: row for row in aggregates}
    active = by_method["active_recompute"]
    frozen = by_method["frozen_probe:recompute"]
    require(active["provider_calls"] == frozen["provider_calls"] > 0, "matched recompute observer calls changed")
    deterministic = [by_method[name] for name in ("trace_rules", "turn_clock", "context_use")]
    require(all(row["provider_calls"] == row["total_tokens"] == 0 and Decimal(row["cost_usd"]) == 0 for row in deterministic), "deterministic overhead is no longer zero-provider")
    comparison = {
        "active_vs_frozen_recompute": {
            "matched_provider_calls_each": active["provider_calls"],
            "active_tokens": active["total_tokens"],
            "frozen_tokens": frozen["total_tokens"],
            "active_minus_frozen_tokens": active["total_tokens"] - frozen["total_tokens"],
            "active_relative_token_difference": active["total_tokens"] / frozen["total_tokens"] - 1,
            "active_cost_usd": active["cost_usd"],
            "frozen_cost_usd": frozen["cost_usd"],
            "semantics": "Active elapsed time is on the target path; frozen elapsed time is off-path compute and is not target delay.",
        },
        "deterministic_zero_provider_methods": [row["method"] for row in deterministic],
        "conclusion": "Zero-carry does not mean zero-cost: provider-backed passive monitors add calls, tokens, latency, and dollars, while deterministic rules, clock, and context-use add no provider calls.",
    }
    return value, aggregates, comparison


def validate_online(path: Path) -> Mapping[str, Any]:
    value = read_json(path, context="online adaptive analysis")
    require(value.get("artifact_type") == "online_adaptive_deployment_analysis", "online artifact type changed")
    require(value.get("source_run_id") == ONLINE_RUN, "online run changed")
    require(value.get("source_manifest_sha256") == ONLINE_MANIFEST_SHA256, "online manifest changed")
    require(value.get("source_pair_manifest_sha256") == ONLINE_PAIRS_SHA256, "online pairs changed")
    require(value.get("deployment_mode") == "online_adaptive", "online estimand changed")
    require(value.get("deployment_policy") == "natural_threshold_per_task_cap", "online policy changed")
    require(value.get("per_task_action_cap") == 1 and value.get("statistical_unit") == "source_task", "online action cap/unit changed")
    rows = value.get("rows")
    require(isinstance(rows, list) and len(rows) == 1120, "online rows incomplete")
    exact_grid(rows, ("method", "operator", "unit_id"), {(m, o, f"extracted-gsm8k-test-{task}::t7/r0") for m in ONLINE_METHODS for o in ONLINE_OPERATORS for task in sorted({int(str(row["task_id"]).split("-")[-1].split("::")[0]) for row in rows})}, context="online rows")
    require(len({row.get("unit_id") for row in rows}) == 40, "online task denominator changed")
    require({row.get("model") for row in rows} == {"gpt-5.6-luna"}, "online model changed")
    require({row.get("benchmark") for row in rows} == {"evolving_intent_gsm8k"}, "online benchmark changed")
    summaries = value.get("metric_summaries")
    effects = value.get("operator_effects")
    require(isinstance(summaries, list) and len(summaries) == 224, "online summaries incomplete")
    require(isinstance(effects, list) and len(effects) == 168, "online effects incomplete")
    require({row.get("n_tasks") for row in summaries} == {40} and {row.get("n_tasks") for row in effects} == {40}, "online analysis denominator changed")
    return value


def validate_yoked(path: Path, validation_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    validation = read_json(validation_path, context="yoked validation")
    require(validation.get("artifact_type") == "two_pass_deployment_validation", "yoked validation type changed")
    require(validation.get("source_run_id") == YOKED_RUN, "yoked validation run changed")
    require(validation.get("source_manifest_sha256") == YOKED_MANIFEST_SHA256, "yoked validation manifest changed")
    require(validation.get("source_pair_manifest_sha256") == YOKED_PAIRS_SHA256, "yoked validation pairs changed")
    require(validation.get("source_schedule_sha256") == YOKED_SCHEDULE_SHA256, "yoked validation schedule changed")
    require(validation.get("primary_ready") is True, "yoked validation is not primary-ready")
    require(all(validation.get(field) == 480 for field in ("expected_cells", "valid_outputs", "valid_jobs", "valid_event_logs", "canonical_regraded_cells")), "yoked validation incomplete")
    require(validation.get("cached_official_cells") == 0, "yoked validation used cached grading")
    value = read_json(path, context="yoked analysis")
    require(value.get("artifact_type") == "two_pass_deployment_analysis", "yoked artifact type changed")
    require(value.get("source_run_id") == YOKED_RUN and value.get("source_manifest_sha256") == YOKED_MANIFEST_SHA256, "yoked analysis provenance changed")
    require(value.get("source_pair_manifest_sha256") == YOKED_PAIRS_SHA256 and value.get("source_schedule_sha256") == YOKED_SCHEDULE_SHA256, "yoked analysis pair/schedule changed")
    require(value.get("deployment_mode") == "two_pass_frozen" and value.get("estimand") == "yoked_anchor", "yoked estimand changed")
    rows = value.get("rows")
    require(isinstance(rows, list) and len(rows) == 480, "yoked rows incomplete")
    require(len({row.get("unit_id") for row in rows}) == 40, "yoked task denominator changed")
    require({row.get("method") for row in rows} == set(YOKED_METHODS) and {row.get("operator") for row in rows} == set(YOKED_OPERATORS), "yoked treatment grid changed")
    require(len(value.get("metric_summaries", [])) == 168 and len(value.get("operator_effects", [])) == 112 and len(value.get("method_effects", [])) == 252, "yoked analysis summaries incomplete")
    return value, validation


def validate_post(path: Path, online_path: Path, yoked_path: Path) -> Mapping[str, Any]:
    value = read_json(path, context="deployment post-analysis")
    require(value.get("artifact_type") == "experiment12_deployment_paper_post_analysis", "post-analysis type changed")
    require(value.get("provider_calls_made") == 0 and value.get("statistical_unit") == "source_task", "post-analysis semantics changed")
    provenance = value.get("provenance")
    require(isinstance(provenance, Mapping), "post-analysis provenance missing")
    require(provenance.get("code_tree_sha256") == EXPECTED_CODE_TREE_SHA256, "post-analysis code provenance changed")
    require(provenance.get("online_analysis_sha256") == sha256_file(online_path), "post-analysis does not consume declared online analysis")
    require(provenance.get("yoked_analysis_sha256") == sha256_file(yoked_path), "post-analysis does not consume declared yoked analysis")
    online = value.get("online_primary")
    yoked = value.get("yoked_controlled_sensitivity")
    require(isinstance(online, Mapping) and isinstance(yoked, Mapping), "post-analysis sections missing")
    expected_online = {"performance_summaries": 28, "operating_resource_summaries": 308, "operator_effects": 21, "method_effects": 84, "method_operator_interactions": 63}
    for field, count in expected_online.items():
        require(isinstance(online.get(field), list) and len(online[field]) == count, f"post online {field} changed")
    expected_yoked = {"summaries": 96, "operator_effects": 8, "method_effects": 18, "method_operator_interactions": 12}
    for field, count in expected_yoked.items():
        require(isinstance(yoked.get(field), list) and len(yoked[field]) == count, f"post yoked {field} changed")
    require(len(value.get("figure_files", [])) == 5 and len(value.get("figure_data_files", [])) == 5, "post-analysis figure inventory changed")
    return value


def validate_sensitivity(path: Path, online_path: Path) -> Mapping[str, Any]:
    value = read_json(path, context="online cumulative two-unit sensitivity")
    require(value.get("artifact_type") == "experiment12_online_cumulative_affected_unit_sensitivity", "sensitivity type changed")
    require(value.get("provider_calls_made") == 0 and value.get("analysis_only") is True, "sensitivity semantics changed")
    source = value.get("source")
    design = value.get("design")
    selection = value.get("affected_units_selection")
    audit = value.get("audit_result")
    require(all(isinstance(item, Mapping) for item in (source, design, selection, audit)), "sensitivity sections missing")
    require(source.get("analysis_sha256") == sha256_file(online_path), "sensitivity source analysis changed")
    require(source.get("source_run_id") == ONLINE_RUN and source.get("source_manifest_sha256") == ONLINE_MANIFEST_SHA256, "sensitivity run provenance changed")
    require(design.get("source_rows") == 1120 and design.get("filtered_rows") == 1064, "sensitivity row counts changed")
    require(design.get("source_tasks") == 40 and design.get("affected_source_units") == 2 and design.get("filtered_tasks") == 38 and design.get("recovery_cells") == 3 and design.get("treatments") == 28, "sensitivity denominator changed")
    documented = selection.get("documented_recovery_cells")
    expected_cells = {"d52046b6eb74a76ecdc3debc", "89df41e0daa1262a43fa5e55", "786d95760ccdb86713c26936"}
    require(isinstance(documented, list) and {row.get("cell_id") for row in documented} == expected_cells, "sensitivity recovery-cell coverage changed")
    affected_units = {row.get("unit_id") for row in documented}
    require(affected_units == {"extracted-gsm8k-test-814::t7/r0", "extracted-gsm8k-test-989::t7/r0"}, "sensitivity affected-unit coverage changed")
    require(selection.get("outcome_fields_used_for_selection") == [] and selection.get("removed_rows") == 56, "sensitivity selection is not outcome-blind paired omission")
    require(audit.get("source_summaries_exactly_reproduced") is True and audit.get("two_units_removed_from_every_treatment") is True and audit.get("all_treatments_share_exact_n38_unit_set") is True, "sensitivity audit failed")
    comparisons = value.get("comparisons")
    recomputed = value.get("recomputed_n38")
    require(isinstance(comparisons, Mapping) and isinstance(recomputed, Mapping), "sensitivity comparison sections missing")
    expected_metrics = {"success", "threshold_firings", "selected_actions", "task_tokens", "observer_tokens", "total_tokens", "latency_ms", "actual_cost_usd"}
    require(set(comparisons.get("metrics", ())) == expected_metrics, "sensitivity does not cover the exact eight metrics")
    require(len(comparisons.get("absolute_summaries", ())) == 224 and len(comparisons.get("operator_effects", ())) == 168, "sensitivity comparison grid incomplete")
    require(len(recomputed.get("metric_summaries", ())) == 224 and len(recomputed.get("operator_effects", ())) == 168, "n=38 recomputed grid incomplete")
    require({row.get("n_tasks") for row in recomputed["metric_summaries"]} == {38} and {row.get("n_tasks") for row in recomputed["operator_effects"]} == {38}, "n=38 recomputed denominators changed")
    require(audit.get("all_metric_summaries_compared") == 224 and audit.get("all_operator_effects_compared") == 168, "sensitivity eight-metric audit coverage changed")
    require(audit.get("success_operator_effects_compared") == 21 and audit.get("threshold_firing_operator_effects_compared") == 21 and audit.get("selected_action_rate_operator_effects_compared") == 21 and audit.get("key_resource_operator_effects_compared") == 105, "sensitivity metric-group coverage changed")
    return value


def validate_receipt(path: Path, expected_type: str, *, context: str) -> Mapping[str, Any]:
    value = read_json(path, context=context)
    require(value.get("artifact_type") == expected_type, f"{context} type changed")
    return value


def validate_audit_receipts(staging_path: Path, analysis_path: Path, online_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    staging = validate_receipt(staging_path, "experiment12_adaptive_semantic_retry_normalization", context="normalization receipt")
    analysis = validate_receipt(analysis_path, "experiment12_staged_stock_adaptive_analysis_receipt", context="staged analyzer receipt")
    require(staging.get("run_id") == ONLINE_RUN and staging.get("source_manifest_sha256") == ONLINE_MANIFEST_SHA256, "normalization run provenance changed")
    require(staging.get("copy_on_write") is True and staging.get("production_files_or_ledger_modified") is False, "normalization is not copy-on-write")
    require(staging.get("outcome_values_used_to_select_patch_scope") is False and staging.get("scientific_score_or_decision_changed") is False, "normalization used scientific outcomes or decisions")
    require(staging.get("all_attempt_tokens_latency_and_cost_retained") is True, "normalization accounting proof missing")
    normalization_cases = staging.get("normalization_cases")
    require(isinstance(normalization_cases, list) and len(normalization_cases) == 3 and all(isinstance(row, Mapping) for row in normalization_cases), "three-cell semantic-normalization inventory missing")
    normalized_recovery_cells = {row.get("cell_id") for row in normalization_cases if row.get("cell_id") in {"d52046b6eb74a76ecdc3debc", "89df41e0daa1262a43fa5e55", "786d95760ccdb86713c26936"}}
    require(normalized_recovery_cells == {"d52046b6eb74a76ecdc3debc", "89df41e0daa1262a43fa5e55", "786d95760ccdb86713c26936"}, "normalization does not cover all three recovered cells")
    ordinary_reconciliations = staging.get("ordinary_failed_attempt_ledger_reconciliations")
    require(isinstance(ordinary_reconciliations, list) and len(ordinary_reconciliations) == 1 and isinstance(ordinary_reconciliations[0], Mapping), "expected exactly one ordinary failed-attempt ledger reconciliation")
    ordinary_reconciliation_text = json.dumps(ordinary_reconciliations[0], sort_keys=True).lower()
    require("503" in ordinary_reconciliation_text and "ledger" in ordinary_reconciliation_text and "failed" in ordinary_reconciliation_text, "ordinary reconciliation is not the declared HTTP 503 ledger-status normalization")
    require(analysis.get("run_id") == ONLINE_RUN and analysis.get("unmodified_stock_analyzer") is True and analysis.get("analysis_rows") == 1120, "stock analyzer audit failed")
    require(analysis.get("staging_receipt_sha256") == sha256_file(staging_path), "analysis receipt points to another staging receipt")
    require(analysis.get("analysis_output_sha256") == sha256_file(online_path), "audited staged analysis differs from declared online analysis")

    receipt_specs = (
        (G / "recovery-adaptive-d52046b6eb74a76ecdc3debc12.json", "experiment12_online_adaptive_single_cell_recovery", "online recovery d520"),
        (G / "recovery-adaptive-89df41e0daa1262a43fa5e5512.json", "experiment12_online_adaptive_trace_judge_recovery", "online recovery 89df"),
        (G / "recovery-adaptive-786d95760ccdb86713c2693612.json", "experiment12_online_adaptive_trace_judge_recovery", "online recovery 786"),
        (ONLINE_ROOT / "results" / "recovery" / "d52046b6eb74a76ecdc3debc" / "archive-receipt.json", "experiment12_online_adaptive_pre_recovery_archive", "online archive d520"),
        (ONLINE_ROOT / "results" / "recovery" / "89df41e0daa1262a43fa5e55" / "archive-receipt.json", "experiment12_online_adaptive_pre_recovery_archive", "online archive 89df"),
        (ONLINE_ROOT / "results" / "recovery" / "786d95760ccdb86713c26936" / "archive-receipt.json", "experiment12_online_adaptive_pre_recovery_archive", "online archive 786"),
        (G / "forensic-audit-adaptive-89df41e0daa1262a43fa5e5512.json", "experiment12_provider_free_recovery_forensic_audit", "online forensic audit 89df"),
        (G / "forensic-audit-adaptive-786d95760ccdb86713c2693612.json", "experiment12_provider_free_recovery_forensic_audit", "online forensic audit 786"),
        (PASS1_ROOT / "results" / "recovery" / "9d8591ea71f67026d743d434" / "recovery-receipt.json", "experiment12_single_missing_shadow_judge_recovery", "two-pass pass-one recovery"),
    )
    records = [
        file_record(staging_path, role="online semantic-retry normalization", artifact_type=str(staging["artifact_type"])),
        file_record(analysis_path, role="unmodified staged-analyzer execution", artifact_type=str(analysis["artifact_type"])),
    ]
    for path, artifact_type, role in receipt_specs:
        receipt = validate_receipt(path, artifact_type, context=role)
        require(receipt.get("run_id") in {ONLINE_RUN, "e12-deploy-twopass-pass1-evolving-luna-40-v1"}, f"{role} run changed")
        if role == "online recovery 89df":
            require(receipt.get("cell_id") == "89df41e0daa1262a43fa5e55" and receipt.get("malformed_checkpoint") == 6 and receipt.get("final_recovery_max_output_tokens") == 640, "89df recovery facts changed")
        if role == "online recovery 786":
            groups = receipt.get("judge_recovery_groups")
            require(receipt.get("cell_id") == "786d95760ccdb86713c26936" and receipt.get("final_recovery_max_output_tokens") == 640, "786 recovery identity/cap changed")
            require(isinstance(groups, list) and {row.get("checkpoint") for row in groups} == {5, 6} and all(row.get("one_cell_output_cap_deviation", {}).get("recovery_max_output_tokens") == 640 for row in groups), "786 checkpoint-5/checkpoint-6 cap deviations changed")
        if role in {"online forensic audit 89df", "online forensic audit 786"}:
            verdict = receipt.get("verdict")
            process = receipt.get("process_evidence_and_attribution")
            require(receipt.get("provider_calls_made_by_audit") == 0 and isinstance(verdict, Mapping) and isinstance(process, Mapping), f"{role} semantics changed")
            require(verdict.get("artifact_and_accounting_integrity") == "passed" and verdict.get("executor_attribution") == "unknown", f"{role} verdict changed")
            require(process.get("executor_identity") == "unknown" and process.get("execution_bound_script_sha256") is None, f"{role} executor attribution changed")
        records.append(file_record(path, role=role, artifact_type=artifact_type))
    disclosure = {
        "online": "Three passive trace-judge cells spanning two of 40 deployment source tasks required same-prefix semantic recovery after output-cap truncation. Cell 89df recovered checkpoint 6 with cap 640; cell 786 used cap 640 at checkpoints 5 and 6. Executor attribution for the 89df and 786 recovery suffixes is unknown. Raw production artifacts were retained; a hash-bound copy counted every physical attempt and normalized one ordinary HTTP 503 ledger status, and a paired n=38 sensitivity omitted both affected source tasks from all 28 treatments.",
        "two_pass": "The two-pass passive pass had one documented trace-judge recovery. Trace judge was not among the four methods selected for the final active-anchored yoked schedule; the recovery remains part of the reproducibility record.",
    }
    return records, disclosure


def figure_record(svg: Path, *, allocation: str, purpose: str) -> dict[str, Any]:
    record = file_record(svg, role="paper figure")
    record.update({"allocation": allocation, "purpose": purpose})
    sidecar = svg.with_suffix(".data.json")
    if not sidecar.exists() and svg.parent == G:
        sidecar = Path(str(svg) + ".data.json")
    require(sidecar.is_file() and not sidecar.is_symlink(), f"missing figure sidecar for {svg}")
    record["data_sidecar"] = file_record(sidecar, role="figure data sidecar")
    return record


def collect_figures(post_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    appendix: list[dict[str, Any]] = []
    main.append(figure_record(G / "active-probe-ladder-confirmatory-v1.svg", allocation="main_figure_1", purpose="Cross-model active observer-effect trend plus explicitly exploratory burden ladder"))
    main.append(figure_record(G / "observer-overhead-confirmatory-v1.svg", allocation="main_figure_2", purpose="Provider overhead of active and passive observation"))

    pr_paths = sorted((EVOLVING_RUN / "results" / "signal-figures").glob("*.svg")) + sorted((BFCL_RUN / "results" / "signal-figures").glob("*.svg"))
    require(len(pr_paths) == 7, "primary precision-recall figure count changed")
    for path in pr_paths:
        main.append(figure_record(path, allocation="main_figure_3_source_panel", purpose="One of all seven powered signal-quality panels; assemble all panels without cherry-picking"))

    post_dir = post_path.parent
    for stem, allocation, purpose in (
        ("online-performance", "main_figure_4a", "Natural-policy success by observation method and state operator"),
        ("online-firing-actions", "main_figure_4b", "Realized firing/action incidence needed to interpret natural-policy success"),
        ("online-success-interactions", "main_figure_5_if_space", "Specific method-by-operator success interactions relative to active recomputation"),
    ):
        main.append(figure_record(post_dir / f"{stem}.svg", allocation=allocation, purpose=purpose))

    appendix.append(figure_record(G / "active-probe-mechanism-exploratory-v1.svg", allocation="appendix", purpose="Exploratory copy/recall/counter/recompute mechanism effects"))
    for root in (EVOLVING_RUN / "results" / "observer-figures", BFCL_RUN / "results" / "observer-figures"):
        for path in sorted(root.glob("*.svg")):
            appendix.append(figure_record(path, allocation="appendix", purpose="Per-benchmark observer-effect/resource detail"))
    bfcl_sensitivity = sorted((BFCL_RUN / "results" / "signal-figures-complete-case").glob("*.svg"))
    require(len(bfcl_sensitivity) == 3, "BFCL complete-case sensitivity figure count changed")
    for path in bfcl_sensitivity:
        appendix.append(figure_record(path, allocation="appendix", purpose="BFCL complete-case signal sensitivity"))
    for stem, purpose in (
        ("online-resources", "Online end-to-end tokens/cost"),
        ("yoked-controlled-sensitivity", "Aggressive checkpoint-1 active-anchored controlled sensitivity"),
    ):
        appendix.append(figure_record(post_dir / f"{stem}.svg", allocation="appendix", purpose=purpose))
    return main, appendix


def success_views(post: Mapping[str, Any]) -> dict[str, Any]:
    online = post["online_primary"]
    summaries = [dict(row) for row in online["performance_summaries"]]
    require(all(row.get("metric") == "success" and row.get("n_tasks") == 40 for row in summaries), "online performance rows changed")
    tops: list[dict[str, Any]] = []
    for operator in ONLINE_OPERATORS:
        candidates = [row for row in summaries if row["operator"] == operator]
        highest = max(float(row["mean"]) for row in candidates)
        winners = sorted(row["method"] for row in candidates if math.isclose(float(row["mean"]), highest, abs_tol=1e-15))
        tops.append({"operator": operator, "highest_success": highest, "methods": winners, "descriptive_only": True})
    method_effects = [dict(row) for row in online["method_effects"]]
    active_comparisons = [row for row in method_effects if row.get("reference_method") == "active_recompute"]
    require(len(active_comparisons) == 24, "active online comparison coverage changed")
    operating = [dict(row) for row in online["operating_resource_summaries"]]
    incidence = [row for row in operating if row.get("metric") in {"firing_incidence", "action_incidence"}]
    require(len(incidence) == 56, "online firing/action incidence coverage changed")
    operator_effects = [dict(row) for row in online["operator_effects"]]
    interactions = [dict(row) for row in online["method_operator_interactions"] if row.get("reference_method") == "active_recompute"]
    require(len(interactions) == 18, "online active interaction coverage changed")

    yoked = post["yoked_controlled_sensitivity"]
    yoked_success = [dict(row) for row in yoked["summaries"] if row.get("metric") == "success"]
    require(len(yoked_success) == 12, "yoked success coverage changed")
    yoked_tops: list[dict[str, Any]] = []
    for operator in YOKED_OPERATORS:
        candidates = [row for row in yoked_success if row["operator"] == operator]
        highest = max(float(row["mean"]) for row in candidates)
        yoked_tops.append({"operator": operator, "highest_success": highest, "methods": sorted(row["method"] for row in candidates if math.isclose(float(row["mean"]), highest, abs_tol=1e-15)), "descriptive_only": True})
    return {
        "online_natural_policy": {
            "success_summaries": summaries,
            "descriptive_point_estimate_leaders": tops,
            "active_recompute_vs_other_method_effects": active_comparisons,
            "operator_minus_none_effects": operator_effects,
            "active_relative_method_operator_interactions": interactions,
            "firing_and_action_incidence": incidence,
            "interpretation": "These are deployed-policy comparisons under unequal natural scalar firing rates; they combine method identity with realized firing/action behavior.",
        },
        "yoked_checkpoint1_sensitivity": {
            "success_summaries": yoked_success,
            "descriptive_point_estimate_leaders": yoked_tops,
            "operator_minus_none_effects": [dict(row) for row in yoked["operator_effects"] if row.get("metric") == "success"],
            "active_recompute_vs_other_method_effects": [dict(row) for row in yoked["method_effects"] if row.get("reference_method") == "active_recompute"],
            "interpretation": "Controlled sensitivity on the same 40 tasks with one active-anchored action at checkpoint 1; not natural timing and not an independent replication.",
        },
    }


def supported_claims(observer: Mapping[str, Any], signal_winners: Sequence[Mapping[str, Any]], sensitivity: Mapping[str, Any], deployment: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_wins = sum("active_recompute" in row["winning_methods"] for row in signal_winners)
    non_active = len(signal_winners) - active_wins
    sensitivity_assessment = sensitivity["audit_result"]["assessment"]
    leaders = deployment["online_natural_policy"]["descriptive_point_estimate_leaders"]
    leader_text = "; ".join(f"{OPERATOR_LABELS[row['operator']]}: {', '.join(METHOD_LABELS[m] for m in row['methods'])} ({row['highest_success']:.3f})" for row in leaders)
    return [
        {
            "id": "S1",
            "status": "supported_as_trend",
            "claim": observer["wording"],
            "qualification": "Seven model-by-benchmark strata are descriptive replications, not independent samples for a pooled universal claim.",
        },
        {
            "id": "S2",
            "status": "supported",
            "claim": f"Signal quality is conditional: active recomputation has the highest AUPRC in {active_wins}/7 powered slices; another passive method or a context/clock baseline leads {non_active}/7.",
            "qualification": "Active signals are measured on carried trajectories and zero-carry signals on clean trajectories; this is an ecological comparison, not a same-trajectory counterfactual.",
        },
        {
            "id": "S3",
            "status": "supported",
            "claim": "Provider-backed active and passive observation both consume calls, tokens, elapsed compute, and dollars; deterministic trace rules, clocks, and context-use add zero provider calls.",
            "qualification": "Passive provider latency is off the target path and must not be described as agent delay.",
        },
        {
            "id": "S4",
            "status": "supported_for_one_deployment_slice",
            "claim": "Natural-policy deployment is method-by-operator specific. Descriptive leaders were " + leader_text + ".",
            "qualification": "Use the exact paired intervals, action incidence, and interactions—not point-estimate ranks alone. Deployment covers Luna on 40 Evolving Intent tasks only.",
        },
        {
            "id": "S5",
            "status": "sensitivity_result",
            "claim": f"The paired n=38 cumulative two-source-task omission audit assessment is `{sensitivity_assessment}`.",
            "qualification": "Report every outcome/action/resource flag listed in the sensitivity receipt, including any changed conclusion.",
        },
        {
            "id": "S6",
            "status": "supported_as_controlled_sensitivity",
            "claim": "Under the common aggressive checkpoint-1 schedule, the best point estimate depends on the state operator; no method is universally dominant.",
            "qualification": "The yoked study reuses the same 40 source tasks and is neither natural timing nor an independent replication.",
        },
    ]


UNSUPPORTED = [
    {
        "claim": "Active observation always harms task success.",
        "reason": "One of seven powered active-recompute effects is positive; four intervals include zero.",
    },
    {
        "claim": "Observer harm is consistent without exceptions across four or five powered models.",
        "reason": "The result is a 6/7 trend with a positive Luna/Evolving exception, not a rule.",
    },
    {
        "claim": "Active signals are universally less accurate than passive signals or trivial baselines.",
        "reason": "Active recomputation leads AUPRC in some powered slices.",
    },
    {
        "claim": "Passive signals universally outperform clock and context baselines.",
        "reason": "Powered signal winners vary by model and benchmark.",
    },
    {
        "claim": "Increasing active chore complexity monotonically improves detection or worsens observer harm.",
        "reason": "The n=20 mechanism arms are heterogeneous and explicitly exploratory.",
    },
    {
        "claim": "Evolving Intent supplies independently graded within-N-turn degradation labels.",
        "reason": "It supplies verified final success; BFCL supplies turn-level action-failure evidence.",
    },
    {
        "claim": "The deployed feedback is an LLM-generated GOOD/BAD decision critique.",
        "reason": "It is a deterministic, bounded, quote-only WATCH reminder.",
    },
    {
        "claim": "Primary online intervention counts are matched across methods.",
        "reason": "The primary study uses unequal natural scalar firing rates with a one-action cap.",
    },
    {
        "claim": "Deployment results generalize across models or to action traces.",
        "reason": "Deployment covers one model and one reasoning benchmark; BFCL was not deployed.",
    },
    {
        "claim": "The final deployment has a truly unmonitored arm or an oracle bound.",
        "reason": "operator=none is monitored no-action; neither final manifest includes an oracle.",
    },
]


def inventory_records(paths: Mapping[str, Path], parsed: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    roles = {
        "readme": "paper exposition backbone",
        "evolving_score": "powered reasoning signal score",
        "bfcl_score": "powered action signal score",
        "bfcl_sensitivity": "BFCL complete-case signal sensitivity",
        "evolving_validation": "powered reasoning validation",
        "bfcl_validation": "powered action validation",
        "ladder": "powered observer-effect and exploratory ladder",
        "mechanism": "exploratory active mechanism",
        "overhead": "observer overhead",
        "online": "primary online adaptive analysis",
        "yoked": "checkpoint-1 yoked analysis",
        "yoked_validation": "checkpoint-1 yoked validation",
        "post": "deployment paper post-analysis",
        "sensitivity": "paired cumulative leave-two-source-tasks-out sensitivity",
    }
    records = []
    for name, role in roles.items():
        artifact_type = parsed.get(name, {}).get("artifact_type")
        records.append(file_record(paths[name], role=role, artifact_type=str(artifact_type) if artifact_type else None))
    return records


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(payload: Mapping[str, Any], json_sha: str, script_sha: str) -> str:
    samples = payload["sample_sizes"]
    observer = payload["observer_effect"]
    lines = [
        "# Experiment 12 paper materials",
        "",
        "Fail-closed, provider-free inventory for the 8-page workshop paper. Numerical claims below are derived from the listed machine-readable artifacts; this document deliberately corrects unsupported wording in `README.md`.",
        "",
        f"- Companion JSON SHA256: `{json_sha}`",
        f"- Generator SHA256: `{script_sha}`",
        f"- Frozen Experiment 12 code tree: `{payload['frozen_code_tree_sha256']}`",
        "",
        "## Evidence at a glance",
        "",
        "| study | scope | statistical unit / denominator | role |",
        "|---|---|---|---|",
        f"| Powered observation: Evolving Intent | 4 models, 2 trajectory arms, 4 zero-carry shadows | {samples['confirmatory_evolving']['trajectory_tasks_per_model_arm']} tasks/model/arm; {samples['confirmatory_evolving']['signal_tasks_per_model']} signal tasks/model | confirmatory reasoning traces |",
        f"| Powered observation: BFCL | 3 models, 2 trajectory arms, 4 zero-carry shadows | {samples['confirmatory_bfcl']['trajectory_tasks_per_model_arm']} tasks/model/arm; {samples['confirmatory_bfcl']['primary_signal_tasks_per_model']} primary signal tasks/model | confirmatory action traces |",
        f"| Active mechanism | 9 model-benchmark strata, 4 probe variants | n={samples['active_mechanism']['tasks_per_arm_stratum']} paired tasks/arm/stratum | exploratory only |",
        f"| Online deployment | Luna × Evolving Intent, 7 methods × 4 operators | {samples['online_deployment']['source_tasks']} paired source tasks; {samples['online_deployment']['cells']} cells | primary ecological deployment |",
        f"| Yoked deployment | Luna × Evolving Intent, 4 methods × 3 operators | {samples['yoked_deployment']['source_tasks']} paired source tasks; {samples['yoked_deployment']['cells']} cells | checkpoint-1 controlled sensitivity |",
        f"| Recovery sensitivity | remove two affected source tasks from all 28 online treatments | {samples['leave_two_units']['source_tasks']} → {samples['leave_two_units']['filtered_tasks']} tasks | cumulative robustness audit |",
        "",
        "## Sharp conclusions supported by the data",
        "",
    ]
    for claim in payload["claims"]["supported"]:
        lines.extend([f"- **{claim['status']}:** {claim['claim']}", f"  - Qualification: {claim['qualification']}"])
    lines.extend(
        [
            "",
            "## Powered active observer effect",
            "",
            observer["headline"]["wording"],
            "",
            "| benchmark | model | clean success | active success | active − clean | paired 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in observer["effects"]:
        lines.append(f"| {row['benchmark']} | {MODEL_LABELS.get(row['model'], row['model'])} | {row['clean_success']:.3f} | {row['active_success']:.3f} | {row['effect']:+.3f} | [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |")

    lines.extend(["", "## Signal-quality winners (AUPRC)", "", "No universal method wins. Full precision, recall, AUPRC, thresholds, and denominators are in the companion JSON.", "", "| benchmark | model | analyzable tasks | highest AUPRC | method(s) |", "|---|---|---:|---:|---|"])
    for row in payload["signal_quality"]["slice_winners"]:
        labels = ", ".join(METHOD_LABELS[m] for m in row["winning_methods"])
        lines.append(f"| {row['benchmark']} | {MODEL_LABELS.get(row['model'], row['model'])} | {row['n_tasks']} | {row['highest_auprc']:.3f} | {labels} |")

    lines.extend(["", "## Online deployment: exact success estimates", "", "These are natural-policy results. Firing/action incidence must be shown beside them.", "", "| operator | method | success | paired 95% CI |", "|---|---|---:|---:|"])
    for row in payload["deployment"]["online_natural_policy"]["success_summaries"]:
        lines.append(f"| {OPERATOR_LABELS[row['operator']]} | {METHOD_LABELS[row['method']]} | {row['mean']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")

    lines.extend(["", "### Active versus other methods within each operator", "", "Positive effects favor the comparison method.", "", "| operator | comparison − active | effect | paired 95% CI |", "|---|---|---:|---:|"])
    for row in payload["deployment"]["online_natural_policy"]["active_recompute_vs_other_method_effects"]:
        lines.append(f"| {OPERATOR_LABELS[row['operator']]} | {METHOD_LABELS[row['comparison_method']]} − active | {row['effect']:+.3f} | [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |")

    sensitivity = payload["cumulative_two_unit_sensitivity"]
    lines.extend(
        [
            "",
            "## Robustness and required disclosures",
            "",
            f"- Paired cumulative n=38 assessment: **{sensitivity['audit_result']['assessment']}**.",
            f"- Scientific-outcome change flags: {sensitivity['audit_result']['scientific_outcome_change_count']}; action-policy flags: {sensitivity['audit_result']['action_policy_change_count']}; resource flags: {sensitivity['audit_result']['resource_sensitivity_count']}.",
            f"- {payload['disclosures']['online']}",
            f"- {payload['disclosures']['two_pass']}",
            "- Evolving Intent has final-success labels, not independently graded within-horizon failure labels. BFCL supplies the action-trace/turn-level early-warning evidence.",
            "- `good_bad_watch_feedback` is a deterministic quote-only WATCH reminder—not an LLM-generated assessment of good and bad decisions.",
            "",
            "## Claims the paper must not make",
            "",
        ]
    )
    for row in payload["claims"]["unsupported"]:
        lines.append(f"- ~~{row['claim']}~~ {row['reason']}")

    lines.extend(["", "## Recommended figure allocation", "", "### Main paper", ""])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload["figures"]["main"]:
        grouped[row["allocation"]].append(row)
    for allocation, rows in grouped.items():
        lines.append(f"- **{allocation}:** {rows[0]['purpose']}")
        for row in rows:
            lines.append(f"  - `{row['path']}` (`{row['sha256']}`)")
    lines.extend(["", "For the seven precision-recall source panels, assemble all four Evolving plus all three BFCL panels into one figure; showing a hand-picked subset would be misleading.", "", "### Appendix", ""])
    for row in payload["figures"]["appendix"]:
        lines.append(f"- `{row['path']}` — {row['purpose']}")
    lines.extend(["", "## Immutable material inventory", "", "Every input, validation, receipt, figure, and sidecar path has an exact SHA256 and byte size in `PAPER_MATERIALS12.json`.", ""])
    return "\n".join(lines)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(paths: Mapping[str, Path]) -> dict[str, Any]:
    code_hash = validate_code_tree()
    backbone = paper_section(paths["readme"])
    evolving_manifest = "b0be97ed6b01c9c3b003a5900b8e596772fffe3c68c4b9cdd47f46ff329c7056"
    evolving_pairs = "ccb98c678dc0d9ff9caee539ccd9859aa406abffc16b4bb9eaaf0abfd0bb6a6c"
    bfcl_manifest = "551a4574a1fe502c9304feff46c956ffb6b19a96af84d54d3e18d3a60fd919c3"
    bfcl_pairs = "3f2802f6f7471a65f758b0a8c60fc60a5a0334e906732c1f1560c1b31e990be4"
    evolving_models = ("deepseek-v4-flash-0731", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-oss-120b")
    bfcl_models = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-oss-120b")
    evolving_validation, evolving_metrics, evolving_winners = validate_confirmation(
        paths["evolving_validation"], paths["evolving_score"], run_id="e12-confirmatory-evolving-core-v2",
        manifest_sha=evolving_manifest, pair_sha=evolving_pairs, benchmark="evolving_intent_gsm8k",
        models=evolving_models, score_n={model: 56 for model in evolving_models}, expected_cells=448, expected_shadows=224,
    )
    bfcl_validation, bfcl_metrics, bfcl_winners = validate_confirmation(
        paths["bfcl_validation"], paths["bfcl_score"], run_id="e12-confirmatory-bfcl-core-v3",
        manifest_sha=bfcl_manifest, pair_sha=bfcl_pairs, benchmark="bfcl_multi_turn",
        models=bfcl_models, score_n={model: 52 for model in bfcl_models}, expected_cells=336, expected_shadows=168,
    )
    bfcl_sensitivity = read_json(paths["bfcl_sensitivity"], context="BFCL complete-case sensitivity")
    require(bfcl_sensitivity.get("source_run_id") == "e12-confirmatory-bfcl-core-v3" and len(bfcl_sensitivity.get("metrics", [])) == 24, "BFCL sensitivity changed")
    ladder, effects, observer_headline = validate_ladder(paths["ladder"])
    mechanism, mechanism_summary = validate_mechanism(paths["mechanism"])
    overhead, overhead_aggregates, overhead_comparison = validate_overhead(paths["overhead"])

    online = validate_online(paths["online"])
    yoked, yoked_validation = validate_yoked(paths["yoked"], paths["yoked_validation"])
    post = validate_post(paths["post"], paths["online"], paths["yoked"])
    sensitivity = validate_sensitivity(paths["sensitivity"], paths["online"])
    receipt_records, disclosures = validate_audit_receipts(paths["staging_receipt"], paths["analysis_receipt"], paths["online"])
    deployment = success_views(post)
    main_figures, appendix_figures = collect_figures(paths["post"])

    parsed = {
        "evolving_score": read_json(paths["evolving_score"], context="evolving score inventory"),
        "bfcl_score": read_json(paths["bfcl_score"], context="BFCL score inventory"),
        "bfcl_sensitivity": bfcl_sensitivity,
        "evolving_validation": evolving_validation,
        "bfcl_validation": bfcl_validation,
        "ladder": ladder,
        "mechanism": mechanism,
        "overhead": overhead,
        "online": online,
        "yoked": yoked,
        "yoked_validation": yoked_validation,
        "post": post,
        "sensitivity": sensitivity,
    }
    evidence_inventory = inventory_records(paths, parsed)
    script_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "experiment12_final_paper_material_inventory",
        "provider_calls_made": 0,
        "frozen_code_tree_sha256": code_hash,
        "paper_backbone": backbone,
        "generator": file_record(script_path, role="paper-material generator"),
        "sample_sizes": {
            "confirmatory_evolving": {"models": list(evolving_models), "trajectory_tasks_per_model_arm": 56, "trajectory_outputs": 448, "shadow_outputs": 224, "signal_tasks_per_model": 56},
            "confirmatory_bfcl": {"models": list(bfcl_models), "trajectory_tasks_per_model_arm": 56, "trajectory_outputs": 336, "shadow_outputs": 168, "primary_signal_tasks_per_model": 52, "complete_case_signal_tasks": {"gpt-5.6-luna": 51, "gpt-5.6-terra": 52, "gpt-oss-120b": 52}},
            "active_mechanism": {"model_benchmark_strata": 9, "probe_arms": 4, "tasks_per_arm_stratum": 20, "paired_task_rows": 720},
            "observer_overhead": {"model_benchmark_strata": 7, "model_tasks": 392, "methods": 8, "task_method_rows": 3136},
            "online_deployment": {"model": "gpt-5.6-luna", "benchmark": "evolving_intent_gsm8k", "source_tasks": 40, "methods": 7, "operators": 4, "cells": 1120},
            "yoked_deployment": {"model": "gpt-5.6-luna", "benchmark": "evolving_intent_gsm8k", "source_tasks": 40, "methods": 4, "operators": 3, "cells": 480, "action_checkpoint": 1},
            "leave_two_units": {"source_tasks": 40, "filtered_tasks": 38, "treatments": 28, "removed_rows": 56, "affected_source_tasks": ["extracted-gsm8k-test-814::t7", "extracted-gsm8k-test-989::t7"]},
        },
        "observer_effect": {"headline": observer_headline, "effects": effects},
        "active_probe_mechanism": mechanism_summary,
        "signal_quality": {
            "comparison_semantics": "ecological class comparison: active metrics use carried trajectories; passive/baseline metrics use clean trajectories",
            "slice_winners": [*evolving_winners, *bfcl_winners],
            "all_primary_metrics": [*evolving_metrics, *bfcl_metrics],
            "bfcl_complete_case_sensitivity_path": relative(paths["bfcl_sensitivity"]),
        },
        "observer_overhead": {"method_aggregates": overhead_aggregates, **overhead_comparison},
        "deployment": deployment,
        "cumulative_two_unit_sensitivity": {
            "source_analysis_sha256": sensitivity["source"]["analysis_sha256"],
            "design": sensitivity["design"],
            "affected_units_selection": sensitivity["affected_units_selection"],
            "audit_result": sensitivity["audit_result"],
            "material_operator_effect_changes": sensitivity["comparisons"]["material_operator_effect_changes"],
            "material_absolute_summary_shifts": sensitivity["comparisons"]["material_absolute_summary_shifts"],
        },
        "claims": {
            "supported": supported_claims(observer_headline, [*evolving_winners, *bfcl_winners], sensitivity, deployment),
            "unsupported": UNSUPPORTED,
        },
        "recovery_audit_summary": {
            "recovered_cells": [
                {"cell_id": "d52046b6eb74a76ecdc3debc", "source_task": "extracted-gsm8k-test-814::t7", "method": "trace_judge"},
                {"cell_id": "89df41e0daa1262a43fa5e55", "source_task": "extracted-gsm8k-test-814::t7", "method": "trace_judge", "recovered_checkpoints": [6], "recovery_max_output_tokens": 640, "executor_attribution": "unknown"},
                {"cell_id": "786d95760ccdb86713c26936", "source_task": "extracted-gsm8k-test-989::t7", "method": "trace_judge", "recovered_checkpoints": [5, 6], "recovery_max_output_tokens": 640, "executor_attribution": "unknown"},
            ],
            "affected_source_tasks": ["extracted-gsm8k-test-814::t7", "extracted-gsm8k-test-989::t7"],
            "ordinary_transport_normalization": {"count": 1, "kind": "HTTP 503 ledger-status normalization"},
            "analysis_disposition": "retain primary n=40 results with complete physical-attempt accounting, explicit deviations, and the cumulative paired n=38 sensitivity",
        },
        "disclosures": disclosures,
        "figures": {"main": main_figures, "appendix": appendix_figures},
        "evidence_inventory": evidence_inventory,
        "audit_receipt_inventory": receipt_records,
        "scope_boundary": {
            "observation_generalization": "seven powered model-by-benchmark slices: four reasoning, three action",
            "deployment_generalization": "one model on one reasoning benchmark",
            "feedback_operator": "deterministic bounded quote-only WATCH reminder",
            "online_rates": "natural scalar thresholds; unequal realized firing/action rates",
            "yoked_schedule": "active-anchored checkpoint 1; controlled sensitivity, not independent replication",
        },
    }
    return payload


def dry_check(paths: Mapping[str, Path]) -> dict[str, Any]:
    code_hash = validate_code_tree()
    paper_section(paths["readme"])
    static_names = (
        "evolving_score", "bfcl_score", "bfcl_sensitivity", "evolving_validation",
        "bfcl_validation", "ladder", "mechanism", "overhead", "yoked", "yoked_validation",
    )
    for name in static_names:
        regular(paths[name], context=f"static input {name}")
    evolving_models = ("deepseek-v4-flash-0731", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-oss-120b")
    bfcl_models = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-oss-120b")
    validate_confirmation(
        paths["evolving_validation"], paths["evolving_score"],
        run_id="e12-confirmatory-evolving-core-v2",
        manifest_sha="b0be97ed6b01c9c3b003a5900b8e596772fffe3c68c4b9cdd47f46ff329c7056",
        pair_sha="ccb98c678dc0d9ff9caee539ccd9859aa406abffc16b4bb9eaaf0abfd0bb6a6c",
        benchmark="evolving_intent_gsm8k", models=evolving_models,
        score_n={model: 56 for model in evolving_models}, expected_cells=448,
        expected_shadows=224,
    )
    validate_confirmation(
        paths["bfcl_validation"], paths["bfcl_score"],
        run_id="e12-confirmatory-bfcl-core-v3",
        manifest_sha="551a4574a1fe502c9304feff46c956ffb6b19a96af84d54d3e18d3a60fd919c3",
        pair_sha="3f2802f6f7471a65f758b0a8c60fc60a5a0334e906732c1f1560c1b31e990be4",
        benchmark="bfcl_multi_turn", models=bfcl_models,
        score_n={model: 52 for model in bfcl_models}, expected_cells=336,
        expected_shadows=168,
    )
    bfcl_sensitivity = read_json(paths["bfcl_sensitivity"], context="BFCL complete-case sensitivity")
    require(bfcl_sensitivity.get("source_run_id") == "e12-confirmatory-bfcl-core-v3" and len(bfcl_sensitivity.get("metrics", [])) == 24, "BFCL sensitivity changed")
    validate_ladder(paths["ladder"])
    validate_mechanism(paths["mechanism"])
    validate_overhead(paths["overhead"])
    validate_yoked(paths["yoked"], paths["yoked_validation"])
    final_names = ("online", "post", "sensitivity", "staging_receipt", "analysis_receipt")
    readiness = {name: paths[name].is_file() and not paths[name].is_symlink() for name in final_names}
    return {
        "dry_check": "passed",
        "provider_calls_made": 0,
        "frozen_code_tree_sha256": code_hash,
        "static_inputs_ready": True,
        "final_inputs_ready": readiness,
        "ready_for_full_build": all(readiness.values()),
        "full_command": "python3 experiments12/scripts/posthoc/build_paper_materials12.py",
        "outputs": [relative(paths["output_json"]), relative(paths["output_md"])],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dry-check", action="store_true")
    for name, default in DEFAULTS.items():
        result.add_argument("--" + name.replace("_", "-"), type=Path, default=default)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = {name: getattr(args, name).resolve() for name in DEFAULTS}
    try:
        if args.dry_check:
            print(json.dumps(dry_check(paths), indent=2, sort_keys=True))
            return 0
        payload = build(paths)
        json_text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        atomic_text(paths["output_json"], json_text)
        json_sha = sha256_file(paths["output_json"])
        script_sha = sha256_file(Path(__file__).resolve())
        atomic_text(paths["output_md"], markdown(payload, json_sha, script_sha))
        require(validate_code_tree() == EXPECTED_CODE_TREE_SHA256, "code tree changed during build")
        print(
            json.dumps(
                {
                    "json": relative(paths["output_json"]),
                    "json_sha256": sha256_file(paths["output_json"]),
                    "markdown": relative(paths["output_md"]),
                    "markdown_sha256": sha256_file(paths["output_md"]),
                    "script_sha256": script_sha,
                    "provider_calls_made": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (MaterialsError, FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
