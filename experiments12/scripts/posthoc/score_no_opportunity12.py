#!/usr/bin/env python3
"""Score frozen confirmatory traces when an event precedes the first checkpoint.

This is an analysis-only clarification. It never changes the immutable extract:
tasks with no pre-event observation opportunity remain positive, cannot fire,
and therefore contribute false negatives to full-task recall. Fixed-rate firing
slots are assigned only among tasks that could actually have emitted a warning.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path

from experiments12.analysis12 import (
    ANALYSIS_VERSION,
    _trace_from_dict,
    require_source_task_disjointness,
    score_locked,
    verify_threshold_binding,
    write_signal_figures,
)
from experiments12.core.artifacts import atomic_write_json, read_json, sha256_file
from experiments12.metrics12 import (
    PredictionMetrics,
    _calibration,
    _lead_time,
    _pr_curve,
    _tie_break_value,
    collapse_task_predictions,
    prediction_metrics,
)


RULE_VERSION = 1


def _score_with_no_opportunity(traces, locked):
    predictions = collapse_task_predictions(traces)
    without_opportunity = [
        item for item in predictions if not item.has_actionable_checkpoint
    ]
    if not without_opportunity:
        return prediction_metrics(traces, locked_threshold=locked), None

    eligible = [item for item in predictions if item.has_actionable_checkpoint]
    target_count = math.floor(
        locked.target_firing_rate * len(predictions) + 1e-12
    )
    if target_count > len(eligible):
        raise ValueError("fixed firing count exceeds warning-eligible tasks")
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item.score,
            _tie_break_value(
                locked.tie_break_seed,
                item.model,
                item.benchmark,
                item.method,
                item.task_id,
            ),
            item.task_id,
        ),
    )
    fired = ranked[:target_count]
    boundary = (
        math.nextafter(max(item.score for item in eligible), math.inf)
        if not fired
        else fired[-1].score
    )
    pr_curve, auprc = _pr_curve(predictions)
    # Calibration is opportunity-conditional: absence of a possible forecast is
    # not silently converted into an observed probability of zero.
    calibration, brier, ece = _calibration(eligible, 10)
    positives = sum(item.label for item in predictions)
    true_positives = sum(item.label for item in fired)
    false_positives = len(fired) - true_positives
    summary = PredictionMetrics(
        model=traces[0].model,
        benchmark=traces[0].benchmark,
        method=traces[0].method,
        split=traces[0].split,
        n_tasks=len(predictions),
        n_positive_tasks=positives,
        locked_threshold=locked.threshold,
        threshold_source="calibration_locked_fixed_rate",
        selection_rule=locked.selection_rule,
        target_firing_rate=locked.target_firing_rate,
        realized_score_boundary=boundary,
        precision=true_positives / len(fired) if fired else None,
        recall=true_positives / positives,
        true_positive_tasks=true_positives,
        false_positive_tasks=false_positives,
        firing_rate=len(fired) / len(predictions),
        auprc=auprc,
        brier=brier,
        expected_calibration_error=ece,
        calibration_bins=calibration,
        pr_curve=pr_curve,
        lead_time=_lead_time(
            traces,
            boundary,
            fired_task_ids={item.task_id for item in fired},
        ),
    )
    detail = {
        "model": summary.model,
        "benchmark": summary.benchmark,
        "method": summary.method,
        "n_tasks": len(predictions),
        "n_warning_eligible": len(eligible),
        "n_without_warning_opportunity": len(without_opportunity),
        "without_warning_opportunity_task_ids": sorted(
            item.task_id for item in without_opportunity
        ),
        "fixed_fire_count": len(fired),
        "calibration_denominator": len(eligible),
        "full_task_recall_denominator": positives,
    }
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--sensitivity-output", required=True)
    parser.add_argument("--sensitivity-figures", required=True)
    args = parser.parse_args()

    source = read_json(args.input)
    locked_artifact = read_json(args.thresholds)
    threshold_sha = sha256_file(args.thresholds)
    thresholds, required_slices, calibration_tasks = verify_threshold_binding(
        source,
        locked_artifact,
        threshold_artifact_sha256=threshold_sha,
    )
    traces = tuple(_trace_from_dict(row) for row in source["signal_traces"])
    require_source_task_disjointness(traces, calibration_tasks)
    threshold_by_key = {
        (row.model, row.benchmark, row.method): row for row in thresholds
    }
    groups = defaultdict(list)
    for trace in traces:
        groups[(trace.model, trace.benchmark, trace.method)].append(trace)
    if set(groups) != set(required_slices):
        raise ValueError("confirmatory slices differ from the frozen threshold set")

    summaries = []
    clarifications = []
    excluded_blocks = set()
    for key in sorted(groups):
        group = tuple(groups[key])
        summary, detail = _score_with_no_opportunity(
            group, threshold_by_key[key]
        )
        summaries.append(summary)
        if detail is not None:
            clarifications.append(detail)
            excluded_blocks.update(
                (summary.model, task_id)
                for task_id in detail["without_warning_opportunity_task_ids"]
            )

    primary = {
        "analysis_version": ANALYSIS_VERSION,
        "no_opportunity_rule_version": RULE_VERSION,
        "source_run_id": source["run_id"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_extract_sha256": sha256_file(args.input),
        "threshold_artifact_sha256": threshold_sha,
        "threshold_source_run_id": locked_artifact["source_run_id"],
        "threshold_source_manifest_sha256": locked_artifact[
            "source_manifest_sha256"
        ],
        "post_hoc_global_scoring_clarification": {
            "rule": (
                "A task whose event precedes the first observation checkpoint "
                "remains in the full-task denominator, is ineligible to fire, "
                "and contributes a no-fire false negative. Fixed-rate slots are "
                "ranked only among tasks with a genuine warning opportunity."
            ),
            "applies_to_all_models_benchmarks_methods": True,
            "outcome_or_task_identity_used_to_define_rule": False,
            "calibration_is_opportunity_conditional": True,
            "clarified_slices": clarifications,
        },
        "metrics": [asdict(row) for row in summaries],
    }
    atomic_write_json(args.output, primary)
    write_signal_figures(summaries, Path(args.figures))

    filtered = tuple(
        trace
        for trace in traces
        if (trace.model, trace.task_id) not in excluded_blocks
    )
    sensitivity = score_locked(
        filtered,
        thresholds,
        required_method_slices=required_slices,
    )
    atomic_write_json(
        args.sensitivity_output,
        {
            "analysis_version": ANALYSIS_VERSION,
            "sensitivity": "complete_case_model_task_block_exclusion",
            "source_run_id": source["run_id"],
            "source_extract_sha256": sha256_file(args.input),
            "threshold_artifact_sha256": threshold_sha,
            "excluded_model_task_blocks": [
                {"model": model, "task_id": task_id}
                for model, task_id in sorted(excluded_blocks)
            ],
            "metrics": [asdict(row) for row in sensitivity],
        },
    )
    write_signal_figures(sensitivity, Path(args.sensitivity_figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
