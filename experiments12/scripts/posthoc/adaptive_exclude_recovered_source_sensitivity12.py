"""Paired sensitivity excluding the one recovered source task everywhere.

The exclusion is deliberately symmetric: the source task is removed from all
7 observation methods x 4 operators (28 treatments), then the frozen paired
summary code is rerun on the remaining 39 source tasks per treatment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments12.adaptive_analysis12 import (
    ADAPTIVE_ANALYSIS_TYPE,
    summarize_adaptive_outcomes,
    write_adaptive_figures,
)
from experiments12.core.artifacts import atomic_write_json, read_json, sha256_file


RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
SOURCE_TASK_ID = "extracted-gsm8k-test-814::t7"
EXPECTED_FULL_ROWS = 1_120
EXPECTED_TREATMENTS = 28
EXPECTED_REMAINING_PER_TREATMENT = 39
SENSITIVITY_TYPE = "experiment12_online_adaptive_recovered_source_exclusion"


def extract_sensitivity(
    analysis: Mapping[str, Any],
    *,
    source_analysis_sha256: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if (
        analysis.get("artifact_type") != ADAPTIVE_ANALYSIS_TYPE
        or analysis.get("source_run_id") != RUN_ID
    ):
        raise ValueError("input is not the staged stock online analysis")
    rows = analysis.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FULL_ROWS:
        raise ValueError("stock analysis does not contain all 1,120 declared cells")
    excluded = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("task_id") == SOURCE_TASK_ID
    ]
    remaining = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("task_id") != SOURCE_TASK_ID
    ]
    all_treatments = {
        (str(row["method"]), str(row["operator"]))
        for row in rows
        if isinstance(row, Mapping)
    }
    excluded_treatments = {
        (str(row["method"]), str(row["operator"])) for row in excluded
    }
    if (
        len(all_treatments) != EXPECTED_TREATMENTS
        or len(excluded) != EXPECTED_TREATMENTS
        or len(excluded_treatments) != EXPECTED_TREATMENTS
        or excluded_treatments != all_treatments
        or len({row["cell_id"] for row in excluded}) != EXPECTED_TREATMENTS
    ):
        raise ValueError("recovered source was not excluded once from every treatment")
    counts: dict[tuple[str, str], int] = {}
    for row in remaining:
        key = (str(row["method"]), str(row["operator"]))
        counts[key] = counts.get(key, 0) + 1
    if set(counts) != all_treatments or set(counts.values()) != {
        EXPECTED_REMAINING_PER_TREATMENT
    }:
        raise ValueError("paired exclusion left an unbalanced treatment matrix")
    summaries, effects = summarize_adaptive_outcomes(
        remaining,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "artifact_type": SENSITIVITY_TYPE,
        "sensitivity_version": 1,
        "source_run_id": RUN_ID,
        "source_analysis_sha256": source_analysis_sha256,
        "exclusion_reason": "single cell required semantic judge-attempt recovery",
        "excluded_source_task_id": SOURCE_TASK_ID,
        "exclusion_scope": "all_observation_method_operator_treatments",
        "treatments": EXPECTED_TREATMENTS,
        "excluded_rows": EXPECTED_TREATMENTS,
        "remaining_rows": len(remaining),
        "remaining_source_tasks_per_treatment": EXPECTED_REMAINING_PER_TREATMENT,
        "excluded_cell_ids": sorted(row["cell_id"] for row in excluded),
        "balanced_paired_design_after_exclusion": True,
        "rows": remaining,
        "metric_summaries": [asdict(row) for row in summaries],
        "operator_effects": [asdict(row) for row in effects],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--analysis", required=True)
    root.add_argument("--analysis-sha256", required=True)
    root.add_argument("--output", required=True)
    root.add_argument("--figures")
    root.add_argument("--bootstrap-iterations", type=int, default=2_000)
    root.add_argument("--bootstrap-seed", type=int, default=12_012)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if sha256_file(args.analysis) != args.analysis_sha256:
            raise ValueError("stock staged analysis hash changed")
        sensitivity = extract_sensitivity(
            read_json(args.analysis),
            source_analysis_sha256=args.analysis_sha256,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
        atomic_write_json(args.output, sensitivity)
        if args.figures:
            figure_payload = {
                "artifact_type": ADAPTIVE_ANALYSIS_TYPE,
                "deployment_mode": "online_adaptive",
                "metric_summaries": sensitivity["metric_summaries"],
            }
            write_adaptive_figures(figure_payload, args.figures)
        return 0
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
