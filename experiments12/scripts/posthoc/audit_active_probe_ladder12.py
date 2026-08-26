"""Independent provider-free audit for active-probe-ladder-confirmatory-v1.

This does not import the ladder builder or analysis/metrics helpers. It starts
from frozen pair manifests and trajectory success labels, independently repeats
the deterministic paired bootstrap, and checks the JSON, CSV, figure sidecar,
SVG semantics, source/output hashes, sample tiers, signs, and monotonicity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from statistics import fmean
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments12"
ARTIFACTS = PACKAGE / "data_results" / "runs"
STEM = PACKAGE / "data_results" / "derived" / "active-probe-ladder-confirmatory-v1"
EXPECTED_CODE_HASH = "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
PROBES = (
    "active_name_copy",
    "active_name_recall",
    "active_counter",
    "active_recompute",
)
SOURCES = (
    ("exploratory_mechanism", "e12-baseline-evolving-allarms-allmodels-v1", "evolving_intent_gsm8k", 20, 5),
    ("exploratory_mechanism", "e12-baseline-bfcl-allarms-fourmodels-v2", "bfcl_multi_turn", 20, 4),
    ("confirmatory_powered", "e12-confirmatory-evolving-core-v2", "evolving_intent_gsm8k", 56, 4),
    ("confirmatory_powered", "e12-confirmatory-bfcl-core-v3", "bfcl_multi_turn", 56, 3),
)


class AuditFailure(AssertionError):
    pass


def require(condition: object, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} is not an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def code_tree_hash() -> str:
    records = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".sqlite3"}:
            continue
        if any(part in {"artifacts", "external", "generated"} for part in path.parts):
            continue
        records.append({"path": str(path.relative_to(PACKAGE)), "sha256": sha_file(path)})
    return sha_json(records)


def bootstrap_index(
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


def quantile(values: list[float], probability: float) -> float:
    values.sort()
    position = probability * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def close(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def raw_outcomes(run_id: str) -> tuple[dict[tuple[str, str, str, str], float], set[str]]:
    root = ARTIFACTS / run_id
    rows: dict[tuple[str, str, str, str], float] = {}
    tasks: set[str] = set()
    for cell in read_jsonl(root / "pairs.jsonl"):
        pair = cell["pair_key"]
        task = f"{pair['task_id']}/r{pair['replicate_id']}"
        trajectory = read_json(root / "trajectories" / f"{cell['cell_id']}.json")
        success = trajectory["evaluation"]["success"]
        require(trajectory["complete"] is True and isinstance(success, bool), f"{run_id} invalid trajectory")
        key = (str(pair["model"]), str(pair["domain"]), task, str(cell["arm"]))
        require(key not in rows, f"{run_id} duplicate raw outcome")
        rows[key] = float(success)
        tasks.add(task)
    return rows, tasks


def recompute_row(
    raw: dict[tuple[str, str, str, str], float],
    *,
    model: str,
    benchmark: str,
    arm: str,
    n_tasks: int,
) -> dict[str, float]:
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for (candidate_model, candidate_benchmark, task, candidate_arm), value in raw.items():
        if candidate_model == model and candidate_benchmark == benchmark and candidate_arm in {"clean", arm}:
            paired[task][candidate_arm] = value
    require(len(paired) == n_tasks, f"{model}/{benchmark}/{arm} task count")
    require(all(set(values) == {"clean", arm} for values in paired.values()), f"{model}/{benchmark}/{arm} pairing")
    ordered = [paired[task] for task in sorted(paired)]
    clean = [values["clean"] for values in ordered]
    active = [values[arm] for values in ordered]
    differences = [active_value - clean_value for clean_value, active_value in zip(clean, active)]
    bootstrap = [
        fmean(
            differences[bootstrap_index(12_012, model, benchmark, iteration, draw, n_tasks)]
            for draw in range(n_tasks)
        )
        for iteration in range(2_000)
    ]
    return {
        "clean_success": fmean(clean),
        "active_success": fmean(active),
        "effect": fmean(differences),
        "ci_low": quantile(bootstrap, 0.025),
        "ci_high": quantile(bootstrap, 0.975),
    }


def point_sign(value: float) -> str:
    return "negative" if value < -1e-12 else "positive" if value > 1e-12 else "zero"


def interval_sign(low: float, high: float) -> str:
    return "negative" if high < 0.0 else "positive" if low > 0.0 else "includes_zero"


def sign_summary(rows: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, int]]:
    points = Counter(point_sign(float(row["effect"])) for row in rows)
    intervals = Counter(interval_sign(float(row["ci_low"]), float(row["ci_high"])) for row in rows)
    return (
        {"negative": points["negative"], "zero": points["zero"], "positive": points["positive"]},
        {"negative": intervals["negative"], "includes_zero": intervals["includes_zero"], "positive": intervals["positive"]},
    )


def main() -> None:
    require(code_tree_hash() == EXPECTED_CODE_HASH, "frozen code/config hash changed")
    payload_path = STEM.with_suffix(".json")
    payload = read_json(payload_path)
    receipt_path = STEM.with_suffix(".receipt.json")
    receipt = read_json(receipt_path)
    require(receipt["provider_calls_made"] == 0, "receipt reports provider calls")
    builder_path = ROOT / str(receipt["builder"])
    require(receipt["builder_sha256"] == sha_file(builder_path), "builder hash mismatch")
    require(receipt["code_tree_sha256_before"] == receipt["code_tree_sha256_after"] == EXPECTED_CODE_HASH, "receipt code hash")
    for path, expected in receipt["source_extracts"].items():
        require(sha_file(ROOT / path) == expected, f"source extract hash mismatch: {path}")
    for path, expected in receipt["outputs"].items():
        require(sha_file(ROOT / path) == expected, f"output hash mismatch: {path}")

    rows = payload["rows"]
    require(isinstance(rows, list) and len(rows) == 43, "effect row count")
    keys = {(row["study"], row["benchmark"], row["model"], row["active_arm"]) for row in rows}
    require(len(keys) == len(rows), "effect row keys are not unique")
    source_by_run = {row["run_id"]: row for row in payload["source_runs"]}
    raw_by_run = {}
    tasks_by_tier_benchmark = {}
    reconstructed = 0
    for study, run_id, benchmark, n_tasks, n_models in SOURCES:
        source = source_by_run[run_id]
        require(source["study"] == study and source["benchmark"] == benchmark, f"{run_id} source identity")
        require(source["n_tasks_per_model_arm"] == n_tasks, f"{run_id} source sample size")
        require(source["manifest_sha256"] == sha_file(ARTIFACTS / run_id / "manifest.json"), f"{run_id} manifest hash")
        require(source["pairs_sha256"] == sha_file(ARTIFACTS / run_id / "pairs.jsonl"), f"{run_id} pairs hash")
        manifest = read_json(ARTIFACTS / run_id / "manifest.json")
        require(manifest["repository"]["code_tree_sha256"] == EXPECTED_CODE_HASH, f"{run_id} manifest code hash")
        require(len(manifest["models"]) == n_models, f"{run_id} model count")
        raw, task_ids = raw_outcomes(run_id)
        raw_by_run[run_id] = raw
        tasks_by_tier_benchmark[(study, benchmark)] = task_ids
        require(len(task_ids) == n_tasks, f"{run_id} unique task count")
        require(source["task_identities_sha256"] == sha_json(sorted(task_ids)), f"{run_id} task digest")
        source_rows = [row for row in rows if row["run_id"] == run_id]
        expected_arms = set(PROBES) if study == "exploratory_mechanism" else {"active_recompute"}
        require(len(source_rows) == n_models * len(expected_arms), f"{run_id} output row count")
        for row in source_rows:
            require(row["active_arm"] in expected_arms and row["n_tasks"] == n_tasks, f"{run_id} output design")
            computed = recompute_row(
                raw,
                model=str(row["model"]),
                benchmark=benchmark,
                arm=str(row["active_arm"]),
                n_tasks=n_tasks,
            )
            for field, value in computed.items():
                require(close(row[field], value), f"{run_id}/{row['model']}/{row['active_arm']} {field}")
            require(close(row["effect_percentage_points"], float(row["effect"]) * 100), "percentage-point effect")
            require(row["point_sign"] == point_sign(float(row["effect"])), "point sign")
            require(row["strict_ci_sign"] == interval_sign(float(row["ci_low"]), float(row["ci_high"])), "interval sign")
            require(row["ci_excludes_zero"] is (row["strict_ci_sign"] != "includes_zero"), "CI flag")
            reconstructed += 1

    for benchmark in ("evolving_intent_gsm8k", "bfcl_multi_turn"):
        exploratory_tasks = tasks_by_tier_benchmark[("exploratory_mechanism", benchmark)]
        confirmatory_tasks = tasks_by_tier_benchmark[("confirmatory_powered", benchmark)]
        require(not exploratory_tasks.intersection(confirmatory_tasks), f"{benchmark} tier task overlap")
    require(reconstructed == 43, "raw reconstructed row count")

    exploratory = [row for row in rows if row["study"] == "exploratory_mechanism"]
    confirmatory = [row for row in rows if row["study"] == "confirmatory_powered"]
    require(len(exploratory) == 36 and len(confirmatory) == 7, "tier row counts")
    explore_points, explore_intervals = sign_summary(exploratory)
    confirm_points, confirm_intervals = sign_summary(confirmatory)
    require(explore_points == {"negative": 18, "zero": 8, "positive": 10}, "exploratory sign count")
    require(explore_intervals == {"negative": 4, "includes_zero": 31, "positive": 1}, "exploratory CI sign count")
    require(confirm_points == {"negative": 6, "zero": 0, "positive": 1}, "confirmatory sign count")
    require(confirm_intervals == {"negative": 3, "includes_zero": 4, "positive": 0}, "confirmatory CI sign count")
    require(payload["descriptive_sign_counts"]["exploratory"]["point_sign_counts"] == explore_points, "payload exploratory signs")
    require(payload["descriptive_sign_counts"]["confirmatory_recompute"]["point_sign_counts"] == confirm_points, "payload confirmatory signs")

    exploratory_groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in exploratory:
        exploratory_groups[(str(row["benchmark"]), str(row["model"]))][str(row["active_arm"])] = float(row["effect"])
    patterns = Counter()
    steps = Counter()
    for group in exploratory_groups.values():
        require(set(group) == set(PROBES), "incomplete exploratory ladder")
        values = [group[probe] for probe in PROBES]
        labels = []
        for left, right in zip(values, values[1:]):
            label = "worsening" if right < left - 1e-12 else "improving" if right > left + 1e-12 else "tie"
            labels.append(label)
            steps[label] += 1
        if all(label == "tie" for label in labels):
            pattern = "flat"
        elif all(label in {"worsening", "tie"} for label in labels):
            pattern = "monotone_worsening"
        elif all(label in {"improving", "tie"} for label in labels):
            pattern = "monotone_improving"
        else:
            pattern = "mixed_direction"
        patterns[pattern] += 1
    require(patterns == Counter({"mixed_direction": 6, "monotone_worsening": 1, "monotone_improving": 1, "flat": 1}), "monotonicity patterns")
    require(steps == Counter({"improving": 10, "worsening": 9, "tie": 8}), "adjacent step counts")
    require(payload["exploratory_monotonicity"]["pattern_counts"] == {"flat": 1, "mixed_direction": 6, "monotone_improving": 1, "monotone_worsening": 1}, "payload monotonicity")
    require(payload["exploratory_monotonicity"]["adjacent_step_counts"] == {"improving": 10, "tie": 8, "worsening": 9}, "payload step counts")

    with STEM.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 43, "CSV row count")
    for csv_row, json_row in zip(csv_rows, rows):
        require(set(csv_row) == set(json_row), "CSV columns")
        require(all(csv_row[key] == str(json_row[key]) for key in csv_row), "CSV value mismatch")

    sidecar = read_json(STEM.with_suffix(".svg.data.json"))
    require(sidecar["source_json_sha256"] == sha_file(payload_path), "figure source hash")
    require(sidecar["width"] == 1480 and sidecar["height"] == 1305, "figure dimensions")
    require(len(sidecar["exploratory_panel"]["rows"]) == 36, "figure exploratory cells")
    require(len(sidecar["confirmatory_panel"]["rows"]) == 7, "figure confirmatory rows")
    require(sidecar["exploratory_panel"]["n_tasks_per_cell"] == 20, "figure exploratory n")
    require(sidecar["confirmatory_panel"]["n_tasks_per_cell"] == 56, "figure confirmatory n")
    require(sidecar["exploratory_panel"]["rows"] == exploratory, "figure exploratory data")
    require(sidecar["confirmatory_panel"]["rows"] == confirmatory, "figure confirmatory data")
    svg_path = STEM.with_suffix(".svg")
    svg_markup = svg_path.read_text(encoding="utf-8")
    svg_root = ET.parse(svg_path).getroot()
    font_sizes = [float(value) for value in re.findall(r"font-size:\s*([0-9.]+)px", svg_markup)]
    require(font_sizes, "SVG font sizes missing")
    minimum_print_text_pt = min(font_sizes) * 7.0 * 72.0 / float(sidecar["width"])
    require(minimum_print_text_pt >= 7.0, "SVG text below 7 pt at 7-inch print width")
    require("font-family: 'Liberation Sans'" in svg_markup, "portable SVG font missing")
    svg_text = " ".join(text.strip() for text in svg_root.itertext() if text.strip())
    for phrase in (
        "Observer effect of carried active probes",
        "Exploratory mechanism screen — n = 20 paired tasks per cell",
        "Powered, prespecified recompute contrast — n = 56 paired tasks per cell",
        "not a validated dose",
        "Descriptive pattern, not a pooled effect",
        "cross-model trend for recompute",
        "not a universal rule",
        "Tasks repeat across models",
    ):
        require(phrase in svg_text, f"SVG phrase missing: {phrase}")

    print(
        json.dumps(
            {
                "status": "pass",
                "provider_calls_made": 0,
                "code_tree_sha256": EXPECTED_CODE_HASH,
                "payload_sha256": sha_file(payload_path),
                "receipt_sha256": sha_file(receipt_path),
                "raw_effect_rows_recomputed": reconstructed,
                "exploratory_effect_rows": len(exploratory),
                "confirmatory_effect_rows": len(confirmatory),
                "exploratory_point_signs": explore_points,
                "confirmatory_point_signs": confirm_points,
                "monotonicity_patterns": dict(patterns),
                "adjacent_step_counts": dict(steps),
                "figure_rows_verified": 43,
                "minimum_print_text_pt_at_7in": round(minimum_print_text_pt, 3),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
