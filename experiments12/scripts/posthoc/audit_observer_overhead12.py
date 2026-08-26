"""Independent, provider-free audit of the generated observer-overhead artifact.

This intentionally does not import the overhead builder or its extraction helpers.
It reconstructs every task/method row from frozen trajectories, clean shadows,
append-only call attempts, and the read-only reconciled budget ledger; then it
checks summaries, CSVs, provenance inventories, and the SVG data sidecar.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from statistics import median
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments12"
ARTIFACTS = PACKAGE / "data_results" / "runs"
STEM = PACKAGE / "data_results" / "derived" / "observer-overhead-confirmatory-v1"
EXPECTED_CODE_HASH = "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
RUNS = (
    (
        "e12-confirmatory-evolving-core-v2",
        "evolving_intent_gsm8k",
        4,
        "b0be97ed6b01c9c3b003a5900b8e596772fffe3c68c4b9cdd47f46ff329c7056",
        "ccb98c678dc0d9ff9caee539ccd9859aa406abffc16b4bb9eaaf0abfd0bb6a6c",
    ),
    (
        "e12-confirmatory-bfcl-core-v3",
        "bfcl_multi_turn",
        3,
        "551a4574a1fe502c9304feff46c956ffb6b19a96af84d54d3e18d3a60fd919c3",
        "3f2802f6f7471a65f758b0a8c60fc60a5a0334e906732c1f1560c1b31e990be4",
    ),
)
METHODS = (
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_probe:current_copy",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
    "context_use",
)
PURPOSE = {
    "active_recompute": "active_probe",
    "frozen_probe:recompute": "frozen_probe",
    "frozen_probe:current_copy": "frozen_probe",
    "frozen_quiz": "frozen_quiz",
    "trace_judge": "trace_judge",
}
OBSERVER_PURPOSES = set(PURPOSE.values())
FIELDS = (
    "checkpoint_count",
    "provider_call_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "latency_ms",
    "cost_micro_usd",
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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def inventory_digest(paths: list[Path], relative_to: Path) -> str:
    return sha_json(
        [
            {"path": str(path.relative_to(relative_to)), "sha256": sha_file(path)}
            for path in sorted(paths)
        ]
    )


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


def decimal_text(value: Decimal | int | float) -> str:
    text = format(Decimal(str(value)), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def micro_usd_text(value: int | str) -> str:
    text = f"{Decimal(str(value)) / Decimal(1_000_000):.7f}"
    return text[:-1] if text.endswith("0") else text


def method_metadata(method: str) -> tuple[str, bool, str]:
    if method == "active_recompute":
        return "active_carry", True, "carried active trajectory"
    provider_backed = method in PURPOSE
    source = "clean shadow" if provider_backed else "deterministic clean-shadow computation"
    return "passive_zero_carry", provider_backed, source


def main() -> None:
    artifact_path = STEM.with_suffix(".json")
    payload = load_json(artifact_path)
    require(code_tree_hash() == EXPECTED_CODE_HASH, "frozen source/config hash changed")
    require(payload.get("code_tree_sha256") == EXPECTED_CODE_HASH, "payload code hash changed")

    receipt_path = STEM.with_suffix(".receipt.json")
    receipt = load_json(receipt_path)
    require(receipt.get("provider_calls_made") == 0, "builder receipt reports provider calls")
    require(receipt.get("builder_sha256") == sha_file(PACKAGE / str(receipt["builder"]).split("experiments12/", 1)[1]), "builder hash mismatch")
    for raw_path, expected_hash in dict(receipt["outputs"]).items():
        require(sha_file(ROOT / raw_path) == expected_hash, f"receipt output hash mismatch: {raw_path}")

    connection = sqlite3.connect(
        f"file:{(ARTIFACTS / '_global_budget.sqlite3').resolve()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        ledger = {
            str(row["reservation_id"]): dict(row)
            for row in connection.execute("SELECT * FROM reservations")
        }
    finally:
        connection.close()

    expected_rows = {
        (
            row["run_id"],
            row["model"],
            row["task_id"],
            row["condition"],
            row["replicate_id"],
            row["method"],
        ): row
        for row in payload["task_method_rows"]
    }
    require(len(expected_rows) == 3_136, "task/method artifact keys are not unique")
    computed: list[dict[str, object]] = []
    source_inventory = {row["run_id"]: row for row in payload["source_runs"]}
    run_reports = []

    for run_id, benchmark, n_models, manifest_hash, pairs_hash in RUNS:
        run_root = ARTIFACTS / run_id
        require(sha_file(run_root / "manifest.json") == manifest_hash, f"{run_id} manifest hash")
        require(sha_file(run_root / "pairs.jsonl") == pairs_hash, f"{run_id} pairs hash")
        manifest = load_json(run_root / "manifest.json")
        require(manifest["stage"] == "confirmatory", f"{run_id} is not confirmatory")
        require(manifest["arms"] == ["clean", "active_recompute"], f"{run_id} arms changed")
        require(manifest["operators"] == ["none"], f"{run_id} operators changed")
        require(manifest["repository"]["code_tree_sha256"] == EXPECTED_CODE_HASH, f"{run_id} code lock")

        pairs = load_jsonl(run_root / "pairs.jsonl")
        require(len(pairs) == n_models * 56 * 2, f"{run_id} cell count")
        paired: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
        for cell in pairs:
            pair_key = cell["pair_key"]
            key = (
                pair_key["domain"],
                pair_key["model"],
                pair_key["task_id"],
                pair_key["replicate_id"],
                cell["operator"],
            )
            require(cell["arm"] not in paired[key], f"{run_id} duplicate pair arm")
            paired[key][cell["arm"]] = cell
        require(len(paired) == n_models * 56, f"{run_id} model-task pair count")
        require(all(set(arms) == {"clean", "active_recompute"} for arms in paired.values()), f"{run_id} incomplete pairing")

        attempts: dict[str, dict[str, object]] = {}
        for event_path in sorted((run_root / "events").rglob("*.jsonl")):
            for value in load_jsonl(event_path):
                if not {"event_id", "reservation_id"}.issubset(value):
                    continue
                event_id = str(value["event_id"])
                require(event_id not in attempts, f"{run_id} duplicate attempt: {event_id}")
                attempts[event_id] = value
        require(bool(attempts), f"{run_id} contains no attempts")
        for event_id, attempt in attempts.items():
            row = ledger.get(str(attempt["reservation_id"]))
            require(row is not None, f"{run_id} attempt lacks ledger row: {event_id}")
            require(
                (row["state"], row["provider"], row["purpose"], row["request_status"])
                == ("reconciled", attempt["provider"], attempt["purpose"], attempt["status"]),
                f"{run_id} attempt/ledger identity mismatch: {event_id}",
            )
            usage = attempt["usage"]
            require(
                (
                    row["input_tokens"],
                    row["output_tokens"],
                    row["cached_input_tokens"],
                    row["reasoning_tokens"],
                )
                == (
                    usage["input_tokens"],
                    usage["output_tokens"],
                    usage["cached_input_tokens"],
                    usage["reasoning_tokens"],
                ),
                f"{run_id} attempt/ledger usage mismatch: {event_id}",
            )
            expected_micro = int(
                (Decimal(str(attempt["estimated_cost_usd"])) * Decimal(1_000_000)).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            require(row["actual_micro_usd"] == expected_micro, f"{run_id} ledger cost mismatch: {event_id}")
            require(row["cost_quality"] in {"reported", "estimated", "upper_bound"}, f"{run_id} cost quality")

        used_event_ids: set[str] = set()
        zero_opportunity_pairs = 0
        checkpoint_mismatch_pairs = 0
        for pair_key, arms in sorted(paired.items()):
            domain, model, canonical_task_id, replicate_id, operator = pair_key
            require(domain == benchmark and operator == "none", f"{run_id} design cell changed")
            require(str(canonical_task_id).count("::") == 1, f"{run_id} task ID is noncanonical")
            task_id, condition = str(canonical_task_id).split("::")
            clean_cell = arms["clean"]
            active_cell = arms["active_recompute"]
            clean = load_json(run_root / "trajectories" / f"{clean_cell['cell_id']}.json")
            active = load_json(run_root / "trajectories" / f"{active_cell['cell_id']}.json")
            shadow = load_json(run_root / "shadow" / f"{clean_cell['cell_id']}.json")
            require(clean["complete"] is True and active["complete"] is True and shadow["complete"] is True, f"{run_id} incomplete source")
            require(clean["arm"] == "clean" and active["arm"] == "active_recompute", f"{run_id} arm identity")
            require(shadow["source_trajectory_sha256"] == clean["transcript_sha256"], f"{run_id} shadow source")

            records: dict[str, list[dict[str, object]]] = {method: [] for method in METHODS}
            records["active_recompute"] = list(active["probe_records"])
            for record in shadow["records"]:
                method = str(record["method"])
                if method == "frozen_probe":
                    method = f"{method}:{record['variant']}"
                require(method in records and method != "active_recompute", f"{run_id} unknown passive method")
                records[method].append(record)
            opportunity_counts = {len(records[method]) for method in METHODS}
            if len(opportunity_counts) != 1:
                checkpoint_mismatch_pairs += 1
            if not records["active_recompute"]:
                zero_opportunity_pairs += 1

            for method in METHODS:
                totals = {
                    "provider_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "latency_ms": 0,
                    "cost_micro_usd": 0,
                }
                providers: set[str] = set()
                observer_models: set[str] = set()
                cost_qualities: set[str] = set()
                for record_index, record in enumerate(records[method]):
                    if method not in PURPOSE:
                        require("call" not in record, f"{run_id}/{method} deterministic call receipt")
                        continue
                    call = record.get("call")
                    require(isinstance(call, dict), f"{run_id}/{method} missing call receipt")
                    event_ids = call["call_event_ids"]
                    require(isinstance(event_ids, list) and event_ids and len(event_ids) == len(set(event_ids)), f"{run_id}/{method} invalid event IDs")
                    local = {key: 0 for key in totals}
                    local["provider_call_count"] = len(event_ids)
                    for event_id in event_ids:
                        require(event_id not in used_event_ids, f"{run_id} observer event reused: {event_id}")
                        used_event_ids.add(event_id)
                        attempt = attempts[event_id]
                        ledger_row = ledger[str(attempt["reservation_id"])]
                        require(attempt["status"] == "succeeded", f"{run_id} observer attempt failed")
                        require(attempt["purpose"] == PURPOSE[method], f"{run_id}/{method} wrong purpose")
                        require(attempt["attempt_number"] == 1, f"{run_id}/{method} observer retry")
                        usage = attempt["usage"]
                        for token_field in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
                            local[token_field] += int(usage[token_field])
                        local["latency_ms"] += int(attempt["elapsed_ms"])
                        local["cost_micro_usd"] += int(ledger_row["actual_micro_usd"])
                        providers.add(str(attempt["provider"]))
                        observer_models.add(str(attempt["model"]))
                        cost_qualities.add(str(ledger_row["cost_quality"]))
                    for token_field in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
                        require(call["usage"][token_field] == local[token_field], f"{run_id}/{method} embedded usage mismatch")
                    require(call["elapsed_ms"] == local["latency_ms"], f"{run_id}/{method} embedded latency mismatch")
                    require(Decimal(str(call["accounted_cost_usd"])) * Decimal(1_000_000) == local["cost_micro_usd"], f"{run_id}/{method} embedded cost mismatch")
                    for field in totals:
                        totals[field] += local[field]
                require(all(len(values) <= 1 for values in (providers, observer_models, cost_qualities)), f"{run_id}/{method} mixes backend accounting")
                observation_class, provider_backed, receipt_source = method_metadata(method)
                row = {
                    "run_id": run_id,
                    "benchmark": benchmark,
                    "model": model,
                    "task_id": task_id,
                    "condition": condition,
                    "replicate_id": replicate_id,
                    "method": method,
                    "observation_class": observation_class,
                    "provider_backed": provider_backed,
                    "receipt_source": receipt_source,
                    "provider": next(iter(providers), None),
                    "observer_model": next(iter(observer_models), None),
                    "cost_quality": next(iter(cost_qualities), None),
                    "checkpoint_count": len(records[method]),
                    **totals,
                }
                row["total_tokens"] = int(row["input_tokens"]) + int(row["output_tokens"])
                row["cost_usd"] = f"{Decimal(int(row['cost_micro_usd'])) / Decimal(1_000_000):.6f}"
                key = (run_id, model, task_id, condition, replicate_id, method)
                require(row == expected_rows.get(key), f"raw reconstruction disagrees with artifact: {key}")
                computed.append(row)

        observer_events = {event_id for event_id, attempt in attempts.items() if attempt["purpose"] in OBSERVER_PURPOSES}
        require(used_event_ids == observer_events, f"{run_id} observer event coverage mismatch")
        inventory = source_inventory[run_id]
        trajectories = list((run_root / "trajectories").glob("*.json"))
        shadows = list((run_root / "shadow").glob("*.json"))
        require(inventory["trajectory_count"] == len(trajectories), f"{run_id} trajectory count")
        require(inventory["trajectory_inventory_sha256"] == inventory_digest(trajectories, run_root), f"{run_id} trajectory inventory")
        require(inventory["shadow_count"] == len(shadows), f"{run_id} shadow count")
        require(inventory["shadow_inventory_sha256"] == inventory_digest(shadows, run_root), f"{run_id} shadow inventory")
        require(inventory["observer_call_event_count"] == len(observer_events), f"{run_id} observer event count")
        require(inventory["observer_call_event_ids_sha256"] == sha_json(sorted(observer_events)), f"{run_id} observer event digest")
        run_reports.append(
            {
                "run_id": run_id,
                "model_task_pairs": len(paired),
                "observer_events": len(observer_events),
                "zero_opportunity_model_tasks": zero_opportunity_pairs,
                "active_passive_checkpoint_mismatches": checkpoint_mismatch_pairs,
                "observer_purposes": dict(sorted(Counter(attempts[event]["purpose"] for event in observer_events).items())),
                "cost_qualities": dict(sorted(Counter(ledger[str(attempts[event]["reservation_id"])]["cost_quality"] for event in observer_events).items())),
            }
        )

    require(len(computed) == 3_136, "independent task/method row count")
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in computed:
        groups[(str(row["benchmark"]), str(row["model"]), str(row["method"]))].append(row)
    expected_summaries = {
        (row["benchmark"], row["model"], row["method"]): row for row in payload["summaries"]
    }
    require(len(groups) == len(expected_summaries) == 56, "summary stratum count")
    for key, group in groups.items():
        summary = {
            "benchmark": key[0],
            "model": key[1],
            "method": key[2],
            "observation_class": group[0]["observation_class"],
            "provider_backed": group[0]["provider_backed"],
            "n_tasks": len(group),
            "tasks_with_provider_calls": sum(int(row["provider_call_count"]) > 0 for row in group),
            "providers": ";".join(sorted({str(row["provider"]) for row in group if row["provider"] is not None})),
            "observer_models": ";".join(sorted({str(row["observer_model"]) for row in group if row["observer_model"] is not None})),
            "cost_qualities": ";".join(sorted({str(row["cost_quality"]) for row in group if row["cost_quality"] is not None})),
        }
        for field in FIELDS:
            values = [int(row[field]) for row in group]
            summary[f"sum_{field}"] = sum(values)
            summary[f"median_{field}"] = decimal_text(median(values))
        summary["sum_cost_usd"] = f"{Decimal(int(summary['sum_cost_micro_usd'])) / Decimal(1_000_000):.6f}"
        summary["median_cost_usd"] = micro_usd_text(summary["median_cost_micro_usd"])
        require(summary == expected_summaries.get(key), f"independent summary mismatch: {key}")

    with STEM.with_suffix(".tasks.csv").open(newline="", encoding="utf-8") as handle:
        task_csv = list(csv.DictReader(handle))
    with STEM.with_suffix(".summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_csv = list(csv.DictReader(handle))
    require(len(task_csv) == 3_136 and len(summary_csv) == 56, "CSV row counts")
    for csv_row, json_row in zip(task_csv, payload["task_method_rows"]):
        require(set(csv_row) == set(json_row), "task CSV columns")
        require(all(csv_row[key] == ("" if json_row[key] is None else str(json_row[key])) for key in csv_row), "task CSV value mismatch")
    for csv_row, json_row in zip(summary_csv, payload["summaries"]):
        require(set(csv_row) == set(json_row), "summary CSV columns")
        require(all(csv_row[key] == ("" if json_row[key] is None else str(json_row[key])) for key in csv_row), "summary CSV value mismatch")

    sidecar_path = STEM.with_suffix(".svg.data.json")
    sidecar = load_json(sidecar_path)
    require(sidecar["source_json_sha256"] == sha_file(artifact_path), "figure source hash")
    require(sidecar["width"] == 1480 and sidecar["height"] == 1572, "figure dimensions")
    figure_rows = sidecar["rows"]
    require(len(figure_rows) == 7 * 8 * 4, "figure plotted-cell count")
    figure_lookup = {(row["benchmark"], row["model"], row["method"], row["metric"]): row for row in figure_rows}
    require(len(figure_lookup) == len(figure_rows), "figure cells are not unique")
    for key, summary in expected_summaries.items():
        for metric in ("median_total_tokens", "median_latency_ms", "median_cost_micro_usd", "median_provider_call_count"):
            plotted = figure_lookup[(key[0], key[1], key[2], metric)]
            require(plotted["median"] == str(Decimal(str(summary[metric]))), f"figure median mismatch: {key}/{metric}")
            require(plotted["n_tasks"] == 56, f"figure sample size mismatch: {key}/{metric}")
    for metric, maximum in sidecar["metric_maxima"].items():
        require(Decimal(str(maximum)) == max(Decimal(str(row[metric])) for row in payload["summaries"]), f"figure scale maximum: {metric}")
    svg_path = STEM.with_suffix(".svg")
    svg_root = ET.parse(svg_path).getroot()
    require(svg_root.attrib.get("width") == str(sidecar["width"]), "SVG width")
    require(svg_root.attrib.get("height") == str(sidecar["height"]), "SVG height")
    svg_markup = svg_path.read_text(encoding="utf-8")
    font_sizes = [float(value) for value in re.findall(r"font-size:\s*([0-9.]+)px", svg_markup)]
    require(font_sizes, "SVG font sizes missing")
    minimum_print_text_pt = min(font_sizes) * 7.0 * 72.0 / float(sidecar["width"])
    require(minimum_print_text_pt >= 7.0, "SVG text below 7 pt at 7-inch print width")
    require("font-family: 'Liberation Sans'" in svg_markup, "portable SVG font missing")
    svg_text = " ".join(text.strip() for text in svg_root.itertext() if text.strip())
    for phrase in (
        "Observer overhead per deployed task",
        "Median across 56 held-out tasks per benchmark/model",
        "agent-task calls excluded",
        "active / carried",
        "passive / zero-carry, provider-backed",
        "passive deterministic (zero provider calls)",
        "Provider input + output tokens",
        "Provider-call latency",
        "on-path delay for active probes",
        "off-path time for provider-backed passive monitors",
        "Deterministic local runtime was not measured",
        "ledger estimates, not invoices",
    ):
        require(phrase in svg_text, f"SVG explanatory phrase missing: {phrase}")

    unique_source_tasks = {
        (row["benchmark"], row["task_id"], row["condition"], row["replicate_id"])
        for row in computed
    }
    report = {
        "status": "pass",
        "provider_calls_made": 0,
        "code_tree_sha256": EXPECTED_CODE_HASH,
        "artifact_sha256": sha_file(artifact_path),
        "receipt_sha256": sha_file(receipt_path),
        "task_method_rows_recomputed": len(computed),
        "summary_strata_recomputed": len(groups),
        "figure_cells_verified": len(figure_rows),
        "minimum_print_text_pt_at_7in": round(minimum_print_text_pt, 3),
        "model_tasks": len(computed) // len(METHODS),
        "unique_benchmark_task_conditions": len(unique_source_tasks),
        "runs": run_reports,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
