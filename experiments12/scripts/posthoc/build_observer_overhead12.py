"""Build provider-free observer-overhead artifacts from frozen confirmatory runs.

This script prices only observation itself.  It excludes the agent's task calls:

* ``active_recompute`` uses probe calls carried inside active trajectories;
* provider-backed passive methods use calls made over immutable clean shadows;
* deterministic passive methods have zero provider calls, tokens, latency, and cost.

Every embedded call receipt is joined back to the append-only attempt log and the
reconciled global ledger before it contributes to an output row.  No provider API
is imported or called.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import csv
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments12.analysis12 import _load_attempt_resources
from experiments12.core.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.manifest12 import RunLayout, code_tree_hash
from experiments12.pairing12 import JobCell
from experiments12.validate12 import validate_run


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments12"
ARTIFACTS = PACKAGE / "data_results" / "runs"
OUTPUT_STEM = PACKAGE / "data_results" / "derived" / "observer-overhead-confirmatory-v1"
EXPECTED_CODE_TREE_SHA256 = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)

RUNS = (
    {
        "run_id": "e12-confirmatory-evolving-core-v2",
        "manifest_sha256": (
            "b0be97ed6b01c9c3b003a5900b8e596772fffe3c68c4b9cdd47f46ff329c7056"
        ),
        "pairs_sha256": (
            "ccb98c678dc0d9ff9caee539ccd9859aa406abffc16b4bb9eaaf0abfd0bb6a6c"
        ),
        "benchmark": "evolving_intent_gsm8k",
        "n_models": 4,
        "n_tasks_per_model": 56,
    },
    {
        "run_id": "e12-confirmatory-bfcl-core-v3",
        "manifest_sha256": (
            "551a4574a1fe502c9304feff46c956ffb6b19a96af84d54d3e18d3a60fd919c3"
        ),
        "pairs_sha256": (
            "3f2802f6f7471a65f758b0a8c60fc60a5a0334e906732c1f1560c1b31e990be4"
        ),
        "benchmark": "bfcl_multi_turn",
        "n_models": 3,
        "n_tasks_per_model": 56,
    },
)

METHODS = (
    {
        "method": "active_recompute",
        "label": "Active\nrecompute",
        "observation_class": "active_carry",
        "provider_backed": True,
        "receipt_source": "carried active trajectory",
        "purpose": "active_probe",
    },
    {
        "method": "frozen_probe:recompute",
        "label": "Passive probe\nrecompute",
        "observation_class": "passive_zero_carry",
        "provider_backed": True,
        "receipt_source": "clean shadow",
        "purpose": "frozen_probe",
    },
    {
        "method": "frozen_probe:current_copy",
        "label": "Passive probe\ncurrent-copy",
        "observation_class": "passive_zero_carry",
        "provider_backed": True,
        "receipt_source": "clean shadow",
        "purpose": "frozen_probe",
    },
    {
        "method": "frozen_quiz",
        "label": "Passive\nquiz",
        "observation_class": "passive_zero_carry",
        "provider_backed": True,
        "receipt_source": "clean shadow",
        "purpose": "frozen_quiz",
    },
    {
        "method": "trace_judge",
        "label": "Passive trace\njudge",
        "observation_class": "passive_zero_carry",
        "provider_backed": True,
        "receipt_source": "clean shadow",
        "purpose": "trace_judge",
    },
    {
        "method": "trace_rules",
        "label": "Trace\nrules",
        "observation_class": "passive_zero_carry",
        "provider_backed": False,
        "receipt_source": "deterministic clean-shadow computation",
        "purpose": None,
    },
    {
        "method": "turn_clock",
        "label": "Turn\nclock",
        "observation_class": "passive_zero_carry",
        "provider_backed": False,
        "receipt_source": "deterministic clean-shadow computation",
        "purpose": None,
    },
    {
        "method": "context_use",
        "label": "Context\nuse",
        "observation_class": "passive_zero_carry",
        "provider_backed": False,
        "receipt_source": "deterministic clean-shadow computation",
        "purpose": None,
    },
)
METHOD_ORDER = {row["method"]: index for index, row in enumerate(METHODS)}


class OverheadInputError(ValueError):
    """A frozen input or accounting join is incomplete or inconsistent."""


def _split_task_id(task_id: str) -> tuple[str, str]:
    if not isinstance(task_id, str) or task_id.count("::") != 1:
        raise OverheadInputError(f"non-canonical task ID: {task_id!r}")
    source, condition = task_id.split("::", 1)
    if not source or not condition:
        raise OverheadInputError(f"non-canonical task ID: {task_id!r}")
    return source, condition


def _pair_key(cell: JobCell) -> tuple[str, str, str, int, str]:
    return (
        cell.pair_key.domain,
        cell.pair_key.model,
        cell.pair_key.task_id,
        cell.pair_key.replicate_id,
        cell.operator,
    )


def _inventory_digest(paths: Iterable[Path], *, relative_to: Path) -> str:
    records = [
        {
            "path": str(path.relative_to(relative_to)),
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    return sha256_json(records)


def _receipt_totals(
    records: Sequence[Mapping[str, Any]],
    *,
    attempts: Mapping[str, Any],
    expected_purpose: str,
    context: str,
    used_event_ids: set[str],
) -> dict[str, Any]:
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
    for index, record in enumerate(records):
        call = record.get("call")
        if not isinstance(call, Mapping):
            raise OverheadInputError(f"{context} record {index} lacks call receipt")
        event_ids = call.get("call_event_ids")
        if (
            not isinstance(event_ids, list)
            or not event_ids
            or any(not isinstance(value, str) or not value for value in event_ids)
            or len(event_ids) != len(set(event_ids))
        ):
            raise OverheadInputError(f"{context} record {index} has invalid event IDs")
        duplicate = used_event_ids.intersection(event_ids)
        if duplicate:
            raise OverheadInputError(
                f"observer call event reused: {sorted(duplicate)[0]}"
            )
        used_event_ids.update(event_ids)

        receipt = {
            "provider_call_count": len(event_ids),
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "latency_ms": 0,
            "cost_micro_usd": 0,
        }
        for event_id in event_ids:
            attempt = attempts.get(event_id)
            if attempt is None:
                raise OverheadInputError(f"{context} call lacks attempt: {event_id}")
            if attempt.status.value != "succeeded" or attempt.purpose != expected_purpose:
                raise OverheadInputError(
                    f"{context} call has wrong status/purpose: {event_id}"
                )
            receipt["input_tokens"] += attempt.input_tokens
            receipt["output_tokens"] += attempt.output_tokens
            receipt["cached_input_tokens"] += attempt.cached_input_tokens
            receipt["reasoning_tokens"] += attempt.reasoning_tokens
            receipt["latency_ms"] += attempt.elapsed_ms
            receipt["cost_micro_usd"] += int(attempt.actual_cost_usd * 1_000_000)
            if attempt.provider is None or attempt.model is None:
                raise OverheadInputError(f"{context} call lacks provider/model: {event_id}")
            providers.add(attempt.provider)
            observer_models.add(attempt.model)
            cost_qualities.add(attempt.cost_quality)

        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            raise OverheadInputError(f"{context} record {index} lacks usage")
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            if usage.get(field) != receipt[field]:
                raise OverheadInputError(
                    f"{context} record {index} {field} disagrees with ledger attempts"
                )
        if call.get("elapsed_ms") != receipt["latency_ms"]:
            raise OverheadInputError(
                f"{context} record {index} latency disagrees with ledger attempts"
            )
        try:
            embedded_cost = Decimal(str(call.get("accounted_cost_usd")))
        except Exception as exc:  # pragma: no cover - defensive fail closed
            raise OverheadInputError(
                f"{context} record {index} has invalid cost"
            ) from exc
        if embedded_cost != Decimal(receipt["cost_micro_usd"]) / Decimal(1_000_000):
            raise OverheadInputError(
                f"{context} record {index} cost disagrees with ledger attempts"
            )
        for field, value in receipt.items():
            totals[field] += value
    for label, values in (
        ("provider", providers),
        ("observer_model", observer_models),
        ("cost_quality", cost_qualities),
    ):
        if len(values) > 1:
            raise OverheadInputError(f"{context} mixes {label} values: {sorted(values)}")
        totals[label] = next(iter(values)) if values else None
    return totals


def _passive_records(shadow: Mapping[str, Any], *, context: str) -> dict[str, list[Any]]:
    records = shadow.get("records")
    if not isinstance(records, list):
        raise OverheadInputError(f"{context} shadow records are invalid")
    result: dict[str, list[Any]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise OverheadInputError(f"{context} shadow record {index} is invalid")
        method = record.get("method")
        if method == "frozen_probe":
            variant = record.get("variant")
            if variant not in {"recompute", "current_copy"}:
                raise OverheadInputError(
                    f"{context} has unexpected frozen-probe variant: {variant!r}"
                )
            method = f"frozen_probe:{variant}"
        if method not in METHOD_ORDER or method == "active_recompute":
            raise OverheadInputError(f"{context} has unexpected passive method: {method!r}")
        result[str(method)].append(record)
    expected = set(METHOD_ORDER) - {"active_recompute"}
    if set(result) != expected and records:
        raise OverheadInputError(
            f"{context} passive method set is incomplete: {sorted(set(result))}"
        )
    if not records:
        result = {method: [] for method in expected}
    lengths = {len(result[method]) for method in expected}
    if len(lengths) != 1:
        raise OverheadInputError(f"{context} passive checkpoint counts differ")
    return result


def _task_row(
    *,
    run_id: str,
    benchmark: str,
    model: str,
    source_task_id: str,
    condition: str,
    replicate_id: int,
    method: Mapping[str, Any],
    checkpoint_count: int,
    totals: Mapping[str, Any],
) -> dict[str, Any]:
    cost_micro = totals["cost_micro_usd"]
    row = {
        "run_id": run_id,
        "benchmark": benchmark,
        "model": model,
        "task_id": source_task_id,
        "condition": condition,
        "replicate_id": replicate_id,
        "method": method["method"],
        "observation_class": method["observation_class"],
        "provider_backed": method["provider_backed"],
        "receipt_source": method["receipt_source"],
        "provider": totals.get("provider"),
        "observer_model": totals.get("observer_model"),
        "cost_quality": totals.get("cost_quality"),
        "checkpoint_count": checkpoint_count,
        "provider_call_count": totals["provider_call_count"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["input_tokens"] + totals["output_tokens"],
        "cached_input_tokens": totals["cached_input_tokens"],
        "reasoning_tokens": totals["reasoning_tokens"],
        "latency_ms": totals["latency_ms"],
        "cost_micro_usd": cost_micro,
        "cost_usd": f"{Decimal(cost_micro) / Decimal(1_000_000):.6f}",
    }
    if not method["provider_backed"] and any(
        row[field]
        for field in (
            "provider_call_count",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "latency_ms",
            "cost_micro_usd",
        )
    ):
        raise OverheadInputError(f"deterministic method has provider overhead: {row}")
    return row


def _extract_run(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_id = str(spec["run_id"])
    layout = RunLayout.for_run(ARTIFACTS, run_id)
    if sha256_file(layout.manifest) != spec["manifest_sha256"]:
        raise OverheadInputError(f"{run_id} manifest hash changed")
    if sha256_file(layout.pairs) != spec["pairs_sha256"]:
        raise OverheadInputError(f"{run_id} pair hash changed")
    report = validate_run(
        layout,
        repository_root=ROOT,
        expected_manifest_sha256=str(spec["manifest_sha256"]),
    )
    if not report.primary_ready or report.errors or report.warnings:
        raise OverheadInputError(f"{run_id} does not pass strict validation")

    manifest = read_json(layout.manifest)
    if (
        manifest.get("stage") != "confirmatory"
        or manifest.get("arms") != ["clean", "active_recompute"]
        or manifest.get("operators") != ["none"]
        or manifest.get("repository", {}).get("code_tree_sha256")
        != EXPECTED_CODE_TREE_SHA256
    ):
        raise OverheadInputError(f"{run_id} manifest design is not the frozen design")

    cells = [JobCell.from_dict(value) for value in read_jsonl(layout.pairs)]
    grouped: dict[tuple[str, str, str, int, str], dict[str, JobCell]] = defaultdict(dict)
    for cell in cells:
        key = _pair_key(cell)
        if cell.arm in grouped[key]:
            raise OverheadInputError(f"{run_id} duplicates arm within pair: {key}")
        grouped[key][cell.arm] = cell
    if any(set(arms) != {"clean", "active_recompute"} for arms in grouped.values()):
        raise OverheadInputError(f"{run_id} lacks paired clean/active cells")
    if len(grouped) != spec["n_models"] * spec["n_tasks_per_model"]:
        raise OverheadInputError(f"{run_id} has the wrong number of model-task pairs")

    attempts = _load_attempt_resources(layout)
    used_event_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, arms in sorted(grouped.items()):
        benchmark, model, canonical_task_id, replicate_id, operator = key
        if benchmark != spec["benchmark"] or operator != "none":
            raise OverheadInputError(f"{run_id} contains an unexpected design cell")
        source_task_id, condition = _split_task_id(canonical_task_id)
        clean_cell = arms["clean"]
        active_cell = arms["active_recompute"]
        clean = read_json(layout.trajectories / f"{clean_cell.cell_id}.json")
        active = read_json(layout.trajectories / f"{active_cell.cell_id}.json")
        shadow = read_json(layout.shadow / f"{clean_cell.cell_id}.json")
        for name, value, cell in (
            ("clean", clean, clean_cell),
            ("active", active, active_cell),
        ):
            if (
                value.get("complete") is not True
                or value.get("cell_id") != cell.cell_id
                or value.get("model") != model
                or value.get("task_id") != source_task_id
                or value.get("condition") != condition
            ):
                raise OverheadInputError(f"{run_id} {name} trajectory identity changed")
        if (
            shadow.get("complete") is not True
            or shadow.get("model") != model
            or shadow.get("task_id") != source_task_id
            or shadow.get("condition") != condition
            or shadow.get("source_trajectory_sha256") != clean.get("transcript_sha256")
        ):
            raise OverheadInputError(f"{run_id} clean shadow identity changed")

        active_records = active.get("probe_records")
        if not isinstance(active_records, list) or any(
            not isinstance(record, Mapping)
            or record.get("event") != "active_probe"
            or record.get("variant") != "recompute"
            for record in active_records
        ):
            raise OverheadInputError(f"{run_id} active records are invalid")
        active_totals = _receipt_totals(
            active_records,
            attempts=attempts,
            expected_purpose="active_probe",
            context=f"{run_id}/{model}/{source_task_id}/active_recompute",
            used_event_ids=used_event_ids,
        )
        rows.append(
            _task_row(
                run_id=run_id,
                benchmark=benchmark,
                model=model,
                source_task_id=source_task_id,
                condition=condition,
                replicate_id=replicate_id,
                method=METHODS[METHOD_ORDER["active_recompute"]],
                checkpoint_count=len(active_records),
                totals=active_totals,
            )
        )

        passive = _passive_records(
            shadow, context=f"{run_id}/{model}/{source_task_id}"
        )
        for method in METHODS[1:]:
            method_records = passive[method["method"]]
            if method["provider_backed"]:
                totals = _receipt_totals(
                    method_records,
                    attempts=attempts,
                    expected_purpose=str(method["purpose"]),
                    context=f"{run_id}/{model}/{source_task_id}/{method['method']}",
                    used_event_ids=used_event_ids,
                )
            else:
                if any("call" in record for record in method_records):
                    raise OverheadInputError(
                        f"deterministic method has call receipt: {method['method']}"
                    )
                totals = {
                    "provider_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "latency_ms": 0,
                    "cost_micro_usd": 0,
                    "provider": None,
                    "observer_model": None,
                    "cost_quality": None,
                }
            rows.append(
                _task_row(
                    run_id=run_id,
                    benchmark=benchmark,
                    model=model,
                    source_task_id=source_task_id,
                    condition=condition,
                    replicate_id=replicate_id,
                    method=method,
                    checkpoint_count=len(method_records),
                    totals=totals,
                )
            )

    observer_purposes = {"active_probe", "frozen_probe", "frozen_quiz", "trace_judge"}
    expected_event_ids = {
        event_id
        for event_id, attempt in attempts.items()
        if attempt.purpose in observer_purposes
    }
    if used_event_ids != expected_event_ids:
        raise OverheadInputError(
            f"{run_id} observer event coverage mismatch: "
            f"used={len(used_event_ids)} expected={len(expected_event_ids)}"
        )
    if any(attempt.attempt_number != 1 for event_id, attempt in attempts.items() if event_id in used_event_ids):
        raise OverheadInputError(f"{run_id} unexpectedly contains observer retries")

    inventory = {
        "run_id": run_id,
        "benchmark": spec["benchmark"],
        "manifest_sha256": spec["manifest_sha256"],
        "pairs_sha256": spec["pairs_sha256"],
        "trajectory_count": len(list(layout.trajectories.glob("*.json"))),
        "trajectory_inventory_sha256": _inventory_digest(
            layout.trajectories.glob("*.json"), relative_to=layout.root
        ),
        "shadow_count": len(list(layout.shadow.glob("*.json"))),
        "shadow_inventory_sha256": _inventory_digest(
            layout.shadow.glob("*.json"), relative_to=layout.root
        ),
        "observer_call_event_count": len(used_event_ids),
        "observer_call_event_ids_sha256": sha256_json(sorted(used_event_ids)),
        "strict_validation": {"errors": 0, "warnings": 0, "primary_ready": True},
    }
    return rows, inventory


def _median(values: Sequence[int]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise OverheadInputError("cannot summarize an empty group")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return Decimal(ordered[midpoint])
    return Decimal(ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _micro_usd_text(value: int | str) -> str:
    """Render integer- or half-micro-dollar summaries without losing precision."""

    text = f"{Decimal(str(value)) / Decimal(1_000_000):.7f}"
    # Integer micro-dollar values need six decimals; half-micro medians need seven.
    return text[:-1] if text.endswith("0") else text


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["model"], row["method"])].append(row)
    result: list[dict[str, Any]] = []
    fields = (
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
    for (benchmark, model, method), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], METHOD_ORDER[item[0][2]])
    ):
        summary: dict[str, Any] = {
            "benchmark": benchmark,
            "model": model,
            "method": method,
            "observation_class": group[0]["observation_class"],
            "provider_backed": group[0]["provider_backed"],
            "n_tasks": len(group),
            "tasks_with_provider_calls": sum(row["provider_call_count"] > 0 for row in group),
            "providers": ";".join(
                sorted({str(row["provider"]) for row in group if row["provider"] is not None})
            ),
            "observer_models": ";".join(
                sorted(
                    {
                        str(row["observer_model"])
                        for row in group
                        if row["observer_model"] is not None
                    }
                )
            ),
            "cost_qualities": ";".join(
                sorted(
                    {
                        str(row["cost_quality"])
                        for row in group
                        if row["cost_quality"] is not None
                    }
                )
            ),
        }
        for field in fields:
            values = [int(row[field]) for row in group]
            summary[f"sum_{field}"] = sum(values)
            summary[f"median_{field}"] = _decimal_text(_median(values))
        summary["sum_cost_usd"] = (
            f"{Decimal(summary['sum_cost_micro_usd']) / Decimal(1_000_000):.6f}"
        )
        summary["median_cost_usd"] = _micro_usd_text(
            summary["median_cost_micro_usd"]
        )
        result.append(summary)
    return result


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
        "gpt-oss-120b": "GPT-OSS 120B",
        "deepseek-v4-flash-0731": "DeepSeek V4 Flash",
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
        "gpt-oss-120b": "GPT-OSS 120B",
        "deepseek-v4-flash-0731": "DeepSeek V4",
        "gpt-5.6-luna": "GPT-5.6 Luna",
        "gpt-5.6-terra": "GPT-5.6 Terra",
    }.get(model, model)
    return f"{benchmark_label} · {model_label}"


def _figure(
    summaries: Sequence[Mapping[str, Any]],
    *,
    source_json_sha256: str,
) -> tuple[str, dict[str, Any]]:
    metrics = (
        ("median_total_tokens", "Provider input + output tokens", "tokens"),
        ("median_latency_ms", "Provider-call latency", "latency"),
        ("median_cost_micro_usd", "Observer cost (estimated USD)", "cost"),
        ("median_provider_call_count", "Provider calls", "calls"),
    )
    strata = sorted({(row["benchmark"], row["model"]) for row in summaries})
    strata.sort(
        key=lambda item: (
            {"evolving_intent_gsm8k": 0, "bfcl_multi_turn": 1}.get(item[0], 9),
            {
                "gpt-oss-120b": 0,
                "deepseek-v4-flash-0731": 1,
                "gpt-5.6-luna": 2,
                "gpt-5.6-terra": 3,
            }.get(item[1], 9),
        )
    )
    lookup = {
        (row["benchmark"], row["model"], row["method"]): row for row in summaries
    }
    width = 1480
    left = 290
    right = 30
    top = 220
    cell_w = (width - left - right) / len(METHODS)
    row_h = 35
    block_header_h = 46
    block_gap = 34
    block_h = block_header_h + len(strata) * row_h
    height = int(top + len(metrics) * block_h + (len(metrics) - 1) * block_gap + 86)
    body: list[str] = []
    body.append('<rect width="100%" height="100%" fill="#FFFFFF"/>')
    body.append(
        '<text x="30" y="46" class="title">Observer overhead per deployed task</text>'
    )
    body.append(
        '<text x="30" y="78" class="subtitle">Median across 56 held-out tasks per benchmark/model; observation calls only (agent-task calls excluded)</text>'
    )
    body.append(
        '<rect x="30" y="94" width="18" height="18" rx="2" fill="#FFF3E8" stroke="#D55E00" stroke-width="2"/>'
        '<text x="58" y="110" class="legend">active / carried</text>'
        '<rect x="225" y="94" width="18" height="18" rx="2" fill="#EAF4FB" stroke="#0072B2" stroke-width="2"/>'
        '<text x="253" y="110" class="legend">passive / zero-carry, provider-backed</text>'
        '<rect x="690" y="94" width="18" height="18" rx="2" fill="#F4F4F4" stroke="#999999"/>'
        '<text x="718" y="110" class="legend">passive deterministic (zero provider calls)</text>'
    )
    body.append(
        '<text x="30" y="142" class="note">Cell shade is log-scaled within each metric. Exact input and output token counts are separate in the data files.</text>'
    )
    for index, method in enumerate(METHODS):
        x = left + (index + 0.5) * cell_w
        label_lines = method["label"].split("\n")
        tspans = "".join(
            f'<tspan x="{x:.1f}" dy="{0 if line_index == 0 else 22}">{_esc(line)}</tspan>'
            for line_index, line in enumerate(label_lines)
        )
        body.append(
            f'<text x="{x:.1f}" y="170" text-anchor="middle" class="method">{tspans}</text>'
        )

    plotted: list[dict[str, Any]] = []
    metric_scales: dict[str, float] = {}
    for metric_index, (field, title, format_kind) in enumerate(metrics):
        block_y = top + metric_index * (block_h + block_gap)
        maximum = max(Decimal(str(row[field])) for row in summaries)
        scale_max = math.log1p(float(maximum)) if maximum > 0 else 1.0
        metric_scales[field] = float(maximum)
        body.append(
            f'<text x="30" y="{block_y + 29:.1f}" class="metric">{_esc(title)}</text>'
        )
        body.append(
            f'<line x1="{left}" y1="{block_y + 41:.1f}" x2="{width - right}" y2="{block_y + 41:.1f}" class="rule"/>'
        )
        previous_benchmark: str | None = None
        for stratum_index, (benchmark, model) in enumerate(strata):
            y = block_y + block_header_h + stratum_index * row_h
            if previous_benchmark is not None and benchmark != previous_benchmark:
                body.append(
                    f'<line x1="30" y1="{y - 2:.1f}" x2="{width - right}" y2="{y - 2:.1f}" class="benchmark-rule"/>'
                )
            label = _figure_label(benchmark, model)
            body.append(
                f'<text x="{left - 12}" y="{y + 24:.1f}" text-anchor="end" class="row-label">{_esc(label)}</text>'
            )
            for method_index, method in enumerate(METHODS):
                summary = lookup[(benchmark, model, method["method"])]
                value = Decimal(str(summary[field]))
                normalized = 0.0 if value <= 0 else math.log1p(float(value)) / scale_max
                if method["observation_class"] == "active_carry":
                    border = "#D55E00"
                    fill_base = (253, 232, 216)
                elif method["provider_backed"]:
                    border = "#0072B2"
                    fill_base = (218, 237, 248)
                else:
                    border = "#999999"
                    fill_base = (245, 245, 245)
                if value > 0:
                    strength = 0.18 + 0.62 * normalized
                    target = (44, 102, 140)
                    rgb = tuple(
                        round(channel * (1 - strength) + target_channel * strength)
                        for channel, target_channel in zip(fill_base, target)
                    )
                    fill = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
                else:
                    fill = f"rgb({fill_base[0]},{fill_base[1]},{fill_base[2]})"
                x = left + method_index * cell_w + 3
                cell_width = cell_w - 6
                if format_kind == "tokens":
                    display = f"{float(value):,.0f}" if value == value.to_integral() else f"{float(value):,.1f}"
                    tooltip = (
                        f"median input={summary['median_input_tokens']}; "
                        f"output={summary['median_output_tokens']}; total={value} tokens"
                    )
                elif format_kind == "latency":
                    display = f"{float(value) / 1000:.1f} s" if value else "0 s"
                    tooltip = f"median summed observer latency={value} ms"
                elif format_kind == "cost":
                    display = f"${float(value) / 1_000_000:.4f}" if value else "$0"
                    tooltip = f"median observer cost={summary['median_cost_usd']} USD"
                else:
                    display = _decimal_text(value)
                    tooltip = f"median provider calls={value}"
                text_color = "#FFFFFF" if normalized > 0.56 else "#1F2933"
                body.append(
                    f'<rect x="{x:.1f}" y="{y + 3:.1f}" width="{cell_width:.1f}" height="{row_h - 6:.1f}" rx="4" fill="{fill}" stroke="{border}" stroke-width="{2 if method["observation_class"] == "active_carry" or method["provider_backed"] else 1}"><title>{_esc(label)} · {_esc(method["method"])} · {_esc(tooltip)}</title></rect>'
                )
                body.append(
                    f'<text x="{x + cell_width / 2:.1f}" y="{y + 24:.1f}" text-anchor="middle" class="cell" style="fill:{text_color}">{_esc(display)}</text>'
                )
                plotted.append(
                    {
                        "benchmark": benchmark,
                        "model": model,
                        "method": method["method"],
                        "metric": field,
                        "median": str(value),
                        "display": display,
                        "n_tasks": summary["n_tasks"],
                    }
                )
            previous_benchmark = benchmark

    body.append(
        f'<text x="30" y="{height - 50}" class="footnote">Latency sums provider API response time: on-path delay for active probes; off-path time for provider-backed passive monitors.</text>'
    )
    body.append(
        f'<text x="30" y="{height - 24}" class="footnote">Deterministic local runtime was not measured; zeros mean no provider calls. Dollar values are ledger estimates, not invoices.</text>'
    )
    style = """
      text { font-family: 'Liberation Sans'; fill: #1F2933; }
      .title { font-size: 36px; font-weight: 750; letter-spacing: -0.25px; }
      .subtitle { font-size: 21px; fill: #52606D; }
      .legend { font-size: 21px; fill: #3E4C59; }
      .note, .footnote { font-size: 21px; fill: #616E7C; }
      .method { font-size: 21px; font-weight: 650; fill: #334E68; }
      .metric { font-size: 25px; font-weight: 750; fill: #102A43; }
      .row-label { font-size: 21px; fill: #334E68; }
      .cell { font-size: 21px; font-weight: 700; }
      .rule { stroke: #BCCCDC; stroke-width: 1; }
      .benchmark-rule { stroke: #829AB1; stroke-width: 1.2; stroke-dasharray: 3 3; }
    """
    description = (
        "Heatmap of median per-task observer token, latency, cost, and provider-call "
        "overhead for active carried and passive zero-carry observation methods, "
        "stratified by benchmark and agent model."
    )
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        '<title id="title">Observer overhead per deployed task</title>\n'
        f'<desc id="desc">{_esc(description)}</desc>\n'
        f'<defs><style>{style}</style></defs>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    sidecar = {
        "schema_version": 1,
        "figure_type": "observer_overhead_median_heatmap",
        "title": "Observer overhead per deployed task",
        "description": description,
        "source_json_sha256": source_json_sha256,
        "statistic": "median across held-out model-task units",
        "metric_scale": "log1p color intensity within metric; labels are untransformed",
        "metric_maxima": metric_scales,
        "rows": plotted,
        "width": width,
        "height": height,
    }
    return svg, sidecar


def main() -> None:
    live_hash_before = code_tree_hash(PACKAGE)
    if live_hash_before != EXPECTED_CODE_TREE_SHA256:
        raise OverheadInputError(
            f"frozen code tree changed before extraction: {live_hash_before}"
        )

    all_rows: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []
    for spec in RUNS:
        rows, inventory = _extract_run(spec)
        all_rows.extend(rows)
        source_inventory.append(inventory)
    all_rows.sort(
        key=lambda row: (
            row["benchmark"],
            row["model"],
            row["task_id"],
            row["replicate_id"],
            METHOD_ORDER[row["method"]],
        )
    )
    summaries = _summaries(all_rows)
    n_model_tasks = len(all_rows) // len(METHODS)
    if len(all_rows) != 392 * len(METHODS) or n_model_tasks != 392:
        raise OverheadInputError("unexpected final model-task/method row count")
    if any(row["n_tasks"] != 56 for row in summaries):
        raise OverheadInputError("every summary stratum must contain 56 tasks")

    payload = {
        "schema_version": 1,
        "artifact": "confirmatory_observer_overhead",
        "code_tree_sha256": live_hash_before,
        "estimand": {
            "unit": "model-task",
            "active": (
                "observed resource use of probe calls carried inside the active task trajectory"
            ),
            "passive_provider": (
                "observed resource use of monitor calls over the immutable clean trajectory"
            ),
            "passive_deterministic": (
                "provider overhead is exactly zero; local CPU time was not instrumented"
            ),
            "latency": (
                "sum of provider attempt elapsed_ms; active is on-path delay, passive is "
                "off-path compute time and does not imply task delay"
            ),
            "tokens": "provider-reported input and output tokens; total_tokens is their sum",
            "calls": "provider attempts; all included observer attempts succeeded on attempt 1",
            "cost": (
                "reconciled ledger-accounted USD estimate; cost_quality and observer model "
                "are retained in every provider-backed task row"
            ),
            "excluded": "all agent task-call resource use",
        },
        "methods": [
            {
                "method": row["method"],
                "observation_class": row["observation_class"],
                "provider_backed": row["provider_backed"],
                "receipt_source": row["receipt_source"],
            }
            for row in METHODS
        ],
        "source_runs": source_inventory,
        "counts": {
            "benchmarks": 2,
            "benchmark_model_strata": 7,
            "model_tasks": n_model_tasks,
            "methods": len(METHODS),
            "task_method_rows": len(all_rows),
        },
        "summaries": summaries,
        "task_method_rows": all_rows,
    }
    json_path = OUTPUT_STEM.with_suffix(".json")
    task_csv_path = OUTPUT_STEM.with_suffix(".tasks.csv")
    summary_csv_path = OUTPUT_STEM.with_suffix(".summary.csv")
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    svg_data_path = OUTPUT_STEM.with_suffix(".svg.data.json")
    receipt_path = OUTPUT_STEM.with_suffix(".receipt.json")

    json_sha = atomic_write_json(json_path, payload)
    task_fields = (
        "run_id",
        "benchmark",
        "model",
        "task_id",
        "condition",
        "replicate_id",
        "method",
        "observation_class",
        "provider_backed",
        "receipt_source",
        "provider",
        "observer_model",
        "cost_quality",
        "checkpoint_count",
        "provider_call_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "latency_ms",
        "cost_micro_usd",
        "cost_usd",
    )
    task_csv_sha = atomic_write_bytes(task_csv_path, _csv_bytes(all_rows, task_fields))
    summary_fields = tuple(summaries[0].keys())
    summary_csv_sha = atomic_write_bytes(
        summary_csv_path, _csv_bytes(summaries, summary_fields)
    )
    svg, sidecar = _figure(summaries, source_json_sha256=json_sha)
    svg_sha = atomic_write_bytes(svg_path, svg.encode("utf-8"))
    svg_data_sha = atomic_write_json(svg_data_path, sidecar)

    live_hash_after = code_tree_hash(PACKAGE)
    if live_hash_after != live_hash_before:
        raise OverheadInputError(
            f"frozen code tree changed during extraction: {live_hash_after}"
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
            str(task_csv_path.relative_to(ROOT)): task_csv_sha,
            str(summary_csv_path.relative_to(ROOT)): summary_csv_sha,
            str(svg_path.relative_to(ROOT)): svg_sha,
            str(svg_data_path.relative_to(ROOT)): svg_data_sha,
        },
    }
    receipt_sha = atomic_write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "code_tree_sha256": live_hash_after,
                "model_tasks": n_model_tasks,
                "task_method_rows": len(all_rows),
                "summary_rows": len(summaries),
                "receipt": str(receipt_path.relative_to(ROOT)),
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
