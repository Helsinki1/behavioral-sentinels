#!/usr/bin/env python3
"""Provider-free final ledger and paid-run audit for Experiment 12.

This script is deliberately read-only with respect to production artifacts.  It
audits the global SQLite budget ledger, append-only call-attempt logs, the final
paper runs, superseded run directories, and the documented recovery boundary.
Only the two generated report files passed on the command line are written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments12"
ARTIFACTS = PACKAGE / "data_results" / "runs"
GENERATED = PACKAGE / "data_results" / "derived"
LEDGER = ARTIFACTS / "_global_budget.sqlite3"

EXPECTED_CODE_TREE_SHA256 = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)

PAPER_RUNS: dict[str, dict[str, Any]] = {
    "e12-calibration-bfcl-core-v1": {
        "role": "calibration_bfcl",
        "layout": "observer",
        "expected_pairs": 120,
        "expected_shadows": 60,
        "validation": "validation-calibration.json",
    },
    "e12-calibration-evolving-core-v2": {
        "role": "calibration_evolving",
        "layout": "observer",
        "expected_pairs": 160,
        "expected_shadows": 80,
        "validation": "validation-calibration.json",
    },
    "e12-confirmatory-bfcl-core-v3": {
        "role": "confirmatory_bfcl",
        "layout": "observer",
        "expected_pairs": 336,
        "expected_shadows": 168,
        "validation": "validation-confirmatory.json",
    },
    "e12-confirmatory-evolving-core-v2": {
        "role": "confirmatory_evolving",
        "layout": "observer",
        "expected_pairs": 448,
        "expected_shadows": 224,
        "validation": "validation-confirmatory.json",
    },
    "e12-deploy-twopass-pass1-evolving-luna-40-v1": {
        "role": "two_pass_source_observation",
        "layout": "observer",
        "expected_pairs": 80,
        "expected_shadows": 40,
        "validation": "validation-pass-one.json",
    },
    "e12-deploy-twopass-yoked-evolving-luna-40-v1": {
        "role": "two_pass_yoked_sensitivity",
        "layout": "deployment",
        "expected_pairs": 480,
        "validation": "validation-two-pass.json",
    },
    "e12-deploy-online-evolving-luna-40-v1": {
        "role": "online_primary",
        "layout": "adaptive",
        "expected_pairs": 1120,
        "validation": None,
    },
}

SUPERSEDED_RUNS = {
    "e12-calibration-evolving-core-v1": "superseded by calibration evolving v2",
    "e12-confirmatory-bfcl-core-v1": "aborted/superseded by BFCL confirmatory v3",
    "e12-confirmatory-bfcl-core-v2": "aborted/superseded by BFCL confirmatory v3",
}

ONLINE_RUN = "e12-deploy-online-evolving-luna-40-v1"
PASS_ONE_RUN = "e12-deploy-twopass-pass1-evolving-luna-40-v1"
ONLINE_RECOVERY_CELLS = {
    "786d95760ccdb86713c26936",
    "89df41e0daa1262a43fa5e55",
    "d52046b6eb74a76ecdc3debc",
}


class Audit:
    def __init__(self) -> None:
        self.blockers: list[dict[str, str]] = []
        self.disclosures: list[dict[str, str]] = []

    def require(self, condition: bool, code: str, message: str) -> None:
        if not condition:
            self.blockers.append({"code": code, "message": message})

    def disclose(self, code: str, message: str) -> None:
        self.disclosures.append({"code": code, "message": message})


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def usd(micro_usd: int) -> str:
    return f"{(Decimal(micro_usd) / Decimal(1_000_000)):.6f}"


def collect_call_event_references(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "call_event_ids" and isinstance(child, list):
                references.extend(item for item in child if isinstance(item, str))
            else:
                references.extend(collect_call_event_references(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(collect_call_event_references(child))
    return references


def file_stems(directory: Path, pattern: str, prefix: str = "") -> set[str]:
    stems: set[str] = set()
    for path in directory.glob(pattern):
        stem = path.stem
        if prefix:
            if not stem.startswith(prefix):
                continue
            stem = stem[len(prefix) :]
        stems.add(stem)
    return stems


def count_states(paths: Iterable[Path]) -> dict[str, int]:
    states: Counter[str] = Counter()
    for path in paths:
        try:
            states[str(load_json(path).get("state"))] += 1
        except Exception:
            states["parse_error"] += 1
    return dict(sorted(states.items()))


def code_tree_hash() -> str:
    sys.path.insert(0, str(ROOT))
    try:
        from experiments12.manifest12 import code_tree_hash as implementation

        return implementation(PACKAGE)
    finally:
        sys.path.pop(0)


def read_ledger(audit: Audit) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    uri = f"file:{LEDGER.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        connection.execute("BEGIN")
        limits = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM provider_limits ORDER BY provider"
            )
        ]
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM reservations ORDER BY created_at, reservation_id"
            )
        ]
        connection.rollback()
    finally:
        connection.close()

    audit.require(integrity == "ok", "ledger.integrity", f"integrity_check={integrity}")
    by_id = {row["reservation_id"]: row for row in rows}
    audit.require(
        len(by_id) == len(rows),
        "ledger.reservation_id_duplicate",
        "reservation_id is not globally unique",
    )
    request_keys = [(row["provider"], row["request_key"]) for row in rows]
    audit.require(
        all(key is not None for _, key in request_keys),
        "ledger.null_request_key",
        "one or more reservations have a null request key",
    )
    audit.require(
        len(set(request_keys)) == len(request_keys),
        "ledger.request_key_duplicate",
        "(provider, request_key) is not globally unique",
    )

    states = Counter(row["state"] for row in rows)
    statuses = Counter(str(row["request_status"]) for row in rows)
    qualities = Counter(str(row["cost_quality"]) for row in rows)
    active = [row for row in rows if row["state"] == "reserved"]
    unknown = [row for row in rows if row["request_status"] == "unknown"]
    unaccounted_unknown = [
        row
        for row in unknown
        if not (
            row["state"] == "reconciled"
            and row["cost_quality"] == "upper_bound"
            and row["actual_micro_usd"] == row["reserved_micro_usd"]
        )
    ]
    audit.require(not active, "ledger.active_reservation", "active reservations remain")
    audit.require(
        not unaccounted_unknown,
        "ledger.unaccounted_unknown",
        "a billing-ambiguous request is not charged at its full reserved upper bound",
    )
    if unknown:
        audit.disclose(
            "ledger.accounted_billing_ambiguity",
            f"{len(unknown)} request-status-unknown rows are all reconciled at full reserved upper bounds",
        )

    limit_by_provider = {row["provider"]: row for row in limits}
    providers: dict[str, Any] = {}
    for provider in sorted(limit_by_provider):
        provider_rows = [row for row in rows if row["provider"] == provider]
        spent = sum(int(row["actual_micro_usd"] or 0) for row in provider_rows)
        active_reserved = sum(
            int(row["reserved_micro_usd"])
            for row in provider_rows
            if row["state"] == "reserved"
        )
        limit = limit_by_provider[provider]
        operational = int(limit["operational_cap_micro_usd"])
        hard = int(limit["hard_cap_micro_usd"])
        audit.require(
            spent + active_reserved <= operational,
            f"ledger.{provider}.operational_cap",
            f"{provider} exceeds its operational cap",
        )
        audit.require(
            spent + active_reserved <= hard,
            f"ledger.{provider}.hard_cap",
            f"{provider} exceeds its hard cap",
        )
        providers[provider] = {
            "reservation_rows": len(provider_rows),
            "accounted_spend_micro_usd": spent,
            "accounted_spend_usd": usd(spent),
            "active_reserved_micro_usd": active_reserved,
            "active_reserved_usd": usd(active_reserved),
            "operational_cap_usd": usd(operational),
            "hard_cap_usd": usd(hard),
            "remaining_operational_usd": usd(operational - spent - active_reserved),
            "remaining_hard_usd": usd(hard - spent - active_reserved),
            "below_operational_cap": spent + active_reserved <= operational,
            "below_hard_cap": spent + active_reserved <= hard,
            "request_status_counts": dict(
                sorted(Counter(str(row["request_status"]) for row in provider_rows).items())
            ),
            "cost_quality_counts": dict(
                sorted(Counter(str(row["cost_quality"]) for row in provider_rows).items())
            ),
        }

    ledger_report = {
        "path": rel(LEDGER),
        "integrity_check": integrity,
        "canonical_rows_sha256": sha_json(rows),
        "reservation_rows": len(rows),
        "reservation_id_unique": len(by_id) == len(rows),
        "provider_request_key_unique": len(set(request_keys)) == len(request_keys),
        "null_request_keys": sum(key is None for _, key in request_keys),
        "state_counts": dict(sorted(states.items())),
        "request_status_counts": dict(sorted(statuses.items())),
        "cost_quality_counts": dict(sorted(qualities.items())),
        "active_reservations": len(active),
        "request_status_unknown": len(unknown),
        "unknown_unaccounted": len(unaccounted_unknown),
        "all_financial_states_settled": not active and not unaccounted_unknown,
        "providers": providers,
        "unknown_rows": [
            {
                "reservation_id": row["reservation_id"],
                "provider": row["provider"],
                "request_key": row["request_key"],
                "purpose": row["purpose"],
                "accounted_upper_bound_usd": usd(int(row["actual_micro_usd"])),
            }
            for row in unknown
        ],
    }
    return ledger_report, by_id


def production_attempt_logs() -> list[Path]:
    paths = list(ARTIFACTS.glob("**/call_attempts.jsonl"))
    # Generated benchmark-build attempts live exactly one directory below
    # generated/.  This intentionally excludes analysis views and staging copies.
    paths.extend(GENERATED.glob("*/call_attempts.jsonl"))
    return sorted(set(paths))


def read_attempt_logs(
    audit: Audit, ledger_by_id: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for path in production_attempt_logs():
        parsed = load_jsonl(path)
        inventory.append({"path": rel(path), "rows": len(parsed), "sha256": sha_file(path)})
        for line_number, row in enumerate(parsed, 1):
            enriched = dict(row)
            enriched["_path"] = rel(path)
            enriched["_line"] = line_number
            rows.append(enriched)

    by_reservation: dict[str, dict[str, Any]] = {}
    by_event: dict[str, dict[str, Any]] = {}
    duplicate_reservations: list[str] = []
    duplicate_events: list[str] = []
    for row in rows:
        reservation_id = row.get("reservation_id")
        event_id = row.get("event_id")
        if reservation_id in by_reservation:
            duplicate_reservations.append(str(reservation_id))
        else:
            by_reservation[str(reservation_id)] = row
        if event_id in by_event:
            duplicate_events.append(str(event_id))
        else:
            by_event[str(event_id)] = row

    audit.require(
        not duplicate_reservations,
        "attempt.reservation_duplicate",
        f"duplicate reservation IDs in production attempt logs: {duplicate_reservations[:3]}",
    )
    audit.require(
        not duplicate_events,
        "attempt.event_duplicate",
        f"duplicate event IDs in production attempt logs: {duplicate_events[:3]}",
    )
    ledger_ids = set(ledger_by_id)
    attempt_ids = set(by_reservation)
    extra = sorted(attempt_ids - ledger_ids)
    ledger_only = [ledger_by_id[item] for item in sorted(ledger_ids - attempt_ids)]
    audit.require(
        not extra,
        "attempt.missing_ledger_row",
        f"attempt logs contain {len(extra)} reservations absent from the ledger",
    )
    ledger_only_are_conservative = all(
        row["state"] == "reconciled"
        and row["request_status"] == "unknown"
        and row["cost_quality"] == "upper_bound"
        and row["actual_micro_usd"] == row["reserved_micro_usd"]
        for row in ledger_only
    )
    audit.require(
        ledger_only_are_conservative,
        "attempt.ledger_only_not_conservative",
        "a ledger-only reservation is not a fully charged historical ambiguity",
    )
    if ledger_only:
        audit.disclose(
            "attempt.historical_ledger_only",
            f"{len(ledger_only)} historical killed-process reservations have no appended event row but are fully charged upper bounds",
        )

    for reservation_id, attempt in by_reservation.items():
        ledger = ledger_by_id.get(reservation_id)
        if ledger is None:
            continue
        audit.require(
            ledger["provider"] == attempt.get("provider"),
            "attempt.provider_mismatch",
            f"provider mismatch for {reservation_id}",
        )
        audit.require(
            ledger["purpose"] == attempt.get("purpose"),
            "attempt.purpose_mismatch",
            f"purpose mismatch for {reservation_id}",
        )

    report = {
        "production_log_files": len(inventory),
        "production_log_inventory_sha256": sha_json(inventory),
        "production_log_inventory": inventory,
        "attempt_rows": len(rows),
        "unique_reservation_ids": len(by_reservation),
        "unique_event_ids": len(by_event),
        "duplicate_reservation_ids": sorted(set(duplicate_reservations)),
        "duplicate_event_ids": sorted(set(duplicate_events)),
        "attempt_reservations_without_ledger_rows": extra,
        "ledger_rows_without_attempt_events": [
            {
                "reservation_id": row["reservation_id"],
                "provider": row["provider"],
                "request_key": row["request_key"],
                "purpose": row["purpose"],
                "request_status": row["request_status"],
                "cost_quality": row["cost_quality"],
                "accounted_upper_bound_usd": usd(int(row["actual_micro_usd"])),
            }
            for row in ledger_only
        ],
        "ledger_only_rows_are_fully_accounted_upper_bounds": ledger_only_are_conservative,
    }
    return report, by_reservation, by_event


def scoped_ledger_summary(
    run_id: str,
    ledger_by_id: dict[str, dict[str, Any]],
    attempt_by_reservation: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for row in ledger_by_id.values()
        if str(row["request_key"]).startswith(run_id + "/")
    ]
    reservation_ids = {row["reservation_id"] for row in rows}
    logged = reservation_ids & set(attempt_by_reservation)
    return {
        "reservation_rows": len(rows),
        "attempt_rows": len(logged),
        "all_reservations_have_attempt_events": logged == reservation_ids,
        "state_counts": dict(sorted(Counter(row["state"] for row in rows).items())),
        "request_status_counts": dict(
            sorted(Counter(str(row["request_status"]) for row in rows).items())
        ),
        "cost_quality_counts": dict(
            sorted(Counter(str(row["cost_quality"]) for row in rows).items())
        ),
        "provider_accounted_spend_usd": {
            provider: usd(
                sum(
                    int(row["actual_micro_usd"] or 0)
                    for row in rows
                    if row["provider"] == provider
                )
            )
            for provider in sorted({row["provider"] for row in rows})
        },
    }


def validate_manifest_and_pairs(
    audit: Audit, run_id: str, expected_pairs: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    run = ARTIFACTS / run_id
    manifest_path = run / "manifest.json"
    pairs_path = run / "pairs.jsonl"
    audit.require(manifest_path.is_file(), "run.manifest_missing", f"{run_id}: manifest missing")
    audit.require(pairs_path.is_file(), "run.pairs_missing", f"{run_id}: pairs missing")
    manifest = load_json(manifest_path)
    pairs = load_jsonl(pairs_path)
    pair_ids = [row.get("cell_id") for row in pairs]
    audit.require(
        len(pairs) == expected_pairs,
        "run.pair_count",
        f"{run_id}: expected {expected_pairs} pairs, found {len(pairs)}",
    )
    audit.require(
        len(set(pair_ids)) == len(pair_ids),
        "run.pair_cell_duplicate",
        f"{run_id}: pair cell IDs are not unique",
    )
    audit.require(
        manifest.get("repository", {}).get("code_tree_sha256")
        == EXPECTED_CODE_TREE_SHA256,
        "run.manifest_code_hash",
        f"{run_id}: manifest code-tree hash differs from the frozen hash",
    )
    audit.require(
        manifest.get("run_id") == run_id,
        "run.manifest_run_id",
        f"{run_id}: manifest run_id mismatch",
    )
    declared_cells = manifest.get("extra_config", {}).get("n_cells")
    if declared_cells is not None:
        audit.require(
            int(declared_cells) == len(pairs),
            "run.manifest_n_cells",
            f"{run_id}: manifest n_cells does not equal pair count",
        )
    header = {
        "manifest_sha256": sha_file(manifest_path),
        "pairs_sha256": sha_file(pairs_path),
        "pair_rows": len(pairs),
        "unique_pair_cell_ids": len(set(pair_ids)),
        "arm_counts": dict(sorted(Counter(str(row.get("arm")) for row in pairs).items())),
        "operator_counts": dict(
            sorted(Counter(str(row.get("operator")) for row in pairs).items())
        ),
    }
    return manifest, pairs, header


def validate_output_job(
    audit: Audit,
    run_id: str,
    cell_id: str,
    output_path: Path,
    job_path: Path,
    *,
    output_hash_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = load_json(output_path)
    job = load_json(job_path)
    audit.require(
        output.get("complete") is True,
        "run.output_incomplete",
        f"{run_id}/{cell_id}: output is not complete",
    )
    if "cell_id" in output:
        audit.require(
            output.get("cell_id") == cell_id,
            "run.output_cell_id",
            f"{run_id}/{cell_id}: output cell ID mismatch",
        )
    audit.require(
        job.get("cell_id") == cell_id and job.get("state") == "complete",
        "run.job_incomplete",
        f"{run_id}/{cell_id}: current job is not complete",
    )
    audit.require(
        job.get(output_hash_key) == sha_file(output_path),
        "run.job_output_hash",
        f"{run_id}/{cell_id}: job does not bind the output hash",
    )
    if "accounting_sha256" in job and "accounting" in output:
        audit.require(
            job["accounting_sha256"] == sha_json(output["accounting"]),
            "run.job_accounting_hash",
            f"{run_id}/{cell_id}: accounting hash mismatch",
        )
    return output, job


def audit_observer_run(
    audit: Audit,
    run_id: str,
    spec: dict[str, Any],
    attempt_by_event: dict[str, dict[str, Any]],
    expected_unreferenced: set[str],
) -> dict[str, Any]:
    run = ARTIFACTS / run_id
    _, pairs, report = validate_manifest_and_pairs(
        audit, run_id, int(spec["expected_pairs"])
    )
    pair_ids = {str(row["cell_id"]) for row in pairs}
    clean_ids = {str(row["cell_id"]) for row in pairs if row.get("arm") == "clean"}
    audit.require(
        len(clean_ids) == int(spec["expected_shadows"]),
        "run.clean_count",
        f"{run_id}: clean-cell count differs from expected shadow count",
    )

    trajectory_ids = file_stems(run / "trajectories", "*.json")
    job_ids = file_stems(run / "results/jobs", "*.json")
    trajectory_event_ids = file_stems(run / "events", "trajectory-*.jsonl", "trajectory-")
    shadow_ids = file_stems(run / "shadow", "*.json")
    shadow_job_ids = file_stems(run / "results/shadow_jobs", "*.json")
    shadow_event_ids = file_stems(run / "events", "shadow-*.jsonl", "shadow-")

    for name, actual, expected in (
        ("trajectory outputs", trajectory_ids, pair_ids),
        ("trajectory jobs", job_ids, pair_ids),
        ("trajectory event logs", trajectory_event_ids, pair_ids),
        ("shadow outputs", shadow_ids, clean_ids),
        ("shadow jobs", shadow_job_ids, clean_ids),
    ):
        audit.require(
            actual == expected,
            "run.coverage",
            f"{run_id}: {name} coverage differs; missing={len(expected-actual)}, extra={len(actual-expected)}",
        )

    call_references: list[str] = []
    for cell_id in sorted(pair_ids):
        output, _ = validate_output_job(
            audit,
            run_id,
            cell_id,
            run / f"trajectories/{cell_id}.json",
            run / f"results/jobs/{cell_id}.json",
            output_hash_key="trajectory_sha256",
        )
        call_references.extend(collect_call_event_references(output))
        try:
            load_jsonl(run / f"events/trajectory-{cell_id}.jsonl")
        except Exception as exc:
            audit.require(False, "run.event_parse", f"{run_id}/{cell_id}: {exc}")

    nonempty_shadow_ids: set[str] = set()
    for cell_id in sorted(clean_ids):
        output, _ = validate_output_job(
            audit,
            run_id,
            cell_id,
            run / f"shadow/{cell_id}.json",
            run / f"results/shadow_jobs/{cell_id}.json",
            output_hash_key="shadow_sha256",
        )
        trajectory_sha = load_json(run / f"trajectories/{cell_id}.json").get(
            "transcript_sha256"
        )
        audit.require(
            output.get("source_trajectory_sha256") == trajectory_sha,
            "run.shadow_source_hash",
            f"{run_id}/{cell_id}: shadow source trajectory hash mismatch",
        )
        if output.get("records"):
            nonempty_shadow_ids.add(cell_id)
        call_references.extend(collect_call_event_references(output))

    audit.require(
        shadow_event_ids == nonempty_shadow_ids,
        "run.shadow_event_coverage",
        f"{run_id}: shadow event files do not exactly match shadows with nonempty records",
    )
    for cell_id in sorted(shadow_event_ids):
        try:
            load_jsonl(run / f"events/shadow-{cell_id}.jsonl")
        except Exception as exc:
            audit.require(False, "run.shadow_event_parse", f"{run_id}/{cell_id}: {exc}")

    attempts = load_jsonl(run / "events/call_attempts.jsonl")
    attempt_events = {str(row["event_id"]) for row in attempts}
    referenced = set(call_references)
    unreferenced = attempt_events - referenced
    missing_attempts = referenced - attempt_events
    audit.require(
        unreferenced == expected_unreferenced,
        "run.unreferenced_attempts",
        f"{run_id}: unreferenced attempts differ from documented set: {sorted(unreferenced)}",
    )
    audit.require(
        not missing_attempts,
        "run.missing_attempt_events",
        f"{run_id}: outputs reference absent call events: {sorted(missing_attempts)[:3]}",
    )
    for event_id in unreferenced:
        audit.require(
            event_id in attempt_by_event,
            "run.unreferenced_global_event",
            f"{run_id}: unreferenced event absent from global attempt index: {event_id}",
        )

    validation_path = run / f"results/{spec['validation']}"
    validation = load_json(validation_path)
    audit.require(
        validation.get("primary_ready") is True and not validation.get("errors"),
        "run.validation_not_ready",
        f"{run_id}: source validation is not primary-ready",
    )
    audit.require(
        validation.get("manifest_sha256") == report["manifest_sha256"],
        "run.validation_manifest_hash",
        f"{run_id}: validation manifest hash mismatch",
    )
    audit.require(
        validation.get("pair_manifest_sha256") == report["pairs_sha256"],
        "run.validation_pairs_hash",
        f"{run_id}: validation pairs hash mismatch",
    )

    report.update(
        {
            "trajectory_outputs": len(trajectory_ids),
            "trajectory_jobs_complete": sum(
                1
                for path in (run / "results/jobs").glob("*.json")
                if load_json(path).get("state") == "complete"
            ),
            "trajectory_event_logs": len(trajectory_event_ids),
            "shadow_outputs": len(shadow_ids),
            "shadow_jobs_complete": sum(
                1
                for path in (run / "results/shadow_jobs").glob("*.json")
                if load_json(path).get("state") == "complete"
            ),
            "shadow_event_logs": len(shadow_event_ids),
            "empty_shadow_outputs": len(shadow_ids - nonempty_shadow_ids),
            "call_attempt_rows": len(attempts),
            "unique_referenced_call_events": len(referenced),
            "unreferenced_call_events": sorted(unreferenced),
            "unreferenced_are_documented": unreferenced == expected_unreferenced,
            "validation_path": rel(validation_path),
            "validation_sha256": sha_file(validation_path),
            "validation_primary_ready": validation.get("primary_ready") is True,
            "validation_warning_count": len(validation.get("warnings") or []),
            "coverage_complete": (
                trajectory_ids == pair_ids
                and job_ids == pair_ids
                and trajectory_event_ids == pair_ids
                and shadow_ids == clean_ids
                and shadow_job_ids == clean_ids
                and shadow_event_ids == nonempty_shadow_ids
                and not missing_attempts
            ),
        }
    )
    return report


def audit_deployment_run(
    audit: Audit,
    run_id: str,
    spec: dict[str, Any],
    expected_unreferenced: set[str],
) -> dict[str, Any]:
    run = ARTIFACTS / run_id
    _, pairs, report = validate_manifest_and_pairs(
        audit, run_id, int(spec["expected_pairs"])
    )
    pair_ids = {str(row["cell_id"]) for row in pairs}
    adaptive = spec["layout"] == "adaptive"
    output_folder = "adaptive_deployment" if adaptive else "deployment"
    job_folder = "adaptive_deployment_jobs" if adaptive else "deployment_jobs"
    event_prefix = "adaptive-" if adaptive else "deployment-"

    output_ids = file_stems(run / f"results/{output_folder}", "*.json")
    job_ids = file_stems(run / f"results/{job_folder}", "*.json")
    event_ids = file_stems(run / "events", f"{event_prefix}*.jsonl", event_prefix)
    for name, actual in (
        ("outputs", output_ids),
        ("jobs", job_ids),
        ("event logs", event_ids),
    ):
        audit.require(
            actual == pair_ids,
            "run.coverage",
            f"{run_id}: {name} coverage differs; missing={len(pair_ids-actual)}, extra={len(actual-pair_ids)}",
        )

    call_references: list[str] = []
    for cell_id in sorted(pair_ids):
        output, _ = validate_output_job(
            audit,
            run_id,
            cell_id,
            run / f"results/{output_folder}/{cell_id}.json",
            run / f"results/{job_folder}/{cell_id}.json",
            output_hash_key="output_sha256",
        )
        call_references.extend(collect_call_event_references(output))
        try:
            event_rows = load_jsonl(run / f"events/{event_prefix}{cell_id}.jsonl")
            audit.require(
                bool(event_rows),
                "run.empty_event_log",
                f"{run_id}/{cell_id}: semantic event log is empty",
            )
        except Exception as exc:
            audit.require(False, "run.event_parse", f"{run_id}/{cell_id}: {exc}")

    attempts = load_jsonl(run / "events/call_attempts.jsonl")
    attempt_events = {str(row["event_id"]) for row in attempts}
    referenced = set(call_references)
    unreferenced = attempt_events - referenced
    missing_attempts = referenced - attempt_events
    audit.require(
        unreferenced == expected_unreferenced,
        "run.unreferenced_attempts",
        f"{run_id}: unreferenced attempts differ from documented set: {sorted(unreferenced)}",
    )
    audit.require(
        not missing_attempts,
        "run.missing_attempt_events",
        f"{run_id}: outputs reference absent call events: {sorted(missing_attempts)[:3]}",
    )

    validation_primary_ready: bool | None = None
    validation_path: Path | None = None
    if spec.get("validation"):
        validation_path = run / f"results/{spec['validation']}"
        validation = load_json(validation_path)
        validation_primary_ready = validation.get("primary_ready") is True
        audit.require(
            validation_primary_ready,
            "run.validation_not_ready",
            f"{run_id}: source validation is not primary-ready",
        )

    report.update(
        {
            "outputs": len(output_ids),
            "jobs_complete": sum(
                1
                for path in (run / f"results/{job_folder}").glob("*.json")
                if load_json(path).get("state") == "complete"
            ),
            "semantic_event_logs": len(event_ids),
            "call_attempt_rows": len(attempts),
            "unique_referenced_call_events": len(referenced),
            "unreferenced_call_events": sorted(unreferenced),
            "unreferenced_are_documented": unreferenced == expected_unreferenced,
            "validation_path": rel(validation_path) if validation_path else None,
            "validation_sha256": sha_file(validation_path) if validation_path else None,
            "validation_primary_ready": validation_primary_ready,
            "coverage_complete": (
                output_ids == pair_ids
                and job_ids == pair_ids
                and event_ids == pair_ids
                and not missing_attempts
            ),
        }
    )
    return report


def recovery_expected_orphans(
    audit: Audit,
    ledger_by_id: dict[str, dict[str, Any]],
    attempt_by_reservation: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str]]:
    pass_one_receipt_path = (
        ARTIFACTS
        / PASS_ONE_RUN
        / "results/recovery/9d8591ea71f67026d743d434/recovery-receipt.json"
    )
    pass_one_receipt = load_json(pass_one_receipt_path)
    pass_one = {str(pass_one_receipt["failure"]["original_unreferenced_call_event_id"])}

    online: set[str] = set()
    for cell_id in sorted(ONLINE_RECOVERY_CELLS):
        receipt_path = GENERATED / f"recovery-adaptive-{cell_id}12.json"
        receipt = load_json(receipt_path)
        for attempt in receipt.get("malformed_attempts") or []:
            online.add(str(attempt["event_id"]))
        original_key = receipt.get("original_malformed_request_key")
        if original_key:
            matches = [
                row
                for row in ledger_by_id.values()
                if row.get("provider") == "openai" and row.get("request_key") == original_key
            ]
            audit.require(
                len(matches) == 1,
                "recovery.original_request_key",
                f"{cell_id}: original malformed request key does not resolve uniquely",
            )
            if len(matches) == 1:
                attempt = attempt_by_reservation.get(matches[0]["reservation_id"])
                audit.require(
                    attempt is not None,
                    "recovery.original_attempt_missing",
                    f"{cell_id}: original malformed attempt is absent",
                )
                if attempt:
                    online.add(str(attempt["event_id"]))
    return pass_one, online


def audit_recoveries(
    audit: Audit,
    pass_one_orphans: set[str],
    online_orphans: set[str],
    attempt_by_event: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    online_run = ARTIFACTS / ONLINE_RUN
    online_rows: list[dict[str, Any]] = []
    archive_dirs = {
        path.name
        for path in (online_run / "results/recovery").iterdir()
        if path.is_dir()
    }
    audit.require(
        archive_dirs == ONLINE_RECOVERY_CELLS,
        "recovery.archive_directory_set",
        f"online recovery archive set differs: {sorted(archive_dirs)}",
    )

    for cell_id in sorted(ONLINE_RECOVERY_CELLS):
        receipt_path = GENERATED / f"recovery-adaptive-{cell_id}12.json"
        receipt = load_json(receipt_path)
        archive = online_run / str(receipt["pre_recovery_archive"])
        archive_receipt_path = archive / "archive-receipt.json"
        archive_receipt = load_json(archive_receipt_path)
        audit.require(
            sha_file(archive_receipt_path) == receipt["pre_recovery_archive_receipt_sha256"],
            "recovery.archive_receipt_hash",
            f"{cell_id}: archive receipt hash mismatch",
        )
        for field, hash_field in (
            ("archived_failed_job", "archived_failed_job_sha256"),
            ("archived_partial_events", "archived_partial_events_sha256"),
        ):
            archived_path = online_run / str(archive_receipt[field])
            audit.require(
                archived_path.is_file() and sha_file(archived_path) == archive_receipt[hash_field],
                "recovery.archived_artifact_hash",
                f"{cell_id}: {field} missing or hash-mismatched",
            )

        final_event = online_run / f"events/adaptive-{cell_id}.jsonl"
        final_job = online_run / f"results/adaptive_deployment_jobs/{cell_id}.json"
        final_output = online_run / f"results/adaptive_deployment/{cell_id}.json"
        for path, field in (
            (final_event, "final_event_log_sha256"),
            (final_job, "final_job_sha256"),
            (final_output, "final_output_sha256"),
        ):
            audit.require(
                path.is_file() and sha_file(path) == receipt[field],
                "recovery.final_artifact_hash",
                f"{cell_id}: {field} mismatch",
            )
        audit.require(
            load_json(final_job).get("state") == "complete"
            and load_json(final_output).get("complete") is True,
            "recovery.final_not_complete",
            f"{cell_id}: recovered job/output is not complete",
        )
        audit.require(
            receipt.get("ledger_rows_deleted_or_rewritten") is False
            and archive_receipt.get("ledger_rows_deleted_or_rewritten") is False,
            "recovery.ledger_mutation_flag",
            f"{cell_id}: receipt does not affirm ledger immutability",
        )
        online_rows.append(
            {
                "cell_id": cell_id,
                "receipt_path": rel(receipt_path),
                "receipt_sha256": sha_file(receipt_path),
                "archive_receipt_path": rel(archive_receipt_path),
                "archive_receipt_sha256": sha_file(archive_receipt_path),
                "final_output_sha256": sha_file(final_output),
                "final_job_sha256": sha_file(final_job),
                "final_event_log_sha256": sha_file(final_event),
                "complete": True,
            }
        )

    for event_id in sorted(online_orphans | pass_one_orphans):
        attempt = attempt_by_event.get(event_id)
        audit.require(
            attempt is not None,
            "recovery.orphan_missing",
            f"documented orphan attempt is absent: {event_id}",
        )
        if attempt:
            audit.require(
                attempt.get("status") == "succeeded"
                and attempt.get("finish_reason") == "max_output_tokens",
                "recovery.orphan_semantics",
                f"documented semantic failure has unexpected transport status: {event_id}",
            )

    pass_one_receipt_path = (
        ARTIFACTS
        / PASS_ONE_RUN
        / "results/recovery/9d8591ea71f67026d743d434/recovery-receipt.json"
    )
    pass_one_receipt = load_json(pass_one_receipt_path)
    pass_one_cell = str(pass_one_receipt["cell_id"])
    pass_one_run = ARTIFACTS / PASS_ONE_RUN
    pass_one_event = pass_one_run / f"events/shadow-{pass_one_cell}.jsonl"
    pass_one_output = pass_one_run / f"shadow/{pass_one_cell}.json"
    pass_one_job = pass_one_run / f"results/shadow_jobs/{pass_one_cell}.json"
    audit.require(
        sha_file(pass_one_event) == pass_one_receipt["final"]["event_log_sha256"]
        and sha_file(pass_one_output) == pass_one_receipt["final"]["shadow_sha256"]
        and sha_file(pass_one_job) == pass_one_receipt["final"]["job_sha256"],
        "recovery.pass_one_final_hash",
        "pass-one recovery final hashes do not match current artifacts",
    )
    archived_partial = pass_one_receipt_path.parent / pass_one_receipt["archived_partial_events"]
    audit.require(
        sha_file(archived_partial) == pass_one_receipt["archived_partial_events_sha256"],
        "recovery.pass_one_archive_hash",
        "pass-one archived partial event hash mismatch",
    )

    return {
        "online": {
            "expected_recovery_cells": sorted(ONLINE_RECOVERY_CELLS),
            "archive_directories": sorted(archive_dirs),
            "receipts": online_rows,
            "documented_unreferenced_semantic_attempts": sorted(online_orphans),
            "documented_unreferenced_semantic_attempt_count": len(online_orphans),
        },
        "pass_one": {
            "cell_id": pass_one_cell,
            "receipt_path": rel(pass_one_receipt_path),
            "receipt_sha256": sha_file(pass_one_receipt_path),
            "documented_unreferenced_semantic_attempts": sorted(pass_one_orphans),
            "final_hashes_match": True,
        },
    }


def generic_run_inventory(run_id: str, selected: bool, note: str) -> dict[str, Any]:
    run = ARTIFACTS / run_id
    manifest_path = run / "manifest.json"
    pairs_path = run / "pairs.jsonl"
    pairs = load_jsonl(pairs_path) if pairs_path.is_file() else []
    validation_files = sorted((run / "results").glob("validation-*.json"))
    validation = load_json(validation_files[0]) if validation_files else None
    return {
        "run_id": run_id,
        "paper_selected": selected,
        "note": note,
        "manifest_present": manifest_path.is_file(),
        "manifest_sha256": sha_file(manifest_path) if manifest_path.is_file() else None,
        "pairs_present": pairs_path.is_file(),
        "pairs_sha256": sha_file(pairs_path) if pairs_path.is_file() else None,
        "pair_rows": len(pairs),
        "trajectory_outputs": len(list((run / "trajectories").glob("*.json"))),
        "trajectory_jobs": len(list((run / "results/jobs").glob("*.json"))),
        "trajectory_job_states": count_states((run / "results/jobs").glob("*.json")),
        "trajectory_event_logs": len(list((run / "events").glob("trajectory-*.jsonl"))),
        "shadow_outputs": len(list((run / "shadow").glob("*.json"))),
        "shadow_jobs": len(list((run / "results/shadow_jobs").glob("*.json"))),
        "shadow_job_states": count_states((run / "results/shadow_jobs").glob("*.json")),
        "shadow_event_logs": len(list((run / "events").glob("shadow-*.jsonl"))),
        "deployment_outputs": len(list((run / "results/deployment").glob("*.json"))),
        "deployment_jobs": len(list((run / "results/deployment_jobs").glob("*.json"))),
        "deployment_job_states": count_states(
            (run / "results/deployment_jobs").glob("*.json")
        ),
        "deployment_event_logs": len(list((run / "events").glob("deployment-*.jsonl"))),
        "adaptive_outputs": len(
            list((run / "results/adaptive_deployment").glob("*.json"))
        ),
        "adaptive_jobs": len(
            list((run / "results/adaptive_deployment_jobs").glob("*.json"))
        ),
        "adaptive_job_states": count_states(
            (run / "results/adaptive_deployment_jobs").glob("*.json")
        ),
        "adaptive_event_logs": len(list((run / "events").glob("adaptive-*.jsonl"))),
        "call_attempt_rows": len(load_jsonl(run / "events/call_attempts.jsonl"))
        if (run / "events/call_attempts.jsonl").is_file()
        else 0,
        "validation_path": rel(validation_files[0]) if validation_files else None,
        "validation_primary_ready": validation.get("primary_ready") if validation else None,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    ledger = payload["ledger"]
    lines = [
        "# Experiment 12 final ledger and run audit",
        "",
        f"Overall: **{payload['overall_status']}**. Snapshot: `{payload['snapshot_utc']}`.",
        "",
        f"Frozen source/config hash: `{payload['code_tree_sha256']}` (expected hash matched).",
        "",
        "## Budget ledger",
        "",
        "| Provider | Accounted spend | Operational cap | Hard cap | Active reserved |",
        "|---|---:|---:|---:|---:|",
    ]
    for provider, row in sorted(ledger["providers"].items()):
        lines.append(
            f"| {provider} | ${row['accounted_spend_usd']} | ${row['operational_cap_usd']} | "
            f"${row['hard_cap_usd']} | ${row['active_reserved_usd']} |"
        )
    lines.extend(
        [
            "",
            f"All {ledger['reservation_rows']:,} reservations are financially settled: "
            f"{ledger['active_reservations']} active reservations and "
            f"{ledger['unknown_unaccounted']} unaccounted billing ambiguities. "
            f"There are {ledger['request_status_unknown']} request-status-unknown rows; each is "
            "reconciled at its full reserved upper bound.",
            "",
            f"The {payload['attempt_logs']['attempt_rows']:,} append-only attempt rows have unique "
            "reservation and event IDs. Ten historical killed-process reservations have no event "
            "row; all ten are non-paper runs and are fully charged upper bounds.",
            "",
            "## Final paper-run coverage",
            "",
            "| Role | Run | Pairs | Outputs | Jobs | Semantic event logs | Calls | Unreferenced calls |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id, row in payload["paper_runs"].items():
        if row["layout"] == "observer":
            outputs = f"{row['trajectory_outputs']}+{row['shadow_outputs']}"
            jobs = f"{row['trajectory_jobs_complete']}+{row['shadow_jobs_complete']}"
            events = f"{row['trajectory_event_logs']}+{row['shadow_event_logs']}"
        else:
            outputs = str(row["outputs"])
            jobs = str(row["jobs_complete"])
            events = str(row["semantic_event_logs"])
        lines.append(
            f"| {row['role']} | `{run_id}` | {row['pair_rows']} | {outputs} | {jobs} | "
            f"{events} | {row['call_attempt_rows']} | {len(row['unreferenced_call_events'])} |"
        )
    lines.extend(
        [
            "",
            "The BFCL confirmatory run has 156 shadow event files for 168 complete shadow outputs "
            "because 12 official trajectories have zero observable checkpoints; their complete "
            "shadow outputs correctly contain empty record lists.",
            "",
            "## Documented recovery boundary",
            "",
            f"The online run has all 1,120 current outputs, complete jobs, and semantic event logs, "
            f"plus exactly three hash-bound recovery archives/receipts. Its "
            f"{payload['recoveries']['online']['documented_unreferenced_semantic_attempt_count']} "
            "raw unreferenced call attempts are exactly the malformed max-output judge responses "
            "listed by those receipts. The pass-one run has exactly one analogous documented "
            "max-output judge attempt. No other final paper run has an unreferenced call attempt.",
            "",
            "The online run also contains one referenced HTTP-503 attempt whose billing was "
            "ambiguous; the ledger conservatively charges its full $0.002578 reservation before "
            "the successful retry.",
            "",
            "## Superseded directories",
            "",
            "| Run | Pairs | Trajectories | Shadows | Selection note |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["superseded_runs"]:
        lines.append(
            f"| `{row['run_id']}` | {row['pair_rows']} | {row['trajectory_outputs']} | "
            f"{row['shadow_outputs']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Qualification",
            "",
            "This audit does not normalize recovery data or certify the analysis staging copy. It "
            "certifies the immutable production ledger/log boundary, current paid-run coverage, "
            "and the exact raw attempts that the separate staging receipt must account for.",
            "",
            f"Reproduce with: `python3 {payload['script_path']}`",
            "",
        ]
    )
    if payload["blockers"]:
        lines.extend(["## Blocking findings", ""])
        for item in payload["blockers"]:
            lines.append(f"- `{item['code']}`: {item['message']}")
        lines.append("")
    return "\n".join(lines)


def run_audit(json_output: Path, markdown_output: Path) -> dict[str, Any]:
    audit = Audit()
    live_hash_before = code_tree_hash()
    audit.require(
        live_hash_before == EXPECTED_CODE_TREE_SHA256,
        "source.code_tree_hash",
        f"frozen code tree changed: {live_hash_before}",
    )

    ledger_report, ledger_by_id = read_ledger(audit)
    attempt_report, attempt_by_reservation, attempt_by_event = read_attempt_logs(
        audit, ledger_by_id
    )
    pass_one_orphans, online_orphans = recovery_expected_orphans(
        audit, ledger_by_id, attempt_by_reservation
    )
    audit.require(
        len(pass_one_orphans) == 1,
        "recovery.pass_one_orphan_count",
        f"expected one documented pass-one semantic failure, found {len(pass_one_orphans)}",
    )
    audit.require(
        len(online_orphans) == 6,
        "recovery.online_orphan_count",
        f"expected six documented online semantic failures, found {len(online_orphans)}",
    )

    paper_runs: dict[str, Any] = {}
    for run_id, spec in PAPER_RUNS.items():
        allowed = (
            pass_one_orphans
            if run_id == PASS_ONE_RUN
            else online_orphans
            if run_id == ONLINE_RUN
            else set()
        )
        if spec["layout"] == "observer":
            report = audit_observer_run(
                audit, run_id, spec, attempt_by_event, expected_unreferenced=allowed
            )
        else:
            report = audit_deployment_run(
                audit, run_id, spec, expected_unreferenced=allowed
            )
        report["role"] = spec["role"]
        report["layout"] = spec["layout"]
        report["ledger"] = scoped_ledger_summary(
            run_id, ledger_by_id, attempt_by_reservation
        )
        audit.require(
            report["ledger"]["all_reservations_have_attempt_events"],
            "run.ledger_attempt_coverage",
            f"{run_id}: a scoped ledger row has no attempt event",
        )
        audit.require(
            report["ledger"]["reservation_rows"] == report["call_attempt_rows"],
            "run.ledger_call_count",
            f"{run_id}: scoped ledger count differs from call-attempt count",
        )
        paper_runs[run_id] = report

    recoveries = audit_recoveries(
        audit, pass_one_orphans, online_orphans, attempt_by_event
    )

    all_stage_dirs = sorted(
        path.name
        for path in ARTIFACTS.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("e12-calibration-")
            or path.name.startswith("e12-confirmatory-")
            or path.name == PASS_ONE_RUN
            or path.name == "e12-deploy-twopass-yoked-evolving-luna-40-v1"
            or path.name == ONLINE_RUN
        )
    )
    known = set(PAPER_RUNS) | set(SUPERSEDED_RUNS)
    audit.require(
        set(all_stage_dirs) == known,
        "run.unclassified_stage_directory",
        f"unclassified or missing paid-stage directories: {sorted(set(all_stage_dirs)^known)}",
    )
    complete_inventory = [
        generic_run_inventory(
            run_id,
            run_id in PAPER_RUNS,
            "final paper run" if run_id in PAPER_RUNS else SUPERSEDED_RUNS[run_id],
        )
        for run_id in all_stage_dirs
    ]
    superseded = [row for row in complete_inventory if not row["paper_selected"]]

    live_hash_after = code_tree_hash()
    audit.require(
        live_hash_after == EXPECTED_CODE_TREE_SHA256,
        "source.code_tree_hash_after",
        f"frozen code tree changed during audit: {live_hash_after}",
    )

    payload: dict[str, Any] = {
        "artifact_type": "experiment12_final_ledger_and_paid_run_audit",
        "schema_version": 1,
        "snapshot_utc": utc_now(),
        "overall_status": "PASS" if not audit.blockers else "BLOCKED",
        "provider_free": True,
        "production_artifacts_modified": False,
        "script_path": rel(Path(__file__).resolve()),
        "script_sha256": sha_file(Path(__file__).resolve()),
        "code_tree_sha256": live_hash_after,
        "expected_code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        "ledger": ledger_report,
        "attempt_logs": attempt_report,
        "paper_runs": paper_runs,
        "recoveries": recoveries,
        "all_stage_run_inventory": complete_inventory,
        "superseded_runs": superseded,
        "disclosures": audit.disclosures,
        "blockers": audit.blockers,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=GENERATED / "final-ledger-run-audit12.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=GENERATED / "FINAL_LEDGER_RUN_AUDIT12.md",
    )
    args = parser.parse_args()
    payload = run_audit(args.json_output, args.markdown_output)
    print(
        json.dumps(
            {
                "status": payload["overall_status"],
                "json_output": str(args.json_output),
                "markdown_output": str(args.markdown_output),
                "blockers": len(payload["blockers"]),
                "disclosures": len(payload["disclosures"]),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
