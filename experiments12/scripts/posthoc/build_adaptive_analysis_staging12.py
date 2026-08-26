"""Build and analyze a copy-on-write view of the recovered adaptive run.

Production receipts remain immutable.  This utility copies the completed run
and its ledger, translates three semantically recovered judge calls into the
retry representation required by the frozen analyzer, and runs the stock
analysis plus a paired leave-two-source-units sensitivity analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence

from experiments12.adaptive_analysis12 import (
    ADAPTIVE_ANALYSIS_TYPE,
    extract_adaptive_run,
    summarize_adaptive_outcomes,
    write_adaptive_figures,
)
from experiments12.adaptive_deployment12 import _accounting
from experiments12.analysis12 import AnalysisInputError
from experiments12.cli12 import REPOSITORY_ROOT
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.manifest12 import RunLayout, code_tree_hash
from experiments12.pairing12 import JobCell


RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
MANIFEST_SHA256 = "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
PAIR_SHA256 = "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
CODE_TREE_SHA256 = "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
EXPECTED_CELLS = 1_120
EXPECTED_TREATMENTS = 28
EXPECTED_PRIMARY_N = 40
EXPECTED_SENSITIVITY_N = 38

SOURCE_ARTIFACTS = REPOSITORY_ROOT / "experiments12" / "data_results" / "runs"
SOURCE_LAYOUT = RunLayout.for_run(SOURCE_ARTIFACTS, RUN_ID)
STAGE_ARTIFACTS = (
    REPOSITORY_ROOT
    / "experiments12"
    / "data_results" / "derived"
    / "adaptive-analysis-staging-v1"
)
STAGE_LAYOUT = RunLayout.for_run(STAGE_ARTIFACTS, RUN_ID)
BUILD_RECEIPT = STAGE_ARTIFACTS / "staging-receipt.json"
ANALYSIS_DIR = STAGE_ARTIFACTS / "analysis"
ANALYSIS_PATH = ANALYSIS_DIR / "adaptive-analysis.json"
ANALYSIS_RECEIPT = STAGE_ARTIFACTS / "analysis-receipt.json"
FIGURE_DIR = ANALYSIS_DIR / "figures"
SENSITIVITY_PATH = ANALYSIS_DIR / "adaptive-analysis-leave-two-units.json"
SENSITIVITY_MD_PATH = ANALYSIS_DIR / "adaptive-analysis-leave-two-units.md"

RAW_ANALYZER_FAILURE = (
    "call attempt disagrees with ledger: d950af6bd8a8421e99f8efc17125fa1b"
)
GENERIC_LEDGER_REPAIR = {
    "event_id": "d950af6bd8a8421e99f8efc17125fa1b",
    "reservation_id": "abf82d1b70f9480db3c05659062e0a0b",
    "request_key": (
        f"{RUN_ID}/b0978e4007c1e796c0521807/adaptive-task-7/attempt-1"
    ),
    "from_request_status": "unknown",
    "to_request_status": "failed",
}


@dataclass(frozen=True, slots=True)
class AttemptLock:
    event_id: str
    attempt_sha256: str
    reservation_id: str
    physical_request_key: str
    logical_attempt_number: int
    logical_status: str


@dataclass(frozen=True, slots=True)
class RecoveryLock:
    cell_id: str
    source_unit_id: str
    checkpoint: int
    logical_request_key: str
    receipt_path: str
    receipt_sha256: str
    receipt_artifact_type: str
    production_event_log_sha256: str
    production_output_sha256: str
    production_job_sha256: str
    expected_tokens: int
    expected_elapsed_ms: int
    expected_cost_usd: str
    attempts: tuple[AttemptLock, ...]


RECOVERIES: tuple[RecoveryLock, ...] = (
    RecoveryLock(
        cell_id="d52046b6eb74a76ecdc3debc",
        source_unit_id="extracted-gsm8k-test-814::t7/r0",
        checkpoint=5,
        logical_request_key=(
            f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5"
        ),
        receipt_path=(
            "experiments12/data_results/derived/"
            "recovery-adaptive-d52046b6eb74a76ecdc3debc12.json"
        ),
        receipt_sha256=(
            "83f8939e08e7809d699e51e62a13b68aad838018669d4792ab6e84645741eca1"
        ),
        receipt_artifact_type="experiment12_online_adaptive_single_cell_recovery",
        production_event_log_sha256=(
            "ea4c354f34828a907a09e4496a84d02d33b2de023afab5ccde39c6e6152e8f75"
        ),
        production_output_sha256=(
            "2228470d0a4000c716b293ea026d43dc73025f9625275b2e0438922b6f83aa82"
        ),
        production_job_sha256=(
            "b107983e334ce559daafa4cbd21885f3ecd1897716a76b6db06b9e36487d4c96"
        ),
        expected_tokens=2_579,
        expected_elapsed_ms=13_844,
        expected_cost_usd="0.021550",
        attempts=(
            AttemptLock(
                event_id="6c1a460c1c704a4485f0957808a17e6b",
                attempt_sha256=(
                    "05cc4a63bc0d53e46c99f5d1cdff83c83dbbc93b8d36497d44a3a5bca377d439"
                ),
                reservation_id="0a22c9a145784a9ebb45747f6758aa6c",
                physical_request_key=(
                    f"{RUN_ID}/d52046b6eb74a76ecdc3debc/"
                    "adaptive-trace-judge-5/attempt-1"
                ),
                logical_attempt_number=1,
                logical_status="failed",
            ),
            AttemptLock(
                event_id="4e421d9936504860ad1afd61a58a5788",
                attempt_sha256=(
                    "395a31ac1a6126fd4ae624d0e4374701dd0e0305de38470b1755cb2db8e207bb"
                ),
                reservation_id="305dfec3f9b64f68aef2fc455393e32a",
                physical_request_key=(
                    f"{RUN_ID}/d52046b6eb74a76ecdc3debc/"
                    "adaptive-trace-judge-5-recovery-semantic-1/attempt-1"
                ),
                logical_attempt_number=2,
                logical_status="succeeded",
            ),
        ),
    ),
    RecoveryLock(
        cell_id="89df41e0daa1262a43fa5e55",
        source_unit_id="extracted-gsm8k-test-814::t7/r0",
        checkpoint=6,
        logical_request_key=(
            f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6"
        ),
        receipt_path=(
            "experiments12/data_results/derived/"
            "recovery-adaptive-89df41e0daa1262a43fa5e5512.json"
        ),
        receipt_sha256=(
            "7fdfe614fe976db85343586e4908785aa90ffe734045e629db9af5b46249329e"
        ),
        receipt_artifact_type="experiment12_online_adaptive_trace_judge_recovery",
        production_event_log_sha256=(
            "185d167b92a5fdc473f416a953a687134d8fc60926d3e4c45be504f4ab8e1b8d"
        ),
        production_output_sha256=(
            "4f4b8ee8dcc3074a41d6844de5bf824ec868b26e2394687dc7975ef770e31210"
        ),
        production_job_sha256=(
            "50423cdb86b68cb307b168b4a8bd1c98c1d462240bf826557064a8d2ab57b660"
        ),
        expected_tokens=5_352,
        expected_elapsed_ms=35_167,
        expected_cost_usd="0.048390",
        attempts=(
            AttemptLock(
                "dbfbaf64ae304165b85c33c030ed6729",
                "91ef59a700814342ff9949c41ff55ed71a212c123d2596774611666378110c82",
                "6e20aea9b36b4b0886bed2534c4cd395",
                f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6/attempt-1",
                1,
                "failed",
            ),
            AttemptLock(
                "11257c27688948aeb07a6397e0017f65",
                "1f4ef3e504f9f30ae19b04714b0898a13de0acac3440d9487d1bac9291646fa1",
                "4d9aa7eba85040beb7a5d70ca787a80a",
                f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-1/attempt-1",
                2,
                "failed",
            ),
            AttemptLock(
                "af9e3a5c95d44bb1aee521449e4886e9",
                "5ab0059228c61f8c3eb3f495a884123621c18b5e3ef3ad10dd5f86e179b43541",
                "3c7bf5b0a12840b19f27e8bf28c35d59",
                f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-2/attempt-1",
                3,
                "failed",
            ),
            AttemptLock(
                "08b9bae667e0495ea9bf9c06e16fc699",
                "a2324955b730a13909aad3b7ec0782b5afa46a1869dd40d04166eb5b29ded43b",
                "9335916349834914a4a80b90382d3373",
                f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-3/attempt-1",
                4,
                "failed",
            ),
            AttemptLock(
                "b1608d3c879d4f77a6adca6c05cd1fc8",
                "249c362cb308b1c6c596e5a9e3a0650241d10035cda1c8b6f30ce8f68f42d85b",
                "30535c6615204ebcba52e21327b88f80",
                f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-final-cap640-v1/attempt-1",
                5,
                "succeeded",
            ),
        ),
    ),
    RecoveryLock(
        cell_id="786d95760ccdb86713c26936",
        source_unit_id="extracted-gsm8k-test-989::t7/r0",
        checkpoint=5,
        logical_request_key=(
            f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5"
        ),
        receipt_path=(
            "experiments12/data_results/derived/"
            "recovery-adaptive-786d95760ccdb86713c2693612.json"
        ),
        receipt_sha256=(
            "0110cc242d6ffdec0c4fd1b1e45a606b5b7bee141a1d35a28fdc16f11d056509"
        ),
        receipt_artifact_type="experiment12_online_adaptive_trace_judge_recovery",
        production_event_log_sha256=(
            "7d434654b547953521a8735ddd7a51bfa1d496c997e2bf3c544d0d3cbb85c064"
        ),
        production_output_sha256=(
            "f73c38ac567f04b5fe2863137eacf90263ebbe5fedef1217c4530ad5b1335ca9"
        ),
        production_job_sha256=(
            "449f2c8fdc3bca52cdb3827a811a5529586cb41f0fba166aaaae9d7298d54cc3"
        ),
        expected_tokens=1_832,
        expected_elapsed_ms=13_448,
        expected_cost_usd="0.017050",
        attempts=(
            AttemptLock(
                "af34404456b84d27833193218a878154",
                "17d52ef53f50a034a8feb55095a3f6031e336be232d60b59c6e722ac0c32ba34",
                "3ab51cba528c46e6baa00f36a782defc",
                f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5/attempt-1",
                1,
                "failed",
            ),
            AttemptLock(
                "50045c026c3b44b8a4e38c5aab0509d9",
                "945ac143e66ec180fbd0496eb2f1711464aadb207ed524952f998e6d545ab38f",
                "e76435343cad49499f991e142b8871e3",
                f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5-recovery-final-cap640-v1/attempt-1",
                2,
                "succeeded",
            ),
        ),
    ),
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))


def _production_bindings(recovery: RecoveryLock) -> dict[str, Any]:
    event_path = SOURCE_LAYOUT.events / f"adaptive-{recovery.cell_id}.jsonl"
    output_path = SOURCE_LAYOUT.results / "adaptive_deployment" / f"{recovery.cell_id}.json"
    job_path = SOURCE_LAYOUT.results / "adaptive_deployment_jobs" / f"{recovery.cell_id}.json"
    bindings = {
        "event_log_sha256": sha256_file(event_path),
        "output_sha256": sha256_file(output_path),
        "job_sha256": sha256_file(job_path),
    }
    expected = {
        "event_log_sha256": recovery.production_event_log_sha256,
        "output_sha256": recovery.production_output_sha256,
        "job_sha256": recovery.production_job_sha256,
    }
    if bindings != expected:
        raise ValueError(f"production binding changed for {recovery.cell_id}")
    return bindings


def _open_ledger(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def preflight() -> dict[str, Any]:
    if sha256_file(SOURCE_LAYOUT.manifest) != MANIFEST_SHA256:
        raise ValueError("source manifest hash changed")
    if sha256_file(SOURCE_LAYOUT.pairs) != PAIR_SHA256:
        raise ValueError("source pair-table hash changed")
    if code_tree_hash(REPOSITORY_ROOT / "experiments12") != CODE_TREE_SHA256:
        raise ValueError("frozen Experiment 12 code-tree hash changed")

    cells = tuple(JobCell.from_dict(row) for row in read_jsonl(SOURCE_LAYOUT.pairs))
    if len(cells) != EXPECTED_CELLS or len({cell.cell_id for cell in cells}) != EXPECTED_CELLS:
        raise ValueError("source pair table does not contain exactly 1,120 unique cells")
    cell_index = {cell.cell_id: cell for cell in cells}
    complete_outputs = tuple(
        (SOURCE_LAYOUT.results / "adaptive_deployment").glob("*.json")
    )
    complete_jobs = tuple(
        (SOURCE_LAYOUT.results / "adaptive_deployment_jobs").glob("*.json")
    )
    adaptive_events = tuple(SOURCE_LAYOUT.events.glob("adaptive-*.jsonl"))
    if tuple(map(len, (complete_outputs, complete_jobs, adaptive_events))) != (
        EXPECTED_CELLS,
        EXPECTED_CELLS,
        EXPECTED_CELLS,
    ):
        raise ValueError("source run does not have exact 1,120-file coverage")
    if any(read_json(path).get("state") != "complete" for path in complete_jobs):
        raise ValueError("source run contains an incomplete job")

    attempt_path = SOURCE_LAYOUT.events / "call_attempts.jsonl"
    attempt_rows = read_jsonl(attempt_path)
    attempt_index = {
        str(row.get("event_id")): row
        for row in attempt_rows
        if isinstance(row, Mapping) and row.get("event_id")
    }
    if len(attempt_index) != len(attempt_rows):
        raise ValueError("source call-attempt stream duplicates an event ID")

    recovery_cells: set[str] = set()
    with _open_ledger(SOURCE_LAYOUT.ledger, read_only=True) as ledger:
        ledger_rows = {
            str(row["reservation_id"]): dict(row)
            for row in ledger.execute(
                "SELECT reservation_id, request_key, request_status, state "
                "FROM reservations WHERE request_key LIKE ?",
                (f"{RUN_ID}/%",),
            )
        }
        nonreconciled = [
            row for row in ledger_rows.values() if row.get("state") != "reconciled"
        ]
        if nonreconciled:
            raise ValueError("source run still has non-reconciled reservations")
        for row in ledger_rows.values():
            key = str(row["request_key"])
            if "-recovery-" in key:
                parts = key.split("/")
                if len(parts) >= 2:
                    recovery_cells.add(parts[1])

        recovery_rows: list[dict[str, Any]] = []
        for recovery in RECOVERIES:
            cell = cell_index.get(recovery.cell_id)
            if cell is None:
                raise ValueError(f"recovery cell is undeclared: {recovery.cell_id}")
            unit = f"{cell.pair_key.task_id}/r{cell.pair_key.replicate_id}"
            if unit != recovery.source_unit_id:
                raise ValueError(f"recovery source-unit binding changed: {recovery.cell_id}")
            receipt_path = REPOSITORY_ROOT / recovery.receipt_path
            if sha256_file(receipt_path) != recovery.receipt_sha256:
                raise ValueError(f"recovery receipt changed: {recovery.cell_id}")
            receipt = read_json(receipt_path)
            if (
                receipt.get("artifact_type") != recovery.receipt_artifact_type
                or receipt.get("run_id") != RUN_ID
                or receipt.get("cell_id") != recovery.cell_id
            ):
                raise ValueError(f"recovery receipt identity changed: {recovery.cell_id}")
            bindings = _production_bindings(recovery)
            for attempt in recovery.attempts:
                raw = attempt_index.get(attempt.event_id)
                if raw is None or sha256_json(raw) != attempt.attempt_sha256:
                    raise ValueError(f"production attempt changed: {attempt.event_id}")
                row = ledger_rows.get(attempt.reservation_id)
                if (
                    row is None
                    or row.get("request_key") != attempt.physical_request_key
                    or raw.get("reservation_id") != attempt.reservation_id
                ):
                    raise ValueError(f"attempt/ledger binding changed: {attempt.event_id}")
            recovery_rows.append(
                {
                    "cell_id": recovery.cell_id,
                    "source_unit_id": recovery.source_unit_id,
                    "checkpoint": recovery.checkpoint,
                    "receipt_path": recovery.receipt_path,
                    "receipt_sha256": recovery.receipt_sha256,
                    "receipt_artifact_type": recovery.receipt_artifact_type,
                    "production_bindings": bindings,
                    "physical_attempts": len(recovery.attempts),
                }
            )

        repair = GENERIC_LEDGER_REPAIR
        repair_attempt = attempt_index.get(str(repair["event_id"]))
        repair_row = ledger_rows.get(str(repair["reservation_id"]))
        if (
            repair_attempt is None
            or repair_attempt.get("status") != repair["to_request_status"]
            or repair_attempt.get("reservation_id") != repair["reservation_id"]
            or repair_row is None
            or repair_row.get("request_key") != repair["request_key"]
            or repair_row.get("request_status") != repair["from_request_status"]
        ):
            raise ValueError("generic failed-attempt ledger mismatch changed")

    expected_recovery_cells = {recovery.cell_id for recovery in RECOVERIES}
    if recovery_cells != expected_recovery_cells:
        raise ValueError(
            "physical recovery request keys do not identify exactly three cells"
        )

    try:
        extract_adaptive_run(
            SOURCE_LAYOUT,
            expected_manifest_sha256=MANIFEST_SHA256,
            bootstrap_iterations=1,
        )
    except AnalysisInputError as exc:
        raw_failure = str(exc)
    else:
        raise ValueError("raw production analyzer unexpectedly succeeded")
    if raw_failure != RAW_ANALYZER_FAILURE:
        raise ValueError(f"raw analyzer failure changed: {raw_failure}")

    return {
        "ready": True,
        "run_id": RUN_ID,
        "cells": EXPECTED_CELLS,
        "recovery_cells": recovery_rows,
        "recovery_cell_count": len(recovery_rows),
        "recovery_source_units": sorted(
            {recovery.source_unit_id for recovery in RECOVERIES}
        ),
        "recovery_source_unit_count": len(
            {recovery.source_unit_id for recovery in RECOVERIES}
        ),
        "raw_analyzer_expected_failure": raw_failure,
        "generic_ledger_repair": dict(GENERIC_LEDGER_REPAIR),
        "provider_calls": 0,
    }


def _copy_ledger(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_ledger(source, read_only=True) as source_db:
        with sqlite3.connect(destination) as target_db:
            source_db.backup(target_db)
    with sqlite3.connect(destination) as target_db:
        target_db.execute("PRAGMA journal_mode=DELETE")


def _patch_attempts_and_ledger(stage_root: Path) -> list[dict[str, Any]]:
    layout = RunLayout.for_run(stage_root, RUN_ID)
    attempt_path = layout.events / "call_attempts.jsonl"
    rows = [dict(row) for row in read_jsonl(attempt_path)]
    index = {str(row["event_id"]): row for row in rows}
    normalized: list[dict[str, Any]] = []

    with _open_ledger(layout.ledger, read_only=False) as ledger:
        repair = GENERIC_LEDGER_REPAIR
        cursor = ledger.execute(
            "UPDATE reservations SET request_status=? "
            "WHERE reservation_id=? AND request_key=? AND request_status=?",
            (
                repair["to_request_status"],
                repair["reservation_id"],
                repair["request_key"],
                repair["from_request_status"],
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("generic staging ledger repair did not match one row")

        for recovery in RECOVERIES:
            event_ids: list[str] = []
            elapsed_ms = 0
            tokens = 0
            cost = Decimal("0")
            attempt_receipts: list[dict[str, Any]] = []
            for attempt in recovery.attempts:
                row = index.get(attempt.event_id)
                if row is None or sha256_json(row) != attempt.attempt_sha256:
                    raise ValueError(f"attempt changed during staging: {attempt.event_id}")
                before_sha = sha256_json(row)
                row["attempt_number"] = attempt.logical_attempt_number
                row["status"] = attempt.logical_status
                logical_key = (
                    f"{recovery.logical_request_key}/attempt-"
                    f"{attempt.logical_attempt_number}"
                )
                cursor = ledger.execute(
                    "UPDATE reservations SET request_key=?, request_status=? "
                    "WHERE reservation_id=? AND request_key=?",
                    (
                        logical_key,
                        attempt.logical_status,
                        attempt.reservation_id,
                        attempt.physical_request_key,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError(
                        f"staging ledger normalization missed {attempt.event_id}"
                    )
                usage = row["usage"]
                event_ids.append(attempt.event_id)
                elapsed_ms += int(row["elapsed_ms"])
                tokens += int(usage["input_tokens"]) + int(usage["output_tokens"])
                cost += Decimal(str(row["estimated_cost_usd"]))
                attempt_receipts.append(
                    {
                        "event_id": attempt.event_id,
                        "reservation_id": attempt.reservation_id,
                        "physical_request_key": attempt.physical_request_key,
                        "logical_request_key": logical_key,
                        "logical_attempt_number": attempt.logical_attempt_number,
                        "logical_status": attempt.logical_status,
                        "production_attempt_sha256": before_sha,
                        "staged_attempt_sha256": sha256_json(row),
                    }
                )
            if (
                tokens != recovery.expected_tokens
                or elapsed_ms != recovery.expected_elapsed_ms
                or cost != Decimal(recovery.expected_cost_usd)
            ):
                raise ValueError(f"recovery totals changed: {recovery.cell_id}")
            normalized.append(
                {
                    "cell_id": recovery.cell_id,
                    "source_unit_id": recovery.source_unit_id,
                    "checkpoint": recovery.checkpoint,
                    "logical_request_key": recovery.logical_request_key,
                    "call_event_ids": event_ids,
                    "physical_attempts": len(event_ids),
                    "tokens": tokens,
                    "elapsed_ms": elapsed_ms,
                    "actual_cost_usd": format(cost, "f"),
                    "attempts": attempt_receipts,
                }
            )
        ledger.commit()
    atomic_write_jsonl(attempt_path, rows)
    return normalized


def _rehash_record(record: dict[str, Any], field: str) -> None:
    core = {key: value for key, value in record.items() if key != field}
    record[field] = sha256_json(core)


def _patch_cell(stage_root: Path, recovery: RecoveryLock) -> dict[str, Any]:
    layout = RunLayout.for_run(stage_root, RUN_ID)
    event_path = layout.events / f"adaptive-{recovery.cell_id}.jsonl"
    output_path = layout.results / "adaptive_deployment" / f"{recovery.cell_id}.json"
    job_path = layout.results / "adaptive_deployment_jobs" / f"{recovery.cell_id}.json"
    events = [dict(row) for row in read_jsonl(event_path)]
    output = dict(read_json(output_path))
    job = dict(read_json(job_path))

    signal_matches = [
        row
        for row in events
        if row.get("event") == "signal_observed"
        and row.get("checkpoint") == recovery.checkpoint
    ]
    if len(signal_matches) != 1:
        raise ValueError(f"staging signal is not unique: {recovery.cell_id}")
    signal = signal_matches[0]
    final = recovery.attempts[-1]
    call = signal.get("call")
    if (
        not isinstance(call, dict)
        or call.get("call_event_ids") != [final.event_id]
        or call.get("elapsed_ms") is None
    ):
        raise ValueError(f"production final call binding changed: {recovery.cell_id}")
    old_signal_sha = str(signal["signal_record_sha256"])
    call["call_event_ids"] = [attempt.event_id for attempt in recovery.attempts]
    call["elapsed_ms"] = recovery.expected_elapsed_ms
    _rehash_record(signal, "signal_record_sha256")
    new_signal_sha = str(signal["signal_record_sha256"])

    decisions = [
        row
        for row in events
        if row.get("event") == "adaptive_decision"
        and row.get("checkpoint") == recovery.checkpoint
    ]
    if len(decisions) != 1 or decisions[0].get("signal_record_sha256") != old_signal_sha:
        raise ValueError(f"production decision binding changed: {recovery.cell_id}")
    decision = decisions[0]
    old_decision_sha = str(decision["decision_sha256"])
    decision["signal_record_sha256"] = new_signal_sha
    _rehash_record(decision, "decision_sha256")
    new_decision_sha = str(decision["decision_sha256"])

    for event in events:
        if (
            event.get("event") == "intervention_applied"
            and event.get("checkpoint") == recovery.checkpoint
        ):
            if (
                event.get("signal_record_sha256") != old_signal_sha
                or event.get("decision_sha256") != old_decision_sha
            ):
                raise ValueError(f"production intervention binding changed: {recovery.cell_id}")
            event["signal_record_sha256"] = new_signal_sha
            event["decision_sha256"] = new_decision_sha

    output["task_records"] = [
        row for row in events[:-1] if row.get("event") == "task_turn"
    ]
    output["signal_records"] = [
        row for row in events[:-1] if row.get("event") == "signal_observed"
    ]
    output["decision_records"] = [
        row for row in events[:-1] if row.get("event") == "adaptive_decision"
    ]
    output["intervention_records"] = [
        row for row in events[:-1] if row.get("event") == "intervention_applied"
    ]
    output["accounting"] = _accounting(
        [*output["task_records"], *output["signal_records"]]
    )
    output["event_log_prefix_sha256"] = sha256_json(events[:-1])
    output_sha = atomic_write_json(output_path, output)
    if events[-1].get("event") != "complete":
        raise ValueError(f"production completion event changed: {recovery.cell_id}")
    events[-1]["output_sha256"] = output_sha
    event_sha = atomic_write_jsonl(event_path, events)
    job["output_sha256"] = output_sha
    job["accounting_sha256"] = sha256_json(output["accounting"])
    job_sha = atomic_write_json(job_path, job)
    return {
        "cell_id": recovery.cell_id,
        "staged_event_log_sha256": event_sha,
        "staged_output_sha256": output_sha,
        "staged_job_sha256": job_sha,
        "staged_accounting_sha256": job["accounting_sha256"],
        "old_signal_record_sha256": old_signal_sha,
        "staged_signal_record_sha256": new_signal_sha,
        "old_decision_sha256": old_decision_sha,
        "staged_decision_sha256": new_decision_sha,
    }


def _validate_existing_stage() -> dict[str, Any] | None:
    if not STAGE_ARTIFACTS.exists():
        return None
    if not BUILD_RECEIPT.is_file():
        raise ValueError("staging destination exists without its build receipt")
    receipt = read_json(BUILD_RECEIPT)
    if (
        receipt.get("artifact_type")
        != "experiment12_adaptive_analysis_copy_on_write_staging"
        or receipt.get("source_manifest_sha256") != MANIFEST_SHA256
        or receipt.get("source_pair_manifest_sha256") != PAIR_SHA256
        or receipt.get("normalized_recovery_cell_count") != 3
        or receipt.get("normalized_source_unit_count") != 2
        or sha256_file(STAGE_LAYOUT.manifest) != MANIFEST_SHA256
        or sha256_file(STAGE_LAYOUT.pairs) != PAIR_SHA256
        or sha256_file(STAGE_LAYOUT.events / "call_attempts.jsonl")
        != receipt.get("staged_call_attempts_sha256")
        or sha256_file(STAGE_LAYOUT.ledger) != receipt.get("staged_ledger_sha256")
    ):
        raise ValueError("existing staging receipt or artifacts changed")
    return dict(receipt)


def build() -> dict[str, Any]:
    existing = _validate_existing_stage()
    if existing is not None:
        return existing
    gate = preflight()
    STAGE_ARTIFACTS.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".adaptive-analysis-staging-v1.", dir=STAGE_ARTIFACTS.parent
        )
    )
    try:
        shutil.copytree(SOURCE_LAYOUT.root, temporary / RUN_ID)
        _copy_ledger(SOURCE_LAYOUT.ledger, temporary / "_global_budget.sqlite3")
        normalized = _patch_attempts_and_ledger(temporary)
        staged_cells = [_patch_cell(temporary, recovery) for recovery in RECOVERIES]
        temporary_layout = RunLayout.for_run(temporary, RUN_ID)
        receipt = {
            "artifact_type": "experiment12_adaptive_analysis_copy_on_write_staging",
            "staging_version": 1,
            "source_run_id": RUN_ID,
            "source_artifacts_root": _relative(SOURCE_ARTIFACTS),
            "staging_artifacts_root": _relative(STAGE_ARTIFACTS),
            "source_manifest_sha256": MANIFEST_SHA256,
            "source_pair_manifest_sha256": PAIR_SHA256,
            "source_code_tree_sha256": CODE_TREE_SHA256,
            "declared_cells": EXPECTED_CELLS,
            "production_inputs_immutable": True,
            "provider_calls": 0,
            "raw_analyzer_expected_failure": gate["raw_analyzer_expected_failure"],
            "generic_ledger_repairs": [dict(GENERIC_LEDGER_REPAIR)],
            "normalization_semantics": (
                "provider-successful but semantically malformed judge outputs are "
                "represented as failed logical attempts followed by the successful "
                "semantic final; usage, elapsed time, and cost of every physical "
                "attempt remain included"
            ),
            "normalized_recovery_cell_count": len(RECOVERIES),
            "normalized_source_units": sorted(
                {recovery.source_unit_id for recovery in RECOVERIES}
            ),
            "normalized_source_unit_count": len(
                {recovery.source_unit_id for recovery in RECOVERIES}
            ),
            "recovery_receipts": [
                {
                    "cell_id": recovery.cell_id,
                    "path": recovery.receipt_path,
                    "sha256": recovery.receipt_sha256,
                    "artifact_type": recovery.receipt_artifact_type,
                }
                for recovery in RECOVERIES
            ],
            "normalized_calls": normalized,
            "staged_cell_bindings": staged_cells,
            "staged_call_attempts_path": _relative(
                temporary_layout.events / "call_attempts.jsonl"
            ).replace(temporary.name, STAGE_ARTIFACTS.name, 1),
            "staged_call_attempts_sha256": sha256_file(
                temporary_layout.events / "call_attempts.jsonl"
            ),
            "staged_ledger_path": _relative(
                temporary_layout.ledger
            ).replace(temporary.name, STAGE_ARTIFACTS.name, 1),
            "staged_ledger_sha256": sha256_file(temporary_layout.ledger),
        }
        atomic_write_json(temporary / BUILD_RECEIPT.name, receipt)
        temporary.rename(STAGE_ARTIFACTS)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    for recovery in RECOVERIES:
        _production_bindings(recovery)
    return dict(read_json(BUILD_RECEIPT))


def _sensitivity_markdown(sensitivity: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# Adaptive analysis: leave-two-source-units sensitivity",
            "",
            "This paired sensitivity removes both source units implicated in the three semantic judge recoveries from every method × operator treatment.",
            "",
            f"- Excluded source units: {', '.join(sensitivity['excluded_source_units'])}",
            f"- Excluded rows: {sensitivity['excluded_rows']}",
            f"- Remaining rows: {sensitivity['remaining_rows']}",
            f"- Remaining source tasks per treatment: {sensitivity['remaining_source_tasks_per_treatment']}",
            f"- Treatments: {sensitivity['treatments']}",
            "- Pairing after exclusion: balanced",
            "",
        )
    )


def _build_sensitivity(analysis: Mapping[str, Any], analysis_sha: str) -> dict[str, Any]:
    rows = analysis.get("rows")
    if not isinstance(rows, (list, tuple)) or len(rows) != EXPECTED_CELLS:
        raise ValueError("stock staged analysis lacks exactly 1,120 rows")
    excluded_units = sorted({recovery.source_unit_id for recovery in RECOVERIES})
    excluded = [row for row in rows if row.get("unit_id") in excluded_units]
    remaining = [row for row in rows if row.get("unit_id") not in excluded_units]
    treatments = {
        (str(row["method"]), str(row["operator"])) for row in rows
    }
    excluded_treatments: dict[tuple[str, str], int] = {}
    remaining_treatments: dict[tuple[str, str], int] = {}
    for row in excluded:
        key = (str(row["method"]), str(row["operator"]))
        excluded_treatments[key] = excluded_treatments.get(key, 0) + 1
    for row in remaining:
        key = (str(row["method"]), str(row["operator"]))
        remaining_treatments[key] = remaining_treatments.get(key, 0) + 1
    if (
        len(treatments) != EXPECTED_TREATMENTS
        or len(excluded) != EXPECTED_TREATMENTS * 2
        or set(excluded_treatments) != treatments
        or set(excluded_treatments.values()) != {2}
        or len(remaining) != EXPECTED_TREATMENTS * EXPECTED_SENSITIVITY_N
        or set(remaining_treatments) != treatments
        or set(remaining_treatments.values()) != {EXPECTED_SENSITIVITY_N}
    ):
        raise ValueError("leave-two exclusion is not the exact balanced paired design")
    summaries, effects = summarize_adaptive_outcomes(remaining)
    sensitivity = {
        "artifact_type": "experiment12_online_adaptive_leave_two_source_units_sensitivity",
        "sensitivity_version": 1,
        "source_run_id": RUN_ID,
        "source_analysis_path": _relative(ANALYSIS_PATH),
        "source_analysis_sha256": analysis_sha,
        "exclusion_reason": "three cells required semantic judge-attempt recovery",
        "excluded_source_units": excluded_units,
        "exclusion_scope": "both_source_units_from_every_method_operator_treatment",
        "treatments": EXPECTED_TREATMENTS,
        "excluded_rows_per_treatment": 2,
        "excluded_rows": len(excluded),
        "remaining_rows": len(remaining),
        "remaining_source_tasks_per_treatment": EXPECTED_SENSITIVITY_N,
        "balanced_paired_design_after_exclusion": True,
        "excluded_cell_ids": sorted(str(row["cell_id"]) for row in excluded),
        "rows": remaining,
        "metric_summaries": [asdict(row) for row in summaries],
        "operator_effects": [asdict(row) for row in effects],
    }
    atomic_write_json(SENSITIVITY_PATH, sensitivity)
    atomic_write_text(SENSITIVITY_MD_PATH, _sensitivity_markdown(sensitivity))
    return sensitivity


def analyze() -> dict[str, Any]:
    build_receipt = _validate_existing_stage()
    if build_receipt is None:
        raise ValueError("staging view has not been built")
    analysis = extract_adaptive_run(
        STAGE_LAYOUT,
        expected_manifest_sha256=MANIFEST_SHA256,
    )
    if (
        analysis.get("artifact_type") != ADAPTIVE_ANALYSIS_TYPE
        or len(analysis.get("rows", ())) != EXPECTED_CELLS
        or {
            int(row["n_tasks"])
            for row in analysis.get("metric_summaries", ())
        }
        != {EXPECTED_PRIMARY_N}
    ):
        raise ValueError("stock staged analysis has an unexpected denominator")
    analysis_sha = atomic_write_json(ANALYSIS_PATH, analysis)
    figures = write_adaptive_figures(analysis, FIGURE_DIR)
    sensitivity = _build_sensitivity(analysis, analysis_sha)
    sensitivity_sha = sha256_file(SENSITIVITY_PATH)
    sensitivity_md_sha = sha256_file(SENSITIVITY_MD_PATH)
    receipt = {
        "artifact_type": "experiment12_adaptive_analysis_staging_receipt",
        "analysis_receipt_version": 1,
        "source_run_id": RUN_ID,
        "source_manifest_sha256": MANIFEST_SHA256,
        "source_pair_manifest_sha256": PAIR_SHA256,
        "staging_receipt_path": _relative(BUILD_RECEIPT),
        "staging_receipt_sha256": sha256_file(BUILD_RECEIPT),
        "analysis_path": _relative(ANALYSIS_PATH),
        "analysis_sha256": analysis_sha,
        "analysis_rows": EXPECTED_CELLS,
        "primary_source_tasks_per_treatment": EXPECTED_PRIMARY_N,
        "treatments": EXPECTED_TREATMENTS,
        "normalized_recovery_cell_count": 3,
        "normalized_source_unit_count": 2,
        "resource_semantics": analysis["resource_semantics"],
        "sensitivity": {
            "path": _relative(SENSITIVITY_PATH),
            "sha256": sensitivity_sha,
            "markdown_path": _relative(SENSITIVITY_MD_PATH),
            "markdown_sha256": sensitivity_md_sha,
            "excluded_source_units": sensitivity["excluded_source_units"],
            "excluded_rows": sensitivity["excluded_rows"],
            "remaining_rows": sensitivity["remaining_rows"],
            "remaining_source_tasks_per_treatment": sensitivity[
                "remaining_source_tasks_per_treatment"
            ],
        },
        "figures": [
            {"path": _relative(path), "sha256": sha256_file(path)}
            for path in figures
        ],
        "provider_calls": 0,
    }
    atomic_write_json(ANALYSIS_RECEIPT, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("command", choices=("preflight", "build", "analyze"))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = {
            "preflight": preflight,
            "build": build,
            "analyze": analyze,
        }[args.command]()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (
        AnalysisInputError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
