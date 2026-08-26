"""Reviewed copy-on-write normalization and analysis for the final online run.

This is intentionally separate from the quarantined, unattributed staging-v1
builder and directory.  It never edits raw events, jobs, outputs, or the global
ledger.  It works on a fresh copied run and publishes only derived analysis
products after every raw-input hash is rechecked.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from experiments12.adaptive_analysis12 import _adaptive_replay_inputs, _attempt_totals
from experiments12.adaptive_deployment12 import (
    ADAPTIVE_JOB_SUBDIR,
    ADAPTIVE_RESULT_SUBDIR,
    _accounting,
    _decision_record,
    _validate_existing,
)
from experiments12.analysis12 import _load_attempt_resources
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.deployment12 import LockedMethodThreshold
from experiments12.manifest12 import RunLayout, code_tree_hash
from experiments12.models12 import JUDGE_MODEL_NAME
from experiments12.pairing12 import JobCell


RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
MANIFEST_SHA256 = "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
PAIRS_SHA256 = "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
CODE_TREE_SHA256 = "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
EXPECTED_CELLS = 1_120
SCHEMA_VERSION = 3
STAGING_RECEIPT_TYPE = "experiment12_adaptive_semantic_retry_normalization"
ANALYSIS_RECEIPT_TYPE = "experiment12_staged_stock_adaptive_analysis_receipt"
CASE_INDEX_TYPE = "experiment12_adaptive_normalization_case_index"

SOURCE_BASE = REPOSITORY_ROOT / "experiments12" / "data_results" / "runs"
SOURCE = RunLayout.for_run(SOURCE_BASE, RUN_ID)
STAGE_BASE = (
    REPOSITORY_ROOT
    / "experiments12"
    / "data_results" / "derived"
    / "adaptive-analysis-staging-reviewed-v1"
)
STAGE = RunLayout.for_run(STAGE_BASE, RUN_ID)
STAGING_RECEIPT = STAGE_BASE / "staging-receipt.json"
ANALYSIS_RECEIPT = STAGE_BASE / "analysis-receipt.json"
CASE_INDEX = STAGE_BASE / "normalization-cases.json"
ANALYSIS_DIR = STAGE_BASE / "analysis"
ANALYSIS_PATH = ANALYSIS_DIR / "adaptive-analysis.json"
FIGURE_DIR = ANALYSIS_DIR / "figures"
SENSITIVITY_PATH = ANALYSIS_DIR / "adaptive-analysis-leave-two-units.json"
SENSITIVITY_MD_PATH = ANALYSIS_DIR / "adaptive-analysis-leave-two-units.md"

PRODUCTION_ANALYSIS = SOURCE.results / "adaptive-analysis.json"
PRODUCTION_FIGURES = SOURCE.results / "adaptive-figures"
PRODUCTION_SENSITIVITY = SOURCE.results / "adaptive-analysis-leave-two-units.json"
PRODUCTION_SENSITIVITY_MD = SOURCE.results / "adaptive-analysis-leave-two-units.md"

RAW_ANALYZER_TEXT = (
    "error: call attempt disagrees with ledger: "
    "d950af6bd8a8421e99f8efc17125fa1b\n"
)


class ReviewedStagingError(ValueError):
    """Raised when a frozen receipt, copy, patch, or audit differs."""


@dataclass(frozen=True, slots=True)
class AttemptLock:
    event_id: str
    attempt_sha256: str
    reservation_id: str
    physical_request_key: str
    logical_attempt_number: int
    logical_status: str


@dataclass(frozen=True, slots=True)
class GroupLock:
    checkpoint: int
    logical_request_key: str
    max_output_tokens: int
    normalize: bool
    attempts: tuple[AttemptLock, ...]


@dataclass(frozen=True, slots=True)
class RecoveryLock:
    cell_id: str
    task_id: str
    operator: str
    receipt_name: str
    receipt_sha256: str
    receipt_type: str
    event_sha256: str
    output_sha256: str
    job_sha256: str
    case_name: str | None
    case_sha256: str | None
    forensic_name: str | None
    forensic_sha256: str | None
    groups: tuple[GroupLock, ...]


def _attempt(
    event_id: str,
    digest: str,
    reservation: str,
    key: str,
    number: int,
    status: str,
) -> AttemptLock:
    return AttemptLock(event_id, digest, reservation, key, number, status)


RECOVERIES: tuple[RecoveryLock, ...] = (
    RecoveryLock(
        cell_id="d52046b6eb74a76ecdc3debc",
        task_id="extracted-gsm8k-test-814::t7",
        operator="lossy_compaction",
        receipt_name="recovery-adaptive-d52046b6eb74a76ecdc3debc12.json",
        receipt_sha256="83f8939e08e7809d699e51e62a13b68aad838018669d4792ab6e84645741eca1",
        receipt_type="experiment12_online_adaptive_single_cell_recovery",
        event_sha256="ea4c354f34828a907a09e4496a84d02d33b2de023afab5ccde39c6e6152e8f75",
        output_sha256="2228470d0a4000c716b293ea026d43dc73025f9625275b2e0438922b6f83aa82",
        job_sha256="b107983e334ce559daafa4cbd21885f3ecd1897716a76b6db06b9e36487d4c96",
        case_name=None,
        case_sha256=None,
        forensic_name=None,
        forensic_sha256=None,
        groups=(
            GroupLock(
                5,
                f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5",
                320,
                True,
                (
                    _attempt(
                        "6c1a460c1c704a4485f0957808a17e6b",
                        "05cc4a63bc0d53e46c99f5d1cdff83c83dbbc93b8d36497d44a3a5bca377d439",
                        "0a22c9a145784a9ebb45747f6758aa6c",
                        f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5/attempt-1",
                        1,
                        "failed",
                    ),
                    _attempt(
                        "4e421d9936504860ad1afd61a58a5788",
                        "395a31ac1a6126fd4ae624d0e4374701dd0e0305de38470b1755cb2db8e207bb",
                        "305dfec3f9b64f68aef2fc455393e32a",
                        f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5-recovery-semantic-1/attempt-1",
                        2,
                        "succeeded",
                    ),
                ),
            ),
        ),
    ),
    RecoveryLock(
        cell_id="89df41e0daa1262a43fa5e55",
        task_id="extracted-gsm8k-test-814::t7",
        operator="public_state_reground",
        receipt_name="recovery-adaptive-89df41e0daa1262a43fa5e5512.json",
        receipt_sha256="7fdfe614fe976db85343586e4908785aa90ffe734045e629db9af5b46249329e",
        receipt_type="experiment12_online_adaptive_trace_judge_recovery",
        event_sha256="185d167b92a5fdc473f416a953a687134d8fc60926d3e4c45be504f4ab8e1b8d",
        output_sha256="4f4b8ee8dcc3074a41d6844de5bf824ec868b26e2394687dc7975ef770e31210",
        job_sha256="50423cdb86b68cb307b168b4a8bd1c98c1d462240bf826557064a8d2ab57b660",
        case_name="recovery-case-adaptive-89df41e0daa1262a43fa5e55.json",
        case_sha256="261bfc54dc5faa3f4d10abbac04d573cd0a5f90123ed7b66f7ba974e67874e03",
        forensic_name="forensic-audit-adaptive-89df41e0daa1262a43fa5e5512.json",
        forensic_sha256="c153f1f5643cdfe574c67d941627da37e2f8ebdcc1dfa982015f073d288c6bf4",
        groups=(
            GroupLock(
                6,
                f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6",
                640,
                True,
                (
                    _attempt("dbfbaf64ae304165b85c33c030ed6729", "91ef59a700814342ff9949c41ff55ed71a212c123d2596774611666378110c82", "6e20aea9b36b4b0886bed2534c4cd395", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6/attempt-1", 1, "failed"),
                    _attempt("11257c27688948aeb07a6397e0017f65", "1f4ef3e504f9f30ae19b04714b0898a13de0acac3440d9487d1bac9291646fa1", "4d9aa7eba85040beb7a5d70ca787a80a", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-1/attempt-1", 2, "failed"),
                    _attempt("af9e3a5c95d44bb1aee521449e4886e9", "5ab0059228c61f8c3eb3f495a884123621c18b5e3ef3ad10dd5f86e179b43541", "3c7bf5b0a12840b19f27e8bf28c35d59", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-2/attempt-1", 3, "failed"),
                    _attempt("08b9bae667e0495ea9bf9c06e16fc699", "a2324955b730a13909aad3b7ec0782b5afa46a1869dd40d04166eb5b29ded43b", "9335916349834914a4a80b90382d3373", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-3/attempt-1", 4, "failed"),
                    _attempt("b1608d3c879d4f77a6adca6c05cd1fc8", "249c362cb308b1c6c596e5a9e3a0650241d10035cda1c8b6f30ce8f68f42d85b", "30535c6615204ebcba52e21327b88f80", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-final-cap640-v1/attempt-1", 5, "succeeded"),
                ),
            ),
        ),
    ),
    RecoveryLock(
        cell_id="786d95760ccdb86713c26936",
        task_id="extracted-gsm8k-test-989::t7",
        operator="public_state_reground",
        receipt_name="recovery-adaptive-786d95760ccdb86713c2693612.json",
        receipt_sha256="0110cc242d6ffdec0c4fd1b1e45a606b5b7bee141a1d35a28fdc16f11d056509",
        receipt_type="experiment12_online_adaptive_trace_judge_recovery",
        event_sha256="7d434654b547953521a8735ddd7a51bfa1d496c997e2bf3c544d0d3cbb85c064",
        output_sha256="f73c38ac567f04b5fe2863137eacf90263ebbe5fedef1217c4530ad5b1335ca9",
        job_sha256="449f2c8fdc3bca52cdb3827a811a5529586cb41f0fba166aaaae9d7298d54cc3",
        case_name="recovery-case-adaptive-786d95760ccdb86713c26936.json",
        case_sha256="e3ab7b317a0e3166712140d97b6475643638dd321c6eaa572d98a54c4757d375",
        forensic_name="forensic-audit-adaptive-786d95760ccdb86713c2693612.json",
        forensic_sha256="eae3bbc7e2ae7ff77402aa441ef55d275baf8b8a8e1e0a6ba1561de8d8b2ce66",
        groups=(
            GroupLock(
                5,
                f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5",
                640,
                True,
                (
                    _attempt("af34404456b84d27833193218a878154", "17d52ef53f50a034a8feb55095a3f6031e336be232d60b59c6e722ac0c32ba34", "3ab51cba528c46e6baa00f36a782defc", f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5/attempt-1", 1, "failed"),
                    _attempt("50045c026c3b44b8a4e38c5aab0509d9", "945ac143e66ec180fbd0496eb2f1711464aadb207ed524952f998e6d545ab38f", "e76435343cad49499f991e142b8871e3", f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5-recovery-final-cap640-v1/attempt-1", 2, "succeeded"),
                ),
            ),
            GroupLock(
                6,
                f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-6",
                640,
                False,
                (
                    _attempt("1a3bdac012414546827ecd5002dd7ee1", "d2238d802bdc431090f1178ac7dc196dcf7227a051fc285ed0cf2e9053b0bdc1", "7f9779e9afa44eaabfc28a856386fea1", f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-6/attempt-1", 1, "succeeded"),
                ),
            ),
        ),
    ),
)


ORDINARY_REPAIR = {
    "event_id": "d950af6bd8a8421e99f8efc17125fa1b",
    "attempt_sha256": "18a6e53c4387b900cf23e9ce9ff3a24b733cf5a396cdc3f1f93acf904516f856",
    "reservation_id": "abf82d1b70f9480db3c05659062e0a0b",
    "ledger_row_sha256": "21897bb95498fc9b1653d9d9a8a210ae5990b3aeae885e68b9353babbef5f994",
    "request_key": f"{RUN_ID}/b0978e4007c1e796c0521807/adaptive-task-7/attempt-1",
    "before": "unknown",
    "after": "failed",
    "transport_error": "http_503",
    "cost_quality": "upper_bound",
    "actual_micro_usd": 2578,
}


LEDGER_COLUMNS = (
    "reservation_id", "provider", "purpose", "request_key", "state",
    "reserved_micro_usd", "actual_micro_usd", "cost_quality", "request_status",
    "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens",
    "provider_total_tokens", "provider_request_id", "created_at", "updated_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReviewedStagingError(f"required regular file is absent or linked: {path}")
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _attempts(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(path):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("event_id"), str):
            raise ReviewedStagingError("invalid call-attempt row")
        event_id = str(raw["event_id"])
        if event_id in result:
            raise ReviewedStagingError(f"duplicate call-attempt event: {event_id}")
        result[event_id] = dict(raw)
    return result


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT " + ", ".join(LEDGER_COLUMNS)
            + " FROM reservations WHERE request_key LIKE ? ORDER BY request_key, reservation_id",
            (RUN_ID + "/%",),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _ledger_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = _ledger_rows(path)
    result = {str(row["reservation_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ReviewedStagingError("duplicate ledger reservation")
    return result


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ReviewedStagingError(f"invalid inventory root: {root}")
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReviewedStagingError(f"inventory contains symlink: {path}")
        if path.is_file():
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return result


def _inventory(layout: RunLayout, *, ledger_path: Path | None = None) -> dict[str, Any]:
    files = _file_inventory(layout.root)
    ledger_rows = _ledger_rows(layout.ledger if ledger_path is None else ledger_path)
    return {
        "files": files,
        "files_sha256": sha256_json(files),
        "ledger_rows": ledger_rows,
        "ledger_rows_sha256": sha256_json(ledger_rows),
        "inventory_sha256": sha256_json({"files": files, "ledger_rows": ledger_rows}),
    }


def _raw_inventory(layout: RunLayout) -> dict[str, Any]:
    derived = {
        "results/adaptive-analysis.json",
        "results/adaptive-analysis-leave-two-units.json",
        "results/adaptive-analysis-leave-two-units.md",
    }
    files = [
        row for row in _file_inventory(layout.root)
        if row["path"] not in derived
        and not row["path"].startswith("results/adaptive-figures/")
    ]
    ledger_rows = _ledger_rows(layout.ledger)
    return {
        "files": files,
        "files_sha256": sha256_json(files),
        "ledger_rows": ledger_rows,
        "ledger_rows_sha256": sha256_json(ledger_rows),
        "inventory_sha256": sha256_json({"files": files, "ledger_rows": ledger_rows}),
    }
