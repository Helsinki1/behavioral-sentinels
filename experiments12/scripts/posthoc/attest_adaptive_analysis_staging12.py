"""Provider-free post-hoc attestation of the audited Experiment 12 staging view.

This utility never dispatches a model and never edits production raw inputs or
the provider ledger.  It binds the existing copy-on-write staging view to its
production source, emits the canonical normalization-case index, runs the
external leave-two-source-units sensitivity, and publishes only derived paper
analysis products.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


from experiments12.core.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.manifest12 import code_tree_hash


RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
MANIFEST_SHA = "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
PAIR_SHA = "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
CODE_SHA = "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
EXPECTED_CELLS = 1_120
EXPECTED_ATTEMPTS = 11_687

SOURCE_BASE = REPO / "experiments12" / "data_results" / "runs"
SOURCE_RUN = SOURCE_BASE / RUN_ID
SOURCE_LEDGER = SOURCE_BASE / "_global_budget.sqlite3"
STAGE_BASE = REPO / "experiments12" / "data_results" / "derived" / "adaptive-analysis-staging-v1"
STAGE_RUN = STAGE_BASE / RUN_ID
STAGE_LEDGER = STAGE_BASE / "_global_budget.sqlite3"
ATTEST_BASE = REPO / "experiments12" / "data_results" / "derived" / "adaptive-analysis-attested-v1"
CASE_INDEX = ATTEST_BASE / "normalization-cases.json"
STAGING_RECEIPT = ATTEST_BASE / "staging-receipt.json"
ANALYSIS_RECEIPT = ATTEST_BASE / "analysis-receipt.json"
ATTEST_ANALYSIS = ATTEST_BASE / "analysis" / "adaptive-analysis.json"
SENSITIVITY = ATTEST_BASE / "analysis" / "adaptive-analysis-leave-two-units.json"
SENSITIVITY_MD = ATTEST_BASE / "analysis" / "adaptive-analysis-leave-two-units.md"

STAGE_RECEIPT = STAGE_BASE / "staging-receipt.json"
STAGE_ANALYSIS_RECEIPT = STAGE_BASE / "analysis-receipt.json"
STAGE_ANALYSIS = STAGE_BASE / "analysis" / "adaptive-analysis.json"
STAGE_FIGURES = STAGE_BASE / "analysis" / "figures"

PROD_ANALYSIS = SOURCE_RUN / "results" / "adaptive-analysis.json"
PROD_FIGURES = SOURCE_RUN / "results" / "adaptive-figures"
PROD_SENSITIVITY = SOURCE_RUN / "results" / "adaptive-analysis-leave-two-units.json"
PROD_SENSITIVITY_MD = SOURCE_RUN / "results" / "adaptive-analysis-leave-two-units.md"

SENSITIVITY_SCRIPT = (
    REPO / "experiments12" / "scripts" / "posthoc" / "online_leave_one_unit_sensitivity12.py"
)

LOCKED_FILES = {
    "existing_staging_receipt": (STAGE_RECEIPT, "28fbf8b4e14449087c0444bae6522bec42624b7b93380c92c70d12e44bc42f15"),
    "existing_analysis_receipt": (STAGE_ANALYSIS_RECEIPT, "dfc03904181b6dc5f48d2ed691cb0889a3d7a03518b583fac59d747e32e0cf65"),
    "staged_analysis": (STAGE_ANALYSIS, "c296291f61b1e0134cac1f68f0d94b2f46286f0710354ed0284c2a454db98b9e"),
    "staged_call_attempts": (STAGE_RUN / "events" / "call_attempts.jsonl", "d5bc26366c75bd7612e4f29b3be62e81ee576d0ec898f52fc4cf87eb480f73d7"),
    "staged_ledger": (STAGE_LEDGER, "5f7c273266bcdec7c4dda7585518353d21e2f8545e4699435be9d35f56cc1e9c"),
    "figure_data": (STAGE_FIGURES / "deployment-evolving_intent_gsm8k-gpt-5.6-luna.data.json", "0d650118f6ee79d6b0eccd4b84dded0e46c737892ceb9697adec1cae20f9feec"),
    "figure_svg": (STAGE_FIGURES / "deployment-evolving_intent_gsm8k-gpt-5.6-luna.svg", "6b0f616a1622b9361c8e31751d4d4b53bda4915c9d5214093516d560c8d26513"),
}

ORDINARY = {
    "event_id": "d950af6bd8a8421e99f8efc17125fa1b",
    "attempt_sha256": "18a6e53c4387b900cf23e9ce9ff3a24b733cf5a396cdc3f1f93acf904516f856",
    "reservation_id": "abf82d1b70f9480db3c05659062e0a0b",
    "source_ledger_row_sha256": "21897bb95498fc9b1653d9d9a8a210ae5990b3aeae885e68b9353babbef5f994",
    "request_key": f"{RUN_ID}/b0978e4007c1e796c0521807/adaptive-task-7/attempt-1",
}


def _a(event: str, digest: str, reservation: str, physical: str, number: int, status: str) -> dict[str, Any]:
    return {
        "event_id": event,
        "production_attempt_sha256": digest,
        "reservation_id": reservation,
        "physical_request_key": physical,
        "logical_attempt_number": number,
        "logical_status": status,
    }


RECOVERIES: tuple[dict[str, Any], ...] = (
    {
        "cell_id": "d52046b6eb74a76ecdc3debc",
        "source_task_id": "extracted-gsm8k-test-814::t7",
        "operator": "lossy_compaction",
        "production": {
            "event": "ea4c354f34828a907a09e4496a84d02d33b2de023afab5ccde39c6e6152e8f75",
            "output": "2228470d0a4000c716b293ea026d43dc73025f9625275b2e0438922b6f83aa82",
            "job": "b107983e334ce559daafa4cbd21885f3ecd1897716a76b6db06b9e36487d4c96",
        },
        "staged": {
            "event": "ff08ab7f4f80c666167de3d53a1f3a3b880b38ccfd98c48a20951a5fd962e524",
            "output": "ea82c239d2b79aaac8f13dfff5a71bb3f59e1fd7e5059fafa093124651502fa8",
            "job": "d8d50ae8bed49b961f6231b67a5d5e7e9c6c63b8badeaab0d311afd5884759d4",
            "signal": "9b077f305695afadf51b44fcf4554269d6a1cae0d8d119ae87f3be4fe427af9a",
            "decision": "75473ed00dfdc15f778728f48163e2981dcf6604c813dead44ebf2c7175d0411",
            "accounting": "8a3e0e64efffd8af94392daef845e61143299bac1d597e35d080c73e9a6bf681",
        },
        "receipt": ("experiments12/data_results/derived/recovery-adaptive-d52046b6eb74a76ecdc3debc12.json", "83f8939e08e7809d699e51e62a13b68aad838018669d4792ab6e84645741eca1"),
        "groups": ({
            "checkpoint": 5,
            "logical_request_key": f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5",
            "attempts": (
                _a("6c1a460c1c704a4485f0957808a17e6b", "05cc4a63bc0d53e46c99f5d1cdff83c83dbbc93b8d36497d44a3a5bca377d439", "0a22c9a145784a9ebb45747f6758aa6c", f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5/attempt-1", 1, "failed"),
                _a("4e421d9936504860ad1afd61a58a5788", "395a31ac1a6126fd4ae624d0e4374701dd0e0305de38470b1755cb2db8e207bb", "305dfec3f9b64f68aef2fc455393e32a", f"{RUN_ID}/d52046b6eb74a76ecdc3debc/adaptive-trace-judge-5-recovery-semantic-1/attempt-1", 2, "succeeded"),
            ),
        },),
    },
    {
        "cell_id": "89df41e0daa1262a43fa5e55",
        "source_task_id": "extracted-gsm8k-test-814::t7",
        "operator": "public_state_reground",
        "production": {"event": "185d167b92a5fdc473f416a953a687134d8fc60926d3e4c45be504f4ab8e1b8d", "output": "4f4b8ee8dcc3074a41d6844de5bf824ec868b26e2394687dc7975ef770e31210", "job": "50423cdb86b68cb307b168b4a8bd1c98c1d462240bf826557064a8d2ab57b660"},
        "staged": {"event": "8d6e94f3ecec6c16e1b4ad3b3b72789e11cfdc42d092a0832eaf748246fa3e16", "output": "671950655ef390189780206f74c8ff5204b27a0cb1a29b85cca8e5b86cbe2ad4", "job": "cc70ee380be3cdf96ae0e0bbe310c7417f9252bf433c8a2e440d14de21b1c808", "signal": "b638c4efaa8fb8dcaa2d111807e075d8d73e49801ab6fab4207bf5b5ba4b11ce", "decision": "216ef0f4cc166f650f673080a9710c902788cc13393bb08d98631a5592488797", "accounting": "ed817f340f3eea591a889c8c65dace68a7ae44c384879da987518c41629a8c5c"},
        "receipt": ("experiments12/data_results/derived/recovery-adaptive-89df41e0daa1262a43fa5e5512.json", "7fdfe614fe976db85343586e4908785aa90ffe734045e629db9af5b46249329e"),
        "groups": ({
            "checkpoint": 6,
            "logical_request_key": f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6",
            "attempts": (
                _a("dbfbaf64ae304165b85c33c030ed6729", "91ef59a700814342ff9949c41ff55ed71a212c123d2596774611666378110c82", "6e20aea9b36b4b0886bed2534c4cd395", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6/attempt-1", 1, "failed"),
                _a("11257c27688948aeb07a6397e0017f65", "1f4ef3e504f9f30ae19b04714b0898a13de0acac3440d9487d1bac9291646fa1", "4d9aa7eba85040beb7a5d70ca787a80a", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-1/attempt-1", 2, "failed"),
                _a("af9e3a5c95d44bb1aee521449e4886e9", "5ab0059228c61f8c3eb3f495a884123621c18b5e3ef3ad10dd5f86e179b43541", "3c7bf5b0a12840b19f27e8bf28c35d59", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-2/attempt-1", 3, "failed"),
                _a("08b9bae667e0495ea9bf9c06e16fc699", "a2324955b730a13909aad3b7ec0782b5afa46a1869dd40d04166eb5b29ded43b", "9335916349834914a4a80b90382d3373", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-semantic-3/attempt-1", 4, "failed"),
                _a("b1608d3c879d4f77a6adca6c05cd1fc8", "249c362cb308b1c6c596e5a9e3a0650241d10035cda1c8b6f30ce8f68f42d85b", "30535c6615204ebcba52e21327b88f80", f"{RUN_ID}/89df41e0daa1262a43fa5e55/adaptive-trace-judge-6-recovery-final-cap640-v1/attempt-1", 5, "succeeded"),
            ),
        },),
    },
    {
        "cell_id": "786d95760ccdb86713c26936",
        "source_task_id": "extracted-gsm8k-test-989::t7",
        "operator": "public_state_reground",
        "production": {"event": "7d434654b547953521a8735ddd7a51bfa1d496c997e2bf3c544d0d3cbb85c064", "output": "f73c38ac567f04b5fe2863137eacf90263ebbe5fedef1217c4530ad5b1335ca9", "job": "449f2c8fdc3bca52cdb3827a811a5529586cb41f0fba166aaaae9d7298d54cc3"},
        "staged": {"event": "3a897df0d197f2fc721dd7b01b880a30101ad7c1463a53aab5de4c3fa2afc452", "output": "4ae268d97ec37655e7da8c31aed7232a2ae829a45c86827ad6c1696bb35b99cb", "job": "fa4343f9efe0ea1985993d7aab9c7016367f031b35370871173864950ae0f1fb", "signal": "1c8eee7dc989b5c09e36e9588987917f6420c5a8fc6311b0f67ba7c0d69e9f18", "decision": "69653f3b427738264354d0d21ff6484e47d4e599d3f5eaef0f282a6b6edf951b", "accounting": "02497662b168b3b024cfae17579ffab62cf6c7871f270bd24a5fd09f9041d399"},
        "receipt": ("experiments12/data_results/derived/recovery-adaptive-786d95760ccdb86713c2693612.json", "0110cc242d6ffdec0c4fd1b1e45a606b5b7bee141a1d35a28fdc16f11d056509"),
        "groups": ({
            "checkpoint": 5,
            "logical_request_key": f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5",
            "attempts": (
                _a("af34404456b84d27833193218a878154", "17d52ef53f50a034a8feb55095a3f6031e336be232d60b59c6e722ac0c32ba34", "3ab51cba528c46e6baa00f36a782defc", f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5/attempt-1", 1, "failed"),
                _a("50045c026c3b44b8a4e38c5aab0509d9", "945ac143e66ec180fbd0496eb2f1711464aadb207ed524952f998e6d545ab38f", "e76435343cad49499f991e142b8871e3", f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-5-recovery-final-cap640-v1/attempt-1", 2, "succeeded"),
            ),
        }, {
            "checkpoint": 6,
            "logical_request_key": f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-6",
            "disclosure_only": True,
            "attempts": (
                _a("1a3bdac012414546827ecd5002dd7ee1", "d2238d802bdc431090f1178ac7dc196dcf7227a051fc285ed0cf2e9053b0bdc1", "7f9779e9afa44eaabfc28a856386fea1", f"{RUN_ID}/786d95760ccdb86713c26936/adaptive-trace-judge-6/attempt-1", 1, "succeeded"),
            ),
        }),
    },
)


LEDGER_COLUMNS = (
    "reservation_id", "provider", "purpose", "request_key", "state",
    "reserved_micro_usd", "actual_micro_usd", "cost_quality", "request_status",
    "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens",
    "provider_total_tokens", "provider_request_id", "created_at", "updated_at",
)


class AttestationError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def _file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AttestationError(f"missing, linked, or non-file artifact: {path}")
    return {"path": _rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _locked_files() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in LOCKED_FILES.items():
        record = _file(path)
        if record["sha256"] != expected:
            raise AttestationError(f"locked {name} hash changed")
        result[name] = record
    return result


def _inventory(root: Path, *, omit_derived: bool = False) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise AttestationError(f"invalid inventory root: {root}")
    omitted = {
        "results/adaptive-analysis.json",
        "results/adaptive-analysis-leave-two-units.json",
        "results/adaptive-analysis-leave-two-units.md",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AttestationError(f"inventory traverses symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if omit_derived and (relative in omitted or relative.startswith("results/adaptive-figures/")):
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"file_count": len(rows), "files": rows, "files_sha256": sha256_json(rows)}


def _ledger(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AttestationError(f"invalid ledger: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT " + ", ".join(LEDGER_COLUMNS)
            + " FROM reservations WHERE request_key LIKE ? ORDER BY reservation_id",
            (RUN_ID + "/%",),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _attempts(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(root / "events" / "call_attempts.jsonl")
    result = {str(row.get("event_id")): dict(row) for row in rows if isinstance(row, Mapping)}
    if len(rows) != EXPECTED_ATTEMPTS or len(result) != len(rows):
        raise AttestationError("attempt inventory count or uniqueness changed")
    return result


def _referenced_attempts(root: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted((root / "events").glob("adaptive-*.jsonl")):
        for row in read_jsonl(path):
            call = row.get("call") if isinstance(row, Mapping) else None
            if isinstance(call, Mapping):
                ids = call.get("call_event_ids")
                if isinstance(ids, list):
                    result.update(str(item) for item in ids)
    return result


def _field_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {"production": before.get(key), "staging": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def _run(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _expected_changed_paths() -> set[str]:
    result = {"events/call_attempts.jsonl"}
    for recovery in RECOVERIES:
        cell = recovery["cell_id"]
        result.update(
            {
                f"events/adaptive-{cell}.jsonl",
                f"results/adaptive_deployment/{cell}.json",
                f"results/adaptive_deployment_jobs/{cell}.json",
            }
        )
    return result


def _case_rows() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for recovery in RECOVERIES:
        groups: list[dict[str, Any]] = []
        for group in recovery["groups"]:
            attempts = [dict(row) for row in group["attempts"]]
            disclosure_only = bool(group.get("disclosure_only", False))
            group_row = {
                "checkpoint": group["checkpoint"],
                "logical_request_key": group["logical_request_key"],
                "semantic_failed_attempts": (
                    0
                    if disclosure_only
                    else sum(row["logical_status"] == "failed" for row in attempts)
                ),
                "logical_attempts": len(attempts),
                "disclosure_only": disclosure_only,
                "attempt_event_ids": [row["event_id"] for row in attempts],
                "attempt_chain_sha256": sha256_json(attempts),
            }
            groups.append(group_row)
        unit_id = f"{recovery['source_task_id']}/r0"
        case = {
            "cell_id": recovery["cell_id"],
            "method": "trace_judge",
            "operator": recovery["operator"],
            "source_task_id": recovery["source_task_id"],
            "replicate_id": 0,
            "unit_id": unit_id,
            "normalization_required": True,
            "groups": groups,
        }
        case["attempt_chain_sha256"] = sha256_json(groups)
        cases.append(case)
    return cases


def _verify_analysis(path: Path) -> dict[str, Any]:
    value = read_json(path)
    rows = value.get("rows") if isinstance(value, Mapping) else None
    summaries = value.get("metric_summaries") if isinstance(value, Mapping) else None
    effects = value.get("operator_effects") if isinstance(value, Mapping) else None
    if (
        value.get("artifact_type") != "online_adaptive_deployment_analysis"
        or value.get("source_run_id") != RUN_ID
        or value.get("source_manifest_sha256") != MANIFEST_SHA
        or value.get("source_pair_manifest_sha256") != PAIR_SHA
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_CELLS
        or len({row.get("unit_id") for row in rows}) != 40
        or len({row.get("method") for row in rows}) != 7
        or len({row.get("operator") for row in rows}) != 4
        or not isinstance(summaries, list)
        or len(summaries) != 224
        or {row.get("n_tasks") for row in summaries} != {40}
        or not isinstance(effects, list)
        or len(effects) != 168
        or {row.get("n_tasks") for row in effects} != {40}
    ):
        raise AttestationError("staged stock analysis dimensions or identity changed")
    return dict(value)


def _verify_locked_recovery_files() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for recovery in RECOVERIES:
        cell = recovery["cell_id"]
        paths = {
            "event": SOURCE_RUN / "events" / f"adaptive-{cell}.jsonl",
            "output": SOURCE_RUN / "results" / "adaptive_deployment" / f"{cell}.json",
            "job": SOURCE_RUN / "results" / "adaptive_deployment_jobs" / f"{cell}.json",
        }
        staged_paths = {
            "event": STAGE_RUN / "events" / f"adaptive-{cell}.jsonl",
            "output": STAGE_RUN / "results" / "adaptive_deployment" / f"{cell}.json",
            "job": STAGE_RUN / "results" / "adaptive_deployment_jobs" / f"{cell}.json",
        }
        for role, path in paths.items():
            if sha256_file(path) != recovery["production"][role]:
                raise AttestationError(f"production {role} oracle changed for {cell}")
            if sha256_file(staged_paths[role]) != recovery["staged"][role]:
                raise AttestationError(f"staged {role} oracle changed for {cell}")
        receipt_path = REPO / recovery["receipt"][0]
        if sha256_file(receipt_path) != recovery["receipt"][1]:
            raise AttestationError(f"recovery receipt changed for {cell}")
        records.append(
            {
                "cell_id": cell,
                "source_task_id": recovery["source_task_id"],
                "unit_id": f"{recovery['source_task_id']}/r0",
                "method": "trace_judge",
                "operator": recovery["operator"],
                "production": {role: _file(path) for role, path in paths.items()},
                "staging": {role: _file(path) for role, path in staged_paths.items()},
                "recovery_receipt": _file(receipt_path),
            }
        )
    return records


def audit() -> dict[str, Any]:
    if code_tree_hash(REPO / "experiments12") != CODE_SHA:
        raise AttestationError("frozen Experiment 12 source/config hash changed")
    locked = _locked_files()
    for root in (SOURCE_RUN, STAGE_RUN):
        if sha256_file(root / "manifest.json") != MANIFEST_SHA:
            raise AttestationError(f"manifest changed under {root}")
        if sha256_file(root / "pairs.jsonl") != PAIR_SHA:
            raise AttestationError(f"pair manifest changed under {root}")

    source_inventory = _inventory(SOURCE_RUN, omit_derived=True)
    stage_inventory = _inventory(STAGE_RUN)
    source_by_path = {row["path"]: row for row in source_inventory["files"]}
    stage_by_path = {row["path"]: row for row in stage_inventory["files"]}
    if set(source_by_path) != set(stage_by_path):
        raise AttestationError("production and staging raw file sets differ")
    changed_paths = {
        path
        for path in source_by_path
        if source_by_path[path]["sha256"] != stage_by_path[path]["sha256"]
    }
    if changed_paths != _expected_changed_paths():
        raise AttestationError("staging raw-file difference set changed")
    recovery_files = _verify_locked_recovery_files()

    source_attempts = _attempts(SOURCE_RUN)
    stage_attempts = _attempts(STAGE_RUN)
    if set(source_attempts) != set(stage_attempts):
        raise AttestationError("attempt event-ID set changed")
    semantic_attempts = {
        attempt["event_id"]
        for recovery in RECOVERIES
        for group in recovery["groups"]
        if not group.get("disclosure_only", False)
        for attempt in group["attempts"]
    }
    changed_attempts = {
        event_id
        for event_id in source_attempts
        if source_attempts[event_id] != stage_attempts[event_id]
    }
    if changed_attempts != semantic_attempts or len(changed_attempts) != 9:
        raise AttestationError("staged semantic attempt change set is not exact")
    attempt_changes: list[dict[str, Any]] = []
    for recovery in RECOVERIES:
        for group in recovery["groups"]:
            if group.get("disclosure_only", False):
                continue
            for attempt in group["attempts"]:
                event_id = attempt["event_id"]
                before = source_attempts[event_id]
                after = stage_attempts[event_id]
                if sha256_json(before) != attempt["production_attempt_sha256"]:
                    raise AttestationError(f"production attempt oracle changed: {event_id}")
                changes = _field_changes(before, after)
                if not changes or not set(changes).issubset({"attempt_number", "status"}):
                    raise AttestationError(f"unexpected staged attempt fields changed: {event_id}")
                if after.get("attempt_number") != attempt["logical_attempt_number"]:
                    raise AttestationError(f"logical attempt number changed: {event_id}")
                if after.get("status") != attempt["logical_status"]:
                    raise AttestationError(f"logical attempt status changed: {event_id}")
                attempt_changes.append(
                    {
                        "cell_id": recovery["cell_id"],
                        "checkpoint": group["checkpoint"],
                        "event_id": event_id,
                        "field_changes": changes,
                    }
                )
    source_unreferenced = set(source_attempts) - _referenced_attempts(SOURCE_RUN)
    stage_unreferenced = set(stage_attempts) - _referenced_attempts(STAGE_RUN)
    expected_unreferenced = {
        attempt["event_id"]
        for recovery in RECOVERIES
        for group in recovery["groups"]
        if not group.get("disclosure_only", False)
        for attempt in group["attempts"]
        if attempt["logical_status"] == "failed"
    }
    if source_unreferenced != expected_unreferenced or stage_unreferenced:
        raise AttestationError("physical-attempt reference boundary changed")

    source_ledger_rows = _ledger(SOURCE_LEDGER)
    stage_ledger_rows = _ledger(STAGE_LEDGER)
    source_ledger = {row["reservation_id"]: row for row in source_ledger_rows}
    stage_ledger = {row["reservation_id"]: row for row in stage_ledger_rows}
    if len(source_ledger) != EXPECTED_ATTEMPTS or set(source_ledger) != set(stage_ledger):
        raise AttestationError("run ledger reservation set changed")
    changed_reservations = {
        reservation
        for reservation in source_ledger
        if source_ledger[reservation] != stage_ledger[reservation]
    }
    expected_reservations = {
        attempt["reservation_id"]
        for recovery in RECOVERIES
        for group in recovery["groups"]
        if not group.get("disclosure_only", False)
        for attempt in group["attempts"]
    } | {ORDINARY["reservation_id"]}
    if changed_reservations != expected_reservations or len(changed_reservations) != 10:
        raise AttestationError("staged ledger change set is not exact")
    ledger_changes: list[dict[str, Any]] = []
    for reservation in sorted(changed_reservations):
        changes = _field_changes(source_ledger[reservation], stage_ledger[reservation])
        if not changes or not set(changes).issubset({"request_key", "request_status"}):
            raise AttestationError(f"accounting field changed for reservation {reservation}")
        ledger_changes.append(
            {
                "reservation_id": reservation,
                "event_id": next(
                    (event for event, row in source_attempts.items() if row.get("reservation_id") == reservation),
                    None,
                ),
                "field_changes": changes,
            }
        )
    ordinary_source = source_ledger[ORDINARY["reservation_id"]]
    ordinary_stage = stage_ledger[ORDINARY["reservation_id"]]
    if (
        sha256_json(ordinary_source) != ORDINARY["source_ledger_row_sha256"]
        or ordinary_source.get("request_status") != "unknown"
        or ordinary_stage.get("request_status") != "failed"
        or ordinary_stage.get("actual_micro_usd") != 2578
        or ordinary_stage.get("cost_quality") != "upper_bound"
    ):
        raise AttestationError("ordinary HTTP-503 ledger reconciliation changed")

    analysis = _verify_analysis(STAGE_ANALYSIS)
    return {
        "locked_files": locked,
        "source_inventory": source_inventory,
        "stage_inventory": stage_inventory,
        "changed_paths": sorted(changed_paths),
        "unchanged_file_count": len(source_by_path) - len(changed_paths),
        "recovery_files": recovery_files,
        "attempt_changes": attempt_changes,
        "source_unreferenced_attempts": sorted(source_unreferenced),
        "stage_unreferenced_attempts": sorted(stage_unreferenced),
        "source_ledger_rows_sha256": sha256_json(source_ledger_rows),
        "stage_ledger_rows_sha256": sha256_json(stage_ledger_rows),
        "ledger_changes": ledger_changes,
        "analysis": analysis,
    }


def _analyzer_gates() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="adaptive-attestation-analyzer-",
        dir=REPO / "experiments12" / "data_results" / "derived",
    ) as temporary_name:
        temporary = Path(temporary_name)
        raw_command = (
            sys.executable,
            "-m",
            "experiments12.adaptive_analysis12",
            "extract",
            "--run-id",
            RUN_ID,
            "--manifest-sha256",
            MANIFEST_SHA,
            "--output",
            str(temporary / "raw-analysis.json"),
            "--figures",
            str(temporary / "raw-figures"),
            "--artifacts",
            str(SOURCE_BASE),
            "--bootstrap-iterations",
            "2000",
            "--bootstrap-seed",
            "12012",
        )
        raw = _run(raw_command)
        expected_failure = (
            "error: call attempt disagrees with ledger: "
            "d950af6bd8a8421e99f8efc17125fa1b"
        )
        if (
            raw["returncode"] != 2
            or expected_failure not in raw["stdout"] + raw["stderr"]
            or (temporary / "raw-analysis.json").exists()
        ):
            raise AttestationError("unmodified analyzer did not fail closed on raw production")

        staged_output = temporary / "staged-analysis.json"
        staged_figures = temporary / "staged-figures"
        staged_command = (
            sys.executable,
            "-m",
            "experiments12.adaptive_analysis12",
            "extract",
            "--run-id",
            RUN_ID,
            "--manifest-sha256",
            MANIFEST_SHA,
            "--output",
            str(staged_output),
            "--figures",
            str(staged_figures),
            "--artifacts",
            str(STAGE_BASE),
            "--bootstrap-iterations",
            "2000",
            "--bootstrap-seed",
            "12012",
        )
        staged = _run(staged_command)
        if staged["returncode"] != 0 or not staged_output.is_file():
            raise AttestationError(
                "unmodified analyzer failed on the audited staging view: "
                + staged["stdout"]
                + staged["stderr"]
            )
        if sha256_file(staged_output) != LOCKED_FILES["staged_analysis"][1]:
            raise AttestationError("fresh stock analysis differs from staged analysis oracle")
        expected_figures = {
            path.name: digest
            for name, (path, digest) in LOCKED_FILES.items()
            if name in {"figure_data", "figure_svg"}
        }
        observed_figures = {
            path.name: sha256_file(path)
            for path in sorted(staged_figures.iterdir())
            if path.is_file()
        }
        if observed_figures != expected_figures:
            raise AttestationError("fresh stock figures differ from staged figure oracles")
        return {
            "raw_production": {
                "expected_failure": True,
                "returncode": raw["returncode"],
                "stdout_sha256": sha256_json(raw["stdout"]),
                "stderr_sha256": sha256_json(raw["stderr"]),
                "diagnostic": expected_failure,
                "reproduction_command": [
                    *raw_command[:10],
                    "<TEMP_OUTPUT>",
                    *raw_command[11:12],
                    "<TEMP_FIGURES>",
                    *raw_command[13:],
                ],
            },
            "audited_staging": {
                "returncode": staged["returncode"],
                "analysis_sha256": sha256_file(staged_output),
                "figure_hashes": observed_figures,
                "stdout_sha256": sha256_json(staged["stdout"]),
                "stderr_sha256": sha256_json(staged["stderr"]),
                "reproduction_command": [
                    *staged_command[:10],
                    "<TEMP_OUTPUT>",
                    *staged_command[11:12],
                    "<TEMP_FIGURES>",
                    *staged_command[13:],
                ],
            },
        }


def _summary_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": inventory["file_count"],
        "files_sha256": inventory["files_sha256"],
    }


def _publish(source: Path, target: Path) -> dict[str, Any]:
    source_record = _file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise AttestationError(f"publication target is unsafe: {target}")
        if sha256_file(target) != source_record["sha256"]:
            raise AttestationError(f"publication would overwrite different data: {target}")
    else:
        atomic_write_bytes(target, source.read_bytes())
    if sha256_file(target) != source_record["sha256"]:
        raise AttestationError(f"published artifact hash mismatch: {target}")
    return {
        "source": source_record,
        "published": _file(target),
    }


def build() -> dict[str, Any]:
    if ATTEST_BASE.exists():
        raise AttestationError(f"attestation output already exists: {ATTEST_BASE}")
    audited = audit()
    analyzer_gates = _analyzer_gates()
    source_before = _summary_inventory(audited["source_inventory"])

    ATTEST_ANALYSIS.parent.mkdir(parents=True, exist_ok=False)
    cases = _case_rows()
    affected_units = sorted({case["unit_id"] for case in cases})
    case_index = {
        "artifact_type": "experiment12_adaptive_normalization_case_index",
        "schema_version": 1,
        "run_id": RUN_ID,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_pair_manifest_sha256": PAIR_SHA,
        "selection_basis": "all source units named by the complete frozen three-cell semantic-recovery set; no outcomes inspected",
        "outcome_fields_used_for_selection": [],
        "cases": cases,
        "affected_units": affected_units,
        "recovery_cells": len(cases),
        "affected_source_units": len(affected_units),
        "provider_calls_made": 0,
    }
    atomic_write_json(CASE_INDEX, case_index)
    case_index_sha = sha256_file(CASE_INDEX)

    script_record = _file(Path(__file__))
    semantic_failures = sum(
        group["semantic_failed_attempts"]
        for case in cases
        for group in case["groups"]
    )
    staging_receipt = {
        "artifact_type": "experiment12_adaptive_semantic_retry_normalization",
        "schema_version": 1,
        "attestation_role": "post_hoc_hash_bound_attestation_of_existing_copy_on_write_stage",
        "created_at_utc": _now(),
        "run_id": RUN_ID,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_pair_manifest_sha256": PAIR_SHA,
        "source_code_tree_sha256": CODE_SHA,
        "attestation_script": script_record,
        "attestation_command": [
            sys.executable,
            _rel(Path(__file__)),
            "build",
        ],
        "copy_on_write": True,
        "production_files_or_ledger_modified": False,
        "outcome_values_used_to_select_patch_scope": False,
        "scientific_score_or_decision_changed": False,
        "all_attempt_tokens_latency_and_cost_retained": True,
        "provider_calls_made": 0,
        "executed_stage_builder_identity": "unknown",
        "executed_stage_builder_sha256": None,
        "provenance_limitation": "the originally executed staging builder was not captured before a later file replacement; this attestation instead binds the complete source/stage diff, independent serialization oracles, and a fresh stock-analyzer replay",
        "preexisting_stage_receipts": {
            "staging_receipt": audited["locked_files"]["existing_staging_receipt"],
            "analysis_receipt": audited["locked_files"]["existing_analysis_receipt"],
        },
        "source_raw_inventory": source_before,
        "staging_raw_inventory": _summary_inventory(audited["stage_inventory"]),
        "same_relative_file_set": True,
        "unchanged_raw_files": audited["unchanged_file_count"],
        "changed_raw_files": len(audited["changed_paths"]),
        "changed_raw_paths": audited["changed_paths"],
        "normalization_cases_path": _rel(CASE_INDEX),
        "normalization_cases_sha256": case_index_sha,
        "normalization_cases": cases,
        "normalized_recovery_cell_count": 3,
        "normalized_source_unit_count": 2,
        "semantic_failed_physical_attempts": semantic_failures,
        "attempt_record_changes": audited["attempt_changes"],
        "source_unreferenced_attempts": audited["source_unreferenced_attempts"],
        "staging_unreferenced_attempts": audited["stage_unreferenced_attempts"],
        "source_run_ledger_rows_sha256": audited["source_ledger_rows_sha256"],
        "staging_run_ledger_rows_sha256": audited["stage_ledger_rows_sha256"],
        "ledger_row_changes": audited["ledger_changes"],
        "ordinary_failed_attempt_ledger_reconciliations": [
            {
                "event_id": ORDINARY["event_id"],
                "reservation_id": ORDINARY["reservation_id"],
                "request_key": ORDINARY["request_key"],
                "transport_error": "HTTP 503",
                "ledger_request_status_before": "unknown",
                "ledger_request_status_after": "failed",
                "actual_micro_usd_retained": 2578,
                "cost_quality_retained": "upper_bound",
                "description": "staged ledger status reconciled to failed for the ordinary HTTP 503 attempt; conservative upper-bound charge retained",
            }
        ],
        "recovery_file_oracles": audited["recovery_files"],
        "fresh_unmodified_analyzer_gates": analyzer_gates,
        "resource_semantics": audited["analysis"].get("resource_semantics"),
    }
    atomic_write_json(STAGING_RECEIPT, staging_receipt)
    staging_receipt_sha = sha256_file(STAGING_RECEIPT)

    atomic_write_bytes(ATTEST_ANALYSIS, STAGE_ANALYSIS.read_bytes())
    if sha256_file(ATTEST_ANALYSIS) != LOCKED_FILES["staged_analysis"][1]:
        raise AttestationError("attested analysis copy changed")
    sensitivity_command = (
        sys.executable,
        str(SENSITIVITY_SCRIPT),
        "--input",
        str(ATTEST_ANALYSIS),
        "--expected-input-sha256",
        sha256_file(ATTEST_ANALYSIS),
        "--normalization-cases",
        str(CASE_INDEX),
        "--expected-normalization-cases-sha256",
        case_index_sha,
        "--output",
        str(SENSITIVITY),
        "--markdown",
        str(SENSITIVITY_MD),
    )
    sensitivity_run = _run(sensitivity_command)
    if sensitivity_run["returncode"] != 0:
        raise AttestationError(
            "cumulative sensitivity failed: "
            + sensitivity_run["stdout"]
            + sensitivity_run["stderr"]
        )
    sensitivity = read_json(SENSITIVITY)
    design = sensitivity.get("design") if isinstance(sensitivity, Mapping) else None
    if (
        sensitivity.get("artifact_type")
        != "experiment12_online_cumulative_affected_unit_sensitivity"
        or sensitivity.get("provider_calls_made") != 0
        or sensitivity.get("analysis_only") is not True
        or not isinstance(design, Mapping)
        or design.get("source_rows") != 1120
        or design.get("filtered_rows") != 1064
        or design.get("source_tasks") != 40
        or design.get("filtered_tasks") != 38
        or design.get("recovery_cells") != 3
        or design.get("affected_source_units") != 2
        or design.get("treatments") != 28
    ):
        raise AttestationError("cumulative sensitivity dimensions changed")

    publication: list[dict[str, Any]] = []
    publication.append(_publish(ATTEST_ANALYSIS, PROD_ANALYSIS))
    for path in sorted(STAGE_FIGURES.iterdir()):
        if path.is_file():
            publication.append(_publish(path, PROD_FIGURES / path.name))
    publication.append(_publish(SENSITIVITY, PROD_SENSITIVITY))
    publication.append(_publish(SENSITIVITY_MD, PROD_SENSITIVITY_MD))

    source_after_inventory = _inventory(SOURCE_RUN, omit_derived=True)
    source_after = _summary_inventory(source_after_inventory)
    if source_after != source_before:
        raise AttestationError("production raw inputs changed during attestation")

    analysis_receipt = {
        "artifact_type": "experiment12_staged_stock_adaptive_analysis_receipt",
        "schema_version": 1,
        "created_at_utc": _now(),
        "run_id": RUN_ID,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_pair_manifest_sha256": PAIR_SHA,
        "staging_receipt_path": _rel(STAGING_RECEIPT),
        "staging_receipt_sha256": staging_receipt_sha,
        "normalization_cases_path": _rel(CASE_INDEX),
        "normalization_cases_sha256": case_index_sha,
        "unmodified_stock_analyzer": True,
        "raw_production_analyzer_failed_closed": True,
        "analysis_rows": EXPECTED_CELLS,
        "source_tasks_per_treatment": 40,
        "analysis_output_path": _rel(PROD_ANALYSIS),
        "analysis_output_sha256": sha256_file(PROD_ANALYSIS),
        "staged_analysis_path": _rel(ATTEST_ANALYSIS),
        "staged_analysis_sha256": sha256_file(ATTEST_ANALYSIS),
        "sensitivity_path": _rel(PROD_SENSITIVITY),
        "sensitivity_sha256": sha256_file(PROD_SENSITIVITY),
        "sensitivity_markdown_path": _rel(PROD_SENSITIVITY_MD),
        "sensitivity_markdown_sha256": sha256_file(PROD_SENSITIVITY_MD),
        "fresh_unmodified_analyzer_gates": analyzer_gates,
        "sensitivity_command": list(sensitivity_command),
        "sensitivity_stdout_sha256": sha256_json(sensitivity_run["stdout"]),
        "sensitivity_stderr_sha256": sha256_json(sensitivity_run["stderr"]),
        "published_derived_artifacts": publication,
        "production_raw_inventory_before": source_before,
        "production_raw_inventory_after": source_after,
        "production_raw_immutable": True,
        "provider_calls_made": 0,
    }
    atomic_write_json(ANALYSIS_RECEIPT, analysis_receipt)
    return {
        "status": "complete",
        "provider_calls_made": 0,
        "staging_receipt": _file(STAGING_RECEIPT),
        "analysis_receipt": _file(ANALYSIS_RECEIPT),
        "analysis": _file(PROD_ANALYSIS),
        "sensitivity": _file(PROD_SENSITIVITY),
        "figures": [
            _file(path)
            for path in sorted(PROD_FIGURES.iterdir())
            if path.is_file()
        ],
    }


def preflight() -> dict[str, Any]:
    audited = audit()
    return {
        "status": "passed",
        "provider_calls_made": 0,
        "source_raw_files": audited["source_inventory"]["file_count"],
        "staged_raw_files": audited["stage_inventory"]["file_count"],
        "unchanged_raw_files": audited["unchanged_file_count"],
        "changed_raw_files": len(audited["changed_paths"]),
        "changed_attempt_records": len(audited["attempt_changes"]),
        "changed_ledger_rows": len(audited["ledger_changes"]),
        "source_unreferenced_attempts": len(audited["source_unreferenced_attempts"]),
        "staging_unreferenced_attempts": len(audited["stage_unreferenced_attempts"]),
        "analysis_rows": len(audited["analysis"]["rows"]),
        "build_command": [sys.executable, _rel(Path(__file__)), "build"],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("command", choices=("preflight", "build"))
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = preflight() if args.command == "preflight" else build()
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (
        AttestationError,
        FileExistsError,
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
