#!/usr/bin/env python3
"""Build the fail-closed final paper-material inventory for Experiment 12.

This is a provider-free release audit.  It reads frozen manifests, completed
results, analysis receipts, figures, and the global budget ledger.  It writes
``PAPER_MATERIALS12.json`` and ``PAPER_MATERIALS12.md`` only after every gate
passes; otherwise it exits 2 and leaves any prior final files untouched.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments12.manifest12 import code_tree_hash  # noqa: E402
from experiments12.adaptive_analysis12 import summarize_adaptive_outcomes  # noqa: E402


EXPERIMENT = ROOT / "experiments12"
ARTIFACTS = EXPERIMENT / "data_results" / "runs"
GENERATED = EXPERIMENT / "data_results" / "derived"
DEFAULT_STAGING = GENERATED / "adaptive-analysis-staging-v1"
OUTPUT_JSON = GENERATED / "PAPER_MATERIALS12.json"
OUTPUT_MD = GENERATED / "PAPER_MATERIALS12.md"

EXPECTED_CODE_TREE_SHA256 = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)
ONLINE_RUN = "e12-deploy-online-evolving-luna-40-v1"
YOKED_RUN = "e12-deploy-twopass-yoked-evolving-luna-40-v1"
PASS_ONE_RUN = "e12-deploy-twopass-pass1-evolving-luna-40-v1"
ONLINE_MANIFEST_SHA256 = (
    "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
)
ONLINE_PAIRS_SHA256 = (
    "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
)
YOKED_MANIFEST_SHA256 = (
    "8cd3194cb460aa5e77a8d9b7b7aeee01f6c81df7d655f22f7a03a529f4062250"
)

RECOVERY_CELLS = {
    "d52046b6eb74a76ecdc3debc": ("trace_judge", "lossy_compaction"),
    "89df41e0daa1262a43fa5e55": ("trace_judge", "public_state_reground"),
    "786d95760ccdb86713c26936": ("trace_judge", "public_state_reground"),
}
RECOVERY_RECEIPT_TYPES = {
    "d52046b6eb74a76ecdc3debc": "experiment12_online_adaptive_single_cell_recovery",
    "89df41e0daa1262a43fa5e55": "experiment12_online_adaptive_trace_judge_recovery",
    "786d95760ccdb86713c26936": "experiment12_online_adaptive_trace_judge_recovery",
}
RECOVERY_RECEIPT_SHA256 = {
    "d52046b6eb74a76ecdc3debc": "83f8939e08e7809d699e51e62a13b68aad838018669d4792ab6e84645741eca1",
    "89df41e0daa1262a43fa5e55": "7fdfe614fe976db85343586e4908785aa90ffe734045e629db9af5b46249329e",
    "786d95760ccdb86713c26936": "0110cc242d6ffdec0c4fd1b1e45a606b5b7bee141a1d35a28fdc16f11d056509",
}
RECOVERY_TASKS = {
    "d52046b6eb74a76ecdc3debc": "extracted-gsm8k-test-814::t7",
    "89df41e0daa1262a43fa5e55": "extracted-gsm8k-test-814::t7",
    "786d95760ccdb86713c26936": "extracted-gsm8k-test-989::t7",
}
RECOVERY_UNIT_IDS = {
    "extracted-gsm8k-test-814::t7/r0",
    "extracted-gsm8k-test-989::t7/r0",
}


RUN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "run_id": "e12-baseline-evolving-allarms-allmodels-v1",
        "role": "exploratory active-probe mechanism / Evolving Intent",
        "manifest_sha256": "f538166a9b1e657429be547e617822b9160df1b37e496526518b4424a6d3b852",
        "cells": 500,
        "tasks": 20,
        "models": 5,
        "arms": 5,
        "operators": 1,
    },
    {
        "run_id": "e12-baseline-bfcl-allarms-fourmodels-v2",
        "role": "exploratory active-probe mechanism / BFCL",
        "manifest_sha256": "eca11658fb6167e0877f4180517c197fbefacc2dffd2d43f6f6530e09e962407",
        "cells": 400,
        "tasks": 20,
        "models": 4,
        "arms": 5,
        "operators": 1,
    },
    {
        "run_id": "e12-calibration-evolving-core-v2",
        "role": "threshold calibration / Evolving Intent",
        "manifest_sha256": "9ce66a541c6ca5a48adef1a42179433bb5a7ce287ae5eb6817c97284552f86f0",
        "cells": 160,
        "tasks": 20,
        "models": 4,
        "arms": 2,
        "operators": 1,
    },
    {
        "run_id": "e12-calibration-bfcl-core-v1",
        "role": "threshold calibration / BFCL",
        "manifest_sha256": "b1043d46b6ebfe3716e2d24741668d3ea1432f76bdb723c1cde3de4f5b897868",
        "cells": 120,
        "tasks": 20,
        "models": 3,
        "arms": 2,
        "operators": 1,
    },
    {
        "run_id": "e12-confirmatory-evolving-core-v2",
        "role": "confirmatory observer effect and signal accuracy / Evolving Intent",
        "manifest_sha256": "b0be97ed6b01c9c3b003a5900b8e596772fffe3c68c4b9cdd47f46ff329c7056",
        "cells": 448,
        "tasks": 56,
        "models": 4,
        "arms": 2,
        "operators": 1,
    },
    {
        "run_id": "e12-confirmatory-bfcl-core-v3",
        "role": "confirmatory observer effect and signal accuracy / BFCL",
        "manifest_sha256": "551a4574a1fe502c9304feff46c956ffb6b19a96af84d54d3e18d3a60fd919c3",
        "cells": 336,
        "tasks": 56,
        "models": 3,
        "arms": 2,
        "operators": 1,
    },
    {
        "run_id": PASS_ONE_RUN,
        "role": "frozen passive pass-one schedule source",
        "manifest_sha256": "b5f4d46f8deb8b899c8d9cb35ae3758f858077e6a94704a30f9c3cabe5d2aa8f",
        "cells": 80,
        "tasks": 40,
        "models": 1,
        "arms": 2,
        "operators": 1,
    },
    {
        "run_id": ONLINE_RUN,
        "role": "primary online adaptive deployment",
        "manifest_sha256": ONLINE_MANIFEST_SHA256,
        "cells": 1120,
        "tasks": 40,
        "models": 1,
        "arms": 7,
        "operators": 4,
    },
    {
        "run_id": YOKED_RUN,
        "role": "frozen two-pass yoked deployment sensitivity",
        "manifest_sha256": YOKED_MANIFEST_SHA256,
        "cells": 480,
        "tasks": 40,
        "models": 1,
        "arms": 4,
        "operators": 3,
    },
)

SHADOW_OUTPUT_COUNTS = {
    "e12-baseline-evolving-allarms-allmodels-v1": 0,
    "e12-baseline-bfcl-allarms-fourmodels-v2": 0,
    "e12-calibration-evolving-core-v2": 80,
    "e12-calibration-bfcl-core-v1": 60,
    "e12-confirmatory-evolving-core-v2": 224,
    "e12-confirmatory-bfcl-core-v3": 168,
    PASS_ONE_RUN: 40,
    ONLINE_RUN: 0,
    YOKED_RUN: 0,
}


FIXED_FILES: tuple[tuple[str, str, str, str], ...] = (
    (
        "experiments12/data_results/derived/lock-calibration-evolving-core-v2.json",
        "2d893d6ab91ddffd8fc28e2991081f5f2c9a7f58e4445ccc9d1dc68a5472f116",
        "immutable_design",
        "Evolving calibration launch lock",
    ),
    (
        "experiments12/data_results/derived/lock-calibration-bfcl-core-v1.json",
        "3eaf259ef67b20487b3f6d0a338ec15554563a9ca9e367be4d7962c0259b2104",
        "immutable_design",
        "BFCL calibration launch lock",
    ),
    (
        "experiments12/data_results/derived/lock-confirmatory-evolving-core-v2.json",
        "d7b4781481d6b09f377a359bf68618f7edf2377c3b06bdb8f9052c6e75c45f28",
        "immutable_design",
        "Evolving confirmatory launch lock",
    ),
    (
        "experiments12/data_results/derived/lock-confirmatory-bfcl-core-v2.json",
        "3a321ed07769b17656fa9812af71986f85879673b5d2d1a4e723fe4188b09cb3",
        "immutable_design",
        "BFCL confirmatory launch lock",
    ),
    (
        "experiments12/data_results/derived/lock-deploy-online-evolving-luna-40-v1.json",
        "569f785640a89c046cda340640988ba4f881f944bbc4344cce5e3c5baedb4427",
        "immutable_design",
        "online deployment launch lock",
    ),
    (
        "experiments12/data_results/derived/lock-deploy-twopass-yoked-evolving-luna-40-v1.json",
        "cd21c1878fc10912501aba2c7c5d222a6fcbb8aa73a53b8679d52c6a16294355",
        "immutable_design",
        "two-pass deployment launch lock",
    ),
    (
        "experiments12/data_results/derived/thresholds-calibration-evolving-core-v2.json",
        "ae27c2dc38197e60864e11f3435cec45e86c13f1f787b585ef9b844440199623",
        "immutable_design",
        "Evolving frozen method thresholds",
    ),
    (
        "experiments12/data_results/derived/thresholds-calibration-bfcl-core-v1.json",
        "a140e5405180cde25bac6702c9164eb800ac27af4714db1962be79031620b84b",
        "immutable_design",
        "BFCL frozen method thresholds",
    ),
    (
        f"experiments12/data_results/runs/{ONLINE_RUN}/results/deployment_threshold_lock.json",
        "061216da43506e13159eada54226c697cd94d0a72da8203c05605a69e14247d2",
        "immutable_design",
        "online deployment threshold lock",
    ),
    (
        f"experiments12/data_results/runs/{YOKED_RUN}/results/deployment_threshold_lock.json",
        "eefb90b329d487435f9774612b4c3836ce0eae83d6bf78c39f3848a470edc2bc",
        "immutable_design",
        "two-pass deployment threshold lock",
    ),
    (
        f"experiments12/data_results/runs/{YOKED_RUN}/results/deployment_schedule.json",
        "fa6ebd579a58369d13343c22870d3772fa8c4f4ddc1b07e2e3120f23a92f635f",
        "immutable_design",
        "frozen checkpoint-1 yoked schedule",
    ),
    (
        f"experiments12/data_results/runs/{YOKED_RUN}/results/deployment_pass_one.json",
        "0e6179bd5a1095e5a576ff64c2013c9ba9ac0c0dd09b2355ff18b91aa1bb4af0",
        "immutable_design",
        "frozen pass-one observations used by the yoked schedule",
    ),
)


FIXED_RESULTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "experiments12/data_results/runs/e12-baseline-evolving-allarms-allmodels-v1/results/validation-baseline-exploratory.json",
        "bd338bc2c6287a94776788df06731c57db2c4633e4dd24250f75aa4db619c214",
        "exploratory_result",
        "Evolving all-arm validation",
    ),
    (
        "experiments12/data_results/runs/e12-baseline-evolving-allarms-allmodels-v1/results/extract-baseline-exploratory.json",
        "75840685649e87da837dd01a66ad8197a178cb3a8fa19c5e88cd67c638801f51",
        "exploratory_result",
        "Evolving active-probe mechanism extract",
    ),
    (
        "experiments12/data_results/runs/e12-baseline-bfcl-allarms-fourmodels-v2/results/validation-baseline-exploratory.json",
        "2381423273eca79082cbde12103a3285e6223f6fa0843f387c966114f8f6bd15",
        "exploratory_result",
        "BFCL all-arm validation",
    ),
    (
        "experiments12/data_results/runs/e12-baseline-bfcl-allarms-fourmodels-v2/results/extract-baseline-exploratory.json",
        "6161557d8ecb18c4564f415ef2046be473f1378898ab9611c3eb93e0c9d0e42b",
        "exploratory_result",
        "BFCL active-probe mechanism extract",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-evolving-core-v2/results/validation-confirmatory.json",
        "318b256ab335aa79b483658a7b950ab264d5a10699a94e0966f58116b6164733",
        "primary_result",
        "Evolving confirmatory validation",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-evolving-core-v2/results/extract-confirmatory.json",
        "26e1a7ff96cad026f1cabf35375053032c0e57e133f5834772a480754d1c23db",
        "primary_result",
        "Evolving confirmatory observer-effect extract",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-evolving-core-v2/results/score-confirmatory.json",
        "d270a92b33fde2d69bbd972cc987727bfa5c6c5b5b24c4cae262f957279044a4",
        "primary_result",
        "Evolving confirmatory signal score",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-bfcl-core-v3/results/validation-confirmatory.json",
        "3014834235c2a547eb6d80ac559a28bb35dcfa8e1359b6a4c16700db45c544bb",
        "primary_result",
        "BFCL confirmatory validation",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-bfcl-core-v3/results/extract-confirmatory.json",
        "48398ece77a0d2800975f13fa1f11db723ce055e8c1a1dd0098ebe802a14a927",
        "primary_result",
        "BFCL confirmatory observer-effect extract",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-bfcl-core-v3/results/score-confirmatory-no-opportunity-v1.json",
        "4b60934f38cb806bcf29a3879763020138761c2631502e9e53bda41a6cb5afd5",
        "primary_result",
        "BFCL no-opportunity primary signal score",
    ),
    (
        "experiments12/data_results/runs/e12-confirmatory-bfcl-core-v3/results/score-confirmatory-complete-case-sensitivity.json",
        "d5c2c358bf9b31a7c0c3d83cf74425bfdd10f9f6617bb4ac48c4bb5b58f28b18",
        "sensitivity_result",
        "BFCL complete-case signal sensitivity",
    ),
    (
        f"experiments12/data_results/runs/{PASS_ONE_RUN}/results/validation-pass-one.json",
        "b652552d679f6b307b549bf57073ee1851764bc7f24f31550a18068151e5dc98",
        "sensitivity_result",
        "pass-one strict validation with disclosed recovery warning",
    ),
)


CAVEATS: tuple[str, ...] = (
    "The carried active recomputation effect is negative in six of seven powered model/benchmark strata, not a universal rule; one powered stratum is positive.",
    "No observation signal wins universally: the active signal has the highest AUPRC in only three of seven powered model/benchmark strata.",
    "Evolving Intent uses final-task failure labels; BFCL uses turn-level failure opportunities, so their signal metrics are not interchangeable.",
    "Deployment covers one model on one reasoning benchmark; there is no BFCL deployment run.",
    "The nominal GOOD/BAD/WATCH operator emits deterministic WATCH with a current-prefix exact quote, not an LLM-written GOOD/BAD critique.",
    "Natural deployment uses unequal scalar thresholds; some methods fire on 100% of tasks, so deployment is not a uniform 20% policy.",
    "The yoked sensitivity uses an aggressive checkpoint-1 anchor on the same 40 tasks; it is a schedule sensitivity, not a second independent sample.",
    "Three semantic-normalization recoveries occurred on two source units; every physical attempt and its cost is retained, and the cumulative paired n=38 analysis excludes both units from all 28 treatments.",
    "The copy-on-write analysis staging reconciles one pre-existing ledger request status from unknown to failed so the stock analyzer can enforce call/ledger agreement; production inputs remain immutable.",
    "Pass one has a disclosed trace-judge recovery warning for an unreferenced provider attempt; its billed resources remain in the ledger.",
    "Deployment has no unmonitored or oracle arm; method comparisons use the declared baseline and observation methods only.",
    "Online resource totals include retries; two-pass resource totals describe pass two only and exclude the frozen pass-one passive cost.",
    "The active-probe complexity ladder is exploratory and non-monotone; no monotonic burden claim is licensed.",
    "BFCL primary scoring applies the no-opportunity rule and must be shown beside its complete-case sensitivity.",
)


class AuditError(ValueError):
    """Raised whenever a final paper-material gate cannot be proven."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise AuditError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise AuditError(f"paper artifact escapes repository: {path}") from exc


def _require_regular(path: Path, *, allow_empty: bool = False) -> None:
    absolute = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"missing or linked paper artifact: {path}")
    if not allow_empty and path.stat().st_size <= 0:
        raise AuditError(f"empty paper artifact: {path}")
    root = ROOT.resolve()
    cursor = absolute.parent
    while cursor != root:
        if cursor.is_symlink():
            raise AuditError(f"paper artifact traverses symlink: {path}")
        if root not in cursor.resolve().parents and cursor.resolve() != root:
            raise AuditError(f"paper artifact escapes repository: {path}")
        cursor = cursor.parent


def _read_json(path: Path) -> Any:
    _require_regular(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON: {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    _require_regular(path)
    rows: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AuditError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
    except UnicodeDecodeError as exc:
        raise AuditError(f"invalid UTF-8 JSONL: {path}") from exc
    if not rows:
        raise AuditError(f"empty JSONL: {path}")
    return rows


class Inventory:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def add(
        self,
        path: Path,
        *,
        category: str,
        role: str,
        expected_sha256: str | None = None,
        allow_jsonl_with_json_suffix: bool = False,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        _require_regular(path, allow_empty=allow_empty)
        label = _relative(path)
        if label in self.entries:
            entry = self.entries[label]
            if entry["category"] != category and category not in entry["also_categories"]:
                entry["also_categories"].append(category)
            if role != entry["role"] and role not in entry["also_roles"]:
                entry["also_roles"].append(role)
            if expected_sha256 is not None and entry["sha256"] != expected_sha256:
                raise AuditError(f"fixed hash changed: {label}")
            return entry
        digest = _sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise AuditError(
                f"fixed hash changed: {label}: expected {expected_sha256}, got {digest}"
            )
        entry: dict[str, Any] = {
            "path": label,
            "category": category,
            "also_categories": [],
            "role": role,
            "also_roles": [],
            "sha256": digest,
            "bytes": path.stat().st_size,
            "format": path.suffix.lstrip(".").lower() or "none",
        }
        if path.suffix == ".json":
            try:
                value = _read_json(path)
            except AuditError:
                if not allow_jsonl_with_json_suffix:
                    raise
                rows = _read_jsonl(path)
                entry["format"] = "jsonl"
                entry["declared_suffix"] = "json"
                entry["rows"] = len(rows)
                self.entries[label] = entry
                return entry
            entry["json_shape"] = "object" if isinstance(value, Mapping) else "array"
            if isinstance(value, Mapping):
                entry["top_level_keys"] = sorted(str(key) for key in value)
                if isinstance(value.get("artifact_type"), str):
                    entry["artifact_type"] = value["artifact_type"]
            else:
                entry["items"] = len(value)
        elif path.suffix == ".jsonl":
            entry["rows"] = len(_read_jsonl(path))
        elif path.suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader)
                    rows = sum(1 for row in reader if row)
            except (StopIteration, UnicodeDecodeError, csv.Error) as exc:
                raise AuditError(f"invalid CSV: {path}") from exc
            if not header or rows < 1:
                raise AuditError(f"CSV lacks header or data rows: {path}")
            entry["header"] = header
            entry["rows"] = rows
        elif path.suffix == ".svg":
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError as exc:
                raise AuditError(f"invalid SVG XML: {path}: {exc}") from exc
            if root.tag.rsplit("}", 1)[-1] != "svg":
                raise AuditError(f"XML artifact is not SVG: {path}")
            entry["svg_root_valid"] = True
        elif path.suffix in {".md", ".py"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise AuditError(f"invalid UTF-8 text artifact: {path}") from exc
            entry["lines"] = len(text.splitlines())
        self.entries[label] = entry
        return entry

    def ordered(self) -> list[dict[str, Any]]:
        return [self.entries[key] for key in sorted(self.entries)]


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise AuditError(f"{context}: expected {expected!r}, got {actual!r}")


def _manifest_and_sample_sizes(inventory: Inventory) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    samples: list[dict[str, Any]] = []
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in RUN_SPECS:
        run_id = str(spec["run_id"])
        run_root = ARTIFACTS / run_id
        manifest_path = run_root / "manifest.json"
        pairs_path = run_root / "pairs.jsonl"
        inventory.add(
            manifest_path,
            category="immutable_design",
            role=f"manifest: {spec['role']}",
            expected_sha256=str(spec["manifest_sha256"]),
        )
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            raise AuditError(f"manifest is not an object: {run_id}")
        _assert_equal(manifest.get("run_id"), run_id, f"{run_id} manifest run_id")
        _assert_equal(
            manifest.get("repository", {}).get("code_tree_sha256"),
            EXPECTED_CODE_TREE_SHA256,
            f"{run_id} frozen code binding",
        )
        benchmark_receipts = manifest.get("benchmark_receipts")
        if not isinstance(benchmark_receipts, list) or not benchmark_receipts:
            raise AuditError(f"{run_id} lacks immutable input receipts")
        for receipt in benchmark_receipts:
            if not isinstance(receipt, Mapping):
                raise AuditError(f"{run_id} has a malformed input receipt")
            receipt_path = ROOT / str(receipt.get("path", ""))
            inventory.add(
                receipt_path,
                category="immutable_input",
                role=f"{run_id} input receipt: {receipt.get('name')}",
                expected_sha256=str(receipt.get("sha256")),
                allow_jsonl_with_json_suffix=True,
                allow_empty=True,
            )
        pair_digest = str(manifest.get("pair_manifest_sha256"))
        inventory.add(
            pairs_path,
            category="immutable_design",
            role=f"randomized treatment cells: {spec['role']}",
            expected_sha256=pair_digest,
        )
        rows_raw = _read_jsonl(pairs_path)
        if any(not isinstance(row, Mapping) for row in rows_raw):
            raise AuditError(f"non-object pair row: {run_id}")
        rows = [dict(row) for row in rows_raw]
        pair_rows[run_id] = rows
        cell_ids = [str(row.get("cell_id")) for row in rows]
        if len(cell_ids) != len(set(cell_ids)):
            raise AuditError(f"duplicate treatment cell: {run_id}")
        try:
            models = sorted({str(row["pair_key"]["model"]) for row in rows})
            arms = sorted({str(row["arm"]) for row in rows})
            operators = sorted({str(row["operator"]) for row in rows})
            task_keys = {
                (
                    str(row["pair_key"]["domain"]),
                    str(row["pair_key"]["task_id"]),
                    str(row["pair_key"]["task_sha256"]),
                )
                for row in rows
            }
            model_task_units = {
                (
                    str(row["pair_key"]["domain"]),
                    str(row["pair_key"]["task_id"]),
                    str(row["pair_key"]["task_sha256"]),
                    str(row["pair_key"]["model"]),
                    int(row["pair_key"]["replicate_id"]),
                )
                for row in rows
            }
            replicates = sorted({int(row["pair_key"]["replicate_id"]) for row in rows})
            benchmarks = sorted({str(row["pair_key"]["domain"]) for row in rows})
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid pair schema: {run_id}") from exc
        for field, actual in (
            ("cells", len(rows)),
            ("tasks", len(task_keys)),
            ("models", len(models)),
            ("arms", len(arms)),
            ("operators", len(operators)),
        ):
            _assert_equal(actual, spec[field], f"{run_id} {field}")
        _assert_equal(replicates, [0], f"{run_id} replicate IDs")
        expected_product = (
            len(task_keys) * len(models) * len(arms) * len(operators) * len(replicates)
        )
        _assert_equal(len(rows), expected_product, f"{run_id} exact treatment product")
        actual_product = {
            (
                (
                    str(row["pair_key"]["domain"]),
                    str(row["pair_key"]["task_id"]),
                    str(row["pair_key"]["task_sha256"]),
                ),
                str(row["pair_key"]["model"]),
                str(row["arm"]),
                str(row["operator"]),
                int(row["pair_key"]["replicate_id"]),
            )
            for row in rows
        }
        expected_product_keys = {
            (task, model, arm, operator, replicate)
            for task in task_keys
            for model in models
            for arm in arms
            for operator in operators
            for replicate in replicates
        }
        _assert_equal(
            actual_product,
            expected_product_keys,
            f"{run_id} complete unique treatment product",
        )
        _assert_equal(sorted(manifest.get("models", [])), models, f"{run_id} model set")
        _assert_equal(sorted(manifest.get("arms", [])), arms, f"{run_id} arm set")
        _assert_equal(
            sorted(manifest.get("operators", [])), operators, f"{run_id} operator set"
        )
        declared_cells = manifest.get("extra_config", {}).get("n_cells")
        if declared_cells is not None:
            _assert_equal(declared_cells, len(rows), f"{run_id} declared cells")
        samples.append(
            {
                "run_id": run_id,
                "role": spec["role"],
                "benchmark": benchmarks,
                "cells_or_trajectories": len(rows),
                "shadow_outputs": SHADOW_OUTPUT_COUNTS[run_id],
                "source_tasks": len(task_keys),
                "model_task_units": len(model_task_units),
                "models": models,
                "methods_or_arms": arms,
                "operators": operators,
                "replicates": replicates,
                "exact_cartesian_product": True,
                "manifest_sha256": spec["manifest_sha256"],
                "pairs_sha256": pair_digest,
            }
        )
    return samples, pair_rows


def _validate_standard_run_results(inventory: Inventory) -> None:
    for label, digest, category, role in (*FIXED_FILES, *FIXED_RESULTS):
        inventory.add(
            ROOT / label,
            category=category,
            role=role,
            expected_sha256=digest,
        )

    validations = (
        ("e12-baseline-evolving-allarms-allmodels-v1", "validation-baseline-exploratory.json", 500, 0),
        ("e12-baseline-bfcl-allarms-fourmodels-v2", "validation-baseline-exploratory.json", 400, 0),
        ("e12-calibration-evolving-core-v2", "validation-calibration.json", 160, 80),
        ("e12-calibration-bfcl-core-v1", "validation-calibration.json", 120, 60),
        ("e12-confirmatory-evolving-core-v2", "validation-confirmatory.json", 448, 224),
        ("e12-confirmatory-bfcl-core-v3", "validation-confirmatory.json", 336, 168),
        (PASS_ONE_RUN, "validation-pass-one.json", 80, 40),
    )
    expected_manifests = {str(spec["run_id"]): str(spec["manifest_sha256"]) for spec in RUN_SPECS}
    expected_pairs = {
        run_id: _sha256(ARTIFACTS / run_id / "pairs.jsonl")
        for run_id, *_rest in validations
    }
    for run_id, filename, cells, shadows in validations:
        path = ARTIFACTS / run_id / "results" / filename
        if "baseline" in run_id:
            category = "exploratory_result"
        elif "calibration" in run_id:
            category = "calibration_result"
        elif run_id == PASS_ONE_RUN:
            category = "sensitivity_result"
        else:
            category = "primary_result"
        inventory.add(
            path,
            category=category,
            role=f"strict validation for {run_id}",
        )
        value = _read_json(path)
        _assert_equal(value.get("primary_ready"), True, f"{run_id} primary_ready")
        for field in ("expected_cells", "valid_trajectories", "trajectory_outputs"):
            _assert_equal(value.get(field), cells, f"{run_id} {field}")
        _assert_equal(value.get("shadow_outputs"), shadows, f"{run_id} shadow outputs")
        _assert_equal(value.get("errors"), [], f"{run_id} validation errors")
        _assert_equal(
            value.get("manifest_sha256"), expected_manifests[run_id], f"{run_id} validation manifest"
        )
        _assert_equal(
            value.get("pair_manifest_sha256"),
            expected_pairs[run_id],
            f"{run_id} validation pairs",
        )
        if run_id == PASS_ONE_RUN:
            warning_codes = [row.get("code") for row in value.get("warnings", [])]
            _assert_equal(
                warning_codes,
                ["call_event.unreferenced"],
                "pass-one disclosed recovery warning",
            )
        else:
            _assert_equal(value.get("warnings"), [], f"{run_id} validation warnings")

    calibration_files = (
        "experiments12/data_results/runs/e12-calibration-evolving-core-v2/results/extract-calibration.json",
        "experiments12/data_results/runs/e12-calibration-bfcl-core-v1/results/extract-calibration.json",
    )
    for label in calibration_files:
        inventory.add(
            ROOT / label,
            category="calibration_result",
            role="calibration analysis extract",
        )

    normalization_path = (
        ARTIFACTS
        / "e12-baseline-evolving-allarms-allmodels-v1/results/analysis-ledger-normalization.json"
    )
    normalization = _read_json(normalization_path)
    _assert_equal(
        normalization.get("artifact_type"),
        "experiment12_analysis_ledger_normalization_receipt",
        "exploratory ledger-normalization receipt type",
    )
    _assert_equal(
        normalization.get("canonical_run_was_modified"),
        False,
        "exploratory canonical run immutability",
    )
    _assert_equal(
        normalization.get("trajectory_or_outcome_bytes_changed"),
        False,
        "exploratory trajectory/outcome immutability",
    )
    _assert_equal(
        normalization.get("cost_or_usage_changed"),
        False,
        "exploratory cost/usage immutability",
    )
    inventory.add(
        normalization_path,
        category="analysis_provenance",
        role="Evolving exploratory ledger-normalization receipt",
    )

    runtime_path = (
        ARTIFACTS / "e12-confirmatory-bfcl-core-v3/results/runtime-environment.json"
    )
    runtime = _read_json(runtime_path)
    _assert_equal(runtime.get("run_id"), "e12-confirmatory-bfcl-core-v3", "BFCL runtime run")
    _assert_equal(
        runtime.get("experiment12_code_tree_sha256"),
        EXPECTED_CODE_TREE_SHA256,
        "BFCL scoring runtime source hash",
    )
    preflight = runtime.get("provider_free_preflight")
    if not isinstance(preflight, Mapping):
        raise AuditError("BFCL scoring runtime lacks provider-free preflight receipt")
    for field in (
        "begin_episode",
        "empty_execute_tools_every_turn",
        "materialize_public_state_every_turn",
        "evaluate_episode",
    ):
        _assert_equal(preflight.get(field), "passed", f"BFCL runtime preflight {field}")
    inventory.add(
        runtime_path,
        category="analysis_provenance",
        role="BFCL confirmatory scoring runtime receipt",
    )

    pass_one_recovery = (
        ARTIFACTS
        / PASS_ONE_RUN
        / "results/recovery/9d8591ea71f67026d743d434"
    )
    if pass_one_recovery.is_symlink() or not pass_one_recovery.is_dir():
        raise AuditError("pass-one recovery archive is missing or linked")
    pass_one_files = sorted(path for path in pass_one_recovery.iterdir() if path.is_file())
    _assert_equal(len(pass_one_files), 6, "pass-one recovery archive file count")
    receipt_path = pass_one_recovery / "recovery-receipt.json"
    receipt = _read_json(receipt_path)
    _assert_equal(
        receipt.get("artifact_type"),
        "experiment12_single_missing_shadow_judge_recovery",
        "pass-one recovery receipt type",
    )
    _assert_equal(receipt.get("run_id"), PASS_ONE_RUN, "pass-one recovery run")
    _assert_equal(
        receipt.get("cell_id"), "9d8591ea71f67026d743d434", "pass-one recovery cell"
    )
    _assert_equal(
        receipt.get("code_tree_sha256"),
        EXPECTED_CODE_TREE_SHA256,
        "pass-one recovery source hash",
    )
    _assert_equal(
        receipt.get("source_code_or_scientific_values_changed"),
        False,
        "pass-one recovery scientific immutability",
    )
    _assert_equal(
        receipt.get("extra_provider_calls_beyond_the_missing_judge"),
        0,
        "pass-one recovery extra calls",
    )
    _assert_equal(
        receipt.get("final", {}).get("validation_primary_ready"),
        True,
        "pass-one recovery final validation",
    )
    for path in pass_one_files:
        inventory.add(
            path,
            category="recovery_provenance",
            role="pass-one trace-judge recovery audit trail",
            expected_sha256=(
                "a699a0f1ab43af0f3cd7a4b2d48bf206e0184ac6e2569ceda0bc9109a4725f42"
                if path == receipt_path
                else None
            ),
        )


def _add_figure_directory(
    inventory: Inventory,
    directory: Path,
    *,
    expected_pairs: int,
    category: str,
    role: str,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise AuditError(f"missing or linked figure directory: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if any(path.is_symlink() for path in files):
        raise AuditError(f"linked figure artifact in {directory}")
    svgs = [path for path in files if path.suffix == ".svg"]
    sidecars = [path for path in files if path.name.endswith(".data.json")]
    extras = [path for path in files if path not in svgs and path not in sidecars]
    if extras:
        raise AuditError(f"unexpected files in figure directory {directory}: {extras}")
    _assert_equal(len(svgs), expected_pairs, f"{directory} SVG count")
    _assert_equal(len(sidecars), expected_pairs, f"{directory} sidecar count")
    svg_stems = {path.stem for path in svgs}
    data_stems = {path.name[: -len(".data.json")] for path in sidecars}
    _assert_equal(data_stems, svg_stems, f"{directory} exact SVG/sidecar pairing")
    for path in [*svgs, *sidecars]:
        inventory.add(path, category=category, role=role)
    return {
        "directory": _relative(directory),
        "svg_count": len(svgs),
        "sidecar_count": len(sidecars),
        "paired": True,
        "file_set_sha256": _sha256_json(
            [
                {"path": _relative(path), "sha256": _sha256(path)}
                for path in sorted([*svgs, *sidecars])
            ]
        ),
    }


def _validate_figures(inventory: Inventory, staging: Path) -> list[dict[str, Any]]:
    specs = (
        (ARTIFACTS / "e12-baseline-evolving-allarms-allmodels-v1/results/figures", 24, "exploratory_result", "Evolving active-probe mechanism figure"),
        (ARTIFACTS / "e12-baseline-bfcl-allarms-fourmodels-v2/results/figures", 24, "exploratory_result", "BFCL active-probe mechanism figure"),
        (ARTIFACTS / "e12-calibration-evolving-core-v2/results/observer-figures", 6, "calibration_result", "Evolving calibration observer figure"),
        (ARTIFACTS / "e12-calibration-bfcl-core-v1/results/observer-figures", 6, "calibration_result", "BFCL calibration observer figure"),
        (ARTIFACTS / "e12-confirmatory-evolving-core-v2/results/observer-figures", 6, "primary_result", "Evolving observer-effect figure"),
        (ARTIFACTS / "e12-confirmatory-bfcl-core-v3/results/observer-figures", 6, "primary_result", "BFCL observer-effect figure"),
        (ARTIFACTS / "e12-confirmatory-evolving-core-v2/results/signal-figures", 4, "primary_result", "Evolving signal precision-recall figure"),
        (ARTIFACTS / "e12-confirmatory-bfcl-core-v3/results/signal-figures", 3, "primary_result", "BFCL primary signal precision-recall figure"),
        (ARTIFACTS / "e12-confirmatory-bfcl-core-v3/results/signal-figures-complete-case", 3, "sensitivity_result", "BFCL complete-case precision-recall figure"),
        (ARTIFACTS / YOKED_RUN / "results/two-pass-figures", 5, "sensitivity_result", "two-pass yoked deployment figure"),
        (staging / "analysis/figures", 1, "primary_result", "online adaptive deployment figure"),
    )
    return [
        _add_figure_directory(
            inventory,
            directory,
            expected_pairs=count,
            category=category,
            role=role,
        )
        for directory, count, category, role in specs
    ]


def _validate_bound_generated_bundle(
    inventory: Inventory,
    *,
    receipt_path: Path,
    expected_outputs: set[str],
    artifact_path: Path,
    expected_artifact: str,
    role: str,
    expected_receipt_sha256: str,
) -> Mapping[str, Any]:
    receipt = _read_json(receipt_path)
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AuditError(f"generated receipt lacks output bindings: {receipt_path}")
    _assert_equal(set(map(str, outputs)), expected_outputs, f"{role} receipt output set")
    _assert_equal(receipt.get("provider_calls_made"), 0, f"{role} provider calls")
    for field in ("code_tree_sha256_before", "code_tree_sha256_after"):
        _assert_equal(receipt.get(field), EXPECTED_CODE_TREE_SHA256, f"{role} {field}")
    builder_path = ROOT / str(receipt.get("builder", ""))
    _assert_equal(
        receipt.get("builder_sha256"),
        _sha256(builder_path),
        f"{role} builder binding",
    )
    inventory.add(
        receipt_path,
        category="paper_derived_result",
        role=f"{role} receipt",
        expected_sha256=expected_receipt_sha256,
    )
    inventory.add(builder_path, category="analysis_provenance", role=f"{role} builder")
    for label, digest in outputs.items():
        inventory.add(
            ROOT / str(label),
            category="paper_derived_result",
            role=role,
            expected_sha256=str(digest),
        )
    artifact = _read_json(artifact_path)
    _assert_equal(artifact.get("artifact"), expected_artifact, f"{role} artifact identity")
    _assert_equal(
        artifact.get("code_tree_sha256"), EXPECTED_CODE_TREE_SHA256, f"{role} source hash"
    )
    return artifact


def _validate_paper_derived_results(inventory: Inventory) -> dict[str, Any]:
    overhead_prefix = "experiments12/data_results/derived/observer-overhead-confirmatory-v1"
    overhead_outputs = {
        f"{overhead_prefix}.json",
        f"{overhead_prefix}.summary.csv",
        f"{overhead_prefix}.tasks.csv",
        f"{overhead_prefix}.svg",
        f"{overhead_prefix}.svg.data.json",
    }
    overhead = _validate_bound_generated_bundle(
        inventory,
        receipt_path=GENERATED / "observer-overhead-confirmatory-v1.receipt.json",
        expected_outputs=overhead_outputs,
        artifact_path=GENERATED / "observer-overhead-confirmatory-v1.json",
        expected_artifact="confirmatory_observer_overhead",
        role="confirmatory observer-compute overhead",
        expected_receipt_sha256="08b5eb493bd65f6ea09efa00fcae2611a74824f94f8e251796c79395114042a9",
    )
    _assert_equal(
        overhead.get("counts"),
        {
            "benchmark_model_strata": 7,
            "benchmarks": 2,
            "methods": 8,
            "model_tasks": 392,
            "task_method_rows": 3136,
        },
        "observer-overhead counts",
    )

    ladder_prefix = "experiments12/data_results/derived/active-probe-ladder-confirmatory-v1"
    ladder_outputs = {
        f"{ladder_prefix}.json",
        f"{ladder_prefix}.csv",
        f"{ladder_prefix}.svg",
        f"{ladder_prefix}.svg.data.json",
    }
    ladder = _validate_bound_generated_bundle(
        inventory,
        receipt_path=GENERATED / "active-probe-ladder-confirmatory-v1.receipt.json",
        expected_outputs=ladder_outputs,
        artifact_path=GENERATED / "active-probe-ladder-confirmatory-v1.json",
        expected_artifact="active_probe_observer_effect_ladder",
        role="active-probe observer-effect ladder",
        expected_receipt_sha256="7721c6545eace98fdaf1106500e478b8e8c8e298325964c021ad29567b617e1a",
    )
    sign_counts = ladder.get("descriptive_sign_counts", {}).get("confirmatory_recompute", {})
    _assert_equal(sign_counts.get("n_strata"), 7, "powered observer-effect strata")
    _assert_equal(
        sign_counts.get("point_sign_counts"),
        {"negative": 6, "positive": 1, "zero": 0},
        "powered observer-effect sign trend",
    )
    confirmatory_sources = [
        row
        for row in ladder.get("source_runs", [])
        if isinstance(row, Mapping) and row.get("study") == "confirmatory_powered"
    ]
    _assert_equal(len(confirmatory_sources), 2, "powered observer-effect benchmark sources")
    powered_models = sorted(
        {
            str(model)
            for row in confirmatory_sources
            for model in row.get("models", [])
        }
    )
    _assert_equal(len(powered_models), 4, "powered observer-effect distinct models")

    provenance_scripts = (
        (GENERATED / "build_observer_overhead12.py", "observer-overhead builder", None),
        (
            GENERATED / "audit_observer_overhead12.py",
            "observer-overhead independent audit",
            "01503e4599116213c8d10d3f11ef16725e908dfbf8c7815b2107ecb7b300ed40",
        ),
        (GENERATED / "build_active_probe_ladder12.py", "active-probe ladder builder", None),
        (
            GENERATED / "audit_active_probe_ladder12.py",
            "active-probe ladder independent audit",
            "060ee476afcfa3abb6388007cf559c1eb4d3ba840b9eddb3767f54119365b423",
        ),
        (GENERATED / "luna_threshold_diagnostic12.py", "threshold-policy diagnostic builder", None),
        (GENERATED / "online_leave_one_unit_sensitivity12.py", "online cumulative n=38 sensitivity generator", None),
        (Path(__file__), "final paper-material inventory generator", None),
    )
    for path, role, digest in provenance_scripts:
        inventory.add(
            path,
            category="analysis_provenance",
            role=role,
            expected_sha256=digest,
        )

    diagnostic_path = GENERATED / "luna-threshold-diagnostic12.json"
    diagnostic = _read_json(diagnostic_path)
    _assert_equal(
        diagnostic.get("artifact_type"),
        "experiment12_provider_free_threshold_policy_diagnostic",
        "threshold diagnostic identity",
    )
    _assert_equal(
        diagnostic.get("provenance", {}).get("code_tree_sha256"),
        EXPECTED_CODE_TREE_SHA256,
        "threshold diagnostic source hash",
    )
    _assert_equal(
        diagnostic.get("provenance", {}).get("provider_calls_made_by_diagnostic"),
        0,
        "threshold diagnostic provider calls",
    )
    inventory.add(
        diagnostic_path,
        category="sensitivity_result",
        role="natural scalar-threshold deployment diagnostic",
    )
    figure_labels = diagnostic.get("figure_files")
    _assert_equal(
        set(figure_labels or []),
        {
            "experiments12/data_results/derived/luna-threshold-rates12.svg",
            "experiments12/data_results/derived/luna-top8-timing12.svg",
        },
        "threshold diagnostic figure set",
    )
    for label in figure_labels:
        inventory.add(
            ROOT / str(label),
            category="sensitivity_result",
            role="natural scalar-threshold policy diagnostic figure",
        )

    interaction_receipt_path = GENERATED / "deployment-interaction-confirmatory-v1.receipt.json"
    interaction_receipt = _read_json(interaction_receipt_path)
    _assert_equal(interaction_receipt.get("artifact_type"), "experiment12_deployment_interaction_receipt", "deployment interaction receipt type")
    _assert_equal(interaction_receipt.get("schema_version"), 1, "deployment interaction receipt version")
    _assert_equal(interaction_receipt.get("provider_calls_made"), 0, "deployment interaction provider calls")
    _assert_equal(interaction_receipt.get("code_tree_sha256"), EXPECTED_CODE_TREE_SHA256, "deployment interaction code binding")
    _assert_equal(interaction_receipt.get("designs_never_pooled"), True, "deployment interaction estimand separation")
    builder_path = ROOT / str(interaction_receipt.get("builder_path", ""))
    _assert_equal(interaction_receipt.get("builder_sha256"), _sha256(builder_path), "deployment interaction builder binding")
    inventory.add(builder_path, category="analysis_provenance", role="deployment interaction builder", expected_sha256="f59f2932954f807c32d6c3a751b558b530469bd93164b00457e1cbba373c03d7")
    inventory.add(GENERATED / "audit_deployment_interaction12.py", category="analysis_provenance", role="deployment interaction independent audit", expected_sha256="a990453d21f03d450b34291e387a3c07945f11b53831d795266cc520ca5946f6")
    inventory.add(interaction_receipt_path, category="paper_derived_result", role="deployment interaction bundle receipt", expected_sha256="8b773e0acd393e8bb95275c8d244703e91c3fa10a13a757b1b46227c57999ae2")
    interaction_outputs = {
        "experiments12/data_results/derived/deployment-interaction-confirmatory-v1.csv": "e621ffe8dbcb96f2f13e178e61a5181a0cc5898108b2b5e269ddb712e2fe6e8d",
        "experiments12/data_results/derived/deployment-interaction-confirmatory-v1.json": "63ee460340ce35c8527eebda22fc80cffe1ac09fa835dc19e4ae646c2721492e",
        "experiments12/data_results/derived/deployment-interaction-confirmatory-v1.svg": "d47dabee6bd291ab6aeffe41cdbe227c3ef8fd4b9f5cc5b5a55ac5bc1e8501c4",
        "experiments12/data_results/derived/deployment-interaction-confirmatory-v1.svg.data.json": "5c44027a5cc89641d0b9e44669cb4236d173cf0ac787bcaf670e2e9837501505",
    }
    _assert_equal(interaction_receipt.get("outputs"), interaction_outputs, "deployment interaction output bindings")
    for label, digest in interaction_outputs.items():
        inventory.add(ROOT / label, category="paper_derived_result", role="confirmatory deployment method-by-operator interaction bundle", expected_sha256=digest)
    for label, digest in interaction_receipt.get("inputs", {}).items():
        _assert_equal(_sha256(ROOT / str(label)), digest, f"deployment interaction input binding {label}")
    interaction = _read_json(GENERATED / "deployment-interaction-confirmatory-v1.json")
    _assert_equal(interaction.get("artifact_type"), "experiment12_deployment_interaction_figure", "deployment interaction artifact type")
    _assert_equal(interaction.get("provider_calls_made"), 0, "deployment interaction artifact provider calls")
    _assert_equal(interaction.get("designs_never_pooled"), True, "deployment interaction panel separation")
    _assert_equal(len(interaction.get("rows", [])), 29, "deployment interaction plotted rows")
    _assert_equal(len(interaction.get("panels", [])), 2, "deployment interaction panels")
    qualitative = interaction.get("leave_two_unit_qualitative_audit", {})
    _assert_equal(qualitative.get("n38_effects_compared"), 21, "deployment success sensitivity coverage")
    _assert_equal(qualitative.get("changed_displayed_effects"), 1, "deployment success sensitivity changes")
    _assert_equal(qualitative.get("any_point_sign_changed"), True, "deployment success point-sign sensitivity")
    _assert_equal(qualitative.get("any_ci_relation_changed"), False, "deployment success CI sensitivity")
    sidecar = _read_json(GENERATED / "deployment-interaction-confirmatory-v1.svg.data.json")
    _assert_equal(sidecar.get("figure_type"), "separate_online_and_yoked_operator_effect_forests", "deployment interaction figure type")
    _assert_equal(sidecar.get("designs_never_pooled"), True, "deployment interaction sidecar separation")
    _assert_equal(len(sidecar.get("rows", [])), 29, "deployment interaction sidecar marks")
    if float(sidecar.get("minimum_text_points_at_target_width", 0)) < 7.0:
        raise AuditError("deployment interaction figure text is below 7pt at target width")
    return {
        "observer_effect_powered_strata": 7,
        "observer_effect_distinct_models": 4,
        "observer_effect_models": powered_models,
        "observer_effect_benchmarks": 2,
        "observer_effect_negative": 6,
        "observer_effect_positive": 1,
        "observer_effect_zero": 0,
        "complexity_ladder_inference": "exploratory_non_monotone",
        "deployment_interaction_rows": 29,
        "deployment_interaction_panels": 2,
        "deployment_success_effect_sensitivity_changes": 1,
        "deployment_success_effect_ci_relation_changes": 0,
    }


def _signal_accuracy_gate() -> dict[str, Any]:
    paths = (
        ARTIFACTS / "e12-confirmatory-evolving-core-v2/results/score-confirmatory.json",
        ARTIFACTS / "e12-confirmatory-bfcl-core-v3/results/score-confirmatory-no-opportunity-v1.json",
    )
    metrics: list[Mapping[str, Any]] = []
    for path in paths:
        value = _read_json(path)
        rows = value.get("metrics")
        if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
            raise AuditError(f"signal score metrics missing: {path}")
        metrics.extend(rows)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in metrics:
        grouped.setdefault((str(row.get("benchmark")), str(row.get("model"))), []).append(row)
    _assert_equal(len(grouped), 7, "powered signal model/benchmark strata")
    active_leads: list[dict[str, Any]] = []
    for (benchmark, model), rows in sorted(grouped.items()):
        active = [row for row in rows if row.get("method") == "active_recompute"]
        _assert_equal(len(active), 1, f"active signal row {benchmark}/{model}")
        values = [float(row["auprc"]) for row in rows]
        active_value = float(active[0]["auprc"])
        if active_value == max(values):
            active_leads.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "active_auprc": active_value,
                    "tied": sum(value == active_value for value in values) > 1,
                }
            )
    _assert_equal(len(active_leads), 3, "strata where active has maximum AUPRC")
    return {
        "powered_strata": len(grouped),
        "active_has_maximum_auprc": len(active_leads),
        "active_leading_strata": active_leads,
        "universal_winner": False,
    }


def _recovery_receipt_path(cell_id: str) -> Path:
    return GENERATED / f"recovery-adaptive-{cell_id}12.json"


def _validate_recoveries(
    inventory: Inventory,
    pair_rows: Mapping[str, list[dict[str, Any]]],
    online_rows: Sequence[Mapping[str, Any]],
    staging_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    online_pairs = {str(row.get("cell_id")): row for row in pair_rows[ONLINE_RUN]}
    analysis_by_cell = {str(row.get("cell_id")): row for row in online_rows}
    receipt_cells: list[dict[str, Any]] = []
    source_units: set[str] = set()
    staging_text = json.dumps(staging_receipt, sort_keys=True)
    for cell_id, treatment in RECOVERY_CELLS.items():
        pair = online_pairs.get(cell_id)
        if pair is None:
            raise AuditError(f"recovery cell absent from frozen pairs: {cell_id}")
        _assert_equal(
            (pair.get("arm"), pair.get("operator")),
            treatment,
            f"recovery treatment identity {cell_id}",
        )
        task_id = str(pair.get("pair_key", {}).get("task_id"))
        replicate_id = pair.get("pair_key", {}).get("replicate_id")
        unit_id = f"{task_id}/r{replicate_id}"
        _assert_equal(task_id, RECOVERY_TASKS[cell_id], f"recovery task {cell_id}")
        _assert_equal(unit_id, f"{RECOVERY_TASKS[cell_id]}/r0", f"recovery unit {cell_id}")
        source_units.add(unit_id)
        row = analysis_by_cell.get(cell_id)
        if row is None:
            raise AuditError(f"recovery cell absent from online analysis: {cell_id}")
        _assert_equal(row.get("unit_id"), unit_id, f"analysis recovery unit {cell_id}")
        _assert_equal(
            (row.get("method"), row.get("operator")),
            treatment,
            f"analysis recovery treatment {cell_id}",
        )
        if cell_id not in staging_text:
            raise AuditError(f"staging receipt omits normalized cell: {cell_id}")

        receipt_path = _recovery_receipt_path(cell_id)
        receipt = _read_json(receipt_path)
        _assert_equal(
            receipt.get("artifact_type"),
            RECOVERY_RECEIPT_TYPES[cell_id],
            f"recovery receipt type {cell_id}",
        )
        _assert_equal(receipt.get("run_id"), ONLINE_RUN, f"recovery run {cell_id}")
        _assert_equal(receipt.get("cell_id"), cell_id, f"recovery cell receipt {cell_id}")
        _assert_equal(
            receipt.get("ledger_rows_deleted_or_rewritten"),
            False,
            f"recovery immutable ledger {cell_id}",
        )
        if cell_id == "89df41e0daa1262a43fa5e55":
            groups = receipt.get("judge_recovery_groups")
            if not isinstance(groups, list) or len(groups) != 1:
                raise AuditError("second recovery lacks its single judge-retry group")
            dispatches = groups[0].get("dispatches", [])
            _assert_equal(len(dispatches), 5, "second recovery physical judge dispatches")
            _assert_equal(
                [row.get("logical_attempt_number") for row in dispatches],
                [1, 2, 3, 4, 5],
                "second recovery logical attempt sequence",
            )
            _assert_equal(
                [row.get("semantic_parse") for row in dispatches],
                ["failed", "failed", "failed", "failed", "succeeded"],
                "second recovery semantic parse sequence",
            )
        inventory.add(
            receipt_path,
            category="recovery_provenance",
            role=f"single-cell semantic recovery receipt {cell_id}",
            expected_sha256=RECOVERY_RECEIPT_SHA256.get(cell_id),
        )
        case_label = receipt.get("case_file")
        if case_label is not None:
            case_path = ROOT / str(case_label)
            inventory.add(
                case_path,
                category="recovery_provenance",
                role=f"predeclared recovery case {cell_id}",
                expected_sha256=str(receipt.get("case_sha256")),
            )

        archive = ARTIFACTS / ONLINE_RUN / "results/recovery" / cell_id
        if archive.is_symlink() or not archive.is_dir():
            raise AuditError(f"missing recovery archive: {cell_id}")
        files = sorted(path for path in archive.iterdir() if path.is_file())
        _assert_equal(len(files), 3, f"recovery archive file count {cell_id}")
        archive_receipt_path = archive / "archive-receipt.json"
        archive_receipt = _read_json(archive_receipt_path)
        _assert_equal(
            archive_receipt.get("artifact_type"),
            "experiment12_online_adaptive_pre_recovery_archive",
            f"recovery archive type {cell_id}",
        )
        _assert_equal(archive_receipt.get("cell_id"), cell_id, f"archive cell {cell_id}")
        _assert_equal(
            archive_receipt.get("archive_created_before_provider_dispatch"),
            True,
            f"archive timing {cell_id}",
        )
        _assert_equal(
            _sha256(archive_receipt_path),
            receipt.get("pre_recovery_archive_receipt_sha256"),
            f"archive receipt binding {cell_id}",
        )
        for path in files:
            inventory.add(
                path,
                category="recovery_provenance",
                role=f"pre-recovery immutable archive {cell_id}",
            )
        partial_path = next((path for path in files if path.name.startswith("partial-events-")), None)
        failed_path = archive / "failed-job.json"
        if partial_path is None:
            raise AuditError(f"recovery archive lacks partial events: {cell_id}")
        _assert_equal(
            _sha256(partial_path),
            receipt.get(
                "archived_partial_events_sha256",
                receipt.get("original_partial_file_sha256"),
            ),
            f"archived partial-event binding {cell_id}",
        )
        _assert_equal(
            _sha256(failed_path),
            receipt.get(
                "archived_failed_job_sha256",
                receipt.get("original_failed_job_sha256"),
            ),
            f"archived failed-job binding {cell_id}",
        )
        production_paths = {
            "final_output_sha256": ARTIFACTS / ONLINE_RUN / "results/adaptive_deployment" / f"{cell_id}.json",
            "final_job_sha256": ARTIFACTS / ONLINE_RUN / "results/adaptive_deployment_jobs" / f"{cell_id}.json",
            "final_event_log_sha256": ARTIFACTS / ONLINE_RUN / "events" / f"adaptive-{cell_id}.jsonl",
        }
        for field, path in production_paths.items():
            _require_regular(path)
            _assert_equal(_sha256(path), receipt.get(field), f"recovered production binding {cell_id}/{field}")
            inventory.add(
                path,
                category="recovery_provenance",
                role=f"recovered canonical production artifact {cell_id}",
            )
        receipt_cells.append(
            {
                "cell_id": cell_id,
                "method": treatment[0],
                "operator": treatment[1],
                "task_id": task_id,
                "unit_id": unit_id,
                "receipt_sha256": _sha256(receipt_path),
                "archive_receipt_sha256": _sha256(archive_receipt_path),
            }
        )
    _assert_equal(source_units, RECOVERY_UNIT_IDS, "complete affected-unit recovery invariant")
    return {
        "cells": receipt_cells,
        "distinct_source_units": sorted(source_units),
        "distinct_source_unit_count": len(source_units),
        "complete_affected_unit_set": True,
        "cumulative_exclusion_required": True,
        "future_recovery_on_new_unit_requires_expanded_cumulative_exclusion": True,
    }


def _validate_online_analysis(
    inventory: Inventory,
    staging: Path,
    pair_rows: Mapping[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    required = {
        "staging_receipt": staging / "staging-receipt.json",
        "analysis_receipt": staging / "analysis-receipt.json",
        "analysis": staging / "analysis/adaptive-analysis.json",
    }
    fixed_hashes = {
        "staging_receipt": "28fbf8b4e14449087c0444bae6522bec42624b7b93380c92c70d12e44bc42f15",
        "analysis_receipt": "dfc03904181b6dc5f48d2ed691cb0889a3d7a03518b583fac59d747e32e0cf65",
        "analysis": "c296291f61b1e0134cac1f68f0d94b2f46286f0710354ed0284c2a454db98b9e",
    }
    values = {name: _read_json(path) for name, path in required.items()}
    for name, path in required.items():
        inventory.add(
            path,
            category="primary_result" if name in {"analysis", "analysis_receipt"} else "recovery_provenance",
            role=f"online staged analysis {name.replace('_', ' ')}",
            expected_sha256=fixed_hashes[name],
        )
    inventory.add(
        GENERATED / "build_adaptive_analysis_staging12.py",
        category="recovery_provenance",
        role="provider-free copy-on-write normalization and analysis builder",
        expected_sha256="792a53a27127482ae890aceae361fc4858d706f58ac928b0244bfd5861b685ce",
    )

    staging_receipt = values["staging_receipt"]
    if not isinstance(staging_receipt, Mapping):
        raise AuditError("online staging receipt is not an object")
    _assert_equal(
        staging_receipt.get("artifact_type"),
        "experiment12_adaptive_analysis_copy_on_write_staging",
        "online staging receipt type",
    )
    _assert_equal(staging_receipt.get("staging_version"), 1, "online staging version")
    _assert_equal(staging_receipt.get("source_run_id"), ONLINE_RUN, "online staging run")
    _assert_equal(staging_receipt.get("declared_cells"), 1120, "online staged cells")
    _assert_equal(staging_receipt.get("provider_calls"), 0, "online staging provider calls")
    _assert_equal(staging_receipt.get("production_inputs_immutable"), True, "online production immutability")
    _assert_equal(
        staging_receipt.get("source_manifest_sha256"),
        ONLINE_MANIFEST_SHA256,
        "staging source manifest",
    )
    _assert_equal(
        staging_receipt.get("source_pair_manifest_sha256"),
        ONLINE_PAIRS_SHA256,
        "staging source pairs",
    )
    _assert_equal(
        staging_receipt.get("source_code_tree_sha256"),
        EXPECTED_CODE_TREE_SHA256,
        "staging source code",
    )
    _assert_equal(
        staging_receipt.get("source_artifacts_root"),
        "experiments12/data_results/runs",
        "online production source root",
    )
    _assert_equal(
        staging_receipt.get("staging_artifacts_root"),
        _relative(staging),
        "online copy-on-write staging root",
    )
    _assert_equal(staging_receipt.get("normalized_recovery_cell_count"), 3, "normalized cell count")
    _assert_equal(staging_receipt.get("normalized_source_unit_count"), 2, "normalized unit count")
    _assert_equal(
        set(staging_receipt.get("normalized_source_units", [])),
        RECOVERY_UNIT_IDS,
        "normalized source-unit set",
    )
    semantics = str(staging_receipt.get("normalization_semantics", ""))
    for text in ("failed logical attempts", "usage", "elapsed time", "cost"):
        if text not in semantics:
            raise AuditError(f"online staging semantics omit {text!r}")

    receipt_rows = staging_receipt.get("recovery_receipts")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != 3:
        raise AuditError("staging receipt lacks all three recovery receipts")
    receipt_index = {str(row.get("cell_id")): row for row in receipt_rows if isinstance(row, Mapping)}
    _assert_equal(set(receipt_index), set(RECOVERY_CELLS), "staging recovery receipt cells")
    for cell_id in RECOVERY_CELLS:
        row = receipt_index[cell_id]
        _assert_equal(row.get("artifact_type"), RECOVERY_RECEIPT_TYPES[cell_id], f"staging recovery type {cell_id}")
        _assert_equal(row.get("sha256"), RECOVERY_RECEIPT_SHA256[cell_id], f"staging recovery hash {cell_id}")
        _assert_equal(row.get("path"), _relative(_recovery_receipt_path(cell_id)), f"staging recovery path {cell_id}")

    normalized_calls = staging_receipt.get("normalized_calls")
    if not isinstance(normalized_calls, list) or len(normalized_calls) != 3:
        raise AuditError("staging receipt must contain exactly three normalized logical calls")
    call_index = {str(row.get("cell_id")): row for row in normalized_calls if isinstance(row, Mapping)}
    _assert_equal(set(call_index), set(RECOVERY_CELLS), "normalized logical-call cells")
    expected_checkpoints = {
        "d52046b6eb74a76ecdc3debc": 5,
        "89df41e0daa1262a43fa5e55": 6,
        "786d95760ccdb86713c26936": 5,
    }
    expected_attempts = {
        "d52046b6eb74a76ecdc3debc": 2,
        "89df41e0daa1262a43fa5e55": 5,
        "786d95760ccdb86713c26936": 2,
    }
    for cell_id, row in call_index.items():
        _assert_equal(row.get("checkpoint"), expected_checkpoints[cell_id], f"normalized checkpoint {cell_id}")
        _assert_equal(row.get("source_unit_id"), f"{RECOVERY_TASKS[cell_id]}/r0", f"normalized source unit {cell_id}")
        _assert_equal(row.get("physical_attempts"), expected_attempts[cell_id], f"normalized physical attempts {cell_id}")
        attempts = row.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != expected_attempts[cell_id]:
            raise AuditError(f"normalized attempt trail changed: {cell_id}")
        _assert_equal(
            [attempt.get("logical_attempt_number") for attempt in attempts],
            list(range(1, expected_attempts[cell_id] + 1)),
            f"normalized logical attempt order {cell_id}",
        )
        _assert_equal(
            [attempt.get("logical_status") for attempt in attempts],
            ["failed"] * (expected_attempts[cell_id] - 1) + ["succeeded"],
            f"normalized semantic outcomes {cell_id}",
        )
        _assert_equal(
            row.get("call_event_ids"),
            [attempt.get("event_id") for attempt in attempts],
            f"normalized call-event order {cell_id}",
        )
        if int(row.get("tokens", 0)) <= 0 or int(row.get("elapsed_ms", 0)) <= 0 or float(row.get("actual_cost_usd", 0)) <= 0:
            raise AuditError(f"normalized call drops physical resources: {cell_id}")
        for attempt in attempts:
            for field in ("production_attempt_sha256", "staged_attempt_sha256"):
                if not isinstance(attempt.get(field), str) or len(str(attempt[field])) != 64:
                    raise AuditError(f"normalized call lacks {field}: {cell_id}")

    repairs = staging_receipt.get("generic_ledger_repairs")
    expected_repair = [{
        "event_id": "d950af6bd8a8421e99f8efc17125fa1b",
        "from_request_status": "unknown",
        "request_key": "e12-deploy-online-evolving-luna-40-v1/b0978e4007c1e796c0521807/adaptive-task-7/attempt-1",
        "reservation_id": "abf82d1b70f9480db3c05659062e0a0b",
        "to_request_status": "failed",
    }]
    _assert_equal(repairs, expected_repair, "generic staged ledger reconciliation")
    _assert_equal(
        staging_receipt.get("raw_analyzer_expected_failure"),
        "call attempt disagrees with ledger: d950af6bd8a8421e99f8efc17125fa1b",
        "documented raw analyzer failure",
    )

    def add_staged_binding(field: str, role: str) -> Path:
        path = ROOT / str(staging_receipt.get(field, ""))
        try:
            path.resolve().relative_to(staging.resolve())
        except ValueError as exc:
            raise AuditError(f"staging path escapes copy-on-write root: {field}") from exc
        inventory.add(
            path,
            category="recovery_provenance",
            role=role,
            expected_sha256=str(staging_receipt.get(field.replace("_path", "_sha256"))),
            allow_empty=field == "staged_ledger_path",
        )
        return path

    add_staged_binding("staged_call_attempts_path", "normalized staged physical call attempts")
    add_staged_binding("staged_ledger_path", "normalized staged budget ledger")
    staged_run = staging / ONLINE_RUN
    inventory.add(staged_run / "manifest.json", category="recovery_provenance", role="staged frozen online manifest", expected_sha256=ONLINE_MANIFEST_SHA256)
    inventory.add(staged_run / "pairs.jsonl", category="recovery_provenance", role="staged frozen online pairs", expected_sha256=ONLINE_PAIRS_SHA256)

    bindings = staging_receipt.get("staged_cell_bindings")
    if not isinstance(bindings, list) or len(bindings) != 3:
        raise AuditError("staging receipt lacks exact recovered-cell bindings")
    binding_index = {str(row.get("cell_id")): row for row in bindings if isinstance(row, Mapping)}
    _assert_equal(set(binding_index), set(RECOVERY_CELLS), "staged recovered-cell bindings")
    for cell_id, binding in binding_index.items():
        output_path = staged_run / "results/adaptive_deployment" / f"{cell_id}.json"
        job_path = staged_run / "results/adaptive_deployment_jobs" / f"{cell_id}.json"
        event_path = staged_run / "events" / f"adaptive-{cell_id}.jsonl"
        for field, path, role in (
            ("staged_output_sha256", output_path, "normalized staged cell output"),
            ("staged_job_sha256", job_path, "normalized staged cell job"),
            ("staged_event_log_sha256", event_path, "normalized staged cell event log"),
        ):
            inventory.add(path, category="recovery_provenance", role=f"{role} {cell_id}", expected_sha256=str(binding.get(field)))
        output = _read_json(output_path)
        _assert_equal(_sha256_json(output.get("accounting")), binding.get("staged_accounting_sha256"), f"staged accounting binding {cell_id}")
        for field in ("old_decision_sha256", "old_signal_record_sha256", "staged_decision_sha256", "staged_signal_record_sha256"):
            if not isinstance(binding.get(field), str) or len(str(binding[field])) != 64:
                raise AuditError(f"staged cell binding lacks digest {field}: {cell_id}")

    analysis_receipt = values["analysis_receipt"]
    _assert_equal(
        analysis_receipt.get("artifact_type"),
        "experiment12_adaptive_analysis_staging_receipt",
        "online analysis receipt type",
    )
    _assert_equal(analysis_receipt.get("analysis_receipt_version"), 1, "online analysis receipt version")
    _assert_equal(analysis_receipt.get("source_run_id"), ONLINE_RUN, "online analysis receipt run")
    _assert_equal(analysis_receipt.get("source_manifest_sha256"), ONLINE_MANIFEST_SHA256, "online receipt manifest")
    _assert_equal(analysis_receipt.get("source_pair_manifest_sha256"), ONLINE_PAIRS_SHA256, "online receipt pairs")
    _assert_equal(analysis_receipt.get("provider_calls"), 0, "online receipt provider calls")
    _assert_equal(analysis_receipt.get("analysis_rows"), 1120, "online receipt rows")
    _assert_equal(analysis_receipt.get("primary_source_tasks_per_treatment"), 40, "online receipt tasks")
    _assert_equal(analysis_receipt.get("treatments"), 28, "online receipt treatments")
    _assert_equal(analysis_receipt.get("normalized_recovery_cell_count"), 3, "online receipt recovered cells")
    _assert_equal(analysis_receipt.get("normalized_source_unit_count"), 2, "online receipt recovered units")
    _assert_equal(
        analysis_receipt.get("staging_receipt_sha256"),
        _sha256(required["staging_receipt"]),
        "online analysis/staging receipt binding",
    )
    _assert_equal(
        analysis_receipt.get("analysis_sha256"),
        _sha256(required["analysis"]),
        "online analysis receipt/output binding",
    )
    _assert_equal(analysis_receipt.get("analysis_path"), _relative(required["analysis"]), "online analysis path binding")
    resource_semantics = str(analysis_receipt.get("resource_semantics", ""))
    for text in ("tokens", "latency", "dollars", "failed retries"):
        if text not in resource_semantics:
            raise AuditError(f"online analysis receipt resource semantics omit {text!r}")
    sensitivity_binding = analysis_receipt.get("sensitivity")
    if not isinstance(sensitivity_binding, Mapping):
        raise AuditError("online analysis receipt lacks sensitivity binding")
    expected_sensitivity_binding = {
        "path": _relative(staging / "analysis/adaptive-analysis-leave-two-units.json"),
        "sha256": "0e2bae5f026be2d44e3c8e9e90986057503f9a8d0ed65eb65e25180d0213a756",
        "markdown_path": _relative(staging / "analysis/adaptive-analysis-leave-two-units.md"),
        "markdown_sha256": "7ec8f718f5a49825a2185b6571638aa91ff0cd1d4f3d58810ca9411b46ddec46",
        "excluded_rows": 56,
        "remaining_rows": 1064,
        "remaining_source_tasks_per_treatment": 38,
        "excluded_source_units": sorted(RECOVERY_UNIT_IDS),
    }
    for field, expected in expected_sensitivity_binding.items():
        _assert_equal(sensitivity_binding.get(field), expected, f"online receipt sensitivity {field}")
    figures = analysis_receipt.get("figures")
    if not isinstance(figures, list) or len(figures) != 2:
        raise AuditError("online analysis receipt figure bundle changed")
    for row in figures:
        path = ROOT / str(row.get("path", ""))
        inventory.add(path, category="figure", role="online deployed method-by-operator figure bundle", expected_sha256=str(row.get("sha256")))

    analysis = values["analysis"]
    _assert_equal(
        analysis.get("artifact_type"),
        "online_adaptive_deployment_analysis",
        "online analysis type",
    )
    _assert_equal(analysis.get("source_run_id"), ONLINE_RUN, "online analysis source run")
    _assert_equal(
        analysis.get("source_manifest_sha256"), ONLINE_MANIFEST_SHA256, "online analysis manifest"
    )
    _assert_equal(
        analysis.get("source_pair_manifest_sha256"), ONLINE_PAIRS_SHA256, "online analysis pairs"
    )
    _assert_equal(analysis.get("deployment_mode"), "online_adaptive", "online deployment mode")
    _assert_equal(analysis.get("statistical_unit"), "source_task", "online statistical unit")
    rows = analysis.get("rows")
    summaries = analysis.get("metric_summaries")
    effects = analysis.get("operator_effects")
    if not all(isinstance(value, list) for value in (rows, summaries, effects)):
        raise AuditError("online analysis arrays are missing")
    if any(not isinstance(row, Mapping) for row in rows):
        raise AuditError("online analysis row is not an object")
    _assert_equal(len(rows), 1120, "online analysis row count")
    _assert_equal(len(summaries), 224, "online metric-summary count")
    _assert_equal(len(effects), 168, "online operator-effect count")
    _assert_equal({row.get("n_tasks") for row in summaries}, {40}, "online summary denominators")
    _assert_equal({row.get("n_tasks") for row in effects}, {40}, "online effect denominators")
    _assert_equal(
        {row.get("bootstrap_iterations") for row in [*summaries, *effects]},
        {2000},
        "online bootstrap iterations",
    )
    _assert_equal(
        {row.get("bootstrap_seed") for row in [*summaries, *effects]},
        {12012},
        "online bootstrap seed",
    )
    cell_ids = {str(row.get("cell_id")) for row in rows}
    pair_cell_ids = {str(row.get("cell_id")) for row in pair_rows[ONLINE_RUN]}
    _assert_equal(cell_ids, pair_cell_ids, "online exact declared-cell coverage")
    units = {str(row.get("unit_id")) for row in rows}
    methods = {str(row.get("method")) for row in rows}
    operators = {str(row.get("operator")) for row in rows}
    _assert_equal(len(units), 40, "online unique source units")
    _assert_equal(len(methods), 7, "online methods")
    _assert_equal(len(operators), 4, "online operators")
    treatment_units: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        expected_unit = f"{row.get('task_id')}/r{row.get('replicate_id')}"
        _assert_equal(row.get("unit_id"), expected_unit, "online unit identity")
        treatment_units.setdefault((str(row.get("method")), str(row.get("operator"))), set()).add(
            str(row.get("unit_id"))
        )
    _assert_equal(len(treatment_units), 28, "online exact method/operator product")
    if any(value != units for value in treatment_units.values()):
        raise AuditError("online treatments do not share the same 40 source units")
    full_summaries, full_effects = summarize_adaptive_outcomes(
        rows,
        bootstrap_iterations=2000,
        bootstrap_seed=12012,
        confidence=0.95,
    )
    _assert_equal(
        summaries,
        [asdict(row) for row in full_summaries],
        "online exact frozen summary reproduction",
    )
    _assert_equal(
        effects,
        [asdict(row) for row in full_effects],
        "online exact frozen effect reproduction",
    )
    return (
        {
            "analysis_path": _relative(required["analysis"]),
            "analysis_sha256": _sha256(required["analysis"]),
            "rows": 1120,
            "source_units": 40,
            "methods": 7,
            "operators": 4,
            "metric_summaries": 224,
            "operator_effects": 168,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 12012,
            "strict_stock_analysis": True,
        },
        staging_receipt,
        rows,
    )


def _validate_online_sensitivity(
    inventory: Inventory,
    staging: Path,
    *,
    analysis_sha256: str,
) -> dict[str, Any]:
    receipt_path = staging / "analysis/adaptive-analysis-leave-two-units.json"
    markdown_path = staging / "analysis/adaptive-analysis-leave-two-units.md"
    receipt = _read_json(receipt_path)
    inventory.add(
        receipt_path,
        category="sensitivity_result",
        role="paired online cumulative leave-two-affected-source-units sensitivity",
        expected_sha256="0e2bae5f026be2d44e3c8e9e90986057503f9a8d0ed65eb65e25180d0213a756",
    )
    inventory.add(
        markdown_path,
        category="sensitivity_result",
        role="paired online cumulative leave-two-affected-source-units sensitivity summary",
        expected_sha256="7ec8f718f5a49825a2185b6571638aa91ff0cd1d4f3d58810ca9411b46ddec46",
    )
    _assert_equal(
        receipt.get("artifact_type"),
        "experiment12_online_adaptive_leave_two_source_units_sensitivity",
        "online sensitivity type",
    )
    _assert_equal(receipt.get("sensitivity_version"), 1, "online sensitivity version")
    _assert_equal(receipt.get("source_run_id"), ONLINE_RUN, "online sensitivity source run")
    _assert_equal(
        receipt.get("source_analysis_sha256"),
        analysis_sha256,
        "online sensitivity source analysis binding",
    )
    _assert_equal(
        receipt.get("source_analysis_path"),
        _relative(staging / "analysis/adaptive-analysis.json"),
        "online sensitivity source analysis path",
    )
    expected_scalars = {
        "excluded_rows": 56,
        "remaining_rows": 1064,
        "remaining_source_tasks_per_treatment": 38,
        "treatments": 28,
        "excluded_rows_per_treatment": 2,
        "balanced_paired_design_after_exclusion": True,
        "exclusion_scope": "both_source_units_from_every_method_operator_treatment",
        "exclusion_reason": "three cells required semantic judge-attempt recovery",
    }
    for field, expected in expected_scalars.items():
        _assert_equal(receipt.get(field), expected, f"online sensitivity {field}")
    _assert_equal(
        set(receipt.get("excluded_source_units", [])),
        RECOVERY_UNIT_IDS,
        "sensitivity omitted units",
    )

    source = _read_json(staging / "analysis/adaptive-analysis.json")
    source_rows = source.get("rows")
    rows = receipt.get("rows")
    summaries = receipt.get("metric_summaries")
    effects = receipt.get("operator_effects")
    if not all(isinstance(value, list) for value in (source_rows, rows, summaries, effects)):
        raise AuditError("online sensitivity lacks source rows, filtered rows, or summaries")
    expected_rows = [row for row in source_rows if str(row.get("unit_id")) not in RECOVERY_UNIT_IDS]
    excluded_rows = [row for row in source_rows if str(row.get("unit_id")) in RECOVERY_UNIT_IDS]
    _assert_equal(len(excluded_rows), 56, "sensitivity exact cumulative exclusions")
    _assert_equal(rows, expected_rows, "sensitivity exact source-row filtering")
    _assert_equal(
        set(receipt.get("excluded_cell_ids", [])),
        {str(row.get("cell_id")) for row in excluded_rows},
        "sensitivity excluded-cell set",
    )
    _assert_equal(len(receipt.get("excluded_cell_ids", [])), 56, "sensitivity excluded-cell count")
    _assert_equal(len(rows), 1064, "n38 row count")
    _assert_equal(len(summaries), 224, "n38 metric summaries")
    _assert_equal(len(effects), 168, "n38 operator effects")
    _assert_equal(
        {row.get("n_tasks") for row in summaries},
        {38},
        "n38 summary denominators",
    )
    _assert_equal(
        {row.get("n_tasks") for row in effects},
        {38},
        "n38 effect denominators",
    )
    _assert_equal(
        {row.get("bootstrap_iterations") for row in [*summaries, *effects]},
        {2000},
        "n38 bootstrap iterations",
    )
    _assert_equal(
        {row.get("bootstrap_seed") for row in [*summaries, *effects]},
        {12012},
        "n38 bootstrap seed",
    )
    treatment_units: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        treatment_units.setdefault((str(row.get("method")), str(row.get("operator"))), set()).add(str(row.get("unit_id")))
    _assert_equal(len(treatment_units), 28, "n38 treatment count")
    remaining_units = set(next(iter(treatment_units.values())))
    _assert_equal(len(remaining_units), 38, "n38 common source-unit count")
    if any(unit_set != remaining_units for unit_set in treatment_units.values()):
        raise AuditError("n38 treatments do not share the exact paired source-unit set")
    filtered_summaries, filtered_effects = summarize_adaptive_outcomes(
        rows,
        bootstrap_iterations=2000,
        bootstrap_seed=12012,
        confidence=0.95,
    )
    _assert_equal(summaries, [asdict(row) for row in filtered_summaries], "n38 exact frozen summary reproduction")
    _assert_equal(effects, [asdict(row) for row in filtered_effects], "n38 exact frozen effect reproduction")

    summary_key = ("model", "benchmark", "observation_class", "method", "operator", "metric")
    effect_key = (*summary_key[:-1], "control_operator", "metric")
    source_summary_index = {tuple(row[field] for field in summary_key): row for row in source.get("metric_summaries", [])}
    filtered_summary_index = {tuple(row[field] for field in summary_key): row for row in summaries}
    source_effect_index = {tuple(row[field] for field in effect_key): row for row in source.get("operator_effects", [])}
    filtered_effect_index = {tuple(row[field] for field in effect_key): row for row in effects}
    _assert_equal(set(source_summary_index), set(filtered_summary_index), "n40/n38 summary comparison coverage")
    _assert_equal(set(source_effect_index), set(filtered_effect_index), "n40/n38 effect comparison coverage")

    def sign(value: float) -> int:
        return 1 if value > 1e-12 else (-1 if value < -1e-12 else 0)

    def interval(row: Mapping[str, Any]) -> str:
        if float(row["ci_low"]) > 1e-12:
            return "positive_excludes_zero"
        if float(row["ci_high"]) < -1e-12:
            return "negative_excludes_zero"
        return "includes_zero"

    material_effects: list[dict[str, Any]] = []
    for key in sorted(source_effect_index):
        old, new = source_effect_index[key], filtered_effect_index[key]
        reasons: list[str] = []
        if sign(float(old["effect"])) != sign(float(new["effect"])):
            reasons.append("point_effect_sign_changed")
        if interval(old) != interval(new):
            reasons.append("confidence_interval_relation_to_zero_changed")
        if reasons:
            material_effects.append({
                **{field: old[field] for field in effect_key},
                "n40_effect": old["effect"],
                "n38_effect": new["effect"],
                "n40_interval": interval(old),
                "n38_interval": interval(new),
                "reasons": reasons,
            })
    material_summaries: list[dict[str, Any]] = []
    for key in sorted(source_summary_index):
        old, new = source_summary_index[key], filtered_summary_index[key]
        old_mean, new_mean = float(old["mean"]), float(new["mean"])
        delta = new_mean - old_mean
        metric = str(old["metric"])
        zero_transition = (abs(old_mean) <= 1e-12) != (abs(new_mean) <= 1e-12)
        material = zero_transition
        if metric in {"success", "selected_actions"}:
            material = material or abs(delta) >= 0.025 - 1e-12
        elif abs(old_mean) > 1e-12:
            material = material or abs(delta) / abs(old_mean) >= 0.05
        if material:
            material_summaries.append({
                **{field: old[field] for field in summary_key},
                "n40_mean": old_mean,
                "n38_mean": new_mean,
                "delta": delta,
            })
    return {
        "receipt_path": _relative(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "omitted_unit_ids": sorted(RECOVERY_UNIT_IDS),
        "recovery_cells": sorted(RECOVERY_CELLS),
        "rows": "1120 -> 1064",
        "paired_source_units": "40 -> 38",
        "removed_rows": 56,
        "removed_treatments": 28,
        "success_operator_effects_compared": 21,
        "key_resource_operator_effects_compared": 105,
        "all_metric_summaries_compared": 224,
        "all_operator_effects_compared": 168,
        "material_operator_effect_changes": material_effects,
        "material_absolute_summary_shifts": material_summaries,
        "exact_frozen_recomputation": True,
    }


def _validate_yoked(
    inventory: Inventory,
    pair_rows: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    root = ARTIFACTS / YOKED_RUN / "results"
    validation_path = root / "validation-two-pass.json"
    analysis_path = root / "two-pass-analysis.json"
    validation = _read_json(validation_path)
    analysis = _read_json(analysis_path)
    inventory.add(
        validation_path,
        category="sensitivity_result",
        role="two-pass yoked deployment strict validation",
    )
    inventory.add(
        analysis_path,
        category="sensitivity_result",
        role="two-pass yoked deployment analysis",
    )
    _assert_equal(
        validation.get("artifact_type"), "two_pass_deployment_validation", "yoked validation type"
    )
    _assert_equal(validation.get("primary_ready"), True, "yoked primary_ready")
    for field in (
        "expected_cells",
        "valid_outputs",
        "valid_jobs",
        "valid_event_logs",
        "canonical_regraded_cells",
    ):
        _assert_equal(validation.get(field), 480, f"yoked validation {field}")
    _assert_equal(validation.get("cached_official_cells"), 0, "yoked cached official cells")
    _assert_equal(
        validation.get("source_manifest_sha256"), YOKED_MANIFEST_SHA256, "yoked manifest binding"
    )
    _assert_equal(
        validation.get("source_schedule_sha256"),
        "fa6ebd579a58369d13343c22870d3772fa8c4f4ddc1b07e2e3120f23a92f635f",
        "yoked schedule binding",
    )
    _assert_equal(
        analysis.get("artifact_type"), "two_pass_deployment_analysis", "yoked analysis type"
    )
    _assert_equal(analysis.get("source_run_id"), YOKED_RUN, "yoked analysis source run")
    _assert_equal(
        analysis.get("source_manifest_sha256"),
        YOKED_MANIFEST_SHA256,
        "yoked analysis manifest binding",
    )
    _assert_equal(
        analysis.get("source_pair_manifest_sha256"),
        validation.get("source_pair_manifest_sha256"),
        "yoked analysis pair binding",
    )
    _assert_equal(
        analysis.get("source_schedule_sha256"),
        validation.get("source_schedule_sha256"),
        "yoked analysis schedule binding",
    )
    _assert_equal(analysis.get("deployment_mode"), "two_pass_frozen", "yoked mode")
    _assert_equal(analysis.get("statistical_unit"), "source_task", "yoked statistical unit")
    rows = analysis.get("rows", [])
    summaries = analysis.get("metric_summaries", [])
    operator_effects = analysis.get("operator_effects", [])
    method_effects = analysis.get("method_effects", [])
    _assert_equal(len(rows), 480, "yoked analysis rows")
    _assert_equal(len(summaries), 168, "yoked metric summaries")
    _assert_equal(len(operator_effects), 112, "yoked operator effects")
    _assert_equal(len(method_effects), 252, "yoked method effects")
    _assert_equal({row.get("n_tasks") for row in summaries}, {40}, "yoked summary denominators")
    _assert_equal(
        {row.get("n_tasks") for row in [*operator_effects, *method_effects]},
        {40},
        "yoked effect denominators",
    )
    _assert_equal(
        {str(row.get("cell_id")) for row in rows},
        {str(row.get("cell_id")) for row in pair_rows[YOKED_RUN]},
        "yoked exact declared-cell coverage",
    )
    _assert_equal(len({row.get("unit_id") for row in rows}), 40, "yoked source units")
    _assert_equal(len({row.get("method") for row in rows}), 4, "yoked methods")
    _assert_equal(len({row.get("operator") for row in rows}), 3, "yoked operators")

    table_dir = root / "two-pass-tables"
    if table_dir.is_symlink() or not table_dir.is_dir():
        raise AuditError("missing yoked table directory")
    tables = sorted(path for path in table_dir.iterdir() if path.is_file())
    _assert_equal(len(tables), 4, "yoked CSV table count")
    if any(path.suffix != ".csv" for path in tables):
        raise AuditError("unexpected non-CSV yoked table")
    for path in tables:
        inventory.add(path, category="sensitivity_result", role="two-pass yoked result table")
    return {
        "validation_path": _relative(validation_path),
        "validation_sha256": _sha256(validation_path),
        "analysis_path": _relative(analysis_path),
        "analysis_sha256": _sha256(analysis_path),
        "rows": 480,
        "source_units": 40,
        "methods": 4,
        "operators": 3,
        "metric_summaries": 168,
        "operator_effects": 112,
        "method_effects": 252,
        "canonical_regraded_cells": 480,
        "cached_official_cells": 0,
        "resource_scope": "pass_two_only",
    }


def _budget_snapshot(inventory: Inventory) -> dict[str, Any]:
    database = ARTIFACTS / "_global_budget.sqlite3"
    _require_regular(database)
    before = database.stat()
    wal = Path(str(database) + "-wal")
    wal_before = wal.stat() if wal.exists() else None
    uri = database.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        limits = [dict(row) for row in connection.execute("SELECT * FROM provider_limits ORDER BY provider")]
        expected_limits = {
            "fireworks": (30_000_000, 24_000_000),
            "openai": (500_000_000, 400_000_000),
        }
        actual_limits = {
            str(row["provider"]): (
                int(row["hard_cap_micro_usd"]),
                int(row["operational_cap_micro_usd"]),
            )
            for row in limits
        }
        _assert_equal(actual_limits, expected_limits, "global provider budget limits")
        columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(reservations)")]
        required_columns = {
            "reservation_id",
            "provider",
            "purpose",
            "request_key",
            "state",
            "reserved_micro_usd",
            "actual_micro_usd",
            "cost_quality",
            "request_status",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "provider_total_tokens",
            "provider_request_id",
            "created_at",
            "updated_at",
        }
        if not required_columns.issubset(columns):
            raise AuditError("global ledger reservation schema changed")
        ordered_columns = [column for column in columns if column in required_columns]
        select_columns = ", ".join(f'"{column}"' for column in ordered_columns)
        logical = hashlib.sha256()
        row_count = 0
        for row in connection.execute(
            f"SELECT {select_columns} FROM reservations ORDER BY reservation_id"
        ):
            logical.update(_canonical(dict(row)))
            logical.update(b"\n")
            row_count += 1
        state_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT provider, state, COUNT(*) AS requests,
                       SUM(reserved_micro_usd) AS reserved_micro_usd,
                       SUM(COALESCE(actual_micro_usd, 0)) AS actual_micro_usd
                FROM reservations
                GROUP BY provider, state
                ORDER BY provider, state
                """
            )
        ]
        active = sum(int(row["requests"]) for row in state_rows if row["state"] == "reserved")
        _assert_equal(active, 0, "active budget reservations")
        _assert_equal(
            {str(row["provider"]) for row in state_rows},
            set(expected_limits),
            "providers represented in global ledger",
        )
        final_states = {str(row["state"]) for row in state_rows}
        if "reconciled" not in final_states or not final_states.issubset(
            {"reconciled", "released"}
        ):
            raise AuditError(
                f"final global ledger has an invalid active/unknown state set: {sorted(final_states)}"
            )
        status_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT provider, COALESCE(request_status, '<null>') AS request_status,
                       COUNT(*) AS requests
                FROM reservations
                GROUP BY provider, request_status
                ORDER BY provider, request_status
                """
            )
        ]
        quality_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT provider, COALESCE(cost_quality, '<null>') AS cost_quality,
                       COUNT(*) AS requests
                FROM reservations
                GROUP BY provider, cost_quality
                ORDER BY provider, cost_quality
                """
            )
        ]
        _assert_equal(
            sum(int(row["requests"]) for row in status_rows),
            row_count,
            "request-status accounting coverage",
        )
        _assert_equal(
            sum(int(row["requests"]) for row in quality_rows),
            row_count,
            "cost-quality accounting coverage",
        )
        provider_rows: list[dict[str, Any]] = []
        for provider, (hard, operational) in expected_limits.items():
            spent = sum(
                int(row["actual_micro_usd"] or 0)
                for row in state_rows
                if row["provider"] == provider and row["state"] == "reconciled"
            )
            reserved = sum(
                int(row["reserved_micro_usd"] or 0)
                for row in state_rows
                if row["provider"] == provider and row["state"] == "reserved"
            )
            if spent > hard:
                raise AuditError(f"{provider} actual spend exceeds hard cap")
            provider_rows.append(
                {
                    "provider": provider,
                    "hard_cap_usd": f"{hard / 1_000_000:.6f}",
                    "operational_cap_usd": f"{operational / 1_000_000:.6f}",
                    "actual_spent_usd": f"{spent / 1_000_000:.6f}",
                    "active_reserved_usd": f"{reserved / 1_000_000:.6f}",
                    "hard_remaining_usd": f"{(hard - spent - reserved) / 1_000_000:.6f}",
                    "operational_remaining_usd": f"{(operational - spent - reserved) / 1_000_000:.6f}",
                }
            )
        connection.commit()
    except sqlite3.Error as exc:
        raise AuditError(f"cannot audit global budget ledger: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()
    after = database.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise AuditError("global budget database changed during audit")
    wal_after = wal.stat() if wal.exists() else None
    if (wal_before is None) != (wal_after is None) or (
        wal_before is not None
        and wal_after is not None
        and (wal_before.st_ino, wal_before.st_size, wal_before.st_mtime_ns)
        != (wal_after.st_ino, wal_after.st_size, wal_after.st_mtime_ns)
    ):
        raise AuditError("global budget WAL changed during audit")
    db_entry = inventory.add(
        database,
        category="budget_receipt",
        role="global provider budget ledger",
    )
    wal_entry: dict[str, Any] | None = None
    if wal.exists():
        wal_entry = inventory.add(
            wal,
            category="budget_receipt",
            role="global provider budget ledger WAL",
            allow_empty=True,
        )
    return {
        "database_path": db_entry["path"],
        "database_sha256": db_entry["sha256"],
        "wal_path": None if wal_entry is None else wal_entry["path"],
        "wal_sha256": None if wal_entry is None else wal_entry["sha256"],
        "logical_reservation_rows_sha256": logical.hexdigest(),
        "reservation_rows": row_count,
        "active_reservations": 0,
        "providers": provider_rows,
        "state_counts": state_rows,
        "request_status_counts": status_rows,
        "cost_quality_counts": quality_rows,
    }


def _claim_gates(
    *,
    observer: Mapping[str, Any],
    signal: Mapping[str, Any],
    online: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    yoked: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "gate": "active_vs_passive_definitions",
            "status": "supported",
            "scope": "active carries probe work inside the task trajectory; passive reads an immutable trace with zero carry",
        },
        {
            "gate": "carried_probe_observer_effect",
            "status": "supported_as_trend_not_rule",
            "scope": f"negative point effect in {observer['observer_effect_negative']}/{observer['observer_effect_powered_strata']} powered model/benchmark strata; one positive exception",
        },
        {
            "gate": "signal_accuracy_tradeoffs",
            "status": "supported_no_universal_winner",
            "scope": f"active has maximum AUPRC in {signal['active_has_maximum_auprc']}/{signal['powered_strata']} powered strata",
        },
        {
            "gate": "online_deployed_method_by_operator_outcomes",
            "status": "supported",
            "scope": f"{online['source_units']} paired tasks, seven methods, four operators, one model, Evolving Intent only",
        },
        {
            "gate": "cumulative_affected_unit_recovery_sensitivity",
            "status": "supported",
            "scope": (
                "all 28 treatments re-estimated after cumulatively omitting two "
                f"affected source units (n=40 to n=38): {', '.join(sensitivity['omitted_unit_ids'])}"
            ),
        },
        {
            "gate": "frozen_two_pass_yoked_schedule",
            "status": "supported_as_sensitivity",
            "scope": f"{yoked['source_units']} same source tasks under checkpoint-1 yoked scheduling; pass-two resources only",
        },
        {
            "gate": "active_probe_complexity_ladder",
            "status": "exploratory_only",
            "scope": "non-monotone descriptive ladder; no monotonic complexity claim",
        },
        {
            "gate": "generative_good_bad_feedback_claim",
            "status": "prohibited_by_implementation",
            "scope": "operator is deterministic WATCH-only exact-quote feedback",
        },
        {
            "gate": "uniform_twenty_percent_deployment_claim",
            "status": "prohibited_by_policy",
            "scope": "natural scalar cutoffs yield unequal firing rates, including 100%",
        },
        {
            "gate": "cross_benchmark_deployment_claim",
            "status": "prohibited_by_design",
            "scope": "no BFCL deployment run",
        },
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Experiment 12 final paper-material inventory",
        "",
        "**FINAL RELEASE GATE: PASS.** Every required artifact and scientific scope check passed; this audit made zero provider calls.",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Frozen Experiment12 source: `{report['frozen_code_tree_sha256']}`",
        f"- Files inventoried: {report['inventory']['file_count']}",
        f"- Inventory digest: `{report['inventory']['entries_sha256']}`",
        "",
        "## Sample sizes",
        "",
        "| Run | Cells | Shadows | Tasks | Models | Arms | Operators | Role |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["sample_sizes"]:
        lines.append(
            "| {run_id} | {cells_or_trajectories} | {shadow_outputs} | {source_tasks} | {models} | {arms} | {operators} | {role} |".format(
                run_id=row["run_id"],
                cells_or_trajectories=row["cells_or_trajectories"],
                shadow_outputs=row["shadow_outputs"],
                source_tasks=row["source_tasks"],
                models=len(row["models"]),
                arms=len(row["methods_or_arms"]),
                operators=len(row["operators"]),
                role=row["role"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim gates",
            "",
            "| Claim | Status | Licensed scope |",
            "|---|---|---|",
        ]
    )
    for row in report["claim_gates"]:
        lines.append(f"| {row['gate']} | **{row['status']}** | {row['scope']} |")
    lines.extend(["", "## Budget closure", "", "| Provider | Spent | Hard cap | Hard remaining | Active reserved |", "|---|---:|---:|---:|---:|"])
    for row in report["budget"]["providers"]:
        lines.append(
            f"| {row['provider']} | ${row['actual_spent_usd']} | ${row['hard_cap_usd']} | ${row['hard_remaining_usd']} | ${row['active_reserved_usd']} |"
        )
    lines.extend(
        [
            "",
            "## Recovery and robustness",
            "",
            "- Three recovered cells map to exactly two frozen source units: "
            + ", ".join(f"`{unit}`" for unit in report["recovery"]["distinct_source_units"])
            + ".",
            f"- Paired cumulative sensitivity removes 56 rows (two per treatment): {report['online_sensitivity']['paired_source_units']} source units.",
            f"- Online analysis: {report['online_primary']['rows']} rows, {report['online_primary']['metric_summaries']} summaries, {report['online_primary']['operator_effects']} paired operator effects.",
            f"- Yoked sensitivity: {report['yoked_sensitivity']['rows']} rows; resource scope is `{report['yoked_sensitivity']['resource_scope']}`.",
            "",
            "## Required caveats",
            "",
        ]
    )
    lines.extend(f"{index}. {text}" for index, text in enumerate(report["required_caveats"], 1))
    lines.extend(
        [
            "",
            "## SHA256 inventory",
            "",
            "The JSON companion contains format-specific metadata (JSON keys, CSV headers/rows, SVG validation, and all budget counts).",
            "",
            "| Category | Path | SHA256 |",
            "|---|---|---|",
        ]
    )
    for row in report["inventory"]["files"]:
        lines.append(f"| {row['category']} | `{row['path']}` | `{row['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def build_report(*, staging: Path) -> dict[str, Any]:
    current_code_hash = code_tree_hash(EXPERIMENT)
    _assert_equal(current_code_hash, EXPECTED_CODE_TREE_SHA256, "frozen Experiment12 source tree")
    inventory = Inventory()
    samples, pair_rows = _manifest_and_sample_sizes(inventory)
    _validate_standard_run_results(inventory)
    observer = _validate_paper_derived_results(inventory)
    signal = _signal_accuracy_gate()
    online, staging_receipt, online_rows = _validate_online_analysis(inventory, staging, pair_rows)
    recovery = _validate_recoveries(inventory, pair_rows, online_rows, staging_receipt)
    online_sensitivity = _validate_online_sensitivity(
        inventory, staging, analysis_sha256=str(online["analysis_sha256"])
    )
    yoked = _validate_yoked(inventory, pair_rows)
    figure_sets = _validate_figures(inventory, staging)
    budget = _budget_snapshot(inventory)

    backbone_paths = (
        (ROOT / "README.md", "user-authored paper backbone"),
        (EXPERIMENT / "README.md", "Experiment12 technical README"),
        (GENERATED / "paper-readiness-map12.json", "pre-analysis claim-readiness map"),
        (GENERATED / "PAPER_READINESS12.md", "pre-analysis claim-readiness summary"),
    )
    for path, role in backbone_paths:
        inventory.add(path, category="paper_backbone", role=role)
    readiness_map = _read_json(GENERATED / "paper-readiness-map12.json")
    mandatory_sensitivity = readiness_map.get(
        "online_semantic_retry_normalization", {}
    ).get("mandatory_sensitivity", {})
    staging_builder_sha = _sha256(GENERATED / "build_adaptive_analysis_staging12.py")
    _assert_equal(
        mandatory_sensitivity.get("builder_sha256"),
        staging_builder_sha,
        "readiness-map staging/sensitivity builder binding",
    )
    _assert_equal(mandatory_sensitivity.get("source_analysis_sha256"), online["analysis_sha256"], "readiness-map online analysis binding")
    _assert_equal(mandatory_sensitivity.get("sensitivity_sha256"), online_sensitivity["receipt_sha256"], "readiness-map sensitivity binding")
    _assert_equal(mandatory_sensitivity.get("analysis_receipt_sha256"), _sha256(staging / "analysis-receipt.json"), "readiness-map analysis receipt binding")
    _assert_equal(
        mandatory_sensitivity.get("denominator"),
        "40 -> 38 paired source tasks",
        "readiness-map cumulative sensitivity denominator",
    )
    _assert_equal(
        set(mandatory_sensitivity.get("receipt_outputs", [])),
        {
            "adaptive-analysis-leave-two-units.json",
            "adaptive-analysis-leave-two-units.md",
        },
        "readiness-map cumulative sensitivity outputs",
    )
    readiness_markdown = (GENERATED / "PAPER_READINESS12.md").read_text(
        encoding="utf-8"
    )
    for required_text in (
        staging_builder_sha,
        str(online["analysis_sha256"]),
        str(online_sensitivity["receipt_sha256"]),
        "cumulative paired n=38 sensitivity",
        "adaptive-analysis-leave-two-units.json",
        "786d95760ccdb86713c26936",
    ):
        if required_text not in readiness_markdown:
            raise AuditError(
                f"readiness Markdown lacks current cumulative-sensitivity binding: {required_text}"
            )

    entries = inventory.ordered()
    claims = _claim_gates(
        observer=observer,
        signal=signal,
        online=online,
        sensitivity=online_sensitivity,
        yoked=yoked,
    )
    return {
        "schema_version": 1,
        "artifact_type": "experiment12_final_paper_material_inventory",
        "generated_at_utc": _utc_now(),
        "provider_calls_made": 0,
        "release_gate": "pass",
        "all_required_gates_pass": True,
        "frozen_code_tree_sha256": current_code_hash,
        "paper_title": "Active and passive observation methods for reasoning and action agent traces",
        "sample_sizes": samples,
        "observer_effect": observer,
        "signal_accuracy": signal,
        "online_primary": online,
        "recovery": recovery,
        "online_sensitivity": online_sensitivity,
        "yoked_sensitivity": yoked,
        "figure_sets": figure_sets,
        "budget": budget,
        "claim_gates": claims,
        "required_caveats": list(CAVEATS),
        "inventory": {
            "file_count": len(entries),
            "entries_sha256": _sha256_json(entries),
            "files": entries,
        },
        "fail_closed_policy": (
            "No final JSON or Markdown is replaced unless every fixed hash, receipt "
            "binding, exact sample/product count, complete affected-unit recovery invariant, "
            "sensitivity denominator, figure pairing, and budget gate passes."
        ),
    }


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--staging-base",
        default=str(DEFAULT_STAGING),
        help="completed immutable-source online analysis staging directory",
    )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = build_report(staging=Path(args.staging_base))
        json_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        markdown_text = _markdown(report)
        _atomic_write(OUTPUT_JSON, json_text)
        _atomic_write(OUTPUT_MD, markdown_text)
    except (
        AttributeError,
        AuditError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"paper-material release gate FAILED; final files not written: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "release_gate": "pass",
                "json": _relative(OUTPUT_JSON),
                "json_sha256": _sha256(OUTPUT_JSON),
                "markdown": _relative(OUTPUT_MD),
                "markdown_sha256": _sha256(OUTPUT_MD),
                "provider_calls_made": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
