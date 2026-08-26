"""Fail-closed, read-only validation of Experiment 12 run artifacts.

The pair manifest is the denominator: validation never silently intersects the
cells that happened to finish.  A report is machine-readable and
``primary_ready`` is true only when every declared trajectory is present once
and every integrity/accounting check succeeds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from experiments12.bfcl_runner12 import (
    BFCL_RUNNER_VERSION,
    validate_bfcl_task_turn_record,
)
from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.schemas import CallAttemptRecord
from experiments12.harness12 import ARM_TO_PROBE
from experiments12.manifest12 import RunLayout, validate_manifest_files
from experiments12.pairing12 import JobCell


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CELL_ID = re.compile(r"^[0-9a-f]{24}$")
_BLOCK_ID = re.compile(r"^[0-9a-f]{20}$")
_GOLD_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "evaluation_label",
        "final_answer",
        "gold",
        "gold_answer",
        "gold_label",
        "ground_truth",
        "ground_truth_label",
        "label",
        "target_answer",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One stable, machine-readable validation finding."""

    code: str
    message: str
    path: str | None = None
    cell_id: str | None = None
    subject: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {"code": self.code, "message": self.message}
        if self.path is not None:
            result["path"] = self.path
        if self.cell_id is not None:
            result["cell_id"] = self.cell_id
        if self.subject is not None:
            result["subject"] = self.subject
        return result


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Complete audit result; warnings never hide a primary-analysis error."""

    run_id: str | None
    manifest_sha256: str | None
    pair_manifest_sha256: str | None
    expected_cells: int
    trajectory_outputs: int
    valid_trajectories: int
    shadow_outputs: int
    call_events: int
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]

    @property
    def primary_ready(self) -> bool:
        return (
            self.expected_cells > 0
            and self.valid_trajectories == self.expected_cells
            and not self.errors
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "pair_manifest_sha256": self.pair_manifest_sha256,
            "expected_cells": self.expected_cells,
            "trajectory_outputs": self.trajectory_outputs,
            "valid_trajectories": self.valid_trajectories,
            "shadow_outputs": self.shadow_outputs,
            "call_events": self.call_events,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
            "primary_ready": self.primary_ready,
        }


class _Findings:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.errors: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []

    def display(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except (OSError, ValueError):
            return path.name

    def error(
        self,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        cell_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        self.errors.append(
            ValidationIssue(
                code,
                message,
                None if path is None else self.display(path),
                cell_id,
                subject,
            )
        )

    def warning(
        self,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        cell_id: str | None = None,
        subject: str | None = None,
    ) -> None:
        self.warnings.append(
            ValidationIssue(
                code,
                message,
                None if path is None else self.display(path),
                cell_id,
                subject,
            )
        )

    @staticmethod
    def _sort(issues: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
        return tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.code,
                    item.path or "",
                    item.cell_id or "",
                    item.subject or "",
                    item.message,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class _EventLog:
    path: Path
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _Trajectory:
    path: Path
    value: Mapping[str, Any]


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _declared_task_identity(cell: JobCell) -> tuple[str, str | None]:
    """Return the harness task ID and optional condition encoded by the runner."""

    pair_task_id = cell.pair_key.task_id
    if "::" not in pair_task_id:
        return pair_task_id, None
    source_task_id, condition = pair_task_id.rsplit("::", 1)
    return source_task_id, condition


def _json_object(path: Path, findings: _Findings, code: str) -> Mapping[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        findings.error(code, f"could not read a complete JSON object: {type(exc).__name__}", path=path)
        return None
    if not _is_object(value):
        findings.error(code, "artifact must be a JSON object", path=path)
        return None
    return value


def _validate_manifest(
    layout: RunLayout,
    repository_root: Path,
    findings: _Findings,
    expected_manifest_sha256: str | None,
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    manifest_hash: str | None = None
    pair_hash: str | None = None
    if not layout.manifest.is_file():
        findings.error("manifest.missing", "run manifest is missing", path=layout.manifest)
        return None, None, None
    if layout.manifest.is_symlink():
        findings.error("manifest.symlink", "run manifest must not be a symlink", path=layout.manifest)
    try:
        manifest_hash = sha256_file(layout.manifest)
    except OSError as exc:
        findings.error("manifest.unreadable", f"could not hash manifest: {type(exc).__name__}", path=layout.manifest)
        return None, None, None
    if expected_manifest_sha256 is None:
        findings.warning(
            "manifest.unpinned",
            "no external manifest SHA256 was supplied; internal frozen hashes were still checked",
            path=layout.manifest,
        )
    elif not _valid_sha(expected_manifest_sha256):
        findings.error("manifest.expected_hash_invalid", "expected manifest hash is not lowercase SHA256")
    elif manifest_hash != expected_manifest_sha256:
        findings.error(
            "manifest.hash_mismatch",
            "manifest bytes do not match the externally pinned SHA256",
            path=layout.manifest,
        )
    manifest = _json_object(layout.manifest, findings, "manifest.invalid")
    if manifest is None:
        return None, manifest_hash, None
    if manifest.get("run_id") != layout.root.name:
        findings.error(
            "manifest.run_id_mismatch",
            "manifest run_id differs from the run-layout directory",
            path=layout.manifest,
        )
    for key in ("models", "arms", "operators"):
        values = manifest.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
            or len(values) != len(set(values))
        ):
            findings.error("manifest.invalid", f"{key} must be a non-empty unique string list", path=layout.manifest)
    if manifest.get("secret_values_recorded") is not False:
        findings.error("manifest.secret_flag", "manifest must explicitly record that no secrets are present", path=layout.manifest)
    if not layout.pairs.is_file():
        findings.error("pairs.missing", "declared pair manifest is missing", path=layout.pairs)
        return manifest, manifest_hash, None
    if layout.pairs.is_symlink():
        findings.error("pairs.symlink", "pair manifest must not be a symlink", path=layout.pairs)
    try:
        pair_hash = sha256_file(layout.pairs)
    except OSError as exc:
        findings.error("pairs.unreadable", f"could not hash pair manifest: {type(exc).__name__}", path=layout.pairs)
        return manifest, manifest_hash, None
    if pair_hash != manifest.get("pair_manifest_sha256"):
        findings.error("pairs.hash_mismatch", "pair manifest bytes differ from the frozen manifest hash", path=layout.pairs)
    try:
        legacy_errors = validate_manifest_files(
            manifest,
            repository_root=repository_root,
            pair_manifest_path=layout.pairs,
        )
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        findings.error("manifest.integrity_check_failed", f"manifest integrity check failed: {type(exc).__name__}", path=layout.manifest)
    else:
        codes = {
            "unsupported manifest version": "manifest.version",
            "pair manifest hash mismatch": "pairs.hash_mismatch",
            "Experiment 12 code/config tree changed": "manifest.code_hash_mismatch",
            "model price catalog changed": "manifest.model_catalog_hash_mismatch",
        }
        existing = {item.code for item in findings.errors}
        for message in legacy_errors:
            code = codes.get(message, "manifest.integrity")
            if code not in existing:
                findings.error(code, message, path=layout.manifest)
                existing.add(code)
    return manifest, manifest_hash, pair_hash


def _pair_cells(
    layout: RunLayout,
    manifest: Mapping[str, Any] | None,
    findings: _Findings,
) -> tuple[JobCell, ...]:
    if not layout.pairs.is_file():
        return ()
    try:
        rows = read_jsonl(layout.pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        findings.error("pairs.invalid_jsonl", f"pair manifest is incomplete or invalid: {type(exc).__name__}", path=layout.pairs)
        return ()
    if not rows:
        findings.error("pairs.empty", "pair manifest contains no declared cells", path=layout.pairs)
        return ()
    cells: list[JobCell] = []
    for index, row in enumerate(rows, 1):
        if not _is_object(row):
            findings.error("pairs.invalid_cell", f"line {index} is not an object", path=layout.pairs)
            continue
        try:
            cell = JobCell.from_dict(row)
        except (TypeError, ValueError) as exc:
            findings.error("pairs.invalid_cell", f"line {index} has an invalid cell: {exc}", path=layout.pairs)
            continue
        if _CELL_ID.fullmatch(cell.cell_id) is None:
            findings.error("pairs.invalid_cell_id", "cell_id must be 24 lowercase hex characters", path=layout.pairs, cell_id=cell.cell_id)
        if _BLOCK_ID.fullmatch(cell.block_id) is None:
            findings.error("pairs.invalid_block_id", "block_id must be 20 lowercase hex characters", path=layout.pairs, cell_id=cell.cell_id)
        if not _valid_sha(cell.pair_key.task_sha256):
            findings.error("pairs.invalid_task_hash", "pair-key task_sha256 must be lowercase SHA256", path=layout.pairs, cell_id=cell.cell_id)
        if "::" in cell.pair_key.task_id:
            source_task_id, condition = _declared_task_identity(cell)
            if not source_task_id or not condition or "::" in source_task_id:
                findings.error("pairs.invalid_task_id", "condition-aware task_id must be source_id::condition", path=layout.pairs, cell_id=cell.cell_id)
        cells.append(cell)
    ids = [cell.cell_id for cell in cells]
    for cell_id in sorted(item for item, count in Counter(ids).items() if count > 1):
        findings.error("pairs.duplicate_cell", "cell_id occurs more than once in pair manifest", path=layout.pairs, cell_id=cell_id)
    if manifest is not None:
        models = manifest.get("models")
        arms = manifest.get("arms")
        operators = manifest.get("operators")
        if isinstance(models, list) and isinstance(arms, list) and isinstance(operators, list):
            for cell in cells:
                if cell.pair_key.model not in models:
                    findings.error("pairs.foreign_model", "cell model is absent from run manifest", cell_id=cell.cell_id, subject=cell.pair_key.model)
                if cell.arm not in arms:
                    findings.error("pairs.foreign_arm", "cell arm is absent from run manifest", cell_id=cell.cell_id, subject=cell.arm)
                if cell.operator not in operators:
                    findings.error("pairs.foreign_operator", "cell operator is absent from run manifest", cell_id=cell.cell_id, subject=cell.operator)
            expected_treatments = {(arm, operator) for arm in arms for operator in operators}
            blocks: dict[str, list[JobCell]] = {}
            for cell in cells:
                blocks.setdefault(cell.block_id, []).append(cell)
            for block_id, block in blocks.items():
                treatments = [(cell.arm, cell.operator) for cell in block]
                if set(treatments) != expected_treatments or len(treatments) != len(expected_treatments):
                    findings.error("pairs.incomplete_block", "block is not the exact arm-by-operator treatment set", subject=block_id)
                if sorted(cell.block_position for cell in block) != list(range(len(block))):
                    findings.error("pairs.invalid_block_positions", "block positions must be contiguous from zero", subject=block_id)
                pair_ids = {cell.pair_key.stable_id for cell in block}
                task_hashes = {cell.pair_key.task_sha256 for cell in block}
                if len(pair_ids) != 1 or len(task_hashes) != 1:
                    findings.error("pairs.mixed_block", "one block contains multiple pair keys or task hashes", subject=block_id)
        extra = manifest.get("extra_config")
        if isinstance(extra, Mapping) and extra.get("n_cells") is not None:
            if isinstance(extra.get("n_cells"), bool) or extra.get("n_cells") != len(rows):
                findings.error("pairs.count_mismatch", "manifest n_cells differs from pair manifest length", path=layout.pairs)
    return tuple(cells)


def _scan_event_logs(
    layout: RunLayout,
    expected_ids: set[str],
    findings: _Findings,
) -> tuple[dict[str, _EventLog], dict[str, tuple[Path, CallAttemptRecord]]]:
    starts: dict[str, _EventLog] = {}
    calls: dict[str, tuple[Path, CallAttemptRecord]] = {}
    roots = ((layout.events, True), (layout.shadow, False))
    for root, strict in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if path.is_symlink():
                findings.error("events.symlink", "event log must not be a symlink", path=path)
                continue
            try:
                raw_records = read_jsonl(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                findings.error("events.invalid_jsonl", f"event log is incomplete or invalid: {type(exc).__name__}", path=path)
                continue
            if not raw_records:
                findings.error("events.empty", "event log is empty", path=path)
                continue
            records = tuple(item for item in raw_records if _is_object(item))
            if len(records) != len(raw_records):
                findings.error("events.invalid_record", "event log contains a non-object record", path=path)
                continue
            if records[0].get("event") == "start":
                cell_id = records[0].get("cell_id")
                if not isinstance(cell_id, str):
                    findings.error("events.start_missing_cell", "trajectory event log start lacks cell_id", path=path)
                    continue
                if cell_id not in expected_ids:
                    findings.error("events.foreign_cell", "trajectory event log is not declared", path=path, cell_id=cell_id)
                    continue
                if cell_id in starts:
                    findings.error("events.duplicate_cell", "multiple trajectory event logs claim the same cell", path=path, cell_id=cell_id)
                    continue
                starts[cell_id] = _EventLog(path, records)
                continue
            recognized = False
            for record in records:
                if "event_id" in record or "reservation_id" in record:
                    recognized = True
                    try:
                        attempt = CallAttemptRecord.from_dict(record)
                    except (KeyError, TypeError, ValueError) as exc:
                        findings.error("call_event.invalid", f"invalid call-attempt record: {exc}", path=path)
                        continue
                    if attempt.event_id in calls:
                        findings.error("call_event.duplicate", "call event ID occurs more than once", path=path, subject=attempt.event_id)
                        continue
                    calls[attempt.event_id] = (path, attempt)
                elif "method" in record and "source_trajectory_sha256" in record:
                    recognized = True
                elif strict:
                    findings.error("events.unrecognized", "event log is neither a trajectory log nor a call-attempt log", path=path)
                    recognized = True
            if strict and not recognized:
                findings.error("events.unrecognized", "event log has no recognized records", path=path)
    for cell_id in sorted(expected_ids - set(starts)):
        findings.error("events.missing_cell", "declared cell has no trajectory event log", cell_id=cell_id)
    return starts, calls


def _nested_gold_paths(value: Any, prefix: str = "task_records") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            path = f"{prefix}.{key}"
            if normalized in _GOLD_KEYS:
                found.append(path)
            # Benchmark messages, native tool payloads, and public environment
            # state may legitimately use a parameter named "answer".  The
            # invariant concerns evaluator metadata attached to task records,
            # not the observed transcript or executor payload.
            if normalized not in {
                "user_message",
                "assistant_message",
                "messages",
                "tool_executions",
                "tool_results",
                "public_state_json",
            }:
                found.extend(_nested_gold_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_nested_gold_paths(child, f"{prefix}[{index}]"))
    return tuple(found)


def _message_pair(record: Mapping[str, Any]) -> tuple[Any, Any]:
    return record.get("user_message"), record.get("assistant_message")


def _record_messages(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Exact public turn history, including native tool calls when present."""

    messages = record.get("messages")
    if messages is None:
        return _message_pair(record)
    if isinstance(messages, list):
        return tuple(messages)
    # Keep validation read-only and fail closed via an impossible reconstruction.
    return (None,)


def _validate_record_calls(
    records: Sequence[Any],
    *,
    path: Path,
    cell_id: str,
    findings: _Findings,
    referenced_calls: dict[str, list[str]],
) -> None:
    for index, record in enumerate(records):
        if not _is_object(record):
            continue
        call = record.get("call")
        if not _is_object(call):
            findings.error("trajectory.call_missing", "task/probe record lacks call metadata", path=path, cell_id=cell_id, subject=str(index))
            continue
        ids = call.get("call_event_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
            findings.error("trajectory.call_ids_invalid", "call_event_ids must be a non-empty string list", path=path, cell_id=cell_id, subject=str(index))
            continue
        if len(ids) != len(set(ids)):
            findings.error("trajectory.call_ids_duplicate", "one call record repeats a call event ID", path=path, cell_id=cell_id, subject=str(index))
        for event_id in ids:
            referenced_calls.setdefault(event_id, []).append(f"trajectory:{cell_id}:{index}")


def _checkpoint_list(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value
    ):
        return None
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        return None
    return result


def _validate_event_log(
    log: _EventLog,
    output: Mapping[str, Any],
    cell: JobCell,
    manifest: Mapping[str, Any] | None,
    findings: _Findings,
) -> None:
    records = log.records
    start = records[0]
    start_hash = start.get("design_sha256")
    design = {key: value for key, value in start.items() if key not in {"event", "design_sha256"}}
    if not _valid_sha(start_hash) or sha256_json(design) != start_hash:
        findings.error("trajectory.design_hash_mismatch", "start-event design hash does not match its content", path=log.path, cell_id=cell.cell_id)
    expected_run = None if manifest is None else manifest.get("run_id")
    expected = {
        "run_id": expected_run,
        "cell_id": cell.cell_id,
        "model": cell.pair_key.model,
        "arm": cell.arm,
    }
    for key, value in expected.items():
        if value is not None and start.get(key) != value:
            findings.error("trajectory.design_mismatch", f"start-event {key} differs from declaration", path=log.path, cell_id=cell.cell_id, subject=key)
    task = start.get("task")
    if not _is_object(task):
        findings.error("trajectory.design_task_missing", "start-event task provenance is missing", path=log.path, cell_id=cell.cell_id)
    else:
        source_task_id, condition = _declared_task_identity(cell)
        expected_task = {
            "domain": cell.pair_key.domain,
            "task_id": source_task_id,
            "task_sha256": cell.pair_key.task_sha256,
        }
        if condition is not None:
            expected_task["condition"] = condition
        for key, value in expected_task.items():
            if task.get(key) != value:
                findings.error("trajectory.design_task_mismatch", f"start-event task {key} differs from pair manifest", path=log.path, cell_id=cell.cell_id, subject=key)
        for hash_key in ("source_sha256", "task_sha256"):
            if not _valid_sha(task.get(hash_key)):
                findings.error("trajectory.design_task_hash_invalid", f"start-event task {hash_key} is invalid", path=log.path, cell_id=cell.cell_id, subject=hash_key)
    if output.get("design_sha256") != start_hash:
        findings.error("trajectory.design_hash_mismatch", "trajectory design hash differs from its start event", path=log.path, cell_id=cell.cell_id)
    allowed = {"start", "task_turn", "active_probe", "complete"}
    kinds = [record.get("event") for record in records]
    if any(kind not in allowed for kind in kinds):
        findings.error("events.unknown_event", "trajectory event log contains an unknown event", path=log.path, cell_id=cell.cell_id)
    completes = [record for record in records if record.get("event") == "complete"]
    if len(completes) != 1 or records[-1].get("event") != "complete":
        findings.error("events.complete_invalid", "trajectory log must end with exactly one complete event", path=log.path, cell_id=cell.cell_id)
    task_events = [record for record in records if record.get("event") == "task_turn"]
    probe_events = [record for record in records if record.get("event") == "active_probe"]
    if output.get("task_records") != task_events:
        findings.error("trajectory.task_events_mismatch", "materialized task records differ from append-only events", path=log.path, cell_id=cell.cell_id)
    if output.get("probe_records") != probe_events:
        findings.error("trajectory.probe_events_mismatch", "materialized probe records differ from append-only events", path=log.path, cell_id=cell.cell_id)
    timeline_messages: list[Any] = []
    for record in records[1:]:
        if record.get("event") in {"task_turn", "active_probe"}:
            timeline_messages.extend(
                _record_messages(record)
                if record.get("event") == "task_turn"
                else _message_pair(record)
            )
    if output.get("messages") != timeline_messages:
        findings.error("trajectory.messages_mismatch", "materialized messages differ from append-only task/probe events", path=log.path, cell_id=cell.cell_id)
    if completes:
        if completes[0].get("transcript_sha256") != output.get("transcript_sha256"):
            findings.error("trajectory.complete_hash_mismatch", "complete-event transcript hash differs from materialized output", path=log.path, cell_id=cell.cell_id)


def _validate_trajectory(
    trajectory: _Trajectory,
    cell: JobCell,
    event_log: _EventLog | None,
    manifest: Mapping[str, Any] | None,
    findings: _Findings,
    referenced_calls: dict[str, list[str]],
) -> bool:
    before = len(findings.errors)
    path, value = trajectory.path, trajectory.value
    expected_run = None if manifest is None else manifest.get("run_id")
    source_task_id, declared_condition = _declared_task_identity(cell)
    required = {
        "complete": True,
        "cell_id": cell.cell_id,
        "model": cell.pair_key.model,
        "domain": cell.pair_key.domain,
        "task_id": source_task_id,
        "task_sha256": cell.pair_key.task_sha256,
        "arm": cell.arm,
    }
    if expected_run is not None:
        required["run_id"] = expected_run
    if declared_condition is not None:
        required["condition"] = declared_condition
    for key, expected in required.items():
        if value.get(key) != expected:
            findings.error("trajectory.declaration_mismatch", f"trajectory {key} differs from declared cell", path=path, cell_id=cell.cell_id, subject=key)
    if not _valid_sha(value.get("task_sha256")):
        findings.error("trajectory.task_hash_invalid", "trajectory task_sha256 is invalid", path=path, cell_id=cell.cell_id)
    if not isinstance(value.get("condition"), str) or not value.get("condition"):
        findings.error("trajectory.condition_invalid", "trajectory condition must be a non-empty string", path=path, cell_id=cell.cell_id)
    task_records = value.get("task_records")
    probe_records = value.get("probe_records")
    messages = value.get("messages")
    if not isinstance(task_records, list) or not task_records:
        findings.error("trajectory.task_records_invalid", "task_records must be a non-empty list", path=path, cell_id=cell.cell_id)
        task_records = []
    if not isinstance(probe_records, list):
        findings.error("trajectory.probe_records_invalid", "probe_records must be a list", path=path, cell_id=cell.cell_id)
        probe_records = []
    if not isinstance(messages, list):
        findings.error("trajectory.messages_invalid", "messages must be a list", path=path, cell_id=cell.cell_id)
        messages = []
    task_turns: list[Any] = []
    for record in task_records:
        if not _is_object(record) or record.get("event") != "task_turn":
            findings.error("trajectory.task_records_invalid", "task_records may contain only task_turn objects", path=path, cell_id=cell.cell_id)
            continue
        task_turns.append(record.get("task_turn"))
    if task_turns != list(range(1, len(task_records) + 1)):
        findings.error("trajectory.task_turns_invalid", "task turns must be contiguous from one", path=path, cell_id=cell.cell_id)
    if cell.pair_key.domain == "bfcl_multi_turn":
        if value.get("bfcl_runner_version") != BFCL_RUNNER_VERSION:
            findings.error(
                "trajectory.bfcl_runner_version_invalid",
                "BFCL trajectory does not use the current bounded-agent schema",
                path=path,
                cell_id=cell.cell_id,
            )
        start_config = (
            event_log.records[0].get("config")
            if event_log is not None and event_log.records
            else None
        )
        max_batches = (
            start_config.get("max_tool_batches_per_turn")
            if _is_object(start_config)
            else None
        )
        if (
            isinstance(max_batches, bool)
            or not isinstance(max_batches, int)
            or max_batches < 1
        ):
            findings.error(
                "trajectory.bfcl_tool_batch_limit_invalid",
                "BFCL start event lacks a valid frozen tool-batch limit",
                path=path,
                cell_id=cell.cell_id,
            )
        else:
            for record in task_records:
                if not _is_object(record):
                    continue
                try:
                    validate_bfcl_task_turn_record(
                        record,
                        max_tool_batches_per_turn=max_batches,
                    )
                except ValueError as exc:
                    findings.error(
                        "trajectory.bfcl_turn_termination_invalid",
                        f"BFCL bounded-agent turn record is invalid: {exc}",
                        path=path,
                        cell_id=cell.cell_id,
                        subject=str(record.get("task_turn")),
                    )
    gold_paths = _nested_gold_paths(task_records)
    for gold_path in gold_paths:
        findings.error("trajectory.gold_in_task_records", "task_records contain a forbidden gold-label key", path=path, cell_id=cell.cell_id, subject=gold_path)
    _validate_record_calls(task_records, path=path, cell_id=cell.cell_id, findings=findings, referenced_calls=referenced_calls)
    _validate_record_calls(probe_records, path=path, cell_id=cell.cell_id, findings=findings, referenced_calls=referenced_calls)
    assistant_messages = [
        record.get("assistant_message", {}).get("content")
        for record in task_records
        if _is_object(record) and _is_object(record.get("assistant_message"))
    ]
    if value.get("task_assistant_messages") != assistant_messages:
        findings.error("trajectory.assistant_messages_mismatch", "task_assistant_messages do not match task_records", path=path, cell_id=cell.cell_id)
    checkpoints = _checkpoint_list(value.get("checkpoint_turns"))
    if checkpoints is None:
        findings.error("trajectory.checkpoints_invalid", "checkpoint_turns must be sorted unique positive integers", path=path, cell_id=cell.cell_id)
        checkpoints = ()
    elif task_records and any(turn >= len(task_records) for turn in checkpoints):
        findings.error("trajectory.checkpoints_invalid", "checkpoints must occur before the final task turn", path=path, cell_id=cell.cell_id)
    expected_probe = ARM_TO_PROBE.get(cell.arm)
    if cell.arm == "clean":
        if value.get("active_probe_variant") is not None:
            findings.error("trajectory.clean_probe_variant", "clean trajectory declares an active probe variant", path=path, cell_id=cell.cell_id)
        if probe_records:
            findings.error("trajectory.clean_has_probe", "clean trajectory contains active_probe records", path=path, cell_id=cell.cell_id)
        expected_messages: list[Any] = []
        for record in task_records:
            if _is_object(record):
                expected_messages.extend(_record_messages(record))
        if messages != expected_messages:
            findings.error("trajectory.clean_has_extra_messages", "clean messages are not exactly the task exchanges", path=path, cell_id=cell.cell_id)
    else:
        if expected_probe is None:
            findings.error("trajectory.unknown_active_arm", "non-clean arm has no frozen active-probe mapping", path=path, cell_id=cell.cell_id)
        if value.get("active_probe_variant") != expected_probe:
            findings.error("trajectory.active_variant_mismatch", "active probe variant differs from frozen arm design", path=path, cell_id=cell.cell_id)
        actual_checkpoints: list[tuple[Any, Any, Any]] = []
        active_timeline: list[Any] = []
        probes_by_turn: dict[int, Mapping[str, Any]] = {}
        for record in probe_records:
            if not _is_object(record) or record.get("event") != "active_probe":
                findings.error("trajectory.probe_records_invalid", "probe_records may contain only active_probe objects", path=path, cell_id=cell.cell_id)
                continue
            actual_checkpoints.append(
                (record.get("after_task_turn"), record.get("checkpoint_index"), record.get("variant"))
            )
            turn = record.get("after_task_turn")
            if isinstance(turn, int) and not isinstance(turn, bool):
                probes_by_turn[turn] = record
        declared = [
            (turn, index, expected_probe)
            for index, turn in enumerate(checkpoints, 1)
        ]
        if actual_checkpoints != declared:
            findings.error("trajectory.active_checkpoints_mismatch", "active probes are not exactly at declared checkpoints", path=path, cell_id=cell.cell_id)
        # The active signal is observed only after the probe response has
        # entered target history. Reconstruct that exact prefix for both the
        # scripted and native-tool record shapes and verify its content address.
        for turn, task_record in enumerate(task_records, 1):
            if _is_object(task_record):
                active_timeline.extend(_record_messages(task_record))
            probe_record = probes_by_turn.get(turn)
            if probe_record is None:
                continue
            active_timeline.extend(_message_pair(probe_record))
            prefix_sha256 = probe_record.get("source_prefix_sha256")
            if not _valid_sha(prefix_sha256):
                findings.error(
                    "trajectory.probe_prefix_hash_invalid",
                    "active probe source_prefix_sha256 must be lowercase SHA256",
                    path=path,
                    cell_id=cell.cell_id,
                    subject=str(turn),
                )
            elif prefix_sha256 != sha256_json(active_timeline):
                findings.error(
                    "trajectory.probe_prefix_hash_mismatch",
                    "active probe source_prefix_sha256 does not match carried history",
                    path=path,
                    cell_id=cell.cell_id,
                    subject=str(turn),
                )
    if not _valid_sha(value.get("transcript_sha256")) or sha256_json(messages) != value.get("transcript_sha256"):
        findings.error("trajectory.transcript_hash_mismatch", "transcript_sha256 does not match canonical messages", path=path, cell_id=cell.cell_id)
    if event_log is not None:
        _validate_event_log(event_log, value, cell, manifest, findings)
    return event_log is not None and len(findings.errors) == before


def _trajectory_outputs(
    layout: RunLayout,
    cells: Sequence[JobCell],
    event_logs: Mapping[str, _EventLog],
    manifest: Mapping[str, Any] | None,
    findings: _Findings,
    referenced_calls: dict[str, list[str]],
) -> tuple[dict[str, _Trajectory], int, dict[str, list[_Trajectory]], dict[Path, str]]:
    expected = {cell.cell_id: cell for cell in cells}
    claimed: dict[str, list[_Trajectory]] = {}
    files: list[Path] = []
    if not layout.trajectories.is_dir():
        findings.error("trajectories.missing_directory", "trajectory output directory is missing", path=layout.trajectories)
    else:
        for path in sorted(item for item in layout.trajectories.rglob("*") if item.is_file() or item.is_symlink()):
            if path.suffix != ".json":
                findings.error("trajectory.foreign_file", "trajectory directory contains a non-JSON output", path=path)
                continue
            files.append(path)
            if path.is_symlink():
                findings.error("trajectory.symlink", "trajectory output must not be a symlink", path=path)
                continue
            value = _json_object(path, findings, "trajectory.invalid_json")
            if value is None:
                continue
            cell_id = value.get("cell_id")
            if not isinstance(cell_id, str):
                findings.error("trajectory.missing_cell_id", "trajectory output lacks a string cell_id", path=path)
                continue
            item = _Trajectory(path, value)
            claimed.setdefault(cell_id, []).append(item)
    for cell_id, outputs in sorted(claimed.items()):
        if cell_id not in expected:
            for output in outputs:
                findings.error("trajectory.foreign_cell", "trajectory output is not declared in pair manifest", path=output.path, cell_id=cell_id)
        if len(outputs) > 1:
            for output in outputs:
                findings.error("trajectory.duplicate_cell", "multiple outputs claim the same cell", path=output.path, cell_id=cell_id)
    for cell_id in sorted(set(expected) - set(claimed)):
        findings.error("trajectory.missing_cell", "declared cell has no trajectory output", cell_id=cell_id)
    accepted: dict[str, _Trajectory] = {}
    valid = 0
    file_hashes: dict[Path, str] = {}
    clean_by_transcript: dict[str, list[_Trajectory]] = {}
    for cell_id, cell in expected.items():
        outputs = claimed.get(cell_id, [])
        if len(outputs) != 1:
            continue
        output = outputs[0]
        accepted[cell_id] = output
        try:
            file_hashes[output.path] = sha256_file(output.path)
        except OSError as exc:
            findings.error("trajectory.unreadable", f"could not hash trajectory file: {type(exc).__name__}", path=output.path, cell_id=cell_id)
        if _validate_trajectory(output, cell, event_logs.get(cell_id), manifest, findings, referenced_calls):
            valid += 1
        if output.value.get("arm") == "clean" and _valid_sha(output.value.get("transcript_sha256")):
            clean_by_transcript.setdefault(output.value["transcript_sha256"], []).append(output)
    return accepted, valid, clean_by_transcript, file_hashes


def _shadow_outputs(
    layout: RunLayout,
    clean_by_transcript: Mapping[str, Sequence[_Trajectory]],
    findings: _Findings,
    referenced_calls: dict[str, list[str]],
) -> int:
    if not layout.shadow.exists():
        return 0
    paths = sorted(layout.shadow.rglob("*.json"))
    for path in paths:
        if path.is_symlink():
            findings.error("shadow.symlink", "shadow output must not be a symlink", path=path)
            continue
        value = _json_object(path, findings, "shadow.invalid_json")
        if value is None:
            continue
        source_hash = value.get("source_trajectory_sha256")
        if not _valid_sha(source_hash):
            findings.error("shadow.source_hash_invalid", "shadow source hash is not lowercase SHA256", path=path)
            continue
        sources = clean_by_transcript.get(source_hash, ())
        if not sources:
            findings.error("shadow.source_missing", "shadow output does not source-hash an existing clean trajectory", path=path, subject=source_hash)
            continue
        if value.get("complete") is not True:
            findings.error("shadow.incomplete", "shadow output is not complete", path=path, subject=source_hash)
        if any(key in value for key in ("messages", "task_records", "probe_records", "active_probe")):
            findings.error("shadow.carries_messages", "shadow output must not contain or replace target-history fields", path=path, subject=source_hash)
        metadata_matches = [
            source
            for source in sources
            if all(
                value.get(key) == source.value.get(key)
                for key in ("model", "domain", "task_id", "condition")
            )
        ]
        if not metadata_matches:
            findings.error("shadow.source_metadata_mismatch", "shadow identity does not match its clean source", path=path, subject=source_hash)
        records = value.get("records")
        if not isinstance(records, list):
            findings.error("shadow.records_invalid", "shadow records must be a list", path=path, subject=source_hash)
            continue
        for index, record in enumerate(records):
            if not _is_object(record):
                findings.error("shadow.records_invalid", "shadow record is not an object", path=path, subject=str(index))
                continue
            if record.get("source_trajectory_sha256") != source_hash:
                findings.error("shadow.record_source_mismatch", "shadow record source hash differs from output", path=path, subject=str(index))
            if any(key in record for key in ("messages", "task_records", "probe_records", "active_probe")):
                findings.error("shadow.carries_messages", "shadow record contains target-history fields", path=path, subject=str(index))
            if "call" in record:
                _validate_record_calls((record,), path=path, cell_id=f"shadow:{source_hash[:12]}", findings=findings, referenced_calls=referenced_calls)
    return len(paths)


def _ledger_rows(path: Path, findings: _Findings) -> dict[str, Mapping[str, Any]] | None:
    if not path.is_file():
        findings.error("ledger.missing", "global budget ledger is missing", path=path)
        return None
    if path.is_symlink():
        findings.error("ledger.symlink", "global budget ledger must not be a symlink", path=path)
        return None
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT reservation_id, provider, purpose, request_key, state FROM reservations"
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        findings.error("ledger.invalid", f"could not read reservation ledger: {type(exc).__name__}", path=path)
        return None
    return {str(row["reservation_id"]): dict(row) for row in rows}


def _validate_accounting(
    layout: RunLayout,
    calls: Mapping[str, tuple[Path, CallAttemptRecord]],
    references: Mapping[str, Sequence[str]],
    findings: _Findings,
    run_id: str | None,
) -> None:
    for event_id, locations in sorted(references.items()):
        if len(locations) > 1:
            findings.error("call_event.reused", "call event ID is referenced by multiple materialized calls", subject=event_id)
        if event_id not in calls:
            findings.error("call_event.missing", "materialized call event ID has no append-only call-attempt record", subject=event_id)
    for event_id, (path, _attempt) in sorted(calls.items()):
        if event_id not in references:
            findings.warning("call_event.unreferenced", "call-attempt event is not referenced by a complete materialized output", path=path, subject=event_id)
    rows = _ledger_rows(layout.ledger, findings)
    if rows is None:
        return
    for event_id in sorted(set(references) & set(calls)):
        path, attempt = calls[event_id]
        row = rows.get(attempt.reservation_id)
        if row is None:
            findings.error("call_event.ledger_missing", "call event reservation is absent from global ledger", path=path, subject=event_id)
            continue
        if row.get("state") != "reconciled":
            findings.error("call_event.ledger_unreconciled", "call event reservation is not reconciled", path=path, subject=event_id)
        if row.get("provider") != attempt.provider or row.get("purpose") != attempt.purpose:
            findings.error("call_event.ledger_mismatch", "call event provider/purpose differs from ledger reservation", path=path, subject=event_id)
        request_key = row.get("request_key")
        if run_id is not None and (
            not isinstance(request_key, str) or not request_key.startswith(run_id + "/")
        ):
            findings.error("call_event.ledger_scope_mismatch", "ledger request key is outside this run's namespace", path=path, subject=event_id)


def validate_run(
    layout: RunLayout,
    *,
    repository_root: str | Path,
    expected_manifest_sha256: str | None = None,
) -> ValidationReport:
    """Audit one run without modifying it or inferring an analysis denominator."""

    if not isinstance(layout, RunLayout):
        raise TypeError("layout must be a RunLayout")
    root = layout.root
    findings = _Findings(root)
    manifest, manifest_hash, pair_hash = _validate_manifest(
        layout,
        Path(repository_root),
        findings,
        expected_manifest_sha256,
    )
    cells = _pair_cells(layout, manifest, findings)
    expected_ids = {cell.cell_id for cell in cells}
    event_logs, calls = _scan_event_logs(layout, expected_ids, findings)
    referenced_calls: dict[str, list[str]] = {}
    _accepted, valid, clean_by_transcript, source_file_hashes = _trajectory_outputs(
        layout,
        cells,
        event_logs,
        manifest,
        findings,
        referenced_calls,
    )
    shadow_count = _shadow_outputs(
        layout,
        clean_by_transcript,
        findings,
        referenced_calls,
    )
    for path, before_hash in source_file_hashes.items():
        try:
            after_hash = sha256_file(path)
        except OSError as exc:
            findings.error("trajectory.changed_during_validation", f"clean/source trajectory became unreadable: {type(exc).__name__}", path=path)
        else:
            if after_hash != before_hash:
                findings.error("trajectory.changed_during_validation", "trajectory bytes changed during validation", path=path)
    run_id = manifest.get("run_id") if manifest is not None and isinstance(manifest.get("run_id"), str) else None
    _validate_accounting(layout, calls, referenced_calls, findings, run_id)
    trajectory_count = sum(
        1
        for path in layout.trajectories.rglob("*.json")
        if path.is_file() or path.is_symlink()
    ) if layout.trajectories.exists() else 0
    return ValidationReport(
        run_id=run_id,
        manifest_sha256=manifest_hash,
        pair_manifest_sha256=pair_hash,
        expected_cells=len(cells),
        trajectory_outputs=trajectory_count,
        valid_trajectories=valid,
        shadow_outputs=shadow_count,
        call_events=len(calls),
        errors=findings._sort(findings.errors),
        warnings=findings._sort(findings.warnings),
    )


def write_validation_report(path: str | Path, report: ValidationReport) -> str:
    """Atomically write a JSON report and return its exact-file SHA256."""

    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    return atomic_write_json(path, report.as_dict())


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_run",
    "write_validation_report",
]
