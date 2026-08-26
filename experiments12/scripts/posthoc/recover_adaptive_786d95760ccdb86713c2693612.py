"""Recover task-989 cell 786d957 after a truncated checkpoint-5 judge.

The default invocation is a provider-free audit.  Paid recovery is deliberately
gated by explicit spend, 640-token-cap, and four-call confirmations.  This
script is locked to the exact 15-event prefix, case JSON, failed job, checkpoint-4
reground, and carried history below.  It never truncates an event log, removes a
reservation, retries a transport call, or reuses a consumed provider request key.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from experiments12.adaptive_deployment12 import (
    ADAPTIVE_DEPLOYMENT_MODE,
    ADAPTIVE_POLICY,
    ADAPTIVE_RESULT_SUBDIR,
    ADAPTIVE_JOB_SUBDIR,
    ADAPTIVE_RUNNER_VERSION,
    ADAPTIVE_SCHEMA_VERSION,
    AdaptiveDeploymentError,
    _accounting,
    _apply_online_action,
    _call_record,
    _decision_record,
    _design,
    _job_state,
    _manifest_mode,
    _method_kind,
    _observe_current_prefix,
    _replay_signal_record,
    _request_key,
    _require_receipt,
    _runtime_config,
    _validate_existing,
    _validate_manifest_matrix,
    validate_adaptive_design,
)
from experiments12.cli12 import REPOSITORY_ROOT, _environment
from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.transport import CompletionResult, Transport
from experiments12.deployment12 import (
    THRESHOLD_LOCK_RECEIPT,
    _validate_evolving_runtime_provenance,
    load_threshold_lock,
)
from experiments12.domains.base import DomainTask
from experiments12.domains.evolving_intent import EvolvingIntentAdapter
from experiments12.harness12 import (
    DEFAULT_REASONING_EFFORT,
    HarnessConfig,
    conservative_input_token_bound,
    grade_final_numeric,
)
from experiments12.manifest12 import RunLayout
from experiments12.monitors.judge import (
    JUDGE_RESPONSE_SCHEMA,
    build_judge_request,
    parse_judge_output,
)
from experiments12.operators12 import CompactionConfig, freeze_initial_instructions
from experiments12.pairing12 import JobCell
from experiments12.passive_spec12 import passive_monitor_spec_from_manifest
from experiments12.runner12 import _stage_ledger, _validate_run_inputs
from experiments12.spec12 import Stage


RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
CELL_ID = "786d95760ccdb86713c26936"
ARTIFACTS_ROOT = REPOSITORY_ROOT / "experiments12" / "data_results" / "runs"
CASE_PATH = (
    REPOSITORY_ROOT
    / "experiments12/data_results/derived/recovery-case-adaptive-786d95760ccdb86713c26936-exact-cp5-v1.json"
)
CASE_SHA256 = "1cb938ef37db8edc3497adf1765924d6e3f67ee9cd30c58d36abc12e6d03932a"
EXPECTED_CODE_TREE_SHA256 = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)
DATASET_PATH = (
    REPOSITORY_ROOT
    / "experiments12/data_results/derived/evolving-deployment-forty-composed-a/"
    "evolving_intent_gsm8k_frozen.json"
)
TASK_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "experiments12/data_results/derived/evolving-deployment-forty-composed-a/tasks-t7.jsonl"
)
BUILD_RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "experiments12/data_results/derived/evolving-deployment-forty-composed-a/build_receipt.json"
)
THRESHOLD_LOCK_PATH = (
    ARTIFACTS_ROOT / RUN_ID / "results/deployment_threshold_lock.json"
)
DATASET_SHA256 = "6bdd6eb969a6c3f93e495e0d21be1055d1423071b2815c938949f44eac4a16ad"
MANIFEST_SHA256 = "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
PAIR_MANIFEST_SHA256 = "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
THRESHOLD_LOCK_SHA256 = "061216da43506e13159eada54226c697cd94d0a72da8203c05605a69e14247d2"
DESIGN_SHA256 = "aa34d4365cf5bd41e9413ceebfd2518e63e8389d280b727060927d130a3238a5"

# Both the canonical JSON sequence and its exact JSONL encoding are locked.
PARTIAL_EVENT_COUNT = 15
PARTIAL_PREFIX_SHA256 = "c6dc693f2aba9632fd44aba23ca0dd71d6334fbf0899286107cb1b578cfc4573"
PARTIAL_FILE_SHA256 = "d9f9b4893158385891e5bc5823217f05eb22285f627702206661ede63a8ed0a2"
FAILED_JOB_SHA256 = "f914a1cf90daae5a3cfd483625162209cabc87d60c8a9f238749cf6b1f353ef5"
FAILED_JOB = {
    "adaptive_runner_version": 1,
    "cell_id": CELL_ID,
    "error_type": "ValueError",
    "state": "failed",
}

STANDARD_JUDGE_5 = _request_key(RUN_ID, CELL_ID, "trace-judge", 5)
STANDARD_JUDGE_6 = _request_key(RUN_ID, CELL_ID, "trace-judge", 6)
FINAL_RECOVERY_JUDGE_5 = (
    f"{RUN_ID}/{CELL_ID}/adaptive-trace-judge-5-recovery-final-cap640-v1"
)
LATER_STANDARD_KEYS = (
    _request_key(RUN_ID, CELL_ID, "task", 6),
    STANDARD_JUDGE_6,
    _request_key(RUN_ID, CELL_ID, "task", 7),
)
MALFORMED_PROVIDER_REQUEST_ID = "req_0ef771202e1f4e149544bc3202554694"
MALFORMED_EVENT_ID = "af34404456b84d27833193218a878154"
MALFORMED_RESERVATION_ID = "3ab51cba528c46e6baa00f36a782defc"
MALFORMED_ATTEMPT_SHA256 = (
    "17d52ef53f50a034a8feb55095a3f6031e336be232d60b59c6e722ac0c32ba34"
)


@dataclass(slots=True)
class FrozenContext:
    layout: RunLayout
    manifest: dict[str, Any]
    cell: JobCell
    task: DomainTask
    threshold: Any
    threshold_lock: Any
    passive_spec: dict[str, Any]
    config: HarnessConfig
    compaction: CompactionConfig
    start: dict[str, Any]


@dataclass(slots=True)
class PrefixState:
    events: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    assistant_task: list[str]
    task_records: list[dict[str, Any]]
    signal_records: list[dict[str, Any]]
    decision_records: list[dict[str, Any]]
    intervention_records: list[dict[str, Any]]
    actions: int
    instructions: Any


def _code_tree_hash() -> str:
    package = REPOSITORY_ROOT / "experiments12"
    records = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".sqlite3"}:
            continue
        if any(part in {"artifacts", "external", "generated"} for part in path.parts):
            continue
        records.append(
            {"path": str(path.relative_to(package)), "sha256": sha256_file(path)}
        )
    return sha256_json(records)


def _validate_case_lock() -> None:
    if sha256_file(CASE_PATH) != CASE_SHA256:
        raise AdaptiveDeploymentError("task-989 recovery case hash changed")
    case = read_json(CASE_PATH)
    malformed = case.get("malformed_attempt")
    expected_suffix = list(LATER_STANDARD_KEYS)
    if (
        case.get("case_version") != 1
        or case.get("run_id") != RUN_ID
        or case.get("cell_id") != CELL_ID
        or case.get("model") != "gpt-5.6-luna"
        or case.get("task_id") != "extracted-gsm8k-test-989::t7"
        or case.get("task_sha256")
        != "aa461e9eaaefb3dbd90b8d6fc21771a2aa604ea5d778e7e408e1ead3aa5c7a8f"
        or case.get("arm") != "trace_judge"
        or case.get("operator") != "public_state_reground"
        or case.get("design_sha256") != DESIGN_SHA256
        or case.get("missing_checkpoint") != 5
        or case.get("actions_at_boundary") != 1
        or case.get("intervention_checkpoint") != 4
        or case.get("intervention_type") != "reground"
        or case.get("intervention_sha256")
        != "d29efeddb1069bf8f6d6d83b04fcdb0432708ee8789a544f8454f572260bf310"
        or case.get("continued_history_sha256")
        != "51ce94326e34b15623a0a33e44d2e70a80464fd2fb200d0e93200d67aaf7b4c2"
        or case.get("partial_event_count") != PARTIAL_EVENT_COUNT
        or case.get("partial_prefix_sha256") != PARTIAL_PREFIX_SHA256
        or case.get("partial_file_sha256") != PARTIAL_FILE_SHA256
        or case.get("failed_job_sha256") != FAILED_JOB_SHA256
        or not isinstance(malformed, Mapping)
        or malformed.get("request_key") != f"{STANDARD_JUDGE_5}/attempt-1"
        or malformed.get("event_id") != MALFORMED_EVENT_ID
        or malformed.get("attempt_sha256") != MALFORMED_ATTEMPT_SHA256
        or malformed.get("reservation_id") != MALFORMED_RESERVATION_ID
        or malformed.get("provider_request_id") != MALFORMED_PROVIDER_REQUEST_ID
        or malformed.get("input_tokens") != 653
        or malformed.get("output_tokens") != 320
        or malformed.get("cached_input_tokens") != 0
        or malformed.get("reasoning_tokens") != 320
        or malformed.get("actual_micro_usd") != 9665
        or case.get("canonical_observer_max_output_tokens") != 320
        or case.get("final_recovery_max_output_tokens") != 640
        or case.get("final_recovery_request_key") != FINAL_RECOVERY_JUDGE_5
        or case.get("later_canonical_request_keys") != expected_suffix
        or case.get("request_input_token_caps") != [2926, 1941, 20000, 20000]
        or case.get("maximum_new_provider_calls") != 4
        or case.get("transport_attempts_per_request") != 1
        or case.get("maximum_additional_reserved_usd") != "0.155649"
        or case.get("frozen_code_tree_sha256") != EXPECTED_CODE_TREE_SHA256
        or case.get("manifest_sha256") != MANIFEST_SHA256
        or case.get("pair_manifest_sha256") != PAIR_MANIFEST_SHA256
        or case.get("threshold_lock_sha256") != THRESHOLD_LOCK_SHA256
    ):
        raise AdaptiveDeploymentError("task-989 recovery case fields changed")


def _load_context() -> FrozenContext:
    if _code_tree_hash() != EXPECTED_CODE_TREE_SHA256:
        raise AdaptiveDeploymentError("frozen source/config tree changed")
    _validate_case_lock()
    layout = RunLayout.for_run(ARTIFACTS_ROOT, RUN_ID)
    if sha256_file(layout.manifest) != MANIFEST_SHA256:
        raise AdaptiveDeploymentError("recovery manifest hash changed")
    if sha256_file(layout.pairs) != PAIR_MANIFEST_SHA256:
        raise AdaptiveDeploymentError("recovery pair-manifest hash changed")
    if sha256_file(THRESHOLD_LOCK_PATH) != THRESHOLD_LOCK_SHA256:
        raise AdaptiveDeploymentError("recovery threshold-lock hash changed")

    adapter = EvolvingIntentAdapter(DATASET_PATH, expected_sha256=DATASET_SHA256)
    tasks = adapter.load_tasks()
    manifest, cells, task_index = _validate_run_inputs(
        layout=layout,
        task_manifest_path=TASK_MANIFEST_PATH,
        tasks=tasks,
    )
    _manifest_mode(manifest)
    _validate_manifest_matrix(manifest=manifest, cells=cells, task_index=task_index)
    _validate_evolving_runtime_provenance(
        manifest=manifest,
        cells=cells,
        task_index=task_index,
        dataset_path=DATASET_PATH,
        build_receipt_path=BUILD_RECEIPT_PATH,
    )
    passive_spec = passive_monitor_spec_from_manifest(manifest)
    threshold_digest = _require_receipt(
        manifest,
        name=THRESHOLD_LOCK_RECEIPT,
        path=THRESHOLD_LOCK_PATH,
    )
    if threshold_digest != THRESHOLD_LOCK_SHA256:
        raise AdaptiveDeploymentError("manifest binds another threshold lock")
    threshold_lock = load_threshold_lock(THRESHOLD_LOCK_PATH)
    validate_adaptive_design(
        cells=cells,
        task_index=task_index,
        threshold_lock=threshold_lock,
    )

    matches = [cell for cell in cells if cell.cell_id == CELL_ID]
    if len(matches) != 1:
        raise AdaptiveDeploymentError("recovery cell is not uniquely declared")
    cell = matches[0]
    if (
        cell.arm != "trace_judge"
        or cell.operator != "public_state_reground"
        or cell.pair_key.model != "gpt-5.6-luna"
        or cell.pair_key.task_id != "extracted-gsm8k-test-989::t7"
        or str(cell.pair_key.task_sha256)
        != "aa461e9eaaefb3dbd90b8d6fc21771a2aa604ea5d778e7e408e1ead3aa5c7a8f"
    ):
        raise AdaptiveDeploymentError("recovery cell treatment/identity changed")
    key = (
        cell.pair_key.domain,
        cell.pair_key.task_id,
        str(cell.pair_key.task_sha256),
    )
    task = task_index[key]
    threshold = threshold_lock.threshold_for(
        cell.pair_key.model, cell.pair_key.domain, cell.arm
    )
    if threshold.threshold != 0.76 or threshold.lock_sha256 != (
        "2ed1ff3d5689941f96e9ce88805460f04d042fd4b96127af6a2afabcc7fcf8e2"
    ):
        raise AdaptiveDeploymentError("recovery threshold record changed")

    config = HarnessConfig()
    compaction = CompactionConfig()
    extra = manifest["extra_config"]
    if extra.get("adaptive_runtime") != _runtime_config(config, compaction):
        raise AdaptiveDeploymentError("recovery runtime differs from the frozen run")
    design = _design(
        run_id=RUN_ID,
        cell=cell,
        task=task,
        threshold=threshold,
        threshold_lock=threshold_lock,
        threshold_lock_sha256=threshold_digest,
        manifest_sha256=sha256_file(layout.manifest),
        pair_manifest_sha256=sha256_file(layout.pairs),
        passive_spec=passive_spec,
        config=config,
        compaction_config=compaction,
    )
    if sha256_json(design) != DESIGN_SHA256:
        raise AdaptiveDeploymentError("recomputed adaptive design changed")
    start = {"event": "start", "design_sha256": DESIGN_SHA256, **design}
    return FrozenContext(
        layout=layout,
        manifest=manifest,
        cell=cell,
        task=task,
        threshold=threshold,
        threshold_lock=threshold_lock,
        passive_spec=dict(passive_spec),
        config=config,
        compaction=compaction,
        start=start,
    )


def _validate_ledger_boundary(context: FrozenContext, events: Sequence[Mapping[str, Any]]) -> None:
    expected_bases: list[tuple[str, str]] = []
    for checkpoint in range(1, 6):
        expected_bases.append((_request_key(RUN_ID, CELL_ID, "task", checkpoint), "adaptive_agent_turn"))
        expected_bases.append(
            (_request_key(RUN_ID, CELL_ID, "trace-judge", checkpoint), "adaptive_trace_judge")
        )
    expected = {f"{base}/attempt-1": purpose for base, purpose in expected_bases}
    uri = f"file:{context.layout.ledger.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM reservations WHERE request_key LIKE ? ORDER BY created_at",
                (f"{RUN_ID}/{CELL_ID}/%",),
            )
        ]
    finally:
        connection.close()
    by_key = {str(row["request_key"]): row for row in rows}
    if set(by_key) != set(expected) or len(by_key) != len(rows):
        raise AdaptiveDeploymentError(
            "cell ledger no longer ends at the exact failed checkpoint-5 call"
        )
    for key, purpose in expected.items():
        row = by_key[key]
        if (
            row["provider"] != "openai"
            or row["purpose"] != purpose
            or row["state"] != "reconciled"
            or row["request_status"] != "succeeded"
            or row["actual_micro_usd"] is None
        ):
            raise AdaptiveDeploymentError("a pre-recovery ledger row is not reconciled")

    malformed = by_key[f"{STANDARD_JUDGE_5}/attempt-1"]
    if (
        malformed["provider_request_id"] != MALFORMED_PROVIDER_REQUEST_ID
        or malformed["reservation_id"] != MALFORMED_RESERVATION_ID
        or malformed["input_tokens"] != 653
        or malformed["output_tokens"] != 320
        or malformed["cached_input_tokens"] != 0
        or malformed["reasoning_tokens"] != 320
        or malformed["provider_total_tokens"] != 973
        or malformed["actual_micro_usd"] != 9665
    ):
        raise AdaptiveDeploymentError("the malformed observer reservation changed")

    call_rows = read_jsonl(context.layout.events / "call_attempts.jsonl")
    call_by_reservation = {
        row.get("reservation_id"): row
        for row in call_rows
        if isinstance(row, Mapping) and isinstance(row.get("reservation_id"), str)
    }
    malformed_call = call_by_reservation.get(malformed["reservation_id"])
    if (
        not isinstance(malformed_call, Mapping)
        or sha256_json(malformed_call) != MALFORMED_ATTEMPT_SHA256
        or malformed_call.get("event_id") != MALFORMED_EVENT_ID
        or malformed_call.get("reservation_id") != MALFORMED_RESERVATION_ID
        or malformed_call.get("status") != "succeeded"
        or malformed_call.get("finish_reason") != "max_output_tokens"
        or malformed_call.get("provider_request_id") != MALFORMED_PROVIDER_REQUEST_ID
        or malformed_call.get("attempt_number") != 1
        or malformed_call.get("usage")
        != {
            "cached_input_tokens": 0,
            "input_tokens": 653,
            "output_tokens": 320,
            "provider_reported_total_tokens": 973,
            "reasoning_tokens": 320,
        }
    ):
        raise AdaptiveDeploymentError("malformed judge call audit receipt changed")
    event_ids = {
        event_id
        for event in events
        for event_id in ((event.get("call") or {}).get("call_event_ids") or ())
    }
    if malformed_call.get("event_id") in event_ids:
        raise AdaptiveDeploymentError("malformed judge call was unexpectedly materialized")
    for event_id in event_ids:
        matches = [row for row in call_rows if row.get("event_id") == event_id]
        if len(matches) != 1 or matches[0].get("status") != "succeeded":
            raise AdaptiveDeploymentError("a materialized prefix call lacks one audit event")


def _validate_and_replay_prefix(context: FrozenContext) -> PrefixState:
    output_path = (
        context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{CELL_ID}.json"
    )
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{CELL_ID}.json"
    event_path = context.layout.events / f"adaptive-{CELL_ID}.jsonl"
    if output_path.exists():
        raise AdaptiveDeploymentError("canonical output already exists")
    if (
        not job_path.is_file()
        or read_json(job_path) != FAILED_JOB
        or sha256_file(job_path) != FAILED_JOB_SHA256
    ):
        raise AdaptiveDeploymentError("failed adaptive job receipt changed")
    if (
        not event_path.is_file()
        or sha256_file(event_path) != PARTIAL_FILE_SHA256
    ):
        raise AdaptiveDeploymentError("partial adaptive event file changed")
    events = read_jsonl(event_path)
    if (
        len(events) != PARTIAL_EVENT_COUNT
        or sha256_json(events) != PARTIAL_PREFIX_SHA256
        or events[0] != context.start
    ):
        raise AdaptiveDeploymentError("partial adaptive event prefix changed")

    first_content = context.task.turns[0].user_message
    instructions = freeze_initial_instructions(
        domain=context.task.domain,
        task_id=context.task.task_id,
        messages=({"role": "user", "content": first_content},),
    )
    messages: list[dict[str, Any]] = []
    assistant_task: list[str] = []
    task_records: list[dict[str, Any]] = []
    signal_records: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    actions = 0
    cursor = 1
    for turn_number, turn in enumerate(context.task.turns[:5], 1):
        record = events[cursor]
        cursor += 1
        user = {"role": "user", "content": turn.user_message}
        messages.append(user)
        assistant = record.get("assistant_message")
        if (
            not isinstance(assistant, Mapping)
            or assistant.get("role") != "assistant"
            or not isinstance(assistant.get("content"), str)
        ):
            raise AdaptiveDeploymentError("partial task assistant record is invalid")
        expected_task = {
            "event": "task_turn",
            "task_turn": turn_number,
            "user_message": user,
            "assistant_message": dict(assistant),
            "request_prefix_sha256": sha256_json(messages),
            "call": record.get("call"),
            "continued_history_sha256": "",
        }
        messages.append(dict(assistant))
        expected_task["continued_history_sha256"] = sha256_json(messages)
        if record != expected_task:
            raise AdaptiveDeploymentError("partial task history does not replay")
        task_records.append(expected_task)
        assistant_task.append(str(assistant["content"]))
        if turn_number == 5:
            continue

        signal = events[cursor]
        cursor += 1
        expected_signal, carried = _replay_signal_record(
            cell=context.cell,
            task=context.task,
            checkpoint=turn_number,
            messages=messages,
            task_records=task_records,
            record=signal,
            passive_spec=context.passive_spec,
        )
        if signal != expected_signal:
            raise AdaptiveDeploymentError("partial observer signal does not replay")
        messages = carried
        signal_records.append(expected_signal)
        decision = events[cursor]
        cursor += 1
        expected_decision = _decision_record(
            signal=expected_signal,
            threshold=context.threshold,
            cap=context.threshold_lock.natural_max_actions_per_task,
            actions_before=actions,
        )
        if decision != expected_decision:
            raise AdaptiveDeploymentError("partial adaptive decision does not replay")
        decision_records.append(expected_decision)
        if expected_decision["action_selected"]:
            continued, intervention = _apply_online_action(
                cell=context.cell,
                task=context.task,
                messages=messages,
                signal=expected_signal,
                decision=expected_decision,
                instructions=instructions,
                compaction_config=context.compaction,
            )
            if cursor >= len(events) or events[cursor] != intervention:
                raise AdaptiveDeploymentError("partial intervention does not replay")
            cursor += 1
            messages = continued
            intervention_records.append(intervention)
            actions += 1

    if (
        cursor != len(events)
        or actions != 1
        or len(intervention_records) != 1
        or intervention_records[0].get("checkpoint") != 4
        or intervention_records[0].get("declared_operator")
        != "public_state_reground"
        or intervention_records[0].get("intervention_type") != "reground"
        or sha256_json(intervention_records[0])
        != "d29efeddb1069bf8f6d6d83b04fcdb0432708ee8789a544f8454f572260bf310"
        or decision_records[-1].get("checkpoint") != 4
        or decision_records[-1].get("action_selected") is not True
        or decision_records[-1].get("actions_after") != 1
    ):
        raise AdaptiveDeploymentError(
            "partial prefix is not the locked checkpoint-4 reground boundary"
        )
    if sha256_json(messages) != (
        "51ce94326e34b15623a0a33e44d2e70a80464fd2fb200d0e93200d67aaf7b4c2"
    ):
        raise AdaptiveDeploymentError("checkpoint-5 target history changed")
    _accounting([*task_records, *signal_records])
    _validate_ledger_boundary(context, events)
    return PrefixState(
        events=list(events),
        messages=messages,
        assistant_task=assistant_task,
        task_records=task_records,
        signal_records=signal_records,
        decision_records=decision_records,
        intervention_records=intervention_records,
        actions=actions,
        instructions=instructions,
    )


def _validate_initial_call_caps(context: FrozenContext, state: PrefixState) -> None:
    judge_request = build_judge_request(
        state.messages,
        5,
        benchmark=context.task.domain,
    )
    judge_bound = conservative_input_token_bound(
        judge_request,
        extra_bytes=len(str(JUDGE_RESPONSE_SCHEMA).encode("utf-8")),
    )
    task6_messages = [
        *state.messages,
        {"role": "user", "content": context.task.turns[5].user_message},
    ]
    task6_bound = conservative_input_token_bound(task6_messages)
    if judge_bound != 2926 or task6_bound != 1941:
        raise AdaptiveDeploymentError("initial recovery input-token bounds changed")


def _archive_failed_state(context: FrozenContext) -> dict[str, Path]:
    """Write-once preservation of artifacts that canonical completion replaces."""

    event_path = context.layout.events / f"adaptive-{CELL_ID}.jsonl"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{CELL_ID}.json"
    archive_root = context.layout.results / "recovery" / CELL_ID
    archived_events = archive_root / "partial-events-15.jsonl"
    archived_job = archive_root / "failed-job.json"
    archive_receipt = archive_root / "archive-receipt.json"
    event_bytes = event_path.read_bytes()
    job_bytes = job_path.read_bytes()
    if sha256_file(event_path) != PARTIAL_FILE_SHA256 or sha256_file(job_path) != FAILED_JOB_SHA256:
        raise AdaptiveDeploymentError("source artifacts changed before recovery archive")

    for destination, payload, digest in (
        (archived_events, event_bytes, PARTIAL_FILE_SHA256),
        (archived_job, job_bytes, FAILED_JOB_SHA256),
    ):
        if destination.exists():
            if sha256_file(destination) != digest or destination.read_bytes() != payload:
                raise AdaptiveDeploymentError("existing recovery archive conflicts with source")
        else:
            written = atomic_write_bytes(destination, payload)
            if written != digest:
                raise AssertionError("exact recovery archive hash drifted")

    receipt = {
        "artifact_type": "experiment12_online_adaptive_pre_recovery_archive",
        "run_id": RUN_ID,
        "cell_id": CELL_ID,
        "case_file": str(CASE_PATH.relative_to(REPOSITORY_ROOT)),
        "case_sha256": CASE_SHA256,
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "frozen_code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
        "archive_created_before_provider_dispatch": True,
        "source_failed_job": f"results/{ADAPTIVE_JOB_SUBDIR}/{CELL_ID}.json",
        "source_failed_job_sha256": FAILED_JOB_SHA256,
        "archived_failed_job": str(archived_job.relative_to(context.layout.root)),
        "archived_failed_job_sha256": sha256_file(archived_job),
        "source_partial_events": f"events/adaptive-{CELL_ID}.jsonl",
        "source_partial_event_count": PARTIAL_EVENT_COUNT,
        "source_partial_prefix_sha256": PARTIAL_PREFIX_SHA256,
        "source_partial_file_sha256": PARTIAL_FILE_SHA256,
        "archived_partial_events": str(archived_events.relative_to(context.layout.root)),
        "archived_partial_events_sha256": sha256_file(archived_events),
        "canonical_event_log_remains_append_only": True,
        "canonical_failed_job_replaced_only_after_complete_output": True,
        "ledger_rows_deleted_or_rewritten": False,
    }
    if archive_receipt.exists():
        if read_json(archive_receipt) != receipt:
            raise AdaptiveDeploymentError("existing recovery archive receipt changed")
    else:
        atomic_write_json(archive_receipt, receipt)
    return {
        "root": archive_root,
        "events": archived_events,
        "job": archived_job,
        "receipt": archive_receipt,
    }


class RecoveryTransport:
    """Permit exactly judge5@640, task6, canonical judge6@320, task7."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.expected = [STANDARD_JUDGE_5, *LATER_STANDARD_KEYS]
        self.seen: list[str] = []
        self.recovery_request_keys: list[str] = []
        self.observer_dispatch_keys: list[str] = []
        self.observer_dispatches: list[dict[str, Any]] = []

    async def complete(
        self,
        model_name: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        request_key = kwargs.get("request_key")
        expected = self.expected[len(self.seen)] if len(self.seen) < len(self.expected) else None
        if request_key != expected:
            raise AdaptiveDeploymentError(
                f"recovery dispatch order changed: expected {expected!r}, got {request_key!r}"
            )
        input_estimate = kwargs.get("input_token_estimate")
        input_cap = (2926, 1941, 20000, 20000)[len(self.seen)]
        if (
            isinstance(input_estimate, bool)
            or not isinstance(input_estimate, int)
            or input_estimate < 1
            or input_estimate > input_cap
        ):
            raise AdaptiveDeploymentError("recovery request exceeds its locked input cap")
        if request_key not in {STANDARD_JUDGE_5, STANDARD_JUDGE_6}:
            if request_key not in LATER_STANDARD_KEYS:
                raise AdaptiveDeploymentError("non-observer recovery key is not canonical")
            if model_name != "gpt-5.6-luna" or kwargs.get("max_output_tokens") != 1800:
                raise AdaptiveDeploymentError("canonical task model/output cap changed")
            result = await self.transport.complete(model_name, messages, **kwargs)
            if len(result.attempts) != 1:
                raise AdaptiveDeploymentError("canonical task suffix attempted more than once")
            self.seen.append(str(request_key))
            return result

        if model_name != "gpt-5.6-sol-judge":
            raise AdaptiveDeploymentError("recovery judge model changed")
        if kwargs.get("max_output_tokens") != 320:
            raise AdaptiveDeploymentError("canonical judge cap changed before dispatch")
        replacement = dict(kwargs)
        if request_key == STANDARD_JUDGE_5:
            dispatch_key = FINAL_RECOVERY_JUDGE_5
            replacement["request_key"] = dispatch_key
            replacement["max_output_tokens"] = 640
            self.recovery_request_keys.append(dispatch_key)
        else:
            dispatch_key = STANDARD_JUDGE_6
        result = await self.transport.complete(model_name, messages, **replacement)
        if len(result.attempts) != 1:
            raise AdaptiveDeploymentError("recovery observer attempted more than once")
        call = _call_record(result)
        dispatch = {
            "logical_request_key": str(request_key),
            "dispatch_request_key": dispatch_key,
            "physical_request_key": f"{dispatch_key}/attempt-1",
            "max_output_tokens": replacement["max_output_tokens"],
            "call_event_ids": list(call["call_event_ids"]),
            "reservation_ids": [attempt.reservation_id for attempt in result.attempts],
            "provider_request_id": result.request_id,
            "finish_reason": result.finish_reason,
            "raw_output_sha256": sha256_json(result.text),
            "raw_output_characters": len(result.text),
        }
        try:
            parse_judge_output(result.text)
        except ValueError as exc:
            dispatch["semantic_parse"] = "failed"
            dispatch["semantic_error_type"] = type(exc).__name__
            self.observer_dispatches.append(dispatch)
            raise AdaptiveDeploymentError(
                f"the single permitted checkpoint-{5 if request_key == STANDARD_JUDGE_5 else 6} judge response was malformed"
            ) from exc
        dispatch["semantic_parse"] = "succeeded"
        self.observer_dispatches.append(dispatch)
        self.observer_dispatch_keys.append(dispatch_key)
        self.seen.append(str(request_key))
        return result


def _append_action_if_selected(
    context: FrozenContext,
    state: PrefixState,
    signal: dict[str, Any],
    decision: dict[str, Any],
    event_path: Path,
) -> None:
    if not decision["action_selected"]:
        return
    state.messages, intervention = _apply_online_action(
        cell=context.cell,
        task=context.task,
        messages=state.messages,
        signal=signal,
        decision=decision,
        instructions=state.instructions,
        compaction_config=context.compaction,
    )
    state.actions += 1
    if state.actions != decision["actions_after"]:
        raise AssertionError("recovery action count drifted")
    append_jsonl(event_path, intervention)
    state.events.append(intervention)
    state.intervention_records.append(intervention)


async def _execute_recovery(
    context: FrozenContext,
    state: PrefixState,
    *,
    env_file: Path,
) -> dict[str, Any]:
    event_path = context.layout.events / f"adaptive-{CELL_ID}.jsonl"
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{CELL_ID}.json"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{CELL_ID}.json"
    archive = _archive_failed_state(context)
    transport = Transport(
        _stage_ledger(context.layout, RUN_ID, Stage.CONFIRMATORY),
        context.layout.events / "call_attempts.jsonl",
        environ=_environment(env_file),
        max_attempts=1,
    )
    recovery = RecoveryTransport(transport)

    # The failed run stopped after task turn 5 and before its signal event.
    signal = await _observe_current_prefix(
        run_id=RUN_ID,
        cell=context.cell,
        task=context.task,
        checkpoint=5,
        checkpoint_index=5,
        messages=state.messages,
        task_records=state.task_records,
        transport=recovery,  # type: ignore[arg-type]
        config=context.config,
        passive_spec=context.passive_spec,
    )
    if signal["source_prefix_before_observation_sha256"] != state.task_records[-1][
        "continued_history_sha256"
    ]:
        raise AssertionError("recovered signal did not observe task turn 5")
    append_jsonl(event_path, signal)
    state.events.append(signal)
    state.signal_records.append(signal)
    decision = _decision_record(
        signal=signal,
        threshold=context.threshold,
        cap=context.threshold_lock.natural_max_actions_per_task,
        actions_before=state.actions,
    )
    append_jsonl(event_path, decision)
    state.events.append(decision)
    state.decision_records.append(decision)
    if decision["action_selected"] or decision["actions_after"] != 1:
        raise AssertionError("checkpoint-5 recovery violated the spent action cap")
    _append_action_if_selected(context, state, signal, decision, event_path)

    for turn_number in (6, 7):
        turn = context.task.turns[turn_number - 1]
        user = {"role": "user", "content": turn.user_message}
        state.messages.append(user)
        request_prefix_sha256 = sha256_json(state.messages)
        result = await recovery.complete(
            context.cell.pair_key.model,
            state.messages,
            purpose="adaptive_agent_turn",
            request_key=_request_key(RUN_ID, CELL_ID, "task", turn_number),
            input_token_estimate=conservative_input_token_bound(state.messages),
            max_output_tokens=context.config.task_max_output_tokens,
            temperature=context.config.temperature,
            reasoning_effort=DEFAULT_REASONING_EFFORT[context.cell.pair_key.model],
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("recovered task turn returned tool calls")
        assistant = {"role": "assistant", "content": result.text}
        state.messages.append(assistant)
        state.assistant_task.append(result.text)
        task_event = {
            "event": "task_turn",
            "task_turn": turn_number,
            "user_message": user,
            "assistant_message": assistant,
            "request_prefix_sha256": request_prefix_sha256,
            "call": _call_record(result),
            "continued_history_sha256": sha256_json(state.messages),
        }
        append_jsonl(event_path, task_event)
        state.events.append(task_event)
        state.task_records.append(task_event)
        if turn_number == 7:
            continue

        signal = await _observe_current_prefix(
            run_id=RUN_ID,
            cell=context.cell,
            task=context.task,
            checkpoint=turn_number,
            checkpoint_index=turn_number,
            messages=state.messages,
            task_records=state.task_records,
            transport=recovery,  # type: ignore[arg-type]
            config=context.config,
            passive_spec=context.passive_spec,
        )
        if signal["source_prefix_before_observation_sha256"] != task_event[
            "continued_history_sha256"
        ]:
            raise AssertionError("recovered signal did not observe task turn 6")
        append_jsonl(event_path, signal)
        state.events.append(signal)
        state.signal_records.append(signal)
        decision = _decision_record(
            signal=signal,
            threshold=context.threshold,
            cap=context.threshold_lock.natural_max_actions_per_task,
            actions_before=state.actions,
        )
        append_jsonl(event_path, decision)
        state.events.append(decision)
        state.decision_records.append(decision)
        if decision["action_selected"] or decision["actions_after"] != 1:
            raise AssertionError("checkpoint-6 recovery violated the spent action cap")
        _append_action_if_selected(context, state, signal, decision, event_path)

    if recovery.seen != recovery.expected:
        raise AssertionError("recovery did not use its exact four-call schedule")
    checkpoints = tuple(range(1, len(context.task.turns)))
    if (
        len(state.task_records) != len(context.task.turns)
        or len(state.signal_records) != len(checkpoints)
        or len(state.decision_records) != len(checkpoints)
    ):
        raise AssertionError("recovery did not materialize complete checkpoint coverage")

    prediction, success = grade_final_numeric(
        state.assistant_task[-1], context.task.evaluation_label
    )
    transcript_sha256 = sha256_json(state.messages)
    accounting = _accounting([*state.task_records, *state.signal_records])
    kind, active_variant = _method_kind(context.cell.arm)
    materialized = {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "adaptive_runner_version": ADAPTIVE_RUNNER_VERSION,
        "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
        "deployment_policy": ADAPTIVE_POLICY,
        "run_id": RUN_ID,
        "cell_id": CELL_ID,
        "design_sha256": DESIGN_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
        "threshold_lock_sha256": THRESHOLD_LOCK_SHA256,
        "threshold_record_sha256": context.threshold.lock_sha256,
        "calibration_run_id": context.threshold_lock.calibration_run_id,
        "calibration_manifest_sha256": context.threshold_lock.calibration_manifest_sha256,
        "model": context.cell.pair_key.model,
        "domain": context.task.domain,
        "task_id": context.task.task_id,
        "condition": context.task.condition,
        "task_sha256": context.task.task_sha256,
        "observation_method": context.cell.arm,
        "observation_kind": kind,
        "active_probe_variant": active_variant,
        "operator": context.cell.operator,
        "per_task_action_cap": context.threshold_lock.natural_max_actions_per_task,
        "checkpoint_turns": list(checkpoints),
        "messages": state.messages,
        "task_assistant_messages": state.assistant_task,
        "task_records": state.task_records,
        "signal_records": state.signal_records,
        "decision_records": state.decision_records,
        "intervention_records": state.intervention_records,
        "probe_records": [
            row
            for row in state.signal_records
            if row["observation_kind"] == "active_carry"
        ],
        "evaluation": {
            "prediction": prediction,
            "evaluation_label_sha256": (
                None
                if context.task.evaluation_label is None
                else sha256_json(context.task.evaluation_label)
            ),
            "success": success,
        },
        "transcript_sha256": transcript_sha256,
        "accounting": accounting,
        "observation_burden": {
            "checkpoints": len(state.signal_records),
            "paid_observer_calls": accounting["by_category"]["observer"]["calls"],
            "carried_probe_calls": sum(
                row["observation_kind"] == "active_carry"
                for row in state.signal_records
            ),
        },
        "event_log_prefix_sha256": sha256_json(state.events),
        "complete": True,
    }
    atomic_write_json(output_path, materialized)
    append_jsonl(
        event_path,
        {
            "event": "complete",
            "design_sha256": DESIGN_SHA256,
            "task_turns": len(state.task_records),
            "signals": len(state.signal_records),
            "selected_actions": state.actions,
            "transcript_sha256": transcript_sha256,
            "output_sha256": sha256_file(output_path),
            "prediction": prediction,
            "success": success,
        },
    )
    _job_state(
        job_path,
        cell=context.cell,
        state="complete",
        detail={
            "output_sha256": sha256_file(output_path),
            "success": success,
            "accounting_sha256": sha256_json(accounting),
        },
    )
    validated = _validate_existing(
        output_file=output_path,
        event_file=event_path,
        start=context.start,
        task=context.task,
        compaction_config=context.compaction,
    )
    receipt_path = (
        REPOSITORY_ROOT
        / "experiments12/data_results/derived/recovery-adaptive-786d95760ccdb86713c2693612.json"
    )
    atomic_write_json(
        receipt_path,
        {
            "artifact_type": "experiment12_online_adaptive_single_cell_recovery",
            "run_id": RUN_ID,
            "cell_id": CELL_ID,
            "case_file": str(CASE_PATH.relative_to(REPOSITORY_ROOT)),
            "case_sha256": CASE_SHA256,
            "recovery_script_sha256": sha256_file(Path(__file__)),
            "frozen_code_tree_sha256": EXPECTED_CODE_TREE_SHA256,
            "original_partial_event_count": PARTIAL_EVENT_COUNT,
            "original_partial_prefix_sha256": PARTIAL_PREFIX_SHA256,
            "original_partial_file_sha256": PARTIAL_FILE_SHA256,
            "original_failed_job_sha256": FAILED_JOB_SHA256,
            "pre_recovery_archive": str(archive["root"].relative_to(context.layout.root)),
            "pre_recovery_archive_receipt_sha256": sha256_file(archive["receipt"]),
            "archived_failed_job_sha256": sha256_file(archive["job"]),
            "archived_partial_events_sha256": sha256_file(archive["events"]),
            "original_malformed_request_key": f"{STANDARD_JUDGE_5}/attempt-1",
            "original_malformed_event_id": MALFORMED_EVENT_ID,
            "original_malformed_attempt_sha256": MALFORMED_ATTEMPT_SHA256,
            "original_malformed_reservation_id": MALFORMED_RESERVATION_ID,
            "original_malformed_provider_request_id": MALFORMED_PROVIDER_REQUEST_ID,
            "canonical_observer_max_output_tokens": 320,
            "final_recovery_max_output_tokens": 640,
            "maximum_new_provider_calls": 4,
            "transport_attempts_per_request": 1,
            "recovery_request_keys": recovery.recovery_request_keys,
            "observer_dispatch_keys": recovery.observer_dispatch_keys,
            "observer_dispatches": recovery.observer_dispatches,
            "later_request_keys_are_canonical": list(LATER_STANDARD_KEYS),
            "ledger_rows_deleted_or_rewritten": False,
            "final_event_log_sha256": sha256_file(event_path),
            "final_output_sha256": sha256_file(output_path),
            "final_job_sha256": sha256_file(job_path),
            "accounting_sha256": sha256_json(validated["accounting"]),
        },
    )
    return validated


def _validate_completed(context: FrozenContext) -> dict[str, Any] | None:
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{CELL_ID}.json"
    if not output_path.exists():
        return None
    event_path = context.layout.events / f"adaptive-{CELL_ID}.jsonl"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{CELL_ID}.json"
    output = _validate_existing(
        output_file=output_path,
        event_file=event_path,
        start=context.start,
        task=context.task,
        compaction_config=context.compaction,
    )
    job = read_json(job_path)
    if (
        job.get("state") != "complete"
        or job.get("cell_id") != CELL_ID
        or job.get("output_sha256") != sha256_file(output_path)
    ):
        raise AdaptiveDeploymentError("completed recovery conflicts with its job receipt")
    return output


async def _main(args: argparse.Namespace) -> int:
    context = _load_context()
    completed = _validate_completed(context)
    if completed is not None:
        print(
            f"already complete: cell={CELL_ID} output={completed['evaluation']['success']} "
            f"output_sha256={sha256_file(context.layout.results / ADAPTIVE_RESULT_SUBDIR / f'{CELL_ID}.json')}"
        )
        return 0
    state = _validate_and_replay_prefix(context)
    _validate_initial_call_caps(context, state)
    if not args.execute:
        archive_note = ""
        if args.archive:
            archive = _archive_failed_state(context)
            archive_note = (
                " archive="
                + str(archive["root"].relative_to(context.layout.root))
            )
        print(
            f"audit passed: cell={CELL_ID} events={len(state.events)} "
            "reground=checkpoint-4 missing=trace_judge@5 "
            "schedule=judge-5-cap640,task-6,judge-6-cap320,task-7 "
            "maximum_new_calls=4 transport_attempts_per_request=1 "
            "maximum_additional_reserved_usd=0.155649"
            f"{archive_note}"
        )
        return 0
    if not args.yes_spend:
        raise AdaptiveDeploymentError("paid recovery requires --execute --yes-spend")
    if args.confirm_final_observer_max_output_tokens != 640:
        raise AdaptiveDeploymentError("explicit confirmation of the 640-token cap is required")
    if args.confirm_maximum_new_provider_calls != 4:
        raise AdaptiveDeploymentError("explicit confirmation of four maximum new calls is required")
    result = await _execute_recovery(
        context,
        state,
        env_file=Path(args.env_file),
    )
    print(
        f"recovery complete: cell={CELL_ID} success={result['evaluation']['success']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="store_true",
        help="provider-free write-once archive of the exact failed boundary",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes-spend", action="store_true")
    parser.add_argument("--confirm-final-observer-max-output-tokens", type=int)
    parser.add_argument("--confirm-maximum-new-provider-calls", type=int)
    parser.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    args = parser.parse_args()
    if args.yes_spend and not args.execute:
        raise AdaptiveDeploymentError("--yes-spend is invalid without --execute")
    if not args.execute and (
        args.confirm_final_observer_max_output_tokens is not None
        or args.confirm_maximum_new_provider_calls is not None
    ):
        raise AdaptiveDeploymentError("spend confirmations are invalid without --execute")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
