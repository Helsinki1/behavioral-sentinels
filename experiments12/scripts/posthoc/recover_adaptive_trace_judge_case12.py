"""Resume one hash-locked online cell after capped trace-judge truncations.

The default command is provider-free.  ``--execute --yes-spend`` is required
for dispatch.  A case JSON locks the exact failed boundary, treatment,
all consumed malformed reservations, and carried-history hash.  The script
replays every existing task/signal/decision/intervention record, permits one
final missing-judge call under a distinct key and explicit larger output cap,
and then uses untouched canonical keys for the remaining suffix.  Production
call logs and ledger rows are append-only; the original failed boundary is
archived before dispatch.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from experiments12.adaptive_deployment12 import (
    ADAPTIVE_DEPLOYMENT_MODE,
    ADAPTIVE_JOB_SUBDIR,
    ADAPTIVE_POLICY,
    ADAPTIVE_RESULT_SUBDIR,
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
from experiments12.models12 import estimate_call_upper_bound_usd
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
ARTIFACTS_ROOT = REPOSITORY_ROOT / "experiments12" / "data_results" / "runs"
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
THRESHOLD_LOCK_PATH = ARTIFACTS_ROOT / RUN_ID / "results/deployment_threshold_lock.json"
DATASET_SHA256 = "6bdd6eb969a6c3f93e495e0d21be1055d1423071b2815c938949f44eac4a16ad"
MANIFEST_SHA256 = "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
PAIR_MANIFEST_SHA256 = "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
THRESHOLD_LOCK_SHA256 = "061216da43506e13159eada54226c697cd94d0a72da8203c05605a69e14247d2"
THRESHOLD_RECORD_SHA256 = "2ed1ff3d5689941f96e9ce88805460f04d042fd4b96127af6a2afabcc7fcf8e2"


@dataclass(frozen=True, slots=True)
class MalformedAttempt:
    semantic_attempt: int
    request_key: str
    event_id: str
    attempt_sha256: str
    reservation_id: str
    provider_request_id: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    actual_micro_usd: int


@dataclass(frozen=True, slots=True)
class RecoveryCase:
    cell_id: str
    model: str
    task_id: str
    task_sha256: str
    arm: str
    operator: str
    design_sha256: str
    missing_checkpoint: int
    actions_at_boundary: int
    continued_history_sha256: str
    partial_event_count: int
    partial_prefix_sha256: str
    partial_file_sha256: str
    failed_job_sha256: str
    pre_recovery_archive_case_sha256: str | None
    malformed_attempts: tuple[MalformedAttempt, ...]
    canonical_observer_max_output_tokens: int
    final_recovery_max_output_tokens: int
    final_recovery_request_key: str
    later_canonical_request_keys: tuple[str, ...]
    request_input_token_caps: tuple[int, ...]

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> "RecoveryCase":
        if sha256_file(path) != expected_sha256:
            raise AdaptiveDeploymentError("recovery case hash changed")
        value = read_json(path)
        case_version = value.get("case_version")
        if case_version not in {2, 3} or value.get("run_id") != RUN_ID:
            raise AdaptiveDeploymentError("recovery case identity changed")
        malformed_attempts = tuple(
            MalformedAttempt(
                semantic_attempt=int(item["semantic_attempt"]),
                request_key=str(item["request_key"]),
                event_id=str(item["event_id"]),
                attempt_sha256=str(item["attempt_sha256"]),
                reservation_id=str(item["reservation_id"]),
                provider_request_id=str(item["provider_request_id"]),
                input_tokens=int(item["input_tokens"]),
                output_tokens=int(item["output_tokens"]),
                cached_input_tokens=int(item["cached_input_tokens"]),
                reasoning_tokens=int(item["reasoning_tokens"]),
                actual_micro_usd=int(item["actual_micro_usd"]),
            )
            for item in value["malformed_attempts"]
        )
        case = cls(
            cell_id=str(value["cell_id"]),
            model=str(value["model"]),
            task_id=str(value["task_id"]),
            task_sha256=str(value["task_sha256"]),
            arm=str(value["arm"]),
            operator=str(value["operator"]),
            design_sha256=str(value["design_sha256"]),
            missing_checkpoint=int(value["missing_checkpoint"]),
            actions_at_boundary=int(value["actions_at_boundary"]),
            continued_history_sha256=str(value["continued_history_sha256"]),
            partial_event_count=int(value["partial_event_count"]),
            partial_prefix_sha256=str(value["partial_prefix_sha256"]),
            partial_file_sha256=str(value["partial_file_sha256"]),
            failed_job_sha256=str(value["failed_job_sha256"]),
            pre_recovery_archive_case_sha256=(
                None
                if value.get("pre_recovery_archive_case_sha256") is None
                else str(value["pre_recovery_archive_case_sha256"])
            ),
            malformed_attempts=malformed_attempts,
            canonical_observer_max_output_tokens=int(
                value["canonical_observer_max_output_tokens"]
            ),
            final_recovery_max_output_tokens=int(
                value["final_recovery_max_output_tokens"]
            ),
            final_recovery_request_key=str(value["final_recovery_request_key"]),
            later_canonical_request_keys=tuple(
                str(item) for item in value["later_canonical_request_keys"]
            ),
            request_input_token_caps=tuple(
                int(item)
                for item in value.get(
                    "request_input_token_caps",
                    (3_375, 2_485) if case_version == 2 else (),
                )
            ),
        )
        canonical_judge = (
            f"{RUN_ID}/{case.cell_id}/adaptive-trace-judge-"
            f"{case.missing_checkpoint}/attempt-1"
        )
        malformed_keys = tuple(item.request_key for item in malformed_attempts)
        malformed_ids = tuple(item.event_id for item in malformed_attempts)
        malformed_reservations = tuple(
            item.reservation_id for item in malformed_attempts
        )
        expected_retry_keys = tuple(
            f"{RUN_ID}/{case.cell_id}/adaptive-trace-judge-"
            f"{case.missing_checkpoint}-recovery-semantic-{attempt}/attempt-1"
            for attempt in range(1, len(malformed_attempts))
        )
        expected_suffix: list[str] = []
        for turn in range(case.missing_checkpoint + 1, 8):
            expected_suffix.append(_request_key(RUN_ID, case.cell_id, "task", turn))
            if turn < 7:
                expected_suffix.append(
                    _request_key(RUN_ID, case.cell_id, "trace-judge", turn)
                )
        if (
            case.arm != "trace_judge"
            or not 1 <= case.missing_checkpoint <= 6
            or case.actions_at_boundary not in {0, 1}
            or len(malformed_attempts) < 1
            or tuple(item.semantic_attempt for item in malformed_attempts)
            != tuple(range(len(malformed_attempts)))
            or malformed_keys != (canonical_judge, *expected_retry_keys)
            or len(set(malformed_keys)) != len(malformed_attempts)
            or len(set(malformed_ids)) != len(malformed_attempts)
            or len(set(malformed_reservations)) != len(malformed_attempts)
            or any(item.output_tokens != 320 for item in malformed_attempts)
            or any(item.cached_input_tokens != 0 for item in malformed_attempts)
            or case.canonical_observer_max_output_tokens != 320
            or case.final_recovery_max_output_tokens != 640
            or case.final_recovery_request_key
            != (
                f"{RUN_ID}/{case.cell_id}/adaptive-trace-judge-"
                f"{case.missing_checkpoint}-recovery-final-cap640-v1"
            )
            or f"{case.final_recovery_request_key}/attempt-1" in malformed_keys
            or case.later_canonical_request_keys != tuple(expected_suffix)
            or len(case.request_input_token_caps)
            != 1 + len(case.later_canonical_request_keys)
            or any(value < 1 for value in case.request_input_token_caps)
            or set(case.later_canonical_request_keys) & set(malformed_keys)
            or any(
                len(digest) != 64
                for digest in (
                    case.task_sha256,
                    case.design_sha256,
                    case.continued_history_sha256,
                    case.partial_prefix_sha256,
                    case.partial_file_sha256,
                    case.failed_job_sha256,
                    *(item.attempt_sha256 for item in malformed_attempts),
                    *(
                        ()
                        if case.pre_recovery_archive_case_sha256 is None
                        else (case.pre_recovery_archive_case_sha256,)
                    ),
                )
            )
        ):
            raise AdaptiveDeploymentError("recovery case fields are invalid")
        return case


@dataclass(slots=True)
class FrozenContext:
    case: RecoveryCase
    case_path: Path
    case_sha256: str
    layout: RunLayout
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


def _load_context(case_path: Path, case_sha256: str) -> FrozenContext:
    case = RecoveryCase.load(case_path, case_sha256)
    layout = RunLayout.for_run(ARTIFACTS_ROOT, RUN_ID)
    for path, digest, label in (
        (layout.manifest, MANIFEST_SHA256, "manifest"),
        (layout.pairs, PAIR_MANIFEST_SHA256, "pair manifest"),
        (THRESHOLD_LOCK_PATH, THRESHOLD_LOCK_SHA256, "threshold lock"),
    ):
        if sha256_file(path) != digest:
            raise AdaptiveDeploymentError(f"recovery {label} hash changed")
    adapter = EvolvingIntentAdapter(DATASET_PATH, expected_sha256=DATASET_SHA256)
    manifest, cells, task_index = _validate_run_inputs(
        layout=layout,
        task_manifest_path=TASK_MANIFEST_PATH,
        tasks=adapter.load_tasks(),
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
        manifest, name=THRESHOLD_LOCK_RECEIPT, path=THRESHOLD_LOCK_PATH
    )
    threshold_lock = load_threshold_lock(THRESHOLD_LOCK_PATH)
    validate_adaptive_design(
        cells=cells, task_index=task_index, threshold_lock=threshold_lock
    )
    matches = [cell for cell in cells if cell.cell_id == case.cell_id]
    if len(matches) != 1:
        raise AdaptiveDeploymentError("recovery cell is not uniquely declared")
    cell = matches[0]
    if (
        cell.arm != case.arm
        or cell.operator != case.operator
        or cell.pair_key.model != case.model
        or cell.pair_key.task_id != case.task_id
        or str(cell.pair_key.task_sha256) != case.task_sha256
    ):
        raise AdaptiveDeploymentError("recovery cell differs from its case lock")
    key = (
        cell.pair_key.domain,
        cell.pair_key.task_id,
        str(cell.pair_key.task_sha256),
    )
    task = task_index[key]
    threshold = threshold_lock.threshold_for(
        cell.pair_key.model, cell.pair_key.domain, cell.arm
    )
    if threshold.threshold != 0.76 or threshold.lock_sha256 != THRESHOLD_RECORD_SHA256:
        raise AdaptiveDeploymentError("recovery threshold changed")
    config = HarnessConfig()
    compaction = CompactionConfig()
    if manifest["extra_config"].get("adaptive_runtime") != _runtime_config(
        config, compaction
    ):
        raise AdaptiveDeploymentError("recovery runtime changed")
    design = _design(
        run_id=RUN_ID,
        cell=cell,
        task=task,
        threshold=threshold,
        threshold_lock=threshold_lock,
        threshold_lock_sha256=threshold_digest,
        manifest_sha256=MANIFEST_SHA256,
        pair_manifest_sha256=PAIR_MANIFEST_SHA256,
        passive_spec=passive_spec,
        config=config,
        compaction_config=compaction,
    )
    if sha256_json(design) != case.design_sha256:
        raise AdaptiveDeploymentError("recomputed recovery design changed")
    return FrozenContext(
        case=case,
        case_path=case_path,
        case_sha256=case_sha256,
        layout=layout,
        cell=cell,
        task=task,
        threshold=threshold,
        threshold_lock=threshold_lock,
        passive_spec=dict(passive_spec),
        config=config,
        compaction=compaction,
        start={"event": "start", "design_sha256": case.design_sha256, **design},
    )


def _validate_ledger_boundary(
    context: FrozenContext, events: Sequence[Mapping[str, Any]]
) -> None:
    case = context.case
    call_rows = read_jsonl(context.layout.events / "call_attempts.jsonl")
    by_event = {
        str(row["event_id"]): row
        for row in call_rows
        if isinstance(row, Mapping) and isinstance(row.get("event_id"), str)
    }
    referenced = {
        event_id
        for event in events
        for event_id in ((event.get("call") or {}).get("call_event_ids") or ())
    }
    malformed_event_ids = {item.event_id for item in case.malformed_attempts}
    if malformed_event_ids & referenced:
        raise AdaptiveDeploymentError("a malformed call is already materialized")
    for event_id in referenced:
        if event_id not in by_event or by_event[event_id].get("status") != "succeeded":
            raise AdaptiveDeploymentError("prefix call attempt is absent or invalid")

    for lock in case.malformed_attempts:
        call = by_event.get(lock.event_id)
        usage = call.get("usage") if isinstance(call, Mapping) else None
        if (
            not isinstance(call, Mapping)
            or not isinstance(usage, Mapping)
            or sha256_json(call) != lock.attempt_sha256
            or call.get("reservation_id") != lock.reservation_id
            or call.get("provider_request_id") != lock.provider_request_id
            or call.get("provider") != "openai"
            or call.get("model") != "gpt-5.6-sol"
            or call.get("purpose") != "adaptive_trace_judge"
            or call.get("status") != "succeeded"
            or call.get("finish_reason") != "max_output_tokens"
            or call.get("attempt_number") != 1
            or usage.get("input_tokens") != lock.input_tokens
            or usage.get("output_tokens") != lock.output_tokens
            or usage.get("cached_input_tokens") != lock.cached_input_tokens
            or usage.get("reasoning_tokens") != lock.reasoning_tokens
            or usage.get("provider_reported_total_tokens")
            != lock.input_tokens + lock.output_tokens
        ):
            raise AdaptiveDeploymentError(
                f"malformed call-attempt receipt {lock.semantic_attempt} changed"
            )
    uri = context.layout.ledger.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        ledger = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM reservations WHERE request_key LIKE ?",
                (f"{RUN_ID}/{context.case.cell_id}/%",),
            )
        ]
    finally:
        connection.close()
    by_reservation = {str(row["reservation_id"]): row for row in ledger}
    referenced_reservations = {
        str(by_event[event_id]["reservation_id"]) for event_id in referenced
    }
    malformed_reservations = {
        item.reservation_id for item in case.malformed_attempts
    }
    expected_reservations = referenced_reservations | malformed_reservations
    if len(by_reservation) != len(ledger) or set(by_reservation) != expected_reservations:
        raise AdaptiveDeploymentError("cell ledger contains calls beyond the locked boundary")
    for lock in case.malformed_attempts:
        malformed = by_reservation[lock.reservation_id]
        if (
            malformed.get("request_key") != lock.request_key
            or malformed.get("provider") != "openai"
            or malformed.get("purpose") != "adaptive_trace_judge"
            or malformed.get("state") != "reconciled"
            or malformed.get("request_status") != "succeeded"
            or malformed.get("provider_request_id") != lock.provider_request_id
            or malformed.get("input_tokens") != lock.input_tokens
            or malformed.get("output_tokens") != lock.output_tokens
            or malformed.get("cached_input_tokens") != lock.cached_input_tokens
            or malformed.get("reasoning_tokens") != lock.reasoning_tokens
            or malformed.get("provider_total_tokens")
            != lock.input_tokens + lock.output_tokens
            or malformed.get("actual_micro_usd") != lock.actual_micro_usd
        ):
            raise AdaptiveDeploymentError(
                f"malformed ledger reservation {lock.semantic_attempt} changed"
            )


def _validate_and_replay_prefix(context: FrozenContext) -> PrefixState:
    case = context.case
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{case.cell_id}.json"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{case.cell_id}.json"
    event_path = context.layout.events / f"adaptive-{case.cell_id}.jsonl"
    if output_path.exists():
        raise AdaptiveDeploymentError("canonical recovery output already exists")
    expected_job = {
        "adaptive_runner_version": 1,
        "cell_id": case.cell_id,
        "error_type": "ValueError",
        "state": "failed",
    }
    if read_json(job_path) != expected_job or sha256_file(job_path) != case.failed_job_sha256:
        raise AdaptiveDeploymentError("failed recovery job changed")
    events = read_jsonl(event_path)
    if (
        sha256_file(event_path) != case.partial_file_sha256
        or len(events) != case.partial_event_count
        or sha256_json(events) != case.partial_prefix_sha256
        or events[0] != context.start
    ):
        raise AdaptiveDeploymentError("partial recovery event prefix changed")

    first_content = context.task.turns[0].user_message
    instructions = freeze_initial_instructions(
        domain=context.task.domain,
        task_id=context.task.task_id,
        messages=({"role": "user", "content": first_content},),
    )
    state = PrefixState(
        events=list(events),
        messages=[],
        assistant_task=[],
        task_records=[],
        signal_records=[],
        decision_records=[],
        intervention_records=[],
        actions=0,
        instructions=instructions,
    )
    cursor = 1
    missing_checkpoint: int | None = None
    while cursor < len(events):
        turn_number = len(state.task_records) + 1
        if turn_number > len(context.task.turns):
            raise AdaptiveDeploymentError("partial prefix has extra task turns")
        raw_task = events[cursor]
        cursor += 1
        turn = context.task.turns[turn_number - 1]
        user = {"role": "user", "content": turn.user_message}
        state.messages.append(user)
        assistant = raw_task.get("assistant_message")
        if (
            not isinstance(assistant, Mapping)
            or assistant.get("role") != "assistant"
            or not isinstance(assistant.get("content"), str)
        ):
            raise AdaptiveDeploymentError("partial task assistant is invalid")
        expected_task = {
            "event": "task_turn",
            "task_turn": turn_number,
            "user_message": user,
            "assistant_message": dict(assistant),
            "request_prefix_sha256": sha256_json(state.messages),
            "call": raw_task.get("call"),
            "continued_history_sha256": "",
        }
        state.messages.append(dict(assistant))
        expected_task["continued_history_sha256"] = sha256_json(state.messages)
        if raw_task != expected_task:
            raise AdaptiveDeploymentError("partial task history does not replay")
        state.task_records.append(expected_task)
        state.assistant_task.append(str(assistant["content"]))
        if cursor == len(events):
            missing_checkpoint = turn_number
            break
        if turn_number == len(context.task.turns):
            raise AdaptiveDeploymentError("complete task has unexplained partial suffix")
        raw_signal = events[cursor]
        cursor += 1
        expected_signal, carried = _replay_signal_record(
            cell=context.cell,
            task=context.task,
            checkpoint=turn_number,
            messages=state.messages,
            task_records=state.task_records,
            record=raw_signal,
            passive_spec=context.passive_spec,
        )
        if raw_signal != expected_signal:
            raise AdaptiveDeploymentError("partial signal does not replay")
        state.messages = carried
        state.signal_records.append(expected_signal)
        if cursor >= len(events):
            raise AdaptiveDeploymentError("partial prefix ends before its decision")
        raw_decision = events[cursor]
        cursor += 1
        expected_decision = _decision_record(
            signal=expected_signal,
            threshold=context.threshold,
            cap=context.threshold_lock.natural_max_actions_per_task,
            actions_before=state.actions,
        )
        if raw_decision != expected_decision:
            raise AdaptiveDeploymentError("partial decision does not replay")
        state.decision_records.append(expected_decision)
        if expected_decision["action_selected"]:
            continued, intervention = _apply_online_action(
                cell=context.cell,
                task=context.task,
                messages=state.messages,
                signal=expected_signal,
                decision=expected_decision,
                instructions=state.instructions,
                compaction_config=context.compaction,
            )
            if cursor >= len(events) or events[cursor] != intervention:
                raise AdaptiveDeploymentError("partial intervention does not replay")
            cursor += 1
            state.messages = continued
            state.intervention_records.append(intervention)
            state.actions += 1
    if (
        cursor != len(events)
        or missing_checkpoint != case.missing_checkpoint
        or state.actions != case.actions_at_boundary
        or sha256_json(state.messages) != case.continued_history_sha256
    ):
        raise AdaptiveDeploymentError("replayed partial boundary differs from case lock")
    _accounting([*state.task_records, *state.signal_records])
    _validate_ledger_boundary(context, events)
    return state


def _archive_failed_state(context: FrozenContext) -> dict[str, Path]:
    case = context.case
    event_path = context.layout.events / f"adaptive-{case.cell_id}.jsonl"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{case.cell_id}.json"
    root = context.layout.results / "recovery" / case.cell_id
    archived_events = root / f"partial-events-{case.partial_event_count}.jsonl"
    archived_job = root / "failed-job.json"
    receipt_path = root / "archive-receipt.json"
    for source, destination, digest in (
        (event_path, archived_events, case.partial_file_sha256),
        (job_path, archived_job, case.failed_job_sha256),
    ):
        payload = source.read_bytes()
        if sha256_file(source) != digest:
            raise AdaptiveDeploymentError("failed source changed before archiving")
        if destination.exists():
            if sha256_file(destination) != digest or destination.read_bytes() != payload:
                raise AdaptiveDeploymentError("existing failed-state archive conflicts")
        elif atomic_write_bytes(destination, payload) != digest:
            raise AssertionError("failed-state archive hash drifted")
    receipt = {
        "artifact_type": "experiment12_online_adaptive_pre_recovery_archive",
        "run_id": RUN_ID,
        "cell_id": case.cell_id,
        # This receipt was created before the three capped semantic retries;
        # keep validating it against the exact case hash it originally bound.
        "case_sha256": (
            case.pre_recovery_archive_case_sha256 or context.case_sha256
        ),
        "archive_created_before_provider_dispatch": True,
        "source_partial_event_count": case.partial_event_count,
        "source_partial_prefix_sha256": case.partial_prefix_sha256,
        "source_partial_file_sha256": case.partial_file_sha256,
        "archived_partial_events": str(archived_events.relative_to(context.layout.root)),
        "archived_partial_events_sha256": sha256_file(archived_events),
        "source_failed_job_sha256": case.failed_job_sha256,
        "archived_failed_job": str(archived_job.relative_to(context.layout.root)),
        "archived_failed_job_sha256": sha256_file(archived_job),
        "ledger_rows_deleted_or_rewritten": False,
    }
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise AdaptiveDeploymentError("existing archive receipt changed")
    else:
        atomic_write_json(receipt_path, receipt)
    return {"root": root, "events": archived_events, "job": archived_job, "receipt": receipt_path}


class RecoveryTransport:
    def __init__(
        self,
        transport: Transport,
        *,
        case: RecoveryCase,
        logical_schedule: Sequence[str],
        missing_judge: str,
    ) -> None:
        self.transport = transport
        self.case = case
        self.logical_schedule = list(logical_schedule)
        self.missing_judge = missing_judge
        self.seen: list[str] = []
        self.judge_groups: list[dict[str, Any]] = []

    async def complete(
        self,
        model_name: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        logical = kwargs.get("request_key")
        expected = (
            self.logical_schedule[len(self.seen)]
            if len(self.seen) < len(self.logical_schedule)
            else None
        )
        if logical != expected:
            raise AdaptiveDeploymentError(
                f"recovery dispatch order changed: expected {expected!r}, got {logical!r}"
            )
        input_estimate = kwargs.get("input_token_estimate")
        input_cap = self.case.request_input_token_caps[len(self.seen)]
        if (
            isinstance(input_estimate, bool)
            or not isinstance(input_estimate, int)
            or input_estimate < 1
            or input_estimate > input_cap
        ):
            raise AdaptiveDeploymentError("recovery request exceeds its locked input cap")
        if "trace-judge" not in str(logical):
            if logical not in self.case.later_canonical_request_keys:
                raise AdaptiveDeploymentError("non-observer recovery key is not canonical")
            result = await self.transport.complete(model_name, messages, **kwargs)
            if len(result.attempts) != 1:
                raise AdaptiveDeploymentError("canonical suffix call was attempted twice")
            self.seen.append(str(logical))
            return result
        if model_name != "gpt-5.6-sol-judge":
            raise AdaptiveDeploymentError("missing judge model changed")
        if kwargs.get("max_output_tokens") != self.case.canonical_observer_max_output_tokens:
            raise AdaptiveDeploymentError("canonical observer cap changed before recovery")
        replacement = dict(kwargs)
        if logical == self.missing_judge:
            replacement["request_key"] = self.case.final_recovery_request_key
        elif logical not in self.case.later_canonical_request_keys:
            raise AdaptiveDeploymentError("unexpected recovery observer call")
        replacement["max_output_tokens"] = self.case.final_recovery_max_output_tokens
        result = await self.transport.complete(model_name, messages, **replacement)
        if len(result.attempts) != 1:
            raise AdaptiveDeploymentError("final capped observer call was attempted twice")
        call = _call_record(result)
        locked_dispatches = [
            {
                "logical_attempt_number": index,
                "semantic_attempt": lock.semantic_attempt,
                "physical_request_key": lock.request_key,
                "call_event_ids": [lock.event_id],
                "reservation_ids": [lock.reservation_id],
                "provider_request_id": lock.provider_request_id,
                "attempt_sha256": lock.attempt_sha256,
                "max_output_tokens": self.case.canonical_observer_max_output_tokens,
                "finish_reason": "max_output_tokens",
                "semantic_parse": "failed",
                "source": "preexisting_case_lock",
            }
            for index, lock in enumerate(self.case.malformed_attempts, start=1)
        ]
        if logical != self.missing_judge:
            locked_dispatches = []
        dispatch_request_key = str(replacement["request_key"])
        dispatch = {
            "logical_attempt_number": len(locked_dispatches) + 1,
            "semantic_attempt": len(locked_dispatches),
            "dispatch_request_key": dispatch_request_key,
            "physical_request_key": f"{dispatch_request_key}/attempt-1",
            "max_output_tokens": self.case.final_recovery_max_output_tokens,
            "call_event_ids": list(call["call_event_ids"]),
            "reservation_ids": [attempt.reservation_id for attempt in result.attempts],
            "provider_request_id": result.request_id,
            "finish_reason": result.finish_reason,
            "raw_output_sha256": sha256_json(result.text),
            "raw_output_characters": len(result.text),
        }
        group = {
            "logical_request_key": str(logical),
            "checkpoint": int(str(logical).rsplit("-", 1)[1]),
            "malformed_requests_consumed": len(locked_dispatches),
            "one_cell_output_cap_deviation": {
                "canonical_max_output_tokens": self.case.canonical_observer_max_output_tokens,
                "recovery_max_output_tokens": self.case.final_recovery_max_output_tokens,
            },
            "dispatches": [*locked_dispatches, dispatch],
        }
        try:
            parse_judge_output(result.text)
        except ValueError as exc:
            dispatch["semantic_parse"] = "failed"
            dispatch["semantic_error_type"] = type(exc).__name__
            self.judge_groups.append(group)
            raise AdaptiveDeploymentError(
                "a one-attempt 640-token recovery observer response was malformed"
            ) from exc
        dispatch["semantic_parse"] = "succeeded"
        self.judge_groups.append(group)
        self.seen.append(str(logical))
        return result


def _append_signal_decision(
    context: FrozenContext,
    state: PrefixState,
    signal: dict[str, Any],
    event_path: Path,
) -> dict[str, Any]:
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
    if decision["action_selected"]:
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
        append_jsonl(event_path, intervention)
        state.events.append(intervention)
        state.intervention_records.append(intervention)
    return decision


def _logical_suffix(context: FrozenContext) -> list[str]:
    checkpoint = context.case.missing_checkpoint
    return [
        _request_key(RUN_ID, context.case.cell_id, "trace-judge", checkpoint),
        *context.case.later_canonical_request_keys,
    ]


def _spend_proposal_from_caps(context: FrozenContext) -> dict[str, Any]:
    schedule = _logical_suffix(context)
    caps = context.case.request_input_token_caps
    requests: list[dict[str, Any]] = []
    total_reserved = Decimal("0")
    for index, (logical, input_cap) in enumerate(zip(schedule, caps, strict=True)):
        observer = "trace-judge" in logical
        model = "gpt-5.6-sol-judge" if observer else context.case.model
        max_output_tokens = (
            context.case.final_recovery_max_output_tokens
            if observer
            else context.config.task_max_output_tokens
        )
        dispatch_key = (
            context.case.final_recovery_request_key if index == 0 else logical
        )
        reserved = estimate_call_upper_bound_usd(model, input_cap, max_output_tokens)
        total_reserved += reserved
        requests.append(
            {
                "logical_request_key": logical,
                "dispatch_request_key": dispatch_key,
                "model": model,
                "input_token_reservation_cap": input_cap,
                "max_output_tokens": max_output_tokens,
                "max_reserved_usd": str(reserved),
            }
        )
    return {
        "maximum_new_provider_calls": len(schedule),
        "transport_attempts_per_request": 1,
        "requests": requests,
        "maximum_additional_reserved_usd": str(total_reserved),
    }


def _spend_proposal(
    context: FrozenContext, state: PrefixState
) -> dict[str, Any]:
    if context.threshold_lock.natural_max_actions_per_task != 1 or state.actions != 1:
        raise AdaptiveDeploymentError("recovery spend bound assumes the action cap is spent")
    judge_request = build_judge_request(
        state.messages,
        context.case.missing_checkpoint,
        benchmark=context.task.domain,
    )
    judge_input_bound = conservative_input_token_bound(
        judge_request,
        extra_bytes=len(str(JUDGE_RESPONSE_SCHEMA).encode("utf-8")),
    )
    task_turn = context.task.turns[context.case.missing_checkpoint]
    task_messages = [
        *state.messages,
        {"role": "user", "content": task_turn.user_message},
    ]
    task_input_bound = conservative_input_token_bound(task_messages)
    caps = context.case.request_input_token_caps
    if judge_input_bound != caps[0] or task_input_bound != caps[1]:
        raise AdaptiveDeploymentError("recovery conservative input bound changed")
    return _spend_proposal_from_caps(context)


async def _execute(
    context: FrozenContext,
    state: PrefixState,
    *,
    env_file: Path,
    spend_proposal: Mapping[str, Any],
) -> dict[str, Any]:
    case = context.case
    event_path = context.layout.events / f"adaptive-{case.cell_id}.jsonl"
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{case.cell_id}.json"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{case.cell_id}.json"
    archive = _archive_failed_state(context)
    transport = Transport(
        _stage_ledger(context.layout, RUN_ID, Stage.CONFIRMATORY),
        context.layout.events / "call_attempts.jsonl",
        environ=_environment(env_file),
        max_attempts=1,
    )
    schedule = _logical_suffix(context)
    recovery = RecoveryTransport(
        transport,
        case=case,
        logical_schedule=schedule,
        missing_judge=schedule[0],
    )
    checkpoint = case.missing_checkpoint
    signal = await _observe_current_prefix(
        run_id=RUN_ID,
        cell=context.cell,
        task=context.task,
        checkpoint=checkpoint,
        checkpoint_index=checkpoint,
        messages=state.messages,
        task_records=state.task_records,
        transport=recovery,  # type: ignore[arg-type]
        config=context.config,
        passive_spec=context.passive_spec,
    )
    if signal["source_prefix_before_observation_sha256"] != state.task_records[-1][
        "continued_history_sha256"
    ]:
        raise AssertionError("recovered signal observed another prefix")
    decision = _append_signal_decision(context, state, signal, event_path)
    if (
        case.actions_at_boundary
        >= context.threshold_lock.natural_max_actions_per_task
        and (decision["action_selected"] or decision["actions_after"] != state.actions)
    ):
        raise AssertionError("recovered checkpoint violated the frozen action cap")

    for turn_number in range(checkpoint + 1, len(context.task.turns) + 1):
        turn = context.task.turns[turn_number - 1]
        user = {"role": "user", "content": turn.user_message}
        state.messages.append(user)
        request_prefix_sha256 = sha256_json(state.messages)
        result = await recovery.complete(
            context.cell.pair_key.model,
            state.messages,
            purpose="adaptive_agent_turn",
            request_key=_request_key(RUN_ID, case.cell_id, "task", turn_number),
            input_token_estimate=conservative_input_token_bound(state.messages),
            max_output_tokens=context.config.task_max_output_tokens,
            temperature=context.config.temperature,
            reasoning_effort=DEFAULT_REASONING_EFFORT[context.cell.pair_key.model],
        )
        if result.tool_calls:
            raise AdaptiveDeploymentError("recovered task returned tool calls")
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
        if turn_number == len(context.task.turns):
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
        _append_signal_decision(context, state, signal, event_path)
    if recovery.seen != schedule:
        raise AssertionError("recovery did not consume its exact logical suffix")
    checkpoints = tuple(range(1, len(context.task.turns)))
    if (
        len(state.task_records) != len(context.task.turns)
        or len(state.signal_records) != len(checkpoints)
        or len(state.decision_records) != len(checkpoints)
    ):
        raise AssertionError("recovery coverage is incomplete")
    prediction, success = grade_final_numeric(
        state.assistant_task[-1], context.task.evaluation_label
    )
    transcript_sha256 = sha256_json(state.messages)
    accounting = _accounting([*state.task_records, *state.signal_records])
    kind, active_variant = _method_kind(context.cell.arm)
    output = {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "adaptive_runner_version": ADAPTIVE_RUNNER_VERSION,
        "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
        "deployment_policy": ADAPTIVE_POLICY,
        "run_id": RUN_ID,
        "cell_id": case.cell_id,
        "design_sha256": case.design_sha256,
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
        "probe_records": [],
        "evaluation": {
            "prediction": prediction,
            "evaluation_label_sha256": sha256_json(context.task.evaluation_label),
            "success": success,
        },
        "transcript_sha256": transcript_sha256,
        "accounting": accounting,
        "observation_burden": {
            "checkpoints": len(state.signal_records),
            "paid_observer_calls": accounting["by_category"]["observer"]["calls"],
            "carried_probe_calls": 0,
        },
        "event_log_prefix_sha256": sha256_json(state.events),
        "complete": True,
    }
    atomic_write_json(output_path, output)
    append_jsonl(
        event_path,
        {
            "event": "complete",
            "design_sha256": case.design_sha256,
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
    _write_completed_recovery_receipt(context, validated)
    return validated


def _write_completed_recovery_receipt(
    context: FrozenContext, validated: Mapping[str, Any]
) -> Path:
    """Verify committed production state and write only its generated receipt."""

    case = context.case
    event_path = context.layout.events / f"adaptive-{case.cell_id}.jsonl"
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{case.cell_id}.json"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{case.cell_id}.json"
    events = read_jsonl(event_path)
    signals = [
        event
        for event in events
        if event.get("event") == "signal_observed"
        and event.get("checkpoint") == case.missing_checkpoint
    ]
    final_tasks = [
        event
        for event in events
        if event.get("event") == "task_turn"
        and event.get("task_turn") == len(context.task.turns)
    ]
    if len(signals) != 1 or len(final_tasks) != 1:
        raise AdaptiveDeploymentError("completed recovery suffix is not unique")
    signal = signals[0]
    final_task = final_tasks[0]
    signal_ids = list((signal.get("call") or {}).get("call_event_ids") or ())
    task_ids = list((final_task.get("call") or {}).get("call_event_ids") or ())
    if len(signal_ids) != 1 or len(task_ids) != 1:
        raise AdaptiveDeploymentError("completed recovery used more than one new attempt")

    call_rows = read_jsonl(context.layout.events / "call_attempts.jsonl")
    by_event: dict[str, Mapping[str, Any]] = {}
    for row in call_rows:
        event_id = row.get("event_id") if isinstance(row, Mapping) else None
        if isinstance(event_id, str):
            if event_id in by_event:
                raise AdaptiveDeploymentError("duplicate call-attempt event ID")
            by_event[event_id] = row
    referenced = {
        event_id
        for event in events
        for event_id in ((event.get("call") or {}).get("call_event_ids") or ())
    }
    for event_id in referenced:
        if event_id not in by_event or by_event[event_id].get("status") != "succeeded":
            raise AdaptiveDeploymentError("completed call attempt is absent or invalid")
    for lock in case.malformed_attempts:
        row = by_event.get(lock.event_id)
        usage = row.get("usage") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not isinstance(usage, Mapping)
            or sha256_json(row) != lock.attempt_sha256
            or row.get("reservation_id") != lock.reservation_id
            or row.get("provider_request_id") != lock.provider_request_id
            or row.get("status") != "succeeded"
            or row.get("finish_reason") != "max_output_tokens"
            or row.get("attempt_number") != 1
            or usage.get("input_tokens") != lock.input_tokens
            or usage.get("output_tokens") != lock.output_tokens
            or usage.get("cached_input_tokens") != lock.cached_input_tokens
            or usage.get("reasoning_tokens") != lock.reasoning_tokens
        ):
            raise AdaptiveDeploymentError("completed malformed-attempt lock changed")

    final_observer = by_event[signal_ids[0]]
    final_task_attempt = by_event[task_ids[0]]
    if (
        final_observer.get("purpose") != "adaptive_trace_judge"
        or final_observer.get("model") != "gpt-5.6-sol"
        or final_observer.get("finish_reason") != "completed"
        or final_observer.get("attempt_number") != 1
        or final_observer.get("event_id")
        in {item.event_id for item in case.malformed_attempts}
        or final_task_attempt.get("purpose") != "adaptive_agent_turn"
        or final_task_attempt.get("model") != case.model
        or final_task_attempt.get("attempt_number") != 1
    ):
        raise AdaptiveDeploymentError("completed recovery attempt semantics changed")

    def validate_materialized_call(
        record: Mapping[str, Any], attempt: Mapping[str, Any]
    ) -> None:
        call = record.get("call")
        usage = attempt.get("usage")
        if not isinstance(call, Mapping) or not isinstance(usage, Mapping):
            raise AdaptiveDeploymentError("completed materialized call is malformed")
        expected_usage = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
        }
        if (
            call.get("call_event_ids") != [attempt.get("event_id")]
            or call.get("request_id") != attempt.get("provider_request_id")
            or call.get("finish_reason") != attempt.get("finish_reason")
            or call.get("elapsed_ms") != attempt.get("elapsed_ms")
            or call.get("accounted_cost_usd") != attempt.get("estimated_cost_usd")
            or call.get("usage") != expected_usage
        ):
            raise AdaptiveDeploymentError("completed call materialization changed")

    suffix_materialized: list[
        tuple[str, Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for logical in _logical_suffix(context):
        number = int(logical.rsplit("-", 1)[1])
        if "trace-judge" in logical:
            matches = [
                event
                for event in events
                if event.get("event") == "signal_observed"
                and event.get("checkpoint") == number
            ]
        else:
            matches = [
                event
                for event in events
                if event.get("event") == "task_turn"
                and event.get("task_turn") == number
            ]
        if len(matches) != 1:
            raise AdaptiveDeploymentError("completed logical suffix is not unique")
        ids = list((matches[0].get("call") or {}).get("call_event_ids") or ())
        if len(ids) != 1 or ids[0] not in by_event:
            raise AdaptiveDeploymentError("completed suffix call count changed")
        attempt = by_event[ids[0]]
        validate_materialized_call(matches[0], attempt)
        if "trace-judge" in logical:
            if (
                attempt.get("purpose") != "adaptive_trace_judge"
                or attempt.get("model") != "gpt-5.6-sol"
                or attempt.get("finish_reason") != "completed"
            ):
                raise AdaptiveDeploymentError("completed suffix observer changed")
        elif (
            attempt.get("purpose") != "adaptive_agent_turn"
            or attempt.get("model") != case.model
        ):
            raise AdaptiveDeploymentError("completed suffix task call changed")
        suffix_materialized.append((logical, matches[0], attempt))

    uri = context.layout.ledger.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        ledger_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM reservations WHERE request_key LIKE ?",
                (f"{RUN_ID}/{case.cell_id}/%",),
            )
        ]
    finally:
        connection.close()
    by_reservation = {str(row["reservation_id"]): row for row in ledger_rows}
    if len(by_reservation) != len(ledger_rows):
        raise AdaptiveDeploymentError("completed cell ledger has duplicate reservations")
    expected_reservations = {
        str(by_event[event_id]["reservation_id"]) for event_id in referenced
    } | {item.reservation_id for item in case.malformed_attempts}
    if set(by_reservation) != expected_reservations:
        raise AdaptiveDeploymentError("completed cell ledger contains an unknown call")
    for event_id in referenced | {item.event_id for item in case.malformed_attempts}:
        attempt = by_event[event_id]
        usage = attempt["usage"]
        ledger = by_reservation[str(attempt["reservation_id"])]
        actual_micro_usd = int(
            Decimal(str(attempt["estimated_cost_usd"])) * Decimal(1_000_000)
        )
        if (
            ledger.get("state") != "reconciled"
            or ledger.get("request_status") != "succeeded"
            or ledger.get("provider_request_id") != attempt.get("provider_request_id")
            or ledger.get("purpose") != attempt.get("purpose")
            or ledger.get("input_tokens") != usage.get("input_tokens")
            or ledger.get("output_tokens") != usage.get("output_tokens")
            or ledger.get("cached_input_tokens") != usage.get("cached_input_tokens")
            or ledger.get("reasoning_tokens") != usage.get("reasoning_tokens")
            or ledger.get("provider_total_tokens")
            != usage.get("provider_reported_total_tokens")
            or ledger.get("actual_micro_usd") != actual_micro_usd
        ):
            raise AdaptiveDeploymentError("completed attempt/ledger join changed")
    spend_proposal = _spend_proposal_from_caps(context)
    proposal_rows = spend_proposal["requests"]
    for index, (logical, _record, attempt) in enumerate(suffix_materialized):
        ledger = by_reservation[str(attempt["reservation_id"])]
        proposal = proposal_rows[index]
        expected_key = f"{proposal['dispatch_request_key']}/attempt-1"
        maximum_reserved = int(
            Decimal(str(proposal["max_reserved_usd"])) * Decimal(1_000_000)
        )
        if (
            proposal.get("logical_request_key") != logical
            or ledger.get("request_key") != expected_key
            or ledger.get("reserved_micro_usd") > maximum_reserved
        ):
            raise AdaptiveDeploymentError("completed recovery request key or bound changed")

    archive_root = context.layout.results / "recovery" / case.cell_id
    archive_receipt_path = archive_root / "archive-receipt.json"
    archived_events = archive_root / f"partial-events-{case.partial_event_count}.jsonl"
    archived_job = archive_root / "failed-job.json"
    archive_receipt = read_json(archive_receipt_path)
    if (
        sha256_file(archived_events) != case.partial_file_sha256
        or sha256_file(archived_job) != case.failed_job_sha256
        or archive_receipt.get("case_sha256")
        != (case.pre_recovery_archive_case_sha256 or context.case_sha256)
        or archive_receipt.get("source_partial_prefix_sha256")
        != case.partial_prefix_sha256
        or archive_receipt.get("archived_partial_events_sha256")
        != case.partial_file_sha256
        or archive_receipt.get("archived_failed_job_sha256")
        != case.failed_job_sha256
        or archive_receipt.get("ledger_rows_deleted_or_rewritten") is not False
    ):
        raise AdaptiveDeploymentError("completed recovery archive changed")

    locked_dispatches = [
        {
            "logical_attempt_number": index,
            "semantic_attempt": item.semantic_attempt,
            "physical_request_key": item.request_key,
            "call_event_ids": [item.event_id],
            "reservation_ids": [item.reservation_id],
            "provider_request_id": item.provider_request_id,
            "attempt_sha256": item.attempt_sha256,
            "max_output_tokens": case.canonical_observer_max_output_tokens,
            "finish_reason": "max_output_tokens",
            "semantic_parse": "failed",
            "source": "preexisting_case_lock",
        }
        for index, item in enumerate(case.malformed_attempts, start=1)
    ]
    final_dispatch = {
        "logical_attempt_number": len(locked_dispatches) + 1,
        "semantic_attempt": len(case.malformed_attempts),
        "dispatch_request_key": case.final_recovery_request_key,
        "physical_request_key": f"{case.final_recovery_request_key}/attempt-1",
        "max_output_tokens": case.final_recovery_max_output_tokens,
        "call_event_ids": signal_ids,
        "reservation_ids": [final_observer["reservation_id"]],
        "provider_request_id": final_observer["provider_request_id"],
        "attempt_sha256": sha256_json(final_observer),
        "finish_reason": final_observer["finish_reason"],
        "raw_output_sha256": sha256_json(signal["raw_output"]),
        "raw_output_characters": len(signal["raw_output"]),
        "semantic_parse": "succeeded",
        "source": "committed_production_signal",
    }
    judge_groups: list[dict[str, Any]] = [
        {
            "logical_request_key": _request_key(
                RUN_ID, case.cell_id, "trace-judge", case.missing_checkpoint
            ),
            "checkpoint": case.missing_checkpoint,
            "malformed_requests_consumed": len(case.malformed_attempts),
            "one_cell_output_cap_deviation": {
                "canonical_max_output_tokens": 320,
                "recovery_max_output_tokens": 640,
            },
            "dispatches": [*locked_dispatches, final_dispatch],
        }
    ]
    for logical, record, attempt in suffix_materialized[1:]:
        if "trace-judge" not in logical:
            continue
        judge_groups.append(
            {
                "logical_request_key": logical,
                "checkpoint": int(logical.rsplit("-", 1)[1]),
                "malformed_requests_consumed": 0,
                "one_cell_output_cap_deviation": {
                    "canonical_max_output_tokens": 320,
                    "recovery_max_output_tokens": 640,
                },
                "dispatches": [
                    {
                        "logical_attempt_number": 1,
                        "semantic_attempt": 0,
                        "dispatch_request_key": logical,
                        "physical_request_key": f"{logical}/attempt-1",
                        "max_output_tokens": 640,
                        "call_event_ids": [attempt["event_id"]],
                        "reservation_ids": [attempt["reservation_id"]],
                        "provider_request_id": attempt["provider_request_id"],
                        "attempt_sha256": sha256_json(attempt),
                        "finish_reason": attempt["finish_reason"],
                        "raw_output_sha256": sha256_json(record["raw_output"]),
                        "raw_output_characters": len(record["raw_output"]),
                        "semantic_parse": "succeeded",
                        "source": "committed_production_signal",
                    }
                ],
            }
        )
    case_path = context.case_path
    if not case_path.is_absolute():
        case_path = REPOSITORY_ROOT / case_path
    receipt = {
        "artifact_type": "experiment12_online_adaptive_trace_judge_recovery",
        "recovery_version": 3,
        "run_id": RUN_ID,
        "cell_id": case.cell_id,
        "source_task_id": case.task_id,
        "case_file": str(case_path.resolve().relative_to(REPOSITORY_ROOT.resolve())),
        "case_sha256": context.case_sha256,
        "original_partial_event_count": case.partial_event_count,
        "original_partial_prefix_sha256": case.partial_prefix_sha256,
        "original_partial_file_sha256": case.partial_file_sha256,
        "original_failed_job_sha256": case.failed_job_sha256,
        "pre_recovery_archive": str(archive_root.relative_to(context.layout.root)),
        "pre_recovery_archive_receipt_sha256": sha256_file(archive_receipt_path),
        "malformed_checkpoint": case.missing_checkpoint,
        "malformed_attempts": [
            {
                "semantic_attempt": item.semantic_attempt,
                "request_key": item.request_key,
                "event_id": item.event_id,
                "attempt_sha256": item.attempt_sha256,
                "reservation_id": item.reservation_id,
                "provider_request_id": item.provider_request_id,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "cached_input_tokens": item.cached_input_tokens,
                "reasoning_tokens": item.reasoning_tokens,
                "actual_micro_usd": item.actual_micro_usd,
            }
            for item in case.malformed_attempts
        ],
        "final_recovery_request_key": case.final_recovery_request_key,
        "canonical_observer_max_output_tokens": case.canonical_observer_max_output_tokens,
        "final_recovery_max_output_tokens": case.final_recovery_max_output_tokens,
        "maximum_new_provider_calls": spend_proposal["maximum_new_provider_calls"],
        "transport_attempts_per_request": 1,
        "spend_proposal": spend_proposal,
        "judge_recovery_groups": judge_groups,
        "final_task_turn": {
            "request_key": case.later_canonical_request_keys[-1],
            "event_id": final_task_attempt["event_id"],
            "attempt_sha256": sha256_json(final_task_attempt),
            "reservation_id": final_task_attempt["reservation_id"],
            "provider_request_id": final_task_attempt["provider_request_id"],
        },
        "logical_suffix": _logical_suffix(context),
        "ledger_rows_deleted_or_rewritten": False,
        "production_was_complete_before_receipt_write": True,
        "final_event_log_sha256": sha256_file(event_path),
        "final_output_sha256": sha256_file(output_path),
        "final_job_sha256": sha256_file(job_path),
        "accounting_sha256": sha256_json(validated["accounting"]),
    }
    receipt_path = (
        REPOSITORY_ROOT
        / f"experiments12/data_results/derived/recovery-adaptive-{case.cell_id}12.json"
    )
    if receipt_path.exists():
        if read_json(receipt_path) != receipt:
            raise AdaptiveDeploymentError("existing completed recovery receipt changed")
    else:
        atomic_write_json(receipt_path, receipt)
    return receipt_path


def _validate_completed(context: FrozenContext) -> dict[str, Any] | None:
    case = context.case
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{case.cell_id}.json"
    if not output_path.exists():
        return None
    event_path = context.layout.events / f"adaptive-{case.cell_id}.jsonl"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{case.cell_id}.json"
    output = _validate_existing(
        output_file=output_path,
        event_file=event_path,
        start=context.start,
        task=context.task,
        compaction_config=context.compaction,
    )
    job = read_json(job_path)
    if job.get("state") != "complete" or job.get("output_sha256") != sha256_file(output_path):
        raise AdaptiveDeploymentError("completed recovery job conflicts")
    return output


async def _main(args: argparse.Namespace) -> int:
    context = _load_context(Path(args.case), args.case_sha256)
    completed = _validate_completed(context)
    if completed is not None:
        receipt = _write_completed_recovery_receipt(context, completed)
        print(
            f"already complete: cell={context.case.cell_id} "
            f"receipt={receipt} receipt_sha256={sha256_file(receipt)}"
        )
        return 0
    state = _validate_and_replay_prefix(context)
    spend_proposal = _spend_proposal(context, state)
    if not args.execute:
        archive = _archive_failed_state(context) if args.archive else None
        suffix = ",".join(_logical_suffix(context))
        print(
            f"audit passed: cell={context.case.cell_id} events={len(state.events)} "
            f"missing=trace_judge@{context.case.missing_checkpoint} suffix={suffix} "
            f"final_observer_cap={context.case.final_recovery_max_output_tokens} "
            f"maximum_new_calls={spend_proposal['maximum_new_provider_calls']} "
            "maximum_additional_reserved_usd="
            f"{spend_proposal['maximum_additional_reserved_usd']}"
            + (" archive=" + str(archive["root"]) if archive else "")
        )
        return 0
    if not args.yes_spend:
        raise AdaptiveDeploymentError("paid recovery requires --execute --yes-spend")
    if (
        args.confirm_final_observer_max_output_tokens
        != context.case.final_recovery_max_output_tokens
    ):
        raise AdaptiveDeploymentError("explicit confirmation of the 640-token cap is required")
    expected_calls = int(spend_proposal["maximum_new_provider_calls"])
    if args.confirm_maximum_new_provider_calls != expected_calls:
        raise AdaptiveDeploymentError(
            "explicit confirmation of the maximum new calls is required"
        )
    result = await _execute(
        context,
        state,
        env_file=Path(args.env_file),
        spend_proposal=spend_proposal,
    )
    print(
        f"recovery complete: cell={context.case.cell_id} "
        f"success={result['evaluation']['success']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--case-sha256", required=True)
    parser.add_argument("--archive", action="store_true")
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
