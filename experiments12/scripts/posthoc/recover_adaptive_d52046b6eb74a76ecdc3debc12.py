"""Recover one online-adaptive cell after a truncated trace-judge response.

The default invocation is a provider-free audit.  Paid recovery is deliberately
gated by both ``--execute`` and ``--yes-spend``.  This script is locked to the
exact 14-event prefix and failed job receipt named below; it never truncates an
event log, removes a reservation, or reuses a provider request key.
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
from experiments12.monitors.judge import parse_judge_output
from experiments12.operators12 import CompactionConfig, freeze_initial_instructions
from experiments12.pairing12 import JobCell
from experiments12.passive_spec12 import passive_monitor_spec_from_manifest
from experiments12.runner12 import _stage_ledger, _validate_run_inputs
from experiments12.spec12 import Stage


RUN_ID = "e12-deploy-online-evolving-luna-40-v1"
CELL_ID = "d52046b6eb74a76ecdc3debc"
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
THRESHOLD_LOCK_PATH = (
    ARTIFACTS_ROOT / RUN_ID / "results/deployment_threshold_lock.json"
)
DATASET_SHA256 = "6bdd6eb969a6c3f93e495e0d21be1055d1423071b2815c938949f44eac4a16ad"
MANIFEST_SHA256 = "7294e3edcd3468aeb8881182f77a3f22c9d9de41dff1dba8f183e9fe1753c9a7"
PAIR_MANIFEST_SHA256 = "16ebad63b1119ed79887e3d62f2ea1b8a7df69021a8254b27923b54314af70c6"
THRESHOLD_LOCK_SHA256 = "061216da43506e13159eada54226c697cd94d0a72da8203c05605a69e14247d2"
DESIGN_SHA256 = "d0a46b93d523c2a30b29c470ef7a9c885a63676436da7133564c9df28009d4ec"

# Both the canonical JSON sequence and its exact JSONL encoding are locked.
PARTIAL_EVENT_COUNT = 14
PARTIAL_PREFIX_SHA256 = "0c7406c8c4da92a8dc4c7243f99da27525e6e54e4c66ec22021df518ae87bb7c"
PARTIAL_FILE_SHA256 = "e2ef3fde80a146faf134e07be4bdba42a488a7709284dc681d71b19998a9e631"
FAILED_JOB_SHA256 = "c38f8ba472f2858d7d4fafacac7f143a67bfd60f372b4abb52f0de9704bda9a5"
FAILED_JOB = {
    "adaptive_runner_version": 1,
    "cell_id": CELL_ID,
    "error_type": "ValueError",
    "state": "failed",
}

STANDARD_JUDGE_5 = _request_key(RUN_ID, CELL_ID, "trace-judge", 5)
STANDARD_JUDGE_6 = _request_key(RUN_ID, CELL_ID, "trace-judge", 6)
RECOVERY_JUDGE_5_PREFIX = (
    f"{RUN_ID}/{CELL_ID}/adaptive-trace-judge-5-recovery-semantic"
)
RECOVERY_JUDGE_6_PREFIX = (
    f"{RUN_ID}/{CELL_ID}/adaptive-trace-judge-6-recovery-semantic"
)
LATER_STANDARD_KEYS = (
    _request_key(RUN_ID, CELL_ID, "task", 6),
    STANDARD_JUDGE_6,
    _request_key(RUN_ID, CELL_ID, "task", 7),
)
MALFORMED_PROVIDER_REQUEST_ID = "req_93198ef701da46ba858ff5def7790ec7"


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


def _load_context() -> FrozenContext:
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
        or cell.operator != "lossy_compaction"
        or cell.pair_key.model != "gpt-5.6-luna"
        or cell.pair_key.task_id != "extracted-gsm8k-test-814::t7"
        or str(cell.pair_key.task_sha256)
        != "ab0b0f048d3951f0fa7343727aeedb74545969d740c08fd59e5037aedda04ef7"
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
        or malformed["output_tokens"] != 320
        or malformed["reasoning_tokens"] != 241
        or malformed["actual_micro_usd"] != 11405
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
        or malformed_call.get("status") != "succeeded"
        or malformed_call.get("finish_reason") != "max_output_tokens"
        or malformed_call.get("provider_request_id") != MALFORMED_PROVIDER_REQUEST_ID
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

    if cursor != len(events) or actions != 0 or intervention_records:
        raise AdaptiveDeploymentError("partial prefix is not the locked no-action boundary")
    if sha256_json(messages) != (
        "f58cf30692da6432e3bd477745bae0670bc2c9f389f4b556c7ea7edce7c35d7b"
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


def _archive_failed_state(context: FrozenContext) -> dict[str, Path]:
    """Write-once preservation of artifacts that canonical completion replaces."""

    event_path = context.layout.events / f"adaptive-{CELL_ID}.jsonl"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{CELL_ID}.json"
    archive_root = context.layout.results / "recovery" / CELL_ID
    archived_events = archive_root / "partial-events-14.jsonl"
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
    """Change only the consumed checkpoint-5 judge key; reject every surprise."""

    def __init__(self, transport: Transport, *, max_semantic_attempts: int) -> None:
        self.transport = transport
        self.max_semantic_attempts = max_semantic_attempts
        self.expected = [STANDARD_JUDGE_5, *LATER_STANDARD_KEYS]
        self.seen: list[str] = []
        self.semantic_failures: list[dict[str, Any]] = []
        self.recovery_request_keys: list[str] = []
        self.observer_dispatch_keys: list[str] = []

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
        if request_key not in {STANDARD_JUDGE_5, STANDARD_JUDGE_6}:
            result = await self.transport.complete(model_name, messages, **kwargs)
            self.seen.append(str(request_key))
            return result

        for semantic_attempt in range(1, self.max_semantic_attempts + 1):
            if request_key == STANDARD_JUDGE_5:
                dispatch_key = f"{RECOVERY_JUDGE_5_PREFIX}-{semantic_attempt}"
            elif semantic_attempt == 1:
                # Checkpoint 6 is unused at the recovery boundary, so retain its
                # canonical key unless a semantically malformed response consumes it.
                dispatch_key = STANDARD_JUDGE_6
            else:
                dispatch_key = f"{RECOVERY_JUDGE_6_PREFIX}-{semantic_attempt}"
            replacement = dict(kwargs)
            replacement["request_key"] = dispatch_key
            result = await self.transport.complete(model_name, messages, **replacement)
            self.observer_dispatch_keys.append(dispatch_key)
            if dispatch_key != request_key:
                self.recovery_request_keys.append(dispatch_key)
            try:
                parse_judge_output(result.text)
            except ValueError as exc:
                call = _call_record(result)
                self.semantic_failures.append(
                    {
                        "logical_request_key": request_key,
                        "dispatch_request_key": dispatch_key,
                        "semantic_attempt": semantic_attempt,
                        "error_type": type(exc).__name__,
                        "raw_output_sha256": sha256_json(result.text),
                        "raw_output_characters": len(result.text),
                        "finish_reason": result.finish_reason,
                        "call_event_ids": list(call["call_event_ids"]),
                    }
                )
                continue
            self.seen.append(str(request_key))
            return result
        checkpoint = 5 if request_key == STANDARD_JUDGE_5 else 6
        raise AdaptiveDeploymentError(
            f"all bounded checkpoint-{checkpoint} recovery responses were malformed"
        )


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
    max_semantic_attempts: int,
) -> dict[str, Any]:
    event_path = context.layout.events / f"adaptive-{CELL_ID}.jsonl"
    output_path = context.layout.results / ADAPTIVE_RESULT_SUBDIR / f"{CELL_ID}.json"
    job_path = context.layout.results / ADAPTIVE_JOB_SUBDIR / f"{CELL_ID}.json"
    archive = _archive_failed_state(context)
    transport = Transport(
        _stage_ledger(context.layout, RUN_ID, Stage.CONFIRMATORY),
        context.layout.events / "call_attempts.jsonl",
        environ=_environment(env_file),
        max_attempts=6,
    )
    recovery = RecoveryTransport(
        transport,
        max_semantic_attempts=max_semantic_attempts,
    )

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
        / "experiments12/data_results/derived/recovery-adaptive-d52046b6eb74a76ecdc3debc12.json"
    )
    atomic_write_json(
        receipt_path,
        {
            "artifact_type": "experiment12_online_adaptive_single_cell_recovery",
            "run_id": RUN_ID,
            "cell_id": CELL_ID,
            "original_partial_event_count": PARTIAL_EVENT_COUNT,
            "original_partial_prefix_sha256": PARTIAL_PREFIX_SHA256,
            "original_partial_file_sha256": PARTIAL_FILE_SHA256,
            "original_failed_job_sha256": FAILED_JOB_SHA256,
            "pre_recovery_archive": str(archive["root"].relative_to(context.layout.root)),
            "pre_recovery_archive_receipt_sha256": sha256_file(archive["receipt"]),
            "archived_failed_job_sha256": sha256_file(archive["job"]),
            "archived_partial_events_sha256": sha256_file(archive["events"]),
            "original_malformed_request_key": f"{STANDARD_JUDGE_5}/attempt-1",
            "original_malformed_provider_request_id": MALFORMED_PROVIDER_REQUEST_ID,
            "recovery_request_keys": recovery.recovery_request_keys,
            "observer_dispatch_keys": recovery.observer_dispatch_keys,
            "semantic_parse_failures_before_success": recovery.semantic_failures,
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
            "missing=trace_judge@5 later_calls=task-6,trace-judge-6,task-7"
            f"{archive_note}"
        )
        return 0
    if not args.yes_spend:
        raise AdaptiveDeploymentError("paid recovery requires --execute --yes-spend")
    result = await _execute_recovery(
        context,
        state,
        env_file=Path(args.env_file),
        max_semantic_attempts=args.max_semantic_attempts,
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
    parser.add_argument("--max-semantic-attempts", type=int, default=3)
    parser.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    args = parser.parse_args()
    if args.yes_spend and not args.execute:
        raise AdaptiveDeploymentError("--yes-spend is invalid without --execute")
    if not 1 <= args.max_semantic_attempts <= 3:
        raise AdaptiveDeploymentError("max semantic attempts must be between one and three")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
