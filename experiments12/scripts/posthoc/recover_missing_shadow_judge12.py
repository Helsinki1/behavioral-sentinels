"""Recover one fail-closed shadow whose only missing record is the final judge.

This does not replay already paid monitor calls.  It verifies the archived
partial event sequence is the exact canonical prefix, issues the one missing
judge request under a new audited request key, then rebuilds and validates the
standard shadow artifact before marking its existing declared cell complete.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

from experiments12.cli12 import REPOSITORY_ROOT, _environment
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.transport import JsonSchemaOutput, Transport
from experiments12.harness12 import _call_record, conservative_input_token_bound
from experiments12.manifest12 import RunLayout, code_tree_hash
from experiments12.monitors.judge import (
    JUDGE_RESPONSE_SCHEMA,
    build_judge_request,
    parse_judge_output,
)
from experiments12.pairing12 import JobCell
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    passive_monitor_spec_from_manifest,
    quiz_generator_spec,
)
from experiments12.runner12 import _job_state, _stage_ledger
from experiments12.shadow12 import (
    SHADOW_VERSION,
    _checkpoint_turns,
    _expected_shadow_coverage,
    _prefix,
    _validate_shadow_materialization,
)
from experiments12.spec12 import Stage


def _signature(record: dict[str, Any]) -> tuple[int, str, str | None]:
    variant = record.get("variant") if record.get("method") == "frozen_probe" else None
    return int(record["checkpoint_turn"]), str(record["method"]), variant


async def _recover(args: argparse.Namespace) -> None:
    layout = RunLayout.for_run(args.artifacts, args.run_id)
    manifest = read_json(layout.manifest)
    if manifest.get("run_id") != args.run_id or manifest.get("stage") != "confirmatory":
        raise ValueError("recovery run identity/stage differs from the frozen manifest")
    if code_tree_hash(REPOSITORY_ROOT / "experiments12") != manifest["repository"][
        "code_tree_sha256"
    ]:
        raise ValueError("source tree differs from the frozen run")

    cells = [JobCell.from_dict(row) for row in read_jsonl(layout.pairs)]
    matches = [cell for cell in cells if cell.cell_id == args.cell_id]
    if len(matches) != 1 or matches[0].arm != "clean" or matches[0].operator != "none":
        raise ValueError("recovery cell is not one declared clean/no-operator cell")
    cell = matches[0]
    trajectory = read_json(layout.trajectories / f"{args.cell_id}.json")
    if trajectory.get("complete") is not True or trajectory.get("arm") != "clean":
        raise ValueError("recovery source is not a complete clean trajectory")

    partial_path = Path(args.partial_events)
    if sha256_file(partial_path) != args.partial_sha256:
        raise ValueError("archived partial event hash differs from the recovery lock")
    partial = read_jsonl(partial_path)
    if any(not isinstance(row, dict) for row in partial):
        raise ValueError("archived partial event file contains a non-object")

    spec = passive_monitor_spec_from_manifest(manifest)
    checkpoints = _checkpoint_turns(trajectory)
    variants = tuple(spec["frozen_probe"]["variants"])
    methods = tuple(spec["required_methods"])
    expected = _expected_shadow_coverage(checkpoints, variants, methods)
    observed = Counter(_signature(row) for row in partial)
    missing_key = (checkpoints[-1], "trace_judge", None)
    if expected - observed != Counter({missing_key: 1}) or observed - expected:
        raise ValueError("partial shadow is not missing exactly the final trace judge")

    canonical_order: list[tuple[int, str, str | None]] = []
    for turn in checkpoints:
        canonical_order.extend(
            [
                (turn, "turn_clock", None),
                (turn, "context_use", None),
                (turn, "trace_rules", None),
                *((turn, "frozen_probe", variant) for variant in variants),
                (turn, "frozen_quiz", None),
                (turn, "trace_judge", None),
            ]
        )
    if [_signature(row) for row in partial] != canonical_order[:-1]:
        raise ValueError("partial shadow records are not the exact canonical prefix")

    source_sha = trajectory["transcript_sha256"]
    prefix_sha256_by_checkpoint = {
        turn: sha256_json(_prefix(trajectory, turn)) for turn in checkpoints
    }
    for record in partial:
        turn = int(record["checkpoint_turn"])
        if (
            record.get("source_trajectory_sha256") != source_sha
            or record.get("source_prefix_sha256") != prefix_sha256_by_checkpoint[turn]
            or record.get("passive_monitor_spec_sha256")
            != PASSIVE_MONITOR_SPEC_SHA256
        ):
            raise ValueError("partial shadow record provenance differs from its source")

    turn = checkpoints[-1]
    prefix = _prefix(trajectory, turn)
    judge = spec["trace_judge"]
    request = build_judge_request(prefix, turn, benchmark=trajectory["domain"])
    schema = JsonSchemaOutput.from_schema("trace_risk", JUDGE_RESPONSE_SCHEMA)
    transport = Transport(
        _stage_ledger(layout, args.run_id, Stage.CONFIRMATORY),
        layout.events / "call_attempts.jsonl",
        environ=_environment(args.env_file),
        max_attempts=6,
    )
    parse_failures: list[dict[str, Any]] = []
    result = verdict = None
    for semantic_attempt in range(1, args.max_semantic_attempts + 1):
        result = await transport.complete(
            judge["model"],
            request,
            purpose="trace_judge",
            request_key=(
                f"{args.run_id}/shadow-recovery/{source_sha[:20]}/"
                f"judge-{len(checkpoints)}-semantic-{semantic_attempt}"
            ),
            input_token_estimate=conservative_input_token_bound(
                request, extra_bytes=len(str(JUDGE_RESPONSE_SCHEMA).encode("utf-8"))
            ),
            max_output_tokens=judge["max_output_tokens"],
            temperature=spec["determinism"]["temperature"],
            reasoning_effort=judge["reasoning_effort"],
            output_schema=schema,
        )
        try:
            verdict = parse_judge_output(result.text)
            break
        except ValueError as exc:
            parse_failures.append(
                {
                    "semantic_attempt": semantic_attempt,
                    "error_type": type(exc).__name__,
                    "raw_output_sha256": sha256_json(result.text),
                    "raw_output_characters": len(result.text),
                    "call_event_ids": list(_call_record(result)["call_event_ids"]),
                }
            )
    if result is None or verdict is None:
        atomic_write_json(
            Path(args.recovery_receipt).with_name("failed-semantic-recovery.json"),
            {"parse_failures": parse_failures},
        )
        raise ValueError("all bounded semantic judge recovery attempts failed")

    judge_record = {
        "method": "trace_judge",
        "checkpoint_turn": turn,
        "actionable_before_turn": turn + 1,
        "score": verdict.risk,
        "fired": None,
        "concerns": list(verdict.concerns),
        "evidence": list(verdict.evidence),
        "monitor_spec_sha256": verdict.spec_sha256,
        "source_trajectory_sha256": source_sha,
        "source_prefix_sha256": prefix_sha256_by_checkpoint[turn],
        "raw_output": result.text,
        "call": _call_record(result),
        "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
    }
    records = [*partial, judge_record]
    quiz_generator = quiz_generator_spec(spec, trajectory["domain"])
    materialized = {
        "schema_version": 1,
        "shadow_version": SHADOW_VERSION,
        "source_trajectory_sha256": source_sha,
        "model": trajectory["model"],
        "domain": trajectory["domain"],
        "task_id": trajectory["task_id"],
        "condition": trajectory["condition"],
        "checkpoint_turns": list(checkpoints),
        "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
        "quiz_generator": dict(quiz_generator),
        "records": records,
        "monitor_methods": sorted({row["method"] for row in records}),
        "complete": True,
    }
    _validate_shadow_materialization(
        materialized,
        trajectory=trajectory,
        source_sha=source_sha,
        checkpoints=checkpoints,
        spec=spec,
        quiz_generator=quiz_generator,
        prefix_sha256_by_checkpoint=prefix_sha256_by_checkpoint,
    )

    event_path = layout.events / f"shadow-{args.cell_id}.jsonl"
    output_path = layout.shadow / f"{args.cell_id}.json"
    job_path = layout.results / "shadow_jobs" / f"{args.cell_id}.json"
    if event_path.exists() or output_path.exists() or job_path.exists():
        raise FileExistsError("recovery destinations must be absent")
    atomic_write_jsonl(event_path, records)
    atomic_write_json(output_path, materialized)
    _job_state(
        job_path,
        cell=cell,
        state="complete",
        detail={
            "shadow_sha256": sha256_file(output_path),
            "monitor_methods": materialized["monitor_methods"],
            "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
        },
    )
    atomic_write_json(
        args.recovery_receipt,
        {
            "artifact_type": "experiment12_single_missing_shadow_judge_recovery",
            "run_id": args.run_id,
            "cell_id": args.cell_id,
            "source_trajectory_sha256": source_sha,
            "archived_partial_events": str(partial_path),
            "archived_partial_events_sha256": args.partial_sha256,
            "archived_partial_record_count": len(partial),
            "recovered_method": "trace_judge",
            "recovered_checkpoint": turn,
            "recovery_request_key_is_distinct": True,
            "semantic_parse_failures_before_success": parse_failures,
            "recovery_call_event_ids": list(
                judge_record["call"]["call_event_ids"]
            ),
            "final_event_log_sha256": sha256_file(event_path),
            "final_shadow_sha256": sha256_file(output_path),
            "final_job_sha256": sha256_file(job_path),
            "code_tree_sha256": manifest["repository"]["code_tree_sha256"],
            "source_code_or_scientific_values_changed": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--partial-events", required=True)
    parser.add_argument("--partial-sha256", required=True)
    parser.add_argument("--recovery-receipt", required=True)
    parser.add_argument("--max-semantic-attempts", type=int, default=3)
    parser.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_semantic_attempts <= 3:
        raise ValueError("max semantic attempts must be between one and three")
    asyncio.run(_recover(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
