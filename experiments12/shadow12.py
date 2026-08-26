"""Passive zero-carry scoring on immutable clean trajectory prefixes."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_json,
    read_json,
    sha256_json,
)
from experiments12.core.transport import JsonSchemaOutput, Transport
from experiments12.harness12 import _call_record, conservative_input_token_bound
from experiments12.models12 import CATALOG
from experiments12.monitors.frozen_probe import build_frozen_probe_fork
from experiments12.monitors.frozen_quiz import QuizQuestion, build_quiz_fork, grade_quiz
from experiments12.monitors.judge import (
    JUDGE_RESPONSE_SCHEMA,
    build_judge_request,
    parse_judge_output,
)
from experiments12.monitors.trace_rules import score_trace_rules
from experiments12.passive_quizzes12 import (
    generate_bfcl_passive_quiz,
    generate_evolving_passive_quiz,
)
from experiments12.probes12 import (
    CURRENT_COPY,
    RECOMPUTE,
    generate_probe_instance,
    grade_probe_response,
)
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    assert_passive_runtime_overrides,
    canonical_passive_monitor_spec,
    quiz_generator_spec,
    validate_passive_monitor_spec,
)


SHADOW_VERSION = 2
STATELESS_FROZEN_VARIANTS = (CURRENT_COPY, RECOMPUTE)


def _prefix(trajectory: Mapping[str, Any], after_turn: int) -> list[dict[str, Any]]:
    records = trajectory.get("task_records")
    if not isinstance(records, list) or not 1 <= after_turn <= len(records):
        raise ValueError("trajectory has no valid task-record prefix")
    messages: list[dict[str, Any]] = []
    for record in records[:after_turn]:
        if record.get("event") != "task_turn":
            raise ValueError("task_records contains a non-task event")
        turn_messages = record.get("messages")
        if turn_messages is None:
            turn_messages = (record.get("user_message"), record.get("assistant_message"))
        if (
            isinstance(turn_messages, (str, bytes))
            or not isinstance(turn_messages, Sequence)
            or not turn_messages
            or any(not isinstance(message, Mapping) for message in turn_messages)
        ):
            raise ValueError("task record contains an invalid public message sequence")
        messages.extend(dict(message) for message in turn_messages)
    return messages


def _latest_input_tokens(record: Mapping[str, Any]) -> int:
    """Return the latest request's context size, not a multi-call turn sum."""

    calls = record.get("calls")
    if isinstance(calls, list) and calls:
        call = calls[-1]
    else:
        call = record.get("call")
    if not isinstance(call, Mapping) or not isinstance(call.get("usage"), Mapping):
        raise ValueError("task record lacks latest-call usage")
    value = call["usage"].get("input_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("task record has invalid latest input-token usage")
    return value


def _observed_rule_flags(
    task_records: Sequence[Mapping[str, Any]], after_turn: int
) -> dict[str, bool]:
    """Map only already observed BFCL failures into the frozen rule vocabulary."""

    observed = {
        "invalid_tool_call": False,
        "execution_error": False,
        "tool_result_error": False,
        "protocol_violation": False,
    }
    for record in task_records[:after_turn]:
        indicators = record.get("failure_indicators")
        if indicators is None:
            continue
        if not isinstance(indicators, Mapping):
            raise ValueError("task failure_indicators must be an object")
        for key in (
            "invalid_call_observed",
            "execution_failure_observed",
            "state_check_failure_observed",
        ):
            if not isinstance(indicators.get(key), bool):
                raise ValueError("task failure_indicators are incomplete")
        observed["invalid_tool_call"] |= indicators["invalid_call_observed"]
        observed["execution_error"] |= indicators["execution_failure_observed"]
        observed["tool_result_error"] |= indicators["execution_failure_observed"]
        observed["protocol_violation"] |= indicators["state_check_failure_observed"]
    return observed


def _monitor_key(
    run_id: str,
    source_sha: str,
    method: str,
    checkpoint: int,
    variant: str | None = None,
) -> str:
    suffix = "" if variant is None else f"-{variant}"
    return f"{run_id}/shadow/{source_sha[:20]}/{method}{suffix}-{checkpoint}"


def _checkpoint_turns(trajectory: Mapping[str, Any]) -> tuple[int, ...]:
    records = trajectory.get("task_records")
    if not isinstance(records, list) or not records:
        raise ValueError("clean trajectory has no completed task records")
    raw = trajectory.get("checkpoint_turns")
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw
    ):
        raise ValueError("clean trajectory checkpoint_turns must be an integer list")
    checkpoints = tuple(raw)
    expected = tuple(range(1, len(records)))
    if checkpoints != expected:
        raise ValueError(
            "clean trajectory checkpoints differ from the frozen after-each-nonfinal schedule"
        )
    return checkpoints


def _expected_shadow_coverage(
    checkpoints: Sequence[int],
    variants: Sequence[str],
    required_methods: Sequence[str],
) -> Counter[tuple[int, str, str | None]]:
    expected: Counter[tuple[int, str, str | None]] = Counter()
    for turn in checkpoints:
        for method in required_methods:
            if method == "frozen_probe":
                for variant in variants:
                    expected[(turn, method, variant)] += 1
            else:
                expected[(turn, method, None)] += 1
    return expected


def _validate_shadow_materialization(
    materialized: Mapping[str, Any],
    *,
    trajectory: Mapping[str, Any],
    source_sha: str,
    checkpoints: Sequence[int],
    spec: Mapping[str, Any],
    quiz_generator: Mapping[str, Any],
    prefix_sha256_by_checkpoint: Mapping[int, str],
) -> None:
    """Reject reuse unless every frozen monitor appears exactly once as declared."""

    if (
        materialized.get("schema_version") != 1
        or materialized.get("shadow_version") != SHADOW_VERSION
    ):
        raise ValueError("shadow output schema/version changed")
    if materialized.get("source_trajectory_sha256") != source_sha:
        raise ValueError("shadow output belongs to another source trajectory")
    for key in ("model", "domain", "task_id", "condition"):
        if materialized.get(key) != trajectory.get(key):
            raise ValueError(f"shadow output {key} differs from its source trajectory")
    if materialized.get("passive_monitor_spec_sha256") != PASSIVE_MONITOR_SPEC_SHA256:
        raise ValueError("shadow output belongs to another passive monitor spec")
    if materialized.get("checkpoint_turns") != list(checkpoints):
        raise ValueError("shadow output checkpoint schedule changed")
    if materialized.get("quiz_generator") != dict(quiz_generator):
        raise ValueError("shadow output quiz generator changed")
    if materialized.get("complete") is not True:
        raise ValueError("shadow output is incomplete")
    records = materialized.get("records")
    if not isinstance(records, list):
        raise ValueError("shadow output records are invalid")
    variants = tuple(spec["frozen_probe"]["variants"])
    methods = tuple(spec["required_methods"])
    observed: Counter[tuple[int, str, str | None]] = Counter()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("shadow output contains a non-object monitor record")
        method = record.get("method")
        turn = record.get("checkpoint_turn")
        score = record.get("score")
        if (
            method not in methods
            or isinstance(turn, bool)
            or not isinstance(turn, int)
            or turn not in checkpoints
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
            or record.get("actionable_before_turn") != turn + 1
        ):
            raise ValueError("shadow output contains an undeclared method/checkpoint")
        variant: str | None = None
        if method == "frozen_probe":
            raw_variant = record.get("variant")
            if raw_variant not in variants:
                raise ValueError("shadow output contains an undeclared frozen-probe variant")
            variant = str(raw_variant)
        if record.get("source_trajectory_sha256") != source_sha:
            raise ValueError("shadow monitor record source hash changed")
        if record.get("source_prefix_sha256") != prefix_sha256_by_checkpoint[turn]:
            raise ValueError("shadow monitor record prefix hash changed")
        if record.get("passive_monitor_spec_sha256") != PASSIVE_MONITOR_SPEC_SHA256:
            raise ValueError("shadow monitor record passive spec hash changed")
        if method == "frozen_quiz":
            if record.get("quiz_generator") != dict(quiz_generator):
                raise ValueError("shadow quiz record generator changed")
        observed[(int(turn), str(method), variant)] += 1
    expected = _expected_shadow_coverage(checkpoints, variants, methods)
    if observed != expected:
        missing = sorted((str(key), count) for key, count in (expected - observed).items())
        extra = sorted((str(key), count) for key, count in (observed - expected).items())
        raise ValueError(f"shadow monitor coverage mismatch: missing={missing}, extra={extra}")
    if materialized.get("monitor_methods") != sorted(methods if checkpoints else ()):
        raise ValueError("shadow output monitor method summary changed")


async def score_clean_trajectory(
    *,
    run_id: str,
    trajectory: Mapping[str, Any],
    transport: Transport,
    event_path: str | Path,
    output_path: str | Path,
    passive_monitor_spec: Mapping[str, Any] | None = None,
    frozen_probe_variants: Sequence[str] | None = None,
    quiz_by_checkpoint: Mapping[int, Sequence[QuizQuestion]] | None = None,
    run_judge: bool | None = None,
    judge_model: str | None = None,
    judge_max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Score one clean path; no monitor message is ever appended to it."""

    if trajectory.get("complete") is not True or trajectory.get("arm") != "clean":
        raise ValueError("passive primary scoring requires a complete clean trajectory")
    source_sha = trajectory.get("transcript_sha256")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("clean trajectory lacks a transcript hash")
    model = trajectory.get("model")
    if model not in CATALOG.models or CATALOG.models[model].role != "target":
        raise ValueError("trajectory target model is outside the frozen slate")
    spec = validate_passive_monitor_spec(
        canonical_passive_monitor_spec()
        if passive_monitor_spec is None
        else passive_monitor_spec
    )
    variants = tuple(spec["frozen_probe"]["variants"])
    if variants != STATELESS_FROZEN_VARIANTS:
        raise ValueError("frozen passive spec has noncanonical stateless probe variants")
    if frozen_probe_variants is not None and tuple(frozen_probe_variants) != variants:
        raise ValueError("runtime frozen-probe variants differ from the passive spec")
    assert_passive_runtime_overrides(
        spec,
        run_judge=run_judge,
        judge_model=judge_model,
    )
    judge = spec["trace_judge"]
    if (
        judge_max_output_tokens is not None
        and judge_max_output_tokens != judge["max_output_tokens"]
    ):
        raise ValueError("runtime judge output limit differs from the passive spec")
    checkpoints = _checkpoint_turns(trajectory)
    prefix_sha256_by_checkpoint = {
        turn: sha256_json(_prefix(trajectory, turn)) for turn in checkpoints
    }
    domain = trajectory.get("domain")
    if not isinstance(domain, str):
        raise ValueError("clean trajectory lacks a domain")
    quiz_generator = quiz_generator_spec(spec, domain)

    quiz_builder = {
        "evolving_intent_gsm8k": generate_evolving_passive_quiz,
        "bfcl_multi_turn": generate_bfcl_passive_quiz,
    }.get(domain)
    if quiz_builder is None:
        raise ValueError(f"domain has no canonical passive quiz builder: {domain!r}")
    canonical_quizzes = {
        turn: quiz_builder(trajectory["task_records"], turn) for turn in checkpoints
    }
    if quiz_by_checkpoint is not None:
        if set(quiz_by_checkpoint) != set(checkpoints):
            raise ValueError("quiz checkpoints differ from the frozen trajectory schedule")
        supplied: dict[int, tuple[QuizQuestion, ...]] = {}
        for turn in checkpoints:
            questions = tuple(quiz_by_checkpoint[turn])
            if not questions or any(
                not isinstance(question, QuizQuestion) for question in questions
            ):
                raise ValueError("each frozen checkpoint requires a nonempty typed quiz")
            supplied[turn] = questions
        if supplied != canonical_quizzes:
            raise ValueError("runtime quiz questions differ from the canonical generator")
    normalized_quizzes = canonical_quizzes

    output = Path(output_path)
    if output.exists():
        existing = read_json(output)
        _validate_shadow_materialization(
            existing,
            trajectory=trajectory,
            source_sha=source_sha,
            checkpoints=checkpoints,
            spec=spec,
            quiz_generator=quiz_generator,
            prefix_sha256_by_checkpoint=prefix_sha256_by_checkpoint,
        )
        return existing
    event_file = Path(event_path)
    if event_file.exists():
        # Monitor calls are cheap relative to corrupting provenance. A future
        # resume implementation can replay completed shadow events by call key;
        # today we fail closed instead of silently rebilling or mixing specs.
        raise FileExistsError("partial shadow event log exists; audit before retrying")
    event_file.parent.mkdir(parents=True, exist_ok=True)
    task_instance_id = (
        f"{trajectory['domain']}/{trajectory['task_id']}/{trajectory['condition']}"
    )
    records: list[dict[str, Any]] = []
    horizon = len(trajectory["task_records"])
    reasoning_effort = spec["determinism"]["reasoning_effort_by_target"][model]
    temperature = spec["determinism"]["temperature"]

    def frozen_record(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(record),
            "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
        }

    for checkpoint_index, turn in enumerate(checkpoints, 1):
        prefix = _prefix(trajectory, turn)
        source_prefix_sha256 = sha256_json(prefix)
        input_tokens = _latest_input_tokens(trajectory["task_records"][turn - 1])
        context_window = CATALOG.models[model].context_window_tokens
        baseline_records = tuple(
            frozen_record(record)
            for record in (
                {
                "method": "turn_clock",
                "checkpoint_turn": turn,
                "actionable_before_turn": turn + 1,
                "score": turn / horizon,
                "fired": None,
                "source_trajectory_sha256": source_sha,
                "source_prefix_sha256": source_prefix_sha256,
                },
                {
                "method": "context_use",
                "checkpoint_turn": turn,
                "actionable_before_turn": turn + 1,
                "score": min(1.0, input_tokens / context_window),
                "raw_input_tokens": input_tokens,
                "context_window_tokens": context_window,
                "fired": None,
                "source_trajectory_sha256": source_sha,
                "source_prefix_sha256": source_prefix_sha256,
                },
            )
        )
        for record in baseline_records:
            append_jsonl(event_file, record)
            records.append(record)

        rule_flags = _observed_rule_flags(trajectory["task_records"], turn)
        rules = score_trace_rules(
            prefix,
            event_flags=rule_flags,
            fire_threshold=spec["trace_rules"]["fire_threshold"],
        )
        rule_record = frozen_record({
            "method": "trace_rules",
            "checkpoint_turn": turn,
            "actionable_before_turn": turn + 1,
            "score": rules.risk,
            "fired": rules.fired,
            "reasons": list(rules.reasons),
            "observed_event_flags": rule_flags,
            "monitor_spec_sha256": rules.spec_sha256,
            "source_trajectory_sha256": source_sha,
            "source_prefix_sha256": source_prefix_sha256,
        })
        append_jsonl(event_file, rule_record)
        records.append(rule_record)

        for variant in variants:
            instance = generate_probe_instance(variant, task_instance_id, checkpoint_index)
            fork = build_frozen_probe_fork(prefix, instance)
            result = await transport.complete(
                model,
                list(fork.messages),
                purpose="frozen_probe",
                request_key=_monitor_key(
                    run_id, source_sha, "frozen", checkpoint_index, variant
                ),
                input_token_estimate=conservative_input_token_bound(fork.messages),
                max_output_tokens=spec["frozen_probe"]["max_output_tokens"],
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            grade = grade_probe_response(instance, result.text)
            record = frozen_record({
                "method": "frozen_probe",
                "variant": variant,
                "checkpoint_turn": turn,
                "actionable_before_turn": turn + 1,
                "score": 0.0 if grade.passed else 1.0,
                "fired": not grade.passed,
                "passed": grade.passed,
                "grade_error": grade.error,
                "monitor_spec_sha256": fork.spec_sha256,
                "source_trajectory_sha256": source_sha,
                "source_prefix_sha256": source_prefix_sha256,
                "raw_output": result.text,
                "call": _call_record(result),
            })
            append_jsonl(event_file, record)
            records.append(record)

        questions = normalized_quizzes[turn]
        fork = build_quiz_fork(prefix, questions, turn)
        result = await transport.complete(
            model,
            fork,
            purpose="frozen_quiz",
            request_key=_monitor_key(run_id, source_sha, "quiz", checkpoint_index),
            input_token_estimate=conservative_input_token_bound(fork),
            max_output_tokens=spec["frozen_quiz"]["max_output_tokens"],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        grade = grade_quiz(
            questions,
            result.text,
            fire_at_wrong=spec["frozen_quiz"]["fire_at_wrong"],
        )
        record = frozen_record({
            "method": "frozen_quiz",
            "checkpoint_turn": turn,
            "actionable_before_turn": turn + 1,
            "score": grade.risk,
            "fired": grade.fired,
            "n_wrong": grade.n_wrong,
            "question_ids": [question.question_id for question in questions],
            "quiz_generator": dict(quiz_generator),
            "monitor_spec_sha256": grade.spec_sha256,
            "source_trajectory_sha256": source_sha,
            "source_prefix_sha256": source_prefix_sha256,
            "raw_output": result.text,
            "call": _call_record(result),
        })
        append_jsonl(event_file, record)
        records.append(record)

        if judge["enabled"]:
            request = build_judge_request(prefix, turn, benchmark=trajectory["domain"])
            schema = JsonSchemaOutput.from_schema("trace_risk", JUDGE_RESPONSE_SCHEMA)
            result = await transport.complete(
                judge["model"],
                request,
                purpose="trace_judge",
                request_key=_monitor_key(run_id, source_sha, "judge", checkpoint_index),
                input_token_estimate=conservative_input_token_bound(
                    request, extra_bytes=len(str(JUDGE_RESPONSE_SCHEMA).encode("utf-8"))
                ),
                max_output_tokens=judge["max_output_tokens"],
                temperature=temperature,
                reasoning_effort=judge["reasoning_effort"],
                output_schema=schema,
            )
            verdict = parse_judge_output(result.text)
            record = frozen_record({
                "method": "trace_judge",
                "checkpoint_turn": turn,
                "actionable_before_turn": turn + 1,
                "score": verdict.risk,
                "fired": None,
                "concerns": list(verdict.concerns),
                "evidence": list(verdict.evidence),
                "monitor_spec_sha256": verdict.spec_sha256,
                "source_trajectory_sha256": source_sha,
                "source_prefix_sha256": source_prefix_sha256,
                "raw_output": result.text,
                "call": _call_record(result),
            })
            append_jsonl(event_file, record)
            records.append(record)

    materialized = {
        "schema_version": 1,
        "shadow_version": SHADOW_VERSION,
        "source_trajectory_sha256": source_sha,
        "model": model,
        "domain": trajectory["domain"],
        "task_id": trajectory["task_id"],
        "condition": trajectory["condition"],
        "checkpoint_turns": list(checkpoints),
        "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
        "quiz_generator": dict(quiz_generator),
        "records": records,
        "monitor_methods": sorted({record["method"] for record in records}),
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
    atomic_write_json(output, materialized)
    return materialized
