"""Provider-free regrading of carried and zero-carry observation records.

Recorded ``score`` and ``grade`` fields are convenient caches, not scientific
ground truth.  These helpers reconstruct every observation from the exact
target-visible prefix and the frozen deterministic generators before a score
may be used to schedule a deployment intervention.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import sha256_json
from experiments12.harness12 import ARM_TO_PROBE
from experiments12.models12 import CATALOG
from experiments12.monitors.frozen_probe import build_frozen_probe_fork
from experiments12.monitors.frozen_quiz import QuizQuestion, grade_quiz
from experiments12.monitors.judge import parse_judge_output
from experiments12.monitors.trace_rules import score_trace_rules
from experiments12.passive_quizzes12 import (
    generate_bfcl_passive_quiz,
    generate_evolving_passive_quiz,
)
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    canonical_passive_monitor_spec,
    effective_passive_method_names,
    quiz_generator_spec,
    validate_passive_monitor_spec,
)
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.shadow12 import (
    _checkpoint_turns,
    _latest_input_tokens,
    _observed_rule_flags,
    _prefix,
    _validate_shadow_materialization,
)


class SignalIntegrityError(ValueError):
    """A stored signal does not reproduce from its frozen public inputs."""


def _records(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise SignalIntegrityError(f"{label} must be a nonempty object sequence")
    return tuple(value)


def _messages(record: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    native = record.get("messages")
    raw: Any = (
        native
        if native is not None
        else (record.get("user_message"), record.get("assistant_message"))
    )
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or not raw
        or any(not isinstance(message, Mapping) for message in raw)
    ):
        raise SignalIntegrityError("task record has an invalid message sequence")
    return tuple(dict(message) for message in raw)


def _task_instance_id(trajectory: Mapping[str, Any]) -> str:
    parts = tuple(trajectory.get(key) for key in ("domain", "task_id", "condition"))
    if any(not isinstance(part, str) or not part for part in parts):
        raise SignalIntegrityError("trajectory lacks its canonical task identity")
    return "/".join(parts)


def _number_equal(actual: Any, expected: float, label: str) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or float(actual) != float(expected)
    ):
        raise SignalIntegrityError(f"{label} does not reproduce")


def _require_fields(
    record: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    changed = sorted(key for key, value in expected.items() if record.get(key) != value)
    if changed:
        raise SignalIntegrityError(
            f"{label} does not reproduce fields: {', '.join(changed)}"
        )


def validate_active_signal_records(
    trajectory: Mapping[str, Any], method: str
) -> tuple[Mapping[str, Any], ...]:
    """Reconstruct and regrade one complete carried-probe trajectory.

    The returned records are the original immutable mappings, but only after
    their checkpoint schedule, prompts, responses, grade cache, and carried
    prefix hashes reproduce exactly.
    """

    try:
        variant = ARM_TO_PROBE[method]
    except (KeyError, TypeError) as exc:
        raise SignalIntegrityError("active method is unknown") from exc
    if (
        not isinstance(trajectory, Mapping)
        or trajectory.get("complete") is not True
        or trajectory.get("arm") != method
        or trajectory.get("active_probe_variant") != variant
    ):
        raise SignalIntegrityError("active trajectory identity changed")
    task_records = _records(trajectory.get("task_records"), "active task records")
    if len(task_records) < 2:
        raise SignalIntegrityError("active trajectory needs at least two task turns")
    checkpoints = tuple(range(1, len(task_records)))
    if trajectory.get("checkpoint_turns") != list(checkpoints):
        raise SignalIntegrityError("active checkpoint schedule is not canonical")
    probe_records = _records(trajectory.get("probe_records"), "active probe records")
    if len(probe_records) != len(checkpoints):
        raise SignalIntegrityError("active probe coverage is incomplete")

    task_instance_id = _task_instance_id(trajectory)
    setup = render_initial_instruction(
        variant, task_instance_id, tuple(range(1, len(checkpoints) + 1))
    )
    if setup is not None:
        first = _messages(task_records[0])[0]
        content = first.get("content")
        delimiter = (
            "\n\n--- BFCL USER MESSAGE ---\n"
            if trajectory.get("domain") == "bfcl_multi_turn"
            else "\n\n--- BENCHMARK MESSAGE ---\n"
        )
        if not isinstance(content, str) or not content.startswith(setup + delimiter):
            raise SignalIntegrityError("active initial carried instruction changed")

    timeline: list[dict[str, Any]] = []
    for turn, task_record in enumerate(task_records, 1):
        if task_record.get("event") != "task_turn" or task_record.get("task_turn") != turn:
            raise SignalIntegrityError("active task turns are not contiguous")
        timeline.extend(_messages(task_record))
        if turn == len(task_records):
            continue
        record = probe_records[turn - 1]
        instance = generate_probe_instance(variant, task_instance_id, turn)
        assistant = record.get("assistant_message")
        response = assistant.get("content") if isinstance(assistant, Mapping) else None
        if not isinstance(response, str):
            raise SignalIntegrityError("active probe response is not text")
        grade = grade_probe_response(instance, response)
        expected_grade = {
            "passed": grade.passed,
            "value_correct": grade.value_correct,
            "exact_format": grade.exact_format,
            "error": grade.error,
            "expected_sha256": sha256_json(instance.expected_answer),
        }
        _require_fields(
            record,
            {
                "event": "active_probe",
                "after_task_turn": turn,
                "checkpoint_index": turn,
                "variant": variant,
                "user_message": {
                    "role": "user",
                    "content": render_probe_prompt(instance),
                },
                "grade": expected_grade,
            },
            f"active probe checkpoint {turn}",
        )
        probe_messages = (
            dict(record["user_message"]),
            dict(assistant),
        )
        timeline.extend(probe_messages)
        if record.get("source_prefix_sha256") != sha256_json(timeline):
            raise SignalIntegrityError(
                f"active probe checkpoint {turn} carried-prefix hash changed"
            )

    messages = trajectory.get("messages")
    if messages != timeline or trajectory.get("transcript_sha256") != sha256_json(timeline):
        raise SignalIntegrityError("active transcript does not reproduce")
    return probe_records


def _canonical_quizzes(
    trajectory: Mapping[str, Any], checkpoints: Sequence[int]
) -> dict[int, tuple[QuizQuestion, ...]]:
    builder = {
        "evolving_intent_gsm8k": generate_evolving_passive_quiz,
        "bfcl_multi_turn": generate_bfcl_passive_quiz,
    }.get(trajectory.get("domain"))
    if builder is None:
        raise SignalIntegrityError("trajectory domain has no frozen passive quiz")
    return {
        turn: builder(trajectory["task_records"], turn) for turn in checkpoints
    }


def _validate_passive_record(
    *,
    trajectory: Mapping[str, Any],
    record: Mapping[str, Any],
    checkpoint_index: int,
    quizzes: Mapping[int, tuple[QuizQuestion, ...]],
    spec: Mapping[str, Any],
) -> str:
    method = record.get("method")
    turn = record.get("checkpoint_turn")
    if isinstance(turn, bool) or not isinstance(turn, int):
        raise SignalIntegrityError("passive checkpoint is invalid")
    if method != "frozen_probe" and record.get("variant") is not None:
        raise SignalIntegrityError("only frozen-probe records may declare a variant")
    prefix = _prefix(trajectory, turn)
    horizon = len(trajectory["task_records"])

    if method == "turn_clock":
        _number_equal(record.get("score"), turn / horizon, "turn-clock score")
        _require_fields(record, {"fired": None}, "turn-clock record")
        return "turn_clock"

    if method == "context_use":
        input_tokens = _latest_input_tokens(trajectory["task_records"][turn - 1])
        model = trajectory.get("model")
        if model not in CATALOG.models or CATALOG.models[model].role != "target":
            raise SignalIntegrityError("passive trajectory model is not a frozen target")
        context_window = CATALOG.models[model].context_window_tokens
        _number_equal(
            record.get("score"),
            min(1.0, input_tokens / context_window),
            "context-use score",
        )
        _require_fields(
            record,
            {
                "raw_input_tokens": input_tokens,
                "context_window_tokens": context_window,
                "fired": None,
            },
            "context-use record",
        )
        return "context_use"

    if method == "trace_rules":
        flags = _observed_rule_flags(trajectory["task_records"], turn)
        result = score_trace_rules(
            prefix,
            event_flags=flags,
            fire_threshold=spec["trace_rules"]["fire_threshold"],
        )
        _number_equal(record.get("score"), result.risk, "trace-rules score")
        _require_fields(
            record,
            {
                "fired": result.fired,
                "reasons": list(result.reasons),
                "observed_event_flags": flags,
                "monitor_spec_sha256": result.spec_sha256,
            },
            "trace-rules record",
        )
        return "trace_rules"

    task_instance_id = _task_instance_id(trajectory)
    raw_output = record.get("raw_output")
    if not isinstance(raw_output, str):
        raise SignalIntegrityError(f"{method} record lacks its raw model output")

    if method == "frozen_probe":
        variant = record.get("variant")
        if variant not in spec["frozen_probe"]["variants"]:
            raise SignalIntegrityError("frozen-probe variant changed")
        instance = generate_probe_instance(variant, task_instance_id, checkpoint_index)
        fork = build_frozen_probe_fork(prefix, instance)
        grade = grade_probe_response(instance, raw_output)
        _number_equal(
            record.get("score"), 0.0 if grade.passed else 1.0, "frozen-probe score"
        )
        _require_fields(
            record,
            {
                "fired": not grade.passed,
                "passed": grade.passed,
                "grade_error": grade.error,
                "monitor_spec_sha256": fork.spec_sha256,
            },
            "frozen-probe record",
        )
        return f"frozen_probe:{variant}"

    if method == "frozen_quiz":
        questions = quizzes[turn]
        grade = grade_quiz(
            questions,
            raw_output,
            fire_at_wrong=spec["frozen_quiz"]["fire_at_wrong"],
        )
        _number_equal(record.get("score"), grade.risk, "frozen-quiz score")
        _require_fields(
            record,
            {
                "fired": grade.fired,
                "n_wrong": grade.n_wrong,
                "question_ids": [question.question_id for question in questions],
                "monitor_spec_sha256": grade.spec_sha256,
            },
            "frozen-quiz record",
        )
        return "frozen_quiz"

    if method == "trace_judge":
        verdict = parse_judge_output(raw_output)
        _number_equal(record.get("score"), verdict.risk, "trace-judge score")
        _require_fields(
            record,
            {
                "fired": None,
                "concerns": list(verdict.concerns),
                "evidence": list(verdict.evidence),
                "monitor_spec_sha256": verdict.spec_sha256,
            },
            "trace-judge record",
        )
        return "trace_judge"

    raise SignalIntegrityError(f"unknown passive method: {method!r}")


def validate_passive_signal_records(
    trajectory: Mapping[str, Any],
    shadow: Mapping[str, Any],
    method: str,
    *,
    passive_monitor_spec: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate a complete canonical shadow and return one effective method.

    Full-shadow coverage is checked even when only one method is requested, so
    a partial hand-built shadow cannot enter a deployment preparation path.
    Every stored score is then regenerated from its prefix and raw output.
    """

    if method not in effective_passive_method_names():
        raise SignalIntegrityError(f"unknown effective passive method: {method!r}")
    if (
        not isinstance(trajectory, Mapping)
        or trajectory.get("complete") is not True
        or trajectory.get("arm") != "clean"
    ):
        raise SignalIntegrityError("passive source must be a complete clean trajectory")
    if not isinstance(shadow, Mapping):
        raise SignalIntegrityError("passive shadow must be an object")
    spec = validate_passive_monitor_spec(
        canonical_passive_monitor_spec()
        if passive_monitor_spec is None
        else passive_monitor_spec
    )
    try:
        checkpoints = _checkpoint_turns(trajectory)
        source_sha = trajectory.get("transcript_sha256")
        complete_messages = _prefix(trajectory, len(trajectory["task_records"]))
        if source_sha != sha256_json(complete_messages) or trajectory.get(
            "messages"
        ) != complete_messages:
            raise SignalIntegrityError("clean source transcript does not reproduce")
        prefix_hashes = {
            turn: sha256_json(_prefix(trajectory, turn)) for turn in checkpoints
        }
        quiz_generator = quiz_generator_spec(spec, str(trajectory.get("domain")))
        quizzes = _canonical_quizzes(trajectory, checkpoints)
        _validate_shadow_materialization(
            shadow,
            trajectory=trajectory,
            source_sha=str(source_sha),
            checkpoints=checkpoints,
            spec=spec,
            quiz_generator=quiz_generator,
            prefix_sha256_by_checkpoint=prefix_hashes,
        )
    except SignalIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SignalIntegrityError(f"passive shadow materialization changed: {exc}") from exc

    selected: list[Mapping[str, Any]] = []
    checkpoint_ordinals = {turn: index for index, turn in enumerate(checkpoints, 1)}
    try:
        for record in shadow["records"]:
            qualified = _validate_passive_record(
                trajectory=trajectory,
                record=record,
                checkpoint_index=checkpoint_ordinals[int(record["checkpoint_turn"])],
                quizzes=quizzes,
                spec=spec,
            )
            if qualified == method:
                selected.append(record)
    except SignalIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SignalIntegrityError(f"passive record does not reproduce: {exc}") from exc
    if tuple(record.get("checkpoint_turn") for record in selected) != checkpoints:
        raise SignalIntegrityError(f"passive method {method} checkpoint coverage changed")
    if any(
        record.get("passive_monitor_spec_sha256") != PASSIVE_MONITOR_SPEC_SHA256
        for record in selected
    ):
        raise SignalIntegrityError("passive method spec hash changed")
    return tuple(selected)


__all__ = [
    "SignalIntegrityError",
    "validate_active_signal_records",
    "validate_passive_signal_records",
]
