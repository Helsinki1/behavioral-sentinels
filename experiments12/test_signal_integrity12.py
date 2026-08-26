from __future__ import annotations

from copy import deepcopy
import json
import unittest

from experiments12.core.artifacts import sha256_json
from experiments12.harness12 import ARM_TO_PROBE
from experiments12.models12 import CATALOG
from experiments12.monitors.frozen_probe import build_frozen_probe_fork
from experiments12.monitors.frozen_quiz import QuizQuestion, grade_quiz
from experiments12.monitors.judge import parse_judge_output
from experiments12.monitors.trace_rules import score_trace_rules
from experiments12.passive_quizzes12 import generate_evolving_passive_quiz
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    canonical_passive_monitor_spec,
    effective_passive_method_names,
    quiz_generator_spec,
)
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_probe_prompt,
)
from experiments12.shadow12 import SHADOW_VERSION, _prefix
from experiments12.signal_integrity12 import (
    SignalIntegrityError,
    validate_active_signal_records,
    validate_passive_signal_records,
)


MODEL = "gpt-5.6-luna"
DOMAIN = "evolving_intent_gsm8k"
TASK_ID = "integrity-task"
CONDITION = "t7"
ACTIVE = "active_recompute"


def _task_records() -> list[dict[str, object]]:
    result = []
    for turn in range(1, 4):
        result.append(
            {
                "event": "task_turn",
                "task_turn": turn,
                "user_message": {
                    "role": "user",
                    "content": f"public benchmark instruction {turn}",
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": f"public assistant answer {turn}",
                },
                "call": {
                    "usage": {
                        "input_tokens": 100 * turn,
                        "output_tokens": 10,
                    }
                },
            }
        )
    return result


def _flat_messages(records: list[dict[str, object]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for record in records:
        messages.extend(
            (
                dict(record["user_message"]),
                dict(record["assistant_message"]),
            )
        )
    return messages


def _active_trajectory() -> dict[str, object]:
    records = _task_records()
    variant = ARM_TO_PROBE[ACTIVE]
    instance_id = f"{DOMAIN}/{TASK_ID}/{CONDITION}"
    messages: list[dict[str, str]] = []
    probes: list[dict[str, object]] = []
    for turn, record in enumerate(records, 1):
        messages.extend(
            (dict(record["user_message"]), dict(record["assistant_message"]))
        )
        if turn == len(records):
            continue
        instance = generate_probe_instance(variant, instance_id, turn)
        response = instance.expected_answer
        grade = grade_probe_response(instance, response)
        user = {"role": "user", "content": render_probe_prompt(instance)}
        assistant = {"role": "assistant", "content": response}
        messages.extend((user, assistant))
        probes.append(
            {
                "event": "active_probe",
                "after_task_turn": turn,
                "checkpoint_index": turn,
                "variant": variant,
                "user_message": user,
                "assistant_message": assistant,
                "grade": {
                    "passed": grade.passed,
                    "value_correct": grade.value_correct,
                    "exact_format": grade.exact_format,
                    "error": grade.error,
                    "expected_sha256": sha256_json(instance.expected_answer),
                },
                "source_prefix_sha256": sha256_json(messages),
            }
        )
    return {
        "complete": True,
        "model": MODEL,
        "domain": DOMAIN,
        "task_id": TASK_ID,
        "condition": CONDITION,
        "arm": ACTIVE,
        "active_probe_variant": variant,
        "checkpoint_turns": [1, 2],
        "task_records": records,
        "probe_records": probes,
        "messages": messages,
        "transcript_sha256": sha256_json(messages),
    }


def _quiz_reply(questions: tuple[QuizQuestion, ...]) -> str:
    answers = []
    for index, question in enumerate(questions, 1):
        expected = question.expected
        value = ", ".join(str(item) for item in expected) if isinstance(expected, tuple) else str(expected)
        answers.append(f"A{index}: {value}")
    return "\n".join(answers)


def _passive_fixture() -> tuple[dict[str, object], dict[str, object]]:
    task_records = _task_records()
    messages = _flat_messages(task_records)
    trajectory: dict[str, object] = {
        "complete": True,
        "model": MODEL,
        "domain": DOMAIN,
        "task_id": TASK_ID,
        "condition": CONDITION,
        "arm": "clean",
        "active_probe_variant": None,
        "checkpoint_turns": [1, 2],
        "task_records": task_records,
        "probe_records": [],
        "messages": messages,
        "transcript_sha256": sha256_json(messages),
    }
    spec = canonical_passive_monitor_spec()
    source_sha = trajectory["transcript_sha256"]
    context_window = CATALOG.models[MODEL].context_window_tokens
    quiz_generator = quiz_generator_spec(spec, DOMAIN)
    instance_id = f"{DOMAIN}/{TASK_ID}/{CONDITION}"
    records: list[dict[str, object]] = []

    def frozen(value: dict[str, object]) -> dict[str, object]:
        return {**value, "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256}

    for checkpoint_index, turn in enumerate((1, 2), 1):
        prefix = _prefix(trajectory, turn)
        prefix_sha = sha256_json(prefix)
        common = {
            "checkpoint_turn": turn,
            "actionable_before_turn": turn + 1,
            "source_trajectory_sha256": source_sha,
            "source_prefix_sha256": prefix_sha,
        }
        records.append(
            frozen(
                {
                    "method": "turn_clock",
                    **common,
                    "score": turn / len(task_records),
                    "fired": None,
                }
            )
        )
        input_tokens = task_records[turn - 1]["call"]["usage"]["input_tokens"]
        records.append(
            frozen(
                {
                    "method": "context_use",
                    **common,
                    "score": min(1.0, input_tokens / context_window),
                    "raw_input_tokens": input_tokens,
                    "context_window_tokens": context_window,
                    "fired": None,
                }
            )
        )
        flags = {
            "invalid_tool_call": False,
            "execution_error": False,
            "tool_result_error": False,
            "protocol_violation": False,
        }
        rules = score_trace_rules(
            prefix,
            event_flags=flags,
            fire_threshold=spec["trace_rules"]["fire_threshold"],
        )
        records.append(
            frozen(
                {
                    "method": "trace_rules",
                    **common,
                    "score": rules.risk,
                    "fired": rules.fired,
                    "reasons": list(rules.reasons),
                    "observed_event_flags": flags,
                    "monitor_spec_sha256": rules.spec_sha256,
                }
            )
        )
        for variant in spec["frozen_probe"]["variants"]:
            instance = generate_probe_instance(variant, instance_id, checkpoint_index)
            output = instance.expected_answer
            grade = grade_probe_response(instance, output)
            fork = build_frozen_probe_fork(prefix, instance)
            records.append(
                frozen(
                    {
                        "method": "frozen_probe",
                        "variant": variant,
                        **common,
                        "score": 0.0,
                        "fired": False,
                        "passed": True,
                        "grade_error": grade.error,
                        "monitor_spec_sha256": fork.spec_sha256,
                        "raw_output": output,
                    }
                )
            )
        questions = generate_evolving_passive_quiz(task_records, turn)
        output = _quiz_reply(questions)
        grade = grade_quiz(
            questions,
            output,
            fire_at_wrong=spec["frozen_quiz"]["fire_at_wrong"],
        )
        records.append(
            frozen(
                {
                    "method": "frozen_quiz",
                    **common,
                    "score": grade.risk,
                    "fired": grade.fired,
                    "n_wrong": grade.n_wrong,
                    "question_ids": [question.question_id for question in questions],
                    "quiz_generator": quiz_generator,
                    "monitor_spec_sha256": grade.spec_sha256,
                    "raw_output": output,
                }
            )
        )
        judge_output = json.dumps(
            {"risk": 0.25, "concerns": ["uncertainty"], "evidence": ["public"]}
        )
        verdict = parse_judge_output(judge_output)
        records.append(
            frozen(
                {
                    "method": "trace_judge",
                    **common,
                    "score": verdict.risk,
                    "fired": None,
                    "concerns": list(verdict.concerns),
                    "evidence": list(verdict.evidence),
                    "monitor_spec_sha256": verdict.spec_sha256,
                    "raw_output": judge_output,
                }
            )
        )
    shadow = {
        "schema_version": 1,
        "shadow_version": SHADOW_VERSION,
        "source_trajectory_sha256": source_sha,
        "model": MODEL,
        "domain": DOMAIN,
        "task_id": TASK_ID,
        "condition": CONDITION,
        "checkpoint_turns": [1, 2],
        "passive_monitor_spec_sha256": PASSIVE_MONITOR_SPEC_SHA256,
        "quiz_generator": quiz_generator,
        "records": records,
        "monitor_methods": sorted(spec["required_methods"]),
        "complete": True,
    }
    return trajectory, shadow


class ActiveSignalIntegrityTests(unittest.TestCase):
    def test_active_records_regrade_and_reject_cached_grade_tampering(self):
        trajectory = _active_trajectory()
        self.assertEqual(len(validate_active_signal_records(trajectory, ACTIVE)), 2)
        changed = deepcopy(trajectory)
        changed["probe_records"][0]["grade"]["passed"] = False
        with self.assertRaisesRegex(SignalIntegrityError, "grade"):
            validate_active_signal_records(changed, ACTIVE)

    def test_active_prompt_and_prefix_tampering_is_rejected(self):
        trajectory = _active_trajectory()
        changed = deepcopy(trajectory)
        changed["probe_records"][0]["user_message"]["content"] += " altered"
        with self.assertRaisesRegex(SignalIntegrityError, "fields"):
            validate_active_signal_records(changed, ACTIVE)


class PassiveSignalIntegrityTests(unittest.TestCase):
    def test_every_effective_passive_method_regrades(self):
        trajectory, shadow = _passive_fixture()
        for method in effective_passive_method_names():
            with self.subTest(method=method):
                self.assertEqual(
                    len(validate_passive_signal_records(trajectory, shadow, method)),
                    2,
                )

    def test_each_monitor_score_cache_is_rejected_when_changed(self):
        trajectory, shadow = _passive_fixture()
        methods = (
            "turn_clock",
            "context_use",
            "trace_rules",
            "frozen_probe:current_copy",
            "frozen_quiz",
            "trace_judge",
        )
        for qualified in methods:
            raw_method, _, variant = qualified.partition(":")
            changed = deepcopy(shadow)
            record = next(
                row
                for row in changed["records"]
                if row["method"] == raw_method
                and (not variant or row.get("variant") == variant)
            )
            record["score"] = 1.0 if float(record["score"]) != 1.0 else 0.0
            with self.subTest(method=qualified):
                with self.assertRaisesRegex(SignalIntegrityError, "score"):
                    validate_passive_signal_records(trajectory, changed, qualified)

    def test_raw_output_tampering_and_partial_shadow_are_rejected(self):
        trajectory, shadow = _passive_fixture()
        changed = deepcopy(shadow)
        quiz = next(row for row in changed["records"] if row["method"] == "frozen_quiz")
        quiz["raw_output"] = "not canonical answers"
        with self.assertRaises(SignalIntegrityError):
            validate_passive_signal_records(trajectory, changed, "frozen_quiz")

        partial = deepcopy(shadow)
        partial["records"] = [
            row for row in partial["records"] if row["method"] != "trace_rules"
        ]
        with self.assertRaisesRegex(SignalIntegrityError, "coverage"):
            validate_passive_signal_records(trajectory, partial, "turn_clock")


if __name__ == "__main__":
    unittest.main()
