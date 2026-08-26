from __future__ import annotations

import unittest

from experiments12.monitors.base import assert_answer_blind
from experiments12.monitors.frozen_quiz import QuizQuestion, build_quiz_fork, grade_quiz
from experiments12.monitors.frozen_probe import build_frozen_probe_fork
from experiments12.monitors.judge import build_judge_request, parse_judge_output
from experiments12.monitors.trace_rules import score_trace_rules
from experiments12.probes12 import CURRENT_COPY, STATEFUL_COUNTER, generate_probe_instance


class PassiveBoundaryTests(unittest.TestCase):
    def test_forbids_gold_and_future(self):
        for key in ("gold", "label", "change_plan", "future"):
            with self.assertRaises(ValueError):
                assert_answer_blind({key: "secret"})

    def test_quiz_fork_does_not_mutate_target(self):
        messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "work"}]
        before = [dict(message) for message in messages]
        questions = [QuizQuestion("state_n", "Current count?", "integer", 4)]
        fork = build_quiz_fork(messages, questions, 2)
        self.assertEqual(messages, before)
        self.assertEqual(len(fork), len(messages) + 1)
        fork[0]["content"] = "changed"
        self.assertEqual(messages[0]["content"], "task")

    def test_frozen_probe_is_exact_but_not_carried(self):
        messages = [{"role": "assistant", "content": "task answer"}]
        instance = generate_probe_instance(CURRENT_COPY, "task-1", 1)
        fork = build_frozen_probe_fork(messages, instance)
        self.assertEqual(len(fork.messages), 2)
        self.assertIn("say my name", fork.messages[-1]["content"])
        self.assertEqual(messages, [{"role": "assistant", "content": "task answer"}])
        with self.assertRaises(ValueError):
            build_frozen_probe_fork(
                messages,
                generate_probe_instance(STATEFUL_COUNTER, "task-1", 1),
            )


class QuizTests(unittest.TestCase):
    def test_deterministic_grading_kinds(self):
        questions = [
            QuizQuestion("q1", "Count?", "integer", 7),
            QuizQuestion("q2", "Names?", "set", ("alice", "bob")),
            QuizQuestion("q3", "Order?", "ordered", ("x", "y")),
            QuizQuestion("q4", "Phrase?", "exact", "still working"),
        ]
        result = grade_quiz(
            questions,
            "A1: seven (7)\nA2: Bob, Alice\nA3: x y\nA4: still working",
        )
        self.assertEqual(result.correct, (True, True, True, True))
        self.assertEqual(result.risk, 0)
        self.assertFalse(result.fired)

    def test_missing_answer_fires(self):
        questions = [QuizQuestion("q1", "Count?", "integer", 7)]
        result = grade_quiz(questions, "I think seven")
        self.assertEqual(result.n_wrong, 1)
        self.assertTrue(result.fired)


class JudgeTests(unittest.TestCase):
    def test_request_contains_only_prefix(self):
        request = build_judge_request(
            [{"role": "user", "content": "visible"}],
            1,
            benchmark="bfcl_multi_turn",
        )
        self.assertIn("visible", request[-1]["content"])
        self.assertNotIn("ground_truth", request[-1]["content"])

    def test_strict_parse(self):
        verdict = parse_judge_output(
            '{"risk":0.7,"concerns":["stale state"],"evidence":["used old path"]}'
        )
        self.assertEqual(verdict.risk, 0.7)
        with self.assertRaises(ValueError):
            parse_judge_output('{"risk":0.7,"concerns":[],"evidence":[],"answer":"x"}')


class RuleTests(unittest.TestCase):
    def test_answer_blind_structural_flags(self):
        result = score_trace_rules(
            [{"role": "assistant", "content": "working"}],
            event_flags={"invalid_tool_call": True},
        )
        self.assertTrue(result.fired)
        self.assertIn("invalid_tool_call", result.reasons)
        with self.assertRaises(ValueError):
            score_trace_rules([], event_flags={"correct": False})

    def test_repeat_signal(self):
        text = "I will repeat this sufficiently long assistant response now"
        result = score_trace_rules(
            [{"role": "assistant", "content": text}, {"role": "assistant", "content": text}]
        )
        self.assertTrue(result.fired)
        self.assertIn("exact_repeated_assistant_output", result.reasons)


if __name__ == "__main__":
    unittest.main()
