"""No-dependency tests for Experiment 12's four carried active probes."""

from __future__ import annotations

import unittest

from experiments12.probes12 import (
    CURRENT_COPY,
    FIXED_OUTPUT_LENGTH,
    INITIAL_RECALL,
    PROBE_DEFINITIONS,
    PROBE_ORDER,
    RECOMPUTE,
    STATEFUL_COUNTER,
    append_carried_probe_exchange,
    component_metadata,
    extract_probe_answer,
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)


class DesignTests(unittest.TestCase):
    def test_exactly_four_variants_and_paper_terms(self) -> None:
        self.assertEqual(tuple(PROBE_DEFINITIONS), PROBE_ORDER)
        self.assertEqual(len(PROBE_DEFINITIONS), 4)
        terms = [PROBE_DEFINITIONS[key].paper_term for key in PROBE_ORDER]
        self.assertEqual(
            terms,
            ["say my name", "say my name", "increment this number", "recompute"],
        )

    def test_component_metadata_is_explicit_and_carried(self) -> None:
        metadata = component_metadata()
        self.assertEqual(set(metadata), set(PROBE_ORDER))
        for row in metadata.values():
            self.assertEqual(row["output_length"], FIXED_OUTPUT_LENGTH)
            self.assertTrue(row["carried"])
            self.assertIn("copyability", row)
            self.assertIn("memory_load", row)
            self.assertIn("reasoning_load", row)
        self.assertEqual(metadata[CURRENT_COPY]["copyability"], 2)
        self.assertEqual(metadata[INITIAL_RECALL]["memory_load"], 2)
        self.assertEqual(metadata[RECOMPUTE]["reasoning_load"], 2)


class GenerationTests(unittest.TestCase):
    def test_every_expected_output_has_identical_exact_length(self) -> None:
        answers = []
        for task_id in ("turnbench/17", "gsm8k/42", "bfcl/alpha"):
            for checkpoint in (1, 2, 7, 31):
                for variant in PROBE_ORDER:
                    instance = generate_probe_instance(variant, task_id, checkpoint)
                    answers.append(instance.expected_answer)
                    self.assertEqual(len(instance.expected_answer), FIXED_OUTPUT_LENGTH)
                    self.assertRegex(instance.expected_answer, r"^PROBE: [A-Z0-9]{8}$")
        self.assertEqual({len(answer) for answer in answers}, {FIXED_OUTPUT_LENGTH})

    def test_generation_is_order_independent_and_deterministic(self) -> None:
        forward = {
            (variant, checkpoint): generate_probe_instance(
                variant, "determinism-task", checkpoint
            )
            for variant in PROBE_ORDER
            for checkpoint in range(1, 6)
        }
        reverse = {
            (variant, checkpoint): generate_probe_instance(
                variant, "determinism-task", checkpoint
            )
            for variant in reversed(PROBE_ORDER)
            for checkpoint in reversed(range(1, 6))
        }
        self.assertEqual(forward, reverse)

    def test_variant_semantics(self) -> None:
        copy_one = generate_probe_instance(CURRENT_COPY, "task-a", 1)
        copy_two = generate_probe_instance(CURRENT_COPY, "task-a", 2)
        self.assertNotEqual(copy_one.expected_answer, copy_two.expected_answer)
        self.assertIn(copy_one.payload, render_probe_prompt(copy_one))

        recall_one = generate_probe_instance(INITIAL_RECALL, "task-a", 1)
        recall_late = generate_probe_instance(INITIAL_RECALL, "task-a", 99)
        self.assertNotEqual(recall_one.expected_answer, recall_late.expected_answer)
        recall_setup = render_initial_instruction(INITIAL_RECALL, "task-a", (1, 99))
        self.assertIsNotNone(recall_setup)
        self.assertIn(recall_one.payload, recall_setup or "")
        self.assertIn(recall_late.payload, recall_setup or "")
        self.assertNotIn(recall_one.payload, render_probe_prompt(recall_one))

        counter_one = generate_probe_instance(STATEFUL_COUNTER, "task-a", 1)
        counter_two = generate_probe_instance(STATEFUL_COUNTER, "task-a", 2)
        self.assertEqual(int(counter_two.payload), int(counter_one.payload) + 1)
        self.assertIsNotNone(render_initial_instruction(STATEFUL_COUNTER, "task-a"))

        recompute = generate_probe_instance(RECOMPUTE, "task-a", 3)
        parameters = recompute.parameter_map
        self.assertEqual(
            int(recompute.payload),
            parameters["left"] * parameters["right"] + parameters["addend"],
        )
        self.assertIn("recompute", render_probe_prompt(recompute).lower())


class GradingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = generate_probe_instance(RECOMPUTE, "grade-task", 4)

    def test_exact_response_passes(self) -> None:
        grade = grade_probe_response(self.instance, self.instance.expected_answer + "\n")
        self.assertTrue(grade.passed)
        self.assertTrue(grade.value_correct)
        self.assertTrue(grade.exact_format)
        self.assertIsNone(grade.error)

    def test_wrong_extra_and_ambiguous_responses_fail_deterministically(self) -> None:
        wrong = grade_probe_response(self.instance, "PROBE: 00000000")
        self.assertFalse(wrong.passed)
        self.assertEqual(wrong.error, "wrong_value")

        extra = grade_probe_response(
            self.instance, "Here it is:\n" + self.instance.expected_answer
        )
        self.assertFalse(extra.passed)
        self.assertTrue(extra.value_correct)
        self.assertEqual(extra.error, "extra_or_noncanonical_output")

        padded = grade_probe_response(self.instance, " " + self.instance.expected_answer)
        self.assertFalse(padded.passed)
        self.assertEqual(padded.error, "extra_or_noncanonical_output")

        ambiguous_text = (
            self.instance.expected_answer + "\n" + self.instance.expected_answer
        )
        ambiguous = grade_probe_response(self.instance, ambiguous_text)
        self.assertFalse(ambiguous.passed)
        self.assertEqual(ambiguous.error, "missing_or_ambiguous_answer")
        self.assertIsNone(extract_probe_answer(ambiguous_text))

    def test_carried_exchange_appends_prompt_and_actual_answer(self) -> None:
        history = [{"role": "user", "content": "main task turn"}]
        actual = self.instance.expected_answer
        grade = append_carried_probe_exchange(history, self.instance, actual)
        self.assertTrue(grade.passed)
        self.assertEqual([message["role"] for message in history[-2:]], ["user", "assistant"])
        self.assertEqual(history[-1]["content"], actual)
        self.assertIn("ACTIVE CARRIED PROBE", history[-2]["content"])
        self.assertIn("exactly 15 characters", history[-2]["content"])


class ValidationTests(unittest.TestCase):
    def test_bad_inputs_are_rejected(self) -> None:
        with self.assertRaises(KeyError):
            generate_probe_instance("unknown", "task", 1)
        with self.assertRaises(ValueError):
            generate_probe_instance(CURRENT_COPY, "", 1)
        with self.assertRaises(ValueError):
            generate_probe_instance(CURRENT_COPY, "task", 0)
        with self.assertRaises(ValueError):
            render_initial_instruction(INITIAL_RECALL, "task")
        with self.assertRaises(ValueError):
            render_initial_instruction(INITIAL_RECALL, "task", (1, 1))
        with self.assertRaises(TypeError):
            extract_probe_answer(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
