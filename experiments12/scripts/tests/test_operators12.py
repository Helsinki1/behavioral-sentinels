"""Synthetic tests for pure Experiment 12 intervention transitions."""

from __future__ import annotations

import copy
import inspect
import json
import unittest

from experiments12.core.artifacts import sha256_json
from experiments12.operators12 import (
    CheckpointSchedule,
    CompactionConfig,
    ContaminationError,
    FeedbackNote,
    InterventionType,
    InterventionValidationError,
    ScheduleMismatchError,
    ScheduleMode,
    ScheduledMember,
    SignalReference,
    apply_intervention,
    build_lossy_compaction,
    build_matched_schedule,
    build_yoked_schedule,
    conservative_token_upper_bound,
    freeze_initial_instructions,
    freeze_public_state,
    freeze_visible_prefix,
    make_feedback_note,
)
from experiments12.spec12 import FEEDBACK_MAX_TOKENS


DOMAIN = "synthetic_domain"
TASK_ID = "task-7"


class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_messages = [
            {"role": "system", "content": "Follow the public task instructions."}
        ]
        self.history = [
            *self.initial_messages,
            {"role": "user", "content": "Make a plan for Ada."},
            {"role": "assistant", "content": "The plan is ready."},
            {"role": "user", "content": "Check the wrong entry and retry."},
            {"role": "assistant", "content": "I found the wrong entry; retry now."},
        ]
        self.prefix = freeze_visible_prefix(
            domain=DOMAIN,
            task_id=TASK_ID,
            after_turn=2,
            messages=self.history,
        )
        self.instructions = freeze_initial_instructions(
            domain=DOMAIN,
            task_id=TASK_ID,
            messages=self.initial_messages,
        )
        self.schedule = build_yoked_schedule(
            group_id="deployment-pair-7",
            eligible_by_member={"cell-a": (1, 2, 3), "cell-b": (1, 2, 3)},
            anchor_member_id="cell-a",
            anchor_action_checkpoints=(2,),
            seed=19,
        )

    def apply(self, kind: InterventionType | str, **kwargs: object):
        return apply_intervention(
            intervention_type=kind,
            prefix=self.prefix,
            schedule=self.schedule,
            member_id="cell-b",
            checkpoint=2,
            **kwargs,
        )

    def test_none_returns_a_fresh_exact_history_and_hashed_provenance(self) -> None:
        before = copy.deepcopy(self.history)
        result = self.apply(InterventionType.NONE)
        self.assertEqual(result.continued_history, before)
        self.assertEqual(self.history, before)
        self.assertIsNot(result.continued_history, self.history)
        mutable_copy = result.continued_history
        mutable_copy[0]["content"] = "mutated outside the record"
        self.assertEqual(result.continued_history[0]["content"], before[0]["content"])
        event = result.as_event()
        self.assertEqual(event["intervention_type"], "none")
        self.assertEqual(event["schedule_sha256"], self.schedule.schedule_sha256)
        provenance = event.pop("provenance_sha256")
        self.assertEqual(provenance, sha256_json(event))
        self.assertEqual(result.dropped_message_count, 0)

    def test_compaction_is_deterministic_lossy_and_uses_only_the_prefix(self) -> None:
        config = CompactionConfig(
            keep_last_messages=2,
            max_excerpt_bytes=70,
            max_summary_bytes=190,
        )
        summary = build_lossy_compaction(self.prefix, config)
        self.assertIn("wrong entry", summary)
        self.assertNotIn("Make a plan for Ada", summary)
        first = self.apply(
            "compact",
            instructions=self.instructions,
            compaction_config=config,
            signal=SignalReference(
                method="trace_rules",
                checkpoint=2,
                source_prefix_sha256=self.prefix.prefix_sha256,
                signal_record_sha256="a" * 64,
            ),
        )
        second = self.apply(
            InterventionType.COMPACT,
            instructions=self.instructions,
            compaction_config=config,
            signal=SignalReference(
                method="trace_rules",
                checkpoint=2,
                source_prefix_sha256=self.prefix.prefix_sha256,
                signal_record_sha256="a" * 64,
            ),
        )
        self.assertEqual(first, second)
        self.assertEqual(first.continued_history[:1], self.initial_messages)
        self.assertEqual(len(first.continued_history), 2)
        self.assertIn(summary, first.continued_history[-1]["content"])
        self.assertEqual(first.dropped_message_count, len(self.history) - 1)
        self.assertEqual(self.history[-1]["content"], "I found the wrong entry; retry now.")

    def test_reground_uses_exact_public_state_and_rejects_contamination(self) -> None:
        state = freeze_public_state(
            domain=DOMAIN,
            task_id=TASK_ID,
            after_turn=2,
            state={"revision": 4, "contacts": {"Ada": {"status": "active"}}},
        )
        result = self.apply(
            InterventionType.REGROUND,
            instructions=self.instructions,
            public_state=state,
        )
        continued = result.continued_history
        self.assertEqual(continued[:1], self.initial_messages)
        self.assertEqual(len(continued), 2)
        self.assertIn(state.state_json, continued[-1]["content"])
        self.assertNotIn("wrong entry", json.dumps(continued))
        self.assertEqual(result.dropped_message_count, len(self.history) - 1)

        contaminated_states = (
            {"private_state": {"x": 1}},
            {"nested": {"goldAnswer": "x"}},
            {"items": [{"future_turns": ["later"]}]},
            {"hiddenState": "secret"},
            {"evaluationResult": {"official_score": 1}},
        )
        for contaminated in contaminated_states:
            with self.subTest(contaminated=contaminated):
                with self.assertRaises(ContaminationError):
                    freeze_public_state(
                        domain=DOMAIN,
                        task_id=TASK_ID,
                        after_turn=2,
                        state=contaminated,
                    )

    def test_feedback_is_exactly_quoted_keeps_trace_and_stays_under_80(self) -> None:
        note = make_feedback_note(
            self.prefix,
            good=("plan",),
            bad=("wrong",),
            watch=("retry",),
        )
        self.assertLessEqual(
            conservative_token_upper_bound(note.render()), FEEDBACK_MAX_TOKENS
        )
        result = self.apply("feedback", feedback=note)
        self.assertEqual(result.continued_history[:-1], self.history)
        self.assertEqual(result.continued_history[-1]["content"], note.render())
        self.assertIn('GOOD: "plan"', note.render())
        self.assertIn('BAD: "wrong"', note.render())
        self.assertIn('WATCH: "retry"', note.render())
        self.assertEqual(result.dropped_message_count, 0)

        invented = FeedbackNote(self.prefix.prefix_sha256, bad=("never appeared",))
        with self.assertRaisesRegex(ContaminationError, "absent"):
            self.apply("feedback", feedback=invented)
        with self.assertRaisesRegex(InterventionValidationError, "80"):
            FeedbackNote(self.prefix.prefix_sha256, good=("x" * 60,))

    def test_matched_and_yoked_schedules_are_deterministic_and_score_free(self) -> None:
        eligible_forward = {"cell-b": (1, 2, 3, 4, 5), "cell-a": (1, 2, 3, 4, 5)}
        eligible_reverse = {"cell-a": (1, 2, 3, 4, 5), "cell-b": (1, 2, 3, 4, 5)}
        first = build_matched_schedule(
            group_id="matched-1",
            eligible_by_member=eligible_forward,
            intervention_count=2,
            seed=42,
        )
        second = build_matched_schedule(
            group_id="matched-1",
            eligible_by_member=eligible_reverse,
            intervention_count=2,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.schedule_sha256, second.schedule_sha256)
        self.assertEqual(first.members[0].action_checkpoints, first.members[1].action_checkpoints)
        self.assertNotIn("score", inspect.signature(build_matched_schedule).parameters)
        self.assertNotIn("score", inspect.signature(build_yoked_schedule).parameters)

        yoked = build_yoked_schedule(
            group_id="yoked-1",
            eligible_by_member={"anchor": (1, 2, 4), "target": (1, 2, 3, 4)},
            anchor_member_id="anchor",
            anchor_action_checkpoints=(2, 4),
        )
        self.assertEqual(yoked.action_checkpoints, (2, 4))
        self.assertTrue(all(member.action_checkpoints == (2, 4) for member in yoked.members))

        with self.assertRaisesRegex(ScheduleMismatchError, "different eligible"):
            build_matched_schedule(
                group_id="bad-match",
                eligible_by_member={"a": (1, 2), "b": (1, 3)},
                intervention_count=1,
                seed=1,
            )
        with self.assertRaisesRegex(ScheduleMismatchError, "cannot receive"):
            build_yoked_schedule(
                group_id="bad-yoke",
                eligible_by_member={"a": (1, 2), "b": (1,)},
                anchor_member_id="a",
                anchor_action_checkpoints=(2,),
            )
        with self.assertRaises(ScheduleMismatchError):
            CheckpointSchedule(
                group_id="manual-bad",
                mode=ScheduleMode.MATCHED,
                members=(
                    ScheduledMember("a", (1, 2), (1,)),
                    ScheduledMember("b", (1, 2), (2,)),
                ),
                seed=0,
            )

    def test_application_fails_closed_on_type_payload_prefix_and_schedule_mismatch(self) -> None:
        with self.assertRaisesRegex(InterventionValidationError, "exactly"):
            self.apply("lossy_compaction", instructions=self.instructions)
        state = freeze_public_state(
            domain=DOMAIN,
            task_id=TASK_ID,
            after_turn=2,
            state={"revision": 1},
        )
        with self.assertRaisesRegex(InterventionValidationError, "cannot carry"):
            self.apply("none", public_state=state)
        wrong_signal = SignalReference(
            method="judge",
            checkpoint=2,
            source_prefix_sha256="b" * 64,
            signal_record_sha256="c" * 64,
        )
        with self.assertRaisesRegex(ContaminationError, "signal provenance"):
            self.apply("none", signal=wrong_signal)
        frozen_signal = SignalReference(
            method="trace_rules",
            checkpoint=2,
            source_prefix_sha256="d" * 64,
            signal_record_sha256="e" * 64,
            schedule_sha256=self.schedule.schedule_sha256,
            frozen_two_pass=True,
        )
        frozen_result = self.apply("none", signal=frozen_signal)
        frozen_event = frozen_result.as_event()
        self.assertTrue(frozen_event["signal_frozen_two_pass"])
        self.assertEqual(frozen_event["signal_source_prefix_sha256"], "d" * 64)
        self.assertEqual(frozen_event["signal_record_sha256"], "e" * 64)
        with self.assertRaisesRegex(ContaminationError, "deployment schedule"):
            self.apply(
                "none",
                signal=SignalReference(
                    method="trace_rules",
                    checkpoint=2,
                    source_prefix_sha256="d" * 64,
                    signal_record_sha256="e" * 64,
                    schedule_sha256="f" * 64,
                    frozen_two_pass=True,
                ),
            )
        with self.assertRaises(ScheduleMismatchError):
            apply_intervention(
                intervention_type="none",
                prefix=self.prefix,
                schedule=self.schedule,
                member_id="unmatched-cell",
                checkpoint=2,
            )
        wrong_checkpoint_prefix = freeze_visible_prefix(
            domain=DOMAIN,
            task_id=TASK_ID,
            after_turn=1,
            messages=self.history[:3],
        )
        with self.assertRaises(ScheduleMismatchError):
            apply_intervention(
                intervention_type="none",
                prefix=wrong_checkpoint_prefix,
                schedule=self.schedule,
                member_id="cell-b",
                checkpoint=1,
            )
        with self.assertRaisesRegex(ContaminationError, "checkpoint"):
            apply_intervention(
                intervention_type="none",
                prefix=wrong_checkpoint_prefix,
                schedule=self.schedule,
                member_id="cell-b",
                checkpoint=2,
            )

        contaminated_message = [
            {"role": "user", "content": "visible", "future_turns": ["hidden"]}
        ]
        with self.assertRaises(ContaminationError):
            freeze_visible_prefix(
                domain=DOMAIN,
                task_id=TASK_ID,
                after_turn=1,
                messages=contaminated_message,
            )
        with self.assertRaisesRegex(ContaminationError, "initial user"):
            freeze_initial_instructions(
                domain=DOMAIN,
                task_id=TASK_ID,
                messages=(
                    {"role": "user", "content": "initial task"},
                    {"role": "user", "content": "future task"},
                ),
            )


if __name__ == "__main__":
    unittest.main()
