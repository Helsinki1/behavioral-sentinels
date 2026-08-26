from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from experiments12.core.schemas import CallAttemptRecord, CallStatus, TokenUsage
from experiments12.core.transport import CompletionResult
from experiments12.core.artifacts import atomic_write_json, read_json
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    canonical_passive_monitor_spec,
)
from experiments12.shadow12 import (
    _prefix,
    generate_evolving_passive_quiz,
    score_clean_trajectory,
)


class FakeTransport:
    def __init__(self):
        self.calls = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append((model, [dict(m) for m in messages], kwargs))
        purpose = kwargs["purpose"]
        text = (
            '{"risk":0.25,"concerns":[],"evidence":[]}'
            if purpose == "trace_judge"
            else "PROBE: AAAAAAAA"
        )
        attempt = CallAttemptRecord(
            event_id=f"e{len(self.calls)}",
            reservation_id=f"r{len(self.calls)}",
            provider="openai",
            model=model,
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="s",
            finished_at="f",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            estimated_cost_usd=Decimal("0.001"),
            elapsed_ms=1,
        )
        return CompletionResult(
            text=text,
            tool_calls=(),
            usage=attempt.usage,
            response_id="resp",
            request_id="req",
            model_id=model,
            finish_reason="stop",
            cost_usd=Decimal("0.001"),
            attempts=(attempt,),
        )


def clean_trajectory():
    task_records = []
    messages = []
    for turn in range(1, 4):
        user = {"role": "user", "content": f"user {turn}"}
        assistant = {"role": "assistant", "content": f"assistant {turn}"}
        messages.extend((user, assistant))
        task_records.append(
            {
                "event": "task_turn",
                "task_turn": turn,
                "user_message": user,
                "assistant_message": assistant,
                "call": {"usage": {"input_tokens": turn * 100}},
            }
        )
    return {
        "complete": True,
        "arm": "clean",
        "transcript_sha256": "a" * 64,
        "model": "gpt-5.6-luna",
        "domain": "evolving_intent_gsm8k",
        "task_id": "x",
        "condition": "t7",
        "checkpoint_turns": [1, 2],
        "task_records": task_records,
        "messages": messages,
        # Gold-like material exists in the source artifact but is never passed
        # into any fake transport message.
        "evaluation": {"secret_gold": "DO_NOT_LEAK"},
    }


class ShadowTests(unittest.IsolatedAsyncioTestCase):
    def test_native_tool_prefix_keeps_full_public_turn(self):
        trajectory = clean_trajectory()
        turn = trajectory["task_records"][0]
        turn["messages"] = [
            turn["user_message"],
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "name": "lookup", "arguments": '{"answer":7}'}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"value":7}'},
            turn["assistant_message"],
        ]
        prefix = _prefix(trajectory, 1)
        self.assertEqual([message["role"] for message in prefix], ["user", "assistant", "tool", "assistant"])
        self.assertIn("tool_calls", prefix[1])

    async def test_prefix_only_zero_carry_and_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeTransport()
            source = clean_trajectory()
            before = list(source["messages"])
            result = await score_clean_trajectory(
                run_id="shadow-test",
                trajectory=source,
                transport=fake,
                event_path=Path(tmp) / "shadow.events.jsonl",
                output_path=Path(tmp) / "shadow.json",
                run_judge=True,
            )
            self.assertEqual(source["messages"], before)
            self.assertEqual(len(fake.calls), 8)
            self.assertFalse(any("DO_NOT_LEAK" in str(call) for call in fake.calls))
            methods = {record["method"] for record in result["records"]}
            self.assertEqual(
                methods,
                {
                    "turn_clock",
                    "context_use",
                    "trace_rules",
                    "frozen_probe",
                    "frozen_quiz",
                    "trace_judge",
                },
            )
            self.assertEqual(
                result["passive_monitor_spec_sha256"],
                PASSIVE_MONITOR_SPEC_SHA256,
            )
            # First checkpoint has only the first completed task exchange.
            self.assertIn("user 1", str(fake.calls[0][1]))
            self.assertNotIn("user 2", str(fake.calls[0][1]))

    async def test_partial_shadow_refuses_silent_rebill(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / "shadow.events.jsonl"
            event.write_text('{"partial":true}\n')
            with self.assertRaises(FileExistsError):
                await score_clean_trajectory(
                    run_id="shadow-test",
                    trajectory=clean_trajectory(),
                    transport=FakeTransport(),
                    event_path=event,
                    output_path=Path(tmp) / "shadow.json",
                )

    def test_evolving_quiz_is_deterministic_and_reads_only_completed_public_turns(self):
        trajectory = clean_trajectory()
        first = generate_evolving_passive_quiz(trajectory["task_records"], 2)
        changed = json.loads(json.dumps(trajectory["task_records"]))
        changed[2]["user_message"]["content"] = "UNOBSERVED FUTURE SECRET 999"
        changed[2]["assistant_message"]["content"] = "UNOBSERVED FUTURE ANSWER"
        second = generate_evolving_passive_quiz(changed, 2)
        self.assertEqual(first, second)
        self.assertEqual(first[0].expected, 2)
        self.assertTrue(all("answer" not in question.text.lower() for question in first))

    async def test_runtime_monitor_overrides_must_exactly_match_canonical_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            changed = canonical_passive_monitor_spec()
            changed["required_methods"].pop()
            with self.assertRaisesRegex(ValueError, "canonical frozen spec"):
                await score_clean_trajectory(
                    run_id="shadow-test",
                    trajectory=clean_trajectory(),
                    transport=FakeTransport(),
                    event_path=base / "missing.events.jsonl",
                    output_path=base / "missing.json",
                    passive_monitor_spec=changed,
                )
            changed = canonical_passive_monitor_spec()
            changed["extra"] = True
            with self.assertRaisesRegex(ValueError, "canonical frozen spec"):
                await score_clean_trajectory(
                    run_id="shadow-test",
                    trajectory=clean_trajectory(),
                    transport=FakeTransport(),
                    event_path=base / "extra.events.jsonl",
                    output_path=base / "extra.json",
                    passive_monitor_spec=changed,
                )
            with self.assertRaisesRegex(ValueError, "judge setting"):
                await score_clean_trajectory(
                    run_id="shadow-test",
                    trajectory=clean_trajectory(),
                    transport=FakeTransport(),
                    event_path=base / "judge.events.jsonl",
                    output_path=base / "judge.json",
                    run_judge=False,
                )
            with self.assertRaisesRegex(ValueError, "variants"):
                await score_clean_trajectory(
                    run_id="shadow-test",
                    trajectory=clean_trajectory(),
                    transport=FakeTransport(),
                    event_path=base / "variant.events.jsonl",
                    output_path=base / "variant.json",
                    frozen_probe_variants=("current_copy",),
                )

    async def test_shadow_reuse_rejects_changed_hash_and_missing_or_extra_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / "shadow.json"
            event = base / "shadow.events.jsonl"
            await score_clean_trajectory(
                run_id="shadow-test",
                trajectory=clean_trajectory(),
                transport=FakeTransport(),
                event_path=event,
                output_path=output,
            )
            original = read_json(output)
            cases = []
            changed_hash = json.loads(json.dumps(original))
            changed_hash["passive_monitor_spec_sha256"] = "f" * 64
            cases.append(changed_hash)
            missing = json.loads(json.dumps(original))
            missing["records"].pop()
            cases.append(missing)
            extra = json.loads(json.dumps(original))
            extra["records"].append(dict(extra["records"][0]))
            cases.append(extra)
            changed_variant = json.loads(json.dumps(original))
            probe = next(
                record
                for record in changed_variant["records"]
                if record["method"] == "frozen_probe"
            )
            probe["variant"] = "initial_recall"
            cases.append(changed_variant)
            for index, payload in enumerate(cases):
                atomic_write_json(output, payload)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    await score_clean_trajectory(
                        run_id="shadow-test",
                        trajectory=clean_trajectory(),
                        transport=FakeTransport(),
                        event_path=event,
                        output_path=output,
                    )


if __name__ == "__main__":
    unittest.main()
