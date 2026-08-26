from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from experiments12.core.artifacts import sha256_json
from experiments12.core.schemas import CallAttemptRecord, CallStatus, TokenUsage
from experiments12.core.transport import CompletionResult
from experiments12.domains.base import DomainTask, DomainTurn, canonical_json_sha256
from experiments12.harness12 import HarnessConfig, grade_final_numeric, run_scripted_task


def task() -> DomainTask:
    return DomainTask(
        domain="evolving_intent_gsm8k",
        task_id="x",
        condition="t7",
        turns=(
            DomainTurn(1, "Start with 2 apples."),
            DomainTurn(2, "Actually use 3 apples."),
            DomainTurn(3, "What is twice that?"),
        ),
        evaluation_label="6",
        source_sha256="a" * 64,
        task_sha256="b" * 64,
    )


class FakeTransport:
    def __init__(self):
        self.calls = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append(([dict(m) for m in messages], kwargs))
        purpose = kwargs["purpose"]
        text = "PROBE: " + "A" * 8 if purpose == "active_probe" else "Answer: 6"
        attempt = CallAttemptRecord(
            event_id=f"event{len(self.calls)}",
            reservation_id=f"reservation{len(self.calls)}",
            provider="openai",
            model=model,
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="start",
            finished_at="finish",
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


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_numeric_grader(self):
        self.assertEqual(grade_final_numeric("work \\boxed{1,234}", "1234"), ("1,234", True))
        self.assertEqual(grade_final_numeric("Answer: 7", "6"), ("7", False))
        self.assertEqual(
            grade_final_numeric(r"work \boxed{8\text{ years old}}", "8"),
            (r"8\text{ years old}", True),
        )
        self.assertEqual(
            grade_final_numeric(r"Answer: \$75.00", "75"),
            (r"\$75.00", True),
        )
        self.assertEqual(
            grade_final_numeric(
                r"Answer: Carlos starts earning a profit in the 13th year.", "13"
            ),
            ("Carlos starts earning a profit in the 13th year", True),
        )
        self.assertEqual(
            grade_final_numeric(r"work \boxed{\frac{1}{2}}", "0.5"),
            (r"\frac{1}{2}", True),
        )
        self.assertEqual(
            grade_final_numeric(r"work \boxed{\frac{1}{2}\text{ cup}}", "0.5"),
            (r"\frac{1}{2}\text{ cup}", True),
        )
        self.assertEqual(
            grade_final_numeric(r"work says 8, but \boxed{7}", "8"),
            ("7", False),
        )
        self.assertEqual(
            grade_final_numeric(r"\boxed{The answer could be 7 or 8}", "8"),
            ("The answer could be 7 or 8", False),
        )

    async def test_clean_has_no_probe_and_active_is_carried(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeTransport()
            clean = await run_scripted_task(
                run_id="r",
                cell_id="clean-cell",
                model="gpt-5.6-luna",
                task=task(),
                arm_name="clean",
                transport=fake,
                event_path=Path(tmp) / "clean.events.jsonl",
                output_path=Path(tmp) / "clean.json",
                config=HarnessConfig(task_max_output_tokens=20),
            )
            self.assertEqual(len(fake.calls), 3)
            self.assertEqual(clean["probe_records"], [])
            self.assertTrue(clean["evaluation"]["success"])

            fake = FakeTransport()
            active = await run_scripted_task(
                run_id="r",
                cell_id="active-cell",
                model="gpt-5.6-luna",
                task=task(),
                arm_name="active_name_copy",
                transport=fake,
                event_path=Path(tmp) / "active.events.jsonl",
                output_path=Path(tmp) / "active.json",
                config=HarnessConfig(task_max_output_tokens=20),
            )
            self.assertEqual(len(fake.calls), 5)
            self.assertEqual(len(active["probe_records"]), 2)
            search_from = 0
            for probe in active["probe_records"]:
                user_index = active["messages"].index(
                    probe["user_message"], search_from
                )
                assistant_index = user_index + 1
                self.assertEqual(
                    active["messages"][assistant_index], probe["assistant_message"]
                )
                self.assertEqual(
                    probe["source_prefix_sha256"],
                    sha256_json(active["messages"][: assistant_index + 1]),
                )
                search_from = assistant_index + 1
            # The second task call sees the first probe prompt and response.
            second_task_messages = fake.calls[2][0]
            self.assertTrue(any("ACTIVE CARRIED PROBE" in m["content"] for m in second_task_messages))

    async def test_completed_materialization_is_idempotent_no_rebill(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeTransport()
            kwargs = dict(
                run_id="r",
                cell_id="cell",
                model="gpt-5.6-luna",
                task=task(),
                arm_name="clean",
                transport=fake,
                event_path=Path(tmp) / "events.jsonl",
                output_path=Path(tmp) / "trajectory.json",
                config=HarnessConfig(task_max_output_tokens=20),
            )
            first = await run_scripted_task(**kwargs)
            n_calls = len(fake.calls)
            second = await run_scripted_task(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(len(fake.calls), n_calls)

    async def test_low_level_runtime_rejects_active_t1_before_any_call(self):
        source = task()
        t1 = DomainTask(
            domain=source.domain,
            task_id=source.task_id,
            condition="t1",
            turns=(source.turns[0],),
            evaluation_label=source.evaluation_label,
            source_sha256=source.source_sha256,
            task_sha256=canonical_json_sha256({"task": "t1"}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeTransport()
            with self.assertRaisesRegex(ValueError, "forbidden for t1"):
                await run_scripted_task(
                    run_id="r",
                    cell_id="active-t1",
                    model="gpt-5.6-luna",
                    task=t1,
                    arm_name="active_counter",
                    transport=fake,
                    event_path=Path(tmp) / "events.jsonl",
                    output_path=Path(tmp) / "trajectory.json",
                )
            self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
