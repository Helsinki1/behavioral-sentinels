"""No-network tests for the Experiment 12 stdlib provider transport."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error

from experiments12.core.artifacts import read_jsonl
from experiments12.core.budget import BudgetLedger, BudgetOverrun
from experiments12.core.transport import (
    JsonSchemaOutput,
    JsonSchemaTool,
    MissingCredentials,
    RequestValidationError,
    Transport,
    TransportError,
)


class FakeResponse:
    def __init__(
        self,
        payload: object | bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.status = status
        self.headers = headers or {}
        self.closed = False

    def read(self, limit: int = -1) -> bytes:
        return self.body if limit < 0 else self.body[:limit]

    def close(self) -> None:
        self.closed = True


class SequenceOpener:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float) -> object:
        self.calls.append((request, timeout))
        if not self.items:
            raise AssertionError("unexpected network dispatch")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def http_error(
    status: int,
    secret_text: str,
    *,
    retry_after: str | None = None,
    request_id: str | None = None,
) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    if request_id is not None:
        headers["X-Request-ID"] = request_id
    return urllib.error.HTTPError(
        "https://provider.invalid/v1/inference",
        status,
        secret_text,
        headers,
        io.BytesIO(secret_text.encode("utf-8")),
    )


def openai_payload(
    *,
    text: str = "OK",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached_tokens: int = 40,
    reasoning_tokens: int = 5,
) -> dict[str, object]:
    return {
        "id": "resp_123",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def fireworks_text_payload(text: str = "OK") -> dict[str, object]:
    return {
        "id": "chatcmpl-fw-1",
        "model": "accounts/fireworks/models/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 90,
            "completion_tokens": 10,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_transport_")
        self.root = Path(self.temp.name)
        self.ledger = BudgetLedger(
            self.root / "ledger.sqlite3",
            operational_caps_usd={"openai": "20", "fireworks": "10"},
        )
        self.events = self.root / "attempts.jsonl"
        self.sleeps: list[float] = []

    def tearDown(self) -> None:
        self.temp.cleanup()

    async def fake_sleep(self, delay: float) -> None:
        self.sleeps.append(delay)

    def transport(
        self,
        opener: SequenceOpener,
        *,
        environ: dict[str, str] | None = None,
        max_attempts: int = 3,
        random_value: float = 0.0,
    ) -> Transport:
        return Transport(
            self.ledger,
            self.events,
            environ=(
                {
                    "OPENAI_API_KEY": "OPENAI_SUPER_SECRET",
                    "FIREWORKS_API_KEY": "FIREWORKS_SUPER_SECRET",
                }
                if environ is None
                else environ
            ),
            urlopen=opener,
            sleep=self.fake_sleep,
            random_fn=lambda: random_value,
            max_attempts=max_attempts,
            base_backoff_seconds=1,
            max_backoff_seconds=10,
            jitter_seconds=1,
        )

    async def test_openai_responses_text_tools_schema_reasoning_and_ids(self) -> None:
        opener = SequenceOpener(
            FakeResponse(openai_payload(text='{"decision":"continue"}'), headers={"X-Request-ID": "req-openai-1"})
        )
        tool = JsonSchemaTool.from_schema(
            "read_state",
            "Read one state field",
            {
                "type": "object",
                "properties": {"field": {"type": "string"}},
                "required": ["field"],
                "additionalProperties": False,
            },
        )
        output = JsonSchemaOutput.from_schema(
            "decision",
            {
                "type": "object",
                "properties": {"decision": {"enum": ["continue", "reset"]}},
                "required": ["decision"],
                "additionalProperties": False,
            },
        )
        result = await self.transport(opener).complete(
            "gpt-5.6-luna",
            [
                {"role": "system", "content": "SYSTEM_PROMPT_MARKER_PRIVATE"},
                {"role": "user", "content": "USER_PROMPT_MARKER_PRIVATE"},
            ],
            purpose="agent_turn",
            request_key="task-1/control/turn-1",
            input_token_estimate=120,
            max_output_tokens=50,
            temperature=0.2,
            reasoning_effort="low",
            tools=(tool,),
            tool_choice="auto",
            output_schema=output,
        )
        self.assertEqual(result.text, '{"decision":"continue"}')
        self.assertEqual(result.response_id, "resp_123")
        self.assertEqual(result.request_id, "req-openai-1")
        self.assertEqual(result.model_id, "gpt-5.6-luna")
        self.assertEqual(result.usage.input_tokens, 100)
        self.assertEqual(result.usage.cached_input_tokens, 40)
        self.assertEqual(result.usage.reasoning_tokens, 5)
        self.assertEqual(len(result.attempts), 1)

        request, timeout = opener.calls[0]
        self.assertEqual(timeout, 120.0)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer OPENAI_SUPER_SECRET")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["tools"][0]["parameters"]["type"], "object")
        self.assertTrue(body["tools"][0]["strict"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(body["tool_choice"], "auto")

        event = read_jsonl(self.events)[0]
        serialized = json.dumps(event)
        self.assertEqual(event["status"], "succeeded")
        self.assertEqual(event["provider_request_id"], "req-openai-1")
        self.assertNotIn("OPENAI_SUPER_SECRET", serialized)
        self.assertNotIn("SYSTEM_PROMPT_MARKER_PRIVATE", serialized)
        self.assertNotIn("USER_PROMPT_MARKER_PRIVATE", serialized)
        self.assertEqual(self.sleeps, [])

    async def test_fireworks_chat_native_tool_call_and_request_shape(self) -> None:
        payload = {
            "id": "chatcmpl-fw-tool",
            "model": "accounts/fireworks/models/gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-7",
                                "type": "function",
                                "function": {
                                    "name": "reset_session",
                                    "arguments": '{"reason":"drift"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 90,
                "completion_tokens": 10,
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        }
        opener = SequenceOpener(
            FakeResponse(payload, headers={"X-Fireworks-Request-ID": "req-fw-7"})
        )
        tool = JsonSchemaTool.from_schema(
            "reset_session",
            "Reset the agent session",
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        )
        result = await self.transport(opener).complete(
            "gpt-oss-120b",
            [{"role": "user", "content": "PRIVATE_FIREWORKS_PROMPT"}],
            purpose="monitor",
            request_key="task-2/shadow/checkpoint-3",
            input_token_estimate=100,
            max_output_tokens=40,
            reasoning_effort="low",
            tools=(tool,),
            tool_choice="reset_session",
        )
        self.assertEqual(result.text, "")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].call_id, "call-7")
        self.assertEqual(result.tool_calls[0].name, "reset_session")
        self.assertEqual(result.tool_calls[0].arguments, {"reason": "drift"})
        self.assertEqual(result.request_id, "req-fw-7")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.usage.reasoning_tokens, 4)

        request, _ = opener.calls[0]
        body = json.loads(request.data)
        self.assertEqual(
            request.full_url,
            "https://api.fireworks.ai/inference/v1/chat/completions",
        )
        self.assertEqual(body["max_tokens"], 40)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["tools"][0]["function"]["name"], "reset_session")
        self.assertEqual(
            body["tool_choice"],
            {"type": "function", "function": {"name": "reset_session"}},
        )
        serialized = json.dumps(read_jsonl(self.events))
        self.assertNotIn("FIREWORKS_SUPER_SECRET", serialized)
        self.assertNotIn("PRIVATE_FIREWORKS_PROMPT", serialized)

    async def test_retry_after_is_honored_and_failed_attempt_is_sanitized(self) -> None:
        leaked = "AUTHORIZATION=Bearer VERY_SECRET_ERROR_TEXT"
        opener = SequenceOpener(
            http_error(429, leaked, retry_after="2", request_id="req-rate-limit"),
            FakeResponse(
                fireworks_text_payload("recovered"),
                headers={"X-Request-ID": "req-recovered"},
            ),
        )
        result = await self.transport(opener, max_attempts=2, random_value=0.5).complete(
            "gpt-oss-120b",
            [{"role": "user", "content": "RETRY_PROMPT_PRIVATE"}],
            purpose="agent_turn",
            request_key="task-3/control/turn-4",
            input_token_estimate=100,
            max_output_tokens=30,
        )
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(self.sleeps, [2.5])
        events = read_jsonl(self.events)
        self.assertEqual([e["status"] for e in events], ["failed", "succeeded"])
        self.assertEqual(events[0]["error_type"], "http_429")
        self.assertEqual(events[0]["provider_request_id"], "req-rate-limit")
        self.assertEqual(events[0]["error_message"], "provider returned an HTTP error")
        serialized = json.dumps(events)
        self.assertNotIn(leaked, serialized)
        self.assertNotIn("VERY_SECRET_ERROR_TEXT", serialized)
        self.assertNotIn("FIREWORKS_SUPER_SECRET", serialized)
        self.assertNotIn("RETRY_PROMPT_PRIVATE", serialized)
        budget = self.ledger.snapshot("fireworks")
        self.assertEqual(budget.reconciled_requests, 2)
        self.assertGreater(budget.upper_bound_spend_usd, Decimal("0"))

    async def test_nonretryable_http_error_has_no_secret_or_retry(self) -> None:
        leaked = "request rejected; key=PRIVATE_KEY_IN_EXCEPTION"
        opener = SequenceOpener(http_error(400, leaked, request_id="req-bad"))
        with self.assertRaises(TransportError) as caught:
            await self.transport(opener, max_attempts=3).complete(
                "gpt-5.6-luna",
                [{"role": "user", "content": "PRIVATE_BAD_REQUEST_BODY"}],
                purpose="judge",
                request_key="task-4/judge/checkpoint-2",
                input_token_estimate=50,
                max_output_tokens=10,
            )
        error = caught.exception
        self.assertEqual(error.category, "http_400")
        self.assertFalse(error.retryable)
        self.assertEqual(error.http_status, 400)
        self.assertEqual(len(error.attempts), 1)
        self.assertNotIn(leaked, str(error))
        self.assertEqual(self.sleeps, [])
        serialized = json.dumps(read_jsonl(self.events))
        self.assertNotIn("PRIVATE_KEY_IN_EXCEPTION", serialized)
        self.assertNotIn("PRIVATE_BAD_REQUEST_BODY", serialized)
        self.assertNotIn("OPENAI_SUPER_SECRET", serialized)

    async def test_invalid_success_response_is_unknown_cost_then_retried(self) -> None:
        opener = SequenceOpener(
            FakeResponse(b"not-json", headers={"X-Request-ID": "req-invalid"}),
            FakeResponse(openai_payload(text="valid"), headers={"X-Request-ID": "req-valid"}),
        )
        result = await self.transport(opener, max_attempts=2).complete(
            "gpt-5.6-luna",
            [{"role": "user", "content": "hello"}],
            purpose="agent_turn",
            request_key="task-5/control/turn-1",
            input_token_estimate=100,
            max_output_tokens=20,
        )
        self.assertEqual(result.text, "valid")
        events = read_jsonl(self.events)
        self.assertEqual(events[0]["error_type"], "invalid_response")
        self.assertEqual(events[0]["provider_request_id"], "req-invalid")
        self.assertEqual(self.sleeps, [1.0])
        self.assertGreater(self.ledger.snapshot("openai").upper_bound_spend_usd, 0)

    async def test_reservation_exists_before_dispatch(self) -> None:
        class InspectingOpener:
            def __init__(self, ledger: BudgetLedger) -> None:
                self.ledger = ledger
                self.called = False

            def __call__(self, request: object, *, timeout: float) -> FakeResponse:
                self.called = True
                snapshot = self.ledger.snapshot("fireworks")
                if snapshot.active_reservations != 1 or snapshot.reserved_usd <= 0:
                    raise AssertionError("network dispatched before reservation")
                return FakeResponse(fireworks_text_payload())

        opener = InspectingOpener(self.ledger)
        transport = Transport(
            self.ledger,
            self.events,
            environ={"FIREWORKS_API_KEY": "SECRET"},
            urlopen=opener,
            sleep=self.fake_sleep,
            random_fn=lambda: 0,
        )
        await transport.complete(
            "gpt-oss-120b",
            [{"role": "user", "content": "hello"}],
            purpose="agent_turn",
            request_key="task-6/control/turn-1",
            input_token_estimate=100,
            max_output_tokens=20,
        )
        self.assertTrue(opener.called)

    async def test_missing_credentials_does_not_reserve_dispatch_or_emit(self) -> None:
        opener = SequenceOpener(FakeResponse(openai_payload()))
        with self.assertRaises(MissingCredentials) as caught:
            await self.transport(opener, environ={}).complete(
                "gpt-5.6-luna",
                [{"role": "user", "content": "hello"}],
                purpose="agent_turn",
                request_key="task-7/control/turn-1",
                input_token_estimate=100,
                max_output_tokens=20,
            )
        self.assertEqual(caught.exception.environment_variable, "OPENAI_API_KEY")
        self.assertEqual(opener.calls, [])
        self.assertFalse(self.events.exists())
        self.assertEqual(self.ledger.snapshot("openai").reconciled_requests, 0)
        self.assertEqual(self.ledger.snapshot("openai").active_reservations, 0)

    async def test_invalid_schema_fails_before_reservation_or_dispatch(self) -> None:
        opener = SequenceOpener(FakeResponse(openai_payload()))
        with self.assertRaises(RequestValidationError):
            JsonSchemaTool.from_schema(
                "bad tool name",
                "invalid",
                {"type": "object"},
            )
        with self.assertRaises(RequestValidationError):
            await self.transport(opener).complete(
                "gpt-5.6-luna",
                [{"role": "user", "content": "hello"}],
                purpose="contains spaces",
                request_key="task-8/control/turn-1",
                input_token_estimate=100,
                max_output_tokens=20,
            )
        self.assertEqual(opener.calls, [])
        self.assertEqual(self.ledger.snapshot("openai").active_reservations, 0)

    async def test_overrun_is_logged_before_budget_exception(self) -> None:
        path = self.root / "small-ledger.sqlite3"
        ledger = BudgetLedger(path, operational_caps_usd={"openai": "0.001"})
        events = self.root / "overrun.jsonl"
        opener = SequenceOpener(
            FakeResponse(
                openai_payload(
                    input_tokens=1_000_000,
                    output_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                ),
                headers={"X-Request-ID": "req-overrun"},
            )
        )
        transport = Transport(
            ledger,
            events,
            environ={"OPENAI_API_KEY": "SECRET"},
            urlopen=opener,
            sleep=self.fake_sleep,
            random_fn=lambda: 0,
        )
        with self.assertRaises(BudgetOverrun):
            await transport.complete(
                "gpt-5.6-luna",
                [{"role": "user", "content": "hello"}],
                purpose="agent_turn",
                request_key="task-9/control/turn-1",
                input_token_estimate=1,
                max_output_tokens=1,
            )
        logged = read_jsonl(events)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["status"], "succeeded")
        self.assertEqual(logged[0]["provider_request_id"], "req-overrun")
        self.assertEqual(ledger.snapshot("openai").stop_reason, "operational")

    async def test_cancellation_is_reconciled_unknown_and_emitted(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        class BlockingOpener:
            def __call__(self, request: object, *, timeout: float) -> FakeResponse:
                entered.set()
                release.wait(timeout=5)
                finished.set()
                return FakeResponse(fireworks_text_payload())

        transport = Transport(
            self.ledger,
            self.events,
            environ={"FIREWORKS_API_KEY": "SECRET"},
            urlopen=BlockingOpener(),
            sleep=self.fake_sleep,
            random_fn=lambda: 0,
        )
        task = asyncio.create_task(
            transport.complete(
                "gpt-oss-120b",
                [{"role": "user", "content": "PRIVATE_CANCELLED_PROMPT"}],
                purpose="agent_turn",
                request_key="task-10/control/turn-1",
                input_token_estimate=100,
                max_output_tokens=20,
            )
        )
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        try:
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 2)
        events = read_jsonl(self.events)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "unknown")
        self.assertEqual(events[0]["error_type"], "cancelled")
        self.assertNotIn("PRIVATE_CANCELLED_PROMPT", json.dumps(events))
        budget = self.ledger.snapshot("fireworks")
        self.assertEqual(budget.active_reservations, 0)
        self.assertEqual(budget.reconciled_requests, 1)
        self.assertGreater(budget.upper_bound_spend_usd, Decimal("0"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
