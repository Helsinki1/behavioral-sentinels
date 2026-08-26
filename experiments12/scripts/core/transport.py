"""Budgeted, retrying stdlib transport for Experiment 12 model calls.

Supported wire protocols:

* OpenAI Responses API.
* Fireworks' OpenAI-compatible Chat Completions API.

Every actual HTTP attempt is reserved in :mod:`experiments12.core.budget`
before dispatch, reconciled afterward (conservatively at the reserved upper
bound when billing is unknown), and immediately appended as one sanitized
``CallAttemptRecord``.  Event records never contain prompts, response bodies,
authorization headers, API keys, or exception text.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence
import urllib.error
import urllib.request
from uuid import uuid4

from experiments12.models12 import (
    CATALOG,
    ModelCatalog,
    ModelSpec,
    estimate_call_upper_bound_usd,
)

from .artifacts import append_jsonl, canonical_json_bytes
from .budget import BudgetLedger, BudgetOverrun, ReconciliationResult
from .schemas import CallAttemptRecord, CallStatus, TokenUsage


_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_./:-]{1,256}$")
_PURPOSE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_REQUEST_KEY_RE = re.compile(r"^[A-Za-z0-9_./:-]{1,240}$")
_RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_USD_QUANTUM = Decimal("0.000001")


class TransportError(RuntimeError):
    """Sanitized terminal transport failure."""

    def __init__(
        self,
        category: str,
        *,
        retryable: bool,
        http_status: int | None,
        attempts: tuple[CallAttemptRecord, ...],
    ) -> None:
        self.category = category
        self.retryable = retryable
        self.http_status = http_status
        self.attempts = attempts
        status = "" if http_status is None else f" (HTTP {http_status})"
        super().__init__(f"provider request failed: {category}{status}")


class MissingCredentials(TransportError):
    def __init__(self, environment_variable: str) -> None:
        self.environment_variable = environment_variable
        super().__init__(
            "missing_credentials",
            retryable=False,
            http_status=None,
            attempts=(),
        )


class RequestValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class JsonSchemaTool:
    """Immutable native function tool definition."""

    name: str
    description: str
    schema_json: str
    strict: bool = True

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise RequestValidationError("tool name is invalid")
        if not isinstance(self.description, str):
            raise RequestValidationError("tool description must be a string")
        if not isinstance(self.strict, bool):
            raise RequestValidationError("tool strict must be boolean")
        schema = self.schema
        if not isinstance(schema, Mapping) or schema.get("type") != "object":
            raise RequestValidationError("tool JSON schema must have object type")

    @property
    def schema(self) -> dict[str, Any]:
        try:
            value = json.loads(self.schema_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RequestValidationError("tool schema_json must be valid JSON") from exc
        if not isinstance(value, dict):
            raise RequestValidationError("tool schema_json must encode an object")
        return value

    @classmethod
    def from_schema(
        cls,
        name: str,
        description: str,
        schema: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> "JsonSchemaTool":
        if not isinstance(schema, Mapping):
            raise RequestValidationError("tool schema must be an object")
        return cls(
            name=name,
            description=description,
            schema_json=canonical_json_bytes(schema).decode("utf-8"),
            strict=strict,
        )


@dataclass(frozen=True, slots=True)
class JsonSchemaOutput:
    """Provider-native structured text output format."""

    name: str
    schema_json: str
    strict: bool = True
    description: str | None = None

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise RequestValidationError("output schema name is invalid")
        if not isinstance(self.strict, bool):
            raise RequestValidationError("output schema strict must be boolean")
        if self.description is not None and not isinstance(self.description, str):
            raise RequestValidationError("output schema description must be a string or None")
        if not isinstance(self.schema, dict):
            raise RequestValidationError("output schema_json must encode an object")

    @property
    def schema(self) -> dict[str, Any]:
        try:
            value = json.loads(self.schema_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RequestValidationError("output schema_json must be valid JSON") from exc
        if not isinstance(value, dict):
            raise RequestValidationError("output schema_json must encode an object")
        return value

    @classmethod
    def from_schema(
        cls,
        name: str,
        schema: Mapping[str, Any],
        *,
        strict: bool = True,
        description: str | None = None,
    ) -> "JsonSchemaOutput":
        if not isinstance(schema, Mapping):
            raise RequestValidationError("output schema must be an object")
        return cls(
            name=name,
            schema_json=canonical_json_bytes(schema).decode("utf-8"),
            strict=strict,
            description=description,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        if not _safe_identifier(self.call_id):
            raise ValueError("provider returned an invalid tool call id")
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise ValueError("provider returned an invalid tool name")
        arguments = self.arguments
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")

    @property
    def arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return value


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage
    response_id: str | None
    request_id: str | None
    model_id: str
    finish_reason: str | None
    cost_usd: Decimal
    attempts: tuple[CallAttemptRecord, ...]


@dataclass(frozen=True, slots=True)
class _WireResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class _ParsedResponse:
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage
    usage_payload: Mapping[str, Any]
    response_id: str | None
    request_id: str | None
    model_id: str
    finish_reason: str | None


class _AttemptFailure(Exception):
    """Internal error containing only sanitized fields."""

    def __init__(
        self,
        category: str,
        safe_message: str,
        *,
        retryable: bool,
        http_status: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.category = category
        self.safe_message = safe_message
        self.retryable = retryable
        self.http_status = http_status
        self.headers = headers or {}
        super().__init__(safe_message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_identifier(value: Any) -> str | None:
    if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value):
        return value
    return None


def _whole_nonnegative(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RequestValidationError(f"{name} must be a non-negative integer")
    if positive and value == 0:
        raise RequestValidationError(f"{name} must be positive")
    return value


def _headers(value: Any) -> dict[str, str]:
    try:
        items = value.items()
    except Exception:
        return {}
    result: dict[str, str] = {}
    for key, item in items:
        if isinstance(key, str) and isinstance(item, str):
            result[key.lower()] = item
    return result


def _request_id(headers: Mapping[str, str], payload: Mapping[str, Any] | None = None) -> str | None:
    for key in ("x-request-id", "openai-request-id", "x-fireworks-request-id"):
        candidate = _safe_identifier(headers.get(key))
        if candidate:
            return candidate
    if payload is not None:
        for key in ("request_id", "_request_id"):
            candidate = _safe_identifier(payload.get(key))
            if candidate:
                return candidate
    return None


def _arguments_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise ValueError("tool arguments must be a JSON object")
    return canonical_json_bytes(parsed).decode("utf-8")


def _normalize_tool_call(value: Any) -> dict[str, Any]:
    if isinstance(value, ToolCall):
        return {"id": value.call_id, "name": value.name, "arguments": value.arguments_json}
    if not isinstance(value, Mapping):
        raise RequestValidationError("assistant tool_calls must be objects")
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", "{}")
    else:
        name = value.get("name")
        arguments = value.get("arguments", "{}")
    call_id = value.get("id", value.get("call_id"))
    if not _safe_identifier(call_id) or not isinstance(name, str) or not _TOOL_NAME_RE.fullmatch(name):
        raise RequestValidationError("assistant tool call id or name is invalid")
    try:
        normalized_arguments = _arguments_json(arguments)
    except ValueError as exc:
        raise RequestValidationError("assistant tool arguments must be a JSON object") from exc
    return {"id": call_id, "name": name, "arguments": normalized_arguments}


def _normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise RequestValidationError("messages must be a non-empty sequence")
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise RequestValidationError("every message must be an object")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise RequestValidationError("message role is invalid")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RequestValidationError("message content must be text")
        item: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            call_id = message.get("tool_call_id", message.get("call_id"))
            if not _safe_identifier(call_id):
                raise RequestValidationError("tool message requires a safe tool_call_id")
            item["tool_call_id"] = call_id
        raw_calls = message.get("tool_calls")
        if raw_calls is not None:
            if role != "assistant" or isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
                raise RequestValidationError("tool_calls are only valid on assistant messages")
            item["tool_calls"] = [_normalize_tool_call(call) for call in raw_calls]
        result.append(item)
    return result


def _openai_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
            continue
        if message["content"] or not message.get("tool_calls"):
            result.append({"role": role, "content": message["content"]})
        for call in message.get("tool_calls", ()):  # prior assistant function calls
            result.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
            )
    return result


def _fireworks_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message["role"], "content": message["content"]}
        if message["role"] == "tool":
            item["tool_call_id"] = message["tool_call_id"]
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in message["tool_calls"]
            ]
        result.append(item)
    return result


def _tool_choice(value: str | None, *, provider: str) -> Any:
    if value is None:
        return None
    if value in {"auto", "none", "required"}:
        return value
    if not _TOOL_NAME_RE.fullmatch(value):
        raise RequestValidationError("tool_choice must be auto, none, required, or a tool name")
    if provider == "openai":
        return {"type": "function", "name": value}
    return {"type": "function", "function": {"name": value}}


def _build_request_body(
    spec: ModelSpec,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_output_tokens: int,
    temperature: float | None,
    reasoning_effort: str | None,
    tools: Sequence[JsonSchemaTool],
    tool_choice: str | None,
    output_schema: JsonSchemaOutput | None,
) -> dict[str, Any]:
    if spec.provider == "openai":
        body: dict[str, Any] = {
            "model": spec.model,
            "input": _openai_messages(messages),
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                    "strict": tool.strict,
                }
                for tool in tools
            ]
        choice = _tool_choice(tool_choice, provider=spec.provider)
        if choice is not None:
            body["tool_choice"] = choice
        if output_schema is not None:
            fmt: dict[str, Any] = {
                "type": "json_schema",
                "name": output_schema.name,
                "schema": output_schema.schema,
                "strict": output_schema.strict,
            }
            if output_schema.description is not None:
                fmt["description"] = output_schema.description
            body["text"] = {"format": fmt}
        return body

    if spec.provider == "fireworks":
        body = {
            "model": spec.model,
            "messages": _fireworks_messages(messages),
            "max_tokens": max_output_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema,
                        "strict": tool.strict,
                    },
                }
                for tool in tools
            ]
        choice = _tool_choice(tool_choice, provider=spec.provider)
        if choice is not None:
            body["tool_choice"] = choice
        if output_schema is not None:
            schema_body: dict[str, Any] = {
                "name": output_schema.name,
                "schema": output_schema.schema,
                "strict": output_schema.strict,
            }
            if output_schema.description is not None:
                schema_body["description"] = output_schema.description
            body["response_format"] = {"type": "json_schema", "json_schema": schema_body}
        return body
    raise RequestValidationError("catalog model has unsupported provider")


def _usage_int(value: Any, name: str, *, required: bool = False) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage {name} is invalid")
    return value


def _parse_usage(payload: Any, *, provider: str) -> tuple[TokenUsage, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("response usage is missing")
    if provider == "openai":
        input_tokens = _usage_int(payload.get("input_tokens"), "input_tokens", required=True)
        output_tokens = _usage_int(payload.get("output_tokens"), "output_tokens", required=True)
        input_details = payload.get("input_tokens_details") or {}
        output_details = payload.get("output_tokens_details") or {}
    else:
        input_tokens = _usage_int(payload.get("prompt_tokens"), "prompt_tokens", required=True)
        output_tokens = _usage_int(
            payload.get("completion_tokens"), "completion_tokens", required=True
        )
        input_details = payload.get("prompt_tokens_details") or {}
        output_details = payload.get("completion_tokens_details") or {}
    if not isinstance(input_details, Mapping) or not isinstance(output_details, Mapping):
        raise ValueError("usage detail fields are invalid")
    cached = _usage_int(
        input_details.get("cached_tokens", input_details.get("cached_input_tokens", 0)),
        "cached_tokens",
    )
    reasoning = _usage_int(
        output_details.get("reasoning_tokens", 0), "reasoning_tokens"
    )
    total = _usage_int(payload.get("total_tokens"), "total_tokens")
    return (
        TokenUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cached_input_tokens=cached or 0,
            reasoning_tokens=reasoning or 0,
            provider_reported_total_tokens=total,
        ),
        payload,
    )


def _parse_tool_call(value: Any) -> ToolCall:
    if not isinstance(value, Mapping):
        raise ValueError("tool call is invalid")
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name")
        arguments = value.get("arguments")
    call_id = value.get("call_id", value.get("id"))
    if not isinstance(name, str):
        raise ValueError("tool call name is missing")
    return ToolCall(
        call_id=str(call_id) if call_id is not None else "",
        name=name,
        arguments_json=_arguments_json(arguments),
    )


def _parse_openai(payload: Any, headers: Mapping[str, str], spec: ModelSpec) -> _ParsedResponse:
    if not isinstance(payload, Mapping):
        raise ValueError("response root is invalid")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("response output is invalid")
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise ValueError("response message content is invalid")
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") in {"function_call", "tool_call"}:
                    calls.append(_parse_tool_call(part))
        elif kind in {"function_call", "tool_call"}:
            calls.append(_parse_tool_call(item))
    usage, usage_payload = _parse_usage(payload.get("usage"), provider="openai")
    model_id = _safe_identifier(payload.get("model")) or spec.model
    response_id = _safe_identifier(payload.get("id"))
    finish = payload.get("status")
    if not isinstance(finish, str):
        finish = None
    incomplete = payload.get("incomplete_details")
    if isinstance(incomplete, Mapping) and isinstance(incomplete.get("reason"), str):
        finish = incomplete["reason"]
    return _ParsedResponse(
        text="".join(text_parts),
        tool_calls=tuple(calls),
        usage=usage,
        usage_payload=usage_payload,
        response_id=response_id,
        request_id=_request_id(headers, payload),
        model_id=model_id,
        finish_reason=finish,
    )


def _parse_fireworks(
    payload: Any, headers: Mapping[str, str], spec: ModelSpec
) -> _ParsedResponse:
    if not isinstance(payload, Mapping):
        raise ValueError("response root is invalid")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("response choices are invalid")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("response message is invalid")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("response text is invalid")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ValueError("response tool_calls are invalid")
    calls = tuple(_parse_tool_call(call) for call in raw_calls)
    usage, usage_payload = _parse_usage(payload.get("usage"), provider="fireworks")
    model_id = _safe_identifier(payload.get("model")) or spec.model
    finish = choice.get("finish_reason")
    if not isinstance(finish, str):
        finish = None
    return _ParsedResponse(
        text=content,
        tool_calls=calls,
        usage=usage,
        usage_payload=usage_payload,
        response_id=_safe_identifier(payload.get("id")),
        request_id=_request_id(headers, payload),
        model_id=model_id,
        finish_reason=finish,
    )


def _reported_cost_usd(usage_payload: Mapping[str, Any]) -> Decimal | None:
    for key in ("total_cost_usd", "cost_usd"):
        value = usage_payload.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            cost = Decimal(str(value))
        except (InvalidOperation, ValueError):
            continue
        if cost.is_finite() and cost >= 0:
            return cost.quantize(_USD_QUANTUM, rounding=ROUND_CEILING)
    return None


def _estimated_actual_cost_usd(spec: ModelSpec, usage: TokenUsage, catalog: ModelCatalog) -> Decimal:
    pricing = spec.pricing
    long_context = pricing.long_context
    if long_context is not None and usage.input_tokens > long_context.threshold_input_tokens:
        uncached_rate = max(
            long_context.input_per_million_usd,
            long_context.cache_write_per_million_usd,
        )
        cached_rate = long_context.cached_input_per_million_usd
        output_rate = long_context.output_per_million_usd
    else:
        uncached_rate = pricing.input_per_million_usd
        if pricing.cache_write_per_million_usd is not None:
            uncached_rate = max(uncached_rate, pricing.cache_write_per_million_usd)
        cached_rate = pricing.cached_input_per_million_usd
        output_rate = pricing.output_per_million_usd
    cached_tokens = min(usage.cached_input_tokens, usage.input_tokens)
    uncached_tokens = usage.input_tokens - cached_tokens
    cost = (
        Decimal(uncached_tokens) * uncached_rate
        + Decimal(cached_tokens) * cached_rate
        + Decimal(usage.output_tokens) * output_rate
    ) / Decimal(catalog.token_unit)
    return cost.quantize(_USD_QUANTUM, rounding=ROUND_CEILING)


def _retry_after_seconds(headers: Mapping[str, str], now: datetime | None = None) -> float | None:
    value = headers.get("retry-after")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        delay = float(value.strip())
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            delay = (target - current).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(delay):
        return None
    return max(0.0, delay)


class Transport:
    """One reusable async transport sharing a run-level budget and event log."""

    def __init__(
        self,
        ledger: BudgetLedger,
        event_log_path: str | Path,
        *,
        catalog: ModelCatalog = CATALOG,
        environ: Mapping[str, str] | None = None,
        urlopen: Callable[..., Any] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        random_fn: Callable[[], float] | None = None,
        timeout_seconds: float = 120.0,
        max_attempts: int = 6,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        jitter_seconds: float = 1.0,
        max_response_bytes: int = 8_000_000,
    ) -> None:
        if not isinstance(ledger, BudgetLedger):
            raise TypeError("ledger must be BudgetLedger")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (
            ("base_backoff_seconds", base_backoff_seconds),
            ("max_backoff_seconds", max_backoff_seconds),
            ("jitter_seconds", jitter_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if max_backoff_seconds < base_backoff_seconds:
            raise ValueError("max_backoff_seconds cannot be below base_backoff_seconds")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be positive")
        self.ledger = ledger
        self.event_log_path = Path(event_log_path)
        self.catalog = catalog
        self.environ = os.environ if environ is None else environ
        self.urlopen = urllib.request.urlopen if urlopen is None else urlopen
        self.sleep = asyncio.sleep if sleep is None else sleep
        self.random_fn = random.random if random_fn is None else random_fn
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = max_attempts
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.jitter_seconds = float(jitter_seconds)
        self.max_response_bytes = max_response_bytes

    async def complete(
        self,
        model_name: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        purpose: str,
        request_key: str,
        input_token_estimate: int,
        max_output_tokens: int,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        tools: Sequence[JsonSchemaTool] = (),
        tool_choice: str | None = None,
        output_schema: JsonSchemaOutput | None = None,
    ) -> CompletionResult:
        """Call one frozen catalog model, with budgeted attempts and safe audit events."""

        try:
            spec = self.catalog.models[model_name]
        except KeyError as exc:
            raise RequestValidationError(f"unknown catalog model {model_name!r}") from exc
        if not _PURPOSE_RE.fullmatch(purpose):
            raise RequestValidationError("purpose must be a short identifier")
        if not _REQUEST_KEY_RE.fullmatch(request_key):
            raise RequestValidationError("request_key must be a short non-secret identifier")
        input_token_estimate = _whole_nonnegative(
            input_token_estimate, "input_token_estimate"
        )
        max_output_tokens = _whole_nonnegative(
            max_output_tokens, "max_output_tokens", positive=True
        )
        if temperature is not None:
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or not 0 <= temperature <= 2
            ):
                raise RequestValidationError("temperature must be between 0 and 2")
            temperature = float(temperature)
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
        ):
            raise RequestValidationError("reasoning_effort must be a non-empty string or None")
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise RequestValidationError("tools must be a sequence")
        if any(not isinstance(tool, JsonSchemaTool) for tool in tools):
            raise RequestValidationError("tools must contain JsonSchemaTool records")
        if output_schema is not None and not isinstance(output_schema, JsonSchemaOutput):
            raise RequestValidationError("output_schema must be JsonSchemaOutput or None")
        normalized_messages = _normalize_messages(messages)
        body = _build_request_body(
            spec,
            normalized_messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort.strip() if reasoning_effort else None,
            tools=tools,
            tool_choice=tool_choice,
            output_schema=output_schema,
        )
        try:
            body_bytes = canonical_json_bytes(body)
        except (TypeError, ValueError) as exc:
            raise RequestValidationError("request body is not valid JSON data") from exc
        api_key = self.environ.get(spec.api_key_env, "")
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingCredentials(spec.api_key_env) from None

        upper_bound = estimate_call_upper_bound_usd(
            model_name,
            input_token_estimate,
            max_output_tokens,
            catalog=self.catalog,
        )
        attempts: list[CallAttemptRecord] = []
        for attempt_number in range(1, self.max_attempts + 1):
            reservation = self.ledger.reserve(
                spec.provider,
                upper_bound,
                purpose=purpose,
                request_key=f"{request_key}/attempt-{attempt_number}",
            )
            request = urllib.request.Request(
                spec.inference_url,
                data=body_bytes,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "behavioral-sentinels-exp12/1",
                },
                method="POST",
            )
            started_at = _utc_now()
            started_clock = time.monotonic()
            try:
                wire = await asyncio.to_thread(self._dispatch, request)
                try:
                    payload = json.loads(wire.body.decode("utf-8"))
                    parsed = (
                        _parse_openai(payload, wire.headers, spec)
                        if spec.provider == "openai"
                        else _parse_fireworks(payload, wire.headers, spec)
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                    raise _AttemptFailure(
                        "invalid_response",
                        "provider returned an invalid response",
                        retryable=True,
                        http_status=wire.status,
                        headers=wire.headers,
                    ) from None
            except asyncio.CancelledError:
                # The worker thread may still reach the provider after the
                # coroutine is cancelled, so billing is unknown and must be
                # conservatively retained before cancellation propagates.
                elapsed_ms = max(0, round((time.monotonic() - started_clock) * 1000))
                result = self.ledger.reconcile_unknown(
                    reservation.reservation_id,
                    raise_on_overrun=False,
                )
                record = CallAttemptRecord(
                    event_id=uuid4().hex,
                    reservation_id=reservation.reservation_id,
                    provider=spec.provider,
                    model=spec.model,
                    purpose=purpose,
                    attempt_number=attempt_number,
                    status=CallStatus.UNKNOWN,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    usage=result.reservation.usage,
                    estimated_cost_usd=result.reservation.actual_usd,
                    elapsed_ms=elapsed_ms,
                    error_type="cancelled",
                    error_message="provider request was cancelled with billing unknown",
                )
                self._emit(record)
                attempts.append(record)
                self._raise_if_overrun(result)
                raise
            except _AttemptFailure as failure:
                elapsed_ms = max(0, round((time.monotonic() - started_clock) * 1000))
                result = self.ledger.reconcile_unknown(
                    reservation.reservation_id,
                    provider_request_id=_request_id(failure.headers),
                    raise_on_overrun=False,
                )
                record = CallAttemptRecord(
                    event_id=uuid4().hex,
                    reservation_id=reservation.reservation_id,
                    provider=spec.provider,
                    model=spec.model,
                    purpose=purpose,
                    attempt_number=attempt_number,
                    status=CallStatus.FAILED,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    usage=result.reservation.usage,
                    estimated_cost_usd=result.reservation.actual_usd,
                    elapsed_ms=elapsed_ms,
                    provider_request_id=result.reservation.provider_request_id,
                    error_type=failure.category,
                    error_message=failure.safe_message,
                )
                self._emit(record)
                attempts.append(record)
                self._raise_if_overrun(result)
                if not failure.retryable or attempt_number >= self.max_attempts:
                    raise TransportError(
                        failure.category,
                        retryable=failure.retryable,
                        http_status=failure.http_status,
                        attempts=tuple(attempts),
                    ) from None
                await self.sleep(self._retry_delay(attempt_number, failure.headers))
                continue
            except Exception:
                # This branch protects against secret-bearing text from an
                # injected/custom transport.  It must never be copied or chained.
                elapsed_ms = max(0, round((time.monotonic() - started_clock) * 1000))
                result = self.ledger.reconcile_unknown(
                    reservation.reservation_id,
                    raise_on_overrun=False,
                )
                record = CallAttemptRecord(
                    event_id=uuid4().hex,
                    reservation_id=reservation.reservation_id,
                    provider=spec.provider,
                    model=spec.model,
                    purpose=purpose,
                    attempt_number=attempt_number,
                    status=CallStatus.FAILED,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    usage=result.reservation.usage,
                    estimated_cost_usd=result.reservation.actual_usd,
                    elapsed_ms=elapsed_ms,
                    error_type="unexpected_error",
                    error_message="unexpected provider transport failure",
                )
                self._emit(record)
                attempts.append(record)
                self._raise_if_overrun(result)
                raise TransportError(
                    "unexpected_error",
                    retryable=False,
                    http_status=None,
                    attempts=tuple(attempts),
                ) from None

            reported_cost = _reported_cost_usd(parsed.usage_payload)
            actual_cost = (
                reported_cost
                if reported_cost is not None
                else _estimated_actual_cost_usd(spec, parsed.usage, self.catalog)
            )
            cost_quality = "reported" if reported_cost is not None else "estimated"
            reconciliation = self.ledger.reconcile(
                reservation.reservation_id,
                actual_cost,
                usage=parsed.usage,
                cost_quality=cost_quality,
                request_status=CallStatus.SUCCEEDED,
                provider_request_id=parsed.request_id,
                raise_on_overrun=False,
            )
            elapsed_ms = max(0, round((time.monotonic() - started_clock) * 1000))
            record = CallAttemptRecord(
                event_id=uuid4().hex,
                reservation_id=reservation.reservation_id,
                provider=spec.provider,
                model=spec.model,
                purpose=purpose,
                attempt_number=attempt_number,
                status=CallStatus.SUCCEEDED,
                started_at=started_at,
                finished_at=_utc_now(),
                usage=parsed.usage,
                estimated_cost_usd=actual_cost,
                elapsed_ms=elapsed_ms,
                provider_request_id=parsed.request_id,
                finish_reason=parsed.finish_reason,
            )
            self._emit(record)
            attempts.append(record)
            self._raise_if_overrun(reconciliation)
            return CompletionResult(
                text=parsed.text,
                tool_calls=parsed.tool_calls,
                usage=parsed.usage,
                response_id=parsed.response_id,
                request_id=parsed.request_id,
                model_id=parsed.model_id,
                finish_reason=parsed.finish_reason,
                cost_usd=actual_cost,
                attempts=tuple(attempts),
            )
        raise AssertionError("attempt loop exhausted without a terminal result")

    def _dispatch(self, request: urllib.request.Request) -> _WireResponse:
        try:
            response = self.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            headers = _headers(exc.headers)
            status = exc.code if isinstance(exc.code, int) else None
            try:
                exc.close()
            except Exception:
                pass
            raise _AttemptFailure(
                f"http_{status}" if status is not None else "http_error",
                "provider returned an HTTP error",
                retryable=status in _RETRYABLE_HTTP,
                http_status=status,
                headers=headers,
            ) from None
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError):
            raise _AttemptFailure(
                "network_error",
                "provider network request failed",
                retryable=True,
            ) from None
        except Exception:
            raise _AttemptFailure(
                "unexpected_error",
                "unexpected provider transport failure",
                retryable=False,
            ) from None

        try:
            status = getattr(response, "status", getattr(response, "code", 200))
            if isinstance(status, bool) or not isinstance(status, int):
                raise _AttemptFailure(
                    "invalid_response",
                    "provider returned an invalid response",
                    retryable=True,
                )
            headers = _headers(getattr(response, "headers", {}))
            if status >= 400:
                raise _AttemptFailure(
                    f"http_{status}",
                    "provider returned an HTTP error",
                    retryable=status in _RETRYABLE_HTTP,
                    http_status=status,
                    headers=headers,
                )
            body = response.read(self.max_response_bytes + 1)
            if not isinstance(body, bytes) or len(body) > self.max_response_bytes:
                raise _AttemptFailure(
                    "invalid_response",
                    "provider returned an invalid response",
                    retryable=True,
                    http_status=status,
                    headers=headers,
                )
            return _WireResponse(status=status, headers=headers, body=body)
        except _AttemptFailure:
            raise
        except Exception:
            raise _AttemptFailure(
                "invalid_response",
                "provider returned an invalid response",
                retryable=True,
            ) from None
        finally:
            try:
                response.close()
            except Exception:
                pass

    def _retry_delay(self, attempt_number: int, headers: Mapping[str, str]) -> float:
        retry_after = _retry_after_seconds(headers)
        base = (
            retry_after
            if retry_after is not None
            else self.base_backoff_seconds * (2 ** max(0, attempt_number - 1))
        )
        base = min(self.max_backoff_seconds, max(0.0, base))
        try:
            sample = float(self.random_fn())
        except Exception:
            sample = 0.0
        if not math.isfinite(sample):
            sample = 0.0
        sample = min(1.0, max(0.0, sample))
        jitter = sample * min(self.jitter_seconds, max(0.0, self.max_backoff_seconds - base))
        return min(self.max_backoff_seconds, base + jitter)

    def _emit(self, record: CallAttemptRecord) -> None:
        append_jsonl(self.event_log_path, record)

    @staticmethod
    def _raise_if_overrun(result: ReconciliationResult) -> None:
        if result.over_hard_cap or result.over_operational_cap:
            raise BudgetOverrun(result)


__all__ = [
    "TransportError",
    "MissingCredentials",
    "RequestValidationError",
    "JsonSchemaTool",
    "JsonSchemaOutput",
    "ToolCall",
    "CompletionResult",
    "Transport",
]
