"""Tiny paid connectivity/tool smoke for the frozen five-model slate.

This is not scientific data. It validates the real wire formats, returned usage,
tool parsing, and shared spend ledger before benchmark trajectories are allowed.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from experiments12.cli12 import DEFAULT_ARTIFACTS, REPOSITORY_ROOT, _environment
from experiments12.core.artifacts import atomic_write_json
from experiments12.core.budget import BudgetLedger
from experiments12.core.transport import JsonSchemaTool, Transport, TransportError
from experiments12.models12 import TARGET_MODEL_NAMES
from experiments12.spec12 import OPERATIONAL_PROVIDER_USD


EFFORT = {
    "gpt-oss-120b": "low",
    "deepseek-v4-flash-0731": "none",
    "qwen3p7-plus": "none",
    "gpt-5.6-luna": "low",
    "gpt-5.6-terra": "low",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _token_bound(messages: list[dict[str, str]], schema: dict[str, Any]) -> int:
    # A byte/token ceiling is deliberately much more conservative than normal
    # English tokenization and keeps pre-dispatch reservations safe.
    return 256 + sum(len(item["content"].encode("utf-8")) for item in messages) + len(
        repr(schema).encode("utf-8")
    )


async def run(args: argparse.Namespace) -> int:
    if not args.yes_spend:
        raise ValueError("live smoke requires --yes-spend")
    models = TARGET_MODEL_NAMES if args.models == "default" else tuple(args.models.split(","))
    unknown = sorted(set(models) - set(TARGET_MODEL_NAMES))
    if unknown:
        raise ValueError(f"smoke supports target slate only: {unknown}")
    output_dir = Path(args.artifacts) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(
        Path(args.artifacts) / "_global_budget.sqlite3",
        operational_caps_usd={
            provider: Decimal(str(value)) for provider, value in OPERATIONAL_PROVIDER_USD.items()
        },
    )
    transport = Transport(
        ledger,
        output_dir / "call_attempts.jsonl",
        environ=_environment(args.env_file),
        max_attempts=3,
        timeout_seconds=120,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }
    tool = JsonSchemaTool.from_schema("record_number", "Record one integer.", schema)
    messages = [
        {"role": "system", "content": "Follow the request and use the supplied tool."},
        {"role": "user", "content": "Call record_number with n=7. Do not answer in text."},
    ]
    records: list[dict[str, Any]] = []
    for model in models:
        started = _now()
        try:
            result = await transport.complete(
                model,
                messages,
                purpose="connectivity_smoke",
                request_key=f"{args.run_id}/{model}/native-tool",
                input_token_estimate=_token_bound(messages, schema),
                max_output_tokens=128,
                reasoning_effort=EFFORT[model],
                tools=(tool,),
                tool_choice="record_number",
            )
            valid = (
                len(result.tool_calls) == 1
                and result.tool_calls[0].name == "record_number"
                and result.tool_calls[0].arguments == {"n": 7}
            )
            records.append(
                {
                    "model": model,
                    "started_at": started,
                    "finished_at": _now(),
                    "status": "ok" if valid else "invalid_tool_result",
                    "resolved_model_id": result.model_id,
                    "response_id": result.response_id,
                    "request_id": result.request_id,
                    "finish_reason": result.finish_reason,
                    "tool_calls": [
                        {"name": call.name, "arguments": call.arguments}
                        for call in result.tool_calls
                    ],
                    "usage": {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "cached_input_tokens": result.usage.cached_input_tokens,
                        "reasoning_tokens": result.usage.reasoning_tokens,
                    },
                    "accounted_cost_usd": str(result.cost_usd),
                    "attempts": len(result.attempts),
                }
            )
        except TransportError as exc:
            records.append(
                {
                    "model": model,
                    "started_at": started,
                    "finished_at": _now(),
                    "status": "transport_error",
                    "category": exc.category,
                    "http_status": exc.http_status,
                    "attempts": len(exc.attempts),
                }
            )
    snapshots = ledger.snapshot()
    payload = {
        "run_id": args.run_id,
        "scientific_data": False,
        "purpose": "provider native-tool connectivity only",
        "records": records,
        "budget": {
            provider: {
                "spent_usd": str(value.spent_usd),
                "reserved_usd": str(value.reserved_usd),
                "remaining_operational_usd": str(value.remaining_operational_usd),
                "remaining_hard_usd": str(value.remaining_hard_usd),
            }
            for provider, value in snapshots.items()
        },
    }
    atomic_write_json(output_dir / "smoke_results.json", payload)
    for record in records:
        print(f"{record['model']}: {record['status']}")
    print(f"results: {output_dir / 'smoke_results.json'}")
    return 0 if all(record["status"] == "ok" for record in records) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes-spend", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", default="default")
    parser.add_argument("--env-file", default=str(REPOSITORY_ROOT / ".env"))
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

