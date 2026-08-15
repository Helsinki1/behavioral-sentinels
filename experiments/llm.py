"""Thin async chat-completion client with retries, shared by runner and judge."""
import asyncio
import os
import random

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

_clients = {}


def get_client(cfg):
    key = cfg["base_url"]
    if key not in _clients:
        _clients[key] = AsyncOpenAI(
            base_url=cfg["base_url"],
            api_key=os.environ[cfg["api_key_env"]],
            timeout=120.0,
            max_retries=0,
        )
    return _clients[key]


async def chat(cfg, messages, max_tokens=None, temperature=0.2):
    """Returns (content, usage_dict). Retries on transient errors."""
    client = get_client(cfg)
    mt = max_tokens or cfg["max_tokens"]
    last_err = None
    for attempt in range(12):
        try:
            resp = await client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                max_tokens=mt,
                temperature=temperature,
                extra_body=cfg.get("extra") or {},
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if not content.strip() and choice.finish_reason == "length" and mt < 3000:
                mt = 3000  # reasoning ate the budget; retry once with more room
                continue
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            }
            return content, usage
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            last_err = e
            await asyncio.sleep(min(90, 2 ** attempt) + random.random() * 5)
        except APIError as e:
            last_err = e
            if getattr(e, "status_code", 500) and e.status_code < 500 and e.status_code != 429:
                raise
            await asyncio.sleep(min(90, 2 ** attempt) + random.random() * 5)
    raise RuntimeError(f"chat failed after retries: {last_err}")
