"""Shared answer-blind boundaries for passive monitors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import sha256_json


FORBIDDEN_OBSERVER_KEYS = frozenset(
    {
        "answer",
        "answers",
        "change_plan",
        "correct",
        "future",
        "gold",
        "ground_truth",
        "label",
        "possible_answer",
        "success",
        "target",
    }
)


def assert_answer_blind(metadata: Mapping[str, Any] | None) -> None:
    """Reject likely outcome/future fields before they reach a monitor worker."""

    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise TypeError("observer metadata must be a mapping")
    forbidden = sorted(str(key) for key in metadata if str(key).lower() in FORBIDDEN_OBSERVER_KEYS)
    if forbidden:
        raise ValueError(f"observer metadata contains forbidden fields: {forbidden}")


def frozen_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep copy so monitor calls cannot mutate target history."""

    if isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of mappings")
    copied = deepcopy(list(messages))
    for index, message in enumerate(copied):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be a mapping")
        if not isinstance(message.get("role"), str):
            raise ValueError(f"message {index} has no string role")
    return copied


def monitor_spec_hash(name: str, version: int, payload: Mapping[str, Any]) -> str:
    if not name or version < 1:
        raise ValueError("monitor name and positive version are required")
    return sha256_json({"name": name, "version": version, "spec": dict(payload)})

