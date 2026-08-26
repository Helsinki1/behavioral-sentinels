"""Exact active-probe calls made on a disposable, zero-carry prefix copy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from experiments12.probes12 import (
    CURRENT_COPY,
    RECOMPUTE,
    ProbeInstance,
    render_probe_prompt,
)

from .base import assert_answer_blind, frozen_messages, monitor_spec_hash


FROZEN_PROBE_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrozenProbeFork:
    messages: tuple[dict[str, Any], ...]
    spec_sha256: str
    variant: str
    checkpoint_index: int


def build_frozen_probe_fork(
    messages: Sequence[Mapping[str, Any]],
    instance: ProbeInstance,
    *,
    observer_metadata: Mapping[str, Any] | None = None,
) -> FrozenProbeFork:
    """Build the matched zero-carry call; caller must discard its response."""

    assert_answer_blind(observer_metadata)
    if not isinstance(instance, ProbeInstance):
        raise TypeError("instance must be a ProbeInstance")
    if instance.variant not in {CURRENT_COPY, RECOMPUTE}:
        raise ValueError(
            "an exact zero-carry match is only valid for stateless current_copy/recompute probes"
        )
    fork = frozen_messages(messages)
    prompt = render_probe_prompt(instance)
    fork.append({"role": "user", "content": prompt})
    spec = monitor_spec_hash(
        "frozen_probe",
        FROZEN_PROBE_VERSION,
        {"variant": instance.variant, "prompt": prompt},
    )
    return FrozenProbeFork(
        messages=tuple(fork),
        spec_sha256=spec,
        variant=instance.variant,
        checkpoint_index=instance.checkpoint_index,
    )
