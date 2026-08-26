"""Deterministic, equal-output active (carried) probes for Experiment 12.

These are *active* observations: at each checkpoint the probe is a new user
message, the target agent produces a new assistant answer, and both messages
remain in the continuing task history.  They are not disposable zero-carry
forks.  The four variants isolate useful components while sharing one exact
15-character answer envelope (``PROBE: XXXXXXXX``):

* ``current_copy`` — **say my name**, with the name/code in the current prompt;
* ``initial_recall`` — **say my name**, but the code appeared only initially;
* ``stateful_counter`` — **increment this number** from carried state; and
* ``recompute`` — **recompute** fresh current-turn arithmetic.

All task/checkpoint material is derived from SHA-256, not process-global random
state, so generation is stable across order, process, and Python version.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Any, Mapping, MutableSequence, Sequence


CURRENT_COPY = "current_copy"
INITIAL_RECALL = "initial_recall"
STATEFUL_COUNTER = "stateful_counter"
RECOMPUTE = "recompute"
PROBE_ORDER = (CURRENT_COPY, INITIAL_RECALL, STATEFUL_COUNTER, RECOMPUTE)

PAYLOAD_WIDTH = 8
ANSWER_PREFIX = "PROBE: "
FIXED_OUTPUT_LENGTH = len(ANSWER_PREFIX) + PAYLOAD_WIDTH
MAX_CHECKPOINT_INDEX = 1_000_000
_PROTOCOL_SALT = "behavioral-sentinels/experiment-12/active-probes/v1"
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_PAYLOAD_RE = re.compile(r"^[A-Z0-9]{8}$")
_ANSWER_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?`?PROBE[ \t]*:[ \t]*`?([A-Z0-9]{8})`?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ProbeComponents:
    """Predeclared experimental components; loads use a 0–2 ordinal scale.

    ``copyability`` means whether the *exact current answer* is visible in the
    checkpoint prompt.  ``copy_source`` records when useful source state is
    instead available in the initial instruction or prior carried answer.
    """

    copyability: int
    memory_load: int
    reasoning_load: int
    output_length: int
    copy_source: str

    def __post_init__(self) -> None:
        for name in ("copyability", "memory_load", "reasoning_load"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2:
                raise ValueError(f"{name} must be an integer from 0 through 2")
        if self.output_length != FIXED_OUTPUT_LENGTH:
            raise ValueError(f"output_length must be exactly {FIXED_OUTPUT_LENGTH}")
        if not self.copy_source:
            raise ValueError("copy_source must be non-empty")


@dataclass(frozen=True, slots=True)
class ProbeDefinition:
    key: str
    paper_term: str
    label: str
    components: ProbeComponents
    requires_initial_instruction: bool


@dataclass(frozen=True, slots=True)
class ProbeInstance:
    """Ground truth for one target task and one 1-based probe checkpoint."""

    variant: str
    task_id: str
    checkpoint_index: int
    expected_answer: str
    parameters: tuple[tuple[str, int | str], ...]

    def __post_init__(self) -> None:
        if self.variant not in PROBE_DEFINITIONS:
            raise ValueError(f"unknown probe variant: {self.variant!r}")
        _validate_task_checkpoint(self.task_id, self.checkpoint_index)
        _validate_answer(self.expected_answer)
        if not isinstance(self.parameters, tuple):
            raise ValueError("parameters must be an immutable tuple")

    @property
    def payload(self) -> str:
        return self.expected_answer[len(ANSWER_PREFIX) :]

    @property
    def parameter_map(self) -> Mapping[str, int | str]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True)
class ProbeGrade:
    passed: bool
    expected_answer: str
    extracted_answer: str | None
    value_correct: bool
    exact_format: bool
    observed_length: int
    error: str | None


PROBE_DEFINITIONS: Mapping[str, ProbeDefinition] = MappingProxyType(
    {
        CURRENT_COPY: ProbeDefinition(
            key=CURRENT_COPY,
            paper_term="say my name",
            label="say my name — current-turn name/code copy",
            components=ProbeComponents(
                copyability=2,
                memory_load=0,
                reasoning_load=0,
                output_length=FIXED_OUTPUT_LENGTH,
                copy_source="current checkpoint prompt",
            ),
            requires_initial_instruction=False,
        ),
        INITIAL_RECALL: ProbeDefinition(
            key=INITIAL_RECALL,
            paper_term="say my name",
            label="say my name — initial-only name/code recall",
            components=ProbeComponents(
                copyability=0,
                memory_load=2,
                reasoning_load=0,
                output_length=FIXED_OUTPUT_LENGTH,
                copy_source="initial instruction only",
            ),
            requires_initial_instruction=True,
        ),
        STATEFUL_COUNTER: ProbeDefinition(
            key=STATEFUL_COUNTER,
            paper_term="increment this number",
            label="increment this number — stateful carried counter",
            components=ProbeComponents(
                copyability=0,
                memory_load=1,
                reasoning_load=1,
                output_length=FIXED_OUTPUT_LENGTH,
                copy_source="prior carried answer supplies source state",
            ),
            requires_initial_instruction=True,
        ),
        RECOMPUTE: ProbeDefinition(
            key=RECOMPUTE,
            paper_term="recompute",
            label="recompute — current-turn arithmetic",
            components=ProbeComponents(
                copyability=0,
                memory_load=0,
                reasoning_load=2,
                output_length=FIXED_OUTPUT_LENGTH,
                copy_source="no answer is available to copy",
            ),
            requires_initial_instruction=False,
        ),
    }
)


def _validate_task_checkpoint(task_id: str, checkpoint_index: int) -> None:
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if (
        isinstance(checkpoint_index, bool)
        or not isinstance(checkpoint_index, int)
        or not 1 <= checkpoint_index <= MAX_CHECKPOINT_INDEX
    ):
        raise ValueError(
            f"checkpoint_index must be an integer from 1 through {MAX_CHECKPOINT_INDEX}"
        )


def _digest(variant: str, task_id: str, checkpoint_index: int | str) -> bytes:
    material = f"{_PROTOCOL_SALT}\0{variant}\0{task_id}\0{checkpoint_index}"
    return hashlib.sha256(material.encode("utf-8")).digest()


def _code(digest: bytes) -> str:
    return "".join(_CODE_ALPHABET[value % len(_CODE_ALPHABET)] for value in digest[:8])


def _initial_recall_code(task_id: str, checkpoint_index: int) -> str:
    return _code(_digest(INITIAL_RECALL, task_id, checkpoint_index))


def _counter_start(task_id: str) -> int:
    raw = int.from_bytes(_digest(STATEFUL_COUNTER, task_id, "initial")[:4], "big")
    return 10_000_000 + raw % 10_000_000


def _answer(payload: str) -> str:
    payload = payload.upper()
    if not _PAYLOAD_RE.fullmatch(payload):
        raise ValueError("probe payload must be exactly eight uppercase alphanumeric characters")
    answer = ANSWER_PREFIX + payload
    _validate_answer(answer)
    return answer


def _validate_answer(answer: str) -> None:
    if (
        not isinstance(answer, str)
        or len(answer) != FIXED_OUTPUT_LENGTH
        or not answer.startswith(ANSWER_PREFIX)
        or not _PAYLOAD_RE.fullmatch(answer[len(ANSWER_PREFIX) :])
    ):
        raise ValueError(
            f"probe answer must match 'PROBE: XXXXXXXX' in exactly {FIXED_OUTPUT_LENGTH} characters"
        )


def generate_probe_instance(
    variant: str,
    task_id: str,
    checkpoint_index: int,
) -> ProbeInstance:
    """Generate one order-independent probe instance and its exact answer."""

    if variant not in PROBE_DEFINITIONS:
        raise KeyError(f"unknown probe variant: {variant}")
    _validate_task_checkpoint(task_id, checkpoint_index)

    if variant == CURRENT_COPY:
        code = _code(_digest(variant, task_id, checkpoint_index))
        payload = code
        parameters: tuple[tuple[str, int | str], ...] = (("code", code),)
    elif variant == INITIAL_RECALL:
        code = _initial_recall_code(task_id, checkpoint_index)
        payload = code
        parameters = (("initial_code", code),)
    elif variant == STATEFUL_COUNTER:
        start = _counter_start(task_id)
        value = start + checkpoint_index
        if value > 99_999_999:  # Defensive; the declared checkpoint cap is far lower.
            raise ValueError("counter no longer fits the fixed eight-character payload")
        payload = f"{value:08d}"
        parameters = (("counter_start", start), ("counter_value", value))
    else:
        digest = _digest(variant, task_id, checkpoint_index)
        left = 1_000 + int.from_bytes(digest[0:2], "big") % 9_000
        right = 1_000 + int.from_bytes(digest[2:4], "big") % 9_000
        addend = int.from_bytes(digest[4:6], "big") % 10_000
        value = left * right + addend
        payload = f"{value:08d}"
        parameters = (
            ("addend", addend),
            ("left", left),
            ("result", value),
            ("right", right),
        )

    return ProbeInstance(
        variant=variant,
        task_id=task_id,
        checkpoint_index=checkpoint_index,
        expected_answer=_answer(payload),
        parameters=parameters,
    )


def render_initial_instruction(
    variant: str,
    task_id: str,
    checkpoint_indices: Sequence[int] | None = None,
) -> str | None:
    """Return setup that belongs in the initial task message, if required.

    The recall arm requires the trajectory's preregistered checkpoint indices
    so its full codebook can appear initially and nowhere later.  Each code is
    used once; a previous carried answer therefore cannot reveal a later one.
    """

    if variant not in PROBE_DEFINITIONS:
        raise KeyError(f"unknown probe variant: {variant}")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    if variant == INITIAL_RECALL:
        if (
            checkpoint_indices is None
            or isinstance(checkpoint_indices, (str, bytes))
            or not checkpoint_indices
        ):
            raise ValueError(
                "initial_recall requires the non-empty preregistered checkpoint_indices"
            )
        checkpoints = tuple(checkpoint_indices)
        for checkpoint_index in checkpoints:
            _validate_task_checkpoint(task_id, checkpoint_index)
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("checkpoint_indices must not contain duplicates")
        codebook = "\n".join(
            f"- checkpoint {checkpoint_index}: "
            f"{_initial_recall_code(task_id, checkpoint_index)}"
            for checkpoint_index in sorted(checkpoints)
        )
        return (
            "ACTIVE CARRIED PROBE — say my name (initial-only name/code recall). "
            "Memorize the checkpoint-specific eight-character session name/codes in "
            "this codebook. At each later active-probe checkpoint, return the code for "
            "that checkpoint in the required PROBE line. Each code is used once and "
            "will not be repeated in a checkpoint prompt.\n" + codebook
        )
    if variant == STATEFUL_COUNTER:
        start = _counter_start(task_id)
        return (
            "ACTIVE CARRIED PROBE — increment this number. "
            f"The counter starts at {start:08d}. At each active-probe checkpoint, "
            "increment the last carried counter value by exactly one and return the "
            "new eight-digit value in the required PROBE line."
        )
    return None


def render_probe_prompt(instance: ProbeInstance) -> str:
    """Render the separate user message that actively intervenes in history."""

    if not isinstance(instance, ProbeInstance):
        raise TypeError("instance must be a ProbeInstance")
    parameters = instance.parameter_map
    common = (
        "Reply with exactly one line and nothing else. The line must contain "
        f"exactly {FIXED_OUTPUT_LENGTH} characters: `PROBE: ` followed by exactly "
        "eight uppercase letters or digits (zero-pad numbers to eight digits)."
    )
    if instance.variant == CURRENT_COPY:
        task = (
            "ACTIVE CARRIED PROBE — say my name (current-turn name/code copy). "
            f"The current name/code is `{parameters['code']}`. Copy that code."
        )
    elif instance.variant == INITIAL_RECALL:
        task = (
            "ACTIVE CARRIED PROBE — say my name (initial-only name/code recall). "
            f"Recall the eight-character session name/code for checkpoint "
            f"{instance.checkpoint_index}, supplied only in the initial codebook; it "
            "is deliberately not repeated here."
        )
    elif instance.variant == STATEFUL_COUNTER:
        task = (
            "ACTIVE CARRIED PROBE — increment this number. Increment the last "
            "counter value carried in this conversation by exactly one. Do not reset "
            "it and do not copy the unchanged prior value."
        )
    else:
        task = (
            "ACTIVE CARRIED PROBE — recompute (current-turn arithmetic). Recompute "
            f"({parameters['left']} × {parameters['right']}) + {parameters['addend']}. "
            "Do not show work."
        )
    return task + "\n" + common


def extract_probe_answer(response: str) -> str | None:
    """Extract one unambiguous answer line and return its canonical spelling."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    matches = _ANSWER_LINE_RE.findall(response)
    if len(matches) != 1:
        return None
    return _answer(matches[0])


def grade_probe_response(instance: ProbeInstance, response: str) -> ProbeGrade:
    """Strictly grade value *and* compliance with the fixed-length envelope."""

    if not isinstance(instance, ProbeInstance):
        raise TypeError("instance must be a ProbeInstance")
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if response.endswith("\r\n"):
        visible = response[:-2]
    elif response.endswith("\n"):
        visible = response[:-1]
    else:
        visible = response
    extracted = extract_probe_answer(response)
    value_correct = extracted == instance.expected_answer
    exact_format = visible == instance.expected_answer
    passed = value_correct and exact_format
    if extracted is None:
        error = "missing_or_ambiguous_answer"
    elif not value_correct:
        error = "wrong_value"
    elif not exact_format:
        error = "extra_or_noncanonical_output"
    else:
        error = None
    return ProbeGrade(
        passed=passed,
        expected_answer=instance.expected_answer,
        extracted_answer=extracted,
        value_correct=value_correct,
        exact_format=exact_format,
        observed_length=len(visible),
        error=error,
    )


def carried_history_messages(
    instance: ProbeInstance,
    response: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the two messages an *active carried* observation adds to history.

    The returned assistant content is the target model's actual response, even
    when it grades incorrectly.  A passive zero-carry observer must not use this
    function and must discard its fork instead.
    """

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    return (
        {"role": "user", "content": render_probe_prompt(instance)},
        {"role": "assistant", "content": response},
    )


def append_carried_probe_exchange(
    continuing_history: MutableSequence[dict[str, str]],
    instance: ProbeInstance,
    response: str,
) -> ProbeGrade:
    """Append prompt **and answer** to continuing history, then return its grade."""

    messages = carried_history_messages(instance, response)
    continuing_history.extend(messages)
    return grade_probe_response(instance, response)


def component_metadata() -> dict[str, dict[str, Any]]:
    """Return a JSON-ready frozen-design table for manifests and analysis."""

    return {
        key: {
            "paper_term": definition.paper_term,
            "label": definition.label,
            "copyability": definition.components.copyability,
            "copy_source": definition.components.copy_source,
            "memory_load": definition.components.memory_load,
            "reasoning_load": definition.components.reasoning_load,
            "output_length": definition.components.output_length,
            "requires_initial_instruction": definition.requires_initial_instruction,
            "carried": True,
        }
        for key, definition in PROBE_DEFINITIONS.items()
    }


__all__ = [
    "ANSWER_PREFIX",
    "CURRENT_COPY",
    "FIXED_OUTPUT_LENGTH",
    "INITIAL_RECALL",
    "MAX_CHECKPOINT_INDEX",
    "PAYLOAD_WIDTH",
    "PROBE_DEFINITIONS",
    "PROBE_ORDER",
    "RECOMPUTE",
    "STATEFUL_COUNTER",
    "ProbeComponents",
    "ProbeDefinition",
    "ProbeGrade",
    "ProbeInstance",
    "append_carried_probe_exchange",
    "carried_history_messages",
    "component_metadata",
    "extract_probe_answer",
    "generate_probe_instance",
    "grade_probe_response",
    "render_initial_instruction",
    "render_probe_prompt",
]
