"""Pure, fail-closed intervention contracts for Experiment 12 deployment.

The module makes no model or environment calls.  It freezes a target-visible
prefix, verifies a score-independent matched/yoked checkpoint schedule, applies
exactly one declared intervention, and returns a fresh continued history plus
content hashes for an append-only provenance event.

``compact`` is a deterministic, bounded, lossy extract of the visible prefix.
``reground`` reconstructs frozen initial instructions and embeds an exact
canonical public-state snapshot.  ``feedback`` retains the complete prefix and
appends only GOOD/BAD/WATCH classifications backed by exact prefix quotes.
``none`` changes no content but still records its matched pseudo-intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

from experiments12.core.artifacts import canonical_json_bytes, sha256_json
from experiments12.domains.base import validate_sha256
from experiments12.spec12 import FEEDBACK_MAX_TOKENS


OPERATORS_VERSION = 1
DEFAULT_COMPACTION_MESSAGES = 4
DEFAULT_COMPACTION_EXCERPT_BYTES = 240
DEFAULT_COMPACTION_SUMMARY_BYTES = 1_600

COMPACTION_CONTRACT = """Clear the prior trace. Continue from the frozen initial
instructions and a deterministic lossy extract of only the completed visible
prefix. The extract is context, not a claim of completeness or correctness."""

REGROUND_CONTRACT = """Clear the prior trace. Continue from the frozen initial
instructions and the exact canonical public environment state available after
this checkpoint. Do not add private, gold, evaluator, hidden, or future state."""

FEEDBACK_CONTRACT = """Keep the trace. Append at most 80 conservatively counted
tokens under GOOD, BAD, and WATCH. Every nonempty item must be an exact quote
from the visible prefix; do not add a score, answer, prediction, or hidden fact."""

_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_INSTRUCTION_ROLES = frozenset({"system", "developer", "user"})
_SENSITIVE_KEYS = frozenset(
    {
        "private",
        "private_state",
        "secret",
        "secret_state",
        "hidden",
        "hidden_state",
        "gold",
        "gold_answer",
        "gold_label",
        "ground_truth",
        "ground_truth_answer",
        "ground_truth_label",
        "correct_answer",
        "reference_answer",
        "label",
        "correctness",
        "evaluation",
        "evaluation_label",
        "evaluation_result",
        "evaluator",
        "evaluator_result",
        "evaluator_state",
        "official_score",
        "official_success",
        "task_score",
        "oracle",
        "oracle_answer",
        "future",
        "future_state",
        "future_turn",
        "future_turns",
        "next_user_message",
        "remaining_turns",
        "unobserved_turns",
    }
)
_SENSITIVE_PREFIXES = (
    "private_",
    "secret_",
    "hidden_",
    "gold_",
    "ground_truth_",
    "evaluation_",
    "evaluator_",
    "official_",
    "oracle_",
    "future_",
    "unobserved_",
)
_SENSITIVE_COMPACT_KEYS = frozenset(
    "".join(character for character in key if character.isalnum())
    for key in _SENSITIVE_KEYS
)


class InterventionValidationError(ValueError):
    """An intervention input is invalid, contaminated, or internally inconsistent."""


class ContaminationError(InterventionValidationError):
    """Evaluator-only, private, or future material crossed a public boundary."""


class ScheduleMismatchError(InterventionValidationError):
    """A cell or checkpoint is not covered by its frozen matched/yoked schedule."""


class InterventionType(str, Enum):
    NONE = "none"
    COMPACT = "compact"
    REGROUND = "reground"
    FEEDBACK = "feedback"


class ScheduleMode(str, Enum):
    MATCHED = "matched"
    YOKED = "yoked"


def _identifier(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise InterventionValidationError(f"{name} must be a bounded single-line string")
    return value


def _positive_integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise InterventionValidationError(f"{name} must be a {qualifier} integer")
    return value


def _sha256(name: str, value: Any) -> str:
    try:
        return validate_sha256(name, value)
    except ValueError as exc:
        raise InterventionValidationError(str(exc)) from exc


def _canonical_text(name: str, value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InterventionValidationError(f"{name} must be finite canonical JSON data") from exc


def _exact_keys(name: str, value: Any, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise InterventionValidationError(f"{name} must be an object")
    actual = set(value)
    if actual != allowed:
        raise ContaminationError(
            f"{name} has unexpected fields; missing={sorted(allowed - actual)}, "
            f"unknown={sorted(actual - allowed)}"
        )
    return value


def _normalize_tool_call(value: Any, where: str) -> dict[str, Any]:
    item = _exact_keys(where, value, {"id", "type", "function"})
    if item["type"] != "function":
        raise InterventionValidationError(f"{where}.type must be function")
    call_id = _identifier(f"{where}.id", item["id"])
    function = _exact_keys(f"{where}.function", item["function"], {"name", "arguments"})
    name = _identifier(f"{where}.function.name", function["name"])
    arguments = function["arguments"]
    if isinstance(arguments, str):
        try:
            arguments_value = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise InterventionValidationError(f"{where} arguments are not valid JSON") from exc
    else:
        arguments_value = arguments
    if not isinstance(arguments_value, Mapping):
        raise InterventionValidationError(f"{where} arguments must encode an object")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _canonical_text(f"{where} arguments", arguments_value),
        },
    }


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    instructions_only: bool,
) -> list[dict[str, Any]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise InterventionValidationError("messages must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        if not isinstance(raw, Mapping):
            raise InterventionValidationError(f"message {index} must be an object")
        role = raw.get("role")
        if role not in (_INSTRUCTION_ROLES if instructions_only else _MESSAGE_ROLES):
            raise InterventionValidationError(f"message {index} role is invalid")
        if role in {"system", "developer", "user"}:
            item = _exact_keys(f"message {index}", raw, {"role", "content"})
        elif role == "tool":
            item = _exact_keys(
                f"message {index}", raw, {"role", "content", "tool_call_id"}
            )
        else:
            allowed = {"role", "content"}
            if "tool_calls" in raw:
                allowed.add("tool_calls")
            item = _exact_keys(f"message {index}", raw, allowed)
        content = item["content"]
        if not isinstance(content, str):
            raise InterventionValidationError(f"message {index} content must be text")
        copied: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            copied["tool_call_id"] = _identifier(
                f"message {index}.tool_call_id", item["tool_call_id"]
            )
        if role == "assistant" and "tool_calls" in item:
            raw_calls = item["tool_calls"]
            if not isinstance(raw_calls, list):
                raise InterventionValidationError(f"message {index}.tool_calls must be an array")
            copied["tool_calls"] = [
                _normalize_tool_call(call, f"message {index}.tool_calls[{call_index}]")
                for call_index, call in enumerate(raw_calls)
            ]
        normalized.append(copied)
    if instructions_only:
        user_positions = [
            index for index, message in enumerate(normalized) if message["role"] == "user"
        ]
        if len(user_positions) > 1 or (user_positions and user_positions[0] != len(normalized) - 1):
            raise ContaminationError(
                "initial instructions may contain at most one final initial user message"
            )
    return normalized


def _normalize_sensitive_key(key: str) -> str:
    normalized = "_".join(part for part in "".join(
        character.lower() if character.isalnum() else " " for character in key
    ).split())
    return normalized


def _assert_public_data(value: Any, path: str = "public_state") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContaminationError(f"{path} contains a non-string key")
            normalized = _normalize_sensitive_key(key)
            compact = "".join(character for character in normalized if character.isalnum())
            if (
                normalized in _SENSITIVE_KEYS
                or normalized.startswith(_SENSITIVE_PREFIXES)
                or compact in _SENSITIVE_COMPACT_KEYS
            ):
                raise ContaminationError(f"{path} contains forbidden field {key!r}")
            _assert_public_data(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_data(item, f"{path}[{index}]")
    elif value is None or isinstance(value, (bool, int, float, str)):
        _canonical_text(path, value)
    else:
        raise InterventionValidationError(f"{path} is not JSON data")


@dataclass(frozen=True, slots=True)
class VisiblePrefix:
    """A frozen target-visible history ending at one completed task turn."""

    domain: str
    task_id: str
    after_turn: int
    messages_json: str
    prefix_sha256: str

    def __post_init__(self) -> None:
        _identifier("domain", self.domain)
        _identifier("task_id", self.task_id)
        _positive_integer("after_turn", self.after_turn)
        try:
            messages = json.loads(self.messages_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InterventionValidationError("messages_json is invalid") from exc
        normalized = _normalize_messages(messages, instructions_only=False)
        if _canonical_text("visible prefix", normalized) != self.messages_json:
            raise InterventionValidationError("messages_json must use normalized canonical JSON")
        _sha256("prefix_sha256", self.prefix_sha256)
        if sha256_json(normalized) != self.prefix_sha256:
            raise InterventionValidationError("visible prefix does not match prefix_sha256")

    @property
    def messages(self) -> list[dict[str, Any]]:
        return json.loads(self.messages_json)


def freeze_visible_prefix(
    *,
    domain: str,
    task_id: str,
    after_turn: int,
    messages: Sequence[Mapping[str, Any]],
) -> VisiblePrefix:
    normalized = _normalize_messages(messages, instructions_only=False)
    return VisiblePrefix(
        domain=_identifier("domain", domain),
        task_id=_identifier("task_id", task_id),
        after_turn=_positive_integer("after_turn", after_turn),
        messages_json=_canonical_text("visible prefix", normalized),
        prefix_sha256=sha256_json(normalized),
    )


@dataclass(frozen=True, slots=True)
class InitialInstructions:
    """Instructions frozen before any target response or future benchmark turn."""

    domain: str
    task_id: str
    messages_json: str
    instructions_sha256: str

    def __post_init__(self) -> None:
        _identifier("domain", self.domain)
        _identifier("task_id", self.task_id)
        try:
            messages = json.loads(self.messages_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InterventionValidationError("instruction messages_json is invalid") from exc
        normalized = _normalize_messages(messages, instructions_only=True)
        if _canonical_text("initial instructions", normalized) != self.messages_json:
            raise InterventionValidationError(
                "instruction messages_json must use normalized canonical JSON"
            )
        _sha256("instructions_sha256", self.instructions_sha256)
        if sha256_json(normalized) != self.instructions_sha256:
            raise InterventionValidationError("initial instructions do not match their hash")

    @property
    def messages(self) -> list[dict[str, str]]:
        return json.loads(self.messages_json)


def freeze_initial_instructions(
    *,
    domain: str,
    task_id: str,
    messages: Sequence[Mapping[str, Any]],
) -> InitialInstructions:
    normalized = _normalize_messages(messages, instructions_only=True)
    return InitialInstructions(
        domain=_identifier("domain", domain),
        task_id=_identifier("task_id", task_id),
        messages_json=_canonical_text("initial instructions", normalized),
        instructions_sha256=sha256_json(normalized),
    )


@dataclass(frozen=True, slots=True)
class PublicStateSnapshot:
    """Exact canonical public environment state at an actionable checkpoint."""

    domain: str
    task_id: str
    after_turn: int
    state_json: str
    state_sha256: str

    def __post_init__(self) -> None:
        _identifier("domain", self.domain)
        _identifier("task_id", self.task_id)
        _positive_integer("after_turn", self.after_turn)
        try:
            state = json.loads(self.state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InterventionValidationError("state_json is invalid") from exc
        if not isinstance(state, dict):
            raise InterventionValidationError("public state must be a JSON object")
        _assert_public_data(state)
        if _canonical_text("public state", state) != self.state_json:
            raise InterventionValidationError("state_json must use canonical JSON")
        _sha256("state_sha256", self.state_sha256)
        if sha256_json(state) != self.state_sha256:
            raise InterventionValidationError("public state does not match state_sha256")

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(self.state_json)


def freeze_public_state(
    *,
    domain: str,
    task_id: str,
    after_turn: int,
    state: Mapping[str, Any],
) -> PublicStateSnapshot:
    if not isinstance(state, Mapping):
        raise InterventionValidationError("public state must be an object")
    copied = json.loads(_canonical_text("public state", state))
    _assert_public_data(copied)
    return PublicStateSnapshot(
        domain=_identifier("domain", domain),
        task_id=_identifier("task_id", task_id),
        after_turn=_positive_integer("after_turn", after_turn),
        state_json=_canonical_text("public state", copied),
        state_sha256=sha256_json(copied),
    )


@dataclass(frozen=True, slots=True)
class SignalReference:
    """Hash-only monitor provenance for same-prefix or frozen two-pass use.

    A same-pass signal must name the exact target prefix.  A deployment signal
    may instead come from an answer-blind pass-one rerun; in that case
    ``frozen_two_pass`` is true and ``schedule_sha256`` binds the signal record
    to the immutable schedule that selected it.  This avoids falsely claiming
    that a pass-one signal was computed from the stochastic pass-two prefix.
    """

    method: str
    checkpoint: int
    source_prefix_sha256: str
    signal_record_sha256: str
    schedule_sha256: str | None = None
    frozen_two_pass: bool = False

    def __post_init__(self) -> None:
        _identifier("signal method", self.method)
        _positive_integer("signal checkpoint", self.checkpoint)
        _sha256("signal source_prefix_sha256", self.source_prefix_sha256)
        _sha256("signal_record_sha256", self.signal_record_sha256)
        if not isinstance(self.frozen_two_pass, bool):
            raise InterventionValidationError("frozen_two_pass must be boolean")
        if self.schedule_sha256 is not None:
            _sha256("signal schedule_sha256", self.schedule_sha256)
        if self.frozen_two_pass and self.schedule_sha256 is None:
            raise InterventionValidationError(
                "a frozen two-pass signal must bind its deployment schedule"
            )
        if not self.frozen_two_pass and self.schedule_sha256 is not None:
            raise InterventionValidationError(
                "a same-pass signal cannot claim a deployment schedule"
            )


def conservative_token_upper_bound(text: str) -> int:
    """One UTF-8 byte per token: deliberately conservative and tokenizer-free."""

    if not isinstance(text, str):
        raise InterventionValidationError("token-counted value must be text")
    return len(text.encode("utf-8"))


def _validate_quote(quote: Any) -> str:
    if (
        not isinstance(quote, str)
        or not quote.strip()
        or len(quote) > 160
        or any(character in quote for character in ("\x00", "\r", "\n"))
    ):
        raise InterventionValidationError("feedback evidence must be single-line text <=160 chars")
    return quote


@dataclass(frozen=True, slots=True)
class FeedbackNote:
    """GOOD/BAD/WATCH classifications containing only exact visible quotes."""

    source_prefix_sha256: str
    good: tuple[str, ...] = ()
    bad: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _sha256("source_prefix_sha256", self.source_prefix_sha256)
        all_quotes: list[str] = []
        for name in ("good", "bad", "watch"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise InterventionValidationError(f"feedback {name} must be a tuple")
            all_quotes.extend(_validate_quote(quote) for quote in values)
        if not all_quotes:
            raise InterventionValidationError("feedback must contain at least one quoted observation")
        if len(all_quotes) != len(set(all_quotes)):
            raise InterventionValidationError("feedback evidence quotes must be unique")
        if conservative_token_upper_bound(self.render()) > FEEDBACK_MAX_TOKENS:
            raise InterventionValidationError(
                f"feedback exceeds the conservative {FEEDBACK_MAX_TOKENS}-token limit"
            )

    @staticmethod
    def _line(name: str, quotes: tuple[str, ...]) -> str:
        rendered = " | ".join(json.dumps(quote, ensure_ascii=False) for quote in quotes)
        return f"{name}: {rendered if rendered else '—'}"

    def render(self) -> str:
        return "\n".join(
            (
                self._line("GOOD", self.good),
                self._line("BAD", self.bad),
                self._line("WATCH", self.watch),
            )
        )

    @property
    def note_sha256(self) -> str:
        return sha256_json(
            {
                "source_prefix_sha256": self.source_prefix_sha256,
                "good": list(self.good),
                "bad": list(self.bad),
                "watch": list(self.watch),
                "rendered": self.render(),
            }
        )

    def validate_for(self, prefix: VisiblePrefix) -> None:
        if self.source_prefix_sha256 != prefix.prefix_sha256:
            raise ContaminationError("feedback was produced from a different visible prefix")
        evidence_fields: list[str] = []
        for message in prefix.messages:
            evidence_fields.append(message["content"])
            for call in message.get("tool_calls", []):
                evidence_fields.extend(
                    (call["id"], call["function"]["name"], call["function"]["arguments"])
                )
            if "tool_call_id" in message:
                evidence_fields.append(message["tool_call_id"])
        for quote in (*self.good, *self.bad, *self.watch):
            if not any(quote in field for field in evidence_fields):
                raise ContaminationError("feedback quote is absent from the visible prefix")


def make_feedback_note(
    prefix: VisiblePrefix,
    *,
    good: Sequence[str] = (),
    bad: Sequence[str] = (),
    watch: Sequence[str] = (),
) -> FeedbackNote:
    if any(isinstance(value, (str, bytes)) for value in (good, bad, watch)):
        raise InterventionValidationError("feedback sections must be sequences of quotes")
    note = FeedbackNote(
        source_prefix_sha256=prefix.prefix_sha256,
        good=tuple(good),
        bad=tuple(bad),
        watch=tuple(watch),
    )
    note.validate_for(prefix)
    return note


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    keep_last_messages: int = DEFAULT_COMPACTION_MESSAGES
    max_excerpt_bytes: int = DEFAULT_COMPACTION_EXCERPT_BYTES
    max_summary_bytes: int = DEFAULT_COMPACTION_SUMMARY_BYTES

    def __post_init__(self) -> None:
        _positive_integer("keep_last_messages", self.keep_last_messages)
        _positive_integer("max_excerpt_bytes", self.max_excerpt_bytes)
        _positive_integer("max_summary_bytes", self.max_summary_bytes)
        if self.max_excerpt_bytes > self.max_summary_bytes:
            raise InterventionValidationError("max_excerpt_bytes cannot exceed max_summary_bytes")
        if self.max_summary_bytes > 4_096:
            raise InterventionValidationError("lossy compaction summary cannot exceed 4096 bytes")

    @property
    def config_sha256(self) -> str:
        return sha256_json(
            {
                "keep_last_messages": self.keep_last_messages,
                "max_excerpt_bytes": self.max_excerpt_bytes,
                "max_summary_bytes": self.max_summary_bytes,
            }
        )


def _truncate_utf8(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    prefix = encoded[: limit - 3]
    while prefix:
        try:
            return prefix.decode("utf-8") + "..."
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return "..."[:limit]


def build_lossy_compaction(prefix: VisiblePrefix, config: CompactionConfig) -> str:
    """Create a deterministic extract from the prefix and nothing else."""

    if not isinstance(prefix, VisiblePrefix) or not isinstance(config, CompactionConfig):
        raise TypeError("prefix and CompactionConfig are required")
    header = f"VISIBLE PREFIX THROUGH TURN {prefix.after_turn}; LOSSY EXTRACT:"
    lines = [header]
    for message in prefix.messages[-config.keep_last_messages :]:
        body: dict[str, Any] = {"role": message["role"], "content": message["content"]}
        if "tool_calls" in message:
            body["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            body["tool_call_id"] = message["tool_call_id"]
        excerpt = _truncate_utf8(
            _canonical_text("compaction excerpt", body),
            config.max_excerpt_bytes,
        )
        candidate = "- " + excerpt
        used = len(("\n".join(lines) + "\n").encode("utf-8"))
        remaining = config.max_summary_bytes - used
        if remaining <= 2:
            break
        lines.append(_truncate_utf8(candidate, remaining))
        if conservative_token_upper_bound("\n".join(lines)) >= config.max_summary_bytes:
            break
    summary = _truncate_utf8("\n".join(lines), config.max_summary_bytes)
    if conservative_token_upper_bound(summary) > config.max_summary_bytes:
        raise AssertionError("compaction byte bound failed")
    return summary


@dataclass(frozen=True, slots=True)
class ScheduledMember:
    member_id: str
    eligible_checkpoints: tuple[int, ...]
    action_checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier("schedule member_id", self.member_id)
        for name in ("eligible_checkpoints", "action_checkpoints"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in values
            ):
                raise InterventionValidationError(f"{name} must be a tuple of positive integers")
            if tuple(sorted(set(values))) != values:
                raise InterventionValidationError(f"{name} must be unique and increasing")
        if not set(self.action_checkpoints).issubset(self.eligible_checkpoints):
            raise ScheduleMismatchError("action checkpoints are not all eligible")


@dataclass(frozen=True, slots=True)
class CheckpointSchedule:
    """One score-free checkpoint declaration shared by all compared cells."""

    group_id: str
    mode: ScheduleMode
    members: tuple[ScheduledMember, ...]
    seed: int
    yoke_anchor_member_id: str | None = None
    schema_version: int = OPERATORS_VERSION

    def __post_init__(self) -> None:
        _identifier("schedule group_id", self.group_id)
        if not isinstance(self.mode, ScheduleMode):
            raise InterventionValidationError("schedule mode must be ScheduleMode")
        if not isinstance(self.members, tuple) or len(self.members) < 2 or any(
            not isinstance(member, ScheduledMember) for member in self.members
        ):
            raise ScheduleMismatchError("a matched/yoked schedule needs at least two members")
        member_ids = tuple(member.member_id for member in self.members)
        if member_ids != tuple(sorted(member_ids)) or len(member_ids) != len(set(member_ids)):
            raise ScheduleMismatchError("schedule members must be unique and sorted")
        action_sets = {member.action_checkpoints for member in self.members}
        if len(action_sets) != 1:
            raise ScheduleMismatchError("all schedule members must have identical action checkpoints")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise InterventionValidationError("schedule seed must be non-negative")
        if self.mode is ScheduleMode.MATCHED:
            if self.yoke_anchor_member_id is not None:
                raise ScheduleMismatchError("matched schedules cannot name a yoke anchor")
            if len({member.eligible_checkpoints for member in self.members}) != 1:
                raise ScheduleMismatchError("matched members must have identical eligible checkpoints")
        else:
            if self.yoke_anchor_member_id not in member_ids:
                raise ScheduleMismatchError("yoked schedule anchor must be one declared member")
        if self.schema_version != OPERATORS_VERSION:
            raise InterventionValidationError("unsupported operator schedule schema version")

    @property
    def action_checkpoints(self) -> tuple[int, ...]:
        return self.members[0].action_checkpoints

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "mode": self.mode.value,
            "seed": self.seed,
            "yoke_anchor_member_id": self.yoke_anchor_member_id,
            "members": [
                {
                    "member_id": member.member_id,
                    "eligible_checkpoints": list(member.eligible_checkpoints),
                    "action_checkpoints": list(member.action_checkpoints),
                }
                for member in self.members
            ],
        }

    @property
    def schedule_sha256(self) -> str:
        return sha256_json(self.as_dict())

    def require_action(self, member_id: str, checkpoint: int) -> ScheduledMember:
        member_id = _identifier("member_id", member_id)
        checkpoint = _positive_integer("checkpoint", checkpoint)
        matches = [member for member in self.members if member.member_id == member_id]
        if len(matches) != 1:
            raise ScheduleMismatchError("intervention member is absent from the frozen schedule")
        member = matches[0]
        if checkpoint not in member.eligible_checkpoints:
            raise ScheduleMismatchError("intervention checkpoint is not eligible for this member")
        if checkpoint not in member.action_checkpoints:
            raise ScheduleMismatchError("intervention checkpoint was not selected by the schedule")
        return member


def _eligible_members(
    eligible_by_member: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not isinstance(eligible_by_member, Mapping) or len(eligible_by_member) < 2:
        raise ScheduleMismatchError("eligible_by_member must contain at least two members")
    normalized: list[tuple[str, tuple[int, ...]]] = []
    for member_id, checkpoints in eligible_by_member.items():
        member_id = _identifier("schedule member_id", member_id)
        if isinstance(checkpoints, (str, bytes)) or not isinstance(checkpoints, Sequence):
            raise ScheduleMismatchError("eligible checkpoints must be a sequence")
        values = tuple(checkpoints)
        if not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
        ):
            raise ScheduleMismatchError("eligible checkpoints must be positive integers")
        if tuple(sorted(set(values))) != values:
            raise ScheduleMismatchError("eligible checkpoints must be unique and increasing")
        normalized.append((member_id, values))
    normalized.sort(key=lambda item: item[0])
    if len({member_id for member_id, _values in normalized}) != len(normalized):
        raise ScheduleMismatchError("schedule member IDs must be unique")
    return tuple(normalized)


def _checkpoint_rank(group_id: str, seed: int, checkpoint: int) -> bytes:
    return hashlib.sha256(f"{group_id}\0{seed}\0{checkpoint}".encode("utf-8")).digest()


def build_matched_schedule(
    *,
    group_id: str,
    eligible_by_member: Mapping[str, Sequence[int]],
    intervention_count: int,
    seed: int,
) -> CheckpointSchedule:
    """Select identical checkpoints without accepting or inspecting signal scores."""

    group_id = _identifier("group_id", group_id)
    eligible = _eligible_members(eligible_by_member)
    if len({values for _member, values in eligible}) != 1:
        raise ScheduleMismatchError("matched members have different eligible checkpoints")
    if (
        isinstance(intervention_count, bool)
        or not isinstance(intervention_count, int)
        or intervention_count < 0
    ):
        raise InterventionValidationError("intervention_count must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InterventionValidationError("schedule seed must be non-negative")
    candidates = eligible[0][1]
    if intervention_count > len(candidates):
        raise ScheduleMismatchError("intervention_count exceeds eligible checkpoints")
    ranked = sorted(candidates, key=lambda checkpoint: _checkpoint_rank(group_id, seed, checkpoint))
    selected = tuple(sorted(ranked[:intervention_count]))
    return CheckpointSchedule(
        group_id=group_id,
        mode=ScheduleMode.MATCHED,
        members=tuple(ScheduledMember(member, values, selected) for member, values in eligible),
        seed=seed,
    )


def build_yoked_schedule(
    *,
    group_id: str,
    eligible_by_member: Mapping[str, Sequence[int]],
    anchor_member_id: str,
    anchor_action_checkpoints: Sequence[int],
    seed: int = 0,
) -> CheckpointSchedule:
    """Copy an anchor's declared action turns; numeric signal scores are not inputs."""

    group_id = _identifier("group_id", group_id)
    anchor_member_id = _identifier("anchor_member_id", anchor_member_id)
    eligible = _eligible_members(eligible_by_member)
    if anchor_member_id not in {member for member, _values in eligible}:
        raise ScheduleMismatchError("yoke anchor is absent from eligible members")
    if isinstance(anchor_action_checkpoints, (str, bytes)) or not isinstance(
        anchor_action_checkpoints, Sequence
    ):
        raise ScheduleMismatchError("anchor action checkpoints must be a sequence")
    actions = tuple(anchor_action_checkpoints)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in actions):
        raise ScheduleMismatchError("anchor action checkpoints must be positive integers")
    if tuple(sorted(set(actions))) != actions:
        raise ScheduleMismatchError("anchor action checkpoints must be unique and increasing")
    for member_id, member_eligible in eligible:
        if not set(actions).issubset(member_eligible):
            raise ScheduleMismatchError(
                f"member {member_id!r} cannot receive every yoked intervention"
            )
    return CheckpointSchedule(
        group_id=group_id,
        mode=ScheduleMode.YOKED,
        members=tuple(ScheduledMember(member, values, actions) for member, values in eligible),
        seed=seed,
        yoke_anchor_member_id=anchor_member_id,
    )


@dataclass(frozen=True, slots=True)
class InterventionApplication:
    intervention_type: InterventionType
    domain: str
    task_id: str
    schedule_sha256: str
    schedule_group_id: str
    schedule_mode: ScheduleMode
    member_id: str
    checkpoint: int
    source_prefix_sha256: str
    instructions_sha256: str | None
    signal_method: str | None
    signal_source_prefix_sha256: str | None
    signal_record_sha256: str | None
    signal_frozen_two_pass: bool
    payload_sha256: str
    added_message_sha256: str | None
    continued_history_json: str
    continued_history_sha256: str
    source_message_count: int
    continued_message_count: int
    dropped_message_count: int
    schema_version: int = OPERATORS_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.intervention_type, InterventionType):
            raise InterventionValidationError("intervention_type must be InterventionType")
        if not isinstance(self.schedule_mode, ScheduleMode):
            raise InterventionValidationError("schedule_mode must be ScheduleMode")
        _identifier("domain", self.domain)
        _identifier("task_id", self.task_id)
        _identifier("schedule_group_id", self.schedule_group_id)
        _identifier("member_id", self.member_id)
        _positive_integer("checkpoint", self.checkpoint)
        for name in (
            "schedule_sha256",
            "source_prefix_sha256",
            "payload_sha256",
            "continued_history_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in (
            "instructions_sha256",
            "signal_source_prefix_sha256",
            "signal_record_sha256",
            "added_message_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha256(name, value)
        if self.signal_method is not None:
            _identifier("signal_method", self.signal_method)
        if not isinstance(self.signal_frozen_two_pass, bool):
            raise InterventionValidationError("signal_frozen_two_pass must be boolean")
        signal_fields = (
            self.signal_method,
            self.signal_source_prefix_sha256,
            self.signal_record_sha256,
        )
        if any(value is None for value in signal_fields) != all(
            value is None for value in signal_fields
        ):
            raise InterventionValidationError("signal provenance fields must be all set or all null")
        if self.signal_frozen_two_pass and self.signal_record_sha256 is None:
            raise InterventionValidationError("two-pass provenance requires a signal")
        for name in ("source_message_count", "continued_message_count", "dropped_message_count"):
            _positive_integer(name, getattr(self, name), allow_zero=True)
        try:
            history = json.loads(self.continued_history_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InterventionValidationError("continued_history_json is invalid") from exc
        normalized = _normalize_messages(history, instructions_only=False)
        if _canonical_text("continued history", normalized) != self.continued_history_json:
            raise InterventionValidationError("continued_history_json must be canonical")
        if len(normalized) != self.continued_message_count:
            raise InterventionValidationError("continued message count mismatch")
        if sha256_json(normalized) != self.continued_history_sha256:
            raise InterventionValidationError("continued history does not match its hash")
        if self.schema_version != OPERATORS_VERSION:
            raise InterventionValidationError("unsupported intervention schema version")

    @property
    def continued_history(self) -> list[dict[str, Any]]:
        """Return a fresh history; mutation cannot change recorded provenance."""

        return json.loads(self.continued_history_json)

    def _provenance_core(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": "intervention_applied",
            "intervention_type": self.intervention_type.value,
            "domain": self.domain,
            "task_id": self.task_id,
            "schedule_sha256": self.schedule_sha256,
            "schedule_group_id": self.schedule_group_id,
            "schedule_mode": self.schedule_mode.value,
            "member_id": self.member_id,
            "checkpoint": self.checkpoint,
            "source_prefix_sha256": self.source_prefix_sha256,
            "instructions_sha256": self.instructions_sha256,
            "signal_method": self.signal_method,
            "signal_source_prefix_sha256": self.signal_source_prefix_sha256,
            "signal_record_sha256": self.signal_record_sha256,
            "signal_frozen_two_pass": self.signal_frozen_two_pass,
            "payload_sha256": self.payload_sha256,
            "added_message_sha256": self.added_message_sha256,
            "continued_history_sha256": self.continued_history_sha256,
            "source_message_count": self.source_message_count,
            "continued_message_count": self.continued_message_count,
            "dropped_message_count": self.dropped_message_count,
        }

    @property
    def provenance_sha256(self) -> str:
        return sha256_json(self._provenance_core())

    def as_event(self) -> dict[str, Any]:
        return {**self._provenance_core(), "provenance_sha256": self.provenance_sha256}


def _validate_instruction_prefix(
    prefix: VisiblePrefix,
    instructions: InitialInstructions | None,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    if instructions is None:
        if required:
            raise InterventionValidationError("this intervention requires frozen instructions")
        return []
    if not isinstance(instructions, InitialInstructions):
        raise InterventionValidationError("instructions must be InitialInstructions")
    if instructions.domain != prefix.domain or instructions.task_id != prefix.task_id:
        raise ContaminationError("instructions belong to a different task")
    instruction_messages = instructions.messages
    if prefix.messages[: len(instruction_messages)] != instruction_messages:
        raise ContaminationError("frozen instructions are not the exact start of this prefix")
    return instruction_messages


def apply_intervention(
    *,
    intervention_type: InterventionType | str,
    prefix: VisiblePrefix,
    schedule: CheckpointSchedule,
    member_id: str,
    checkpoint: int,
    instructions: InitialInstructions | None = None,
    public_state: PublicStateSnapshot | None = None,
    feedback: FeedbackNote | None = None,
    compaction_config: CompactionConfig | None = None,
    signal: SignalReference | None = None,
) -> InterventionApplication:
    """Apply one scheduled intervention without mutating the visible prefix."""

    try:
        kind = intervention_type if isinstance(intervention_type, InterventionType) else InterventionType(intervention_type)
    except (TypeError, ValueError) as exc:
        raise InterventionValidationError(
            "intervention_type must be exactly none, compact, reground, or feedback"
        ) from exc
    if not isinstance(prefix, VisiblePrefix) or not isinstance(schedule, CheckpointSchedule):
        raise TypeError("VisiblePrefix and CheckpointSchedule are required")
    checkpoint = _positive_integer("checkpoint", checkpoint)
    if prefix.after_turn != checkpoint:
        raise ContaminationError("visible prefix checkpoint does not match the scheduled action")
    schedule.require_action(member_id, checkpoint)
    if signal is not None:
        if not isinstance(signal, SignalReference):
            raise InterventionValidationError("signal must be SignalReference")
        if signal.checkpoint != checkpoint:
            raise ContaminationError("signal provenance belongs to a different checkpoint")
        if signal.frozen_two_pass:
            if signal.schedule_sha256 != schedule.schedule_sha256:
                raise ContaminationError(
                    "frozen signal provenance belongs to a different deployment schedule"
                )
        elif signal.source_prefix_sha256 != prefix.prefix_sha256:
            raise ContaminationError("signal provenance belongs to a different prefix")

    instruction_messages = _validate_instruction_prefix(
        prefix,
        instructions,
        required=kind in {InterventionType.COMPACT, InterventionType.REGROUND},
    )
    source_messages = prefix.messages
    added_message: dict[str, str] | None = None
    dropped = 0

    if kind is InterventionType.NONE:
        if any(value is not None for value in (public_state, feedback, compaction_config)):
            raise InterventionValidationError("none intervention cannot carry an operator payload")
        continued = json.loads(prefix.messages_json)
        payload = {"intervention_type": kind.value}
    elif kind is InterventionType.FEEDBACK:
        if feedback is None or public_state is not None or compaction_config is not None:
            raise InterventionValidationError("feedback intervention requires only FeedbackNote")
        if not isinstance(feedback, FeedbackNote):
            raise InterventionValidationError("feedback must be FeedbackNote")
        feedback.validate_for(prefix)
        added_message = {
            "role": "user",
            "content": feedback.render(),
        }
        continued = [*json.loads(prefix.messages_json), added_message]
        payload = {
            "intervention_type": kind.value,
            "feedback_note_sha256": feedback.note_sha256,
            "feedback_token_upper_bound": conservative_token_upper_bound(feedback.render()),
        }
    elif kind is InterventionType.COMPACT:
        if public_state is not None or feedback is not None:
            raise InterventionValidationError("compact intervention cannot carry state or feedback")
        config = compaction_config or CompactionConfig()
        if not isinstance(config, CompactionConfig):
            raise InterventionValidationError("compaction_config must be CompactionConfig")
        summary = build_lossy_compaction(prefix, config)
        added_message = {
            "role": "user",
            "content": "LOSSY SELF-SUMMARY; VERIFY AGAINST NEW INPUTS\n" + summary,
        }
        continued = [*instruction_messages, added_message]
        dropped = max(0, len(source_messages) - len(instruction_messages))
        payload = {
            "intervention_type": kind.value,
            "compaction_config_sha256": config.config_sha256,
            "summary_sha256": sha256_json(summary),
            "source_prefix_sha256": prefix.prefix_sha256,
        }
    else:
        if public_state is None or feedback is not None or compaction_config is not None:
            raise InterventionValidationError("reground intervention requires only PublicStateSnapshot")
        if not isinstance(public_state, PublicStateSnapshot):
            raise InterventionValidationError("public_state must be PublicStateSnapshot")
        if (
            public_state.domain != prefix.domain
            or public_state.task_id != prefix.task_id
            or public_state.after_turn != checkpoint
        ):
            raise ContaminationError("public state belongs to a different task or checkpoint")
        _assert_public_data(public_state.state)
        added_message = {
            "role": "user",
            "content": (
                f"EXACT PUBLIC STATE AFTER TURN {checkpoint}; NO PRIVATE OR FUTURE DATA\n"
                "PUBLIC_STATE_JSON:\n"
                + public_state.state_json
            ),
        }
        continued = [*instruction_messages, added_message]
        dropped = max(0, len(source_messages) - len(instruction_messages))
        payload = {
            "intervention_type": kind.value,
            "public_state_sha256": public_state.state_sha256,
            "public_state_after_turn": public_state.after_turn,
            "instructions_sha256": instructions.instructions_sha256 if instructions else None,
        }

    normalized_continued = _normalize_messages(continued, instructions_only=False)
    continued_json = _canonical_text("continued history", normalized_continued)
    added_sha = None if added_message is None else sha256_json(added_message)
    return InterventionApplication(
        intervention_type=kind,
        domain=prefix.domain,
        task_id=prefix.task_id,
        schedule_sha256=schedule.schedule_sha256,
        schedule_group_id=schedule.group_id,
        schedule_mode=schedule.mode,
        member_id=_identifier("member_id", member_id),
        checkpoint=checkpoint,
        source_prefix_sha256=prefix.prefix_sha256,
        instructions_sha256=(
            None if instructions is None else instructions.instructions_sha256
        ),
        signal_method=None if signal is None else signal.method,
        signal_source_prefix_sha256=(
            None if signal is None else signal.source_prefix_sha256
        ),
        signal_record_sha256=None if signal is None else signal.signal_record_sha256,
        signal_frozen_two_pass=False if signal is None else signal.frozen_two_pass,
        payload_sha256=sha256_json(payload),
        added_message_sha256=added_sha,
        continued_history_json=continued_json,
        continued_history_sha256=sha256_json(normalized_continued),
        source_message_count=len(source_messages),
        continued_message_count=len(normalized_continued),
        dropped_message_count=dropped,
    )


__all__ = [
    "OPERATORS_VERSION",
    "COMPACTION_CONTRACT",
    "REGROUND_CONTRACT",
    "FEEDBACK_CONTRACT",
    "InterventionValidationError",
    "ContaminationError",
    "ScheduleMismatchError",
    "InterventionType",
    "ScheduleMode",
    "VisiblePrefix",
    "freeze_visible_prefix",
    "InitialInstructions",
    "freeze_initial_instructions",
    "PublicStateSnapshot",
    "freeze_public_state",
    "SignalReference",
    "conservative_token_upper_bound",
    "FeedbackNote",
    "make_feedback_note",
    "CompactionConfig",
    "build_lossy_compaction",
    "ScheduledMember",
    "CheckpointSchedule",
    "build_matched_schedule",
    "build_yoked_schedule",
    "InterventionApplication",
    "apply_intervention",
]
