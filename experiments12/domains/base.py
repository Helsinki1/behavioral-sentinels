"""Small, answer-blind contracts shared by Experiment 12 domains.

Domain adapters may retain evaluator-only labels, but observer workers receive
only :class:`ObserverCheckpoint`.  A checkpoint is constructed from a completed
assistant prefix, so it cannot accidentally contain future user turns.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answers",
        "change_plan",
        "correct",
        "final_answer",
        "final_label",
        "future",
        "future_turns",
        "gold",
        "gold_answer",
        "ground_truth",
        "label",
        "predecessor_functions",
        "predecessors",
        "success",
        "target",
        "target_answer",
    }
)


class DomainError(RuntimeError):
    """Base class for domain integration failures."""


class DomainValidationError(DomainError, ValueError):
    """A frozen input does not satisfy its declared contract."""


class DomainUnavailableError(DomainError):
    """Required external code, data, or authorization is unavailable."""


class ArtifactIntegrityError(DomainValidationError):
    """An external artifact does not match its declared digest or revision."""


class PermissionGateError(DomainUnavailableError):
    """Use of a permission-gated external benchmark was refused."""


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{name} must be a non-empty string")


def validate_sha256(name: str, value: str) -> str:
    """Return a normalized SHA-256 digest or fail closed."""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.lower()):
        raise DomainValidationError(f"{name} must be a 64-character SHA256 digest")
    return value.lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_hashed_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, str]:
    """Read one exact file and verify its optional expected digest."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DomainUnavailableError(f"required file is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise DomainUnavailableError(f"required path is not a file: {resolved}")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise DomainUnavailableError(f"could not read required file: {resolved}") from exc
    digest = sha256_bytes(data)
    if expected_sha256 is not None:
        expected = validate_sha256("expected_sha256", expected_sha256)
        if digest != expected:
            raise ArtifactIntegrityError(
                f"SHA256 mismatch for {resolved}: expected {expected}, got {digest}"
            )
    return resolved, data, digest


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON using the repository's stable, whitespace-free encoding."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DomainValidationError("value is not canonical JSON data") from exc
    return sha256_bytes(payload)


@dataclass(frozen=True, slots=True)
class InputArtifact:
    """One immutable external input committed to a run manifest."""

    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _nonempty("role", self.role)
        _nonempty("path", self.path)
        validate_sha256("sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class DomainTurn:
    """One benchmark-supplied user turn."""

    index: int
    user_message: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise DomainValidationError("turn index must be a positive integer")
        _nonempty("user_message", self.user_message)


@dataclass(frozen=True, slots=True)
class ObservedTurn:
    """One completed user/assistant exchange visible at a checkpoint."""

    index: int
    user_message: str
    assistant_message: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 1:
            raise DomainValidationError("observed turn index must be positive")
        _nonempty("user_message", self.user_message)
        if not isinstance(self.assistant_message, str):
            raise DomainValidationError("assistant_message must be a string")


@dataclass(frozen=True, slots=True)
class ObserverCheckpoint:
    """The complete and only domain payload supplied to an observer."""

    domain: str
    task_id: str
    condition: str
    after_turn: int
    turns: tuple[ObservedTurn, ...]
    source_sha256: str
    task_sha256: str
    public_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("domain", "task_id", "condition"):
            _nonempty(name, getattr(self, name))
        if not isinstance(self.turns, tuple) or any(
            not isinstance(turn, ObservedTurn) for turn in self.turns
        ):
            raise DomainValidationError("checkpoint turns must be ObservedTurn records")
        if self.after_turn != len(self.turns) or self.after_turn < 1:
            raise DomainValidationError(
                "checkpoint after_turn must equal its non-empty completed prefix"
            )
        if tuple(turn.index for turn in self.turns) != tuple(
            range(1, self.after_turn + 1)
        ):
            raise DomainValidationError("checkpoint turns must be contiguous from turn 1")
        validate_sha256("source_sha256", self.source_sha256)
        validate_sha256("task_sha256", self.task_sha256)
        _validate_public_metadata(self.public_metadata)

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        """Return a fresh chat-style copy for a passive monitor request."""

        messages: list[dict[str, str]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        return tuple(messages)

    def to_observer_dict(self) -> dict[str, Any]:
        """Serialize without any evaluator label or unobserved turn."""

        return {
            "domain": self.domain,
            "task_id": self.task_id,
            "condition": self.condition,
            "after_turn": self.after_turn,
            "turns": [
                {
                    "index": turn.index,
                    "user_message": turn.user_message,
                    "assistant_message": turn.assistant_message,
                }
                for turn in self.turns
            ],
            "source_sha256": self.source_sha256,
            "task_sha256": self.task_sha256,
            "public_metadata": dict(self.public_metadata),
        }


def _validate_public_metadata(value: tuple[tuple[str, str], ...]) -> None:
    if not isinstance(value, tuple):
        raise DomainValidationError("public_metadata must be a tuple")
    keys: list[str] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise DomainValidationError("public_metadata entries must be key/value pairs")
        key, item_value = item
        _nonempty("public metadata key", key)
        if key.lower() in _FORBIDDEN_PUBLIC_KEYS:
            raise DomainValidationError(f"observer metadata key is forbidden: {key}")
        if not isinstance(item_value, str):
            raise DomainValidationError("public metadata values must be strings")
        keys.append(key)
    if len(keys) != len({key.lower() for key in keys}):
        raise DomainValidationError("public_metadata keys must be unique")


@dataclass(frozen=True, slots=True)
class DomainTask:
    """A frozen scripted task plus evaluator-only outcome information.

    Generic harnesses should pass :meth:`checkpoint` output—not this object—to
    passive or active-observer workers.
    """

    domain: str
    task_id: str
    condition: str
    turns: tuple[DomainTurn, ...]
    evaluation_label: str | None
    source_sha256: str
    task_sha256: str
    public_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("domain", "task_id", "condition"):
            _nonempty(name, getattr(self, name))
        if not isinstance(self.turns, tuple) or any(
            not isinstance(turn, DomainTurn) for turn in self.turns
        ):
            raise DomainValidationError("turns must be a tuple of DomainTurn records")
        if not self.turns:
            raise DomainValidationError("a task must contain at least one turn")
        if tuple(turn.index for turn in self.turns) != tuple(range(1, len(self.turns) + 1)):
            raise DomainValidationError("task turns must be contiguous from turn 1")
        if self.evaluation_label is not None:
            _nonempty("evaluation_label", self.evaluation_label)
        validate_sha256("source_sha256", self.source_sha256)
        validate_sha256("task_sha256", self.task_sha256)
        _validate_public_metadata(self.public_metadata)

    @property
    def instance_id(self) -> str:
        return f"{self.domain}/{self.task_id}/{self.condition}"

    def next_turn(self, completed_turns: int) -> DomainTurn | None:
        """Return the next user turn without exposing any later turn."""

        if (
            isinstance(completed_turns, bool)
            or not isinstance(completed_turns, int)
            or completed_turns < 0
            or completed_turns > len(self.turns)
        ):
            raise DomainValidationError("completed_turns is outside the task")
        if completed_turns == len(self.turns):
            return None
        return self.turns[completed_turns]

    def checkpoint(self, assistant_messages: Sequence[str]) -> ObserverCheckpoint:
        """Build an answer-blind checkpoint for exactly the completed prefix."""

        if isinstance(assistant_messages, (str, bytes)):
            raise DomainValidationError("assistant_messages must be a sequence")
        completed = tuple(assistant_messages)
        if not completed or len(completed) > len(self.turns):
            raise DomainValidationError("checkpoint requires a non-empty valid prefix")
        if any(not isinstance(message, str) for message in completed):
            raise DomainValidationError("assistant messages must be strings")
        observed = tuple(
            ObservedTurn(turn.index, turn.user_message, assistant_message)
            for turn, assistant_message in zip(
                self.turns[: len(completed)], completed, strict=True
            )
        )
        return ObserverCheckpoint(
            domain=self.domain,
            task_id=self.task_id,
            condition=self.condition,
            after_turn=len(observed),
            turns=observed,
            source_sha256=self.source_sha256,
            task_sha256=self.task_sha256,
            public_metadata=self.public_metadata,
        )

    def manifest_record(self) -> dict[str, Any]:
        """Return provenance without placing the plaintext gold in a manifest."""

        return {
            "domain": self.domain,
            "task_id": self.task_id,
            "condition": self.condition,
            "num_turns": len(self.turns),
            "source_sha256": self.source_sha256,
            "task_sha256": self.task_sha256,
            "evaluation_label_sha256": (
                None
                if self.evaluation_label is None
                else sha256_bytes(self.evaluation_label.encode("utf-8"))
            ),
            "public_metadata": dict(self.public_metadata),
        }


@runtime_checkable
class DomainAdapter(Protocol):
    """Minimal adapter surface consumed by a generic task harness."""

    @property
    def domain(self) -> str: ...

    @property
    def input_artifacts(self) -> tuple[InputArtifact, ...]: ...

    def load_tasks(self) -> tuple[DomainTask, ...]: ...


@dataclass(frozen=True, slots=True)
class ExternalLoaderBoundary:
    """A subprocess specification that keeps upstream imports out of this process."""

    external_root: str
    root_environment_variable: str
    pinned_commit: str
    verified_inputs: tuple[InputArtifact, ...]

    def __post_init__(self) -> None:
        _nonempty("external_root", self.external_root)
        _nonempty("root_environment_variable", self.root_environment_variable)
        _nonempty("pinned_commit", self.pinned_commit)
        if not isinstance(self.verified_inputs, tuple) or any(
            not isinstance(item, InputArtifact) for item in self.verified_inputs
        ):
            raise DomainValidationError("verified_inputs must be InputArtifact records")

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return an explicit environment; caller chooses which base values survive."""

        result = dict(base or {})
        result[self.root_environment_variable] = self.external_root
        return result

    def command(
        self,
        loader_script: str | Path,
        *args: str,
        python_executable: str | None = None,
    ) -> tuple[str, ...]:
        """Construct, but do not execute, an isolated bridge command."""

        try:
            script = Path(loader_script).expanduser().resolve(strict=True)
        except OSError as exc:
            raise DomainUnavailableError(f"loader script is unavailable: {loader_script}") from exc
        if not script.is_file():
            raise DomainUnavailableError(f"loader script is not a file: {script}")
        executable = python_executable or sys.executable
        _nonempty("python_executable", executable)
        if any(not isinstance(argument, str) for argument in args):
            raise DomainValidationError("loader arguments must be strings")
        return (executable, str(script), *args)
