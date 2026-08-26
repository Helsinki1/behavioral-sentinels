"""Pinned, out-of-process contract for BFCL V4 multi-turn episodes.

This module contains no BFCL code or data.  ``BFCL_ROOT`` must point to a
separately supplied checkout of the Gorilla repository at :data:`PINNED_COMMIT`.
The checkout is inspected as data, never imported into this process.

An external bridge supplied by the runner owns all official BFCL behavior.  It
speaks one JSON object per stdin/stdout line using this strict request envelope::

    {"schema_version": 1, "kind": "request", "operation": "...",
     "request_id": "...", "payload": {...}}

and one of these strict response envelopes::

    {"schema_version": 1, "kind": "response", "operation": "...",
     "request_id": "...", "ok": true, "payload": {...}}
    {"schema_version": 1, "kind": "response", "operation": "...",
     "request_id": "...", "ok": false,
     "error": {"code": "...", "message": "..."}}

Every object rejects unknown fields.  The supported operations are ``hello``,
``load_tasks``, ``begin_episode``, ``execute_tools``,
``materialize_public_state``, and ``evaluate_episode``.  The bridge stays alive
for an episode so the official executor can retain its environment.

There is deliberately no per-turn ``correct`` field.  Turn-level indicators
are derived only from directly observed invalid calls, execution failures, and
an official state check that actually returned ``failed``.  Official task
correctness is represented only by the final episode evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence, TextIO
from uuid import uuid4

from experiments12.core.transport import JsonSchemaTool, ToolCall

from .base import (
    ArtifactIntegrityError,
    DomainTask,
    DomainTurn,
    DomainUnavailableError,
    DomainValidationError,
    ExternalLoaderBoundary,
    InputArtifact,
    canonical_json_sha256,
    read_hashed_file,
    validate_sha256,
)


DOMAIN = "bfcl_multi_turn"
REPOSITORY = "https://github.com/ShishirPatil/gorilla"
PINNED_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
LICENSE_IDENTIFIER = "Apache-2.0"
ROOT_ENVIRONMENT_VARIABLE = "BFCL_ROOT"
BRIDGE_PROTOCOL = "bfcl-v4-jsonl"
BRIDGE_SCHEMA_VERSION = 1

V4_MULTI_TURN_FILES: tuple[tuple[str, str], ...] = (
    ("multi_turn_base", "BFCL_v4_multi_turn_base.json"),
    ("multi_turn_miss_func", "BFCL_v4_multi_turn_miss_func.json"),
    ("multi_turn_miss_param", "BFCL_v4_multi_turn_miss_param.json"),
    ("multi_turn_long_context", "BFCL_v4_multi_turn_long_context.json"),
)
V4_MULTI_TURN_CATEGORIES = tuple(category for category, _name in V4_MULTI_TURN_FILES)

# These are every answer and function-document file consumed by the real V4
# bridge.  They remain explicit so a new upstream file cannot silently enter an
# experiment through directory enumeration.
V4_POSSIBLE_ANSWER_FILES: tuple[tuple[str, str], ...] = V4_MULTI_TURN_FILES
V4_FUNCTION_DOC_FILES: tuple[tuple[str, str], ...] = (
    ("GorillaFileSystem", "gorilla_file_system.json"),
    ("MathAPI", "math_api.json"),
    ("MessageAPI", "message_api.json"),
    ("TwitterAPI", "posting_api.json"),
    ("TicketAPI", "ticket_api.json"),
    ("TradingBot", "trading_bot.json"),
    ("TravelAPI", "travel_booking.json"),
    ("VehicleControlAPI", "vehicle_control.json"),
)

# Exact upstream Python sources imported by bfcl_bridge12.py, including package
# initializers and the one shared backend helper imported transitively.  Third-
# party packages are runtime dependencies, not vendored BFCL inputs.
V4_OFFICIAL_SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("package_init", "bfcl_eval/__init__.py"),
    ("constants_init", "bfcl_eval/constants/__init__.py"),
    (
        "executable_backend_config",
        "bfcl_eval/constants/executable_backend_config.py",
    ),
    ("eval_checker_init", "bfcl_eval/eval_checker/__init__.py"),
    (
        "multi_turn_eval_init",
        "bfcl_eval/eval_checker/multi_turn_eval/__init__.py",
    ),
    (
        "multi_turn_executor",
        "bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py",
    ),
    (
        "multi_turn_checker",
        "bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py",
    ),
    (
        "backend_package_init",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/__init__.py",
    ),
    (
        "backend_gorilla_file_system",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/gorilla_file_system.py",
    ),
    (
        "backend_math_api",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/math_api.py",
    ),
    (
        "backend_message_api",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/message_api.py",
    ),
    (
        "backend_posting_api",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/posting_api.py",
    ),
    (
        "backend_ticket_api",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/ticket_api.py",
    ),
    (
        "backend_trading_bot",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/trading_bot.py",
    ),
    (
        "backend_travel_booking",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/travel_booking.py",
    ),
    (
        "backend_vehicle_control",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/vehicle_control.py",
    ),
    (
        "backend_long_context",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/long_context.py",
    ),
)

# The audited repository has historically been used both from its Git root and
# from the nested leaderboard directory.  A valid checkout must contain all
# four files in one of these known layouts; files are never searched by glob.
V4_DATA_DIRECTORY_CANDIDATES = (
    "berkeley-function-call-leaderboard/bfcl_eval/data",
    "berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn",
    "bfcl_eval/data",
    "bfcl_eval/data/multi_turn",
)
BFCL_PACKAGE_DIRECTORY_CANDIDATES = (
    "berkeley-function-call-leaderboard",
    ".",
)
LICENSE_PATH_CANDIDATES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "berkeley-function-call-leaderboard/LICENSE",
    "berkeley-function-call-leaderboard/LICENSE.txt",
)

BRIDGE_CAPABILITIES = frozenset(
    {
        "load_tasks",
        "begin_episode",
        "execute_tools",
        "materialize_public_state",
        "evaluate_episode",
    }
)

_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,79}$")
_MAX_IDENTIFIER_LENGTH = 512


class BFCLBridgeError(DomainUnavailableError):
    """A sanitized failure reported by, or while talking to, the bridge."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"official BFCL bridge request failed: {code}")


class ToolExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    INVALID_CALL = "invalid_call"
    EXECUTION_FAILURE = "execution_failure"


class StateCheckStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


def _identifier(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise DomainValidationError(f"{name} must be a bounded single-line string")
    return value


def _nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{name} must be non-empty text")
    return value


def _positive_integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DomainValidationError(f"{name} must be a {qualifier} integer")
    return value


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DomainValidationError(f"{name} must be a JSON object")
    return value


def _array(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise DomainValidationError(f"{name} must be a JSON array")
    return value


def _exact_keys(
    name: str,
    value: Any,
    required: frozenset[str] | set[str],
) -> Mapping[str, Any]:
    item = _mapping(name, value)
    expected = set(required)
    actual = set(item)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise DomainValidationError(
            f"{name} has invalid fields; missing={missing}, unknown={unknown}"
        )
    return item


def _canonical_json_text(name: str, value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise DomainValidationError(f"{name} is not canonical JSON data") from exc


def _decode_json_line(line: str, *, max_bytes: int) -> Mapping[str, Any]:
    if not isinstance(line, str) or not line.endswith("\n"):
        raise DomainValidationError("BFCL bridge response must be one newline-terminated line")
    if len(line.encode("utf-8")) > max_bytes:
        raise DomainValidationError("BFCL bridge response exceeds the configured line limit")
    try:
        value = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise DomainValidationError("BFCL bridge emitted invalid JSON") from exc
    return _mapping("BFCL bridge response", value)


def _path_inside(root: Path, candidate: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise DomainUnavailableError(f"required BFCL path is unavailable: {candidate}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError(f"BFCL path escapes the explicit checkout: {candidate}") from exc
    return resolved


def _git_directory(root: Path) -> Path:
    marker = root / ".git"
    if not marker.exists():
        raise ArtifactIntegrityError("BFCL_ROOT must be a Git checkout with .git metadata")
    if marker.is_dir():
        return marker.resolve()
    try:
        content = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ArtifactIntegrityError(f"could not read BFCL Git metadata: {marker}") from exc
    if not content.lower().startswith("gitdir:"):
        raise ArtifactIntegrityError("BFCL .git file is not a valid gitdir marker")
    git_dir = Path(content[len("gitdir:") :].strip())
    if not git_dir.is_absolute():
        git_dir = marker.parent / git_dir
    try:
        return git_dir.resolve(strict=True)
    except OSError as exc:
        raise ArtifactIntegrityError(f"BFCL Git directory is unavailable: {git_dir}") from exc


def _read_git_head(root: Path) -> str:
    git_dir = _git_directory(root)
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip().lower()
    except OSError as exc:
        raise ArtifactIntegrityError(f"could not read BFCL checkout HEAD: {head_path}") from exc
    if _GIT_OBJECT_RE.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        raise ArtifactIntegrityError("BFCL checkout HEAD is neither a commit nor a ref")
    ref = head[5:].strip()
    ref_path = Path(ref)
    if (
        not ref
        or ref_path.is_absolute()
        or ".." in ref_path.parts
        or not ref_path.parts
        or ref_path.parts[0] != "refs"
    ):
        raise ArtifactIntegrityError("BFCL checkout HEAD contains an invalid ref")
    loose = git_dir / ref
    if loose.is_file():
        value = loose.read_text(encoding="utf-8").strip().lower()
        if not _GIT_OBJECT_RE.fullmatch(value):
            raise ArtifactIntegrityError(f"invalid BFCL object ID in {loose}")
        return value
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            try:
                value, candidate_ref = line.split(" ", 1)
            except ValueError:
                continue
            if candidate_ref == ref and _GIT_OBJECT_RE.fullmatch(value.lower()):
                return value.lower()
    raise ArtifactIntegrityError(f"could not resolve BFCL checkout HEAD ref: {ref}")


def _validate_apache_license(data: bytes, path: Path) -> None:
    try:
        text = " ".join(data.decode("utf-8").lower().split())
    except UnicodeDecodeError as exc:
        raise ArtifactIntegrityError(f"BFCL license is not UTF-8 text: {path}") from exc
    markers = (
        "apache license",
        "version 2.0, january 2004",
        "terms and conditions for use, reproduction, and distribution",
        "accepting warranty or additional liability",
        "end of terms and conditions",
    )
    if any(marker not in text for marker in markers):
        raise ArtifactIntegrityError(
            f"BFCL checkout does not contain the audited {LICENSE_IDENTIFIER} text"
        )


def _license_artifact(root: Path) -> InputArtifact:
    for relative in LICENSE_PATH_CANDIDATES:
        candidate = root / relative
        if not candidate.is_file():
            continue
        resolved = _path_inside(root, candidate)
        resolved, data, digest = read_hashed_file(resolved)
        _validate_apache_license(data, resolved)
        return InputArtifact(f"bfcl:license:{LICENSE_IDENTIFIER}", str(resolved), digest)
    raise DomainUnavailableError("BFCL checkout is missing its Apache-2.0 license file")


def _v4_artifacts(root: Path) -> tuple[tuple[InputArtifact, ...], dict[str, str]]:
    chosen: Path | None = None
    for relative_directory in V4_DATA_DIRECTORY_CANDIDATES:
        directory = root / relative_directory
        if directory.is_dir() and all(
            (directory / filename).is_file() for _category, filename in V4_MULTI_TURN_FILES
        ):
            chosen = directory
            break
    if chosen is None:
        required = [filename for _category, filename in V4_MULTI_TURN_FILES]
        raise DomainUnavailableError(
            "BFCL checkout is missing a complete V4 multi-turn file set: " + ", ".join(required)
        )

    artifacts: list[InputArtifact] = []
    source_by_category: dict[str, str] = {}
    for category, filename in V4_MULTI_TURN_FILES:
        resolved = _path_inside(root, chosen / filename)
        resolved, data, digest = read_hashed_file(resolved)
        if not data.strip():
            raise ArtifactIntegrityError(f"BFCL V4 data file is empty: {resolved}")
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ArtifactIntegrityError(f"BFCL V4 data file is an unmaterialized LFS pointer: {resolved}")
        artifacts.append(InputArtifact(f"bfcl:v4:{category}", str(resolved), digest))
        source_by_category[category] = digest
    return tuple(artifacts), source_by_category


def _bfcl_package_directory(root: Path) -> Path:
    required_relatives = (
        *(
            f"bfcl_eval/data/possible_answer/{filename}"
            for _category, filename in V4_POSSIBLE_ANSWER_FILES
        ),
        *(
            f"bfcl_eval/data/multi_turn_func_doc/{filename}"
            for _class_name, filename in V4_FUNCTION_DOC_FILES
        ),
        *(relative for _label, relative in V4_OFFICIAL_SOURCE_FILES),
    )
    for relative_directory in BFCL_PACKAGE_DIRECTORY_CANDIDATES:
        candidate = root / relative_directory
        if candidate.is_dir() and all(
            (candidate / relative).is_file() for relative in required_relatives
        ):
            return _path_inside(root, candidate)
    raise DomainUnavailableError(
        "BFCL checkout is missing the explicit V4 answers, function docs, "
        "or official evaluator sources required by the bridge"
    )


def _dependency_artifact(
    root: Path,
    path: Path,
    role: str,
    *,
    allow_empty: bool = False,
) -> InputArtifact:
    resolved = _path_inside(root, path)
    resolved, data, digest = read_hashed_file(resolved)
    if not allow_empty and not data.strip():
        raise ArtifactIntegrityError(f"required BFCL bridge input is empty: {resolved}")
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ArtifactIntegrityError(
            f"required BFCL bridge input is an unmaterialized LFS pointer: {resolved}"
        )
    return InputArtifact(role, str(resolved), digest)


def _bridge_dependency_artifacts(
    root: Path,
) -> tuple[
    tuple[InputArtifact, ...],
    tuple[InputArtifact, ...],
    tuple[InputArtifact, ...],
]:
    package = _bfcl_package_directory(root)
    possible_answers = tuple(
        _dependency_artifact(
            root,
            package / "bfcl_eval" / "data" / "possible_answer" / filename,
            f"bfcl:v4:possible_answer:{category}",
        )
        for category, filename in V4_POSSIBLE_ANSWER_FILES
    )
    function_docs = tuple(
        _dependency_artifact(
            root,
            package / "bfcl_eval" / "data" / "multi_turn_func_doc" / filename,
            f"bfcl:v4:function_doc:{class_name}",
        )
        for class_name, filename in V4_FUNCTION_DOC_FILES
    )
    official_sources = tuple(
        _dependency_artifact(
            root,
            package / relative,
            f"bfcl:v4:official_source:{label}",
            allow_empty=relative.endswith("/__init__.py") or relative == "bfcl_eval/__init__.py",
        )
        for label, relative in V4_OFFICIAL_SOURCE_FILES
    )
    return possible_answers, function_docs, official_sources


def _tool_payload(tool: JsonSchemaTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.schema,
        "strict": tool.strict,
    }


def _parse_tool(value: Any, where: str) -> JsonSchemaTool:
    item = _exact_keys(where, value, {"name", "description", "parameters", "strict"})
    if not isinstance(item["name"], str) or not isinstance(item["description"], str):
        raise DomainValidationError(f"{where} name and description must be strings")
    if not isinstance(item["strict"], bool):
        raise DomainValidationError(f"{where} strict must be boolean")
    parameters = _mapping(f"{where}.parameters", item["parameters"])
    try:
        return JsonSchemaTool.from_schema(
            item["name"],
            item["description"],
            parameters,
            strict=item["strict"],
        )
    except ValueError as exc:
        raise DomainValidationError(f"{where} is not a valid native tool schema") from exc


@dataclass(frozen=True, slots=True)
class BFCLTaskTurn:
    """One official user turn and the native tools exposed for that turn."""

    index: int
    user_message: str
    tools: tuple[JsonSchemaTool, ...]

    def __post_init__(self) -> None:
        _positive_integer("BFCL turn index", self.index)
        _nonempty_text("BFCL user message", self.user_message)
        if not isinstance(self.tools, tuple) or any(
            not isinstance(tool, JsonSchemaTool) for tool in self.tools
        ):
            raise DomainValidationError("BFCL turn tools must be JsonSchemaTool records")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise DomainValidationError("BFCL turn tool names must be unique")

    def bridge_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "user_message": self.user_message,
            "tools": [_tool_payload(tool) for tool in self.tools],
        }


@dataclass(frozen=True, slots=True)
class BFCLTaskRecord:
    """An answer-blind official BFCL task returned by the external loader."""

    task_id: str
    category: str
    turns: tuple[BFCLTaskTurn, ...]
    source_sha256: str
    task_sha256: str

    def __post_init__(self) -> None:
        _identifier("BFCL task_id", self.task_id)
        if self.category not in V4_MULTI_TURN_CATEGORIES:
            raise DomainValidationError(f"unsupported BFCL V4 category: {self.category}")
        if not isinstance(self.turns, tuple) or not self.turns or any(
            not isinstance(turn, BFCLTaskTurn) for turn in self.turns
        ):
            raise DomainValidationError("BFCL task turns must be a non-empty tuple")
        if tuple(turn.index for turn in self.turns) != tuple(range(1, len(self.turns) + 1)):
            raise DomainValidationError("BFCL task turns must be contiguous from turn 1")
        validate_sha256("BFCL source_sha256", self.source_sha256)
        validate_sha256("BFCL task_sha256", self.task_sha256)
        if canonical_json_sha256(self.bridge_core()) != self.task_sha256:
            raise ArtifactIntegrityError("BFCL bridge task payload does not match task_sha256")

    def bridge_core(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "turns": [turn.bridge_payload() for turn in self.turns],
        }

    def as_domain_task(self) -> DomainTask:
        """Expose the common task surface without fabricating an evaluation label."""

        return DomainTask(
            domain=DOMAIN,
            task_id=self.task_id,
            condition="official_native_tools",
            turns=tuple(DomainTurn(turn.index, turn.user_message) for turn in self.turns),
            evaluation_label=None,
            source_sha256=self.source_sha256,
            task_sha256=self.task_sha256,
            public_metadata=(
                ("bfcl_category", self.category),
                ("tool_interface", "native"),
            ),
        )


@dataclass(frozen=True, slots=True)
class BFCLStartedEpisode:
    episode_id: str
    task_id: str

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        _identifier("task_id", self.task_id)


@dataclass(frozen=True, slots=True)
class BFCLToolExecutionResult:
    """One directly observed result from the official BFCL executor."""

    call_id: str
    name: str
    status: ToolExecutionStatus
    output_json: str

    def __post_init__(self) -> None:
        _identifier("tool result call_id", self.call_id)
        _identifier("tool result name", self.name)
        if not isinstance(self.status, ToolExecutionStatus):
            raise DomainValidationError("tool result status is invalid")
        try:
            value = json.loads(self.output_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DomainValidationError("tool result output_json is invalid") from exc
        if _canonical_json_text("tool result output", value) != self.output_json:
            raise DomainValidationError("tool result output_json must use canonical JSON")

    @property
    def output(self) -> Any:
        return json.loads(self.output_json)


@dataclass(frozen=True, slots=True)
class BFCLTurnFailureIndicators:
    """Only objectively observed process failures; never a correctness label."""

    invalid_call_observed: bool
    execution_failure_observed: bool
    state_check_failure_observed: bool
    state_check_available: bool

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.invalid_call_observed,
                self.execution_failure_observed,
                self.state_check_failure_observed,
                self.state_check_available,
            )
        ):
            raise DomainValidationError("turn failure indicators must be boolean")
        if self.state_check_failure_observed and not self.state_check_available:
            raise DomainValidationError("a failed state check must have been available")

    @property
    def any_observed_failure(self) -> bool:
        return (
            self.invalid_call_observed
            or self.execution_failure_observed
            or self.state_check_failure_observed
        )


@dataclass(frozen=True, slots=True)
class BFCLTurnExecution:
    episode_id: str
    task_id: str
    turn_index: int
    results: tuple[BFCLToolExecutionResult, ...]
    state_check: StateCheckStatus

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        _identifier("task_id", self.task_id)
        _positive_integer("turn_index", self.turn_index)
        if not isinstance(self.results, tuple) or any(
            not isinstance(result, BFCLToolExecutionResult) for result in self.results
        ):
            raise DomainValidationError("execution results must be BFCLToolExecutionResult records")
        if len({result.call_id for result in self.results}) != len(self.results):
            raise DomainValidationError("execution result call IDs must be unique")
        if not isinstance(self.state_check, StateCheckStatus):
            raise DomainValidationError("state_check is invalid")

    @property
    def failure_indicators(self) -> BFCLTurnFailureIndicators:
        return BFCLTurnFailureIndicators(
            invalid_call_observed=any(
                result.status is ToolExecutionStatus.INVALID_CALL for result in self.results
            ),
            execution_failure_observed=any(
                result.status is ToolExecutionStatus.EXECUTION_FAILURE for result in self.results
            ),
            state_check_failure_observed=self.state_check is StateCheckStatus.FAILED,
            state_check_available=self.state_check is not StateCheckStatus.NOT_RUN,
        )

    def tool_messages(self) -> tuple[dict[str, str], ...]:
        """Return tool-result messages suitable for the next transport request."""

        return tuple(
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": result.output_json,
            }
            for result in self.results
        )


@dataclass(frozen=True, slots=True)
class BFCLPublicState:
    episode_id: str
    task_id: str
    after_turn: int
    state_json: str
    state_sha256: str

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        _identifier("task_id", self.task_id)
        _positive_integer("after_turn", self.after_turn, allow_zero=True)
        try:
            state = json.loads(self.state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DomainValidationError("public state_json is invalid") from exc
        if not isinstance(state, dict):
            raise DomainValidationError("public BFCL state must be a JSON object")
        if _canonical_json_text("public BFCL state", state) != self.state_json:
            raise DomainValidationError("public state_json must use canonical JSON")
        validate_sha256("state_sha256", self.state_sha256)
        if canonical_json_sha256(state) != self.state_sha256:
            raise ArtifactIntegrityError("public BFCL state does not match state_sha256")

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(self.state_json)


@dataclass(frozen=True, slots=True)
class BFCLOfficialEpisodeEvaluation:
    """The only official task-correctness result exposed by this adapter."""

    episode_id: str
    task_id: str
    official_score: Decimal
    official_success: bool
    official_result_json: str

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        _identifier("task_id", self.task_id)
        if not isinstance(self.official_score, Decimal) or not self.official_score.is_finite():
            raise DomainValidationError("official_score must be a finite decimal")
        if not Decimal("0") <= self.official_score <= Decimal("1"):
            raise DomainValidationError("official_score must be between 0 and 1")
        if not isinstance(self.official_success, bool):
            raise DomainValidationError("official_success must be boolean")
        try:
            result = json.loads(self.official_result_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DomainValidationError("official_result_json is invalid") from exc
        if not isinstance(result, dict):
            raise DomainValidationError("official_result must be a JSON object")
        if _canonical_json_text("official_result", result) != self.official_result_json:
            raise DomainValidationError("official_result_json must use canonical JSON")

    @property
    def official_result(self) -> dict[str, Any]:
        return json.loads(self.official_result_json)


@dataclass(frozen=True, slots=True)
class BFCLReadiness:
    root: str
    checkout_commit: str
    repository: str
    license_identifier: str
    license_artifact: InputArtifact
    v4_data_artifacts: tuple[InputArtifact, ...]
    v4_possible_answer_artifacts: tuple[InputArtifact, ...]
    v4_function_doc_artifacts: tuple[InputArtifact, ...]
    v4_official_source_artifacts: tuple[InputArtifact, ...]

    @property
    def ready_for_external_bridge(self) -> bool:
        return (
            self.checkout_commit == PINNED_COMMIT
            and self.license_identifier == LICENSE_IDENTIFIER
            and len(self.v4_data_artifacts) == len(V4_MULTI_TURN_FILES)
            and len(self.v4_possible_answer_artifacts)
            == len(V4_POSSIBLE_ANSWER_FILES)
            and len(self.v4_function_doc_artifacts) == len(V4_FUNCTION_DOC_FILES)
            and len(self.v4_official_source_artifacts)
            == len(V4_OFFICIAL_SOURCE_FILES)
        )


def bridge_event_schemas() -> dict[str, Any]:
    """Return the exact field-level JSONL contract for bridge implementers.

    This compact descriptor is intentionally simpler than full JSON Schema;
    ``required`` is also the complete allowed field set for every object.
    Nested task/tool/result objects are validated by the records in this module.
    """

    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "request_envelope": {
            "required": ["schema_version", "kind", "operation", "request_id", "payload"],
            "kind": "request",
            "additional_properties": False,
        },
        "success_envelope": {
            "required": [
                "schema_version",
                "kind",
                "operation",
                "request_id",
                "ok",
                "payload",
            ],
            "kind": "response",
            "ok": True,
            "additional_properties": False,
        },
        "error_envelope": {
            "required": [
                "schema_version",
                "kind",
                "operation",
                "request_id",
                "ok",
                "error",
            ],
            "kind": "response",
            "ok": False,
            "error_required": ["code", "message"],
            "additional_properties": False,
        },
        "operations": {
            "hello": {"request": [], "response": ["protocol", "bfcl_commit", "license", "capabilities"]},
            "load_tasks": {"request": ["categories", "task_ids"], "response": ["tasks"]},
            "begin_episode": {
                "request": ["episode_id", "task_id"],
                "response": ["episode_id", "task_id", "started"],
            },
            "execute_tools": {
                "request": ["episode_id", "task_id", "turn_index", "tool_calls"],
                "response": ["episode_id", "task_id", "turn_index", "results", "state_check"],
            },
            "materialize_public_state": {
                "request": ["episode_id", "task_id", "after_turn"],
                "response": ["episode_id", "task_id", "after_turn", "state", "state_sha256"],
            },
            "evaluate_episode": {
                "request": ["episode_id", "task_id"],
                "response": [
                    "episode_id",
                    "task_id",
                    "official_score",
                    "official_success",
                    "official_result",
                ],
            },
        },
    }


class BFCLAdapter:
    """Verify an explicit BFCL checkout and create isolated bridge clients."""

    domain = DOMAIN

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        env = os.environ if environment is None else environment
        root_value = env.get(ROOT_ENVIRONMENT_VARIABLE)
        if not isinstance(root_value, str) or not root_value.strip():
            raise DomainUnavailableError(
                f"{ROOT_ENVIRONMENT_VARIABLE} must explicitly name the BFCL checkout"
            )
        try:
            root = Path(root_value).expanduser().resolve(strict=True)
        except OSError as exc:
            raise DomainUnavailableError(f"BFCL checkout is unavailable: {root_value}") from exc
        if not root.is_dir():
            raise DomainUnavailableError(f"BFCL_ROOT is not a directory: {root}")
        checkout_commit = _read_git_head(root)
        if checkout_commit != PINNED_COMMIT:
            raise ArtifactIntegrityError(
                f"BFCL checkout must be {PINNED_COMMIT}, got {checkout_commit}"
            )
        license_artifact = _license_artifact(root)
        v4_artifacts, source_by_category = _v4_artifacts(root)
        (
            possible_answer_artifacts,
            function_doc_artifacts,
            official_source_artifacts,
        ) = _bridge_dependency_artifacts(root)
        self._root = root
        self._source_by_category = source_by_category
        self._input_artifacts = (
            license_artifact,
            *v4_artifacts,
            *possible_answer_artifacts,
            *function_doc_artifacts,
            *official_source_artifacts,
        )
        self._readiness = BFCLReadiness(
            root=str(root),
            checkout_commit=checkout_commit,
            repository=REPOSITORY,
            license_identifier=LICENSE_IDENTIFIER,
            license_artifact=license_artifact,
            v4_data_artifacts=v4_artifacts,
            v4_possible_answer_artifacts=possible_answer_artifacts,
            v4_function_doc_artifacts=function_doc_artifacts,
            v4_official_source_artifacts=official_source_artifacts,
        )

    @property
    def input_artifacts(self) -> tuple[InputArtifact, ...]:
        return self._input_artifacts

    @property
    def readiness(self) -> BFCLReadiness:
        return self._readiness

    def loader_boundary(self) -> ExternalLoaderBoundary:
        """Recheck provenance and return a subprocess-only upstream boundary."""

        checkout_commit = _read_git_head(self._root)
        if checkout_commit != PINNED_COMMIT:
            raise ArtifactIntegrityError(
                f"BFCL checkout changed revisions after validation: {checkout_commit}"
            )
        for artifact in self._input_artifacts:
            _resolved, data, _digest = read_hashed_file(
                artifact.path,
                expected_sha256=artifact.sha256,
            )
            if artifact.role == f"bfcl:license:{LICENSE_IDENTIFIER}":
                _validate_apache_license(data, Path(artifact.path))
        return ExternalLoaderBoundary(
            external_root=str(self._root),
            root_environment_variable=ROOT_ENVIRONMENT_VARIABLE,
            pinned_commit=PINNED_COMMIT,
            verified_inputs=self._input_artifacts,
        )

    def bridge_client(
        self,
        bridge_script: str | Path,
        *,
        python_executable: str | None = None,
        base_environment: Mapping[str, str] | None = None,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        max_line_bytes: int = 64 * 1024 * 1024,
    ) -> "BFCLBridgeClient":
        return BFCLBridgeClient(
            self.loader_boundary(),
            bridge_script,
            source_sha256_by_category=self._source_by_category,
            python_executable=python_executable,
            base_environment=base_environment,
            popen_factory=popen_factory,
            max_line_bytes=max_line_bytes,
        )

    def load_tasks(self) -> tuple[DomainTask, ...]:
        raise DomainUnavailableError(
            "BFCL is interactive; load official tasks through bridge_client().load_tasks()"
        )


class BFCLBridgeClient:
    """Synchronous client for one persistent, isolated official BFCL process."""

    def __init__(
        self,
        boundary: ExternalLoaderBoundary,
        bridge_script: str | Path,
        *,
        source_sha256_by_category: Mapping[str, str],
        python_executable: str | None = None,
        base_environment: Mapping[str, str] | None = None,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        max_line_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if not isinstance(boundary, ExternalLoaderBoundary):
            raise TypeError("boundary must be ExternalLoaderBoundary")
        if boundary.pinned_commit != PINNED_COMMIT:
            raise ArtifactIntegrityError("bridge boundary does not use the pinned BFCL commit")
        if isinstance(max_line_bytes, bool) or not isinstance(max_line_bytes, int) or max_line_bytes < 1:
            raise DomainValidationError("max_line_bytes must be a positive integer")
        source_map = dict(source_sha256_by_category)
        if set(source_map) != set(V4_MULTI_TURN_CATEGORIES):
            raise DomainValidationError("source_sha256_by_category must cover every V4 category")
        for category, digest in source_map.items():
            validate_sha256(f"source digest for {category}", digest)
        self.boundary = boundary
        self.command = boundary.command(
            bridge_script,
            "--jsonl",
            python_executable=python_executable,
        )
        bridge_path, _bridge_data, bridge_sha256 = read_hashed_file(self.command[1])
        self.bridge_artifact = InputArtifact("bfcl:bridge", str(bridge_path), bridge_sha256)
        safe_base = dict(base_environment or {})
        safe_base.setdefault("PYTHONIOENCODING", "utf-8")
        safe_base.setdefault("PYTHONUNBUFFERED", "1")
        self.environment = boundary.environment(safe_base)
        self.source_sha256_by_category = source_map
        self.popen_factory = popen_factory
        self.max_line_bytes = max_line_bytes
        self._process: subprocess.Popen[str] | None = None

    @property
    def input_artifacts(self) -> tuple[InputArtifact, ...]:
        """Return every upstream input plus the exact external bridge script."""

        return (*self.boundary.verified_inputs, self.bridge_artifact)

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> "BFCLBridgeClient":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.running:
            return
        if self._process is not None:
            raise BFCLBridgeError("bridge_not_restartable")
        read_hashed_file(
            self.bridge_artifact.path,
            expected_sha256=self.bridge_artifact.sha256,
        )
        try:
            process = self.popen_factory(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                cwd=self.boundary.external_root,
                env=self.environment,
            )
        except (OSError, subprocess.SubprocessError):
            raise BFCLBridgeError("bridge_start_failed") from None
        self._process = process
        try:
            payload = self._exchange_started("hello", {})
            self._validate_hello(payload)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.SubprocessError):
                    pass
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _streams(self) -> tuple[TextIO, TextIO]:
        process = self._process
        if process is None or process.poll() is not None:
            raise BFCLBridgeError("bridge_not_running")
        if process.stdin is None or process.stdout is None:
            raise BFCLBridgeError("bridge_missing_stdio")
        return process.stdin, process.stdout

    def _exchange(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.running:
            self.start()
        return self._exchange_started(operation, payload)

    def _exchange_started(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation not in bridge_event_schemas()["operations"]:
            raise DomainValidationError(f"unsupported BFCL bridge operation: {operation}")
        request_id = uuid4().hex
        event = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "kind": "request",
            "operation": operation,
            "request_id": request_id,
            "payload": dict(payload),
        }
        line = _canonical_json_text("BFCL bridge request", event) + "\n"
        if len(line.encode("utf-8")) > self.max_line_bytes:
            raise DomainValidationError("BFCL bridge request exceeds the configured line limit")
        stdin, stdout = self._streams()
        try:
            stdin.write(line)
            stdin.flush()
            response_line = stdout.readline(self.max_line_bytes + 1)
        except (OSError, UnicodeError):
            raise BFCLBridgeError("bridge_io_failed") from None
        if not response_line:
            raise BFCLBridgeError("bridge_closed_without_response")
        response = _decode_json_line(response_line, max_bytes=self.max_line_bytes)
        common = {"schema_version", "kind", "operation", "request_id", "ok"}
        if response.get("ok") is True:
            response = _exact_keys("BFCL success response", response, common | {"payload"})
        elif response.get("ok") is False:
            response = _exact_keys("BFCL error response", response, common | {"error"})
        else:
            raise DomainValidationError("BFCL response ok must be boolean")
        if response["schema_version"] != BRIDGE_SCHEMA_VERSION:
            raise DomainValidationError("BFCL bridge response schema_version mismatch")
        if response["kind"] != "response":
            raise DomainValidationError("BFCL bridge response kind must be response")
        if response["operation"] != operation or response["request_id"] != request_id:
            raise DomainValidationError("BFCL bridge response does not match its request")
        if response["ok"] is False:
            error = _exact_keys("BFCL bridge error", response["error"], {"code", "message"})
            code = error["code"]
            if not isinstance(code, str) or not _SAFE_CODE_RE.fullmatch(code):
                code = "invalid_bridge_error"
            if not isinstance(error["message"], str):
                raise DomainValidationError("BFCL bridge error message must be a string")
            raise BFCLBridgeError(code)
        return _mapping("BFCL bridge payload", response["payload"])

    @staticmethod
    def _validate_hello(payload: Mapping[str, Any]) -> None:
        payload = _exact_keys(
            "BFCL hello payload",
            payload,
            {"protocol", "bfcl_commit", "license", "capabilities"},
        )
        if payload["protocol"] != BRIDGE_PROTOCOL:
            raise ArtifactIntegrityError("BFCL bridge protocol mismatch")
        if payload["bfcl_commit"] != PINNED_COMMIT:
            raise ArtifactIntegrityError("BFCL bridge does not attest the pinned commit")
        if payload["license"] != LICENSE_IDENTIFIER:
            raise ArtifactIntegrityError("BFCL bridge license identifier mismatch")
        capabilities = _array("BFCL bridge capabilities", payload["capabilities"])
        if any(not isinstance(item, str) for item in capabilities):
            raise DomainValidationError("BFCL bridge capabilities must be strings")
        if len(capabilities) != len(set(capabilities)) or set(capabilities) != BRIDGE_CAPABILITIES:
            raise DomainValidationError("BFCL bridge capabilities do not match the contract")

    def load_tasks(
        self,
        *,
        categories: Sequence[str] = V4_MULTI_TURN_CATEGORIES,
        task_ids: Sequence[str] = (),
    ) -> tuple[BFCLTaskRecord, ...]:
        if isinstance(categories, (str, bytes)) or not isinstance(categories, Sequence):
            raise DomainValidationError("categories must be a sequence")
        category_tuple = tuple(categories)
        if (
            not category_tuple
            or any(category not in V4_MULTI_TURN_CATEGORIES for category in category_tuple)
            or len(category_tuple) != len(set(category_tuple))
        ):
            raise DomainValidationError("categories must be unique BFCL V4 multi-turn categories")
        if isinstance(task_ids, (str, bytes)) or not isinstance(task_ids, Sequence):
            raise DomainValidationError("task_ids must be a sequence")
        task_id_tuple = tuple(_identifier("task_id", item) for item in task_ids)
        if len(task_id_tuple) != len(set(task_id_tuple)):
            raise DomainValidationError("task_ids must be unique")
        payload = self._exchange(
            "load_tasks",
            {"categories": list(category_tuple), "task_ids": list(task_id_tuple)},
        )
        payload = _exact_keys("load_tasks payload", payload, {"tasks"})
        raw_tasks = _array("load_tasks.tasks", payload["tasks"])
        tasks = tuple(
            self._parse_task(value, f"load_tasks.tasks[{index}]")
            for index, value in enumerate(raw_tasks)
        )
        if not tasks:
            raise DomainValidationError("BFCL bridge returned no official tasks")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise DomainValidationError("BFCL bridge returned duplicate task IDs")
        if any(task.category not in category_tuple for task in tasks):
            raise DomainValidationError("BFCL bridge returned an unrequested category")
        if task_id_tuple and {task.task_id for task in tasks} != set(task_id_tuple):
            raise DomainValidationError("BFCL bridge did not return exactly the requested task IDs")
        if not task_id_tuple and {task.category for task in tasks} != set(category_tuple):
            raise DomainValidationError("BFCL bridge did not cover every requested category")
        return tasks

    def _parse_task(self, value: Any, where: str) -> BFCLTaskRecord:
        item = _exact_keys(where, value, {"task_id", "category", "turns", "task_sha256"})
        task_id = _identifier(f"{where}.task_id", item["task_id"])
        category = item["category"]
        if category not in V4_MULTI_TURN_CATEGORIES:
            raise DomainValidationError(f"{where}.category is unsupported")
        raw_turns = _array(f"{where}.turns", item["turns"])
        turns: list[BFCLTaskTurn] = []
        for index, raw_turn in enumerate(raw_turns):
            turn = _exact_keys(
                f"{where}.turns[{index}]",
                raw_turn,
                {"index", "user_message", "tools"},
            )
            raw_tools = _array(f"{where}.turns[{index}].tools", turn["tools"])
            tools = tuple(
                _parse_tool(tool, f"{where}.turns[{index}].tools[{tool_index}]")
                for tool_index, tool in enumerate(raw_tools)
            )
            turns.append(
                BFCLTaskTurn(
                    index=_positive_integer(f"{where}.turns[{index}].index", turn["index"]),
                    user_message=_nonempty_text(
                        f"{where}.turns[{index}].user_message", turn["user_message"]
                    ),
                    tools=tools,
                )
            )
        return BFCLTaskRecord(
            task_id=task_id,
            category=category,
            turns=tuple(turns),
            source_sha256=self.source_sha256_by_category[category],
            task_sha256=validate_sha256(f"{where}.task_sha256", item["task_sha256"]),
        )

    def begin_episode(self, episode_id: str, task_id: str) -> BFCLStartedEpisode:
        episode_id = _identifier("episode_id", episode_id)
        task_id = _identifier("task_id", task_id)
        payload = self._exchange(
            "begin_episode", {"episode_id": episode_id, "task_id": task_id}
        )
        payload = _exact_keys(
            "begin_episode payload", payload, {"episode_id", "task_id", "started"}
        )
        if payload["episode_id"] != episode_id or payload["task_id"] != task_id:
            raise DomainValidationError("begin_episode response identity mismatch")
        if payload["started"] is not True:
            raise DomainValidationError("begin_episode response must affirm started=true")
        return BFCLStartedEpisode(episode_id, task_id)

    def execute_tools(
        self,
        episode_id: str,
        task_id: str,
        turn_index: int,
        tool_calls: Sequence[ToolCall],
    ) -> BFCLTurnExecution:
        episode_id = _identifier("episode_id", episode_id)
        task_id = _identifier("task_id", task_id)
        turn_index = _positive_integer("turn_index", turn_index)
        if isinstance(tool_calls, (str, bytes)) or not isinstance(tool_calls, Sequence):
            raise DomainValidationError("tool_calls must be a sequence")
        calls = tuple(tool_calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise DomainValidationError("tool_calls must contain transport ToolCall records")
        if len({call.call_id for call in calls}) != len(calls):
            raise DomainValidationError("tool call IDs must be unique")
        request_calls = [
            {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
            for call in calls
        ]
        payload = self._exchange(
            "execute_tools",
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "turn_index": turn_index,
                "tool_calls": request_calls,
            },
        )
        payload = _exact_keys(
            "execute_tools payload",
            payload,
            {"episode_id", "task_id", "turn_index", "results", "state_check"},
        )
        if (
            payload["episode_id"] != episode_id
            or payload["task_id"] != task_id
            or payload["turn_index"] != turn_index
        ):
            raise DomainValidationError("execute_tools response identity mismatch")
        raw_results = _array("execute_tools.results", payload["results"])
        results: list[BFCLToolExecutionResult] = []
        for index, raw_result in enumerate(raw_results):
            result = _exact_keys(
                f"execute_tools.results[{index}]",
                raw_result,
                {"call_id", "name", "status", "output"},
            )
            try:
                status = ToolExecutionStatus(result["status"])
            except (TypeError, ValueError) as exc:
                raise DomainValidationError("official tool execution status is invalid") from exc
            results.append(
                BFCLToolExecutionResult(
                    call_id=_identifier("tool result call_id", result["call_id"]),
                    name=_identifier("tool result name", result["name"]),
                    status=status,
                    output_json=_canonical_json_text("tool result output", result["output"]),
                )
            )
        expected = [(call.call_id, call.name) for call in calls]
        actual = [(result.call_id, result.name) for result in results]
        if actual != expected:
            raise DomainValidationError("tool results must exactly match requested calls in order")
        try:
            state_check = StateCheckStatus(payload["state_check"])
        except (TypeError, ValueError) as exc:
            raise DomainValidationError("official state_check status is invalid") from exc
        return BFCLTurnExecution(
            episode_id=episode_id,
            task_id=task_id,
            turn_index=turn_index,
            results=tuple(results),
            state_check=state_check,
        )

    def materialize_public_state(
        self,
        episode_id: str,
        task_id: str,
        after_turn: int,
    ) -> BFCLPublicState:
        episode_id = _identifier("episode_id", episode_id)
        task_id = _identifier("task_id", task_id)
        after_turn = _positive_integer("after_turn", after_turn, allow_zero=True)
        payload = self._exchange(
            "materialize_public_state",
            {"episode_id": episode_id, "task_id": task_id, "after_turn": after_turn},
        )
        payload = _exact_keys(
            "materialize_public_state payload",
            payload,
            {"episode_id", "task_id", "after_turn", "state", "state_sha256"},
        )
        if (
            payload["episode_id"] != episode_id
            or payload["task_id"] != task_id
            or payload["after_turn"] != after_turn
        ):
            raise DomainValidationError("public-state response identity mismatch")
        state = _mapping("materialize_public_state.state", payload["state"])
        return BFCLPublicState(
            episode_id=episode_id,
            task_id=task_id,
            after_turn=after_turn,
            state_json=_canonical_json_text("materialize_public_state.state", state),
            state_sha256=validate_sha256("state_sha256", payload["state_sha256"]),
        )

    def evaluate_episode(
        self,
        episode_id: str,
        task_id: str,
    ) -> BFCLOfficialEpisodeEvaluation:
        episode_id = _identifier("episode_id", episode_id)
        task_id = _identifier("task_id", task_id)
        payload = self._exchange(
            "evaluate_episode", {"episode_id": episode_id, "task_id": task_id}
        )
        payload = _exact_keys(
            "evaluate_episode payload",
            payload,
            {
                "episode_id",
                "task_id",
                "official_score",
                "official_success",
                "official_result",
            },
        )
        if payload["episode_id"] != episode_id or payload["task_id"] != task_id:
            raise DomainValidationError("official evaluation response identity mismatch")
        score_value = payload["official_score"]
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise DomainValidationError("official_score must be a JSON number")
        if isinstance(score_value, float) and not math.isfinite(score_value):
            raise DomainValidationError("official_score must be finite")
        try:
            score = Decimal(str(score_value))
        except (InvalidOperation, ValueError) as exc:
            raise DomainValidationError("official_score is invalid") from exc
        success = payload["official_success"]
        if not isinstance(success, bool):
            raise DomainValidationError("official_success must be boolean")
        official_result = _mapping("official_result", payload["official_result"])
        return BFCLOfficialEpisodeEvaluation(
            episode_id=episode_id,
            task_id=task_id,
            official_score=score,
            official_success=success,
            official_result_json=_canonical_json_text("official_result", official_result),
        )


__all__ = [
    "DOMAIN",
    "REPOSITORY",
    "PINNED_COMMIT",
    "LICENSE_IDENTIFIER",
    "ROOT_ENVIRONMENT_VARIABLE",
    "BRIDGE_PROTOCOL",
    "BRIDGE_SCHEMA_VERSION",
    "BRIDGE_CAPABILITIES",
    "V4_MULTI_TURN_FILES",
    "V4_MULTI_TURN_CATEGORIES",
    "V4_POSSIBLE_ANSWER_FILES",
    "V4_FUNCTION_DOC_FILES",
    "V4_OFFICIAL_SOURCE_FILES",
    "V4_DATA_DIRECTORY_CANDIDATES",
    "BFCL_PACKAGE_DIRECTORY_CANDIDATES",
    "BFCLBridgeError",
    "ToolExecutionStatus",
    "StateCheckStatus",
    "BFCLTaskTurn",
    "BFCLTaskRecord",
    "BFCLStartedEpisode",
    "BFCLToolExecutionResult",
    "BFCLTurnFailureIndicators",
    "BFCLTurnExecution",
    "BFCLPublicState",
    "BFCLOfficialEpisodeEvaluation",
    "BFCLReadiness",
    "bridge_event_schemas",
    "BFCLAdapter",
    "BFCLBridgeClient",
]
