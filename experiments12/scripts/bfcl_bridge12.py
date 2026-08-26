#!/usr/bin/env python3
"""Standalone JSONL bridge to the pinned BFCL V4 multi-turn evaluator.

The parent process sees only public prompts, native tool definitions, observed
tool outputs, redacted public state, and one final official score.  Hidden
initial configuration and possible answers never cross the process boundary.

This file intentionally depends only on the Python standard library.  At run
time it imports only the pinned checkout's official executor and checker; it
never imports a model handler or makes a model/network call.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import keyword
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL = "bfcl-v4-jsonl"
PINNED_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
LICENSE_IDENTIFIER = "Apache-2.0"
CAPABILITIES = [
    "begin_episode",
    "evaluate_episode",
    "execute_tools",
    "load_tasks",
    "materialize_public_state",
]

CATEGORY_FILES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}
FUNCTION_DOC_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}
STATELESS_CLASSES = frozenset({"MathAPI"})
OMIT_STATE_CLASSES = frozenset()
ADDITIONAL_FUNCTION_PROMPT = (
    "I have updated some more functions you can choose from. What about now?"
)

_TYPE_MAPPING = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}
_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "authorization",
    "authentication",
    "private",
    "ground_truth",
    "possible_answer",
    "binding_card",
    "card_number",
)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,511}$")
_SAFE_ERROR_TYPE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_INSTANCE_PUNCTUATION_RE = re.compile(r"[-./:]")
_PROTOCOL_STDOUT = sys.stdout


class BridgeFault(Exception):
    """A stable, non-sensitive protocol error."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(slots=True)
class TaskData:
    task_id: str
    category: str
    entry: dict[str, Any]
    ground_truth: list[list[str]]
    public_record: dict[str, Any]
    exposed_names_by_turn: tuple[frozenset[str], ...]
    schema_by_turn: tuple[dict[str, dict[str, Any]], ...]


@dataclass(slots=True)
class Episode:
    episode_id: str
    task: TaskData
    model_name: str
    current_turn: int = 1
    decoded_batches: list[list[list[str]]] = field(default_factory=list)
    instances: dict[str, Any] = field(default_factory=dict)
    public_snapshots: dict[int, dict[str, Any]] = field(default_factory=dict)
    evaluation_payload: dict[str, Any] | None = None


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BridgeFault("invalid_request", f"{where} must be a JSON object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise BridgeFault("invalid_request", f"{where} must be a JSON array")
    return value


def _exact(value: Any, fields: Sequence[str], where: str) -> dict[str, Any]:
    item = _mapping(value, where)
    if set(item) != set(fields):
        raise BridgeFault("invalid_request", f"{where} has invalid fields")
    return item


def _identifier(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise BridgeFault("invalid_request", f"{where} is invalid")
    return value


def _read_git_head(root: Path) -> str:
    marker = root / ".git"
    if marker.is_dir():
        git_dir = marker.resolve()
    elif marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if not text.lower().startswith("gitdir:"):
            raise BridgeFault("checkout_invalid", "BFCL Git metadata is invalid")
        git_dir = Path(text.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = marker.parent / git_dir
        git_dir = git_dir.resolve()
    else:
        raise BridgeFault("checkout_invalid", "BFCL Git metadata is unavailable")

    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    if not head.startswith("ref: refs/"):
        raise BridgeFault("checkout_invalid", "BFCL Git HEAD is invalid")
    ref = head[5:].strip()
    if Path(ref).is_absolute() or ".." in Path(ref).parts:
        raise BridgeFault("checkout_invalid", "BFCL Git HEAD is invalid")
    loose = git_dir / ref
    if loose.is_file():
        value = loose.read_text(encoding="utf-8").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", value):
            return value
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1] == ref:
                value = parts[0].lower()
                if re.fullmatch(r"[0-9a-f]{40}", value):
                    return value
    raise BridgeFault("checkout_invalid", "BFCL Git HEAD cannot be resolved")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise BridgeFault("checkout_invalid", "required BFCL data is unavailable") from None
    records: list[dict[str, Any]] = []
    try:
        for line in lines:
            if line.strip():
                value = json.loads(
                    line,
                    parse_constant=lambda _token: (_ for _ in ()).throw(ValueError()),
                )
                if not isinstance(value, dict):
                    raise ValueError
                records.append(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        raise BridgeFault("checkout_invalid", "BFCL data is malformed") from None
    if not records:
        raise BridgeFault("checkout_invalid", "required BFCL data is empty")
    return records


def _normalize_schema(value: Any) -> Any:
    """Reproduce BFCL's Gorilla-to-OpenAPI type conversion."""

    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "type" and isinstance(item, str):
            result[key] = _TYPE_MAPPING.get(item, "string")
        else:
            result[key] = _normalize_schema(item)
    return result


def _public_tool(function_doc: Mapping[str, Any]) -> dict[str, Any]:
    try:
        name = function_doc["name"]
        description = function_doc["description"]
        parameters = _normalize_schema(function_doc["parameters"])
    except (KeyError, TypeError):
        raise BridgeFault("checkout_invalid", "BFCL function documentation is malformed") from None
    if not isinstance(name, str) or not _SAFE_IDENTIFIER_RE.fullmatch(name):
        raise BridgeFault("checkout_invalid", "BFCL function name is invalid")
    if not isinstance(description, str) or not isinstance(parameters, dict):
        raise BridgeFault("checkout_invalid", "BFCL function documentation is malformed")
    parameters["type"] = "object"
    if not isinstance(parameters.get("properties", {}), dict):
        raise BridgeFault("checkout_invalid", "BFCL function parameters are malformed")
    # BFCL has optional/defaulted parameters, so declaring provider strict mode
    # would alter the benchmark's callable interface.
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": False,
    }


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_").replace(" ", "_")
    return key.startswith("_") or normalized == "parent" or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


class _Omitted:
    pass


_OMITTED = _Omitted()


def _json_public(value: Any, seen: set[int] | None = None, depth: int = 0) -> Any:
    """Convert official instances to bounded, non-secret JSON state."""

    if depth > 80:
        return _OMITTED
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _OMITTED
    if isinstance(value, Decimal):
        return format(value, "f") if value.is_finite() else _OMITTED

    seen = set() if seen is None else seen
    track_identity = isinstance(value, (dict, list, tuple, set, frozenset)) or hasattr(
        value, "__dict__"
    )
    identity = id(value)
    if track_identity:
        if identity in seen:
            return _OMITTED
        seen.add(identity)
    try:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key in sorted(value, key=lambda item: str(item)):
                if not isinstance(key, (str, int, float, bool)):
                    continue
                text_key = str(key)
                if _is_sensitive_key(text_key):
                    continue
                converted = _json_public(value[key], seen, depth + 1)
                if converted is not _OMITTED:
                    result[text_key] = converted
            return result
        if isinstance(value, (list, tuple)):
            result_list = []
            for item in value:
                converted = _json_public(item, seen, depth + 1)
                if converted is not _OMITTED:
                    result_list.append(converted)
            return result_list
        if isinstance(value, (set, frozenset)):
            converted_items = []
            for item in value:
                converted = _json_public(item, seen, depth + 1)
                if converted is not _OMITTED:
                    converted_items.append(converted)
            return sorted(converted_items, key=_canonical)
        if hasattr(value, "__dict__"):
            attributes = {
                key: item
                for key, item in vars(value).items()
                if isinstance(key, str) and not _is_sensitive_key(key)
            }
            converted = _json_public(attributes, seen, depth + 1)
            if converted is _OMITTED:
                return _OMITTED
            return {"object_type": type(value).__name__, "attributes": converted}
        return _OMITTED
    finally:
        if track_identity:
            seen.discard(identity)


def _public_state(instances: Mapping[str, Any]) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for class_name in sorted(instances):
        if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_CLASSES:
            continue
        converted = _json_public(instances[class_name])
        if converted is not _OMITTED:
            classes[class_name] = converted
    state = {"classes": classes}
    # Serialization here is also a fail-closed privacy/canonicalization gate.
    _canonical(state)
    return state


def _schema_accepts(value: Any, schema: Any) -> bool:
    if not isinstance(schema, dict):
        return True
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            return False
        if any(item not in value for item in required):
            return False
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        return all(
            key not in properties or _schema_accepts(item, properties[key])
            for key, item in value.items()
        )
    if expected == "array":
        return isinstance(value, list) and all(
            _schema_accepts(item, schema.get("items", {})) for item in value
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _python_literal(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{repr(key)}: {_python_literal(value[key])}" for key in sorted(value)
        ) + "}"
    raise BridgeFault("invalid_request", "tool arguments contain unsupported JSON")


def _function_call(name: str, arguments: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(arguments):
        if not key.isidentifier() or keyword.iskeyword(key):
            raise BridgeFault("invalid_request", "tool argument name is invalid")
        parts.append(f"{key}={_python_literal(arguments[key])}")
    return f"{name}({', '.join(parts)})"


class BFCLRuntime:
    def __init__(self) -> None:
        root_value = os.environ.get("BFCL_ROOT")
        if not isinstance(root_value, str) or not root_value.strip():
            raise BridgeFault("checkout_unavailable", "BFCL_ROOT is required")
        try:
            self.root = Path(root_value).resolve(strict=True)
        except OSError:
            raise BridgeFault("checkout_unavailable", "BFCL_ROOT is unavailable") from None
        if not self.root.is_dir() or _read_git_head(self.root) != PINNED_COMMIT:
            raise BridgeFault("checkout_revision_mismatch", "BFCL checkout revision differs")
        self.package_root = self.root / "berkeley-function-call-leaderboard"
        self.data_root = self.package_root / "bfcl_eval" / "data"
        if not self.package_root.is_dir() or not self.data_root.is_dir():
            raise BridgeFault("checkout_invalid", "BFCL V4 package layout is unavailable")
        package_text = str(self.package_root.resolve())
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
        self._function_docs: dict[str, list[dict[str, Any]]] = {}
        self._tasks_by_category: dict[str, tuple[TaskData, ...]] = {}
        self._tasks_by_id: dict[str, TaskData] = {}
        self._official_modules: tuple[Any, Any, Any, Any] | None = None
        self.episodes: dict[str, Episode] = {}

    def _load_function_docs(self, class_name: str) -> list[dict[str, Any]]:
        cached = self._function_docs.get(class_name)
        if cached is not None:
            return deepcopy(cached)
        filename = FUNCTION_DOC_FILES.get(class_name)
        if filename is None:
            raise BridgeFault("checkout_invalid", "BFCL task names an unsupported backend")
        records = _read_jsonl(self.data_root / "multi_turn_func_doc" / filename)
        public = [_public_tool(record) for record in records]
        names = [item["name"] for item in public]
        if len(names) != len(set(names)):
            raise BridgeFault("checkout_invalid", "BFCL function names are duplicated")
        self._function_docs[class_name] = public
        return deepcopy(public)

    def _make_task(
        self,
        category: str,
        entry: dict[str, Any],
        ground_truth: list[list[str]],
    ) -> TaskData:
        try:
            task_id = entry["id"]
            questions = entry["question"]
            involved_classes = entry["involved_classes"]
        except KeyError:
            raise BridgeFault("checkout_invalid", "BFCL task data is malformed") from None
        if (
            not isinstance(task_id, str)
            or not isinstance(questions, list)
            or not questions
            or not isinstance(involved_classes, list)
            or any(not isinstance(item, str) for item in involved_classes)
            or len(questions) != len(ground_truth)
        ):
            raise BridgeFault("checkout_invalid", "BFCL task data is malformed")

        all_tools: list[dict[str, Any]] = []
        for class_name in involved_classes:
            all_tools.extend(self._load_function_docs(class_name))
        all_names = [tool["name"] for tool in all_tools]
        if len(all_names) != len(set(all_names)):
            raise BridgeFault("checkout_invalid", "BFCL task exposes ambiguous tool names")

        holdout: dict[str, list[dict[str, Any]]] = {}
        raw_holdout = entry.get("missed_function", {})
        if not isinstance(raw_holdout, dict):
            raise BridgeFault("checkout_invalid", "BFCL held-out functions are malformed")
        for turn_key, raw_names in raw_holdout.items():
            if not isinstance(turn_key, str) or not isinstance(raw_names, list):
                raise BridgeFault("checkout_invalid", "BFCL held-out functions are malformed")
            held_tools: list[dict[str, Any]] = []
            for name in raw_names:
                if not isinstance(name, str):
                    raise BridgeFault("checkout_invalid", "BFCL held-out functions are malformed")
                for index, tool in enumerate(all_tools):
                    if tool["name"] == name:
                        held_tools.append(all_tools.pop(index))
                        break
                else:
                    raise BridgeFault("checkout_invalid", "BFCL held-out function is unavailable")
            holdout[turn_key] = held_tools

        turns: list[dict[str, Any]] = []
        exposed_names_by_turn: list[frozenset[str]] = []
        schema_by_turn: list[dict[str, dict[str, Any]]] = []
        current_tools = all_tools
        for zero_index, messages in enumerate(questions):
            if str(zero_index) in holdout:
                current_tools = [*current_tools, *holdout[str(zero_index)]]
                if messages != []:
                    raise BridgeFault("checkout_invalid", "BFCL held-out turn is not empty")
                user_message = ADDITIONAL_FUNCTION_PROMPT
            else:
                if (
                    not isinstance(messages, list)
                    or len(messages) != 1
                    or not isinstance(messages[0], dict)
                    or set(messages[0]) != {"role", "content"}
                    or messages[0]["role"] != "user"
                    or not isinstance(messages[0]["content"], str)
                    or not messages[0]["content"].strip()
                ):
                    raise BridgeFault("checkout_invalid", "BFCL turn message is unsupported")
                user_message = messages[0]["content"]
            turn_tools = deepcopy(current_tools)
            turns.append(
                {
                    "index": zero_index + 1,
                    "user_message": user_message,
                    "tools": turn_tools,
                }
            )
            exposed_names_by_turn.append(
                frozenset(tool["name"] for tool in turn_tools)
            )
            schema_by_turn.append(
                {tool["name"]: tool["parameters"] for tool in turn_tools}
            )

        core = {"task_id": task_id, "category": category, "turns": turns}
        public_record = {**core, "task_sha256": _digest(core)}
        return TaskData(
            task_id=task_id,
            category=category,
            entry=entry,
            ground_truth=ground_truth,
            public_record=public_record,
            exposed_names_by_turn=tuple(exposed_names_by_turn),
            schema_by_turn=tuple(schema_by_turn),
        )

    def _load_category(self, category: str) -> tuple[TaskData, ...]:
        cached = self._tasks_by_category.get(category)
        if cached is not None:
            return cached
        filename = CATEGORY_FILES.get(category)
        if filename is None:
            raise BridgeFault("invalid_request", "unsupported BFCL category")
        entries = _read_jsonl(self.data_root / filename)
        answers = _read_jsonl(self.data_root / "possible_answer" / filename)
        if len(entries) != len(answers):
            raise BridgeFault("checkout_invalid", "BFCL task and answer counts differ")
        tasks: list[TaskData] = []
        for entry, answer in zip(entries, answers):
            if entry.get("id") != answer.get("id"):
                raise BridgeFault("checkout_invalid", "BFCL task and answer IDs differ")
            ground_truth = answer.get("ground_truth")
            if (
                not isinstance(ground_truth, list)
                or any(
                    not isinstance(turn, list)
                    or any(not isinstance(call, str) for call in turn)
                    for turn in ground_truth
                )
            ):
                raise BridgeFault("checkout_invalid", "BFCL possible answer is malformed")
            task = self._make_task(category, entry, ground_truth)
            if task.task_id in self._tasks_by_id:
                raise BridgeFault("checkout_invalid", "BFCL task ID is duplicated")
            self._tasks_by_id[task.task_id] = task
            tasks.append(task)
        result = tuple(tasks)
        self._tasks_by_category[category] = result
        return result

    def _task(self, task_id: str) -> TaskData:
        cached = self._tasks_by_id.get(task_id)
        if cached is not None:
            return cached
        category = task_id.rsplit("_", 1)[0]
        if category in CATEGORY_FILES:
            self._load_category(category)
        task = self._tasks_by_id.get(task_id)
        if task is None:
            raise BridgeFault("task_not_found", "requested BFCL task is unavailable")
        return task

    def _official(self) -> tuple[Any, Any, Any, Any]:
        if self._official_modules is not None:
            return self._official_modules
        try:
            with redirect_stdout(sys.stderr):
                from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
                from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
                    multi_turn_checker,
                    multi_turn_irrelevance_checker,
                )
        except (ImportError, OSError):
            raise BridgeFault(
                "official_dependency_unavailable",
                "the official BFCL executor dependencies are unavailable",
            ) from None
        self._official_modules = (
            multi_turn_utils.execute_multi_turn_func_call,
            multi_turn_checker,
            multi_turn_irrelevance_checker,
            multi_turn_utils,
        )
        return self._official_modules

    def _execute_official(
        self, episode: Episode, calls: list[str]
    ) -> tuple[list[str], dict[str, Any]]:
        execute, _checker, _irrelevance, _module = self._official()
        try:
            with redirect_stdout(sys.stderr):
                outputs, instances = execute(
                    func_call_list=calls,
                    initial_config=episode.task.entry["initial_config"],
                    involved_classes=episode.task.entry["involved_classes"],
                    model_name=episode.model_name,
                    test_entry_id=episode.task.task_id,
                    long_context="long_context" in episode.task.category,
                    is_evaL_run=False,
                )
        except (ImportError, ModuleNotFoundError):
            raise BridgeFault(
                "official_dependency_unavailable",
                "the official BFCL backend dependencies are unavailable",
            ) from None
        except Exception:
            raise BridgeFault(
                "official_executor_failed", "the official BFCL executor failed"
            ) from None
        if (
            not isinstance(outputs, list)
            or any(not isinstance(item, str) for item in outputs)
            or len(outputs) != len(calls)
            or not isinstance(instances, dict)
        ):
            raise BridgeFault(
                "official_executor_failed", "the official BFCL executor returned malformed data"
            )
        return outputs, instances

    def _episode(self, episode_id: Any, task_id: Any) -> Episode:
        episode_id = _identifier(episode_id, "episode_id")
        task_id = _identifier(task_id, "task_id")
        episode = self.episodes.get(episode_id)
        if episode is None:
            raise BridgeFault("episode_not_found", "BFCL episode is unavailable")
        if episode.task.task_id != task_id:
            raise BridgeFault("episode_mismatch", "BFCL episode identity differs")
        return episode

    def hello(self, payload: Any) -> dict[str, Any]:
        _exact(payload, (), "hello payload")
        return {
            "protocol": PROTOCOL,
            "bfcl_commit": PINNED_COMMIT,
            "license": LICENSE_IDENTIFIER,
            "capabilities": CAPABILITIES,
        }

    def load_tasks(self, payload: Any) -> dict[str, Any]:
        request = _exact(payload, ("categories", "task_ids"), "load_tasks payload")
        categories = _array(request["categories"], "categories")
        task_ids = _array(request["task_ids"], "task_ids")
        if (
            not categories
            or any(not isinstance(item, str) or item not in CATEGORY_FILES for item in categories)
            or len(categories) != len(set(categories))
            or any(not isinstance(item, str) for item in task_ids)
            or len(task_ids) != len(set(task_ids))
        ):
            raise BridgeFault("invalid_request", "task selection is invalid")
        available: dict[str, TaskData] = {}
        for category in categories:
            available.update((task.task_id, task) for task in self._load_category(category))
        if task_ids:
            try:
                selected = [available[task_id] for task_id in task_ids]
            except KeyError:
                raise BridgeFault("task_not_found", "requested BFCL task is unavailable") from None
        else:
            selected = [
                task for category in categories for task in self._tasks_by_category[category]
            ]
        return {"tasks": [deepcopy(task.public_record) for task in selected]}

    def begin_episode(self, payload: Any) -> dict[str, Any]:
        request = _exact(payload, ("episode_id", "task_id"), "begin_episode payload")
        episode_id = _identifier(request["episode_id"], "episode_id")
        task_id = _identifier(request["task_id"], "task_id")
        if episode_id in self.episodes:
            raise BridgeFault("episode_exists", "BFCL episode ID already exists")
        task = self._task(task_id)
        model_name = "bfcl_bridge_" + hashlib.sha256(
            episode_id.encode("utf-8")
        ).hexdigest()[:24]
        episode = Episode(
            episode_id=episode_id,
            task=task,
            model_name=model_name,
            decoded_batches=[[] for _turn in task.public_record["turns"]],
        )
        _outputs, instances = self._execute_official(episode, [])
        episode.instances = instances
        episode.public_snapshots[0] = _public_state(instances)
        self.episodes[episode_id] = episode
        return {"episode_id": episode_id, "task_id": task_id, "started": True}

    def execute_tools(self, payload: Any) -> dict[str, Any]:
        request = _exact(
            payload,
            ("episode_id", "task_id", "turn_index", "tool_calls"),
            "execute_tools payload",
        )
        episode = self._episode(request["episode_id"], request["task_id"])
        turn_index = request["turn_index"]
        if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
            raise BridgeFault("invalid_request", "turn_index is invalid")
        if episode.current_turn > len(episode.decoded_batches):
            raise BridgeFault("episode_complete", "all BFCL turns are already ended")
        if turn_index != episode.current_turn:
            raise BridgeFault("turn_order", "BFCL turns must be ended in order")
        if episode.evaluation_payload is not None:
            raise BridgeFault("episode_evaluated", "BFCL episode is already evaluated")
        raw_calls = _array(request["tool_calls"], "tool_calls")
        if not raw_calls:
            episode.public_snapshots[turn_index] = _public_state(episode.instances)
            episode.current_turn += 1
            return {
                "episode_id": episode.episode_id,
                "task_id": episode.task.task_id,
                "turn_index": turn_index,
                "results": [],
                "state_check": "not_run",
            }

        exposed = episode.task.exposed_names_by_turn[turn_index - 1]
        schemas = episode.task.schema_by_turn[turn_index - 1]
        parsed: list[tuple[str, str, dict[str, Any], str | None]] = []
        call_ids: set[str] = set()
        valid_call_strings: list[str] = []
        valid_positions: list[int] = []
        for index, raw_call in enumerate(raw_calls):
            call = _exact(
                raw_call, ("call_id", "name", "arguments"), f"tool_calls[{index}]"
            )
            call_id = _identifier(call["call_id"], "call_id")
            name = _identifier(call["name"], "tool name")
            arguments = _mapping(call["arguments"], "tool arguments")
            if call_id in call_ids:
                raise BridgeFault("invalid_request", "tool call IDs must be unique")
            call_ids.add(call_id)
            invalid_reason: str | None = None
            call_string: str | None = None
            if name not in exposed:
                invalid_reason = "tool_not_exposed"
            elif not _schema_accepts(arguments, schemas[name]):
                invalid_reason = "arguments_do_not_match_schema"
            else:
                try:
                    call_string = _function_call(name, arguments)
                except BridgeFault:
                    invalid_reason = "arguments_cannot_be_executed"
            parsed.append((call_id, name, arguments, invalid_reason))
            if invalid_reason is None and call_string is not None:
                valid_positions.append(index)
                valid_call_strings.append(call_string)

        official_outputs: list[str] = []
        if valid_call_strings:
            official_outputs, instances = self._execute_official(
                episode, valid_call_strings
            )
            episode.instances = instances
            episode.decoded_batches[turn_index - 1].append(valid_call_strings)

        output_by_position = dict(zip(valid_positions, official_outputs))
        results: list[dict[str, Any]] = []
        for index, (call_id, name, _arguments, invalid_reason) in enumerate(parsed):
            if invalid_reason is not None:
                status = "invalid_call"
                output: Any = {"error": invalid_reason}
            else:
                official_output = output_by_position[index]
                if official_output.startswith("Error during execution:"):
                    status = "execution_failure"
                    output = {"error": "official_execution_failed"}
                else:
                    status = "succeeded"
                    output = official_output
            results.append(
                {"call_id": call_id, "name": name, "status": status, "output": output}
            )
        return {
            "episode_id": episode.episode_id,
            "task_id": episode.task.task_id,
            "turn_index": turn_index,
            "results": results,
            "state_check": "not_run",
        }

    def materialize_public_state(self, payload: Any) -> dict[str, Any]:
        request = _exact(
            payload,
            ("episode_id", "task_id", "after_turn"),
            "materialize_public_state payload",
        )
        episode = self._episode(request["episode_id"], request["task_id"])
        after_turn = request["after_turn"]
        if (
            isinstance(after_turn, bool)
            or not isinstance(after_turn, int)
            or after_turn < 0
            or after_turn not in episode.public_snapshots
        ):
            raise BridgeFault(
                "public_state_unavailable", "requested public state is unavailable"
            )
        state = deepcopy(episode.public_snapshots[after_turn])
        return {
            "episode_id": episode.episode_id,
            "task_id": episode.task.task_id,
            "after_turn": after_turn,
            "state": state,
            "state_sha256": _digest(state),
        }

    def evaluate_episode(self, payload: Any) -> dict[str, Any]:
        request = _exact(
            payload, ("episode_id", "task_id"), "evaluate_episode payload"
        )
        episode = self._episode(request["episode_id"], request["task_id"])
        if episode.current_turn != len(episode.decoded_batches) + 1:
            raise BridgeFault(
                "episode_not_complete", "all BFCL turns must be ended before evaluation"
            )
        if episode.evaluation_payload is not None:
            return deepcopy(episode.evaluation_payload)

        _execute, checker, irrelevance_checker, _module = self._official()
        checker_name = episode.model_name + "_official_check"
        try:
            with redirect_stdout(sys.stderr):
                accuracy = checker(
                    episode.decoded_batches,
                    episode.task.ground_truth,
                    episode.task.entry,
                    episode.task.category,
                    checker_name,
                )
                irrelevance_required = episode.task.category in {
                    "multi_turn_miss_func",
                    "multi_turn_miss_param",
                }
                irrelevance = (
                    irrelevance_checker(
                        episode.decoded_batches, episode.task.ground_truth
                    )
                    if irrelevance_required
                    else {"valid": True}
                )
        except (ImportError, ModuleNotFoundError):
            raise BridgeFault(
                "official_dependency_unavailable",
                "the official BFCL checker dependencies are unavailable",
            ) from None
        except Exception:
            raise BridgeFault(
                "official_checker_failed", "the official BFCL checker failed"
            ) from None
        if not isinstance(accuracy, dict) or not isinstance(accuracy.get("valid"), bool):
            raise BridgeFault(
                "official_checker_failed", "the official BFCL checker returned malformed data"
            )
        if not isinstance(irrelevance, dict) or not isinstance(
            irrelevance.get("valid"), bool
        ):
            raise BridgeFault(
                "official_checker_failed", "the official BFCL checker returned malformed data"
            )

        accuracy_valid = accuracy["valid"]
        irrelevance_valid = irrelevance["valid"]
        success = accuracy_valid and irrelevance_valid
        failure_type: str | None = None
        if not accuracy_valid:
            candidate = accuracy.get("error_type")
            if isinstance(candidate, str) and _SAFE_ERROR_TYPE_RE.fullmatch(candidate):
                failure_type = candidate
            else:
                failure_type = "multi_turn:official_check_failed"
        elif not irrelevance_valid:
            candidate = irrelevance.get("error_type")
            if isinstance(candidate, str) and _SAFE_ERROR_TYPE_RE.fullmatch(candidate):
                failure_type = candidate
            else:
                failure_type = "multi_turn:irrelevance_check_failed"
        official_result = {
            "checker": "BFCL_v4_multi_turn_checker",
            "valid": success,
            "state_and_response_valid": accuracy_valid,
            "irrelevance_checked": irrelevance_required,
            "irrelevance_valid": irrelevance_valid,
            "failure_type": failure_type,
        }
        result = {
            "episode_id": episode.episode_id,
            "task_id": episode.task.task_id,
            "official_score": 1 if success else 0,
            "official_success": success,
            "official_result": official_result,
        }
        episode.evaluation_payload = deepcopy(result)
        return result

    def dispatch(self, operation: str, payload: Any) -> dict[str, Any]:
        handlers = {
            "hello": self.hello,
            "load_tasks": self.load_tasks,
            "begin_episode": self.begin_episode,
            "execute_tools": self.execute_tools,
            "materialize_public_state": self.materialize_public_state,
            "evaluate_episode": self.evaluate_episode,
        }
        handler = handlers.get(operation)
        if handler is None:
            raise BridgeFault("unsupported_operation", "operation is unsupported")
        return handler(payload)


def _response(
    *,
    operation: str,
    request_id: str,
    payload: dict[str, Any] | None = None,
    fault: BridgeFault | None = None,
) -> dict[str, Any]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "kind": "response",
        "operation": operation,
        "request_id": request_id,
    }
    if fault is None:
        return {**common, "ok": True, "payload": payload}
    return {
        **common,
        "ok": False,
        "error": {"code": fault.code, "message": fault.message},
    }


def _serve_jsonl() -> int:
    try:
        runtime: BFCLRuntime | None = BFCLRuntime()
        startup_fault: BridgeFault | None = None
    except BridgeFault as exc:
        runtime = None
        startup_fault = exc
    except Exception:
        runtime = None
        startup_fault = BridgeFault(
            "bridge_initialization_failed", "the BFCL bridge could not initialize"
        )

    for line in sys.stdin:
        operation = "invalid"
        request_id = "invalid"
        try:
            request = json.loads(
                line,
                parse_constant=lambda _token: (_ for _ in ()).throw(ValueError()),
            )
            request = _exact(
                request,
                ("schema_version", "kind", "operation", "request_id", "payload"),
                "request envelope",
            )
            if isinstance(request.get("operation"), str):
                operation = request["operation"]
            if isinstance(request.get("request_id"), str):
                request_id = request["request_id"]
            if request["schema_version"] != SCHEMA_VERSION or request["kind"] != "request":
                raise BridgeFault("invalid_request", "request envelope metadata differs")
            operation = _identifier(request["operation"], "operation")
            request_id = _identifier(request["request_id"], "request_id")
            if startup_fault is not None or runtime is None:
                raise startup_fault or BridgeFault(
                    "bridge_initialization_failed", "the BFCL bridge could not initialize"
                )
            result = runtime.dispatch(operation, request["payload"])
            event = _response(
                operation=operation, request_id=request_id, payload=result
            )
        except BridgeFault as exc:
            event = _response(
                operation=operation, request_id=request_id, fault=exc
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            event = _response(
                operation=operation,
                request_id=request_id,
                fault=BridgeFault("invalid_request", "request is not valid JSON"),
            )
        except Exception:
            event = _response(
                operation=operation,
                request_id=request_id,
                fault=BridgeFault("bridge_internal_error", "the BFCL bridge failed"),
            )
        try:
            _PROTOCOL_STDOUT.write(_canonical(event) + "\n")
            _PROTOCOL_STDOUT.flush()
        except (OSError, BrokenPipeError):
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pinned BFCL V4 JSONL bridge")
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args(argv)
    if not args.jsonl:
        parser.error("--jsonl is required")
    return _serve_jsonl()


if __name__ == "__main__":
    raise SystemExit(main())
