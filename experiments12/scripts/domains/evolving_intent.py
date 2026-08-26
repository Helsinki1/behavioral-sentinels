"""Frozen-data adapter for Microsoft's Evolving Intent GSM8K benchmark.

This module does not generate benchmark data.  It consumes an explicit rendered
JSON artifact with one paired ``t1`` and ``t7`` record per task.  Keeping the two
conditions explicit is important: the first turn of an evolving episode is not
the official one-turn, fully specified baseline.

Accepted JSON shape::

    {"tasks": [
      {"task_id": "gsm8k-12", "condition": "t1",
       "turns": ["..."], "label": "42"},
      {"task_id": "gsm8k-12", "condition": "t7",
       "turns": ["...", "...", "...", "...", "...", "...", "..."],
       "label": "42"}
    ]}

Extra source fields are deliberately ignored.  In particular, latent change
plans, future turns, predecessor structures, and gold metadata never enter an
observer checkpoint.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .base import (
    ArtifactIntegrityError,
    DomainTask,
    DomainTurn,
    DomainUnavailableError,
    DomainValidationError,
    InputArtifact,
    canonical_json_sha256,
    read_hashed_file,
)


DOMAIN = "evolving_intent_gsm8k"
PINNED_COMMIT = "993d6be9597ac03854b46362ccd647eb1bfd267a"
CONDITION_TURNS = {"t1": 1, "t7": 7}


def _task_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DomainValidationError("task_id must be a string or integer")
    result = str(value).strip()
    if not result:
        raise DomainValidationError("task_id must not be empty")
    return result


def _label(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DomainValidationError("label must be a non-empty string or integer")
    result = str(value).strip()
    if not result:
        raise DomainValidationError("label must not be empty")
    return result


def _turn_text(value: Any, index: int) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        present = [key for key in ("user_message", "user", "content") if key in value]
        if len(present) != 1:
            raise DomainValidationError(
                f"turn {index} must contain exactly one user text field"
            )
        text = value[present[0]]
    else:
        raise DomainValidationError(f"turn {index} must be a string or mapping")
    if not isinstance(text, str) or not text.strip():
        raise DomainValidationError(f"turn {index} user text must be non-empty")
    return text


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "tasks" not in value:
            raise DomainValidationError("dataset object must contain a tasks array")
        value = value["tasks"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DomainValidationError("dataset must be a JSON array or contain one")
    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise DomainValidationError(f"task record {index} must be an object")
        records.append(record)
    if not records:
        raise DomainValidationError("dataset contains no tasks")
    return records


def _task_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if "task_ids" not in value:
            raise DomainValidationError("task ID object must contain task_ids")
        value = value["task_ids"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DomainValidationError("task IDs must be a JSON array")
    normalized = tuple(_task_id(item) for item in value)
    if not normalized:
        raise DomainValidationError("task ID list is empty")
    if len(normalized) != len(set(normalized)):
        raise DomainValidationError("task ID list contains duplicates")
    return normalized


class EvolvingIntentAdapter:
    """Validate and expose one immutable rendered Evolving Intent artifact."""

    domain = DOMAIN

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        expected_sha256: str | None = None,
        task_ids_path: str | Path | None = None,
        expected_task_ids_sha256: str | None = None,
        upstream_root: str | Path | None = None,
    ) -> None:
        dataset_path, dataset_bytes, dataset_digest = read_hashed_file(
            dataset_path, expected_sha256=expected_sha256
        )
        self._dataset_path = dataset_path
        self._dataset_bytes = dataset_bytes
        self._dataset_sha256 = dataset_digest
        artifacts = [InputArtifact("rendered_dataset", str(dataset_path), dataset_digest)]

        self._expected_task_ids: tuple[str, ...] | None = None
        if task_ids_path is not None:
            ids_path, ids_bytes, ids_digest = read_hashed_file(
                task_ids_path, expected_sha256=expected_task_ids_sha256
            )
            try:
                ids_json = json.loads(ids_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DomainValidationError(f"invalid task ID JSON: {ids_path}") from exc
            self._expected_task_ids = _task_ids(ids_json)
            artifacts.append(InputArtifact("task_ids", str(ids_path), ids_digest))
        elif expected_task_ids_sha256 is not None:
            raise DomainValidationError(
                "expected_task_ids_sha256 requires an explicit task_ids_path"
            )

        self._input_artifacts = tuple(artifacts)
        self._tasks: tuple[DomainTask, ...] | None = None
        self._upstream_root: Path | None = None
        if upstream_root is not None:
            try:
                root = Path(upstream_root).expanduser().resolve(strict=True)
            except OSError as exc:
                raise DomainUnavailableError(
                    f"explicit Evolving Intent root is unavailable: {upstream_root}"
                ) from exc
            if not root.is_dir():
                raise DomainUnavailableError(f"upstream root is not a directory: {root}")
            self._upstream_root = root

    @property
    def input_artifacts(self) -> tuple[InputArtifact, ...]:
        return self._input_artifacts

    @property
    def source_sha256(self) -> str:
        return self._dataset_sha256

    def load_tasks(self) -> tuple[DomainTask, ...]:
        if self._tasks is not None:
            return self._tasks
        try:
            raw = json.loads(self._dataset_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DomainValidationError(
                f"invalid rendered dataset JSON: {self._dataset_path}"
            ) from exc

        parsed: dict[tuple[str, str], tuple[tuple[DomainTurn, ...], str]] = {}
        task_order: list[str] = []
        for record_index, record in enumerate(_records(raw)):
            missing = {"task_id", "condition", "turns", "label"} - set(record)
            if missing:
                raise DomainValidationError(
                    f"task record {record_index} is missing {sorted(missing)}"
                )
            task_id = _task_id(record["task_id"])
            condition = record["condition"]
            if not isinstance(condition, str) or condition not in CONDITION_TURNS:
                raise DomainValidationError(
                    f"task {task_id} has unsupported condition {condition!r}"
                )
            raw_turns = record["turns"]
            if isinstance(raw_turns, (str, bytes)) or not isinstance(raw_turns, Sequence):
                raise DomainValidationError(f"task {task_id}/{condition} turns must be an array")
            expected_count = CONDITION_TURNS[condition]
            if len(raw_turns) != expected_count:
                raise DomainValidationError(
                    f"task {task_id}/{condition} must have exactly {expected_count} turns"
                )
            turns = tuple(
                DomainTurn(index, _turn_text(turn, index))
                for index, turn in enumerate(raw_turns, start=1)
            )
            label = _label(record["label"])
            key = (task_id, condition)
            if key in parsed:
                raise DomainValidationError(f"duplicate task record: {task_id}/{condition}")
            parsed[key] = (turns, label)
            if task_id not in task_order:
                task_order.append(task_id)

        discovered_ids = set(task_order)
        if self._expected_task_ids is not None:
            expected_ids = set(self._expected_task_ids)
            if discovered_ids != expected_ids:
                missing = sorted(expected_ids - discovered_ids)
                extra = sorted(discovered_ids - expected_ids)
                raise ArtifactIntegrityError(
                    f"rendered task IDs differ from manifest; missing={missing}, extra={extra}"
                )
            task_order = list(self._expected_task_ids)

        tasks: list[DomainTask] = []
        for task_id in task_order:
            present = {condition for candidate_id, condition in parsed if candidate_id == task_id}
            if present != set(CONDITION_TURNS):
                raise DomainValidationError(
                    f"task {task_id} must contain paired t1 and t7 records; got {sorted(present)}"
                )
            for condition in ("t1", "t7"):
                turns, label = parsed[(task_id, condition)]
                task_payload = {
                    "domain": DOMAIN,
                    "upstream_commit": PINNED_COMMIT,
                    "source_sha256": self._dataset_sha256,
                    "task_id": task_id,
                    "condition": condition,
                    "turns": [turn.user_message for turn in turns],
                }
                tasks.append(
                    DomainTask(
                        domain=DOMAIN,
                        task_id=task_id,
                        condition=condition,
                        turns=turns,
                        evaluation_label=label,
                        source_sha256=self._dataset_sha256,
                        task_sha256=canonical_json_sha256(task_payload),
                        public_metadata=(
                            ("benchmark", "Evolving Intent GSM8K"),
                            ("upstream_commit", PINNED_COMMIT),
                        ),
                    )
                )

        self._tasks = tuple(tasks)
        return self._tasks

    def import_upstream_simulator(self) -> type[Any]:
        """Explicitly import upstream ``EvolvingIntent`` only when requested.

        Loading a frozen rendered artifact never imports upstream code.  This
        opt-in method executes code from the caller-supplied checkout and may
        therefore require dependencies not used by this adapter itself.
        """

        if self._upstream_root is None:
            raise DomainUnavailableError(
                "upstream simulator import requires an explicit upstream_root; "
                "environment-variable or ambient imports are not accepted"
            )
        return _import_upstream_simulator(self._upstream_root)


def _import_upstream_simulator(root: Path) -> type[Any]:
    expected = root / "situated_simulation" / "user_simulation.py"
    if not expected.is_file():
        raise DomainUnavailableError(
            f"upstream simulator module is missing from explicit root: {expected}"
        )

    package = sys.modules.get("situated_simulation")
    if package is not None:
        package_file = getattr(package, "__file__", None)
        if package_file is None or not Path(package_file).resolve().is_relative_to(root):
            raise DomainUnavailableError(
                "situated_simulation is already imported from a different root"
            )

    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("situated_simulation.user_simulation")
    except Exception as exc:
        raise DomainUnavailableError(
            f"could not import upstream simulator from explicit root {root}: {exc}"
        ) from exc
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

    module_file = getattr(module, "__file__", None)
    if module_file is None or not Path(module_file).resolve().is_relative_to(root):
        raise DomainUnavailableError("imported simulator did not originate in explicit root")
    simulator = getattr(module, "EvolvingIntent", None)
    if not isinstance(simulator, type):
        raise DomainUnavailableError("upstream module has no EvolvingIntent class")
    return simulator
