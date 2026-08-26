"""Predeclared, block-randomized cells and fail-closed completeness checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Iterable, Mapping, Sequence

from experiments12.core.artifacts import sha256_json
from experiments12.core.schemas import PairKey


ALLOWED_JOB_STATES = frozenset({"complete", "failed", "missing"})


@dataclass(frozen=True, slots=True)
class TaskRef:
    benchmark: str
    task_id: str
    task_sha256: str

    def __post_init__(self) -> None:
        if not self.benchmark or not self.task_id:
            raise ValueError("benchmark and task_id are required")
        if len(self.task_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.task_sha256):
            raise ValueError("task_sha256 must be lowercase SHA256")


@dataclass(frozen=True, slots=True)
class JobCell:
    cell_id: str
    block_id: str
    block_position: int
    pair_key: PairKey
    arm: str
    operator: str
    seed: int

    def __post_init__(self) -> None:
        if not self.cell_id or not self.block_id or not self.arm or not self.operator:
            raise ValueError("job identifiers/arm/operator cannot be empty")
        if self.block_position < 0 or self.seed < 0:
            raise ValueError("block position and seed must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "block_id": self.block_id,
            "block_position": self.block_position,
            "pair_key": {
                "model": self.pair_key.model,
                "domain": self.pair_key.domain,
                "task_id": self.pair_key.task_id,
                "replicate_id": self.pair_key.replicate_id,
                "task_sha256": self.pair_key.task_sha256,
            },
            "arm": self.arm,
            "operator": self.operator,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobCell":
        if not isinstance(value, Mapping):
            raise ValueError("job cell must be an object")
        expected = {
            "cell_id",
            "block_id",
            "block_position",
            "pair_key",
            "arm",
            "operator",
            "seed",
        }
        if set(value) != expected or not isinstance(value.get("pair_key"), Mapping):
            raise ValueError("job cell has missing or unexpected fields")
        cell = cls(
            cell_id=str(value["cell_id"]),
            block_id=str(value["block_id"]),
            block_position=int(value["block_position"]),
            pair_key=PairKey.from_dict(value["pair_key"]),
            arm=str(value["arm"]),
            operator=str(value["operator"]),
            seed=int(value["seed"]),
        )
        if cell.as_dict() != dict(value):
            raise ValueError("job cell values are not in canonical form")
        return cell


@dataclass(frozen=True, slots=True)
class CompletenessReport:
    expected: int
    complete: int
    failed: int
    missing: int
    duplicate_results: tuple[str, ...]
    failed_cells: tuple[str, ...]
    missing_cells: tuple[str, ...]

    @property
    def primary_ready(self) -> bool:
        return (
            self.expected == self.complete
            and not self.duplicate_results
            and not self.failed_cells
            and not self.missing_cells
        )


def _uint64(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def make_pair_manifest(
    *,
    tasks: Sequence[TaskRef],
    models: Sequence[str],
    arms: Sequence[str],
    operators: Sequence[str] = ("none",),
    replicates: int = 1,
    randomization_seed: int,
) -> tuple[JobCell, ...]:
    """Declare every required cell before dispatch and shuffle within blocks."""

    if not tasks or not models or not arms or not operators:
        raise ValueError("tasks/models/arms/operators must be nonempty")
    if replicates < 1 or randomization_seed < 0:
        raise ValueError("replicates must be positive and seed non-negative")
    for values, name in ((models, "model"), (arms, "arm"), (operators, "operator")):
        if len(values) != len(set(values)) or any(not value for value in values):
            raise ValueError(f"{name} values must be nonempty and unique")
    task_keys = [(task.benchmark, task.task_id, task.task_sha256) for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("duplicate task refs")

    blocks: list[list[JobCell]] = []
    for model in models:
        for task in tasks:
            for replicate in range(replicates):
                pair_key = PairKey(
                    model=model,
                    domain=task.benchmark,
                    task_id=task.task_id,
                    replicate_id=replicate,
                    task_sha256=task.task_sha256,
                )
                block_id = sha256_json(
                    {
                        "pair": pair_key.stable_id,
                        "task_sha256": task.task_sha256,
                        "randomization_seed": randomization_seed,
                    }
                )[:20]
                treatments = [(arm, operator) for operator in operators for arm in arms]
                rng = random.Random(
                    _uint64(randomization_seed, model, task.benchmark, task.task_id, replicate)
                )
                rng.shuffle(treatments)
                block: list[JobCell] = []
                for position, (arm, operator) in enumerate(treatments):
                    cell_payload = {
                        "pair": pair_key.stable_id,
                        "task_sha256": task.task_sha256,
                        "arm": arm,
                        "operator": operator,
                        "seed": randomization_seed,
                    }
                    block.append(
                        JobCell(
                            cell_id=sha256_json(cell_payload)[:24],
                            block_id=block_id,
                            block_position=position,
                            pair_key=pair_key,
                            arm=arm,
                            operator=operator,
                            seed=_uint64(randomization_seed, "cell", sha256_json(cell_payload)),
                        )
                    )
                blocks.append(block)

    # Round-robin over randomized within-task blocks. This prevents one whole
    # treatment from aligning with provider drift while retaining pair locality.
    rng = random.Random(_uint64(randomization_seed, "block-order"))
    rng.shuffle(blocks)
    cells: list[JobCell] = []
    width = max(len(block) for block in blocks)
    for position in range(width):
        for block in blocks:
            cells.append(block[position])
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise RuntimeError("cell ID collision")
    return tuple(cells)


def manifest_sha256(cells: Iterable[JobCell]) -> str:
    return sha256_json([cell.as_dict() for cell in cells])


def check_completeness(
    expected_cells: Sequence[JobCell],
    result_states: Iterable[tuple[str, str]],
) -> CompletenessReport:
    """Compare against the declaration; never intersect whatever is present."""

    expected_ids = {cell.cell_id for cell in expected_cells}
    if len(expected_ids) != len(expected_cells):
        raise ValueError("expected manifest contains duplicate cells")
    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    for cell_id, state in result_states:
        if state not in ALLOWED_JOB_STATES:
            raise ValueError(f"unknown job state: {state}")
        if cell_id not in expected_ids:
            raise ValueError(f"result does not belong to pair manifest: {cell_id}")
        if cell_id in seen:
            duplicates.add(cell_id)
        else:
            seen[cell_id] = state
    failed = sorted(cell_id for cell_id, state in seen.items() if state == "failed")
    explicit_missing = {cell_id for cell_id, state in seen.items() if state == "missing"}
    absent = expected_ids - set(seen)
    missing = sorted(explicit_missing | absent)
    complete = sum(state == "complete" for state in seen.values())
    return CompletenessReport(
        expected=len(expected_cells),
        complete=complete,
        failed=len(failed),
        missing=len(missing),
        duplicate_results=tuple(sorted(duplicates)),
        failed_cells=tuple(failed),
        missing_cells=tuple(missing),
    )
