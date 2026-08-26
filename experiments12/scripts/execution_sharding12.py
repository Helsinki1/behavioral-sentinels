"""Deterministic, execution-only sharding for frozen Experiment 12 cells.

Shards are assigned by position in the already-validated, frozen pair table.
The assignment is deliberately absent from run manifests: changing the number
of workers changes only who executes a cell, never the scientific design.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence, TypeVar


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class ExecutionShard:
    """One member of an exact modulo partition of frozen declared order."""

    count: int = 1
    index: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ValueError("shard_count must be an integer")
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError("shard_index must be an integer")
        if self.count < 1:
            raise ValueError("shard_count must be positive")
        if self.index < 0 or self.index >= self.count:
            raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")

    def select(self, declared: Sequence[_T]) -> tuple[_T, ...]:
        """Select this worker's cells without changing frozen declared order."""

        return tuple(
            item
            for position, item in enumerate(declared)
            if position % self.count == self.index
        )


def add_execution_shard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the uniform execution-only shard flags to a paid-run parser."""

    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of disjoint execution workers (does not alter the manifest)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based worker index; frozen pair-order position modulo shard count",
    )


__all__ = ["ExecutionShard", "add_execution_shard_arguments"]
