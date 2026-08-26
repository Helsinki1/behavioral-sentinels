"""Compose independently frozen Evolving-Intent builds without model calls."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments12.build_evolving12 import PINNED_COMMIT, SEED
from experiments12.core.artifacts import (
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
)
from experiments12.domains.evolving_intent import DOMAIN, EvolvingIntentAdapter


COMBINE_VERSION = 1


class CompositionError(ValueError):
    """A child build or structural-screen receipt is incompatible."""


@dataclass(frozen=True, slots=True)
class FrozenChild:
    root: Path
    source_id: int
    task_id: str
    config: Mapping[str, Any]
    receipt: Mapping[str, Any]
    dataset: Mapping[str, Any]
    config_sha256: str
    receipt_sha256: str
    dataset_sha256: str


def _object(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise CompositionError(f"missing {label}: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise CompositionError(f"{label} must be a JSON object")
    return value


def _verify_declared_files(config: Mapping[str, Any]) -> None:
    declared: list[Any] = []
    declared.extend(config.get("inputs", ()))
    declared.extend(config.get("prompt_files", ()))
    bridge = config.get("bridge")
    if isinstance(bridge, Mapping):
        declared.append(
            {"path": bridge.get("path"), "sha256": bridge.get("sha256")}
        )
    for item in declared:
        if not isinstance(item, Mapping):
            raise CompositionError("child build has an invalid input receipt")
        path, digest = item.get("path"), item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise CompositionError("child input receipt lacks path/SHA256")
        candidate = Path(path)
        if not candidate.is_file() or sha256_file(candidate) != digest:
            raise CompositionError(f"child build input changed: {candidate.name}")


def load_child(path: str | Path) -> FrozenChild:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise CompositionError(f"child build is not a directory: {root}")
    config_path = root / "build_config.json"
    receipt_path = root / "build_receipt.json"
    dataset_path = root / "evolving_intent_gsm8k_frozen.json"
    config = _object(config_path, "child build config")
    receipt = _object(receipt_path, "child build receipt")
    dataset = _object(dataset_path, "child frozen dataset")

    if receipt.get("build_sha256") != sha256_json(config):
        raise CompositionError("child build receipt does not bind its config")
    frozen = receipt.get("frozen_dataset")
    if not isinstance(frozen, Mapping) or frozen.get("sha256") != sha256_file(dataset_path):
        raise CompositionError("child build receipt does not bind its dataset")
    source_ids, task_ids = config.get("source_ids"), config.get("task_ids")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != 1
        or isinstance(source_ids[0], bool)
        or not isinstance(source_ids[0], int)
        or not isinstance(task_ids, list)
        or len(task_ids) != 1
        or not isinstance(task_ids[0], str)
    ):
        raise CompositionError("each child must contain exactly one source task")
    if (
        config.get("benchmark") != DOMAIN
        or config.get("upstream_commit") != PINNED_COMMIT
        or config.get("seed") != SEED
        or config.get("shared_across_target_arms_and_models") is not True
        or receipt.get("generation") != config.get("generation")
        or receipt.get("bridge_runtime") != config.get("bridge", {}).get("runtime")
    ):
        raise CompositionError("child scientific metadata is inconsistent")
    _verify_declared_files(config)
    tasks = EvolvingIntentAdapter(
        dataset_path, expected_sha256=sha256_file(dataset_path)
    ).load_tasks()
    if (
        len(tasks) != 2
        or {task.condition for task in tasks} != {"t1", "t7"}
        or {task.task_id for task in tasks} != {task_ids[0]}
    ):
        raise CompositionError("child dataset is not one valid t1/t7 pair")
    raw_tasks = dataset.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 2:
        raise CompositionError("child public dataset has the wrong task count")
    return FrozenChild(
        root=root,
        source_id=source_ids[0],
        task_id=task_ids[0],
        config=config,
        receipt=receipt,
        dataset=dataset,
        config_sha256=sha256_file(config_path),
        receipt_sha256=sha256_file(receipt_path),
        dataset_sha256=sha256_file(dataset_path),
    )


def _common_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"source_ids", "task_ids"}
    }


def compose_frozen_builds(
    *,
    child_dirs: Sequence[str | Path],
    screen_plan_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    if not child_dirs:
        raise CompositionError("at least one child build is required")
    children = tuple(load_child(path) for path in child_dirs)
    if len({child.source_id for child in children}) != len(children):
        raise CompositionError("child source IDs must be unique")
    common = _common_config(children[0].config)
    if any(_common_config(child.config) != common for child in children[1:]):
        raise CompositionError("child builds do not share one generation/runtime config")

    plan_path = Path(screen_plan_path).expanduser().resolve(strict=True)
    plan = _object(plan_path, "structural screen plan")
    candidate_ids = plan.get("candidate_source_ids_in_order")
    target = plan.get("target_valid_tasks")
    if (
        not isinstance(candidate_ids, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or isinstance(target, bool)
        or not isinstance(target, int)
        or target < 1
    ):
        raise CompositionError("structural screen plan is invalid")
    source_ids = [child.source_id for child in children]
    try:
        ordered = sorted(source_ids, key=candidate_ids.index)
    except ValueError as exc:
        raise CompositionError("a child source is outside the frozen candidate list") from exc
    if source_ids != ordered or len(children) != target:
        raise CompositionError("children must be the ordered target number of screened tasks")
    generation = children[0].config.get("generation")
    if not isinstance(generation, Mapping) or any(
        plan.get(key) != generation.get(key)
        for key in (
            "generator_model",
            "generator_reasoning_effort",
            "judge_model",
            "judge_reasoning_effort",
        )
    ):
        raise CompositionError("screen plan and child generation settings differ")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"composition output already exists: {destination}")
    destination.mkdir(parents=True)
    records: list[Any] = []
    for child in children:
        records.extend(child.dataset["tasks"])
    dataset_payload = {
        "schema_version": 1,
        "benchmark": DOMAIN,
        "upstream_commit": PINNED_COMMIT,
        "seed": SEED,
        "shared_across_target_arms_and_models": True,
        "composition_version": COMBINE_VERSION,
        "tasks": records,
    }
    dataset_path = destination / "evolving_intent_gsm8k_frozen.json"
    atomic_write_json(dataset_path, dataset_payload)
    loaded = EvolvingIntentAdapter(
        dataset_path, expected_sha256=sha256_file(dataset_path)
    ).load_tasks()
    if len(loaded) != 2 * len(children):
        raise CompositionError("composed dataset failed independent adapter validation")

    input_tokens = output_tokens = cached_tokens = reasoning_tokens = 0
    cost = Decimal("0")
    call_count = 0
    for child in children:
        calls = child.receipt.get("calls")
        if not isinstance(calls, list):
            raise CompositionError("child receipt has no call ledger")
        call_count += len(calls)
        for call in calls:
            if not isinstance(call, Mapping) or not isinstance(call.get("usage"), Mapping):
                raise CompositionError("child call receipt is incomplete")
            usage = call["usage"]
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
            cached_tokens += int(usage.get("cached_input_tokens", 0))
            reasoning_tokens += int(usage.get("reasoning_tokens", 0))
            cost += Decimal(str(call.get("accounted_cost_usd")))
    receipt_payload = {
        "schema_version": 1,
        "composition_version": COMBINE_VERSION,
        "benchmark": DOMAIN,
        "upstream_commit": PINNED_COMMIT,
        "seed": SEED,
        "shared_across_target_arms_and_models": True,
        "screen_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "generation": dict(generation),
        "bridge_runtime": children[0].receipt.get("bridge_runtime"),
        "source_ids": source_ids,
        "task_ids": [child.task_id for child in children],
        "children": [
            {
                "source_id": child.source_id,
                "task_id": child.task_id,
                "build_dir": str(child.root),
                "build_config_sha256": child.config_sha256,
                "build_receipt_sha256": child.receipt_sha256,
                "frozen_dataset_sha256": child.dataset_sha256,
            }
            for child in children
        ],
        "accounting": {
            "calls": call_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "accounted_cost_usd": str(cost),
        },
        "frozen_dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
    }
    receipt_path = destination / "build_receipt.json"
    atomic_write_json(receipt_path, receipt_payload)
    return dataset_path, receipt_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--children", required=True, help="comma-separated child dirs")
    parser.add_argument("--screen-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    children = tuple(item.strip() for item in args.children.split(",") if item.strip())
    try:
        dataset, receipt = compose_frozen_builds(
            child_dirs=children,
            screen_plan_path=args.screen_plan,
            output_dir=args.output_dir,
        )
    except (CompositionError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(f"composed dataset: {dataset}")
    print(f"dataset_sha256: {sha256_file(dataset)}")
    print(f"receipt_sha256: {sha256_file(receipt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

