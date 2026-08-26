from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments12.core.artifacts import atomic_write_jsonl, sha256_file
from experiments12.domains.base import DomainTask, DomainTurn, canonical_json_sha256
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.runner12 import (
    _validate_run_inputs,
    freeze_task_manifest,
    load_task_manifest,
    pair_task_id,
    resolve_declared_tasks,
    task_manifest_rows,
)
from experiments12.spec12 import Stage


ROOT = Path(__file__).resolve().parent.parent


def task(task_id: str, condition: str, turns: int) -> DomainTask:
    source = canonical_json_sha256({"dataset": 12})
    messages = tuple(DomainTurn(i, f"turn {i}") for i in range(1, turns + 1))
    payload = {
        "domain": "evolving_intent_gsm8k",
        "task_id": task_id,
        "condition": condition,
        "turns": [turn.user_message for turn in messages],
        "source": source,
    }
    return DomainTask(
        domain="evolving_intent_gsm8k",
        task_id=task_id,
        condition=condition,
        turns=messages,
        evaluation_label="42",
        source_sha256=source,
        task_sha256=canonical_json_sha256(payload),
    )


class RunnerTests(unittest.TestCase):
    def test_manifest_is_condition_aware_answer_blind_and_write_once(self):
        tasks = (task("12", "t1", 1), task("12", "t7", 7))
        rows = task_manifest_rows(tasks)
        self.assertEqual([row["task_id"] for row in rows], ["12::t1", "12::t7"])
        self.assertFalse(any("label" in row or "turns" in row for row in rows))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            freeze_task_manifest(path, tasks)
            self.assertEqual(list(load_task_manifest(path)), rows)
            with self.assertRaises(FileExistsError):
                freeze_task_manifest(path, tasks)

    def test_resolution_fails_on_changed_content(self):
        original = task("12", "t7", 7)
        rows = task_manifest_rows((original,))
        self.assertEqual(
            resolve_declared_tasks((original,), rows)[
                (original.domain, pair_task_id(original), original.task_sha256)
            ],
            original,
        )
        changed = task("12", "t7", 6)
        with self.assertRaises(ValueError):
            resolve_declared_tasks((changed,), rows)

    def test_source_ids_cannot_alias_condition_separator(self):
        with self.assertRaises(ValueError):
            pair_task_id(task("bad::id", "t1", 1))

    def test_runtime_rejects_adversarial_active_t1_pair_manifest(self):
        t1 = task("12", "t1", 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "tasks.jsonl"
            freeze_task_manifest(task_path, (t1,))
            row = load_task_manifest(task_path)[0]
            cells = make_pair_manifest(
                tasks=(
                    TaskRef(
                        str(row["benchmark"]),
                        str(row["task_id"]),
                        str(row["task_sha256"]),
                    ),
                ),
                models=("gpt-5.6-luna",),
                arms=("active_counter",),
                randomization_seed=7,
            )
            layout = RunLayout.for_run(root / "artifacts", "bad-t1")
            layout.create()
            atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])
            receipt = ArtifactReceipt.from_file(
                "task_manifest", task_path, workspace=ROOT
            )
            manifest = build_manifest(
                run_id="bad-t1",
                stage=Stage.SMOKE,
                repository_root=ROOT,
                pair_manifest_sha256=sha256_file(layout.pairs),
                models=("gpt-5.6-luna",),
                arms=("active_counter",),
                operators=("none",),
                randomization_seed=7,
                benchmark_receipts=(receipt,),
                extra_config={"n_cells": 1},
            )
            write_manifest_once(layout.manifest, manifest)
            with self.assertRaisesRegex(ValueError, "forbidden for t1"):
                _validate_run_inputs(
                    layout=layout,
                    task_manifest_path=task_path,
                    tasks=(t1,),
                )


if __name__ == "__main__":
    unittest.main()
