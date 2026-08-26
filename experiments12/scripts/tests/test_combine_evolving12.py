from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments12.build_evolving12 import PINNED_COMMIT, SEED
from experiments12.combine_evolving12 import CompositionError, compose_frozen_builds
from experiments12.core.artifacts import atomic_write_json, read_json, sha256_file, sha256_json
from experiments12.domains.evolving_intent import DOMAIN, EvolvingIntentAdapter


class EvolvingCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="evolving_compose_")
        self.root = Path(self.temp.name)
        self.shared = self.root / "locked.txt"
        self.shared.write_text("locked", encoding="utf-8")
        self.plan = self.root / "screen.json"
        atomic_write_json(
            self.plan,
            {
                "candidate_source_ids_in_order": [12, 36, 40],
                "target_valid_tasks": 3,
                "generator_model": "generator",
                "generator_reasoning_effort": "none",
                "judge_model": "judge",
                "judge_reasoning_effort": "none",
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def child(self, source_id: int, *, generator: str = "generator") -> Path:
        root = self.root / f"child-{source_id}-{generator}"
        root.mkdir()
        task_id = f"task-{source_id}"
        generation = {
            "generator_model": generator,
            "generator_reasoning_effort": "none",
            "judge_model": "judge",
            "judge_reasoning_effort": "none",
            "seed": SEED,
        }
        runtime = {"python_version": "3.12.0", "attestation_sha256": "a" * 64}
        config = {
            "schema_version": 1,
            "benchmark": DOMAIN,
            "upstream_commit": PINNED_COMMIT,
            "seed": SEED,
            "source_ids": [source_id],
            "task_ids": [task_id],
            "generation": generation,
            "provider_compatibility": {},
            "bridge": {
                "path": str(self.shared),
                "sha256": sha256_file(self.shared),
                "runtime": runtime,
            },
            "inputs": [
                {"role": "fixture", "path": str(self.shared), "sha256": sha256_file(self.shared)}
            ],
            "prompt_files": [],
            "shared_across_target_arms_and_models": True,
            "target_arm": None,
            "target_model": None,
        }
        atomic_write_json(root / "build_config.json", config)
        dataset = {
            "tasks": [
                {"task_id": task_id, "condition": "t1", "turns": ["one"], "label": "1"},
                {
                    "task_id": task_id,
                    "condition": "t7",
                    "turns": [f"turn {index}" for index in range(1, 8)],
                    "label": "1",
                },
            ]
        }
        atomic_write_json(root / "evolving_intent_gsm8k_frozen.json", dataset)
        receipt = {
            "build_sha256": sha256_json(config),
            "generation": generation,
            "bridge_runtime": runtime,
            "calls": [
                {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cached_input_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                    "accounted_cost_usd": "0.01",
                }
            ],
            "frozen_dataset": {
                "sha256": sha256_file(root / "evolving_intent_gsm8k_frozen.json")
            },
        }
        atomic_write_json(root / "build_receipt.json", receipt)
        return root

    def test_compose_valid_children_and_reject_tampering(self) -> None:
        children = [self.child(value) for value in (12, 36, 40)]
        dataset, receipt = compose_frozen_builds(
            child_dirs=children,
            screen_plan_path=self.plan,
            output_dir=self.root / "combined",
        )
        self.assertEqual(len(EvolvingIntentAdapter(dataset).load_tasks()), 6)
        self.assertEqual(read_json(receipt)["accounting"]["calls"], 3)
        self.assertEqual(read_json(receipt)["accounting"]["accounted_cost_usd"], "0.03")

        tampered = self.root / "tampered-child"
        source = children[0]
        tampered.mkdir()
        for path in source.iterdir():
            (tampered / path.name).write_bytes(path.read_bytes())
        payload = read_json(tampered / "evolving_intent_gsm8k_frozen.json")
        payload["tasks"][0]["turns"] = ["changed"]
        atomic_write_json(tampered / "evolving_intent_gsm8k_frozen.json", payload)
        with self.assertRaisesRegex(CompositionError, "does not bind its dataset"):
            compose_frozen_builds(
                child_dirs=(tampered, children[1], children[2]),
                screen_plan_path=self.plan,
                output_dir=self.root / "must-not-exist",
            )

    def test_rejects_wrong_order_or_generation(self) -> None:
        children = [self.child(value) for value in (12, 36)]
        incompatible = self.child(40, generator="other")
        with self.assertRaisesRegex(CompositionError, "generation/runtime"):
            compose_frozen_builds(
                child_dirs=(*children, incompatible),
                screen_plan_path=self.plan,
                output_dir=self.root / "incompatible",
            )
        with self.assertRaisesRegex(CompositionError, "ordered"):
            compose_frozen_builds(
                child_dirs=(children[1], children[0], self.child(40)),
                screen_plan_path=self.plan,
                output_dir=self.root / "wrong-order",
            )


if __name__ == "__main__":
    unittest.main()
