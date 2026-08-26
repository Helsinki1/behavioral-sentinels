"""Offline, standard-library-only tests for external domain boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments12.domains.base import (
    ArtifactIntegrityError,
    DomainUnavailableError,
    DomainValidationError,
    PermissionGateError,
)
from experiments12.domains.evolving_intent import EvolvingIntentAdapter
from experiments12.domains import turnbench_ms
from experiments12.domains.turnbench_ms import (
    PINNED_COMMIT as TURNBENCH_COMMIT,
    REPOSITORY as TURNBENCH_REPOSITORY,
    REQUIRED_RUNTIME_PATHS,
    ROOT_ENVIRONMENT_VARIABLE,
    TurnBenchMSAdapter,
)


def _write_json(path: Path, value: object) -> str:
    data = json.dumps(value, sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _paired_dataset(*, t7_count: int = 7, t7_label: str = "42") -> dict[str, object]:
    return {
        "tasks": [
            {
                "task_id": "gsm8k-12",
                "condition": "t1",
                "turns": ["Solve the fully specified problem."],
                "label": "42",
                "gold": "must never enter observer data",
                "metadata": {"change_plan": ["secret"], "future": "secret"},
            },
            {
                "task_id": "gsm8k-12",
                "condition": "t7",
                "turns": [
                    {
                        "user_message": f"Public user turn {index}",
                        "future": "discarded",
                        "gold_answer": "discarded",
                    }
                    for index in range(1, t7_count + 1)
                ],
                "label": t7_label,
                "change_plan": ["secret transition"],
                "predecessor_functions": ["secret latent structure"],
            },
        ]
    }


class EvolvingIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_evolving_")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_loads_paired_t1_t7_and_builds_prefix_only_checkpoint(self) -> None:
        dataset_path = self.root / "frozen.json"
        dataset_hash = _write_json(dataset_path, _paired_dataset())
        ids_path = self.root / "task_ids.json"
        ids_hash = _write_json(ids_path, ["gsm8k-12"])

        adapter = EvolvingIntentAdapter(
            dataset_path,
            expected_sha256=dataset_hash,
            task_ids_path=ids_path,
            expected_task_ids_sha256=ids_hash,
        )
        tasks = adapter.load_tasks()
        self.assertEqual([(task.condition, len(task.turns)) for task in tasks], [("t1", 1), ("t7", 7)])
        self.assertEqual(len(adapter.input_artifacts), 2)
        self.assertEqual(adapter.source_sha256, dataset_hash)

        evolving = tasks[1]
        checkpoint = evolving.checkpoint(("agent answer 1", "agent answer 2"))
        observer = checkpoint.to_observer_dict()
        serialized = json.dumps(observer, sort_keys=True).lower()
        self.assertEqual(checkpoint.after_turn, 2)
        self.assertEqual(len(checkpoint.turns), 2)
        self.assertNotIn("public user turn 3", serialized)
        for secret_key in (
            "change_plan",
            "future",
            "gold",
            "label",
            "predecessor",
            "evaluation_label",
        ):
            self.assertNotIn(secret_key, serialized)
        self.assertEqual(evolving.evaluation_label, "42")
        self.assertNotIn("42", serialized)
        self.assertNotIn("evaluation_label", evolving.manifest_record())

    def test_rejects_bad_turn_count_and_missing_pair(self) -> None:
        wrong_turns = self.root / "wrong-turns.json"
        _write_json(wrong_turns, _paired_dataset(t7_count=6))
        with self.assertRaisesRegex(DomainValidationError, "exactly 7"):
            EvolvingIntentAdapter(wrong_turns).load_tasks()

        missing_pair = self.root / "missing-pair.json"
        value = _paired_dataset()
        value["tasks"] = value["tasks"][:1]  # type: ignore[index]
        _write_json(missing_pair, value)
        with self.assertRaisesRegex(DomainValidationError, "paired t1 and t7"):
            EvolvingIntentAdapter(missing_pair).load_tasks()

    def test_preserves_condition_specific_labels(self) -> None:
        dataset = self.root / "different-condition-labels.json"
        _write_json(dataset, _paired_dataset(t7_label="43"))
        tasks = EvolvingIntentAdapter(dataset).load_tasks()
        self.assertEqual([task.evaluation_label for task in tasks], ["42", "43"])

    def test_hash_and_task_id_manifest_fail_closed(self) -> None:
        dataset_path = self.root / "frozen.json"
        _write_json(dataset_path, _paired_dataset())
        with self.assertRaises(ArtifactIntegrityError):
            EvolvingIntentAdapter(dataset_path, expected_sha256="0" * 64)

        ids_path = self.root / "task_ids.json"
        _write_json(ids_path, ["different-id"])
        with self.assertRaisesRegex(ArtifactIntegrityError, "differ from manifest"):
            EvolvingIntentAdapter(dataset_path, task_ids_path=ids_path).load_tasks()

    def test_upstream_import_has_no_ambient_or_environment_fallback(self) -> None:
        dataset_path = self.root / "frozen.json"
        _write_json(dataset_path, _paired_dataset())
        adapter = EvolvingIntentAdapter(dataset_path)
        with self.assertRaisesRegex(DomainUnavailableError, "explicit upstream_root"):
            adapter.import_upstream_simulator()


class TurnBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_turnbench_")
        self.root = Path(self.temp.name)
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        self.receipt = self.root / "permission.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_receipt(self, **overrides: object) -> None:
        value: dict[str, object] = {
            "schema_version": 1,
            "repository": TURNBENCH_REPOSITORY,
            "commit": TURNBENCH_COMMIT,
            "permission_granted": True,
            "scope": ["research_use", "local_execution"],
            "granted_by": "upstream rights holder",
            "granted_at": "2026-08-26",
            "evidence_sha256": "e" * 64,
        }
        value.update(overrides)
        _write_json(self.receipt, value)

    def _prepare_checkout(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for index, relative_path in enumerate(turnbench_ms.PINNED_PATH_SHA256, start=1):
            path = self.checkout / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f"fixture-{index}".encode("utf-8")
            path.write_bytes(data)
            hashes[relative_path] = hashlib.sha256(data).hexdigest()
        for relative_path in REQUIRED_RUNTIME_PATHS:
            path = self.checkout / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# external fixture\n", encoding="utf-8")
        git_dir = self.checkout / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text(TURNBENCH_COMMIT + "\n", encoding="utf-8")
        return hashes

    def test_requires_affirmative_receipt_before_checkout(self) -> None:
        missing = self.root / "missing-receipt.json"
        with self.assertRaises(PermissionGateError):
            TurnBenchMSAdapter(
                missing,
                environment={ROOT_ENVIRONMENT_VARIABLE: str(self.checkout)},
            )

        self._write_receipt(permission_granted=False)
        with self.assertRaisesRegex(PermissionGateError, "does not affirm"):
            TurnBenchMSAdapter(
                self.receipt,
                environment={ROOT_ENVIRONMENT_VARIABLE: str(self.checkout)},
            )

    def test_requires_explicit_root_environment_variable(self) -> None:
        self._write_receipt()
        with self.assertRaisesRegex(PermissionGateError, ROOT_ENVIRONMENT_VARIABLE):
            TurnBenchMSAdapter(self.receipt, environment={})

    def test_verified_checkout_exposes_boundary_and_limitations(self) -> None:
        self._write_receipt()
        fixture_hashes = self._prepare_checkout()
        with patch.object(turnbench_ms, "PINNED_PATH_SHA256", fixture_hashes):
            adapter = TurnBenchMSAdapter(
                self.receipt,
                environment={ROOT_ENVIRONMENT_VARIABLE: str(self.checkout)},
            )

        readiness = adapter.readiness
        self.assertTrue(readiness.ready_for_external_loader)
        self.assertEqual(readiness.checkout_commit, TURNBENCH_COMMIT)
        self.assertFalse(readiness.official_process_labels_available)
        self.assertFalse(readiness.official_process_extractor_available)
        self.assertTrue(any("no license" in note.lower() for note in readiness.notes))
        self.assertTrue(any("not released" in note.lower() for note in readiness.notes))

        boundary = adapter.loader_boundary()
        self.assertEqual(boundary.external_root, str(self.checkout.resolve()))
        self.assertEqual(
            boundary.environment({"SAFE": "1"}),
            {"SAFE": "1", ROOT_ENVIRONMENT_VARIABLE: str(self.checkout.resolve())},
        )
        with self.assertRaisesRegex(DomainUnavailableError, "interactive"):
            adapter.load_tasks()

        (self.checkout / REQUIRED_RUNTIME_PATHS[0]).write_text(
            "# changed after validation\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ArtifactIntegrityError, "SHA256 mismatch"):
            adapter.loader_boundary()

    def test_path_hash_mismatch_fails_closed(self) -> None:
        self._write_receipt()
        self._prepare_checkout()
        one_wrong = dict(turnbench_ms.PINNED_PATH_SHA256)
        first = next(iter(one_wrong))
        one_wrong[first] = "0" * 64
        with patch.object(turnbench_ms, "PINNED_PATH_SHA256", one_wrong):
            with self.assertRaises(ArtifactIntegrityError):
                TurnBenchMSAdapter(
                    self.receipt,
                    environment={ROOT_ENVIRONMENT_VARIABLE: str(self.checkout)},
                )


if __name__ == "__main__":
    unittest.main()
