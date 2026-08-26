from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from experiments12.cli12 import (
    _confirmatory_analysis_lock,
    _dotenv,
    _evolving_provenance_receipts,
    _load_tasks,
    _reject_active_t1_rows,
    main,
)
from experiments12.core.artifacts import atomic_write_json, atomic_write_jsonl, sha256_file
from experiments12.passive_spec12 import effective_passive_method_names
from experiments12.source_registry12 import load_source_registry
from experiments12.domains.evolving_intent import PINNED_COMMIT
from experiments12.spec12 import Stage


class CliTests(unittest.TestCase):
    def test_dotenv_parser_returns_values_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("# comment\nOPENAI_API_KEY='secret'\nFIREWORKS_API_KEY=other\n")
            values = _dotenv(path)
            self.assertEqual(set(values), {"OPENAI_API_KEY", "FIREWORKS_API_KEY"})
            self.assertEqual(values["OPENAI_API_KEY"], "secret")

    def test_task_loader_hashes_rows_if_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text('{"benchmark":"b","task_id":"1","turns":["x"]}\n')
            tasks = _load_tasks(path)
            self.assertEqual(tasks[0].task_id, "1")
            self.assertEqual(len(tasks[0].task_sha256), 64)

    def test_initialization_rejects_active_arms_for_t1(self):
        rows = [
            {
                "benchmark": "evolving_intent_gsm8k",
                "task_id": "gsm8k-12::t1",
                "condition": "t1",
            }
        ]
        _reject_active_t1_rows(rows, ("clean",))
        with self.assertRaisesRegex(ValueError, "forbidden for t1"):
            _reject_active_t1_rows(rows, ("clean", "active_counter"))

    def test_evolving_dataset_and_build_receipt_are_bound_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.json"
            dataset.write_text('{"tasks":[]}', encoding="utf-8")
            digest = sha256_file(dataset)
            receipt = root / "build_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "benchmark": "evolving_intent_gsm8k",
                        "upstream_commit": PINNED_COMMIT,
                        "shared_across_target_arms_and_models": True,
                        "frozen_dataset": {"sha256": digest},
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "task_id": "x::t7",
                    "source_sha256": digest,
                }
            ]
            frozen = _evolving_provenance_receipts(
                rows,
                dataset_path=str(dataset),
                build_receipt_path=str(receipt),
            )
            self.assertEqual(
                [item.name for item in frozen],
                ["evolving_rendered_dataset", "evolving_build_receipt"],
            )
            with self.assertRaisesRegex(ValueError, "require"):
                _evolving_provenance_receipts(
                    rows, dataset_path=str(dataset), build_receipt_path=None
                )
            bad = json.loads(receipt.read_text(encoding="utf-8"))
            bad["frozen_dataset"]["sha256"] = "f" * 64
            receipt.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not attest"):
                _evolving_provenance_receipts(
                    rows,
                    dataset_path=str(dataset),
                    build_receipt_path=str(receipt),
                )

    def test_scientific_init_failure_creates_no_run_or_ledger(self):
        with tempfile.TemporaryDirectory(prefix="cli-launch-gate12-") as raw:
            root = Path(raw)
            tasks = root / "calibration.jsonl"
            sources = load_source_registry()["benchmarks"][
                "evolving_intent_gsm8k"
            ]["allocations"]["calibration"]["source_ids"]
            atomic_write_jsonl(
                tasks,
                [
                    {
                        "benchmark": "evolving_intent_gsm8k",
                        "source_task_id": f"extracted-gsm8k-test-{source}",
                        "task_id": f"extracted-gsm8k-test-{source}::t7",
                        "condition": "t7",
                    }
                    for source in sources
                ],
            )
            artifacts = root / "runs"
            result = main(
                [
                    "init",
                    "--run-id",
                    "missing-planning-lock",
                    "--stage",
                    "calibration",
                    "--tasks",
                    str(tasks),
                    "--source-registry",
                    "experiments12/source_allocation12.json",
                    "--models",
                    "gpt-5.6-luna",
                    "--arms",
                    "clean,active_recompute",
                    "--artifacts",
                    str(artifacts),
                ]
            )
            self.assertEqual(result, 2)
            self.assertFalse((artifacts / "missing-planning-lock").exists())
            self.assertFalse((artifacts / "_global_budget.sqlite3").exists())

    def test_confirmatory_analysis_lock_is_exact_and_benchmark_scoped(self):
        with tempfile.TemporaryDirectory(prefix="cli-threshold-lock12-") as raw:
            root = Path(raw)
            threshold = root / "thresholds.json"
            atomic_write_json(
                threshold,
                {
                    "artifact_type": "locked_fixed_rate_thresholds",
                    "source_manifest_sha256": "a" * 64,
                    "required_passive_methods": list(effective_passive_method_names()),
                    "calibration_source_tasks": [
                        {
                            "benchmark": "bfcl_multi_turn",
                            "source_task_id": "multi_turn_base_20",
                        }
                    ],
                },
            )
            rows = [
                {
                    "benchmark": "bfcl_multi_turn",
                    "source_task_id": "multi_turn_base_40",
                    "task_id": "multi_turn_base_40::official_native_tools",
                }
            ]
            lock = _confirmatory_analysis_lock(
                stage=Stage.CONFIRMATORY,
                task_rows=rows,
                thresholds_path=threshold,
            )
            self.assertEqual(
                lock,
                {
                    "threshold_artifact_sha256": sha256_file(threshold),
                    "calibration_manifest_sha256": "a" * 64,
                },
            )
            with self.assertRaisesRegex(ValueError, "benchmark differs"):
                _confirmatory_analysis_lock(
                    stage=Stage.CONFIRMATORY,
                    task_rows=[
                        {
                            **rows[0],
                            "benchmark": "evolving_intent_gsm8k",
                        }
                    ],
                    thresholds_path=threshold,
                )


if __name__ == "__main__":
    unittest.main()
