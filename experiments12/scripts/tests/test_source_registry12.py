from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments12.core.artifacts import atomic_write_json, read_json, sha256_file
from experiments12.source_registry12 import (
    CANONICAL_REALIZED_ALLOCATION_RECEIPTS,
    SourceRegistryError,
    bind_task_allocation,
    load_source_registry,
    normalize_source_id,
    validate_source_allocation_binding,
    validate_source_registry,
)
from experiments12.spec12 import Stage


class SourceRegistryTests(unittest.TestCase):
    def test_canonical_registry_is_globally_disjoint_and_records_actual_smoke(self):
        registry = load_source_registry()
        evolving = registry["benchmarks"]["evolving_intent_gsm8k"]
        self.assertEqual(
            evolving["allocations"]["smoke"]["source_ids"],
            ["12", "36", "40", "43", "50"],
        )
        self.assertEqual(
            evolving["diagnostic_exclusions"],
            [
                {"source_id": "14", "reason_code": "prior_generation_diagnostic"},
                {"source_id": "16", "reason_code": "prior_generation_diagnostic"},
                {"source_id": "49", "reason_code": "structural_screen_rejection"},
            ],
        )
        bfcl_wave = registry["benchmarks"]["bfcl_multi_turn"]["allocations"][
            "smoke"
        ]["waves"]["all_models"]
        self.assertEqual(
            bfcl_wave,
            [
                "multi_turn_base_3",
                "multi_turn_base_4",
                "multi_turn_miss_func_0",
                "multi_turn_miss_param_0",
                "multi_turn_long_context_0",
            ],
        )

    def test_cross_stage_or_reserve_overlap_fails(self):
        registry = load_source_registry()
        changed = deepcopy(registry)
        changed["benchmarks"]["evolving_intent_gsm8k"]["allocations"][
            "calibration"
        ]["source_ids"][0] = "60"
        with self.assertRaisesRegex(SourceRegistryError, "multiple stages"):
            validate_source_registry(changed)

        changed = deepcopy(registry)
        changed["benchmarks"]["bfcl_multi_turn"]["structural_failure_reserve"][
            0
        ] = "multi_turn_base_10"
        with self.assertRaisesRegex(SourceRegistryError, "overlaps"):
            validate_source_registry(changed)

    def test_task_selection_must_exactly_match_stage_or_smoke_wave(self):
        registry = load_source_registry()
        expected = registry["benchmarks"]["evolving_intent_gsm8k"]["allocations"][
            "calibration"
        ]["source_ids"]
        rows = [
            {
                "benchmark": "evolving_intent_gsm8k",
                "source_task_id": f"extracted-gsm8k-test-{source_id}",
                "task_id": f"extracted-gsm8k-test-{source_id}::t7",
            }
            for source_id in expected
        ]
        binding = bind_task_allocation(rows, stage=Stage.CALIBRATION)
        self.assertEqual(binding.source_ids, tuple(expected))
        rows[-1] = {**rows[-1], "source_task_id": "extracted-gsm8k-test-49"}
        with self.assertRaisesRegex(SourceRegistryError, "differ"):
            bind_task_allocation(rows, stage=Stage.CALIBRATION)

        bfcl = registry["benchmarks"]["bfcl_multi_turn"]["allocations"]["smoke"]
        smoke_rows = [
            {
                "benchmark": "bfcl_multi_turn",
                "source_task_id": source_id,
                "task_id": source_id + "::official_native_tools",
            }
            for source_id in bfcl["waves"]["all_models"]
        ]
        wave = bind_task_allocation(
            smoke_rows, stage=Stage.SMOKE, smoke_wave="all_models"
        )
        self.assertEqual(list(wave.source_ids), bfcl["waves"]["all_models"])

    def test_source_normalization_is_strict(self):
        self.assertEqual(
            normalize_source_id(
                "evolving_intent_gsm8k", "extracted-gsm8k-test-0036::t7"
            ),
            "36",
        )
        self.assertEqual(
            normalize_source_id(
                "bfcl_multi_turn", "multi_turn_miss_param_20::official_native_tools"
            ),
            "multi_turn_miss_param_20",
        )
        with self.assertRaises(SourceRegistryError):
            normalize_source_id("evolving_intent_gsm8k", "gsm8k-secret")

    def test_realized_baseline_accepts_only_first_ordered_reserve_replacement(self):
        receipt = Path("experiments12/data_results/inputs/evolving_baseline_screen12.json")
        selected = read_json(receipt)["selected_source_ids"]
        rows = [
            {
                "benchmark": "evolving_intent_gsm8k",
                "source_task_id": f"extracted-gsm8k-test-{source_id}",
                "task_id": f"extracted-gsm8k-test-{source_id}::t7",
            }
            for source_id in selected
        ]
        binding = bind_task_allocation(
            rows,
            stage=Stage.BASELINE_GATE,
            realized_allocation_path=receipt,
        )
        self.assertEqual(binding.realized_allocation_sha256, sha256_file(receipt))
        self.assertEqual(binding.structural_rejection_source_ids, ("122",))
        self.assertEqual(binding.replacement_source_ids, ("1021",))

        with tempfile.TemporaryDirectory(prefix="allocation-receipt12-") as raw:
            changed = read_json(receipt)
            changed["selected_source_ids"][-1] = 1024
            bad = Path(raw) / "skipped-reserve.json"
            atomic_write_json(bad, changed)
            rows[-1] = {
                **rows[-1],
                "source_task_id": "extracted-gsm8k-test-1024",
                "task_id": "extracted-gsm8k-test-1024::t7",
            }
            with self.assertRaisesRegex(SourceRegistryError, "canonical receipt"):
                bind_task_allocation(
                    rows,
                    stage=Stage.BASELINE_GATE,
                    realized_allocation_path=bad,
                )

    def test_realized_calibration_continues_after_consumed_baseline_reserve(self):
        receipt = Path("experiments12/data_results/inputs/evolving_calibration_screen12.json")
        selected = read_json(receipt)["selected_source_ids"]
        rows = [
            {
                "benchmark": "evolving_intent_gsm8k",
                "source_task_id": f"extracted-gsm8k-test-{source_id}",
                "task_id": f"extracted-gsm8k-test-{source_id}::t7",
            }
            for source_id in selected
        ]
        binding = bind_task_allocation(
            rows,
            stage=Stage.CALIBRATION,
            realized_allocation_path=receipt,
        )
        self.assertEqual(binding.realized_allocation_sha256, sha256_file(receipt))
        self.assertEqual(
            binding.structural_rejection_source_ids,
            ("203", "208", "213", "256", "296", "1040", "1044"),
        )
        self.assertEqual(
            binding.replacement_source_ids,
            ("1024", "1025", "1026", "1045", "1047"),
        )
        reproduced = validate_source_allocation_binding(binding.as_dict())
        self.assertEqual(reproduced, binding)

    def test_confirmatory_and_deployment_receipts_advance_one_global_reserve(self):
        with tempfile.TemporaryDirectory(prefix="allocation-later-stages12-") as raw:
            root = Path(raw)
            registry = load_source_registry()
            benchmark = "evolving_intent_gsm8k"
            allocations = registry["benchmarks"][benchmark]["allocations"]

            confirmatory_primary = allocations["confirmatory"]["source_ids"]
            confirmatory_receipt = root / "confirmatory.json"
            atomic_write_json(
                confirmatory_receipt,
                {
                    "schema_version": 1,
                    "purpose": "Select confirmatory tasks outcome-blind.",
                    "candidate_source_ids_in_order": [
                        *confirmatory_primary,
                        "1049",
                    ],
                    "target_valid_tasks": len(confirmatory_primary),
                    "maximum_attempts_per_candidate": 1,
                    "acceptance": {
                        "target_model_outcomes_available_at_selection": False
                    },
                    "structural_rejections": {
                        confirmatory_primary[0]: "structural renderer failure"
                    },
                    "selected_source_ids": [*confirmatory_primary[1:], "1049"],
                },
            )

            deployment_primary = allocations["deployment"]["source_ids"]
            deployment_receipt = root / "deployment.json"
            atomic_write_json(
                deployment_receipt,
                {
                    "schema_version": 1,
                    "purpose": "Select deployment tasks outcome-blind.",
                    "candidate_source_ids_in_order": [*deployment_primary, "1058"],
                    "target_valid_tasks": len(deployment_primary),
                    "maximum_attempts_per_candidate": 1,
                    "acceptance": {
                        "target_model_outcomes_available_at_selection": False
                    },
                    "structural_rejections": {
                        deployment_primary[0]: "structural renderer failure"
                    },
                    "selected_source_ids": [*deployment_primary[1:], "1058"],
                },
            )

            receipt_overrides = {
                (benchmark, "confirmatory"): confirmatory_receipt,
                (benchmark, "deployment"): deployment_receipt,
            }
            with patch.dict(
                CANONICAL_REALIZED_ALLOCATION_RECEIPTS,
                receipt_overrides,
                clear=False,
            ):
                confirmatory = bind_task_allocation(
                    [
                        {"benchmark": benchmark, "source_task_id": source}
                        for source in [*confirmatory_primary[1:], "1049"]
                    ],
                    stage=Stage.CONFIRMATORY,
                    realized_allocation_path=confirmatory_receipt,
                )
                self.assertEqual(confirmatory.replacement_source_ids, ("1049",))
                self.assertEqual(
                    validate_source_allocation_binding(confirmatory.as_dict()),
                    confirmatory,
                )

                deployment = bind_task_allocation(
                    [
                        {"benchmark": benchmark, "source_task_id": source}
                        for source in [*deployment_primary[1:], "1058"]
                    ],
                    stage="deployment",
                    realized_allocation_path=deployment_receipt,
                )
                self.assertEqual(deployment.replacement_source_ids, ("1058",))
                self.assertEqual(
                    validate_source_allocation_binding(deployment.as_dict()),
                    deployment,
                )


if __name__ == "__main__":
    unittest.main()
