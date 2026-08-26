from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
)
from experiments12.core.budget import BudgetLedger
from experiments12.core.schemas import PairKey
from experiments12.manifest12 import RunLayout
from experiments12.models12 import TARGET_MODEL_NAMES
from experiments12.pairing12 import JobCell
from experiments12.plan12 import main as planning_main
from experiments12.planning_lock12 import (
    BASELINE_PROFILE_VERSION,
    BASELINE_PROFILE_TYPE,
    PlanningLockError,
    assert_scientific_launch,
    build_baseline_resource_profile,
    build_projection_lock,
    freeze_projection_lock,
    validate_baseline_resource_profile,
    validate_projection_lock_static,
)
from experiments12.source_registry12 import bind_task_allocation, load_source_registry
from experiments12.spec12 import Stage


class PlanningLockTests(unittest.TestCase):
    def _profile(self, path: Path, *, benchmark="evolving_intent_gsm8k") -> Path:
        registry = load_source_registry()
        sources = sorted(
            registry["benchmarks"][benchmark]["allocations"]["baseline_gate"][
                "source_ids"
            ]
        )
        condition = "t7" if benchmark == "evolving_intent_gsm8k" else "official_native_tools"
        allocation = bind_task_allocation(
            [
                {"benchmark": benchmark, "source_task_id": source}
                for source in sources
            ],
            stage=Stage.BASELINE_GATE,
        )
        profiles = []
        for model in sorted(TARGET_MODEL_NAMES):
            profiles.append(
                {
                    "model": model,
                    "benchmark": benchmark,
                    "condition": condition,
                    "source_allocation": allocation.as_dict(),
                    "n_tasks": 20,
                    "n_success": 12,
                    "success_rate": "0.6",
                    "source_task_ids": sources,
                    "p95_calls": 8,
                    "p95_input_tokens": 4000,
                    "p95_output_tokens": 600,
                    "p95_checkpoints": 6,
                    "planning_input_tokens": 5000,
                    "planning_output_tokens": 750,
                }
            )
        value = validate_baseline_resource_profile(
            {
                "schema_version": BASELINE_PROFILE_VERSION,
                "artifact_type": BASELINE_PROFILE_TYPE,
                "created_from_complete_clean_baseline_trajectories": True,
                "quantile_rule": "nearest_rank_p95",
                "profile_headroom_multiplier": "1.25",
                "source_manifest_sha256s": ["a" * 64],
                "profiles": profiles,
            }
        )
        atomic_write_json(path, value)
        return path

    def test_build_validate_and_launch_check_reproduce_cost_and_sample_lock(self):
        with tempfile.TemporaryDirectory(prefix="planning-lock12-") as raw:
            root = Path(raw)
            profile = self._profile(root / "profile.json")
            ledger = root / "ledger.sqlite3"
            BudgetLedger(ledger)
            lock = build_projection_lock(
                baseline_profile_path=profile,
                registry_path=Path("experiments12/source_allocation12.json"),
                ledger_path=ledger,
                stage=Stage.CONFIRMATORY,
                benchmark="evolving_intent_gsm8k",
                models=TARGET_MODEL_NAMES,
                arms=("clean", "active_recompute"),
                operators=("none",),
            )
            self.assertEqual(lock["sample_size"]["planned_n_tasks"], 56)
            self.assertEqual(lock["sample_size"]["required_n_tasks"], 54)
            self.assertTrue(all(row["fits_stage_scope"] for row in lock["provider_projections"]))
            lock_path = root / "lock.json"
            freeze_projection_lock(
                lock_path,
                lock,
                baseline_profile_path=profile,
                registry_path=Path("experiments12/source_allocation12.json"),
            )
            registry = load_source_registry()
            sources = registry["benchmarks"]["evolving_intent_gsm8k"]["allocations"][
                "confirmatory"
            ]["source_ids"]
            rows = [
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "source_task_id": f"extracted-gsm8k-test-{source}",
                    "task_id": f"extracted-gsm8k-test-{source}::t7",
                }
                for source in sources
            ]
            binding = assert_scientific_launch(
                task_rows=rows,
                stage=Stage.CONFIRMATORY,
                models=TARGET_MODEL_NAMES,
                arms=("clean", "active_recompute"),
                operators=("none",),
                replicates=1,
                ledger_path=ledger,
                registry_path=Path("experiments12/source_allocation12.json"),
                projection_lock_path=lock_path,
                baseline_profile_path=profile,
            )
            self.assertEqual(binding.projection_lock_sha256, sha256_file(lock_path))

            changed = deepcopy(read_json(lock_path))
            changed["provider_projections"][0]["projected_usd"] = "0.000001"
            with self.assertRaisesRegex(PlanningLockError, "does not reproduce"):
                validate_projection_lock_static(
                    changed,
                    baseline_profile_path=profile,
                    registry_path=Path("experiments12/source_allocation12.json"),
                )

    def test_projection_fails_before_lock_when_remaining_budget_is_too_small(self):
        with tempfile.TemporaryDirectory(prefix="planning-budget12-") as raw:
            root = Path(raw)
            profile = self._profile(root / "profile.json")
            ledger_path = root / "ledger.sqlite3"
            BudgetLedger(
                ledger_path,
                operational_caps_usd={"openai": Decimal("0.01"), "fireworks": Decimal("0.01")},
            )
            with self.assertRaisesRegex(PlanningLockError, "exceeds"):
                build_projection_lock(
                    baseline_profile_path=profile,
                    registry_path=Path("experiments12/source_allocation12.json"),
                    ledger_path=ledger_path,
                    stage=Stage.CONFIRMATORY,
                    benchmark="evolving_intent_gsm8k",
                    models=TARGET_MODEL_NAMES,
                    arms=("clean", "active_recompute"),
                    operators=("none",),
                )

    def test_baseline_needs_no_profile_but_later_stages_need_both_locks(self):
        with tempfile.TemporaryDirectory(prefix="planning-stage-gates12-") as raw:
            root = Path(raw)
            registry_path = Path("experiments12/source_allocation12.json")
            registry = load_source_registry()

            baseline_receipt = Path("experiments12/evolving_baseline_screen12.json")
            baseline_rows = [
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "source_task_id": str(source),
                }
                for source in read_json(baseline_receipt)["selected_source_ids"]
            ]
            baseline = assert_scientific_launch(
                task_rows=baseline_rows,
                stage=Stage.BASELINE_GATE,
                models=("gpt-5.6-luna",),
                arms=("clean",),
                operators=("none",),
                replicates=1,
                ledger_path=root / "absent-ledger.sqlite3",
                registry_path=registry_path,
                projection_lock_path=None,
                baseline_profile_path=None,
                realized_allocation_path=baseline_receipt,
            )
            self.assertEqual(baseline.stage, Stage.BASELINE_GATE.value)
            self.assertFalse((root / "absent-ledger.sqlite3").exists())

            for stage_name, allocation_stage, design_family in (
                (Stage.CALIBRATION, None, "observer_effect"),
                (Stage.CONFIRMATORY, None, "observer_effect"),
                (Stage.CONFIRMATORY, "deployment", "deployment"),
            ):
                split = stage_name.value if allocation_stage is None else allocation_stage
                rows = [
                    {
                        "benchmark": "evolving_intent_gsm8k",
                        "source_task_id": source,
                    }
                    for source in registry["benchmarks"]["evolving_intent_gsm8k"]
                    ["allocations"][split]["source_ids"]
                ]
                with self.assertRaisesRegex(
                    PlanningLockError, "requires baseline profile and cost/sample-size lock"
                ):
                    assert_scientific_launch(
                        task_rows=rows,
                        stage=stage_name,
                        allocation_stage=allocation_stage,
                        design_family=design_family,
                        models=("gpt-5.6-luna",),
                        arms=(
                            ("clean", "active_recompute")
                            if design_family == "observer_effect"
                            else ("active_recompute", "turn_clock")
                        ),
                        operators=(
                            ("none",)
                            if design_family == "observer_effect"
                            else ("none", "lossy_compaction")
                        ),
                        replicates=1,
                        ledger_path=root / "absent-ledger.sqlite3",
                        registry_path=registry_path,
                        projection_lock_path=None,
                        baseline_profile_path=None,
                    )
            self.assertFalse((root / "absent-ledger.sqlite3").exists())

    def test_calibration_projection_and_launch_bind_realized_replacements(self):
        with tempfile.TemporaryDirectory(prefix="planning-realized-cal12-") as raw:
            root = Path(raw)
            registry_path = Path("experiments12/source_allocation12.json")
            profile = self._profile(root / "profile.json")
            ledger = root / "ledger.sqlite3"
            BudgetLedger(ledger)
            receipt = Path("experiments12/evolving_calibration_screen12.json")
            selected = read_json(receipt)["selected_source_ids"]
            lock = build_projection_lock(
                baseline_profile_path=profile,
                registry_path=registry_path,
                ledger_path=ledger,
                stage=Stage.CALIBRATION,
                benchmark="evolving_intent_gsm8k",
                models=("gpt-5.6-luna",),
                arms=("clean", "active_recompute"),
                operators=("none",),
                realized_allocation_path=receipt,
            )
            self.assertEqual(lock["source_allocation"]["source_ids"], [str(x) for x in selected])
            lock_path = root / "calibration-lock.json"
            freeze_projection_lock(
                lock_path,
                lock,
                baseline_profile_path=profile,
                registry_path=registry_path,
                realized_allocation_path=receipt,
            )
            # Once frozen, the tracked canonical receipt is enough to
            # reproduce the lock without an out-of-band path override.
            validate_projection_lock_static(
                read_json(lock_path),
                baseline_profile_path=profile,
                registry_path=registry_path,
            )
            rows = [
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "source_task_id": str(source),
                }
                for source in selected
            ]
            binding = assert_scientific_launch(
                task_rows=rows,
                stage=Stage.CALIBRATION,
                models=("gpt-5.6-luna",),
                arms=("clean", "active_recompute"),
                operators=("none",),
                replicates=1,
                ledger_path=ledger,
                registry_path=registry_path,
                projection_lock_path=lock_path,
                baseline_profile_path=profile,
                realized_allocation_path=receipt,
            )
            self.assertEqual(
                binding.allocation.replacement_source_ids,
                ("1024", "1025", "1026", "1045", "1047"),
            )

    def test_deployment_uses_confirmatory_cap_and_distinct_deployment_allocation(self):
        with tempfile.TemporaryDirectory(prefix="planning-deployment12-") as raw:
            root = Path(raw)
            profile = self._profile(root / "profile.json")
            ledger = root / "ledger.sqlite3"
            BudgetLedger(ledger)
            methods = ("active_recompute", "turn_clock")
            operators = ("none", "lossy_compaction")
            lock = build_projection_lock(
                baseline_profile_path=profile,
                registry_path=Path("experiments12/source_allocation12.json"),
                ledger_path=ledger,
                stage=Stage.CONFIRMATORY,
                allocation_stage="deployment",
                design_family="deployment",
                benchmark="evolving_intent_gsm8k",
                models=TARGET_MODEL_NAMES,
                arms=methods,
                operators=operators,
            )
            self.assertEqual(lock["stage"], "confirmatory")
            self.assertEqual(lock["allocation_stage"], "deployment")
            self.assertEqual(lock["sample_size"]["planned_n_tasks"], 40)
            self.assertEqual(lock["sample_size"]["required_n_tasks"], 38)
            lock_path = root / "deployment-lock.json"
            freeze_projection_lock(
                lock_path,
                lock,
                baseline_profile_path=profile,
                registry_path=Path("experiments12/source_allocation12.json"),
            )
            sources = load_source_registry()["benchmarks"][
                "evolving_intent_gsm8k"
            ]["allocations"]["deployment"]["source_ids"]
            rows = [
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "source_task_id": f"extracted-gsm8k-test-{source}",
                    "task_id": f"extracted-gsm8k-test-{source}::t7",
                }
                for source in sources
            ]
            binding = assert_scientific_launch(
                task_rows=rows,
                stage=Stage.CONFIRMATORY,
                allocation_stage="deployment",
                design_family="deployment",
                models=TARGET_MODEL_NAMES,
                arms=methods,
                operators=operators,
                replicates=1,
                ledger_path=ledger,
                registry_path=Path("experiments12/source_allocation12.json"),
                projection_lock_path=lock_path,
                baseline_profile_path=profile,
            )
            self.assertEqual(binding.allocation.stage, "deployment")
            with self.assertRaisesRegex(PlanningLockError, "differs"):
                assert_scientific_launch(
                    task_rows=rows,
                    stage=Stage.CONFIRMATORY,
                    allocation_stage="deployment",
                    design_family="deployment",
                    models=TARGET_MODEL_NAMES,
                    arms=methods,
                    operators=("none", "public_state_reground"),
                    replicates=1,
                    ledger_path=ledger,
                    registry_path=Path("experiments12/source_allocation12.json"),
                    projection_lock_path=lock_path,
                    baseline_profile_path=profile,
                )

    def test_profile_builder_uses_complete_clean_baseline_calls_and_p95(self):
        with tempfile.TemporaryDirectory(prefix="planning-profile12-") as raw:
            root = Path(raw)
            layout = RunLayout.for_run(root / "runs", "baseline")
            layout.create()
            registry = load_source_registry()
            sources = [
                str(value)
                for value in read_json(
                    Path("experiments12/evolving_baseline_screen12.json")
                )["selected_source_ids"]
            ]
            cells = []
            for index, source in enumerate(sources):
                cell = JobCell(
                    cell_id=f"cell-{index}",
                    block_id=f"block-{index}",
                    block_position=0,
                    pair_key=PairKey(
                        model="gpt-5.6-luna",
                        domain="evolving_intent_gsm8k",
                        task_id=f"extracted-gsm8k-test-{source}::t7",
                        task_sha256=(f"{index:064x}")[-64:],
                    ),
                    arm="clean",
                    operator="none",
                    seed=index,
                )
                cells.append(cell)
            atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in cells])
            atomic_write_json(
                layout.manifest,
                {
                    "stage": "baseline_gate",
                    "pair_manifest_sha256": sha256_file(layout.pairs),
                },
            )
            for index, cell in enumerate(cells):
                atomic_write_json(
                    layout.trajectories / f"{cell.cell_id}.json",
                    {
                        "complete": True,
                        "arm": "clean",
                        "model": "gpt-5.6-luna",
                        "domain": "evolving_intent_gsm8k",
                        "condition": "t7",
                        "evaluation": {"success": index < 12},
                        "checkpoint_turns": [1, 2, 3, 4, 5, 6],
                        "task_records": [
                            {
                                "call": {
                                    "call_event_ids": [f"event-{index}"],
                                    "usage": {
                                        "input_tokens": 100 + index,
                                        "output_tokens": 20 + index,
                                    },
                                }
                            }
                        ],
                    },
                )
            profile = build_baseline_resource_profile((layout,))
            row = profile["profiles"][0]
            self.assertEqual(row["n_tasks"], 20)
            self.assertEqual(row["n_success"], 12)
            self.assertEqual(row["success_rate"], "0.6")
            self.assertEqual(
                row["source_allocation"]["replacement_source_ids"], ["1021"]
            )
            self.assertEqual(row["p95_input_tokens"], 118)
            self.assertEqual(row["planning_input_tokens"], 148)
            frozen_profile_path = root / "baseline-profile-v2.json"
            self.assertEqual(
                planning_main(
                    [
                        "profile",
                        "--artifacts-root",
                        str(root / "runs"),
                        "--run-id",
                        "baseline",
                        "--output",
                        str(frozen_profile_path),
                    ]
                ),
                0,
            )
            frozen_row = read_json(frozen_profile_path)["profiles"][0]
            self.assertEqual(
                (frozen_row["n_success"], frozen_row["success_rate"]),
                (12, "0.6"),
            )
            corrupted = deepcopy(profile)
            corrupted["profiles"][0]["success_rate"] = "0.65"
            with self.assertRaisesRegex(PlanningLockError, "does not reproduce"):
                validate_baseline_resource_profile(corrupted)
            first_path = layout.trajectories / "cell-0.json"
            first = read_json(first_path)
            first.pop("evaluation")
            atomic_write_json(first_path, first)
            with self.assertRaisesRegex(PlanningLockError, "binary official success"):
                build_baseline_resource_profile((layout,))
            first["evaluation"] = {"success": True}
            atomic_write_json(first_path, first)
            (layout.trajectories / "cell-0.json").unlink()
            with self.assertRaisesRegex(PlanningLockError, "exactly cover"):
                build_baseline_resource_profile((layout,))

    def test_planning_cli_freezes_and_revalidates_lock_without_provider_calls(self):
        with tempfile.TemporaryDirectory(prefix="planning-cli12-") as raw:
            root = Path(raw)
            profile = self._profile(root / "profile.json")
            self.assertEqual(
                planning_main(
                    [
                        "validate-profile",
                        "--profile",
                        str(profile),
                        "--registry",
                        "experiments12/source_allocation12.json",
                    ]
                ),
                0,
            )
            ledger = root / "ledger.sqlite3"
            BudgetLedger(ledger)
            lock = root / "calibration-lock.json"
            common = [
                "--baseline-profile",
                str(profile),
                "--registry",
                "experiments12/source_allocation12.json",
                "--ledger",
                str(ledger),
            ]
            self.assertEqual(
                planning_main(
                    [
                        "lock",
                        *common,
                        "--stage",
                        "calibration",
                        "--benchmark",
                        "evolving_intent_gsm8k",
                        "--models",
                        "gpt-5.6-luna",
                        "--arms",
                        "clean,active_recompute",
                        "--operators",
                        "none",
                        "--output",
                        str(lock),
                    ]
                ),
                0,
            )
            self.assertTrue(lock.is_file())
            self.assertEqual(
                planning_main(
                    ["validate-lock", *common, "--lock", str(lock)]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
