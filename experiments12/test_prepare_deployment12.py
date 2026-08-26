from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments12.analysis12 import make_threshold_artifact
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
    sha256_json,
)
from experiments12.core.schemas import CallAttemptRecord, CallStatus, TokenUsage
from experiments12.core.transport import CompletionResult
from experiments12.deployment12 import (
    DEPLOYMENT_SCHEDULE_RECEIPT,
    PASS_ONE_RECEIPT,
    THRESHOLD_LOCK_RECEIPT,
    DeploymentArtifactError,
    DeploymentEstimand,
    deployment_runtime_config,
    load_deployment_schedule,
    load_pass_one_observations,
    load_threshold_lock,
)
from experiments12.deployment_pass_one12 import initialize_evolving_pass_one
from experiments12.domains.evolving_intent import PINNED_COMMIT
from experiments12.harness12 import ARM_TO_PROBE
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.metrics12 import (
    CheckpointScore,
    ObservationTrace,
    select_fixed_firing_rate_threshold,
)
from experiments12.passive_spec12 import (
    PASSIVE_MONITOR_SPEC_SHA256,
    effective_passive_method_names,
)
from experiments12.probes12 import (
    generate_probe_instance,
    grade_probe_response,
    render_initial_instruction,
    render_probe_prompt,
)
from experiments12.planning_lock12 import ScientificLaunchBinding
from experiments12.prepare_deployment12 import (
    CALIBRATION_EXTRACT_RECEIPT,
    CALIBRATION_MANIFEST_RECEIPT,
    CALIBRATION_THRESHOLDS_RECEIPT,
    DEPLOYMENT_MODE,
    DEPLOYMENT_PAIR_RECEIPT,
    SOURCE_OBSERVATION_MANIFEST_RECEIPT,
    _active_records,
    _passive_records,
    deployment_pass_one_source_contract,
    deployment_threshold_lock_from_analysis,
    prepare_deployment_run,
    verify_analysis_threshold_derivation,
)
from experiments12.spec12 import Operator, Stage
from experiments12.source_registry12 import SourceAllocationBinding
from experiments12.runner12 import load_pair_cells
from experiments12.shadow12 import score_clean_trajectory


ROOT = Path(__file__).resolve().parent.parent
MODEL = "gpt-5.6-luna"
BENCHMARK = "evolving_intent_gsm8k"
ACTIVE = "active_name_copy"
PASSIVE = "trace_rules"
TASK_ID = "deployment-task"
PAIR_TASK_ID = f"{TASK_ID}::t7"


def ready_report():
    return SimpleNamespace(primary_ready=True, errors=())


class ShadowFixtureTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, model, messages, **kwargs):
        self.calls += 1
        purpose = kwargs["purpose"]
        text = (
            '{"risk":0.25,"concerns":[],"evidence":[]}'
            if purpose == "trace_judge"
            else "PROBE: AAAAAAAA"
        )
        attempt = CallAttemptRecord(
            event_id=f"shadow-event-{self.calls}",
            reservation_id=f"shadow-reservation-{self.calls}",
            provider="openai",
            model=model,
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="start",
            finished_at="finish",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            estimated_cost_usd=Decimal("0.001"),
            elapsed_ms=1,
        )
        return CompletionResult(
            text=text,
            tool_calls=(),
            usage=attempt.usage,
            response_id=f"shadow-response-{self.calls}",
            request_id=f"shadow-request-{self.calls}",
            model_id=model,
            finish_reason="stop",
            cost_usd=Decimal("0.001"),
            attempts=(attempt,),
        )


class PreparationFixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_prepare_")
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.dataset = self.root / "dataset.json"
        atomic_write_json(self.dataset, {"tasks": []})
        self.dataset_sha256 = sha256_file(self.dataset)
        self.build_receipt = self.root / "build-receipt.json"
        atomic_write_json(
            self.build_receipt,
            {
                "benchmark": BENCHMARK,
                "upstream_commit": PINNED_COMMIT,
                "shared_across_target_arms_and_models": True,
                "frozen_dataset": {"sha256": self.dataset_sha256},
            },
        )
        self.task_sha256 = sha256_json({"task": TASK_ID, "condition": "t7"})
        self.tasks = self.root / "tasks.jsonl"
        self.task_rows = [
            {
                "task_manifest_version": 1,
                "benchmark": BENCHMARK,
                "task_id": PAIR_TASK_ID,
                "source_task_id": TASK_ID,
                "condition": "t7",
                "num_turns": 3,
                "source_sha256": self.dataset_sha256,
                "task_sha256": self.task_sha256,
            }
        ]
        atomic_write_jsonl(self.tasks, self.task_rows)
        self.registry = self.root / "source-registry.json"
        self.baseline_profile = self.root / "baseline-profile.json"
        self.planning_lock = self.root / "deployment-planning-lock.json"
        atomic_write_json(self.registry, {"fixture": "registry"})
        atomic_write_json(self.baseline_profile, {"fixture": "baseline"})
        atomic_write_json(self.planning_lock, {"fixture": "planning"})
        self.launch_binding = ScientificLaunchBinding(
            allocation=SourceAllocationBinding(
                registry_sha256=sha256_file(self.registry),
                benchmark=BENCHMARK,
                stage="deployment",
                wave=None,
                source_ids=(TASK_ID,),
            ),
            projection_lock_sha256=sha256_file(self.planning_lock),
            projected_provider_usd={"fireworks": "0.1", "openai": "0.2"},
            required_n_tasks=1,
        )
        self._calibration_lock()
        self._source_run()

    def close(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _call(tag: str) -> dict[str, object]:
        return {
            "call_event_ids": [f"event-{tag}"],
            "resolved_model_id": MODEL,
            "response_id": f"response-{tag}",
            "request_id": f"request-{tag}",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            },
            "accounted_cost_usd": "0.001",
            "elapsed_ms": 2,
        }

    def _trajectory(self, cell, *, active: bool) -> dict[str, object]:
        task_records: list[dict[str, object]] = []
        probe_records: list[dict[str, object]] = []
        messages: list[dict[str, str]] = []
        checkpoints = [1, 2]
        variant = ARM_TO_PROBE[ACTIVE]
        instance_id = f"{BENCHMARK}/{TASK_ID}/t7"
        setup = render_initial_instruction(
            variant, instance_id, tuple(range(1, len(checkpoints) + 1))
        )
        for turn in range(1, 4):
            user_content = f"user-{turn}"
            if active and turn == 1 and setup:
                user_content = setup + "\n\n--- BENCHMARK MESSAGE ---\n" + user_content
            user = {"role": "user", "content": user_content}
            assistant = {"role": "assistant", "content": f"assistant-{turn}"}
            task_record = {
                "event": "task_turn",
                "task_turn": turn,
                "user_message": user,
                "assistant_message": assistant,
                "call": self._call(f"{cell.cell_id}-task-{turn}"),
            }
            task_records.append(task_record)
            messages.extend((user, assistant))
            if active and turn in checkpoints:
                instance = generate_probe_instance(variant, instance_id, turn)
                response = instance.expected_answer
                grade = grade_probe_response(instance, response)
                probe_user = {
                    "role": "user",
                    "content": render_probe_prompt(instance),
                }
                probe_assistant = {"role": "assistant", "content": response}
                messages.extend((probe_user, probe_assistant))
                probe_records.append(
                    {
                        "event": "active_probe",
                        "after_task_turn": turn,
                        "checkpoint_index": checkpoints.index(turn) + 1,
                        "variant": variant,
                        "user_message": probe_user,
                        "assistant_message": probe_assistant,
                        "grade": {
                            "passed": grade.passed,
                            "value_correct": grade.value_correct,
                            "exact_format": grade.exact_format,
                            "error": grade.error,
                            "expected_sha256": sha256_json(instance.expected_answer),
                        },
                        "call": self._call(f"{cell.cell_id}-probe-{turn}"),
                        "source_prefix_sha256": sha256_json(messages),
                    }
                )
        return {
            "complete": True,
            "run_id": "source-run",
            "cell_id": cell.cell_id,
            "model": MODEL,
            "domain": BENCHMARK,
            "task_id": TASK_ID,
            "condition": "t7",
            "task_sha256": self.task_sha256,
            "arm": ACTIVE if active else "clean",
            "active_probe_variant": variant if active else None,
            "checkpoint_turns": checkpoints,
            "messages": messages,
            "task_records": task_records,
            "probe_records": probe_records,
            "transcript_sha256": sha256_json(messages),
        }

    def _source_run(self) -> None:
        with patch(
            "experiments12.deployment_pass_one12.assert_scientific_launch",
            return_value=self.launch_binding,
        ):
            initialize_evolving_pass_one(
                run_id="source-run",
                task_manifest_path=self.tasks,
                calibration_threshold_path=self.analysis_thresholds,
                source_registry_path=self.registry,
                baseline_profile_path=self.baseline_profile,
                planning_lock_path=self.planning_lock,
                models=(MODEL,),
                methods=(ACTIVE, PASSIVE),
                deployment_operators=(
                    Operator.NONE.value,
                    Operator.REGROUND.value,
                ),
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                natural_max_actions_per_task=1,
                matched_actions_per_method=1,
                yoke_anchor_method=PASSIVE,
                randomization_seed=31,
                evolving_dataset_path=self.dataset,
                evolving_build_receipt_path=self.build_receipt,
                artifacts_root=self.artifacts,
            )
        self.source = RunLayout.for_run(self.artifacts, "source-run")
        cells = load_pair_cells(self.source.pairs)
        for cell in cells:
            trajectory = self._trajectory(cell, active=cell.arm == ACTIVE)
            atomic_write_json(
                self.source.trajectories / f"{cell.cell_id}.json", trajectory
            )
            if cell.arm != "clean":
                continue
            shadow_path = self.source.shadow / f"{cell.cell_id}.json"
            result = asyncio.run(
                score_clean_trajectory(
                    run_id="source-run",
                    trajectory=trajectory,
                    transport=ShadowFixtureTransport(),
                    event_path=self.source.events / f"shadow-{cell.cell_id}.jsonl",
                    output_path=shadow_path,
                )
            )
            atomic_write_json(
                self.source.results / "shadow_jobs" / f"{cell.cell_id}.json",
                {
                    "runner_version": 1,
                    "cell_id": cell.cell_id,
                    "state": "complete",
                    "shadow_sha256": sha256_file(shadow_path),
                    "monitor_methods": result["monitor_methods"],
                    "passive_monitor_spec_sha256": result[
                        "passive_monitor_spec_sha256"
                    ],
                },
            )
        self.source_cells = cells

    def _calibration_lock(self) -> None:
        calibration = RunLayout.for_run(self.artifacts, "calibration-run")
        calibration.create()
        atomic_write_jsonl(calibration.pairs, [{"fixture": True}])
        manifest = build_manifest(
            run_id="calibration-run",
            stage=Stage.CALIBRATION,
            repository_root=ROOT,
            pair_manifest_sha256=sha256_file(calibration.pairs),
            models=(MODEL,),
            arms=(ACTIVE,),
            operators=(Operator.NONE.value,),
            randomization_seed=23,
            benchmark_receipts=(),
            extra_config={"n_cells": 1, "replicates": 1},
        )
        write_manifest_once(calibration.manifest, manifest)
        calibration_manifest_sha256 = sha256_file(calibration.manifest)
        traces = tuple(
            ObservationTrace(
                model=MODEL,
                benchmark=BENCHMARK,
                method=method,
                task_id=f"calibration-{index}/r0",
                split="calibration",
                checkpoints=(
                    CheckpointScore(1, 0.1 + index * 0.1, True),
                    CheckpointScore(2, 0.8 + index * 0.05, True),
                ),
                event_checkpoint=2,
                source_task_id=f"calibration-{index}",
            )
            for index, method in enumerate((ACTIVE, PASSIVE), 1)
        )
        source = {
            "run_id": "calibration-run",
            "stage": "calibration",
            "manifest_sha256": calibration_manifest_sha256,
            "split": "calibration",
            "required_passive_methods": list(effective_passive_method_names()),
            "signal_traces": [asdict(trace) for trace in traces],
        }
        self.calibration_extract = self.root / "calibration-extract.json"
        atomic_write_json(self.calibration_extract, source)
        selections = tuple(
            select_fixed_firing_rate_threshold((trace,), target_firing_rate=0.5)
            for trace in traces
        )
        threshold = make_threshold_artifact(
            source,
            traces,
            selections,
            target_firing_rate=0.5,
            source_extract_sha256=sha256_file(self.calibration_extract),
        )
        self.analysis_thresholds = self.root / "analysis-thresholds.json"
        atomic_write_json(self.analysis_thresholds, threshold)

    def prepare(
        self,
        run_id: str = "deployment-run",
        *,
        realized_allocation_path: Path | None = None,
    ):
        return prepare_deployment_run(
            source_run_id="source-run",
            deployment_run_id=run_id,
            task_manifest_path=self.tasks,
            calibration_threshold_path=self.analysis_thresholds,
            calibration_extract_path=self.calibration_extract,
            source_registry_path=self.registry,
            baseline_profile_path=self.baseline_profile,
            planning_lock_path=self.planning_lock,
            realized_allocation_path=realized_allocation_path,
            methods=(ACTIVE, PASSIVE),
            operators=(Operator.NONE.value, Operator.REGROUND.value),
            estimand=DeploymentEstimand.NATURAL_THRESHOLD,
            natural_max_actions_per_task=1,
            matched_actions_per_method=1,
            yoke_anchor_method=PASSIVE,
            randomization_seed=31,
            artifacts_root=self.artifacts,
            evolving_dataset_path=self.dataset,
            evolving_build_receipt_path=self.build_receipt,
        )


class ThresholdConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = PreparationFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_analysis_derivation_reproduces_and_global_overlap_is_rejected(self):
        payload, digest = verify_analysis_threshold_derivation(
            self.fixture.analysis_thresholds,
            self.fixture.calibration_extract,
        )
        self.assertEqual(digest, sha256_file(self.fixture.calibration_extract))
        lock = deployment_threshold_lock_from_analysis(
            payload,
            deployment_task_rows=self.fixture.task_rows,
            models=(MODEL,),
            methods=(ACTIVE, PASSIVE),
            natural_max_actions_per_task=1,
            matched_actions_per_method=1,
            yoke_anchor_method=PASSIVE,
        )
        self.assertEqual(
            {(row.model, row.benchmark, row.method) for row in lock.methods},
            {
                (MODEL, BENCHMARK, ACTIVE),
                (MODEL, BENCHMARK, PASSIVE),
            },
        )
        overlap = [dict(self.fixture.task_rows[0])]
        overlap[0]["source_task_id"] = "calibration-1"
        with self.assertRaisesRegex(DeploymentArtifactError, "overlap globally"):
            deployment_threshold_lock_from_analysis(
                payload,
                deployment_task_rows=overlap,
                models=(MODEL,),
                methods=(ACTIVE, PASSIVE),
                natural_max_actions_per_task=1,
                matched_actions_per_method=1,
                yoke_anchor_method=PASSIVE,
            )

        aliased_threshold = deepcopy(payload)
        aliased_threshold["calibration_source_tasks"][0]["source_task_id"] = "770"
        aliased_threshold["calibration_source_tasks"].sort(
            key=lambda row: (row["benchmark"], row["source_task_id"])
        )
        aliased_deployment = [dict(self.fixture.task_rows[0])]
        aliased_deployment[0]["source_task_id"] = "extracted-gsm8k-test-770"
        with self.assertRaisesRegex(DeploymentArtifactError, "overlap globally"):
            deployment_threshold_lock_from_analysis(
                aliased_threshold,
                deployment_task_rows=aliased_deployment,
                models=(MODEL,),
                methods=(ACTIVE, PASSIVE),
                natural_max_actions_per_task=1,
                matched_actions_per_method=1,
                yoke_anchor_method=PASSIVE,
            )

    def test_threshold_extract_tampering_fails(self):
        extract = read_json(self.fixture.calibration_extract)
        extract["signal_traces"][0]["checkpoints"][0]["score"] = 0.99
        atomic_write_json(self.fixture.calibration_extract, extract)
        with self.assertRaisesRegex(DeploymentArtifactError, "extract differs"):
            verify_analysis_threshold_derivation(
                self.fixture.analysis_thresholds,
                self.fixture.calibration_extract,
            )


class PrefixExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = PreparationFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_active_and_passive_prefixes_are_recomputed(self):
        active_cell = next(
            cell for cell in self.fixture.source_cells if cell.arm == ACTIVE
        )
        active = read_json(
            self.fixture.source.trajectories / f"{active_cell.cell_id}.json"
        )
        self.assertEqual(len(_active_records(active, ACTIVE)), 2)
        active["probe_records"][0]["source_prefix_sha256"] = "f" * 64
        with self.assertRaisesRegex(DeploymentArtifactError, "active signal integrity"):
            _active_records(active, ACTIVE)

        clean_cell = next(
            cell for cell in self.fixture.source_cells if cell.arm == "clean"
        )
        clean = read_json(
            self.fixture.source.trajectories / f"{clean_cell.cell_id}.json"
        )
        shadow = read_json(self.fixture.source.shadow / f"{clean_cell.cell_id}.json")
        self.assertEqual(len(_passive_records(clean, shadow, PASSIVE)), 2)
        shadow["records"][0]["source_prefix_sha256"] = "e" * 64
        with self.assertRaisesRegex(DeploymentArtifactError, "passive signal integrity"):
            _passive_records(clean, shadow, PASSIVE)


class ProductionPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = PreparationFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    @patch("experiments12.deployment_pass_one12.assert_scientific_launch")
    def test_production_initializer_freezes_minimal_bound_source_run(
        self, mocked_launch
    ):
        mocked_launch.return_value = self.fixture.launch_binding
        result = initialize_evolving_pass_one(
            run_id="initialized-source",
            task_manifest_path=self.fixture.tasks,
            calibration_threshold_path=self.fixture.analysis_thresholds,
            source_registry_path=self.fixture.registry,
            baseline_profile_path=self.fixture.baseline_profile,
            planning_lock_path=self.fixture.planning_lock,
            models=(MODEL,),
            methods=(ACTIVE, PASSIVE),
            deployment_operators=(Operator.NONE.value, Operator.REGROUND.value),
            estimand=DeploymentEstimand.NATURAL_THRESHOLD,
            natural_max_actions_per_task=1,
            matched_actions_per_method=1,
            yoke_anchor_method=PASSIVE,
            randomization_seed=41,
            evolving_dataset_path=self.fixture.dataset,
            evolving_build_receipt_path=self.fixture.build_receipt,
            artifacts_root=self.fixture.artifacts,
        )
        self.assertEqual(result.declared_cells, 2)
        self.assertEqual(result.source_arms, ("clean", ACTIVE))
        manifest = read_json(result.manifest_path)
        self.assertEqual(manifest["arms"], ["clean", ACTIVE])
        self.assertEqual(manifest["operators"], [Operator.NONE.value])
        contract = manifest["extra_config"]["deployment_pass_one_source"]
        self.assertEqual(contract["deployment_methods"], [ACTIVE, PASSIVE])
        self.assertEqual(
            contract["deployment_operators"],
            [Operator.NONE.value, Operator.REGROUND.value],
        )
        self.assertEqual(contract["statistical_unit"], "source_task")
        self.assertEqual(
            contract["deployment_estimand"],
            DeploymentEstimand.NATURAL_THRESHOLD.value,
        )
        self.assertEqual(contract["natural_max_actions_per_task"], 1)
        self.assertEqual(contract["matched_actions_per_method"], 1)
        self.assertEqual(contract["yoke_anchor_method"], PASSIVE)
        self.assertEqual(contract["randomization_seed"], 41)
        names = {row["name"] for row in manifest["benchmark_receipts"]}
        self.assertIn(CALIBRATION_THRESHOLDS_RECEIPT, names)
        mocked_launch.assert_called_once()
        launch = mocked_launch.call_args.kwargs
        self.assertEqual(launch["allocation_stage"], "deployment")
        self.assertEqual(launch["design_family"], "deployment")
        self.assertEqual(launch["arms"], (ACTIVE, PASSIVE))
        self.assertEqual(
            launch["operators"],
            (Operator.NONE.value, Operator.REGROUND.value),
        )
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            initialize_evolving_pass_one(
                run_id="initialized-source",
                task_manifest_path=self.fixture.tasks,
                calibration_threshold_path=self.fixture.analysis_thresholds,
                source_registry_path=self.fixture.registry,
                baseline_profile_path=self.fixture.baseline_profile,
                planning_lock_path=self.fixture.planning_lock,
                models=(MODEL,),
                methods=(ACTIVE, PASSIVE),
                deployment_operators=(
                    Operator.NONE.value,
                    Operator.REGROUND.value,
                ),
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                natural_max_actions_per_task=1,
                matched_actions_per_method=1,
                yoke_anchor_method=PASSIVE,
                randomization_seed=41,
                evolving_dataset_path=self.fixture.dataset,
                evolving_build_receipt_path=self.fixture.build_receipt,
                artifacts_root=self.fixture.artifacts,
            )

    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    @patch("experiments12.prepare_deployment12.extract_run")
    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    def test_full_preparation_binds_every_artifact_and_allows_passive_arms(
        self, mocked_validate, mocked_extract, mocked_launch
    ):
        mocked_extract.return_value = read_json(self.fixture.calibration_extract)
        mocked_launch.return_value = self.fixture.launch_binding
        result = self.fixture.prepare()
        self.assertEqual(result.declared_cells, 4)
        self.assertEqual(result.pass_one_traces, 2)
        self.assertEqual(mocked_validate.call_count, 1)
        mocked_extract.assert_called_once()
        manifest = read_json(result.layout.manifest)
        self.assertEqual(manifest["arms"], [ACTIVE, PASSIVE])
        self.assertEqual(manifest["extra_config"]["deployment_mode"], DEPLOYMENT_MODE)
        self.assertEqual(
            manifest["extra_config"]["deployment_runtime"],
            deployment_runtime_config(),
        )
        self.assertEqual(
            manifest["extra_config"]["analysis_lock"]["threshold_artifact_sha256"],
            sha256_file(self.fixture.analysis_thresholds),
        )
        names = {row["name"] for row in manifest["benchmark_receipts"]}
        self.assertTrue(
            {
                "task_manifest",
                DEPLOYMENT_PAIR_RECEIPT,
                SOURCE_OBSERVATION_MANIFEST_RECEIPT,
                CALIBRATION_MANIFEST_RECEIPT,
                CALIBRATION_EXTRACT_RECEIPT,
                CALIBRATION_THRESHOLDS_RECEIPT,
                PASS_ONE_RECEIPT,
                THRESHOLD_LOCK_RECEIPT,
                DEPLOYMENT_SCHEDULE_RECEIPT,
                "evolving_rendered_dataset",
                "evolving_build_receipt",
                "source_allocation_registry",
                "measured_baseline_resource_profile",
                "cost_sample_size_projection_lock",
            }.issubset(names)
        )
        pass_one = load_pass_one_observations(
            result.layout.results / "deployment_pass_one.json"
        )
        self.assertEqual({row.method for row in pass_one.traces}, {ACTIVE, PASSIVE})
        self.assertNotIn("evaluation", str(pass_one.as_dict()))
        lock = load_threshold_lock(
            result.layout.results / "deployment_threshold_lock.json"
        )
        schedule = load_deployment_schedule(
            result.layout.results / "deployment_schedule.json"
        )
        self.assertEqual(schedule.threshold_lock_sha256, result.threshold_lock_sha256)
        self.assertEqual(len(lock.methods), 2)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self.fixture.prepare()

        mocked_launch.assert_called_once()
        call = mocked_launch.call_args.kwargs
        self.assertEqual(call["stage"], Stage.CONFIRMATORY)
        self.assertEqual(call["allocation_stage"], "deployment")
        self.assertEqual(call["design_family"], "deployment")
        self.assertIsNone(call["realized_allocation_path"])

    @patch(
        "experiments12.prepare_deployment12.validate_run",
        return_value=SimpleNamespace(
            primary_ready=False,
            errors=(SimpleNamespace(code="trajectory.missing_cell"),),
        ),
    )
    def test_incomplete_source_run_fails_before_creating_destination(self, _mocked):
        with self.assertRaisesRegex(DeploymentArtifactError, "not complete"):
            self.fixture.prepare("rejected-run")
        self.assertFalse((self.fixture.artifacts / "rejected-run").exists())

    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    def test_preparation_rejects_a_nonproduction_pass_one_contract(self, _mocked):
        manifest = read_json(self.fixture.source.manifest)
        manifest["extra_config"]["deployment_pass_one_source"]["replicates"] = 2
        atomic_write_json(self.fixture.source.manifest, manifest)
        with self.assertRaisesRegex(DeploymentArtifactError, "exact deployment pass-one"):
            self.fixture.prepare("bad-source-contract")
        self.assertFalse((self.fixture.artifacts / "bad-source-contract").exists())

    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    def test_preparation_requires_exact_initializer_metadata_and_analysis_lock(self, _mocked):
        manifest = read_json(self.fixture.source.manifest)
        del manifest["extra_config"]["initializer_version"]
        atomic_write_json(self.fixture.source.manifest, manifest)
        with self.assertRaisesRegex(DeploymentArtifactError, "production initializer metadata"):
            self.fixture.prepare("bad-initializer-metadata")

        self.fixture.close()
        self.fixture = PreparationFixture()
        manifest = read_json(self.fixture.source.manifest)
        manifest["extra_config"]["analysis_lock"]["calibration_manifest_sha256"] = "f" * 64
        atomic_write_json(self.fixture.source.manifest, manifest)
        with self.assertRaisesRegex(DeploymentArtifactError, "exact deployment analysis lock"):
            self.fixture.prepare("bad-source-analysis-lock")

    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    @patch("experiments12.prepare_deployment12.extract_run")
    def test_signal_scores_and_shadow_events_are_replayed_before_scheduling(
        self, mocked_extract, mocked_launch, _mocked
    ):
        mocked_extract.return_value = read_json(self.fixture.calibration_extract)
        mocked_launch.return_value = self.fixture.launch_binding
        clean_cell = next(cell for cell in self.fixture.source_cells if cell.arm == "clean")
        shadow_path = self.fixture.source.shadow / f"{clean_cell.cell_id}.json"
        event_path = self.fixture.source.events / f"shadow-{clean_cell.cell_id}.jsonl"
        job_path = self.fixture.source.results / "shadow_jobs" / f"{clean_cell.cell_id}.json"
        shadow = read_json(shadow_path)
        rules = next(row for row in shadow["records"] if row["method"] == "trace_rules")
        rules["score"] = 1.0 if rules["score"] != 1.0 else 0.0
        atomic_write_json(shadow_path, shadow)
        with self.assertRaisesRegex(DeploymentArtifactError, "append-only events"):
            self.fixture.prepare("shadow-event-mismatch")

        atomic_write_jsonl(event_path, shadow["records"])
        job = read_json(job_path)
        job["shadow_sha256"] = sha256_file(shadow_path)
        atomic_write_json(job_path, job)
        with self.assertRaisesRegex(DeploymentArtifactError, "passive signal integrity"):
            self.fixture.prepare("shadow-score-tamper")

    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    @patch("experiments12.prepare_deployment12.extract_run")
    def test_active_cached_grade_is_replayed_before_scheduling(
        self, mocked_extract, mocked_launch, _mocked
    ):
        mocked_extract.return_value = read_json(self.fixture.calibration_extract)
        mocked_launch.return_value = self.fixture.launch_binding
        active_cell = next(cell for cell in self.fixture.source_cells if cell.arm == ACTIVE)
        path = self.fixture.source.trajectories / f"{active_cell.cell_id}.json"
        trajectory = read_json(path)
        trajectory["probe_records"][0]["grade"]["passed"] = not trajectory[
            "probe_records"
        ][0]["grade"]["passed"]
        atomic_write_json(path, trajectory)
        with self.assertRaisesRegex(DeploymentArtifactError, "active signal integrity"):
            self.fixture.prepare("active-grade-tamper")

    @patch("experiments12.prepare_deployment12.extract_run")
    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    def test_launch_binding_must_match_exact_projection_lock(
        self, mocked_launch, _mocked_validate, mocked_extract
    ):
        mocked_extract.return_value = read_json(self.fixture.calibration_extract)
        mocked_launch.return_value = ScientificLaunchBinding(
            allocation=self.fixture.launch_binding.allocation,
            projection_lock_sha256="f" * 64,
            projected_provider_usd={"fireworks": "0.1", "openai": "0.2"},
            required_n_tasks=1,
        )
        with self.assertRaisesRegex(DeploymentArtifactError, "launch binding differs"):
            self.fixture.prepare("bad-launch-run")
        self.assertFalse((self.fixture.artifacts / "bad-launch-run").exists())

    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    def test_source_projection_receipt_must_match_source_launch(
        self, mocked_launch, _mocked_validate
    ):
        mocked_launch.return_value = self.fixture.launch_binding
        manifest = read_json(self.fixture.source.manifest)
        manifest["extra_config"]["scientific_launch_lock"][
            "projection_lock_sha256"
        ] = "e" * 64
        atomic_write_json(self.fixture.source.manifest, manifest)
        with self.assertRaisesRegex(
            DeploymentArtifactError, "not bound to this deployment planning lock"
        ):
            self.fixture.prepare("bad-source-lock-run")
        self.assertFalse((self.fixture.artifacts / "bad-source-lock-run").exists())

    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    def test_source_launch_binding_must_match_every_final_field(
        self, mocked_launch, _mocked_validate
    ):
        mocked_launch.return_value = self.fixture.launch_binding
        manifest = read_json(self.fixture.source.manifest)
        manifest["extra_config"]["scientific_launch_lock"]["required_n_tasks"] = 999
        atomic_write_json(self.fixture.source.manifest, manifest)
        with self.assertRaisesRegex(DeploymentArtifactError, "differs from final deployment"):
            self.fixture.prepare("bad-source-launch-details")

    @patch("experiments12.prepare_deployment12.extract_run")
    @patch("experiments12.prepare_deployment12.validate_run", return_value=ready_report())
    @patch("experiments12.prepare_deployment12.assert_scientific_launch")
    def test_optional_realized_allocation_is_bound_end_to_end(
        self, mocked_launch, _mocked_validate, mocked_extract
    ):
        mocked_extract.return_value = read_json(self.fixture.calibration_extract)
        realized = self.fixture.root / "realized-allocation.json"
        atomic_write_json(realized, {"outcome_blind": True})
        allocation = SourceAllocationBinding(
            registry_sha256=sha256_file(self.fixture.registry),
            benchmark=BENCHMARK,
            stage="deployment",
            wave=None,
            source_ids=(TASK_ID,),
            realized_allocation_sha256=sha256_file(realized),
        )
        binding = ScientificLaunchBinding(
            allocation=allocation,
            projection_lock_sha256=sha256_file(self.fixture.planning_lock),
            projected_provider_usd={"fireworks": "0.1", "openai": "0.2"},
            required_n_tasks=1,
        )
        mocked_launch.return_value = binding
        manifest = read_json(self.fixture.source.manifest)
        manifest["extra_config"]["scientific_launch_lock"] = binding.as_dict()
        manifest["benchmark_receipts"].append(
            asdict(
                ArtifactReceipt.from_file(
                    "realized_source_allocation", realized, workspace=ROOT
                )
            )
        )
        atomic_write_json(self.fixture.source.manifest, manifest)

        result = self.fixture.prepare(
            "realized-run", realized_allocation_path=realized
        )
        output = read_json(result.layout.manifest)
        receipts = {
            row["name"]: row["sha256"] for row in output["benchmark_receipts"]
        }
        self.assertEqual(receipts["realized_source_allocation"], sha256_file(realized))
        self.assertEqual(
            output["extra_config"]["scientific_launch_lock"], binding.as_dict()
        )


if __name__ == "__main__":
    unittest.main()
