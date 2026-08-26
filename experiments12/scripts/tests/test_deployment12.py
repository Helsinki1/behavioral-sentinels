from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments12.core.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    read_json,
    sha256_file,
)
from experiments12.core.schemas import CallAttemptRecord, CallStatus, PairKey, TokenUsage
from experiments12.core.transport import CompletionResult
from experiments12.deployment12 import (
    DEPLOYMENT_SCHEDULE_RECEIPT,
    PASS_ONE_RECEIPT,
    THRESHOLD_LOCK_RECEIPT,
    DeploymentArtifactError,
    DeploymentEstimand,
    FeedbackEvidence,
    LockedMethodThreshold,
    PassOneCheckpoint,
    PassOneMethodTrace,
    PassOneObservationArtifact,
    ThresholdLockArtifact,
    build_deployment_schedule,
    build_pass_one_observation_artifact,
    deployment_runtime_config,
    deployment_completeness,
    execute_deployment_run,
    extract_deployment_outcomes,
    freeze_deployment_schedule,
    freeze_pass_one_observations,
    freeze_threshold_lock,
    load_deployment_schedule,
    main,
    pass_one_trace_from_records,
    run_deployment_task,
    threshold_lock_from_calibration,
    validate_deployment_schedule,
)
from experiments12.domains.base import DomainTask, DomainTurn, canonical_json_sha256
from experiments12.harness12 import ARM_TO_PROBE, HarnessConfig
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.metrics12 import ThresholdSelection
from experiments12.operators12 import CompactionConfig
from experiments12.pairing12 import JobCell
from experiments12.runner12 import freeze_task_manifest, pair_task_id
from experiments12.spec12 import Operator, Stage


ROOT = Path(__file__).resolve().parents[3]
MODEL = "gpt-5.6-luna"
METHODS = ("active_recompute", "trace_rules", "turn_clock")
OPERATORS = (
    Operator.NONE.value,
    Operator.COMPACT.value,
    Operator.REGROUND.value,
    Operator.FEEDBACK.value,
)


def task() -> DomainTask:
    turns = (
        DomainTurn(1, "Start with 2 apples."),
        DomainTurn(2, "Actually use 3 apples."),
        DomainTurn(3, "What is twice that?"),
    )
    source = canonical_json_sha256({"dataset": "deployment-test"})
    digest = canonical_json_sha256(
        {
            "domain": "evolving_intent_gsm8k",
            "task_id": "x",
            "condition": "t7",
            "turns": [turn.user_message for turn in turns],
            "source": source,
        }
    )
    return DomainTask(
        domain="evolving_intent_gsm8k",
        task_id="x",
        condition="t7",
        turns=turns,
        evaluation_label="6",
        source_sha256=source,
        task_sha256=digest,
        public_metadata=(("split", "confirmatory"),),
    )


def cells(
    methods: tuple[str, ...] = METHODS,
    operators: tuple[str, ...] = OPERATORS,
) -> tuple[JobCell, ...]:
    current = task()
    pair = PairKey(
        model=MODEL,
        domain=current.domain,
        task_id=pair_task_id(current),
        replicate_id=0,
        task_sha256=current.task_sha256,
    )
    result = []
    for method in methods:
        for operator in operators:
            result.append(
                JobCell(
                    cell_id=canonical_json_sha256(
                        {"method": method, "operator": operator}
                    )[:24],
                    block_id="block-x",
                    block_position=len(result),
                    pair_key=pair,
                    arm=method,
                    operator=operator,
                    seed=12,
                )
            )
    return tuple(result)


SCORES = {
    "active_recompute": (0.9, 0.1),
    "trace_rules": (0.1, 0.9),
    "turn_clock": (0.7, 0.6),
}


def pass_one(methods: tuple[str, ...] = METHODS, scores=SCORES) -> PassOneObservationArtifact:
    current = task()
    traces = []
    for method in methods:
        rows = tuple(
            PassOneCheckpoint(
                checkpoint=index,
                score=score,
                source_prefix_sha256=canonical_json_sha256(
                    {"method": method, "checkpoint": index, "kind": "prefix"}
                ),
                signal_record_sha256=canonical_json_sha256(
                    {"method": method, "checkpoint": index, "kind": "signal"}
                ),
            )
            for index, score in enumerate(scores[method], 1)
        )
        traces.append(
            PassOneMethodTrace(
                model=MODEL,
                benchmark=current.domain,
                task_id=pair_task_id(current),
                task_sha256=current.task_sha256,
                replicate_id=0,
                method=method,
                active_variant=ARM_TO_PROBE.get(method),
                source_trajectory_sha256=canonical_json_sha256(
                    {"method": method, "trajectory": "pass-one"}
                ),
                task_horizon=3,
                checkpoints=rows,
            )
        )
    return PassOneObservationArtifact(
        source_run_id="pass-one-run",
        source_manifest_sha256="a" * 64,
        traces=tuple(sorted(traces, key=lambda row: row.identity)),
    )


def threshold_lock(methods: tuple[str, ...] = METHODS) -> ThresholdLockArtifact:
    rows = tuple(
        sorted(
            (
                LockedMethodThreshold(
                    model=MODEL,
                    benchmark=task().domain,
                    method=method,
                    threshold=0.8,
                    target_firing_rate=0.5,
                    achieved_firing_rate=0.5,
                    calibration_n_tasks=20,
                    calibration_digest=canonical_json_sha256(
                        {"method": method, "calibration": "locked"}
                    ),
                    selection_rule="task_score_rank_hash_ties",
                    tie_break_seed=12012,
                    calibration_target_fire_count=10,
                )
                for method in methods
            ),
            key=lambda row: (row.model, row.benchmark, row.method),
        )
    )
    return ThresholdLockArtifact(
        calibration_run_id="calibration-run",
        calibration_manifest_sha256="b" * 64,
        natural_max_actions_per_task=2,
        matched_actions_per_method=1,
        yoke_anchor_method="trace_rules" if "trace_rules" in methods else methods[0],
        methods=rows,
    )


def _quote(checkpoint: int) -> str:
    return task().turns[checkpoint - 1].user_message


def feedback_for(
    declared: tuple[JobCell, ...],
    actions_by_method: Mapping[str, tuple[int, ...]],
) -> dict[tuple[str, int], FeedbackEvidence]:
    return {
        (cell.cell_id, checkpoint): FeedbackEvidence(watch=(_quote(checkpoint),))
        for cell in declared
        if cell.operator == Operator.FEEDBACK.value
        for checkpoint in actions_by_method[cell.arm]
    }


def build(
    estimand: DeploymentEstimand,
    *,
    declared: tuple[JobCell, ...] | None = None,
    observations: PassOneObservationArtifact | None = None,
    lock: ThresholdLockArtifact | None = None,
    feedback: Mapping[tuple[str, int], FeedbackEvidence] | None = None,
):
    declared = declared or cells()
    observations = observations or pass_one()
    lock = lock or threshold_lock()
    if feedback is None:
        expected = {
            DeploymentEstimand.NATURAL_THRESHOLD: {
                "active_recompute": (1,),
                "trace_rules": (2,),
                "turn_clock": (),
            },
            DeploymentEstimand.MATCHED_RATE_TOP_K: {
                "active_recompute": (1,),
                "trace_rules": (2,),
                "turn_clock": (1,),
            },
            DeploymentEstimand.YOKED_ANCHOR: {
                "active_recompute": (2,),
                "trace_rules": (2,),
                "turn_clock": (2,),
            },
        }[estimand]
        feedback = feedback_for(declared, expected)
    return build_deployment_schedule(
        estimand=estimand,
        cells=declared,
        pair_manifest_sha256="c" * 64,
        pass_one=observations,
        pass_one_artifact_sha256="d" * 64,
        threshold_lock=lock,
        threshold_lock_sha256="e" * 64,
        feedback_plans=feedback,
    )


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], dict]] = []

    async def complete(self, model, messages, **kwargs):
        copied = [dict(message) for message in messages]
        self.calls.append((model, copied, dict(kwargs)))
        purpose = kwargs["purpose"]
        text = "PROBE: 00000000" if purpose == "deployment_active_probe" else "Answer: 6"
        attempt = CallAttemptRecord(
            event_id=f"event-{len(self.calls)}",
            reservation_id=f"reservation-{len(self.calls)}",
            provider="openai",
            model=model,
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="start",
            finished_at="finish",
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            estimated_cost_usd=Decimal("0.001"),
            elapsed_ms=3,
        )
        return CompletionResult(
            text=text,
            tool_calls=(),
            usage=attempt.usage,
            response_id=f"response-{len(self.calls)}",
            request_id=f"request-{len(self.calls)}",
            model_id=model,
            finish_reason="stop",
            cost_usd=Decimal("0.001"),
            attempts=(attempt,),
        )


class TwoPassBuilderTests(unittest.TestCase):
    def test_threshold_lock_accepts_calibration_selections_only(self):
        selection = ThresholdSelection(
            model=MODEL,
            benchmark=task().domain,
            method="trace_rules",
            split="calibration",
            threshold=0.8,
            target_firing_rate=0.5,
            achieved_firing_rate=0.5,
            n_tasks=20,
            calibration_digest="6" * 64,
            selection_rule="task_score_rank_hash_ties",
            tie_break_seed=12012,
            target_fire_count=10,
        )
        lock = threshold_lock_from_calibration(
            calibration_run_id="calibration",
            calibration_manifest_sha256="7" * 64,
            selections=(selection,),
            natural_max_actions_per_task=2,
            matched_actions_per_method=1,
            yoke_anchor_method="trace_rules",
        )
        self.assertEqual(lock.methods[0].threshold, 0.8)
        with self.assertRaisesRegex(DeploymentArtifactError, "calibration"):
            threshold_lock_from_calibration(
                calibration_run_id="bad",
                calibration_manifest_sha256="7" * 64,
                selections=(object(),),
                natural_max_actions_per_task=2,
                matched_actions_per_method=1,
                yoke_anchor_method="trace_rules",
            )

    def test_pass_one_extractor_uses_prefix_records_and_rejects_outcomes(self):
        active = pass_one_trace_from_records(
            model=MODEL,
            benchmark=task().domain,
            task_id=pair_task_id(task()),
            task_sha256=task().task_sha256,
            replicate_id=0,
            method="active_recompute",
            source_trajectory_sha256="1" * 64,
            task_horizon=3,
            records=(
                {
                    "event": "active_probe",
                    "after_task_turn": 1,
                    "variant": "recompute",
                    "source_prefix_sha256": "2" * 64,
                    "source_trajectory_sha256": "1" * 64,
                    "grade": {"passed": False},
                },
            ),
        )
        passive = pass_one_trace_from_records(
            model=MODEL,
            benchmark=task().domain,
            task_id=pair_task_id(task()),
            task_sha256=task().task_sha256,
            replicate_id=0,
            method="frozen_probe:recompute",
            source_trajectory_sha256="3" * 64,
            task_horizon=3,
            records=(
                {
                    "method": "frozen_probe",
                    "variant": "recompute",
                    "checkpoint_turn": 1,
                    "actionable_before_turn": 2,
                    "score": 0.25,
                    "source_prefix_sha256": "4" * 64,
                    "source_trajectory_sha256": "3" * 64,
                },
            ),
        )
        artifact = build_pass_one_observation_artifact(
            source_run_id="pass-one",
            source_manifest_sha256="5" * 64,
            traces=(passive, active),
        )
        self.assertEqual([row.method for row in artifact.traces], [
            "active_recompute",
            "frozen_probe:recompute",
        ])
        self.assertEqual(active.checkpoints[0].score, 1.0)
        contaminated = {
            "method": "trace_rules",
            "checkpoint_turn": 1,
            "score": 0.5,
            "source_prefix_sha256": "4" * 64,
            "evaluation": {"success": False},
        }
        with self.assertRaisesRegex(DeploymentArtifactError, "outcome field"):
            pass_one_trace_from_records(
                model=MODEL,
                benchmark=task().domain,
                task_id=pair_task_id(task()),
                task_sha256=task().task_sha256,
                replicate_id=0,
                method="trace_rules",
                source_trajectory_sha256="3" * 64,
                task_horizon=3,
                records=(contaminated,),
            )

    def test_primary_natural_schedules_are_method_specific(self):
        artifact = build(DeploymentEstimand.NATURAL_THRESHOLD)
        actions = {
            group.observation_method: tuple(row.checkpoint for row in group.actions)
            for group in artifact.groups
        }
        self.assertEqual(
            actions,
            {"active_recompute": (1,), "trace_rules": (2,), "turn_clock": ()},
        )
        for group in artifact.groups:
            self.assertEqual(
                group.schedule.action_checkpoints,
                actions[group.observation_method],
            )
        self.assertNotEqual(actions["active_recompute"], actions["trace_rules"])

    def test_matched_rate_is_score_ranked_not_hash_ranked_and_preserves_timing(self):
        artifact = build(DeploymentEstimand.MATCHED_RATE_TOP_K)
        actions = {
            group.observation_method: tuple(row.checkpoint for row in group.actions)
            for group in artifact.groups
        }
        self.assertEqual(
            actions,
            {"active_recompute": (1,), "trace_rules": (2,), "turn_clock": (1,)},
        )
        self.assertTrue(
            all(len(group.actions) == 1 for group in artifact.groups)
        )
        clock = next(group for group in artifact.groups if group.observation_method == "turn_clock")
        self.assertFalse(clock.actions[0].natural_threshold_fired)

    def test_matched_rate_spends_one_global_budget_not_one_per_task(self):
        first_cells = cells(operators=(Operator.NONE.value, Operator.COMPACT.value))
        second_cells = tuple(
            replace(
                cell,
                cell_id=canonical_json_sha256(
                    {"original": cell.cell_id, "replicate_id": 1}
                )[:24],
                block_id="block-y",
                block_position=index,
                pair_key=replace(cell.pair_key, replicate_id=1),
            )
            for index, cell in enumerate(first_cells)
        )
        second_scores = {
            "active_recompute": (0.2, 0.3),
            "trace_rules": (0.95, 0.1),
            "turn_clock": (0.1, 0.99),
        }
        second_traces = []
        for original in pass_one().traces:
            method = original.method
            second_traces.append(
                replace(
                    original,
                    replicate_id=1,
                    source_trajectory_sha256=canonical_json_sha256(
                        {"method": method, "replicate_id": 1, "trajectory": "pass-one"}
                    ),
                    checkpoints=tuple(
                        PassOneCheckpoint(
                            checkpoint=checkpoint,
                            score=score,
                            source_prefix_sha256=canonical_json_sha256(
                                {
                                    "method": method,
                                    "replicate_id": 1,
                                    "checkpoint": checkpoint,
                                    "kind": "prefix",
                                }
                            ),
                            signal_record_sha256=canonical_json_sha256(
                                {
                                    "method": method,
                                    "replicate_id": 1,
                                    "checkpoint": checkpoint,
                                    "kind": "signal",
                                }
                            ),
                        )
                        for checkpoint, score in enumerate(second_scores[method], 1)
                    ),
                )
            )
        observations = replace(
            pass_one(),
            traces=tuple(
                sorted((*pass_one().traces, *second_traces), key=lambda row: row.identity)
            ),
        )
        artifact = build_deployment_schedule(
            estimand=DeploymentEstimand.MATCHED_RATE_TOP_K,
            cells=(*first_cells, *second_cells),
            pair_manifest_sha256="c" * 64,
            pass_one=observations,
            pass_one_artifact_sha256="d" * 64,
            threshold_lock=threshold_lock(),
            threshold_lock_sha256="e" * 64,
            feedback_plans={},
        )

        actions = {
            (group.observation_method, group.replicate_id): tuple(
                action.checkpoint for action in group.actions
            )
            for group in artifact.groups
        }
        self.assertEqual(
            actions,
            {
                ("active_recompute", 0): (1,),
                ("active_recompute", 1): (),
                ("trace_rules", 0): (),
                ("trace_rules", 1): (1,),
                ("turn_clock", 0): (),
                ("turn_clock", 1): (2,),
            },
        )
        for method in METHODS:
            self.assertEqual(
                sum(len(group.actions) for group in artifact.groups if group.observation_method == method),
                1,
            )

    def test_yoking_copies_locked_anchor_trigger_receipts(self):
        artifact = build(DeploymentEstimand.YOKED_ANCHOR)
        for group in artifact.groups:
            self.assertEqual(tuple(row.checkpoint for row in group.actions), (2,))
            self.assertEqual(group.actions[0].trigger_method, "trace_rules")
            self.assertTrue(group.actions[0].natural_threshold_fired)
        source_hashes = {group.actions[0].signal_record_sha256 for group in artifact.groups}
        self.assertEqual(len(source_hashes), 1)

    def test_missing_method_threshold_and_pass_one_trace_fail_closed(self):
        declared = cells()
        with self.assertRaisesRegex(DeploymentArtifactError, "threshold lock"):
            build_deployment_schedule(
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                cells=declared,
                pair_manifest_sha256="c" * 64,
                pass_one=pass_one(),
                pass_one_artifact_sha256="d" * 64,
                threshold_lock=threshold_lock(("active_recompute", "trace_rules")),
                threshold_lock_sha256="e" * 64,
                feedback_plans={},
            )

        no_control = cells(METHODS, (Operator.COMPACT.value, Operator.REGROUND.value))
        with self.assertRaisesRegex(DeploymentArtifactError, "no-intervention"):
            build_deployment_schedule(
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                cells=no_control,
                pair_manifest_sha256="c" * 64,
                pass_one=pass_one(),
                pass_one_artifact_sha256="d" * 64,
                threshold_lock=threshold_lock(),
                threshold_lock_sha256="e" * 64,
                feedback_plans={},
            )
        with self.assertRaisesRegex(DeploymentArtifactError, "pass one"):
            build_deployment_schedule(
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                cells=declared,
                pair_manifest_sha256="c" * 64,
                pass_one=pass_one(("active_recompute", "trace_rules")),
                pass_one_artifact_sha256="d" * 64,
                threshold_lock=threshold_lock(),
                threshold_lock_sha256="e" * 64,
                feedback_plans={},
            )

    def test_outcomes_and_unknown_fields_are_rejected_from_pass_one(self):
        raw = pass_one().as_dict()
        raw["outcomes"] = [{"success": False}]
        with self.assertRaises(DeploymentArtifactError):
            PassOneObservationArtifact.from_dict(raw)
        raw = pass_one().as_dict()
        raw["outcome_fields_present"] = True
        with self.assertRaisesRegex(DeploymentArtifactError, "outcome-free"):
            PassOneObservationArtifact.from_dict(raw)

    def test_recomputation_detects_score_threshold_and_action_tampering(self):
        original = build(DeploymentEstimand.NATURAL_THRESHOLD)
        index = {
            (task().domain, pair_task_id(task()), task().task_sha256): task()
        }
        validate_deployment_schedule(
            original,
            cells=cells(),
            task_index=index,
            pass_one=pass_one(),
            threshold_lock=threshold_lock(),
        )

        raw = original.as_dict()
        raw["groups"][0]["actions"][0]["score"] = 0.95
        tampered = type(original).from_dict(raw)
        with self.assertRaisesRegex(DeploymentArtifactError, "does not reproduce"):
            validate_deployment_schedule(
                tampered,
                cells=cells(),
                task_index=index,
                pass_one=pass_one(),
                threshold_lock=threshold_lock(),
            )

        changed_rows = list(threshold_lock().methods)
        first = changed_rows[0]
        changed_rows[0] = LockedMethodThreshold(
            model=first.model,
            benchmark=first.benchmark,
            method=first.method,
            threshold=0.95,
            target_firing_rate=first.target_firing_rate,
            achieved_firing_rate=first.achieved_firing_rate,
            calibration_n_tasks=first.calibration_n_tasks,
            calibration_digest=first.calibration_digest,
            selection_rule=first.selection_rule,
            tie_break_seed=first.tie_break_seed,
            calibration_target_fire_count=first.calibration_target_fire_count,
        )
        drifted = ThresholdLockArtifact(
            calibration_run_id="calibration-run",
            calibration_manifest_sha256="b" * 64,
            natural_max_actions_per_task=2,
            matched_actions_per_method=1,
            yoke_anchor_method="trace_rules",
            methods=tuple(changed_rows),
        )
        with self.assertRaisesRegex(DeploymentArtifactError, "does not reproduce"):
            validate_deployment_schedule(
                original,
                cells=cells(),
                task_index=index,
                pass_one=pass_one(),
                threshold_lock=drifted,
            )

    def test_schedule_freeze_refuses_post_outcome_build_and_is_write_once(self):
        artifact = build(DeploymentEstimand.NATURAL_THRESHOLD)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schedule = root / "schedule.json"
            outcomes = root / "outcomes"
            freeze_deployment_schedule(
                schedule, artifact, outcome_artifacts_root=outcomes
            )
            self.assertEqual(load_deployment_schedule(schedule), artifact)
            with self.assertRaises(FileExistsError):
                freeze_deployment_schedule(
                    schedule, artifact, outcome_artifacts_root=outcomes
                )
            outcomes.mkdir()
            atomic_write_json(outcomes / "cell.json", {"success": True})
            with self.assertRaisesRegex(DeploymentArtifactError, "already exist"):
                freeze_deployment_schedule(
                    root / "late.json", artifact, outcome_artifacts_root=outcomes
                )
            outcome_file = root / "outcome-file.json"
            atomic_write_json(outcome_file, {"success": True})
            with self.assertRaisesRegex(DeploymentArtifactError, "already exist"):
                freeze_deployment_schedule(
                    root / "later.json",
                    artifact,
                    outcome_artifacts_root=outcome_file,
                )

    def test_cli_spend_gate_precedes_artifact_access(self):
        self.assertEqual(
            main(
                [
                    "run-evolving",
                    "--run-id",
                    "r",
                    "--dataset",
                    "/missing",
                    "--dataset-sha256",
                    "0" * 64,
                    "--build-receipt",
                    "/missing",
                    "--tasks",
                    "/missing",
                    "--pass-one",
                    "/missing",
                    "--thresholds",
                    "/missing",
                    "--schedule",
                    "/missing",
                ]
            ),
            2,
        )


class DeploymentRunnerTests(unittest.IsolatedAsyncioTestCase):
    def _runner_design(self, method: str, scores: tuple[float, float]):
        declared = cells((method,), OPERATORS)
        observations = pass_one((method,), {method: scores})
        lock = threshold_lock((method,))
        actions = tuple(
            index for index, score in enumerate(scores, 1) if score >= 0.8
        )
        feedback = feedback_for(declared, {method: actions})
        artifact = build_deployment_schedule(
            estimand=DeploymentEstimand.NATURAL_THRESHOLD,
            cells=declared,
            pair_manifest_sha256="c" * 64,
            pass_one=observations,
            pass_one_artifact_sha256="d" * 64,
            threshold_lock=lock,
            threshold_lock_sha256="e" * 64,
            feedback_plans=feedback,
        )
        return declared, artifact, artifact.groups[0]

    async def _run(self, cell, artifact, group, root):
        fake = FakeTransport()
        result = await run_deployment_task(
            run_id="deployment-test",
            cell=cell,
            task=task(),
            group=group,
            schedule_artifact=artifact,
            schedule_artifact_sha256="f" * 64,
            transport=fake,
            event_path=root / f"{cell.cell_id}.jsonl",
            output_path=root / f"{cell.cell_id}.json",
            yes_spend=True,
            config=HarnessConfig(task_max_output_tokens=20, probe_max_output_tokens=20),
        )
        return fake, result

    async def test_active_observes_every_checkpoint_and_actions_only_when_frozen(self):
        declared, artifact, group = self._runner_design(
            "active_recompute", (0.9, 0.95)
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cell in declared:
                fake, result = await self._run(cell, artifact, group, root)
                self.assertEqual(
                    [call[2]["purpose"] for call in fake.calls],
                    [
                        "deployment_agent_turn",
                        "deployment_active_probe",
                        "deployment_agent_turn",
                        "deployment_active_probe",
                        "deployment_agent_turn",
                    ],
                )
                self.assertEqual(len(result["probe_records"]), 2)
                self.assertTrue(
                    all(not row["grade"]["passed"] for row in result["probe_records"])
                )
                self.assertEqual(
                    [row["checkpoint"] for row in result["intervention_records"]],
                    [1, 2],
                )
                self.assertTrue(
                    all(row["signal_frozen_two_pass"] for row in result["intervention_records"])
                )
                self.assertTrue(
                    all(row["signal_record_sha256"] for row in result["intervention_records"])
                )
                self.assertEqual(result["accounting"]["total"]["calls"], 5)
                self.assertTrue(result["evaluation"]["success"])

    async def test_passive_is_zero_carry_but_uses_frozen_signal_receipt(self):
        declared, artifact, group = self._runner_design("trace_rules", (0.1, 0.9))
        chosen = next(cell for cell in declared if cell.operator == Operator.REGROUND.value)
        with tempfile.TemporaryDirectory() as tmp:
            fake, result = await self._run(chosen, artifact, group, Path(tmp))
            self.assertEqual(len(fake.calls), 3)
            self.assertEqual(result["probe_records"], [])
            self.assertEqual(len(result["intervention_records"]), 1)
            event = result["intervention_records"][0]
            self.assertEqual(event["checkpoint"], 2)
            self.assertEqual(event["signal_method"], "trace_rules")
            self.assertTrue(event["signal_frozen_two_pass"])
            state = next(
                message["content"]
                for message in result["messages"]
                if "PUBLIC_STATE_JSON" in message["content"]
            )
            self.assertIn("Actually use 3 apples.", state)
            self.assertNotIn("What is twice that?", state)

    async def test_no_intervention_operator_keeps_observation_burden(self):
        declared, artifact, group = self._runner_design("active_recompute", (0.9, 0.1))
        chosen = next(cell for cell in declared if cell.operator == Operator.NONE.value)
        with tempfile.TemporaryDirectory() as tmp:
            fake, result = await self._run(chosen, artifact, group, Path(tmp))
            self.assertEqual(len(fake.calls), 5)
            self.assertEqual(len(result["intervention_records"]), 1)
            self.assertEqual(
                result["intervention_records"][0]["intervention_type"], "none"
            )
            self.assertEqual(
                result["intervention_records"][0]["dropped_message_count"], 0
            )

    async def test_low_level_gate_and_idempotency_fail_closed(self):
        declared, artifact, group = self._runner_design("trace_rules", (0.1, 0.9))
        chosen = next(cell for cell in declared if cell.operator == Operator.NONE.value)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = FakeTransport()
            kwargs = dict(
                run_id="deployment-test",
                cell=chosen,
                task=task(),
                group=group,
                schedule_artifact=artifact,
                schedule_artifact_sha256="f" * 64,
                transport=fake,
                event_path=root / "events.jsonl",
                output_path=root / "output.json",
                config=HarnessConfig(task_max_output_tokens=20),
            )
            with self.assertRaisesRegex(DeploymentArtifactError, "yes_spend"):
                await run_deployment_task(**kwargs)
            self.assertEqual(fake.calls, [])
            first = await run_deployment_task(**kwargs, yes_spend=True)
            calls = len(fake.calls)
            second = await run_deployment_task(**kwargs, yes_spend=True)
            self.assertEqual(first, second)
            self.assertEqual(len(fake.calls), calls)
            changed = read_json(root / "output.json")
            changed["accounting"]["total"]["calls"] = 99
            atomic_write_json(root / "output.json", changed)
            with self.assertRaisesRegex(DeploymentArtifactError, "accounting"):
                await run_deployment_task(**kwargs, yes_spend=True)
            self.assertEqual(len(fake.calls), calls)

    async def test_dispatcher_binds_all_three_receipts_and_shared_budget(self):
        method = "trace_rules"
        declared, synthetic_schedule, _group = self._runner_design(method, (0.9, 0.1))
        observations = pass_one((method,), {method: (0.9, 0.1)})
        lock = threshold_lock((method,))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "deployment-budget-test"
            layout = RunLayout.for_run(root, run_id)
            layout.create()
            task_manifest = root / "tasks.jsonl"
            dataset = root / "dataset.json"
            build_receipt = root / "build-receipt.json"
            pass_path = root / "pass-one.json"
            threshold_path = root / "thresholds.json"
            schedule_path = root / "schedule.json"
            atomic_write_bytes(
                dataset, canonical_json_bytes({"dataset": "deployment-test"})
            )
            atomic_write_json(
                build_receipt,
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "upstream_commit": (
                        "993d6be9597ac03854b46362ccd647eb1bfd267a"
                    ),
                    "shared_across_target_arms_and_models": True,
                    "frozen_dataset": {"sha256": sha256_file(dataset)},
                },
            )
            freeze_task_manifest(task_manifest, (task(),))
            pass_digest = freeze_pass_one_observations(pass_path, observations)
            threshold_digest = freeze_threshold_lock(threshold_path, lock)
            atomic_write_jsonl(layout.pairs, [cell.as_dict() for cell in declared])
            pair_digest = sha256_file(layout.pairs)
            feedback = feedback_for(declared, {method: (1,)})
            schedule = build_deployment_schedule(
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                cells=declared,
                pair_manifest_sha256=pair_digest,
                pass_one=observations,
                pass_one_artifact_sha256=pass_digest,
                threshold_lock=lock,
                threshold_lock_sha256=threshold_digest,
                feedback_plans=feedback,
            )
            schedule_digest = freeze_deployment_schedule(
                schedule_path,
                schedule,
                outcome_artifacts_root=layout.results / "deployment",
            )
            runtime_harness = HarnessConfig(task_max_output_tokens=20)
            runtime_compaction = CompactionConfig()
            manifest = build_manifest(
                run_id=run_id,
                stage=Stage.SMOKE,
                repository_root=ROOT,
                pair_manifest_sha256=pair_digest,
                models=(MODEL,),
                arms=(method,),
                operators=OPERATORS,
                randomization_seed=12,
                benchmark_receipts=(
                    ArtifactReceipt.from_file(
                        "task_manifest", task_manifest, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        "evolving_rendered_dataset", dataset, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        "evolving_build_receipt", build_receipt, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(PASS_ONE_RECEIPT, pass_path, workspace=ROOT),
                    ArtifactReceipt.from_file(
                        THRESHOLD_LOCK_RECEIPT, threshold_path, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        DEPLOYMENT_SCHEDULE_RECEIPT, schedule_path, workspace=ROOT
                    ),
                ),
                extra_config={
                    "n_cells": len(declared),
                    "deployment_mode": "two_pass_frozen",
                    "deployment_estimand": DeploymentEstimand.NATURAL_THRESHOLD.value,
                    "deployment_runtime": deployment_runtime_config(
                        runtime_harness, runtime_compaction
                    ),
                },
            )
            write_manifest_once(layout.manifest, manifest)
            fake = FakeTransport()
            ledger_sentinel = object()
            with patch(
                "experiments12.deployment12._stage_ledger",
                return_value=ledger_sentinel,
            ) as stage_ledger, patch(
                "experiments12.deployment12.Transport", return_value=fake
            ) as transport_class:
                with self.assertRaisesRegex(
                    DeploymentArtifactError, "runtime configuration"
                ):
                    await execute_deployment_run(
                        run_id=run_id,
                        task_manifest_path=task_manifest,
                        pass_one_path=pass_path,
                        threshold_lock_path=threshold_path,
                        schedule_path=schedule_path,
                        tasks=(task(),),
                        yes_spend=True,
                        artifacts_root=root,
                        max_new_cells=2,
                        config=HarnessConfig(task_max_output_tokens=21),
                        compaction_config=runtime_compaction,
                        evolving_dataset_path=dataset,
                        evolving_build_receipt_path=build_receipt,
                    )
                with self.assertRaisesRegex(
                    DeploymentArtifactError, "runtime configuration"
                ):
                    await execute_deployment_run(
                        run_id=run_id,
                        task_manifest_path=task_manifest,
                        pass_one_path=pass_path,
                        threshold_lock_path=threshold_path,
                        schedule_path=schedule_path,
                        tasks=(task(),),
                        yes_spend=True,
                        artifacts_root=root,
                        max_new_cells=2,
                        config=runtime_harness,
                        compaction_config=CompactionConfig(max_summary_bytes=1024),
                        evolving_dataset_path=dataset,
                        evolving_build_receipt_path=build_receipt,
                    )
                summary = await execute_deployment_run(
                    run_id=run_id,
                    task_manifest_path=task_manifest,
                    pass_one_path=pass_path,
                    threshold_lock_path=threshold_path,
                    schedule_path=schedule_path,
                    tasks=(task(),),
                    yes_spend=True,
                    artifacts_root=root,
                    max_new_cells=2,
                    config=runtime_harness,
                    compaction_config=runtime_compaction,
                    evolving_dataset_path=dataset,
                    evolving_build_receipt_path=build_receipt,
                )
            stage_ledger.assert_called_once_with(layout, run_id, Stage.SMOKE)
            self.assertEqual(layout.ledger, root / "_global_budget.sqlite3")
            self.assertIs(transport_class.call_args.args[0], ledger_sentinel)
            self.assertEqual(summary.completed_cells, 2)
            self.assertEqual(summary.skipped_cells, 2)
            self.assertEqual(len(fake.calls), 6)
            self.assertNotEqual(synthetic_schedule.pair_manifest_sha256, pair_digest)

            report = deployment_completeness(layout, declared)
            self.assertEqual(report.complete, 2)
            self.assertEqual(report.missing, 2)
            outputs = {
                cell.cell_id: read_json(
                    layout.results / "deployment" / f"{cell.cell_id}.json"
                )
                for cell in declared[:2]
            }
            with self.assertRaisesRegex(DeploymentArtifactError, "exactly cover"):
                extract_deployment_outcomes(declared, outputs)


if __name__ == "__main__":
    unittest.main()
