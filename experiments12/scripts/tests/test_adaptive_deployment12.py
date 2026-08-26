from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments12.adaptive_deployment12 import (
    ADAPTIVE_POLICY,
    ADAPTIVE_DEPLOYMENT_MODE,
    PRIMARY_MAX_ACTIONS_PER_TASK,
    PRIMARY_REPLICATES,
    AdaptiveDeploymentError,
    _runtime_config,
    _validate_existing,
    execute_adaptive_run,
    main,
    parser,
    prepare_adaptive_run,
    run_adaptive_task,
    validate_adaptive_design,
)
from experiments12.core.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.schemas import CallAttemptRecord, CallStatus, PairKey, TokenUsage
from experiments12.core.transport import CompletionResult
from experiments12.deployment12 import (
    THRESHOLD_LOCK_RECEIPT,
    LockedMethodThreshold,
    ThresholdLockArtifact,
    freeze_threshold_lock,
)
from experiments12.domains.base import DomainTask, DomainTurn, canonical_json_sha256
from experiments12.harness12 import HarnessConfig
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.operators12 import CompactionConfig
from experiments12.pairing12 import JobCell
from experiments12.passive_spec12 import canonical_passive_monitor_spec
from experiments12.planning_lock12 import ScientificLaunchBinding
from experiments12.runner12 import freeze_task_manifest, pair_task_id
from experiments12.source_registry12 import SourceAllocationBinding
from experiments12.spec12 import Operator, Stage


MODEL = "gpt-5.6-luna"


def task() -> DomainTask:
    turns = (
        DomainTurn(1, "Start with 2 apples."),
        DomainTurn(2, "Actually use 3 apples."),
        DomainTurn(3, "What is twice that?"),
    )
    source = canonical_json_sha256({"dataset": "adaptive-test"})
    digest = canonical_json_sha256(
        {
            "domain": "evolving_intent_gsm8k",
            "task_id": "adaptive-x",
            "condition": "t7",
            "turns": [turn.user_message for turn in turns],
            "source": source,
        }
    )
    return DomainTask(
        domain="evolving_intent_gsm8k",
        task_id="adaptive-x",
        condition="t7",
        turns=turns,
        evaluation_label="6",
        source_sha256=source,
        task_sha256=digest,
        public_metadata=(("split", "confirmatory"),),
    )


def cell(method: str, operator: str, *, suffix: str = "") -> JobCell:
    current = task()
    pair = PairKey(
        model=MODEL,
        domain=current.domain,
        task_id=pair_task_id(current),
        replicate_id=0,
        task_sha256=current.task_sha256,
    )
    return JobCell(
        cell_id=canonical_json_sha256(
            {"method": method, "operator": operator, "suffix": suffix}
        )[:24],
        block_id="adaptive-block",
        block_position=0,
        pair_key=pair,
        arm=method,
        operator=operator,
        seed=12012,
    )


def threshold_lock(
    methods: tuple[str, ...],
    *,
    thresholds: dict[str, float] | None = None,
    cap: int = 1,
) -> ThresholdLockArtifact:
    values = thresholds or {method: 0.5 for method in methods}
    rows = tuple(
        sorted(
            (
                LockedMethodThreshold(
                    model=MODEL,
                    benchmark=task().domain,
                    method=method,
                    threshold=values[method],
                    target_firing_rate=0.5,
                    achieved_firing_rate=0.5,
                    calibration_n_tasks=20,
                    calibration_digest=canonical_json_sha256(
                        {"method": method, "source": "adaptive-calibration"}
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
        natural_max_actions_per_task=cap,
        matched_actions_per_method=1,
        yoke_anchor_method=methods[0],
        methods=rows,
    )


class FakeTransport:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[tuple[str, list[dict], dict]] = []
        self.fail_on_call = fail_on_call

    async def complete(self, model, messages, **kwargs):
        copied = [dict(message) for message in messages]
        self.calls.append((model, copied, dict(kwargs)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("synthetic transport interruption")
        purpose = kwargs["purpose"]
        if purpose == "adaptive_trace_judge":
            text = '{"risk":0.9,"concerns":["drift"],"evidence":["apples"]}'
        elif purpose == "adaptive_frozen_quiz":
            text = "A1: 1\nA2: 1\nA3: start with 2 apples"
        elif purpose in {"adaptive_active_probe", "adaptive_frozen_probe"}:
            text = "PROBE: 00000000"
        else:
            text = "Answer: 6"
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
            usage=TokenUsage(input_tokens=10 + len(self.calls), output_tokens=2),
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


class AdaptiveRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def run_cell(
        self,
        *,
        root: Path,
        method: str,
        operator: str,
        threshold: float = 0.5,
        cap: int = 1,
        transport: FakeTransport | None = None,
        yes_spend: bool = True,
        suffix: str = "",
    ):
        locked = threshold_lock(
            (method,), thresholds={method: threshold}, cap=cap
        )
        selected = cell(method, operator, suffix=suffix)
        fake = transport or FakeTransport()
        result = await run_adaptive_task(
            run_id="adaptive-test",
            cell=selected,
            task=task(),
            threshold=locked.methods[0],
            threshold_lock=locked,
            threshold_lock_sha256="c" * 64,
            manifest_sha256="d" * 64,
            pair_manifest_sha256="e" * 64,
            passive_monitor_spec=canonical_passive_monitor_spec(),
            transport=fake,
            event_path=root / f"{selected.cell_id}.jsonl",
            output_path=root / f"{selected.cell_id}.json",
            yes_spend=yes_spend,
            config=HarnessConfig(task_max_output_tokens=30),
        )
        return selected, fake, result

    async def test_active_setup_carries_and_later_probe_sees_compacted_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, fake, result = await self.run_cell(
                root=Path(tmp),
                method="active_counter",
                operator=Operator.COMPACT.value,
            )
        purposes = [call[2]["purpose"] for call in fake.calls]
        self.assertEqual(
            purposes,
            [
                "adaptive_agent_turn",
                "adaptive_active_probe",
                "adaptive_agent_turn",
                "adaptive_active_probe",
                "adaptive_agent_turn",
            ],
        )
        self.assertIn("counter starts", fake.calls[0][1][0]["content"])
        self.assertIn("LOSSY SELF-SUMMARY", str(fake.calls[3][1]))
        self.assertEqual([row["action_selected"] for row in result["decision_records"]], [True, False])
        self.assertEqual(len(result["intervention_records"]), 1)
        for signal in result["signal_records"]:
            self.assertTrue(signal["carried_into_target"])
            self.assertNotEqual(
                signal["source_prefix_before_observation_sha256"],
                signal["source_prefix_sha256"],
            )
        intervention = result["intervention_records"][0]
        self.assertFalse(intervention["signal_frozen_two_pass"])
        self.assertEqual(
            intervention["signal_record_sha256"],
            result["signal_records"][0]["signal_record_sha256"],
        )
        self.assertEqual(result["deployment_mode"], ADAPTIVE_DEPLOYMENT_MODE)
        self.assertTrue(result["evaluation"]["success"])

    async def test_coherent_history_omission_fails_exact_provider_free_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected, _fake, _result = await self.run_cell(
                root=root,
                method="active_counter",
                operator=Operator.COMPACT.value,
            )
            output_path = root / f"{selected.cell_id}.json"
            event_path = root / f"{selected.cell_id}.jsonl"
            output = read_json(output_path)
            events = read_jsonl(event_path)

            # This coordinated rewrite used to pass: the second request claims
            # a prefix with neither carried probe nor compaction history, while
            # the final transcript is replaced with a self-consistent empty one.
            second_task = next(
                row
                for row in events
                if row.get("event") == "task_turn" and row.get("task_turn") == 2
            )
            second_task["request_prefix_sha256"] = sha256_json(
                [second_task["user_message"]]
            )
            output["task_records"] = [
                row for row in events[:-1] if row.get("event") == "task_turn"
            ]
            output["messages"] = []
            output["transcript_sha256"] = sha256_json([])
            output["event_log_prefix_sha256"] = sha256_json(events[:-1])
            atomic_write_json(output_path, output)
            events[-1]["transcript_sha256"] = output["transcript_sha256"]
            events[-1]["output_sha256"] = sha256_file(output_path)
            atomic_write_jsonl(event_path, events)

            with self.assertRaisesRegex(
                AdaptiveDeploymentError, "request/history|final carried history"
            ):
                _validate_existing(
                    output_file=output_path,
                    event_file=event_path,
                    start=events[0],
                    task=task(),
                    compaction_config=CompactionConfig(),
                )

    async def test_passive_probe_is_zero_carry_and_resenses_regrounded_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            _selected, fake, result = await self.run_cell(
                root=Path(tmp),
                method="frozen_probe:recompute",
                operator=Operator.REGROUND.value,
                threshold=0.0,
            )
        self.assertEqual(
            [call[2]["purpose"] for call in fake.calls],
            [
                "adaptive_agent_turn",
                "adaptive_frozen_probe",
                "adaptive_agent_turn",
                "adaptive_frozen_probe",
                "adaptive_agent_turn",
            ],
        )
        discarded_prompt = fake.calls[1][1][-1]["content"]
        self.assertNotIn(discarded_prompt, str(fake.calls[2][1]))
        self.assertIn("PUBLIC_STATE_JSON", str(fake.calls[2][1]))
        self.assertIn("PUBLIC_STATE_JSON", str(fake.calls[3][1]))
        for signal in result["signal_records"]:
            self.assertFalse(signal["carried_into_target"])
            self.assertEqual(
                signal["source_prefix_before_observation_sha256"],
                signal["source_prefix_sha256"],
            )
        self.assertEqual(
            result["accounting"]["by_category"]["observer"]["calls"], 2
        )
        self.assertEqual([row["action_selected"] for row in result["decision_records"]], [True, False])

    async def test_none_and_operator_controls_keep_equal_observation_burden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _none_cell, none_transport, none = await self.run_cell(
                root=root,
                method="turn_clock",
                operator=Operator.NONE.value,
                threshold=0.2,
                suffix="none",
            )
            _compact_cell, compact_transport, compact = await self.run_cell(
                root=root,
                method="turn_clock",
                operator=Operator.COMPACT.value,
                threshold=0.2,
                suffix="compact",
            )
        self.assertEqual(len(none_transport.calls), len(compact_transport.calls))
        self.assertEqual(none["observation_burden"], compact["observation_burden"])
        self.assertEqual(none["observation_burden"]["checkpoints"], 2)
        self.assertEqual(none["observation_burden"]["paid_observer_calls"], 0)
        self.assertEqual(none["intervention_records"][0]["intervention_type"], "none")
        self.assertEqual(
            compact["intervention_records"][0]["intervention_type"], "compact"
        )
        # The action cap does not stop later sensing in either cell.
        self.assertEqual(len(none["signal_records"]), 2)
        self.assertEqual(len(compact["signal_records"]), 2)

    async def test_passive_quiz_judge_rules_and_context_use_current_semantics(self):
        expectations = {
            "frozen_quiz": 2,
            "trace_judge": 2,
            "trace_rules": 0,
            "context_use": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (method, observer_calls) in enumerate(expectations.items()):
                _selected, _fake, result = await self.run_cell(
                    root=root,
                    method=method,
                    operator=Operator.NONE.value,
                    threshold=0.0,
                    suffix=str(index),
                )
                self.assertEqual(len(result["signal_records"]), 2)
                self.assertEqual(
                    result["accounting"]["by_category"]["observer"]["calls"],
                    observer_calls,
                )
                self.assertTrue(
                    all(
                        row["passive_monitor_spec_sha256"] is not None
                        for row in result["signal_records"]
                    )
                )

    async def test_gate_idempotency_tampering_and_partial_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            locked = threshold_lock(("trace_rules",), thresholds={"trace_rules": 0.0})
            selected = cell("trace_rules", Operator.NONE.value)
            fake = FakeTransport()
            kwargs = dict(
                run_id="adaptive-test",
                cell=selected,
                task=task(),
                threshold=locked.methods[0],
                threshold_lock=locked,
                threshold_lock_sha256="c" * 64,
                manifest_sha256="d" * 64,
                pair_manifest_sha256="e" * 64,
                passive_monitor_spec=canonical_passive_monitor_spec(),
                transport=fake,
                event_path=root / "events.jsonl",
                output_path=root / "output.json",
                config=HarnessConfig(task_max_output_tokens=30),
            )
            with self.assertRaisesRegex(AdaptiveDeploymentError, "yes_spend"):
                await run_adaptive_task(**kwargs)
            self.assertEqual(fake.calls, [])
            first = await run_adaptive_task(**kwargs, yes_spend=True)
            calls = len(fake.calls)
            second = await run_adaptive_task(**kwargs, yes_spend=True)
            self.assertEqual(first, second)
            self.assertEqual(len(fake.calls), calls)
            changed = dict(first)
            changed["messages"] = [*changed["messages"], {"role": "user", "content": "tampered"}]
            atomic_write_json(root / "output.json", changed)
            with self.assertRaisesRegex(AdaptiveDeploymentError, "torn|changed"):
                await run_adaptive_task(**kwargs, yes_spend=True)

            interrupted = FakeTransport(fail_on_call=2)
            partial_kwargs = {
                **kwargs,
                "cell": cell("trace_rules", Operator.NONE.value, suffix="partial"),
                "transport": interrupted,
                "event_path": root / "partial.jsonl",
                "output_path": root / "partial-output.json",
            }
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                await run_adaptive_task(**partial_kwargs, yes_spend=True)
            replacement = FakeTransport()
            partial_kwargs["transport"] = replacement
            with self.assertRaisesRegex(AdaptiveDeploymentError, "partial"):
                await run_adaptive_task(**partial_kwargs, yes_spend=True)
            self.assertEqual(replacement.calls, [])

    async def test_dispatcher_binds_online_manifest_threshold_and_evolving_sources(self):
        methods = ("active_recompute", "trace_rules")
        operators = (Operator.NONE.value, Operator.COMPACT.value)
        locked = threshold_lock(methods, thresholds={method: 0.0 for method in methods})
        declared = tuple(
            cell(method, operator, suffix=f"dispatch-{method}-{operator}")
            for method in methods
            for operator in operators
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "adaptive-dispatch"
            layout = RunLayout.for_run(root, run_id)
            layout.create()
            task_manifest = root / "tasks.jsonl"
            dataset = root / "dataset.json"
            build_receipt = root / "build-receipt.json"
            threshold_path = root / "threshold-lock.json"
            registry = root / "source-registry.json"
            baseline_profile = root / "baseline-profile.json"
            planning_lock = root / "planning-lock.json"
            atomic_write_bytes(
                dataset, canonical_json_bytes({"dataset": "adaptive-test"})
            )
            atomic_write_json(
                build_receipt,
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "upstream_commit": "993d6be9597ac03854b46362ccd647eb1bfd267a",
                    "shared_across_target_arms_and_models": True,
                    "frozen_dataset": {"sha256": sha256_file(dataset)},
                },
            )
            freeze_task_manifest(task_manifest, (task(),))
            threshold_sha = freeze_threshold_lock(threshold_path, locked)
            atomic_write_json(registry, {"synthetic": "dispatcher-lock"})
            atomic_write_json(baseline_profile, {"synthetic": "dispatcher-lock"})
            atomic_write_json(planning_lock, {"synthetic": "dispatcher-lock"})
            launch_binding = ScientificLaunchBinding(
                allocation=SourceAllocationBinding(
                    registry_sha256=sha256_file(registry),
                    benchmark=task().domain,
                    stage="deployment",
                    wave=None,
                    source_ids=(task().task_id,),
                ),
                projection_lock_sha256=sha256_file(planning_lock),
                projected_provider_usd={"openai": "1", "fireworks": "1"},
                required_n_tasks=1,
            )
            atomic_write_jsonl(layout.pairs, [item.as_dict() for item in declared])
            pair_sha = sha256_file(layout.pairs)
            manifest = build_manifest(
                run_id=run_id,
                stage=Stage.CONFIRMATORY,
                repository_root=Path(__file__).resolve().parents[3],
                pair_manifest_sha256=pair_sha,
                models=(MODEL,),
                arms=methods,
                operators=operators,
                randomization_seed=12012,
                benchmark_receipts=(
                    ArtifactReceipt.from_file(
                        "task_manifest", task_manifest, workspace=root
                    ),
                    ArtifactReceipt.from_file(
                        "evolving_rendered_dataset", dataset, workspace=root
                    ),
                    ArtifactReceipt.from_file(
                        "evolving_build_receipt", build_receipt, workspace=root
                    ),
                    ArtifactReceipt.from_file(
                        THRESHOLD_LOCK_RECEIPT, threshold_path, workspace=root
                    ),
                    ArtifactReceipt.from_file(
                        "source_allocation_registry", registry, workspace=root
                    ),
                    ArtifactReceipt.from_file(
                        "measured_baseline_resource_profile",
                        baseline_profile,
                        workspace=root,
                    ),
                    ArtifactReceipt.from_file(
                        "cost_sample_size_projection_lock",
                        planning_lock,
                        workspace=root,
                    ),
                ),
                extra_config={
                    "n_tasks": 1,
                    "n_cells": len(declared),
                    "replicates": 1,
                    "deployment_mode": ADAPTIVE_DEPLOYMENT_MODE,
                    "deployment_policy": ADAPTIVE_POLICY,
                    "threshold_lock_sha256": threshold_sha,
                    "natural_max_actions_per_task": 1,
                    "calibration_manifest_sha256": locked.calibration_manifest_sha256,
                    "adaptive_runtime": _runtime_config(
                        HarnessConfig(task_max_output_tokens=30), CompactionConfig()
                    ),
                    "scientific_launch_lock": launch_binding.as_dict(),
                },
            )
            write_manifest_once(layout.manifest, manifest)
            fake = FakeTransport()
            summary = await execute_adaptive_run(
                run_id=run_id,
                task_manifest_path=task_manifest,
                threshold_lock_path=threshold_path,
                tasks=(task(),),
                yes_spend=True,
                artifacts_root=root,
                max_new_cells=2,
                transport=fake,
                evolving_dataset_path=dataset,
                evolving_build_receipt_path=build_receipt,
                config=HarnessConfig(task_max_output_tokens=30),
            )
            self.assertEqual(summary.completed_cells, 2)
            self.assertEqual(summary.skipped_cells, 2)
            self.assertEqual(summary.failed_cells, 0)
            self.assertTrue(
                (layout.results / "adaptive_deployment" / f"{declared[0].cell_id}.json").is_file()
            )
            with self.assertRaisesRegex(AdaptiveDeploymentError, "scientific"):
                changed = read_json(layout.manifest)
                changed["extra_config"]["scientific_launch_lock"][
                    "source_allocation"
                ]["stage"] = "confirmatory"
                atomic_write_json(layout.manifest, changed)
                await execute_adaptive_run(
                    run_id=run_id,
                    task_manifest_path=task_manifest,
                    threshold_lock_path=threshold_path,
                    tasks=(task(),),
                    yes_spend=True,
                    artifacts_root=root,
                    transport=FakeTransport(),
                    evolving_dataset_path=dataset,
                    evolving_build_receipt_path=build_receipt,
                    config=HarnessConfig(task_max_output_tokens=30),
                )
            atomic_write_json(layout.manifest, manifest)
            with self.assertRaisesRegex(AdaptiveDeploymentError, "manifest"):
                changed = read_json(layout.manifest)
                changed["extra_config"]["deployment_mode"] = "two_pass_frozen"
                atomic_write_json(layout.manifest, changed)
                await execute_adaptive_run(
                    run_id=run_id,
                    task_manifest_path=task_manifest,
                    threshold_lock_path=threshold_path,
                    tasks=(task(),),
                    yes_spend=True,
                    artifacts_root=root,
                    transport=FakeTransport(),
                    evolving_dataset_path=dataset,
                    evolving_build_receipt_path=build_receipt,
                )


class AdaptiveDesignTests(unittest.TestCase):
    def test_primary_online_cap_is_one_and_preparation_rejects_other_values(self):
        args = parser().parse_args(
            [
                "prepare-evolving",
                "--run-id",
                "cap-check",
                "--dataset",
                "/missing",
                "--dataset-sha256",
                "0" * 64,
                "--build-receipt",
                "/missing",
                "--tasks",
                "/missing",
                "--calibration-thresholds",
                "/missing",
                "--calibration-extract",
                "/missing",
                "--calibration-manifest",
                "/missing",
                "--source-registry",
                "/missing",
                "--baseline-profile",
                "/missing",
                "--planning-lock",
                "/missing",
                "--models",
                MODEL,
            ]
        )
        self.assertEqual(args.max_actions_per_task, PRIMARY_MAX_ACTIONS_PER_TASK)
        self.assertEqual(args.replicates, PRIMARY_REPLICATES)
        with self.assertRaisesRegex(AdaptiveDeploymentError, "exactly one"):
            prepare_adaptive_run(
                deployment_run_id="cap-check",
                task_manifest_path="/missing",
                calibration_threshold_path="/missing",
                calibration_extract_path="/missing",
                calibration_manifest_path="/missing",
                source_registry_path="/missing",
                baseline_profile_path="/missing",
                planning_lock_path="/missing",
                tasks=(task(),),
                models=(MODEL,),
                methods=("trace_rules",),
                operators=(Operator.NONE.value, Operator.COMPACT.value),
                natural_max_actions_per_task=2,
                randomization_seed=12012,
            )
        with self.assertRaisesRegex(AdaptiveDeploymentError, "exactly one replicate"):
            prepare_adaptive_run(
                deployment_run_id="replicate-check",
                task_manifest_path="/missing",
                calibration_threshold_path="/missing",
                calibration_extract_path="/missing",
                calibration_manifest_path="/missing",
                source_registry_path="/missing",
                baseline_profile_path="/missing",
                planning_lock_path="/missing",
                tasks=(task(),),
                models=(MODEL,),
                methods=("active_recompute", "trace_rules"),
                operators=(Operator.NONE.value, Operator.COMPACT.value),
                natural_max_actions_per_task=1,
                randomization_seed=12012,
                replicates=2,
            )

    def test_design_requires_exact_thresholds_and_none_operator_pair(self):
        methods = ("active_recompute", "trace_rules")
        locked = threshold_lock(methods)
        declared = tuple(
            cell(method, operator, suffix=f"{method}-{operator}")
            for method in methods
            for operator in (Operator.NONE.value, Operator.COMPACT.value)
        )
        index = {
            (task().domain, pair_task_id(task()), task().task_sha256): task()
        }
        validate_adaptive_design(
            cells=declared, task_index=index, threshold_lock=locked
        )
        no_control = tuple(
            item for item in declared if item.operator != Operator.NONE.value
        )
        with self.assertRaisesRegex(AdaptiveDeploymentError, "operator=none"):
            validate_adaptive_design(
                cells=no_control, task_index=index, threshold_lock=locked
            )
        with self.assertRaisesRegex(AdaptiveDeploymentError, "exactly cover"):
            validate_adaptive_design(
                cells=declared,
                task_index=index,
                threshold_lock=threshold_lock(("active_recompute",)),
            )

    def test_cli_spend_gate_precedes_artifact_access(self):
        self.assertEqual(
            main(
                [
                    "run-evolving",
                    "--run-id",
                    "missing-run",
                    "--dataset",
                    "/missing",
                    "--dataset-sha256",
                    "0" * 64,
                    "--build-receipt",
                    "/missing",
                    "--tasks",
                    "/missing",
                    "--thresholds",
                    "/missing",
                ]
            ),
            2,
        )

    def test_provider_free_preparation_binds_calibration_chain_and_mode(self):
        methods = ("active_recompute", "trace_rules")
        locked = threshold_lock(methods)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_manifest = root / "tasks.jsonl"
            dataset = root / "dataset.json"
            build_receipt = root / "build-receipt.json"
            thresholds = root / "thresholds.json"
            extract = root / "extract.json"
            registry = root / "source-registry.json"
            baseline_profile = root / "baseline-profile.json"
            planning_lock = root / "planning-lock.json"
            calibration_layout = RunLayout.for_run(root, "calibration-run")
            calibration_layout.create()
            freeze_task_manifest(task_manifest, (task(),))
            atomic_write_bytes(
                dataset, canonical_json_bytes({"dataset": "adaptive-test"})
            )
            atomic_write_json(
                build_receipt,
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "upstream_commit": "993d6be9597ac03854b46362ccd647eb1bfd267a",
                    "shared_across_target_arms_and_models": True,
                    "frozen_dataset": {"sha256": sha256_file(dataset)},
                },
            )
            atomic_write_json(extract, {"synthetic": "shared-verifier-is-mocked"})
            atomic_write_json(thresholds, {"synthetic": "shared-verifier-is-mocked"})
            atomic_write_json(registry, {"synthetic": "launch-gate-is-mocked"})
            atomic_write_json(
                baseline_profile, {"synthetic": "launch-gate-is-mocked"}
            )
            atomic_write_json(planning_lock, {"synthetic": "launch-gate-is-mocked"})
            atomic_write_json(
                calibration_layout.manifest,
                {"run_id": "calibration-run", "stage": "calibration"},
            )
            calibration_sha = sha256_file(calibration_layout.manifest)
            payload = {
                "source_run_id": "calibration-run",
                "source_manifest_sha256": calibration_sha,
            }
            launch_binding = ScientificLaunchBinding(
                allocation=SourceAllocationBinding(
                    registry_sha256=sha256_file(registry),
                    benchmark=task().domain,
                    stage="deployment",
                    wave=None,
                    source_ids=(task().task_id,),
                ),
                projection_lock_sha256=sha256_file(planning_lock),
                projected_provider_usd={"openai": "1", "fireworks": "1"},
                required_n_tasks=1,
            )
            with patch(
                "experiments12.adaptive_deployment12.verify_analysis_threshold_derivation",
                return_value=(payload, sha256_file(extract)),
            ) as derivation, patch(
                "experiments12.adaptive_deployment12.verify_calibration_extract_against_run",
                return_value={},
            ) as calibration_check, patch(
                "experiments12.adaptive_deployment12.deployment_threshold_lock_from_analysis",
                return_value=locked,
            ) as converter, patch(
                "experiments12.adaptive_deployment12.assert_scientific_launch",
                return_value=launch_binding,
            ) as launch_gate:
                result = prepare_adaptive_run(
                    deployment_run_id="online-run",
                    task_manifest_path=task_manifest,
                    calibration_threshold_path=thresholds,
                    calibration_extract_path=extract,
                    calibration_manifest_path=calibration_layout.manifest,
                    source_registry_path=registry,
                    baseline_profile_path=baseline_profile,
                    planning_lock_path=planning_lock,
                    tasks=(task(),),
                    models=(MODEL,),
                    methods=methods,
                    operators=(Operator.NONE.value, Operator.COMPACT.value),
                    natural_max_actions_per_task=1,
                    randomization_seed=12012,
                    artifacts_root=root,
                    evolving_dataset_path=dataset,
                    evolving_build_receipt_path=build_receipt,
                )
            derivation.assert_called_once()
            calibration_check.assert_called_once()
            converter.assert_called_once()
            launch_gate.assert_called_once()
            launch_kwargs = launch_gate.call_args.kwargs
            self.assertEqual(launch_kwargs["stage"], Stage.CONFIRMATORY)
            self.assertEqual(launch_kwargs["allocation_stage"], "deployment")
            self.assertEqual(launch_kwargs["design_family"], "deployment")
            manifest = read_json(result.manifest_path)
            self.assertEqual(
                manifest["extra_config"]["deployment_mode"],
                ADAPTIVE_DEPLOYMENT_MODE,
            )
            self.assertEqual(
                manifest["extra_config"]["deployment_policy"], ADAPTIVE_POLICY
            )
            self.assertEqual(
                manifest["extra_config"]["analysis_lock"],
                {
                    "threshold_artifact_sha256": sha256_file(thresholds),
                    "calibration_manifest_sha256": calibration_sha,
                },
            )
            self.assertEqual(
                manifest["extra_config"]["scientific_launch_lock"],
                launch_binding.as_dict(),
            )
            receipt_names = {
                receipt["name"] for receipt in manifest["benchmark_receipts"]
            }
            self.assertTrue(
                {
                    "calibration_manifest",
                    "calibration_analysis_extract",
                    "calibration_analysis_thresholds",
                    "deployment_threshold_lock",
                    "source_allocation_registry",
                    "measured_baseline_resource_profile",
                    "cost_sample_size_projection_lock",
                }.issubset(receipt_names)
            )
            self.assertTrue(result.threshold_lock_path.is_file())


if __name__ == "__main__":
    unittest.main()
