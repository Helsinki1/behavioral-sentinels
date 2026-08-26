"""Provider-free launch gate for the complete Experiment 12 data path.

This is deliberately an integration test rather than another isolated unit
fixture.  It freezes two disjoint observer runs, drives the real transport and
ledger against an in-memory HTTP responder, validates/extracts/calibrates them,
then freezes and executes a two-pass deployment and writes paper figure
sidecars.  No provider SDK, credential, network request, generated input, or
workspace artifact is used.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments12.analysis12 import (
    _trace_from_dict,
    calibrate_thresholds,
    extract_run,
    load_threshold_artifact,
    make_threshold_artifact,
    require_source_task_disjointness,
    write_observer_figures,
    write_signal_figures,
)
from experiments12.cli12 import main as cli_main
from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
)
from experiments12.core.transport import Transport
from experiments12.deployment12 import (
    DEPLOYMENT_SCHEDULE_RECEIPT,
    PASS_ONE_RECEIPT,
    THRESHOLD_LOCK_RECEIPT,
    TWO_PASS_DEPLOYMENT_MODE,
    DeploymentArtifactError,
    DeploymentEstimand,
    build_deployment_schedule,
    build_pass_one_observation_artifact,
    deployment_runtime_config,
    deployment_completeness,
    execute_deployment_run,
    extract_deployment_outcomes,
    freeze_deployment_schedule,
    freeze_pass_one_observations,
    freeze_threshold_lock,
    pass_one_trace_from_records,
    threshold_lock_from_calibration,
)
from experiments12.domains.base import DomainTask, DomainTurn, canonical_json_sha256
from experiments12.domains.evolving_intent import PINNED_COMMIT
from experiments12.figures12 import DeploymentBar, write_deployment_grouped_bars
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    write_manifest_once,
)
from experiments12.metrics12 import grouped_prediction_metrics
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.runner12 import (
    _stage_ledger,
    execute_scripted_run,
    freeze_task_manifest,
    load_pair_cells,
    pair_task_id,
)
from experiments12.spec12 import Operator, Stage
from experiments12.validate12 import validate_run


ROOT = Path(__file__).resolve().parents[3]
MODEL = "gpt-5.6-luna"
ACTIVE = "active_recompute"
PASSIVE = "trace_rules"


class _MockResponse:
    def __init__(self, payload: object, request_id: str) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.status = 200
        self.headers = {"X-Request-ID": request_id}

    def read(self, limit: int = -1) -> bytes:
        return self.body if limit < 0 else self.body[:limit]

    def close(self) -> None:
        return None


class _MockOpenAI:
    """Return deterministic Responses payloads while recording every request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, request: object, *, timeout: float) -> _MockResponse:
        body = json.loads(request.data)
        self.calls.append(body)
        number = len(self.calls)
        items = body.get("input", [])
        text = json.dumps(items, sort_keys=True)
        last_user = next(
            (
                str(item.get("content", ""))
                for item in reversed(items)
                if isinstance(item, dict) and item.get("role") == "user"
            ),
            "",
        )
        output_format = body.get("text", {}).get("format", {})
        if output_format.get("name") == "trace_risk":
            answer = '{"risk":0.75,"concerns":["mock"],"evidence":[]}'
        elif "EXPERIMENT12_ZERO_CARRY_QUIZ" in last_user:
            answer = "A1: 0\nA2: 0\nA3: none\nA4: none"
        elif "ACTIVE CARRIED PROBE" in last_user:
            answer = "PROBE: 00000000"
        else:
            case = next(
                (name for name in ("A", "B", "C", "D") if f"CASE_{name}" in text),
                None,
            )
            clean_correct = case in {"A", "C"}
            carried = "ACTIVE CARRIED PROBE" in text
            correct = (not clean_correct) if carried else clean_correct
            answer = "Answer: 6" if correct else "Answer: 7"
        payload = {
            "id": f"resp_{number}",
            "model": body["model"],
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": answer}],
                }
            ],
            "usage": {
                "input_tokens": 30,
                "output_tokens": 5,
                "total_tokens": 35,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        return _MockResponse(payload, f"mock-request-{number}")


def _task(source_sha256: str, case: str, split: str) -> DomainTask:
    turns = tuple(
        DomainTurn(index, message)
        for index, message in enumerate(
            (
                f"CASE_{case}: Start with three apples.",
                f"CASE_{case}: Keep that value and prepare to double it.",
                f"CASE_{case}: What is twice the current value?",
            ),
            1,
        )
    )
    task_id = f"case-{case.lower()}"
    task_sha256 = canonical_json_sha256(
        {
            "domain": "evolving_intent_gsm8k",
            "task_id": task_id,
            "condition": "t7",
            "turns": [turn.user_message for turn in turns],
            "source_sha256": source_sha256,
        }
    )
    return DomainTask(
        domain="evolving_intent_gsm8k",
        task_id=task_id,
        condition="t7",
        turns=turns,
        evaluation_label="6",
        source_sha256=source_sha256,
        task_sha256=task_sha256,
        public_metadata=(("split", split),),
    )


class LaunchGateTests(unittest.IsolatedAsyncioTestCase):
    def _init_observer_run(
        self,
        *,
        artifacts: Path,
        run_id: str,
        stage: Stage,
        tasks: tuple[DomainTask, ...],
        dataset: Path,
        build_receipt: Path,
    ) -> tuple[RunLayout, Path]:
        task_manifest = artifacts.parent / f"{run_id}-tasks.jsonl"
        freeze_task_manifest(task_manifest, tasks)
        # These synthetic CASE_* identities intentionally sit outside the
        # tracked scientific source registry.  Unit tests exercise that real
        # gate separately; this provider-free integration fixture mocks only
        # the gate result so it can continue testing the downstream data path.
        with patch("experiments12.cli12.assert_scientific_launch", return_value=None):
            result = cli_main(
                [
                    "init",
                    "--run-id",
                    run_id,
                    "--stage",
                    stage.value,
                    "--tasks",
                    str(task_manifest),
                    "--evolving-dataset",
                    str(dataset),
                    "--evolving-build-receipt",
                    str(build_receipt),
                    "--models",
                    MODEL,
                    "--arms",
                    f"clean,{ACTIVE}",
                    "--operators",
                    Operator.NONE.value,
                    "--seed",
                    "12012",
                    "--artifacts",
                    str(artifacts),
                ]
            )
        self.assertEqual(result, 0)
        return RunLayout.for_run(artifacts, run_id), task_manifest

    async def _execute_observer_run(
        self,
        *,
        layout: RunLayout,
        task_manifest: Path,
        tasks: tuple[DomainTask, ...],
        dataset: Path,
        build_receipt: Path,
        opener: _MockOpenAI,
    ) -> None:
        def transport_factory(ledger, event_log_path, **_kwargs):
            return Transport(
                ledger,
                event_log_path,
                environ={"OPENAI_API_KEY": "mock-only"},
                urlopen=opener,
            )

        with patch("experiments12.runner12.Transport", side_effect=transport_factory):
            summary = await execute_scripted_run(
                run_id=layout.root.name,
                task_manifest_path=task_manifest,
                tasks=tasks,
                artifacts_root=layout.root.parent,
                environ={},
                phase="both",
                evolving_dataset_path=dataset,
                evolving_build_receipt_path=build_receipt,
            )
        self.assertEqual(summary.declared_cells, 2 * len(tasks))
        self.assertEqual(summary.completed_cells, 3 * len(tasks))
        self.assertEqual(summary.failed_cells, 0)

    async def test_mocked_freeze_observe_calibrate_deploy_analyze_and_plot(self):
        with tempfile.TemporaryDirectory(prefix="experiment12-launch-gate-") as tmp:
            root = Path(tmp)
            artifacts = root / "runs"
            dataset = root / "rendered-evolving.json"
            build_receipt = root / "rendered-evolving.receipt.json"
            atomic_write_json(dataset, {"mock_cases": ["A", "B", "C", "D"]})
            dataset_sha256 = sha256_file(dataset)
            atomic_write_json(
                build_receipt,
                {
                    "benchmark": "evolving_intent_gsm8k",
                    "upstream_commit": PINNED_COMMIT,
                    "shared_across_target_arms_and_models": True,
                    "frozen_dataset": {"sha256": dataset_sha256},
                },
            )
            calibration_tasks = tuple(
                _task(dataset_sha256, case, "calibration") for case in ("A", "B")
            )
            pass_one_tasks = tuple(
                _task(dataset_sha256, case, "deployment_pass_one")
                for case in ("C", "D")
            )
            opener = _MockOpenAI()

            calibration_layout, calibration_tasks_path = self._init_observer_run(
                artifacts=artifacts,
                run_id="mock-calibration",
                stage=Stage.CALIBRATION,
                tasks=calibration_tasks,
                dataset=dataset,
                build_receipt=build_receipt,
            )
            await self._execute_observer_run(
                layout=calibration_layout,
                task_manifest=calibration_tasks_path,
                tasks=calibration_tasks,
                dataset=dataset,
                build_receipt=build_receipt,
                opener=opener,
            )
            for cell in load_pair_cells(calibration_layout.pairs):
                trajectory = read_json(
                    calibration_layout.trajectories / f"{cell.cell_id}.json"
                )
                if cell.arm == "clean":
                    shadow = read_json(
                        calibration_layout.shadow / f"{cell.cell_id}.json"
                    )
                    self.assertEqual(
                        shadow["source_trajectory_sha256"],
                        trajectory["transcript_sha256"],
                    )
                    self.assertFalse(
                        {"messages", "task_records", "probe_records"}.intersection(
                            shadow
                        )
                    )
                    self.assertNotIn(
                        "ACTIVE CARRIED PROBE", json.dumps(trajectory["messages"])
                    )
                else:
                    self.assertEqual(len(trajectory["probe_records"]), 2)
                    self.assertIn(
                        "ACTIVE CARRIED PROBE", json.dumps(trajectory["messages"])
                    )
            calibration_manifest_sha256 = sha256_file(calibration_layout.manifest)
            calibration_report = validate_run(
                calibration_layout,
                repository_root=ROOT,
                expected_manifest_sha256=calibration_manifest_sha256,
            )
            self.assertTrue(calibration_report.primary_ready, calibration_report.as_dict())
            calibration_extract = extract_run(
                calibration_layout,
                expected_manifest_sha256=calibration_manifest_sha256,
                split="calibration",
            )
            calibration_extract_path = root / "calibration-extract.json"
            atomic_write_json(calibration_extract_path, calibration_extract)
            calibration_traces = tuple(
                _trace_from_dict(row) for row in calibration_extract["signal_traces"]
            )
            thresholds = calibrate_thresholds(
                calibration_traces, target_firing_rate=0.5
            )
            threshold_artifact = make_threshold_artifact(
                calibration_extract,
                calibration_traces,
                thresholds,
                target_firing_rate=0.5,
                source_extract_sha256=sha256_file(calibration_extract_path),
            )
            analysis_threshold_path = root / "analysis-thresholds.json"
            atomic_write_json(analysis_threshold_path, threshold_artifact)
            (
                loaded_thresholds,
                _required_slices,
                calibration_source_tasks,
                _required_passive,
            ) = load_threshold_artifact(read_json(analysis_threshold_path))
            observer_figure_root = root / "figures" / "observer"
            write_observer_figures(calibration_extract, observer_figure_root)
            self.assertGreaterEqual(len(tuple(observer_figure_root.glob("*.svg"))), 6)

            pass_layout, pass_tasks_path = self._init_observer_run(
                artifacts=artifacts,
                run_id="mock-pass-one",
                stage=Stage.BASELINE_GATE,
                tasks=pass_one_tasks,
                dataset=dataset,
                build_receipt=build_receipt,
            )
            await self._execute_observer_run(
                layout=pass_layout,
                task_manifest=pass_tasks_path,
                tasks=pass_one_tasks,
                dataset=dataset,
                build_receipt=build_receipt,
                opener=opener,
            )
            pass_manifest_sha256 = sha256_file(pass_layout.manifest)
            pass_report = validate_run(
                pass_layout,
                repository_root=ROOT,
                expected_manifest_sha256=pass_manifest_sha256,
            )
            self.assertTrue(pass_report.primary_ready, pass_report.as_dict())
            pass_extract = extract_run(
                pass_layout,
                expected_manifest_sha256=pass_manifest_sha256,
                split="deployment_pass_one",
            )
            pass_signal_traces = tuple(
                _trace_from_dict(row) for row in pass_extract["signal_traces"]
            )
            require_source_task_disjointness(
                pass_signal_traces, calibration_source_tasks
            )
            locked_map = {
                (row.model, row.benchmark, row.method): row
                for row in loaded_thresholds
            }
            signal_summaries = grouped_prediction_metrics(
                pass_signal_traces, locked_thresholds=locked_map
            )
            signal_figure_root = root / "figures" / "signal"
            write_signal_figures(signal_summaries, signal_figure_root)
            self.assertEqual(len(tuple(signal_figure_root.glob("*.svg"))), 1)

            pass_cells = load_pair_cells(pass_layout.pairs)
            pass_traces = []
            for cell in pass_cells:
                trajectory = read_json(
                    pass_layout.trajectories / f"{cell.cell_id}.json"
                )
                common = {
                    "model": cell.pair_key.model,
                    "benchmark": cell.pair_key.domain,
                    "task_id": cell.pair_key.task_id,
                    "task_sha256": str(cell.pair_key.task_sha256),
                    "replicate_id": cell.pair_key.replicate_id,
                    "task_horizon": len(trajectory["task_records"]),
                }
                if cell.arm == ACTIVE:
                    pass_traces.append(
                        pass_one_trace_from_records(
                            **common,
                            method=ACTIVE,
                            source_trajectory_sha256=trajectory[
                                "transcript_sha256"
                            ],
                            records=trajectory["probe_records"],
                        )
                    )
                elif cell.arm == "clean":
                    shadow = read_json(pass_layout.shadow / f"{cell.cell_id}.json")
                    pass_traces.append(
                        pass_one_trace_from_records(
                            **common,
                            method=PASSIVE,
                            source_trajectory_sha256=shadow[
                                "source_trajectory_sha256"
                            ],
                            records=tuple(
                                record
                                for record in shadow["records"]
                                if record["method"] == PASSIVE
                            ),
                        )
                    )
            pass_artifact = build_pass_one_observation_artifact(
                source_run_id=pass_layout.root.name,
                source_manifest_sha256=pass_manifest_sha256,
                traces=pass_traces,
            )
            pass_artifact_path = root / "deployment-pass-one.json"
            pass_artifact_sha256 = freeze_pass_one_observations(
                pass_artifact_path, pass_artifact
            )
            deployment_selections = tuple(
                row for row in loaded_thresholds if row.method in {ACTIVE, PASSIVE}
            )
            deployment_thresholds = threshold_lock_from_calibration(
                calibration_run_id=calibration_layout.root.name,
                calibration_manifest_sha256=calibration_manifest_sha256,
                selections=deployment_selections,
                natural_max_actions_per_task=1,
                matched_actions_per_method=1,
                yoke_anchor_method=PASSIVE,
            )
            deployment_threshold_path = root / "deployment-thresholds.json"
            deployment_threshold_sha256 = freeze_threshold_lock(
                deployment_threshold_path, deployment_thresholds
            )

            deployment_run_id = "mock-deployment"
            deployment_layout = RunLayout.for_run(artifacts, deployment_run_id)
            deployment_layout.create()
            deployment_cells = make_pair_manifest(
                tasks=tuple(
                    TaskRef(task.domain, pair_task_id(task), task.task_sha256)
                    for task in pass_one_tasks
                ),
                models=(MODEL,),
                arms=(ACTIVE, PASSIVE),
                operators=(Operator.NONE.value, Operator.REGROUND.value),
                randomization_seed=12012,
            )
            atomic_write_jsonl(
                deployment_layout.pairs,
                [cell.as_dict() for cell in deployment_cells],
            )
            pair_sha256 = sha256_file(deployment_layout.pairs)
            schedule = build_deployment_schedule(
                estimand=DeploymentEstimand.NATURAL_THRESHOLD,
                cells=deployment_cells,
                pair_manifest_sha256=pair_sha256,
                pass_one=pass_artifact,
                pass_one_artifact_sha256=pass_artifact_sha256,
                threshold_lock=deployment_thresholds,
                threshold_lock_sha256=deployment_threshold_sha256,
                feedback_plans={},
            )
            self.assertTrue(all(group.actions for group in schedule.groups))
            self.assertTrue(
                all(
                    action.trigger_method == group.observation_method
                    for group in schedule.groups
                    for action in group.actions
                )
            )
            schedule_path = root / "deployment-schedule.json"
            schedule_sha256 = freeze_deployment_schedule(
                schedule_path,
                schedule,
                outcome_artifacts_root=deployment_layout.results / "deployment",
            )
            deployment_manifest = build_manifest(
                run_id=deployment_run_id,
                stage=Stage.SMOKE,
                repository_root=ROOT,
                pair_manifest_sha256=pair_sha256,
                models=(MODEL,),
                arms=(ACTIVE, PASSIVE),
                operators=(Operator.NONE.value, Operator.REGROUND.value),
                randomization_seed=12012,
                benchmark_receipts=(
                    ArtifactReceipt.from_file(
                        "task_manifest", pass_tasks_path, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        "evolving_rendered_dataset", dataset, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        "evolving_build_receipt", build_receipt, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        PASS_ONE_RECEIPT, pass_artifact_path, workspace=ROOT
                    ),
                    ArtifactReceipt.from_file(
                        THRESHOLD_LOCK_RECEIPT,
                        deployment_threshold_path,
                        workspace=ROOT,
                    ),
                    ArtifactReceipt.from_file(
                        DEPLOYMENT_SCHEDULE_RECEIPT,
                        schedule_path,
                        workspace=ROOT,
                    ),
                ),
                extra_config={
                    "n_cells": len(deployment_cells),
                    "deployment_estimand": DeploymentEstimand.NATURAL_THRESHOLD.value,
                    "deployment_mode": TWO_PASS_DEPLOYMENT_MODE,
                    "deployment_runtime": deployment_runtime_config(),
                },
            )
            write_manifest_once(deployment_layout.manifest, deployment_manifest)
            deployment_transport = Transport(
                _stage_ledger(
                    deployment_layout, deployment_run_id, Stage.SMOKE
                ),
                deployment_layout.events / "call_attempts.jsonl",
                environ={"OPENAI_API_KEY": "mock-only"},
                urlopen=opener,
            )
            calls_before_rejection = len(opener.calls)
            with self.assertRaisesRegex(
                DeploymentArtifactError, "frozen dataset and build receipt"
            ):
                await execute_deployment_run(
                    run_id=deployment_run_id,
                    task_manifest_path=pass_tasks_path,
                    pass_one_path=pass_artifact_path,
                    threshold_lock_path=deployment_threshold_path,
                    schedule_path=schedule_path,
                    tasks=pass_one_tasks,
                    yes_spend=True,
                    artifacts_root=artifacts,
                    transport=deployment_transport,
                )
            self.assertEqual(len(opener.calls), calls_before_rejection)
            deployment_summary = await execute_deployment_run(
                run_id=deployment_run_id,
                task_manifest_path=pass_tasks_path,
                pass_one_path=pass_artifact_path,
                threshold_lock_path=deployment_threshold_path,
                schedule_path=schedule_path,
                tasks=pass_one_tasks,
                yes_spend=True,
                artifacts_root=artifacts,
                transport=deployment_transport,
                evolving_dataset_path=dataset,
                evolving_build_receipt_path=build_receipt,
            )
            self.assertEqual(deployment_summary.completed_cells, len(deployment_cells))
            self.assertEqual(deployment_summary.failed_cells, 0)
            self.assertTrue(
                deployment_completeness(
                    deployment_layout, deployment_cells
                ).primary_ready
            )
            deployment_outputs = {
                cell.cell_id: read_json(
                    deployment_layout.results
                    / "deployment"
                    / f"{cell.cell_id}.json"
                )
                for cell in deployment_cells
            }
            deployment_rows = extract_deployment_outcomes(
                deployment_cells, deployment_outputs
            )
            for row in deployment_rows:
                output = deployment_outputs[row["cell_id"]]
                expected_probes = 2 if row["method"] == ACTIVE else 0
                self.assertEqual(len(output["probe_records"]), expected_probes)
                self.assertTrue(
                    all(
                        event["signal_frozen_two_pass"]
                        for event in output["intervention_records"]
                    )
                )

            grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
            for row in deployment_rows:
                grouped[(row["method"], row["operator"])].append(row["success"])
            bars = tuple(
                DeploymentBar(
                    observation_class=(
                        "active" if method == ACTIVE else "passive-observational"
                    ),
                    operator=operator,
                    method=method,
                    value=sum(values) / len(values),
                    n_tasks=len(values),
                )
                for (method, operator), values in sorted(grouped.items())
            )
            deployment_figure = write_deployment_grouped_bars(
                bars, root / "figures" / "deployment.svg"
            )
            self.assertTrue(deployment_figure.svg_path.is_file())
            self.assertEqual(
                read_json(deployment_figure.data_path)["statistical_unit"], "task"
            )
            self.assertTrue(opener.calls)
            self.assertTrue(
                all(call.get("model") for call in opener.calls),
                "every mocked dispatch still traversed the real model request builder",
            )


if __name__ == "__main__":
    unittest.main()
