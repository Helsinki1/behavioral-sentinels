from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from experiments12.core.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from experiments12.core.budget import BudgetLedger
from experiments12.core.schemas import CallAttemptRecord, CallStatus, TokenUsage, record_to_dict
from experiments12.harness12 import ARM_TO_PROBE, HARNESS_VERSION
from experiments12.manifest12 import RunLayout, build_manifest, write_manifest_once
from experiments12.pairing12 import TaskRef, make_pair_manifest
from experiments12.spec12 import Stage
from experiments12.validate12 import validate_run, write_validation_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK_SHA = "b" * 64
SOURCE_SHA = "a" * 64


class SyntheticRun:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_validate_")
        self.base = Path(self.temp.name)
        self.layout = RunLayout.for_run(self.base, "synthetic-run")
        self.layout.create()
        self.cells = make_pair_manifest(
            tasks=(TaskRef("evolving_intent_gsm8k", "task-1::t7", TASK_SHA),),
            models=("gpt-5.6-luna",),
            arms=("clean", "active_name_copy"),
            operators=("none",),
            replicates=1,
            randomization_seed=17,
        )
        atomic_write_jsonl(self.layout.pairs, [cell.as_dict() for cell in self.cells])
        manifest = build_manifest(
            run_id="synthetic-run",
            stage=Stage.CONFIRMATORY,
            repository_root=REPOSITORY_ROOT,
            pair_manifest_sha256=sha256_file(self.layout.pairs),
            models=("gpt-5.6-luna",),
            arms=("clean", "active_name_copy"),
            operators=("none",),
            randomization_seed=17,
            benchmark_receipts=(),
            extra_config={"n_cells": len(self.cells)},
        )
        write_manifest_once(self.layout.manifest, manifest)
        self.manifest_sha256 = sha256_file(self.layout.manifest)
        self.ledger = BudgetLedger(self.layout.ledger)
        self.call_attempts: list[dict[str, object]] = []
        self._call_counter = 0
        self.trajectory_paths: dict[str, Path] = {}
        self.event_paths: dict[str, Path] = {}
        for cell in self.cells:
            self._write_trajectory(cell)
        atomic_write_jsonl(self.layout.events / "call_attempts.jsonl", self.call_attempts)
        clean = next(
            read_json(path)
            for cell_id, path in self.trajectory_paths.items()
            if next(cell for cell in self.cells if cell.cell_id == cell_id).arm == "clean"
        )
        self.shadow_path = self.layout.shadow / "clean.shadow.json"
        atomic_write_json(
            self.shadow_path,
            {
                "schema_version": 1,
                "shadow_version": 1,
                "source_trajectory_sha256": clean["transcript_sha256"],
                "model": clean["model"],
                "domain": clean["domain"],
                "task_id": clean["task_id"],
                "condition": clean["condition"],
                "records": [
                    {
                        "method": "trace_rules",
                        "checkpoint_turn": 1,
                        "score": 0.2,
                        "source_trajectory_sha256": clean["transcript_sha256"],
                    }
                ],
                "monitor_methods": ["trace_rules"],
                "complete": True,
            },
        )

    def close(self) -> None:
        self.temp.cleanup()

    def cell(self, arm: str):
        return next(cell for cell in self.cells if cell.arm == arm)

    def _call(self, cell_id: str, purpose: str, index: int) -> dict[str, object]:
        self._call_counter += 1
        reservation = self.ledger.reserve(
            "openai",
            "0.01",
            purpose=purpose,
            request_key=f"synthetic-run/{cell_id}/{purpose}-{index}/attempt-1",
        )
        usage = TokenUsage(input_tokens=20 + index, output_tokens=3)
        self.ledger.reconcile(
            reservation.reservation_id,
            "0.001",
            usage=usage,
            request_status=CallStatus.SUCCEEDED,
        )
        event_id = f"event-{self._call_counter:04d}"
        attempt = CallAttemptRecord(
            event_id=event_id,
            reservation_id=reservation.reservation_id,
            provider="openai",
            model="gpt-5.6-luna",
            purpose=purpose,
            attempt_number=1,
            status=CallStatus.SUCCEEDED,
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            usage=usage,
            estimated_cost_usd=Decimal("0.001"),
            elapsed_ms=10,
        )
        self.call_attempts.append(record_to_dict(attempt))
        return {
            "call_event_ids": [event_id],
            "resolved_model_id": "gpt-5.6-luna",
            "response_id": f"response-{self._call_counter}",
            "request_id": f"request-{self._call_counter}",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            },
            "accounted_cost_usd": "0.001",
            "elapsed_ms": 10,
        }

    def _write_trajectory(self, cell) -> None:
        checkpoints = [1, 2]
        task_manifest = {
            "domain": cell.pair_key.domain,
            "task_id": "task-1",
            "condition": "t7",
            "num_turns": 3,
            "source_sha256": SOURCE_SHA,
            "task_sha256": TASK_SHA,
            "evaluation_label_sha256": "c" * 64,
            "public_metadata": {},
        }
        design = {
            "run_id": "synthetic-run",
            "cell_id": cell.cell_id,
            "model": cell.pair_key.model,
            "task": task_manifest,
            "arm": cell.arm,
            "config": {
                "checkpoint_every": 1,
                "task_max_output_tokens": 100,
                "probe_max_output_tokens": 20,
                "temperature": None,
            },
            "checkpoint_turns": checkpoints,
            "harness_version": HARNESS_VERSION,
        }
        start = {"event": "start", "design_sha256": sha256_json(design), **design}
        events: list[dict[str, object]] = [start]
        messages: list[dict[str, str]] = []
        task_records: list[dict[str, object]] = []
        probe_records: list[dict[str, object]] = []
        active_variant = ARM_TO_PROBE.get(cell.arm)
        assistants: list[str] = []
        for turn in range(1, 4):
            user = {"role": "user", "content": f"task user {turn}"}
            assistant = {"role": "assistant", "content": f"task assistant {turn}"}
            task_record = {
                "event": "task_turn",
                "task_turn": turn,
                "user_message": user,
                "assistant_message": assistant,
                "call": self._call(cell.cell_id, "agent_turn", turn),
            }
            events.append(task_record)
            task_records.append(task_record)
            messages.extend((user, assistant))
            assistants.append(assistant["content"])
            if active_variant is not None and turn in checkpoints:
                probe_user = {"role": "user", "content": f"probe {turn}"}
                probe_assistant = {"role": "assistant", "content": "PROBE: AAAAAAAA"}
                probe_record = {
                    "event": "active_probe",
                    "after_task_turn": turn,
                    "checkpoint_index": checkpoints.index(turn) + 1,
                    "variant": active_variant,
                    "user_message": probe_user,
                    "assistant_message": probe_assistant,
                    "grade": {
                        "passed": True,
                        "value_correct": True,
                        "exact_format": True,
                        "error": None,
                        "expected_sha256": "d" * 64,
                    },
                    "call": self._call(cell.cell_id, "active_probe", checkpoints.index(turn) + 1),
                }
                messages.extend((probe_user, probe_assistant))
                probe_record["source_prefix_sha256"] = sha256_json(messages)
                events.append(probe_record)
                probe_records.append(probe_record)
        transcript_hash = sha256_json(messages)
        complete = {
            "event": "complete",
            "task_turns": 3,
            "transcript_sha256": transcript_hash,
            "prediction": "6",
            "success": True,
        }
        events.append(complete)
        output = {
            "schema_version": 1,
            "harness_version": HARNESS_VERSION,
            "run_id": "synthetic-run",
            "cell_id": cell.cell_id,
            "design_sha256": start["design_sha256"],
            "model": cell.pair_key.model,
            "domain": cell.pair_key.domain,
            "task_id": "task-1",
            "condition": "t7",
            "task_sha256": TASK_SHA,
            "arm": cell.arm,
            "active_probe_variant": active_variant,
            "checkpoint_turns": checkpoints,
            "messages": messages,
            "task_assistant_messages": assistants,
            "task_records": task_records,
            "probe_records": probe_records,
            "evaluation": {
                "prediction": "6",
                "evaluation_label_sha256": "c" * 64,
                "success": True,
            },
            "transcript_sha256": transcript_hash,
            "accounting": {"by_category": {}, "resolved_model_ids": [cell.pair_key.model]},
            "complete": True,
        }
        output_path = self.layout.trajectories / f"{cell.cell_id}.json"
        event_path = self.layout.events / f"{cell.cell_id}.events.jsonl"
        atomic_write_json(output_path, output)
        atomic_write_jsonl(event_path, events)
        self.trajectory_paths[cell.cell_id] = output_path
        self.event_paths[cell.cell_id] = event_path


def codes(report) -> set[str]:
    return {issue.code for issue in report.errors}


class ValidateRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = SyntheticRun()

    def tearDown(self) -> None:
        self.run.close()

    def validate(self):
        return validate_run(
            self.run.layout,
            repository_root=REPOSITORY_ROOT,
            expected_manifest_sha256=self.run.manifest_sha256,
        )

    def test_complete_synthetic_run_is_primary_ready_and_report_is_json(self) -> None:
        before = {
            path: sha256_file(path)
            for path in self.run.trajectory_paths.values()
        }
        report = self.validate()
        self.assertTrue(report.primary_ready, report.as_dict())
        self.assertEqual(report.expected_cells, 2)
        self.assertEqual(report.valid_trajectories, 2)
        self.assertEqual(report.shadow_outputs, 1)
        self.assertEqual(report.errors, ())
        self.assertEqual(report.warnings, ())
        self.assertEqual(
            before,
            {path: sha256_file(path) for path in self.run.trajectory_paths.values()},
        )
        report_path = self.run.layout.results / "validation.json"
        digest = write_validation_report(report_path, report)
        self.assertEqual(digest, sha256_file(report_path))
        self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["primary_ready"])

    def test_manifest_and_pair_hash_mutations_fail_closed(self) -> None:
        manifest = read_json(self.run.layout.manifest)
        manifest["stage"] = "calibration"
        atomic_write_json(self.run.layout.manifest, manifest)
        report = self.validate()
        self.assertIn("manifest.hash_mismatch", codes(report))
        self.assertFalse(report.primary_ready)

        # Restore the manifest, then mutate the exact pair bytes it pins.
        atomic_write_json(self.run.layout.manifest, {**manifest, "stage": "confirmatory"})
        self.run.manifest_sha256 = sha256_file(self.run.layout.manifest)
        rows = read_jsonl(self.run.layout.pairs)
        rows[0]["seed"] += 1
        atomic_write_jsonl(self.run.layout.pairs, rows)
        report = self.validate()
        self.assertIn("pairs.hash_mismatch", codes(report))
        self.assertFalse(report.primary_ready)

    def test_missing_duplicate_and_foreign_outputs_are_not_intersected_away(self) -> None:
        clean = self.run.cell("clean")
        clean_path = self.run.trajectory_paths[clean.cell_id]
        clean_path.unlink()
        report = self.validate()
        self.assertIn("trajectory.missing_cell", codes(report))
        self.assertFalse(report.primary_ready)

        # Recreate from the append-only event material, then duplicate the cell.
        events = read_jsonl(self.run.event_paths[clean.cell_id])
        original = {
            "cell_id": clean.cell_id,
            "complete": True,
        }
        # A duplicate need not otherwise be valid: cardinality fails first.
        atomic_write_json(clean_path, original)
        atomic_write_json(self.run.layout.trajectories / "duplicate.json", original)
        foreign = dict(read_json(self.run.trajectory_paths[self.run.cell("active_name_copy").cell_id]))
        foreign["cell_id"] = "f" * 24
        atomic_write_json(self.run.layout.trajectories / "foreign.json", foreign)
        report = self.validate()
        self.assertIn("trajectory.duplicate_cell", codes(report))
        self.assertIn("trajectory.foreign_cell", codes(report))
        self.assertFalse(report.primary_ready)
        self.assertTrue(events)  # The fixture remains entirely offline/synthetic.

    def test_trajectory_identity_design_transcript_and_gold_are_checked(self) -> None:
        active = self.run.cell("active_name_copy")
        path = self.run.trajectory_paths[active.cell_id]
        output = read_json(path)
        output["model"] = "foreign-model"
        output["messages"][0]["content"] = "tampered"
        output["task_records"][0]["gold_label"] = "secret"
        atomic_write_json(path, output)
        event_rows = read_jsonl(self.run.event_paths[active.cell_id])
        event_rows[0]["config"]["checkpoint_every"] = 2
        atomic_write_jsonl(self.run.event_paths[active.cell_id], event_rows)
        report = self.validate()
        self.assertTrue(
            {
                "trajectory.declaration_mismatch",
                "trajectory.transcript_hash_mismatch",
                "trajectory.gold_in_task_records",
                "trajectory.design_hash_mismatch",
            }.issubset(codes(report))
        )
        self.assertFalse(report.primary_ready)

    def test_clean_and_active_probe_boundaries_are_exact(self) -> None:
        clean = self.run.cell("clean")
        clean_path = self.run.trajectory_paths[clean.cell_id]
        clean_output = read_json(clean_path)
        clean_output["active_probe_variant"] = "name_copy"
        clean_output["probe_records"] = [
            {
                "event": "active_probe",
                "after_task_turn": 1,
                "checkpoint_index": 1,
                "variant": "name_copy",
            }
        ]
        atomic_write_json(clean_path, clean_output)
        active = self.run.cell("active_name_copy")
        active_path = self.run.trajectory_paths[active.cell_id]
        active_output = read_json(active_path)
        active_output["probe_records"] = active_output["probe_records"][:1]
        atomic_write_json(active_path, active_output)
        report = self.validate()
        self.assertTrue(
            {
                "trajectory.clean_probe_variant",
                "trajectory.clean_has_probe",
                "trajectory.active_checkpoints_mismatch",
            }.issubset(codes(report))
        )
        self.assertFalse(report.primary_ready)

    def test_active_probe_prefix_hash_is_required_and_recomputed(self) -> None:
        active = self.run.cell("active_name_copy")
        path = self.run.trajectory_paths[active.cell_id]
        output = read_json(path)
        output["probe_records"][0]["source_prefix_sha256"] = "f" * 64
        atomic_write_json(path, output)
        event_rows = read_jsonl(self.run.event_paths[active.cell_id])
        probe = next(row for row in event_rows if row.get("event") == "active_probe")
        probe["source_prefix_sha256"] = "f" * 64
        atomic_write_jsonl(self.run.event_paths[active.cell_id], event_rows)
        report = self.validate()
        self.assertIn("trajectory.probe_prefix_hash_mismatch", codes(report))
        self.assertFalse(report.primary_ready)

    def test_shadow_must_content_address_an_unchanged_clean_source(self) -> None:
        shadow = read_json(self.run.shadow_path)
        shadow["source_trajectory_sha256"] = "e" * 64
        shadow["records"][0]["source_trajectory_sha256"] = "e" * 64
        shadow["messages"] = [{"role": "user", "content": "must not carry"}]
        atomic_write_json(self.run.shadow_path, shadow)
        clean_path = self.run.trajectory_paths[self.run.cell("clean").cell_id]
        before = sha256_file(clean_path)
        report = self.validate()
        self.assertIn("shadow.source_missing", codes(report))
        self.assertEqual(before, sha256_file(clean_path))
        self.assertFalse(report.primary_ready)

    def test_call_events_must_exist_once_and_resolve_to_global_ledger(self) -> None:
        call_path = self.run.layout.events / "call_attempts.jsonl"
        rows = read_jsonl(call_path)
        missing_event_id = rows[0]["event_id"]
        rows = rows[1:]
        rows[0]["reservation_id"] = "not-in-ledger"
        ledger_gap_id = rows[0]["event_id"]
        atomic_write_jsonl(call_path, rows)
        report = self.validate()
        self.assertIn("call_event.missing", codes(report))
        self.assertIn("call_event.ledger_missing", codes(report))
        subjects = {issue.subject for issue in report.errors}
        self.assertIn(missing_event_id, subjects)
        self.assertIn(ledger_gap_id, subjects)
        self.assertFalse(report.primary_ready)


if __name__ == "__main__":
    unittest.main()
