"""Focused stdlib-only tests for Experiment 12 core infrastructure."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
import tempfile
import threading
import unittest

from experiments12.core.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_json,
    verify_sha256,
)
from experiments12.core.budget import (
    BudgetLedger,
    BudgetLimitExceeded,
    BudgetOverrun,
    DuplicateRequestKey,
    HARD_CAPS_USD,
    InvalidBudget,
    ReservationStateError,
)
from experiments12.core.schemas import (
    CallAttemptRecord,
    CallStatus,
    MonitorRecord,
    PairKey,
    TokenUsage,
    TrajectoryRecord,
    TrajectoryStatus,
    TurnRecord,
    record_to_dict,
)


class SchemaTests(unittest.TestCase):
    def test_records_are_immutable_and_round_trip(self) -> None:
        pair = PairKey(
            model="test-model",
            domain="coding",
            task_id="task-7",
            replicate_id=2,
            task_sha256="a" * 64,
        )
        turn = TurnRecord(
            turn=1,
            user_message="user exact text",
            assistant_message="assistant exact text",
            call_event_ids=("call-1",),
            hallucination=False,
        )
        trajectory = TrajectoryRecord(
            run_id="run-1",
            experiment_id="experiment-12",
            pair_key=pair,
            arm="control",
            system_message="system exact text",
            turns=(turn,),
            status=TrajectoryStatus.COMPLETE,
            started_at="2026-08-26T10:00:00Z",
            finished_at="2026-08-26T10:01:00Z",
        )
        restored = TrajectoryRecord.from_dict(record_to_dict(trajectory))
        self.assertEqual(restored, trajectory)
        self.assertEqual(pair.stable_id, "test-model/coding/task-7/r2")
        with self.assertRaises(FrozenInstanceError):
            pair.task_id = "changed"  # type: ignore[misc]

    def test_call_usage_and_cost_round_trip(self) -> None:
        record = CallAttemptRecord(
            event_id="event-1",
            reservation_id="reservation-1",
            provider="openai",
            model="test-model",
            purpose="agent_turn",
            attempt_number=2,
            status=CallStatus.SUCCEEDED,
            started_at="2026-08-26T10:00:00Z",
            finished_at="2026-08-26T10:00:01Z",
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=20,
                cached_input_tokens=50,
                reasoning_tokens=5,
                provider_reported_total_tokens=120,
            ),
            estimated_cost_usd=Decimal("0.012345"),
            elapsed_ms=1000,
        )
        restored = CallAttemptRecord.from_dict(record_to_dict(record))
        self.assertEqual(restored, record)
        self.assertEqual(record.usage.total_tokens, 120)

    def test_monitor_schema_encodes_action_time_and_no_outcome(self) -> None:
        record = MonitorRecord(
            monitor_event_id="monitor-1",
            source_trajectory_sha256="b" * 64,
            monitor_spec_sha256="c" * 64,
            checkpoint_turn=6,
            observable_after_turn=6,
            actionable_before_turn=7,
            fired=True,
        )
        plain = record_to_dict(record)
        self.assertNotIn("hallucination", plain)
        self.assertNotIn("outcome", plain)
        self.assertEqual(MonitorRecord.from_dict(plain), record)
        with self.assertRaises(ValueError):
            MonitorRecord(
                monitor_event_id="bad",
                source_trajectory_sha256="b" * 64,
                monitor_spec_sha256="c" * 64,
                checkpoint_turn=6,
                observable_after_turn=6,
                actionable_before_turn=6,
                fired=False,
            )

    def test_validation_rejects_bad_hash_and_negative_usage(self) -> None:
        with self.assertRaises(ValueError):
            PairKey("m", "d", "t", task_sha256="not-a-hash")
        with self.assertRaises(ValueError):
            TokenUsage(input_tokens=-1)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_artifacts_")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_canonical_hash_ignores_mapping_insertion_order(self) -> None:
        left = {"b": 2, "a": [1, 3]}
        right = {"a": [1, 3], "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(sha256_json(left), sha256_json(right))

    def test_atomic_json_returns_exact_digest_and_preserves_old_on_error(self) -> None:
        path = self.root / "nested" / "artifact.json"
        digest = atomic_write_json(path, {"version": 1, "ok": True})
        self.assertEqual(digest, sha256_file(path))
        self.assertTrue(verify_sha256(path, digest))
        before = path.read_bytes()
        with self.assertRaises((TypeError, ValueError)):
            atomic_write_json(path, {"bad": object()})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(read_json(path), {"ok": True, "version": 1})

    def test_atomic_jsonl_and_locked_appends(self) -> None:
        path = self.root / "events.jsonl"
        atomic_write_jsonl(path, ({"event": i} for i in range(3)))

        def add(i: int) -> None:
            append_jsonl(path, {"event": i})

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(add, range(3, 103)))
        records = read_jsonl(path)
        self.assertEqual(len(records), 103)
        self.assertEqual({r["event"] for r in records}, set(range(103)))

    def test_torn_jsonl_is_rejected(self) -> None:
        path = self.root / "torn.jsonl"
        path.write_bytes(b'{"ok":true}\n{"torn":')
        with self.assertRaisesRegex(ValueError, "torn JSONL"):
            read_jsonl(path)

    def test_append_digest_is_for_exact_line(self) -> None:
        path = self.root / "single.jsonl"
        record = {"kind": "call", "n": 1}
        digest = append_jsonl(path, record)
        self.assertEqual(digest, sha256_bytes(canonical_json_bytes(record) + b"\n"))


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="experiment12_budget_")
        self.path = Path(self.temp.name) / "ledger.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_compiled_hard_caps_and_configurable_operational_stops(self) -> None:
        ledger = BudgetLedger(
            self.path,
            operational_caps_usd={"openai": "2.50", "fireworks": "3"},
        )
        openai = ledger.snapshot("openai")
        fireworks = ledger.snapshot("fireworks")
        self.assertEqual(HARD_CAPS_USD["openai"], Decimal("500"))
        self.assertEqual(HARD_CAPS_USD["fireworks"], Decimal("30"))
        self.assertEqual(openai.hard_cap_usd, Decimal("500"))
        self.assertEqual(openai.operational_cap_usd, Decimal("2.5"))
        self.assertEqual(fireworks.hard_cap_usd, Decimal("30"))
        self.assertEqual(fireworks.operational_cap_usd, Decimal("3"))
        with self.assertRaises(InvalidBudget):
            ledger.configure_operational_cap("openai", "500.000001")

    def test_reserve_reconcile_and_usage_rollup(self) -> None:
        ledger = BudgetLedger(self.path, operational_caps_usd={"openai": "2"})
        reservation = ledger.reserve(
            "openai",
            "0.50",
            purpose="agent_turn",
            request_key="task/arm/turn/attempt-1",
        )
        during = ledger.snapshot("openai")
        self.assertEqual(during.spent_usd, Decimal("0"))
        self.assertEqual(during.reserved_usd, Decimal("0.5"))
        usage = TokenUsage(input_tokens=100, output_tokens=25, cached_input_tokens=40)
        result = ledger.reconcile(
            reservation.reservation_id,
            "0.20",
            usage=usage,
            provider_request_id="provider-request-1",
        )
        self.assertEqual(result.budget.spent_usd, Decimal("0.2"))
        self.assertEqual(result.budget.reserved_usd, Decimal("0"))
        self.assertEqual(result.budget.input_tokens, 100)
        self.assertEqual(result.budget.output_tokens, 25)
        self.assertEqual(result.reservation.usage, usage)
        # Identical reconciliation is idempotent; it cannot double-count cost.
        again = ledger.reconcile(
            reservation.reservation_id,
            "0.20",
            usage=usage,
            provider_request_id="provider-request-1",
        )
        self.assertEqual(again.budget.spent_usd, Decimal("0.2"))
        with self.assertRaises(DuplicateRequestKey):
            ledger.reserve(
                "openai",
                "0.50",
                purpose="agent_turn",
                request_key="task/arm/turn/attempt-1",
            )

    def test_limit_check_is_transactional_across_workers(self) -> None:
        first = BudgetLedger(self.path, operational_caps_usd={"openai": "0.75"})
        second = BudgetLedger(self.path)
        barrier = threading.Barrier(2)

        def reserve(ledger: BudgetLedger, key: str) -> str:
            barrier.wait()
            try:
                ledger.reserve("openai", "0.50", purpose="retry", request_key=key)
                return "reserved"
            except BudgetLimitExceeded as exc:
                self.assertEqual(exc.limit_kind, "operational")
                return "blocked"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(
                pool.map(
                    lambda args: reserve(*args),
                    ((first, "worker-1"), (second, "worker-2")),
                )
            )
        self.assertEqual(sorted(outcomes), ["blocked", "reserved"])
        self.assertEqual(first.snapshot("openai").reserved_usd, Decimal("0.5"))

    def test_unknown_call_keeps_upper_bound_cost(self) -> None:
        ledger = BudgetLedger(self.path, operational_caps_usd={"fireworks": "1"})
        reservation = ledger.reserve(
            "fireworks", "0.40", purpose="quiz", request_key="unknown-attempt"
        )
        result = ledger.reconcile_unknown(reservation.reservation_id)
        self.assertEqual(result.reservation.request_status, CallStatus.UNKNOWN)
        self.assertEqual(result.reservation.cost_quality, "upper_bound")
        self.assertEqual(result.budget.spent_usd, Decimal("0.4"))
        self.assertEqual(result.budget.upper_bound_spend_usd, Decimal("0.4"))

    def test_reconciliation_overrun_is_recorded_before_raising(self) -> None:
        ledger = BudgetLedger(self.path, operational_caps_usd={"openai": "1"})
        reservation = ledger.reserve("openai", "0.50", purpose="agent_turn")
        with self.assertRaises(BudgetOverrun) as caught:
            ledger.reconcile(reservation.reservation_id, "1.25")
        self.assertTrue(caught.exception.result.over_operational_cap)
        self.assertFalse(caught.exception.result.over_hard_cap)
        snapshot = ledger.snapshot("openai")
        self.assertEqual(snapshot.spent_usd, Decimal("1.25"))
        self.assertEqual(snapshot.stop_reason, "operational")
        with self.assertRaises(BudgetLimitExceeded):
            ledger.reserve("openai", "0.000001", purpose="retry")

    def test_release_is_only_for_unsent_requests(self) -> None:
        ledger = BudgetLedger(self.path)
        reservation = ledger.reserve("fireworks", "0.10", purpose="agent_turn")
        released = ledger.release(reservation.reservation_id)
        self.assertEqual(released.state, "released")
        self.assertEqual(ledger.snapshot("fireworks").reserved_usd, Decimal("0"))
        reconciled = ledger.reserve("fireworks", "0.10", purpose="agent_turn")
        ledger.reconcile(reconciled.reservation_id, "0.05")
        with self.assertRaises(ReservationStateError):
            ledger.release(reconciled.reservation_id)

    def test_provider_budgets_are_independent(self) -> None:
        ledger = BudgetLedger(
            self.path,
            operational_caps_usd={"openai": "0.20", "fireworks": "0.20"},
        )
        ledger.reserve("openai", "0.20", purpose="agent_turn")
        ledger.reserve("fireworks", "0.20", purpose="agent_turn")
        self.assertEqual(ledger.snapshot("openai").reserved_usd, Decimal("0.2"))
        self.assertEqual(ledger.snapshot("fireworks").reserved_usd, Decimal("0.2"))

    def test_run_scope_is_transactional_and_namespaced(self) -> None:
        ledger = BudgetLedger(
            self.path,
            request_scope="smoke-a",
            scope_caps_usd={"openai": "0.10", "fireworks": "0.05"},
        )
        with self.assertRaises(ValueError):
            ledger.reserve(
                "openai", "0.01", purpose="agent_turn", request_key="another/call"
            )
        first = ledger.reserve(
            "openai", "0.06", purpose="agent_turn", request_key="smoke-a/call-1"
        )
        with self.assertRaises(BudgetLimitExceeded) as caught:
            ledger.reserve(
                "openai", "0.05", purpose="agent_turn", request_key="smoke-a/call-2"
            )
        self.assertEqual(caught.exception.limit_kind, "run-stage:smoke-a")
        ledger.reconcile(first.reservation_id, "0.02")
        second = ledger.reserve(
            "openai", "0.05", purpose="agent_turn", request_key="smoke-a/call-2"
        )
        self.assertEqual(second.state, "reserved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
