"""Transactional provider budgets for Experiment 12.

Every provider request attempt, including a retry, must reserve its conservative
maximum cost before it is sent.  Reconciliation replaces that reservation with
reported/estimated actual cost.  If a timeout leaves billing unknown,
``reconcile_unknown`` accounts the full reservation as an upper bound rather
than silently treating the attempt as free.

The hard safety ceilings are deliberately compiled into this module:

* OpenAI: $500
* Fireworks: $30

Operational stop levels are run-specific and configurable, but can never exceed
those hard ceilings.  SQLite ``BEGIN IMMEDIATE`` transactions make concurrent
worker reservations serializable across processes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .schemas import CallStatus, TokenUsage


LEDGER_SCHEMA_VERSION = 1
_MICRO_USD = Decimal("1000000")

HARD_CAPS_USD: Mapping[str, Decimal] = MappingProxyType(
    {"openai": Decimal("500"), "fireworks": Decimal("30")}
)

_PROVIDER_ALIASES = {
    "openai": "openai",
    "fireworks": "fireworks",
    "fireworks.ai": "fireworks",
}
_COST_QUALITIES = {"reported", "estimated", "upper_bound"}
_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class BudgetError(RuntimeError):
    """Base class for ledger and budget failures."""


class UnknownProvider(BudgetError):
    pass


class InvalidBudget(BudgetError):
    pass


class LedgerSchemaError(BudgetError):
    pass


class ReservationStateError(BudgetError):
    pass


class DuplicateRequestKey(BudgetError):
    pass


class BudgetLimitExceeded(BudgetError):
    """Raised before a request when its reservation would exceed a limit."""

    def __init__(
        self,
        provider: str,
        limit_kind: str,
        requested_usd: Decimal,
        projected_usd: Decimal,
        cap_usd: Decimal,
    ) -> None:
        self.provider = provider
        self.limit_kind = limit_kind
        self.requested_usd = requested_usd
        self.projected_usd = projected_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"{provider} {limit_kind} budget exceeded: request reserves "
            f"${requested_usd}, projected ${projected_usd}, cap ${cap_usd}"
        )


class BudgetOverrun(BudgetError):
    """Raised after accounting an unexpectedly expensive completed request.

    Accounting is committed before this exception is raised, so the ledger
    remains conservative and future requests stop.
    """

    def __init__(self, result: "ReconciliationResult") -> None:
        self.result = result
        kinds = []
        if result.over_hard_cap:
            kinds.append("hard")
        if result.over_operational_cap:
            kinds.append("operational")
        super().__init__(
            f"{result.reservation.provider} reconciliation exceeded "
            f"{' and '.join(kinds)} budget; cost was recorded"
        )


def canonical_provider(provider: str) -> str:
    if not isinstance(provider, str):
        raise UnknownProvider("provider must be a string")
    key = provider.strip().lower()
    try:
        return _PROVIDER_ALIASES[key]
    except KeyError as exc:
        raise UnknownProvider(f"no hard cap is defined for provider {provider!r}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _to_micro_usd(value: Decimal | str | int | float, name: str) -> int:
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise InvalidBudget(f"{name} must be a decimal dollar amount") from exc
    if not amount.is_finite() or amount < 0:
        raise InvalidBudget(f"{name} must be finite and non-negative")
    return int((amount * _MICRO_USD).to_integral_value(rounding=ROUND_CEILING))


def _from_micro_usd(value: int) -> Decimal:
    return Decimal(value) / _MICRO_USD


def _usage(value: TokenUsage | Mapping[str, Any] | None) -> TokenUsage:
    if value is None:
        return TokenUsage()
    if isinstance(value, TokenUsage):
        return value
    return TokenUsage.from_dict(value)


@dataclass(frozen=True, slots=True)
class ReservationRecord:
    reservation_id: str
    provider: str
    purpose: str
    request_key: str | None
    state: str
    reserved_usd: Decimal
    actual_usd: Decimal | None
    cost_quality: str | None
    request_status: CallStatus | None
    usage: TokenUsage
    provider_request_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ProviderBudget:
    provider: str
    hard_cap_usd: Decimal
    operational_cap_usd: Decimal
    spent_usd: Decimal
    reserved_usd: Decimal
    upper_bound_spend_usd: Decimal
    remaining_hard_usd: Decimal
    remaining_operational_usd: Decimal
    reconciled_requests: int
    active_reservations: int
    released_reservations: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    reservation: ReservationRecord
    budget: ProviderBudget
    over_hard_cap: bool
    over_operational_cap: bool


class BudgetLedger:
    """A process-safe ledger backed by one SQLite file per experiment run."""

    def __init__(
        self,
        path: str | Path,
        *,
        operational_caps_usd: Mapping[str, Decimal | str | int | float | None] | None = None,
        request_scope: str | None = None,
        scope_caps_usd: Mapping[str, Decimal | str | int | float] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.path = Path(path)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        if request_scope is not None and not _SCOPE_RE.fullmatch(request_scope):
            raise ValueError("request_scope must be a short safe run identifier")
        if scope_caps_usd is not None and request_scope is None:
            raise ValueError("scope_caps_usd requires request_scope")
        self.request_scope = request_scope
        self._scope_caps_micro: dict[str, int] = {}
        for provider, value in (scope_caps_usd or {}).items():
            canonical = canonical_provider(provider)
            cap_micro = _to_micro_usd(value, "scope cap")
            hard_micro = _to_micro_usd(HARD_CAPS_USD[canonical], "hard cap")
            if cap_micro > hard_micro:
                raise InvalidBudget(
                    f"{canonical} scope cap cannot exceed hard cap ${HARD_CAPS_USD[canonical]}"
                )
            self._scope_caps_micro[canonical] = cap_micro
        self.path.parent.mkdir(parents=True, exist_ok=True)
        caps: dict[str, int] = {}
        for provider, value in (operational_caps_usd or {}).items():
            canonical = canonical_provider(provider)
            hard_micro = _to_micro_usd(HARD_CAPS_USD[canonical], "hard cap")
            operational_micro = hard_micro if value is None else _to_micro_usd(value, "operational cap")
            if operational_micro > hard_micro:
                raise InvalidBudget(
                    f"{canonical} operational cap cannot exceed hard cap ${HARD_CAPS_USD[canonical]}"
                )
            caps[canonical] = operational_micro
        self._initialize(caps)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self, configured_caps: Mapping[str, int]) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
        finally:
            connection.close()
        with self._transaction() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, LEDGER_SCHEMA_VERSION):
                raise LedgerSchemaError(
                    f"ledger schema {version} is not supported; expected {LEDGER_SCHEMA_VERSION}"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_limits (
                    provider TEXT PRIMARY KEY,
                    hard_cap_micro_usd INTEGER NOT NULL CHECK (hard_cap_micro_usd >= 0),
                    operational_cap_micro_usd INTEGER NOT NULL
                        CHECK (operational_cap_micro_usd >= 0),
                    updated_at TEXT NOT NULL,
                    CHECK (operational_cap_micro_usd <= hard_cap_micro_usd)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL REFERENCES provider_limits(provider),
                    purpose TEXT NOT NULL,
                    request_key TEXT,
                    state TEXT NOT NULL CHECK (state IN ('reserved', 'reconciled', 'released')),
                    reserved_micro_usd INTEGER NOT NULL CHECK (reserved_micro_usd >= 0),
                    actual_micro_usd INTEGER CHECK (actual_micro_usd >= 0),
                    cost_quality TEXT CHECK (
                        cost_quality IS NULL OR
                        cost_quality IN ('reported', 'estimated', 'upper_bound')
                    ),
                    request_status TEXT CHECK (
                        request_status IS NULL OR
                        request_status IN ('succeeded', 'failed', 'unknown')
                    ),
                    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0
                        CHECK (cached_input_tokens >= 0),
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0 CHECK (reasoning_tokens >= 0),
                    provider_total_tokens INTEGER CHECK (provider_total_tokens >= 0),
                    provider_request_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS reservations_request_key
                ON reservations(provider, request_key)
                WHERE request_key IS NOT NULL
                """
            )
            now = _utc_now()
            for provider, hard_usd in HARD_CAPS_USD.items():
                hard_micro = _to_micro_usd(hard_usd, "hard cap")
                row = connection.execute(
                    "SELECT hard_cap_micro_usd FROM provider_limits WHERE provider = ?",
                    (provider,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO provider_limits(
                            provider, hard_cap_micro_usd, operational_cap_micro_usd, updated_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (provider, hard_micro, hard_micro, now),
                    )
                elif int(row["hard_cap_micro_usd"]) != hard_micro:
                    raise LedgerSchemaError(
                        f"stored {provider} hard cap does not match compiled safety cap"
                    )
            for provider, cap_micro in configured_caps.items():
                connection.execute(
                    """
                    UPDATE provider_limits
                    SET operational_cap_micro_usd = ?, updated_at = ?
                    WHERE provider = ?
                    """,
                    (cap_micro, now, provider),
                )
            connection.execute(f"PRAGMA user_version = {LEDGER_SCHEMA_VERSION}")

    def configure_operational_cap(
        self,
        provider: str,
        cap_usd: Decimal | str | int | float | None,
    ) -> ProviderBudget:
        canonical = canonical_provider(provider)
        hard_micro = _to_micro_usd(HARD_CAPS_USD[canonical], "hard cap")
        cap_micro = hard_micro if cap_usd is None else _to_micro_usd(cap_usd, "operational cap")
        if cap_micro > hard_micro:
            raise InvalidBudget(
                f"{canonical} operational cap cannot exceed hard cap ${HARD_CAPS_USD[canonical]}"
            )
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE provider_limits
                SET operational_cap_micro_usd = ?, updated_at = ?
                WHERE provider = ?
                """,
                (cap_micro, _utc_now(), canonical),
            )
            result = self._snapshot_connection(connection, canonical)
        return result

    def reserve(
        self,
        provider: str,
        max_cost_usd: Decimal | str | int | float,
        *,
        purpose: str,
        request_key: str | None = None,
    ) -> ReservationRecord:
        """Atomically reserve worst-case cost before one provider attempt."""

        canonical = canonical_provider(provider)
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        if request_key is not None and (not isinstance(request_key, str) or not request_key.strip()):
            raise ValueError("request_key must be a non-empty string or None")
        if self.request_scope is not None:
            prefix = self.request_scope + "/"
            if request_key is None or not request_key.startswith(prefix):
                raise ValueError("request_key is outside this ledger's run scope")
        amount_micro = _to_micro_usd(max_cost_usd, "max_cost_usd")
        reservation_id = uuid4().hex
        now = _utc_now()
        with self._transaction() as connection:
            if request_key is not None:
                existing = connection.execute(
                    "SELECT * FROM reservations WHERE provider = ? AND request_key = ?",
                    (canonical, request_key),
                ).fetchone()
                if existing is not None:
                    record = self._reservation_from_row(existing)
                    if (
                        record.state == "reserved"
                        and record.purpose == purpose
                        and _to_micro_usd(record.reserved_usd, "reserved_usd") == amount_micro
                    ):
                        return record
                    raise DuplicateRequestKey(
                        f"request_key {request_key!r} already has a {record.state} reservation"
                    )

            budget = self._snapshot_connection(connection, canonical)
            scope_cap_micro = self._scope_caps_micro.get(canonical)
            if scope_cap_micro is not None:
                # Safe because request_scope excludes SQL wildcard characters.
                scoped_row = connection.execute(
                    """
                    SELECT COALESCE(SUM(
                        CASE
                            WHEN state = 'reconciled' THEN actual_micro_usd
                            WHEN state = 'reserved' THEN reserved_micro_usd
                            ELSE 0
                        END
                    ), 0) AS committed_micro
                    FROM reservations
                    WHERE provider = ? AND request_key LIKE ?
                    """,
                    (canonical, f"{self.request_scope}/%"),
                ).fetchone()
                scoped_projected = int(scoped_row["committed_micro"]) + amount_micro
                if scoped_projected > scope_cap_micro:
                    raise BudgetLimitExceeded(
                        canonical,
                        f"run-stage:{self.request_scope}",
                        _from_micro_usd(amount_micro),
                        _from_micro_usd(scoped_projected),
                        _from_micro_usd(scope_cap_micro),
                    )
            projected_micro = (
                _to_micro_usd(budget.spent_usd, "spent_usd")
                + _to_micro_usd(budget.reserved_usd, "reserved_usd")
                + amount_micro
            )
            hard_micro = _to_micro_usd(budget.hard_cap_usd, "hard_cap_usd")
            operational_micro = _to_micro_usd(
                budget.operational_cap_usd, "operational_cap_usd"
            )
            if projected_micro > hard_micro:
                raise BudgetLimitExceeded(
                    canonical,
                    "hard",
                    _from_micro_usd(amount_micro),
                    _from_micro_usd(projected_micro),
                    budget.hard_cap_usd,
                )
            if projected_micro > operational_micro:
                raise BudgetLimitExceeded(
                    canonical,
                    "operational",
                    _from_micro_usd(amount_micro),
                    _from_micro_usd(projected_micro),
                    budget.operational_cap_usd,
                )
            connection.execute(
                """
                INSERT INTO reservations(
                    reservation_id, provider, purpose, request_key, state,
                    reserved_micro_usd, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'reserved', ?, ?, ?)
                """,
                (
                    reservation_id,
                    canonical,
                    purpose.strip(),
                    request_key,
                    amount_micro,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            return self._reservation_from_row(row)

    def reconcile(
        self,
        reservation_id: str,
        actual_cost_usd: Decimal | str | int | float,
        *,
        usage: TokenUsage | Mapping[str, Any] | None = None,
        cost_quality: str = "reported",
        request_status: CallStatus = CallStatus.SUCCEEDED,
        provider_request_id: str | None = None,
        raise_on_overrun: bool = True,
    ) -> ReconciliationResult:
        """Commit actual cost, releasing any unused reservation atomically."""

        if cost_quality not in _COST_QUALITIES:
            raise ValueError(f"cost_quality must be one of {sorted(_COST_QUALITIES)}")
        if not isinstance(request_status, CallStatus):
            raise ValueError("request_status must be CallStatus")
        actual_micro = _to_micro_usd(actual_cost_usd, "actual_cost_usd")
        token_usage = _usage(usage)
        now = _utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise ReservationStateError(f"unknown reservation {reservation_id!r}")
            current = self._reservation_from_row(row)
            if current.state == "reconciled":
                if (
                    _to_micro_usd(current.actual_usd or Decimal(0), "actual_usd") == actual_micro
                    and current.usage == token_usage
                    and current.cost_quality == cost_quality
                    and current.request_status is request_status
                ):
                    budget = self._snapshot_connection(connection, current.provider)
                    result = self._result(current, budget)
                else:
                    raise ReservationStateError(
                        f"reservation {reservation_id!r} was already reconciled differently"
                    )
            elif current.state != "reserved":
                raise ReservationStateError(
                    f"cannot reconcile {current.state} reservation {reservation_id!r}"
                )
            else:
                connection.execute(
                    """
                    UPDATE reservations SET
                        state = 'reconciled', actual_micro_usd = ?, cost_quality = ?,
                        request_status = ?, input_tokens = ?, output_tokens = ?,
                        cached_input_tokens = ?, reasoning_tokens = ?,
                        provider_total_tokens = ?, provider_request_id = ?, updated_at = ?
                    WHERE reservation_id = ?
                    """,
                    (
                        actual_micro,
                        cost_quality,
                        request_status.value,
                        token_usage.input_tokens,
                        token_usage.output_tokens,
                        token_usage.cached_input_tokens,
                        token_usage.reasoning_tokens,
                        token_usage.provider_reported_total_tokens,
                        provider_request_id,
                        now,
                        reservation_id,
                    ),
                )
                updated_row = connection.execute(
                    "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
                ).fetchone()
                current = self._reservation_from_row(updated_row)
                budget = self._snapshot_connection(connection, current.provider)
                result = self._result(current, budget)
        if raise_on_overrun and (result.over_hard_cap or result.over_operational_cap):
            raise BudgetOverrun(result)
        return result

    def reconcile_unknown(
        self,
        reservation_id: str,
        *,
        usage: TokenUsage | Mapping[str, Any] | None = None,
        provider_request_id: str | None = None,
        raise_on_overrun: bool = True,
    ) -> ReconciliationResult:
        """Conservatively account the entire reservation after an ambiguous failure."""

        reservation = self.get_reservation(reservation_id)
        return self.reconcile(
            reservation_id,
            reservation.reserved_usd,
            usage=usage,
            cost_quality="upper_bound",
            request_status=CallStatus.UNKNOWN,
            provider_request_id=provider_request_id,
            raise_on_overrun=raise_on_overrun,
        )

    def release(self, reservation_id: str) -> ReservationRecord:
        """Release a reservation only when the request was definitely not sent."""

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise ReservationStateError(f"unknown reservation {reservation_id!r}")
            record = self._reservation_from_row(row)
            if record.state == "reconciled":
                raise ReservationStateError(
                    f"cannot release reconciled reservation {reservation_id!r}"
                )
            if record.state == "reserved":
                connection.execute(
                    """
                    UPDATE reservations SET state = 'released', updated_at = ?
                    WHERE reservation_id = ?
                    """,
                    (_utc_now(), reservation_id),
                )
                row = connection.execute(
                    "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
                ).fetchone()
                record = self._reservation_from_row(row)
            return record

    def get_reservation(self, reservation_id: str) -> ReservationRecord:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ReservationStateError(f"unknown reservation {reservation_id!r}")
        return self._reservation_from_row(row)

    def list_reservations(
        self,
        *,
        provider: str | None = None,
        state: str | None = None,
    ) -> list[ReservationRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if provider is not None:
            clauses.append("provider = ?")
            parameters.append(canonical_provider(provider))
        if state is not None:
            if state not in {"reserved", "reconciled", "released"}:
                raise ValueError("invalid reservation state")
            clauses.append("state = ?")
            parameters.append(state)
        query = "SELECT * FROM reservations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, reservation_id"
        connection = self._connect()
        try:
            rows = connection.execute(query, parameters).fetchall()
        finally:
            connection.close()
        return [self._reservation_from_row(row) for row in rows]

    def snapshot(self, provider: str | None = None) -> ProviderBudget | dict[str, ProviderBudget]:
        connection = self._connect()
        try:
            if provider is not None:
                return self._snapshot_connection(connection, canonical_provider(provider))
            return {
                name: self._snapshot_connection(connection, name)
                for name in HARD_CAPS_USD
            }
        finally:
            connection.close()

    @staticmethod
    def _reservation_from_row(row: sqlite3.Row) -> ReservationRecord:
        return ReservationRecord(
            reservation_id=row["reservation_id"],
            provider=row["provider"],
            purpose=row["purpose"],
            request_key=row["request_key"],
            state=row["state"],
            reserved_usd=_from_micro_usd(int(row["reserved_micro_usd"])),
            actual_usd=(
                None
                if row["actual_micro_usd"] is None
                else _from_micro_usd(int(row["actual_micro_usd"]))
            ),
            cost_quality=row["cost_quality"],
            request_status=(
                None if row["request_status"] is None else CallStatus(row["request_status"])
            ),
            usage=TokenUsage(
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                cached_input_tokens=int(row["cached_input_tokens"]),
                reasoning_tokens=int(row["reasoning_tokens"]),
                provider_reported_total_tokens=(
                    None
                    if row["provider_total_tokens"] is None
                    else int(row["provider_total_tokens"])
                ),
            ),
            provider_request_id=row["provider_request_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _snapshot_connection(connection: sqlite3.Connection, provider: str) -> ProviderBudget:
        row = connection.execute(
            """
            SELECT
                limits.provider,
                limits.hard_cap_micro_usd,
                limits.operational_cap_micro_usd,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled'
                    THEN r.actual_micro_usd ELSE 0 END), 0) AS spent_micro,
                COALESCE(SUM(CASE WHEN r.state = 'reserved'
                    THEN r.reserved_micro_usd ELSE 0 END), 0) AS reserved_micro,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled' AND r.cost_quality = 'upper_bound'
                    THEN r.actual_micro_usd ELSE 0 END), 0) AS upper_bound_micro,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled' THEN 1 ELSE 0 END), 0)
                    AS reconciled_requests,
                COALESCE(SUM(CASE WHEN r.state = 'reserved' THEN 1 ELSE 0 END), 0)
                    AS active_reservations,
                COALESCE(SUM(CASE WHEN r.state = 'released' THEN 1 ELSE 0 END), 0)
                    AS released_reservations,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled' THEN r.input_tokens ELSE 0 END), 0)
                    AS input_tokens,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled' THEN r.output_tokens ELSE 0 END), 0)
                    AS output_tokens,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled'
                    THEN r.cached_input_tokens ELSE 0 END), 0) AS cached_input_tokens,
                COALESCE(SUM(CASE WHEN r.state = 'reconciled'
                    THEN r.reasoning_tokens ELSE 0 END), 0) AS reasoning_tokens
            FROM provider_limits AS limits
            LEFT JOIN reservations AS r ON r.provider = limits.provider
            WHERE limits.provider = ?
            GROUP BY limits.provider
            """,
            (provider,),
        ).fetchone()
        if row is None:
            raise UnknownProvider(provider)
        hard = int(row["hard_cap_micro_usd"])
        operational = int(row["operational_cap_micro_usd"])
        spent = int(row["spent_micro"])
        reserved = int(row["reserved_micro"])
        committed = spent + reserved
        stop_reason = None
        if committed >= hard:
            stop_reason = "hard"
        elif committed >= operational:
            stop_reason = "operational"
        return ProviderBudget(
            provider=provider,
            hard_cap_usd=_from_micro_usd(hard),
            operational_cap_usd=_from_micro_usd(operational),
            spent_usd=_from_micro_usd(spent),
            reserved_usd=_from_micro_usd(reserved),
            upper_bound_spend_usd=_from_micro_usd(int(row["upper_bound_micro"])),
            remaining_hard_usd=_from_micro_usd(max(0, hard - committed)),
            remaining_operational_usd=_from_micro_usd(max(0, operational - committed)),
            reconciled_requests=int(row["reconciled_requests"]),
            active_reservations=int(row["active_reservations"]),
            released_reservations=int(row["released_reservations"]),
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            cached_input_tokens=int(row["cached_input_tokens"]),
            reasoning_tokens=int(row["reasoning_tokens"]),
            stop_reason=stop_reason,
        )

    @staticmethod
    def _result(reservation: ReservationRecord, budget: ProviderBudget) -> ReconciliationResult:
        committed = budget.spent_usd + budget.reserved_usd
        return ReconciliationResult(
            reservation=reservation,
            budget=budget,
            over_hard_cap=committed > budget.hard_cap_usd,
            over_operational_cap=committed > budget.operational_cap_usd,
        )


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "HARD_CAPS_USD",
    "BudgetError",
    "UnknownProvider",
    "InvalidBudget",
    "LedgerSchemaError",
    "ReservationStateError",
    "DuplicateRequestKey",
    "BudgetLimitExceeded",
    "BudgetOverrun",
    "ReservationRecord",
    "ProviderBudget",
    "ReconciliationResult",
    "BudgetLedger",
    "canonical_provider",
]
