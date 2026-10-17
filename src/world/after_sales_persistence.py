"""Typed persistence primitives for World-authoritative after-sales state.

This module is deliberately independent of benchmark scenarios and of either
World storage backend.  It defines the rows, owner-aware read policy, append
rules, idempotency journal, and replay format that both the in-memory and
SQLite World implementations must use.

The tables store immutable records.  Stateful objects such as disputes and
exchanges append a new version instead of replacing history.  Commerce effects
on orders, inventory, and the ledger are represented in the same commit, but
are applied by the enclosing World transaction.  A scenario is never allowed
to instantiate or mutate these tables directly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

from protocol.after_sales import (
    AfterSalesOrderBinding,
    AfterSalesRecord,
    DisputeCase,
    DisputeEvidence,
    ExchangeCase,
    RefundCase,
    RefundDecision,
    ReturnAuthorization,
    ReturnReceipt,
    ReturnRequest,
    Ruling,
    after_sales_binding_to_dict,
    after_sales_binding_from_json,
    after_sales_record_from_json,
    after_sales_record_to_dict,
    validate_after_sales_order_binding,
    validate_after_sales_record,
)
from world.after_sales_core import (
    AfterSalesPolicyRevision,
    CoreRecord,
    DisputeResponseRecord,
    LedgerReconciliationRequest,
    LedgerReconciliationResult,
    LedgerReconciliationSource,
    PaidCancellationRecord,
    core_after_sales_record_to_dict,
    core_after_sales_record_from_dict,
    after_sales_policy_from_dict,
    after_sales_policy_to_dict,
    ledger_reconciliation_source_from_dict,
    ledger_reconciliation_source_to_dict,
    validate_after_sales_policy,
    validate_core_after_sales_record,
    validate_ledger_reconciliation_source,
)
from world.errors import IdempotencyConflict


AFTER_SALES_OPERATION_SCHEMA = "cwe.world-after-sales-operation.v1"
AFTER_SALES_COMMIT_SCHEMA = "cwe.world-after-sales-commit.v1"

AfterSalesTableName: TypeAlias = Literal[
    "after_sales_policies",
    "after_sales_bindings",
    "paid_cancellations",
    "return_requests",
    "return_authorizations",
    "return_receipts",
    "refund_cases",
    "refund_decisions",
    "exchange_cases",
    "dispute_cases",
    "dispute_evidence",
    "dispute_responses",
    "after_sales_rulings",
    "ledger_reconciliation_sources",
    "ledger_reconciliation_requests",
    "ledger_reconciliation_results",
]

PersistedAfterSalesRow: TypeAlias = (
    AfterSalesPolicyRevision
    | AfterSalesOrderBinding
    | AfterSalesRecord
    | CoreRecord
    | LedgerReconciliationSource
)

CommerceEffectKind: TypeAlias = Literal[
    "cancel_paid_order",
    "mark_returned",
    "issue_refund",
    "complete_exchange",
]


@dataclass(frozen=True, slots=True)
class CommerceEffect:
    """A trusted commerce-state change committed with after-sales records.

    All identities and monetary values are derived by World.  The enclosing
    World validates the referenced rows and applies the effect atomically.
    """

    kind: CommerceEffectKind
    order_id: str
    sku_id: str
    qty: int
    amount: int
    currency: str
    inventory_release_qty: int = 0
    payment_before_digest: str | None = None
    packing_before_digest: str | None = None
    fulfillment_stage: Literal["authorized", "created", "packed"] | None = None
    replacement_sku_id: str | None = None
    financial_effect: Literal["none", "void", "refund"] = "none"


@dataclass(frozen=True, slots=True)
class AfterSalesWrite:
    """One immutable row append in an atomic after-sales commit."""

    table: AfterSalesTableName
    key: str
    value: PersistedAfterSalesRow


@dataclass(frozen=True, slots=True)
class AfterSalesOperationRecord:
    """Actor-scoped idempotency and authority evidence for one command."""

    operation_id: str
    operation: str
    order_id: str
    binding_digest: str
    actor_id: str
    idempotency_key: str
    request_fingerprint: str
    logical_tick: int
    result_table: AfterSalesTableName
    result_key: str
    result_digest: str
    effect_digest: str
    operation_digest: str
    schema_id: str = AFTER_SALES_OPERATION_SCHEMA


@dataclass(frozen=True, slots=True)
class AfterSalesCommit:
    """One replayable unit that must share a World transaction boundary."""

    commit_id: str
    expected_logical_tick: int
    logical_tick: int
    context_digest: str
    writes: tuple[AfterSalesWrite, ...]
    commerce_effects: tuple[CommerceEffect, ...]
    operation: AfterSalesOperationRecord
    commit_digest: str
    schema_id: str = AFTER_SALES_COMMIT_SCHEMA


class AfterSalesPersistenceError(ValueError):
    """Persisted after-sales rows or commits violate the storage contract."""


class AfterSalesTables:
    """Typed, indexed, owner-aware in-memory projection of after-sales rows.

    This is the canonical table behavior shared by the future World adapters.
    It is not an independent commerce world.  Orders, inventory, ledger, and
    logical time remain owned by the enclosing World transaction.
    """

    def __init__(self) -> None:
        self._rows: dict[AfterSalesTableName, dict[str, PersistedAfterSalesRow]] = {
            table: {} for table in _TABLE_TYPES
        }
        self._binding_by_order: dict[str, str] = {}
        self._operations: dict[str, AfterSalesOperationRecord] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}
        self._commits: list[AfterSalesCommit] = []

    def clone(self) -> "AfterSalesTables":
        """Return a structural copy suitable for transactional staging."""

        clone = AfterSalesTables()
        clone._rows = {name: dict(rows) for name, rows in self._rows.items()}
        clone._binding_by_order = dict(self._binding_by_order)
        clone._operations = dict(self._operations)
        clone._idempotency = dict(self._idempotency)
        clone._commits = list(self._commits)
        return clone

    def replace_from(self, other: "AfterSalesTables") -> None:
        """Install a fully validated staged state after an atomic commit."""

        if not isinstance(other, AfterSalesTables):
            raise TypeError("replacement must be AfterSalesTables")
        self._rows = {name: dict(rows) for name, rows in other._rows.items()}
        self._binding_by_order = dict(other._binding_by_order)
        self._operations = dict(other._operations)
        self._idempotency = dict(other._idempotency)
        self._commits = list(other._commits)

    def append(self, write: AfterSalesWrite) -> Literal["append", "idempotent"]:
        """Validate and append one immutable row, or classify an exact retry."""

        validate_after_sales_write(write)
        rows = self._rows[write.table]
        existing = rows.get(write.key)
        if existing is not None:
            if record_digest(existing) == record_digest(write.value):
                return "idempotent"
            raise IdempotencyConflict(
                f"after-sales row {write.table}:{write.key} already exists"
            )
        if write.table == "after_sales_bindings":
            binding = cast(AfterSalesOrderBinding, write.value)
            prior_key = self._binding_by_order.get(binding.order_id)
            if prior_key is not None:
                prior = cast(
                    AfterSalesOrderBinding,
                    self._rows["after_sales_bindings"][prior_key],
                )
                if prior.binding_digest == binding.binding_digest:
                    return "idempotent"
                raise AfterSalesPersistenceError(
                    f"order {binding.order_id} already has an after-sales binding"
                )
            self._binding_by_order[binding.order_id] = write.key
        rows[write.key] = write.value
        return "append"

    def append_operation(
        self, operation: AfterSalesOperationRecord
    ) -> Literal["append", "idempotent"]:
        validate_after_sales_operation(operation)
        scope = (
            operation.actor_id,
            operation.operation,
            operation.idempotency_key,
        )
        prior_id = self._idempotency.get(scope)
        if prior_id is not None:
            prior = self._operations[prior_id]
            if prior.operation_digest == operation.operation_digest:
                return "idempotent"
            raise IdempotencyConflict(
                "after-sales idempotency key was reused with different content"
            )
        if operation.operation_id in self._operations:
            prior = self._operations[operation.operation_id]
            if prior.operation_digest == operation.operation_digest:
                return "idempotent"
            raise IdempotencyConflict("after-sales operation id collision")
        self._operations[operation.operation_id] = operation
        self._idempotency[scope] = operation.operation_id
        return "append"

    def append_commit(self, commit: AfterSalesCommit) -> None:
        validate_after_sales_commit(commit)
        for prior in self._commits:
            if prior.commit_id == commit.commit_id:
                if prior.commit_digest == commit.commit_digest:
                    return
                raise IdempotencyConflict("after-sales commit id collision")
        if self._commits:
            if commit.logical_tick <= self._commits[-1].logical_tick:
                raise AfterSalesPersistenceError(
                    "after-sales commit logical time must be strictly increasing"
                )
        self._commits.append(commit)

    def read(
        self,
        table: AfterSalesTableName,
        key: str,
        *,
        caller: str | None,
    ) -> PersistedAfterSalesRow | None:
        row = self._rows[table].get(key)
        if row is None:
            return None
        if _can_read(row, caller):
            return row
        return None

    def all(
        self,
        table: AfterSalesTableName,
        *,
        caller: str | None,
    ) -> Iterator[tuple[str, PersistedAfterSalesRow]]:
        for key, row in sorted(self._rows[table].items()):
            if _can_read(row, caller):
                yield key, row

    def internal_all(
        self, table: AfterSalesTableName
    ) -> Iterator[tuple[str, PersistedAfterSalesRow]]:
        """World-internal deterministic iteration, never exposed to an actor."""

        yield from sorted(self._rows[table].items())

    def binding_for_order(
        self, order_id: str, *, caller: str | None
    ) -> AfterSalesOrderBinding | None:
        key = self._binding_by_order.get(order_id)
        if key is None:
            return None
        return cast(
            AfterSalesOrderBinding | None,
            self.read("after_sales_bindings", key, caller=caller),
        )

    def latest_policy(
        self, merchant_id: str, *, caller: str | None
    ) -> AfterSalesPolicyRevision | None:
        matches = [
            cast(AfterSalesPolicyRevision, row)
            for _, row in self.internal_all("after_sales_policies")
            if cast(AfterSalesPolicyRevision, row).merchant_id == merchant_id
        ]
        if not matches:
            return None
        policy = max(matches, key=lambda row: row.revision)
        return policy if _can_read(policy, caller) else None

    def operation_for_retry(
        self, *, actor_id: str, operation: str, idempotency_key: str
    ) -> AfterSalesOperationRecord | None:
        operation_id = self._idempotency.get(
            (actor_id, operation, idempotency_key)
        )
        return None if operation_id is None else self._operations[operation_id]

    @property
    def operations(self) -> tuple[AfterSalesOperationRecord, ...]:
        return tuple(
            sorted(self._operations.values(), key=lambda row: (row.logical_tick, row.operation_id))
        )

    @property
    def commits(self) -> tuple[AfterSalesCommit, ...]:
        return tuple(self._commits)

    def state_digest(self) -> str:
        return canonical_digest(self.to_wire())

    def to_wire(self) -> dict[str, Any]:
        return {
            "tables": {
                name: [
                    {"key": key, "value": record_to_wire(value)}
                    for key, value in self.internal_all(name)
                ]
                for name in sorted(_TABLE_TYPES)
            },
            "operations": [operation_to_wire(row) for row in self.operations],
        }


def build_after_sales_operation(
    *,
    operation: str,
    order_id: str,
    binding_digest: str,
    actor_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    logical_tick: int,
    result_table: AfterSalesTableName,
    result_key: str,
    result_digest: str,
    effects: Iterable[CommerceEffect],
) -> AfterSalesOperationRecord:
    effect_rows = tuple(effect_to_wire(effect) for effect in effects)
    contract = {
        "schema_id": AFTER_SALES_OPERATION_SCHEMA,
        "operation": _text(operation, "operation"),
        "order_id": _text(order_id, "order_id"),
        "binding_digest": _digest_text(binding_digest, "binding_digest"),
        "actor_id": _text(actor_id, "actor_id"),
        "idempotency_key": _text(idempotency_key, "idempotency_key"),
        "request_fingerprint": _digest_text(
            request_fingerprint, "request_fingerprint"
        ),
        "logical_tick": _nonnegative_int(logical_tick, "logical_tick"),
        "result_table": result_table,
        "result_key": _text(result_key, "result_key"),
        "result_digest": _digest_text(result_digest, "result_digest"),
        "effect_digest": canonical_digest(effect_rows),
    }
    operation_digest = canonical_digest(contract)
    result = AfterSalesOperationRecord(
        operation_id=f"after-sales-op:{operation_digest[:32]}",
        operation=cast(str, contract["operation"]),
        order_id=cast(str, contract["order_id"]),
        binding_digest=cast(str, contract["binding_digest"]),
        actor_id=cast(str, contract["actor_id"]),
        idempotency_key=cast(str, contract["idempotency_key"]),
        request_fingerprint=cast(str, contract["request_fingerprint"]),
        logical_tick=cast(int, contract["logical_tick"]),
        result_table=result_table,
        result_key=cast(str, contract["result_key"]),
        result_digest=cast(str, contract["result_digest"]),
        effect_digest=cast(str, contract["effect_digest"]),
        operation_digest=operation_digest,
    )
    validate_after_sales_operation(result)
    return result


def build_after_sales_commit(
    *,
    expected_logical_tick: int,
    logical_tick: int,
    context_digest: str,
    writes: Iterable[AfterSalesWrite],
    commerce_effects: Iterable[CommerceEffect],
    operation: AfterSalesOperationRecord,
) -> AfterSalesCommit:
    writes_tuple = tuple(writes)
    effects_tuple = tuple(commerce_effects)
    contract = {
        "schema_id": AFTER_SALES_COMMIT_SCHEMA,
        "expected_logical_tick": expected_logical_tick,
        "logical_tick": logical_tick,
        "context_digest": context_digest,
        "writes": [write_to_wire(row) for row in writes_tuple],
        "commerce_effects": [effect_to_wire(row) for row in effects_tuple],
        "operation_digest": operation.operation_digest,
    }
    digest = canonical_digest(contract)
    commit = AfterSalesCommit(
        commit_id=f"after-sales-commit:{digest[:32]}",
        expected_logical_tick=expected_logical_tick,
        logical_tick=logical_tick,
        context_digest=context_digest,
        writes=writes_tuple,
        commerce_effects=effects_tuple,
        operation=operation,
        commit_digest=digest,
    )
    validate_after_sales_commit(commit)
    return commit


def validate_after_sales_write(write: AfterSalesWrite) -> None:
    if not isinstance(write, AfterSalesWrite):
        raise AfterSalesPersistenceError("after-sales write has wrong type")
    expected = _TABLE_TYPES.get(write.table)
    if expected is None or not isinstance(write.value, expected):
        raise AfterSalesPersistenceError(
            f"row type does not match after-sales table {write.table}"
        )
    if write.key != record_key(write.table, write.value):
        raise AfterSalesPersistenceError("after-sales write key is not canonical")
    _validate_row(write.value)


def validate_after_sales_operation(operation: AfterSalesOperationRecord) -> None:
    if not isinstance(operation, AfterSalesOperationRecord):
        raise AfterSalesPersistenceError("operation has wrong type")
    if operation.schema_id != AFTER_SALES_OPERATION_SCHEMA:
        raise AfterSalesPersistenceError("unsupported after-sales operation schema")
    _text(operation.operation_id, "operation_id")
    _text(operation.operation, "operation")
    _text(operation.order_id, "order_id")
    _digest_text(operation.binding_digest, "binding_digest")
    _text(operation.actor_id, "actor_id")
    _text(operation.idempotency_key, "idempotency_key")
    _digest_text(operation.request_fingerprint, "request_fingerprint")
    _nonnegative_int(operation.logical_tick, "logical_tick")
    if operation.result_table not in _TABLE_TYPES:
        raise AfterSalesPersistenceError("invalid operation result table")
    _text(operation.result_key, "result_key")
    _digest_text(operation.result_digest, "result_digest")
    _digest_text(operation.effect_digest, "effect_digest")
    _digest_text(operation.operation_digest, "operation_digest")
    contract = operation_to_wire(operation, include_identity=False)
    if canonical_digest(contract) != operation.operation_digest:
        raise AfterSalesPersistenceError("after-sales operation digest mismatch")
    if operation.operation_id != f"after-sales-op:{operation.operation_digest[:32]}":
        raise AfterSalesPersistenceError("after-sales operation id mismatch")


def validate_after_sales_commit(commit: AfterSalesCommit) -> None:
    if not isinstance(commit, AfterSalesCommit):
        raise AfterSalesPersistenceError("commit has wrong type")
    if commit.schema_id != AFTER_SALES_COMMIT_SCHEMA:
        raise AfterSalesPersistenceError("unsupported after-sales commit schema")
    if commit.logical_tick != commit.expected_logical_tick + 1:
        raise AfterSalesPersistenceError("after-sales commit must advance one tick")
    _digest_text(commit.context_digest, "context_digest")
    for write in commit.writes:
        validate_after_sales_write(write)
    for effect in commit.commerce_effects:
        validate_commerce_effect(effect)
    validate_after_sales_operation(commit.operation)
    expected = canonical_digest(commit_to_wire(commit, include_identity=False))
    if expected != commit.commit_digest:
        raise AfterSalesPersistenceError("after-sales commit digest mismatch")
    if commit.commit_id != f"after-sales-commit:{commit.commit_digest[:32]}":
        raise AfterSalesPersistenceError("after-sales commit id mismatch")
    result = [
        row
        for row in commit.writes
        if row.table == commit.operation.result_table
        and row.key == commit.operation.result_key
    ]
    if len(result) != 1:
        raise AfterSalesPersistenceError(
            "after-sales operation result must be one row in the same commit"
        )
    if record_digest(result[0].value) != commit.operation.result_digest:
        raise AfterSalesPersistenceError("operation result digest mismatch")
    if canonical_digest(tuple(effect_to_wire(row) for row in commit.commerce_effects)) != (
        commit.operation.effect_digest
    ):
        raise AfterSalesPersistenceError("operation effect digest mismatch")


def validate_commerce_effect(effect: CommerceEffect) -> None:
    if not isinstance(effect, CommerceEffect):
        raise AfterSalesPersistenceError("commerce effect has wrong type")
    _text(effect.order_id, "effect.order_id")
    _text(effect.sku_id, "effect.sku_id")
    if isinstance(effect.qty, bool) or not isinstance(effect.qty, int) or effect.qty <= 0:
        raise AfterSalesPersistenceError("effect.qty must be positive")
    _nonnegative_int(effect.amount, "effect.amount")
    _nonnegative_int(effect.inventory_release_qty, "effect.inventory_release_qty")
    if not isinstance(effect.currency, str) or len(effect.currency) != 3:
        raise AfterSalesPersistenceError("effect.currency must be ISO-4217 text")
    if effect.kind == "complete_exchange":
        _text(effect.replacement_sku_id, "effect.replacement_sku_id")
    elif effect.replacement_sku_id is not None:
        raise AfterSalesPersistenceError(
            "replacement_sku_id is allowed only for exchange completion"
        )
    expected_financial = {
        "cancel_paid_order": frozenset({"void", "refund"}),
        "mark_returned": frozenset({"none"}),
        "issue_refund": frozenset({"refund"}),
        "complete_exchange": frozenset({"none"}),
    }[effect.kind]
    if effect.financial_effect not in expected_financial:
        raise AfterSalesPersistenceError("commerce effect financial type mismatch")
    if effect.kind in {"cancel_paid_order", "complete_exchange"}:
        if effect.inventory_release_qty != effect.qty:
            raise AfterSalesPersistenceError(
                "cancellation and exchange must release the bound quantity"
            )
    elif effect.kind == "mark_returned" and effect.inventory_release_qty != 0:
        raise AfterSalesPersistenceError(
            "physical return receipt cannot release inventory before resolution"
        )
    elif effect.inventory_release_qty > effect.qty:
        raise AfterSalesPersistenceError(
            "refund inventory release cannot exceed the bound quantity"
        )
    if effect.kind == "cancel_paid_order":
        _digest_text(effect.payment_before_digest, "effect.payment_before_digest")
        if effect.fulfillment_stage == "authorized":
            if effect.packing_before_digest is not None:
                raise AfterSalesPersistenceError(
                    "authorized cancellation cannot bind packing state"
                )
        elif effect.fulfillment_stage in {"created", "packed"}:
            _digest_text(effect.packing_before_digest, "effect.packing_before_digest")
        else:
            raise AfterSalesPersistenceError(
                "cancellation fulfillment stage is invalid"
            )
    elif effect.kind == "issue_refund":
        _digest_text(effect.payment_before_digest, "effect.payment_before_digest")
        if effect.packing_before_digest is not None or effect.fulfillment_stage is not None:
            raise AfterSalesPersistenceError(
                "ordinary refund cannot mutate pre-dispatch packing state"
            )
    elif any(
        value is not None
        for value in (
            effect.payment_before_digest,
            effect.packing_before_digest,
            effect.fulfillment_stage,
        )
    ):
        raise AfterSalesPersistenceError(
            "payment and packing causal fields are cancellation-only"
        )


def replay_after_sales(commits: Iterable[AfterSalesCommit]) -> AfterSalesTables:
    """Rebuild after-sales tables from the ordered committed journal."""

    tables = AfterSalesTables()
    for commit in commits:
        validate_after_sales_commit(commit)
        staged = tables.clone()
        for write in commit.writes:
            staged.append(write)
        staged.append_operation(commit.operation)
        staged.append_commit(commit)
        tables.replace_from(staged)
    return tables


def record_key(
    table: AfterSalesTableName, value: PersistedAfterSalesRow
) -> str:
    if table == "after_sales_policies":
        row = cast(AfterSalesPolicyRevision, value)
        return f"{row.merchant_id}:{row.revision}"
    if table == "after_sales_bindings":
        return cast(AfterSalesOrderBinding, value).binding_digest
    fields = {
        "paid_cancellations": "cancellation_id",
        "return_requests": "request_id",
        "return_authorizations": "authorization_id",
        "return_receipts": "receipt_id",
        "refund_cases": "case_id",
        "refund_decisions": "decision_id",
        "exchange_cases": "case_id",
        "dispute_cases": "dispute_id",
        "dispute_evidence": "evidence_id",
        "dispute_responses": "response_id",
        "after_sales_rulings": "ruling_id",
        "ledger_reconciliation_sources": "source_id",
        "ledger_reconciliation_requests": "request_id",
        "ledger_reconciliation_results": "result_id",
    }
    field = fields[table]
    stable = _text(getattr(value, field), field)
    if table in {"exchange_cases", "dispute_cases"}:
        return f"{stable}:v{cast(Any, value).version}"
    return stable


def record_table(value: PersistedAfterSalesRow) -> AfterSalesTableName:
    """Return the canonical logical table for a persisted after-sales row.

    History APIs expose heterogeneous immutable records.  Keeping this mapping
    beside the strict codecs prevents HTTP adapters and benchmark code from
    guessing a table from a schema id or Python class name.
    """

    for table, expected_type in _TABLE_TYPES.items():
        if isinstance(value, expected_type):
            return table
    raise AfterSalesPersistenceError(
        f"unsupported after-sales record type: {type(value).__name__}"
    )


def physical_after_sales_record_lookup_key(
    table: AfterSalesTableName, key: str
) -> str:
    """Return the physical World key for one exact logical result row.

    Callers must already know both the typed table and its canonical logical
    key.  This helper deliberately does not parse, truncate, or otherwise
    guess versioned identifiers.
    """

    if table not in _TABLE_TYPES:
        raise AfterSalesPersistenceError("invalid after-sales result table")
    if table == "after_sales_policies":
        raise AfterSalesPersistenceError(
            "policies belong in the dedicated after_sales_policies collection"
        )
    logical_key = _text(key, "after-sales result key")
    return f"{len(table)}:{table}:{len(logical_key)}:{logical_key}"


def physical_after_sales_record_key(write: AfterSalesWrite) -> str:
    """Key one typed domain row in World's generic physical collection.

    ``AfterSalesTables`` keeps domain-specific names while CommerceWorld owns a
    single durable ``after_sales_records`` collection.  Prefixing the canonical
    domain key prevents collisions between, for example, a return request and
    a reconciliation request that happen to share an identifier.
    """

    validate_after_sales_write(write)
    return physical_after_sales_record_lookup_key(write.table, write.key)


_SAFE_RESULT_REFERENCE_FIELDS: dict[AfterSalesTableName, tuple[str, ...]] = {
    "after_sales_policies": (),
    "after_sales_bindings": (),
    "paid_cancellations": ("cancellation_id",),
    "return_requests": ("request_id",),
    "return_authorizations": ("request_id", "authorization_id"),
    "return_receipts": ("request_id", "authorization_id", "receipt_id"),
    "refund_cases": ("case_id",),
    "refund_decisions": ("case_id", "decision_id"),
    "exchange_cases": ("case_id",),
    "dispute_cases": ("dispute_id",),
    "dispute_evidence": ("dispute_id", "evidence_id"),
    "dispute_responses": ("dispute_id", "response_id"),
    "after_sales_rulings": ("dispute_id", "ruling_id"),
    "ledger_reconciliation_sources": ("source_id",),
    "ledger_reconciliation_requests": ("request_id",),
    "ledger_reconciliation_results": ("request_id", "result_id"),
}


def after_sales_result_references(
    write: AfterSalesWrite,
    operation: AfterSalesOperationRecord,
) -> dict[str, str]:
    """Project safe stable identifiers from one exact committed result row.

    The operation's table, physical key, and digest must all identify the
    supplied typed row.  Reference fields are selected by an explicit
    per-table allowlist.  Owner identities, monetary values, bindings, policy,
    and any other row content can therefore never enter an actor-facing ack.
    """

    validate_after_sales_write(write)
    validate_after_sales_operation(operation)
    if write.table != operation.result_table or write.key != operation.result_key:
        raise AfterSalesPersistenceError(
            "after-sales ack result row does not match operation identity"
        )
    if record_digest(write.value) != operation.result_digest:
        raise AfterSalesPersistenceError(
            "after-sales ack result row does not match operation digest"
        )
    references: dict[str, str] = {}
    for field in _SAFE_RESULT_REFERENCE_FIELDS[write.table]:
        value = _text(getattr(write.value, field), f"result reference {field}")
        references[field] = value
    return references


def record_digest(value: PersistedAfterSalesRow) -> str:
    if isinstance(value, AfterSalesPolicyRevision):
        return value.policy_digest
    if isinstance(value, AfterSalesOrderBinding):
        return value.binding_digest
    if isinstance(value, LedgerReconciliationSource):
        return value.source_digest
    return cast(Any, value).record_digest


def record_to_wire(value: PersistedAfterSalesRow) -> dict[str, Any]:
    if isinstance(value, AfterSalesPolicyRevision):
        return after_sales_policy_to_dict(value)
    if isinstance(value, AfterSalesOrderBinding):
        return after_sales_binding_to_dict(value)
    if isinstance(
        value,
        (
            PaidCancellationRecord,
            DisputeResponseRecord,
            LedgerReconciliationRequest,
            LedgerReconciliationResult,
        ),
    ):
        return core_after_sales_record_to_dict(value)
    if isinstance(value, LedgerReconciliationSource):
        return ledger_reconciliation_source_to_dict(value)
    return after_sales_record_to_dict(cast(AfterSalesRecord, value))


def record_from_wire(
    table: AfterSalesTableName, value: Mapping[str, Any]
) -> PersistedAfterSalesRow:
    """Strict SQLite/HTTP-independent codec for one typed table row."""

    if not isinstance(value, Mapping):
        raise AfterSalesPersistenceError("after-sales wire row must be an object")
    if table == "after_sales_policies":
        row: PersistedAfterSalesRow = after_sales_policy_from_dict(value)
    elif table == "after_sales_bindings":
        row = after_sales_binding_from_json(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    elif table == "ledger_reconciliation_sources":
        row = ledger_reconciliation_source_from_dict(value)
    elif table in {
        "paid_cancellations",
        "dispute_responses",
        "ledger_reconciliation_requests",
        "ledger_reconciliation_results",
    }:
        row = core_after_sales_record_from_dict(value)
    else:
        row = after_sales_record_from_json(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
    validate_after_sales_write(
        AfterSalesWrite(table, record_key(table, row), row)
    )
    return row


def write_to_wire(write: AfterSalesWrite) -> dict[str, Any]:
    return {
        "table": write.table,
        "key": write.key,
        "value": record_to_wire(write.value),
    }


def effect_to_wire(effect: CommerceEffect) -> dict[str, Any]:
    validate_commerce_effect(effect)
    return cast(dict[str, Any], _json_value(asdict(effect)))


def operation_to_wire(
    operation: AfterSalesOperationRecord, *, include_identity: bool = True
) -> dict[str, Any]:
    value = cast(dict[str, Any], _json_value(asdict(operation)))
    if not include_identity:
        value.pop("operation_id")
        value.pop("operation_digest")
    return value


def commit_to_wire(
    commit: AfterSalesCommit, *, include_identity: bool = True
) -> dict[str, Any]:
    value = {
        "schema_id": commit.schema_id,
        "expected_logical_tick": commit.expected_logical_tick,
        "logical_tick": commit.logical_tick,
        "context_digest": commit.context_digest,
        "writes": [write_to_wire(row) for row in commit.writes],
        "commerce_effects": [effect_to_wire(row) for row in commit.commerce_effects],
        "operation_digest": commit.operation.operation_digest,
    }
    if include_identity:
        value = {
            "commit_id": commit.commit_id,
            **value,
            "commit_digest": commit.commit_digest,
        }
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_row(value: PersistedAfterSalesRow) -> None:
    if isinstance(value, AfterSalesPolicyRevision):
        validate_after_sales_policy(value)
    elif isinstance(value, AfterSalesOrderBinding):
        validate_after_sales_order_binding(value)
    elif isinstance(value, LedgerReconciliationSource):
        validate_ledger_reconciliation_source(value)
    elif isinstance(
        value,
        (
            PaidCancellationRecord,
            DisputeResponseRecord,
            LedgerReconciliationRequest,
            LedgerReconciliationResult,
        ),
    ):
        validate_core_after_sales_record(value)
    else:
        validate_after_sales_record(cast(AfterSalesRecord, value))


def _binding(value: PersistedAfterSalesRow) -> AfterSalesOrderBinding | None:
    if isinstance(value, AfterSalesOrderBinding):
        return value
    if isinstance(value, AfterSalesPolicyRevision):
        return None
    return cast(Any, value).binding


def _can_read(value: PersistedAfterSalesRow, caller: str | None) -> bool:
    if caller is None or caller.startswith("platform:") or caller == "world":
        return True
    if isinstance(value, AfterSalesPolicyRevision):
        return caller == value.merchant_id
    binding = _binding(value)
    assert binding is not None
    return caller in {binding.owner_id, binding.merchant_id}


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, (dict, Mapping, MappingProxyType)):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AfterSalesPersistenceError(f"{label} must be non-empty text")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AfterSalesPersistenceError(f"{label} must be non-negative integer")
    return value


def _digest_text(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise AfterSalesPersistenceError(f"{label} must be lowercase sha256")
    return text


_TABLE_TYPES: dict[AfterSalesTableName, type[Any] | tuple[type[Any], ...]] = {
    "after_sales_policies": AfterSalesPolicyRevision,
    "after_sales_bindings": AfterSalesOrderBinding,
    "paid_cancellations": PaidCancellationRecord,
    "return_requests": ReturnRequest,
    "return_authorizations": ReturnAuthorization,
    "return_receipts": ReturnReceipt,
    "refund_cases": RefundCase,
    "refund_decisions": RefundDecision,
    "exchange_cases": ExchangeCase,
    "dispute_cases": DisputeCase,
    "dispute_evidence": DisputeEvidence,
    "dispute_responses": DisputeResponseRecord,
    "after_sales_rulings": Ruling,
    "ledger_reconciliation_sources": LedgerReconciliationSource,
    "ledger_reconciliation_requests": LedgerReconciliationRequest,
    "ledger_reconciliation_results": LedgerReconciliationResult,
}


__all__ = [
    "AFTER_SALES_COMMIT_SCHEMA",
    "AFTER_SALES_OPERATION_SCHEMA",
    "AfterSalesCommit",
    "AfterSalesOperationRecord",
    "AfterSalesPersistenceError",
    "AfterSalesTableName",
    "AfterSalesTables",
    "AfterSalesWrite",
    "CommerceEffect",
    "PersistedAfterSalesRow",
    "after_sales_result_references",
    "build_after_sales_commit",
    "build_after_sales_operation",
    "canonical_digest",
    "commit_to_wire",
    "effect_to_wire",
    "operation_to_wire",
    "physical_after_sales_record_key",
    "physical_after_sales_record_lookup_key",
    "record_digest",
    "record_from_wire",
    "record_key",
    "record_table",
    "record_to_wire",
    "replay_after_sales",
    "validate_after_sales_commit",
    "validate_after_sales_operation",
    "validate_after_sales_write",
    "validate_commerce_effect",
    "write_to_wire",
]
