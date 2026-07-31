"""SQLite-backed storage for the World state tables."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast
from uuid import uuid4

from protocol.errors import SchemaError
from protocol.cart_quote_state import (
    CartQuoteStaleError,
    PersistentCartQuote,
    apply_cart_quote_record,
    assert_cart_quote_snapshots_current,
    assert_cart_quote_usable,
    persistent_cart_quote_from_json,
    persistent_cart_quote_to_json,
    validate_persistent_cart_quote,
)
from protocol.cart_quote_request import (
    CartQuoteRequestAuthorityError,
    CartQuoteRequestLine,
    CartQuoteRequestStaleError,
    PersistentCartQuoteRequest,
    assert_cart_quote_request_usable,
    build_persistent_cart_quote_request,
    coerce_persistent_cart_quote_request,
    persistent_cart_quote_request_to_dict,
    validate_persistent_cart_quote_request,
)
from protocol.evidence_records import (
    EvidenceRecord,
    MandateRevision,
    MandateRevisionAuthority,
    coerce_evidence_record,
    coerce_mandate_revision,
    evidence_record_to_json,
    mandate_revision_to_json,
)
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventReceipt,
    ProtocolEventSchemaError,
    apply_protocol_event,
    apply_protocol_receipt,
    build_protocol_event_receipt,
    protocol_event_from_json,
    protocol_event_receipt_from_json,
    protocol_event_receipt_to_json,
    protocol_event_to_json,
    replay_protocol_events,
    replay_protocol_receipts,
    validate_protocol_event,
    validate_protocol_event_receipt,
    validate_protocol_receipt_stream,
)
from protocol.matching import (
    MatchAcceptance,
    MatchAcceptanceRejected,
    MatchCertificate,
    MatchValidationError,
    SearchSession,
    canonical_digest,
    coerce_match_acceptance,
    coerce_match_certificate,
    coerce_search_session,
    issue_match_certificate as build_match_certificate,
    match_acceptance_to_wire,
    match_certificate_to_wire,
    search_session_to_wire,
    validate_search_session,
)
from protocol.listing_claims import (
    ListingClaim,
    coerce_listing_claim,
    listing_claim_to_json,
)
from protocol.negotiation_state import (
    PROPOSE_OFFER,
    NegotiationBinding,
    NegotiationEvent,
    NegotiationSchemaError,
    NegotiationThread,
    apply_negotiation_event,
    build_negotiation_event,
    build_next_negotiation_event,
    negotiation_event_from_json,
    negotiation_event_to_json,
    negotiation_thread_from_json,
    negotiation_thread_to_json,
    replay_negotiation_events,
)
from protocol.pricing_policy import (
    GENESIS_PRICING_POLICY_DIGEST,
    PricingPolicyRevision,
    apply_pricing_policy_revision,
    build_pricing_policy_revision,
    pricing_policy_revision_from_json,
    pricing_policy_revision_to_json,
)
from protocol.supply_authority import (
    DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    SupplyPurchaseAuthority,
    build_supply_purchase_authority,
    supply_purchase_authority_from_json,
    supply_purchase_authority_id,
    supply_purchase_authority_to_json,
    validate_supply_purchase_authority,
    validate_supply_purchase_authority_ttl_ticks,
)

from world.catalog_mutations import (
    CatalogMutationIntent,
    apply_catalog_mutation_intent,
    catalog_listings_semantically_equal,
    catalog_mutation_fingerprint,
    catalog_owner_for_actor,
    normalize_catalog_mutation_intent,
)
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesPolicyRevision,
    after_sales_intent_fingerprint,
    authoritative_order_digest,
    build_after_sales_policy_revision,
    derive_after_sales_binding,
    derive_ledger_reconciliation_source,
    normalize_after_sales_intent,
)
from world.after_sales_persistence import (
    AfterSalesCommit,
    AfterSalesOperationRecord,
    AfterSalesTables,
    AfterSalesWrite,
    CommerceEffect,
    build_after_sales_operation,
    canonical_digest as after_sales_digest,
    physical_after_sales_record_key,
    physical_after_sales_record_lookup_key,
    record_digest as after_sales_record_digest,
    record_from_wire as after_sales_record_from_wire,
    record_to_wire as after_sales_record_to_wire,
    write_to_wire as after_sales_write_to_wire,
)
from world.after_sales_service import (
    AfterSalesCommandResult,
    AfterSalesContextConflict,
    AfterSalesPlanner,
    TrustedAfterSalesContext,
    TrustedReplacementContext,
    load_trusted_after_sales_evidence,
    resolve_current_versioned_record,
    stage_policy_publication,
)
from world.market_governance_persistence import (
    GovernancePolicyEnvelope,
    GovernanceRecordEnvelope,
    envelope_key as governance_envelope_key,
    policy_envelope_from_wire,
    policy_envelope_to_wire,
    record_envelope_from_wire,
    record_envelope_to_wire,
)
from world.cart_pricing import (
    CartQuoteIntent,
    PricingPolicyIntent,
    QuoteAuthority,
    build_authoritative_cart_quote,
    cart_quote_intent_fingerprint,
    catalog_snapshot_digest,
    inventory_snapshot_digest,
    normalize_cart_quote_intent,
    normalize_pricing_policy_intent,
    pricing_policy_set_digest,
    pricing_policy_intent_fingerprint,
    pricing_policy_revision_key,
)
from world.evidence_contracts import (
    authority_operation_key,
    authorize_mandate_read,
    authorize_persisted_evidence_read,
    coerce_mandate_authority,
    evidence_record_key,
    mandate_authority_to_wire,
    mandate_revision_key,
    validate_evidence_append,
    validate_listing_claim_append,
    validate_mandate_append,
    validate_mandate_authority_registration,
)
from world.negotiations import (
    NegotiationIntent,
    negotiation_event_id,
    negotiation_intent_fingerprint,
    negotiation_listing_digest,
    normalize_negotiation_intent,
)
from world.errors import (
    AfterSalesReferenceRejected,
    DisputeNotActionable,
    ExchangeNotActionable,
    FulfillmentNotActionable,
    IdempotencyConflict,
    InvalidOrderTransition,
    LogicalTimeError,
    LifecycleAuthorizationError,
    OrderNotRefundable,
    OrderNotSettleable,
    OutOfStock,
    ShipmentNotActionable,
    TableNotFound,
    WorldError,
    WriteNotAuthorized,
)
from world.state import (
    World,
    _ALREADY_SETTLED,
    _REFUNDABLE,
    _SETTLEABLE,
    _SHIPMENT_TRANSITIONS,
    _available_qty,
    _active_pricing_policy_for_listing,
    _catalog_revision,
    _captured_return_window,
    _enforce_return_window,
    _authorize_lifecycle_actor,
    _authorize_exchange_actor,
    _authorize_psp,
    _require_inventory_owner,
    _require_listing_owner,
    _reputation_settlement_event_id,
    _reserve_inventory,
    _revisioned_listing,
    _order_identity_matches,
    _match_acceptance_key,
    _mandate_budget_minor,
    _money_cents,
    _negotiation_parties,
    _validate_event_order_binding,
    _validate_group_payment_record,
    _validate_seeded_evidence_contracts,
    _validate_protocol_event_reference,
    _protocol_effect_idempotency_key,
    _protocol_payment_receipt,
    _protocol_process_receipt_id,
    _protocol_process_receipt_replay,
    _validate_protocol_process_precondition,
    _same_dispute_claim,
    _supply_projection,
    _validate_partial_receipt,
    _validate_replacement_identity,
    _validate_requested_and_fulfilled,
    _validate_same_order_identity,
    _validate_dispute_order,
    _validate_open_dispute_shape,
    _validate_reputation_settlement_identity,
    _validate_ruling,
    order_operation_reference_digest,
    protocol_operation_effect_reference_digest,
    _after_sales_refund_receipt,
    _after_sales_context_evidence_digests,
    _after_sales_context_hint,
    _charge_receipt_for_order,
    _effects_for_persisted_after_sales,
    _normalize_after_sales_policy_intent,
    _after_sales_policy_for_context,
    _receipt_tick,
)
from world.payment_fulfillment import (
    PackingRecord,
    PaymentStateRecord,
    derive_dispatch_packing_sequence,
    derive_packing_transition,
    derive_payment_authorization,
    derive_payment_capture,
    derive_payment_resolution,
    normalize_packing_intent,
    normalize_payment_intent,
    packing_record_from_dict,
    packing_record_key,
    packing_record_to_dict,
    payment_state_from_dict,
    payment_state_key,
    payment_state_to_dict,
)
from world.tables import (
    AfterSalesPolicyTable,
    AfterSalesRecordTable,
    AuthorityOperationTable,
    CatalogTable,
    DisputeTable,
    EvidenceRecordTable,
    ExchangeTable,
    GovernancePolicyTable,
    GovernanceRecordTable,
    FulfillmentTable,
    InventoryTable,
    LedgerTable,
    ListingClaimTable,
    MandateAuthorityTable,
    MandateRevisionTable,
    MatchAcceptanceTable,
    MatchCertificateTable,
    NegotiationEventTable,
    NegotiationThreadTable,
    OrderTable,
    OrderGroupTable,
    OrderTimelineTable,
    PackingRecordTable,
    PaymentStateTable,
    ProtocolEventTable,
    ProtocolReceiptTable,
    PricingPolicyRevisionTable,
    PersistentCartQuoteTable,
    PersistentCartQuoteRequestTable,
    ReputationSettlementTable,
    ReputationTable,
    ReviewTable,
    RulingTable,
    ShipmentTable,
    SearchSessionTable,
    SupplyPurchaseAuthorityTable,
    Table,
)
from world.transactions import (
    validate_listing_owner,
    validate_persisted_order_identity,
    validate_transaction_identity,
)
from world.types import (
    AgentId,
    AllocationBatch,
    AuthorityOperationRecord,
    Dispute,
    DisputeId,
    DisputeState,
    Exchange,
    ExchangeId,
    FeeComponent,
    FulfillmentAllocation,
    InventoryRow,
    Listing,
    Money,
    Order,
    OrderGroup,
    OrderGroupId,
    OrderId,
    OrderState,
    OrderTimeline,
    Receipt,
    QuoteId,
    ReputationScore,
    ReputationSettlement,
    ReputationSettlementSource,
    Ruling,
    RulingId,
    Shipment,
    ShipmentId,
    ShipmentResolution,
    ShipmentStatus,
    ShipmentStatusEvent,
    SkuId,
    Review,
    ReviewId,
    SupplyEventRecord,
    SupplyState,
    TxnId,
    TableWrite,
    WorldCommitRecord,
    WorldSnapshot,
)


_AUTHORITY_TABLES = frozenset(
    {
        "protocol_events",
        "protocol_receipts",
        "evidence_records",
        "mandate_authorities",
        "mandate_revisions",
        "listing_claims",
        "authority_operations",
        "negotiation_events",
        "negotiation_threads",
        "pricing_policy_revisions",
        "persistent_cart_quote_requests",
        "persistent_cart_quotes",
        "supply_purchase_authorities",
        "order_groups",
        "payment_states",
        "packing_records",
        "after_sales_policies",
        "after_sales_records",
        "governance_policies",
        "governance_records",
    }
)

_GOVERNANCE_AUTHORITY_SCOPES = frozenset(
    {
        "publish_governance_policy",
        "aggregate_reviews",
        "ingest_review_observation",
        "ingest_market_observation",
        "resolve_governance_case",
        "apply_governance_reputation",
        "create_remediation_plan",
        "verify_remediation_step",
        "persist_ranking_context",
    }
)


class DatabaseWorld:
    """World-compatible facade backed by ``SQLiteWorldStore``."""

    def __init__(self, path: str | Path = ":memory:", *, mode: Literal["E0", "E1"] = "E0") -> None:
        self._store = SQLiteWorldStore(path)
        self._mode = mode
        self._tables: dict[str, Table[Any, Any]] = {
            "catalog": CatalogTable(),
            "inventory": InventoryTable(),
            "orders": OrderTable(),
            "ledger": LedgerTable(),
            "reputation": ReputationTable(),
            "reputation_settlements": ReputationSettlementTable(),
            "disputes": DisputeTable(),
            "rulings": RulingTable(),
            "reviews": ReviewTable(),
            "fulfillments": FulfillmentTable(),
            "exchanges": ExchangeTable(),
            "shipments": ShipmentTable(),
            "order_timelines": OrderTimelineTable(),
            "order_groups": OrderGroupTable(),
            "search_sessions": SearchSessionTable(),
            "match_acceptances": MatchAcceptanceTable(),
            "match_certificates": MatchCertificateTable(),
            "supply_purchase_authorities": SupplyPurchaseAuthorityTable(),
            "negotiation_events": NegotiationEventTable(),
            "negotiation_threads": NegotiationThreadTable(),
            "pricing_policy_revisions": PricingPolicyRevisionTable(),
            "persistent_cart_quote_requests": PersistentCartQuoteRequestTable(),
            "persistent_cart_quotes": PersistentCartQuoteTable(),
            "protocol_events": ProtocolEventTable(),
            "protocol_receipts": ProtocolReceiptTable(),
            "evidence_records": EvidenceRecordTable(),
            "mandate_authorities": MandateAuthorityTable(),
            "mandate_revisions": MandateRevisionTable(),
            "listing_claims": ListingClaimTable(),
            "authority_operations": AuthorityOperationTable(),
            "payment_states": PaymentStateTable(),
            "packing_records": PackingRecordTable(),
            "after_sales_policies": AfterSalesPolicyTable(),
            "after_sales_records": AfterSalesRecordTable(),
            "governance_policies": GovernancePolicyTable(),
            "governance_records": GovernanceRecordTable(),
        }

    def read(self, table: str, key: Any, *, caller: str | None = None) -> Any | None:
        self._table(table)
        return self._store.read(table, key, caller=caller)

    def begin_evidence_window(self) -> int:
        return self._store.begin_evidence_window()

    @property
    def commit_journal(self) -> tuple[WorldCommitRecord, ...]:
        return self._store.commit_journal

    def commits_since(self, cursor: int) -> tuple[WorldCommitRecord, ...]:
        return self._store.commits_since(cursor)

    def authorize_payment(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        return self._store.authorize_payment(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def capture_payment(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        return self._store.capture_payment(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_packing_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PackingRecord:
        return self._store.apply_packing_intent(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def publish_after_sales_policy(
        self,
        policy_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesPolicyRevision:
        return self._store.publish_after_sales_policy(
            policy_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_after_sales_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        return self._store.apply_after_sales_intent(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def complete_ledger_reconciliation(
        self,
        order_id: str,
        request_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        return self._store.complete_ledger_reconciliation(
            order_id,
            request_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def payment_history(
        self, order_id: str, *, caller: str
    ) -> tuple[PaymentStateRecord, ...]:
        return self._store.payment_history(order_id, caller=caller)

    def ledger_history(self, order_id: str, *, caller: str) -> tuple[Receipt, ...]:
        return self._store.ledger_history(order_id, caller=caller)

    def packing_history(
        self, order_id: str, *, caller: str
    ) -> tuple[PackingRecord, ...]:
        return self._store.packing_history(order_id, caller=caller)

    def after_sales_history(self, order_id: str, *, caller: str) -> tuple[Any, ...]:
        return self._store.after_sales_history(order_id, caller=caller)

    def after_sales_result_record(
        self,
        operation: AfterSalesOperationRecord,
        *,
        caller: str,
    ) -> AfterSalesWrite | None:
        return self._store.after_sales_result_record(operation, caller=caller)

    def after_sales_policy(
        self, merchant_id: str, *, caller: str
    ) -> AfterSalesPolicyRevision | None:
        return self._store.after_sales_policy(merchant_id, caller=caller)

    def load_after_sales_context(
        self,
        order_id: str,
        *,
        intent: Mapping[str, Any] | None = None,
        original_actor: str | None = None,
        evidence_digests: Mapping[str, str] | None = None,
    ) -> TrustedAfterSalesContext:
        """Delegate the exact planner context through the database facade."""

        return self._store.load_after_sales_context(
            order_id,
            intent=intent,
            original_actor=original_actor,
            evidence_digests=evidence_digests,
        )

    def publish_governance_policy(
        self,
        policy_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.publish_governance_policy(
            policy_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_governance_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.apply_governance_intent(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def aggregate_reviews(
        self,
        sku_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.aggregate_reviews(
            sku_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def ingest_review_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.ingest_review_observation(
            record_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def ingest_market_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.ingest_market_observation(
            record_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def resolve_governance_case(
        self,
        decision_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.resolve_governance_case(
            decision_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_governance_reputation(
        self,
        source_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.apply_governance_reputation(
            source_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def create_remediation_plan(
        self,
        plan_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.create_remediation_plan(
            plan_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def verify_remediation_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.verify_remediation_step(
            plan_id,
            step_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def persist_ranking_context(
        self,
        ranking_result: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._store.persist_ranking_context(
            ranking_result,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def governance_history(
        self, record_kind: str, stable_id: str, *, caller: str
    ) -> tuple[Any, ...]:
        return self._store.governance_history(
            record_kind, stable_id, caller=caller
        )

    def ranking_context_projection(
        self, context_id: str, *, caller: str
    ) -> dict[str, Any]:
        return self._store.ranking_context_projection(context_id, caller=caller)

    def read_order_state_revision(
        self,
        order_id: OrderId | str,
        *,
        caller: str,
    ) -> int | None:
        return self._store.read_order_state_revision(order_id, caller=caller)

    def read_order_operation_reference(
        self,
        order_id: OrderId | str,
        *,
        caller: str,
    ) -> str | None:
        return self._store.read_order_operation_reference(order_id, caller=caller)

    def protocol_events_for_stream(
        self,
        binding_digest: str,
        *,
        caller: str,
    ) -> tuple[ProtocolEvent, ...]:
        return self._store.protocol_events_for_stream(
            binding_digest,
            caller=caller,
        )

    def protocol_receipts_for_stream(
        self,
        binding_digest: str,
        *,
        order_id: OrderId | str,
        caller: str,
    ) -> tuple[ProtocolEventReceipt, ...]:
        return self._store.protocol_receipts_for_stream(
            binding_digest,
            order_id=order_id,
            caller=caller,
        )

    def read_negotiation_event(
        self, event_id: str, *, caller: str
    ) -> NegotiationEvent | None:
        return self._store.read_negotiation_event(event_id, caller=caller)

    def read_negotiation_thread(
        self, negotiation_id: str, *, caller: str
    ) -> NegotiationThread | None:
        return self._store.read_negotiation_thread(
            negotiation_id, caller=caller
        )

    def negotiation_events_for_thread(
        self, negotiation_id: str, *, caller: str
    ) -> tuple[NegotiationEvent, ...]:
        return self._store.negotiation_events_for_thread(
            negotiation_id, caller=caller
        )

    def publish_protocol_event(
        self,
        event: ProtocolEvent,
        *,
        by_actor: str,
    ) -> ProtocolEvent:
        return self._store.publish_protocol_event(event, by_actor=by_actor)

    def append_protocol_receipt(
        self,
        receipt: ProtocolEventReceipt,
        *,
        by_actor: str,
        original_actor: str,
    ) -> ProtocolEventReceipt:
        return self._store.append_protocol_receipt(
            receipt,
            by_actor=by_actor,
            original_actor=original_actor,
        )

    def process_protocol_event(
        self,
        *,
        event_id: str,
        by_actor: str,
        original_actor: str,
        reason: str,
        idempotency_key: str,
    ) -> ProtocolEventReceipt:
        """Execute a registered core operation and persist its linked receipt."""

        return self._store.process_protocol_event(
            event_id=event_id,
            by_actor=by_actor,
            original_actor=original_actor,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def apply_catalog_mutation(
        self,
        intent: CatalogMutationIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Listing:
        return self._store.apply_catalog_mutation(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_negotiation_intent(
        self,
        intent: NegotiationIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        max_rounds: int,
        deadline_ticks: int,
    ) -> NegotiationEvent:
        return self._store.apply_negotiation_intent(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            max_rounds=max_rounds,
            deadline_ticks=deadline_ticks,
        )

    def publish_pricing_policy(
        self,
        intent: PricingPolicyIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PricingPolicyRevision:
        return self._store.publish_pricing_policy(
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def pricing_policy_revisions(
        self,
        market_id: str,
        merchant_id: str,
        policy_id: str,
        *,
        caller: str,
    ) -> tuple[PricingPolicyRevision, ...]:
        return self._store.pricing_policy_revisions(
            market_id,
            merchant_id,
            policy_id,
            caller=caller,
        )

    def issue_cart_quote(
        self,
        intent: CartQuoteIntent | Mapping[str, Any],
        *,
        market_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        return self._store.issue_cart_quote(
            intent,
            market_id=market_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            quote_ttl_ticks=quote_ttl_ticks,
        )

    def create_cart_quote_request(
        self,
        intent: CartQuoteIntent | Mapping[str, Any],
        *,
        market_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        request_ttl_ticks: int = 10,
    ) -> PersistentCartQuoteRequest:
        return self._store.create_cart_quote_request(
            intent,
            market_id=market_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            request_ttl_ticks=request_ttl_ticks,
        )

    def issue_cart_quote_from_request(
        self,
        request_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        return self._store.issue_cart_quote_from_request(
            request_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            quote_ttl_ticks=quote_ttl_ticks,
        )

    def checkout_cart(
        self,
        *,
        quote_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> OrderGroup:
        return self._store.checkout_cart(
            quote_id=quote_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def persist_evidence_record(
        self,
        record: EvidenceRecord,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> EvidenceRecord:
        return self._store.persist_evidence_record(
            record,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def read_evidence_record(
        self,
        record_id: str,
        *,
        caller: str,
        version: int | None = None,
        record_digest: str | None = None,
    ) -> EvidenceRecord | None:
        return self._store.read_evidence_record(
            record_id,
            caller=caller,
            version=version,
            record_digest=record_digest,
        )

    def register_mandate_authority(
        self,
        authority: MandateRevisionAuthority,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevisionAuthority:
        return self._store.register_mandate_authority(
            authority,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def append_mandate_revision(
        self,
        revision: MandateRevision,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevision:
        return self._store.append_mandate_revision(
            revision,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def mandate_revisions(
        self, mandate_id: str, *, caller: str
    ) -> tuple[MandateRevision, ...]:
        return self._store.mandate_revisions(mandate_id, caller=caller)

    def apply_listing_claim(
        self,
        claim: ListingClaim,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ListingClaim:
        return self._store.apply_listing_claim(
            claim,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def read_listing_claim(
        self, claim_id: str, *, caller: str
    ) -> ListingClaim | None:
        return self._store.read_listing_claim(claim_id, caller=caller)

    def listing_claims_for_listing(
        self, listing_id: str, *, caller: str
    ) -> tuple[ListingClaim, ...]:
        return self._store.listing_claims_for_listing(listing_id, caller=caller)

    def write(self, table: str, key: Any, value: Any, *, by_action: str) -> None:
        target = self._table(table)
        if table in {
            "protocol_events",
            "protocol_receipts",
            "evidence_records",
            "mandate_authorities",
            "mandate_revisions",
            "listing_claims",
            "negotiation_events",
            "negotiation_threads",
            "pricing_policy_revisions",
            "persistent_cart_quote_requests",
            "persistent_cart_quotes",
            "supply_purchase_authorities",
            "order_groups",
        }:
            raise WriteNotAuthorized(
                "authority records must use their validating World APIs"
            )
        action = by_action.removeprefix("world.")
        if by_action not in target.allowed_actions and action not in target.allowed_actions:
            raise WriteNotAuthorized(
                f"{by_action!r} is not authorized to write table {table!r}"
            )
        if table == "catalog" and isinstance(value, Listing):
            prior = self._store.read("catalog", key, caller="platform:catalog")
            value = _revisioned_listing(value, prior)
        self._store.write(table, key, value, by_action=by_action)

    def read_supply_state(self, sku_id: SkuId, *, caller: str) -> SupplyState:
        return self._store.read_supply_state(sku_id, caller=caller)

    def issue_supply_purchase_authorities(
        self,
        sku_ids: tuple[str, ...],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        ttl_ticks: int = DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    ) -> tuple[SupplyPurchaseAuthority, ...]:
        return self._store.issue_supply_purchase_authorities(
            sku_ids,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
            ttl_ticks=ttl_ticks,
        )

    def read_supply_purchase_authority(
        self, authority_id: str, *, caller: str
    ) -> SupplyPurchaseAuthority | None:
        return self._store.read_supply_purchase_authority(
            authority_id, caller=caller
        )

    def apply_settlement_reputation(
        self,
        *,
        merchant_id: AgentId,
        order_id: OrderId,
        txn_id: TxnId,
        by_actor: str,
        original_actor: str,
        source_request_id: str,
        idempotency_key: str,
    ) -> ReputationScore:
        return self._store.apply_settlement_reputation(
            merchant_id=merchant_id,
            order_id=order_id,
            txn_id=txn_id,
            by_actor=by_actor,
            original_actor=original_actor,
            source_request_id=source_request_id,
            idempotency_key=idempotency_key,
        )

    def create_search_session(
        self,
        *,
        session: SearchSession,
        by_actor: str,
        idempotency_key: str,
    ) -> SearchSession:
        return self._store.create_search_session(
            session=session,
            by_actor=by_actor,
            idempotency_key=idempotency_key,
        )

    def resolve_search_session(
        self,
        *,
        buyer_id: str,
        offer_id: str,
        caller: str,
        unique_only: bool = True,
        current_only: bool = True,
    ) -> SearchSession | None:
        return self._store.resolve_search_session(
            buyer_id=buyer_id,
            offer_id=offer_id,
            caller=caller,
            unique_only=unique_only,
            current_only=current_only,
        )

    def issue_match_certificate(
        self,
        *,
        acceptance: MatchAcceptance,
        by_actor: str,
        original_actor: str,
    ) -> MatchCertificate:
        return self._store.issue_match_certificate(
            acceptance=acceptance,
            by_actor=by_actor,
            original_actor=original_actor,
        )

    def resolve_match_certificate(
        self,
        *,
        buyer_id: str,
        order_id: str,
        caller: str,
        current_only: bool = True,
    ) -> MatchCertificate | None:
        return self._store.resolve_match_certificate(
            buyer_id=buyer_id,
            order_id=order_id,
            caller=caller,
            current_only=current_only,
        )

    def apply_supply_event(
        self,
        *,
        sku_id: SkuId,
        qty_delta: int,
        eta_day: int | None,
        unit_price_cents: int | None,
        expected_version: int | None,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> SupplyState:
        return self._store.apply_supply_event(
            sku_id=sku_id,
            qty_delta=qty_delta,
            eta_day=eta_day,
            unit_price_cents=unit_price_cents,
            expected_version=expected_version,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def settle_order(
        self,
        *,
        order: Order,
        receipt: Receipt,
        by_role: str,
        idempotency_key: str,
    ) -> Receipt:
        """Atomically settle an order in the durable store.

        ``by_role`` and ``idempotency_key`` retain the in-memory ``World``
        interface. Durable idempotency is derived from the persisted order and
        ledger, so it survives process restarts and retries with a fresh key.
        """
        return self._store.settle_order(
            order=order,
            receipt=receipt,
            by_role=by_role,
            idempotency_key=idempotency_key,
        )

    def refund_order(
        self,
        *,
        order: Order,
        refund_receipt: Receipt,
        by_role: str,
        idempotency_key: str,
    ) -> Receipt:
        """Atomically refund an order in the durable store."""
        return self._store.refund_order(
            order=order,
            refund_receipt=refund_receipt,
            by_role=by_role,
            idempotency_key=idempotency_key,
        )

    def settle_order_partial(
        self,
        *,
        order: Order,
        fulfilled_qty: int,
        receipt: Receipt | None,
        by_actor: str,
        idempotency_key: str,
    ) -> FulfillmentAllocation:
        """Atomically allocate a partial fill and its outstanding backorder."""
        return self._store.settle_order_partial(
            order=order,
            fulfilled_qty=fulfilled_qty,
            receipt=receipt,
            by_actor=by_actor,
            idempotency_key=idempotency_key,
        )

    def allocate_orders_atomic(
        self,
        *,
        allocation_id: str,
        merchant_id: AgentId,
        sku_id: SkuId,
        priority_order_ids: tuple[OrderId, ...],
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AllocationBatch:
        """Atomically allocate one scarce SKU across prioritized orders."""
        return self._store.allocate_orders_atomic(
            allocation_id=allocation_id,
            merchant_id=merchant_id,
            sku_id=sku_id,
            priority_order_ids=priority_order_ids,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def record_shipment_status(
        self,
        *,
        shipment_id: ShipmentId,
        event_id: str,
        status: ShipmentStatus,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        return self._store.record_shipment_status(
            shipment_id=shipment_id,
            event_id=event_id,
            status=status,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def resolve_shipment(
        self,
        *,
        shipment_id: ShipmentId,
        resolution: ShipmentResolution,
        replacement_sku_id: SkuId | None,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        return self._store.resolve_shipment(
            shipment_id=shipment_id,
            resolution=resolution,
            replacement_sku_id=replacement_sku_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def dispatch_order(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Move the owning merchant's paid order to ``DISPATCHED``."""
        return self._store.dispatch_order(order_id=order_id, by_actor=by_actor)

    def cancel_order(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Cancel a party's unpaid order."""
        return self._store.cancel_order(order_id=order_id, by_actor=by_actor)

    def mark_order_returned(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Record the owning merchant's receipt of a physical return."""
        return self._store.mark_order_returned(order_id=order_id, by_actor=by_actor)

    @property
    def logical_time(self) -> int:
        return self._store.logical_time

    def advance_logical_time(self, *, to_tick: int, by_actor: str) -> int:
        return self._store.advance_logical_time(to_tick=to_tick, by_actor=by_actor)

    def exchange_order(
        self,
        *,
        exchange_id: ExchangeId,
        original_order_id: OrderId,
        replacement_order: Order,
        by_actor: str,
        idempotency_key: str,
    ) -> Exchange:
        """Atomically create a like-for-like replacement without ledger writes."""
        return self._store.exchange_order(
            exchange_id=exchange_id,
            original_order_id=original_order_id,
            replacement_order=replacement_order,
            by_actor=by_actor,
            idempotency_key=idempotency_key,
        )

    def open_dispute(self, *, dispute: Dispute, by_actor: str) -> Dispute:
        """Open one party-bound dispute for a paid order."""
        return self._store.open_dispute(dispute=dispute, by_actor=by_actor)

    def rule_dispute(self, *, ruling: Ruling, by_actor: str) -> Ruling:
        """Atomically record an adjudicator ruling."""
        return self._store.rule_dispute(ruling=ruling, by_actor=by_actor)

    def apply(self, initial_state: dict[str, Any]) -> None:
        revision_seed = initial_state.get("order_state_revisions")
        for table_name, rows in initial_state.items():
            if table_name == "order_state_revisions":
                continue
            if table_name == "logical_time":
                self._store.seed_logical_time(rows)
                continue
            self._table(table_name)
            self._store.clear_table(table_name)
            if rows is None:
                continue
            if isinstance(rows, dict):
                for key, value in rows.items():
                    if table_name in _AUTHORITY_TABLES:
                        self._store.write(
                            table_name,
                            key,
                            value,
                            by_action=_SEED_ACTIONS[table_name],
                        )
                    else:
                        self.write(
                            table_name,
                            key,
                            value,
                            by_action=_SEED_ACTIONS[table_name],
                        )
                continue
            for row in rows:
                if table_name in _AUTHORITY_TABLES:
                    self._store.write(
                        table_name,
                        _row_key(table_name, row),
                        row,
                        by_action=_SEED_ACTIONS[table_name],
                    )
                else:
                    self.write(
                        table_name,
                        _row_key(table_name, row),
                        row,
                        by_action=_SEED_ACTIONS[table_name],
                    )
        if revision_seed is not None:
            self._store.seed_order_state_revisions(revision_seed)
        self._store.validate_persisted_protocol_records()
        self._store.validate_persisted_negotiations()
        self._store.validate_persisted_evidence_contracts()

    def reset(self, mode: Literal["E0", "E1"]) -> None:
        self._store.reset(mode)
        self._mode = mode

    def snapshot(self) -> WorldSnapshot:
        return self._store.snapshot()

    def search_catalog(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        *,
        limit: int | None = None,
    ) -> list[Listing]:
        return self._store.search_catalog(query, filters, limit=limit)

    def iter_table(self, table: str) -> Iterator[tuple[Any, Any]]:
        self._table(table)
        yield from self._store.iter_table(table)

    def close(self) -> None:
        self._store.close()

    def _table(self, table: str) -> Table[Any, Any]:
        try:
            return self._tables[table]
        except KeyError as exc:
            raise TableNotFound(f"unknown world table: {table}") from exc


class SQLiteWorldStore:
    """Durable current-state tables plus a minimal mutation log."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = RLock()
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()
        try:
            self.validate_persisted_governance()
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def publish_governance_policy(
        self,
        policy_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "publish_governance_policy",
            policy_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_governance_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "apply_governance_intent",
            intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def aggregate_reviews(
        self,
        sku_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "aggregate_reviews",
            sku_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def ingest_review_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "ingest_review_observation",
            record_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def ingest_market_observation(
        self,
        record_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "ingest_market_observation",
            record_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def resolve_governance_case(
        self,
        decision_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "resolve_governance_case",
            decision_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def apply_governance_reputation(
        self,
        source_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "apply_governance_reputation",
            source_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def create_remediation_plan(
        self,
        plan_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "create_remediation_plan",
            plan_intent,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def verify_remediation_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "verify_remediation_step",
            plan_id,
            step_id,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def persist_ranking_context(
        self,
        ranking_result: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Any:
        return self._run_governance_method(
            "persist_ranking_context",
            ranking_result,
            by_actor=by_actor,
            original_actor=original_actor,
            idempotency_key=idempotency_key,
        )

    def governance_history(
        self, record_kind: str, stable_id: str, *, caller: str
    ) -> tuple[Any, ...]:
        with self._lock:
            shadow = self._governance_shadow()
            return shadow.governance_history(record_kind, stable_id, caller=caller)

    def ranking_context_projection(
        self, context_id: str, *, caller: str
    ) -> dict[str, Any]:
        with self._lock:
            shadow = self._governance_shadow()
            return shadow.ranking_context_projection(context_id, caller=caller)

    def _governance_shadow(self) -> World:
        snapshot = self.snapshot()
        initial = {
            field.name: getattr(snapshot, field.name) for field in fields(snapshot)
        }
        shadow = World(mode="E1")
        shadow.apply(initial)
        return shadow

    def _run_governance_method(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the shared World authority engine inside one SQLite tx.

        The shadow is a deterministic transaction planner hydrated from the
        locked durable snapshot.  It is not a second truth source.  Only its
        single validated World commit is applied to SQLite before COMMIT.
        """

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                shadow = self._governance_shadow()
                result = getattr(shadow, method_name)(*args, **kwargs)
                commits = shadow.commit_journal
                if not commits:
                    self._conn.commit()
                    return result
                if len(commits) != 1:
                    raise WorldError(
                        "one governance command must produce exactly one World commit"
                    )
                commit = commits[0]
                for write in commit.table_writes:
                    if write.table == "logical_time":
                        if self._clock_value() != write.before:
                            raise WorldError("governance logical time changed before commit")
                        _record_mutation(
                            self._conn,
                            table="logical_time",
                            key="world",
                            action=commit.authority_action,
                            before=write.before,
                            after=write.after,
                        )
                        self._set_clock(cast(int, write.after))
                        continue
                    current = self._fetch_value(write.table, write.key)
                    if current != write.before:
                        raise WorldError(
                            f"{write.table}:{write.key} changed before governance commit"
                        )
                    _upsert(self._conn, write.table, write.after)
                    _record_mutation(
                        self._conn,
                        table=write.table,
                        key=write.key,
                        action=commit.authority_action,
                        before=write.before,
                        after=write.after,
                    )
                self._append_world_commit(
                    commit_kind=commit.commit_kind,
                    operation=commit.operation,
                    authority_action=commit.authority_action,
                    actor_id=commit.actor_id,
                    idempotency_key=commit.idempotency_key,
                    subject_id=commit.subject_id,
                    table_writes=commit.table_writes,
                    invariants_held=commit.invariants_held,
                    request_fingerprint=commit.request_fingerprint,
                )
                self._conn.commit()
                return result
            except BaseException:
                self._conn.rollback()
                raise

    @property
    def logical_time(self) -> int:
        with self._lock:
            return self._clock_value()

    def seed_logical_time(self, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LogicalTimeError("seed logical_time must be a non-negative integer")
        with self._lock, self._conn:
            self._set_clock(value)

    def advance_logical_time(self, *, to_tick: int, by_actor: str) -> int:
        if by_actor != "runtime:clock":
            raise LogicalTimeError("only runtime:clock may advance World logical time")
        if isinstance(to_tick, bool) or not isinstance(to_tick, int) or to_tick < 0:
            raise LogicalTimeError("logical time must be a non-negative integer")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                current = self._clock_value()
                if to_tick < current:
                    raise LogicalTimeError(
                        f"logical time cannot regress from {current} to {to_tick}"
                    )
                if to_tick != current:
                    _record_mutation(
                        self._conn,
                        table="logical_time",
                        key="world",
                        action="world.advance_clock",
                        before=current,
                        after=to_tick,
                    )
                    self._set_clock(to_tick)
                self._conn.commit()
                return to_tick
            except BaseException:
                self._conn.rollback()
                raise

    def checkout_cart(
        self,
        *,
        quote_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> OrderGroup:
        """Atomically settle a persisted quote using only its World identifier."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may checkout a persistent cart quote"
            )
        if not isinstance(quote_id, str) or not quote_id.strip():
            raise OrderNotSettleable("cart checkout quote_id must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict("cart checkout idempotency key must be non-empty")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                quote = cast(
                    "PersistentCartQuote | None",
                    self._fetch_value("persistent_cart_quotes", quote_id),
                )
                if quote is None:
                    raise OrderNotSettleable(
                        "cart checkout references an unknown quote"
                    )
                validate_persistent_cart_quote(quote)
                if original_actor != quote.buyer_id:
                    raise LifecycleAuthorizationError(
                        "cart checkout must be authenticated by the quoted buyer"
                    )
                fingerprint = canonical_digest({"quote_id": quote_id})
                scope = f"cart-checkout:{quote.market_id}"
                replay = self._authority_operation_replay(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, OrderGroup):
                        raise WorldError(
                            "cart checkout idempotency outcome is missing"
                        )
                    self._conn.commit()
                    return outcome
                group_id = OrderGroupId(f"group:{quote.quote_id}")
                if self._fetch_value("order_groups", str(group_id)) is not None:
                    raise IdempotencyConflict(
                        "persistent cart quote was already checked out under "
                        "another key"
                    )
                if any(
                    line.backorder_qty
                    or line.unfilled_qty
                    or not line.fulfill_now_qty
                    for line in quote.lines
                ):
                    raise OrderNotSettleable(
                        "checkout currently requires every persisted quote line "
                        "to be fully available"
                    )

                authority = cast(
                    "MandateRevisionAuthority | None",
                    self._fetch_value("mandate_authorities", quote.mandate_id),
                )
                revision_rows = self._conn.execute(
                    "SELECT * FROM mandate_revisions WHERE mandate_id = ? "
                    "ORDER BY revision",
                    (quote.mandate_id,),
                ).fetchall()
                revisions = tuple(
                    cast(
                        MandateRevision,
                        _row_to_value("mandate_revisions", row),
                    )
                    for row in revision_rows
                )
                if authority is None or not revisions:
                    raise WriteNotAuthorized(
                        "cart checkout mandate authority is missing"
                    )
                current_mandate = revisions[-1]
                before_tick = self._clock_value()
                event_tick = before_tick + 1
                assert_cart_quote_usable(
                    quote,
                    buyer_id=authority.buyer_id,
                    principal_id=authority.principal_id,
                    mandate_id=authority.mandate_id,
                    mandate_revision=current_mandate.revision,
                    mandate_digest=current_mandate.revision_digest,
                    logical_tick=event_tick,
                )
                if quote.grand_total_minor > _mandate_budget_minor(revisions):
                    raise WriteNotAuthorized(
                        "cart checkout exceeds the current principal mandate budget"
                    )

                policy_rows = self._conn.execute(
                    "SELECT * FROM pricing_policy_revisions ORDER BY revision_key"
                ).fetchall()
                all_policies = tuple(
                    cast(
                        PricingPolicyRevision,
                        _row_to_value("pricing_policy_revisions", row),
                    )
                    for row in policy_rows
                )
                listings: dict[str, Listing] = {}
                inventory: dict[str, InventoryRow] = {}
                selected_policies: dict[str, PricingPolicyRevision] = {}
                for line in quote.lines:
                    listing = cast(
                        "Listing | None",
                        self._fetch_value("catalog", line.sku_id),
                    )
                    stock = self._fetch_value("inventory", line.sku_id)
                    if listing is None or not isinstance(stock, InventoryRow):
                        raise CartQuoteStaleError(
                            f"authoritative cart source disappeared for "
                            f"{line.sku_id!r}"
                        )
                    listings[line.sku_id] = listing
                    inventory[line.sku_id] = stock
                    selected_policies[line.sku_id] = (
                        _active_pricing_policy_for_listing(
                            all_policies,
                            market_id=quote.market_id,
                            listing=listing,
                            logical_tick=event_tick,
                        )
                    )
                assert_cart_quote_snapshots_current(
                    quote,
                    catalog_revisions={
                        sku_id: _catalog_revision(listing)
                        for sku_id, listing in listings.items()
                    },
                    catalog_digests={
                        sku_id: catalog_snapshot_digest(listing)
                        for sku_id, listing in listings.items()
                    },
                    inventory_revisions={
                        sku_id: stock.version
                        for sku_id, stock in inventory.items()
                    },
                    inventory_digests={
                        sku_id: inventory_snapshot_digest(stock)
                        for sku_id, stock in inventory.items()
                    },
                    merchant_by_sku={
                        sku_id: str(listing.merchant_id)
                        for sku_id, listing in listings.items()
                    },
                )
                policies = tuple(
                    {
                        policy.policy_digest: policy
                        for policy in selected_policies.values()
                    }.values()
                )
                if (
                    pricing_policy_set_digest(policies)
                    != quote.pricing_policy_digest
                    or max(policy.revision for policy in policies)
                    != quote.pricing_policy_revision
                ):
                    raise CartQuoteStaleError(
                        "cart pricing policy snapshot changed"
                    )

                orders: list[Order] = []
                receipts: list[Receipt] = []
                timelines: list[OrderTimeline] = []
                reserved: dict[str, InventoryRow] = {}
                for index, line in enumerate(quote.lines, start=1):
                    stock = inventory[line.sku_id]
                    if _available_qty(stock) < line.fulfill_now_qty:
                        raise OutOfStock(
                            f"insufficient inventory for quoted sku "
                            f"{line.sku_id!r}"
                        )
                    order_id = OrderId(
                        f"order:{quote.quote_id}:{index:03d}"
                    )
                    txn_id = TxnId(f"txn:{quote.quote_id}:{index:03d}")
                    if (
                        self._fetch_value("orders", str(order_id)) is not None
                        or self._fetch_value("ledger", str(txn_id)) is not None
                    ):
                        raise IdempotencyConflict(
                            f"cart checkout durable row collision for {order_id}"
                        )
                    unit_price = _money_from_minor(
                        line.unit_price_minor,
                        quote.currency,
                    )
                    order = Order(
                        order_id=order_id,
                        buyer_id=AgentId(quote.buyer_id),
                        merchant_id=AgentId(line.merchant_id),
                        sku_id=SkuId(line.sku_id),
                        qty=line.fulfill_now_qty,
                        agreed_price=unit_price,
                        state=OrderState.SETTLED,
                    )
                    receipt = Receipt(
                        txn_id=txn_id,
                        ts=f"world-tick:{event_tick}",
                        order_id=order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        sku_id=order.sku_id,
                        qty=order.qty,
                        price=unit_price,
                        idempotency_key=idempotency_key,
                    )
                    timeline = OrderTimeline(
                        order_id=order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        settled_at_tick=event_tick,
                        return_window_ticks=_captured_return_window(
                            listings[line.sku_id]
                        ),
                    )
                    orders.append(order)
                    receipts.append(receipt)
                    timelines.append(timeline)
                    reserved[line.sku_id] = _reserve_inventory(
                        stock, qty=line.fulfill_now_qty
                    )

                fees = tuple(
                    FeeComponent(
                        fee_id=charge.charge_id,
                        kind=charge.kind,
                        scope=charge.scope,
                        amount=Money(
                            Decimal(charge.amount_minor) / Decimal(100),
                            quote.currency,
                        ),
                        basis=charge.basis,
                    )
                    for charge in quote.charges
                )
                group = OrderGroup(
                    order_group_id=group_id,
                    quote_id=QuoteId(quote.quote_id),
                    buyer_id=AgentId(quote.buyer_id),
                    merchant_ids=tuple(
                        AgentId(value)
                        for value in sorted(
                            {line.merchant_id for line in quote.lines}
                        )
                    ),
                    order_ids=tuple(order.order_id for order in orders),
                    txn_ids=tuple(receipt.txn_id for receipt in receipts),
                    subtotal=Money(
                        Decimal(quote.subtotal_minor) / Decimal(100),
                        quote.currency,
                    ),
                    fee_breakdown=fees,
                    grand_total=Money(
                        Decimal(quote.grand_total_minor) / Decimal(100),
                        quote.currency,
                    ),
                    quote_hash=quote.quote_digest,
                    idempotency_key=idempotency_key,
                )
                _validate_group_payment_record(group, receipts, quote)

                for order, receipt, timeline in zip(
                    orders, receipts, timelines, strict=True
                ):
                    _upsert(self._conn, "orders", order)
                    _record_mutation(
                        self._conn,
                        table="orders",
                        key=str(order.order_id),
                        action="world.checkout_cart_quote",
                        before=None,
                        after=order,
                    )
                    _upsert(
                        self._conn,
                        "inventory",
                        reserved[str(order.sku_id)],
                    )
                    _record_mutation(
                        self._conn,
                        table="inventory",
                        key=str(order.sku_id),
                        action="world.checkout_cart_quote",
                        before=inventory[str(order.sku_id)],
                        after=reserved[str(order.sku_id)],
                    )
                    _upsert(self._conn, "ledger", receipt)
                    _record_mutation(
                        self._conn,
                        table="ledger",
                        key=str(receipt.txn_id),
                        action="world.checkout_cart_quote",
                        before=None,
                        after=receipt,
                    )
                    _upsert(self._conn, "order_timelines", timeline)
                    _record_mutation(
                        self._conn,
                        table="order_timelines",
                        key=str(order.order_id),
                        action="world.checkout_cart_quote",
                        before=None,
                        after=timeline,
                    )
                _upsert(self._conn, "order_groups", group)
                _record_mutation(
                    self._conn,
                    table="order_groups",
                    key=str(group_id),
                    action="world.checkout_cart_quote",
                    before=None,
                    after=group,
                )
                operation = self._record_authority_operation(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="order_groups",
                    outcome_key=str(group_id),
                    authority_action="world.checkout_cart_quote",
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.checkout_cart_quote",
                    before=before_tick,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                commit_writes: list[TableWrite] = []
                for order, receipt, timeline in zip(
                    orders, receipts, timelines, strict=True
                ):
                    commit_writes.extend(
                        (
                            TableWrite(
                                table="orders",
                                key=str(order.order_id),
                                op="create",
                                before=None,
                                after=order,
                            ),
                            TableWrite(
                                table="inventory",
                                key=str(order.sku_id),
                                op="update",
                                before=inventory[str(order.sku_id)],
                                after=reserved[str(order.sku_id)],
                            ),
                            TableWrite(
                                table="ledger",
                                key=str(receipt.txn_id),
                                op="create",
                                before=None,
                                after=receipt,
                            ),
                            TableWrite(
                                table="order_timelines",
                                key=str(order.order_id),
                                op="create",
                                before=None,
                                after=timeline,
                            ),
                        )
                    )
                commit_writes.extend(
                    (
                        TableWrite(
                            table="order_groups",
                            key=str(group_id),
                            op="create",
                            before=None,
                            after=group,
                        ),
                        TableWrite(
                            table="authority_operations",
                            key=operation.operation_key,
                            op="create",
                            before=None,
                            after=operation,
                        ),
                        TableWrite(
                            table="logical_time",
                            key="world",
                            op="update",
                            before=before_tick,
                            after=event_tick,
                        ),
                    )
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="checkout_cart_quote",
                    authority_action="world.checkout_cart_quote",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    subject_id=str(group_id),
                    table_writes=tuple(commit_writes),
                    invariants_held=(
                        "world-loaded-quote",
                        "atomic-cross-merchant",
                        "mandate-budget-current",
                        "catalog-inventory-policy-fresh",
                        "inventory-no-oversell",
                        "line-receipts-plus-group-charges-equal-grand-total",
                        "quote-digest-bound-group-payment",
                        "actor-scoped-idempotency",
                    ),
                    request_fingerprint=fingerprint,
                )
                self._conn.commit()
                return group
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise IdempotencyConflict(
                    "conflicting concurrent cart checkout"
                ) from exc
            except BaseException:
                self._conn.rollback()
                raise

    def read_negotiation_event(
        self, event_id: str, *, caller: str
    ) -> NegotiationEvent | None:
        value = self.read("negotiation_events", event_id, caller=caller)
        return cast("NegotiationEvent | None", value)

    def read_negotiation_thread(
        self, negotiation_id: str, *, caller: str
    ) -> NegotiationThread | None:
        value = self.read("negotiation_threads", negotiation_id, caller=caller)
        return cast("NegotiationThread | None", value)

    def negotiation_events_for_thread(
        self, negotiation_id: str, *, caller: str
    ) -> tuple[NegotiationEvent, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM negotiation_events "
                "WHERE negotiation_id = ? ORDER BY sequence_no, event_id",
                (negotiation_id,),
            ).fetchall()
            events = tuple(
                cast(
                    NegotiationEvent,
                    _row_to_value("negotiation_events", row),
                )
                for row in rows
            )
            return tuple(
                event
                for event in events
                if _visible("negotiation_events", event, caller)
            )

    def apply_negotiation_intent(
        self,
        intent: NegotiationIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        max_rounds: int,
        deadline_ticks: int,
    ) -> NegotiationEvent:
        """Atomically append an event and replace its materialized thread."""

        if by_actor != "platform:negotiation":
            raise WriteNotAuthorized(
                "only platform:negotiation may apply negotiation intents"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict(
                "negotiation idempotency key must be a non-empty string"
            )
        normalized = normalize_negotiation_intent(intent)
        fingerprint = negotiation_intent_fingerprint(
            normalized,
            max_rounds=max_rounds,
            deadline_ticks=deadline_ticks,
        )
        negotiation_id = normalized["negotiation_id"]
        operation_scope = f"negotiation-event:{negotiation_id}"
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope=operation_scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    event = self._fetch_value(replay[0], replay[1])
                    if not isinstance(event, NegotiationEvent):
                        raise WorldError(
                            "negotiation idempotency outcome is missing"
                        )
                    self._conn.commit()
                    return event

                before_thread = cast(
                    "NegotiationThread | None",
                    self._fetch_value(
                        "negotiation_threads", negotiation_id
                    ),
                )
                action_kind = normalized["action_kind"]
                before_tick = self._clock_value()
                event_tick = before_tick + 1
                event_id = negotiation_event_id(
                    negotiation_id, original_actor, idempotency_key
                )
                if action_kind == PROPOSE_OFFER:
                    if before_thread is not None:
                        raise NegotiationSchemaError(
                            "negotiation already exists"
                        )
                    buyer_id, merchant_id = _negotiation_parties(
                        original_actor, normalized["counterparty_id"]
                    )
                    listing = cast(
                        "Listing | None",
                        self._fetch_value("catalog", normalized["sku_id"]),
                    )
                    if listing is None:
                        raise NegotiationSchemaError(
                            "negotiation listing does not exist"
                        )
                    if str(listing.merchant_id) != merchant_id:
                        raise WriteNotAuthorized(
                            "negotiation merchant does not own the listing"
                        )
                    binding = NegotiationBinding(
                        negotiation_id=negotiation_id,
                        buyer_id=buyer_id,
                        merchant_id=merchant_id,
                        offer_id=normalized["offer_id"],
                        sku_id=normalized["sku_id"],
                        listing_digest=negotiation_listing_digest(listing),
                        listing_revision=_catalog_revision(listing),
                        currency=listing.list_price.currency,
                        qty=int(normalized.get("qty", 1)),
                        max_rounds=max_rounds,
                        opened_at_tick=event_tick,
                        expires_at_tick=event_tick + deadline_ticks,
                    )
                    event = build_negotiation_event(
                        binding,
                        event_id=event_id,
                        action_kind=action_kind,
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        unit_price=int(normalized["unit_price"]),
                        round_no=int(normalized.get("round_no", 1)),
                        sequence_no=1,
                        previous_digest=None,
                        server_tick=event_tick,
                    )
                else:
                    if before_thread is None:
                        raise NegotiationSchemaError(
                            "negotiation does not exist"
                        )
                    if original_actor not in {
                        before_thread.buyer_id,
                        before_thread.merchant_id,
                    }:
                        raise WriteNotAuthorized(
                            "actor is not a negotiation participant"
                        )
                    expected_peer = (
                        before_thread.merchant_id
                        if original_actor == before_thread.buyer_id
                        else before_thread.buyer_id
                    )
                    if normalized["counterparty_id"] != expected_peer:
                        raise NegotiationSchemaError(
                            "negotiation counterparty mismatch"
                        )
                    if (
                        normalized["offer_id"] != before_thread.offer_id
                        or normalized["sku_id"] != before_thread.sku_id
                    ):
                        raise NegotiationSchemaError(
                            "negotiation lineage mismatch"
                        )
                    if max_rounds != before_thread.max_rounds or (
                        deadline_ticks
                        != before_thread.expires_at_tick
                        - before_thread.opened_at_tick
                    ):
                        raise NegotiationSchemaError(
                            "negotiation policy configuration changed mid-thread"
                        )
                    event = build_next_negotiation_event(
                        before_thread,
                        event_id=event_id,
                        action_kind=action_kind,
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        server_tick=event_tick,
                        unit_price=normalized.get("unit_price"),
                        round_no=normalized.get("round_no"),
                    )
                    binding = NegotiationBinding(
                        negotiation_id=before_thread.negotiation_id,
                        buyer_id=before_thread.buyer_id,
                        merchant_id=before_thread.merchant_id,
                        offer_id=before_thread.offer_id,
                        sku_id=before_thread.sku_id,
                        listing_digest=before_thread.listing_digest,
                        listing_revision=before_thread.listing_revision,
                        currency=before_thread.currency,
                        qty=before_thread.qty,
                        max_rounds=before_thread.max_rounds,
                        opened_at_tick=before_thread.opened_at_tick,
                        expires_at_tick=before_thread.expires_at_tick,
                    )
                transition = apply_negotiation_event(
                    before_thread,
                    event,
                    binding=binding,
                    server_tick=event_tick,
                )
                _upsert(self._conn, "negotiation_events", event)
                _record_mutation(
                    self._conn,
                    table="negotiation_events",
                    key=event.event_id,
                    action="world.apply_negotiation_intent",
                    before=None,
                    after=event,
                )
                _upsert(self._conn, "negotiation_threads", transition.thread)
                _record_mutation(
                    self._conn,
                    table="negotiation_threads",
                    key=negotiation_id,
                    action="world.apply_negotiation_intent",
                    before=before_thread,
                    after=transition.thread,
                )
                self._record_authority_operation(
                    scope=operation_scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="negotiation_events",
                    outcome_key=event.event_id,
                    authority_action="world.apply_negotiation_intent",
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.apply_negotiation_intent",
                    before=before_tick,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._conn.commit()
                return event
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise IdempotencyConflict(
                    "conflicting concurrent negotiation event"
                ) from exc
            except BaseException:
                self._conn.rollback()
                raise

    def publish_pricing_policy(
        self,
        intent: PricingPolicyIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PricingPolicyRevision:
        """Atomically persist one World-derived pricing-policy revision."""

        if by_actor != "platform:pricing":
            raise WriteNotAuthorized(
                "only platform:pricing may publish pricing policies"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict(
                "pricing policy idempotency key must not be blank"
            )
        normalized = normalize_pricing_policy_intent(intent)
        fingerprint = pricing_policy_intent_fingerprint(normalized)
        market_id = normalized["market_id"]
        scope = f"pricing-policy:{market_id}"
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PricingPolicyRevision):
                        raise WorldError(
                            "pricing policy idempotency outcome is missing"
                        )
                    self._conn.commit()
                    return outcome

                merchant_id = catalog_owner_for_actor(original_actor)
                currency = self._pricing_policy_scope_currency(
                    merchant_id=merchant_id,
                    listing_ids=normalized["listing_ids"],
                    product_ids=normalized["product_ids"],
                )
                rows = self._conn.execute(
                    "SELECT * FROM pricing_policy_revisions "
                    "WHERE market_id = ? AND merchant_id = ? AND policy_id = ? "
                    "ORDER BY revision",
                    (market_id, merchant_id, normalized["policy_id"]),
                ).fetchall()
                stream = tuple(
                    cast(
                        PricingPolicyRevision,
                        _row_to_value("pricing_policy_revisions", row),
                    )
                    for row in rows
                )
                previous = stream[-1] if stream else None
                revision_number = 1 if previous is None else previous.revision + 1
                predecessor = (
                    GENESIS_PRICING_POLICY_DIGEST
                    if previous is None
                    else previous.policy_digest
                )
                before_tick = self._clock_value()
                event_tick = before_tick + 1
                expires_after = normalized["expires_after_ticks"]
                revision = build_pricing_policy_revision(
                    market_id=market_id,
                    merchant_id=merchant_id,
                    owner_id=merchant_id,
                    policy_id=normalized["policy_id"],
                    revision=revision_number,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    currency=currency,
                    listing_ids=normalized["listing_ids"],
                    product_ids=normalized["product_ids"],
                    quantity_tiers=normalized["quantity_tiers"],
                    bundle_discounts=normalized["bundle_discounts"],
                    bundle_stacking=normalized["bundle_stacking"],
                    components=normalized["components"],
                    published_at_tick=event_tick,
                    effective_from_tick=(
                        event_tick + normalized["effective_after_ticks"]
                    ),
                    expires_at_tick=(
                        None
                        if expires_after is None
                        else event_tick + expires_after
                    ),
                    predecessor_digest=predecessor,
                )
                all_rows = self._conn.execute(
                    "SELECT * FROM pricing_policy_revisions "
                    "ORDER BY revision_key"
                ).fetchall()
                disposition, _ = apply_pricing_policy_revision(
                    (
                        cast(
                            PricingPolicyRevision,
                            _row_to_value("pricing_policy_revisions", row),
                        )
                        for row in all_rows
                    ),
                    revision,
                    trusted_owner_id=merchant_id,
                    server_tick=event_tick,
                )
                if disposition != "append":
                    raise WorldError(
                        "new pricing policy unexpectedly classified as retry"
                    )
                key = pricing_policy_revision_key(
                    market_id,
                    merchant_id,
                    revision.policy_id,
                    revision.revision,
                )
                _upsert(self._conn, "pricing_policy_revisions", revision)
                _record_mutation(
                    self._conn,
                    table="pricing_policy_revisions",
                    key=key,
                    action="world.publish_pricing_policy",
                    before=None,
                    after=revision,
                )
                self._record_authority_operation(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="pricing_policy_revisions",
                    outcome_key=key,
                    authority_action="world.publish_pricing_policy",
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.publish_pricing_policy",
                    before=before_tick,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._conn.commit()
                return revision
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise IdempotencyConflict(
                    "conflicting concurrent pricing policy revision"
                ) from exc
            except BaseException:
                self._conn.rollback()
                raise

    def pricing_policy_revisions(
        self,
        market_id: str,
        merchant_id: str,
        policy_id: str,
        *,
        caller: str,
    ) -> tuple[PricingPolicyRevision, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pricing_policy_revisions "
                "WHERE market_id = ? AND merchant_id = ? AND policy_id = ? "
                "ORDER BY revision",
                (market_id, merchant_id, policy_id),
            ).fetchall()
            revisions = tuple(
                cast(
                    PricingPolicyRevision,
                    _row_to_value("pricing_policy_revisions", row),
                )
                for row in rows
            )
            return tuple(
                row
                for row in revisions
                if _visible("pricing_policy_revisions", row, caller)
            )

    def _pricing_policy_scope_currency(
        self,
        *,
        merchant_id: str,
        listing_ids: tuple[str, ...],
        product_ids: tuple[str, ...],
    ) -> str:
        scoped: dict[str, Listing] = {}
        for sku_id in listing_ids:
            listing = cast(
                "Listing | None", self._fetch_value("catalog", sku_id)
            )
            if listing is None:
                raise WriteNotAuthorized(
                    f"pricing policy references unknown listing {sku_id!r}"
                )
            if str(listing.merchant_id) != merchant_id:
                raise WriteNotAuthorized(
                    f"pricing policy actor does not own listing {sku_id!r}"
                )
            scoped[sku_id] = listing
        for product_id in product_ids:
            rows = self._conn.execute(
                "SELECT * FROM catalog WHERE product_id = ? AND merchant_id = ? "
                "ORDER BY sku_id",
                (product_id, merchant_id),
            ).fetchall()
            if not rows:
                raise WriteNotAuthorized(
                    f"pricing policy actor owns no listing for product {product_id!r}"
                )
            scoped.update(
                (
                    str(row["sku_id"]),
                    cast(Listing, _row_to_value("catalog", row)),
                )
                for row in rows
            )
        currencies = {row.list_price.currency for row in scoped.values()}
        if len(currencies) != 1:
            raise WorldError("pricing policy scope must use exactly one currency")
        return next(iter(currencies))

    def create_cart_quote_request(
        self,
        intent: CartQuoteIntent | Mapping[str, Any],
        *,
        market_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        request_ttl_ticks: int = 10,
    ) -> PersistentCartQuoteRequest:
        """Persist a World-derived merchant quote authorization atomically."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may create a cart quote request"
            )
        if not isinstance(market_id, str) or not market_id.strip():
            raise WorldError("cart quote request market_id must be non-empty")
        if (
            isinstance(request_ttl_ticks, bool)
            or not isinstance(request_ttl_ticks, int)
            or request_ttl_ticks <= 0
        ):
            raise WorldError("cart quote request ttl must be a positive integer")
        normalized = normalize_cart_quote_intent(intent)
        fingerprint = canonical_digest(
            {
                "intent": cart_quote_intent_fingerprint(
                    normalized,
                    market_id=market_id,
                    quote_ttl_ticks=request_ttl_ticks,
                ),
                "request_ttl_ticks": request_ttl_ticks,
            }
        )
        scope = f"cart-quote-request:{market_id}"
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PersistentCartQuoteRequest):
                        raise WorldError(
                            "cart quote request idempotency outcome is missing"
                        )
                    self._conn.commit()
                    return outcome
                authority = cast(
                    "MandateRevisionAuthority | None",
                    self._fetch_value(
                        "mandate_authorities", normalized["mandate_id"]
                    ),
                )
                if authority is None:
                    raise WriteNotAuthorized("cart quote request mandate is missing")
                if original_actor not in {
                    authority.buyer_id,
                    authority.principal_id,
                }:
                    raise WriteNotAuthorized(
                        "cart quote request actor is not the mandate buyer or principal"
                    )
                revisions = tuple(
                    cast(
                        MandateRevision,
                        _row_to_value("mandate_revisions", row),
                    )
                    for row in self._conn.execute(
                        "SELECT * FROM mandate_revisions WHERE mandate_id = ? "
                        "ORDER BY revision",
                        (normalized["mandate_id"],),
                    ).fetchall()
                )
                if not revisions:
                    raise WriteNotAuthorized(
                        "cart quote request mandate has no revision"
                    )
                mandate = revisions[-1]
                if (
                    mandate.buyer_id != authority.buyer_id
                    or mandate.principal_id != authority.principal_id
                ):
                    raise WriteNotAuthorized(
                        "cart quote request mandate authority is inconsistent"
                    )
                lines: list[CartQuoteRequestLine] = []
                currencies: set[str] = set()
                for line in normalized["lines"]:
                    listing = cast(
                        "Listing | None",
                        self._fetch_value("catalog", line["sku_id"]),
                    )
                    if listing is None:
                        raise WorldError(
                            f"cart quote request listing {line['sku_id']!r} is missing"
                        )
                    currencies.add(listing.list_price.currency)
                    lines.append(
                        CartQuoteRequestLine(
                            sku_id=str(listing.sku_id),
                            product_id=listing.product_id,
                            merchant_id=str(listing.merchant_id),
                            qty=line["qty"],
                            catalog_revision=_catalog_revision(listing),
                            catalog_digest=catalog_snapshot_digest(listing),
                        )
                    )
                if len(currencies) != 1:
                    raise WorldError("cart quote request cannot mix currencies")
                if len({line.merchant_id for line in lines}) != 1:
                    raise WorldError(
                        "merchant quote authorization requires one merchant; "
                        "buyers may request a direct cross-merchant quote"
                    )
                before_tick = self._clock_value()
                event_tick = before_tick + 1
                request_id = "cart-request:" + canonical_digest(
                    {
                        "market_id": market_id,
                        "created_by": original_actor,
                        "idempotency_key": idempotency_key,
                        "fingerprint": fingerprint,
                    }
                )[:32]
                request = build_persistent_cart_quote_request(
                    request_id=request_id,
                    market_id=market_id,
                    buyer_id=authority.buyer_id,
                    principal_id=authority.principal_id,
                    created_by=original_actor,
                    issuer_id="world",
                    mandate_id=authority.mandate_id,
                    mandate_revision=mandate.revision,
                    mandate_digest=mandate.revision_digest,
                    idempotency_key=idempotency_key,
                    fill_policy=normalized["fill_policy"],
                    backorder_policy=normalized["backorder_policy"],
                    currency=next(iter(currencies)),
                    lines=lines,
                    issued_at_tick=event_tick,
                    expires_at_tick=event_tick + request_ttl_ticks,
                )
                _upsert(self._conn, "persistent_cart_quote_requests", request)
                _record_mutation(
                    self._conn,
                    table="persistent_cart_quote_requests",
                    key=request.request_id,
                    action="world.create_cart_quote_request",
                    before=None,
                    after=request,
                )
                operation = self._record_authority_operation(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="persistent_cart_quote_requests",
                    outcome_key=request.request_id,
                    authority_action="world.create_cart_quote_request",
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.create_cart_quote_request",
                    before=before_tick,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="create_cart_quote_request",
                    authority_action="world.create_cart_quote_request",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    subject_id=request.request_id,
                    table_writes=(
                        TableWrite(
                            table="persistent_cart_quote_requests",
                            key=request.request_id,
                            op="create",
                            before=None,
                            after=request,
                        ),
                        TableWrite(
                            table="authority_operations",
                            key=operation.operation_key,
                            op="create",
                            before=None,
                            after=operation,
                        ),
                        TableWrite(
                            table="logical_time",
                            key="world",
                            op="update",
                            before=before_tick,
                            after=event_tick,
                        ),
                    ),
                    invariants_held=(
                        "buyer-or-principal-authorized",
                        "mandate-revision-bound",
                        "catalog-owner-derived",
                        "budget-values-excluded",
                        "actor-scoped-idempotency",
                    ),
                    request_fingerprint=fingerprint,
                )
                self._conn.commit()
                return request
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise IdempotencyConflict(
                    "conflicting concurrent cart quote request authorization"
                ) from exc
            except BaseException:
                self._conn.rollback()
                raise

    def issue_cart_quote_from_request(
        self,
        request_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        """Issue a merchant quote from a persisted opaque request."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may issue a requested cart quote"
            )
        if original_actor.split(":", 1)[0] != "merchant":
            raise LifecycleAuthorizationError(
                "requested cart quote actor must be a fully qualified merchant"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                request = cast(
                    "PersistentCartQuoteRequest | None",
                    self._fetch_value(
                        "persistent_cart_quote_requests", request_id
                    ),
                )
                if request is None:
                    raise WriteNotAuthorized(
                        "cart quote request is unknown or unauthorized"
                    )
                validate_persistent_cart_quote_request(request)
                fingerprint = canonical_digest(
                    {
                        "request_id": request.request_id,
                        "request_digest": request.request_digest,
                        "quote_ttl_ticks": quote_ttl_ticks,
                    }
                )
                scope = f"cart-quote:{request.market_id}"
                replay = self._authority_operation_replay(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PersistentCartQuote):
                        raise WorldError("cart quote idempotency outcome is missing")
                    self._conn.commit()
                    return outcome
                revisions = tuple(
                    cast(
                        MandateRevision,
                        _row_to_value("mandate_revisions", row),
                    )
                    for row in self._conn.execute(
                        "SELECT * FROM mandate_revisions WHERE mandate_id = ? "
                        "ORDER BY revision",
                        (request.mandate_id,),
                    ).fetchall()
                )
                if not revisions:
                    raise WriteNotAuthorized("cart quote request mandate is missing")
                before_tick = self._clock_value()
                event_tick = before_tick + 1
                current_mandate = revisions[-1]
                try:
                    assert_cart_quote_request_usable(
                        request,
                        merchant_id=original_actor,
                        mandate_revision=current_mandate.revision,
                        mandate_digest=current_mandate.revision_digest,
                        logical_tick=event_tick,
                    )
                except CartQuoteRequestStaleError as exc:
                    raise CartQuoteStaleError(str(exc)) from exc
                except CartQuoteRequestAuthorityError as exc:
                    raise WriteNotAuthorized(str(exc)) from exc
                all_policies = tuple(
                    cast(
                        PricingPolicyRevision,
                        _row_to_value("pricing_policy_revisions", row),
                    )
                    for row in self._conn.execute(
                        "SELECT * FROM pricing_policy_revisions ORDER BY revision_key"
                    ).fetchall()
                )
                listings: dict[str, Listing] = {}
                inventory: dict[str, InventoryRow] = {}
                policies: dict[str, PricingPolicyRevision] = {}
                for request_line in request.lines:
                    listing = cast(
                        "Listing | None",
                        self._fetch_value("catalog", request_line.sku_id),
                    )
                    stock = self._fetch_value("inventory", request_line.sku_id)
                    if listing is None or not isinstance(stock, InventoryRow):
                        raise OutOfStock("requested cart source is unavailable")
                    if (
                        str(listing.merchant_id) != request_line.merchant_id
                        or _catalog_revision(listing)
                        != request_line.catalog_revision
                        or catalog_snapshot_digest(listing)
                        != request_line.catalog_digest
                    ):
                        raise CartQuoteStaleError(
                            "cart quote request catalog binding changed"
                        )
                    listings[request_line.sku_id] = listing
                    inventory[request_line.sku_id] = stock
                    policies[request_line.sku_id] = (
                        _active_pricing_policy_for_listing(
                            all_policies,
                            market_id=request.market_id,
                            listing=listing,
                            logical_tick=event_tick,
                        )
                    )
                normalized = normalize_cart_quote_intent(
                    {
                        "mandate_id": request.mandate_id,
                        "lines": [
                            {"sku_id": line.sku_id, "qty": line.qty}
                            for line in request.lines
                        ],
                        "fill_policy": request.fill_policy,
                        "backorder_policy": request.backorder_policy,
                    }
                )
                quote = build_authoritative_cart_quote(
                    intent=normalized,
                    market_id=request.market_id,
                    requested_by=original_actor,
                    issuer_id="world",
                    idempotency_key=idempotency_key,
                    authority=QuoteAuthority(
                        principal_id=request.principal_id,
                        buyer_id=request.buyer_id,
                        mandate_revision=request.mandate_revision,
                        mandate_digest=request.mandate_digest,
                    ),
                    listings=listings,
                    inventory=inventory,
                    policies_by_sku=policies,
                    issued_at_tick=event_tick,
                    quote_ttl_ticks=min(
                        quote_ttl_ticks, request.expires_at_tick - event_tick
                    ),
                    request_id=request.request_id,
                )
                existing = tuple(
                    cast(
                        PersistentCartQuote,
                        _row_to_value("persistent_cart_quotes", row),
                    )
                    for row in self._conn.execute(
                        "SELECT * FROM persistent_cart_quotes ORDER BY quote_id"
                    ).fetchall()
                )
                disposition, _ = apply_cart_quote_record(
                    existing,
                    quote,
                    trusted_issuer_id="world",
                    server_tick=event_tick,
                )
                if disposition != "append":
                    raise WorldError(
                        "new cart quote unexpectedly classified as retry"
                    )
                _upsert(self._conn, "persistent_cart_quotes", quote)
                _record_mutation(
                    self._conn,
                    table="persistent_cart_quotes",
                    key=quote.quote_id,
                    action="world.issue_cart_quote",
                    before=None,
                    after=quote,
                )
                operation = self._record_authority_operation(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="persistent_cart_quotes",
                    outcome_key=quote.quote_id,
                    authority_action="world.issue_cart_quote",
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.issue_cart_quote",
                    before=before_tick,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="issue_cart_quote",
                    authority_action="world.issue_cart_quote",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    subject_id=quote.quote_id,
                    table_writes=(
                        TableWrite(
                            table="persistent_cart_quotes",
                            key=quote.quote_id,
                            op="create",
                            before=None,
                            after=quote,
                        ),
                        TableWrite(
                            table="authority_operations",
                            key=operation.operation_key,
                            op="create",
                            before=None,
                            after=operation,
                        ),
                        TableWrite(
                            table="logical_time",
                            key="world",
                            op="update",
                            before=before_tick,
                            after=event_tick,
                        ),
                    ),
                    invariants_held=(
                        "world-owned-pricing",
                        "principal-mandate-bound",
                        "catalog-inventory-policy-snapshots",
                        "actor-scoped-idempotency",
                    ),
                    request_fingerprint=fingerprint,
                )
                self._conn.commit()
                return quote
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise IdempotencyConflict(
                    "conflicting concurrent requested cart quote"
                ) from exc
            except BaseException:
                self._conn.rollback()
                raise

    def issue_cart_quote(
        self,
        intent: CartQuoteIntent | Mapping[str, Any],
        *,
        market_id: str,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        quote_ttl_ticks: int = 10,
    ) -> PersistentCartQuote:
        """Build and durably persist a quote entirely inside one SQL transaction."""

        if by_actor != "platform:checkout":
            raise LifecycleAuthorizationError(
                "only platform:checkout may request an authoritative cart quote"
            )
        if original_actor.split(":", 1)[0] != "buyer":
            raise LifecycleAuthorizationError(
                "cart quote original actor must be a fully qualified buyer"
            )
        if not isinstance(market_id, str) or not market_id.strip():
            raise WorldError("cart quote market_id must be non-empty")
        normalized = normalize_cart_quote_intent(intent)
        fingerprint = cart_quote_intent_fingerprint(
            normalized,
            market_id=market_id,
            quote_ttl_ticks=quote_ttl_ticks,
        )
        scope = f"cart-quote:{market_id}"
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PersistentCartQuote):
                        raise WorldError("cart quote idempotency outcome is missing")
                    self._conn.commit()
                    return outcome

                authority = cast(
                    "MandateRevisionAuthority | None",
                    self._fetch_value(
                        "mandate_authorities", normalized["mandate_id"]
                    ),
                )
                if authority is None:
                    raise WriteNotAuthorized("cart quote mandate authority is missing")
                if authority.buyer_id != original_actor:
                    raise WriteNotAuthorized(
                        "cart quote actor does not own the persisted mandate"
                    )
                revision_rows = self._conn.execute(
                    "SELECT * FROM mandate_revisions WHERE mandate_id = ? "
                    "ORDER BY revision",
                    (normalized["mandate_id"],),
                ).fetchall()
                revisions = tuple(
                    cast(
                        MandateRevision,
                        _row_to_value("mandate_revisions", row),
                    )
                    for row in revision_rows
                )
                if not revisions:
                    raise WriteNotAuthorized("cart quote mandate has no revision")
                current_mandate = revisions[-1]
                if (
                    current_mandate.buyer_id != authority.buyer_id
                    or current_mandate.principal_id != authority.principal_id
                ):
                    raise WriteNotAuthorized(
                        "cart quote mandate authority is inconsistent"
                    )

                before_tick = self._clock_value()
                event_tick = before_tick + 1
                all_policy_rows = self._conn.execute(
                    "SELECT * FROM pricing_policy_revisions ORDER BY revision_key"
                ).fetchall()
                all_policies = tuple(
                    cast(
                        PricingPolicyRevision,
                        _row_to_value("pricing_policy_revisions", row),
                    )
                    for row in all_policy_rows
                )
                listings: dict[str, Listing] = {}
                inventory: dict[str, InventoryRow] = {}
                selected_policies: dict[str, PricingPolicyRevision] = {}
                for line in normalized["lines"]:
                    sku_id = line["sku_id"]
                    listing = cast(
                        "Listing | None", self._fetch_value("catalog", sku_id)
                    )
                    stock = self._fetch_value("inventory", sku_id)
                    if listing is None or not isinstance(stock, InventoryRow):
                        raise OutOfStock(
                            "cart quote requires authoritative listing and "
                            f"inventory for {sku_id!r}"
                        )
                    listings[sku_id] = listing
                    inventory[sku_id] = stock
                    selected_policies[sku_id] = _active_pricing_policy_for_listing(
                        all_policies,
                        market_id=market_id,
                        listing=listing,
                        logical_tick=event_tick,
                    )
                quote = build_authoritative_cart_quote(
                    intent=normalized,
                    market_id=market_id,
                    requested_by=original_actor,
                    issuer_id="world",
                    idempotency_key=idempotency_key,
                    authority=QuoteAuthority(
                        principal_id=authority.principal_id,
                        buyer_id=authority.buyer_id,
                        mandate_revision=current_mandate.revision,
                        mandate_digest=current_mandate.revision_digest,
                    ),
                    listings=listings,
                    inventory=inventory,
                    policies_by_sku=selected_policies,
                    issued_at_tick=event_tick,
                    quote_ttl_ticks=quote_ttl_ticks,
                )
                if quote.grand_total_minor > _mandate_budget_minor(revisions):
                    raise WriteNotAuthorized(
                        "authoritative cart quote exceeds the principal mandate budget"
                    )
                existing_quote_rows = self._conn.execute(
                    "SELECT * FROM persistent_cart_quotes ORDER BY quote_id"
                ).fetchall()
                disposition, _ = apply_cart_quote_record(
                    (
                        cast(
                            PersistentCartQuote,
                            _row_to_value("persistent_cart_quotes", row),
                        )
                        for row in existing_quote_rows
                    ),
                    quote,
                    trusted_issuer_id="world",
                    server_tick=event_tick,
                )
                if disposition != "append":
                    raise WorldError(
                        "new cart quote unexpectedly classified as retry"
                    )
                _upsert(self._conn, "persistent_cart_quotes", quote)
                _record_mutation(
                    self._conn,
                    table="persistent_cart_quotes",
                    key=quote.quote_id,
                    action="world.issue_cart_quote",
                    before=None,
                    after=quote,
                )
                self._record_authority_operation(
                    scope=scope,
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="persistent_cart_quotes",
                    outcome_key=quote.quote_id,
                    authority_action="world.issue_cart_quote",
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.issue_cart_quote",
                    before=before_tick,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._conn.commit()
                return quote
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise IdempotencyConflict(
                    "conflicting concurrent cart quote request"
                ) from exc
            except BaseException:
                self._conn.rollback()
                raise

    def apply_catalog_mutation(
        self,
        intent: CatalogMutationIntent | Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Listing:
        """Durably apply one owner-validated, actor-idempotent catalog intent."""

        if by_actor != "platform:catalog":
            raise WriteNotAuthorized(
                "only platform:catalog may apply actor catalog mutations"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise IdempotencyConflict(
                "catalog mutation idempotency key must not be blank"
            )
        normalized = normalize_catalog_mutation_intent(intent)
        fingerprint = catalog_mutation_fingerprint(normalized)
        sku_id = SkuId(normalized["sku_id"])
        operation_key = authority_operation_key(
            "catalog-mutation", original_actor, idempotency_key
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                prior_operation = cast(
                    "AuthorityOperationRecord | None",
                    self._fetch_value("authority_operations", operation_key),
                )
                if prior_operation is not None:
                    if prior_operation.request_fingerprint != fingerprint:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was reused "
                            "for another request"
                        )
                    if prior_operation.outcome_listing is None:
                        raise WorldError(
                            "catalog mutation idempotency outcome is missing"
                        )
                    self._conn.commit()
                    return prior_operation.outcome_listing

                before = cast(
                    "Listing | None", self._fetch_value("catalog", str(sku_id))
                )
                desired = apply_catalog_mutation_intent(
                    before, normalized, original_actor=original_actor
                )
                changed = (
                    before is None
                    or not catalog_listings_semantically_equal(desired, before)
                )
                outcome = (
                    _revisioned_listing(desired, before)
                    if changed
                    else cast(Listing, before)
                )
                if changed:
                    _upsert(self._conn, "catalog", outcome)
                    _record_mutation(
                        self._conn,
                        table="catalog",
                        key=str(sku_id),
                        action="world.apply_catalog_mutation",
                        before=before,
                        after=outcome,
                    )
                self._record_authority_operation(
                    scope="catalog-mutation",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="catalog",
                    outcome_key=str(sku_id),
                    outcome_listing=outcome,
                    authority_action="world.apply_catalog_mutation",
                )
                self._conn.commit()
                return outcome
            except BaseException:
                self._conn.rollback()
                raise

    def persist_evidence_record(
        self,
        record: EvidenceRecord,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> EvidenceRecord:
        """Append one issuer-authenticated evidence version durably."""

        if by_actor != "platform:evidence":
            raise WriteNotAuthorized(
                "only platform:evidence may persist evidence records"
            )
        if not idempotency_key:
            raise IdempotencyConflict("evidence idempotency key must not be blank")
        fingerprint = record.record_digest
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope="evidence",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, EvidenceRecord):
                        raise WorldError("evidence idempotency outcome is missing")
                    self._conn.commit()
                    return outcome

                key = evidence_record_key(record.record_id, record.version)
                exact = self._fetch_value("evidence_records", key)
                if exact is not None:
                    if exact != record or record.issuer_id != original_actor:
                        raise IdempotencyConflict(
                            "evidence version retry differs from persisted content"
                        )
                    operation = self._record_authority_operation(
                        scope="evidence",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        request_fingerprint=fingerprint,
                        outcome_table="evidence_records",
                        outcome_key=key,
                    )
                    self._append_world_commit(
                        commit_kind="transaction",
                        operation="bind_evidence_idempotency",
                        authority_action="world.persist_evidence_record",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        subject_id=record.record_id,
                        table_writes=(
                            TableWrite(
                                table="authority_operations",
                                key=operation.operation_key,
                                op="create",
                                before=None,
                                after=operation,
                            ),
                        ),
                        invariants_held=(
                            "actor-scoped-idempotency",
                            "zero-evidence-write",
                        ),
                        request_fingerprint=fingerprint,
                    )
                    self._conn.commit()
                    return cast(EvidenceRecord, exact)

                row = self._conn.execute(
                    "SELECT * FROM evidence_records WHERE record_id = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (record.record_id,),
                ).fetchone()
                current = (
                    None
                    if row is None
                    else cast(EvidenceRecord, _row_to_value("evidence_records", row))
                )
                validate_evidence_append(
                    current,
                    record,
                    original_actor=original_actor,
                    logical_time=self._clock_value(),
                )
                _upsert(self._conn, "evidence_records", record)
                operation = self._record_authority_operation(
                    scope="evidence",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="evidence_records",
                    outcome_key=key,
                )
                _record_mutation(
                    self._conn,
                    table="evidence_records",
                    key=key,
                    action="world.persist_evidence_record",
                    before=None,
                    after=record,
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="persist_evidence_record",
                    authority_action="world.persist_evidence_record",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    subject_id=record.record_id,
                    table_writes=(
                        TableWrite(
                            table="evidence_records",
                            key=key,
                            op="create",
                            before=None,
                            after=record,
                        ),
                        TableWrite(
                            table="authority_operations",
                            key=operation.operation_key,
                            op="create",
                            before=None,
                            after=operation,
                        ),
                    ),
                    invariants_held=(
                        "issuer-authenticated",
                        "owner-acl-bound",
                        "version-contiguous",
                        "digest-verified",
                        "append-only",
                    ),
                    request_fingerprint=fingerprint,
                )
                self._conn.commit()
                return record
            except BaseException:
                self._conn.rollback()
                raise

    def read_evidence_record(
        self,
        record_id: str,
        *,
        caller: str,
        version: int | None = None,
        record_digest: str | None = None,
    ) -> EvidenceRecord | None:
        """Read current or historical evidence through its persisted ACL."""

        if version is not None and record_digest is not None:
            raise ValueError("choose evidence version or digest, not both")
        with self._lock:
            if version is not None:
                row = self._fetch(
                    "evidence_records", evidence_record_key(record_id, version)
                )
            elif record_digest is not None:
                row = self._conn.execute(
                    "SELECT * FROM evidence_records "
                    "WHERE record_id = ? AND record_digest = ?",
                    (record_id, record_digest),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM evidence_records WHERE record_id = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (record_id,),
                ).fetchone()
            if row is None:
                return None
            record = cast(EvidenceRecord, _row_to_value("evidence_records", row))
            return authorize_persisted_evidence_read(record, reader_id=caller)

    def register_mandate_authority(
        self,
        authority: MandateRevisionAuthority,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevisionAuthority:
        """Register one immutable principal-to-buyer authority durably."""

        if by_actor != "platform:mandate":
            raise WriteNotAuthorized(
                "only platform:mandate may register mandate authority"
            )
        if not idempotency_key:
            raise IdempotencyConflict("mandate idempotency key must not be blank")
        fingerprint = canonical_digest(mandate_authority_to_wire(authority))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope="mandate-authority",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, MandateRevisionAuthority):
                        raise WorldError("mandate authority outcome is missing")
                    self._conn.commit()
                    return outcome
                current = cast(
                    "MandateRevisionAuthority | None",
                    self._fetch_value("mandate_authorities", authority.mandate_id),
                )
                disposition = validate_mandate_authority_registration(
                    current, authority, original_actor=original_actor
                )
                if disposition == "append":
                    _upsert(self._conn, "mandate_authorities", authority)
                    _record_mutation(
                        self._conn,
                        table="mandate_authorities",
                        key=authority.mandate_id,
                        action="world.register_mandate_authority",
                        before=None,
                        after=authority,
                    )
                self._record_authority_operation(
                    scope="mandate-authority",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="mandate_authorities",
                    outcome_key=authority.mandate_id,
                )
                self._conn.commit()
                return authority if current is None else current
            except BaseException:
                self._conn.rollback()
                raise

    def append_mandate_revision(
        self,
        revision: MandateRevision,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> MandateRevision:
        """Append one principal-authenticated mandate revision durably."""

        if by_actor != "platform:mandate":
            raise WriteNotAuthorized(
                "only platform:mandate may append mandate revisions"
            )
        if not idempotency_key:
            raise IdempotencyConflict("mandate idempotency key must not be blank")
        fingerprint = revision.revision_digest
        key = mandate_revision_key(revision.mandate_id, revision.revision)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._authority_operation_replay(
                    scope="mandate-revision",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, MandateRevision):
                        raise WorldError("mandate revision outcome is missing")
                    self._conn.commit()
                    return outcome
                authority = cast(
                    "MandateRevisionAuthority | None",
                    self._fetch_value("mandate_authorities", revision.mandate_id),
                )
                if authority is None:
                    raise WriteNotAuthorized(
                        f"mandate {revision.mandate_id!r} has no registered authority"
                    )
                exact = self._fetch_value("mandate_revisions", key)
                if exact is not None:
                    if exact != revision or revision.principal_id != original_actor:
                        raise IdempotencyConflict(
                            "mandate revision retry differs from persisted content"
                        )
                    self._record_authority_operation(
                        scope="mandate-revision",
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                        request_fingerprint=fingerprint,
                        outcome_table="mandate_revisions",
                        outcome_key=key,
                    )
                    self._conn.commit()
                    return cast(MandateRevision, exact)
                current_row = self._conn.execute(
                    "SELECT * FROM mandate_revisions WHERE mandate_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (revision.mandate_id,),
                ).fetchone()
                current = (
                    None
                    if current_row is None
                    else cast(
                        MandateRevision,
                        _row_to_value("mandate_revisions", current_row),
                    )
                )
                validate_mandate_append(
                    current,
                    revision,
                    authority,
                    original_actor=original_actor,
                    logical_time=self._clock_value(),
                )
                _upsert(self._conn, "mandate_revisions", revision)
                self._record_authority_operation(
                    scope="mandate-revision",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    outcome_table="mandate_revisions",
                    outcome_key=key,
                )
                _record_mutation(
                    self._conn,
                    table="mandate_revisions",
                    key=key,
                    action="world.append_mandate_revision",
                    before=None,
                    after=revision,
                )
                self._conn.commit()
                return revision
            except BaseException:
                self._conn.rollback()
                raise

    def mandate_revisions(
        self, mandate_id: str, *, caller: str
    ) -> tuple[MandateRevision, ...]:
        """Read a revision history only to its bound principal or buyer."""

        with self._lock:
            authority = cast(
                "MandateRevisionAuthority | None",
                self._fetch_value("mandate_authorities", mandate_id),
            )
            if authority is None:
                return ()
            authorize_mandate_read(authority, reader_id=caller)
            rows = self._conn.execute(
                "SELECT * FROM mandate_revisions WHERE mandate_id = ? "
                "ORDER BY revision",
                (mandate_id,),
            ).fetchall()
            return tuple(
                cast(MandateRevision, _row_to_value("mandate_revisions", row))
                for row in rows
            )

    def apply_listing_claim(
        self,
        claim: ListingClaim,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> ListingClaim:
        """Apply one merchant-owned, evidence-bound listing claim transition."""

        if by_actor != "platform:claims":
            raise WriteNotAuthorized(
                "only platform:claims may apply listing claims"
            )
        if not idempotency_key:
            raise IdempotencyConflict("claim idempotency key must not be blank")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._conn.execute(
                    "SELECT claim_id, event_digest FROM listing_claim_keys "
                    "WHERE merchant_id = ? AND idempotency_key = ?",
                    (original_actor, idempotency_key),
                ).fetchone()
                if replay is not None:
                    outcome = self._fetch_value(
                        "listing_claims", str(replay["claim_id"])
                    )
                    if not isinstance(outcome, ListingClaim):
                        raise WorldError("listing claim outcome is missing")
                    if (
                        claim.claim_id != outcome.claim_id
                        or claim.listing_id != outcome.listing_id
                        or claim.merchant_id != outcome.merchant_id
                        or claim.subject != outcome.subject
                        or claim.issuer_id != outcome.issuer_id
                        or claim.current.idempotency_key != idempotency_key
                        or claim.current.event_digest != str(replay["event_digest"])
                    ):
                        raise IdempotencyConflict(
                            "merchant claim idempotency key was reused for another request"
                        )
                    self._conn.commit()
                    return outcome
                current = cast(
                    "ListingClaim | None",
                    self._fetch_value("listing_claims", claim.claim_id),
                )
                listing = cast(
                    "Listing | None",
                    self._fetch_value("catalog", claim.listing_id),
                )

                def evidence_lookup(
                    record_id: str, digest: str
                ) -> EvidenceRecord | None:
                    row = self._conn.execute(
                        "SELECT * FROM evidence_records "
                        "WHERE record_id = ? AND record_digest = ?",
                        (record_id, digest),
                    ).fetchone()
                    return (
                        None
                        if row is None
                        else cast(
                            EvidenceRecord,
                            _row_to_value("evidence_records", row),
                        )
                    )

                disposition = validate_listing_claim_append(
                    current,
                    claim,
                    listing=listing,
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    logical_time=self._clock_value(),
                    evidence_lookup=evidence_lookup,
                )
                if disposition == "append":
                    _upsert(self._conn, "listing_claims", claim)
                    _record_mutation(
                        self._conn,
                        table="listing_claims",
                        key=claim.claim_id,
                        action="world.apply_listing_claim",
                        before=current,
                        after=claim,
                    )
                self._conn.commit()
                return claim if disposition == "append" else cast(ListingClaim, current)
            except BaseException:
                self._conn.rollback()
                raise

    def read_listing_claim(
        self, claim_id: str, *, caller: str
    ) -> ListingClaim | None:
        with self._lock:
            claim = cast(
                "ListingClaim | None",
                self._fetch_value("listing_claims", claim_id),
            )
            if claim is None:
                return None
            trusted = caller == "runtime" or caller.startswith(
                ("runtime:", "platform:")
            )
            if claim.state == "draft" and caller != claim.merchant_id and not trusted:
                return None
            return claim

    def listing_claims_for_listing(
        self, listing_id: str, *, caller: str
    ) -> tuple[ListingClaim, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM listing_claims WHERE listing_id = ? ORDER BY claim_id",
                (listing_id,),
            ).fetchall()
            claims = tuple(
                cast(ListingClaim, _row_to_value("listing_claims", row))
                for row in rows
            )
            trusted = caller == "runtime" or caller.startswith(
                ("runtime:", "platform:")
            )
            return tuple(
                claim
                for claim in claims
                if claim.state != "draft"
                or claim.merchant_id == caller
                or trusted
            )

    def _authority_operation_replay(
        self,
        *,
        scope: str,
        actor_id: str,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> tuple[str, str] | None:
        row = self._conn.execute(
            "SELECT request_fingerprint, outcome_table, outcome_key "
            "FROM authority_operations "
            "WHERE scope = ? AND actor_id = ? AND idempotency_key = ?",
            (scope, actor_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if str(row["request_fingerprint"]) != request_fingerprint:
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key!r} was reused for another request"
            )
        return str(row["outcome_table"]), str(row["outcome_key"])

    def _record_authority_operation(
        self,
        *,
        scope: str,
        actor_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        outcome_table: str,
        outcome_key: str,
        outcome_listing: Listing | None = None,
        authority_action: str | None = None,
    ) -> AuthorityOperationRecord:
        operation_key = authority_operation_key(scope, actor_id, idempotency_key)
        record = AuthorityOperationRecord(
            operation_key=operation_key,
            scope=scope,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            outcome_table=outcome_table,
            outcome_key=outcome_key,
            outcome_listing=outcome_listing,
        )
        _upsert(self._conn, "authority_operations", record)
        _record_mutation(
            self._conn,
            table="authority_operations",
            key=operation_key,
            action=authority_action or f"world.{scope}",
            before=None,
            after=record,
        )
        return record

    def _clock_value(self) -> int:
        row = self._conn.execute(
            "SELECT logical_time FROM world_clock WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise WorldError("world clock row is missing")
        return int(row["logical_time"])

    def _set_clock(self, value: int) -> None:
        self._conn.execute(
            "UPDATE world_clock SET logical_time = ? WHERE singleton = 1",
            (value,),
        )

    def read(self, table: str, key: Any, *, caller: str | None = None) -> Any | None:
        row = self._fetch(table, str(key))
        if row is None:
            return None
        value = _row_to_value(table, row)
        if _visible(table, value, caller):
            return value
        return None

    def read_order_state_revision(
        self,
        order_id: OrderId | str,
        *,
        caller: str,
    ) -> int | None:
        with self._lock:
            order = cast("Order | None", self.read("orders", order_id, caller=caller))
            if order is None:
                return None
            return self._order_revision_for_existing(str(order.order_id))

    def read_order_operation_reference(
        self,
        order_id: OrderId | str,
        *,
        caller: str,
    ) -> str | None:
        with self._lock:
            order = cast("Order | None", self.read("orders", order_id, caller=caller))
            if order is None:
                return None
            revision = self._order_revision_for_existing(str(order.order_id))
            return order_operation_reference_digest(order, revision)

    def protocol_events_for_stream(
        self,
        binding_digest: str,
        *,
        caller: str,
    ) -> tuple[ProtocolEvent, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM protocol_events
                WHERE binding_digest = ?
                ORDER BY event_sequence, event_id
                """,
                (binding_digest,),
            ).fetchall()
            events = tuple(
                cast(ProtocolEvent, _row_to_value("protocol_events", row))
                for row in rows
            )
            return tuple(
                event for event in events if _visible("protocol_events", event, caller)
            )

    def protocol_receipts_for_stream(
        self,
        binding_digest: str,
        *,
        order_id: OrderId | str,
        caller: str,
    ) -> tuple[ProtocolEventReceipt, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM protocol_receipts
                WHERE binding_digest = ? AND order_id = ?
                ORDER BY logical_tick, receipt_id
                """,
                (binding_digest, str(order_id)),
            ).fetchall()
            receipts = tuple(
                cast(ProtocolEventReceipt, _row_to_value("protocol_receipts", row))
                for row in rows
            )
            return tuple(
                receipt
                for receipt in receipts
                if _visible("protocol_receipts", receipt, caller)
            )

    def publish_protocol_event(
        self,
        event: ProtocolEvent,
        *,
        by_actor: str,
    ) -> ProtocolEvent:
        if by_actor != "platform:events":
            raise WriteNotAuthorized(
                "only platform:events may publish protocol events"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                order = cast(
                    "Order | None",
                    self._fetch_value("orders", event.binding.order_id),
                )
                if order is None:
                    raise ProtocolEventSchemaError(
                        f"protocol event order {event.binding.order_id!r} does not exist"
                    )
                _validate_event_order_binding(event, order)
                revision = self._order_revision_for_existing(str(order.order_id))
                certificates = MatchCertificateTable()
                for key, certificate in self.iter_table("match_certificates"):
                    certificates.write(str(key), cast(MatchCertificate, certificate))
                _validate_protocol_event_reference(
                    event,
                    order,
                    revision=revision,
                    logical_time=self._clock_value(),
                    certificates=certificates,
                )
                events = self._protocol_events_internal(
                    event.binding.binding_digest
                )
                disposition, _ = apply_protocol_event(
                    events,
                    event,
                    binding=event.binding,
                    server_tick=self._clock_value(),
                    current_order_state=order.state.value,
                    current_state_revision=revision,
                )
                if disposition == "idempotent":
                    self._conn.commit()
                    return next(
                        existing
                        for existing in events
                        if existing.event_digest == event.event_digest
                    )
                _upsert(self._conn, "protocol_events", event)
                _record_mutation(
                    self._conn,
                    table="protocol_events",
                    key=event.event_id,
                    action="world.publish_protocol_event",
                    before=None,
                    after=event,
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="publish_protocol_event",
                    authority_action="world.publish_protocol_event",
                    actor_id=by_actor,
                    idempotency_key=event.idempotency_key,
                    subject_id=str(order.order_id),
                    table_writes=(
                        TableWrite(
                            table="protocol_events",
                            key=event.event_id,
                            op="create",
                            before=None,
                            after=event,
                        ),
                    ),
                    invariants_held=(
                        "platform-events-only",
                        "authoritative-order-binding",
                        "authoritative-state-revision",
                        "reference-verified",
                        "append-only",
                    ),
                )
                self._conn.commit()
                return event
            except BaseException:
                self._conn.rollback()
                raise

    def append_protocol_receipt(
        self,
        receipt: ProtocolEventReceipt,
        *,
        by_actor: str,
        original_actor: str,
    ) -> ProtocolEventReceipt:
        if by_actor != "platform:events":
            raise WriteNotAuthorized(
                "only platform:events may append protocol receipts"
            )
        if original_actor != receipt.actor_id:
            raise LifecycleAuthorizationError(
                "authenticated receipt actor does not match sealed actor identity"
            )
        if receipt.decision == "process":
            raise WriteNotAuthorized(
                "protocol event processing requires a registered CommerceWorld "
                "operation handler and committed effect digest"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                event = cast(
                    "ProtocolEvent | None",
                    self._fetch_value("protocol_events", receipt.event_id),
                )
                if event is None or event.event_digest != receipt.event_digest:
                    raise ProtocolEventSchemaError(
                        "receipt does not reference one exact persisted protocol event"
                    )
                if original_actor != event.binding.recipient_id:
                    raise LifecycleAuthorizationError(
                        "only the persisted event recipient may decide the event"
                    )
                order = cast(
                    "Order | None",
                    self._fetch_value("orders", event.binding.order_id),
                )
                if order is None:
                    raise ProtocolEventSchemaError(
                        f"protocol receipt order {event.binding.order_id!r} does not exist"
                    )
                _validate_event_order_binding(event, order)
                revision = self._order_revision_for_existing(str(order.order_id))
                events = self._protocol_events_internal(
                    event.binding.binding_digest
                )
                receipts = self._protocol_receipts_internal(
                    event.binding.binding_digest,
                    str(order.order_id),
                )
                disposition, _ = apply_protocol_receipt(
                    receipts,
                    events,
                    receipt,
                    binding=event.binding,
                    server_tick=self._clock_value(),
                    current_order_state=order.state.value,
                    current_state_revision=revision,
                )
                if disposition == "idempotent":
                    self._conn.commit()
                    return next(
                        existing
                        for existing in receipts
                        if existing.receipt_digest == receipt.receipt_digest
                    )
                _upsert(self._conn, "protocol_receipts", receipt)
                _record_mutation(
                    self._conn,
                    table="protocol_receipts",
                    key=receipt.receipt_id,
                    action="world.append_protocol_receipt",
                    before=None,
                    after=receipt,
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="append_protocol_receipt",
                    authority_action="world.append_protocol_receipt",
                    actor_id=original_actor,
                    idempotency_key=receipt.idempotency_key,
                    subject_id=str(order.order_id),
                    table_writes=(
                        TableWrite(
                            table="protocol_receipts",
                            key=receipt.receipt_id,
                            op="create",
                            before=None,
                            after=receipt,
                        ),
                    ),
                    invariants_held=(
                        "platform-events-mediated",
                        "recipient-identity-bound",
                        "exact-event-reference",
                        "authoritative-observed-state",
                        "evidence-only-no-business-effect",
                    ),
                )
                self._conn.commit()
                return receipt
            except BaseException:
                self._conn.rollback()
                raise

    def process_protocol_event(
        self,
        *,
        event_id: str,
        by_actor: str,
        original_actor: str,
        reason: str,
        idempotency_key: str,
    ) -> ProtocolEventReceipt:
        """Atomically run one operation and append its durable process receipt.

        The event precondition read, registered commerce mutation, committed
        effect lookup, and receipt append share one ``BEGIN IMMEDIATE`` unit.
        A failure at any point rolls back both business state and receipt state.
        Exact retries return the original receipt without repeating the effect.
        """

        if by_actor != "platform:events":
            raise WriteNotAuthorized(
                "only platform:events may process protocol events"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ProtocolEventSchemaError("protocol process reason must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ProtocolEventSchemaError(
                "protocol process idempotency key must be non-empty"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                event = cast(
                    "ProtocolEvent | None",
                    self._fetch_value("protocol_events", event_id),
                )
                if event is None:
                    raise ProtocolEventSchemaError("protocol event does not exist")
                if original_actor != event.binding.recipient_id:
                    raise LifecycleAuthorizationError(
                        "only the persisted event recipient may process it"
                    )
                receipts = self._protocol_receipts_internal(
                    event.binding.binding_digest,
                    event.binding.order_id,
                )
                replay = _protocol_process_receipt_replay(
                    receipts,
                    event=event,
                    actor_id=original_actor,
                    reason=reason,
                    idempotency_key=idempotency_key,
                )
                if replay is not None:
                    self._conn.commit()
                    return replay
                order = cast(
                    "Order | None",
                    self._fetch_value("orders", event.binding.order_id),
                )
                if order is None:
                    raise ProtocolEventSchemaError(
                        "protocol event order does not exist"
                    )
                _validate_event_order_binding(event, order)
                revision = self._order_revision_for_existing(str(order.order_id))
                decision_tick = self._clock_value()
                certificates = MatchCertificateTable()
                for key, certificate in self.iter_table("match_certificates"):
                    certificates.write(str(key), cast(MatchCertificate, certificate))
                operation = _validate_protocol_process_precondition(
                    event,
                    order=order,
                    revision=revision,
                    logical_time=decision_tick,
                    original_actor=original_actor,
                    certificates=certificates,
                )
                business_commit_cursor = self.begin_evidence_window()

                effect_key = _protocol_effect_idempotency_key(event)
                if operation == "settle_order":
                    payment = _protocol_payment_receipt(
                        event,
                        order=order,
                        logical_time=decision_tick,
                        refund=False,
                    )
                    outcome = self.settle_order(
                        order=order,
                        receipt=payment,
                        by_role="platform",
                        idempotency_key=effect_key,
                        _manage_transaction=False,
                    )
                    outcome_table, outcome_key = "ledger", str(outcome.txn_id)
                elif operation == "dispatch_order":
                    self.dispatch_order(
                        order_id=order.order_id,
                        by_actor=event.binding.recipient_id,
                        _manage_transaction=False,
                    )
                    outcome_table = "shipments"
                    outcome_key = f"shipment:{order.order_id}"
                elif operation == "refund_order":
                    allocation = cast(
                        "FulfillmentAllocation | None",
                        self._fetch_value("fulfillments", str(order.order_id)),
                    )
                    paid_qty = (
                        order.qty
                        if allocation is None
                        else allocation.fulfilled_qty
                    )
                    refund = _protocol_payment_receipt(
                        event,
                        order=order,
                        logical_time=decision_tick,
                        refund=True,
                        paid_qty=paid_qty,
                    )
                    outcome = self.refund_order(
                        order=order,
                        refund_receipt=refund,
                        by_role="platform",
                        idempotency_key=effect_key,
                        _manage_transaction=False,
                    )
                    outcome_table, outcome_key = "ledger", str(outcome.txn_id)
                else:  # pragma: no cover - pure registry validation owns this.
                    raise ProtocolEventSchemaError(
                        f"unsupported protocol operation {operation!r}"
                    )

                if self._fetch_value(outcome_table, outcome_key) is None:
                    raise WorldError(
                        "processed protocol operation has no committed outcome row"
                    )
                effect_reference = protocol_operation_effect_reference_digest(
                    event,
                    operation=operation,
                    outcome_table=outcome_table,
                    outcome_key=outcome_key,
                )
                receipt = build_protocol_event_receipt(
                    event,
                    receipt_id=_protocol_process_receipt_id(
                        event,
                        actor_id=original_actor,
                        idempotency_key=idempotency_key,
                    ),
                    decision="process",
                    actor_id=original_actor,
                    observed_order_state=event.required_order_state,
                    observed_state_revision=event.required_state_revision,
                    reason=reason,
                    effect_reference_digests=(effect_reference,),
                    logical_tick=decision_tick,
                    idempotency_key=idempotency_key,
                )
                events = self._protocol_events_internal(
                    event.binding.binding_digest
                )
                validate_protocol_receipt_stream(
                    (*receipts, receipt),
                    events=events,
                    binding=event.binding,
                )
                if self._fetch("protocol_receipts", receipt.receipt_id) is not None:
                    raise IdempotencyConflict(
                        "protocol process receipt identity already exists"
                    )
                self._persist_protocol_process_receipt_in_transaction(receipt)
                self._merge_protocol_process_commit_in_transaction(
                    business_commit_cursor=business_commit_cursor,
                    receipt=receipt,
                    order_id=str(order.order_id),
                    operation=operation,
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                )
                self._conn.commit()
                return receipt
            except BaseException:
                self._conn.rollback()
                raise

    def _persist_protocol_process_receipt_in_transaction(
        self,
        receipt: ProtocolEventReceipt,
    ) -> None:
        """Append one process receipt on the caller's open SQLite transaction."""

        _upsert(self._conn, "protocol_receipts", receipt)
        _record_mutation(
            self._conn,
            table="protocol_receipts",
            key=receipt.receipt_id,
            action="world.process_protocol_event",
            before=None,
            after=receipt,
        )

    def _merge_protocol_process_commit_in_transaction(
        self,
        *,
        business_commit_cursor: int,
        receipt: ProtocolEventReceipt,
        order_id: str,
        operation: str,
        original_actor: str,
        idempotency_key: str,
    ) -> None:
        """Merge a nested business commit and process receipt before COMMIT.

        The in-memory World rewrites its one nested settlement/dispatch/refund
        commit into one ``process_protocol_event`` commit.  SQLite must expose
        the identical replay surface: the receipt and business effect share the
        same database transaction, so recording them as separate or omitting
        the receipt would make a valid durable state impossible to replay.
        """

        rows = self._conn.execute(
            "SELECT sequence, commit_id, commit_json FROM world_commit_records "
            "WHERE sequence >= ? ORDER BY sequence",
            (business_commit_cursor,),
        ).fetchall()
        if len(rows) != 1:
            raise WorldError(
                "registered protocol operation did not produce one World commit"
            )
        raw = rows[0]
        business_commit = _world_commit_from_json(str(raw["commit_json"]))
        receipt_write = TableWrite(
            table="protocol_receipts",
            key=receipt.receipt_id,
            op="create",
            before=None,
            after=receipt,
        )
        combined_writes = (*business_commit.table_writes, receipt_write)
        invariants = (
            *business_commit.invariants_held,
            "platform-events-mediated",
            "recipient-identity-bound",
            "fresh-event-precondition",
            f"registered-operation:{operation}",
            "committed-effect-reference",
            "business-and-receipt-atomic",
            "append-only",
        )
        combined = replace(
            business_commit,
            operation="process_protocol_event",
            authority_action="world.process_protocol_event",
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            subject_id=order_id,
            table_writes=combined_writes,
            invariants_held=invariants,
        )
        updated = self._conn.execute(
            "UPDATE world_commit_records SET commit_json = ? "
            "WHERE sequence = ? AND commit_id = ?",
            (
                _world_commit_to_json(combined),
                int(raw["sequence"]),
                str(raw["commit_id"]),
            ),
        )
        if updated.rowcount != 1:
            raise WorldError("protocol process World commit identity changed")

    def seed_order_state_revisions(self, values: Any) -> None:
        if not isinstance(values, dict):
            raise ValueError("order_state_revisions seed must be a mapping")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = {
                    str(row["order_id"])
                    for row in self._conn.execute("SELECT order_id FROM orders")
                }
                for raw_order_id, revision in values.items():
                    order_id = str(raw_order_id)
                    if order_id not in existing:
                        raise ValueError(
                            "order_state_revisions references a missing order: "
                            f"{order_id!r}"
                        )
                    if (
                        isinstance(revision, bool)
                        or not isinstance(revision, int)
                        or revision < 1
                    ):
                        raise ValueError(
                            "order state revisions must be positive integers"
                        )
                    self._conn.execute(
                        """
                        INSERT INTO order_state_revisions(order_id, revision)
                        VALUES (?, ?)
                        ON CONFLICT(order_id) DO UPDATE SET revision=excluded.revision
                        """,
                        (order_id, revision),
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _order_revision_for_existing(self, order_id: str) -> int:
        row = self._conn.execute(
            "SELECT revision FROM order_state_revisions WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        if row is not None:
            return int(row["revision"])
        self._conn.execute(
            "INSERT INTO order_state_revisions(order_id, revision) VALUES (?, 1)",
            (order_id,),
        )
        return 1

    def _protocol_events_internal(
        self,
        binding_digest: str,
    ) -> tuple[ProtocolEvent, ...]:
        rows = self._conn.execute(
            """
            SELECT * FROM protocol_events
            WHERE binding_digest = ?
            ORDER BY event_sequence, event_id
            """,
            (binding_digest,),
        ).fetchall()
        return tuple(
            cast(ProtocolEvent, _row_to_value("protocol_events", row))
            for row in rows
        )

    def _protocol_receipts_internal(
        self,
        binding_digest: str,
        order_id: str,
    ) -> tuple[ProtocolEventReceipt, ...]:
        rows = self._conn.execute(
            """
            SELECT * FROM protocol_receipts
            WHERE binding_digest = ? AND order_id = ?
            ORDER BY logical_tick, receipt_id
            """,
            (binding_digest, order_id),
        ).fetchall()
        return tuple(
            cast(ProtocolEventReceipt, _row_to_value("protocol_receipts", row))
            for row in rows
        )

    def validate_persisted_protocol_records(self) -> None:
        """Structurally replay every persisted event and receipt stream."""

        with self._lock:
            bindings = [
                str(row["binding_digest"])
                for row in self._conn.execute(
                    "SELECT DISTINCT binding_digest FROM protocol_events "
                    "ORDER BY binding_digest"
                )
            ]
            for binding_digest in bindings:
                events = self._protocol_events_internal(binding_digest)
                if not events:
                    continue
                binding = events[0].binding
                replay_protocol_events(binding, events)
                receipts = self._protocol_receipts_internal(
                    binding_digest,
                    binding.order_id,
                )
                replay_protocol_receipts(binding, events, receipts)
            orphan = self._conn.execute(
                """
                SELECT r.receipt_id
                FROM protocol_receipts AS r
                LEFT JOIN protocol_events AS e
                  ON e.binding_digest = r.binding_digest
                WHERE e.binding_digest IS NULL
                ORDER BY r.receipt_id
                LIMIT 1
                """
            ).fetchone()
            if orphan is not None:
                raise ProtocolEventSchemaError(
                    "persisted protocol receipt has no event stream: "
                    f"{orphan['receipt_id']!r}"
                )

    def validate_persisted_negotiations(self) -> None:
        """Replay every persisted negotiation and match its thread exactly."""

        with self._lock:
            negotiation_ids = [
                str(row["negotiation_id"])
                for row in self._conn.execute(
                    "SELECT negotiation_id FROM negotiation_threads "
                    "UNION SELECT negotiation_id FROM negotiation_events "
                    "ORDER BY negotiation_id"
                )
            ]
            for negotiation_id in negotiation_ids:
                events = tuple(
                    cast(
                        NegotiationEvent,
                        _row_to_value("negotiation_events", row),
                    )
                    for row in self._conn.execute(
                        "SELECT * FROM negotiation_events "
                        "WHERE negotiation_id = ? ORDER BY sequence_no, event_id",
                        (negotiation_id,),
                    )
                )
                thread = cast(
                    "NegotiationThread | None",
                    self._fetch_value("negotiation_threads", negotiation_id),
                )
                if not events or thread is None:
                    raise NegotiationSchemaError(
                        "persisted negotiation requires events and thread"
                    )
                first = events[0]
                binding = NegotiationBinding(
                    negotiation_id=first.negotiation_id,
                    buyer_id=first.buyer_id,
                    merchant_id=first.merchant_id,
                    offer_id=first.offer_id,
                    sku_id=first.sku_id,
                    listing_digest=first.listing_digest,
                    listing_revision=first.listing_revision,
                    currency=first.currency,
                    qty=first.qty,
                    max_rounds=first.max_rounds,
                    opened_at_tick=first.opened_at_tick,
                    expires_at_tick=first.expires_at_tick,
                )
                if replay_negotiation_events(binding, events).thread != thread:
                    raise NegotiationSchemaError(
                        "persisted negotiation thread differs from strict replay"
                    )

    def validate_persisted_evidence_contracts(self) -> None:
        """Validate persisted authority rows using the shared World rules."""

        with self._lock:
            catalog = CatalogTable()
            evidence = EvidenceRecordTable()
            authorities = MandateAuthorityTable()
            revisions = MandateRevisionTable()
            claims = ListingClaimTable()
            operations = AuthorityOperationTable()
            negotiation_events = NegotiationEventTable()
            pricing_policies = PricingPolicyRevisionTable()
            cart_quote_requests = PersistentCartQuoteRequestTable()
            cart_quotes = PersistentCartQuoteTable()
            order_groups = OrderGroupTable()
            ledger = LedgerTable()
            governance_policies = GovernancePolicyTable()
            governance_records = GovernanceRecordTable()
            materialized: tuple[tuple[str, Table[Any, Any]], ...] = (
                ("catalog", catalog),
                ("evidence_records", evidence),
                ("mandate_authorities", authorities),
                ("mandate_revisions", revisions),
                ("listing_claims", claims),
                ("authority_operations", operations),
                ("negotiation_events", negotiation_events),
                ("pricing_policy_revisions", pricing_policies),
                ("persistent_cart_quote_requests", cart_quote_requests),
                ("persistent_cart_quotes", cart_quotes),
                ("order_groups", order_groups),
                ("ledger", ledger),
                ("governance_policies", governance_policies),
                ("governance_records", governance_records),
            )
            for table_name, table in materialized:
                persisted_rows = list(self.iter_table(table_name))
                if table_name == "governance_policies":
                    persisted_rows.sort(
                        key=lambda item: (
                            item[1].kind,
                            item[1].stable_id,
                            item[1].revision,
                        )
                    )
                elif table_name == "governance_records":
                    persisted_rows.sort(
                        key=lambda item: (
                            item[1].kind,
                            item[1].stable_id,
                            item[1].version,
                        )
                    )
                for key, value in persisted_rows:
                    table.write(str(key), value)
            _validate_seeded_evidence_contracts(
                catalog=catalog,
                evidence=evidence,
                authorities=authorities,
                revisions=revisions,
                claims=claims,
                operations=operations,
                negotiation_events=negotiation_events,
                pricing_policies=pricing_policies,
                cart_quote_requests=cart_quote_requests,
                cart_quotes=cart_quotes,
                order_groups=order_groups,
                ledger=ledger,
                governance_policies=governance_policies,
                governance_records=governance_records,
                logical_time=self._clock_value(),
            )

    def validate_persisted_governance(self) -> None:
        """Fail startup on corrupt governance rows, chains, or operation joins.

        This validation is intentionally governance-scoped.  Calling the
        broader seed validator during database construction would make this
        subsystem depend on unrelated authority-operation kinds.  The typed
        table implementations still provide the shared digest, exact-wire,
        canonical-key, and append-only-chain rules used by the memory World.
        """

        with self._lock:
            policy_table = GovernancePolicyTable()
            record_table = GovernanceRecordTable()
            specifications: tuple[
                tuple[str, str, Table[Any, Any]], ...
            ] = (
                ("governance_policies", "revision", policy_table),
                ("governance_records", "version", record_table),
            )
            for table_name, sequence_column, target in specifications:
                rows = self._conn.execute(
                    f"SELECT * FROM {table_name} "
                    f"ORDER BY record_kind, stable_id, {sequence_column}"
                ).fetchall()
                for row in rows:
                    envelope = _row_to_value(table_name, row)
                    physical = (
                        str(row["envelope_key"]),
                        str(row["record_kind"]),
                        str(row["stable_id"]),
                        int(row[sequence_column]),
                        str(row["service_actor"]),
                        str(row["original_actor"]),
                        int(row["logical_tick"]),
                        str(row["envelope_digest"]),
                    )
                    semantic = (
                        governance_envelope_key(envelope),
                        envelope.kind,
                        envelope.stable_id,
                        getattr(envelope, sequence_column),
                        envelope.service_actor,
                        envelope.original_actor,
                        envelope.logical_tick,
                        envelope.envelope_digest,
                    )
                    if physical != semantic:
                        raise WorldError(
                            f"{table_name} protected columns differ from sealed envelope"
                        )
                    if envelope.logical_tick > self._clock_value():
                        raise LogicalTimeError(
                            "governance envelope cannot be after World logical time"
                        )
                    target.write(str(row["envelope_key"]), envelope)

            operation_table = AuthorityOperationTable()
            for key, operation in self.iter_table("authority_operations"):
                operation_table.write(str(key), operation)
                if operation.outcome_table not in {
                    "governance_policies",
                    "governance_records",
                }:
                    continue
                allowed_scope = operation.scope in _GOVERNANCE_AUTHORITY_SCOPES or (
                    operation.scope.startswith("apply_governance_intent:")
                )
                if not allowed_scope:
                    raise WorldError(
                        "governance authority operation has an unsupported scope"
                    )
                if (
                    operation.scope == "publish_governance_policy"
                ) != (operation.outcome_table == "governance_policies"):
                    raise WorldError(
                        "governance authority scope and outcome collection differ"
                    )
                target = (
                    policy_table
                    if operation.outcome_table == "governance_policies"
                    else record_table
                )
                outcome = target.read(operation.outcome_key, caller="world")
                if outcome is None:
                    raise WorldError(
                        "governance authority operation references a missing outcome"
                    )
                if (
                    operation.actor_id != outcome.original_actor
                    or operation.idempotency_key != outcome.idempotency_key
                    or operation.outcome_listing is not None
                    or not _is_lower_sha256(operation.request_fingerprint)
                ):
                    raise WorldError(
                        "governance authority operation binding is invalid"
                    )

    def write(self, table: str, key: Any, value: Any, *, by_action: str) -> None:
        with self._lock:
            if table in _AUTHORITY_TABLES and str(key) != str(_row_key(table, value)):
                raise IdempotencyConflict(
                    f"{table} key does not match the sealed record identity"
                )
            before = self._fetch_value(table, str(key))
            if table in {
                "ledger",
                "fulfillments",
                "exchanges",
                "protocol_events",
                "protocol_receipts",
                "negotiation_events",
                "negotiation_threads",
                "pricing_policy_revisions",
                "persistent_cart_quote_requests",
                "persistent_cart_quotes",
                "supply_purchase_authorities",
                "order_groups",
                "evidence_records",
                "mandate_authorities",
                "mandate_revisions",
                "authority_operations",
            } and before is not None:
                raise WriteNotAuthorized(f"{table} row already exists: {key}")
            with self._conn:
                _upsert(self._conn, table, value)
                _record_mutation(
                    self._conn,
                    table=table,
                    key=str(key),
                    action=by_action,
                    before=before,
                    after=value,
                )

    def read_supply_state(self, sku_id: SkuId, *, caller: str) -> SupplyState:
        with self._lock:
            inventory = cast(
                "InventoryRow | None", self._fetch_value("inventory", str(sku_id))
            )
            listing = cast(
                "Listing | None", self._fetch_value("catalog", str(sku_id))
            )
            if inventory is None or listing is None:
                raise OutOfStock(f"unknown supply sku {sku_id}")
            role = caller.split(":", 1)[0]
            if role not in {"platform", "runtime"} and caller != str(inventory.merchant_id):
                raise LifecycleAuthorizationError(
                    f"actor {caller!r} cannot read exact supply for {sku_id}"
                )
            if inventory.merchant_id != listing.merchant_id:
                raise WorldError(f"catalog/inventory owner mismatch for {sku_id}")
            return _supply_projection(listing, inventory)

    def issue_supply_purchase_authorities(
        self,
        sku_ids: tuple[str, ...],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
        ttl_ticks: int = DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    ) -> tuple[SupplyPurchaseAuthority, ...]:
        """Durably mint one atomic authority batch from current supply."""

        if by_actor != "platform:supply":
            raise WriteNotAuthorized(
                "only platform:supply may issue supply purchase authorities"
            )
        if not isinstance(original_actor, str) or not original_actor.startswith(
            "buyer:"
        ):
            raise LifecycleAuthorizationError(
                "supply purchase authority requires an authenticated buyer"
            )
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise IdempotencyConflict(
                "supply purchase authority requires an idempotency key"
            )
        try:
            validate_supply_purchase_authority_ttl_ticks(ttl_ticks)
        except SchemaError as exc:
            raise WorldError(str(exc)) from exc
        if (
            not isinstance(sku_ids, tuple)
            or not sku_ids
            or len(sku_ids) > 64
            or any(not isinstance(value, str) or not value for value in sku_ids)
            or len(set(sku_ids)) != len(sku_ids)
        ):
            raise WorldError(
                "supply purchase authority requires 1..64 unique sku ids"
            )
        authority_ids = tuple(
            supply_purchase_authority_id(
                buyer_id=original_actor,
                request_idempotency_key=idempotency_key,
                sku_id=sku_id,
            )
            for sku_id in sku_ids
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = tuple(
                    self._fetch_value("supply_purchase_authorities", authority_id)
                    for authority_id in authority_ids
                )
                if any(row is not None for row in existing):
                    if not all(row is not None for row in existing):
                        raise IdempotencyConflict(
                            "supply authority request has an incomplete durable replay"
                        )
                    replay = cast(tuple[SupplyPurchaseAuthority, ...], existing)
                    for row, sku_id in zip(replay, sku_ids, strict=True):
                        validate_supply_purchase_authority(row)
                        if (
                            row.buyer_id != original_actor
                            or row.sku_id != sku_id
                            or row.request_idempotency_key != idempotency_key
                            or row.expires_at_tick - row.issued_at_tick != ttl_ticks
                        ):
                            raise IdempotencyConflict(
                                "supply authority idempotency key was reused"
                            )
                    self._conn.commit()
                    return replay

                issued_at_tick = self._clock_value()
                authorities: list[SupplyPurchaseAuthority] = []
                for sku_id in sku_ids:
                    listing = cast(
                        "Listing | None",
                        self._fetch_value("catalog", sku_id),
                    )
                    inventory = cast(
                        "InventoryRow | None",
                        self._fetch_value("inventory", sku_id),
                    )
                    if listing is None or inventory is None:
                        raise OutOfStock(f"unknown supply sku {sku_id}")
                    if listing.merchant_id != inventory.merchant_id:
                        raise WorldError(
                            f"catalog/inventory owner mismatch for {sku_id}"
                        )
                    state = _supply_projection(listing, inventory)
                    authorities.append(build_supply_purchase_authority(
                        buyer_id=original_actor,
                        merchant_id=str(listing.merchant_id),
                        sku_id=sku_id,
                        unit_price_cents=state.unit_price_cents,
                        currency=listing.list_price.currency,
                        available_qty=state.available_qty,
                        supply_version=state.version,
                        issued_at_tick=issued_at_tick,
                        expires_at_tick=issued_at_tick + ttl_ticks,
                        request_idempotency_key=idempotency_key,
                    ))
                writes: list[TableWrite] = []
                for authority in authorities:
                    _upsert(self._conn, "supply_purchase_authorities", authority)
                    _record_mutation(
                        self._conn,
                        table="supply_purchase_authorities",
                        key=authority.authority_id,
                        action="world.issue_supply_purchase_authority",
                        before=None,
                        after=authority,
                    )
                    writes.append(TableWrite(
                        table="supply_purchase_authorities",
                        key=authority.authority_id,
                        op="create",
                        before=None,
                        after=authority,
                    ))
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="issue_supply_purchase_authority",
                    authority_action="world.issue_supply_purchase_authority",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    subject_id=authorities[0].authority_id,
                    table_writes=tuple(writes),
                    invariants_held=(
                        "world-derived-listing-and-supply",
                        "buyer-scoped",
                        "sealed",
                        "bounded-expiry",
                        "atomic-batch",
                        "idempotent",
                        "market-clock-neutral",
                    ),
                    request_fingerprint=canonical_digest({
                        "sku_ids": list(sku_ids),
                        "ttl_ticks": ttl_ticks,
                    }),
                )
                self._conn.commit()
                return tuple(authorities)
            except BaseException:
                self._conn.rollback()
                raise

    def read_supply_purchase_authority(
        self,
        authority_id: str,
        *,
        caller: str,
    ) -> SupplyPurchaseAuthority | None:
        with self._lock:
            row = cast(
                "SupplyPurchaseAuthority | None",
                self._fetch_value("supply_purchase_authorities", authority_id),
            )
            if row is None:
                return None
            if caller.split(":", 1)[0] in {"platform", "runtime"}:
                return row
            return row if caller == row.buyer_id else None

    def apply_settlement_reputation(
        self,
        *,
        merchant_id: AgentId,
        order_id: OrderId,
        txn_id: TxnId,
        by_actor: str,
        original_actor: str,
        source_request_id: str,
        idempotency_key: str,
    ) -> ReputationScore:
        """Durably apply one reputation sample per settled transaction."""

        if by_actor != "platform:reputation":
            raise WriteNotAuthorized(
                "only platform:reputation may apply settlement reputation"
            )
        if original_actor != "platform:psp":
            raise WriteNotAuthorized(
                "settlement reputation must originate from platform:psp"
            )
        if not source_request_id or not idempotency_key:
            raise IdempotencyConflict(
                "settlement reputation requires source request and idempotency keys"
            )
        source = ReputationSettlementSource(
            source_actor=original_actor,
            source_request_id=source_request_id,
            source_idempotency_key=idempotency_key,
        )

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                order = cast(
                    "Order | None", self._fetch_value("orders", str(order_id))
                )
                receipt = cast(
                    "Receipt | None", self._fetch_value("ledger", str(txn_id))
                )
                _validate_reputation_settlement_identity(
                    merchant_id=merchant_id,
                    order_id=order_id,
                    txn_id=txn_id,
                    order=order,
                    receipt=receipt,
                )

                source_row = self._conn.execute(
                    """
                    SELECT s.source_request_id, e.event_json
                    FROM reputation_settlement_sources AS s
                    JOIN reputation_settlements AS e ON e.event_id = s.event_id
                    WHERE s.source_actor = ? AND s.source_idempotency_key = ?
                    """,
                    (original_actor, idempotency_key),
                ).fetchone()
                if source_row is not None:
                    existing = _reputation_settlement_from_json(
                        str(source_row["event_json"])
                    )
                    if (
                        existing.order_id != order_id
                        or existing.txn_id != txn_id
                        or existing.merchant_id != merchant_id
                    ):
                        raise IdempotencyConflict(
                            "reputation source idempotency key was reused for a different settlement"
                        )
                    if str(source_row["source_request_id"]) != source_request_id:
                        raise IdempotencyConflict(
                            "reputation source request changed under an existing key"
                        )
                    self._conn.commit()
                    return existing.outcome

                event_row = self._conn.execute(
                    """
                    SELECT event_json FROM reputation_settlements WHERE txn_id = ?
                    """,
                    (str(txn_id),),
                ).fetchone()
                if event_row is not None:
                    existing = _reputation_settlement_from_json(
                        str(event_row["event_json"])
                    )
                    if (
                        existing.order_id != order_id
                        or existing.merchant_id != merchant_id
                    ):
                        raise IdempotencyConflict(
                            "reputation transaction identity conflicts with persisted event"
                        )
                    updated = replace(
                        existing,
                        sources=tuple(
                            sorted(
                                (*existing.sources, source),
                                key=lambda item: (
                                    item.source_actor,
                                    item.source_idempotency_key,
                                    item.source_request_id,
                                ),
                            )
                        ),
                    )
                    _upsert(self._conn, "reputation_settlements", updated)
                    _record_mutation(
                        self._conn,
                        table="reputation_settlements",
                        key=existing.event_id,
                        action="world.update_reputation",
                        before=existing,
                        after=updated,
                    )
                    self._conn.commit()
                    return existing.outcome

                prior = cast(
                    "ReputationScore | None",
                    self._fetch_value("reputation", str(merchant_id)),
                )
                if prior is None:
                    raise WorldError(
                        f"settlement merchant {merchant_id} has no tracked reputation row"
                    )
                settled = int(prior.n_settled) + 1
                outcome = ReputationScore(
                    merchant_id=prior.merchant_id,
                    rolling_avg=(
                        float(prior.rolling_avg) * int(prior.n_settled) + 5.0
                    )
                    / settled,
                    n_settled=settled,
                    n_disputed=int(prior.n_disputed),
                )
                event = ReputationSettlement(
                    event_id=_reputation_settlement_event_id(txn_id),
                    order_id=order_id,
                    txn_id=txn_id,
                    merchant_id=merchant_id,
                    sources=(source,),
                    outcome=outcome,
                )
                _upsert(self._conn, "reputation", outcome)
                _upsert(self._conn, "reputation_settlements", event)
                _record_mutation(
                    self._conn,
                    table="reputation",
                    key=str(merchant_id),
                    action="world.update_reputation",
                    before=prior,
                    after=outcome,
                )
                _record_mutation(
                    self._conn,
                    table="reputation_settlements",
                    key=event.event_id,
                    action="world.update_reputation",
                    before=None,
                    after=event,
                )
                self._conn.commit()
                return outcome
            except BaseException:
                self._conn.rollback()
                raise

    def create_search_session(
        self,
        *,
        session: SearchSession,
        by_actor: str,
        idempotency_key: str,
    ) -> SearchSession:
        if by_actor != "platform:aggregator":
            raise WriteNotAuthorized(
                "only platform:aggregator may create search sessions"
            )
        validate_search_session(session)
        if idempotency_key != session.search_idempotency_key:
            raise IdempotencyConflict(
                "search envelope and session idempotency keys differ"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                prior = self._conn.execute(
                    """
                    SELECT session_json FROM search_sessions
                    WHERE buyer_id = ? AND search_idempotency_key = ?
                    """,
                    (session.buyer_id, session.search_idempotency_key),
                ).fetchone()
                if prior is not None:
                    existing = coerce_search_session(json.loads(prior["session_json"]))
                    if existing != session:
                        raise IdempotencyConflict(
                            "search idempotency key was reused for a different request"
                        )
                    self._conn.commit()
                    return existing
                self._validate_search_session_against_store(session)
                encoded = _canonical_json(search_session_to_wire(session))
                self._conn.execute(
                    """
                    INSERT INTO search_sessions
                      (session_id, buyer_id, search_idempotency_key, session_json,
                       created_at)
                    VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        session.session_id,
                        session.buyer_id,
                        session.search_idempotency_key,
                        encoded,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO search_session_offers
                      (session_id, buyer_id, offer_id, expires_at_tick)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            session.session_id,
                            session.buyer_id,
                            offer.offer_id,
                            session.expires_at_tick,
                        )
                        for offer in session.offers
                    ),
                )
                _record_mutation(
                    self._conn,
                    table="search_sessions",
                    key=session.session_id,
                    action="world.create_search_session",
                    before=None,
                    after=session,
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="create_search_session",
                    authority_action="world.create_search_session",
                    actor_id=by_actor,
                    idempotency_key=idempotency_key,
                    subject_id=session.session_id,
                    table_writes=(
                        TableWrite(
                            table="search_sessions",
                            key=session.session_id,
                            op="create",
                            before=None,
                            after=session,
                        ),
                    ),
                    invariants_held=(
                        "server-authored-session",
                        "catalog-revision-bound",
                        "inventory-revision-bound",
                        "buyer-scoped-idempotency",
                    ),
                )
                self._conn.commit()
                return session
            except BaseException:
                self._conn.rollback()
                raise

    def resolve_search_session(
        self,
        *,
        buyer_id: str,
        offer_id: str,
        caller: str,
        unique_only: bool = True,
        current_only: bool = True,
    ) -> SearchSession | None:
        if not (caller.startswith("platform:") or caller.startswith("runtime:")):
            return None
        with self._lock:
            now = self._clock_value()
            where = "" if not current_only else "AND o.expires_at_tick > ?"
            params: tuple[Any, ...] = (
                (buyer_id, offer_id)
                if not current_only
                else (buyer_id, offer_id, now)
            )
            rows = self._conn.execute(
                f"""
                SELECT s.session_json
                FROM search_session_offers AS o
                JOIN search_sessions AS s ON s.session_id = o.session_id
                WHERE o.buyer_id = ? AND o.offer_id = ? {where}
                ORDER BY o.session_id
                """,
                params,
            ).fetchall()
            matches = [
                coerce_search_session(json.loads(row["session_json"]))
                for row in rows
            ]
            if not matches or (unique_only and len(matches) != 1):
                return None
            return matches[0]

    def issue_match_certificate(
        self,
        *,
        acceptance: MatchAcceptance,
        by_actor: str,
        original_actor: str,
    ) -> MatchCertificate:
        if by_actor != "platform:aggregator":
            raise WriteNotAuthorized(
                "only platform:aggregator may issue match certificates"
            )
        if original_actor != acceptance.buyer_id:
            raise MatchAcceptanceRejected(
                "acceptance buyer does not match original actor"
            )
        acceptance_key = _match_acceptance_key(
            acceptance.buyer_id, acceptance.idempotency_key
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                session_row = self._conn.execute(
                    "SELECT session_json FROM search_sessions WHERE session_id = ?",
                    (acceptance.session_id,),
                ).fetchone()
                if session_row is None:
                    raise MatchAcceptanceRejected(
                        "unknown persisted search session"
                    )
                session = coerce_search_session(json.loads(session_row["session_json"]))

                prior_acceptance_row = self._conn.execute(
                    "SELECT acceptance_json FROM match_acceptances WHERE acceptance_key = ?",
                    (acceptance_key,),
                ).fetchone()
                existing_certificate: MatchCertificate | None = None
                if prior_acceptance_row is not None:
                    prior_acceptance = coerce_match_acceptance(
                        json.loads(prior_acceptance_row["acceptance_json"])
                    )
                    if prior_acceptance != acceptance:
                        raise IdempotencyConflict(
                            "match acceptance changed under an existing idempotency key"
                        )
                    cert_row = self._conn.execute(
                        """
                        SELECT certificate_json FROM match_certificates
                        WHERE acceptance_digest = ?
                        """,
                        (acceptance.acceptance_digest,),
                    ).fetchone()
                    if cert_row is None:
                        raise WorldError(
                            "persisted match acceptance has no certificate"
                        )
                    existing_certificate = coerce_match_certificate(
                        json.loads(cert_row["certificate_json"])
                    )

                listing, inventory = self._matching_rows(acceptance.sku_id)
                try:
                    certificate = build_match_certificate(
                        session,
                        acceptance,
                        current_tick=self._clock_value(),
                        current_catalog_revision=_catalog_revision(listing),
                        current_inventory_revision=inventory.version,
                        existing_certificate=existing_certificate,
                    )
                except MatchValidationError as exc:
                    raise MatchAcceptanceRejected(str(exc)) from exc
                if existing_certificate is not None:
                    self._conn.commit()
                    return existing_certificate

                self._conn.execute(
                    """
                    INSERT INTO match_acceptances
                      (acceptance_key, buyer_id, idempotency_key,
                       acceptance_digest, acceptance_json, created_at)
                    VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        acceptance_key,
                        acceptance.buyer_id,
                        acceptance.idempotency_key,
                        acceptance.acceptance_digest,
                        _canonical_json(match_acceptance_to_wire(acceptance)),
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO match_certificates
                      (cert_id, acceptance_digest, buyer_id, order_id,
                       expires_at_tick, certificate_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?,
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        certificate.cert_id,
                        certificate.acceptance_digest,
                        certificate.buyer_id,
                        certificate.order_id,
                        certificate.expires_at_tick,
                        _canonical_json(match_certificate_to_wire(certificate)),
                    ),
                )
                _record_mutation(
                    self._conn,
                    table="match_acceptances",
                    key=acceptance_key,
                    action="world.issue_match_certificate",
                    before=None,
                    after=acceptance,
                )
                _record_mutation(
                    self._conn,
                    table="match_certificates",
                    key=certificate.cert_id,
                    action="world.issue_match_certificate",
                    before=None,
                    after=certificate,
                )
                self._conn.commit()
                return certificate
            except BaseException:
                self._conn.rollback()
                raise

    def resolve_match_certificate(
        self,
        *,
        buyer_id: str,
        order_id: str,
        caller: str,
        current_only: bool = True,
    ) -> MatchCertificate | None:
        if not (caller.startswith("platform:") or caller.startswith("runtime:")):
            return None
        with self._lock:
            now = self._clock_value()
            if current_only:
                rows = self._conn.execute(
                    """
                    SELECT certificate_json FROM match_certificates
                    WHERE buyer_id = ? AND order_id = ? AND expires_at_tick > ?
                    ORDER BY cert_id
                    """,
                    (buyer_id, order_id, now),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT certificate_json FROM match_certificates
                    WHERE buyer_id = ? AND order_id = ?
                    ORDER BY cert_id
                    """,
                    (buyer_id, order_id),
                ).fetchall()
            matches = [
                coerce_match_certificate(json.loads(row["certificate_json"]))
                for row in rows
            ]
            return matches[0] if len(matches) == 1 else None

    def _matching_rows(self, sku_id: str) -> tuple[Listing, InventoryRow]:
        listing = cast("Listing | None", self._fetch_value("catalog", sku_id))
        inventory = cast("InventoryRow | None", self._fetch_value("inventory", sku_id))
        if listing is None or inventory is None:
            raise MatchValidationError("matched SKU lacks catalog or inventory state")
        if str(listing.merchant_id) != str(inventory.merchant_id):
            raise MatchValidationError("catalog and inventory merchant mismatch")
        return listing, inventory

    def _validate_search_session_against_store(self, session: SearchSession) -> None:
        now = self._clock_value()
        if session.issued_at_tick != now:
            raise MatchValidationError("search session issued_at_tick is not current")
        if session.expires_at_tick <= now:
            raise MatchValidationError("search session is already expired")
        for offer in session.offers:
            listing, inventory = self._matching_rows(offer.sku_id)
            if offer.session_id != session.session_id:
                raise MatchValidationError("offer session_id mismatch")
            if offer.issued_at_tick != session.issued_at_tick:
                raise MatchValidationError("offer issued_at_tick mismatch")
            if offer.expires_at_tick > session.expires_at_tick:
                raise MatchValidationError("offer expiry exceeds session expiry")
            if offer.merchant_id != str(listing.merchant_id):
                raise MatchValidationError("offer merchant does not own listing")
            if offer.unit_price_cents != _money_cents(listing.list_price):
                raise MatchValidationError("offer price differs from catalog")
            if offer.currency != listing.list_price.currency:
                raise MatchValidationError("offer currency differs from catalog")
            if offer.catalog_revision != _catalog_revision(listing):
                raise MatchValidationError("offer catalog revision is stale")
            if offer.inventory_revision != inventory.version:
                raise MatchValidationError("offer inventory revision is stale")
            if offer.qty > _available_qty(inventory):
                raise MatchValidationError("offer quantity exceeds available inventory")

    def apply_supply_event(
        self,
        *,
        sku_id: SkuId,
        qty_delta: int,
        eta_day: int | None,
        unit_price_cents: int | None,
        expected_version: int | None,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> SupplyState:
        """SQLite equivalent of ``World.apply_supply_event``."""

        if by_actor != "platform:supply":
            raise LifecycleAuthorizationError(
                f"only platform:supply may apply supply events; got {by_actor!r}"
            )
        if isinstance(qty_delta, bool) or not isinstance(qty_delta, int):
            raise ValueError("qty_delta must be an integer")
        if eta_day is not None and (
            isinstance(eta_day, bool) or not isinstance(eta_day, int) or eta_day < 0
        ):
            raise ValueError("eta_day must be a non-negative integer")
        if unit_price_cents is not None and (
            isinstance(unit_price_cents, bool)
            or not isinstance(unit_price_cents, int)
            or unit_price_cents <= 0
        ):
            raise ValueError("unit_price_cents must be a positive integer")
        if expected_version is not None and (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version <= 0
        ):
            raise ValueError("expected_version must be a positive integer")
        if not idempotency_key.strip():
            raise ValueError("supply event idempotency key must not be blank")

        fingerprint = _canonical_json(
            {
                "sku_id": str(sku_id),
                "qty_delta": qty_delta,
                "eta_day": eta_day,
                "unit_price_cents": unit_price_cents,
                "expected_version": expected_version,
                "original_actor": original_actor,
            }
        )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._conn.execute(
                    """
                    SELECT request_fingerprint, outcome_json
                    FROM supply_event_records
                    WHERE scope = ? AND idempotency_key = ?
                    """,
                    (by_actor, idempotency_key),
                ).fetchone()
                if replay is not None:
                    if replay["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was reused for another supply event"
                        )
                    value = json.loads(str(replay["outcome_json"]))
                    self._conn.commit()
                    return SupplyState(
                        sku_id=SkuId(str(value["sku_id"])),
                        merchant_id=AgentId(str(value["merchant_id"])),
                        available_qty=int(value["available_qty"]),
                        reserved_qty=int(value["reserved_qty"]),
                        eta_day=int(value["eta_day"]),
                        unit_price_cents=int(value["unit_price_cents"]),
                        version=int(value["version"]),
                    )

                inventory = cast(
                    "InventoryRow | None",
                    self._fetch_value("inventory", str(sku_id)),
                )
                listing = cast(
                    "Listing | None", self._fetch_value("catalog", str(sku_id))
                )
                if inventory is None or listing is None:
                    raise OutOfStock(f"unknown supply sku {sku_id}")
                if inventory.merchant_id != listing.merchant_id:
                    raise WorldError(f"catalog/inventory owner mismatch for {sku_id}")
                if original_actor not in {str(inventory.merchant_id), "runtime:supply"}:
                    raise LifecycleAuthorizationError(
                        f"actor {original_actor!r} does not own supply {sku_id}"
                    )
                if expected_version is not None and inventory.version != expected_version:
                    raise IdempotencyConflict(
                        f"stale supply version {expected_version}; current is {inventory.version}"
                    )
                next_qty = inventory.qty_available + qty_delta
                if next_qty < inventory.qty_reserved or next_qty < 0:
                    raise OutOfStock(
                        f"supply event would reduce {sku_id} below reserved inventory"
                    )
                projected_inventory = replace(
                    inventory,
                    qty_available=next_qty,
                    eta_day=inventory.eta_day if eta_day is None else eta_day,
                    version=inventory.version + 1,
                )
                listing_after = listing
                if unit_price_cents is not None:
                    listing_after = _revisioned_listing(
                        replace(
                            listing,
                            list_price=Money(
                                Decimal(unit_price_cents) / Decimal(100),
                                listing.list_price.currency,
                            ),
                        ),
                        listing,
                    )
                outcome = _supply_projection(listing_after, projected_inventory)
                durable_record = SupplyEventRecord(
                    scope=by_actor,
                    idempotency_key=idempotency_key,
                    sku_id=sku_id,
                    qty_delta=qty_delta,
                    eta_day=eta_day,
                    unit_price_cents=unit_price_cents,
                    expected_version=expected_version,
                    original_actor=original_actor,
                    outcome=outcome,
                )
                inventory_after = replace(
                    projected_inventory,
                    supply_events=inventory.supply_events + (durable_record,),
                )
                writes: list[tuple[str, str, Any, Any]] = [
                    ("inventory", str(sku_id), inventory, inventory_after)
                ]
                if listing_after != listing:
                    writes.append(("catalog", str(sku_id), listing, listing_after))
                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.apply_supply_event",
                        before=before,
                        after=after,
                    )
                self._conn.execute(
                    """
                    INSERT INTO supply_event_records
                      (scope, idempotency_key, request_fingerprint, outcome_json, created_at)
                    VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (by_actor, idempotency_key, fingerprint, _value_to_json(outcome)),
                )
                self._conn.commit()
                return outcome
            except BaseException:
                self._conn.rollback()
                raise

    def settle_order(
        self,
        *,
        order: Order,
        receipt: Receipt,
        by_role: str,
        idempotency_key: str,
        _manage_transaction: bool = True,
    ) -> Receipt:
        """Settle ``order`` as one SQLite write transaction.

        ``BEGIN IMMEDIATE`` acquires the database write reservation before any
        precondition read. This matters when multiple ``DatabaseWorld``
        instances (and therefore multiple SQLite connections) race for the same
        inventory row: the losing transaction observes the winner's committed
        reservation instead of acting on a stale snapshot.
        """
        receipt = replace(receipt, effect="charge")
        fingerprint = _idempotency_fingerprint("settle", order, receipt)
        with self._lock:
            if _manage_transaction:
                self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._idempotency_replay(
                    scope=by_role,
                    key=idempotency_key,
                    operation="settle",
                    fingerprint=fingerprint,
                )
                if replay is not None:
                    if _manage_transaction:
                        self._conn.commit()
                    return replay
                existing = self._fetch_value("orders", str(order.order_id))
                if existing is not None:
                    existing_order = cast(Order, existing)
                    validate_persisted_order_identity(existing_order, order)
                    validate_transaction_identity(
                        order, receipt, None, expected_effect="charge"
                    )
                    if existing_order.state in _ALREADY_SETTLED:
                        prior = self._receipt_for_order(order.order_id, refund=False)
                        if prior is None:
                            raise WorldError(
                                f"order {order.order_id} is {existing_order.state.value} "
                                "but has no ledger receipt — refusing to re-settle"
                            )
                        self._record_idempotency(
                            scope=by_role,
                            key=idempotency_key,
                            operation="settle",
                            fingerprint=fingerprint,
                            outcome=prior,
                        )
                        if _manage_transaction:
                            self._conn.commit()
                        return prior
                    if existing_order.state not in _SETTLEABLE:
                        raise OrderNotSettleable(
                            f"order {order.order_id} in state "
                            f"{existing_order.state.value!r} is not settleable"
                        )

                inventory_value = self._fetch_value("inventory", str(order.sku_id))
                if inventory_value is None:
                    raise OutOfStock(f"no inventory for sku {order.sku_id}")
                inventory = cast(InventoryRow, inventory_value)
                validate_transaction_identity(
                    order, receipt, inventory, expected_effect="charge"
                )
                listing = self._fetch_value("catalog", str(order.sku_id))
                validate_listing_owner(order, listing)
                payment_history = self._payment_history_unchecked(
                    str(order.order_id)
                )
                previous_payment = payment_history[-1] if payment_history else None
                if (
                    previous_payment is not None
                    and previous_payment.state != "authorized"
                ):
                    raise WorldError(
                        f"order {order.order_id} has payment state "
                        f"{previous_payment.state!r} before settlement"
                    )
                reservation_already_held = (
                    previous_payment is not None
                    and previous_payment.state == "authorized"
                )
                if reservation_already_held:
                    if inventory.qty_reserved < order.qty:
                        raise WorldError(
                            f"authorized payment for {order.order_id} has no "
                            "matching inventory reservation"
                        )
                elif inventory.qty_available - inventory.qty_reserved < order.qty:
                    raise OutOfStock(f"insufficient inventory for sku {order.sku_id}")
                if self._fetch("ledger", str(receipt.txn_id)) is not None:
                    raise WriteNotAuthorized(f"ledger row already exists: {receipt.txn_id}")

                settled_order = replace(order, state=OrderState.SETTLED)
                reserved_inventory = (
                    inventory
                    if reservation_already_held
                    else replace(
                        inventory,
                        qty_reserved=inventory.qty_reserved + order.qty,
                        version=inventory.version + 1,
                    )
                )
                if self._fetch("order_timelines", str(order.order_id)) is not None:
                    raise WorldError(
                        f"order {order.order_id} has timing evidence before settlement"
                    )
                event_tick = self._clock_value() + 1
                timeline = OrderTimeline(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    settled_at_tick=event_tick,
                    return_window_ticks=_captured_return_window(listing),
                )
                payment = derive_payment_capture(
                    settled_order,
                    receipt,
                    previous=previous_payment,
                    original_actor="platform:psp",
                    server_tick=event_tick,
                    idempotency_key=f"settlement-capture:{idempotency_key}",
                )
                payment_key = payment_state_key(payment)
                writes: list[tuple[str, str, Any, Any]] = [
                    (
                        "orders",
                        str(order.order_id),
                        existing,
                        settled_order,
                    ),
                    ("ledger", str(receipt.txn_id), None, receipt),
                    (
                        "order_timelines",
                        str(order.order_id),
                        None,
                        timeline,
                    ),
                    ("payment_states", payment_key, None, payment),
                ]
                if reserved_inventory != inventory:
                    writes.insert(
                        1,
                        (
                            "inventory",
                            str(order.sku_id),
                            inventory,
                            reserved_inventory,
                        ),
                    )
                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.settle_order",
                        before=before,
                        after=after,
                    )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.settle_order",
                    before=event_tick - 1,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._record_idempotency(
                    scope=by_role,
                    key=idempotency_key,
                    operation="settle",
                    fingerprint=fingerprint,
                    outcome=receipt,
                )
                commit_writes = tuple(
                    TableWrite(
                        table=table,
                        key=key,
                        op="create" if before is None else "update",
                        before=before,
                        after=after,
                    )
                    for table, key, before, after in writes
                ) + (
                    TableWrite(
                        table="logical_time",
                        key="world",
                        op="update",
                        before=event_tick - 1,
                        after=event_tick,
                    ),
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="settle",
                    authority_action="world.settle_order",
                    actor_id=by_role,
                    idempotency_key=idempotency_key,
                    subject_id=str(order.order_id),
                    table_writes=commit_writes,
                    invariants_held=(
                        "atomic",
                        "allowlist:SETTLEABLE",
                        "idempotent",
                        "world-clock",
                        "first-class-payment-captured",
                        "single-inventory-reservation",
                    ),
                    request_fingerprint=fingerprint,
                )
                if _manage_transaction:
                    self._conn.commit()
                return receipt
            except BaseException:
                if _manage_transaction:
                    self._conn.rollback()
                raise

    def settle_order_partial(
        self,
        *,
        order: Order,
        fulfilled_qty: int,
        receipt: Receipt | None,
        by_actor: str,
        idempotency_key: str,
    ) -> FulfillmentAllocation:
        """Durably allocate filled and backordered units in one transaction."""
        if receipt is not None:
            receipt = replace(receipt, effect="charge")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                _authorize_psp(by_actor, operation="allocate fulfillment")
                _validate_requested_and_fulfilled(order, fulfilled_qty)
                if order.state not in _SETTLEABLE:
                    raise OrderNotSettleable(
                        f"order {order.order_id} in state {order.state.value!r} "
                        "is not eligible for an initial fulfillment allocation"
                    )
                backordered_qty = order.qty - fulfilled_qty
                allocation = FulfillmentAllocation(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    sku_id=order.sku_id,
                    requested_qty=order.qty,
                    fulfilled_qty=fulfilled_qty,
                    backordered_qty=backordered_qty,
                    receipt_txn_id=receipt.txn_id if receipt is not None else None,
                    created_by=AgentId(by_actor),
                    idempotency_key=idempotency_key,
                )

                prior_key = self._conn.execute(
                    """
                    SELECT * FROM fulfillments
                    WHERE created_by = ? AND idempotency_key = ?
                    """,
                    (by_actor, idempotency_key),
                ).fetchone()
                if prior_key is not None:
                    prior = cast(
                        FulfillmentAllocation,
                        _row_to_value("fulfillments", prior_key),
                    )
                    prior_receipt = (
                        None
                        if prior.receipt_txn_id is None
                        else self._fetch_value(
                            "ledger",
                            str(prior.receipt_txn_id),
                        )
                    )
                    persisted_order = self._fetch_value(
                        "orders",
                        str(prior.order_id),
                    )
                    if (
                        prior == allocation
                        and prior_receipt == receipt
                        and isinstance(persisted_order, Order)
                        and _order_identity_matches(persisted_order, order)
                    ):
                        self._conn.commit()
                        return prior
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used "
                        "for a different fulfillment request"
                    )
                if self._fetch("fulfillments", str(order.order_id)) is not None:
                    raise FulfillmentNotActionable(
                        f"order {order.order_id} already has a fulfillment allocation"
                    )

                existing_value = self._fetch_value("orders", str(order.order_id))
                existing = cast("Order | None", existing_value)
                if existing is not None:
                    _validate_same_order_identity(existing, order)
                    if existing.state not in _SETTLEABLE:
                        raise OrderNotSettleable(
                            f"order {order.order_id} in state "
                            f"{existing.state.value!r} is not fulfillable"
                        )
                inventory_value = self._fetch_value(
                    "inventory",
                    str(order.sku_id),
                )
                if inventory_value is None:
                    raise OutOfStock(f"no inventory row for sku {order.sku_id}")
                inventory = cast(InventoryRow, inventory_value)
                listing = self._fetch_value("catalog", str(order.sku_id))
                if listing is None:
                    raise FulfillmentNotActionable(
                        f"listing {order.sku_id} does not exist"
                    )
                validate_listing_owner(order, listing)
                if inventory.merchant_id != order.merchant_id:
                    raise FulfillmentNotActionable(
                        "inventory merchant does not own the requested order"
                    )
                if inventory.qty_available - inventory.qty_reserved < fulfilled_qty:
                    raise OutOfStock(f"insufficient inventory for sku {order.sku_id}")
                _validate_partial_receipt(
                    order=order,
                    fulfilled_qty=fulfilled_qty,
                    receipt=receipt,
                    inventory=inventory,
                    idempotency_key=idempotency_key,
                )
                if receipt is not None and self._fetch(
                    "ledger",
                    str(receipt.txn_id),
                ) is not None:
                    raise WriteNotAuthorized(
                        f"ledger row already exists: {receipt.txn_id}"
                    )

                target_state = (
                    OrderState.BACKORDERED
                    if fulfilled_qty == 0
                    else OrderState.PARTIALLY_SETTLED
                    if backordered_qty > 0
                    else OrderState.SETTLED
                )
                persisted = replace(order, state=target_state)
                writes: list[tuple[str, str, Any, Any]] = [
                    ("orders", str(order.order_id), existing, persisted),
                ]
                if fulfilled_qty > 0:
                    reserved = replace(
                        inventory,
                        qty_reserved=inventory.qty_reserved + fulfilled_qty,
                        version=inventory.version + 1,
                    )
                    writes.append((
                        "inventory",
                        str(order.sku_id),
                        inventory,
                        reserved,
                    ))
                    assert receipt is not None
                    writes.append(("ledger", str(receipt.txn_id), None, receipt))
                writes.append((
                    "fulfillments",
                    str(order.order_id),
                    None,
                    allocation,
                ))
                event_tick: int | None = None
                captured_payment: PaymentStateRecord | None = None
                if fulfilled_qty > 0:
                    if self._fetch("order_timelines", str(order.order_id)) is not None:
                        raise WorldError(
                            f"order {order.order_id} has timing evidence before settlement"
                        )
                    event_tick = self._clock_value() + 1
                    timeline = OrderTimeline(
                        order_id=order.order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        settled_at_tick=event_tick,
                        return_window_ticks=_captured_return_window(listing),
                    )
                    writes.append((
                        "order_timelines",
                        str(order.order_id),
                        None,
                        timeline,
                    ))
                    if backordered_qty == 0:
                        assert receipt is not None
                        captured_payment = derive_payment_capture(
                            persisted,
                            receipt,
                            previous=None,
                            original_actor="platform:psp",
                            server_tick=event_tick,
                            idempotency_key=(
                                f"partial-settlement-capture:{idempotency_key}"
                            ),
                        )
                        writes.append((
                            "payment_states",
                            payment_state_key(captured_payment),
                            None,
                            captured_payment,
                        ))
                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.settle_order_partial",
                        before=before,
                        after=after,
                    )
                if event_tick is not None:
                    _record_mutation(
                        self._conn,
                        table="logical_time",
                        key="world",
                        action="world.settle_order_partial",
                        before=event_tick - 1,
                        after=event_tick,
                    )
                    self._set_clock(event_tick)
                commit_writes = [
                    TableWrite(
                        table=table,
                        key=key,
                        op="create" if before is None else "update",
                        before=before,
                        after=after,
                    )
                    for table, key, before, after in writes
                ]
                if event_tick is not None:
                    commit_writes.append(
                        TableWrite(
                            table="logical_time",
                            key="world",
                            op="update",
                            before=event_tick - 1,
                            after=event_tick,
                        )
                    )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="partial_settle",
                    authority_action="world.settle_order_partial",
                    actor_id=by_actor,
                    idempotency_key=idempotency_key,
                    subject_id=str(order.order_id),
                    table_writes=tuple(commit_writes),
                    invariants_held=(
                        "atomic",
                        "requested=fulfilled+backordered",
                        "no-oversell",
                        "zero-fill-no-payment",
                        "idempotent",
                        *((
                            "first-class-payment-captured",
                        ) if captured_payment is not None else ()),
                    ),
                )
                self._conn.commit()
                return allocation
            except BaseException:
                self._conn.rollback()
                raise

    def allocate_orders_atomic(
        self,
        *,
        allocation_id: str,
        merchant_id: AgentId,
        sku_id: SkuId,
        priority_order_ids: tuple[OrderId, ...],
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AllocationBatch:
        """SQLite equivalent of ``World.allocate_orders_atomic``.

        Priority and quantities are derived from durable World rows while an
        ``IMMEDIATE`` transaction holds the write lock.  The persisted outcome
        record makes an exact retry stable across process restarts.
        """

        if by_actor != "platform:fulfillment":
            raise LifecycleAuthorizationError(
                f"only platform:fulfillment may allocate orders; got {by_actor!r}"
            )
        if original_actor != str(merchant_id):
            raise LifecycleAuthorizationError(
                "allocation original_actor must equal the owning merchant"
            )
        if not allocation_id.strip() or not idempotency_key.strip():
            raise FulfillmentNotActionable(
                "allocation id and idempotency key must not be blank"
            )
        if not priority_order_ids or len(set(priority_order_ids)) != len(
            priority_order_ids
        ):
            raise FulfillmentNotActionable(
                "priority_order_ids must be non-empty and unique"
            )

        fingerprint = _canonical_json({
            "allocation_id": allocation_id,
            "merchant_id": str(merchant_id),
            "sku_id": str(sku_id),
            "priority_order_ids": [str(value) for value in priority_order_ids],
            "original_actor": original_actor,
        })
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._conn.execute(
                    """
                    SELECT request_fingerprint, outcome_json
                    FROM allocation_batch_records
                    WHERE scope = ? AND idempotency_key = ?
                    """,
                    (by_actor, idempotency_key),
                ).fetchone()
                if replay is not None:
                    if replay["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was reused for another allocation"
                        )
                    outcome = _allocation_batch_from_json(
                        str(replay["outcome_json"])
                    )
                    self._conn.commit()
                    return outcome

                id_conflict = self._conn.execute(
                    """
                    SELECT 1 FROM allocation_batch_records
                    WHERE allocation_id = ?
                    """,
                    (allocation_id,),
                ).fetchone()
                if id_conflict is not None:
                    raise IdempotencyConflict(
                        f"allocation id {allocation_id!r} already exists"
                    )

                eligible = tuple(
                    cast(Order, _row_to_value("orders", row))
                    for row in self._conn.execute(
                        """
                        SELECT * FROM orders
                        WHERE merchant_id = ? AND sku_id = ?
                          AND state IN (?, ?) AND request_order > 0
                        ORDER BY request_order, buyer_id, order_id
                        """,
                        (
                            str(merchant_id),
                            str(sku_id),
                            OrderState.PROPOSED.value,
                            OrderState.ACCEPTED.value,
                        ),
                    ).fetchall()
                )
                authoritative = tuple(row.order_id for row in eligible)
                if priority_order_ids != authoritative:
                    raise FulfillmentNotActionable(
                        "priority_order_ids do not cover authoritative pending orders in priority order"
                    )
                if not eligible:
                    raise FulfillmentNotActionable("allocation has no eligible orders")

                inventory = cast(
                    "InventoryRow | None",
                    self._fetch_value("inventory", str(sku_id)),
                )
                listing = cast(
                    "Listing | None", self._fetch_value("catalog", str(sku_id))
                )
                if inventory is None or listing is None:
                    raise OutOfStock(f"unknown allocation sku {sku_id}")
                if (
                    inventory.merchant_id != merchant_id
                    or listing.merchant_id != merchant_id
                ):
                    raise FulfillmentNotActionable(
                        "allocation catalog/inventory is not owned by the merchant"
                    )
                existing_fulfillment = self._conn.execute(
                    """
                    SELECT order_id FROM fulfillments
                    WHERE order_id IN ({}) LIMIT 1
                    """.format(",".join("?" for _ in priority_order_ids)),
                    tuple(str(value) for value in priority_order_ids),
                ).fetchone()
                if existing_fulfillment is not None:
                    raise FulfillmentNotActionable(
                        "an allocation order already has a fulfillment record"
                    )

                event_tick = self._clock_value() + 1
                remaining = inventory.qty_available - inventory.qty_reserved
                by_id = {order.order_id: order for order in eligible}
                allocations: list[FulfillmentAllocation] = []
                receipts: list[Receipt] = []
                orders_after: list[tuple[Order, Order]] = []
                timelines: list[OrderTimeline] = []
                total_fulfilled = 0
                for order_id in priority_order_ids:
                    order = by_id[order_id]
                    fulfilled = min(order.qty, remaining)
                    backordered = order.qty - fulfilled
                    remaining -= fulfilled
                    total_fulfilled += fulfilled
                    per_order_key = f"{idempotency_key}:{order.order_id}"
                    receipt = None
                    if fulfilled > 0:
                        receipt = Receipt(
                            txn_id=TxnId(
                                f"allocation:{allocation_id}:{order.order_id}"
                            ),
                            ts=f"world-tick:{event_tick}",
                            order_id=order.order_id,
                            buyer_id=order.buyer_id,
                            merchant_id=order.merchant_id,
                            sku_id=order.sku_id,
                            qty=fulfilled,
                            price=order.agreed_price,
                            idempotency_key=per_order_key,
                        )
                        if self._fetch("ledger", str(receipt.txn_id)) is not None:
                            raise WriteNotAuthorized(
                                f"ledger row already exists: {receipt.txn_id}"
                            )
                        receipts.append(receipt)
                    allocation = FulfillmentAllocation(
                        order_id=order.order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        sku_id=order.sku_id,
                        requested_qty=order.qty,
                        fulfilled_qty=fulfilled,
                        backordered_qty=backordered,
                        receipt_txn_id=(
                            None if receipt is None else receipt.txn_id
                        ),
                        created_by=AgentId(original_actor),
                        idempotency_key=per_order_key,
                        allocation_id=allocation_id,
                    )
                    allocations.append(allocation)
                    target = (
                        OrderState.BACKORDERED
                        if fulfilled == 0
                        else OrderState.PARTIALLY_SETTLED
                        if backordered > 0
                        else OrderState.SETTLED
                    )
                    orders_after.append((order, replace(order, state=target)))
                    if fulfilled > 0:
                        if self._fetch(
                            "order_timelines", str(order.order_id)
                        ) is not None:
                            raise FulfillmentNotActionable(
                                f"order {order.order_id} already has timeline evidence"
                            )
                        timelines.append(OrderTimeline(
                            order_id=order.order_id,
                            buyer_id=order.buyer_id,
                            merchant_id=order.merchant_id,
                            settled_at_tick=event_tick,
                            return_window_ticks=_captured_return_window(listing),
                        ))

                inventory_after = replace(
                    inventory,
                    qty_reserved=inventory.qty_reserved + total_fulfilled,
                    version=inventory.version + 1,
                )
                batch = AllocationBatch(
                    allocation_id=allocation_id,
                    merchant_id=merchant_id,
                    sku_id=sku_id,
                    priority_order_ids=priority_order_ids,
                    allocations=tuple(allocations),
                    created_by=AgentId(original_actor),
                    idempotency_key=idempotency_key,
                )
                writes: list[tuple[str, str, Any | None, Any]] = [
                    ("orders", str(before.order_id), before, after)
                    for before, after in orders_after
                ]
                if total_fulfilled:
                    writes.append((
                        "inventory", str(sku_id), inventory, inventory_after
                    ))
                receipt_by_order = {row.order_id: row for row in receipts}
                for allocation in allocations:
                    receipt = receipt_by_order.get(allocation.order_id)
                    if receipt is not None:
                        writes.append((
                            "ledger", str(receipt.txn_id), None, receipt
                        ))
                    writes.append((
                        "fulfillments",
                        str(allocation.order_id),
                        None,
                        allocation,
                    ))
                writes.extend(
                    ("order_timelines", str(row.order_id), None, row)
                    for row in timelines
                )
                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.allocate_orders_atomic",
                        before=before,
                        after=after,
                    )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.allocate_orders_atomic",
                    before=event_tick - 1,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._conn.execute(
                    """
                    INSERT INTO allocation_batch_records
                      (allocation_id, scope, idempotency_key,
                       request_fingerprint, outcome_json, created_at)
                    VALUES (?, ?, ?, ?, ?,
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        allocation_id,
                        by_actor,
                        idempotency_key,
                        fingerprint,
                        _value_to_json(batch),
                    ),
                )
                self._conn.commit()
                return batch
            except BaseException:
                self._conn.rollback()
                raise

    def record_shipment_status(
        self,
        *,
        shipment_id: ShipmentId,
        event_id: str,
        status: ShipmentStatus,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        """Durably append one trusted logistics event."""

        if by_actor != "platform:fulfillment":
            raise LifecycleAuthorizationError(
                f"only platform:fulfillment may record shipment status; got {by_actor!r}"
            )
        if original_actor != "runtime:logistics":
            raise LifecycleAuthorizationError(
                "shipment status original_actor must be runtime:logistics"
            )
        if not str(shipment_id).strip() or not event_id.strip() or not idempotency_key.strip():
            raise ShipmentNotActionable(
                "shipment id, event id, and idempotency key must not be blank"
            )
        fingerprint = _canonical_json({
            "shipment_id": str(shipment_id),
            "event_id": event_id,
            "status": status.value,
            "original_actor": original_actor,
        })
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._conn.execute(
                    """
                    SELECT request_fingerprint, outcome_json
                    FROM shipment_event_records
                    WHERE scope = ? AND idempotency_key = ?
                    """,
                    (by_actor, idempotency_key),
                ).fetchone()
                if replay is not None:
                    if replay["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was reused for another shipment event"
                        )
                    outcome = _shipment_from_json(str(replay["outcome_json"]))
                    self._conn.commit()
                    return outcome

                shipment = cast(
                    "Shipment | None",
                    self._fetch_value("shipments", str(shipment_id)),
                )
                if shipment is None:
                    raise ShipmentNotActionable(
                        f"shipment {shipment_id} does not exist"
                    )
                for event in shipment.status_history:
                    if event.event_id != event_id:
                        continue
                    if event.status != status:
                        raise IdempotencyConflict(
                            f"shipment event id {event_id!r} was reused with another status"
                        )
                    self._conn.execute(
                        """
                        INSERT INTO shipment_event_records
                          (scope, idempotency_key, shipment_id, event_id,
                           request_fingerprint, outcome_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?,
                                strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                        """,
                        (
                            by_actor,
                            idempotency_key,
                            str(shipment_id),
                            event_id,
                            fingerprint,
                            _value_to_json(shipment),
                        ),
                    )
                    self._conn.commit()
                    return shipment
                if shipment.resolution in {
                    ShipmentResolution.REPLACEMENT,
                    ShipmentResolution.REFUND,
                }:
                    raise ShipmentNotActionable(
                        "a replaced or refunded shipment cannot receive more status events"
                    )
                if status not in _SHIPMENT_TRANSITIONS[shipment.status]:
                    raise ShipmentNotActionable(
                        f"shipment cannot transition from {shipment.status.value!r} to {status.value!r}"
                    )
                event_tick = self._clock_value() + 1
                updated = replace(
                    shipment,
                    status=status,
                    status_history=shipment.status_history
                    + (
                        ShipmentStatusEvent(
                            event_id,
                            status,
                            event_tick,
                            idempotency_key=idempotency_key,
                            shipment_version=shipment.version + 1,
                        ),
                    ),
                    version=shipment.version + 1,
                )
                _upsert(self._conn, "shipments", updated)
                _record_mutation(
                    self._conn,
                    table="shipments",
                    key=str(shipment_id),
                    action="world.record_shipment_status",
                    before=shipment,
                    after=updated,
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.record_shipment_status",
                    before=event_tick - 1,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._conn.execute(
                    """
                    INSERT INTO shipment_event_records
                      (scope, idempotency_key, shipment_id, event_id,
                       request_fingerprint, outcome_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?,
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        by_actor,
                        idempotency_key,
                        str(shipment_id),
                        event_id,
                        fingerprint,
                        _value_to_json(updated),
                    ),
                )
                self._conn.commit()
                return updated
            except BaseException:
                self._conn.rollback()
                raise

    def resolve_shipment(
        self,
        *,
        shipment_id: ShipmentId,
        resolution: ShipmentResolution,
        replacement_sku_id: SkuId | None,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> Shipment:
        """Durably commit an atomic shipment exception resolution."""

        if by_actor != "platform:fulfillment":
            raise LifecycleAuthorizationError(
                f"only platform:fulfillment may resolve shipments; got {by_actor!r}"
            )
        if not str(shipment_id).strip() or not idempotency_key.strip():
            raise ShipmentNotActionable(
                "shipment id and idempotency key must not be blank"
            )
        fingerprint = _canonical_json({
            "shipment_id": str(shipment_id),
            "resolution": resolution.value,
            "replacement_sku_id": (
                None if replacement_sku_id is None else str(replacement_sku_id)
            ),
            "original_actor": original_actor,
        })
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._conn.execute(
                    """
                    SELECT request_fingerprint, outcome_json
                    FROM shipment_resolution_records
                    WHERE scope = ? AND idempotency_key = ?
                    """,
                    (by_actor, idempotency_key),
                ).fetchone()
                if replay is not None:
                    if replay["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was reused for another shipment resolution"
                        )
                    outcome = _shipment_from_json(str(replay["outcome_json"]))
                    self._conn.commit()
                    return outcome

                shipment = cast(
                    "Shipment | None",
                    self._fetch_value("shipments", str(shipment_id)),
                )
                if shipment is None:
                    raise ShipmentNotActionable(
                        f"shipment {shipment_id} does not exist"
                    )
                if original_actor not in {
                    str(shipment.buyer_id),
                    str(shipment.merchant_id),
                }:
                    raise LifecycleAuthorizationError(
                        f"actor {original_actor!r} is not a party to shipment {shipment_id}"
                    )
                if shipment.resolution is not None:
                    if (
                        shipment.resolution == resolution
                        and shipment.replacement_sku_id == replacement_sku_id
                    ):
                        self._conn.execute(
                            """
                            INSERT INTO shipment_resolution_records
                              (scope, idempotency_key, shipment_id,
                               request_fingerprint, outcome_json, created_at)
                            VALUES (?, ?, ?, ?, ?,
                                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                            """,
                            (
                                by_actor,
                                idempotency_key,
                                str(shipment_id),
                                fingerprint,
                                _value_to_json(shipment),
                            ),
                        )
                        self._conn.commit()
                        return shipment
                    raise ShipmentNotActionable(
                        f"shipment {shipment_id} already has a different resolution"
                    )
                if resolution == ShipmentResolution.WAIT:
                    if shipment.status not in {
                        ShipmentStatus.DELAYED,
                        ShipmentStatus.MISSING_SCAN,
                        ShipmentStatus.LOST,
                    }:
                        raise ShipmentNotActionable(
                            "wait is valid only for an exceptional shipment"
                        )
                    if replacement_sku_id is not None:
                        raise ShipmentNotActionable(
                            "wait resolution cannot name a replacement sku"
                        )
                else:
                    if shipment.status != ShipmentStatus.LOST:
                        raise ShipmentNotActionable(
                            "replacement/refund requires authoritative lost status"
                        )
                    if resolution == ShipmentResolution.REPLACEMENT:
                        if replacement_sku_id is None:
                            raise ShipmentNotActionable(
                                "replacement resolution requires replacement_sku_id"
                            )
                    elif replacement_sku_id is not None:
                        raise ShipmentNotActionable(
                            "refund resolution cannot name a replacement sku"
                        )

                order = cast(
                    "Order | None",
                    self._fetch_value("orders", str(shipment.order_id)),
                )
                if order is None or (
                    order.buyer_id != shipment.buyer_id
                    or order.merchant_id != shipment.merchant_id
                    or order.sku_id != shipment.original_sku_id
                ):
                    raise ShipmentNotActionable(
                        "shipment does not match its authoritative order"
                    )
                if order.state != OrderState.DISPATCHED:
                    raise ShipmentNotActionable(
                        f"shipment order must be dispatched; found {order.state.value!r}"
                    )
                paid_qty = _paid_quantity_from_store(self, order)
                event_tick = self._clock_value() + 1
                updated = replace(
                    shipment,
                    resolution=resolution,
                    replacement_sku_id=replacement_sku_id,
                    version=shipment.version + 1,
                    resolution_idempotency_key=idempotency_key,
                    resolved_by=AgentId(original_actor),
                    resolution_version=shipment.version + 1,
                    resolution_history_length=len(shipment.status_history),
                )
                writes: list[tuple[str, str, Any | None, Any]] = [(
                    "shipments", str(shipment_id), shipment, updated
                )]
                original_inventory: InventoryRow | None = None
                original_after: InventoryRow | None = None
                replacement_inventory: InventoryRow | None = None
                replacement_after: InventoryRow | None = None

                if resolution in {
                    ShipmentResolution.REPLACEMENT,
                    ShipmentResolution.REFUND,
                }:
                    original_inventory = cast(
                        "InventoryRow | None",
                        self._fetch_value(
                            "inventory", str(shipment.original_sku_id)
                        ),
                    )
                    if (
                        original_inventory is None
                        or original_inventory.merchant_id != shipment.merchant_id
                        or original_inventory.qty_reserved < paid_qty
                        or original_inventory.qty_available < paid_qty
                    ):
                        raise ShipmentNotActionable(
                            "lost shipment lacks its authoritative inventory reservation"
                        )
                    original_after = replace(
                        original_inventory,
                        qty_available=original_inventory.qty_available - paid_qty,
                        qty_reserved=original_inventory.qty_reserved - paid_qty,
                        version=original_inventory.version + 1,
                    )

                if resolution == ShipmentResolution.REPLACEMENT:
                    assert replacement_sku_id is not None
                    replacement_inventory = cast(
                        "InventoryRow | None",
                        self._fetch_value("inventory", str(replacement_sku_id)),
                    )
                    replacement_listing = cast(
                        "Listing | None",
                        self._fetch_value("catalog", str(replacement_sku_id)),
                    )
                    if (
                        replacement_inventory is None
                        or replacement_listing is None
                        or replacement_inventory.merchant_id != shipment.merchant_id
                        or replacement_listing.merchant_id != shipment.merchant_id
                    ):
                        raise ShipmentNotActionable(
                            "replacement sku is not owned by the shipment merchant"
                        )
                    if replacement_sku_id == shipment.original_sku_id:
                        assert original_after is not None
                        if _available_qty(original_after) < paid_qty:
                            raise OutOfStock(
                                f"insufficient replacement inventory for {replacement_sku_id}"
                            )
                        replacement_after = replace(
                            original_after,
                            qty_reserved=original_after.qty_reserved + paid_qty,
                            version=original_inventory.version + 1,
                        )
                        original_after = replacement_after
                        replacement_inventory = original_inventory
                    else:
                        if _available_qty(replacement_inventory) < paid_qty:
                            raise OutOfStock(
                                f"insufficient replacement inventory for {replacement_sku_id}"
                            )
                        replacement_after = replace(
                            replacement_inventory,
                            qty_reserved=replacement_inventory.qty_reserved + paid_qty,
                            version=replacement_inventory.version + 1,
                        )
                    assert original_inventory is not None and original_after is not None
                    writes.append((
                        "inventory",
                        str(shipment.original_sku_id),
                        original_inventory,
                        original_after,
                    ))
                    if replacement_sku_id != shipment.original_sku_id:
                        assert replacement_inventory is not None and replacement_after is not None
                        writes.append((
                            "inventory",
                            str(replacement_sku_id),
                            replacement_inventory,
                            replacement_after,
                        ))

                if resolution == ShipmentResolution.REFUND:
                    assert original_inventory is not None and original_after is not None
                    refunded_order = replace(order, state=OrderState.REFUNDED)
                    refund_receipt = Receipt(
                        txn_id=TxnId(f"refund:shipment:{shipment_id}"),
                        ts=f"world-tick:{event_tick}",
                        order_id=order.order_id,
                        buyer_id=order.buyer_id,
                        merchant_id=order.merchant_id,
                        sku_id=order.sku_id,
                        qty=paid_qty,
                        price=order.agreed_price,
                        idempotency_key=idempotency_key,
                        effect="refund",
                    )
                    if self._fetch("ledger", str(refund_receipt.txn_id)) is not None:
                        raise WriteNotAuthorized(
                            f"ledger row already exists: {refund_receipt.txn_id}"
                        )
                    timeline_before = cast(
                        "OrderTimeline | None",
                        self._fetch_value("order_timelines", str(order.order_id)),
                    )
                    timeline_after = (
                        OrderTimeline(
                            order_id=order.order_id,
                            buyer_id=order.buyer_id,
                            merchant_id=order.merchant_id,
                            refunded_at_tick=event_tick,
                        )
                        if timeline_before is None
                        else replace(timeline_before, refunded_at_tick=event_tick)
                    )
                    writes.extend([
                        (
                            "inventory",
                            str(shipment.original_sku_id),
                            original_inventory,
                            original_after,
                        ),
                        ("orders", str(order.order_id), order, refunded_order),
                        ("ledger", str(refund_receipt.txn_id), None, refund_receipt),
                        (
                            "order_timelines",
                            str(order.order_id),
                            timeline_before,
                            timeline_after,
                        ),
                    ])

                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.resolve_shipment",
                        before=before,
                        after=after,
                    )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.resolve_shipment",
                    before=event_tick - 1,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                self._conn.execute(
                    """
                    INSERT INTO shipment_resolution_records
                      (scope, idempotency_key, shipment_id,
                       request_fingerprint, outcome_json, created_at)
                    VALUES (?, ?, ?, ?, ?,
                            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                    """,
                    (
                        by_actor,
                        idempotency_key,
                        str(shipment_id),
                        fingerprint,
                        _value_to_json(updated),
                    ),
                )
                self._conn.commit()
                return updated
            except BaseException:
                self._conn.rollback()
                raise

    def dispatch_order(
        self,
        *,
        order_id: OrderId,
        by_actor: str,
        _manage_transaction: bool = True,
    ) -> Order:
        """Move ``SETTLED`` to ``DISPATCHED`` in one SQLite transaction."""
        return self._transition_order(
            order_id=order_id,
            by_actor=by_actor,
            allowed_actor_sides=frozenset({"merchant"}),
            allowed_from=frozenset({
                OrderState.SETTLED,
                OrderState.PARTIALLY_SETTLED,
            }),
            replay_states=frozenset({
                OrderState.DISPATCHED,
                OrderState.RETURNED,
                OrderState.REFUNDED,
            }),
            target=OrderState.DISPATCHED,
            action="dispatch",
            timeline_field="dispatched_at_tick",
            create_shipment=True,
            create_packing=True,
            _manage_transaction=_manage_transaction,
        )

    def cancel_order(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Move ``PROPOSED`` or ``ACCEPTED`` to ``CANCELLED`` atomically."""
        return self._transition_order(
            order_id=order_id,
            by_actor=by_actor,
            allowed_actor_sides=frozenset({"buyer", "merchant"}),
            allowed_from=frozenset({
                OrderState.PROPOSED,
                OrderState.ACCEPTED,
                OrderState.BACKORDERED,
            }),
            replay_states=frozenset({OrderState.CANCELLED}),
            target=OrderState.CANCELLED,
            action="cancel",
        )

    def mark_order_returned(self, *, order_id: OrderId, by_actor: str) -> Order:
        """Move ``DISPATCHED`` to ``RETURNED`` atomically."""
        return self._transition_order(
            order_id=order_id,
            by_actor=by_actor,
            allowed_actor_sides=frozenset({"merchant"}),
            allowed_from=frozenset({OrderState.DISPATCHED}),
            replay_states=frozenset({OrderState.RETURNED, OrderState.REFUNDED}),
            target=OrderState.RETURNED,
            action="return",
            timeline_field="returned_at_tick",
        )

    def exchange_order(
        self,
        *,
        exchange_id: ExchangeId,
        original_order_id: OrderId,
        replacement_order: Order,
        by_actor: str,
        idempotency_key: str,
    ) -> Exchange:
        """Durably replace a returned order without creating ledger rows."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                original_value = self._fetch_value(
                    "orders",
                    str(original_order_id),
                )
                if original_value is None:
                    raise ExchangeNotActionable(
                        f"original order {original_order_id} does not exist"
                    )
                original = cast(Order, original_value)
                _authorize_exchange_actor(original, by_actor)
                if replacement_order.state not in _SETTLEABLE:
                    raise ExchangeNotActionable(
                        "replacement order must begin PROPOSED or ACCEPTED"
                    )
                if not str(exchange_id).strip() or not idempotency_key.strip():
                    raise ExchangeNotActionable(
                        "exchange id and idempotency key must not be blank"
                    )
                allocation_value = self._fetch_value(
                    "fulfillments",
                    str(original.order_id),
                )
                allocation = cast(
                    "FulfillmentAllocation | None",
                    allocation_value,
                )
                qty = (
                    original.qty
                    if allocation is None
                    else allocation.fulfilled_qty
                )
                if qty <= 0:
                    raise ExchangeNotActionable(
                        f"order {original.order_id} has no fulfilled units"
                    )
                exchange = Exchange(
                    exchange_id=exchange_id,
                    original_order_id=original.order_id,
                    replacement_order_id=replacement_order.order_id,
                    buyer_id=original.buyer_id,
                    merchant_id=original.merchant_id,
                    original_sku_id=original.sku_id,
                    replacement_sku_id=replacement_order.sku_id,
                    qty=qty,
                    created_by=AgentId(by_actor),
                    idempotency_key=idempotency_key,
                )
                prior_key = self._conn.execute(
                    """
                    SELECT * FROM exchanges
                    WHERE created_by = ? AND idempotency_key = ?
                    """,
                    (by_actor, idempotency_key),
                ).fetchone()
                if prior_key is not None:
                    prior = cast(Exchange, _row_to_value("exchanges", prior_key))
                    persisted_replacement = self._fetch_value(
                        "orders",
                        str(prior.replacement_order_id),
                    )
                    if (
                        prior == exchange
                        and isinstance(persisted_replacement, Order)
                        and _order_identity_matches(
                            persisted_replacement,
                            replacement_order,
                        )
                    ):
                        self._conn.commit()
                        return prior
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used "
                        "for a different exchange request"
                    )
                conflict = self._conn.execute(
                    """
                    SELECT * FROM exchanges
                    WHERE exchange_id = ? OR original_order_id = ?
                       OR replacement_order_id = ?
                    LIMIT 1
                    """,
                    (
                        str(exchange_id),
                        str(original.order_id),
                        str(replacement_order.order_id),
                    ),
                ).fetchone()
                if conflict is not None:
                    prior = cast(Exchange, _row_to_value("exchanges", conflict))
                    if prior.exchange_id == exchange_id:
                        detail = f"exchange id {exchange_id} already exists"
                    elif prior.original_order_id == original.order_id:
                        detail = f"order {original.order_id} was already exchanged"
                    else:
                        detail = (
                            f"order {replacement_order.order_id} is already a replacement"
                        )
                    raise ExchangeNotActionable(detail)
                if original.state != OrderState.RETURNED:
                    raise ExchangeNotActionable(
                        f"order {original.order_id} must be RETURNED before exchange; "
                        f"found {original.state.value!r}"
                    )
                if allocation is not None and allocation.backordered_qty != 0:
                    raise ExchangeNotActionable(
                        "an order with outstanding backordered units cannot be exchanged"
                    )
                _validate_replacement_identity(original, replacement_order, qty=qty)
                if self._fetch(
                    "orders",
                    str(replacement_order.order_id),
                ) is not None:
                    raise ExchangeNotActionable(
                        f"replacement order id {replacement_order.order_id} already exists"
                    )

                original_listing = self._fetch_value(
                    "catalog",
                    str(original.sku_id),
                )
                replacement_listing = self._fetch_value(
                    "catalog",
                    str(replacement_order.sku_id),
                )
                _require_listing_owner(original, original_listing)
                _require_listing_owner(replacement_order, replacement_listing)
                original_inventory_value = self._fetch_value(
                    "inventory",
                    str(original.sku_id),
                )
                replacement_inventory_value = self._fetch_value(
                    "inventory",
                    str(replacement_order.sku_id),
                )
                if (
                    original_inventory_value is None
                    or replacement_inventory_value is None
                ):
                    raise ExchangeNotActionable(
                        "both original and replacement SKUs require inventory rows"
                    )
                original_inventory = cast(
                    InventoryRow,
                    original_inventory_value,
                )
                replacement_inventory = cast(
                    InventoryRow,
                    replacement_inventory_value,
                )
                _require_inventory_owner(original, original_inventory)
                _require_inventory_owner(replacement_order, replacement_inventory)
                if original_inventory.qty_reserved < qty:
                    raise ExchangeNotActionable(
                        "original inventory no longer holds the returned reservation"
                    )

                inventory_writes: list[tuple[str, str, Any, Any]] = []
                if original.sku_id == replacement_order.sku_id:
                    released = replace(
                        original_inventory,
                        qty_reserved=original_inventory.qty_reserved - qty,
                        version=original_inventory.version + 1,
                    )
                    if released.qty_available - released.qty_reserved < qty:
                        raise OutOfStock(
                            "insufficient inventory for replacement sku "
                            f"{original.sku_id}"
                        )
                    final_inventory = replace(
                        released,
                        qty_reserved=released.qty_reserved + qty,
                        version=original_inventory.version + 1,
                    )
                    inventory_writes.append((
                        "inventory",
                        str(original.sku_id),
                        original_inventory,
                        final_inventory,
                    ))
                else:
                    if (
                        replacement_inventory.qty_available
                        - replacement_inventory.qty_reserved
                        < qty
                    ):
                        raise OutOfStock(
                            "insufficient inventory for replacement sku "
                            f"{replacement_order.sku_id}"
                        )
                    released = replace(
                        original_inventory,
                        qty_reserved=original_inventory.qty_reserved - qty,
                        version=original_inventory.version + 1,
                    )
                    replacement_reserved = replace(
                        replacement_inventory,
                        qty_reserved=replacement_inventory.qty_reserved + qty,
                        version=replacement_inventory.version + 1,
                    )
                    inventory_writes.extend((
                        (
                            "inventory",
                            str(original.sku_id),
                            original_inventory,
                            released,
                        ),
                        (
                            "inventory",
                            str(replacement_order.sku_id),
                            replacement_inventory,
                            replacement_reserved,
                        ),
                    ))

                exchanged_original = replace(
                    original,
                    state=OrderState.EXCHANGED,
                )
                settled_replacement = replace(
                    replacement_order,
                    state=OrderState.SETTLED,
                )
                writes: list[tuple[str, str, Any, Any]] = [
                    (
                        "orders",
                        str(original.order_id),
                        original,
                        exchanged_original,
                    ),
                    (
                        "orders",
                        str(replacement_order.order_id),
                        None,
                        settled_replacement,
                    ),
                    *inventory_writes,
                    (
                        "exchanges",
                        str(exchange.exchange_id),
                        None,
                        exchange,
                    ),
                ]
                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.exchange_order",
                        before=before,
                        after=after,
                    )
                self._conn.commit()
                return exchange
            except BaseException:
                self._conn.rollback()
                raise

    def _transition_order(
        self,
        *,
        order_id: OrderId,
        by_actor: str,
        allowed_actor_sides: frozenset[str],
        allowed_from: frozenset[OrderState],
        replay_states: frozenset[OrderState],
        target: OrderState,
        action: str,
        timeline_field: str | None = None,
        create_shipment: bool = False,
        create_packing: bool = False,
        _manage_transaction: bool = True,
    ) -> Order:
        """Apply one allow-listed, non-regressing order transition."""
        with self._lock:
            if _manage_transaction:
                self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing_value = self._fetch_value("orders", str(order_id))
                if existing_value is None:
                    raise InvalidOrderTransition(f"order {order_id} does not exist")
                existing = cast(Order, existing_value)
                _authorize_lifecycle_actor(
                    existing,
                    by_actor,
                    allowed_actor_sides,
                )
                if existing.state in replay_states:
                    if _manage_transaction:
                        self._conn.commit()
                    return existing
                if existing.state not in allowed_from:
                    allowed = ", ".join(
                        sorted(state.value for state in allowed_from)
                    )
                    raise InvalidOrderTransition(
                        f"cannot {action} order {order_id} from "
                        f"{existing.state.value!r}; allowed states: {allowed}"
                    )
                updated = replace(existing, state=target)
                timing_value = self._fetch_value("order_timelines", str(order_id))
                timing_before = cast("OrderTimeline | None", timing_value)
                event_tick = self._clock_value() + 1 if timeline_field is not None else None
                timing_after = timing_before
                if timeline_field is not None:
                    if timing_before is None:
                        listing = self._fetch_value("catalog", str(existing.sku_id))
                        window = (
                            _captured_return_window(listing)
                            if timeline_field == "dispatched_at_tick"
                            else None
                        )
                        timing_after = OrderTimeline(
                            order_id=existing.order_id,
                            buyer_id=existing.buyer_id,
                            merchant_id=existing.merchant_id,
                            return_window_ticks=window,
                            **{timeline_field: event_tick},
                        )
                    else:
                        timing_after = replace(
                            timing_before,
                            **{timeline_field: event_tick},
                        )
                shipment_after: Shipment | None = None
                if create_shipment:
                    assert event_tick is not None
                    prior_shipment = self._conn.execute(
                        "SELECT shipment_id FROM shipments WHERE order_id = ? LIMIT 1",
                        (str(existing.order_id),),
                    ).fetchone()
                    if prior_shipment is not None:
                        raise ShipmentNotActionable(
                            f"order {existing.order_id} already has shipment "
                            f"{prior_shipment['shipment_id']} before dispatch"
                        )
                    shipment_id = ShipmentId(f"shipment:{existing.order_id}")
                    if self._fetch("shipments", str(shipment_id)) is not None:
                        raise ShipmentNotActionable(
                            f"shipment id {shipment_id} already exists"
                        )
                    shipment_after = Shipment(
                        shipment_id=shipment_id,
                        order_id=existing.order_id,
                        buyer_id=existing.buyer_id,
                        merchant_id=existing.merchant_id,
                        original_sku_id=existing.sku_id,
                        status=ShipmentStatus.IN_TRANSIT,
                        status_history=(
                            ShipmentStatusEvent(
                                event_id=f"shipment:{existing.order_id}:created",
                                status=ShipmentStatus.IN_TRANSIT,
                                logical_time=event_tick,
                                shipment_version=1,
                            ),
                        ),
                        version=1,
                    )
                packing_rows: tuple[PackingRecord, ...] = ()
                if create_packing:
                    if event_tick is None:
                        raise WorldError(
                            "packing dispatch requires a World event tick"
                        )
                    payment_history = self._payment_history_unchecked(
                        str(existing.order_id)
                    )
                    if not payment_history:
                        if existing.state != OrderState.PARTIALLY_SETTLED:
                            raise WorldError(
                                "dispatch requires first-class captured payment state"
                            )
                    else:
                        prior_packing = self._packing_history_unchecked(
                            str(existing.order_id)
                        )
                        packing_actor = (
                            by_actor
                            if by_actor == str(existing.merchant_id)
                            else "platform:fulfillment"
                        )
                        packing_rows = derive_dispatch_packing_sequence(
                            updated,
                            payment_history[-1],
                            previous=(
                                prior_packing[-1] if prior_packing else None
                            ),
                            original_actor=packing_actor,
                            server_tick=event_tick,
                            idempotency_key=f"dispatch:{existing.order_id}",
                        )
                _upsert(self._conn, "orders", updated)
                _record_mutation(
                    self._conn,
                    table="orders",
                    key=str(order_id),
                    action=f"world.{action}_order",
                    before=existing,
                    after=updated,
                )
                if timing_after is not None and timing_after != timing_before:
                    _upsert(self._conn, "order_timelines", timing_after)
                    _record_mutation(
                        self._conn,
                        table="order_timelines",
                        key=str(order_id),
                        action=f"world.{action}_order",
                        before=timing_before,
                        after=timing_after,
                    )
                if shipment_after is not None:
                    _upsert(self._conn, "shipments", shipment_after)
                    _record_mutation(
                        self._conn,
                        table="shipments",
                        key=str(shipment_after.shipment_id),
                        action=f"world.{action}_order",
                        before=None,
                        after=shipment_after,
                    )
                for packing in packing_rows:
                    _upsert(self._conn, "packing_records", packing)
                    _record_mutation(
                        self._conn,
                        table="packing_records",
                        key=packing_record_key(packing),
                        action=f"world.{action}_order",
                        before=None,
                        after=packing,
                    )
                if event_tick is not None:
                    _record_mutation(
                        self._conn,
                        table="logical_time",
                        key="world",
                        action=f"world.{action}_order",
                        before=event_tick - 1,
                        after=event_tick,
                    )
                    self._set_clock(event_tick)
                commit_writes: list[TableWrite] = [
                    TableWrite(
                        table="orders",
                        key=str(order_id),
                        op="update",
                        before=existing,
                        after=updated,
                    )
                ]
                if timing_after is not None and timing_after != timing_before:
                    commit_writes.append(
                        TableWrite(
                            table="order_timelines",
                            key=str(order_id),
                            op="create" if timing_before is None else "update",
                            before=timing_before,
                            after=timing_after,
                        )
                    )
                if shipment_after is not None:
                    commit_writes.append(
                        TableWrite(
                            table="shipments",
                            key=str(shipment_after.shipment_id),
                            op="create",
                            before=None,
                            after=shipment_after,
                        )
                    )
                commit_writes.extend(
                    TableWrite(
                        table="packing_records",
                        key=packing_record_key(packing),
                        op="create",
                        before=None,
                        after=packing,
                    )
                    for packing in packing_rows
                )
                if event_tick is not None:
                    commit_writes.append(
                        TableWrite(
                            table="logical_time",
                            key="world",
                            op="update",
                            before=event_tick - 1,
                            after=event_tick,
                        )
                    )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation=action,
                    authority_action=f"world.{action}",
                    actor_id=by_actor,
                    idempotency_key=None,
                    subject_id=str(order_id),
                    table_writes=tuple(commit_writes),
                    invariants_held=(
                        "atomic",
                        f"allowlist:{target.value.upper()}",
                        "non-regressing",
                        *(("unique-shipment-per-order",) if create_shipment else ()),
                        *(("complete-packing-history",) if packing_rows else ()),
                    ),
                )
                if _manage_transaction:
                    self._conn.commit()
                return updated
            except BaseException:
                if _manage_transaction:
                    self._conn.rollback()
                raise

    def open_dispute(self, *, dispute: Dispute, by_actor: str) -> Dispute:
        """Open one party-bound dispute as one durable transaction."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                _validate_open_dispute_shape(dispute, by_actor)
                existing_value = self._fetch_value(
                    "disputes",
                    str(dispute.dispute_id),
                )
                if existing_value is not None:
                    existing = cast(Dispute, existing_value)
                    if _same_dispute_claim(existing, dispute):
                        self._conn.commit()
                        return existing
                    raise DisputeNotActionable(
                        f"dispute id {dispute.dispute_id} already names a "
                        "different claim"
                    )
                order_value = self._fetch_value("orders", str(dispute.order_id))
                order = cast("Order | None", order_value)
                _validate_dispute_order(dispute, order)
                prior = self._conn.execute(
                    "SELECT dispute_id FROM disputes WHERE order_id = ? LIMIT 1",
                    (str(dispute.order_id),),
                ).fetchone()
                if prior is not None:
                    raise DisputeNotActionable(
                        f"order {dispute.order_id} already has dispute "
                        f"{prior['dispute_id']}"
                    )
                _upsert(self._conn, "disputes", dispute)
                _record_mutation(
                    self._conn,
                    table="disputes",
                    key=str(dispute.dispute_id),
                    action="world.open_dispute",
                    before=None,
                    after=dispute,
                )
                self._conn.commit()
                return dispute
            except BaseException:
                self._conn.rollback()
                raise

    def rule_dispute(self, *, ruling: Ruling, by_actor: str) -> Ruling:
        """Atomically mark a dispute RULED and append its single ruling."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if by_actor != "platform:adjudicator":
                    raise LifecycleAuthorizationError(
                        "only platform:adjudicator may rule a dispute"
                    )
                dispute_value = self._fetch_value(
                    "disputes",
                    str(ruling.dispute_id),
                )
                if dispute_value is None:
                    raise DisputeNotActionable(
                        f"dispute {ruling.dispute_id} does not exist"
                    )
                dispute = cast(Dispute, dispute_value)
                existing_value = self._fetch_value(
                    "rulings",
                    str(ruling.ruling_id),
                )
                if existing_value is not None:
                    existing = cast(Ruling, existing_value)
                    if existing == ruling and dispute.state == DisputeState.RULED:
                        self._conn.commit()
                        return existing
                    raise DisputeNotActionable(
                        f"ruling id {ruling.ruling_id} already names a "
                        "different outcome"
                    )
                prior = self._conn.execute(
                    "SELECT ruling_id FROM rulings WHERE dispute_id = ? LIMIT 1",
                    (str(ruling.dispute_id),),
                ).fetchone()
                if prior is not None:
                    raise DisputeNotActionable(
                        f"dispute {ruling.dispute_id} already has ruling "
                        f"{prior['ruling_id']}"
                    )
                if dispute.state not in {
                    DisputeState.OPEN,
                    DisputeState.UNDER_REVIEW,
                }:
                    raise DisputeNotActionable(
                        f"dispute {ruling.dispute_id} in state "
                        f"{dispute.state.value!r} cannot be ruled"
                    )
                order_value = self._fetch_value("orders", str(dispute.order_id))
                order = cast("Order | None", order_value)
                _validate_ruling(ruling, dispute, order)
                ruled_dispute = replace(dispute, state=DisputeState.RULED)
                _upsert(self._conn, "disputes", ruled_dispute)
                _record_mutation(
                    self._conn,
                    table="disputes",
                    key=str(dispute.dispute_id),
                    action="world.rule_dispute",
                    before=dispute,
                    after=ruled_dispute,
                )
                _upsert(self._conn, "rulings", ruling)
                _record_mutation(
                    self._conn,
                    table="rulings",
                    key=str(ruling.ruling_id),
                    action="world.rule_dispute",
                    before=None,
                    after=ruling,
                )
                self._conn.commit()
                return ruling
            except BaseException:
                self._conn.rollback()
                raise

    def refund_order(
        self,
        *,
        order: Order,
        refund_receipt: Receipt,
        by_role: str,
        idempotency_key: str,
        _manage_transaction: bool = True,
    ) -> Receipt:
        """Refund ``order`` and restock its reservation in one transaction."""
        refund_receipt = replace(refund_receipt, effect="refund")
        fingerprint = _idempotency_fingerprint("refund", order, refund_receipt)
        with self._lock:
            if _manage_transaction:
                self._conn.execute("BEGIN IMMEDIATE")
            try:
                replay = self._idempotency_replay(
                    scope=by_role,
                    key=idempotency_key,
                    operation="refund",
                    fingerprint=fingerprint,
                )
                if replay is not None:
                    if _manage_transaction:
                        self._conn.commit()
                    return replay
                existing_value = self._fetch_value("orders", str(order.order_id))
                if existing_value is None:
                    raise OrderNotRefundable(
                        f"order {order.order_id} does not exist — nothing to refund"
                    )
                existing = cast(Order, existing_value)
                validate_persisted_order_identity(existing, order)
                allocation_value = self._fetch_value(
                    "fulfillments",
                    str(existing.order_id),
                )
                allocation = cast(
                    "FulfillmentAllocation | None",
                    allocation_value,
                )
                paid_qty = (
                    existing.qty
                    if allocation is None
                    else allocation.fulfilled_qty
                )
                validate_transaction_identity(
                    existing,
                    refund_receipt,
                    None,
                    expected_qty=paid_qty,
                    expected_effect="refund",
                )
                if existing.state == OrderState.REFUNDED:
                    prior = self._receipt_for_order(order.order_id, refund=True)
                    if prior is None:
                        raise WorldError(
                            f"order {order.order_id} is REFUNDED but has no refund "
                            "receipt — refusing to re-refund"
                        )
                    self._record_idempotency(
                        scope=by_role,
                        key=idempotency_key,
                        operation="refund",
                        fingerprint=fingerprint,
                        outcome=prior,
                    )
                    if _manage_transaction:
                        self._conn.commit()
                    return prior
                if existing.state not in _REFUNDABLE:
                    raise OrderNotRefundable(
                        f"order {order.order_id} in state {existing.state.value!r} "
                        "is not refundable (only a settled/dispatched/returned "
                        "order can be refunded)"
                    )
                replacement = self._conn.execute(
                    """
                    SELECT exchange_id FROM exchanges
                    WHERE replacement_order_id = ? LIMIT 1
                    """,
                    (str(existing.order_id),),
                ).fetchone()
                if replacement is not None:
                    raise OrderNotRefundable(
                        f"replacement order {existing.order_id} has no independent "
                        "payment to refund"
                    )
                if paid_qty <= 0:
                    raise OrderNotRefundable(
                        f"order {existing.order_id} has no fulfilled units"
                    )
                if self._fetch("ledger", str(refund_receipt.txn_id)) is not None:
                    raise WriteNotAuthorized(
                        f"ledger row already exists: {refund_receipt.txn_id}"
                    )

                timeline_value = self._fetch_value(
                    "order_timelines", str(existing.order_id)
                )
                timeline_before = cast("OrderTimeline | None", timeline_value)
                event_tick = self._clock_value() + 1
                _enforce_return_window(timeline_before, event_tick=event_tick)
                payment_history = self._payment_history_unchecked(
                    str(existing.order_id)
                )
                previous_payment = payment_history[-1] if payment_history else None
                refunded_payment: PaymentStateRecord | None = None
                payment_key: str | None = None
                if previous_payment is not None:
                    if previous_payment.state not in {
                        "captured",
                        "partially_refunded",
                    }:
                        raise WorldError(
                            f"order {existing.order_id} has payment state "
                            f"{previous_payment.state!r} before refund"
                        )
                    refunded_payment = derive_payment_resolution(
                        existing,
                        previous_payment,
                        outcome="refunded",
                        refund_receipt=refund_receipt,
                        original_actor="platform:psp",
                        server_tick=event_tick,
                        idempotency_key=f"refund-resolution:{idempotency_key}",
                    )
                    payment_key = payment_state_key(refunded_payment)
                    fully_refunded = refunded_payment.state == "refunded"
                else:
                    if refund_receipt.price != existing.agreed_price:
                        raise OrderNotRefundable(
                            "partial refund requires first-class captured payment state"
                        )
                    fully_refunded = True
                refunded_order = (
                    replace(existing, state=OrderState.REFUNDED)
                    if fully_refunded
                    else existing
                )
                timeline = (
                    None
                    if not fully_refunded
                    else (
                        OrderTimeline(
                            order_id=existing.order_id,
                            buyer_id=existing.buyer_id,
                            merchant_id=existing.merchant_id,
                            return_authorized_at_tick=event_tick,
                            refunded_at_tick=event_tick,
                        )
                        if timeline_before is None
                        else replace(
                            timeline_before,
                            return_authorized_at_tick=event_tick,
                            refunded_at_tick=event_tick,
                        )
                    )
                )
                writes: list[tuple[str, str, Any, Any]] = []
                if fully_refunded:
                    writes.append(
                        ("orders", str(existing.order_id), existing, refunded_order)
                    )
                inventory_value = self._fetch_value("inventory", str(existing.sku_id))
                validate_transaction_identity(
                    existing,
                    refund_receipt,
                    inventory_value,
                    expected_qty=paid_qty,
                    expected_effect="refund",
                )
                validate_listing_owner(
                    existing, self._fetch_value("catalog", str(existing.sku_id))
                )
                if fully_refunded and inventory_value is not None:
                    inventory = cast(InventoryRow, inventory_value)
                    if inventory.qty_reserved < paid_qty:
                        raise OrderNotRefundable(
                            "inventory reservation is below the fulfilled quantity"
                        )
                    restocked = replace(
                        inventory,
                        qty_reserved=inventory.qty_reserved - paid_qty,
                        version=inventory.version + 1,
                    )
                    writes.append(
                        ("inventory", str(existing.sku_id), inventory, restocked)
                    )
                writes.append(("ledger", str(refund_receipt.txn_id), None, refund_receipt))
                if refunded_payment is not None and payment_key is not None:
                    writes.append(
                        (
                            "payment_states",
                            payment_key,
                            None,
                            refunded_payment,
                        )
                    )
                if timeline is not None:
                    writes.append((
                        "order_timelines",
                        str(existing.order_id),
                        timeline_before,
                        timeline,
                    ))

                for table, key, before, after in writes:
                    _upsert(self._conn, table, after)
                    _record_mutation(
                        self._conn,
                        table=table,
                        key=key,
                        action="world.refund_order",
                        before=before,
                        after=after,
                    )
                self._record_idempotency(
                    scope=by_role,
                    key=idempotency_key,
                    operation="refund",
                    fingerprint=fingerprint,
                    outcome=refund_receipt,
                )
                _record_mutation(
                    self._conn,
                    table="logical_time",
                    key="world",
                    action="world.refund_order",
                    before=event_tick - 1,
                    after=event_tick,
                )
                self._set_clock(event_tick)
                commit_writes = tuple(
                    TableWrite(
                        table=table,
                        key=key,
                        op="create" if before is None else "update",
                        before=before,
                        after=after,
                    )
                    for table, key, before, after in writes
                ) + (
                    TableWrite(
                        table="logical_time",
                        key="world",
                        op="update",
                        before=event_tick - 1,
                        after=event_tick,
                    ),
                )
                self._append_world_commit(
                    commit_kind="transaction",
                    operation="refund",
                    authority_action="world.refund_order",
                    actor_id=by_role,
                    idempotency_key=idempotency_key,
                    subject_id=str(existing.order_id),
                    table_writes=commit_writes,
                    invariants_held=(
                        "atomic",
                        "allowlist:REFUNDABLE",
                        "idempotent",
                        "world-clock",
                        *(
                            (
                                "captured-payment-refunded"
                                if fully_refunded
                                else "captured-payment-partially-refunded",
                            )
                            if refunded_payment is not None
                            else ()
                        ),
                    ),
                    request_fingerprint=fingerprint,
                )
                if _manage_transaction:
                    self._conn.commit()
                return refund_receipt
            except BaseException:
                if _manage_transaction:
                    self._conn.rollback()
                raise

    # -- First-class payment, packing, and after-sales authority ---------

    def begin_evidence_window(self) -> int:
        """Return the durable commit cursor for a new evidence window."""

        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM world_commit_records"
            ).fetchone()
            return 0 if row is None else int(row["n"])

    @property
    def commit_journal(self) -> tuple[WorldCommitRecord, ...]:
        """Return the complete durable World commit journal in sequence order."""

        with self._lock:
            rows = self._conn.execute(
                "SELECT commit_json FROM world_commit_records ORDER BY sequence"
            ).fetchall()
            return tuple(
                _world_commit_from_json(str(row["commit_json"])) for row in rows
            )

    def commits_since(self, cursor: int) -> tuple[WorldCommitRecord, ...]:
        """Return durable commits at or after ``cursor``."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("commit cursor must be a non-negative integer")
        with self._lock:
            end = self.begin_evidence_window()
            if cursor > end:
                raise ValueError("commit cursor is beyond the end of the journal")
            rows = self._conn.execute(
                "SELECT commit_json FROM world_commit_records "
                "WHERE sequence >= ? ORDER BY sequence",
                (cursor,),
            ).fetchall()
            return tuple(
                _world_commit_from_json(str(row["commit_json"])) for row in rows
            )

    def authorize_payment(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        """Authorize payment and reserve inventory in one SQLite transaction."""

        normalized = normalize_payment_intent(intent)
        if normalized["op"] != "authorize":
            raise WorldError("authorize_payment requires an authorize intent")
        if by_actor != "platform:psp":
            raise WriteNotAuthorized("only platform:psp may authorize payment")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                order = cast(
                    "Order | None",
                    self._fetch_value("orders", cast(str, normalized["order_id"])),
                )
                if order is None:
                    raise WorldError("payment order does not exist")
                if original_actor != str(order.buyer_id):
                    raise WriteNotAuthorized("payment principal does not own order")
                fingerprint = after_sales_digest(
                    {
                        "intent": normalized,
                        "service_actor": by_actor,
                        "original_actor": original_actor,
                    }
                )
                replay = self._authority_operation_replay(
                    scope="apply_payment_intent",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PaymentStateRecord):
                        raise WorldError("payment idempotency outcome is missing")
                    self._conn.commit()
                    return outcome
                inventory = self._fetch_value("inventory", str(order.sku_id))
                if inventory is None or _available_qty(inventory) < order.qty:
                    raise OutOfStock(f"insufficient inventory for {order.sku_id}")
                _require_inventory_owner(order, inventory)
                tick = self._clock_value() + 1
                payment = derive_payment_authorization(
                    order,
                    original_actor=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
                payment_key = payment_state_key(payment)
                reserved = _reserve_inventory(inventory, qty=order.qty)
                self._commit_simple_authority_transition(
                    operation="authorize_payment",
                    authority_action="world.apply_payment_intent",
                    scope="apply_payment_intent",
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    subject_id=str(order.order_id),
                    outcome_table="payment_states",
                    outcome_key=payment_key,
                    tick=tick,
                    mutations=(
                        ("inventory", str(order.sku_id), inventory, reserved),
                        ("payment_states", payment_key, None, payment),
                    ),
                    invariants=(
                        "psp-authenticated",
                        "principal-order-bound",
                        "inventory-reserved",
                        "single-clock-advance",
                    ),
                )
                self._conn.commit()
                return payment
            except BaseException:
                self._conn.rollback()
                raise

    def capture_payment(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PaymentStateRecord:
        """Seal captured payment state to the exact persisted charge receipt."""

        normalized = normalize_payment_intent(intent)
        if normalized["op"] != "capture":
            raise WorldError("capture_payment requires a capture intent")
        if by_actor != "platform:psp":
            raise WriteNotAuthorized("only platform:psp may capture payment")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                order = cast(
                    "Order | None",
                    self._fetch_value("orders", cast(str, normalized["order_id"])),
                )
                if order is None:
                    raise WorldError("payment order does not exist")
                if original_actor != str(order.buyer_id):
                    raise WriteNotAuthorized("payment principal does not own order")
                fingerprint = after_sales_digest(
                    {
                        "intent": normalized,
                        "service_actor": by_actor,
                        "original_actor": original_actor,
                    }
                )
                replay = self._authority_operation_replay(
                    scope="apply_payment_intent",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PaymentStateRecord):
                        raise WorldError("payment idempotency outcome is missing")
                    self._conn.commit()
                    return outcome
                history = self._payment_history_unchecked(str(order.order_id))
                receipt = _charge_receipt_for_order(
                    self._ledger_projection(),
                    str(order.order_id),
                    payment=None,
                    history=history,
                )
                if receipt is None:
                    raise WorldError(
                        "payment capture requires a persisted charge receipt"
                    )
                previous = history[-1] if history else None
                tick = self._clock_value() + 1
                payment = derive_payment_capture(
                    order,
                    receipt,
                    previous=previous,
                    original_actor=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
                payment_key = payment_state_key(payment)
                self._commit_simple_authority_transition(
                    operation="capture_payment",
                    authority_action="world.apply_payment_intent",
                    scope="apply_payment_intent",
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    subject_id=str(order.order_id),
                    outcome_table="payment_states",
                    outcome_key=payment_key,
                    tick=tick,
                    mutations=(("payment_states", payment_key, None, payment),),
                    invariants=(
                        "psp-authenticated",
                        "principal-order-bound",
                        "ledger-receipt-bound",
                        "single-clock-advance",
                    ),
                )
                self._conn.commit()
                return payment
            except BaseException:
                self._conn.rollback()
                raise

    def apply_packing_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> PackingRecord:
        """Apply a compact fulfillment intent to durable packing state."""

        normalized = normalize_packing_intent(intent)
        if by_actor != "platform:fulfillment" and not by_actor.startswith(
            "merchant:"
        ):
            raise WriteNotAuthorized(
                "packing requires platform:fulfillment or the owning merchant route"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                order = cast(
                    "Order | None",
                    self._fetch_value("orders", cast(str, normalized["order_id"])),
                )
                if order is None:
                    raise WorldError("packing order does not exist")
                if original_actor != str(order.merchant_id):
                    raise WriteNotAuthorized("packing principal is not order merchant")
                if by_actor.startswith("merchant:") and by_actor != original_actor:
                    raise WriteNotAuthorized(
                        "merchant packing route is not owner bound"
                    )
                fingerprint = after_sales_digest(
                    {
                        "intent": normalized,
                        "service_actor": by_actor,
                        "original_actor": original_actor,
                    }
                )
                replay = self._authority_operation_replay(
                    scope="apply_packing_intent",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, PackingRecord):
                        raise WorldError("packing idempotency outcome is missing")
                    self._conn.commit()
                    return outcome
                payments = self._payment_history_unchecked(str(order.order_id))
                if not payments:
                    raise WorldError("packing requires first-class payment state")
                packings = self._packing_history_unchecked(str(order.order_id))
                previous = packings[-1] if packings else None
                target = {
                    "create": "created",
                    "pack": "packed",
                    "cancel": "cancelled",
                    "hand_off": "handed_off",
                }[normalized["op"]]
                tick = self._clock_value() + 1
                packing = derive_packing_transition(
                    order,
                    payments[-1],
                    previous=previous,
                    target_state=cast(Any, target),
                    original_actor=by_actor,
                    server_tick=tick,
                    idempotency_key=idempotency_key,
                )
                packing_key = packing_record_key(packing)
                self._commit_simple_authority_transition(
                    operation=f"packing_{normalized['op']}",
                    authority_action="world.apply_packing_intent",
                    scope="apply_packing_intent",
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    subject_id=str(order.order_id),
                    outcome_table="packing_records",
                    outcome_key=packing_key,
                    tick=tick,
                    mutations=(("packing_records", packing_key, None, packing),),
                    invariants=(
                        "fulfillment-service-authenticated",
                        "merchant-order-bound",
                        "payment-state-bound",
                        "single-clock-advance",
                    ),
                )
                self._conn.commit()
                return packing
            except BaseException:
                self._conn.rollback()
                raise

    def publish_after_sales_policy(
        self,
        policy_intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesPolicyRevision:
        """Publish a World-sealed merchant after-sales policy revision."""

        if by_actor != "platform:policy":
            raise WriteNotAuthorized(
                "only platform:policy may publish after-sales policy"
            )
        normalized = _normalize_after_sales_policy_intent(policy_intent)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                fingerprint = after_sales_digest(
                    {
                        "intent": normalized,
                        "service_actor": by_actor,
                        "original_actor": original_actor,
                    }
                )
                replay = self._authority_operation_replay(
                    scope="publish_after_sales_policy",
                    actor_id=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if replay is not None:
                    outcome = self._fetch_value(replay[0], replay[1])
                    if not isinstance(outcome, AfterSalesPolicyRevision):
                        raise WorldError("policy idempotency outcome is missing")
                    self._conn.commit()
                    return outcome
                current_row = self._conn.execute(
                    "SELECT * FROM after_sales_policies WHERE merchant_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (original_actor,),
                ).fetchone()
                current = (
                    None
                    if current_row is None
                    else cast(
                        AfterSalesPolicyRevision,
                        _row_to_value("after_sales_policies", current_row),
                    )
                )
                tick = self._clock_value() + 1
                policy = build_after_sales_policy_revision(
                    **normalized,
                    merchant_id=original_actor,
                    revision=1 if current is None else current.revision + 1,
                    previous_digest=None if current is None else current.policy_digest,
                    effective_tick=tick,
                    published_by_id=by_actor,
                )
                plan = stage_policy_publication(
                    self._after_sales_tables(),
                    policy,
                    original_actor=by_actor,
                    server_tick=tick,
                    trusted_publisher_ids=("platform:policy",),
                )
                self._commit_simple_authority_transition(
                    operation="publish_after_sales_policy",
                    authority_action="world.publish_after_sales_policy",
                    scope="publish_after_sales_policy",
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    subject_id=original_actor,
                    outcome_table="after_sales_policies",
                    outcome_key=plan.write.key,
                    tick=tick,
                    mutations=((
                        "after_sales_policies",
                        plan.write.key,
                        None,
                        policy,
                    ),),
                    invariants=(
                        "policy-service-authenticated",
                        "merchant-principal-bound",
                        "contiguous-policy-revision",
                        "single-clock-advance",
                    ),
                )
                self._conn.commit()
                return policy
            except BaseException:
                self._conn.rollback()
                raise

    def apply_after_sales_intent(
        self,
        intent: Mapping[str, Any],
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        """Plan and commit a compact after-sales intent in one transaction."""

        if by_actor != "platform:after-sales":
            raise WriteNotAuthorized(
                "only platform:after-sales may apply after-sales intents"
            )
        normalized = normalize_after_sales_intent(intent)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                context = self.load_after_sales_context(
                    cast(str, normalized["order_id"]),
                    intent=normalized,
                    original_actor=original_actor,
                )
                binding = context.tables.binding_for_order(
                    str(context.order.order_id), caller="world"
                )
                policy = _after_sales_policy_for_context(
                    context.tables,
                    merchant_id=str(context.order.merchant_id),
                    binding=binding,
                )
                if binding is None:
                    binding = derive_after_sales_binding(
                        order=context.order,
                        receipt=context.charge_receipt,
                        shipment=context.shipment,
                        policy=policy,
                        payment=context.payment,
                    )
                fingerprint = after_sales_intent_fingerprint(
                    normalized,
                    binding=binding,
                    policy=policy,
                    original_actor=original_actor,
                    evidence_records=context.evidence_records,
                )
                retry = self._after_sales_retry_result(
                    operation=cast(str, normalized["op"]),
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                )
                if retry is not None:
                    self._conn.commit()
                    return retry
                planned = AfterSalesPlanner().plan(
                    context,
                    normalized,
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                )
                if planned.commit is not None:
                    self._commit_after_sales(planned.commit)
                    self._conn.commit()
                    return planned
                self._conn.commit()
                return planned
            except BaseException:
                self._conn.rollback()
                raise

    def complete_ledger_reconciliation(
        self,
        order_id: str,
        request_id: str,
        *,
        by_actor: str,
        original_actor: str,
        idempotency_key: str,
    ) -> AfterSalesCommandResult:
        if by_actor != "platform:accounting":
            raise WriteNotAuthorized(
                "only platform:accounting may complete reconciliation"
            )
        if original_actor != by_actor:
            raise WriteNotAuthorized(
                "ledger reconciliation principal must be platform accounting"
            )
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                context = self.load_after_sales_context(order_id)
                planned = AfterSalesPlanner().plan_reconciliation_result(
                    context,
                    request_id=request_id,
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                )
                replay = self._after_sales_retry_result(
                    operation="complete_ledger_reconciliation",
                    original_actor=original_actor,
                    idempotency_key=idempotency_key,
                    request_fingerprint=planned.operation.request_fingerprint,
                    scope="complete_ledger_reconciliation",
                )
                if replay is not None:
                    self._conn.commit()
                    return replay
                if planned.commit is None:
                    raise WorldError("reconciliation plan has no commit")
                self._commit_after_sales_records(
                    planned.commit,
                    authority_action="world.complete_ledger_reconciliation",
                    scope="complete_ledger_reconciliation",
                )
                self._conn.commit()
                return planned
            except BaseException:
                self._conn.rollback()
                raise

    def load_after_sales_context(
        self,
        order_id: str,
        *,
        intent: Mapping[str, Any] | None = None,
        original_actor: str | None = None,
        evidence_digests: Mapping[str, str] | None = None,
    ) -> TrustedAfterSalesContext:
        """Load the authoritative SQLite rows consumed by the planner."""

        with self._lock:
            order = cast("Order | None", self._fetch_value("orders", order_id))
            if order is None:
                if (
                    isinstance(original_actor, str)
                    and original_actor.split(":", 1)[0] in {"buyer", "merchant"}
                ):
                    raise AfterSalesReferenceRejected(
                        "after_sales_order_not_found"
                    )
                raise WorldError(f"after-sales order does not exist: {order_id}")
            payments = self._payment_history_unchecked(order_id)
            if not payments:
                raise AfterSalesCoreTransitionError(
                    "after-sales requires first-class payment state"
                )
            payment = payments[-1]
            charge = _charge_receipt_for_order(
                self._ledger_projection(),
                order_id,
                payment=payment,
                history=payments,
            )
            packings = self._packing_history_unchecked(order_id)
            packing = packings[-1] if packings else None
            shipment_row = self._conn.execute(
                "SELECT * FROM shipments WHERE order_id = ?", (order_id,)
            ).fetchone()
            shipment = (
                None
                if shipment_row is None
                else cast(Shipment, _row_to_value("shipments", shipment_row))
            )
            timeline = cast(
                "OrderTimeline | None",
                self._fetch_value("order_timelines", order_id),
            )
            settled_at = (
                timeline.settled_at_tick
                if timeline is not None and timeline.settled_at_tick is not None
                else _receipt_tick(charge)
            )
            tables = self._after_sales_tables()
            binding = tables.binding_for_order(order_id, caller="world")
            policy = _after_sales_policy_for_context(
                tables,
                merchant_id=str(order.merchant_id),
                binding=binding,
            )

            def evidence_lookup(record_id: str) -> EvidenceRecord | None:
                expected = (
                    None
                    if evidence_digests is None
                    else evidence_digests.get(record_id)
                )
                if expected is None:
                    row = self._conn.execute(
                        "SELECT * FROM evidence_records WHERE record_id = ? "
                        "ORDER BY version DESC LIMIT 1",
                        (record_id,),
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT * FROM evidence_records "
                        "WHERE record_id = ? AND record_digest = ?",
                        (record_id, expected),
                    ).fetchone()
                return (
                    None
                    if row is None
                    else cast(
                        EvidenceRecord,
                        _row_to_value("evidence_records", row),
                    )
                )

            evidence_records = load_trusted_after_sales_evidence(
                intent,
                order=order,
                policy=policy,
                logical_tick=self._clock_value(),
                original_actor=original_actor,
                lookup=evidence_lookup,
            )
            if evidence_digests is not None and set(evidence_digests) != {
                record.record_id for record in evidence_records
            }:
                raise WorldError("trusted evidence digest bindings are incomplete")
            replacement = self._after_sales_replacement_context(
                order,
                intent=intent,
                tables=tables,
            )
            ledger_sources = ()
            if binding is not None:
                ledger_sources = tuple(
                    derive_ledger_reconciliation_source(
                        receipt,
                        binding=binding,
                        effect=receipt.effect,
                        server_tick=_receipt_tick(receipt),
                    )
                    for _, receipt in self.iter_table("ledger")
                    if str(receipt.order_id) == order_id
                )
            return TrustedAfterSalesContext(
                logical_tick=self._clock_value(),
                order=order,
                payment=payment,
                charge_receipt=charge,
                packing=packing,
                shipment=shipment,
                settled_at_tick=settled_at,
                tables=tables,
                replacement=replacement,
                evidence_records=evidence_records,
                ledger_sources=ledger_sources,
            )

    def _after_sales_replacement_context(
        self,
        order: Order,
        *,
        intent: Mapping[str, Any] | None,
        tables: AfterSalesTables,
    ) -> TrustedReplacementContext | None:
        if intent is None:
            return None
        candidate = intent.get("replacement_sku_id")
        case_id: str | None = None
        if candidate is None and intent.get("case_id") is not None:
            requested_case = str(intent["case_id"])
            rows = [
                row
                for _, row in tables.internal_all("exchange_cases")
            ]
            if rows:
                latest = resolve_current_versioned_record(
                    tables,
                    "exchange_cases",
                    rows,
                    reference=requested_case,
                    identity_field="case_id",
                )
                candidate = latest.replacement_sku_id
                case_id = latest.case_id
        if candidate is None:
            return None
        sku_id = str(candidate)
        listing = self._fetch_value("catalog", sku_id)
        inventory = self._fetch_value("inventory", sku_id)
        if not isinstance(listing, Listing) or not isinstance(
            inventory, InventoryRow
        ):
            return None
        replacement_order = Order(
            order_id=OrderId(
                f"replacement:{case_id}"
                if case_id is not None
                else f"replacement:pending:{order.order_id}:{sku_id}"
            ),
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            sku_id=SkuId(sku_id),
            qty=order.qty,
            agreed_price=order.agreed_price,
            state=OrderState.SETTLED,
            request_order=order.request_order,
        )
        return TrustedReplacementContext(
            sku_id=sku_id,
            merchant_id=str(listing.merchant_id),
            available_qty=_available_qty(inventory),
            listing_digest=after_sales_digest(
                {
                    "sku_id": str(listing.sku_id),
                    "merchant_id": str(listing.merchant_id),
                    "product_id": listing.product_id,
                    "name": listing.name,
                    "category": listing.category,
                    "attributes": listing.attributes,
                    "list_price": str(listing.list_price.amount),
                    "currency": listing.list_price.currency,
                }
            ),
            inventory_digest=after_sales_digest(
                {
                    "sku_id": str(inventory.sku_id),
                    "merchant_id": str(inventory.merchant_id),
                    "qty_available": inventory.qty_available,
                    "qty_reserved": inventory.qty_reserved,
                    "eta_day": inventory.eta_day,
                    "version": inventory.version,
                }
            ),
            projected_order_digest=authoritative_order_digest(replacement_order),
        )

    def payment_history(
        self, order_id: str, *, caller: str
    ) -> tuple[PaymentStateRecord, ...]:
        with self._lock:
            return tuple(
                row
                for row in self._payment_history_unchecked(order_id)
                if _visible("payment_states", row, caller)
            )

    def ledger_history(self, order_id: str, *, caller: str) -> tuple[Receipt, ...]:
        """Return exact immutable receipts after order-party authorization."""

        with self._lock:
            order = self._fetch_value("orders", order_id)
            if order is None or not _visible("orders", order, caller):
                return ()
            rows = self._conn.execute(
                "SELECT * FROM ledger WHERE order_id = ? ORDER BY txn_id",
                (order_id,),
            ).fetchall()
            receipts = tuple(
                cast(Receipt, _row_to_value("ledger", row)) for row in rows
            )
            return tuple(
                receipt
                for receipt in receipts
                if _visible("ledger", receipt, caller)
            )

    def packing_history(
        self, order_id: str, *, caller: str
    ) -> tuple[PackingRecord, ...]:
        with self._lock:
            return tuple(
                row
                for row in self._packing_history_unchecked(order_id)
                if _visible("packing_records", row, caller)
            )

    def after_sales_history(self, order_id: str, *, caller: str) -> tuple[Any, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM after_sales_records WHERE order_id = ? "
                "ORDER BY logical_tick, record_kind, domain_key",
                (order_id,),
            ).fetchall()
            writes = tuple(
                cast(AfterSalesWrite, _row_to_value("after_sales_records", row))
                for row in rows
            )
            return tuple(
                write.value
                for write in writes
                if _visible("after_sales_records", write, caller)
            )

    def after_sales_result_record(
        self,
        operation: AfterSalesOperationRecord,
        *,
        caller: str,
    ) -> AfterSalesWrite | None:
        """Resolve one operation's exact persisted result under row ACLs."""

        with self._lock:
            physical_key = physical_after_sales_record_lookup_key(
                operation.result_table, operation.result_key
            )
            write = self._fetch_value("after_sales_records", physical_key)
            if not isinstance(write, AfterSalesWrite):
                return None
            if not _visible("after_sales_records", write, caller):
                return None
            if (
                write.table != operation.result_table
                or write.key != operation.result_key
                or after_sales_record_digest(write.value)
                != operation.result_digest
            ):
                return None
            return write

    def after_sales_policy(
        self, merchant_id: str, *, caller: str
    ) -> AfterSalesPolicyRevision | None:
        """Return the latest typed merchant policy to an authorized reader."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM after_sales_policies WHERE merchant_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (merchant_id,),
            ).fetchone()
            if row is None:
                return None
            policy = cast(
                AfterSalesPolicyRevision,
                _row_to_value("after_sales_policies", row),
            )
            return policy if _visible("after_sales_policies", policy, caller) else None

    def _payment_history_unchecked(
        self, order_id: str
    ) -> tuple[PaymentStateRecord, ...]:
        rows = self._conn.execute(
            "SELECT * FROM payment_states WHERE order_id = ? ORDER BY version",
            (order_id,),
        ).fetchall()
        return tuple(
            cast(PaymentStateRecord, _row_to_value("payment_states", row))
            for row in rows
        )

    def _packing_history_unchecked(self, order_id: str) -> tuple[PackingRecord, ...]:
        rows = self._conn.execute(
            "SELECT * FROM packing_records WHERE order_id = ? ORDER BY version",
            (order_id,),
        ).fetchall()
        return tuple(
            cast(PackingRecord, _row_to_value("packing_records", row))
            for row in rows
        )

    def _ledger_projection(self) -> LedgerTable:
        table = LedgerTable()
        for key, receipt in self.iter_table("ledger"):
            table.write(key, receipt)
        return table

    def _after_sales_tables(self) -> AfterSalesTables:
        projection = AfterSalesTables()
        for _, policy in self.iter_table("after_sales_policies"):
            projection.append(
                AfterSalesWrite(
                    "after_sales_policies",
                    f"{policy.merchant_id}:{policy.revision}",
                    policy,
                )
            )
        for _, write in self.iter_table("after_sales_records"):
            projection.append(write)
        return projection

    def _after_sales_retry_result(
        self,
        *,
        operation: str,
        original_actor: str,
        idempotency_key: str,
        request_fingerprint: str,
        scope: str = "apply_after_sales_intent",
    ) -> AfterSalesCommandResult | None:
        key = authority_operation_key(
            scope, original_actor, idempotency_key
        )
        authority = cast(
            "AuthorityOperationRecord | None",
            self._fetch_value("authority_operations", key),
        )
        if authority is None:
            return None
        if authority.request_fingerprint != request_fingerprint:
            raise IdempotencyConflict(
                "after-sales idempotency key was reused for another intent"
            )
        stored = cast(
            "AfterSalesWrite | None",
            self._fetch_value("after_sales_records", authority.outcome_key),
        )
        if stored is None:
            raise WorldError("after-sales idempotency outcome is missing")
        effects = _effects_for_persisted_after_sales(stored.value)
        binding = getattr(stored.value, "binding", None)
        if binding is None:
            raise WorldError("after-sales result has no order binding")
        operation_row = build_after_sales_operation(
            operation=operation,
            order_id=binding.order_id,
            binding_digest=binding.binding_digest,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            logical_tick=getattr(stored.value, "logical_tick"),
            result_table=stored.table,
            result_key=stored.key,
            result_digest=after_sales_record_digest(stored.value),
            effects=effects,
        )
        return AfterSalesCommandResult("idempotent", operation_row, None)

    def _commit_after_sales(self, commit: AfterSalesCommit) -> None:
        current = self.load_after_sales_context(
            commit.operation.order_id,
            intent=_after_sales_context_hint(commit),
            original_actor=commit.operation.actor_id,
            evidence_digests=_after_sales_context_evidence_digests(commit),
        )
        if commit.expected_logical_tick != self._clock_value():
            raise AfterSalesContextConflict("World logical time changed")
        if commit.context_digest != current.context_digest:
            raise AfterSalesContextConflict("World after-sales context changed")
        if len(commit.commerce_effects) > 1:
            raise WorldError("after-sales commit has multiple commerce effects")
        if not commit.commerce_effects:
            self._commit_after_sales_records(commit)
            return
        effect = commit.commerce_effects[0]
        if effect.kind == "cancel_paid_order":
            self._commit_paid_cancellation(commit, effect, current)
        elif effect.kind == "mark_returned":
            self._commit_return_receipt(commit, effect, current)
        elif effect.kind == "issue_refund":
            self._commit_after_sales_refund(commit, effect, current)
        elif effect.kind == "complete_exchange":
            self._commit_after_sales_exchange(commit, effect, current)
        else:
            raise WorldError(
                f"unsupported after-sales commerce effect: {effect.kind}"
            )

    def _commit_after_sales_records(
        self,
        commit: AfterSalesCommit,
        *,
        extra_mutations: tuple[tuple[str, str, Any | None, Any], ...] = (),
        authority_action: str = "world.apply_after_sales_intent",
        scope: str = "apply_after_sales_intent",
        extra_invariants: tuple[str, ...] = (),
    ) -> None:
        core_mutations = self._after_sales_core_projection_mutations(commit)
        mutations: list[tuple[str, str, Any | None, Any]] = [
            *extra_mutations,
            *core_mutations,
        ]
        for write in commit.writes:
            if write.table == "after_sales_policies":
                raise WorldError("after-sales command cannot publish policy")
            physical_key = physical_after_sales_record_key(write)
            mutations.append(("after_sales_records", physical_key, None, write))
        result_write = next(
            write
            for write in commit.writes
            if write.table == commit.operation.result_table
            and write.key == commit.operation.result_key
        )
        self._commit_simple_authority_transition(
            operation=commit.operation.operation,
            authority_action=authority_action,
            scope=scope,
            original_actor=commit.operation.actor_id,
            idempotency_key=commit.operation.idempotency_key,
            request_fingerprint=commit.operation.request_fingerprint,
            subject_id=commit.operation.order_id,
            outcome_table="after_sales_records",
            outcome_key=physical_after_sales_record_key(result_write),
            tick=commit.logical_tick,
            mutations=tuple(mutations),
            invariants=(
                "after-sales-context-cas",
                "typed-causal-records",
                "principal-provenance-preserved",
                "single-clock-advance",
                *extra_invariants,
            ),
        )

    def _after_sales_core_projection_mutations(
        self, commit: AfterSalesCommit
    ) -> tuple[tuple[str, str, Any | None, Any], ...]:
        mutations: list[tuple[str, str, Any | None, Any]] = []
        for write in commit.writes:
            value = write.value
            if write.table == "dispute_cases":
                dispute_id = str(value.dispute_id)
                before = self._fetch_value("disputes", dispute_id)
                after = Dispute(
                    dispute_id=DisputeId(dispute_id),
                    order_id=OrderId(value.binding.order_id),
                    filed_by=AgentId(value.filed_by_id),
                    against=AgentId(value.against_id),
                    reason=value.reason,
                    state=DisputeState(value.state),
                )
                if before is not None and (
                    str(before.order_id) != value.binding.order_id
                    or str(before.filed_by) != value.filed_by_id
                    or str(before.against) != value.against_id
                    or before.reason != value.reason
                ):
                    raise AfterSalesContextConflict(
                        "typed dispute conflicts with core dispute identity"
                    )
                mutations.append(("disputes", dispute_id, before, after))
            elif write.table == "after_sales_rulings":
                ruling_id = str(value.ruling_id)
                refund_amount = (
                    None
                    if value.refund_amount == 0
                    else Money(
                        (Decimal(value.refund_amount) / Decimal(100)).quantize(
                            Decimal("0.01")
                        ),
                        value.binding.currency,
                    )
                )
                ruling = Ruling(
                    ruling_id=RulingId(ruling_id),
                    dispute_id=DisputeId(str(value.dispute_id)),
                    in_favor_of=AgentId(str(value.winner_id)),
                    rationale=str(value.rationale),
                    refund_amount=refund_amount,
                )
                before = self._fetch_value("rulings", ruling_id)
                mutations.append(("rulings", ruling_id, before, ruling))
        return tuple(mutations)

    def _commit_return_receipt(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        if order.state != OrderState.DISPATCHED:
            raise AfterSalesContextConflict(
                "physical return receipt requires a dispatched order"
            )
        if effect.order_id != str(order.order_id) or effect.qty > order.qty:
            raise AfterSalesContextConflict("return receipt order identity changed")
        returned = replace(order, state=OrderState.RETURNED)
        timeline = cast(
            "OrderTimeline | None",
            self._fetch_value("order_timelines", str(order.order_id)),
        )
        updated_timeline = (
            OrderTimeline(
                order_id=order.order_id,
                buyer_id=order.buyer_id,
                merchant_id=order.merchant_id,
                returned_at_tick=commit.logical_tick,
            )
            if timeline is None
            else replace(timeline, returned_at_tick=commit.logical_tick)
        )
        self._commit_after_sales_records(
            commit,
            extra_mutations=(
                ("orders", str(order.order_id), order, returned),
                (
                    "order_timelines",
                    str(order.order_id),
                    timeline,
                    updated_timeline,
                ),
            ),
            extra_invariants=(
                "physical-return-recorded",
                "inventory-reservation-preserved",
                "payment-state-preserved",
            ),
        )

    def _commit_after_sales_refund(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        if order.state not in {
            OrderState.SETTLED,
            OrderState.DISPATCHED,
            OrderState.RETURNED,
        }:
            raise AfterSalesContextConflict("order is not refundable")
        if context.payment.record_digest != effect.payment_before_digest:
            raise AfterSalesContextConflict("payment changed before refund")
        if effect.amount <= 0:
            raise WorldError("refund commerce effect amount must be positive")
        inventory = self._fetch_value("inventory", str(order.sku_id))
        if not isinstance(inventory, InventoryRow):
            raise WorldError("refund requires indexed inventory")
        if inventory.qty_reserved < effect.inventory_release_qty:
            raise WorldError("refund inventory reservation underflow")
        refund = _after_sales_refund_receipt(
            order,
            amount=effect.amount,
            tick=commit.logical_tick,
            idempotency_key=commit.operation.idempotency_key,
        )
        payment = derive_payment_resolution(
            order,
            context.payment,
            outcome="refunded",
            refund_receipt=refund,
            original_actor="platform:psp",
            server_tick=commit.logical_tick,
            idempotency_key=f"payment:{commit.operation.idempotency_key}",
        )
        fully_refunded = payment.state == "refunded"
        refunded_order = (
            replace(order, state=OrderState.REFUNDED) if fully_refunded else order
        )
        timeline = cast(
            "OrderTimeline | None",
            self._fetch_value("order_timelines", str(order.order_id)),
        )
        updated_timeline = (
            None
            if not fully_refunded
            else (
                OrderTimeline(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    merchant_id=order.merchant_id,
                    refunded_at_tick=commit.logical_tick,
                )
                if timeline is None
                else replace(timeline, refunded_at_tick=commit.logical_tick)
            )
        )
        mutations: list[tuple[str, str, Any | None, Any]] = []
        if fully_refunded:
            mutations.append(
                ("orders", str(order.order_id), order, refunded_order)
            )
        if effect.inventory_release_qty:
            released = replace(
                inventory,
                qty_reserved=inventory.qty_reserved
                - effect.inventory_release_qty,
                version=inventory.version + 1,
            )
            mutations.append(
                ("inventory", str(order.sku_id), inventory, released)
            )
        mutations.extend(
            (
                ("ledger", str(refund.txn_id), None, refund),
                ("payment_states", payment_state_key(payment), None, payment),
            )
        )
        if updated_timeline is not None:
            mutations.append(
                (
                    "order_timelines",
                    str(order.order_id),
                    timeline,
                    updated_timeline,
                )
            )
        self._commit_after_sales_records(
            commit,
            extra_mutations=tuple(mutations),
            extra_invariants=(
                "refund-ledger-payment-atomic",
                (
                    "order-fully-refunded"
                    if fully_refunded
                    else "order-state-preserved-for-partial-refund"
                ),
                (
                    "returned-quantity-released"
                    if effect.inventory_release_qty
                    else "dispute-does-not-restock"
                ),
            ),
        )

    def _commit_after_sales_exchange(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        if order.state != OrderState.RETURNED:
            raise AfterSalesContextConflict(
                "exchange completion requires a returned order"
            )
        result_write = next(
            write
            for write in commit.writes
            if write.table == commit.operation.result_table
            and write.key == commit.operation.result_key
        )
        case = result_write.value
        if getattr(case, "state", None) != "completed":
            raise WorldError("exchange completion result is not completed")
        replacement_sku = str(effect.replacement_sku_id)
        original_inventory = self._fetch_value("inventory", str(order.sku_id))
        replacement_inventory = self._fetch_value("inventory", replacement_sku)
        replacement_listing = self._fetch_value("catalog", replacement_sku)
        if not isinstance(original_inventory, InventoryRow) or not isinstance(
            replacement_inventory, InventoryRow
        ):
            raise WorldError("exchange requires both inventory rows")
        if not isinstance(replacement_listing, Listing) or str(
            replacement_listing.merchant_id
        ) != str(order.merchant_id):
            raise WorldError("replacement listing is not merchant-owned")
        if original_inventory.qty_reserved < effect.inventory_release_qty:
            raise WorldError("exchange original reservation underflow")
        if _available_qty(replacement_inventory) < effect.qty:
            raise OutOfStock("replacement inventory cannot satisfy exchange")
        replacement_order = Order(
            order_id=OrderId(f"replacement:{case.case_id}"),
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            sku_id=SkuId(replacement_sku),
            qty=effect.qty,
            agreed_price=order.agreed_price,
            state=OrderState.SETTLED,
            request_order=order.request_order,
        )
        if authoritative_order_digest(replacement_order) != case.completion_order_digest:
            raise AfterSalesContextConflict(
                "replacement order does not match planned completion digest"
            )
        exchanged = replace(order, state=OrderState.EXCHANGED)
        exchange = Exchange(
            exchange_id=ExchangeId(str(case.case_id)),
            original_order_id=order.order_id,
            replacement_order_id=replacement_order.order_id,
            buyer_id=order.buyer_id,
            merchant_id=order.merchant_id,
            original_sku_id=order.sku_id,
            replacement_sku_id=SkuId(replacement_sku),
            qty=effect.qty,
            created_by=AgentId(commit.operation.actor_id),
            idempotency_key=commit.operation.idempotency_key,
        )
        if str(order.sku_id) == replacement_sku:
            final_inventory = replace(
                original_inventory,
                version=original_inventory.version + 1,
            )
            inventory_mutations = ((
                "inventory",
                str(order.sku_id),
                original_inventory,
                final_inventory,
            ),)
        else:
            released = replace(
                original_inventory,
                qty_reserved=original_inventory.qty_reserved
                - effect.inventory_release_qty,
                version=original_inventory.version + 1,
            )
            reserved = replace(
                replacement_inventory,
                qty_reserved=replacement_inventory.qty_reserved + effect.qty,
                version=replacement_inventory.version + 1,
            )
            inventory_mutations = (
                ("inventory", str(order.sku_id), original_inventory, released),
                (
                    "inventory",
                    replacement_sku,
                    replacement_inventory,
                    reserved,
                ),
            )
        self._commit_after_sales_records(
            commit,
            extra_mutations=(
                ("orders", str(order.order_id), order, exchanged),
                (
                    "orders",
                    str(replacement_order.order_id),
                    None,
                    replacement_order,
                ),
                *inventory_mutations,
                ("exchanges", str(exchange.exchange_id), None, exchange),
            ),
            extra_invariants=(
                "returned-original",
                "replacement-world-grounded",
                "no-second-payment",
                "no-oversell",
            ),
        )

    def _commit_paid_cancellation(
        self,
        commit: AfterSalesCommit,
        effect: CommerceEffect,
        context: TrustedAfterSalesContext,
    ) -> None:
        order = context.order
        inventory = self._fetch_value("inventory", str(order.sku_id))
        if not isinstance(inventory, InventoryRow):
            raise WorldError("paid cancellation requires indexed inventory row")
        if inventory.qty_reserved < effect.inventory_release_qty:
            raise WorldError("paid cancellation reservation underflow")
        if context.payment.record_digest != effect.payment_before_digest:
            raise AfterSalesContextConflict("payment changed before cancellation")
        if effect.packing_before_digest is not None and (
            context.packing is None
            or context.packing.record_digest != effect.packing_before_digest
        ):
            raise AfterSalesContextConflict("packing changed before cancellation")
        cancelled_order = replace(order, state=OrderState.CANCELLED)
        released_inventory = replace(
            inventory,
            qty_reserved=inventory.qty_reserved - effect.inventory_release_qty,
            version=inventory.version + 1,
        )
        refund: Receipt | None = None
        if effect.financial_effect == "refund":
            refund = _after_sales_refund_receipt(
                order,
                amount=effect.amount,
                tick=commit.logical_tick,
                idempotency_key=commit.operation.idempotency_key,
            )
            payment = derive_payment_resolution(
                cancelled_order,
                context.payment,
                outcome="refunded",
                refund_receipt=refund,
                original_actor="platform:psp",
                server_tick=commit.logical_tick,
                idempotency_key=f"payment:{commit.operation.idempotency_key}",
            )
        else:
            payment = derive_payment_resolution(
                cancelled_order,
                context.payment,
                outcome="voided",
                refund_receipt=None,
                original_actor="platform:psp",
                server_tick=commit.logical_tick,
                idempotency_key=f"payment:{commit.operation.idempotency_key}",
            )
        packing: PackingRecord | None = None
        if context.packing is not None:
            packing = derive_packing_transition(
                cancelled_order,
                payment,
                previous=context.packing,
                target_state="cancelled",
                original_actor="platform:fulfillment",
                server_tick=commit.logical_tick,
                idempotency_key=f"packing:{commit.operation.idempotency_key}",
            )
        mutations: list[tuple[str, str, Any | None, Any]] = [
            ("orders", str(order.order_id), order, cancelled_order),
            ("inventory", str(order.sku_id), inventory, released_inventory),
        ]
        if refund is not None:
            mutations.append(("ledger", str(refund.txn_id), None, refund))
        mutations.append(
            ("payment_states", payment_state_key(payment), None, payment)
        )
        if packing is not None:
            mutations.append(
                ("packing_records", packing_record_key(packing), None, packing)
            )
        for write in commit.writes:
            if write.table == "after_sales_policies":
                raise WorldError("after-sales command cannot publish policy")
            physical_key = physical_after_sales_record_key(write)
            mutations.append(("after_sales_records", physical_key, None, write))
        result_write = next(
            write
            for write in commit.writes
            if write.table == commit.operation.result_table
            and write.key == commit.operation.result_key
        )
        self._commit_simple_authority_transition(
            operation=commit.operation.operation,
            authority_action="world.apply_after_sales_intent",
            scope="apply_after_sales_intent",
            original_actor=commit.operation.actor_id,
            idempotency_key=commit.operation.idempotency_key,
            request_fingerprint=commit.operation.request_fingerprint,
            subject_id=commit.operation.order_id,
            outcome_table="after_sales_records",
            outcome_key=physical_after_sales_record_key(result_write),
            tick=commit.logical_tick,
            mutations=tuple(mutations),
            invariants=(
                "after-sales-context-cas",
                "payment-packing-causal-binding",
                "atomic-commercial-effect",
                "principal-provenance-preserved",
                "single-clock-advance",
            ),
        )

    def _commit_simple_authority_transition(
        self,
        *,
        operation: str,
        authority_action: str,
        scope: str,
        original_actor: str,
        idempotency_key: str,
        request_fingerprint: str,
        subject_id: str,
        outcome_table: str,
        outcome_key: str,
        tick: int,
        mutations: tuple[tuple[str, str, Any | None, Any], ...],
        invariants: tuple[str, ...],
    ) -> None:
        before_tick = self._clock_value()
        if tick != before_tick + 1:
            raise LogicalTimeError(
                "authority transaction must advance one World tick"
            )
        authority_key = authority_operation_key(
            scope, original_actor, idempotency_key
        )
        authority = AuthorityOperationRecord(
            operation_key=authority_key,
            scope=scope,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            outcome_table=outcome_table,
            outcome_key=outcome_key,
        )
        all_mutations = (
            *mutations,
            ("authority_operations", authority_key, None, authority),
        )
        writes: list[TableWrite] = []
        for table_name, key, before, after in all_mutations:
            current = self._fetch_value(table_name, key)
            if current != before:
                raise AfterSalesContextConflict(
                    f"{table_name}:{key} changed before authority commit"
                )
            _upsert(self._conn, table_name, after)
            _record_mutation(
                self._conn,
                table=table_name,
                key=key,
                action=authority_action,
                before=before,
                after=after,
            )
            writes.append(
                TableWrite(
                    table=table_name,
                    key=key,
                    op="create" if before is None else "update",
                    before=before,
                    after=after,
                )
            )
        _record_mutation(
            self._conn,
            table="logical_time",
            key="world",
            action=authority_action,
            before=before_tick,
            after=tick,
        )
        self._set_clock(tick)
        writes.append(
            TableWrite(
                table="logical_time",
                key="world",
                op="update",
                before=before_tick,
                after=tick,
            )
        )
        self._append_world_commit(
            commit_kind="transaction",
            operation=operation,
            authority_action=authority_action,
            actor_id=original_actor,
            idempotency_key=idempotency_key,
            subject_id=subject_id,
            table_writes=tuple(writes),
            invariants_held=invariants,
            request_fingerprint=request_fingerprint,
        )

    def _append_world_commit(
        self,
        *,
        commit_kind: str,
        operation: str,
        authority_action: str,
        actor_id: str | None,
        idempotency_key: str | None,
        subject_id: str,
        table_writes: tuple[TableWrite, ...],
        invariants_held: tuple[str, ...],
        request_fingerprint: str | None = None,
    ) -> WorldCommitRecord:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM world_commit_records"
        ).fetchone()
        sequence = 0 if row is None else int(row["n"])
        record = WorldCommitRecord(
            sequence=sequence,
            commit_id=f"world-commit:{sequence:08d}",
            commit_kind=commit_kind,
            operation=operation,
            authority_action=authority_action,
            actor_id=actor_id,
            idempotency_key=idempotency_key,
            subject_id=subject_id,
            table_writes=table_writes,
            invariants_held=invariants_held,
            request_fingerprint=request_fingerprint,
        )
        self._conn.execute(
            "INSERT INTO world_commit_records "
            "(sequence, commit_id, commit_json, created_at) "
            "VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (sequence, record.commit_id, _world_commit_to_json(record)),
        )
        return record

    def clear_table(self, table: str) -> None:
        _assert_table(table)
        with self._lock, self._conn:
            if table == "search_sessions":
                self._conn.execute("DELETE FROM search_session_offers")
            if table == "reputation_settlements":
                self._conn.execute("DELETE FROM reputation_settlement_sources")
            if table == "listing_claims":
                self._conn.execute("DELETE FROM listing_claim_keys")
            self._conn.execute(f"DELETE FROM {table}")
            if table == "orders":
                self._conn.execute("DELETE FROM order_state_revisions")

    def reset(self, mode: Literal["E0", "E1"]) -> None:
        persistent = (
            {
                "orders",
                "ledger",
                "reputation",
                "reputation_settlements",
                "disputes",
                "rulings",
                "fulfillments",
                "exchanges",
                "shipments",
                "order_timelines",
                "order_groups",
                "search_sessions",
                "match_acceptances",
                "match_certificates",
                "supply_purchase_authorities",
                "negotiation_events",
                "negotiation_threads",
                "pricing_policy_revisions",
                "persistent_cart_quote_requests",
                "persistent_cart_quotes",
                "protocol_events",
                "protocol_receipts",
                "evidence_records",
                "mandate_authorities",
                "mandate_revisions",
                "listing_claims",
                "authority_operations",
                "payment_states",
                "packing_records",
                "after_sales_policies",
                "after_sales_records",
                "governance_policies",
                "governance_records",
                "reviews",
            }
            if mode == "E1"
            else set()
        )
        with self._lock, self._conn:
            if "search_sessions" not in persistent:
                self._conn.execute("DELETE FROM search_session_offers")
            if "reputation_settlements" not in persistent:
                self._conn.execute("DELETE FROM reputation_settlement_sources")
            if "listing_claims" not in persistent:
                self._conn.execute("DELETE FROM listing_claim_keys")
            for table in reversed(_TABLE_ORDER):
                if table not in persistent:
                    self._conn.execute(f"DELETE FROM {table}")
            if mode == "E0":
                self._conn.execute("DELETE FROM idempotency_records")
                self._conn.execute("DELETE FROM supply_event_records")
                self._conn.execute("DELETE FROM allocation_batch_records")
                self._conn.execute("DELETE FROM shipment_event_records")
                self._conn.execute("DELETE FROM shipment_resolution_records")
                self._conn.execute("DELETE FROM order_state_revisions")
                self._set_clock(0)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            catalog=tuple(value for _, value in self.iter_table("catalog")),
            inventory=dict(self.iter_table("inventory")),
            orders=tuple(value for _, value in self.iter_table("orders")),
            ledger=tuple(value for _, value in self.iter_table("ledger")),
            reputation=dict(self.iter_table("reputation")),
            reputation_settlements=tuple(
                value for _, value in self.iter_table("reputation_settlements")
            ),
            disputes=tuple(value for _, value in self.iter_table("disputes")),
            rulings=tuple(value for _, value in self.iter_table("rulings")),
            reviews=tuple(value for _, value in self.iter_table("reviews")),
            fulfillments=tuple(
                value for _, value in self.iter_table("fulfillments")
            ),
            exchanges=tuple(value for _, value in self.iter_table("exchanges")),
            shipments=tuple(value for _, value in self.iter_table("shipments")),
            logical_time=self._clock_value(),
            order_timelines=tuple(
                value for _, value in self.iter_table("order_timelines")
            ),
            order_groups=tuple(
                value for _, value in self.iter_table("order_groups")
            ),
            search_sessions=tuple(
                value for _, value in self.iter_table("search_sessions")
            ),
            match_acceptances=dict(self.iter_table("match_acceptances")),
            match_certificates=tuple(
                value for _, value in self.iter_table("match_certificates")
            ),
            supply_purchase_authorities=tuple(
                value
                for _, value in self.iter_table(
                    "supply_purchase_authorities"
                )
            ),
            protocol_events=tuple(
                value for _, value in self.iter_table("protocol_events")
            ),
            protocol_receipts=tuple(
                value for _, value in self.iter_table("protocol_receipts")
            ),
            negotiation_events=tuple(
                value for _, value in self.iter_table("negotiation_events")
            ),
            negotiation_threads=tuple(
                value for _, value in self.iter_table("negotiation_threads")
            ),
            pricing_policy_revisions=tuple(
                value
                for _, value in self.iter_table("pricing_policy_revisions")
            ),
            persistent_cart_quote_requests=tuple(
                value
                for _, value in self.iter_table(
                    "persistent_cart_quote_requests"
                )
            ),
            persistent_cart_quotes=tuple(
                value for _, value in self.iter_table("persistent_cart_quotes")
            ),
            evidence_records=tuple(
                value for _, value in self.iter_table("evidence_records")
            ),
            mandate_authorities=tuple(
                value for _, value in self.iter_table("mandate_authorities")
            ),
            mandate_revisions=tuple(
                value for _, value in self.iter_table("mandate_revisions")
            ),
            listing_claims=tuple(
                value for _, value in self.iter_table("listing_claims")
            ),
            authority_operations=tuple(
                value for _, value in self.iter_table("authority_operations")
            ),
            payment_states=tuple(
                value for _, value in self.iter_table("payment_states")
            ),
            packing_records=tuple(
                value for _, value in self.iter_table("packing_records")
            ),
            after_sales_policies=tuple(
                value for _, value in self.iter_table("after_sales_policies")
            ),
            after_sales_records=tuple(
                value for _, value in self.iter_table("after_sales_records")
            ),
            governance_policies=tuple(
                value for _, value in self.iter_table("governance_policies")
            ),
            governance_records=tuple(
                value for _, value in self.iter_table("governance_records")
            ),
            order_state_revisions={
                str(row["order_id"]): int(row["revision"])
                for row in self._conn.execute(
                    "SELECT order_id, revision FROM order_state_revisions "
                    "ORDER BY order_id"
                )
            },
        )

    def search_catalog(
        self,
        query: str,
        filters: dict[str, object] | None = None,
        *,
        limit: int | None = None,
    ) -> list[Listing]:
        filters = filters or {}
        if limit is not None and limit <= 0:
            return []
        terms = query.casefold().split()
        where = ["status = 'active'"]
        params: list[Any] = []
        indexed_filters: set[str] = set()
        for key in ("merchant_id", "product_id", "category"):
            expected = filters.get(key)
            if expected is None or isinstance(expected, (set, tuple, list, frozenset)):
                continue
            where.append(f"{key} = ?")
            params.append(str(expected))
            indexed_filters.add(key)
        sql = "SELECT * FROM catalog WHERE " + " AND ".join(where) + " ORDER BY sku_id"
        remaining_filters = {
            key: value for key, value in filters.items() if key not in indexed_filters
        }
        matches: list[Listing] = []
        for raw in self._conn.execute(sql, params):
            row = cast(Listing, _row_to_value("catalog", raw))
            haystack = " ".join(
                [
                    str(row.sku_id),
                    row.category,
                    row.name,
                    " ".join(f"{k} {v}" for k, v in sorted(row.attributes.items())),
                ]
            ).casefold()
            if terms and not all(term in haystack for term in terms):
                continue
            if not _matches_filters(row, remaining_filters):
                continue
            matches.append(row)
            if limit is not None and len(matches) >= limit:
                break
        return matches

    def iter_table(self, table: str) -> Iterator[tuple[Any, Any]]:
        _assert_table(table)
        key = _KEY_COLUMNS[table]
        rows = self._conn.execute(f"SELECT * FROM {table} ORDER BY {key}").fetchall()
        for row in rows:
            value = _row_to_value(table, row)
            if table in {
                "match_acceptances",
                "evidence_records",
                "mandate_revisions",
                "pricing_policy_revisions",
                "payment_states",
                "packing_records",
                "after_sales_policies",
                "after_sales_records",
                "governance_policies",
                "governance_records",
            }:
                yield str(row[_KEY_COLUMNS[table]]), value
            else:
                yield getattr(value, _KEY_ATTRS[table]), value

    def _fetch(self, table: str, key: str) -> sqlite3.Row | None:
        _assert_table(table)
        key_column = _KEY_COLUMNS[table]
        row = self._conn.execute(
            f"SELECT * FROM {table} WHERE {key_column} = ?",
            (key,),
        ).fetchone()
        return cast("sqlite3.Row | None", row)

    def _fetch_json(self, table: str, key: str) -> str | None:
        row = self._fetch(table, key)
        if row is None:
            return None
        return _value_to_json(_row_to_value(table, row))

    def _fetch_value(self, table: str, key: str) -> Any | None:
        row = self._fetch(table, key)
        if row is None:
            return None
        return _row_to_value(table, row)

    def _receipt_for_order(self, order_id: OrderId, *, refund: bool) -> Receipt | None:
        """Return the durable settle/refund receipt for ``order_id``.

        Economic direction is an authoritative receipt field.  Transaction-id
        naming never participates in classification.  Ordering by SQLite
        ``rowid`` returns the original append when legacy databases contain
        more than one matching entry.
        """
        rows = self._conn.execute(
            "SELECT * FROM ledger WHERE order_id = ? ORDER BY rowid",
            (str(order_id),),
        ).fetchall()
        for row in rows:
            receipt = cast(Receipt, _row_to_value("ledger", row))
            is_refund = receipt.effect == "refund"
            if is_refund == refund:
                return receipt
        return None

    def _idempotency_replay(
        self,
        *,
        scope: str,
        key: str,
        operation: str,
        fingerprint: str,
    ) -> Receipt | None:
        """Return an exact durable replay or reject a conflicting key reuse."""
        row = self._conn.execute(
            """
            SELECT operation, request_fingerprint, outcome_txn_id
            FROM idempotency_records
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, key),
        ).fetchone()
        if row is None:
            return None
        if row["operation"] != operation or row["request_fingerprint"] != fingerprint:
            raise IdempotencyConflict(
                f"idempotency key {key!r} was already used for a different request"
            )
        outcome = self._fetch_value("ledger", str(row["outcome_txn_id"]))
        if outcome is None:
            raise WorldError(
                f"idempotency record {scope!r}/{key!r} has no ledger outcome"
            )
        return cast(Receipt, outcome)

    def _record_idempotency(
        self,
        *,
        scope: str,
        key: str,
        operation: str,
        fingerprint: str,
        outcome: Receipt,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO idempotency_records
              (scope, idempotency_key, operation, request_fingerprint,
               outcome_txn_id, created_at)
            VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (scope, key, operation, fingerprint, str(outcome.txn_id)),
        )

    def _migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        ledger_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(ledger)")
        }
        if "effect" not in ledger_columns:
            self._conn.execute(
                "ALTER TABLE ledger ADD COLUMN effect TEXT NOT NULL DEFAULT 'charge'"
            )
        if "price_amount_text" not in ledger_columns:
            self._conn.execute(
                "ALTER TABLE ledger ADD COLUMN price_amount_text TEXT"
            )
            self._conn.execute(
                "UPDATE ledger SET price_amount_text = printf('%.2f', price_minor / 100.0)"
            )
        catalog_columns = {
            str(row["name"]) for row in self._conn.execute("PRAGMA table_info(catalog)")
        }
        if "product_id" not in catalog_columns:
            self._conn.execute("ALTER TABLE catalog ADD COLUMN product_id TEXT")
        authority_operation_columns = {
            str(row["name"])
            for row in self._conn.execute(
                "PRAGMA table_info(authority_operations)"
            )
        }
        if "outcome_listing_json" not in authority_operation_columns:
            self._conn.execute(
                "ALTER TABLE authority_operations "
                "ADD COLUMN outcome_listing_json TEXT"
            )
        inventory_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(inventory)")
        }
        if "eta_day" not in inventory_columns:
            self._conn.execute(
                "ALTER TABLE inventory ADD COLUMN eta_day INTEGER NOT NULL DEFAULT 0"
            )
        if "version" not in inventory_columns:
            self._conn.execute(
                "ALTER TABLE inventory ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
        if "supply_events_json" not in inventory_columns:
            self._conn.execute(
                "ALTER TABLE inventory ADD COLUMN supply_events_json "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        order_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(orders)")
        }
        if "request_order" not in order_columns:
            self._conn.execute(
                "ALTER TABLE orders ADD COLUMN request_order INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO order_state_revisions(order_id, revision)
            SELECT order_id, 1 FROM orders
            """
        )
        fulfillment_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(fulfillments)")
        }
        if "allocation_id" not in fulfillment_columns:
            self._conn.execute(
                "ALTER TABLE fulfillments ADD COLUMN allocation_id TEXT"
            )
        shipment_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(shipments)")
        }
        for name, declaration in (
            ("resolution_idempotency_key", "TEXT"),
            ("resolved_by", "TEXT"),
            ("resolution_version", "INTEGER"),
            ("resolution_history_length", "INTEGER"),
        ):
            if name not in shipment_columns:
                self._conn.execute(
                    f"ALTER TABLE shipments ADD COLUMN {name} {declaration}"
                )
        certificate_columns = {
            str(row["name"])
            for row in self._conn.execute(
                "PRAGMA table_info(match_certificates)"
            )
        }
        for name, declaration in (
            ("buyer_id", "TEXT"),
            ("order_id", "TEXT"),
            ("expires_at_tick", "INTEGER"),
        ):
            if name not in certificate_columns:
                self._conn.execute(
                    f"ALTER TABLE match_certificates ADD COLUMN {name} {declaration}"
                )
        for row in self._conn.execute(
            """
            SELECT cert_id, certificate_json FROM match_certificates
            WHERE buyer_id IS NULL OR order_id IS NULL OR expires_at_tick IS NULL
            """
        ).fetchall():
            certificate = coerce_match_certificate(
                json.loads(row["certificate_json"])
            )
            self._conn.execute(
                """
                UPDATE match_certificates
                SET buyer_id = ?, order_id = ?, expires_at_tick = ?
                WHERE cert_id = ?
                """,
                (
                    certificate.buyer_id,
                    certificate.order_id,
                    certificate.expires_at_tick,
                    certificate.cert_id,
                ),
            )
        indexed_sessions = {
            str(row["session_id"])
            for row in self._conn.execute(
                "SELECT DISTINCT session_id FROM search_session_offers"
            )
        }
        for row in self._conn.execute(
            "SELECT session_id, session_json FROM search_sessions"
        ).fetchall():
            if str(row["session_id"]) in indexed_sessions:
                continue
            session = coerce_search_session(json.loads(row["session_json"]))
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO search_session_offers
                  (session_id, buyer_id, offer_id, expires_at_tick)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        session.session_id,
                        session.buyer_id,
                        offer.offer_id,
                        session.expires_at_tick,
                    )
                    for offer in session.offers
                ),
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_catalog_product ON catalog (product_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_allocation ON orders "
            "(merchant_id, sku_id, state, request_order, buyer_id, order_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fulfillments_allocation ON fulfillments "
            "(allocation_id, order_id)"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_shipments_resolution_operation "
            "ON shipments (resolution_idempotency_key) "
            "WHERE resolution_idempotency_key IS NOT NULL"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_match_certificates_order "
            "ON match_certificates "
            "(buyer_id, order_id, expires_at_tick, cert_id)"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0001_initial', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0002_catalog_product_id', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0003_idempotency_records', strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0004_fulfillment_exchange', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_clock(singleton, logical_time) VALUES (1, 0)"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0005_authoritative_logical_clock', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0006_versioned_supply_state', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0007_atomic_order_allocation', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0008_authoritative_shipments', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0009_t6_durable_operations', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0010_persistent_matching', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0011_matching_lookup_indexes', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0012_reputation_settlement_idempotency', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0013_protocol_events_and_order_revisions', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0014_evidence_mandates_and_listing_claims', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0015_actor_scoped_catalog_mutations', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0016_world_backed_negotiations', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0017_world_backed_pricing_policies', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0018_persistent_cart_quotes', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0019_atomic_order_groups', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0020_cart_quote_request_authorizations', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0021_authoritative_ledger_effect', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO world_schema_migrations(version, applied_at) "
            "VALUES ('0022_supply_purchase_authorities', "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
        )
        self._conn.commit()


_TABLE_ORDER = (
    "catalog",
    "inventory",
    "orders",
    "ledger",
    "reputation",
    "reputation_settlements",
    "disputes",
    "rulings",
    "reviews",
    "fulfillments",
    "exchanges",
    "shipments",
    "order_timelines",
    "order_groups",
    "search_sessions",
    "match_acceptances",
    "match_certificates",
    "supply_purchase_authorities",
    "negotiation_events",
    "negotiation_threads",
    "pricing_policy_revisions",
    "persistent_cart_quote_requests",
    "persistent_cart_quotes",
    "protocol_events",
    "protocol_receipts",
    "evidence_records",
    "mandate_authorities",
    "mandate_revisions",
    "listing_claims",
    "authority_operations",
    "payment_states",
    "packing_records",
    "after_sales_policies",
    "after_sales_records",
    "governance_policies",
    "governance_records",
)

_KEY_COLUMNS = {
    "catalog": "sku_id",
    "inventory": "sku_id",
    "orders": "order_id",
    "ledger": "txn_id",
    "reputation": "merchant_id",
    "reputation_settlements": "event_id",
    "disputes": "dispute_id",
    "rulings": "ruling_id",
    "reviews": "review_id",
    "fulfillments": "order_id",
    "exchanges": "exchange_id",
    "shipments": "shipment_id",
    "order_timelines": "order_id",
    "order_groups": "order_group_id",
    "search_sessions": "session_id",
    "match_acceptances": "acceptance_key",
    "match_certificates": "cert_id",
    "supply_purchase_authorities": "authority_id",
    "negotiation_events": "event_id",
    "negotiation_threads": "negotiation_id",
    "pricing_policy_revisions": "revision_key",
    "persistent_cart_quote_requests": "request_id",
    "persistent_cart_quotes": "quote_id",
    "protocol_events": "event_id",
    "protocol_receipts": "receipt_id",
    "evidence_records": "evidence_key",
    "mandate_authorities": "mandate_id",
    "mandate_revisions": "revision_key",
    "listing_claims": "claim_id",
    "authority_operations": "operation_key",
    "payment_states": "record_key",
    "packing_records": "record_key",
    "after_sales_policies": "policy_key",
    "after_sales_records": "physical_key",
    "governance_policies": "envelope_key",
    "governance_records": "envelope_key",
}

_KEY_ATTRS = {
    **_KEY_COLUMNS,
    "match_acceptances": "idempotency_key",
}

_SEED_ACTIONS = {
    "catalog": "world.update_catalog",
    "inventory": "world.update_inventory",
    "orders": "world.create_order",
    "ledger": "world.update_ledger",
    "reputation": "world.update_reputation",
    "reputation_settlements": "world.update_reputation",
    "disputes": "world.update_dispute",
    "rulings": "world.update_ruling",
    "reviews": "world.add_review",
    "fulfillments": "world.record_fulfillment",
    "exchanges": "world.record_exchange",
    "shipments": "world.record_shipment",
    "order_timelines": "world.record_order_timeline",
    "order_groups": "world.checkout_cart",
    "search_sessions": "world.create_search_session",
    "match_acceptances": "world.issue_match_certificate",
    "match_certificates": "world.issue_match_certificate",
    "supply_purchase_authorities": "world.issue_supply_purchase_authority",
    "negotiation_events": "world.apply_negotiation_intent",
    "negotiation_threads": "world.apply_negotiation_intent",
    "pricing_policy_revisions": "world.publish_pricing_policy",
    "persistent_cart_quote_requests": "world.create_cart_quote_request",
    "persistent_cart_quotes": "world.issue_cart_quote",
    "protocol_events": "world.publish_protocol_event",
    "protocol_receipts": "world.append_protocol_receipt",
    "evidence_records": "world.persist_evidence_record",
    "mandate_authorities": "world.register_mandate_authority",
    "mandate_revisions": "world.append_mandate_revision",
    "listing_claims": "world.apply_listing_claim",
    "authority_operations": "world.persist_evidence_record",
    "payment_states": "world.apply_payment_intent",
    "packing_records": "world.apply_packing_intent",
    "after_sales_policies": "world.publish_after_sales_policy",
    "after_sales_records": "world.apply_after_sales_intent",
    "governance_policies": "world.publish_governance_policy",
    "governance_records": "world.apply_governance_intent",
}


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_table(table: str) -> None:
    if table not in _KEY_COLUMNS:
        raise TableNotFound(f"unknown world table: {table}")


def _row_key(table: str, row: Any) -> Any:
    if table == "payment_states":
        return payment_state_key(row)
    if table == "packing_records":
        return packing_record_key(row)
    if table == "after_sales_policies":
        return f"{row.merchant_id}:{row.revision}"
    if table == "after_sales_records":
        return physical_after_sales_record_key(row)
    if table in {"governance_policies", "governance_records"}:
        return governance_envelope_key(row)
    if table == "match_acceptances":
        return _match_acceptance_key(row.buyer_id, row.idempotency_key)
    if table == "evidence_records":
        return evidence_record_key(row.record_id, row.version)
    if table == "mandate_revisions":
        return mandate_revision_key(row.mandate_id, row.revision)
    if table == "pricing_policy_revisions":
        return pricing_policy_revision_key(
            row.market_id, row.merchant_id, row.policy_id, row.revision
        )
    try:
        return getattr(row, _KEY_ATTRS[table])
    except KeyError as exc:
        raise TableNotFound(f"unknown world table: {table}") from exc
    except AttributeError as exc:
        raise ValueError(f"{table} row missing key attribute {_KEY_ATTRS[table]!r}") from exc


def _record_mutation(
    conn: sqlite3.Connection,
    *,
    table: str,
    key: str,
    action: str,
    before: Any | None,
    after: Any,
) -> None:
    """Append one mutation and its row diff on the caller's transaction."""
    if table == "orders":
        if before is None:
            conn.execute(
                """
                INSERT INTO order_state_revisions(order_id, revision)
                VALUES (?, 1)
                ON CONFLICT(order_id) DO UPDATE SET revision=1
                """,
                (key,),
            )
        else:
            # A migrated legacy order without a row starts at revision one, so
            # its first committed update becomes revision two.
            conn.execute(
                """
                INSERT INTO order_state_revisions(order_id, revision)
                VALUES (?, 2)
                ON CONFLICT(order_id) DO UPDATE SET revision=revision + 1
                """,
                (key,),
            )
    mutation_id = f"mut-{uuid4().hex}"
    before_json = None if before is None else _value_to_json(before)
    after_json = _value_to_json(after)
    conn.execute(
        """
        INSERT INTO world_mutations
          (mutation_id, table_name, row_key, action_kind, before_json,
           after_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        """,
        (mutation_id, table, key, action, before_json, after_json),
    )
    conn.execute(
        """
        INSERT INTO transaction_diffs
          (diff_id, mutation_id, table_name, row_key, op, before_json,
           after_json, ordinal)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            f"diff-{uuid4().hex}",
            mutation_id,
            table,
            key,
            "insert" if before is None else "update",
            before_json,
            after_json,
        ),
    )


def _upsert(conn: sqlite3.Connection, table: str, value: Any) -> None:
    if table == "payment_states":
        conn.execute(
            """
            INSERT INTO payment_states
              (record_key, payment_id, order_id, owner_id, merchant_id,
               payment_state, version, logical_tick, record_digest, record_json,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                payment_state_key(value),
                value.payment_id,
                value.order_id,
                value.owner_id,
                value.merchant_id,
                value.state,
                value.version,
                value.logical_tick,
                value.record_digest,
                _canonical_json(payment_state_to_dict(value)),
            ),
        )
        return
    if table == "governance_policies":
        envelope = cast(GovernancePolicyEnvelope, value)
        conn.execute(
            """
            INSERT INTO governance_policies
              (envelope_key, record_kind, stable_id, revision, service_actor,
               original_actor, logical_tick, envelope_digest, envelope_json,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                governance_envelope_key(envelope),
                envelope.kind,
                envelope.stable_id,
                envelope.revision,
                envelope.service_actor,
                envelope.original_actor,
                envelope.logical_tick,
                envelope.envelope_digest,
                _canonical_json(policy_envelope_to_wire(envelope)),
            ),
        )
        return
    if table == "governance_records":
        envelope = cast(GovernanceRecordEnvelope, value)
        conn.execute(
            """
            INSERT INTO governance_records
              (envelope_key, record_kind, stable_id, version, service_actor,
               original_actor, logical_tick, envelope_digest, envelope_json,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                governance_envelope_key(envelope),
                envelope.kind,
                envelope.stable_id,
                envelope.version,
                envelope.service_actor,
                envelope.original_actor,
                envelope.logical_tick,
                envelope.envelope_digest,
                _canonical_json(record_envelope_to_wire(envelope)),
            ),
        )
        return
    if table == "packing_records":
        conn.execute(
            """
            INSERT INTO packing_records
              (record_key, packing_id, order_id, owner_id, merchant_id,
               packing_state, version, logical_tick, record_digest, record_json,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                packing_record_key(value),
                value.packing_id,
                value.order_id,
                value.owner_id,
                value.merchant_id,
                value.state,
                value.version,
                value.logical_tick,
                value.record_digest,
                _canonical_json(packing_record_to_dict(value)),
            ),
        )
        return
    if table == "after_sales_policies":
        conn.execute(
            """
            INSERT INTO after_sales_policies
              (policy_key, policy_id, merchant_id, revision, logical_tick,
               policy_digest, policy_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                f"{value.merchant_id}:{value.revision}",
                value.policy_id,
                value.merchant_id,
                value.revision,
                value.effective_tick,
                value.policy_digest,
                _canonical_json(after_sales_record_to_wire(value)),
            ),
        )
        return
    if table == "after_sales_records":
        write = cast(AfterSalesWrite, value)
        binding = (
            write.value
            if write.table == "after_sales_bindings"
            else getattr(write.value, "binding")
        )
        conn.execute(
            """
            INSERT INTO after_sales_records
              (physical_key, record_kind, domain_key, order_id, owner_id,
               merchant_id, logical_tick, record_digest, record_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                physical_after_sales_record_key(write),
                write.table,
                write.key,
                binding.order_id,
                binding.owner_id,
                binding.merchant_id,
                int(getattr(write.value, "logical_tick", 0)),
                after_sales_record_digest(write.value),
                _canonical_json(after_sales_write_to_wire(write)),
            ),
        )
        return
    if table == "catalog":
        conn.execute(
            """
            INSERT INTO catalog
              (sku_id, merchant_id, product_id, category, name, attributes_json,
               list_price_minor, list_price_currency, version, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(sku_id) DO UPDATE SET
              merchant_id=excluded.merchant_id,
              product_id=excluded.product_id,
              category=excluded.category,
              name=excluded.name,
              attributes_json=excluded.attributes_json,
              list_price_minor=excluded.list_price_minor,
              list_price_currency=excluded.list_price_currency,
              version=catalog.version + 1,
              updated_at=excluded.updated_at
            """,
            (
                str(value.sku_id),
                str(value.merchant_id),
                value.product_id,
                value.category,
                value.name,
                _canonical_json(value.attributes),
                _money_minor(value.list_price),
                value.list_price.currency,
                _catalog_revision(value),
            ),
        )
        return
    if table == "inventory":
        conn.execute(
            """
            INSERT INTO inventory
              (sku_id, merchant_id, qty_available, qty_reserved, eta_day, version,
               supply_events_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(sku_id) DO UPDATE SET
              merchant_id=excluded.merchant_id,
              qty_available=excluded.qty_available,
              qty_reserved=excluded.qty_reserved,
              eta_day=excluded.eta_day,
              version=excluded.version,
              supply_events_json=excluded.supply_events_json,
              updated_at=excluded.updated_at
            """,
            (
                str(value.sku_id),
                str(value.merchant_id),
                value.qty_available,
                value.qty_reserved,
                value.eta_day,
                value.version,
                _canonical_json(value.supply_events),
            ),
        )
        return
    if table == "orders":
        conn.execute(
            """
            INSERT INTO orders
              (order_id, buyer_id, merchant_id, sku_id, qty, agreed_price_minor,
               agreed_price_currency, state, request_order, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(order_id) DO UPDATE SET
              buyer_id=excluded.buyer_id,
              merchant_id=excluded.merchant_id,
              sku_id=excluded.sku_id,
              qty=excluded.qty,
              agreed_price_minor=excluded.agreed_price_minor,
              agreed_price_currency=excluded.agreed_price_currency,
              state=excluded.state,
              request_order=excluded.request_order,
              updated_at=excluded.updated_at
            """,
            (
                str(value.order_id),
                str(value.buyer_id),
                str(value.merchant_id),
                str(value.sku_id),
                value.qty,
                _money_minor(value.agreed_price),
                value.agreed_price.currency,
                value.state.value,
                value.request_order,
            ),
        )
        return
    if table == "order_timelines":
        conn.execute(
            """
            INSERT INTO order_timelines
              (order_id, buyer_id, merchant_id, settled_at_tick,
               dispatched_at_tick, return_window_ticks,
               return_authorized_at_tick, returned_at_tick, refunded_at_tick)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
              buyer_id=excluded.buyer_id,
              merchant_id=excluded.merchant_id,
              settled_at_tick=excluded.settled_at_tick,
              dispatched_at_tick=excluded.dispatched_at_tick,
              return_window_ticks=excluded.return_window_ticks,
              return_authorized_at_tick=excluded.return_authorized_at_tick,
              returned_at_tick=excluded.returned_at_tick,
              refunded_at_tick=excluded.refunded_at_tick
            """,
            (
                str(value.order_id),
                str(value.buyer_id),
                str(value.merchant_id),
                value.settled_at_tick,
                value.dispatched_at_tick,
                value.return_window_ticks,
                value.return_authorized_at_tick,
                value.returned_at_tick,
                value.refunded_at_tick,
            ),
        )
        return
    if table == "order_groups":
        conn.execute(
            """
            INSERT INTO order_groups
              (order_group_id, quote_id, buyer_id, idempotency_key,
               group_json, created_at)
            VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                str(value.order_group_id),
                str(value.quote_id),
                str(value.buyer_id),
                value.idempotency_key,
                _value_to_json(value),
            ),
        )
        return
    if table == "fulfillments":
        conn.execute(
            """
            INSERT INTO fulfillments
              (order_id, buyer_id, merchant_id, sku_id, requested_qty,
               fulfilled_qty, backordered_qty, receipt_txn_id, created_by,
               idempotency_key, allocation_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(order_id) DO UPDATE SET
              buyer_id=excluded.buyer_id,
              merchant_id=excluded.merchant_id,
              sku_id=excluded.sku_id,
              requested_qty=excluded.requested_qty,
              fulfilled_qty=excluded.fulfilled_qty,
              backordered_qty=excluded.backordered_qty,
              receipt_txn_id=excluded.receipt_txn_id,
              created_by=excluded.created_by,
              idempotency_key=excluded.idempotency_key,
              allocation_id=excluded.allocation_id
            """,
            (
                str(value.order_id),
                str(value.buyer_id),
                str(value.merchant_id),
                str(value.sku_id),
                value.requested_qty,
                value.fulfilled_qty,
                value.backordered_qty,
                (
                    None
                    if value.receipt_txn_id is None
                    else str(value.receipt_txn_id)
                ),
                str(value.created_by),
                value.idempotency_key,
                value.allocation_id,
            ),
        )
        return
    if table == "exchanges":
        conn.execute(
            """
            INSERT INTO exchanges
              (exchange_id, original_order_id, replacement_order_id, buyer_id,
               merchant_id, original_sku_id, replacement_sku_id, qty,
               created_by, idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(exchange_id) DO UPDATE SET
              original_order_id=excluded.original_order_id,
              replacement_order_id=excluded.replacement_order_id,
              buyer_id=excluded.buyer_id,
              merchant_id=excluded.merchant_id,
              original_sku_id=excluded.original_sku_id,
              replacement_sku_id=excluded.replacement_sku_id,
              qty=excluded.qty,
              created_by=excluded.created_by,
              idempotency_key=excluded.idempotency_key
            """,
            (
                str(value.exchange_id),
                str(value.original_order_id),
                str(value.replacement_order_id),
                str(value.buyer_id),
                str(value.merchant_id),
                str(value.original_sku_id),
                str(value.replacement_sku_id),
                value.qty,
                str(value.created_by),
                value.idempotency_key,
            ),
        )
        return
    if table == "shipments":
        conn.execute(
            """
            INSERT INTO shipments
              (shipment_id, order_id, buyer_id, merchant_id, original_sku_id,
               status, status_history_json, resolution, replacement_sku_id,
               version, resolution_idempotency_key, resolved_by,
               resolution_version, resolution_history_length, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(shipment_id) DO UPDATE SET
              order_id=excluded.order_id,
              buyer_id=excluded.buyer_id,
              merchant_id=excluded.merchant_id,
              original_sku_id=excluded.original_sku_id,
              status=excluded.status,
              status_history_json=excluded.status_history_json,
              resolution=excluded.resolution,
              replacement_sku_id=excluded.replacement_sku_id,
              version=excluded.version,
              resolution_idempotency_key=excluded.resolution_idempotency_key,
              resolved_by=excluded.resolved_by,
              resolution_version=excluded.resolution_version,
              resolution_history_length=excluded.resolution_history_length,
              updated_at=excluded.updated_at
            """,
            (
                str(value.shipment_id),
                str(value.order_id),
                str(value.buyer_id),
                str(value.merchant_id),
                str(value.original_sku_id),
                value.status.value,
                _canonical_json(value.status_history),
                None if value.resolution is None else value.resolution.value,
                (
                    None
                    if value.replacement_sku_id is None
                    else str(value.replacement_sku_id)
                ),
                value.version,
                value.resolution_idempotency_key,
                None if value.resolved_by is None else str(value.resolved_by),
                value.resolution_version,
                value.resolution_history_length,
            ),
        )
        return
    if table == "ledger":
        if value.effect not in {"charge", "refund"}:
            raise WorldError("ledger receipt effect must be charge or refund")
        conn.execute(
            """
            INSERT INTO ledger
              (txn_id, ts, order_id, buyer_id, merchant_id, sku_id, qty,
               price_minor, price_amount_text, price_currency, idempotency_key,
               effect)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(value.txn_id),
                value.ts,
                str(value.order_id),
                str(value.buyer_id),
                str(value.merchant_id),
                str(value.sku_id),
                value.qty,
                _money_minor(value.price),
                str(value.price.amount),
                value.price.currency,
                value.idempotency_key,
                value.effect,
            ),
        )
        return
    if table == "reputation":
        conn.execute(
            """
            INSERT INTO reputation
              (merchant_id, rolling_avg, n_settled, n_disputed, updated_at)
            VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(merchant_id) DO UPDATE SET
              rolling_avg=excluded.rolling_avg,
              n_settled=excluded.n_settled,
              n_disputed=excluded.n_disputed,
              updated_at=excluded.updated_at
            """,
            (str(value.merchant_id), value.rolling_avg, value.n_settled, value.n_disputed),
        )
        return
    if table == "reviews":
        conn.execute(
            """
            INSERT INTO reviews
              (review_id, reviewer_id, sku_id, merchant_id, rating, review_text,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(review_id) DO UPDATE SET
              reviewer_id=excluded.reviewer_id,
              sku_id=excluded.sku_id,
              merchant_id=excluded.merchant_id,
              rating=excluded.rating,
              review_text=excluded.review_text
            """,
            (
                str(value.review_id),
                str(value.reviewer_id),
                str(value.sku_id),
                str(value.merchant_id),
                value.rating,
                value.text,
            ),
        )
        return
    if table == "reputation_settlements":
        conn.execute(
            """
            INSERT INTO reputation_settlements
              (event_id, order_id, txn_id, merchant_id, event_json, created_at,
               updated_at)
            VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(event_id) DO UPDATE SET
              order_id=excluded.order_id,
              txn_id=excluded.txn_id,
              merchant_id=excluded.merchant_id,
              event_json=excluded.event_json,
              updated_at=excluded.updated_at
            """,
            (
                value.event_id,
                str(value.order_id),
                str(value.txn_id),
                str(value.merchant_id),
                _value_to_json(value),
            ),
        )
        conn.execute(
            "DELETE FROM reputation_settlement_sources WHERE event_id = ?",
            (value.event_id,),
        )
        for source in value.sources:
            conn.execute(
                """
                INSERT INTO reputation_settlement_sources
                  (source_actor, source_idempotency_key, source_request_id,
                   event_id)
                VALUES (?, ?, ?, ?)
                """,
                (
                    source.source_actor,
                    source.source_idempotency_key,
                    source.source_request_id,
                    value.event_id,
                ),
            )
        return
    if table == "disputes":
        conn.execute(
            """
            INSERT INTO disputes
              (dispute_id, order_id, filed_by, against, reason, state, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(dispute_id) DO UPDATE SET
              order_id=excluded.order_id,
              filed_by=excluded.filed_by,
              against=excluded.against,
              reason=excluded.reason,
              state=excluded.state,
              updated_at=excluded.updated_at
            """,
            (
                str(value.dispute_id),
                str(value.order_id),
                str(value.filed_by),
                str(value.against),
                value.reason,
                value.state.value,
            ),
        )
        return
    if table == "rulings":
        refund_minor = _money_minor(value.refund_amount) if value.refund_amount is not None else None
        refund_currency = value.refund_amount.currency if value.refund_amount is not None else None
        conn.execute(
            """
            INSERT INTO rulings
              (ruling_id, dispute_id, in_favor_of, rationale, refund_amount_minor,
               refund_amount_currency, created_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(ruling_id) DO UPDATE SET
              dispute_id=excluded.dispute_id,
              in_favor_of=excluded.in_favor_of,
              rationale=excluded.rationale,
              refund_amount_minor=excluded.refund_amount_minor,
              refund_amount_currency=excluded.refund_amount_currency
            """,
            (
                str(value.ruling_id),
                str(value.dispute_id),
                str(value.in_favor_of),
                value.rationale,
                refund_minor,
                refund_currency,
            ),
        )
        return
    if table == "search_sessions":
        validate_search_session(value)
        conn.execute(
            """
            INSERT INTO search_sessions
              (session_id, buyer_id, search_idempotency_key, session_json,
               created_at)
            VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.session_id,
                value.buyer_id,
                value.search_idempotency_key,
                _canonical_json(search_session_to_wire(value)),
            ),
        )
        conn.execute(
            "DELETE FROM search_session_offers WHERE session_id = ?",
            (value.session_id,),
        )
        conn.executemany(
            """
            INSERT INTO search_session_offers
              (session_id, buyer_id, offer_id, expires_at_tick)
            VALUES (?, ?, ?, ?)
            """,
            (
                (
                    value.session_id,
                    value.buyer_id,
                    offer.offer_id,
                    value.expires_at_tick,
                )
                for offer in value.offers
            ),
        )
        return
    if table == "match_acceptances":
        conn.execute(
            """
            INSERT INTO match_acceptances
              (acceptance_key, buyer_id, idempotency_key, acceptance_digest,
               acceptance_json, created_at)
            VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                _match_acceptance_key(value.buyer_id, value.idempotency_key),
                value.buyer_id,
                value.idempotency_key,
                value.acceptance_digest,
                _canonical_json(match_acceptance_to_wire(value)),
            ),
        )
        return
    if table == "match_certificates":
        conn.execute(
            """
            INSERT INTO match_certificates
              (cert_id, acceptance_digest, buyer_id, order_id,
               expires_at_tick, certificate_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.cert_id,
                value.acceptance_digest,
                value.buyer_id,
                value.order_id,
                value.expires_at_tick,
                _canonical_json(match_certificate_to_wire(value)),
            ),
        )
        return
    if table == "supply_purchase_authorities":
        validate_supply_purchase_authority(value)
        conn.execute(
            """
            INSERT INTO supply_purchase_authorities
              (authority_id, authority_digest, buyer_id, merchant_id, sku_id,
               order_id, supply_version, expires_at_tick, authority_json,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.authority_id,
                value.authority_digest,
                value.buyer_id,
                value.merchant_id,
                value.sku_id,
                value.order_id,
                value.supply_version,
                value.expires_at_tick,
                supply_purchase_authority_to_json(value),
            ),
        )
        return
    if table == "negotiation_events":
        conn.execute(
            """
            INSERT INTO negotiation_events
              (event_id, negotiation_id, buyer_id, merchant_id, actor_id,
               idempotency_key, sequence_no, event_digest, event_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.event_id,
                value.negotiation_id,
                value.buyer_id,
                value.merchant_id,
                value.actor_id,
                value.idempotency_key,
                value.sequence_no,
                value.event_digest,
                negotiation_event_to_json(value),
            ),
        )
        return
    if table == "negotiation_threads":
        conn.execute(
            """
            INSERT INTO negotiation_threads
              (negotiation_id, buyer_id, merchant_id, status, event_count,
               head_event_digest, thread_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT(negotiation_id) DO UPDATE SET
              buyer_id=excluded.buyer_id,
              merchant_id=excluded.merchant_id,
              status=excluded.status,
              event_count=excluded.event_count,
              head_event_digest=excluded.head_event_digest,
              thread_json=excluded.thread_json,
              updated_at=excluded.updated_at
            """,
            (
                value.negotiation_id,
                value.buyer_id,
                value.merchant_id,
                value.status,
                value.event_count,
                value.head_event_digest,
                negotiation_thread_to_json(value),
            ),
        )
        return
    if table == "pricing_policy_revisions":
        conn.execute(
            """
            INSERT INTO pricing_policy_revisions
              (revision_key, market_id, merchant_id, owner_id, policy_id,
               revision, actor_id, idempotency_key, policy_digest,
               policy_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                pricing_policy_revision_key(
                    value.market_id,
                    value.merchant_id,
                    value.policy_id,
                    value.revision,
                ),
                value.market_id,
                value.merchant_id,
                value.owner_id,
                value.policy_id,
                value.revision,
                value.actor_id,
                value.idempotency_key,
                value.policy_digest,
                pricing_policy_revision_to_json(value),
            ),
        )
        return
    if table == "persistent_cart_quote_requests":
        validate_persistent_cart_quote_request(value)
        conn.execute(
            """
            INSERT INTO persistent_cart_quote_requests
              (request_id, market_id, buyer_id, principal_id, created_by,
               mandate_id, mandate_revision, idempotency_key, request_digest,
               issued_at_tick, expires_at_tick, request_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.request_id,
                value.market_id,
                value.buyer_id,
                value.principal_id,
                value.created_by,
                value.mandate_id,
                value.mandate_revision,
                value.idempotency_key,
                value.request_digest,
                value.issued_at_tick,
                value.expires_at_tick,
                _canonical_json(persistent_cart_quote_request_to_dict(value)),
            ),
        )
        return
    if table == "persistent_cart_quotes":
        conn.execute(
            """
            INSERT INTO persistent_cart_quotes
              (quote_id, request_id, market_id, buyer_id, principal_id,
               requested_by, issuer_id, mandate_id, mandate_revision,
               idempotency_key, quote_digest, issued_at_tick, expires_at_tick,
               quote_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.quote_id,
                value.request_id,
                value.market_id,
                value.buyer_id,
                value.principal_id,
                value.requested_by,
                value.issuer_id,
                value.mandate_id,
                value.mandate_revision,
                value.idempotency_key,
                value.quote_digest,
                value.issued_at_tick,
                value.expires_at_tick,
                persistent_cart_quote_to_json(value),
            ),
        )
        return
    if table == "protocol_events":
        validate_protocol_event(value)
        conn.execute(
            """
            INSERT INTO protocol_events
              (event_id, market_id, stream_id, order_id, buyer_id, merchant_id,
               recipient_id, authority_id, binding_digest, event_sequence,
               event_digest, actor_id, idempotency_key, issued_at_tick,
               expires_at_tick, event_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.event_id,
                value.binding.market_id,
                value.binding.stream_id,
                value.binding.order_id,
                value.binding.buyer_id,
                value.binding.merchant_id,
                value.binding.recipient_id,
                value.binding.authority_id,
                value.binding.binding_digest,
                value.sequence,
                value.event_digest,
                value.actor_id,
                value.idempotency_key,
                value.issued_at_tick,
                value.expires_at_tick,
                protocol_event_to_json(value),
            ),
        )
        return
    if table == "protocol_receipts":
        validate_protocol_event_receipt(value)
        conn.execute(
            """
            INSERT INTO protocol_receipts
              (receipt_id, market_id, stream_id, order_id, buyer_id,
               merchant_id, recipient_id, authority_id, binding_digest,
               event_id, event_digest, receipt_digest, actor_id,
               idempotency_key, decision, logical_tick, receipt_json,
               created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            """,
            (
                value.receipt_id,
                value.binding.market_id,
                value.binding.stream_id,
                value.binding.order_id,
                value.binding.buyer_id,
                value.binding.merchant_id,
                value.binding.recipient_id,
                value.binding.authority_id,
                value.binding.binding_digest,
                value.event_id,
                value.event_digest,
                value.receipt_digest,
                value.actor_id,
                value.idempotency_key,
                value.decision,
                value.logical_tick,
                protocol_event_receipt_to_json(value),
            ),
        )
        return
    if table == "evidence_records":
        conn.execute(
            "INSERT INTO evidence_records "
            "(evidence_key, record_id, version, record_digest, subject_id, "
            " issuer_id, owner_id, record_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (
                evidence_record_key(value.record_id, value.version),
                value.record_id,
                value.version,
                value.record_digest,
                value.subject_id,
                value.issuer_id,
                value.owner_id,
                evidence_record_to_json(value),
            ),
        )
        return
    if table == "mandate_authorities":
        conn.execute(
            "INSERT INTO mandate_authorities "
            "(mandate_id, principal_id, buyer_id, authority_json, created_at) "
            "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (
                value.mandate_id,
                value.principal_id,
                value.buyer_id,
                _canonical_json(mandate_authority_to_wire(value)),
            ),
        )
        return
    if table == "mandate_revisions":
        conn.execute(
            "INSERT INTO mandate_revisions "
            "(revision_key, mandate_id, revision, revision_digest, principal_id, "
            " buyer_id, revision_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (
                mandate_revision_key(value.mandate_id, value.revision),
                value.mandate_id,
                value.revision,
                value.revision_digest,
                value.principal_id,
                value.buyer_id,
                mandate_revision_to_json(value),
            ),
        )
        return
    if table == "listing_claims":
        conn.execute(
            "INSERT INTO listing_claims "
            "(claim_id, listing_id, merchant_id, claim_state, current_version, "
            " current_event_digest, claim_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
            "ON CONFLICT(claim_id) DO UPDATE SET "
            "listing_id=excluded.listing_id, merchant_id=excluded.merchant_id, "
            "claim_state=excluded.claim_state, "
            "current_version=excluded.current_version, "
            "current_event_digest=excluded.current_event_digest, "
            "claim_json=excluded.claim_json, updated_at=excluded.updated_at",
            (
                value.claim_id,
                value.listing_id,
                value.merchant_id,
                value.state,
                value.version,
                value.current.event_digest,
                listing_claim_to_json(value),
            ),
        )
        conn.execute("DELETE FROM listing_claim_keys WHERE claim_id = ?", (value.claim_id,))
        conn.executemany(
            "INSERT INTO listing_claim_keys "
            "(merchant_id, idempotency_key, claim_id, event_digest) "
            "VALUES (?, ?, ?, ?)",
            (
                (
                    value.merchant_id,
                    version.idempotency_key,
                    value.claim_id,
                    version.event_digest,
                )
                for version in value.versions
            ),
        )
        return
    if table == "authority_operations":
        conn.execute(
            "INSERT INTO authority_operations "
            "(operation_key, scope, actor_id, idempotency_key, "
            " request_fingerprint, outcome_table, outcome_key, "
            " outcome_listing_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (
                value.operation_key,
                value.scope,
                value.actor_id,
                value.idempotency_key,
                value.request_fingerprint,
                value.outcome_table,
                value.outcome_key,
                (
                    None
                    if value.outcome_listing is None
                    else _value_to_json(value.outcome_listing)
                ),
            ),
        )
        return
    raise TableNotFound(f"unknown world table: {table}")


def _row_to_value(table: str, row: sqlite3.Row) -> Any:
    if table == "payment_states":
        return payment_state_from_dict(json.loads(str(row["record_json"])))
    if table == "packing_records":
        return packing_record_from_dict(json.loads(str(row["record_json"])))
    if table == "after_sales_policies":
        return after_sales_record_from_wire(
            "after_sales_policies", json.loads(str(row["policy_json"]))
        )
    if table == "after_sales_records":
        payload = json.loads(str(row["record_json"]))
        if not isinstance(payload, dict) or set(payload) != {"table", "key", "value"}:
            raise WorldError("after-sales record wrapper fields are not exact")
        record_table = cast(Any, payload["table"])
        value = after_sales_record_from_wire(record_table, payload["value"])
        write = AfterSalesWrite(record_table, str(payload["key"]), value)
        if physical_after_sales_record_key(write) != str(row["physical_key"]):
            raise WorldError("after-sales physical key mismatch")
        return write
    if table == "governance_policies":
        envelope = policy_envelope_from_wire(
            json.loads(str(row["envelope_json"]))
        )
        if governance_envelope_key(envelope) != str(row["envelope_key"]):
            raise WorldError("governance policy physical key mismatch")
        return envelope
    if table == "governance_records":
        envelope = record_envelope_from_wire(
            json.loads(str(row["envelope_json"]))
        )
        if governance_envelope_key(envelope) != str(row["envelope_key"]):
            raise WorldError("governance record physical key mismatch")
        return envelope
    if table == "catalog":
        return Listing(
            sku_id=SkuId(row["sku_id"]),
            category=row["category"],
            name=row["name"],
            attributes=json.loads(row["attributes_json"]),
            list_price=_money_from_minor(row["list_price_minor"], row["list_price_currency"]),
            merchant_id=AgentId(row["merchant_id"]),
            product_id=row["product_id"],
        )
    if table == "inventory":
        raw_supply_events = json.loads(str(row["supply_events_json"]))
        return InventoryRow(
            sku_id=SkuId(row["sku_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            qty_available=row["qty_available"],
            qty_reserved=row["qty_reserved"],
            eta_day=row["eta_day"],
            version=row["version"],
            supply_events=tuple(
                SupplyEventRecord(
                    scope=str(value["scope"]),
                    idempotency_key=str(value["idempotency_key"]),
                    sku_id=SkuId(str(value["sku_id"])),
                    qty_delta=int(value["qty_delta"]),
                    eta_day=(
                        None if value.get("eta_day") is None else int(value["eta_day"])
                    ),
                    unit_price_cents=(
                        None
                        if value.get("unit_price_cents") is None
                        else int(value["unit_price_cents"])
                    ),
                    expected_version=(
                        None
                        if value.get("expected_version") is None
                        else int(value["expected_version"])
                    ),
                    original_actor=str(value["original_actor"]),
                    outcome=_supply_state_from_mapping(value["outcome"]),
                )
                for value in raw_supply_events
            ),
        )
    if table == "orders":
        return Order(
            order_id=OrderId(row["order_id"]),
            buyer_id=AgentId(row["buyer_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            sku_id=SkuId(row["sku_id"]),
            qty=row["qty"],
            agreed_price=_money_from_minor(row["agreed_price_minor"], row["agreed_price_currency"]),
            state=OrderState(row["state"]),
            request_order=row["request_order"],
        )
    if table == "order_timelines":
        return OrderTimeline(
            order_id=OrderId(row["order_id"]),
            buyer_id=AgentId(row["buyer_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            settled_at_tick=row["settled_at_tick"],
            dispatched_at_tick=row["dispatched_at_tick"],
            return_window_ticks=row["return_window_ticks"],
            return_authorized_at_tick=row["return_authorized_at_tick"],
            returned_at_tick=row["returned_at_tick"],
            refunded_at_tick=row["refunded_at_tick"],
        )
    if table == "order_groups":
        return _order_group_from_json(str(row["group_json"]))
    if table == "fulfillments":
        receipt_txn_id = row["receipt_txn_id"]
        return FulfillmentAllocation(
            order_id=OrderId(row["order_id"]),
            buyer_id=AgentId(row["buyer_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            sku_id=SkuId(row["sku_id"]),
            requested_qty=row["requested_qty"],
            fulfilled_qty=row["fulfilled_qty"],
            backordered_qty=row["backordered_qty"],
            receipt_txn_id=(
                None if receipt_txn_id is None else TxnId(receipt_txn_id)
            ),
            created_by=AgentId(row["created_by"]),
            idempotency_key=row["idempotency_key"],
            allocation_id=row["allocation_id"],
        )
    if table == "exchanges":
        return Exchange(
            exchange_id=ExchangeId(row["exchange_id"]),
            original_order_id=OrderId(row["original_order_id"]),
            replacement_order_id=OrderId(row["replacement_order_id"]),
            buyer_id=AgentId(row["buyer_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            original_sku_id=SkuId(row["original_sku_id"]),
            replacement_sku_id=SkuId(row["replacement_sku_id"]),
            qty=row["qty"],
            created_by=AgentId(row["created_by"]),
            idempotency_key=row["idempotency_key"],
        )
    if table == "shipments":
        history = json.loads(str(row["status_history_json"]))
        resolution = row["resolution"]
        replacement_sku_id = row["replacement_sku_id"]
        return Shipment(
            shipment_id=ShipmentId(row["shipment_id"]),
            order_id=OrderId(row["order_id"]),
            buyer_id=AgentId(row["buyer_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            original_sku_id=SkuId(row["original_sku_id"]),
            status=ShipmentStatus(row["status"]),
            status_history=tuple(
                ShipmentStatusEvent(
                    event_id=str(value["event_id"]),
                    status=ShipmentStatus(str(value["status"])),
                    logical_time=int(value["logical_time"]),
                    idempotency_key=(
                        None
                        if value.get("idempotency_key") is None
                        else str(value["idempotency_key"])
                    ),
                    shipment_version=(
                        None
                        if value.get("shipment_version") is None
                        else int(value["shipment_version"])
                    ),
                )
                for value in history
            ),
            resolution=(
                None if resolution is None else ShipmentResolution(resolution)
            ),
            replacement_sku_id=(
                None
                if replacement_sku_id is None
                else SkuId(replacement_sku_id)
            ),
            version=int(row["version"]),
            resolution_idempotency_key=row["resolution_idempotency_key"],
            resolved_by=(
                None if row["resolved_by"] is None else AgentId(row["resolved_by"])
            ),
            resolution_version=row["resolution_version"],
            resolution_history_length=row["resolution_history_length"],
        )
    if table == "ledger":
        return Receipt(
            txn_id=TxnId(row["txn_id"]),
            ts=row["ts"],
            order_id=OrderId(row["order_id"]),
            buyer_id=AgentId(row["buyer_id"]),
            merchant_id=AgentId(row["merchant_id"]),
            sku_id=SkuId(row["sku_id"]),
            qty=row["qty"],
            price=Money(
                Decimal(
                    str(
                        row["price_amount_text"]
                        if row["price_amount_text"] is not None
                        else Decimal(row["price_minor"]) / Decimal(100)
                    )
                ),
                row["price_currency"],
            ),
            idempotency_key=row["idempotency_key"],
            effect=str(row["effect"]),
        )
    if table == "reputation":
        return ReputationScore(
            merchant_id=AgentId(row["merchant_id"]),
            rolling_avg=row["rolling_avg"],
            n_settled=row["n_settled"],
            n_disputed=row["n_disputed"],
        )
    if table == "reviews":
        return Review(
            review_id=ReviewId(str(row["review_id"])),
            reviewer_id=AgentId(str(row["reviewer_id"])),
            sku_id=SkuId(str(row["sku_id"])),
            merchant_id=AgentId(str(row["merchant_id"])),
            rating=int(row["rating"]),
            text=str(row["review_text"]),
        )
    if table == "reputation_settlements":
        return _reputation_settlement_from_json(str(row["event_json"]))
    if table == "disputes":
        return Dispute(
            dispute_id=DisputeId(row["dispute_id"]),
            order_id=OrderId(row["order_id"]),
            filed_by=AgentId(row["filed_by"]),
            against=AgentId(row["against"]),
            reason=row["reason"],
            state=DisputeState(row["state"]),
        )
    if table == "rulings":
        refund = None
        if row["refund_amount_minor"] is not None:
            refund = _money_from_minor(row["refund_amount_minor"], row["refund_amount_currency"])
        return Ruling(
            ruling_id=RulingId(row["ruling_id"]),
            dispute_id=DisputeId(row["dispute_id"]),
            in_favor_of=AgentId(row["in_favor_of"]),
            rationale=row["rationale"],
            refund_amount=refund,
        )
    if table == "search_sessions":
        return coerce_search_session(json.loads(str(row["session_json"])))
    if table == "match_acceptances":
        return coerce_match_acceptance(json.loads(str(row["acceptance_json"])))
    if table == "match_certificates":
        return coerce_match_certificate(json.loads(str(row["certificate_json"])))
    if table == "supply_purchase_authorities":
        return supply_purchase_authority_from_json(str(row["authority_json"]))
    if table == "negotiation_events":
        return negotiation_event_from_json(str(row["event_json"]))
    if table == "negotiation_threads":
        return negotiation_thread_from_json(str(row["thread_json"]))
    if table == "pricing_policy_revisions":
        return pricing_policy_revision_from_json(str(row["policy_json"]))
    if table == "persistent_cart_quote_requests":
        return coerce_persistent_cart_quote_request(
            json.loads(str(row["request_json"]))
        )
    if table == "persistent_cart_quotes":
        return persistent_cart_quote_from_json(str(row["quote_json"]))
    if table == "protocol_events":
        return protocol_event_from_json(str(row["event_json"]))
    if table == "protocol_receipts":
        return protocol_event_receipt_from_json(str(row["receipt_json"]))
    if table == "evidence_records":
        return coerce_evidence_record(json.loads(str(row["record_json"])))
    if table == "mandate_authorities":
        return coerce_mandate_authority(json.loads(str(row["authority_json"])))
    if table == "mandate_revisions":
        return coerce_mandate_revision(json.loads(str(row["revision_json"])))
    if table == "listing_claims":
        return coerce_listing_claim(json.loads(str(row["claim_json"])))
    if table == "authority_operations":
        raw_listing = row["outcome_listing_json"]
        return AuthorityOperationRecord(
            operation_key=str(row["operation_key"]),
            scope=str(row["scope"]),
            actor_id=str(row["actor_id"]),
            idempotency_key=str(row["idempotency_key"]),
            request_fingerprint=str(row["request_fingerprint"]),
            outcome_table=str(row["outcome_table"]),
            outcome_key=str(row["outcome_key"]),
            outcome_listing=(
                None
                if raw_listing is None
                else _listing_from_mapping(json.loads(str(raw_listing)))
            ),
        )
    raise TableNotFound(f"unknown world table: {table}")


def _visible(table: str, value: Any, caller: str | None) -> bool:
    if table in {"governance_policies", "governance_records"}:
        if caller is None or caller == "world" or caller == value.service_actor:
            return True
        if caller == "runtime" or bool(caller and caller.startswith("runtime:")):
            return True
        return caller in {
            *value.owner_ids,
            *value.subject_ids,
            value.original_actor,
        }
    if (
        caller is None
        or caller == "platform"
        or caller.startswith("platform:")
    ):
        return True
    if table in {
        "search_sessions",
        "match_acceptances",
        "match_certificates",
        "reputation_settlements",
    }:
        return caller == "runtime" or bool(caller and caller.startswith("runtime:"))
    if table in {"payment_states", "packing_records"}:
        return caller in {value.owner_id, value.merchant_id} or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table == "after_sales_policies":
        return caller == value.merchant_id or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table == "after_sales_records":
        binding = (
            value.value
            if value.table == "after_sales_bindings"
            else getattr(value.value, "binding")
        )
        return caller in {binding.owner_id, binding.merchant_id} or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table in {"protocol_events", "protocol_receipts"}:
        return (
            caller == str(value.binding.recipient_id)
            or caller == "runtime"
            or bool(caller and caller.startswith("runtime:"))
        )
    if table in {"negotiation_events", "negotiation_threads"}:
        return caller in {value.buyer_id, value.merchant_id} or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table == "pricing_policy_revisions":
        return caller in {value.owner_id, value.merchant_id} or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table == "persistent_cart_quote_requests":
        return caller in {
            value.buyer_id,
            value.principal_id,
            value.created_by,
        } or caller == "runtime" or bool(caller and caller.startswith("runtime:"))
    if table == "persistent_cart_quotes":
        return caller in {
            value.buyer_id,
            value.principal_id,
        } or caller == "runtime" or bool(caller and caller.startswith("runtime:"))
    if table == "supply_purchase_authorities":
        return caller == value.buyer_id or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table == "evidence_records":
        authorize_persisted_evidence_read(value, reader_id=str(caller))
        return True
    if table in {"mandate_authorities", "mandate_revisions"}:
        return caller in {value.principal_id, value.buyer_id} or caller == "runtime" or bool(
            caller and caller.startswith("runtime:")
        )
    if table == "listing_claims":
        return value.state != "draft" or caller == value.merchant_id
    if table == "authority_operations":
        return caller == "runtime" or bool(caller and caller.startswith("runtime:"))
    if table in {"catalog", "reputation", "inventory", "reviews"}:
        return True
    if table in {
        "orders",
        "ledger",
        "fulfillments",
        "exchanges",
        "shipments",
        "order_timelines",
    }:
        return caller in (str(value.buyer_id), str(value.merchant_id))
    if table == "order_groups":
        return caller == str(value.buyer_id) or caller in {
            str(merchant_id) for merchant_id in value.merchant_ids
        }
    if table == "disputes":
        return caller in (str(value.filed_by), str(value.against))
    if table == "rulings":
        return True
    return False


def _matches_filters(row: Listing, filters: dict[str, object]) -> bool:
    for key, expected in filters.items():
        actual = getattr(row, key, row.attributes.get(key))
        if isinstance(expected, (set, tuple, list, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _money_minor(value: Money) -> int:
    return int((value.amount * Decimal("100")).to_integral_exact())


def _money_from_minor(value: int, currency: str) -> Money:
    return Money((Decimal(value) / Decimal("100")).quantize(Decimal("0.01")), currency)


def _listing_from_mapping(value: Any) -> Listing:
    if not isinstance(value, dict):
        raise WorldError("catalog mutation outcome listing is not an object")
    price = value.get("list_price")
    if not isinstance(price, dict):
        raise WorldError("catalog mutation outcome price is not an object")
    return Listing(
        sku_id=SkuId(str(value["sku_id"])),
        category=str(value["category"]),
        name=str(value["name"]),
        attributes=dict(value.get("attributes", {})),
        list_price=Money(
            Decimal(str(price["amount"])), str(price.get("currency", "USD"))
        ),
        merchant_id=AgentId(str(value["merchant_id"])),
        product_id=(
            None if value.get("product_id") is None else str(value["product_id"])
        ),
    )


def _order_group_from_json(payload: str) -> OrderGroup:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise WorldError("order group row is not an object")

    def money(field: str) -> Money:
        raw = value.get(field)
        if not isinstance(raw, dict):
            raise WorldError(f"order group {field} is not an object")
        return Money(
            Decimal(str(raw.get("amount"))),
            str(raw.get("currency", "USD")),
        )

    raw_fees = value.get("fee_breakdown")
    if not isinstance(raw_fees, list):
        raise WorldError("order group fee_breakdown is not an array")
    fees: list[FeeComponent] = []
    for raw in raw_fees:
        if not isinstance(raw, dict) or not isinstance(raw.get("amount"), dict):
            raise WorldError("order group fee row is invalid")
        amount = cast(dict[str, Any], raw["amount"])
        fees.append(
            FeeComponent(
                fee_id=str(raw.get("fee_id", "")),
                kind=str(raw.get("kind", "")),
                scope=str(raw.get("scope", "")),
                amount=Money(
                    Decimal(str(amount.get("amount"))),
                    str(amount.get("currency", "USD")),
                ),
                basis=str(raw.get("basis", "")),
            )
        )
    return OrderGroup(
        order_group_id=OrderGroupId(str(value["order_group_id"])),
        quote_id=QuoteId(str(value["quote_id"])),
        buyer_id=AgentId(str(value["buyer_id"])),
        merchant_ids=tuple(AgentId(str(item)) for item in value["merchant_ids"]),
        order_ids=tuple(OrderId(str(item)) for item in value["order_ids"]),
        txn_ids=tuple(TxnId(str(item)) for item in value["txn_ids"]),
        subtotal=money("subtotal"),
        fee_breakdown=tuple(fees),
        grand_total=money("grand_total"),
        quote_hash=str(value["quote_hash"]),
        idempotency_key=str(value["idempotency_key"]),
        state=str(value.get("state", "settled")),
    )


def _idempotency_fingerprint(operation: str, order: Order, receipt: Receipt) -> str:
    """Canonical request identity stored with a scoped idempotency key."""
    return _canonical_json({
        "operation": operation,
        "order": order,
        "receipt": receipt,
    })


def _world_commit_to_json(record: WorldCommitRecord) -> str:
    """Encode one durable commit without losing typed World row identity."""

    payload = {
        "sequence": record.sequence,
        "commit_id": record.commit_id,
        "commit_kind": record.commit_kind,
        "operation": record.operation,
        "authority_action": record.authority_action,
        "actor_id": record.actor_id,
        "idempotency_key": record.idempotency_key,
        "subject_id": record.subject_id,
        "table_writes": [
            {
                "table": write.table,
                "key": write.key,
                "op": write.op,
                "before": _world_commit_value_to_wire(write.table, write.before),
                "after": _world_commit_value_to_wire(write.table, write.after),
            }
            for write in record.table_writes
        ],
        "invariants_held": list(record.invariants_held),
        "request_fingerprint": record.request_fingerprint,
        "schema_version": record.schema_version,
    }
    return _canonical_json(payload)


def _world_commit_from_json(payload: str) -> WorldCommitRecord:
    """Decode and type-check a durable commit journal row."""

    value = json.loads(payload)
    if not isinstance(value, dict):
        raise WorldError("world commit row is not an object")
    if value.get("schema_version") != "cwe.world-commit.v1":
        raise WorldError("unsupported durable world commit schema")
    raw_writes = value.get("table_writes")
    if not isinstance(raw_writes, list) or not raw_writes:
        raise WorldError("durable world commit has no table writes")
    writes: list[TableWrite] = []
    for raw in raw_writes:
        if not isinstance(raw, dict):
            raise WorldError("durable world commit write is not an object")
        table = str(raw["table"])
        writes.append(
            TableWrite(
                table=table,
                key=str(raw["key"]),
                op=str(raw["op"]),
                before=_world_commit_value_from_wire(table, raw.get("before")),
                after=_world_commit_value_from_wire(table, raw.get("after")),
            )
        )
    return WorldCommitRecord(
        sequence=int(value["sequence"]),
        commit_id=str(value["commit_id"]),
        commit_kind=str(value["commit_kind"]),
        operation=str(value["operation"]),
        authority_action=str(value["authority_action"]),
        actor_id=(None if value.get("actor_id") is None else str(value["actor_id"])),
        idempotency_key=(
            None
            if value.get("idempotency_key") is None
            else str(value["idempotency_key"])
        ),
        subject_id=str(value["subject_id"]),
        table_writes=tuple(writes),
        invariants_held=tuple(str(row) for row in value["invariants_held"]),
        request_fingerprint=(
            None
            if value.get("request_fingerprint") is None
            else str(value["request_fingerprint"])
        ),
    )


def _world_commit_value_to_wire(table: str, value: Any) -> Any:
    if value is None:
        return None
    if table == "payment_states":
        return payment_state_to_dict(cast(PaymentStateRecord, value))
    if table == "packing_records":
        return packing_record_to_dict(cast(PackingRecord, value))
    if table == "after_sales_policies":
        return after_sales_record_to_wire(value)
    if table == "after_sales_records":
        return after_sales_write_to_wire(cast(AfterSalesWrite, value))
    if table == "governance_policies":
        return policy_envelope_to_wire(cast(GovernancePolicyEnvelope, value))
    if table == "governance_records":
        return record_envelope_to_wire(cast(GovernanceRecordEnvelope, value))
    if table == "persistent_cart_quote_requests":
        return persistent_cart_quote_request_to_dict(
            cast(PersistentCartQuoteRequest, value)
        )
    if table == "persistent_cart_quotes":
        return json.loads(persistent_cart_quote_to_json(cast(PersistentCartQuote, value)))
    if table == "search_sessions":
        return search_session_to_wire(cast(SearchSession, value))
    return json.loads(_value_to_json(value))


def _world_commit_value_from_wire(table: str, value: Any) -> Any:
    if value is None:
        return None
    if table == "logical_time":
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorldError("world commit logical time is invalid")
        return value
    if not isinstance(value, dict):
        raise WorldError(f"world commit {table} row is not an object")
    if table == "payment_states":
        return payment_state_from_dict(value)
    if table == "packing_records":
        return packing_record_from_dict(value)
    if table == "after_sales_policies":
        return after_sales_record_from_wire("after_sales_policies", value)
    if table == "after_sales_records":
        if set(value) != {"table", "key", "value"}:
            raise WorldError("world commit after-sales wrapper fields are not exact")
        record_table = str(value["table"])
        return AfterSalesWrite(
            record_table,
            str(value["key"]),
            after_sales_record_from_wire(record_table, value["value"]),
        )
    if table == "governance_policies":
        return policy_envelope_from_wire(value)
    if table == "governance_records":
        return record_envelope_from_wire(value)
    if table == "persistent_cart_quote_requests":
        return coerce_persistent_cart_quote_request(value)
    if table == "persistent_cart_quotes":
        return persistent_cart_quote_from_json(_canonical_json(value))
    if table == "protocol_events":
        return protocol_event_from_json(_canonical_json(value))
    if table == "protocol_receipts":
        return protocol_event_receipt_from_json(_canonical_json(value))
    if table == "order_groups":
        return _order_group_from_json(_canonical_json(value))
    if table == "search_sessions":
        return coerce_search_session(value)
    if table == "inventory":
        return InventoryRow(
            sku_id=SkuId(str(value["sku_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            qty_available=int(value["qty_available"]),
            qty_reserved=int(value["qty_reserved"]),
            eta_day=int(value.get("eta_day", 0)),
            version=int(value.get("version", 0)),
            supply_events=tuple(
                SupplyEventRecord(
                    scope=str(event["scope"]),
                    idempotency_key=str(event["idempotency_key"]),
                    sku_id=SkuId(str(event["sku_id"])),
                    qty_delta=int(event["qty_delta"]),
                    eta_day=(
                        None
                        if event.get("eta_day") is None
                        else int(event["eta_day"])
                    ),
                    unit_price_cents=(
                        None
                        if event.get("unit_price_cents") is None
                        else int(event["unit_price_cents"])
                    ),
                    expected_version=(
                        None
                        if event.get("expected_version") is None
                        else int(event["expected_version"])
                    ),
                    original_actor=str(event["original_actor"]),
                    outcome=_supply_state_from_mapping(event["outcome"]),
                )
                for event in value.get("supply_events", [])
            ),
        )
    if table == "orders":
        price = cast(Mapping[str, Any], value["agreed_price"])
        return Order(
            order_id=OrderId(str(value["order_id"])),
            buyer_id=AgentId(str(value["buyer_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            sku_id=SkuId(str(value["sku_id"])),
            qty=int(value["qty"]),
            agreed_price=Money(
                Decimal(str(price["amount"])), str(price["currency"])
            ),
            state=OrderState(str(value["state"])),
            request_order=int(value.get("request_order", 0)),
        )
    if table == "ledger":
        price = cast(Mapping[str, Any], value["price"])
        return Receipt(
            txn_id=TxnId(str(value["txn_id"])),
            ts=str(value["ts"]),
            order_id=OrderId(str(value["order_id"])),
            buyer_id=AgentId(str(value["buyer_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            sku_id=SkuId(str(value["sku_id"])),
            qty=int(value["qty"]),
            price=Money(Decimal(str(price["amount"])), str(price["currency"])),
            idempotency_key=str(value["idempotency_key"]),
            effect=str(value.get("effect", "charge")),
        )
    if table == "order_timelines":
        return OrderTimeline(
            order_id=OrderId(str(value["order_id"])),
            buyer_id=AgentId(str(value["buyer_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            settled_at_tick=(
                None
                if value.get("settled_at_tick") is None
                else int(value["settled_at_tick"])
            ),
            dispatched_at_tick=(
                None
                if value.get("dispatched_at_tick") is None
                else int(value["dispatched_at_tick"])
            ),
            return_window_ticks=(
                None
                if value.get("return_window_ticks") is None
                else int(value["return_window_ticks"])
            ),
            return_authorized_at_tick=(
                None
                if value.get("return_authorized_at_tick") is None
                else int(value["return_authorized_at_tick"])
            ),
            returned_at_tick=(
                None
                if value.get("returned_at_tick") is None
                else int(value["returned_at_tick"])
            ),
            refunded_at_tick=(
                None
                if value.get("refunded_at_tick") is None
                else int(value["refunded_at_tick"])
            ),
        )
    if table == "fulfillments":
        receipt_txn_id = value.get("receipt_txn_id")
        return FulfillmentAllocation(
            order_id=OrderId(str(value["order_id"])),
            buyer_id=AgentId(str(value["buyer_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            sku_id=SkuId(str(value["sku_id"])),
            requested_qty=int(value["requested_qty"]),
            fulfilled_qty=int(value["fulfilled_qty"]),
            backordered_qty=int(value["backordered_qty"]),
            receipt_txn_id=(
                None if receipt_txn_id is None else TxnId(str(receipt_txn_id))
            ),
            created_by=AgentId(str(value["created_by"])),
            idempotency_key=str(value["idempotency_key"]),
            allocation_id=(
                None
                if value.get("allocation_id") is None
                else str(value["allocation_id"])
            ),
        )
    if table == "shipments":
        return _shipment_from_json(_canonical_json(value))
    if table == "disputes":
        return Dispute(
            dispute_id=DisputeId(str(value["dispute_id"])),
            order_id=OrderId(str(value["order_id"])),
            filed_by=AgentId(str(value["filed_by"])),
            against=AgentId(str(value["against"])),
            reason=str(value["reason"]),
            state=DisputeState(str(value["state"])),
        )
    if table == "rulings":
        raw_refund = value.get("refund_amount")
        refund = None
        if isinstance(raw_refund, dict):
            refund = Money(
                Decimal(str(raw_refund["amount"])),
                str(raw_refund["currency"]),
            )
        return Ruling(
            ruling_id=RulingId(str(value["ruling_id"])),
            dispute_id=DisputeId(str(value["dispute_id"])),
            in_favor_of=AgentId(str(value["in_favor_of"])),
            rationale=str(value["rationale"]),
            refund_amount=refund,
        )
    if table == "reviews":
        return Review(
            review_id=ReviewId(str(value["review_id"])),
            reviewer_id=AgentId(str(value["reviewer_id"])),
            sku_id=SkuId(str(value["sku_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            rating=int(value["rating"]),
            text=str(value.get("text", "")),
        )
    if table == "reputation":
        return ReputationScore(
            merchant_id=AgentId(str(value["merchant_id"])),
            rolling_avg=float(value["rolling_avg"]),
            n_settled=int(value["n_settled"]),
            n_disputed=int(value["n_disputed"]),
        )
    if table == "evidence_records":
        return coerce_evidence_record(value)
    if table == "supply_purchase_authorities":
        from protocol.supply_authority import coerce_supply_purchase_authority

        return coerce_supply_purchase_authority(value)
    if table == "exchanges":
        return Exchange(
            exchange_id=ExchangeId(str(value["exchange_id"])),
            original_order_id=OrderId(str(value["original_order_id"])),
            replacement_order_id=OrderId(str(value["replacement_order_id"])),
            buyer_id=AgentId(str(value["buyer_id"])),
            merchant_id=AgentId(str(value["merchant_id"])),
            original_sku_id=SkuId(str(value["original_sku_id"])),
            replacement_sku_id=SkuId(str(value["replacement_sku_id"])),
            qty=int(value["qty"]),
            created_by=AgentId(str(value["created_by"])),
            idempotency_key=str(value["idempotency_key"]),
        )
    if table == "authority_operations":
        raw_listing = value.get("outcome_listing")
        return AuthorityOperationRecord(
            operation_key=str(value["operation_key"]),
            scope=str(value["scope"]),
            actor_id=str(value["actor_id"]),
            idempotency_key=str(value["idempotency_key"]),
            request_fingerprint=str(value["request_fingerprint"]),
            outcome_table=str(value["outcome_table"]),
            outcome_key=str(value["outcome_key"]),
            outcome_listing=(
                None if raw_listing is None else _listing_from_mapping(raw_listing)
            ),
        )
    raise WorldError(f"unsupported typed durable commit table: {table}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"), sort_keys=True)


def _value_to_json(value: Any) -> str:
    return _canonical_json(value)


def _reputation_settlement_from_json(value: str) -> ReputationSettlement:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise WorldError("reputation settlement row is not an object")
    outcome = payload.get("outcome")
    sources = payload.get("sources")
    if not isinstance(outcome, dict) or not isinstance(sources, list):
        raise WorldError("reputation settlement row has invalid evidence")
    return ReputationSettlement(
        event_id=str(payload["event_id"]),
        order_id=OrderId(str(payload["order_id"])),
        txn_id=TxnId(str(payload["txn_id"])),
        merchant_id=AgentId(str(payload["merchant_id"])),
        sources=tuple(
            ReputationSettlementSource(
                source_actor=str(source["source_actor"]),
                source_request_id=str(source["source_request_id"]),
                source_idempotency_key=str(source["source_idempotency_key"]),
            )
            for source in sources
            if isinstance(source, dict)
        ),
        outcome=ReputationScore(
            merchant_id=AgentId(str(outcome["merchant_id"])),
            rolling_avg=float(outcome["rolling_avg"]),
            n_settled=int(outcome["n_settled"]),
            n_disputed=int(outcome["n_disputed"]),
        ),
    )


def _supply_state_from_mapping(value: Any) -> SupplyState:
    if not isinstance(value, dict):
        raise ValueError("supply outcome must be an object")
    return SupplyState(
        sku_id=SkuId(str(value["sku_id"])),
        merchant_id=AgentId(str(value["merchant_id"])),
        available_qty=int(value["available_qty"]),
        reserved_qty=int(value["reserved_qty"]),
        eta_day=int(value["eta_day"]),
        unit_price_cents=int(value["unit_price_cents"]),
        version=int(value["version"]),
    )


def _allocation_batch_from_json(value: str) -> AllocationBatch:
    payload = json.loads(value)
    allocations = tuple(
        FulfillmentAllocation(
            order_id=OrderId(str(row["order_id"])),
            buyer_id=AgentId(str(row["buyer_id"])),
            merchant_id=AgentId(str(row["merchant_id"])),
            sku_id=SkuId(str(row["sku_id"])),
            requested_qty=int(row["requested_qty"]),
            fulfilled_qty=int(row["fulfilled_qty"]),
            backordered_qty=int(row["backordered_qty"]),
            receipt_txn_id=(
                None
                if row["receipt_txn_id"] is None
                else TxnId(str(row["receipt_txn_id"]))
            ),
            created_by=AgentId(str(row["created_by"])),
            idempotency_key=str(row["idempotency_key"]),
            allocation_id=(
                None
                if row.get("allocation_id") is None
                else str(row["allocation_id"])
            ),
        )
        for row in payload["allocations"]
    )
    return AllocationBatch(
        allocation_id=str(payload["allocation_id"]),
        merchant_id=AgentId(str(payload["merchant_id"])),
        sku_id=SkuId(str(payload["sku_id"])),
        priority_order_ids=tuple(
            OrderId(str(row)) for row in payload["priority_order_ids"]
        ),
        allocations=allocations,
        created_by=AgentId(str(payload["created_by"])),
        idempotency_key=str(payload["idempotency_key"]),
    )


def _shipment_from_json(value: str) -> Shipment:
    row = json.loads(value)
    return Shipment(
        shipment_id=ShipmentId(str(row["shipment_id"])),
        order_id=OrderId(str(row["order_id"])),
        buyer_id=AgentId(str(row["buyer_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        original_sku_id=SkuId(str(row["original_sku_id"])),
        status=ShipmentStatus(str(row["status"])),
        status_history=tuple(
            ShipmentStatusEvent(
                event_id=str(event["event_id"]),
                status=ShipmentStatus(str(event["status"])),
                logical_time=int(event["logical_time"]),
                idempotency_key=(
                    None
                    if event.get("idempotency_key") is None
                    else str(event["idempotency_key"])
                ),
                shipment_version=(
                    None
                    if event.get("shipment_version") is None
                    else int(event["shipment_version"])
                ),
            )
            for event in row["status_history"]
        ),
        resolution=(
            None
            if row.get("resolution") is None
            else ShipmentResolution(str(row["resolution"]))
        ),
        replacement_sku_id=(
            None
            if row.get("replacement_sku_id") is None
            else SkuId(str(row["replacement_sku_id"]))
        ),
        version=int(row["version"]),
        resolution_idempotency_key=(
            None
            if row.get("resolution_idempotency_key") is None
            else str(row["resolution_idempotency_key"])
        ),
        resolved_by=(
            None if row.get("resolved_by") is None else AgentId(str(row["resolved_by"]))
        ),
        resolution_version=(
            None
            if row.get("resolution_version") is None
            else int(row["resolution_version"])
        ),
        resolution_history_length=(
            None
            if row.get("resolution_history_length") is None
            else int(row["resolution_history_length"])
        ),
    )


def _paid_quantity_from_store(store: SQLiteWorldStore, order: Order) -> int:
    allocation = cast(
        "FulfillmentAllocation | None",
        store._fetch_value("fulfillments", str(order.order_id)),
    )
    return order.qty if allocation is None else allocation.fulfilled_qty


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if not isinstance(value, type) and is_dataclass(value):
        return {
            field.name: getattr(value, field.name)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS world_schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS world_clock (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  logical_time INTEGER NOT NULL CHECK (logical_time >= 0)
);

CREATE TABLE IF NOT EXISTS catalog (
  sku_id TEXT PRIMARY KEY,
  merchant_id TEXT NOT NULL,
  product_id TEXT,
  category TEXT NOT NULL,
  name TEXT NOT NULL,
  attributes_json TEXT NOT NULL,
  list_price_minor INTEGER NOT NULL,
  list_price_currency TEXT NOT NULL DEFAULT 'USD',
  status TEXT NOT NULL DEFAULT 'active',
  version INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_catalog_merchant ON catalog (merchant_id);
CREATE INDEX IF NOT EXISTS idx_catalog_category ON catalog (category);
CREATE INDEX IF NOT EXISTS idx_catalog_product ON catalog (product_id);

CREATE TABLE IF NOT EXISTS inventory (
  sku_id TEXT PRIMARY KEY,
  merchant_id TEXT NOT NULL,
  qty_available INTEGER NOT NULL CHECK (qty_available >= 0),
  qty_reserved INTEGER NOT NULL CHECK (qty_reserved >= 0),
  eta_day INTEGER NOT NULL DEFAULT 0 CHECK (eta_day >= 0),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
  supply_events_json TEXT NOT NULL DEFAULT '[]',
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inventory_merchant ON inventory (merchant_id);

CREATE TABLE IF NOT EXISTS search_sessions (
  session_id TEXT PRIMARY KEY,
  buyer_id TEXT NOT NULL,
  search_idempotency_key TEXT NOT NULL,
  session_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (buyer_id, search_idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_search_sessions_buyer
  ON search_sessions (buyer_id, created_at);

CREATE TABLE IF NOT EXISTS search_session_offers (
  session_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  offer_id TEXT NOT NULL,
  expires_at_tick INTEGER NOT NULL,
  PRIMARY KEY (session_id, offer_id)
);
CREATE INDEX IF NOT EXISTS idx_search_session_offers_lookup
  ON search_session_offers (buyer_id, offer_id, expires_at_tick, session_id);

CREATE TABLE IF NOT EXISTS match_acceptances (
  acceptance_key TEXT PRIMARY KEY,
  buyer_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  acceptance_digest TEXT NOT NULL UNIQUE,
  acceptance_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (buyer_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_match_acceptances_buyer
  ON match_acceptances (buyer_id, created_at);

CREATE TABLE IF NOT EXISTS match_certificates (
  cert_id TEXT PRIMARY KEY,
  acceptance_digest TEXT NOT NULL UNIQUE,
  buyer_id TEXT,
  order_id TEXT,
  expires_at_tick INTEGER,
  certificate_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS supply_purchase_authorities (
  authority_id TEXT PRIMARY KEY,
  authority_digest TEXT NOT NULL UNIQUE,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  sku_id TEXT NOT NULL,
  order_id TEXT NOT NULL UNIQUE,
  supply_version INTEGER NOT NULL CHECK (supply_version > 0),
  expires_at_tick INTEGER NOT NULL CHECK (expires_at_tick > 0),
  authority_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_supply_authority_buyer
  ON supply_purchase_authorities (buyer_id, authority_id);
CREATE INDEX IF NOT EXISTS idx_supply_authority_sku
  ON supply_purchase_authorities (sku_id, supply_version, authority_id);

CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  sku_id TEXT NOT NULL,
  qty INTEGER NOT NULL CHECK (qty > 0),
  agreed_price_minor INTEGER NOT NULL,
  agreed_price_currency TEXT NOT NULL DEFAULT 'USD',
  state TEXT NOT NULL,
  request_order INTEGER NOT NULL DEFAULT 0 CHECK (request_order >= 0),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_buyer ON orders (buyer_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_orders_merchant ON orders (merchant_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_orders_sku ON orders (sku_id);

CREATE TABLE IF NOT EXISTS order_state_revisions (
  order_id TEXT PRIMARY KEY,
  revision INTEGER NOT NULL CHECK (revision > 0)
);

CREATE TABLE IF NOT EXISTS order_groups (
  order_group_id TEXT PRIMARY KEY,
  quote_id TEXT NOT NULL UNIQUE,
  buyer_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  group_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_order_groups_buyer
  ON order_groups (buyer_id, order_group_id);

CREATE TABLE IF NOT EXISTS evidence_records (
  evidence_key TEXT PRIMARY KEY,
  record_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK (version > 0),
  record_digest TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  issuer_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  record_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (record_id, version),
  UNIQUE (record_id, record_digest)
);
CREATE INDEX IF NOT EXISTS idx_evidence_records_current
  ON evidence_records (record_id, version DESC);

CREATE TABLE IF NOT EXISTS mandate_authorities (
  mandate_id TEXT PRIMARY KEY,
  principal_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  authority_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mandate_authorities_buyer
  ON mandate_authorities (buyer_id, mandate_id);

CREATE TABLE IF NOT EXISTS mandate_revisions (
  revision_key TEXT PRIMARY KEY,
  mandate_id TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision > 0),
  revision_digest TEXT NOT NULL,
  principal_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  revision_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (mandate_id, revision),
  UNIQUE (mandate_id, revision_digest)
);
CREATE INDEX IF NOT EXISTS idx_mandate_revisions_history
  ON mandate_revisions (mandate_id, revision);

CREATE TABLE IF NOT EXISTS listing_claims (
  claim_id TEXT PRIMARY KEY,
  listing_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  claim_state TEXT NOT NULL,
  current_version INTEGER NOT NULL CHECK (current_version > 0),
  current_event_digest TEXT NOT NULL,
  claim_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_listing_claims_listing
  ON listing_claims (listing_id, claim_id);

CREATE TABLE IF NOT EXISTS listing_claim_keys (
  merchant_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  claim_id TEXT NOT NULL,
  event_digest TEXT NOT NULL,
  PRIMARY KEY (merchant_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS authority_operations (
  operation_key TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  outcome_table TEXT NOT NULL,
  outcome_key TEXT NOT NULL,
  outcome_listing_json TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (scope, actor_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS negotiation_events (
  event_id TEXT PRIMARY KEY,
  negotiation_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  sequence_no INTEGER NOT NULL CHECK (sequence_no > 0),
  event_digest TEXT NOT NULL UNIQUE,
  event_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (negotiation_id, sequence_no),
  UNIQUE (negotiation_id, actor_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_negotiation_events_thread
  ON negotiation_events (negotiation_id, sequence_no, event_id);
CREATE INDEX IF NOT EXISTS idx_negotiation_events_parties
  ON negotiation_events (buyer_id, merchant_id, negotiation_id);

CREATE TABLE IF NOT EXISTS negotiation_threads (
  negotiation_id TEXT PRIMARY KEY,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active', 'accepted', 'rejected', 'withdrawn')),
  event_count INTEGER NOT NULL CHECK (event_count > 0),
  head_event_digest TEXT NOT NULL,
  thread_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_negotiation_threads_parties
  ON negotiation_threads (buyer_id, merchant_id, negotiation_id);

CREATE TABLE IF NOT EXISTS pricing_policy_revisions (
  revision_key TEXT PRIMARY KEY,
  market_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  policy_id TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision > 0),
  actor_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  policy_digest TEXT NOT NULL UNIQUE,
  policy_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (market_id, merchant_id, policy_id, revision),
  UNIQUE (market_id, actor_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_pricing_policy_stream
  ON pricing_policy_revisions
  (market_id, merchant_id, policy_id, revision);
CREATE INDEX IF NOT EXISTS idx_pricing_policy_scope
  ON pricing_policy_revisions (market_id, merchant_id, revision);

CREATE TABLE IF NOT EXISTS persistent_cart_quote_requests (
  request_id TEXT PRIMARY KEY,
  market_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  principal_id TEXT NOT NULL,
  created_by TEXT NOT NULL,
  mandate_id TEXT NOT NULL,
  mandate_revision INTEGER NOT NULL CHECK (mandate_revision > 0),
  idempotency_key TEXT NOT NULL,
  request_digest TEXT NOT NULL UNIQUE,
  issued_at_tick INTEGER NOT NULL CHECK (issued_at_tick >= 0),
  expires_at_tick INTEGER NOT NULL CHECK (expires_at_tick > issued_at_tick),
  request_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (market_id, created_by, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_cart_quote_requests_buyer
  ON persistent_cart_quote_requests
  (market_id, buyer_id, issued_at_tick, request_id);
CREATE INDEX IF NOT EXISTS idx_cart_quote_requests_expiry
  ON persistent_cart_quote_requests
  (market_id, expires_at_tick, request_id);

CREATE TABLE IF NOT EXISTS persistent_cart_quotes (
  quote_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  market_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  principal_id TEXT NOT NULL,
  requested_by TEXT NOT NULL,
  issuer_id TEXT NOT NULL,
  mandate_id TEXT NOT NULL,
  mandate_revision INTEGER NOT NULL CHECK (mandate_revision > 0),
  idempotency_key TEXT NOT NULL,
  quote_digest TEXT NOT NULL UNIQUE,
  issued_at_tick INTEGER NOT NULL CHECK (issued_at_tick >= 0),
  expires_at_tick INTEGER NOT NULL CHECK (expires_at_tick > issued_at_tick),
  quote_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (market_id, request_id),
  UNIQUE (market_id, requested_by, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_cart_quotes_buyer
  ON persistent_cart_quotes (market_id, buyer_id, issued_at_tick, quote_id);
CREATE INDEX IF NOT EXISTS idx_cart_quotes_expiry
  ON persistent_cart_quotes (market_id, expires_at_tick, quote_id);

CREATE TABLE IF NOT EXISTS protocol_events (
  event_id TEXT PRIMARY KEY,
  market_id TEXT NOT NULL,
  stream_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  recipient_id TEXT NOT NULL,
  authority_id TEXT NOT NULL,
  binding_digest TEXT NOT NULL,
  event_sequence INTEGER NOT NULL CHECK (event_sequence > 0),
  event_digest TEXT NOT NULL UNIQUE,
  actor_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  issued_at_tick INTEGER NOT NULL CHECK (issued_at_tick >= 0),
  expires_at_tick INTEGER NOT NULL CHECK (expires_at_tick > issued_at_tick),
  event_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (binding_digest, event_sequence),
  UNIQUE (binding_digest, actor_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_protocol_events_order
  ON protocol_events (order_id, stream_id, event_sequence, event_id);
CREATE INDEX IF NOT EXISTS idx_protocol_events_recipient
  ON protocol_events (recipient_id, issued_at_tick, event_id);

CREATE TABLE IF NOT EXISTS protocol_receipts (
  receipt_id TEXT PRIMARY KEY,
  market_id TEXT NOT NULL,
  stream_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  recipient_id TEXT NOT NULL,
  authority_id TEXT NOT NULL,
  binding_digest TEXT NOT NULL,
  event_id TEXT NOT NULL,
  event_digest TEXT NOT NULL,
  receipt_digest TEXT NOT NULL UNIQUE,
  actor_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('acknowledge', 'reject', 'process')),
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  receipt_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (binding_digest, event_digest),
  UNIQUE (binding_digest, actor_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_protocol_receipts_order
  ON protocol_receipts (order_id, stream_id, logical_tick, receipt_id);
CREATE INDEX IF NOT EXISTS idx_protocol_receipts_recipient
  ON protocol_receipts (recipient_id, logical_tick, receipt_id);

CREATE TABLE IF NOT EXISTS order_timelines (
  order_id TEXT PRIMARY KEY,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  settled_at_tick INTEGER CHECK (settled_at_tick IS NULL OR settled_at_tick >= 0),
  dispatched_at_tick INTEGER CHECK (dispatched_at_tick IS NULL OR dispatched_at_tick >= 0),
  return_window_ticks INTEGER CHECK (return_window_ticks IS NULL OR return_window_ticks > 0),
  return_authorized_at_tick INTEGER
    CHECK (return_authorized_at_tick IS NULL OR return_authorized_at_tick >= 0),
  returned_at_tick INTEGER CHECK (returned_at_tick IS NULL OR returned_at_tick >= 0),
  refunded_at_tick INTEGER CHECK (refunded_at_tick IS NULL OR refunded_at_tick >= 0)
);
CREATE INDEX IF NOT EXISTS idx_order_timelines_buyer
  ON order_timelines (buyer_id, order_id);
CREATE INDEX IF NOT EXISTS idx_order_timelines_merchant
  ON order_timelines (merchant_id, order_id);

CREATE TABLE IF NOT EXISTS fulfillments (
  order_id TEXT PRIMARY KEY,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  sku_id TEXT NOT NULL,
  requested_qty INTEGER NOT NULL CHECK (requested_qty > 0),
  fulfilled_qty INTEGER NOT NULL CHECK (fulfilled_qty >= 0),
  backordered_qty INTEGER NOT NULL CHECK (backordered_qty >= 0),
  receipt_txn_id TEXT,
  created_by TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  allocation_id TEXT,
  created_at TEXT NOT NULL,
  CHECK (requested_qty = fulfilled_qty + backordered_qty),
  CHECK ((fulfilled_qty = 0 AND receipt_txn_id IS NULL)
      OR (fulfilled_qty > 0 AND receipt_txn_id IS NOT NULL)),
  UNIQUE (created_by, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_fulfillments_buyer
  ON fulfillments (buyer_id, created_at);
CREATE INDEX IF NOT EXISTS idx_fulfillments_merchant
  ON fulfillments (merchant_id, created_at);

CREATE TABLE IF NOT EXISTS exchanges (
  exchange_id TEXT PRIMARY KEY,
  original_order_id TEXT NOT NULL UNIQUE,
  replacement_order_id TEXT NOT NULL UNIQUE,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  original_sku_id TEXT NOT NULL,
  replacement_sku_id TEXT NOT NULL,
  qty INTEGER NOT NULL CHECK (qty > 0),
  created_by TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (created_by, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_exchanges_buyer
  ON exchanges (buyer_id, created_at);
CREATE INDEX IF NOT EXISTS idx_exchanges_merchant
  ON exchanges (merchant_id, created_at);

CREATE TABLE IF NOT EXISTS shipments (
  shipment_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL UNIQUE,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  original_sku_id TEXT NOT NULL,
  status TEXT NOT NULL,
  status_history_json TEXT NOT NULL,
  resolution TEXT,
  replacement_sku_id TEXT,
  version INTEGER NOT NULL CHECK (version > 0),
  resolution_idempotency_key TEXT,
  resolved_by TEXT,
  resolution_version INTEGER,
  resolution_history_length INTEGER,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shipments_buyer
  ON shipments (buyer_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_shipments_merchant
  ON shipments (merchant_id, updated_at);

CREATE TABLE IF NOT EXISTS ledger (
  txn_id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  order_id TEXT NOT NULL,
  buyer_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  sku_id TEXT NOT NULL,
  qty INTEGER NOT NULL CHECK (qty > 0),
  price_minor INTEGER NOT NULL,
  price_amount_text TEXT NOT NULL,
  price_currency TEXT NOT NULL DEFAULT 'USD',
  idempotency_key TEXT NOT NULL,
  effect TEXT NOT NULL DEFAULT 'charge' CHECK (effect IN ('charge', 'refund'))
);
CREATE INDEX IF NOT EXISTS idx_ledger_order ON ledger (order_id);
CREATE INDEX IF NOT EXISTS idx_ledger_buyer ON ledger (buyer_id, ts);
CREATE INDEX IF NOT EXISTS idx_ledger_merchant ON ledger (merchant_id, ts);

CREATE TABLE IF NOT EXISTS idempotency_records (
  scope TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  operation TEXT NOT NULL CHECK (operation IN ('settle', 'refund')),
  request_fingerprint TEXT NOT NULL,
  outcome_txn_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS supply_event_records (
  scope TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  outcome_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS allocation_batch_records (
  allocation_id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  outcome_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS shipment_event_records (
  scope TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  shipment_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  outcome_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS shipment_resolution_records (
  scope TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  shipment_id TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  outcome_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (scope, idempotency_key)
);

CREATE TABLE IF NOT EXISTS reputation (
  merchant_id TEXT PRIMARY KEY,
  rolling_avg REAL NOT NULL,
  n_settled INTEGER NOT NULL CHECK (n_settled >= 0),
  n_disputed INTEGER NOT NULL CHECK (n_disputed >= 0),
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY,
  reviewer_id TEXT NOT NULL,
  sku_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  review_text TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_listing
  ON reviews (sku_id, merchant_id, review_id);

CREATE TABLE IF NOT EXISTS reputation_settlements (
  event_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  txn_id TEXT NOT NULL UNIQUE,
  merchant_id TEXT NOT NULL,
  event_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reputation_settlements_merchant
  ON reputation_settlements (merchant_id, txn_id);

CREATE TABLE IF NOT EXISTS reputation_settlement_sources (
  source_actor TEXT NOT NULL,
  source_idempotency_key TEXT NOT NULL,
  source_request_id TEXT NOT NULL,
  event_id TEXT NOT NULL REFERENCES reputation_settlements(event_id)
    ON DELETE CASCADE,
  PRIMARY KEY (source_actor, source_idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_reputation_settlement_sources_event
  ON reputation_settlement_sources (event_id);

CREATE TABLE IF NOT EXISTS disputes (
  dispute_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  filed_by TEXT NOT NULL,
  against TEXT NOT NULL,
  reason TEXT NOT NULL,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_disputes_order ON disputes (order_id);
CREATE INDEX IF NOT EXISTS idx_disputes_filed_by ON disputes (filed_by, updated_at);
CREATE INDEX IF NOT EXISTS idx_disputes_against ON disputes (against, updated_at);

CREATE TABLE IF NOT EXISTS rulings (
  ruling_id TEXT PRIMARY KEY,
  dispute_id TEXT NOT NULL,
  in_favor_of TEXT NOT NULL,
  rationale TEXT NOT NULL,
  refund_amount_minor INTEGER,
  refund_amount_currency TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rulings_dispute ON rulings (dispute_id);

CREATE TABLE IF NOT EXISTS payment_states (
  record_key TEXT PRIMARY KEY,
  payment_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  payment_state TEXT NOT NULL,
  version INTEGER NOT NULL CHECK (version > 0),
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  record_digest TEXT NOT NULL UNIQUE,
  record_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (payment_id, version)
);
CREATE INDEX IF NOT EXISTS idx_payment_states_order
  ON payment_states (order_id, version);
CREATE INDEX IF NOT EXISTS idx_payment_states_parties
  ON payment_states (owner_id, merchant_id, order_id);

CREATE TABLE IF NOT EXISTS packing_records (
  record_key TEXT PRIMARY KEY,
  packing_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  packing_state TEXT NOT NULL,
  version INTEGER NOT NULL CHECK (version > 0),
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  record_digest TEXT NOT NULL UNIQUE,
  record_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (packing_id, version)
);
CREATE INDEX IF NOT EXISTS idx_packing_records_order
  ON packing_records (order_id, version);
CREATE INDEX IF NOT EXISTS idx_packing_records_parties
  ON packing_records (owner_id, merchant_id, order_id);

CREATE TABLE IF NOT EXISTS after_sales_policies (
  policy_key TEXT PRIMARY KEY,
  policy_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision > 0),
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  policy_digest TEXT NOT NULL UNIQUE,
  policy_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (merchant_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_after_sales_policies_merchant
  ON after_sales_policies (merchant_id, revision);

CREATE TABLE IF NOT EXISTS after_sales_records (
  physical_key TEXT PRIMARY KEY,
  record_kind TEXT NOT NULL,
  domain_key TEXT NOT NULL,
  order_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  merchant_id TEXT NOT NULL,
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  record_digest TEXT NOT NULL UNIQUE,
  record_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (record_kind, domain_key)
);
CREATE INDEX IF NOT EXISTS idx_after_sales_records_order
  ON after_sales_records (order_id, logical_tick, record_kind, domain_key);
CREATE INDEX IF NOT EXISTS idx_after_sales_records_parties
  ON after_sales_records (owner_id, merchant_id, order_id);

CREATE TABLE IF NOT EXISTS governance_policies (
  envelope_key TEXT PRIMARY KEY,
  record_kind TEXT NOT NULL,
  stable_id TEXT NOT NULL,
  revision INTEGER NOT NULL CHECK (revision > 0),
  service_actor TEXT NOT NULL,
  original_actor TEXT NOT NULL,
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  envelope_digest TEXT NOT NULL UNIQUE,
  envelope_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (record_kind, stable_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_governance_policies_stream
  ON governance_policies (record_kind, stable_id, revision);

CREATE TABLE IF NOT EXISTS governance_records (
  envelope_key TEXT PRIMARY KEY,
  record_kind TEXT NOT NULL,
  stable_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK (version > 0),
  service_actor TEXT NOT NULL,
  original_actor TEXT NOT NULL,
  logical_tick INTEGER NOT NULL CHECK (logical_tick >= 0),
  envelope_digest TEXT NOT NULL UNIQUE,
  envelope_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (record_kind, stable_id, version)
);
CREATE INDEX IF NOT EXISTS idx_governance_records_stream
  ON governance_records (record_kind, stable_id, version);

CREATE TABLE IF NOT EXISTS world_commit_records (
  sequence INTEGER PRIMARY KEY CHECK (sequence >= 0),
  commit_id TEXT NOT NULL UNIQUE,
  commit_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS world_mutations (
  mutation_id TEXT PRIMARY KEY,
  table_name TEXT NOT NULL,
  row_key TEXT NOT NULL,
  action_kind TEXT NOT NULL,
  before_json TEXT,
  after_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_world_mutations_table_row
  ON world_mutations (table_name, row_key, created_at);

CREATE TABLE IF NOT EXISTS transaction_diffs (
  diff_id TEXT PRIMARY KEY,
  mutation_id TEXT NOT NULL REFERENCES world_mutations (mutation_id),
  table_name TEXT NOT NULL,
  row_key TEXT NOT NULL,
  op TEXT NOT NULL CHECK (op IN ('insert', 'update', 'delete')),
  before_json TEXT,
  after_json TEXT,
  ordinal INTEGER NOT NULL,
  UNIQUE (mutation_id, ordinal)
);

CREATE TABLE IF NOT EXISTS world_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  snapshot_hash TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""
