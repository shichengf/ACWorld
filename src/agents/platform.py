"""Commerce Intelligence Layer — Platform Service (provisional location).

In the ACWorld architecture, the Platform Service is **not** an
``Agent`` — it is the Commerce Intelligence Layer hosted by the runtime,
between the buyer and merchant agents. It is addressable on the bus as
``platform`` but never registered via ``Runtime.register(agent)``.

This file is **provisional**: it survives under ``src/agents/`` only
because the import surface and tests still expect a ``make_platform_agent``
factory and three policy-module classes. The intended layout (per
``WORLD_CLASSES.md`` §8) is a sibling ``platform/`` package whose modules
are not built from agent primitives. That refactor is tracked separately;
this module compiles under the new ``Agent`` shape in the meantime.

Policy modules retained here for reference until they move to ``platform/``:

    * ``ForwardSearchSkill`` — aggregator pass-through against a single
      merchant. Multi-merchant ranking + sponsored-slot policy layer on later.
    * ``AtomicSettleSkill``  — PSP: only call site that writes
      ``ledger`` + ``inventory`` atomically via
      ``World.write(by_action='settle')``.
    * ``RejectDisputeSkill`` — adjudicator stub: every dispute rejected;
      richer rulings (lenient / strict / evidence-weighted / LLM-backed)
      are alternative configurations of the same Layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Literal, cast

from agents.types import SkillGroup
from agents.evidence_policy import ClaimStateRejected, EvidenceClaimsPolicy
from agents.platform_errors import CommerceActionRejected, RejectionCategory
from agents.negotiation import (
    NEGOTIATION_ACTIONS,
    NegotiationPolicy,
    NegotiationRejected,
    mediation_metadata,
)
from protocol.envelope import Envelope, to_json
from protocol.cart_quote_request import persistent_cart_quote_request_to_dict
from protocol.cart_quote_request import CartQuoteRequestStaleError
from protocol.cart_quote_state import CartQuoteStaleError, persistent_cart_quote_to_dict
from protocol.errors import SchemaError
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventAuthorityError,
    ProtocolEventReceipt,
    ReceiptDecision,
    build_next_protocol_event,
    build_protocol_event_binding,
    build_protocol_event_receipt,
    protocol_event_receipt_to_dict,
    protocol_event_to_dict,
    ProtocolEventStaleError,
)
from protocol.matching import (
    MatchAcceptance,
    MatchAcceptanceRejected,
    OfferSnapshot,
    SearchSession,
    canonical_digest,
    match_certificate_id,
    match_certificate_to_wire,
    offer_snapshot_to_wire,
    seal_match_acceptance,
    seal_search_session,
    search_session_to_wire,
    validate_match_certificate,
)
from protocol.market_governance import (
    Campaign,
    GovernanceCase,
    MarketSignal,
    RankingContext,
    RemediationPlan,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
)
from protocol.negotiation_state import (
    negotiated_order_id,
    validate_negotiation_thread,
)
from protocol.remediation_audit import (
    REMEDIATION_AUDITOR_SERVICE_ID,
    build_remediation_audit_request,
)
from protocol.supply_authority import (
    DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS,
    validate_supply_purchase_authority,
    validate_supply_purchase_authority_ttl_ticks,
)
from runtime.errors import PrivateUtilityLeak
from protocol.schemas import (
    AdjustPricePayload,
    ProtocolEventDecisionPayload,
    ReceiveShipmentPayload,
    UpdateListingPayload,
)
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesIntentError,
    normalize_after_sales_intent,
)
from world.catalog_mutations import normalize_catalog_mutation_intent
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceResolutionDecision,
    GovernanceResponseAttestation,
    GovernanceIntentError,
    GovernanceTransitionError,
    RemediationBlueprint,
    ReputationPolicyRevision,
    ReviewAccountBinding,
    normalize_governance_intent,
)
from world.errors import (
    AfterSalesReferenceRejected,
    CatalogMutationRejected,
    DisputeNotActionable,
    ExchangeNotActionable,
    FulfillmentNotActionable,
    InsufficientFunds,
    InvalidOrderTransition,
    LifecycleAuthorizationError,
    NoMerchantConsent,
    OrderIdentityMismatch,
    OrderNotRefundable,
    OrderNotSettleable,
    OutOfStock,
    ReturnWindowClosed,
    ShipmentNotActionable,
    IdempotencyConflict,
    WriteNotAuthorized,
)
from world.transactions import scope_idempotency_key
from world.match_authorizations import match_authorization_acceptance_key
from world.negotiations import negotiation_listing_digest
from world.types import (
    AgentId,
    AllocationBatch,
    Dispute,
    DisputeId,
    DisputeState,
    ExchangeId,
    InventoryRow,
    Listing,
    Money,
    Order,
    OrderId,
    OrderState,
    Receipt,
    ReputationScore,
    Ruling,
    RulingId,
    Shipment,
    ShipmentId,
    ShipmentResolution,
    ShipmentStatus,
    SkuId,
    SupplyState,
    TxnId,
)

if TYPE_CHECKING:
    from agents.base import Agent
    from agents.interfaces import Memory, SkillLoader
    from agents.types import AgentContext, AgentInputs
    from agents.world_client import WorldClient
    from runtime.audit import AuditLog
    from runtime.evidence import PlatformDecisionJournal
    from runtime.privacy import SecretRegistry
    from world.state import World  # only for the in-process convenience param type


PLATFORM_PSP_ACTIONS = frozenset({"platform.settle_payment", "settle"})
PLATFORM_REPUTATION_ACTIONS = frozenset({"platform.update_reputation", "update_reputation"})
PLATFORM_AGGREGATOR_ACTIONS = frozenset({"commerce.search", "commerce.accept_offer"})
# Merchant commerce-namespace intents the platform translates into compact
# World operations.  Catalog ownership, revisioning, and idempotency are
# validated by ``world.apply_catalog_mutation``; inventory delivery remains a
# separate World write.  Merchants never receive direct-to-World privilege.
PLATFORM_CATALOG_ACTIONS = frozenset({
    "commerce.get_sku",
    "commerce.adjust_price",
    "commerce.update_listing",
    "commerce.receive_shipment",
})
PLATFORM_ORDER_LIFECYCLE_ACTIONS = frozenset({
    "commerce.dispatch",
    "commerce.cancel_order",
    "commerce.mark_returned",
})
PLATFORM_EXCHANGE_ACTIONS = frozenset({"commerce.request_exchange"})
PLATFORM_AFTER_SALES_ACTIONS = frozenset({
    "commerce.cancel_paid_order",
    "commerce.request_return",
    "commerce.authorize_return",
    "commerce.deny_return",
    "commerce.receive_return",
    "commerce.open_refund_case",
    "commerce.approve_refund",
    "commerce.deny_refund",
    "commerce.request_exchange",
    "commerce.authorize_exchange",
    "commerce.deny_exchange",
    "commerce.complete_exchange",
    "commerce.open_dispute",
    "commerce.submit_dispute_evidence",
    "commerce.respond_to_dispute",
    "commerce.request_ledger_reconciliation",
})
PLATFORM_AFTER_SALES_READ_ACTIONS = frozenset({
    "commerce.read_payment_history",
    "commerce.read_ledger_history",
    "commerce.read_packing_history",
    "commerce.read_after_sales_history",
    "commerce.read_after_sales_policy",
})
PLATFORM_GOVERNANCE_ACTOR_ACTIONS = frozenset({
    "commerce.publish_campaign",
    "commerce.disclose_placement",
    "commerce.activate_campaign",
    "commerce.submit_review",
    "commerce.reject_review_manipulation",
    "commerce.reject_coordination",
    "commerce.accept_remediation_plan",
    "commerce.complete_remediation_step",
    "commerce.read_governance_history",
})
PLATFORM_GOVERNANCE_TRUSTED_ACTIONS = frozenset({
    "platform.publish_governance_policy",
    "platform.aggregate_reviews",
    "platform.ingest_review_observation",
    "platform.ingest_market_observation",
    "platform.resolve_governance_case",
    "platform.apply_governance_reputation",
    "platform.create_remediation_plan",
    "platform.verify_remediation_step",
    "platform.persist_ranking_context",
})
_AFTER_SALES_ACK_REFERENCE_FIELDS = frozenset({
    "cancellation_id",
    "request_id",
    "authorization_id",
    "receipt_id",
    "case_id",
    "decision_id",
    "dispute_id",
    "evidence_id",
    "response_id",
    "ruling_id",
    "source_id",
    "result_id",
})
PLATFORM_ADJUDICATOR_ACTIONS = frozenset({
    "open_dispute",  # legacy wire spelling
    "platform.open_dispute",
    "platform.rule_dispute",
})
PLATFORM_CART_ACTIONS = frozenset({
    "commerce.create_cart_quote_request",
    "commerce.request_cart_quote",
    "platform.checkout_cart",
})
PLATFORM_SUPPLY_ACTIONS = frozenset({
    "commerce.read_supply_state",
    "commerce.update_supply",
    "platform.apply_supply_event",
})
PLATFORM_FULFILLMENT_ACTIONS = frozenset({
    "commerce.allocate_fulfillment",
    "commerce.read_shipment",
    "commerce.resolve_shipment",
    "platform.record_shipment_status",
})
PLATFORM_EVENT_ACTIONS = frozenset({
    "platform.issue_protocol_event",
    "platform.advance_market_clock",
    "commerce.acknowledge_protocol_event",
    "commerce.reject_protocol_event",
    "commerce.process_protocol_event",
})

_AFTER_SALES_ACTOR_AUTHORITY_MESSAGES: dict[str, frozenset[str]] = {
    "actor cannot cancel this paid order": frozenset(
        {"commerce.cancel_paid_order"}
    ),
    "only the order owner may request return": frozenset(
        {"commerce.request_return"}
    ),
    "actor cannot authorize returns": frozenset(
        {"commerce.authorize_return", "commerce.deny_return"}
    ),
    "actor cannot receive returns": frozenset({"commerce.receive_return"}),
    "actor cannot open refund case": frozenset(
        {"commerce.open_refund_case"}
    ),
    "actor cannot decide refunds": frozenset(
        {"commerce.approve_refund", "commerce.deny_refund"}
    ),
    "only the order owner may request exchange": frozenset(
        {"commerce.request_exchange"}
    ),
    "actor cannot decide or complete exchange": frozenset(
        {
            "commerce.authorize_exchange",
            "commerce.deny_exchange",
            "commerce.complete_exchange",
        }
    ),
    "only an order party may open dispute": frozenset(
        {"commerce.open_dispute"}
    ),
    "actor cannot submit dispute evidence": frozenset(
        {"commerce.submit_dispute_evidence"}
    ),
    "only the opposing party may file dispute response": frozenset(
        {"commerce.respond_to_dispute"}
    ),
    "actor cannot request ledger reconciliation": frozenset(
        {"commerce.request_ledger_reconciliation"}
    ),
}

_CART_ACTOR_AUTHORITY_REASONS: dict[tuple[str, str], str] = {
    (
        "commerce.create_cart_quote_request",
        "cart quote request mandate is missing",
    ): "cart_request_mandate_missing",
    (
        "commerce.create_cart_quote_request",
        "cart quote request actor is not the mandate buyer or principal",
    ): "cart_request_actor_mismatch",
    (
        "commerce.create_cart_quote_request",
        "cart quote request mandate has no revision",
    ): "cart_request_mandate_missing",
    (
        "commerce.create_cart_quote_request",
        "cart quote request mandate authority is inconsistent",
    ): "cart_request_mandate_inconsistent",
    (
        "commerce.request_cart_quote",
        "cart quote request is unknown or unauthorized",
    ): "cart_request_not_authorized",
    (
        "commerce.request_cart_quote",
        "cart quote request mandate is missing",
    ): "cart_request_mandate_missing",
    (
        "commerce.request_cart_quote",
        "merchant is not authorized by request",
    ): "cart_request_not_authorized",
    (
        "commerce.request_cart_quote",
        "merchant quote requests must contain only that merchant's listings",
    ): "cart_request_merchant_mismatch",
    (
        "commerce.request_cart_quote",
        "cart quote mandate authority is missing",
    ): "cart_quote_mandate_missing",
    (
        "commerce.request_cart_quote",
        "cart quote actor does not own the persisted mandate",
    ): "cart_quote_actor_mismatch",
    (
        "commerce.request_cart_quote",
        "cart quote mandate has no revision",
    ): "cart_quote_mandate_missing",
    (
        "commerce.request_cart_quote",
        "cart quote mandate authority is inconsistent",
    ): "cart_quote_mandate_inconsistent",
    (
        "commerce.request_cart_quote",
        "authoritative cart quote exceeds the principal mandate budget",
    ): "cart_quote_exceeds_mandate",
    (
        "platform.checkout_cart",
        "cart checkout must be authenticated by the quoted buyer",
    ): "cart_checkout_buyer_mismatch",
    (
        "platform.checkout_cart",
        "cart checkout mandate authority is missing",
    ): "cart_checkout_mandate_missing",
    (
        "platform.checkout_cart",
        "cart checkout exceeds the current principal mandate budget",
    ): "cart_checkout_exceeds_mandate",
}

_EVIDENCE_ACTOR_AUTHORITY_REASONS: dict[tuple[str, str, str], str] = {
    (
        "platform:evidence",
        "commerce.publish_evidence_record",
        "evidence record issuer must match the authenticated original actor",
    ): "evidence_actor_mismatch",
    (
        "platform:mandate",
        "delegate.register_mandate_authority",
        "mandate authority principal must match authenticated original actor",
    ): "mandate_actor_mismatch",
    (
        "platform:mandate",
        "delegate.append_mandate_revision",
        "mandate revision principal must match authenticated original actor",
    ): "mandate_actor_mismatch",
    (
        "platform:claims",
        "commerce.apply_listing_claim",
        "authenticated merchant does not own the claimed listing",
    ): "listing_claim_owner_mismatch",
    (
        "platform:claims",
        "commerce.apply_listing_claim",
        "listing claim merchant does not own listing",
    ): "listing_claim_owner_mismatch",
    (
        "platform:claims",
        "commerce.apply_listing_claim",
        "listing claim issuer must match authenticated owning merchant",
    ): "listing_claim_owner_mismatch",
    (
        "platform:claims",
        "commerce.apply_listing_claim",
        "claim issuer must be the owning merchant",
    ): "listing_claim_owner_mismatch",
    (
        "platform:claims",
        "commerce.apply_listing_claim",
        "only the owning merchant can modify a listing claim",
    ): "listing_claim_owner_mismatch",
    (
        "platform:claims",
        "commerce.apply_listing_claim",
        "claim intent changes immutable claim identity",
    ): "listing_claim_identity_mismatch",
}

_PROTOCOL_EVENT_ACTOR_REASONS: dict[str, str] = {
    "protocol event does not exist": "protocol_event_not_found",
    "only the bound protocol event recipient may decide it": (
        "protocol_event_recipient_mismatch"
    ),
    "protocol operation actor is not the event recipient": (
        "protocol_event_recipient_mismatch"
    ),
}

_UNHANDLED = object()


def _actor_action_rejection(
    env: Envelope,
    exc: Exception,
) -> CommerceActionRejected | None:
    """Normalize only known actor-caused commerce refusals.

    The same exception type can mean a corrupt internal command when it is
    raised for a Runtime or Platform service request.  Route, action kind, and
    full actor role therefore participate in the allow-list.  This deliberately
    does not catch broad ``WorldError``, ``SchemaError``, authorization errors,
    privacy errors, or transport failures.
    """

    kind = str(env.action.get("kind", ""))
    role = env.from_.split(":", 1)[0]
    if role not in {"buyer", "consumer", "merchant"}:
        return None

    reason: str | None = None
    category: RejectionCategory = "state"
    source_error_type = type(exc).__name__

    if (
        env.to == "platform:psp"
        and kind in PLATFORM_PSP_ACTIONS
        and isinstance(
            exc,
            (
                InsufficientFunds,
                FulfillmentNotActionable,
                NoMerchantConsent,
                OrderIdentityMismatch,
                OrderNotSettleable,
                OutOfStock,
                IdempotencyConflict,
            ),
        )
    ):
        reason = _rejection_reason(
            exc,
            (
                (InsufficientFunds, "insufficient_funds"),
                (FulfillmentNotActionable, "fulfillment_not_actionable"),
                (NoMerchantConsent, "merchant_consent_missing"),
                (OrderIdentityMismatch, "order_identity_mismatch"),
                (OrderNotSettleable, "order_not_settleable"),
                (OutOfStock, "out_of_stock"),
                (IdempotencyConflict, "settlement_conflict"),
            ),
        )
        category = (
            "identity"
            if isinstance(exc, OrderIdentityMismatch)
            else "conflict"
            if isinstance(exc, IdempotencyConflict)
            else "business"
        )
    elif (
        env.to in {"platform:after-sales", "platform:psp"}
        and kind == "commerce.request_return"
        and isinstance(exc, (OrderNotRefundable, ReturnWindowClosed))
    ):
        if isinstance(exc, ReturnWindowClosed):
            reason = "return_window_closed"
            category = "stale"
        else:
            reason = "order_not_refundable"
    elif (
        env.to == "platform:psp"
        and kind in PLATFORM_ORDER_LIFECYCLE_ACTIONS
        and isinstance(exc, InvalidOrderTransition)
    ):
        reason = "invalid_order_transition"
    elif (
        env.to == "platform:adjudicator"
        and kind in PLATFORM_ADJUDICATOR_ACTIONS
        and isinstance(exc, (DisputeNotActionable, IdempotencyConflict))
    ):
        reason = (
            "dispute_not_actionable"
            if isinstance(exc, DisputeNotActionable)
            else "dispute_conflict"
        )
        category = "conflict" if isinstance(exc, IdempotencyConflict) else "state"
    elif (
        env.to == "platform:aggregator"
        and kind == "commerce.accept_offer"
        and isinstance(exc, (MatchAcceptanceRejected, IdempotencyConflict))
    ):
        reason = (
            "match_acceptance_rejected"
            if isinstance(exc, MatchAcceptanceRejected)
            else "match_acceptance_conflict"
        )
        category = (
            "conflict" if isinstance(exc, IdempotencyConflict) else "stale"
        )
    elif (
        env.to == "platform:aggregator"
        and kind == "commerce.search"
        and isinstance(exc, _AggregatorPolicyError)
    ):
        reason = exc.reason_code
        category = exc.category
    elif (
        env.to == "platform:aggregator"
        and kind == "commerce.search"
        and isinstance(exc, IdempotencyConflict)
    ):
        reason = "search_operation_conflict"
        category = "conflict"
    elif (
        env.to == "platform:catalog"
        and kind in PLATFORM_CATALOG_ACTIONS
        and isinstance(
            exc,
            (CatalogMutationRejected, IdempotencyConflict, OutOfStock),
        )
    ):
        reason = _rejection_reason(
            exc,
            (
                (CatalogMutationRejected, "catalog_mutation_rejected"),
                (IdempotencyConflict, "catalog_mutation_conflict"),
                (OutOfStock, "catalog_inventory_unavailable"),
            ),
        )
        category = "conflict" if isinstance(exc, IdempotencyConflict) else "state"
    elif (
        env.to == "platform:checkout"
        and kind in PLATFORM_CART_ACTIONS
        and isinstance(
            exc,
            (
                CartQuoteRequestStaleError,
                CartQuoteStaleError,
                IdempotencyConflict,
                OrderNotSettleable,
                OutOfStock,
            ),
        )
    ):
        reason = _rejection_reason(
            exc,
            (
                (CartQuoteRequestStaleError, "cart_quote_request_stale"),
                (CartQuoteStaleError, "cart_quote_stale"),
                (OrderNotSettleable, "cart_not_settleable"),
                (OutOfStock, "cart_inventory_unavailable"),
                (IdempotencyConflict, "cart_operation_conflict"),
            ),
        )
        category = (
            "stale"
            if isinstance(
                exc, (CartQuoteRequestStaleError, CartQuoteStaleError)
            )
            else "conflict"
            if isinstance(exc, IdempotencyConflict)
            else "business"
        )
    elif (
        env.to == "platform:supply"
        and kind in PLATFORM_SUPPLY_ACTIONS
        and isinstance(exc, (OutOfStock, IdempotencyConflict))
    ):
        reason = (
            "supply_unavailable"
            if isinstance(exc, OutOfStock)
            else "supply_operation_conflict"
        )
        category = "conflict" if isinstance(exc, IdempotencyConflict) else "state"
    elif (
        env.to == "platform:fulfillment"
        and kind in PLATFORM_FULFILLMENT_ACTIONS
        and isinstance(
            exc,
            (
                FulfillmentNotActionable,
                IdempotencyConflict,
                OutOfStock,
                ShipmentNotActionable,
            ),
        )
    ):
        reason = _rejection_reason(
            exc,
            (
                (FulfillmentNotActionable, "fulfillment_not_actionable"),
                (ShipmentNotActionable, "shipment_not_actionable"),
                (OutOfStock, "out_of_stock"),
                (IdempotencyConflict, "fulfillment_conflict"),
            ),
        )
        category = "conflict" if isinstance(exc, IdempotencyConflict) else "state"
    elif (
        env.to in {"platform:after-sales", "platform:psp"}
        and kind in PLATFORM_AFTER_SALES_ACTIONS
        and isinstance(
            exc,
            (
                AfterSalesCoreTransitionError,
                AfterSalesIntentError,
                DisputeNotActionable,
                ExchangeNotActionable,
                IdempotencyConflict,
                OutOfStock,
            ),
        )
    ):
        reason = _rejection_reason(
            exc,
            (
                (AfterSalesIntentError, "invalid_after_sales_intent"),
                (AfterSalesCoreTransitionError, "after_sales_transition_rejected"),
                (DisputeNotActionable, "dispute_not_actionable"),
                (ExchangeNotActionable, "exchange_not_actionable"),
                (OutOfStock, "out_of_stock"),
                (IdempotencyConflict, "after_sales_conflict"),
            ),
        )
        category = (
            "conflict" if isinstance(exc, IdempotencyConflict) else "state"
        )
    elif (
        kind in PLATFORM_GOVERNANCE_ACTOR_ACTIONS
        and isinstance(
            exc,
            (GovernanceIntentError, GovernanceTransitionError, IdempotencyConflict),
        )
    ):
        reason = _rejection_reason(
            exc,
            (
                (GovernanceIntentError, "invalid_governance_intent"),
                (GovernanceTransitionError, "governance_transition_rejected"),
                (IdempotencyConflict, "governance_operation_conflict"),
            ),
        )
        category = (
            "conflict" if isinstance(exc, IdempotencyConflict) else "state"
        )
    elif (
        env.to == "platform:events"
        and kind in PLATFORM_EVENT_ACTIONS
        and isinstance(
            exc,
            (
                IdempotencyConflict,
                ProtocolEventAuthorityError,
                ProtocolEventStaleError,
            ),
        )
    ):
        if isinstance(exc, ProtocolEventStaleError):
            reason = "protocol_event_stale"
            category = "stale"
        elif isinstance(exc, ProtocolEventAuthorityError):
            reason = "protocol_event_recipient_role_mismatch"
            category = "identity"
        else:
            reason = "protocol_event_conflict"
            category = "conflict"
    elif (
        env.to in {"platform:after-sales", "platform:psp"}
        and kind in PLATFORM_AFTER_SALES_ACTIONS
        and isinstance(exc, AfterSalesReferenceRejected)
    ):
        reason = exc.reason_code
        category = (
            "stale"
            if reason.endswith("_stale")
            else "identity"
            if reason.endswith(("_not_authorized", "_identity_mismatch"))
            else "state"
        )
    elif (
        env.to == "platform:claims"
        and kind == "commerce.apply_listing_claim"
        and isinstance(exc, ClaimStateRejected)
    ):
        reason = exc.reason_code
        category = "state"

    if reason is None:
        authority_rejection = _actor_state_authority_rejection(env, exc)
        if authority_rejection is not None:
            reason, category = authority_rejection
            # The in-process World preserves WriteNotAuthorized while the HTTP
            # World seam intentionally restores the same 403 as
            # LifecycleAuthorizationError.  Journal a topology-independent
            # source label for this exact, actor-owned refusal.
            source_error_type = "ActorAuthorityRejected"

    if reason is None:
        return None
    return CommerceActionRejected(
        reason,
        category=category,
        source_error_type=source_error_type,
    )


def _rejection_reason(
    exc: Exception,
    options: tuple[tuple[type[Exception], str], ...],
) -> str:
    """Return the first subtype-aware stable reason for a narrow allow-list."""

    for error_type, reason_code in options:
        if isinstance(exc, error_type):
            return reason_code
    raise AssertionError("actor rejection allow-list and reason map diverged")


def _actor_state_authority_rejection(
    env: Envelope,
    exc: Exception,
) -> tuple[str, RejectionCategory] | None:
    """Classify only exact actor-owned state/identity refusals.

    These failures depend on authoritative World state and therefore cannot be
    decided by the side-effect-free terminal action contract.  The allow-list
    includes route, action, exception family, and an exact stable message (or
    the evidence ACL's fixed prefix/suffix).  An unknown authorization or
    schema error still propagates as an infrastructure defect.  In particular,
    this function never treats a broad ``SchemaError`` or
    ``WriteNotAuthorized`` as a normal model outcome.
    """

    kind = str(env.action.get("kind", ""))
    message = str(exc)
    authorization_error = isinstance(
        exc, (LifecycleAuthorizationError, WriteNotAuthorized)
    )

    if (
        authorization_error
        and env.to in {"platform:after-sales", "platform:psp"}
        and kind
        in _AFTER_SALES_ACTOR_AUTHORITY_MESSAGES.get(message, frozenset())
    ):
        return "after_sales_order_authority_rejected", "identity"

    if env.to == "platform:checkout" and authorization_error:
        reason = _CART_ACTOR_AUTHORITY_REASONS.get((kind, message))
        if reason is not None:
            category: RejectionCategory = (
                "business" if reason.endswith("exceeds_mandate") else "identity"
            )
            return reason, category

    evidence_reason = _EVIDENCE_ACTOR_AUTHORITY_REASONS.get(
        (env.to, kind, message)
    )
    if evidence_reason is not None and isinstance(
        exc, (LifecycleAuthorizationError, WriteNotAuthorized, SchemaError)
    ):
        return evidence_reason, "identity"

    if (
        env.to == "platform:claims"
        and kind == "commerce.apply_listing_claim"
        and isinstance(exc, (LifecycleAuthorizationError, SchemaError))
        and message.startswith("reader ")
        and message.endswith(" is not authorized for evidence")
    ):
        return "listing_claim_evidence_not_authorized", "identity"

    if (
        env.to == "platform:events"
        and kind
        in {
            "commerce.acknowledge_protocol_event",
            "commerce.reject_protocol_event",
            "commerce.process_protocol_event",
        }
        and isinstance(
            exc, (LifecycleAuthorizationError, WriteNotAuthorized, SchemaError)
        )
    ):
        reason = _PROTOCOL_EVENT_ACTOR_REASONS.get(message)
        if reason is not None:
            return reason, "identity"

    return None


class PlatformService:
    """Minimal in-process Platform Service.

    This is the current bridge between the RFD service shape and the existing
    scaffold. It owns marketplace decisions and commits allowed facts to
    ``World`` through action-gated writes.

    ``audit`` is optional but recommended. When supplied, ``PSPPolicy.settle``
    uses it to enforce the **merchant-consent invariant**: a settle at a
    price below the listing's list_price requires the audit log to contain
    a ``commerce.accept_offer`` or ``commerce.counter_offer`` envelope FROM
    the named merchant at the same sku + price. Constructing PlatformService
    without ``audit`` puts PSP in lenient mode (no consent check) — useful
    for unit tests that don't care, but unsafe for any agent-driven run.
    """

    def __init__(
        self,
        *,
        world_client: "WorldClient | None" = None,
        world: "World | None" = None,
        audit: "AuditLog | None" = None,
        decision_journal: "PlatformDecisionJournal | None" = None,
        platform_policy: "dict[str, Any] | None" = None,
        max_search_results: int | None = None,
        max_transactions_per_buyer: int | None = None,
        supply_authority_ttl_ticks: int = (
            DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS
        ),
        secrets: "SecretRegistry | None" = None,
        negotiation_participants: frozenset[str] | None = None,
        negotiation_max_rounds: int = 8,
        negotiation_deadline_ticks: int = 32,
        after_sales_adjudication_policy: (
            Callable[[Mapping[str, Any], str], tuple[str, str]] | None
        ) = None,
    ) -> None:
        # The platform reaches World ONLY through a VCP WorldClient — it never
        # holds a World object or calls it directly. ``world=`` is the in-process
        # convenience: it builds a client whose transport is a local
        # WorldService.handle — the SAME world.* service surface, no HTTP, no
        # shared World handle, no shared lock. The 4-process launcher passes
        # ``world_client=`` (HTTP) explicitly.
        if world_client is None:
            if world is None:
                raise ValueError("PlatformService needs world_client (or world for in-process)")
            from agents.world_client import in_process_world_client
            world_client = in_process_world_client(world)
        ranking_policy = None
        ranking_governance: dict[str, Any] | None = None
        if platform_policy:
            name = str(platform_policy.get("name", "")).strip()
            config = platform_policy.get("config", {})
            if not name:
                raise ValueError("platform_policy requires a registered 'name'")
            if not isinstance(config, dict):
                raise ValueError("platform_policy.config must be a mapping")
            from episode.extensions import PLATFORM_POLICIES

            ranking_policy = PLATFORM_POLICIES.get(name)(dict(config))
            if not callable(ranking_policy):
                raise TypeError(f"platform policy factory {name!r} did not return a callable")
            raw_governance = platform_policy.get("governance")
            if raw_governance is not None:
                if not isinstance(raw_governance, dict) or set(raw_governance) != {
                    "policy_id",
                    "policy_version",
                }:
                    raise ValueError(
                        "platform_policy.governance requires exactly policy_id "
                        "and policy_version"
                    )
                policy_id = raw_governance.get("policy_id")
                policy_version = raw_governance.get("policy_version")
                if not isinstance(policy_id, str) or not policy_id.strip():
                    raise ValueError("governance ranking policy_id must be non-empty")
                if (
                    isinstance(policy_version, bool)
                    or not isinstance(policy_version, int)
                    or policy_version <= 0
                ):
                    raise ValueError(
                        "governance ranking policy_version must be positive"
                    )
                ranking_governance = {
                    "policy_id": policy_id,
                    "policy_version": policy_version,
                }
        if max_search_results is not None and max_search_results <= 0:
            raise ValueError("max_search_results must be positive")
        if max_transactions_per_buyer is not None and max_transactions_per_buyer <= 0:
            raise ValueError("max_transactions_per_buyer must be positive")
        try:
            validate_supply_purchase_authority_ttl_ticks(
                supply_authority_ttl_ticks
            )
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc
        if decision_journal is None and audit is not None:
            from runtime.evidence import PlatformDecisionJournal

            decision_journal = PlatformDecisionJournal(
                audit.path.with_name("platform.decisions.jsonl"),
                run_id=audit.run_id,
            )
        self._decision_journal = decision_journal
        self._audit = audit
        self._world_client = world_client
        self._psp = PSPPolicy(
            world_client=world_client,
            audit=audit,
            max_transactions_per_buyer=max_transactions_per_buyer,
        )
        self._reputation = ReputationPolicy(world_client=world_client)
        self._aggregator = AggregatorPolicy(
            world_client=world_client,
            ranking_policy=ranking_policy,
            governance_policy=ranking_governance,
            max_results=max_search_results,
        )
        self._catalog = CatalogPolicy(world_client=world_client)
        self._cart = CartPolicy(world_client=world_client)
        self._supply = SupplyPolicy(
            world_client=world_client,
            authority_ttl_ticks=supply_authority_ttl_ticks,
        )
        self._fulfillment = FulfillmentPolicy(world_client=world_client)
        self._events = ProtocolEventPolicy(world_client=world_client)
        self._evidence_claims = EvidenceClaimsPolicy(world_client=world_client)
        self._negotiation = NegotiationPolicy(
            world_client=world_client,
            secrets=secrets,
            participant_ids=negotiation_participants,
            max_rounds=negotiation_max_rounds,
            deadline_ticks=negotiation_deadline_ticks,
        )
        self._lifecycle = OrderLifecyclePolicy(world_client=world_client)
        self._after_sales = AfterSalesPolicy(
            world_client=world_client,
            adjudication_policy=after_sales_adjudication_policy,
        )
        self._governance = GovernancePolicy(world_client=world_client)
        self._adjudicator = AdjudicatorPolicy(
            world_client=world_client,
            audit=audit,
        )

    def handle(self, env: Envelope) -> Envelope | list[Envelope] | None:
        """Dispatch and append exactly one Platform validation decision."""

        try:
            result = self._dispatch(env)
        except NegotiationRejected as exc:
            # A typed negotiation action can be recognized yet rejected by a
            # normal market rule, for example when it exceeds the configured
            # round limit.  The authoritative outcome is the rejected Platform
            # decision.  It is evidence about the actor's attempted action,
            # not an infrastructure failure, and therefore must not abort the
            # Runtime after the decision has already been journaled.
            self._record_decision(
                env,
                decision="rejected",
                reason_code=exc.reason_code,
                error_type=type(exc).__name__,
            )
            return None
        except Exception as exc:
            rejection = _actor_action_rejection(env, exc)
            if rejection is not None:
                self._record_decision(
                    env,
                    decision="rejected",
                    reason_code=rejection.reason_code,
                    error_type=rejection.source_error_type,
                )
                return None
            if isinstance(exc, PrivateUtilityLeak):
                self._record_platform_security_event(env, exc)
            self._record_decision(
                env,
                decision="rejected",
                reason_code=str(
                    getattr(exc, "reason_code", "platform_exception_rejected")
                ),
                error_type=type(exc).__name__,
            )
            raise
        if result is _UNHANDLED:
            self._record_decision(
                env,
                decision="rejected",
                reason_code="unsupported_platform_action",
            )
            return None
        handled = cast("Envelope | list[Envelope] | None", result)
        if _is_policy_decline(handled):
            self._record_decision(
                env,
                decision="rejected",
                reason_code="platform_policy_declined",
                response=handled,
            )
        else:
            self._record_decision(
                env,
                decision="accepted",
                reason_code="validated_and_executed",
                response=handled,
            )
        return handled

    def deliver_protocol_event(
        self,
        *,
        market_id: str,
        stream_id: str,
        order_id: OrderId | str,
        recipient_id: str,
        event_id: str,
        event_kind: str,
        ttl_ticks: int,
        idempotency_key: str,
        reference_kind: Literal["operation", "certificate"] = "operation",
        reference_digest: str | None = None,
        reference_authorization_id: str | None = None,
        ts: str = "1970-01-01T00:00:00Z",
        in_reply_to: str | None = None,
    ) -> Envelope:
        """Issue and persist one authority event, then build its delivery.

        Scenario code may configure this call, but it cannot provide order
        parties, state, revision, logical tick, or an operation-state digest.
        Those fields are derived inside Platform from authoritative World VCP.
        """

        return self._events.deliver(
            market_id=market_id,
            stream_id=stream_id,
            order_id=order_id,
            recipient_id=recipient_id,
            event_id=event_id,
            event_kind=event_kind,
            ttl_ticks=ttl_ticks,
            idempotency_key=idempotency_key,
            reference_kind=reference_kind,
            reference_digest=reference_digest,
            reference_authorization_id=reference_authorization_id,
            ts=ts,
            in_reply_to=in_reply_to,
        )

    def _dispatch(self, env: Envelope) -> Envelope | list[Envelope] | object | None:
        """Return the policy result or the private ``_UNHANDLED`` sentinel."""

        kind = str(env.action.get("kind", ""))
        if env.to == "platform:psp" and kind in PLATFORM_PSP_ACTIONS:
            receipt = self._psp.settle(env)
            receipt_payload = _payload_mapping(receipt)
            order_id = str(receipt_payload.get("order_id", ""))
            settled_order = self._world_client.read(
                "orders",
                OrderId(order_id),
                caller="platform:psp",
            )
            if not isinstance(settled_order, Order):
                raise RuntimeError(
                    "PSP settlement receipt is not backed by an authoritative order"
                )
            reputation = self._reputation.successful_settlement_update_request(
                merchant_id=str(settled_order.merchant_id),
                order_id=order_id,
                txn_id=str(receipt_payload.get("txn_id", "")),
                ts=env.ts,
                correlation_id=env.msg_id,
                idempotency_key=env.idempotency_key,
            )
            return receipt if reputation is None else [receipt, reputation]
        if env.to == "platform:psp" and kind == "commerce.request_return":
            # Timing and other authority fields must fail before either the
            # legacy adapter or the modern exact-shape lifecycle path runs.
            _reject_time_claims(_payload_mapping(env))
            if self._is_legacy_psp_return(env, kind=kind):
                return self._legacy_psp_return(env)
        if env.to in {"platform:after-sales", "platform:psp"} and kind in (
            PLATFORM_AFTER_SALES_ACTIONS | PLATFORM_AFTER_SALES_READ_ACTIONS
        ):
            return self._after_sales.handle(env)
        if kind in (
            PLATFORM_GOVERNANCE_ACTOR_ACTIONS
            | PLATFORM_GOVERNANCE_TRUSTED_ACTIONS
        ):
            return self._governance.handle(env)
        if env.to == "platform:psp" and kind in PLATFORM_ORDER_LIFECYCLE_ACTIONS:
            return self._lifecycle.handle(env)
        if env.to == "platform:adjudicator" and kind in PLATFORM_ADJUDICATOR_ACTIONS:
            return self._adjudicator.handle(env)
        if env.to == "platform:reputation" and kind in PLATFORM_REPUTATION_ACTIONS:
            return self._reputation.update(env)
        if env.to == "platform:aggregator":
            if kind == "commerce.search":
                return self._aggregator.search(env)
            if kind == "commerce.accept_offer":
                return self._aggregator.certify(env)
        if env.to == "platform:catalog" and kind in PLATFORM_CATALOG_ACTIONS:
            return self._catalog.handle(env)
        if env.to == "platform:checkout" and kind in PLATFORM_CART_ACTIONS:
            if kind == "commerce.create_cart_quote_request":
                return self._cart.authorize(env)
            if kind == "commerce.request_cart_quote":
                return self._cart.quote(env)
            return self._cart.checkout(env)
        if env.to == "platform:supply" and kind in PLATFORM_SUPPLY_ACTIONS:
            return self._supply.handle(env)
        if (
            env.to == "platform:fulfillment"
            and kind in PLATFORM_FULFILLMENT_ACTIONS
        ):
            return self._fulfillment.handle(env)
        if env.to == "platform:negotiation" and kind in NEGOTIATION_ACTIONS:
            return self._negotiation.mediate(env)
        if env.to == "platform:events" and kind in PLATFORM_EVENT_ACTIONS:
            return self._events.handle(env)
        if kind in EvidenceClaimsPolicy.HANDLES and env.to in {
            "platform:evidence",
            "platform:mandate",
            "platform:claims",
        }:
            return self._evidence_claims.handle(env)
        return _UNHANDLED

    @staticmethod
    def _is_legacy_psp_return(env: Envelope, *, kind: str) -> bool:
        """Recognize only the two historical immediate-refund wire shapes."""

        if env.to != "platform:psp" or kind != "commerce.request_return":
            return False
        payload = env.action.get("payload")
        return isinstance(payload, Mapping) and set(payload) in (
            {"order_id"},
            {"order_id", "reason"},
        )

    def _legacy_psp_return(self, env: Envelope) -> Envelope:
        """Adapt a historical return request through the exact intent schema.

        The compatibility surface remains immediate atomic refund, but it does
        not invent quantity or authority.  Platform derives the paid quantity
        from authoritative World allocation state, fills only stable defaults,
        and requires the same exact request-return normalizer used by the full
        after-sales lifecycle before calling the existing World refund
        transaction.  Modern complete intents continue through
        :class:`AfterSalesPolicy` unchanged.
        """

        payload = _payload_mapping(env)
        order_id = str(payload.get("order_id", ""))
        order = self._world_client.read(
            "orders", OrderId(order_id), caller="platform:psp"
        )
        if order is None:
            raise OrderNotRefundable(f"refund for unknown order {order_id!r}")
        allocation = self._world_client.read(
            "fulfillments", order.order_id, caller="platform:psp"
        )
        requested_qty = (
            int(allocation.fulfilled_qty)
            if allocation is not None
            else int(order.qty)
        )
        if requested_qty <= 0:
            raise OrderNotRefundable(
                f"order {order.order_id} has no fulfilled units to refund"
            )
        normalized = normalize_after_sales_intent(
            {
                "op": "request_return",
                "order_id": str(order.order_id),
                "requested_qty": requested_qty,
                "reason": str(payload.get("reason") or "legacy_return_request"),
                "evidence_ids": (),
            }
        )
        adapted_payload = {
            key: value for key, value in normalized.items() if key != "op"
        }
        adapted = Envelope(
            msg_id=env.msg_id,
            ts=env.ts,
            from_=env.from_,
            to=env.to,
            in_reply_to=env.in_reply_to,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "commerce.request_return",
                "payload": adapted_payload,
            },
            signature=env.signature,
        )
        return self._psp.refund(adapted)

    def _record_decision(
        self,
        env: Envelope,
        *,
        decision: str,
        reason_code: str,
        response: Envelope | list[Envelope] | None = None,
        error_type: str | None = None,
    ) -> None:
        if self._decision_journal is None:
            return
        if isinstance(response, list):
            responses = response
        elif response is None:
            responses = []
        else:
            responses = [response]
        response_kinds = tuple(
            str(item.action.get("kind", ""))
            for item in responses
            if item is not None and isinstance(item.action, dict)
        )
        response_msg_ids = tuple(item.msg_id for item in responses if item is not None)
        response_sha256s = tuple(
            hashlib.sha256(to_json(item).encode("utf-8")).hexdigest()
            for item in responses
            if item is not None
        )
        self._decision_journal.append(
            request=env,
            decision=decision,
            reason_code=reason_code,
            response_kinds=response_kinds,
            response_msg_ids=response_msg_ids,
            response_sha256s=response_sha256s,
            decision_metadata=mediation_metadata(response),
            error_type=error_type,
        )

    def _record_platform_security_event(
        self,
        env: Envelope,
        leak: PrivateUtilityLeak,
    ) -> None:
        append = getattr(self._audit, "append_security", None)
        if not callable(append):
            return
        from runtime.types import SecurityEvent

        finding = getattr(leak, "finding", None)
        append(
            SecurityEvent(
                msg_id=env.msg_id,
                sender_id=env.from_,
                action_kind=str(env.action.get("kind", "")),
                secret_owner=str(getattr(finding, "secret_owner", "") or ""),
                secret_name=str(getattr(finding, "secret_name", "") or ""),
                field_path=str(getattr(finding, "field_path", "") or ""),
                reason=str(getattr(finding, "reason", "") or "private_utility_leak"),
                recipient_id=str(
                    (env.action.get("payload") or {}).get("counterparty_id", env.to)
                ),
                owner_role=str(getattr(finding, "owner_role", "") or ""),
                ts=env.ts,
            )
        )


def _is_policy_decline(result: Envelope | list[Envelope] | None) -> bool:
    """Recognize the Platform's structured policy-decline response shape."""

    if isinstance(result, list):
        values = result
    elif result is None:
        values = []
    else:
        values = [result]
    for value in values:
        if value is None or not isinstance(value.action, dict):
            continue
        payload = value.action.get("payload")
        if not isinstance(payload, dict):
            continue
        if (
            value.action.get("kind") == "platform.catalog_listing"
            and payload.get("status") == "declined"
        ):
            return True
        answer = payload.get("answer")
        if isinstance(answer, dict) and answer.get("decision") == "declined":
            return True
    return False


class ProtocolEventPolicy:
    """Authority issuance and recipient decisions for durable protocol events."""

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def deliver(
        self,
        *,
        market_id: str,
        stream_id: str,
        order_id: OrderId | str,
        recipient_id: str,
        event_id: str,
        event_kind: str,
        ttl_ticks: int,
        idempotency_key: str,
        reference_kind: Literal["operation", "certificate"] = "operation",
        reference_digest: str | None = None,
        reference_authorization_id: str | None = None,
        ts: str,
        in_reply_to: str | None,
    ) -> Envelope:
        if isinstance(ttl_ticks, bool) or not isinstance(ttl_ticks, int) or ttl_ticks <= 0:
            raise SchemaError("protocol event ttl_ticks must be a positive integer")
        for name, value in (
            ("market_id", market_id),
            ("stream_id", stream_id),
            ("recipient_id", recipient_id),
            ("event_id", event_id),
            ("event_kind", event_kind),
            ("idempotency_key", idempotency_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"protocol event {name} must be non-empty")
        state = self._world_client.read_order_protocol_state(
            order_id,
            caller="platform:events",
        )
        if state is None:
            raise SchemaError(f"protocol event order {order_id!s} is not visible")
        buyer_id = str(state["buyer_id"])
        merchant_id = str(state["merchant_id"])
        if recipient_id not in {buyer_id, merchant_id}:
            raise SchemaError("protocol event recipient is not an order party")
        binding = build_protocol_event_binding(
            market_id=market_id,
            stream_id=stream_id,
            order_id=str(state["order_id"]),
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            recipient_id=recipient_id,
            authority_id="platform:events",
        )
        if reference_kind == "operation":
            if reference_authorization_id is not None:
                raise SchemaError(
                    "operation protocol event cannot name a match authorization"
                )
            derived_reference = str(state["operation_reference_digest"])
            if reference_digest not in (None, derived_reference):
                raise SchemaError(
                    "operation reference digest must be derived from World"
                )
            reference_digest = derived_reference
        elif reference_kind == "certificate":
            if reference_authorization_id is not None:
                if (
                    not isinstance(reference_authorization_id, str)
                    or not reference_authorization_id.strip()
                ):
                    raise SchemaError(
                        "certificate match authorization id must be non-empty"
                    )
                acceptance = self._world_client.read_match_acceptance(
                    buyer_id=buyer_id,
                    idempotency_key=match_authorization_acceptance_key(
                        reference_authorization_id
                    ),
                    caller="platform:events",
                )
                if acceptance is None:
                    raise SchemaError(
                        "certificate match authorization is not persisted"
                    )
                certificate = self._world_client.read_match_certificate(
                    cert_id=match_certificate_id(acceptance.acceptance_digest),
                    caller="platform:events",
                )
                if certificate is None:
                    raise SchemaError(
                        "certificate match authorization has no persisted certificate"
                    )
                validate_match_certificate(
                    certificate,
                    acceptance=acceptance,
                    expected_buyer_id=buyer_id,
                    expected_order_id=str(state["order_id"]),
                )
                derived_reference = certificate.certificate_digest
                if reference_digest not in (None, derived_reference):
                    raise SchemaError(
                        "certificate digest disagrees with the persisted authorization"
                    )
                reference_digest = derived_reference
            elif not isinstance(reference_digest, str) or not reference_digest:
                raise SchemaError(
                    "certificate protocol event needs a persisted authorization or digest"
                )
        else:
            raise SchemaError(
                f"unsupported protocol event reference kind {reference_kind!r}"
            )

        events = self._world_client.read_protocol_events(
            binding.binding_digest,
            caller="platform:events",
        )
        for existing in events:
            if (
                existing.actor_id == "platform:events"
                and existing.idempotency_key == idempotency_key
            ):
                if (
                    existing.event_id != event_id
                    or existing.event_kind != event_kind
                    or existing.reference_kind != reference_kind
                    or existing.reference_digest != reference_digest
                    or existing.binding != binding
                ):
                    raise IdempotencyConflict(
                        "protocol event idempotency key was reused with different intent"
                    )
                return self._delivery_envelope(
                    existing,
                    ts=ts,
                    in_reply_to=in_reply_to,
                )

        logical_time = state["logical_time"]
        revision = state["state_revision"]
        if (
            isinstance(logical_time, bool)
            or not isinstance(logical_time, int)
            or logical_time < 0
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise SchemaError("World returned invalid protocol time or revision")
        event = build_next_protocol_event(
            events,
            binding,
            event_id=event_id,
            event_kind=event_kind,
            required_order_state=str(state["state"]),
            required_state_revision=revision,
            reference_kind=reference_kind,
            reference_digest=reference_digest,
            issued_at_tick=logical_time,
            expires_at_tick=logical_time + ttl_ticks,
            actor_id="platform:events",
            idempotency_key=idempotency_key,
        )
        persisted = self._world_client.publish_protocol_event(event)
        return self._delivery_envelope(
            persisted,
            ts=ts,
            in_reply_to=in_reply_to,
        )

    def handle(self, env: Envelope) -> Envelope:
        kind = str(env.action.get("kind", ""))
        if kind == "platform.issue_protocol_event":
            return self._issue(env)
        if kind == "platform.advance_market_clock":
            return self._advance_market_clock(env)
        payload: ProtocolEventDecisionPayload | dict[str, Any] = _payload_mapping(env)
        if set(payload) != {"event_id", "reason"}:
            raise SchemaError(
                "protocol event decision requires exactly event_id and reason"
            )
        event_id = str(payload["event_id"])
        reason = str(payload["reason"])
        if not event_id or not reason.strip():
            raise SchemaError("protocol event decision fields must be non-empty")
        event = self._world_client.read_protocol_event(
            event_id,
            caller="platform:events",
        )
        if event is None:
            raise SchemaError("protocol event does not exist")
        if env.from_ != event.binding.recipient_id:
            raise LifecycleAuthorizationError(
                "only the bound protocol event recipient may decide it"
            )
        decision_by_kind = {
            "commerce.acknowledge_protocol_event": "acknowledge",
            "commerce.reject_protocol_event": "reject",
            "commerce.process_protocol_event": "process",
        }
        try:
            decision = decision_by_kind[kind]
        except KeyError as exc:
            raise SchemaError(f"unsupported protocol event decision {kind!r}") from exc
        if decision == "process":
            persisted = self._world_client.process_protocol_event(
                event_id=event_id,
                original_actor=env.from_,
                reason=reason,
                idempotency_key=env.idempotency_key,
            )
            return self._receipt_envelope(env, persisted)

        prior_receipts = self._world_client.read_protocol_receipts(
            event.binding.binding_digest,
            order_id=event.binding.order_id,
            caller="platform:events",
        )
        for existing in prior_receipts:
            if (
                existing.actor_id == env.from_
                and existing.idempotency_key == env.idempotency_key
            ):
                if (
                    existing.event_digest != event.event_digest
                    or existing.decision != decision
                    or existing.reason != reason
                ):
                    raise IdempotencyConflict(
                        "protocol receipt idempotency key was reused with different intent"
                    )
                return self._receipt_envelope(env, existing)

        state = self._world_client.read_order_protocol_state(
            event.binding.order_id,
            caller="platform:events",
        )
        if state is None:
            raise SchemaError("protocol event order is no longer visible")
        logical_time = state["logical_time"]
        revision = state["state_revision"]
        if (
            isinstance(logical_time, bool)
            or not isinstance(logical_time, int)
            or isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise SchemaError("World returned invalid protocol decision state")
        identity = hashlib.sha256(
            (
                f"{event.binding.binding_digest}\x1f{event.event_digest}\x1f"
                f"{env.from_}\x1f{env.idempotency_key}"
            ).encode("utf-8")
        ).hexdigest()
        receipt = build_protocol_event_receipt(
            event,
            receipt_id=f"protocol-receipt:{identity}",
            decision=cast("ReceiptDecision", decision),
            actor_id=env.from_,
            observed_order_state=str(state["state"]),
            observed_state_revision=revision,
            reason=reason,
            effect_reference_digests=(),
            logical_tick=logical_time,
            idempotency_key=env.idempotency_key,
        )
        persisted = self._world_client.append_protocol_receipt(
            receipt,
            original_actor=env.from_,
        )
        return self._receipt_envelope(env, persisted)

    def _issue(self, env: Envelope) -> Envelope:
        """Validate a runtime trigger and issue through the normal core path.

        This is the transport-neutral entry point used by deterministic market
        schedulers and declarative scenarios.  It deliberately calls
        :meth:`deliver` rather than constructing an event from scenario data.
        """

        if env.from_ != "runtime:events":
            raise LifecycleAuthorizationError(
                "only runtime:events may request protocol event issuance"
            )
        payload = _payload_mapping(env)
        required = {
            "market_id",
            "stream_id",
            "order_id",
            "recipient_id",
            "event_id",
            "event_kind",
            "ttl_ticks",
        }
        optional = {
            "reference_kind",
            "reference_digest",
            "reference_authorization_id",
        }
        missing = required - set(payload)
        extras = set(payload) - required - optional
        if missing or extras:
            raise SchemaError(
                "protocol event issuance has invalid fields: "
                f"missing={sorted(missing)!r}, extras={sorted(extras)!r}"
            )
        reference_kind = payload.get("reference_kind", "operation")
        if reference_kind not in {"operation", "certificate"}:
            raise SchemaError("protocol event reference_kind is invalid")
        reference_digest = payload.get("reference_digest")
        if reference_digest is not None and not isinstance(reference_digest, str):
            raise SchemaError("protocol event reference_digest must be a string")
        reference_authorization_id = payload.get("reference_authorization_id")
        if reference_authorization_id is not None and not isinstance(
            reference_authorization_id, str
        ):
            raise SchemaError(
                "protocol event reference_authorization_id must be a string"
            )
        ttl_ticks = payload["ttl_ticks"]
        if isinstance(ttl_ticks, bool) or not isinstance(ttl_ticks, int):
            raise SchemaError("protocol event ttl_ticks must be an integer")
        return self.deliver(
            market_id=str(payload["market_id"]),
            stream_id=str(payload["stream_id"]),
            order_id=str(payload["order_id"]),
            recipient_id=str(payload["recipient_id"]),
            event_id=str(payload["event_id"]),
            event_kind=str(payload["event_kind"]),
            ttl_ticks=ttl_ticks,
            idempotency_key=env.idempotency_key,
            reference_kind=cast("Literal['operation', 'certificate']", reference_kind),
            reference_digest=reference_digest,
            reference_authorization_id=reference_authorization_id,
            ts=env.ts,
            in_reply_to=env.msg_id,
        )

    def _advance_market_clock(self, env: Envelope) -> Envelope:
        """Forward a trusted scheduler tick to the authoritative World clock."""

        if env.from_ != "runtime:clock":
            raise LifecycleAuthorizationError(
                "only runtime:clock may advance the market clock"
            )
        payload = _payload_mapping(env)
        if set(payload) != {"to_tick"}:
            raise SchemaError("market clock advance requires exactly to_tick")
        to_tick = payload["to_tick"]
        if isinstance(to_tick, bool) or not isinstance(to_tick, int) or to_tick < 0:
            raise SchemaError("market clock to_tick must be a non-negative integer")
        logical_time = self._world_client.advance_logical_time(to_tick=to_tick)
        return Envelope(
            msg_id=f"{env.msg_id}:market-clock-advanced",
            ts=env.ts,
            from_="platform:events",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.market_clock_advanced",
                "payload": {"logical_time": logical_time},
            },
        )

    def _delivery_envelope(
        self,
        event: ProtocolEvent,
        *,
        ts: str,
        in_reply_to: str | None,
    ) -> Envelope:
        return Envelope(
            msg_id=f"{event.event_id}:delivery",
            ts=ts,
            from_="platform:events",
            to=event.binding.recipient_id,
            in_reply_to=in_reply_to,
            idempotency_key=event.idempotency_key,
            action={
                "kind": "platform.deliver_protocol_event",
                "payload": {
                    "event": protocol_event_to_dict(event),
                },
            },
        )

    @staticmethod
    def _receipt_envelope(
        request: Envelope,
        receipt: ProtocolEventReceipt,
    ) -> Envelope:
        return Envelope(
            msg_id=f"{request.msg_id}:protocol-receipt",
            ts=request.ts,
            from_="platform:events",
            to=request.from_,
            in_reply_to=request.msg_id,
            idempotency_key=request.idempotency_key,
            action={
                "kind": "platform.protocol_event_receipt",
                "payload": {
                    "receipt": protocol_event_receipt_to_dict(receipt),
                },
            },
        )


class PSPPolicy:
    """Baseline payment policy that settles by writing World facts."""

    def __init__(
        self,
        *,
        world_client: "WorldClient",
        audit: "AuditLog | None" = None,
        max_transactions_per_buyer: int | None = None,
    ) -> None:
        self._world_client = world_client
        self._audit = audit
        self._max_transactions_per_buyer = max_transactions_per_buyer
        self._orders_by_buyer: dict[str, set[str]] = {}

    def settle(self, env: Envelope) -> Envelope:
        payload = env.action.get("payload", {})
        if not isinstance(payload, dict):
            raise SchemaError("platform.settle_payment payload must be an object")
        order = self._settlement_order(payload)
        self._verify_buyer_actor(env, order)
        prior_orders = self._orders_by_buyer.setdefault(str(order.buyer_id), set())
        if (
            self._max_transactions_per_buyer is not None
            and str(order.order_id) not in prior_orders
            and len(prior_orders) >= self._max_transactions_per_buyer
        ):
            raise OrderNotSettleable(
                f"buyer {order.buyer_id} reached max_transactions_per_buyer="
                f"{self._max_transactions_per_buyer}"
            )

        # Hard guard: World must contain authority for this exact price and
        # commercial identity. Pre-flight before any World write so failure
        # leaves no half-state. Audit text is evidence, never authority.
        authorization_kind = self._verify_settlement_authorization(
            payload, order, request_idempotency_key=env.idempotency_key
        )
        if authorization_kind == "match_certificate":
            self._verify_merchant_consent(order)

        allow_partial = payload.get("allow_partial", False)
        if not isinstance(allow_partial, bool):
            raise SchemaError("platform.settle_payment.allow_partial must be boolean")

        if allow_partial:
            # This is the current single-shot S22 contract: the first request
            # records the complete fulfilled/backordered split.  A later fill
            # of outstanding units is intentionally not implemented yet; World
            # rejects a second allocation for the order instead of pretending
            # that replenishment completed it.
            prior = self._world_client.read(
                "fulfillments", order.order_id, caller="platform:psp"
            )
            scoped_key = scope_idempotency_key(str(order.buyer_id), env.idempotency_key)
            if prior is not None and prior.idempotency_key in {
                env.idempotency_key,
                scoped_key,
            }:
                fulfilled_qty = int(prior.fulfilled_qty)
                receipt = (
                    None
                    if prior.receipt_txn_id is None
                    else self._world_client.read(
                        "ledger", prior.receipt_txn_id, caller="platform:psp"
                    )
                )
            else:
                inventory = self._world_client.read(
                    "inventory", order.sku_id, caller="platform:psp"
                )
                if inventory is None:
                    fulfilled_qty = 0
                else:
                    fulfilled_qty = min(order.qty, max(0, _available_qty(inventory)))
                receipt = (
                    None
                    if fulfilled_qty == 0
                    else _coerce_partial_receipt(
                        payload,
                        order=order,
                        env=env,
                        fulfilled_qty=fulfilled_qty,
                    )
                )
            allocation = self._world_client.settle_order_partial(
                order=order,
                fulfilled_qty=fulfilled_qty,
                receipt=receipt,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
            )
            prior_orders.add(str(order.order_id))
            return Envelope(
                msg_id=f"{env.msg_id}:allocated",
                ts=env.ts,
                from_="platform:psp",
                to=env.from_,
                in_reply_to=env.msg_id,
                idempotency_key=env.idempotency_key,
                action={
                    "kind": "platform.fulfillment_allocation",
                    "payload": {
                        "order_id": order.order_id,
                        "txn_id": allocation.txn_id,
                        "status": allocation.status,
                        "requested_qty": allocation.requested_qty,
                        "fulfilled_qty": allocation.fulfilled_qty,
                        "backordered_qty": allocation.backordered_qty,
                    },
                },
            )

        # The atomic, idempotent settle transaction now lives in World — one
        # commit under the world lock, keyed by idempotency_key. PSP no longer
        # holds the world lock or issues the three writes itself. Transitional:
        # still a direct in-process call; the over-VCP ``world.settle_order``
        # emit lands with the re-entrant turn model.
        receipt = _coerce_receipt(payload, order=order, env=env)
        settled = self._world_client.settle_order(
            order=order,
            receipt=receipt,
            by_role="platform",
            idempotency_key=env.idempotency_key,
        )
        prior_orders.add(str(order.order_id))

        return Envelope(
            msg_id=f"{env.msg_id}:settled",
            ts=env.ts,
            from_="platform:psp",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.settlement_receipt",
                "payload": {
                    "order_id": order.order_id,
                    "txn_id": settled.txn_id,
                    "status": "settled",
                },
            },
        )

    def _settlement_order(self, payload: dict[str, Any]) -> Order:
        """Resolve the exact order bound by a certificate-only request.

        The actor terminal deliberately permits ``{"cert_id": ...}`` so the
        buyer need not copy World-owned commercial bindings back onto the
        wire.  Those bindings must come from the authoritative certificate,
        not from agent memory or a reconciler.  An empty retained conformance
        probe is a normal actor rejection and must never fall through to
        ``Decimal(None)`` as an infrastructure error.
        """

        if "supply_authority_id" in payload:
            allowed = {
                "supply_authority_id",
                "supply_authority_digest",
                "sku_id",
                "qty",
                "allow_partial",
            }
            required = allowed - {"allow_partial"}
            if not required.issubset(payload) or not set(payload) <= allowed:
                raise OrderNotSettleable(
                    "supply-authority settlement fields are not exact"
                )
            authority = self._world_client.read_supply_purchase_authority(
                str(payload["supply_authority_id"]),
                caller="platform:psp",
            )
            if authority is None:
                raise OrderNotSettleable(
                    "supply purchase authority is not present in World"
                )
            validate_supply_purchase_authority(authority)
            if (
                payload.get("supply_authority_digest")
                != authority.authority_digest
                or payload.get("sku_id") != authority.sku_id
            ):
                raise OrderNotSettleable(
                    "supply settlement does not match its sealed authority"
                )
            qty = payload.get("qty")
            if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
                raise OrderNotSettleable(
                    "supply-authority settlement qty must be positive"
                )
            return Order(
                order_id=OrderId(authority.order_id),
                buyer_id=AgentId(authority.buyer_id),
                merchant_id=AgentId(authority.merchant_id),
                sku_id=SkuId(authority.sku_id),
                qty=qty,
                agreed_price=Money(
                    Decimal(authority.unit_price_cents) / Decimal(100),
                    authority.currency,
                ),
                state=OrderState.ACCEPTED,
            )
        if set(payload) <= {"cert_id"}:
            cert_id = payload.get("cert_id")
            if not isinstance(cert_id, str) or not cert_id.strip():
                raise OrderNotSettleable(
                    "certificate-only settlement requires a non-empty cert_id"
                )
            certificate = self._world_client.read_match_certificate(
                cert_id, caller="platform:psp"
            )
            if certificate is None:
                raise OrderNotSettleable(
                    "match certificate is not present in authoritative World state"
                )
            return Order(
                order_id=OrderId(certificate.order_id),
                buyer_id=AgentId(certificate.buyer_id),
                merchant_id=AgentId(certificate.merchant_id),
                sku_id=SkuId(certificate.sku_id),
                qty=certificate.qty,
                agreed_price=Money(
                    Decimal(certificate.unit_price_cents) / Decimal(100),
                    certificate.currency,
                ),
                state=OrderState.ACCEPTED,
            )
        return _coerce_order(payload)

    def refund(self, env: Envelope) -> Envelope:
        """Process a buyer's ``commerce.request_return`` into a refund — the
        settle-family inverse. Eligibility gate: the merchant must have published
        the sku as ``returnable`` (the SAME world attribute the buyer grounds for
        a ``returnable`` must_have — one source for the boundary). On eligible,
        run the atomic + idempotent ``world.refund_order`` (REFUNDED + restock +
        reversing ledger) and reply ``commerce.issue_refund``.

        Raises:
            OrderNotRefundable: unknown order, a non-returnable sku, or an order
                not in a refundable state (the world transition enforces the
                state allow-list).
        """
        payload = env.action.get("payload", {})
        if not isinstance(payload, dict):
            raise SchemaError("commerce.request_return payload must be an object")
        _reject_time_claims(payload)
        order = self._world_client.read("orders", OrderId(str(payload["order_id"])), caller="platform:psp")
        if order is None:
            raise OrderNotRefundable(f"refund for unknown order {payload.get('order_id')!r}")
        self._verify_buyer_actor(env, order)
        listing = self._world_client.read("catalog", order.sku_id, caller="platform:psp")
        returnable = bool((getattr(listing, "attributes", {}) or {}).get("returnable")) if listing else False
        if not returnable:
            raise OrderNotRefundable(
                f"sku {order.sku_id} is not returnable — refund refused"
            )
        allocation = self._world_client.read(
            "fulfillments",
            order.order_id,
            caller="platform:psp",
        )
        paid_qty = (
            int(allocation.fulfilled_qty)
            if allocation is not None
            else int(order.qty)
        )
        if paid_qty <= 0:
            raise OrderNotRefundable(
                f"order {order.order_id} has no fulfilled units to refund"
            )
        refund_receipt = Receipt(
            txn_id=TxnId(f"refund:{order.order_id}"), ts=env.ts, order_id=order.order_id,
            buyer_id=order.buyer_id, merchant_id=order.merchant_id, sku_id=order.sku_id,
            qty=paid_qty, price=order.agreed_price, idempotency_key=env.idempotency_key,
        )
        refunded = self._world_client.refund_order(
            order=order, refund_receipt=refund_receipt,
            by_role="platform", idempotency_key=env.idempotency_key,
        )
        return Envelope(
            msg_id=f"{env.msg_id}:refunded",
            ts=env.ts,
            from_="platform:psp",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "commerce.issue_refund",
                "payload": {
                    "order_id": order.order_id,
                    "txn_id": refunded.txn_id,
                    "status": "refunded",
                },
            },
        )

    @staticmethod
    def _verify_buyer_actor(env: Envelope, order: Order) -> None:
        """Bind a PSP request to the exact buyer recorded on the order."""
        if env.from_ != str(order.buyer_id):
            raise OrderIdentityMismatch(
                f"PSP caller {env.from_!r} does not own order buyer_id "
                f"{str(order.buyer_id)!r}"
            )


    # --- Merchant consent guard --------------------------------------

    def _verify_settlement_authorization(
        self,
        payload: dict[str, Any],
        order: Order,
        *,
        request_idempotency_key: str,
    ) -> Literal[
        "match_certificate",
        "negotiation_agreement",
        "supply_purchase_authority",
    ]:
        """Authorize one exact order through one World-backed authority path.

        Published-price purchases retain the persisted search-session and
        match-certificate contract.  A negotiated price may instead name a
        World-owned, terminal accepted negotiation thread.  The two paths are
        deliberately disjoint: a caller cannot submit a bad negotiation
        reference and then fall back to weaker audit-text consent.
        """

        cert_id = payload.get("cert_id")
        negotiation_id = payload.get("negotiation_id")
        supply_authority_id = payload.get("supply_authority_id")
        if sum(
            value not in (None, "")
            for value in (cert_id, negotiation_id, supply_authority_id)
        ) > 1:
            raise OrderNotSettleable(
                "settlement must name exactly one transaction authority"
            )
        if supply_authority_id not in (None, ""):
            self._verify_supply_purchase_authority(
                payload,
                order,
                request_idempotency_key=request_idempotency_key,
            )
            return "supply_purchase_authority"
        if negotiation_id not in (None, ""):
            self._verify_negotiation_agreement(payload, order)
            return "negotiation_agreement"
        self._verify_match_certificate(
            payload,
            order,
            request_idempotency_key=request_idempotency_key,
        )
        return "match_certificate"

    def _verify_supply_purchase_authority(
        self,
        payload: Mapping[str, Any],
        order: Order,
        *,
        request_idempotency_key: str,
    ) -> None:
        """Verify a persisted supply authority against current World state."""

        authority_id = payload.get("supply_authority_id")
        if not isinstance(authority_id, str) or not authority_id.strip():
            raise OrderNotSettleable(
                "supply settlement requires an authority id"
            )
        authority = self._world_client.read_supply_purchase_authority(
            authority_id,
            caller="platform:psp",
        )
        if authority is None:
            raise OrderNotSettleable(
                "supply purchase authority is not present in World"
            )
        try:
            validate_supply_purchase_authority(authority)
        except (SchemaError, ValueError) as exc:
            raise OrderNotSettleable(
                "persisted supply purchase authority is invalid"
            ) from exc
        expected = (
            authority.authority_digest,
            authority.buyer_id,
            authority.merchant_id,
            authority.sku_id,
            authority.order_id,
            authority.unit_price_cents,
            authority.currency,
        )
        observed = (
            payload.get("supply_authority_digest"),
            str(order.buyer_id),
            str(order.merchant_id),
            str(order.sku_id),
            str(order.order_id),
            _to_cents_half_up(order.agreed_price.amount),
            order.agreed_price.currency,
        )
        if observed != expected:
            raise OrderNotSettleable(
                "supply purchase authority does not bind this settlement"
            )

        if self._is_exact_supply_settlement_replay(
            payload,
            order,
            request_idempotency_key=request_idempotency_key,
        ):
            return
        logical_time = self._world_client.read_logical_time(caller="platform:psp")
        if (
            logical_time < authority.issued_at_tick
            or logical_time >= authority.expires_at_tick
        ):
            raise OrderNotSettleable("supply purchase authority is stale")
        listing = self._world_client.read(
            "catalog", order.sku_id, caller="platform:psp"
        )
        supply = self._world_client.read_supply_state(
            str(order.sku_id), caller="platform:psp"
        )
        if listing is None:
            raise OrderNotSettleable("supply authority listing is missing")
        current = (
            str(listing.merchant_id),
            _to_cents_half_up(listing.list_price.amount),
            listing.list_price.currency,
            supply.version,
        )
        sealed = (
            authority.merchant_id,
            authority.unit_price_cents,
            authority.currency,
            authority.supply_version,
        )
        if current != sealed:
            raise OrderNotSettleable(
                "supply purchase authority no longer matches current supply"
            )
        if authority.available_qty <= 0 or supply.available_qty <= 0:
            raise OrderNotSettleable(
                "supply purchase authority has no purchasable inventory"
            )
        allow_partial = payload.get("allow_partial", False)
        if not isinstance(allow_partial, bool):
            raise SchemaError(
                "platform.settle_payment.allow_partial must be boolean"
            )
        if not allow_partial and order.qty > supply.available_qty:
            raise OrderNotSettleable(
                "all-or-nothing supply purchase exceeds available inventory"
            )

    def _is_exact_supply_settlement_replay(
        self,
        payload: Mapping[str, Any],
        order: Order,
        *,
        request_idempotency_key: str,
    ) -> bool:
        if self._is_exact_settlement_replay(
            payload,
            order,
            request_idempotency_key=request_idempotency_key,
        ):
            return True
        persisted = self._world_client.read(
            "orders", order.order_id, caller="platform:psp"
        )
        allocation = self._world_client.read(
            "fulfillments", order.order_id, caller="platform:psp"
        )
        scoped = scope_idempotency_key(
            str(order.buyer_id), request_idempotency_key
        )
        return bool(
            persisted is not None
            and allocation is not None
            and persisted.buyer_id == order.buyer_id
            and persisted.merchant_id == order.merchant_id
            and persisted.sku_id == order.sku_id
            and persisted.qty == order.qty
            and persisted.agreed_price == order.agreed_price
            and allocation.idempotency_key
            in {request_idempotency_key, scoped}
        )

    def _verify_negotiation_agreement(
        self,
        payload: Mapping[str, Any],
        order: Order,
    ) -> None:
        """Verify an accepted World negotiation as exact order authority."""

        negotiation_id = payload.get("negotiation_id")
        if not isinstance(negotiation_id, str) or not negotiation_id.strip():
            raise OrderNotSettleable(
                "negotiated settlement requires a non-empty negotiation_id"
            )
        thread = self._world_client.read_negotiation_thread(
            negotiation_id,
            caller="platform:psp",
        )
        if thread is None:
            raise OrderNotSettleable(
                "negotiation agreement is not present in authoritative World state"
            )
        try:
            validate_negotiation_thread(thread)
        except (SchemaError, ValueError) as exc:
            raise OrderNotSettleable(
                "authoritative negotiation agreement is invalid"
            ) from exc
        agreement = thread.agreement
        if thread.status != "accepted" or agreement is None:
            raise OrderNotSettleable(
                "negotiation thread is not terminally accepted"
            )
        try:
            expected_order_id = negotiated_order_id(
                negotiation_id, agreement.offer_id
            )
        except (SchemaError, ValueError) as exc:
            raise OrderNotSettleable(str(exc)) from exc
        if str(order.order_id) != expected_order_id:
            raise OrderNotSettleable(
                "negotiation agreement does not authorize this order identity"
            )

        current_tick = self._world_client.read_logical_time(caller="platform:psp")
        if current_tick < agreement.accepted_at_tick or current_tick >= agreement.expires_at_tick:
            raise OrderNotSettleable(
                "negotiation agreement is not valid at World logical time"
            )
        listing = self._world_client.read(
            "catalog", order.sku_id, caller="platform:psp"
        )
        if listing is None:
            raise OrderNotSettleable(
                "negotiation agreement listing is missing from World"
            )
        if (
            agreement.listing_revision != _catalog_revision_from_listing(listing)
            or agreement.listing_digest != negotiation_listing_digest(listing)
        ):
            raise OrderNotSettleable(
                "negotiation agreement is stale for the current listing"
            )

        expected = (
            str(order.buyer_id),
            str(order.merchant_id),
            str(order.sku_id),
            int(order.qty),
            _to_cents_half_up(order.agreed_price.amount),
            order.agreed_price.currency,
        )
        observed = (
            agreement.buyer_id,
            agreement.merchant_id,
            agreement.sku_id,
            agreement.qty,
            agreement.unit_price,
            agreement.currency,
        )
        if observed != expected:
            raise OrderNotSettleable(
                "negotiation agreement does not bind the exact settlement order"
            )

    def _verify_match_certificate(
        self,
        payload: dict[str, Any],
        order: Order,
        *,
        request_idempotency_key: str,
    ) -> None:
        """Validate a certificate loaded from authoritative World state.

        Audit text is evidence, not authority.  A payload can name only a
        certificate that World persisted atomically with its exact acceptance;
        every settlement field and both current revisions are rechecked before
        any order, inventory, or ledger write.
        """
        listing = self._world_client.read(
            "catalog", order.sku_id, caller="platform:psp"
        )
        cert_id = payload.get("cert_id")
        if cert_id in (None, ""):
            # Narrow compatibility for legacy 1x1 buyers that accepted a
            # persisted offer but did not echo the certificate id into their
            # settlement. Both the current session and certificate must be
            # uniquely resolvable from trusted World state.
            matching = self._world_client.resolve_search_session(
                buyer_id=str(order.buyer_id),
                offer_id=f"agg:{order.sku_id}",
                caller="platform:psp",
            )
            if matching is None:
                any_matching = self._world_client.resolve_search_session(
                    buyer_id=str(order.buyer_id),
                    offer_id=f"agg:{order.sku_id}",
                    caller="platform:psp",
                    unique_only=False,
                    current_only=False,
                )
                if any_matching is None:
                    return
                raise OrderNotSettleable(
                    "legacy settlement has no unique current search session"
                )
            certificate = self._world_client.resolve_match_certificate(
                buyer_id=str(order.buyer_id),
                order_id=str(order.order_id),
                caller="platform:psp",
            )
            if certificate is None or certificate.session_id != matching.session_id:
                raise OrderNotSettleable(
                    "legacy settlement has no unique persisted match certificate"
                )
        else:
            certificate = self._world_client.read_match_certificate(
                str(cert_id), caller="platform:psp"
            )
        if certificate is None:
            raise OrderNotSettleable(
                "match certificate is not present in authoritative World state"
            )
        session = self._world_client.read_search_session(
            certificate.session_id, caller="platform:psp"
        )
        acceptance = self._world_client.read_match_acceptance(
            buyer_id=certificate.buyer_id,
            idempotency_key=certificate.idempotency_key,
            caller="platform:psp",
        )
        try:
            supply = self._world_client.read_supply_state(
                str(order.sku_id), caller="platform:psp"
            )
        except (ValueError, SchemaError) as exc:
            raise OrderNotSettleable("match certificate supply state is missing") from exc
        if session is None or acceptance is None or listing is None:
            raise OrderNotSettleable("match certificate backing state is incomplete")
        exact_replay = self._is_exact_settlement_replay(
            payload,
            order,
            request_idempotency_key=request_idempotency_key,
        )
        try:
            validate_match_certificate(
                certificate,
                session=session,
                acceptance=acceptance,
                current_tick=(
                    None
                    if exact_replay
                    else self._world_client.read_logical_time(caller="platform:psp")
                ),
                current_catalog_revision=(
                    None
                    if exact_replay
                    else _catalog_revision_from_listing(listing)
                ),
                current_inventory_revision=None if exact_replay else supply.version,
                expected_buyer_id=str(order.buyer_id),
                expected_order_id=str(order.order_id),
            )
        except (SchemaError, ValueError) as exc:
            raise OrderNotSettleable(str(exc)) from exc
        cents = _to_cents_half_up(order.agreed_price.amount)
        expected = (
            str(order.merchant_id),
            str(order.sku_id),
            cents,
            order.agreed_price.currency,
            order.qty,
        )
        observed = (
            certificate.merchant_id,
            certificate.sku_id,
            certificate.unit_price_cents,
            certificate.currency,
            certificate.qty,
        )
        if observed != expected:
            raise OrderNotSettleable(
                "match certificate does not bind the exact settlement order"
            )

    def _is_exact_settlement_replay(
        self,
        payload: Mapping[str, Any],
        order: Order,
        *,
        request_idempotency_key: str,
    ) -> bool:
        persisted = self._world_client.read(
            "orders", order.order_id, caller="platform:psp"
        )
        if persisted is None or persisted.state not in {
            OrderState.PARTIALLY_SETTLED,
            OrderState.SETTLED,
            OrderState.DISPATCHED,
            OrderState.RETURNED,
            OrderState.REFUNDED,
        }:
            return False
        if (
            persisted.buyer_id != order.buyer_id
            or persisted.merchant_id != order.merchant_id
            or persisted.sku_id != order.sku_id
            or persisted.qty != order.qty
            or persisted.agreed_price != order.agreed_price
        ):
            return False
        txn_id = TxnId(str(payload.get("txn_id", f"txn:{order.order_id}")))
        receipt = self._world_client.read(
            "ledger", txn_id, caller="platform:psp"
        )
        return bool(
            receipt is not None
            and receipt.order_id == order.order_id
            and receipt.buyer_id == order.buyer_id
            and receipt.merchant_id == order.merchant_id
            and receipt.sku_id == order.sku_id
            and receipt.qty == order.qty
            and receipt.price == order.agreed_price
            and receipt.idempotency_key == request_idempotency_key
        )

    def _verify_merchant_consent(self, order: Order) -> None:
        """Verify the legacy published-price commitment from World.

        This method is reached only when settlement did not name a World-backed
        negotiation agreement.  The only remaining compatibility path is the
        listing price published by the listing owner.  Historical
        ``commerce.accept_offer`` or ``commerce.counter_offer`` audit records
        are deliberately ignored. They are evidence of messages, not
        authoritative commerce state.

        A negotiated discount must instead supply ``negotiation_id`` and pass
        :meth:`_verify_negotiation_agreement`.  Audit-free construction remains
        a narrow unit-test compatibility seam. Formal and executable benchmark
        episodes always provide an audit and therefore cannot use that seam.
        """
        if self._audit is None:
            return
        agreed_cents = _to_cents_half_up(order.agreed_price.amount)
        listing = self._world_client.read("catalog", order.sku_id, caller="platform:psp")
        if listing is None:
            raise NoMerchantConsent(
                f"settlement sku {order.sku_id!r} has no World listing authority"
            )
        listing_owner = str(listing.merchant_id)
        order_owner = str(order.merchant_id)
        owner_matches = (
            listing_owner == order_owner
            or (
                listing_owner == "merchant"
                and order_owner.split(":", 1)[0] == "merchant"
            )
        )
        if not owner_matches:
            raise NoMerchantConsent(
                f"order names merchant {order_owner!r}, but sku {order.sku_id!r} "
                f"is owned by {listing_owner!r}"
            )
        list_cents = _to_cents_half_up(listing.list_price.amount)
        if agreed_cents != list_cents:
            raise NoMerchantConsent(
                f"settle at {agreed_cents} cents for sku {str(order.sku_id)!r} "
                f"from {order_owner!r} has no World-backed authority: it does "
                f"not match the published price and no accepted negotiation "
                f"agreement was supplied"
            )


class AfterSalesPolicy:
    """Stateless mediation for World-authoritative after-sales transitions.

    Buyer and Merchant send only compact intents.  This policy derives the
    operation from the authenticated action kind, preserves the full actor id,
    and forwards through ``WorldClient``.  It owns no lifecycle dictionary and
    never calls the legacy refund or exchange transactions.
    """

    _OPERATION_BY_ACTION = {
        "commerce.cancel_paid_order": "cancel_paid_order",
        "commerce.request_return": "request_return",
        "commerce.authorize_return": "authorize_return",
        "commerce.deny_return": "deny_return",
        "commerce.receive_return": "receive_return",
        "commerce.open_refund_case": "open_refund_case",
        "commerce.approve_refund": "approve_refund",
        "commerce.deny_refund": "deny_refund",
        "commerce.request_exchange": "request_exchange",
        "commerce.authorize_exchange": "authorize_exchange",
        "commerce.deny_exchange": "deny_exchange",
        "commerce.complete_exchange": "complete_exchange",
        "commerce.open_dispute": "open_dispute",
        "commerce.submit_dispute_evidence": "submit_dispute_evidence",
        "commerce.respond_to_dispute": "respond_to_dispute",
        "commerce.request_ledger_reconciliation": (
            "request_ledger_reconciliation"
        ),
    }
    _RESOURCE_BY_ACTION = {
        "commerce.read_payment_history": "payment_history",
        "commerce.read_ledger_history": "ledger_history",
        "commerce.read_packing_history": "packing_history",
        "commerce.read_after_sales_history": "after_sales_history",
        "commerce.read_after_sales_policy": "policy",
    }

    def __init__(
        self,
        *,
        world_client: "WorldClient",
        adjudication_policy: (
            Callable[[Mapping[str, Any], str], tuple[str, str]] | None
        ) = None,
    ) -> None:
        self._world_client = world_client
        self._adjudication_policy = (
            adjudication_policy or _verified_evidence_count_adjudication
        )

    def handle(self, env: Envelope) -> Envelope:
        kind = str(env.action.get("kind", ""))
        if kind in self._RESOURCE_BY_ACTION:
            return self._read(env, resource=self._RESOURCE_BY_ACTION[kind])
        try:
            operation = self._OPERATION_BY_ACTION[kind]
        except KeyError as exc:
            raise SchemaError(f"unsupported after-sales action {kind!r}") from exc
        payload = _payload_mapping(env)
        if "op" in payload:
            raise SchemaError("after-sales operation is derived from action kind")
        normalized = dict(
            normalize_after_sales_intent({"op": operation, **payload})
        )
        result = self._world_client.apply_after_sales_intent(
            intent=normalized,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        acknowledgement = self._acknowledgement(
            env, order_id=str(normalized["order_id"]), result=result
        )

        # These service decisions remain explicit World operations with their
        # own authenticated actors and idempotency scopes.  They are not hidden
        # Platform state and are therefore present in commit/replay evidence.
        if operation == "respond_to_dispute":
            history = self._world_client.read_after_sales_resource(
                resource="after_sales_history",
                original_actor="platform:adjudicator",
                order_id=str(normalized["order_id"]),
            )
            ruling_operation, rationale = self._adjudication_policy(
                history, str(normalized["dispute_id"])
            )
            if ruling_operation not in {
                "rule_for_filer",
                "rule_for_respondent",
                "rule_split",
            }:
                raise SchemaError("adjudication policy returned an invalid ruling")
            if not isinstance(rationale, str) or not rationale.strip():
                raise SchemaError("adjudication policy returned no rationale")
            self._world_client.apply_after_sales_intent(
                intent={
                    "op": ruling_operation,
                    "order_id": str(normalized["order_id"]),
                    "dispute_id": str(normalized["dispute_id"]),
                    "rationale": rationale,
                },
                original_actor="platform:adjudicator",
                idempotency_key=f"{env.idempotency_key}:adjudicator",
            )
        elif operation == "request_ledger_reconciliation":
            operation_row = _after_sales_operation(result)
            self._world_client.complete_ledger_reconciliation(
                order_id=str(normalized["order_id"]),
                request_id=str(operation_row["result_key"]),
                idempotency_key=f"{env.idempotency_key}:accounting",
            )
        return acknowledgement

    def _read(self, env: Envelope, *, resource: str) -> Envelope:
        payload = _payload_mapping(env)
        if resource == "policy":
            if set(payload) != {"merchant_id"}:
                raise SchemaError(
                    "after-sales policy read requires exactly merchant_id"
                )
            projection = self._world_client.read_after_sales_resource(
                resource=resource,
                original_actor=env.from_,
                merchant_id=_required_text(payload, "merchant_id"),
            )
        else:
            if set(payload) != {"order_id"}:
                raise SchemaError(
                    "after-sales history read requires exactly order_id"
                )
            projection = self._world_client.read_after_sales_resource(
                resource=resource,
                original_actor=env.from_,
                order_id=_required_text(payload, "order_id"),
            )
        return Envelope(
            msg_id=f"{env.msg_id}:after-sales-snapshot",
            ts=env.ts,
            from_="platform:after-sales",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.after_sales_snapshot",
                "payload": projection,
            },
        )

    @staticmethod
    def _acknowledgement(
        env: Envelope, *, order_id: str, result: Mapping[str, Any]
    ) -> Envelope:
        operation = _after_sales_operation(result)
        disposition = result.get("disposition")
        if disposition not in {"committed", "idempotent"}:
            raise SchemaError("World returned an invalid after-sales disposition")
        safe = {
            "operation": str(operation["operation"]),
            "order_id": order_id,
            "disposition": disposition,
            "result_table": str(operation["result_table"]),
            "result_key": str(operation["result_key"]),
            "result_digest": str(operation["result_digest"]),
            "references": _after_sales_references(result),
        }
        if any(
            not safe[name]
            for name in safe
            if name not in {"disposition", "references"}
        ):
            raise SchemaError("World returned an incomplete after-sales operation")
        return Envelope(
            msg_id=f"{env.msg_id}:after-sales-updated",
            ts=env.ts,
            from_="platform:after-sales",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={"kind": "platform.after_sales_updated", "payload": safe},
        )


class GovernancePolicy:
    """Stateless mediator for World-authoritative marketplace governance."""

    _ORCHESTRATOR_ACTOR = "platform:orchestrator"

    _ACTOR_OPERATIONS = {
        "commerce.publish_campaign": ("publish_campaign", "platform:ads"),
        "commerce.disclose_placement": (
            "disclose_placement",
            "platform:ads",
        ),
        "commerce.activate_campaign": ("activate_campaign", "platform:ads"),
        "commerce.submit_review": ("submit_review", "platform:reviews"),
        "commerce.reject_review_manipulation": (
            "reject_review_manipulation",
            "platform:governance",
        ),
        "commerce.reject_coordination": (
            "reject_coordination",
            "platform:governance",
        ),
        "commerce.accept_remediation_plan": (
            "accept_remediation_plan",
            "platform:remediation",
        ),
        "commerce.complete_remediation_step": (
            "complete_remediation_step",
            "platform:remediation",
        ),
    }
    _TRUSTED_SERVICES = {
        "platform.publish_governance_policy": None,
        "platform.aggregate_reviews": "platform:reviews",
        "platform.ingest_review_observation": "platform:reviews",
        "platform.ingest_market_observation": "platform:governance",
        "platform.resolve_governance_case": "platform:governance",
        "platform.apply_governance_reputation": "platform:reputation",
        "platform.create_remediation_plan": "platform:remediation",
        "platform.verify_remediation_step": "platform:remediation",
        "platform.persist_ranking_context": "platform:ranking",
    }
    _TRUSTED_REQUESTERS = {
        "platform.publish_governance_policy": frozenset({"runtime:governance"}),
        "platform.aggregate_reviews": frozenset(
            {"runtime:reviews", _ORCHESTRATOR_ACTOR}
        ),
        "platform.ingest_review_observation": frozenset({"runtime:reviews"}),
        "platform.ingest_market_observation": frozenset({"runtime:governance"}),
        "platform.resolve_governance_case": frozenset(
            {"runtime:governance", _ORCHESTRATOR_ACTOR}
        ),
        "platform.apply_governance_reputation": frozenset(
            {"runtime:reputation", _ORCHESTRATOR_ACTOR}
        ),
        "platform.create_remediation_plan": frozenset(
            {"runtime:remediation", _ORCHESTRATOR_ACTOR}
        ),
        "platform.verify_remediation_step": frozenset(
            {"runtime:remediation", _ORCHESTRATOR_ACTOR}
        ),
        "platform.persist_ranking_context": frozenset({"platform:ranking"}),
    }
    _POLICY_SERVICE_BY_KIND = {
        "ads_campaign_terms": "platform:ads",
        "review_account_binding": "platform:reviews",
        "reputation_policy_revision": "platform:reputation",
        "remediation_blueprint": "platform:remediation",
    }

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def handle(self, env: Envelope) -> Envelope | list[Envelope]:
        kind = str(env.action.get("kind", ""))
        if kind == "commerce.read_governance_history":
            return self._history(env)
        if kind in self._ACTOR_OPERATIONS:
            return self._actor_intent(env, kind=kind)
        if kind in self._TRUSTED_SERVICES:
            return self._trusted_operation(env, kind=kind)
        raise SchemaError(f"unsupported governance action {kind!r}")

    def _actor_intent(
        self, env: Envelope, *, kind: str
    ) -> Envelope | list[Envelope]:
        operation, service_actor = self._ACTOR_OPERATIONS[kind]
        if env.to != service_actor:
            raise SchemaError(
                f"{kind} must be addressed to {service_actor}"
            )
        expected_role = "buyer" if operation == "submit_review" else "merchant"
        if env.from_.split(":", 1)[0] != expected_role:
            raise SchemaError(f"{kind} requires an authenticated {expected_role}")
        payload = _payload_mapping(env)
        if "op" in payload:
            raise SchemaError("governance operation is derived from action kind")
        intent = dict(normalize_governance_intent({"op": operation, **payload}))
        result = self._world_client.apply_governance_intent(
            intent,
            by_actor=service_actor,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        updated = self._updated(
            env,
            service_actor=service_actor,
            operation=operation,
            result=result,
        )
        if operation == "submit_review":
            if not isinstance(result, ReviewEvidence):
                raise SchemaError("World returned a non-review submission")
            return [
                updated,
                self._internal_request(
                    env,
                    suffix="aggregate",
                    to_actor="platform:reviews",
                    action_kind="platform.aggregate_reviews",
                    payload={"sku_id": result.sku_id},
                ),
            ]
        if operation == "accept_remediation_plan":
            if not isinstance(result, RemediationPlan):
                raise SchemaError(
                    "World returned a non-plan remediation acceptance"
                )
            notice = self._remediation_plan_notice(env, result)
            responses = [updated, notice]
            audit_request = self._remediation_audit_request(notice, result)
            if audit_request is not None:
                responses.append(audit_request)
            return responses
        if operation == "complete_remediation_step":
            if not isinstance(result, RemediationPlan):
                raise SchemaError(
                    "World returned a non-plan remediation completion"
                )
            return [
                updated,
                self._internal_request(
                    env,
                    suffix="verify",
                    to_actor="platform:remediation",
                    action_kind="platform.verify_remediation_step",
                    payload={
                        "plan_id": result.plan_id,
                        "step_id": _required_text(payload, "step_id"),
                    },
                ),
            ]
        return updated

    def _trusted_operation(
        self, env: Envelope, *, kind: str
    ) -> Envelope | list[Envelope]:
        allowed_requesters = self._TRUSTED_REQUESTERS[kind]
        if env.from_ not in allowed_requesters:
            raise LifecycleAuthorizationError(
                f"{kind} requires one of the registered control-plane actors"
            )
        payload = _payload_mapping(env)
        service_actor = self._TRUSTED_SERVICES[kind]
        if service_actor is not None and env.to != service_actor:
            raise LifecycleAuthorizationError(
                f"{kind} must be addressed to {service_actor}"
            )
        resolution_template: dict[str, Any] | None = None
        remediation_template: dict[str, Any] | None = None

        if kind == "platform.publish_governance_policy":
            _require_exact_fields(payload, {"policy_intent"}, action=kind)
            policy_intent = _required_mapping(payload, "policy_intent", action=kind)
            policy_kind = policy_intent.get("kind")
            if not isinstance(policy_kind, str):
                raise SchemaError("governance policy intent requires kind")
            try:
                service_actor = self._POLICY_SERVICE_BY_KIND[policy_kind]
            except KeyError as exc:
                raise SchemaError(
                    f"unsupported governance policy kind {policy_kind!r}"
                ) from exc
            if env.to != service_actor:
                raise LifecycleAuthorizationError(
                    f"{kind} for {policy_kind} must be addressed to {service_actor}"
                )
            result = self._world_client.publish_governance_policy(
                policy_intent,
                by_actor=service_actor,
                original_actor=service_actor,
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.aggregate_reviews":
            _require_exact_fields(payload, {"sku_id"}, action=kind)
            assert service_actor is not None
            result = self._world_client.aggregate_reviews(
                _required_text(payload, "sku_id"),
                by_actor=service_actor,
                original_actor=service_actor,
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.ingest_review_observation":
            _require_exact_fields(payload, {"record_id"}, action=kind)
            assert service_actor == "platform:reviews"
            record_id = _required_text(payload, "record_id")
            evidence = self._world_client.read_evidence_record(
                record_id,
                caller=service_actor,
            )
            if evidence is None:
                raise SchemaError("review observation evidence is unavailable")
            result = self._world_client.ingest_review_observation(
                record_id,
                by_actor=service_actor,
                original_actor=evidence.issuer_id,
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.ingest_market_observation":
            allowed_shapes = (
                {"record_id", "detector_id"},
                {"record_id", "detector_id", "resolution_template"},
                {
                    "record_id",
                    "detector_id",
                    "resolution_template",
                    "remediation_template",
                },
            )
            if set(payload) not in allowed_shapes:
                raise SchemaError(
                    f"{kind} requires detector fields and optional ordered "
                    "resolution/remediation templates"
                )
            if "resolution_template" in payload:
                resolution_template = _governance_resolution_template(
                    _required_mapping(payload, "resolution_template", action=kind),
                    action=kind,
                )
            if "remediation_template" in payload:
                remediation_template = _governance_remediation_template(
                    _required_mapping(payload, "remediation_template", action=kind),
                    action=kind,
                )
            assert service_actor is not None
            result = self._world_client.ingest_market_observation(
                _required_text(payload, "record_id"),
                by_actor=service_actor,
                original_actor=_required_text(payload, "detector_id"),
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.resolve_governance_case":
            if set(payload) == {"decision_intent"}:
                pass
            elif (
                set(payload) == {"decision_intent", "remediation_template"}
                and env.from_ == self._ORCHESTRATOR_ACTOR
            ):
                remediation_template = _governance_remediation_template(
                    _required_mapping(payload, "remediation_template", action=kind),
                    action=kind,
                )
            else:
                raise SchemaError(
                    f"{kind} requires decision_intent; only the registered "
                    "Platform orchestrator may propagate remediation_template"
                )
            assert service_actor is not None
            result = self._world_client.resolve_governance_case(
                _required_mapping(payload, "decision_intent", action=kind),
                by_actor=service_actor,
                original_actor=service_actor,
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.apply_governance_reputation":
            _require_exact_fields(payload, {"source_intent"}, action=kind)
            assert service_actor is not None
            result = self._world_client.apply_governance_reputation(
                _required_mapping(payload, "source_intent", action=kind),
                by_actor=service_actor,
                original_actor=service_actor,
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.create_remediation_plan":
            _require_exact_fields(payload, {"plan_intent"}, action=kind)
            assert service_actor is not None
            result = self._world_client.create_remediation_plan(
                _required_mapping(payload, "plan_intent", action=kind),
                by_actor=service_actor,
                original_actor=service_actor,
                idempotency_key=env.idempotency_key,
            )
        elif kind == "platform.verify_remediation_step":
            _require_exact_fields(payload, {"plan_id", "step_id"}, action=kind)
            assert service_actor is not None
            result = self._world_client.verify_remediation_step(
                _required_text(payload, "plan_id"),
                _required_text(payload, "step_id"),
                by_actor=service_actor,
                original_actor=service_actor,
                idempotency_key=env.idempotency_key,
            )
        else:
            _require_exact_fields(
                payload, {"ranking_result", "buyer_id"}, action=kind
            )
            assert service_actor == "platform:ranking"
            result = self._world_client.persist_ranking_context(
                _required_mapping(payload, "ranking_result", action=kind),
                by_actor=service_actor,
                original_actor=_required_text(payload, "buyer_id"),
                idempotency_key=env.idempotency_key,
            )
        assert service_actor is not None
        updated_to = self._trusted_ack_target(env, kind=kind)
        updated = self._updated(
            env,
            service_actor=service_actor,
            operation=kind.removeprefix("platform."),
            result=result,
            to_actor=updated_to,
        )
        if kind == "platform.ingest_review_observation":
            if not isinstance(result, ReviewEvidence):
                raise SchemaError("World returned a non-review observation")
            return [
                updated,
                self._internal_request(
                    env,
                    suffix="aggregate",
                    to_actor="platform:reviews",
                    action_kind="platform.aggregate_reviews",
                    payload={"sku_id": result.sku_id},
                ),
            ]
        if kind == "platform.ingest_market_observation":
            if not isinstance(result, GovernanceCase):
                raise SchemaError("World returned an invalid governance case")
            responses = [updated, *self._governance_case_notices(env, result)]
            if resolution_template is not None:
                internal_payload: dict[str, Any] = {
                    "decision_intent": {
                        "case_id": result.case_id,
                        **resolution_template,
                    }
                }
                if remediation_template is not None:
                    internal_payload["remediation_template"] = remediation_template
                responses.append(
                    self._internal_request(
                        env,
                        suffix="auto-resolve",
                        to_actor="platform:governance",
                        action_kind="platform.resolve_governance_case",
                        payload=internal_payload,
                    )
                )
            return responses
        if kind == "platform.resolve_governance_case":
            if not isinstance(result, GovernanceCase):
                raise SchemaError("World returned an invalid resolved case")
            responses = [updated, *self._governance_case_notices(env, result)]
            if remediation_template is not None:
                responses.append(
                    self._internal_request(
                        env,
                        suffix="create-plan",
                        to_actor="platform:remediation",
                        action_kind="platform.create_remediation_plan",
                        payload={
                            "plan_intent": {
                                "case_id": result.case_id,
                                **remediation_template,
                            }
                        },
                    )
                )
            return responses
        if kind == "platform.create_remediation_plan":
            if not isinstance(result, RemediationPlan):
                raise SchemaError("World returned an invalid remediation plan")
            return [updated, self._remediation_plan_notice(env, result)]
        if kind == "platform.verify_remediation_step":
            if not isinstance(result, RemediationPlan):
                raise SchemaError("World returned an invalid verified plan")
            notice = self._remediation_plan_notice(env, result)
            responses = [updated, notice]
            audit_request = self._remediation_audit_request(notice, result)
            if audit_request is not None:
                responses.append(audit_request)
            if result.status == "completed":
                responses.append(
                    self._internal_request(
                        env,
                        suffix="reputation",
                        to_actor="platform:reputation",
                        action_kind="platform.apply_governance_reputation",
                        payload={
                            "source_intent": {
                                "source_kind": "remediation_plan",
                                "source_id": result.plan_id,
                            }
                        },
                    )
                )
            return responses
        return updated

    def _governance_case_notices(
        self,
        env: Envelope,
        governance_case: GovernanceCase,
    ) -> list[Envelope]:
        """Notify only World-derived case subjects with safe version references."""

        history = self._world_client.governance_history(
            "governance_case",
            governance_case.case_id,
            caller="platform:governance",
        )
        if not history or not isinstance(history[0], GovernanceCase):
            raise SchemaError("World governance case history is unavailable")
        latest = history[-1]
        if (
            not isinstance(latest, GovernanceCase)
            or latest.case_digest != governance_case.case_digest
            or latest.version != governance_case.version
        ):
            raise SchemaError("World governance case history is not current")
        opened_reference = _governance_reference(history[0])
        current_reference = _governance_reference(governance_case)
        world_tick = self._world_client.read_logical_time(
            caller="platform:governance"
        )
        response_actions: list[str] = []
        if governance_case.status == "open":
            action = {
                "review_integrity": "commerce.reject_review_manipulation",
                "competition": "commerce.reject_coordination",
            }.get(governance_case.case_kind)
            if action is not None:
                response_actions.append(action)
        notices: list[Envelope] = []
        for index, merchant_id in enumerate(
            sorted(governance_case.subject_merchant_ids), start=1
        ):
            notices.append(
                Envelope(
                    msg_id=f"{env.msg_id}:case-notice:{index}",
                    ts=env.ts,
                    from_="platform:governance",
                    to=merchant_id,
                    in_reply_to=env.msg_id,
                    idempotency_key=(
                        f"{env.idempotency_key}:case-notice:{index}"
                    ),
                    action={
                        "kind": "platform.governance_case_notice",
                        "payload": {
                            "opened_case_reference": opened_reference,
                            "current_case_reference": current_reference,
                            "case_kind": governance_case.case_kind,
                            "status": governance_case.status,
                            "response_actions": list(response_actions),
                            "world_tick": world_tick,
                        },
                    },
                )
            )
        return notices

    def _remediation_plan_notice(
        self,
        env: Envelope,
        plan: RemediationPlan,
    ) -> Envelope:
        """Expose World-derived plan identity, progress, and legal next actions."""

        world_tick = self._world_client.read_logical_time(
            caller="platform:remediation"
        )
        response_actions: list[str]
        if plan.status == "draft":
            response_actions = ["commerce.accept_remediation_plan"]
        elif plan.status == "active" and any(
            step.status == "pending" for step in plan.steps
        ):
            response_actions = ["commerce.complete_remediation_step"]
        else:
            response_actions = []
        return Envelope(
            msg_id=f"{env.msg_id}:plan-notice",
            ts=env.ts,
            from_="platform:remediation",
            to=plan.owner_merchant_id,
            in_reply_to=env.msg_id,
            idempotency_key=f"{env.idempotency_key}:plan-notice",
            action={
                "kind": "platform.remediation_plan_notice",
                "payload": {
                    "plan_reference": _governance_reference(plan),
                    "governance_case_id": plan.governance_case_id,
                    "status": plan.status,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "sequence_no": step.sequence_no,
                            "action_kind": step.action_kind,
                            "status": step.status,
                        }
                        for step in plan.steps
                    ],
                    "response_actions": response_actions,
                    "world_tick": world_tick,
                },
            },
        )

    def _remediation_audit_request(
        self,
        notice: Envelope,
        plan: RemediationPlan,
    ) -> Envelope | None:
        """Request external evidence for the next live pending World step."""

        if plan.status != "active":
            return None
        pending = next((step for step in plan.steps if step.status == "pending"), None)
        if pending is None:
            return None
        world_tick = self._world_client.read_logical_time(
            caller="platform:remediation"
        )
        payload = build_remediation_audit_request(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            plan_digest=plan.plan_digest,
            owner_merchant_id=plan.owner_merchant_id,
            step_id=pending.step_id,
            sequence_no=pending.sequence_no,
            action_kind=pending.action_kind,
            world_tick=world_tick,
        )
        return Envelope(
            msg_id=f"{notice.msg_id}:audit-request",
            ts=notice.ts,
            from_="platform:remediation",
            to=REMEDIATION_AUDITOR_SERVICE_ID,
            # The merchant notice is emitted first, while both Platform
            # responses remain children of the authenticated request received
            # by platform:remediation.  This preserves the causal rule that a
            # child sender must be its parent's recipient.
            in_reply_to=notice.in_reply_to,
            idempotency_key=f"{notice.idempotency_key}:audit-request",
            action={
                "kind": "platform.remediation_audit_request",
                "payload": payload,
            },
        )

    def _internal_request(
        self,
        env: Envelope,
        *,
        suffix: str,
        to_actor: str,
        action_kind: str,
        payload: dict[str, Any],
    ) -> Envelope:
        return Envelope(
            msg_id=f"{env.msg_id}:{suffix}",
            ts=env.ts,
            from_=self._ORCHESTRATOR_ACTOR,
            to=to_actor,
            in_reply_to=env.msg_id,
            idempotency_key=f"{env.idempotency_key}:{suffix}",
            action={"kind": action_kind, "payload": payload},
        )

    def _trusted_ack_target(self, env: Envelope, *, kind: str) -> str:
        if env.from_ != self._ORCHESTRATOR_ACTOR:
            return env.from_
        return {
            "platform.aggregate_reviews": "runtime:reviews",
            "platform.resolve_governance_case": "runtime:governance",
            "platform.create_remediation_plan": "runtime:remediation",
            "platform.verify_remediation_step": "runtime:remediation",
            "platform.apply_governance_reputation": "runtime:reputation",
        }.get(kind, "runtime:governance")

    def _history(self, env: Envelope) -> Envelope:
        if env.to != "platform:governance":
            raise SchemaError(
                "commerce.read_governance_history must target platform:governance"
            )
        if env.from_.split(":", 1)[0] not in {"buyer", "merchant"}:
            raise SchemaError("governance history requires a commerce actor")
        payload = _payload_mapping(env)
        _require_exact_fields(
            payload, {"record_kind", "stable_id"}, action=env.action["kind"]
        )
        record_kind = _required_text(payload, "record_kind")
        stable_id = _required_text(payload, "stable_id")
        rows = self._world_client.governance_history(
            record_kind, stable_id, caller=env.from_
        )
        return Envelope(
            msg_id=f"{env.msg_id}:governance-snapshot",
            ts=env.ts,
            from_="platform:governance",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.governance_snapshot",
                "payload": {
                    "resource": "governance_history",
                    "record_kind": record_kind,
                    "stable_id": stable_id,
                    "records": [_governance_safe_result(row) for row in rows],
                },
            },
        )

    @staticmethod
    def _updated(
        env: Envelope,
        *,
        service_actor: str,
        operation: str,
        result: Any,
        to_actor: str | None = None,
    ) -> Envelope:
        safe = {"operation": operation, **_governance_safe_result(result)}
        return Envelope(
            msg_id=f"{env.msg_id}:governance-updated",
            ts=env.ts,
            from_=service_actor,
            to=env.from_ if to_actor is None else to_actor,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={"kind": "platform.governance_updated", "payload": safe},
        )


class OrderLifecyclePolicy:
    """Commit non-financial order transitions through authoritative World APIs.

    The protocol sender is the only source of actor provenance. Payload fields
    that could claim a buyer, merchant, or forwarding actor are rejected; the
    stored order is read first and the full ``env.from_`` value must be an exact
    party match before World receives the command.
    """

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def handle(self, env: Envelope) -> Envelope:
        payload = _payload_mapping(env)
        _reject_provenance_fields(payload)
        order_id = _required_text(payload, "order_id")
        order = self._world_client.read(
            "orders", OrderId(order_id), caller="platform:psp"
        )
        if order is None:
            raise SchemaError(f"unknown order_id {order_id!r}")

        kind = str(env.action.get("kind", ""))
        if kind == "commerce.dispatch":
            self._require_exact_merchant(env, order)
            updated = self._world_client.dispatch_order(
                order_id=order.order_id,
                original_actor=env.from_,
            )
            operation = "dispatch"
        elif kind == "commerce.cancel_order":
            self._require_exact_party(env, order)
            updated = self._world_client.cancel_order(
                order_id=order.order_id,
                original_actor=env.from_,
            )
            operation = "cancel"
        elif kind == "commerce.mark_returned":
            self._require_exact_merchant(env, order)
            updated = self._world_client.mark_order_returned(
                order_id=order.order_id,
                original_actor=env.from_,
            )
            operation = "mark_returned"
        else:
            raise SchemaError(f"unsupported order lifecycle action {kind!r}")
        return _lifecycle_ack(
            env,
            service="platform:psp",
            payload={
                "operation": operation,
                "order_id": str(updated.order_id),
                "state": updated.state.value,
            },
        )

    def exchange(self, env: Envelope) -> Envelope:
        """Commit a like-for-like exchange without trusting buyer-supplied value.

        The request contains only immutable identifiers.  Buyer, merchant,
        quantity, currency, and unit price are copied from the authoritative
        original order, then World atomically validates the returned state,
        replacement ownership/stock, idempotency, and zero-ledger invariant.
        """
        payload = _payload_mapping(env)
        _reject_provenance_fields(payload)
        permitted = {
            "order_id",
            "replacement_sku_id",
            "replacement_order_id",
            "exchange_id",
        }
        unexpected = sorted(set(payload) - permitted)
        if unexpected:
            raise SchemaError(
                "commerce.request_exchange accepts only authoritative identity "
                f"references; unexpected fields: {unexpected}"
            )
        order_id = _required_text(payload, "order_id")
        replacement_sku_id = _required_text(payload, "replacement_sku_id")
        original = self._world_client.read(
            "orders", OrderId(order_id), caller="platform:psp"
        )
        if original is None:
            raise SchemaError(f"unknown order_id {order_id!r}")
        if env.from_ != str(original.buyer_id):
            raise LifecycleAuthorizationError(
                f"actor {env.from_!r} does not own order {order_id!r}"
            )
        replacement_order_id = str(
            payload.get("replacement_order_id") or f"{order_id}:replacement"
        )
        exchange_id = str(payload.get("exchange_id") or f"exchange:{order_id}")
        replacement = Order(
            order_id=OrderId(replacement_order_id),
            buyer_id=original.buyer_id,
            merchant_id=original.merchant_id,
            sku_id=SkuId(replacement_sku_id),
            qty=original.qty,
            agreed_price=original.agreed_price,
            state=OrderState.PROPOSED,
        )
        exchanged = self._world_client.exchange_order(
            exchange_id=ExchangeId(exchange_id),
            original_order_id=original.order_id,
            replacement_order=replacement,
            idempotency_key=env.idempotency_key,
        )
        return _lifecycle_ack(
            env,
            service="platform:psp",
            payload={
                "operation": "exchange",
                "exchange_id": str(exchanged.exchange_id),
                "order_id": str(exchanged.original_order_id),
                "replacement_order_id": str(exchanged.replacement_order_id),
                "replacement_sku_id": str(exchanged.replacement_sku_id),
                "state": "exchanged",
            },
        )

    @staticmethod
    def _require_exact_merchant(env: Envelope, order: Order) -> None:
        if env.from_ != str(order.merchant_id):
            raise LifecycleAuthorizationError(
                f"actor {env.from_!r} does not own order merchant_id "
                f"{str(order.merchant_id)!r}"
            )

    @staticmethod
    def _require_exact_party(env: Envelope, order: Order) -> None:
        if env.from_ not in {str(order.buyer_id), str(order.merchant_id)}:
            raise LifecycleAuthorizationError(
                f"actor {env.from_!r} is not a party to order {order.order_id}"
            )


class AdjudicatorPolicy:
    """Persist party-bound disputes and evidence-backed platform rulings.

    A filed dispute is always recorded first.  When its request cites a
    delivery-exception evidence id that was already emitted by the trusted PSP
    for the same order, the baseline policy deterministically rules for the
    order's buyer.  This keeps the decision reproducible and auditable without
    trusting an agent-supplied evidence string.  With no audit source (for
    example a bare unit construction), or with no matching trusted evidence,
    the dispute remains open for an explicit adjudicator action.
    """

    def __init__(
        self,
        *,
        world_client: "WorldClient",
        audit: "AuditLog | None" = None,
    ) -> None:
        self._world_client = world_client
        self._audit = audit

    def handle(self, env: Envelope) -> Envelope | list[Envelope]:
        kind = str(env.action.get("kind", ""))
        if kind in {"open_dispute", "platform.open_dispute"}:
            return self.open_dispute(env)
        if kind == "platform.rule_dispute":
            return self.rule_dispute(env)
        raise SchemaError(f"unsupported adjudicator action {kind!r}")

    def open_dispute(self, env: Envelope) -> Envelope | list[Envelope]:
        payload = _payload_mapping(env)
        _reject_provenance_fields(payload)
        order = self._order_for_dispute(payload)
        actor = env.from_
        if actor == str(order.buyer_id):
            against = str(order.merchant_id)
        elif actor == str(order.merchant_id):
            against = str(order.buyer_id)
        else:
            raise LifecycleAuthorizationError(
                f"actor {actor!r} is not a party to order {order.order_id}"
            )
        claimed_against = payload.get("against")
        if claimed_against not in (None, "") and str(claimed_against) != against:
            raise DisputeNotActionable(
                "against must name the opposing party derived from the stored order"
            )
        reason = str(payload.get("reason", payload.get("claim", ""))).strip()
        if not reason:
            raise SchemaError("platform.open_dispute requires reason or claim")
        dispute_id = str(
            payload.get("dispute_id") or f"dispute:{env.idempotency_key}"
        )
        dispute = Dispute(
            dispute_id=DisputeId(dispute_id),
            order_id=order.order_id,
            filed_by=AgentId(actor),
            against=AgentId(against),
            reason=reason,
            state=DisputeState.OPEN,
        )
        opened = self._world_client.open_dispute(
            dispute=dispute,
            original_actor=actor,
        )
        opened_ack = _lifecycle_ack(
            env,
            service="platform:adjudicator",
            payload={
                "operation": "open_dispute",
                "dispute_id": str(opened.dispute_id),
                "order_id": str(opened.order_id),
                "state": opened.state.value,
            },
        )
        ruling_notice = self._rule_from_trusted_delivery_evidence(
            env=env,
            order=order,
            dispute=opened,
        )
        return opened_ack if ruling_notice is None else [opened_ack, ruling_notice]

    def _rule_from_trusted_delivery_evidence(
        self,
        *,
        env: Envelope,
        order: Order,
        dispute: Dispute,
    ) -> Envelope | None:
        """Rule only evidence already committed to the protocol audit.

        The cited id alone is not authority.  It must match an earlier
        ``platform.lifecycle_updated`` envelope from the exact PSP service,
        name this exact order, and carry the structured
        ``delivery_exception`` status.  Replaying the same open request is
        idempotent because both World dispute creation and the single-ruling
        write accept the identical immutable object and reject conflicts.
        """

        payload = _payload_mapping(env)
        raw_cited = payload.get("evidence_ids", ())
        if isinstance(raw_cited, (str, bytes)) or not isinstance(
            raw_cited, (list, tuple)
        ):
            raise SchemaError("platform.open_dispute evidence_ids must be a list")
        cited = tuple(str(item).strip() for item in raw_cited)
        if any(not item for item in cited):
            raise SchemaError(
                "platform.open_dispute evidence_ids must contain non-empty strings"
            )
        trusted = self._trusted_delivery_evidence(order_id=str(order.order_id))
        matched = tuple(sorted(set(cited).intersection(trusted)))
        if not matched:
            return None

        dispute_suffix = str(dispute.dispute_id)
        if dispute_suffix.startswith("dispute:"):
            dispute_suffix = dispute_suffix.removeprefix("dispute:")
        ruling = Ruling(
            ruling_id=RulingId(f"ruling:{dispute_suffix}"),
            dispute_id=dispute.dispute_id,
            in_favor_of=order.buyer_id,
            rationale=(
                "trusted delivery_exception evidence: " + ",".join(matched)
            ),
            refund_amount=None,
        )
        ruled = self._world_client.rule_dispute(
            ruling=ruling,
            original_actor="platform:adjudicator",
        )
        return Envelope(
            msg_id=f"{env.msg_id}:ruling",
            ts=env.ts,
            from_="platform:adjudicator",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=f"{env.idempotency_key}:ruling",
            action={
                "kind": "platform.rule_dispute",
                "payload": {
                    "ruling_id": str(ruled.ruling_id),
                    "dispute_id": str(ruled.dispute_id),
                    "order_id": str(order.order_id),
                    "in_favor_of": str(ruled.in_favor_of),
                    "evidence_ids": list(matched),
                    "status": "ruled",
                },
            },
        )

    def _trusted_delivery_evidence(self, *, order_id: str) -> frozenset[str]:
        if self._audit is None:
            return frozenset()
        evidence: set[str] = set()
        for past in self._audit.replay():
            action = past.action if isinstance(past.action, dict) else {}
            payload = action.get("payload")
            if (
                past.from_ != "platform:psp"
                or str(action.get("kind", "")) != "platform.lifecycle_updated"
                or not isinstance(payload, dict)
                or str(payload.get("order_id", "")) != order_id
                or str(payload.get("status", "")) != "delivery_exception"
            ):
                continue
            evidence_id = str(payload.get("evidence_id", "")).strip()
            if evidence_id:
                evidence.add(evidence_id)
        return frozenset(evidence)

    def rule_dispute(self, env: Envelope) -> Envelope:
        if env.from_ != "platform:adjudicator":
            raise LifecycleAuthorizationError(
                "only platform:adjudicator may issue a ruling"
            )
        payload = _payload_mapping(env)
        _reject_provenance_fields(payload)
        dispute_id = _required_text(payload, "dispute_id")
        in_favor_of = _required_text(payload, "in_favor_of")
        rationale = _required_text(payload, "rationale")
        refund_value = payload.get("refund_amount")
        ruling = Ruling(
            ruling_id=RulingId(
                str(payload.get("ruling_id") or f"ruling:{dispute_id}")
            ),
            dispute_id=DisputeId(dispute_id),
            in_favor_of=AgentId(in_favor_of),
            rationale=rationale,
            refund_amount=(
                None if refund_value is None else _coerce_money(refund_value)
            ),
        )
        ruled = self._world_client.rule_dispute(
            ruling=ruling,
            original_actor=env.from_,
        )
        return _lifecycle_ack(
            env,
            service="platform:adjudicator",
            payload={
                "operation": "rule_dispute",
                "ruling_id": str(ruled.ruling_id),
                "dispute_id": str(ruled.dispute_id),
                "in_favor_of": str(ruled.in_favor_of),
                "status": "ruled",
            },
        )

    def _order_for_dispute(self, payload: dict[str, Any]) -> Order:
        order_id = payload.get("order_id")
        if order_id in (None, ""):
            txn_id = payload.get("tx_id", payload.get("txn_id"))
            if txn_id in (None, ""):
                raise SchemaError(
                    "platform.open_dispute requires order_id or tx_id"
                )
            receipt = self._world_client.read(
                "ledger", TxnId(str(txn_id)), caller="platform:adjudicator"
            )
            if receipt is None:
                raise SchemaError(f"unknown tx_id {str(txn_id)!r}")
            order_id = receipt.order_id
        order = self._world_client.read(
            "orders", OrderId(str(order_id)), caller="platform:adjudicator"
        )
        if order is None:
            raise SchemaError(f"unknown order_id {str(order_id)!r}")
        return order


class _AggregatorPolicyError(Exception):
    """Narrow actor-facing refusal from the governed search boundary.

    This marker is deliberately distinct from ``SchemaError`` and
    ``WorldError``.  Only deterministic mandate validation performed before a
    governed search mutates World may raise it.  Unexpected World, transport,
    privacy, and protocol failures therefore remain infrastructure failures.
    """

    def __init__(
        self,
        reason_code: str,
        *,
        category: RejectionCategory = "state",
    ) -> None:
        self.reason_code = reason_code
        self.category = category
        super().__init__(reason_code)


class AggregatorPolicy:
    """Commerce Intelligence Layer aggregator (minimal).

    Handles ``commerce.search`` envelopes addressed to ``platform:aggregator``.
    Reads the catalog directly from ``World`` (this is the documented escape
    hatch for platform-internal components — see module docstring) and packages
    matching listings as a ``platform.rank_offers`` reply.

    Why not ``WorldTools(caller_id="platform:aggregator")``? ``WorldTools`` is
    the *agent-facing* read view that enforces caller scoping; the platform
    legitimately needs cross-agent visibility to rank offers, and the partition
    table already grants platform read-all. Reusing ``WorldTools`` here would
    add no security and would force it to special-case the platform caller.

    Stub status: no merchant fan-out (single-source: World). A future
    multi-merchant ranking policy would query each merchant for an actual
    bid; for the minimal Phase 4 demo, ``list_price`` is treated as the
    initial offer and any negotiation runs through the buyer's
    ``negotiation`` skill against the merchant directly.
    """

    def __init__(
        self,
        *,
        world_client: "WorldClient",
        ranking_policy: Any = None,
        governance_policy: Mapping[str, Any] | None = None,
        max_results: int | None = None,
    ) -> None:
        self._world_client = world_client
        self._ranking_policy = ranking_policy
        self._governance_policy = (
            None if governance_policy is None else dict(governance_policy)
        )
        self._max_results = max_results

    def search(self, env: Envelope) -> Envelope:
        payload = env.action.get("payload", {})
        if not isinstance(payload, dict):
            raise SchemaError("commerce.search payload must be an object")
        query = str(payload.get("query", ""))
        limit = max(0, int(payload.get("limit", 10)))
        if self._max_results is not None:
            limit = min(limit, self._max_results)

        filter_contract = str(payload.get("filter_contract", "")).strip()
        if filter_contract and filter_contract != "typed_constraints.v1":
            raise SchemaError(
                f"unsupported commerce.search filter_contract {filter_contract!r}"
            )
        typed_filters = (
            _typed_search_filters(payload.get("filters", {}))
            if filter_contract
            else None
        )
        if self._governance_policy is None:
            # Non-governed legacy scenarios predate persisted mandate
            # revisions.  Preserve their synthetic identifier exactly.
            mandate_id = str(
                payload.get("mandate_id") or f"legacy-mandate:{env.from_}"
            )
        else:
            # A governed ranking context is bound to an authoritative mandate
            # revision.  Validate that binding before creating a search
            # session, so an actor mistake cannot leave a partial World commit
            # and cannot surface later as a broad WorldError.
            mandate_id = self._validated_governed_mandate_id(env, payload)
        query_digest = canonical_digest(
            {
                "query": query,
                "limit": limit,
                "filter_contract": filter_contract,
                "filters": typed_filters or {},
            }
        )
        session_id = "search-session:" + canonical_digest(
            {
                "buyer_id": env.from_,
                "search_idempotency_key": env.idempotency_key,
            }
        )[:24]
        existing = self._world_client.read_search_session(
            session_id, caller="platform:aggregator"
        )
        if existing is not None:
            if (
                existing.buyer_id != env.from_
                or existing.mandate_id != mandate_id
                or existing.search_request_id != env.idempotency_key
                or existing.search_idempotency_key != env.idempotency_key
                or existing.query_digest != query_digest
            ):
                raise IdempotencyConflict(
                    "search request changed under an existing idempotency key"
                )
            return self._search_response(env, existing)

        current_tick = self._world_client.read_logical_time(
            caller="platform:aggregator"
        )
        expires_at_tick = current_tick + 32

        # Recall-oriented retrieval: OR-match content tokens against each
        # row's haystack (name + category + attributes), score by overlap,
        # rank desc. The buyer's discovery-search Stage 1-4 does the hard
        # filtering (budget, must_have, eta, stock) on the returned
        # candidates — the aggregator's job is to surface plausible matches,
        # not to enforce buyer-side constraints.
        #
        # Legacy searches intentionally leave ``filters`` to the buyer-side
        # decision skill.  Callers may opt into ``typed_constraints.v1`` for a
        # small, validated public predicate vocabulary.  The opt-in preserves
        # every historical scenario while making Platform-side query
        # reformulation executable against authoritative listing attributes.
        terms = _tokenize_query(query)

        # Build a bounded recall pool over VCP. Querying individual tokens keeps
        # the historical OR-recall semantics without fetching the full catalog.
        # Oversampling leaves room for out-of-stock rows while the public result
        # remains exactly top-k. Very long natural-language queries are capped so
        # request work scales with top-k, not with prompt length or B×M.
        pool_limit = max(limit * 5, limit)
        by_sku: dict[str, Any] = {}
        if terms:
            for term in sorted(terms)[:8]:
                for listing in self._world_client.search_catalog(
                    term,
                    caller="platform:aggregator",
                    limit=pool_limit,
                ):
                    by_sku[str(listing.sku_id)] = listing
        else:
            for listing in self._world_client.search_catalog(
                "",
                caller="platform:aggregator",
                limit=pool_limit,
            ):
                by_sku[str(listing.sku_id)] = listing
        all_listings = [by_sku[key] for key in sorted(by_sku)]
        ranked: list[tuple[int, Any]] = []
        for listing in all_listings:
            haystack = _listing_haystack(listing)
            if terms:
                score = sum(1 for t in terms if t in haystack)
                if score == 0:
                    continue
            else:
                score = 0
            ranked.append((score, listing))
        # Recall fallback: a non-empty query that matched NOTHING (e.g. a
        # content-poor catalog whose listings carry no descriptive name/category,
        # so a natural-language query scores zero everywhere) would otherwise
        # strand the buyer with an empty result. The aggregator's job is recall,
        # not buyer-side filtering — so surface every listing and let the buyer's
        # Stage 1-4 filter rigidly. Bounded by ``limit`` below.
        if terms and not ranked:
            fallback = self._world_client.search_catalog(
                "",
                caller="platform:aggregator",
                limit=pool_limit,
            )
            ranked = [(0, listing) for listing in fallback]
        # Sort by (descending score, ascending sku_id) for determinism.
        ranked.sort(key=lambda pair: (-pair[0], str(pair[1].sku_id)))
        listings = [listing for _, listing in ranked]
        if typed_filters is not None:
            listings = [
                listing
                for listing in listings
                if _listing_matches_typed_search_filters(listing, typed_filters)
            ]

        candidates: list[dict[str, Any]] = []
        candidate_limit = pool_limit if self._ranking_policy is not None else limit
        for listing in listings:
            if len(candidates) >= candidate_limit:
                break
            inv = self._world_client.read(
                "inventory", listing.sku_id, caller="platform:aggregator"
            )
            if inv is None or _available_qty(inv) <= 0:
                continue
            supply = self._world_client.read_supply_state(
                str(listing.sku_id), caller="platform:aggregator"
            )
            if supply.available_qty <= 0:
                continue
            candidate = {
                # Historical public alias remains stable for existing 1x1
                # buyers. Security comes from session membership + digest, not
                # from pretending this human-readable id is globally unique.
                "offer_id": f"agg:{listing.sku_id}",
                "merchant_id": str(listing.merchant_id),
                "sku_id": str(listing.sku_id),
                "unit_price": _to_cents_half_up(listing.list_price.amount),
                "unit_price_cents": _to_cents_half_up(listing.list_price.amount),
                "currency": listing.list_price.currency,
                "qty": 1,
                "fulfillment": {"method": "standard",
                                "eta_days": _fulfillment_eta_days(listing)},
                "claims": sorted(_extract_claims(listing.attributes)),
                "expires_at": f"world-tick:{expires_at_tick}",
                "catalog_revision": _catalog_revision_from_listing(listing),
                "inventory_revision": supply.version,
            }
            candidates.append(candidate)

        if self._ranking_policy is not None:
            source = {str(item["offer_id"]): item for item in candidates}
            ranked_candidates = self._ranking_policy(tuple(candidates), limit=limit)
            if not isinstance(ranked_candidates, list):
                raise TypeError("platform ranking policy must return a list")
            selected: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in ranked_candidates:
                if not isinstance(item, dict):
                    raise TypeError("platform ranking policy rows must be mappings")
                offer_id = str(item.get("offer_id", ""))
                if offer_id not in source:
                    raise ValueError(
                        f"platform ranking policy invented unknown offer_id {offer_id!r}"
                    )
                if offer_id in seen:
                    raise ValueError(
                        f"platform ranking policy returned duplicate offer_id {offer_id!r}"
                    )
                seen.add(offer_id)
                selected.append(dict(source[offer_id]))
                if len(selected) >= limit:
                    break
            candidates = selected

        session = seal_search_session(
            SearchSession(
                session_id=session_id,
                buyer_id=env.from_,
                mandate_id=mandate_id,
                # Transport message ids may be regenerated when the same
                # logical request crosses another VCP topology. Persist the
                # idempotency identity so World snapshots and certificates
                # remain stable across in-process and HTTP execution.
                search_request_id=env.idempotency_key,
                search_idempotency_key=env.idempotency_key,
                query_digest=query_digest,
                issued_at_tick=current_tick,
                expires_at_tick=expires_at_tick,
                offers=tuple(
                    OfferSnapshot(
                        offer_id=str(candidate["offer_id"]),
                        session_id=session_id,
                        buyer_id=env.from_,
                        mandate_id=mandate_id,
                        merchant_id=str(candidate["merchant_id"]),
                        sku_id=str(candidate["sku_id"]),
                        unit_price_cents=int(candidate["unit_price_cents"]),
                        currency=str(candidate["currency"]),
                        qty=int(candidate["qty"]),
                        catalog_revision=int(candidate["catalog_revision"]),
                        inventory_revision=int(candidate["inventory_revision"]),
                        issued_at_tick=current_tick,
                        expires_at_tick=expires_at_tick,
                    )
                    for candidate in candidates
                ),
            )
        )
        persisted = self._world_client.create_search_session(
            session=session,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        return self._search_response(env, persisted)

    def _validated_governed_mandate_id(
        self,
        env: Envelope,
        payload: Mapping[str, Any],
    ) -> str:
        raw_mandate_id = payload.get("mandate_id")
        if not isinstance(raw_mandate_id, str) or not raw_mandate_id.strip():
            raise _AggregatorPolicyError("search_mandate_missing")
        revisions = self._world_client.read_mandate_revisions(
            raw_mandate_id,
            caller="platform:aggregator",
        )
        if not revisions:
            raise _AggregatorPolicyError("search_mandate_not_found")
        latest = max(revisions, key=lambda revision: revision.revision)
        if latest.buyer_id != env.from_:
            raise _AggregatorPolicyError(
                "search_mandate_owner_mismatch",
                category="identity",
            )
        return raw_mandate_id

    def _search_response(self, env: Envelope, session: SearchSession) -> Envelope:
        candidates: list[dict[str, Any]] = []
        for offer in session.offers:
            typed = offer_snapshot_to_wire(offer)
            # Additive v1 aliases keep existing buyers operational.  The
            # revisioned fields above remain the authoritative contract.
            typed["unit_price"] = offer.unit_price_cents
            typed["expires_at"] = f"world-tick:{offer.expires_at_tick}"
            listing = self._world_client.read(
                "catalog", SkuId(offer.sku_id), caller="platform:aggregator"
            )
            typed["fulfillment"] = {
                "method": "standard",
                "eta_days": 0 if listing is None else _fulfillment_eta_days(listing),
            }
            typed["claims"] = (
                []
                if listing is None
                else sorted(_extract_claims(listing.attributes))
            )
            candidates.append(typed)
        ranking_reference: dict[str, str] | None = None
        ranking_projection: dict[str, Any] | None = None
        if self._governance_policy is not None and session.offers:
            context = self._world_client.persist_ranking_context(
                {
                    "request_id": session.session_id,
                    "mandate_id": session.mandate_id,
                    "candidate_sku_ids": [
                        offer.sku_id for offer in session.offers
                    ],
                    "ranked_sku_ids": [offer.sku_id for offer in session.offers],
                    "policy_id": str(self._governance_policy["policy_id"]),
                    "policy_version": int(
                        self._governance_policy["policy_version"]
                    ),
                },
                by_actor="platform:ranking",
                original_actor=env.from_,
                idempotency_key=f"ranking-context:{env.idempotency_key}",
            )
            safe_context = _governance_safe_result(context)
            ranking_reference = cast(
                "dict[str, str]", safe_context["references"]
            )
            ranking_projection = self._world_client.ranking_context_projection(
                context.context_id,
                caller="platform:aggregator",
            )
            if (
                ranking_projection["context_id"] != context.context_id
                or ranking_projection["context_digest"] != context.context_digest
            ):
                raise SchemaError(
                    "ranking projection does not match persisted context"
                )
        return Envelope(
            msg_id=f"{env.msg_id}:ranked",
            ts=env.ts,
            from_="platform:aggregator",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.rank_offers",
                "payload": {
                    "schema_version": "cwe.platform-rank-offers.v2",
                    "session_id": session.session_id,
                    "session_digest": session.session_digest,
                    "mandate_id": session.mandate_id,
                    "issued_at_tick": session.issued_at_tick,
                    "expires_at_tick": session.expires_at_tick,
                    "search_session": search_session_to_wire(session),
                    "candidates": candidates,
                    "ranking_context_reference": ranking_reference,
                    "ranking_context_projection": ranking_projection,
                },
            },
        )

    def certify(self, env: Envelope) -> Envelope:
        """Issue a ``platform.create_match_certificate`` for a buyer's accept_offer.

        Buyer's discovery-search Path B picks a winning offer from the
        aggregator's ``platform.rank_offers`` reply, then emits
        ``commerce.accept_offer`` to ``platform:aggregator``. The aggregator
        confirms the match by issuing a certificate back to the buyer. The
        buyer's ``purchase-confirmation`` Branch A picks up the certificate,
        runs the freshness check + authority gate, and emits
        ``platform.settle_payment`` to ``platform:psp``.

        Certification is a World-backed exact join.  The aggregator cannot
        certify a caller-invented offer, price, revision, actor, or order.
        """
        payload = env.action.get("payload", {})
        if not isinstance(payload, dict):
            raise SchemaError("commerce.accept_offer payload must be an object")
        offer_id = payload.get("offer_id")
        if not isinstance(offer_id, str) or not offer_id.strip():
            raise SchemaError("commerce.accept_offer requires a non-empty offer_id")

        session: SearchSession | None
        if "session_id" in payload:
            strict_fields = {
                "session_id",
                "session_digest",
                "offer_id",
                "offer_digest",
                "mandate_id",
                "order_id",
                "merchant_id",
                "sku_id",
                "unit_price_cents",
                "currency",
                "qty",
                "catalog_revision",
                "inventory_revision",
            }
            _require_exact_fields(
                payload,
                strict_fields,
                action="commerce.accept_offer",
            )
            acceptance = seal_match_acceptance(
                MatchAcceptance(
                    request_msg_id=env.idempotency_key,
                    idempotency_key=env.idempotency_key,
                    session_id=payload["session_id"],
                    session_digest=payload["session_digest"],
                    offer_id=offer_id,
                    offer_digest=payload["offer_digest"],
                    buyer_id=env.from_,
                    mandate_id=payload["mandate_id"],
                    order_id=payload["order_id"],
                    merchant_id=payload["merchant_id"],
                    sku_id=payload["sku_id"],
                    unit_price_cents=payload["unit_price_cents"],
                    currency=payload["currency"],
                    qty=payload["qty"],
                    catalog_revision=payload["catalog_revision"],
                    inventory_revision=payload["inventory_revision"],
                )
            )
            session = self._world_client.read_search_session(
                acceptance.session_id,
                caller="platform:aggregator",
            )
            if session is None:
                raise MatchAcceptanceRejected("unknown search session")
        else:
            # Backward compatibility is deliberately narrow: only one current
            # persisted session for this exact buyer/offer may be inferred.
            legacy_fields = {"offer_id"}
            if "mandate_id" in payload:
                legacy_fields.add("mandate_id")
            _require_exact_fields(
                payload,
                legacy_fields,
                action="legacy commerce.accept_offer",
            )
            supplied_mandate = payload.get("mandate_id")
            if supplied_mandate is not None and (
                not isinstance(supplied_mandate, str)
                or not supplied_mandate.strip()
            ):
                raise SchemaError(
                    "legacy commerce.accept_offer mandate_id must be non-empty text"
                )
            session = self._world_client.resolve_search_session(
                buyer_id=env.from_,
                offer_id=offer_id,
            )
            if session is None:
                raise MatchAcceptanceRejected(
                    "legacy accept_offer has no unique current search session"
                )
            matches = [offer for offer in session.offers if offer.offer_id == offer_id]
            if len(matches) != 1:
                raise MatchAcceptanceRejected(
                    "legacy accepted offer is not unique"
                )
            offer = matches[0]
            order_mandate = supplied_mandate or session.mandate_id
            acceptance = seal_match_acceptance(
                MatchAcceptance(
                    request_msg_id=env.idempotency_key,
                    idempotency_key=env.idempotency_key,
                    session_id=session.session_id,
                    session_digest=session.session_digest,
                    offer_id=offer.offer_id,
                    offer_digest=offer.offer_digest,
                    buyer_id=env.from_,
                    mandate_id=session.mandate_id,
                    order_id=f"ord-{order_mandate}-{offer.offer_id}",
                    merchant_id=offer.merchant_id,
                    sku_id=offer.sku_id,
                    unit_price_cents=offer.unit_price_cents,
                    currency=offer.currency,
                    qty=offer.qty,
                    catalog_revision=offer.catalog_revision,
                    inventory_revision=offer.inventory_revision,
                )
            )

        certificate = self._world_client.issue_match_certificate(
            acceptance=acceptance,
            original_actor=env.from_,
        )
        return Envelope(
            msg_id=f"{env.msg_id}:cert",
            ts=env.ts,
            from_="platform:aggregator",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.create_match_certificate",
                "payload": match_certificate_to_wire(certificate),
            },
        )


class SupplyPolicy:
    """Versioned supply reads and owner/trusted supply-event mediation."""

    def __init__(
        self,
        *,
        world_client: "WorldClient",
        authority_ttl_ticks: int = (
            DEFAULT_SUPPLY_PURCHASE_AUTHORITY_TTL_TICKS
        ),
    ) -> None:
        self._world_client = world_client
        self._authority_ttl_ticks = (
            validate_supply_purchase_authority_ttl_ticks(
                authority_ttl_ticks
            )
        )

    def handle(self, env: Envelope) -> Envelope:
        kind = str(env.action.get("kind", ""))
        payload = _payload_mapping(env)
        if kind == "commerce.read_supply_state":
            return self._read(env, payload)
        if kind == "commerce.update_supply":
            if env.from_.split(":", 1)[0] != "merchant":
                raise SchemaError("commerce.update_supply requires a merchant actor")
            return self._apply(env, payload, original_actor=env.from_)
        if kind == "platform.apply_supply_event":
            if env.from_ != "runtime:supply":
                raise SchemaError(
                    "platform.apply_supply_event requires runtime:supply"
                )
            return self._apply(env, payload, original_actor=env.from_)
        raise SchemaError(f"unsupported supply action {kind!r}")

    def _read(self, env: Envelope, payload: Mapping[str, Any]) -> Envelope:
        raw = payload.get("sku_ids")
        if not isinstance(raw, list) or not raw or len(raw) > 64:
            raise SchemaError("supply read requires 1..64 sku_ids")
        sku_ids = tuple(str(value) for value in raw)
        if any(not value for value in sku_ids) or len(set(sku_ids)) != len(sku_ids):
            raise SchemaError("supply read sku_ids must be non-empty and unique")
        states = tuple(
            self._world_client.read_supply_state(
                sku_id,
                caller="platform:supply",
            )
            for sku_id in sku_ids
        )
        if env.from_.split(":", 1)[0] == "merchant" and any(
            str(row.merchant_id) != env.from_ for row in states
        ):
            raise SchemaError("merchant may read only its own exact supply state")
        purchase_options: list[dict[str, Any]] = []
        if env.from_.split(":", 1)[0] == "buyer":
            authorities = self._world_client.issue_supply_purchase_authorities(
                sku_ids,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
                ttl_ticks=self._authority_ttl_ticks,
            )
            if len(authorities) != len(states):
                raise SchemaError(
                    "World returned an incomplete supply authority batch"
                )
            purchase_options = [
                {
                    "authority_id": authority.authority_id,
                    "authority_digest": authority.authority_digest,
                    "sku_id": authority.sku_id,
                    "order_id": authority.order_id,
                    "merchant_id": authority.merchant_id,
                    "unit_price_cents": authority.unit_price_cents,
                    "currency": authority.currency,
                    "available_qty": authority.available_qty,
                    "supply_version": authority.supply_version,
                    "expires_at_tick": authority.expires_at_tick,
                }
                for authority in authorities
                if authority.available_qty > 0
            ]
        return Envelope(
            msg_id=f"{env.msg_id}:supply-state",
            ts=env.ts,
            from_="platform:supply",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.supply_state",
                "payload": {
                    "states": [_supply_state_payload(row) for row in states],
                    "purchase_options": purchase_options,
                },
            },
        )

    def _apply(
        self,
        env: Envelope,
        payload: Mapping[str, Any],
        *,
        original_actor: str,
    ) -> Envelope:
        sku_id = str(payload.get("sku_id", ""))
        if not sku_id:
            raise SchemaError("supply event requires sku_id")
        values: dict[str, int | None] = {}
        for name, default in (
            ("qty_delta", 0),
            ("eta_day", None),
            ("unit_price_cents", None),
            ("expected_version", None),
        ):
            value = payload.get(name, default)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise SchemaError(f"supply event {name} must be an integer")
            values[name] = cast("int | None", value)
        if (
            values["qty_delta"] == 0
            and values["eta_day"] is None
            and values["unit_price_cents"] is None
        ):
            raise SchemaError("supply event must change quantity, ETA, or price")
        state = self._world_client.apply_supply_event(
            sku_id=sku_id,
            qty_delta=int(values["qty_delta"] or 0),
            eta_day=values["eta_day"],
            unit_price_cents=values["unit_price_cents"],
            expected_version=values["expected_version"],
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )
        return Envelope(
            msg_id=f"{env.msg_id}:supply-applied",
            ts=env.ts,
            from_="platform:supply",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.supply_event_applied",
                "payload": {"supply_state": _supply_state_payload(state)},
            },
        )


def _supply_state_payload(row: SupplyState) -> dict[str, Any]:
    return {
        "sku_id": str(row.sku_id),
        "merchant_id": str(row.merchant_id),
        "available_qty": row.available_qty,
        "reserved_qty": row.reserved_qty,
        "eta_day": row.eta_day,
        "unit_price_cents": row.unit_price_cents,
        "version": row.version,
    }


class FulfillmentPolicy:
    """Preserve merchant identity while World computes a scarce-stock batch."""

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def handle(self, env: Envelope) -> Envelope:
        kind = str(env.action.get("kind", ""))
        if kind == "commerce.read_shipment":
            return self._read_shipment(env)
        if kind == "commerce.resolve_shipment":
            return self._resolve_shipment(env)
        if kind == "platform.record_shipment_status":
            return self._record_shipment_status(env)
        if env.from_.split(":", 1)[0] != "merchant":
            raise SchemaError("commerce.allocate_fulfillment requires a merchant")
        payload = _payload_mapping(env)
        if "merchant_id" in payload:
            raise SchemaError("allocation merchant identity is derived from sender")
        allocation_id = str(payload.get("allocation_id", ""))
        sku_id = str(payload.get("sku_id", ""))
        raw_order_ids = payload.get("priority_order_ids")
        if not allocation_id or not sku_id:
            raise SchemaError("allocation requires allocation_id and sku_id")
        if not isinstance(raw_order_ids, list) or not raw_order_ids:
            raise SchemaError("allocation priority_order_ids must be a non-empty list")
        order_ids = tuple(OrderId(str(value)) for value in raw_order_ids)
        if any(not str(value) for value in order_ids) or len(set(order_ids)) != len(
            order_ids
        ):
            raise SchemaError("allocation order ids must be unique and non-empty")
        batch = self._world_client.allocate_orders_atomic(
            allocation_id=allocation_id,
            merchant_id=AgentId(env.from_),
            sku_id=SkuId(sku_id),
            priority_order_ids=order_ids,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        return Envelope(
            msg_id=f"{env.msg_id}:allocation-batch",
            ts=env.ts,
            from_="platform:fulfillment",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.allocation_batch",
                "payload": {"allocation_batch": _allocation_batch_payload(batch)},
            },
        )

    def _read_shipment(self, env: Envelope) -> Envelope:
        if env.from_.split(":", 1)[0] not in {"buyer", "merchant"}:
            raise SchemaError("shipment reads require a buyer or merchant")
        payload = _payload_mapping(env)
        shipment_id = ShipmentId(str(payload.get("shipment_id", "")))
        if not str(shipment_id):
            raise SchemaError("shipment read requires shipment_id")
        shipment = self._world_client.read_shipment(shipment_id)
        if env.from_ not in {
            str(shipment.buyer_id),
            str(shipment.merchant_id),
        }:
            raise SchemaError("actor is not a party to the shipment")
        return self._shipment_reply(
            env,
            shipment,
            kind="platform.shipment_state",
            suffix="shipment-state",
        )

    def _record_shipment_status(self, env: Envelope) -> Envelope:
        if env.from_ != "runtime:logistics":
            raise SchemaError(
                "platform.record_shipment_status requires runtime:logistics"
            )
        payload = _payload_mapping(env)
        shipment_id = ShipmentId(str(payload.get("shipment_id", "")))
        event_id = str(payload.get("event_id", ""))
        try:
            status = ShipmentStatus(str(payload.get("status", "")))
        except ValueError as exc:
            raise SchemaError("invalid shipment status") from exc
        if not str(shipment_id) or not event_id:
            raise SchemaError("shipment status requires shipment_id and event_id")
        shipment = self._world_client.record_shipment_status(
            shipment_id=shipment_id,
            event_id=event_id,
            status=status,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        return self._shipment_reply(
            env,
            shipment,
            kind="platform.shipment_state",
            suffix="shipment-status",
        )

    def _resolve_shipment(self, env: Envelope) -> Envelope:
        if env.from_.split(":", 1)[0] not in {"buyer", "merchant"}:
            raise SchemaError("shipment resolution requires a buyer or merchant")
        payload = _payload_mapping(env)
        shipment_id = ShipmentId(str(payload.get("shipment_id", "")))
        try:
            resolution = ShipmentResolution(str(payload.get("resolution", "")))
        except ValueError as exc:
            raise SchemaError("invalid shipment resolution") from exc
        raw_replacement = payload.get("replacement_sku_id")
        shipment = self._world_client.resolve_shipment(
            shipment_id=shipment_id,
            resolution=resolution,
            replacement_sku_id=(
                None
                if raw_replacement in (None, "")
                else SkuId(str(raw_replacement))
            ),
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        return self._shipment_reply(
            env,
            shipment,
            kind="platform.shipment_resolved",
            suffix="shipment-resolved",
        )

    def _shipment_reply(
        self,
        env: Envelope,
        shipment: Shipment,
        *,
        kind: str,
        suffix: str,
    ) -> Envelope:
        payload: dict[str, Any] = {"shipment": _shipment_payload(shipment)}
        if (
            kind == "platform.shipment_state"
            and env.from_.split(":", 1)[0] in {"buyer", "merchant"}
        ):
            payload["replacement_options"] = self._replacement_options(shipment)
        return Envelope(
            msg_id=f"{env.msg_id}:{suffix}",
            ts=env.ts,
            from_="platform:fulfillment",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": kind,
                "payload": payload,
            },
        )

    def _replacement_options(self, shipment: Shipment) -> list[dict[str, Any]]:
        """Return bounded, World-backed alternatives owned by the merchant."""

        listings = self._world_client.search_catalog(
            "",
            {"merchant_id": str(shipment.merchant_id)},
            caller="platform:fulfillment",
            limit=64,
        )
        options: list[dict[str, Any]] = []
        for listing in listings:
            if str(listing.sku_id) == str(shipment.original_sku_id):
                continue
            state = self._world_client.read_supply_state(
                str(listing.sku_id),
                caller="platform:fulfillment",
            )
            available = state.available_qty
            if available <= 0:
                continue
            options.append({
                "sku_id": str(listing.sku_id),
                "merchant_id": str(listing.merchant_id),
                "available_qty": available,
                "unit_price_cents": state.unit_price_cents,
                "currency": listing.list_price.currency,
                "supply_version": state.version,
            })
        return options


def _allocation_batch_payload(row: AllocationBatch) -> dict[str, Any]:
    return {
        "allocation_id": row.allocation_id,
        "merchant_id": str(row.merchant_id),
        "sku_id": str(row.sku_id),
        "priority_order_ids": [str(value) for value in row.priority_order_ids],
        "allocations": [
            {
                "order_id": str(value.order_id),
                "buyer_id": str(value.buyer_id),
                "merchant_id": str(value.merchant_id),
                "sku_id": str(value.sku_id),
                "requested_qty": value.requested_qty,
                "fulfilled_qty": value.fulfilled_qty,
                "backordered_qty": value.backordered_qty,
                "receipt_txn_id": (
                    None
                    if value.receipt_txn_id is None
                    else str(value.receipt_txn_id)
                ),
                "created_by": str(value.created_by),
                "idempotency_key": value.idempotency_key,
            }
            for value in row.allocations
        ],
        "created_by": str(row.created_by),
        "idempotency_key": row.idempotency_key,
    }


def _shipment_payload(row: Shipment) -> dict[str, Any]:
    return {
        "shipment_id": str(row.shipment_id),
        "order_id": str(row.order_id),
        "buyer_id": str(row.buyer_id),
        "merchant_id": str(row.merchant_id),
        "original_sku_id": str(row.original_sku_id),
        "status": row.status.value,
        "status_history": [
            {
                "event_id": value.event_id,
                "status": value.status.value,
                "logical_time": value.logical_time,
            }
            for value in row.status_history
        ],
        "resolution": (
            None if row.resolution is None else row.resolution.value
        ),
        "replacement_sku_id": (
            None
            if row.replacement_sku_id is None
            else str(row.replacement_sku_id)
        ),
        "version": row.version,
    }


class CartPolicy:
    """Stateless Platform adapter over World-owned cart authorization and checkout."""

    QUOTE_TTL_TICKS = 10

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def authorize(
        self, env: Envelope
    ) -> Envelope | list[Envelope]:
        """Create a buyer/principal request and notify its allowed merchants."""

        role = env.from_.split(":", 1)[0]
        if role not in {"buyer", "consumer"}:
            raise SchemaError(
                "cart quote authorization requires a buyer or principal actor"
            )
        payload = _payload_mapping(env)
        allowed = {
            "market_id",
            "mandate_id",
            "lines",
            "fill_policy",
            "backorder_policy",
            "request_ttl_ticks",
        }
        if set(payload) - allowed:
            raise SchemaError("cart quote authorization contains unsupported fields")
        market_id = payload.get("market_id")
        if not isinstance(market_id, str) or not market_id.strip():
            raise SchemaError("cart quote authorization requires market_id")
        ttl = payload.get("request_ttl_ticks", self.QUOTE_TTL_TICKS)
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
            raise SchemaError("request_ttl_ticks must be a positive integer")
        request = self._world_client.create_cart_quote_request(
            self._intent(payload),
            market_id=market_id,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
            request_ttl_ticks=ttl,
        )
        public = persistent_cart_quote_request_to_dict(request)
        # Show only the commercial scope the merchant must decide whether to
        # quote.  Buyer/principal/mandate and authoritative revision/digest
        # bindings stay inside World.  The merchant later submits only the
        # opaque request id; World reloads and revalidates the complete record.
        merchant_view = {
            "request_id": request.request_id,
            "lines": [
                {"sku_id": line.sku_id, "qty": line.qty}
                for line in request.lines
            ],
            "fill_policy": request.fill_policy,
            "backorder_policy": request.backorder_policy,
            "expires_at_tick": request.expires_at_tick,
        }
        replies = [
            Envelope(
                msg_id=f"{env.msg_id}:cart-request",
                ts=env.ts,
                from_="platform:checkout",
                to=env.from_,
                in_reply_to=env.msg_id,
                idempotency_key=env.idempotency_key,
                action={
                    "kind": "platform.cart_quote_request",
                    "payload": {"request": public},
                },
            )
        ]
        replies.extend(
            Envelope(
                msg_id=f"{env.msg_id}:cart-request:{index:03d}",
                ts=env.ts,
                from_="platform:checkout",
                to=merchant_id,
                in_reply_to=env.msg_id,
                idempotency_key=env.idempotency_key,
                action={
                    "kind": "platform.cart_quote_request",
                    "payload": {"request": merchant_view},
                },
            )
            for index, merchant_id in enumerate(
                request.allowed_merchant_ids, start=1
            )
        )
        return replies[0] if len(replies) == 1 else replies

    def quote(self, env: Envelope) -> Envelope | list[Envelope]:
        """Issue a World-persisted quote from compact actor input only."""

        payload = _payload_mapping(env)
        role = env.from_.split(":", 1)[0]
        if role == "merchant":
            allowed = {"request_id", "quote_ttl_ticks"}
            if set(payload) - allowed or "request_id" not in payload:
                raise SchemaError(
                    "merchant cart quote requires only request_id and optional ttl"
                )
            ttl = payload.get("quote_ttl_ticks", self.QUOTE_TTL_TICKS)
            if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
                raise SchemaError("quote_ttl_ticks must be a positive integer")
            quote = self._world_client.issue_cart_quote_from_request(
                str(payload["request_id"]),
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
                quote_ttl_ticks=ttl,
            )
        elif role == "buyer":
            allowed = {
                "market_id",
                "mandate_id",
                "lines",
                "fill_policy",
                "backorder_policy",
                "quote_ttl_ticks",
            }
            if set(payload) - allowed:
                raise SchemaError("buyer cart quote contains unsupported fields")
            market_id = payload.get("market_id")
            if not isinstance(market_id, str) or not market_id.strip():
                raise SchemaError("buyer cart quote requires market_id")
            ttl = payload.get("quote_ttl_ticks", self.QUOTE_TTL_TICKS)
            if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
                raise SchemaError("quote_ttl_ticks must be a positive integer")
            quote = self._world_client.issue_cart_quote(
                self._intent(payload),
                market_id=market_id,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
                quote_ttl_ticks=ttl,
            )
        else:
            raise SchemaError("cart quote requester must be buyer or merchant")
        wire = persistent_cart_quote_to_dict(quote)
        merchant_wire = {
            "quote_id": quote.quote_id,
            "request_id": quote.request_id,
            "market_id": quote.market_id,
            "buyer_id": quote.buyer_id,
            "currency": quote.currency,
            "lines": wire["lines"],
            "subtotal_minor": quote.subtotal_minor,
            "charges": wire["charges"],
            "shipping_minor": quote.shipping_minor,
            "tax_minor": quote.tax_minor,
            "fee_minor": quote.fee_minor,
            "grand_total_minor": quote.grand_total_minor,
            "issued_at_tick": quote.issued_at_tick,
            "expires_at_tick": quote.expires_at_tick,
        }
        recipients = (
            ((env.from_, merchant_wire), (quote.buyer_id, wire))
            if role == "merchant" and env.from_ != quote.buyer_id
            else ((env.from_, wire),)
        )
        replies = [
            Envelope(
                msg_id=f"{env.msg_id}:cart-quote:{index:03d}",
                ts=env.ts,
                from_="platform:checkout",
                to=recipient,
                in_reply_to=env.msg_id,
                idempotency_key=env.idempotency_key,
                action={
                    "kind": "platform.cart_quote",
                    "payload": {"quote": recipient_wire},
                },
            )
            for index, (recipient, recipient_wire) in enumerate(
                recipients, start=1
            )
        ]
        return replies[0] if len(replies) == 1 else replies

    def checkout(self, env: Envelope) -> Envelope:
        """Forward only a quote id under the authenticated buyer delegation."""

        if env.from_.split(":", 1)[0] != "buyer":
            raise SchemaError("cart checkout buyer actor is required")
        payload = _payload_mapping(env)
        if set(payload) != {"quote_id"}:
            raise SchemaError("cart checkout requires exactly quote_id")
        quote_id = payload.get("quote_id")
        if not isinstance(quote_id, str) or not quote_id.strip():
            raise SchemaError("cart checkout quote_id must be non-empty")
        group = self._world_client.checkout_cart(
            quote_id=quote_id,
            original_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        return Envelope(
            msg_id=f"{env.msg_id}:cart-settled",
            ts=env.ts,
            from_="platform:checkout",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.cart_settlement",
                "payload": {"order_group": group},
            },
        )

    @staticmethod
    def _intent(payload: Mapping[str, Any]) -> dict[str, Any]:
        intent: dict[str, Any] = {
            "mandate_id": payload.get("mandate_id"),
            "lines": payload.get("lines"),
        }
        if "fill_policy" in payload:
            intent["fill_policy"] = payload["fill_policy"]
        if "backorder_policy" in payload:
            intent["backorder_policy"] = payload["backorder_policy"]
        return intent

class CatalogPolicy:
    """Forwards merchant commerce intents to authority-validating World APIs.

    The merchant lane has **no direct-to-world write privilege** for the
    catalog / inventory tables (PARTITION_ALLOW enforces this at the
    router; ``WorldService`` enforces it again at the world boundary
    per WORLD_DESIGN.md §6). The merchant emits the commerce intent —
    ``commerce.adjust_price`` / ``commerce.update_listing`` /
    ``commerce.receive_shipment`` — to ``platform:catalog``; this
    policy authenticates the original actor from ``Envelope.from_`` and passes
    only a compact intent plus the original idempotency key.  World reads the
    current row, validates ownership, stamps the revision, and atomically binds
    the result to actor-scoped idempotency.

    Three forwards:

    * ``commerce.adjust_price``     → ``world.apply_catalog_mutation``.
    * ``commerce.update_listing``   → ``world.apply_catalog_mutation`` for
                                       publish, update, and delist intents.
    * ``commerce.receive_shipment`` → ``world.update_inventory``
                                       (increment ``qty_available`` by
                                       ``qty``; preserves ``qty_reserved``).

    The policy is **deterministic** — same envelope in, same world
    write out. Validation errors come back as a ``commerce.respond_inquiry``
    decline (category=``policy``); the platform never silently drops a
    forward.
    """

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def handle(self, env: Envelope) -> Envelope:
        kind = str(env.action.get("kind", ""))
        payload = dict(env.action.get("payload", {}))
        sender_id = env.from_
        sender_role = sender_id.split(":", 1)[0]
        if kind == "commerce.get_sku":
            return self._get_sku(env, payload, sender_id)
        if sender_role != "merchant":
            return self._decline(env, "sender_not_permitted")
        # Ownership is always derived from env.from_; payload-supplied
        # merchant_id is a spoofing vector and must never be accepted.
        # Reject early at any of the three places it could appear so
        # downstream code never has to defend against it.
        if "merchant_id" in payload:
            return self._decline(env, "merchant_id_in_payload_forbidden")
        nested_fields = payload.get("fields")
        if isinstance(nested_fields, dict) and "merchant_id" in nested_fields:
            return self._decline(env, "merchant_id_in_payload_forbidden")
        try:
            if kind == "commerce.adjust_price":
                return self._adjust_price(env, payload, sender_id)
            if kind == "commerce.update_listing":
                return self._update_listing(env, payload, sender_id)
            if kind == "commerce.receive_shipment":
                return self._receive_shipment(env, payload, sender_id)
        except (
            _CatalogPolicyError,
            CatalogMutationRejected,
            IdempotencyConflict,
        ) as exc:
            return self._decline(env, str(exc))
        except (LifecycleAuthorizationError, WriteNotAuthorized) as exc:
            # World exposes the same exact actor ownership refusal as
            # WriteNotAuthorized in process and LifecycleAuthorizationError
            # over HTTP.  Do not swallow other authorization failures: those
            # may indicate a broken Platform-to-World service identity.
            if str(exc) in {"sender_not_permitted", "sku_not_owned_by_sender"}:
                return self._decline(env, str(exc))
            raise
        return self._decline(env, "unsupported_catalog_action")

    # --- per-action forwards ----------------------------------------

    def _get_sku(
        self,
        env: Envelope,
        payload: dict[str, Any],
        sender_id: str,
    ) -> Envelope:
        """Return one World-backed listing through the normal Platform path.

        Buyer catalog reads remain public.  A merchant request is deliberately
        owner-scoped because this endpoint is also used for merchant catalog
        operations and must not become a cross-merchant private work queue.
        The optional request id is correlation only; listing identity and
        ownership are always read from World.
        """

        if set(payload) - {"sku_id", "request_id"}:
            return self._listing_decline(
                env,
                request_id=env.in_reply_to or env.msg_id,
                reason="unknown_catalog_read_field",
            )
        try:
            sku_id = _required(payload, "sku_id", SkuId)
        except _CatalogPolicyError as exc:
            return self._listing_decline(
                env,
                request_id=env.in_reply_to or env.msg_id,
                reason=str(exc),
            )
        request_id = payload.get("request_id", env.in_reply_to or env.msg_id)
        if not isinstance(request_id, str) or not request_id.strip():
            return self._listing_decline(
                env,
                request_id=env.in_reply_to or env.msg_id,
                reason="invalid_request_id",
            )
        listing = self._world_client.read(
            "catalog", sku_id, caller="platform:catalog"
        )
        if listing is None:
            return self._listing_response(
                env,
                request_id=request_id,
                status="not_found",
                listing=None,
            )
        if sender_id.split(":", 1)[0] == "merchant":
            try:
                _assert_owner(sender_id, listing.merchant_id)
            except _CatalogPolicyError:
                return self._listing_decline(
                    env,
                    request_id=request_id,
                    reason="listing_not_owned_by_requester",
                )
        elif sender_id.split(":", 1)[0] != "buyer":
            return self._listing_decline(
                env,
                request_id=request_id,
                reason="sender_not_permitted",
            )
        return self._listing_response(
            env,
            request_id=request_id,
            status="ok",
            listing={
                "sku_id": str(listing.sku_id),
                "product_id": listing.product_id,
                "merchant_id": str(listing.merchant_id),
                "category": listing.category,
                "name": listing.name,
                "list_price_cents": _to_cents_half_up(
                    listing.list_price.amount
                ),
                "currency": listing.list_price.currency,
                "attributes": json.loads(
                    json.dumps(
                        dict(listing.attributes or {}),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
                "catalog_revision": _catalog_revision_from_listing(listing),
            },
        )

    def _adjust_price(self, env: Envelope, payload: "AdjustPricePayload | dict[str, Any]",
                      sender_id: str) -> Envelope:
        unsupported = set(payload) - {"sku_id", "list_price"}
        if unsupported:
            raise _CatalogPolicyError(
                "unsupported_catalog_payload_fields:"
                + ",".join(sorted(unsupported))
            )
        sku_id = _required(payload, "sku_id", SkuId)
        new_list_price = payload.get("list_price")
        if not isinstance(new_list_price, int) or isinstance(new_list_price, bool) or new_list_price <= 0:
            raise _CatalogPolicyError("invalid_list_price")
        intent = _normalize_catalog_actor_intent(
            {
                "op": "adjust_price",
                "sku_id": str(sku_id),
                "fields": {"list_price": new_list_price},
            }
        )
        self._world_client.apply_catalog_mutation(
            intent,
            original_actor=sender_id,
            idempotency_key=env.idempotency_key,
        )
        return self._ack(env, {
            "kind": "world.apply_catalog_mutation",
            "op": "adjust_price",
            "sku_id": str(sku_id),
            "list_price_minor": int(new_list_price),
            "status": "ok",
        })

    def _update_listing(self, env: Envelope, payload: "UpdateListingPayload | dict[str, Any]",
                        sender_id: str) -> Envelope:
        unsupported = set(payload) - {
            "op",
            "sku_id",
            "fields",
            "product",
            "op_id",
            # Correlation evidence remains on the audited actor envelope.  It
            # is not part of the authority-owned World mutation intent.
            "verification_source_id",
        }
        if unsupported:
            raise _CatalogPolicyError(
                "unsupported_catalog_payload_fields:"
                + ",".join(sorted(unsupported))
            )
        op = str(payload.get("op", "")).lower()
        sku_id = _required(payload, "sku_id", SkuId)
        fields = payload.get("fields", {})
        if fields is None and op == "delist":
            fields = {}
        if not isinstance(fields, Mapping):
            raise _CatalogPolicyError("catalog mutation fields must be an object")
        intent = _normalize_catalog_actor_intent(
            {"op": op, "sku_id": str(sku_id), "fields": dict(fields)}
        )
        self._world_client.apply_catalog_mutation(
            intent,
            original_actor=sender_id,
            idempotency_key=env.idempotency_key,
        )
        return self._ack(
            env,
            {
                "kind": "world.apply_catalog_mutation",
                "op": op,
                "sku_id": str(sku_id),
                "status": "ok",
            },
        )

    def _receive_shipment(self, env: Envelope, payload: "ReceiveShipmentPayload | dict[str, Any]",
                          sender_id: str) -> Envelope:
        sku_id = _required(payload, "sku_id", SkuId)
        qty = payload.get("qty")
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            raise _CatalogPolicyError("invalid_qty")
        listing = self._world_client.read("catalog", sku_id, caller="platform:catalog")
        if listing is not None:
            _assert_owner(sender_id, listing.merchant_id)
        existing_inv = self._world_client.read("inventory", sku_id, caller="platform:catalog")
        if existing_inv is None:
            merchant_id_str = str(listing.merchant_id) if listing is not None else sender_id.split(":", 1)[0]
            new_row = InventoryRow(
                sku_id=sku_id,
                merchant_id=AgentId(merchant_id_str),
                qty_available=int(qty),
                qty_reserved=0,
            )
        else:
            if isinstance(existing_inv, InventoryRow):
                available = existing_inv.qty_available
                reserved = existing_inv.qty_reserved
                merchant_id_str = str(existing_inv.merchant_id)
            else:
                available = int(existing_inv)
                reserved = 0
                merchant_id_str = (
                    str(listing.merchant_id) if listing is not None
                    else sender_id.split(":", 1)[0]
                )
            new_row = InventoryRow(
                sku_id=sku_id,
                merchant_id=AgentId(merchant_id_str),
                qty_available=int(available) + int(qty),
                qty_reserved=int(reserved),
            )
        self._world_client.write("inventory", sku_id, new_row,
                           by_action="world.update_inventory")
        return self._ack(env, {"kind": "world.update_inventory",
                                "sku_id": str(sku_id),
                                "qty_delta": int(qty),
                                "qty_available": new_row.qty_available,
                                "status": "ok"})

    # --- envelope helpers ------------------------------------------

    @staticmethod
    def _ack(env: Envelope, payload: dict[str, Any]) -> Envelope:
        return Envelope(
            msg_id=f"{env.msg_id}:catalog",
            ts=env.ts,
            from_="platform:catalog",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={"kind": "platform.catalog_ack",
                    "payload": payload},
        )

    @staticmethod
    def _listing_response(
        env: Envelope,
        *,
        request_id: str,
        status: str,
        listing: dict[str, Any] | None,
        decline_reason: str | None = None,
    ) -> Envelope:
        return Envelope(
            msg_id=f"{env.msg_id}:catalog-listing",
            ts=env.ts,
            from_="platform:catalog",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.catalog_listing",
                "payload": {
                    "schema_version": "cwe.platform-catalog-listing.v1",
                    "request_id": request_id,
                    "status": status,
                    "listing": listing,
                    "decline_reason": decline_reason,
                },
            },
        )

    @classmethod
    def _listing_decline(
        cls,
        env: Envelope,
        *,
        request_id: str,
        reason: str,
    ) -> Envelope:
        return cls._listing_response(
            env,
            request_id=request_id,
            status="declined",
            listing=None,
            decline_reason=reason,
        )

    @staticmethod
    def _decline(env: Envelope, reason: str) -> Envelope:
        return Envelope(
            msg_id=f"{env.msg_id}:catalog-decline",
            ts=env.ts,
            from_="platform:catalog",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={"kind": "commerce.respond_inquiry",
                    "payload": {"sku_id": str(env.action.get("payload", {}).get("sku_id", "")),
                                "product": "", "category": "policy",
                                "answer": {"decision": "declined",
                                            "decline_reason": reason}}},
        )


class _CatalogPolicyError(Exception):
    """Internal error raised inside ``CatalogPolicy`` to produce a decline."""


def _normalize_catalog_actor_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate model-authored catalog data before the authoritative VCP call.

    World repeats this normalization under its own transaction boundary.  The
    Platform copy distinguishes a malformed actor intent from an unexpected
    World ``SchemaError`` so only the former becomes a typed policy decline.
    """

    try:
        return dict(normalize_catalog_mutation_intent(value))
    except SchemaError as exc:
        raise _CatalogPolicyError(str(exc)) from exc


def _required(payload: dict[str, Any], key: str, coerce: Any) -> Any:
    if key not in payload or payload[key] in (None, ""):
        raise _CatalogPolicyError(f"missing_{key}")
    return coerce(str(payload[key]))


def _assert_owner(sender_id: str, merchant_id: AgentId) -> None:
    """Ensure the merchant addressed by ``sender_id`` owns the listing.

    Sender ``"merchant"`` and ``"merchant:<role>"`` both own rows whose
    ``merchant_id`` starts with ``"merchant"`` — the agent id may carry
    a role suffix; the listing's ``merchant_id`` is the side prefix.
    Multi-merchant deployments override by passing a fully-qualified
    ``merchant_id`` on each Listing; this baseline trusts the side-prefix
    match.
    """
    listing_owner = str(merchant_id)
    if listing_owner == "merchant":
        owns = sender_id.split(":", 1)[0] == "merchant"
    else:
        owns = sender_id == listing_owner
    if not owns:
        raise _CatalogPolicyError("sku_not_owned_by_sender")


_LEGACY_MERCHANT_SUBROLES = frozenset({
    "catalog", "fulfillment", "pricing", "retrieval", "support", "owner",
})


def _merchant_owner_id(sender_id: str) -> str:
    """Map old role-address senders to ``merchant`` and preserve real actor ids.

    The original protocol examples used addresses such as ``merchant:catalog``
    for subroles of one generic merchant. Population agents use ids such as
    ``merchant:m17``. Keeping this compatibility shim lets legacy actions retain
    ownership while multi-merchant listings receive an exact owner id.
    """
    side, sep, suffix = sender_id.partition(":")
    if side != "merchant":
        raise _CatalogPolicyError("sku_not_owned_by_sender")
    if not sep or suffix in _LEGACY_MERCHANT_SUBROLES:
        return "merchant"
    return sender_id


class ReputationPolicy:
    """Baseline reputation policy that writes a precomputed score to World."""

    def __init__(self, *, world_client: "WorldClient") -> None:
        self._world_client = world_client

    def update(self, env: Envelope) -> Envelope:
        payload = env.action.get("payload", {})
        if not isinstance(payload, dict):
            raise SchemaError("platform.update_reputation payload must be an object")
        if env.from_ == "platform:psp":
            merchant_id = str(payload.get("merchant_id", "")).strip()
            order_id = str(payload.get("order_id", "")).strip()
            txn_id = str(payload.get("txn_id", "")).strip()
            if not merchant_id or not order_id or not txn_id:
                raise SchemaError(
                    "PSP reputation update requires merchant_id, order_id, and txn_id"
                )
            score = self._world_client.apply_settlement_reputation(
                merchant_id=AgentId(merchant_id),
                order_id=OrderId(order_id),
                txn_id=TxnId(txn_id),
                original_actor=env.from_,
                source_request_id=env.msg_id,
                idempotency_key=env.idempotency_key,
            )
        else:
            score = _coerce_reputation_score(payload)
            self._world_client.write(
                "reputation",
                score.merchant_id,
                score,
                by_action="world.update_reputation",
            )
        return Envelope(
            msg_id=f"{env.msg_id}:reputation",
            ts=env.ts,
            from_="platform:reputation",
            to=str(payload.get("notify_to") or env.from_),
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={
                "kind": "platform.reputation_updated",
                "payload": {"merchant_id": score.merchant_id},
            },
        )

    def successful_settlement_update_request(
        self,
        *,
        merchant_id: str,
        order_id: str,
        txn_id: str,
        ts: str,
        correlation_id: str,
        idempotency_key: str,
    ) -> Envelope | None:
        """Advance an explicitly tracked reputation row after a real settle.

        Markets that do not seed reputation retain the historical behavior.
        A tracked merchant receives one deterministic five-star success sample;
        the World row and notification are both downstream of the authoritative
        settlement, so a proposal alone can never improve reputation.
        """
        if not merchant_id or not order_id or not txn_id:
            return None
        prior = self._world_client.read(
            "reputation", AgentId(merchant_id), caller="platform:reputation"
        )
        if prior is None:
            return None
        return Envelope(
            msg_id=f"{correlation_id}:reputation:update",
            ts=ts,
            from_="platform:psp",
            to="platform:reputation",
            in_reply_to=correlation_id,
            idempotency_key=f"{idempotency_key}:reputation",
            action={
                "kind": "platform.update_reputation",
                "payload": {
                    "merchant_id": merchant_id,
                    "order_id": order_id,
                    "txn_id": txn_id,
                    "notify_to": merchant_id,
                },
            },
        )

# --- Skills (deterministic baseline) ----------------------------------

class ForwardSearchSkill:
    """Forward ``search`` to the merchant and pass hits back to the buyer.

    Baseline: single merchant, no ranking, no merging. Replaced by an
    aggregator skill that ranks across N merchants in the multi-merchant
    variant.
    """

    name: str = "forward_search"
    group: SkillGroup = SkillGroup.DISCOVERY

    HANDLES: frozenset[str] = frozenset({"search", "get_sku"})

    def run(self, msg: "Envelope", ctx: "AgentContext") -> "Envelope | None":
        """Forward to the single registered merchant; return hits unchanged."""
        raise NotImplementedError(
            "build envelope to merchant with same payload; "
            "(multi-merchant: round-robin / rank across merchants)"
        )


class AtomicSettleSkill:
    """Wrap ``settle`` in one atomic transaction across ledger + inventory.

    The only code path that calls ``World.write(by_action='settle')``.
    """

    name: str = "atomic_settle"
    group: SkillGroup = SkillGroup.AUTHORIZATION

    HANDLES: frozenset[str] = frozenset({"settle"})

    def __init__(self, *, world_client: "WorldClient") -> None:
        """Hold the platform's VCP world client."""
        self._psp = PSPPolicy(world_client=world_client)

    def run(self, msg: "Envelope", ctx: "AgentContext") -> "Envelope | None":
        """Decrement inventory + append ledger entry atomically; return receipt envelope.

        Idempotent on ``(env.from_, env.idempotency_key)``.
        """
        return self._psp.settle(msg)


class RejectDisputeSkill:
    """Stub skill: reject dispute with the canned error."""

    name: str = "reject_dispute"
    group: SkillGroup = SkillGroup.OPERATIONS

    HANDLES: frozenset[str] = frozenset({"open_dispute"})

    def run(self, msg: "Envelope", ctx: "AgentContext") -> "Envelope | None":
        """Return error envelope with ``action.kind='dispute_rejected'``."""
        raise NotImplementedError(
            "build reply envelope to msg.from_ with action.kind='dispute_rejected', "
            "payload={'error': 'dispute_not_supported'}"
        )


# --- Factory ----------------------------------------------------------

def make_platform_agent(
    *,
    world_client: "WorldClient",
    inputs: "AgentInputs",
    skill_loader: "SkillLoader | None" = None,
    memory: "Memory | None" = None,
    agent_id: str = "platform",
) -> "Agent":
    """Provisional factory; **the Platform Service is not an Agent** per
    DESIGN.md §5.4. Kept here only so existing imports / tests continue to
    work until the ``platform/`` package lands.

    The real construction path will be ``Runtime.platform = PlatformService(...)``
    at runtime startup, with policy modules ablated independently of any
    ``Agent`` shape.
    """
    raise NotImplementedError(
        "memory = memory or InMemoryStore(); "
        "skill_loader = skill_loader or EmptySkillLoader(); "
        "return Agent(id=agent_id, skill_loader=skill_loader, "
        "enabled_skills=skill_loader.manifests(), memory=memory, inputs=inputs, "
        "inbound_action_kinds=PLATFORM_INBOUND, outbound_action_kinds=PLATFORM_OUTBOUND); "
        "# TODO: replace with PlatformService construction once platform/ lands"
    )


_PROVENANCE_FIELDS = frozenset({
    "buyer_id",
    "merchant_id",
    "filed_by",
    "original_actor",
    "by_actor",
})

_TIME_AUTHORITY_FIELDS = frozenset({
    "ts",
    "timestamp",
    "today",
    "current_time",
    "current_tick",
    "logical_time",
    "delivered_at",
    "settled_at_tick",
    "dispatched_at_tick",
    "return_authorized_at_tick",
    "return_window_ticks",
    "deadline_tick",
})


def _payload_mapping(env: Envelope) -> dict[str, Any]:
    payload = env.action.get("payload", {})
    if not isinstance(payload, dict):
        raise SchemaError("lifecycle payload must be an object")
    return dict(payload)


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], *, action: Any
) -> None:
    actual = set(payload)
    if actual != expected:
        raise SchemaError(
            f"{action} requires exact fields; "
            f"missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(str(value) for value in actual - expected)!r}"
        )


def _required_mapping(
    payload: Mapping[str, Any], key: str, *, action: str
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SchemaError(f"{action}.{key} must be an object")
    if any(not isinstance(name, str) or not name for name in value):
        raise SchemaError(f"{action}.{key} field names must be non-empty strings")
    return dict(value)


def _governance_resolution_template(
    value: dict[str, Any], *, action: str
) -> dict[str, Any]:
    _require_exact_fields(
        value,
        {"resolution_kind", "policy_id", "policy_version"},
        action=f"{action}.resolution_template",
    )
    resolution_kind = _required_text(value, "resolution_kind")
    if resolution_kind not in {
        "violation_confirmed",
        "no_violation",
        "compliant_rejection_recorded",
    }:
        raise SchemaError(
            f"{action}.resolution_template has an unsupported resolution_kind"
        )
    policy_version = value.get("policy_version")
    if (
        isinstance(policy_version, bool)
        or not isinstance(policy_version, int)
        or policy_version < 1
    ):
        raise SchemaError(
            f"{action}.resolution_template.policy_version must be positive"
        )
    return {
        "resolution_kind": resolution_kind,
        "policy_id": _required_text(value, "policy_id"),
        "policy_version": policy_version,
    }


def _governance_remediation_template(
    value: dict[str, Any], *, action: str
) -> dict[str, str]:
    _require_exact_fields(
        value,
        {"blueprint_id", "sku_id"},
        action=f"{action}.remediation_template",
    )
    return {
        "blueprint_id": _required_text(value, "blueprint_id"),
        "sku_id": _required_text(value, "sku_id"),
    }


def _governance_safe_result(value: Any) -> dict[str, Any]:
    """Project a typed World result without exposing private policy inputs."""

    extra: dict[str, Any]
    if isinstance(value, AdsCampaignTerms):
        kind, stable_id, version, digest, status = (
            "ads_campaign_terms",
            value.campaign_id,
            1,
            value.terms_digest,
            "published",
        )
        extra = {}
    elif isinstance(value, ReviewAccountBinding):
        kind, stable_id, version, digest, status = (
            "review_account_binding",
            value.reviewer_id,
            1,
            value.binding_digest,
            "published",
        )
        extra = {}
    elif isinstance(value, ReputationPolicyRevision):
        kind, stable_id, version, digest, status = (
            "reputation_policy_revision",
            value.policy_id,
            value.revision,
            value.policy_digest,
            "published",
        )
        extra = {}
    elif isinstance(value, RemediationBlueprint):
        kind, stable_id, version, digest, status = (
            "remediation_blueprint",
            value.blueprint_id,
            1,
            value.blueprint_digest,
            "published",
        )
        extra = {}
    elif isinstance(value, Campaign):
        kind, stable_id, version, digest, status = (
            "campaign",
            value.campaign_id,
            value.version,
            value.campaign_digest,
            value.status,
        )
        extra = {
            "placements": [
                {
                    "placement_id": item.placement_id,
                    "sku_id": item.sku_id,
                    "disclosure_status": item.disclosure_status,
                    "disclosure_text": item.disclosure_text,
                }
                for item in value.placements
            ]
        }
    elif isinstance(value, ReviewEvidence):
        kind, stable_id, version, digest, status = (
            "review_evidence",
            value.review_id,
            value.version,
            value.evidence_digest,
            "recorded",
        )
        extra = {
            "sku_id": value.sku_id,
            "rating": value.rating,
            "verified_purchase": value.verified_purchase,
        }
    elif isinstance(value, ReviewAggregate):
        kind, stable_id, version, digest, status = (
            "review_aggregate",
            value.aggregate_id,
            value.version,
            value.aggregate_digest,
            "computed",
        )
        extra = {
            "sku_id": value.sku_id,
            "review_count": value.review_count,
            "verified_review_count": value.verified_review_count,
            "rating_sum": value.rating_sum,
            "verified_rating_sum": value.verified_rating_sum,
        }
    elif isinstance(value, MarketSignal):
        kind, stable_id, version, digest, status = (
            "market_signal",
            value.signal_id,
            value.version,
            value.signal_digest,
            "recorded",
        )
        extra = {"signal_kind": value.signal_kind}
    elif isinstance(value, GovernanceCase):
        kind, stable_id, version, digest, status = (
            "governance_case",
            value.case_id,
            value.version,
            value.case_digest,
            value.status,
        )
        extra = {
            "case_kind": value.case_kind,
            "resolution_code": value.resolution_code,
            "subject_merchant_ids": list(value.subject_merchant_ids),
        }
    elif isinstance(value, GovernanceResponseAttestation):
        kind, stable_id, version, digest, status = (
            "governance_response_attestation",
            value.response_id,
            1,
            value.attestation_digest,
            "recorded",
        )
        extra = {
            "case_id": value.case_id,
            "subject_merchant_id": value.subject_merchant_id,
            "response_kind": value.response_kind,
        }
    elif isinstance(value, GovernanceResolutionDecision):
        kind, stable_id, version, digest, status = (
            "governance_resolution_decision",
            value.decision_id,
            1,
            value.decision_digest,
            "recorded",
        )
        extra = {
            "case_id": value.case_id,
            "resolution_code": value.resolution_code,
        }
    elif isinstance(value, ReputationEvent):
        kind, stable_id, version, digest, status = (
            "reputation_event",
            value.merchant_id,
            value.version,
            value.event_digest,
            "recorded",
        )
        extra = {
            "event_id": value.event_id,
            "merchant_id": value.merchant_id,
            "event_kind": value.event_kind,
            "outcome_bps": value.outcome_bps,
        }
    elif isinstance(value, RemediationPlan):
        kind, stable_id, version, digest, status = (
            "remediation_plan",
            value.plan_id,
            value.version,
            value.plan_digest,
            value.status,
        )
        extra = {
            "governance_case_id": value.governance_case_id,
            "owner_merchant_id": value.owner_merchant_id,
            "steps": [
                {
                    "step_id": item.step_id,
                    "sequence_no": item.sequence_no,
                    "action_kind": item.action_kind,
                    "status": item.status,
                }
                for item in value.steps
            ],
        }
    elif isinstance(value, RankingContext):
        kind, stable_id, version, digest, status = (
            "ranking_context",
            value.context_id,
            value.version,
            value.context_digest,
            "recorded",
        )
        extra = {
            "request_id": value.request_id,
            "ranked_sku_ids": list(value.ranked_sku_ids),
        }
    else:
        raise SchemaError(
            f"unsupported governance result {type(value).__name__}"
        )
    return {
        "result_kind": kind,
        "result_id": stable_id,
        "version": version,
        "status": status,
        "references": {
            "record_kind": kind,
            "stable_id": stable_id,
            "record_digest": digest,
        },
        **extra,
    }


def _governance_reference(value: Any) -> dict[str, str]:
    reference = _governance_safe_result(value).get("references")
    if not isinstance(reference, dict) or set(reference) != {
        "record_kind",
        "stable_id",
        "record_digest",
    }:
        raise SchemaError("governance result has an invalid safe reference")
    if any(not isinstance(item, str) or not item for item in reference.values()):
        raise SchemaError("governance safe reference fields must be non-empty")
    return dict(reference)


def _after_sales_operation(result: Mapping[str, Any]) -> dict[str, Any]:
    if set(result) != {"disposition", "operation", "references"}:
        raise SchemaError(
            "World after-sales result requires exactly disposition, operation, "
            "and references"
        )
    operation = result.get("operation")
    if not isinstance(operation, dict):
        raise SchemaError("World after-sales operation must be an object")
    required = {
        "operation",
        "order_id",
        "result_table",
        "result_key",
        "result_digest",
    }
    if not required.issubset(operation):
        raise SchemaError("World after-sales operation is missing safe result fields")
    return operation


def _after_sales_references(result: Mapping[str, Any]) -> dict[str, str]:
    references = result.get("references")
    if not isinstance(references, dict):
        raise SchemaError("World after-sales references must be an object")
    if not references:
        raise SchemaError("World after-sales references must not be empty")
    unexpected = set(references).difference(_AFTER_SALES_ACK_REFERENCE_FIELDS)
    if unexpected:
        raise SchemaError(
            "World after-sales references contain unsafe fields: "
            + ", ".join(sorted(str(field) for field in unexpected))
        )
    if any(
        not isinstance(field, str)
        or not isinstance(value, str)
        or not value.strip()
        for field, value in references.items()
    ):
        raise SchemaError(
            "World after-sales references require non-empty string fields and values"
        )
    return dict(references)


def _verified_evidence_count_adjudication(
    projection: Mapping[str, Any], dispute_id: str
) -> tuple[str, str]:
    """Choose a ruling from verified evidence already persisted in World.

    Each distinct cited EvidenceRecord digest counts once for its submitting
    side.  More verified citations wins; equal support yields a split ruling.
    The rationale seals the exact decision-input digest so the ruling commit is
    causally tied to the World projection used by this replaceable baseline.
    """

    if projection.get("resource") != "after_sales_history":
        raise SchemaError("adjudication requires authoritative after-sales history")
    raw_records = projection.get("records")
    if not isinstance(raw_records, list):
        raise SchemaError("after-sales history records are missing")

    cases: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    for wrapper in raw_records:
        if not isinstance(wrapper, Mapping):
            raise SchemaError("after-sales history contains an invalid wrapper")
        table = wrapper.get("table")
        value = wrapper.get("value")
        if not isinstance(value, dict) or value.get("dispute_id") != dispute_id:
            continue
        if table == "dispute_cases":
            cases.append(value)
        elif table == "dispute_evidence":
            evidence_rows.append(value)
        elif table == "dispute_responses":
            response_rows.append(value)
    if not cases:
        raise SchemaError("adjudication cannot find the persisted dispute")
    latest = max(cases, key=lambda row: int(row.get("version", 0)))
    filer_id = str(latest.get("filed_by_id", ""))
    respondent_id = str(latest.get("against_id", ""))
    if not filer_id or not respondent_id:
        raise SchemaError("persisted dispute parties are incomplete")

    cited: dict[str, set[tuple[str, str]]] = {
        filer_id: set(),
        respondent_id: set(),
    }
    for row in evidence_rows:
        actor = str(row.get("actor_id", ""))
        if actor not in cited:
            continue
        record_id = str(row.get("evidence_id", ""))
        digest = str(row.get("source_digest", ""))
        if record_id and digest:
            cited[actor].add((record_id, digest))
    for row in response_rows:
        actor = str(row.get("actor_id", ""))
        if actor not in cited:
            continue
        facts = row.get("facts")
        entries = facts.get("evidence") if isinstance(facts, Mapping) else None
        if not isinstance(entries, list):
            raise SchemaError("persisted dispute response evidence is invalid")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise SchemaError("persisted evidence citation is invalid")
            record_id = str(entry.get("record_id", ""))
            digest = str(entry.get("record_digest", ""))
            if record_id and digest:
                cited[actor].add((record_id, digest))

    filer_count = len(cited[filer_id])
    respondent_count = len(cited[respondent_id])
    if filer_count > respondent_count:
        operation = "rule_for_filer"
    elif respondent_count > filer_count:
        operation = "rule_for_respondent"
    else:
        operation = "rule_split"
    inputs = {
        "dispute_id": dispute_id,
        "dispute_digest": str(latest.get("record_digest", "")),
        "filer_id": filer_id,
        "respondent_id": respondent_id,
        "filer_evidence": sorted(cited[filer_id]),
        "respondent_evidence": sorted(cited[respondent_id]),
        "operation": operation,
    }
    decision_digest = hashlib.sha256(
        json.dumps(
            inputs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return (
        operation,
        "verified evidence count "
        f"filer={filer_count} respondent={respondent_count}; "
        f"decision_digest={decision_digest}",
    )


def _reject_provenance_fields(payload: dict[str, Any]) -> None:
    forbidden = sorted(_PROVENANCE_FIELDS.intersection(payload))
    if forbidden:
        raise SchemaError(
            "actor provenance comes from envelope.from; forbidden payload fields: "
            + ", ".join(forbidden)
        )


def _reject_time_claims(payload: dict[str, Any]) -> None:
    forbidden = sorted(_TIME_AUTHORITY_FIELDS.intersection(payload))
    if forbidden:
        raise SchemaError(
            "return timing comes from World logical time; forbidden payload fields: "
            + ", ".join(forbidden)
        )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"lifecycle payload requires a non-empty {key}")
    return value.strip()


def _lifecycle_ack(
    env: Envelope,
    *,
    service: str,
    payload: dict[str, Any],
) -> Envelope:
    return Envelope(
        msg_id=f"{env.msg_id}:lifecycle",
        ts=env.ts,
        from_=service,
        to=env.from_,
        in_reply_to=env.msg_id,
        idempotency_key=env.idempotency_key,
        action={"kind": "platform.lifecycle_updated", "payload": payload},
    )


def _coerce_order(payload: dict[str, Any]) -> Order:
    order = payload.get("order")
    if isinstance(order, Order):
        return order

    price = _coerce_money(payload.get("agreed_price", payload.get("price")))
    return Order(
        order_id=OrderId(str(payload["order_id"])),
        buyer_id=AgentId(str(payload["buyer_id"])),
        merchant_id=AgentId(str(payload["merchant_id"])),
        sku_id=SkuId(str(payload["sku_id"])),
        qty=int(payload["qty"]),
        agreed_price=price,
        state=OrderState(str(payload.get("state", OrderState.ACCEPTED.value))),
    )


def _coerce_receipt(payload: dict[str, Any], *, order: Order, env: Envelope) -> Receipt:
    receipt = payload.get("receipt")
    if isinstance(receipt, Receipt):
        return receipt
    return Receipt(
        txn_id=TxnId(str(payload.get("txn_id", f"txn:{order.order_id}"))),
        ts=str(payload.get("ts", env.ts)),
        order_id=order.order_id,
        buyer_id=order.buyer_id,
        merchant_id=order.merchant_id,
        sku_id=order.sku_id,
        qty=order.qty,
        price=order.agreed_price,
        idempotency_key=env.idempotency_key,
    )


def _coerce_partial_receipt(
    payload: dict[str, Any],
    *,
    order: Order,
    env: Envelope,
    fulfilled_qty: int,
) -> Receipt:
    receipt = payload.get("receipt")
    if isinstance(receipt, Receipt):
        return receipt
    return Receipt(
        txn_id=TxnId(str(payload.get("txn_id", f"txn:{order.order_id}"))),
        ts=str(payload.get("ts", env.ts)),
        order_id=order.order_id,
        buyer_id=order.buyer_id,
        merchant_id=order.merchant_id,
        sku_id=order.sku_id,
        qty=fulfilled_qty,
        price=order.agreed_price,
        idempotency_key=env.idempotency_key,
    )


def _coerce_money(value: Any) -> Money:
    if isinstance(value, Money):
        return value
    if isinstance(value, dict):
        return Money(Decimal(str(value["amount"])), str(value.get("currency", "USD")))
    return Money(Decimal(str(value)))


def _available_qty(row: InventoryRow | int) -> int:
    if isinstance(row, InventoryRow):
        return row.qty_available - row.qty_reserved
    return int(row)


def _catalog_revision_from_listing(listing: Listing) -> int:
    value = (listing.attributes or {}).get("catalog_revision", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError("catalog_revision must be a positive integer")
    return int(value)


_TYPED_SEARCH_NUMERIC_FILTERS: dict[str, tuple[str, str]] = {
    "shipping_days_max": ("shipping_days", "max"),
    "warranty_months_min": ("warranty_months", "min"),
    "return_days_min": ("return_days", "min"),
    "energy_score_min": ("energy_score", "min"),
}
_TYPED_SEARCH_FILTER_KEYS = frozenset({
    *_TYPED_SEARCH_NUMERIC_FILTERS,
    "required_features",
})


def _typed_search_filters(raw: Any) -> dict[str, Any]:
    """Validate the opt-in public search predicate vocabulary.

    Price or budget filters are deliberately absent.  A buyer's ceiling is
    private utility and must be applied by its own decision policy rather than
    copied into an outbound Platform request.
    """

    if not isinstance(raw, dict):
        raise SchemaError("typed commerce.search filters must be an object")
    unknown = sorted(set(map(str, raw)) - _TYPED_SEARCH_FILTER_KEYS)
    if unknown:
        raise SchemaError(f"unsupported typed commerce.search filter(s): {unknown!r}")
    output: dict[str, Any] = {}
    for key in _TYPED_SEARCH_NUMERIC_FILTERS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(f"typed commerce.search {key} must be a non-negative integer")
        output[key] = value
    features = raw.get("required_features", [])
    if (
        not isinstance(features, list)
        or not all(isinstance(item, str) and item.strip() for item in features)
        or len(features) != len(set(features))
    ):
        raise SchemaError(
            "typed commerce.search required_features must be a duplicate-free list "
            "of non-empty strings"
        )
    if features:
        output["required_features"] = list(features)
    return output


def _listing_matches_typed_search_filters(
    listing: Any,
    filters: Mapping[str, Any],
) -> bool:
    attrs = getattr(listing, "attributes", {}) or {}
    if not isinstance(attrs, dict):
        return False
    for key, (attribute, direction) in _TYPED_SEARCH_NUMERIC_FILTERS.items():
        if key not in filters:
            continue
        observed = attrs.get(attribute)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            return False
        expected = filters[key]
        if direction == "max" and observed > expected:
            return False
        if direction == "min" and observed < expected:
            return False
    required = filters.get("required_features", [])
    return all(_listing_has_feature(attrs, feature) for feature in required)


def _listing_has_feature(attributes: Mapping[str, Any], feature: str) -> bool:
    needle = feature.casefold().strip()
    direct = attributes.get(feature)
    if direct is True:
        return True
    features = attributes.get("features", ())
    if isinstance(features, (list, tuple, set, frozenset)) and any(
        str(value).casefold().strip() == needle for value in features
    ):
        return True
    return any(
        str(key).casefold().strip() == needle and bool(value)
        for key, value in attributes.items()
    )


_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "without", "from", "into", "onto",
    "this", "that", "these", "those", "are", "was", "were", "have",
    "has", "had", "but", "not", "all", "any", "can", "will", "would",
    "should", "could", "may", "might", "must", "shall",
    "within", "about", "between", "after", "before",
    "comfortable", "delivery", "days", "day", "your", "you", "our",
    "their", "his", "her",
})


def _tokenize_query(query: str) -> list[str]:
    """Lowercase alphanumeric tokens of len ≥ 3, dropping common stopwords.

    The aggregator does recall-oriented retrieval; the buyer's natural-
    language query ("wool crew socks with merino wool, comfortable for
    daily wear, delivery within 5 days") boils down to a handful of
    content tokens ({"wool", "crew", "socks", "merino", "wear"}) after
    this pass. Any token that survives is OR-matched against each
    listing's haystack. Internal punctuation is treated as a separator
    so e.g. "half-cent" → ["half", "cent"], matching a haystack that
    also splits on punctuation the same way.
    """
    # Replace any non-alphanumeric with a space, then split on whitespace.
    normalized = "".join(
        ch if ch.isalnum() else " " for ch in query.casefold()
    )
    out: list[str] = []
    for token in normalized.split():
        if len(token) < 3:
            continue
        if token in _QUERY_STOPWORDS:
            continue
        out.append(token)
    return out


def _listing_haystack(listing: "Any") -> str:
    """Flatten a Listing's searchable fields into one lowercased string.

    Punctuation is stripped to match the query tokenizer's behavior:
    a query token like ``"half-cent"`` becomes ``"halfcent"`` after
    ``_tokenize_query``, and the haystack normalizes the same way so
    a listing name ``"Half-cent sock"`` is reachable.
    """
    parts = [
        str(listing.sku_id),
        listing.category,
        listing.name,
    ]
    attrs = listing.attributes or {}
    for k in sorted(attrs.keys()):
        parts.append(f"{k} {attrs[k]}")
    raw = " ".join(parts).casefold()
    # Replace any non-alphanumeric (other than space) with a space so
    # punctuation doesn't block substring matches.
    return "".join(ch if ch.isalnum() or ch == " " else " " for ch in raw)


def _to_cents_half_up(amount: Decimal) -> int:
    """Convert a Money decimal amount to integer minor units (half-up).

    Python's built-in ``round`` is banker's rounding (half to even), which
    surprises pricing math at the cent boundary. Decimal's ``quantize`` with
    ``ROUND_HALF_UP`` gives the intuitive "$0.005 → 1 cent" behavior.
    """
    cents = (amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(cents)


def _fulfillment_eta_days(listing: "Any") -> "int | None":
    """Resolve the offer's fulfillment ETA — the SEAM the rank_offers contract
    defines as a *merchant-provided fulfillment attribute*, not a hardwired
    listing field (4b–4d item B).

    v1 sources it from the catalog's ``shipping_days`` attribute; a future
    merchant fulfillment-policy override plugs in HERE, so it can change without
    touching the rank_offers schema (``fulfillment.eta_days`` stays the contract
    surface). Returns ``None`` when the merchant publishes no ETA — the offer then
    carries ``eta_days: null`` and the buyer/scorer treat delivery as
    UNVERIFIABLE rather than fabricate one (the old hardcoded ``4`` was exactly
    that fabrication, and it made every ``shipping_within_N`` mandate reject all
    offers at the delivery gate)."""
    raw = (getattr(listing, "attributes", {}) or {}).get("shipping_days")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _extract_claims(attrs: "dict[str, Any] | None") -> set[str]:
    """Project ``Listing.attributes`` into a flat set of normalized claim tokens.

    Only fields where the cross-vendor format is uniform enough to project
    safely:

      * ``material`` / ``Material`` — comma-separated everywhere
        ("wool, merino"). Split, lowercase, strip.
      * ``certifications`` / ``Certifications`` — same format, lower variance.

    ``tags`` are deliberately NOT projected — each brand uses a different
    convention (``allbirds::carbon-score => 5.9`` vs
    ``Brand_Principles:vegan`` vs space-separated). The buyer's Stage 3
    grounding picks up tag-based claims via ``world.get_listing``.
    """
    if not attrs:
        return set()
    claims: set[str] = set()
    material = attrs.get("material") or attrs.get("Material")
    if material:
        claims.update(t.strip().lower() for t in str(material).split(","))
    certs = attrs.get("certifications") or attrs.get("Certifications")
    if certs:
        claims.update(t.strip().lower() for t in str(certs).split(","))
    return {c for c in claims if c}


def _coerce_reputation_score(payload: dict[str, Any]) -> ReputationScore:
    score = payload.get("score")
    if isinstance(score, ReputationScore):
        return score
    return ReputationScore(
        merchant_id=AgentId(str(payload["merchant_id"])),
        rolling_avg=float(payload["rolling_avg"]),
        n_settled=int(payload["n_settled"]),
        n_disputed=int(payload["n_disputed"]),
    )
