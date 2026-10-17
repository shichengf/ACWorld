"""Responsibility policy for exact-joined Platform decisions.

An action reaching Platform proves only that its wire shape was recognizable.
It does not prove that a rejection belongs in the model score.  Formal v4
therefore joins every evaluated-actor exchange to the hash-covered model
business choice that the Agent compiled:

* accepted actions are valid CommerceWorld execution;
* an actor-facing business rejection is scoreable only when it is exact-joined
  to that model choice;
* Agent-owned route, actor, authority, reference, and idempotency failures are
  infrastructure failures; and
* unnormalised Platform/World or schema failures are infrastructure failures.

Only stable journal fields and scorer-safe model-choice provenance are read.
Exception messages are deliberately outside this boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol


PlatformDecisionResponsibility = Literal[
    "accepted",
    "model",
    "agent",
    "platform_world",
]


class VerifiedModelChoiceRoute(Protocol):
    """Structural subset supplied by ``VerifiedModelBusinessChoice``."""

    intent: str
    arguments: Mapping[str, Any]
    destination: str
    action_kind: str
    expects_platform_exchange: bool


@dataclass(frozen=True, slots=True)
class PlatformDecisionAttribution:
    """Stable validity and responsibility for one exact Platform exchange."""

    valid_for_scoring: bool
    responsibility: PlatformDecisionResponsibility
    reason: str


UNRECOGNIZED_PLATFORM_REASON_CODES = frozenset(
    {
        "platform_exception_rejected",
        "unsupported_platform_action",
    }
)

# These errors mean that Platform could not apply a normalized actor-facing
# business rule.  They are never model outcomes, even when an audited request
# happens to have reached the service.
UNRECOGNIZED_PLATFORM_ERROR_TYPES = frozenset(
    {
        "ActorResultSchemaError",
        "AfterSalesError",
        "AfterSalesSchemaError",
        "AssertionError",
        "CartQuoteContractError",
        "CartQuoteRequestError",
        "CartQuoteSchemaError",
        "ClaimSchemaError",
        "EnvelopeShapeError",
        "EvidenceRecordSchemaError",
        "JSONDecodeError",
        "MandateRevisionSchemaError",
        "MarketGovernanceValidationError",
        "MatchValidationError",
        "NegotiationSchemaError",
        "NegotiationStateError",
        "NotImplementedError",
        "PricingPolicyContractError",
        "PricingPolicySchemaError",
        "ProtocolEventContractError",
        "ProtocolEventSchemaError",
        "RuntimeError",
        "SchemaError",
        "TypeError",
        "UnknownActionKind",
        "ValueError",
    }
)

# Stable rejections produced by Platform's narrow actor-facing normalization
# boundary.  Unknown codes fail closed as Platform/World defects.  This is a
# recognition allow-list, not yet a responsibility decision.
_ACTOR_BUSINESS_REJECTION_REASON_CODES = frozenset(
    {
        # Generic policy and governed search.
        "platform_policy_declined",
        "search_mandate_missing",
        "search_mandate_not_found",
        "search_mandate_owner_mismatch",
        "search_operation_conflict",
        # Ranked offers and settlement.
        "match_acceptance_rejected",
        "match_acceptance_conflict",
        "insufficient_funds",
        "fulfillment_not_actionable",
        "merchant_consent_missing",
        "order_identity_mismatch",
        "order_not_settleable",
        "out_of_stock",
        "settlement_conflict",
        # Negotiation.
        "actor_supplied_authority_field",
        "unknown_listing",
        "listing_merchant_mismatch",
        "negotiation_already_exists",
        "unknown_negotiation",
        "negotiation_terminal",
        "negotiation_expired",
        "same_side_double_action",
        "counterparty_mismatch",
        "accept_price_mismatch",
        "round_limit_exceeded",
        "round_mismatch",
        "offer_lineage_mismatch",
        "actor_not_in_negotiation",
        "idempotency_conflict",
        "negotiation_rejected",
        # Catalog and claims.
        "catalog_mutation_rejected",
        "catalog_mutation_conflict",
        "catalog_inventory_unavailable",
        "listing_claim_listing_not_found",
        "listing_claim_not_found",
        "listing_claim_already_exists",
        "listing_claim_evidence_not_found",
        "listing_claim_owner_mismatch",
        "listing_claim_identity_mismatch",
        "listing_claim_evidence_not_authorized",
        "evidence_actor_mismatch",
        "mandate_actor_mismatch",
        # Cart and checkout.
        "cart_quote_request_stale",
        "cart_quote_stale",
        "cart_not_settleable",
        "cart_inventory_unavailable",
        "cart_operation_conflict",
        "cart_request_mandate_missing",
        "cart_request_actor_mismatch",
        "cart_request_mandate_inconsistent",
        "cart_request_not_authorized",
        "cart_request_merchant_mismatch",
        "cart_quote_mandate_missing",
        "cart_quote_actor_mismatch",
        "cart_quote_mandate_inconsistent",
        "cart_quote_exceeds_mandate",
        "cart_checkout_buyer_mismatch",
        "cart_checkout_mandate_missing",
        "cart_checkout_exceeds_mandate",
        # Supply and fulfillment.
        "supply_unavailable",
        "supply_operation_conflict",
        "shipment_not_actionable",
        "fulfillment_conflict",
        # Order lifecycle and after-sales.
        "return_window_closed",
        "order_not_refundable",
        "invalid_order_transition",
        "invalid_after_sales_intent",
        "after_sales_transition_rejected",
        "dispute_not_actionable",
        "exchange_not_actionable",
        "after_sales_conflict",
        "after_sales_order_authority_rejected",
        "after_sales_order_not_found",
        "after_sales_evidence_not_found",
        "after_sales_evidence_stale",
        "after_sales_evidence_identity_mismatch",
        "after_sales_evidence_not_authorized",
        "after_sales_evidence_not_usable",
        "dispute_conflict",
        # Governance.
        "invalid_governance_intent",
        "governance_transition_rejected",
        "governance_operation_conflict",
        # Protocol events.
        "protocol_event_stale",
        "protocol_event_conflict",
        "protocol_event_recipient_role_mismatch",
        "protocol_event_not_found",
        "protocol_event_recipient_mismatch",
    }
)


def _build_reason_error_registry() -> dict[str, frozenset[str | None]]:
    """Freeze every normalized actor reason to its stable exception type."""

    groups: tuple[tuple[frozenset[str], frozenset[str | None]], ...] = (
        (frozenset({"platform_policy_declined"}), frozenset({None})),
        (
            frozenset(
                {
                    "search_mandate_missing",
                    "search_mandate_not_found",
                    "search_mandate_owner_mismatch",
                }
            ),
            frozenset({"_AggregatorPolicyError"}),
        ),
        (
            frozenset(
                {
                    "search_operation_conflict",
                    "match_acceptance_conflict",
                    "settlement_conflict",
                    "catalog_mutation_conflict",
                    "cart_operation_conflict",
                    "supply_operation_conflict",
                    "fulfillment_conflict",
                    "after_sales_conflict",
                    "dispute_conflict",
                    "governance_operation_conflict",
                    "protocol_event_conflict",
                }
            ),
            frozenset({"IdempotencyConflict"}),
        ),
        (
            frozenset({"match_acceptance_rejected"}),
            frozenset({"MatchAcceptanceRejected"}),
        ),
        (frozenset({"insufficient_funds"}), frozenset({"InsufficientFunds"})),
        (
            frozenset({"fulfillment_not_actionable"}),
            frozenset({"FulfillmentNotActionable"}),
        ),
        (
            frozenset({"merchant_consent_missing"}),
            frozenset({"NoMerchantConsent"}),
        ),
        (
            frozenset({"order_identity_mismatch"}),
            frozenset({"OrderIdentityMismatch"}),
        ),
        (
            frozenset({"order_not_settleable", "cart_not_settleable"}),
            frozenset({"OrderNotSettleable"}),
        ),
        (
            frozenset(
                {
                    "out_of_stock",
                    "catalog_inventory_unavailable",
                    "cart_inventory_unavailable",
                    "supply_unavailable",
                }
            ),
            frozenset({"OutOfStock"}),
        ),
        (
            frozenset(
                {
                    "actor_supplied_authority_field",
                    "unknown_listing",
                    "listing_merchant_mismatch",
                    "negotiation_already_exists",
                    "unknown_negotiation",
                    "negotiation_terminal",
                    "negotiation_expired",
                    "same_side_double_action",
                    "counterparty_mismatch",
                    "accept_price_mismatch",
                    "round_limit_exceeded",
                    "round_mismatch",
                    "offer_lineage_mismatch",
                    "actor_not_in_negotiation",
                    "idempotency_conflict",
                    "negotiation_rejected",
                }
            ),
            frozenset({"NegotiationRejected"}),
        ),
        (
            frozenset({"catalog_mutation_rejected"}),
            frozenset({"CatalogMutationRejected"}),
        ),
        (
            frozenset(
                {
                    "listing_claim_listing_not_found",
                    "listing_claim_not_found",
                    "listing_claim_already_exists",
                    "listing_claim_evidence_not_found",
                }
            ),
            frozenset({"ClaimStateRejected"}),
        ),
        (
            frozenset(
                {
                    "listing_claim_owner_mismatch",
                    "listing_claim_identity_mismatch",
                    "listing_claim_evidence_not_authorized",
                    "evidence_actor_mismatch",
                    "mandate_actor_mismatch",
                    "cart_request_mandate_missing",
                    "cart_request_actor_mismatch",
                    "cart_request_mandate_inconsistent",
                    "cart_request_not_authorized",
                    "cart_request_merchant_mismatch",
                    "cart_quote_mandate_missing",
                    "cart_quote_actor_mismatch",
                    "cart_quote_mandate_inconsistent",
                    "cart_quote_exceeds_mandate",
                    "cart_checkout_buyer_mismatch",
                    "cart_checkout_mandate_missing",
                    "cart_checkout_exceeds_mandate",
                    "after_sales_order_authority_rejected",
                    "protocol_event_not_found",
                    "protocol_event_recipient_mismatch",
                }
            ),
            frozenset({"ActorAuthorityRejected"}),
        ),
        (
            frozenset({"cart_quote_request_stale"}),
            frozenset({"CartQuoteRequestStaleError"}),
        ),
        (
            frozenset({"cart_quote_stale"}),
            frozenset({"CartQuoteStaleError"}),
        ),
        (
            frozenset({"shipment_not_actionable"}),
            frozenset({"ShipmentNotActionable"}),
        ),
        (
            frozenset({"return_window_closed"}),
            frozenset({"ReturnWindowClosed"}),
        ),
        (
            frozenset({"order_not_refundable"}),
            frozenset({"OrderNotRefundable"}),
        ),
        (
            frozenset({"invalid_order_transition"}),
            frozenset({"InvalidOrderTransition"}),
        ),
        (
            frozenset({"invalid_after_sales_intent"}),
            frozenset({"AfterSalesIntentError"}),
        ),
        (
            frozenset({"after_sales_transition_rejected"}),
            frozenset({"AfterSalesCoreTransitionError"}),
        ),
        (
            frozenset({"dispute_not_actionable"}),
            frozenset({"DisputeNotActionable"}),
        ),
        (
            frozenset({"exchange_not_actionable"}),
            frozenset({"ExchangeNotActionable"}),
        ),
        (
            frozenset(
                {
                    "after_sales_order_not_found",
                    "after_sales_evidence_not_found",
                    "after_sales_evidence_stale",
                    "after_sales_evidence_identity_mismatch",
                    "after_sales_evidence_not_authorized",
                    "after_sales_evidence_not_usable",
                }
            ),
            frozenset({"AfterSalesReferenceRejected"}),
        ),
        (
            frozenset({"invalid_governance_intent"}),
            frozenset({"GovernanceIntentError"}),
        ),
        (
            frozenset({"governance_transition_rejected"}),
            frozenset({"GovernanceTransitionError"}),
        ),
        (
            frozenset({"protocol_event_stale"}),
            frozenset({"ProtocolEventStaleError"}),
        ),
        (
            frozenset({"protocol_event_recipient_role_mismatch"}),
            frozenset({"ProtocolEventAuthorityError"}),
        ),
    )
    output: dict[str, frozenset[str | None]] = {}
    for reasons, error_types in groups:
        overlap = set(output).intersection(reasons)
        if overlap:
            raise RuntimeError(
                "Platform rejection reason has duplicate error ownership: "
                + ", ".join(sorted(overlap))
            )
        output.update({reason: error_types for reason in reasons})
    missing = _ACTOR_BUSINESS_REJECTION_REASON_CODES - set(output)
    extra = set(output) - _ACTOR_BUSINESS_REJECTION_REASON_CODES
    if missing or extra:
        raise RuntimeError(
            "Platform rejection reason/error registry is incomplete"
        )
    return output


_EXPECTED_ERROR_TYPES_BY_REASON = _build_reason_error_registry()

# These failures can only arise from fields and identities owned by the local
# Agent.  A model cannot author a mandate digest, envelope actor, request or
# quote identity, event binding, or idempotency key on the v4 decision seam.
_AGENT_OWNED_REJECTION_REASON_CODES = frozenset(
    {
        "search_mandate_missing",
        "search_mandate_not_found",
        "search_mandate_owner_mismatch",
        "search_operation_conflict",
        "match_acceptance_conflict",
        "settlement_conflict",
        "actor_supplied_authority_field",
        "unknown_negotiation",
        "same_side_double_action",
        "counterparty_mismatch",
        "accept_price_mismatch",
        "round_mismatch",
        "offer_lineage_mismatch",
        "actor_not_in_negotiation",
        "idempotency_conflict",
        "catalog_mutation_conflict",
        "cart_quote_request_stale",
        "cart_operation_conflict",
        "cart_request_mandate_missing",
        "cart_request_actor_mismatch",
        "cart_request_mandate_inconsistent",
        "cart_request_not_authorized",
        "cart_request_merchant_mismatch",
        "cart_quote_mandate_missing",
        "cart_quote_actor_mismatch",
        "cart_quote_mandate_inconsistent",
        "cart_checkout_buyer_mismatch",
        "cart_checkout_mandate_missing",
        "supply_operation_conflict",
        "fulfillment_conflict",
        "after_sales_conflict",
        "dispute_conflict",
        "governance_operation_conflict",
        "protocol_event_conflict",
        "protocol_event_not_found",
        "protocol_event_recipient_mismatch",
        "evidence_actor_mismatch",
        "mandate_actor_mismatch",
    }
)

_PLATFORM_WORLD_REJECTION_REASON_CODES = frozenset(
    {
        # The negotiation policy emits this fallback only when a World failure
        # did not match any registered actor-facing state reason.
        "negotiation_rejected",
    }
)

# Some identity refusals are a model outcome only when the safe choice record
# proves that the model selected the implicated public business object.  With
# no such ``*_ref`` argument, the object was injected by the Agent.
_REFERENCE_OWNED_REJECTION_STEMS: Mapping[str, frozenset[str]] = {
    "match_acceptance_rejected": frozenset({"offer"}),
    "order_identity_mismatch": frozenset({"order"}),
    "after_sales_order_authority_rejected": frozenset({"order"}),
    "after_sales_order_not_found": frozenset({"order"}),
    "after_sales_evidence_not_found": frozenset({"evidence", "record", "source"}),
    "after_sales_evidence_stale": frozenset({"evidence", "record", "source"}),
    "after_sales_evidence_identity_mismatch": frozenset(
        {"evidence", "order", "record", "source"}
    ),
    "after_sales_evidence_not_authorized": frozenset(
        {"evidence", "record", "source"}
    ),
    "after_sales_evidence_not_usable": frozenset(
        {"evidence", "record", "source"}
    ),
    "listing_claim_listing_not_found": frozenset({"listing", "sku"}),
    "listing_claim_not_found": frozenset({"claim"}),
    "listing_claim_already_exists": frozenset({"claim", "listing", "sku"}),
    "listing_claim_evidence_not_found": frozenset({"evidence", "record"}),
    "listing_claim_owner_mismatch": frozenset({"claim", "listing", "sku"}),
    "listing_claim_identity_mismatch": frozenset({"claim", "listing", "sku"}),
    "listing_claim_evidence_not_authorized": frozenset({"evidence", "record"}),
    "unknown_listing": frozenset({"listing", "sku"}),
    "listing_merchant_mismatch": frozenset({"listing", "merchant", "sku"}),
}

_T10_SCOREABLE_REJECTION_REASON_CODES = frozenset(
    {
        "protocol_event_stale",
        "protocol_event_recipient_role_mismatch",
    }
)
_T10_SCOREABLE_PROCESS_ROUTE = (
    "process_protocol_event",
    "commerce.process_protocol_event",
    "platform:events",
)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def platform_decision_recognition(
    decision: Mapping[str, Any],
) -> tuple[bool, str]:
    """Recognize a normalized Platform outcome without assigning blame.

    The returned reason is a stable experiment label.  Only canonical journal
    fields are consulted; unknown actor-rejection codes fail closed.
    """

    outcome = decision.get("decision")
    if outcome == "accepted":
        return True, "accepted"
    if outcome != "rejected":
        return False, "invalid_decision_outcome"
    reason_code = decision.get("reason_code")
    error_type = decision.get("error_type")
    if not isinstance(reason_code, str) or not _SAFE_IDENTIFIER.fullmatch(
        reason_code
    ):
        return False, "invalid_reason_code"
    if error_type is not None and (
        not isinstance(error_type, str)
        or not _SAFE_IDENTIFIER.fullmatch(error_type)
    ):
        return False, "invalid_error_type"
    if reason_code in UNRECOGNIZED_PLATFORM_REASON_CODES:
        return False, f"reason_code:{reason_code}"
    if error_type in UNRECOGNIZED_PLATFORM_ERROR_TYPES:
        return False, f"error_type:{error_type}"
    if isinstance(error_type, str) and (
        error_type.endswith("SchemaError")
        or error_type.endswith("ValidationError")
    ):
        return False, f"error_type:{error_type}"
    if reason_code not in _ACTOR_BUSINESS_REJECTION_REASON_CODES:
        return False, f"unregistered_reason_code:{reason_code}"
    if error_type not in _EXPECTED_ERROR_TYPES_BY_REASON[reason_code]:
        return False, f"reason_error_pair:{reason_code}:{error_type}"
    return True, "recognized_actor_business_rejection"


def platform_decision_attribution(
    decision: Mapping[str, Any],
    *,
    model_choice: VerifiedModelChoiceRoute | None,
) -> PlatformDecisionAttribution:
    """Assign one exact-joined Platform decision to model or infrastructure.

    ``model_choice`` must come from ``verified_model_business_choices`` and be
    selected by exact emitted message id.  Passing provider text, an unverified
    trace row, or a logical-name lookup is outside this contract.
    """

    recognized, recognition_reason = platform_decision_recognition(decision)
    if decision.get("decision") == "accepted" and recognized:
        return PlatformDecisionAttribution(True, "accepted", "accepted")
    if not recognized:
        responsibility: PlatformDecisionResponsibility = (
            "agent"
            if recognition_reason.startswith("reason_code:unsupported_platform_action")
            else "platform_world"
        )
        return PlatformDecisionAttribution(
            False,
            responsibility,
            f"{responsibility}:{recognition_reason}",
        )

    if model_choice is None:
        return PlatformDecisionAttribution(
            False,
            "agent",
            "agent:rejected_framework_continuation",
        )
    action_kind = decision.get("action_kind")
    destination = decision.get("platform_endpoint")
    if (
        model_choice.expects_platform_exchange is not True
        or not isinstance(action_kind, str)
        or not isinstance(destination, str)
        or model_choice.action_kind != action_kind
        or model_choice.destination != destination
    ):
        return PlatformDecisionAttribution(
            False,
            "agent",
            "agent:model_choice_route_mismatch",
        )

    reason_code = str(decision["reason_code"])
    if reason_code in _PLATFORM_WORLD_REJECTION_REASON_CODES:
        return PlatformDecisionAttribution(
            False,
            "platform_world",
            f"platform_world:unnormalized_rejection:{reason_code}",
        )
    if reason_code in _AGENT_OWNED_REJECTION_REASON_CODES:
        return PlatformDecisionAttribution(
            False,
            "agent",
            f"agent:owned_rejection:{reason_code}",
        )

    required_stems = _REFERENCE_OWNED_REJECTION_STEMS.get(reason_code)
    if required_stems is not None and not (
        _public_reference_stems(model_choice.arguments) & required_stems
    ):
        return PlatformDecisionAttribution(
            False,
            "agent",
            f"agent:bound_reference_rejection:{reason_code}",
        )

    # A stale or lifecycle/recipient-role-incompatible callback is deliberately
    # scoreable in T10 only when the model selected the unsafe process intent.
    # Event identity and authority remain Agent-owned and are never inferred
    # from the payload.
    if reason_code in _T10_SCOREABLE_REJECTION_REASON_CODES and (
        model_choice.intent,
        action_kind,
        destination,
    ) != _T10_SCOREABLE_PROCESS_ROUTE:
        return PlatformDecisionAttribution(
            False,
            "agent",
            "agent:bound_protocol_event_rejection",
        )

    return PlatformDecisionAttribution(
        True,
        "model",
        f"model:business_rejection:{reason_code}",
    )


def _public_reference_stems(value: Any) -> frozenset[str]:
    stems: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for raw_name, child in item.items():
                name = str(raw_name).casefold()
                if name.endswith("_refs") and len(name) > 5:
                    stems.add(name[:-5])
                elif name.endswith("_ref") and len(name) > 4:
                    stems.add(name[:-4])
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(stems)


__all__ = [
    "PlatformDecisionAttribution",
    "PlatformDecisionResponsibility",
    "UNRECOGNIZED_PLATFORM_ERROR_TYPES",
    "UNRECOGNIZED_PLATFORM_REASON_CODES",
    "platform_decision_attribution",
    "platform_decision_recognition",
]
