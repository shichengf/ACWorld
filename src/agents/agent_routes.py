"""Provider-neutral business intents and deterministic CommerceWorld compilation.

The model chooses among business functions.  It never authors a VCP envelope,
actor identity, message id, idempotency key, or framework-owned authority
field.  The read-only prompt observation may still contain inbound VCP
metadata and action names because those are facts about the current event, not
fields accepted back from model output.  An explicit participant-message intent
may ask for a business recipient, while every Platform/service destination is
fixed by :class:`AgentRouteRegistry`.

The registry validates each typed business decision and compiles actions into the
existing actor-terminal contract.  No function in this module can call
Platform or mutate World directly.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents.commerce_grounding import (
    CATALOG_UPDATE_ROUTE,
    GROUNDED_COMMERCE_AUTHORITY_V1,
    LISTING_CLAIM_ROUTE,
    ResolvedAuthorityPrerequisites,
)
from agents.agent_phase import RouteBinding, RouteRegistry
from agents.decision_errors import (
    DeterministicCompilerError,
    FrameworkAuthorityError,
    ModelBusinessDecisionError,
    PlatformContractError,
)
from protocol.actor_terminal import (
    BENCHMARK_PLATFORM_ACTOR_ROUTES,
    ActorTerminalContractError,
    preview_actor_terminal_action,
)
from protocol.actions import ActionKind, PARTITION_ALLOW
from protocol.envelope import Envelope


_INTENT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FINISH_INTENT = "agent_finish_turn"
INBOUND_SENDER_DESTINATION = "@inbound_sender"
_SHIPMENT_WAIT_INTENT = "act_wait_for_current_shipment"
_SHIPMENT_REPLACE_INTENT = "act_replace_current_shipment"
_SHIPMENT_REFUND_INTENT = "act_refund_current_shipment"
_CLAIM_PUBLISH_INTENT = "act_publish_owned_listing_claim"
_CLAIM_CORRECT_INTENT = "act_correct_owned_listing_claim"
_CLAIM_RETRACT_INTENT = "act_retract_owned_listing_claim"
_CLAIM_VARIANTS = frozenset({"claim_publish", "claim_correct", "claim_retract"})
_CLAIM_COMPILATION_SCHEMA_V1 = "cwe.agent-claim-compilation-templates.v1"

# These Platform operations all consume an order business reference.  In a
# public benchmark phase the reference comes from authenticated task/Platform
# authority, not from an arbitrary provider string.  The model chooses the
# business operation (and, for reconciliation, its reason); Agent owns the
# concrete order binding.
AFTER_SALES_ORDER_BOUND_OPERATIONS = frozenset(
    {
        "request_ledger_reconciliation",
        "read_payment_history",
        "read_ledger_history",
        "read_packing_history",
        "read_after_sales_history",
    }
)
_AFTER_SALES_RECONCILIATION_OPERATION = "request_ledger_reconciliation"


class SemanticDecisionError(ModelBusinessDecisionError):
    """A model-authored intent is not a valid business decision.

    Framework authority, Platform contract, and deterministic compiler failures
    use their dedicated unscoreable exception classes and must never enter the
    model-repair loop.
    """


@dataclass(frozen=True, slots=True)
class AgentBusinessIntentSpec:
    """One Agent-private intent schema compiled from a model business decision."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    category: str = "action"
    business_parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not _INTENT_NAME_RE.fullmatch(self.name):
            raise ValueError(f"invalid Agent business intent name: {self.name!r}")
        if not self.description.strip():
            raise ValueError("business intent description must be non-empty")
        if self.parameters.get("type") != "object":
            raise ValueError("business intent parameters must be an object schema")
        if (
            self.business_parameters is not None
            and self.business_parameters.get("type") != "object"
        ):
            raise ValueError("business parameters must be an object schema")
        if self.category not in {"read", "action", "control"}:
            raise ValueError("business intent category must be read, action, or control")


@dataclass(frozen=True, slots=True)
class WorldReadCall:
    """A business observation request normalized to a WorldTools name."""

    tool: str
    args: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SemanticAction:
    """A validated business decision ready for Agent-owned VCP construction."""

    destination: str
    action_kind: str
    payload: Mapping[str, Any]
    decision_rationale: str | None = None


def semantic_action_observation_class(
    action: SemanticAction,
    *,
    resolved_destination: str,
) -> str:
    """Classify one validated high-level action without claiming execution.

    The classification is deliberately about the Agent/compiler boundary.  It
    lets later evidence analysis distinguish a model that never produced a
    valid business intent from one that produced a valid business action whose
    route did not create an evaluated-actor-to-Platform exchange.  Runtime,
    Platform, World, Tracker, and replay evidence remain authoritative for
    whether the route was accepted or changed commerce state.
    """

    role = resolved_destination.split(":", 1)[0]
    if role == "platform":
        return "platform_business_action"
    if resolved_destination == "runtime:evidence":
        return "runtime_evidence_action"
    if role == "consumer" and action.action_kind == "delegate.reject_purchase":
        return "principal_reply_action"
    if role in _ACTOR_ROLES:
        return "participant_message_action"
    return "other_compiled_action"


@dataclass(frozen=True, slots=True)
class _ActionBinding:
    spec: AgentBusinessIntentSpec
    route: RouteBinding
    payload_mode: str = "arguments"
    semantic_variant: str | None = None

    def __post_init__(self) -> None:
        if self.route.source_name != self.spec.name:
            raise ValueError("registered route source must equal its intent source name")

    @property
    def operation(self) -> str:
        return self.route.operation

    @property
    def action_kind(self) -> str:
        return self.route.action_kind

    @property
    def destination(self) -> str:
        return self.route.destination

    @property
    def roles(self) -> frozenset[str]:
        return self.route.roles


@dataclass(frozen=True, slots=True)
class _AuthorityIntentProjection:
    """Model-visible schema produced from authenticated business authority.

    A projection deliberately has no action kind or destination.  The sole
    internal route remains the :class:`RouteBinding` registered beside the
    intent source; authority code can narrow a business schema but cannot
    create, redirect, or re-assert a wire route.
    """

    parameters: Mapping[str, Any]
    description: str | None = None


@dataclass(frozen=True, slots=True)
class AfterSalesOrderAuthority:
    """Actor-bound order choices for one public after-sales phase.

    ``task_order_ids`` preserves the ordering in the actor-visible task
    business input.  ``current_order_ids`` is the subset that is legal now.
    A sequential reconciliation phase normally has exactly one current order;
    that singleton is framework-bound and therefore omitted from model
    arguments.  The tuple form deliberately supports future phases where
    choosing between several simultaneously legal public order references is
    itself part of the business decision.
    """

    actor_id: str
    task_order_ids: tuple[str, ...]
    current_order_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or self.actor_id.split(":", 1)[0] not in {
            "buyer",
            "merchant",
        }:
            raise FrameworkAuthorityError("after-sales order authority has an invalid actor")
        if (
            not self.task_order_ids
            or any(
                not isinstance(order_id, str) or not order_id.strip()
                for order_id in self.task_order_ids
            )
            or len(self.task_order_ids) != len(set(self.task_order_ids))
        ):
            raise FrameworkAuthorityError("after-sales task order authority is malformed")
        if (
            any(
                not isinstance(order_id, str) or not order_id.strip()
                for order_id in self.current_order_ids
            )
            or len(self.current_order_ids) != len(set(self.current_order_ids))
            or not set(self.current_order_ids).issubset(self.task_order_ids)
        ):
            raise FrameworkAuthorityError("current after-sales order authority is malformed")

    def bind(self, arguments: Mapping[str, Any]) -> str:
        """Resolve the provider's optional public reference to one order."""

        if not self.current_order_ids:
            raise FrameworkAuthorityError("after-sales order authority has no current order")
        supplied = arguments.get("order_id")
        if len(self.current_order_ids) == 1:
            if supplied is not None:
                raise SemanticDecisionError(
                    "current after-sales order is Agent-bound, not model-authored"
                )
            return self.current_order_ids[0]
        if not isinstance(supplied, str) or supplied not in self.current_order_ids:
            raise SemanticDecisionError(
                "selected after-sales order is outside current public authority"
            )
        return supplied


def _register_action_binding(
    spec: AgentBusinessIntentSpec,
    *,
    operation: str,
    action_kind: str,
    destination: str,
    roles: frozenset[str],
    payload_mode: str = "arguments",
    semantic_variant: str | None = None,
) -> _ActionBinding:
    """Register one semantic source and its one Agent-private wire route."""

    return _ActionBinding(
        spec=spec,
        route=RouteBinding(
            operation=operation,
            action_kind=action_kind,
            destination=destination,
            roles=roles,
            source_name=spec.name,
        ),
        payload_mode=payload_mode,
        semantic_variant=semantic_variant,
    )


def _bind_authority_projection(
    binding: _ActionBinding,
    projection: _AuthorityIntentProjection,
) -> AgentBusinessIntentSpec:
    """Attach a schema projection to an already registered intent source."""

    return AgentBusinessIntentSpec(
        name=binding.spec.name,
        description=projection.description or binding.spec.description,
        parameters=copy.deepcopy(dict(projection.parameters)),
    )


_ACTOR_ROLES = frozenset({"buyer", "merchant"})

# The selector is the deterministic owner of the per-turn capability surface.
# These declarations name the built-in Skill bundles that may request each
# business action. They deliberately contain VCP action kinds, not intent
# source names, so one declaration covers schema-preserving compiler revisions.
_SKILL_ACTION_KINDS: dict[str, dict[str, frozenset[str]]] = {
    "buyer": {
        "mandate-parsing": frozenset({"delegate.reject_purchase"}),
        "marketplace-message-safety": frozenset({"commerce.send_message"}),
        "discovery-search": frozenset(
            {
                "commerce.search",
                "commerce.get_sku",
                "commerce.accept_offer",
                "commerce.send_message",
                "commerce.submit_decision_record",
                "delegate.reject_purchase",
            }
        ),
        "negotiation": frozenset(
            {
                "commerce.propose_offer",
                "commerce.counter_offer",
                "commerce.accept_offer",
                "commerce.reject_offer",
                "commerce.withdraw_offer",
                "platform.settle_payment",
            }
        ),
        "cart-checkout": frozenset(
            {
                "commerce.request_cart_quote",
                "platform.checkout_cart",
                "delegate.reject_purchase",
            }
        ),
        "supply-fulfillment": frozenset(
            {
                "commerce.read_supply_state",
                "commerce.read_shipment",
                "commerce.resolve_shipment",
                "commerce.send_message",
                "platform.settle_payment",
                "delegate.reject_purchase",
            }
        ),
        "purchase-confirmation": frozenset(
            {
                "platform.settle_payment",
                "commerce.reject_offer",
                "delegate.reject_purchase",
            }
        ),
        "after-sales-lifecycle": frozenset(
            {
                "commerce.cancel_order",
                "commerce.cancel_paid_order",
                "commerce.open_refund_case",
                "commerce.request_return",
                "commerce.request_exchange",
                "commerce.open_dispute",
                "commerce.submit_dispute_evidence",
                "commerce.respond_to_dispute",
                "commerce.request_ledger_reconciliation",
                "commerce.read_payment_history",
                "commerce.read_ledger_history",
                "commerce.read_packing_history",
                "commerce.read_after_sales_history",
                "commerce.read_after_sales_policy",
                "commerce.send_message",
            }
        ),
        "return-refund": frozenset({"commerce.request_return"}),
        "authenticated-review": frozenset({"commerce.submit_review"}),
        "protocol-event-handling": frozenset(
            {
                "commerce.acknowledge_protocol_event",
                "commerce.reject_protocol_event",
                "commerce.process_protocol_event",
                "commerce.publish_evidence_record",
            }
        ),
    },
    "merchant": {
        # Pre-step and guard Skills contribute reasoning/memory constraints but
        # own no terminal intent by themselves.  Explicit empty ownership is
        # fail-closed and still lets them compose with a selected main Skill.
        "price-discovery": frozenset(),
        "stockout-aware-pricing": frozenset(),
        "reputation-aware-pricing": frozenset(),
        "claim-truthfulness": frozenset({"commerce.update_listing"}),
        "private-utility-guard": frozenset(),
        "pricing-negotiate": frozenset(
            {
                "commerce.counter_offer",
                "commerce.accept_offer",
                "commerce.reject_offer",
                "commerce.withdraw_offer",
            }
        ),
        "aging-markdown": frozenset({"commerce.adjust_price"}),
        "demand-driven-markup": frozenset({"commerce.adjust_price"}),
        "peer-pricing": frozenset({"commerce.adjust_price"}),
        "catalog-serve": frozenset(
            {
                "commerce.get_sku",
                "commerce.respond_inquiry",
            }
        ),
        "inquiry-handle": frozenset(
            {
                "commerce.get_sku",
                "commerce.respond_inquiry",
                "commerce.send_message",
            }
        ),
        "listing-publish": frozenset(
            {
                "commerce.update_listing",
                "commerce.respond_inquiry",
            }
        ),
        "order-intake": frozenset(
            {
                "commerce.accept_offer",
                "commerce.respond_inquiry",
            }
        ),
        "order-cancel": frozenset(
            {
                "commerce.cancel_order",
                "commerce.respond_inquiry",
            }
        ),
        "cart-quote-handle": frozenset({"commerce.request_cart_quote"}),
        "fulfillment": frozenset(
            {
                "commerce.dispatch",
                "commerce.mark_returned",
                "commerce.issue_refund",
                "commerce.respond_inquiry",
            }
        ),
        "supply-logistics": frozenset(
            {
                "commerce.read_supply_state",
                "commerce.update_supply",
                "commerce.allocate_fulfillment",
                "commerce.read_shipment",
                "commerce.resolve_shipment",
                "commerce.send_message",
            }
        ),
        "return-adjudicate": frozenset(
            {
                "commerce.issue_refund",
                "commerce.respond_inquiry",
            }
        ),
        "after-sales-lifecycle": frozenset(
            {
                "commerce.cancel_paid_order",
                "commerce.authorize_return",
                "commerce.deny_return",
                "commerce.receive_return",
                "commerce.approve_refund",
                "commerce.deny_refund",
                "commerce.authorize_exchange",
                "commerce.deny_exchange",
                "commerce.complete_exchange",
                "commerce.open_dispute",
                "commerce.submit_dispute_evidence",
                "commerce.respond_to_dispute",
                "commerce.request_ledger_reconciliation",
                "commerce.read_payment_history",
                "commerce.read_ledger_history",
                "commerce.read_packing_history",
                "commerce.read_after_sales_history",
                "commerce.read_after_sales_policy",
                "commerce.read_shipment",
                "commerce.send_message",
            }
        ),
        "dispute-defense": frozenset(
            {
                "commerce.open_dispute",
                "commerce.submit_dispute_evidence",
                "commerce.respond_to_dispute",
                "commerce.send_message",
            }
        ),
        "inbound-restock": frozenset({"commerce.receive_shipment"}),
        "restock-signal": frozenset({"commerce.send_message"}),
        "listing-claim-manage": frozenset(
            {
                "commerce.apply_listing_claim",
                "commerce.update_listing",
                "commerce.submit_decision_record",
                "commerce.respond_inquiry",
            }
        ),
        "market-governance": frozenset(
            {
                "commerce.publish_campaign",
                "commerce.disclose_placement",
                "commerce.activate_campaign",
                "commerce.reject_review_manipulation",
                "commerce.reject_coordination",
                "commerce.accept_remediation_plan",
                "commerce.complete_remediation_step",
                "commerce.read_governance_history",
            }
        ),
        "protocol-event-handle": frozenset(
            {
                "commerce.acknowledge_protocol_event",
                "commerce.reject_protocol_event",
                "commerce.process_protocol_event",
                "commerce.publish_evidence_record",
            }
        ),
    },
}

# Native World reads follow the same least-capability rule as actions.  The
# legacy text Agent exposed a global read whitelist and relied on Skill prose
# to choose from it.  Native models receive only the reads needed by the
# selector-activated business workflow, which removes contradictory and
# near-duplicate choices without changing World authorization.
_SKILL_READ_INTENTS: dict[str, dict[str, frozenset[str]]] = {
    "buyer": {
        "mandate-parsing": frozenset(),
        "marketplace-message-safety": frozenset(),
        "discovery-search": frozenset(
            {
                "read_search_catalog",
                "read_listing",
                "read_stock_availability",
                "read_merchant_reputation",
                "read_friends",
                "read_friend_reviews",
                "read_review_evidence",
                "read_evidence_record",
                "read_mandate_revisions",
                "read_listing_claim",
                "read_listing_claims",
            }
        ),
        "negotiation": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_merchant_reputation",
                "read_own_balance",
            }
        ),
        "cart-checkout": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_own_balance",
            }
        ),
        "supply-fulfillment": frozenset({"read_own_order"}),
        "purchase-confirmation": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_own_balance",
                "read_own_order",
            }
        ),
        "after-sales-lifecycle": frozenset(
            {
                "read_own_order",
                "read_evidence_record",
            }
        ),
        "return-refund": frozenset({"read_own_order"}),
        "authenticated-review": frozenset({"read_own_order", "read_listing"}),
        "protocol-event-handling": frozenset(
            {
                "read_evidence_record",
                "read_mandate_revisions",
                "read_listing_claim",
                "read_listing_claims",
            }
        ),
    },
    "merchant": {
        "price-discovery": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_merchant_reputation",
            }
        ),
        "stockout-aware-pricing": frozenset(
            {
                "read_listing",
                "read_stock_availability",
            }
        ),
        "reputation-aware-pricing": frozenset({"read_merchant_reputation"}),
        "claim-truthfulness": frozenset(
            {
                "read_listing",
                "read_listing_claim",
                "read_listing_claims",
                "read_evidence_record",
            }
        ),
        "private-utility-guard": frozenset(),
        "pricing-negotiate": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_merchant_reputation",
            }
        ),
        "aging-markdown": frozenset({"read_listing", "read_stock_availability"}),
        "demand-driven-markup": frozenset({"read_listing", "read_stock_availability"}),
        "peer-pricing": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_merchant_reputation",
            }
        ),
        "catalog-serve": frozenset({"read_listing", "read_stock_availability"}),
        "inquiry-handle": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_listing_claims",
            }
        ),
        "listing-publish": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_listing_claims",
            }
        ),
        "order-intake": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_own_order",
            }
        ),
        "order-cancel": frozenset({"read_own_order"}),
        "cart-quote-handle": frozenset({"read_listing"}),
        "fulfillment": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_own_order",
            }
        ),
        "supply-logistics": frozenset(
            {
                "read_listing",
                "read_stock_availability",
                "read_own_order",
            }
        ),
        "return-adjudicate": frozenset(
            {
                "read_own_order",
                "read_evidence_record",
            }
        ),
        "after-sales-lifecycle": frozenset(
            {
                "read_own_order",
                "read_evidence_record",
            }
        ),
        "dispute-defense": frozenset(
            {
                "read_own_order",
                "read_evidence_record",
            }
        ),
        "inbound-restock": frozenset({"read_listing", "read_stock_availability"}),
        "restock-signal": frozenset({"read_stock_availability"}),
        "listing-claim-manage": frozenset(
            {
                "read_listing",
                "read_listing_claim",
                "read_listing_claims",
                "read_evidence_record",
            }
        ),
        "market-governance": frozenset(
            {
                "read_merchant_reputation",
                "read_evidence_record",
            }
        ),
        "protocol-event-handle": frozenset({"read_evidence_record"}),
    },
}

# Runtime evidence submissions do not change commerce state.  They remain
# available on every selected turn so a benchmark task can close with a typed
# result without granting an unrelated Platform or World capability.
_ALWAYS_SKILL_SCOPED_ACTIONS = frozenset(
    {
        "commerce.submit_decision_record",
        "delegate.report_result",
    }
)

# Actor-result actions require a scenario-registered causal root in addition to
# role and Skill ownership.  The Agent computes that per turn from spawn-frozen
# roots and its own direct outbound lineage, then passes the decision into this
# registry for both advertisement and compilation.
ACTOR_REPORT_ACTION_KINDS = _ALWAYS_SKILL_SCOPED_ACTIONS

# An action kind can have more than one canonical Platform route.  These exact
# overrides prevent an otherwise valid role/skill pair from selecting the
# wrong service (for example a merchant accepting its own aggregator offer).
_ROUTE_SKILLS: dict[tuple[str, str, str], frozenset[str]] = {
    ("buyer", "commerce.accept_offer", "platform:aggregator"): frozenset(
        {
            "discovery-search",
        }
    ),
    ("buyer", "commerce.accept_offer", "platform:negotiation"): frozenset(
        {
            "negotiation",
        }
    ),
    ("merchant", "commerce.accept_offer", "platform:negotiation"): frozenset(
        {
            "pricing-negotiate",
            "order-intake",
        }
    ),
    ("buyer", "commerce.request_return", "platform:after-sales"): frozenset(
        {
            "after-sales-lifecycle",
        }
    ),
    ("buyer", "commerce.request_return", "platform:psp"): frozenset(
        {
            "return-refund",
        }
    ),
}


def _object_schema(
    properties: Mapping[str, Any] | None = None,
    *,
    required: Sequence[str] = (),
    additional: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(dict(properties or {})),
        "required": list(required),
        "additionalProperties": additional,
    }


_TEXT = {"type": "string", "minLength": 1}
_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_NONNEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
_POSITIVE_INTEGER = {"type": "integer", "minimum": 1}
_BOOLEAN = {"type": "boolean"}
_OBJECT = {"type": "object"}
_TEXT_LIST = {"type": "array", "items": _TEXT}
_OBJECT_LIST = {"type": "array", "items": _OBJECT}
_MONEY = _object_schema(
    {
        "amount": {
            "oneOf": [
                {"type": "integer"},
                {
                    "type": "string",
                    "pattern": r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$",
                },
            ]
        },
        "currency": _TEXT,
    },
    required=("amount", "currency"),
)
_CATALOG_LISTING_FIELDS = _object_schema(
    {
        "list_price": _POSITIVE_INTEGER,
        "attributes": _OBJECT,
        "permitted_claims": _TEXT_LIST,
        "must_not_claim": _TEXT_LIST,
        "inventory": {"type": "integer", "minimum": 0, "maximum": 0},
        "status": _STRING,
        "category": _STRING,
        "name": _STRING,
        "currency": _STRING,
        "product_id": _STRING,
    }
)
_TYPED_SEARCH_FILTERS = _object_schema(
    {
        "shipping_days_max": _NONNEGATIVE_INTEGER,
        "warranty_months_min": _NONNEGATIVE_INTEGER,
        "return_days_min": _NONNEGATIVE_INTEGER,
        "energy_score_min": _NONNEGATIVE_INTEGER,
        "required_features": {
            "type": "array",
            "items": _TEXT,
            "uniqueItems": True,
        },
    }
)
_EVIDENCE_RECORD = _object_schema(
    {
        "schema_id": _TEXT,
        "record_id": _TEXT,
        "kind": _TEXT,
        "subject_id": _TEXT,
        "issuer_id": _TEXT,
        "facts": _OBJECT,
        "trust": _OBJECT,
        "version": _POSITIVE_INTEGER,
        "owner_id": _TEXT,
        "read_acl": {"type": "array", "items": _TEXT, "uniqueItems": True},
        "issued_at_tick": _NONNEGATIVE_INTEGER,
        "record_digest": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
    required=(
        "schema_id",
        "record_id",
        "kind",
        "subject_id",
        "issuer_id",
        "facts",
        "trust",
        "version",
        "owner_id",
        "read_acl",
        "issued_at_tick",
        "record_digest",
    ),
)


#: What an actor states about its own observation. Issuer, ownership, read
#: authority, logical time, and the digest are sealed by platform:evidence.
_EVIDENCE_OBSERVATION = _object_schema(
    {
        "kind": _TEXT,
        "subject_id": _TEXT,
        "facts": _OBJECT,
        "trust": _OBJECT,
    },
    required=("kind", "subject_id", "facts"),
)


def _world_read_specs() -> tuple[tuple[AgentBusinessIntentSpec, str], ...]:
    rows = (
        (
            "read_search_catalog",
            "Search the caller-visible catalog.",
            "world.search_catalog",
            _object_schema(
                {"query": _STRING, "filters": _OBJECT, "limit": _POSITIVE_INTEGER},
                required=("query",),
            ),
        ),
        (
            "read_listing",
            "Read one authoritative listing by SKU.",
            "world.get_listing",
            _object_schema({"sku_id": _TEXT}, required=("sku_id",)),
        ),
        (
            "read_stock_availability",
            "Check whether a caller-visible SKU has enough stock.",
            "world.is_in_stock",
            _object_schema({"sku_id": _TEXT, "qty": _POSITIVE_INTEGER}, required=("sku_id",)),
        ),
        (
            "read_merchant_reputation",
            "Read a merchant's public reputation.",
            "world.get_merchant_reputation",
            _object_schema({"merchant_id": _TEXT}, required=("merchant_id",)),
        ),
        (
            "read_own_balance",
            "Read the calling actor's own ledger balance.",
            "world.get_balance",
            _object_schema(),
        ),
        (
            "read_own_order",
            "Read an order when the caller is a party.",
            "world.get_order",
            _object_schema({"order_id": _TEXT}, required=("order_id",)),
        ),
        (
            "read_friends",
            "Read the calling buyer's own friend identifiers.",
            "world.get_friends",
            _object_schema(),
        ),
        (
            "read_friend_reviews",
            "Read reviews authored by the caller's friends.",
            "world.get_friend_reviews",
            _object_schema({"sku_id": _TEXT, "merchant_id": _TEXT}),
        ),
        (
            "read_review_evidence",
            "Read friend reviews with stable evidence identifiers.",
            "world.get_review_evidence",
            _object_schema({"sku_id": _TEXT, "merchant_id": _TEXT}),
        ),
        (
            "read_evidence_record",
            "Read the current caller-authorized durable evidence record by its stable identity.",
            "world.get_evidence_record",
            _object_schema({"record_id": _TEXT}, required=("record_id",)),
        ),
        (
            "read_mandate_revisions",
            "Read the caller-authorized mandate revision history.",
            "world.get_mandate_revisions",
            _object_schema({"mandate_id": _TEXT}, required=("mandate_id",)),
        ),
        (
            "read_listing_claim",
            "Read one caller-visible listing claim.",
            "world.get_listing_claim",
            _object_schema({"claim_id": _TEXT}, required=("claim_id",)),
        ),
        (
            "read_listing_claims",
            "List published claims for one listing.",
            "world.list_listing_claims",
            _object_schema({"listing_id": _TEXT}, required=("listing_id",)),
        ),
    )
    return tuple(
        (
            AgentBusinessIntentSpec(name, description, schema, category="read"),
            tool,
        )
        for name, description, tool, schema in rows
    )


WORLD_READ_TOOL_SPECS: tuple[AgentBusinessIntentSpec, ...] = tuple(
    spec for spec, _tool in _world_read_specs()
)


def _validator_schema(validator: Any) -> dict[str, Any]:
    name = getattr(validator, "__name__", "")
    schemas = {
        "_text": _TEXT,
        "_string": _STRING,
        "_integer": _INTEGER,
        "_nonnegative_integer": _NONNEGATIVE_INTEGER,
        "_positive_integer": _POSITIVE_INTEGER,
        "_boolean": _BOOLEAN,
        "_object": _OBJECT,
        "_text_list": _TEXT_LIST,
        "_object_list": _OBJECT_LIST,
        "_money": _MONEY,
        "_rating": {"type": "integer", "minimum": 1, "maximum": 5},
        "_catalog_fields": _CATALOG_LISTING_FIELDS,
        "_typed_search_filters": _TYPED_SEARCH_FILTERS,
        "_nonempty_unique_text_list": {
            "type": "array",
            "items": _TEXT,
            "minItems": 1,
            "uniqueItems": True,
        },
        "_bounded_supply_sku_ids": {
            "type": "array",
            "items": _TEXT,
            "minItems": 1,
            "maxItems": 64,
            "uniqueItems": True,
        },
        "_evidence_record": _EVIDENCE_RECORD,
        "_evidence_observation": _EVIDENCE_OBSERVATION,
    }
    if name in schemas:
        return copy.deepcopy(schemas[name])
    if name == "_cart_lines":
        return {
            "type": "array",
            "minItems": 1,
            "items": _object_schema(
                {"sku_id": _TEXT, "qty": _POSITIVE_INTEGER},
                required=("sku_id", "qty"),
            ),
        }
    if name == "validate":
        closure = inspect.getclosurevars(validator).nonlocals
        allowed = closure.get("allowed")
        if isinstance(allowed, frozenset) and all(isinstance(v, str) for v in allowed):
            return {"type": "string", "enum": sorted(allowed)}
    return copy.deepcopy(_OBJECT)


def _platform_payload_schema(kind: str, destination: str) -> dict[str, Any]:
    # The terminal contract is the authoritative validator.  Its internal
    # field map is reflected only to build the internal compiler schema; compile
    # time always validates again through preview_actor_terminal_action.
    from protocol import actor_terminal as terminal

    payload_spec = terminal._PLATFORM_SPECS.get((kind, destination))  # type: ignore[attr-defined]
    if payload_spec is not None:
        properties = {
            name: _validator_schema(validator)
            for name, validator in {**payload_spec.required, **payload_spec.optional}.items()
        }
        required = tuple(payload_spec.required)
        if kind == "commerce.search" and destination == "platform:aggregator":
            # ``limit`` and mandate authority are framework-owned in native
            # mode.  Keep the old fields locally accepted for compatibility
            # with scripted adapters, but let an Agent fill them when they are
            # absent and override them when scenario authority is available.
            required = ("query",)
        if kind == "platform.checkout_cart" and destination == "platform:checkout":
            # Choosing to checkout the currently reviewed quote is the model
            # action.  The quote identity is bound from Platform context.
            required = ()
    elif kind == "commerce.accept_offer" and destination == "platform:aggregator":
        properties = {
            "mandate_id": _TEXT,
            "offer_id": _TEXT,
            "session_id": _TEXT,
            "session_digest": _TEXT,
            "offer_digest": _TEXT,
            "sku_id": _TEXT,
            "merchant_id": _TEXT,
            "qty": _POSITIVE_INTEGER,
            "unit_price_cents": _NONNEGATIVE_INTEGER,
            "currency": _TEXT,
            "catalog_revision": _NONNEGATIVE_INTEGER,
            "inventory_revision": _NONNEGATIVE_INTEGER,
            "order_id": _TEXT,
        }
        # The model chooses an offer.  The Agent binds the active mandate and
        # projects every provider call to the compact terminal contract.
        required = ("offer_id",)
    else:
        # Cart request has two retained wire forms.  The compiler's canonical
        # terminal validator enforces that one complete form was selected.
        properties = {
            "request_id": _TEXT,
            "market_id": _TEXT,
            "mandate_id": _TEXT,
            "lines": {"type": "array", "items": _OBJECT},
            "fill_policy": {"type": "string"},
            "backorder_policy": {"type": "string"},
        }
        required = ()
    if kind == "platform.settle_payment":
        # Accept the compact order/payment-rail or certificate form in
        # addition to the established full settlement request.
        required = ()
    elif kind == "commerce.allocate_fulfillment" and destination == "platform:fulfillment":
        # Actor terminal applies the stronger non-empty/unique post-validator.
        # Reflect it in the business schema so an admitted model
        # call cannot fail merely because the generated list was empty or
        # duplicated.
        properties["priority_order_ids"] = {
            "type": "array",
            "items": _TEXT,
            "minItems": 1,
            "uniqueItems": True,
        }
    elif kind == "commerce.read_supply_state" and destination == "platform:supply":
        # SupplyPolicy and actor terminal both require a bounded unique batch.
        properties["sku_ids"] = {
            "type": "array",
            "items": _TEXT,
            "minItems": 1,
            "maxItems": 64,
            "uniqueItems": True,
        }
    if kind == "commerce.accept_offer":
        properties["decision_rationale"] = _TEXT
        required = (*required, "decision_rationale")
    return _object_schema(properties, required=required)


def _business_platform_parameters(
    kind: str,
    destination: str,
    local: Mapping[str, Any],
) -> dict[str, Any]:
    """Project protocol-heavy actions into high-level business functions.

    The local schema remains backward compatible with deterministic scripted
    adapters.  A live model never receives actor authority, session digests,
    revisions, order identifiers, or the scenario's retrieval bound as fields
    it is expected to manufacture.
    """

    if (kind, destination) == ("commerce.search", "platform:aggregator"):
        local_properties = local.get("properties", {})
        properties = {
            "query": copy.deepcopy(local_properties["query"]),
        }
        if "filters" in local_properties:
            properties["filters"] = copy.deepcopy(local_properties["filters"])
        return _object_schema(properties, required=("query",))
    if (kind, destination) == ("commerce.accept_offer", "platform:aggregator"):
        return _object_schema(
            {
                "offer_id": _TEXT,
                "decision_rationale": _TEXT,
            },
            required=("offer_id", "decision_rationale"),
        )
    if (kind, destination) == ("platform.checkout_cart", "platform:checkout"):
        return _object_schema({}, required=())
    if (kind, destination) == ("platform.settle_payment", "platform:psp"):
        # The Agent compiles the settlement target from the Platform-owned match
        # certificate or the authenticated negotiation authority.  The model
        # chooses to settle; it never names the certificate or the thread.
        return _without_properties(local, ("cert_id", "negotiation_id"))
    if (kind, destination) == ("commerce.update_supply", "platform:supply"):
        # Optimistic-concurrency versions are Agent bookkeeping, not a business
        # choice the model is asked to transcribe.
        return _without_properties(local, ("expected_version",))
    # commerce.publish_evidence_record -> platform:evidence needs no projection:
    # its actor-facing payload is already only the observation, and
    # platform:evidence derives the read ACL and the rest of the authority
    # fields from World relationships.
    return copy.deepcopy(dict(local))


def _without_properties(
    schema: Mapping[str, Any],
    drop: Sequence[str],
) -> dict[str, Any]:
    """Copy ``schema`` without the named properties.

    Only the model-facing projection loses the fields.  The local protocol
    schema is untouched, so the Agent still validates the compiled payload it
    authors from sealed authority.
    """

    removed = set(drop)
    properties = {
        name: value
        for name, value in dict(schema.get("properties", {})).items()
        if name not in removed
    }
    required = [name for name in schema.get("required", []) if name not in removed]
    return _object_schema(
        properties,
        required=required,
        additional=bool(schema.get("additionalProperties", False)),
    )


def _source_name_for_route(kind: str, destination: str) -> str:
    raw = "act_" + re.sub(r"[^A-Za-z0-9]+", "_", kind).strip("_")
    suffix = destination.split(":", 1)[-1].replace("-", "_")
    raw = f"{raw}_{suffix}"
    if len(raw) <= 64:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{raw[:55]}_{digest}"


def _platform_business_operation(kind: str, destination: str) -> str:
    """Name the business operation while registering its Platform route."""

    if kind == "commerce.accept_offer":
        return (
            "accept_ranked_offer"
            if destination == "platform:aggregator"
            else "accept_negotiated_offer"
        )
    if kind == "commerce.request_return":
        return "request_payment_return" if destination == "platform:psp" else "request_order_return"
    return kind.split(".", 1)[-1]


def _platform_bindings() -> tuple[_ActionBinding, ...]:
    rows: list[_ActionBinding] = []
    for kind, destination in sorted(BENCHMARK_PLATFORM_ACTOR_ROUTES):
        name = _source_name_for_route(kind, destination)
        roles = _platform_sender_roles(kind, destination)
        local_parameters = _platform_payload_schema(kind, destination)
        spec = AgentBusinessIntentSpec(
            name=name,
            description=(
                f"Perform the business action {kind} through the "
                f"CommerceWorld {destination.split(':', 1)[-1]} service."
            ),
            parameters=local_parameters,
            business_parameters=_business_platform_parameters(
                kind,
                destination,
                local_parameters,
            ),
        )
        rows.append(
            _register_action_binding(
                spec,
                operation=_platform_business_operation(kind, destination),
                action_kind=kind,
                destination=destination,
                roles=roles,
            )
        )
    return tuple(rows)


def _platform_sender_roles(kind: str, destination: str) -> frozenset[str]:
    """Return actor roles authorized for one exact Platform route."""

    try:
        partition = PARTITION_ALLOW[ActionKind(kind)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"semantic Platform action has no partition: {kind}") from exc
    roles = frozenset(partition["senders"]) & _ACTOR_ROLES
    if (kind, destination) == ("commerce.accept_offer", "platform:aggregator"):
        # The generic negotiation action is symmetric, but the aggregator
        # certificate path is a buyer choice only.
        roles &= frozenset({"buyer"})
    if not roles:
        raise ValueError(
            f"semantic Platform action has no buyer/merchant sender: {kind} -> {destination}"
        )
    return roles


def _specialized_bindings() -> tuple[_ActionBinding, ...]:
    """Register non-generic business variants in the sole route registry.

    These are ordinary Agent-owned Platform/Runtime routes (for example the
    three shipment resolutions).  Calling them "direct" was a leftover from
    the removed direct-LLM adapter era and incorrectly suggested a bypass of
    Episode/Runtime/Platform/World.
    """
    report_schema = _object_schema(
        {
            "outcome": {
                "type": "string",
                "enum": ["completed", "declined", "failed", "partial"],
            },
            "summary": _TEXT,
            "details": _OBJECT,
        },
        required=("outcome", "summary", "details"),
    )
    return (
        _register_action_binding(
            AgentBusinessIntentSpec(
                _SHIPMENT_WAIT_INTENT,
                "Keep the current delayed shipment open and wait for delivery.",
                _object_schema({}, required=()),
            ),
            operation="wait_for_shipment",
            action_kind="commerce.resolve_shipment",
            destination="platform:fulfillment",
            roles=_ACTOR_ROLES,
            semantic_variant="wait",
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                _SHIPMENT_REPLACE_INTENT,
                "Replace the current shipment with one selected replacement SKU.",
                _object_schema(
                    {"replacement_sku_id": _TEXT},
                    required=("replacement_sku_id",),
                ),
            ),
            operation="replace_shipment",
            action_kind="commerce.resolve_shipment",
            destination="platform:fulfillment",
            roles=_ACTOR_ROLES,
            semantic_variant="replacement",
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                _SHIPMENT_REFUND_INTENT,
                "Refund the order attached to the current shipment.",
                _object_schema({}, required=()),
            ),
            operation="refund_shipment",
            action_kind="commerce.resolve_shipment",
            destination="platform:fulfillment",
            roles=_ACTOR_ROLES,
            semantic_variant="refund",
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                _CLAIM_PUBLISH_INTENT,
                "Publish one existing draft claim with selected evidence.",
                _object_schema(),
            ),
            operation="publish_listing_claim",
            action_kind="commerce.apply_listing_claim",
            destination="platform:claims",
            roles=frozenset({"merchant"}),
            semantic_variant="claim_publish",
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                _CLAIM_CORRECT_INTENT,
                "Correct one existing published claim with selected evidence.",
                _object_schema(),
            ),
            operation="correct_listing_claim",
            action_kind="commerce.apply_listing_claim",
            destination="platform:claims",
            roles=frozenset({"merchant"}),
            semantic_variant="claim_correct",
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                _CLAIM_RETRACT_INTENT,
                "Retract one existing published claim for a stated reason.",
                _object_schema(),
            ),
            operation="retract_listing_claim",
            action_kind="commerce.apply_listing_claim",
            destination="platform:claims",
            roles=frozenset({"merchant"}),
            semantic_variant="claim_retract",
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                "act_submit_decision_record",
                "Submit the actor's structured task conclusion to Runtime evidence.",
                report_schema,
            ),
            operation="submit_decision_record",
            action_kind="commerce.submit_decision_record",
            destination="runtime:evidence",
            roles=_ACTOR_ROLES,
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                "act_report_result",
                "Submit the actor's structured result to Runtime evidence.",
                report_schema,
            ),
            operation="report_task_result",
            action_kind="delegate.report_result",
            destination="runtime:evidence",
            roles=_ACTOR_ROLES,
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                "act_reply_to_inquiry",
                "Reply to the actor that sent the current inquiry.",
                _object_schema({"payload": _OBJECT}, required=("payload",)),
            ),
            operation="respond_inquiry",
            action_kind="commerce.respond_inquiry",
            destination=INBOUND_SENDER_DESTINATION,
            payload_mode="nested_payload",
            roles=frozenset({"merchant"}),
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                "act_reject_purchase",
                "Reject the current principal purchase mandate safely.",
                _object_schema(
                    {"mandate_id": _TEXT, "reason": _TEXT},
                    required=("reason",),
                ),
                business_parameters=_object_schema(
                    {"reason": _TEXT},
                    required=("reason",),
                ),
            ),
            operation="reject_purchase",
            action_kind="delegate.reject_purchase",
            destination=INBOUND_SENDER_DESTINATION,
            roles=frozenset({"buyer"}),
        ),
        _register_action_binding(
            AgentBusinessIntentSpec(
                "act_send_message",
                "Send a bounded typed commerce message to a named buyer or merchant participant.",
                _object_schema(
                    {"recipient_id": _TEXT, "payload": _OBJECT},
                    required=("recipient_id", "payload"),
                ),
            ),
            operation="send_message",
            action_kind="commerce.send_message",
            destination="@argument:recipient_id",
            payload_mode="recipient_and_payload",
            roles=_ACTOR_ROLES,
        ),
    )


def _coerce_task_result_contract(value: Any) -> dict[str, Any] | None:
    """Validate one framework-owned, value-free delegated-task contract."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FrameworkAuthorityError("task result contract must be an object")
    active = value.get("active")
    action_kind = value.get("action_kind")
    endpoint = value.get("endpoint")
    result_format = value.get("result_format", "named_submission")
    submission_kind = value.get("submission_kind")
    payload_schema = value.get("payload_schema")
    if not isinstance(active, bool):
        raise FrameworkAuthorityError("task result contract active flag must be boolean")
    if action_kind not in ACTOR_REPORT_ACTION_KINDS:
        raise FrameworkAuthorityError("task result contract has an unsupported actor action")
    if endpoint != "runtime:evidence":
        raise FrameworkAuthorityError("task result contract must use Runtime evidence")
    if result_format not in {"named_submission", "direct_details"}:
        raise FrameworkAuthorityError("task result contract has an unsupported result format")
    if result_format == "named_submission":
        if not isinstance(submission_kind, str) or not submission_kind.strip():
            raise FrameworkAuthorityError("named task result contract has no submission kind")
    elif submission_kind is not None:
        raise FrameworkAuthorityError("direct-details task result cannot name a submission kind")
    if (
        not isinstance(payload_schema, Mapping)
        or payload_schema.get("type") != "object"
        or not isinstance(payload_schema.get("properties"), Mapping)
    ):
        raise FrameworkAuthorityError("task result contract has no object payload schema")
    return {
        "active": active,
        "action_kind": str(action_kind),
        "endpoint": str(endpoint),
        "result_format": str(result_format),
        "submission_kind": submission_kind,
        "payload_schema": copy.deepcopy(dict(payload_schema)),
    }


def _coerce_allowed_routes(
    value: Any,
) -> frozenset[tuple[str, str]] | None:
    """Normalize framework-owned semantic registry routes."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple, frozenset)):
        raise FrameworkAuthorityError("public task allowed routes must be a sequence")
    routes: set[tuple[str, str]] = set()
    for row in value:
        if (
            not isinstance(row, (list, tuple))
            or len(row) != 2
            or any(not isinstance(item, str) or not item.strip() for item in row)
        ):
            raise FrameworkAuthorityError(
                "public task allowed routes must contain action/destination pairs"
            )
        routes.add((row[0], row[1]))
    if len(routes) != len(value):
        raise FrameworkAuthorityError("public task allowed routes must be unique")
    return frozenset(routes)


def _task_result_authority_projection(
    contract: Mapping[str, Any],
) -> _AuthorityIntentProjection:
    """Project public task authority into a model-visible result schema."""

    result_format = str(contract["result_format"])
    submission_kind = contract.get("submission_kind")
    parameters = _object_schema(
        {
            "outcome": {
                "type": "string",
                "enum": ["completed", "declined", "failed", "partial"],
            },
            "summary": _TEXT,
            "payload": copy.deepcopy(dict(contract["payload_schema"])),
        },
        required=("outcome", "summary", "payload"),
    )
    result_label = (
        str(submission_kind)
        if result_format == "named_submission"
        else "evidence-grounded decision"
    )
    wrapper_description = (
        "result kind, Runtime record, and protocol wrapper"
        if result_format == "named_submission"
        else "Runtime record and direct details wrapper"
    )
    return _AuthorityIntentProjection(
        description=(
            f"Submit the current task's {result_label} result. Provide only "
            "the semantic payload requested by the public task contract. "
            f"CommerceWorld binds the {wrapper_description}."
        ),
        parameters=parameters,
    )


def _inquiry_response_authority_projection(
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    authority = context.get("inquiry_response_authority")
    if not isinstance(authority, Mapping):
        return None
    raw_content_ids = authority.get("content_ids")
    if (
        not isinstance(raw_content_ids, (list, tuple))
        or not raw_content_ids
        or any(not isinstance(value, str) or not value.strip() for value in raw_content_ids)
        or len(raw_content_ids) != len(set(raw_content_ids))
    ):
        raise FrameworkAuthorityError("inquiry response authority has no finite content identities")
    content_ids = list(raw_content_ids)
    content_set = {
        "type": "array",
        "items": {"type": "string", "enum": content_ids},
        "uniqueItems": True,
    }
    return _AuthorityIntentProjection(
        description=(
            "Classify the current inquiry content into accepted and rejected "
            "business references using the grounded listing response."
        ),
        parameters=_object_schema(
            {
                "payload": _object_schema(
                    {
                        "security_response": _object_schema(
                            {
                                "resolution": {
                                    "type": "string",
                                    "enum": ["policy_safe"],
                                },
                                "rejected_content_ids": copy.deepcopy(content_set),
                                "accepted_content_ids": copy.deepcopy(content_set),
                            },
                            required=(
                                "resolution",
                                "rejected_content_ids",
                                "accepted_content_ids",
                            ),
                        )
                    },
                    required=("security_response",),
                )
            },
            required=("payload",),
        ),
    )


def _cart_quote_authority_projection(
    role: str,
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    """Return the high-level quote choice for a context-bound actor.

    Buyer line selection remains the capability under evaluation.  The model
    therefore supplies strict SKU/quantity lines, while Agent binds market,
    mandate, and policy authority.  A merchant only chooses whether to answer
    the current Platform request, whose identifier is already inbound state.
    """

    if role == "buyer":
        authority = context.get("cart_quote_authority")
        if not isinstance(authority, Mapping):
            raise FrameworkAuthorityError("buyer cart phase has no principal cart authority")
        required_authority = (
            "market_id",
            "mandate_id",
            "fill_policy",
            "backorder_policy",
        )
        if any(
            not isinstance(authority.get(name), str) or not str(authority[name]).strip()
            for name in required_authority
        ):
            raise FrameworkAuthorityError("buyer cart authority is malformed")
        grounding = _grounding_authority(context)
        grounded_skus = list(grounding.visible_sku_ids) if grounding is not None else []
        if not grounded_skus:
            return None
        parameters = _object_schema(
            {
                "lines": {
                    "type": "array",
                    "minItems": 1,
                    "items": _object_schema(
                        {
                            "sku_id": {
                                "type": "string",
                                "enum": grounded_skus,
                            },
                            "qty": _POSITIVE_INTEGER,
                        },
                        required=("sku_id", "qty"),
                    ),
                }
            },
            required=("lines",),
        )
        description = (
            "Request a World-authoritative quote for the selected cart lines. "
            "CommerceWorld binds the active market, mandate, and cart policies."
        )
    elif role == "merchant":
        request_id = context.get("cart_quote_request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise FrameworkAuthorityError(
                "merchant cart phase has no authenticated Platform request"
            )
        grounding = _grounding_authority(context)
        grounded_skus = list(grounding.visible_sku_ids) if grounding is not None else []
        if not grounded_skus:
            return None
        parameters = _object_schema(
            {
                "line_quotes": {
                    "type": "array",
                    "minItems": 1,
                    "items": _object_schema(
                        {
                            "sku_id": {
                                "type": "string",
                                "enum": grounded_skus,
                            },
                            "qty": _POSITIVE_INTEGER,
                            "unit_price_minor": _NONNEGATIVE_INTEGER,
                            "line_total_minor": _NONNEGATIVE_INTEGER,
                            "applied_rule_kinds": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "catalog_base",
                                        "quantity_tier",
                                        "bundle_discount",
                                    ],
                                },
                            },
                        },
                        required=(
                            "sku_id",
                            "qty",
                            "unit_price_minor",
                            "line_total_minor",
                            "applied_rule_kinds",
                        ),
                    ),
                },
                "charges": {
                    "type": "array",
                    "items": _object_schema(
                        {
                            "kind": {
                                "type": "string",
                                "enum": ["shipping", "tax", "fee"],
                            },
                            "amount_minor": _NONNEGATIVE_INTEGER,
                        },
                        required=("kind", "amount_minor"),
                    ),
                },
                "subtotal_minor": _NONNEGATIVE_INTEGER,
                "grand_total_minor": _NONNEGATIVE_INTEGER,
            },
            required=(
                "line_quotes",
                "charges",
                "subtotal_minor",
                "grand_total_minor",
            ),
        )
        description = (
            "Propose the line prices and total for the current cart from the "
            "actor-visible quantity tiers, bundle terms, and charge rules. "
            "The local Agent binds the private request identifier."
        )
    else:  # pragma: no cover - callers validate actor roles first.
        raise FrameworkAuthorityError("cart quote intents require a buyer or merchant role")
    return _AuthorityIntentProjection(
        description=description,
        parameters=parameters,
    )


def _commerce_action_authority(
    context: Mapping[str, Any],
    *,
    kind: str,
) -> Mapping[str, Any] | None:
    """Return one authority derived from actor-visible CommerceWorld state."""

    grounding = _grounding_authority(context)
    if grounding is None:
        return None
    route = {
        "catalog_update": CATALOG_UPDATE_ROUTE,
        "listing_claim": LISTING_CLAIM_ROUTE,
    }.get(kind)
    if route is None:
        raise FrameworkAuthorityError("commerce action authority kind is unknown")
    grounded = grounding.authority_for(route)
    if grounded is None:
        return None
    authority = grounded.to_mutable_value()
    if authority.get("schema_version") != GROUNDED_COMMERCE_AUTHORITY_V1:
        raise FrameworkAuthorityError("commerce action authority has an unsupported schema")
    if authority.get("kind") != kind:
        raise FrameworkAuthorityError("commerce action authority kind is inconsistent")
    return authority


def _grounding_authority(
    context: Mapping[str, Any],
) -> ResolvedAuthorityPrerequisites | None:
    value = context.get("grounding_authority")
    if value is None:
        return None
    if not isinstance(value, ResolvedAuthorityPrerequisites):
        raise FrameworkAuthorityError("grounding authority snapshot is malformed")
    return value


def _completed_listing_claim_ids(context: Mapping[str, Any]) -> frozenset[str]:
    """Return claim completions from the unified grounding snapshot only."""

    grounding = _grounding_authority(context)
    if grounding is None:
        return frozenset()
    return frozenset(grounding.completed_claim_ids)


def _commerce_grounding_routes(
    context: Mapping[str, Any],
    *,
    pending: bool,
) -> frozenset[tuple[str, str]]:
    grounding = _grounding_authority(context)
    if grounding is None:
        return frozenset()
    return grounding.pending_routes if pending else grounding.required_routes


def _catalog_update_authority_projection(
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    """Project owned-listing authority without choosing a wire route."""

    authority = _commerce_action_authority(context, kind="catalog_update")
    if authority is None:
        return None
    sku_id = authority.get("sku_id")
    attribute_schemas = authority.get("attribute_schemas")
    if (
        not isinstance(sku_id, str)
        or not sku_id.strip()
        or not isinstance(attribute_schemas, Mapping)
        or not attribute_schemas
        or any(not isinstance(name, str) or not name.strip() for name in attribute_schemas)
        or any(
            value_type
            not in {
                "string",
                "integer",
                "number",
                "boolean",
                "object",
                "array",
                "null",
            }
            for value_type in attribute_schemas.values()
        )
    ):
        raise PlatformContractError("grounded catalog update authority is malformed")
    grounding = _grounding_authority(context)
    current_listing_id = grounding.current_listing_id if grounding is not None else None
    if current_listing_id is not None and current_listing_id != sku_id:
        raise PlatformContractError(
            "catalog update authority differs from the active Runtime lineage"
        )
    changes_schema = _object_schema(
        {str(name): {"type": str(value_type)} for name, value_type in attribute_schemas.items()},
    )
    changes_schema["minProperties"] = 1
    return _AuthorityIntentProjection(
        description=(
            "Update one or more evidence-grounded attributes on the owned "
            "listing. CommerceWorld binds operation, listing id, and owner."
        ),
        parameters=_object_schema(
            {"changes": changes_schema},
            required=("changes",),
        ),
    )


def _claim_compilation_inventory(
    context: Mapping[str, Any],
) -> frozenset[str] | None:
    """Return the exact private claim inventory without exposing its templates."""

    raw = context.get("claim_compilation_templates")
    if raw is None:
        return None
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "claims"}
        or raw.get("schema_version") != _CLAIM_COMPILATION_SCHEMA_V1
        or not isinstance(raw.get("claims"), list)
        or not raw["claims"]
    ):
        raise DeterministicCompilerError("private claim compilation binding is malformed")
    claim_ids: list[str] = []
    for row in raw["claims"]:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"claim_id", "operation_templates"}
            or not isinstance(row.get("claim_id"), str)
            or not str(row["claim_id"]).strip()
            or not isinstance(row.get("operation_templates"), Mapping)
        ):
            raise DeterministicCompilerError("private claim compilation row is malformed")
        claim_ids.append(str(row["claim_id"]))
    if len(claim_ids) != len(set(claim_ids)):
        raise DeterministicCompilerError("private claim compilation identities are ambiguous")
    authorized_value = context.get("claim_compilation_authorized_claim_ids")
    if (
        not isinstance(authorized_value, (list, tuple, frozenset))
        or not authorized_value
        or any(
            not isinstance(claim_id, str) or not claim_id.strip() for claim_id in authorized_value
        )
    ):
        raise DeterministicCompilerError(
            "private claim compilation binding has no authorized claim inventory"
        )
    authorized = frozenset(authorized_value)
    if len(authorized) != len(tuple(authorized_value)) or authorized != frozenset(claim_ids):
        raise DeterministicCompilerError(
            "private claim compilation identities differ from task authority"
        )
    return authorized


def _listing_claim_authority(
    context: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, str], ...]] | None:
    """Validate claim and evidence identities observed through CommerceWorld."""

    authority = _commerce_action_authority(context, kind="listing_claim")
    if authority is None:
        return None
    claims_value = authority.get("claims")
    evidence_value = authority.get("evidence_records")
    if (
        not isinstance(claims_value, list)
        or not claims_value
        or not isinstance(evidence_value, list)
    ):
        raise PlatformContractError("grounded listing claim authority is malformed")
    evidence: list[dict[str, str]] = []
    for row in evidence_value:
        if not isinstance(row, Mapping):
            raise PlatformContractError("grounded evidence authority row is malformed")
        record_id = row.get("record_id")
        subject_id = row.get("subject_id")
        if (
            not isinstance(record_id, str)
            or not record_id.strip()
            or not isinstance(subject_id, str)
            or not subject_id.strip()
        ):
            raise PlatformContractError("grounded evidence authority row is malformed")
        evidence.append({"record_id": record_id, "subject_id": subject_id})
    if len({row["record_id"] for row in evidence}) != len(evidence):
        raise PlatformContractError("grounded evidence authority ids must be unique")
    claims: list[dict[str, Any]] = []
    for row in claims_value:
        if not isinstance(row, Mapping):
            raise PlatformContractError("listing claim authority row must be an object")
        claim_id = row.get("claim_id")
        listing_id = row.get("listing_id")
        subject = row.get("subject")
        state = row.get("state")
        content_schema = row.get("content_schema")
        used_evidence_record_ids = row.get("used_evidence_record_ids")
        if (
            not isinstance(claim_id, str)
            or not claim_id.strip()
            or not isinstance(listing_id, str)
            or not listing_id.strip()
            or not isinstance(subject, str)
            or not subject.strip()
            or state not in {"draft", "published", "corrected"}
            or not isinstance(content_schema, Mapping)
            or not content_schema
            or any(not isinstance(name, str) or not name.strip() for name in content_schema)
            or any(
                value_type
                not in {
                    "string",
                    "integer",
                    "number",
                    "boolean",
                    "object",
                    "array",
                    "null",
                }
                for value_type in content_schema.values()
            )
            or not isinstance(used_evidence_record_ids, list)
            or any(
                not isinstance(record_id, str) or not record_id.strip()
                for record_id in used_evidence_record_ids
            )
            or len(used_evidence_record_ids) != len(set(used_evidence_record_ids))
        ):
            raise PlatformContractError("grounded listing claim authority row is malformed")
        claims.append(
            {
                "claim_id": claim_id,
                "listing_id": listing_id,
                "subject": subject,
                "state": state,
                "content_schema": {
                    str(name): str(value_type) for name, value_type in content_schema.items()
                },
                "used_evidence_record_ids": tuple(used_evidence_record_ids),
            }
        )
    claim_ids = [row["claim_id"] for row in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise PlatformContractError("grounded listing claim authority ids must be unique")
    grounding = _grounding_authority(context)
    current_listing_id = grounding.current_listing_id if grounding is not None else None
    if current_listing_id is not None and any(
        row["listing_id"] != current_listing_id for row in claims
    ):
        raise PlatformContractError(
            "listing claim authority differs from the active Runtime lineage"
        )
    compilation_inventory = _claim_compilation_inventory(context)
    if compilation_inventory is not None:
        completed = _completed_listing_claim_ids(context)
        observed = frozenset(row["claim_id"] for row in claims)
        if not observed <= compilation_inventory or not completed <= compilation_inventory:
            raise DeterministicCompilerError(
                "private claim inventory differs from World claim authority"
            )
        if compilation_inventory - observed - completed:
            # Keep the claim route read-only until every frozen task claim has
            # current World authority.  This makes any later absent template
            # identity provably completed rather than an arbitrary extra row.
            return None
    return tuple(claims), tuple(evidence)


def _claim_compilation_binding(
    context: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]] | None:
    """Validate private deterministic fields for every selectable claim action.

    This binding is scenario-frozen Agent context, never a provider
    observation.  Any missing or malformed injected field is therefore a
    deterministic framework failure rather than a model validation error.
    """

    raw = context.get("claim_compilation_templates")
    if raw is None:
        return None
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema_version", "claims"}
        or raw.get("schema_version") != _CLAIM_COMPILATION_SCHEMA_V1
        or not isinstance(raw.get("claims"), list)
    ):
        raise DeterministicCompilerError("private claim compilation binding is malformed")
    authority_by_id = {str(row["claim_id"]): row for row in claims}
    if len(authority_by_id) != len(claims):
        raise PlatformContractError("listing claim authority ids are ambiguous")
    completed = _completed_listing_claim_ids(context)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    bound_claim_ids: set[str] = set()
    seen_template_ids: set[str] = set()
    for row in raw["claims"]:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"claim_id", "operation_templates"}
            or not isinstance(row.get("claim_id"), str)
            or not str(row["claim_id"]).strip()
            or not isinstance(row.get("operation_templates"), Mapping)
        ):
            raise DeterministicCompilerError("private claim compilation row is malformed")
        claim_id = str(row["claim_id"])
        claim = authority_by_id.get(claim_id)
        if claim_id in seen_template_ids or (claim is None and claim_id not in completed):
            raise DeterministicCompilerError(
                "private claim compilation identities differ from World authority"
            )
        seen_template_ids.add(claim_id)
        templates = row["operation_templates"]
        if claim is None or claim_id in completed:
            # A committed claim is no longer selectable.  A retraction also
            # leaves active World authority, so validate its frozen template
            # structurally without comparing it to the post-action state.
            legal_operations = frozenset(templates)
            if legal_operations not in {
                frozenset({"publish"}),
                frozenset({"correct", "retract"}),
            }:
                raise DeterministicCompilerError(
                    "private claim compilation binding has invalid operation closure"
                )
        else:
            bound_claim_ids.add(claim_id)
            legal_operations = (
                frozenset({"publish"})
                if claim["state"] == "draft"
                else frozenset({"correct", "retract"})
            )
        if set(templates) != legal_operations:
            raise DeterministicCompilerError(
                "private claim compilation binding omits a selectable operation"
            )
        for operation in sorted(legal_operations):
            template = templates[operation]
            if not isinstance(template, Mapping):
                raise DeterministicCompilerError(
                    "private claim compilation template is not an object"
                )
            normalized = copy.deepcopy(dict(template))
            if operation == "publish":
                valid = not normalized
            elif operation == "correct":
                content = normalized.get("content")
                valid = (
                    set(normalized) == {"content"}
                    and isinstance(content, Mapping)
                    and bool(content)
                )
                if valid and claim is not None:
                    content_schema = _object_schema(
                        {
                            name: {"type": value_type}
                            for name, value_type in claim["content_schema"].items()
                        },
                        required=tuple(claim["content_schema"]),
                    )
                    try:
                        _validate_schema_arguments(content_schema, content)
                    except SemanticDecisionError as exc:
                        raise DeterministicCompilerError(
                            "private corrected-claim content violates World schema"
                        ) from exc
            else:
                reason = normalized.get("reason")
                valid = (
                    set(normalized) == {"reason"}
                    and isinstance(reason, str)
                    and bool(reason.strip())
                )
            if not valid:
                raise DeterministicCompilerError(
                    f"private {operation} claim compilation template is malformed"
                )
            output[(claim_id, operation)] = normalized
    if bound_claim_ids != set(authority_by_id) - completed:
        raise DeterministicCompilerError(
            "private claim compilation binding does not close over World authority"
        )
    return output


def _listing_claim_authority_projection(
    semantic_variant: str | None,
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    """Expose one operation-complete claim intent for each legal business choice."""

    if semantic_variant not in _CLAIM_VARIANTS:
        return None
    authority = _listing_claim_authority(context)
    if authority is None:
        return None
    claims, evidence_records = authority
    compilation_binding = _claim_compilation_binding(context, claims)
    operation = str(semantic_variant).removeprefix("claim_")
    required_states = (
        frozenset({"draft"}) if operation == "publish" else frozenset({"published", "corrected"})
    )
    completed = _completed_listing_claim_ids(context)
    candidates = tuple(
        row
        for row in claims
        if row["state"] in required_states and row["claim_id"] not in completed
    )
    if not candidates:
        return None
    properties: dict[str, Any] = {
        "claim_id": {
            "type": "string",
            "enum": sorted(row["claim_id"] for row in candidates),
        },
    }
    required = ["claim_id"]
    if operation in {"publish", "correct"}:
        candidate_subjects = {row["subject"] for row in candidates}
        evidence_record_ids = sorted(
            row["record_id"] for row in evidence_records if row["subject_id"] in candidate_subjects
        )
        if not evidence_record_ids:
            return None
        properties["evidence_record_ids"] = {
            "type": "array",
            "items": {"type": "string", "enum": evidence_record_ids},
            "minItems": 1,
        }
        required.append("evidence_record_ids")
    if operation == "correct" and compilation_binding is None:
        content_shapes = {tuple(sorted(row["content_schema"].items())) for row in candidates}
        if len(content_shapes) != 1:
            raise PlatformContractError(
                "correctable claims require one public content shape per intent"
            )
        content_schema = dict(next(iter(content_shapes)))
        properties["content"] = _object_schema(
            {name: {"type": value_type} for name, value_type in content_schema.items()},
            required=tuple(content_schema),
        )
        required.append("content")
    if operation == "retract":
        if compilation_binding is None:
            properties["reason"] = _TEXT
            required.append("reason")
        candidate_subjects = {row["subject"] for row in candidates}
        evidence_record_ids = sorted(
            row["record_id"] for row in evidence_records if row["subject_id"] in candidate_subjects
        )
        if compilation_binding is not None and not evidence_record_ids:
            return None
        properties["evidence_record_ids"] = {
            "type": "array",
            "items": {"type": "string", "enum": evidence_record_ids},
            **({"minItems": 1} if compilation_binding is not None else {}),
        }
        if compilation_binding is not None:
            required.append("evidence_record_ids")
    descriptions = {
        "publish": "Publish one selected draft claim with grounded World evidence.",
        "correct": (
            "Correct one selected published claim using grounded World evidence."
            if compilation_binding is not None
            else "Correct one selected published claim with replacement content and grounded World evidence."
        ),
        "retract": (
            "Retract one selected published claim using grounded World evidence."
            if compilation_binding is not None
            else "Retract one selected published claim for a stated evidence-grounded reason."
        ),
    }
    return _AuthorityIntentProjection(
        description=(
            descriptions[operation]
            + " CommerceWorld binds listing id, subject, and authenticated owner."
        ),
        parameters=_object_schema(properties, required=required),
    )


def _settlement_authority_projection(
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    """Project the commercial choice for authenticated settlement authority."""

    negotiated = context.get("negotiated_settlement_authority")
    if isinstance(negotiated, Mapping):
        authority_payload = negotiated.get("payload")
        actor_id = context.get("actor_id")
        negotiated_required = {
            "negotiation_id",
            "order_id",
            "buyer_id",
            "merchant_id",
            "sku_id",
            "qty",
            "agreed_price",
        }
        if (
            not isinstance(authority_payload, Mapping)
            or not isinstance(actor_id, str)
            or not actor_id.startswith("buyer:")
            or not negotiated_required.issubset(authority_payload)
        ):
            raise PlatformContractError("accepted negotiation settlement authority is malformed")
        try:
            preview_actor_terminal_action(
                Envelope(
                    msg_id="semantic-authority-preview",
                    ts="2000-01-01T00:00:00Z",
                    from_=actor_id,
                    to="platform:psp",
                    in_reply_to="semantic-inbound",
                    idempotency_key="semantic-authority-preview",
                    action={
                        "kind": "platform.settle_payment",
                        "payload": dict(authority_payload),
                    },
                )
            )
        except ActorTerminalContractError as exc:
            raise DeterministicCompilerError(
                "accepted negotiation settlement authority violates actor terminal"
            ) from exc
        return _AuthorityIntentProjection(
            description=(
                "Settle the exact accepted negotiation currently delivered by "
                "CommerceWorld. Agent binds all agreement and actor identities."
            ),
            parameters=_object_schema(
                {
                    "settlement_choice": {
                        "type": "string",
                        "enum": ["settle_accepted_agreement"],
                    }
                },
                required=("settlement_choice",),
            ),
        )
    supply = context.get("supply_settlement_authority")
    if isinstance(supply, Mapping):
        states = supply.get("states")
        options = supply.get("settlement_options")
        buyer_id = supply.get("buyer_id")
        if (
            not isinstance(options, list)
            or not isinstance(buyer_id, str)
            or not buyer_id.startswith("buyer:")
        ):
            raise PlatformContractError("supply settlement authority is malformed")
        sku_ids = sorted(
            {
                str(row["sku_id"])
                for row in options
                if isinstance(row, Mapping)
                and isinstance(row.get("sku_id"), str)
                and row["sku_id"].strip()
            }
        )
        if not sku_ids:
            return None
        if len(sku_ids) != len(options):
            raise PlatformContractError("supply settlement options are not uniquely keyed")
        for sku_id in sku_ids:
            state_matches = [
                row for row in states if isinstance(row, Mapping) and row.get("sku_id") == sku_id
            ]
            option_matches = [
                row for row in options if isinstance(row, Mapping) and row.get("sku_id") == sku_id
            ]
            if len(state_matches) != 1 or len(option_matches) != 1:
                raise PlatformContractError("supply settlement SKU authority is not one-to-one")
            state = state_matches[0]
            option = option_matches[0]
            merchant_id = state.get("merchant_id")
            cents = state.get("unit_price_cents")
            authority_id = option.get("authority_id")
            authority_digest = option.get("authority_digest")
            currency = option.get("currency", "USD")
            if (
                not isinstance(merchant_id, str)
                or not merchant_id.startswith("merchant:")
                or isinstance(cents, bool)
                or not isinstance(cents, int)
                or cents < 0
                or isinstance(state.get("available_qty"), bool)
                or not isinstance(state.get("available_qty"), int)
                or int(state["available_qty"]) <= 0
                or not isinstance(authority_id, str)
                or not authority_id.strip()
                or not isinstance(authority_digest, str)
                or not authority_digest.strip()
                or not isinstance(currency, str)
                or not currency.strip()
                or option.get("merchant_id") != merchant_id
                or option.get("unit_price_cents") != cents
                or option.get("available_qty") != state.get("available_qty")
                or option.get("supply_version") != state.get("version")
            ):
                raise PlatformContractError("supply settlement option is malformed")
            try:
                preview_actor_terminal_action(
                    Envelope(
                        msg_id="semantic-authority-preview",
                        ts="2000-01-01T00:00:00Z",
                        from_=buyer_id,
                        to="platform:psp",
                        in_reply_to="semantic-inbound",
                        idempotency_key="semantic-authority-preview",
                        action={
                            "kind": "platform.settle_payment",
                            "payload": {
                                "supply_authority_id": authority_id,
                                "supply_authority_digest": authority_digest,
                                "sku_id": sku_id,
                                "qty": 1,
                                "allow_partial": False,
                            },
                        },
                    )
                )
            except ActorTerminalContractError as exc:
                raise DeterministicCompilerError(
                    "supply settlement option violates actor terminal"
                ) from exc
        return _AuthorityIntentProjection(
            description=(
                "Purchase one SKU from the current authoritative supply reply. "
                "Choose quantity and partial-fill policy. The Agent forwards "
                "the World authority that binds actor, merchant, order, and "
                "exact current price."
            ),
            parameters=_object_schema(
                {
                    "sku_id": {"type": "string", "enum": sku_ids},
                    "qty": _POSITIVE_INTEGER,
                    "allow_partial": _BOOLEAN,
                },
                required=("sku_id", "qty", "allow_partial"),
            ),
        )
    return None


def _shipment_authority_projection(
    semantic_variant: str | None,
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    """Return a conditionally complete intent for one shipment resolution."""

    authority = context.get("shipment_resolution_authority")
    if not isinstance(authority, Mapping) or semantic_variant is None:
        return None
    shipment_id = authority.get("shipment_id")
    if not isinstance(shipment_id, str) or not shipment_id.strip():
        raise PlatformContractError("shipment resolution authority is malformed")
    if semantic_variant in {"wait", "refund"}:
        return _AuthorityIntentProjection(
            parameters=_object_schema({}, required=()),
        )
    if semantic_variant != "replacement":
        return None
    raw_candidates = authority.get("replacement_candidate_sku_ids")
    candidates = (
        sorted({str(value) for value in raw_candidates if isinstance(value, str) and value.strip()})
        if isinstance(raw_candidates, (list, tuple))
        else []
    )
    if not candidates:
        return None
    return _AuthorityIntentProjection(
        parameters=_object_schema(
            {
                "replacement_sku_id": {
                    "type": "string",
                    "enum": candidates,
                }
            },
            required=("replacement_sku_id",),
        ),
    )


def _after_sales_order_authority(
    binding: _ActionBinding,
    context: Mapping[str, Any],
) -> AfterSalesOrderAuthority | None:
    """Return actor-matched authority for an order-bearing after-sales route."""

    if binding.operation not in AFTER_SALES_ORDER_BOUND_OPERATIONS:
        return None
    value = context.get("after_sales_order_authority")
    if value is None:
        return None
    if not isinstance(value, AfterSalesOrderAuthority):
        raise FrameworkAuthorityError("after-sales order authority is malformed")
    actor_id = context.get("actor_id")
    if actor_id != value.actor_id:
        raise FrameworkAuthorityError("after-sales order authority differs from the current actor")
    return value


def _after_sales_order_authority_projection(
    binding: _ActionBinding,
    context: Mapping[str, Any],
) -> _AuthorityIntentProjection | None:
    """Expose only business choices legal under current order authority."""

    authority = _after_sales_order_authority(binding, context)
    if authority is None or not authority.current_order_ids:
        return None
    properties: dict[str, Any] = {}
    required: list[str] = []
    if len(authority.current_order_ids) > 1:
        # Multiple simultaneous choices are model-visible public references.
        # The enum is both provider guidance and compile-time authority.
        properties["order_id"] = {
            "type": "string",
            "enum": list(authority.current_order_ids),
        }
        required.append("order_id")
    if binding.operation == _AFTER_SALES_RECONCILIATION_OPERATION:
        properties["reason"] = copy.deepcopy(_TEXT)
        required.append("reason")
        description = (
            "Request ledger reconciliation for the current authorized order. "
            "CommerceWorld Agent binds the order reference."
        )
    else:
        description = (
            "Read this history for the current authorized order. CommerceWorld "
            "Agent binds the order reference."
        )
    return _AuthorityIntentProjection(
        parameters=_object_schema(properties, required=tuple(required)),
        description=description,
    )


class AgentRouteRegistry:
    """Agent-private business-intent catalogue and deterministic VCP compiler."""

    def __init__(self) -> None:
        reads = _world_read_specs()
        self._read_by_name = {spec.name: (spec, tool) for spec, tool in reads}
        bindings = (*_platform_bindings(), *_specialized_bindings())
        if len({binding.spec.name for binding in bindings}) != len(bindings):
            raise ValueError("Agent business intent names must be unique")
        self._action_by_name = {binding.spec.name: binding for binding in bindings}
        self._route_registry = RouteRegistry(tuple(binding.route for binding in bindings))
        self._finish_spec = AgentBusinessIntentSpec(
            _FINISH_INTENT,
            "End this turn without emitting a commerce action.",
            _object_schema({"reason": _TEXT}),
            category="control",
        )

    def business_operation_for_source(self, source_name: str) -> str:
        """Return the model-visible operation behind one Agent-private schema.

        This is the sole naming seam used by the business-decision bridge.
        It deliberately distinguishes identical VCP action kinds that reach
        different services, while keeping every service address Agent-private.
        """

        if source_name in self._read_by_name:
            suffix = source_name.removeprefix("read_")
            return f"observe_{suffix}"
        if source_name == self._finish_spec.name:
            return "finish"
        try:
            binding = self._action_by_name[source_name]
        except KeyError as exc:
            raise FrameworkAuthorityError("business decision source is not registered") from exc
        return binding.operation

    def business_route_binding_for_source(
        self,
        source_name: str,
    ) -> RouteBinding | None:
        """Project one action source into its sole Agent-private route binding."""

        binding = self._action_by_name.get(source_name)
        if binding is None:
            if source_name in self._read_by_name or source_name == self._finish_spec.name:
                return None
            raise FrameworkAuthorityError("business decision source is not registered")
        return binding.route

    def business_route_registry_digest(self) -> str:
        """Hash the one internal operation-to-route registry."""

        rows = []
        for source_name in sorted(self._action_by_name):
            binding = self.business_route_binding_for_source(source_name)
            assert binding is not None
            rows.append(
                {
                    "operation": binding.operation,
                    "action_kind": binding.action_kind,
                    "destination": binding.destination,
                    "roles": sorted(binding.roles),
                    "source_name": binding.source_name,
                }
            )
        rendered = json.dumps(
            rows,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def business_route_registry(self) -> RouteRegistry:
        """Return the complete immutable Agent-private route registry."""

        return self._route_registry

    def intent_specs_for(
        self,
        role: str,
        allowed_action_kinds: Sequence[str] | frozenset[str],
        *,
        include_world_reads: bool = True,
        activated_skill_names: Sequence[str] | None = None,
        allow_actor_reports: bool = True,
        include_finish: bool = True,
        task_result_contract: Mapping[str, Any] | None = None,
        allowed_routes: Sequence[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
        bind_cart_quote_authority: bool = False,
        actor_context: Mapping[str, Any] | None = None,
    ) -> tuple[AgentBusinessIntentSpec, ...]:
        """Return deterministic role- and Skill-scoped private intent schemas.

        ``activated_skill_names=None`` retains the no-selector compatibility
        surface.  Without a public phase, a concrete sequence fails closed to
        actions owned by those selected built-in Skill bundles.  With a public
        phase, its exact registered routes replace the broader Skill action
        surface while Skills continue to scope Agent-owned World reads.
        """

        if role not in {"buyer", "merchant"}:
            raise FrameworkAuthorityError("business intents require a buyer or merchant role")
        context = dict(actor_context or {})
        route_scope = _coerce_allowed_routes(allowed_routes)
        phase_actor_id = context.get("actor_id")
        phase_is_bound = (
            isinstance(phase_actor_id, str) and phase_actor_id.split(":", 1)[0] in _ACTOR_ROLES
        )
        supply_authority = context.get("supply_settlement_authority")
        supply_states = (
            supply_authority.get("states") if isinstance(supply_authority, Mapping) else None
        )
        supply_options = (
            supply_authority.get("settlement_options")
            if isinstance(supply_authority, Mapping)
            else None
        )
        supply_has_no_purchasable_option = bool(
            isinstance(supply_states, list)
            and supply_states
            and isinstance(supply_options, list)
            and not supply_options
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("available_qty"), int)
                and not isinstance(row.get("available_qty"), bool)
                and int(row["available_qty"]) == 0
                for row in supply_states
            )
        )
        grounding_required_routes = _commerce_grounding_routes(
            context,
            pending=False,
        )
        grounding_pending_routes = _commerce_grounding_routes(
            context,
            pending=True,
        )
        if not grounding_pending_routes.issubset(grounding_required_routes):
            raise FrameworkAuthorityError(
                "pending commerce grounding routes exceed required routes"
            )
        grounded_bound_routes: set[tuple[str, str]] = set()
        if _commerce_action_authority(context, kind="catalog_update") is not None:
            grounded_bound_routes.add(CATALOG_UPDATE_ROUTE)
        if _commerce_action_authority(context, kind="listing_claim") is not None:
            grounded_bound_routes.add(LISTING_CLAIM_ROUTE)
        if grounding_required_routes - frozenset(grounded_bound_routes) != grounding_pending_routes:
            raise FrameworkAuthorityError(
                "commerce grounding pending state is inconsistent with authority"
            )
        if grounding_pending_routes and not include_world_reads:
            raise FrameworkAuthorityError(
                "commerce grounding is pending but the phase forbids World reads"
            )
        if (
            phase_is_bound
            and route_scope is not None
            and (
                "platform.settle_payment",
                "platform:psp",
            )
            in route_scope
        ):
            cert_id = context.get("match_certificate_id")
            if (
                _settlement_authority_projection(context) is None
                and not (isinstance(cert_id, str) and bool(cert_id.strip()))
                and not supply_has_no_purchasable_option
            ):
                # A public phase promises that this exact business route is
                # executable now.  Falling back to the retained wire schema
                # would spend a provider call on an action Platform must reject.
                raise FrameworkAuthorityError(
                    "public settlement phase has no authoritative settlement binding"
                )
        if (
            phase_is_bound
            and route_scope is not None
            and (
                "commerce.resolve_shipment",
                "platform:fulfillment",
            )
            in route_scope
            and not isinstance(context.get("shipment_resolution_authority"), Mapping)
        ):
            raise FrameworkAuthorityError(
                "public shipment phase has no authoritative shipment binding"
            )
        # A spawn-frozen public phase is a narrower authority than the broad
        # Skill catalogue.  Its exact registered routes therefore own the
        # action surface for that turn.  Skills continue to control prompt
        # bodies and World reads, while Runtime and Platform still validate
        # the compiled action.  This lets a legitimate cross-domain phase,
        # such as a delivered shipment entering after-sales handling, expose
        # its declared action without granting the Skill's ambient actions.
        allowed = (
            frozenset(str(kind) for kind in allowed_action_kinds)
            if route_scope is not None
            else self.allowed_action_kinds_for(
                role,
                allowed_action_kinds,
                activated_skill_names=activated_skill_names,
            )
        )
        intent_specs: list[AgentBusinessIntentSpec] = []
        if include_world_reads:
            allowed_reads = self.allowed_read_source_names_for(
                role,
                activated_skill_names=activated_skill_names,
            )
            intent_specs.extend(
                spec for name, (spec, _tool) in self._read_by_name.items() if name in allowed_reads
            )
        result_contract = _coerce_task_result_contract(task_result_contract)
        if route_scope is not None:
            registered = frozenset(
                (binding.action_kind, binding.destination)
                for binding in self._action_by_name.values()
                if role in binding.roles
            )
            unknown = route_scope - registered
            if unknown:
                raise FrameworkAuthorityError(
                    "public task contract names an unregistered actor route"
                )
        exposed_routes: set[tuple[str, str]] = set()
        for binding in self._action_by_name.values():
            if binding.action_kind not in allowed or role not in binding.roles:
                continue
            if (
                route_scope is not None
                and (binding.action_kind, binding.destination) not in route_scope
            ):
                continue
            if route_scope is None and not _route_is_skill_authorized(
                role,
                binding,
                activated_skill_names=activated_skill_names,
            ):
                continue
            if (
                binding.action_kind == "commerce.resolve_shipment"
                and binding.destination == "platform:fulfillment"
            ):
                shipment_bound = isinstance(context.get("shipment_resolution_authority"), Mapping)
                if shipment_bound:
                    # Replace the wire-shaped generic intent with three
                    # conditionally complete business choices.  A replacement
                    # choice cannot omit its SKU, while wait/refund cannot
                    # smuggle one onto the wire.
                    if binding.semantic_variant is None:
                        continue
                    shipment_projection = _shipment_authority_projection(
                        binding.semantic_variant,
                        context,
                    )
                    if shipment_projection is None:
                        continue
                    intent_specs.append(
                        _bind_authority_projection(
                            binding,
                            shipment_projection,
                        )
                    )
                    exposed_routes.add((binding.action_kind, binding.destination))
                    continue
                if binding.semantic_variant is not None:
                    continue
            if (
                binding.action_kind == "commerce.apply_listing_claim"
                and binding.destination == "platform:claims"
            ):
                route = LISTING_CLAIM_ROUTE
                claim_authority = _listing_claim_authority(context)
                if binding.semantic_variant in _CLAIM_VARIANTS:
                    claim_projection = _listing_claim_authority_projection(
                        binding.semantic_variant,
                        context,
                    )
                    if claim_projection is None:
                        continue
                    intent_specs.append(
                        _bind_authority_projection(
                            binding,
                            claim_projection,
                        )
                    )
                    exposed_routes.add((binding.action_kind, binding.destination))
                    continue
                if route in grounding_required_routes:
                    # Under grounded compilation the operation-complete intents
                    # replace the generic wire union.  Before World reads have
                    # completed, advertise reads only and never expose a
                    # terminal intent that Platform would have to reject.
                    exposed_routes.add(route)
                    continue
                if claim_authority is not None:
                    # The branch-complete intents replace the generic wire union.
                    # Counting the declared route here keeps a completed claim
                    # phase valid even when only its result/response intent
                    # remains model-visible.
                    exposed_routes.add((binding.action_kind, binding.destination))
                    continue
            catalog_update_projection = (
                _catalog_update_authority_projection(context)
                if binding.operation == "update_listing"
                else None
            )
            if catalog_update_projection is not None:
                intent_specs.append(
                    _bind_authority_projection(
                        binding,
                        catalog_update_projection,
                    )
                )
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if (
                binding.action_kind,
                binding.destination,
            ) == CATALOG_UPDATE_ROUTE and CATALOG_UPDATE_ROUTE in grounding_required_routes:
                exposed_routes.add(CATALOG_UPDATE_ROUTE)
                continue
            if binding.action_kind in ACTOR_REPORT_ACTION_KINDS:
                # Runtime evidence is not an ambient escape hatch from a
                # commerce workflow.  It is model-visible only when a public,
                # phase-active task contract names exactly one result action.
                if not allow_actor_reports or result_contract is None:
                    continue
                if (
                    result_contract["active"] is not True
                    or binding.action_kind != result_contract["action_kind"]
                ):
                    continue
                intent_specs.append(
                    _bind_authority_projection(
                        binding,
                        _task_result_authority_projection(result_contract),
                    )
                )
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if binding.action_kind == "commerce.send_message" and isinstance(
                context.get("message_payload_authority"), Mapping
            ):
                # Communication correlation and authoritative state echoes are
                # compiled by Agent.  The model chooses only whether to notify
                # the current participant.
                intent_specs.append(
                    AgentBusinessIntentSpec(
                        name=binding.spec.name,
                        description=(
                            "Notify the current business counterparty using the "
                            "Agent-bound lifecycle context."
                        ),
                        parameters=_object_schema(),
                        category="action",
                    )
                )
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if binding.action_kind == "commerce.get_sku" and isinstance(
                context.get("catalog_lookup_request_id"), str
            ):
                intent_specs.append(
                    AgentBusinessIntentSpec(
                        name=binding.spec.name,
                        description="Read one owned listing for the current inquiry.",
                        parameters=_object_schema(
                            {"sku_id": _TEXT},
                            required=("sku_id",),
                        ),
                        category="action",
                    )
                )
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            inquiry_projection = (
                _inquiry_response_authority_projection(context)
                if binding.operation == "respond_inquiry"
                else None
            )
            if inquiry_projection is not None:
                intent_specs.append(_bind_authority_projection(binding, inquiry_projection))
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            after_sales_order_authority = _after_sales_order_authority(
                binding,
                context,
            )
            if after_sales_order_authority is not None:
                projection = _after_sales_order_authority_projection(
                    binding,
                    context,
                )
                if projection is not None:
                    intent_specs.append(_bind_authority_projection(binding, projection))
                # An exhausted sequential authority advertises no action, but
                # remains a resolved route rather than falling back to the
                # generic protocol-shaped order_id schema.
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            settlement_projection = (
                _settlement_authority_projection(context)
                if binding.operation == "settle_payment"
                else None
            )
            if settlement_projection is not None:
                intent_specs.append(
                    _bind_authority_projection(
                        binding,
                        settlement_projection,
                    )
                )
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if (
                binding.action_kind == "platform.settle_payment"
                and binding.destination == "platform:psp"
                and supply_has_no_purchasable_option
            ):
                # The route remains part of the public phase, but the current
                # World reply proves that no SKU is purchasable.  Do not fall
                # back to a generic settlement intent that Platform must
                # reject, and let the actor use the phase's decline route.
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if (
                binding.action_kind == "commerce.request_cart_quote"
                and binding.destination == "platform:checkout"
                and (
                    bind_cart_quote_authority
                    or (
                        route_scope is not None
                        and (binding.action_kind, binding.destination) in route_scope
                    )
                )
            ):
                cart_projection = _cart_quote_authority_projection(
                    role,
                    context,
                )
                if cart_projection is None:
                    if not include_world_reads:
                        raise FrameworkAuthorityError(
                            "buyer cart phase requires World grounding but forbids World reads"
                        )
                else:
                    intent_specs.append(
                        _bind_authority_projection(
                            binding,
                            cart_projection,
                        )
                    )
                # A buyer begins with reads only.  The route remains declared
                # and becomes model-visible after this exact turn observes
                # at least one listing from World.
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if (
                binding.action_kind == "platform.checkout_cart"
                and binding.destination == "platform:checkout"
                and route_scope is not None
                and (binding.action_kind, binding.destination) in route_scope
            ):
                quote_id = context.get("cart_quote_id")
                if not isinstance(quote_id, str) or not quote_id.strip():
                    raise FrameworkAuthorityError(
                        "public checkout phase has no authenticated quote authority"
                    )
                intent_specs.append(binding.spec)
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            if (
                binding.action_kind == "commerce.search"
                and binding.destination == "platform:aggregator"
                and context.get("search_filters_allowed") is False
            ):
                model_schema = binding.spec.business_parameters or binding.spec.parameters
                model_properties = model_schema.get("properties")
                if (
                    not isinstance(model_properties, Mapping)
                    or "query" not in model_properties
                ):
                    raise FrameworkAuthorityError(
                        "complete-candidate search has no model-facing query"
                    )
                intent_specs.append(
                    AgentBusinessIntentSpec(
                        name=binding.spec.name,
                        description=(
                            "Search the complete candidate set with a broad business query."
                        ),
                        parameters=binding.spec.parameters,
                        business_parameters=_object_schema(
                            {"query": copy.deepcopy(model_properties["query"])},
                            required=("query",),
                        ),
                        category=binding.spec.category,
                    )
                )
                exposed_routes.add((binding.action_kind, binding.destination))
                continue
            intent_specs.append(binding.spec)
            exposed_routes.add((binding.action_kind, binding.destination))
        if route_scope is not None:
            missing = route_scope - exposed_routes
            if missing:
                rendered = ", ".join(
                    f"{kind} -> {destination}" for kind, destination in sorted(missing)
                )
                raise FrameworkAuthorityError(
                    "public task routes are unavailable under the active actor "
                    f"or evidence authority: {rendered}"
                )
        if include_finish:
            intent_specs.append(self._finish_spec)
        return tuple(intent_specs)

    def registered_action_routes_for(
        self,
        role: str,
    ) -> frozenset[tuple[str, str]]:
        """Return the exact public action routes registered for ``role``.

        This is a declaration audit surface.  It does not select Skills,
        expose a private route to a model, or compile an action.  Public task contract
        validation uses it to prove that every named route is implementable
        without consulting an ideal trajectory or oracle.
        """

        if role not in _ACTOR_ROLES:
            raise SemanticDecisionError("business intents require a buyer or merchant role")
        return frozenset(
            (binding.action_kind, binding.destination)
            for binding in self._action_by_name.values()
            if role in binding.roles
        )

    def allowed_read_source_names_for(
        self,
        role: str,
        *,
        activated_skill_names: Sequence[str] | None,
    ) -> frozenset[str]:
        """Return Agent-private World-read intents owned by selected Skills."""

        if role not in _ACTOR_ROLES:
            raise SemanticDecisionError("business intents require a buyer or merchant role")
        if activated_skill_names is None:
            return frozenset(self._read_by_name)
        known = _SKILL_READ_INTENTS[role]
        allowed: set[str] = set()
        for name in activated_skill_names:
            try:
                allowed.update(known[str(name)])
            except KeyError as exc:
                raise SemanticDecisionError(
                    f"semantic read ownership is not registered for Skill {name!r}"
                ) from exc
        unknown = allowed - set(self._read_by_name)
        if unknown:  # pragma: no cover - import-time declarations are static
            raise SemanticDecisionError(
                "business observation ownership names an unknown intent source"
            )
        return frozenset(allowed)

    def allowed_action_kinds_for(
        self,
        role: str,
        declared_action_kinds: Sequence[str] | frozenset[str],
        *,
        activated_skill_names: Sequence[str] | None,
    ) -> frozenset[str]:
        """Intersect a role declaration with deterministic Skill ownership."""

        if role not in _ACTOR_ROLES:
            raise SemanticDecisionError("business intents require a buyer or merchant role")
        declared = frozenset(str(kind) for kind in declared_action_kinds)
        if activated_skill_names is None:
            return declared
        known = _SKILL_ACTION_KINDS[role]
        permitted = set(_ALWAYS_SKILL_SCOPED_ACTIONS)
        for name in activated_skill_names:
            try:
                permitted.update(known[str(name)])
            except KeyError as exc:
                raise SemanticDecisionError(
                    f"semantic action ownership is not registered for Skill {name!r}"
                ) from exc
        return declared & frozenset(permitted)

    def world_read_tool_name(self, source_name: str) -> str | None:
        """Return the WorldTools name behind an Agent-private read intent."""

        row = self._read_by_name.get(source_name)
        return None if row is None else row[1]

    def compile_action(
        self,
        *,
        role: str,
        source_name: str,
        arguments: Mapping[str, Any],
        allowed_action_kinds: Sequence[str] | frozenset[str],
        activated_skill_names: Sequence[str] | None = None,
        allow_actor_reports: bool = True,
        actor_context: Mapping[str, Any] | None = None,
    ) -> SemanticAction:
        """Validate and compile one Agent-internal business operation."""

        try:
            binding = self._action_by_name[source_name]
        except KeyError as exc:
            raise SemanticDecisionError("source is not a registered business action") from exc
        if not isinstance(arguments, Mapping):
            raise SemanticDecisionError("business operation arguments must be an object")
        arguments = dict(arguments)
        context = dict(actor_context or {})
        route_scope = _coerce_allowed_routes(context.get("task_allowed_routes"))
        allowed = (
            frozenset(str(kind) for kind in allowed_action_kinds)
            if route_scope is not None
            else self.allowed_action_kinds_for(
                role,
                allowed_action_kinds,
                activated_skill_names=activated_skill_names,
            )
        )
        if binding.action_kind in ACTOR_REPORT_ACTION_KINDS and not allow_actor_reports:
            raise SemanticDecisionError(
                "actor report action has no registered causal root on this turn"
            )
        if role not in binding.roles or binding.action_kind not in allowed:
            raise SemanticDecisionError("semantic action is outside the actor's role surface")
        if route_scope is None and not _route_is_skill_authorized(
            role,
            binding,
            activated_skill_names=activated_skill_names,
        ):
            raise SemanticDecisionError("semantic action is outside the activated Skill surface")
        if (
            route_scope is not None
            and (binding.action_kind, binding.destination) not in route_scope
        ):
            raise SemanticDecisionError("semantic action is outside the public task route surface")
        if (
            binding.action_kind == "commerce.search"
            and binding.destination == "platform:aggregator"
            and context.get("search_filters_allowed") is False
            and "filters" in arguments
        ):
            raise SemanticDecisionError(
                "complete-candidate search does not accept model-authored filters"
            )
        grounding_required_routes = _commerce_grounding_routes(
            context,
            pending=False,
        )
        binding_route = (binding.action_kind, binding.destination)
        if binding_route in grounding_required_routes:
            authority_kind = (
                "catalog_update" if binding_route == CATALOG_UPDATE_ROUTE else "listing_claim"
            )
            if (
                _commerce_action_authority(
                    context,
                    kind=authority_kind,
                )
                is None
            ):
                raise FrameworkAuthorityError(
                    "commerce action cannot compile before authoritative World grounding"
                )
        if (
            route_scope is not None
            and isinstance(context.get("actor_id"), str)
            and binding.operation == "settle_payment"
            and _settlement_authority_projection(context) is None
            and not (
                isinstance(context.get("match_certificate_id"), str)
                and bool(str(context["match_certificate_id"]).strip())
            )
        ):
            raise FrameworkAuthorityError(
                "public settlement phase has no authoritative settlement binding"
            )
        if (
            route_scope is not None
            and isinstance(context.get("actor_id"), str)
            and binding.action_kind == "commerce.resolve_shipment"
            and binding.destination == "platform:fulfillment"
            and not isinstance(context.get("shipment_resolution_authority"), Mapping)
        ):
            raise FrameworkAuthorityError(
                "public shipment phase has no authoritative shipment binding"
            )
        after_sales_order_authority = _after_sales_order_authority(
            binding,
            context,
        )
        after_sales_order_projection = _after_sales_order_authority_projection(
            binding,
            context,
        )
        if after_sales_order_authority is not None and after_sales_order_projection is None:
            raise FrameworkAuthorityError(
                "after-sales order action has no current authorized order"
            )
        if after_sales_order_projection is not None:
            _validate_schema_arguments(
                after_sales_order_projection.parameters,
                arguments,
            )
        result_contract = _coerce_task_result_contract(context.get("task_result_contract"))
        if binding.action_kind in ACTOR_REPORT_ACTION_KINDS:
            if result_contract is None:
                raise FrameworkAuthorityError(
                    "actor report action requires an explicit public task result contract"
                )
            if (
                result_contract["active"] is not True
                or binding.action_kind != result_contract["action_kind"]
            ):
                raise FrameworkAuthorityError(
                    "actor report action is outside the public task result phase"
                )
            result_projection = _task_result_authority_projection(result_contract)
            _validate_schema_arguments(result_projection.parameters, arguments)
            details: dict[str, Any]
            if result_contract["result_format"] == "named_submission":
                details = {
                    "submission": {
                        "kind": result_contract["submission_kind"],
                        "payload": copy.deepcopy(arguments["payload"]),
                    }
                }
            else:
                details = copy.deepcopy(arguments["payload"])
            payload = {
                "outcome": arguments["outcome"],
                "summary": arguments["summary"],
                "details": details,
            }
            rationale = None
        else:
            is_cart_quote_route = binding.operation == "request_cart_quote"
            cart_quote_projection = (
                _cart_quote_authority_projection(role, context)
                if is_cart_quote_route
                and (
                    route_scope is not None
                    or isinstance(context.get("cart_quote_authority"), Mapping)
                    or isinstance(context.get("cart_quote_request_id"), str)
                )
                else None
            )
            if is_cart_quote_route and route_scope is not None and cart_quote_projection is None:
                raise FrameworkAuthorityError(
                    "public cart quote cannot compile before World-grounded SKU authority"
                )
            cart_quote_authority_bound = is_cart_quote_route and cart_quote_projection is not None
            settlement_projection = (
                _settlement_authority_projection(context)
                if binding.operation == "settle_payment"
                else None
            )
            shipment_projection = (
                _shipment_authority_projection(binding.semantic_variant, context)
                if binding.operation
                in {
                    "wait_for_shipment",
                    "replace_shipment",
                    "refund_shipment",
                }
                else None
            )
            catalog_update_projection = (
                _catalog_update_authority_projection(context)
                if binding.operation == "update_listing"
                else None
            )
            claim_projection = (
                _listing_claim_authority_projection(
                    binding.semantic_variant,
                    context,
                )
                if binding.operation
                in {
                    "publish_listing_claim",
                    "correct_listing_claim",
                    "retract_listing_claim",
                }
                else None
            )
            claim_authority = (
                _listing_claim_authority(context)
                if binding.action_kind == "commerce.apply_listing_claim"
                and binding.destination == "platform:claims"
                else None
            )
            message_payload_authority = (
                context.get("message_payload_authority")
                if binding.action_kind == "commerce.send_message"
                else None
            )
            catalog_lookup_request_id = (
                context.get("catalog_lookup_request_id")
                if binding.action_kind == "commerce.get_sku"
                else None
            )
            inquiry_response_projection = (
                _inquiry_response_authority_projection(context)
                if binding.operation == "respond_inquiry"
                else None
            )
            if (
                binding.semantic_variant in {"wait", "replacement", "refund"}
                and shipment_projection is None
            ):
                raise FrameworkAuthorityError(
                    "shipment resolution choice has no current shipment authority"
                )
            if binding.semantic_variant in _CLAIM_VARIANTS and claim_projection is None:
                raise FrameworkAuthorityError(
                    "listing claim choice has no current public claim authority"
                )
            if (
                claim_authority is not None
                and binding.action_kind == "commerce.apply_listing_claim"
                and binding.semantic_variant not in _CLAIM_VARIANTS
            ):
                raise FrameworkAuthorityError(
                    "generic listing claim union is unavailable under bound authority"
                )
            if after_sales_order_projection is not None:
                _validate_schema_arguments(
                    after_sales_order_projection.parameters,
                    arguments,
                )
            elif settlement_projection is not None:
                _validate_schema_arguments(
                    settlement_projection.parameters,
                    arguments,
                )
            elif shipment_projection is not None:
                _validate_schema_arguments(
                    shipment_projection.parameters,
                    arguments,
                )
            elif catalog_update_projection is not None:
                _validate_schema_arguments(
                    catalog_update_projection.parameters,
                    arguments,
                )
            elif claim_projection is not None:
                _validate_schema_arguments(
                    claim_projection.parameters,
                    arguments,
                )
            elif cart_quote_authority_bound:
                assert cart_quote_projection is not None
                _validate_schema_arguments(
                    cart_quote_projection.parameters,
                    arguments,
                )
            elif isinstance(message_payload_authority, Mapping):
                _validate_schema_arguments(_object_schema(), arguments)
            elif isinstance(catalog_lookup_request_id, str):
                _validate_schema_arguments(
                    _object_schema({"sku_id": _TEXT}, required=("sku_id",)),
                    arguments,
                )
            elif inquiry_response_projection is not None:
                _validate_schema_arguments(
                    inquiry_response_projection.parameters,
                    arguments,
                )
            else:
                # Validate against the model-facing projection, not the local
                # protocol schema.  Otherwise a field the projection hides is
                # merely undocumented: a model that emits it anyway would still
                # have it compiled into the outgoing payload.
                _validate_schema_arguments(
                    binding.spec.business_parameters or binding.spec.parameters,
                    arguments,
                )
            rationale = arguments.pop("decision_rationale", None)
            payload = arguments
        destination = binding.destination
        if binding.payload_mode == "nested_payload":
            payload = arguments["payload"]
        elif binding.payload_mode == "recipient_and_payload":
            message_payload_authority = context.get("message_payload_authority")
            if isinstance(message_payload_authority, Mapping):
                recipient = context.get("reply_recipient_id")
                if not isinstance(recipient, str) or not recipient.strip():
                    raise FrameworkAuthorityError(
                        "Agent-bound participant message has no reply recipient"
                    )
                destination = recipient
                payload = copy.deepcopy(dict(message_payload_authority))
            else:
                destination = str(arguments["recipient_id"])
                payload = arguments["payload"]
            address_parts = destination.split(":")
            if address_parts[0] not in _ACTOR_ROLES or any(
                not part.strip() for part in address_parts
            ):
                raise SemanticDecisionError(
                    "semantic participant messages may target only buyer or merchant actors"
                )
        elif not (binding.action_kind in ACTOR_REPORT_ACTION_KINDS and result_contract is not None):
            payload = arguments
        if not isinstance(payload, Mapping):
            raise SemanticDecisionError("semantic action payload must be an object")

        payload = dict(payload)
        if binding.action_kind == "commerce.get_sku":
            request_id = context.get("catalog_lookup_request_id")
            if isinstance(request_id, str) and request_id.strip():
                payload["request_id"] = request_id
        if after_sales_order_authority is not None:
            payload["order_id"] = after_sales_order_authority.bind(arguments)
        catalog_authority = (
            _commerce_action_authority(
                context,
                kind="catalog_update",
            )
            if (
                binding.action_kind == "commerce.update_listing"
                and binding.destination == "platform:catalog"
            )
            else None
        )
        if catalog_authority is not None:
            payload = {
                "op": "update",
                "sku_id": str(catalog_authority["sku_id"]),
                "fields": {
                    "attributes": copy.deepcopy(dict(arguments["changes"])),
                },
            }

        if binding.semantic_variant in _CLAIM_VARIANTS:
            claim_authority = _listing_claim_authority(context)
            if claim_authority is None:  # guarded before local validation.
                raise FrameworkAuthorityError(
                    "listing claim choice has no current public claim authority"
                )
            claims, evidence_records = claim_authority
            claim_compilation = _claim_compilation_binding(context, claims)
            operation = str(binding.semantic_variant).removeprefix("claim_")
            claim_id = arguments["claim_id"]
            matches = [row for row in claims if row["claim_id"] == claim_id]
            if len(matches) != 1:
                raise SemanticDecisionError(
                    "selected claim is not unique in public claim authority"
                )
            claim = matches[0]
            selected_evidence_ids = list(dict.fromkeys(arguments.get("evidence_record_ids", ())))
            evidence_by_id = {row["record_id"]: row for row in evidence_records}
            if any(
                evidence_id not in evidence_by_id
                or evidence_by_id[evidence_id]["subject_id"] != claim["subject"]
                for evidence_id in selected_evidence_ids
            ):
                raise SemanticDecisionError(
                    "selected evidence is not World-grounded for this claim subject"
                )
            if set(selected_evidence_ids) & set(claim["used_evidence_record_ids"]):
                raise SemanticDecisionError(
                    "selected evidence was already used in this claim history"
                )
            payload = {
                "claim_id": claim["claim_id"],
                "listing_id": claim["listing_id"],
                "operation": operation,
                "subject": claim["subject"],
            }
            if operation in {"publish", "correct"} or (
                operation == "retract" and claim_compilation is not None
            ):
                payload["evidence_record_ids"] = selected_evidence_ids
            if claim_compilation is not None:
                try:
                    compiled_fields = claim_compilation[(claim_id, operation)]
                except KeyError as exc:  # validated inventory should make this unreachable.
                    raise DeterministicCompilerError(
                        "selected claim operation has no private compiler binding"
                    ) from exc
                payload.update(copy.deepcopy(compiled_fields))
            elif operation == "correct":
                payload["content"] = copy.deepcopy(dict(arguments["content"]))
            elif operation == "retract":
                payload["reason"] = arguments["reason"]
                if selected_evidence_ids:
                    payload["evidence_record_ids"] = selected_evidence_ids
        if binding.action_kind == "delegate.reject_purchase":
            mandate_id = context.get("mandate_id", payload.get("mandate_id"))
            if not isinstance(mandate_id, str) or not mandate_id.strip():
                raise FrameworkAuthorityError(
                    "semantic purchase rejection has no active mandate authority"
                )
            payload["mandate_id"] = mandate_id
            principal_id = context.get("principal_id")
            if principal_id is not None:
                if not _actor_id_has_role(principal_id, frozenset({"consumer"})):
                    raise FrameworkAuthorityError(
                        "semantic purchase rejection has an invalid principal route"
                    )
                destination = principal_id

        if binding.action_kind == "commerce.respond_inquiry":
            reply_recipient_id = context.get("reply_recipient_id")
            if reply_recipient_id is not None:
                if not _actor_id_has_role(
                    reply_recipient_id,
                    frozenset({"buyer", "merchant"}),
                ):
                    raise FrameworkAuthorityError(
                        "semantic inquiry response has an invalid participant route"
                    )
                destination = reply_recipient_id

        if binding.action_kind == "commerce.send_message":
            message_payload = context.get("message_payload_authority")
            if isinstance(message_payload, Mapping):
                payload = copy.deepcopy(dict(message_payload))

        if (
            binding.action_kind == "commerce.search"
            and binding.destination == "platform:aggregator"
        ):
            # Retrieval cardinality is a scenario/Platform policy, not a model
            # preference.  A standalone legacy Agent may still supply its old
            # explicit limit, while scenario-built Agents deterministically
            # override it with population.matching.top_k.
            search_limit = context.get("search_limit", payload.get("limit"))
            if (
                isinstance(search_limit, bool)
                or not isinstance(search_limit, int)
                or search_limit <= 0
            ):
                raise FrameworkAuthorityError(
                    "semantic marketplace search has no positive framework limit"
                )
            payload["limit"] = search_limit
            mandate_id = context.get("mandate_id", payload.get("mandate_id"))
            if isinstance(mandate_id, str) and mandate_id.strip():
                payload["mandate_id"] = mandate_id
            benchmark_task_id = context.get("benchmark_task_id")
            if isinstance(benchmark_task_id, str) and benchmark_task_id.strip():
                payload["benchmark_task_id"] = benchmark_task_id
            hard_constraint_ids = context.get("hard_constraint_ids")
            if isinstance(hard_constraint_ids, (list, tuple)):
                normalized_constraint_ids = [
                    constraint_id
                    for constraint_id in hard_constraint_ids
                    if isinstance(constraint_id, str) and constraint_id.strip()
                ]
                if normalized_constraint_ids:
                    payload["hard_constraint_ids"] = normalized_constraint_ids
            if payload.get("filters"):
                payload["filter_contract"] = "typed_constraints.v1"

        # Aggregator acceptance is always a high-level choice.  Certificate
        # fields, revisions and order ids are Platform/World authority and are
        # never trusted from provider output, even when a legacy scripted
        # adapter includes them.  The Agent binds its frozen mandate and emits
        # the exact compact terminal shape.
        if (
            binding.action_kind == "commerce.accept_offer"
            and binding.destination == "platform:aggregator"
        ):
            mandate_id = context.get("mandate_id", payload.get("mandate_id"))
            if not isinstance(mandate_id, str) or not mandate_id.strip():
                raise FrameworkAuthorityError(
                    "semantic aggregator acceptance has no active mandate authority"
                )
            offer_id = payload["offer_id"]
            ranked = context.get("rank_offers")
            candidates = ranked.get("candidates") if isinstance(ranked, Mapping) else None
            if isinstance(candidates, list):
                matches = [
                    row
                    for row in candidates
                    if isinstance(row, Mapping) and row.get("offer_id") == offer_id
                ]
                if len(matches) != 1:
                    raise SemanticDecisionError(
                        "selected offer is not unique in the active ranked response"
                    )
                candidate = matches[0]
                required_candidate_fields = (
                    "session_id",
                    "offer_digest",
                    "sku_id",
                    "merchant_id",
                    "qty",
                    "unit_price_cents",
                    "currency",
                    "catalog_revision",
                    "inventory_revision",
                )
                session_digest = ranked.get("session_digest")
                if (
                    not isinstance(session_digest, str)
                    or not session_digest.strip()
                    or any(name not in candidate for name in required_candidate_fields)
                ):
                    raise PlatformContractError(
                        "active ranked response lacks certified offer authority"
                    )
                payload = {
                    "mandate_id": mandate_id,
                    "offer_id": offer_id,
                    "session_id": candidate["session_id"],
                    "session_digest": session_digest,
                    "offer_digest": candidate["offer_digest"],
                    "sku_id": candidate["sku_id"],
                    "merchant_id": candidate["merchant_id"],
                    "qty": candidate["qty"],
                    "unit_price_cents": candidate["unit_price_cents"],
                    "currency": candidate["currency"],
                    "catalog_revision": candidate["catalog_revision"],
                    "inventory_revision": candidate["inventory_revision"],
                    "order_id": f"ord-{mandate_id}-{offer_id}",
                }
            else:
                # Legacy direct tests and non-session adapters retain the
                # narrow compact route.  A real ranked response always takes
                # the certified branch above so repeated searches remain
                # unambiguous.
                payload = {
                    "mandate_id": mandate_id,
                    "offer_id": offer_id,
                }

        if (
            binding.action_kind == "platform.settle_payment"
            and binding.destination == "platform:psp"
        ):
            cert_id = context.get("match_certificate_id")
            if isinstance(cert_id, str) and cert_id.strip():
                # A high-level "settle" choice on a match-certificate turn is
                # compiled to the exact Platform-owned certificate.  Provider
                # output cannot redirect payment to another order or actor.
                payload = {"cert_id": cert_id}
            else:
                negotiated = context.get("negotiated_settlement_authority")
                if isinstance(negotiated, Mapping):
                    authority_payload = negotiated.get("payload")
                    if not isinstance(authority_payload, Mapping):
                        raise PlatformContractError(
                            "accepted negotiation settlement authority is incomplete"
                        )
                    # The model has explicitly chosen to settle.  Every
                    # commercial identifier and the agreed price came from the
                    # authenticated Platform relay, never from model output.
                    payload = copy.deepcopy(dict(authority_payload))
                else:
                    supply = context.get("supply_settlement_authority")
                    if isinstance(supply, Mapping):
                        states = supply.get("states")
                        options = supply.get("settlement_options")
                        actor_id = supply.get("buyer_id")
                        sku_id = arguments.get("sku_id")
                        state_matches = (
                            [
                                row
                                for row in states
                                if isinstance(row, Mapping) and row.get("sku_id") == sku_id
                            ]
                            if isinstance(states, list)
                            else []
                        )
                        option_matches = (
                            [
                                row
                                for row in options
                                if isinstance(row, Mapping) and row.get("sku_id") == sku_id
                            ]
                            if isinstance(options, list)
                            else []
                        )
                        if (
                            len(state_matches) != 1
                            or len(option_matches) != 1
                            or not isinstance(actor_id, str)
                            or not actor_id.strip()
                        ):
                            raise SemanticDecisionError(
                                "selected supply purchase lacks unique authority"
                            )
                        option = option_matches[0]
                        authority_id = option.get("authority_id")
                        authority_digest = option.get("authority_digest")
                        if (
                            not isinstance(authority_id, str)
                            or not authority_id.strip()
                            or not isinstance(authority_digest, str)
                            or not authority_digest.strip()
                        ):
                            raise PlatformContractError(
                                "selected supply purchase authority is malformed"
                            )
                        payload = {
                            "supply_authority_id": authority_id,
                            "supply_authority_digest": authority_digest,
                            "sku_id": str(sku_id),
                            "qty": arguments["qty"],
                            "allow_partial": arguments["allow_partial"],
                        }

        if (
            binding.action_kind == "commerce.resolve_shipment"
            and binding.destination == "platform:fulfillment"
            and binding.semantic_variant is not None
        ):
            shipment = context.get("shipment_resolution_authority")
            shipment_id = shipment.get("shipment_id") if isinstance(shipment, Mapping) else None
            if not isinstance(shipment_id, str) or not shipment_id.strip():
                raise FrameworkAuthorityError(
                    "shipment resolution has no authoritative shipment id"
                )
            payload = {
                "shipment_id": shipment_id,
                "resolution": binding.semantic_variant,
            }
            if binding.semantic_variant == "replacement":
                payload["replacement_sku_id"] = arguments["replacement_sku_id"]

        if (
            binding.action_kind == "commerce.request_cart_quote"
            and binding.destination == "platform:checkout"
        ):
            if role == "buyer":
                cart_authority = context.get("cart_quote_authority")
                if isinstance(cart_authority, Mapping):
                    # SKU and quantity selection remain the model's planning
                    # decision.  The spawn-frozen mandate owns only the
                    # surrounding protocol authority.
                    authority_fields = (
                        "market_id",
                        "mandate_id",
                        "fill_policy",
                        "backorder_policy",
                    )
                    if any(name not in cart_authority for name in authority_fields):
                        raise FrameworkAuthorityError("semantic buyer cart authority is incomplete")
                    payload = {
                        **{name: copy.deepcopy(cart_authority[name]) for name in authority_fields},
                        "lines": copy.deepcopy(arguments["lines"]),
                    }
            elif role == "merchant":
                request_id = context.get("cart_quote_request_id")
                if isinstance(request_id, str) and request_id.strip():
                    # The identifier was issued by Platform in the current
                    # inbound request.  The merchant chooses to participate;
                    # it never transcribes or redirects that authority.
                    payload = {"request_id": request_id}

        if (
            binding.action_kind == "platform.checkout_cart"
            and binding.destination == "platform:checkout"
        ):
            quote_id = context.get("cart_quote_id")
            if isinstance(quote_id, str) and quote_id.strip():
                payload = {"quote_id": quote_id}
            elif route_scope is not None and isinstance(context.get("actor_id"), str):
                raise FrameworkAuthorityError(
                    "public checkout phase has no authenticated quote authority"
                )

        if not destination.startswith("@"):
            try:
                preview_actor_terminal_action(
                    Envelope(
                        msg_id="semantic-preview",
                        ts="2000-01-01T00:00:00Z",
                        from_=role,
                        to=destination,
                        in_reply_to="semantic-inbound",
                        idempotency_key="semantic-preview",
                        action={"kind": binding.action_kind, "payload": dict(payload)},
                    )
                )
            except ActorTerminalContractError as exc:
                if route_scope is not None and isinstance(context.get("actor_id"), str):
                    raise DeterministicCompilerError(
                        "framework-bound semantic action violates the actor terminal contract"
                    ) from exc
                raise SemanticDecisionError(
                    "semantic action arguments violate the actor terminal contract"
                ) from exc
        return SemanticAction(
            destination=destination,
            action_kind=binding.action_kind,
            payload=dict(payload),
            decision_rationale=str(rationale) if rationale is not None else None,
        )


def _route_is_skill_authorized(
    role: str,
    binding: _ActionBinding,
    *,
    activated_skill_names: Sequence[str] | None,
) -> bool:
    """Apply exact-route Skill ownership after action-kind filtering."""

    if activated_skill_names is None:
        return True
    required = _ROUTE_SKILLS.get((role, binding.action_kind, binding.destination))
    if required is None:
        return True
    return bool(required.intersection(map(str, activated_skill_names)))


def resolve_semantic_destination(
    action: SemanticAction,
    *,
    inbound_sender: str,
) -> str:
    """Resolve a framework-owned reply route and validate its actor class."""

    if action.destination != INBOUND_SENDER_DESTINATION:
        return action.destination
    allowed_by_kind = {
        "commerce.respond_inquiry": frozenset({"buyer", "merchant"}),
        "delegate.reject_purchase": frozenset({"consumer"}),
    }
    allowed = allowed_by_kind.get(action.action_kind)
    if allowed is None:
        raise DeterministicCompilerError("semantic reply action has no destination policy")
    if inbound_sender.split(":", 1)[0] not in allowed:
        raise FrameworkAuthorityError(
            "semantic reply action cannot target this inbound sender role"
        )
    return inbound_sender


def _actor_id_has_role(actor_id: Any, allowed_roles: frozenset[str]) -> bool:
    """Return whether a framework-owned actor id has one allowed role."""

    return (
        isinstance(actor_id, str)
        and bool(actor_id.strip())
        and actor_id.split(":", 1)[0] in allowed_roles
        and all(part.strip() for part in actor_id.split(":"))
    )


def _validate_schema_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    if not isinstance(arguments, Mapping):
        raise SemanticDecisionError("business intent arguments must be an object")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise SemanticDecisionError("business intent schema has no properties")
    required = frozenset(schema.get("required", ()))
    keys = frozenset(arguments)
    missing = required - keys
    unknown = keys - frozenset(properties)
    if missing:
        raise SemanticDecisionError("business intent arguments are missing required fields")
    if unknown and schema.get("additionalProperties") is False:
        raise SemanticDecisionError("business intent arguments contain unsupported fields")
    if len(arguments) < int(schema.get("minProperties", 0)):
        raise SemanticDecisionError("business intent arguments contain too few fields")
    for name, value in arguments.items():
        field_schema = properties.get(name)
        if isinstance(field_schema, Mapping):
            _validate_json_value(value, field_schema, field=name)


def _validate_json_value(value: Any, schema: Mapping[str, Any], *, field: str) -> None:
    if "oneOf" in schema:
        if any(_matches_json_value(value, row) for row in schema["oneOf"]):
            return
        raise SemanticDecisionError(f"business intent argument {field} has the wrong type")
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str) or (schema.get("minLength", 0) and not value.strip()):
            raise SemanticDecisionError(f"business intent argument {field} must be text")
        if "enum" in schema and value not in schema["enum"]:
            raise SemanticDecisionError(
                f"business intent argument {field} has an unsupported value"
            )
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise SemanticDecisionError(f"business intent argument {field} must be an integer")
        if "minimum" in schema and value < schema["minimum"]:
            raise SemanticDecisionError(f"business intent argument {field} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SemanticDecisionError(f"business intent argument {field} exceeds its maximum")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SemanticDecisionError(f"business intent argument {field} must be a number")
    elif expected == "boolean" and not isinstance(value, bool):
        raise SemanticDecisionError(f"business intent argument {field} must be a boolean")
    elif expected == "null" and value is not None:
        raise SemanticDecisionError(f"business intent argument {field} must be null")
    elif expected == "object":
        if not isinstance(value, Mapping):
            raise SemanticDecisionError(f"business intent argument {field} must be an object")
        if "properties" in schema:
            _validate_schema_arguments(schema, value)
    elif expected == "array":
        if not isinstance(value, (list, tuple)):
            raise SemanticDecisionError(f"business intent argument {field} must be a list")
        if len(value) < int(schema.get("minItems", 0)):
            raise SemanticDecisionError(f"business intent argument {field} has too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise SemanticDecisionError(f"business intent argument {field} has too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_json_value(item, item_schema, field=f"{field}[{index}]")
        if schema.get("uniqueItems") is True:
            seen: set[str] = set()
            for item in value:
                try:
                    identity = json.dumps(
                        item,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise SemanticDecisionError(
                        f"business intent argument {field} must contain JSON values"
                    ) from exc
                if identity in seen:
                    raise SemanticDecisionError(
                        f"business intent argument {field} must not contain duplicate items"
                    )
                seen.add(identity)


def _matches_json_value(value: Any, schema: Mapping[str, Any]) -> bool:
    try:
        _validate_json_value(value, schema, field="value")
    except SemanticDecisionError:
        return False
    return True


DEFAULT_AGENT_ROUTE_REGISTRY = AgentRouteRegistry()


def canonical_semantic_digest(value: Mapping[str, Any]) -> tuple[int, str]:
    """Return canonical JSON character length and SHA-256 without retention."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SemanticDecisionError("semantic value must be canonical JSON") from exc
    return len(rendered), hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def business_route_registry_digest() -> str:
    """Return the canonical internal route registry digest."""

    return DEFAULT_AGENT_ROUTE_REGISTRY.business_route_registry_digest()


__all__ = [
    "DEFAULT_AGENT_ROUTE_REGISTRY",
    "AgentBusinessIntentSpec",
    "INBOUND_SENDER_DESTINATION",
    "SemanticAction",
    "semantic_action_observation_class",
    "SemanticDecisionError",
    "AgentRouteRegistry",
    "WORLD_READ_TOOL_SPECS",
    "WorldReadCall",
    "canonical_semantic_digest",
    "business_route_registry_digest",
    "resolve_semantic_destination",
]
