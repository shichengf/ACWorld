"""VCP service surface for World."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from typing import Any, cast

from protocol.actions import ActionKind, is_send_allowed
from protocol.cart_quote_request import persistent_cart_quote_request_to_dict
from protocol.cart_quote_state import persistent_cart_quote_to_dict
from protocol.envelope import Envelope
from protocol.errors import PartitionViolation, SchemaError
from protocol.evidence_records import (
    EvidenceReadDenied,
    coerce_evidence_record,
    coerce_mandate_revision,
    evidence_record_to_dict,
    mandate_revision_to_dict,
)
from protocol.tool_errors import world_tool_error_payload
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventReceipt,
    protocol_event_from_json,
    protocol_event_receipt_from_json,
    protocol_event_receipt_to_dict,
    protocol_event_to_dict,
)
from protocol.matching import (
    coerce_match_acceptance,
    coerce_search_session,
    match_acceptance_to_wire,
    match_certificate_to_wire,
    search_session_to_wire,
)
from protocol.listing_claims import (
    coerce_listing_claim,
    listing_claim_to_wire,
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
    negotiation_event_to_dict,
    negotiation_thread_to_dict,
)
from protocol.pricing_policy import pricing_policy_revision_to_wire
from protocol.supply_authority import supply_purchase_authority_to_dict
from world.evidence_contracts import (
    coerce_mandate_authority,
    mandate_authority_to_wire,
)
from world.after_sales_core import after_sales_policy_to_dict
from world.after_sales_persistence import (
    after_sales_result_references,
    operation_to_wire as after_sales_operation_to_wire,
    record_key as after_sales_record_key,
    record_table as after_sales_record_table,
    record_to_wire as after_sales_record_to_wire,
)
from world.payment_fulfillment import (
    packing_record_to_dict,
    payment_state_to_dict,
)
from world.market_governance_core import (
    AdsCampaignTerms,
    GovernanceResolutionDecision,
    GovernanceResponseAttestation,
    RemediationBlueprint,
    ReputationPolicyRevision,
    ReviewAccountBinding,
)
from world.market_governance_persistence import (
    policy_payload_to_wire,
    record_payload_to_wire,
)
from world.transactions import scope_idempotency_key
from world.types import (
    AgentId,
    AllocationBatch,
    Dispute,
    DisputeId,
    DisputeState,
    Exchange,
    ExchangeId,
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
    ReputationScore,
    Ruling,
    RulingId,
    Shipment,
    ShipmentId,
    ShipmentResolution,
    ShipmentStatus,
    ShipmentStatusEvent,
    SkuId,
    SupplyState,
    TxnId,
)
from world.errors import WriteNotAuthorized


_ORDER_LIFECYCLE_ACTIONS = frozenset({
    "world.dispatch_order",
    "world.cancel_order",
    "world.mark_order_returned",
})

# These action names remain valid *internal* ``World.write(..., by_action=...)``
# audit labels, but exposing them as row-wise VCP writes would let a caller
# bypass the atomic settle/refund transaction invariants.
_DIRECT_TRANSACTION_WRITES = frozenset({
    "world.create_order",
    "world.reserve_inventory",
    "world.update_order_status",
    "world.update_ledger",
})

_WORLD_WRITE_ACTORS = {
    "world.update_catalog": "platform:catalog",
    "world.update_inventory": "platform:catalog",
    "world.update_reputation": "platform:reputation",
}


class WorldService:
    """Envelope-in/envelope-out adapter over ``World`` state."""

    def __init__(self, world: Any) -> None:
        self._world = world

    def handle(self, env: Envelope) -> Envelope | list[Envelope] | None:
        if env.to != "world":
            raise PartitionViolation(
                "World VCP accepts only the canonical 'world' endpoint"
            )
        kind = str(env.action.get("kind", ""))
        payload = env.action.get("payload", {})
        # Defense-in-depth: enforce sender role per WORLD_DESIGN §6
        # ("Permission: sender role may perform the world.* action").
        # The runtime router checks this on send too, but envelopes can
        # also arrive over the FastAPI surface directly — World is the
        # last gate. We only enforce on world.* kinds we recognize;
        # unknown kinds fall through to the "no-op" branch.
        if kind.startswith("world.") and kind != "world.response":
            self._authorize(kind, env.from_)
        if kind in {
            "world.read_payment_history",
            "world.read_ledger_history",
            "world.read_packing_history",
            "world.read_after_sales_history",
            "world.read_after_sales_policy",
        }:
            return self._response(
                env, self._read_after_sales_resource(kind, payload, env)
            )
        if kind == "world.governance_history":
            return self._response(
                env, self._governance_history(payload, env)
            )
        if kind.startswith("world.read_"):
            try:
                result = self._read(kind, payload, caller=env.from_)
            except (EvidenceReadDenied, WriteNotAuthorized) as exc:
                # A valid, read-only actor tool may be denied by the
                # authoritative row ACL.  The request itself is still real
                # audited evidence.  Return an opaque typed tool outcome so
                # the parked Agent turn can resume, just as the synchronous
                # WorldTools path reports a caught tool exception.
                if env.from_.split(":", 1)[0] not in {"buyer", "merchant"}:
                    raise
                result = world_tool_error_payload(
                    error_type=type(exc).__name__,
                    message="read not authorized",
                )
            return self._response(env, result)
        if kind == "world.resolve_search_session":
            result = self._read(kind, payload, caller=env.from_)
            return self._response(env, result)
        if kind == "world.settle_order":
            return self._response(env, self._settle(payload, env))
        if kind == "world.settle_order_partial":
            return self._response(env, self._settle_partial(payload, env))
        if kind == "world.checkout_cart":
            return self._response(env, self._checkout_cart(payload, env))
        if kind == "world.create_cart_quote_request":
            return self._response(
                env, self._create_cart_quote_request(payload, env)
            )
        if kind == "world.issue_cart_quote":
            return self._response(env, self._issue_cart_quote(payload, env))
        if kind == "world.issue_supply_purchase_authority":
            return self._response(
                env,
                self._issue_supply_purchase_authority(payload, env),
            )
        if kind == "world.apply_supply_event":
            return self._response(env, self._apply_supply_event(payload, env))
        if kind == "world.allocate_orders_atomic":
            return self._response(env, self._allocate_orders_atomic(payload, env))
        if kind == "world.record_shipment_status":
            return self._response(env, self._record_shipment_status(payload, env))
        if kind == "world.resolve_shipment":
            return self._response(env, self._resolve_shipment(payload, env))
        if kind == "world.create_search_session":
            return self._response(env, self._create_search_session(payload, env))
        if kind == "world.issue_match_certificate":
            return self._response(env, self._issue_match_certificate(payload, env))
        if kind == "world.publish_protocol_event":
            return self._response(env, self._publish_protocol_event(payload, env))
        if kind == "world.append_protocol_receipt":
            return self._response(env, self._append_protocol_receipt(payload, env))
        if kind == "world.process_protocol_event":
            return self._response(env, self._process_protocol_event(payload, env))
        if kind == "world.persist_evidence_record":
            return self._response(env, self._persist_evidence_record(payload, env))
        if kind == "world.register_mandate_authority":
            return self._response(
                env, self._register_mandate_authority(payload, env)
            )
        if kind == "world.append_mandate_revision":
            return self._response(env, self._append_mandate_revision(payload, env))
        if kind == "world.apply_listing_claim":
            return self._response(env, self._apply_listing_claim(payload, env))
        if kind == "world.apply_catalog_mutation":
            return self._response(env, self._apply_catalog_mutation(payload, env))
        if kind == "world.apply_negotiation_intent":
            return self._response(
                env, self._apply_negotiation_intent(payload, env)
            )
        if kind == "world.publish_pricing_policy":
            return self._response(
                env, self._publish_pricing_policy(payload, env)
            )
        if kind == "world.publish_after_sales_policy":
            return self._response(
                env, self._publish_after_sales_policy(payload, env)
            )
        if kind == "world.publish_governance_policy":
            return self._response(
                env, self._publish_governance_policy(payload, env)
            )
        if kind == "world.apply_governance_intent":
            return self._response(
                env, self._apply_governance_intent(payload, env)
            )
        if kind == "world.aggregate_reviews":
            return self._response(env, self._aggregate_reviews(payload, env))
        if kind == "world.ingest_review_observation":
            return self._response(
                env, self._ingest_review_observation(payload, env)
            )
        if kind == "world.ingest_market_observation":
            return self._response(
                env, self._ingest_market_observation(payload, env)
            )
        if kind == "world.resolve_governance_case":
            return self._response(
                env, self._resolve_governance_case(payload, env)
            )
        if kind == "world.apply_governance_reputation":
            return self._response(
                env, self._apply_governance_reputation(payload, env)
            )
        if kind == "world.create_remediation_plan":
            return self._response(
                env, self._create_remediation_plan(payload, env)
            )
        if kind == "world.verify_remediation_step":
            return self._response(
                env, self._verify_remediation_step(payload, env)
            )
        if kind == "world.persist_ranking_context":
            return self._response(
                env, self._persist_ranking_context(payload, env)
            )
        if kind == "world.ranking_context_projection":
            return self._response(
                env, self._ranking_context_projection(payload, env)
            )
        if kind == "world.apply_payment_intent":
            return self._response(env, self._apply_payment_intent(payload, env))
        if kind == "world.apply_packing_intent":
            return self._response(env, self._apply_packing_intent(payload, env))
        if kind == "world.apply_after_sales_intent":
            return self._response(
                env, self._apply_after_sales_intent(payload, env)
            )
        if kind == "world.complete_ledger_reconciliation":
            return self._response(
                env, self._complete_ledger_reconciliation(payload, env)
            )
        if kind == "world.refund_order":
            return self._response(env, self._refund(payload, env))
        if kind == "world.exchange_order":
            return self._response(env, self._exchange(payload, env))
        if kind in _ORDER_LIFECYCLE_ACTIONS:
            return self._response(env, self._order_lifecycle(kind, payload, env))
        if kind == "world.open_dispute":
            return self._response(env, self._open_dispute(payload, env))
        if kind == "world.rule_dispute":
            return self._response(env, self._rule_dispute(payload, env))
        if kind == "world.advance_clock":
            return self._response(env, self._advance_clock(payload, env))
        if (
            kind == "world.update_reputation"
            and payload.get("operation") == "settlement_success"
        ):
            return self._response(
                env, self._apply_settlement_reputation(payload, env)
            )
        if kind.startswith("world.") and kind != "world.response":
            result = self._write(kind, payload, env)
            return self._response(env, result)
        return None

    @staticmethod
    def _authorize(kind: str, sender_id: str) -> None:
        """Raise :class:`PartitionViolation` if ``sender_id``'s role lane
        may not emit ``kind`` to ``world``.

        Pure lookup against :data:`protocol.actions.PARTITION_ALLOW`;
        no agent registry needed. Sender role is the side prefix of
        ``sender_id`` (``"merchant"`` for ``"merchant:fulfillment"``).
        """
        try:
            action_kind = ActionKind(kind)
        except ValueError:
            return
        sender_role = sender_id.split(":", 1)[0]
        if not is_send_allowed(action_kind, sender_role, "world"):
            raise PartitionViolation(
                f"{sender_role!r} may not send {kind!r} to 'world'"
            )

    def _read(self, kind: str, payload: dict[str, Any], *, caller: str) -> Any:
        if kind == "world.read_catalog":
            if "query" in payload:
                query = str(payload.get("query", ""))
                filters = payload.get("filters")
                limit = int(payload.get("limit", 10))
                if hasattr(self._world, "search_catalog"):
                    return self._world.search_catalog(query, filters, limit=limit)
                table = self._world._tables["catalog"]
                return table.search(query, filters)[:limit]
            return self._world.read("catalog", SkuId(str(payload["sku_id"])), caller=caller)
        # Lazy lookup: only extract the key the current read needs.
        # Building these eagerly would dereference irrelevant payload keys
        # (e.g. ``merchant_id`` on an inventory read) and KeyError out.
        if kind == "world.read_inventory":
            # Raw counts: platform/runtime see all; a merchant sees ONLY its own
            # stock (never a competitor's levels). Buyers are blocked at the
            # partition layer and reach availability via world.read_availability.
            row = self._world.read("inventory", SkuId(str(payload["sku_id"])), caller=caller)
            role = caller.split(":", 1)[0]
            if role in ("platform", "runtime"):
                return row
            if row is not None and str(getattr(row, "merchant_id", "")) == caller:
                return row
            return None
        if kind == "world.read_supply_state":
            return self._world.read_supply_state(
                SkuId(str(payload["sku_id"])),
                caller=caller,
            )
        if kind == "world.read_supply_purchase_authority":
            row = self._world.read_supply_purchase_authority(
                str(payload["authority_id"]),
                caller=caller,
            )
            return None if row is None else supply_purchase_authority_to_dict(row)
        if kind == "world.read_shipment":
            return self._world.read(
                "shipments",
                ShipmentId(str(payload["shipment_id"])),
                caller=caller,
            )
        if kind == "world.read_search_session":
            session = self._world.read(
                "search_sessions", str(payload["session_id"]), caller=caller
            )
            return None if session is None else search_session_to_wire(session)
        if kind == "world.resolve_search_session":
            unique_only = payload.get("unique_only", True)
            current_only = payload.get("current_only", True)
            if not isinstance(unique_only, bool) or not isinstance(
                current_only, bool
            ):
                raise SchemaError(
                    "search-session resolution flags must be booleans"
                )
            session = self._world.resolve_search_session(
                buyer_id=str(payload["buyer_id"]),
                offer_id=str(payload["offer_id"]),
                caller=caller,
                unique_only=unique_only,
                current_only=current_only,
            )
            return None if session is None else search_session_to_wire(session)
        if kind == "world.read_match_acceptance":
            acceptance = self._world.read(
                "match_acceptances",
                _match_acceptance_key(
                    str(payload["buyer_id"]), str(payload["idempotency_key"])
                ),
                caller=caller,
            )
            return (
                None
                if acceptance is None
                else match_acceptance_to_wire(acceptance)
            )
        if kind == "world.read_match_certificate":
            if "cert_id" in payload:
                certificate = self._world.read(
                    "match_certificates", str(payload["cert_id"]), caller=caller
                )
            else:
                current_only = payload.get("current_only", True)
                if not isinstance(current_only, bool):
                    raise SchemaError(
                        "match-certificate resolution flag must be boolean"
                    )
                certificate = self._world.resolve_match_certificate(
                    buyer_id=str(payload["buyer_id"]),
                    order_id=str(payload["order_id"]),
                    caller=caller,
                    current_only=current_only,
                )
            return (
                None
                if certificate is None
                else match_certificate_to_wire(certificate)
            )
        if kind == "world.read_availability":
            # Availability-only view: returns ONLY the in-stock bool, never the
            # exact qty_available/qty_reserved counts (those are commercially
            # sensitive and the synchronous WorldTools.is_in_stock hides them).
            # Routed through the same WorldTools method so the bool is single-
            # sourced and re-entrant == synchronous.
            from world.tools import WorldTools

            tools = WorldTools(self._world, caller_id=caller)
            sku = SkuId(str(payload["sku_id"]))
            return {"sku_id": str(sku),
                    "in_stock": tools.is_in_stock(sku, qty=int(payload.get("qty", 1)))}
        if kind == "world.read_reputation":
            return self._world.read("reputation", AgentId(str(payload["merchant_id"])), caller=caller)
        if kind == "world.read_policy":
            if set(payload) != {"policy_kind", "merchant_id"}:
                raise SchemaError(
                    "world.read_policy requires exactly policy_kind and merchant_id"
                )
            if payload["policy_kind"] != "after_sales":
                raise SchemaError("unsupported public policy_kind")
            merchant_id = str(payload["merchant_id"])
            if not merchant_id:
                raise SchemaError("merchant_id must be non-empty")
            policy = self._world.after_sales_policy(
                merchant_id, caller="platform:after-sales"
            )
            return None if policy is None else _public_after_sales_policy(policy)
        if kind == "world.read_after_sales_authority":
            if set(payload) != {"order_id"}:
                raise SchemaError(
                    "after-sales authority read requires exactly order_id"
                )
            order_id = payload["order_id"]
            if not isinstance(order_id, str) or not order_id.strip():
                raise SchemaError(
                    "after-sales authority order_id must be non-empty"
                )
            from world.after_sales_authority_projection import (
                after_sales_authority_projection_to_wire,
            )
            from world.tools import WorldTools

            projection = WorldTools(
                self._world,
                caller_id=caller,
            ).get_after_sales_authority(order_id)
            return after_sales_authority_projection_to_wire(projection)
        if kind == "world.read_protocol_event_authority":
            if set(payload) != {"event_id"}:
                raise SchemaError(
                    "protocol event authority read requires exactly event_id"
                )
            event_id = payload["event_id"]
            if not isinstance(event_id, str) or not event_id.strip():
                raise SchemaError(
                    "protocol event authority event_id must be non-empty"
                )
            from world.tools import WorldTools

            return WorldTools(
                self._world,
                caller_id=caller,
            ).get_protocol_event_authority(event_id)
        if kind == "world.read_order":
            return self._world.read("orders", OrderId(str(payload["order_id"])), caller=caller)
        if kind == "world.read_order_protocol_state":
            order_id = OrderId(str(payload["order_id"]))
            order = self._world.read("orders", order_id, caller=caller)
            if order is None:
                return None
            revision = self._world.read_order_state_revision(
                order_id,
                caller=caller,
            )
            reference_digest = self._world.read_order_operation_reference(
                order_id,
                caller=caller,
            )
            if revision is None or reference_digest is None:
                raise SchemaError("visible order has no authoritative protocol state")
            return {
                "order_id": str(order.order_id),
                "buyer_id": str(order.buyer_id),
                "merchant_id": str(order.merchant_id),
                "state": order.state.value,
                "state_revision": revision,
                "operation_reference_digest": reference_digest,
                "logical_time": int(self._world.logical_time),
            }
        if kind == "world.read_protocol_event":
            event = self._world.read(
                "protocol_events",
                str(payload["event_id"]),
                caller=caller,
            )
            return None if event is None else protocol_event_to_dict(event)
        if kind == "world.read_protocol_events":
            events = self._world.protocol_events_for_stream(
                str(payload["binding_digest"]),
                caller=caller,
            )
            return [protocol_event_to_dict(event) for event in events]
        if kind == "world.read_protocol_receipts":
            receipts = self._world.protocol_receipts_for_stream(
                str(payload["binding_digest"]),
                order_id=OrderId(str(payload["order_id"])),
                caller=caller,
            )
            return [
                protocol_event_receipt_to_dict(receipt) for receipt in receipts
            ]
        if kind == "world.read_negotiation_event":
            if set(payload) != {"event_id"}:
                raise SchemaError(
                    "negotiation event read requires exactly event_id"
                )
            event = self._world.read_negotiation_event(
                str(payload["event_id"]), caller=caller
            )
            return None if event is None else negotiation_event_to_dict(event)
        if kind == "world.read_negotiation_thread":
            if set(payload) != {"negotiation_id"}:
                raise SchemaError(
                    "negotiation thread read requires exactly negotiation_id"
                )
            thread = self._world.read_negotiation_thread(
                str(payload["negotiation_id"]), caller=caller
            )
            return None if thread is None else negotiation_thread_to_dict(thread)
        if kind == "world.read_negotiation_events":
            if set(payload) != {"negotiation_id"}:
                raise SchemaError(
                    "negotiation stream read requires exactly negotiation_id"
                )
            return [
                negotiation_event_to_dict(event)
                for event in self._world.negotiation_events_for_thread(
                    str(payload["negotiation_id"]), caller=caller
                )
            ]
        if kind == "world.read_pricing_policies":
            expected = {"market_id", "merchant_id", "policy_id"}
            if set(payload) != expected:
                raise SchemaError(
                    "pricing policy read requires exactly market_id, "
                    "merchant_id, and policy_id"
                )
            return [
                pricing_policy_revision_to_wire(revision)
                for revision in self._world.pricing_policy_revisions(
                    str(payload["market_id"]),
                    str(payload["merchant_id"]),
                    str(payload["policy_id"]),
                    caller=caller,
                )
            ]
        if kind == "world.read_cart_quote_request":
            if set(payload) != {"request_id"}:
                raise SchemaError(
                    "cart quote request read requires exactly request_id"
                )
            request = self._world.read(
                "persistent_cart_quote_requests",
                str(payload["request_id"]),
                caller=caller,
            )
            return (
                None
                if request is None
                else persistent_cart_quote_request_to_dict(request)
            )
        if kind == "world.read_cart_quote":
            if set(payload) != {"quote_id"}:
                raise SchemaError("cart quote read requires exactly quote_id")
            quote = self._world.read(
                "persistent_cart_quotes",
                str(payload["quote_id"]),
                caller=caller,
            )
            return None if quote is None else persistent_cart_quote_to_dict(quote)
        if kind == "world.read_evidence_record":
            allowed = {"record_id", "version", "record_digest"}
            if not set(payload).issubset(allowed) or "record_id" not in payload:
                raise SchemaError(
                    "evidence read requires record_id and at most version or digest"
                )
            if "version" in payload and "record_digest" in payload:
                raise SchemaError("evidence read cannot combine version and digest")
            record_id = payload["record_id"]
            if not isinstance(record_id, str) or not record_id.strip():
                raise SchemaError("evidence record_id must be a non-empty string")
            version = payload.get("version")
            if version is not None and (
                isinstance(version, bool)
                or not isinstance(version, int)
                or version <= 0
            ):
                raise SchemaError("evidence version must be a positive integer")
            record_digest = payload.get("record_digest")
            if record_digest is not None and (
                not isinstance(record_digest, str) or not record_digest.strip()
            ):
                raise SchemaError("evidence record_digest must be a non-empty string")
            record = self._world.read_evidence_record(
                record_id,
                caller=caller,
                version=version,
                record_digest=record_digest,
            )
            return None if record is None else evidence_record_to_dict(record)
        if kind == "world.read_mandate_revisions":
            if set(payload) != {"mandate_id"}:
                raise SchemaError("mandate read requires exactly mandate_id")
            mandate_id = payload["mandate_id"]
            if not isinstance(mandate_id, str) or not mandate_id.strip():
                raise SchemaError("mandate_id must be a non-empty string")
            return [
                mandate_revision_to_dict(revision)
                for revision in self._world.mandate_revisions(
                    mandate_id, caller=caller
                )
            ]
        if kind == "world.read_listing_claim":
            if set(payload) != {"claim_id"}:
                raise SchemaError("listing claim read requires exactly claim_id")
            claim_id = payload["claim_id"]
            if not isinstance(claim_id, str) or not claim_id.strip():
                raise SchemaError("claim_id must be a non-empty string")
            claim = self._world.read_listing_claim(
                claim_id, caller=caller
            )
            return None if claim is None else listing_claim_to_wire(claim)
        if kind == "world.read_listing_claims":
            if set(payload) != {"listing_id"}:
                raise SchemaError("listing claims read requires exactly listing_id")
            listing_id = payload["listing_id"]
            if not isinstance(listing_id, str) or not listing_id.strip():
                raise SchemaError("listing_id must be a non-empty string")
            return [
                listing_claim_to_wire(claim)
                for claim in self._world.listing_claims_for_listing(
                    listing_id, caller=caller
                )
            ]
        if kind == "world.read_ledger":
            return self._world.read("ledger", TxnId(str(payload["txn_id"])), caller=caller)
        if kind == "world.read_fulfillment":
            return self._world.read(
                "fulfillments", OrderId(str(payload["order_id"])), caller=caller
            )
        if kind == "world.read_dispute":
            return self._world.read(
                "disputes", DisputeId(str(payload["dispute_id"])), caller=caller
            )
        if kind == "world.read_ruling":
            return self._read_ruling(payload, caller=caller)
        if kind == "world.read_clock":
            return {"logical_time": int(self._world.logical_time)}
        if kind == "world.read_order_timeline":
            return self._world.read(
                "order_timelines",
                OrderId(str(payload["order_id"])),
                caller=caller,
            )
        if kind == "world.read_order_group":
            return self._world.read(
                "order_groups",
                OrderGroupId(str(payload["order_group_id"])),
                caller=caller,
            )
        if kind in {
            "world.read_friends",
            "world.read_friend_reviews",
            "world.read_review_evidence",
        }:
            # Reuse the caller-scoped WorldTools facade so the VCP path and the
            # in-process path share one social-read implementation.
            from world.tools import WorldTools

            tools = WorldTools(self._world, caller_id=caller)
            if kind == "world.read_friends":
                return list(tools.get_friends())
            if kind == "world.read_review_evidence":
                # Call the facade instead of rebuilding its response here.  The
                # second copy of this shape drifted from the first and reported
                # a blank ``sku_id`` for an unfiltered read, which the Agent
                # rejects as a malformed business reference and which aborts the
                # episode as an environment defect.
                return tools.get_review_evidence(
                    sku_id=payload.get("sku_id"),
                    merchant_id=payload.get("merchant_id"),
                )
            return tools.get_friend_reviews(
                sku_id=payload.get("sku_id"), merchant_id=payload.get("merchant_id")
            )
        return None

    def _read_after_sales_resource(
        self,
        kind: str,
        payload: dict[str, Any],
        env: Envelope,
    ) -> dict[str, Any]:
        """Read an owner-scoped lifecycle projection through Platform.

        The service address is authenticated by the envelope.  World uses the
        preserved ``original_actor`` for row visibility, so Platform cannot
        accidentally widen a buyer or merchant history read.
        """

        self._require_service_actor(env, "platform:after-sales")
        if kind == "world.read_after_sales_policy":
            if set(payload) != {"merchant_id", "original_actor"}:
                raise SchemaError(
                    "after-sales policy read requires exactly merchant_id and "
                    "original_actor"
                )
            original_actor = _original_actor(payload)
            _require_commerce_actor(original_actor)
            merchant_id = str(payload["merchant_id"])
            if not merchant_id:
                raise SchemaError("merchant_id must be non-empty")
            policy = self._world.after_sales_policy(
                merchant_id, caller=env.from_
            )
            return {
                "resource": "policy",
                "records": (
                    [] if policy is None else [_public_after_sales_policy(policy)]
                ),
            }

        if set(payload) != {"order_id", "original_actor"}:
            raise SchemaError(
                "after-sales history read requires exactly order_id and "
                "original_actor"
            )
        actor = _original_actor(payload)
        _require_after_sales_principal(actor)
        order_id = str(payload["order_id"])
        if not order_id:
            raise SchemaError("order_id must be non-empty")
        if kind == "world.read_payment_history":
            return {
                "resource": "payment_history",
                "records": [
                    payment_state_to_dict(row)
                    for row in self._world.payment_history(order_id, caller=actor)
                ],
            }
        if kind == "world.read_ledger_history":
            return {
                "resource": "ledger_history",
                "records": [
                    _receipt_to_wire(row)
                    for row in self._world.ledger_history(order_id, caller=actor)
                ],
            }
        if kind == "world.read_packing_history":
            return {
                "resource": "packing_history",
                "records": [
                    packing_record_to_dict(row)
                    for row in self._world.packing_history(order_id, caller=actor)
                ],
            }
        if kind == "world.read_after_sales_history":
            records: list[dict[str, Any]] = []
            for row in self._world.after_sales_history(order_id, caller=actor):
                table = after_sales_record_table(row)
                records.append(
                    {
                        "table": table,
                        "key": after_sales_record_key(table, row),
                        "value": after_sales_record_to_wire(row),
                    }
                )
            return {"resource": "after_sales_history", "records": records}
        raise SchemaError(f"unsupported after-sales read action {kind!r}")

    def _publish_governance_policy(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        intent, original_actor = _governance_intent_request(
            payload, action="world.publish_governance_policy"
        )
        result = self._world.publish_governance_policy(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _apply_governance_intent(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        intent, original_actor = _governance_intent_request(
            payload, action="world.apply_governance_intent"
        )
        result = self._world.apply_governance_intent(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _aggregate_reviews(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"sku_id", "original_actor"}),
            action="world.aggregate_reviews",
        )
        result = self._world.aggregate_reviews(
            _governance_text(request, "sku_id"),
            by_actor=env.from_,
            original_actor=_governance_original_actor(request),
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _ingest_market_observation(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"record_id", "original_actor"}),
            action="world.ingest_market_observation",
        )
        result = self._world.ingest_market_observation(
            _governance_text(request, "record_id"),
            by_actor=env.from_,
            original_actor=_governance_original_actor(request),
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _ingest_review_observation(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"record_id", "original_actor"}),
            action="world.ingest_review_observation",
        )
        result = self._world.ingest_review_observation(
            _governance_text(request, "record_id"),
            by_actor=env.from_,
            original_actor=_governance_original_actor(request),
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _resolve_governance_case(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        intent, original_actor = _governance_intent_request(
            payload, action="world.resolve_governance_case"
        )
        result = self._world.resolve_governance_case(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _apply_governance_reputation(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        intent, original_actor = _governance_intent_request(
            payload, action="world.apply_governance_reputation"
        )
        result = self._world.apply_governance_reputation(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _create_remediation_plan(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        intent, original_actor = _governance_intent_request(
            payload, action="world.create_remediation_plan"
        )
        result = self._world.create_remediation_plan(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _verify_remediation_step(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"plan_id", "step_id", "original_actor"}),
            action="world.verify_remediation_step",
        )
        result = self._world.verify_remediation_step(
            _governance_text(request, "plan_id"),
            _governance_text(request, "step_id"),
            by_actor=env.from_,
            original_actor=_governance_original_actor(request),
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _persist_ranking_context(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"ranking_result", "original_actor"}),
            action="world.persist_ranking_context",
        )
        ranking_result = request["ranking_result"]
        if not isinstance(ranking_result, Mapping):
            raise SchemaError(
                "world.persist_ranking_context.ranking_result must be an object"
            )
        result = self._world.persist_ranking_context(
            dict(ranking_result),
            by_actor=env.from_,
            original_actor=_governance_original_actor(request),
            idempotency_key=_governance_idempotency_key(env),
        )
        return _governance_result_to_wire(result)

    def _ranking_context_projection(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"context_id"}),
            action="world.ranking_context_projection",
        )
        return self._world.ranking_context_projection(
            _governance_text(request, "context_id"),
            caller=env.from_,
        )

    def _governance_history(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:governance")
        request = _strict_governance_payload(
            payload,
            fields=frozenset({"record_kind", "stable_id", "original_actor"}),
            action="world.governance_history",
        )
        record_kind = _governance_text(request, "record_kind")
        stable_id = _governance_text(request, "stable_id")
        actor = _governance_original_actor(request)
        rows = self._world.governance_history(
            record_kind, stable_id, caller=actor
        )
        return {
            "resource": "governance_history",
            "record_kind": record_kind,
            "stable_id": stable_id,
            "records": [
                _governance_payload_to_wire(record_kind, row) for row in rows
            ],
        }

    def _read_ruling(self, payload: dict[str, Any], *, caller: str) -> Ruling | None:
        """Read a ruling only when its underlying dispute is caller-visible."""
        ruling: Ruling | None = None
        if payload.get("ruling_id") not in (None, ""):
            ruling = self._world.read(
                "rulings", RulingId(str(payload["ruling_id"])), caller=None
            )
        elif payload.get("dispute_id") not in (None, ""):
            dispute_id = str(payload["dispute_id"])
            rows = (
                self._world.iter_table("rulings")
                if hasattr(self._world, "iter_table")
                else self._world._tables["rulings"].all()
            )
            for _, candidate in rows:
                if str(candidate.dispute_id) == dispute_id:
                    ruling = candidate
                    break
        else:
            raise SchemaError("world.read_ruling requires ruling_id or dispute_id")
        if ruling is None:
            return None
        dispute = self._world.read(
            "disputes", ruling.dispute_id, caller=caller
        )
        return ruling if dispute is not None else None

    def _publish_after_sales_policy(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:policy")
        intent, original_actor = _authority_intent_payload(
            payload, action="world.publish_after_sales_policy"
        )
        _require_actor_role(original_actor, "merchant")
        policy = self._world.publish_after_sales_policy(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )
        return after_sales_policy_to_dict(policy)

    def _apply_payment_intent(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:psp")
        intent, original_actor = _authority_intent_payload(
            payload, action="world.apply_payment_intent"
        )
        _require_actor_role(original_actor, "buyer")
        operation = intent.get("op")
        if operation == "authorize":
            row = self._world.authorize_payment(
                intent,
                by_actor=env.from_,
                original_actor=original_actor,
                idempotency_key=env.idempotency_key,
            )
        elif operation == "capture":
            row = self._world.capture_payment(
                intent,
                by_actor=env.from_,
                original_actor=original_actor,
                idempotency_key=env.idempotency_key,
            )
        else:
            raise SchemaError(
                "world.apply_payment_intent op must be authorize or capture"
            )
        return payment_state_to_dict(row)

    def _apply_packing_intent(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:fulfillment")
        intent, original_actor = _authority_intent_payload(
            payload, action="world.apply_packing_intent"
        )
        _require_actor_role(original_actor, "merchant")
        row = self._world.apply_packing_intent(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )
        return packing_record_to_dict(row)

    def _apply_after_sales_intent(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:after-sales")
        intent, original_actor = _authority_intent_payload(
            payload, action="world.apply_after_sales_intent"
        )
        _require_after_sales_principal(original_actor)
        result = self._world.apply_after_sales_intent(
            intent,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )
        result_record = self._world.after_sales_result_record(
            result.operation, caller=original_actor
        )
        if result_record is None:
            raise PartitionViolation(
                "after-sales result row is not visible to the authenticated principal"
            )
        return {
            "disposition": result.disposition,
            "operation": after_sales_operation_to_wire(result.operation),
            "references": after_sales_result_references(
                result_record, result.operation
            ),
        }

    def _complete_ledger_reconciliation(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:accounting")
        if set(payload) != {"order_id", "request_id", "original_actor"}:
            raise SchemaError(
                "world.complete_ledger_reconciliation requires exactly "
                "order_id, request_id, and original_actor"
            )
        original_actor = _original_actor(payload)
        if original_actor != env.from_:
            raise PartitionViolation(
                "ledger reconciliation principal must be platform:accounting"
            )
        order_id = str(payload["order_id"])
        request_id = str(payload["request_id"])
        if not order_id or not request_id:
            raise SchemaError("reconciliation ids must be non-empty")
        result = self._world.complete_ledger_reconciliation(
            order_id,
            request_id,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )
        return {
            "disposition": result.disposition,
            "operation": after_sales_operation_to_wire(result.operation),
        }

    def _write(
        self,
        kind: str,
        payload: dict[str, Any],
        env: Envelope,
    ) -> dict[str, Any]:
        if kind in _DIRECT_TRANSACTION_WRITES:
            raise PartitionViolation(
                f"{kind!r} is an internal transaction label, not a VCP write; "
                "use world.settle_order, world.refund_order, or a lifecycle action"
            )
        expected_actor = _WORLD_WRITE_ACTORS.get(kind)
        if expected_actor is not None:
            self._require_service_actor(env, expected_actor)
        table, key, value = _world_write_args(kind, payload)
        self._world.write(table, key, value, by_action=kind)
        return {"table": table, "key": str(key), "status": "ok"}

    def _publish_protocol_event(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:events")
        if set(payload) != {"event"}:
            raise SchemaError("protocol event publish requires exactly event")
        event = _coerce_protocol_event(payload["event"])
        persisted = self._world.publish_protocol_event(
            event,
            by_actor=env.from_,
        )
        return protocol_event_to_dict(persisted)

    def _append_protocol_receipt(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:events")
        if set(payload) != {"receipt", "original_actor"}:
            raise SchemaError(
                "protocol receipt append requires exactly receipt and original_actor"
            )
        receipt = _coerce_protocol_event_receipt(payload["receipt"])
        persisted = self._world.append_protocol_receipt(
            receipt,
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
        )
        return protocol_event_receipt_to_dict(persisted)

    def _process_protocol_event(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        """Execute one registered event operation through authoritative World."""

        self._require_service_actor(env, "platform:events")
        expected = {"event_id", "original_actor", "reason"}
        if set(payload) != expected:
            raise SchemaError(
                "protocol event processing requires exactly event_id, "
                "original_actor, and reason"
            )
        for field in expected:
            value = payload[field]
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(
                    f"protocol event processing {field} must be non-empty"
                )
        persisted = self._world.process_protocol_event(
            event_id=payload["event_id"],
            by_actor=env.from_,
            original_actor=payload["original_actor"],
            reason=payload["reason"],
            idempotency_key=env.idempotency_key,
        )
        return protocol_event_receipt_to_dict(persisted)

    def _persist_evidence_record(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:evidence")
        if set(payload) != {"record", "original_actor"}:
            raise SchemaError(
                "evidence persist requires exactly record and original_actor"
            )
        record = coerce_evidence_record(payload["record"])
        persisted = self._world.persist_evidence_record(
            record,
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            idempotency_key=env.idempotency_key,
        )
        return evidence_record_to_dict(persisted)

    def _register_mandate_authority(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, object]:
        self._require_service_actor(env, "platform:mandate")
        if set(payload) != {"authority", "original_actor"}:
            raise SchemaError(
                "mandate authority registration requires authority and original_actor"
            )
        authority = coerce_mandate_authority(payload["authority"])
        persisted = self._world.register_mandate_authority(
            authority,
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            idempotency_key=env.idempotency_key,
        )
        return mandate_authority_to_wire(persisted)

    def _append_mandate_revision(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:mandate")
        if set(payload) != {"revision", "original_actor"}:
            raise SchemaError(
                "mandate revision append requires revision and original_actor"
            )
        revision = coerce_mandate_revision(payload["revision"])
        persisted = self._world.append_mandate_revision(
            revision,
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            idempotency_key=env.idempotency_key,
        )
        return mandate_revision_to_dict(persisted)

    def _apply_listing_claim(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:claims")
        if set(payload) != {"claim", "original_actor"}:
            raise SchemaError(
                "listing claim apply requires claim and original_actor"
            )
        claim = coerce_listing_claim(payload["claim"])
        persisted = self._world.apply_listing_claim(
            claim,
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            idempotency_key=env.idempotency_key,
        )
        return listing_claim_to_wire(persisted)

    def _apply_catalog_mutation(
        self, payload: dict[str, Any], env: Envelope
    ) -> Listing:
        self._require_service_actor(env, "platform:catalog")
        if set(payload) != {"intent", "original_actor"}:
            raise SchemaError(
                "catalog mutation requires exactly intent and original_actor"
            )
        original_actor = payload["original_actor"]
        if not isinstance(original_actor, str) or not original_actor:
            raise SchemaError("catalog mutation original_actor must be non-empty")
        return cast(Listing, self._world.apply_catalog_mutation(
            payload["intent"],
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        ))

    def _apply_negotiation_intent(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:negotiation")
        expected = {
            "intent",
            "original_actor",
            "max_rounds",
            "deadline_ticks",
        }
        if set(payload) != expected:
            raise SchemaError(
                "negotiation apply requires exactly intent, original_actor, "
                "max_rounds, and deadline_ticks"
            )
        original_actor = payload["original_actor"]
        if not isinstance(original_actor, str) or not original_actor.strip():
            raise SchemaError("negotiation original_actor must be non-empty")
        event = self._world.apply_negotiation_intent(
            payload["intent"],
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
            max_rounds=payload["max_rounds"],
            deadline_ticks=payload["deadline_ticks"],
        )
        return negotiation_event_to_dict(event)

    def _publish_pricing_policy(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:pricing")
        if set(payload) != {"intent", "original_actor"}:
            raise SchemaError(
                "pricing policy publish requires exactly intent and original_actor"
            )
        original_actor = payload["original_actor"]
        if not isinstance(original_actor, str) or not original_actor.strip():
            raise SchemaError("pricing policy original_actor must be non-empty")
        revision = self._world.publish_pricing_policy(
            payload["intent"],
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )
        return pricing_policy_revision_to_wire(revision)

    def _apply_settlement_reputation(
        self, payload: dict[str, Any], env: Envelope
    ) -> ReputationScore:
        """Route a PSP-backed reputation event through authoritative World."""

        self._require_service_actor(env, "platform:reputation")
        expected = {
            "operation",
            "merchant_id",
            "order_id",
            "txn_id",
            "original_actor",
            "source_request_id",
        }
        if set(payload) != expected:
            raise SchemaError(
                "settlement reputation requires exactly operation, merchant_id, "
                "order_id, txn_id, original_actor, and source_request_id"
            )
        if payload["operation"] != "settlement_success":
            raise SchemaError("unsupported reputation operation")
        return self._world.apply_settlement_reputation(
            merchant_id=AgentId(str(payload["merchant_id"])),
            order_id=OrderId(str(payload["order_id"])),
            txn_id=TxnId(str(payload["txn_id"])),
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            source_request_id=str(payload["source_request_id"]),
            idempotency_key=env.idempotency_key,
        )

    def _settle(self, payload: dict[str, Any], env: Envelope) -> dict[str, Any]:
        """Route ``world.settle_order`` to the atomic + idempotent World
        transaction (the multi-row commit ``_world_write_args`` can't express,
        keyed by the envelope's idempotency_key). Returns the settle summary."""
        self._require_service_actor(env, "platform:psp")
        order = _coerce_order(payload)
        receipt = _coerce_settle_receipt(payload, order, env)
        scoped_key = scope_idempotency_key(str(order.buyer_id), env.idempotency_key)
        settled = self._world.settle_order(
            order=order,
            receipt=receipt,
            by_role=env.from_.split(":", 1)[0],
            idempotency_key=scoped_key,
        )
        return {
            "order_id": str(settled.order_id),
            "txn_id": str(settled.txn_id),
            "status": "settled",
        }

    def _create_cart_quote_request(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:checkout")
        expected = {
            "intent",
            "market_id",
            "original_actor",
            "request_ttl_ticks",
        }
        if set(payload) != expected:
            raise SchemaError(
                "cart quote request creation requires exactly intent, market_id, "
                "original_actor, and request_ttl_ticks"
            )
        request = self._world.create_cart_quote_request(
            payload["intent"],
            market_id=str(payload["market_id"]),
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            idempotency_key=env.idempotency_key,
            request_ttl_ticks=payload["request_ttl_ticks"],
        )
        return persistent_cart_quote_request_to_dict(request)

    def _issue_cart_quote(
        self, payload: dict[str, Any], env: Envelope
    ) -> dict[str, Any]:
        self._require_service_actor(env, "platform:checkout")
        common = {"original_actor", "quote_ttl_ticks"}
        if set(payload) == common | {"request_id"}:
            quote = self._world.issue_cart_quote_from_request(
                str(payload["request_id"]),
                by_actor=env.from_,
                original_actor=str(payload["original_actor"]),
                idempotency_key=env.idempotency_key,
                quote_ttl_ticks=payload["quote_ttl_ticks"],
            )
        elif set(payload) == common | {"intent", "market_id"}:
            quote = self._world.issue_cart_quote(
                payload["intent"],
                market_id=str(payload["market_id"]),
                by_actor=env.from_,
                original_actor=str(payload["original_actor"]),
                idempotency_key=env.idempotency_key,
                quote_ttl_ticks=payload["quote_ttl_ticks"],
            )
        else:
            raise SchemaError(
                "cart quote issuance requires original_actor and quote_ttl_ticks "
                "plus either request_id or intent with market_id"
            )
        return persistent_cart_quote_to_dict(quote)

    def _issue_supply_purchase_authority(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> list[dict[str, Any]]:
        self._require_service_actor(env, "platform:supply")
        if set(payload) != {"sku_ids", "original_actor", "ttl_ticks"}:
            raise SchemaError(
                "supply authority issuance requires sku_ids, original_actor, "
                "and ttl_ticks"
            )
        raw_skus = payload["sku_ids"]
        if not isinstance(raw_skus, list):
            raise SchemaError("supply authority sku_ids must be an array")
        rows = self._world.issue_supply_purchase_authorities(
            tuple(raw_skus),
            by_actor=env.from_,
            original_actor=str(payload["original_actor"]),
            idempotency_key=env.idempotency_key,
            ttl_ticks=payload["ttl_ticks"],
        )
        return [supply_purchase_authority_to_dict(row) for row in rows]

    def _checkout_cart(self, payload: dict[str, Any], env: Envelope) -> OrderGroup:
        self._require_service_actor(env, "platform:checkout")
        if set(payload) != {"quote_id", "original_actor"}:
            raise SchemaError(
                "cart checkout requires exactly quote_id and original_actor"
            )
        original_actor = str(payload["original_actor"])
        return self._world.checkout_cart(
            quote_id=str(payload["quote_id"]),
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=env.idempotency_key,
        )

    def _apply_supply_event(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> SupplyState:
        self._require_service_actor(env, "platform:supply")
        original_actor = _original_actor(payload)
        qty_delta = payload.get("qty_delta", 0)
        eta_day = payload.get("eta_day")
        unit_price_cents = payload.get("unit_price_cents")
        expected_version = payload.get("expected_version")
        for name, value in (
            ("qty_delta", qty_delta),
            ("eta_day", eta_day),
            ("unit_price_cents", unit_price_cents),
            ("expected_version", expected_version),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise SchemaError(f"world.apply_supply_event.{name} must be an integer")
        return self._world.apply_supply_event(
            sku_id=SkuId(str(payload.get("sku_id", ""))),
            qty_delta=int(qty_delta),
            eta_day=eta_day,
            unit_price_cents=unit_price_cents,
            expected_version=expected_version,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=scope_idempotency_key(
                original_actor, env.idempotency_key
            ),
        )

    def _allocate_orders_atomic(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> AllocationBatch:
        self._require_service_actor(env, "platform:fulfillment")
        original_actor = _original_actor(payload)
        allocation_id = str(payload.get("allocation_id", ""))
        sku_id = str(payload.get("sku_id", ""))
        raw_order_ids = payload.get("priority_order_ids")
        if not allocation_id or not sku_id:
            raise SchemaError(
                "world.allocate_orders_atomic requires allocation_id and sku_id"
            )
        if not isinstance(raw_order_ids, list) or not raw_order_ids:
            raise SchemaError(
                "world.allocate_orders_atomic.priority_order_ids must be a non-empty list"
            )
        order_ids = tuple(OrderId(str(value)) for value in raw_order_ids)
        if any(not str(value) for value in order_ids) or len(set(order_ids)) != len(
            order_ids
        ):
            raise SchemaError(
                "world.allocate_orders_atomic.priority_order_ids must be unique and non-empty"
            )
        return self._world.allocate_orders_atomic(
            allocation_id=allocation_id,
            merchant_id=AgentId(original_actor),
            sku_id=SkuId(sku_id),
            priority_order_ids=order_ids,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=scope_idempotency_key(
                original_actor, env.idempotency_key
            ),
        )

    def _record_shipment_status(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> Shipment:
        self._require_service_actor(env, "platform:fulfillment")
        original_actor = _original_actor(payload)
        if original_actor != "runtime:logistics":
            raise PartitionViolation(
                "shipment status original_actor must be runtime:logistics"
            )
        try:
            status = ShipmentStatus(str(payload.get("status", "")))
        except ValueError as exc:
            raise SchemaError("invalid shipment status") from exc
        return self._world.record_shipment_status(
            shipment_id=ShipmentId(str(payload.get("shipment_id", ""))),
            event_id=str(payload.get("event_id", "")),
            status=status,
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=scope_idempotency_key(
                original_actor, env.idempotency_key
            ),
        )

    def _resolve_shipment(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> Shipment:
        self._require_service_actor(env, "platform:fulfillment")
        original_actor = _original_actor(payload)
        try:
            resolution = ShipmentResolution(str(payload.get("resolution", "")))
        except ValueError as exc:
            raise SchemaError("invalid shipment resolution") from exc
        raw_replacement = payload.get("replacement_sku_id")
        return self._world.resolve_shipment(
            shipment_id=ShipmentId(str(payload.get("shipment_id", ""))),
            resolution=resolution,
            replacement_sku_id=(
                None
                if raw_replacement in (None, "")
                else SkuId(str(raw_replacement))
            ),
            by_actor=env.from_,
            original_actor=original_actor,
            idempotency_key=scope_idempotency_key(
                original_actor, env.idempotency_key
            ),
        )

    def _refund(self, payload: dict[str, Any], env: Envelope) -> dict[str, Any]:
        """Route ``world.refund_order`` to the atomic + idempotent World refund
        (REFUNDED + restock + reversing ledger, all under ``World._lock`` inside
        this process). The platform only emits the envelope; the transaction —
        and its allowlist/idempotency — live entirely here. Returns the refund
        summary."""
        self._require_service_actor(env, "platform:psp")
        order = _coerce_order(payload)
        refund_receipt = _coerce_refund_receipt(payload, order, env)
        scoped_key = scope_idempotency_key(str(order.buyer_id), env.idempotency_key)
        refunded = self._world.refund_order(
            order=order,
            refund_receipt=refund_receipt,
            by_role=env.from_.split(":", 1)[0],
            idempotency_key=scoped_key,
        )
        return {
            "order_id": str(refunded.order_id),
            "txn_id": str(refunded.txn_id),
            "status": "refunded",
        }

    def _exchange(self, payload: dict[str, Any], env: Envelope) -> Exchange:
        """Route the platform's actor-preserving exchange command atomically."""
        self._require_service_actor(env, "platform:psp")
        replacement_raw = payload.get("replacement_order")
        replacement = (
            replacement_raw
            if isinstance(replacement_raw, Order)
            else _coerce_order({"order": replacement_raw})
        )
        # The buyer identity was checked by the platform before this internal
        # service call.  World authorizes the exact PSP service actor and scopes
        # idempotency to the replacement's authoritative buyer identity.
        scoped_key = scope_idempotency_key(
            str(replacement.buyer_id), env.idempotency_key
        )
        return self._world.exchange_order(
            exchange_id=ExchangeId(str(payload.get("exchange_id", ""))),
            original_order_id=OrderId(str(payload.get("original_order_id", ""))),
            replacement_order=replacement,
            by_actor=env.from_,
            idempotency_key=scoped_key,
        )

    def _settle_partial(
        self,
        payload: dict[str, Any],
        env: Envelope,
    ) -> dict[str, Any]:
        """Commit one authoritative full/partial/backorder allocation.

        Unlike the legacy all-or-nothing path, this service preserves the
        originating buyer separately and requires the exact PSP service actor.
        The fulfillment quantity is still validated atomically by World.
        """
        self._require_service_actor(env, "platform:psp")
        order = _coerce_order(payload)
        original_actor = _original_actor(payload)
        if original_actor != str(order.buyer_id):
            raise PartitionViolation(
                "partial settlement original_actor must own the order"
            )
        fulfilled_qty = payload.get("fulfilled_qty")
        if isinstance(fulfilled_qty, bool) or not isinstance(fulfilled_qty, int):
            raise SchemaError("world.settle_order_partial.fulfilled_qty must be an integer")
        receipt = (
            None
            if fulfilled_qty == 0
            else _coerce_partial_receipt(payload, order, env, fulfilled_qty)
        )
        # Preserve retry compatibility for an allocation persisted before
        # actor-scoped service keys were introduced. New allocations always use
        # the scoped key; an exact legacy same-order/raw-key record replays in
        # its original namespace.
        scoped_key = scope_idempotency_key(original_actor, env.idempotency_key)
        prior = self._world.read(
            "fulfillments",
            order.order_id,
            caller="platform:psp",
        )
        effective_key = (
            env.idempotency_key
            if prior is not None and prior.idempotency_key == env.idempotency_key
            else scoped_key
        )
        if receipt is not None:
            receipt = replace(receipt, idempotency_key=effective_key)
        allocation = self._world.settle_order_partial(
            order=order,
            fulfilled_qty=fulfilled_qty,
            receipt=receipt,
            by_actor=env.from_,
            idempotency_key=effective_key,
        )
        return {
            "order_id": str(allocation.order_id),
            "txn_id": (
                str(allocation.receipt_txn_id)
                if allocation.receipt_txn_id is not None
                else None
            ),
            "status": (
                "settled"
                if allocation.backordered_qty == 0
                else "backordered"
                if allocation.fulfilled_qty == 0
                else "partially_settled"
            ),
            "requested_qty": allocation.requested_qty,
            "fulfilled_qty": allocation.fulfilled_qty,
            "backordered_qty": allocation.backordered_qty,
        }

    def _order_lifecycle(
        self,
        kind: str,
        payload: dict[str, Any],
        env: Envelope,
    ) -> Order:
        """Commit an order transition while preserving the originating actor.

        World only accepts these commands from the exact PSP service address.
        The buyer/merchant identity is carried separately as ``original_actor``
        and is checked by the authoritative World primitive against the stored
        order; it is never inferred from a caller-supplied buyer/merchant field.
        """
        self._require_service_actor(env, "platform:psp")
        order_id = _required_id(payload, "order_id", OrderId)
        actor = _original_actor(payload)
        if kind == "world.dispatch_order":
            return self._world.dispatch_order(order_id=order_id, by_actor=actor)
        if kind == "world.cancel_order":
            return self._world.cancel_order(order_id=order_id, by_actor=actor)
        if kind == "world.mark_order_returned":
            return self._world.mark_order_returned(order_id=order_id, by_actor=actor)
        raise ValueError(f"unsupported lifecycle action: {kind}")

    def _open_dispute(self, payload: dict[str, Any], env: Envelope) -> Dispute:
        self._require_service_actor(env, "platform:adjudicator")
        dispute = _coerce_dispute(payload)
        return self._world.open_dispute(
            dispute=dispute,
            by_actor=_original_actor(payload),
        )

    def _rule_dispute(self, payload: dict[str, Any], env: Envelope) -> Ruling:
        self._require_service_actor(env, "platform:adjudicator")
        ruling = _coerce_ruling(payload)
        return self._world.rule_dispute(
            ruling=ruling,
            by_actor=_original_actor(payload),
        )

    def _advance_clock(self, payload: dict[str, Any], env: Envelope) -> dict[str, int]:
        if env.from_ != "runtime:clock":
            raise PartitionViolation("only runtime:clock may advance World logical time")
        if set(payload) != {"to_tick"}:
            raise SchemaError("world.advance_clock accepts exactly the to_tick field")
        value = payload["to_tick"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError("world.advance_clock.to_tick must be an integer")
        current = self._world.advance_logical_time(
            to_tick=value,
            by_actor=env.from_,
        )
        return {"logical_time": current}

    def _create_search_session(
        self, payload: dict[str, Any], env: Envelope
    ) -> Any:
        self._require_service_actor(env, "platform:aggregator")
        if set(payload) != {"session", "original_actor"}:
            raise SchemaError(
                "world.create_search_session requires exactly session and original_actor"
            )
        session = coerce_search_session(payload["session"])
        original_actor = str(payload["original_actor"])
        if original_actor != session.buyer_id:
            raise PartitionViolation("search session actor does not match buyer")
        persisted = self._world.create_search_session(
            session=session,
            by_actor=env.from_,
            idempotency_key=env.idempotency_key,
        )
        return search_session_to_wire(persisted)

    def _issue_match_certificate(
        self, payload: dict[str, Any], env: Envelope
    ) -> Any:
        self._require_service_actor(env, "platform:aggregator")
        if set(payload) != {"acceptance", "original_actor"}:
            raise SchemaError(
                "world.issue_match_certificate requires exactly acceptance and original_actor"
            )
        acceptance = coerce_match_acceptance(payload["acceptance"])
        original_actor = str(payload["original_actor"])
        if original_actor != acceptance.buyer_id:
            raise PartitionViolation("match acceptance actor does not match buyer")
        if env.idempotency_key != acceptance.idempotency_key:
            raise SchemaError(
                "match acceptance and envelope idempotency keys must match"
            )
        certificate = self._world.issue_match_certificate(
            acceptance=acceptance,
            by_actor=env.from_,
            original_actor=original_actor,
        )
        return match_certificate_to_wire(certificate)

    @staticmethod
    def _require_service_actor(env: Envelope, expected: str) -> None:
        if env.from_ != expected:
            raise PartitionViolation(
                f"{env.from_!r} may not execute an operation reserved for "
                f"{expected!r}"
            )

    @staticmethod
    def _response(env: Envelope, payload: Any) -> Envelope:
        return Envelope(
            msg_id=f"{env.msg_id}:world",
            ts=env.ts,
            from_="world",
            to=env.from_,
            in_reply_to=env.msg_id,
            idempotency_key=env.idempotency_key,
            action={"kind": "world.response", "payload": payload},
        )


def _world_write_args(kind: str, payload: dict[str, Any]) -> tuple[str, Any, Any]:
    if kind == "world.update_catalog":
        listing = _coerce_listing(payload)
        return "catalog", listing.sku_id, listing
    if kind in {"world.reserve_inventory", "world.update_inventory"}:
        inventory = _coerce_inventory(payload)
        return "inventory", inventory.sku_id, inventory
    if kind in {"world.create_order", "world.update_order_status"}:
        order = _coerce_order(payload)
        return "orders", order.order_id, order
    if kind == "world.update_ledger":
        receipt = _coerce_receipt(payload)
        return "ledger", receipt.txn_id, receipt
    if kind == "world.update_reputation":
        score = _coerce_reputation(payload)
        return "reputation", score.merchant_id, score
    raise ValueError(f"unsupported world action: {kind}")


def _match_acceptance_key(buyer_id: str, key: str) -> str:
    return f"{buyer_id}\x1f{key}"


def _coerce_protocol_event(value: Any) -> ProtocolEvent:
    if isinstance(value, ProtocolEvent):
        return value
    if not isinstance(value, dict):
        raise SchemaError("protocol event must be an object")
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return protocol_event_from_json(payload)
    except (TypeError, ValueError, SchemaError) as exc:
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"invalid protocol event payload: {exc}") from exc


def _coerce_protocol_event_receipt(value: Any) -> ProtocolEventReceipt:
    if isinstance(value, ProtocolEventReceipt):
        return value
    if not isinstance(value, dict):
        raise SchemaError("protocol event receipt must be an object")
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return protocol_event_receipt_from_json(payload)
    except (TypeError, ValueError, SchemaError) as exc:
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"invalid protocol event receipt payload: {exc}") from exc


def _coerce_money(value: Any) -> Money:
    if isinstance(value, Money):
        return value
    if isinstance(value, dict):
        return Money(amount=Decimal(str(value["amount"])), currency=value.get("currency", "USD"))
    return Money(amount=Decimal(str(value)))


def _coerce_listing(payload: dict[str, Any]) -> Listing:
    row = payload.get("listing", payload)
    if isinstance(row, Listing):
        return row
    return Listing(
        sku_id=SkuId(str(row["sku_id"])),
        category=str(row["category"]),
        name=str(row["name"]),
        attributes=dict(row.get("attributes", {})),
        list_price=_coerce_money(row["list_price"]),
        merchant_id=AgentId(str(row["merchant_id"])),
        product_id=(str(row["product_id"]) if row.get("product_id") is not None else None),
    )


def _coerce_inventory(payload: dict[str, Any]) -> InventoryRow:
    row = payload.get("inventory", payload)
    if isinstance(row, InventoryRow):
        return row
    return InventoryRow(
        sku_id=SkuId(str(row["sku_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        qty_available=int(row["qty_available"]),
        qty_reserved=int(row.get("qty_reserved", 0)),
        eta_day=int(row.get("eta_day", 0)),
        version=int(row.get("version", 1)),
    )


def _coerce_supply_state(payload: dict[str, Any] | SupplyState) -> SupplyState:
    if isinstance(payload, SupplyState):
        return payload
    row = payload.get("supply_state", payload)
    return SupplyState(
        sku_id=SkuId(str(row["sku_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        available_qty=int(row["available_qty"]),
        reserved_qty=int(row["reserved_qty"]),
        eta_day=int(row["eta_day"]),
        unit_price_cents=int(row["unit_price_cents"]),
        version=int(row["version"]),
    )


def _coerce_fulfillment(payload: dict[str, Any]) -> FulfillmentAllocation:
    row = payload.get("fulfillment", payload)
    if isinstance(row, FulfillmentAllocation):
        return row
    return FulfillmentAllocation(
        order_id=OrderId(str(row["order_id"])),
        buyer_id=AgentId(str(row["buyer_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        sku_id=SkuId(str(row["sku_id"])),
        requested_qty=int(row["requested_qty"]),
        fulfilled_qty=int(row["fulfilled_qty"]),
        backordered_qty=int(row["backordered_qty"]),
        receipt_txn_id=(
            TxnId(str(row["receipt_txn_id"]))
            if row.get("receipt_txn_id") is not None
            else None
        ),
        created_by=AgentId(str(row["created_by"])),
        idempotency_key=str(row["idempotency_key"]),
    )


def _coerce_allocation_batch(
    payload: dict[str, Any] | AllocationBatch,
) -> AllocationBatch:
    if isinstance(payload, AllocationBatch):
        return payload
    row = payload.get("allocation_batch", payload)
    if not isinstance(row, dict):
        raise SchemaError("allocation batch payload must be an object")
    raw_allocations = row.get("allocations")
    raw_priority = row.get("priority_order_ids")
    if not isinstance(raw_allocations, (list, tuple)) or not isinstance(
        raw_priority, (list, tuple)
    ):
        raise SchemaError("allocation batch requires allocations and priority_order_ids")
    return AllocationBatch(
        allocation_id=str(row["allocation_id"]),
        merchant_id=AgentId(str(row["merchant_id"])),
        sku_id=SkuId(str(row["sku_id"])),
        priority_order_ids=tuple(OrderId(str(value)) for value in raw_priority),
        allocations=tuple(
            _coerce_fulfillment(cast("dict[str, Any]", value))
            if not isinstance(value, FulfillmentAllocation)
            else value
            for value in raw_allocations
        ),
        created_by=AgentId(str(row["created_by"])),
        idempotency_key=str(row["idempotency_key"]),
    )


def _coerce_shipment(payload: dict[str, Any] | Shipment) -> Shipment:
    if isinstance(payload, Shipment):
        return payload
    row = payload.get("shipment", payload)
    if not isinstance(row, dict):
        raise SchemaError("shipment payload must be an object")
    history = row.get("status_history")
    if not isinstance(history, (list, tuple)):
        raise SchemaError("shipment status_history must be a sequence")
    try:
        return Shipment(
            shipment_id=ShipmentId(str(row["shipment_id"])),
            order_id=OrderId(str(row["order_id"])),
            buyer_id=AgentId(str(row["buyer_id"])),
            merchant_id=AgentId(str(row["merchant_id"])),
            original_sku_id=SkuId(str(row["original_sku_id"])),
            status=ShipmentStatus(str(row["status"])),
            status_history=tuple(
                value
                if isinstance(value, ShipmentStatusEvent)
                else ShipmentStatusEvent(
                    event_id=str(value["event_id"]),
                    status=ShipmentStatus(str(value["status"])),
                    logical_time=int(value["logical_time"]),
                )
                for value in history
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
            version=int(row.get("version", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid shipment payload: {exc}") from exc


def _coerce_exchange(payload: dict[str, Any]) -> Exchange:
    """Normalize an exchange row returned over the HTTP/VCP transport."""
    value = payload.get("exchange", payload)
    if not isinstance(value, dict):
        raise SchemaError("exchange payload must be an object")
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


def _coerce_order(payload: dict[str, Any]) -> Order:
    row = payload.get("order", payload)
    if isinstance(row, Order):
        return row
    return Order(
        order_id=OrderId(str(row["order_id"])),
        buyer_id=AgentId(str(row["buyer_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        sku_id=SkuId(str(row["sku_id"])),
        qty=int(row["qty"]),
        agreed_price=_coerce_money(row["agreed_price"]),
        state=OrderState(str(row.get("state", OrderState.ACCEPTED.value))),
        request_order=int(row.get("request_order", 0)),
    )


def _coerce_order_timeline(payload: dict[str, Any]) -> OrderTimeline:
    row = payload.get("order_timeline", payload)
    if isinstance(row, OrderTimeline):
        return row
    if not isinstance(row, dict):
        raise SchemaError("order timeline must be an object")
    try:
        return OrderTimeline(
            order_id=OrderId(str(row["order_id"])),
            buyer_id=AgentId(str(row["buyer_id"])),
            merchant_id=AgentId(str(row["merchant_id"])),
            settled_at_tick=_optional_tick(row.get("settled_at_tick")),
            dispatched_at_tick=_optional_tick(row.get("dispatched_at_tick")),
            return_window_ticks=_optional_tick(row.get("return_window_ticks")),
            return_authorized_at_tick=_optional_tick(
                row.get("return_authorized_at_tick")
            ),
            returned_at_tick=_optional_tick(row.get("returned_at_tick")),
            refunded_at_tick=_optional_tick(row.get("refunded_at_tick")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid order timeline payload: {exc}") from exc


def _optional_tick(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("logical ticks must be non-negative integers")
    return value


def _coerce_receipt(payload: dict[str, Any]) -> Receipt:
    row = payload.get("receipt", payload)
    if isinstance(row, Receipt):
        return row
    return Receipt(
        txn_id=TxnId(str(row["txn_id"])),
        ts=str(row["ts"]),
        order_id=OrderId(str(row["order_id"])),
        buyer_id=AgentId(str(row["buyer_id"])),
        merchant_id=AgentId(str(row["merchant_id"])),
        sku_id=SkuId(str(row["sku_id"])),
        qty=int(row["qty"]),
        price=_coerce_money(row["price"]),
        idempotency_key=str(row["idempotency_key"]),
        effect=str(row.get("effect", "charge")),
    )


def _coerce_order_group(payload: dict[str, Any]) -> OrderGroup:
    """Normalize an order-group row returned over HTTP/VCP."""

    # See ``WorldService._checkout_cart`` for why this dependency is lazy.
    from protocol.cart import coerce_order_group

    return coerce_order_group(payload)


def _coerce_settle_receipt(payload: dict[str, Any], order: Order, env: Envelope) -> Receipt:
    """Receipt for a settle: an explicit ``receipt`` in the payload wins;
    otherwise synthesize from the order + envelope (ts + idempotency_key)."""
    row = payload.get("receipt")
    if isinstance(row, Receipt):
        return row
    if isinstance(row, dict):
        return _coerce_receipt(row)
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
    order: Order,
    env: Envelope,
    fulfilled_qty: int,
) -> Receipt:
    """Build the receipt for only the units actually allocated."""
    row = payload.get("receipt")
    if isinstance(row, Receipt):
        return row
    if isinstance(row, dict):
        return _coerce_receipt(row)
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


def _coerce_refund_receipt(payload: dict[str, Any], order: Order, env: Envelope) -> Receipt:
    """Reversing receipt for a refund: an explicit ``refund_receipt`` in the
    payload wins; otherwise synthesize from the order + envelope. The ``refund:``
    txn_id prefix marks it as the reversal (distinct from the settle receipt)."""
    row = payload.get("refund_receipt")
    if isinstance(row, Receipt):
        return replace(row, effect="refund")
    if isinstance(row, dict):
        return replace(_coerce_receipt(row), effect="refund")
    return Receipt(
        txn_id=TxnId(str(payload.get("txn_id", f"refund:{order.order_id}"))),
        ts=str(payload.get("ts", env.ts)),
        order_id=order.order_id,
        buyer_id=order.buyer_id,
        merchant_id=order.merchant_id,
        sku_id=order.sku_id,
        qty=order.qty,
        price=order.agreed_price,
        idempotency_key=env.idempotency_key,
        effect="refund",
    )


def _coerce_reputation(payload: dict[str, Any]) -> ReputationScore:
    row = payload.get("score", payload)
    if isinstance(row, ReputationScore):
        return row
    return ReputationScore(
        merchant_id=AgentId(str(row["merchant_id"])),
        rolling_avg=float(row["rolling_avg"]),
        n_settled=int(row["n_settled"]),
        n_disputed=int(row["n_disputed"]),
    )


def _coerce_dispute(payload: dict[str, Any]) -> Dispute:
    row = payload.get("dispute", payload)
    if isinstance(row, Dispute):
        return row
    if not isinstance(row, dict):
        raise SchemaError("dispute must be an object")
    try:
        return Dispute(
            dispute_id=DisputeId(str(row["dispute_id"])),
            order_id=OrderId(str(row["order_id"])),
            filed_by=AgentId(str(row["filed_by"])),
            against=AgentId(str(row["against"])),
            reason=str(row["reason"]),
            state=DisputeState(str(row.get("state", DisputeState.OPEN.value))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid dispute payload: {exc}") from exc


def _coerce_ruling(payload: dict[str, Any]) -> Ruling:
    row = payload.get("ruling", payload)
    if isinstance(row, Ruling):
        return row
    if not isinstance(row, dict):
        raise SchemaError("ruling must be an object")
    try:
        refund = row.get("refund_amount")
        return Ruling(
            ruling_id=RulingId(str(row["ruling_id"])),
            dispute_id=DisputeId(str(row["dispute_id"])),
            in_favor_of=AgentId(str(row["in_favor_of"])),
            rationale=str(row["rationale"]),
            refund_amount=None if refund is None else _coerce_money(refund),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid ruling payload: {exc}") from exc


def _authority_intent_payload(
    payload: dict[str, Any], *, action: str
) -> tuple[dict[str, Any], str]:
    if set(payload) != {"intent", "original_actor"}:
        raise SchemaError(
            f"{action} requires exactly intent and original_actor"
        )
    intent = payload["intent"]
    if not isinstance(intent, Mapping):
        raise SchemaError(f"{action}.intent must be an object")
    return dict(intent), _original_actor(payload)


_GOVERNANCE_POLICY_KINDS = frozenset(
    {
        "ads_campaign_terms",
        "review_account_binding",
        "reputation_policy_revision",
        "remediation_blueprint",
    }
)
_GOVERNANCE_RECORD_KINDS = frozenset(
    {
        "campaign",
        "review_evidence",
        "review_aggregate",
        "market_signal",
        "governance_case",
        "governance_response_attestation",
        "governance_resolution_decision",
        "reputation_event",
        "remediation_plan",
        "ranking_context",
    }
)


def _strict_governance_payload(
    payload: Any,
    *,
    fields: frozenset[str],
    action: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SchemaError(f"{action} payload must be an object")
    actual = frozenset(payload.keys())
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(repr(value) for value in actual - fields)
        raise SchemaError(
            f"{action} requires exact fields; missing={missing!r}, "
            f"unknown={unknown!r}"
        )
    return dict(payload)


def _governance_intent_request(
    payload: Any, *, action: str
) -> tuple[dict[str, Any], str]:
    request = _strict_governance_payload(
        payload,
        fields=frozenset({"intent", "original_actor"}),
        action=action,
    )
    intent = request["intent"]
    if not isinstance(intent, Mapping):
        raise SchemaError(f"{action}.intent must be an object")
    return dict(intent), _governance_original_actor(request)


def _governance_original_actor(payload: Mapping[str, Any]) -> str:
    value = payload.get("original_actor")
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("governance request requires non-empty original_actor")
    return value


def _governance_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"governance request requires non-empty {field}")
    return value


def _governance_idempotency_key(env: Envelope) -> str:
    value = env.idempotency_key
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("governance mutation requires an idempotency key")
    return value


def _governance_result_to_wire(value: Any) -> dict[str, Any]:
    if isinstance(value, AdsCampaignTerms):
        kind = "ads_campaign_terms"
        payload = policy_payload_to_wire(value)
    elif isinstance(value, ReviewAccountBinding):
        kind = "review_account_binding"
        payload = policy_payload_to_wire(value)
    elif isinstance(value, ReputationPolicyRevision):
        kind = "reputation_policy_revision"
        payload = policy_payload_to_wire(value)
    elif isinstance(value, RemediationBlueprint):
        kind = "remediation_blueprint"
        payload = policy_payload_to_wire(value)
    elif isinstance(value, Campaign):
        kind = "campaign"
        payload = record_payload_to_wire(value)
    elif isinstance(value, ReviewEvidence):
        kind = "review_evidence"
        payload = record_payload_to_wire(value)
    elif isinstance(value, ReviewAggregate):
        kind = "review_aggregate"
        payload = record_payload_to_wire(value)
    elif isinstance(value, MarketSignal):
        kind = "market_signal"
        payload = record_payload_to_wire(value)
    elif isinstance(value, GovernanceCase):
        kind = "governance_case"
        payload = record_payload_to_wire(value)
    elif isinstance(value, GovernanceResponseAttestation):
        kind = "governance_response_attestation"
        payload = record_payload_to_wire(value)
    elif isinstance(value, GovernanceResolutionDecision):
        kind = "governance_resolution_decision"
        payload = record_payload_to_wire(value)
    elif isinstance(value, ReputationEvent):
        kind = "reputation_event"
        payload = record_payload_to_wire(value)
    elif isinstance(value, RemediationPlan):
        kind = "remediation_plan"
        payload = record_payload_to_wire(value)
    elif isinstance(value, RankingContext):
        kind = "ranking_context"
        payload = record_payload_to_wire(value)
    else:
        raise SchemaError(
            f"World returned unsupported governance result {type(value).__name__}"
        )
    return {"result_kind": kind, "value": payload}


def _governance_payload_to_wire(record_kind: str, value: Any) -> dict[str, Any]:
    if record_kind in _GOVERNANCE_POLICY_KINDS:
        return policy_payload_to_wire(value)
    if record_kind in _GOVERNANCE_RECORD_KINDS:
        return record_payload_to_wire(value)
    raise SchemaError(f"unsupported governance record kind {record_kind!r}")


def _require_actor_role(actor: str, expected_role: str) -> None:
    if actor.split(":", 1)[0] != expected_role:
        raise PartitionViolation(
            f"{actor!r} is not an authenticated {expected_role} actor"
        )


def _require_commerce_actor(actor: str) -> None:
    if actor.split(":", 1)[0] not in {"buyer", "merchant"}:
        raise PartitionViolation(
            f"{actor!r} is not an authenticated buyer or merchant"
        )


def _require_after_sales_principal(actor: str) -> None:
    if actor == "platform:adjudicator":
        return
    _require_commerce_actor(actor)


def _public_after_sales_policy(policy: Any) -> dict[str, Any]:
    """Remove internal service allowlists from a published policy row."""

    row = after_sales_policy_to_dict(policy)
    public_fields = (
        "schema_id",
        "policy_id",
        "merchant_id",
        "revision",
        "effective_tick",
        "return_window_ticks",
        "max_refund_bps",
        "split_refund_bps",
        "owner_paid_cancel_allowed",
        "merchant_paid_cancel_allowed",
        "allowed_return_conditions",
        "policy_digest",
    )
    return {field: row[field] for field in public_fields}


def _receipt_to_wire(receipt: Receipt) -> dict[str, Any]:
    """Canonical public receipt projection with authoritative effect."""

    if receipt.effect not in {"charge", "refund"}:
        raise SchemaError("World ledger receipt has an invalid effect")
    return {
        "txn_id": str(receipt.txn_id),
        "ts": receipt.ts,
        "order_id": str(receipt.order_id),
        "buyer_id": str(receipt.buyer_id),
        "merchant_id": str(receipt.merchant_id),
        "sku_id": str(receipt.sku_id),
        "qty": receipt.qty,
        "price": {
            "amount": str(receipt.price.amount),
            "currency": receipt.price.currency,
        },
        "idempotency_key": receipt.idempotency_key,
        "effect": receipt.effect,
    }


def _original_actor(payload: dict[str, Any]) -> str:
    actor = payload.get("original_actor")
    if not isinstance(actor, str) or not actor.strip():
        raise SchemaError("lifecycle write requires a non-empty original_actor")
    return actor


def _required_id(payload: dict[str, Any], key: str, coerce: Any) -> Any:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaError(f"lifecycle write requires a non-empty {key}")
    return coerce(value)
