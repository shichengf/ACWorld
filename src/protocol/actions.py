"""The action enum and the role-partition allow-table.

The action kinds come from the DESIGN.md §4 protocol surface and the VCP_SPEC
namespaces. Action set is additive: new actions append; existing ones are
never renamed or removed.
"""

from __future__ import annotations

from enum import Enum


class ActionKind(str, Enum):
    """All action kinds the protocol enumerates.

    Wire strings for buyer/merchant outbound actions are namespaced
    (``commerce.*`` / ``platform.*`` / ``world.*``). Bare-name members
    (e.g. ``SETTLE``, ``UPDATE_REPUTATION``) are internal aliases used
    by service-side dispatch tables; they may share an action semantically
    with a namespaced sibling but carry a distinct wire string.
    """

    # Discovery (commerce namespace; wire form used by buyer/merchant)
    SEARCH = "commerce.search"
    GET_SKU = "commerce.get_sku"
    # Negotiation (commerce namespace; wire form used by buyer/merchant)
    PROPOSE_OFFER = "commerce.propose_offer"
    COUNTER_OFFER = "commerce.counter_offer"
    ACCEPT_OFFER = "commerce.accept_offer"
    REJECT_OFFER = "commerce.reject_offer"
    WITHDRAW_OFFER = "commerce.withdraw_offer"
    # Mandate / cart (bare; no commerce-namespace mirror yet)
    CREATE_ORDER = "create_order"
    AUTHORIZE_PAYMENT = "authorize_payment"
    COMMERCE_CREATE_CART_QUOTE_REQUEST = "commerce.create_cart_quote_request"
    COMMERCE_REQUEST_CART_QUOTE = "commerce.request_cart_quote"
    COMMERCE_READ_SUPPLY_STATE = "commerce.read_supply_state"
    COMMERCE_UPDATE_SUPPLY = "commerce.update_supply"
    COMMERCE_ALLOCATE_FULFILLMENT = "commerce.allocate_fulfillment"
    COMMERCE_READ_SHIPMENT = "commerce.read_shipment"
    COMMERCE_RESOLVE_SHIPMENT = "commerce.resolve_shipment"
    # Persistent authority event decisions.  The actor supplies only its
    # decision intent; Platform derives event identity, observed World state,
    # logical time, and any permitted effect references.
    COMMERCE_ACKNOWLEDGE_PROTOCOL_EVENT = "commerce.acknowledge_protocol_event"
    COMMERCE_REJECT_PROTOCOL_EVENT = "commerce.reject_protocol_event"
    COMMERCE_PROCESS_PROTOCOL_EVENT = "commerce.process_protocol_event"
    COMMERCE_PUBLISH_EVIDENCE_RECORD = "commerce.publish_evidence_record"
    # Authenticated external services use a Platform namespaced action.  This
    # keeps runtime supplied observations distinct from commerce-agent intent
    # while preserving the same Platform to World evidence boundary.
    PLATFORM_PUBLISH_EVIDENCE_RECORD = "platform.publish_evidence_record"
    COMMERCE_APPLY_LISTING_CLAIM = "commerce.apply_listing_claim"
    # Settlement (bare; SETTLE is internal-only, never on the wire)
    SETTLE = "settle"
    # Authoritative after-sales intents.  Actors can send these only to
    # Platform.  Platform preserves the authenticated full actor id and calls
    # the matching compact World API; actors never write lifecycle state.
    COMMERCE_CANCEL_PAID_ORDER = "commerce.cancel_paid_order"
    REQUEST_RETURN = "commerce.request_return"
    COMMERCE_AUTHORIZE_RETURN = "commerce.authorize_return"
    COMMERCE_DENY_RETURN = "commerce.deny_return"
    COMMERCE_RECEIVE_RETURN = "commerce.receive_return"
    COMMERCE_OPEN_REFUND_CASE = "commerce.open_refund_case"
    COMMERCE_APPROVE_REFUND = "commerce.approve_refund"
    COMMERCE_DENY_REFUND = "commerce.deny_refund"
    COMMERCE_REQUEST_EXCHANGE = "commerce.request_exchange"
    COMMERCE_AUTHORIZE_EXCHANGE = "commerce.authorize_exchange"
    COMMERCE_DENY_EXCHANGE = "commerce.deny_exchange"
    COMMERCE_COMPLETE_EXCHANGE = "commerce.complete_exchange"
    COMMERCE_OPEN_DISPUTE = "commerce.open_dispute"
    COMMERCE_SUBMIT_DISPUTE_EVIDENCE = "commerce.submit_dispute_evidence"
    COMMERCE_RESPOND_TO_DISPUTE = "commerce.respond_to_dispute"
    COMMERCE_REQUEST_LEDGER_RECONCILIATION = (
        "commerce.request_ledger_reconciliation"
    )
    COMMERCE_READ_PAYMENT_HISTORY = "commerce.read_payment_history"
    COMMERCE_READ_LEDGER_HISTORY = "commerce.read_ledger_history"
    COMMERCE_READ_PACKING_HISTORY = "commerce.read_packing_history"
    COMMERCE_READ_AFTER_SALES_HISTORY = "commerce.read_after_sales_history"
    COMMERCE_READ_AFTER_SALES_POLICY = "commerce.read_after_sales_policy"
    # Marketplace-governance actor intents.  These are compact requests to a
    # stateless Platform mediator.  Campaign state, verified-purchase facts,
    # case outcomes, remediation progress, and ranking evidence are derived
    # and persisted only by World.
    COMMERCE_PUBLISH_CAMPAIGN = "commerce.publish_campaign"
    COMMERCE_DISCLOSE_PLACEMENT = "commerce.disclose_placement"
    COMMERCE_ACTIVATE_CAMPAIGN = "commerce.activate_campaign"
    COMMERCE_SUBMIT_REVIEW = "commerce.submit_review"
    COMMERCE_REJECT_REVIEW_MANIPULATION = (
        "commerce.reject_review_manipulation"
    )
    COMMERCE_REJECT_COORDINATION = "commerce.reject_coordination"
    COMMERCE_ACCEPT_REMEDIATION_PLAN = (
        "commerce.accept_remediation_plan"
    )
    COMMERCE_COMPLETE_REMEDIATION_STEP = (
        "commerce.complete_remediation_step"
    )
    COMMERCE_READ_GOVERNANCE_HISTORY = "commerce.read_governance_history"
    # Platform
    UPDATE_REPUTATION = "update_reputation"
    OPEN_DISPUTE = "open_dispute"
    RULE = "rule"
    PLATFORM_SETTLE_PAYMENT = "platform.settle_payment"
    PLATFORM_UPDATE_REPUTATION = "platform.update_reputation"
    PLATFORM_SETTLEMENT_RECEIPT = "platform.settlement_receipt"
    # Additive response for the partial-settlement path.  It is deliberately
    # distinct from a payment receipt because a zero-fill allocation creates no
    # ledger entry.
    PLATFORM_FULFILLMENT_ALLOCATION = "platform.fulfillment_allocation"
    PLATFORM_REPUTATION_UPDATED = "platform.reputation_updated"
    PLATFORM_RANK_OFFERS = "platform.rank_offers"
    PLATFORM_CREATE_MATCH_CERTIFICATE = "platform.create_match_certificate"
    # Authoritative lifecycle / adjudication replies and requests.  The older
    # bare ``open_dispute`` / ``rule`` members remain wire-compatible aliases;
    # new scenarios use the namespaced forms documented by VCP_SPEC.
    PLATFORM_OPEN_DISPUTE = "platform.open_dispute"
    PLATFORM_RULE_DISPUTE = "platform.rule_dispute"
    PLATFORM_LIFECYCLE_UPDATED = "platform.lifecycle_updated"
    PLATFORM_AFTER_SALES_UPDATED = "platform.after_sales_updated"
    PLATFORM_AFTER_SALES_SNAPSHOT = "platform.after_sales_snapshot"
    # Trusted governance control-plane requests and safe Platform replies.
    # The Platform owns no governance state; each request is forwarded through
    # the matching world.* VCP method.
    PLATFORM_PUBLISH_GOVERNANCE_POLICY = (
        "platform.publish_governance_policy"
    )
    PLATFORM_AGGREGATE_REVIEWS = "platform.aggregate_reviews"
    PLATFORM_INGEST_REVIEW_OBSERVATION = (
        "platform.ingest_review_observation"
    )
    PLATFORM_INGEST_MARKET_OBSERVATION = (
        "platform.ingest_market_observation"
    )
    PLATFORM_RESOLVE_GOVERNANCE_CASE = (
        "platform.resolve_governance_case"
    )
    PLATFORM_APPLY_GOVERNANCE_REPUTATION = (
        "platform.apply_governance_reputation"
    )
    PLATFORM_CREATE_REMEDIATION_PLAN = (
        "platform.create_remediation_plan"
    )
    PLATFORM_VERIFY_REMEDIATION_STEP = (
        "platform.verify_remediation_step"
    )
    PLATFORM_PERSIST_RANKING_CONTEXT = (
        "platform.persist_ranking_context"
    )
    PLATFORM_GOVERNANCE_UPDATED = "platform.governance_updated"
    PLATFORM_GOVERNANCE_SNAPSHOT = "platform.governance_snapshot"
    # World-derived actor notifications.  These carry only safe references and
    # next-action affordances.  They never contain scenario-authored case or
    # remediation identities.
    PLATFORM_GOVERNANCE_CASE_NOTICE = "platform.governance_case_notice"
    PLATFORM_REMEDIATION_PLAN_NOTICE = "platform.remediation_plan_notice"
    PLATFORM_REMEDIATION_AUDIT_REQUEST = (
        "platform.remediation_audit_request"
    )
    # CatalogPolicy ack for forwarded commerce.* catalog intents
    # (adjust_price / update_listing / receive_shipment). Distinct from
    # PLATFORM_SETTLEMENT_RECEIPT, which is the PSP's settle confirmation;
    # the two were conflated in the initial merchant-branch merge.
    PLATFORM_CATALOG_ACK = "platform.catalog_ack"
    # Read-only catalog response.  ``commerce.get_sku`` is mediated by
    # Platform so the requesting actor, World read, and returned listing form
    # one decision-journaled exchange without creating a dummy World write.
    PLATFORM_CATALOG_LISTING = "platform.catalog_listing"
    PLATFORM_CART_QUOTE_REQUEST = "platform.cart_quote_request"
    PLATFORM_CART_QUOTE = "platform.cart_quote"
    PLATFORM_CHECKOUT_CART = "platform.checkout_cart"
    PLATFORM_CART_SETTLEMENT = "platform.cart_settlement"
    PLATFORM_APPLY_SUPPLY_EVENT = "platform.apply_supply_event"
    PLATFORM_SUPPLY_STATE = "platform.supply_state"
    PLATFORM_SUPPLY_EVENT_APPLIED = "platform.supply_event_applied"
    PLATFORM_ALLOCATION_BATCH = "platform.allocation_batch"
    PLATFORM_RECORD_SHIPMENT_STATUS = "platform.record_shipment_status"
    PLATFORM_SHIPMENT_STATE = "platform.shipment_state"
    PLATFORM_SHIPMENT_RESOLVED = "platform.shipment_resolved"
    # Trusted environment/runtime request to issue an authoritative protocol
    # event.  The request carries only routing/configuration fields; Platform
    # derives order parties, state, revision, clock, and operation reference
    # from World before producing ``platform.deliver_protocol_event``.
    PLATFORM_ISSUE_PROTOCOL_EVENT = "platform.issue_protocol_event"
    PLATFORM_ADVANCE_MARKET_CLOCK = "platform.advance_market_clock"
    PLATFORM_MARKET_CLOCK_ADVANCED = "platform.market_clock_advanced"
    PLATFORM_DELIVER_PROTOCOL_EVENT = "platform.deliver_protocol_event"
    PLATFORM_PROTOCOL_EVENT_RECEIPT = "platform.protocol_event_receipt"
    PLATFORM_EVIDENCE_RECORD_PERSISTED = "platform.evidence_record_persisted"
    PLATFORM_MANDATE_AUTHORITY_REGISTERED = "platform.mandate_authority_registered"
    PLATFORM_MANDATE_REVISION_APPENDED = "platform.mandate_revision_appended"
    PLATFORM_LISTING_CLAIM_UPDATED = "platform.listing_claim_updated"
    # Delegate (consumer/principal ↔ buyer)
    DELEGATE_CREATE_PURCHASE_MANDATE = "delegate.create_purchase_mandate"
    DELEGATE_REGISTER_MANDATE_AUTHORITY = "delegate.register_mandate_authority"
    DELEGATE_APPEND_MANDATE_REVISION = "delegate.append_mandate_revision"
    DELEGATE_APPROVE_PURCHASE = "delegate.approve_purchase"
    DELEGATE_REJECT_PURCHASE = "delegate.reject_purchase"
    # Non-authoritative actor output accepted and journaled by Runtime.  These
    # records describe what an actor concluded; they are never Platform
    # decisions or World facts and therefore address ``runtime:evidence``.
    DELEGATE_REPORT_RESULT = "delegate.report_result"
    COMMERCE_SUBMIT_DECISION_RECORD = "commerce.submit_decision_record"
    # Large-catalog benchmark. These additive actions keep model-authored
    # catalog reads and decisions on the ordinary Agent -> Platform -> World
    # path without changing the original 200-task interface.
    COMMERCE_LARGE_CATALOG_SEARCH = "commerce.large_catalog_search"
    COMMERCE_LARGE_CATALOG_CART_SEARCH = "commerce.large_catalog_cart_search"
    COMMERCE_LARGE_CATALOG_OBSERVE = "commerce.large_catalog_observe"
    COMMERCE_LARGE_CATALOG_DECIDE = "commerce.large_catalog_decide"
    PLATFORM_LARGE_CATALOG_RESULT = "platform.large_catalog_result"
    PLATFORM_LARGE_CATALOG_DECISION = "platform.large_catalog_decision"
    # Commerce-namespace intents — the merchant emits these to platform,
    # which validates and forwards as the matching ``world.*`` write
    # (catalog / inventory) or as a buyer-side notification (dispatch /
    # respond_inquiry / send_message / cancel_order / issue_refund).
    # The merchant never writes world directly: PARTITION_ALLOW puts
    # the receiver of every commerce.* intent at ``platform`` (or buyer
    # for the notification kinds), and WorldService also rejects
    # merchant-origin ``world.update_*`` envelopes as defense-in-depth.
    # Earlier bare-name variants (``adjust_price`` / ``update_listing``
    # / ``receive_shipment`` / ``dispatch`` / ``respond_inquiry`` /
    # ``cancel_order`` / ``issue_refund`` / ``send_message``) were
    # retired in Phase 2 — only the ``commerce.*`` wire form remains.
    COMMERCE_ADJUST_PRICE = "commerce.adjust_price"
    COMMERCE_UPDATE_LISTING = "commerce.update_listing"
    COMMERCE_RECEIVE_SHIPMENT = "commerce.receive_shipment"
    COMMERCE_DISPATCH = "commerce.dispatch"
    COMMERCE_MARK_RETURNED = "commerce.mark_returned"
    COMMERCE_RESPOND_INQUIRY = "commerce.respond_inquiry"
    COMMERCE_CANCEL_ORDER = "commerce.cancel_order"
    COMMERCE_ISSUE_REFUND = "commerce.issue_refund"
    COMMERCE_SEND_MESSAGE = "commerce.send_message"
    # World
    WORLD_READ_CATALOG = "world.read_catalog"
    WORLD_READ_INVENTORY = "world.read_inventory"
    WORLD_READ_AVAILABILITY = "world.read_availability"
    WORLD_READ_POLICY = "world.read_policy"
    WORLD_READ_REPUTATION = "world.read_reputation"
    WORLD_READ_ORDER = "world.read_order"
    WORLD_READ_LEDGER = "world.read_ledger"
    WORLD_READ_FULFILLMENT = "world.read_fulfillment"
    WORLD_READ_DISPUTE = "world.read_dispute"
    WORLD_READ_RULING = "world.read_ruling"
    WORLD_READ_CLOCK = "world.read_clock"
    WORLD_READ_ORDER_TIMELINE = "world.read_order_timeline"
    WORLD_READ_ORDER_GROUP = "world.read_order_group"
    WORLD_READ_SUPPLY_STATE = "world.read_supply_state"
    WORLD_READ_SUPPLY_PURCHASE_AUTHORITY = "world.read_supply_purchase_authority"
    WORLD_READ_SHIPMENT = "world.read_shipment"
    WORLD_READ_ORDER_PROTOCOL_STATE = "world.read_order_protocol_state"
    WORLD_READ_PROTOCOL_EVENT = "world.read_protocol_event"
    WORLD_READ_PROTOCOL_EVENTS = "world.read_protocol_events"
    WORLD_READ_PROTOCOL_RECEIPTS = "world.read_protocol_receipts"
    # Agent-private, caller-scoped current callback authority.  The model sees
    # only the resulting business facts and never this World route or event id.
    WORLD_READ_PROTOCOL_EVENT_AUTHORITY = "world.read_protocol_event_authority"
    WORLD_READ_NEGOTIATION_EVENT = "world.read_negotiation_event"
    WORLD_READ_NEGOTIATION_THREAD = "world.read_negotiation_thread"
    WORLD_READ_NEGOTIATION_EVENTS = "world.read_negotiation_events"
    WORLD_READ_PRICING_POLICIES = "world.read_pricing_policies"
    WORLD_READ_CART_QUOTE_REQUEST = "world.read_cart_quote_request"
    WORLD_READ_CART_QUOTE = "world.read_cart_quote"
    # Caller-scoped after-sales reads are emitted only by Platform.  The
    # authenticated buyer/merchant is carried as ``original_actor`` and World
    # applies row ownership before returning a canonical wire projection.
    WORLD_READ_PAYMENT_HISTORY = "world.read_payment_history"
    WORLD_READ_LEDGER_HISTORY = "world.read_ledger_history"
    WORLD_READ_PACKING_HISTORY = "world.read_packing_history"
    WORLD_READ_AFTER_SALES_HISTORY = "world.read_after_sales_history"
    WORLD_READ_AFTER_SALES_POLICY = "world.read_after_sales_policy"
    # Agent-private, caller-scoped business authority.  Unlike the public
    # history tools this projection contains only legal business affordances
    # and sealed internal references; it is never exposed as an LLM tool.
    WORLD_READ_AFTER_SALES_AUTHORITY = "world.read_after_sales_authority"
    # Persistent discovery/matching records.  These are World-owned facts,
    # not Platform process memory: a Platform restart must not invalidate an
    # offer that is still fresh, and a forged certificate must never be
    # accepted merely because it appeared in an audit stream.
    WORLD_READ_SEARCH_SESSION = "world.read_search_session"
    WORLD_READ_MATCH_ACCEPTANCE = "world.read_match_acceptance"
    WORLD_READ_MATCH_CERTIFICATE = "world.read_match_certificate"
    WORLD_RESOLVE_SEARCH_SESSION = "world.resolve_search_session"
    WORLD_READ_FRIENDS = "world.read_friends"
    WORLD_READ_FRIEND_REVIEWS = "world.read_friend_reviews"
    WORLD_READ_REVIEW_EVIDENCE = "world.read_review_evidence"
    WORLD_UPDATE_CATALOG = "world.update_catalog"
    WORLD_APPLY_CATALOG_MUTATION = "world.apply_catalog_mutation"
    WORLD_APPLY_NEGOTIATION_INTENT = "world.apply_negotiation_intent"
    WORLD_PUBLISH_PRICING_POLICY = "world.publish_pricing_policy"
    WORLD_CREATE_CART_QUOTE_REQUEST = "world.create_cart_quote_request"
    WORLD_ISSUE_CART_QUOTE = "world.issue_cart_quote"
    WORLD_ISSUE_SUPPLY_PURCHASE_AUTHORITY = "world.issue_supply_purchase_authority"
    WORLD_CREATE_ORDER = "world.create_order"
    WORLD_RESERVE_INVENTORY = "world.reserve_inventory"
    WORLD_UPDATE_INVENTORY = "world.update_inventory"
    WORLD_UPDATE_ORDER_STATUS = "world.update_order_status"
    WORLD_UPDATE_LEDGER = "world.update_ledger"
    WORLD_UPDATE_REPUTATION = "world.update_reputation"
    WORLD_SETTLE_ORDER = "world.settle_order"
    WORLD_SETTLE_ORDER_PARTIAL = "world.settle_order_partial"
    WORLD_REFUND_ORDER = "world.refund_order"
    WORLD_DISPATCH_ORDER = "world.dispatch_order"
    WORLD_CANCEL_ORDER = "world.cancel_order"
    WORLD_MARK_ORDER_RETURNED = "world.mark_order_returned"
    WORLD_EXCHANGE_ORDER = "world.exchange_order"
    WORLD_OPEN_DISPUTE = "world.open_dispute"
    WORLD_RULE_DISPUTE = "world.rule_dispute"
    WORLD_ADVANCE_CLOCK = "world.advance_clock"
    WORLD_CHECKOUT_CART = "world.checkout_cart"
    WORLD_APPLY_SUPPLY_EVENT = "world.apply_supply_event"
    WORLD_ALLOCATE_ORDERS_ATOMIC = "world.allocate_orders_atomic"
    WORLD_RECORD_SHIPMENT_STATUS = "world.record_shipment_status"
    WORLD_RESOLVE_SHIPMENT = "world.resolve_shipment"
    WORLD_CREATE_SEARCH_SESSION = "world.create_search_session"
    WORLD_ISSUE_MATCH_CERTIFICATE = "world.issue_match_certificate"
    WORLD_PUBLISH_PROTOCOL_EVENT = "world.publish_protocol_event"
    WORLD_APPEND_PROTOCOL_RECEIPT = "world.append_protocol_receipt"
    WORLD_PROCESS_PROTOCOL_EVENT = "world.process_protocol_event"
    WORLD_PERSIST_EVIDENCE_RECORD = "world.persist_evidence_record"
    WORLD_READ_EVIDENCE_RECORD = "world.read_evidence_record"
    WORLD_REGISTER_MANDATE_AUTHORITY = "world.register_mandate_authority"
    WORLD_APPEND_MANDATE_REVISION = "world.append_mandate_revision"
    WORLD_READ_MANDATE_REVISIONS = "world.read_mandate_revisions"
    WORLD_APPLY_LISTING_CLAIM = "world.apply_listing_claim"
    WORLD_READ_LISTING_CLAIM = "world.read_listing_claim"
    WORLD_READ_LISTING_CLAIMS = "world.read_listing_claims"
    WORLD_APPLY_AFTER_SALES_INTENT = "world.apply_after_sales_intent"
    WORLD_COMPLETE_LEDGER_RECONCILIATION = (
        "world.complete_ledger_reconciliation"
    )
    WORLD_PUBLISH_AFTER_SALES_POLICY = "world.publish_after_sales_policy"
    WORLD_APPLY_PAYMENT_INTENT = "world.apply_payment_intent"
    WORLD_APPLY_PACKING_INTENT = "world.apply_packing_intent"
    WORLD_PUBLISH_GOVERNANCE_POLICY = "world.publish_governance_policy"
    WORLD_APPLY_GOVERNANCE_INTENT = "world.apply_governance_intent"
    WORLD_AGGREGATE_REVIEWS = "world.aggregate_reviews"
    WORLD_INGEST_REVIEW_OBSERVATION = "world.ingest_review_observation"
    WORLD_INGEST_MARKET_OBSERVATION = "world.ingest_market_observation"
    WORLD_RESOLVE_GOVERNANCE_CASE = "world.resolve_governance_case"
    WORLD_APPLY_GOVERNANCE_REPUTATION = "world.apply_governance_reputation"
    WORLD_CREATE_REMEDIATION_PLAN = "world.create_remediation_plan"
    WORLD_VERIFY_REMEDIATION_STEP = "world.verify_remediation_step"
    WORLD_PERSIST_RANKING_CONTEXT = "world.persist_ranking_context"
    WORLD_RANKING_CONTEXT_PROJECTION = "world.ranking_context_projection"
    WORLD_GOVERNANCE_HISTORY = "world.governance_history"
    WORLD_COMPUTE_STATE_DIFF = "world.compute_state_diff"
    WORLD_LARGE_CATALOG_SEARCH = "world.large_catalog_search"
    WORLD_LARGE_CATALOG_CART_SEARCH = "world.large_catalog_cart_search"
    WORLD_LARGE_CATALOG_OBSERVE = "world.large_catalog_observe"
    WORLD_LARGE_CATALOG_COMMIT = "world.large_catalog_commit"
    WORLD_RESPONSE = "world.response"


# Allow-table: ActionKind -> {"senders": frozenset[str], "receivers": frozenset[str]}.
# Roles are matched by the side prefix of an agent's id (e.g. "buyer", "buyer:intent"
# both match "buyer").
PARTITION_ALLOW: dict[ActionKind, dict[str, frozenset[str]]] = {
    ActionKind.SETTLE: {
        "senders": frozenset({"buyer", "platform"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_SETTLE_PAYMENT: {
        "senders": frozenset({"buyer", "platform"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_UPDATE_REPUTATION: {
        "senders": frozenset({"runtime", "platform"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_SETTLEMENT_RECEIPT: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer", "merchant", "runtime"}),
    },
    ActionKind.PLATFORM_FULFILLMENT_ALLOCATION: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer", "merchant", "runtime"}),
    },
    ActionKind.PLATFORM_REPUTATION_UPDATED: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"runtime", "buyer", "merchant"}),
    },
    ActionKind.PLATFORM_CATALOG_ACK: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"merchant"}),
    },
    ActionKind.PLATFORM_DELIVER_PROTOCOL_EVENT: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer", "merchant"}),
    },
    ActionKind.PLATFORM_PROTOCOL_EVENT_RECEIPT: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer", "merchant", "runtime"}),
    },
    ActionKind.PLATFORM_EVIDENCE_RECORD_PERSISTED: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer", "merchant", "runtime"}),
    },
    ActionKind.PLATFORM_MANDATE_AUTHORITY_REGISTERED: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"consumer", "runtime"}),
    },
    ActionKind.PLATFORM_MANDATE_REVISION_APPENDED: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"consumer", "runtime"}),
    },
    ActionKind.PLATFORM_LISTING_CLAIM_UPDATED: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"merchant", "runtime"}),
    },
    ActionKind.PLATFORM_ISSUE_PROTOCOL_EVENT: {
        "senders": frozenset({"runtime"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_ADVANCE_MARKET_CLOCK: {
        "senders": frozenset({"runtime"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_MARKET_CLOCK_ADVANCED: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"runtime"}),
    },
    # Discovery — buyer search goes through platform:aggregator (Phase 4).
    ActionKind.SEARCH: {
        "senders": frozenset({"buyer"}),
        "receivers": frozenset({"platform"}),         # platform:aggregator
    },
    ActionKind.GET_SKU: {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_RANK_OFFERS: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer"}),
    },
    ActionKind.PLATFORM_CREATE_MATCH_CERTIFICATE: {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer"}),
    },
    # Negotiation — symmetric: either side may initiate or respond.
    ActionKind.PROPOSE_OFFER: {
        "senders": frozenset({"buyer", "merchant", "platform"}),
        "receivers": frozenset({"buyer", "merchant", "platform"}),
    },
    ActionKind.COUNTER_OFFER: {
        "senders": frozenset({"buyer", "merchant", "platform"}),
        "receivers": frozenset({"buyer", "merchant", "platform"}),
    },
    ActionKind.ACCEPT_OFFER: {
        # Buyer accepts a rank_offers winner by sending to platform:aggregator
        # (which issues the match certificate). Negotiation accept_offer between
        # buyer ↔ merchant goes peer-to-peer.
        "senders": frozenset({"buyer", "merchant", "platform"}),
        "receivers": frozenset({"buyer", "merchant", "platform"}),
    },
    ActionKind.REJECT_OFFER: {
        "senders": frozenset({"buyer", "merchant", "platform"}),
        "receivers": frozenset({"buyer", "merchant", "platform"}),
    },
    ActionKind.WITHDRAW_OFFER: {
        "senders": frozenset({"buyer", "merchant", "platform"}),
        "receivers": frozenset({"buyer", "merchant", "platform"}),
    },
    # Delegate lane (consumer/principal ↔ buyer)
    ActionKind.DELEGATE_CREATE_PURCHASE_MANDATE: {
        "senders": frozenset({"consumer"}),
        "receivers": frozenset({"buyer"}),
    },
    ActionKind.DELEGATE_REGISTER_MANDATE_AUTHORITY: {
        "senders": frozenset({"consumer"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.DELEGATE_APPEND_MANDATE_REVISION: {
        "senders": frozenset({"consumer"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.DELEGATE_APPROVE_PURCHASE: {
        "senders": frozenset({"consumer"}),
        "receivers": frozenset({"buyer"}),
    },
    ActionKind.DELEGATE_REJECT_PURCHASE: {
        "senders": frozenset({"buyer", "consumer"}),
        "receivers": frozenset({"buyer", "consumer"}),
    },
    ActionKind.DELEGATE_REPORT_RESULT: {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"runtime"}),
    },
    ActionKind.COMMERCE_SUBMIT_DECISION_RECORD: {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"runtime"}),
    },
    ActionKind.COMMERCE_ACKNOWLEDGE_PROTOCOL_EVENT: {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.COMMERCE_REJECT_PROTOCOL_EVENT: {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.COMMERCE_PROCESS_PROTOCOL_EVENT: {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.COMMERCE_PUBLISH_EVIDENCE_RECORD: {
        "senders": frozenset({"buyer", "merchant", "runtime"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.PLATFORM_PUBLISH_EVIDENCE_RECORD: {
        "senders": frozenset({"runtime"}),
        "receivers": frozenset({"platform"}),
    },
    ActionKind.COMMERCE_APPLY_LISTING_CLAIM: {
        "senders": frozenset({"merchant"}),
        "receivers": frozenset({"platform"}),
    },
}

# World reads — public / scoped-by-caller per WORLD_DESIGN.md §7.
for _kind in (
    ActionKind.WORLD_READ_CATALOG,
    ActionKind.WORLD_READ_AVAILABILITY,
    ActionKind.WORLD_READ_POLICY,
    ActionKind.WORLD_READ_REPUTATION,
    ActionKind.WORLD_READ_ORDER,
    ActionKind.WORLD_READ_LEDGER,
    ActionKind.WORLD_READ_FULFILLMENT,
    ActionKind.WORLD_READ_DISPUTE,
    ActionKind.WORLD_READ_RULING,
    ActionKind.WORLD_READ_CLOCK,
    ActionKind.WORLD_READ_ORDER_TIMELINE,
    ActionKind.WORLD_READ_ORDER_GROUP,
    ActionKind.WORLD_READ_FRIENDS,
    ActionKind.WORLD_READ_FRIEND_REVIEWS,
    ActionKind.WORLD_READ_REVIEW_EVIDENCE,
    ActionKind.WORLD_READ_SHIPMENT,
    ActionKind.WORLD_READ_ORDER_PROTOCOL_STATE,
    ActionKind.WORLD_READ_PROTOCOL_EVENT,
    ActionKind.WORLD_READ_PROTOCOL_EVENTS,
    ActionKind.WORLD_READ_PROTOCOL_RECEIPTS,
    ActionKind.WORLD_READ_PROTOCOL_EVENT_AUTHORITY,
    ActionKind.WORLD_READ_NEGOTIATION_EVENT,
    ActionKind.WORLD_READ_NEGOTIATION_THREAD,
    ActionKind.WORLD_READ_NEGOTIATION_EVENTS,
    ActionKind.WORLD_READ_PRICING_POLICIES,
    ActionKind.WORLD_READ_CART_QUOTE_REQUEST,
    ActionKind.WORLD_READ_CART_QUOTE,
    ActionKind.WORLD_READ_EVIDENCE_RECORD,
    ActionKind.WORLD_READ_LISTING_CLAIM,
    ActionKind.WORLD_READ_LISTING_CLAIMS,
    ActionKind.WORLD_READ_AFTER_SALES_AUTHORITY,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"buyer", "merchant", "platform", "runtime"}),
        "receivers": frozenset({"world"}),
    }

for _kind in (
    ActionKind.WORLD_READ_CART_QUOTE_REQUEST,
    ActionKind.WORLD_READ_CART_QUOTE,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset(
            {"buyer", "consumer", "merchant", "platform", "runtime"}
        ),
        "receivers": frozenset({"world"}),
    }

# A mandate is private to its principal and delegated buyer.  Row-level ACL is
# enforced by World; the protocol partition additionally permits the principal
# lane to perform the same caller-bound read over HTTP.
PARTITION_ALLOW[ActionKind.WORLD_READ_MANDATE_REVISIONS] = {
    "senders": frozenset({"buyer", "consumer", "platform", "runtime"}),
    "receivers": frozenset({"world"}),
}

# Match records contain actor- and order-bound transaction material.  Agents
# receive the public typed Platform replies; only trusted Platform services and
# the runtime verifier may read the authoritative persisted rows directly.
for _kind in (
    ActionKind.WORLD_READ_SEARCH_SESSION,
    ActionKind.WORLD_READ_MATCH_ACCEPTANCE,
    ActionKind.WORLD_READ_MATCH_CERTIFICATE,
    ActionKind.WORLD_RESOLVE_SEARCH_SESSION,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform", "runtime"}),
        "receivers": frozenset({"world"}),
    }

PARTITION_ALLOW[ActionKind.WORLD_CHECKOUT_CART] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"world"}),
}

for _kind in (
    ActionKind.WORLD_APPLY_CATALOG_MUTATION,
    ActionKind.WORLD_APPLY_NEGOTIATION_INTENT,
    ActionKind.WORLD_PUBLISH_PRICING_POLICY,
    ActionKind.WORLD_CREATE_CART_QUOTE_REQUEST,
    ActionKind.WORLD_ISSUE_CART_QUOTE,
    ActionKind.WORLD_PERSIST_EVIDENCE_RECORD,
    ActionKind.WORLD_REGISTER_MANDATE_AUTHORITY,
    ActionKind.WORLD_APPEND_MANDATE_REVISION,
    ActionKind.WORLD_APPLY_LISTING_CLAIM,
    ActionKind.WORLD_PROCESS_PROTOCOL_EVENT,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"world"}),
    }

PARTITION_ALLOW[ActionKind.WORLD_READ_SUPPLY_STATE] = {
    "senders": frozenset({"platform", "runtime", "merchant"}),
    "receivers": frozenset({"world"}),
}
PARTITION_ALLOW[ActionKind.WORLD_READ_SUPPLY_PURCHASE_AUTHORITY] = {
    "senders": frozenset({"platform", "runtime"}),
    "receivers": frozenset({"world"}),
}

# Raw inventory LEVELS are commercially sensitive (exact qty_available /
# qty_reserved). Only platform/runtime (settle + fulfillment genuinely need the
# counts) and the OWNING merchant may read the row; a buyer gets availability via
# world.read_availability (a bool, no counts). The owning-merchant scoping is
# enforced row-level in WorldService (a merchant may not read a competitor's
# levels). Buyer is removed from the SENDERS here so a third-party agent cannot
# harvest every merchant's stock by hand-crafting a raw read_inventory envelope
# (the privacy boundary must hold at the protocol layer, not just the tool layer).
PARTITION_ALLOW[ActionKind.WORLD_READ_INVENTORY] = {
    "senders": frozenset({"platform", "runtime", "merchant"}),
    "receivers": frozenset({"world"}),
}

# Deterministic time is advanced only by the runtime control-plane lane; World
# additionally requires the exact ``runtime:clock`` address.
PARTITION_ALLOW[ActionKind.WORLD_ADVANCE_CLOCK] = {
    "senders": frozenset({"runtime"}),
    "receivers": frozenset({"world"}),
}

# Platform-only world writes (PSP / settlement).
for _kind in (
    ActionKind.WORLD_CREATE_ORDER,
    ActionKind.WORLD_RESERVE_INVENTORY,
    ActionKind.WORLD_UPDATE_ORDER_STATUS,
    ActionKind.WORLD_UPDATE_LEDGER,
    ActionKind.WORLD_SETTLE_ORDER,
    ActionKind.WORLD_SETTLE_ORDER_PARTIAL,
    ActionKind.WORLD_REFUND_ORDER,
    ActionKind.WORLD_DISPATCH_ORDER,
    ActionKind.WORLD_CANCEL_ORDER,
    ActionKind.WORLD_MARK_ORDER_RETURNED,
    ActionKind.WORLD_EXCHANGE_ORDER,
    ActionKind.WORLD_OPEN_DISPUTE,
    ActionKind.WORLD_RULE_DISPUTE,
    ActionKind.WORLD_APPLY_SUPPLY_EVENT,
    ActionKind.WORLD_ALLOCATE_ORDERS_ATOMIC,
    ActionKind.WORLD_RECORD_SHIPMENT_STATUS,
    ActionKind.WORLD_RESOLVE_SHIPMENT,
    ActionKind.WORLD_CREATE_SEARCH_SESSION,
    ActionKind.WORLD_ISSUE_MATCH_CERTIFICATE,
    ActionKind.WORLD_ISSUE_SUPPLY_PURCHASE_AUTHORITY,
    ActionKind.WORLD_PUBLISH_PROTOCOL_EVENT,
    ActionKind.WORLD_APPEND_PROTOCOL_RECEIPT,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"world"}),
    }

# world.update_* writes are platform-only — the merchant must NOT write
# directly to world (per the design: agents change world state only
# through VCP with the platform). The merchant emits the commerce.*
    # intent (adjust_price / update_listing / receive_shipment / dispatch)
    # to ``platform:catalog``.  Catalog intents become the actor-aware
    # ``world.apply_catalog_mutation`` transaction; shipment inventory remains
    # a separate validated world write.
PARTITION_ALLOW[ActionKind.WORLD_UPDATE_CATALOG] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"world"}),
}
PARTITION_ALLOW[ActionKind.WORLD_UPDATE_INVENTORY] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"world"}),
}
PARTITION_ALLOW[ActionKind.WORLD_UPDATE_REPUTATION] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"world"}),
}
PARTITION_ALLOW[ActionKind.WORLD_COMPUTE_STATE_DIFF] = {
    "senders": frozenset({"runtime"}),
    "receivers": frozenset({"world"}),
}
PARTITION_ALLOW[ActionKind.WORLD_RESPONSE] = {
    "senders": frozenset({"world"}),
    "receivers": frozenset({"buyer", "merchant", "platform", "runtime"}),
}

for _kind in (
    ActionKind.COMMERCE_LARGE_CATALOG_SEARCH,
    ActionKind.COMMERCE_LARGE_CATALOG_CART_SEARCH,
    ActionKind.COMMERCE_LARGE_CATALOG_OBSERVE,
    ActionKind.COMMERCE_LARGE_CATALOG_DECIDE,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"platform"}),
    }

for _kind in (
    ActionKind.WORLD_LARGE_CATALOG_SEARCH,
    ActionKind.WORLD_LARGE_CATALOG_CART_SEARCH,
    ActionKind.WORLD_LARGE_CATALOG_OBSERVE,
    ActionKind.WORLD_LARGE_CATALOG_COMMIT,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"world"}),
    }

for _kind in (
    ActionKind.PLATFORM_LARGE_CATALOG_RESULT,
    ActionKind.PLATFORM_LARGE_CATALOG_DECISION,
):
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"buyer", "merchant"}),
    }

# ---- commerce.* — merchant-side intents that platform forwards -------------
#
# These commerce-namespace actions are merchant-emitted intents. After the
# "no direct world writes from merchant" policy, the merchant addresses
# ``platform:catalog`` (and other platform sub-services). Platform authenticates
# the actor and forwards compact intents; World validates ownership and applies
# the mutation. Receivers therefore stop at platform, not at world.
PARTITION_ALLOW[ActionKind.COMMERCE_UPDATE_LISTING] = {
    # Owner directives (delegate.update_listing) land merchant->merchant
    # (merchant:owner -> merchant:catalog). The merchant's outbound
    # commerce.update_listing goes to ``platform:catalog`` which calls the
    # actor-aware ``world.apply_catalog_mutation`` transaction. The merchant
    # never addresses ``world`` directly.
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"merchant", "platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_ADJUST_PRICE] = {
    # commerce.adjust_price is the merchant's price-change intent. Platform
    # authenticates the actor and World validates owner/current row through
    # world.apply_catalog_mutation. Merchant never writes catalog directly.
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_CATALOG_LISTING] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant"}),
}
# Free-text channel — chat / pulse / inquiry between any pair, plus an
# owner-driven pulse (merchant:owner -> merchant:fulfillment); receiver
# side {merchant, buyer, platform} covers all the existing flows.
PARTITION_ALLOW[ActionKind.COMMERCE_SEND_MESSAGE] = {
    "senders": frozenset({"buyer", "merchant", "platform"}),
    "receivers": frozenset({"buyer", "merchant", "platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_RESPOND_INQUIRY] = {
    # Pre-purchase Q&A reply, also used to decline owner directives or to
    # send a public-reason return decline. Receivers include merchant so
    # owner-side directive declines can land on merchant:owner.
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"buyer", "merchant", "platform"}),
}
# Mandate / cart (bare wire form — no commerce.* counterpart yet)
PARTITION_ALLOW[ActionKind.CREATE_ORDER] = {
    "senders": frozenset({"buyer", "platform"}),
    "receivers": frozenset({"merchant", "platform"}),
}
PARTITION_ALLOW[ActionKind.AUTHORIZE_PAYMENT] = {
    "senders": frozenset({"buyer", "platform"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_REQUEST_CART_QUOTE] = {
    "senders": frozenset({"buyer", "merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_CREATE_CART_QUOTE_REQUEST] = {
    "senders": frozenset({"buyer", "consumer"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_CART_QUOTE_REQUEST] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "consumer", "merchant", "runtime"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_CART_QUOTE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_CHECKOUT_CART] = {
    "senders": frozenset({"buyer"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_CART_SETTLEMENT] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_READ_SUPPLY_STATE] = {
    "senders": frozenset({"buyer", "merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_UPDATE_SUPPLY] = {
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_APPLY_SUPPLY_EVENT] = {
    "senders": frozenset({"runtime"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_SUPPLY_STATE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant", "runtime"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_SUPPLY_EVENT_APPLIED] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"merchant", "runtime"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_ALLOCATE_FULFILLMENT] = {
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_ALLOCATION_BATCH] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"merchant", "runtime"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_READ_SHIPMENT] = {
    "senders": frozenset({"buyer", "merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_RESOLVE_SHIPMENT] = {
    "senders": frozenset({"buyer", "merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_RECORD_SHIPMENT_STATUS] = {
    "senders": frozenset({"runtime"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_SHIPMENT_STATE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant", "runtime"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_SHIPMENT_RESOLVED] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant"}),
}
# Settlement / fulfillment
PARTITION_ALLOW[ActionKind.COMMERCE_DISPATCH] = {
    # commerce.dispatch is the merchant's fulfillment intent. Buyer gets
    # the delivery notice; platform forwards inventory decrement +
    # order-status flip to world. Merchant never writes order/inventory
    # tables directly.
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"buyer", "platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_MARK_RETURNED] = {
    # A physical return is acknowledged by the exact merchant that owns the
    # order. Platform preserves that full actor id when committing the state
    # transition in World.
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_RECEIVE_SHIPMENT] = {
    # delegate.receive_shipment lands merchant->merchant (owner pulse to
    # fulfillment role). The merchant's outbound commerce.receive_shipment
    # is the restock intent; platform receives, validates ownership,
    # forwards as ``world.update_inventory`` (qty_available += delta).
    # Merchant never writes inventory directly.
    "senders": frozenset({"merchant"}),
    "receivers": frozenset({"merchant", "platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_CANCEL_ORDER] = {
    "senders": frozenset({"buyer", "merchant", "platform"}),
    "receivers": frozenset({"buyer", "merchant", "platform"}),
}
PARTITION_ALLOW[ActionKind.REQUEST_RETURN] = {
    "senders": frozenset({"buyer"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.COMMERCE_REQUEST_EXCHANGE] = {
    "senders": frozenset({"buyer"}),
    "receivers": frozenset({"platform"}),
}
for _kind, _senders in {
    ActionKind.COMMERCE_CANCEL_PAID_ORDER: frozenset({"buyer", "merchant"}),
    ActionKind.COMMERCE_AUTHORIZE_RETURN: frozenset({"merchant"}),
    ActionKind.COMMERCE_DENY_RETURN: frozenset({"merchant"}),
    ActionKind.COMMERCE_RECEIVE_RETURN: frozenset({"merchant"}),
    ActionKind.COMMERCE_OPEN_REFUND_CASE: frozenset({"buyer"}),
    ActionKind.COMMERCE_APPROVE_REFUND: frozenset({"merchant"}),
    ActionKind.COMMERCE_DENY_REFUND: frozenset({"merchant"}),
    ActionKind.COMMERCE_AUTHORIZE_EXCHANGE: frozenset({"merchant"}),
    ActionKind.COMMERCE_DENY_EXCHANGE: frozenset({"merchant"}),
    ActionKind.COMMERCE_COMPLETE_EXCHANGE: frozenset({"merchant"}),
    ActionKind.COMMERCE_OPEN_DISPUTE: frozenset({"buyer", "merchant"}),
    ActionKind.COMMERCE_SUBMIT_DISPUTE_EVIDENCE: frozenset(
        {"buyer", "merchant"}
    ),
    ActionKind.COMMERCE_RESPOND_TO_DISPUTE: frozenset({"buyer", "merchant"}),
    ActionKind.COMMERCE_REQUEST_LEDGER_RECONCILIATION: frozenset(
        {"buyer", "merchant"}
    ),
}.items():
    PARTITION_ALLOW[_kind] = {
        "senders": _senders,
        "receivers": frozenset({"platform"}),
    }
for _kind in {
    ActionKind.COMMERCE_READ_PAYMENT_HISTORY,
    ActionKind.COMMERCE_READ_LEDGER_HISTORY,
    ActionKind.COMMERCE_READ_PACKING_HISTORY,
    ActionKind.COMMERCE_READ_AFTER_SALES_HISTORY,
    ActionKind.COMMERCE_READ_AFTER_SALES_POLICY,
}:
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"buyer", "merchant"}),
        "receivers": frozenset({"platform"}),
    }
PARTITION_ALLOW[ActionKind.PLATFORM_AFTER_SALES_UPDATED] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_AFTER_SALES_SNAPSHOT] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant"}),
}
for _kind in {
    ActionKind.WORLD_APPLY_AFTER_SALES_INTENT,
    ActionKind.WORLD_COMPLETE_LEDGER_RECONCILIATION,
    ActionKind.WORLD_PUBLISH_AFTER_SALES_POLICY,
    ActionKind.WORLD_APPLY_PAYMENT_INTENT,
    ActionKind.WORLD_APPLY_PACKING_INTENT,
    ActionKind.WORLD_READ_PAYMENT_HISTORY,
    ActionKind.WORLD_READ_LEDGER_HISTORY,
    ActionKind.WORLD_READ_PACKING_HISTORY,
    ActionKind.WORLD_READ_AFTER_SALES_HISTORY,
    ActionKind.WORLD_READ_AFTER_SALES_POLICY,
}:
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"world"}),
    }

# Marketplace governance remains Platform-mediated.  Actor requests never
# address World, and all trusted control-plane operations cross the same VCP
# boundary rather than reaching a shared World object.
for _kind, _senders in {
    ActionKind.COMMERCE_PUBLISH_CAMPAIGN: frozenset({"merchant"}),
    ActionKind.COMMERCE_DISCLOSE_PLACEMENT: frozenset({"merchant"}),
    ActionKind.COMMERCE_ACTIVATE_CAMPAIGN: frozenset({"merchant"}),
    ActionKind.COMMERCE_SUBMIT_REVIEW: frozenset({"buyer"}),
    ActionKind.COMMERCE_REJECT_REVIEW_MANIPULATION: frozenset({"merchant"}),
    ActionKind.COMMERCE_REJECT_COORDINATION: frozenset({"merchant"}),
    ActionKind.COMMERCE_ACCEPT_REMEDIATION_PLAN: frozenset({"merchant"}),
    ActionKind.COMMERCE_COMPLETE_REMEDIATION_STEP: frozenset({"merchant"}),
    ActionKind.COMMERCE_READ_GOVERNANCE_HISTORY: frozenset(
        {"buyer", "merchant"}
    ),
}.items():
    PARTITION_ALLOW[_kind] = {
        "senders": _senders,
        "receivers": frozenset({"platform"}),
    }

for _kind in {
    ActionKind.PLATFORM_PUBLISH_GOVERNANCE_POLICY,
    ActionKind.PLATFORM_AGGREGATE_REVIEWS,
    ActionKind.PLATFORM_INGEST_REVIEW_OBSERVATION,
    ActionKind.PLATFORM_INGEST_MARKET_OBSERVATION,
    ActionKind.PLATFORM_RESOLVE_GOVERNANCE_CASE,
    ActionKind.PLATFORM_APPLY_GOVERNANCE_REPUTATION,
    ActionKind.PLATFORM_CREATE_REMEDIATION_PLAN,
    ActionKind.PLATFORM_VERIFY_REMEDIATION_STEP,
    ActionKind.PLATFORM_PERSIST_RANKING_CONTEXT,
}:
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"runtime", "platform"}),
        "receivers": frozenset({"platform"}),
    }

PARTITION_ALLOW[ActionKind.PLATFORM_GOVERNANCE_UPDATED] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant", "runtime", "platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_GOVERNANCE_SNAPSHOT] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant", "runtime", "platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_GOVERNANCE_CASE_NOTICE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"merchant"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_REMEDIATION_PLAN_NOTICE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"merchant"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_REMEDIATION_AUDIT_REQUEST] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"runtime"}),
}

for _kind in {
    ActionKind.WORLD_PUBLISH_GOVERNANCE_POLICY,
    ActionKind.WORLD_APPLY_GOVERNANCE_INTENT,
    ActionKind.WORLD_AGGREGATE_REVIEWS,
    ActionKind.WORLD_INGEST_REVIEW_OBSERVATION,
    ActionKind.WORLD_INGEST_MARKET_OBSERVATION,
    ActionKind.WORLD_RESOLVE_GOVERNANCE_CASE,
    ActionKind.WORLD_APPLY_GOVERNANCE_REPUTATION,
    ActionKind.WORLD_CREATE_REMEDIATION_PLAN,
    ActionKind.WORLD_VERIFY_REMEDIATION_STEP,
    ActionKind.WORLD_PERSIST_RANKING_CONTEXT,
    ActionKind.WORLD_RANKING_CONTEXT_PROJECTION,
    ActionKind.WORLD_GOVERNANCE_HISTORY,
}:
    PARTITION_ALLOW[_kind] = {
        "senders": frozenset({"platform"}),
        "receivers": frozenset({"world"}),
    }
PARTITION_ALLOW[ActionKind.COMMERCE_ISSUE_REFUND] = {
    "senders": frozenset({"merchant", "platform"}),
    "receivers": frozenset({"buyer", "platform"}),
}
# Platform / dispute / reputation
PARTITION_ALLOW[ActionKind.OPEN_DISPUTE] = {
    "senders": frozenset({"buyer", "merchant", "platform"}),
    "receivers": frozenset({"buyer", "merchant", "platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_OPEN_DISPUTE] = {
    "senders": frozenset({"buyer", "merchant", "platform"}),
    "receivers": frozenset({"platform"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_RULE_DISPUTE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"platform", "buyer", "merchant"}),
}
PARTITION_ALLOW[ActionKind.PLATFORM_LIFECYCLE_UPDATED] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"platform", "buyer", "merchant", "runtime"}),
}
PARTITION_ALLOW[ActionKind.UPDATE_REPUTATION] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"world"}),
}
PARTITION_ALLOW[ActionKind.RULE] = {
    "senders": frozenset({"platform"}),
    "receivers": frozenset({"buyer", "merchant"}),
}


def is_send_allowed(kind: ActionKind, sender_role: str, receiver_role: str) -> bool:
    """Return ``True`` iff ``sender_role`` may emit ``kind`` to ``receiver_role``.

    ``sender_role`` and ``receiver_role`` are side prefixes (``buyer`` /
    ``merchant`` / ``platform``), not full agent ids.
    """
    entry = PARTITION_ALLOW.get(kind)
    if entry is None:
        return False
    return sender_role in entry["senders"] and receiver_role in entry["receivers"]


def is_deferred(kind: ActionKind) -> bool:
    """Return ``True`` iff ``kind`` may be handled asynchronously."""
    return kind in {
        ActionKind.PLATFORM_SETTLE_PAYMENT,
        ActionKind.WORLD_UPDATE_LEDGER,
        ActionKind.WORLD_COMPUTE_STATE_DIFF,
    }
