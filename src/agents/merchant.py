"""Merchant-side agent factory.

Per AGENT_CLASSES.md §2 / §8, the merchant is a single ``Agent("merchant")``
(or ``Agent("merchant:<n>")`` for multi-merchant populations) whose behavior
is composed from a set of enabled SKILL.md bundles loaded via ``SkillLoader``.
No ``MerchantMainSkill`` Python class — skills are bundles on disk, not
subclasses.

The skill bundles live under ``skills/merchant/<name>/SKILL.md``; the factory
just decides **which** bundles are enabled and lets the loader handle the
rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from agents.base import Agent
from agents.memory import InMemoryStore
from agents.skills import EmptySkillLoader, FilesystemSkillLoader
from protocol.actions import ActionKind

if TYPE_CHECKING:
    from agents.inference import InferenceChannel
    from agents.interfaces import Memory, SkillLoader
    from agents.types import AgentInputs


# --- Default enabled skill set ---------------------------------------
#
# Names match ``skills/merchant/<name>/SKILL.md`` directory names. This is the
# merchant lane's hook (AGENT_CLASSES.md §8): the factory just decides which
# bundles are enabled; the loader exposes manifests at spawn and loads full
# SKILL.md bodies on demand. Order is spawn / scan order.
#
#   price-discovery         INTERPRETATION  — "fair price" -> defensible band
#   catalog-serve           DISCOVERY       — truthful GroundedOffer to buyers
#   listing-publish         DISCOVERY       — owner-driven catalog authoring
#                                             (publish / update / delist) via
#                                             commerce.update_listing
#   inquiry-handle          DISCOVERY       — buyer-side pre-purchase Q&A
#                                             (shipping / availability / attr /
#                                             policy) via commerce.respond_inquiry
#   pricing-negotiate       ECONOMIC        — accept / counter / reject
#   stockout-aware-pricing  ECONOMIC        — pre-step that biases counter on
#                                             scarce + in-demand SKUs
#   reputation-aware-pricing TRUST          — pre-step that biases counter by
#                                             the merchant's own platform
#                                             reputation (steady-state regime;
#                                             scarcity dominates when both apply)
#   claim-truthfulness      TRUST           — pre-step that filters offer
#                                             claims down to the subset
#                                             grounded in the listing's
#                                             attributes (truthfulness contract)
#   aging-markdown          ECONOMIC        — proactive list-price step-down
#                                             on aged, slow-moving stock
#   demand-driven-markup    ECONOMIC        — proactive list-price step-up on
#                                             fresh + fast + high-demand stock
#                                             (mirror of aging-markdown; capped
#                                             at target_band[1]; defers during
#                                             scarcity)
#   peer-pricing            ECONOMIC        — proactive list-price alignment
#                                             with comparable items already
#                                             listed in the world catalog
#                                             (median / quartile strategies);
#                                             capped at target_band[1] and
#                                             strictly above floor; defers to
#                                             aging-markdown,
#                                             demand-driven-markup, and
#                                             stockout-aware-pricing
#   order-cancel            OPERATIONS      — pre-dispatch cancellation;
#                                             status-gated full-refund or a
#                                             public-reason decline
#   order-intake            OPERATIONS      — record commerce.create_order into
#                                             agent-side TRANSACTION memory so
#                                             fulfillment / return-adjudicate /
#                                             dispute-defense can find the order
#                                             when later turns need it
#   fulfillment             OPERATIONS      — dispatch + policy-bound returns
#   return-adjudicate       OPERATIONS      — policy-bound return decision
#                                             (approve/partial/decline) via
#                                             commerce.issue_refund or a
#                                             commerce.respond_inquiry policy
#                                             decline
#   dispute-defense         OPERATIONS      — evidence brief when buyer
#                                             escalates via platform.open_dispute
#   inbound-restock         OPERATIONS      — record an owner-shipped lot (+qty)
#                                             via commerce.receive_shipment;
#                                             mirror of fulfillment's dispatch.
#   restock-signal          OPERATIONS      — internal note when stock low
#   private-utility-guard   SAFETY          — floor / urgency never on the wire
DEFAULT_MERCHANT_SKILLS: tuple[str, ...] = (
    # Phase 4's omnibus ``pricing`` skill was removed in B4 commit 2 —
    # ``pricing-negotiate`` + the stockout / reputation pre-steps cover
    # its surface, and the selector's defer rule would have dropped it
    # on every offer turn anyway.
    "price-discovery",
    "catalog-serve",
    "listing-publish",
    "inquiry-handle",
    "pricing-negotiate",
    "stockout-aware-pricing",
    "reputation-aware-pricing",
    "claim-truthfulness",
    "aging-markdown",
    "demand-driven-markup",
    "peer-pricing",
    "order-cancel",
    "order-intake",
    "fulfillment",
    "return-adjudicate",
    "dispute-defense",
    "inbound-restock",
    "restock-signal",
    "private-utility-guard",
    "listing-claim-manage",
    "cart-quote-handle",
    "supply-logistics",
    "after-sales-lifecycle",
    "market-governance",
    "protocol-event-handle",
)


# --- Side partition --------------------------------------------------
#
# The single P0 ``Agent("merchant")`` covers the merchant-side roles of the
# VCP_SPEC.md §6 matrix (catalog / retrieval / pricing / fulfillment /
# support). Values come from the ``ActionKind`` enum (not bare strings) so the
# eventual VCP namespace migration (commerce.* / world.*) propagates
# mechanically. The runtime's allow-table is authoritative; these are the
# agent's declared kinds.
MERCHANT_INBOUND: frozenset[str] = frozenset({
    # Phase 4: buyer search goes through ``platform:aggregator``, not direct
    # to merchant — SEARCH / GET_SKU receivers are platform-only in the
    # partition table. Don't declare them inbound on the merchant or the
    # router rejects the merchant at register time.
    ActionKind.PROPOSE_OFFER.value,
    ActionKind.COUNTER_OFFER.value,
    ActionKind.ACCEPT_OFFER.value,
    ActionKind.REJECT_OFFER.value,
    ActionKind.WITHDRAW_OFFER.value,
    ActionKind.CREATE_ORDER.value,
    # commerce.settle goes buyer/platform -> platform only (main's allow-table);
    # the merchant hears settlement via platform.settlement_receipt below.
    ActionKind.COMMERCE_RECEIVE_SHIPMENT.value,  # delegate.receive_shipment manifest from merchant:owner
    ActionKind.COMMERCE_UPDATE_LISTING.value,    # delegate.update_listing manifest from merchant:owner
    ActionKind.COMMERCE_CANCEL_ORDER.value,      # commerce.cancel_order from buyer (pre-dispatch)
    ActionKind.OPEN_DISPUTE.value,      # platform.open_dispute notification from platform:adjudicator
    ActionKind.RULE.value,              # platform.rule outcome lands on merchant after dispute
    ActionKind.PLATFORM_SETTLEMENT_RECEIPT.value,  # PSP confirms an order settled
    ActionKind.PLATFORM_REPUTATION_UPDATED.value,  # platform pushes a reputation delta
    ActionKind.PLATFORM_CATALOG_ACK.value,         # CatalogPolicy ack for forwarded commerce.* writes
    ActionKind.PLATFORM_CATALOG_LISTING.value,     # owner-scoped World-backed listing read
    ActionKind.PLATFORM_LIFECYCLE_UPDATED.value,
    ActionKind.PLATFORM_RULE_DISPUTE.value,
    ActionKind.PLATFORM_SUPPLY_STATE.value,
    ActionKind.PLATFORM_SUPPLY_EVENT_APPLIED.value,
    ActionKind.PLATFORM_ALLOCATION_BATCH.value,
    ActionKind.PLATFORM_SHIPMENT_STATE.value,
    ActionKind.PLATFORM_SHIPMENT_RESOLVED.value,
    ActionKind.PLATFORM_DELIVER_PROTOCOL_EVENT.value,
    ActionKind.PLATFORM_PROTOCOL_EVENT_RECEIPT.value,
    ActionKind.PLATFORM_EVIDENCE_RECORD_PERSISTED.value,
    ActionKind.PLATFORM_LISTING_CLAIM_UPDATED.value,
    ActionKind.PLATFORM_AFTER_SALES_UPDATED.value,
    ActionKind.PLATFORM_AFTER_SALES_SNAPSHOT.value,
    # T8 continuations are World-derived Platform notices consumed by the
    # market-governance Skill.  Declaring them keeps the role manifest aligned
    # with the runtime bundles and the Router partition.
    ActionKind.PLATFORM_GOVERNANCE_UPDATED.value,
    ActionKind.PLATFORM_GOVERNANCE_SNAPSHOT.value,
    ActionKind.PLATFORM_GOVERNANCE_CASE_NOTICE.value,
    ActionKind.PLATFORM_REMEDIATION_PLAN_NOTICE.value,
    ActionKind.COMMERCE_SEND_MESSAGE.value,
})
MERCHANT_OUTBOUND: frozenset[str] = frozenset({
    ActionKind.COUNTER_OFFER.value,
    ActionKind.ACCEPT_OFFER.value,
    ActionKind.REJECT_OFFER.value,
    ActionKind.WITHDRAW_OFFER.value,
    # commerce.* intents addressed to platform:catalog — the platform
    # validates ownership / payload shape and forwards as the matching
    # ``world.*`` write. The merchant never writes world tables directly
    # (catalog / inventory / orders); the partition table enforces this
    # at the router, and WorldService also rejects merchant-origin
    # world.update_* envelopes as defense-in-depth.
    ActionKind.COMMERCE_DISPATCH.value,           # merchant:fulfillment -> platform -> world.update_inventory + world.update_order_status
    ActionKind.COMMERCE_MARK_RETURNED.value,
    ActionKind.COMMERCE_RECEIVE_SHIPMENT.value,   # merchant:fulfillment -> platform -> world.update_inventory (+qty)
    ActionKind.COMMERCE_UPDATE_LISTING.value,     # merchant:catalog -> platform -> world.apply_catalog_mutation
    ActionKind.COMMERCE_ADJUST_PRICE.value,       # merchant -> platform -> world.apply_catalog_mutation
    ActionKind.GET_SKU.value,                     # merchant -> platform -> World-backed listing response
    ActionKind.COMMERCE_CANCEL_ORDER.value,       # merchant-initiated cancel (fraud / oversold)
    ActionKind.COMMERCE_RESPOND_INQUIRY.value,    # merchant:retrieval -> buyer Q&A reply
    ActionKind.COMMERCE_ISSUE_REFUND.value,
    ActionKind.OPEN_DISPUTE.value,       # merchant:support may escalate to adjudicator
    ActionKind.COMMERCE_SEND_MESSAGE.value,
    # T5 merchant quote response. Platform performs the calculation and owns
    # quote state; the merchant returns only the delivered request id.
    ActionKind.COMMERCE_REQUEST_CART_QUOTE.value,
    ActionKind.COMMERCE_READ_SUPPLY_STATE.value,
    ActionKind.COMMERCE_UPDATE_SUPPLY.value,
    ActionKind.COMMERCE_ALLOCATE_FULFILLMENT.value,
    ActionKind.COMMERCE_READ_SHIPMENT.value,
    ActionKind.COMMERCE_RESOLVE_SHIPMENT.value,
    ActionKind.DELEGATE_REPORT_RESULT.value,
    ActionKind.COMMERCE_SUBMIT_DECISION_RECORD.value,
    ActionKind.COMMERCE_ACKNOWLEDGE_PROTOCOL_EVENT.value,
    ActionKind.COMMERCE_REJECT_PROTOCOL_EVENT.value,
    ActionKind.COMMERCE_PROCESS_PROTOCOL_EVENT.value,
    ActionKind.COMMERCE_PUBLISH_EVIDENCE_RECORD.value,
    ActionKind.COMMERCE_APPLY_LISTING_CLAIM.value,
    ActionKind.COMMERCE_CANCEL_PAID_ORDER.value,
    ActionKind.COMMERCE_AUTHORIZE_RETURN.value,
    ActionKind.COMMERCE_DENY_RETURN.value,
    ActionKind.COMMERCE_RECEIVE_RETURN.value,
    ActionKind.COMMERCE_APPROVE_REFUND.value,
    ActionKind.COMMERCE_DENY_REFUND.value,
    ActionKind.COMMERCE_AUTHORIZE_EXCHANGE.value,
    ActionKind.COMMERCE_DENY_EXCHANGE.value,
    ActionKind.COMMERCE_COMPLETE_EXCHANGE.value,
    ActionKind.COMMERCE_OPEN_DISPUTE.value,
    ActionKind.COMMERCE_SUBMIT_DISPUTE_EVIDENCE.value,
    ActionKind.COMMERCE_RESPOND_TO_DISPUTE.value,
    ActionKind.COMMERCE_REQUEST_LEDGER_RECONCILIATION.value,
    ActionKind.COMMERCE_READ_PAYMENT_HISTORY.value,
    ActionKind.COMMERCE_READ_LEDGER_HISTORY.value,
    ActionKind.COMMERCE_READ_PACKING_HISTORY.value,
    ActionKind.COMMERCE_READ_AFTER_SALES_HISTORY.value,
    ActionKind.COMMERCE_READ_AFTER_SALES_POLICY.value,
    # T8 merchant governance intents. Every action is a compact request to a
    # typed Platform endpoint; none grants direct World mutation authority.
    ActionKind.COMMERCE_PUBLISH_CAMPAIGN.value,
    ActionKind.COMMERCE_DISCLOSE_PLACEMENT.value,
    ActionKind.COMMERCE_ACTIVATE_CAMPAIGN.value,
    ActionKind.COMMERCE_REJECT_REVIEW_MANIPULATION.value,
    ActionKind.COMMERCE_REJECT_COORDINATION.value,
    ActionKind.COMMERCE_ACCEPT_REMEDIATION_PLAN.value,
    ActionKind.COMMERCE_COMPLETE_REMEDIATION_STEP.value,
    ActionKind.COMMERCE_READ_GOVERNANCE_HISTORY.value,
    # World reads — public / scoped-by-caller per WORLD_DESIGN §7. The
    # merchant only reads world; every write goes through platform.
    ActionKind.WORLD_READ_CATALOG.value,
    ActionKind.WORLD_READ_INVENTORY.value,
    ActionKind.WORLD_READ_POLICY.value,
    ActionKind.WORLD_READ_ORDER.value,
    ActionKind.WORLD_READ_REPUTATION.value,
    ActionKind.WORLD_READ_LEDGER.value,
    ActionKind.WORLD_READ_DISPUTE.value,
    ActionKind.WORLD_READ_RULING.value,
})


# --- Factory ---------------------------------------------------------

def _default_skills_root() -> Path:
    """Locate ``skills/merchant/`` relative to the repo, regardless of cwd.

    Layout assumption: this file lives at ``<repo>/src/agents/merchant.py``,
    so the skill root is ``<repo>/skills/merchant``.
    """
    return Path(__file__).resolve().parents[2] / "skills" / "merchant"


def make_merchant_agent(
    *,
    inputs: "AgentInputs",
    channel: "InferenceChannel",
    skill_loader: "SkillLoader | None" = None,
    memory: "Memory | None" = None,
    agent_id: str = "merchant",
    enabled_skill_names: tuple[str, ...] = DEFAULT_MERCHANT_SKILLS,
    inventory_view: "Any | None" = None,
    merchant_data: "Any | None" = None,
    selector: "Any | None" = None,
    actor_report_root_msg_ids: frozenset[str] = frozenset(),
    semantic_root_principals: "Mapping[str, str] | None" = None,
) -> "Agent":
    """Construct a merchant agent.

    Args:
        inputs: persona / policy. The policy slice carries ``floor_price``
            (private, lives in ``PRIVATE_UTILITY``) and any negotiation
            tuning knobs (``margin_target_bps``, ``max_negotiation_rounds``).
        skill_loader: pre-built loader; if ``None``, a default
            ``FilesystemSkillLoader`` rooted at ``<repo>/skills/merchant/``
            is constructed with ``enabled_skill_names``. Empty tuple →
            ``EmptySkillLoader`` (useful for early integration demos).
        memory: per-agent memory store; defaults to ``InMemoryStore()``.
        channel: typed business-decision transport used by the Agent.
        agent_id: bus address; ``"merchant"`` by default, ``"merchant:<n>"``
            for multi-merchant populations.
        enabled_skill_names: SKILL.md bundle names to enable; honored only
            when ``skill_loader`` is ``None``.

    Returns:
        An ``Agent`` whose ``enabled_skills`` is the loader's manifest tuple
        at spawn time. Full skill bodies are loaded on demand inside the
        turn that activates them.
    """
    if memory is None:
        memory = InMemoryStore()

    if skill_loader is None:
        if enabled_skill_names:
            skill_loader = FilesystemSkillLoader(
                roots=(_default_skills_root(),),
                enabled_names=enabled_skill_names,
            )
        else:
            skill_loader = EmptySkillLoader()

    return Agent(
        id=agent_id,
        skill_loader=skill_loader,
        enabled_skills=skill_loader.manifests(),
        memory=memory,
        inputs=inputs,
        channel=channel,
        inbound_action_kinds=MERCHANT_INBOUND,
        outbound_action_kinds=MERCHANT_OUTBOUND,
        inventory_view=inventory_view,
        merchant_data=merchant_data,
        selector=selector,
        actor_report_root_msg_ids=actor_report_root_msg_ids,
        semantic_root_principals=semantic_root_principals,
    )
