"""Buyer-side agent factory.

Per AGENT_CLASSES.md §2 / §8, the buyer is a single ``Agent("buyer")`` whose
behavior is composed from a set of enabled SKILL.md bundles loaded via
``SkillLoader``. No ``BuyerMainSkill`` Python class — skills are bundles on
disk, not subclasses.

The skill bundles live under ``skills/buyer/<name>/SKILL.md``; the factory
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
# Names match ``skills/buyer/<name>/SKILL.md``. Listed in journey order
# (parse → discover → confirm) for prompt readability.
DEFAULT_BUYER_SKILLS: tuple[str, ...] = (
    "mandate-parsing",
    "marketplace-message-safety",
    "discovery-search",
    "negotiation",
    "cart-checkout",
    "supply-fulfillment",
    "purchase-confirmation",
    "after-sales-lifecycle",
    "return-refund",
    "authenticated-review",
    "protocol-event-handling",
)


# --- VCP action surface for the buyer --------------------------------
#
# Names are VCP-namespaced strings (delegate.* / commerce.* / platform.*).
# Used for runtime partition checks and for shaping ``Agent.receive`` —
# this scaffold only stores the set; enforcement is a Router concern.

BUYER_INBOUND: frozenset[str] = frozenset({
    # principal → buyer
    "delegate.create_purchase_mandate",
    "delegate.approve_purchase",
    "delegate.reject_purchase",
    # platform → buyer
    "platform.rank_offers",
    "platform.create_match_certificate",
    "platform.settlement_receipt",
    "platform.fulfillment_allocation",
    "platform.lifecycle_updated",
    "platform.rule_dispute",
    ActionKind.PLATFORM_CATALOG_LISTING.value,
    ActionKind.PLATFORM_SUPPLY_STATE.value,
    ActionKind.PLATFORM_SHIPMENT_STATE.value,
    ActionKind.PLATFORM_SHIPMENT_RESOLVED.value,
    ActionKind.PLATFORM_DELIVER_PROTOCOL_EVENT.value,
    ActionKind.PLATFORM_PROTOCOL_EVENT_RECEIPT.value,
    ActionKind.PLATFORM_EVIDENCE_RECORD_PERSISTED.value,
    ActionKind.PLATFORM_AFTER_SALES_UPDATED.value,
    ActionKind.PLATFORM_AFTER_SALES_SNAPSHOT.value,
    ActionKind.PLATFORM_GOVERNANCE_UPDATED.value,
    ActionKind.PLATFORM_GOVERNANCE_SNAPSHOT.value,
    # merchant → buyer (negotiation lane + Branch B in purchase-confirmation)
    "commerce.accept_offer",
    "commerce.counter_offer",
    "commerce.reject_offer",
    "commerce.withdraw_offer",
    # Untrusted marketplace chat is still a real protocol input.  Declaring
    # it here lets adversarial-content scenarios exercise the buyer through
    # Runtime rather than injecting text directly into an inference prompt.
    ActionKind.COMMERCE_SEND_MESSAGE.value,
})

BUYER_OUTBOUND: frozenset[str] = frozenset({
    # buyer → platform aggregator
    "commerce.search",
    ActionKind.GET_SKU.value,
    # T5's evaluated buyer requests a World-authored quote and checks out only
    # by the returned quote id.  Both actions terminate at platform:checkout.
    ActionKind.COMMERCE_REQUEST_CART_QUOTE.value,
    ActionKind.PLATFORM_CHECKOUT_CART.value,
    # buyer → platform PSP (settlement)
    "platform.settle_payment",
    "platform.open_dispute",
    # buyer → merchant (negotiation + acceptance + walking away)
    "commerce.accept_offer",
    "commerce.reject_offer",
    "commerce.withdraw_offer",
    "commerce.propose_offer",
    "commerce.counter_offer",
    "commerce.cancel_order",
    "commerce.request_return",
    "commerce.request_exchange",
    ActionKind.COMMERCE_CANCEL_PAID_ORDER.value,
    ActionKind.COMMERCE_OPEN_REFUND_CASE.value,
    ActionKind.COMMERCE_OPEN_DISPUTE.value,
    ActionKind.COMMERCE_SUBMIT_DISPUTE_EVIDENCE.value,
    ActionKind.COMMERCE_RESPOND_TO_DISPUTE.value,
    ActionKind.COMMERCE_REQUEST_LEDGER_RECONCILIATION.value,
    ActionKind.COMMERCE_READ_PAYMENT_HISTORY.value,
    ActionKind.COMMERCE_READ_LEDGER_HISTORY.value,
    ActionKind.COMMERCE_READ_PACKING_HISTORY.value,
    ActionKind.COMMERCE_READ_AFTER_SALES_HISTORY.value,
    ActionKind.COMMERCE_READ_AFTER_SALES_POLICY.value,
    ActionKind.COMMERCE_READ_SUPPLY_STATE.value,
    ActionKind.COMMERCE_READ_SHIPMENT.value,
    ActionKind.COMMERCE_RESOLVE_SHIPMENT.value,
    ActionKind.COMMERCE_SEND_MESSAGE.value,
    ActionKind.DELEGATE_REPORT_RESULT.value,
    ActionKind.COMMERCE_SUBMIT_DECISION_RECORD.value,
    ActionKind.COMMERCE_ACKNOWLEDGE_PROTOCOL_EVENT.value,
    ActionKind.COMMERCE_REJECT_PROTOCOL_EVENT.value,
    ActionKind.COMMERCE_PROCESS_PROTOCOL_EVENT.value,
    ActionKind.COMMERCE_PUBLISH_EVIDENCE_RECORD.value,
    ActionKind.COMMERCE_SUBMIT_REVIEW.value,
    # buyer → principal (failure modes)
    "delegate.reject_purchase",
})


# --- Factory ---------------------------------------------------------

def _default_skills_root() -> Path:
    """Locate ``skills/buyer/`` relative to the repo, regardless of cwd.

    Layout assumption: this file lives at ``<repo>/src/agents/buyer.py``,
    so the skill root is ``<repo>/skills/buyer``.
    """
    return Path(__file__).resolve().parents[2] / "skills" / "buyer"


def make_buyer_agent(
    *,
    inputs: "AgentInputs",
    channel: "InferenceChannel",
    skill_loader: "SkillLoader | None" = None,
    memory: "Memory | None" = None,
    agent_id: str = "buyer",
    enabled_skill_names: tuple[str, ...] = DEFAULT_BUYER_SKILLS,
    selector: "Any | None" = None,
    actor_report_root_msg_ids: frozenset[str] = frozenset(),
    semantic_search_limit: int | None = None,
    semantic_root_principals: "Mapping[str, str] | None" = None,
) -> "Agent":
    """Construct the buyer agent.

    Args:
        inputs: persona / mandate / policy. The mandate slice is required for
            the buyer; ``success_oracle`` is never rendered into the prompt
            (it never lands in ``AgentInputs`` to begin with).
        skill_loader: pre-built loader; if ``None``, a default
            ``FilesystemSkillLoader`` rooted at ``<repo>/skills/buyer/`` is
            constructed with ``enabled_skill_names``. If the name tuple is
            empty, an ``EmptySkillLoader`` is used instead (useful for early
            integration demos before any SKILL.md exists).
        memory: per-agent memory store; defaults to ``InMemoryStore()``.
        channel: typed business-decision transport used by the Agent.
        agent_id: bus address; ``"buyer"`` by default. ``"buyer:<n>"`` is
            reserved for future multi-buyer populations.
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
        inbound_action_kinds=BUYER_INBOUND,
        outbound_action_kinds=BUYER_OUTBOUND,
        selector=selector,
        actor_report_root_msg_ids=actor_report_root_msg_ids,
        semantic_search_limit=semantic_search_limit,
        semantic_root_principals=semantic_root_principals,
    )
