"""Buyer-lane skill selector.

The selector covers both the ordinary discovery and purchase journey and
the reusable operational protocols exposed by CommerceWorld.  Gates match
typed inbound actions and public payload shapes.  They do not depend on a
scenario identifier, benchmark family, or model-specific behavior.

Activation ordering:

* On a fresh ``delegate.create_purchase_mandate``: mandate-parsing must
  run BEFORE discovery-search (the latter reads PREFERENCE.goal that the
  former writes).
* On a ``platform.rank_offers``: discovery-search owns Path B; on its
  "negotiable_candidates" branch, the LLM is expected to follow with
  ``load_skill: negotiation`` — selector includes both so the order is
  available.
* On a ``commerce.counter_offer`` / ``commerce.accept_offer`` /
  ``commerce.reject_offer`` from the merchant: negotiation handles the
  reply, and on accept it may chain into purchase-confirmation Branch B.
* On a ``platform.create_match_certificate`` / ``platform.settlement_receipt``
  / ``delegate.{approve,reject}_purchase``: purchase-confirmation owns
  the rest of the journey.
* Cart, supply, after-sales, marketplace-message, and protocol-event
  envelopes activate their matching operational skill.  A purchase mandate
  can also activate cart or supply handling when it contains the matching
  public workflow shape.
* ``private-utility-guard`` isn't a separate buyer skill today; the
  buyer's privacy discipline lives in each skill's body. If a guard
  skill is added later, it joins the tail of every activated tuple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.skill_selector import _SelectorContext, run_gates

if TYPE_CHECKING:
    from agents.interfaces import Memory
    from protocol.envelope import Envelope


# --- gate predicates -------------------------------------------------


def _payload(ctx: _SelectorContext) -> dict[str, Any]:
    """Return the inbound payload as a plain mapping."""

    action = ctx.env.action
    raw = action.get("payload") if isinstance(action, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _contains_mapping_shape(value: Any, required: frozenset[str], *, depth: int = 3) -> bool:
    """Return whether a bounded public payload subtree has ``required`` keys.

    Purchase mandates may wrap an operational contract in an application
    chosen field.  Shape matching keeps the selector reusable without naming
    that wrapper or coupling activation to a scenario id.
    """

    if depth < 0 or not isinstance(value, dict):
        return False
    if required.issubset(value):
        return True
    return any(
        _contains_mapping_shape(item, required, depth=depth - 1)
        for item in value.values()
        if isinstance(item, dict)
    )


_AFTER_SALES_MESSAGE_KEYS = frozenset(
    {
        "order_id",
        "order_ids",
        "request_id",
        "authorization_id",
        "case_id",
        "dispute_id",
        "event",
    }
)


def _looks_like_after_sales_message(payload: dict[str, Any]) -> bool:
    return bool(_AFTER_SALES_MESSAGE_KEYS.intersection(payload))


def _message_mentions(ctx: _SelectorContext, terms: frozenset[str]) -> bool:
    """Match only public instruction-like text in a marketplace message."""

    if ctx.env.action.get("kind") != "commerce.send_message":
        return False
    fragments: list[str] = []

    def visit(value: Any, *, key: str = "", depth: int = 3) -> None:
        if depth < 0:
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, key=str(child_key).casefold(), depth=depth - 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key=key, depth=depth - 1)
        elif isinstance(value, str) and key in {
            "goal",
            "instruction",
            "operation",
            "reason",
            "summary",
        }:
            fragments.append(value.casefold())

    visit(_payload(ctx))
    text = " ".join(fragments)
    return any(term in text for term in terms)

def _mandate_parsing_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activation predicate for skills/buyer/mandate-parsing/SKILL.md."""
    kind = ctx.env.action.get("kind", "")
    if kind == "delegate.create_purchase_mandate":
        return True, "inbound is a fresh purchase mandate"
    return False, f"not a mandate inbound (got {kind!r})"


def _discovery_search_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activation predicate for skills/buyer/discovery-search/SKILL.md.

    Two paths: outbound search (kicked off after mandate-parsing writes
    the goal) and inbound rank_offers reply. We activate on either —
    the LLM picks the path based on context.
    """
    kind = ctx.env.action.get("kind", "")
    if kind == "delegate.create_purchase_mandate":
        return True, "fresh mandate → search to follow mandate-parsing"
    if kind == "platform.rank_offers":
        return True, "rank_offers reply → run Path B filter + commit"
    if kind == "commerce.send_message" and not _looks_like_after_sales_message(
        _payload(ctx)
    ):
        return True, "market message may signal an authorized discovery handoff"
    return False, f"not a discovery trigger (got {kind!r})"


def _negotiation_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activation predicate for skills/buyer/negotiation/SKILL.md.

    Three triggers: an inbound from the merchant in the offer lane;
    OR the buyer is initiating a counter (i.e. discovery-search wrote
    ``last_discovery_outcome=negotiable``); OR an inbound rank_offers
    (in case Path B determines no affordable candidate exists and routes
    here). We activate on the inbound kinds AND on rank_offers; the
    skill body's preconditions decide whether to actually act.
    """
    from agents.types import MemoryType  # local import to avoid cycle
    kind = ctx.env.action.get("kind", "")
    if kind in ("commerce.counter_offer", "commerce.accept_offer",
                "commerce.reject_offer"):
        return True, f"merchant-side negotiation envelope ({kind})"
    if kind == "platform.rank_offers":
        # Path B might route here via "negotiable_candidates" memory.
        # Pre-activate so the LLM has the option without a re-load.
        return True, "rank_offers may trigger negotiation handoff"
    # Buyer-initiated next-round counter (this turn's inbound is a
    # merchant counter_offer; covered by the first branch above).
    outcome = ctx.memory.read(MemoryType.TRANSACTION, "last_discovery_outcome")
    if outcome == "negotiable":
        return True, "discovery-search wrote negotiable_candidates"
    return False, f"no negotiation trigger (kind={kind!r})"


def _purchase_confirmation_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activation predicate for skills/buyer/purchase-confirmation/SKILL.md.

    Four trigger kinds plus the post-accept chain from negotiation.
    """
    kind = ctx.env.action.get("kind", "")
    if kind in (
        "platform.create_match_certificate",
        "platform.settlement_receipt",
        "platform.fulfillment_allocation",
        "platform.settle_payment_declined",   # future graceful PSP decline
        "delegate.approve_purchase",
        "delegate.reject_purchase",
    ):
        return True, f"purchase-confirmation handles {kind}"
    if kind == "commerce.accept_offer":
        # Merchant accepted the buyer's offer (Branch B trigger).
        return True, "merchant accept_offer → purchase-confirmation Branch B"
    return False, f"not a confirmation trigger (got {kind!r})"


def _return_refund_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activation predicate for skills/buyer/return-refund/SKILL.md (s5).

    Two triggers: the settlement receipt WHEN the mandate asked for a
    post-purchase return (``TRANSACTION.return_after_purchase``, written by
    mandate-parsing) — then the buyer emits ``commerce.request_return``; and the
    ``commerce.issue_refund`` that closes the loop. Without the mandate flag this
    never fires, so ordinary buys are unaffected.
    """
    from agents.types import MemoryType  # local import to avoid cycle
    kind = ctx.env.action.get("kind", "")
    if kind == "commerce.issue_refund":
        return True, "refund issued → return-refund closes the loop"
    if kind == "platform.settlement_receipt" and ctx.memory.read(
        MemoryType.TRANSACTION, "return_after_purchase"
    ):
        return True, "settled + mandate wants a post-purchase return"
    return False, f"no return trigger (kind={kind!r})"


def _cart_checkout_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activate the persistent multi-line cart quote and checkout skill."""

    kind = ctx.env.action.get("kind", "")
    if kind in {"platform.cart_quote", "platform.cart_settlement"}:
        return True, f"cart lifecycle reply ({kind})"
    payload = _payload(ctx)
    if kind == "delegate.create_purchase_mandate":
        if _contains_mapping_shape(payload, frozenset({"market_id", "lines"})):
            return True, "mandate contains a specified multi-line market cart request"
        if _contains_mapping_shape(
            payload,
            frozenset({"market_id", "mandate_id", "fill_policy", "backorder_policy"}),
        ):
            return True, "mandate contains cart-planning quote authority"
    return False, f"no cart workflow trigger (kind={kind!r})"


def _supply_fulfillment_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activate buyer supply-state, allocation, and shipment handling."""

    kind = ctx.env.action.get("kind", "")
    if kind in {
        "platform.supply_state",
        "platform.shipment_state",
        "platform.shipment_resolved",
        "platform.fulfillment_allocation",
    }:
        return True, f"supply or fulfillment reply ({kind})"
    payload = _payload(ctx)
    if kind == "delegate.create_purchase_mandate" and (
        _contains_mapping_shape(payload, frozenset({"sku_ids", "requested_qty"}))
        or _contains_mapping_shape(payload, frozenset({"shipment_id", "authority"}))
    ):
        return True, "mandate contains a supply or shipment workflow"
    return False, f"no supply or fulfillment trigger (kind={kind!r})"


def _after_sales_lifecycle_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activate authoritative cancel, return, exchange, and dispute handling."""

    kind = ctx.env.action.get("kind", "")
    if kind in {
        "platform.after_sales_updated",
        "platform.after_sales_snapshot",
        "platform.lifecycle_updated",
        "platform.rule_dispute",
        "platform.shipment_state",
    }:
        return True, f"after-sales platform reply ({kind})"
    if kind == "commerce.send_message" and _looks_like_after_sales_message(
        _payload(ctx)
    ):
        return True, "typed counterpart message carries after-sales references"
    return False, f"no after-sales trigger (kind={kind!r})"


def _marketplace_message_safety_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activate the untrusted-content boundary for every market message."""

    kind = ctx.env.action.get("kind", "")
    if kind == "commerce.send_message":
        return True, "all marketplace message content is untrusted input"
    return False, f"not a marketplace message (kind={kind!r})"


def _protocol_event_handling_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activate persistent callback validation and receipt handling."""

    kind = ctx.env.action.get("kind", "")
    if kind in {
        "platform.deliver_protocol_event",
        "platform.protocol_event_receipt",
    }:
        return True, f"protocol event lifecycle envelope ({kind})"
    return False, f"not a protocol event envelope (kind={kind!r})"


def _authenticated_review_gate(ctx: _SelectorContext) -> tuple[bool, str]:
    """Activate verified-purchase review submission and acknowledgement."""

    kind = ctx.env.action.get("kind", "")
    if kind in {"platform.governance_updated", "platform.governance_snapshot"}:
        payload = _payload(ctx)
        if payload.get("operation") in {"submit_review", "aggregate_reviews"}:
            return True, "Platform returned an authenticated review projection"
    if _message_mentions(ctx, frozenset({"authenticated review", "verified review"})):
        return True, "authorized message requests a verified-purchase review"
    return False, f"not an authenticated review workflow (kind={kind!r})"


# --- registry --------------------------------------------------------

_BUYER_GATES = {
    "mandate-parsing": _mandate_parsing_gate,
    "marketplace-message-safety": _marketplace_message_safety_gate,
    "discovery-search": _discovery_search_gate,
    "negotiation": _negotiation_gate,
    "cart-checkout": _cart_checkout_gate,
    "supply-fulfillment": _supply_fulfillment_gate,
    "purchase-confirmation": _purchase_confirmation_gate,
    "after-sales-lifecycle": _after_sales_lifecycle_gate,
    "return-refund": _return_refund_gate,
    "authenticated-review": _authenticated_review_gate,
    "protocol-event-handling": _protocol_event_handling_gate,
}


# --- ordering --------------------------------------------------------
#
# Buyer skills today have a clean dependency chain. If multiple fire on
# the same envelope, they run in this canonical order. Skills not present
# in the activated set are simply skipped; the order is total over the
# universe of buyer skills.
_BUYER_ORDER = (
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


def _order(activated: list[str]) -> tuple[str, ...]:
    """Return activated skills in canonical execution order."""
    active = set(activated)
    return tuple(name for name in _BUYER_ORDER if name in active)


# --- selector --------------------------------------------------------

class BuyerSkillSelector:
    """Concrete ``SkillSelector`` for buyer-lane agents."""

    def __init__(self, *, strict: bool = False) -> None:
        self._strict = strict

    def select(
        self,
        env: "Envelope",
        *,
        memory: "Memory",
        merchant_data: Any | None = None,
        inventory_view: Any | None = None,
    ) -> tuple[str, ...]:
        ctx = _SelectorContext(
            env=env,
            memory=memory,
            merchant_data=merchant_data,
            inventory_view=inventory_view,
        )
        fired = run_gates(
            ctx,
            _BUYER_GATES,
            lane="buyer",
            strict=self._strict,
        )
        return _order(fired)


__all__ = ["BuyerSkillSelector"]
