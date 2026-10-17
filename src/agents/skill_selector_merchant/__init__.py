"""Merchant-lane skill selector.

Registry of merchant gates + pairwise defer rules + execution
ordering + the ``MerchantSkillSelector`` class.

Selectors use the public ACWorld skill registry and deterministic predicates.

The selector composes three deterministic stages:

1. **Run gates** — every registered gate fires; collect names whose gate
   returned True.
2. **Apply defer rules** — pairwise "if A and B both fired, drop B"
   transforms. Today's rules: scarcity dominates reputation; aged stock
   dominates demand-driven markup; peer-pricing defers to all three
   list-price movers.
3. **Order** — sort by canonical execution order (pre-steps before main
   steps; structural guards last).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.skill_selector import _SelectorContext, run_gates

# Per-gate imports — one per merchant skill, kept alphabetical for
# readability. Each module exposes a single ``gate`` callable.
from agents.skill_selector_merchant.aging_markdown import gate as _aging_markdown
from agents.skill_selector_merchant.after_sales_lifecycle import (
    gate as _after_sales_lifecycle,
)
from agents.skill_selector_merchant.cart_quote_handle import gate as _cart_quote_handle
from agents.skill_selector_merchant.catalog_serve import gate as _catalog_serve
from agents.skill_selector_merchant.claim_truthfulness import gate as _claim_truthfulness
from agents.skill_selector_merchant.demand_driven_markup import gate as _demand_driven_markup
from agents.skill_selector_merchant.dispute_defense import gate as _dispute_defense
from agents.skill_selector_merchant.fulfillment import gate as _fulfillment
from agents.skill_selector_merchant.inbound_restock import gate as _inbound_restock
from agents.skill_selector_merchant.inquiry_handle import gate as _inquiry_handle
from agents.skill_selector_merchant.listing_publish import gate as _listing_publish
from agents.skill_selector_merchant.listing_claim_manage import gate as _listing_claim_manage
from agents.skill_selector_merchant.market_governance import gate as _market_governance
from agents.skill_selector_merchant.order_cancel import gate as _order_cancel
from agents.skill_selector_merchant.order_intake import gate as _order_intake
from agents.skill_selector_merchant.peer_pricing import gate as _peer_pricing
from agents.skill_selector_merchant.price_discovery import gate as _price_discovery
from agents.skill_selector_merchant.pricing_negotiate import gate as _pricing_negotiate
from agents.skill_selector_merchant.private_utility_guard import gate as _private_utility_guard
from agents.skill_selector_merchant.protocol_event_handle import gate as _protocol_event_handle
from agents.skill_selector_merchant.reputation_aware_pricing import gate as _reputation_aware_pricing
from agents.skill_selector_merchant.restock_signal import gate as _restock_signal
from agents.skill_selector_merchant.return_adjudicate import gate as _return_adjudicate
from agents.skill_selector_merchant.stockout_aware_pricing import gate as _stockout_aware_pricing
from agents.skill_selector_merchant.supply_logistics import gate as _supply_logistics

if TYPE_CHECKING:
    from agents.interfaces import Memory
    from protocol.envelope import Envelope


# --- registry --------------------------------------------------------

_MERCHANT_GATES = {
    "aging-markdown": _aging_markdown,
    "after-sales-lifecycle": _after_sales_lifecycle,
    "cart-quote-handle": _cart_quote_handle,
    "catalog-serve": _catalog_serve,
    "claim-truthfulness": _claim_truthfulness,
    "demand-driven-markup": _demand_driven_markup,
    "dispute-defense": _dispute_defense,
    "fulfillment": _fulfillment,
    "inbound-restock": _inbound_restock,
    "inquiry-handle": _inquiry_handle,
    "listing-publish": _listing_publish,
    "listing-claim-manage": _listing_claim_manage,
    "market-governance": _market_governance,
    "order-cancel": _order_cancel,
    "order-intake": _order_intake,
    "peer-pricing": _peer_pricing,
    "price-discovery": _price_discovery,
    "pricing-negotiate": _pricing_negotiate,
    "private-utility-guard": _private_utility_guard,
    "protocol-event-handle": _protocol_event_handle,
    "reputation-aware-pricing": _reputation_aware_pricing,
    "restock-signal": _restock_signal,
    "return-adjudicate": _return_adjudicate,
    "stockout-aware-pricing": _stockout_aware_pricing,
    "supply-logistics": _supply_logistics,
}


# --- defer rules -----------------------------------------------------
#
# Pairwise: when both keys' gates fired, drop the value. Order doesn't
# matter (rules apply transitively in a fixed iteration order). Each
# rule corresponds to a real "if both apply, which wins" question;
# rules whose pairs are already mutually exclusive by gate predicates
# are NOT listed here (they would be dead code — the selector would
# never see both fire to begin with).
_DEFER_RULES: list[tuple[str, str]] = [
    # Scarcity dominates reputation when both apply
    # (per reputation-aware-pricing's "no scarcity bias already" rule).
    # Both fire on an offer turn; gates do NOT mutually exclude.
    ("stockout-aware-pricing", "reputation-aware-pricing"),
    # Peer-pricing defers to the idle-turn list-price movers when they
    # also fire. aging-markdown and demand-driven-markup don't constrain
    # peer-pricing's "merchant has active listings" gate, so co-fire is
    # real; this rule resolves it.
    ("aging-markdown", "peer-pricing"),
    ("demand-driven-markup", "peer-pricing"),
]

# Pairs intentionally NOT listed above (mutually exclusive by gate logic;
# documented here so a future contributor doesn't "fix" a missing rule):
#
# * (aging-markdown, demand-driven-markup) — aging requires
#   inventory_age_days > policy.markdown_rules.after_days; demand-driven
#   requires inventory_age_days <= policy.markdown_rules.after_days. The
#   strict-vs-non-strict comparison means at most one fires per SKU.
# * (stockout-aware-pricing, peer-pricing) — stockout requires a
#   decision-offer turn (is_decision_offer_turn); peer-pricing requires
#   an idle turn (is_idle_turn). Offer turns are in _REACTIVE_KINDS, so
#   is_idle_turn returns False for them. No overlap possible.


def _apply_defer_rules(fired: list[str]) -> list[str]:
    active = set(fired)
    for dominant, dropped in _DEFER_RULES:
        if dominant in active and dropped in active:
            active.discard(dropped)
    return [name for name in fired if name in active]


# --- ordering --------------------------------------------------------
#
# Canonical execution order across the universe of merchant skills.
# Activated skills appear in this order in the output tuple; non-activated
# ones are skipped. Order matters because pre-steps (stockout / reputation
# / claim-truthfulness) write memory the "main" skills read.
_MERCHANT_ORDER = (
    # Interpretation (writes target_band before pricing-negotiate reads)
    "price-discovery",
    # Pre-steps for offer turns (write bias to memory)
    "stockout-aware-pricing",
    "reputation-aware-pricing",
    "claim-truthfulness",
    # Main offer-turn skill
    "pricing-negotiate",
    # Idle-turn list-price movers (only one of these can fire per turn
    # post-defer, but listed in their natural priority)
    "aging-markdown",
    "demand-driven-markup",
    "peer-pricing",
    # Discovery / catalog
    "catalog-serve",
    "inquiry-handle",
    "listing-publish",
    # Operations / lifecycle
    "order-intake",
    "order-cancel",
    "cart-quote-handle",
    "fulfillment",
    "supply-logistics",
    "return-adjudicate",
    "after-sales-lifecycle",
    "dispute-defense",
    "inbound-restock",
    "restock-signal",
    # Catalog evidence, governance, and durable event continuations
    "listing-claim-manage",
    "market-governance",
    "protocol-event-handle",
    # Structural guards (last — pre-emit check on every outbound)
    "private-utility-guard",
)


def _order(activated: list[str]) -> tuple[str, ...]:
    """Return activated skills in canonical execution order."""
    active = set(activated)
    return tuple(name for name in _MERCHANT_ORDER if name in active)


# --- selector --------------------------------------------------------

class MerchantSkillSelector:
    """Concrete ``SkillSelector`` for merchant-lane agents."""

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
            _MERCHANT_GATES,
            lane="merchant",
            strict=self._strict,
        )
        deferred = _apply_defer_rules(fired)
        return _order(deferred)


__all__ = ["MerchantSkillSelector"]
