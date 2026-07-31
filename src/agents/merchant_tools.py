"""Merchant-lane tool functions.

The runtime capabilities the merchant agent's SKILL.md bundles are allowed to
call (the ``allowed_tools`` surface in ``skills/merchant/*/SKILL.md``). Two
kinds of capability live here:

* **State accessors over the injected ``MerchantData`` store**:
  ``get_listing`` / ``read_offer_mandate`` / ``get_policy`` / ``get_order``
  / ``get_category_demand`` / ``get_competitors`` / … . These read the
  seller's own catalog, price, floor, inventory, policy, orders, and the
  market intelligence views derived from the merchant research dataset.
* **Composite envelope constructors** that mix state + decision logic:
  ``propose_markdown`` (validates against the listing's floor before
  rendering ``commerce.adjust_price``). ``cancel_order`` /
  ``adjudicate_return`` (status / window gating before rendering refund or
  decline). ``build_dispute_evidence`` renders the evidence brief from the
  order record. ``update_listing`` / ``record_order`` / ``receive_shipment``
  (lifecycle envelopes that touch ``MerchantData``).

**Pure helpers live one layer down** in :mod:`protocol.pricing` (counters,
band biases, markdown/markup steps, availability/reputation bands) and
:mod:`protocol.envelopes` (``build_grounded_offer``, ``build_inquiry_response``,
``assert_no_private_leak``). This module re-exports them so existing
``from agents.merchant_tools import compute_counter`` callers keep working
without change. New code should import from :mod:`protocol` directly so
the buyer / platform lanes can share the same math without a merchant
dependency.

Design rules:

* **Money is integer minor units everywhere** (VCP_SPEC.md §3). $3.50 == 350.
* These functions are **dependency-free and pure** over an injected
  ``MerchantData``. Excel I/O lives in ``merchant_data.py`` (lazy pandas) so
  this module stays testable without pandas/openpyxl.
* A "product" is a **category** (e.g. ``自动猫砂盆`` / automatic litter box).
  It is the unit the research dataset is keyed by and the unit the virtual seller
  lists on the platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-exports — pure helpers (math + envelope shape) now live in the
# protocol layer so every lane can reach for the same primitives. The
# imports below preserve the merchant-side public surface:
# ``from agents.merchant_tools import compute_counter`` etc. still resolve.
from protocol.envelopes import (  # noqa: F401  (re-export)
    ClaimNotPermitted,
    MandateLeak,
    assert_no_private_leak,
    build_grounded_offer,
    build_inquiry_response,
)
from protocol.pricing import (  # noqa: F401  (re-export)
    aged_stock_discount,
    apply_pricing_bias,
    apply_reputation_bias,
    availability_band,
    compute_concession_counter,
    compute_counter,
    compute_min_offer_step,
    compute_patience_loss,
    demand_step_markup,
)
from protocol.pricing import reputation_band as _reputation_band_primitives
from protocol.schemas import (  # noqa: F401  (re-export for skill-side callers)
    AdjustPricePayload,
    ReceiveShipmentPayload,
    UpdateListingFieldsPublish,
    UpdateListingPayload,
)
from agents.merchant_private_state import (  # noqa: F401  (re-export)
    PUBLIC_ATTRIBUTE_KEYS,
    MerchantPrivateState,
    split_attributes,
)

# B7 split: dataclasses + MerchantData moved to sibling modules so this
# file holds only the tool functions. Re-export the public surface so
# ``from agents.merchant_tools import MerchantData / ToolError / Policy``
# (and every other type) continues to work.
from agents.merchant_tools_types import (  # noqa: F401  (re-export)
    CompetitorInfo,
    DemandSignals,
    OfferMandateView,
    OpportunityGaps,
    Order,
    OrderNotFound,
    Policy,
    PriceBenchmark,
    Positioning,
    ProductNotFound,
    Reputation,
    ToolError,
    Visibility,
)
from agents.merchant_tools_state import (  # noqa: F401  (re-export)
    DEFAULT_CATEGORY_ALIASES,
    MerchantData,
)
# Private helpers still needed by tool functions below — pulled out of
# the state module via leading-underscore-aware names.
from agents.merchant_tools_state import _aliases_for, _need  # noqa: F401


# Dataclasses, MerchantData, and internal parsing helpers moved to
# sibling modules in B7. This file now holds only the tool functions.


# --- Tier 1: transactional -------------------------------------------

def get_my_catalog(data: MerchantData) -> list[str]:
    """Every product (category) the seller lists or has intel for, sorted."""
    return sorted(set(data.private_states) | set(data.demand) | set(data.visibility))


def get_listing(data: MerchantData, product: str) -> MerchantPrivateState:
    """Return the merchant's private state for ``product``.

    Historically returned a now-removed ``Listing`` dataclass that also
    held world-authoritative inventory + list_price snapshots; B5 split
    those off to :class:`InventoryView` and ``world.read_catalog``. The
    function name is preserved so call sites that just want the per-SKU
    handle don't have to rename; the returned shape is now
    :class:`MerchantPrivateState`, which carries floor / velocity / age /
    seed_list_price / seed_inventory.
    """
    return _need(data.private_states, product, "listing")


def get_inventory(data: MerchantData, product: str) -> int:
    """Return current inventory count for ``product``.

    Reads through the live :class:`InventoryView` when one is attached
    (authoritative — propagates PSP settles); otherwise returns the
    spawn-time ``seed_inventory`` snapshot as a backward-compat fallback
    for tests that don't construct a view.
    """
    if data.inventory_view is not None:
        return int(data.inventory_view.qty_available(product))
    return int(_need(data.private_states, product, "listing").seed_inventory)


def read_offer_mandate(data: MerchantData, product: str) -> OfferMandateView:
    return _need(data.mandates, product, "offer mandate")


def get_policy(data: MerchantData, product: str | None = None) -> Policy:
    return data.policy


def get_order(data: MerchantData, order_id: str) -> Order:
    if order_id not in data.orders:
        raise OrderNotFound(order_id)
    return data.orders[order_id]


# --- Tier 2: market intelligence -------------------------------------

def get_category_demand(data: MerchantData, product: str) -> DemandSignals:
    return _need(data.demand, product, "demand signals")


def get_price_benchmarks(data: MerchantData, product: str) -> PriceBenchmark:
    if product not in data.benchmarks:
        return PriceBenchmark(product=product, band_low=None, band_high=None)
    return data.benchmarks[product]


def get_competitors(data: MerchantData, product: str) -> list[CompetitorInfo]:
    return list(data.competitors.get(product, []))


def get_market_visibility(data: MerchantData, product: str) -> Visibility:
    return _need(data.visibility, product, "visibility")


def get_positioning_angles(data: MerchantData, product: str) -> Positioning:
    return data.positioning.get(product, Positioning(product=product))


def get_opportunity_gaps(data: MerchantData, product: str) -> OpportunityGaps:
    return data.gaps.get(product, OpportunityGaps(product=product))


# --- Pure helpers (compute_counter / build_grounded_offer / …) -------
#
# The pure math + envelope-shape helpers (``compute_counter``,
# ``compute_concession_counter``, ``compute_min_offer_step``,
# ``compute_patience_loss``, ``build_grounded_offer``,
# ``assert_no_private_leak``) used to live here. They now live in
# :mod:`protocol.pricing` and :mod:`protocol.envelopes` and are re-exported
# from the top of this module for backward compatibility. New code should
# import from :mod:`protocol` directly.


# --- Stock-management tools (aging-markdown / stockout-aware-pricing /
# --- restock-signal) ------------------------------------------------

@dataclass(frozen=True)
class InventoryStatus:
    """Combined inventory + velocity observation.

    ``days_to_stockout`` = ``count / velocity_per_day`` when velocity is
    known and positive; ``None`` otherwise (callers should treat unknown
    velocity conservatively — neither scarce nor comfortable).
    """
    product: str
    count: int
    inventory_age_days: int
    last_restock_at: str | None = None
    velocity_per_day: float | None = None
    days_to_stockout: float | None = None


def get_inventory_status(data: MerchantData, product: str) -> InventoryStatus:
    """The one observation tool the three stock-management skills share.

    Aging-markdown reads ``inventory_age_days`` (+ ``days_to_stockout`` to
    rule out scarcity); stockout-aware-pricing reads ``days_to_stockout``;
    restock-signal reads ``count``.

    Source-of-truth resolution (B5,
    the ACWorld merchant-data authority design):

    * ``count`` from ``data.inventory_view.qty_available(product)`` when
      a view is attached; otherwise from the spawn-time
      ``seed_inventory`` snapshot. The view is the authoritative path
      — it reads through to world inventory, so a settle that
      decremented qty_available is visible on the next call. The seed
      fallback exists only for tests that don't construct a view.

    * ``velocity_per_day`` / ``inventory_age_days`` / ``arrived_at`` from
      ``MerchantPrivateState`` — merchant-private bookkeeping that was
      never world-authoritative.
    """
    state = get_listing(data, product)
    if data.inventory_view is not None:
        count = int(data.inventory_view.qty_available(product))
    else:
        count = int(state.seed_inventory)
    v = state.velocity_per_day
    dts = float(count) / float(v) if (v is not None and v > 0) else None
    return InventoryStatus(
        product=product,
        count=count,
        inventory_age_days=int(state.inventory_age_days),
        last_restock_at=state.arrived_at,
        velocity_per_day=(float(v) if v is not None else None),
        days_to_stockout=dts,
    )


def propose_markdown(data: MerchantData, product: str,
                     new_list_price: int, *, sku_id: str = "S1") -> dict[str, Any]:
    """Construct a ``commerce.adjust_price`` action dict for a markdown.

    Validates at the boundary so the LLM cannot accidentally emit an unsafe
    adjust — the error messages are intentionally generic and never echo
    the floor or the violation distance.

    Raises:
        ToolError: ``new_list_price`` is non-integer, equal/above the
            current list price (not a markdown), or at/below the floor.
    """
    if int(new_list_price) != new_list_price:
        raise ToolError("new_list_price must be integer minor units")
    state = get_listing(data, product)
    # ``seed_list_price`` is a spawn-time snapshot, not authoritative —
    # the "is this actually a markdown?" check is a tool-side guardrail
    # against the LLM emitting an upward move through the markdown path.
    # If the world's current price has drifted from the seed, the worst
    # case is that the LLM mis-selects ``propose_markdown`` vs
    # ``propose_price_set``; CatalogPolicy ultimately validates the
    # resulting envelope against the world catalog.
    if new_list_price >= state.seed_list_price:
        raise ToolError("new_list_price must be below the current list price")
    if new_list_price <= state.floor_price:
        raise ToolError("new_list_price not permitted")
    payload: AdjustPricePayload = {
        "sku_id": sku_id, "list_price": int(new_list_price),
    }
    # ``product`` is a merchant-side label; the schema is sku_id-keyed,
    # but the platform tolerates the extra field. Keep it for skill-side
    # readability and replay diffing.
    return {
        "kind": "commerce.adjust_price",
        "payload": {**payload, "product": product},
    }


def propose_price_set(data: MerchantData, product: str,
                      new_list_price: int, *, sku_id: str = "S1",
                      reason: str | None = None) -> dict[str, Any]:
    """Construct a ``commerce.adjust_price`` action dict for an **arbitrary**
    list-price move — up *or* down.

    The generic counterpart to :func:`propose_markdown` (which is strictly
    a downward move) and to :func:`demand_step_markup` (which is a single
    bounded upward step). Use this when the merchant has chosen a new
    price from an external signal — peer pricing, owner directive,
    repositioning — that may be above or below the current list price.

    Validates at the boundary:

    * ``new_list_price`` is an integer minor unit.
    * ``new_list_price > floor_price`` (the private-utility invariant —
      never cross or disclose the floor).
    * ``new_list_price != lst.list_price`` is **not** enforced — a no-op
      adjust still goes through (idempotent). The skill body should
      short-circuit before calling if the value is unchanged.

    The optional ``reason`` is **not** placed on the wire — it is for
    agent-side audit / memory only. The world write only sees the new
    list_price.

    Raises:
        ToolError: ``non_integer_money`` or ``list_price_at_or_below_floor``.
    """
    if int(new_list_price) != new_list_price:
        raise ToolError("non_integer_money")
    state = get_listing(data, product)
    if new_list_price <= state.floor_price:
        # Generic decline reason — never echo the floor distance.
        raise ToolError("list_price_at_or_below_floor")
    payload: AdjustPricePayload = {
        "sku_id": sku_id, "list_price": int(new_list_price),
    }
    return {
        "kind": "commerce.adjust_price",
        "payload": {**payload, "product": product},
    }


def receive_shipment(data: MerchantData, product: str, qty: int,
                     arrived_at: str, *, lot_id: str | None = None,
                     sku_id: str = "S1") -> dict[str, Any]:
    """Construct a ``commerce.receive_shipment`` action dict.

    Mirror of ``propose_markdown`` in shape: pure constructor, no IO, no
    clock. The runtime is what actually flips ``Listing.inventory`` and
    refreshes ``arrived_at`` — this helper only builds the envelope
    payload the merchant emits.

    Validates at the boundary so the LLM cannot emit an unsafe restock:

    * ``qty`` must be a strictly positive integer (no floats, no zero,
      no negatives) — minor-units convention does not apply to qty, but
      the integer rule does.
    * ``product`` must resolve to an existing listing via
      ``get_listing``; raises ``KeyError`` otherwise (consistent with
      the other read helpers — the skill itself is responsible for
      translating that into a ``commerce.respond_inquiry`` decline with
      ``reason: "unknown_sku"``).

    Raises:
        ToolError: ``qty`` is not a positive integer.
    """
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ToolError("qty must be a positive integer")
    # Surface KeyError as ToolError so the agent boundary stays consistent
    # with propose_markdown's failure mode.
    get_listing(data, product)
    payload: ReceiveShipmentPayload = {
        "sku_id": sku_id, "product": product,
        "qty": int(qty), "arrived_at": str(arrived_at),
    }
    if lot_id:
        payload["lot_id"] = str(lot_id)
    return {"kind": "commerce.receive_shipment", "payload": payload}


def build_receive_shipment_envelope(
    *,
    sku_id: str,
    product: str,
    qty: int,
    arrived_at: str,
    lot_id: str | None = None,
) -> dict[str, Any]:
    """Construct a ``commerce.receive_shipment`` action dict — the merchant's
    restock intent addressed to ``platform:catalog``.

    The merchant lane **has no direct-to-world write privilege** for
    inventory rows (PARTITION_ALLOW enforces this; ``WorldService`` also
    rejects merchant-origin ``world.update_inventory`` envelopes as
    defense-in-depth). To restock a SKU the merchant emits this
    commerce-namespace envelope; the platform's :class:`CatalogPolicy`
    validates ownership and forwards as a ``world.update_inventory``
    write that increments ``qty_available`` by ``qty``.

    Pure constructor, no IO. The MerchantData-aware variant
    :func:`receive_shipment` (which also asserts the SKU exists in the
    merchant's local store) is the in-agent ergonomics; this builder is
    the bare wire shape for notebook / external use.

    Validates at the boundary:

    * ``qty`` strictly positive integer.
    * ``sku_id`` and ``product`` non-empty strings.

    Returns:
        ``{"kind": "commerce.receive_shipment", "payload": {...}}``

    Raises:
        ToolError: ``invalid_qty``, ``invalid_sku_id``, ``invalid_product``.
    """
    if not sku_id or not isinstance(sku_id, str):
        raise ToolError("invalid_sku_id")
    if not product or not isinstance(product, str):
        raise ToolError("invalid_product")
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ToolError("invalid_qty")
    payload: ReceiveShipmentPayload = {
        "sku_id": str(sku_id), "product": str(product),
        "qty": int(qty), "arrived_at": str(arrived_at),
    }
    if lot_id:
        payload["lot_id"] = str(lot_id)
    return {"kind": "commerce.receive_shipment", "payload": payload}


# return-adjudicate: reason codes the merchant honors at full vs partial vs deny.
# Pure data — the skill body is what owns the policy intent; this is the lookup
# the helper uses so the LLM cannot author a custom reason without warning.
_FULL_REFUND_REASONS: frozenset[str] = frozenset((
    "defective", "damaged_on_arrival", "wrong_item", "not_as_described",
))
_PARTIAL_REFUND_REASONS: frozenset[str] = frozenset((
    "changed_mind",   # consumer-friendly but condition-discounted if opened/used
))


def _days_between(iso_a: str, iso_b: str) -> int:
    """Whole-day delta b - a; both ISO ``YYYY-MM-DD`` (no time). Pure."""
    from datetime import date
    da = date.fromisoformat(str(iso_a)[:10])
    db = date.fromisoformat(str(iso_b)[:10])
    return (db - da).days


# Order statuses that are still pre-dispatch and therefore cancellable.
_CANCELLABLE_STATUSES: frozenset[str] = frozenset(("created", "paid", "authorized"))


def cancel_order(data: MerchantData, order_id: str, *,
                 reason: str = "buyer_requested",
                 sku_id: str = "S1") -> dict[str, Any]:
    """Adjudicate a pre-dispatch order cancellation.

    Returns exactly one action dict — either a ``commerce.issue_refund``
    (full refund + cancellation accepted) or a ``commerce.respond_inquiry``
    of ``category: "policy"`` (declined because the order is no longer
    cancellable: already shipped / delivered / cancelled).

    Decision tree (deterministic):

    1. Unknown order -> respond_inquiry decline ``"unknown_order"``.
    2. Already cancelled -> decline ``"already_cancelled"``.
    3. Already shipped or delivered -> decline
       ``"order_already_shipped"``. The buyer should use
       ``commerce.request_return`` after delivery (``return-adjudicate``
       handles that path).
    4. Status in ``_CANCELLABLE_STATUSES`` (``created``/``paid``/
       ``authorized``) -> approve full refund for ``order.amount`` with
       ``reason: "cancellation:<reason>"``. The PSP executes the ledger
       reverse; the merchant only requests it via the envelope.

    All amounts are integer minor units. Public decline reasons never
    contain margin, floor, or any private value.
    """
    def _decline(decline_reason: str) -> dict[str, Any]:
        return build_inquiry_response(
            product="", category="policy",
            answer={"decision": "declined",
                    "decline_reason": decline_reason,
                    "order_id": order_id},
            sku_id=sku_id,
        )
    if order_id not in data.orders:
        return _decline("unknown_order")
    order = data.orders[order_id]
    status = (order.status or "").lower()
    if status == "cancelled":
        return _decline("already_cancelled")
    if status in ("shipped", "delivered"):
        return _decline("order_already_shipped")
    if status not in _CANCELLABLE_STATUSES:
        # Unknown / out-of-band status — refuse rather than guess.
        return _decline("not_cancellable")
    return {"kind": "commerce.issue_refund",
            "payload": {"order_id": order_id,
                        "amount": int(order.amount),
                        "reason": f"cancellation:{reason}",
                        "product": order.product, "sku_id": sku_id}}


def record_order(data: MerchantData, order_id: str, *,
                 product: str, qty: int, amount: int,
                 buyer_id: str = "buyer",
                 status: str = "settled",
                 delivered_at: str | None = None) -> dict[str, Any]:
    """Persist an order into ``MerchantData.orders`` and return a structured
    ack the order-intake skill emits to the requesting buyer/platform.

    The merchant agent's TRANSACTION memory becomes the source of truth for
    its own orders — every downstream skill that needs to look up an order
    (``fulfillment``, ``return-adjudicate``, ``dispute-defense``) reads
    through this record. In the real runtime the row also lives in the
    World's ``orders`` table; here the agent-side mirror is what the skills
    operate against.

    Validation at the boundary:

    * ``qty`` strictly positive integer.
    * ``amount`` non-negative integer minor units (zero is allowed for
      free / gifted samples, even though they rarely settle).
    * ``order_id`` must not already exist — recording the same order twice
      is the merchant equivalent of a double-charge, so the boundary
      raises ``ToolError("duplicate_order")``. Idempotent handlers should
      check existence in memory before calling.

    Raises:
        ToolError: ``invalid_qty``, ``non_integer_money``, ``duplicate_order``.
    """
    if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
        raise ToolError("invalid_qty")
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        raise ToolError("non_integer_money")
    if order_id in data.orders:
        raise ToolError("duplicate_order")
    data.orders[order_id] = Order(
        order_id=order_id, product=product, qty=int(qty), amount=int(amount),
        status=str(status), delivered_at=delivered_at,
    )
    return build_inquiry_response(
        product=product, category="general",
        answer={
            "ack": "order_recorded",
            "order_id": order_id, "qty": int(qty), "amount": int(amount),
        },
    )


def build_dispute_evidence(data: MerchantData, order_id: str, dispute_id: str,
                            *, buyer_claim: str = "", refund_history: list[dict[str, Any]] | None = None,
                            claim_record: tuple[str, ...] = (),
                            sku_id: str = "S1") -> dict[str, Any]:
    """Construct a ``commerce.send_message`` envelope carrying a structured
    evidence brief, addressed to the platform adjudicator.

    Pure constructor. The skill body (``dispute-defense``) decides *when* to
    package evidence; this helper renders the brief into the exact payload
    the adjudicator parses.

    The brief is intentionally *factual* — order state, refund history (if
    any), the original ``permitted_claims`` the merchant made at offer time,
    and the public policy. It contains **no** private margin, floor, or
    urgency cue; the adjudicator works with the same surface the buyer saw
    plus the merchant's transactional record.

    Raises:
        ToolError: order does not exist (the merchant cannot defend a
            dispute against a SKU it never sold).
    """
    if order_id not in data.orders:
        raise ToolError("unknown_order")
    order = data.orders[order_id]
    pol = data.policy
    evidence: dict[str, Any] = {
        "evidence_brief": True,
        "dispute_id": str(dispute_id),
        "order_id": order_id,
        "order_summary": {
            "product": order.product,
            "qty": int(order.qty),
            "amount": int(order.amount),
            "status": order.status,
            "delivered_at": order.delivered_at,
        },
        "policy_summary": {
            "refund_policy": pol.refund_policy,
            "return_window_days": int(pol.return_window_days),
        },
        "claim_record": list(claim_record),
        "refund_history": list(refund_history or []),
        "buyer_claim": str(buyer_claim or ""),
    }
    return {"kind": "commerce.send_message",
            "payload": {"sku_id": sku_id, "product": order.product,
                        "to_role": "platform:adjudicator",
                        "evidence": evidence}}


def adjudicate_return(data: MerchantData, order_id: str, reason: str, *,
                      today: str, condition: str | None = None,
                      partial_pct: int = 50,
                      sku_id: str = "S1") -> dict[str, Any]:
    """Adjudicate a buyer return request against policy and stated reason.

    Returns exactly one action dict — either a ``commerce.issue_refund``
    (approved full or partial) or a ``commerce.respond_inquiry`` of
    ``category: "policy"`` (declined, with a public decline reason).

    Decision tree (deterministic):

    1. Unknown order → respond_inquiry decline ``"unknown_order"``.
    2. ``delivered_at`` missing → decline ``"not_delivered"``.
    3. ``today - delivered_at > return_window_days`` → decline
       ``"return_window_closed"``.
    4. Reason in ``_FULL_REFUND_REASONS`` → approve **full** refund for
       ``order.amount``.
    5. Reason in ``_PARTIAL_REFUND_REASONS``:
       * ``condition in {"new", None}`` → approve **full**.
       * ``condition in {"used", "opened"}`` → approve **partial**
         (``amount * partial_pct / 100``, integer math).
    6. Any other reason → decline ``"reason_not_covered"``.

    All amounts are integer minor units. Partial refunds round down.
    Public decline reasons NEVER include the floor, the order's margin,
    or any private value. The skill body and ``private-utility-guard``
    enforce that invariant before the emit.

    Raises:
        ToolError: on internally inconsistent input (``partial_pct``
            outside ``[0, 100]``, ``today`` not ISO).
    """
    if not isinstance(partial_pct, int) or not (0 <= partial_pct <= 100):
        raise ToolError("invalid_partial_pct")
    if not isinstance(today, str) or len(today) < 10:
        raise ToolError("invalid_today")
    def _decline(decline_reason: str, **extra: Any) -> dict[str, Any]:
        ans: dict[str, Any] = {"decision": "declined",
                                "decline_reason": decline_reason,
                                "order_id": order_id}
        ans.update(extra)
        return build_inquiry_response(
            product="", category="policy", answer=ans, sku_id=sku_id,
        )
    if order_id not in data.orders:
        return _decline("unknown_order")
    order = data.orders[order_id]
    if order.delivered_at is None:
        return _decline("not_delivered")
    days = _days_between(order.delivered_at, today)
    if days < 0:
        # delivered_at is in the future relative to today — clamp to 0 for the
        # window check (no decline) so test seeding doesn't depend on a clock.
        days = 0
    pol = data.policy
    window = int(pol.return_window_days)
    if window <= 0 or days > window:
        return _decline("return_window_closed")
    full_amount = int(order.amount)
    if reason in _FULL_REFUND_REASONS:
        amount = full_amount
    elif reason in _PARTIAL_REFUND_REASONS:
        if condition in (None, "new"):
            amount = full_amount
        elif condition in ("used", "opened"):
            amount = (full_amount * int(partial_pct)) // 100
        else:
            return _decline("reason_not_covered",
                            note=f"condition {condition!r} not recognized")
    else:
        return _decline("reason_not_covered")
    return {"kind": "commerce.issue_refund",
            "payload": {"order_id": order_id, "amount": int(amount),
                        "reason": reason,
                        "product": order.product, "sku_id": sku_id}}


def update_listing(data: MerchantData, op: str, product: str, *,
                   sku_id: str = "S1",
                   list_price: int | None = None,
                   attributes: dict[str, Any] | None = None,
                   permitted_claims: tuple[str, ...] | None = None,
                   must_not_claim: tuple[str, ...] | None = None,
                   inventory: int | None = None,
                   status: str | None = None,
                   op_id: str | None = None) -> dict[str, Any]:
    """Construct a ``commerce.update_listing`` action dict for catalog authoring.

    Mirror of ``propose_markdown`` / ``receive_shipment`` in shape: pure
    constructor, no IO, no clock. Platform authenticates the envelope sender;
    World applies ``world.apply_catalog_mutation`` and owns revisioning and
    idempotency. This helper only builds the compact actor payload.

    Three operations:

    * ``publish`` — author a new SKU. ``list_price``, ``attributes``,
      ``permitted_claims`` are required. The product must NOT already
      exist; raises ``ToolError("already_listed")`` if it does. The
      merchant's private ``floor_price`` lives only in ``MerchantData``
      and ``MemoryType.PRIVATE_UTILITY`` — it is **never** a parameter
      to this constructor and **never** ships in the payload.
    * ``update`` — edit attributes / claims / status on an existing listing.
      The ``list_price`` field is NOT permitted here — the markup/markdown
      skills own price moves. Raises ``ToolError("price_not_permitted_on_update")``
      if passed.
    * ``delist`` — set status to "delisted". Only ``product`` is needed.

    Common validation:

    * ``list_price``, when present, must be integer minor units.
    * Product existence is checked at the boundary so the LLM can't author
      a duplicate publish or update a non-existent SKU.

    Raises:
        ToolError: any boundary check fails (unknown_op, already_listed,
            unknown_sku, non_integer_money, price_not_permitted_on_update,
            missing_required_field).
    """
    if op not in ("publish", "update", "delist"):
        raise ToolError("unknown_op")
    exists = product in data.private_states
    if op == "publish":
        if exists:
            raise ToolError("already_listed")
        if list_price is None:
            raise ToolError("missing_required_field:list_price")
        if not isinstance(list_price, int) or isinstance(list_price, bool):
            raise ToolError("non_integer_money")
        if attributes is None or permitted_claims is None:
            raise ToolError("missing_required_field:attributes_or_claims")
        publish_fields: UpdateListingFieldsPublish = {
            "list_price": int(list_price),
            "attributes": dict(attributes),
            "permitted_claims": list(permitted_claims),
            "must_not_claim": list(must_not_claim or ()),
            "inventory": int(inventory if inventory is not None else 0),
            "status": status or "active",
        }
        fields: dict[str, Any] = dict(publish_fields)
    elif op == "update":
        if not exists:
            raise ToolError("unknown_sku")
        if list_price is not None:
            raise ToolError("price_not_permitted_on_update")
        fields = {}
        if attributes is not None:
            fields["attributes"] = dict(attributes)
        if permitted_claims is not None:
            fields["permitted_claims"] = list(permitted_claims)
        if must_not_claim is not None:
            fields["must_not_claim"] = list(must_not_claim)
        if status is not None:
            fields["status"] = str(status)
        if not fields:
            raise ToolError("missing_required_field:fields")
    else:  # delist
        if not exists:
            raise ToolError("unknown_sku")
        fields = {"status": "delisted"}
    payload: dict[str, Any] = {"op": op, "sku_id": sku_id, "product": product,
                                "fields": fields}
    if op_id:
        payload["op_id"] = str(op_id)
    return {"kind": "commerce.update_listing", "payload": payload}


def _claim_to_attribute_key(claim: str) -> str:
    """Normalize a claim string into a candidate attribute key.

    ``"self-cleaning"`` -> ``"self_cleaning"``,
    ``"app control"``  -> ``"app_control"``,
    ``"Large Capacity"`` -> ``"large_capacity"``.
    """
    return str(claim).strip().lower().replace("-", "_").replace(" ", "_")


def filter_truthful_claims(
    permitted_claims: tuple[str, ...] | list[str],
    attributes: dict[str, Any],
    *,
    alias_map: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the subset of ``permitted_claims`` truthfully grounded in ``attributes``.

    A claim is "grounded" when its normalized key (or its alias) maps to a
    truthy value in ``attributes``. Numeric attributes count as truthy when
    non-zero; lists count as truthy when non-empty.

    ``permitted_claims`` is the **legal envelope** (the mandate's
    permitted set); ``filter_truthful_claims`` returns the subset that
    the listing's attribute facts actually back. The merchant must
    never claim something the filter excluded — that is the
    truthfulness contract this skill enforces.

    The alias map handles claims whose grounding attribute uses a
    different name (``"app control"`` may be grounded by an underlying
    ``"wifi"`` attribute, for example).

    Pure. Returns claims in their original order.
    """
    out: list[str] = []
    aliases = {str(k).lower(): str(v) for k, v in (alias_map or {}).items()}
    for c in permitted_claims:
        key = aliases.get(str(c).lower()) or _claim_to_attribute_key(c)
        val = attributes.get(key)
        if isinstance(val, bool):
            ok = val
        elif isinstance(val, (int, float)):
            ok = val != 0
        elif isinstance(val, (list, tuple, set, dict, str)):
            ok = bool(val)
        else:
            ok = val is not None
        if ok:
            out.append(c)
    return tuple(out)


def get_reputation(data: MerchantData) -> Reputation:
    """Tier 1 — merchant's own platform reputation snapshot.

    The merchant never writes reputation (platform monopoly per VCP_SPEC §6);
    it only reads. ``reputation-aware-pricing`` uses this on offer turns to
    bias the negotiation band; trust-related downstream skills may read it
    for other reasons.
    """
    return data.reputation


def reputation_band(rep: Reputation, *,
                    high_rating: float = 4.5, high_reviews: int = 50,
                    low_rating: float = 3.5, low_reviews: int = 10) -> str:
    """Merchant-facing wrapper around :func:`protocol.pricing.reputation_band`.

    Accepts the merchant's :class:`Reputation` dataclass directly (the
    protocol-layer version takes ``(avg_rating, review_count)`` primitives
    so it stays lane-agnostic). The qualitative bands and thresholds are
    identical — see the protocol docstring for the band semantics.
    """
    return _reputation_band_primitives(
        rep.avg_rating, rep.review_count,
        high_rating=high_rating, high_reviews=high_reviews,
        low_rating=low_rating, low_reviews=low_reviews,
    )
