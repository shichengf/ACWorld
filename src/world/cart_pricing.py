"""World-owned intents and deterministic pricing for persistent cart checkout.

Agents submit compact policy and quote intents.  They never submit merchant
authority, revision numbers, logical ticks, digests, sealed quotes, or checkout
rows.  The World supplies those facts from its catalog, inventory, mandate,
policy, and clock tables before calling the pure builders in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from protocol.cart_quote_state import (
    BackorderPolicy,
    FillPolicy,
    PersistentCartQuote,
    build_cart_quote_charge,
    build_cart_quote_line,
    build_persistent_cart_quote,
)
from protocol.errors import SchemaError
from protocol.matching import MatchValidationError, canonical_digest
from protocol.pricing_policy import (
    BundleDiscount,
    BundleStacking,
    PricingComponent,
    PricingPolicyRevision,
    QuantityTier,
    build_bundle_condition,
    build_bundle_discount,
    build_pricing_component,
    build_quantity_tier,
)
from world.errors import OutOfStock
from world.types import InventoryRow, Listing


class PricingPolicyIntent(TypedDict):
    market_id: str
    policy_id: str
    listing_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    quantity_tiers: tuple[QuantityTier, ...]
    bundle_discounts: tuple[BundleDiscount, ...]
    bundle_stacking: BundleStacking
    components: tuple[PricingComponent, ...]
    effective_after_ticks: int
    expires_after_ticks: int | None


class CartLineIntent(TypedDict):
    sku_id: str
    qty: int


class CartQuoteIntent(TypedDict):
    mandate_id: str
    lines: tuple[CartLineIntent, ...]
    fill_policy: FillPolicy
    backorder_policy: BackorderPolicy


_POLICY_FIELDS = frozenset(
    {
        "market_id",
        "policy_id",
        "listing_ids",
        "product_ids",
        "quantity_tiers",
        "bundle_discounts",
        "bundle_stacking",
        "components",
        "effective_after_ticks",
        "expires_after_ticks",
    }
)
_POLICY_REQUIRED = frozenset(
    {"market_id", "policy_id", "listing_ids", "product_ids", "quantity_tiers"}
)
_QUOTE_FIELDS = frozenset(
    {"mandate_id", "lines", "fill_policy", "backorder_policy"}
)
_QUOTE_REQUIRED = frozenset({"mandate_id", "lines"})


def normalize_pricing_policy_intent(value: Any) -> PricingPolicyIntent:
    row = _mapping(value, "pricing policy intent")
    _exact_fields(row, _POLICY_FIELDS, _POLICY_REQUIRED, "pricing policy intent")
    listing_ids = _canonical_texts(row.get("listing_ids"), "listing_ids")
    product_ids = _canonical_texts(row.get("product_ids"), "product_ids")
    if not listing_ids and not product_ids:
        raise SchemaError("pricing policy requires a listing_id or product_id")
    tiers = tuple(
        _quantity_tier(item, index)
        for index, item in enumerate(_array(row.get("quantity_tiers"), "quantity_tiers"))
    )
    bundles = tuple(
        sorted(
            (
                _bundle_discount(item, index)
                for index, item in enumerate(
                    _array(row.get("bundle_discounts", []), "bundle_discounts")
                )
            ),
            key=lambda item: item.bundle_id,
        )
    )
    components = tuple(
        sorted(
            (
                _component(item, index)
                for index, item in enumerate(
                    _array(row.get("components", []), "components")
                )
            ),
            key=lambda item: (item.kind, item.component_id),
        )
    )
    stacking = row.get("bundle_stacking", "best_only")
    if stacking not in {"best_only", "cumulative"}:
        raise SchemaError("bundle_stacking must be best_only or cumulative")
    effective_after = _integer(
        row.get("effective_after_ticks", 0),
        "effective_after_ticks",
        minimum=0,
    )
    raw_expiry = row.get("expires_after_ticks")
    expires_after = (
        None
        if raw_expiry is None
        else _integer(raw_expiry, "expires_after_ticks", minimum=1)
    )
    if expires_after is not None and expires_after <= effective_after:
        raise SchemaError("policy expiry must follow its effective tick")
    normalized = PricingPolicyIntent(
        market_id=_text(row.get("market_id"), "market_id"),
        policy_id=_text(row.get("policy_id"), "policy_id"),
        listing_ids=listing_ids,
        product_ids=product_ids,
        quantity_tiers=tuple(sorted(tiers, key=lambda item: item.minimum_quantity)),
        bundle_discounts=bundles,
        bundle_stacking=cast(BundleStacking, stacking),
        components=components,
        effective_after_ticks=effective_after,
        expires_after_ticks=expires_after,
    )
    pricing_policy_intent_fingerprint(normalized)
    return normalized


def pricing_policy_intent_fingerprint(intent: PricingPolicyIntent) -> str:
    return _digest(
        {
            "market_id": intent["market_id"],
            "policy_id": intent["policy_id"],
            "listing_ids": list(intent["listing_ids"]),
            "product_ids": list(intent["product_ids"]),
            "quantity_tiers": [
                {
                    "minimum_quantity": item.minimum_quantity,
                    "maximum_quantity": item.maximum_quantity,
                    "unit_price_minor": item.unit_price_minor,
                }
                for item in intent["quantity_tiers"]
            ],
            "bundle_discounts": [
                {
                    "bundle_id": item.bundle_id,
                    "conditions": [
                        {
                            "selector_kind": condition.selector_kind,
                            "selector_id": condition.selector_id,
                            "minimum_quantity": condition.minimum_quantity,
                        }
                        for condition in item.conditions
                    ],
                    "discount_minor": item.discount_minor,
                    "discount_bps": item.discount_bps,
                }
                for item in intent["bundle_discounts"]
            ],
            "bundle_stacking": intent["bundle_stacking"],
            "components": [
                {
                    "component_id": item.component_id,
                    "kind": item.kind,
                    "fixed_minor": item.fixed_minor,
                    "per_unit_minor": item.per_unit_minor,
                    "subtotal_rate_bps": item.subtotal_rate_bps,
                    "minimum_subtotal_minor": item.minimum_subtotal_minor,
                    "maximum_subtotal_minor": item.maximum_subtotal_minor,
                }
                for item in intent["components"]
            ],
            "effective_after_ticks": intent["effective_after_ticks"],
            "expires_after_ticks": intent["expires_after_ticks"],
        }
    )


def normalize_cart_quote_intent(value: Any) -> CartQuoteIntent:
    row = _mapping(value, "cart quote intent")
    _exact_fields(row, _QUOTE_FIELDS, _QUOTE_REQUIRED, "cart quote intent")
    raw_lines = _array(row.get("lines"), "lines")
    if not raw_lines or len(raw_lines) > 64:
        raise SchemaError("cart quote requires between 1 and 64 lines")
    lines: list[CartLineIntent] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_lines):
        line = _mapping(item, f"lines[{index}]")
        _exact_fields(
            line,
            frozenset({"sku_id", "qty"}),
            frozenset({"sku_id", "qty"}),
            f"lines[{index}]",
        )
        sku_id = _text(line.get("sku_id"), f"lines[{index}].sku_id")
        if sku_id in seen:
            raise SchemaError("cart quote cannot repeat a sku_id")
        seen.add(sku_id)
        lines.append(
            CartLineIntent(
                sku_id=sku_id,
                qty=_integer(line.get("qty"), f"lines[{index}].qty", minimum=1),
            )
        )
    fill = row.get("fill_policy", "all_or_none")
    backorder = row.get("backorder_policy", "reject")
    if fill not in {"all_or_none", "allow_partial"}:
        raise SchemaError("invalid fill_policy")
    if backorder not in {"reject", "allow"}:
        raise SchemaError("invalid backorder_policy")
    normalized = CartQuoteIntent(
        mandate_id=_text(row.get("mandate_id"), "mandate_id"),
        lines=tuple(lines),
        fill_policy=cast(FillPolicy, fill),
        backorder_policy=cast(BackorderPolicy, backorder),
    )
    cart_quote_intent_fingerprint(normalized, market_id="", quote_ttl_ticks=1)
    return normalized


def cart_quote_intent_fingerprint(
    intent: CartQuoteIntent,
    *,
    market_id: str,
    quote_ttl_ticks: int,
) -> str:
    if not isinstance(quote_ttl_ticks, int) or isinstance(quote_ttl_ticks, bool):
        raise SchemaError("quote_ttl_ticks must be a positive integer")
    if quote_ttl_ticks <= 0:
        raise SchemaError("quote_ttl_ticks must be a positive integer")
    return _digest(
        {
            "market_id": market_id,
            "mandate_id": intent["mandate_id"],
            "lines": [dict(line) for line in intent["lines"]],
            "fill_policy": intent["fill_policy"],
            "backorder_policy": intent["backorder_policy"],
            "quote_ttl_ticks": quote_ttl_ticks,
        }
    )


def pricing_policy_revision_key(
    market_id: str, merchant_id: str, policy_id: str, revision: int
) -> str:
    return (
        f"{len(market_id)}:{market_id}:{len(merchant_id)}:{merchant_id}:"
        f"{len(policy_id)}:{policy_id}:{revision:020d}"
    )


def cart_quote_id(
    market_id: str, requested_by: str, idempotency_key: str
) -> str:
    return f"quote:{_digest({'market_id': market_id, 'requested_by': requested_by, 'idempotency_key': idempotency_key})[:32]}"


def cart_quote_request_id(
    market_id: str, requested_by: str, idempotency_key: str
) -> str:
    return f"request:{_digest({'market_id': market_id, 'requested_by': requested_by, 'idempotency_key': idempotency_key})[:32]}"


def catalog_snapshot_digest(listing: Listing) -> str:
    return _digest(
        {
            "sku_id": str(listing.sku_id),
            "product_id": listing.product_id,
            "merchant_id": str(listing.merchant_id),
            "category": listing.category,
            "name": listing.name,
            "attributes": listing.attributes,
            "list_price_minor": _money_minor(listing),
            "currency": listing.list_price.currency,
        }
    )


def inventory_snapshot_digest(inventory: InventoryRow) -> str:
    return _digest(
        {
            "sku_id": str(inventory.sku_id),
            "merchant_id": str(inventory.merchant_id),
            "qty_available": inventory.qty_available,
            "qty_reserved": inventory.qty_reserved,
            "eta_day": inventory.eta_day,
            "version": inventory.version,
        }
    )


def pricing_policy_set_digest(
    policies: Sequence[PricingPolicyRevision],
) -> str:
    return _digest(
        [
            {
                "market_id": policy.market_id,
                "merchant_id": policy.merchant_id,
                "policy_id": policy.policy_id,
                "revision": policy.revision,
                "policy_digest": policy.policy_digest,
            }
            for policy in sorted(
                policies,
                key=lambda item: (item.merchant_id, item.policy_id, item.revision),
            )
        ]
    )


@dataclass(frozen=True, slots=True)
class QuoteAuthority:
    principal_id: str
    buyer_id: str
    mandate_revision: int
    mandate_digest: str


def build_authoritative_cart_quote(
    *,
    intent: CartQuoteIntent,
    market_id: str,
    requested_by: str,
    issuer_id: str,
    idempotency_key: str,
    authority: QuoteAuthority,
    listings: Mapping[str, Listing],
    inventory: Mapping[str, InventoryRow],
    policies_by_sku: Mapping[str, PricingPolicyRevision],
    issued_at_tick: int,
    quote_ttl_ticks: int,
    request_id: str | None = None,
) -> PersistentCartQuote:
    """Price and seal one cart using only authoritative World rows."""

    if set(listings) != {line["sku_id"] for line in intent["lines"]}:
        raise SchemaError("authoritative listing set does not match cart intent")
    if set(inventory) != set(listings) or set(policies_by_sku) != set(listings):
        raise SchemaError("cart quote authority inputs are incomplete")
    currency: str | None = None
    quantities: dict[str, int] = {}
    product_quantities: dict[str, int] = {}
    availability: dict[str, tuple[int, int, int, int | None]] = {}
    for line in intent["lines"]:
        sku_id = line["sku_id"]
        listing = listings[sku_id]
        stock = inventory[sku_id]
        if str(listing.merchant_id) != str(stock.merchant_id):
            raise SchemaError("catalog and inventory merchant ownership disagree")
        if currency is None:
            currency = listing.list_price.currency
        elif currency != listing.list_price.currency:
            raise SchemaError("cart quote cannot mix currencies")
        available = max(0, stock.qty_available - stock.qty_reserved)
        fill_now = min(line["qty"], available)
        remainder = line["qty"] - fill_now
        backorder = remainder if intent["backorder_policy"] == "allow" else 0
        unfilled = remainder - backorder
        if intent["fill_policy"] == "all_or_none" and unfilled:
            # The intent is structurally valid.  It cannot be fulfilled only
            # because authoritative inventory changed after the request was
            # created, so preserve that business-state distinction across
            # both World implementations and the HTTP VCP boundary.
            raise OutOfStock(f"insufficient inventory for {sku_id!r}")
        expected = (
            issued_at_tick + max(1, stock.eta_day)
            if backorder
            else None
        )
        availability[sku_id] = (fill_now, backorder, unfilled, expected)
        priced = fill_now + backorder
        quantities[sku_id] = priced
        if listing.product_id is not None:
            product_quantities[listing.product_id] = (
                product_quantities.get(listing.product_id, 0) + priced
            )

    policy_lines: dict[str, list[str]] = {}
    policy_objects: dict[str, PricingPolicyRevision] = {}
    base_prices: dict[str, int] = {}
    for sku_id, policy in policies_by_sku.items():
        key = policy.policy_digest
        policy_objects[key] = policy
        policy_lines.setdefault(key, []).append(sku_id)
    for key, sku_ids in policy_lines.items():
        policy = policy_objects[key]
        total_qty = sum(quantities[sku_id] for sku_id in sku_ids)
        tier = _active_tier(policy.quantity_tiers, total_qty)
        for sku_id in sku_ids:
            base_prices[sku_id] = tier.unit_price_minor

    discounted_prices = dict(base_prices)
    for key, sku_ids in policy_lines.items():
        policy = policy_objects[key]
        subtotal = sum(base_prices[sku] * quantities[sku] for sku in sku_ids)
        discount = _bundle_discount_minor(
            policy,
            subtotal=subtotal,
            listing_quantities=quantities,
            product_quantities=product_quantities,
        )
        _allocate_discount(
            discounted_prices,
            sku_ids=tuple(sorted(sku_ids)),
            quantities=quantities,
            discount_minor=discount,
        )

    lines = []
    for index, line in enumerate(intent["lines"], start=1):
        sku_id = line["sku_id"]
        listing = listings[sku_id]
        stock = inventory[sku_id]
        fill_now, backorder, unfilled, expected = availability[sku_id]
        lines.append(
            build_cart_quote_line(
                line_id=f"line:{index:03d}:{sku_id}",
                sku_id=sku_id,
                product_id=listing.product_id,
                merchant_id=str(listing.merchant_id),
                requested_qty=line["qty"],
                fulfill_now_qty=fill_now,
                backorder_qty=backorder,
                unfilled_qty=unfilled,
                unit_price_minor=discounted_prices[sku_id],
                catalog_revision=_catalog_revision(listing),
                catalog_digest=catalog_snapshot_digest(listing),
                inventory_revision=stock.version,
                inventory_digest=inventory_snapshot_digest(stock),
                expected_restock_tick=expected,
            )
        )

    charges = []
    for key, sku_ids in sorted(policy_lines.items()):
        policy = policy_objects[key]
        subtotal = sum(base_prices[sku] * quantities[sku] for sku in sku_ids)
        total_qty = sum(quantities[sku] for sku in sku_ids)
        for component in policy.components:
            upper = component.maximum_subtotal_minor
            if subtotal < component.minimum_subtotal_minor or (
                upper is not None and subtotal >= upper
            ):
                continue
            amount = (
                component.fixed_minor
                + component.per_unit_minor * total_qty
                + subtotal * component.subtotal_rate_bps // 10_000
            )
            charges.append(
                build_cart_quote_charge(
                    charge_id=(
                        f"charge:{policy.merchant_id}:{policy.policy_id}:"
                        f"{component.component_id}"
                    ),
                    kind=component.kind,
                    scope="merchant",
                    scope_id=policy.merchant_id,
                    amount_minor=amount,
                    basis=(
                        f"fixed={component.fixed_minor};per_unit="
                        f"{component.per_unit_minor};rate_bps="
                        f"{component.subtotal_rate_bps};subtotal={subtotal};"
                        f"quantity={total_qty}"
                    ),
                    policy_revision=policy.revision,
                    policy_digest=policy.policy_digest,
                )
            )
    policies = tuple(policy_objects.values())
    return build_persistent_cart_quote(
        quote_id=cart_quote_id(market_id, requested_by, idempotency_key),
        request_id=(
            request_id
            if request_id is not None
            else cart_quote_request_id(market_id, requested_by, idempotency_key)
        ),
        market_id=market_id,
        buyer_id=authority.buyer_id,
        principal_id=authority.principal_id,
        requested_by=requested_by,
        issuer_id=issuer_id,
        mandate_id=intent["mandate_id"],
        mandate_revision=authority.mandate_revision,
        mandate_digest=authority.mandate_digest,
        idempotency_key=idempotency_key,
        fill_policy=intent["fill_policy"],
        backorder_policy=intent["backorder_policy"],
        currency=currency or "USD",
        lines=lines,
        charges=charges,
        pricing_policy_revision=max(policy.revision for policy in policies),
        pricing_policy_digest=pricing_policy_set_digest(policies),
        issued_at_tick=issued_at_tick,
        expires_at_tick=issued_at_tick + quote_ttl_ticks,
    )


def _active_tier(tiers: tuple[QuantityTier, ...], quantity: int) -> QuantityTier:
    if quantity <= 0:
        raise SchemaError("pricing policy cannot price zero units")
    for tier in tiers:
        if quantity >= tier.minimum_quantity and (
            tier.maximum_quantity is None or quantity <= tier.maximum_quantity
        ):
            return tier
    raise SchemaError("pricing policy has no tier for requested quantity")


def _bundle_discount_minor(
    policy: PricingPolicyRevision,
    *,
    subtotal: int,
    listing_quantities: Mapping[str, int],
    product_quantities: Mapping[str, int],
) -> int:
    satisfied: list[tuple[str, int]] = []
    for discount in policy.bundle_discounts:
        if all(
            (
                listing_quantities.get(condition.selector_id, 0)
                if condition.selector_kind == "listing"
                else product_quantities.get(condition.selector_id, 0)
            )
            >= condition.minimum_quantity
            for condition in discount.conditions
        ):
            amount = (
                cast(int, discount.discount_minor)
                if discount.discount_minor is not None
                else subtotal * cast(int, discount.discount_bps) // 10_000
            )
            satisfied.append((discount.bundle_id, amount))
    if not satisfied:
        return 0
    if policy.bundle_stacking == "best_only":
        return min(subtotal, sorted(satisfied, key=lambda item: (-item[1], item[0]))[0][1])
    return min(subtotal, sum(amount for _, amount in satisfied))


def _allocate_discount(
    prices: dict[str, int],
    *,
    sku_ids: tuple[str, ...],
    quantities: Mapping[str, int],
    discount_minor: int,
) -> None:
    remaining = discount_minor
    if not remaining:
        return
    for sku_id in sku_ids:
        qty = quantities[sku_id]
        if qty <= 0:
            continue
        reduction = min(prices[sku_id] - 1, remaining // qty)
        prices[sku_id] -= reduction
        remaining -= reduction * qty
    if remaining:
        # PersistentCartQuote v1 stores integer unit prices and therefore
        # cannot truthfully encode a residual smaller than every line's unit
        # count.  Fail closed instead of silently changing the policy result.
        raise SchemaError(
            "bundle discount cannot be represented by integer per-unit quote lines"
        )


def _quantity_tier(value: Any, index: int) -> QuantityTier:
    row = _mapping(value, f"quantity_tiers[{index}]")
    _exact_fields(
        row,
        frozenset({"minimum_quantity", "maximum_quantity", "unit_price_minor"}),
        frozenset({"minimum_quantity", "unit_price_minor"}),
        f"quantity_tiers[{index}]",
    )
    raw_max = row.get("maximum_quantity")
    return build_quantity_tier(
        minimum_quantity=_integer(
            row.get("minimum_quantity"),
            f"quantity_tiers[{index}].minimum_quantity",
            minimum=1,
        ),
        maximum_quantity=(
            None
            if raw_max is None
            else _integer(
                raw_max,
                f"quantity_tiers[{index}].maximum_quantity",
                minimum=1,
            )
        ),
        unit_price_minor=_integer(
            row.get("unit_price_minor"),
            f"quantity_tiers[{index}].unit_price_minor",
            minimum=1,
        ),
    )


def _bundle_discount(value: Any, index: int) -> BundleDiscount:
    row = _mapping(value, f"bundle_discounts[{index}]")
    _exact_fields(
        row,
        frozenset({"bundle_id", "conditions", "discount_minor", "discount_bps"}),
        frozenset({"bundle_id", "conditions"}),
        f"bundle_discounts[{index}]",
    )
    conditions = []
    for condition_index, item in enumerate(
        _array(row.get("conditions"), f"bundle_discounts[{index}].conditions")
    ):
        condition = _mapping(
            item,
            f"bundle_discounts[{index}].conditions[{condition_index}]",
        )
        _exact_fields(
            condition,
            frozenset({"selector_kind", "selector_id", "minimum_quantity"}),
            frozenset({"selector_kind", "selector_id", "minimum_quantity"}),
            f"bundle_discounts[{index}].conditions[{condition_index}]",
        )
        kind = condition.get("selector_kind")
        if kind not in {"listing", "product"}:
            raise SchemaError("unsupported bundle selector_kind")
        conditions.append(
            build_bundle_condition(
                selector_kind=cast(Literal["listing", "product"], kind),
                selector_id=_text(condition.get("selector_id"), "selector_id"),
                minimum_quantity=_integer(
                    condition.get("minimum_quantity"),
                    "minimum_quantity",
                    minimum=1,
                ),
            )
        )
    raw_minor = row.get("discount_minor")
    raw_bps = row.get("discount_bps")
    return build_bundle_discount(
        bundle_id=_text(row.get("bundle_id"), f"bundle_discounts[{index}].bundle_id"),
        conditions=conditions,
        discount_minor=(
            None
            if raw_minor is None
            else _integer(raw_minor, "discount_minor", minimum=1)
        ),
        discount_bps=(
            None
            if raw_bps is None
            else _integer(raw_bps, "discount_bps", minimum=1)
        ),
    )


def _component(value: Any, index: int) -> PricingComponent:
    row = _mapping(value, f"components[{index}]")
    fields = frozenset(
        {
            "component_id",
            "kind",
            "fixed_minor",
            "per_unit_minor",
            "subtotal_rate_bps",
            "minimum_subtotal_minor",
            "maximum_subtotal_minor",
        }
    )
    _exact_fields(
        row,
        fields,
        frozenset({"component_id", "kind"}),
        f"components[{index}]",
    )
    kind = row.get("kind")
    if kind not in {"shipping", "tax", "fee"}:
        raise SchemaError("invalid pricing component kind")
    raw_max = row.get("maximum_subtotal_minor")
    return build_pricing_component(
        component_id=_text(row.get("component_id"), "component_id"),
        kind=cast(Literal["shipping", "tax", "fee"], kind),
        fixed_minor=_integer(row.get("fixed_minor", 0), "fixed_minor", minimum=0),
        per_unit_minor=_integer(
            row.get("per_unit_minor", 0), "per_unit_minor", minimum=0
        ),
        subtotal_rate_bps=_integer(
            row.get("subtotal_rate_bps", 0), "subtotal_rate_bps", minimum=0
        ),
        minimum_subtotal_minor=_integer(
            row.get("minimum_subtotal_minor", 0),
            "minimum_subtotal_minor",
            minimum=0,
        ),
        maximum_subtotal_minor=(
            None
            if raw_max is None
            else _integer(raw_max, "maximum_subtotal_minor", minimum=0)
        ),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise SchemaError(f"{label} must be an object with string fields")
    return cast(Mapping[str, Any], value)


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{label} must be an array")
    return value


def _exact_fields(
    row: Mapping[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    extra = set(row) - allowed
    missing = required - set(row)
    if extra or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise SchemaError(f"{label} has " + "; ".join(details))


def _canonical_texts(value: Any, label: str) -> tuple[str, ...]:
    rows = _array(value, label)
    values = tuple(sorted(_text(item, f"{label} item") for item in rows))
    if len(set(values)) != len(values):
        raise SchemaError(f"{label} contains duplicates")
    return values


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaError(f"{label} must be an integer >= {minimum}")
    return value


def _money_minor(listing: Listing) -> int:
    value = listing.list_price.amount * 100
    if value != value.to_integral_value():
        raise SchemaError("listing price has fractional minor units")
    return int(value)


def _catalog_revision(listing: Listing) -> int:
    value = listing.attributes.get("catalog_revision", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError("listing catalog_revision must be a positive integer")
    return value


def _digest(value: Any) -> str:
    try:
        return cast(str, canonical_digest(value))
    except (MatchValidationError, TypeError, ValueError) as exc:
        raise SchemaError(f"value is not canonical JSON: {exc}") from exc


__all__ = [
    "CartQuoteIntent",
    "PricingPolicyIntent",
    "QuoteAuthority",
    "build_authoritative_cart_quote",
    "cart_quote_id",
    "cart_quote_intent_fingerprint",
    "cart_quote_request_id",
    "catalog_snapshot_digest",
    "inventory_snapshot_digest",
    "normalize_cart_quote_intent",
    "normalize_pricing_policy_intent",
    "pricing_policy_intent_fingerprint",
    "pricing_policy_revision_key",
    "pricing_policy_set_digest",
]
