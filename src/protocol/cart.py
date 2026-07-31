"""Typed cart quote wire coercion, arithmetic validation, and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from typing import Any, Mapping

from protocol.errors import SchemaError
from world.types import (
    AgentId,
    CartLine,
    CartQuote,
    FeeComponent,
    Money,
    OrderGroup,
    OrderGroupId,
    OrderId,
    QuoteId,
    SkuId,
    TxnId,
)


CART_QUOTE_SCHEMA = "cwe.cart-quote.v1"


def money_from_cents(cents: int, currency: str = "USD") -> Money:
    if isinstance(cents, bool) or not isinstance(cents, int):
        raise SchemaError("money cents must be an integer")
    return Money(Decimal(cents) / Decimal(100), currency)


def money_to_cents(value: Money) -> int:
    cents = value.amount * Decimal(100)
    if cents != cents.to_integral_value():
        raise SchemaError("cart money must use integer minor units")
    return int(cents)


def cart_quote_hash(quote: CartQuote) -> str:
    body = json.dumps(
        _quote_contract(quote),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def seal_cart_quote(quote: CartQuote) -> CartQuote:
    return replace(quote, quote_hash=cart_quote_hash(quote))


def validate_cart_quote(quote: CartQuote) -> None:
    if not quote.lines:
        raise SchemaError("cart quote needs at least one line")
    if len({str(line.sku_id) for line in quote.lines}) != len(quote.lines):
        raise SchemaError("cart quote contains duplicate sku lines")
    currencies = {
        quote.subtotal.currency,
        quote.grand_total.currency,
        *(line.unit_price.currency for line in quote.lines),
        *(line.line_total.currency for line in quote.lines),
        *(fee.amount.currency for fee in quote.fee_breakdown),
    }
    if len(currencies) != 1:
        raise SchemaError("cart quote currencies must match")
    for line in quote.lines:
        if line.qty <= 0:
            raise SchemaError("cart line qty must be positive")
        if money_to_cents(line.unit_price) <= 0:
            raise SchemaError("cart line unit price must be positive")
        if money_to_cents(line.line_total) != money_to_cents(line.unit_price) * line.qty:
            raise SchemaError("cart line total disagrees with unit price and qty")
        if not 0 <= line.bundle_discount_bps <= 10_000:
            raise SchemaError("cart line bundle discount is invalid")
    subtotal = sum(money_to_cents(line.line_total) for line in quote.lines)
    if money_to_cents(quote.subtotal) != subtotal:
        raise SchemaError("cart quote subtotal disagrees with lines")
    fees = sum(money_to_cents(fee.amount) for fee in quote.fee_breakdown)
    if money_to_cents(quote.grand_total) != subtotal + fees:
        raise SchemaError("cart quote grand total disagrees with fees")
    if quote.issued_at_tick < 0 or quote.expires_at_tick <= quote.issued_at_tick:
        raise SchemaError("cart quote expiry is invalid")
    if len(quote.quote_hash) != 64 or quote.quote_hash != cart_quote_hash(quote):
        raise SchemaError("cart quote hash mismatch")


def coerce_cart_quote(value: Any) -> CartQuote:
    row = value.get("quote", value) if isinstance(value, Mapping) else value
    if isinstance(row, CartQuote):
        return row
    if not isinstance(row, Mapping):
        raise SchemaError("cart quote must be an object")
    try:
        quote = CartQuote(
            quote_id=QuoteId(str(row["quote_id"])),
            buyer_id=AgentId(str(row["buyer_id"])),
            requested_by=AgentId(str(row["requested_by"])),
            lines=tuple(_coerce_line(item) for item in row["lines"]),
            subtotal=_coerce_money(row["subtotal"]),
            fee_breakdown=tuple(_coerce_fee(item) for item in row["fee_breakdown"]),
            grand_total=_coerce_money(row["grand_total"]),
            issued_at_tick=int(row["issued_at_tick"]),
            expires_at_tick=int(row["expires_at_tick"]),
            quote_hash=str(row["quote_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid cart quote: {exc}") from exc
    return quote


def coerce_order_group(value: Any) -> OrderGroup:
    row = value.get("order_group", value) if isinstance(value, Mapping) else value
    if isinstance(row, OrderGroup):
        return row
    if not isinstance(row, Mapping):
        raise SchemaError("order group must be an object")
    try:
        return OrderGroup(
            order_group_id=OrderGroupId(str(row["order_group_id"])),
            quote_id=QuoteId(str(row["quote_id"])),
            buyer_id=AgentId(str(row["buyer_id"])),
            merchant_ids=tuple(AgentId(str(item)) for item in row["merchant_ids"]),
            order_ids=tuple(OrderId(str(item)) for item in row["order_ids"]),
            txn_ids=tuple(TxnId(str(item)) for item in row["txn_ids"]),
            subtotal=_coerce_money(row["subtotal"]),
            fee_breakdown=tuple(_coerce_fee(item) for item in row["fee_breakdown"]),
            grand_total=_coerce_money(row["grand_total"]),
            quote_hash=str(row["quote_hash"]),
            idempotency_key=str(row["idempotency_key"]),
            state=str(row.get("state", "settled")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid order group: {exc}") from exc


def _quote_contract(quote: CartQuote) -> dict[str, Any]:
    return {
        "schema_version": CART_QUOTE_SCHEMA,
        "quote_id": str(quote.quote_id),
        "buyer_id": str(quote.buyer_id),
        "requested_by": str(quote.requested_by),
        "lines": [
            {
                "sku_id": str(line.sku_id),
                "merchant_id": str(line.merchant_id),
                "qty": line.qty,
                "list_unit_price_cents": money_to_cents(line.list_unit_price),
                "unit_price_cents": money_to_cents(line.unit_price),
                "line_total_cents": money_to_cents(line.line_total),
                "tier_min_qty": line.tier_min_qty,
                "bundle_discount_bps": line.bundle_discount_bps,
            }
            for line in quote.lines
        ],
        "subtotal_cents": money_to_cents(quote.subtotal),
        "fee_breakdown": [
            {
                "fee_id": fee.fee_id,
                "kind": fee.kind,
                "scope": fee.scope,
                "amount_cents": money_to_cents(fee.amount),
                "basis": fee.basis,
            }
            for fee in quote.fee_breakdown
        ],
        "grand_total_cents": money_to_cents(quote.grand_total),
        "issued_at_tick": quote.issued_at_tick,
        "expires_at_tick": quote.expires_at_tick,
    }


def _coerce_line(value: Any) -> CartLine:
    if isinstance(value, CartLine):
        return value
    if not isinstance(value, Mapping):
        raise SchemaError("cart line must be an object")
    return CartLine(
        sku_id=SkuId(str(value["sku_id"])),
        merchant_id=AgentId(str(value["merchant_id"])),
        qty=int(value["qty"]),
        list_unit_price=_coerce_money(value["list_unit_price"]),
        unit_price=_coerce_money(value["unit_price"]),
        line_total=_coerce_money(value["line_total"]),
        tier_min_qty=(
            int(value["tier_min_qty"])
            if value.get("tier_min_qty") is not None
            else None
        ),
        bundle_discount_bps=int(value.get("bundle_discount_bps", 0)),
    )


def _coerce_fee(value: Any) -> FeeComponent:
    if isinstance(value, FeeComponent):
        return value
    if not isinstance(value, Mapping):
        raise SchemaError("fee component must be an object")
    return FeeComponent(
        fee_id=str(value["fee_id"]),
        kind=str(value["kind"]),
        scope=str(value["scope"]),
        amount=_coerce_money(value["amount"]),
        basis=str(value["basis"]),
    )


def _coerce_money(value: Any) -> Money:
    if isinstance(value, Money):
        return value
    if not isinstance(value, Mapping):
        raise SchemaError("money must be an object")
    return Money(Decimal(str(value["amount"])), str(value.get("currency", "USD")))


__all__ = [
    "CART_QUOTE_SCHEMA",
    "cart_quote_hash",
    "coerce_cart_quote",
    "coerce_order_group",
    "money_from_cents",
    "money_to_cents",
    "seal_cart_quote",
    "validate_cart_quote",
]
