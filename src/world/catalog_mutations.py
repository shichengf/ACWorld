"""Pure validation and transition logic for authoritative catalog intents.

Actors submit small commerce intents.  They never submit a sealed ``Listing``
or any authority, revision, fingerprint, or idempotency material.  Platform
authenticates the actor from the envelope and World applies the transition to
its current row under the World transaction lock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from typing import Any, Literal, TypedDict, cast

from protocol.errors import SchemaError
from protocol.matching import MatchValidationError, canonical_digest
from world.evidence_contracts import reject_embedded_authority_records
from world.errors import CatalogMutationRejected, WriteNotAuthorized
from world.types import AgentId, Listing, Money, SkuId


CatalogMutationOperation = Literal[
    "publish", "update", "delist", "adjust_price"
]


class CatalogMutationIntent(TypedDict):
    """Compact actor intent accepted by the World catalog state machine."""

    op: CatalogMutationOperation
    sku_id: str
    fields: dict[str, Any]


_LEGACY_MERCHANT_SUBROLES = frozenset(
    {"catalog", "fulfillment", "pricing", "retrieval", "support", "owner"}
)
_TOP_LEVEL_FIELDS = frozenset({"op", "sku_id", "fields"})
_PUBLISH_FIELDS = frozenset(
    {
        "list_price",
        "attributes",
        "permitted_claims",
        "must_not_claim",
        "inventory",
        "status",
        "category",
        "name",
        "currency",
        "product_id",
    }
)
_UPDATE_FIELDS = frozenset(
    {
        "attributes",
        "permitted_claims",
        "must_not_claim",
        "status",
        "category",
        "name",
        "product_id",
    }
)
_FORBIDDEN_AUTHORITY_FIELDS = frozenset(
    {
        "merchant_id",
        "original_actor",
        "by_actor",
        "idempotency_key",
        "request_fingerprint",
        "catalog_revision",
        "floor_price",
    }
)


def normalize_catalog_mutation_intent(value: Any) -> CatalogMutationIntent:
    """Validate and return a canonical, detached compact catalog intent."""

    if not isinstance(value, Mapping):
        raise SchemaError("catalog mutation intent must be an object")
    extra = set(value) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(value)
    if extra or missing:
        raise SchemaError(
            "catalog mutation intent requires exactly op, sku_id, and fields"
        )
    op = value.get("op")
    if op not in {"publish", "update", "delist", "adjust_price"}:
        raise SchemaError("unknown_op")
    sku_id = value.get("sku_id")
    if not isinstance(sku_id, str) or not sku_id.strip():
        raise SchemaError("missing_sku_id")
    fields_value = value.get("fields")
    if not isinstance(fields_value, Mapping):
        raise SchemaError("catalog mutation fields must be an object")
    fields = {str(key): item for key, item in fields_value.items()}
    if len(fields) != len(fields_value) or any(
        not isinstance(key, str) or not key for key in fields_value
    ):
        raise SchemaError("catalog mutation field names must be non-empty strings")
    forbidden = sorted(_FORBIDDEN_AUTHORITY_FIELDS.intersection(fields))
    if forbidden:
        raise SchemaError(
            "catalog authority fields are World-owned: " + ", ".join(forbidden)
        )
    allowed = (
        _PUBLISH_FIELDS
        if op == "publish"
        else _UPDATE_FIELDS
        if op == "update"
        else frozenset()
        if op == "delist"
        else frozenset({"list_price"})
    )
    unexpected = sorted(set(fields) - allowed)
    if unexpected:
        raise SchemaError(
            f"unsupported {op} fields: " + ", ".join(unexpected)
        )
    _validate_operation_fields(cast(CatalogMutationOperation, op), fields)
    intent: CatalogMutationIntent = {
        "op": cast(CatalogMutationOperation, op),
        "sku_id": sku_id.strip(),
        "fields": _detach_json(fields),
    }
    # This is also the fail-closed JSON-domain check used by the request
    # fingerprint.  It rejects NaN, non-string keys, and non-JSON objects.
    catalog_mutation_fingerprint(intent)
    return intent


def catalog_mutation_fingerprint(intent: CatalogMutationIntent) -> str:
    """Return the stable fingerprint bound to actor-scoped idempotency."""

    try:
        return cast(str, canonical_digest(intent))
    except (MatchValidationError, TypeError, ValueError) as exc:
        raise SchemaError(f"catalog mutation intent is not canonical JSON: {exc}") from exc


def apply_catalog_mutation_intent(
    current: Listing | None,
    intent: CatalogMutationIntent,
    *,
    original_actor: str,
) -> Listing:
    """Apply a validated intent to ``current`` without stamping a revision."""

    owner = catalog_owner_for_actor(original_actor)
    op = intent["op"]
    fields = intent["fields"]
    sku_id = SkuId(intent["sku_id"])
    if op == "publish":
        if current is not None:
            raise CatalogMutationRejected("already_listed")
        attributes = _public_attributes(fields)
        return Listing(
            sku_id=sku_id,
            category=str(fields.get("category", "")),
            name=str(fields.get("name", "")),
            attributes=attributes,
            list_price=Money(
                Decimal(cast(int, fields["list_price"])) / Decimal("100"),
                str(fields.get("currency", "USD")),
            ),
            merchant_id=AgentId(owner),
            product_id=(
                str(fields["product_id"])
                if fields.get("product_id") is not None
                else None
            ),
        )
    if current is None:
        raise CatalogMutationRejected("unknown_sku")
    assert_catalog_owner(original_actor, current)
    if op == "adjust_price":
        return replace(
            current,
            list_price=Money(
                Decimal(cast(int, fields["list_price"])) / Decimal("100"),
                current.list_price.currency,
            ),
        )
    if op == "delist":
        attributes = dict(current.attributes)
        attributes["status"] = "delisted"
        return replace(current, attributes=attributes)
    attributes = dict(current.attributes)
    attributes.update(cast(dict[str, Any], fields.get("attributes", {})))
    _merge_public_metadata(attributes, fields)
    return replace(
        current,
        category=str(fields.get("category", current.category)),
        name=str(fields.get("name", current.name)),
        attributes=attributes,
        product_id=(
            str(fields["product_id"])
            if "product_id" in fields and fields["product_id"] is not None
            else current.product_id
        ),
    )


def catalog_listings_semantically_equal(left: Listing, right: Listing) -> bool:
    """Compare public catalog content while ignoring World-owned revision."""

    left_attributes = dict(left.attributes)
    right_attributes = dict(right.attributes)
    left_attributes.pop("catalog_revision", None)
    right_attributes.pop("catalog_revision", None)
    return cast(
        bool,
        replace(left, attributes=left_attributes)
        == replace(right, attributes=right_attributes),
    )


def catalog_owner_for_actor(actor_id: str) -> str:
    """Derive an owner id from an authenticated merchant actor address."""

    if not isinstance(actor_id, str) or not actor_id:
        raise WriteNotAuthorized("catalog mutation actor must be non-empty")
    side, separator, suffix = actor_id.partition(":")
    if side != "merchant":
        raise WriteNotAuthorized("sender_not_permitted")
    if not separator or suffix in _LEGACY_MERCHANT_SUBROLES:
        return "merchant"
    return actor_id


def assert_catalog_owner(actor_id: str, listing: Listing) -> None:
    """Reject a merchant that does not exactly own the authoritative row."""

    if catalog_owner_for_actor(actor_id) != str(listing.merchant_id):
        raise WriteNotAuthorized("sku_not_owned_by_sender")


def _validate_operation_fields(op: CatalogMutationOperation, fields: dict[str, Any]) -> None:
    if op in {"publish", "adjust_price"}:
        value = fields.get("list_price")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SchemaError("invalid_list_price")
    if op == "publish":
        currency = fields.get("currency", "USD")
        if not isinstance(currency, str) or not currency.strip():
            raise SchemaError("invalid_currency")
        inventory = fields.get("inventory", 0)
        if isinstance(inventory, bool) or not isinstance(inventory, int) or inventory < 0:
            raise SchemaError("invalid_inventory")
        if inventory:
            raise SchemaError("inventory_requires_receive_shipment")
    for key in ("category", "name", "status", "product_id"):
        if key in fields and fields[key] is not None and not isinstance(fields[key], str):
            raise SchemaError(f"invalid_{key}")
    attributes = fields.get("attributes", {})
    if not isinstance(attributes, Mapping):
        raise SchemaError("invalid_attributes")
    if any(not isinstance(key, str) or not key for key in attributes):
        raise SchemaError("listing attribute names must be non-empty strings")
    nested_forbidden = sorted(_FORBIDDEN_AUTHORITY_FIELDS.intersection(attributes))
    if nested_forbidden:
        raise SchemaError(
            "listing attributes cannot carry authority state: "
            + ", ".join(nested_forbidden)
        )
    reject_embedded_authority_records(dict(attributes))
    for key in ("permitted_claims", "must_not_claim"):
        if key not in fields:
            continue
        values = fields[key]
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise SchemaError(f"invalid_{key}")


def _public_attributes(fields: dict[str, Any]) -> dict[str, Any]:
    attributes = dict(cast(Mapping[str, Any], fields.get("attributes", {})))
    _merge_public_metadata(attributes, fields)
    return attributes


def _merge_public_metadata(attributes: dict[str, Any], fields: Mapping[str, Any]) -> None:
    for key in ("permitted_claims", "must_not_claim"):
        if key in fields:
            attributes[key] = list(cast(list[str] | tuple[str, ...], fields[key]))
    if "status" in fields:
        attributes["status"] = str(fields["status"])


def _detach_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SchemaError("catalog mutation JSON keys must be strings")
        return {key: _detach_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach_json(item) for item in value]
    return value


__all__ = [
    "CatalogMutationIntent",
    "CatalogMutationOperation",
    "apply_catalog_mutation_intent",
    "assert_catalog_owner",
    "catalog_listings_semantically_equal",
    "catalog_mutation_fingerprint",
    "catalog_owner_for_actor",
    "normalize_catalog_mutation_intent",
]
