"""Static authority policy for model-facing business-intent schemas.

The language model chooses business intent.  It does not author protocol
correlation, authority, or replay fields.  Business resource references may
remain model choices only when the Agent has projected an authoritative,
finite set into the model-facing schema.

This module is intentionally independent from the Agent-private intent registry. It
can therefore audit a concrete, context-bound model surface without
importing a benchmark task, an ideal trajectory, or a scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# These fields are plumbing or sealed authority.  Even a one-value enum would
# ask the model to transcribe a framework value and is therefore forbidden.
FRAMEWORK_ONLY_PROVIDER_FIELDS = frozenset({
    "acceptance_request_id",
    "allocation_id",
    "actor_id",
    "authority_digest",
    "authority_id",
    "authorization_id",
    "campaign_id",
    "cart_id",
    "case_id",
    "cert_id",
    "counterparty_id",
    "decision_id",
    "event_id",
    "idempotency_key",
    "inbound_msg_id",
    "in_reply_to",
    "mandate_id",
    "msg_id",
    "negotiation_id",
    "offer_digest",
    "operation_id",
    "original_actor",
    "owner_id",
    "placement_id",
    "plan_id",
    "policy_id",
    "quote_id",
    "request_id",
    "request_msg_id",
    "response_msg_id",
    "receipt_id",
    "recipient_id",
    "result_id",
    "round_no",
    "session_digest",
    "session_id",
    "step_id",
    "source_msg_id",
    "submission_msg_id",
    "supply_authority_digest",
    "supply_authority_id",
    "transaction_id",
    "txn_id",
})


# Internal CommerceWorld identities never cross the provider boundary.  The
# Agent projects a model-selectable identity into an opaque public reference
# and reverses the projection after validating the decision.  Suffix rules are
# deliberate: new domains inherit this boundary without extending an
# allowlist.
INTERNAL_REFERENCE_SUFFIX = "_id"
INTERNAL_REFERENCE_ARRAY_SUFFIX = "_ids"
PUBLIC_REFERENCE_SUFFIX = "_ref"
PUBLIC_REFERENCE_ARRAY_SUFFIX = "_refs"
PUBLIC_REFERENCE_ALIAS_PREFIX = "business-ref-"
PUBLIC_REFERENCE_ALIAS_HEX_CHARS = 20
_LOWER_HEX = frozenset("0123456789abcdef")

# Correlation identities commonly acquire a semantic prefix (for example
# ``source_msg_id`` or ``receipt_txn_id``).  The final stem still identifies
# framework communication state and must remain Agent-owned.
_FRAMEWORK_CORRELATION_STEM_FAMILIES = frozenset({
    "event",
    "msg",
    "request",
    "session",
    "transaction",
    "txn",
})


def framework_owned_reference_field(field_name: str) -> bool:
    """Return whether an ID/reference stem belongs exclusively to the Agent.

    The bridge uses this predicate before projecting ``*_id`` fields.  Public
    spelling must not turn a protocol correlation or authority identity into a
    model choice, so the same predicate recognizes both internal and projected
    suffixes.
    """

    if not isinstance(field_name, str):
        return False
    normalized = field_name.casefold()
    for suffix in (
        INTERNAL_REFERENCE_ARRAY_SUFFIX,
        PUBLIC_REFERENCE_ARRAY_SUFFIX,
        INTERNAL_REFERENCE_SUFFIX,
        PUBLIC_REFERENCE_SUFFIX,
    ):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            stem = normalized[: -len(suffix)]
            family_owned = (
                stem.rsplit("_", 1)[-1]
                in _FRAMEWORK_CORRELATION_STEM_FAMILIES
            )
            return (
                family_owned
                or f"{stem}{INTERNAL_REFERENCE_SUFFIX}"
                in FRAMEWORK_ONLY_PROVIDER_FIELDS
                or f"{stem}{INTERNAL_REFERENCE_ARRAY_SUFFIX}"
                in FRAMEWORK_ONLY_PROVIDER_FIELDS
            )
    return False


def _framework_owned_field(field_name: str) -> bool:
    """Return whether one provider argument is deterministic plumbing.

    Suffix rules prevent a newly added protocol from bypassing the audit only
    because its exact field name has not yet been added to the allowlist.
    """

    normalized = field_name.casefold()
    return (
        normalized in FRAMEWORK_ONLY_PROVIDER_FIELDS
        or framework_owned_reference_field(field_name)
        or normalized.endswith("_idempotency_key")
        or normalized.endswith("_digest")
        or normalized == "revision"
        or normalized.endswith("_revision")
        or normalized == "version"
        or normalized.endswith("_version")
        or normalized == "tick"
        or normalized.endswith("_tick")
        or normalized in {"currency", "sequence_no"}
    )


def _internal_reference_field(field_name: str) -> bool:
    """Recognize a singular Agent-private CommerceWorld identity."""

    return field_name.casefold().endswith(INTERNAL_REFERENCE_SUFFIX)


def _internal_reference_array_field(field_name: str) -> bool:
    """Recognize plural Agent-private CommerceWorld identities."""

    return field_name.casefold().endswith(INTERNAL_REFERENCE_ARRAY_SUFFIX)


def _public_reference_field(field_name: str) -> bool:
    """Recognize one provider-visible opaque business reference."""

    return field_name.casefold().endswith(PUBLIC_REFERENCE_SUFFIX)


def _public_reference_array_field(field_name: str) -> bool:
    """Recognize provider-visible opaque business references."""

    return field_name.casefold().endswith(PUBLIC_REFERENCE_ARRAY_SUFFIX)


def public_reference_alias_is_valid(value: Any) -> bool:
    """Return whether a value matches the frozen opaque-reference format."""

    if not isinstance(value, str) or not value.startswith(
        PUBLIC_REFERENCE_ALIAS_PREFIX
    ):
        return False
    digest = value[len(PUBLIC_REFERENCE_ALIAS_PREFIX) :]
    return (
        len(digest) == PUBLIC_REFERENCE_ALIAS_HEX_CHARS
        and all(character in _LOWER_HEX for character in digest)
    )


def business_intent_authority_violations(
    schema: Mapping[str, Any],
    *,
    root: str = "arguments",
) -> tuple[str, ...]:
    """Return deterministic authority-boundary violations for one schema.

    The input is the exact business-intent schema emitted after Agent context
    binding.  The same rule applies to observation and terminal intent
    schemas: the model only receives finite opaque ``*_ref``/``*_refs``
    choices, never an internal ``*_id``/``*_ids`` field.
    """

    violations: list[str] = []
    if not isinstance(schema, Mapping):
        return (f"{root} is not a JSON schema object",)
    _inspect_schema(schema, path=root, field_name=None, violations=violations)
    return tuple(sorted(set(violations)))


def require_business_intent_authority_safe(
    schema: Mapping[str, Any],
    *,
    root: str = "arguments",
) -> None:
    """Raise when a business schema exposes Agent-owned identity or authority."""

    violations = business_intent_authority_violations(schema, root=root)
    if violations:
        raise ValueError(
            "business intent schema violates the Agent authority boundary: "
            + "; ".join(violations)
        )


def _inspect_schema(
    schema: Mapping[str, Any],
    *,
    path: str,
    field_name: str | None,
    violations: list[str],
) -> None:
    if field_name is not None:
        if _internal_reference_array_field(field_name) or _internal_reference_field(
            field_name
        ):
            violations.append(
                f"{path} is a framework-owned internal ID leak"
                if _framework_owned_field(field_name)
                else f"{path} is an internal ID leak"
            )
        elif _framework_owned_field(field_name):
            violations.append(f"{path} is framework-owned")
        elif _public_reference_array_field(field_name) and not _finite_item_enum(
            schema
        ):
            violations.append(
                f"{path} items are not a non-empty finite public reference enum"
            )
        elif _public_reference_field(field_name) and not _finite_enum(schema):
            violations.append(
                f"{path} is not a non-empty finite public reference enum"
            )

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for raw_name, child in properties.items():
            name = str(raw_name)
            if isinstance(child, Mapping):
                _inspect_schema(
                    child,
                    path=f"{path}.{name}",
                    field_name=name,
                    violations=violations,
                )

    items = schema.get("items")
    if isinstance(items, Mapping):
        _inspect_schema(
            items,
            path=f"{path}[]",
            field_name=None,
            violations=violations,
        )

    for union_name in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(union_name)
        if isinstance(branches, Sequence) and not isinstance(
            branches, (str, bytes, bytearray)
        ):
            for index, branch in enumerate(branches):
                if isinstance(branch, Mapping):
                    _inspect_schema(
                        branch,
                        path=f"{path}.{union_name}[{index}]",
                        # The enclosing field has already been checked against
                        # its complete schema.  Branches still need traversal
                        # for nested properties, but repeating ``field_name``
                        # would produce branch-order-dependent duplicate
                        # diagnostics.
                        field_name=None,
                        violations=violations,
                    )

    # These keywords contain schemas but not concrete provider field names.
    # Traverse them so nesting or a local definition cannot hide an internal
    # ID.  Reference fields themselves are still checked when encountered in
    # a descendant ``properties`` map.
    for container_name in (
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
    ):
        children = schema.get(container_name)
        if isinstance(children, Mapping):
            for raw_name, child in children.items():
                if isinstance(child, Mapping):
                    _inspect_schema(
                        child,
                        path=f"{path}.{container_name}.{raw_name}",
                        field_name=None,
                        violations=violations,
                    )

    for child_name in ("additionalProperties", "contains"):
        child = schema.get(child_name)
        if isinstance(child, Mapping):
            _inspect_schema(
                child,
                path=f"{path}.{child_name}",
                field_name=None,
                violations=violations,
            )


def _finite_enum(schema: Mapping[str, Any]) -> bool:
    values = schema.get("enum")
    return bool(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes, bytearray))
        and values
        and all(public_reference_alias_is_valid(value) for value in values)
        and len(values) == len({repr(value) for value in values})
    )


def _finite_item_enum(schema: Mapping[str, Any]) -> bool:
    if schema.get("const") == [] and schema.get("maxItems") == 0:
        return True
    items = schema.get("items")
    return isinstance(items, Mapping) and _finite_enum(items)


__all__ = [
    "FRAMEWORK_ONLY_PROVIDER_FIELDS",
    "INTERNAL_REFERENCE_ARRAY_SUFFIX",
    "INTERNAL_REFERENCE_SUFFIX",
    "PUBLIC_REFERENCE_ALIAS_HEX_CHARS",
    "PUBLIC_REFERENCE_ALIAS_PREFIX",
    "PUBLIC_REFERENCE_ARRAY_SUFFIX",
    "PUBLIC_REFERENCE_SUFFIX",
    "framework_owned_reference_field",
    "public_reference_alias_is_valid",
    "business_intent_authority_violations",
    "require_business_intent_authority_safe",
]
