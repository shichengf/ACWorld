"""World authority rules for evidence, mandates, and listing claims.

Protocol modules define canonical immutable values. This module binds those
values to authenticated actors, World logical time, catalog ownership, and
persisted predecessor state. Scenarios and agents must use these rules through
World methods rather than maintaining task-local registries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from protocol.errors import IdempotencyConflict
from protocol.evidence_records import (
    EVIDENCE_RECORD_SCHEMA_ID,
    MANDATE_REVISION_SCHEMA_ID,
    EvidenceReadAuthority,
    EvidenceRecord,
    MandateRevision,
    MandateRevisionAuthority,
    authorize_evidence_read,
    validate_evidence_record,
    validate_mandate_revision_transition,
)
from protocol.listing_claims import (
    LISTING_CLAIM_SCHEMA,
    ClaimAuthorityError,
    ClaimSchemaError,
    ListingClaim,
    validate_listing_claim,
)
from protocol.cart_quote_state import PERSISTENT_CART_QUOTE_SCHEMA
from protocol.pricing_policy import PRICING_POLICY_SCHEMA
from world.errors import WorldError, WriteNotAuthorized
from world.types import Listing


AppendDisposition = Literal["append", "idempotent"]


def mandate_authority_to_wire(
    authority: MandateRevisionAuthority,
) -> dict[str, object]:
    """Return the exact JSON shape used by World persistence and VCP."""

    return {
        "principal_id": authority.principal_id,
        "buyer_id": authority.buyer_id,
        "mandate_id": authority.mandate_id,
        "allowed_fields": list(authority.allowed_fields),
    }


def coerce_mandate_authority(value: Any) -> MandateRevisionAuthority:
    """Coerce an exact authority object without trusting extra fields."""

    if isinstance(value, MandateRevisionAuthority):
        return value
    if not isinstance(value, Mapping):
        raise WorldError("mandate authority must be an object")
    expected = {"principal_id", "buyer_id", "mandate_id", "allowed_fields"}
    if set(value) != expected:
        raise WorldError("mandate authority fields must match the v1 contract")
    raw_fields = value["allowed_fields"]
    if not isinstance(raw_fields, list) or not all(
        isinstance(item, str) for item in raw_fields
    ):
        raise WorldError("mandate authority allowed_fields must be a string array")
    fields = tuple(raw_fields)
    if fields != tuple(sorted(set(fields))):
        raise WorldError("mandate authority allowed_fields must be unique and sorted")
    for field in ("principal_id", "buyer_id", "mandate_id"):
        raw = value[field]
        if not isinstance(raw, str) or not raw.strip():
            raise WorldError(f"mandate authority {field} must be a non-empty string")
    return MandateRevisionAuthority(
        principal_id=value["principal_id"],
        buyer_id=value["buyer_id"],
        mandate_id=value["mandate_id"],
        allowed_fields=fields,
    )


def evidence_record_key(record_id: str, version: int) -> str:
    """Return an injective key for one immutable evidence version."""

    return f"{len(record_id)}:{record_id}:{version:020d}"


def mandate_revision_key(mandate_id: str, revision: int) -> str:
    """Return an injective key for one immutable mandate revision."""

    return f"{len(mandate_id)}:{mandate_id}:{revision:020d}"


def authority_operation_key(scope: str, actor_id: str, key: str) -> str:
    """Return an injective key for one actor-scoped authority operation."""

    return (
        f"{len(scope)}:{scope}:"
        f"{len(actor_id)}:{actor_id}:"
        f"{len(key)}:{key}"
    )


def validate_evidence_append(
    current: EvidenceRecord | None,
    candidate: EvidenceRecord,
    *,
    original_actor: str,
    logical_time: int,
) -> AppendDisposition:
    """Bind an evidence version to its issuer and persisted predecessor."""

    validate_evidence_record(candidate)
    if candidate.issuer_id != original_actor:
        raise WriteNotAuthorized(
            "evidence record issuer must match the authenticated original actor"
        )
    if candidate.issued_at_tick != logical_time:
        raise WorldError("evidence issued_at_tick must equal World logical time")
    if current is None:
        if candidate.version != 1:
            raise WorldError("first evidence record version must be 1")
        return "append"
    if candidate.version == current.version:
        if candidate == current:
            return "idempotent"
        raise IdempotencyConflict(
            "evidence record version was reused for different canonical content"
        )
    if candidate.version != current.version + 1:
        raise WorldError(
            f"evidence record version must advance from {current.version} "
            f"to {current.version + 1}"
        )
    immutable = (
        "record_id",
        "kind",
        "subject_id",
        "issuer_id",
        "owner_id",
    )
    changed = [
        field
        for field in immutable
        if getattr(candidate, field) != getattr(current, field)
    ]
    if changed:
        raise WorldError(
            "evidence record immutable identity changed: " + ", ".join(changed)
        )
    if candidate.issued_at_tick <= current.issued_at_tick:
        raise WorldError("evidence issued_at_tick must increase across versions")
    return "append"


def evidence_read_authority(record: EvidenceRecord) -> EvidenceReadAuthority:
    """Construct the trusted read binding from a persisted World row."""

    return EvidenceReadAuthority(
        record_id=record.record_id,
        issuer_id=record.issuer_id,
        owner_id=record.owner_id,
        read_acl=record.read_acl,
        version=record.version,
        record_digest=record.record_digest,
    )


def authorize_persisted_evidence_read(
    record: EvidenceRecord, *, reader_id: str
) -> EvidenceRecord:
    """Authorize an actor read using only metadata loaded from World."""

    if _trusted_service(reader_id):
        validate_evidence_record(record)
        return record
    return authorize_evidence_read(
        record,
        reader_id=reader_id,
        authority=evidence_read_authority(record),
    )


def validate_mandate_authority_registration(
    current: MandateRevisionAuthority | None,
    candidate: MandateRevisionAuthority,
    *,
    original_actor: str,
) -> AppendDisposition:
    """Persist an immutable authority only when its principal authenticates."""

    if candidate.principal_id != original_actor:
        raise WriteNotAuthorized(
            "mandate authority principal must match authenticated original actor"
        )
    if current is None:
        return "append"
    if current == candidate:
        return "idempotent"
    raise IdempotencyConflict(
        "mandate authority is immutable once registered for a mandate"
    )


def validate_mandate_append(
    current: MandateRevision | None,
    candidate: MandateRevision,
    authority: MandateRevisionAuthority,
    *,
    original_actor: str,
    logical_time: int,
) -> AppendDisposition:
    """Bind a contiguous revision to persisted principal authority and time."""

    if candidate.principal_id != original_actor:
        raise WriteNotAuthorized(
            "mandate revision principal must match authenticated original actor"
        )
    if candidate.logical_tick != logical_time:
        raise WorldError("mandate revision logical_tick must equal World logical time")
    return validate_mandate_revision_transition(current, candidate, authority)


def authorize_mandate_read(
    authority: MandateRevisionAuthority, *, reader_id: str
) -> None:
    """Limit mandate history to the bound principal, buyer, and services."""

    if _trusted_service(reader_id):
        return
    if reader_id not in {authority.principal_id, authority.buyer_id}:
        raise WriteNotAuthorized(
            f"actor {reader_id!r} cannot read mandate {authority.mandate_id!r}"
        )


def validate_listing_claim_append(
    current: ListingClaim | None,
    candidate: ListingClaim,
    *,
    listing: Listing | None,
    original_actor: str,
    idempotency_key: str,
    logical_time: int,
    evidence_lookup: Callable[[str, str], EvidenceRecord | None],
) -> AppendDisposition:
    """Validate one World-backed listing-claim transition.

    ``evidence_lookup`` resolves an exact ``(record_id, digest)`` historical
    evidence version. A claim can therefore remain replayable after a source
    record receives a later version.
    """

    validate_listing_claim(candidate)
    if listing is None:
        raise ClaimSchemaError("listing claim references an unknown listing")
    if candidate.listing_id != str(listing.sku_id):
        raise ClaimSchemaError("listing claim identity does not match listing SKU")
    if candidate.merchant_id != str(listing.merchant_id):
        raise ClaimAuthorityError("listing claim merchant does not own listing")
    if candidate.issuer_id != original_actor or candidate.merchant_id != original_actor:
        raise ClaimAuthorityError(
            "listing claim issuer must match authenticated owning merchant"
        )

    if current is None:
        if len(candidate.versions) != 1 or candidate.current.version != 1:
            raise ClaimSchemaError("new listing claim must contain only version 1")
        disposition: AppendDisposition = "append"
    elif candidate == current:
        disposition = "idempotent"
    else:
        if (
            candidate.claim_id != current.claim_id
            or candidate.listing_id != current.listing_id
            or candidate.merchant_id != current.merchant_id
            or candidate.subject != current.subject
            or candidate.issuer_id != current.issuer_id
        ):
            raise ClaimSchemaError("listing claim immutable identity changed")
        if len(candidate.versions) != len(current.versions) + 1:
            raise ClaimSchemaError("listing claim must append exactly one version")
        if candidate.versions[:-1] != current.versions:
            raise ClaimSchemaError("listing claim history is not append-only")
        disposition = "append"

    if candidate.current.idempotency_key != idempotency_key:
        raise IdempotencyConflict(
            "claim version and envelope idempotency keys must match"
        )
    if candidate.current.logical_tick != logical_time:
        raise ClaimSchemaError("claim logical_tick must equal World logical time")
    if disposition == "idempotent":
        return disposition

    for reference in candidate.current.evidence:
        record = evidence_lookup(reference.source_id, reference.source_digest)
        if record is None:
            raise ClaimSchemaError(
                f"claim evidence source {reference.source_id!r} is not persisted"
            )
        authorize_persisted_evidence_read(record, reader_id=original_actor)
        if record.record_digest != reference.source_digest:
            raise ClaimSchemaError("claim evidence source digest mismatch")
        if record.subject_id != candidate.subject:
            raise ClaimSchemaError("claim evidence subject binding mismatch")
        if reference.observed_at_tick < record.issued_at_tick:
            raise ClaimSchemaError(
                "claim evidence cannot be observed before its source record"
            )
    return disposition


_EXECUTABLE_CATALOG_ATTRIBUTE_KEYS = frozenset(
    {
        "pricing_tiers",
        "bundle_rules",
        "fee_rules",
    }
)


def reject_embedded_authority_records(attributes: Mapping[str, object]) -> None:
    """Reject authoritative or executable state hidden in public attributes.

    Listing attributes are descriptive, public catalog metadata.  Versioned
    authority records and executable pricing programs have dedicated World
    tables and must never acquire a second source of truth in the catalog.
    The walk is recursive so wrapping a forbidden program in display metadata
    does not bypass the boundary.
    """

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            executable_keys = sorted(
                key
                for key in value
                if isinstance(key, str)
                and key in _EXECUTABLE_CATALOG_ATTRIBUTE_KEYS
            )
            if executable_keys:
                raise WorldError(
                    "executable pricing programs must use World pricing tables, "
                    "not listing attributes: " + ", ".join(executable_keys)
                )
            schema_id = value.get("schema_id")
            schema_version = value.get("schema_version")
            if schema_id in {
                EVIDENCE_RECORD_SCHEMA_ID,
                MANDATE_REVISION_SCHEMA_ID,
                PRICING_POLICY_SCHEMA,
                PERSISTENT_CART_QUOTE_SCHEMA,
            } or schema_version == LISTING_CLAIM_SCHEMA:
                raise WorldError(
                    "authority records must use World tables, not listing attributes"
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(attributes)


def _trusted_service(actor_id: str) -> bool:
    return actor_id == "runtime" or actor_id.startswith(("runtime:", "platform:"))


__all__ = [
    "AppendDisposition",
    "authority_operation_key",
    "authorize_mandate_read",
    "authorize_persisted_evidence_read",
    "coerce_mandate_authority",
    "evidence_read_authority",
    "evidence_record_key",
    "mandate_revision_key",
    "mandate_authority_to_wire",
    "reject_embedded_authority_records",
    "validate_evidence_append",
    "validate_listing_claim_append",
    "validate_mandate_append",
    "validate_mandate_authority_registration",
]
