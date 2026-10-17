"""Replay-stable merchant claim and evidence contracts for catalog listings.

Listing attributes are descriptive World data.  They are not published claims.
Only an explicit :class:`ListingClaim` whose current state is ``published`` or
``corrected`` can be exposed as a merchant assertion.  The pure transitions in
this module create an append-only, digest-linked history that a later Platform
or World integration can persist without importing benchmark code.

The module deliberately owns no registry and performs no state writes.  A
caller must persist the returned claim atomically.  Exact idempotent retries
return the supplied persisted claim.  Reuse of an idempotency key with any
different request fact fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

from protocol.errors import IdempotencyConflict, SchemaError


CLAIM_EVIDENCE_SCHEMA = "cwe.claim-evidence.v1"
CLAIM_VERSION_SCHEMA = "cwe.claim-version.v1"
LISTING_CLAIM_SCHEMA = "cwe.listing-claim.v1"
CLAIM_REQUEST_SCHEMA = "cwe.claim-request.v1"
GENESIS_DIGEST = "0" * 64

ClaimStatus: TypeAlias = Literal["draft", "published", "corrected", "retracted"]
ClaimOperation: TypeAlias = Literal["draft", "publish", "correct", "retract"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]

_CLAIM_STATUSES = frozenset({"draft", "published", "corrected", "retracted"})
_CLAIM_OPERATIONS = frozenset({"draft", "publish", "correct", "retract"})
_STATE_BY_OPERATION: dict[str, ClaimStatus] = {
    "draft": "draft",
    "publish": "published",
    "correct": "corrected",
    "retract": "retracted",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EVIDENCE_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "evidence_id",
        "source_id",
        "claim_id",
        "listing_id",
        "merchant_id",
        "subject",
        "source_digest",
        "observed_at_tick",
        "evidence_digest",
    }
)
_VERSION_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "event_id",
        "claim_id",
        "listing_id",
        "merchant_id",
        "subject",
        "issuer_id",
        "operation",
        "state",
        "version",
        "previous_event_digest",
        "idempotency_key",
        "content",
        "content_digest",
        "evidence",
        "reason",
        "logical_tick",
        "request_digest",
        "event_digest",
    }
)
_CLAIM_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "claim_id",
        "listing_id",
        "merchant_id",
        "subject",
        "issuer_id",
        "state",
        "current_version",
        "current_event_digest",
        "versions",
    }
)


class ClaimSchemaError(SchemaError):
    """A claim, version, or evidence object violates its strict contract."""


class ClaimTransitionError(ClaimSchemaError):
    """A requested claim lifecycle transition is not allowed."""


class ClaimAuthorityError(ClaimTransitionError):
    """An actor other than the owning merchant attempted a mutation."""


@dataclass(frozen=True)
class ClaimEvidence:
    """One immutable reference to evidence for one listing claim subject."""

    evidence_id: str
    source_id: str
    claim_id: str
    listing_id: str
    merchant_id: str
    subject: str
    source_digest: str
    observed_at_tick: int
    evidence_digest: str = ""


@dataclass(frozen=True)
class ClaimVersion:
    """One append-only state transition in a listing claim's history."""

    event_id: str
    claim_id: str
    listing_id: str
    merchant_id: str
    subject: str
    issuer_id: str
    operation: ClaimOperation
    state: ClaimStatus
    version: int
    previous_event_digest: str
    idempotency_key: str
    content: Mapping[str, JsonValue]
    content_digest: str
    evidence: tuple[ClaimEvidence, ...]
    reason: str | None
    logical_tick: int
    request_digest: str
    event_digest: str = ""


ClaimEvent: TypeAlias = ClaimVersion


@dataclass(frozen=True)
class ListingClaim:
    """A claim identity and its complete append-only version history."""

    claim_id: str
    listing_id: str
    merchant_id: str
    subject: str
    issuer_id: str
    versions: tuple[ClaimVersion, ...]

    @property
    def current(self) -> ClaimVersion:
        return self.versions[-1]

    @property
    def state(self) -> ClaimStatus:
        return self.current.state

    @property
    def version(self) -> int:
        return self.current.version

    @property
    def content_digest(self) -> str:
        return self.current.content_digest

    @property
    def logical_tick(self) -> int:
        return self.current.logical_tick


def canonical_claim_digest(value: Any) -> str:
    """Hash one strict JSON value using a replay-stable canonical encoding."""

    frozen = _freeze_json(value, path="value")
    body = _canonical_json(_thaw_json(frozen)).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def claim_content_digest(content: Mapping[str, Any]) -> str:
    """Return the canonical digest of a non-empty claim content object."""

    frozen = _freeze_content(content)
    return canonical_claim_digest(_thaw_json(frozen))


def seal_claim_evidence(evidence: ClaimEvidence) -> ClaimEvidence:
    """Validate and seal evidence with a digest over all source bindings."""

    _validate_evidence_fields(evidence, require_digest=False)
    sealed = ClaimEvidence(
        evidence_id=evidence.evidence_id,
        source_id=evidence.source_id,
        claim_id=evidence.claim_id,
        listing_id=evidence.listing_id,
        merchant_id=evidence.merchant_id,
        subject=evidence.subject,
        source_digest=evidence.source_digest,
        observed_at_tick=evidence.observed_at_tick,
        evidence_digest=canonical_claim_digest(_evidence_contract(evidence)),
    )
    validate_claim_evidence(sealed)
    return sealed


def validate_claim_evidence(evidence: ClaimEvidence) -> None:
    """Validate evidence bindings and its canonical digest."""

    _validate_evidence_fields(evidence, require_digest=True)
    expected = canonical_claim_digest(_evidence_contract(evidence))
    if evidence.evidence_digest != expected:
        raise ClaimSchemaError("claim evidence digest mismatch")


def claim_evidence_to_wire(evidence: ClaimEvidence) -> dict[str, Any]:
    """Return a fresh strict JSON-compatible evidence object."""

    validate_claim_evidence(evidence)
    return {**_evidence_contract(evidence), "evidence_digest": evidence.evidence_digest}


def coerce_claim_evidence(value: Any) -> ClaimEvidence:
    """Coerce and validate one exact evidence wire object."""

    if isinstance(value, ClaimEvidence):
        validate_claim_evidence(value)
        return value
    row = _mapping(value, "claim evidence")
    _require_exact_keys(row, _EVIDENCE_WIRE_KEYS, "claim evidence")
    _require_schema(row, CLAIM_EVIDENCE_SCHEMA)
    evidence = ClaimEvidence(
        evidence_id=_wire_text(row, "evidence_id"),
        source_id=_wire_text(row, "source_id"),
        claim_id=_wire_text(row, "claim_id"),
        listing_id=_wire_text(row, "listing_id"),
        merchant_id=_wire_text(row, "merchant_id"),
        subject=_wire_text(row, "subject"),
        source_digest=_wire_digest(row, "source_digest"),
        observed_at_tick=_wire_int(row, "observed_at_tick"),
        evidence_digest=_wire_digest(row, "evidence_digest"),
    )
    validate_claim_evidence(evidence)
    return evidence


def create_listing_claim_draft(
    *,
    claim_id: str,
    listing_id: str,
    merchant_id: str,
    subject: str,
    issuer_id: str,
    content: Mapping[str, Any],
    evidence: Sequence[ClaimEvidence] = (),
    logical_tick: int,
    idempotency_key: str,
    existing_claim: ListingClaim | None = None,
) -> ListingClaim:
    """Create a draft, or return an exact previously persisted draft retry."""

    for label, value in (
        ("claim_id", claim_id),
        ("listing_id", listing_id),
        ("merchant_id", merchant_id),
        ("subject", subject),
        ("issuer_id", issuer_id),
        ("idempotency_key", idempotency_key),
    ):
        _require_text(value, label)
    if issuer_id != merchant_id:
        raise ClaimAuthorityError("claim issuer must be the owning merchant")
    _require_nonnegative_int(logical_tick, "logical_tick")
    frozen_content = _freeze_content(content)
    normalized_evidence = _normalize_evidence(
        evidence,
        claim_id=claim_id,
        listing_id=listing_id,
        merchant_id=merchant_id,
        subject=subject,
        logical_tick=logical_tick,
    )
    request_digest = _request_digest(
        operation="draft",
        claim_id=claim_id,
        listing_id=listing_id,
        merchant_id=merchant_id,
        subject=subject,
        issuer_id=issuer_id,
        idempotency_key=idempotency_key,
        content=frozen_content,
        evidence=normalized_evidence,
        reason=None,
        logical_tick=logical_tick,
    )
    if existing_claim is not None:
        validate_listing_claim(existing_claim)
        _require_claim_identity(
            existing_claim,
            claim_id=claim_id,
            listing_id=listing_id,
            merchant_id=merchant_id,
            subject=subject,
            issuer_id=issuer_id,
        )
        return _return_retry_or_conflict(
            existing_claim,
            idempotency_key=idempotency_key,
            operation="draft",
            request_digest=request_digest,
        )

    version = _build_version(
        claim_id=claim_id,
        listing_id=listing_id,
        merchant_id=merchant_id,
        subject=subject,
        issuer_id=issuer_id,
        operation="draft",
        version=1,
        previous_event_digest=GENESIS_DIGEST,
        idempotency_key=idempotency_key,
        content=frozen_content,
        evidence=normalized_evidence,
        reason=None,
        logical_tick=logical_tick,
        request_digest=request_digest,
    )
    claim = ListingClaim(
        claim_id=claim_id,
        listing_id=listing_id,
        merchant_id=merchant_id,
        subject=subject,
        issuer_id=issuer_id,
        versions=(version,),
    )
    validate_listing_claim(claim)
    return claim


def publish_claim(
    claim: ListingClaim,
    *,
    actor_id: str,
    evidence: Sequence[ClaimEvidence],
    logical_tick: int,
    idempotency_key: str,
) -> ListingClaim:
    """Publish a draft using explicit matched evidence."""

    validate_listing_claim(claim)
    _require_owner(claim, actor_id)
    _require_text(idempotency_key, "idempotency_key")
    _require_nonnegative_int(logical_tick, "logical_tick")
    existing = _version_for_idempotency_key(claim, idempotency_key)
    content = existing.content if existing is not None else claim.current.content
    normalized_evidence = _normalize_evidence_for_claim(claim, evidence, logical_tick)
    request_digest = _request_digest_for_claim(
        claim,
        operation="publish",
        idempotency_key=idempotency_key,
        content=content,
        evidence=normalized_evidence,
        reason=None,
        logical_tick=logical_tick,
    )
    if existing is not None:
        return _return_retry_or_conflict(
            claim,
            idempotency_key=idempotency_key,
            operation="publish",
            request_digest=request_digest,
        )
    if claim.state != "draft":
        raise ClaimTransitionError("only a draft claim can be published")
    if not normalized_evidence:
        raise ClaimTransitionError("publishing a claim requires evidence")
    return _append_version(
        claim,
        operation="publish",
        idempotency_key=idempotency_key,
        content=content,
        evidence=normalized_evidence,
        reason=None,
        logical_tick=logical_tick,
        request_digest=request_digest,
    )


def correct_claim(
    claim: ListingClaim,
    *,
    actor_id: str,
    content: Mapping[str, Any],
    evidence: Sequence[ClaimEvidence],
    logical_tick: int,
    idempotency_key: str,
) -> ListingClaim:
    """Append a corrected assertion while preserving all prior provenance."""

    validate_listing_claim(claim)
    _require_owner(claim, actor_id)
    _require_text(idempotency_key, "idempotency_key")
    _require_nonnegative_int(logical_tick, "logical_tick")
    frozen_content = _freeze_content(content)
    normalized_evidence = _normalize_evidence_for_claim(claim, evidence, logical_tick)
    request_digest = _request_digest_for_claim(
        claim,
        operation="correct",
        idempotency_key=idempotency_key,
        content=frozen_content,
        evidence=normalized_evidence,
        reason=None,
        logical_tick=logical_tick,
    )
    existing = _version_for_idempotency_key(claim, idempotency_key)
    if existing is not None:
        return _return_retry_or_conflict(
            claim,
            idempotency_key=idempotency_key,
            operation="correct",
            request_digest=request_digest,
        )
    if claim.state not in {"published", "corrected"}:
        raise ClaimTransitionError("only a published claim can be corrected")
    if not normalized_evidence:
        raise ClaimTransitionError("correcting a claim requires evidence")
    if claim_content_digest(frozen_content) == claim.current.content_digest:
        raise ClaimTransitionError("a correction must change canonical claim content")
    return _append_version(
        claim,
        operation="correct",
        idempotency_key=idempotency_key,
        content=frozen_content,
        evidence=normalized_evidence,
        reason=None,
        logical_tick=logical_tick,
        request_digest=request_digest,
    )


def retract_claim(
    claim: ListingClaim,
    *,
    actor_id: str,
    reason: str,
    evidence: Sequence[ClaimEvidence] = (),
    logical_tick: int,
    idempotency_key: str,
) -> ListingClaim:
    """Retract a public assertion without deleting its content or evidence."""

    validate_listing_claim(claim)
    _require_owner(claim, actor_id)
    _require_text(reason, "reason")
    _require_text(idempotency_key, "idempotency_key")
    _require_nonnegative_int(logical_tick, "logical_tick")
    existing = _version_for_idempotency_key(claim, idempotency_key)
    content = existing.content if existing is not None else claim.current.content
    normalized_evidence = _normalize_evidence_for_claim(claim, evidence, logical_tick)
    request_digest = _request_digest_for_claim(
        claim,
        operation="retract",
        idempotency_key=idempotency_key,
        content=content,
        evidence=normalized_evidence,
        reason=reason,
        logical_tick=logical_tick,
    )
    if existing is not None:
        return _return_retry_or_conflict(
            claim,
            idempotency_key=idempotency_key,
            operation="retract",
            request_digest=request_digest,
        )
    if claim.state not in {"published", "corrected"}:
        raise ClaimTransitionError("only a published claim can be retracted")
    return _append_version(
        claim,
        operation="retract",
        idempotency_key=idempotency_key,
        content=content,
        evidence=normalized_evidence,
        reason=reason,
        logical_tick=logical_tick,
        request_digest=request_digest,
    )


def published_claim_content(claim: ListingClaim) -> dict[str, Any] | None:
    """Return public claim content, never raw listing attributes or a draft."""

    validate_listing_claim(claim)
    if claim.state not in {"published", "corrected"}:
        return None
    return cast(dict[str, Any], _thaw_json(claim.current.content))


def claim_version_to_wire(version: ClaimVersion) -> dict[str, Any]:
    """Return a fresh JSON-compatible version after full validation."""

    validate_claim_version(version)
    return {
        **_version_contract(version),
        "event_digest": version.event_digest,
    }


def coerce_claim_version(value: Any) -> ClaimVersion:
    """Coerce and validate one exact version wire object."""

    if isinstance(value, ClaimVersion):
        validate_claim_version(value)
        return value
    row = _mapping(value, "claim version")
    _require_exact_keys(row, _VERSION_WIRE_KEYS, "claim version")
    _require_schema(row, CLAIM_VERSION_SCHEMA)
    raw_evidence = row["evidence"]
    if not isinstance(raw_evidence, (list, tuple)):
        raise ClaimSchemaError("claim version evidence must be an array")
    operation_text = _wire_text(row, "operation")
    state_text = _wire_text(row, "state")
    if operation_text not in _CLAIM_OPERATIONS:
        raise ClaimSchemaError(f"unsupported claim operation: {operation_text!r}")
    if state_text not in _CLAIM_STATUSES:
        raise ClaimSchemaError(f"unsupported claim state: {state_text!r}")
    reason = row["reason"]
    if reason is not None and not isinstance(reason, str):
        raise ClaimSchemaError("reason must be a string or null")
    version = ClaimVersion(
        event_id=_wire_text(row, "event_id"),
        claim_id=_wire_text(row, "claim_id"),
        listing_id=_wire_text(row, "listing_id"),
        merchant_id=_wire_text(row, "merchant_id"),
        subject=_wire_text(row, "subject"),
        issuer_id=_wire_text(row, "issuer_id"),
        operation=cast(ClaimOperation, operation_text),
        state=cast(ClaimStatus, state_text),
        version=_wire_int(row, "version"),
        previous_event_digest=_wire_digest(row, "previous_event_digest"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        content=_freeze_content(_mapping(row["content"], "claim content")),
        content_digest=_wire_digest(row, "content_digest"),
        evidence=tuple(coerce_claim_evidence(item) for item in raw_evidence),
        reason=reason,
        logical_tick=_wire_int(row, "logical_tick"),
        request_digest=_wire_digest(row, "request_digest"),
        event_digest=_wire_digest(row, "event_digest"),
    )
    validate_claim_version(version)
    return version


def listing_claim_to_wire(claim: ListingClaim) -> dict[str, Any]:
    """Serialize a complete claim history for durable storage or replay."""

    validate_listing_claim(claim)
    return {
        "schema_version": LISTING_CLAIM_SCHEMA,
        "claim_id": claim.claim_id,
        "listing_id": claim.listing_id,
        "merchant_id": claim.merchant_id,
        "subject": claim.subject,
        "issuer_id": claim.issuer_id,
        "state": claim.state,
        "current_version": claim.version,
        "current_event_digest": claim.current.event_digest,
        "versions": [claim_version_to_wire(version) for version in claim.versions],
    }


def coerce_listing_claim(value: Any) -> ListingClaim:
    """Coerce strict persisted claim data and verify the complete digest chain."""

    if isinstance(value, ListingClaim):
        validate_listing_claim(value)
        return value
    row = _mapping(value, "listing claim")
    _require_exact_keys(row, _CLAIM_WIRE_KEYS, "listing claim")
    _require_schema(row, LISTING_CLAIM_SCHEMA)
    raw_versions = row["versions"]
    if not isinstance(raw_versions, (list, tuple)):
        raise ClaimSchemaError("listing claim versions must be an array")
    state_text = _wire_text(row, "state")
    if state_text not in _CLAIM_STATUSES:
        raise ClaimSchemaError(f"unsupported claim state: {state_text!r}")
    persisted_version = _wire_int(row, "current_version")
    persisted_digest = _wire_digest(row, "current_event_digest")
    claim = ListingClaim(
        claim_id=_wire_text(row, "claim_id"),
        listing_id=_wire_text(row, "listing_id"),
        merchant_id=_wire_text(row, "merchant_id"),
        subject=_wire_text(row, "subject"),
        issuer_id=_wire_text(row, "issuer_id"),
        versions=tuple(coerce_claim_version(item) for item in raw_versions),
    )
    validate_listing_claim(claim)
    if claim.state != state_text:
        raise ClaimSchemaError("persisted claim state does not match history")
    if claim.version != persisted_version:
        raise ClaimSchemaError("persisted current_version does not match history")
    if claim.current.event_digest != persisted_digest:
        raise ClaimSchemaError("persisted current_event_digest does not match history")
    return claim


def listing_claim_to_json(claim: ListingClaim) -> str:
    """Serialize one claim as canonical JSON."""

    return _canonical_json(listing_claim_to_wire(claim))


def listing_claim_from_json(payload: str) -> ListingClaim:
    """Parse strict JSON and reject duplicate object keys at every depth."""

    if not isinstance(payload, str):
        raise ClaimSchemaError("listing claim JSON must be a string")
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise ClaimSchemaError(f"invalid listing claim JSON: {exc}") from exc
    return coerce_listing_claim(value)


def validate_claim_version(version: ClaimVersion) -> None:
    """Validate one independently persisted version and all evidence."""

    if not isinstance(version, ClaimVersion):
        raise ClaimSchemaError("claim version must be ClaimVersion")
    for field in (
        "event_id",
        "claim_id",
        "listing_id",
        "merchant_id",
        "subject",
        "issuer_id",
        "idempotency_key",
    ):
        _require_text(getattr(version, field), field)
    if version.operation not in _CLAIM_OPERATIONS:
        raise ClaimSchemaError(f"unsupported claim operation: {version.operation!r}")
    if version.state not in _CLAIM_STATUSES:
        raise ClaimSchemaError(f"unsupported claim state: {version.state!r}")
    if version.state != _STATE_BY_OPERATION[version.operation]:
        raise ClaimSchemaError("claim version state does not match operation")
    _require_positive_int(version.version, "version")
    _require_digest(version.previous_event_digest, "previous_event_digest")
    _require_nonnegative_int(version.logical_tick, "logical_tick")
    _validate_reason(version.operation, version.reason)
    frozen_content = _freeze_content(version.content)
    expected_content_digest = claim_content_digest(frozen_content)
    _require_digest(version.content_digest, "content_digest")
    if version.content_digest != expected_content_digest:
        raise ClaimSchemaError("claim content digest mismatch")
    if not isinstance(version.evidence, tuple):
        raise ClaimSchemaError("claim version evidence must be a tuple")
    evidence_ids: set[str] = set()
    source_ids: set[str] = set()
    for evidence in version.evidence:
        validate_claim_evidence(evidence)
        _validate_evidence_binding(evidence, version)
        if evidence.evidence_id in evidence_ids:
            raise ClaimSchemaError("duplicate evidence_id in claim version")
        if evidence.source_id in source_ids:
            raise ClaimSchemaError("duplicate source_id in claim version")
        evidence_ids.add(evidence.evidence_id)
        source_ids.add(evidence.source_id)
        if evidence.observed_at_tick > version.logical_tick:
            raise ClaimSchemaError("claim evidence cannot be observed in the future")
    if tuple(sorted(version.evidence, key=lambda item: item.evidence_id)) != version.evidence:
        raise ClaimSchemaError("claim version evidence must use canonical evidence_id order")
    if version.operation in {"publish", "correct"} and not version.evidence:
        raise ClaimSchemaError(f"{version.operation} claim version requires evidence")
    expected_request = _request_digest(
        operation=version.operation,
        claim_id=version.claim_id,
        listing_id=version.listing_id,
        merchant_id=version.merchant_id,
        subject=version.subject,
        issuer_id=version.issuer_id,
        idempotency_key=version.idempotency_key,
        content=version.content,
        evidence=version.evidence,
        reason=version.reason,
        logical_tick=version.logical_tick,
    )
    _require_digest(version.request_digest, "request_digest")
    if version.request_digest != expected_request:
        raise ClaimSchemaError("claim request digest mismatch")
    if version.event_id != _event_id(version.request_digest):
        raise ClaimSchemaError("claim event_id is not deterministic")
    _require_digest(version.event_digest, "event_digest")
    expected_event = canonical_claim_digest(_version_contract(version))
    if version.event_digest != expected_event:
        raise ClaimSchemaError("claim event digest mismatch")


def validate_listing_claim(claim: ListingClaim) -> None:
    """Verify claim ownership, append-only ordering, and the full digest chain."""

    if not isinstance(claim, ListingClaim):
        raise ClaimSchemaError("listing claim must be ListingClaim")
    for field in ("claim_id", "listing_id", "merchant_id", "subject", "issuer_id"):
        _require_text(getattr(claim, field), field)
    if claim.issuer_id != claim.merchant_id:
        raise ClaimAuthorityError("claim issuer must be the owning merchant")
    if not isinstance(claim.versions, tuple) or not claim.versions:
        raise ClaimSchemaError("listing claim must have a non-empty version tuple")

    event_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    evidence_ids: set[str] = set()
    source_ids: set[str] = set()
    previous: ClaimVersion | None = None
    for index, version in enumerate(claim.versions, start=1):
        validate_claim_version(version)
        _validate_version_binding(version, claim)
        if version.version != index:
            raise ClaimSchemaError("claim versions must be contiguous from one")
        if version.event_id in event_ids:
            raise ClaimSchemaError("duplicate claim event_id")
        if version.idempotency_key in idempotency_keys:
            raise ClaimSchemaError("duplicate claim idempotency_key")
        event_ids.add(version.event_id)
        idempotency_keys.add(version.idempotency_key)
        for evidence in version.evidence:
            if evidence.evidence_id in evidence_ids:
                raise ClaimSchemaError("duplicate evidence_id in claim history")
            if evidence.source_id in source_ids:
                raise ClaimSchemaError("duplicate source_id in claim history")
            evidence_ids.add(evidence.evidence_id)
            source_ids.add(evidence.source_id)

        if previous is None:
            if version.operation != "draft" or version.state != "draft":
                raise ClaimSchemaError("claim history must begin with a draft")
            if version.previous_event_digest != GENESIS_DIGEST:
                raise ClaimSchemaError("draft must use the genesis previous digest")
        else:
            if version.previous_event_digest != previous.event_digest:
                raise ClaimSchemaError("claim history previous digest mismatch")
            if version.logical_tick <= previous.logical_tick:
                raise ClaimSchemaError("claim logical ticks must strictly increase")
            _validate_history_transition(previous, version)
        previous = version


def _append_version(
    claim: ListingClaim,
    *,
    operation: ClaimOperation,
    idempotency_key: str,
    content: Mapping[str, JsonValue],
    evidence: tuple[ClaimEvidence, ...],
    reason: str | None,
    logical_tick: int,
    request_digest: str,
) -> ListingClaim:
    if logical_tick <= claim.logical_tick:
        raise ClaimTransitionError("claim logical_tick must advance")
    version = _build_version(
        claim_id=claim.claim_id,
        listing_id=claim.listing_id,
        merchant_id=claim.merchant_id,
        subject=claim.subject,
        issuer_id=claim.issuer_id,
        operation=operation,
        version=claim.version + 1,
        previous_event_digest=claim.current.event_digest,
        idempotency_key=idempotency_key,
        content=content,
        evidence=evidence,
        reason=reason,
        logical_tick=logical_tick,
        request_digest=request_digest,
    )
    updated = ListingClaim(
        claim_id=claim.claim_id,
        listing_id=claim.listing_id,
        merchant_id=claim.merchant_id,
        subject=claim.subject,
        issuer_id=claim.issuer_id,
        versions=claim.versions + (version,),
    )
    validate_listing_claim(updated)
    return updated


def _build_version(
    *,
    claim_id: str,
    listing_id: str,
    merchant_id: str,
    subject: str,
    issuer_id: str,
    operation: ClaimOperation,
    version: int,
    previous_event_digest: str,
    idempotency_key: str,
    content: Mapping[str, JsonValue],
    evidence: tuple[ClaimEvidence, ...],
    reason: str | None,
    logical_tick: int,
    request_digest: str,
) -> ClaimVersion:
    unsigned = ClaimVersion(
        event_id=_event_id(request_digest),
        claim_id=claim_id,
        listing_id=listing_id,
        merchant_id=merchant_id,
        subject=subject,
        issuer_id=issuer_id,
        operation=operation,
        state=_STATE_BY_OPERATION[operation],
        version=version,
        previous_event_digest=previous_event_digest,
        idempotency_key=idempotency_key,
        content=content,
        content_digest=claim_content_digest(content),
        evidence=evidence,
        reason=reason,
        logical_tick=logical_tick,
        request_digest=request_digest,
    )
    sealed = ClaimVersion(
        event_id=unsigned.event_id,
        claim_id=unsigned.claim_id,
        listing_id=unsigned.listing_id,
        merchant_id=unsigned.merchant_id,
        subject=unsigned.subject,
        issuer_id=unsigned.issuer_id,
        operation=unsigned.operation,
        state=unsigned.state,
        version=unsigned.version,
        previous_event_digest=unsigned.previous_event_digest,
        idempotency_key=unsigned.idempotency_key,
        content=unsigned.content,
        content_digest=unsigned.content_digest,
        evidence=unsigned.evidence,
        reason=unsigned.reason,
        logical_tick=unsigned.logical_tick,
        request_digest=unsigned.request_digest,
        event_digest=canonical_claim_digest(_version_contract(unsigned)),
    )
    validate_claim_version(sealed)
    return sealed


def _request_digest_for_claim(
    claim: ListingClaim,
    *,
    operation: ClaimOperation,
    idempotency_key: str,
    content: Mapping[str, JsonValue],
    evidence: tuple[ClaimEvidence, ...],
    reason: str | None,
    logical_tick: int,
) -> str:
    return _request_digest(
        operation=operation,
        claim_id=claim.claim_id,
        listing_id=claim.listing_id,
        merchant_id=claim.merchant_id,
        subject=claim.subject,
        issuer_id=claim.issuer_id,
        idempotency_key=idempotency_key,
        content=content,
        evidence=evidence,
        reason=reason,
        logical_tick=logical_tick,
    )


def _request_digest(
    *,
    operation: ClaimOperation,
    claim_id: str,
    listing_id: str,
    merchant_id: str,
    subject: str,
    issuer_id: str,
    idempotency_key: str,
    content: Mapping[str, JsonValue],
    evidence: tuple[ClaimEvidence, ...],
    reason: str | None,
    logical_tick: int,
) -> str:
    return canonical_claim_digest(
        {
            "schema_version": CLAIM_REQUEST_SCHEMA,
            "operation": operation,
            "claim_id": claim_id,
            "listing_id": listing_id,
            "merchant_id": merchant_id,
            "subject": subject,
            "issuer_id": issuer_id,
            "idempotency_key": idempotency_key,
            "content_digest": claim_content_digest(content),
            "evidence_digests": [item.evidence_digest for item in evidence],
            "reason": reason,
            "logical_tick": logical_tick,
        }
    )


def _return_retry_or_conflict(
    claim: ListingClaim,
    *,
    idempotency_key: str,
    operation: ClaimOperation,
    request_digest: str,
) -> ListingClaim:
    existing = _version_for_idempotency_key(claim, idempotency_key)
    if existing is None:
        raise IdempotencyConflict("claim does not contain the supplied idempotency key")
    if existing.operation != operation or existing.request_digest != request_digest:
        raise IdempotencyConflict(
            "claim idempotency key was reused with a different request"
        )
    return claim


def _version_for_idempotency_key(
    claim: ListingClaim, idempotency_key: str
) -> ClaimVersion | None:
    matches = [
        version for version in claim.versions if version.idempotency_key == idempotency_key
    ]
    if len(matches) > 1:
        raise ClaimSchemaError("duplicate claim idempotency_key")
    return matches[0] if matches else None


def _normalize_evidence_for_claim(
    claim: ListingClaim,
    evidence: Sequence[ClaimEvidence],
    logical_tick: int,
) -> tuple[ClaimEvidence, ...]:
    return _normalize_evidence(
        evidence,
        claim_id=claim.claim_id,
        listing_id=claim.listing_id,
        merchant_id=claim.merchant_id,
        subject=claim.subject,
        logical_tick=logical_tick,
    )


def _normalize_evidence(
    evidence: Sequence[ClaimEvidence],
    *,
    claim_id: str,
    listing_id: str,
    merchant_id: str,
    subject: str,
    logical_tick: int,
) -> tuple[ClaimEvidence, ...]:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise ClaimSchemaError("claim evidence must be a sequence")
    normalized = tuple(sorted((coerce_claim_evidence(item) for item in evidence), key=lambda item: item.evidence_id))
    evidence_ids: set[str] = set()
    source_ids: set[str] = set()
    for item in normalized:
        if item.claim_id != claim_id:
            raise ClaimSchemaError("claim evidence claim_id mismatch")
        if item.listing_id != listing_id:
            raise ClaimSchemaError("claim evidence listing_id mismatch")
        if item.merchant_id != merchant_id:
            raise ClaimSchemaError("claim evidence merchant_id mismatch")
        if item.subject != subject:
            raise ClaimSchemaError("claim evidence subject mismatch")
        if item.observed_at_tick > logical_tick:
            raise ClaimSchemaError("claim evidence cannot be observed in the future")
        if item.evidence_id in evidence_ids:
            raise ClaimSchemaError("duplicate evidence_id in claim version")
        if item.source_id in source_ids:
            raise ClaimSchemaError("duplicate source_id in claim version")
        evidence_ids.add(item.evidence_id)
        source_ids.add(item.source_id)
    return normalized


def _evidence_contract(evidence: ClaimEvidence) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_EVIDENCE_SCHEMA,
        "evidence_id": evidence.evidence_id,
        "source_id": evidence.source_id,
        "claim_id": evidence.claim_id,
        "listing_id": evidence.listing_id,
        "merchant_id": evidence.merchant_id,
        "subject": evidence.subject,
        "source_digest": evidence.source_digest,
        "observed_at_tick": evidence.observed_at_tick,
    }


def _version_contract(version: ClaimVersion) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_VERSION_SCHEMA,
        "event_id": version.event_id,
        "claim_id": version.claim_id,
        "listing_id": version.listing_id,
        "merchant_id": version.merchant_id,
        "subject": version.subject,
        "issuer_id": version.issuer_id,
        "operation": version.operation,
        "state": version.state,
        "version": version.version,
        "previous_event_digest": version.previous_event_digest,
        "idempotency_key": version.idempotency_key,
        "content": _thaw_json(version.content),
        "content_digest": version.content_digest,
        "evidence": [claim_evidence_to_wire(item) for item in version.evidence],
        "reason": version.reason,
        "logical_tick": version.logical_tick,
        "request_digest": version.request_digest,
    }


def _validate_evidence_fields(
    evidence: ClaimEvidence, *, require_digest: bool
) -> None:
    if not isinstance(evidence, ClaimEvidence):
        raise ClaimSchemaError("claim evidence must be ClaimEvidence")
    for field in (
        "evidence_id",
        "source_id",
        "claim_id",
        "listing_id",
        "merchant_id",
        "subject",
    ):
        _require_text(getattr(evidence, field), field)
    _require_digest(evidence.source_digest, "source_digest")
    _require_nonnegative_int(evidence.observed_at_tick, "observed_at_tick")
    if require_digest:
        _require_digest(evidence.evidence_digest, "evidence_digest")


def _validate_evidence_binding(
    evidence: ClaimEvidence, version: ClaimVersion
) -> None:
    for field in ("claim_id", "listing_id", "merchant_id", "subject"):
        if getattr(evidence, field) != getattr(version, field):
            raise ClaimSchemaError(f"claim evidence {field} mismatch")


def _validate_version_binding(version: ClaimVersion, claim: ListingClaim) -> None:
    for field in ("claim_id", "listing_id", "merchant_id", "subject", "issuer_id"):
        if getattr(version, field) != getattr(claim, field):
            raise ClaimSchemaError(f"claim version {field} mismatch")


def _validate_history_transition(previous: ClaimVersion, current: ClaimVersion) -> None:
    allowed: dict[ClaimStatus, frozenset[ClaimOperation]] = {
        "draft": frozenset({"publish"}),
        "published": frozenset({"correct", "retract"}),
        "corrected": frozenset({"correct", "retract"}),
        "retracted": frozenset(),
    }
    if current.operation not in allowed[previous.state]:
        raise ClaimSchemaError(
            f"invalid claim transition: {previous.state} -> {current.operation}"
        )
    if current.operation == "publish" and current.content_digest != previous.content_digest:
        raise ClaimSchemaError("publish must preserve draft claim content")
    if current.operation == "correct" and current.content_digest == previous.content_digest:
        raise ClaimSchemaError("correction must change canonical claim content")
    if current.operation == "retract" and current.content_digest != previous.content_digest:
        raise ClaimSchemaError("retraction must preserve the prior claim content")


def _validate_reason(operation: ClaimOperation, reason: str | None) -> None:
    if operation == "retract":
        _require_text(reason, "reason")
    elif reason is not None:
        raise ClaimSchemaError("reason is only allowed on claim retraction")


def _require_owner(claim: ListingClaim, actor_id: str) -> None:
    _require_text(actor_id, "actor_id")
    if actor_id != claim.merchant_id:
        raise ClaimAuthorityError("only the owning merchant can modify a listing claim")


def _require_claim_identity(
    claim: ListingClaim,
    *,
    claim_id: str,
    listing_id: str,
    merchant_id: str,
    subject: str,
    issuer_id: str,
) -> None:
    expected = {
        "claim_id": claim_id,
        "listing_id": listing_id,
        "merchant_id": merchant_id,
        "subject": subject,
        "issuer_id": issuer_id,
    }
    for field, value in expected.items():
        if getattr(claim, field) != value:
            raise IdempotencyConflict(f"claim identity changed under retry: {field}")


def _freeze_content(value: Mapping[str, Any]) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ClaimSchemaError("claim content must be an object")
    frozen = _freeze_json(value, path="content")
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise ClaimSchemaError("claim content must be an object")
    if not frozen:
        raise ClaimSchemaError("claim content must not be empty")
    return frozen


def _freeze_json(value: Any, *, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ClaimSchemaError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ClaimSchemaError(f"{path} contains a non-string object key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ClaimSchemaError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ClaimSchemaError(f"claim value is not canonical JSON: {exc}") from exc


def _event_id(request_digest: str) -> str:
    _require_digest(request_digest, "request_digest")
    return f"claim-event:{request_digest[:32]}"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaimSchemaError(f"{label} must be an object")
    return value


def _require_schema(row: Mapping[str, Any], expected: str) -> None:
    if row.get("schema_version") != expected:
        raise ClaimSchemaError(f"unsupported schema_version for {expected}")


def _require_exact_keys(
    row: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(row.keys())
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise ClaimSchemaError(f"{label} has invalid fields: {', '.join(details)}")


def _wire_text(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_text(value, key)
    return cast(str, value)


def _wire_int(row: Mapping[str, Any], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClaimSchemaError(f"{key} must be an integer")
    return cast(int, value)


def _wire_digest(row: Mapping[str, Any], key: str) -> str:
    value = row[key]
    _require_digest(value, key)
    return cast(str, value)


def _require_text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ClaimSchemaError(f"{label} must be a non-empty string")


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ClaimSchemaError(f"{label} must be a lowercase SHA-256 hex digest")


def _require_positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClaimSchemaError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClaimSchemaError(f"{label} must be a non-negative integer")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClaimSchemaError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


__all__ = [
    "CLAIM_EVIDENCE_SCHEMA",
    "CLAIM_REQUEST_SCHEMA",
    "CLAIM_VERSION_SCHEMA",
    "GENESIS_DIGEST",
    "LISTING_CLAIM_SCHEMA",
    "ClaimAuthorityError",
    "ClaimEvent",
    "ClaimEvidence",
    "ClaimOperation",
    "ClaimSchemaError",
    "ClaimStatus",
    "ClaimTransitionError",
    "ClaimVersion",
    "ListingClaim",
    "canonical_claim_digest",
    "claim_content_digest",
    "claim_evidence_to_wire",
    "claim_version_to_wire",
    "coerce_claim_evidence",
    "coerce_claim_version",
    "coerce_listing_claim",
    "correct_claim",
    "create_listing_claim_draft",
    "listing_claim_from_json",
    "listing_claim_to_json",
    "listing_claim_to_wire",
    "publish_claim",
    "published_claim_content",
    "retract_claim",
    "seal_claim_evidence",
    "validate_claim_evidence",
    "validate_claim_version",
    "validate_listing_claim",
]
