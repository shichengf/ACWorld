"""Core contracts for persisted evidence and principal-authorized mandates.

These objects belong to the CommerceWorld domain protocol.  An
:class:`EvidenceRecord` is an independently persisted record with its own
owner, read access-control list, issuer, version, logical issuance time, and
content digest.  A :class:`MandateRevision` is a principal-authored change to
one buyer mandate, linked to the previous revision by digest.

Neither object is a catalog listing and neither may be encoded as ad-hoc
listing attributes.  Platform and World integration can persist their
canonical JSON, expose them through authorized read APIs, and replay the same
bytes without importing a benchmark task module.

The access helpers intentionally require trusted bindings supplied by the
Runtime or World store.  Constructing those bindings from the untrusted record
being checked would defeat the issuer and ACL checks and is explicitly not a
supported use.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, TypeAlias, TypedDict, cast

from protocol.errors import SchemaError


EVIDENCE_RECORD_SCHEMA_ID = "cwe.evidence-record.v1"
MANDATE_REVISION_SCHEMA_ID = "cwe.mandate-revision.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
MandateRevisionDisposition: TypeAlias = Literal["append", "idempotent"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EVIDENCE_WIRE_FIELDS = frozenset(
    {
        "schema_id",
        "record_id",
        "kind",
        "subject_id",
        "issuer_id",
        "facts",
        "trust",
        "version",
        "owner_id",
        "read_acl",
        "issued_at_tick",
        "record_digest",
    }
)
_MANDATE_WIRE_FIELDS = frozenset(
    {
        "schema_id",
        "principal_id",
        "buyer_id",
        "mandate_id",
        "revision",
        "previous_digest",
        "changes",
        "authorized_fields",
        "logical_tick",
        "revision_digest",
    }
)


class EvidenceRecordSchemaError(SchemaError):
    """An evidence record is not an exact, valid v1 core object."""


class EvidenceRecordDigestMismatch(EvidenceRecordSchemaError):
    """An evidence record digest does not match its canonical content."""


class EvidenceReadAuthorityError(EvidenceRecordSchemaError):
    """Trusted issuer, owner, ACL, version, or persistence binding differs."""


class EvidenceReadDenied(EvidenceReadAuthorityError):
    """The requesting actor is neither the owner nor an ACL member."""


class MandateRevisionSchemaError(SchemaError):
    """A mandate revision is not an exact, valid v1 core object."""


class MandateRevisionDigestMismatch(MandateRevisionSchemaError):
    """A mandate revision digest does not match its canonical content."""


class MandateRevisionAuthorityError(MandateRevisionSchemaError):
    """A revision is not authorized for the bound principal and buyer."""


class MandateRevisionOrderError(MandateRevisionSchemaError):
    """A revision is stale, skips a version, or breaks the digest chain."""


class MandateRevisionConflict(MandateRevisionOrderError):
    """One revision number was reused for different canonical content."""


class EvidenceRecordWire(TypedDict):
    """Exact JSON-compatible wire shape of :class:`EvidenceRecord`."""

    schema_id: str
    record_id: str
    kind: str
    subject_id: str
    issuer_id: str
    facts: dict[str, Any]
    trust: dict[str, Any]
    version: int
    owner_id: str
    read_acl: list[str]
    issued_at_tick: int
    record_digest: str


class MandateRevisionWire(TypedDict):
    """Exact JSON-compatible wire shape of :class:`MandateRevision`."""

    schema_id: str
    principal_id: str
    buyer_id: str
    mandate_id: str
    revision: int
    previous_digest: str | None
    changes: dict[str, Any]
    authorized_fields: list[str]
    logical_tick: int
    revision_digest: str


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One immutable, owner-scoped piece of CommerceWorld evidence.

    ``facts`` contains the claims made by ``issuer_id``.  ``trust`` contains
    structured trust metadata, for example a verification method and a score
    in basis points.  The protocol commits to the trust metadata but does not
    prescribe one domain-specific trust model.
    """

    record_id: str
    kind: str
    subject_id: str
    issuer_id: str
    facts: Mapping[str, JsonValue]
    trust: Mapping[str, JsonValue]
    version: int
    owner_id: str
    read_acl: tuple[str, ...]
    issued_at_tick: int
    record_digest: str = ""
    schema_id: str = EVIDENCE_RECORD_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class EvidenceReadAuthority:
    """Trusted World metadata against which a record read is authorized.

    Callers must load this binding from protected persistence or construct it
    from trusted Runtime state.  They must not copy it from an untrusted
    :class:`EvidenceRecord` supplied by the reader.
    """

    record_id: str
    issuer_id: str
    owner_id: str
    read_acl: tuple[str, ...]
    version: int
    record_digest: str

    def __post_init__(self) -> None:
        for name in ("record_id", "issuer_id", "owner_id"):
            _require_text(
                name,
                getattr(self, name),
                error_type=EvidenceReadAuthorityError,
            )
        _require_canonical_string_tuple(
            self.read_acl,
            "read_acl",
            error_type=EvidenceReadAuthorityError,
        )
        _require_positive_int(
            "version", self.version, error_type=EvidenceReadAuthorityError
        )
        _require_digest(
            "record_digest",
            self.record_digest,
            error_type=EvidenceReadAuthorityError,
        )


@dataclass(frozen=True, slots=True)
class MandateRevision:
    """One canonical principal-authored patch to a buyer mandate.

    ``changes`` is a flat mapping from mandate field paths to replacement
    JSON values.  ``authorized_fields`` must contain exactly those paths.
    Authority is verified against a separately trusted
    :class:`MandateRevisionAuthority`; self-declared fields never grant access.
    """

    principal_id: str
    buyer_id: str
    mandate_id: str
    revision: int
    previous_digest: str | None
    changes: Mapping[str, JsonValue]
    authorized_fields: tuple[str, ...]
    logical_tick: int
    revision_digest: str = ""
    schema_id: str = MANDATE_REVISION_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class MandateRevisionAuthority:
    """Trusted authority binding for one principal, buyer, and mandate."""

    principal_id: str
    buyer_id: str
    mandate_id: str
    allowed_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("principal_id", "buyer_id", "mandate_id"):
            _require_text(
                name,
                getattr(self, name),
                error_type=MandateRevisionAuthorityError,
            )
        _require_canonical_string_tuple(
            self.allowed_fields,
            "allowed_fields",
            error_type=MandateRevisionAuthorityError,
        )


def build_evidence_record(
    *,
    record_id: str,
    kind: str,
    subject_id: str,
    issuer_id: str,
    facts: Mapping[str, Any],
    trust: Mapping[str, Any],
    version: int,
    owner_id: str,
    read_acl: Iterable[str],
    issued_at_tick: int,
) -> EvidenceRecord:
    """Build, recursively freeze, and seal one core evidence record."""

    return seal_evidence_record(
        EvidenceRecord(
            record_id=record_id,
            kind=kind,
            subject_id=subject_id,
            issuer_id=issuer_id,
            facts=_freeze_object(
                facts, path="facts", error_type=EvidenceRecordSchemaError
            ),
            trust=_freeze_object(
                trust, path="trust", error_type=EvidenceRecordSchemaError
            ),
            version=version,
            owner_id=owner_id,
            read_acl=_canonicalize_string_iterable(
                read_acl, "read_acl", error_type=EvidenceRecordSchemaError
            ),
            issued_at_tick=issued_at_tick,
        )
    )


def seal_evidence_record(record: EvidenceRecord) -> EvidenceRecord:
    """Return a deeply immutable record carrying its canonical SHA-256 digest."""

    if not isinstance(record, EvidenceRecord):
        raise EvidenceRecordSchemaError("evidence record must be EvidenceRecord")
    normalized = EvidenceRecord(
        schema_id=record.schema_id,
        record_id=record.record_id,
        kind=record.kind,
        subject_id=record.subject_id,
        issuer_id=record.issuer_id,
        facts=_freeze_object(
            record.facts, path="facts", error_type=EvidenceRecordSchemaError
        ),
        trust=_freeze_object(
            record.trust, path="trust", error_type=EvidenceRecordSchemaError
        ),
        version=record.version,
        owner_id=record.owner_id,
        read_acl=record.read_acl,
        issued_at_tick=record.issued_at_tick,
        record_digest="",
    )
    _validate_evidence_fields(normalized)
    sealed = EvidenceRecord(
        schema_id=normalized.schema_id,
        record_id=normalized.record_id,
        kind=normalized.kind,
        subject_id=normalized.subject_id,
        issuer_id=normalized.issuer_id,
        facts=normalized.facts,
        trust=normalized.trust,
        version=normalized.version,
        owner_id=normalized.owner_id,
        read_acl=normalized.read_acl,
        issued_at_tick=normalized.issued_at_tick,
        record_digest=evidence_record_digest(normalized),
    )
    validate_evidence_record(sealed)
    return sealed


def evidence_record_digest(record: EvidenceRecord) -> str:
    """Digest every semantic field of an evidence record except the digest."""

    return _sha256_json(
        _evidence_record_contract(record), error_type=EvidenceRecordSchemaError
    )


def validate_evidence_record(record: EvidenceRecord) -> None:
    """Validate schema, field types, deep JSON content, and canonical digest."""

    if not isinstance(record, EvidenceRecord):
        raise EvidenceRecordSchemaError("evidence record must be EvidenceRecord")
    _validate_evidence_fields(record)
    _require_digest(
        "record_digest", record.record_digest, error_type=EvidenceRecordSchemaError
    )
    if record.record_digest != evidence_record_digest(record):
        raise EvidenceRecordDigestMismatch("evidence record digest mismatch")


def evidence_record_to_dict(record: EvidenceRecord) -> EvidenceRecordWire:
    """Return a fresh JSON-compatible mapping after complete validation."""

    validate_evidence_record(record)
    return cast(
        EvidenceRecordWire,
        {
            **_evidence_record_contract(record),
            "record_digest": record.record_digest,
        },
    )


def evidence_record_to_json(record: EvidenceRecord) -> str:
    """Serialize a record to canonical JSON for persistence, HTTP, and replay."""

    return _canonical_json(
        evidence_record_to_dict(record), error_type=EvidenceRecordSchemaError
    )


def coerce_evidence_record(value: Any) -> EvidenceRecord:
    """Strictly coerce a decoded JSON object; reject unknown or missing fields."""

    if isinstance(value, EvidenceRecord):
        validate_evidence_record(value)
        return value
    row = _exact_mapping(
        value,
        expected=_EVIDENCE_WIRE_FIELDS,
        label="evidence record",
        error_type=EvidenceRecordSchemaError,
    )
    if row["schema_id"] != EVIDENCE_RECORD_SCHEMA_ID:
        raise EvidenceRecordSchemaError(
            f"unsupported evidence schema_id: {row['schema_id']!r}"
        )
    record = EvidenceRecord(
        schema_id=_strict_text(
            "schema_id", row["schema_id"], error_type=EvidenceRecordSchemaError
        ),
        record_id=_strict_text(
            "record_id", row["record_id"], error_type=EvidenceRecordSchemaError
        ),
        kind=_strict_text("kind", row["kind"], error_type=EvidenceRecordSchemaError),
        subject_id=_strict_text(
            "subject_id", row["subject_id"], error_type=EvidenceRecordSchemaError
        ),
        issuer_id=_strict_text(
            "issuer_id", row["issuer_id"], error_type=EvidenceRecordSchemaError
        ),
        facts=_freeze_object(
            row["facts"], path="facts", error_type=EvidenceRecordSchemaError
        ),
        trust=_freeze_object(
            row["trust"], path="trust", error_type=EvidenceRecordSchemaError
        ),
        version=_strict_int(
            "version", row["version"], error_type=EvidenceRecordSchemaError
        ),
        owner_id=_strict_text(
            "owner_id", row["owner_id"], error_type=EvidenceRecordSchemaError
        ),
        read_acl=_strict_string_array(
            row["read_acl"], "read_acl", error_type=EvidenceRecordSchemaError
        ),
        issued_at_tick=_strict_int(
            "issued_at_tick",
            row["issued_at_tick"],
            error_type=EvidenceRecordSchemaError,
        ),
        record_digest=_strict_text(
            "record_digest",
            row["record_digest"],
            error_type=EvidenceRecordSchemaError,
        ),
    )
    validate_evidence_record(record)
    return record


def evidence_record_from_json(payload: str) -> EvidenceRecord:
    """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""

    value = _strict_json_loads(
        payload, label="evidence record", error_type=EvidenceRecordSchemaError
    )
    return coerce_evidence_record(value)


def verify_evidence_read_authority(
    record: EvidenceRecord,
    *,
    reader_id: str,
    authority: EvidenceReadAuthority,
) -> None:
    """Authorize one read using trusted metadata, never record-supplied ACL.

    The issuer, owner, ACL, version, record identity, and exact persisted digest
    are compared before the reader is checked.  Resealing a record with a
    forged issuer or expanded ACL therefore cannot grant read access.
    """

    validate_evidence_record(record)
    _require_text(
        "reader_id", reader_id, error_type=EvidenceReadAuthorityError
    )
    if not isinstance(authority, EvidenceReadAuthority):
        raise EvidenceReadAuthorityError(
            "authority must be a trusted EvidenceReadAuthority binding"
        )
    checks = {
        "record_id": (record.record_id, authority.record_id),
        "issuer_id": (record.issuer_id, authority.issuer_id),
        "owner_id": (record.owner_id, authority.owner_id),
        "read_acl": (record.read_acl, authority.read_acl),
        "version": (record.version, authority.version),
        "record_digest": (record.record_digest, authority.record_digest),
    }
    mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatches:
        raise EvidenceReadAuthorityError(
            "evidence read authority mismatch: " + ", ".join(mismatches)
        )
    if reader_id != authority.owner_id and reader_id not in authority.read_acl:
        raise EvidenceReadDenied(f"reader {reader_id!r} is not authorized for evidence")


def authorize_evidence_read(
    record: EvidenceRecord,
    *,
    reader_id: str,
    authority: EvidenceReadAuthority,
) -> EvidenceRecord:
    """Verify read authority and return the validated record for the caller."""

    verify_evidence_read_authority(record, reader_id=reader_id, authority=authority)
    return record


def build_mandate_revision(
    *,
    principal_id: str,
    buyer_id: str,
    mandate_id: str,
    revision: int,
    previous_digest: str | None,
    changes: Mapping[str, Any],
    authorized_fields: Iterable[str],
    logical_tick: int,
) -> MandateRevision:
    """Build, recursively freeze, and seal one principal-authored revision."""

    return seal_mandate_revision(
        MandateRevision(
            principal_id=principal_id,
            buyer_id=buyer_id,
            mandate_id=mandate_id,
            revision=revision,
            previous_digest=previous_digest,
            changes=_freeze_object(
                changes, path="changes", error_type=MandateRevisionSchemaError
            ),
            authorized_fields=_canonicalize_string_iterable(
                authorized_fields,
                "authorized_fields",
                error_type=MandateRevisionSchemaError,
            ),
            logical_tick=logical_tick,
        )
    )


def seal_mandate_revision(revision: MandateRevision) -> MandateRevision:
    """Return a deeply immutable revision carrying its canonical digest."""

    if not isinstance(revision, MandateRevision):
        raise MandateRevisionSchemaError("mandate revision must be MandateRevision")
    normalized = MandateRevision(
        schema_id=revision.schema_id,
        principal_id=revision.principal_id,
        buyer_id=revision.buyer_id,
        mandate_id=revision.mandate_id,
        revision=revision.revision,
        previous_digest=revision.previous_digest,
        changes=_freeze_object(
            revision.changes,
            path="changes",
            error_type=MandateRevisionSchemaError,
        ),
        authorized_fields=revision.authorized_fields,
        logical_tick=revision.logical_tick,
        revision_digest="",
    )
    _validate_mandate_revision_fields(normalized)
    sealed = MandateRevision(
        schema_id=normalized.schema_id,
        principal_id=normalized.principal_id,
        buyer_id=normalized.buyer_id,
        mandate_id=normalized.mandate_id,
        revision=normalized.revision,
        previous_digest=normalized.previous_digest,
        changes=normalized.changes,
        authorized_fields=normalized.authorized_fields,
        logical_tick=normalized.logical_tick,
        revision_digest=mandate_revision_digest(normalized),
    )
    validate_mandate_revision(sealed)
    return sealed


def mandate_revision_digest(revision: MandateRevision) -> str:
    """Digest every semantic revision field except ``revision_digest``."""

    return _sha256_json(
        _mandate_revision_contract(revision),
        error_type=MandateRevisionSchemaError,
    )


def validate_mandate_revision(revision: MandateRevision) -> None:
    """Validate schema, patch shape, chain header, and canonical digest."""

    if not isinstance(revision, MandateRevision):
        raise MandateRevisionSchemaError("mandate revision must be MandateRevision")
    _validate_mandate_revision_fields(revision)
    _require_digest(
        "revision_digest",
        revision.revision_digest,
        error_type=MandateRevisionSchemaError,
    )
    if revision.revision_digest != mandate_revision_digest(revision):
        raise MandateRevisionDigestMismatch("mandate revision digest mismatch")


def mandate_revision_to_dict(revision: MandateRevision) -> MandateRevisionWire:
    """Return a fresh JSON-compatible mapping after complete validation."""

    validate_mandate_revision(revision)
    return cast(
        MandateRevisionWire,
        {
            **_mandate_revision_contract(revision),
            "revision_digest": revision.revision_digest,
        },
    )


def mandate_revision_to_json(revision: MandateRevision) -> str:
    """Serialize a revision to canonical JSON for persistence and replay."""

    return _canonical_json(
        mandate_revision_to_dict(revision), error_type=MandateRevisionSchemaError
    )


def coerce_mandate_revision(value: Any) -> MandateRevision:
    """Strictly coerce a decoded JSON object; reject unknown or missing fields."""

    if isinstance(value, MandateRevision):
        validate_mandate_revision(value)
        return value
    row = _exact_mapping(
        value,
        expected=_MANDATE_WIRE_FIELDS,
        label="mandate revision",
        error_type=MandateRevisionSchemaError,
    )
    if row["schema_id"] != MANDATE_REVISION_SCHEMA_ID:
        raise MandateRevisionSchemaError(
            f"unsupported mandate revision schema_id: {row['schema_id']!r}"
        )
    previous = row["previous_digest"]
    if previous is not None and not isinstance(previous, str):
        raise MandateRevisionSchemaError("previous_digest must be a string or null")
    revision = MandateRevision(
        schema_id=_strict_text(
            "schema_id", row["schema_id"], error_type=MandateRevisionSchemaError
        ),
        principal_id=_strict_text(
            "principal_id",
            row["principal_id"],
            error_type=MandateRevisionSchemaError,
        ),
        buyer_id=_strict_text(
            "buyer_id", row["buyer_id"], error_type=MandateRevisionSchemaError
        ),
        mandate_id=_strict_text(
            "mandate_id", row["mandate_id"], error_type=MandateRevisionSchemaError
        ),
        revision=_strict_int(
            "revision", row["revision"], error_type=MandateRevisionSchemaError
        ),
        previous_digest=previous,
        changes=_freeze_object(
            row["changes"],
            path="changes",
            error_type=MandateRevisionSchemaError,
        ),
        authorized_fields=_strict_string_array(
            row["authorized_fields"],
            "authorized_fields",
            error_type=MandateRevisionSchemaError,
        ),
        logical_tick=_strict_int(
            "logical_tick",
            row["logical_tick"],
            error_type=MandateRevisionSchemaError,
        ),
        revision_digest=_strict_text(
            "revision_digest",
            row["revision_digest"],
            error_type=MandateRevisionSchemaError,
        ),
    )
    validate_mandate_revision(revision)
    return revision


def mandate_revision_from_json(payload: str) -> MandateRevision:
    """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""

    value = _strict_json_loads(
        payload, label="mandate revision", error_type=MandateRevisionSchemaError
    )
    return coerce_mandate_revision(value)


def verify_mandate_revision_authority(
    revision: MandateRevision,
    authority: MandateRevisionAuthority,
) -> None:
    """Verify identity and field permissions against trusted principal state."""

    validate_mandate_revision(revision)
    if not isinstance(authority, MandateRevisionAuthority):
        raise MandateRevisionAuthorityError(
            "authority must be a trusted MandateRevisionAuthority binding"
        )
    checks = {
        "principal_id": (revision.principal_id, authority.principal_id),
        "buyer_id": (revision.buyer_id, authority.buyer_id),
        "mandate_id": (revision.mandate_id, authority.mandate_id),
    }
    mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
    if mismatches:
        raise MandateRevisionAuthorityError(
            "mandate authority mismatch: " + ", ".join(mismatches)
        )
    unauthorized = sorted(
        set(revision.authorized_fields) - set(authority.allowed_fields)
    )
    if unauthorized:
        raise MandateRevisionAuthorityError(
            f"unauthorized mandate fields: {unauthorized!r}"
        )


def validate_mandate_revision_transition(
    previous: MandateRevision | None,
    candidate: MandateRevision,
    authority: MandateRevisionAuthority,
) -> MandateRevisionDisposition:
    """Purely validate one append or exact idempotent re-delivery.

    New revisions must be contiguous, point to the exact previous digest, and
    have a strictly increasing logical tick.  Re-delivery of the exact current
    revision is idempotent.  Reuse of its revision number with any other digest
    is a conflict.
    """

    verify_mandate_revision_authority(candidate, authority)
    if previous is None:
        if candidate.revision != 1:
            raise MandateRevisionOrderError(
                f"first mandate revision must be 1, got {candidate.revision}"
            )
        return "append"

    verify_mandate_revision_authority(previous, authority)
    if candidate.revision == previous.revision:
        if candidate.revision_digest == previous.revision_digest:
            return "idempotent"
        raise MandateRevisionConflict(
            f"conflicting content for mandate revision {candidate.revision}"
        )
    if candidate.revision < previous.revision:
        raise MandateRevisionOrderError(
            f"stale mandate revision {candidate.revision}; current is {previous.revision}"
        )
    if candidate.revision != previous.revision + 1:
        raise MandateRevisionOrderError(
            f"mandate revision jump from {previous.revision} to {candidate.revision}"
        )
    if candidate.previous_digest != previous.revision_digest:
        raise MandateRevisionOrderError("mandate previous_digest mismatch")
    if candidate.logical_tick <= previous.logical_tick:
        raise MandateRevisionOrderError(
            "mandate logical_tick must increase between revisions"
        )
    return "append"


def validate_mandate_revision_sequence(
    revisions: Iterable[MandateRevision],
    authority: MandateRevisionAuthority,
) -> tuple[MandateRevision, ...]:
    """Validate a persistence-order sequence and collapse exact re-deliveries."""

    accepted: list[MandateRevision] = []
    current: MandateRevision | None = None
    for candidate in revisions:
        disposition = validate_mandate_revision_transition(
            current, candidate, authority
        )
        if disposition == "append":
            accepted.append(candidate)
            current = candidate
    return tuple(accepted)


def _evidence_record_contract(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "schema_id": record.schema_id,
        "record_id": record.record_id,
        "kind": record.kind,
        "subject_id": record.subject_id,
        "issuer_id": record.issuer_id,
        "facts": _thaw_json(record.facts),
        "trust": _thaw_json(record.trust),
        "version": record.version,
        "owner_id": record.owner_id,
        "read_acl": list(record.read_acl),
        "issued_at_tick": record.issued_at_tick,
    }


def _mandate_revision_contract(revision: MandateRevision) -> dict[str, Any]:
    return {
        "schema_id": revision.schema_id,
        "principal_id": revision.principal_id,
        "buyer_id": revision.buyer_id,
        "mandate_id": revision.mandate_id,
        "revision": revision.revision,
        "previous_digest": revision.previous_digest,
        "changes": _thaw_json(revision.changes),
        "authorized_fields": list(revision.authorized_fields),
        "logical_tick": revision.logical_tick,
    }


def _validate_evidence_fields(record: EvidenceRecord) -> None:
    if record.schema_id != EVIDENCE_RECORD_SCHEMA_ID:
        raise EvidenceRecordSchemaError(
            f"unsupported evidence schema_id: {record.schema_id!r}"
        )
    for name in ("record_id", "kind", "subject_id", "issuer_id", "owner_id"):
        _require_text(name, getattr(record, name), error_type=EvidenceRecordSchemaError)
    _freeze_object(record.facts, path="facts", error_type=EvidenceRecordSchemaError)
    _freeze_object(record.trust, path="trust", error_type=EvidenceRecordSchemaError)
    _require_positive_int(
        "version", record.version, error_type=EvidenceRecordSchemaError
    )
    _require_canonical_string_tuple(
        record.read_acl, "read_acl", error_type=EvidenceRecordSchemaError
    )
    _require_nonnegative_int(
        "issued_at_tick",
        record.issued_at_tick,
        error_type=EvidenceRecordSchemaError,
    )


def _validate_mandate_revision_fields(revision: MandateRevision) -> None:
    if revision.schema_id != MANDATE_REVISION_SCHEMA_ID:
        raise MandateRevisionSchemaError(
            f"unsupported mandate revision schema_id: {revision.schema_id!r}"
        )
    for name in ("principal_id", "buyer_id", "mandate_id"):
        _require_text(
            name, getattr(revision, name), error_type=MandateRevisionSchemaError
        )
    _require_positive_int(
        "revision", revision.revision, error_type=MandateRevisionSchemaError
    )
    if revision.revision == 1:
        if revision.previous_digest is not None:
            raise MandateRevisionSchemaError(
                "first mandate revision previous_digest must be null"
            )
    else:
        _require_digest(
            "previous_digest",
            revision.previous_digest,
            error_type=MandateRevisionSchemaError,
        )
    frozen_changes = _freeze_object(
        revision.changes, path="changes", error_type=MandateRevisionSchemaError
    )
    if not frozen_changes:
        raise MandateRevisionSchemaError("changes must contain at least one field")
    for key in frozen_changes:
        _require_text(
            "changes field path", key, error_type=MandateRevisionSchemaError
        )
    _require_canonical_string_tuple(
        revision.authorized_fields,
        "authorized_fields",
        error_type=MandateRevisionSchemaError,
    )
    if tuple(sorted(frozen_changes)) != revision.authorized_fields:
        raise MandateRevisionSchemaError(
            "authorized_fields must exactly match the changed field paths"
        )
    _require_nonnegative_int(
        "logical_tick",
        revision.logical_tick,
        error_type=MandateRevisionSchemaError,
    )


def _freeze_object(
    value: Any,
    *,
    path: str,
    error_type: type[SchemaError],
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise error_type(f"{path} must be an object")
    frozen = _freeze_json(value, path=path, error_type=error_type)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise error_type(f"{path} must be an object")
    return frozen


def _freeze_json(
    value: Any,
    *,
    path: str,
    error_type: type[SchemaError],
) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type(f"{path} contains a non-string object key")
            frozen[key] = _freeze_json(
                item, path=f"{path}.{key}", error_type=error_type
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(
                item,
                path=f"{path}[{index}]",
                error_type=error_type,
            )
            for index, item in enumerate(value)
        )
    raise error_type(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonicalize_string_iterable(
    values: Iterable[str],
    label: str,
    *,
    error_type: type[SchemaError],
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise error_type(f"{label} must be an array of strings")
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise error_type(f"{label} must be an array of strings") from exc
    for row in rows:
        _require_text(label, row, error_type=error_type)
    if len(set(rows)) != len(rows):
        raise error_type(f"{label} must not contain duplicates")
    return tuple(sorted(rows))


def _strict_string_array(
    value: Any,
    label: str,
    *,
    error_type: type[SchemaError],
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise error_type(f"{label} must be an array of strings")
    rows = tuple(value)
    _require_canonical_string_tuple(rows, label, error_type=error_type)
    return rows


def _require_canonical_string_tuple(
    values: Any,
    label: str,
    *,
    error_type: type[SchemaError],
) -> None:
    if not isinstance(values, tuple):
        raise error_type(f"{label} must be a canonical string tuple")
    for row in values:
        _require_text(label, row, error_type=error_type)
    if len(set(values)) != len(values):
        raise error_type(f"{label} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise error_type(f"{label} must be sorted canonically")


def _exact_mapping(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
    error_type: type[SchemaError],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{label} must be an object")
    actual = frozenset(value.keys())
    non_string = sorted(repr(key) for key in actual if not isinstance(key, str))
    if non_string:
        raise error_type(f"{label} has non-string fields: {non_string!r}")
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing!r}")
        if unknown:
            details.append(f"unknown={unknown!r}")
        raise error_type(f"{label} has invalid fields: {', '.join(details)}")
    return cast(Mapping[str, Any], value)


def _strict_json_loads(
    payload: str,
    *,
    label: str,
    error_type: type[SchemaError],
) -> Any:
    if not isinstance(payload, str):
        raise error_type(f"{label} JSON must be a string")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise error_type(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise error_type(f"{label} JSON contains non-finite number {value!r}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise error_type(f"invalid {label} JSON: {exc}") from exc


def _sha256_json(value: Any, *, error_type: type[SchemaError]) -> str:
    body = _canonical_json(value, error_type=error_type).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _canonical_json(value: Any, *, error_type: type[SchemaError]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise error_type(f"value is not canonical JSON: {exc}") from exc


def _strict_text(name: str, value: Any, *, error_type: type[SchemaError]) -> str:
    _require_text(name, value, error_type=error_type)
    return cast(str, value)


def _strict_int(name: str, value: Any, *, error_type: type[SchemaError]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{name} must be an integer")
    return cast(int, value)


def _require_text(name: str, value: Any, *, error_type: type[SchemaError]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{name} must be a non-empty string")


def _require_digest(name: str, value: Any, *, error_type: type[SchemaError]) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise error_type(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(
    name: str,
    value: Any,
    *,
    error_type: type[SchemaError],
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error_type(f"{name} must be a positive integer")


def _require_nonnegative_int(
    name: str,
    value: Any,
    *,
    error_type: type[SchemaError],
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_type(f"{name} must be a non-negative integer")


__all__ = [
    "EVIDENCE_RECORD_SCHEMA_ID",
    "MANDATE_REVISION_SCHEMA_ID",
    "EvidenceReadAuthority",
    "EvidenceReadAuthorityError",
    "EvidenceReadDenied",
    "EvidenceRecord",
    "EvidenceRecordDigestMismatch",
    "EvidenceRecordSchemaError",
    "EvidenceRecordWire",
    "MandateRevision",
    "MandateRevisionAuthority",
    "MandateRevisionAuthorityError",
    "MandateRevisionConflict",
    "MandateRevisionDigestMismatch",
    "MandateRevisionDisposition",
    "MandateRevisionOrderError",
    "MandateRevisionSchemaError",
    "MandateRevisionWire",
    "authorize_evidence_read",
    "build_evidence_record",
    "build_mandate_revision",
    "coerce_evidence_record",
    "coerce_mandate_revision",
    "evidence_record_digest",
    "evidence_record_from_json",
    "evidence_record_to_dict",
    "evidence_record_to_json",
    "mandate_revision_digest",
    "mandate_revision_from_json",
    "mandate_revision_to_dict",
    "mandate_revision_to_json",
    "seal_evidence_record",
    "seal_mandate_revision",
    "validate_evidence_record",
    "validate_mandate_revision",
    "validate_mandate_revision_sequence",
    "validate_mandate_revision_transition",
    "verify_evidence_read_authority",
    "verify_mandate_revision_authority",
]
