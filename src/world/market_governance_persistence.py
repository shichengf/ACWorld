"""Strict persistence envelopes for World-owned marketplace governance.

This module does not own a commerce world.  It defines the two append-only
collections required by the T8 integration contract and the compare-and-swap
commit value that an enclosing :class:`World` transaction applies.  Catalog,
orders, ledger, evidence, mandates, reviews, reputation, authority operations,
logical time, and the global commit journal remain existing World state.

The two physical collections are deliberately fixed:

``governance_policies``
    Trusted policy inputs.

``governance_records``
    Derived governance outcomes.

Every row is a strict tagged union.  Nested placements remain inside a
campaign and remediation steps remain inside a remediation plan.  Detector
observations are never stored here; they remain ordinary ``EvidenceRecord``
rows and only their derived signal and case enter this projection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, TypeAlias, cast

from protocol.market_governance import (
    Campaign,
    GovernanceCase,
    MarketSignal,
    RankingContext,
    RemediationPlan,
    ReputationEvent,
    ReviewAggregate,
    ReviewEvidence,
    campaign_to_wire,
    coerce_campaign,
    coerce_governance_case,
    coerce_market_signal,
    coerce_ranking_context,
    coerce_remediation_plan,
    coerce_reputation_event,
    coerce_review_aggregate,
    coerce_review_evidence,
    governance_case_to_wire,
    market_signal_to_wire,
    ranking_context_to_wire,
    remediation_plan_to_wire,
    reputation_event_to_wire,
    review_aggregate_to_wire,
    review_evidence_to_wire,
    validate_version_successor,
)
from world.errors import IdempotencyConflict
from world.market_governance_core import (
    ADS_CAMPAIGN_TERMS_SCHEMA,
    GOVERNANCE_RESOLUTION_DECISION_SCHEMA,
    GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA,
    REMEDIATION_BLUEPRINT_SCHEMA,
    REPUTATION_POLICY_SCHEMA,
    REVIEW_ACCOUNT_BINDING_SCHEMA,
    AdsCampaignTerms,
    AdsPlacementTerms,
    GovernanceResolutionDecision,
    GovernanceResponseAttestation,
    RemediationBlueprint,
    RemediationBlueprintStep,
    ReputationPolicyRevision,
    ReviewAccountBinding,
    validate_ads_campaign_terms,
    validate_remediation_blueprint,
    validate_reputation_policy,
    validate_reputation_policy_transition,
    validate_review_account_binding,
)


GOVERNANCE_POLICY_ENVELOPE_SCHEMA = "cwe.world-governance-policy-envelope.v1"
GOVERNANCE_RECORD_ENVELOPE_SCHEMA = "cwe.world-governance-record-envelope.v1"
GOVERNANCE_PROJECTION_EFFECT_SCHEMA = "cwe.world-governance-projection-effect.v1"
GOVERNANCE_OPERATION_SCHEMA = "cwe.world-governance-operation.v1"
GOVERNANCE_COMMIT_SCHEMA = "cwe.world-governance-commit.v1"

GovernanceCollection: TypeAlias = Literal[
    "governance_policies", "governance_records"
]
GovernancePolicyKind: TypeAlias = Literal[
    "ads_campaign_terms",
    "review_account_binding",
    "reputation_policy_revision",
    "remediation_blueprint",
]
GovernanceRecordKind: TypeAlias = Literal[
    "campaign",
    "review_evidence",
    "review_aggregate",
    "market_signal",
    "governance_case",
    "governance_response_attestation",
    "governance_resolution_decision",
    "reputation_event",
    "remediation_plan",
    "ranking_context",
]
ProjectionTable: TypeAlias = Literal["reviews", "reputation"]

GovernancePolicyPayload: TypeAlias = (
    AdsCampaignTerms
    | ReviewAccountBinding
    | ReputationPolicyRevision
    | RemediationBlueprint
)
GovernanceRecordPayload: TypeAlias = (
    Campaign
    | ReviewEvidence
    | ReviewAggregate
    | MarketSignal
    | GovernanceCase
    | GovernanceResponseAttestation
    | GovernanceResolutionDecision
    | ReputationEvent
    | RemediationPlan
    | RankingContext
)


class MarketGovernancePersistenceError(ValueError):
    """A governance envelope or staged commit is not replay-safe."""


@dataclass(frozen=True, slots=True, order=True)
class GovernanceIndex:
    """One protected exact-match index stored outside the payload."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class GovernancePolicyEnvelope:
    kind: GovernancePolicyKind
    stable_id: str
    revision: int
    previous_envelope_digest: str | None
    semantic_digest: str
    owner_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    indexes: tuple[GovernanceIndex, ...]
    logical_tick: int
    service_actor: str
    original_actor: str
    idempotency_key: str
    request_fingerprint: str
    payload: GovernancePolicyPayload
    envelope_digest: str
    schema_id: str = GOVERNANCE_POLICY_ENVELOPE_SCHEMA

    def __getattr__(self, name: str) -> Any:
        """Expose typed payload fields for snapshot/query ergonomics.

        Persistence metadata remains explicit on the envelope.  Delegation is
        read-only and never participates in serialization or validation.
        """

        return getattr(self.payload, name)


@dataclass(frozen=True, slots=True)
class GovernanceRecordEnvelope:
    kind: GovernanceRecordKind
    stable_id: str
    version: int
    previous_envelope_digest: str | None
    semantic_digest: str
    owner_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    indexes: tuple[GovernanceIndex, ...]
    logical_tick: int
    service_actor: str
    original_actor: str
    idempotency_key: str
    request_fingerprint: str
    payload: GovernanceRecordPayload
    envelope_digest: str
    schema_id: str = GOVERNANCE_RECORD_ENVELOPE_SCHEMA

    def __getattr__(self, name: str) -> Any:
        """Expose the immutable typed record through its persisted envelope."""

        return getattr(self.payload, name)


GovernanceEnvelope: TypeAlias = GovernancePolicyEnvelope | GovernanceRecordEnvelope


@dataclass(frozen=True, slots=True)
class GovernanceWrite:
    collection: GovernanceCollection
    key: str
    value: GovernanceEnvelope


@dataclass(frozen=True, slots=True)
class GovernanceProjectionEffect:
    """Explicit effect on an existing World projection.

    The effect is not a third governance store.  The enclosing World validates
    and applies it to ``reviews`` or ``reputation`` in the same transaction as
    the typed governance rows.
    """

    table: ProjectionTable
    key: str
    before_digest: str | None
    payload: Mapping[str, Any]
    effect_digest: str
    schema_id: str = GOVERNANCE_PROJECTION_EFFECT_SCHEMA


@dataclass(frozen=True, slots=True, order=True)
class GovernanceResultRef:
    collection: GovernanceCollection
    key: str
    semantic_digest: str


@dataclass(frozen=True, slots=True)
class GovernanceOperationRecord:
    operation_id: str
    operation: str
    service_actor: str
    original_actor: str
    idempotency_key: str
    request_fingerprint: str
    logical_tick: int
    result_refs: tuple[GovernanceResultRef, ...]
    projection_effect_digest: str
    operation_digest: str
    schema_id: str = GOVERNANCE_OPERATION_SCHEMA


@dataclass(frozen=True, slots=True)
class GovernanceCommit:
    commit_id: str
    expected_logical_tick: int
    logical_tick: int
    context_digest: str
    writes: tuple[GovernanceWrite, ...]
    projection_effects: tuple[GovernanceProjectionEffect, ...]
    operation: GovernanceOperationRecord
    commit_digest: str
    schema_id: str = GOVERNANCE_COMMIT_SCHEMA


class GovernanceTables:
    """Typed staging projection for the two physical collections.

    ``operations`` and ``commits`` mirror rows that the enclosing World writes
    to its existing authority-operation table and global journal.  They are
    retained here solely to validate exact retry and deterministic replay.
    """

    def __init__(self) -> None:
        self._policies: dict[str, GovernancePolicyEnvelope] = {}
        self._records: dict[str, GovernanceRecordEnvelope] = {}
        self._operations: dict[str, GovernanceOperationRecord] = {}
        self._idempotency: dict[tuple[str, str, str, str], str] = {}
        self._commits: list[GovernanceCommit] = []

    def clone(self) -> "GovernanceTables":
        clone = GovernanceTables()
        clone._policies = dict(self._policies)
        clone._records = dict(self._records)
        clone._operations = dict(self._operations)
        clone._idempotency = dict(self._idempotency)
        clone._commits = list(self._commits)
        return clone

    def replace_from(self, other: "GovernanceTables") -> None:
        if not isinstance(other, GovernanceTables):
            raise TypeError("replacement must be GovernanceTables")
        self._policies = dict(other._policies)
        self._records = dict(other._records)
        self._operations = dict(other._operations)
        self._idempotency = dict(other._idempotency)
        self._commits = list(other._commits)

    def append(self, write: GovernanceWrite) -> Literal["append", "idempotent"]:
        validate_governance_write(write)
        rows: dict[str, GovernanceEnvelope]
        if write.collection == "governance_policies":
            rows = cast(dict[str, GovernanceEnvelope], self._policies)
        else:
            rows = cast(dict[str, GovernanceEnvelope], self._records)
        existing = rows.get(write.key)
        if existing is not None:
            if existing.envelope_digest == write.value.envelope_digest:
                return "idempotent"
            raise IdempotencyConflict(
                f"governance row {write.collection}:{write.key} already exists"
            )
        self._validate_predecessor(write.value)
        rows[write.key] = write.value
        return "append"

    def _validate_predecessor(self, row: GovernanceEnvelope) -> None:
        collection: GovernanceCollection
        version: int
        if isinstance(row, GovernancePolicyEnvelope):
            collection = "governance_policies"
            version = row.revision
        else:
            collection = "governance_records"
            version = row.version
        if version == 1:
            if row.previous_envelope_digest is not None:
                raise MarketGovernancePersistenceError(
                    "first governance envelope cannot have a predecessor"
                )
            return
        predecessor_key = envelope_key_from_parts(
            collection, row.kind, row.stable_id, version - 1
        )
        predecessor = (
            self._policies.get(predecessor_key)
            if collection == "governance_policies"
            else self._records.get(predecessor_key)
        )
        if predecessor is None:
            raise MarketGovernancePersistenceError(
                "governance envelope version is not contiguous"
            )
        if row.previous_envelope_digest != predecessor.envelope_digest:
            raise MarketGovernancePersistenceError(
                "governance envelope predecessor digest mismatch"
            )
        if (
            row.owner_ids != predecessor.owner_ids
            or row.subject_ids != predecessor.subject_ids
            or row.service_actor != predecessor.service_actor
        ):
            raise MarketGovernancePersistenceError(
                "governance stream authority or subjects changed"
            )
        if row.logical_tick < predecessor.logical_tick:
            raise MarketGovernancePersistenceError("governance time moved backwards")
        if isinstance(row, GovernancePolicyEnvelope) and isinstance(
            predecessor, GovernancePolicyEnvelope
        ):
            if not isinstance(row.payload, ReputationPolicyRevision) or not isinstance(
                predecessor.payload, ReputationPolicyRevision
            ):
                raise MarketGovernancePersistenceError(
                    "only reputation policies currently support revisions"
                )
            validate_reputation_policy_transition(
                predecessor.payload,
                row.payload,
                original_actor=row.original_actor,
                server_tick=row.logical_tick,
                trusted_publisher_ids=(row.service_actor,),
            )
            if row.payload.previous_digest != predecessor.semantic_digest:
                raise MarketGovernancePersistenceError(
                    "policy semantic predecessor digest mismatch"
                )
        elif isinstance(row, GovernanceRecordEnvelope) and isinstance(
            predecessor, GovernanceRecordEnvelope
        ):
            if isinstance(
                row.payload,
                (GovernanceResponseAttestation, GovernanceResolutionDecision),
            ):
                raise MarketGovernancePersistenceError(
                    "immutable governance attestation cannot have a successor"
                )
            previous_payload = cast(Any, predecessor.payload)
            current_payload = cast(Any, row.payload)
            validate_version_successor(
                previous_payload,
                current_payload,
                expected_authority=row.service_actor,
                expected_owner_id=(row.owner_ids[0] if len(row.owner_ids) == 1 else None),
            )
            if current_payload.previous_digest != predecessor.semantic_digest:
                raise MarketGovernancePersistenceError(
                    "record semantic predecessor digest mismatch"
                )

    def append_operation(
        self, operation: GovernanceOperationRecord
    ) -> Literal["append", "idempotent"]:
        validate_governance_operation(operation)
        scope = (
            operation.service_actor,
            operation.original_actor,
            operation.operation,
            operation.idempotency_key,
        )
        prior_id = self._idempotency.get(scope)
        if prior_id is not None:
            prior = self._operations[prior_id]
            if prior.operation_digest == operation.operation_digest:
                return "idempotent"
            raise IdempotencyConflict(
                "governance idempotency key was reused with different content"
            )
        prior = self._operations.get(operation.operation_id)
        if prior is not None:
            if prior.operation_digest == operation.operation_digest:
                return "idempotent"
            raise IdempotencyConflict("governance operation id collision")
        self._operations[operation.operation_id] = operation
        self._idempotency[scope] = operation.operation_id
        return "append"

    def operation_for_retry(
        self,
        *,
        service_actor: str,
        original_actor: str,
        operation: str,
        idempotency_key: str,
    ) -> GovernanceOperationRecord | None:
        operation_id = self._idempotency.get(
            (service_actor, original_actor, operation, idempotency_key)
        )
        return None if operation_id is None else self._operations[operation_id]

    def apply_commit(self, commit: GovernanceCommit) -> Literal["append", "idempotent"]:
        validate_governance_commit(commit)
        for prior in self._commits:
            if prior.commit_id == commit.commit_id:
                if prior.commit_digest == commit.commit_digest:
                    return "idempotent"
                raise IdempotencyConflict("governance commit id collision")
        if self._commits and commit.logical_tick <= self._commits[-1].logical_tick:
            raise MarketGovernancePersistenceError(
                "governance commit logical time must be strictly increasing"
            )
        staged = self.clone()
        dispositions = tuple(staged.append(write) for write in commit.writes)
        operation_disposition = staged.append_operation(commit.operation)
        if any(item == "idempotent" for item in dispositions) or (
            operation_disposition == "idempotent"
        ):
            raise MarketGovernancePersistenceError(
                "partial governance retry is not an atomic commit"
            )
        staged._commits.append(commit)
        self.replace_from(staged)
        return "append"

    def read(
        self,
        collection: GovernanceCollection,
        key: str,
        *,
        caller: str | None,
    ) -> GovernanceEnvelope | None:
        row: GovernanceEnvelope | None = (
            self._policies.get(key)
            if collection == "governance_policies"
            else self._records.get(key)
        )
        if row is None or not _can_read(row, caller):
            return None
        return row

    def history(
        self,
        kind: GovernancePolicyKind | GovernanceRecordKind,
        stable_id: str,
        *,
        caller: str | None,
    ) -> tuple[GovernanceEnvelope, ...]:
        rows: Iterable[GovernanceEnvelope]
        if kind in _POLICY_PAYLOAD_TYPES:
            rows = self._policies.values()
        else:
            rows = self._records.values()
        visible = [
            row
            for row in rows
            if row.kind == kind
            and row.stable_id == stable_id
            and _can_read(row, caller)
        ]
        return tuple(sorted(visible, key=envelope_version))

    def internal_all(
        self, collection: GovernanceCollection
    ) -> Iterator[tuple[str, GovernanceEnvelope]]:
        rows: Mapping[str, GovernanceEnvelope] = (
            self._policies
            if collection == "governance_policies"
            else self._records
        )
        yield from sorted(rows.items())

    @property
    def operations(self) -> tuple[GovernanceOperationRecord, ...]:
        return tuple(
            sorted(
                self._operations.values(),
                key=lambda row: (row.logical_tick, row.operation_id),
            )
        )

    @property
    def commits(self) -> tuple[GovernanceCommit, ...]:
        return tuple(self._commits)

    def state_digest(self) -> str:
        return canonical_digest(self.to_wire())

    def to_wire(self) -> dict[str, Any]:
        return {
            "governance_policies": [
                {"key": key, "value": policy_envelope_to_wire(cast(GovernancePolicyEnvelope, row))}
                for key, row in self.internal_all("governance_policies")
            ],
            "governance_records": [
                {"key": key, "value": record_envelope_to_wire(cast(GovernanceRecordEnvelope, row))}
                for key, row in self.internal_all("governance_records")
            ],
            "operations": [operation_to_wire(row) for row in self.operations],
        }


def build_policy_envelope(
    payload: GovernancePolicyPayload,
    *,
    logical_tick: int,
    service_actor: str,
    original_actor: str,
    idempotency_key: str,
    request_fingerprint: str,
    owner_ids: Iterable[str] = (),
    subject_ids: Iterable[str] = (),
    previous: GovernancePolicyEnvelope | None = None,
) -> GovernancePolicyEnvelope:
    kind, stable_id, revision, semantic_digest, derived_owners, derived_subjects = (
        _policy_identity(payload)
    )
    owners = _canonical_texts((*derived_owners, *owner_ids), "owner_ids")
    subjects = _canonical_texts((*derived_subjects, *subject_ids), "subject_ids")
    if kind == "ads_campaign_terms" and not owners:
        raise MarketGovernancePersistenceError(
            "campaign terms require trusted catalog owner metadata"
        )
    previous_digest = None
    if previous is not None:
        validate_policy_envelope(previous)
        if (
            previous.kind != kind
            or previous.stable_id != stable_id
            or previous.revision + 1 != revision
        ):
            raise MarketGovernancePersistenceError("policy predecessor identity mismatch")
        previous_digest = previous.envelope_digest
    candidate = GovernancePolicyEnvelope(
        kind=kind,
        stable_id=stable_id,
        revision=revision,
        previous_envelope_digest=previous_digest,
        semantic_digest=semantic_digest,
        owner_ids=owners,
        subject_ids=subjects,
        indexes=_policy_indexes(payload),
        logical_tick=_nonnegative_int(logical_tick, "logical_tick"),
        service_actor=_text(service_actor, "service_actor"),
        original_actor=_text(original_actor, "original_actor"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
        request_fingerprint=_digest_text(request_fingerprint, "request_fingerprint"),
        payload=payload,
        envelope_digest="",
    )
    sealed = _seal_policy_envelope(candidate)
    validate_policy_envelope(sealed)
    return sealed


def build_record_envelope(
    payload: GovernanceRecordPayload,
    *,
    logical_tick: int,
    service_actor: str,
    original_actor: str,
    idempotency_key: str,
    request_fingerprint: str,
    subject_ids: Iterable[str] = (),
    previous: GovernanceRecordEnvelope | None = None,
) -> GovernanceRecordEnvelope:
    kind, stable_id, version, semantic_digest, owners, derived_subjects = (
        _record_identity(payload)
    )
    subjects = _canonical_texts((*derived_subjects, *subject_ids), "subject_ids")
    if kind == "governance_resolution_decision" and not subjects:
        raise MarketGovernancePersistenceError(
            "governance decision requires trusted case subjects"
        )
    previous_digest = None
    if previous is not None:
        validate_record_envelope(previous)
        if (
            previous.kind != kind
            or previous.stable_id != stable_id
            or previous.version + 1 != version
        ):
            raise MarketGovernancePersistenceError("record predecessor identity mismatch")
        previous_digest = previous.envelope_digest
    candidate = GovernanceRecordEnvelope(
        kind=kind,
        stable_id=stable_id,
        version=version,
        previous_envelope_digest=previous_digest,
        semantic_digest=semantic_digest,
        owner_ids=owners,
        subject_ids=subjects,
        indexes=_record_indexes(payload),
        logical_tick=_nonnegative_int(logical_tick, "logical_tick"),
        service_actor=_text(service_actor, "service_actor"),
        original_actor=_text(original_actor, "original_actor"),
        idempotency_key=_text(idempotency_key, "idempotency_key"),
        request_fingerprint=_digest_text(request_fingerprint, "request_fingerprint"),
        payload=payload,
        envelope_digest="",
    )
    sealed = _seal_record_envelope(candidate)
    validate_record_envelope(sealed)
    return sealed


def build_projection_effect(
    *,
    table: ProjectionTable,
    key: str,
    payload: Mapping[str, Any],
    before_digest: str | None = None,
) -> GovernanceProjectionEffect:
    normalized = _validate_projection_payload(table, payload)
    if before_digest is not None:
        _digest_text(before_digest, "before_digest")
    contract = {
        "schema_id": GOVERNANCE_PROJECTION_EFFECT_SCHEMA,
        "table": table,
        "key": _text(key, "key"),
        "before_digest": before_digest,
        "payload": normalized,
    }
    result = GovernanceProjectionEffect(
        table=table,
        key=cast(str, contract["key"]),
        before_digest=before_digest,
        payload=normalized,
        effect_digest=canonical_digest(contract),
    )
    validate_projection_effect(result)
    return result


def build_governance_operation(
    *,
    operation: str,
    service_actor: str,
    original_actor: str,
    idempotency_key: str,
    request_fingerprint: str,
    logical_tick: int,
    result_refs: Iterable[GovernanceResultRef],
    projection_effects: Iterable[GovernanceProjectionEffect] = (),
) -> GovernanceOperationRecord:
    refs = tuple(sorted(result_refs))
    effects = tuple(projection_effects)
    if not refs:
        raise MarketGovernancePersistenceError("governance operation requires a result")
    contract = {
        "schema_id": GOVERNANCE_OPERATION_SCHEMA,
        "operation": _text(operation, "operation"),
        "service_actor": _text(service_actor, "service_actor"),
        "original_actor": _text(original_actor, "original_actor"),
        "idempotency_key": _text(idempotency_key, "idempotency_key"),
        "request_fingerprint": _digest_text(
            request_fingerprint, "request_fingerprint"
        ),
        "logical_tick": _nonnegative_int(logical_tick, "logical_tick"),
        "result_refs": [result_ref_to_wire(item) for item in refs],
        "projection_effect_digest": canonical_digest(
            [projection_effect_to_wire(item) for item in effects]
        ),
    }
    digest = canonical_digest(contract)
    result = GovernanceOperationRecord(
        operation_id=f"governance-op:{digest[:32]}",
        operation=cast(str, contract["operation"]),
        service_actor=cast(str, contract["service_actor"]),
        original_actor=cast(str, contract["original_actor"]),
        idempotency_key=cast(str, contract["idempotency_key"]),
        request_fingerprint=cast(str, contract["request_fingerprint"]),
        logical_tick=cast(int, contract["logical_tick"]),
        result_refs=refs,
        projection_effect_digest=cast(str, contract["projection_effect_digest"]),
        operation_digest=digest,
    )
    validate_governance_operation(result)
    return result


def build_governance_commit(
    *,
    expected_logical_tick: int,
    logical_tick: int,
    context_digest: str,
    writes: Iterable[GovernanceWrite],
    projection_effects: Iterable[GovernanceProjectionEffect],
    operation: GovernanceOperationRecord,
) -> GovernanceCommit:
    write_rows = tuple(writes)
    effect_rows = tuple(projection_effects)
    contract = {
        "schema_id": GOVERNANCE_COMMIT_SCHEMA,
        "expected_logical_tick": expected_logical_tick,
        "logical_tick": logical_tick,
        "context_digest": context_digest,
        "writes": [write_to_wire(row) for row in write_rows],
        "projection_effects": [projection_effect_to_wire(row) for row in effect_rows],
        "operation_digest": operation.operation_digest,
    }
    digest = canonical_digest(contract)
    result = GovernanceCommit(
        commit_id=f"governance-commit:{digest[:32]}",
        expected_logical_tick=expected_logical_tick,
        logical_tick=logical_tick,
        context_digest=context_digest,
        writes=write_rows,
        projection_effects=effect_rows,
        operation=operation,
        commit_digest=digest,
    )
    validate_governance_commit(result)
    return result


def result_ref(write: GovernanceWrite) -> GovernanceResultRef:
    validate_governance_write(write)
    return GovernanceResultRef(
        collection=write.collection,
        key=write.key,
        semantic_digest=write.value.semantic_digest,
    )


def validate_policy_envelope(row: GovernancePolicyEnvelope) -> None:
    if not isinstance(row, GovernancePolicyEnvelope):
        raise MarketGovernancePersistenceError("policy envelope has wrong type")
    if row.schema_id != GOVERNANCE_POLICY_ENVELOPE_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported policy envelope schema")
    kind, stable_id, revision, semantic, derived_owners, derived_subjects = (
        _policy_identity(row.payload)
    )
    if (row.kind, row.stable_id, row.revision, row.semantic_digest) != (
        kind,
        stable_id,
        revision,
        semantic,
    ):
        raise MarketGovernancePersistenceError("policy envelope metadata mismatch")
    _validate_envelope_common(row)
    if not set(derived_owners).issubset(row.owner_ids) or not set(
        derived_subjects
    ).issubset(row.subject_ids):
        raise MarketGovernancePersistenceError("policy protected subjects mismatch")
    if kind == "ads_campaign_terms" and not row.owner_ids:
        raise MarketGovernancePersistenceError("campaign policy owner is missing")
    if _policy_authority(row.payload) != row.service_actor:
        raise MarketGovernancePersistenceError("policy service authority mismatch")
    if row.envelope_digest != canonical_digest(
        policy_envelope_to_wire(row, include_digest=False)
    ):
        raise MarketGovernancePersistenceError("policy envelope digest mismatch")


def validate_record_envelope(row: GovernanceRecordEnvelope) -> None:
    if not isinstance(row, GovernanceRecordEnvelope):
        raise MarketGovernancePersistenceError("record envelope has wrong type")
    if row.schema_id != GOVERNANCE_RECORD_ENVELOPE_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported record envelope schema")
    kind, stable_id, version, semantic, owners, derived_subjects = _record_identity(
        row.payload
    )
    if (row.kind, row.stable_id, row.version, row.semantic_digest) != (
        kind,
        stable_id,
        version,
        semantic,
    ):
        raise MarketGovernancePersistenceError("record envelope metadata mismatch")
    _validate_envelope_common(row)
    if row.owner_ids != owners or not set(derived_subjects).issubset(row.subject_ids):
        raise MarketGovernancePersistenceError("record protected subjects mismatch")
    if _record_authority(row.payload) != row.service_actor:
        raise MarketGovernancePersistenceError("record service authority mismatch")
    if row.payload.idempotency_key != row.idempotency_key:
        raise MarketGovernancePersistenceError(
            "record payload and operation idempotency keys differ"
        )
    if kind == "governance_resolution_decision" and not row.subject_ids:
        raise MarketGovernancePersistenceError("decision subjects are missing")
    if row.envelope_digest != canonical_digest(
        record_envelope_to_wire(row, include_digest=False)
    ):
        raise MarketGovernancePersistenceError("record envelope digest mismatch")


def _validate_envelope_common(row: GovernanceEnvelope) -> None:
    _text(row.stable_id, "stable_id")
    if envelope_version(row) <= 0:
        raise MarketGovernancePersistenceError("envelope version must be positive")
    if envelope_version(row) == 1:
        if row.previous_envelope_digest is not None:
            raise MarketGovernancePersistenceError("first envelope has a predecessor")
    else:
        _digest_text(row.previous_envelope_digest, "previous_envelope_digest")
    _digest_text(row.semantic_digest, "semantic_digest")
    _digest_text(row.envelope_digest, "envelope_digest")
    if row.owner_ids != _canonical_texts(row.owner_ids, "owner_ids"):
        raise MarketGovernancePersistenceError("owner_ids are not canonical")
    if row.subject_ids != _canonical_texts(row.subject_ids, "subject_ids"):
        raise MarketGovernancePersistenceError("subject_ids are not canonical")
    if row.indexes != tuple(sorted(set(row.indexes))):
        raise MarketGovernancePersistenceError("governance indexes are not canonical")
    for index in row.indexes:
        _text(index.name, "index.name")
        _text(index.value, "index.value")
    _nonnegative_int(row.logical_tick, "logical_tick")
    _text(row.service_actor, "service_actor")
    _text(row.original_actor, "original_actor")
    _text(row.idempotency_key, "idempotency_key")
    _digest_text(row.request_fingerprint, "request_fingerprint")


def validate_governance_write(write: GovernanceWrite) -> None:
    if not isinstance(write, GovernanceWrite):
        raise MarketGovernancePersistenceError("governance write has wrong type")
    if write.collection == "governance_policies":
        if not isinstance(write.value, GovernancePolicyEnvelope):
            raise MarketGovernancePersistenceError(
                "policy collection requires a policy envelope"
            )
        validate_policy_envelope(write.value)
    elif write.collection == "governance_records":
        if not isinstance(write.value, GovernanceRecordEnvelope):
            raise MarketGovernancePersistenceError(
                "record collection requires a record envelope"
            )
        validate_record_envelope(write.value)
    else:
        raise MarketGovernancePersistenceError("unknown governance collection")
    if write.key != envelope_key(write.value):
        raise MarketGovernancePersistenceError("governance key is not canonical")


def validate_projection_effect(effect: GovernanceProjectionEffect) -> None:
    if not isinstance(effect, GovernanceProjectionEffect):
        raise MarketGovernancePersistenceError("projection effect has wrong type")
    if effect.schema_id != GOVERNANCE_PROJECTION_EFFECT_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported projection effect schema")
    _text(effect.key, "projection key")
    if effect.before_digest is not None:
        _digest_text(effect.before_digest, "before_digest")
    normalized = _validate_projection_payload(effect.table, effect.payload)
    contract = {
        "schema_id": effect.schema_id,
        "table": effect.table,
        "key": effect.key,
        "before_digest": effect.before_digest,
        "payload": normalized,
    }
    if effect.effect_digest != canonical_digest(contract):
        raise MarketGovernancePersistenceError("projection effect digest mismatch")


def validate_governance_operation(operation: GovernanceOperationRecord) -> None:
    if not isinstance(operation, GovernanceOperationRecord):
        raise MarketGovernancePersistenceError("operation has wrong type")
    if operation.schema_id != GOVERNANCE_OPERATION_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported operation schema")
    _text(operation.operation, "operation")
    _text(operation.service_actor, "service_actor")
    _text(operation.original_actor, "original_actor")
    _text(operation.idempotency_key, "idempotency_key")
    _digest_text(operation.request_fingerprint, "request_fingerprint")
    _nonnegative_int(operation.logical_tick, "logical_tick")
    if not operation.result_refs or operation.result_refs != tuple(
        sorted(set(operation.result_refs))
    ):
        raise MarketGovernancePersistenceError("operation results are not canonical")
    for ref in operation.result_refs:
        _text(ref.key, "result key")
        _digest_text(ref.semantic_digest, "result semantic_digest")
    _digest_text(operation.projection_effect_digest, "projection_effect_digest")
    expected = canonical_digest(operation_to_wire(operation, include_identity=False))
    if operation.operation_digest != expected:
        raise MarketGovernancePersistenceError("operation digest mismatch")
    if operation.operation_id != f"governance-op:{expected[:32]}":
        raise MarketGovernancePersistenceError("operation id mismatch")


def validate_governance_commit(commit: GovernanceCommit) -> None:
    if not isinstance(commit, GovernanceCommit):
        raise MarketGovernancePersistenceError("commit has wrong type")
    if commit.schema_id != GOVERNANCE_COMMIT_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported commit schema")
    if commit.logical_tick != commit.expected_logical_tick + 1:
        raise MarketGovernancePersistenceError("governance commit must advance one tick")
    _digest_text(commit.context_digest, "context_digest")
    if not commit.writes:
        raise MarketGovernancePersistenceError("governance commit has no typed rows")
    for write in commit.writes:
        validate_governance_write(write)
        row = write.value
        if (
            row.logical_tick != commit.logical_tick
            or row.service_actor != commit.operation.service_actor
            or row.original_actor != commit.operation.original_actor
            or row.idempotency_key != commit.operation.idempotency_key
            or row.request_fingerprint != commit.operation.request_fingerprint
        ):
            raise MarketGovernancePersistenceError(
                "governance row is outside operation authority boundary"
            )
    for effect in commit.projection_effects:
        validate_projection_effect(effect)
    validate_governance_operation(commit.operation)
    refs = tuple(sorted(result_ref(write) for write in commit.writes))
    if not set(commit.operation.result_refs).issubset(refs):
        raise MarketGovernancePersistenceError(
            "operation result is not in the same governance commit"
        )
    effect_digest = canonical_digest(
        [projection_effect_to_wire(row) for row in commit.projection_effects]
    )
    if effect_digest != commit.operation.projection_effect_digest:
        raise MarketGovernancePersistenceError("projection effect binding mismatch")
    expected = canonical_digest(commit_to_wire(commit, include_identity=False))
    if commit.commit_digest != expected:
        raise MarketGovernancePersistenceError("commit digest mismatch")
    if commit.commit_id != f"governance-commit:{expected[:32]}":
        raise MarketGovernancePersistenceError("commit id mismatch")


def envelope_version(row: GovernanceEnvelope) -> int:
    return row.revision if isinstance(row, GovernancePolicyEnvelope) else row.version


def envelope_key(row: GovernanceEnvelope) -> str:
    collection: GovernanceCollection = (
        "governance_policies"
        if isinstance(row, GovernancePolicyEnvelope)
        else "governance_records"
    )
    return envelope_key_from_parts(
        collection, row.kind, row.stable_id, envelope_version(row)
    )


def envelope_key_from_parts(
    collection: GovernanceCollection,
    kind: str,
    stable_id: str,
    version: int,
) -> str:
    if collection not in {"governance_policies", "governance_records"}:
        raise MarketGovernancePersistenceError("unknown governance collection")
    _text(kind, "kind")
    _text(stable_id, "stable_id")
    if _positive_int(version, "version") != version:
        raise AssertionError("unreachable")
    return f"{kind}:{stable_id}:{version}"


def policy_envelope_to_wire(
    row: GovernancePolicyEnvelope, *, include_digest: bool = True
) -> dict[str, Any]:
    value = {
        "schema_id": row.schema_id,
        "kind": row.kind,
        "stable_id": row.stable_id,
        "revision": row.revision,
        "previous_envelope_digest": row.previous_envelope_digest,
        "semantic_digest": row.semantic_digest,
        "owner_ids": list(row.owner_ids),
        "subject_ids": list(row.subject_ids),
        "indexes": [index_to_wire(item) for item in row.indexes],
        "logical_tick": row.logical_tick,
        "service_actor": row.service_actor,
        "original_actor": row.original_actor,
        "idempotency_key": row.idempotency_key,
        "request_fingerprint": row.request_fingerprint,
        "payload": policy_payload_to_wire(row.payload),
    }
    if include_digest:
        value["envelope_digest"] = row.envelope_digest
    return value


def record_envelope_to_wire(
    row: GovernanceRecordEnvelope, *, include_digest: bool = True
) -> dict[str, Any]:
    value = {
        "schema_id": row.schema_id,
        "kind": row.kind,
        "stable_id": row.stable_id,
        "version": row.version,
        "previous_envelope_digest": row.previous_envelope_digest,
        "semantic_digest": row.semantic_digest,
        "owner_ids": list(row.owner_ids),
        "subject_ids": list(row.subject_ids),
        "indexes": [index_to_wire(item) for item in row.indexes],
        "logical_tick": row.logical_tick,
        "service_actor": row.service_actor,
        "original_actor": row.original_actor,
        "idempotency_key": row.idempotency_key,
        "request_fingerprint": row.request_fingerprint,
        "payload": record_payload_to_wire(row.payload),
    }
    if include_digest:
        value["envelope_digest"] = row.envelope_digest
    return value


def policy_envelope_from_wire(value: Mapping[str, Any]) -> GovernancePolicyEnvelope:
    row = _strict_mapping(value, "policy envelope", _POLICY_ENVELOPE_FIELDS)
    if row["schema_id"] != GOVERNANCE_POLICY_ENVELOPE_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported policy envelope schema")
    kind = _policy_kind(row["kind"])
    payload = policy_payload_from_wire(kind, _mapping(row["payload"], "payload"))
    result = GovernancePolicyEnvelope(
        kind=kind,
        stable_id=_wire_text(row, "stable_id"),
        revision=_wire_positive_int(row, "revision"),
        previous_envelope_digest=_wire_optional_digest(row, "previous_envelope_digest"),
        semantic_digest=_wire_digest(row, "semantic_digest"),
        owner_ids=_wire_text_tuple(row, "owner_ids"),
        subject_ids=_wire_text_tuple(row, "subject_ids"),
        indexes=_wire_indexes(row, "indexes"),
        logical_tick=_wire_nonnegative_int(row, "logical_tick"),
        service_actor=_wire_text(row, "service_actor"),
        original_actor=_wire_text(row, "original_actor"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        request_fingerprint=_wire_digest(row, "request_fingerprint"),
        payload=payload,
        envelope_digest=_wire_digest(row, "envelope_digest"),
    )
    validate_policy_envelope(result)
    return result


def record_envelope_from_wire(value: Mapping[str, Any]) -> GovernanceRecordEnvelope:
    row = _strict_mapping(value, "record envelope", _RECORD_ENVELOPE_FIELDS)
    if row["schema_id"] != GOVERNANCE_RECORD_ENVELOPE_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported record envelope schema")
    kind = _record_kind(row["kind"])
    payload = record_payload_from_wire(kind, _mapping(row["payload"], "payload"))
    result = GovernanceRecordEnvelope(
        kind=kind,
        stable_id=_wire_text(row, "stable_id"),
        version=_wire_positive_int(row, "version"),
        previous_envelope_digest=_wire_optional_digest(row, "previous_envelope_digest"),
        semantic_digest=_wire_digest(row, "semantic_digest"),
        owner_ids=_wire_text_tuple(row, "owner_ids"),
        subject_ids=_wire_text_tuple(row, "subject_ids"),
        indexes=_wire_indexes(row, "indexes"),
        logical_tick=_wire_nonnegative_int(row, "logical_tick"),
        service_actor=_wire_text(row, "service_actor"),
        original_actor=_wire_text(row, "original_actor"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        request_fingerprint=_wire_digest(row, "request_fingerprint"),
        payload=payload,
        envelope_digest=_wire_digest(row, "envelope_digest"),
    )
    validate_record_envelope(result)
    return result


def policy_payload_to_wire(payload: GovernancePolicyPayload) -> dict[str, Any]:
    if isinstance(payload, AdsCampaignTerms):
        validate_ads_campaign_terms(payload)
        return {
            "schema_id": payload.schema_id,
            "campaign_id": payload.campaign_id,
            "budget_cents": payload.budget_cents,
            "currency": payload.currency,
            "starts_at_tick": payload.starts_at_tick,
            "ends_at_tick": payload.ends_at_tick,
            "placements": [
                {
                    "sku_id": item.sku_id,
                    "bid_cents": item.bid_cents,
                    "fee_cents": item.fee_cents,
                }
                for item in payload.placements
            ],
            "issued_by_id": payload.issued_by_id,
            "terms_digest": payload.terms_digest,
        }
    if isinstance(payload, ReviewAccountBinding):
        validate_review_account_binding(payload, expected_authority=payload.authority_id)
        return cast(dict[str, Any], _json_value(asdict(payload)))
    if isinstance(payload, ReputationPolicyRevision):
        validate_reputation_policy(payload)
        return cast(dict[str, Any], _json_value(asdict(payload)))
    if isinstance(payload, RemediationBlueprint):
        validate_remediation_blueprint(payload)
        return cast(dict[str, Any], _json_value(asdict(payload)))
    raise MarketGovernancePersistenceError("unsupported governance policy payload")


def policy_payload_from_wire(
    kind: GovernancePolicyKind, value: Mapping[str, Any]
) -> GovernancePolicyPayload:
    if kind == "ads_campaign_terms":
        row = _strict_mapping(value, kind, _ADS_TERMS_FIELDS)
        placements = tuple(
            AdsPlacementTerms(
                sku_id=_wire_text(item, "sku_id"),
                bid_cents=_wire_nonnegative_int(item, "bid_cents"),
                fee_cents=_wire_nonnegative_int(item, "fee_cents"),
            )
            for item in (
                _strict_mapping(
                    _mapping(entry, "placement"),
                    "placement",
                    _ADS_PLACEMENT_FIELDS,
                )
                for entry in _wire_list(row, "placements")
            )
        )
        result: GovernancePolicyPayload = AdsCampaignTerms(
            campaign_id=_wire_text(row, "campaign_id"),
            budget_cents=_wire_positive_int(row, "budget_cents"),
            currency=_wire_text(row, "currency"),
            starts_at_tick=_wire_nonnegative_int(row, "starts_at_tick"),
            ends_at_tick=_wire_nonnegative_int(row, "ends_at_tick"),
            placements=placements,
            issued_by_id=_wire_text(row, "issued_by_id"),
            terms_digest=_wire_digest(row, "terms_digest"),
            schema_id=_wire_schema(row, ADS_CAMPAIGN_TERMS_SCHEMA),
        )
        validate_ads_campaign_terms(cast(AdsCampaignTerms, result))
        return result
    if kind == "review_account_binding":
        row = _strict_mapping(value, kind, _REVIEW_ACCOUNT_FIELDS)
        result = ReviewAccountBinding(
            reviewer_id=_wire_text(row, "reviewer_id"),
            account_created_at_tick=_wire_nonnegative_int(
                row, "account_created_at_tick"
            ),
            burst_group_id=_wire_optional_text(row, "burst_group_id"),
            authority_id=_wire_text(row, "authority_id"),
            binding_digest=_wire_digest(row, "binding_digest"),
            schema_id=_wire_schema(row, REVIEW_ACCOUNT_BINDING_SCHEMA),
        )
        validate_review_account_binding(result, expected_authority=result.authority_id)
        return result
    if kind == "reputation_policy_revision":
        row = _strict_mapping(value, kind, _REPUTATION_POLICY_FIELDS)
        result = ReputationPolicyRevision(
            policy_id=_wire_text(row, "policy_id"),
            revision=_wire_positive_int(row, "revision"),
            previous_digest=_wire_optional_digest(row, "previous_digest"),
            effective_tick=_wire_nonnegative_int(row, "effective_tick"),
            published_by_id=_wire_text(row, "published_by_id"),
            fulfilled_order_bps=_wire_bps(row, "fulfilled_order_bps"),
            disputed_order_bps=_wire_bps(row, "disputed_order_bps"),
            refund_bps=_wire_bps(row, "refund_bps"),
            remediation_verified_bps=_wire_bps(row, "remediation_verified_bps"),
            compliance_violation_bps=_wire_bps(row, "compliance_violation_bps"),
            policy_digest=_wire_digest(row, "policy_digest"),
            schema_id=_wire_schema(row, REPUTATION_POLICY_SCHEMA),
        )
        validate_reputation_policy(result)
        return result
    row = _strict_mapping(value, kind, _REMEDIATION_BLUEPRINT_FIELDS)
    steps = tuple(
        RemediationBlueprintStep(
            action_kind=_wire_text(item, "action_kind"),
            prerequisite_sequence_nos=_wire_int_tuple(
                item, "prerequisite_sequence_nos"
            ),
        )
        for item in (
            _strict_mapping(
                _mapping(entry, "blueprint step"),
                "blueprint step",
                _REMEDIATION_BLUEPRINT_STEP_FIELDS,
            )
            for entry in _wire_list(row, "steps")
        )
    )
    result = RemediationBlueprint(
        blueprint_id=_wire_text(row, "blueprint_id"),
        governance_case_kind=_wire_text(row, "governance_case_kind"),
        steps=steps,
        issued_by_id=_wire_text(row, "issued_by_id"),
        blueprint_digest=_wire_digest(row, "blueprint_digest"),
        schema_id=_wire_schema(row, REMEDIATION_BLUEPRINT_SCHEMA),
    )
    validate_remediation_blueprint(result)
    return result


def record_payload_to_wire(payload: GovernanceRecordPayload) -> dict[str, Any]:
    if isinstance(payload, Campaign):
        return campaign_to_wire(payload)
    if isinstance(payload, ReviewEvidence):
        return review_evidence_to_wire(payload)
    if isinstance(payload, ReviewAggregate):
        return review_aggregate_to_wire(payload)
    if isinstance(payload, MarketSignal):
        return market_signal_to_wire(payload)
    if isinstance(payload, GovernanceCase):
        return governance_case_to_wire(payload)
    if isinstance(payload, GovernanceResponseAttestation):
        _validate_response_standalone(payload)
        return cast(dict[str, Any], _json_value(asdict(payload)))
    if isinstance(payload, GovernanceResolutionDecision):
        _validate_decision_standalone(payload)
        return cast(dict[str, Any], _json_value(asdict(payload)))
    if isinstance(payload, ReputationEvent):
        return reputation_event_to_wire(payload)
    if isinstance(payload, RemediationPlan):
        return remediation_plan_to_wire(payload)
    if isinstance(payload, RankingContext):
        return ranking_context_to_wire(payload, server_id=payload.authored_by)
    raise MarketGovernancePersistenceError("unsupported governance record payload")


def record_payload_from_wire(
    kind: GovernanceRecordKind, value: Mapping[str, Any]
) -> GovernanceRecordPayload:
    if kind == "campaign":
        return coerce_campaign(value)
    if kind == "review_evidence":
        return coerce_review_evidence(value)
    if kind == "review_aggregate":
        return coerce_review_aggregate(value)
    if kind == "market_signal":
        return coerce_market_signal(value)
    if kind == "governance_case":
        return coerce_governance_case(value)
    if kind == "reputation_event":
        return coerce_reputation_event(value)
    if kind == "remediation_plan":
        return coerce_remediation_plan(value)
    if kind == "ranking_context":
        authored_by = value.get("authored_by")
        return coerce_ranking_context(value, server_id=_text(authored_by, "authored_by"))
    if kind == "governance_response_attestation":
        row = _strict_mapping(value, kind, _RESPONSE_FIELDS)
        result: GovernanceRecordPayload = GovernanceResponseAttestation(
            response_id=_wire_text(row, "response_id"),
            case_id=_wire_text(row, "case_id"),
            case_digest=_wire_digest(row, "case_digest"),
            subject_merchant_id=_wire_text(row, "subject_merchant_id"),
            response_kind=_wire_text(row, "response_kind"),
            signal_digests=_wire_digest_tuple(row, "signal_digests"),
            submitted_at_tick=_wire_nonnegative_int(row, "submitted_at_tick"),
            authored_by=_wire_text(row, "authored_by"),
            idempotency_key=_wire_text(row, "idempotency_key"),
            request_fingerprint=_wire_digest(row, "request_fingerprint"),
            provenance_digests=_wire_digest_tuple(row, "provenance_digests"),
            attestation_digest=_wire_digest(row, "attestation_digest"),
            schema_id=_wire_schema(row, GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA),
        )
        _validate_response_standalone(cast(GovernanceResponseAttestation, result))
        return result
    row = _strict_mapping(value, kind, _DECISION_FIELDS)
    result = GovernanceResolutionDecision(
        decision_id=_wire_text(row, "decision_id"),
        case_id=_wire_text(row, "case_id"),
        case_digest=_wire_digest(row, "case_digest"),
        resolution_kind=_wire_text(row, "resolution_kind"),
        target_status=_wire_text(row, "target_status"),
        resolution_code=_wire_text(row, "resolution_code"),
        policy_id=_wire_text(row, "policy_id"),
        policy_version=_wire_positive_int(row, "policy_version"),
        response_digests=_wire_digest_tuple(row, "response_digests"),
        decided_at_tick=_wire_nonnegative_int(row, "decided_at_tick"),
        authored_by=_wire_text(row, "authored_by"),
        idempotency_key=_wire_text(row, "idempotency_key"),
        decision_digest=_wire_digest(row, "decision_digest"),
        schema_id=_wire_schema(row, GOVERNANCE_RESOLUTION_DECISION_SCHEMA),
    )
    _validate_decision_standalone(result)
    return result


def write_to_wire(write: GovernanceWrite) -> dict[str, Any]:
    validate_governance_write(write)
    value = (
        policy_envelope_to_wire(cast(GovernancePolicyEnvelope, write.value))
        if write.collection == "governance_policies"
        else record_envelope_to_wire(cast(GovernanceRecordEnvelope, write.value))
    )
    return {"collection": write.collection, "key": write.key, "value": value}


def projection_effect_to_wire(effect: GovernanceProjectionEffect) -> dict[str, Any]:
    validate_projection_effect(effect)
    return {
        "schema_id": effect.schema_id,
        "table": effect.table,
        "key": effect.key,
        "before_digest": effect.before_digest,
        "payload": _json_value(effect.payload),
        "effect_digest": effect.effect_digest,
    }


def result_ref_to_wire(ref: GovernanceResultRef) -> dict[str, Any]:
    return {
        "collection": ref.collection,
        "key": ref.key,
        "semantic_digest": ref.semantic_digest,
    }


def operation_to_wire(
    operation: GovernanceOperationRecord, *, include_identity: bool = True
) -> dict[str, Any]:
    value = {
        "schema_id": operation.schema_id,
        "operation": operation.operation,
        "service_actor": operation.service_actor,
        "original_actor": operation.original_actor,
        "idempotency_key": operation.idempotency_key,
        "request_fingerprint": operation.request_fingerprint,
        "logical_tick": operation.logical_tick,
        "result_refs": [result_ref_to_wire(item) for item in operation.result_refs],
        "projection_effect_digest": operation.projection_effect_digest,
    }
    if include_identity:
        value["operation_id"] = operation.operation_id
        value["operation_digest"] = operation.operation_digest
    return value


def commit_to_wire(
    commit: GovernanceCommit, *, include_identity: bool = True
) -> dict[str, Any]:
    value = {
        "schema_id": commit.schema_id,
        "expected_logical_tick": commit.expected_logical_tick,
        "logical_tick": commit.logical_tick,
        "context_digest": commit.context_digest,
        "writes": [write_to_wire(row) for row in commit.writes],
        "projection_effects": [
            projection_effect_to_wire(row) for row in commit.projection_effects
        ],
        "operation_digest": commit.operation.operation_digest,
    }
    if include_identity:
        value["commit_id"] = commit.commit_id
        value["commit_digest"] = commit.commit_digest
    return value


def replay_governance(commits: Iterable[GovernanceCommit]) -> GovernanceTables:
    tables = GovernanceTables()
    for commit in commits:
        tables.apply_commit(commit)
    return tables


def canonical_digest(value: Any) -> str:
    try:
        body = json.dumps(
            _json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MarketGovernancePersistenceError(
            f"value is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(body).hexdigest()


def _seal_policy_envelope(row: GovernancePolicyEnvelope) -> GovernancePolicyEnvelope:
    from dataclasses import replace

    return replace(
        row,
        envelope_digest=canonical_digest(
            policy_envelope_to_wire(row, include_digest=False)
        ),
    )


def _seal_record_envelope(row: GovernanceRecordEnvelope) -> GovernanceRecordEnvelope:
    from dataclasses import replace

    return replace(
        row,
        envelope_digest=canonical_digest(
            record_envelope_to_wire(row, include_digest=False)
        ),
    )


def _policy_identity(
    payload: GovernancePolicyPayload,
) -> tuple[
    GovernancePolicyKind,
    str,
    int,
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    if isinstance(payload, AdsCampaignTerms):
        validate_ads_campaign_terms(payload)
        return (
            "ads_campaign_terms",
            payload.campaign_id,
            1,
            payload.terms_digest,
            (),
            (),
        )
    if isinstance(payload, ReviewAccountBinding):
        validate_review_account_binding(payload, expected_authority=payload.authority_id)
        return (
            "review_account_binding",
            payload.reviewer_id,
            1,
            payload.binding_digest,
            (payload.reviewer_id,),
            (),
        )
    if isinstance(payload, ReputationPolicyRevision):
        validate_reputation_policy(payload)
        return (
            "reputation_policy_revision",
            payload.policy_id,
            payload.revision,
            payload.policy_digest,
            (payload.published_by_id,),
            (),
        )
    if isinstance(payload, RemediationBlueprint):
        validate_remediation_blueprint(payload)
        return (
            "remediation_blueprint",
            payload.blueprint_id,
            1,
            payload.blueprint_digest,
            (payload.issued_by_id,),
            (),
        )
    raise MarketGovernancePersistenceError("unsupported governance policy payload")


def _record_identity(
    payload: GovernanceRecordPayload,
) -> tuple[
    GovernanceRecordKind,
    str,
    int,
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    if isinstance(payload, Campaign):
        campaign_to_wire(payload)
        return (
            "campaign",
            payload.campaign_id,
            payload.version,
            payload.campaign_digest,
            (payload.owner_merchant_id,),
            (),
        )
    if isinstance(payload, ReviewEvidence):
        review_evidence_to_wire(payload)
        return (
            "review_evidence",
            payload.review_id,
            payload.version,
            payload.evidence_digest,
            (payload.reviewer_id,),
            (payload.merchant_id,),
        )
    if isinstance(payload, ReviewAggregate):
        review_aggregate_to_wire(payload)
        return (
            "review_aggregate",
            payload.aggregate_id,
            payload.version,
            payload.aggregate_digest,
            (payload.merchant_id,),
            (),
        )
    if isinstance(payload, MarketSignal):
        market_signal_to_wire(payload)
        return (
            "market_signal",
            payload.signal_id,
            payload.version,
            payload.signal_digest,
            (),
            payload.subject_merchant_ids,
        )
    if isinstance(payload, GovernanceCase):
        governance_case_to_wire(payload)
        return (
            "governance_case",
            payload.case_id,
            payload.version,
            payload.case_digest,
            (),
            payload.subject_merchant_ids,
        )
    if isinstance(payload, GovernanceResponseAttestation):
        _validate_response_standalone(payload)
        return (
            "governance_response_attestation",
            payload.response_id,
            1,
            payload.attestation_digest,
            (payload.subject_merchant_id,),
            (),
        )
    if isinstance(payload, GovernanceResolutionDecision):
        _validate_decision_standalone(payload)
        return (
            "governance_resolution_decision",
            payload.decision_id,
            1,
            payload.decision_digest,
            (),
            (),
        )
    if isinstance(payload, ReputationEvent):
        reputation_event_to_wire(payload)
        return (
            "reputation_event",
            payload.merchant_id,
            payload.version,
            payload.event_digest,
            (payload.merchant_id,),
            (),
        )
    if isinstance(payload, RemediationPlan):
        remediation_plan_to_wire(payload)
        return (
            "remediation_plan",
            payload.plan_id,
            payload.version,
            payload.plan_digest,
            (payload.owner_merchant_id,),
            (),
        )
    if isinstance(payload, RankingContext):
        ranking_context_to_wire(payload, server_id=payload.authored_by)
        return (
            "ranking_context",
            payload.context_id,
            payload.version,
            payload.context_digest,
            (payload.buyer_id,),
            (),
        )
    raise MarketGovernancePersistenceError("unsupported governance record payload")


def _policy_authority(payload: GovernancePolicyPayload) -> str:
    if isinstance(payload, AdsCampaignTerms):
        return payload.issued_by_id
    if isinstance(payload, ReviewAccountBinding):
        return payload.authority_id
    if isinstance(payload, ReputationPolicyRevision):
        return payload.published_by_id
    return payload.issued_by_id


def _record_authority(payload: GovernanceRecordPayload) -> str:
    return payload.authored_by


def _policy_indexes(payload: GovernancePolicyPayload) -> tuple[GovernanceIndex, ...]:
    values: list[GovernanceIndex] = []
    if isinstance(payload, AdsCampaignTerms):
        values.append(GovernanceIndex("campaign_id", payload.campaign_id))
        values.extend(GovernanceIndex("sku_id", item.sku_id) for item in payload.placements)
    elif isinstance(payload, ReviewAccountBinding):
        values.append(GovernanceIndex("reviewer_id", payload.reviewer_id))
    elif isinstance(payload, ReputationPolicyRevision):
        values.append(GovernanceIndex("policy_id", payload.policy_id))
    else:
        values.extend(
            (
                GovernanceIndex("blueprint_id", payload.blueprint_id),
                GovernanceIndex("case_kind", payload.governance_case_kind),
            )
        )
    return tuple(sorted(set(values)))


def _record_indexes(payload: GovernanceRecordPayload) -> tuple[GovernanceIndex, ...]:
    values: list[GovernanceIndex] = []
    if isinstance(payload, Campaign):
        values.extend(
            (
                GovernanceIndex("campaign_id", payload.campaign_id),
                GovernanceIndex("merchant_id", payload.owner_merchant_id),
            )
        )
        values.extend(GovernanceIndex("sku_id", item.sku_id) for item in payload.placements)
    elif isinstance(payload, ReviewEvidence):
        values.extend(
            (
                GovernanceIndex("review_id", payload.review_id),
                GovernanceIndex("sku_id", payload.sku_id),
                GovernanceIndex("merchant_id", payload.merchant_id),
                GovernanceIndex("reviewer_id", payload.reviewer_id),
            )
        )
        if payload.order_id is not None:
            values.append(GovernanceIndex("order_id", payload.order_id))
    elif isinstance(payload, ReviewAggregate):
        values.extend(
            (
                GovernanceIndex("aggregate_id", payload.aggregate_id),
                GovernanceIndex("sku_id", payload.sku_id),
                GovernanceIndex("merchant_id", payload.merchant_id),
            )
        )
    elif isinstance(payload, MarketSignal):
        values.append(GovernanceIndex("signal_id", payload.signal_id))
        values.extend(
            GovernanceIndex("merchant_id", item)
            for item in payload.subject_merchant_ids
        )
    elif isinstance(payload, GovernanceCase):
        values.append(GovernanceIndex("case_id", payload.case_id))
        values.extend(
            GovernanceIndex("merchant_id", item)
            for item in payload.subject_merchant_ids
        )
    elif isinstance(payload, GovernanceResponseAttestation):
        values.extend(
            (
                GovernanceIndex("response_id", payload.response_id),
                GovernanceIndex("case_id", payload.case_id),
                GovernanceIndex("merchant_id", payload.subject_merchant_id),
            )
        )
    elif isinstance(payload, GovernanceResolutionDecision):
        values.extend(
            (
                GovernanceIndex("decision_id", payload.decision_id),
                GovernanceIndex("case_id", payload.case_id),
            )
        )
    elif isinstance(payload, ReputationEvent):
        values.extend(
            (
                GovernanceIndex("event_id", payload.event_id),
                GovernanceIndex("merchant_id", payload.merchant_id),
                GovernanceIndex("source_ref", payload.source_ref),
            )
        )
    elif isinstance(payload, RemediationPlan):
        values.extend(
            (
                GovernanceIndex("plan_id", payload.plan_id),
                GovernanceIndex("case_id", payload.governance_case_id),
                GovernanceIndex("merchant_id", payload.owner_merchant_id),
            )
        )
    else:
        values.extend(
            (
                GovernanceIndex("context_id", payload.context_id),
                GovernanceIndex("request_id", payload.request_id),
                GovernanceIndex("buyer_id", payload.buyer_id),
            )
        )
    return tuple(sorted(set(values)))


def _validate_response_standalone(value: GovernanceResponseAttestation) -> None:
    if value.schema_id != GOVERNANCE_RESPONSE_ATTESTATION_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported response schema")
    for name in (
        "response_id",
        "case_id",
        "subject_merchant_id",
        "response_kind",
        "authored_by",
        "idempotency_key",
    ):
        _text(getattr(value, name), name)
    _digest_text(value.case_digest, "case_digest")
    _digest_text(value.request_fingerprint, "request_fingerprint")
    _canonical_digests(value.signal_digests, "signal_digests")
    _canonical_digests(value.provenance_digests, "provenance_digests")
    contract = {
        "schema_id": value.schema_id,
        "response_id": value.response_id,
        "case_id": value.case_id,
        "case_digest": value.case_digest,
        "subject_merchant_id": value.subject_merchant_id,
        "response_kind": value.response_kind,
        "signal_digests": list(value.signal_digests),
        "submitted_at_tick": value.submitted_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
        "request_fingerprint": value.request_fingerprint,
        "provenance_digests": list(value.provenance_digests),
    }
    if value.attestation_digest != canonical_digest(contract):
        raise MarketGovernancePersistenceError("response semantic digest mismatch")


def _validate_decision_standalone(value: GovernanceResolutionDecision) -> None:
    if value.schema_id != GOVERNANCE_RESOLUTION_DECISION_SCHEMA:
        raise MarketGovernancePersistenceError("unsupported decision schema")
    for name in (
        "decision_id",
        "case_id",
        "resolution_kind",
        "target_status",
        "resolution_code",
        "policy_id",
        "authored_by",
        "idempotency_key",
    ):
        _text(getattr(value, name), name)
    _digest_text(value.case_digest, "case_digest")
    _canonical_digests(value.response_digests, "response_digests")
    contract = {
        "schema_id": value.schema_id,
        "decision_id": value.decision_id,
        "case_id": value.case_id,
        "case_digest": value.case_digest,
        "resolution_kind": value.resolution_kind,
        "target_status": value.target_status,
        "resolution_code": value.resolution_code,
        "policy_id": value.policy_id,
        "policy_version": value.policy_version,
        "response_digests": list(value.response_digests),
        "decided_at_tick": value.decided_at_tick,
        "authored_by": value.authored_by,
        "idempotency_key": value.idempotency_key,
    }
    if value.decision_digest != canonical_digest(contract):
        raise MarketGovernancePersistenceError("decision semantic digest mismatch")


def _validate_projection_payload(
    table: ProjectionTable, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if table == "reviews":
        row = _strict_mapping(payload, "review projection", _REVIEW_PROJECTION_FIELDS)
        normalized = {
            "review_id": _wire_text(row, "review_id"),
            "reviewer_id": _wire_text(row, "reviewer_id"),
            "sku_id": _wire_text(row, "sku_id"),
            "merchant_id": _wire_text(row, "merchant_id"),
            "rating": _wire_int_range(row, "rating", 1, 5),
            "text": _wire_text_allow_empty(row, "text"),
        }
        return normalized
    if table == "reputation":
        row = _strict_mapping(
            payload, "reputation projection", _REPUTATION_PROJECTION_FIELDS
        )
        return {
            "merchant_id": _wire_text(row, "merchant_id"),
            "event_id": _wire_text(row, "event_id"),
            "event_digest": _wire_digest(row, "event_digest"),
            "event_kind": _wire_text(row, "event_kind"),
            "outcome_bps": _wire_bps(row, "outcome_bps"),
        }
    raise MarketGovernancePersistenceError("unsupported projection table")


def _can_read(row: GovernanceEnvelope, caller: str | None) -> bool:
    if caller is None:
        return False
    if caller == "world" or caller == row.service_actor:
        return True
    return caller in set((*row.owner_ids, *row.subject_ids, row.original_actor))


def index_to_wire(value: GovernanceIndex) -> dict[str, str]:
    return {"name": value.name, "value": value.value}


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise MarketGovernancePersistenceError("non-finite float is forbidden")
        return value
    raise MarketGovernancePersistenceError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def _strict_mapping(
    value: Mapping[str, Any], label: str, fields: frozenset[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarketGovernancePersistenceError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise MarketGovernancePersistenceError(
            f"{label} fields are not exact: missing={missing!r}, unknown={unknown!r}"
        )
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MarketGovernancePersistenceError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketGovernancePersistenceError(f"{name} must be non-empty text")
    return value


def _text_allow_empty(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise MarketGovernancePersistenceError(f"{name} must be text")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MarketGovernancePersistenceError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MarketGovernancePersistenceError(f"{name} must be a positive integer")
    return value


def _digest_text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MarketGovernancePersistenceError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_texts(values: Iterable[str], name: str) -> tuple[str, ...]:
    return tuple(sorted({_text(value, name) for value in values}))


def _canonical_digests(values: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    for value in result:
        _digest_text(value, name)
    return result


def _wire_text(row: Mapping[str, Any], name: str) -> str:
    return _text(row[name], name)


def _wire_text_allow_empty(row: Mapping[str, Any], name: str) -> str:
    return _text_allow_empty(row[name], name)


def _wire_optional_text(row: Mapping[str, Any], name: str) -> str | None:
    value = row[name]
    return None if value is None else _text(value, name)


def _wire_nonnegative_int(row: Mapping[str, Any], name: str) -> int:
    return _nonnegative_int(row[name], name)


def _wire_positive_int(row: Mapping[str, Any], name: str) -> int:
    return _positive_int(row[name], name)


def _wire_int_range(
    row: Mapping[str, Any], name: str, lower: int, upper: int
) -> int:
    value = row[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= upper
    ):
        raise MarketGovernancePersistenceError(f"{name} is outside {lower}..{upper}")
    return value


def _wire_bps(row: Mapping[str, Any], name: str) -> int:
    return _wire_int_range(row, name, 0, 10_000)


def _wire_digest(row: Mapping[str, Any], name: str) -> str:
    return _digest_text(row[name], name)


def _wire_optional_digest(row: Mapping[str, Any], name: str) -> str | None:
    value = row[name]
    return None if value is None else _digest_text(value, name)


def _wire_list(row: Mapping[str, Any], name: str) -> list[Any]:
    value = row[name]
    if not isinstance(value, list):
        raise MarketGovernancePersistenceError(f"{name} must be a list")
    return value


def _wire_text_tuple(row: Mapping[str, Any], name: str) -> tuple[str, ...]:
    return tuple(_text(value, name) for value in _wire_list(row, name))


def _wire_digest_tuple(row: Mapping[str, Any], name: str) -> tuple[str, ...]:
    return tuple(_digest_text(value, name) for value in _wire_list(row, name))


def _wire_int_tuple(row: Mapping[str, Any], name: str) -> tuple[int, ...]:
    return tuple(_positive_int(value, name) for value in _wire_list(row, name))


def _wire_indexes(row: Mapping[str, Any], name: str) -> tuple[GovernanceIndex, ...]:
    result = []
    for item in _wire_list(row, name):
        mapping = _strict_mapping(
            _mapping(item, "index"), "index", frozenset({"name", "value"})
        )
        result.append(
            GovernanceIndex(_wire_text(mapping, "name"), _wire_text(mapping, "value"))
        )
    return tuple(result)


def _wire_schema(row: Mapping[str, Any], expected: str) -> str:
    value = _wire_text(row, "schema_id")
    if value != expected:
        raise MarketGovernancePersistenceError(f"unsupported payload schema {value!r}")
    return value


def _policy_kind(value: Any) -> GovernancePolicyKind:
    if value not in _POLICY_PAYLOAD_TYPES:
        raise MarketGovernancePersistenceError("unknown governance policy tag")
    return cast(GovernancePolicyKind, value)


def _record_kind(value: Any) -> GovernanceRecordKind:
    if value not in _RECORD_PAYLOAD_TYPES:
        raise MarketGovernancePersistenceError("unknown governance record tag")
    return cast(GovernanceRecordKind, value)


_POLICY_PAYLOAD_TYPES = {
    "ads_campaign_terms": AdsCampaignTerms,
    "review_account_binding": ReviewAccountBinding,
    "reputation_policy_revision": ReputationPolicyRevision,
    "remediation_blueprint": RemediationBlueprint,
}
_RECORD_PAYLOAD_TYPES = {
    "campaign": Campaign,
    "review_evidence": ReviewEvidence,
    "review_aggregate": ReviewAggregate,
    "market_signal": MarketSignal,
    "governance_case": GovernanceCase,
    "governance_response_attestation": GovernanceResponseAttestation,
    "governance_resolution_decision": GovernanceResolutionDecision,
    "reputation_event": ReputationEvent,
    "remediation_plan": RemediationPlan,
    "ranking_context": RankingContext,
}

_POLICY_ENVELOPE_FIELDS = frozenset(
    {
        "schema_id",
        "kind",
        "stable_id",
        "revision",
        "previous_envelope_digest",
        "semantic_digest",
        "owner_ids",
        "subject_ids",
        "indexes",
        "logical_tick",
        "service_actor",
        "original_actor",
        "idempotency_key",
        "request_fingerprint",
        "payload",
        "envelope_digest",
    }
)
_RECORD_ENVELOPE_FIELDS = frozenset(
    (_POLICY_ENVELOPE_FIELDS - {"revision"}) | {"version"}
)
_ADS_TERMS_FIELDS = frozenset(
    {
        "schema_id",
        "campaign_id",
        "budget_cents",
        "currency",
        "starts_at_tick",
        "ends_at_tick",
        "placements",
        "issued_by_id",
        "terms_digest",
    }
)
_ADS_PLACEMENT_FIELDS = frozenset({"sku_id", "bid_cents", "fee_cents"})
_REVIEW_ACCOUNT_FIELDS = frozenset(
    {
        "reviewer_id",
        "account_created_at_tick",
        "burst_group_id",
        "authority_id",
        "binding_digest",
        "schema_id",
    }
)
_REPUTATION_POLICY_FIELDS = frozenset(
    {
        "policy_id",
        "revision",
        "previous_digest",
        "effective_tick",
        "published_by_id",
        "fulfilled_order_bps",
        "disputed_order_bps",
        "refund_bps",
        "remediation_verified_bps",
        "compliance_violation_bps",
        "policy_digest",
        "schema_id",
    }
)
_REMEDIATION_BLUEPRINT_FIELDS = frozenset(
    {
        "blueprint_id",
        "governance_case_kind",
        "steps",
        "issued_by_id",
        "blueprint_digest",
        "schema_id",
    }
)
_REMEDIATION_BLUEPRINT_STEP_FIELDS = frozenset(
    {"action_kind", "prerequisite_sequence_nos"}
)
_RESPONSE_FIELDS = frozenset(
    {
        "response_id",
        "case_id",
        "case_digest",
        "subject_merchant_id",
        "response_kind",
        "signal_digests",
        "submitted_at_tick",
        "authored_by",
        "idempotency_key",
        "request_fingerprint",
        "provenance_digests",
        "attestation_digest",
        "schema_id",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "decision_id",
        "case_id",
        "case_digest",
        "resolution_kind",
        "target_status",
        "resolution_code",
        "policy_id",
        "policy_version",
        "response_digests",
        "decided_at_tick",
        "authored_by",
        "idempotency_key",
        "decision_digest",
        "schema_id",
    }
)
_REVIEW_PROJECTION_FIELDS = frozenset(
    {"review_id", "reviewer_id", "sku_id", "merchant_id", "rating", "text"}
)
_REPUTATION_PROJECTION_FIELDS = frozenset(
    {"merchant_id", "event_id", "event_digest", "event_kind", "outcome_bps"}
)


__all__ = [
    "GOVERNANCE_COMMIT_SCHEMA",
    "GOVERNANCE_OPERATION_SCHEMA",
    "GOVERNANCE_POLICY_ENVELOPE_SCHEMA",
    "GOVERNANCE_PROJECTION_EFFECT_SCHEMA",
    "GOVERNANCE_RECORD_ENVELOPE_SCHEMA",
    "GovernanceCollection",
    "GovernanceCommit",
    "GovernanceEnvelope",
    "GovernanceIndex",
    "GovernanceOperationRecord",
    "GovernancePolicyEnvelope",
    "GovernancePolicyKind",
    "GovernanceProjectionEffect",
    "GovernanceRecordEnvelope",
    "GovernanceRecordKind",
    "GovernanceResultRef",
    "GovernanceTables",
    "GovernanceWrite",
    "MarketGovernancePersistenceError",
    "build_governance_commit",
    "build_governance_operation",
    "build_policy_envelope",
    "build_projection_effect",
    "build_record_envelope",
    "canonical_digest",
    "commit_to_wire",
    "envelope_key",
    "envelope_key_from_parts",
    "envelope_version",
    "operation_to_wire",
    "policy_envelope_from_wire",
    "policy_envelope_to_wire",
    "policy_payload_from_wire",
    "policy_payload_to_wire",
    "projection_effect_to_wire",
    "record_envelope_from_wire",
    "record_envelope_to_wire",
    "record_payload_from_wire",
    "record_payload_to_wire",
    "replay_governance",
    "result_ref",
    "validate_governance_commit",
    "validate_governance_operation",
    "validate_governance_write",
    "validate_policy_envelope",
    "validate_projection_effect",
    "validate_record_envelope",
    "write_to_wire",
]
