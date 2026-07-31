"""Platform policy boundary for World-backed evidence, mandates, and claims."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from protocol.envelope import Envelope
from protocol.errors import SchemaError
from agents.evidence_acl_policy import derive_evidence_read_acl
from protocol.evidence_records import (
    EvidenceRecord,
    build_evidence_record,
    coerce_evidence_record,
    coerce_mandate_revision,
    evidence_record_to_dict,
    mandate_revision_to_dict,
)
from protocol.listing_claims import (
    ClaimEvidence,
    ListingClaim,
    claim_content_digest,
    correct_claim,
    create_listing_claim_draft,
    listing_claim_to_wire,
    publish_claim,
    retract_claim,
    seal_claim_evidence,
)
from world.evidence_contracts import (
    coerce_mandate_authority,
    mandate_authority_to_wire,
)
from world.types import SkuId

if TYPE_CHECKING:
    from agents.world_client import WorldClient


class ClaimStateRejected(SchemaError):
    """A compact merchant claim selected nonexistent authoritative state."""

    ALLOWED_REASON_CODES = frozenset(
        {
            "listing_claim_listing_not_found",
            "listing_claim_not_found",
            "listing_claim_already_exists",
            "listing_claim_evidence_not_found",
        }
    )

    def __init__(self, reason_code: str) -> None:
        if reason_code not in self.ALLOWED_REASON_CODES:
            raise ValueError("unsupported claim state rejection reason")
        self.reason_code = reason_code
        super().__init__(reason_code)


class EvidenceClaimsPolicy:
    """Validate typed actor requests before authoritative World mutation."""

    HANDLES = frozenset(
        {
            "commerce.publish_evidence_record",
            "platform.publish_evidence_record",
            "delegate.register_mandate_authority",
            "delegate.append_mandate_revision",
            "commerce.apply_listing_claim",
        }
    )

    def __init__(self, *, world_client: WorldClient) -> None:
        self._world_client = world_client

    #: Fields an actor may state about its own observation.
    ACTOR_EVIDENCE_FIELDS = frozenset({"kind", "subject_id", "facts", "trust"})

    def _seal_actor_evidence_record(
        self,
        env: Envelope,
        payload: dict[str, Any],
    ) -> EvidenceRecord:
        """Build a sealed record from an actor observation and World authority.

        Every authority-bearing field is derived here rather than copied from
        the request, so presenting one on the wire is rejected instead of
        silently honoured.
        """

        observation = _single(payload, "observation", "commerce.publish_evidence_record")
        if not isinstance(observation, dict):
            raise SchemaError("evidence observation must be an object")
        unknown = set(observation) - self.ACTOR_EVIDENCE_FIELDS
        if unknown:
            raise SchemaError(
                "evidence observation carries Platform-owned fields: "
                + ", ".join(sorted(unknown))
            )
        missing = {"kind", "subject_id", "facts"} - set(observation)
        if missing:
            raise SchemaError(
                "evidence observation is missing " + ", ".join(sorted(missing))
            )
        for field in ("kind", "subject_id"):
            if not isinstance(observation[field], str) or not observation[field].strip():
                raise SchemaError(f"evidence observation {field} must be non-empty text")
        for field in ("facts", "trust"):
            value = observation.get(field, {})
            if not isinstance(value, dict):
                raise SchemaError(f"evidence observation {field} must be an object")

        actor_id = env.from_
        subject_id = str(observation["subject_id"])
        read_acl = derive_evidence_read_acl(
            world_client=self._world_client,
            owner_id=actor_id,
            subject_id=subject_id,
        )
        return build_evidence_record(
            # A Platform-minted identity keeps one actor from squatting or
            # overwriting another actor's record id, and keeps retries of the
            # same request idempotent.
            record_id=_actor_evidence_record_id(actor_id, env.idempotency_key),
            kind=str(observation["kind"]),
            subject_id=subject_id,
            issuer_id=actor_id,
            facts=dict(observation["facts"]),
            trust=dict(observation.get("trust", {})),
            version=1,
            owner_id=actor_id,
            read_acl=read_acl,
            issued_at_tick=self._world_client.read_logical_time(
                caller="platform:evidence"
            ),
        )

    def handle(self, env: Envelope) -> Envelope | list[Envelope]:
        kind = str(env.action.get("kind", ""))
        payload = _exact_payload(env.action.get("payload"), kind)
        if kind in {
            "commerce.publish_evidence_record",
            "platform.publish_evidence_record",
        }:
            if env.to != "platform:evidence":
                raise SchemaError("evidence records must target platform:evidence")
            if kind == "platform.publish_evidence_record" and not env.from_.startswith(
                "runtime:"
            ):
                raise SchemaError(
                    "Platform evidence publication requires an external runtime service"
                )
            if kind == "commerce.publish_evidence_record" and _is_modelled_actor(
                env.from_
            ):
                # A Buyer or Merchant states what it observed. Issuer, ownership,
                # read authority, logical time, and the digest are sealed here
                # from authenticated context, so a hand-built envelope cannot
                # widen disclosure even though it never passed a model surface.
                # Trusted runtime and Platform services keep the sealed-record
                # path below; their authenticated identity is already authority.
                record = self._seal_actor_evidence_record(env, payload)
            else:
                record = coerce_evidence_record(_single(payload, "record", kind))
            persisted = self._world_client.persist_evidence_record(
                record=record,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
            )
            acknowledgement = _reply(
                env,
                from_="platform:evidence",
                kind="platform.evidence_record_persisted",
                payload={"record": evidence_record_to_dict(persisted)},
            )
            if persisted.kind != "independent_governance_audit":
                return acknowledgement
            if not persisted.owner_id.startswith("merchant:"):
                raise SchemaError(
                    "independent governance audit must be owned by a merchant"
                )
            # The owner notification is emitted only after the World commit
            # returned the sealed record.  It is not a substitute for the
            # external service acknowledgement and cannot create evidence.
            return [
                acknowledgement,
                Envelope(
                    msg_id=f"{env.msg_id}:owner-notice",
                    ts=env.ts,
                    from_="platform:evidence",
                    to=persisted.owner_id,
                    in_reply_to=env.msg_id,
                    idempotency_key=f"{env.idempotency_key}:owner-notice",
                    action={
                        "kind": "platform.evidence_record_persisted",
                        "payload": {"record": evidence_record_to_dict(persisted)},
                    },
                ),
            ]
        if kind == "delegate.register_mandate_authority":
            if env.to != "platform:mandate":
                raise SchemaError("mandate authority must target platform:mandate")
            authority = coerce_mandate_authority(
                _single(payload, "authority", kind)
            )
            persisted = self._world_client.register_mandate_authority(
                authority=authority,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
            )
            return _reply(
                env,
                from_="platform:mandate",
                kind="platform.mandate_authority_registered",
                payload={"authority": mandate_authority_to_wire(persisted)},
            )
        if kind == "delegate.append_mandate_revision":
            if env.to != "platform:mandate":
                raise SchemaError("mandate revision must target platform:mandate")
            revision = coerce_mandate_revision(_single(payload, "revision", kind))
            persisted = self._world_client.append_mandate_revision(
                revision=revision,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
            )
            return _reply(
                env,
                from_="platform:mandate",
                kind="platform.mandate_revision_appended",
                payload={"revision": mandate_revision_to_dict(persisted)},
            )
        if kind == "commerce.apply_listing_claim":
            if env.to != "platform:claims":
                raise SchemaError("listing claims must target platform:claims")
            claim, replayed = self._derive_listing_claim(payload, env)
            persisted = (
                claim
                if replayed
                else self._world_client.apply_listing_claim(
                    claim=claim,
                    original_actor=env.from_,
                    idempotency_key=env.idempotency_key,
                )
            )
            return _reply(
                env,
                from_="platform:claims",
                kind="platform.listing_claim_updated",
                payload={"claim": listing_claim_to_wire(persisted)},
            )
        raise SchemaError(f"unsupported evidence/claim action: {kind!r}")

    def _derive_listing_claim(
        self, payload: dict[str, Any], env: Envelope
    ) -> tuple[ListingClaim, bool]:
        """Turn a compact merchant intent into a canonical claim transition."""

        intent = _claim_intent(payload)
        claim_id = intent["claim_id"]
        listing_id = intent["listing_id"]
        subject = intent["subject"]
        operation = intent["operation"]

        listing = self._world_client.read(
            "catalog", SkuId(listing_id), caller="platform:claims"
        )
        if listing is None:
            raise ClaimStateRejected("listing_claim_listing_not_found")
        if str(listing.merchant_id) != env.from_:
            raise SchemaError(
                "authenticated merchant does not own the claimed listing"
            )
        current = self._world_client.read_listing_claim(
            claim_id, caller="platform:claims"
        )
        if current is not None and (
            current.listing_id != listing_id
            or current.subject != subject
            or current.merchant_id != env.from_
            or current.issuer_id != env.from_
        ):
            raise SchemaError("claim intent changes immutable claim identity")

        prior = None if current is None else next(
            (
                version
                for version in current.versions
                if version.idempotency_key == env.idempotency_key
            ),
            None,
        )
        if prior is not None:
            _validate_compact_claim_retry(intent, prior)
            # The exact operation is already authoritative World state.  A
            # replay returns that state and performs no new business write.
            return current, True

        logical_tick = self._world_client.read_logical_time(
            caller="platform:claims"
        )
        evidence = self._claim_evidence(
            intent,
            current=current,
            merchant_id=env.from_,
            logical_tick=logical_tick,
        )
        if operation == "draft":
            if current is not None:
                raise ClaimStateRejected("listing_claim_already_exists")
            return create_listing_claim_draft(
                claim_id=claim_id,
                listing_id=listing_id,
                merchant_id=env.from_,
                subject=subject,
                issuer_id=env.from_,
                content=intent["content"],
                logical_tick=logical_tick,
                idempotency_key=env.idempotency_key,
            ), False
        if current is None:
            raise ClaimStateRejected("listing_claim_not_found")
        if operation == "publish":
            return publish_claim(
                current,
                actor_id=env.from_,
                evidence=evidence,
                logical_tick=logical_tick,
                idempotency_key=env.idempotency_key,
            ), False
        if operation == "correct":
            return correct_claim(
                current,
                actor_id=env.from_,
                content=intent["content"],
                evidence=evidence,
                logical_tick=logical_tick,
                idempotency_key=env.idempotency_key,
            ), False
        return retract_claim(
            current,
            actor_id=env.from_,
            reason=intent["reason"],
            evidence=evidence,
            logical_tick=logical_tick,
            idempotency_key=env.idempotency_key,
        ), False

    def _claim_evidence(
        self,
        intent: dict[str, Any],
        *,
        current: ListingClaim | None,
        merchant_id: str,
        logical_tick: int,
    ) -> tuple[ClaimEvidence, ...]:
        source_ids = intent.get("evidence_record_ids", [])
        next_version = 1 if current is None else current.version + 1
        output: list[ClaimEvidence] = []
        for index, source_id in enumerate(source_ids, start=1):
            record = self._world_client.read_evidence_record(
                source_id, caller=merchant_id
            )
            if record is None:
                raise ClaimStateRejected("listing_claim_evidence_not_found")
            output.append(seal_claim_evidence(ClaimEvidence(
                evidence_id=(
                    f"claim-evidence:{intent['claim_id']}:{next_version}:"
                    f"{index}:{source_id}"
                ),
                source_id=record.record_id,
                claim_id=intent["claim_id"],
                listing_id=intent["listing_id"],
                merchant_id=merchant_id,
                subject=intent["subject"],
                source_digest=record.record_digest,
                observed_at_tick=logical_tick,
            )))
        return tuple(output)


def _exact_payload(value: Any, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{kind} payload must be an object")
    return value


def _is_modelled_actor(actor_id: str) -> bool:
    """True for the Buyer and Merchant roles a model can drive."""

    return actor_id.split(":", 1)[0] in {"buyer", "merchant"}


def _actor_evidence_record_id(actor_id: str, idempotency_key: str) -> str:
    """Mint one deterministic, actor-scoped evidence record identity."""

    digest = hashlib.sha256(
        json.dumps([actor_id, idempotency_key], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"evidence:{digest[:32]}"


def _single(payload: dict[str, Any], field: str, kind: str) -> Any:
    if set(payload) != {field}:
        raise SchemaError(f"{kind} payload requires exactly {field}")
    return payload[field]


def _claim_intent(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact public compact intent and reject authority fields."""

    base = {"claim_id", "listing_id", "subject", "operation"}
    operation = payload.get("operation")
    fields = {
        "draft": base | {"content"},
        "publish": base | {"evidence_record_ids"},
        "correct": base | {"content", "evidence_record_ids"},
        "retract": base | {"reason"},
    }.get(operation)
    if fields is None:
        raise SchemaError(
            "claim operation must be draft, publish, correct, or retract"
        )
    # Retraction may optionally cite evidence, but no other extra fields are
    # accepted.  This rejects merchant/issuer/version/tick/digest/history.
    allowed = fields | ({"evidence_record_ids"} if operation == "retract" else set())
    if not fields.issubset(payload) or not set(payload).issubset(allowed):
        raise SchemaError(
            f"compact {operation} claim intent has invalid fields"
        )
    for field in ("claim_id", "listing_id", "subject"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise SchemaError(f"claim intent {field} must be a non-empty string")
    if "content" in fields:
        content = payload.get("content")
        if not isinstance(content, dict) or not content:
            raise SchemaError("claim intent content must be a non-empty object")
    if "reason" in fields:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SchemaError("claim retraction reason must be non-empty")
    sources = payload.get("evidence_record_ids", [])
    if not isinstance(sources, list) or not all(
        isinstance(item, str) and item.strip() for item in sources
    ):
        raise SchemaError("evidence_record_ids must be a string array")
    if len(sources) != len(set(sources)):
        raise SchemaError("evidence_record_ids must not contain duplicates")
    if operation in {"publish", "correct"} and not sources:
        raise SchemaError(f"{operation} requires at least one evidence record")
    return {
        **payload,
        "evidence_record_ids": sorted(sources),
    }


def _validate_compact_claim_retry(intent: dict[str, Any], version: Any) -> None:
    """Fail a reused actor key unless the compact request is exactly the same."""

    if intent["operation"] != version.operation:
        raise SchemaError("claim idempotency key was reused for another operation")
    if "content" in intent and claim_content_digest(intent["content"]) != version.content_digest:
        raise SchemaError("claim idempotency retry changed content")
    if intent.get("reason") != version.reason:
        raise SchemaError("claim idempotency retry changed reason")
    supplied = tuple(sorted(intent.get("evidence_record_ids", [])))
    persisted = tuple(sorted(item.source_id for item in version.evidence))
    if supplied != persisted:
        raise SchemaError("claim idempotency retry changed evidence references")


def _reply(
    request: Envelope,
    *,
    from_: str,
    kind: str,
    payload: dict[str, Any],
) -> Envelope:
    return Envelope(
        msg_id=f"{request.msg_id}:{kind}",
        ts=request.ts,
        from_=from_,
        to=request.from_,
        in_reply_to=request.msg_id,
        idempotency_key=request.idempotency_key,
        action={"kind": kind, "payload": payload},
    )


__all__ = ["EvidenceClaimsPolicy"]
