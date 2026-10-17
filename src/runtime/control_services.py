"""Registered non-agent Runtime control services.

These services are deterministic environment components.  They are not buyer
or merchant agents, own no private agent memory, and can only answer the exact
Platform actions declared by their service kind.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from protocol.envelope import Envelope
from protocol.errors import SchemaError
from protocol.evidence_records import build_evidence_record, evidence_record_to_dict
from protocol.remediation_audit import (
    REMEDIATION_AUDITOR_SERVICE_ID,
    validate_remediation_audit_request,
)

if TYPE_CHECKING:
    from episode.types import ControlServiceSpec
    from world.state import World


class RuntimeControlService(Protocol):
    service_id: str
    inbound_action_kinds: frozenset[str]

    def handle(self, env: Envelope) -> Envelope | list[Envelope] | None:
        ...


class RemediationAuditorService:
    """Attest only a current pending step from authoritative World state."""

    inbound_action_kinds = frozenset(
        {
            "platform.remediation_audit_request",
            "platform.evidence_record_persisted",
        }
    )

    def __init__(self, *, service_id: str, world: "World") -> None:
        if service_id != REMEDIATION_AUDITOR_SERVICE_ID:
            raise ValueError(
                "remediation auditor must use the canonical runtime service id"
            )
        self.service_id = service_id
        self._world = world

    def handle(self, env: Envelope) -> Envelope | None:
        kind = str(env.action.get("kind", ""))
        if env.to != self.service_id or kind not in self.inbound_action_kinds:
            raise SchemaError("control service received an unsupported envelope")
        if kind == "platform.evidence_record_persisted":
            # Commit acknowledgements terminate at the external service.  The
            # Platform separately notifies the evidence owner.
            return None
        if env.from_ != "platform:remediation" or env.in_reply_to is None:
            raise SchemaError(
                "remediation audit requires a causally linked Platform request"
            )
        payload = validate_remediation_audit_request(
            env.action.get("payload")
        )
        self._verify_current_pending_step(payload)
        record_id = (
            "evidence:remediation-audit:"
            + str(payload["request_fingerprint"])[:24]
        )
        record = build_evidence_record(
            record_id=record_id,
            kind="independent_governance_audit",
            subject_id=str(payload["step_id"]),
            issuer_id=self.service_id,
            facts={
                "evidence_kind": str(payload["action_kind"]),
                "result": "passed",
                "plan_id": str(payload["plan_id"]),
                "plan_version": int(payload["plan_version"]),
                "plan_digest": str(payload["plan_digest"]),
                "sequence_no": int(payload["sequence_no"]),
                "request_msg_id": env.msg_id,
                "request_fingerprint": str(payload["request_fingerprint"]),
            },
            trust={"method": "independent_audit", "confidence_bps": 10_000},
            version=1,
            owner_id=str(payload["owner_merchant_id"]),
            read_acl=(
                str(payload["owner_merchant_id"]),
                "platform:governance",
                "platform:remediation",
            ),
            issued_at_tick=int(payload["world_tick"]),
        )
        return Envelope(
            msg_id=f"{env.msg_id}:publish-evidence",
            ts=env.ts,
            from_=self.service_id,
            to="platform:evidence",
            in_reply_to=env.msg_id,
            idempotency_key=f"{env.idempotency_key}:publish-evidence",
            action={
                "kind": "platform.publish_evidence_record",
                "payload": {"record": evidence_record_to_dict(record)},
            },
        )

    def _verify_current_pending_step(self, request: Mapping[str, Any]) -> None:
        rows = self._world.governance_history(
            "remediation_plan",
            str(request["plan_id"]),
            caller="platform:remediation",
        )
        if not rows:
            raise SchemaError("remediation audit references an unknown plan")
        plan = rows[-1]
        if (
            plan.version != request["plan_version"]
            or plan.plan_digest != request["plan_digest"]
            or plan.owner_merchant_id != request["owner_merchant_id"]
            or plan.status != "active"
            or self._world.logical_time != request["world_tick"]
        ):
            raise SchemaError("remediation audit request is stale or not authoritative")
        step = next(
            (row for row in plan.steps if row.step_id == request["step_id"]),
            None,
        )
        if step is None or (
            step.status != "pending"
            or step.sequence_no != request["sequence_no"]
            or step.action_kind != request["action_kind"]
        ):
            raise SchemaError("remediation audit request does not name a pending step")


def build_control_service(
    spec: "ControlServiceSpec", *, world: "World"
) -> RuntimeControlService:
    if spec.kind != "remediation_auditor":
        raise ValueError(f"unknown Runtime control service kind {spec.kind!r}")
    if spec.config:
        raise ValueError("remediation_auditor does not accept configuration fields")
    return RemediationAuditorService(service_id=spec.service_id, world=world)


__all__ = [
    "RemediationAuditorService",
    "RuntimeControlService",
    "build_control_service",
]
