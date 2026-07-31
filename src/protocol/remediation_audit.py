"""Canonical request binding for external remediation audit services."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from protocol.errors import SchemaError


REMEDIATION_AUDITOR_SERVICE_ID = "runtime:remediation-auditor"
REMEDIATION_AUDIT_REQUEST_FIELDS = frozenset(
    {
        "plan_id",
        "plan_version",
        "plan_digest",
        "owner_merchant_id",
        "step_id",
        "sequence_no",
        "action_kind",
        "world_tick",
        "request_fingerprint",
    }
)


def build_remediation_audit_request(
    *,
    plan_id: str,
    plan_version: int,
    plan_digest: str,
    owner_merchant_id: str,
    step_id: str,
    sequence_no: int,
    action_kind: str,
    world_tick: int,
) -> dict[str, Any]:
    """Build a self-verifying request bound to one World plan version."""

    body = {
        "plan_id": _text(plan_id, "plan_id"),
        "plan_version": _positive_int(plan_version, "plan_version"),
        "plan_digest": _digest(plan_digest, "plan_digest"),
        "owner_merchant_id": _text(owner_merchant_id, "owner_merchant_id"),
        "step_id": _text(step_id, "step_id"),
        "sequence_no": _positive_int(sequence_no, "sequence_no"),
        "action_kind": _text(action_kind, "action_kind"),
        "world_tick": _non_negative_int(world_tick, "world_tick"),
    }
    return {**body, "request_fingerprint": _fingerprint(body)}


def validate_remediation_audit_request(value: Any) -> dict[str, Any]:
    """Validate exact shape and return the canonical detached request."""

    if not isinstance(value, Mapping) or set(value) != REMEDIATION_AUDIT_REQUEST_FIELDS:
        raise SchemaError(
            "remediation audit request fields must be exactly: "
            + ", ".join(sorted(REMEDIATION_AUDIT_REQUEST_FIELDS))
        )
    expected = build_remediation_audit_request(
        plan_id=value.get("plan_id"),
        plan_version=value.get("plan_version"),
        plan_digest=value.get("plan_digest"),
        owner_merchant_id=value.get("owner_merchant_id"),
        step_id=value.get("step_id"),
        sequence_no=value.get("sequence_no"),
        action_kind=value.get("action_kind"),
        world_tick=value.get("world_tick"),
    )
    if value.get("request_fingerprint") != expected["request_fingerprint"]:
        raise SchemaError("remediation audit request fingerprint mismatch")
    return expected


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be non-empty text")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer")
    return value


def _digest(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise SchemaError(f"{name} must be a lowercase SHA-256 digest")
    return text


__all__ = [
    "REMEDIATION_AUDITOR_SERVICE_ID",
    "build_remediation_audit_request",
    "validate_remediation_audit_request",
]
