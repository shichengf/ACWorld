"""Agent authority for one authenticated CommerceWorld protocol-event turn.

The language model chooses only whether to process or reject the event and a
short reason.  Event identity, stream identity, order parties, lifecycle
revision, and World clock are framework authority.  They are validated from
the exact current Platform delivery and World records, then bound back into the
wire action by this module.  No benchmark case, scenario oracle, or expected
answer participates in that boundary.

An expired or lifecycle-stale event is still a legitimate delivery.  The
protocol permits its recipient to reject it durably, so such an authority
exposes only ``reject_current_event``.  Contradictory framework records and an
event that already has a receipt fail before model inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from agents.business_decision import BusinessIntentSpec
from agents.decision_errors import (
    ModelBusinessDecisionError,
    PlatformContractError,
    SemanticBoundaryError,
)
from protocol.envelope import Envelope
from protocol.errors import SchemaError
from protocol.event_receipts import (
    ProtocolEvent,
    ProtocolEventReceipt,
    protocol_event_from_json,
    protocol_event_receipt_from_json,
    protocol_event_to_dict,
    validate_protocol_event,
    validate_protocol_event_receipt,
)


_EVENTS_ENDPOINT = "platform:events"
_DELIVERY_KIND = "platform.deliver_protocol_event"
_PROCESS_OPERATION = "process_protocol_event"
_REJECT_OPERATION = "reject_protocol_event"
_MAX_REASON_LENGTH = 500
_ORDER_STATE_FIELDS = frozenset(
    {
        "order_id",
        "buyer_id",
        "merchant_id",
        "state",
        "state_revision",
        "operation_reference_digest",
        "logical_time",
    }
)


class ProtocolEventTurnError(SemanticBoundaryError):
    """Base error at the Agent protocol-event compiler boundary."""


class ProtocolEventFrameworkAuthorityError(
    PlatformContractError,
    ProtocolEventTurnError,
):
    """Authenticated Platform or current World authority is inconsistent."""


class ProtocolEventModelChoiceError(ModelBusinessDecisionError, ProtocolEventTurnError):
    """The model selected a business intent or argument outside its contract."""


@dataclass(frozen=True, slots=True)
class CompiledProtocolEventAction:
    """Exact wire action plus the framework authority used to compile it.

    The additional fields are local evidence.  Only ``destination`` and
    Routing and envelope fields are added by the Agent; none are model-authored.
    """

    operation: str
    payload: Mapping[str, Any]
    event: ProtocolEvent
    inbound_msg_id: str
    observed_order_state: str
    observed_state_revision: int
    logical_tick: int


@dataclass(frozen=True, slots=True)
class ProtocolEventTurnAuthority:
    """Validated authority for one exact pending protocol-event delivery."""

    actor_id: str
    inbound_msg_id: str
    event: ProtocolEvent
    observed_order_state: str
    observed_state_revision: int
    logical_tick: int
    process_allowed: bool
    stale_reason: str | None

    @classmethod
    def from_inbound(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        current_event: ProtocolEvent | Mapping[str, Any],
        current_order_state: Mapping[str, Any],
        current_receipts: Iterable[ProtocolEventReceipt | Mapping[str, Any]] = (),
        current_tick: int,
    ) -> "ProtocolEventTurnAuthority":
        """Validate one delivery against exact current World authority.

        ``current_event`` must be the persisted row returned for the delivered
        event id.  ``current_order_state`` is the exact result of
        ``world.read_order_protocol_state``.  ``current_receipts`` is the
        current stream's recipient decision set.  Callers must obtain all
        three through actor-authorized World reads for this turn.
        """

        _require_commercial_actor(actor_id)
        if not isinstance(inbound, Envelope):
            raise ProtocolEventFrameworkAuthorityError(
                "protocol-event authority requires an authenticated Envelope"
            )
        if inbound.from_ != _EVENTS_ENDPOINT:
            raise ProtocolEventFrameworkAuthorityError(
                "protocol-event delivery has the wrong Platform sender"
            )
        if inbound.to != actor_id:
            raise ProtocolEventFrameworkAuthorityError(
                "protocol-event delivery recipient does not match the Agent actor"
            )
        if set(inbound.action) != {"kind", "payload"}:
            raise ProtocolEventFrameworkAuthorityError(
                "protocol-event delivery action must contain exactly kind and payload"
            )
        if inbound.action.get("kind") != _DELIVERY_KIND:
            raise ProtocolEventFrameworkAuthorityError(
                "current inbound is not a protocol-event delivery"
            )
        payload = inbound.action.get("payload")
        if not isinstance(payload, Mapping) or set(payload) not in (
            {"event"},
            {"event", "event_authority"},
        ):
            raise ProtocolEventFrameworkAuthorityError(
                "protocol-event delivery payload must contain exactly one event "
                "and at most one exact authority package"
            )

        delivered = _coerce_event(payload["event"], "delivered protocol event")
        persisted = _coerce_event(current_event, "current World protocol event")
        if delivered != persisted or protocol_event_to_dict(delivered) != protocol_event_to_dict(
            persisted
        ):
            raise ProtocolEventFrameworkAuthorityError(
                "delivered protocol event is not the exact current World event"
            )
        event = persisted
        if event.binding.authority_id != _EVENTS_ENDPOINT or event.actor_id != _EVENTS_ENDPOINT:
            raise ProtocolEventFrameworkAuthorityError(
                "protocol event was not issued by platform:events authority"
            )
        if event.binding.recipient_id != actor_id:
            raise ProtocolEventFrameworkAuthorityError(
                "protocol event binding recipient does not match the Agent actor"
            )
        role = actor_id.split(":", 1)[0]
        expected_party = event.binding.buyer_id if role == "buyer" else event.binding.merchant_id
        if expected_party != actor_id:
            raise ProtocolEventFrameworkAuthorityError(
                "protocol event recipient role does not match the bound order party"
            )

        state = _coerce_order_state(current_order_state)
        if (
            state["order_id"] != event.binding.order_id
            or state["buyer_id"] != event.binding.buyer_id
            or state["merchant_id"] != event.binding.merchant_id
        ):
            raise ProtocolEventFrameworkAuthorityError(
                "current World order identity or parties disagree with the event binding"
            )
        tick = _framework_nonnegative_int(current_tick, "current World tick")
        if state["logical_time"] != tick:
            raise ProtocolEventFrameworkAuthorityError(
                "current World clock disagrees with the order protocol-state snapshot"
            )
        if tick < event.issued_at_tick:
            raise ProtocolEventFrameworkAuthorityError(
                "current World clock predates protocol-event issuance"
            )

        receipts = tuple(
            _coerce_receipt(value, "current World protocol receipt") for value in current_receipts
        )
        already_decided = False
        for receipt in receipts:
            if (
                receipt.binding.binding_digest != event.binding.binding_digest
                or receipt.binding != event.binding
            ):
                raise ProtocolEventFrameworkAuthorityError(
                    "current receipt set contains a cross-stream or cross-order record"
                )
            if receipt.event_id == event.event_id or receipt.event_digest == event.event_digest:
                if receipt.event_id != event.event_id or receipt.event_digest != event.event_digest:
                    raise ProtocolEventFrameworkAuthorityError(
                        "current receipt identity conflicts with the delivered event"
                    )
                already_decided = True

        stale: list[str] = []
        if tick > event.expires_at_tick:
            stale.append("expired")
        if state["state"] != event.required_order_state:
            stale.append("order state changed")
        if state["state_revision"] != event.required_state_revision:
            stale.append("order revision changed")
        if (
            event.reference_kind == "operation"
            and state["operation_reference_digest"] != event.reference_digest
        ):
            stale.append("operation reference changed")
        if already_decided:
            stale.append("already decided")
        stale_reason = None if not stale else "; ".join(stale)
        return cls(
            actor_id=actor_id,
            inbound_msg_id=inbound.msg_id,
            event=event,
            observed_order_state=state["state"],
            observed_state_revision=state["state_revision"],
            logical_tick=tick,
            process_allowed=not stale,
            stale_reason=stale_reason,
        )

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        """Return both model choices for the exact current event.

        Whether a callback is current, stale, expired, or duplicated is the
        business judgment under evaluation.  The Agent binds the event and its
        authority, but it must not remove the unsafe choice and thereby answer
        the benchmark for the model.
        """

        reject = _decision_spec(
            _REJECT_OPERATION,
            "Reject the current marketplace event with a concise reason.",
        )
        return (
            _decision_spec(
                _PROCESS_OPERATION,
                "Process the current marketplace event.",
            ),
            reject,
        )

    def compile(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CompiledProtocolEventAction:
        """Compile one high-level model choice into authority-bound business data."""

        if not isinstance(arguments, Mapping):
            raise ProtocolEventModelChoiceError("protocol-event tool arguments must be an object")
        if operation not in {_PROCESS_OPERATION, _REJECT_OPERATION}:
            raise ProtocolEventModelChoiceError(
                "model selected an intent outside the current protocol-event turn"
            )
        _require_choice_keys(arguments, required={"reason"})
        reason = _choice_reason(arguments.get("reason"))
        return CompiledProtocolEventAction(
            operation=operation,
            payload={
                "event_id": self.event.event_id,
                "reason": reason,
            },
            event=self.event,
            inbound_msg_id=self.inbound_msg_id,
            observed_order_state=self.observed_order_state,
            observed_state_revision=self.observed_state_revision,
            logical_tick=self.logical_tick,
        )


def _coerce_event(value: Any, label: str) -> ProtocolEvent:
    if isinstance(value, ProtocolEvent):
        try:
            validate_protocol_event(value)
        except SchemaError as exc:
            raise ProtocolEventFrameworkAuthorityError(f"invalid {label}: {exc}") from exc
        return value
    if not isinstance(value, Mapping):
        raise ProtocolEventFrameworkAuthorityError(f"{label} must be an object")
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return protocol_event_from_json(payload)
    except (TypeError, ValueError, SchemaError) as exc:
        raise ProtocolEventFrameworkAuthorityError(f"invalid {label}: {exc}") from exc


def _coerce_receipt(value: Any, label: str) -> ProtocolEventReceipt:
    if isinstance(value, ProtocolEventReceipt):
        try:
            validate_protocol_event_receipt(value)
        except SchemaError as exc:
            raise ProtocolEventFrameworkAuthorityError(f"invalid {label}: {exc}") from exc
        return value
    if not isinstance(value, Mapping):
        raise ProtocolEventFrameworkAuthorityError(f"{label} must be an object")
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return protocol_event_receipt_from_json(payload)
    except (TypeError, ValueError, SchemaError) as exc:
        raise ProtocolEventFrameworkAuthorityError(f"invalid {label}: {exc}") from exc


def _coerce_order_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolEventFrameworkAuthorityError(
            "current World order protocol state must be an object"
        )
    if set(value) != _ORDER_STATE_FIELDS:
        missing = sorted(_ORDER_STATE_FIELDS - set(value))
        unknown = sorted(set(value) - _ORDER_STATE_FIELDS)
        raise ProtocolEventFrameworkAuthorityError(
            "current World order protocol state has invalid fields: "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    row = dict(value)
    for field in ("order_id", "buyer_id", "merchant_id", "state"):
        row[field] = _framework_text(row[field], field)
    row["state_revision"] = _framework_nonnegative_int(row["state_revision"], "state_revision")
    row["logical_time"] = _framework_nonnegative_int(row["logical_time"], "logical_time")
    row["operation_reference_digest"] = _framework_digest(
        row["operation_reference_digest"], "operation_reference_digest"
    )
    if row["buyer_id"] == row["merchant_id"]:
        raise ProtocolEventFrameworkAuthorityError(
            "current World order buyer and merchant must be distinct"
        )
    return row


def _require_commercial_actor(actor_id: Any) -> None:
    if not isinstance(actor_id, str):
        raise ProtocolEventFrameworkAuthorityError(
            "protocol-event Agent actor must be a full buyer or merchant identity"
        )
    parts = actor_id.split(":")
    if len(parts) < 2 or not all(parts) or parts[0] not in {"buyer", "merchant"}:
        raise ProtocolEventFrameworkAuthorityError(
            "protocol-event Agent actor must be a full buyer or merchant identity"
        )


def _framework_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolEventFrameworkAuthorityError(f"{label} must be non-empty text")
    return value


def _framework_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolEventFrameworkAuthorityError(f"{label} must be a nonnegative integer")
    return value


def _framework_digest(value: Any, label: str) -> str:
    digest = _framework_text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProtocolEventFrameworkAuthorityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_choice_keys(arguments: Mapping[str, Any], *, required: set[str]) -> None:
    if any(not isinstance(key, str) for key in arguments):
        raise ProtocolEventModelChoiceError("protocol-event tool argument names must be strings")
    actual = set(arguments)
    missing = required - actual
    unknown = actual - required
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ProtocolEventModelChoiceError(
            "invalid protocol-event tool arguments: " + "; ".join(details)
        )


def _choice_reason(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolEventModelChoiceError("reason must be non-empty text")
    reason = value.strip()
    if len(reason) > _MAX_REASON_LENGTH:
        raise ProtocolEventModelChoiceError(
            f"reason exceeds the {_MAX_REASON_LENGTH}-character limit"
        )
    return reason


def _decision_spec(operation: str, description: str) -> BusinessIntentSpec:
    return BusinessIntentSpec(
        intent=operation,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_REASON_LENGTH,
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
        category="act",
        source_name=operation,
    )


__all__ = [
    "CompiledProtocolEventAction",
    "ProtocolEventFrameworkAuthorityError",
    "ProtocolEventModelChoiceError",
    "ProtocolEventTurnAuthority",
    "ProtocolEventTurnError",
]
