"""Platform-mediated negotiation with deterministic lineage and privacy checks.

The policy is deliberately not an ``Agent``.  Buyer and Merchant actors submit
typed offer actions to ``platform:negotiation``.  The Platform validates actor
identity, listing ownership, offer lineage, round order, deadline, and the
privacy boundary before returning a transparent relay to the counterparty.

The relay is authored by ``platform:negotiation``.  The submitting actor is
bound inside server-authored ``platform_mediation`` metadata and remains
available in the audited request.  A Platform service must never impersonate
the original buyer or merchant on the wire merely to preserve a convenient
peer-to-peer shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Mapping

from protocol.envelope import Envelope
from protocol.errors import SchemaError
from protocol.negotiation_state import (
    ACCEPT_OFFER,
    COUNTER_OFFER,
    NEGOTIATION_ACTIONS,
    PROPOSE_OFFER,
    NegotiationStateError,
)
from protocol.negotiation_turn_projection import (
    build_negotiation_turn_projection,
    negotiation_turn_projection_to_dict,
)
from runtime.errors import PrivateUtilityLeak
from runtime.privacy import SecretRegistry, find_leak
from world.errors import IdempotencyConflict, WriteNotAuthorized
from world.negotiations import negotiation_status_for_action

if TYPE_CHECKING:
    from agents.world_client import WorldClient


class NegotiationRejected(SchemaError):
    """A sanitized, machine-classified Platform negotiation rejection."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class NegotiationPolicy:
    """Stateless mediator over authoritative World negotiation state."""

    def __init__(
        self,
        *,
        world_client: WorldClient,
        secrets: SecretRegistry | None = None,
        participant_ids: frozenset[str] | None = None,
        max_rounds: int = 8,
        deadline_ticks: int = 32,
    ) -> None:
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds <= 0:
            raise ValueError("negotiation max_rounds must be a positive integer")
        if (
            isinstance(deadline_ticks, bool)
            or not isinstance(deadline_ticks, int)
            or deadline_ticks <= 0
        ):
            raise ValueError("negotiation deadline_ticks must be a positive integer")
        self._world = world_client
        self._secrets = secrets or SecretRegistry()
        self._participants = participant_ids
        self._max_rounds = max_rounds
        self._deadline_ticks = deadline_ticks

    def mediate(self, env: Envelope) -> Envelope:
        """Validate one submission and return its only authorized relay."""

        kind = str(env.action.get("kind", ""))
        if kind not in NEGOTIATION_ACTIONS:
            raise NegotiationRejected("unsupported_negotiation_action")
        if env.to != "platform:negotiation":
            raise NegotiationRejected("wrong_negotiation_endpoint")
        payload = _payload(env)
        negotiation_id = _required_text(payload, "negotiation_id")
        offer_id = _required_text(payload, "offer_id")
        sku_id = _required_text(payload, "sku_id")
        counterparty_id = _required_text(payload, "counterparty_id")
        _require_opposite_sides(env.from_, counterparty_id)
        self._require_participants(env.from_, counterparty_id)

        _reject_authority_fields(payload)
        compact: dict[str, Any] = {
            "action_kind": kind,
            "negotiation_id": negotiation_id,
            "offer_id": offer_id,
            "sku_id": sku_id,
            "counterparty_id": counterparty_id,
        }
        if kind in {PROPOSE_OFFER, COUNTER_OFFER, ACCEPT_OFFER}:
            compact["unit_price"] = _required_price(payload)
        elif "unit_price" in payload:
            compact["unit_price"] = _required_price(payload)
        if "round_no" in payload:
            claimed = payload["round_no"]
            if isinstance(claimed, bool) or not isinstance(claimed, int) or claimed <= 0:
                raise NegotiationRejected("round_mismatch")
            compact["round_no"] = claimed
        if kind == PROPOSE_OFFER and "qty" in payload:
            qty = payload["qty"]
            if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
                raise NegotiationRejected("invalid_qty")
            compact["qty"] = qty

        if kind != PROPOSE_OFFER:
            persisted = self._world.read_negotiation_thread(
                negotiation_id, caller="platform:negotiation"
            )
            if persisted is not None:
                if offer_id != persisted.offer_id:
                    raise NegotiationRejected("offer_lineage_mismatch")
                if sku_id != persisted.sku_id:
                    raise NegotiationRejected("listing_lineage_mismatch")
                if env.from_ not in {
                    persisted.buyer_id,
                    persisted.merchant_id,
                }:
                    raise NegotiationRejected("participant_lineage_mismatch")
                expected_peer = (
                    persisted.merchant_id
                    if env.from_ == persisted.buyer_id
                    else persisted.buyer_id
                )
                if counterparty_id != expected_peer:
                    raise NegotiationRejected("counterparty_mismatch")

        # Check the public relay before any authoritative write.  A private
        # budget/floor probe therefore cannot leave a persistent event behind.
        privacy_probe = Envelope(
            msg_id=f"platform-negotiation:{env.msg_id}",
            ts=env.ts,
            from_=env.from_,
            to=counterparty_id,
            in_reply_to=env.msg_id,
            idempotency_key=f"platform-negotiation:{env.idempotency_key}",
            action={"kind": kind, "payload": dict(payload)},
        )
        finding = find_leak(privacy_probe, self._secrets)
        if finding is not None:
            raise PrivateUtilityLeak(
                "platform negotiation relay would disclose private utility",
                finding=finding,
            )

        try:
            event = self._world.apply_negotiation_intent(
                compact,
                original_actor=env.from_,
                idempotency_key=env.idempotency_key,
                max_rounds=self._max_rounds,
                deadline_ticks=self._deadline_ticks,
            )
        except (NegotiationStateError, WriteNotAuthorized, IdempotencyConflict) as exc:
            raise NegotiationRejected(_reason_code(exc)) from exc

        mediation = {
            "mediated_by": "platform:negotiation",
            "submission_msg_id": env.msg_id,
            "submitted_by": env.from_,
            "recipient_id": event.counterparty_id,
            "negotiation_id": event.negotiation_id,
            "canonical_round": event.round_no,
            "status": negotiation_status_for_action(event.action_kind),
            "listing_digest": event.listing_digest,
        }
        thread = self._world.read_negotiation_thread(
            event.negotiation_id,
            caller="platform:negotiation",
        )
        if thread is None:
            raise RuntimeError(
                "World committed a negotiation event without its current thread"
            )
        relay_payload = dict(payload)
        relay_payload.update(
            {
                "negotiation_id": event.negotiation_id,
                "offer_id": event.offer_id,
                "sku_id": event.sku_id,
                "currency": event.currency,
                # The World operation returns the immutable thread binding in
                # this event. Reuse it directly so a successful commit is not
                # followed by a second read that could fail before relay.
                "qty": event.qty,
                "counterparty_id": event.counterparty_id,
                "round_no": event.round_no,
                "platform_mediation": mediation,
                # This is a server-authored, privacy-safe projection used by
                # the local Agent compiler.  The provider never echoes it.
                # Platform and World still revalidate the eventual action.
                "world_thread_projection": negotiation_turn_projection_to_dict(
                    build_negotiation_turn_projection(
                        thread,
                        # Price already appears at the established public
                        # payload path for priced actions.  Do not duplicate it
                        # under a new nested path that could bypass the privacy
                        # registry's field-specific policy.
                        disclose_price=False,
                    )
                ),
            }
        )
        if event.action_kind in {PROPOSE_OFFER, COUNTER_OFFER, ACCEPT_OFFER}:
            relay_payload["unit_price"] = event.unit_price
        else:
            # World binds a terminal rejection/withdrawal to the current offer
            # price for deterministic replay, but that authoritative lineage
            # field is not part of the public terminal action.  In particular,
            # echoing it can disclose an exact buyer budget or merchant floor
            # even when the actor correctly submitted a price-free rejection.
            relay_payload.pop("unit_price", None)
        return Envelope(
            msg_id=f"platform-negotiation:{env.msg_id}",
            ts=env.ts,
            from_="platform:negotiation",
            to=event.counterparty_id,
            in_reply_to=env.msg_id,
            idempotency_key=f"platform-negotiation:{env.idempotency_key}",
            action={"kind": event.action_kind, "payload": relay_payload},
        )

    def _require_participants(self, sender_id: str, counterparty_id: str) -> None:
        if self._participants is None:
            return
        if sender_id not in self._participants or counterparty_id not in self._participants:
            raise NegotiationRejected("unknown_participant")

def mediation_metadata(result: Any) -> dict[str, Any] | None:
    """Extract the public, server-authored metadata from a mediated relay."""

    values = result if isinstance(result, list) else [result]
    for value in values:
        action = getattr(value, "action", None)
        payload = action.get("payload") if isinstance(action, dict) else None
        metadata = payload.get("platform_mediation") if isinstance(payload, dict) else None
        if isinstance(metadata, dict):
            return _jsonable(metadata)
    return None


def _payload(env: Envelope) -> dict[str, Any]:
    value = env.action.get("payload")
    if not isinstance(value, Mapping):
        raise NegotiationRejected("negotiation_payload_not_object")
    return dict(value)


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise NegotiationRejected(f"missing_{key}")
    return value.strip()


def _required_price(payload: Mapping[str, Any]) -> int:
    value = payload.get("unit_price")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NegotiationRejected("invalid_unit_price")
    return value


def _require_opposite_sides(sender_id: str, counterparty_id: str) -> None:
    roles = {_role(sender_id), _role(counterparty_id)}
    if roles != {"buyer", "merchant"} or sender_id == counterparty_id:
        raise NegotiationRejected("invalid_counterparty_roles")


def _role(actor_id: str) -> str:
    return actor_id.split(":", 1)[0]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    )


_AUTHORITY_FIELDS = frozenset(
    {
        "agreement",
        "buyer_id",
        "event_digest",
        "event_id",
        "expires_at_tick",
        "head_event_digest",
        "listing_digest",
        "listing_revision",
        "max_rounds",
        "merchant_id",
        "opened_at_tick",
        "platform_mediation",
        "previous_digest",
        "sequence_no",
        "server_tick",
        "status",
        "thread_digest",
    }
)


def _reject_authority_fields(payload: Mapping[str, Any]) -> None:
    tampered = sorted(_AUTHORITY_FIELDS.intersection(payload))
    if tampered:
        raise NegotiationRejected("actor_supplied_authority_field")


def _reason_code(exc: BaseException) -> str:
    message = str(exc).casefold()
    mappings = (
        ("listing does not exist", "unknown_listing"),
        ("does not own the listing", "listing_merchant_mismatch"),
        ("already exists", "negotiation_already_exists"),
        ("does not exist", "unknown_negotiation"),
        ("terminal", "negotiation_terminal"),
        ("deadline has expired", "negotiation_expired"),
        ("same side cannot act consecutively", "same_side_double_action"),
        ("counterparty", "counterparty_mismatch"),
        ("acceptance price mismatch", "accept_price_mismatch"),
        ("round_no exceeds max_rounds", "round_limit_exceeded"),
        ("round limit exceeded", "round_limit_exceeded"),
        ("round mismatch", "round_mismatch"),
        ("lineage mismatch", "offer_lineage_mismatch"),
        ("not a negotiation participant", "actor_not_in_negotiation"),
        ("idempotency key", "idempotency_conflict"),
    )
    for fragment, code in mappings:
        if fragment in message:
            return code
    return "negotiation_rejected"


__all__ = [
    "NEGOTIATION_ACTIONS",
    "NegotiationPolicy",
    "NegotiationRejected",
    "mediation_metadata",
]
