"""Agent-side authority for high-level, Platform-mediated negotiation tools.

The language model chooses only commercial values.  It never authors actor
identity, peer identity, negotiation identity, offer lineage, SKU, currency,
or round numbers.  Those fields are derived from one authenticated inbound
envelope and, for an existing thread, the current authoritative World state.

This module is deliberately independent of benchmark cases and scenario
oracles.  It is also independent of model transport and the Agent-private intent
registry, so Buyer and Merchant agents can share the same validation and
compilation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

from agents.business_decision import BusinessIntentSpec
from agents.decision_errors import (
    ModelBusinessDecisionError,
    PlatformContractError,
    SemanticBoundaryError,
)
from protocol.envelope import Envelope
from protocol.matching import (
    MatchValidationError,
    OfferSnapshot,
    SearchSession,
    coerce_search_session,
)
from protocol.negotiation_state import (
    ACCEPT_OFFER,
    COUNTER_OFFER,
    PROPOSE_OFFER,
    NEGOTIATION_ACTIONS,
    NegotiationThread,
    coerce_negotiation_thread,
)
from protocol.negotiation_turn_projection import (
    NEGOTIATION_TURN_PROJECTION_SCHEMA,
    NegotiationTurnProjection,
    coerce_negotiation_turn_projection,
)
from protocol.errors import SchemaError
from world.negotiations import negotiation_status_for_action


_RANK_SCHEMA = "cwe.platform-rank-offers.v2"
_NEGOTIATION_ENDPOINT = "platform:negotiation"
_RANK_ENDPOINT = "platform:aggregator"
_PRICED_ACTIONS = frozenset({PROPOSE_OFFER, COUNTER_OFFER, ACCEPT_OFFER})
_ACTIVE_INBOUND_ACTIONS = frozenset({PROPOSE_OFFER, COUNTER_OFFER})
_DIGEST_LENGTH = 64
_MAX_NOTE_LENGTH = 500

_PROPOSE_OPERATION = "propose_offer"
_COUNTER_OPERATION = "counter_offer"
_ACCEPT_OPERATION = "accept_negotiated_offer"
_REJECT_OPERATION = "reject_offer"


class NegotiationTurnError(SemanticBoundaryError):
    """Base error raised at the Agent negotiation compiler boundary."""


class NegotiationFrameworkAuthorityError(
    PlatformContractError,
    NegotiationTurnError,
):
    """Authenticated Platform or World authority is absent or inconsistent."""


class NegotiationModelChoiceError(ModelBusinessDecisionError, NegotiationTurnError):
    """A model selected a business intent or value outside its contract."""


@dataclass(frozen=True, slots=True)
class NegotiationOfferAuthority:
    """One immutable ranked offer available for a new negotiation."""

    offer_id: str
    session_id: str
    session_digest: str
    mandate_id: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    currency: str
    available_qty: int
    listed_unit_price: int
    offer_digest: str
    catalog_revision: int
    inventory_revision: int


@dataclass(frozen=True, slots=True)
class CompiledNegotiationAction:
    """Business operation and authority-bound payload ready for Agent routing."""

    operation: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NegotiationTurnAuthority:
    """Validated authority for one ranked or mediated Agent turn."""

    source: Literal["rank_offers", "mediated_turn"]
    actor_id: str
    inbound_msg_id: str
    offers: tuple[NegotiationOfferAuthority, ...] = ()
    negotiation_id: str | None = None
    offer_id: str | None = None
    peer_id: str | None = None
    sku_id: str | None = None
    currency: str | None = None
    qty: int | None = None
    current_unit_price: int | None = None
    canonical_round: int | None = None
    max_rounds: int | None = None
    inbound_action_kind: str | None = None
    thread_status: str | None = None

    @classmethod
    def from_inbound(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        current_tick: int | None = None,
        current_thread: NegotiationThread | Mapping[str, Any] | None = None,
    ) -> "NegotiationTurnAuthority":
        """Build authority from the authenticated current inbound envelope.

        A ranked response requires ``current_tick`` so an expired search
        session cannot become a model choice.  A mediated response requires
        ``current_thread`` so a delayed or cross-thread relay cannot be used to
        compile a new World mutation.
        """

        _require_actor_id(actor_id)
        if inbound.to != actor_id:
            raise NegotiationFrameworkAuthorityError(
                "authenticated inbound recipient does not match the Agent actor"
            )
        kind = _action_kind(inbound)
        if kind == "platform.rank_offers":
            return cls._from_ranked_response(
                inbound,
                actor_id=actor_id,
                current_tick=current_tick,
            )
        if kind in NEGOTIATION_ACTIONS:
            return cls._from_mediated_response(
                inbound,
                actor_id=actor_id,
                current_thread=current_thread,
            )
        raise NegotiationFrameworkAuthorityError(
            "current inbound is not a negotiation authority surface"
        )

    @classmethod
    def _from_ranked_response(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        current_tick: int | None,
    ) -> "NegotiationTurnAuthority":
        if inbound.from_ != _RANK_ENDPOINT:
            raise NegotiationFrameworkAuthorityError(
                "ranked negotiation authority has the wrong Platform sender"
            )
        if _role(actor_id) != "buyer":
            raise NegotiationFrameworkAuthorityError(
                "ranked offer authority may be issued only to a buyer actor"
            )
        now = _framework_nonnegative_int(current_tick, "current World tick")
        payload = _payload(inbound)
        required_top_level = {
            "schema_version",
            "session_id",
            "session_digest",
            "mandate_id",
            "issued_at_tick",
            "expires_at_tick",
            "search_session",
            "candidates",
        }
        missing = required_top_level - set(payload)
        if missing:
            raise NegotiationFrameworkAuthorityError(
                "ranked response is missing authority fields: " + ", ".join(sorted(missing))
            )
        if payload.get("schema_version") != _RANK_SCHEMA:
            raise NegotiationFrameworkAuthorityError(
                "ranked response schema version is not supported"
            )
        try:
            session = coerce_search_session(payload["search_session"])
        except (SchemaError, MatchValidationError) as exc:
            raise NegotiationFrameworkAuthorityError(
                f"ranked response has an invalid search session: {exc}"
            ) from exc
        _same_framework("session_id", payload.get("session_id"), session.session_id)
        _same_framework("session_digest", payload.get("session_digest"), session.session_digest)
        _same_framework("mandate_id", payload.get("mandate_id"), session.mandate_id)
        _same_framework("issued_at_tick", payload.get("issued_at_tick"), session.issued_at_tick)
        _same_framework("expires_at_tick", payload.get("expires_at_tick"), session.expires_at_tick)
        _same_framework("buyer_id", session.buyer_id, actor_id)
        if now < session.issued_at_tick or now >= session.expires_at_tick:
            raise NegotiationFrameworkAuthorityError(
                "ranked negotiation authority is not fresh at the current World tick"
            )

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise NegotiationFrameworkAuthorityError("ranked response candidates must be an array")
        if len(raw_candidates) != len(session.offers):
            raise NegotiationFrameworkAuthorityError(
                "ranked candidates do not exactly project the search session"
            )
        offers_by_id = {offer.offer_id: offer for offer in session.offers}
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        authorities: list[NegotiationOfferAuthority] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                raise NegotiationFrameworkAuthorityError(
                    "ranked response candidate must be an object"
                )
            offer_id = _framework_text(raw_candidate.get("offer_id"), "offer_id")
            offer_digest = _framework_digest(raw_candidate.get("offer_digest"), "offer_digest")
            if offer_id in seen_ids or offer_digest in seen_digests:
                raise NegotiationFrameworkAuthorityError(
                    "ranked response contains a duplicate offer authority"
                )
            seen_ids.add(offer_id)
            seen_digests.add(offer_digest)
            offer = offers_by_id.get(offer_id)
            if offer is None:
                raise NegotiationFrameworkAuthorityError(
                    "ranked candidate is not a member of the search session"
                )
            _validate_candidate_projection(raw_candidate, offer)
            if not _commercial_actor_id(
                offer.buyer_id, expected_role="buyer"
            ) or not _commercial_actor_id(offer.merchant_id, expected_role="merchant"):
                raise NegotiationFrameworkAuthorityError(
                    "ranked offer has invalid buyer or merchant actor identity"
                )
            if offer.buyer_id == offer.merchant_id:
                raise NegotiationFrameworkAuthorityError(
                    "ranked offer cannot bind one actor to both sides"
                )
            authorities.append(_offer_authority(session, offer))
        if set(offers_by_id) != seen_ids:
            raise NegotiationFrameworkAuthorityError(
                "ranked candidates omit a search-session offer"
            )
        return cls(
            source="rank_offers",
            actor_id=actor_id,
            inbound_msg_id=inbound.msg_id,
            offers=tuple(authorities),
        )

    @classmethod
    def _from_mediated_response(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        current_thread: NegotiationThread | Mapping[str, Any] | None,
    ) -> "NegotiationTurnAuthority":
        if inbound.from_ != _NEGOTIATION_ENDPOINT:
            raise NegotiationFrameworkAuthorityError(
                "negotiation relay has the wrong Platform sender"
            )
        if current_thread is None:
            raise NegotiationFrameworkAuthorityError(
                "mediated negotiation authority requires current World thread state"
            )
        try:
            thread = (
                coerce_negotiation_turn_projection(current_thread)
                if isinstance(current_thread, Mapping)
                and current_thread.get("schema_id") == NEGOTIATION_TURN_PROJECTION_SCHEMA
                else coerce_negotiation_thread(current_thread)
            )
        except SchemaError as exc:
            raise NegotiationFrameworkAuthorityError(
                f"current World negotiation thread is invalid: {exc}"
            ) from exc
        kind = _action_kind(inbound)
        payload = _payload(inbound)
        metadata = payload.get("platform_mediation")
        if not isinstance(metadata, Mapping):
            raise NegotiationFrameworkAuthorityError(
                "negotiation relay lacks Platform mediation metadata"
            )
        required_metadata = {
            "mediated_by",
            "submission_msg_id",
            "submitted_by",
            "recipient_id",
            "negotiation_id",
            "canonical_round",
            "status",
            "listing_digest",
        }
        if set(metadata) != required_metadata:
            raise NegotiationFrameworkAuthorityError(
                "negotiation mediation metadata has missing or unknown fields"
            )
        if metadata.get("mediated_by") != _NEGOTIATION_ENDPOINT:
            raise NegotiationFrameworkAuthorityError("invalid negotiation mediator")
        submission_msg_id = _framework_text(metadata.get("submission_msg_id"), "submission_msg_id")
        if inbound.in_reply_to != submission_msg_id:
            raise NegotiationFrameworkAuthorityError(
                "negotiation relay does not reply to its bound submission"
            )
        submitted_by = _framework_text(metadata.get("submitted_by"), "submitted_by")
        recipient_id = _framework_text(metadata.get("recipient_id"), "recipient_id")
        _same_framework("recipient_id", recipient_id, actor_id)
        if not _commercial_actor_id(submitted_by) or {
            _role(submitted_by),
            _role(actor_id),
        } != {"buyer", "merchant"}:
            raise NegotiationFrameworkAuthorityError(
                "negotiation relay does not bind opposite commercial roles"
            )
        if submitted_by == actor_id:
            raise NegotiationFrameworkAuthorityError(
                "negotiation relay cannot submit and receive as the same actor"
            )

        negotiation_id = _framework_text(payload.get("negotiation_id"), "negotiation_id")
        offer_id = _framework_text(payload.get("offer_id"), "offer_id")
        sku_id = _framework_text(payload.get("sku_id"), "sku_id")
        currency = _framework_currency(payload.get("currency"))
        qty = _framework_positive_int(payload.get("qty"), "qty")
        round_no = _framework_positive_int(payload.get("round_no"), "round_no")
        counterparty_id = _framework_text(payload.get("counterparty_id"), "counterparty_id")
        # A transparent relay retains the World event's counterparty, which is
        # the authenticated recipient.  The compiler uses submitted_by as the
        # peer for the recipient's next action.
        _same_framework("counterparty_id", counterparty_id, actor_id)
        _same_framework("metadata negotiation_id", metadata.get("negotiation_id"), negotiation_id)
        _same_framework("canonical_round", metadata.get("canonical_round"), round_no)
        listing_digest = _framework_digest(metadata.get("listing_digest"), "listing_digest")
        expected_status = negotiation_status_for_action(kind)
        _same_framework("mediation status", metadata.get("status"), expected_status)

        unit_price: int | None = None
        if kind in _PRICED_ACTIONS:
            unit_price = _framework_positive_int(payload.get("unit_price"), "unit_price")
        elif "unit_price" in payload:
            raise NegotiationFrameworkAuthorityError(
                "terminal rejection or withdrawal relay must not expose unit_price"
            )

        _validate_relay_against_current_thread(
            thread=thread,
            kind=kind,
            actor_id=actor_id,
            submitted_by=submitted_by,
            negotiation_id=negotiation_id,
            offer_id=offer_id,
            sku_id=sku_id,
            currency=currency,
            qty=qty,
            round_no=round_no,
            unit_price=unit_price,
            listing_digest=listing_digest,
        )
        return cls(
            source="mediated_turn",
            actor_id=actor_id,
            inbound_msg_id=inbound.msg_id,
            negotiation_id=negotiation_id,
            offer_id=offer_id,
            peer_id=submitted_by,
            sku_id=sku_id,
            currency=currency,
            qty=qty,
            current_unit_price=(
                unit_price
                if isinstance(thread, NegotiationTurnProjection)
                and thread.current_unit_price is None
                else thread.current_unit_price
            ),
            canonical_round=round_no,
            max_rounds=thread.max_rounds,
            inbound_action_kind=kind,
            thread_status=thread.status,
        )

    def _counter_offer_available(self) -> bool:
        """Return whether a counter can stay within authenticated round authority."""

        return bool(
            self.source == "mediated_turn"
            and isinstance(self.canonical_round, int)
            and not isinstance(self.canonical_round, bool)
            and isinstance(self.max_rounds, int)
            and not isinstance(self.max_rounds, bool)
            and 0 < self.canonical_round < self.max_rounds
        )

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        """Return provider-neutral business choices legal for this turn."""

        if self.source == "rank_offers":
            offer_ids = [offer.offer_id for offer in self.offers]
            if not offer_ids:
                return ()
            return (
                _decision_spec(
                    _PROPOSE_OPERATION,
                    "Open a mediated negotiation for one ranked offer.",
                    properties={
                        "offer_id": {"type": "string", "enum": offer_ids},
                        "unit_price": {"type": "integer", "minimum": 1},
                        "qty": {"type": "integer", "minimum": 1},
                        "note": {"type": "string", "minLength": 1, "maxLength": _MAX_NOTE_LENGTH},
                    },
                    required=("offer_id", "unit_price", "qty"),
                ),
            )
        if (
            self.thread_status != "active"
            or self.inbound_action_kind not in _ACTIVE_INBOUND_ACTIONS
        ):
            return ()
        qty = cast(int, self.qty)
        counter = (
            (
                _decision_spec(
                    _COUNTER_OPERATION,
                    "Counter the current authoritative offer.",
                    properties={
                        "unit_price": {"type": "integer", "minimum": 1},
                        # Quantity is immutable after the proposal in the current
                        # World state machine. The optional value may affirm it,
                        # but cannot silently rewrite the thread binding.
                        "qty": {"type": "integer", "const": qty},
                        "note": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_NOTE_LENGTH,
                        },
                    },
                    required=("unit_price",),
                ),
            )
            if self._counter_offer_available()
            else ()
        )
        return (
            *counter,
            _decision_spec(
                _ACCEPT_OPERATION,
                "Accept the current authoritative price and quantity.",
                properties={},
                required=(),
            ),
            _decision_spec(
                _REJECT_OPERATION,
                "Reject the current offer without disclosing private utility.",
                properties={
                    "reason": {"type": "string", "minLength": 1, "maxLength": _MAX_NOTE_LENGTH}
                },
                required=("reason",),
            ),
        )

    def compile(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CompiledNegotiationAction:
        """Compile a high-level model choice to authority-bound business data."""

        if not isinstance(arguments, Mapping):
            raise NegotiationModelChoiceError("negotiation tool arguments must be an object")
        if self.source == "rank_offers":
            return self._compile_start(operation, arguments)
        return self._compile_response(operation, arguments)

    def _compile_start(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CompiledNegotiationAction:
        if operation != _PROPOSE_OPERATION:
            raise NegotiationModelChoiceError(
                "model selected an intent outside the ranked negotiation turn"
            )
        _require_choice_keys(
            arguments,
            required={"offer_id", "unit_price", "qty"},
            optional={"note"},
        )
        offer_id = _choice_text(arguments.get("offer_id"), "offer_id")
        matches = [offer for offer in self.offers if offer.offer_id == offer_id]
        if len(matches) != 1:
            raise NegotiationModelChoiceError(
                "selected offer_id is outside the current ranked offer enum"
            )
        offer = matches[0]
        unit_price = _choice_positive_int(arguments.get("unit_price"), "unit_price")
        qty = _choice_positive_int(arguments.get("qty"), "qty")
        if qty > offer.available_qty:
            raise NegotiationModelChoiceError(
                "proposed quantity exceeds the selected ranked offer authority"
            )
        payload: dict[str, Any] = {
            "negotiation_id": f"neg:{offer.mandate_id}:{offer.offer_id}",
            "offer_id": offer.offer_id,
            "sku_id": offer.sku_id,
            "counterparty_id": offer.merchant_id,
            "unit_price": unit_price,
            "round_no": 1,
            "qty": qty,
        }
        note = _optional_choice_note(arguments)
        if note is not None:
            payload["reason"] = note
        return CompiledNegotiationAction(
            operation=_PROPOSE_OPERATION,
            payload=payload,
        )

    def _compile_response(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CompiledNegotiationAction:
        if (
            self.thread_status != "active"
            or self.inbound_action_kind not in _ACTIVE_INBOUND_ACTIONS
        ):
            raise NegotiationFrameworkAuthorityError(
                "current World negotiation thread has no legal response turn"
            )
        if operation not in {
            _COUNTER_OPERATION,
            _ACCEPT_OPERATION,
            _REJECT_OPERATION,
        }:
            raise NegotiationModelChoiceError(
                "model selected an intent outside the current negotiation turn"
            )
        negotiation_id = cast(str, self.negotiation_id)
        offer_id = cast(str, self.offer_id)
        sku_id = cast(str, self.sku_id)
        peer_id = cast(str, self.peer_id)
        qty = cast(int, self.qty)
        price = cast(int, self.current_unit_price)
        round_no = cast(int, self.canonical_round)
        payload: dict[str, Any] = {
            "negotiation_id": negotiation_id,
            "offer_id": offer_id,
            "sku_id": sku_id,
            "counterparty_id": peer_id,
        }
        if operation == _COUNTER_OPERATION:
            if not self._counter_offer_available():
                raise NegotiationModelChoiceError(
                    "model selected counter_offer at the final negotiation round"
                )
            _require_choice_keys(
                arguments,
                required={"unit_price"},
                optional={"qty", "note"},
            )
            payload["unit_price"] = _choice_positive_int(arguments.get("unit_price"), "unit_price")
            if "qty" in arguments:
                selected_qty = _choice_positive_int(arguments.get("qty"), "qty")
                if selected_qty != qty:
                    raise NegotiationModelChoiceError(
                        "counter quantity cannot rewrite the World thread binding"
                    )
            payload["round_no"] = round_no + 1
            note = _optional_choice_note(arguments)
            if note is not None:
                payload["reason"] = note
            compiled_operation = _COUNTER_OPERATION
        elif operation == _ACCEPT_OPERATION:
            _require_choice_keys(arguments, required=set(), optional=set())
            payload["unit_price"] = price
            payload["round_no"] = round_no
            compiled_operation = _ACCEPT_OPERATION
        else:
            _require_choice_keys(arguments, required={"reason"}, optional=set())
            payload["round_no"] = round_no
            payload["reason"] = _choice_note(arguments.get("reason"), "reason")
            compiled_operation = _REJECT_OPERATION
        return CompiledNegotiationAction(
            operation=compiled_operation,
            payload=payload,
        )


def _validate_candidate_projection(candidate: Mapping[str, Any], offer: OfferSnapshot) -> None:
    expected = {
        "schema_version": "cwe.offer-snapshot.v1",
        "offer_id": offer.offer_id,
        "session_id": offer.session_id,
        "buyer_id": offer.buyer_id,
        "mandate_id": offer.mandate_id,
        "merchant_id": offer.merchant_id,
        "sku_id": offer.sku_id,
        "unit_price_cents": offer.unit_price_cents,
        "currency": offer.currency,
        "qty": offer.qty,
        "catalog_revision": offer.catalog_revision,
        "inventory_revision": offer.inventory_revision,
        "issued_at_tick": offer.issued_at_tick,
        "expires_at_tick": offer.expires_at_tick,
        "offer_digest": offer.offer_digest,
    }
    for field, expected_value in expected.items():
        _same_framework(field, candidate.get(field), expected_value)
    if "unit_price" in candidate:
        _same_framework("unit_price alias", candidate.get("unit_price"), offer.unit_price_cents)
    if "expires_at" in candidate:
        _same_framework(
            "expires_at alias",
            candidate.get("expires_at"),
            f"world-tick:{offer.expires_at_tick}",
        )


def _offer_authority(session: SearchSession, offer: OfferSnapshot) -> NegotiationOfferAuthority:
    return NegotiationOfferAuthority(
        offer_id=offer.offer_id,
        session_id=offer.session_id,
        session_digest=session.session_digest,
        mandate_id=offer.mandate_id,
        buyer_id=offer.buyer_id,
        merchant_id=offer.merchant_id,
        sku_id=offer.sku_id,
        currency=offer.currency,
        available_qty=offer.qty,
        listed_unit_price=offer.unit_price_cents,
        offer_digest=offer.offer_digest,
        catalog_revision=offer.catalog_revision,
        inventory_revision=offer.inventory_revision,
    )


def _validate_relay_against_current_thread(
    *,
    thread: NegotiationThread | NegotiationTurnProjection,
    kind: str,
    actor_id: str,
    submitted_by: str,
    negotiation_id: str,
    offer_id: str,
    sku_id: str,
    currency: str,
    qty: int,
    round_no: int,
    unit_price: int | None,
    listing_digest: str,
) -> None:
    comparisons = {
        "negotiation_id": (negotiation_id, thread.negotiation_id),
        "offer_id": (offer_id, thread.offer_id),
        "sku_id": (sku_id, thread.sku_id),
        "currency": (currency, thread.currency),
        "qty": (qty, thread.qty),
        "round_no": (round_no, thread.round_no),
        "listing_digest": (listing_digest, thread.listing_digest),
        "submitted_by": (submitted_by, thread.last_actor_id),
        "recipient_id": (actor_id, thread.last_counterparty_id),
        "status": (negotiation_status_for_action(kind), thread.status),
    }
    mismatches = [
        field
        for field, (actual, expected) in comparisons.items()
        if actual != expected or type(actual) is not type(expected)
    ]
    if (
        unit_price is not None
        and thread.current_unit_price is not None
        and unit_price != thread.current_unit_price
    ):
        mismatches.append("unit_price")
    participants = {thread.buyer_id, thread.merchant_id}
    if participants != {actor_id, submitted_by}:
        mismatches.append("participants")
    if mismatches:
        raise NegotiationFrameworkAuthorityError(
            "negotiation relay is stale or cross-bound against current World state: "
            + ", ".join(sorted(set(mismatches)))
        )


def _decision_spec(
    operation: str,
    description: str,
    *,
    properties: Mapping[str, Any],
    required: tuple[str, ...],
) -> BusinessIntentSpec:
    return BusinessIntentSpec(
        intent=operation,
        description=description,
        parameters={
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
        category="act",
        source_name=operation,
    )


def _payload(inbound: Envelope) -> dict[str, Any]:
    value = inbound.action.get("payload")
    if not isinstance(value, Mapping):
        raise NegotiationFrameworkAuthorityError("negotiation authority payload must be an object")
    return dict(value)


def _action_kind(inbound: Envelope) -> str:
    value = inbound.action.get("kind")
    if not isinstance(value, str) or not value:
        raise NegotiationFrameworkAuthorityError("negotiation authority action kind is missing")
    return value


def _require_actor_id(actor_id: str) -> None:
    if not _commercial_actor_id(actor_id):
        raise NegotiationFrameworkAuthorityError(
            "negotiation Agent actor must be a full buyer or merchant identity"
        )


def _role(actor_id: str) -> str:
    return actor_id.split(":", 1)[0] if isinstance(actor_id, str) else ""


def _commercial_actor_id(value: Any, *, expected_role: str | None = None) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split(":")
    if len(parts) < 2 or not all(parts):
        return False
    role = parts[0]
    return role in {"buyer", "merchant"} and (expected_role is None or role == expected_role)


def _same_framework(label: str, actual: Any, expected: Any) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise NegotiationFrameworkAuthorityError(f"{label} authority mismatch")


def _framework_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegotiationFrameworkAuthorityError(f"{label} must be non-empty text")
    return value.strip()


def _framework_currency(value: Any) -> str:
    currency = _framework_text(value, "currency")
    if len(currency) != 3 or currency != currency.upper() or not currency.isalpha():
        raise NegotiationFrameworkAuthorityError("currency must be a three-letter uppercase code")
    return currency


def _framework_digest(value: Any, label: str) -> str:
    digest = _framework_text(value, label)
    if len(digest) != _DIGEST_LENGTH or any(c not in "0123456789abcdef" for c in digest):
        raise NegotiationFrameworkAuthorityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _framework_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NegotiationFrameworkAuthorityError(f"{label} must be a nonnegative integer")
    return value


def _framework_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NegotiationFrameworkAuthorityError(f"{label} must be a positive integer")
    return value


def _require_choice_keys(
    arguments: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
) -> None:
    if any(not isinstance(key, str) for key in arguments):
        raise NegotiationModelChoiceError("negotiation tool argument names must be strings")
    actual = set(arguments)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise NegotiationModelChoiceError(
            "invalid negotiation tool arguments: " + "; ".join(details)
        )


def _choice_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegotiationModelChoiceError(f"{label} must be non-empty text")
    return value.strip()


def _choice_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NegotiationModelChoiceError(f"{label} must be a positive integer")
    return value


def _choice_note(value: Any, label: str) -> str:
    note = _choice_text(value, label)
    if len(note) > _MAX_NOTE_LENGTH:
        raise NegotiationModelChoiceError(f"{label} exceeds the {_MAX_NOTE_LENGTH}-character limit")
    return note


def _optional_choice_note(arguments: Mapping[str, Any]) -> str | None:
    if "note" not in arguments:
        return None
    return _choice_note(arguments.get("note"), "note")


__all__ = [
    "CompiledNegotiationAction",
    "NegotiationFrameworkAuthorityError",
    "NegotiationModelChoiceError",
    "NegotiationOfferAuthority",
    "NegotiationTurnAuthority",
    "NegotiationTurnError",
]
