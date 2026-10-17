"""Agent authority for accepting a World-backed Platform ranked offer.

The Platform response is the only source of search-session and offer identity.
The model may select one advertised offer or make a local high-level
disposition.  It never authors session digests, actor identities, listing
identity, price, revisions, or the order identifier used on the wire.

This boundary is intentionally independent of benchmark tasks and scenario
oracles.  It can be shared by every buyer that consumes
``platform.rank_offers``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from agents.business_decision import BusinessIntentSpec
from agents.decision_errors import (
    ModelBusinessDecisionError,
    PlatformContractError,
    SemanticBoundaryError,
)
from protocol.envelope import Envelope
from protocol.errors import SchemaError
from protocol.matching import (
    MatchValidationError,
    OfferSnapshot,
    SearchSession,
    coerce_search_session,
)


_RANK_ENDPOINT = "platform:aggregator"
_RANK_KIND = "platform.rank_offers"
_RANK_SCHEMA = "cwe.platform-rank-offers.v2"
_DIGEST_LENGTH = 64
_MAX_REASON_LENGTH = 500
_ACCEPT_OPERATION = "accept_ranked_offer"
_REJECT_OPERATION = "reject_ranked_offers"
_CONTINUE_OPERATION = "continue_ranked_selection"
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "session_id",
        "session_digest",
        "mandate_id",
        "issued_at_tick",
        "expires_at_tick",
        "search_session",
        "candidates",
        "ranking_context_reference",
        "ranking_context_projection",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "offer_id",
        "session_id",
        "buyer_id",
        "mandate_id",
        "merchant_id",
        "sku_id",
        "unit_price_cents",
        "currency",
        "qty",
        "catalog_revision",
        "inventory_revision",
        "issued_at_tick",
        "expires_at_tick",
        "offer_digest",
        "unit_price",
        "expires_at",
        "fulfillment",
        "claims",
    }
)


class RankedOfferTurnError(SemanticBoundaryError):
    """Base error for the ranked-offer Agent boundary."""


class RankedOfferFrameworkAuthorityError(
    PlatformContractError,
    RankedOfferTurnError,
):
    """The authenticated Platform authority is absent or contradictory."""


class RankedOfferModelChoiceError(ModelBusinessDecisionError, RankedOfferTurnError):
    """The model selected a business intent or argument outside the advertised set."""


@dataclass(frozen=True, slots=True)
class RankedOfferAuthority:
    """One exact World-backed offer that the model may select."""

    offer_id: str
    session_id: str
    session_digest: str
    mandate_id: str
    buyer_id: str
    merchant_id: str
    sku_id: str
    unit_price_cents: int
    currency: str
    qty: int
    catalog_revision: int
    inventory_revision: int
    offer_digest: str
    order_id: str


@dataclass(frozen=True, slots=True)
class CompiledRankedOfferDecision:
    """Authority-bound result of one ranked-offer business decision."""

    disposition: Literal["accept", "reject", "continue"]
    operation: str | None
    payload: Mapping[str, Any] | None
    selected_offer: RankedOfferAuthority | None = None


@dataclass(frozen=True, slots=True)
class RankedOfferTurnAuthority:
    """Validated authority for one current ``platform.rank_offers`` turn."""

    actor_id: str
    inbound_msg_id: str
    search_request_msg_id: str
    issued_at_tick: int
    expires_at_tick: int
    offers: tuple[RankedOfferAuthority, ...]

    @classmethod
    def from_inbound(
        cls,
        inbound: Envelope,
        *,
        actor_id: str,
        current_tick: int,
    ) -> "RankedOfferTurnAuthority":
        """Fail closed before model invocation if rank authority is invalid."""

        if not _actor_id(actor_id, role="buyer"):
            raise RankedOfferFrameworkAuthorityError(
                "ranked-offer authority requires a full buyer actor identity"
            )
        if inbound.from_ != _RANK_ENDPOINT:
            raise RankedOfferFrameworkAuthorityError(
                "ranked-offer authority has the wrong Platform sender"
            )
        if inbound.to != actor_id:
            raise RankedOfferFrameworkAuthorityError(
                "ranked-offer authority is bound to another buyer actor"
            )
        if inbound.action.get("kind") != _RANK_KIND:
            raise RankedOfferFrameworkAuthorityError(
                "current inbound is not a ranked-offer authority surface"
            )
        search_request_msg_id = _framework_text(inbound.in_reply_to, "ranked response in_reply_to")
        now = _framework_nonnegative_int(current_tick, "current World tick")
        payload = _payload(inbound)
        if set(payload) != _TOP_LEVEL_FIELDS:
            _raise_field_difference(
                actual=set(payload),
                expected=set(_TOP_LEVEL_FIELDS),
                label="ranked response",
            )
        if payload["schema_version"] != _RANK_SCHEMA:
            raise RankedOfferFrameworkAuthorityError(
                "ranked response schema version is not supported"
            )
        try:
            session = coerce_search_session(payload["search_session"])
        except (SchemaError, MatchValidationError) as exc:
            raise RankedOfferFrameworkAuthorityError(
                f"ranked response has an invalid World search session: {exc}"
            ) from exc
        _validate_session_binding(
            inbound=inbound,
            payload=payload,
            session=session,
            actor_id=actor_id,
            current_tick=now,
        )
        _validate_ranking_context(payload)
        offers = _validate_candidates(payload["candidates"], session, current_tick=now)
        return cls(
            actor_id=actor_id,
            inbound_msg_id=_framework_text(inbound.msg_id, "ranked response msg_id"),
            search_request_msg_id=search_request_msg_id,
            issued_at_tick=session.issued_at_tick,
            expires_at_tick=session.expires_at_tick,
            offers=offers,
        )

    def decision_schema(self) -> tuple[BusinessIntentSpec, ...]:
        """Expose provider-neutral choices with offer identity as a closed enum."""

        specs: list[BusinessIntentSpec] = []
        offer_ids = [offer.offer_id for offer in self.offers]
        if offer_ids:
            specs.append(
                _decision_spec(
                    _ACCEPT_OPERATION,
                    "Accept one currently ranked marketplace offer.",
                    properties={
                        "offer_id": {"type": "string", "enum": offer_ids},
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_REASON_LENGTH,
                        },
                    },
                    required=("offer_id",),
                )
            )
        specs.extend(
            (
                _decision_spec(
                    _REJECT_OPERATION,
                    "Reject the current ranked set without accepting an offer.",
                    properties={
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_REASON_LENGTH,
                        }
                    },
                    required=("reason",),
                ),
                _decision_spec(
                    _CONTINUE_OPERATION,
                    "Continue local reasoning without committing a commerce action.",
                    properties={},
                    required=(),
                ),
            )
        )
        return tuple(specs)

    def compile(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> CompiledRankedOfferDecision:
        """Compile a model business choice to authority-bound business data."""

        if not isinstance(arguments, Mapping):
            raise RankedOfferModelChoiceError("ranked-offer tool arguments must be an object")
        if operation == _ACCEPT_OPERATION:
            _require_choice_keys(arguments, required={"offer_id"}, optional={"reason"})
            offer_id = _choice_text(arguments.get("offer_id"), "offer_id")
            matches = [offer for offer in self.offers if offer.offer_id == offer_id]
            if len(matches) != 1:
                raise RankedOfferModelChoiceError(
                    "selected offer_id is outside the current grounded offer enum"
                )
            if "reason" in arguments:
                _choice_reason(arguments.get("reason"))
            offer = matches[0]
            return CompiledRankedOfferDecision(
                disposition="accept",
                operation=_ACCEPT_OPERATION,
                payload={
                    "session_id": offer.session_id,
                    "session_digest": offer.session_digest,
                    "offer_id": offer.offer_id,
                    "offer_digest": offer.offer_digest,
                    "mandate_id": offer.mandate_id,
                    "order_id": offer.order_id,
                    "merchant_id": offer.merchant_id,
                    "sku_id": offer.sku_id,
                    "unit_price_cents": offer.unit_price_cents,
                    "currency": offer.currency,
                    "qty": offer.qty,
                    "catalog_revision": offer.catalog_revision,
                    "inventory_revision": offer.inventory_revision,
                },
                selected_offer=offer,
            )
        if operation == _REJECT_OPERATION:
            _require_choice_keys(arguments, required={"reason"}, optional=set())
            _choice_reason(arguments.get("reason"))
            return CompiledRankedOfferDecision("reject", None, None)
        if operation == _CONTINUE_OPERATION:
            _require_choice_keys(arguments, required=set(), optional=set())
            return CompiledRankedOfferDecision("continue", None, None)
        raise RankedOfferModelChoiceError(
            "model selected an intent outside the current ranked-offer turn"
        )


def _validate_session_binding(
    *,
    inbound: Envelope,
    payload: Mapping[str, Any],
    session: SearchSession,
    actor_id: str,
    current_tick: int,
) -> None:
    comparisons = {
        "session_id": (payload["session_id"], session.session_id),
        "session_digest": (payload["session_digest"], session.session_digest),
        "mandate_id": (payload["mandate_id"], session.mandate_id),
        "issued_at_tick": (payload["issued_at_tick"], session.issued_at_tick),
        "expires_at_tick": (payload["expires_at_tick"], session.expires_at_tick),
        "buyer_id": (session.buyer_id, actor_id),
        "response idempotency": (
            inbound.idempotency_key,
            session.search_idempotency_key,
        ),
        "search request authority": (
            session.search_request_id,
            session.search_idempotency_key,
        ),
    }
    mismatches = [
        label
        for label, (actual, expected) in comparisons.items()
        if actual != expected or type(actual) is not type(expected)
    ]
    if mismatches:
        raise RankedOfferFrameworkAuthorityError(
            "ranked response is cross-bound against its World search session: "
            + ", ".join(sorted(mismatches))
        )
    if current_tick < session.issued_at_tick or current_tick >= session.expires_at_tick:
        raise RankedOfferFrameworkAuthorityError(
            "ranked response is expired or not yet valid at the current World tick"
        )


def _validate_candidates(
    value: Any,
    session: SearchSession,
    *,
    current_tick: int,
) -> tuple[RankedOfferAuthority, ...]:
    if not isinstance(value, list):
        raise RankedOfferFrameworkAuthorityError("ranked response candidates must be an array")
    if len(value) != len(session.offers):
        raise RankedOfferFrameworkAuthorityError(
            "ranked candidates do not exactly project the World search session"
        )
    by_id = {offer.offer_id: offer for offer in session.offers}
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    result: list[RankedOfferAuthority] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise RankedOfferFrameworkAuthorityError("ranked candidate must be an object")
        if set(raw) != _CANDIDATE_FIELDS:
            _raise_field_difference(
                actual=set(raw),
                expected=set(_CANDIDATE_FIELDS),
                label="ranked candidate",
            )
        offer_id = _framework_text(raw.get("offer_id"), "offer_id")
        offer_digest = _framework_digest(raw.get("offer_digest"), "offer_digest")
        if offer_id in seen_ids or offer_digest in seen_digests:
            raise RankedOfferFrameworkAuthorityError(
                "ranked response contains duplicate offer authority"
            )
        seen_ids.add(offer_id)
        seen_digests.add(offer_digest)
        offer = by_id.get(offer_id)
        if offer is None:
            raise RankedOfferFrameworkAuthorityError(
                "ranked candidate is not in the World search session"
            )
        if current_tick < offer.issued_at_tick or current_tick >= offer.expires_at_tick:
            raise RankedOfferFrameworkAuthorityError(
                "ranked response contains expired or not-yet-valid offer authority"
            )
        _validate_candidate_projection(raw, offer)
        if not _actor_id(offer.buyer_id, role="buyer") or not _actor_id(
            offer.merchant_id, role="merchant"
        ):
            raise RankedOfferFrameworkAuthorityError(
                "ranked offer has an invalid commercial actor binding"
            )
        result.append(
            RankedOfferAuthority(
                offer_id=offer.offer_id,
                session_id=offer.session_id,
                session_digest=session.session_digest,
                mandate_id=offer.mandate_id,
                buyer_id=offer.buyer_id,
                merchant_id=offer.merchant_id,
                sku_id=offer.sku_id,
                unit_price_cents=offer.unit_price_cents,
                currency=offer.currency,
                qty=offer.qty,
                catalog_revision=offer.catalog_revision,
                inventory_revision=offer.inventory_revision,
                offer_digest=offer.offer_digest,
                order_id=f"ord-{offer.mandate_id}-{offer.offer_id}",
            )
        )
    if set(by_id) != seen_ids:
        raise RankedOfferFrameworkAuthorityError(
            "ranked candidates omit World search-session authority"
        )
    return tuple(result)


def _validate_candidate_projection(raw: Mapping[str, Any], offer: OfferSnapshot) -> None:
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
        "unit_price": offer.unit_price_cents,
        "expires_at": f"world-tick:{offer.expires_at_tick}",
    }
    mismatches = [
        label
        for label, expected_value in expected.items()
        if raw.get(label) != expected_value or type(raw.get(label)) is not type(expected_value)
    ]
    fulfillment = raw.get("fulfillment")
    if not isinstance(fulfillment, Mapping) or set(fulfillment) != {
        "method",
        "eta_days",
    }:
        mismatches.append("fulfillment")
    else:
        if (
            not isinstance(fulfillment.get("method"), str)
            or not str(fulfillment.get("method")).strip()
        ):
            mismatches.append("fulfillment.method")
        eta = fulfillment.get("eta_days")
        if eta is not None and (isinstance(eta, bool) or not isinstance(eta, int) or eta < 0):
            mismatches.append("fulfillment.eta_days")
    claims = raw.get("claims")
    if (
        not isinstance(claims, list)
        or any(not isinstance(claim, str) or not claim.strip() for claim in claims)
        or len(set(claims)) != len(claims)
    ):
        mismatches.append("claims")
    if mismatches:
        raise RankedOfferFrameworkAuthorityError(
            "ranked candidate contradicts its World offer snapshot: "
            + ", ".join(sorted(set(mismatches)))
        )


def _validate_ranking_context(payload: Mapping[str, Any]) -> None:
    reference = payload["ranking_context_reference"]
    projection = payload["ranking_context_projection"]
    if reference is None and projection is None:
        return
    if not isinstance(reference, Mapping) or not isinstance(projection, Mapping):
        raise RankedOfferFrameworkAuthorityError(
            "ranking context reference and projection must be jointly present"
        )


def _payload(inbound: Envelope) -> dict[str, Any]:
    value = inbound.action.get("payload")
    if not isinstance(value, Mapping):
        raise RankedOfferFrameworkAuthorityError("ranked response payload must be an object")
    return dict(value)


def _actor_id(value: Any, *, role: str) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split(":")
    return len(parts) >= 2 and parts[0] == role and all(parts)


def _framework_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankedOfferFrameworkAuthorityError(f"{label} must be non-empty text")
    return value.strip()


def _framework_digest(value: Any, label: str) -> str:
    digest = _framework_text(value, label)
    if len(digest) != _DIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RankedOfferFrameworkAuthorityError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _framework_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RankedOfferFrameworkAuthorityError(f"{label} must be a nonnegative integer")
    return int(value)


def _raise_field_difference(
    *,
    actual: set[Any],
    expected: set[str],
    label: str,
) -> None:
    missing = sorted(expected - actual)
    unknown = sorted(str(value) for value in actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    raise RankedOfferFrameworkAuthorityError(
        f"{label} has invalid authority fields: " + "; ".join(details)
    )


def _require_choice_keys(
    arguments: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
) -> None:
    if any(not isinstance(key, str) for key in arguments):
        raise RankedOfferModelChoiceError("ranked-offer tool argument names must be strings")
    actual = set(arguments)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise RankedOfferModelChoiceError(
            "invalid ranked-offer tool arguments: " + "; ".join(details)
        )


def _choice_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankedOfferModelChoiceError(f"{label} must be non-empty text")
    return value.strip()


def _choice_reason(value: Any) -> str:
    reason = _choice_text(value, "reason")
    if len(reason) > _MAX_REASON_LENGTH:
        raise RankedOfferModelChoiceError(
            f"reason exceeds the {_MAX_REASON_LENGTH}-character limit"
        )
    return reason


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


__all__ = [
    "CompiledRankedOfferDecision",
    "RankedOfferAuthority",
    "RankedOfferFrameworkAuthorityError",
    "RankedOfferModelChoiceError",
    "RankedOfferTurnAuthority",
    "RankedOfferTurnError",
]
