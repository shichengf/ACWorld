"""Pure VCP envelope/payload constructors.

These functions build dicts matching the VCP_SPEC.md §4 payload shapes
(``GroundedOffer``, ``InquiryResponse``, etc.) and a structural leak-guard
that catches PRIVATE_UTILITY values before they hit the wire.

They live in the protocol layer so any lane (merchant / buyer / platform)
that wants to emit a wire-stable envelope reaches for the same constructor
and gets the same shape. The constructors are intentionally narrow:

* validate at the boundary (integer minor-units, allowed enums);
* refuse to embed quantitative cues that VCP_SPEC forbids on the wire
  (e.g. raw inventory counts in availability answers);
* return plain dicts — no I/O, no clock, no lane-private state.

Anything that *decides* which envelope to build (the skill body) stays on
its lane. These constructors only *render* the decision.
"""

from __future__ import annotations

import uuid
from typing import Any

from protocol.errors import ProtocolError


# --- errors -----------------------------------------------------------

class ClaimNotPermitted(ProtocolError):
    """A grounded offer carried a claim outside ``permitted_claims``.

    The truthfulness contract (VCP_SPEC §4.3 / claim-truthfulness skill)
    is enforced at envelope-construction time so a misbehaving skill
    cannot ship a non-permitted claim regardless of LLM output.
    """


class EnvelopeShapeError(ProtocolError):
    """An envelope constructor was called with an invalid shape — wrong
    enum value, non-integer money, forbidden quantitative cue, etc."""


class MandateLeak(ProtocolError):
    """A PRIVATE_UTILITY value appeared in an outbound payload.

    Raised by :func:`assert_no_private_leak` before the runtime hands the
    envelope to the bus. The merchant's ``private-utility-guard`` skill is
    the structural check that calls this on every emit.

    Mirrors :class:`agents.errors.MandateLeak` so existing callers that
    catch the agent-level error keep working; the protocol-level class is
    the canonical one going forward.
    """


# --- VCP_SPEC enums ---------------------------------------------------

_INQUIRY_CATEGORIES: frozenset[str] = frozenset((
    "shipping", "availability", "attribute", "policy", "general", "unknown_sku",
))


# --- GroundedOffer ----------------------------------------------------

def build_grounded_offer(
    *,
    product: str,
    sku_id: str,
    qty: int,
    unit_price: int,
    permitted_claims: tuple[str, ...],
    claims: tuple[str, ...] = (),
    eta_days: int = 3,
    fulfillment_method: str = "standard",
) -> dict[str, Any]:
    """Construct a VCP ``GroundedOffer`` payload (VCP_SPEC.md §4.3).

    Enforces:

    * integer minor-unit ``unit_price``;
    * ``claims ⊆ permitted_claims`` (truthfulness contract).

    Raises:
        EnvelopeShapeError: ``unit_price`` is not an integer.
        ClaimNotPermitted: a claim is outside the permitted set.
    """
    if int(unit_price) != unit_price:
        raise EnvelopeShapeError("unit_price must be integer minor units")
    bad = [c for c in claims if c not in permitted_claims]
    if bad:
        raise ClaimNotPermitted(f"claims not permitted: {bad}")
    return {
        "offer_id": str(uuid.uuid4()),
        "sku_id": sku_id,
        "product": product,
        "qty": int(qty),
        "unit_price": int(unit_price),
        "fulfillment": {"method": fulfillment_method, "eta_days": int(eta_days)},
        "claims": list(claims),
        "expires_at": None,
        "idempotency_key": str(uuid.uuid4()),
    }


# --- InquiryResponse --------------------------------------------------

def build_inquiry_response(product: str, category: str, answer: Any, *,
                           sku_id: str = "S1") -> dict[str, Any]:
    """Construct a ``commerce.respond_inquiry`` action dict.

    Pure constructor. Validates ``category`` against a fixed set and refuses
    to embed a quantitative inventory integer in an ``availability`` answer
    — the buyer must see the qualitative band (see
    :func:`protocol.pricing.availability_band`).

    Raises:
        EnvelopeShapeError: ``unknown_category`` or
            ``numeric_inventory_not_permitted``.
    """
    if category not in _INQUIRY_CATEGORIES:
        raise EnvelopeShapeError("unknown_category")
    if category == "availability" and isinstance(answer, int) and not isinstance(answer, bool):
        raise EnvelopeShapeError("numeric_inventory_not_permitted")
    return {"kind": "commerce.respond_inquiry",
            "payload": {"sku_id": sku_id, "product": product,
                        "category": category, "answer": answer}}


# --- Leak guard -------------------------------------------------------

def assert_no_private_leak(payload: Any, private_values: list[Any]) -> None:
    """Raise :class:`MandateLeak` if any PRIVATE_UTILITY value appears in
    ``payload`` (structural check).

    The cheap, last-mile guard the ``private-utility-guard`` skill runs
    before any send — floor price, walk-away, urgency cue, etc., must
    never reach the wire. Pure (no I/O).

    LAYER NOTE (not a competing algorithm): this is the AGENT-SIDE pre-send
    guard — an agent self-checks its OWN declared private values (conservative
    ``str(v) in repr(payload)`` substring). The AUTHORITATIVE boundary is the
    runtime :meth:`runtime.router.Router.check_payload`, enforcing the
    scenario-registered :class:`~runtime.privacy.SecretRegistry` path-aware and
    cross-party (and recording a sanitized security event on a block). The two
    are complementary layers, not two answers to the same question.
    """
    blob = repr(payload)
    for v in private_values:
        if v is None:
            continue
        if str(v) in blob:
            raise MandateLeak(f"private-utility value {v!r} present in outbound payload")


__all__ = [
    "ClaimNotPermitted",
    "EnvelopeShapeError",
    "MandateLeak",
    "assert_no_private_leak",
    "build_grounded_offer",
    "build_inquiry_response",
]
