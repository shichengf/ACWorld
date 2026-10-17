"""Shared, pure merchant-consent verification.

ONE implementation of the settle-consent contract, called by BOTH the runtime
platform (``agents.platform.PlatformService``) and the offline step scorer, so
the two never drift. Pure: no I/O, no world/audit handles — callers pass the
already-extracted facts.

Contract (unchanged from the original platform guard):
1. **List-price settle** — ``agreed_cents == list_cents`` is consent (the public
   catalog price is the merchant's commitment).
2. **Negotiated settle** — an audited ``commerce.accept_offer`` /
   ``commerce.counter_offer`` FROM the named merchant, for this sku, at this
   exact integer-cents ``unit_price``.
3. Otherwise: no consent.

v1 decision (documented): the match is on ``(sender == merchant, kind, sku_id,
unit_price)`` only. Quantity and offer-id lineage are NOT part of the consent
predicate — this preserves the existing runtime behavior. A buyer-authored
envelope never counts (``from_id`` must equal the merchant). The scorer is
responsible for passing only commitments that CAUSALLY PRECEDE the settle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

CONSENT_KINDS = ("commerce.accept_offer", "commerce.counter_offer")


@dataclass(frozen=True)
class PriceCommitment:
    """A merchant-authored price signal extracted from one audited envelope."""
    from_id: str
    kind: str
    sku_id: str
    unit_price: "int | None"
    msg_id: str = ""


@dataclass(frozen=True)
class ConsentResult:
    ok: bool
    basis: str                       # "list_price" | "negotiated" | "none"
    evidence_msg_id: "str | None" = None
    reason: str = ""


def verify_merchant_consent(
    *,
    agreed_cents: int,
    sku_id: str,
    merchant_id: str,
    list_cents: "int | None",
    commitments: "Iterable[PriceCommitment]",
) -> ConsentResult:
    """Return a structured :class:`ConsentResult` for a settle at ``agreed_cents``."""
    if list_cents is not None and agreed_cents == list_cents:
        return ConsentResult(ok=True, basis="list_price",
                             reason="agreed price equals public list price")
    for c in commitments:
        if (c.from_id == merchant_id and c.kind in CONSENT_KINDS
                and c.sku_id == sku_id and c.unit_price == agreed_cents):
            return ConsentResult(ok=True, basis="negotiated", evidence_msg_id=c.msg_id or None,
                                 reason="merchant accept/counter at this sku+price")
    return ConsentResult(
        ok=False, basis="none",
        reason=(f"no list-price match and no merchant accept/counter from {merchant_id!r} "
                f"for sku {sku_id!r} at {agreed_cents} cents"))


__all__ = ["CONSENT_KINDS", "PriceCommitment", "ConsentResult", "verify_merchant_consent"]
