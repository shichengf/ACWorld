"""Platform-derived read authority for actor-published evidence.

An actor asks to publish evidence about a subject.  It does not get to say who
may later read that evidence: a model that authored its own ``read_acl`` could
widen disclosure to actors with no relationship to the subject, or narrow it to
hide evidence from a counterparty that is entitled to see it.

The Platform therefore derives the ACL from authoritative World relationships
and overwrites whatever arrived on the wire.  Two rules are implemented:

* the subject names an order that World knows -> the order's Buyer and Merchant
  are participants, because both are already parties to that transaction;
* anything else -> the publisher alone, which is the conservative default and
  never discloses more than the actor already had.

``platform:*`` services are deliberately absent from every derived ACL. Trusted
service reads are authorized by role inside the World read path, not by
enumerating services here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.world_client import WorldClient


def derive_evidence_read_acl(
    *,
    world_client: WorldClient,
    owner_id: str,
    subject_id: str,
) -> tuple[str, ...]:
    """Return the sorted read ACL the Platform will seal into a record."""

    participants = {owner_id}
    participants.update(_order_participants(world_client, subject_id))
    # A governance case binding would extend the participant set here. World
    # exposes no read-only case projection today, so a case-scoped record falls
    # back to the owner-only rule rather than guessing at its participants.
    return tuple(sorted(participants))


def _order_participants(world_client: WorldClient, subject_id: str) -> frozenset[str]:
    """Return the Buyer and Merchant of the order named by ``subject_id``."""

    if not subject_id.strip():
        return frozenset()
    try:
        state: dict[str, Any] | None = world_client.read_order_protocol_state(
            subject_id,
            caller="platform:evidence",
        )
    except Exception:
        # An unreadable or non-order subject is not an error: it simply carries
        # no transaction relationship, so the owner-only rule applies.
        return frozenset()
    if not isinstance(state, dict):
        return frozenset()
    participants = {
        str(state[field])
        for field in ("buyer_id", "merchant_id")
        if isinstance(state.get(field), str) and str(state[field]).strip()
    }
    return frozenset(participants)
