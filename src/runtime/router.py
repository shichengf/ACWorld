"""Role-partition + private-utility guard.

The router is the gate the bus consults on every send. It is a pure function
over envelopes; no I/O, no state beyond the partition table it was constructed
with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from protocol.actions import ActionKind, PARTITION_ALLOW, is_send_allowed
from protocol.errors import PartitionViolation
from runtime.errors import PrivateUtilityLeak
from runtime.privacy import (
    _COUNTERPARTY,
    _PRIVATE_FIELD_OWNER_SIDE,
    MoneySecret,
    SecretRegistry,
    find_leak,
)

if TYPE_CHECKING:
    from agents.interfaces import Memory
    from protocol.envelope import Envelope


class Router:
    """Stateless validator for outbound envelopes."""

    def __init__(
        self,
        secrets: "SecretRegistry | None" = None,
        *,
        memories: "dict[str, Memory] | None" = None,
    ) -> None:
        """Construct the router. Reads the partition table from ``protocol.actions``.

        ``secrets`` is the explicit per-episode :class:`~runtime.privacy.SecretRegistry`
        the payload check enforces (built from the scenario answer key). With no
        registry (``Router()``) the privacy check is a no-op — the legacy default
        used on bare constructions and on routing-only tests.

        ``memories`` is a backward-compat shim: if a legacy caller passes agent
        memories, ONLY explicitly recognized monetary secret fields (``max_budget``
        / ``floor_price`` / …) are converted into a registry — never a whole-bucket
        scan (which would collide ``qty=3`` with ``max_negotiation_rounds=3``).
        """
        if secrets is not None:
            self._secrets = secrets
        elif memories:
            self._secrets = _registry_from_memories(memories)
        else:
            self._secrets = SecretRegistry()

    def check_outbound(self, env: "Envelope", sender_id: str) -> None:
        """Raise ``PartitionViolation`` if ``sender_id`` may not emit ``env.action.kind``.

        ``sender_id`` is the agent id (``buyer:intent``); the side prefix
        (``buyer``) is derived and matched against the allow-table.
        """
        kind = ActionKind(env.action["kind"])
        sender_role = _role(sender_id)
        receiver_role = _role(env.to)
        if not is_send_allowed(kind, sender_role, receiver_role):
            raise PartitionViolation(
                f"{sender_role!r} may not send {kind.value!r} to {receiver_role!r}"
            )

    def check_payload(self, env: "Envelope", sender_id: str) -> None:
        """Raise ``PrivateUtilityLeak`` if ``env`` would disclose a registered
        secret across a counterparty boundary.

        Delegates to the path-aware v1 detector (:func:`runtime.privacy.find_leak`):
        an explicit private field name, or a registered secret amount in a
        non-transaction-price field, crossing the buyer<->merchant boundary. A
        legitimate transaction price (an offer ``unit_price`` / a settle
        ``agreed_price``) equal to a secret is exempt by (ActionKind, path).
        The raised error carries a sanitized :class:`~runtime.privacy.LeakFinding`
        (no raw value) so the bus can write a security-event sidecar record.
        """
        # No early-return on an empty monetary registry: explicit private
        # FIELD-NAME detection needs no registered amount and must still fire.
        finding = find_leak(env, self._secrets)
        if finding is not None:
            raise PrivateUtilityLeak(
                "payload would disclose a registered secret across a counterparty "
                "boundary",
                finding=finding,
            )

    def validate_register(
        self,
        agent_id: str,
        outbound_action_kinds: frozenset[str],
        inbound_action_kinds: frozenset[str],
    ) -> None:
        """Validate at register time that an agent's declared partitions are coherent.

        Per WORLD_CLASSES.md §6, role partition is checked at register time, not
        just at send. A buyer agent declaring it emits ``update_reputation``
        fails registration.
        """
        side = _role(agent_id)
        for kind_value in outbound_action_kinds:
            kind = ActionKind(kind_value)
            allowed = PARTITION_ALLOW.get(kind)
            if allowed is None or side not in allowed["senders"]:
                raise PartitionViolation(f"{side!r} may not emit {kind.value!r}")
        for kind_value in inbound_action_kinds:
            kind = ActionKind(kind_value)
            allowed = PARTITION_ALLOW.get(kind)
            if allowed is None or side not in allowed["receivers"]:
                raise PartitionViolation(f"{side!r} may not receive {kind.value!r}")


def _role(address: str) -> str:
    return address.split(":", 1)[0]


def _registry_from_memories(memories: "dict[str, Memory]") -> SecretRegistry:
    """Backward-compat shim: build a registry from legacy agent memories,
    converting ONLY explicitly recognized monetary secret fields. Never scans the
    whole ``PRIVATE_UTILITY`` bucket (booleans / round counts must not become
    secrets that collide with payload quantities)."""
    from agents.types import MemoryType

    secrets: list[MoneySecret] = []
    for agent_id, memory in memories.items():
        if memory is None:
            continue
        for key in memory.keys(MemoryType.PRIVATE_UTILITY):
            owner_side = _PRIVATE_FIELD_OWNER_SIDE.get(key)
            if owner_side is None:
                continue  # only recognized monetary secret fields, never the bucket
            value = memory.read(MemoryType.PRIVATE_UTILITY, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            secrets.append(MoneySecret(
                owner_id=agent_id, name=key, amount_cents=int(value),
                counterparty_roles=_COUNTERPARTY.get(owner_side, frozenset()),
            ))
    return SecretRegistry(money=tuple(secrets))
