"""Skill activation selector — Python-side dispatch for "which skills fire."

Lifts the implicit skill-activation DAG out of SKILL.md ``when_to_use`` /
"defers to X" text and into testable Python predicates. The selector runs
at the head of every ``Agent.receive`` turn (after B5's cache-invalidation
hooks, before the bounded turn loop), returning an ordered tuple of skill
names the LLM is permitted to ``load_skill`` this turn.

Selection is deterministic and constrained by the public ACWorld skill registry.

This module is the lane-agnostic shape: ``_SelectorContext`` (the data
gates read from) + the ``SkillSelector`` Protocol. Lane-specific
implementations live alongside (``agents/skill_selector_buyer.py``,
``agents/skill_selector_merchant/``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from agents.interfaces import Memory
    from protocol.envelope import Envelope


#: Module-level logger; selectors emit per-gate decisions at DEBUG so a
#: live-script (or test) can see *why* a skill did or didn't activate
#: without the selector having to stash trace state on itself. Use
#: ``logging.getLogger("agents.skill_selector").setLevel(logging.DEBUG)``
#: from a driver to enable.
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SelectorContext:
    """The bundle gate functions read from.

    A frozen dataclass rather than positional args so adding a new
    dependency (e.g. ``audit`` for "did the merchant emit X recently?")
    doesn't ripple through every gate signature.

    Fields:
        env: this turn's inbound envelope.
        memory: the agent's persistent memory store (read across turns).
        merchant_data: merchant-side ``MerchantData`` snapshot.
            ``None`` for non-merchant lanes; gates that need it must
            guard with ``if ctx.merchant_data is None: return False, ...``.
        inventory_view: merchant-side world-inventory reader; same
            None-guard convention.
    """
    env: "Envelope"
    memory: "Memory"
    merchant_data: Any | None = None
    inventory_view: Any | None = None


#: One gate is a pure function from context to (should_fire, reason).
#: The reason string is for ``logging.debug`` and (eventually) any
#: introspection surface; it's not consumed by the selector logic.
Gate = Callable[[_SelectorContext], tuple[bool, str]]


class SkillSelector(Protocol):
    """Lane-specific selector. Implementations live in sibling modules."""

    def select(
        self,
        env: "Envelope",
        *,
        memory: "Memory",
        merchant_data: Any | None = None,
        inventory_view: Any | None = None,
    ) -> tuple[str, ...]:
        """Return the ordered tuple of skill names activated this turn."""
        ...


class SkillSelectionError(RuntimeError):
    """A selector gate failed while strict skill selection was enabled.

    Gate failures are framework failures, not model decisions.  The exception
    deliberately exposes only stable identifiers so formal result handling can
    classify it without persisting arbitrary gate exception text.
    """

    error_code = "skill_selector_failure"

    def __init__(self, *, lane: str, skill_name: str, cause: Exception) -> None:
        self.lane = lane
        self.skill_name = skill_name
        self.cause_type = type(cause).__name__
        super().__init__(
            f"{lane}/{skill_name} selector gate failed with {self.cause_type}"
        )


def run_gates(
    ctx: _SelectorContext,
    gates: dict[str, Gate],
    lane: str = "agent",
    *,
    strict: bool = False,
) -> list[str]:
    """Iterate ``gates``, return the names whose gate fired.

    Emits one ``DEBUG`` log line per gate with the reason string —
    enables a live driver to see exactly why each skill did or didn't
    fire without the selector having to stash trace state.
    """
    fired: list[str] = []
    for name, gate in gates.items():
        try:
            should_fire, reason = gate(ctx)
        except Exception as exc:  # noqa: BLE001 — surface gate bugs visibly
            if strict:
                # Formal logs retain only stable framework identifiers. A gate
                # exception may contain scenario values or provider-derived
                # text and must not be copied into console or experiment logs.
                log.debug(
                    "%s/%s gate raised %s",
                    lane,
                    name,
                    type(exc).__name__,
                )
                raise SkillSelectionError(
                    lane=lane,
                    skill_name=name,
                    cause=exc,
                ) from exc
            log.debug("%s/%s gate raised: %s", lane, name, exc)
            continue
        decision = "fire" if should_fire else "skip"
        log.debug("%s/%s %s: %s", lane, name, decision, reason)
        if should_fire:
            fired.append(name)
    return fired


__all__ = [
    "Gate",
    "SkillSelectionError",
    "SkillSelector",
    "_SelectorContext",
    "run_gates",
    "log",
]
