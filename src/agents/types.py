"""Passive data types for the agents package.

The primitives from AGENT_CLASSES.md §1 are split across files:
    * ``Agent``           — concrete class, lives in ``base.py``
    * ``SkillLoader``     — Protocol, lives in ``interfaces.py``
    * ``Memory``          — Protocol, lives in ``interfaces.py``
    * ``SkillManifest``   — passive data, lives here
    * ``SkillDocument``   — passive data, lives here
    * ``AgentInputs``     — passive data, lives here

The two enums (``SkillGroup``, ``MemoryType``) are also passive data and live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from world.tools import WorldTools


# --- Skill grouping ----------------------------------------------------

class SkillGroup(str, Enum):
    """The seven research-shaped groups from VIBE_COMMERCE §6."""
    INTERPRETATION = "interpretation"   # vibe parsing, taste grounding, constraint extraction
    DISCOVERY = "discovery"             # query formulation, search, filtering, ranking
    ECONOMIC = "economic"               # budget tracking, price-quality, negotiation
    TRUST = "trust"                     # reputation reading, claim verification
    AUTHORIZATION = "authorization"     # mandate verification, payment release, atomic settlement
    OPERATIONS = "operations"           # fulfillment, return, refund, dispute, ruling
    SAFETY = "safety"                   # role partition, private-utility guard, manipulation resistance


# --- Memory typing -----------------------------------------------------

class MemoryType(str, Enum):
    """The four memory types from VIBE_COMMERCE §7.

    The runtime enforces that ``PRIVATE_UTILITY`` values never appear in an
    outbound envelope's payload (AGENT_CLASSES.md §4).
    """
    PREFERENCE = "preference"           # taste, brand affinity
    TRANSACTION = "transaction"         # past orders, returns
    TRUST = "trust"                     # reputation, dispute outcomes
    PRIVATE_UTILITY = "private_utility"  # max budget, walk-away price, floor price


# --- Skill bundles ----------------------------------------------------

@dataclass(frozen=True)
class SkillManifest:
    """Lightweight, spawn-time view of a skill bundle (AGENT_CLASSES.md §4.2).

    The agent sees only manifests at spawn — never full SKILL.md bodies — so
    many skills can be enabled without bloating the system prompt. The full
    ``SkillDocument`` is fetched on demand via ``SkillLoader.load``.

    ``digest`` is a stable content hash over the manifest frontmatter, the
    full SKILL.md body, and bundled references / scripts. Replay records the
    digest so ``skill@A`` and ``skill@B`` are distinguishable in reports.
    """
    name: str
    description: str
    when_to_use: str | None = None
    group: "SkillGroup | None" = None
    path: str = ""
    digest: str = ""


@dataclass(frozen=True)
class SkillDocument:
    """Full skill bundle loaded on demand for the turn that needs it.

    Per AGENT_CLASSES.md §4.1, the document body and its referenced files /
    bundled scripts enter context only for the turn that activates the skill;
    they are not persisted in cross-turn state.
    """
    manifest: SkillManifest
    instructions: str
    references: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()


# --- Scenario inputs --------------------------------------------------

@dataclass(frozen=True)
class AgentInputs:
    """The three optional dicts a scenario hands an agent at spawn.

    See AGENT_CLASSES.md §5. Composed into the agent's system prompt **once**
    at construction; never re-rendered per turn.
    """
    persona: dict[str, Any] | None = None    # who this agent is (DESIGN §5.7 axes)
    mandate: dict[str, Any] | None = None    # buyer side: goal, hard_constraints, authority
    policy: dict[str, Any] | None = None     # merchant: floor_price, refund_policy
                                              # platform: lenient_market / strict_market


# --- Per-turn reasoning trace recorder (item 5) -----------------------

class TurnTrace:
    """Mutable per-turn buffer the agent fills inside ``receive``.

    The turn loop calls :meth:`finalize` once at its terminal step with the
    ``history`` it already assembled. The transport reads this buffer after the
    turn, enriches it with the agent's private-memory snapshot, and flushes a
    ``TraceRecord`` to the trace sidecar (never to ``audit.jsonl``). When no
    trace sink is wired, no recorder is created and the loop's ``finalize`` call
    is simply skipped — tracing is a pure add-on with zero on-wire effect.
    """

    def __init__(self) -> None:
        self.steps: tuple[dict[str, Any], ...] = ()
        self.result: Any = None          # the emitted Envelope, or None for no_reply
        self.terminal: str | None = None
        self.finalized: bool = False
        # Live framework-owned history for an exception that terminates the
        # decision before its normal terminal step.  The raw channel response is
        # never stored here.  ``finalize_failure`` snapshots only completed
        # load/memory/tool operations already accepted by the Agent loop.
        self._bound_steps: list[dict[str, Any]] | None = None
        #: R1 stitch key — set on a re-entrant turn so the assembled TraceRecord
        #: ties its multi-round steps together. ``None`` for synchronous turns.
        self.decision_id: str | None = None

    def bind_steps(self, steps: "list[dict[str, Any]]") -> None:
        """Bind the live framework history for truthful failure finalization."""

        self._bound_steps = steps

    def finalize(
        self, steps: "list[dict[str, Any]]", *, result: Any, terminal: str | None = None
    ) -> None:
        """Record the turn's step stream + terminal outcome. Called once."""
        self.steps = tuple(steps)
        self.result = result
        self.terminal = terminal or ("emit_envelope" if result is not None else "no_reply")
        self.finalized = True

    def finalize_failure(self, *, terminal: str) -> None:
        """Finalize a failed decision without claiming an emitted envelope."""

        self.steps = tuple(self._bound_steps or ())
        self.result = None
        self.terminal = terminal
        self.finalized = True


# --- Per-turn context -------------------------------------------------

@dataclass(frozen=True)
class AgentContext:
    """The context the runtime passes into ``Agent.receive`` and ``Skill.run``.

    Constructed fresh per send by the runtime; bound to the receiving agent's id.
    """
    world: "WorldTools"           # caller-scoped read view (WORLD_CLASSES §2.2)
    turn: int                     # 0-indexed turn within the current episode
    scratchpad: tuple[str, ...]   # last N envelopes summarized (default N=10)
    trace: "TurnTrace | None" = None  # item 5: per-turn reasoning recorder (None = tracing off)
    remote_world: bool = False    # R1: world reads go over the bus (suspend/resume), not in-process


# --- Tool surface -----------------------------------------------------

# Whitelist of World reads the Agent may compile from typed business intents.
# The model never invokes these transport-level names directly.
ALLOWED_WORLD_TOOLS: frozenset[str] = frozenset({
    "world.search_catalog",
    "world.get_listing",
    "world.is_in_stock",
    "world.get_merchant_reputation",
    "world.get_balance",
    "world.get_order",
    "world.get_friends",
    "world.get_friend_reviews",
    "world.get_review_evidence",
    "world.get_evidence_record",
    "world.get_mandate_revisions",
    "world.get_listing_claim",
    "world.list_listing_claims",
})
