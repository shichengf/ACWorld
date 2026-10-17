"""Exception hierarchy for the agents package."""

from __future__ import annotations

# ``MandateLeak`` is canonically defined in :mod:`protocol.envelopes` because
# the structural leak guard (:func:`protocol.envelopes.assert_no_private_leak`)
# is the actual raiser. The agents-package alias keeps existing
# ``from agents.errors import MandateLeak`` callers working — both the
# raise site and the catch site reference the same class, so ``except``
# clauses match cleanly across layers.
from protocol.envelopes import MandateLeak  # noqa: F401  (re-export)


class AgentError(Exception):
    """Base error for the agents package."""


class SkillNotFound(AgentError):
    """No skill in the agent's skill list handles the inbound action kind."""


class MultiEmitForbidden(AgentError):
    """A skill returned more than one envelope. Skill contract: 0 or 1 only.

    Multi-emit (e.g. platform fanout) is a bus concern, not a skill concern.
    See AGENT_CLASSES.md §9 Q2.
    """


class FrontmatterError(AgentError):
    """A ``SKILL.md`` file's YAML frontmatter is missing, malformed, or
    missing required fields (``name``, ``description``).

    Surfaced at scenario / agent construction time by ``FilesystemSkillLoader``.
    """


class BudgetExceeded(AgentError):
    """The agent's bounded internal turn loop hit its step budget.

    Raised by ``Agent.receive`` when the LLM produces more than the
    configured maximum number of internal steps without terminating
    (``emit_envelope`` or ``no_reply``). See AGENT_DESIGN.md §3.1.
    """


class SkillNotActivated(AgentError):
    """The LLM tried to ``load_skill`` a name not in the selector's
    activated tuple for this turn.

    B4 (ACWorld skill-selection design): when a ``SkillSelector`` is attached
    to an agent, ``Agent.receive`` filters the per-step prompt's manifest
    list to the activated tuple AND enforces that the LLM cannot load
    anything outside it. The two layers are complementary — filtering
    means the LLM normally doesn't try, enforcement catches the case
    where the LLM hallucinates a name from training-data priors anyway.

    Raised inside the bounded turn loop; the script / runtime can catch
    and report cleanly. With ``selector=None`` (the default), this
    exception is never raised — all skills in ``enabled_skills`` remain
    loadable as before.
    """


class BudgetReconcileViolation(AgentError):
    """The reconciler refused to emit a settle that would exceed
    ``PRIVATE_UTILITY.max_budget``.

    Raised by ``_reconcile_outbound_payload`` when the price it would
    write into ``platform.settle_payment.payload.agreed_price`` (from
    the selected offer in TRANSACTION memory) plus ``cumulative_spend``
    would total above ``max_budget``. This catches the case where
    ``discovery-search`` committed an over-budget offer to
    ``selected_offer`` (a skill-body violation) and ``purchase-confirmation``
    then tries to settle it.

    Without this guard, the reconciler would silently overwrite the
    LLM's invented ``agreed_price`` with the over-budget value from
    ``selected_offer``, the PSP would accept it as a list-price settle,
    and the buyer would unknowingly exceed budget — discovered by the
    live-LLM driver in 2026-05. The PSP's consent check passes a
    list-price settle through unconditionally, so the budget invariant
    must be enforced *before* the envelope reaches the bus.
    """


class GroundingRequired(AgentError):
    """The buyer tried to accept a purchase carrying a feature ``must_have``
    without having grounded the chosen sku — item 3 grounding teeth (code).

    A feature must_have (noise_cancellation, "merino wool", …) may be satisfied
    only by a GROUNDING read of the listing's attributes, never by the offer's
    marketing claims or the LLM's say-so. So before emitting a
    ``commerce.accept_offer`` for a mandate with feature must_haves, the buyer
    must have grounded the chosen sku — evidenced by a ``world.get_listing`` /
    ``world.search_catalog`` read of it in this decision's step history, OR a
    recorded ``grounded_attributes`` on ``TRANSACTION.selected_offer`` (the
    cross-turn case). This is PRESENCE (show-you-grounded); whether the grounded
    evidence actually SATISFIES the must_have is the scorer's fabricated-grounding
    check. Without a real read it is fabricated grounding — refused at the agent
    boundary so an ungrounded feature-buy never reaches the bus.
    """


class DecisionRecordIncomplete(AgentError):
    """A buyer's purchase acceptance lacks a usable, consistently bound decision.

    ``TRANSACTION.selected_offer`` must identify the emitted offer. When a
    ``TRANSACTION.budget_audit`` exists, its candidate entries must record
    eligibility and a rejection stage and reason for any ineligible item.
    Optional choice fields must agree between the record and emitted action
    when both provide them. Free-text rationale is optional.

    These guards check record presence and consistency before the action
    reaches the bus. Task scoring checks the commercial choice and recorded
    eligibility. Social evidence is task-dependent; fetching friend reviews
    is not a universal requirement of this guard.
    """
