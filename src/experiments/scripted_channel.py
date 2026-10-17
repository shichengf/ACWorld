"""Deterministic, model-free business-decision channel for environment studies.

CommerceWorld environment studies need multi-actor traces that exercise the
real ``Episode -> Runtime -> Agent -> Platform -> World`` path without paying an
LLM provider.  This module supplies one reusable, domain-agnostic channel that
satisfies the same :class:`agents.inference.InferenceChannel` Protocol the
``OpenAIChannel`` satisfies, so the local Agent still owns every CommerceWorld
interface operation: World reads, public-reference resolution, argument
validation, authority, route registry, envelopes, idempotency, correlation, and
protocol continuation.

The channel recovers the provider-neutral public request from the prompt the
Agent already renders (:meth:`agents.business_decision.LLMDecisionRequestV1.to_prompt`),
**fully re-validates it through the authoritative
:class:`~agents.business_decision.LLMDecisionRequestV1` constructor before any
policy runs**, hands a read-only public view to a deterministic policy callback,
and serializes the callback's
:class:`~agents.business_decision.LLMBusinessDecisionV1` back into the exact wire
text the Agent will parse -- itself re-validated against the advertised intent
set.  The callback never receives a World, Platform, task case, server oracle,
or internal identifier as an argument; the only inputs are the actor-visible
business facts, the actor's role/goal/phase, and the public allowed-intent set.

Reconstruction deliberately reuses ``LLMDecisionRequestV1`` / ``BusinessIntentSpec``
/ ``LLMBusinessDecisionV1.parse`` from :mod:`agents.business_decision` rather
than adding a parser there: that module is a frozen ``CommerceWorld-v2.1``
suite-manifest file, so the shared validator is reused, not duplicated or
modified.

Security boundary.  A Python callback can, in principle, close over a World
handle, an oracle, or a network socket; the structural guarantees here cover the
*argument surface* only.  A scripted policy must therefore be treated as
trusted, frozen, and audited code, and the environment-study process must run
with the network disabled (a Gate 1b / process-level responsibility).  This
channel makes and holds no network resource itself: :attr:`provider_calls` is
always ``0``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from agents.business_decision import (
    LLM_BUSINESS_DECISION_SCHEMA_V1,
    LLM_DECISION_REQUEST_SCHEMA_V1,
    BusinessDecisionContractError,
    BusinessIntentSpec,
    LLMBusinessDecisionV1,
    LLMDecisionRequestV1,
)
from agents.inference import BusinessDecisionResponseV1

_ROLE_VALUES = frozenset({"buyer", "merchant"})
# Placeholder Agent-private source name.  ``BusinessIntentSpec`` requires a
# non-empty source name, but the public request projection deliberately omits
# it (it is Agent-private).  Reconstruction only needs the intent name and its
# public parameter schema for validation; the source name is never read by the
# authoritative validators and never leaves this channel.
_RECONSTRUCTED_SOURCE_NAME = "scripted-environment-study"
# The public request has exactly these root keys (see
# ``LLMDecisionRequestV1.to_public_dict``); anything else is a tampered prompt.
_ALLOWED_REQUEST_KEYS = frozenset(
    {"schema_version", "role", "goal", "phase", "observations", "allowed_intents", "response_contract"}
)
_EXPECTED_RESPONSE_CONTRACT = {
    "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
    "required": ["schema_version", "intent", "arguments"],
    "optional": ["message"],
}


class ScriptedBusinessDecisionError(BusinessDecisionContractError):
    """A scripted policy produced no valid provider-neutral business decision.

    Subclasses :class:`~agents.business_decision.BusinessDecisionContractError`
    (itself a ``ValueError``) so callers can treat a scripted rejection and a
    real model-interface rejection uniformly.  The channel raises it whenever the
    prompt is malformed, the request fails authoritative validation, the policy
    returns the wrong type, or the decision selects an intent outside the
    advertised set, an internal address or route, or schema-invalid arguments.
    """


@dataclass(frozen=True, slots=True)
class ScriptedDecisionContextV1:
    """The complete, genuinely read-only view a scripted policy may see.

    Every mapping is a deep-frozen copy of the actor-visible public request: a
    policy cannot mutate it, and it holds no reference to World, Platform, the
    task case, a server oracle, or any internal identifier, so a policy can only
    decide from the same business facts a model would receive.
    """

    decision_id: str | None
    role: str
    goal: str
    phase: str
    observations: tuple[Mapping[str, Any], ...]
    allowed_intents: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observations", tuple(_freeze_public(row) for row in self.observations)
        )
        object.__setattr__(
            self, "allowed_intents", tuple(_freeze_public(row) for row in self.allowed_intents)
        )

    @property
    def allowed_intent_names(self) -> frozenset[str]:
        """Return the set of intent names the policy may select."""

        return frozenset(str(row["intent"]) for row in self.allowed_intents)

    def allowed_intent(self, name: str) -> Mapping[str, Any]:
        """Return the public spec (intent/description/category/parameters).

        Raises:
            KeyError: ``name`` is not one of the advertised intents.
        """

        for row in self.allowed_intents:
            if row.get("intent") == name:
                return row
        raise KeyError(name)


ScriptedBusinessDecisionPolicyV1 = Callable[
    [ScriptedDecisionContextV1], LLMBusinessDecisionV1
]


@dataclass(frozen=True, slots=True)
class ScriptedDecisionRecordV1:
    """Hashes-only provenance for one scripted decision. No private raw text."""

    sequence: int
    actor_id: str
    role: str
    decision_id: str | None
    request_sha256: str
    decision_sha256: str


class ScriptedBusinessDecisionChannelV1:
    """A model-free :class:`InferenceChannel` bound to one actor and policy.

    Construct one instance per Buyer or Merchant actor; instances never share
    mutable state.  The channel makes no network call and holds no provider
    credential, so :attr:`provider_calls` is always ``0``.
    """

    # The Agent inspects these Protocol attributes before dispatching.
    supports_business_decisions = True
    # ``decision_id`` is accepted as local evidence context and validated, but,
    # as with every business-decision channel, it is never serialized outward.
    supports_decision_evidence_context = True

    def __init__(
        self,
        *,
        actor_id: str,
        role: str,
        policy: ScriptedBusinessDecisionPolicyV1,
    ) -> None:
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("scripted channel requires a non-empty actor id")
        if role not in _ROLE_VALUES:
            raise ValueError("scripted channel role must be 'buyer' or 'merchant'")
        if not callable(policy):
            raise ValueError("scripted channel policy must be callable")
        self._actor_id = actor_id
        self._role = role
        self._policy = policy
        self._log: list[ScriptedDecisionRecordV1] = []

    @property
    def actor_id(self) -> str:
        return self._actor_id

    @property
    def role(self) -> str:
        return self._role

    @property
    def provider_calls(self) -> int:
        """Always ``0``: a scripted channel never contacts a provider."""

        return 0

    @property
    def decision_log(self) -> tuple[ScriptedDecisionRecordV1, ...]:
        """Return the ordered hashes-only record of scripted decisions."""

        return tuple(self._log)

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        """Recover the request, validate it, run the policy, and return text."""

        if decision_id is not None and (
            not isinstance(decision_id, str)
            or not decision_id
            or len(decision_id) > 512
            or any(ord(char) < 32 for char in decision_id)
        ):
            raise ValueError("business decision evidence context is invalid")

        # Full authoritative validation happens here, before the policy runs, so
        # a policy never observes a request that would be rejected.
        request = _recover_public_request(
            user_prompt, expected_role=self._role, decision_id=decision_id
        )
        context = ScriptedDecisionContextV1(
            decision_id=decision_id,
            role=request.role,
            goal=request.goal,
            phase=request.phase,
            observations=tuple(copy.deepcopy(dict(row)) for row in request.observations),
            allowed_intents=tuple(spec.to_public_dict() for spec in request.allowed_intents),
        )

        try:
            decision = self._policy(context)
        except BusinessDecisionContractError as exc:
            raise ScriptedBusinessDecisionError(
                "scripted policy produced an invalid business decision"
            ) from exc
        if not isinstance(decision, LLMBusinessDecisionV1):
            raise ScriptedBusinessDecisionError(
                "scripted policy must return an LLMBusinessDecisionV1"
            )

        content = _canonical_decision_text(decision)
        # Re-validate against the advertised intent set with the authoritative
        # parser.  This is the fail-closed boundary: an intent outside the phase,
        # an internal service address, an action namespace, or a schema-invalid
        # argument raises here, before the text reaches the Agent.
        try:
            LLMBusinessDecisionV1.parse(content, allowed_intents=request.allowed_intents)
        except BusinessDecisionContractError as exc:
            raise ScriptedBusinessDecisionError(
                "scripted decision failed provider-neutral revalidation"
            ) from exc

        self._log.append(
            ScriptedDecisionRecordV1(
                sequence=len(self._log),
                actor_id=self._actor_id,
                role=self._role,
                decision_id=decision_id,
                request_sha256=request.public_sha256,
                decision_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
        return BusinessDecisionResponseV1(
            content=content,
            response_chars=len(content),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


def _reject_constant(value: str) -> None:
    raise ScriptedBusinessDecisionError(f"business prompt has a non-finite constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScriptedBusinessDecisionError(f"business prompt has a duplicate key {key!r}")
        result[key] = value
    return result


def _recover_public_request(
    user_prompt: str,
    *,
    expected_role: str,
    decision_id: str | None,
) -> LLMDecisionRequestV1:
    """Recover the request and revalidate it with the authoritative constructor.

    Mirrors the deterministic-baseline recovery used by the benchmark family
    channels (the Agent renders ``preamble + json.dumps(request.to_public_dict())``
    and the preamble has no ``{``), but then rebuilds an
    :class:`LLMDecisionRequestV1`, so every observation, intent schema, and
    provider-surface rule is enforced by the same code the benchmark uses.  A
    malformed request -- a non-object observation, a duplicate key, a non-finite
    number, an unknown root field, or an internal address -- fails closed here
    rather than reaching the policy.
    """

    if not isinstance(user_prompt, str):
        raise ScriptedBusinessDecisionError("business prompt must be text")
    start = user_prompt.find("{")
    if start < 0:
        raise ScriptedBusinessDecisionError("business prompt has no request object")
    try:
        value = json.loads(
            user_prompt[start:],
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except ScriptedBusinessDecisionError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ScriptedBusinessDecisionError(
            "business prompt request is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ScriptedBusinessDecisionError("business prompt request must be one object")
    if value.get("schema_version") != LLM_DECISION_REQUEST_SCHEMA_V1:
        raise ScriptedBusinessDecisionError("business prompt has the wrong request schema")
    unknown = set(value) - _ALLOWED_REQUEST_KEYS
    if unknown:
        raise ScriptedBusinessDecisionError("business prompt has unknown request fields")
    if value.get("response_contract") != _EXPECTED_RESPONSE_CONTRACT:
        raise ScriptedBusinessDecisionError("business prompt response contract is invalid")

    # Strict scalar types -- never str()-coerce, or a non-string goal carrying an
    # internal address would be stringified into a "valid" request the policy sees.
    role = value.get("role")
    goal = value.get("goal")
    phase = value.get("phase")
    if not isinstance(role, str) or not isinstance(goal, str) or not isinstance(phase, str):
        raise ScriptedBusinessDecisionError("business request role/goal/phase must be strings")

    raw_intents = value.get("allowed_intents")
    if not isinstance(raw_intents, list) or not raw_intents:
        raise ScriptedBusinessDecisionError("business prompt has no valid allowed intents")
    raw_observations = value.get("observations", [])
    if not isinstance(raw_observations, list):
        raise ScriptedBusinessDecisionError("business prompt observations must be a list")

    try:
        specs = tuple(_reconstruct_intent_spec(row) for row in raw_intents)
        request = LLMDecisionRequestV1(
            decision_id=decision_id or "reconstructed-public-request",
            role=role,
            goal=goal,
            phase=phase,
            observations=tuple(raw_observations),
            allowed_intents=specs,
        )
        # Authoritative provider-surface validation BEFORE any policy runs: this
        # rejects an internal service address or action namespace embedded in the
        # goal, phase, or an observation string, so a policy never sees one.
        request.to_public_dict()
    except BusinessDecisionContractError as exc:
        raise ScriptedBusinessDecisionError(
            "business prompt failed authoritative request validation"
        ) from exc

    if request.role != expected_role:
        raise ScriptedBusinessDecisionError(
            "scripted channel received a request for a different actor role"
        )
    return request


_ALLOWED_INTENT_KEYS = frozenset({"intent", "description", "category", "parameters"})


def _reconstruct_intent_spec(row: Mapping[str, Any]) -> BusinessIntentSpec:
    """Rebuild one intent spec from the public projection, strictly typed.

    Unknown intent fields and non-string/non-object types are rejected (never
    coerced).  The Agent-private ``source_name`` is filled with a fixed
    placeholder that never leaves this channel.
    """

    if not isinstance(row, Mapping):
        raise ScriptedBusinessDecisionError("business intent row must be an object")
    if set(row) - _ALLOWED_INTENT_KEYS:
        raise ScriptedBusinessDecisionError("business intent row has unknown fields")
    intent = row.get("intent")
    description = row.get("description")
    category = row.get("category")
    parameters = row.get("parameters")
    if (
        not isinstance(intent, str)
        or not isinstance(description, str)
        or not isinstance(category, str)
        or not isinstance(parameters, Mapping)
    ):
        raise ScriptedBusinessDecisionError("business intent row has invalid field types")
    return BusinessIntentSpec(
        intent=intent,
        description=description,
        parameters=copy.deepcopy(dict(parameters)),
        category=category,
        source_name=_RECONSTRUCTED_SOURCE_NAME,
    )


# A factory is single-use: it may be claimed for exactly one Episode, then it is
# sealed forever.  This makes cross-episode decision-log accumulation for the
# same actor id structurally impossible.
_FACTORY_NEW = "new"
_FACTORY_CLAIMED = "claimed"
_FACTORY_SEALED = "sealed"


class ScriptedChannelFactoryV1:
    """Create exactly one scripted channel per actor, for exactly one Episode.

    Guarantees ``agent_id -> one dedicated channel instance``: distinct actors
    never share a channel, and re-requesting the same actor returns the same
    bound instance (a role conflict for a known id fails closed), so decision
    logs can never cross between actors.  Suitable as the ``channels(agent_id,
    role)`` factory an ``Episode``/``EpisodeBatch`` calls per registered agent.

    Lifecycle ``new -> claimed -> sealed``.  Channels are served only while the
    factory is *claimed* for one Episode (via :meth:`claim_for_episode`); a
    second claim fails closed, and the factory cannot serve channels again once
    sealed.  A run therefore cannot accumulate a second episode's decisions onto
    the same actor ids -- callers must build a fresh factory per Episode.  The
    read-only accessors (:meth:`channel_for`, :meth:`actor_ids`,
    :attr:`provider_calls`) remain available in every state so a report can be
    assembled after the factory is sealed.
    """

    def __init__(
        self,
        *,
        policies: Mapping[str, ScriptedBusinessDecisionPolicyV1],
        actor_policies: Mapping[str, ScriptedBusinessDecisionPolicyV1] | None = None,
    ) -> None:
        actor_bindings = dict(actor_policies or {})
        if not policies and not actor_bindings:
            raise ValueError(
                "scripted channel factory requires a role or actor policy"
            )
        for role in policies:
            if role not in _ROLE_VALUES:
                raise ValueError(f"scripted channel factory role must be buyer/merchant: {role!r}")
        for actor_id, policy in actor_bindings.items():
            if (
                not isinstance(actor_id, str)
                or actor_id.split(":", 1)[0] not in _ROLE_VALUES
                or not callable(policy)
            ):
                raise ValueError(
                    "scripted actor policy requires a namespaced buyer/merchant id "
                    "and a callable policy"
                )
        self._policies = dict(policies)
        self._actor_policies = actor_bindings
        self._channels: dict[str, ScriptedBusinessDecisionChannelV1] = {}
        self._state = _FACTORY_NEW
        self._episode_id: str | None = None

    @property
    def lifecycle_state(self) -> str:
        """Return the factory lifecycle state: ``new``/``claimed``/``sealed``."""

        return self._state

    @property
    def claimed_episode_id(self) -> str | None:
        """Return the episode id this factory was claimed for, if any."""

        return self._episode_id

    @contextmanager
    def claim_for_episode(self, episode_id: str) -> Iterator["ScriptedChannelFactoryV1"]:
        """Claim this factory for exactly one Episode, then seal it on exit.

        A factory may be claimed only once: a second claim (even after the first
        episode finished or raised) fails closed, so the same actor ids can never
        collect decisions from two episodes.  Channels are servable only inside
        the claim.
        """

        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("scripted channel factory claim requires a non-empty episode id")
        if self._state != _FACTORY_NEW:
            raise ValueError(
                f"scripted channel factory is already used (state={self._state!r}); "
                "build a fresh factory per episode"
            )
        self._state = _FACTORY_CLAIMED
        self._episode_id = episode_id
        try:
            yield self
        finally:
            self._state = _FACTORY_SEALED

    def __call__(self, agent_id: str, role: str) -> ScriptedBusinessDecisionChannelV1:
        if self._state != _FACTORY_CLAIMED:
            raise ValueError(
                "scripted channel factory is not claimed for an episode "
                f"(state={self._state!r}); use claim_for_episode() first"
            )
        actor_policy = self._actor_policies.get(agent_id)
        if actor_policy is None and role not in self._policies:
            raise ValueError(f"scripted channel factory has no policy for role {role!r}")
        if agent_id in self._actor_policies and agent_id.split(":", 1)[0] != role:
            raise ValueError(
                f"actor-specific policy {agent_id!r} was requested with role {role!r}"
            )
        existing = self._channels.get(agent_id)
        if existing is not None:
            if existing.role != role:
                raise ValueError(
                    f"actor {agent_id!r} was already bound to role {existing.role!r}"
                )
            return existing
        channel = ScriptedBusinessDecisionChannelV1(
            actor_id=agent_id,
            role=role,
            policy=actor_policy or self._policies[role],
        )
        self._channels[agent_id] = channel
        return channel

    def channel_for(self, agent_id: str) -> ScriptedBusinessDecisionChannelV1:
        """Return the channel bound to ``agent_id`` (KeyError if never created)."""

        return self._channels[agent_id]

    @property
    def provider_calls(self) -> int:
        """Total provider calls across every created channel -- always ``0``."""

        return sum(channel.provider_calls for channel in self._channels.values())

    def actor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._channels))


def _freeze_public(value: Any) -> Any:
    """Return an immutable deep copy so a policy cannot mutate its context view."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_public(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_public(item) for item in value)
    return value


def _canonical_decision_text(decision: LLMBusinessDecisionV1) -> str:
    return json.dumps(
        decision.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "ScriptedBusinessDecisionChannelV1",
    "ScriptedBusinessDecisionError",
    "ScriptedBusinessDecisionPolicyV1",
    "ScriptedChannelFactoryV1",
    "ScriptedDecisionContextV1",
    "ScriptedDecisionRecordV1",
]
