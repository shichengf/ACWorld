"""Trusted causal context resolver for actor-authored evidence reports.

The resolver consumes envelopes *after* Runtime validation and in the exact
order in which Runtime audited them.  It snapshots each envelope into canonical
JSON, indexes it by ``msg_id``, and treats ``in_reply_to`` as a causal edge.

For actor reports addressed to ``runtime:evidence``, bindings are derived only
from the report envelope, its observed causal chain, and an optional immutable
context registered by episode orchestration.  Report payload values are never
consulted for actor, principal, task, mandate, or episode context identity.

This module is intentionally passive resolver core.  It does not register VCP
actions, edit Runtime routing, write evidence artifacts, or integrate itself
with episode orchestration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterable, Mapping, cast

from protocol.actor_result import (
    COMMERCE_SUBMIT_DECISION_RECORD,
    DELEGATE_REPORT_RESULT,
    ActorResultAction,
    ActorResultBinding,
)
from protocol.envelope import Envelope, to_json
from protocol.errors import SchemaError


EVIDENCE_ENDPOINT = "runtime:evidence"
DELEGATION_ROOT_ACTION = "delegate.create_purchase_mandate"
OWNER_DIRECTIVE_ROOT_ACTION = "commerce.send_message"
DEFAULT_ROOT_ACTION_KINDS = frozenset(
    {DELEGATION_ROOT_ACTION, OWNER_DIRECTIVE_ROOT_ACTION}
)
DEFAULT_REPORT_ACTION_KINDS = frozenset(
    {DELEGATE_REPORT_RESULT, COMMERCE_SUBMIT_DECISION_RECORD}
)
_CONTEXT_FIELDS = ("task_id", "mandate_id", "context_id")
_MISSING = object()
_PLATFORM_CAUSAL_HANDOFFS: Mapping[str, frozenset[str]] = {
    "commerce.submit_review": frozenset({"platform.aggregate_reviews"}),
    "platform.ingest_review_observation": frozenset(
        {"platform.aggregate_reviews"}
    ),
    "platform.ingest_market_observation": frozenset(
        {"platform.resolve_governance_case"}
    ),
    "platform.resolve_governance_case": frozenset(
        {"platform.create_remediation_plan"}
    ),
    "commerce.complete_remediation_step": frozenset(
        {"platform.verify_remediation_step"}
    ),
    "platform.verify_remediation_step": frozenset(
        {"platform.apply_governance_reputation"}
    ),
}


class ActorContextError(SchemaError):
    """Base failure for trusted actor-context resolution."""


class ActorContextRegistrationError(ActorContextError):
    """An injected immutable context is malformed, ambiguous, or out of scope."""


class DuplicateEnvelopeConflict(ActorContextError):
    """One ``msg_id`` was reused with different canonical envelope bytes."""


class MissingCausalParent(ActorContextError):
    """An envelope references a parent that was not observed earlier."""


class CausalCycleError(ActorContextError):
    """The observed ``in_reply_to`` graph contains a causal cycle."""


class CausalLinkError(ActorContextError):
    """A child was not authored by the recipient of its claimed parent."""


class ActorContextBindingError(ActorContextError):
    """A report chain failed root, actor, principal, episode, or action binding."""


@dataclass(frozen=True, slots=True)
class RegisteredActorContext:
    """Immutable context injected by trusted episode orchestration.

    A registration is keyed by the exact root ``msg_id`` and binds the root
    sender/recipient/action before any actor report exists.  This is required
    for generic owner directives whose payload does not carry a complete
    delegation context.
    """

    root_msg_id: str
    root_action_kind: str
    actor_id: str
    principal_id: str
    task_id: str
    mandate_id: str
    context_id: str
    report_action_kinds: tuple[ActorResultAction, ...] = (
        DELEGATE_REPORT_RESULT,
        COMMERCE_SUBMIT_DECISION_RECORD,
    )

    def __post_init__(self) -> None:
        for name in (
            "root_msg_id",
            "root_action_kind",
            "actor_id",
            "principal_id",
            "task_id",
            "mandate_id",
            "context_id",
        ):
            _require_text(getattr(self, name), name, ActorContextRegistrationError)
        if not isinstance(self.report_action_kinds, tuple) or not self.report_action_kinds:
            raise ActorContextRegistrationError(
                "report_action_kinds must be a non-empty tuple"
            )
        _require_unique(self.report_action_kinds, "report action")
        unknown = set(self.report_action_kinds) - DEFAULT_REPORT_ACTION_KINDS
        if unknown:
            raise ActorContextRegistrationError(
                f"unsupported report action scope: {sorted(unknown)!r}"
            )


@dataclass(frozen=True, slots=True)
class _ObservedEnvelope:
    envelope: Envelope
    canonical_json: str
    canonical_sha256: str
    sequence: int


@dataclass(frozen=True, slots=True)
class _TrustedRootContext:
    root_msg_id: str
    root_action_kind: str
    actor_id: str
    principal_id: str
    task_id: str
    mandate_id: str
    context_id: str
    report_action_kinds: frozenset[str]


class ActorContextResolver:
    """Maintain a canonical causal graph and resolve trusted report bindings."""

    def __init__(
        self,
        *,
        episode_context_id: str,
        actor_contexts: Iterable[RegisteredActorContext] = (),
        root_action_kinds: Iterable[str] = DEFAULT_ROOT_ACTION_KINDS,
        report_action_kinds: Iterable[str] = DEFAULT_REPORT_ACTION_KINDS,
    ) -> None:
        _require_text(
            episode_context_id,
            "episode_context_id",
            ActorContextRegistrationError,
        )
        roots = frozenset(root_action_kinds)
        reports = frozenset(report_action_kinds)
        _validate_action_set(roots, "root_action_kinds")
        _validate_action_set(reports, "report_action_kinds")
        unknown_reports = reports - DEFAULT_REPORT_ACTION_KINDS
        if unknown_reports:
            raise ActorContextRegistrationError(
                f"unsupported report actions: {sorted(unknown_reports)!r}"
            )

        registrations: dict[str, RegisteredActorContext] = {}
        for context in actor_contexts:
            if not isinstance(context, RegisteredActorContext):
                raise ActorContextRegistrationError(
                    "actor_contexts must contain RegisteredActorContext values"
                )
            if context.context_id != episode_context_id:
                raise ActorContextRegistrationError(
                    "registered actor context belongs to another episode"
                )
            if context.root_action_kind not in roots:
                raise ActorContextRegistrationError(
                    "registered root action is outside resolver scope"
                )
            if not set(context.report_action_kinds).issubset(reports):
                raise ActorContextRegistrationError(
                    "registered report action is outside resolver scope"
                )
            existing = registrations.get(context.root_msg_id)
            if existing is not None and existing != context:
                raise ActorContextRegistrationError(
                    "root_msg_id has conflicting immutable actor contexts"
                )
            registrations[context.root_msg_id] = context

        self._episode_context_id = episode_context_id
        self._root_action_kinds = roots
        self._report_action_kinds = reports
        self._registrations: Mapping[str, RegisteredActorContext] = MappingProxyType(
            registrations
        )
        self._observed: dict[str, _ObservedEnvelope] = {}
        self._bindings: dict[str, ActorResultBinding] = {}
        self._lock = RLock()

    @property
    def episode_context_id(self) -> str:
        return self._episode_context_id

    @property
    def observed_count(self) -> int:
        with self._lock:
            return len(self._observed)

    def has_message(self, msg_id: str) -> bool:
        with self._lock:
            return msg_id in self._observed

    def canonical_envelope_sha256(self, msg_id: str) -> str:
        with self._lock:
            try:
                return self._observed[msg_id].canonical_sha256
            except KeyError as exc:
                raise MissingCausalParent(f"unknown envelope {msg_id!r}") from exc

    def validate_report_candidate(self, envelope: Envelope) -> ActorResultBinding:
        """Purely preview one actor report against the observed causal graph.

        Runtime invokes this while an Agent turn is still open, before the
        candidate is audited.  A model-selected report whose causal chain has
        no trusted actor context therefore becomes a Tracker-bound protocol
        failure instead of aborting the episode after the turn was recorded as
        a successful emission.  The preview never adds a graph node or caches
        a binding; :meth:`ingest` remains the only live acceptance path.
        """

        snapshot, canonical = _snapshot_envelope(envelope)
        with self._lock:
            existing = self._observed.get(snapshot.msg_id)
            if existing is not None:
                if existing.canonical_json != canonical:
                    raise DuplicateEnvelopeConflict(
                        f"msg_id {snapshot.msg_id!r} has different canonical bytes"
                    )
                snapshot = existing.envelope
            else:
                self._validate_new_causal_node(snapshot)
            return self._binding_for_report(snapshot)

    def observe(self, envelope: Envelope) -> bool:
        """Snapshot one ordered audited envelope.

        Returns ``True`` for a new graph node and ``False`` for an exact
        canonical duplicate.  Reusing a ``msg_id`` for any changed byte is a
        hard conflict.  A non-root parent must already be present because the
        Runtime audit stream is ordered.
        """

        snapshot, canonical = _snapshot_envelope(envelope)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self._lock:
            existing = self._observed.get(snapshot.msg_id)
            if existing is not None:
                if existing.canonical_json != canonical:
                    raise DuplicateEnvelopeConflict(
                        f"msg_id {snapshot.msg_id!r} has different canonical bytes"
                    )
                return False

            self._validate_new_causal_node(snapshot)

            registration = self._registrations.get(snapshot.msg_id)
            if registration is not None:
                self._validate_registered_root(snapshot, registration)

            self._observed[snapshot.msg_id] = _ObservedEnvelope(
                envelope=snapshot,
                canonical_json=canonical,
                canonical_sha256=digest,
                sequence=len(self._observed),
            )
            return True

    def ingest(self, envelope: Envelope) -> ActorResultBinding | None:
        """Observe an envelope and resolve it when it is evidence-related."""

        self.observe(envelope)
        kind = _action_kind(envelope)
        if kind in DEFAULT_REPORT_ACTION_KINDS or envelope.to == EVIDENCE_ENDPOINT:
            return self.resolve(envelope.msg_id)
        return None

    def resolve(self, request: str | Envelope) -> ActorResultBinding:
        """Resolve one already-observed report without reading its payload."""

        with self._lock:
            msg_id = request if isinstance(request, str) else request.msg_id
            try:
                node = self._observed[msg_id]
            except KeyError as exc:
                raise ActorContextBindingError(
                    f"report envelope {msg_id!r} was not observed"
                ) from exc
            if isinstance(request, Envelope):
                _, canonical = _snapshot_envelope(request)
                if canonical != node.canonical_json:
                    raise DuplicateEnvelopeConflict(
                        f"report msg_id {msg_id!r} does not match observed bytes"
                    )
            cached = self._bindings.get(msg_id)
            if cached is not None:
                return cached

            binding = self._binding_for_report(node.envelope)
            self._bindings[msg_id] = binding
            return binding

    def _validate_new_causal_node(self, snapshot: Envelope) -> None:
        parent_id = snapshot.in_reply_to
        if parent_id == snapshot.msg_id:
            raise CausalCycleError("envelope cannot reply to itself")
        if parent_id is None:
            return
        parent = self._observed.get(parent_id)
        if parent is None:
            raise MissingCausalParent(
                f"envelope {snapshot.msg_id!r} references missing parent {parent_id!r}"
            )
        if not _causal_child_sender_is_valid(parent.envelope, snapshot):
            raise CausalLinkError(
                "causal child sender is not the claimed parent's recipient"
            )

    def _binding_for_report(self, envelope: Envelope) -> ActorResultBinding:
        action_kind = _action_kind(envelope)
        if envelope.to != EVIDENCE_ENDPOINT:
            raise ActorContextBindingError(
                "actor result must be addressed to runtime:evidence"
            )
        if action_kind not in self._report_action_kinds:
            raise ActorContextBindingError(
                f"unsupported evidence action scope {action_kind!r}"
            )
        if envelope.in_reply_to is None:
            raise ActorContextBindingError(
                "actor result must reply to an observed causal parent"
            )

        chain = self._causal_chain(envelope)
        root = chain[-1]
        trusted = self._trusted_root_context(root)
        if envelope.from_ != trusted.actor_id:
            raise ActorContextBindingError(
                "report actor does not match causal root recipient"
            )
        if action_kind not in trusted.report_action_kinds:
            raise ActorContextBindingError(
                "report action is outside the root's registered scope"
            )
        return ActorResultBinding(
            action_kind=cast(ActorResultAction, action_kind),
            actor_id=envelope.from_,
            principal_id=trusted.principal_id,
            task_id=trusted.task_id,
            mandate_id=trusted.mandate_id,
            context_id=trusted.context_id,
            in_reply_to=envelope.in_reply_to,
            idempotency_key=envelope.idempotency_key,
        )

    def _causal_chain(self, report: Envelope) -> tuple[Envelope, ...]:
        parent_id = report.in_reply_to
        if parent_id is None:  # guarded by resolve; keeps helper total
            raise ActorContextBindingError("report has no causal parent")
        chain: list[Envelope] = []
        seen = {report.msg_id}
        child = report
        while True:
            if parent_id in seen:
                raise CausalCycleError("cycle detected in actor report causal chain")
            seen.add(parent_id)
            try:
                parent = self._observed[parent_id].envelope
            except KeyError as exc:
                raise MissingCausalParent(
                    f"causal chain references missing parent {parent_id!r}"
                ) from exc
            if child.in_reply_to != parent.msg_id:
                raise CausalLinkError("causal edge changed after observation")
            if not _causal_child_sender_is_valid(parent, child):
                raise CausalLinkError(
                    "causal child sender is not the claimed parent's recipient"
                )
            chain.append(parent)
            if parent.in_reply_to is None:
                return tuple(chain)
            child = parent
            parent_id = parent.in_reply_to

    def _trusted_root_context(self, root: Envelope) -> _TrustedRootContext:
        if root.in_reply_to is not None:
            raise ActorContextBindingError("causal root unexpectedly has a parent")
        kind = _action_kind(root)
        if kind not in self._root_action_kinds:
            raise ActorContextBindingError(
                f"causal root action {kind!r} is outside trusted scope"
            )
        registration = self._registrations.get(root.msg_id)
        if registration is None and kind != DELEGATION_ROOT_ACTION:
            raise ActorContextBindingError(
                "generic owner directive requires immutable orchestration context"
            )
        return self._context_from_root(root, registration)

    def _validate_registered_root(
        self,
        root: Envelope,
        registration: RegisteredActorContext,
    ) -> None:
        if root.in_reply_to is not None:
            raise ActorContextBindingError("registered root must not have a parent")
        if _action_kind(root) != registration.root_action_kind:
            raise ActorContextBindingError("registered root action binding mismatch")
        if root.to != registration.actor_id:
            raise ActorContextBindingError("registered root actor binding mismatch")
        if root.from_ != registration.principal_id:
            raise ActorContextBindingError("registered root principal binding mismatch")
        self._context_from_root(root, registration)

    def _context_from_root(
        self,
        root: Envelope,
        registration: RegisteredActorContext | None,
    ) -> _TrustedRootContext:
        context_values = _root_context_values(root)
        if registration is None:
            missing = [name for name in _CONTEXT_FIELDS if name not in context_values]
            if missing:
                raise ActorContextBindingError(
                    "root delegation has incomplete trusted context: "
                    + ", ".join(missing)
                )
            task_id = context_values["task_id"]
            mandate_id = context_values["mandate_id"]
            context_id = context_values["context_id"]
            report_actions = self._report_action_kinds
        else:
            expected = {
                "task_id": registration.task_id,
                "mandate_id": registration.mandate_id,
                "context_id": registration.context_id,
            }
            mismatches = [
                name
                for name, actual in context_values.items()
                if actual != expected[name]
            ]
            if mismatches:
                raise ActorContextBindingError(
                    "root payload conflicts with immutable context: "
                    + ", ".join(mismatches)
                )
            task_id = registration.task_id
            mandate_id = registration.mandate_id
            context_id = registration.context_id
            report_actions = frozenset(registration.report_action_kinds)

        if context_id != self._episode_context_id:
            raise ActorContextBindingError("causal root belongs to another episode")
        actor_id = root.to
        principal_id = root.from_
        if registration is not None:
            if actor_id != registration.actor_id:
                raise ActorContextBindingError("root actor binding mismatch")
            if principal_id != registration.principal_id:
                raise ActorContextBindingError("root principal binding mismatch")
            if _action_kind(root) != registration.root_action_kind:
                raise ActorContextBindingError("root action binding mismatch")
        return _TrustedRootContext(
            root_msg_id=root.msg_id,
            root_action_kind=_action_kind(root),
            actor_id=actor_id,
            principal_id=principal_id,
            task_id=task_id,
            mandate_id=mandate_id,
            context_id=context_id,
            report_action_kinds=frozenset(report_actions),
        )


def _snapshot_envelope(envelope: Envelope) -> tuple[Envelope, str]:
    if not isinstance(envelope, Envelope):
        raise ActorContextError("resolver accepts only Envelope values")
    _validate_envelope_shape(envelope)
    try:
        normalized = json.loads(
            to_json(envelope),
            parse_constant=_reject_nonfinite_constant,
        )
        canonical = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        # ``protocol.envelope.to_json`` rejects NaN/Infinity before the
        # parse_constant hook can run.  Keep this public error stable across
        # either rejection site and do not echo model-controlled payload text.
        raise ActorContextError(
            "envelope contains non-finite or non-canonical JSON"
        ) from exc
    if not isinstance(normalized, dict):  # pragma: no cover - to_json guarantees
        raise ActorContextError("canonical envelope must be an object")
    data = dict(normalized)
    data["from_"] = data.pop("from")
    snapshot = Envelope(**data)
    _validate_envelope_shape(snapshot)
    return snapshot, canonical


def _validate_envelope_shape(envelope: Envelope) -> None:
    for name in ("msg_id", "from_", "to", "idempotency_key"):
        _require_text(getattr(envelope, name), name, ActorContextError)
    if envelope.in_reply_to is not None:
        _require_text(envelope.in_reply_to, "in_reply_to", ActorContextError)
    if not isinstance(envelope.action, Mapping):
        raise ActorContextError("envelope action must be an object")
    _require_text(envelope.action.get("kind"), "action.kind", ActorContextError)
    if "payload" not in envelope.action:
        raise ActorContextError("envelope action requires payload")


def _root_context_values(root: Envelope) -> dict[str, str]:
    payload = root.action.get("payload")
    if not isinstance(payload, Mapping):
        raise ActorContextBindingError("root delegation payload must be an object")
    nested = payload.get("delegation_context", _MISSING)
    direct = {name: payload[name] for name in _CONTEXT_FIELDS if name in payload}
    if nested is not _MISSING and direct:
        raise ActorContextBindingError(
            "root delegation context cannot use nested and direct forms together"
        )
    raw: Mapping[str, Any]
    if nested is not _MISSING:
        if not isinstance(nested, Mapping):
            raise ActorContextBindingError("delegation_context must be an object")
        keys = set(nested)
        expected = set(_CONTEXT_FIELDS)
        if keys != expected:
            missing = sorted(expected - keys)
            unknown = sorted(keys - expected)
            raise ActorContextBindingError(
                f"delegation_context has invalid fields: missing={missing!r}, unknown={unknown!r}"
            )
        raw = nested
    else:
        raw = direct
    result: dict[str, str] = {}
    for name, value in raw.items():
        _require_text(value, f"root {name}", ActorContextBindingError)
        result[name] = value
    return result


def _is_platform_causal_handoff(parent: Envelope, child: Envelope) -> bool:
    """Allow only registered stateless Platform orchestration edges.

    A Platform service may return an internal request authored by the exact
    ``platform:orchestrator`` identity.  The child still names the accepted
    request as its parent, but its sender is the orchestrator rather than the
    service endpoint.  This narrow action map preserves the ordinary
    recipient-authors-child rule for every other envelope.
    """

    if (
        not parent.to.startswith("platform:")
        or child.from_ != "platform:orchestrator"
        or not child.to.startswith("platform:")
    ):
        return False
    allowed = _PLATFORM_CAUSAL_HANDOFFS.get(_action_kind(parent), frozenset())
    return _action_kind(child) in allowed


def _causal_child_sender_is_valid(parent: Envelope, child: Envelope) -> bool:
    """Apply the same exact causal-edge rule during observe and resolve."""

    return child.from_ == parent.to or _is_platform_causal_handoff(parent, child)


def _action_kind(envelope: Envelope) -> str:
    value = envelope.action.get("kind")
    if not isinstance(value, str) or not value:
        raise ActorContextError("envelope action.kind must be a non-empty string")
    return value


def _validate_action_set(values: frozenset[str], label: str) -> None:
    if not values:
        raise ActorContextRegistrationError(f"{label} must not be empty")
    for value in values:
        _require_text(value, f"{label} item", ActorContextRegistrationError)


def _require_text(value: Any, label: str, error_type: type[ActorContextError]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string")


def _require_unique(values: Iterable[Any], label: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ActorContextRegistrationError(f"duplicate {label} {value!r}")
        seen.add(value)


def _reject_nonfinite_constant(value: str) -> None:
    raise ActorContextError(f"envelope contains non-finite JSON number {value}")


__all__ = [
    "DEFAULT_REPORT_ACTION_KINDS",
    "DEFAULT_ROOT_ACTION_KINDS",
    "DELEGATION_ROOT_ACTION",
    "EVIDENCE_ENDPOINT",
    "OWNER_DIRECTIVE_ROOT_ACTION",
    "ActorContextBindingError",
    "ActorContextError",
    "ActorContextRegistrationError",
    "ActorContextResolver",
    "CausalCycleError",
    "CausalLinkError",
    "DuplicateEnvelopeConflict",
    "MissingCausalParent",
    "RegisteredActorContext",
]
