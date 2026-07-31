"""Value-free public phase contracts for native CommerceWorld Agents.

The contract in this module is scenario-owned and spawn-frozen.  It limits the
business functions that an Agent may see on one inbound turn.  It is not a
scoring contract and it never consumes an oracle, an ideal trajectory, or an
expected answer.

``allowed_routes.destination`` names the static destination from the semantic
tool registry.  Dynamic values such as ``@inbound_sender`` and
``@argument:recipient_id`` are therefore valid.  The existing semantic
compiler resolves those templates later and Runtime remains the authority for
the final recipient.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from agents.types import ALLOWED_WORLD_TOOLS
from protocol.envelope import Envelope


PUBLIC_TASK_EXECUTION_SCHEMA_VERSION = "cwe.public-task-execution.v1"
WORLD_READ_POLICIES = frozenset({"deny", "skill_scoped"})
FINISH_POLICIES = frozenset({
    "forbid",
    "allow_wait",
    "framework_terminal",
    "framework_continue",
})
ACTOR_ROLES = frozenset({"buyer", "merchant"})
ACTOR_REPORT_ROUTES = frozenset({
    ("commerce.submit_decision_record", "runtime:evidence"),
    ("delegate.report_result", "runtime:evidence"),
})

_FORBIDDEN_AUTHORITY_KEYS = frozenset({
    "success_oracle",
    "ground_truth",
    "ideal_trajectory",
    "expected_answer",
})


class PublicTaskExecutionContractError(ValueError):
    """A public task contract is malformed or cannot select one phase."""


@dataclass(frozen=True, slots=True)
class PublicTaskRoute:
    """One semantic registry route allowed during a public phase."""

    action_kind: str
    destination: str


@dataclass(frozen=True, slots=True)
class PublicTaskPhaseMatch:
    """Public inbound facts used to select a phase."""

    actor_roles: frozenset[str]
    inbound_action_kinds: frozenset[str]
    inbound_sender_roles: frozenset[str]
    inbound_senders: frozenset[str]
    payload_equals: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PublicTaskResultBinding:
    """One high-level actor result that the Agent binds to Runtime evidence."""

    action_kind: str
    destination: str
    result_format: str
    submission_kind: str | None
    payload_schema: Mapping[str, Any]

    def to_semantic_contract(self) -> dict[str, Any]:
        """Return the existing semantic result compiler input."""

        return {
            "active": True,
            "action_kind": self.action_kind,
            "endpoint": self.destination,
            "result_format": self.result_format,
            "submission_kind": self.submission_kind,
            "payload_schema": copy.deepcopy(dict(self.payload_schema)),
        }


@dataclass(frozen=True, slots=True)
class PublicTaskAcceptedReferenceGate:
    """Delay a framework continuation until all accepted public refs exist."""

    reference_field: str
    required_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicTaskWorldReadRequirement:
    """One exact World-read prerequisite for a model-selectable action."""

    tool: str
    argument_field: str | None
    required_values: tuple[str, ...]
    minimum_successes: int


@dataclass(frozen=True, slots=True)
class PublicTaskPreActionReadGate:
    """Hide one action until its declared World reads have completed."""

    action_kind: str
    requirements: tuple[PublicTaskWorldReadRequirement, ...]
    exclusive_until_complete: bool


@dataclass(frozen=True, slots=True)
class PublicTaskFrameworkContinuation:
    """One deterministic Agent-owned continuation before model inference."""

    action_kind: str
    destination: str
    payload: Mapping[str, Any]
    accepted_reference_gate: PublicTaskAcceptedReferenceGate | None = None


@dataclass(frozen=True, slots=True)
class ResolvedPublicTaskPhase:
    """A validated phase selected for one concrete actor turn."""

    phase_id: str
    match: PublicTaskPhaseMatch
    allowed_routes: tuple[PublicTaskRoute, ...]
    world_reads: str
    finish: str
    result: PublicTaskResultBinding | None
    pre_action_read_gate: PublicTaskPreActionReadGate | None
    framework_continuation: PublicTaskFrameworkContinuation | None

    @property
    def semantic_routes(self) -> tuple[PublicTaskRoute, ...]:
        """Return all action routes, including the optional result route."""

        routes = list(self.allowed_routes)
        if self.result is not None:
            result_route = PublicTaskRoute(
                action_kind=self.result.action_kind,
                destination=self.result.destination,
            )
            if result_route not in routes:
                routes.append(result_route)
        return tuple(routes)

    def to_actor_context(self) -> dict[str, Any]:
        """Project this phase into framework-owned semantic Agent context."""

        output: dict[str, Any] = {
            "task_phase_id": self.phase_id,
            "task_allowed_routes": tuple(
                (route.action_kind, route.destination)
                for route in self.semantic_routes
            ),
            "task_allowed_action_kinds": tuple(
                dict.fromkeys(route.action_kind for route in self.semantic_routes)
            ),
            "task_world_read_policy": self.world_reads,
            "task_world_reads_allowed": self.world_reads == "skill_scoped",
            "task_finish_policy": self.finish,
        }
        if self.result is not None:
            output["task_result_contract"] = self.result.to_semantic_contract()
        if self.pre_action_read_gate is not None:
            output["task_pre_action_read_gate"] = {
                "action_kind": self.pre_action_read_gate.action_kind,
                "exclusive_until_complete": (
                    self.pre_action_read_gate.exclusive_until_complete
                ),
                "requirements": [
                    {
                        "tool": row.tool,
                        "argument_field": row.argument_field,
                        "required_values": list(row.required_values),
                        "minimum_successes": row.minimum_successes,
                    }
                    for row in self.pre_action_read_gate.requirements
                ],
            }
        if self.framework_continuation is not None:
            continuation: dict[str, Any] = {
                "action_kind": self.framework_continuation.action_kind,
                "destination": self.framework_continuation.destination,
                "payload": copy.deepcopy(dict(self.framework_continuation.payload)),
            }
            gate = self.framework_continuation.accepted_reference_gate
            if gate is not None:
                continuation["accepted_reference_gate"] = {
                    "reference_field": gate.reference_field,
                    "required_values": list(gate.required_values),
                }
            output["task_framework_continuation"] = continuation
        return output


@dataclass(frozen=True, slots=True)
class PublicTaskExecutionContract:
    """One validated collection of mutually exclusive public phases."""

    phases: tuple[ResolvedPublicTaskPhase, ...]


def validate_public_task_execution_contract(
    value: Any,
) -> PublicTaskExecutionContract:
    """Validate and normalize a ``cwe.public-task-execution.v1`` contract."""

    if not isinstance(value, Mapping):
        raise PublicTaskExecutionContractError(
            "public task execution contract must be an object"
        )
    _reject_hidden_authority(value)
    if value.get("schema_version") != PUBLIC_TASK_EXECUTION_SCHEMA_VERSION:
        raise PublicTaskExecutionContractError(
            "public task execution contract has an unsupported schema"
        )
    phases_value = value.get("phases")
    if not isinstance(phases_value, list) or not phases_value:
        raise PublicTaskExecutionContractError(
            "public task execution contract requires non-empty phases"
        )
    phases = tuple(_validate_phase(row) for row in phases_value)
    ids = [phase.phase_id for phase in phases]
    if len(ids) != len(set(ids)):
        raise PublicTaskExecutionContractError(
            "public task execution phase ids must be unique"
        )
    return PublicTaskExecutionContract(phases=phases)


def resolve_public_task_phase(
    contract: PublicTaskExecutionContract | Mapping[str, Any],
    *,
    actor_id: str,
    inbound: Envelope,
) -> ResolvedPublicTaskPhase:
    """Resolve exactly one phase from public actor and inbound facts.

    A present contract is fail-closed.  Neither a missing match nor an
    ambiguous match may fall back to the Agent's ambient action surface.
    """

    normalized = (
        contract
        if isinstance(contract, PublicTaskExecutionContract)
        else validate_public_task_execution_contract(contract)
    )
    actor_role = _actor_role(actor_id, label="actor id")
    inbound_kind = inbound.action.get("kind")
    payload = inbound.action.get("payload")
    if not isinstance(inbound_kind, str) or not inbound_kind.strip():
        raise PublicTaskExecutionContractError(
            "inbound envelope has no public action kind"
        )
    if not isinstance(payload, Mapping):
        raise PublicTaskExecutionContractError(
            "inbound envelope payload must be an object"
        )
    sender_role = _actor_role(inbound.from_, label="inbound sender")
    matches = [
        phase
        for phase in normalized.phases
        if actor_role in phase.match.actor_roles
        and inbound_kind in phase.match.inbound_action_kinds
        and (
            not phase.match.inbound_sender_roles
            or sender_role in phase.match.inbound_sender_roles
        )
        and (
            not phase.match.inbound_senders
            or inbound.from_ in phase.match.inbound_senders
        )
        and _payload_contains(payload, phase.match.payload_equals)
    ]
    if not matches:
        raise PublicTaskExecutionContractError(
            "public task execution contract matched no phase"
        )
    if len(matches) != 1:
        raise PublicTaskExecutionContractError(
            "public task execution contract matched multiple phases"
        )
    return matches[0]


def phase_actor_context(phase: ResolvedPublicTaskPhase) -> dict[str, Any]:
    """Public helper used by Agent and contract-focused audits."""

    return phase.to_actor_context()


def public_task_execution_contract_digest(value: Mapping[str, Any]) -> str:
    """Hash one validated public contract without retaining its content."""

    validate_public_task_execution_contract(value)
    normalized = _strict_json_copy(value, label="public task execution contract")
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def public_task_phase_ambiguities(
    contract: PublicTaskExecutionContract | Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return statically overlapping public phase pairs.

    This check uses only public match predicates.  It is intentionally
    independent of an oracle, reviewed trajectory, or expected task answer.
    A returned pair means that at least one syntactically possible inbound
    envelope could satisfy both phases, so the contract is not unambiguous by
    construction even if today's reference trajectory happens not to hit the
    overlap.
    """

    normalized = (
        contract
        if isinstance(contract, PublicTaskExecutionContract)
        else validate_public_task_execution_contract(contract)
    )
    overlaps: list[tuple[str, str]] = []
    for index, left in enumerate(normalized.phases):
        for right in normalized.phases[index + 1 :]:
            if _phase_matches_overlap(left.match, right.match):
                overlaps.append((left.phase_id, right.phase_id))
    return tuple(overlaps)


def _phase_matches_overlap(
    left: PublicTaskPhaseMatch,
    right: PublicTaskPhaseMatch,
) -> bool:
    return (
        bool(left.actor_roles & right.actor_roles)
        and bool(left.inbound_action_kinds & right.inbound_action_kinds)
        and _sender_matches_overlap(left, right)
        and _payload_matches_compatible(left.payload_equals, right.payload_equals)
    )


def _sender_matches_overlap(
    left: PublicTaskPhaseMatch,
    right: PublicTaskPhaseMatch,
) -> bool:
    """Return whether one public sender id can satisfy both predicates."""

    if left.inbound_senders and right.inbound_senders:
        candidates = left.inbound_senders & right.inbound_senders
    elif left.inbound_senders:
        candidates = left.inbound_senders
    elif right.inbound_senders:
        candidates = right.inbound_senders
    else:
        if left.inbound_sender_roles and right.inbound_sender_roles:
            return bool(left.inbound_sender_roles & right.inbound_sender_roles)
        return True
    return any(
        (
            not left.inbound_sender_roles
            or _actor_role(sender, label="inbound sender")
            in left.inbound_sender_roles
        )
        and (
            not right.inbound_sender_roles
            or _actor_role(sender, label="inbound sender")
            in right.inbound_sender_roles
        )
        for sender in candidates
    )


def _payload_matches_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    for key in left.keys() & right.keys():
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, Mapping) or isinstance(right_value, Mapping):
            if not isinstance(left_value, Mapping) or not isinstance(
                right_value,
                Mapping,
            ):
                return False
            if not _payload_matches_compatible(left_value, right_value):
                return False
        elif type(left_value) is not type(right_value) or left_value != right_value:
            return False
    return True


def _validate_phase(value: Any) -> ResolvedPublicTaskPhase:
    if not isinstance(value, Mapping):
        raise PublicTaskExecutionContractError("public task phase must be an object")
    phase_id = _nonempty_text(value.get("phase_id"), label="phase id")
    match = _validate_match(value.get("match"))
    allowed_value = value.get("allowed_routes")
    if not isinstance(allowed_value, list):
        raise PublicTaskExecutionContractError(
            f"public task phase {phase_id!r} allowed_routes must be a list"
        )
    allowed_routes = tuple(_validate_route(row) for row in allowed_value)
    if len(allowed_routes) != len(set(allowed_routes)):
        raise PublicTaskExecutionContractError(
            f"public task phase {phase_id!r} contains duplicate routes"
        )
    if any(
        (route.action_kind, route.destination) in ACTOR_REPORT_ROUTES
        for route in allowed_routes
    ):
        raise PublicTaskExecutionContractError(
            "actor report routes must be declared only through phase.result"
        )
    world_reads = value.get("world_reads")
    if world_reads not in WORLD_READ_POLICIES:
        raise PublicTaskExecutionContractError(
            f"public task phase {phase_id!r} has invalid world_reads policy"
        )
    finish = value.get("finish")
    if finish not in FINISH_POLICIES:
        raise PublicTaskExecutionContractError(
            f"public task phase {phase_id!r} has invalid finish policy"
        )
    result = _validate_result(value.get("result"))
    pre_action_read_gate = _validate_pre_action_read_gate(
        value.get("pre_action_read_gate"),
        allowed_routes=allowed_routes,
        world_reads=str(world_reads),
    )
    framework_continuation = _validate_framework_continuation(
        value.get("framework_continuation")
    )
    if finish == "framework_terminal" and (
        allowed_routes
        or result is not None
        or pre_action_read_gate is not None
        or framework_continuation is not None
        or world_reads != "deny"
    ):
        raise PublicTaskExecutionContractError(
            "framework-terminal phase cannot expose actions, results, or World reads"
        )
    if finish == "framework_continue" and (
        allowed_routes
        or result is not None
        or pre_action_read_gate is not None
        or framework_continuation is None
        or world_reads != "deny"
    ):
        raise PublicTaskExecutionContractError(
            "framework-continue phase requires exactly one deterministic continuation"
        )
    if (
        finish != "framework_continue"
        and framework_continuation is not None
        and not (
            finish == "allow_wait"
            and framework_continuation.accepted_reference_gate is not None
        )
    ):
        raise PublicTaskExecutionContractError(
            "framework_continuation requires framework_continue or a gated allow_wait"
        )
    return ResolvedPublicTaskPhase(
        phase_id=phase_id,
        match=match,
        allowed_routes=allowed_routes,
        world_reads=str(world_reads),
        finish=str(finish),
        result=result,
        pre_action_read_gate=pre_action_read_gate,
        framework_continuation=framework_continuation,
    )


def _validate_match(value: Any) -> PublicTaskPhaseMatch:
    if not isinstance(value, Mapping):
        raise PublicTaskExecutionContractError("public task phase match must be an object")
    actor_roles = _text_set(value.get("actor_roles"), label="actor_roles")
    if not actor_roles.issubset(ACTOR_ROLES):
        raise PublicTaskExecutionContractError(
            "public task phase actor_roles must contain only buyer or merchant"
        )
    action_kinds = _text_set(
        value.get("inbound_action_kinds"),
        label="inbound_action_kinds",
    )
    sender_roles = _text_set(
        value.get("inbound_sender_roles", []),
        label="inbound_sender_roles",
        allow_empty=True,
    )
    senders_value = value.get("inbound_senders", [])
    inbound_senders = _text_set(
        senders_value,
        label="inbound_senders",
        allow_empty=True,
    )
    if not sender_roles and not inbound_senders:
        raise PublicTaskExecutionContractError(
            "public task phase match requires sender roles or exact senders"
        )
    payload_equals = value.get("payload_equals", {})
    if not isinstance(payload_equals, Mapping):
        raise PublicTaskExecutionContractError(
            "public task phase payload_equals must be an object"
        )
    normalized_payload_equals = _strict_json_copy(
        payload_equals,
        label="payload_equals",
    )
    if not isinstance(normalized_payload_equals, dict):  # pragma: no cover
        raise AssertionError("validated payload_equals did not remain an object")
    return PublicTaskPhaseMatch(
        actor_roles=actor_roles,
        inbound_action_kinds=action_kinds,
        inbound_sender_roles=sender_roles,
        inbound_senders=inbound_senders,
        payload_equals=normalized_payload_equals,
    )


def _validate_route(value: Any) -> PublicTaskRoute:
    if not isinstance(value, Mapping):
        raise PublicTaskExecutionContractError("public task route must be an object")
    return PublicTaskRoute(
        action_kind=_nonempty_text(value.get("action_kind"), label="route action_kind"),
        destination=_nonempty_text(value.get("destination"), label="route destination"),
    )


def _validate_result(value: Any) -> PublicTaskResultBinding | None:
    if value is None:
        return None
    route = _validate_route(value)
    if (route.action_kind, route.destination) not in ACTOR_REPORT_ROUTES:
        raise PublicTaskExecutionContractError(
            "public task result must target the registered Runtime evidence route"
        )
    result_format = value.get("result_format", "named_submission")
    if result_format not in {"named_submission", "direct_details"}:
        raise PublicTaskExecutionContractError(
            "public task result has an unsupported result_format"
        )
    submission_kind = value.get("submission_kind")
    if result_format == "named_submission":
        submission_kind = _nonempty_text(
            submission_kind,
            label="result submission_kind",
        )
    elif submission_kind is not None:
        raise PublicTaskExecutionContractError(
            "direct-details public task result cannot name a submission_kind"
        )
    payload_schema = value.get("payload_schema")
    if (
        not isinstance(payload_schema, Mapping)
        or payload_schema.get("type") != "object"
        or not isinstance(payload_schema.get("properties"), Mapping)
    ):
        raise PublicTaskExecutionContractError(
            "public task result requires an object payload_schema"
        )
    normalized_payload_schema = _strict_json_copy(
        payload_schema,
        label="result payload_schema",
    )
    if not isinstance(normalized_payload_schema, dict):  # pragma: no cover
        raise AssertionError("validated payload_schema did not remain an object")
    return PublicTaskResultBinding(
        action_kind=route.action_kind,
        destination=route.destination,
        result_format=str(result_format),
        submission_kind=submission_kind,
        payload_schema=normalized_payload_schema,
    )


def _validate_pre_action_read_gate(
    value: Any,
    *,
    allowed_routes: tuple[PublicTaskRoute, ...],
    world_reads: str,
) -> PublicTaskPreActionReadGate | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PublicTaskExecutionContractError(
            "pre_action_read_gate must be an object"
        )
    action_kind = _nonempty_text(
        value.get("action_kind"),
        label="pre-action read gate action_kind",
    )
    if action_kind not in {row.action_kind for row in allowed_routes}:
        raise PublicTaskExecutionContractError(
            "pre-action read gate action is outside the phase routes"
        )
    if world_reads != "skill_scoped":
        raise PublicTaskExecutionContractError(
            "pre-action read gate requires skill-scoped World reads"
        )
    raw_requirements = value.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise PublicTaskExecutionContractError(
            "pre-action read gate requires a non-empty requirement list"
        )
    requirements: list[PublicTaskWorldReadRequirement] = []
    exclusive_until_complete = value.get("exclusive_until_complete", False)
    if not isinstance(exclusive_until_complete, bool):
        raise PublicTaskExecutionContractError(
            "pre-action read gate exclusive_until_complete must be a boolean"
        )
    identities: set[tuple[str, str | None]] = set()
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            raise PublicTaskExecutionContractError(
                "pre-action World-read requirement must be an object"
            )
        tool = _nonempty_text(
            raw.get("tool"),
            label="pre-action World-read tool",
        )
        if tool not in ALLOWED_WORLD_TOOLS:
            raise PublicTaskExecutionContractError(
                "pre-action read gate names an unsupported World tool"
            )
        argument_field = raw.get("argument_field")
        if argument_field is not None:
            argument_field = _nonempty_text(
                argument_field,
                label="pre-action World-read argument field",
            )
        raw_values = raw.get("required_values", [])
        if (
            not isinstance(raw_values, list)
            or any(not isinstance(item, str) or not item for item in raw_values)
            or len(raw_values) != len(set(raw_values))
        ):
            raise PublicTaskExecutionContractError(
                "pre-action World-read values must be a unique text list"
            )
        minimum_successes = raw.get("minimum_successes", 0)
        if (
            isinstance(minimum_successes, bool)
            or not isinstance(minimum_successes, int)
            or minimum_successes < 0
        ):
            raise PublicTaskExecutionContractError(
                "pre-action World-read minimum_successes must be a non-negative integer"
            )
        if bool(argument_field) is not bool(raw_values):
            raise PublicTaskExecutionContractError(
                "pre-action World-read exact targets require both a field and values"
            )
        if not raw_values and minimum_successes < 1:
            raise PublicTaskExecutionContractError(
                "pre-action World-read count requirement must be positive"
            )
        identity = (tool, argument_field)
        if identity in identities:
            raise PublicTaskExecutionContractError(
                "pre-action read gate repeats a World-read requirement"
            )
        identities.add(identity)
        requirements.append(
            PublicTaskWorldReadRequirement(
                tool=tool,
                argument_field=argument_field,
                required_values=tuple(raw_values),
                minimum_successes=minimum_successes,
            )
        )
    return PublicTaskPreActionReadGate(
        action_kind=action_kind,
        requirements=tuple(requirements),
        exclusive_until_complete=exclusive_until_complete,
    )


def _validate_framework_continuation(
    value: Any,
) -> PublicTaskFrameworkContinuation | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PublicTaskExecutionContractError(
            "framework_continuation must be an object"
        )
    action_kind = _nonempty_text(
        value.get("action_kind"),
        label="framework continuation action_kind",
    )
    if action_kind != "commerce.send_message":
        raise PublicTaskExecutionContractError(
            "framework continuation supports only commerce.send_message"
        )
    destination = value.get("destination")
    if destination not in {"@self", "@bound_counterparty"}:
        raise PublicTaskExecutionContractError(
            "framework continuation destination must be @self or @bound_counterparty"
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise PublicTaskExecutionContractError(
            "framework continuation payload must be an object"
        )
    normalized_payload = _strict_json_copy(
        payload,
        label="framework continuation payload",
    )
    if not isinstance(normalized_payload, dict):  # pragma: no cover
        raise AssertionError("validated continuation payload did not remain an object")
    raw_gate = value.get("accepted_reference_gate")
    gate: PublicTaskAcceptedReferenceGate | None = None
    if raw_gate is not None:
        if destination != "@bound_counterparty" or not isinstance(raw_gate, Mapping):
            raise PublicTaskExecutionContractError(
                "accepted reference gate requires a bound-counterparty continuation"
            )
        if set(raw_gate) != {"reference_field", "required_values"}:
            raise PublicTaskExecutionContractError(
                "accepted reference gate has an unsupported field"
            )
        reference_field = _nonempty_text(
            raw_gate.get("reference_field"),
            label="accepted reference gate field",
        )
        required_values = raw_gate.get("required_values")
        if (
            not isinstance(required_values, list)
            or not required_values
            or any(not isinstance(item, str) or not item for item in required_values)
            or len(required_values) != len(set(required_values))
        ):
            raise PublicTaskExecutionContractError(
                "accepted reference gate values must be a non-empty unique text list"
            )
        gate = PublicTaskAcceptedReferenceGate(
            reference_field=reference_field,
            required_values=tuple(required_values),
        )
    return PublicTaskFrameworkContinuation(
        action_kind=action_kind,
        destination=str(destination),
        payload=normalized_payload,
        accepted_reference_gate=gate,
    )


def _payload_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping) or not _payload_contains(
                actual_value,
                expected_value,
            ):
                return False
        elif type(actual_value) is not type(expected_value) or actual_value != expected_value:
            return False
    return True


def _actor_role(actor_id: str, *, label: str) -> str:
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise PublicTaskExecutionContractError(f"{label} must be non-empty text")
    role = actor_id.split(":", 1)[0]
    if not role:
        raise PublicTaskExecutionContractError(f"{label} has no role")
    return role


def _text_set(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        suffix = "a unique text list" if allow_empty else "a non-empty unique text list"
        raise PublicTaskExecutionContractError(f"{label} must be {suffix}")
    return frozenset(value)


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicTaskExecutionContractError(f"{label} must be non-empty text")
    return value


def _strict_json_copy(value: Any, *, label: str) -> Any:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return json.loads(rendered)
    except (TypeError, ValueError, RecursionError) as exc:
        raise PublicTaskExecutionContractError(f"{label} must be strict JSON") from exc


def _reject_hidden_authority(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_AUTHORITY_KEYS:
                raise PublicTaskExecutionContractError(
                    "public task execution contract contains hidden authority"
                )
            _reject_hidden_authority(item)
    elif isinstance(value, list):
        for item in value:
            _reject_hidden_authority(item)
