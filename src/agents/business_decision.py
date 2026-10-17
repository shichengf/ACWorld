"""Provider-neutral business decisions for CommerceWorld Agents.

The model-facing contract in this module deliberately contains no Runtime,
Platform, World, VCP, routing, correlation, or authority fields.  A model may
choose a business intent and business-owned values; :mod:`agents.agent_phase`
is responsible for translating that choice into the internal CommerceWorld
wire contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from agents.provider_boundary_policy import (
    benchmark_internal_namespace_field_v1,
    benchmark_metadata_field_is_private_v1,
    provider_boundary_field_is_private_v1,
    provider_boundary_semantic_path_v1,
)


LLM_DECISION_REQUEST_SCHEMA_V1 = "cwe.llm-decision-request.v1"
LLM_BUSINESS_DECISION_SCHEMA_V1 = "cwe.llm-business-decision.v1"
MODEL_BUSINESS_CHOICE_SCHEMA_V1 = "cwe.model-business-choice.v1"

# Frozen provider-facing reference policy.  Commerce resource identities stay
# inside the Agent: structured ``*_id``/``*_ids`` fields are advertised as
# opaque ``*_ref``/``*_refs`` fields and their values are domain-separated
# deterministic aliases.  The bridge owns the reversible projection.
PUBLIC_REFERENCE_POLICY_V1: Mapping[str, Any] = MappingProxyType(
    {
        "internal_singular_suffix": "_id",
        "internal_plural_suffix": "_ids",
        "public_singular_suffix": "_ref",
        "public_plural_suffix": "_refs",
        "alias_prefix": "business-ref-",
        "alias_hash": "sha256",
        "alias_hash_hex_chars": 20,
        "alias_domain_separator": "cwe.public-reference.v1",
        "candidate_sources": (
            "finite_schema_enum_union_explicit_reference_relations_and_"
            "authenticated_business_observations"
        ),
        "framework_reference_handling": "hide_and_inject_unique_authenticated_value",
    }
)

_INTENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_ROLE_VALUES = frozenset({"buyer", "merchant"})
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "action_kind",
        "allowed_routes",
        "authority",
        "authority_digest",
        "authority_id",
        "benchmark_task_id",
        "cert_id",
        "claim_compilation_authorized_claim_ids",
        "claim_compilation_templates",
        "contract_id",
        "destination",
        "expected_answer",
        "ground_truth",
        "idempotency_key",
        "ideal_trajectory",
        "in_reply_to",
        "msg_id",
        "oracle",
        "route",
        "read_acl",
        "request_fingerprint",
        "signature",
        "success_oracle",
        "task_id",
        "write_acl",
    }
)
_FORBIDDEN_ADDRESS_PREFIXES = (
    "platform:",
    "runtime:",
    "world:",
)
_FORBIDDEN_MODEL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:buyer|consumer|merchant|platform|runtime|world):"
    r"[^\s,;\]\[(){}]+"
    r"|(?<![A-Za-z0-9_])(?:commerce|platform|delegate)\.[a-z0-9_]+",
    flags=re.IGNORECASE,
)
_MAX_JSON_CHARS = 1_048_576

# Stable Agent-owned feedback for a rejected provider-neutral decision.  These
# strings are safe to persist in Tracker evidence and to return to the model;
# they never interpolate provider-authored keys, values, identifiers, or raw
# response text.
BUSINESS_REPAIR_ERROR_MESSAGES_V1: Mapping[str, str] = MappingProxyType(
    {
        "response_contract_invalid": (
            "The business decision did not match the advertised response contract."
        ),
        "intent_not_authorized": (
            "The selected business intent is not available in the current phase."
        ),
        "arguments_invalid": (
            "The business decision arguments do not match the advertised schema."
        ),
        "private_field_rejected": (
            "The business decision included a field owned by the local Agent."
        ),
        "world_read_already_completed": (
            "The requested business observation was already completed."
        ),
        "world_read_repeated": (
            "The business decision repeated an identical completed observation."
        ),
        "world_read_limit": ("The business decision exceeded the bounded observation limit."),
        "terminal_contract_invalid": (
            "The business decision did not select one valid terminal intent."
        ),
        "authority_rejected": (
            "The business decision is outside the actor's current business authority."
        ),
        "business_validation_rejected": (
            "The business decision failed deterministic Agent validation."
        ),
    }
)


class BusinessDecisionContractError(ValueError):
    """The provider-neutral business decision contract is malformed."""


@dataclass(frozen=True, slots=True)
class BusinessIntentSpec:
    """One model-visible business choice with a JSON object argument schema.

    ``source_name`` is Agent-private and is intentionally omitted from
    :meth:`to_public_dict`.  It lets the bridge retain compatibility with a
    domain compiler without leaking its compiler source or route identity.
    """

    intent: str
    description: str
    parameters: Mapping[str, Any]
    category: str
    source_name: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.intent, str) or not _INTENT_RE.fullmatch(self.intent):
            raise BusinessDecisionContractError(f"invalid business intent name: {self.intent!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise BusinessDecisionContractError("business intent description must be non-empty")
        if self.category not in {"observe", "act", "control"}:
            raise BusinessDecisionContractError(
                "business intent category must be observe, act, or control"
            )
        if not isinstance(self.source_name, str) or not self.source_name:
            raise BusinessDecisionContractError(
                "business intent requires an Agent-private source name"
            )
        if not isinstance(self.parameters, Mapping) or self.parameters.get("type") != "object":
            raise BusinessDecisionContractError(
                "business intent parameters must be an object JSON schema"
            )
        _require_strict_json(self.parameters, label="business intent schema")
        _require_public_value(
            self.parameters,
            path="parameters",
            semantic_path=("parameters",),
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Return the exact model-visible projection."""

        return {
            "intent": self.intent,
            "description": self.description,
            "category": self.category,
            "parameters": copy.deepcopy(dict(self.parameters)),
        }


@dataclass(frozen=True, slots=True)
class LLMDecisionRequestV1:
    """One provider-neutral request emitted by the local Agent."""

    decision_id: str
    role: str
    goal: str
    phase: str
    observations: tuple[Mapping[str, Any], ...]
    allowed_intents: tuple[BusinessIntentSpec, ...]
    schema_version: str = LLM_DECISION_REQUEST_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != LLM_DECISION_REQUEST_SCHEMA_V1:
            raise BusinessDecisionContractError("unsupported LLM decision request schema")
        for value, label in (
            (self.decision_id, "decision_id"),
            (self.goal, "goal"),
            (self.phase, "phase"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise BusinessDecisionContractError(f"{label} must be non-empty text")
        if self.role not in _ROLE_VALUES:
            raise BusinessDecisionContractError("LLM decision role is invalid")
        if not self.allowed_intents:
            raise BusinessDecisionContractError(
                "LLM decision request needs at least one business intent"
            )
        names = [row.intent for row in self.allowed_intents]
        if len(names) != len(set(names)):
            raise BusinessDecisionContractError(
                "LLM decision request contains duplicate business intents"
            )
        for index, observation in enumerate(self.observations):
            if not isinstance(observation, Mapping):
                raise BusinessDecisionContractError(f"observation {index} must be an object")
            _require_strict_json(observation, label=f"observation {index}")
            _require_public_value(
                observation,
                path=f"observations[{index}]",
                semantic_path=("observations",),
            )

    def to_public_dict(self) -> dict[str, Any]:
        """Return the complete provider request with no Agent-private fields."""

        payload = {
            "schema_version": self.schema_version,
            "role": self.role,
            "goal": self.goal,
            "phase": self.phase,
            "observations": [copy.deepcopy(dict(row)) for row in self.observations],
            "allowed_intents": [row.to_public_dict() for row in self.allowed_intents],
            "response_contract": {
                "schema_version": LLM_BUSINESS_DECISION_SCHEMA_V1,
                "required": ["schema_version", "intent", "arguments"],
                "optional": ["message"],
            },
        }
        _require_public_value(payload, path="request", semantic_path=("request",))
        _require_provider_surface_value(payload, path="request")
        _require_strict_json(payload, label="LLM decision request")
        return payload

    def to_prompt(self) -> str:
        """Render a deterministic JSON-only provider prompt."""

        return (
            "Choose exactly one allowed business intent. Return one JSON object "
            "matching response_contract. Do not invent hidden identifiers or "
            "communication fields.\n\n"
            + json.dumps(
                self.to_public_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )

    @property
    def public_sha256(self) -> str:
        return _canonical_sha256(self.to_public_dict())


@dataclass(frozen=True, slots=True)
class LLMBusinessDecisionV1:
    """A parsed model-authored business choice."""

    intent: str
    arguments: Mapping[str, Any]
    message: str | None = None
    schema_version: str = LLM_BUSINESS_DECISION_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != LLM_BUSINESS_DECISION_SCHEMA_V1:
            raise BusinessDecisionContractError("unsupported LLM business decision schema")
        if not isinstance(self.intent, str) or not _INTENT_RE.fullmatch(self.intent):
            raise BusinessDecisionContractError("business decision intent is invalid")
        if not isinstance(self.arguments, Mapping):
            raise BusinessDecisionContractError("business decision arguments must be an object")
        _require_strict_json(self.arguments, label="business decision arguments")
        _require_public_value(
            self.arguments,
            path="arguments",
            semantic_path=("arguments",),
        )
        if self.message is not None and (
            not isinstance(self.message, str) or len(self.message) > 8_000
        ):
            raise BusinessDecisionContractError("business decision message must be bounded text")

    @classmethod
    def parse(
        cls,
        raw: str,
        *,
        allowed_intents: Sequence[BusinessIntentSpec],
    ) -> "LLMBusinessDecisionV1":
        """Parse strict provider text and enforce the advertised intent set."""

        if not isinstance(raw, str) or not raw.strip():
            raise BusinessDecisionContractError("model business decision is empty")
        if len(raw) > _MAX_JSON_CHARS:
            raise BusinessDecisionContractError("model business decision is too large")
        text = _strip_single_code_fence(raw.strip())
        try:
            value = json.loads(
                text,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise BusinessDecisionContractError(
                "model business decision is not strict JSON"
            ) from exc
        if not isinstance(value, dict):
            raise BusinessDecisionContractError("model business decision must be one object")
        if set(value) - {"schema_version", "intent", "arguments", "message"}:
            raise BusinessDecisionContractError(
                "model business decision contains unsupported fields"
            )
        if set(value) < {"schema_version", "intent", "arguments"}:
            raise BusinessDecisionContractError(
                "model business decision is missing required fields"
            )
        _require_provider_surface_value(value, path="response")
        decision = cls(
            schema_version=value["schema_version"],
            intent=value["intent"],
            arguments=value["arguments"],
            message=value.get("message"),
        )
        specs = {row.intent: row for row in allowed_intents}
        if decision.intent not in specs:
            raise BusinessDecisionContractError(
                "model selected a business intent outside the current phase"
            )
        _validate_simple_object_schema(
            specs[decision.intent].parameters,
            decision.arguments,
        )
        return decision

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "intent": self.intent,
            "arguments": copy.deepcopy(dict(self.arguments)),
        }
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(frozen=True, slots=True)
class ModelBusinessChoiceV1:
    """Sanitized, schema-validated provenance for one model business choice.

    The record deliberately excludes the provider response body, optional
    ``message``, and any reasoning field.  Finite enum/const values, opaque
    public references, booleans, numbers, and structured containers remain
    available to capability scorers.  Other model-authored strings are
    irreversibly reduced to their character length and SHA-256 digest.
    """

    intent: str
    arguments: Mapping[str, Any]
    arguments_chars: int
    arguments_sha256: str
    schema_version: str = MODEL_BUSINESS_CHOICE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_BUSINESS_CHOICE_SCHEMA_V1:
            raise BusinessDecisionContractError("unsupported model business choice schema")
        if not isinstance(self.intent, str) or not _INTENT_RE.fullmatch(self.intent):
            raise BusinessDecisionContractError("model business choice intent is invalid")
        if not isinstance(self.arguments, Mapping):
            raise BusinessDecisionContractError("model business choice arguments must be an object")
        _require_strict_json(self.arguments, label="model business choice arguments")
        chars, digest = canonical_business_value_digest(self.arguments)
        if (
            isinstance(self.arguments_chars, bool)
            or self.arguments_chars != chars
            or self.arguments_sha256 != digest
        ):
            raise BusinessDecisionContractError(
                "model business choice argument evidence is inconsistent"
            )

    @classmethod
    def from_validated_decision(
        cls,
        decision: LLMBusinessDecisionV1,
        *,
        spec: BusinessIntentSpec,
    ) -> "ModelBusinessChoiceV1":
        """Create safe provenance from a decision on its public schema surface."""

        if decision.intent != spec.intent:
            raise BusinessDecisionContractError(
                "model business choice intent does not match its public schema"
            )
        _validate_simple_object_schema(spec.parameters, decision.arguments)
        sanitized = _sanitize_model_authored_value(
            spec.parameters,
            decision.arguments,
            public_reference=False,
        )
        if not isinstance(sanitized, Mapping):
            raise BusinessDecisionContractError(
                "model business choice arguments must remain an object"
            )
        chars, digest = canonical_business_value_digest(sanitized)
        return cls(
            intent=decision.intent,
            arguments=copy.deepcopy(dict(sanitized)),
            arguments_chars=chars,
            arguments_sha256=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent": self.intent,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "arguments_chars": self.arguments_chars,
            "arguments_sha256": self.arguments_sha256,
        }


def canonical_business_value_digest(value: Any) -> tuple[int, str]:
    """Return canonical JSON character length and SHA-256 for safe evidence."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise BusinessDecisionContractError(
            "business choice evidence must be canonical JSON"
        ) from exc
    if len(rendered) > _MAX_JSON_CHARS:
        raise BusinessDecisionContractError("business choice evidence is too large")
    return len(rendered), hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sanitize_model_authored_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    public_reference: bool,
) -> Any:
    """Project a validated provider value into scorer-safe provenance."""

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", {})
        output: dict[str, Any] = {}
        for raw_name, item in value.items():
            name = str(raw_name)
            child = properties.get(name) if isinstance(properties, Mapping) else None
            if not isinstance(child, Mapping):
                child = additional if isinstance(additional, Mapping) else {}
            output[name] = _sanitize_model_authored_value(
                child,
                item,
                public_reference=(
                    name.casefold().endswith("_ref") or name.casefold().endswith("_refs")
                ),
            )
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = schema.get("items")
        item_schema = items if isinstance(items, Mapping) else {}
        return [
            _sanitize_model_authored_value(
                item_schema,
                item,
                public_reference=public_reference,
            )
            for item in value
        ]
    if isinstance(value, str):
        if public_reference or _schema_retains_finite_string(schema, value):
            return value
        return {
            "text_chars": len(value),
            "text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    return copy.deepcopy(value)


def _schema_retains_finite_string(schema: Mapping[str, Any], value: str) -> bool:
    enum = schema.get("enum")
    if (
        isinstance(enum, Sequence)
        and not isinstance(enum, (str, bytes, bytearray))
        and any(isinstance(candidate, str) and candidate == value for candidate in enum)
    ):
        return True
    if isinstance(schema.get("const"), str) and schema.get("const") == value:
        return True
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes, bytearray)):
            continue
        for branch in branches:
            if isinstance(branch, Mapping) and _schema_retains_finite_string(branch, value):
                return True
    return False


def _validate_simple_object_schema(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    """Validate the deterministic JSON-schema subset used by Agent intents.

    Provider-side structured-output enforcement is advisory: different model
    gateways implement different subsets.  The local Agent therefore checks
    every admitted nested value before a domain compiler sees it, keeping a
    schema-invalid model choice scoreable instead of misclassifying a later
    compiler exception as infrastructure failure.
    """

    _validate_schema_value(schema, arguments, path="arguments")


def _validate_schema_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    path: str,
) -> None:
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, Sequence) or isinstance(enum, (str, bytes, bytearray)) or not enum:
            raise BusinessDecisionContractError("business intent contains an invalid enum")
        if not any(_strict_json_equal(value, candidate) for candidate in enum):
            raise BusinessDecisionContractError(
                f"business decision value at {path} is outside its visible choices"
            )

    if "const" in schema and not _strict_json_equal(value, schema["const"]):
        raise BusinessDecisionContractError(
            f"business decision value at {path} does not match its fixed choice"
        )

    _validate_schema_combinators(schema, value, path=path)

    declared_type = schema.get("type")
    if declared_type is not None:
        allowed_types = (
            tuple(declared_type)
            if isinstance(declared_type, Sequence)
            and not isinstance(declared_type, (str, bytes, bytearray))
            else (declared_type,)
        )
        if (
            not allowed_types
            or any(not isinstance(item, str) for item in allowed_types)
            or not any(_matches_json_type(value, item) for item in allowed_types)
        ):
            raise BusinessDecisionContractError(
                f"business decision value at {path} has the wrong JSON type"
            )

    if isinstance(value, Mapping):
        _validate_schema_object(schema, value, path=path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _validate_schema_array(schema, value, path=path)
    elif isinstance(value, str):
        _validate_schema_string(schema, value, path=path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_schema_number(schema, value, path=path)


def _validate_schema_combinators(
    schema: Mapping[str, Any],
    value: Any,
    *,
    path: str,
) -> None:
    for keyword in ("allOf", "anyOf", "oneOf"):
        raw_branches = schema.get(keyword)
        if raw_branches is None:
            continue
        if (
            not isinstance(raw_branches, Sequence)
            or isinstance(raw_branches, (str, bytes, bytearray))
            or not raw_branches
            or any(not isinstance(branch, Mapping) for branch in raw_branches)
        ):
            raise BusinessDecisionContractError(f"business intent {keyword} contract is invalid")
        matches = 0
        for branch in raw_branches:
            try:
                _validate_schema_value(branch, value, path=path)
            except BusinessDecisionContractError:
                continue
            matches += 1
        expected = len(raw_branches) if keyword == "allOf" else 1
        valid = matches == expected if keyword in {"allOf", "oneOf"} else matches >= 1
        if not valid:
            raise BusinessDecisionContractError(
                f"business decision value at {path} does not match {keyword}"
            )


def _validate_schema_object(
    schema: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    path: str,
) -> None:
    raw_properties = schema.get("properties", {})
    if not isinstance(raw_properties, Mapping):
        raise BusinessDecisionContractError("business intent object properties are invalid")
    properties = {str(name): child for name, child in raw_properties.items()}
    if any(not isinstance(child, Mapping) for child in properties.values()):
        raise BusinessDecisionContractError("business intent object property schema is invalid")
    raw_required = schema.get("required", ())
    if (
        not isinstance(raw_required, Sequence)
        or isinstance(raw_required, (str, bytes, bytearray))
        or any(not isinstance(name, str) for name in raw_required)
    ):
        raise BusinessDecisionContractError("business intent required list is invalid")
    required = set(raw_required)
    if required - set(value):
        raise BusinessDecisionContractError(
            f"business decision object at {path} is missing required fields"
        )

    additional = schema.get("additionalProperties", True)
    extra = set(value) - set(properties)
    if additional is False and extra:
        raise BusinessDecisionContractError(
            f"business decision object at {path} contains unsupported fields"
        )
    if not isinstance(additional, (bool, Mapping)):
        raise BusinessDecisionContractError(
            "business intent additionalProperties contract is invalid"
        )
    for name, item in value.items():
        child = properties.get(name)
        if child is None:
            if isinstance(additional, Mapping):
                _validate_schema_value(
                    additional,
                    item,
                    path=f"{path}.additional",
                )
            continue
        _validate_schema_value(child, item, path=f"{path}.{name}")


def _validate_schema_array(
    schema: Mapping[str, Any],
    value: Sequence[Any],
    *,
    path: str,
) -> None:
    minimum = _nonnegative_schema_integer(schema.get("minItems"), "minItems")
    maximum = _nonnegative_schema_integer(schema.get("maxItems"), "maxItems")
    if minimum is not None and len(value) < minimum:
        raise BusinessDecisionContractError(f"business decision array at {path} is too short")
    if maximum is not None and len(value) > maximum:
        raise BusinessDecisionContractError(f"business decision array at {path} is too long")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise BusinessDecisionContractError("business intent array bounds are invalid")
    unique = schema.get("uniqueItems", False)
    if not isinstance(unique, bool):
        raise BusinessDecisionContractError("business intent uniqueItems contract is invalid")
    if unique:
        canonical = [_strict_json_marker(item) for item in value]
        if len(canonical) != len(set(canonical)):
            raise BusinessDecisionContractError(
                f"business decision array at {path} contains duplicate items"
            )
    items = schema.get("items")
    if items is None:
        return
    if not isinstance(items, Mapping):
        raise BusinessDecisionContractError("business intent array item schema is invalid")
    for index, item in enumerate(value):
        _validate_schema_value(items, item, path=f"{path}[{index}]")


def _validate_schema_string(
    schema: Mapping[str, Any],
    value: str,
    *,
    path: str,
) -> None:
    minimum = _nonnegative_schema_integer(schema.get("minLength"), "minLength")
    maximum = _nonnegative_schema_integer(schema.get("maxLength"), "maxLength")
    if minimum is not None and len(value) < minimum:
        raise BusinessDecisionContractError(f"business decision text at {path} is too short")
    if maximum is not None and len(value) > maximum:
        raise BusinessDecisionContractError(f"business decision text at {path} is too long")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise BusinessDecisionContractError("business intent text bounds are invalid")
    raw_pattern = schema.get("pattern")
    if raw_pattern is not None:
        if not isinstance(raw_pattern, str):
            raise BusinessDecisionContractError("business intent text pattern is invalid")
        try:
            matches = re.search(raw_pattern, value) is not None
        except re.error as exc:
            raise BusinessDecisionContractError("business intent text pattern is invalid") from exc
        if not matches:
            raise BusinessDecisionContractError(
                f"business decision text at {path} does not match its pattern"
            )


def _validate_schema_number(
    schema: Mapping[str, Any],
    value: int | float,
    *,
    path: str,
) -> None:
    for keyword, comparison in (
        ("minimum", lambda candidate, bound: candidate >= bound),
        ("maximum", lambda candidate, bound: candidate <= bound),
    ):
        bound = schema.get(keyword)
        if bound is None:
            continue
        if not isinstance(bound, (int, float)) or isinstance(bound, bool):
            raise BusinessDecisionContractError(f"business intent {keyword} contract is invalid")
        if not comparison(value, bound):
            raise BusinessDecisionContractError(
                f"business decision number at {path} is outside its bounds"
            )


def _nonnegative_schema_integer(value: Any, keyword: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BusinessDecisionContractError(f"business intent {keyword} contract is invalid")
    return value


def _matches_json_type(value: Any, expected: str) -> bool:
    return {
        "array": lambda: isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "null": lambda: value is None,
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": lambda: isinstance(value, Mapping),
        "string": lambda: isinstance(value, str),
    }.get(expected, lambda: False)()


def _strict_json_marker(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _strict_json_equal(left: Any, right: Any) -> bool:
    return _strict_json_marker(left) == _strict_json_marker(right)


def _require_public_value(
    value: Any,
    *,
    path: str,
    semantic_path: tuple[str, ...],
) -> None:
    if isinstance(value, Mapping):
        semantic_path = provider_boundary_semantic_path_v1(semantic_path, value)
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if benchmark_internal_namespace_field_v1(
                normalized
            ) or benchmark_metadata_field_is_private_v1(
                semantic_path,
                normalized,
            ):
                raise BusinessDecisionContractError(f"{path}.{key} is Agent-private")
            if normalized in _FORBIDDEN_PUBLIC_KEYS:
                raise BusinessDecisionContractError(f"{path}.{key} is Agent-private")
            if provider_boundary_field_is_private_v1(
                semantic_path,
                normalized,
            ):
                raise BusinessDecisionContractError(f"{path}.{key} is Agent-private")
            _require_public_value(
                item,
                path=f"{path}.{key}",
                semantic_path=(*semantic_path, normalized),
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _require_public_value(
                item,
                path=f"{path}[{index}]",
                semantic_path=semantic_path,
            )
        return
    if isinstance(value, str) and value.casefold().startswith(_FORBIDDEN_ADDRESS_PREFIXES):
        raise BusinessDecisionContractError(f"{path} contains an internal service address")


def _require_provider_surface_value(value: Any, *, path: str) -> None:
    """Reject actor addresses and wire action names at the provider boundary.

    Internal authority schemas are represented by ``BusinessIntentSpec``
    before the Agent aliases their values, so their validation cannot apply
    this rule.  Requests call it only after projection; responses call it only
    before public references are restored to Agent-private identities.
    """

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            _require_provider_surface_value(item, path=f"{path}.{raw_key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _require_provider_surface_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _FORBIDDEN_MODEL_TOKEN_RE.search(value):
        raise BusinessDecisionContractError(
            f"{path} contains an Agent-private address or action namespace"
        )


def _require_strict_json(value: Any, *, label: str) -> None:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise BusinessDecisionContractError(f"{label} must be strict JSON") from exc
    if len(rendered) > _MAX_JSON_CHARS:
        raise BusinessDecisionContractError(f"{label} is too large")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def business_decision_contract_digest() -> str:
    """Hash the exact model-visible decision protocol and leak policy."""

    return _canonical_sha256(
        {
            "request_schema": LLM_DECISION_REQUEST_SCHEMA_V1,
            "decision_schema": LLM_BUSINESS_DECISION_SCHEMA_V1,
            "model_choice_schema": MODEL_BUSINESS_CHOICE_SCHEMA_V1,
            "model_choice_string_policy": {
                "retain": ["finite_enum", "const", "public_reference"],
                "otherwise": ["text_chars", "text_sha256"],
            },
            "intent_pattern": _INTENT_RE.pattern,
            "roles": sorted(_ROLE_VALUES),
            "forbidden_public_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
            "forbidden_address_prefixes": list(_FORBIDDEN_ADDRESS_PREFIXES),
            "forbidden_model_token_pattern": _FORBIDDEN_MODEL_TOKEN_RE.pattern,
            "public_reference_policy": dict(PUBLIC_REFERENCE_POLICY_V1),
            "local_schema_validation": [
                "additionalProperties",
                "allOf",
                "anyOf",
                "const",
                "enum",
                "maximum",
                "maxItems",
                "maxLength",
                "minimum",
                "minItems",
                "minLength",
                "oneOf",
                "pattern",
                "properties",
                "required",
                "type",
                "uniqueItems",
            ],
            "repair_error_messages": dict(BUSINESS_REPAIR_ERROR_MESSAGES_V1),
        }
    )


def business_repair_error_v1(error: BaseException) -> tuple[str, str]:
    """Classify a rejection without retaining provider-authored fragments."""

    message = str(error)
    if message in {
        "the requested World reads were already completed",
    }:
        code = "world_read_already_completed"
    elif message in {
        "business decision repeated an identical World read batch",
        "business decision repeated an identical World read",
        "business decision repeated an unchanged platform observation",
    }:
        code = "world_read_repeated"
    elif message in {
        "business decision exceeds the distinct World read limit",
    }:
        code = "world_read_limit"
    elif "outside the current phase" in message or "outside the active Agent phase" in message:
        code = "intent_not_authorized"
    elif "Agent-private" in message or "internal service address" in message:
        code = "private_field_rejected"
    elif message.startswith("business decision") and any(
        marker in message
        for marker in (
            "argument",
            "arguments",
            "missing required",
            "unsupported",
        )
    ):
        code = "arguments_invalid"
    elif any(
        marker in message
        for marker in (
            "must contain exactly one",
            "must contain one",
            "must select exactly one",
            "terminal intent",
            "terminal commerce action",
            "finish-turn",
        )
    ):
        code = "terminal_contract_invalid"
    elif any(
        marker in message
        for marker in (
            "authority",
            "not authorized",
            "outside the actor role",
            "outside the current actor",
        )
    ):
        code = "authority_rejected"
    elif message.startswith("model business decision") or message.startswith(
        "unsupported LLM business decision"
    ):
        code = "response_contract_invalid"
    else:
        code = "business_validation_rejected"
    return code, BUSINESS_REPAIR_ERROR_MESSAGES_V1[code]


def _strip_single_code_fence(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


__all__ = [
    "BUSINESS_REPAIR_ERROR_MESSAGES_V1",
    "BusinessDecisionContractError",
    "BusinessIntentSpec",
    "LLM_BUSINESS_DECISION_SCHEMA_V1",
    "LLM_DECISION_REQUEST_SCHEMA_V1",
    "LLMBusinessDecisionV1",
    "LLMDecisionRequestV1",
    "MODEL_BUSINESS_CHOICE_SCHEMA_V1",
    "ModelBusinessChoiceV1",
    "PUBLIC_REFERENCE_POLICY_V1",
    "business_decision_contract_digest",
    "business_repair_error_v1",
    "canonical_business_value_digest",
]
