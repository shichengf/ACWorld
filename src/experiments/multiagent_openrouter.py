"""Frozen paid-contract substrate for the one-shot 5x5 OpenRouter trace.

This module does not execute a paid request.  It closes the C4 authorization
boundary: an exact public OpenRouter catalog row is validated without aliases,
the shared-market scenario and complete local execution inputs are frozen, and
the worst-case token charge must fit below the declared hard cap.  C5 may only
construct network channels from a successfully verified instance of this
contract and after a fresh, explicit ``--allow-paid`` authorization.

The provider still returns only provider-neutral business decisions.  The
CommerceWorld Agent owns reads, identifiers, routes, authority, envelopes,
retries, correlation, and continuation exactly as it does in the zero-model
preflight.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
import ssl
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from agents.inference import (
    BusinessDecisionResponseV1,
    ChannelTransportError,
    OpenAIChannel,
    ProviderResponseError,
    _raise_for_in_band_provider_error,
)
from episode.capability_materializer import scenario_content_hash_v2
from episode.capability_runtime import RuntimeEvidenceBundleV2, canonical_sha256
from episode.runner import EpisodeBatch
from episode.scenario import materialize_initial_world_tables, population_for_scenario
from episode.termination import EPISODE_TERMINATION_ARTIFACT, load_verified_scoreable_termination
from evals.serialize import to_canonical
from experiments.environment_study import (
    ARTIFACT_INDEX_NAME,
    DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES,
    SourceManifestV1,
    build_artifact_index,
    build_source_manifest,
    iter_loaded_module_files,
    repo_source_paths,
    unlisted_loaded_sources,
    verify_source_manifest,
    write_json_artifact,
)
from experiments.multiagent_preflight import build_multiagent_scenario
from runtime.tracker_evidence import (
    verified_model_business_choices,
    verify_tracker_evidence,
)


MULTIAGENT_OPENROUTER_CONTRACT_SCHEMA = (
    "cwe.multiagent-openrouter-trace-contract.v1"
)
MULTIAGENT_LLM_TRACE_REPORT_SCHEMA = "cwe.multiagent-llm-trace-report.v1"
OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.5-flash"
EXPECTED_CANONICAL_SLUG = "google/gemini-3.5-flash-20260519"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_BILLABLE_ATTEMPTS = 50
MAX_INPUT_BYTES_PER_ATTEMPT = 20_000
MAX_OUTPUT_TOKENS_PER_ATTEMPT = 1_000
HARD_COST_CAP_USD = Decimal("3.00")
FORMAT_REPAIRS_PER_ACTOR = 1
TRANSPORT_RETRY_BACKOFF_SECONDS = (60, 300)
REQUEST_TIMEOUT_SECONDS = 240.0

REQUIRED_MODEL_PARAMETERS = frozenset(
    {"max_tokens", "reasoning", "reasoning_effort", "response_format"}
)
PROVIDER_ROUTE: Mapping[str, Any] = MappingProxyType(
    {
        # Provider fallback is allowed only between endpoints serving the same
        # exact model id.  No model-list fallback or router model is supplied.
        "sort": "price",
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
        # OpenRouter's max_price fields are USD per million tokens.  They
        # prevent a provider price increase after catalog capture from
        # crossing the frozen budget even within the short preflight/run gap.
        "max_price": {"prompt": 1.5, "completion": 9.0},
    }
)
REQUEST_SETTINGS: Mapping[str, Any] = MappingProxyType(
    {
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_OUTPUT_TOKENS_PER_ATTEMPT,
        "reasoning": {"effort": "minimal", "exclude": True},
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_DIAGNOSTIC_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_ACTOR_ROLES = frozenset({"buyer", "merchant"})
_SAFE_STOP_REASONS = frozenset(
    {
        "completed",
        "infrastructure_error",
        "model_protocol_error",
        "resource_limit",
        "security_guard",
    }
)
_FORBIDDEN_REPORT_KEY_MARKERS = (
    "api_key",
    "authorization",
    "capability_score",
    "content",
    "private_value",
    "prompt_text",
    "ranking",
    "raw_",
    "response_text",
    "reward",
    "secret",
    "strict_success",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class MultiAgentOpenRouterError(ValueError):
    """The paid-trace catalog, contract, budget, or report is invalid."""


class PaidAuthorizationRequired(MultiAgentOpenRouterError):
    """A caller tried to cross the paid boundary without fresh authorization."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MultiAgentOpenRouterError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiAgentOpenRouterError(f"{label} must be non-empty text")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MultiAgentOpenRouterError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _require_float(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiAgentOpenRouterError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= minimum:
        raise MultiAgentOpenRouterError(f"{label} must be > {minimum}")
    return parsed


def _require_exact_keys(
    raw: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    keys = {str(key) for key in raw}
    required_set = set(required)
    missing = required_set - keys
    unknown = keys - (required_set | set(optional))
    if missing:
        raise MultiAgentOpenRouterError(
            f"{label} is missing required keys: {sorted(missing)}"
        )
    if unknown:
        raise MultiAgentOpenRouterError(
            f"{label} has unknown keys: {sorted(unknown)}"
        )


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise MultiAgentOpenRouterError(f"{label} must be a non-negative decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MultiAgentOpenRouterError(
            f"{label} must be a non-negative decimal"
        ) from exc
    if not parsed.is_finite() or parsed < 0:
        raise MultiAgentOpenRouterError(f"{label} must be a non-negative decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validate_sanitized_diagnostics(value: Any, *, path: str = "diagnostics") -> None:
    """Reject raw provider/private fields and benchmark scoring projections."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise MultiAgentOpenRouterError(
                    f"{path} keys must be non-empty strings"
                )
            lowered = key.lower()
            if any(marker in lowered for marker in _FORBIDDEN_REPORT_KEY_MARKERS):
                raise MultiAgentOpenRouterError(
                    f"{path} contains forbidden diagnostic field {key!r}"
                )
            _validate_sanitized_diagnostics(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_sanitized_diagnostics(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _SAFE_DIAGNOSTIC_TEXT_RE.fullmatch(value) is None:
            raise MultiAgentOpenRouterError(
                f"{path} text must be a bounded canonical token or digest"
            )
        return
    if value is None or isinstance(value, (int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise MultiAgentOpenRouterError(f"{path} must be finite JSON")
        return
    raise MultiAgentOpenRouterError(f"{path} must contain only JSON-safe values")


@dataclass(frozen=True, slots=True)
class OpenRouterModelSnapshotV1:
    """Sanitized exact-model facts captured from the public models catalog."""

    model_id: str
    canonical_slug: str
    catalog_row_sha256: str
    context_length: int
    max_completion_tokens: int
    supported_parameters: tuple[str, ...]
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    prompt_price_per_token: str
    completion_price_per_token: str
    internal_reasoning_price_per_token: str
    request_price: str
    expiration_date: None = None

    def __post_init__(self) -> None:
        if self.model_id != DEFAULT_OPENROUTER_MODEL:
            raise MultiAgentOpenRouterError(
                "OpenRouter snapshot must use the exact configured model id"
            )
        if self.canonical_slug != EXPECTED_CANONICAL_SLUG:
            raise MultiAgentOpenRouterError(
                "OpenRouter canonical slug changed; explicit review is required"
            )
        _require_digest(self.catalog_row_sha256, label="catalog_row_sha256")
        _require_int(self.context_length, label="context_length", minimum=1)
        _require_int(
            self.max_completion_tokens,
            label="max_completion_tokens",
            minimum=MAX_OUTPUT_TOKENS_PER_ATTEMPT,
        )
        if tuple(sorted(set(self.supported_parameters))) != self.supported_parameters:
            raise MultiAgentOpenRouterError(
                "supported_parameters must be unique and sorted"
            )
        missing = REQUIRED_MODEL_PARAMETERS - set(self.supported_parameters)
        if missing:
            raise MultiAgentOpenRouterError(
                "exact model lacks required parameters: " + ", ".join(sorted(missing))
            )
        if "text" not in self.input_modalities or "text" not in self.output_modalities:
            raise MultiAgentOpenRouterError("exact model must support text input and output")
        for label, value in (
            ("prompt_price_per_token", self.prompt_price_per_token),
            ("completion_price_per_token", self.completion_price_per_token),
            (
                "internal_reasoning_price_per_token",
                self.internal_reasoning_price_per_token,
            ),
            ("request_price", self.request_price),
        ):
            _decimal(value, label=label)
        if self.expiration_date is not None:
            raise MultiAgentOpenRouterError("exact model is expired or scheduled to expire")

    @property
    def prompt_price(self) -> Decimal:
        return _decimal(self.prompt_price_per_token, label="prompt price")

    @property
    def completion_price(self) -> Decimal:
        return _decimal(self.completion_price_per_token, label="completion price")

    @property
    def reasoning_price(self) -> Decimal:
        return _decimal(
            self.internal_reasoning_price_per_token, label="reasoning price"
        )

    @property
    def fixed_request_price(self) -> Decimal:
        return _decimal(self.request_price, label="request price")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "canonical_slug": self.canonical_slug,
            "catalog_row_sha256": self.catalog_row_sha256,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "supported_parameters": list(self.supported_parameters),
            "input_modalities": list(self.input_modalities),
            "output_modalities": list(self.output_modalities),
            "prompt_price_per_token": self.prompt_price_per_token,
            "completion_price_per_token": self.completion_price_per_token,
            "internal_reasoning_price_per_token": (
                self.internal_reasoning_price_per_token
            ),
            "request_price": self.request_price,
            "expiration_date": self.expiration_date,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpenRouterModelSnapshotV1":
        _require_exact_keys(
            raw,
            required=(
                "model_id",
                "canonical_slug",
                "catalog_row_sha256",
                "context_length",
                "max_completion_tokens",
                "supported_parameters",
                "input_modalities",
                "output_modalities",
                "prompt_price_per_token",
                "completion_price_per_token",
                "internal_reasoning_price_per_token",
                "request_price",
                "expiration_date",
            ),
            label="OpenRouter model snapshot",
        )
        sequences: dict[str, tuple[str, ...]] = {}
        for name in (
            "supported_parameters",
            "input_modalities",
            "output_modalities",
        ):
            value = raw.get(name)
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise MultiAgentOpenRouterError(f"{name} must be a text list")
            sequences[name] = tuple(value)
        expiration = raw.get("expiration_date")
        if expiration is not None:
            raise MultiAgentOpenRouterError("exact model is expired or scheduled to expire")
        return cls(
            model_id=_require_text(raw.get("model_id"), label="model_id"),
            canonical_slug=_require_text(
                raw.get("canonical_slug"), label="canonical_slug"
            ),
            catalog_row_sha256=_require_digest(
                raw.get("catalog_row_sha256"), label="catalog_row_sha256"
            ),
            context_length=_require_int(
                raw.get("context_length"), label="context_length", minimum=1
            ),
            max_completion_tokens=_require_int(
                raw.get("max_completion_tokens"),
                label="max_completion_tokens",
                minimum=1,
            ),
            supported_parameters=sequences["supported_parameters"],
            input_modalities=sequences["input_modalities"],
            output_modalities=sequences["output_modalities"],
            prompt_price_per_token=_require_text(
                raw.get("prompt_price_per_token"), label="prompt_price_per_token"
            ),
            completion_price_per_token=_require_text(
                raw.get("completion_price_per_token"),
                label="completion_price_per_token",
            ),
            internal_reasoning_price_per_token=_require_text(
                raw.get("internal_reasoning_price_per_token"),
                label="internal_reasoning_price_per_token",
            ),
            request_price=_require_text(
                raw.get("request_price"), label="request_price"
            ),
            expiration_date=None,
        )


def model_snapshot_from_catalog(
    payload: Mapping[str, Any],
    *,
    exact_model_id: str = DEFAULT_OPENROUTER_MODEL,
) -> OpenRouterModelSnapshotV1:
    """Select and validate one exact catalog row; aliases never match."""

    if exact_model_id != DEFAULT_OPENROUTER_MODEL:
        raise MultiAgentOpenRouterError("automatic model substitution is forbidden")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise MultiAgentOpenRouterError("OpenRouter catalog requires a data list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("id") == exact_model_id
    ]
    if len(matches) != 1:
        raise MultiAgentOpenRouterError(
            "OpenRouter catalog must contain the exact model id exactly once"
        )
    row = dict(matches[0])
    pricing = row.get("pricing")
    architecture = row.get("architecture")
    top_provider = row.get("top_provider")
    if not isinstance(pricing, Mapping):
        raise MultiAgentOpenRouterError("exact model has no pricing object")
    if pricing.get("overrides"):
        raise MultiAgentOpenRouterError(
            "tiered model pricing needs explicit budget review"
        )
    if not isinstance(architecture, Mapping) or not isinstance(top_provider, Mapping):
        raise MultiAgentOpenRouterError(
            "exact model lacks architecture or active top-provider metadata"
        )
    supported = row.get("supported_parameters")
    inputs = architecture.get("input_modalities")
    outputs = architecture.get("output_modalities")
    if not isinstance(supported, list) or not all(
        isinstance(item, str) for item in supported
    ):
        raise MultiAgentOpenRouterError("supported_parameters must be a text list")
    if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
        raise MultiAgentOpenRouterError("input_modalities must be a text list")
    if not isinstance(outputs, list) or not all(
        isinstance(item, str) for item in outputs
    ):
        raise MultiAgentOpenRouterError("output_modalities must be a text list")

    context_length = _require_int(
        row.get("context_length"), label="context_length", minimum=1
    )
    provider_context = _require_int(
        top_provider.get("context_length"),
        label="top_provider.context_length",
        minimum=1,
    )
    max_completion = _require_int(
        top_provider.get("max_completion_tokens"),
        label="top_provider.max_completion_tokens",
        minimum=MAX_OUTPUT_TOKENS_PER_ATTEMPT,
    )
    if min(context_length, provider_context) < (
        MAX_INPUT_BYTES_PER_ATTEMPT + MAX_OUTPUT_TOKENS_PER_ATTEMPT
    ):
        raise MultiAgentOpenRouterError("exact model context is below the frozen request bound")

    def price(name: str) -> str:
        raw = pricing.get(name, "0")
        return _decimal_text(_decimal(raw, label=f"pricing.{name}"))

    return OpenRouterModelSnapshotV1(
        model_id=exact_model_id,
        canonical_slug=_require_text(
            row.get("canonical_slug"), label="canonical_slug"
        ),
        catalog_row_sha256=_digest(row),
        context_length=context_length,
        max_completion_tokens=max_completion,
        supported_parameters=tuple(sorted(set(supported))),
        input_modalities=tuple(sorted(set(inputs))),
        output_modalities=tuple(sorted(set(outputs))),
        prompt_price_per_token=price("prompt"),
        completion_price_per_token=price("completion"),
        internal_reasoning_price_per_token=price("internal_reasoning"),
        request_price=price("request"),
        expiration_date=row.get("expiration_date"),
    )


def fetch_openrouter_model_snapshot(
    *,
    endpoint: str = OPENROUTER_MODELS_ENDPOINT,
    timeout: float = 30.0,
    api_key: str | None = None,
) -> OpenRouterModelSnapshotV1:
    """Read the public model catalog; this function never calls inference."""

    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise MultiAgentOpenRouterError("catalog timeout must be numeric")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise MultiAgentOpenRouterError("catalog timeout must be positive and finite")
    headers = {
        "Accept": "application/json",
        "User-Agent": "CommerceWorld-KDD2027-M2M-Preflight/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise MultiAgentOpenRouterError(
            f"OpenRouter models endpoint returned HTTP {exc.code}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise MultiAgentOpenRouterError(
            "OpenRouter models endpoint is unavailable"
        ) from exc
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiAgentOpenRouterError(
            "OpenRouter models endpoint returned invalid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MultiAgentOpenRouterError("OpenRouter model catalog must be an object")
    return model_snapshot_from_catalog(payload)


@dataclass(frozen=True, slots=True)
class FrozenTraceActorV1:
    actor_id: str
    role: str
    actor_input_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.actor_id, str)
            or self.actor_id.split(":", 1)[0] != self.role
            or self.role not in _SAFE_ACTOR_ROLES
        ):
            raise MultiAgentOpenRouterError("trace actor id/role binding is invalid")
        _require_digest(self.actor_input_sha256, label="actor_input_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "actor_id": self.actor_id,
            "role": self.role,
            "actor_input_sha256": self.actor_input_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FrozenTraceActorV1":
        _require_exact_keys(
            raw,
            required=("actor_id", "role", "actor_input_sha256"),
            label="trace actor",
        )
        return cls(
            actor_id=_require_text(raw.get("actor_id"), label="actor_id"),
            role=_require_text(raw.get("role"), label="role"),
            actor_input_sha256=_require_digest(
                raw.get("actor_input_sha256"), label="actor_input_sha256"
            ),
        )


def _trace_component_sources() -> Mapping[str, tuple[str, ...]]:
    components = {
        name: tuple(paths)
        for name, paths in DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES.items()
    }
    path = "src/experiments/multiagent_openrouter.py"
    experiments = components.get("experiments_closure", ())
    if path not in experiments:
        components["experiments_closure"] = (*experiments, path)
    return MappingProxyType(components)


def build_multiagent_openrouter_source_manifest(
    repo_root: str | Path = _REPO_ROOT,
) -> SourceManifestV1:
    return build_source_manifest(repo_root, _trace_component_sources())


def _frozen_actors() -> tuple[FrozenTraceActorV1, ...]:
    population = population_for_scenario(build_multiagent_scenario(5))
    actors: list[FrozenTraceActorV1] = []
    for buyer in population.buyers:
        actors.append(
            FrozenTraceActorV1(
                actor_id=buyer.buyer_id,
                role="buyer",
                actor_input_sha256=_digest(
                    to_canonical(
                        {
                            "persona": buyer.persona,
                            "mandate": buyer.mandate,
                            "initial_state": buyer.initial_state,
                        }
                    )
                ),
            )
        )
    for merchant in population.merchants:
        actors.append(
            FrozenTraceActorV1(
                actor_id=merchant.merchant_id,
                role="merchant",
                actor_input_sha256=_digest(
                    to_canonical(
                        {
                            "persona": merchant.persona,
                            "policy": merchant.policy,
                            "catalog_scope": merchant.catalog_scope,
                            "initial_state": merchant.initial_state,
                        }
                    )
                ),
            )
        )
    return tuple(sorted(actors, key=lambda actor: actor.actor_id))


def worst_case_cost(snapshot: OpenRouterModelSnapshotV1) -> Decimal:
    """Conservative cap: completion and reasoning prices are both charged."""

    per_attempt = (
        Decimal(MAX_INPUT_BYTES_PER_ATTEMPT) * snapshot.prompt_price
        + Decimal(MAX_OUTPUT_TOKENS_PER_ATTEMPT)
        * (snapshot.completion_price + snapshot.reasoning_price)
        + snapshot.fixed_request_price
    )
    return per_attempt * Decimal(MAX_BILLABLE_ATTEMPTS)


@dataclass(frozen=True, slots=True)
class MultiAgentOpenRouterTraceContractV1:
    """Frozen C4 contract.  Possession does not authorize provider spending."""

    model: OpenRouterModelSnapshotV1
    request_model_id: str
    actors: tuple[FrozenTraceActorV1, ...]
    scenario_sha256: str
    initial_world_sha256: str
    logical_schedule_sha256: str
    system_prompt_sha256: str
    skill_bundle_sha256: str
    decision_schema_sha256: str
    route_registry_sha256: str
    source_manifest_sha256: str
    provider_route: Mapping[str, Any]
    request_settings: Mapping[str, Any]
    max_billable_attempts: int
    max_input_bytes_per_attempt: int
    max_output_tokens_per_attempt: int
    format_repairs_per_actor: int
    transport_retry_backoff_seconds: tuple[int, ...]
    request_timeout_seconds: float
    hard_cost_cap_usd: str
    worst_case_cost_usd: str
    contract_id: str
    schema_version: str = MULTIAGENT_OPENROUTER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MULTIAGENT_OPENROUTER_CONTRACT_SCHEMA:
            raise MultiAgentOpenRouterError("unsupported paid trace contract schema")
        if len(self.actors) != 10:
            raise MultiAgentOpenRouterError("paid trace contract requires exactly ten actors")
        actor_ids = tuple(actor.actor_id for actor in self.actors)
        if actor_ids != tuple(sorted(set(actor_ids))):
            raise MultiAgentOpenRouterError("paid trace actors must be unique and sorted")
        if sum(actor.role == "buyer" for actor in self.actors) != 5 or sum(
            actor.role == "merchant" for actor in self.actors
        ) != 5:
            raise MultiAgentOpenRouterError("paid trace requires five buyers and five merchants")
        if self.request_model_id != self.model.canonical_slug:
            raise MultiAgentOpenRouterError(
                "paid POST must use the revalidated dated canonical model slug"
            )
        for label, value in (
            ("scenario_sha256", self.scenario_sha256),
            ("initial_world_sha256", self.initial_world_sha256),
            ("logical_schedule_sha256", self.logical_schedule_sha256),
            ("system_prompt_sha256", self.system_prompt_sha256),
            ("skill_bundle_sha256", self.skill_bundle_sha256),
            ("decision_schema_sha256", self.decision_schema_sha256),
            ("route_registry_sha256", self.route_registry_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
        ):
            _require_digest(value, label=label)
        if dict(self.provider_route) != _thaw(PROVIDER_ROUTE):
            raise MultiAgentOpenRouterError("provider routing differs from the frozen route")
        if _thaw(self.request_settings) != _thaw(REQUEST_SETTINGS):
            raise MultiAgentOpenRouterError("request settings differ from the frozen settings")
        if self.max_billable_attempts != MAX_BILLABLE_ATTEMPTS:
            raise MultiAgentOpenRouterError("billable-attempt bound changed")
        if self.max_input_bytes_per_attempt != MAX_INPUT_BYTES_PER_ATTEMPT:
            raise MultiAgentOpenRouterError("input-byte bound changed")
        if self.max_output_tokens_per_attempt != MAX_OUTPUT_TOKENS_PER_ATTEMPT:
            raise MultiAgentOpenRouterError("output-token bound changed")
        if self.format_repairs_per_actor != FORMAT_REPAIRS_PER_ACTOR:
            raise MultiAgentOpenRouterError("Agent format-repair policy changed")
        if self.transport_retry_backoff_seconds != TRANSPORT_RETRY_BACKOFF_SECONDS:
            raise MultiAgentOpenRouterError("Agent transport-retry policy changed")
        if self.request_timeout_seconds != REQUEST_TIMEOUT_SECONDS:
            raise MultiAgentOpenRouterError("provider request timeout changed")
        cap = _decimal(self.hard_cost_cap_usd, label="hard_cost_cap_usd")
        worst = _decimal(self.worst_case_cost_usd, label="worst_case_cost_usd")
        if cap != HARD_COST_CAP_USD or worst != worst_case_cost(self.model):
            raise MultiAgentOpenRouterError("paid trace cost calculation is inconsistent")
        if worst > cap:
            raise MultiAgentOpenRouterError("worst-case trace cost exceeds the hard cap")
        expected = compute_multiagent_openrouter_contract_id(self)
        if self.contract_id != expected:
            raise MultiAgentOpenRouterError("paid trace contract_id is inconsistent")
        object.__setattr__(self, "provider_route", _deep_freeze(self.provider_route))
        object.__setattr__(self, "request_settings", _deep_freeze(self.request_settings))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "request_model_id": self.request_model_id,
            "actors": [actor.to_dict() for actor in self.actors],
            "scenario_sha256": self.scenario_sha256,
            "initial_world_sha256": self.initial_world_sha256,
            "logical_schedule_sha256": self.logical_schedule_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "skill_bundle_sha256": self.skill_bundle_sha256,
            "decision_schema_sha256": self.decision_schema_sha256,
            "route_registry_sha256": self.route_registry_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "provider_route": _thaw(self.provider_route),
            "request_settings": _thaw(self.request_settings),
            "max_billable_attempts": self.max_billable_attempts,
            "max_input_bytes_per_attempt": self.max_input_bytes_per_attempt,
            "max_output_tokens_per_attempt": self.max_output_tokens_per_attempt,
            "format_repairs_per_actor": self.format_repairs_per_actor,
            "transport_retry_backoff_seconds": list(
                self.transport_retry_backoff_seconds
            ),
            "request_timeout_seconds": self.request_timeout_seconds,
            "hard_cost_cap_usd": self.hard_cost_cap_usd,
            "worst_case_cost_usd": self.worst_case_cost_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"contract_id": self.contract_id, **self.identity_payload()}

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> "MultiAgentOpenRouterTraceContractV1":
        required = (
            "schema_version",
            "contract_id",
            "model",
            "request_model_id",
            "actors",
            "scenario_sha256",
            "initial_world_sha256",
            "logical_schedule_sha256",
            "system_prompt_sha256",
            "skill_bundle_sha256",
            "decision_schema_sha256",
            "route_registry_sha256",
            "source_manifest_sha256",
            "provider_route",
            "request_settings",
            "max_billable_attempts",
            "max_input_bytes_per_attempt",
            "max_output_tokens_per_attempt",
            "format_repairs_per_actor",
            "transport_retry_backoff_seconds",
            "request_timeout_seconds",
            "hard_cost_cap_usd",
            "worst_case_cost_usd",
        )
        _require_exact_keys(raw, required=required, label="paid trace contract")
        model = raw.get("model")
        actors = raw.get("actors")
        route = raw.get("provider_route")
        settings = raw.get("request_settings")
        backoff = raw.get("transport_retry_backoff_seconds")
        if not isinstance(model, Mapping):
            raise MultiAgentOpenRouterError("contract model must be an object")
        if not isinstance(actors, list) or not all(
            isinstance(actor, Mapping) for actor in actors
        ):
            raise MultiAgentOpenRouterError("contract actors must be an object list")
        if not isinstance(route, Mapping) or not isinstance(settings, Mapping):
            raise MultiAgentOpenRouterError("contract route/settings must be objects")
        if not isinstance(backoff, list):
            raise MultiAgentOpenRouterError("contract retry backoff must be a list")
        return cls(
            schema_version=_require_text(
                raw.get("schema_version"), label="schema_version"
            ),
            contract_id=_require_digest(raw.get("contract_id"), label="contract_id"),
            model=OpenRouterModelSnapshotV1.from_dict(model),
            request_model_id=_require_text(
                raw.get("request_model_id"), label="request_model_id"
            ),
            actors=tuple(FrozenTraceActorV1.from_dict(actor) for actor in actors),
            scenario_sha256=_require_digest(
                raw.get("scenario_sha256"), label="scenario_sha256"
            ),
            initial_world_sha256=_require_digest(
                raw.get("initial_world_sha256"), label="initial_world_sha256"
            ),
            logical_schedule_sha256=_require_digest(
                raw.get("logical_schedule_sha256"), label="logical_schedule_sha256"
            ),
            system_prompt_sha256=_require_digest(
                raw.get("system_prompt_sha256"), label="system_prompt_sha256"
            ),
            skill_bundle_sha256=_require_digest(
                raw.get("skill_bundle_sha256"), label="skill_bundle_sha256"
            ),
            decision_schema_sha256=_require_digest(
                raw.get("decision_schema_sha256"), label="decision_schema_sha256"
            ),
            route_registry_sha256=_require_digest(
                raw.get("route_registry_sha256"), label="route_registry_sha256"
            ),
            source_manifest_sha256=_require_digest(
                raw.get("source_manifest_sha256"), label="source_manifest_sha256"
            ),
            provider_route=dict(route),
            request_settings=dict(settings),
            max_billable_attempts=_require_int(
                raw.get("max_billable_attempts"), label="max_billable_attempts"
            ),
            max_input_bytes_per_attempt=_require_int(
                raw.get("max_input_bytes_per_attempt"),
                label="max_input_bytes_per_attempt",
            ),
            max_output_tokens_per_attempt=_require_int(
                raw.get("max_output_tokens_per_attempt"),
                label="max_output_tokens_per_attempt",
            ),
            format_repairs_per_actor=_require_int(
                raw.get("format_repairs_per_actor"), label="format_repairs_per_actor"
            ),
            transport_retry_backoff_seconds=tuple(
                _require_int(value, label="transport retry backoff", minimum=1)
                for value in backoff
            ),
            request_timeout_seconds=_require_float(
                raw.get("request_timeout_seconds"),
                label="request_timeout_seconds",
            ),
            hard_cost_cap_usd=_require_text(
                raw.get("hard_cost_cap_usd"), label="hard_cost_cap_usd"
            ),
            worst_case_cost_usd=_require_text(
                raw.get("worst_case_cost_usd"), label="worst_case_cost_usd"
            ),
        )


def compute_multiagent_openrouter_contract_id(
    contract: MultiAgentOpenRouterTraceContractV1,
) -> str:
    return _digest(contract.identity_payload())


def build_multiagent_openrouter_contract(
    snapshot: OpenRouterModelSnapshotV1,
    *,
    repo_root: str | Path = _REPO_ROOT,
    manifest: SourceManifestV1 | None = None,
) -> tuple[MultiAgentOpenRouterTraceContractV1, SourceManifestV1]:
    """Freeze the exact 5x5 trace before any provider request is possible."""

    root = Path(repo_root).resolve(strict=True)
    frozen_manifest = manifest or build_multiagent_openrouter_source_manifest(root)
    issues = verify_source_manifest(frozen_manifest, root)
    if issues:
        raise MultiAgentOpenRouterError(
            "source manifest does not match disk: " + "; ".join(issues)
        )
    scenario = build_multiagent_scenario(5)
    population = population_for_scenario(scenario)
    component = frozen_manifest.component_digest
    kwargs: dict[str, Any] = {
        "model": snapshot,
        "request_model_id": snapshot.canonical_slug,
        "actors": _frozen_actors(),
        "scenario_sha256": scenario_content_hash_v2(scenario),
        "initial_world_sha256": str(
            canonical_sha256(to_canonical(materialize_initial_world_tables(scenario)))
        ),
        "logical_schedule_sha256": _digest(
            to_canonical(
                {
                    "initial_events": population.initial_events,
                    "world_events": scenario.world_events,
                }
            )
        ),
        "system_prompt_sha256": component("agent_prompts"),
        "skill_bundle_sha256": component("skill_cards"),
        "decision_schema_sha256": component("business_decision_contract"),
        "route_registry_sha256": component("route_registry"),
        "source_manifest_sha256": _digest(frozen_manifest.to_dict()),
        "provider_route": dict(PROVIDER_ROUTE),
        "request_settings": _thaw(REQUEST_SETTINGS),
        "max_billable_attempts": MAX_BILLABLE_ATTEMPTS,
        "max_input_bytes_per_attempt": MAX_INPUT_BYTES_PER_ATTEMPT,
        "max_output_tokens_per_attempt": MAX_OUTPUT_TOKENS_PER_ATTEMPT,
        "format_repairs_per_actor": FORMAT_REPAIRS_PER_ACTOR,
        "transport_retry_backoff_seconds": TRANSPORT_RETRY_BACKOFF_SECONDS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "hard_cost_cap_usd": _decimal_text(HARD_COST_CAP_USD),
        "worst_case_cost_usd": _decimal_text(worst_case_cost(snapshot)),
    }
    provisional = object.__new__(MultiAgentOpenRouterTraceContractV1)
    for name, value in kwargs.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "schema_version", MULTIAGENT_OPENROUTER_CONTRACT_SCHEMA)
    object.__setattr__(provisional, "contract_id", "")
    contract_id = compute_multiagent_openrouter_contract_id(provisional)
    return MultiAgentOpenRouterTraceContractV1(
        **kwargs,
        contract_id=contract_id,
    ), frozen_manifest


def require_paid_authorization(*, allow_paid: bool) -> None:
    """The only C4 authorization gate; false fails before credentials or POST."""

    if allow_paid is not True:
        raise PaidAuthorizationRequired(
            "paid 5x5 trace requires a fresh explicit --allow-paid authorization"
        )


PAID_TRACE_ARTIFACT_ORDER = (
    "source-manifest.json",
    "contract.json",
    "paid-attempt-ledger.jsonl",
    "hashes-only-decision-log.json",
    "report.json",
)


class ClassifiedOpenRouterTransportError(ChannelTransportError):
    """Sanitized retry classification consumed by the existing Agent policy."""

    _CODES = frozenset(
        {
            "network",
            "provider_429",
            "provider_5xx",
            "provider_timeout",
            "provider_4xx",
            "malformed_provider_response",
        }
    )

    def __init__(self, error_code: str) -> None:
        if error_code not in self._CODES:
            raise ValueError("unknown OpenRouter transport classification")
        super().__init__(f"OpenRouter request failed ({error_code})")
        self.error_code = error_code


def _provider_error_code(exc: BaseException) -> str:
    """Map transport/provider failures to the Agent's bounded retry taxonomy."""

    if isinstance(exc, ProviderResponseError):
        code = exc.code
        error_type = exc.error_type
        if code in {408, 504}:
            return "provider_timeout"
        if code == 429:
            return "provider_429"
        if isinstance(code, int) and 500 <= code <= 599:
            return "provider_5xx"
        if isinstance(code, int) and 400 <= code <= 499:
            return "provider_4xx"
        if error_type in {"provider_timeout", "request_timeout", "timeout"}:
            return "provider_timeout"
        if error_type == "rate_limit_exceeded":
            return "provider_429"
        if error_type in {
            "provider_overloaded",
            "provider_server_error",
            "provider_unavailable",
            "server_error",
            "service_unavailable",
        }:
            return "provider_5xx"
        return "malformed_provider_response"

    cause = exc.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        if cause.code in {408, 504}:
            return "provider_timeout"
        if cause.code == 429:
            return "provider_429"
        if 500 <= cause.code <= 599:
            return "provider_5xx"
        return "provider_4xx"
    if isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(
        cause, (TimeoutError, socket.timeout)
    ):
        return "provider_timeout"
    if isinstance(cause, urllib.error.URLError) and isinstance(
        cause.reason, (TimeoutError, socket.timeout)
    ):
        return "provider_timeout"
    if isinstance(exc, (ConnectionError, ssl.SSLError, http.client.HTTPException)):
        return "network"
    if isinstance(cause, (OSError, http.client.HTTPException)):
        return "network"
    return "malformed_provider_response"


@dataclass(frozen=True, slots=True)
class PaidAttemptRecordV1:
    """One persisted provider attempt with hashes/usage only, never raw text."""

    attempt_index: int
    actor_id: str
    decision_id_sha256: str
    request_sha256: str
    request_input_bytes: int
    reserved_cost_usd: str
    outcome: str
    error_code: str | None = None
    response_model: str | None = None
    provider_sha256: str | None = None
    generation_id_sha256: str | None = None
    response_content_sha256: str | None = None
    response_chars: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: str = "0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cwe.multiagent-openrouter-attempt.v1",
            "attempt_index": self.attempt_index,
            "actor_id": self.actor_id,
            "decision_id_sha256": self.decision_id_sha256,
            "request_sha256": self.request_sha256,
            "request_input_bytes": self.request_input_bytes,
            "reserved_cost_usd": self.reserved_cost_usd,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "response_model": self.response_model,
            "provider_sha256": self.provider_sha256,
            "generation_id_sha256": self.generation_id_sha256,
            "response_content_sha256": self.response_content_sha256,
            "response_chars": self.response_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
        }


class PaidTraceLedgerV1:
    """Thread-safe, append-only paid-attempt ledger with a pre-call reserve."""

    def __init__(
        self,
        *,
        path: str | Path,
        contract: MultiAgentOpenRouterTraceContractV1,
    ) -> None:
        self.path = Path(path)
        self.contract = contract
        self._lock = threading.Lock()
        self._reservations: dict[int, dict[str, Any]] = {}
        self._records: list[PaidAttemptRecordV1] = []
        self._reserved_cost = Decimal("0")
        self._actual_cost = Decimal("0")
        if self.path.exists():
            raise MultiAgentOpenRouterError("paid attempt ledger already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)

    def _append(self, value: Mapping[str, Any]) -> None:
        payload = _canonical_bytes(value) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(
        self,
        *,
        actor_id: str,
        decision_id: str | None,
        request_payload: Mapping[str, Any],
        request_input_bytes: int,
    ) -> int:
        with self._lock:
            if request_input_bytes > self.contract.max_input_bytes_per_attempt:
                raise MultiAgentOpenRouterError(
                    "provider prompt exceeds the frozen input-byte bound"
                )
            attempt_index = len(self._reservations) + 1
            if attempt_index > self.contract.max_billable_attempts:
                raise MultiAgentOpenRouterError("paid provider attempt cap exhausted")
            reserved = (
                Decimal(request_input_bytes) * self.contract.model.prompt_price
                + Decimal(self.contract.max_output_tokens_per_attempt)
                * (
                    self.contract.model.completion_price
                    + self.contract.model.reasoning_price
                )
                + self.contract.model.fixed_request_price
            )
            if self._reserved_cost + reserved > _decimal(
                self.contract.hard_cost_cap_usd, label="hard cost cap"
            ):
                raise MultiAgentOpenRouterError("paid cost reserve would exceed hard cap")
            request_sha256 = _digest(request_payload)
            decision_sha256 = hashlib.sha256(
                (decision_id or "none").encode("utf-8")
            ).hexdigest()
            self._reserved_cost += reserved
            self._reservations[attempt_index] = {
                "actor_id": actor_id,
                "decision_id_sha256": decision_sha256,
                "request_sha256": request_sha256,
                "request_input_bytes": request_input_bytes,
                "reserved_cost_usd": _decimal_text(reserved),
            }
            self._append(
                {
                    "schema_version": "cwe.multiagent-openrouter-ledger-event.v1",
                    "event": "reserved",
                    "attempt_index": attempt_index,
                    **self._reservations[attempt_index],
                }
            )
            return attempt_index

    def finish(
        self,
        attempt_index: int,
        *,
        outcome: str,
        error_code: str | None = None,
        response: Mapping[str, Any] | None = None,
    ) -> PaidAttemptRecordV1:
        with self._lock:
            reservation = self._reservations.get(attempt_index)
            if reservation is None or any(
                row.attempt_index == attempt_index for row in self._records
            ):
                raise MultiAgentOpenRouterError("paid attempt finish is ambiguous")
            fields = (
                _safe_response_accounting(response, contract=self.contract)
                if response is not None
                else {}
            )
            cost = _decimal(fields.get("cost_usd", "0"), label="provider response cost")
            reserved = _decimal(
                reservation["reserved_cost_usd"], label="reserved provider cost"
            )
            if cost > reserved:
                raise MultiAgentOpenRouterError(
                    "provider response cost exceeded its frozen pre-call reserve"
                )
            if self._actual_cost + cost > _decimal(
                self.contract.hard_cost_cap_usd, label="hard cost cap"
            ):
                raise MultiAgentOpenRouterError("actual provider cost exceeded hard cap")
            record = PaidAttemptRecordV1(
                attempt_index=attempt_index,
                actor_id=str(reservation["actor_id"]),
                decision_id_sha256=str(reservation["decision_id_sha256"]),
                request_sha256=str(reservation["request_sha256"]),
                request_input_bytes=int(reservation["request_input_bytes"]),
                reserved_cost_usd=str(reservation["reserved_cost_usd"]),
                outcome=outcome,
                error_code=error_code,
                **fields,
            )
            self._actual_cost += cost
            self._records.append(record)
            self._append(
                {
                    "schema_version": "cwe.multiagent-openrouter-ledger-event.v1",
                    "event": "finished",
                    **record.to_dict(),
                }
            )
            return record

    @property
    def records(self) -> tuple[PaidAttemptRecordV1, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def provider_attempts(self) -> int:
        with self._lock:
            return len(self._reservations)

    def actor_provider_calls(self) -> tuple[tuple[str, int], ...]:
        with self._lock:
            counts: dict[str, int] = {}
            for reservation in self._reservations.values():
                actor_id = str(reservation["actor_id"])
                counts[actor_id] = counts.get(actor_id, 0) + 1
            return tuple(sorted(counts.items()))

    @property
    def actual_cost(self) -> Decimal:
        with self._lock:
            return self._actual_cost

    @property
    def billed_cost_upper_bound(self) -> Decimal:
        """Exact response cost plus a conservative reserve for unknown attempts."""

        with self._lock:
            total = Decimal("0")
            for record in self._records:
                if record.outcome == "response":
                    total += _decimal(record.cost_usd, label="recorded provider cost")
                else:
                    total += _decimal(
                        record.reserved_cost_usd,
                        label="unknown-attempt provider reserve",
                    )
            return total


def _safe_response_accounting(
    response: Mapping[str, Any] | None,
    *,
    contract: MultiAgentOpenRouterTraceContractV1,
) -> dict[str, Any]:
    if response is None:
        return {}
    response_model = response.get("model")
    # OpenRouter accepts the dated canonical slug used by the frozen POST, but
    # normalizes the successful response's ``model`` field to the catalog
    # ``id``.  Both strings come from the same revalidated catalog row and are
    # frozen in the contract.  No third model identity is accepted.
    if response_model not in {
        contract.request_model_id,
        contract.model.model_id,
    }:
        raise MultiAgentOpenRouterError("provider returned a different model")
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise MultiAgentOpenRouterError("provider response has no usage accounting")
    prompt_tokens = _require_int(usage.get("prompt_tokens"), label="prompt_tokens")
    completion_tokens = _require_int(
        usage.get("completion_tokens"), label="completion_tokens"
    )
    if completion_tokens > contract.max_output_tokens_per_attempt:
        raise MultiAgentOpenRouterError(
            "provider completion exceeded the frozen output-token bound"
        )
    details = usage.get("completion_tokens_details")
    reasoning_tokens = 0
    if details is not None:
        if not isinstance(details, Mapping):
            raise MultiAgentOpenRouterError(
                "provider completion token details must be an object"
            )
        raw_reasoning = details.get("reasoning_tokens", 0)
        reasoning_tokens = _require_int(raw_reasoning, label="reasoning_tokens")
    raw_cost = usage.get("cost")
    if raw_cost is None:
        raise MultiAgentOpenRouterError(
            "provider response has no authoritative cost accounting"
        )
    cost = _decimal(raw_cost, label="usage.cost")
    provider = response.get("provider")
    generation_id = response.get("id")
    choice_content: str | None = None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            choice_content = str(message["content"])
    return {
        "response_model": str(response_model),
        "provider_sha256": (
            hashlib.sha256(str(provider).encode("utf-8")).hexdigest()
            if isinstance(provider, str) and provider
            else None
        ),
        "generation_id_sha256": (
            hashlib.sha256(str(generation_id).encode("utf-8")).hexdigest()
            if isinstance(generation_id, str) and generation_id
            else None
        ),
        "response_content_sha256": (
            hashlib.sha256(choice_content.encode("utf-8")).hexdigest()
            if choice_content is not None
            else None
        ),
        "response_chars": len(choice_content) if choice_content is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": _decimal_text(cost),
    }


class BudgetedOpenRouterChannelV1(OpenAIChannel):
    """Exact-model OpenAI-compatible channel with an on-disk pre-call budget."""

    def __init__(
        self,
        *,
        actor_id: str,
        api_key: str,
        contract: MultiAgentOpenRouterTraceContractV1,
        ledger: PaidTraceLedgerV1,
    ) -> None:
        self.actor_id = actor_id
        self.contract = contract
        self.ledger = ledger
        self._active_decision_id: str | None = None
        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            model=contract.request_model_id,
            timeout=contract.request_timeout_seconds,
            provider_options=_thaw(contract.provider_route),
            business_response_format=True,
            extra_headers={"X-OpenRouter-Title": "CommerceWorld 5x5 Trace"},
        )

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        if self._active_decision_id is not None:
            raise MultiAgentOpenRouterError("provider channel cannot be re-entered")
        self._active_decision_id = decision_id or "none"
        try:
            return super().complete_business_decision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                decision_id=decision_id,
            )
        except ClassifiedOpenRouterTransportError:
            raise
        except (ProviderResponseError, ChannelTransportError) as exc:
            raise ClassifiedOpenRouterTransportError(_provider_error_code(exc)) from exc
        finally:
            self._active_decision_id = None

    def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload.update(_thaw(self.contract.request_settings))
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise MultiAgentOpenRouterError("provider request messages are invalid")
        prompt_bytes = 0
        for message in messages:
            if not isinstance(message, Mapping) or not isinstance(
                message.get("content"), str
            ):
                raise MultiAgentOpenRouterError("provider request message is invalid")
            prompt_bytes += len(str(message["content"]).encode("utf-8"))
        attempt_index = self.ledger.reserve(
            actor_id=self.actor_id,
            decision_id=self._active_decision_id,
            request_payload=payload,
            request_input_bytes=prompt_bytes,
        )
        try:
            response: dict[str, Any] = super()._post_chat_completions(payload)
            # OpenRouter may encode an upstream failure in an HTTP-200 body.
            # Classify it before the response can be persisted as successful.
            _raise_for_in_band_provider_error(response)
            self.ledger.finish(attempt_index, outcome="response", response=response)
            return response
        except MultiAgentOpenRouterError:
            # A response can fail exact-model/usage/cost validation after the
            # paid call.  Persist a terminal error without attempting to parse
            # or retain its body, then stop the shared market.
            self.ledger.finish(
                attempt_index,
                outcome="infrastructure_error",
                error_code="malformed_provider_response",
            )
            raise
        except (ProviderResponseError, ChannelTransportError) as exc:
            code = _provider_error_code(exc)
            self.ledger.finish(
                attempt_index,
                outcome="transport_error",
                error_code=code,
            )
            raise ClassifiedOpenRouterTransportError(code) from exc


_FACTORY_NEW = "new"
_FACTORY_CLAIMED = "claimed"
_FACTORY_SEALED = "sealed"


class OpenRouterChannelFactoryV1:
    """One exact-model channel per actor and exactly one shared Episode."""

    def __init__(
        self,
        *,
        api_key: str,
        contract: MultiAgentOpenRouterTraceContractV1,
        ledger: PaidTraceLedgerV1,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise MultiAgentOpenRouterError("OpenRouter API key is required")
        self._api_key = api_key
        self.contract = contract
        self.ledger = ledger
        self._actors = {actor.actor_id: actor.role for actor in contract.actors}
        self._channels: dict[str, BudgetedOpenRouterChannelV1] = {}
        self._state = _FACTORY_NEW
        self._episode_id: str | None = None

    @contextmanager
    def claim_for_episode(self, episode_id: str) -> Iterator["OpenRouterChannelFactoryV1"]:
        if self._state != _FACTORY_NEW or not isinstance(episode_id, str) or not episode_id:
            raise MultiAgentOpenRouterError("paid channel factory is already used or invalid")
        self._state = _FACTORY_CLAIMED
        self._episode_id = episode_id
        try:
            yield self
        finally:
            self._state = _FACTORY_SEALED

    def __call__(self, agent_id: str, role: str) -> BudgetedOpenRouterChannelV1:
        if self._state != _FACTORY_CLAIMED:
            raise MultiAgentOpenRouterError("paid channel factory is not claimed")
        if self._actors.get(agent_id) != role:
            raise MultiAgentOpenRouterError("paid channel actor id/role is not frozen")
        existing = self._channels.get(agent_id)
        if existing is not None:
            return existing
        channel = BudgetedOpenRouterChannelV1(
            actor_id=agent_id,
            api_key=self._api_key,
            contract=self.contract,
            ledger=self.ledger,
        )
        self._channels[agent_id] = channel
        return channel

    @property
    def lifecycle_state(self) -> str:
        return self._state

    def actor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._channels))


@dataclass(frozen=True, slots=True)
class MultiAgentLLMTraceReportV1:
    """Capability-free operational report for the one paid shared-market trace."""

    contract_id: str
    valid: bool
    exact_model_id: str
    provider_attempts: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_cost_usd: str
    actor_provider_calls: tuple[tuple[str, int], ...]
    joined_business_decisions: int
    world_commits: int
    final_world_sha256: str | None
    replay_verified: bool
    stop_reason: str
    diagnostics: Mapping[str, Any]
    report_id: str
    schema_version: str = MULTIAGENT_LLM_TRACE_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MULTIAGENT_LLM_TRACE_REPORT_SCHEMA:
            raise MultiAgentOpenRouterError("unsupported paid trace report schema")
        _require_digest(self.contract_id, label="contract_id")
        if self.exact_model_id != DEFAULT_OPENROUTER_MODEL:
            raise MultiAgentOpenRouterError("paid report model differs from the exact model")
        for label, value in (
            ("provider_attempts", self.provider_attempts),
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("reasoning_tokens", self.reasoning_tokens),
            ("joined_business_decisions", self.joined_business_decisions),
            ("world_commits", self.world_commits),
        ):
            _require_int(value, label=label)
        if self.provider_attempts > MAX_BILLABLE_ATTEMPTS:
            raise MultiAgentOpenRouterError("provider attempt cap was exceeded")
        cost = _decimal(self.total_cost_usd, label="total_cost_usd")
        if cost > HARD_COST_CAP_USD:
            raise MultiAgentOpenRouterError("paid report exceeds the hard cost cap")
        actor_ids = tuple(actor_id for actor_id, _ in self.actor_provider_calls)
        if actor_ids != tuple(sorted(set(actor_ids))):
            raise MultiAgentOpenRouterError("actor provider coverage must be unique and sorted")
        if len(actor_ids) > 10:
            raise MultiAgentOpenRouterError("actor provider coverage exceeds ten actors")
        for actor_id, calls in self.actor_provider_calls:
            if actor_id.split(":", 1)[0] not in _SAFE_ACTOR_ROLES:
                raise MultiAgentOpenRouterError("actor provider coverage id is invalid")
            _require_int(calls, label=f"actor provider calls[{actor_id}]")
        if sum(calls for _, calls in self.actor_provider_calls) != self.provider_attempts:
            raise MultiAgentOpenRouterError(
                "actor provider calls do not sum to provider_attempts"
            )
        if self.final_world_sha256 is not None:
            _require_digest(self.final_world_sha256, label="final_world_sha256")
        if self.stop_reason not in _SAFE_STOP_REASONS:
            raise MultiAgentOpenRouterError("paid trace stop_reason is not a known enum")
        if not isinstance(self.diagnostics, Mapping):
            raise MultiAgentOpenRouterError("report diagnostics must be an object")
        _validate_sanitized_diagnostics(self.diagnostics)
        if self.valid:
            if len(actor_ids) != 10 or any(
                calls < 1 for _, calls in self.actor_provider_calls
            ):
                raise MultiAgentOpenRouterError(
                    "a valid paid trace requires real provider coverage for all ten actors"
                )
            if not self.replay_verified or self.final_world_sha256 is None:
                raise MultiAgentOpenRouterError(
                    "a valid paid trace requires a verified replay and final World digest"
                )
        expected = compute_multiagent_llm_trace_report_id(self)
        if self.report_id != expected:
            raise MultiAgentOpenRouterError("paid trace report_id is inconsistent")
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "valid": self.valid,
            "exact_model_id": self.exact_model_id,
            "provider_attempts": self.provider_attempts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_cost_usd": self.total_cost_usd,
            "actor_provider_calls": [list(row) for row in self.actor_provider_calls],
            "joined_business_decisions": self.joined_business_decisions,
            "world_commits": self.world_commits,
            "final_world_sha256": self.final_world_sha256,
            "replay_verified": self.replay_verified,
            "stop_reason": self.stop_reason,
            "diagnostics": _thaw(self.diagnostics),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"report_id": self.report_id, **self.identity_payload()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MultiAgentLLMTraceReportV1":
        required = (
            "schema_version",
            "report_id",
            "contract_id",
            "valid",
            "exact_model_id",
            "provider_attempts",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_cost_usd",
            "actor_provider_calls",
            "joined_business_decisions",
            "world_commits",
            "final_world_sha256",
            "replay_verified",
            "stop_reason",
            "diagnostics",
        )
        _require_exact_keys(raw, required=required, label="paid trace report")
        coverage = raw.get("actor_provider_calls")
        diagnostics = raw.get("diagnostics")
        if not isinstance(coverage, list):
            raise MultiAgentOpenRouterError("actor_provider_calls must be a list")
        parsed_coverage: list[tuple[str, int]] = []
        for row in coverage:
            if not isinstance(row, list) or len(row) != 2:
                raise MultiAgentOpenRouterError(
                    "actor_provider_calls entries must be [actor_id, calls]"
                )
            parsed_coverage.append(
                (
                    _require_text(row[0], label="actor provider id"),
                    _require_int(row[1], label="actor provider calls"),
                )
            )
        if not isinstance(diagnostics, Mapping):
            raise MultiAgentOpenRouterError("report diagnostics must be an object")
        valid = raw.get("valid")
        replay_verified = raw.get("replay_verified")
        if not isinstance(valid, bool) or not isinstance(replay_verified, bool):
            raise MultiAgentOpenRouterError("report validity fields must be booleans")
        final_digest = raw.get("final_world_sha256")
        if final_digest is not None:
            final_digest = _require_digest(
                final_digest, label="final_world_sha256"
            )
        return cls(
            schema_version=_require_text(
                raw.get("schema_version"), label="schema_version"
            ),
            report_id=_require_digest(raw.get("report_id"), label="report_id"),
            contract_id=_require_digest(raw.get("contract_id"), label="contract_id"),
            valid=valid,
            exact_model_id=_require_text(
                raw.get("exact_model_id"), label="exact_model_id"
            ),
            provider_attempts=_require_int(
                raw.get("provider_attempts"), label="provider_attempts"
            ),
            prompt_tokens=_require_int(
                raw.get("prompt_tokens"), label="prompt_tokens"
            ),
            completion_tokens=_require_int(
                raw.get("completion_tokens"), label="completion_tokens"
            ),
            reasoning_tokens=_require_int(
                raw.get("reasoning_tokens"), label="reasoning_tokens"
            ),
            total_cost_usd=_require_text(
                raw.get("total_cost_usd"), label="total_cost_usd"
            ),
            actor_provider_calls=tuple(parsed_coverage),
            joined_business_decisions=_require_int(
                raw.get("joined_business_decisions"),
                label="joined_business_decisions",
            ),
            world_commits=_require_int(
                raw.get("world_commits"), label="world_commits"
            ),
            final_world_sha256=final_digest,
            replay_verified=replay_verified,
            stop_reason=_require_text(raw.get("stop_reason"), label="stop_reason"),
            diagnostics=dict(diagnostics),
        )


def compute_multiagent_llm_trace_report_id(report: MultiAgentLLMTraceReportV1) -> str:
    return _digest(report.identity_payload())


def build_multiagent_llm_trace_report(**fields: Any) -> MultiAgentLLMTraceReportV1:
    """Build a report whose stable identity is derived, never caller-declared."""

    provisional = object.__new__(MultiAgentLLMTraceReportV1)
    for name, value in fields.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "schema_version", MULTIAGENT_LLM_TRACE_REPORT_SCHEMA)
    object.__setattr__(provisional, "report_id", "")
    report_id = compute_multiagent_llm_trace_report_id(provisional)
    return MultiAgentLLMTraceReportV1(**fields, report_id=report_id)


@dataclass(frozen=True, slots=True)
class PersistedMultiAgentLLMTraceV1:
    report: MultiAgentLLMTraceReportV1
    artifacts_dir: str
    episode_dir: str
    artifact_index: Mapping[str, Any]


def _termination_state(episode_dir: Path) -> tuple[bool, str]:
    path = episode_dir / EPISODE_TERMINATION_ARTIFACT
    if not path.exists():
        return True, "completed"
    scoreable = load_verified_scoreable_termination(episode_dir)
    if scoreable is None:
        return False, "infrastructure_error"
    stop_reason = scoreable.get("stop_reason")
    if stop_reason not in _SAFE_STOP_REASONS:
        raise MultiAgentOpenRouterError("termination stop reason is not recognized")
    return True, str(stop_reason)


def _tracker_summary(
    evidence: RuntimeEvidenceBundleV2,
    actor_ids: tuple[str, ...],
) -> tuple[bool, int, int, int]:
    record_count = 0
    complete_count = 0
    joined_choices = 0
    for actor_id in actor_ids:
        verdict = verify_tracker_evidence(
            evidence,
            evaluated_actor_id=actor_id,
            strict_ideal=False,
        )
        if not verdict.verified:
            return False, record_count, complete_count, joined_choices
        record_count += verdict.record_count
        complete_count += verdict.complete_record_count
        joined_choices += len(
            verified_model_business_choices(
                evidence,
                evaluated_actor_id=actor_id,
                strict_ideal=False,
            )
        )
    return True, record_count, complete_count, joined_choices


def _provider_tracker_join(
    evidence: RuntimeEvidenceBundleV2,
    records: Sequence[PaidAttemptRecordV1],
) -> bool:
    tracker_decision_hashes = {
        hashlib.sha256(str(row["decision_id"]).encode("utf-8")).hexdigest()
        for row in evidence.trace_rows
        if isinstance(row.get("decision_id"), str) and row.get("decision_id")
    }
    response_records = [row for row in records if row.outcome == "response"]
    return bool(response_records) and all(
        row.decision_id_sha256 in tracker_decision_hashes for row in response_records
    )


def _platform_action_is_read_only(action_kind: object) -> bool:
    """Return the platform actions that must acknowledge without a commit.

    CommerceWorld's actor-facing read operations use the ``commerce.read_*``
    namespace.  Aggregator search predates that naming rule and is the one
    explicit read-only exception.  This predicate deliberately describes the
    operation class instead of copying every domain's read route list.
    """

    return isinstance(action_kind, str) and (
        action_kind == "commerce.search" or action_kind.startswith("commerce.read_")
    )


def _model_choice_causal_join(
    evidence: RuntimeEvidenceBundleV2,
    actor_ids: tuple[str, ...],
) -> tuple[bool, int]:
    """Join Platform-bound model choices to decisions and World commits.

    ``verified_model_business_choices`` has already joined every choice to its
    exact Agent-compiled audited envelope.  Agent-to-agent messages do not
    manufacture a Platform acknowledgement.  For the remaining Platform-bound
    choices, rejection is legitimate and has no World commit. Accepted writes
    must have one unique actor/idempotency-bound commit; accepted actor-facing
    reads must have no commit because their audited Platform response is the
    terminal causal evidence.
    """

    choices = tuple(
        choice
        for actor_id in actor_ids
        for choice in verified_model_business_choices(
            evidence,
            evaluated_actor_id=actor_id,
            strict_ideal=False,
        )
    )
    exchanges = evidence.platform_exchanges()
    by_request: dict[str, list[Any]] = {}
    for exchange in exchanges:
        request_id = exchange.request.get("msg_id")
        if isinstance(request_id, str):
            by_request.setdefault(request_id, []).append(exchange)
    commits_by_operation: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for commit in evidence.world_events:
        actor_id = commit.get("actor_id")
        idempotency_key = commit.get("idempotency_key")
        if isinstance(actor_id, str) and isinstance(idempotency_key, str):
            commits_by_operation.setdefault((actor_id, idempotency_key), []).append(
                commit
            )

    for choice in choices:
        if not choice.expects_platform_exchange:
            continue
        matches = by_request.get(choice.emitted_msg_id, [])
        if len(matches) != 1:
            return False, len(choices)
        decision = matches[0].decision
        actor_id = decision.get("actor_id")
        idempotency_key = decision.get("idempotency_key")
        outcome = decision.get("decision")
        if not isinstance(actor_id, str) or not isinstance(idempotency_key, str):
            return False, len(choices)
        commits = commits_by_operation.get((actor_id, idempotency_key), [])
        if outcome == "accepted":
            if _platform_action_is_read_only(decision.get("action_kind")):
                if commits:
                    return False, len(choices)
            elif len(commits) != 1:
                return False, len(choices)
        elif outcome == "rejected":
            if commits:
                return False, len(choices)
        else:
            return False, len(choices)
    return bool(choices), len(choices)


def _market_diagnostics(evidence: RuntimeEvidenceBundleV2) -> dict[str, Any]:
    tables = evidence.final_world.get("tables")
    if not isinstance(tables, Mapping):
        return {"final_table_shape_valid": False}
    orders = tables.get("orders")
    ledger = tables.get("ledger")
    inventory = tables.get("inventory")
    order_rows = orders if isinstance(orders, list) else []
    ledger_rows = ledger if isinstance(ledger, list) else []
    inventory_rows = inventory if isinstance(inventory, Mapping) else {}
    after_sales_order = next(
        (
            row
            for row in order_rows
            if isinstance(row, Mapping)
            and row.get("order_id") == "order:cwenv-m2m-after-sales"
        ),
        None,
    )
    return {
        "final_table_shape_valid": True,
        "order_count": len(order_rows),
        "ledger_entry_count": len(ledger_rows),
        "inventory_row_count": len(inventory_rows),
        "reserved_inventory_units": sum(
            int(row.get("qty_reserved", 0))
            for row in inventory_rows.values()
            if isinstance(row, Mapping)
            and isinstance(row.get("qty_reserved", 0), int)
            and not isinstance(row.get("qty_reserved", 0), bool)
        ),
        "after_sales_state": (
            after_sales_order.get("state")
            if isinstance(after_sales_order, Mapping)
            else None
        ),
    }


def _write_hashes_only_decision_log(
    path: Path,
    records: Sequence[PaidAttemptRecordV1],
) -> None:
    write_json_artifact(
        {
            "schema_version": "cwe.multiagent-openrouter-decision-log.v1",
            "storage": "hashes_and_usage_only",
            "records": [record.to_dict() for record in records],
        },
        path,
    )


def run_persisted_multiagent_llm_trace(
    *,
    allow_paid: bool,
    api_key: str,
    out_root: str | Path,
    artifacts_dir: str | Path,
    repo_root: str | Path = _REPO_ROOT,
    snapshot: OpenRouterModelSnapshotV1 | None = None,
) -> PersistedMultiAgentLLMTraceV1:
    """Execute C5 once; no resume or source-drift stitching is supported."""

    require_paid_authorization(allow_paid=allow_paid)
    if not isinstance(api_key, str) or not api_key.strip():
        raise MultiAgentOpenRouterError("OpenRouter API key is required")
    root = Path(repo_root).resolve(strict=True)
    output = Path(out_root)
    artifacts = Path(artifacts_dir)
    if (output.exists() and any(output.iterdir())) or (
        artifacts.exists() and any(artifacts.iterdir())
    ):
        raise MultiAgentOpenRouterError(
            "paid trace requires fresh empty output and artifact directories"
        )
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    # The catalog is re-read immediately before contract creation unless an
    # exact validated snapshot is injected by a test.  No model alias or
    # alternative model list exists on this path.
    current_snapshot = snapshot or fetch_openrouter_model_snapshot(api_key=api_key)
    contract, manifest = build_multiagent_openrouter_contract(
        current_snapshot,
        repo_root=root,
    )
    write_json_artifact(manifest.to_dict(), artifacts / "source-manifest.json")
    write_json_artifact(contract.to_dict(), artifacts / "contract.json")

    ledger = PaidTraceLedgerV1(
        path=artifacts / "paid-attempt-ledger.jsonl",
        contract=contract,
    )
    factory = OpenRouterChannelFactoryV1(
        api_key=api_key,
        contract=contract,
        ledger=ledger,
    )
    scenario = build_multiagent_scenario(5)
    episode_dir = output / scenario.scenario_id
    evidence: RuntimeEvidenceBundleV2 | None = None
    replay_verified = False
    termination_valid = False
    stop_reason = "infrastructure_error"
    tracker_verified = False
    tracker_records = 0
    complete_tracker_records = 0
    joined_choices = 0
    exact_provider_tracker_join = False
    exact_platform_world_join = False
    model_choice_causal_join = False
    sources_declared = False
    caught_error_type: str | None = None
    accepted_platform_operations = 0
    rejected_platform_operations = 0
    loaded_source_count = 0
    unlisted_source_count = 0
    try:
        with factory.claim_for_episode(scenario.scenario_id):
            EpisodeBatch(
                scenarios=[scenario],
                channels=factory,
                out_root=output,
                strict_skill_selection=True,
                strict_tracker_capture=True,
            ).run()
        evidence = RuntimeEvidenceBundleV2.load(episode_dir)
        from episode.replay import verify_episode_replay

        replay_verified = bool(
            verify_episode_replay(episode_dir, strict=True).replay_ok
        )
        termination_valid, stop_reason = _termination_state(episode_dir)
        actor_ids = tuple(actor.actor_id for actor in contract.actors)
        (
            tracker_verified,
            tracker_records,
            complete_tracker_records,
            joined_choices,
        ) = _tracker_summary(evidence, actor_ids)
        exact_provider_tracker_join = _provider_tracker_join(
            evidence, ledger.records
        )
        model_choice_causal_join, causal_choice_count = _model_choice_causal_join(
            evidence, actor_ids
        )
        if causal_choice_count != joined_choices:
            raise MultiAgentOpenRouterError(
                "model-choice causal join count contradicts Tracker extraction"
            )
        accepted = evidence.platform_exchanges(decision="accepted")
        rejected = evidence.platform_exchanges(decision="rejected")
        accepted_platform_operations = len(accepted)
        rejected_platform_operations = len(rejected)
        # platform_exchanges() builds the Runtime exact-join context over all
        # accepted/rejected decisions and every World commit.  Returning means
        # the complete Platform/World graph was unique and consistent.
        exact_platform_world_join = True
        loaded = repo_source_paths(iter_loaded_module_files(), root)
        loaded_source_count = len(loaded)
        unlisted = unlisted_loaded_sources(manifest, loaded)
        unlisted_source_count = len(unlisted)
        sources_declared = not unlisted
    except Exception as exc:  # noqa: BLE001 - persist a sanitized invalid report
        caught_error_type = type(exc).__name__

    records = ledger.records
    actor_calls = ledger.actor_provider_calls()
    provider_attempts = ledger.provider_attempts
    all_actor_calls = (
        tuple(actor_id for actor_id, _ in actor_calls)
        == tuple(actor.actor_id for actor in contract.actors)
        and all(calls >= 1 for _, calls in actor_calls)
    )
    initial_matches = bool(
        evidence is not None
        and evidence.initial_digest == contract.initial_world_sha256
        and scenario_content_hash_v2(scenario) == contract.scenario_sha256
    )
    response_models_exact = all(
        record.response_model
        in {contract.request_model_id, contract.model.model_id}
        for record in records
        if record.outcome == "response"
    )
    every_reservation_finished = provider_attempts == len(records)
    valid = all(
        (
            evidence is not None,
            replay_verified,
            termination_valid,
            tracker_verified,
            exact_provider_tracker_join,
            exact_platform_world_join,
            model_choice_causal_join,
            sources_declared,
            initial_matches,
            response_models_exact,
            every_reservation_finished,
            all_actor_calls,
            factory.lifecycle_state == _FACTORY_SEALED,
            caught_error_type is None,
        )
    )
    diagnostics: dict[str, Any] = {
        "checks": {
            "all_actor_provider_coverage": all_actor_calls,
            "every_reservation_finished": every_reservation_finished,
            "exact_platform_world_join": exact_platform_world_join,
            "exact_provider_tracker_join": exact_provider_tracker_join,
            "model_choice_causal_join": model_choice_causal_join,
            "initial_world_and_scenario_frozen": initial_matches,
            "provider_model_exact": response_models_exact,
            "replay_verified": replay_verified,
            "sources_declared": sources_declared,
            "termination_valid": termination_valid,
            "tracker_verified": tracker_verified,
        },
        "accepted_platform_operations": accepted_platform_operations,
        "rejected_platform_operations": rejected_platform_operations,
        "tracker_record_count": tracker_records,
        "complete_tracker_record_count": complete_tracker_records,
        "loaded_source_count": loaded_source_count,
        "unlisted_source_count": unlisted_source_count,
        "caught_error_type": caught_error_type,
        "cost_accounting_exact": all(
            record.outcome == "response" for record in records
        ),
    }
    if evidence is not None:
        diagnostics["market"] = _market_diagnostics(evidence)
    report = build_multiagent_llm_trace_report(
        contract_id=contract.contract_id,
        valid=valid,
        exact_model_id=contract.model.model_id,
        provider_attempts=provider_attempts,
        prompt_tokens=sum(record.prompt_tokens for record in records),
        completion_tokens=sum(record.completion_tokens for record in records),
        reasoning_tokens=sum(record.reasoning_tokens for record in records),
        total_cost_usd=_decimal_text(ledger.billed_cost_upper_bound),
        actor_provider_calls=actor_calls,
        joined_business_decisions=joined_choices,
        world_commits=len(evidence.world_events) if evidence is not None else 0,
        final_world_sha256=evidence.final_digest if evidence is not None else None,
        replay_verified=replay_verified,
        stop_reason=stop_reason,
        diagnostics=diagnostics,
    )
    _write_hashes_only_decision_log(
        artifacts / "hashes-only-decision-log.json", records
    )
    write_json_artifact(report.to_dict(), artifacts / "report.json")
    index = build_artifact_index(artifacts, PAID_TRACE_ARTIFACT_ORDER)
    write_json_artifact(index, artifacts / ARTIFACT_INDEX_NAME)
    return PersistedMultiAgentLLMTraceV1(
        report=report,
        artifacts_dir=str(artifacts),
        episode_dir=str(episode_dir),
        artifact_index=index,
    )


__all__ = [
    "BudgetedOpenRouterChannelV1",
    "ClassifiedOpenRouterTransportError",
    "DEFAULT_OPENROUTER_MODEL",
    "EXPECTED_CANONICAL_SLUG",
    "FORMAT_REPAIRS_PER_ACTOR",
    "HARD_COST_CAP_USD",
    "MAX_BILLABLE_ATTEMPTS",
    "MAX_INPUT_BYTES_PER_ATTEMPT",
    "MAX_OUTPUT_TOKENS_PER_ATTEMPT",
    "MULTIAGENT_LLM_TRACE_REPORT_SCHEMA",
    "MULTIAGENT_OPENROUTER_CONTRACT_SCHEMA",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODELS_ENDPOINT",
    "OpenRouterChannelFactoryV1",
    "PAID_TRACE_ARTIFACT_ORDER",
    "PROVIDER_ROUTE",
    "REQUEST_SETTINGS",
    "REQUEST_TIMEOUT_SECONDS",
    "TRANSPORT_RETRY_BACKOFF_SECONDS",
    "FrozenTraceActorV1",
    "MultiAgentLLMTraceReportV1",
    "MultiAgentOpenRouterError",
    "MultiAgentOpenRouterTraceContractV1",
    "OpenRouterModelSnapshotV1",
    "PaidAttemptRecordV1",
    "PaidAuthorizationRequired",
    "PaidTraceLedgerV1",
    "PersistedMultiAgentLLMTraceV1",
    "build_multiagent_openrouter_contract",
    "build_multiagent_openrouter_source_manifest",
    "build_multiagent_llm_trace_report",
    "compute_multiagent_llm_trace_report_id",
    "compute_multiagent_openrouter_contract_id",
    "fetch_openrouter_model_snapshot",
    "model_snapshot_from_catalog",
    "require_paid_authorization",
    "run_persisted_multiagent_llm_trace",
    "worst_case_cost",
]
