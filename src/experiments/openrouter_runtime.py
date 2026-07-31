"""OpenRouter channel construction for public ACWorld benchmark runs."""

from __future__ import annotations

import http.client
import socket
import ssl
import urllib.error
from typing import Any

from agents.inference import (
    ChannelTransportError,
    InferenceChannel,
    OpenAIChannel,
    ProviderResponseError,
)
from experiments.benchmark_executor import (
    ExecutionConfigurationError,
    BenchmarkExecutor,
)
from experiments.benchmark_plan import MAIN_MODELS_V2


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TIMEOUT_SECONDS = 240.0
FATAL_RUN_ERRORS = (
    "BenchmarkChannelContractError",
    "PreProviderCallAccountingError",
    "SkillSelectionInfrastructureError",
    "TrackerCaptureInfrastructureError",
)

_PROVIDER_429_TYPES = frozenset({"rate_limit_exceeded"})
_PROVIDER_TIMEOUT_TYPES = frozenset(
    {"provider_timeout", "request_timeout", "timeout"}
)
_PROVIDER_5XX_TYPES = frozenset(
    {
        "provider_overloaded",
        "provider_server_error",
        "provider_unavailable",
        "server",
        "server_error",
        "service_unavailable",
        "unmapped",
    }
)
_PROVIDER_4XX_TYPES = frozenset(
    {
        "authentication",
        "authentication_error",
        "authorization_error",
        "bad_request",
        "content_policy",
        "content_policy_error",
        "content_policy_violation",
        "forbidden",
        "invalid_request",
        "invalid_request_error",
        "permission_error",
        "request",
        "request_error",
        "unauthorized",
    }
)


class BenchmarkRunError(RuntimeError):
    """A public benchmark run rejected its configuration."""


class BenchmarkChannelContractError(BenchmarkRunError, ExecutionConfigurationError):
    """The evaluated channel does not provide the required Agent interface."""


class ClassifiedTransportError(ChannelTransportError):
    """Sanitized transport failure with a runner owned retry class."""

    def __init__(self, error_code: str) -> None:
        if error_code not in {
            "network",
            "provider_429",
            "provider_5xx",
            "provider_timeout",
            "provider_4xx",
            "malformed_provider_response",
        }:
            raise ValueError("unknown sanitized transport error code")
        super().__init__(f"OpenRouter request failed ({error_code})")
        self.error_code = error_code


def _classify_provider_error(*, code: int | None, error_type: str | None) -> str:
    if code is not None:
        if isinstance(code, bool) or not isinstance(code, int):
            return "malformed_provider_response"
        if code in {408, 504}:
            return "provider_timeout"
        if code == 429:
            return "provider_429"
        if 500 <= code <= 599:
            return "provider_5xx"
        if 400 <= code <= 499:
            return "provider_4xx"
        return "malformed_provider_response"
    if error_type in _PROVIDER_429_TYPES:
        return "provider_429"
    if error_type in _PROVIDER_TIMEOUT_TYPES:
        return "provider_timeout"
    if error_type in _PROVIDER_5XX_TYPES:
        return "provider_5xx"
    if error_type in _PROVIDER_4XX_TYPES:
        return "provider_4xx"
    return "malformed_provider_response"


class _ClassifyingOpenRouterChannel:
    """Wrap an OpenAI compatible channel with sanitized error classes."""

    def __init__(self, inner: OpenAIChannel) -> None:
        self._inner = inner

    @property
    def supports_decision_evidence_context(self) -> bool:
        return bool(getattr(self._inner, "supports_decision_evidence_context", False))

    @property
    def supports_business_decisions(self) -> bool:
        return bool(getattr(self._inner, "supports_business_decisions", False))

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> Any:
        try:
            return self._inner.complete_business_decision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                decision_id=decision_id,
            )
        except (TimeoutError, socket.timeout) as exc:
            raise ClassifiedTransportError("provider_timeout") from exc
        except (ConnectionError, ssl.SSLError, http.client.HTTPException) as exc:
            raise ClassifiedTransportError("network") from exc
        except ProviderResponseError as exc:
            raise ClassifiedTransportError(
                _classify_provider_error(code=exc.code, error_type=exc.error_type)
            ) from exc
        except ChannelTransportError as exc:
            cause = exc.__cause__
            if isinstance(cause, urllib.error.HTTPError):
                if cause.code in {408, 504}:
                    code = "provider_timeout"
                elif cause.code == 429:
                    code = "provider_429"
                elif 500 <= cause.code <= 599:
                    code = "provider_5xx"
                else:
                    code = "provider_4xx"
            elif isinstance(cause, (TimeoutError, socket.timeout)):
                code = "provider_timeout"
            elif isinstance(cause, urllib.error.URLError) and isinstance(
                cause.reason, (TimeoutError, socket.timeout)
            ):
                code = "provider_timeout"
            elif isinstance(cause, (OSError, http.client.HTTPException)):
                code = "network"
            else:
                code = "malformed_provider_response"
            raise ClassifiedTransportError(code) from exc


def openrouter_channel_factory(
    *,
    api_key: str,
    allowed_model_ids: tuple[str, ...] = MAIN_MODELS_V2,
) -> Any:
    """Create exact model channels using the public benchmark settings."""

    if not isinstance(api_key, str) or not api_key.strip():
        raise BenchmarkRunError("an OpenRouter API key is required")
    if (
        not allowed_model_ids
        or len(allowed_model_ids) != len(set(allowed_model_ids))
        or any(
            not isinstance(model_id, str) or "/" not in model_id
            for model_id in allowed_model_ids
        )
    ):
        raise BenchmarkRunError("OpenRouter model allowlist is invalid")

    def make(model_id: str, _actor_id: str, _role: str) -> InferenceChannel:
        if model_id not in allowed_model_ids:
            raise BenchmarkRunError("model is outside the active OpenRouter panel")
        return _ClassifyingOpenRouterChannel(
            OpenAIChannel(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                model=model_id,
                timeout=OPENROUTER_TIMEOUT_SECONDS,
                provider_options={"sort": "latency", "allow_fallbacks": True},
            )
        )

    return make


def require_business_decision_executor(
    executor: BenchmarkExecutor,
) -> BenchmarkExecutor:
    """Require the typed business decision interface before provider access."""

    base_factory = executor.model_channels
    if base_factory is None:
        raise BenchmarkChannelContractError(
            "benchmark execution has no evaluated model channel factory"
        )

    def checked_factory(
        model_id: str,
        actor_id: str,
        role: str,
    ) -> InferenceChannel:
        channel = base_factory(model_id, actor_id, role)
        missing: list[str] = []
        if getattr(channel, "supports_business_decisions", None) is not True:
            missing.append("supports_business_decisions=True")
        if getattr(channel, "supports_decision_evidence_context", None) is not True:
            missing.append("supports_decision_evidence_context=True")
        if not callable(getattr(channel, "complete_business_decision", None)):
            missing.append("callable complete_business_decision")
        if missing:
            raise BenchmarkChannelContractError(
                "evaluated model channel lacks required Agent business capability: "
                f"{', '.join(missing)}"
            )
        return channel

    return executor.with_model_channels(checked_factory)


__all__ = [
    "BenchmarkChannelContractError",
    "BenchmarkRunError",
    "ClassifiedTransportError",
    "FATAL_RUN_ERRORS",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_TIMEOUT_SECONDS",
    "openrouter_channel_factory",
    "require_business_decision_executor",
]
