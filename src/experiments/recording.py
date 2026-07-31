"""Non-invasive inference observations around an existing channel."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable

from agents.inference import (
    BusinessDecisionResponseV1,
    InferenceChannel,
    ModelDecisionParseError,
)


MODEL_PROTOCOL_RESPONSE_ERROR_TYPE = "ModelDecisionParseError"
PRE_PROVIDER_CALL_ERROR_CODE = "pre_provider_call_failure"
RETRYABLE_TRANSPORT_OBSERVATION_CODES = frozenset(
    {
        "network",
        "provider_429",
        "provider_5xx",
        "provider_timeout",
        "malformed_provider_response",
    }
)
MAX_SAME_DECISION_TRANSPORT_RETRIES = 2


@dataclass(frozen=True)
class CallObservation:
    """Provider-independent data available without changing ``OpenAIChannel``."""

    call_index: int
    latency_seconds: float
    system_chars: int
    user_chars: int
    output_chars: int | None
    system_prompt_sha256: str
    user_prompt_sha256: str
    output_sha256: str | None
    decision_id: str | None = None
    error_type: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


class RecordingChannel:
    """Record timing, lengths, and digests without retaining request content."""

    def __init__(
        self,
        channel: InferenceChannel,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._channel = channel
        self._clock = clock
        self._observations: list[CallObservation] = []

    @property
    def observations(self) -> tuple[CallObservation, ...]:
        return tuple(self._observations)

    @property
    def supports_decision_evidence_context(self) -> bool:
        """Whether calls can be joined to the Agent Tracker decision."""

        return bool(
            getattr(
                self._channel,
                "supports_decision_evidence_context",
                False,
            )
        )

    @property
    def supports_business_decisions(self) -> bool:
        return bool(getattr(self._channel, "supports_business_decisions", False))

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        """Record one provider-neutral business decision without raw retention."""

        if not self.supports_business_decisions:
            raise TypeError("wrapped channel does not support business decisions")
        complete = getattr(self._channel, "complete_business_decision", None)
        if not callable(complete):
            raise TypeError("wrapped channel has no business completion method")
        if not self.supports_decision_evidence_context:
            decision_id = None
        elif not isinstance(decision_id, str) or not decision_id:
            raise ValueError("business decision evidence requires a decision_id")
        started = self._clock()
        try:
            response = complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                decision_id=decision_id,
            )
        except Exception as exc:
            self._record_error(
                started=started,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                exc=exc,
                decision_id=decision_id,
            )
            raise
        if not isinstance(response, BusinessDecisionResponseV1):
            raise TypeError("business completion returned an unsupported response")
        self._record_success_evidence(
            started=started,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_chars=response.response_chars,
            output_sha256=response.response_sha256,
            decision_id=decision_id,
        )
        return response

    def _record_error(
        self,
        *,
        started: float,
        system_prompt: str,
        user_prompt: str,
        exc: Exception,
        decision_id: str | None,
    ) -> None:
        """Append one failure observation without retaining exception text."""

        output_chars, output_sha256 = _safe_response_evidence(exc)
        self._observations.append(
            CallObservation(
                call_index=len(self._observations),
                latency_seconds=max(0.0, self._clock() - started),
                system_chars=len(system_prompt),
                user_chars=len(user_prompt),
                output_chars=output_chars,
                system_prompt_sha256=_sha256(system_prompt),
                user_prompt_sha256=_sha256(user_prompt),
                output_sha256=output_sha256,
                decision_id=decision_id,
                error_type=type(exc).__name__,
                error_code=(
                    str(code) if (code := getattr(exc, "error_code", None)) is not None else None
                ),
            )
        )

    def _record_success_evidence(
        self,
        *,
        started: float,
        system_prompt: str,
        user_prompt: str,
        output_chars: int,
        output_sha256: str,
        decision_id: str | None,
    ) -> None:
        """Append exact response evidence without materializing response text."""

        self._observations.append(
            CallObservation(
                call_index=len(self._observations),
                latency_seconds=max(0.0, self._clock() - started),
                system_chars=len(system_prompt),
                user_chars=len(user_prompt),
                output_chars=output_chars,
                system_prompt_sha256=_sha256(system_prompt),
                user_prompt_sha256=_sha256(user_prompt),
                output_sha256=output_sha256,
                decision_id=decision_id,
            )
        )

    def summary(self) -> dict[str, int | float]:
        return {
            "call_count": len(self._observations),
            "latency_seconds": sum(item.latency_seconds for item in self._observations),
            "input_chars": sum(item.system_chars + item.user_chars for item in self._observations),
            "output_chars": sum(item.output_chars or 0 for item in self._observations),
            "error_count": sum(item.error_type is not None for item in self._observations),
        }


class RecordedInferenceFailure(RuntimeError):
    """Safe infrastructure failure carrying completed call observations.

    Provider exceptions can contain response bodies, request headers, or other
    sensitive text.  Direct executors therefore replace them at their boundary
    with this fixed-message exception.  The attached observations are built
    exclusively from :class:`CallObservation` plus executor-owned identity
    labels; raw prompts, outputs, provider messages, and credentials cannot be
    represented here.
    """

    def __init__(
        self,
        message: str,
        *,
        observations: tuple[dict[str, Any], ...],
    ) -> None:
        super().__init__(message)
        self.observations = observations


def recorded_inference_failure(
    message: str,
    channel: RecordingChannel,
    *,
    actor_id: str,
    role: str,
    model_id: str,
) -> RecordedInferenceFailure:
    """Project a failed recording into a persistence-safe exception.

    ``message`` must be an executor-owned constant.  It is deliberately not
    derived from the provider exception that caused the failed call.
    """

    return RecordedInferenceFailure(
        message,
        observations=tuple(
            {
                "actor_id": actor_id,
                "role": role,
                "source": "model",
                "model_id": model_id,
                **observation.to_dict(),
            }
            for observation in channel.observations
        ),
    )


def _sha256(content: str) -> str:
    """Return a stable content fingerprint without retaining the content."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _safe_response_evidence(exc: Exception) -> tuple[int | None, str | None]:
    """Read only validated response telemetry from a model protocol error."""

    if not isinstance(exc, ModelDecisionParseError):
        return None, None
    if not exc.provider_response_received:
        return None, None
    chars = exc.response_chars
    digest = exc.response_sha256
    if (
        not isinstance(chars, int)
        or isinstance(chars, bool)
        or chars < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        return None, None
    return chars, digest


def is_model_protocol_response_observation(row: Mapping[str, Any]) -> bool:
    """Return whether a safe row represents scoreable model protocol behavior.

    This is intentionally narrow. Only a model-controlled typed decision that
    received a provider response, failed as ``ModelDecisionParseError``, and
    retained exact response length and SHA-256 qualifies. Legacy parse errors,
    reference-policy failures, transport errors, and provider errors remain
    infrastructure failures.
    """

    output_chars = row.get("output_chars")
    output_sha256 = row.get("output_sha256")
    return (
        row.get("source") == "model"
        and row.get("error_type") == MODEL_PROTOCOL_RESPONSE_ERROR_TYPE
        and row.get("error_code") is None
        and isinstance(output_chars, int)
        and not isinstance(output_chars, bool)
        and output_chars >= 0
        and isinstance(output_sha256, str)
        and len(output_sha256) == 64
        and all(char in "0123456789abcdef" for char in output_sha256)
    )


def recovered_same_decision_transport_error_indices(
    rows: Sequence[Mapping[str, Any]],
) -> frozenset[int]:
    """Identify transport-error chains recovered by an exact later attempt.

    Recovery is intentionally evidence based. Every adjacent observation in a
    chain must belong to the same actor and decision, use the next call index,
    and repeat the exact system/user prompt and provider-facing tool hashes.
    The chain is recovered only when it terminates in a provider response; an
    unrelated later success can therefore never hide an infrastructure
    failure. Unmatched errors are simply omitted and remain fatal to callers.
    """

    def _is_retryable_error(row: Mapping[str, Any]) -> bool:
        return row.get("error_code") in RETRYABLE_TRANSPORT_OBSERVATION_CODES

    def _is_accepted_response(row: Mapping[str, Any]) -> bool:
        return row.get("error_type") is None or is_model_protocol_response_observation(row)

    def _is_exact_next_attempt(
        prior: Mapping[str, Any],
        following: Mapping[str, Any],
    ) -> bool:
        call_index = prior.get("call_index")
        actor_id = prior.get("actor_id")
        if (
            not isinstance(actor_id, str)
            or not actor_id
            or actor_id != following.get("actor_id")
            or prior.get("decision_id") != following.get("decision_id")
            or isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or following.get("call_index") != call_index + 1
        ):
            return False
        for field in ("system_prompt_sha256", "user_prompt_sha256"):
            value = prior.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                or value != following.get(field)
            ):
                return False
        return True

    recovered: set[int] = set()
    pending: list[int] = []
    for index, row in enumerate(rows):
        if not pending:
            if _is_retryable_error(row):
                pending.append(index)
            continue

        prior = rows[pending[-1]]
        if not _is_exact_next_attempt(prior, row):
            pending = [index] if _is_retryable_error(row) else []
            continue

        if _is_retryable_error(row):
            pending.append(index)
            if len(pending) > MAX_SAME_DECISION_TRANSPORT_RETRIES:
                # The entire over-limit chain stays unrecovered. A later
                # provider response cannot turn excess calls into valid
                # evidence merely because the 48-call ledger had capacity.
                pending = []
            continue
        if _is_accepted_response(row):
            recovered.update(pending)
        pending = []
    return frozenset(recovered)


__all__ = [
    "CallObservation",
    "PRE_PROVIDER_CALL_ERROR_CODE",
    "MAX_SAME_DECISION_TRANSPORT_RETRIES",
    "RETRYABLE_TRANSPORT_OBSERVATION_CODES",
    "RecordedInferenceFailure",
    "RecordingChannel",
    "is_model_protocol_response_observation",
    "recovered_same_decision_transport_error_indices",
    "recorded_inference_failure",
]
