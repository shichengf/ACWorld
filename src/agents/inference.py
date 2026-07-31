"""Typed provider transport used by CommerceWorld Agents.

An LLM returns one provider-neutral business decision.  The local Agent owns
all CommerceWorld routing, identifiers, authority, World reads, envelopes,
retries, correlation, and protocol continuation.
"""

from __future__ import annotations

import json
import hashlib
import http.client
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Protocol

from agents.errors import AgentError


# --- Errors ----------------------------------------------------------


class ModelDecisionParseError(AgentError):
    """A provider response did not contain a valid business decision.

    This is a model-interface failure, not a transport failure. In particular,
    malformed or schema-invalid model text is never interpreted as a commerce
    action by the Agent.
    """

    def __init__(
        self,
        message: str,
        *,
        response_chars: int | None = None,
        response_sha256: str | None = None,
    ) -> None:
        """Build a safe model-interface error.

        ``response_chars`` and ``response_sha256`` are present only when an
        HTTP provider returned a response before business-decision parsing
        failed. They let formal accounting count the paid call without
        retaining model-authored text.
        """

        if (response_chars is None) is not (response_sha256 is None):
            raise ValueError("response evidence must provide both length and digest")
        if response_chars is not None and response_chars < 0:
            raise ValueError("response_chars must be non-negative")
        if response_sha256 is not None and (
            len(response_sha256) != 64
            or any(char not in "0123456789abcdef" for char in response_sha256)
        ):
            raise ValueError("response_sha256 must be a lowercase SHA-256 digest")
        super().__init__(message)
        self.response_chars = response_chars
        self.response_sha256 = response_sha256

    @property
    def provider_response_received(self) -> bool:
        """Whether this parse failure followed a valid provider response."""

        return self.response_chars is not None


class InvalidAgentEnvelope(AgentError, ValueError):
    """Agent compilation produced an invalid outbound-envelope shape."""


class ChannelTransportError(AgentError):
    """A networked channel failed to reach its endpoint or got a non-2xx reply."""


class ProviderResponseError(ChannelTransportError):
    """An HTTP-success response carried an in-band provider failure.

    OpenAI-compatible routers can return HTTP 200 while placing an upstream
    failure in the Chat Completions JSON envelope.  Retain only the numeric
    status and the canonical ``metadata.error_type`` identifier.  Provider
    messages, provider names, and raw metadata are deliberately excluded from
    both the exception object and its string representation.
    """

    def __init__(
        self,
        *,
        code: int | None,
        error_type: str | None,
    ) -> None:
        if code is not None and type(code) is not int:
            raise ValueError("provider response error code must be an integer or null")
        if error_type is not None and not _is_canonical_provider_error_type(error_type):
            raise ValueError("provider response error type must be canonical or null")

        details: list[str] = []
        if code is not None:
            details.append(f"code={code}")
        if error_type is not None:
            details.append(f"error_type={error_type}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"OpenAIChannel provider response error{suffix}")
        self.code = code
        self.error_type = error_type


# --- Channel Protocol ------------------------------------------------


class InferenceChannel(Protocol):
    """The sole typed LLM transport seam consumed by ``Agent.receive``."""

    supports_business_decisions: bool
    supports_decision_evidence_context: bool

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> "BusinessDecisionResponseV1": ...

# --- HTTP channel (OpenAI-compatible) ---------------------------------

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_HTTP_RESPONSE_READ_CHUNK_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class BusinessDecisionResponseV1:
    """Raw provider text plus hashes-only evidence for a business decision."""

    content: str = field(repr=False)
    response_chars: int
    response_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ValueError("business decision response content must be text")
        if self.response_chars < 0 or self.response_chars != len(self.content):
            raise ValueError("business decision response length is inconsistent")
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.response_sha256 != expected:
            raise ValueError("business decision response digest is inconsistent")


def _http_response_socket(response: Any) -> Any:
    """Return the socket whose read timeout bounds an urllib response.

    ``urllib`` exposes no public per-response deadline API.  CPython's HTTP
    response keeps the connected socket under ``fp.raw._sock`` for both HTTP
    and HTTPS.  Failing closed here is preferable to silently claiming a
    wall-clock timeout while an unbounded ``read()`` remains possible.
    """

    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if not callable(getattr(sock, "settimeout", None)):
        raise OSError("response socket does not support deadline reads")
    return sock


def _read_http_response_before_deadline(
    response: Any,
    *,
    deadline: float,
) -> bytes:
    """Read one HTTP body without allowing keepalives to extend its deadline."""

    reader = getattr(response, "read1", None)
    if not callable(reader):
        raise OSError("response does not support bounded reads")
    sock = _http_response_socket(response)
    chunks: list[bytes] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP response deadline elapsed")
        sock.settimeout(remaining)
        chunk = reader(_HTTP_RESPONSE_READ_CHUNK_BYTES)
        if time.monotonic() >= deadline:
            raise TimeoutError("HTTP response deadline elapsed")
        if not chunk:
            content_bytes_remaining = getattr(response, "length", None)
            if type(content_bytes_remaining) is int and content_bytes_remaining > 0:
                # http.client.HTTPResponse decrements ``length`` after each
                # successful read.  EOF with a positive remainder is a
                # truncated transport, not a complete malformed JSON body.
                # Surface it as a connection failure so the formal wrapper
                # can use the one exact-decision recovery request.
                raise ConnectionError("HTTP response ended before Content-Length was satisfied")
            return b"".join(chunks)
        if not isinstance(chunk, bytes):
            raise OSError("HTTP response body chunk must be bytes")
        chunks.append(chunk)


def _is_canonical_provider_error_type(value: str) -> bool:
    """Return whether ``value`` is a bounded lowercase snake-case identifier."""

    return (
        1 <= len(value) <= 64
        and value[0] in "abcdefghijklmnopqrstuvwxyz"
        and all(char in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in value)
    )


def _provider_error_fields(raw: Any) -> tuple[int | None, str | None]:
    """Extract the only two persistence-safe fields from a provider error."""

    if not isinstance(raw, Mapping):
        return None, None

    raw_code = raw.get("code")
    code = raw_code if type(raw_code) is int else None

    metadata = raw.get("metadata")
    raw_error_type = metadata.get("error_type") if isinstance(metadata, Mapping) else None
    error_type = (
        raw_error_type
        if isinstance(raw_error_type, str) and _is_canonical_provider_error_type(raw_error_type)
        else None
    )
    return code, error_type


def _raise_for_in_band_provider_error(data: Mapping[str, Any]) -> None:
    """Fail safely when a 2xx Chat Completions envelope represents an error."""

    if "error" in data and data.get("error") is not None:
        code, error_type = _provider_error_fields(data.get("error"))
        raise ProviderResponseError(code=code, error_type=error_type)

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return

    has_choice_error = "error" in choice and choice.get("error") is not None
    if has_choice_error or choice.get("finish_reason") == "error":
        code, error_type = _provider_error_fields(choice.get("error"))
        raise ProviderResponseError(code=code, error_type=error_type)


class OpenAIChannel:
    """Business-decision transport for a Chat Completions–compatible endpoint.

    Covers OpenAI itself, OpenRouter, Anthropic via the OpenAI-compat shim,
    local OpenAI-compatible servers, and the planned
    cc-router endpoint. Configuration precedence (later wins):

        1. Class defaults (``base_url=https://api.openai.com/v1``).
        2. Environment variables: ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``,
           ``OPENAI_MODEL``.
        3. Constructor kwargs.

    Uses only stdlib ``urllib`` so the package stays dependency-free; if
    you'd rather lean on the official ``openai`` SDK, write a thin
    adapter that satisfies the same Protocol.

    The model sees and returns only provider-neutral business JSON.  This
    channel accepts and returns plain business-decision JSON only; the Agent
    owns every CommerceWorld interface operation.
    """

    base_url: str
    api_key: str
    model: str
    timeout: float
    extra_headers: "dict[str, str]"
    provider_options: "dict[str, Any]"
    supports_business_decisions = True
    # ``decision_id`` is local evidence context. It is validated by this
    # transport but never serialized into the provider request.
    supports_decision_evidence_context = True

    def __init__(
        self,
        *,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        model: "str | None" = None,
        timeout: "float | None" = None,
        extra_headers: "dict[str, str] | None" = None,
        provider_options: "Mapping[str, Any] | None" = None,
        business_response_format: bool = False,
    ) -> None:
        """Resolve config from kwargs > env > defaults.

        Raises:
            ValueError: ``api_key`` or ``model`` cannot be resolved.
        """
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("OPENAI_MODEL") or ""
        resolved_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT_SECONDS
        if (
            isinstance(resolved_timeout, bool)
            or not isinstance(resolved_timeout, (int, float))
            or not math.isfinite(float(resolved_timeout))
            or float(resolved_timeout) <= 0
        ):
            raise ValueError("OpenAIChannel: timeout must be a positive finite number")
        self.timeout = float(resolved_timeout)
        self.extra_headers = dict(extra_headers or {})
        self.provider_options = dict(provider_options or {})
        if not isinstance(business_response_format, bool):
            raise ValueError("OpenAIChannel: business_response_format must be boolean")
        self.business_response_format = business_response_format
        if not self.api_key:
            raise ValueError(
                "OpenAIChannel: api_key is required (pass api_key=... or set OPENAI_API_KEY)"
            )
        if not self.model:
            raise ValueError(
                "OpenAIChannel: model is required (pass model=... or set OPENAI_MODEL)"
            )

    def complete_business_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        decision_id: str | None = None,
    ) -> BusinessDecisionResponseV1:
        """Return plain business-decision JSON text."""

        if decision_id is not None and (
            not isinstance(decision_id, str)
            or not decision_id
            or len(decision_id) > 512
            or any(ord(char) < 32 for char in decision_id)
        ):
            raise ValueError("business decision evidence context is invalid")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.business_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.provider_options:
            payload["provider"] = dict(self.provider_options)
        data = self._post_chat_completions(payload)
        _raise_for_in_band_provider_error(data)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChannelTransportError(
                "OpenAIChannel business response lacks assistant content"
            ) from exc
        if not isinstance(content, str):
            raise ChannelTransportError("OpenAIChannel business response content is not text")
        return BusinessDecisionResponseV1(
            content=content,
            response_chars=len(content),
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def _post_chat_completions(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST one Chat Completions payload and decode its JSON object."""

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        deadline = time.monotonic() + self.timeout
        try:
            connection_timeout = deadline - time.monotonic()
            if connection_timeout <= 0:
                raise TimeoutError("HTTP connection deadline elapsed")
            with urllib.request.urlopen(req, timeout=connection_timeout) as resp:
                raw = _read_http_response_before_deadline(
                    resp,
                    deadline=deadline,
                )
        except urllib.error.HTTPError as exc:
            raise ChannelTransportError(f"OpenAIChannel HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ChannelTransportError(f"OpenAIChannel transport error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ChannelTransportError(
                "OpenAIChannel request exceeded its wall-clock timeout"
            ) from exc
        except http.client.HTTPException as exc:
            # IncompleteRead, BadStatusLine, and related framing failures are
            # transient connection failures.  Preserve only the exception
            # type as the causal class; never serialize a partial provider
            # body or status-line detail.
            raise ChannelTransportError(
                f"OpenAIChannel HTTP framing error: {type(exc).__name__}"
            ) from exc
        except OSError as exc:
            raise ChannelTransportError(
                f"OpenAIChannel transport error: {type(exc).__name__}"
            ) from exc

        try:
            decoded = raw.decode("utf-8")
            data = json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ChannelTransportError(f"OpenAIChannel response was not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ChannelTransportError("OpenAIChannel response root must be an object")
        return data
