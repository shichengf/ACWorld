"""Strict actor-authored result evidence for principal-facing reports.

This module defines the payload shared by the
``delegate.report_result`` and ``commerce.submit_decision_record`` actions.
Both actions address ``runtime:evidence``.  Runtime validates and journals the
report before it can become benchmark evidence, so HTTP and in-process runs use
the same acceptance boundary.

An :class:`ActorResultReport` is evidence written by an actor.  It is **not** a
Platform decision, a World commit, a payment receipt, or proof that the
reported commerce state exists.  Authoritative state must still be joined to
the Platform decision journal and World commit journal.

The wire contract has three defenses against re-labelling an old report:

* actor and principal identities are duplicated in the payload and checked
  against the enclosing envelope;
* task, mandate, context, ``in_reply_to``, and the envelope idempotency key are
  checked against a runtime-supplied :class:`ActorResultBinding`;
* ``content_sha256`` commits to the content *and* every binding field, including
  ``schema_id`` and action kind.

The JSON form is canonical (UTF-8, sorted keys, compact separators, no NaN), so
the same bytes can be hashed, transported over HTTP, and replayed later.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias, TypedDict, cast

from protocol.errors import SchemaError


ACTOR_RESULT_SCHEMA_ID = "cwe.actor-result.v1"
ACTOR_EVIDENCE_AUTHORITY = "actor-authored-evidence-only"

DELEGATE_REPORT_RESULT = "delegate.report_result"
COMMERCE_SUBMIT_DECISION_RECORD = "commerce.submit_decision_record"

ActorResultAction: TypeAlias = Literal[
    "delegate.report_result",
    "commerce.submit_decision_record",
]
ActorRecordType: TypeAlias = Literal["result", "decision_record"]
ActorOutcome: TypeAlias = Literal["completed", "partial", "declined", "failed"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]


_ACTION_TO_RECORD_TYPE: dict[str, ActorRecordType] = {
    DELEGATE_REPORT_RESULT: "result",
    COMMERCE_SUBMIT_DECISION_RECORD: "decision_record",
}
_OUTCOMES = frozenset({"completed", "partial", "declined", "failed"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ActorResultSchemaError(SchemaError):
    """The actor-result payload is not an exact v1 wire object."""


class ActorResultDigestMismatch(ActorResultSchemaError):
    """The report's digest does not match its bound canonical content."""


class ActorResultBindingError(ActorResultSchemaError):
    """The report does not belong to the enclosing envelope/runtime context."""


class ActorReportContentWire(TypedDict):
    """Wire shape for the schema-stable content header.

    Family-specific result fields belong under ``details``.  This keeps the
    outer evidence contract strict without making the core protocol know about
    benchmark task families.
    """

    outcome: ActorOutcome
    summary: str
    details: dict[str, Any]


# The model-facing submission deliberately has the same compact shape as the
# content block.  It contains no actor, principal, task, causal, idempotency, or
# digest fields that an untrusted policy could self-assert.  Runtime supplies
# those bindings and seals the accepted ActorResultReport.
ActorReportSubmissionWire: TypeAlias = ActorReportContentWire


class ActorResultReportWire(TypedDict):
    """Exact v1 wire shape for an actor-authored report payload."""

    schema_id: str
    authority: str
    action_kind: ActorResultAction
    record_type: ActorRecordType
    actor_id: str
    principal_id: str
    task_id: str
    mandate_id: str
    context_id: str
    in_reply_to: str
    idempotency_key: str
    content: ActorReportContentWire
    content_sha256: str


@dataclass(frozen=True)
class ActorReportContent:
    """Typed, immutable actor-authored content.

    ``details`` is recursively frozen during coercion.  It can carry task or
    domain-specific evidence, but it cannot smuggle extra top-level protocol
    fields into the report.
    """

    outcome: ActorOutcome
    summary: str
    details: Mapping[str, JsonValue]


@dataclass(frozen=True)
class ActorResultReport:
    """One non-authoritative actor result/decision record."""

    schema_id: str
    authority: str
    action_kind: ActorResultAction
    record_type: ActorRecordType
    actor_id: str
    principal_id: str
    task_id: str
    mandate_id: str
    context_id: str
    in_reply_to: str
    idempotency_key: str
    content: ActorReportContent
    content_sha256: str


@dataclass(frozen=True)
class ActorResultBinding:
    """Authoritative envelope and episode facts used to accept a report.

    Runtime constructs this object from the actual envelope plus the active
    task/mandate context.  Report-supplied values must never be used to build
    the expected binding.
    """

    action_kind: ActorResultAction
    actor_id: str
    principal_id: str
    task_id: str
    mandate_id: str
    context_id: str
    in_reply_to: str
    idempotency_key: str

    def __post_init__(self) -> None:
        _validate_action_kind(self.action_kind)
        for name in (
            "actor_id",
            "principal_id",
            "task_id",
            "mandate_id",
            "context_id",
            "in_reply_to",
            "idempotency_key",
        ):
            _require_nonempty_string(name, getattr(self, name))


def build_actor_result_report(
    *,
    binding: ActorResultBinding,
    outcome: ActorOutcome,
    summary: str,
    details: Mapping[str, Any],
) -> ActorResultReport:
    """Build and seal one report from trusted binding facts and actor content."""

    content = _coerce_content(
        {"outcome": outcome, "summary": summary, "details": details}
    )
    report = ActorResultReport(
        schema_id=ACTOR_RESULT_SCHEMA_ID,
        authority=ACTOR_EVIDENCE_AUTHORITY,
        action_kind=binding.action_kind,
        record_type=_ACTION_TO_RECORD_TYPE[binding.action_kind],
        actor_id=binding.actor_id,
        principal_id=binding.principal_id,
        task_id=binding.task_id,
        mandate_id=binding.mandate_id,
        context_id=binding.context_id,
        in_reply_to=binding.in_reply_to,
        idempotency_key=binding.idempotency_key,
        content=content,
        content_sha256="",
    )
    sealed = _replace_digest(report, actor_result_content_sha256(report))
    validate_actor_result_report(sealed)
    return sealed


def coerce_actor_report_submission(value: Any) -> ActorReportContent:
    """Strictly parse the compact model-facing actor result submission.

    Models are not asked to calculate SHA-256 or repeat trusted identities.
    Runtime combines this content with an independently resolved
    :class:`ActorResultBinding` and calls :func:`build_actor_result_report`.
    """

    return _coerce_content(value)


def actor_report_submission_to_dict(
    content: ActorReportContent,
) -> ActorReportSubmissionWire:
    """Return a fresh JSON-compatible compact submission mapping."""

    _validate_content(content)
    return cast(
        ActorReportSubmissionWire,
        {
            "outcome": content.outcome,
            "summary": content.summary,
            "details": _thaw_json(content.details),
        },
    )


def coerce_actor_result_report(value: Any) -> ActorResultReport:
    """Strictly coerce a decoded JSON object into a validated report.

    Unknown or missing top-level fields are rejected.  Values are not silently
    converted with ``str()`` or ``int()`` because such coercion can hide wire
    schema drift.
    """

    if isinstance(value, ActorResultReport):
        validate_actor_result_report(value)
        return value
    if not isinstance(value, Mapping):
        raise ActorResultSchemaError("actor result report must be an object")

    row = dict(value)
    expected = {
        "schema_id",
        "authority",
        "action_kind",
        "record_type",
        "actor_id",
        "principal_id",
        "task_id",
        "mandate_id",
        "context_id",
        "in_reply_to",
        "idempotency_key",
        "content",
        "content_sha256",
    }
    unknown = sorted(set(row) - expected)
    missing = sorted(expected - set(row))
    if unknown:
        raise ActorResultSchemaError(f"unknown actor result fields: {unknown}")
    if missing:
        raise ActorResultSchemaError(f"missing actor result fields: {missing}")

    report = ActorResultReport(
        schema_id=_strict_string("schema_id", row["schema_id"]),
        authority=_strict_string("authority", row["authority"]),
        action_kind=cast(
            ActorResultAction, _strict_string("action_kind", row["action_kind"])
        ),
        record_type=cast(
            ActorRecordType, _strict_string("record_type", row["record_type"])
        ),
        actor_id=_strict_string("actor_id", row["actor_id"]),
        principal_id=_strict_string("principal_id", row["principal_id"]),
        task_id=_strict_string("task_id", row["task_id"]),
        mandate_id=_strict_string("mandate_id", row["mandate_id"]),
        context_id=_strict_string("context_id", row["context_id"]),
        in_reply_to=_strict_string("in_reply_to", row["in_reply_to"]),
        idempotency_key=_strict_string(
            "idempotency_key", row["idempotency_key"]
        ),
        content=_coerce_content(row["content"]),
        content_sha256=_strict_string("content_sha256", row["content_sha256"]),
    )
    validate_actor_result_report(report)
    return report


def actor_result_to_dict(report: ActorResultReport) -> ActorResultReportWire:
    """Return a fresh JSON-compatible mapping after full self-validation."""

    validate_actor_result_report(report)
    return cast(
        ActorResultReportWire,
        {
            "schema_id": report.schema_id,
            "authority": report.authority,
            "action_kind": report.action_kind,
            "record_type": report.record_type,
            "actor_id": report.actor_id,
            "principal_id": report.principal_id,
            "task_id": report.task_id,
            "mandate_id": report.mandate_id,
            "context_id": report.context_id,
            "in_reply_to": report.in_reply_to,
            "idempotency_key": report.idempotency_key,
            "content": {
                "outcome": report.content.outcome,
                "summary": report.content.summary,
                "details": _thaw_json(report.content.details),
            },
            "content_sha256": report.content_sha256,
        },
    )


def actor_result_to_json(report: ActorResultReport) -> str:
    """Serialize to canonical JSON suitable for HTTP and replay artifacts."""

    return _canonical_json(actor_result_to_dict(report))


def actor_result_from_json(payload: str) -> ActorResultReport:
    """Parse strict JSON, rejecting duplicate keys at any nesting level."""

    if not isinstance(payload, str):
        raise ActorResultSchemaError("actor result JSON must be a string")
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise ActorResultSchemaError(f"invalid actor result JSON: {exc}") from exc
    return coerce_actor_result_report(value)


def actor_result_content_sha256(report: ActorResultReport) -> str:
    """Hash canonical actor content together with every semantic binding field.

    Despite the concise wire name ``content_sha256``, this is intentionally not
    a hash of the free-form ``details`` alone.  It commits the content to the
    schema, action, actor, principal, task, mandate, context, reply target, and
    idempotency key. Editing any binding invalidates the digest.
    """

    contract = {
        "schema_id": report.schema_id,
        "authority": report.authority,
        "action_kind": report.action_kind,
        "record_type": report.record_type,
        "actor_id": report.actor_id,
        "principal_id": report.principal_id,
        "task_id": report.task_id,
        "mandate_id": report.mandate_id,
        "context_id": report.context_id,
        "in_reply_to": report.in_reply_to,
        "idempotency_key": report.idempotency_key,
        "content": {
            "outcome": report.content.outcome,
            "summary": report.content.summary,
            "details": _thaw_json(report.content.details),
        },
    }
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def validate_actor_result_report(report: ActorResultReport) -> None:
    """Validate exact schema, authority marker, content, and digest."""

    if not isinstance(report, ActorResultReport):
        raise ActorResultSchemaError("actor result must be ActorResultReport")
    if report.schema_id != ACTOR_RESULT_SCHEMA_ID:
        raise ActorResultSchemaError(
            f"unsupported actor result schema_id: {report.schema_id!r}"
        )
    if report.authority != ACTOR_EVIDENCE_AUTHORITY:
        raise ActorResultSchemaError(
            "actor result authority must be actor-authored-evidence-only"
        )
    _validate_action_kind(report.action_kind)
    expected_type = _ACTION_TO_RECORD_TYPE[report.action_kind]
    if report.record_type != expected_type:
        raise ActorResultSchemaError(
            f"record_type {report.record_type!r} does not match action "
            f"{report.action_kind!r}"
        )
    for name in (
        "actor_id",
        "principal_id",
        "task_id",
        "mandate_id",
        "context_id",
        "in_reply_to",
        "idempotency_key",
    ):
        _require_nonempty_string(name, getattr(report, name))
    _validate_content(report.content)
    if not isinstance(report.content_sha256, str) or not _SHA256_RE.fullmatch(
        report.content_sha256
    ):
        raise ActorResultSchemaError("content_sha256 must be 64 lowercase hex characters")
    expected_digest = actor_result_content_sha256(report)
    if report.content_sha256 != expected_digest:
        raise ActorResultDigestMismatch("actor result content_sha256 mismatch")


def verify_actor_result_binding(
    report: ActorResultReport, expected: ActorResultBinding
) -> None:
    """Verify a valid report against actual envelope and runtime context facts.

    Callers must build ``expected`` from trusted Runtime state.  A structurally
    valid report from another task or episode is rejected rather than accepted
    as replay evidence.
    """

    validate_actor_result_report(report)
    checks = {
        "action_kind": (report.action_kind, expected.action_kind),
        "actor_id": (report.actor_id, expected.actor_id),
        "principal_id": (report.principal_id, expected.principal_id),
        "task_id": (report.task_id, expected.task_id),
        "mandate_id": (report.mandate_id, expected.mandate_id),
        "context_id": (report.context_id, expected.context_id),
        "in_reply_to": (report.in_reply_to, expected.in_reply_to),
        "idempotency_key": (
            report.idempotency_key,
            expected.idempotency_key,
        ),
    }
    mismatches = [name for name, (actual, wanted) in checks.items() if actual != wanted]
    if mismatches:
        raise ActorResultBindingError(
            "actor result binding mismatch: " + ", ".join(mismatches)
        )


def _coerce_content(value: Any) -> ActorReportContent:
    if isinstance(value, ActorReportContent):
        _validate_content(value)
        return value
    if not isinstance(value, Mapping):
        raise ActorResultSchemaError("actor result content must be an object")
    row = dict(value)
    expected = {"outcome", "summary", "details"}
    unknown = sorted(set(row) - expected)
    missing = sorted(expected - set(row))
    if unknown:
        raise ActorResultSchemaError(f"unknown actor result content fields: {unknown}")
    if missing:
        raise ActorResultSchemaError(f"missing actor result content fields: {missing}")
    outcome = _strict_string("content.outcome", row["outcome"])
    summary = _strict_string("content.summary", row["summary"])
    if not isinstance(row["details"], Mapping):
        raise ActorResultSchemaError("content.details must be an object")
    frozen_details = _freeze_json(row["details"], path="content.details")
    if not isinstance(frozen_details, Mapping):  # pragma: no cover - guarded above
        raise ActorResultSchemaError("content.details must be an object")
    content = ActorReportContent(
        outcome=cast(ActorOutcome, outcome),
        summary=summary,
        details=frozen_details,
    )
    _validate_content(content)
    return content


def _validate_content(content: ActorReportContent) -> None:
    if not isinstance(content, ActorReportContent):
        raise ActorResultSchemaError("content must be ActorReportContent")
    if content.outcome not in _OUTCOMES:
        raise ActorResultSchemaError(f"unsupported content.outcome: {content.outcome!r}")
    _require_nonempty_string("content.summary", content.summary)
    if not isinstance(content.details, Mapping):
        raise ActorResultSchemaError("content.details must be an object")
    _freeze_json(content.details, path="content.details")


def _validate_action_kind(value: Any) -> None:
    if not isinstance(value, str) or value not in _ACTION_TO_RECORD_TYPE:
        raise ActorResultSchemaError(f"unsupported actor result action_kind: {value!r}")


def _strict_string(name: str, value: Any) -> str:
    _require_nonempty_string(name, value)
    return cast(str, value)


def _require_nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ActorResultSchemaError(f"{name} must be a non-empty string")


def _freeze_json(value: Any, *, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ActorResultSchemaError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ActorResultSchemaError(f"{path} contains a non-string object key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ActorResultSchemaError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ActorResultSchemaError(f"actor result is not canonical JSON: {exc}") from exc


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActorResultSchemaError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _replace_digest(report: ActorResultReport, digest: str) -> ActorResultReport:
    return ActorResultReport(
        schema_id=report.schema_id,
        authority=report.authority,
        action_kind=report.action_kind,
        record_type=report.record_type,
        actor_id=report.actor_id,
        principal_id=report.principal_id,
        task_id=report.task_id,
        mandate_id=report.mandate_id,
        context_id=report.context_id,
        in_reply_to=report.in_reply_to,
        idempotency_key=report.idempotency_key,
        content=report.content,
        content_sha256=digest,
    )


__all__ = [
    "ACTOR_EVIDENCE_AUTHORITY",
    "ACTOR_RESULT_SCHEMA_ID",
    "COMMERCE_SUBMIT_DECISION_RECORD",
    "DELEGATE_REPORT_RESULT",
    "ActorOutcome",
    "ActorRecordType",
    "ActorReportContent",
    "ActorReportContentWire",
    "ActorReportSubmissionWire",
    "ActorResultAction",
    "ActorResultBinding",
    "ActorResultBindingError",
    "ActorResultDigestMismatch",
    "ActorResultReport",
    "ActorResultReportWire",
    "ActorResultSchemaError",
    "actor_result_content_sha256",
    "actor_result_from_json",
    "actor_result_to_dict",
    "actor_result_to_json",
    "actor_report_submission_to_dict",
    "build_actor_result_report",
    "coerce_actor_report_submission",
    "coerce_actor_result_report",
    "validate_actor_result_report",
    "verify_actor_result_binding",
]
