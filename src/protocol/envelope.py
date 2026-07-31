"""The wire envelope: the shape every agent uses to talk over the bus.

This shape is wire-stable; fields can be added, never removed or renamed.
The serialization helpers here are the one place the wire form is materialized.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal
from typing import Any

from protocol.actions import ActionKind
from protocol.errors import SchemaError, UnknownActionKind


@dataclass(frozen=True)
class Envelope:
    """One bus envelope.

    The Python attribute ``from_`` corresponds to the wire field ``from``
    (``from`` is a Python keyword). ``to_json`` / ``from_json`` handle the
    rename in both directions.
    """
    msg_id: str
    ts: str
    from_: str
    to: str
    in_reply_to: str | None
    idempotency_key: str
    action: dict[str, Any]
    signature: str | None = None  # populated only in signed-envelope scenarios


def to_json(env: Envelope) -> str:
    """Serialize ``env`` to canonical JSON (sorted keys; ``from_`` -> ``from``).

    Output must be byte-stable for replay; the implementer must use
    ``sort_keys=True`` and a fixed float repr.
    """
    data = asdict(env)
    data["from"] = data.pop("from_")
    return json.dumps(
        data,
        allow_nan=False,
        default=_json_default,
        separators=(",", ":"),
        sort_keys=True,
    )


def from_json(payload: str) -> Envelope:
    """Parse JSON into an ``Envelope``; raise ``SchemaError`` on malformed input."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid envelope JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError("envelope payload must be a JSON object")
    if "from" in data:
        data["from_"] = data.pop("from")
    try:
        env = Envelope(**data)
    except TypeError as exc:
        raise SchemaError(f"invalid envelope fields: {exc}") from exc
    validate(env)
    return env


def validate(env: Envelope) -> None:
    """Pure schema validation. Does not check role partition (Router does that).

    Raises:
        SchemaError: any required field missing/empty/wrong-typed.
        UnknownActionKind: ``env.action['kind']`` is not in the ActionKind enum.
    """
    for name in ("msg_id", "ts", "from_", "to", "idempotency_key"):
        value = getattr(env, name)
        if not isinstance(value, str) or not value:
            raise SchemaError(f"{name} must be a non-empty string")
    if not _address_like(env.from_) or not _address_like(env.to):
        raise SchemaError("from/to must be '<side>[:<role>]' addresses")
    if env.in_reply_to is not None and not isinstance(env.in_reply_to, str):
        raise SchemaError("in_reply_to must be a string or null")
    if not isinstance(env.action, dict):
        raise SchemaError("action must be an object")
    if not isinstance(env.action.get("kind"), str) or not env.action["kind"]:
        raise SchemaError("action.kind must be a non-empty string")
    if "payload" not in env.action:
        raise SchemaError("action.payload is required")
    try:
        ActionKind(env.action["kind"])
    except ValueError as exc:
        raise UnknownActionKind(env.action["kind"]) from exc


def _address_like(value: str) -> bool:
    parts = value.split(":")
    return bool(parts[0]) and all(part for part in parts)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return asdict(value)  # type: ignore[arg-type]
    return str(value)
