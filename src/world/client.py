"""HTTP client for the World VCP service (WORLD_DESIGN.md §3 / §11 slice 3).

The :class:`VCPWorldClient` is the agent-facing complement to
:mod:`world.api`: it serializes an :class:`~protocol.envelope.Envelope`,
POSTs it to ``/vcp``, and parses the reply envelope. It is the only HTTP-
aware code in this package; everything else (services, tables, stores)
stays transport-agnostic.

Synchronous (``httpx.Client``) — the simulation loop is single-threaded
and the agent-side ergonomics of ``reply = client.send(env)`` are nicer
than an async equivalent for notebook use. A future async variant can
sit next to this one if needed.

The wire field rename ``from_`` (Python attribute) ↔ ``from`` (JSON
field) is centralized in :func:`protocol.envelope.to_json` /
:func:`from_json`; this client reuses both ends.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from protocol.cart_quote_state import CartQuoteStaleError
from protocol.envelope import Envelope, from_json, to_json
from protocol.event_receipts import (
    ProtocolEventAuthorityError,
    ProtocolEventStaleError,
)
from protocol.matching import MatchAcceptanceRejected
from protocol.negotiation_state import NegotiationStateError
from world.errors import (
    AfterSalesReferenceRejected,
    CatalogMutationRejected,
    DisputeNotActionable,
    ExchangeNotActionable,
    FulfillmentNotActionable,
    IdempotencyConflict,
    InsufficientFunds,
    InvalidOrderTransition,
    LifecycleAuthorizationError,
    OrderIdentityMismatch,
    OrderNotRefundable,
    OrderNotSettleable,
    OutOfStock,
    ReturnWindowClosed,
    ShipmentNotActionable,
)
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesIntentError,
)
from world.market_governance_core import (
    GovernanceBindingError,
    GovernanceIntentError,
    GovernanceTransitionError,
)


class VCPWorldClient:
    """Synchronous VCP client for a World ``/vcp`` endpoint.

    Two construction modes:

    * **Real HTTP** — pass ``base_url`` (e.g. ``"http://127.0.0.1:8765"``).
      The client builds its own :class:`httpx.Client`. Used by the
      notebook demo against a live uvicorn server.
    * **In-process** — pass ``http_client`` (any :class:`httpx.Client`
      subclass, including :class:`fastapi.testclient.TestClient`). The
      VCP client uses it directly. Used by unit tests that drive the
      FastAPI app synchronously without a real socket.

    Exactly one of ``base_url`` or ``http_client`` must be provided.

    Args:
        agent_id: VCP address the client signs envelopes as
            (``"merchant:fulfillment"``, ``"buyer"``, …). Used as
            ``Envelope.from_`` when callers use the helper builders.
        base_url: target World HTTP endpoint. No trailing slash.
        http_client: pre-built :class:`httpx.Client` (or subclass).
        timeout: per-request timeout in seconds (real-HTTP mode only).
        trust_env: whether the underlying ``httpx.Client`` should
            consult ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY``.
            **Defaults to** ``False`` — the World service is always
            local, and a corp / loopback proxy (e.g. Clash on
            ``127.0.0.1:7897``) intercepting localhost traffic is a
            common source of mysterious 502s. Real-HTTP mode only.
        owns_client: when ``True``, :meth:`close` closes the underlying
            client. Defaults to ``True`` in real-HTTP mode (we built it)
            and ``False`` in in-process mode (caller owns the lifetime).
    """

    def __init__(
        self,
        *,
        agent_id: str,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 10.0,
        trust_env: bool = False,
        owns_client: bool | None = None,
    ) -> None:
        if (base_url is None) == (http_client is None):
            raise ValueError("provide exactly one of base_url or http_client")
        self.agent_id = agent_id
        if http_client is not None:
            self._client = http_client
            self._owns_client = bool(owns_client) if owns_client is not None else False
        else:
            self._client = httpx.Client(
                base_url=base_url, timeout=timeout, trust_env=trust_env,
            )
            self._owns_client = bool(owns_client) if owns_client is not None else True

    def close(self) -> None:
        """Release the underlying connection pool (when we own it)."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "VCPWorldClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- VCP transport ------------------------------------------------

    def send(self, env: Envelope) -> Envelope | list[Envelope] | None:
        """POST one VCP envelope and parse its ordered reply envelope(s).

        World endpoints normally return one ``world.response``.  Other VCP
        participants, including Platform, may atomically return an ordered
        list containing an acknowledgement and an internal causal request.
        Returns ``None`` when the service produced no reply.

        Raises:
            httpx.HTTPStatusError: non-2xx response from ``/vcp``.
            protocol.SchemaError / UnknownActionKind: reply envelope did
                not pass schema validation.
        """
        response = self._client.post("/vcp", content=to_json(env),
                                     headers={"content-type": "application/json"})
        _raise_typed_status(response)
        body = response.text
        if not body or body == "null":
            return None
        decoded = json.loads(body)
        if isinstance(decoded, list):
            return [
                from_json(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                for item in decoded
            ]
        return from_json(body)

    def post_batch(self, envs: list[Envelope]) -> list[Envelope | None]:
        """POST ``/vcp/batch`` with an ordered list of envelopes."""
        payload = "[" + ",".join(to_json(e) for e in envs) + "]"
        response = self._client.post("/vcp/batch", content=payload,
                                     headers={"content-type": "application/json"})
        _raise_typed_status(response)
        items = response.json()
        out: list[Envelope | None] = []
        for item in items:
            if item is None:
                out.append(None)
                continue
            out.append(_envelope_from_payload(item))
        return out

    def healthz(self) -> bool:
        """Return ``True`` if ``GET /healthz`` reports ``ok``."""
        r = self._client.get("/healthz")
        return r.status_code == 200 and r.json().get("status") == "ok"

    # ---- typed convenience wrappers ----------------------------------

    def read_inventory(
        self, sku_id: str, *,
        msg_id: str | None = None,
        idempotency_key: str | None = None,
        ts: str = "1970-01-01T00:00:00Z",
    ) -> dict[str, Any] | None:
        """Issue ``world.read_inventory`` for ``sku_id`` and return the row dict.

        The reply payload is the raw dict the World service produced
        (an ``InventoryRow`` rendered through ``asdict``). Returns
        ``None`` when the row does not exist.
        """
        env = _build_envelope(
            from_=self.agent_id,
            to="world",
            kind="world.read_inventory",
            payload={"sku_id": sku_id},
            msg_id=msg_id or f"read-inv-{sku_id}",
            idempotency_key=idempotency_key or f"read-inv-{sku_id}",
            ts=ts,
        )
        reply = self.send(env)
        if reply is None:
            return None
        if isinstance(reply, list):
            raise RuntimeError("World inventory read returned multiple VCP replies")
        return reply.action.get("payload")

    def update_inventory(
        self, *,
        sku_id: str,
        merchant_id: str,
        qty_available: int,
        qty_reserved: int = 0,
        msg_id: str | None = None,
        idempotency_key: str | None = None,
        ts: str = "1970-01-01T00:00:00Z",
    ) -> dict[str, Any]:
        """Issue ``world.update_inventory`` upserting the SKU's inventory row.

        Returns the World service's write ack payload
        (``{"table": "inventory", "key": ..., "status": "ok"}``).
        """
        env = _build_envelope(
            from_=self.agent_id,
            to="world",
            kind="world.update_inventory",
            payload={
                "sku_id": sku_id,
                "merchant_id": merchant_id,
                "qty_available": int(qty_available),
                "qty_reserved": int(qty_reserved),
            },
            msg_id=msg_id or f"upd-inv-{sku_id}",
            idempotency_key=idempotency_key or f"upd-inv-{sku_id}",
            ts=ts,
        )
        reply = self.send(env)
        if reply is None:
            return {"status": "no_reply"}
        if isinstance(reply, list):
            raise RuntimeError("World inventory write returned multiple VCP replies")
        return reply.action.get("payload", {})


# --- internal helpers -------------------------------------------------


def _raise_typed_status(response: httpx.Response) -> None:
    """Preserve stable domain errors across a VCP HTTP hop.

    Schema failures deliberately remain ``HTTPStatusError`` so the public
    actor-to-Platform validation contract does not change.
    """

    if response.is_success:
        return
    try:
        payload = response.json()
    except (TypeError, ValueError):
        response.raise_for_status()
        return
    if isinstance(payload, dict):
        kind = payload.get("kind")
        message = str(payload.get("error", response.text))
        if kind == "authorization_error":
            raise LifecycleAuthorizationError(message)
        if kind == "idempotency_conflict":
            raise IdempotencyConflict(message)
        if kind == "catalog_mutation_rejected":
            raise CatalogMutationRejected(message)
        if kind == "match_acceptance_rejected":
            raise MatchAcceptanceRejected(message)
        if kind == "after_sales_validation_error":
            raise AfterSalesIntentError(message)
        if kind == "after_sales_transition_error":
            raise AfterSalesCoreTransitionError(message)
        if kind == "after_sales_reference_rejected":
            raise AfterSalesReferenceRejected(message)
        if kind == "governance_validation_error":
            raise GovernanceIntentError(message)
        if kind == "governance_binding_error":
            raise GovernanceBindingError(message)
        if kind == "governance_transition_error":
            raise GovernanceTransitionError(message)
        if kind == "cart_quote_stale":
            raise CartQuoteStaleError(message)
        if kind == "negotiation_state_error":
            raise NegotiationStateError(message)
        if kind == "insufficient_funds":
            raise InsufficientFunds(message)
        if kind == "out_of_stock":
            raise OutOfStock(message)
        if kind == "order_not_settleable":
            raise OrderNotSettleable(message)
        if kind == "order_not_refundable":
            raise OrderNotRefundable(message)
        if kind == "return_window_closed":
            raise ReturnWindowClosed(message)
        if kind == "invalid_order_transition":
            raise InvalidOrderTransition(message)
        if kind == "dispute_not_actionable":
            raise DisputeNotActionable(message)
        if kind == "fulfillment_not_actionable":
            raise FulfillmentNotActionable(message)
        if kind == "shipment_not_actionable":
            raise ShipmentNotActionable(message)
        if kind == "exchange_not_actionable":
            raise ExchangeNotActionable(message)
        if kind == "order_identity_mismatch":
            raise OrderIdentityMismatch(message)
        if kind == "protocol_event_authority":
            raise ProtocolEventAuthorityError(message)
        if kind == "protocol_event_stale":
            raise ProtocolEventStaleError(message)
    response.raise_for_status()

def _build_envelope(
    *,
    from_: str,
    to: str,
    kind: str,
    payload: dict[str, Any],
    msg_id: str,
    idempotency_key: str,
    ts: str,
) -> Envelope:
    return Envelope(
        msg_id=msg_id,
        ts=ts,
        from_=from_,
        to=to,
        in_reply_to=None,
        idempotency_key=idempotency_key,
        action={"kind": kind, "payload": payload},
    )


def _envelope_from_payload(payload: dict[str, Any]) -> Envelope:
    data = dict(payload)
    if "from" in data:
        data["from_"] = data.pop("from")
    return Envelope(**data)


__all__ = ["VCPWorldClient"]
