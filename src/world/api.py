"""FastAPI adapter for the World VCP service."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from protocol.cart_quote_state import CartQuoteStaleError
from protocol.envelope import Envelope, to_json, validate
from protocol.errors import PartitionViolation, SchemaError, UnknownActionKind
from protocol.event_receipts import (
    ProtocolEventAuthorityError,
    ProtocolEventStaleError,
)
from protocol.matching import MatchAcceptanceRejected
from protocol.negotiation_state import NegotiationStateError
from protocol.evidence_records import EvidenceReadDenied
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
    WriteNotAuthorized,
)
from world.after_sales_core import (
    AfterSalesCoreTransitionError,
    AfterSalesIntentError,
)
from world.after_sales_persistence import AfterSalesPersistenceError
from world.after_sales_service import AfterSalesServiceError
from world.payment_fulfillment import PaymentFulfillmentError
from world.market_governance_core import (
    GovernanceBindingError,
    GovernanceIntentError,
    GovernanceTransitionError,
)
from world.market_governance_persistence import MarketGovernancePersistenceError
from world.market_governance_service import MarketGovernanceServiceError
from world.service import WorldService


_WORLD_DOMAIN_CONFLICT_KINDS: dict[type[Exception], str] = {
    InsufficientFunds: "insufficient_funds",
    OutOfStock: "out_of_stock",
    OrderNotSettleable: "order_not_settleable",
    OrderNotRefundable: "order_not_refundable",
    ReturnWindowClosed: "return_window_closed",
    InvalidOrderTransition: "invalid_order_transition",
    DisputeNotActionable: "dispute_not_actionable",
    FulfillmentNotActionable: "fulfillment_not_actionable",
    ShipmentNotActionable: "shipment_not_actionable",
    ExchangeNotActionable: "exchange_not_actionable",
    OrderIdentityMismatch: "order_identity_mismatch",
    ProtocolEventAuthorityError: "protocol_event_authority",
    ProtocolEventStaleError: "protocol_event_stale",
}


def _world_domain_conflict_kind(exc: Exception) -> str:
    """Return the most specific registered wire kind for a domain conflict."""

    for error_type in type(exc).__mro__:
        kind = _WORLD_DOMAIN_CONFLICT_KINDS.get(error_type)
        if kind is not None:
            return kind
    raise TypeError(f"unregistered World domain conflict {type(exc).__name__}")


def create_app(service: WorldService) -> FastAPI:
    app = FastAPI(title="Commerce World")

    @app.exception_handler(PartitionViolation)
    async def _partition(_req: Any, exc: PartitionViolation) -> JSONResponse:
        # 403 Forbidden — the sender role is not on the allow-list for
        # this action against ``world``. WORLD_DESIGN.md §6 invariant
        # ("Permission: sender role may perform the world.* action").
        return JSONResponse(status_code=403, content={"error": str(exc),
                                                       "kind": "partition_violation"})

    @app.exception_handler(SchemaError)
    async def _schema(_req: Any, exc: SchemaError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc),
                                                       "kind": "schema_error"})

    @app.exception_handler(NegotiationStateError)
    async def _negotiation_state(
        _req: Any, exc: NegotiationStateError
    ) -> JSONResponse:
        # Preserve an expected negotiation state rejection across the internal
        # Platform-to-World VCP hop.  The Platform converts this typed domain
        # outcome into its own rejected decision; unrelated schema failures
        # remain ordinary HTTP validation errors.
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "kind": "negotiation_state_error"},
        )

    @app.exception_handler(UnknownActionKind)
    async def _kind(_req: Any, exc: UnknownActionKind) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc),
                                                       "kind": "unknown_action_kind"})

    @app.exception_handler(IdempotencyConflict)
    async def _idempotency(
        _req: Any, exc: IdempotencyConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "kind": "idempotency_conflict"},
        )

    @app.exception_handler(MatchAcceptanceRejected)
    async def _match_acceptance_rejected(
        _req: Any, exc: MatchAcceptanceRejected
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "kind": "match_acceptance_rejected",
            },
        )

    @app.exception_handler(CatalogMutationRejected)
    async def _catalog_conflict(
        _req: Any, exc: CatalogMutationRejected
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "kind": "catalog_mutation_rejected"},
        )

    @app.exception_handler(CartQuoteStaleError)
    async def _cart_quote_stale(
        _req: Any, exc: CartQuoteStaleError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc), "kind": "cart_quote_stale"},
        )

    @app.exception_handler(InsufficientFunds)
    @app.exception_handler(OutOfStock)
    @app.exception_handler(OrderNotSettleable)
    @app.exception_handler(OrderNotRefundable)
    @app.exception_handler(ReturnWindowClosed)
    @app.exception_handler(InvalidOrderTransition)
    @app.exception_handler(DisputeNotActionable)
    @app.exception_handler(FulfillmentNotActionable)
    @app.exception_handler(ShipmentNotActionable)
    @app.exception_handler(ExchangeNotActionable)
    @app.exception_handler(OrderIdentityMismatch)
    @app.exception_handler(ProtocolEventAuthorityError)
    @app.exception_handler(ProtocolEventStaleError)
    async def _world_domain_conflict(
        _req: Any,
        exc: (
            InsufficientFunds
            | OutOfStock
            | OrderNotSettleable
            | OrderNotRefundable
            | ReturnWindowClosed
            | InvalidOrderTransition
            | DisputeNotActionable
            | FulfillmentNotActionable
            | ShipmentNotActionable
            | ExchangeNotActionable
            | OrderIdentityMismatch
            | ProtocolEventAuthorityError
            | ProtocolEventStaleError
        ),
    ) -> JSONResponse:
        """Preserve expected, side-effect-free World rejections over HTTP.

        Only explicit actor-triggered domain conflicts are registered here.
        The broad ``WorldError`` base, schema and authorization failures, and
        internal after-sales/governance service failures keep their existing
        security or infrastructure classifications.
        """

        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "kind": _world_domain_conflict_kind(exc),
            },
        )

    @app.exception_handler(AfterSalesIntentError)
    async def _after_sales_validation(
        _req: Any, exc: AfterSalesIntentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "kind": "after_sales_validation_error",
            },
        )

    @app.exception_handler(AfterSalesCoreTransitionError)
    async def _after_sales_transition(
        _req: Any, exc: AfterSalesCoreTransitionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "kind": "after_sales_transition_error",
            },
        )

    @app.exception_handler(AfterSalesReferenceRejected)
    async def _after_sales_reference_rejected(
        _req: Any,
        exc: AfterSalesReferenceRejected,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": exc.reason_code,
                "kind": "after_sales_reference_rejected",
            },
        )

    @app.exception_handler(AfterSalesServiceError)
    @app.exception_handler(PaymentFulfillmentError)
    @app.exception_handler(AfterSalesPersistenceError)
    async def _after_sales_internal(
        _req: Any,
        _exc: (
            AfterSalesServiceError
            | PaymentFulfillmentError
            | AfterSalesPersistenceError
        ),
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal after-sales failure",
                "kind": "after_sales_internal_error",
            },
        )

    @app.exception_handler(GovernanceIntentError)
    async def _governance_validation(
        _req: Any, exc: GovernanceIntentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "kind": "governance_validation_error",
            },
        )

    @app.exception_handler(GovernanceBindingError)
    async def _governance_binding(
        _req: Any, exc: GovernanceBindingError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "kind": "governance_binding_error",
            },
        )

    @app.exception_handler(GovernanceTransitionError)
    async def _governance_transition(
        _req: Any, exc: GovernanceTransitionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error": str(exc),
                "kind": "governance_transition_error",
            },
        )

    @app.exception_handler(MarketGovernanceServiceError)
    @app.exception_handler(MarketGovernancePersistenceError)
    async def _governance_internal(
        _req: Any,
        _exc: MarketGovernanceServiceError | MarketGovernancePersistenceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal governance failure",
                "kind": "governance_internal_error",
            },
        )

    @app.exception_handler(LifecycleAuthorizationError)
    @app.exception_handler(WriteNotAuthorized)
    @app.exception_handler(EvidenceReadDenied)
    async def _authorization(
        _req: Any,
        exc: LifecycleAuthorizationError | WriteNotAuthorized | EvidenceReadDenied,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"error": str(exc), "kind": "authorization_error"},
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/vcp")
    def vcp(payload: dict[str, Any]) -> Any:
        env = _envelope_from_body(payload)
        result = service.handle(env)
        return _encode_result(result, request_kind=str(env.action.get("kind", "")))

    @app.post("/vcp/batch")
    def vcp_batch(payloads: list[dict[str, Any]]) -> list[Any]:
        encoded: list[Any] = []
        for payload in payloads:
            env = _envelope_from_body(payload)
            encoded.append(
                _encode_result(
                    service.handle(env),
                    request_kind=str(env.action.get("kind", "")),
                )
            )
        return encoded

    return app


def _envelope_from_body(payload: dict[str, Any]) -> Envelope:
    data = dict(payload)
    if "from" in data:
        data["from_"] = data.pop("from")
    env = Envelope(**data)
    validate(env)
    return env


def _encode_result(
    result: Envelope | list[Envelope] | None,
    *,
    request_kind: str = "",
) -> Any:
    if result is None:
        return None
    if isinstance(result, list):
        return [_encode_envelope(env, request_kind=request_kind) for env in result]
    return _encode_envelope(result, request_kind=request_kind)


def _encode_envelope(env: Envelope, *, request_kind: str) -> dict[str, Any]:
    """Encode one reply, applying transport-only compatibility projections."""
    encoded = json.loads(to_json(env))
    if (
        request_kind == "world.read_inventory"
        and encoded.get("action", {}).get("kind") == "world.response"
    ):
        payload = encoded["action"].get("payload")
        if isinstance(payload, dict):
            encoded["action"]["payload"] = {
                field: payload[field]
                for field in (
                    "sku_id",
                    "merchant_id",
                    "qty_available",
                    "qty_reserved",
                )
                if field in payload
            }
    return encoded
