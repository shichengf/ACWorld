"""FastAPI adapter fronting the :class:`~agents.platform.PlatformService` behind
``/vcp`` — the marketplace operator (aggregator + PSP + reputation + catalog) as
its own VCP server for the item-7 four-process launcher.

So buyer / merchant reach the platform over the wire like any other VCP
participant (``commerce.search`` → ``platform:aggregator``,
``platform.settle_payment`` → ``platform:psp``, …).

RESIDUAL (item 4c, deferred): the platform's OWN access to the world is still an
in-process call (``self._world.settle_order`` / catalog reads), so this server
MUST be constructed with the SAME ``World`` object the world server fronts. The
buyer/merchant ↔ platform boundary is genuinely over VCP here; only the
platform's internal world access is co-located until ``world.settle_order`` /
catalog reads move over VCP.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from protocol.envelope import Envelope, to_json, validate
from protocol.errors import PartitionViolation, SchemaError, UnknownActionKind
from protocol.matching import MatchAcceptanceRejected
from world.errors import (
    IdempotencyConflict,
    LifecycleAuthorizationError,
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


def create_platform_app(platform: Any) -> FastAPI:
    """A ``/vcp`` server delegating to ``platform.handle`` (a ``PlatformService``)."""
    app = FastAPI(title="Commerce Platform")

    @app.exception_handler(PartitionViolation)
    async def _partition(_req: Any, exc: PartitionViolation) -> JSONResponse:
        return JSONResponse(status_code=403, content={"error": str(exc),
                                                       "kind": "partition_violation"})

    @app.exception_handler(SchemaError)
    async def _schema(_req: Any, exc: SchemaError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc),
                                                       "kind": "schema_error"})

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

    @app.exception_handler(LifecycleAuthorizationError)
    @app.exception_handler(WriteNotAuthorized)
    async def _authorization(
        _req: Any,
        exc: LifecycleAuthorizationError | WriteNotAuthorized,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={"error": str(exc), "kind": "authorization_error"},
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

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/vcp")
    def vcp(payload: dict[str, Any]) -> Any:
        env = _envelope_from_body(payload)
        return _encode_result(platform.handle(env))

    return app


def _envelope_from_body(payload: dict[str, Any]) -> Envelope:
    data = dict(payload)
    if "from" in data:
        data["from_"] = data.pop("from")
    env = Envelope(**data)
    validate(env)
    return env


def _encode_result(result: "Envelope | list[Envelope] | None") -> Any:
    if result is None:
        return None
    if isinstance(result, list):
        return [json.loads(to_json(env)) for env in result]
    return json.loads(to_json(result))
