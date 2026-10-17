"""FastAPI adapter for the in-process runtime."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from protocol.envelope import Envelope, to_json, validate
from protocol.errors import PartitionViolation, SchemaError, UnknownActionKind


def create_app(runtime: Any) -> FastAPI:
    app = FastAPI(title="Commerce Runtime")

    @app.exception_handler(PartitionViolation)
    async def _partition(_req: Any, exc: PartitionViolation) -> JSONResponse:
        # 403 Forbidden — the router rejected the envelope. Same wire
        # shape as the world FastAPI returns so callers can branch on
        # ``response.json()["kind"] == "partition_violation"``.
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

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/vcp")
    def vcp(payload: dict[str, Any]) -> Any:
        return _encode_result(runtime.send(_envelope_from_body(payload)))

    @app.post("/vcp/batch")
    def vcp_batch(payloads: list[dict[str, Any]]) -> list[Any]:
        return [_encode_result(runtime.send(_envelope_from_body(payload))) for payload in payloads]

    return app


def _envelope_from_body(payload: dict[str, Any]) -> Envelope:
    data = dict(payload)
    if "from" in data:
        data["from_"] = data.pop("from")
    env = Envelope(**data)
    validate(env)
    return env


def _encode_result(result: Envelope | list[Envelope] | None) -> Any:
    if result is None:
        return None
    if isinstance(result, list):
        return [json.loads(to_json(env)) for env in result]
    return json.loads(to_json(result))
