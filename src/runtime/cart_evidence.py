"""Exact Runtime, Platform, and World evidence for persistent cart checkout.

The verifier is a reusable CommerceWorld evidence contract.  It proves that a
compact buyer or merchant request accepted by Platform caused the exact World
authorization, quote, and atomic checkout rows found in the replayed snapshot.
It is deliberately independent of benchmark task ids and scoring rubrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from protocol.cart_quote_request import (
    coerce_persistent_cart_quote_request,
    persistent_cart_quote_request_to_dict,
)
from protocol.cart_quote_state import (
    persistent_cart_quote_from_json,
    persistent_cart_quote_to_dict,
)
from protocol.matching import canonical_digest
from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
    wire_envelope_sha256,
)
from world.cart_pricing import (
    cart_quote_id,
    cart_quote_intent_fingerprint,
    cart_quote_request_id,
    normalize_cart_quote_intent,
)
from world.evidence_contracts import authority_operation_key


CART_EVIDENCE_CONTRACT = "commerceworld.persistent-cart.v1"
CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT = (
    "commerceworld.persistent-cart-authorization-prefix.v1"
)
CART_QUOTE_PREFIX_EVIDENCE_CONTRACT = "commerceworld.persistent-cart-quote-prefix.v1"
_ENDPOINT = "platform:checkout"
_DEFAULT_TTL = 10
_PRECLAIMED_COMMIT_IDS_OPTION = "preclaimed_commit_ids"
_ALLOW_REPEATED_QUOTES_OPTION = "allow_repeated_quotes"
_CART_AUTHORITY_PAIRS = {
    ("create_cart_quote_request", "world.create_cart_quote_request"),
    ("issue_cart_quote", "world.issue_cart_quote"),
    ("checkout_cart_quote", "world.checkout_cart_quote"),
}


@dataclass(frozen=True, slots=True)
class VerifiedCartEvidence:
    """One exact request, quote, and checkout authority graph."""

    quote_exchange: LinkedPlatformExchange
    checkout_exchange: LinkedPlatformExchange
    authorization_exchange: LinkedPlatformExchange | None
    request: dict[str, Any] | None
    quote: dict[str, Any]
    order_group: dict[str, Any]
    request_commit: dict[str, Any] | None
    quote_commit: dict[str, Any]
    checkout_commit: dict[str, Any]
    quote_exchanges: tuple[LinkedPlatformExchange, ...]
    quotes: tuple[dict[str, Any], ...]
    quote_commits: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedCartAuthorizationPrefixEvidence:
    """Exact merchant-lane prefix before the evaluated merchant answers.

    This result proves only that the deterministic Buyer authorization request
    reached Platform and produced one authoritative World request row.  It
    deliberately contains no quote or checkout fields, so a task scorer cannot
    turn counterpart setup into evaluated-Merchant quote or settlement credit.
    """

    authorization_exchange: LinkedPlatformExchange
    request: dict[str, Any]
    request_commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedCartQuotePrefixEvidence:
    """Exact authoritative quote prefix before buyer checkout.

    A buyer lane contains one direct quote operation.  A merchant lane also
    contains the preceding buyer authorization operation.  The result exposes
    only those completed operations and cannot be mistaken for settlement
    evidence.
    """

    quote_exchange: LinkedPlatformExchange
    authorization_exchange: LinkedPlatformExchange | None
    request: dict[str, Any] | None
    quote: dict[str, Any]
    request_commit: dict[str, Any] | None
    quote_commit: dict[str, Any]
    quote_exchanges: tuple[LinkedPlatformExchange, ...]
    quotes: tuple[dict[str, Any], ...]
    quote_commits: tuple[dict[str, Any], ...]


def _cart_scoped_world_commits(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Remove only commits already claimed by another exact contract.

    Cart is a composable CommerceWorld operation.  A buyer may complete an
    exact search or match before requesting a quote.  The caller must pass the
    commit ids returned by those independently verified contracts; the global
    scorer closure then proves that every excluded commit was in fact claimed.
    Cart commits themselves can never be excluded here.
    """

    raw = options.get(_PRECLAIMED_COMMIT_IDS_OPTION, ())
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(value, str) or not value for value in raw
    ):
        raise ExactJoinError("preclaimed cart commit ids must be a text sequence")
    excluded = tuple(raw)
    if len(excluded) != len(set(excluded)):
        raise ExactJoinError("preclaimed cart commit ids contain duplicates")
    by_id = {
        str(row.get("commit_id")): row
        for row in context.world_commits
        if isinstance(row.get("commit_id"), str) and row.get("commit_id")
    }
    if len(by_id) != len(context.world_commits):
        raise ExactJoinError("cart World journal has missing or duplicate commit ids")
    missing = set(excluded) - set(by_id)
    if missing:
        raise ExactJoinError("preclaimed cart commit id is absent from World journal")
    for commit_id in excluded:
        row = by_id[commit_id]
        if (row.get("operation"), row.get("authority_action")) in _CART_AUTHORITY_PAIRS:
            raise ExactJoinError("a cart authority commit cannot be preclaimed")
    excluded_set = set(excluded)
    return tuple(row for row in context.world_commits if row.get("commit_id") not in excluded_set)


def _verified_cart_authorization(
    context: ExactJoinContext,
    authorization_exchange: LinkedPlatformExchange,
    *,
    market_id: str,
    buyer_id: str,
    evaluated_actor_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join one Buyer authorization to its exact World row and projections."""

    request_rows = _unique_rows(
        context.final_tables, "persistent_cart_quote_requests", "request_id"
    )
    if len(request_rows) != 1:
        raise ExactJoinError("merchant cart flow needs one World request")
    request = next(iter(request_rows.values()))
    typed_request = coerce_persistent_cart_quote_request(request)
    canonical_request = json.loads(
        _canonical_text(persistent_cart_quote_request_to_dict(typed_request))
    )
    if canonical_request != request:
        raise ExactJoinError("persistent cart request is not canonical")

    request_envelope = authorization_exchange.request
    if (
        request_envelope.get("from") != buyer_id
        or request_envelope.get("to") != _ENDPOINT
        or authorization_exchange.decision.get("actor_id") != buyer_id
        or authorization_exchange.decision.get("platform_endpoint") != _ENDPOINT
    ):
        raise ExactJoinError("cart authorization exchange changed actor or endpoint")
    idempotency_key = _idempotency_key(authorization_exchange)
    auth_payload = _payload(request_envelope)
    allowed_payload_fields = {
        "market_id",
        "mandate_id",
        "lines",
        "fill_policy",
        "backorder_policy",
        "request_ttl_ticks",
    }
    if set(auth_payload) - allowed_payload_fields:
        raise ExactJoinError("cart authorization contains unsupported fields")
    intent = normalize_cart_quote_intent(
        {
            "mandate_id": auth_payload.get("mandate_id"),
            "lines": auth_payload.get("lines"),
            "fill_policy": auth_payload.get("fill_policy", "all_or_none"),
            "backorder_policy": auth_payload.get("backorder_policy", "reject"),
        }
    )
    if auth_payload.get("market_id") != market_id:
        raise ExactJoinError("cart authorization market id changed")
    request_ttl = _positive_int(
        auth_payload.get("request_ttl_ticks", _DEFAULT_TTL),
        "request_ttl_ticks",
    )
    request_fingerprint = canonical_digest(
        {
            "intent": cart_quote_intent_fingerprint(
                intent,
                market_id=market_id,
                quote_ttl_ticks=request_ttl,
            ),
            "request_ttl_ticks": request_ttl,
        }
    )
    expected_lines = tuple((str(line["sku_id"]), int(line["qty"])) for line in intent["lines"])
    persisted_lines = tuple((line.sku_id, line.qty) for line in typed_request.lines)
    if persisted_lines != expected_lines:
        raise ExactJoinError("persisted cart request changed requested SKU quantities")
    if (
        typed_request.market_id != market_id
        or typed_request.buyer_id != buyer_id
        or typed_request.created_by != buyer_id
        or typed_request.issuer_id != "world"
        or typed_request.idempotency_key != idempotency_key
        or typed_request.mandate_id != intent["mandate_id"]
        or typed_request.fill_policy != intent["fill_policy"]
        or typed_request.backorder_policy != intent["backorder_policy"]
        or typed_request.allowed_merchant_ids != (evaluated_actor_id,)
        or any(line.merchant_id != evaluated_actor_id for line in typed_request.lines)
    ):
        raise ExactJoinError("persistent cart request authority binding changed")

    mandate_authorities = _table_rows(context.initial_tables, "mandate_authorities")
    matching_authorities = [
        row for row in mandate_authorities if row.get("mandate_id") == typed_request.mandate_id
    ]
    if len(matching_authorities) != 1:
        raise ExactJoinError("cart request mandate authority is not unique")
    mandate_authority = matching_authorities[0]
    if (
        mandate_authority.get("buyer_id") != typed_request.buyer_id
        or mandate_authority.get("principal_id") != typed_request.principal_id
    ):
        raise ExactJoinError("cart request parties differ from mandate authority")
    mandate_revisions = _table_rows(context.initial_tables, "mandate_revisions")
    matching_revisions = [
        row
        for row in mandate_revisions
        if row.get("mandate_id") == typed_request.mandate_id
        and row.get("revision") == typed_request.mandate_revision
    ]
    if len(matching_revisions) != 1:
        raise ExactJoinError("cart request mandate revision is not unique")
    mandate_revision = matching_revisions[0]
    if (
        mandate_revision.get("buyer_id") != typed_request.buyer_id
        or mandate_revision.get("principal_id") != typed_request.principal_id
        or mandate_revision.get("revision_digest") != typed_request.mandate_digest
    ):
        raise ExactJoinError("cart request mandate revision binding changed")

    initial_tick = _nonnegative_int(
        context.initial_tables.get("logical_time"), "initial logical_time"
    )
    if (
        typed_request.issued_at_tick != initial_tick + 1
        or typed_request.expires_at_tick != typed_request.issued_at_tick + request_ttl
    ):
        raise ExactJoinError("cart request World-clock binding changed")
    expected_request_id = (
        "cart-request:"
        + canonical_digest(
            {
                "market_id": market_id,
                "created_by": buyer_id,
                "idempotency_key": idempotency_key,
                "fingerprint": request_fingerprint,
            }
        )[:32]
    )
    if typed_request.request_id != expected_request_id:
        raise ExactJoinError("cart request identity changed")

    request_commit = _authority_commit(
        context,
        operation="create_cart_quote_request",
        authority_action="world.create_cart_quote_request",
        actor_id=buyer_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        outcome_table="persistent_cart_quote_requests",
        outcome_key=typed_request.request_id,
        outcome_row=request,
    )
    _verify_cart_authorization_commit(
        request_commit,
        request=request,
        market_id=market_id,
        buyer_id=buyer_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        initial_tick=initial_tick,
    )

    if len(authorization_exchange.responses) != 2:
        raise ExactJoinError("cart authorization needs exactly two responses")
    buyer_response = _one_response(
        authorization_exchange,
        kind="platform.cart_quote_request",
        recipient=buyer_id,
    )
    merchant_response = _one_response(
        authorization_exchange,
        kind="platform.cart_quote_request",
        recipient=evaluated_actor_id,
    )
    _verify_cart_authorization_response(
        authorization_exchange,
        buyer_response,
        recipient_id=buyer_id,
        expected_msg_id=f"{request_envelope['msg_id']}:cart-request",
    )
    _verify_cart_authorization_response(
        authorization_exchange,
        merchant_response,
        recipient_id=evaluated_actor_id,
        expected_msg_id=f"{request_envelope['msg_id']}:cart-request:001",
    )
    if _payload(buyer_response) != {"request": request}:
        raise ExactJoinError("buyer cart request response changed the World row")
    expected_projection = {
        "request_id": typed_request.request_id,
        "lines": [{"sku_id": line.sku_id, "qty": line.qty} for line in typed_request.lines],
        "fill_policy": typed_request.fill_policy,
        "backorder_policy": typed_request.backorder_policy,
        "expires_at_tick": typed_request.expires_at_tick,
    }
    if _payload(merchant_response) != {"request": expected_projection}:
        raise ExactJoinError("merchant cart request projection changed")
    return request, request_commit


def _verify_cart_authorization_response(
    exchange: LinkedPlatformExchange,
    response: Mapping[str, Any],
    *,
    recipient_id: str,
    expected_msg_id: str,
) -> None:
    request = exchange.request
    if (
        set(response)
        != {
            "msg_id",
            "ts",
            "from",
            "to",
            "in_reply_to",
            "idempotency_key",
            "action",
            "signature",
        }
        or response.get("msg_id") != expected_msg_id
        or response.get("ts") != request.get("ts")
        or response.get("from") != _ENDPOINT
        or response.get("to") != recipient_id
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
        or response.get("signature") is not None
        or (response.get("action") or {}).get("kind") != "platform.cart_quote_request"
    ):
        raise ExactJoinError("cart authorization response correlation changed")


def _verify_cart_authorization_commit(
    commit: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    market_id: str,
    buyer_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    initial_tick: int,
) -> None:
    request_id = _text(request.get("request_id"), "request_id")
    expected_identity = {
        "schema_version": "cwe.world-commit.v1",
        "commit_kind": "transaction",
        "operation": "create_cart_quote_request",
        "authority_action": "world.create_cart_quote_request",
        "actor_id": buyer_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "subject_id": request_id,
    }
    if any(commit.get(name) != value for name, value in expected_identity.items()):
        raise ExactJoinError("cart authorization commit identity changed")
    if commit.get("invariants_held") != [
        "buyer-or-principal-authorized",
        "mandate-revision-bound",
        "catalog-owner-derived",
        "budget-values-excluded",
        "actor-scoped-idempotency",
    ]:
        raise ExactJoinError("cart authorization commit invariants changed")
    scope = f"cart-quote-request:{market_id}"
    operation_key = authority_operation_key(scope, buyer_id, idempotency_key)
    authority_row = {
        "operation_key": operation_key,
        "scope": scope,
        "actor_id": buyer_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "outcome_table": "persistent_cart_quote_requests",
        "outcome_key": request_id,
        "outcome_listing": None,
    }
    writes = commit.get("table_writes")
    if not isinstance(writes, list) or len(writes) != 3:
        raise ExactJoinError("cart authorization commit write count changed")
    request_write, authority_write, clock_write = writes
    if request_write != {
        "table": "persistent_cart_quote_requests",
        "key": request_id,
        "op": "create",
        "before": None,
        "after": dict(request),
    }:
        raise ExactJoinError("cart authorization request write changed")
    if authority_write != {
        "table": "authority_operations",
        "key": operation_key,
        "op": "create",
        "before": None,
        "after": authority_row,
    }:
        raise ExactJoinError("cart authorization authority write changed")
    if clock_write != {
        "table": "logical_time",
        "key": "world",
        "op": "update",
        "before": initial_tick,
        "after": initial_tick + 1,
    }:
        raise ExactJoinError("cart authorization clock write changed")


def verify_cart_authorization_prefix_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedCartAuthorizationPrefixEvidence:
    """Verify the only valid incomplete prefix of a merchant cart task.

    The accepted prefix is intentionally narrow: one Buyer authorization
    exchange, one canonical request row, one matching World commit, no quote,
    no order group, and no unrelated World commit.  A complete cart flow must
    continue to use :func:`verify_cart_evidence_contract`.
    """

    unknown_options = sorted(
        set(options)
        - {
            "market_id",
            "buyer_id",
            "evaluated_actor_id",
            _PRECLAIMED_COMMIT_IDS_OPTION,
            _ALLOW_REPEATED_QUOTES_OPTION,
        }
    )
    if unknown_options:
        raise ExactJoinError(
            "unknown cart authorization prefix options: " + ", ".join(unknown_options)
        )
    _allow_repeated_quotes(options)
    market_id = _text(options.get("market_id"), "market_id")
    buyer_id = _text(options.get("buyer_id"), "buyer_id")
    evaluated_actor_id = _text(options.get("evaluated_actor_id"), "evaluated_actor_id")
    if not evaluated_actor_id.startswith("merchant:"):
        raise ExactJoinError("cart authorization prefix is merchant-lane only")

    cart_exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("platform_endpoint") == _ENDPOINT
    )
    accepted_exchanges = tuple(
        row for row in cart_exchanges if row.decision.get("decision") == "accepted"
    )
    rejected_exchanges = tuple(
        row for row in cart_exchanges if row.decision.get("decision") == "rejected"
    )
    if len(accepted_exchanges) != 1:
        raise ExactJoinError(
            "cart authorization prefix needs exactly one accepted Platform request"
        )
    if rejected_exchanges or len(cart_exchanges) != 1:
        raise ExactJoinError("cart authorization prefix must precede every evaluated actor request")
    authorization_exchange = _one_exchange(
        accepted_exchanges,
        kind="commerce.create_cart_quote_request",
        actor_id=buyer_id,
    )

    initial_requests = _unique_rows(
        context.initial_tables, "persistent_cart_quote_requests", "request_id"
    )
    initial_quotes = _unique_rows(context.initial_tables, "persistent_cart_quotes", "quote_id")
    initial_groups = _unique_rows(context.initial_tables, "order_groups", "order_group_id")
    request_rows = _unique_rows(
        context.final_tables, "persistent_cart_quote_requests", "request_id"
    )
    quote_rows = _unique_rows(context.final_tables, "persistent_cart_quotes", "quote_id")
    group_rows = _unique_rows(context.final_tables, "order_groups", "order_group_id")
    initial_orders = _unique_rows(context.initial_tables, "orders", "order_id")
    initial_ledger = _unique_rows(context.initial_tables, "ledger", "txn_id")
    final_orders = _unique_rows(context.final_tables, "orders", "order_id")
    final_ledger = _unique_rows(context.final_tables, "ledger", "txn_id")
    if initial_requests or initial_quotes or initial_groups:
        raise ExactJoinError("cart authorization prefix requires a fresh episode")
    if initial_orders or initial_ledger:
        raise ExactJoinError("cart authorization prefix requires no prior settlement")
    if len(request_rows) != 1 or quote_rows or group_rows or final_orders or final_ledger:
        raise ExactJoinError("cart authorization prefix must stop before quote and checkout")
    cart_commits = _cart_scoped_world_commits(context, options)
    if len(cart_commits) != 1:
        raise ExactJoinError("cart authorization prefix has an unexpected World commit")

    request, request_commit = _verified_cart_authorization(
        context,
        authorization_exchange,
        market_id=market_id,
        buyer_id=buyer_id,
        evaluated_actor_id=evaluated_actor_id,
    )
    if request_commit != cart_commits[0]:
        raise ExactJoinError("cart authorization prefix claimed another commit")

    return VerifiedCartAuthorizationPrefixEvidence(
        authorization_exchange=authorization_exchange,
        request=request,
        request_commit=request_commit,
    )


def _allow_repeated_quotes(options: Mapping[str, Any]) -> bool:
    value = options.get(_ALLOW_REPEATED_QUOTES_OPTION, False)
    if not isinstance(value, bool):
        raise ExactJoinError("allow_repeated_quotes must be boolean")
    return value


def _quote_exchanges(
    cart_exchanges: tuple[LinkedPlatformExchange, ...],
    *,
    evaluated_actor_id: str,
    allow_repeated_quotes: bool,
) -> tuple[LinkedPlatformExchange, ...]:
    rows = tuple(
        sorted(
            (
                row
                for row in cart_exchanges
                if row.decision.get("action_kind") == "commerce.request_cart_quote"
                and row.decision.get("actor_id") == evaluated_actor_id
            ),
            key=lambda row: row.request_position,
        )
    )
    if not rows:
        raise ExactJoinError(
            f"cart flow needs one commerce.request_cart_quote exchange for {evaluated_actor_id}"
        )
    if not allow_repeated_quotes and len(rows) != 1:
        raise ExactJoinError(
            f"cart flow needs one commerce.request_cart_quote exchange for {evaluated_actor_id}"
        )
    return rows


def _quote_row_for_exchange(
    quote_rows: Mapping[str, dict[str, Any]],
    quote_exchange: LinkedPlatformExchange,
    *,
    recipient_id: str,
) -> dict[str, Any]:
    response = _one_response(
        quote_exchange,
        kind="platform.cart_quote",
        recipient=recipient_id,
    )
    projected = _payload(response).get("quote")
    quote_id = projected.get("quote_id") if isinstance(projected, Mapping) else None
    if not isinstance(quote_id, str) or not quote_id:
        raise ExactJoinError("cart quote response has no quote identity")
    quote = quote_rows.get(quote_id)
    if quote is None:
        raise ExactJoinError("cart quote response does not identify a World quote")
    return quote


def _verify_quote_attempt(
    context: ExactJoinContext,
    quote_exchange: LinkedPlatformExchange,
    quote: dict[str, Any],
    *,
    market_id: str,
    buyer_id: str,
    evaluated_actor_id: str,
    merchant_lane: bool,
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    typed_quote = persistent_cart_quote_from_json(_canonical_text(quote))
    if persistent_cart_quote_to_dict(typed_quote) != quote:
        raise ExactJoinError("persistent cart quote is not canonical")
    if (
        typed_quote.market_id != market_id
        or typed_quote.buyer_id != buyer_id
        or typed_quote.requested_by != evaluated_actor_id
        or typed_quote.issuer_id != "world"
        or typed_quote.idempotency_key != _idempotency_key(quote_exchange)
    ):
        raise ExactJoinError("cart quote authority binding changed")

    quote_request = quote_exchange.request
    if (
        quote_request.get("from") != evaluated_actor_id
        or quote_request.get("to") != _ENDPOINT
        or quote_exchange.decision.get("actor_id") != evaluated_actor_id
        or quote_exchange.decision.get("platform_endpoint") != _ENDPOINT
    ):
        raise ExactJoinError("cart quote changed actor or endpoint")
    quote_payload = _payload(quote_request)
    quote_ttl = _positive_int(
        quote_payload.get("quote_ttl_ticks", _DEFAULT_TTL),
        "quote_ttl_ticks",
    )
    if merchant_lane:
        if set(quote_payload) - {"request_id", "quote_ttl_ticks"}:
            raise ExactJoinError("merchant submitted non-opaque quote fields")
        if request is None or quote_payload.get("request_id") != request.get("request_id"):
            raise ExactJoinError("merchant quote changed request id")
        quote_fingerprint = canonical_digest(
            {
                "request_id": request["request_id"],
                "request_digest": request["request_digest"],
                "quote_ttl_ticks": quote_ttl,
            }
        )
        expected_expiry = min(
            typed_quote.issued_at_tick + quote_ttl,
            int(request["expires_at_tick"]),
        )
        expected_request_id = str(request["request_id"])
    else:
        allowed_payload_fields = {
            "market_id",
            "mandate_id",
            "lines",
            "fill_policy",
            "backorder_policy",
            "quote_ttl_ticks",
        }
        if set(quote_payload) - allowed_payload_fields:
            raise ExactJoinError("buyer cart quote contains unsupported fields")
        if quote_payload.get("market_id") != market_id:
            raise ExactJoinError("direct quote market id changed")
        quote_intent = normalize_cart_quote_intent(
            {
                "mandate_id": quote_payload.get("mandate_id"),
                "lines": quote_payload.get("lines"),
                "fill_policy": quote_payload.get("fill_policy", "all_or_none"),
                "backorder_policy": quote_payload.get("backorder_policy", "reject"),
            }
        )
        quote_fingerprint = cart_quote_intent_fingerprint(
            quote_intent,
            market_id=market_id,
            quote_ttl_ticks=quote_ttl,
        )
        expected_expiry = typed_quote.issued_at_tick + quote_ttl
        expected_request_id = cart_quote_request_id(
            market_id,
            evaluated_actor_id,
            _idempotency_key(quote_exchange),
        )
    if (
        typed_quote.quote_id
        != cart_quote_id(
            market_id,
            evaluated_actor_id,
            _idempotency_key(quote_exchange),
        )
        or typed_quote.request_id != expected_request_id
        or typed_quote.expires_at_tick != expected_expiry
    ):
        raise ExactJoinError("cart quote identity or expiry changed")
    _verify_quote_mandate_binding(context, typed_quote)

    quote_commit = _authority_commit(
        context,
        operation="issue_cart_quote",
        authority_action="world.issue_cart_quote",
        actor_id=evaluated_actor_id,
        idempotency_key=_idempotency_key(quote_exchange),
        request_fingerprint=quote_fingerprint,
        outcome_table="persistent_cart_quotes",
        outcome_key=typed_quote.quote_id,
        outcome_row=quote,
    )
    _verify_cart_quote_commit(
        quote_commit,
        quote=quote,
        market_id=market_id,
        actor_id=evaluated_actor_id,
        idempotency_key=_idempotency_key(quote_exchange),
        request_fingerprint=quote_fingerprint,
    )
    _verify_quote_prefix_responses(
        quote_exchange,
        quote=quote,
        buyer_id=buyer_id,
        evaluated_actor_id=evaluated_actor_id,
        merchant_lane=merchant_lane,
    )
    return quote_commit


def verify_cart_quote_prefix_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedCartQuotePrefixEvidence:
    """Verify the completed quote prefix immediately before buyer checkout.

    This is deliberately a separate contract from complete cart settlement.
    It accepts only a fresh, replayed cart whose exact Platform exchanges and
    World journal stop after one authoritative quote.  The verifier never
    claims an operation by commit id alone.
    """

    unknown_options = sorted(
        set(options)
        - {
            "market_id",
            "buyer_id",
            "evaluated_actor_id",
            _PRECLAIMED_COMMIT_IDS_OPTION,
            _ALLOW_REPEATED_QUOTES_OPTION,
        }
    )
    if unknown_options:
        raise ExactJoinError("unknown cart quote prefix options: " + ", ".join(unknown_options))
    market_id = _text(options.get("market_id"), "market_id")
    buyer_id = _text(options.get("buyer_id"), "buyer_id")
    evaluated_actor_id = _text(options.get("evaluated_actor_id"), "evaluated_actor_id")
    merchant_lane = evaluated_actor_id.startswith("merchant:")
    if not merchant_lane and evaluated_actor_id != buyer_id:
        raise ExactJoinError("cart quote prefix actor is neither buyer nor merchant")
    allow_repeated_quotes = _allow_repeated_quotes(options)

    cart_exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("platform_endpoint") == _ENDPOINT
    )
    if any(row.decision.get("decision") != "accepted" for row in cart_exchanges):
        raise ExactJoinError("cart quote prefix contains a rejected Platform request")
    quote_exchanges = _quote_exchanges(
        cart_exchanges,
        evaluated_actor_id=evaluated_actor_id,
        allow_repeated_quotes=allow_repeated_quotes,
    )
    authorization_exchange = (
        _one_exchange(
            cart_exchanges,
            kind="commerce.create_cart_quote_request",
            actor_id=buyer_id,
        )
        if merchant_lane
        else None
    )
    expected_exchange_count = len(quote_exchanges) + int(merchant_lane)
    if len(cart_exchanges) != expected_exchange_count:
        raise ExactJoinError("cart quote prefix has unexpected Platform requests")

    initial_requests = _unique_rows(
        context.initial_tables, "persistent_cart_quote_requests", "request_id"
    )
    initial_quotes = _unique_rows(context.initial_tables, "persistent_cart_quotes", "quote_id")
    initial_groups = _unique_rows(context.initial_tables, "order_groups", "order_group_id")
    initial_orders = _unique_rows(context.initial_tables, "orders", "order_id")
    initial_ledger = _unique_rows(context.initial_tables, "ledger", "txn_id")
    request_rows = _unique_rows(
        context.final_tables, "persistent_cart_quote_requests", "request_id"
    )
    quote_rows = _unique_rows(context.final_tables, "persistent_cart_quotes", "quote_id")
    group_rows = _unique_rows(context.final_tables, "order_groups", "order_group_id")
    final_orders = _unique_rows(context.final_tables, "orders", "order_id")
    final_ledger = _unique_rows(context.final_tables, "ledger", "txn_id")
    if initial_requests or initial_quotes or initial_groups:
        raise ExactJoinError("cart quote prefix requires a fresh cart episode")
    if initial_orders or initial_ledger:
        raise ExactJoinError("cart quote prefix requires no prior settlement")
    if len(quote_rows) != len(quote_exchanges):
        raise ExactJoinError("cart quote prefix quote exchanges and World rows differ")
    if group_rows or final_orders or final_ledger:
        raise ExactJoinError("cart quote prefix must stop before checkout")
    if (merchant_lane and len(request_rows) != 1) or (not merchant_lane and request_rows):
        raise ExactJoinError("cart quote prefix has the wrong request-row scope")

    request: dict[str, Any] | None = None
    request_commit: dict[str, Any] | None = None
    if merchant_lane:
        if authorization_exchange is None:
            raise ExactJoinError("merchant quote prefix has no authorization exchange")
        request, request_commit = _verified_cart_authorization(
            context,
            authorization_exchange,
            market_id=market_id,
            buyer_id=buyer_id,
            evaluated_actor_id=evaluated_actor_id,
        )
        if len(authorization_exchange.responses) != len(authorization_exchange.response_positions):
            raise ExactJoinError("cart authorization response positions are incomplete")
        merchant_delivery_positions = [
            position
            for response, position in zip(
                authorization_exchange.responses,
                authorization_exchange.response_positions,
                strict=True,
            )
            if response.get("to") == evaluated_actor_id
            and (response.get("action") or {}).get("kind") == "platform.cart_quote_request"
        ]
        if len(merchant_delivery_positions) != 1 or any(
            exchange.request_position <= merchant_delivery_positions[0]
            for exchange in quote_exchanges
        ):
            raise ExactJoinError("merchant quote request precedes its authorization delivery")

    initial_tick = _nonnegative_int(
        context.initial_tables.get("logical_time"), "initial logical_time"
    )
    quotes = tuple(
        _quote_row_for_exchange(
            quote_rows,
            exchange,
            recipient_id=evaluated_actor_id,
        )
        for exchange in quote_exchanges
    )
    if len({str(row.get("quote_id")) for row in quotes}) != len(quotes):
        raise ExactJoinError("cart quote exchanges identify the same World quote")
    if {str(row.get("quote_id")) for row in quotes} != set(quote_rows):
        raise ExactJoinError("cart quote prefix has an unclaimed World quote")
    quote_commits = tuple(
        _verify_quote_attempt(
            context,
            exchange,
            quote,
            market_id=market_id,
            buyer_id=buyer_id,
            evaluated_actor_id=evaluated_actor_id,
            merchant_lane=merchant_lane,
            request=request,
        )
        for exchange, quote in zip(quote_exchanges, quotes, strict=True)
    )
    observed_sequences = tuple(
        _nonnegative_int(row.get("sequence"), "cart quote commit sequence") for row in quote_commits
    )
    if observed_sequences != tuple(
        range(observed_sequences[0], observed_sequences[0] + len(quote_commits))
    ):
        raise ExactJoinError("cart quote commits are not consecutive")
    if request_commit is not None and observed_sequences[0] != (
        _nonnegative_int(request_commit.get("sequence"), "cart request commit sequence") + 1
    ):
        raise ExactJoinError("merchant cart request and quote commits are not consecutive")
    issue_ticks = tuple(
        _nonnegative_int(row.get("issued_at_tick"), "cart quote issued_at_tick") for row in quotes
    )
    if issue_ticks != tuple(range(issue_ticks[0], issue_ticks[0] + len(quotes))):
        raise ExactJoinError("cart quote World ticks are not consecutive")
    if request is not None and issue_ticks[0] != (
        _nonnegative_int(request.get("issued_at_tick"), "cart request issued_at_tick") + 1
    ):
        raise ExactJoinError("merchant request and quote World ticks are not consecutive")
    final_tick = _nonnegative_int(context.final_tables.get("logical_time"), "final logical_time")
    if final_tick != issue_ticks[-1] or issue_ticks[0] <= initial_tick:
        raise ExactJoinError("cart quote prefix World-clock binding changed")

    claimed_commits = tuple(row for row in (request_commit, *quote_commits) if row is not None)
    if claimed_commits != _cart_scoped_world_commits(context, options):
        raise ExactJoinError("cart quote prefix has an unclaimed or reordered World commit")

    quote_exchange = quote_exchanges[-1]
    quote = quotes[-1]
    quote_commit = quote_commits[-1]

    return VerifiedCartQuotePrefixEvidence(
        quote_exchange=quote_exchange,
        authorization_exchange=authorization_exchange,
        request=request,
        quote=quote,
        request_commit=request_commit,
        quote_commit=quote_commit,
        quote_exchanges=quote_exchanges,
        quotes=quotes,
        quote_commits=quote_commits,
    )


def _verify_quote_mandate_binding(
    context: ExactJoinContext,
    quote: Any,
) -> None:
    authorities = [
        row
        for row in _table_rows(context.initial_tables, "mandate_authorities")
        if row.get("mandate_id") == quote.mandate_id
    ]
    if len(authorities) != 1:
        raise ExactJoinError("cart quote mandate authority is not unique")
    authority = authorities[0]
    if (
        authority.get("buyer_id") != quote.buyer_id
        or authority.get("principal_id") != quote.principal_id
    ):
        raise ExactJoinError("cart quote differs from mandate authority")
    revisions = [
        row
        for row in _table_rows(context.initial_tables, "mandate_revisions")
        if row.get("mandate_id") == quote.mandate_id
        and row.get("revision") == quote.mandate_revision
    ]
    if len(revisions) != 1:
        raise ExactJoinError("cart quote mandate revision is not unique")
    revision = revisions[0]
    if (
        revision.get("buyer_id") != quote.buyer_id
        or revision.get("principal_id") != quote.principal_id
        or revision.get("revision_digest") != quote.mandate_digest
    ):
        raise ExactJoinError("cart quote differs from mandate revision")


def _verify_cart_quote_commit(
    commit: Mapping[str, Any],
    *,
    quote: Mapping[str, Any],
    market_id: str,
    actor_id: str,
    idempotency_key: str,
    request_fingerprint: str,
) -> None:
    quote_id = _text(quote.get("quote_id"), "quote_id")
    expected_identity = {
        "schema_version": "cwe.world-commit.v1",
        "commit_kind": "transaction",
        "operation": "issue_cart_quote",
        "authority_action": "world.issue_cart_quote",
        "actor_id": actor_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "subject_id": quote_id,
    }
    if any(commit.get(name) != value for name, value in expected_identity.items()):
        raise ExactJoinError("cart quote commit identity changed")
    if commit.get("invariants_held") != [
        "world-owned-pricing",
        "principal-mandate-bound",
        "catalog-inventory-policy-snapshots",
        "actor-scoped-idempotency",
    ]:
        raise ExactJoinError("cart quote commit invariants changed")
    scope = f"cart-quote:{market_id}"
    operation_key = authority_operation_key(scope, actor_id, idempotency_key)
    authority_row = {
        "operation_key": operation_key,
        "scope": scope,
        "actor_id": actor_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "outcome_table": "persistent_cart_quotes",
        "outcome_key": quote_id,
        "outcome_listing": None,
    }
    issued_at_tick = _nonnegative_int(quote.get("issued_at_tick"), "issued_at_tick")
    if commit.get("table_writes") != [
        {
            "table": "persistent_cart_quotes",
            "key": quote_id,
            "op": "create",
            "before": None,
            "after": dict(quote),
        },
        {
            "table": "authority_operations",
            "key": operation_key,
            "op": "create",
            "before": None,
            "after": authority_row,
        },
        {
            "table": "logical_time",
            "key": "world",
            "op": "update",
            "before": issued_at_tick - 1,
            "after": issued_at_tick,
        },
    ]:
        raise ExactJoinError("cart quote commit write set changed")


def _merchant_quote_projection(quote: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "quote_id",
        "request_id",
        "market_id",
        "buyer_id",
        "currency",
        "lines",
        "subtotal_minor",
        "charges",
        "shipping_minor",
        "tax_minor",
        "fee_minor",
        "grand_total_minor",
        "issued_at_tick",
        "expires_at_tick",
    )
    return {name: quote[name] for name in fields}


def _verify_quote_prefix_responses(
    exchange: LinkedPlatformExchange,
    *,
    quote: Mapping[str, Any],
    buyer_id: str,
    evaluated_actor_id: str,
    merchant_lane: bool,
) -> None:
    responses = exchange.responses
    request = exchange.request
    if merchant_lane:
        expected_merchant = _expected_quote_response(
            request,
            recipient_id=evaluated_actor_id,
            recipient_quote=_merchant_quote_projection(quote),
            response_index=1,
        )
        expected_buyer = _expected_quote_response(
            request,
            recipient_id=buyer_id,
            recipient_quote=quote,
            response_index=2,
        )
        _verify_quote_decision_response_metadata(
            exchange,
            (expected_merchant, expected_buyer),
        )
        if len(responses) not in {1, 2}:
            raise ExactJoinError("merchant quote prefix has an invalid response count")
        merchant_response = _one_response(
            exchange,
            kind="platform.cart_quote",
            recipient=evaluated_actor_id,
        )
        _verify_quote_response_envelope(
            exchange,
            merchant_response,
            recipient_id=evaluated_actor_id,
            expected_msg_id=f"{exchange.request['msg_id']}:cart-quote:001",
        )
        if _payload(merchant_response) != {"quote": _merchant_quote_projection(quote)}:
            raise ExactJoinError("merchant quote response changed the World projection")
        buyer_responses = [row for row in responses if row.get("to") == buyer_id]
        if len(responses) == 1:
            if buyer_responses:
                raise ExactJoinError("merchant quote prefix lost its merchant response")
            return
        if len(buyer_responses) != 1:
            raise ExactJoinError("merchant quote prefix has no unique buyer response")
        buyer_response = buyer_responses[0]
        _verify_quote_response_envelope(
            exchange,
            buyer_response,
            recipient_id=buyer_id,
            expected_msg_id=f"{exchange.request['msg_id']}:cart-quote:002",
        )
        if _payload(buyer_response) != {"quote": dict(quote)}:
            raise ExactJoinError("buyer quote response changed the World row")
        return

    if len(responses) != 1:
        raise ExactJoinError("buyer quote prefix needs one response")
    expected_buyer = _expected_quote_response(
        request,
        recipient_id=buyer_id,
        recipient_quote=quote,
        response_index=1,
    )
    _verify_quote_decision_response_metadata(exchange, (expected_buyer,))
    buyer_response = _one_response(
        exchange,
        kind="platform.cart_quote",
        recipient=buyer_id,
    )
    _verify_quote_response_envelope(
        exchange,
        buyer_response,
        recipient_id=buyer_id,
        expected_msg_id=f"{exchange.request['msg_id']}:cart-quote:001",
    )
    if _payload(buyer_response) != {"quote": dict(quote)}:
        raise ExactJoinError("buyer quote response changed the World row")


def _expected_quote_response(
    request: Mapping[str, Any],
    *,
    recipient_id: str,
    recipient_quote: Mapping[str, Any],
    response_index: int,
) -> dict[str, Any]:
    return {
        "msg_id": f"{request['msg_id']}:cart-quote:{response_index:03d}",
        "ts": request["ts"],
        "from": _ENDPOINT,
        "to": recipient_id,
        "in_reply_to": request["msg_id"],
        "idempotency_key": request["idempotency_key"],
        "action": {
            "kind": "platform.cart_quote",
            "payload": {"quote": dict(recipient_quote)},
        },
        "signature": None,
    }


def _verify_quote_decision_response_metadata(
    exchange: LinkedPlatformExchange,
    expected_responses: tuple[Mapping[str, Any], ...],
) -> None:
    expected_ids = [str(row["msg_id"]) for row in expected_responses]
    expected_kinds = ["platform.cart_quote"] * len(expected_responses)
    expected_hashes = [wire_envelope_sha256(row) for row in expected_responses]
    decision = exchange.decision
    if (
        decision.get("response_msg_ids") != expected_ids
        or decision.get("response_kinds") != expected_kinds
        or decision.get("response_sha256s") != expected_hashes
    ):
        raise ExactJoinError("cart quote decision changed produced response metadata")


def _verify_quote_response_envelope(
    exchange: LinkedPlatformExchange,
    response: Mapping[str, Any],
    *,
    recipient_id: str,
    expected_msg_id: str,
) -> None:
    request = exchange.request
    if (
        set(response)
        != {
            "msg_id",
            "ts",
            "from",
            "to",
            "in_reply_to",
            "idempotency_key",
            "action",
            "signature",
        }
        or response.get("msg_id") != expected_msg_id
        or response.get("ts") != request.get("ts")
        or response.get("from") != _ENDPOINT
        or response.get("to") != recipient_id
        or response.get("in_reply_to") != request.get("msg_id")
        or response.get("idempotency_key") != request.get("idempotency_key")
        or response.get("signature") is not None
        or (response.get("action") or {}).get("kind") != "platform.cart_quote"
    ):
        raise ExactJoinError("cart quote response correlation changed")


def verify_cart_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedCartEvidence:
    unknown_options = sorted(
        set(options)
        - {
            "market_id",
            "buyer_id",
            "evaluated_actor_id",
            _PRECLAIMED_COMMIT_IDS_OPTION,
            _ALLOW_REPEATED_QUOTES_OPTION,
        }
    )
    if unknown_options:
        raise ExactJoinError(
            "unknown persistent cart evidence options: " + ", ".join(unknown_options)
        )
    # Validate the cross-contract partition, while leaving unrelated commits
    # to the caller's global authority-closure check.  This contract claims
    # only the exact cart operations it returns below.
    cart_scoped_commits = _cart_scoped_world_commits(context, options)
    market_id = _text(options.get("market_id"), "market_id")
    buyer_id = _text(options.get("buyer_id"), "buyer_id")
    evaluated_actor_id = _text(options.get("evaluated_actor_id"), "evaluated_actor_id")
    merchant_lane = evaluated_actor_id.startswith("merchant:")
    if not merchant_lane and evaluated_actor_id != buyer_id:
        raise ExactJoinError("persistent cart evaluated actor is neither buyer nor merchant")
    allow_repeated_quotes = _allow_repeated_quotes(options)

    cart_exchanges = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("platform_endpoint") == _ENDPOINT
    )
    if any(row.decision.get("decision") != "accepted" for row in cart_exchanges):
        raise ExactJoinError("persistent cart flow contains a rejected Platform request")

    quote_exchanges = _quote_exchanges(
        cart_exchanges,
        evaluated_actor_id=evaluated_actor_id,
        allow_repeated_quotes=allow_repeated_quotes,
    )
    checkout_exchange = _one_exchange(
        cart_exchanges,
        kind="platform.checkout_cart",
        actor_id=buyer_id,
    )
    authorization_exchange = (
        _one_exchange(
            cart_exchanges,
            kind="commerce.create_cart_quote_request",
            actor_id=buyer_id,
        )
        if merchant_lane
        else None
    )
    expected_exchange_count = len(quote_exchanges) + 1 + int(merchant_lane)
    if len(cart_exchanges) != expected_exchange_count:
        raise ExactJoinError("persistent cart flow has unexpected Platform requests")

    request_rows = _unique_rows(
        context.final_tables, "persistent_cart_quote_requests", "request_id"
    )
    quote_rows = _unique_rows(context.final_tables, "persistent_cart_quotes", "quote_id")
    group_rows = _unique_rows(context.final_tables, "order_groups", "order_group_id")
    initial_requests = _unique_rows(
        context.initial_tables, "persistent_cart_quote_requests", "request_id"
    )
    initial_quotes = _unique_rows(context.initial_tables, "persistent_cart_quotes", "quote_id")
    initial_groups = _unique_rows(context.initial_tables, "order_groups", "order_group_id")
    if initial_requests or initial_quotes or initial_groups:
        raise ExactJoinError("selected cart evidence requires a fresh episode scope")

    request: dict[str, Any] | None = None
    request_commit: dict[str, Any] | None = None
    if merchant_lane:
        if len(request_rows) != 1 or authorization_exchange is None:
            raise ExactJoinError("merchant cart flow needs one World request")
        request, request_commit = _verified_cart_authorization(
            context,
            authorization_exchange,
            market_id=market_id,
            buyer_id=buyer_id,
            evaluated_actor_id=evaluated_actor_id,
        )
    elif request_rows:
        raise ExactJoinError("direct buyer cart flow created a quote request row")

    if len(quote_rows) != len(quote_exchanges) or len(group_rows) != 1:
        raise ExactJoinError("cart quote exchanges, World quotes, and order-group scope differ")
    quotes = tuple(
        _quote_row_for_exchange(
            quote_rows,
            exchange,
            recipient_id=evaluated_actor_id,
        )
        for exchange in quote_exchanges
    )
    quote_ids = tuple(str(row.get("quote_id")) for row in quotes)
    if len(set(quote_ids)) != len(quote_ids) or set(quote_ids) != set(quote_rows):
        raise ExactJoinError("cart quote exchanges do not claim every World quote exactly")
    initial_tick = _nonnegative_int(
        context.initial_tables.get("logical_time"), "initial logical_time"
    )
    quote_commits = tuple(
        _verify_quote_attempt(
            context,
            exchange,
            quote_row,
            market_id=market_id,
            buyer_id=buyer_id,
            evaluated_actor_id=evaluated_actor_id,
            merchant_lane=merchant_lane,
            request=request,
        )
        for exchange, quote_row in zip(quote_exchanges, quotes, strict=True)
    )
    quote_sequences = tuple(
        _nonnegative_int(row.get("sequence"), "cart quote commit sequence") for row in quote_commits
    )
    if quote_sequences != tuple(range(quote_sequences[0], quote_sequences[0] + len(quote_commits))):
        raise ExactJoinError("cart quote commits are not consecutive")
    if request_commit is not None and quote_sequences[0] != (
        _nonnegative_int(request_commit.get("sequence"), "cart request commit sequence") + 1
    ):
        raise ExactJoinError("merchant cart request and quote commits are not consecutive")
    issue_ticks = tuple(
        _nonnegative_int(row.get("issued_at_tick"), "cart quote issued_at_tick") for row in quotes
    )
    if (
        issue_ticks != tuple(range(issue_ticks[0], issue_ticks[0] + len(quotes)))
        or issue_ticks[0] <= initial_tick
    ):
        raise ExactJoinError("cart quote World ticks are not consecutive")
    if request is not None and issue_ticks[0] != (
        _nonnegative_int(request.get("issued_at_tick"), "cart request issued_at_tick") + 1
    ):
        raise ExactJoinError("merchant request and quote World ticks are not consecutive")

    checkout_payload = _payload(checkout_exchange.request)
    checkout_quote_id = checkout_payload.get("quote_id")
    if set(checkout_payload) != {"quote_id"} or checkout_quote_id not in quote_ids:
        raise ExactJoinError("checkout must contain only one verified persisted quote id")
    selected_index = quote_ids.index(str(checkout_quote_id))
    quote_exchange = quote_exchanges[selected_index]
    quote = quotes[selected_index]
    quote_commit = quote_commits[selected_index]
    order_group = next(iter(group_rows.values()))
    group_id = _text(order_group.get("order_group_id"), "order_group_id")
    checkout_commit = _authority_commit(
        context,
        operation="checkout_cart_quote",
        authority_action="world.checkout_cart_quote",
        actor_id=buyer_id,
        idempotency_key=_idempotency_key(checkout_exchange),
        request_fingerprint=canonical_digest({"quote_id": checkout_quote_id}),
        outcome_table="order_groups",
        outcome_key=group_id,
        outcome_row=order_group,
    )
    if (
        _nonnegative_int(checkout_commit.get("sequence"), "cart checkout commit sequence")
        != quote_sequences[-1] + 1
    ):
        raise ExactJoinError("cart quote and checkout commits are not consecutive")
    settlement = _one_response(
        checkout_exchange,
        kind="platform.cart_settlement",
        recipient=buyer_id,
    )
    if _payload(settlement).get("order_group") != order_group:
        raise ExactJoinError("cart settlement response changed the World group")

    claimed_cart_commits = tuple(
        row for row in (request_commit, *quote_commits, checkout_commit) if row is not None
    )
    observed_cart_commits = tuple(
        row
        for row in cart_scoped_commits
        if (row.get("operation"), row.get("authority_action")) in _CART_AUTHORITY_PAIRS
    )
    if claimed_cart_commits != observed_cart_commits:
        raise ExactJoinError("cart flow has an unclaimed or reordered cart commit")

    return VerifiedCartEvidence(
        quote_exchange=quote_exchange,
        checkout_exchange=checkout_exchange,
        authorization_exchange=authorization_exchange,
        request=request,
        quote=quote,
        order_group=order_group,
        request_commit=request_commit,
        quote_commit=quote_commit,
        checkout_commit=checkout_commit,
        quote_exchanges=quote_exchanges,
        quotes=quotes,
        quote_commits=quote_commits,
    )


def _authority_commit(
    context: ExactJoinContext,
    *,
    operation: str,
    authority_action: str,
    actor_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    outcome_table: str,
    outcome_key: str,
    outcome_row: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        row
        for row in context.world_commits
        if row.get("operation") == operation
        and row.get("authority_action") == authority_action
        and row.get("actor_id") == actor_id
        and row.get("idempotency_key") == idempotency_key
        and row.get("request_fingerprint") == request_fingerprint
        and row.get("subject_id") == outcome_key
    ]
    if len(matches) != 1:
        raise ExactJoinError(f"{operation} has no unique authority commit")
    commit = matches[0]
    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        raise ExactJoinError(f"{operation} commit has no table writes")
    outcomes = [
        row
        for row in writes
        if isinstance(row, Mapping)
        and row.get("table") == outcome_table
        and row.get("key") == outcome_key
        and row.get("op") == "create"
    ]
    if len(outcomes) != 1 or outcomes[0].get("after") != outcome_row:
        raise ExactJoinError(f"{operation} outcome write changed")
    authority_rows = [
        row
        for row in writes
        if isinstance(row, Mapping)
        and row.get("table") == "authority_operations"
        and isinstance(row.get("after"), Mapping)
    ]
    if len(authority_rows) != 1:
        raise ExactJoinError(f"{operation} has no unique authority operation write")
    authority = authority_rows[0]["after"]
    expected = {
        "actor_id": actor_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "outcome_table": outcome_table,
        "outcome_key": outcome_key,
    }
    if any(authority.get(name) != value for name, value in expected.items()):
        raise ExactJoinError(f"{operation} authority operation changed")
    return commit


def _one_exchange(
    exchanges: tuple[LinkedPlatformExchange, ...],
    *,
    kind: str,
    actor_id: str,
) -> LinkedPlatformExchange:
    rows = [
        row
        for row in exchanges
        if row.decision.get("action_kind") == kind and row.decision.get("actor_id") == actor_id
    ]
    if len(rows) != 1:
        raise ExactJoinError(f"cart flow needs one {kind} exchange for {actor_id}")
    return rows[0]


def _one_response(
    exchange: LinkedPlatformExchange,
    *,
    kind: str,
    recipient: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in exchange.responses
        if row.get("to") == recipient and (row.get("action") or {}).get("kind") == kind
    ]
    if len(rows) != 1:
        raise ExactJoinError(f"cart exchange needs one {kind} response to {recipient}")
    return rows[0]


def _unique_rows(
    tables: Mapping[str, Any], table: str, key_field: str
) -> dict[str, dict[str, Any]]:
    raw = _table_rows(tables, table)
    rows: dict[str, dict[str, Any]] = {}
    for value in raw:
        row = value
        key = row.get(key_field)
        if not isinstance(key, str) or not key or key in rows:
            raise ExactJoinError(f"World table {table!r} has invalid row identity")
        rows[key] = row
    return rows


def _table_rows(tables: Mapping[str, Any], table: str) -> list[dict[str, Any]]:
    raw = tables.get(table, [])
    if not isinstance(raw, list):
        raise ExactJoinError(f"World table {table!r} must be an array")
    rows: list[dict[str, Any]] = []
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"World table {table!r} row {ordinal} is invalid")
        rows.append(dict(value))
    return rows


def _payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("cart action payload must be an object")
    return dict(payload)


def _idempotency_key(exchange: LinkedPlatformExchange) -> str:
    return _text(exchange.request.get("idempotency_key"), "idempotency_key")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExactJoinError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExactJoinError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError(f"{label} must be a non-negative integer")
    return value


def _canonical_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(CART_EVIDENCE_CONTRACT, verify_cart_evidence_contract)
)
DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT,
        verify_cart_authorization_prefix_evidence_contract,
    )
)
DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        CART_QUOTE_PREFIX_EVIDENCE_CONTRACT,
        verify_cart_quote_prefix_evidence_contract,
    )
)


__all__ = [
    "CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT",
    "CART_EVIDENCE_CONTRACT",
    "CART_QUOTE_PREFIX_EVIDENCE_CONTRACT",
    "VerifiedCartAuthorizationPrefixEvidence",
    "VerifiedCartEvidence",
    "VerifiedCartQuotePrefixEvidence",
    "verify_cart_authorization_prefix_evidence_contract",
    "verify_cart_evidence_contract",
    "verify_cart_quote_prefix_evidence_contract",
]
