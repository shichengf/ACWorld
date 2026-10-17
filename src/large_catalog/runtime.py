"""ACWorld execution path for the large-catalog stress suite."""

from __future__ import annotations

import heapq
import json
import time
from typing import Any, Mapping, Protocol, Sequence

from agents.inference import InferenceChannel
from large_catalog.database import CatalogDataError, CatalogDatabase
from large_catalog.models import CatalogListing, LargeCatalogTask, SearchRequest
from protocol.actions import ActionKind, is_send_allowed
from protocol.envelope import Envelope, validate


class LargeCatalogRuntimeError(RuntimeError):
    """The large-catalog Agent/Platform/World path failed."""


MAX_CART_REQUIREMENT_CANDIDATES = 1_000


class DecisionPolicy(Protocol):
    def decide(
        self,
        task: LargeCatalogTask,
        observations: Sequence[Mapping[str, Any]],
        allowed_intents: Sequence[str],
        *,
        decision_id: str,
    ) -> Mapping[str, Any]: ...


_INTENT_ACTION = {
    "search_catalog": ActionKind.COMMERCE_LARGE_CATALOG_SEARCH,
    "search_cart_candidates": ActionKind.COMMERCE_LARGE_CATALOG_CART_SEARCH,
    "observe_listing": ActionKind.COMMERCE_LARGE_CATALOG_OBSERVE,
}


def _role(address: str) -> str:
    return address.split(":", 1)[0]


def _envelope(
    *,
    sequence: int,
    sender: str,
    recipient: str,
    kind: ActionKind,
    payload: Mapping[str, Any],
    reply_to: str | None = None,
) -> Envelope:
    env = Envelope(
        msg_id=f"lc-msg-{sequence:04d}",
        ts=f"2026-01-01T00:00:{sequence:02d}Z",
        from_=sender,
        to=recipient,
        in_reply_to=reply_to,
        idempotency_key=f"lc-idem-{sequence:04d}",
        action={"kind": kind.value, "payload": dict(payload)},
    )
    validate(env)
    if not is_send_allowed(kind, _role(sender), _role(recipient)):
        raise LargeCatalogRuntimeError(
            f"VCP partition rejects {sender} -> {recipient} {kind.value}"
        )
    return env


class LargeCatalogWorld:
    """Authoritative catalog plus one isolated transaction overlay."""

    def __init__(self, database: CatalogDatabase, task: LargeCatalogTask) -> None:
        self.database = database
        self.task = task
        self.effects: list[dict[str, Any]] = []
        self.quotes: dict[str, dict[str, Any]] = {}

    def handle(self, envelope: Envelope, *, sequence: int) -> Envelope:
        validate(envelope)
        kind = ActionKind(envelope.action["kind"])
        payload = dict(envelope.action["payload"])
        if kind is ActionKind.WORLD_LARGE_CATALOG_SEARCH:
            raw_filters = payload.get("filters")
            if raw_filters is None:
                filters: dict[str, Any] = {}
            elif isinstance(raw_filters, Mapping):
                filters = dict(raw_filters)
            else:
                raise CatalogDataError("catalog search filters must be an object")
            raw_page_size = payload.get("page_size")
            result = self.database.search(
                SearchRequest(
                    query=str(payload.get("query") or ""),
                    filters=filters,
                    sort=str(payload.get("sort") or "price_asc"),
                    cursor=payload.get("cursor"),
                    page_size=20 if raw_page_size is None else int(raw_page_size),
                )
            ).to_public_dict()
        elif kind is ActionKind.WORLD_LARGE_CATALOG_CART_SEARCH:
            result = self._search_carts(payload)
        elif kind is ActionKind.WORLD_LARGE_CATALOG_OBSERVE:
            ref = str(payload.get("listing_ref", ""))
            listing = self.database.listing(ref)
            result = (
                {"found": False, "listing_ref": ref}
                if listing is None
                else {"found": True, "listing": listing.public_record()}
            )
        elif kind is ActionKind.WORLD_LARGE_CATALOG_COMMIT:
            result = self._commit(payload)
        else:
            raise LargeCatalogRuntimeError(f"World cannot handle {kind.value}")
        return _envelope(
            sequence=sequence,
            sender="world",
            recipient=envelope.from_,
            kind=ActionKind.WORLD_RESPONSE,
            payload={"ok": True, "result": result},
            reply_to=envelope.msg_id,
        )

    def _search_carts(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        requirements = payload.get("requirements")
        constraints = payload.get("constraints", {})
        if (
            not isinstance(requirements, Sequence)
            or isinstance(requirements, (str, bytes))
            or not 1 <= len(requirements) <= 5
            or not isinstance(constraints, Mapping)
        ):
            raise CatalogDataError("cart search needs one to five requirements")
        raw_page_size = payload.get("page_size")
        page_size = 20 if raw_page_size is None else int(raw_page_size)
        if not 1 <= page_size <= 100:
            raise CatalogDataError("cart page_size must be between 1 and 100")
        cursor = payload.get("cursor")
        offset = 0 if cursor is None else int(str(cursor))
        pools: list[tuple[CatalogListing, ...]] = []
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                raise CatalogDataError("cart requirement must be an object")
            queries = requirement.get("queries")
            if queries is None:
                queries = [str(requirement.get("query", ""))]
            by_ref: dict[str, CatalogListing] = {}
            for query in queries:
                query_text = str(query).strip()
                raw_filters = requirement.get("filters")
                if raw_filters is None:
                    filters = {"in_stock": True}
                elif isinstance(raw_filters, Mapping):
                    filters = dict(raw_filters)
                else:
                    raise CatalogDataError(
                        "cart requirement filters must be an object"
                    )
                count = self.database.search(
                    SearchRequest(
                        query=query_text,
                        filters=filters,
                        sort="price_asc",
                        page_size=1,
                    )
                ).total_hits
                if count > MAX_CART_REQUIREMENT_CANDIDATES:
                    raise CatalogDataError(
                        "cart requirement is too broad; refine its product query "
                        "or filters"
                    )
                page = self.database.full_candidates(
                    query=query_text,
                    filters=filters,
                    sort="price_asc",
                )
                by_ref.update({row.listing_ref: row for row in page})
                if len(by_ref) > MAX_CART_REQUIREMENT_CANDIDATES:
                    raise CatalogDataError(
                        "combined cart requirement is too broad; use fewer "
                        "alternatives or tighter filters"
                    )
            rows = tuple(
                sorted(by_ref.values(), key=lambda row: (row.price_minor, row.listing_ref))
            )
            if not rows:
                return {
                    "items": [],
                    "returned_through": 0,
                    "has_more": False,
                    "next_cursor": None,
                    "unseen_total_lower_bound_minor": None,
                    "applied_constraints": dict(constraints),
                }
            pools.append(rows)
        budget = constraints.get("budget_minor")
        min_merchants = int(constraints.get("min_merchants", 1))
        distinct = bool(constraints.get("distinct_listings", True))

        def valid(indices: tuple[int, ...]) -> bool:
            rows = tuple(pool[index] for pool, index in zip(pools, indices))
            total = sum(row.price_minor for row in rows)
            return (
                (budget is None or total <= int(budget))
                and (not distinct or len({row.listing_ref for row in rows}) == len(rows))
                and len({row.merchant_id for row in rows}) >= min_merchants
            )

        def total(indices: tuple[int, ...]) -> int:
            return sum(pool[index].price_minor for pool, index in zip(pools, indices))

        initial = tuple(0 for _ in pools)
        heap: list[tuple[int, tuple[str, ...], tuple[int, ...]]] = [
            (
                total(initial),
                tuple(pool[0].listing_ref for pool in pools),
                initial,
            )
        ]
        seen = {initial}
        valid_rows: list[tuple[int, tuple[CatalogListing, ...]]] = []
        target = offset + page_size + 1
        # Best-first enumeration is independent of the branch-and-bound oracle.
        while heap and len(valid_rows) < target:
            current_total, _refs, indices = heapq.heappop(heap)
            if budget is not None and current_total > int(budget):
                break
            if valid(indices):
                valid_rows.append(
                    (
                        current_total,
                        tuple(pool[index] for pool, index in zip(pools, indices)),
                    )
                )
            for position, pool in enumerate(pools):
                next_index = indices[position] + 1
                if next_index >= len(pool):
                    continue
                candidate = list(indices)
                candidate[position] = next_index
                candidate_tuple = tuple(candidate)
                if candidate_tuple in seen:
                    continue
                seen.add(candidate_tuple)
                candidate_rows = tuple(
                    item_pool[item_index]
                    for item_pool, item_index in zip(pools, candidate_tuple)
                )
                heapq.heappush(
                    heap,
                    (
                        sum(row.price_minor for row in candidate_rows),
                        tuple(row.listing_ref for row in candidate_rows),
                        candidate_tuple,
                    ),
                )
        visible = valid_rows[offset : offset + page_size]
        has_more = len(valid_rows) > offset + page_size or bool(heap)
        items = [
            {
                "cart_ref": "cart:" + "|".join(row.listing_ref for row in rows),
                "lines": [
                    {
                        "listing_ref": row.listing_ref,
                        "merchant": row.merchant_id,
                        "name": row.name,
                        "price_minor": row.price_minor,
                    }
                    for row in rows
                ],
                "listed_total_minor": row_total,
            }
            for row_total, rows in visible
        ]
        return {
            "items": items,
            "returned_through": offset + len(visible),
            "has_more": has_more,
            "next_cursor": str(offset + page_size) if has_more else None,
            "unseen_total_lower_bound_minor": heap[0][0] if heap else None,
            "applied_constraints": dict(constraints),
        }

    def _commit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        actor_id = str(payload.get("actor_id", ""))
        intent = str(payload.get("intent", ""))
        arguments = payload.get("arguments", {})
        if not actor_id or not isinstance(arguments, Mapping):
            raise LargeCatalogRuntimeError("World commit lacks actor or arguments")
        effect: dict[str, Any] = {
            "effect_ref": f"lc-effect-{len(self.effects) + 1:03d}",
            "actor_id": actor_id,
            "intent": intent,
            "arguments": dict(arguments),
        }
        if intent == "request_cart_quote":
            lines = arguments.get("lines")
            if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
                raise LargeCatalogRuntimeError("quote request needs lines")
            subtotal = 0
            normalized_lines: list[dict[str, Any]] = []
            for line in lines:
                if not isinstance(line, Mapping):
                    raise LargeCatalogRuntimeError("quote line must be an object")
                ref = str(line.get("listing_ref", ""))
                quantity = int(line.get("quantity", 1))
                listing = self.database.listing(ref)
                if listing is None or quantity <= 0:
                    raise LargeCatalogRuntimeError("quote line is invalid")
                line_total = listing.price_minor * quantity
                subtotal += line_total
                normalized_lines.append(
                    {
                        "listing_ref": ref,
                        "quantity": quantity,
                        "unit_price_minor": listing.price_minor,
                        "line_total_minor": line_total,
                    }
                )
            fee = int(self.task.oracle.get("fee_minor", 0))
            discount = int(self.task.oracle.get("discount_minor", 0))
            quote_ref = f"quote:{self.task.task_id}:{len(self.quotes) + 1}"
            quote = {
                "quote_ref": quote_ref,
                "lines": normalized_lines,
                "subtotal_minor": subtotal,
                "fee_minor": fee,
                "discount_minor": discount,
                "total_minor": subtotal + fee - discount,
            }
            self.quotes[quote_ref] = quote
            effect["quote"] = quote
        elif intent == "checkout_quote":
            quote_ref = str(arguments.get("quote_ref", ""))
            if quote_ref not in self.quotes:
                raise LargeCatalogRuntimeError("checkout references no current quote")
            effect["order"] = {
                "order_ref": f"order:{self.task.task_id}",
                "quote_ref": quote_ref,
                "total_minor": self.quotes[quote_ref]["total_minor"],
            }
        elif intent == "decline_quote":
            quote_ref = str(arguments.get("quote_ref", ""))
            if quote_ref not in self.quotes:
                raise LargeCatalogRuntimeError("decline references no current quote")
            effect["recorded"] = True
            effect["state_change"] = False
        elif intent in {
            "select_listing",
            "submit_cart",
            "publish_comparison_claim",
            "submit_quote",
        }:
            effect["recorded"] = True
        elif intent in {
            "reject_purchase",
            "submit_product_answer",
            "submit_comparison",
            "decline_comparison_claim",
        }:
            effect["recorded"] = True
            effect["state_change"] = False
        else:
            raise LargeCatalogRuntimeError(f"unsupported commit intent: {intent}")
        self.effects.append(effect)
        return effect


class LargeCatalogPlatform:
    """Shared intermediary that validates and forwards every request."""

    def __init__(self, world: LargeCatalogWorld, task: LargeCatalogTask) -> None:
        self.world = world
        self.task = task

    def handle(
        self,
        envelope: Envelope,
        *,
        world_sequence: int,
        reply_sequence: int,
    ) -> tuple[Envelope, Envelope]:
        validate(envelope)
        kind = ActionKind(envelope.action["kind"])
        payload = dict(envelope.action["payload"])
        if envelope.from_ != f"{self.task.role}:agent":
            raise LargeCatalogRuntimeError("Platform received a request from the wrong actor")
        if kind in {
            ActionKind.COMMERCE_LARGE_CATALOG_SEARCH,
            ActionKind.COMMERCE_LARGE_CATALOG_CART_SEARCH,
            ActionKind.COMMERCE_LARGE_CATALOG_OBSERVE,
        }:
            world_kind = {
                ActionKind.COMMERCE_LARGE_CATALOG_SEARCH: ActionKind.WORLD_LARGE_CATALOG_SEARCH,
                ActionKind.COMMERCE_LARGE_CATALOG_CART_SEARCH: ActionKind.WORLD_LARGE_CATALOG_CART_SEARCH,
                ActionKind.COMMERCE_LARGE_CATALOG_OBSERVE: ActionKind.WORLD_LARGE_CATALOG_OBSERVE,
            }[kind]
            world_request = _envelope(
                sequence=world_sequence,
                sender="platform:catalog",
                recipient="world",
                kind=world_kind,
                payload=payload,
                reply_to=envelope.msg_id,
            )
            world_response = self.world.handle(world_request, sequence=reply_sequence)
            result = dict(world_response.action["payload"])["result"]
            agent_response = _envelope(
                sequence=reply_sequence + 1,
                sender="platform:catalog",
                recipient=envelope.from_,
                kind=ActionKind.PLATFORM_LARGE_CATALOG_RESULT,
                payload={"request_intent": payload.get("intent"), "result": result},
                reply_to=envelope.msg_id,
            )
            return world_request, agent_response
        if kind is ActionKind.COMMERCE_LARGE_CATALOG_DECIDE:
            intent = str(payload.get("intent", ""))
            if intent not in self.task.allowed_intents or intent in {
                "search_catalog",
                "search_cart_candidates",
                "observe_listing",
            }:
                raise LargeCatalogRuntimeError("Platform rejects unavailable final intent")
            arguments = payload.get("arguments")
            if not isinstance(arguments, Mapping):
                raise LargeCatalogRuntimeError("decision arguments must be an object")
            self._validate_references(intent, arguments)
            world_request = _envelope(
                sequence=world_sequence,
                sender="platform:catalog",
                recipient="world",
                kind=ActionKind.WORLD_LARGE_CATALOG_COMMIT,
                payload={
                    "actor_id": envelope.from_,
                    "intent": intent,
                    "arguments": dict(arguments),
                },
                reply_to=envelope.msg_id,
            )
            world_response = self.world.handle(world_request, sequence=reply_sequence)
            agent_response = _envelope(
                sequence=reply_sequence + 1,
                sender="platform:catalog",
                recipient=envelope.from_,
                kind=ActionKind.PLATFORM_LARGE_CATALOG_DECISION,
                payload={
                    "accepted": True,
                    "intent": intent,
                    "world_result": dict(world_response.action["payload"])["result"],
                },
                reply_to=envelope.msg_id,
            )
            return world_request, agent_response
        raise LargeCatalogRuntimeError(f"Platform cannot handle {kind.value}")

    def _validate_references(self, intent: str, arguments: Mapping[str, Any]) -> None:
        refs: list[str] = []
        if "listing_ref" in arguments:
            refs.append(str(arguments["listing_ref"]))
        if isinstance(arguments.get("listing_refs"), Sequence):
            refs.extend(str(item) for item in arguments["listing_refs"])
        if isinstance(arguments.get("lines"), Sequence):
            for line in arguments["lines"]:
                if isinstance(line, Mapping) and "listing_ref" in line:
                    refs.append(str(line["listing_ref"]))
        for ref in refs:
            listing = self.world.database.listing(ref)
            if listing is None:
                raise LargeCatalogRuntimeError("Platform rejects an unknown listing")
            if intent in {"select_listing", "submit_cart", "request_cart_quote"} and not listing.in_stock:
                raise LargeCatalogRuntimeError("Platform rejects an unavailable listing")
        if intent == "checkout_quote":
            quote_ref = str(arguments.get("quote_ref", ""))
            quote = self.world.quotes.get(quote_ref)
            if quote is None:
                raise LargeCatalogRuntimeError("Platform rejects an unknown quote")
            budget = self.task.public_context.get("budget_minor")
            if type(budget) is int and int(quote["total_minor"]) > budget:
                raise LargeCatalogRuntimeError("Platform rejects an over-budget checkout")
        elif intent == "publish_comparison_claim":
            claim = arguments.get("claim")
            if not isinstance(claim, Mapping) or len(refs) != 2:
                raise LargeCatalogRuntimeError("Platform rejects a malformed comparison claim")
            left = self.world.database.listing(refs[0])
            right = self.world.database.listing(refs[1])
            if left is None or right is None or not _claim_is_supported(
                claim,
                left,
                right,
            ):
                raise LargeCatalogRuntimeError("Platform rejects an unsupported claim")
        elif intent == "submit_quote":
            lines = arguments.get("lines")
            if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
                raise LargeCatalogRuntimeError("Platform rejects a malformed quote")
            subtotal = 0
            for line in lines:
                if not isinstance(line, Mapping):
                    raise LargeCatalogRuntimeError("Platform rejects a malformed quote line")
                listing = self.world.database.listing(str(line.get("listing_ref", "")))
                quantity = int(line.get("quantity", 0))
                if listing is None or quantity <= 0:
                    raise LargeCatalogRuntimeError("Platform rejects an invalid quote line")
                expected_line = listing.price_minor * quantity
                if (
                    int(line.get("unit_price_minor", -1)) != listing.price_minor
                    or int(line.get("line_total_minor", -1)) != expected_line
                ):
                    raise LargeCatalogRuntimeError("Platform rejects incorrect line arithmetic")
                subtotal += expected_line
            discount = int(arguments.get("discount_minor", -1))
            fee = int(arguments.get("fee_minor", -1))
            total = int(arguments.get("total_minor", -1))
            if (
                int(arguments.get("subtotal_minor", -1)) != subtotal
                or total != subtotal - discount + fee
            ):
                raise LargeCatalogRuntimeError("Platform rejects incorrect quote arithmetic")


def _claim_is_supported(
    claim: Mapping[str, Any],
    left: CatalogListing,
    right: CatalogListing,
) -> bool:
    kind = str(claim.get("type", ""))
    if kind == "lower_price":
        subject = str(claim.get("subject_ref", ""))
        prices = {
            left.listing_ref: left.price_minor,
            right.listing_ref: right.price_minor,
        }
        return subject in prices and prices[subject] < prices[
            right.listing_ref if subject == left.listing_ref else left.listing_ref
        ]
    if kind == "both_in_stock":
        return left.in_stock and right.in_stock
    if kind == "price_difference_at_most":
        threshold = claim.get("threshold_minor")
        return type(threshold) is int and abs(left.price_minor - right.price_minor) <= threshold
    return False


class ModelPolicy:
    """Provider-neutral typed decision policy with no Skill instructions."""

    def __init__(self, channel: InferenceChannel) -> None:
        self.channel = channel
        self.calls = 0

    def decide(
        self,
        task: LargeCatalogTask,
        observations: Sequence[Mapping[str, Any]],
        allowed_intents: Sequence[str],
        *,
        decision_id: str,
    ) -> Mapping[str, Any]:
        request = {
            "role": task.role,
            "user_request": task.prompt,
            "public_context": dict(task.public_context),
            "observations": list(observations),
            "allowed_intents": {
                intent: _intent_schema(intent) for intent in allowed_intents
            },
            "required_response": {
                "intent": "one allowed intent",
                "arguments": "an object matching that intent",
            },
        }
        system = (
            "You are the assigned commerce agent. Work through the available "
            "catalog observations and then choose one business action. Return "
            "only one strict JSON object with keys intent and arguments. Do not "
            "write VCP routing fields. No external knowledge is authoritative."
        )
        response = self.channel.complete_business_decision(
            system_prompt=system,
            user_prompt=json.dumps(request, ensure_ascii=False, sort_keys=True),
            decision_id=decision_id,
        )
        self.calls += 1
        parsed = _parse_decision_object(response.content)
        if not isinstance(parsed, Mapping):
            raise LargeCatalogRuntimeError("model decision must be a JSON object")
        if not isinstance(parsed.get("intent"), str):
            raise LargeCatalogRuntimeError("model decision lacks a string intent")
        if not isinstance(parsed.get("arguments"), Mapping):
            raise LargeCatalogRuntimeError("model decision arguments must be an object")
        intent = str(parsed["intent"])
        arguments = _canonicalize_arguments(intent, parsed["arguments"])
        return {"intent": intent, "arguments": arguments}


def _parse_decision_object(content: str) -> Mapping[str, Any]:
    """Read one JSON decision while tolerating harmless textual wrapping."""

    stripped = content.strip()
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, Mapping):
        return direct

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    end_positions: set[tuple[int, int]] = set()
    for start, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, relative_end = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        end = start + relative_end
        if (
            isinstance(value, Mapping)
            and "intent" in value
            and "arguments" in value
            and (start, end) not in end_positions
        ):
            candidates.append((start, end, value))
            end_positions.add((start, end))
    objects = [
        value
        for start, end, value in candidates
        if not any(
            outer_start < start and end <= outer_end
            for outer_start, outer_end, _outer_value in candidates
        )
    ]
    if len(objects) == 1:
        return objects[0]
    if not objects:
        raise LargeCatalogRuntimeError("model response contains no valid JSON object")
    raise LargeCatalogRuntimeError("model response contains multiple JSON objects")


def _canonicalize_arguments(
    intent: str,
    raw_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize an unambiguous display-name alias at the model boundary."""

    arguments = dict(raw_arguments)
    if intent != "submit_quote":
        return arguments
    raw_lines = arguments.get("lines")
    if not isinstance(raw_lines, Sequence) or isinstance(raw_lines, (str, bytes)):
        return arguments
    lines: list[Any] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, Mapping):
            lines.append(raw_line)
            continue
        line = dict(raw_line)
        line_total = line.get("line_total_minor")
        line_amount = line.get("line_amount_minor")
        if line_total is None and line_amount is not None:
            line["line_total_minor"] = line.pop("line_amount_minor")
        elif line_total is not None and line_amount is not None:
            if line_total != line_amount:
                raise LargeCatalogRuntimeError(
                    "quote line gives conflicting total and amount values"
                )
            line.pop("line_amount_minor")
        lines.append(line)
    arguments["lines"] = lines
    return arguments


def _intent_schema(intent: str) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {
        "search_catalog": {
            "query": "string",
            "filters": "object",
            "sort": "price_asc | price_desc | name_asc | merchant_asc",
            "cursor": "string or null",
            "page_size": "integer 1..100",
        },
        "search_cart_candidates": {
            "requirements": (
                "list of specific query/filter objects; refine any requirement "
                "matching more than 1000 listings"
            ),
            "constraints": "budget/min_merchants/distinct_listings object",
            "cursor": "string or null",
            "page_size": "integer 1..100",
        },
        "observe_listing": {"listing_ref": "string"},
        "select_listing": {"listing_ref": "string", "evidence_refs": "list[string]"},
        "reject_purchase": {"reason_code": "string", "evidence_refs": "list[string]"},
        "submit_product_answer": {
            "listing_ref": "string",
            "fields": "object",
            "evidence_refs": "list[string]",
        },
        "submit_comparison": {
            "listing_refs": "list[string]",
            "results": "object",
            "evidence_refs": "list[string]",
        },
        "submit_cart": {"lines": "list[{listing_ref, quantity}]", "evidence_refs": "list[string]"},
        "request_cart_quote": {
            "lines": "list[{listing_ref, quantity}]",
            "evidence_refs": "list[string]",
        },
        "checkout_quote": {"quote_ref": "string"},
        "decline_quote": {"quote_ref": "string", "reason_code": "string"},
        "publish_comparison_claim": {
            "listing_refs": "list[string]",
            "claim": "object",
            "evidence_refs": "list[string]",
        },
        "decline_comparison_claim": {
            "listing_refs": "list[string]",
            "reason_code": "string",
            "evidence_refs": "list[string]",
        },
        "submit_quote": {
            "lines": (
                "list[{listing_ref: string, quantity: integer, "
                "unit_price_minor: integer, line_total_minor: integer}]"
            ),
            "subtotal_minor": "integer",
            "discount_minor": "integer",
            "fee_minor": "integer",
            "total_minor": "integer",
            "evidence_refs": "list[string]",
        },
    }
    return schemas[intent]


def run_episode(
    *,
    task: LargeCatalogTask,
    database: CatalogDatabase,
    policy: DecisionPolicy,
    max_steps: int = 20,
) -> dict[str, Any]:
    """Run one model or deterministic policy through Agent, VCP, Platform, World."""

    world = LargeCatalogWorld(database, task)
    platform = LargeCatalogPlatform(world, task)
    observations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    sequence = 0
    final_intent: str | None = None
    final_arguments: dict[str, Any] | None = None
    terminal = "step_limit"
    started = time.monotonic()
    for step in range(max_steps):
        decision = policy.decide(
            task,
            observations,
            task.allowed_intents,
            decision_id=f"{task.task_id}:decision:{step + 1}",
        )
        intent = str(decision.get("intent", ""))
        arguments = decision.get("arguments")
        if intent not in task.allowed_intents or not isinstance(arguments, Mapping):
            terminal = "model_protocol_error"
            trace.append(
                {
                    "event_ref": f"trace-{len(trace) + 1:03d}",
                    "stage": "decision",
                    "intent": intent,
                    "accepted": False,
                    "error": "unavailable intent or invalid arguments",
                }
            )
            break
        sequence += 1
        action_kind = _INTENT_ACTION.get(
            intent, ActionKind.COMMERCE_LARGE_CATALOG_DECIDE
        )
        payload = {"intent": intent, **dict(arguments)}
        if action_kind is ActionKind.COMMERCE_LARGE_CATALOG_DECIDE:
            payload = {"intent": intent, "arguments": dict(arguments)}
        actor_request = _envelope(
            sequence=sequence,
            sender=f"{task.role}:agent",
            recipient="platform:catalog",
            kind=action_kind,
            payload=payload,
        )
        trace.append(
            {
                "event_ref": f"trace-{len(trace) + 1:03d}",
                "stage": "decision",
                "intent": intent,
                "arguments": dict(arguments),
                "envelope": actor_request.action,
            }
        )
        try:
            world_request, response = platform.handle(
                actor_request,
                world_sequence=sequence + 1,
                reply_sequence=sequence + 2,
            )
        except (
            CatalogDataError,
            LargeCatalogRuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            terminal = "platform_rejection"
            trace.append(
                {
                    "event_ref": f"trace-{len(trace) + 1:03d}",
                    "stage": "validation",
                    "intent": intent,
                    "accepted": False,
                    "error": type(exc).__name__,
                }
            )
            final_intent = intent
            final_arguments = dict(arguments)
            break
        sequence += 3
        trace.append(
            {
                "event_ref": f"trace-{len(trace) + 1:03d}",
                "stage": "validation",
                "intent": intent,
                "accepted": True,
                "world_action": world_request.action,
            }
        )
        result = dict(response.action["payload"])
        observations.append(
            {
                "observation_ref": f"observation-{len(observations) + 1:03d}",
                "intent": intent,
                "arguments": dict(arguments),
                "result": result,
            }
        )
        trace.append(
            {
                "event_ref": f"trace-{len(trace) + 1:03d}",
                "stage": "world_effect" if intent not in _INTENT_ACTION else "evidence",
                "intent": intent,
                "result": result,
            }
        )
        if intent not in _INTENT_ACTION and intent != "request_cart_quote":
            final_intent = intent
            final_arguments = dict(arguments)
            terminal = "completed"
            break
    return {
        "task_id": task.task_id,
        "role": task.role,
        "terminal": terminal,
        "final_intent": final_intent,
        "final_arguments": final_arguments,
        "observations": observations,
        "effects": list(world.effects),
        "trace": trace,
        "model_calls": int(getattr(policy, "calls", 0)),
        "latency_seconds": time.monotonic() - started,
    }
