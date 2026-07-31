"""Deterministic, World-authoritative oracle for S18 multi-merchant choice.

S18 compares merchant-specific listings for one shared ``product_id``.  The
headline result never trusts a model-authored rationale or a merchant's
free-text claim.  Feasibility and the winning set are rebuilt from the initial
World snapshot; evidence verification additionally requires the buyer's commit
decision to contain exact framework-executed catalog reads for the complete
comparison set and an audited aggregator result that exposed that set.

The fixed v1 ordering is deliberately simple and inspectable: among listings
that satisfy inventory, budget, must-have, ETA, minimum merchant reputation,
and minimum return-window constraints, minimize total list price, then ETA,
then maximize reputation and return window.  Scenario seeds make the first
criterion unique, so later criteria are deterministic tie-breakers rather than
hidden utility weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from episode.scenario import buyer_mandate
from evals import primitives as P


@dataclass(frozen=True)
class MultiMerchantCandidate:
    """One comparable listing and its authoritative feasibility facts."""

    sku_id: str
    merchant_id: str
    product_id: str
    list_unit_cents: int | None
    available: int
    shipping_days: int | None
    merchant_reputation_bps: int | None
    return_window_days: int | None
    feasible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiMerchantOracle:
    """World-derived S18 comparison result."""

    applicable: bool
    valid: bool
    discriminating: bool
    candidate_skus: tuple[str, ...] = ()
    feasible_skus: tuple[str, ...] = ()
    optimal_skus: tuple[str, ...] = ()
    candidates: tuple[MultiMerchantCandidate, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class MultiMerchantEvidenceResult:
    """Result of checking the selected listing and its exact comparison trail."""

    applicable: bool
    ok: bool
    discriminating: bool
    reason: str
    evidence_ids: tuple[str, ...] = ()


def _variant_id(scenario: Any) -> str:
    benchmark = getattr(scenario, "benchmark", None)
    return str(getattr(benchmark, "variant_id", ""))


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _available(snapshot: Any, sku_id: str) -> int:
    for key, row in (getattr(snapshot, "inventory", {}) or {}).items():
        if str(key) != sku_id:
            continue
        available = getattr(row, "qty_available", row)
        reserved = getattr(row, "qty_reserved", 0)
        try:
            return int(available) - int(reserved or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def derive_multi_merchant_oracle(
    scenario: Any, initial_snapshot: Any
) -> MultiMerchantOracle:
    """Derive S18's feasible and globally optimal sets from World state only."""

    if _variant_id(scenario) != "S18":
        return MultiMerchantOracle(
            applicable=False,
            valid=False,
            discriminating=False,
            reason="not the S18 multi-merchant benchmark lane",
        )

    catalog = tuple(getattr(initial_snapshot, "catalog", ()) or ())
    by_product: dict[str, list[Any]] = {}
    for listing in catalog:
        product_id = str(getattr(listing, "product_id", "") or "")
        if product_id:
            by_product.setdefault(product_id, []).append(listing)

    groups = [
        rows
        for rows in by_product.values()
        if len({str(item.merchant_id) for item in rows}) >= 2
    ]
    if len(groups) != 1:
        return MultiMerchantOracle(
            applicable=True,
            valid=False,
            discriminating=False,
            reason=(
                "S18 requires exactly one product_id represented by at least two merchants"
            ),
        )

    listings = sorted(groups[0], key=lambda item: str(item.sku_id))
    mandate = buyer_mandate(scenario)
    hard = mandate.get("hard_constraints") or {}
    budget = _strict_int(hard.get("budget"))
    requested_qty = _strict_int(mandate.get("quantity")) or 1
    delivery_limit = _strict_int(hard.get("delivery_days"))
    min_reputation = _strict_int(hard.get("min_merchant_reputation_bps"))
    min_return = _strict_int(hard.get("min_return_window_days"))
    must_have = tuple(str(item) for item in hard.get("must_have", ()) or ())

    candidates: list[MultiMerchantCandidate] = []
    rank_by_sku: dict[str, tuple[int, int, int, int]] = {}
    for listing in listings:
        sku_id = str(listing.sku_id)
        attrs = getattr(listing, "attributes", {}) or {}
        unit_cents = P.listing_unit_cents(listing)
        available = _available(initial_snapshot, sku_id)
        shipping_days = _strict_int(attrs.get("shipping_days"))
        reputation = _strict_int(attrs.get("merchant_reputation_bps"))
        return_days = _strict_int(attrs.get("return_window_days"))
        reasons: list[str] = []

        if not P.meets_all(listing, must_have):
            reasons.append("must-have constraints are not satisfied")
        if available < requested_qty:
            reasons.append("insufficient inventory for requested quantity")
        if unit_cents is None:
            reasons.append("listing price is not machine-readable")
        elif budget is not None and unit_cents * requested_qty > budget:
            reasons.append("list-price total exceeds buyer budget")
        if delivery_limit is not None and (
            shipping_days is None or shipping_days > delivery_limit
        ):
            reasons.append("shipping ETA exceeds the hard limit")
        if min_reputation is not None and (
            reputation is None or reputation < min_reputation
        ):
            reasons.append("merchant reputation is below the hard minimum")
        if min_return is not None and (
            return_days is None or return_days < min_return
        ):
            reasons.append("return window is below the hard minimum")

        feasible = not reasons
        candidate = MultiMerchantCandidate(
            sku_id=sku_id,
            merchant_id=str(listing.merchant_id),
            product_id=str(listing.product_id),
            list_unit_cents=unit_cents,
            available=available,
            shipping_days=shipping_days,
            merchant_reputation_bps=reputation,
            return_window_days=return_days,
            feasible=feasible,
            reasons=tuple(reasons),
        )
        candidates.append(candidate)
        if feasible:
            # Missing fields cannot reach this branch when the corresponding
            # threshold is configured by S18.  The sentinels keep this helper
            # total for additive future variants.
            rank_by_sku[sku_id] = (
                int(unit_cents or 0) * requested_qty,
                shipping_days if shipping_days is not None else 10**9,
                -(reputation if reputation is not None else -1),
                -(return_days if return_days is not None else -1),
            )

    feasible_skus = tuple(item.sku_id for item in candidates if item.feasible)
    if not feasible_skus:
        return MultiMerchantOracle(
            applicable=True,
            valid=False,
            discriminating=False,
            candidate_skus=tuple(item.sku_id for item in candidates),
            candidates=tuple(candidates),
            reason="no listing is feasible under the declared S18 comparison constraints",
        )

    best_rank = min(rank_by_sku[sku] for sku in feasible_skus)
    optimal = tuple(sku for sku in feasible_skus if rank_by_sku[sku] == best_rank)
    discriminating = len(optimal) < len(feasible_skus)
    return MultiMerchantOracle(
        applicable=True,
        valid=True,
        discriminating=discriminating,
        candidate_skus=tuple(item.sku_id for item in candidates),
        feasible_skus=feasible_skus,
        optimal_skus=optimal,
        candidates=tuple(candidates),
        reason=(
            "one or more feasible listings are strictly dominated by the fixed S18 ordering"
            if discriminating
            else "all feasible listings tie under the fixed S18 ordering"
        ),
    )


def _step_data(step: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(step, dict):
        return "", {}
    if "kind" in step:
        data = step.get("data")
        return str(step.get("kind", "")), data if isinstance(data, dict) else {}
    return str(step.get("step", "")), step


def _money_amount(value: Any) -> Decimal | None:
    if isinstance(value, dict):
        value = value.get("amount")
    else:
        value = getattr(value, "amount", value)
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _projected_attributes(listing: Any) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in (getattr(listing, "attributes", {}) or {}).items()
        if str(key).casefold().replace(" ", "_") != "key_features"
    }


def _listing_result_matches(result: Any, listing: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return (
        str(result.get("sku_id")) == str(listing.sku_id)
        and str(result.get("merchant_id")) == str(listing.merchant_id)
        and _money_amount(result.get("list_price")) == listing.list_price.amount
        and result.get("attributes") == _projected_attributes(listing)
    )


def _payload_listing(payload: Any, sku_id: str) -> Any:
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if isinstance(row, dict) and str(row.get("sku_id")) == sku_id:
            return row
    return None


def _settle_envelope(audit_index: Any, order: Any) -> Any:
    for env in audit_index.settle_envelopes():
        payload = env.action.get("payload") if isinstance(env.action, dict) else None
        if isinstance(payload, dict) and str(payload.get("order_id")) == str(order.order_id):
            return env
    return None


def verify_multi_merchant_comparison(
    *, scenario: Any, initial_snapshot: Any, audit_index: Any, settled_order: Any
) -> MultiMerchantEvidenceResult:
    """Verify S18 choice plus exact decision-bound global comparison evidence."""

    oracle = derive_multi_merchant_oracle(scenario, initial_snapshot)
    if not oracle.applicable:
        return MultiMerchantEvidenceResult(
            False, False, False, oracle.reason
        )
    if not oracle.valid:
        return MultiMerchantEvidenceResult(
            True, False, oracle.discriminating, oracle.reason
        )
    if settled_order is None:
        return MultiMerchantEvidenceResult(
            True, False, oracle.discriminating, "S18 has no authoritative settled order"
        )

    settled_sku = str(settled_order.sku_id)
    if settled_sku not in oracle.optimal_skus:
        return MultiMerchantEvidenceResult(
            True,
            False,
            oracle.discriminating,
            "settled listing is not globally optimal among World-feasible merchants",
        )

    listings = {
        str(item.sku_id): item
        for item in (getattr(initial_snapshot, "catalog", ()) or ())
        if str(item.sku_id) in oracle.candidate_skus
    }
    candidate_set = set(oracle.candidate_skus)
    buyer_id = str(settled_order.buyer_id)
    settle = _settle_envelope(audit_index, settled_order)
    if settle is None:
        return MultiMerchantEvidenceResult(
            True, False, oracle.discriminating, "settlement has no matching audited request"
        )
    settle_position = audit_index.position(str(settle.msg_id))

    # Find the actual accept decision that contains the complete comparison.
    record = None
    read_slots: dict[str, tuple[dict[str, Any], str, str | None]] = {}
    for trace in audit_index.trace:
        if str(trace.get("agent_id")) != buyer_id:
            continue
        if trace.get("forced_flush") or trace.get("incomplete"):
            continue
        chosen = trace.get("chosen") or {}
        if chosen.get("decision") != "accept" or not str(
            chosen.get("offer_id", "")
        ).endswith(settled_sku):
            continue
        considered = {
            str(item.get("sku_id"))
            for item in (trace.get("considered") or ())
            if isinstance(item, dict)
        }
        if candidate_set - considered:
            continue

        slots: dict[str, tuple[dict[str, Any], str, str | None]] = {}
        decision_id = str(trace.get("decision_id") or "")
        for step_index, step in enumerate(trace.get("steps") or ()):
            kind, data = _step_data(step)
            if kind != "tool_call":
                continue
            for result_index, result in enumerate(data.get("results") or ()):
                if not isinstance(result, dict) or result.get("tool") not in {
                    "world.get_listing",
                    "world.search_catalog",
                }:
                    continue
                raw = result.get("result")
                rows = raw if isinstance(raw, list) else [raw]
                for item_index, item in enumerate(rows):
                    if not isinstance(item, dict):
                        continue
                    sku_id = str(item.get("sku_id", ""))
                    if sku_id in candidate_set:
                        evidence_id = (
                            f"{decision_id}:grounding:{step_index}:"
                            f"{result_index}:{item_index}"
                        )
                        source = result.get("source_msg_id")
                        slots[sku_id] = (
                            item,
                            evidence_id,
                            str(source) if source else None,
                        )
        if candidate_set <= set(slots):
            record = trace
            read_slots = slots
            break

    if record is None:
        return MultiMerchantEvidenceResult(
            True,
            False,
            oracle.discriminating,
            "buyer commit lacks one decision-bound framework catalog read per merchant listing",
        )

    emitted_id = str(record.get("emitted_msg_id") or "")
    commit = audit_index.by_msg_id.get(emitted_id)
    commit_position = audit_index.position(emitted_id)
    if (
        commit is None
        or str(commit.from_) != buyer_id
        or str(commit.to) != "platform:aggregator"
        or str(commit.action.get("kind", "")) != "commerce.accept_offer"
        or commit_position < 0
        or commit_position >= settle_position
    ):
        return MultiMerchantEvidenceResult(
            True, False, oracle.discriminating, "comparison decision is not the causal buyer accept"
        )

    considered_by_sku = {
        str(item.get("sku_id")): item
        for item in (record.get("considered") or ())
        if isinstance(item, dict)
    }
    evidence_ids: list[str] = [emitted_id, str(settle.msg_id)]
    for sku_id in oracle.candidate_skus:
        listing = listings[sku_id]
        result, evidence_id, source_id = read_slots[sku_id]
        considered = considered_by_sku[sku_id]
        if not _listing_result_matches(result, listing):
            return MultiMerchantEvidenceResult(
                True,
                False,
                oracle.discriminating,
                f"framework catalog result for {sku_id} differs from initial World state",
            )
        if (
            considered.get("grounded_evidence_id") != evidence_id
            or considered.get("grounded_attributes") != result.get("attributes")
        ):
            return MultiMerchantEvidenceResult(
                True,
                False,
                oracle.discriminating,
                f"candidate {sku_id} is not bound to its exact decision-local catalog result",
            )
        evidence_ids.append(evidence_id)

        # HTTP/re-entrant reads have a source envelope; synchronous reads do not.
        # When present, fail closed unless it is the buyer's exact World response
        # and causally precedes both accept and settlement.
        if source_id is not None:
            response = audit_index.by_msg_id.get(source_id)
            request = (
                audit_index.by_msg_id.get(str(response.in_reply_to))
                if response is not None and response.in_reply_to
                else None
            )
            response_payload = (
                response.action.get("payload")
                if response is not None and isinstance(response.action, dict)
                else None
            )
            source_listing = _payload_listing(response_payload, sku_id)
            request_payload = (
                request.action.get("payload")
                if request is not None and isinstance(request.action, dict)
                else None
            )
            if (
                response is None
                or request is None
                or str(response.action.get("kind", "")) != "world.response"
                or str(response.from_) != "world"
                or str(response.to) != buyer_id
                or str(request.from_) != buyer_id
                or str(request.action.get("kind", "")) != "world.read_catalog"
                or not isinstance(request_payload, dict)
                or str(request_payload.get("sku_id")) != sku_id
                or not _listing_result_matches(source_listing, listing)
                or audit_index.position(source_id) < 0
                or audit_index.position(source_id) >= commit_position
                or audit_index.position(source_id) >= settle_position
            ):
                return MultiMerchantEvidenceResult(
                    True,
                    False,
                    oracle.discriminating,
                    f"catalog evidence for {sku_id} does not match its audited World response",
                )
            evidence_ids.append(source_id)

    # The platform search result is the authoritative candidate-universe edge:
    # require all and only the comparable World listings, with exact owner,
    # public list price, and ETA.  Marketing prose is never consulted.
    ranked = None
    for env in audit_index.by_kind.get("platform.rank_offers", ()):
        if str(env.to) != buyer_id or audit_index.position(str(env.msg_id)) >= commit_position:
            continue
        payload = env.action.get("payload") if isinstance(env.action, dict) else None
        rows = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue
        if {str(item.get("sku_id")) for item in rows if isinstance(item, dict)} == candidate_set:
            ranked = env
            break
    if ranked is None:
        return MultiMerchantEvidenceResult(
            True,
            False,
            oracle.discriminating,
            "audited platform search did not expose the complete comparable merchant set",
        )

    ranked_rows = {
        str(item.get("sku_id")): item
        for item in ranked.action["payload"]["candidates"]
        if isinstance(item, dict)
    }
    by_sku = {item.sku_id: item for item in oracle.candidates}
    for sku_id in oracle.candidate_skus:
        expected = by_sku[sku_id]
        row = ranked_rows[sku_id]
        eta = row.get("fulfillment", {}).get("eta_days") if isinstance(
            row.get("fulfillment"), dict
        ) else None
        if (
            str(row.get("merchant_id")) != expected.merchant_id
            or _strict_int(row.get("unit_price")) != expected.list_unit_cents
            or _strict_int(eta) != expected.shipping_days
        ):
            return MultiMerchantEvidenceResult(
                True,
                False,
                oracle.discriminating,
                f"aggregator candidate {sku_id} differs from authoritative World state",
            )
    evidence_ids.append(str(ranked.msg_id))

    return MultiMerchantEvidenceResult(
        True,
        True,
        oracle.discriminating,
        "globally optimal merchant listing selected from complete exact World evidence",
        tuple(evidence_ids),
    )


__all__ = [
    "MultiMerchantCandidate",
    "MultiMerchantOracle",
    "MultiMerchantEvidenceResult",
    "derive_multi_merchant_oracle",
    "verify_multi_merchant_comparison",
]
