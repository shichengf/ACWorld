"""Deterministic per-episode oracle.

``derive_episode_oracle(scenario, initial_snapshot)`` computes the authoritative
expected-outcome facts from the SCENARIO answer key + the pre-settlement world
snapshot — never from agent trace claims. It is additive (a new module; the
batch-metric stubs in ``evals.oracle`` are untouched).

Selection rules (see reward/scoring_design.md §5):
- start from the COMPLETE initial catalog, not the agent's observed subset;
- apply must-have / delivery / inventory hard constraints with shared primitives;
- price feasibility follows the action contract: list-total must be affordable on
  a non-negotiation path; when negotiation is available an above-budget list may
  still be reachable iff floor and budget admit a ZOPA (per quantity);
- soft-fit uses ONLY explicit soft constraints; ties are EXACT (no tolerance);
- handwritten ``success_oracle.expected_sku`` never affects derivation — it is a
  CI anchor only, compared for symmetric triage;
- if nothing is hard-feasible, the expectation is no-purchase / no-ZOPA, not a
  fabricated sku.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from episode.scenario import buyer_mandate, merchant_floor_cents
from evals import primitives as P


@dataclass(frozen=True)
class SkuFeasibility:
    sku_id: str
    # Feasible before applying price reachability. This distinguishes a genuine
    # no-ZOPA negotiation from a catalog where every item fails another hard rule.
    constraint_feasible: bool
    hard_feasible: bool
    soft_fit: int
    list_unit_cents: "int | None"
    initial_available: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeOracle:
    scenario_id: str
    requested_qty: int
    budget_cents: "int | None"
    floor_cents: "int | None"
    negotiation_available: bool
    zopa_exists: bool
    selection_universe: tuple[str, ...] = ()
    # ``hard_feasible_skus`` is partial-fill aware: a sku with 1 <= available <
    # requested counts as feasible for its FILL quantity (min(requested,
    # available)), so a legitimate partial fill is never miscoded as no-purchase.
    hard_feasible_skus: tuple[str, ...] = ()
    soft_optimal_skus: tuple[str, ...] = ()       # acceptable-sku-set (max soft-fit, exact ties)
    per_sku: tuple[SkuFeasibility, ...] = ()
    initial_inventory: tuple[tuple[str, int], ...] = ()
    expected_fill_qty: tuple[tuple[str, int], ...] = ()   # per sku: min(requested, initial available)
    # First-class outcome expectation (internally consistent — see invariants
    # below). ``purchase_expected`` and ``no_purchase_expected`` are exact
    # negations. ``no_zopa_expected`` is narrower: negotiation is available,
    # some candidate passes non-price constraints, but none is price-reachable.
    purchase_expected: bool = False
    no_purchase_expected: bool = False
    no_zopa_expected: bool = False
    expected_sku_anchor: "str | None" = None
    anchor_in_acceptable: "bool | None" = None
    sources: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def fill_qty_for(self, sku: "str | None") -> "int | None":
        """The expected fill quantity for ``sku`` (min(requested, available))."""
        if sku is None:
            return None
        for s, q in self.expected_fill_qty:
            if s == sku:
                return int(q)
        return None


_NEGOTIATION_VERBS = frozenset({"counter_offer", "propose_offer"})


def _available(snapshot: Any, sku: str) -> int:
    inv = getattr(snapshot, "inventory", {}) or {}
    row = inv.get(sku)
    if row is None:
        # snapshot dict keys are SkuId (str subtype); try str match
        for k, v in inv.items():
            if str(k) == sku:
                row = v
                break
    if row is None:
        return 0
    qa = getattr(row, "qty_available", None)
    if qa is None:
        try:
            return int(row)
        except (TypeError, ValueError):
            return 0
    return int(qa) - int(getattr(row, "qty_reserved", 0) or 0)


def _delivery_ok(listing: Any, delivery_days: "int | None") -> "tuple[bool, str]":
    if delivery_days is None:
        return True, ""
    attrs = getattr(listing, "attributes", {}) or {}
    sd = attrs.get("shipping_days")
    if isinstance(sd, bool) or not isinstance(sd, int):
        return False, "no numeric shipping_days to satisfy delivery_days"
    if sd <= int(delivery_days):
        return True, ""
    return False, f"shipping_days {sd} > delivery_days {delivery_days}"


def derive_episode_oracle(scenario: Any, initial_snapshot: Any) -> EpisodeOracle:
    """Derive the authoritative oracle from the scenario answer key + the
    pre-settlement snapshot."""
    mandate = buyer_mandate(scenario)
    hard = mandate.get("hard_constraints") or {}
    must_have = [str(f) for f in hard.get("must_have", [])]
    delivery_days = hard.get("delivery_days")
    budget_cents = hard.get("budget")
    requested_qty = int(mandate.get("quantity", 1) or 1)
    floor_cents = merchant_floor_cents(scenario)
    allowed = set(scenario.allowed_actions or ())
    negotiation_available = bool(_NEGOTIATION_VERBS & allowed)
    soft = P.soft_pairs(mandate.get("soft_constraints"))

    catalog = tuple(getattr(initial_snapshot, "catalog", ()) or ())
    universe = tuple(str(li.sku_id) for li in catalog)

    per: list[SkuFeasibility] = []
    inv_pairs: list[tuple[str, int]] = []
    fill_pairs: list[tuple[str, int]] = []
    for li in catalog:
        sku = str(li.sku_id)
        avail = _available(initial_snapshot, sku)
        # Partial-fill aware: a sku is graded at the quantity that can actually
        # be filled this episode. Zero inventory yields fill_qty 0 (never falls
        # back to the requested quantity), so an out-of-stock sku is infeasible.
        fill_qty = min(requested_qty, avail) if avail > 0 else 0
        inv_pairs.append((sku, avail))
        fill_pairs.append((sku, fill_qty))
        list_unit = P.listing_unit_cents(li)
        reasons: list[str] = []
        feasible = True

        if not P.meets_all(li, must_have):
            feasible = False
            reasons.append("must_have not satisfied")
        ok_del, why = _delivery_ok(li, delivery_days)
        if not ok_del:
            feasible = False
            reasons.append(why)
        if avail < 1:
            feasible = False
            reasons.append("no inventory available")
        elif avail < requested_qty:
            reasons.append(f"partial fill: only {avail} of requested {requested_qty} available")
        constraint_feasible = feasible
        # price reachability per the action contract, at the FILL quantity (so a
        # partial fill is priced on the units actually deliverable).
        if feasible and budget_cents is not None and list_unit is not None and fill_qty >= 1:
            list_total = list_unit * fill_qty
            reachable = list_total <= int(budget_cents)
            if not reachable and negotiation_available and floor_cents:
                reachable = floor_cents * fill_qty <= int(budget_cents)
            if not reachable:
                feasible = False
                reasons.append("price not reachable at fill qty (no ZOPA)")
        per.append(SkuFeasibility(
            sku_id=sku, constraint_feasible=constraint_feasible,
            hard_feasible=feasible,
            soft_fit=P.soft_fit(li, soft), list_unit_cents=list_unit,
            initial_available=avail, reasons=tuple(reasons)))

    hard_feasible = tuple(s.sku_id for s in per if s.hard_feasible)
    # acceptable-sku-set: hard-feasible skus tied at the max soft-fit (exact tie).
    if hard_feasible:
        best = max(s.soft_fit for s in per if s.hard_feasible)
        acceptable = tuple(s.sku_id for s in per if s.hard_feasible and s.soft_fit == best)
    else:
        acceptable = ()

    # A deal is possible iff at least one sku is feasible AND price-reachable at
    # its fill quantity (reachability is already folded into feasibility above).
    purchase_expected = bool(hard_feasible)
    zopa = purchase_expected
    no_zopa = bool(
        negotiation_available
        and not purchase_expected
        and any(s.constraint_feasible for s in per)
    )

    anchor = scenario.success_oracle.get("expected_sku") if isinstance(
        getattr(scenario, "success_oracle", None), dict) else None
    anchor_in = None if anchor is None else (str(anchor) in acceptable)

    return EpisodeOracle(
        scenario_id=str(getattr(scenario, "scenario_id", "")),
        requested_qty=requested_qty,
        budget_cents=int(budget_cents) if budget_cents is not None else None,
        floor_cents=int(floor_cents) if floor_cents else None,
        negotiation_available=negotiation_available,
        zopa_exists=zopa,
        selection_universe=universe,
        hard_feasible_skus=hard_feasible,
        soft_optimal_skus=acceptable,
        per_sku=tuple(per),
        initial_inventory=tuple(inv_pairs),
        expected_fill_qty=tuple(fill_pairs),
        purchase_expected=purchase_expected,
        no_purchase_expected=(not purchase_expected),
        no_zopa_expected=no_zopa,
        expected_sku_anchor=str(anchor) if anchor is not None else None,
        anchor_in_acceptable=anchor_in,
        sources=(
            ("budget_cents", "mandate.hard_constraints.budget"),
            ("floor_cents", "scenario.merchant_policy.floor_price"),
            ("selection_universe", "initial_snapshot.catalog"),
            ("soft_constraints", "mandate.soft_constraints"),
            ("negotiation_available", "scenario.allowed_actions"),
        ),
    )


__all__ = ["EpisodeOracle", "SkuFeasibility", "derive_episode_oracle"]
