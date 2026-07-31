"""Deterministic market-level metrics for many-to-many commerce episodes.

This module deliberately has no dependency on an agent, prompt, or LLM judge.
It evaluates normalized, structured records and therefore can be used by both
in-process and HTTP benchmark runners.

Money is represented as integer *minor currency units* throughout (for example,
cents).  Every monetary input in one call must use the same currency and scale.
Ratios are returned as :class:`~decimal.Decimal` values so repeated evaluation is
bit-for-bit deterministic.

Metric definitions are intentionally mechanical:

* trade rate is distinct transacting buyers divided by the buyer population;
* consumer and producer surplus are ``sum(q * (value - price))`` and
  ``sum(q * (price - floor))``; social welfare is their sum;
* allocative efficiency is non-negative realized welfare divided by exact or
  supplied optimal welfare (signed realized welfare remains separately visible);
* inventory correctness is the pass rate across aggregate capacity checks and
  any supplied transaction-local decrement checks;
* exposure fairness is Jain's index over eligible merchants, including eligible
  merchants with zero exposure;
* leakage and violation rates are flagged events divided by observed events.

``None`` is the only N/A representation.  It is used when a denominator or a
required oracle does not exist; it is never silently replaced with zero:

* ``trade_rate`` is N/A when the buyer population is empty;
* ``inventory_correctness`` is N/A when no transaction mutates inventory;
* ``exposure_fairness`` is N/A when there are no eligible merchants or no
  positive exposure weight;
* privacy/protocol rates are N/A when their corresponding event stream is empty;
* ``allocative_efficiency`` is N/A when optimal welfare is zero or the built-in
  unit-demand oracle is not applicable.

The built-in allocation oracle computes an exact maximum-weight capacitated
bipartite matching over sparse valuation edges.  It assumes one unit of demand
per buyer and accepts arbitrary non-negative listing capacities.  If realized
transactions contain multiple units or multiple trades for one buyer, callers
must provide :class:`AllocationOracle`; the evaluator does not invent a demand
model.  Large matching problems also require an explicit oracle so metric
evaluation cannot accidentally dominate benchmark runtime.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Literal


_RATIO: Literal["ratio"] = "ratio"
_MONEY: Literal["minor_currency_units"] = "minor_currency_units"
_MAX_BUILTIN_MATCHING_EDGES = 2_000
_DECIMAL_PRECISION = 28


class MarketMetricsInputError(ValueError):
    """Raised when market records are incomplete, ambiguous, or inconsistent."""


@dataclass(frozen=True, slots=True)
class BuyerValuation:
    """One buyer's unit value for one merchant listing.

    A buyer may have many records (one sparse edge per listing), but an edge must
    be unique within an evaluation call.  The built-in oracle treats the buyer
    as unit-demand across all of those edges.
    """

    buyer_id: str
    merchant_id: str
    listing_id: str
    unit_value_minor: int


@dataclass(frozen=True, slots=True)
class MerchantFloor:
    """Private unit floor and initial capacity for one merchant listing."""

    merchant_id: str
    listing_id: str
    unit_floor_minor: int
    capacity: int = 1


@dataclass(frozen=True, slots=True)
class MarketTransaction:
    """One completed transaction included in the market outcome.

    ``inventory_before`` and ``inventory_after`` are optional but must be
    supplied together.  Aggregate sold quantity is always checked against the
    matching :class:`MerchantFloor.capacity`; supplied observations add one
    transaction-local decrement check each.
    """

    transaction_id: str
    buyer_id: str
    merchant_id: str
    listing_id: str
    unit_price_minor: int
    quantity: int = 1
    inventory_before: int | None = None
    inventory_after: int | None = None


@dataclass(frozen=True, slots=True)
class Exposure:
    """One (optionally position-weighted) listing or merchant impression."""

    exposure_id: str
    buyer_id: str
    merchant_id: str
    listing_id: str | None = None
    weight: Decimal = Decimal(1)


@dataclass(frozen=True, slots=True)
class PrivacyEvent:
    """One deterministic privacy check and whether it detected a leak."""

    event_id: str
    actor_id: str
    leaked: bool


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    """One deterministic protocol check and whether it detected a violation."""

    event_id: str
    actor_id: str
    violated: bool


@dataclass(frozen=True, slots=True)
class AllocationOracle:
    """Externally established optimal welfare for a non-unit-demand market.

    ``oracle_id`` should identify the deterministic solver, fixture, or answer
    key used to establish the value.  The evaluator verifies that the oracle is
    non-negative and not below realized welfare.
    """

    optimal_social_welfare_minor: int
    oracle_id: str


MetricUnit = Literal["ratio", "minor_currency_units"]


@dataclass(frozen=True, slots=True)
class MarketMetricValue:
    """One metric plus the exact operands used to derive it.

    ``value is None`` means N/A and requires a non-empty ``reason``.  Monetary
    totals do not have a denominator.  Ratio metrics expose both numerator and
    denominator to make aggregation auditable.
    """

    value: Decimal | int | None
    unit: MetricUnit
    numerator: Decimal | int | None = None
    denominator: Decimal | int | None = None
    reason: str | None = None

    @property
    def is_applicable(self) -> bool:
        """Whether this metric has a defined value for the supplied market."""

        return self.value is not None


@dataclass(frozen=True, slots=True)
class MarketMetrics:
    """The complete deterministic market-level score bundle."""

    trade_rate: MarketMetricValue
    consumer_surplus: MarketMetricValue
    producer_surplus: MarketMetricValue
    social_welfare: MarketMetricValue
    allocative_efficiency: MarketMetricValue
    inventory_correctness: MarketMetricValue
    exposure_fairness: MarketMetricValue
    privacy_leakage_rate: MarketMetricValue
    protocol_violation_rate: MarketMetricValue


@dataclass(slots=True)
class _ResidualEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketMetricsInputError(f"{field} must be a non-empty string")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MarketMetricsInputError(f"{field} must be an integer")
    if value < minimum:
        raise MarketMetricsInputError(f"{field} must be >= {minimum}")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise MarketMetricsInputError(f"{field} must be a bool")
    return value


def _require_weight(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise MarketMetricsInputError(f"{field} must be a Decimal or integer, not a float")
    weight = value if isinstance(value, Decimal) else Decimal(value)
    if not weight.is_finite() or weight < 0:
        raise MarketMetricsInputError(f"{field} must be finite and >= 0")
    return weight


def _unique_population(values: Iterable[str] | None, field: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    materialized = tuple(values)
    for index, value in enumerate(materialized):
        _require_id(value, f"{field}[{index}]")
    if len(materialized) != len(set(materialized)):
        raise MarketMetricsInputError(f"{field} must not contain duplicates")
    return tuple(sorted(materialized))


def _ratio(
    numerator: Decimal | int,
    denominator: Decimal | int,
) -> MarketMetricValue:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        value = Decimal(numerator) / Decimal(denominator)
    return MarketMetricValue(
        value=value,
        unit=_RATIO,
        numerator=numerator,
        denominator=denominator,
    )


def _not_applicable(unit: MetricUnit, reason: str) -> MarketMetricValue:
    return MarketMetricValue(value=None, unit=unit, reason=reason)


def _money(value: int) -> MarketMetricValue:
    return MarketMetricValue(value=value, unit=_MONEY, numerator=value)


def _add_edge(
    graph: list[list[_ResidualEdge]],
    source: int,
    target: int,
    capacity: int,
    cost: int,
) -> None:
    forward = _ResidualEdge(target, len(graph[target]), capacity, cost)
    reverse = _ResidualEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)


def _shortest_residual_path(
    graph: Sequence[Sequence[_ResidualEdge]],
    source: int,
    sink: int,
) -> tuple[int | None, list[tuple[int, int] | None]]:
    """Bellman-Ford over a small residual network, with stable tie handling."""

    distances: list[int | None] = [None] * len(graph)
    previous: list[tuple[int, int] | None] = [None] * len(graph)
    distances[source] = 0
    for _ in range(len(graph) - 1):
        changed = False
        for node, edges in enumerate(graph):
            distance = distances[node]
            if distance is None:
                continue
            for edge_index, edge in enumerate(edges):
                if edge.capacity <= 0:
                    continue
                candidate = distance + edge.cost
                target_distance = distances[edge.to]
                if target_distance is None or candidate < target_distance:
                    distances[edge.to] = candidate
                    previous[edge.to] = (node, edge_index)
                    changed = True
        if not changed:
            break
    return distances[sink], previous


def _optimal_unit_demand_welfare(
    valuations: Sequence[BuyerValuation],
    floors: dict[tuple[str, str], MerchantFloor],
) -> int:
    """Return exact welfare for a sparse unit-demand, capacitated market."""

    buyers = sorted({valuation.buyer_id for valuation in valuations})
    offers = sorted(floors)
    source = 0
    buyer_offset = 1
    offer_offset = buyer_offset + len(buyers)
    sink = offer_offset + len(offers)
    graph: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]
    buyer_nodes = {buyer_id: buyer_offset + i for i, buyer_id in enumerate(buyers)}
    offer_nodes = {offer: offer_offset + i for i, offer in enumerate(offers)}

    for buyer_id in buyers:
        _add_edge(graph, source, buyer_nodes[buyer_id], 1, 0)
    for offer in offers:
        capacity = floors[offer].capacity
        if capacity > 0:
            _add_edge(graph, offer_nodes[offer], sink, capacity, 0)
    for valuation in sorted(
        valuations,
        key=lambda item: (item.buyer_id, item.merchant_id, item.listing_id),
    ):
        offer = (valuation.merchant_id, valuation.listing_id)
        surplus = valuation.unit_value_minor - floors[offer].unit_floor_minor
        if surplus > 0 and floors[offer].capacity > 0:
            _add_edge(graph, buyer_nodes[valuation.buyer_id], offer_nodes[offer], 1, -surplus)

    minimum_cost = 0
    while True:
        path_cost, previous = _shortest_residual_path(graph, source, sink)
        if path_cost is None or path_cost >= 0:
            break
        node = sink
        while node != source:
            step = previous[node]
            if step is None:  # pragma: no cover - guarded by a finite sink distance
                raise RuntimeError("residual path is missing a predecessor")
            previous_node, edge_index = step
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous_node
        minimum_cost += path_cost
    return -minimum_cost


def _validate_records(
    valuations: Sequence[BuyerValuation],
    floors: Sequence[MerchantFloor],
    transactions: Sequence[MarketTransaction],
    exposures: Sequence[Exposure],
    privacy_events: Sequence[PrivacyEvent],
    protocol_events: Sequence[ProtocolEvent],
    buyer_ids: tuple[str, ...] | None,
    merchant_ids: tuple[str, ...] | None,
    allocation_oracle: AllocationOracle | None,
) -> tuple[
    dict[tuple[str, str, str], BuyerValuation],
    dict[tuple[str, str], MerchantFloor],
    tuple[str, ...],
    tuple[str, ...],
]:
    valuation_index: dict[tuple[str, str, str], BuyerValuation] = {}
    for index, valuation in enumerate(valuations):
        if not isinstance(valuation, BuyerValuation):
            raise MarketMetricsInputError(f"valuations[{index}] must be BuyerValuation")
        _require_id(valuation.buyer_id, f"valuations[{index}].buyer_id")
        _require_id(valuation.merchant_id, f"valuations[{index}].merchant_id")
        _require_id(valuation.listing_id, f"valuations[{index}].listing_id")
        _require_int(valuation.unit_value_minor, f"valuations[{index}].unit_value_minor")
        valuation_key = (valuation.buyer_id, valuation.merchant_id, valuation.listing_id)
        if valuation_key in valuation_index:
            raise MarketMetricsInputError(f"duplicate valuation edge: {valuation_key!r}")
        valuation_index[valuation_key] = valuation

    floor_index: dict[tuple[str, str], MerchantFloor] = {}
    for index, floor in enumerate(floors):
        if not isinstance(floor, MerchantFloor):
            raise MarketMetricsInputError(f"merchant_floors[{index}] must be MerchantFloor")
        _require_id(floor.merchant_id, f"merchant_floors[{index}].merchant_id")
        _require_id(floor.listing_id, f"merchant_floors[{index}].listing_id")
        _require_int(floor.unit_floor_minor, f"merchant_floors[{index}].unit_floor_minor")
        _require_int(floor.capacity, f"merchant_floors[{index}].capacity")
        floor_key = (floor.merchant_id, floor.listing_id)
        if floor_key in floor_index:
            raise MarketMetricsInputError(f"duplicate merchant floor: {floor_key!r}")
        floor_index[floor_key] = floor

    inferred_buyers = tuple(sorted({key[0] for key in valuation_index}))
    inferred_merchants = tuple(sorted({key[0] for key in floor_index}))
    buyers = inferred_buyers if buyer_ids is None else buyer_ids
    merchants = inferred_merchants if merchant_ids is None else merchant_ids
    buyer_set = set(buyers)
    merchant_set = set(merchants)

    for key in valuation_index:
        buyer_id, merchant_id, listing_id = key
        if buyer_id not in buyer_set:
            raise MarketMetricsInputError(f"valuation references buyer outside buyer_ids: {buyer_id!r}")
        if merchant_id not in merchant_set:
            raise MarketMetricsInputError(
                f"valuation references merchant outside merchant_ids: {merchant_id!r}"
            )
        if (merchant_id, listing_id) not in floor_index:
            raise MarketMetricsInputError(
                f"valuation has no matching merchant floor: {(merchant_id, listing_id)!r}"
            )
    for merchant_id, _listing_id in floor_index:
        if merchant_id not in merchant_set:
            raise MarketMetricsInputError(
                f"merchant floor references merchant outside merchant_ids: {merchant_id!r}"
            )

    seen_transactions: set[str] = set()
    for index, transaction in enumerate(transactions):
        if not isinstance(transaction, MarketTransaction):
            raise MarketMetricsInputError(
                f"transactions[{index}] must be MarketTransaction"
            )
        _require_id(transaction.transaction_id, f"transactions[{index}].transaction_id")
        _require_id(transaction.buyer_id, f"transactions[{index}].buyer_id")
        _require_id(transaction.merchant_id, f"transactions[{index}].merchant_id")
        _require_id(transaction.listing_id, f"transactions[{index}].listing_id")
        _require_int(transaction.unit_price_minor, f"transactions[{index}].unit_price_minor")
        _require_int(transaction.quantity, f"transactions[{index}].quantity", minimum=1)
        if transaction.transaction_id in seen_transactions:
            raise MarketMetricsInputError(
                f"duplicate transaction_id: {transaction.transaction_id!r}"
            )
        seen_transactions.add(transaction.transaction_id)
        edge = (transaction.buyer_id, transaction.merchant_id, transaction.listing_id)
        if edge not in valuation_index:
            raise MarketMetricsInputError(f"transaction has no matching valuation: {edge!r}")
        offer = (transaction.merchant_id, transaction.listing_id)
        if offer not in floor_index:
            raise MarketMetricsInputError(f"transaction has no matching merchant floor: {offer!r}")
        before, after = transaction.inventory_before, transaction.inventory_after
        if (before is None) != (after is None):
            raise MarketMetricsInputError(
                f"transaction {transaction.transaction_id!r} must supply both inventory values or neither"
            )
        if before is not None:
            _require_int(before, f"transactions[{index}].inventory_before")
            _require_int(after, f"transactions[{index}].inventory_after")

    seen_exposures: set[str] = set()
    for index, exposure in enumerate(exposures):
        if not isinstance(exposure, Exposure):
            raise MarketMetricsInputError(f"exposures[{index}] must be Exposure")
        _require_id(exposure.exposure_id, f"exposures[{index}].exposure_id")
        _require_id(exposure.buyer_id, f"exposures[{index}].buyer_id")
        _require_id(exposure.merchant_id, f"exposures[{index}].merchant_id")
        _require_weight(exposure.weight, f"exposures[{index}].weight")
        if exposure.exposure_id in seen_exposures:
            raise MarketMetricsInputError(f"duplicate exposure_id: {exposure.exposure_id!r}")
        seen_exposures.add(exposure.exposure_id)
        if exposure.buyer_id not in buyer_set:
            raise MarketMetricsInputError(
                f"exposure references buyer outside buyer_ids: {exposure.buyer_id!r}"
            )
        if exposure.merchant_id not in merchant_set:
            raise MarketMetricsInputError(
                f"exposure references merchant outside merchant_ids: {exposure.merchant_id!r}"
            )
        if exposure.listing_id is not None:
            _require_id(exposure.listing_id, f"exposures[{index}].listing_id")
            offer = (exposure.merchant_id, exposure.listing_id)
            if offer not in floor_index:
                raise MarketMetricsInputError(f"exposure has no matching merchant floor: {offer!r}")

    seen_privacy_events: set[str] = set()
    for index, privacy_event in enumerate(privacy_events):
        if not isinstance(privacy_event, PrivacyEvent):
            raise MarketMetricsInputError(
                f"privacy_events[{index}] must be PrivacyEvent"
            )
        _require_id(privacy_event.event_id, f"privacy_events[{index}].event_id")
        _require_id(privacy_event.actor_id, f"privacy_events[{index}].actor_id")
        _require_bool(privacy_event.leaked, f"privacy_events[{index}].leaked")
        if privacy_event.event_id in seen_privacy_events:
            raise MarketMetricsInputError(
                f"duplicate privacy_events event_id: {privacy_event.event_id!r}"
            )
        seen_privacy_events.add(privacy_event.event_id)

    seen_protocol_events: set[str] = set()
    for index, protocol_event in enumerate(protocol_events):
        if not isinstance(protocol_event, ProtocolEvent):
            raise MarketMetricsInputError(
                f"protocol_events[{index}] must be ProtocolEvent"
            )
        _require_id(protocol_event.event_id, f"protocol_events[{index}].event_id")
        _require_id(protocol_event.actor_id, f"protocol_events[{index}].actor_id")
        _require_bool(protocol_event.violated, f"protocol_events[{index}].violated")
        if protocol_event.event_id in seen_protocol_events:
            raise MarketMetricsInputError(
                f"duplicate protocol_events event_id: {protocol_event.event_id!r}"
            )
        seen_protocol_events.add(protocol_event.event_id)

    if allocation_oracle is not None:
        if not isinstance(allocation_oracle, AllocationOracle):
            raise MarketMetricsInputError("allocation_oracle must be AllocationOracle")
        _require_int(
            allocation_oracle.optimal_social_welfare_minor,
            "allocation_oracle.optimal_social_welfare_minor",
        )
        _require_id(allocation_oracle.oracle_id, "allocation_oracle.oracle_id")

    return valuation_index, floor_index, buyers, merchants


def compute_market_metrics(
    *,
    valuations: Iterable[BuyerValuation],
    merchant_floors: Iterable[MerchantFloor],
    transactions: Iterable[MarketTransaction] = (),
    exposures: Iterable[Exposure] = (),
    privacy_events: Iterable[PrivacyEvent] = (),
    protocol_events: Iterable[ProtocolEvent] = (),
    buyer_ids: Iterable[str] | None = None,
    merchant_ids: Iterable[str] | None = None,
    allocation_oracle: AllocationOracle | None = None,
) -> MarketMetrics:
    """Compute deterministic market-level metrics from structured records.

    ``buyer_ids`` and ``merchant_ids`` define the population denominators.  If
    omitted, buyers are inferred from valuations and merchants from floors.
    Passing the explicit sets is recommended when zero-option buyers or
    zero-listing merchants must remain in the denominator.

    Every completed transaction must have exactly one matching valuation and
    merchant floor.  This strictness prevents a missing private value from being
    silently interpreted as zero.
    """

    valuation_records = tuple(valuations)
    floor_records = tuple(merchant_floors)
    transaction_records = tuple(transactions)
    exposure_records = tuple(exposures)
    privacy_records = tuple(privacy_events)
    protocol_records = tuple(protocol_events)
    explicit_buyers = _unique_population(buyer_ids, "buyer_ids")
    explicit_merchants = _unique_population(merchant_ids, "merchant_ids")
    valuation_index, floor_index, buyers, merchants = _validate_records(
        valuation_records,
        floor_records,
        transaction_records,
        exposure_records,
        privacy_records,
        protocol_records,
        explicit_buyers,
        explicit_merchants,
        allocation_oracle,
    )

    traded_buyers = {transaction.buyer_id for transaction in transaction_records}
    trade_rate = (
        _ratio(len(traded_buyers), len(buyers))
        if buyers
        else _not_applicable(_RATIO, "buyer population is empty")
    )

    consumer_surplus = 0
    producer_surplus = 0
    sold_quantities: Counter[tuple[str, str]] = Counter()
    for transaction in transaction_records:
        edge = (transaction.buyer_id, transaction.merchant_id, transaction.listing_id)
        offer = (transaction.merchant_id, transaction.listing_id)
        valuation = valuation_index[edge]
        floor = floor_index[offer]
        consumer_surplus += (
            valuation.unit_value_minor - transaction.unit_price_minor
        ) * transaction.quantity
        producer_surplus += (
            transaction.unit_price_minor - floor.unit_floor_minor
        ) * transaction.quantity
        sold_quantities[offer] += transaction.quantity
    social_welfare = consumer_surplus + producer_surplus

    if allocation_oracle is not None:
        optimal_welfare = allocation_oracle.optimal_social_welfare_minor
        oracle_source = f"explicit oracle {allocation_oracle.oracle_id!r}"
        allocative_reason: str | None = None
    else:
        trade_counts = Counter(transaction.buyer_id for transaction in transaction_records)
        non_unit_realization = any(
            transaction.quantity != 1 for transaction in transaction_records
        ) or any(count > 1 for count in trade_counts.values())
        if non_unit_realization:
            optimal_welfare = None
            allocative_reason = (
                "realized allocation is not unit-demand; provide AllocationOracle"
            )
        elif len(valuation_records) > _MAX_BUILTIN_MATCHING_EDGES:
            optimal_welfare = None
            allocative_reason = (
                f"built-in exact matching is limited to {_MAX_BUILTIN_MATCHING_EDGES} "
                "valuation edges; provide AllocationOracle"
            )
        else:
            optimal_welfare = _optimal_unit_demand_welfare(valuation_records, floor_index)
            allocative_reason = None
        oracle_source = "built-in exact unit-demand matching"

    if optimal_welfare is None:
        allocative_efficiency = _not_applicable(_RATIO, allocative_reason or "oracle unavailable")
    elif social_welfare > optimal_welfare:
        raise MarketMetricsInputError(
            f"realized social welfare {social_welfare} exceeds {oracle_source} "
            f"optimal welfare {optimal_welfare}"
        )
    elif optimal_welfare == 0:
        allocative_efficiency = _not_applicable(_RATIO, "optimal social welfare is zero")
    else:
        # A welfare-destroying allocation scores zero rather than producing a
        # negative ratio; the signed social_welfare total remains separately visible.
        allocative_efficiency = _ratio(max(social_welfare, 0), optimal_welfare)

    if not transaction_records:
        inventory_correctness = _not_applicable(_RATIO, "no inventory-mutating transactions")
    else:
        inventory_checks: list[bool] = []
        for offer, sold_quantity in sorted(sold_quantities.items()):
            inventory_checks.append(sold_quantity <= floor_index[offer].capacity)
        for transaction in transaction_records:
            if transaction.inventory_before is not None:
                before = transaction.inventory_before
                after = transaction.inventory_after
                # The validator guarantees both are integers here.
                assert after is not None
                inventory_checks.append(
                    before >= transaction.quantity
                    and after == before - transaction.quantity
                )
        inventory_correctness = _ratio(sum(inventory_checks), len(inventory_checks))

    if not merchants:
        exposure_fairness = _not_applicable(_RATIO, "eligible merchant population is empty")
    else:
        with localcontext() as context:
            context.prec = _DECIMAL_PRECISION
            context.rounding = ROUND_HALF_EVEN
            exposure_by_merchant = {merchant_id: Decimal(0) for merchant_id in merchants}
            for index, exposure in enumerate(exposure_records):
                exposure_by_merchant[exposure.merchant_id] += _require_weight(
                    exposure.weight, f"exposures[{index}].weight"
                )
            total_exposure = sum(exposure_by_merchant.values(), Decimal(0))
            squared_total = total_exposure * total_exposure
            fairness_denominator = Decimal(len(merchants)) * sum(
                (weight * weight for weight in exposure_by_merchant.values()),
                Decimal(0),
            )
        if total_exposure == 0:
            exposure_fairness = _not_applicable(_RATIO, "total exposure weight is zero")
        else:
            exposure_fairness = _ratio(squared_total, fairness_denominator)

    privacy_leakage_rate = (
        _ratio(sum(event.leaked for event in privacy_records), len(privacy_records))
        if privacy_records
        else _not_applicable(_RATIO, "privacy event stream is empty")
    )
    protocol_violation_rate = (
        _ratio(sum(event.violated for event in protocol_records), len(protocol_records))
        if protocol_records
        else _not_applicable(_RATIO, "protocol event stream is empty")
    )

    return MarketMetrics(
        trade_rate=trade_rate,
        consumer_surplus=_money(consumer_surplus),
        producer_surplus=_money(producer_surplus),
        social_welfare=_money(social_welfare),
        allocative_efficiency=allocative_efficiency,
        inventory_correctness=inventory_correctness,
        exposure_fairness=exposure_fairness,
        privacy_leakage_rate=privacy_leakage_rate,
        protocol_violation_rate=protocol_violation_rate,
    )


__all__ = [
    "AllocationOracle",
    "BuyerValuation",
    "Exposure",
    "MarketMetricValue",
    "MarketMetrics",
    "MarketMetricsInputError",
    "MarketTransaction",
    "MerchantFloor",
    "PrivacyEvent",
    "ProtocolEvent",
    "compute_market_metrics",
]
