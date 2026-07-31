"""Adapt one executed many-to-many Episode into a verified market artifact.

The adapter reads only authoritative, persisted Episode artifacts plus the
scenario's server-side ``success_oracle.market_oracle`` answer key.  It never
infers a buyer valuation or merchant floor from model text, public list prices,
or a participant's revealed negotiation behavior.

Privacy and protocol rates are deliberately left N/A until a dedicated,
structured observation stream exists.  A blocked security event is an attempted
violation, not a leaked secret, and therefore is not relabeled as either a
positive or negative ``PrivacyEvent`` merely to manufacture a denominator.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from episode.scenario import population_for_scenario
from evals.market_metrics import (
    AllocationOracle,
    BuyerValuation,
    Exposure,
    MarketMetricsInputError,
    MarketTransaction,
    MerchantFloor,
)
from evals.serialize import WORLD_SCHEMA_VERSION
from experiments.market_artifacts import (
    MarketArtifactError,
    build_market_artifact,
    verify_market_artifact,
)
from protocol.envelope import from_json

if TYPE_CHECKING:
    from episode.types import ScenarioSpec
    from experiments.plan import RunSpec


MARKET_ORACLE_SCHEMA = "cwe.market-oracle.v1"
_MARKET_METRIC_NAMES = (
    "trade_rate",
    "consumer_surplus",
    "producer_surplus",
    "social_welfare",
    "allocative_efficiency",
    "inventory_correctness",
    "exposure_fairness",
    "privacy_leakage_rate",
    "protocol_violation_rate",
)


class MarketEpisodeError(ValueError):
    """Episode artifacts or the hidden market answer key are inconsistent."""


def build_market_artifact_from_episode(
    *,
    scenario: "ScenarioSpec",
    run: "RunSpec",
    artifact_dir: str | Path,
    observations: Sequence[Mapping[str, Any]] = (),
    stop_reason: str | None = None,
    truncated: bool = False,
    model_call_limit: int | None = None,
    model_calls_used: int | None = None,
) -> dict[str, Any]:
    """Build a self-verifying market result from one completed or truncated Episode.

    ``artifact_dir`` must contain ``world.initial.json``, ``world.final.json``,
    and ``audit.jsonl``.  Missing or inconsistent private answer-key records are
    a hard error: computing surplus from an guessed value would make the paper
    metric irreproducible.
    """

    if run.suite != "many_to_many":
        raise MarketEpisodeError(
            "market Episode artifacts are only defined for the many_to_many suite"
        )
    root = Path(artifact_dir)
    population = population_for_scenario(scenario)
    buyer_ids = tuple(buyer.buyer_id for buyer in population.buyers)
    merchant_ids = tuple(merchant.merchant_id for merchant in population.merchants)
    if (len(buyer_ids), len(merchant_ids)) != (run.buyers, run.merchants):
        raise MarketEpisodeError(
            "scenario population does not match run identity: "
            f"scenario={len(buyer_ids)}x{len(merchant_ids)}, "
            f"run={run.buyers}x{run.merchants}"
        )

    valuations, floors, allocation_oracle = _parse_market_oracle(
        scenario,
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
    )
    initial = _load_snapshot(root / "world.initial.json", phase="initial")
    final = _load_snapshot(root / "world.final.json", phase="final")
    _validate_answer_key_against_world(
        valuations=valuations,
        floors=floors,
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
        initial=initial,
    )
    transactions = _market_transactions(initial=initial, final=final)
    audit_records = _read_jsonl(root / "audit.jsonl", required=True)
    exposures = _market_exposures(
        audit_records,
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
        floors=floors,
    )

    task_family = run.task_family
    if task_family is None:
        raise MarketEpisodeError(
            f"many-to-many variant {run.variant_id} has no task family"
        )
    try:
        artifact = build_market_artifact(
            market_id=run.run_key,
            variant_id=run.variant_id,
            task_family=task_family,
            seed=run.seed,
            buyer_ids=buyer_ids,
            merchant_ids=merchant_ids,
            valuations=valuations,
            merchant_floors=floors,
            transactions=transactions,
            exposures=exposures,
            # Do not fabricate all-false observations.  These two metrics remain
            # explicitly N/A in cwe.market-artifact.v2 when the streams are empty.
            privacy_events=(),
            protocol_events=(),
            allocation_oracle=allocation_oracle,
            execution={
                "source": "episode_artifacts",
                "scenario_id": scenario.scenario_id,
                "suite": run.suite,
                "model_id": run.model_id,
                "evaluated_role": run.evaluated_role,
                "rollout": run.rollout,
                "run_key": run.run_key,
                "model_call_count": sum(
                    1
                    for row in observations
                    if row.get("source") == "model" and not row.get("error_type")
                ),
                # Execution completion is distinct from task success: a model may
                # naturally finish without solving the task.  A scoreable early
                # abort is the only incomplete execution represented here.
                "completed": not truncated,
                "stop_reason": stop_reason,
                "truncated": truncated,
                "model_call_limit": model_call_limit,
                "model_calls_used": model_calls_used,
                "market_oracle_schema": MARKET_ORACLE_SCHEMA,
                "source_artifact_digests": {
                    "audit": _file_digest(root / "audit.jsonl"),
                    "world_initial": _file_digest(root / "world.initial.json"),
                    "world_final": _file_digest(root / "world.final.json"),
                },
                "privacy_observation_stream": "unavailable_n_a",
                "protocol_observation_stream": "unavailable_n_a",
            },
        )
        verify_market_artifact(artifact)
    except (MarketArtifactError, MarketMetricsInputError) as exc:
        raise MarketEpisodeError(f"invalid market artifact derived from Episode: {exc}") from exc
    return artifact


def market_metrics_for_result(
    artifact: Mapping[str, Any],
) -> dict[str, bool | float | int | None]:
    """Flatten exact market metrics into scalar experiment-result columns.

    Exact Decimal strings remain preserved in ``market.json``.  Ratio values
    are converted to floats only in the convenience result inventory, whose
    schema accepts scalar JSON values and is used for aggregate plotting.
    """

    try:
        verify_market_artifact(artifact)
    except MarketArtifactError as exc:
        raise MarketEpisodeError(f"cannot flatten an invalid market artifact: {exc}") from exc
    metrics = artifact["metrics"]
    flattened: dict[str, bool | float | int | None] = {}
    for name in _MARKET_METRIC_NAMES:
        row = metrics[name]
        raw = row["value"]
        flattened[f"market_{name}_applicable"] = bool(row["applicable"])
        if raw is None:
            value: float | int | None = None
        elif row["unit"] == "minor_currency_units":
            value = int(raw)
        else:
            value = float(Decimal(str(raw)))
        flattened[f"market_{name}"] = value
    return flattened


def _parse_market_oracle(
    scenario: "ScenarioSpec",
    *,
    buyer_ids: Sequence[str],
    merchant_ids: Sequence[str],
) -> tuple[tuple[BuyerValuation, ...], tuple[MerchantFloor, ...], AllocationOracle | None]:
    raw = scenario.success_oracle.get("market_oracle")
    if not isinstance(raw, Mapping):
        raise MarketEpisodeError(
            "many-to-many scenario requires server-side success_oracle.market_oracle"
        )
    schema = raw.get("schema_version", MARKET_ORACLE_SCHEMA)
    if schema != MARKET_ORACLE_SCHEMA:
        raise MarketEpisodeError(f"unsupported market oracle schema: {schema!r}")
    valuation_rows = _mapping_rows(raw.get("buyer_valuations"), "buyer_valuations")
    floor_rows = _mapping_rows(raw.get("merchant_floors"), "merchant_floors")
    buyer_set = set(buyer_ids)
    merchant_set = set(merchant_ids)

    valuations: list[BuyerValuation] = []
    for index, row in enumerate(valuation_rows):
        buyer_id = _id(row.get("buyer_id"), f"buyer_valuations[{index}].buyer_id")
        merchant_id = _id(
            row.get("merchant_id"), f"buyer_valuations[{index}].merchant_id"
        )
        if buyer_id not in buyer_set:
            raise MarketEpisodeError(
                f"buyer_valuations[{index}] references unknown buyer {buyer_id!r}"
            )
        if merchant_id not in merchant_set:
            raise MarketEpisodeError(
                f"buyer_valuations[{index}] references unknown merchant {merchant_id!r}"
            )
        valuations.append(BuyerValuation(
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            listing_id=_id(
                row.get("listing_id"), f"buyer_valuations[{index}].listing_id"
            ),
            unit_value_minor=_integer(
                row.get("unit_value_minor"),
                f"buyer_valuations[{index}].unit_value_minor",
                minimum=0,
            ),
        ))

    floors: list[MerchantFloor] = []
    for index, row in enumerate(floor_rows):
        merchant_id = _id(
            row.get("merchant_id"), f"merchant_floors[{index}].merchant_id"
        )
        if merchant_id not in merchant_set:
            raise MarketEpisodeError(
                f"merchant_floors[{index}] references unknown merchant {merchant_id!r}"
            )
        floors.append(MerchantFloor(
            merchant_id=merchant_id,
            listing_id=_id(
                row.get("listing_id"), f"merchant_floors[{index}].listing_id"
            ),
            unit_floor_minor=_integer(
                row.get("unit_floor_minor"),
                f"merchant_floors[{index}].unit_floor_minor",
                minimum=0,
            ),
            capacity=_integer(
                row.get("capacity", 1),
                f"merchant_floors[{index}].capacity",
                minimum=0,
            ),
        ))

    allocation_raw = raw.get("allocation_oracle")
    allocation_oracle: AllocationOracle | None = None
    if allocation_raw is not None:
        if not isinstance(allocation_raw, Mapping):
            raise MarketEpisodeError("allocation_oracle must be an object or null")
        allocation_oracle = AllocationOracle(
            optimal_social_welfare_minor=_integer(
                allocation_raw.get("optimal_social_welfare_minor"),
                "allocation_oracle.optimal_social_welfare_minor",
                minimum=0,
            ),
            oracle_id=_id(allocation_raw.get("oracle_id"), "allocation_oracle.oracle_id"),
        )
    return tuple(valuations), tuple(floors), allocation_oracle


def _validate_answer_key_against_world(
    *,
    valuations: Sequence[BuyerValuation],
    floors: Sequence[MerchantFloor],
    buyer_ids: Sequence[str],
    merchant_ids: Sequence[str],
    initial: Mapping[str, Any],
) -> None:
    catalog = _catalog_index(initial)
    floor_index: dict[tuple[str, str], MerchantFloor] = {}
    for floor in floors:
        key = (floor.merchant_id, floor.listing_id)
        if key in floor_index:
            raise MarketEpisodeError(f"duplicate market floor edge: {key!r}")
        listing = catalog.get(floor.listing_id)
        if listing is None:
            raise MarketEpisodeError(
                f"market floor listing is absent from initial catalog: {floor.listing_id!r}"
            )
        if str(listing.get("merchant_id")) != floor.merchant_id:
            raise MarketEpisodeError(
                f"market floor owner disagrees with catalog for {floor.listing_id!r}"
            )
        available = _inventory_available(initial, floor.listing_id)
        if available != floor.capacity:
            raise MarketEpisodeError(
                f"market floor capacity disagrees with initial inventory for "
                f"{floor.listing_id!r}: oracle={floor.capacity}, world={available}"
            )
        floor_index[key] = floor

    expected_merchants = set(merchant_ids)
    covered_merchants = {floor.merchant_id for floor in floors}
    if covered_merchants != expected_merchants:
        raise MarketEpisodeError(
            "market floors do not cover the exact merchant population: "
            f"missing={sorted(expected_merchants - covered_merchants)}, "
            f"unexpected={sorted(covered_merchants - expected_merchants)}"
        )

    seen_values: set[tuple[str, str, str]] = set()
    valued_buyers: set[str] = set()
    for valuation in valuations:
        edge = (valuation.buyer_id, valuation.merchant_id, valuation.listing_id)
        if edge in seen_values:
            raise MarketEpisodeError(f"duplicate market valuation edge: {edge!r}")
        if (valuation.merchant_id, valuation.listing_id) not in floor_index:
            raise MarketEpisodeError(f"market valuation has no matching floor: {edge!r}")
        seen_values.add(edge)
        valued_buyers.add(valuation.buyer_id)
    missing_buyers = set(buyer_ids) - valued_buyers
    if missing_buyers:
        raise MarketEpisodeError(
            f"market oracle has no valuation edge for buyers {sorted(missing_buyers)}"
        )


def _market_transactions(
    *, initial: Mapping[str, Any], final: Mapping[str, Any]
) -> tuple[MarketTransaction, ...]:
    final_tables = _tables(final)
    ledger = _row_list(final_tables.get("ledger"), "final.tables.ledger")
    orders = {
        str(row.get("order_id")): row
        for row in _row_list(final_tables.get("orders"), "final.tables.orders")
    }
    rows: list[MarketTransaction] = []
    for index, receipt in enumerate(ledger):
        transaction_id = _id(receipt.get("txn_id"), f"ledger[{index}].txn_id")
        if transaction_id.startswith("refund:"):
            continue
        order_id = _id(receipt.get("order_id"), f"ledger[{index}].order_id")
        order = orders.get(order_id)
        if order is None:
            raise MarketEpisodeError(
                f"settlement receipt {transaction_id!r} has no final order {order_id!r}"
            )
        buyer_id = _id(receipt.get("buyer_id"), f"ledger[{index}].buyer_id")
        merchant_id = _id(receipt.get("merchant_id"), f"ledger[{index}].merchant_id")
        listing_id = _id(receipt.get("sku_id"), f"ledger[{index}].sku_id")
        for field, receipt_value, order_key in (
            ("buyer_id", buyer_id, "buyer_id"),
            ("merchant_id", merchant_id, "merchant_id"),
            ("sku_id", listing_id, "sku_id"),
        ):
            if str(order.get(order_key)) != receipt_value:
                raise MarketEpisodeError(
                    f"settlement receipt {transaction_id!r} {field} disagrees with order"
                )
        price = receipt.get("price")
        if not isinstance(price, Mapping):
            raise MarketEpisodeError(f"ledger[{index}].price must be an object")
        rows.append(MarketTransaction(
            transaction_id=transaction_id,
            buyer_id=buyer_id,
            merchant_id=merchant_id,
            listing_id=listing_id,
            unit_price_minor=_money_minor(price, f"ledger[{index}].price"),
            quantity=_integer(receipt.get("qty"), f"ledger[{index}].qty", minimum=1),
        ))

    # Reconstruct intermediate decrements in authoritative ledger order, while
    # anchoring the last observation to the actual final snapshot.  This catches
    # an aggregate inventory drift without claiming a timestamped observation
    # that the Episode did not persist.
    positions: dict[str, list[int]] = defaultdict(list)
    for index, transaction in enumerate(rows):
        positions[transaction.listing_id].append(index)
    for listing_id, indices in positions.items():
        before = _inventory_available(initial, listing_id)
        actual_final = _inventory_available(final, listing_id)
        for ordinal, index in enumerate(indices):
            transaction = rows[index]
            expected_after = before - transaction.quantity
            after = actual_final if ordinal == len(indices) - 1 else expected_after
            rows[index] = replace(
                transaction,
                inventory_before=before,
                inventory_after=after,
            )
            before = expected_after
    return tuple(rows)


def _market_exposures(
    audit_records: Iterable[Mapping[str, Any]],
    *,
    buyer_ids: Sequence[str],
    merchant_ids: Sequence[str],
    floors: Sequence[MerchantFloor],
) -> tuple[Exposure, ...]:
    buyers = set(buyer_ids)
    merchants = set(merchant_ids)
    floor_by_listing = {floor.listing_id: floor for floor in floors}
    rows: list[Exposure] = []
    for record_index, record in enumerate(audit_records):
        raw = record.get("envelope")
        if not isinstance(raw, str):
            raise MarketEpisodeError(
                f"audit[{record_index}].envelope must be canonical JSON text"
            )
        try:
            envelope = from_json(raw)
        except Exception as exc:  # protocol errors have several public subclasses
            raise MarketEpisodeError(
                f"cannot parse audit envelope at record {record_index}: {exc}"
            ) from exc
        if envelope.action.get("kind") != "platform.rank_offers":
            continue
        buyer_id = str(envelope.to)
        if buyer_id not in buyers:
            continue
        payload = envelope.action.get("payload")
        if not isinstance(payload, Mapping):
            raise MarketEpisodeError(
                f"rank event {envelope.msg_id!r} payload must be an object"
            )
        candidates = payload.get("candidates", ())
        if not isinstance(candidates, list):
            raise MarketEpisodeError(
                f"rank event {envelope.msg_id!r} candidates must be a list"
            )
        width = len(candidates)
        for rank, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, Mapping):
                raise MarketEpisodeError(
                    f"rank event {envelope.msg_id!r} candidate {rank} must be an object"
                )
            listing_id = _id(
                candidate.get("sku_id", candidate.get("listing_id")),
                f"rank event {envelope.msg_id!r} candidate {rank} listing",
            )
            floor = floor_by_listing.get(listing_id)
            if floor is None:
                raise MarketEpisodeError(
                    f"rank event {envelope.msg_id!r} exposes listing without a market floor: "
                    f"{listing_id!r}"
                )
            merchant_id = str(candidate.get("merchant_id", floor.merchant_id))
            if merchant_id != floor.merchant_id or merchant_id not in merchants:
                raise MarketEpisodeError(
                    f"rank event {envelope.msg_id!r} candidate owner disagrees with answer key"
                )
            rows.append(Exposure(
                exposure_id=f"exposure:{envelope.msg_id}:{rank}",
                buyer_id=buyer_id,
                merchant_id=merchant_id,
                listing_id=listing_id,
                weight=Decimal(width - rank + 1),
            ))
    return tuple(rows)


def _load_snapshot(path: Path, *, phase: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketEpisodeError(f"cannot read {phase} World snapshot {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MarketEpisodeError(f"{phase} World snapshot root must be an object")
    if raw.get("schema_version") != WORLD_SCHEMA_VERSION or raw.get("phase") != phase:
        raise MarketEpisodeError(
            f"invalid {phase} World snapshot schema/phase in {path}"
        )
    _tables(raw)
    return raw


def _tables(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    tables = snapshot.get("tables")
    if not isinstance(tables, Mapping):
        raise MarketEpisodeError("World snapshot tables must be an object")
    return tables


def _catalog_index(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    catalog = _row_list(_tables(snapshot).get("catalog"), "tables.catalog")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(catalog):
        sku = _id(row.get("sku_id"), f"catalog[{index}].sku_id")
        if sku in result:
            raise MarketEpisodeError(f"duplicate catalog sku_id: {sku!r}")
        result[sku] = row
    return result


def _inventory_available(snapshot: Mapping[str, Any], listing_id: str) -> int:
    inventory = _tables(snapshot).get("inventory")
    if not isinstance(inventory, Mapping):
        raise MarketEpisodeError("World snapshot inventory must be an object")
    row = inventory.get(listing_id)
    if isinstance(row, Mapping):
        available = _integer(
            row.get("qty_available"), f"inventory[{listing_id!r}].qty_available", minimum=0
        )
        reserved = _integer(
            row.get("qty_reserved", 0),
            f"inventory[{listing_id!r}].qty_reserved",
            minimum=0,
        )
        return available - reserved
    if row is not None:
        return _integer(row, f"inventory[{listing_id!r}]", minimum=0)
    raise MarketEpisodeError(f"World snapshot has no inventory for {listing_id!r}")


def _mapping_rows(raw: Any, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise MarketEpisodeError(f"market_oracle.{field} must be a non-empty list")
    if not all(isinstance(row, Mapping) for row in raw):
        raise MarketEpisodeError(f"market_oracle.{field} rows must be objects")
    return tuple(raw)


def _row_list(raw: Any, field: str) -> tuple[Mapping[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise MarketEpisodeError(f"{field} must be a list of objects")
    return tuple(raw)


def _read_jsonl(path: Path, *, required: bool) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        if required:
            raise MarketEpisodeError(f"missing Episode artifact: {path}")
        return ()
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise MarketEpisodeError(f"{path}:{index} must be a JSON object")
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketEpisodeError(f"cannot read Episode JSONL {path}: {exc}") from exc
    return tuple(rows)


def _id(raw: Any, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise MarketEpisodeError(f"{field} must be a non-empty string")
    return raw


def _integer(raw: Any, field: str, *, minimum: int) -> int:
    if isinstance(raw, bool) or raw is None:
        raise MarketEpisodeError(f"{field} must be an integer >= {minimum}")
    try:
        decimal = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise MarketEpisodeError(f"{field} must be an integer >= {minimum}") from exc
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        raise MarketEpisodeError(f"{field} must be an integer >= {minimum}")
    value = int(decimal)
    if value < minimum:
        raise MarketEpisodeError(f"{field} must be an integer >= {minimum}")
    return value


def _money_minor(raw: Mapping[str, Any], field: str) -> int:
    """Convert a canonical ``Money`` row (major USD units) to exact cents."""

    currency = raw.get("currency", "USD")
    if currency != "USD":
        raise MarketEpisodeError(
            f"{field}.currency must be 'USD' for this market oracle, got {currency!r}"
        )
    amount = raw.get("amount")
    if isinstance(amount, bool) or amount is None:
        raise MarketEpisodeError(f"{field}.amount must be a non-negative USD amount")
    try:
        major = Decimal(str(amount))
    except (InvalidOperation, ValueError) as exc:
        raise MarketEpisodeError(
            f"{field}.amount must be a non-negative USD amount"
        ) from exc
    minor = major * Decimal(100)
    if (
        not minor.is_finite()
        or minor < 0
        or minor != minor.to_integral_value()
    ):
        raise MarketEpisodeError(
            f"{field}.amount must have at most two decimal places and be non-negative"
        )
    return int(minor)


def _file_digest(path: Path) -> str:
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise MarketEpisodeError(f"cannot hash Episode artifact {path}: {exc}") from exc
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


__all__ = [
    "MARKET_ORACLE_SCHEMA",
    "MarketEpisodeError",
    "build_market_artifact_from_episode",
    "market_metrics_for_result",
]
