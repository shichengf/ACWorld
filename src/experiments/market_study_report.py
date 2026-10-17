"""Auditable reporting for bounded many-to-many environment case studies.

The core :mod:`experiments.market_artifacts` artifact intentionally contains
only metrics that can be reconstructed from authoritative World state and its
hidden market oracle.  A paper case study often wants several additional
diagnostics -- for example efficiency relative to the *shown* top-k graph,
contention, fallback, and independent replay.  Those diagnostics require
evidence that the market artifact does not itself preserve.

This module keeps that boundary explicit.  It first verifies the market
artifact, then derives extra diagnostics only when complete structured sidecar
evidence is supplied.  Missing evidence is represented as an unavailable row
with a reason; it is never converted to a zero or a successful check.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.market_metrics import (
    BuyerValuation,
    MerchantFloor,
    compute_market_metrics,
)
from evals.serialize import to_canonical
from experiments.market_artifacts import MarketArtifactError, verify_market_artifact


MARKET_STUDY_REPORT_SCHEMA = "cwe.market-study-report.v1"
_PRECISION = 28


class MarketStudyReportError(ValueError):
    """A market study report or its structured sidecar evidence is invalid."""


def build_market_study_report(
    artifact: Mapping[str, Any],
    *,
    top_k: int | None = None,
    ranked_candidates: Mapping[str, Sequence[str]] | None = None,
    attempt_sequences: Mapping[str, Sequence[str]] | None = None,
    replay_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic report from a verified market artifact.

    ``ranked_candidates`` and ``attempt_sequences`` must cover the exact buyer
    population when supplied.  Candidate rows are cross-checked against the
    artifact's listing-resolved exposure stream.  Consequently, top-k
    reachability cannot be claimed from a prompt, a final answer, or an
    incomplete logging stream.

    ``replay_evidence`` accepts the stable fields returned by
    :class:`episode.replay.ReplayVerificationResult`.  Local paths are omitted
    from the persisted report.
    """

    try:
        verify_market_artifact(artifact)
    except MarketArtifactError as exc:
        raise MarketStudyReportError(f"invalid source market artifact: {exc}") from exc

    inputs = _mapping(artifact.get("inputs"), "artifact.inputs")
    population = _mapping(artifact.get("population"), "artifact.population")
    metrics = _mapping(artifact.get("metrics"), "artifact.metrics")
    execution = _mapping(artifact.get("execution"), "artifact.execution")
    buyer_ids = _string_sequence(population.get("buyer_ids"), "population.buyer_ids")
    merchant_ids = _string_sequence(population.get("merchant_ids"), "population.merchant_ids")
    valuations = _valuations(inputs)
    floors = _floors(inputs)
    transactions = _mapping_rows(inputs.get("transactions"), "inputs.transactions")
    exposures = _mapping_rows(inputs.get("exposures"), "inputs.exposures")
    listing_owner, listing_capacity = _listing_indexes(floors)

    candidates = _normalize_actor_sequences(
        ranked_candidates,
        actor_ids=buyer_ids,
        known_listings=frozenset(listing_owner),
        field="ranked_candidates",
        unique=True,
    )
    if candidates is None:
        if top_k is not None:
            raise MarketStudyReportError("top_k requires complete ranked_candidates evidence")
    else:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise MarketStudyReportError(
                "complete ranked_candidates evidence requires a positive top_k"
            )
        for buyer_id, listing_ids in candidates.items():
            if len(listing_ids) > top_k:
                raise MarketStudyReportError(
                    f"ranked_candidates[{buyer_id!r}] exceeds top_k={top_k}"
                )
        _validate_candidates_against_exposures(candidates, exposures)

    attempts = _normalize_actor_sequences(
        attempt_sequences,
        actor_ids=buyer_ids,
        known_listings=frozenset(listing_owner),
        field="attempt_sequences",
        unique=False,
    )
    if attempts is not None:
        _validate_attempts(
            attempts,
            transactions=transactions,
            ranked_candidates=candidates,
        )

    normalized_replay = _normalize_replay_evidence(replay_evidence, execution=execution)
    evidence = {
        "top_k": top_k,
        "ranked_candidates": candidates,
        "attempt_sequences": attempts,
        "replay": normalized_replay,
    }
    report: dict[str, Any] = {
        "schema_version": MARKET_STUDY_REPORT_SCHEMA,
        "source": {
            "market_identity": _market_identity(artifact),
            "market_schema_version": artifact.get("schema_version"),
            "market_artifact_digest": _digest(artifact),
            "input_digest": artifact.get("input_digest"),
            "execution_digest": artifact.get("execution_digest"),
        },
        "scope": {
            "buyers": len(buyer_ids),
            "merchants": len(merchant_ids),
            "buyer_ids": list(buyer_ids),
            "merchant_ids": list(merchant_ids),
            "variant_id": artifact.get("variant_id"),
            "task_family": artifact.get("task_family"),
            "model_id": execution.get("model_id"),
        },
        "evidence": to_canonical(evidence),
        "outcomes": {
            "trades_and_welfare": _trade_and_welfare(transactions, metrics),
            "global_oracle_efficiency": _global_efficiency(metrics),
            "top_k_reachable_efficiency": _top_k_efficiency(
                candidates=candidates,
                valuations=valuations,
                floors=floors,
                transactions=transactions,
                metrics=metrics,
                buyer_ids=buyer_ids,
                merchant_ids=merchant_ids,
            ),
            "exposure_and_concentration": _exposure_report(
                exposures,
                merchant_ids=merchant_ids,
                metrics=metrics,
            ),
            "contention": _contention_report(
                attempts=attempts,
                candidates=candidates,
                buyer_ids=buyer_ids,
            ),
            "fallback": _fallback_report(
                attempts=attempts,
                transactions=transactions,
            ),
            "oversell_and_inventory": _oversell_report(
                transactions,
                listing_capacity=listing_capacity,
                metrics=metrics,
            ),
            "privacy": _observation_metric(
                metrics,
                "privacy_leakage_rate",
                stream_name="privacy event stream",
            ),
            "protocol": _observation_metric(
                metrics,
                "protocol_violation_rate",
                stream_name="protocol event stream",
            ),
            "replay": _replay_report(normalized_replay),
        },
        "evaluation": {
            "judge": "deterministic_python",
            "llm_judge": False,
            "source_market_artifact_verified": True,
            "unavailable_is_not_zero": True,
            "contention_boundary": (
                "attempt overlap when complete attempt evidence exists; otherwise "
                "candidate overlap is labeled as a proxy"
            ),
            "privacy_protocol_boundary": (
                "blocked attempts are not relabeled as observed non-leaks/non-violations"
            ),
        },
    }
    report = to_canonical(report)
    if not isinstance(report, dict):  # pragma: no cover - dict construction guard
        raise MarketStudyReportError("market study report did not normalize to an object")
    report["report_digest"] = _digest(report)
    return report


def verify_market_study_report(
    report: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any],
) -> None:
    """Rebuild a report from its authenticated evidence and reject any drift."""

    if report.get("schema_version") != MARKET_STUDY_REPORT_SCHEMA:
        raise MarketStudyReportError(
            f"unsupported market study report schema: {report.get('schema_version')!r}"
        )
    evidence = _mapping(report.get("evidence"), "report.evidence")
    rebuilt = build_market_study_report(
        artifact,
        top_k=evidence.get("top_k"),
        ranked_candidates=_optional_sequence_mapping(
            evidence.get("ranked_candidates"), "report.evidence.ranked_candidates"
        ),
        attempt_sequences=_optional_sequence_mapping(
            evidence.get("attempt_sequences"), "report.evidence.attempt_sequences"
        ),
        replay_evidence=_optional_mapping(evidence.get("replay"), "report.evidence.replay"),
    )
    if dict(report) != rebuilt:
        raise MarketStudyReportError("market study report does not match deterministic rebuild")


def write_market_study_report(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    artifact: Mapping[str, Any],
) -> Path:
    """Verify and persist a canonical market-study JSON report."""

    verify_market_study_report(report, artifact=artifact)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_market_study_report(
    path: str | Path,
    *,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and deterministically verify a persisted report."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MarketStudyReportError(f"cannot read market study report {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MarketStudyReportError("market study report root must be an object")
    verify_market_study_report(raw, artifact=artifact)
    return raw


def _trade_and_welfare(
    transactions: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    buyers = {str(row["buyer_id"]) for row in transactions}
    merchants = {str(row["merchant_id"]) for row in transactions}
    return {
        "transaction_count": len(transactions),
        "trading_buyers": len(buyers),
        "trading_merchants": len(merchants),
        "trade_rate": _metric(metrics, "trade_rate"),
        "consumer_surplus_minor": _metric(metrics, "consumer_surplus")["value"],
        "producer_surplus_minor": _metric(metrics, "producer_surplus")["value"],
        "social_welfare_minor": _metric(metrics, "social_welfare")["value"],
    }


def _global_efficiency(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = _metric(metrics, "allocative_efficiency")
    return {
        "applicable": row["applicable"],
        "value": row["value"],
        "realized_welfare_minor": _metric(metrics, "social_welfare")["value"],
        "optimal_welfare_minor": row["denominator"],
        "reason": row["reason"],
        "oracle_scope": "all valuation edges in the hidden market oracle",
    }


def _top_k_efficiency(
    *,
    candidates: dict[str, list[str]] | None,
    valuations: Sequence[BuyerValuation],
    floors: Sequence[MerchantFloor],
    transactions: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    buyer_ids: Sequence[str],
    merchant_ids: Sequence[str],
) -> dict[str, Any]:
    if candidates is None:
        return _unavailable(
            "complete ranked candidate sets and listing-resolved exposures were not supplied"
        )
    transaction_counts = Counter(str(row["buyer_id"]) for row in transactions)
    if any(count > 1 for count in transaction_counts.values()) or any(
        _int(row.get("quantity"), "transaction.quantity", minimum=1) != 1 for row in transactions
    ):
        return _unavailable("top-k reachable oracle is defined for this unit-demand study only")

    valuation_index = {(row.buyer_id, row.listing_id): row for row in valuations}
    restricted: list[BuyerValuation] = []
    for buyer_id, listing_ids in candidates.items():
        for listing_id in listing_ids:
            row = valuation_index.get((buyer_id, listing_id))
            if row is None:
                raise MarketStudyReportError(
                    f"ranked candidate {listing_id!r} for {buyer_id!r} has no valuation edge"
                )
            restricted.append(row)

    probe = compute_market_metrics(
        valuations=restricted,
        merchant_floors=floors,
        buyer_ids=buyer_ids,
        merchant_ids=merchant_ids,
    ).allocative_efficiency
    if probe.denominator is None:
        optimal = 0 if probe.reason == "optimal social welfare is zero" else None
    else:
        optimal = int(probe.denominator)
    if optimal is None:
        return {
            **_unavailable(probe.reason or "candidate-restricted oracle unavailable"),
            "candidate_edges": len(restricted),
            "oracle_scope": "complete logged top-k candidate graph",
        }

    realized = _int(
        _metric(metrics, "social_welfare")["value"],
        "metrics.social_welfare.value",
        minimum=None,
    )
    oversell = _has_oversell(transactions, floors)
    if oversell:
        return {
            **_unavailable("realized transactions exceed candidate-restricted capacity"),
            "candidate_edges": len(restricted),
            "realized_welfare_minor": realized,
            "optimal_welfare_minor": optimal,
            "oracle_scope": "complete logged top-k candidate graph",
        }
    if realized > optimal:
        raise MarketStudyReportError(
            "realized welfare exceeds the exact candidate-restricted oracle"
        )
    if optimal == 0:
        return {
            **_unavailable("candidate-restricted optimal social welfare is zero"),
            "candidate_edges": len(restricted),
            "realized_welfare_minor": realized,
            "optimal_welfare_minor": 0,
            "oracle_scope": "complete logged top-k candidate graph",
        }
    return {
        "applicable": True,
        "value": _ratio_string(max(realized, 0), optimal),
        "numerator": max(realized, 0),
        "denominator": optimal,
        "reason": None,
        "candidate_edges": len(restricted),
        "realized_welfare_minor": realized,
        "optimal_welfare_minor": optimal,
        "oracle_scope": "complete logged top-k candidate graph",
    }


def _exposure_report(
    exposures: Sequence[Mapping[str, Any]],
    *,
    merchant_ids: Sequence[str],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    by_merchant = {merchant_id: Decimal(0) for merchant_id in merchant_ids}
    for index, row in enumerate(exposures):
        merchant_id = str(row.get("merchant_id", ""))
        if merchant_id not in by_merchant:
            raise MarketStudyReportError(
                f"inputs.exposures[{index}] references unknown merchant {merchant_id!r}"
            )
        by_merchant[merchant_id] += _decimal(
            row.get("weight", "1"), f"inputs.exposures[{index}].weight"
        )
    total = sum(by_merchant.values(), Decimal(0))
    base = {
        "exposure_events": len(exposures),
        "exposure_weight_by_merchant": {
            merchant_id: str(value) for merchant_id, value in sorted(by_merchant.items())
        },
        "total_exposure_weight": str(total),
        "jain_fairness": _metric(metrics, "exposure_fairness"),
    }
    if total == 0:
        return {
            **base,
            "concentration": _unavailable("total exposure weight is zero"),
        }
    with localcontext() as context:
        context.prec = _PRECISION
        context.rounding = ROUND_HALF_EVEN
        shares = {merchant_id: value / total for merchant_id, value in by_merchant.items()}
        hhi = sum((share * share for share in shares.values()), Decimal(0))
        effective = Decimal(1) / hhi
    return {
        **base,
        "concentration": {
            "applicable": True,
            "hhi": str(hhi),
            "top_merchant_share": str(max(shares.values())),
            "effective_merchants": str(effective),
            "reason": None,
        },
    }


def _contention_report(
    *,
    attempts: dict[str, list[str]] | None,
    candidates: dict[str, list[str]] | None,
    buyer_ids: Sequence[str],
) -> dict[str, Any]:
    if attempts is not None:
        edges = attempts
        method = "observed_attempt_overlap"
        proxy = False
    elif candidates is not None:
        edges = candidates
        method = "candidate_overlap_proxy"
        proxy = True
    else:
        return _unavailable("neither complete attempt nor ranked-candidate evidence was supplied")

    buyers_by_listing: dict[str, set[str]] = defaultdict(set)
    for buyer_id, listing_ids in edges.items():
        for listing_id in set(listing_ids):
            buyers_by_listing[listing_id].add(buyer_id)
    contested = {
        listing_id: len(buyers)
        for listing_id, buyers in buyers_by_listing.items()
        if len(buyers) >= 2
    }
    buyers_in_contention = {
        buyer_id for listing_id in contested for buyer_id in buyers_by_listing[listing_id]
    }

    first_choice_buyers: dict[str, set[str]] = defaultdict(set)
    for buyer_id, listing_ids in edges.items():
        if listing_ids:
            first_choice_buyers[listing_ids[0]].add(buyer_id)
    first_choice_contested = {
        listing_id: len(buyers)
        for listing_id, buyers in first_choice_buyers.items()
        if len(buyers) >= 2
    }
    buyers_in_first_choice_contention = {
        buyer_id
        for listing_id in first_choice_contested
        for buyer_id in first_choice_buyers[listing_id]
    }
    return {
        "applicable": True,
        "method": method,
        "is_proxy": proxy,
        "reason": (
            "candidate overlap is opportunity-set overlap, not evidence of an attempted trade"
            if proxy
            else None
        ),
        "contested_listing_count": len(contested),
        "contested_listings": dict(sorted(contested.items())),
        "buyers_in_contention": len(buyers_in_contention),
        "buyer_contention_rate": _ratio_string(len(buyers_in_contention), len(buyer_ids)),
        "first_choice_contested_listing_count": len(first_choice_contested),
        "first_choice_contested_listings": dict(sorted(first_choice_contested.items())),
        "buyers_in_first_choice_contention": len(buyers_in_first_choice_contention),
        "first_choice_contention_rate": _ratio_string(
            len(buyers_in_first_choice_contention), len(buyer_ids)
        ),
    }


def _fallback_report(
    *,
    attempts: dict[str, list[str]] | None,
    transactions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if attempts is None:
        return _unavailable("complete ordered clearing-attempt sequences were not supplied")
    transactions_by_buyer: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in transactions:
        transactions_by_buyer[str(row["buyer_id"])].append(row)
    if any(len(rows) > 1 for rows in transactions_by_buyer.values()):
        return _unavailable("fallback diagnostic is defined for at most one trade per buyer")

    fallback_buyers: list[str] = []
    for buyer_id, rows in sorted(transactions_by_buyer.items()):
        settled_listing = str(rows[0]["listing_id"])
        sequence = attempts[buyer_id]
        settled_index = sequence.index(settled_listing)
        if any(item != settled_listing for item in sequence[:settled_index]):
            fallback_buyers.append(buyer_id)
    multi_attempt_buyers = [
        buyer_id for buyer_id, sequence in attempts.items() if len(dict.fromkeys(sequence)) >= 2
    ]
    trading_buyers = len(transactions_by_buyer)
    rate: str | None = None
    reason: str | None = None
    if trading_buyers:
        rate = _ratio_string(len(fallback_buyers), trading_buyers)
    else:
        reason = "no transacting buyers"
    return {
        "applicable": rate is not None,
        "fallback_trade_rate": rate,
        "fallback_trade_count": len(fallback_buyers),
        "fallback_buyer_ids": fallback_buyers,
        "buyers_with_multiple_distinct_attempts": len(multi_attempt_buyers),
        "reason": reason,
    }


def _oversell_report(
    transactions: Sequence[Mapping[str, Any]],
    *,
    listing_capacity: Mapping[str, int],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    sold: Counter[str] = Counter()
    for row in transactions:
        sold[str(row["listing_id"])] += _int(row.get("quantity"), "transaction.quantity", minimum=1)
    violations = [
        {
            "listing_id": listing_id,
            "sold_quantity": sold_quantity,
            "capacity": listing_capacity[listing_id],
        }
        for listing_id, sold_quantity in sorted(sold.items())
        if sold_quantity > listing_capacity[listing_id]
    ]
    return {
        "oversell_check_applicable": True,
        "oversell_detected": bool(violations),
        "violated_listing_count": len(violations),
        "violations": violations,
        "sold_quantity_by_listing": dict(sorted(sold.items())),
        "inventory_correctness": _metric(metrics, "inventory_correctness"),
    }


def _observation_metric(
    metrics: Mapping[str, Any],
    metric_name: str,
    *,
    stream_name: str,
) -> dict[str, Any]:
    row = _metric(metrics, metric_name)
    if not row["applicable"]:
        return {
            "applicable": False,
            "value": None,
            "observed_events": 0,
            "flagged_events": None,
            "reason": row["reason"] or f"{stream_name} is unavailable",
            "interpretation": "unavailable; not zero",
        }
    return {
        "applicable": True,
        "value": row["value"],
        "observed_events": row["denominator"],
        "flagged_events": row["numerator"],
        "reason": None,
        "interpretation": "deterministic structured observation stream",
    }


def _replay_report(replay: Mapping[str, Any] | None) -> dict[str, Any]:
    if replay is None:
        return _unavailable("no independent replay-verifier evidence was supplied")
    strict = bool(replay["strict"])
    replay_ok = bool(replay["replay_ok"])
    digests_equal = replay["expected_state_digest"] == replay["replay_state_digest"]
    supported = strict and replay_ok and digests_equal
    reason: str | None = None
    if not strict:
        reason = "replay evidence is not from a strict verifier"
    elif not replay_ok:
        reason = "strict replay verifier reported failure"
    elif not digests_equal:  # guarded during normalization; defensive
        reason = "expected and replay state digests differ"
    return {
        "applicable": True,
        "strict": strict,
        "replay_ok": replay_ok,
        "state_digests_equal": digests_equal,
        "claim_supported": supported,
        "reason": reason,
        "source_kind": replay["source_kind"],
        "schema_version": replay.get("schema_version"),
        "events_verified": replay.get("events_verified"),
        "transactions_replayed": replay.get("transactions_replayed"),
        "expected_state_digest": replay["expected_state_digest"],
        "replay_state_digest": replay["replay_state_digest"],
    }


def _normalize_replay_evidence(
    raw: Mapping[str, Any] | None,
    *,
    execution: Mapping[str, Any],
) -> dict[str, Any] | None:
    if raw is None:
        if not all(
            key in execution for key in ("replay_ok", "state_digest", "replay_state_digest")
        ):
            return None
        raw = {
            "source_kind": "execution_metadata",
            "replay_ok": execution["replay_ok"],
            "strict": False,
            "expected_state_digest": execution["state_digest"],
            "replay_state_digest": execution["replay_state_digest"],
        }
    replay_ok = raw.get("replay_ok")
    strict = raw.get("strict")
    if not isinstance(replay_ok, bool) or not isinstance(strict, bool):
        raise MarketStudyReportError("replay evidence requires boolean replay_ok and strict")
    expected = _nonempty_string(raw.get("expected_state_digest"), "replay.expected_state_digest")
    replayed = _nonempty_string(raw.get("replay_state_digest"), "replay.replay_state_digest")
    if replay_ok and expected != replayed:
        raise MarketStudyReportError(
            "replay evidence cannot set replay_ok=true when state digests differ"
        )
    source_kind = raw.get("source_kind", raw.get("kind"))
    source_kind = _nonempty_string(source_kind, "replay.source_kind")
    events = _optional_int(raw.get("events_verified"), "replay.events_verified")
    transactions = _optional_int(raw.get("transactions_replayed"), "replay.transactions_replayed")
    schema = raw.get("schema_version")
    if schema is not None:
        schema = _nonempty_string(schema, "replay.schema_version")
    return {
        "source_kind": source_kind,
        "schema_version": schema,
        "strict": strict,
        "replay_ok": replay_ok,
        "events_verified": events,
        "transactions_replayed": transactions,
        "expected_state_digest": expected,
        "replay_state_digest": replayed,
    }


def _normalize_actor_sequences(
    raw: Mapping[str, Sequence[str]] | None,
    *,
    actor_ids: Sequence[str],
    known_listings: frozenset[str],
    field: str,
    unique: bool,
) -> dict[str, list[str]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise MarketStudyReportError(f"{field} must be an object or null")
    expected = set(actor_ids)
    actual = {str(key) for key in raw}
    if actual != expected:
        raise MarketStudyReportError(
            f"{field} must cover the exact buyer population: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    result: dict[str, list[str]] = {}
    for actor_id in actor_ids:
        listing_ids = _string_sequence(raw[actor_id], f"{field}[{actor_id!r}]")
        if unique and len(listing_ids) != len(set(listing_ids)):
            raise MarketStudyReportError(f"{field}[{actor_id!r}] contains duplicates")
        unknown = sorted(set(listing_ids) - known_listings)
        if unknown:
            raise MarketStudyReportError(
                f"{field}[{actor_id!r}] references unknown listings {unknown}"
            )
        result[actor_id] = list(listing_ids)
    return result


def _validate_candidates_against_exposures(
    candidates: Mapping[str, Sequence[str]],
    exposures: Sequence[Mapping[str, Any]],
) -> None:
    exposed: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(exposures):
        buyer_id = str(row.get("buyer_id", ""))
        listing_id = row.get("listing_id")
        if listing_id is None:
            raise MarketStudyReportError(
                "ranked candidates require listing-resolved exposure evidence; "
                f"inputs.exposures[{index}].listing_id is null"
            )
        exposed[buyer_id].add(str(listing_id))
    for buyer_id, listing_ids in candidates.items():
        if exposed.get(buyer_id, set()) != set(listing_ids):
            raise MarketStudyReportError(
                f"ranked_candidates[{buyer_id!r}] does not match artifact exposures"
            )
    unexpected_buyers = sorted(set(exposed) - set(candidates))
    if unexpected_buyers:
        raise MarketStudyReportError(
            f"artifact exposures contain buyers outside candidate evidence: {unexpected_buyers}"
        )


def _validate_attempts(
    attempts: Mapping[str, Sequence[str]],
    *,
    transactions: Sequence[Mapping[str, Any]],
    ranked_candidates: Mapping[str, Sequence[str]] | None,
) -> None:
    for buyer_id, listing_ids in attempts.items():
        if ranked_candidates is not None:
            outside = sorted(set(listing_ids) - set(ranked_candidates[buyer_id]))
            if outside:
                raise MarketStudyReportError(
                    f"attempt_sequences[{buyer_id!r}] contains listings outside its top-k set: "
                    f"{outside}"
                )
    for row in transactions:
        buyer_id = str(row["buyer_id"])
        listing_id = str(row["listing_id"])
        if listing_id not in attempts[buyer_id]:
            raise MarketStudyReportError(
                f"settled listing {listing_id!r} is absent from {buyer_id!r}'s attempt sequence"
            )


def _valuations(inputs: Mapping[str, Any]) -> tuple[BuyerValuation, ...]:
    rows = _mapping_rows(inputs.get("valuations"), "inputs.valuations")
    try:
        return tuple(BuyerValuation(**row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise MarketStudyReportError(f"invalid valuation inputs: {exc}") from exc


def _floors(inputs: Mapping[str, Any]) -> tuple[MerchantFloor, ...]:
    rows = _mapping_rows(inputs.get("merchant_floors"), "inputs.merchant_floors")
    try:
        return tuple(MerchantFloor(**row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise MarketStudyReportError(f"invalid floor inputs: {exc}") from exc


def _listing_indexes(
    floors: Sequence[MerchantFloor],
) -> tuple[dict[str, str], dict[str, int]]:
    owner: dict[str, str] = {}
    capacity: dict[str, int] = {}
    for row in floors:
        if row.listing_id in owner:
            raise MarketStudyReportError(
                "market-study candidate evidence requires globally unique listing_id values"
            )
        owner[row.listing_id] = row.merchant_id
        capacity[row.listing_id] = row.capacity
    return owner, capacity


def _has_oversell(
    transactions: Sequence[Mapping[str, Any]],
    floors: Sequence[MerchantFloor],
) -> bool:
    capacity = {row.listing_id: row.capacity for row in floors}
    sold: Counter[str] = Counter()
    for row in transactions:
        sold[str(row["listing_id"])] += _int(row.get("quantity"), "transaction.quantity", minimum=1)
    return any(quantity > capacity[listing_id] for listing_id, quantity in sold.items())


def _metric(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    row = metrics.get(name)
    if not isinstance(row, Mapping):
        raise MarketStudyReportError(f"artifact metric {name!r} is missing")
    return dict(row)


def _market_identity(artifact: Mapping[str, Any]) -> str:
    raw = artifact.get("market_id", artifact.get("run_key"))
    return _nonempty_string(raw, "artifact market identity")


def _mapping(raw: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise MarketStudyReportError(f"{field} must be an object")
    return raw


def _mapping_rows(raw: Any, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise MarketStudyReportError(f"{field} must be a list of objects")
    return tuple(raw)


def _string_sequence(raw: Any, field: str) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in raw
    ):
        raise MarketStudyReportError(f"{field} must be a sequence of non-empty strings")
    return tuple(raw)


def _optional_sequence_mapping(
    raw: Any,
    field: str,
) -> Mapping[str, Sequence[str]] | None:
    if raw is None:
        return None
    mapping = _mapping(raw, field)
    return {str(key): _string_sequence(value, f"{field}.{key}") for key, value in mapping.items()}


def _optional_mapping(raw: Any, field: str) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    return _mapping(raw, field)


def _nonempty_string(raw: Any, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise MarketStudyReportError(f"{field} must be a non-empty string")
    return raw


def _int(raw: Any, field: str, *, minimum: int | None = 0) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise MarketStudyReportError(f"{field} must be an integer")
    if minimum is not None and raw < minimum:
        raise MarketStudyReportError(f"{field} must be >= {minimum}")
    return int(raw)


def _optional_int(raw: Any, field: str) -> int | None:
    if raw is None:
        return None
    return _int(raw, field)


def _decimal(raw: Any, field: str) -> Decimal:
    if isinstance(raw, bool):
        raise MarketStudyReportError(f"{field} must be a non-negative decimal")
    try:
        value = Decimal(str(raw))
    except Exception as exc:
        raise MarketStudyReportError(f"{field} must be a non-negative decimal") from exc
    if not value.is_finite() or value < 0:
        raise MarketStudyReportError(f"{field} must be a non-negative decimal")
    return value


def _ratio_string(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise MarketStudyReportError("ratio denominator must be positive")
    with localcontext() as context:
        context.prec = _PRECISION
        context.rounding = ROUND_HALF_EVEN
        return str(Decimal(numerator) / Decimal(denominator))


def _unavailable(reason: str) -> dict[str, Any]:
    return {"applicable": False, "value": None, "reason": reason}


def _digest(value: Mapping[str, Any]) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


__all__ = [
    "MARKET_STUDY_REPORT_SCHEMA",
    "MarketStudyReportError",
    "build_market_study_report",
    "load_market_study_report",
    "verify_market_study_report",
    "write_market_study_report",
]
