"""Task-agnostic completion of exact operation evidence.

Benchmark rubrics intentionally score only the capability named by a task.
An agent can nevertheless submit another valid CommerceWorld operation.  The
operation must remain a model mistake for that rubric, but its authoritative
World commit still needs an exact Runtime, Platform, and World join.

This module completes those joins without granting rubric credit.  It runs
after a task scorer and invokes only registered operation evidence contracts.
The caller subsequently proves that every World commit was claimed exactly
once, so this module cannot turn an unverified commit into evidence.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterable, Mapping, Protocol

from runtime.after_sales_evidence import (
    ACTION_TO_OPERATION,
    AFTER_SALES_EVIDENCE_CONTRACT,
    VerifiedAfterSalesEvidence,
)
from runtime.authority_operation_evidence import (
    EVIDENCE_RECORD_EVIDENCE_CONTRACT,
    LISTING_CLAIM_EVIDENCE_CONTRACT,
    MARKET_CLOCK_EVIDENCE_CONTRACT,
    SEARCH_SESSION_EVIDENCE_CONTRACT,
    VerifiedEvidenceRecordEvidence,
    VerifiedListingClaimEvidence,
    VerifiedMarketClockEvidence,
    VerifiedSearchSessionEvidence,
)
from runtime.cart_evidence import (
    CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT,
    CART_EVIDENCE_CONTRACT,
    CART_QUOTE_PREFIX_EVIDENCE_CONTRACT,
    VerifiedCartAuthorizationPrefixEvidence,
    VerifiedCartEvidence,
    VerifiedCartQuotePrefixEvidence,
)
from runtime.exact_join import (
    CATALOG_MUTATION_EVIDENCE_CONTRACT,
    PROTOCOL_EVENT_EVIDENCE_CONTRACT,
    VerifiedCatalogMutationEvidence,
    VerifiedProtocolEventEvidence,
)
from runtime.market_governance_evidence import (
    ACTOR_ACTIONS,
    MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
    TRUSTED_ACTIONS,
    VerifiedMarketGovernanceEvidence,
)
from runtime.match_certificate_evidence import (
    MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
    VerifiedMatchCertificateEvidence,
)
from runtime.negotiation_evidence import (
    NEGOTIATION_EVIDENCE_CONTRACT,
    VerifiedNegotiationEvidence,
)
from runtime.supply_fulfillment_evidence import (
    SUPPLY_FULFILLMENT_ACTION_KINDS,
    SUPPLY_FULFILLMENT_AUTHORITY_PAIRS,
    SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
    VerifiedSupplyFulfillmentEvidence,
    governance_preclaimed_commit_attestations,
)


_CART_AUTHORITY_PAIRS = frozenset(
    {
        ("create_cart_quote_request", "world.create_cart_quote_request"),
        ("issue_cart_quote", "world.issue_cart_quote"),
        ("checkout_cart_quote", "world.checkout_cart_quote"),
    }
)
_AFTER_SALES_OPERATIONS = frozenset(
    {
        *ACTION_TO_OPERATION.values(),
        "rule_for_filer",
        "rule_for_respondent",
        "rule_split",
        "complete_ledger_reconciliation",
    }
)
_GOVERNANCE_OPERATIONS = frozenset(
    {
        *(row[0] for row in ACTOR_ACTIONS.values()),
        *(row[0] for row in TRUSTED_ACTIONS.values()),
        "publish_governance_policy",
    }
)
_PROTOCOL_EVENT_OPERATIONS = frozenset(
    {
        "publish_protocol_event",
        "process_protocol_event",
        "append_protocol_receipt",
    }
)


class OperationEvidenceSource(Protocol):
    """Minimal evidence surface needed by the task-agnostic completer."""

    world_events: tuple[dict[str, Any], ...]
    platform_decisions: tuple[dict[str, Any], ...]

    def verified_operation_evidence(
        self,
        contract_id: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> object: ...


def complete_unclaimed_operation_evidence(
    evidence: OperationEvidenceSource,
    authority_calls: Iterable[tuple[str, str, object]],
) -> None:
    """Invoke exact contracts for valid commits omitted by a task rubric.

    The order is deliberate.  A Match contract owns its source Search commit,
    so Match runs before standalone Search.  Negotiation commits are disjoint
    from discovery commits and can then be verified as one policy-bound graph.
    """

    calls = tuple(authority_calls)
    claimed = _claimed_commit_ids(calls)
    world_commits = tuple(evidence.world_events)

    _complete_match(evidence, world_commits=world_commits, claimed=claimed)
    _complete_search(evidence, world_commits=world_commits, claimed=claimed)
    _complete_negotiation(evidence, world_commits=world_commits, claimed=claimed)
    _complete_after_sales(evidence, world_commits=world_commits, claimed=claimed)
    verified_governance = _governance_result_from_calls(calls)
    completed_governance = _complete_market_governance(
        evidence,
        world_commits=world_commits,
        claimed=claimed,
    )
    if completed_governance is not None:
        verified_governance = completed_governance
    _complete_protocol_events(
        evidence,
        world_commits=world_commits,
        claimed=claimed,
    )
    _complete_catalog_mutations(
        evidence,
        world_commits=world_commits,
        claimed=claimed,
    )
    _complete_listing_claims(
        evidence,
        world_commits=world_commits,
        claimed=claimed,
    )
    _complete_evidence_records(
        evidence,
        world_commits=world_commits,
        claimed=claimed,
    )
    _complete_cart(evidence, world_commits=world_commits, claimed=claimed)
    _complete_market_clock(evidence, world_commits=world_commits, claimed=claimed)
    _complete_supply_fulfillment(
        evidence,
        world_commits=world_commits,
        claimed=claimed,
        verified_governance=verified_governance,
    )


def _governance_result_from_calls(
    calls: Iterable[tuple[str, str, object]],
) -> VerifiedMarketGovernanceEvidence | None:
    results = tuple(
        result
        for contract_id, _options_hash, result in calls
        if contract_id == MARKET_GOVERNANCE_EVIDENCE_CONTRACT
        and isinstance(result, VerifiedMarketGovernanceEvidence)
    )
    if not results:
        return None
    selected = max(results, key=lambda row: len(_commit_ids(row)))
    selected_ids = set(_commit_ids(selected))
    if any(not set(_commit_ids(row)).issubset(selected_ids) for row in results):
        raise ValueError("recorded governance exact results disagree on claimed commits")
    return selected


def _complete_match(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    matching = _commits_for_operation(
        world_commits,
        operation="issue_match_certificate",
        authority_action="world.issue_match_certificate",
    )
    unclaimed = tuple(row for row in matching if _commit_id(row) not in claimed)
    if not unclaimed:
        return
    already_claimed = tuple(row for row in matching if _commit_id(row) in claimed)
    if already_claimed:
        raise ValueError("Match evidence is only partially claimed before task-agnostic completion")
    candidate = evidence.verified_operation_evidence(
        MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
        options={"allow_rejected": True},
    )
    if not isinstance(candidate, VerifiedMatchCertificateEvidence):
        raise TypeError("match certificate evidence contract returned wrong type")
    claimed.update(_commit_ids(candidate))


def _complete_search(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    searches = _commits_for_operation(
        world_commits,
        operation="create_search_session",
        authority_action="world.create_search_session",
    )
    if not any(_commit_id(row) not in claimed for row in searches):
        return
    excluded_session_ids = tuple(
        str(row.get("subject_id")) for row in searches if _commit_id(row) in claimed
    )
    candidate = evidence.verified_operation_evidence(
        SEARCH_SESSION_EVIDENCE_CONTRACT,
        options={"exclude_session_ids": excluded_session_ids},
    )
    if not isinstance(candidate, VerifiedSearchSessionEvidence):
        raise TypeError("search session evidence contract returned wrong type")
    claimed.update(_commit_ids(candidate))


def _complete_negotiation(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    negotiations = _commits_for_operation(
        world_commits,
        operation="negotiation_event",
        authority_action="world.apply_negotiation_intent",
    )
    unclaimed = tuple(row for row in negotiations if _commit_id(row) not in claimed)
    if not unclaimed:
        return
    already_claimed = tuple(row for row in negotiations if _commit_id(row) in claimed)
    if already_claimed:
        raise ValueError(
            "Negotiation evidence is only partially claimed before task-agnostic completion"
        )

    policy_rows = tuple(_negotiation_event_row(row) for row in negotiations)
    negotiation_ids = tuple(sorted({str(row["negotiation_id"]) for row in policy_rows}))
    max_rounds = {int(row["max_rounds"]) for row in policy_rows}
    deadline_ticks = {
        int(row["expires_at_tick"]) - int(row["opened_at_tick"]) for row in policy_rows
    }
    if len(max_rounds) != 1 or len(deadline_ticks) != 1:
        raise ValueError("Negotiation commits do not share one Platform policy binding")
    selected_max_rounds = next(iter(max_rounds))
    selected_deadline_ticks = next(iter(deadline_ticks))
    if selected_max_rounds <= 0 or selected_deadline_ticks <= 0:
        raise ValueError("Negotiation commits have an invalid Platform policy binding")

    candidate = evidence.verified_operation_evidence(
        NEGOTIATION_EVIDENCE_CONTRACT,
        options={
            "expected_negotiation_ids": negotiation_ids,
            "max_rounds": selected_max_rounds,
            "deadline_ticks": selected_deadline_ticks,
        },
    )
    if not isinstance(candidate, VerifiedNegotiationEvidence):
        raise TypeError("negotiation evidence contract returned wrong type")
    claimed.update(_commit_ids(candidate))


def _complete_supply_fulfillment(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
    verified_governance: VerifiedMarketGovernanceEvidence | None,
) -> None:
    transactions = tuple(row for row in world_commits if row.get("commit_kind") == "transaction")
    unclaimed = tuple(row for row in transactions if _commit_id(row) not in claimed)
    if not unclaimed:
        return
    if any(
        (row.get("operation"), row.get("authority_action"))
        not in SUPPLY_FULFILLMENT_AUTHORITY_PAIRS
        for row in unclaimed
    ):
        return
    claimed_transaction_ids = tuple(
        _commit_id(row) for row in transactions if _commit_id(row) in claimed
    )

    actors = tuple(
        str(row.get("actor_id"))
        for row in evidence.platform_decisions
        if row.get("decision") == "accepted"
        and row.get("action_kind") in SUPPLY_FULFILLMENT_ACTION_KINDS
        and isinstance(row.get("actor_id"), str)
        and row.get("actor_id")
    )
    if not actors:
        raise ValueError("Supply commits have no accepted Platform actor binding")
    candidate = evidence.verified_operation_evidence(
        SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
        options={
            "allow_rejected": True,
            "evaluated_actor_id": actors[0],
            "preclaimed_commit_ids": claimed_transaction_ids,
            "preclaimed_commit_attestations": (
                ()
                if verified_governance is None
                else governance_preclaimed_commit_attestations(verified_governance)
            ),
        },
    )
    if not isinstance(candidate, VerifiedSupplyFulfillmentEvidence):
        raise TypeError("supply and fulfillment evidence contract returned wrong type")
    claimed.update(_commit_ids(candidate))


def _complete_after_sales(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=_is_after_sales_commit,
        label="After-sales",
    )
    if not family:
        return
    candidate = evidence.verified_operation_evidence(
        AFTER_SALES_EVIDENCE_CONTRACT,
        options={"allow_rejected": True},
    )
    if not isinstance(candidate, VerifiedAfterSalesEvidence):
        raise TypeError("after-sales evidence contract returned wrong type")
    _claim_exact_family(candidate, family=family, claimed=claimed, label="After-sales")


def _complete_market_governance(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> VerifiedMarketGovernanceEvidence | None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=_is_governance_commit,
        label="Market-governance",
    )
    if not family:
        return None
    candidate = evidence.verified_operation_evidence(
        MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
        options={"allow_rejected": True},
    )
    if not isinstance(candidate, VerifiedMarketGovernanceEvidence):
        raise TypeError("market-governance evidence contract returned wrong type")
    _claim_exact_family(
        candidate,
        family=family,
        claimed=claimed,
        label="Market-governance",
    )
    return candidate


def _complete_protocol_events(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=_is_protocol_event_commit,
        label="Protocol-event",
    )
    if not family:
        return
    publish_commits = tuple(
        row for row in family if row.get("operation") == "publish_protocol_event"
    )
    if not publish_commits:
        raise ValueError("Protocol-event completion cannot infer a receipt-only event scope")
    event_ids = tuple(_protocol_event_id(row) for row in publish_commits)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Protocol-event publish commits repeat an event_id")
    candidate = evidence.verified_operation_evidence(
        PROTOCOL_EVENT_EVIDENCE_CONTRACT,
        options={"expected_event_ids": event_ids},
    )
    if not isinstance(candidate, VerifiedProtocolEventEvidence):
        raise TypeError("protocol-event evidence contract returned wrong type")
    _claim_exact_family(
        candidate,
        family=family,
        claimed=claimed,
        label="Protocol-event",
    )


def _complete_catalog_mutations(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=lambda row: (
            row.get("operation") == "catalog_mutation"
            or row.get("authority_action") == "world.apply_catalog_mutation"
        ),
        label="Catalog-mutation",
    )
    if not family:
        return
    candidate = evidence.verified_operation_evidence(
        CATALOG_MUTATION_EVIDENCE_CONTRACT,
    )
    if not isinstance(candidate, VerifiedCatalogMutationEvidence):
        raise TypeError("catalog-mutation evidence contract returned wrong type")
    _claim_exact_family(
        candidate,
        family=family,
        claimed=claimed,
        label="Catalog-mutation",
    )


def _complete_listing_claims(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=lambda row: (
            row.get("operation") == "apply_listing_claim"
            or row.get("authority_action") == "world.apply_listing_claim"
        ),
        label="Listing-claim",
    )
    if not family:
        return
    candidate = evidence.verified_operation_evidence(LISTING_CLAIM_EVIDENCE_CONTRACT)
    if not isinstance(candidate, VerifiedListingClaimEvidence):
        raise TypeError("listing-claim evidence contract returned wrong type")
    _claim_exact_family(
        candidate,
        family=family,
        claimed=claimed,
        label="Listing-claim",
    )


def _complete_evidence_records(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=lambda row: (
            row.get("operation") in {"persist_evidence_record", "bind_evidence_idempotency"}
            or row.get("authority_action") == "world.persist_evidence_record"
        ),
        label="Evidence-record",
    )
    if not family:
        return
    candidate = evidence.verified_operation_evidence(EVIDENCE_RECORD_EVIDENCE_CONTRACT)
    if not isinstance(candidate, VerifiedEvidenceRecordEvidence):
        raise TypeError("evidence-record contract returned wrong type")
    _claim_exact_family(
        candidate,
        family=family,
        claimed=claimed,
        label="Evidence-record",
    )


def _complete_cart(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=_is_cart_commit,
        label="Cart",
    )
    if not family:
        return
    contract_id, expected_type, scope = _cart_contract_scope(family)
    candidate = evidence.verified_operation_evidence(
        contract_id,
        options={
            **scope,
            "preclaimed_commit_ids": tuple(
                _commit_id(row) for row in world_commits if _commit_id(row) in claimed
            ),
        },
    )
    if not isinstance(candidate, expected_type):
        raise TypeError("cart evidence contract returned wrong type")
    _claim_exact_family(candidate, family=family, claimed=claimed, label="Cart")


def _complete_market_clock(
    evidence: OperationEvidenceSource,
    *,
    world_commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> None:
    family = _unclaimed_complete_family(
        world_commits,
        claimed=claimed,
        predicate=lambda row: (
            row.get("operation") == "advance_clock"
            or row.get("authority_action") == "world.advance_logical_time"
        ),
        label="Market-clock",
    )
    if not family:
        return
    candidate = evidence.verified_operation_evidence(MARKET_CLOCK_EVIDENCE_CONTRACT)
    if not isinstance(candidate, VerifiedMarketClockEvidence):
        raise TypeError("market-clock evidence contract returned wrong type")
    _claim_exact_family(
        candidate,
        family=family,
        claimed=claimed,
        label="Market-clock",
    )


def _unclaimed_complete_family(
    world_commits: tuple[dict[str, Any], ...],
    *,
    claimed: set[str],
    predicate: Any,
    label: str,
) -> tuple[dict[str, Any], ...]:
    """Return a wholly unclaimed operation family or fail on a partial claim.

    Existing operation contracts consume their complete family graph.  Calling
    one after another scorer has already claimed only part of that graph would
    make the same commit appear in two authority results.  Refusing the mixed
    state here keeps completion compositional and fail closed.
    """

    family = tuple(row for row in world_commits if predicate(row))
    unclaimed = tuple(row for row in family if _commit_id(row) not in claimed)
    if not unclaimed:
        return ()
    already_claimed = tuple(row for row in family if _commit_id(row) in claimed)
    if already_claimed:
        raise ValueError(f"{label} evidence is only partially claimed before completion")
    return tuple(sorted(family, key=_commit_sequence))


def _claim_exact_family(
    candidate: object,
    *,
    family: tuple[dict[str, Any], ...],
    claimed: set[str],
    label: str,
) -> None:
    """Accept only the exact contract's one-to-one family commit closure."""

    expected = {_commit_id(row) for row in family}
    observed = _commit_ids(candidate)
    if observed != expected:
        missing = sorted(expected - observed)
        foreign = sorted(observed - expected)
        raise ValueError(
            f"{label} exact contract returned a different commit partition; "
            f"missing={missing!r}, foreign={foreign!r}"
        )
    overlap = sorted(observed & claimed)
    if overlap:
        raise ValueError(f"{label} exact contract claimed commits twice: {overlap!r}")
    claimed.update(observed)


def _is_after_sales_commit(row: Mapping[str, Any]) -> bool:
    operation = row.get("operation")
    authority = row.get("authority_action")
    return operation in _AFTER_SALES_OPERATIONS or authority in {
        "world.apply_after_sales_intent",
        "world.complete_ledger_reconciliation",
    }


def _is_governance_commit(row: Mapping[str, Any]) -> bool:
    operation = row.get("operation")
    authority = row.get("authority_action")
    return operation in _GOVERNANCE_OPERATIONS or authority in {
        f"world.{value}" for value in _GOVERNANCE_OPERATIONS
    }


def _is_protocol_event_commit(row: Mapping[str, Any]) -> bool:
    operation = row.get("operation")
    authority = row.get("authority_action")
    writes = row.get("table_writes")
    has_protocol_write = isinstance(writes, list) and any(
        isinstance(write, Mapping)
        and write.get("table") in {"protocol_events", "protocol_receipts"}
        for write in writes
    )
    return (
        operation in _PROTOCOL_EVENT_OPERATIONS
        or authority in {f"world.{value}" for value in _PROTOCOL_EVENT_OPERATIONS}
        or has_protocol_write
    )


def _protocol_event_id(commit: Mapping[str, Any]) -> str:
    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        raise ValueError("Protocol-event publish commit has no table-write journal")
    event_rows = [
        write.get("after")
        for write in writes
        if isinstance(write, Mapping)
        and write.get("table") == "protocol_events"
        and isinstance(write.get("after"), Mapping)
    ]
    if len(event_rows) != 1:
        raise ValueError("Protocol-event publish commit does not create one event row")
    value = event_rows[0].get("event_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Protocol-event publish row has no event_id")
    return value


def _is_cart_commit(row: Mapping[str, Any]) -> bool:
    operation = row.get("operation")
    authority = row.get("authority_action")
    return any(
        operation == expected_operation or authority == expected_authority
        for expected_operation, expected_authority in _CART_AUTHORITY_PAIRS
    )


def _cart_contract_scope(
    family: tuple[dict[str, Any], ...],
) -> tuple[str, type[object], dict[str, Any]]:
    operations = tuple(str(row.get("operation")) for row in family)
    authorization_count = operations.count("create_cart_quote_request")
    quote_count = operations.count("issue_cart_quote")
    checkout_count = operations.count("checkout_cart_quote")
    expected_order = (
        (("create_cart_quote_request",) if authorization_count else ())
        + (("issue_cart_quote",) * quote_count)
        + (("checkout_cart_quote",) if checkout_count else ())
    )
    if (
        authorization_count > 1
        or checkout_count > 1
        or quote_count < int(bool(checkout_count))
        or operations != expected_order
    ):
        raise ValueError(f"Cart commits do not form a supported exact prefix: {operations!r}")
    expected_contract: tuple[str, type[object]]
    if authorization_count == 1 and quote_count == checkout_count == 0:
        expected_contract = (
            CART_AUTHORIZATION_PREFIX_EVIDENCE_CONTRACT,
            VerifiedCartAuthorizationPrefixEvidence,
        )
    elif quote_count >= 1 and checkout_count == 0:
        expected_contract = (
            CART_QUOTE_PREFIX_EVIDENCE_CONTRACT,
            VerifiedCartQuotePrefixEvidence,
        )
    elif quote_count >= 1 and checkout_count == 1:
        expected_contract = (CART_EVIDENCE_CONTRACT, VerifiedCartEvidence)
    else:
        raise ValueError(f"Cart commits do not form a supported exact prefix: {operations!r}")

    quote_commit = next(
        (row for row in family if row.get("operation") == "issue_cart_quote"),
        None,
    )
    if quote_commit is not None:
        quote = _single_created_row(quote_commit, table="persistent_cart_quotes")
        market_id = _required_text(quote.get("market_id"), "cart quote market_id")
        buyer_id = _required_text(quote.get("buyer_id"), "cart quote buyer_id")
        evaluated_actor_id = _required_text(
            quote.get("requested_by"),
            "cart quote requested_by",
        )
    else:
        request = _single_created_row(
            family[0],
            table="persistent_cart_quote_requests",
        )
        market_id = _required_text(request.get("market_id"), "cart request market_id")
        buyer_id = _required_text(request.get("buyer_id"), "cart request buyer_id")
        merchant_ids = request.get("allowed_merchant_ids")
        if (
            not isinstance(merchant_ids, list)
            or len(merchant_ids) != 1
            or not isinstance(merchant_ids[0], str)
            or not merchant_ids[0]
        ):
            raise ValueError("Cart authorization prefix does not identify one evaluated merchant")
        evaluated_actor_id = merchant_ids[0]
    return (
        expected_contract[0],
        expected_contract[1],
        {
            "market_id": market_id,
            "buyer_id": buyer_id,
            "evaluated_actor_id": evaluated_actor_id,
            "allow_repeated_quotes": quote_count > 1,
        },
    )


def _single_created_row(
    commit: Mapping[str, Any],
    *,
    table: str,
) -> Mapping[str, Any]:
    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        raise ValueError(f"{table} commit has no table-write journal")
    rows = [
        write.get("after")
        for write in writes
        if isinstance(write, Mapping)
        and write.get("table") == table
        and write.get("op") == "create"
        and isinstance(write.get("after"), Mapping)
    ]
    if len(rows) != 1:
        raise ValueError(f"{table} commit does not create one authoritative row")
    return rows[0]


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _commit_sequence(commit: Mapping[str, Any]) -> int:
    value = commit.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("World commit sequence is invalid")
    return value


def _commits_for_operation(
    commits: Iterable[dict[str, Any]],
    *,
    operation: str,
    authority_action: str,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        row
        for row in commits
        if row.get("operation") == operation and row.get("authority_action") == authority_action
    )


def _negotiation_event_row(commit: Mapping[str, Any]) -> dict[str, Any]:
    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        raise ValueError("Negotiation commit has no table-write journal")
    rows = [
        write.get("after")
        for write in writes
        if isinstance(write, Mapping)
        and write.get("table") == "negotiation_events"
        and isinstance(write.get("after"), Mapping)
    ]
    if len(rows) != 1:
        raise ValueError("Negotiation commit does not append one event row")
    row = dict(rows[0])
    required = (
        "negotiation_id",
        "max_rounds",
        "opened_at_tick",
        "expires_at_tick",
    )
    if any(name not in row for name in required):
        raise ValueError("Negotiation event row lacks its policy binding")
    return row


def _commit_id(commit: Mapping[str, Any]) -> str:
    value = commit.get("commit_id")
    if not isinstance(value, str) or not value:
        raise ValueError("World commit has no commit_id")
    return value


def _claimed_commit_ids(
    authority_calls: Iterable[tuple[str, str, object]],
) -> set[str]:
    claimed: set[str] = set()
    for _contract_id, _options_sha256, result in authority_calls:
        claimed.update(_commit_ids(result))
    return claimed


def _commit_ids(value: object, *, seen: set[int] | None = None) -> set[str]:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return set()
    visited = seen if seen is not None else set()
    identity = id(value)
    if identity in visited:
        return set()
    visited.add(identity)
    found: set[str] = set()
    if isinstance(value, Mapping):
        commit_id = value.get("commit_id")
        if isinstance(commit_id, str) and commit_id:
            found.add(commit_id)
        for nested in value.values():
            found.update(_commit_ids(nested, seen=visited))
        return found
    if isinstance(value, (tuple, list, set, frozenset)):
        for nested in value:
            found.update(_commit_ids(nested, seen=visited))
        return found
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            found.update(_commit_ids(getattr(value, field.name), seen=visited))
    return found


__all__ = [
    "OperationEvidenceSource",
    "complete_unclaimed_operation_evidence",
]
