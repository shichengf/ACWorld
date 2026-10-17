"""Exact evidence for CommerceWorld supply, fulfillment, and logistics.

The contract in this module is intentionally task agnostic.  It proves one
continuous authority graph:

``Actor request -> accepted Platform decision -> Platform response ->
World transaction commit -> replayed final World tables``.

Supply and shipment reads are checked against the authoritative World state at
the point of each read.  Repeated reads remain separate request and response
occurrences; they are not collapsed into ambient evidence or idempotent
retries.  Mutations are joined by actor-scoped idempotency keys,
then their table-write chain is applied to the initial snapshot.  Independently
verified external commits may be supplied as a composable sequence.  They are
replayed at their original request positions.  All remaining World
transactions must be supply or fulfillment operations claimed exactly once,
and the combined write chain must reconstruct the final snapshot.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from runtime.exact_join import (
    DEFAULT_OPERATION_EVIDENCE_REGISTRY,
    ExactJoinContext,
    ExactJoinError,
    LinkedPlatformExchange,
    OperationEvidenceContract,
    wire_envelope_sha256,
)
from runtime.world_state_replay import apply_commit_writes
from protocol.errors import SchemaError
from protocol.supply_authority import (
    coerce_supply_purchase_authority,
    supply_purchase_authority_id,
)
from world.transactions import scope_idempotency_key


SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT = "commerceworld.supply-fulfillment-logistics.v1"

_READ_KINDS = frozenset(
    {
        "commerce.read_supply_state",
        "commerce.read_shipment",
    }
)
_WRITE_KINDS = frozenset(
    {
        "commerce.update_supply",
        "platform.apply_supply_event",
        "platform.record_shipment_status",
        "platform.settle_payment",
        "platform.update_reputation",
        "commerce.allocate_fulfillment",
        "commerce.resolve_shipment",
    }
)
_RELEVANT_KINDS = _READ_KINDS | _WRITE_KINDS
_PRECLAIMED_COMMIT_IDS_OPTION = "preclaimed_commit_ids"
_PRECLAIMED_COMMIT_ATTESTATIONS_OPTION = "preclaimed_commit_attestations"
SUPPLY_FULFILLMENT_AUTHORITY_PAIRS = frozenset(
    {
        ("apply_supply_event", "world.apply_supply_event"),
        ("record_shipment_status", "world.record_shipment_status"),
        ("settle", "world.settle_order"),
        ("partial_settle", "world.settle_order_partial"),
        ("apply_settlement_reputation", "world.update_reputation"),
        ("allocate_orders_atomic", "world.allocate_orders_atomic"),
        ("resolve_shipment", "world.resolve_shipment"),
        (
            "issue_supply_purchase_authority",
            "world.issue_supply_purchase_authority",
        ),
    }
)
SUPPLY_FULFILLMENT_ACTION_KINDS = _WRITE_KINDS


@dataclass(frozen=True, slots=True)
class VerifiedSupplyFulfillmentRead:
    """One actor-scoped read joined to its exact authoritative projection."""

    action_kind: str
    exchange: LinkedPlatformExchange
    response: dict[str, Any]
    projection: Any


@dataclass(frozen=True, slots=True)
class VerifiedSupplyFulfillmentOperation:
    """One accepted mutation joined to its unique World transaction."""

    action_kind: str
    operation: str
    request_fingerprint: str
    exchange: LinkedPlatformExchange
    response: dict[str, Any] | None
    commit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedSupplyFulfillmentEvidence:
    """Complete exact evidence for one supply and fulfillment episode."""

    evaluated_actor_id: str
    reads: tuple[VerifiedSupplyFulfillmentRead, ...]
    operations: tuple[VerifiedSupplyFulfillmentOperation, ...]
    read_authority_commits: tuple[dict[str, Any], ...]
    claimed_commit_ids: tuple[str, ...]
    rejected_exchanges: tuple[LinkedPlatformExchange, ...] = ()

    def operations_for(
        self,
        *,
        action_kind: str | None = None,
        actor_id: str | None = None,
    ) -> tuple[VerifiedSupplyFulfillmentOperation, ...]:
        rows = self.operations
        if action_kind is not None:
            rows = tuple(row for row in rows if row.action_kind == action_kind)
        if actor_id is not None:
            rows = tuple(row for row in rows if row.exchange.decision.get("actor_id") == actor_id)
        return rows

    def reads_for(
        self,
        *,
        action_kind: str | None = None,
        actor_id: str | None = None,
    ) -> tuple[VerifiedSupplyFulfillmentRead, ...]:
        rows = self.reads
        if action_kind is not None:
            rows = tuple(row for row in rows if row.action_kind == action_kind)
        if actor_id is not None:
            rows = tuple(row for row in rows if row.exchange.decision.get("actor_id") == actor_id)
        return rows

    def rejections_for(
        self,
        *,
        action_kind: str | None = None,
        actor_id: str | None = None,
    ) -> tuple[LinkedPlatformExchange, ...]:
        """Return verified, zero-effect Platform rejections."""

        rows = self.rejected_exchanges
        if action_kind is not None:
            rows = tuple(
                row
                for row in rows
                if row.decision.get("action_kind") == action_kind
            )
        if actor_id is not None:
            rows = tuple(
                row for row in rows if row.decision.get("actor_id") == actor_id
            )
        return rows


@dataclass(frozen=True, slots=True)
class VerifiedPreclaimedCommitAttestation:
    """One commit already verified by another registered exact contract."""

    contract_id: str
    commit_id: str
    commit_sha256: str
    request_msg_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_id": self.contract_id,
            "commit_id": self.commit_id,
            "commit_sha256": self.commit_sha256,
            "request_msg_id": self.request_msg_id,
        }


def governance_preclaimed_commit_attestations(
    verified: object,
) -> tuple[dict[str, str], ...]:
    """Project a verified governance result into composable commit attestations.

    Raw commits are deliberately not accepted here.  The caller must first
    obtain a typed result from the registered market-governance exact contract.
    """

    from runtime.market_governance_evidence import (
        MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
        VerifiedMarketGovernanceEvidence,
    )

    if not isinstance(verified, VerifiedMarketGovernanceEvidence):
        raise TypeError("governance preclaim requires verified exact-contract evidence")
    rows: list[VerifiedPreclaimedCommitAttestation] = []
    for request in verified.requests:
        request_msg_id = _required_text(
            request.exchange.request.get("msg_id"),
            "governance request msg_id",
        )
        for operation in request.operations:
            commit_id = _required_text(
                operation.commit.get("commit_id"),
                "governance commit_id",
            )
            rows.append(
                VerifiedPreclaimedCommitAttestation(
                    contract_id=MARKET_GOVERNANCE_EVIDENCE_CONTRACT,
                    commit_id=commit_id,
                    commit_sha256=_commit_sha256(operation.commit),
                    request_msg_id=request_msg_id,
                )
            )
    rows.sort(key=lambda row: row.commit_id)
    if len({row.commit_id for row in rows}) != len(rows):
        raise ExactJoinError("verified governance result repeats a commit id")
    return tuple(row.to_dict() for row in rows)


@dataclass(frozen=True, slots=True)
class _CommitContract:
    operation: str
    authority_action: str
    commit_actor_id: str
    subject_id: str
    response_kind: str
    allowed_tables: frozenset[str]


def verify_supply_fulfillment_evidence_contract(
    context: ExactJoinContext,
    options: Mapping[str, Any],
) -> VerifiedSupplyFulfillmentEvidence:
    """Verify a complete supply, fulfillment, and logistics authority graph."""

    unknown = sorted(
        set(options)
        - {
            "allow_rejected",
            "evaluated_actor_id",
            "expected_read_kind",
            _PRECLAIMED_COMMIT_IDS_OPTION,
            _PRECLAIMED_COMMIT_ATTESTATIONS_OPTION,
        }
    )
    if unknown:
        raise ExactJoinError(
            "unknown supply and fulfillment evidence options: " + ", ".join(unknown)
        )
    evaluated_actor_id = _required_text(options.get("evaluated_actor_id"), "evaluated_actor_id")
    expected_read_kind_value = options.get("expected_read_kind")
    expected_read_kind = (
        None
        if expected_read_kind_value is None
        else _required_text(expected_read_kind_value, "expected_read_kind")
    )
    if expected_read_kind is not None and expected_read_kind not in _READ_KINDS:
        raise ExactJoinError("expected_read_kind is not a supply or shipment read")
    allow_rejected = options.get("allow_rejected", False)
    if not isinstance(allow_rejected, bool):
        raise ExactJoinError("allow_rejected must be boolean")

    relevant = tuple(
        exchange
        for exchange in context.exchanges
        if exchange.decision.get("action_kind") in _RELEVANT_KINDS
    )
    accepted: list[LinkedPlatformExchange] = []
    rejected: list[LinkedPlatformExchange] = []
    for exchange in relevant:
        decision = exchange.decision.get("decision")
        if decision == "accepted":
            accepted.append(exchange)
            continue
        if decision != "rejected":
            raise ExactJoinError("supply and fulfillment decision has an invalid outcome")
        if exchange.responses or any(
            exchange.decision.get(field) != []
            for field in ("response_kinds", "response_msg_ids", "response_sha256s")
        ):
            raise ExactJoinError("rejected supply or fulfillment request emitted a response")
        if not allow_rejected:
            raise ExactJoinError(
                "supply and fulfillment evidence contains a rejected Platform request"
            )
        rejected.append(exchange)

    all_transaction_commits = tuple(
        row for row in context.world_commits if row.get("commit_kind") == "transaction"
    )
    commit_ids = [
        _required_text(row.get("commit_id"), "World commit_id") for row in all_transaction_commits
    ]
    if len(commit_ids) != len(set(commit_ids)):
        raise ExactJoinError("World transaction commit ids are not unique")

    raw_preclaimed = options.get(_PRECLAIMED_COMMIT_IDS_OPTION, ())
    if not isinstance(raw_preclaimed, (list, tuple)) or any(
        not isinstance(value, str) or not value for value in raw_preclaimed
    ):
        raise ExactJoinError("preclaimed operation commit ids must be a text sequence")
    preclaimed_ids = tuple(raw_preclaimed)
    if len(preclaimed_ids) != len(set(preclaimed_ids)):
        raise ExactJoinError("preclaimed operation commit ids contain duplicates")
    preclaimed_attestations = _preclaimed_commit_attestations(
        options.get(_PRECLAIMED_COMMIT_ATTESTATIONS_OPTION, ())
    )
    attested_ids = tuple(row.commit_id for row in preclaimed_attestations)
    if set(attested_ids) - set(preclaimed_ids):
        raise ExactJoinError(
            "preclaimed commit attestation does not name a preclaimed commit"
        )
    commits_by_id = {
        _required_text(row.get("commit_id"), "World commit_id"): row
        for row in all_transaction_commits
    }
    if set(preclaimed_ids) - set(commits_by_id):
        raise ExactJoinError("preclaimed operation commit is absent from World journal")
    preclaimed_commits = tuple(commits_by_id[commit_id] for commit_id in preclaimed_ids)
    if any(
        (row.get("operation"), row.get("authority_action"))
        in SUPPLY_FULFILLMENT_AUTHORITY_PAIRS
        for row in preclaimed_commits
    ):
        raise ExactJoinError("a supply or fulfillment commit cannot be preclaimed")
    preclaimed_id_set = set(preclaimed_ids)
    transaction_commits = tuple(
        row for row in all_transaction_commits if row.get("commit_id") not in preclaimed_id_set
    )

    # This mutable copy is a verifier-local replay state.  Scenarios cannot
    # provide it and no benchmark-specific expectation participates in it.
    state = copy.deepcopy(context.initial_tables)
    external_commits = _external_commits_by_request_position(
        context,
        preclaimed_commits,
        attestations=preclaimed_attestations,
    )
    external_index = 0
    applied_commit_sequences: list[int] = []

    def apply_external_before(request_position: int | None) -> None:
        nonlocal external_index
        while external_index < len(external_commits):
            position, commit = external_commits[external_index]
            if request_position is not None and position >= request_position:
                break
            apply_commit_writes(
                state,
                commit,
                allowed_tables=frozenset(state),
            )
            applied_commit_sequences.append(_commit_sequence(commit))
            external_index += 1

    claimed: set[str] = set()
    reads: list[VerifiedSupplyFulfillmentRead] = []
    operations: list[VerifiedSupplyFulfillmentOperation] = []
    read_authority_commits: list[dict[str, Any]] = []

    for exchange in sorted(accepted, key=lambda row: row.request_position):
        apply_external_before(exchange.request_position)
        action_kind = _required_text(exchange.decision.get("action_kind"), "Platform action_kind")
        if action_kind in _READ_KINDS:
            if (
                action_kind == "commerce.read_supply_state"
                and str(exchange.decision.get("actor_id", "")).startswith("buyer:")
            ):
                authority_commit = _claim_supply_authority_commit(
                    exchange,
                    commits=transaction_commits,
                    claimed=claimed,
                )
                apply_commit_writes(
                    state,
                    authority_commit,
                    allowed_tables=frozenset({
                        "supply_purchase_authorities",
                    }),
                )
                applied_commit_sequences.append(
                    _commit_sequence(authority_commit)
                )
                read_authority_commits.append(authority_commit)
            reads.append(_verify_read(exchange, state=state))
            continue

        request_payload = _authoritative_mutation_payload(exchange, state=state)
        contract = _commit_contract(exchange, request_payload=request_payload)
        commit = _claim_commit(
            exchange,
            contract=contract,
            commits=transaction_commits,
            claimed=claimed,
        )
        apply_commit_writes(
            state,
            commit,
            allowed_tables=contract.allowed_tables,
        )
        applied_commit_sequences.append(_commit_sequence(commit))
        response = (
            _one_response(
                exchange,
                kind=contract.response_kind,
                sender=str(exchange.decision.get("platform_endpoint")),
                target=(
                    _required_text(
                        _payload(exchange.request).get("notify_to"),
                        "reputation notification target",
                    )
                    if action_kind == "platform.update_reputation"
                    else None
                ),
                allowed_additional_kinds=(
                    frozenset({"platform.update_reputation"})
                    if action_kind == "platform.settle_payment"
                    else frozenset()
                ),
            )
            if exchange.responses
            else None
        )
        expected_response_payload = _verify_mutation_response(
            exchange,
            response=response,
            commit=commit,
            state=state,
            request_payload=request_payload,
        )
        if response is None:
            _verify_undelivered_mutation_response(
                exchange,
                contract=contract,
                payload=expected_response_payload,
            )
        operations.append(
            VerifiedSupplyFulfillmentOperation(
                action_kind=action_kind,
                operation=contract.operation,
                request_fingerprint=_required_text(
                    exchange.decision.get("normalized_action_sha256"),
                    "normalized action fingerprint",
                ),
                exchange=exchange,
                response=response,
                commit=commit,
            )
        )

    _verify_settlement_reputation_links(operations)

    apply_external_before(None)

    unclaimed = [row for row in transaction_commits if row.get("commit_id") not in claimed]
    if unclaimed:
        raise ExactJoinError(
            "supply and fulfillment evidence contains an unclaimed World transaction"
        )
    operation_sequences = [_commit_sequence(row.commit) for row in operations]
    if operation_sequences != sorted(operation_sequences):
        raise ExactJoinError(
            "Platform mutation order differs from authoritative World commit order"
        )
    expected_commit_sequences = [
        _commit_sequence(row)
        for row in sorted(all_transaction_commits, key=_commit_sequence)
    ]
    if applied_commit_sequences != expected_commit_sequences:
        raise ExactJoinError(
            "composed operation contracts differ from authoritative World commit order"
        )
    if not _tables_equal(state, context.final_tables):
        raise ExactJoinError("claimed World writes do not reconstruct the exact final World tables")

    _verify_independent_read_identities(reads)

    selected_reads = [
        row
        for row in reads
        if row.exchange.decision.get("actor_id") == evaluated_actor_id
        and (expected_read_kind is None or row.action_kind == expected_read_kind)
    ]
    other_actor_reads = [
        row
        for row in reads
        if row.exchange.decision.get("actor_id") == evaluated_actor_id and row not in selected_reads
    ]
    if other_actor_reads:
        raise ExactJoinError("evaluated actor performed an unexpected authority read")

    return VerifiedSupplyFulfillmentEvidence(
        evaluated_actor_id=evaluated_actor_id,
        reads=tuple(reads),
        operations=tuple(operations),
        read_authority_commits=tuple(read_authority_commits),
        claimed_commit_ids=tuple(
            _required_text(row.get("commit_id"), "claimed commit_id")
            for row in sorted(transaction_commits, key=_commit_sequence)
            if row.get("commit_id") in claimed
        ),
        rejected_exchanges=tuple(rejected),
    )


def _verify_independent_read_identities(
    reads: Sequence[VerifiedSupplyFulfillmentRead],
) -> None:
    """Keep repeated authoritative reads distinct and exactly attributable.

    The shared linker already proves each request, decision, audited response
    occurrence, and response hash.  Reads add one stricter rule: a second
    business read must be a new request with a new response identity.  This
    prevents a duplicated ``msg_id`` or an exact retry from masquerading as
    independent evidence while still allowing an actor to query the same
    authoritative state more than once.
    """

    request_ids = [
        _required_text(row.exchange.request.get("msg_id"), "read request msg_id")
        for row in reads
    ]
    if len(request_ids) != len(set(request_ids)):
        raise ExactJoinError("authoritative reads reuse a request msg_id")

    response_ids = [
        _required_text(row.response.get("msg_id"), "read response msg_id")
        for row in reads
    ]
    if len(response_ids) != len(set(response_ids)):
        raise ExactJoinError("authoritative reads reuse a response msg_id")


def verify_settlement_reputation_followup_commits(
    context: ExactJoinContext,
) -> tuple[dict[str, Any], ...]:
    """Verify PSP-triggered reputation commits for projection composition.

    This narrow helper exists because a successful settlement can update the
    operational reputation projection after a governance snapshot was sealed.
    It accepts only the exact causal chain from one verified settlement
    request and receipt to one accepted reputation request and one World
    commit.  It does not accept arbitrary reputation writes.
    """

    updates = tuple(
        row
        for row in context.exchanges
        if row.decision.get("action_kind") == "platform.update_reputation"
        and row.decision.get("decision") == "accepted"
    )
    update_commits = tuple(
        row
        for row in context.world_commits
        if row.get("operation") == "apply_settlement_reputation"
        or row.get("authority_action") == "world.update_reputation"
    )
    claimed_updates: set[str] = set()
    operations: list[VerifiedSupplyFulfillmentOperation] = []
    for exchange in updates:
        contract = _commit_contract(exchange)
        commit = _claim_commit(
            exchange,
            contract=contract,
            commits=update_commits,
            claimed=claimed_updates,
        )
        response = _one_response(
            exchange,
            kind=contract.response_kind,
            sender="platform:reputation",
            target=_required_text(
                _payload(exchange.request).get("notify_to"),
                "reputation notification target",
            ),
        )
        _verify_mutation_response(
            exchange,
            response=response,
            commit=commit,
            state={},
        )
        operations.append(
            VerifiedSupplyFulfillmentOperation(
                action_kind="platform.update_reputation",
                operation=contract.operation,
                request_fingerprint=_required_text(
                    exchange.decision.get("normalized_action_sha256"),
                    "normalized action fingerprint",
                ),
                exchange=exchange,
                response=response,
                commit=commit,
            )
        )
    if len(claimed_updates) != len(update_commits):
        raise ExactJoinError(
            "settlement reputation has an unclaimed or duplicate World commit"
        )

    settlement_commits = tuple(
        row
        for row in context.world_commits
        if row.get("operation") in {"settle", "partial_settle"}
        or row.get("authority_action")
        in {"world.settle_order", "world.settle_order_partial"}
    )
    claimed_settlements: set[str] = set()
    for exchange in context.exchanges:
        if (
            exchange.decision.get("action_kind") != "platform.settle_payment"
            or exchange.decision.get("decision") != "accepted"
        ):
            continue
        raw_payload = _payload(exchange.request)
        request_payload = raw_payload
        if set(raw_payload) == {"cert_id"}:
            request_payload = {
                "order_id": _certificate_only_order_id(
                    exchange,
                    state=context.final_tables,
                )
            }
        contract = _commit_contract(
            exchange,
            request_payload=request_payload,
        )
        commit = _claim_commit(
            exchange,
            contract=contract,
            commits=settlement_commits,
            claimed=claimed_settlements,
        )
        if set(raw_payload) == {"cert_id"}:
            request_payload = _authoritative_mutation_payload(
                exchange,
                state=_state_before_commit(context, commit),
            )
            if request_payload.get("order_id") != contract.subject_id:
                raise ExactJoinError(
                    "certificate-only settlement changed its authoritative order"
                )
        response = _one_response(
            exchange,
            kind=contract.response_kind,
            sender="platform:psp",
            allowed_additional_kinds=frozenset({"platform.update_reputation"}),
        )
        _verify_mutation_response(
            exchange,
            response=response,
            commit=commit,
            state={},
            request_payload=request_payload,
        )
        operations.append(
            VerifiedSupplyFulfillmentOperation(
                action_kind="platform.settle_payment",
                operation=contract.operation,
                request_fingerprint=_required_text(
                    exchange.decision.get("normalized_action_sha256"),
                    "normalized action fingerprint",
                ),
                exchange=exchange,
                response=response,
                commit=commit,
            )
        )
    _verify_settlement_reputation_links(operations)
    return tuple(
        row.commit
        for row in sorted(
            (row for row in operations if row.action_kind == "platform.update_reputation"),
            key=lambda row: _commit_sequence(row.commit),
        )
    )


def _state_before_commit(
    context: ExactJoinContext,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay exact World commits that precede one transaction commit."""

    state = copy.deepcopy(context.initial_tables)
    target_sequence = _commit_sequence(target)
    for commit in sorted(context.world_commits, key=_commit_sequence):
        if _commit_sequence(commit) >= target_sequence:
            break
        apply_commit_writes(
            state,
            commit,
            allowed_tables=frozenset(state),
        )
    return state


def _external_commits_by_request_position(
    context: ExactJoinContext,
    commits: tuple[dict[str, Any], ...],
    *,
    attestations: tuple[VerifiedPreclaimedCommitAttestation, ...],
) -> tuple[tuple[int, dict[str, Any]], ...]:
    """Bind independently claimed commits to their original Platform requests."""

    attestations_by_commit = {row.commit_id: row for row in attestations}
    if len(attestations_by_commit) != len(attestations):
        raise ExactJoinError("preclaimed commit attestations contain duplicates")
    verified_attestations: dict[str, VerifiedPreclaimedCommitAttestation] = {}
    if attestations:
        # Attestation bytes are only a request to compose evidence.  Re-run the
        # named exact contract over this same immutable context, then compare
        # every claimed commit, digest, and request identity.  A caller cannot
        # turn a raw or forged commit into trusted evidence by constructing an
        # attestation-shaped mapping.
        from runtime.market_governance_evidence import (
            VerifiedMarketGovernanceEvidence,
            verify_market_governance_evidence_contract,
        )

        governance = verify_market_governance_evidence_contract(context, {})
        if not isinstance(governance, VerifiedMarketGovernanceEvidence):
            raise ExactJoinError("governance exact contract returned an invalid result")
        verified_attestations = {
            row.commit_id: row
            for row in (
                VerifiedPreclaimedCommitAttestation(**wire)
                for wire in governance_preclaimed_commit_attestations(governance)
            )
        }

    positioned: list[tuple[int, dict[str, Any]]] = []
    for commit in commits:
        commit_id = _required_text(commit.get("commit_id"), "World commit_id")
        attestation = attestations_by_commit.get(commit_id)
        if attestation is not None:
            verified = verified_attestations.get(commit_id)
            if verified is None or attestation != verified:
                raise ExactJoinError(
                    "preclaimed commit attestation was not returned by its exact contract"
                )
            matches = [
                exchange
                for exchange in context.exchanges
                if exchange.request.get("msg_id") == attestation.request_msg_id
            ]
            if len(matches) != 1 or matches[0].decision.get("decision") != "accepted":
                raise ExactJoinError(
                    "attested preclaimed commit has no unique accepted Platform request"
                )
            positioned.append((matches[0].request_position, commit))
            continue

        commit_key = _required_text(commit.get("idempotency_key"), "World idempotency_key")
        matches: list[LinkedPlatformExchange] = []
        for exchange in context.exchanges:
            decision = exchange.decision
            if decision.get("decision") != "accepted":
                continue
            raw_key = decision.get("idempotency_key")
            actor_id = decision.get("actor_id")
            if not isinstance(raw_key, str) or not raw_key:
                continue
            if not isinstance(actor_id, str) or not actor_id:
                continue
            if commit_key in {raw_key, scope_idempotency_key(actor_id, raw_key)}:
                matches.append(exchange)
        if not matches:
            raise ExactJoinError(
                "preclaimed operation commit does not bind to one accepted Platform request"
            )
        if len(matches) > 1:
            positioned.append(
                (_exact_retry_request_position(commit, matches), commit)
            )
        else:
            positioned.append((matches[0].request_position, commit))

    positioned.sort(key=lambda row: (row[0], _commit_sequence(row[1])))
    # One accepted Platform request may causally produce several commits that
    # are owned by different exact contracts.  Their authoritative World
    # sequence, not an invented one-request-one-commit restriction, defines
    # replay order.
    return tuple(positioned)


def _exact_retry_request_position(
    commit: Mapping[str, Any],
    matches: Sequence[LinkedPlatformExchange],
) -> int:
    """Bind a reused exact authority effect to its first identical request."""

    identities: set[tuple[str, str, str, str, str]] = set()
    for exchange in matches:
        decision = exchange.decision
        identities.add(
            (
                _required_text(decision.get("actor_id"), "retry actor_id"),
                _required_text(
                    decision.get("platform_endpoint"), "retry Platform endpoint"
                ),
                _required_text(decision.get("action_kind"), "retry action_kind"),
                _required_text(decision.get("idempotency_key"), "retry idempotency_key"),
                _required_text(
                    decision.get("normalized_action_sha256"),
                    "retry normalized action fingerprint",
                ),
            )
        )
    if len(identities) != 1:
        raise ExactJoinError(
            "preclaimed operation retries changed actor, route, action, key, or fingerprint"
        )
    actor_id, _endpoint, _kind, raw_key, _fingerprint = next(iter(identities))
    commit_key = _required_text(commit.get("idempotency_key"), "World idempotency_key")
    if commit_key not in {raw_key, scope_idempotency_key(actor_id, raw_key)}:
        raise ExactJoinError("preclaimed retry does not own the World commit key")
    return min(row.request_position for row in matches)


def _preclaimed_commit_attestations(
    value: Any,
) -> tuple[VerifiedPreclaimedCommitAttestation, ...]:
    if not isinstance(value, (list, tuple)):
        raise ExactJoinError("preclaimed commit attestations must be a sequence")
    output: list[VerifiedPreclaimedCommitAttestation] = []
    required_keys = {
        "contract_id",
        "commit_id",
        "commit_sha256",
        "request_msg_id",
    }
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != required_keys:
            raise ExactJoinError("preclaimed commit attestation has an invalid schema")
        output.append(
            VerifiedPreclaimedCommitAttestation(
                contract_id=_required_text(raw.get("contract_id"), "attestation contract_id"),
                commit_id=_required_text(raw.get("commit_id"), "attestation commit_id"),
                commit_sha256=_required_text(
                    raw.get("commit_sha256"), "attestation commit_sha256"
                ),
                request_msg_id=_required_text(
                    raw.get("request_msg_id"), "attestation request_msg_id"
                ),
            )
        )
    commit_ids = [row.commit_id for row in output]
    if len(commit_ids) != len(set(commit_ids)):
        raise ExactJoinError("preclaimed commit attestations contain duplicates")
    return tuple(output)


def _verify_read(
    exchange: LinkedPlatformExchange,
    *,
    state: Mapping[str, Any],
) -> VerifiedSupplyFulfillmentRead:
    kind = _required_text(exchange.decision.get("action_kind"), "read action_kind")
    request = _payload(exchange.request)
    endpoint = _required_text(exchange.decision.get("platform_endpoint"), "read Platform endpoint")
    projection: Any
    if kind == "commerce.read_supply_state":
        if endpoint != "platform:supply":
            raise ExactJoinError("supply read used the wrong Platform endpoint")
        raw_skus = request.get("sku_ids")
        if not isinstance(raw_skus, list) or not raw_skus:
            raise ExactJoinError("supply read has no sku_ids")
        sku_ids = tuple(_required_text(value, "supply read sku_id") for value in raw_skus)
        if len(sku_ids) != len(set(sku_ids)):
            raise ExactJoinError("supply read sku_ids are not unique")
        projection = [_supply_projection(state, sku_id) for sku_id in sku_ids]
        response = _one_response(
            exchange,
            kind="platform.supply_state",
            sender="platform:supply",
        )
        actor_id = _required_text(exchange.decision.get("actor_id"), "read actor_id")
        purchase_options = (
            [
                _supply_purchase_option(
                    state,
                    row,
                    buyer_id=actor_id,
                    request_idempotency_key=_required_text(
                        exchange.request.get("idempotency_key"),
                        "supply read idempotency_key",
                    ),
                )
                for row in projection
                if _integer(row.get("available_qty"), "available_qty") > 0
            ]
            if actor_id.startswith("buyer:")
            else []
        )
        if _payload(response) != {
            "states": projection,
            "purchase_options": purchase_options,
        }:
            raise ExactJoinError("supply read response differs from authoritative World")
    elif kind == "commerce.read_shipment":
        if endpoint != "platform:fulfillment":
            raise ExactJoinError("shipment read used the wrong Platform endpoint")
        shipment_id = _required_text(request.get("shipment_id"), "shipment_id")
        projection = _shipment_projection(state, shipment_id)
        actor_id = _required_text(exchange.decision.get("actor_id"), "read actor_id")
        if actor_id not in {
            projection.get("buyer_id"),
            projection.get("merchant_id"),
        }:
            raise ExactJoinError("shipment read actor is not a transaction party")
        response = _one_response(
            exchange,
            kind="platform.shipment_state",
            sender="platform:fulfillment",
        )
        if _payload(response) != {
            "shipment": projection,
            "replacement_options": _shipment_replacement_options(state, projection),
        }:
            raise ExactJoinError("shipment read response differs from authoritative World")
    else:  # pragma: no cover - guarded by the caller's closed set
        raise ExactJoinError(f"unsupported authoritative read {kind!r}")
    return VerifiedSupplyFulfillmentRead(
        action_kind=kind,
        exchange=exchange,
        response=response,
        projection=projection,
    )


def _certificate_only_order_id(
    exchange: LinkedPlatformExchange,
    *,
    state: Mapping[str, Any],
) -> str:
    """Resolve a compact settlement's order identity without trusting text."""

    cert_id = _required_text(
        _payload(exchange.request).get("cert_id"), "settlement cert_id"
    )
    raw_certificates = state.get("match_certificates")
    if not isinstance(raw_certificates, list):
        raise ExactJoinError("World match_certificates table has an invalid shape")
    matches = [
        row
        for row in raw_certificates
        if isinstance(row, Mapping) and row.get("cert_id") == cert_id
    ]
    if len(matches) != 1:
        raise ExactJoinError(
            "certificate-only settlement has no unique authoritative certificate"
        )
    actor_id = _required_text(
        exchange.decision.get("actor_id"), "settlement actor_id"
    )
    if matches[0].get("buyer_id") != actor_id:
        raise ExactJoinError("certificate-only settlement belongs to another buyer")
    return _required_text(
        matches[0].get("order_id"), "certificate order_id"
    )


def _authoritative_mutation_payload(
    exchange: LinkedPlatformExchange,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve compact settlement intent from authoritative World state.

    A buyer may settle with only a match-certificate id or with a compact
    World-issued supply authority.  In either form the commercial identity
    belongs to authoritative World state and must not be copied from actor
    text.  Resolve those fields at the request's replay position, while the
    authority is still fresh, and leave every other mutation payload
    unchanged.
    """

    payload = _payload(exchange.request)
    if exchange.decision.get("action_kind") != "platform.settle_payment":
        return payload
    if "supply_authority_id" in payload:
        return _supply_authority_settlement_payload(
            exchange,
            payload=payload,
            state=state,
        )
    if set(payload) != {"cert_id"}:
        return payload

    cert_id = _required_text(payload.get("cert_id"), "settlement cert_id")
    raw_certificates = state.get("match_certificates")
    if not isinstance(raw_certificates, list):
        raise ExactJoinError("World match_certificates table has an invalid shape")
    matches = [
        dict(row)
        for row in raw_certificates
        if isinstance(row, Mapping) and row.get("cert_id") == cert_id
    ]
    if len(matches) != 1:
        raise ExactJoinError(
            "certificate-only settlement has no unique authoritative certificate"
        )
    certificate = matches[0]
    actor_id = _required_text(
        exchange.decision.get("actor_id"), "settlement actor_id"
    )
    if certificate.get("buyer_id") != actor_id:
        raise ExactJoinError("certificate-only settlement belongs to another buyer")

    logical_time = _integer(state.get("logical_time"), "World logical_time")
    expires_at_tick = _integer(
        certificate.get("expires_at_tick"), "certificate expires_at_tick"
    )
    if logical_time >= expires_at_tick:
        raise ExactJoinError("certificate-only settlement used an expired certificate")

    sku_id = _required_text(certificate.get("sku_id"), "certificate sku_id")
    merchant_id = _required_text(
        certificate.get("merchant_id"), "certificate merchant_id"
    )
    qty = _integer(certificate.get("qty"), "certificate qty")
    if qty <= 0:
        raise ExactJoinError("certificate qty must be positive")
    inventory_revision = _integer(
        certificate.get("inventory_revision"), "certificate inventory_revision"
    )
    raw_inventory = state.get("inventory")
    inventory = (
        raw_inventory.get(sku_id)
        if isinstance(raw_inventory, Mapping)
        else None
    )
    if not isinstance(inventory, Mapping):
        raise ExactJoinError("certificate-only settlement has no World inventory row")
    if (
        inventory.get("merchant_id") != merchant_id
        or _integer(inventory.get("version"), "inventory version")
        != inventory_revision
        or _integer(inventory.get("qty_available"), "qty_available")
        - _integer(inventory.get("qty_reserved"), "qty_reserved")
        < qty
    ):
        raise ExactJoinError(
            "certificate-only settlement does not bind fresh World inventory"
        )

    raw_catalog = state.get("catalog")
    listings = [
        row
        for row in raw_catalog
        if isinstance(row, Mapping) and row.get("sku_id") == sku_id
    ] if isinstance(raw_catalog, list) else []
    if len(listings) != 1 or listings[0].get("merchant_id") != merchant_id:
        raise ExactJoinError(
            "certificate-only settlement has no exact World listing owner"
        )
    attributes = listings[0].get("attributes")
    catalog_revision = (
        attributes.get("catalog_revision", 1)
        if isinstance(attributes, Mapping)
        else 1
    )
    if (
        isinstance(catalog_revision, bool)
        or not isinstance(catalog_revision, int)
        or catalog_revision < 1
        or catalog_revision
        != _integer(
            certificate.get("catalog_revision"), "certificate catalog_revision"
        )
    ):
        raise ExactJoinError(
            "certificate-only settlement does not bind the current catalog revision"
        )
    price = listings[0].get("list_price")
    amount = price.get("amount") if isinstance(price, Mapping) else None
    currency = _required_text(certificate.get("currency"), "certificate currency")
    unit_price_cents = _integer(
        certificate.get("unit_price_cents"), "certificate unit_price_cents"
    )
    try:
        listing_cents = int(
            (Decimal(str(amount)) * 100).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExactJoinError("World listing has invalid exact money") from exc
    if (
        not isinstance(price, Mapping)
        or price.get("currency") != currency
        or listing_cents != unit_price_cents
    ):
        raise ExactJoinError(
            "certificate-only settlement does not bind the current World price"
        )

    return {
        "cert_id": cert_id,
        "order_id": _certificate_only_order_id(exchange, state=state),
        "buyer_id": actor_id,
        "merchant_id": merchant_id,
        "sku_id": sku_id,
        "qty": qty,
        "agreed_price": {
            "amount": str(Decimal(unit_price_cents) / Decimal(100)),
            "currency": currency,
        },
    }


def _supply_authority_settlement_payload(
    exchange: LinkedPlatformExchange,
    *,
    payload: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and validate one compact direct-supply settlement.

    The request carries only the World authority identity, its digest, the
    chosen SKU and quantity, plus the partial-fill choice.  Price, currency,
    merchant, buyer, and order identity are reconstructed from the immutable
    World row.  This mirrors the Platform payment service and prevents either
    actor text or benchmark expectations from becoming transaction authority.
    """

    allowed = {
        "supply_authority_id",
        "supply_authority_digest",
        "sku_id",
        "qty",
        "allow_partial",
    }
    required = allowed - {"allow_partial"}
    if not required.issubset(payload) or not set(payload) <= allowed:
        raise ExactJoinError("supply settlement fields are not exact")

    authority_id = _required_text(
        payload.get("supply_authority_id"), "supply_authority_id"
    )
    raw_authorities = state.get("supply_purchase_authorities")
    if not isinstance(raw_authorities, list):
        raise ExactJoinError("World supply authority table has an invalid shape")
    matches = [
        row
        for row in raw_authorities
        if isinstance(row, Mapping) and row.get("authority_id") == authority_id
    ]
    if len(matches) != 1:
        raise ExactJoinError("supply settlement has no unique World authority")
    try:
        authority = coerce_supply_purchase_authority(matches[0])
    except (SchemaError, TypeError, ValueError) as exc:
        raise ExactJoinError("World supply authority row is invalid") from exc

    actor_id = _required_text(
        exchange.decision.get("actor_id"), "settlement actor_id"
    )
    if (
        actor_id != authority.buyer_id
        or payload.get("supply_authority_digest")
        != authority.authority_digest
        or payload.get("sku_id") != authority.sku_id
    ):
        raise ExactJoinError("supply settlement changed its authority binding")

    qty = _integer(payload.get("qty"), "supply settlement qty")
    if qty <= 0:
        raise ExactJoinError("supply settlement qty must be positive")
    allow_partial = payload.get("allow_partial", False)
    if not isinstance(allow_partial, bool):
        raise ExactJoinError("supply settlement allow_partial must be boolean")

    logical_time = _integer(state.get("logical_time"), "World logical_time")
    if (
        logical_time < authority.issued_at_tick
        or logical_time >= authority.expires_at_tick
    ):
        raise ExactJoinError("supply settlement used a stale World authority")

    projection = _supply_projection(state, authority.sku_id)
    if (
        projection.get("merchant_id") != authority.merchant_id
        or projection.get("unit_price_cents") != authority.unit_price_cents
        or projection.get("version") != authority.supply_version
    ):
        raise ExactJoinError("supply authority no longer matches current World supply")
    listing = _listing_for_supply(state, authority.sku_id)
    if _listing_currency(listing) != authority.currency:
        raise ExactJoinError("supply authority currency changed")
    available_qty = _integer(
        projection.get("available_qty"), "current available supply"
    )
    if authority.available_qty <= 0 or available_qty <= 0:
        raise ExactJoinError("supply authority has no purchasable inventory")
    if not allow_partial and qty > available_qty:
        raise ExactJoinError(
            "all-or-nothing supply settlement exceeds current availability"
        )

    return {
        "supply_authority_id": authority.authority_id,
        "supply_authority_digest": authority.authority_digest,
        "order_id": authority.order_id,
        "buyer_id": authority.buyer_id,
        "merchant_id": authority.merchant_id,
        "sku_id": authority.sku_id,
        "qty": qty,
        "agreed_price": {
            "amount": str(
                Decimal(authority.unit_price_cents) / Decimal(100)
            ),
            "currency": authority.currency,
        },
        "allow_partial": allow_partial,
    }


def _commit_contract(
    exchange: LinkedPlatformExchange,
    *,
    request_payload: Mapping[str, Any] | None = None,
) -> _CommitContract:
    action_kind = _required_text(exchange.decision.get("action_kind"), "mutation action_kind")
    actor_id = _required_text(exchange.decision.get("actor_id"), "mutation actor_id")
    endpoint = _required_text(
        exchange.decision.get("platform_endpoint"), "mutation Platform endpoint"
    )
    payload = (
        _payload(exchange.request)
        if request_payload is None
        else dict(request_payload)
    )
    if action_kind in {"commerce.update_supply", "platform.apply_supply_event"}:
        if endpoint != "platform:supply":
            raise ExactJoinError("supply event used the wrong Platform endpoint")
        if action_kind == "platform.apply_supply_event" and actor_id != "runtime:supply":
            raise ExactJoinError("authoritative supply event has the wrong actor")
        return _CommitContract(
            operation="apply_supply_event",
            authority_action="world.apply_supply_event",
            commit_actor_id="platform:supply",
            subject_id=_required_text(payload.get("sku_id"), "supply sku_id"),
            response_kind="platform.supply_event_applied",
            allowed_tables=frozenset({"inventory", "catalog"}),
        )
    if action_kind == "platform.record_shipment_status":
        if endpoint != "platform:fulfillment" or actor_id != "runtime:logistics":
            raise ExactJoinError("authoritative shipment event has the wrong route")
        return _CommitContract(
            operation="record_shipment_status",
            authority_action="world.record_shipment_status",
            commit_actor_id="platform:fulfillment",
            subject_id=_required_text(payload.get("shipment_id"), "shipment_id"),
            response_kind="platform.shipment_state",
            allowed_tables=frozenset({"shipments", "logical_time"}),
        )
    if action_kind == "platform.settle_payment":
        if endpoint != "platform:psp" or not actor_id.startswith("buyer:"):
            raise ExactJoinError("settlement has the wrong buyer or Platform route")
        response_kinds = {
            str((row.get("action") or {}).get("kind"))
            for row in exchange.responses
            if (row.get("action") or {}).get("kind")
            != "platform.update_reputation"
        }
        if response_kinds == {"platform.fulfillment_allocation"}:
            return _CommitContract(
                operation="partial_settle",
                authority_action="world.settle_order_partial",
                commit_actor_id="platform:psp",
                subject_id=_required_text(payload.get("order_id"), "order_id"),
                response_kind="platform.fulfillment_allocation",
                allowed_tables=frozenset(
                    {
                        "orders",
                        "inventory",
                        "fulfillments",
                        "ledger",
                        "order_timelines",
                        "payment_states",
                        "logical_time",
                    }
                ),
            )
        if response_kinds == {"platform.settlement_receipt"}:
            return _CommitContract(
                operation="settle",
                authority_action="world.settle_order",
                commit_actor_id="platform",
                subject_id=_required_text(payload.get("order_id"), "order_id"),
                response_kind="platform.settlement_receipt",
                allowed_tables=frozenset(
                    {
                        "orders",
                        "inventory",
                        "ledger",
                        "order_timelines",
                        "payment_states",
                        "logical_time",
                    }
                ),
            )
        raise ExactJoinError("settlement has no supported exact Platform response")
    if action_kind == "platform.update_reputation":
        if endpoint != "platform:reputation" or actor_id != "platform:psp":
            raise ExactJoinError(
                "settlement reputation has the wrong Platform authority route"
            )
        txn_id = _required_text(payload.get("txn_id"), "reputation txn_id")
        merchant_id = _required_text(
            payload.get("merchant_id"), "reputation merchant_id"
        )
        if payload.get("notify_to") != merchant_id:
            raise ExactJoinError(
                "settlement reputation notification is not merchant-bound"
            )
        return _CommitContract(
            operation="apply_settlement_reputation",
            authority_action="world.update_reputation",
            commit_actor_id="platform:reputation",
            subject_id=f"reputation-settlement:{txn_id}",
            response_kind="platform.reputation_updated",
            allowed_tables=frozenset({"reputation", "reputation_settlements"}),
        )
    if action_kind == "commerce.allocate_fulfillment":
        if endpoint != "platform:fulfillment" or not actor_id.startswith("merchant:"):
            raise ExactJoinError("allocation has the wrong merchant or Platform route")
        return _CommitContract(
            operation="allocate_orders_atomic",
            authority_action="world.allocate_orders_atomic",
            commit_actor_id="platform:fulfillment",
            subject_id=_required_text(payload.get("allocation_id"), "allocation_id"),
            response_kind="platform.allocation_batch",
            allowed_tables=frozenset(
                {
                    "orders",
                    "inventory",
                    "ledger",
                    "fulfillments",
                    "order_timelines",
                    "logical_time",
                }
            ),
        )
    if action_kind == "commerce.resolve_shipment":
        if endpoint != "platform:fulfillment" or actor_id.split(":", 1)[0] not in {
            "buyer",
            "merchant",
        }:
            raise ExactJoinError("shipment resolution has the wrong actor or route")
        return _CommitContract(
            operation="resolve_shipment",
            authority_action="world.resolve_shipment",
            commit_actor_id="platform:fulfillment",
            subject_id=_required_text(payload.get("shipment_id"), "shipment_id"),
            response_kind="platform.shipment_resolved",
            allowed_tables=frozenset(
                {
                    "shipments",
                    "inventory",
                    "orders",
                    "ledger",
                    "order_timelines",
                    "logical_time",
                }
            ),
        )
    raise ExactJoinError(f"unsupported supply or fulfillment mutation {action_kind!r}")


def _claim_commit(
    exchange: LinkedPlatformExchange,
    *,
    contract: _CommitContract,
    commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> dict[str, Any]:
    actor_id = _required_text(exchange.decision.get("actor_id"), "request actor_id")
    raw_key = _required_text(exchange.request.get("idempotency_key"), "idempotency_key")
    scoped_key = scope_idempotency_key(actor_id, raw_key)
    expected_keys = {scoped_key}
    if exchange.decision.get("action_kind") == "platform.update_reputation":
        # This request is emitted by the trusted PSP rather than by an
        # independently authenticated principal.  The World records the exact
        # Platform-issued key, while actor-originated requests remain scoped.
        expected_keys.add(raw_key)
    matches = [
        row
        for row in commits
        if row.get("operation") == contract.operation
        and row.get("authority_action") == contract.authority_action
        and row.get("actor_id") == contract.commit_actor_id
        and row.get("idempotency_key") in expected_keys
        and row.get("request_fingerprint") is None
        and row.get("subject_id") == contract.subject_id
    ]
    if len(matches) != 1:
        raise ExactJoinError(
            f"{contract.operation} has no unique actor and idempotency-bound World commit"
        )
    commit = matches[0]
    commit_id = _required_text(commit.get("commit_id"), "World commit_id")
    if commit_id in claimed:
        raise ExactJoinError("two Platform requests claimed one World transaction")
    invariants = commit.get("invariants_held")
    if not (
        invariants is True
        or (
            isinstance(invariants, list)
            and bool(invariants)
            and all(isinstance(value, str) and value for value in invariants)
        )
    ):
        raise ExactJoinError("claimed World transaction did not preserve invariants")
    claimed.add(commit_id)
    return commit


def _claim_supply_authority_commit(
    exchange: LinkedPlatformExchange,
    *,
    commits: tuple[dict[str, Any], ...],
    claimed: set[str],
) -> dict[str, Any]:
    """Claim the World mint transaction caused by one buyer supply read."""

    actor_id = _required_text(exchange.decision.get("actor_id"), "request actor_id")
    raw_key = _required_text(
        exchange.request.get("idempotency_key"), "supply read idempotency_key"
    )
    request = _payload(exchange.request)
    raw_skus = request.get("sku_ids")
    if not isinstance(raw_skus, list) or not raw_skus:
        raise ExactJoinError("supply authority read has no sku ids")
    sku_ids = tuple(_required_text(row, "supply sku id") for row in raw_skus)
    expected_subject = supply_purchase_authority_id(
        buyer_id=actor_id,
        request_idempotency_key=raw_key,
        sku_id=sku_ids[0],
    )
    matches = [
        row
        for row in commits
        if row.get("operation") == "issue_supply_purchase_authority"
        and row.get("authority_action")
        == "world.issue_supply_purchase_authority"
        and row.get("actor_id") == actor_id
        and row.get("idempotency_key") == raw_key
        and row.get("subject_id") == expected_subject
    ]
    if len(matches) != 1:
        raise ExactJoinError(
            "buyer supply read has no unique World authority mint commit"
        )
    commit = matches[0]
    commit_id = _required_text(commit.get("commit_id"), "World commit_id")
    if commit_id in claimed:
        raise ExactJoinError("two supply reads claimed one authority commit")
    writes = commit.get("table_writes")
    if not isinstance(writes, list):
        raise ExactJoinError("supply authority commit has no writes")
    if {
        row.get("table") for row in writes if isinstance(row, Mapping)
    } != {"supply_purchase_authorities"}:
        raise ExactJoinError(
            "supply authority issuance changed commercial World state"
        )
    authority_writes = [
        row
        for row in writes
        if isinstance(row, Mapping)
        and row.get("table") == "supply_purchase_authorities"
    ]
    if len(authority_writes) != len(sku_ids):
        raise ExactJoinError("supply authority commit has the wrong batch size")
    observed: set[str] = set()
    for write in authority_writes:
        if write.get("op") != "create" or write.get("before") is not None:
            raise ExactJoinError("supply authority row is not append-only")
        after = write.get("after")
        if not isinstance(after, Mapping):
            raise ExactJoinError("supply authority commit row is invalid")
        try:
            authority = coerce_supply_purchase_authority(after)
        except (SchemaError, TypeError, ValueError) as exc:
            raise ExactJoinError("supply authority commit row is invalid") from exc
        sku_id = authority.sku_id
        if (
            authority.authority_id
            != supply_purchase_authority_id(
                buyer_id=actor_id,
                request_idempotency_key=raw_key,
                sku_id=sku_id,
            )
            or authority.buyer_id != actor_id
            or authority.request_idempotency_key != raw_key
        ):
            raise ExactJoinError("supply authority commit binding changed")
        observed.add(sku_id)
    if observed != set(sku_ids):
        raise ExactJoinError("supply authority commit sku set changed")
    claimed.add(commit_id)
    return commit


def _verify_mutation_response(
    exchange: LinkedPlatformExchange,
    *,
    response: Mapping[str, Any] | None,
    commit: Mapping[str, Any],
    state: Mapping[str, Any],
    request_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action_kind = _required_text(exchange.decision.get("action_kind"), "mutation action_kind")
    request = (
        _payload(exchange.request)
        if request_payload is None
        else dict(request_payload)
    )
    response_payload = None if response is None else _payload(response)
    writes = _writes_by_table(commit)
    scoped_key = scope_idempotency_key(
        _required_text(exchange.decision.get("actor_id"), "mutation actor_id"),
        _required_text(exchange.request.get("idempotency_key"), "idempotency_key"),
    )
    if action_kind in {"commerce.update_supply", "platform.apply_supply_event"}:
        sku_id = _required_text(request.get("sku_id"), "supply sku_id")
        projection = _supply_projection(state, sku_id)
        if response_payload is not None and response_payload != {"supply_state": projection}:
            raise ExactJoinError("supply event response differs from World outcome")
        inventory = _one_after_write(writes, table="inventory", key=sku_id)
        raw_events = inventory.get("supply_events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ExactJoinError("supply event commit has no authoritative event row")
        event = raw_events[-1]
        if not isinstance(event, Mapping):
            raise ExactJoinError("supply event history row is invalid")
        expected_event = {
            "sku_id": sku_id,
            "qty_delta": int(request.get("qty_delta", 0)),
            "eta_day": request.get("eta_day"),
            "unit_price_cents": request.get("unit_price_cents"),
            "expected_version": request.get("expected_version"),
            "original_actor": exchange.decision.get("actor_id"),
            "scope": "platform:supply",
            "idempotency_key": scoped_key,
            "outcome": projection,
        }
        if dict(event) != expected_event:
            raise ExactJoinError("supply event request differs from World event history")
        return {"supply_state": projection}
    if action_kind == "platform.record_shipment_status":
        shipment_id = _required_text(request.get("shipment_id"), "shipment_id")
        projection = _shipment_projection(state, shipment_id)
        if response_payload is not None and response_payload != {"shipment": projection}:
            raise ExactJoinError("shipment event response differs from World outcome")
        shipment = _one_after_write(writes, table="shipments", key=shipment_id)
        history = shipment.get("status_history")
        if not isinstance(history, list) or not history:
            raise ExactJoinError("shipment event commit has no status history")
        last = history[-1]
        if not isinstance(last, Mapping) or any(
            last.get(name) != expected
            for name, expected in {
                "event_id": request.get("event_id"),
                "status": request.get("status"),
                "idempotency_key": scoped_key,
            }.items()
        ):
            raise ExactJoinError("shipment event request differs from World history")
        return {"shipment": projection}
    if action_kind == "commerce.resolve_shipment":
        shipment_id = _required_text(request.get("shipment_id"), "shipment_id")
        projection = _shipment_projection(state, shipment_id)
        if response_payload is not None and response_payload != {"shipment": projection}:
            raise ExactJoinError("shipment resolution response differs from World outcome")
        shipment = _one_after_write(writes, table="shipments", key=shipment_id)
        expected = {
            "resolution": request.get("resolution"),
            "replacement_sku_id": request.get("replacement_sku_id"),
            "resolved_by": exchange.decision.get("actor_id"),
            "resolution_idempotency_key": scoped_key,
        }
        if any(shipment.get(name) != value for name, value in expected.items()):
            raise ExactJoinError("shipment resolution request differs from World row")
        return {"shipment": projection}
    if action_kind == "platform.settle_payment":
        order_id = _required_text(request.get("order_id"), "order_id")
        order = _one_after_write(writes, table="orders", key=order_id)
        for name in ("order_id", "buyer_id", "merchant_id", "sku_id", "qty"):
            if order.get(name) != request.get(name):
                raise ExactJoinError(f"settlement World order changed {name!r}")
        if order.get("agreed_price") != request.get("agreed_price"):
            raise ExactJoinError("settlement World order changed agreed_price")
        ledger_rows = _after_writes(writes, table="ledger")
        if len(ledger_rows) != 1:
            raise ExactJoinError("settlement needs one exact World ledger receipt")
        ledger = ledger_rows[0]
        if commit.get("operation") == "partial_settle":
            allocation = _one_after_write(writes, table="fulfillments", key=order_id)
            expected_response = {
                "order_id": order_id,
                "txn_id": allocation.get("receipt_txn_id"),
                "status": order.get("state"),
                "requested_qty": allocation.get("requested_qty"),
                "fulfilled_qty": allocation.get("fulfilled_qty"),
                "backordered_qty": allocation.get("backordered_qty"),
            }
        else:
            expected_response = {
                "order_id": order_id,
                "txn_id": ledger.get("txn_id"),
                "status": "settled",
            }
        if response_payload is not None and response_payload != expected_response:
            raise ExactJoinError("settlement response differs from World transaction")
        return expected_response
    if action_kind == "platform.update_reputation":
        merchant_id = _required_text(
            request.get("merchant_id"), "reputation merchant_id"
        )
        order_id = _required_text(request.get("order_id"), "reputation order_id")
        txn_id = _required_text(request.get("txn_id"), "reputation txn_id")
        event_id = f"reputation-settlement:{txn_id}"
        reputation = _one_after_write(
            writes,
            table="reputation",
            key=merchant_id,
        )
        event = _one_after_write(
            writes,
            table="reputation_settlements",
            key=event_id,
        )
        if any(
            event.get(name) != expected
            for name, expected in {
                "event_id": event_id,
                "merchant_id": merchant_id,
                "order_id": order_id,
                "txn_id": txn_id,
                "outcome": reputation,
            }.items()
        ):
            raise ExactJoinError(
                "settlement reputation World event changed transaction identity"
            )
        sources = event.get("sources")
        expected_source = {
            "source_actor": "platform:psp",
            "source_request_id": exchange.request.get("msg_id"),
            "source_idempotency_key": exchange.request.get("idempotency_key"),
        }
        if sources != [expected_source]:
            raise ExactJoinError(
                "settlement reputation World event changed its causal source"
            )
        expected_response = {"merchant_id": merchant_id}
        if response_payload is not None and response_payload != expected_response:
            raise ExactJoinError(
                "settlement reputation response differs from World outcome"
            )
        return expected_response
    if action_kind == "commerce.allocate_fulfillment":
        allocation_id = _required_text(request.get("allocation_id"), "allocation_id")
        raw_order_ids = request.get("priority_order_ids")
        if not isinstance(raw_order_ids, list) or not raw_order_ids:
            raise ExactJoinError("allocation request has no priority order ids")
        allocations: list[dict[str, Any]] = []
        fulfillment_writes = {
            str(row.get("key")): row
            for row in writes.get("fulfillments", [])
            if isinstance(row, Mapping)
        }
        for order_id in raw_order_ids:
            write = fulfillment_writes.get(str(order_id))
            if not isinstance(write, Mapping) or not isinstance(write.get("after"), Mapping):
                raise ExactJoinError("allocation omitted a requested World fulfillment")
            allocation = dict(write["after"])
            if allocation.pop("allocation_id", None) != allocation_id:
                raise ExactJoinError("allocation World row changed allocation_id")
            allocations.append(allocation)
        expected_batch = {
            "allocation_id": allocation_id,
            "merchant_id": exchange.decision.get("actor_id"),
            "sku_id": request.get("sku_id"),
            "priority_order_ids": raw_order_ids,
            "allocations": allocations,
            "created_by": exchange.decision.get("actor_id"),
            "idempotency_key": scoped_key,
        }
        if response_payload is not None and response_payload != {
            "allocation_batch": expected_batch
        }:
            raise ExactJoinError("allocation response differs from World transaction")
        return {"allocation_batch": expected_batch}
    raise ExactJoinError(f"unsupported mutation response {action_kind!r}")


def _verify_settlement_reputation_links(
    operations: Sequence[VerifiedSupplyFulfillmentOperation],
) -> None:
    settlements = tuple(
        row for row in operations if row.action_kind == "platform.settle_payment"
    )
    reputation_updates = tuple(
        row for row in operations if row.action_kind == "platform.update_reputation"
    )
    claimed_settlements: set[str] = set()
    for update in reputation_updates:
        request = update.exchange.request
        payload = _payload(request)
        matches: list[VerifiedSupplyFulfillmentOperation] = []
        for settlement in settlements:
            settlement_payload = _settlement_commit_order_payload(
                settlement.commit
            )
            response_payload = (
                _payload(settlement.response)
                if settlement.response is not None
                else _verify_mutation_response(
                    settlement.exchange,
                    response=None,
                    commit=settlement.commit,
                    state={},
                    request_payload=settlement_payload,
                )
            )
            if (
                request.get("in_reply_to")
                == settlement.exchange.request.get("msg_id")
                and request.get("idempotency_key")
                == f"{settlement.exchange.request.get('idempotency_key')}:reputation"
                and payload.get("order_id") == settlement_payload.get("order_id")
                and payload.get("merchant_id") == settlement_payload.get("merchant_id")
                and payload.get("txn_id") == response_payload.get("txn_id")
            ):
                matches.append(settlement)
        if len(matches) != 1:
            raise ExactJoinError(
                "settlement reputation does not bind to one exact settled order"
            )
        settlement_commit_id = _required_text(
            matches[0].commit.get("commit_id"), "settlement commit_id"
        )
        if settlement_commit_id in claimed_settlements:
            raise ExactJoinError(
                "two settlement reputation updates claim one settled order"
            )
        if _commit_sequence(update.commit) <= _commit_sequence(matches[0].commit):
            raise ExactJoinError(
                "settlement reputation committed before its settled order"
            )
        claimed_settlements.add(settlement_commit_id)


def _settlement_commit_order_payload(
    commit: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the exact settled order written by one verified commit."""

    order_id = _required_text(commit.get("subject_id"), "settlement subject_id")
    order = _one_after_write(
        _writes_by_table(commit),
        table="orders",
        key=order_id,
    )
    return {
        field: order.get(field)
        for field in (
            "order_id",
            "buyer_id",
            "merchant_id",
            "sku_id",
            "qty",
            "agreed_price",
        )
    }


def verified_settlement_order(
    operation: VerifiedSupplyFulfillmentOperation,
) -> dict[str, Any]:
    """Return the World order proven by one verified settlement operation."""

    if (
        not isinstance(operation, VerifiedSupplyFulfillmentOperation)
        or operation.action_kind != "platform.settle_payment"
        or operation.operation not in {"settle", "partial_settle"}
    ):
        raise ExactJoinError("operation is not a verified settlement")
    return _settlement_commit_order_payload(operation.commit)


def _one_response(
    exchange: LinkedPlatformExchange,
    *,
    kind: str,
    sender: str,
    target: str | None = None,
    allowed_additional_kinds: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    matching = [
        row
        for row in exchange.responses
        if (row.get("action") or {}).get("kind") == kind
    ]
    additional = [
        str((row.get("action") or {}).get("kind"))
        for row in exchange.responses
        if (row.get("action") or {}).get("kind") != kind
    ]
    if len(matching) != 1 or any(
        value not in allowed_additional_kinds for value in additional
    ):
        raise ExactJoinError(f"{kind} requires one exact Platform response")
    response = matching[0]
    action = response.get("action")
    expected = {
        "from": sender,
        "to": exchange.decision.get("actor_id") if target is None else target,
        "in_reply_to": exchange.request.get("msg_id"),
        "idempotency_key": exchange.request.get("idempotency_key"),
    }
    if any(response.get(name) != value for name, value in expected.items()):
        raise ExactJoinError(f"{kind} response envelope identity changed")
    if not isinstance(action, Mapping) or action.get("kind") != kind:
        raise ExactJoinError(f"{kind} response has the wrong typed action")
    return dict(response)


def _verify_undelivered_mutation_response(
    exchange: LinkedPlatformExchange,
    *,
    contract: _CommitContract,
    payload: Mapping[str, Any],
) -> None:
    request = exchange.request
    suffix = {
        "apply_supply_event": "supply-applied",
        "record_shipment_status": "shipment-status",
        "resolve_shipment": "shipment-resolved",
        "allocate_orders_atomic": "allocation-batch",
        "partial_settle": "allocated",
        "settle": "settled",
    }.get(contract.operation)
    if suffix is None:
        raise ExactJoinError("undelivered mutation has no exact response schema")
    expected = {
        "msg_id": f"{request.get('msg_id')}:{suffix}",
        "ts": request.get("ts"),
        "from": exchange.decision.get("platform_endpoint"),
        "to": exchange.decision.get("actor_id"),
        "in_reply_to": request.get("msg_id"),
        "idempotency_key": request.get("idempotency_key"),
        "signature": None,
        "action": {
            "kind": contract.response_kind,
            "payload": dict(payload),
        },
    }
    decision = exchange.decision
    if (
        decision.get("response_kinds") != [contract.response_kind]
        or decision.get("response_msg_ids") != [expected["msg_id"]]
        or decision.get("response_sha256s") != [wire_envelope_sha256(expected)]
        or exchange.response_positions
    ):
        raise ExactJoinError(f"{contract.response_kind} undelivered response metadata changed")


def _supply_projection(state: Mapping[str, Any], sku_id: str) -> dict[str, Any]:
    inventory = state.get("inventory")
    catalog = state.get("catalog")
    if not isinstance(inventory, Mapping) or not isinstance(catalog, list):
        raise ExactJoinError("World supply tables have invalid shapes")
    row = inventory.get(sku_id)
    listing_matches = [
        value for value in catalog if isinstance(value, Mapping) and value.get("sku_id") == sku_id
    ]
    if not isinstance(row, Mapping) or len(listing_matches) != 1:
        raise ExactJoinError(f"World has no unique supply state for {sku_id!r}")
    listing = listing_matches[0]
    available = _integer(row.get("qty_available"), "qty_available")
    reserved = _integer(row.get("qty_reserved"), "qty_reserved")
    money = listing.get("list_price")
    amount = money.get("amount") if isinstance(money, Mapping) else None
    try:
        cents_decimal = (Decimal(str(amount)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ExactJoinError("World listing has invalid exact money") from exc
    return {
        "sku_id": sku_id,
        "merchant_id": row.get("merchant_id"),
        "available_qty": available - reserved,
        "reserved_qty": reserved,
        "eta_day": _integer(row.get("eta_day"), "eta_day"),
        "unit_price_cents": int(cents_decimal),
        "version": _integer(row.get("version"), "supply version"),
    }


def _listing_for_supply(
    state: Mapping[str, Any],
    sku_id: str,
) -> Mapping[str, Any]:
    catalog = state.get("catalog")
    if not isinstance(catalog, list):
        raise ExactJoinError("World catalog table has an invalid shape")
    matches = [
        row
        for row in catalog
        if isinstance(row, Mapping) and row.get("sku_id") == sku_id
    ]
    if len(matches) != 1:
        raise ExactJoinError(f"World has no unique listing for {sku_id!r}")
    return matches[0]


def _listing_currency(listing: Mapping[str, Any]) -> str:
    money = listing.get("list_price")
    currency = money.get("currency") if isinstance(money, Mapping) else None
    return _required_text(currency, "listing currency")


def _supply_purchase_option(
    state: Mapping[str, Any],
    projection: Mapping[str, Any],
    *,
    buyer_id: str,
    request_idempotency_key: str,
) -> dict[str, Any]:
    sku_id = _required_text(projection.get("sku_id"), "purchase option sku_id")
    authority_id = supply_purchase_authority_id(
        buyer_id=buyer_id,
        request_idempotency_key=request_idempotency_key,
        sku_id=sku_id,
    )
    raw_authorities = state.get("supply_purchase_authorities")
    if not isinstance(raw_authorities, list):
        raise ExactJoinError("World supply authority table has an invalid shape")
    matches = [
        row
        for row in raw_authorities
        if isinstance(row, Mapping) and row.get("authority_id") == authority_id
    ]
    if len(matches) != 1:
        raise ExactJoinError("World has no unique supply purchase authority")
    try:
        authority = coerce_supply_purchase_authority(matches[0])
    except (SchemaError, TypeError, ValueError) as exc:
        raise ExactJoinError("World supply purchase authority is invalid") from exc
    expected_binding = (
        buyer_id,
        sku_id,
        projection.get("merchant_id"),
        projection.get("unit_price_cents"),
        projection.get("available_qty"),
        projection.get("version"),
        request_idempotency_key,
    )
    observed_binding = (
        authority.buyer_id,
        authority.sku_id,
        authority.merchant_id,
        authority.unit_price_cents,
        authority.available_qty,
        authority.supply_version,
        authority.request_idempotency_key,
    )
    if observed_binding != expected_binding:
        raise ExactJoinError("supply purchase authority binding changed")
    return {
        "authority_id": authority_id,
        "authority_digest": authority.authority_digest,
        "sku_id": sku_id,
        "order_id": authority.order_id,
        "merchant_id": _required_text(
            projection.get("merchant_id"), "purchase option merchant_id"
        ),
        "unit_price_cents": _integer(
            projection.get("unit_price_cents"), "purchase option price"
        ),
        "currency": authority.currency,
        "available_qty": authority.available_qty,
        "supply_version": _integer(
            projection.get("version"), "purchase option supply version"
        ),
        "expires_at_tick": authority.expires_at_tick,
    }


def _shipment_replacement_options(
    state: Mapping[str, Any],
    shipment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    catalog = state.get("catalog")
    if not isinstance(catalog, list):
        raise ExactJoinError("World catalog table has an invalid shape")
    merchant_id = _required_text(
        shipment.get("merchant_id"), "shipment merchant_id"
    )
    original_sku_id = _required_text(
        shipment.get("original_sku_id"), "shipment original_sku_id"
    )
    candidates = sorted(
        (
            row
            for row in catalog
            if isinstance(row, Mapping)
            and row.get("merchant_id") == merchant_id
            and row.get("status", "active") == "active"
        ),
        key=lambda row: str(row.get("sku_id", "")),
    )[:64]
    options: list[dict[str, Any]] = []
    for listing in candidates:
        sku_id = _required_text(listing.get("sku_id"), "replacement sku_id")
        if sku_id == original_sku_id:
            continue
        projection = _supply_projection(state, sku_id)
        if _integer(projection.get("available_qty"), "replacement availability") <= 0:
            continue
        options.append({
            "sku_id": sku_id,
            "merchant_id": merchant_id,
            "available_qty": projection["available_qty"],
            "unit_price_cents": projection["unit_price_cents"],
            "currency": _listing_currency(listing),
            "supply_version": projection["version"],
        })
    return options


def _shipment_projection(state: Mapping[str, Any], shipment_id: str) -> dict[str, Any]:
    shipments = state.get("shipments")
    if not isinstance(shipments, list):
        raise ExactJoinError("World shipments table has an invalid shape")
    matches = [
        row
        for row in shipments
        if isinstance(row, Mapping) and row.get("shipment_id") == shipment_id
    ]
    if len(matches) != 1:
        raise ExactJoinError(f"World has no unique shipment {shipment_id!r}")
    row = matches[0]
    history = row.get("status_history")
    if not isinstance(history, list):
        raise ExactJoinError("World shipment status_history is invalid")
    public_history: list[dict[str, Any]] = []
    for value in history:
        if not isinstance(value, Mapping):
            raise ExactJoinError("World shipment status event is invalid")
        public_history.append(
            {
                "event_id": value.get("event_id"),
                "status": value.get("status"),
                "logical_time": value.get("logical_time"),
            }
        )
    return {
        "shipment_id": shipment_id,
        "order_id": row.get("order_id"),
        "buyer_id": row.get("buyer_id"),
        "merchant_id": row.get("merchant_id"),
        "original_sku_id": row.get("original_sku_id"),
        "status": row.get("status"),
        "status_history": public_history,
        "resolution": row.get("resolution"),
        "replacement_sku_id": row.get("replacement_sku_id"),
        "version": row.get("version"),
    }


def _writes_by_table(commit: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    raw = commit.get("table_writes")
    if not isinstance(raw, list):
        raise ExactJoinError("World commit has no table_writes")
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise ExactJoinError("World table write is invalid")
        table = _required_text(value.get("table"), "World write table")
        rows.setdefault(table, []).append(value)
    return rows


def _one_after_write(
    writes: Mapping[str, list[Mapping[str, Any]]],
    *,
    table: str,
    key: str,
) -> dict[str, Any]:
    matches = [row for row in writes.get(table, []) if row.get("key") == key]
    if len(matches) != 1 or not isinstance(matches[0].get("after"), Mapping):
        raise ExactJoinError(f"World commit has no unique {table!r}:{key!r} outcome")
    return dict(matches[0]["after"])


def _after_writes(
    writes: Mapping[str, list[Mapping[str, Any]]],
    *,
    table: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in writes.get(table, []):
        after = value.get("after")
        if not isinstance(after, Mapping):
            raise ExactJoinError(f"World {table!r} write has no after row")
        rows.append(dict(after))
    return rows


def _payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    action = envelope.get("action")
    payload = action.get("payload") if isinstance(action, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ExactJoinError("supply and fulfillment action payload must be an object")
    return dict(payload)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExactJoinError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExactJoinError(f"{label} must be an integer")
    return int(value)


def _commit_sequence(commit: Mapping[str, Any]) -> int:
    value = commit.get("sequence")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactJoinError("World transaction has an invalid sequence")
    return value


def _commit_sha256(commit: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            dict(commit),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExactJoinError("World commit is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _tables_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare World tables while treating row-array serialization as unordered."""

    if set(left) != set(right):
        return False
    for table in left:
        left_value = left[table]
        right_value = right[table]
        if isinstance(left_value, list) and isinstance(right_value, list):
            left_rows = Counter(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                for value in left_value
            )
            right_rows = Counter(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                for value in right_value
            )
            if left_rows != right_rows:
                return False
        elif left_value != right_value:
            return False
    return True


DEFAULT_OPERATION_EVIDENCE_REGISTRY.register(
    OperationEvidenceContract(
        SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT,
        verify_supply_fulfillment_evidence_contract,
    )
)


__all__ = [
    "SUPPLY_FULFILLMENT_ACTION_KINDS",
    "SUPPLY_FULFILLMENT_AUTHORITY_PAIRS",
    "SUPPLY_FULFILLMENT_EVIDENCE_CONTRACT",
    "VerifiedPreclaimedCommitAttestation",
    "VerifiedSupplyFulfillmentEvidence",
    "VerifiedSupplyFulfillmentOperation",
    "VerifiedSupplyFulfillmentRead",
    "governance_preclaimed_commit_attestations",
    "verified_settlement_order",
    "verify_settlement_reputation_followup_commits",
    "verify_supply_fulfillment_evidence_contract",
]
