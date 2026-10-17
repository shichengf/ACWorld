"""Exact closure between verified operation joins and World transactions.

Operation-specific evidence contracts prove that an accepted Platform request
caused a particular World commit.  A scorer must additionally prove the
converse: every transaction in the episode was claimed by one of those
verified operations.  Otherwise an unrelated durable mutation can coexist
with an otherwise valid task trace and remain ambient to the rubric.

This module is deliberately operation agnostic.  Family scorers provide the
commits returned by their exact joins and the operation/authority pairs that
are legal for that family.  The verifier rejects missing, duplicate,
mismatched, disallowed, and unclaimed transaction commits.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


AuthorityPair = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ExactCommitClaimVerdict:
    """Serializable transaction-closure verdict for one scorer lane."""

    verified: bool
    transaction_commit_ids: tuple[str, ...]
    claimed_commit_ids: tuple[str, ...]
    unclaimed_commit_ids: tuple[str, ...]
    foreign_claim_ids: tuple[str, ...]
    duplicate_transaction_ids: tuple[str, ...]
    duplicate_claim_ids: tuple[str, ...]
    mismatched_claim_ids: tuple[str, ...]
    disallowed_commit_ids: tuple[str, ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "transaction_commit_ids": list(self.transaction_commit_ids),
            "claimed_commit_ids": list(self.claimed_commit_ids),
            "unclaimed_commit_ids": list(self.unclaimed_commit_ids),
            "foreign_claim_ids": list(self.foreign_claim_ids),
            "duplicate_transaction_ids": list(self.duplicate_transaction_ids),
            "duplicate_claim_ids": list(self.duplicate_claim_ids),
            "mismatched_claim_ids": list(self.mismatched_claim_ids),
            "disallowed_commit_ids": list(self.disallowed_commit_ids),
            "issues": list(self.issues),
        }


def verify_exact_transaction_commit_claims(
    world_commits: Iterable[Mapping[str, Any]],
    claimed_commits: Iterable[Mapping[str, Any]],
    *,
    allowed_authority_pairs: Iterable[AuthorityPair],
) -> ExactCommitClaimVerdict:
    """Require a one-to-one claim for every transaction World commit.

    ``claimed_commits`` must be the exact commit objects returned or selected
    by independently verified causal joins.  Merely naming an allowed
    operation is insufficient.  Each claim must equal the journal row with the
    same ``commit_id``, and the claimed-id set must equal the complete
    transaction-id set.
    """

    allowed = frozenset(allowed_authority_pairs)
    if not allowed or any(
        not isinstance(operation, str)
        or not operation
        or not isinstance(authority, str)
        or not authority
        for operation, authority in allowed
    ):
        raise ValueError("allowed authority pairs must be non-empty text pairs")

    transactions = tuple(
        dict(row) for row in world_commits if row.get("commit_kind") == "transaction"
    )
    claims = tuple(dict(row) for row in claimed_commits)
    transaction_ids = tuple(_commit_id(row) for row in transactions)
    claim_ids = tuple(_commit_id(row) for row in claims)
    duplicate_transactions = _duplicates(transaction_ids)
    duplicate_claims = _duplicates(claim_ids)

    transaction_by_id = {
        commit_id: row
        for commit_id, row in zip(transaction_ids, transactions, strict=True)
        if commit_id
    }
    claim_by_id = {
        commit_id: row
        for commit_id, row in zip(claim_ids, claims, strict=True)
        if commit_id
    }
    transaction_id_set = set(transaction_by_id)
    claim_id_set = set(claim_by_id)
    foreign_claims = tuple(sorted(claim_id_set - transaction_id_set))
    unclaimed = tuple(sorted(transaction_id_set - claim_id_set))
    mismatched = tuple(
        sorted(
            commit_id
            for commit_id in transaction_id_set & claim_id_set
            if claim_by_id[commit_id] != transaction_by_id[commit_id]
            or claim_by_id[commit_id].get("commit_kind") != "transaction"
        )
    )
    disallowed = tuple(
        sorted(
            commit_id
            for commit_id, row in transaction_by_id.items()
            if (str(row.get("operation", "")), str(row.get("authority_action", "")))
            not in allowed
        )
    )

    issues: list[str] = []
    if "" in transaction_ids:
        issues.append("transaction commit is missing a commit_id")
    if "" in claim_ids:
        issues.append("claimed commit is missing a commit_id")
    if duplicate_transactions:
        issues.append("transaction commit ids are not unique")
    if duplicate_claims:
        issues.append("a transaction commit was claimed more than once")
    if foreign_claims:
        issues.append("a claimed commit is absent from the World journal")
    if mismatched:
        issues.append("a claimed commit differs from its World journal row")
    if disallowed:
        issues.append("the World journal contains a disallowed transaction authority")
    if unclaimed:
        issues.append("the World journal contains an unclaimed transaction commit")

    return ExactCommitClaimVerdict(
        verified=not issues,
        transaction_commit_ids=tuple(sorted(transaction_id_set)),
        claimed_commit_ids=tuple(sorted(claim_id_set)),
        unclaimed_commit_ids=unclaimed,
        foreign_claim_ids=foreign_claims,
        duplicate_transaction_ids=duplicate_transactions,
        duplicate_claim_ids=duplicate_claims,
        mismatched_claim_ids=mismatched,
        disallowed_commit_ids=disallowed,
        issues=tuple(issues),
    )


def _commit_id(row: Mapping[str, Any]) -> str:
    value = row.get("commit_id")
    return value if isinstance(value, str) and value else ""


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(value for value, count in Counter(values).items() if value and count > 1)
    )


__all__ = [
    "AuthorityPair",
    "ExactCommitClaimVerdict",
    "verify_exact_transaction_commit_claims",
]
