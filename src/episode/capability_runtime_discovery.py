"""Reusable exact discovery prefixes for runtime benchmark scorers.

An evaluated agent may perform a valid search, or search and accept an offer,
before attempting the task family's primary operation.  These actions are
real CommerceWorld operations and therefore create authoritative World
commits.  Family scorers use this module to verify those commits with the
registered Search and Match contracts.  The commits can then be composed with
another operation contract without granting task-specific semantic credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from episode.capability_runtime import (
    RuntimeEvidenceBundleV2,
    RuntimeEvidenceError,
)
from runtime.authority_operation_evidence import (
    SEARCH_SESSION_EVIDENCE_CONTRACT,
    VerifiedSearchSessionJoin,
    VerifiedSearchSessionEvidence,
)
from runtime.match_certificate_evidence import (
    MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
    VerifiedMatchCertificateEvidence,
)


@dataclass(frozen=True, slots=True)
class VerifiedDiscoveryPrefixV2:
    """Optional Search and Match commits verified by their exact contracts."""

    commits: tuple[dict[str, Any], ...]
    search_sessions: tuple[VerifiedSearchSessionJoin, ...]
    search_verified: bool
    match_verified: bool
    search_error: str | None
    match_error: str | None

    @property
    def commit_ids(self) -> tuple[str, ...]:
        return tuple(str(row.get("commit_id")) for row in self.commits)

    @property
    def search_evidence(self) -> VerifiedSearchSessionEvidence | None:
        """Project every exact Search join, including Match-owned predecessors.

        The projection does not invoke another evidence contract.  Its joins
        come only from the already verified standalone Search and Match
        results, so a family scorer can inspect the task's discovery state
        without claiming the Match-owned Search commit a second time.
        """

        if not self.search_sessions:
            return None
        session_ids = tuple(
            str(row.session.get("session_id", "")) for row in self.search_sessions
        )
        return VerifiedSearchSessionEvidence(
            sessions=self.search_sessions,
            all_session_ids=session_ids,
            excluded_session_ids=(),
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "search_contract_verified": self.search_verified,
            "match_contract_verified": self.match_verified,
            "search_error": self.search_error,
            "match_error": self.match_error,
            "preclaimed_commit_ids": list(self.commit_ids),
        }


def verify_optional_discovery_prefix_v2(
    evidence: RuntimeEvidenceBundleV2,
    *,
    buyer_id: str,
) -> VerifiedDiscoveryPrefixV2:
    """Verify any discovery work performed by ``buyer_id``.

    A Match contract owns both its source Search commit and its Match commit.
    The standalone Search call therefore excludes sessions already covered by
    Match.  This keeps the formal global commit closure exactly once.
    """

    verified_match: VerifiedMatchCertificateEvidence | None = None
    match_error: str | None = None
    match_actions = tuple(
        row
        for row in evidence.actions(kind="commerce.accept_offer", actor_id=buyer_id)
        if row.get("to") == "platform:aggregator"
    )
    if match_actions:
        try:
            candidate = evidence.verified_operation_evidence(
                MATCH_CERTIFICATE_EVIDENCE_CONTRACT,
                options={
                    "expected_buyer_id": buyer_id,
                    "allow_rejected": True,
                },
            )
            if not isinstance(candidate, VerifiedMatchCertificateEvidence):
                raise RuntimeEvidenceError(
                    "match certificate evidence contract returned wrong type"
                )
            verified_match = candidate
        except RuntimeEvidenceError as exc:
            match_error = f"{type(exc).__name__}: {exc}"

    matched_session_ids = tuple(
        row.session.session_id
        for row in (() if verified_match is None else verified_match.certificates)
    )
    verified_search: VerifiedSearchSessionEvidence | None = None
    search_error: str | None = None
    search_actions = tuple(
        row
        for row in evidence.actions(kind="commerce.search", actor_id=buyer_id)
        if row.get("to") == "platform:aggregator"
    )
    if search_actions:
        try:
            candidate = evidence.verified_operation_evidence(
                SEARCH_SESSION_EVIDENCE_CONTRACT,
                options={"exclude_session_ids": matched_session_ids},
            )
            if not isinstance(candidate, VerifiedSearchSessionEvidence):
                raise RuntimeEvidenceError("search session evidence contract returned wrong type")
            verified_search = candidate
        except RuntimeEvidenceError as exc:
            search_error = f"{type(exc).__name__}: {exc}"

    commits = tuple(
        sorted(
            (
                *(
                    row.commit
                    for row in (() if verified_search is None else verified_search.sessions)
                ),
                *(
                    commit
                    for row in (() if verified_match is None else verified_match.certificates)
                    for commit in (row.search_commit, row.commit)
                ),
            ),
            key=lambda row: int(row.get("sequence", -1)),
        )
    )
    standalone_searches = (
        () if verified_search is None else verified_search.sessions
    )
    matched_searches = tuple(
        VerifiedSearchSessionJoin(
            exchange=row.search_exchanges[0],
            response=row.rank_responses[0],
            session=_strip_schema_versions(
                row.rank_responses[0]["action"]["payload"]["search_session"]
            ),
            commit=row.search_commit,
        )
        for row in (() if verified_match is None else verified_match.certificates)
    )
    search_sessions = tuple(
        sorted(
            (*standalone_searches, *matched_searches),
            key=lambda row: int(row.commit.get("sequence", -1)),
        )
    )
    return VerifiedDiscoveryPrefixV2(
        commits=commits,
        search_sessions=search_sessions,
        search_verified=verified_search is not None,
        match_verified=verified_match is not None,
        search_error=search_error,
        match_error=match_error,
    )


def _strip_schema_versions(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_schema_versions(child)
            for key, child in value.items()
            if key != "schema_version"
        }
    if isinstance(value, (list, tuple)):
        return [_strip_schema_versions(child) for child in value]
    return value


__all__ = [
    "VerifiedDiscoveryPrefixV2",
    "verify_optional_discovery_prefix_v2",
]
