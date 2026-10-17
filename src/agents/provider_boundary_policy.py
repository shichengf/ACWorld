"""Frozen field policy for the CommerceWorld provider boundary.

The names in this module are reserved benchmark/runtime metadata namespaces,
not fuzzy substrings.  In particular, business fields such as
``product_family`` and ``expected_delivery_at`` are intentionally outside the
set, while the exact benchmark fields ``family`` and ``difficulty`` are not
provider-visible.

Business resource identities follow the separate public-reference policy:
the Agent projects authenticated ``*_id``/``*_ids`` values to opaque
``*_ref``/``*_refs`` values before a request reaches a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


BENCHMARK_INTERNAL_METADATA_FIELDS_V1 = frozenset(
    {
        "benchmark_task_id",
        "capability",
        "capability_id",
        "difficulty",
        "difficulty_axis",
        "evaluated_role",
        "family",
        "lane",
        "mutation",
        "mutation_id",
        "mutation_label",
        "oracle_id",
        "task_id",
    }
)

# ``benchmark_`` is an internal namespace, except for the two structural
# containers which are projected through their own field allowlists before the
# final AgentPhase boundary.  A catalog attribute such as
# ``benchmark_t3_entity`` is never a business fact for the model.
_PUBLIC_PROJECTION_CONTAINER_FIELDS_V1 = frozenset({"benchmark_contract", "benchmark_facts"})


def benchmark_internal_namespace_field_v1(field_name: str) -> bool:
    normalized = str(field_name).casefold()
    return normalized.startswith("benchmark_") and (
        normalized not in _PUBLIC_PROJECTION_CONTAINER_FIELDS_V1
    )


_BENCHMARK_CONTEXT_PATH_SUFFIXES_V1 = (
    ("task_context",),
    ("benchmark_contract",),
    ("task_facts",),
    ("benchmark_facts",),
    ("persistent_task_business_facts", "task"),
    ("persistent_task_business_facts", "benchmark"),
)

# These exact business-entity paths use words that are also benchmark labels.
# Keep the allowlist intentionally small: a top-level ``task_facts.family`` is
# metadata, while ``task_facts.product.family`` is an ordinary catalog fact.
_PUBLIC_BUSINESS_METADATA_FIELD_PARENT_SUFFIXES_V1 = {
    "family": (
        ("product",),
        ("product_profile",),
    ),
    "difficulty": (
        ("product",),
        ("product_profile",),
    ),
}


def benchmark_metadata_field_is_private_v1(
    parent_path: Sequence[str],
    field_name: str,
) -> bool:
    """Return whether ``field_name`` is metadata at a reserved context path.

    The same spelling outside these paths is ordinary business data.  This is
    what keeps, for example, a product's public ``family`` or a handling
    instruction's public ``difficulty`` while removing benchmark labels from
    ``task_context`` and ``benchmark_contract``.
    """

    normalized_path = tuple(str(part).casefold() for part in parent_path)
    normalized_field = str(field_name).casefold()
    if normalized_field not in BENCHMARK_INTERNAL_METADATA_FIELDS_V1:
        return False
    if any(
        len(normalized_path) >= len(suffix) and normalized_path[-len(suffix) :] == suffix
        for suffix in _PUBLIC_BUSINESS_METADATA_FIELD_PARENT_SUFFIXES_V1.get(
            normalized_field,
            (),
        )
    ):
        return False
    # Once a value enters a reserved benchmark/task namespace, metadata stays
    # private at every descendant depth.  Matching only the immediate parent
    # would let ``task_facts.nested.capability`` bypass the boundary.  The
    # namespace itself may be nested below request/observation containers, so
    # search for each exact path sequence rather than relying on a root offset.
    return any(
        any(
            normalized_path[index : index + len(namespace)] == namespace
            for index in range(len(normalized_path) - len(namespace) + 1)
        )
        for namespace in _BENCHMARK_CONTEXT_PATH_SUFFIXES_V1
        if len(normalized_path) >= len(namespace)
    )


# Public business concurrency/state facts.  These exact spellings are useful
# to a model comparing visible offers, inventory, policy, or fulfilment state.
# Similar-looking framework values remain private by default; in particular a
# generic ``revision``/``version`` or an unlisted ``*_version`` is not allowed.
PROVIDER_PUBLIC_BUSINESS_STATE_FIELDS_V1 = frozenset(
    {
        "catalog_revision",
        "inventory_revision",
        "state_revision",
        "catalog_version",
        "inventory_version",
        "listing_version",
        "order_version",
        "plan_version",
        "policy_version",
        "refund_version",
        "resolution_version",
        "return_version",
        "shipment_version",
        "supply_version",
    }
)

_PROVIDER_RECORD_CONTEXT_FIELDS_V1 = frozenset(
    {
        "evidence",
        "evidence_record",
        "evidence_records",
        "provenance",
        "provenance_record",
        "provenance_records",
        "source",
        "source_record",
        "source_records",
    }
)

_PROVIDER_PRIVATE_INTEGRITY_FIELDS_V1 = frozenset(
    {
        "authority_digests",
        "digest",
        "fingerprint",
        "hash",
        "idempotency_key",
        "idempotency_keys",
        "offer_digests",
        "replay",
        "replay_hashes",
        "request_fingerprint",
        "revision",
        "session_digests",
        "source_digests",
        "source_hashes",
        "version",
    }
)

_PROVIDER_PRIVATE_INTEGRITY_SUFFIXES_V1 = (
    "_digest",
    "_fingerprint",
    "_hash",
    "_idempotency_key",
    "_revision",
    "_sha256",
    "_version",
)


def provider_boundary_semantic_path_v1(
    parent_path: Sequence[str],
    value: Any,
) -> tuple[str, ...]:
    """Annotate a canonical evidence-record mapping for path-aware policy.

    A World evidence read can arrive under a generic ``result`` container, so
    its parent key alone is insufficient.  The canonical record schema (or
    its already-ref-projected equivalent) supplies a narrow structural marker
    without allowing arbitrary objects named ``result`` to expose digests.
    """

    normalized_path = tuple(str(part).casefold() for part in parent_path)
    if not isinstance(value, Mapping):
        return normalized_path
    keys = {str(key).casefold() for key in value}
    canonical_schema = value.get("schema_id") == "cwe.evidence-record.v1"
    projected_record = (
        "record_digest" in keys
        and "kind" in keys
        and "facts" in keys
        and "trust" in keys
        and bool({"record_id", "record_ref"} & keys)
    )
    if (canonical_schema or projected_record) and (
        not normalized_path or normalized_path[-1] != "evidence_record"
    ):
        return (*normalized_path, "evidence_record")
    return normalized_path


def provider_boundary_field_is_private_v1(
    parent_path: Sequence[str],
    field_name: str,
) -> bool:
    """Classify integrity/version fields at the final provider boundary.

    The decision is deliberately based on the exact business field and its
    parent namespace, never just a broad suffix allow.  ``record_digest`` and
    the canonical record's plain ``version`` are public only on an evidence,
    source, or provenance record.  Authority/session/offer/source/replay
    digests and hashes therefore remain private in every context.
    """

    normalized_path = tuple(str(part).casefold() for part in parent_path)
    normalized = str(field_name).casefold()
    if normalized == "schema_version":
        return False
    if normalized in PROVIDER_PUBLIC_BUSINESS_STATE_FIELDS_V1:
        return False
    record_context = bool(
        normalized_path and normalized_path[-1] in _PROVIDER_RECORD_CONTEXT_FIELDS_V1
    )
    if normalized == "record_digest":
        return not record_context
    if normalized == "version" and record_context:
        return False
    return bool(
        normalized in _PROVIDER_PRIVATE_INTEGRITY_FIELDS_V1
        or normalized.endswith(_PROVIDER_PRIVATE_INTEGRITY_SUFFIXES_V1)
    )


_REPUTATION_HISTORY_PATH_SUFFIX_V1 = (
    "ranking_context_projection",
    "candidate_annotations",
    "reputation",
)


def provider_boundary_public_field_name_v1(
    parent_path: Sequence[str],
    field_name: str,
) -> str:
    """Return the one path-bound semantic rename at the provider boundary.

    A ranking projection's contiguous reputation-event ``version`` is a
    public history count used by the business policy.  Generic versions remain
    private; even another object named ``reputation`` does not qualify.
    """

    normalized_path = tuple(str(part).casefold() for part in parent_path)
    normalized = str(field_name).casefold()
    if (
        normalized == "version"
        and len(normalized_path) >= len(_REPUTATION_HISTORY_PATH_SUFFIX_V1)
        and normalized_path[-len(_REPUTATION_HISTORY_PATH_SUFFIX_V1) :]
        == _REPUTATION_HISTORY_PATH_SUFFIX_V1
    ):
        return "history_count"
    return str(field_name)


__all__ = [
    "BENCHMARK_INTERNAL_METADATA_FIELDS_V1",
    "PROVIDER_PUBLIC_BUSINESS_STATE_FIELDS_V1",
    "benchmark_internal_namespace_field_v1",
    "benchmark_metadata_field_is_private_v1",
    "provider_boundary_field_is_private_v1",
    "provider_boundary_public_field_name_v1",
    "provider_boundary_semantic_path_v1",
]
