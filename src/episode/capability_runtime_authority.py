"""Authority boundary checks for verified ACWorld benchmark scenarios.

Scenario fixtures may seed validated World rows and may inject external
principal or environment events.  They may not impersonate trusted Platform
or World services.  A platform response must be produced by the real Platform
service after validation, and every World mutation must enter through a core
World API.  Keeping this rule in the formal preflight prevents a task adapter
from manufacturing evidence that the environment itself did not produce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agents.turn import PUBLIC_WORLD_READ_ACTION_KINDS
from episode.scenario import population_for_scenario
from episode.types import ScenarioSpec


_TRUSTED_SERVICE_PREFIXES = ("platform:", "world:")

# These fields are not ordinary product attributes.  They name accepted agent
# answers or mutable commerce subsystems that need their own authority,
# lifecycle, persistence, idempotency, and replay contracts.  A task adapter
# may not hide them inside an envelope or ``Listing.attributes`` merely because
# the deterministic scorer can read them later.
_SCORER_SIDECAR_KEYS = frozenset(
    {
        "benchmark_submission",
        "_benchmark_submission",
        "governance_report",
    }
)
_CORE_DOMAIN_STATE_KEYS = frozenset(
    {
        "evidence_documents",
        "claim_decisions",
        "claim_corrections",
        "published_claims",
        "published_comparative_claims",
        "proposed_claims",
        "last_grounded_response",
        # Pricing, cart, and settlement policy are versioned World records.
        # A Listing may carry ordinary product facts, but not executable price
        # tiers, bundle programs, or fee rules that Platform silently treats
        # as authority.
        "pricing_tiers",
        "bundle_rules",
        "fee_rules",
        # Payment authorization/capture and fulfillment packing are lifecycle
        # records.  They cannot be represented as product metadata or actor
        # memory merely to make a cancellation scenario scoreable.
        "payment_state",
        "payment_authorization",
        "packing_state",
        "packing_record",
        # After-sales facts require their own actor-bound lifecycle records.
        "dispute_reason",
        "pending_runtime_capability",
        "review_evidence",
        "review_pollution_percent",
        "review_integrity_requests",
        "rejected_review_request_ids",
        "market_signals",
        "collusion_proposals",
        "rejected_collusion_proposal_ids",
        "reputation_history",
        "recency_weighted_score",
        "remediation_plan",
        "completed_remediation_steps",
        "sponsorship_campaign",
        "sponsorship_disclosed",
        "placement",
    }
)

# Formal scenarios may seed only tables that are hydrated through a validated
# CommerceWorld initialization path.  An unknown top-level key is otherwise a
# tempting place for a task adapter to keep the state machine that the real
# environment is missing.  Add a key here only together with its typed core
# records, validation, persistence, replay, and HTTP/VCP parity path.
_CORE_SCENARIO_INITIAL_STATE_KEYS = frozenset(
    {
        "catalog",
        "orders",
        "ledger",
        "logical_time",
        "order_timelines",
        "shipments",
        "reputation",
        "friendships",
        "reviews",
        "evidence_records",
        "order_settlement_setup",
        "after_sales_setup",
        "market_governance_setup",
        "mandate_authorities",
        "mandate_revisions",
        "listing_claims",
        "match_authorizations",
        "pricing_policy_fixtures",
    }
)


def runtime_scenario_authority_violations_v2(
    scenario: ScenarioSpec,
) -> tuple[str, ...]:
    """Return deterministic violations of the formal scenario authority rule.

    ``PopulationSpec.initial_events`` is an input boundary.  It may contain
    principal messages, authenticated buyer or merchant messages, and external
    environment observations.  It may not contain a pre-fabricated response
    from Platform or World.  External ``runtime:*`` observations must first be
    addressed to a Platform service and use a Platform action, so they cannot
    bypass validation or commit state directly.
    """

    issues: list[str] = []
    population = population_for_scenario(scenario)
    population_actor_ids = {
        *(buyer.buyer_id for buyer in population.buyers),
        *(merchant.merchant_id for merchant in population.merchants),
    }

    for key in sorted(set(scenario.initial_state) - _CORE_SCENARIO_INITIAL_STATE_KEYS):
        issues.append(
            f"initial_state.{key} has no registered CommerceWorld core "
            "initialization path"
        )

    # Scorer-only answer material is never legitimate scenario state, even if
    # it is nested under a table that the World knows how to hydrate.
    for path, key in _find_keys(scenario.initial_state, _SCORER_SIDECAR_KEYS):
        issues.append(
            "scenario carries scorer sidecar field "
            f"{_join_path('initial_state', path, key)!r}"
        )

    catalog = scenario.initial_state.get("catalog", ())
    if isinstance(catalog, Sequence) and not isinstance(catalog, (str, bytes)):
        for ordinal, row in enumerate(catalog):
            if not isinstance(row, Mapping):
                continue
            for path, key in _find_keys(
                row.get("attributes"),
                _SCORER_SIDECAR_KEYS | _CORE_DOMAIN_STATE_KEYS,
            ):
                issues.append(
                    f"initial_state.catalog[{ordinal}] stores core-owned state "
                    f"{_join_path('attributes', path, key)!r}"
                )

    # Actor-local memory is useful for private preferences and history.  It is
    # not an alternative persistence layer for orders, pricing, governance, or
    # after-sales state.  Those records must live in their typed World tables.
    actors = (*population.buyers, *population.merchants)
    for actor in actors:
        actor_id = getattr(actor, "buyer_id", None) or getattr(
            actor, "merchant_id", "unknown"
        )
        for path, key in _find_keys(
            actor.initial_state,
            _SCORER_SIDECAR_KEYS | _CORE_DOMAIN_STATE_KEYS,
        ):
            issues.append(
                f"actor {actor_id!r} initial_state stores core-owned state "
                f"{_join_path('initial_state', path, key)!r}"
            )

    for ordinal, event in enumerate(population.initial_events):
        label = f"initial_events[{ordinal}]"
        sender = str(event.get("from", event.get("from_", ""))).strip()
        recipient = str(event.get("to", "")).strip()
        action = event.get("action")
        kind = (
            str(action.get("kind", "")).strip()
            if isinstance(action, dict)
            else ""
        )

        if sender.startswith(_TRUSTED_SERVICE_PREFIXES):
            issues.append(
                f"{label} impersonates trusted service sender {sender!r}"
            )
        payload = action.get("payload") if isinstance(action, Mapping) else None
        owner_directive = bool(
            sender == "merchant:owner"
            and recipient in population_actor_ids
            and recipient.startswith("merchant:")
            and kind == "commerce.send_message"
            and isinstance(payload, Mapping)
            and payload.get("category") == "owner_directive"
        )
        if (
            sender.startswith(("buyer:", "merchant:"))
            and sender not in population_actor_ids
            and not owner_directive
        ):
            issues.append(
                f"{label} impersonates unregistered scenario actor {sender!r}"
            )
        if recipient == "world" or recipient.startswith("world:"):
            issues.append(f"{label} addresses World directly")
        if kind == "world" or kind.startswith("world."):
            issues.append(f"{label} invokes World action {kind!r} directly")
        if sender.startswith("runtime:"):
            if not recipient.startswith("platform:"):
                issues.append(
                    f"{label} external runtime event must enter through Platform"
                )
            if not kind.startswith("platform."):
                issues.append(
                    f"{label} external runtime event must use a Platform action"
                )

    return tuple(issues)


def runtime_evidence_core_ownership_violations_v2(
    evidence: Any,
) -> tuple[str, ...]:
    """Reject benchmark evidence that smuggles missing core state through data.

    This is deliberately separate from the sender-authority scan above.  A
    scenario can use only legitimate buyer and merchant senders and still
    bypass CommerceWorld by attaching its answer to a normal commerce action,
    or by treating claims, reviews, campaigns, disputes, and remediation as
    arbitrary catalog attributes.  Formal preflight calls this check on the
    *executed* episode, so both scripted reference policies and live model
    actions are covered.

    The check is a stop gate, not a scorer.  It never repairs evidence or
    interprets the desired answer.  The listed state must instead be carried by
    a typed Runtime, Platform, or World subsystem before the task is eligible.
    """

    issues: list[str] = []
    envelopes = getattr(evidence, "envelopes", ())
    trace_rows = getattr(evidence, "trace_rows", ())
    framework_read_request_ids = {
        str(envelope.get("in_reply_to"))
        for envelope in envelopes
        if isinstance(envelope, Mapping)
        and envelope.get("from") == "world"
        and isinstance(envelope.get("in_reply_to"), str)
        and isinstance(envelope.get("action"), Mapping)
        and envelope["action"].get("kind") == "world.response"
    }
    # A scoreable protocol failure can stop an episode after Runtime emitted a
    # public read but before World returned its response.  Tracker records that
    # exact Runtime-generated request id as pending.  Treat it as framework
    # transport without weakening the ban on actor-authored World envelopes.
    framework_read_request_ids.update(
        _pending_framework_world_read_ids(trace_rows)
    )
    for ordinal, envelope in enumerate(envelopes):
        if not isinstance(envelope, Mapping):
            continue
        sender = str(envelope.get("from", envelope.get("from_", ""))).strip()
        recipient = str(envelope.get("to", "")).strip()
        action = envelope.get("action")
        kind = (
            str(action.get("kind", "")).strip()
            if isinstance(action, Mapping)
            else ""
        )
        if sender.startswith(("buyer:", "merchant:")):
            framework_read = _is_framework_world_read(
                envelope,
                kind=kind,
                recipient=recipient,
                framework_request_ids=framework_read_request_ids,
            )
            if (
                recipient == "world" or recipient.startswith("world:")
            ) and not framework_read:
                issues.append(
                    f"audit envelope[{ordinal}] lets actor {sender!r} address "
                    "World directly"
                )
            if (kind == "world" or kind.startswith("world.")) and not framework_read:
                issues.append(
                    f"audit envelope[{ordinal}] lets actor {sender!r} invoke "
                    f"World action {kind!r} directly"
                )
        payload = action.get("payload") if isinstance(action, Mapping) else None
        for path, key in _find_keys(payload, _SCORER_SIDECAR_KEYS):
            issues.append(
                f"audit envelope[{ordinal}] carries scorer sidecar field "
                f"{_join_path('action.payload', path, key)!r}"
            )

    # Agent memory and tool traces are useful observations, but they are not an
    # authority boundary.  In particular, a task adapter may not hide its
    # accepted answer in a memory write and later let the scorer treat that
    # write as if Platform or World had validated it.  Runtime-owned actor
    # result records are the supported channel for semantic agent evidence.
    for ordinal, row in enumerate(trace_rows):
        for path, key in _find_keys(row, _SCORER_SIDECAR_KEYS):
            issues.append(
                f"trace row[{ordinal}] carries scorer sidecar field "
                f"{_join_path('trace', path, key)!r}"
            )

    for phase, snapshot in (
        ("initial", getattr(evidence, "initial_world", None)),
        ("final", getattr(evidence, "final_world", None)),
    ):
        if not isinstance(snapshot, Mapping):
            continue
        tables = snapshot.get("tables")
        catalog = tables.get("catalog") if isinstance(tables, Mapping) else None
        if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
            continue
        for ordinal, row in enumerate(catalog):
            if not isinstance(row, Mapping):
                continue
            attributes = row.get("attributes")
            for path, key in _find_keys(
                attributes,
                _SCORER_SIDECAR_KEYS | _CORE_DOMAIN_STATE_KEYS,
            ):
                issues.append(
                    f"{phase} World catalog[{ordinal}] stores core-owned state "
                    f"{_join_path('attributes', path, key)!r}"
                )
            if (
                isinstance(attributes, Mapping)
                and attributes.get("benchmark_t3_entity") == "authoritative_record"
            ):
                issues.append(
                    f"{phase} World catalog[{ordinal}] represents a non-product "
                    "authoritative record as a Listing"
                )

    # Initial and final snapshots commonly contain the same seeded violation.
    # Preserve deterministic first-observation order without flooding reports.
    return tuple(dict.fromkeys(issues))


def _is_framework_world_read(
    envelope: Mapping[str, Any],
    *,
    kind: str,
    recipient: str,
    framework_request_ids: set[str],
) -> bool:
    """Recognize only Runtime-generated public read-tool transport.

    Actor-authored commerce envelopes cannot choose the deterministic
    ``<parent>:read:<ordinal>`` message identity produced by
    :func:`agents.turn.tool_call_to_read_envelope`.  Requiring the exact
    recipient, read whitelist, identity/idempotency binding, causal parent,
    and either a matching World response or Runtime Tracker pending-read row
    keeps all direct writes and forged reads fail closed while admitting
    legitimate HTTP/re-entrant grounding and scoreable interrupted reads.
    """

    if recipient != "world" or kind not in PUBLIC_WORLD_READ_ACTION_KINDS:
        return False
    msg_id = envelope.get("msg_id")
    parent = envelope.get("in_reply_to")
    if not isinstance(msg_id, str) or not isinstance(parent, str) or not parent:
        return False
    prefix = f"{parent}:read:"
    if not msg_id.startswith(prefix) or not msg_id.removeprefix(prefix).isdigit():
        return False
    return (
        envelope.get("idempotency_key") == msg_id
        and msg_id in framework_request_ids
    )


def _pending_framework_world_read_ids(trace_rows: Any) -> set[str]:
    """Return Runtime-tracked public reads interrupted before World replied."""

    request_ids: set[str] = set()
    if not isinstance(trace_rows, Sequence) or isinstance(
        trace_rows, (str, bytes)
    ):
        return request_ids
    for row in trace_rows:
        if not isinstance(row, Mapping):
            continue
        steps = row.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            continue
        for step in steps:
            if not isinstance(step, Mapping) or step.get("kind") != "tool_call":
                continue
            data = step.get("data")
            results = data.get("results") if isinstance(data, Mapping) else None
            if not isinstance(results, Sequence) or isinstance(
                results, (str, bytes)
            ):
                continue
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                tool = result.get("tool")
                request_id = result.get("pending_request_msg_id")
                if (
                    result.get("status") == "pending_response"
                    and isinstance(tool, str)
                    and tool.startswith("world.")
                    and isinstance(request_id, str)
                    and request_id
                ):
                    request_ids.add(request_id)
    return request_ids


def _find_keys(
    value: Any,
    forbidden: frozenset[str],
    *,
    path: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, ...], str], ...]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            if key in forbidden:
                found.append((path, key))
            found.extend(
                _find_keys(nested, forbidden, path=(*path, key))
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for ordinal, nested in enumerate(value):
            found.extend(
                _find_keys(nested, forbidden, path=(*path, f"[{ordinal}]"))
            )
    return tuple(found)


def _join_path(prefix: str, path: tuple[str, ...], key: str) -> str:
    pieces = [prefix, *path, key]
    rendered = pieces[0]
    for piece in pieces[1:]:
        rendered += piece if piece.startswith("[") else f".{piece}"
    return rendered


__all__ = [
    "runtime_evidence_core_ownership_violations_v2",
    "runtime_scenario_authority_violations_v2",
]
