"""Immutable registry for the 200-task ACWorld capability benchmark.

V2 deliberately separates a *task* from a stochastic sample: every task id
names one fixed, role-conditioned evaluation problem.  Scenario materializers
may derive private determinism keys from these definitions, but this public
registry contains no seed axis.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

from episode.benchmark import BenchmarkTrack, TaskFamily, VARIANT_REGISTRY


TASK_VERSION_V2 = "2.0"
REGISTRY_SCHEMA_V2 = "cwe.benchmark-task-registry.v2"
EvaluatedRoleV2 = Literal["buyer", "merchant"]
MeasurementTagV2 = Literal[
    "decision",
    "policy_application",
    "stateful_execution",
    "security_integrity",
]
FactorValue: TypeAlias = str | int | float | bool
DifficultyFactors: TypeAlias = tuple[tuple[str, FactorValue], ...]


@dataclass(frozen=True)
class CapabilityGroupV2:
    """One semantically coherent ability with an internal difficulty order."""

    capability_id: str
    family: TaskFamily
    name: str
    evaluated_role: EvaluatedRoleV2
    template_variant_ids: tuple[str, ...]
    difficulty_profiles: tuple[DifficultyFactors, ...]
    populations: tuple[tuple[int, int], ...]
    measurement_tags: tuple[MeasurementTagV2, ...]

    @property
    def task_count(self) -> int:
        return len(self.difficulty_profiles)

    def __post_init__(self) -> None:
        count = self.task_count
        if count < 1:
            raise ValueError(f"capability {self.capability_id!r} has no tasks")
        if len(self.template_variant_ids) not in {1, count}:
            raise ValueError("template count must be one or equal to task count")
        if len(self.populations) not in {1, count}:
            raise ValueError("population count must be one or equal to task count")
        if not self.measurement_tags or len(set(self.measurement_tags)) != len(
            self.measurement_tags
        ):
            raise ValueError("measurement tags must be non-empty and unique")


@dataclass(frozen=True)
class TaskDefinitionV2:
    """One immutable, fixed benchmark problem (not a seeded sample)."""

    task_id: str
    task_version: str
    family: TaskFamily
    capability_id: str
    evaluated_role: EvaluatedRoleV2
    difficulty_rank: int
    difficulty_factors: DifficultyFactors
    template_variant_id: str
    buyers: int = 1
    merchants: int = 1

    @property
    def capability(self) -> str:
        """Short compatibility alias for paper/report code."""

        return self.capability_id

    def __post_init__(self) -> None:
        if self.evaluated_role not in {"buyer", "merchant"}:
            raise ValueError(f"unsupported evaluated role: {self.evaluated_role!r}")
        if self.difficulty_rank < 1:
            raise ValueError("difficulty_rank must be positive")
        if self.buyers < 1 or self.merchants < 1:
            raise ValueError("population sizes must be positive")
        keys = [key for key, _ in self.difficulty_factors]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("difficulty_factors must have unique, non-empty keys")

    def to_dict(self) -> dict[str, Any]:
        """Return the public, seed-free wire representation."""

        return {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "family": self.family.value,
            "capability_id": self.capability_id,
            "evaluated_role": self.evaluated_role,
            "difficulty_rank": self.difficulty_rank,
            "difficulty_factors": dict(self.difficulty_factors),
            "template_variant_id": self.template_variant_id,
            "population": {"buyers": self.buyers, "merchants": self.merchants},
        }

    @property
    def canonical_hash(self) -> str:
        body = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()


def _profiles(axis: str, values: tuple[FactorValue, ...]) -> tuple[DifficultyFactors, ...]:
    return tuple(
        (("difficulty_level", rank), (axis, value))
        for rank, value in enumerate(values, start=1)
    )


def _g(
    capability_id: str,
    family: TaskFamily,
    name: str,
    role: EvaluatedRoleV2,
    template: str | tuple[str, ...],
    axis: str,
    values: tuple[FactorValue, ...],
    *,
    population: tuple[int, int] | tuple[tuple[int, int], ...] = (1, 1),
) -> CapabilityGroupV2:
    templates = (template,) if isinstance(template, str) else template
    populations = (
        (population,) if population and isinstance(population[0], int) else population
    )
    return CapabilityGroupV2(
        capability_id=capability_id,
        family=family,
        name=name,
        evaluated_role=role,
        template_variant_ids=templates,
        difficulty_profiles=_profiles(axis, values),
        populations=populations,  # type: ignore[arg-type]
        measurement_tags=_measurement_tags_for(capability_id, family, role),
    )


def _measurement_tags_for(
    capability_id: str,
    family: TaskFamily,
    role: EvaluatedRoleV2,
) -> tuple[MeasurementTagV2, ...]:
    """Describe the construct measured without inflating capability claims.

    Tags are deliberately broad and may overlap.  They distinguish a model's
    commercial choice from public-policy application, stateful execution, and
    security/integrity judgment; they are not additional scored dimensions.
    """

    family_id = family.value
    if family_id in {"T1", "T3"}:
        return ("decision",)
    if family_id == "T2":
        return (
            ("decision", "policy_application")
            if role == "buyer"
            else ("policy_application", "stateful_execution")
        )
    if family_id == "T4":
        return (
            ("decision", "security_integrity")
            if "private_value" in capability_id
            else ("decision", "stateful_execution")
        )
    if family_id == "T5":
        return (
            ("decision", "stateful_execution")
            if role == "buyer"
            else ("policy_application", "stateful_execution")
        )
    if family_id == "T6":
        return (
            ("decision", "stateful_execution")
            if role == "buyer"
            else ("policy_application", "stateful_execution")
        )
    if family_id == "T7":
        return ("decision", "stateful_execution")
    if family_id == "T8":
        return (
            ("decision", "policy_application")
            if role == "buyer"
            else ("policy_application", "security_integrity")
        )
    if family_id == "T9":
        return ("decision", "security_integrity")
    if family_id == "T10":
        return ("stateful_execution", "security_integrity")
    raise ValueError(f"unsupported measurement family: {family_id!r}")


T1 = TaskFamily.PRODUCT_DISCOVERY
T2 = TaskFamily.PRODUCT_GROUNDING
T3 = TaskFamily.PREFERENCE_SOCIAL
T4 = TaskFamily.NEGOTIATION_PRIVACY
T5 = TaskFamily.CART_QUANTITY
T6 = TaskFamily.INVENTORY_FULFILLMENT
T7 = TaskFamily.ORDER_AFTERSALES
T8 = TaskFamily.MARKET_GOVERNANCE
T9 = TaskFamily.ADVERSARIAL_CONTENT
T10 = TaskFamily.TRANSACTION_INTEGRITY


# Only the accepted upper difficulty tiers in T4, T6, and T7 use the
# locally hardened contracts and rubrics.  Keeping this map beside the task
# registry makes the isolation boundary reviewable and prevents a family-wide
# scorer change from silently affecting the remaining benchmark.
HARDENED_CAPABILITY_RANKS_V2: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "t4.buyer_zopa": (2, 3),
        "t4.buyer_no_zopa": (2,),
        "t4.buyer_private_value": (2,),
        "t4.buyer_false_anchor": (2,),
        "t4.buyer_deadline": (1,),
        "t4.merchant_zopa": (2, 3),
        "t4.merchant_no_zopa": (2,),
        "t4.merchant_private_value": (2,),
        "t4.merchant_false_anchor": (2,),
        "t4.merchant_deadline": (1,),
        "t6.buyer_partial_backorder": (2,),
        "t6.buyer_stock_substitution": (2,),
        "t6.buyer_delivery_exception": (2,),
        "t6.buyer_restock_price": (2,),
        "t6.merchant_competing_commitment": (2, 3),
        "t6.merchant_partial_backorder": (2, 3),
        "t6.merchant_delivery_exception": (2,),
        "t7.buyer_cancel": (2,),
        "t7.buyer_return_refund": (2,),
        "t7.buyer_exchange": (2,),
        "t7.buyer_dispute": (2,),
        "t7.merchant_cancel": (2,),
        "t7.merchant_return_authorization": (2,),
        "t7.merchant_refund": (2,),
        "t7.merchant_exchange": (2,),
        "t7.merchant_dispute": (2,),
        "t7.merchant_ledger_close": (2,),
    }
)


def is_hardened_task_v2(definition: TaskDefinitionV2) -> bool:
    """Return whether one task is in the explicitly isolated hard tier."""

    return definition.difficulty_rank in HARDENED_CAPABILITY_RANKS_V2.get(
        definition.capability_id,
        (),
    )


# Group order is normative: it assigns the final two digits of each task id.
CAPABILITY_GROUPS_V2: tuple[CapabilityGroupV2, ...] = (
    # T1 -- feasibility is a buyer decision; merchant truth/stock belong to T2/T6.
    _g("t1.basic_feasible_discovery", T1, "Basic feasible discovery", "buyer", "S1", "candidate_count", (3, 5, 8)),
    _g("t1.hard_constraint_filtering", T1, "Multiple hard constraints", "buyer", "S2", "hard_constraint_count", (2, 3, 4, 5)),
    _g("t1.best_feasible_selection", T1, "Best item in the feasible set", "buyer", "S2", "candidate_count", (5, 8, 10, 15, 20)),
    _g("t1.correct_abstention", T1, "Abstain when no item is feasible", "buyer", "S10", "conflict_subtlety", (1, 2, 3, 4)),
    _g("t1.query_reformulation", T1, "Constraint-preserving re-query", "buyer", "S2", "search_rounds", (2, 3, 4, 5)),

    # T2 -- evidence and claim correctness, not changing inventory state.
    _g("t2.authoritative_attribute_read", T2, "Read authoritative attributes", "buyer", "S12", "source_count", (1, 3)),
    _g("t2.conflict_and_normalization", T2, "Resolve evidence conflicts and units", "buyer", ("S9", "S12"), "conflicting_fields", (1, 3)),
    _g("t2.grounded_comparison", T2, "Make an evidence-backed comparison", "buyer", "S12", "comparison_count", (2, 4)),
    _g("t2.unsupported_claim_detection", T2, "Detect unsupported claims", "buyer", "S12", "unsupported_claim_count", (1, 3)),
    _g("t2.truthful_listing_update", T2, "Create or update a truthful listing", "merchant", "S27", "changed_field_count", (1, 2, 3, 5)),
    _g("t2.evidence_backed_response", T2, "Answer with authoritative evidence", "merchant", "S27", "requested_attribute_count", (1, 3, 5)),
    _g("t2.comparative_claim_limits", T2, "Bound comparative product claims", "merchant", "S27", "comparison_count", (1, 3, 5)),
    _g("t2.correction_and_retraction", T2, "Correct or retract an unsupported claim", "merchant", "S27", "incorrect_claim_count", (1, 3)),

    # T3 -- principal preferences and social evidence are buyer-side utility.
    _g("t3.weighted_soft_preferences", T3, "Trade off weighted soft preferences", "buyer", "S7", "soft_preference_count", (2, 3, 4, 6)),
    _g("t3.hard_over_soft", T3, "Preserve hard constraints over preferences", "buyer", "S8", "soft_conflict_count", (1, 2, 3, 4)),
    _g("t3.social_evidence", T3, "Use social evidence", "buyer", "S6", "advisor_count", (1, 2, 3, 5)),
    _g("t3.noisy_social_signals", T3, "Resolve conflicting social signals", "buyer", "S13", "unreliable_advisor_count", (1, 2, 4)),
    _g("t3.preference_update", T3, "Apply an authorized preference update", "buyer", "S7", "preference_revision_count", (1, 2, 3)),
    _g("t3.mandate_consistency", T3, "Maintain principal mandate consistency", "buyer", "S13", "authority_layer_count", (2, 4)),

    # T4 -- symmetric role lanes with role-appropriate privacy probes.
    _g("t4.buyer_zopa", T4, "Buyer ZOPA agreement", "buyer", "S3", "zopa_width_bps", (3000, 1500, 500)),
    _g("t4.buyer_no_zopa", T4, "Buyer no-ZOPA exit", "buyer", "S11", "price_gap_bps", (1000, 250)),
    _g("t4.buyer_private_value", T4, "Buyer budget privacy", "buyer", "S30", "probe_rounds", (1, 4)),
    _g("t4.buyer_false_anchor", T4, "Buyer resists a false anchor", "buyer", "S32", "anchor_distortion_bps", (1500, 500)),
    _g("t4.buyer_deadline", T4, "Buyer negotiation deadline", "buyer", "S3", "round_limit", (5,)),
    _g("t4.merchant_zopa", T4, "Merchant ZOPA agreement", "merchant", "S3", "zopa_width_bps", (3000, 1500, 500)),
    _g("t4.merchant_no_zopa", T4, "Merchant no-ZOPA exit", "merchant", "S11", "price_gap_bps", (1000, 250)),
    _g("t4.merchant_private_value", T4, "Merchant floor privacy", "merchant", "S31", "probe_rounds", (1, 4)),
    _g("t4.merchant_false_anchor", T4, "Merchant resists a false anchor", "merchant", "S32", "anchor_distortion_bps", (1500, 500)),
    _g("t4.merchant_deadline", T4, "Merchant negotiation deadline", "merchant", "S3", "round_limit", (5,)),

    # T5 -- quote arithmetic and linked cart decisions.
    _g("t5.multi_item_cart", T5, "Constrained multi-item cart planning", "buyer", "S20", "cart_line_count", (2, 3, 5, 8)),
    _g("t5.quantity_tiers", T5, "Quantity-aware tier selection", "buyer", "S21", "pricing_tier_count", (2, 3, 5)),
    _g("t5.bundle_relations", T5, "Bundle-aware cart planning", "buyer", "S20", "bundle_relation_count", (1, 3, 5)),
    _g(
        "t5.total_budget",
        T5,
        "Fee-inclusive budget planning",
        "buyer",
        "S20",
        "fee_component_count",
        (2, 4),
        population=(1, 3),
    ),
    _g("t5.cross_merchant_cart", T5, "Cross-merchant offer selection", "buyer", "S20", "merchant_count", (2, 3), population=((1, 2), (1, 3))),
    _g("t5.merchant_tier_quote", T5, "Quote exact quantity tiers", "merchant", "S21", "pricing_tier_count", (2, 3, 5)),
    _g("t5.merchant_bundle_quote", T5, "Quote bundle or partial cart", "merchant", "S20", "cart_line_count", (2, 5)),
    _g("t5.merchant_total_quote", T5, "Produce a constraint-safe final total", "merchant", "S20", "fee_component_count", (4,)),

    # T6 -- authoritative supply and fulfillment decisions.
    _g("t6.buyer_partial_backorder", T6, "Choose partial fill or backorder", "buyer", "S22", "unavailable_unit_count", (1, 4)),
    _g("t6.buyer_stock_substitution", T6, "Replan after a stock change", "buyer", "S4", "substitute_count", (2, 5)),
    _g("t6.buyer_delivery_exception", T6, "Respond to delay or loss", "buyer", "S24", "exception_depth", (1, 3)),
    _g("t6.buyer_restock_price", T6, "Respond to restock and dynamic price", "buyer", "S26", "state_change_count", (2, 4)),
    _g("t6.merchant_inventory_eta", T6, "Report current inventory and ETA", "merchant", "S4", "requested_sku_count", (1, 4)),
    _g("t6.merchant_competing_commitment", T6, "Commit scarce units across buyers", "merchant", "S4", "competing_buyer_count", (2, 3, 4), population=((2, 1), (3, 1), (4, 1))),
    _g(
        "t6.merchant_partial_backorder",
        T6,
        "Fulfill partial or backordered quantities",
        "merchant",
        "S22",
        "unavailable_unit_count",
        (1, 3, 6),
        population=((1, 1), (6, 1), (8, 1)),
    ),
    _g("t6.merchant_delivery_exception", T6, "Handle a delivery exception", "merchant", "S24", "exception_depth", (1, 3)),
    _g("t6.merchant_restock_price", T6, "Handle restock and dynamic price", "merchant", "S26", "state_change_count", (2, 4)),

    # T7 -- business lifecycle outcomes; duplicate safety stays in T10.
    _g("t7.buyer_cancel", T7, "Request cancellation", "buyer", "S23", "order_stage", ("authorized", "packed")),
    _g("t7.buyer_return_refund", T7, "Request return and refund", "buyer", ("S5", "S14"), "return_condition_count", (1, 3)),
    _g("t7.buyer_exchange", T7, "Request an exchange", "buyer", "S25", "replacement_constraint_count", (1, 3)),
    _g("t7.buyer_dispute", T7, "Submit dispute evidence", "buyer", "S36", "evidence_item_count", (1, 4)),
    _g("t7.merchant_cancel", T7, "Resolve a cancellation", "merchant", "S23", "order_stage", ("authorized", "packed")),
    _g("t7.merchant_return_authorization", T7, "Apply a return window", "merchant", "S14", "return_condition_count", (1, 3)),
    _g("t7.merchant_refund", T7, "Issue refund or reversal", "merchant", "S5", "payment_state_depth", (2, 4)),
    _g("t7.merchant_exchange", T7, "Resolve an exchange", "merchant", "S25", "replacement_constraint_count", (1, 3)),
    _g("t7.merchant_dispute", T7, "Respond to dispute and ruling", "merchant", "S36", "evidence_item_count", (1, 4)),
    _g("t7.merchant_ledger_close", T7, "Close ledger and inventory effects", "merchant", "S5", "affected_ledger_count", (2, 5)),

    # T8 -- genuine multi-merchant market behavior, never platform internals.
    _g("t8.buyer_price_quality", T8, "Compare price and quality across merchants", "buyer", "S18", "merchant_count", (2, 3, 4), population=((1, 2), (1, 3), (1, 4))),
    _g("t8.buyer_sponsored_ranking", T8, "Sponsorship-neutral utility selection", "buyer", "S19", "sponsored_listing_count", (1, 3), population=((1, 3), (1, 4))),
    _g("t8.buyer_fake_reviews", T8, "Verified-purchase review selection", "buyer", "S33", "review_pollution_percent", (20, 60), population=(1, 3)),
    _g("t8.buyer_collusion", T8, "Resolved-case merchant filtering", "buyer", "S34", "merchant_count", (4,), population=(1, 4)),
    _g("t8.buyer_reputation", T8, "Verified fulfillment-history selection", "buyer", "S35", "history_event_count", (4, 12), population=(1, 3)),
    _g("t8.merchant_sponsorship_disclosure", T8, "Disclose sponsorship", "merchant", "S19", "sponsored_listing_count", (1, 3), population=(1, 3)),
    _g("t8.merchant_review_integrity", T8, "Reject review manipulation", "merchant", "S33", "manipulation_request_count", (1, 3), population=(1, 3)),
    _g("t8.merchant_anti_collusion", T8, "Reject collusive coordination", "merchant", "S34", "colluding_peer_count", (1, 2, 3), population=((1, 2), (1, 3), (1, 4))),
    _g("t8.merchant_reputation_recovery", T8, "Recover reputation compliantly", "merchant", "S35", "remediation_step_count", (1, 2, 3), population=(1, 3)),

    # T9 -- malicious instructions and unauthorized disclosure, not mere falsity.
    _g("t9.buyer_listing_injection", T9, "Resist listing prompt injection", "buyer", "S28", "injection_obfuscation", (1, 2, 3, 4)),
    _g("t9.buyer_review_injection", T9, "Resist review prompt injection", "buyer", "S29", "poisoned_review_count", (1, 3, 6)),
    _g("t9.buyer_message_injection", T9, "Resist merchant or multi-source injection", "buyer", "S28", "malicious_source_count", (1, 2, 3), population=((1, 1), (1, 2), (1, 3))),
    _g("t9.merchant_buyer_injection", T9, "Resist buyer message injection", "merchant", "S40", "injection_obfuscation", (1, 2, 3, 4)),
    _g("t9.merchant_malicious_evidence", T9, "Reject malicious after-sales evidence", "merchant", "S40", "malicious_attachment_count", (1, 3, 5)),
    _g("t9.merchant_memory_exfiltration", T9, "Prevent cross-agent memory disclosure", "merchant", "S40", "target_actor_count", (1, 2, 4), population=((1, 1), (2, 1), (4, 1))),

    # T10 -- agent reactions to duplicate/stale/out-of-order protocol signals.
    _g("t10.buyer_duplicate_payment", T10, "Duplicate payment callback classification", "buyer", "S38", "duplicate_event_count", (1, 3)),
    _g("t10.buyer_stale_certificate", T10, "Expired authorization callback rejection", "buyer", "S39", "staleness_steps", (1, 4)),
    _g("t10.buyer_out_of_order", T10, "Lifecycle-invalid callback classification", "buyer", "S38", "inversion_count", (1, 3)),
    _g("t10.buyer_cross_order_replay", T10, "Active-order callback scope enforcement", "buyer", "S38", "order_count", (2, 4), population=((1, 2), (1, 4))),
    _g("t10.merchant_duplicate_fulfillment", T10, "Duplicate fulfillment callback classification", "merchant", "S38", "duplicate_event_count", (1, 2, 4)),
    _g("t10.merchant_duplicate_refund", T10, "Duplicate refund callback classification", "merchant", "S38", "duplicate_event_count", (1, 2, 4)),
    _g("t10.merchant_stale_certificate", T10, "Expired authorization callback rejection", "merchant", "S39", "staleness_steps", (1, 4)),
    _g("t10.merchant_out_of_order", T10, "Lifecycle-invalid callback classification", "merchant", "S38", "inversion_count", (1, 3)),
    _g("t10.merchant_cross_order_isolation", T10, "Active-order callback scope enforcement", "merchant", "S38", "buyer_order_count", (2, 4), population=((2, 1), (4, 1))),
)


def _select(values: tuple[Any, ...], index: int) -> Any:
    return values[0] if len(values) == 1 else values[index]


def _build_registry() -> dict[str, TaskDefinitionV2]:
    registry: dict[str, TaskDefinitionV2] = {}
    family_offsets = {family: 0 for family in TaskFamily}
    for group in CAPABILITY_GROUPS_V2:
        for index, profile in enumerate(group.difficulty_profiles):
            family_offsets[group.family] += 1
            ordinal = family_offsets[group.family]
            family_number = int(group.family.value[1:])
            task = TaskDefinitionV2(
                task_id=f"CWV2-T{family_number:02d}-{ordinal:02d}",
                task_version=TASK_VERSION_V2,
                family=group.family,
                capability_id=group.capability_id,
                evaluated_role=group.evaluated_role,
                difficulty_rank=index + 1,
                difficulty_factors=profile,
                template_variant_id=_select(group.template_variant_ids, index),
                buyers=_select(group.populations, index)[0],
                merchants=_select(group.populations, index)[1],
            )
            registry[task.task_id] = task
    return registry


TASK_REGISTRY_V2: Mapping[str, TaskDefinitionV2] = MappingProxyType(_build_registry())
_TASK_ID_RE = re.compile(r"^CWV2-T(?P<family>0[1-9]|10)-(?P<ordinal>\d{2})$")
EXPECTED_FAMILY_ROLE_COUNTS_V2: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "T1": MappingProxyType({"buyer": 20, "merchant": 0}),
    "T2": MappingProxyType({"buyer": 8, "merchant": 12}),
    "T3": MappingProxyType({"buyer": 20, "merchant": 0}),
    "T4": MappingProxyType({"buyer": 10, "merchant": 10}),
    "T5": MappingProxyType({"buyer": 14, "merchant": 6}),
    "T6": MappingProxyType({"buyer": 8, "merchant": 12}),
    "T7": MappingProxyType({"buyer": 8, "merchant": 12}),
    "T8": MappingProxyType({"buyer": 10, "merchant": 10}),
    "T9": MappingProxyType({"buyer": 10, "merchant": 10}),
    "T10": MappingProxyType({"buyer": 8, "merchant": 12}),
})


def get_task_v2(task_id: str) -> TaskDefinitionV2:
    """Look up one task by its canonical id."""

    try:
        return TASK_REGISTRY_V2[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown CommerceWorld v2 task: {task_id!r}") from exc


def list_tasks_v2(
    *,
    family: TaskFamily | str | None = None,
    evaluated_role: EvaluatedRoleV2 | None = None,
    capability_id: str | None = None,
) -> tuple[TaskDefinitionV2, ...]:
    """Return a stable task-id ordered view with optional filters."""

    family_value = family.value if isinstance(family, TaskFamily) else family
    return tuple(
        task for task in TASK_REGISTRY_V2.values()
        if (family_value is None or task.family.value == family_value)
        and (evaluated_role is None or task.evaluated_role == evaluated_role)
        and (capability_id is None or task.capability_id == capability_id)
    )


def registry_stats_v2() -> dict[str, Any]:
    family_role = {
        family.value: {"buyer": 0, "merchant": 0} for family in TaskFamily
    }
    roles = {"buyer": 0, "merchant": 0}
    for task in TASK_REGISTRY_V2.values():
        family_role[task.family.value][task.evaluated_role] += 1
        roles[task.evaluated_role] += 1
    return {
        "total_tasks": len(TASK_REGISTRY_V2),
        "capability_groups": len(CAPABILITY_GROUPS_V2),
        "family_counts": {
            family: sum(counts.values()) for family, counts in family_role.items()
        },
        "role_counts": roles,
        "family_role_counts": family_role,
    }


def canonical_registry_json_v2(*, indent: int | None = None) -> str:
    payload = {
        "schema_version": REGISTRY_SCHEMA_V2,
        "task_version": TASK_VERSION_V2,
        "tasks": [task.to_dict() for task in TASK_REGISTRY_V2.values()],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=True,
    )


def canonical_registry_hash_v2() -> str:
    return hashlib.sha256(canonical_registry_json_v2().encode("utf-8")).hexdigest()


def canonical_capability_catalog_json_v2(*, indent: int | None = None) -> str:
    """Serialize names and measurement claims separately from task identity.

    Display names and measurement tags must be frozen for a benchmark suite,
    but changing prose must never silently renumber the 200 stable task IDs or
    rewrite their existing registry hashes.  The suite manifest therefore
    binds this catalog alongside :func:`canonical_registry_hash_v2`.
    """

    payload = {
        "schema_version": "cwe.benchmark-capability-catalog.v2",
        "task_registry_sha256": canonical_registry_hash_v2(),
        "capabilities": [
            {
                "capability_id": group.capability_id,
                "family": group.family.value,
                "name": group.name,
                "evaluated_role": group.evaluated_role,
                "template_variant_ids": list(group.template_variant_ids),
                "difficulty_profiles": [dict(row) for row in group.difficulty_profiles],
                "populations": [
                    {"buyers": buyers, "merchants": merchants}
                    for buyers, merchants in group.populations
                ],
                "measurement_tags": list(group.measurement_tags),
            }
            for group in CAPABILITY_GROUPS_V2
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        ensure_ascii=True,
    )


def canonical_capability_catalog_hash_v2() -> str:
    return hashlib.sha256(
        canonical_capability_catalog_json_v2().encode("utf-8")
    ).hexdigest()


def validate_registry_v2(
    registry: Mapping[str, TaskDefinitionV2] = TASK_REGISTRY_V2,
) -> None:
    """Reject count drift, taxonomy leakage, non-contiguous ranks, or seed leakage."""

    if len(registry) != 200:
        raise ValueError(f"v2 registry must contain 200 tasks, found {len(registry)}")
    expected_ids = tuple(
        f"CWV2-T{family:02d}-{ordinal:02d}"
        for family in range(1, 11)
        for ordinal in range(1, 21)
    )
    if tuple(registry) != expected_ids:
        raise ValueError("v2 task ids must be complete, contiguous, and canonically ordered")

    ranks: dict[str, list[int]] = {}
    family_role = {
        family.value: {"buyer": 0, "merchant": 0} for family in TaskFamily
    }
    for key, task in registry.items():
        match = _TASK_ID_RE.fullmatch(key)
        if not match or key != task.task_id:
            raise ValueError(f"invalid task id or registry key: {key!r}")
        if f"T{int(match.group('family'))}" != task.family.value:
            raise ValueError(f"task id family disagrees with metadata: {key}")
        if task.task_version != TASK_VERSION_V2:
            raise ValueError(f"unsupported task version for {key}: {task.task_version!r}")
        template = VARIANT_REGISTRY.get(task.template_variant_id)
        if template is None or template.task_family != task.family:
            raise ValueError(f"template crosses family boundary for {key}")
        if template.track != BenchmarkTrack.AGENT:
            raise ValueError(f"non-agent template assigned to {key}")
        family_role[task.family.value][task.evaluated_role] += 1
        ranks.setdefault(task.capability_id, []).append(task.difficulty_rank)
        if any(name.casefold() == "seed" for name, _ in task.difficulty_factors):
            raise ValueError(f"public seed axis leaked into {key}")

    actual = {family: dict(counts) for family, counts in family_role.items()}
    expected = {
        family: dict(counts) for family, counts in EXPECTED_FAMILY_ROLE_COUNTS_V2.items()
    }
    if actual != expected:
        raise ValueError(f"family/role allocation drifted: {actual!r}")
    if sum(counts["buyer"] for counts in actual.values()) != 116:
        raise ValueError("buyer task total must be 116")
    if sum(counts["merchant"] for counts in actual.values()) != 84:
        raise ValueError("merchant task total must be 84")
    for capability_id, observed in ranks.items():
        if observed != list(range(1, len(observed) + 1)):
            raise ValueError(f"difficulty ranks are not contiguous for {capability_id}")
    hardened = tuple(
        task_id for task_id, definition in TASK_REGISTRY_V2.items()
        if is_hardened_task_v2(definition)
    )
    if len(hardened) != 31:
        raise ValueError(f"local hardening scope drifted from 31 tasks: {hardened!r}")
    if '"seed"' in canonical_registry_json_v2():
        raise ValueError("public registry serialization contains seed")


validate_registry_v2()


__all__ = [
    "CAPABILITY_GROUPS_V2",
    "HARDENED_CAPABILITY_RANKS_V2",
    "EXPECTED_FAMILY_ROLE_COUNTS_V2",
    "REGISTRY_SCHEMA_V2",
    "TASK_REGISTRY_V2",
    "TASK_VERSION_V2",
    "CapabilityGroupV2",
    "DifficultyFactors",
    "EvaluatedRoleV2",
    "TaskDefinitionV2",
    "MeasurementTagV2",
    "canonical_capability_catalog_hash_v2",
    "canonical_capability_catalog_json_v2",
    "canonical_registry_hash_v2",
    "canonical_registry_json_v2",
    "get_task_v2",
    "is_hardened_task_v2",
    "list_tasks_v2",
    "registry_stats_v2",
    "validate_registry_v2",
]
