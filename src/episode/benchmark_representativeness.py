"""Machine-checkable workflow/schema representativeness for Benchmark v2.

The benchmark uses controlled synthetic identities and transactions.  This
module validates a narrower and defensible claim: every fixed capability maps
to a documented commerce workflow, marketplace mechanism, or established
agent-security / transaction-integrity risk.  It deliberately does *not* claim
that task frequencies reproduce any deployed marketplace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

from episode.benchmark import TaskFamily
from episode.capability_benchmark import CAPABILITY_GROUPS_V2, TASK_REGISTRY_V2


REPRESENTATIVENESS_SCHEMA_V1 = "cwe.benchmark-representativeness.v1"
RepresentativenessCategoryV1 = Literal[
    "standard_commerce_lifecycle",
    "platform_market_mechanism",
    "agent_security_privacy",
    "transaction_integrity",
]


@dataclass(frozen=True, slots=True)
class WorkflowAlignmentV1:
    """One family-level alignment to public commerce concepts."""

    family: TaskFamily
    workflow_id: str
    workflow_name: str
    lifecycle_stage: str
    entities: tuple[str, ...]
    real_scenarios: tuple[str, ...]
    public_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.workflow_id,
            self.workflow_name,
            self.lifecycle_stage,
        )
        if any(not value.strip() for value in values):
            raise ValueError("representativeness workflow metadata must be non-empty")
        if not self.entities or not self.real_scenarios or not self.public_sources:
            raise ValueError("representativeness workflow alignment is incomplete")
        if any(not source.startswith("https://") for source in self.public_sources):
            raise ValueError("representativeness sources must be public HTTPS URLs")


_WORKFLOW_ALIGNMENTS_V1: Mapping[TaskFamily, WorkflowAlignmentV1] = MappingProxyType(
    {
        TaskFamily.PRODUCT_DISCOVERY: WorkflowAlignmentV1(
            family=TaskFamily.PRODUCT_DISCOVERY,
            workflow_id="catalog_discovery_and_feasibility",
            workflow_name="Catalog discovery and feasible-item selection",
            lifecycle_stage="pre_purchase",
            entities=("query", "listing", "price", "availability", "search_session"),
            real_scenarios=(
                "filter products by availability, price, category, and attributes",
                "select the best feasible listing or explicitly abstain",
                "reformulate a query without dropping purchase constraints",
            ),
            public_sources=(
                "https://shopify.dev/docs/api/storefront/latest/input-objects/ProductFilter",
            ),
        ),
        TaskFamily.PRODUCT_GROUNDING: WorkflowAlignmentV1(
            family=TaskFamily.PRODUCT_GROUNDING,
            workflow_id="catalog_truth_and_evidence",
            workflow_name="Catalog evidence, claim support, and correction",
            lifecycle_stage="pre_purchase_and_catalog_management",
            entities=("listing", "attribute", "evidence_record", "claim", "revision"),
            real_scenarios=(
                "reconcile conflicting product attributes and units",
                "support a comparison with authoritative product evidence",
                "correct or retract an unsupported merchant claim",
            ),
            public_sources=(
                "https://support.google.com/merchants/answer/7052112?hl=en",
                "https://www.ftc.gov/sites/default/files/attachments/training-materials/policy_substantiation.pdf",
            ),
        ),
        TaskFamily.PREFERENCE_SOCIAL: WorkflowAlignmentV1(
            family=TaskFamily.PREFERENCE_SOCIAL,
            workflow_id="preference_and_social_decision_support",
            workflow_name="Preference trade-offs and social decision support",
            lifecycle_stage="pre_purchase",
            entities=("mandate", "hard_constraint", "soft_preference", "advisor", "review"),
            real_scenarios=(
                "rank feasible products under weighted preferences",
                "preserve hard requirements over conflicting soft preferences",
                "use reviewer or advisor signals while discounting unreliable sources",
            ),
            public_sources=(
                "https://shopify.dev/docs/api/storefront/latest/queries/productrecommendations",
                "https://developers.google.com/product-review-feeds/schema/",
            ),
        ),
        TaskFamily.NEGOTIATION_PRIVACY: WorkflowAlignmentV1(
            family=TaskFamily.NEGOTIATION_PRIVACY,
            workflow_id="offer_negotiation_and_private_utility",
            workflow_name="Offer negotiation, agreement, and private thresholds",
            lifecycle_stage="offer_and_pre_settlement",
            entities=("negotiation_thread", "offer", "counteroffer", "agreement", "expiry"),
            real_scenarios=(
                "reach agreement inside a zone of possible agreement",
                "decline when buyer and seller price ranges do not overlap",
                "protect budget or floor values against probing and false anchors",
            ),
            public_sources=(
                "https://developer.ebay.com/api-docs/user-guides/static/trading-user-guide/best-offers.html",
                "https://developer.ebay.com/api-docs/user-guides/static/trading-user-guide/best-offers-counter.html",
            ),
        ),
        TaskFamily.CART_QUANTITY: WorkflowAlignmentV1(
            family=TaskFamily.CART_QUANTITY,
            workflow_id="cart_pricing_quote_and_checkout",
            workflow_name="Multi-line cart construction, pricing, and checkout",
            lifecycle_stage="cart_and_checkout",
            entities=("cart_line", "quantity_tier", "bundle", "quote", "charge", "order_group"),
            real_scenarios=(
                "construct a multi-item or cross-merchant cart",
                "apply quantity tiers, bundle discounts, shipping, tax, and fees",
                "produce or accept an exact quote before atomic checkout",
            ),
            public_sources=(
                "https://shopify.dev/docs/api/storefront/latest/objects/CartLine",
                "https://shopify.dev/docs/api/storefront/latest/objects/Cart",
            ),
        ),
        TaskFamily.INVENTORY_FULFILLMENT: WorkflowAlignmentV1(
            family=TaskFamily.INVENTORY_FULFILLMENT,
            workflow_id="inventory_supply_and_fulfillment",
            workflow_name="Inventory allocation, supply changes, and fulfillment",
            lifecycle_stage="allocation_and_fulfillment",
            entities=("inventory", "reservation", "allocation", "backorder", "shipment", "restock"),
            real_scenarios=(
                "choose partial fill, backorder, or a substitute after a stock change",
                "allocate scarce units across competing buyers",
                "respond to restock, dynamic price, delay, or shipment loss",
            ),
            public_sources=(
                "https://shopify.dev/docs/apps/build/orders-fulfillment/inventory-management-apps",
                "https://shopify.dev/docs/api/admin-graphql/latest/enums/FulfillmentOrderStatus",
            ),
        ),
        TaskFamily.ORDER_AFTERSALES: WorkflowAlignmentV1(
            family=TaskFamily.ORDER_AFTERSALES,
            workflow_id="order_after_sales_and_financial_reversal",
            workflow_name="Order lifecycle, returns, refunds, exchanges, and disputes",
            lifecycle_stage="post_purchase",
            entities=("order", "packing", "return", "refund", "exchange", "dispute", "ledger"),
            real_scenarios=(
                "cancel an authorized or packed order with the correct downstream effects",
                "authorize a return, restock received goods, and issue a refund",
                "resolve an exchange or dispute and close ledger and inventory effects",
            ),
            public_sources=(
                "https://shopify.dev/docs/apps/build/orders-fulfillment/returns-apps",
                "https://docs.stripe.com/refunds",
                "https://docs.stripe.com/disputes/how-disputes-work",
            ),
        ),
        TaskFamily.MARKET_GOVERNANCE: WorkflowAlignmentV1(
            family=TaskFamily.MARKET_GOVERNANCE,
            workflow_id="marketplace_governance_and_trust",
            workflow_name="Multi-merchant ranking, disclosure, reviews, and market integrity",
            lifecycle_stage="marketplace_governance",
            entities=("placement", "sponsorship", "review", "collusion_case", "reputation", "remediation"),
            real_scenarios=(
                "account for sponsored placement when comparing merchants",
                "remain robust to review pollution and reject review manipulation",
                "detect or reject collusion and perform compliant reputation remediation",
            ),
            public_sources=(
                "https://advertising.amazon.com/solutions/products/sponsored-products",
                "https://support.google.com/merchants/answer/6098512?hl=en-GB",
                "https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/dealings-competitors/price-fixing",
            ),
        ),
        TaskFamily.ADVERSARIAL_CONTENT: WorkflowAlignmentV1(
            family=TaskFamily.ADVERSARIAL_CONTENT,
            workflow_id="untrusted_commerce_content_security",
            workflow_name="Indirect prompt injection and cross-actor privacy protection",
            lifecycle_stage="cross_cutting_agent_security",
            entities=("listing_content", "review_content", "message", "attachment", "actor_memory"),
            real_scenarios=(
                "ignore malicious instructions embedded in listings or reviews",
                "reject hostile buyer or merchant messages and after-sales attachments",
                "prevent one actor from exfiltrating another actor's memory",
            ),
            public_sources=(
                "https://www.nist.gov/news-events/news/2025/01/technical-blog-strengthening-ai-agent-hijacking-evaluations",
                "https://csrc.nist.gov/glossary/term/indirect_prompt_injection",
            ),
        ),
        TaskFamily.TRANSACTION_INTEGRITY: WorkflowAlignmentV1(
            family=TaskFamily.TRANSACTION_INTEGRITY,
            workflow_id="transaction_timing_idempotency_and_replay_safety",
            workflow_name="Idempotent event handling, ordering, expiry, and isolation",
            lifecycle_stage="cross_cutting_transaction_integrity",
            entities=("protocol_event", "object_version", "idempotency_key", "receipt", "order_binding"),
            real_scenarios=(
                "deduplicate repeated payment, fulfillment, or refund callbacks",
                "reject stale certificates and out-of-order state transitions",
                "prevent an event from one order being replayed against another order",
            ),
            public_sources=(
                "https://docs.stripe.com/api/idempotent_requests?lang=curl",
                "https://docs.stripe.com/webhooks?lang=node",
            ),
        ),
    }
)

_T4_PRIVATE_VALUE_CAPABILITIES = frozenset(
    {"t4.buyer_private_value", "t4.merchant_private_value"}
)

EXPECTED_CATEGORY_COUNTS_V1: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        "standard_commerce_lifecycle": MappingProxyType(
            {"capabilities": 54, "tasks": 136}
        ),
        "platform_market_mechanism": MappingProxyType(
            {"capabilities": 9, "tasks": 20}
        ),
        "agent_security_privacy": MappingProxyType(
            {"capabilities": 8, "tasks": 24}
        ),
        "transaction_integrity": MappingProxyType(
            {"capabilities": 9, "tasks": 20}
        ),
    }
)


def _category_for(
    family: TaskFamily,
    capability_id: str,
) -> RepresentativenessCategoryV1:
    if family is TaskFamily.MARKET_GOVERNANCE:
        return "platform_market_mechanism"
    if family is TaskFamily.ADVERSARIAL_CONTENT or (
        capability_id in _T4_PRIVATE_VALUE_CAPABILITIES
    ):
        return "agent_security_privacy"
    if family is TaskFamily.TRANSACTION_INTEGRITY:
        return "transaction_integrity"
    return "standard_commerce_lifecycle"


def representativeness_report_v1() -> dict[str, Any]:
    """Build the complete 80-capability workflow/schema coverage report."""

    rows: list[dict[str, Any]] = []
    category_counts = {
        category: {"capabilities": 0, "tasks": 0}
        for category in EXPECTED_CATEGORY_COUNTS_V1
    }
    for group in CAPABILITY_GROUPS_V2:
        alignment = _WORKFLOW_ALIGNMENTS_V1[group.family]
        category = _category_for(group.family, group.capability_id)
        tasks = tuple(
            task.task_id
            for task in TASK_REGISTRY_V2.values()
            if task.capability_id == group.capability_id
        )
        axes = {
            name
            for profile in group.difficulty_profiles
            for name, _value in profile
            if name != "difficulty_level"
        }
        if len(axes) != 1:
            raise ValueError(
                f"capability {group.capability_id!r} has no single difficulty axis"
            )
        [difficulty_axis] = axes
        difficulty_values = [
            dict(profile)[difficulty_axis] for profile in group.difficulty_profiles
        ]
        category_counts[category]["capabilities"] += 1
        category_counts[category]["tasks"] += len(tasks)
        rows.append(
            {
                "capability_id": group.capability_id,
                "family": group.family.value,
                "name": group.name,
                "evaluated_role": group.evaluated_role,
                "task_count": len(tasks),
                "task_ids": list(tasks),
                "difficulty_axis": difficulty_axis,
                "difficulty_values": difficulty_values,
                "measurement_tags": list(group.measurement_tags),
                "representativeness_category": category,
                "workflow_id": alignment.workflow_id,
                "workflow_name": alignment.workflow_name,
                "lifecycle_stage": alignment.lifecycle_stage,
                "entities": list(alignment.entities),
                "real_scenarios": list(alignment.real_scenarios),
                "public_sources": list(alignment.public_sources),
            }
        )
    return {
        "schema_version": REPRESENTATIVENESS_SCHEMA_V1,
        "claim_scope": "workflow_and_schema_coverage_not_empirical_frequency",
        "capability_count": len(rows),
        "task_count": sum(row["task_count"] for row in rows),
        "category_counts": category_counts,
        "capabilities": rows,
    }


def validate_representativeness_v1() -> None:
    """Fail closed if the public mapping drifts from the executable registry."""

    if set(_WORKFLOW_ALIGNMENTS_V1) != set(TaskFamily):
        raise ValueError("representativeness workflows do not cover all families")
    report = representativeness_report_v1()
    if report["capability_count"] != 80 or report["task_count"] != 200:
        raise ValueError("representativeness report does not cover the full benchmark")
    capability_ids = [row["capability_id"] for row in report["capabilities"]]
    expected_ids = [group.capability_id for group in CAPABILITY_GROUPS_V2]
    if capability_ids != expected_ids or len(capability_ids) != len(set(capability_ids)):
        raise ValueError("representativeness capability identities drifted")
    task_ids = [
        task_id
        for row in report["capabilities"]
        for task_id in row["task_ids"]
    ]
    if set(task_ids) != set(TASK_REGISTRY_V2) or len(task_ids) != len(set(task_ids)):
        raise ValueError("representativeness task coverage is incomplete or ambiguous")
    observed = {
        category: dict(counts)
        for category, counts in report["category_counts"].items()
    }
    expected = {
        category: dict(counts)
        for category, counts in EXPECTED_CATEGORY_COUNTS_V1.items()
    }
    if observed != expected:
        raise ValueError(
            f"representativeness category counts drifted: {observed!r}"
        )


def canonical_representativeness_json_v1(*, indent: int | None = None) -> str:
    validate_representativeness_v1()
    return json.dumps(
        representativeness_report_v1(),
        ensure_ascii=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def canonical_representativeness_hash_v1() -> str:
    return hashlib.sha256(
        canonical_representativeness_json_v1().encode("utf-8")
    ).hexdigest()


def main() -> None:
    """Print the canonical report for artifact or appendix generation."""

    print(canonical_representativeness_json_v1(indent=2))


validate_representativeness_v1()


__all__ = [
    "EXPECTED_CATEGORY_COUNTS_V1",
    "REPRESENTATIVENESS_SCHEMA_V1",
    "RepresentativenessCategoryV1",
    "WorkflowAlignmentV1",
    "canonical_representativeness_hash_v1",
    "canonical_representativeness_json_v1",
    "main",
    "representativeness_report_v1",
    "validate_representativeness_v1",
]


if __name__ == "__main__":
    main()
