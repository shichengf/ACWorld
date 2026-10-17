"""Benchmark taxonomy and parameter axes.

The paper-facing benchmark has ten task families.  The historical ``sN``
identifiers remain useful as *variant* identifiers, but they are not separate
headline categories.  Keeping this registry executable prevents the paper,
scenario files, CLI filters, and reports from drifting into different
taxonomies.

Three variants (S15--S17) validate the benchmark harness rather than an agent;
S37 is a platform-policy diagnostic.  They stay in the registry but are placed
on separate tracks so the main Buyer/Merchant leaderboard can exclude them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskFamily(str, Enum):
    PRODUCT_DISCOVERY = "T1"
    PRODUCT_GROUNDING = "T2"
    PREFERENCE_SOCIAL = "T3"
    NEGOTIATION_PRIVACY = "T4"
    CART_QUANTITY = "T5"
    INVENTORY_FULFILLMENT = "T6"
    ORDER_AFTERSALES = "T7"
    MARKET_GOVERNANCE = "T8"
    ADVERSARIAL_CONTENT = "T9"
    TRANSACTION_INTEGRITY = "T10"


TASK_FAMILY_NAMES: dict[TaskFamily, str] = {
    TaskFamily.PRODUCT_DISCOVERY: "Product discovery and feasibility",
    TaskFamily.PRODUCT_GROUNDING: "Product grounding and truthfulness",
    TaskFamily.PREFERENCE_SOCIAL: "Preference trade-offs and social decisions",
    TaskFamily.NEGOTIATION_PRIVACY: "Negotiation and private utility",
    TaskFamily.CART_QUANTITY: "Multi-item and quantity planning",
    TaskFamily.INVENTORY_FULFILLMENT: "Inventory, fulfillment, and supply changes",
    TaskFamily.ORDER_AFTERSALES: "Order lifecycle and after-sales resolution",
    TaskFamily.MARKET_GOVERNANCE: "Multi-merchant markets and platform governance",
    TaskFamily.ADVERSARIAL_CONTENT: "Adversarial content and prompt injection",
    TaskFamily.TRANSACTION_INTEGRITY: "Transaction timing and replay safety",
}


class BenchmarkTrack(str, Enum):
    AGENT = "agent"
    PLATFORM_DIAGNOSTIC = "platform_diagnostic"
    CONFORMANCE = "conformance"


class Difficulty(str, Enum):
    BASELINE = "baseline"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CatalogScale(str, Enum):
    SMOKE = "smoke"
    STANDARD = "standard"
    MEDIUM = "medium"
    FULL = "full"


CATALOG_SCALE_TARGETS: dict[CatalogScale, int | None] = {
    CatalogScale.SMOKE: 5,
    CatalogScale.STANDARD: 50,
    CatalogScale.MEDIUM: 250,
    CatalogScale.FULL: None,
}


class CatalogSource(str, Enum):
    INLINE = "inline"
    REAL_CSV = "real_csv"


@dataclass(frozen=True)
class VariantDefinition:
    variant_id: str
    name: str
    task_family: TaskFamily | None
    track: BenchmarkTrack = BenchmarkTrack.AGENT


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Machine-readable benchmark metadata attached to a ``ScenarioSpec``."""

    variant_id: str
    task_family: TaskFamily | None
    track: BenchmarkTrack
    difficulty: Difficulty = Difficulty.BASELINE
    catalog_scale: CatalogScale = CatalogScale.SMOKE
    catalog_source: CatalogSource = CatalogSource.INLINE
    catalog_merchants: tuple[str, ...] = ()
    in_stock_only: bool = False
    scenario_version: str = "1.0"

    @property
    def target_listings(self) -> int | None:
        return CATALOG_SCALE_TARGETS[self.catalog_scale]


def _v(number: int, family: TaskFamily | None, name: str,
       track: BenchmarkTrack = BenchmarkTrack.AGENT) -> VariantDefinition:
    return VariantDefinition(f"S{number}", name, family, track)


# One authoritative mapping for the complete roadmap.  Scenario implementations
# may land incrementally; registry membership does not claim implementation.
VARIANT_REGISTRY: dict[str, VariantDefinition] = {
    v.variant_id: v for v in (
        _v(1, TaskFamily.PRODUCT_DISCOVERY, "direct_purchase"),
        _v(2, TaskFamily.PRODUCT_DISCOVERY, "constraint_matching"),
        _v(3, TaskFamily.NEGOTIATION_PRIVACY, "negotiation"),
        _v(4, TaskFamily.INVENTORY_FULFILLMENT, "inventory_edge"),
        _v(5, TaskFamily.ORDER_AFTERSALES, "return_refund"),
        _v(6, TaskFamily.PREFERENCE_SOCIAL, "friend_conflict"),
        _v(7, TaskFamily.PREFERENCE_SOCIAL, "soft_compromise"),
        _v(8, TaskFamily.PREFERENCE_SOCIAL, "rigid_over_friend"),
        _v(9, TaskFamily.PRODUCT_GROUNDING, "multiword_material"),
        _v(10, TaskFamily.PRODUCT_DISCOVERY, "no_feasible_sku"),
        _v(11, TaskFamily.NEGOTIATION_PRIVACY, "no_zopa_negotiation"),
        _v(12, TaskFamily.PRODUCT_GROUNDING, "grounding_provenance"),
        _v(13, TaskFamily.PREFERENCE_SOCIAL, "authoritative_friend"),
        _v(14, TaskFamily.ORDER_AFTERSALES, "return_window"),
        _v(15, None, "replay_determinism", BenchmarkTrack.CONFORMANCE),
        _v(16, None, "router_secret_parity", BenchmarkTrack.CONFORMANCE),
        _v(17, None, "trace_completion_integrity", BenchmarkTrack.CONFORMANCE),
        _v(18, TaskFamily.MARKET_GOVERNANCE, "multi_merchant_price_quality"),
        _v(19, TaskFamily.MARKET_GOVERNANCE, "sponsored_ranking_bias"),
        _v(20, TaskFamily.CART_QUANTITY, "multi_item_cart"),
        _v(21, TaskFamily.CART_QUANTITY, "quantity_discount"),
        _v(22, TaskFamily.INVENTORY_FULFILLMENT, "partial_fill_backorder"),
        _v(23, TaskFamily.ORDER_AFTERSALES, "pre_dispatch_cancel"),
        _v(24, TaskFamily.INVENTORY_FULFILLMENT, "lost_or_delayed_shipment"),
        _v(25, TaskFamily.ORDER_AFTERSALES, "exchange_instead_of_refund"),
        _v(26, TaskFamily.INVENTORY_FULFILLMENT, "restock_then_dynamic_price"),
        _v(27, TaskFamily.PRODUCT_GROUNDING, "listing_update_truthfulness"),
        _v(28, TaskFamily.ADVERSARIAL_CONTENT, "prompt_injection_in_listing"),
        _v(29, TaskFamily.ADVERSARIAL_CONTENT, "prompt_injection_in_review"),
        _v(30, TaskFamily.NEGOTIATION_PRIVACY, "merchant_budget_probe"),
        _v(31, TaskFamily.NEGOTIATION_PRIVACY, "buyer_floor_probe"),
        _v(32, TaskFamily.NEGOTIATION_PRIVACY, "false_discount_anchor"),
        _v(33, TaskFamily.MARKET_GOVERNANCE, "fake_reviews"),
        _v(34, TaskFamily.MARKET_GOVERNANCE, "collusive_merchants"),
        _v(35, TaskFamily.MARKET_GOVERNANCE, "reputation_recovery"),
        _v(36, TaskFamily.ORDER_AFTERSALES, "dispute_with_evidence"),
        _v(37, TaskFamily.MARKET_GOVERNANCE, "adjudicator_bias",
           BenchmarkTrack.PLATFORM_DIAGNOSTIC),
        _v(38, TaskFamily.TRANSACTION_INTEGRITY, "payment_replay_cross_rail"),
        _v(39, TaskFamily.TRANSACTION_INTEGRITY, "match_certificate_stale"),
        _v(40, TaskFamily.ADVERSARIAL_CONTENT, "buyer_message_prompt_injection"),
    )
}


_SCENARIO_ID_RE = re.compile(r"^s(?P<number>\d+)(?:_|$)", re.IGNORECASE)


def variant_id_from_scenario_id(scenario_id: str) -> str | None:
    match = _SCENARIO_ID_RE.match(str(scenario_id))
    return f"S{int(match.group('number'))}" if match else None


def variant_for_scenario(scenario_id: str) -> VariantDefinition | None:
    variant_id = variant_id_from_scenario_id(scenario_id)
    return VARIANT_REGISTRY.get(variant_id or "")


def _enum(enum_type: type[Enum], value: Any, field: str) -> Any:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        choices = ", ".join(str(e.value) for e in enum_type)
        raise ValueError(f"invalid benchmark.{field} {value!r}; expected one of: {choices}") from exc


def metadata_for_scenario(
    scenario_id: str,
    raw: dict[str, Any] | None = None,
) -> BenchmarkMetadata:
    """Infer metadata from ``sN`` and apply an optional YAML override block."""

    raw = dict(raw or {})
    definition = variant_for_scenario(scenario_id)
    inferred_id = definition.variant_id if definition else variant_id_from_scenario_id(scenario_id)
    variant_id = str(raw.get("variant_id") or inferred_id or scenario_id)

    family_value = raw.get("task_family")
    if family_value is None:
        family = definition.task_family if definition else None
    else:
        family = _enum(TaskFamily, family_value, "task_family")

    track_value = raw.get("track")
    track = (definition.track if definition and track_value is None
             else _enum(BenchmarkTrack, track_value or BenchmarkTrack.AGENT.value, "track"))
    merchants = raw.get("catalog_merchants") or ()
    if not isinstance(merchants, (list, tuple)):
        raise ValueError("benchmark.catalog_merchants must be a list")

    return BenchmarkMetadata(
        variant_id=variant_id,
        task_family=family,
        track=track,
        difficulty=_enum(Difficulty, raw.get("difficulty", Difficulty.BASELINE.value),
                         "difficulty"),
        catalog_scale=_enum(CatalogScale,
                            raw.get("catalog_scale", CatalogScale.SMOKE.value),
                            "catalog_scale"),
        catalog_source=_enum(CatalogSource,
                             raw.get("catalog_source", CatalogSource.INLINE.value),
                             "catalog_source"),
        catalog_merchants=tuple(str(m) for m in merchants),
        in_stock_only=bool(raw.get("in_stock_only", False)),
        scenario_version=str(raw.get("scenario_version", "1.0")),
    )


def family_counts(scenario_ids: list[str]) -> dict[str, int]:
    """Count implemented scenarios per paper-facing family."""

    counts = {family.value: 0 for family in TaskFamily}
    for scenario_id in scenario_ids:
        definition = variant_for_scenario(scenario_id)
        if definition and definition.task_family is not None:
            counts[definition.task_family.value] += 1
    return counts


__all__ = [
    "BenchmarkMetadata",
    "BenchmarkTrack",
    "CATALOG_SCALE_TARGETS",
    "CatalogScale",
    "CatalogSource",
    "Difficulty",
    "TASK_FAMILY_NAMES",
    "TaskFamily",
    "VARIANT_REGISTRY",
    "VariantDefinition",
    "family_counts",
    "metadata_for_scenario",
    "variant_for_scenario",
    "variant_id_from_scenario_id",
]
