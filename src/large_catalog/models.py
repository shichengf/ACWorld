"""Typed records shared by the large-catalog suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CatalogListing:
    listing_ref: str
    merchant_id: str
    source_sku: str
    name: str
    variant: str
    category: str
    price_minor: int
    currency: str
    in_stock: bool
    material: str
    key_features: str
    tags: str
    warranty: str
    product_url: str

    def public_summary(self) -> dict[str, Any]:
        return {
            "listing_ref": self.listing_ref,
            "merchant": self.merchant_id,
            "name": self.name,
            "variant": self.variant,
            "category": self.category,
            "price_minor": self.price_minor,
            "currency": self.currency,
            "in_stock": self.in_stock,
        }

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    filters: Mapping[str, Any] = field(default_factory=dict)
    sort: str = "price_asc"
    cursor: str | None = None
    page_size: int = 20


@dataclass(frozen=True, slots=True)
class SearchPage:
    items: tuple[CatalogListing, ...]
    total_hits: int
    next_cursor: str | None
    applied_filters: Mapping[str, Any]
    applied_sort: str
    unseen_price_lower_bound_minor: int | None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "items": [row.public_summary() for row in self.items],
            "total_hits": self.total_hits,
            "next_cursor": self.next_cursor,
            "applied_filters": dict(self.applied_filters),
            "applied_sort": self.applied_sort,
            "unseen_price_lower_bound_minor": self.unseen_price_lower_bound_minor,
        }


@dataclass(frozen=True, slots=True)
class TaskPredicate:
    predicate_id: str
    stage: str
    description: str
    mandatory: bool = True


@dataclass(frozen=True, slots=True)
class LargeCatalogTask:
    task_id: str
    scenario: str
    family: str
    capability: str
    role: str
    prompt: str
    prompt_zh: str
    public_context: Mapping[str, Any]
    oracle: Mapping[str, Any]
    allowed_intents: tuple[str, ...]
    predicates: tuple[TaskPredicate, ...]

    def to_dict(self, *, include_oracle: bool = True) -> dict[str, Any]:
        row = {
            "task_id": self.task_id,
            "scenario": self.scenario,
            "family": self.family,
            "capability": self.capability,
            "role": self.role,
            "prompt": self.prompt,
            "prompt_zh": self.prompt_zh,
            "public_context": dict(self.public_context),
            "allowed_intents": list(self.allowed_intents),
            "predicates": [asdict(item) for item in self.predicates],
        }
        if include_oracle:
            row["oracle"] = dict(self.oracle)
        return row


@dataclass(frozen=True, slots=True)
class ProcessReward:
    stage: str
    predicate: str
    points: int
    maximum: int
    event_ref: str | None


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    model_id: str
    role: str
    score: float
    strict_success: bool
    terminal: str
    process_rewards: tuple[ProcessReward, ...]
    trace: tuple[Mapping[str, Any], ...]
    model_calls: int
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "role": self.role,
            "score": self.score,
            "strict_success": self.strict_success,
            "terminal": self.terminal,
            "process_rewards": [asdict(item) for item in self.process_rewards],
            "trace": [dict(item) for item in self.trace],
            "model_calls": self.model_calls,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }
