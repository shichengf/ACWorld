"""Deterministic construction of sixty diverse large-catalog tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from large_catalog.database import CatalogDatabase
from large_catalog.models import CatalogListing, LargeCatalogTask, TaskPredicate
from large_catalog.oracle import (
    CartOracle,
    cart_oracle,
    comparison_oracle,
    independently_verify_cart,
    selection_oracle,
)


class TaskBuildError(RuntimeError):
    """The available catalog cannot support the fixed suite design."""


READ_INTENTS = ("search_catalog", "observe_listing")
SELECTION_INTENTS = (*READ_INTENTS, "select_listing", "reject_purchase")
ANSWER_INTENTS = (*READ_INTENTS, "submit_product_answer")
COMPARISON_INTENTS = (*READ_INTENTS, "submit_comparison")
CART_INTENTS = (
    "search_catalog",
    "search_cart_candidates",
    "observe_listing",
    "submit_cart",
    "reject_purchase",
)
CHECKOUT_INTENTS = (
    *CART_INTENTS,
    "request_cart_quote",
    "checkout_quote",
    "decline_quote",
)
CLAIM_INTENTS = (*READ_INTENTS, "publish_comparison_claim", "decline_comparison_claim")
QUOTE_INTENTS = (*READ_INTENTS, "submit_quote")

# Human-readable concepts selected from the prepared catalog. This cart subset
# stays bounded so exact combination search remains inexpensive.
CURATED_CART_TERMS = (
    "retinol", "hydrogel", "starlit", "clementine", "bulb", "barware",
    "toothbrush", "napkins", "grinder", "goggles", "potato", "stationery",
    "tofu", "peptide", "ceramide", "windproof", "seafood", "ramen",
    "octopus", "nightstand", "carrot", "toner", "puzzle",
)


def _predicates(
    kind: str,
    *,
    include_search: bool,
    answer_fields: Sequence[str] = (),
) -> tuple[TaskPredicate, ...]:
    search = (
        TaskPredicate("search_preserves_constraints", "search", "Search preserves hard constraints."),
        TaskPredicate("search_supports_global_judgment", "search", "Search evidence supports the final judgment."),
    )
    evidence = (
        TaskPredicate("decisive_records_observed", "evidence", "Decisive listings are read from the World."),
        TaskPredicate("evidence_matches_decision", "evidence", "Cited evidence supports the submitted decision."),
    )
    common = (search if include_search else ()) + evidence
    if kind == "answer":
        field_checks = tuple(
            TaskPredicate(
                f"answer_field_correct.{field}",
                "decision",
                f"The {field} answer matches the authoritative record.",
            )
            for field in answer_fields
        )
        return common + (
            TaskPredicate(
                "answer_targets_requested_listing",
                "decision",
                "The answer refers to the requested listing.",
            ),
        ) + field_checks + (
            TaskPredicate("platform_accepts_valid_action", "validation", "Platform validation has the expected result."),
        )
    if kind in {"comparison", "claim", "quote"}:
        return common + (
            TaskPredicate("structured_content_correct", "decision", "The structured answer is correct."),
            TaskPredicate("unsupported_content_omitted", "decision", "No unsupported fact is asserted."),
            TaskPredicate("decision_matches_oracle", "decision", "The decision matches the full-catalog oracle."),
            TaskPredicate("platform_accepts_valid_action", "validation", "Platform validation has the expected result."),
        )
    if kind == "reject":
        return common + (
            TaskPredicate("infeasibility_correct", "decision", "The full catalog is infeasible."),
            TaskPredicate("rejection_reason_correct", "decision", "The rejection reason matches the evidence."),
            TaskPredicate("decision_matches_oracle", "decision", "The decision matches the full-catalog oracle."),
            TaskPredicate("platform_accepts_valid_action", "validation", "Platform validation has the expected result."),
        )
    return common + (
        TaskPredicate("hard_constraints_satisfied", "decision", "The selected result is feasible."),
        TaskPredicate("preference_logic_correct", "decision", "The stated preference or combination rule is followed."),
        TaskPredicate("decision_matches_oracle", "decision", "The decision belongs to the global accepted set."),
        TaskPredicate("platform_accepts_valid_action", "validation", "Platform validation has the expected result."),
        TaskPredicate("world_effect_matches_decision", "world_effect", "The World records the selected result."),
        TaskPredicate("world_effect_matches_oracle", "world_effect", "The World result matches the oracle."),
    )


def _money(minor: int) -> str:
    return f"${minor / 100:.2f}"


def _term_label(value: str) -> str:
    return value.replace("_", " ")


class TaskBuilder:
    """Build tasks from actual rows, never from fabricated products."""

    def __init__(self, database: CatalogDatabase) -> None:
        self.database = database
        self._cart_terms = self._eligible_terms(
            CURATED_CART_TERMS,
            maximum_candidates=120,
        )
        self._cart_term_index = 0
        self._anchors = database.diverse_listings(limit=240)
        self._out_of_stock_anchors = database.diverse_listings(
            limit=24,
            in_stock=False,
        )
        self._high_price_anchors = database.diverse_listings(
            limit=24,
            min_price_minor=50_000,
            max_price_minor=100_000_000,
        )
        if len(self._anchors) < 120:
            raise TaskBuildError("catalog lacks enough diverse in-stock anchors")
        if not self._out_of_stock_anchors:
            raise TaskBuildError("catalog lacks an out-of-stock factual example")
        if not self._high_price_anchors:
            raise TaskBuildError("catalog lacks a high-price factual example")
        self._anchor_index = 0
        self._tasks: list[LargeCatalogTask] = []

    def build(self) -> tuple[LargeCatalogTask, ...]:
        self._build_search()
        self._build_facts()
        self._build_preferences()
        self._build_carts()
        self._build_merchant()
        tasks = tuple(self._tasks)
        self._validate_suite(tasks)
        return tasks

    def _eligible_terms(
        self,
        candidates: Sequence[str],
        *,
        maximum_candidates: int,
    ) -> tuple[str, ...]:
        terms: list[str] = []
        for term in candidates:
            rows = self.database.full_candidates(
                query=term,
                filters={"in_stock": True},
                sort="price_asc",
            )
            if (
                8 <= len(rows) <= maximum_candidates
                and len({row.merchant_id for row in rows}) >= 2
                and any(row.category for row in rows)
            ):
                terms.append(term)
        required = min(20, len(candidates))
        if len(terms) < required:
            raise TaskBuildError(
                f"catalog supports only {len(terms)}/{required} curated search terms"
            )
        return tuple(terms)

    def _next_cart_term(self) -> tuple[str, tuple[CatalogListing, ...]]:
        term = self._cart_terms[self._cart_term_index % len(self._cart_terms)]
        self._cart_term_index += 1
        rows = self.database.full_candidates(
            query=term,
            filters={"in_stock": True},
            sort="price_asc",
        )
        return term, rows

    def _rows_for(self, term: str) -> tuple[CatalogListing, ...]:
        rows = self.database.full_candidates(
            query=term,
            filters={"in_stock": True},
            sort="price_asc",
        )
        if not rows:
            raise TaskBuildError(f"curated query has no in-stock result: {term}")
        return rows

    def _next_anchor(self) -> CatalogListing:
        row = self._anchors[self._anchor_index % len(self._anchors)]
        self._anchor_index += 1
        return row

    def _add(
        self,
        *,
        code: str,
        index: int,
        family: str,
        capability: str,
        role: str,
        prompt: str,
        prompt_zh: str,
        public_context: Mapping[str, Any],
        oracle: Mapping[str, Any],
        intents: Sequence[str],
        score_kind: str,
    ) -> None:
        self._tasks.append(
            LargeCatalogTask(
                task_id=f"LC-{code}-{index:02d}",
                scenario=code,
                family=family,
                capability=capability,
                role=role,
                prompt=prompt,
                prompt_zh=prompt_zh,
                public_context=dict(public_context),
                oracle=dict(oracle),
                allowed_intents=tuple(intents),
                predicates=_predicates(
                    score_kind,
                    include_search=not (
                        "listing_refs" in public_context
                        or "quote_lines" in public_context
                    ),
                    answer_fields=(
                        tuple(str(field) for field in public_context["requested_fields"])
                        if score_kind == "answer"
                        else ()
                    ),
                ),
            )
        )

    def _build_search(self) -> None:
        # A1: ordinary full-catalog discovery with progressively richer hard filters.
        search_specs = (
            ("retinol", 800, None),
            ("hydrogel", 1_200, "Skin Care"),
            ("starlit", 1_000, None),
        )
        for index, (term, budget, category) in enumerate(search_specs, start=1):
            self._rows_for(term)
            filters: dict[str, Any] = {
                "in_stock": True,
                "max_price_minor": budget,
            }
            qualifier = ""
            qualifier_zh = ""
            if category is not None:
                filters["category"] = category
                qualifier = f" in {category}"
                qualifier_zh = f"，类别为“{category}”"
            oracle = selection_oracle(
                self.database, query=term, filters=filters
            )
            self._require_nonempty(oracle.accepted_refs, "A1")
            self._add(
                code="A1",
                index=index,
                family="Discovery",
                capability="t1.basic_feasible_discovery",
                role="buyer",
                prompt=(
                    f"Find an in-stock {_term_label(term)} product{qualifier} for no "
                    f"more than {_money(budget)}. Choose the least "
                    "expensive option that fits."
                ),
                prompt_zh=(
                    f"帮我找一个有货的“{_term_label(term)}”商品{qualifier_zh}，预算不超过"
                    f"{_money(budget)}。请选择符合要求的最低价商品。"
                ),
                public_context={"query": term, "filters": filters},
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="selection",
            )

        # A2: category preference with direct, fallback, and reject outcomes.
        fallback_specs = (
            ("retinol", "Health & Beauty", 1_500),
            ("collagen", "Skin Care", 600),
            ("hydrogel", "eye masks", 100),
        )
        for index, (term, preferred, budget) in enumerate(fallback_specs, start=1):
            self._rows_for(term)
            filters = {"in_stock": True, "max_price_minor": budget}
            oracle = selection_oracle(
                self.database,
                query=term,
                filters=filters,
                preference={"kind": "category_then_price", "category": preferred},
            )
            outcome = "reject" if not oracle.accepted_refs else "selection"
            self._add(
                code="A2",
                index=index,
                family="Discovery",
                capability="t1.query_reformulation",
                role="buyer",
                prompt=(
                    f"I need an in-stock {_term_label(term)} product for no more than "
                    f"{_money(budget)}. I prefer {preferred}, but another category is "
                    "acceptable if no preferred option fits. Do not buy if nothing "
                    "meets the budget."
                ),
                prompt_zh=(
                    f"我需要一个有货的“{_term_label(term)}”商品，预算不超过{_money(budget)}。"
                    f"我更偏好“{preferred}”类别，但如果没有合适商品，其他类别也可以。"
                    "如果预算内没有商品，就不要购买。"
                ),
                public_context={
                    "query": term,
                    "filters": filters,
                    "preference": {
                        "kind": "category_then_price",
                        "category": preferred,
                    },
                },
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind=outcome,
            )

        # A3: natural priorities over the complete feasible set.
        preference_specs = (
            ("vegan", 1_000, "lowest_price", None),
            ("clementine", 2_500, "category_then_price", "Décor"),
            ("filter", 2_000, "feature_then_price", "reusable"),
        )
        for index, (term, budget, kind, value) in enumerate(preference_specs, start=1):
            self._rows_for(term)
            filters = {"in_stock": True, "max_price_minor": budget}
            preference: dict[str, Any] = {"kind": kind}
            if kind == "category_then_price":
                preference["category"] = value
                priority = f"Prefer {preference['category']}; then choose the lowest price."
                priority_zh = f"优先选择“{preference['category']}”类别，再比较价格。"
            elif kind == "feature_then_price":
                feature = str(value)
                preference["feature"] = feature
                priority = f"Prefer a listing that mentions {feature}; then compare prices."
                priority_zh = f"优先选择商品信息中提到“{feature}”的商品，再比较价格。"
            else:
                priority = "Price matters most."
                priority_zh = "价格最重要。"
            oracle = selection_oracle(
                self.database, query=term, filters=filters, preference=preference
            )
            self._require_nonempty(oracle.accepted_refs, "A3")
            self._add(
                code="A3",
                index=index,
                family="Discovery",
                capability="t1.best_feasible_selection",
                role="buyer",
                prompt=(
                    f"Find an in-stock {_term_label(term)} product under {_money(budget)}. "
                    f"{priority}"
                ),
                prompt_zh=(
                    f"帮我找一个有货、价格低于{_money(budget)}的“{_term_label(term)}”商品。"
                    f"{priority_zh}"
                ),
                public_context={
                    "query": term,
                    "filters": filters,
                    "preference": preference,
                },
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="selection",
            )

        # A4: global proof of infeasibility rather than first-page absence.
        reject_specs = (("retinol", 50), ("hydrogel", 50), ("starlit", 25))
        for index, (term, budget) in enumerate(reject_specs, start=1):
            self._rows_for(term)
            filters = {"in_stock": True, "max_price_minor": budget}
            oracle = selection_oracle(self.database, query=term, filters=filters)
            if oracle.accepted_refs:
                raise TaskBuildError("A4 rejection task unexpectedly has a feasible result")
            self._add(
                code="A4",
                index=index,
                family="Discovery",
                capability="t1.correct_abstention",
                role="buyer",
                prompt=(
                    f"I need an in-stock {_term_label(term)} product for no more than "
                    f"{_money(budget)}. Do not choose an over-budget or unavailable item."
                ),
                prompt_zh=(
                    f"我需要一个有货的“{_term_label(term)}”商品，预算不超过{_money(budget)}。"
                    "不要选择超预算或缺货的商品。"
                ),
                public_context={"query": term, "filters": filters},
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="reject",
            )

    def _build_facts(self) -> None:
        # B1: exact authoritative attributes, including one intentionally missing field.
        requested_sets = (
            ("price_minor", "in_stock"),
            ("category", "key_features"),
            ("material", "warranty"),
        )
        for index, fields in enumerate(requested_sets, start=1):
            row = (
                self._out_of_stock_anchors[0]
                if index == 3
                else self._high_price_anchors[0]
                if index == 2
                else self._next_anchor()
            )
            expected = {
                field: getattr(row, field) if getattr(row, field) not in {"", None} else None
                for field in fields
            }
            self._add(
                code="B1",
                index=index,
                family="Grounding",
                capability="t2.authoritative_attribute_read",
                role="buyer",
                prompt=(
                    f"Please check “{row.name}” ({self._variant_label(row.variant)}) and "
                    f"report its {self._natural_fields(fields)}. Report only details in "
                    "the current product record."
                ),
                prompt_zh=(
                    f"请核对“{row.name}”（{self._variant_label(row.variant, chinese=True)}），并报告"
                    f"{self._natural_fields_zh(fields)}。只报告当前商品记录中确实提供的信息。"
                ),
                public_context={"listing_refs": [row.listing_ref], "requested_fields": list(fields)},
                oracle={
                    "kind": "answer",
                    "listing_ref": row.listing_ref,
                    "expected_fields": expected,
                },
                intents=ANSWER_INTENTS,
                score_kind="answer",
            )

        # B2: pairwise comparisons with separately scoreable clauses.
        for index in range(1, 4):
            left = self._next_anchor()
            right = self._next_anchor()
            oracle = comparison_oracle(self.database, (left.listing_ref, right.listing_ref))
            prompt_tail = (
                "Which one has the lower listed price?"
                if index == 1
                else "Which is cheaper, and are both in stock?"
                if index == 2
                else "Are they in the same category, and is their price difference no more than $5?"
            )
            zh_tail = (
                "哪一款当前标价更低？"
                if index == 1
                else "哪一款更便宜，而且两款商品是否都有货？"
                if index == 2
                else "它们是否属于同一类别，而且价格差是否不超过5美元？"
            )
            self._add(
                code="B2",
                index=index,
                family="Grounding",
                capability="t2.grounded_comparison",
                role="buyer",
                prompt=(
                    f"Compare “{left.name}” ({self._variant_label(left.variant)}) with "
                    f"“{right.name}” ({self._variant_label(right.variant)}). {prompt_tail} "
                    "Check both records before answering."
                ),
                prompt_zh=(
                    f"比较“{left.name}”（{self._variant_label(left.variant, chinese=True)}）和"
                    f"“{right.name}”（{self._variant_label(right.variant, chinese=True)}）。"
                    f"{zh_tail}回答前请核对两条记录。"
                ),
                public_context={"listing_refs": [left.listing_ref, right.listing_ref]},
                oracle=oracle,
                intents=COMPARISON_INTENTS,
                score_kind="comparison",
            )

        # B3: real same-name variants are not collapsed into one product.
        pairs = self._variant_pairs(3)
        for index, (left, right) in enumerate(pairs, start=1):
            oracle = comparison_oracle(self.database, (left.listing_ref, right.listing_ref))
            self._add(
                code="B3",
                index=index,
                family="Grounding",
                capability="t2.conflict_and_normalization",
                role="buyer",
                prompt=(
                    f"These two listings are both called “{left.name}”, but they may be "
                    "different variants. Check each record, identify the variant and "
                    "price, and tell me whether they are actually the same offer."
                ),
                prompt_zh=(
                    f"这两条商品都叫“{left.name}”，但可能是不同变体。请分别核对记录，"
                    "说明各自的变体和价格，并判断它们是否真的是同一个报价。"
                ),
                public_context={"listing_refs": [left.listing_ref, right.listing_ref]},
                oracle={
                    **oracle,
                    "expected_variants": [left.variant, right.variant],
                    "expected_prices": [left.price_minor, right.price_minor],
                },
                intents=COMPARISON_INTENTS,
                score_kind="comparison",
            )

        # B4: merchant may answer only from its own listing evidence.
        fields_by_index = (
            ("price_minor", "in_stock"),
            ("material", "warranty"),
            ("variant", "category", "price_minor"),
        )
        for index, fields in enumerate(fields_by_index, start=1):
            row = self._next_anchor()
            expected = {
                field: getattr(row, field) if getattr(row, field) not in {"", None} else None
                for field in fields
            }
            self._add(
                code="B4",
                index=index,
                family="Grounding",
                capability="t2.evidence_backed_response",
                role="merchant",
                prompt=(
                    f"A customer asks about your listing “{row.name}” "
                    f"({self._variant_label(row.variant)}). Answer its "
                    f"{self._natural_fields(fields)} using your current listing record. "
                    "Say when a requested detail is not provided."
                ),
                prompt_zh=(
                    f"顾客询问你发布的“{row.name}”"
                    f"（{self._variant_label(row.variant, chinese=True)}）。请根据当前"
                    f"商品记录回答其{self._natural_fields_zh(fields)}。记录未提供的信息要明确说明。"
                ),
                public_context={"listing_refs": [row.listing_ref], "requested_fields": list(fields)},
                oracle={
                    "kind": "answer",
                    "listing_ref": row.listing_ref,
                    "expected_fields": expected,
                },
                intents=ANSWER_INTENTS,
                score_kind="answer",
            )

    def _build_preferences(self) -> None:
        # C1: a stated dollar premium, not hidden utility points.
        premium_specs = (
            ("retinol", "Serum", 2_000, 1_400),
            ("collagen", "Skin Care", 2_000, 100),
            ("clementine", "Skin Care", 2_500, 300),
        )
        for index, (term, preferred, budget, premium) in enumerate(
            premium_specs, start=1
        ):
            self._rows_for(term)
            filters = {"in_stock": True, "max_price_minor": budget}
            preference = {
                "kind": "category_premium",
                "category": preferred,
                "premium_minor": premium,
            }
            oracle = selection_oracle(
                self.database, query=term, filters=filters, preference=preference
            )
            self._require_nonempty(oracle.accepted_refs, "C1")
            self._add(
                code="C1",
                index=index,
                family="Preference",
                capability="t3.weighted_soft_preferences",
                role="buyer",
                prompt=(
                    f"Find an in-stock {_term_label(term)} product under {_money(budget)}. "
                    f"I prefer {preferred}, but I will pay at most {_money(premium)} more "
                    "than the cheapest eligible option for that preference."
                ),
                prompt_zh=(
                    f"帮我找一个有货、价格低于{_money(budget)}的“{_term_label(term)}”商品。"
                    f"我更偏好“{preferred}”，但相较最低价合格商品，我最多只愿意为该偏好"
                    f"多付{_money(premium)}。"
                ),
                public_context={"query": term, "filters": filters, "preference": preference},
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="selection",
            )

        # C2: hard budget and stock constraints dominate a named soft preference.
        hard_soft_specs = (
            ("retinol", "Serum", 800),
            ("filter", "Coffee", 700),
            ("filter", "Coffee", 50),
        )
        for index, (term, preferred, budget) in enumerate(
            hard_soft_specs, start=1
        ):
            self._rows_for(term)
            filters = {"in_stock": True, "max_price_minor": budget}
            preference = {"kind": "category_then_price", "category": preferred}
            oracle = selection_oracle(
                self.database, query=term, filters=filters, preference=preference
            )
            self._add(
                code="C2",
                index=index,
                family="Preference",
                capability="t3.hard_over_soft",
                role="buyer",
                prompt=(
                    f"I would like a {preferred} {_term_label(term)} product, but it must "
                    f"be in stock and cost no more than {_money(budget)}. Choose another "
                    "eligible category if needed; buy nothing if every option breaks a "
                    "hard requirement."
                ),
                prompt_zh=(
                    f"我更想要“{preferred}”类别的“{_term_label(term)}”商品，但商品必须有货，"
                    f"且价格不超过{_money(budget)}。必要时可选其他合格类别；如果都违反硬性要求，"
                    "就不要购买。"
                ),
                public_context={"query": term, "filters": filters, "preference": preference},
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="reject" if not oracle.accepted_refs else "selection",
            )

        # C3: the latest natural-language instruction supersedes the earlier preference.
        update_specs = (
            ("hydrogel", "Skin Care", 1_500),
            ("collagen", "Skin Care", 700),
            ("filter", "Coffee", 1_000),
        )
        for index, (term, initial, budget) in enumerate(update_specs, start=1):
            self._rows_for(term)
            if index == 1:
                final_preference = {
                    "kind": "category_then_price",
                    "category": "eye masks",
                }
                update = f"Actually, prefer {final_preference['category']} instead."
                update_zh = f"不过我改主意了，请优先选择“{final_preference['category']}”。"
            elif index == 2:
                final_preference = {"kind": "lowest_price"}
                update = f"Keep the request, but lower my budget to {_money(budget)}."
                update_zh = f"需求不变，但请把预算收紧到{_money(budget)}。"
            else:
                final_preference = {"kind": "lowest_price"}
                update = "I no longer care about category; choose the cheapest eligible item."
                update_zh = "我不再在意类别，请选择最低价的合格商品。"
            filters = {"in_stock": True, "max_price_minor": budget}
            oracle = selection_oracle(
                self.database,
                query=term,
                filters=filters,
                preference=final_preference,
            )
            self._require_nonempty(oracle.accepted_refs, "C3")
            self._add(
                code="C3",
                index=index,
                family="Preference",
                capability="t3.preference_update",
                role="buyer",
                prompt=(
                    f"Find an in-stock {_term_label(term)} product under {_money(budget)}. "
                    f"I initially preferred {initial}. {update}"
                ),
                prompt_zh=(
                    f"帮我找一个有货、价格低于{_money(budget)}的“{_term_label(term)}”商品。"
                    f"我原本偏好“{initial}”。{update_zh}"
                ),
                public_context={
                    "query": term,
                    "filters": filters,
                    "latest_preference": final_preference,
                },
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="selection",
            )

        # C4: explicit lexicographic preferences.
        consistent_specs = (
            ("retinol", 2_000, "category_then_price", "Serum"),
            ("filter", 1_500, "feature_then_price", "reusable"),
            ("vegan", 1_200, "merchant_then_price", None),
        )
        for index, (term, budget, kind, value) in enumerate(
            consistent_specs, start=1
        ):
            rows = self._rows_for(term)
            preference: dict[str, Any] = {"kind": kind}
            if kind == "category_then_price":
                preference["category"] = value
                phrase = f"Prefer {preference['category']}, then choose the lowest price."
                phrase_zh = f"先优先选择“{preference['category']}”，再选择价格最低的商品。"
            elif kind == "feature_then_price":
                preference["feature"] = value
                phrase = f"Prefer a listing mentioning {preference['feature']}, then price."
                phrase_zh = f"先优先选择提到“{preference['feature']}”的商品，再比较价格。"
            else:
                preference["merchant"] = rows[1].merchant_id
                merchant_label = self._merchant_label(preference["merchant"])
                phrase = f"Prefer merchant {merchant_label}, then price."
                phrase_zh = f"先优先选择商家“{merchant_label}”，再比较价格。"
            filters = {"in_stock": True, "max_price_minor": budget}
            oracle = selection_oracle(
                self.database, query=term, filters=filters, preference=preference
            )
            self._require_nonempty(oracle.accepted_refs, "C4")
            self._add(
                code="C4",
                index=index,
                family="Preference",
                capability="t3.mandate_consistency",
                role="buyer",
                prompt=(
                    f"Find an in-stock {_term_label(term)} product under {_money(budget)}. "
                    f"{phrase}"
                ),
                prompt_zh=(
                    f"帮我找一个有货、价格低于{_money(budget)}的“{_term_label(term)}”商品。"
                    f"{phrase_zh}"
                ),
                public_context={"query": term, "filters": filters, "preference": preference},
                oracle=oracle.to_dict(),
                intents=SELECTION_INTENTS,
                score_kind="selection",
            )

    def _build_carts(self) -> None:
        # D1: require a cart spanning at least two merchants.
        for index, lines in enumerate((2, 3, 3), start=1):
            requirements = self._requirements(lines)
            unconstrained = cart_oracle(
                self.database,
                requirements=requirements,
                constraints={"min_merchants": 2, "distinct_listings": True},
            )
            if unconstrained.objective_total_minor is None:
                raise TaskBuildError("D1 has no cross-merchant cart")
            budget = (
                unconstrained.objective_total_minor - 1
                if index == 3
                else unconstrained.objective_total_minor + 200
            )
            constraints = {
                "budget_minor": budget,
                "min_merchants": 2,
                "distinct_listings": True,
            }
            oracle = cart_oracle(
                self.database, requirements=requirements, constraints=constraints
            )
            self._verify_cart(requirements, constraints, oracle)
            needs = self._requirement_phrase(requirements)
            needs_zh = self._requirement_phrase_zh(requirements)
            self._add_cart(
                code="D1",
                index=index,
                capability="t5.cross_merchant_cart",
                requirements=requirements,
                constraints=constraints,
                oracle=oracle,
                prompt=(
                    f"Buy one in-stock item for each of these needs: {needs}. Use at least "
                    f"two merchants, keep the listed total within {_money(budget)}, and "
                    "choose the least expensive valid cart."
                ),
                prompt_zh=(
                    f"请为以下需求各购买一件有货商品：{needs_zh}。至少选择两家商家，"
                    f"商品标价总额不得超过{_money(budget)}，并选择最低价的可行购物车。"
                ),
            )

        # D2: three to five requirements under one total budget.
        for index, lines in enumerate((3, 4, 5), start=1):
            requirements = self._requirements(lines)
            unconstrained = cart_oracle(
                self.database,
                requirements=requirements,
                constraints={"distinct_listings": True},
            )
            if unconstrained.objective_total_minor is None:
                raise TaskBuildError("D2 has no cart")
            budget = (
                unconstrained.objective_total_minor - 1
                if index == 3
                else unconstrained.objective_total_minor + (100 * index)
            )
            constraints = {"budget_minor": budget, "distinct_listings": True}
            oracle = cart_oracle(
                self.database, requirements=requirements, constraints=constraints
            )
            self._verify_cart(requirements, constraints, oracle)
            needs = self._requirement_phrase(requirements)
            needs_zh = self._requirement_phrase_zh(requirements)
            self._add_cart(
                code="D2",
                index=index,
                capability="t5.total_budget",
                requirements=requirements,
                constraints=constraints,
                oracle=oracle,
                prompt=(
                    f"Choose one in-stock item for each of these needs: {needs}. Keep the "
                    f"whole cart within {_money(budget)} and minimize the listed total."
                ),
                prompt_zh=(
                    f"请为以下需求各选择一件有货商品：{needs_zh}。整个购物车不得超过"
                    f"{_money(budget)}，并使商品标价总额最低。"
                ),
            )

        # D3: one requirement explicitly allows a substitute query.
        for index in range(1, 4):
            primary, primary_rows = self._next_cart_term()
            primary_refs = {row.listing_ref for row in primary_rows}
            while True:
                alternate, alternate_rows = self._next_cart_term()
                alternate_refs = {row.listing_ref for row in alternate_rows}
                if not primary_refs.intersection(alternate_refs):
                    break
            combined_refs = primary_refs | alternate_refs
            while True:
                companion, companion_rows = self._next_cart_term()
                companion_refs = {row.listing_ref for row in companion_rows}
                if not combined_refs.intersection(companion_refs):
                    break
            requirements = [
                {
                    "label": f"{primary} or {alternate}",
                    "queries": [primary, alternate],
                    "filters": {"in_stock": True},
                },
                {
                    "label": companion,
                    "query": companion,
                    "filters": {"in_stock": True},
                },
            ]
            unconstrained = cart_oracle(
                self.database,
                requirements=requirements,
                constraints={"distinct_listings": True},
            )
            if unconstrained.objective_total_minor is None:
                raise TaskBuildError("D3 has no substitute cart")
            budget = (
                unconstrained.objective_total_minor - 1
                if index == 3
                else unconstrained.objective_total_minor + 150
            )
            constraints = {"budget_minor": budget, "distinct_listings": True}
            oracle = cart_oracle(
                self.database, requirements=requirements, constraints=constraints
            )
            self._verify_cart(requirements, constraints, oracle)
            self._add_cart(
                code="D3",
                index=index,
                capability="t5.bundle_relations",
                requirements=requirements,
                constraints=constraints,
                oracle=oracle,
                prompt=(
                    f"I need one {_term_label(primary)} item and one "
                    f"{_term_label(companion)} item. If no suitable {_term_label(primary)} "
                    f"fits, {_term_label(alternate)} is an acceptable substitute. Keep "
                    f"the cart within {_money(budget)}."
                ),
                prompt_zh=(
                    f"我需要一个“{_term_label(primary)}”商品和一个“{_term_label(companion)}”"
                    f"商品。如果没有合适的“{_term_label(primary)}”，“{_term_label(alternate)}”"
                    f"可以作为替代。购物车总额不得超过{_money(budget)}。"
                ),
            )

        # D4: authoritative fees determine checkout versus decline.
        for index in range(1, 4):
            requirements = self._requirements(2)
            cart = cart_oracle(
                self.database,
                requirements=requirements,
                constraints={"distinct_listings": True},
            )
            if cart.objective_total_minor is None:
                raise TaskBuildError("D4 has no quoteable cart")
            fee = (125, 975, 350)[index - 1]
            discount = (0, 0, 500)[index - 1]
            quote_total = cart.objective_total_minor + fee - discount
            budget = quote_total + 100 if index in {1, 3} else quote_total - 1
            expected_action = "checkout_quote" if quote_total <= budget else "decline_quote"
            needs = self._requirement_phrase(requirements)
            needs_zh = self._requirement_phrase_zh(requirements)
            oracle_row = {
                **cart.to_dict(),
                "kind": "quote_decision",
                "fee_minor": fee,
                "discount_minor": discount,
                "quote_total_minor": quote_total,
                "budget_minor": budget,
                "expected_action": expected_action,
            }
            self._add(
                code="D4",
                index=index,
                family="Multi-item",
                capability="t5.total_budget",
                role="buyer",
                prompt=(
                    f"Build the least expensive cart for these needs: {needs}. My final "
                    f"budget, including fees and discounts, is {_money(budget)}. Get the "
                    "current quote and check out only if the total fits."
                ),
                prompt_zh=(
                    f"请为以下需求构造最低价购物车：{needs_zh}。我的最终预算（包括费用和折扣）"
                    f"为{_money(budget)}。请先获取当前报价，只有总额在预算内时才结算。"
                ),
                public_context={
                    "requirements": requirements,
                    "constraints": {"distinct_listings": True},
                    "budget_minor": budget,
                },
                oracle=oracle_row,
                intents=CHECKOUT_INTENTS,
                score_kind="selection",
            )

    def _build_merchant(self) -> None:
        # E1: merchant product inquiry.
        for index in range(1, 4):
            row = self._next_anchor()
            fields = (
                ("price_minor", "in_stock"),
                ("material", "warranty"),
                ("variant", "price_minor"),
            )[index - 1]
            expected = {
                field: getattr(row, field) if getattr(row, field) not in {"", None} else None
                for field in fields
            }
            self._add(
                code="E1",
                index=index,
                family="Grounding",
                capability="t2.evidence_backed_response",
                role="merchant",
                prompt=(
                    f"A buyer asks about your listing “{row.name}”. Reply with its "
                    f"{self._natural_fields(fields)}. Use the listing record and do not "
                    "fill in a detail that is absent."
                ),
                prompt_zh=(
                    f"买家询问你发布的“{row.name}”。请根据商品记录回答其"
                    f"{self._natural_fields_zh(fields)}，不要补写记录中不存在的信息。"
                ),
                public_context={"listing_refs": [row.listing_ref], "requested_fields": list(fields)},
                oracle={
                    "kind": "answer",
                    "listing_ref": row.listing_ref,
                    "expected_fields": expected,
                },
                intents=ANSWER_INTENTS,
                score_kind="answer",
            )

        # E2: only evidence-supported comparative claims may be published.
        for index in range(1, 4):
            left = self._next_anchor()
            right = self._next_anchor()
            comparison = comparison_oracle(
                self.database, (left.listing_ref, right.listing_ref)
            )
            if index == 1:
                claim = {"type": "lower_price", "subject_ref": comparison["lower_price_ref"]}
                supported = comparison["lower_price_ref"] is not None
                subject = (
                    left
                    if comparison["lower_price_ref"] == left.listing_ref
                    else right
                )
                other = right if subject is left else left
                claim_text = (
                    f"“{subject.name}” currently costs less than “{other.name}”"
                )
                claim_text_zh = f"“{subject.name}”当前价格低于“{other.name}”"
            elif index == 2:
                claim = {"type": "both_in_stock"}
                supported = bool(comparison["both_in_stock"])
                claim_text = (
                    f"both “{left.name}” and “{right.name}” are currently in stock"
                )
                claim_text_zh = f"“{left.name}”和“{right.name}”目前都有货"
            else:
                claim = {
                    "type": "price_difference_at_most",
                    "threshold_minor": comparison["price_difference_minor"],
                }
                supported = True
                threshold = _money(comparison["price_difference_minor"])
                claim_text = (
                    f"the listed prices of “{left.name}” and “{right.name}” differ "
                    f"by no more than {threshold}"
                )
                claim_text_zh = (
                    f"“{left.name}”和“{right.name}”的标价差不超过{threshold}"
                )
            self._add(
                code="E2",
                index=index,
                family="Grounding",
                capability="t2.comparative_claim_limits",
                role="merchant",
                prompt=(
                    f"We would like to tell customers that {claim_text}. Check both "
                    "current listings and publish the comparison only if it is true."
                ),
                prompt_zh=(
                    f"我们想告诉顾客，{claim_text_zh}。请核对当前两条商品记录，"
                    "确认属实后再发布。"
                ),
                public_context={
                    "listing_refs": [left.listing_ref, right.listing_ref],
                    "requested_claim": claim,
                },
                oracle={
                    "kind": "claim",
                    "expected_action": "publish_comparison_claim" if supported else "decline_comparison_claim",
                    "claim": claim,
                    "comparison": comparison,
                },
                intents=CLAIM_INTENTS,
                score_kind="claim",
            )

        # E3: deliberately unsupported marketing claims must be declined.
        for index in range(1, 4):
            left = self._next_anchor()
            right = self._next_anchor()
            comparison = comparison_oracle(
                self.database, (left.listing_ref, right.listing_ref)
            )
            false_claim = (
                {"type": "lower_price", "subject_ref": right.listing_ref}
                if comparison["lower_price_ref"] == left.listing_ref
                else {"type": "lower_price", "subject_ref": left.listing_ref}
                if comparison["lower_price_ref"] == right.listing_ref
                else {"type": "exclusive_lowest_price"}
            )
            if false_claim["type"] == "lower_price":
                subject = (
                    left
                    if false_claim["subject_ref"] == left.listing_ref
                    else right
                )
                other = right if subject is left else left
                false_text = (
                    f"“{subject.name}” currently costs less than “{other.name}”"
                )
                false_text_zh = f"“{subject.name}”当前价格低于“{other.name}”"
            else:
                false_text = (
                    f"“{left.name}” is the only lowest-priced option between "
                    "these two listings"
                )
                false_text_zh = f"“{left.name}”是这两条商品中唯一的最低价选择"
            self._add(
                code="E3",
                index=index,
                family="Grounding",
                capability="t2.correction_and_retraction",
                role="merchant",
                prompt=(
                    f"We are considering telling customers that {false_text}. Check both "
                    "current listings first. Do not publish the claim if it is false."
                ),
                prompt_zh=(
                    f"我们正在考虑告诉顾客，{false_text_zh}。请先核对当前两条商品记录；"
                    "如果该说法不实，就不要发布。"
                ),
                public_context={
                    "listing_refs": [left.listing_ref, right.listing_ref],
                    "requested_claim": false_claim,
                },
                oracle={
                    "kind": "claim",
                    "expected_action": "decline_comparison_claim",
                    "claim": false_claim,
                    "comparison": comparison,
                },
                intents=CLAIM_INTENTS,
                score_kind="claim",
            )

        # E4: exact multi-line arithmetic from authoritative prices.
        for index, line_count in enumerate((2, 3, 4), start=1):
            rows = [self._next_anchor() for _ in range(line_count)]
            quantities = [position + 1 for position in range(line_count)]
            subtotal = sum(row.price_minor * qty for row, qty in zip(rows, quantities))
            discount = 0 if index == 1 else subtotal // (20 if index == 2 else 10)
            fee = (200, 350, 500)[index - 1]
            total = subtotal - discount + fee
            lines = [
                {"listing_ref": row.listing_ref, "quantity": qty}
                for row, qty in zip(rows, quantities)
            ]
            self._add(
                code="E4",
                index=index,
                family="Multi-item",
                capability="t5.merchant_total_quote",
                role="merchant",
                prompt=(
                    f"Prepare a quote for the {line_count} requested listing lines. Apply "
                    f"the stated discount of {_money(discount)} and fee of {_money(fee)}, "
                    "then report each line amount and the exact final total."
                ),
                prompt_zh=(
                    f"请为这{line_count}条指定商品准备报价。应用{_money(discount)}的折扣和"
                    f"{_money(fee)}的费用，并报告每行金额及准确的最终总额。"
                ),
                public_context={"quote_lines": lines, "discount_minor": discount, "fee_minor": fee},
                oracle={
                    "kind": "quote",
                    "lines": [
                        {
                            **line,
                            "unit_price_minor": row.price_minor,
                            "line_total_minor": row.price_minor * line["quantity"],
                        }
                        for row, line in zip(rows, lines)
                    ],
                    "subtotal_minor": subtotal,
                    "discount_minor": discount,
                    "fee_minor": fee,
                    "total_minor": total,
                },
                intents=QUOTE_INTENTS,
                score_kind="quote",
            )

    def _add_cart(
        self,
        *,
        code: str,
        index: int,
        capability: str,
        requirements: Sequence[Mapping[str, Any]],
        constraints: Mapping[str, Any],
        oracle: CartOracle,
        prompt: str,
        prompt_zh: str,
    ) -> None:
        self._add(
            code=code,
            index=index,
            family="Multi-item",
            capability=capability,
            role="buyer",
            prompt=prompt,
            prompt_zh=prompt_zh,
            public_context={
                "requirements": list(requirements),
                "constraints": dict(constraints),
            },
            oracle=oracle.to_dict(),
            intents=CART_INTENTS,
            score_kind="reject" if not oracle.accepted_carts else "selection",
        )

    def _requirements(self, count: int) -> list[dict[str, Any]]:
        requirements: list[dict[str, Any]] = []
        used: set[str] = set()
        used_refs: set[str] = set()
        while len(requirements) < count:
            term, rows = self._next_cart_term()
            if term in used:
                continue
            refs = {row.listing_ref for row in rows}
            if refs.intersection(used_refs):
                continue
            used.add(term)
            used_refs.update(refs)
            requirements.append(
                {
                    "label": _term_label(term),
                    "query": term,
                    "filters": {"in_stock": True},
                }
            )
        return requirements

    def _verify_cart(
        self,
        requirements: Sequence[Mapping[str, Any]],
        constraints: Mapping[str, Any],
        oracle: CartOracle,
    ) -> None:
        if not independently_verify_cart(
            self.database,
            requirements=requirements,
            constraints=constraints,
            expected=oracle,
        ):
            raise TaskBuildError("independent cart solver disagrees with branch and bound")

    def _variant_pairs(self, count: int) -> tuple[tuple[CatalogListing, CatalogListing], ...]:
        rows = self.database.connection.execute(
            """
            SELECT merchant_id, lower(name) AS normalized_name
            FROM listings
            WHERE name <> '' AND variant <> ''
            GROUP BY merchant_id, lower(name)
            HAVING COUNT(*) >= 2 AND COUNT(DISTINCT lower(variant)) >= 2
            ORDER BY normalized_name, merchant_id
            LIMIT ?
            """,
            (count * 4,),
        ).fetchall()
        pairs: list[tuple[CatalogListing, CatalogListing]] = []
        for row in rows:
            variants = self.database.connection.execute(
                """
                SELECT * FROM listings
                WHERE merchant_id = ? AND lower(name) = ?
                ORDER BY lower(variant), price_minor, listing_ref
                LIMIT 2
                """,
                (row["merchant_id"], row["normalized_name"]),
            ).fetchall()
            if len(variants) == 2:
                from large_catalog.database import _row_to_listing

                pairs.append((_row_to_listing(variants[0]), _row_to_listing(variants[1])))
            if len(pairs) == count:
                break
        if len(pairs) != count:
            raise TaskBuildError("catalog lacks three unambiguous variant pairs")
        return tuple(pairs)

    @staticmethod
    def _requirement_phrase(requirements: Sequence[Mapping[str, Any]]) -> str:
        labels = [str(row["label"]) for row in requirements]
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return f"{', '.join(labels[:-1])}, and {labels[-1]}"

    @staticmethod
    def _requirement_phrase_zh(requirements: Sequence[Mapping[str, Any]]) -> str:
        return "、".join(f"“{row['label']}”" for row in requirements)

    @staticmethod
    def _merchant_label(merchant_id: str) -> str:
        head, separator, tail = merchant_id.rpartition("-")
        if separator and 2 <= len(tail) <= 4:
            return f"{head}.{tail}"
        return merchant_id

    @staticmethod
    def _variant_label(variant: str, *, chinese: bool = False) -> str:
        if not variant or variant.casefold() in {"default", "default title"}:
            return "标准款" if chinese else "standard version"
        return variant

    @staticmethod
    def _natural_fields(fields: Sequence[str]) -> str:
        labels = {
            "price_minor": "listed price",
            "in_stock": "stock status",
            "category": "category",
            "key_features": "listed features",
            "material": "material",
            "warranty": "warranty",
            "variant": "variant",
        }
        return " and ".join(labels[field] for field in fields)

    @staticmethod
    def _natural_fields_zh(fields: Sequence[str]) -> str:
        labels = {
            "price_minor": "标价",
            "in_stock": "库存状态",
            "category": "类别",
            "key_features": "已列出的特征",
            "material": "材质",
            "warranty": "保修信息",
            "variant": "变体",
        }
        return "、".join(labels[field] for field in fields)

    @staticmethod
    def _require_nonempty(values: Sequence[str], label: str) -> None:
        if not values:
            raise TaskBuildError(f"{label} unexpectedly has no accepted answer")

    def _validate_suite(self, tasks: Sequence[LargeCatalogTask]) -> None:
        if len(tasks) != 60 or len({task.task_id for task in tasks}) != 60:
            raise TaskBuildError("suite must contain exactly sixty unique tasks")
        if sum(task.role == "buyer" for task in tasks) != 45:
            raise TaskBuildError("suite must contain forty-five Buyer tasks")
        if sum(task.role == "merchant" for task in tasks) != 15:
            raise TaskBuildError("suite must contain fifteen Merchant tasks")
        if len({task.scenario for task in tasks}) != 20:
            raise TaskBuildError("suite must contain twenty scenario types")
        if len({task.capability for task in tasks}) != 18:
            raise TaskBuildError("suite must cover exactly eighteen existing capabilities")
        if {task.family for task in tasks} != {
            "Discovery",
            "Grounding",
            "Preference",
            "Multi-item",
        }:
            raise TaskBuildError("suite must use exactly four existing families")
        refs: set[str] = set()
        for task in tasks:
            refs.update(str(ref) for ref in task.public_context.get("listing_refs", ()))
            refs.update(
                str(line["listing_ref"])
                for line in task.public_context.get("quote_lines", ())
                if isinstance(line, Mapping) and "listing_ref" in line
            )
            refs.update(str(ref) for ref in task.oracle.get("accepted_refs", ()))
            for cart in task.oracle.get("accepted_carts", ()):
                refs.update(str(ref) for ref in cart)
        listings = [self.database.listing(ref) for ref in sorted(refs)]
        concrete = [row for row in listings if row is not None]
        merchant_count = len({row.merchant_id for row in concrete})
        category_count = len({row.category for row in concrete if row.category})
        if merchant_count < 50:
            raise TaskBuildError(
                f"suite touches only {merchant_count} merchants; at least 50 required"
            )
        if category_count < 30:
            raise TaskBuildError(
                f"suite touches only {category_count} categories; at least 30 required"
            )
        prices = [row.price_minor for row in concrete]
        bands = {
            "under_10": any(price < 1_000 for price in prices),
            "10_to_25": any(1_000 <= price < 2_500 for price in prices),
            "25_to_100": any(2_500 <= price < 10_000 for price in prices),
            "100_to_500": any(10_000 <= price < 50_000 for price in prices),
            "500_plus": any(price >= 50_000 for price in prices),
        }
        if not all(bands.values()):
            raise TaskBuildError(f"suite does not cover all five price bands: {bands}")


def build_tasks(database_path: str | Path) -> tuple[LargeCatalogTask, ...]:
    with CatalogDatabase(database_path) as database:
        return TaskBuilder(database).build()


def write_tasks(
    tasks: Iterable[LargeCatalogTask],
    destination: str | Path,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "suite": "ACWorld Large-Catalog Stress Suite",
                "tasks": [task.to_dict(include_oracle=True) for task in tasks],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_oracle_reports(
    tasks: Iterable[LargeCatalogTask],
    destination: str | Path,
) -> None:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        payload = {
            "task_id": task.task_id,
            "scenario": task.scenario,
            "family": task.family,
            "capability": task.capability,
            "oracle": task.oracle,
            "predicate_support": [
                {
                    "predicate": predicate.predicate_id,
                    "stage": predicate.stage,
                    "meaning": predicate.description,
                }
                for predicate in task.predicates
            ],
        }
        (root / f"{task.task_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_tasks(path: str | Path) -> tuple[LargeCatalogTask, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = []
    for row in raw["tasks"]:
        tasks.append(
            LargeCatalogTask(
                task_id=row["task_id"],
                scenario=row["scenario"],
                family=row["family"],
                capability=row["capability"],
                role=row["role"],
                prompt=row["prompt"],
                prompt_zh=row["prompt_zh"],
                public_context=row["public_context"],
                oracle=row["oracle"],
                allowed_intents=tuple(row["allowed_intents"]),
                predicates=tuple(TaskPredicate(**item) for item in row["predicates"]),
            )
        )
    return tuple(tasks)
