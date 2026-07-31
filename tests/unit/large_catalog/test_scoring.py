from __future__ import annotations

from typing import Any

from large_catalog.models import LargeCatalogTask, TaskPredicate
from large_catalog.scoring import score_run


def _task(
    *,
    oracle: dict[str, Any],
    public_context: dict[str, Any],
    predicates: tuple[str, ...],
) -> LargeCatalogTask:
    return LargeCatalogTask(
        task_id="LC-TEST-01",
        scenario="TEST",
        family="Test",
        capability="test.capability",
        role="buyer",
        prompt="Test",
        prompt_zh="测试",
        public_context=public_context,
        oracle=oracle,
        allowed_intents=("search_catalog", "search_cart_candidates", "select_listing"),
        predicates=tuple(
            TaskPredicate(
                predicate_id=predicate,
                stage=(
                    "search"
                    if predicate.startswith("search_")
                    else "evidence"
                    if predicate in {
                        "decisive_records_observed",
                        "evidence_matches_decision",
                    }
                    else "decision"
                ),
                description=predicate,
            )
            for predicate in predicates
        ),
    )


def test_observation_reference_supports_selected_listing() -> None:
    listing_ref = "merchant:test:row:1"
    task = _task(
        oracle={"kind": "selection", "accepted_refs": [listing_ref]},
        public_context={"query": "item", "filters": {"in_stock": True}},
        predicates=("decisive_records_observed", "evidence_matches_decision"),
    )
    run = {
        "terminal": "completed",
        "final_intent": "select_listing",
        "final_arguments": {
            "listing_ref": listing_ref,
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "observe_listing",
                "arguments": {"listing_ref": listing_ref},
                "result": {
                    "result": {
                        "found": True,
                        "listing": {"listing_ref": listing_ref},
                    }
                },
            }
        ],
        "effects": [{"intent": "select_listing"}],
        "trace": [
            {
                "event_ref": "trace-001",
                "stage": "decision",
                "intent": "observe_listing",
            },
            {
                "event_ref": "trace-002",
                "stage": "evidence",
                "intent": "observe_listing",
            },
        ],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_rejection_reason_is_case_insensitive() -> None:
    task = _task(
        oracle={"kind": "selection", "accepted_refs": []},
        public_context={"query": "missing", "filters": {"in_stock": True}},
        predicates=("infeasibility_correct", "rejection_reason_correct"),
    )
    run = {
        "terminal": "completed",
        "final_intent": "reject_purchase",
        "final_arguments": {
            "reason_code": "NO_VALID_LISTING",
            "evidence_refs": [],
        },
        "observations": [],
        "effects": [{"intent": "reject_purchase"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_cart_search_ignores_nonoperational_requirement_labels() -> None:
    task = _task(
        oracle={"kind": "cart", "accepted_carts": []},
        public_context={
            "requirements": [
                {
                    "label": "toothbrush",
                    "query": "toothbrush",
                    "filters": {"in_stock": True},
                }
            ],
            "constraints": {"budget_minor": 500},
        },
        predicates=("search_preserves_constraints",),
    )
    run = {
        "terminal": "completed",
        "final_intent": "reject_purchase",
        "final_arguments": {"reason_code": "no_valid_cart", "evidence_refs": []},
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_cart_candidates",
                "arguments": {
                    "requirements": [
                        {
                            "query": "toothbrush",
                            "filters": {"in_stock": True},
                        }
                    ],
                    "constraints": {"budget_minor": 500},
                },
                "result": {"result": {"items": []}},
            }
        ],
        "effects": [{"intent": "reject_purchase"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_search_summary_is_sufficient_for_selection() -> None:
    listing_ref = "merchant:test:row:2"
    task = _task(
        oracle={"kind": "selection", "accepted_refs": [listing_ref]},
        public_context={"query": "retinol", "filters": {"in_stock": True}},
        predicates=("decisive_records_observed", "evidence_matches_decision"),
    )
    run = {
        "terminal": "completed",
        "final_intent": "select_listing",
        "final_arguments": {
            "listing_ref": listing_ref,
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_catalog",
                "arguments": {
                    "query": "retinol",
                    "filters": {"in_stock": True},
                    "sort": "price_asc",
                },
                "result": {
                    "result": {
                        "items": [
                            {
                                "listing_ref": listing_ref,
                                "name": "Retinol serum",
                                "price_minor": 499,
                                "in_stock": True,
                            }
                        ]
                    }
                },
            }
        ],
        "effects": [{"intent": "select_listing"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_search_summary_does_not_replace_full_record_for_answer() -> None:
    listing_ref = "merchant:test:row:3"
    task = _task(
        oracle={
            "kind": "answer",
            "listing_ref": listing_ref,
            "expected_fields": {"warranty": "24 months"},
        },
        public_context={
            "query": "retinol",
            "filters": {"in_stock": True},
            "listing_refs": [listing_ref],
            "requested_fields": ["warranty"],
        },
        predicates=("decisive_records_observed",),
    )
    run = {
        "terminal": "completed",
        "final_intent": "submit_product_answer",
        "final_arguments": {
            "listing_ref": listing_ref,
            "fields": {"warranty": "24 months"},
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_catalog",
                "arguments": {
                    "query": "retinol",
                    "filters": {"in_stock": True},
                    "sort": "price_asc",
                },
                "result": {
                    "result": {
                        "items": [
                            {
                                "listing_ref": listing_ref,
                                "name": "Retinol serum",
                                "price_minor": 499,
                                "in_stock": True,
                            }
                        ]
                    }
                },
            }
        ],
        "effects": [{"intent": "submit_product_answer"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert not result.strict_success
    assert result.score == 0.0


def test_cart_summary_is_sufficient_for_selected_lines() -> None:
    left = "merchant:test:row:4"
    right = "merchant:test:row:5"
    task = _task(
        oracle={"kind": "cart", "accepted_carts": [[left, right]]},
        public_context={
            "requirements": [
                {"query": "serum", "filters": {"in_stock": True}},
                {"query": "toner", "filters": {"in_stock": True}},
            ],
            "constraints": {"budget_minor": 2000},
        },
        predicates=("decisive_records_observed", "evidence_matches_decision"),
    )
    run = {
        "terminal": "completed",
        "final_intent": "submit_cart",
        "final_arguments": {
            "lines": [
                {"listing_ref": left, "quantity": 1},
                {"listing_ref": right, "quantity": 1},
            ],
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_cart_candidates",
                "arguments": {
                    "requirements": [
                        {"query": "serum", "filters": {"in_stock": True}},
                        {"query": "toner", "filters": {"in_stock": True}},
                    ],
                    "constraints": {"budget_minor": 2000},
                },
                "result": {
                    "result": {
                        "items": [
                            {
                                "lines": [
                                    {"listing_ref": left},
                                    {"listing_ref": right},
                                ]
                            }
                        ]
                    }
                },
            }
        ],
        "effects": [{"intent": "submit_cart"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_nonempty_preference_search_is_a_complete_selection_path() -> None:
    listing_ref = "merchant:preferred:row:1"
    task = _task(
        oracle={"kind": "selection", "accepted_refs": [listing_ref]},
        public_context={
            "query": "retinol",
            "filters": {"in_stock": True, "max_price_minor": 1500},
            "preference": {
                "kind": "category_then_price",
                "category": "Health & Beauty",
            },
        },
        predicates=(
            "search_preserves_constraints",
            "search_supports_global_judgment",
            "decisive_records_observed",
            "evidence_matches_decision",
        ),
    )
    run = {
        "terminal": "completed",
        "final_intent": "select_listing",
        "final_arguments": {
            "listing_ref": listing_ref,
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_catalog",
                "arguments": {
                    "query": "retinol",
                    "filters": {
                        "in_stock": True,
                        "max_price_minor": 1500,
                        "category": "Health & Beauty",
                    },
                    "sort": "price_asc",
                },
                "result": {
                    "result": {
                        "items": [
                            {
                                "listing_ref": listing_ref,
                                "price_minor": 499,
                            }
                        ]
                    }
                },
            }
        ],
        "effects": [{"intent": "select_listing"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_record_visibility_is_independent_of_search_correctness() -> None:
    listing_ref = "merchant:test:row:6"
    task = _task(
        oracle={"kind": "selection", "accepted_refs": [listing_ref]},
        public_context={"query": "retinol", "filters": {"in_stock": True}},
        predicates=("search_preserves_constraints", "decisive_records_observed"),
    )
    run = {
        "terminal": "completed",
        "final_intent": "select_listing",
        "final_arguments": {
            "listing_ref": listing_ref,
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_catalog",
                "arguments": {
                    "query": "different query",
                    "filters": {"in_stock": True},
                    "sort": "price_asc",
                },
                "result": {
                    "result": {
                        "items": [{"listing_ref": listing_ref, "price_minor": 499}]
                    }
                },
            }
        ],
        "effects": [{"intent": "select_listing"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")
    earned = {row.predicate: row.points for row in result.process_rewards}

    assert not result.strict_success
    assert result.score == 0.5
    assert earned == {
        "search_preserves_constraints": 0,
        "decisive_records_observed": 1,
    }


def test_punctuation_only_field_edits_remain_grounded() -> None:
    listing_ref = "merchant:test:row:7"
    task = _task(
        oracle={
            "kind": "answer",
            "listing_ref": listing_ref,
            "expected_fields": {
                "category": "Robes / Dresses",
                "key_features": "Satin finish, side-seam pockets.",
            },
        },
        public_context={
            "listing_refs": [listing_ref],
            "requested_fields": ["category", "key_features"],
        },
        predicates=("structured_content_correct", "decision_matches_oracle"),
    )
    run = {
        "terminal": "completed",
        "final_intent": "submit_product_answer",
        "final_arguments": {
            "listing_ref": listing_ref,
            "fields": {
                "category": "robes dresses",
                "key_features": "SATIN finish; side seam pockets",
            },
            "evidence_refs": ["observation-001"],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "observe_listing",
                "arguments": {"listing_ref": listing_ref},
                "result": {
                    "result": {
                        "found": True,
                        "listing": {
                            "listing_ref": listing_ref,
                            "category": "Robes / Dresses",
                            "key_features": "Satin finish, side-seam pockets.",
                        },
                    }
                },
            }
        ],
        "effects": [{"intent": "submit_product_answer"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def _missing_field_answer(
    fields: dict[str, Any],
) -> tuple[LargeCatalogTask, dict[str, Any]]:
    listing_ref = "merchant:test:row:missing"
    task = _task(
        oracle={
            "kind": "answer",
            "listing_ref": listing_ref,
            "expected_fields": {"material": None, "warranty": None},
        },
        public_context={
            "listing_refs": [listing_ref],
            "requested_fields": ["material", "warranty"],
        },
        predicates=(
            "answer_targets_requested_listing",
            "answer_field_correct.material",
            "answer_field_correct.warranty",
        ),
    )
    run = {
        "terminal": "completed",
        "final_intent": "submit_product_answer",
        "final_arguments": {
            "listing_ref": listing_ref,
            "fields": fields,
            "evidence_refs": ["observation-001"],
        },
        "observations": [],
        "effects": [{"intent": "submit_product_answer"}],
        "trace": [],
    }
    return task, run


def test_missing_field_markers_match_null_oracle_values() -> None:
    task, run = _missing_field_answer(
        {
            "material": "Not provided in the current listing record.",
            "warranty": "",
        }
    )

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_field_errors_receive_field_level_credit() -> None:
    task, run = _missing_field_answer(
        {
            "material": "80% polyester, 20% cotton",
            "warranty": "Not specified",
        }
    )

    result = score_run(task, run, model_id="test")
    earned = {row.predicate: row.points for row in result.process_rewards}

    assert not result.strict_success
    assert result.score == 2 / 3
    assert earned == {
        "answer_targets_requested_listing": 1,
        "answer_field_correct.material": 0,
        "answer_field_correct.warranty": 1,
    }


def test_incomplete_wrong_answer_gets_less_credit() -> None:
    task, run = _missing_field_answer(
        {"material": "80% polyester, 20% cotton"}
    )

    result = score_run(task, run, model_id="test")
    earned = {row.predicate: row.points for row in result.process_rewards}

    assert not result.strict_success
    assert result.score == 1 / 3
    assert earned == {
        "answer_targets_requested_listing": 1,
        "answer_field_correct.material": 0,
        "answer_field_correct.warranty": 0,
    }


def test_explanatory_material_claim_is_not_a_missing_value_marker() -> None:
    task, run = _missing_field_answer(
        {
            "material": (
                "The material field is not explicitly filled in, but the key "
                "features state it is made from recycled ocean plastic."
            ),
            "warranty": "Warranty detail is not provided in the listing record.",
        }
    )

    result = score_run(task, run, model_id="test")
    earned = {row.predicate: row.points for row in result.process_rewards}

    assert earned == {
        "answer_targets_requested_listing": 1,
        "answer_field_correct.material": 0,
        "answer_field_correct.warranty": 1,
    }


def _claim_task_and_run(claim: dict[str, Any]) -> tuple[LargeCatalogTask, dict[str, Any]]:
    lower_ref = "merchant:test:row:8"
    higher_ref = "merchant:test:row:9"
    task = _task(
        oracle={
            "kind": "claim",
            "expected_action": "publish_comparison_claim",
            "claim": {"type": "lower_price", "subject_ref": lower_ref},
            "comparison": {
                "listing_refs": [lower_ref, higher_ref],
                "lower_price_ref": lower_ref,
                "price_difference_minor": 501,
                "both_in_stock": True,
            },
        },
        public_context={
            "listing_refs": [lower_ref, higher_ref],
            "requested_claim": {"type": "lower_price", "subject_ref": lower_ref},
        },
        predicates=(
            "structured_content_correct",
            "unsupported_content_omitted",
            "decision_matches_oracle",
        ),
    )
    observations = [
        {
            "observation_ref": f"observation-{index:03d}",
            "intent": "observe_listing",
            "arguments": {"listing_ref": listing_ref},
            "result": {
                "result": {
                    "found": True,
                    "listing": {
                        "listing_ref": listing_ref,
                        "price_minor": price_minor,
                        "currency": "USD",
                    },
                }
            },
        }
        for index, (listing_ref, price_minor) in enumerate(
            ((lower_ref, 499), (higher_ref, 1000)),
            start=1,
        )
    ]
    run = {
        "terminal": "completed",
        "final_intent": "publish_comparison_claim",
        "final_arguments": {
            "listing_refs": [lower_ref, higher_ref],
            "claim": claim,
            "evidence_refs": ["observation-001", "observation-002"],
        },
        "observations": observations,
        "effects": [{"intent": "publish_comparison_claim"}],
        "trace": [],
    }
    return task, run


def test_verified_claim_details_are_not_treated_as_unsupported() -> None:
    task, run = _claim_task_and_run(
        {
            "type": "lower_price",
            "subject_ref": "merchant:test:row:8",
            "object_ref": "merchant:test:row:9",
            "subject_price_minor": 499,
            "comparison_price_minor": 1000,
            "currency": "USD",
        }
    )

    result = score_run(task, run, model_id="test")

    assert result.strict_success
    assert result.score == 1.0


def test_unverified_claim_details_remain_unsupported() -> None:
    task, run = _claim_task_and_run(
        {
            "type": "lower_price",
            "subject_ref": "merchant:test:row:8",
            "comparison_price_minor": 999,
        }
    )

    result = score_run(task, run, model_id="test")

    assert not result.strict_success
    assert result.score == 0.0


def test_cart_constraint_alias_is_not_silently_accepted() -> None:
    task = _task(
        oracle={"kind": "cart", "accepted_carts": []},
        public_context={
            "requirements": [
                {"query": "serum", "filters": {"in_stock": True}},
            ],
            "constraints": {"budget_minor": 500},
        },
        predicates=("search_preserves_constraints",),
    )
    run = {
        "terminal": "completed",
        "final_intent": "reject_purchase",
        "final_arguments": {
            "reason_code": "no_valid_cart",
            "evidence_refs": [],
        },
        "observations": [
            {
                "observation_ref": "observation-001",
                "intent": "search_cart_candidates",
                "arguments": {
                    "requirements": [
                        {"query": "serum", "filters": {"in_stock": True}},
                    ],
                    "constraints": {"budget": 500},
                },
                "result": {"result": {"items": []}},
            }
        ],
        "effects": [{"intent": "reject_purchase"}],
        "trace": [],
    }

    result = score_run(task, run, model_id="test")

    assert not result.strict_success
    assert result.score == 0.0
