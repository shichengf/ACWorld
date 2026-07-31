from __future__ import annotations

import pytest

from large_catalog.runtime import (
    LargeCatalogRuntimeError,
    _canonicalize_arguments,
    _intent_schema,
    _parse_decision_object,
)


DECISION = '{"intent":"search_catalog","arguments":{"query":"retinol"}}'


@pytest.mark.parametrize(
    "content",
    (
        DECISION,
        f"```json\n{DECISION}\n```",
        f"I will search first.\n{DECISION}",
        f"{DECISION}\nThis is the next action.",
        (
            '{"intent":"search_catalog","arguments":{"query":"retinol"},'
            '"confidence":0.8}'
        ),
    ),
)
def test_parser_accepts_one_json_decision_with_harmless_wrapping(content: str) -> None:
    parsed = _parse_decision_object(content)
    assert parsed["intent"] == "search_catalog"
    assert parsed["arguments"] == {"query": "retinol"}


@pytest.mark.parametrize(
    "content",
    (
        "{'intent':'search_catalog','arguments':{}}",
        f"{DECISION}\n{DECISION}",
        "I cannot decide.",
    ),
)
def test_parser_rejects_ambiguous_or_non_json_output(content: str) -> None:
    with pytest.raises(LargeCatalogRuntimeError):
        _parse_decision_object(content)


def test_quote_schema_discloses_each_line_field() -> None:
    assert _intent_schema("submit_quote")["lines"] == (
        "list[{listing_ref: string, quantity: integer, "
        "unit_price_minor: integer, line_total_minor: integer}]"
    )


def test_quote_line_amount_alias_is_canonicalized() -> None:
    arguments = _canonicalize_arguments(
        "submit_quote",
        {
            "lines": [
                {
                    "listing_ref": "listing:1",
                    "quantity": 2,
                    "unit_price_minor": 125,
                    "line_amount_minor": 250,
                }
            ]
        },
    )
    assert arguments["lines"] == [
        {
            "listing_ref": "listing:1",
            "quantity": 2,
            "unit_price_minor": 125,
            "line_total_minor": 250,
        }
    ]


def test_quote_line_conflicting_alias_is_rejected() -> None:
    with pytest.raises(
        LargeCatalogRuntimeError,
        match="conflicting total and amount",
    ):
        _canonicalize_arguments(
            "submit_quote",
            {
                "lines": [
                    {
                        "listing_ref": "listing:1",
                        "quantity": 2,
                        "unit_price_minor": 125,
                        "line_total_minor": 250,
                        "line_amount_minor": 251,
                    }
                ]
            },
        )
