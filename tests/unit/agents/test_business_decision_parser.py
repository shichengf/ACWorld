from __future__ import annotations

import pytest

from agents.business_decision import (
    BusinessDecisionContractError,
    BusinessIntentSpec,
    LLMBusinessDecisionV1,
)


SPEC = BusinessIntentSpec(
    intent="search",
    description="Perform the search business choice.",
    parameters={
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    },
    category="act",
    source_name="search",
)
DECISION = (
    '{"schema_version":"cwe.llm-business-decision.v1","intent":"search",'
    '"arguments":{"query":"lamp"}}'
)


@pytest.mark.parametrize(
    "content",
    (
        DECISION,
        f"```json\n{DECISION}\n```",
        f"```\n{DECISION}\n```",
        f"Here is my decision:\n{DECISION}",
        f"{DECISION}\nI will refine this next turn.",
        f"Let me search.\n{DECISION}\nThat covers the request.",
        f"```json\n{DECISION}\n```\nHope that helps.",
        f"\n\n   {DECISION}\n",
        (
            '{"schema_version":"cwe.llm-business-decision.v1","intent":"search",'
            '"arguments":{"query":"lamp"},"reasoning":"cheapest first"}'
        ),
        '{"intent":"search","arguments":{"query":"lamp"}}',
    ),
)
def test_parser_accepts_one_decision_with_harmless_presentation(content: str) -> None:
    decision = LLMBusinessDecisionV1.parse(content, allowed_intents=[SPEC])
    assert decision.intent == "search"
    assert decision.arguments == {"query": "lamp"}


@pytest.mark.parametrize(
    "content",
    (
        f"{DECISION}\n{DECISION}",
        "I cannot decide.",
        "{'intent':'search','arguments':{'query':'lamp'}}",
        '{"intent":"search","intent":"search","arguments":{"query":"lamp"}}',
        '{"intent":"search","arguments":{"query":NaN}}',
        '{"intent":"search","arguments":"lamp"}',
    ),
)
def test_parser_rejects_ambiguous_or_non_json_output(content: str) -> None:
    with pytest.raises(BusinessDecisionContractError):
        LLMBusinessDecisionV1.parse(content, allowed_intents=[SPEC])


@pytest.mark.parametrize(
    "content",
    (
        '{"intent":"settle","arguments":{"query":"lamp"}}',
        '{"intent":"search","arguments":{"query":"lamp","budget_cents":10}}',
        '{"intent":"search","arguments":{}}',
        '{"intent":"search","arguments":{"query":7}}',
        '{"schema_version":"cwe.llm-business-decision.v1","arguments":{"query":"lamp"}}',
    ),
)
def test_widened_presentation_does_not_relax_business_rules(content: str) -> None:
    """Presentation is tolerated; intent authority and arguments stay strict."""

    with pytest.raises(BusinessDecisionContractError):
        LLMBusinessDecisionV1.parse(content, allowed_intents=[SPEC])


def test_narration_key_is_dropped_before_the_provider_surface_check() -> None:
    """An ignored narration field never reaches the Agent as business content."""

    decision = LLMBusinessDecisionV1.parse(
        '{"intent":"search","arguments":{"query":"lamp"},'
        '"notes":"I will check platform:aggregator next."}',
        allowed_intents=[SPEC],
    )
    assert decision.to_dict() == {
        "schema_version": "cwe.llm-business-decision.v1",
        "intent": "search",
        "arguments": {"query": "lamp"},
    }
