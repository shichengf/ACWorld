"""The single provider-facing system prompt for typed business decisions."""

from __future__ import annotations

from pathlib import Path


_PROMPT_PATH = Path(__file__).parent / "prompts" / "business.system.md"


def _load_business_decision_system_prompt_v1() -> str:
    prompt = _PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError("business decision system prompt is empty")
    return prompt


BUSINESS_DECISION_SYSTEM_PROMPT_V1 = _load_business_decision_system_prompt_v1()


__all__ = ["BUSINESS_DECISION_SYSTEM_PROMPT_V1"]
