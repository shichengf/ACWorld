"""S40 buyer-message prompt-injection generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s40"
FAMILY_NAME = "buyer_message_prompt_injection"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S40")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
