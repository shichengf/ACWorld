"""S37 paired adjudicator-bias platform diagnostic generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s37"
FAMILY_NAME = "adjudicator_bias"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S37")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
