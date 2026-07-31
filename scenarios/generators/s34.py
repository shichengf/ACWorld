"""S34 collusive-merchant resistance generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s34"
FAMILY_NAME = "collusive_merchants"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S34")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
