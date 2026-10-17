"""S33 fake-review resistance generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s33"
FAMILY_NAME = "fake_reviews"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S33")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
