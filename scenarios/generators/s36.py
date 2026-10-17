"""S36 evidence-backed dispute generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s36"
FAMILY_NAME = "dispute_with_evidence"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S36")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
