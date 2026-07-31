"""S32 false-discount-anchor generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s32"
FAMILY_NAME = "false_discount_anchor"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S32")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
