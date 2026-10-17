"""S38 cross-rail payment replay generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s38"
FAMILY_NAME = "payment_replay_cross_rail"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S38")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
