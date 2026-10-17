"""S39 stale match-certificate generator."""
from typing import Any

from .advanced_common import generate_variant

FAMILY_ID = "s39"
FAMILY_NAME = "match_certificate_stale"


def generate() -> list[tuple[str, dict[str, Any]]]:
    return generate_variant("S39")


__all__ = ["FAMILY_ID", "FAMILY_NAME", "generate"]
