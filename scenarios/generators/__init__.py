"""Scenario generators — one module per family. The corpus design is
documented in ``scenarios/README.md``; the scenario shape is the
``ScenarioSpec`` dataclass in ``src/episode/types.py``.

Each implemented ``sN`` module exposes ``FAMILY_ID``, ``FAMILY_NAME`` and a
``generate() -> list[(filename, spec_dict)]`` that is a pure deterministic
function of ``common.SEEDS``. ``generate.py`` is the deterministic driver.
"""

from __future__ import annotations

from . import (
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s18,
    s19,
    s20,
    s21,
    s22,
    s23,
    s24,
    s25,
    s26,
    s27,
    s28,
    s29,
    s30,
    s31,
    s32,
    s33,
    s34,
    s35,
    s36,
    s37,
    s38,
    s39,
    s40,
)

FAMILIES = [
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s18,
    s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31,
    s32, s33, s34, s35, s36, s37, s38, s39, s40,
]

__all__ = [
    "FAMILIES", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
    "s10", "s11", "s12", "s13", "s14", "s18", "s19", "s20", "s21",
    "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30",
    "s31", "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39",
    "s40",
]
