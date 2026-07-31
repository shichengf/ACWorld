"""Activation predicate for skills/merchant/private-utility-guard/SKILL.md.

Always-on structural pre-emit check. Returns ``True`` unconditionally
(except for the no-op case of no merchant_data attached, which means
the guard has nothing to check against). The selector places this last
in the activated tuple so it runs as the final pre-emit step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.skill_selector import _SelectorContext


def gate(ctx: "_SelectorContext") -> tuple[bool, str]:
    return True, "structural pre-emit guard always-on"
