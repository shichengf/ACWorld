"""``RemoteWorldTools`` — the world view for an agent running in its own process.

In R1 re-entrant mode an agent never reads World in-process: every grounding read
becomes a ``world.read_*`` envelope routed through the dispatcher. This stand-in
satisfies :class:`~agents.types.AgentContext`'s ``world`` slot while holding no
World — and turns any accidental in-process read into a loud error instead of a
silent use of a World that isn't there (the agent process genuinely has none).
"""

from __future__ import annotations

from typing import Any

from world.tools import WorldTools


class _RefusingWorld:
    """Stands in for ``World`` in a process that has none; any access raises."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            "R1 remote agent has no in-process World — grounding reads must go "
            "over the bus as world.read_* envelopes, not via WorldTools "
            f"(attempted: World.{name})"
        )


class RemoteWorldTools(WorldTools):
    """A :class:`WorldTools` for a split-process agent: type-compatible, no World.

    Re-entrant turns suspend on ``tool_call`` before ever touching ``ctx.world``,
    so these methods are never called on the happy path; if one is (a bug where
    re-entrant mode wasn't propagated), it raises clearly rather than reading a
    World that does not exist in this process.
    """

    def __init__(self, *, caller_id: str) -> None:
        # The refusing sentinel is intentionally not a World; the inherited read
        # methods (``self._world.read(...)``) therefore raise on any call.
        super().__init__(_RefusingWorld(), caller_id=caller_id)  # type: ignore[arg-type]
