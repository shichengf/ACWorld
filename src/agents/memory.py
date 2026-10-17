"""Default in-process ``Memory`` implementation.

Holds the typed read/write store for one agent. The runtime's private-utility
guard reads from this store on every send (see ``runtime.router.Router``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.types import MemoryType

if TYPE_CHECKING:
    pass


class InMemoryStore:
    """A simple per-agent typed store. Default implementation of the ``Memory`` Protocol.

    Storage shape: ``dict[MemoryType, dict[str, Any]]``. One bucket per
    ``MemoryType``; reads return ``None`` for missing keys (per the Protocol
    contract in ``agents.interfaces.Memory``).
    """

    _store: "dict[MemoryType, dict[str, Any]]"

    def __init__(self) -> None:
        """Initialize an empty store with one bucket per ``MemoryType``."""
        self._store = {t: {} for t in MemoryType}

    def read(self, type: MemoryType, key: str) -> Any:
        """Return the value at ``(type, key)`` or ``None`` if absent."""
        return self._store[type].get(key)

    def write(self, type: MemoryType, key: str, value: Any) -> None:
        """Store ``value`` at ``(type, key)``."""
        self._store[type][key] = value

    def delete(self, type: MemoryType, key: str) -> None:
        """Remove a value while keeping deletion idempotent."""

        self._store[type].pop(key, None)

    def keys(self, type: MemoryType) -> "list[str]":
        """Return keys present under ``type`` in insertion order."""
        return list(self._store[type].keys())
