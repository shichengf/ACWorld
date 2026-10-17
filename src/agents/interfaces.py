"""Cross-package contracts the agents layer exposes.

Two of the AGENT_CLASSES.md primitives live here as Protocols:
    * ``SkillLoader`` — exposes manifests at spawn; loads full ``SkillDocument``
      bundles on demand per AGENT_CLASSES.md §4.1 (progressive disclosure).
    * ``Memory``      — the typed read/write store with the private-utility
      invariant per AGENT_CLASSES.md §4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agents.types import MemoryType, SkillDocument, SkillManifest


class SkillLoader(Protocol):
    """Progressive-disclosure loader for SKILL.md bundles.

    At spawn, the agent sees only manifests (``name`` / ``description`` /
    ``when_to_use`` / selected frontmatter / ``digest``). When a turn matches
    a manifest, ``load(name)`` returns the full ``SkillDocument``; the
    document enters context only for that turn.
    """

    def manifests(self) -> "tuple[SkillManifest, ...]":
        """Return all enabled manifests in registration order."""
        ...

    def load(self, name: str) -> "SkillDocument":
        """Return the full ``SkillDocument`` for ``name``.

        Raises:
            SkillNotFound: ``name`` is not in the manifest set.
        """
        ...


class Memory(Protocol):
    """The agent's typed read/write store.

    Non-negotiable invariant (AGENT_CLASSES.md §4): values written under
    ``MemoryType.PRIVATE_UTILITY`` must never appear in an outbound envelope's
    payload. The runtime checks every send.
    """

    def read(self, type: "MemoryType", key: str) -> Any:
        """Return the value at ``(type, key)`` or ``None`` if absent."""
        ...

    def write(self, type: "MemoryType", key: str, value: Any) -> None:
        """Store ``value`` at ``(type, key)``."""
        ...

    def delete(self, type: "MemoryType", key: str) -> None:
        """Remove ``(type, key)`` when a provisional framework write rolls back."""
        ...

    def keys(self, type: "MemoryType") -> "list[str]":
        """Return keys present under ``type``."""
        ...
