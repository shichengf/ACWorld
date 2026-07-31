"""``SkillLoader`` implementations — empty + filesystem-scanning.

Per AGENT_CLASSES.md §5, a loader scans configured roots, exposes only
manifests at spawn, and loads full ``SkillDocument`` bodies on demand. The
loader also computes a stable digest over frontmatter + body + referenced
files / scripts so replay can distinguish ``skill@A`` from ``skill@B``.

This module ships two loaders:

* :class:`EmptySkillLoader` — exposes no manifests; trivial implementation
  used by factories that don't yet have a SKILL.md directory wired up.
* :class:`FilesystemSkillLoader` — the default; reads SKILL.md bundles from
  one or more roots on disk.

A future cross-process loader (HTTP / stdio JSON-RPC per AGENT_DESIGN.md §4)
implements the same :class:`agents.interfaces.SkillLoader` Protocol.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agents.errors import FrontmatterError, SkillNotFound
from agents.types import SkillDocument, SkillGroup, SkillManifest

if TYPE_CHECKING:
    pass


# --- Frontmatter helpers ---------------------------------------------

_FRONTMATTER_DELIM = "---"


def _split_frontmatter(text: str) -> "tuple[str, str]":
    """Split ``text`` into ``(frontmatter_yaml, body)``.

    A SKILL.md file with no frontmatter delimiter yields ``("", body)``;
    callers decide whether that's an error.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return "", text
    closing = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            closing = i
            break
    if closing is None:
        return "", text
    frontmatter = "".join(lines[1:closing])
    body = "".join(lines[closing + 1 :])
    return frontmatter, body


def _parse_yaml(text: str) -> "dict[str, Any]":
    """Parse a small YAML subset used in SKILL.md frontmatter.

    Falls back to ``yaml.safe_load`` if PyYAML is installed; otherwise uses
    a hand-rolled parser that supports the fields actually used by
    SKILL.md (string scalars, booleans, inline ``[a, b, c]`` lists, and
    block lists — a bare ``key:`` followed by indented ``- item`` lines,
    which is the shape ``allowed_tools`` uses in every bundle).

    The hand-rolled fallback keeps the package import-time cost zero on
    systems without PyYAML.
    """
    try:
        import yaml  # type: ignore[import-not-found]

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise FrontmatterError(f"frontmatter must be a mapping, got {type(loaded).__name__}")
        return cast("dict[str, Any]", loaded)
    except ImportError:
        return _parse_yaml_fallback(text)


def _parse_yaml_fallback(text: str) -> "dict[str, Any]":
    """Minimal YAML parser for the SKILL.md frontmatter subset.

    Supports ``key: value`` scalars, inline ``[a, b, c]`` lists, and block
    lists (a bare ``key:`` followed by indented ``- item`` lines). A bare
    ``key:`` with no following list items reads as ``null``.
    """
    out: dict[str, Any] = {}
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].split("#", 1)[0].rstrip()
        i += 1
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            raise FrontmatterError(f"frontmatter list item has no parent key — {line!r}")
        if ":" not in line:
            raise FrontmatterError(f"frontmatter line missing ':' — {line!r}")
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value != "":
            out[key] = _coerce_yaml_value(value)
            continue
        # Empty value: gather an optional block list of "- item" lines.
        items: list[Any] = []
        while i < n:
            peek = lines[i].split("#", 1)[0].rstrip()
            if not peek.strip():
                i += 1
                continue
            if not peek.lstrip().startswith("- "):
                break
            items.append(_coerce_yaml_value(peek.lstrip()[2:].strip()))
            i += 1
        out[key] = items if items else None
    return out


def _coerce_yaml_value(text: str) -> Any:
    """Coerce a trimmed YAML scalar / list into Python."""
    if text == "" or text.lower() in {"null", "~"}:
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_coerce_yaml_value(item.strip()) for item in inner.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


# --- Digest ----------------------------------------------------------

def _digest_bytes(*parts: bytes) -> str:
    """Stable hex digest over the concatenated parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()


# --- Empty loader ----------------------------------------------------

class EmptySkillLoader:
    """A ``SkillLoader`` that exposes no manifests and rejects every load.

    Used by factories that don't yet have a SKILL.md directory wired up
    (early integration demos, deterministic baselines that don't load
    skills). Satisfies the ``SkillLoader`` Protocol so ``Agent`` construction
    never needs to special-case the "no skills" path.
    """

    def manifests(self) -> "tuple[SkillManifest, ...]":
        """Return an empty manifest tuple."""
        return ()

    def load(self, name: str) -> SkillDocument:
        """Always raise — there are no skills to load.

        Raises:
            SkillNotFound: every call.
        """
        raise SkillNotFound(name)


# --- Filesystem loader -----------------------------------------------

class FilesystemSkillLoader:
    """Default ``SkillLoader``: scans skill directory roots on disk.

    Roots are tried in precedence order (AGENT_CLASSES.md §5):

        1. enterprise-managed
        2. personal      (``~/.cwe/skills/<name>/SKILL.md``)
        3. project       (``.cwe/skills/<name>/SKILL.md``)
        4. plugin        (always namespaced as ``plugin:skill-name``)

    First match per ``name`` wins. ``digest`` is a stable SHA-256 over the
    normalized frontmatter, body, references content, and scripts content.
    Replay records the digest so ``skill@A`` and ``skill@B`` are
    distinguishable in reports.
    """

    _by_name: "dict[str, SkillManifest]"
    _cache: "dict[str, SkillDocument]"
    _roots: "tuple[Path, ...]"

    def __init__(
        self, *, roots: "tuple[Path, ...]", enabled_names: "tuple[str, ...]"
    ) -> None:
        """Scan ``roots`` and index manifests for ``enabled_names`` only.

        Raises:
            SkillNotFound: an enabled name does not resolve to any SKILL.md.
            FrontmatterError: a SKILL.md's frontmatter fails validation.
        """
        self._roots = tuple(Path(r) for r in roots)
        self._cache = {}
        self._by_name = {}

        for name in enabled_names:
            skill_dir = self._resolve(name)
            if skill_dir is None:
                raise SkillNotFound(f"{name!r} not found under {self._roots}")
            self._by_name[name] = self._build_manifest(name, skill_dir)

    # ---- public API ------------------------------------------------

    def manifests(self) -> "tuple[SkillManifest, ...]":
        """Return every enabled manifest in registration (i.e. enabled_names) order."""
        return tuple(self._by_name.values())

    def load(self, name: str) -> SkillDocument:
        """Return the full ``SkillDocument`` for ``name``.

        Result is memoized so repeated loads within a turn are cheap.

        Raises:
            SkillNotFound: ``name`` is not in the enabled manifest set.
        """
        if name not in self._by_name:
            raise SkillNotFound(name)
        if name in self._cache:
            return self._cache[name]

        manifest = self._by_name[name]
        skill_dir = Path(manifest.path).parent
        body_path = Path(manifest.path)
        body = body_path.read_text(encoding="utf-8")
        _, instructions = _split_frontmatter(body)

        references = self._collect_files(skill_dir / "references")
        scripts = self._collect_files(skill_dir / "scripts")

        doc = SkillDocument(
            manifest=manifest,
            instructions=instructions,
            references=references,
            scripts=scripts,
        )
        self._cache[name] = doc
        return doc

    # ---- helpers ---------------------------------------------------

    def _resolve(self, name: str) -> "Path | None":
        """Locate the skill directory containing ``<root>/<name>/SKILL.md``.

        Returns the directory path on first hit; ``None`` if no root has it.
        """
        for root in self._roots:
            candidate = root / name / "SKILL.md"
            if candidate.is_file():
                return candidate.parent
        return None

    def _build_manifest(self, name: str, skill_dir: Path) -> SkillManifest:
        """Parse frontmatter from ``<skill_dir>/SKILL.md`` and compute digest."""
        skill_md = skill_dir / "SKILL.md"
        raw = skill_md.read_text(encoding="utf-8")
        fm_text, body = _split_frontmatter(raw)
        if not fm_text.strip():
            raise FrontmatterError(f"{skill_md} has no YAML frontmatter")

        fm = _parse_yaml(fm_text)
        if "name" not in fm or "description" not in fm:
            raise FrontmatterError(
                f"{skill_md} frontmatter must declare 'name' and 'description'"
            )
        fm_name = str(fm["name"])
        if fm_name != name:
            raise FrontmatterError(
                f"{skill_md} frontmatter name {fm_name!r} ≠ directory name {name!r}"
            )

        group: "SkillGroup | None" = None
        raw_group = fm.get("group")
        if raw_group is not None:
            try:
                group = SkillGroup(str(raw_group))
            except ValueError as exc:
                raise FrontmatterError(
                    f"{skill_md} frontmatter group {raw_group!r} is not a valid SkillGroup"
                ) from exc

        # Digest covers the full file text + bundled references / scripts so
        # any byte change to the bundle changes the digest.
        ref_bytes = self._collect_bytes(skill_dir / "references")
        script_bytes = self._collect_bytes(skill_dir / "scripts")
        digest = _digest_bytes(raw.encode("utf-8"), ref_bytes, script_bytes)

        return SkillManifest(
            name=fm_name,
            description=str(fm["description"]),
            when_to_use=(None if "when_to_use" not in fm else str(fm["when_to_use"])),
            group=group,
            path=str(skill_md),
            digest=digest,
        )

    @staticmethod
    def _collect_files(directory: Path) -> "tuple[str, ...]":
        """Return relative file paths under ``directory`` in sorted order.

        Used by ``load()`` to surface bundled references / scripts to the
        agent as paths (it can then read them through a tool call). Returns
        empty tuple if the directory does not exist.
        """
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                str(p.relative_to(directory.parent))
                for p in directory.rglob("*")
                if p.is_file()
            )
        )

    @staticmethod
    def _collect_bytes(directory: Path) -> bytes:
        """Concatenate every file under ``directory`` in sorted order, for digest."""
        if not directory.is_dir():
            return b""
        chunks: list[bytes] = []
        for p in sorted(directory.rglob("*")):
            if p.is_file():
                chunks.append(p.read_bytes())
        return b"".join(chunks)
