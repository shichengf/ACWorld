"""Regenerate every generator-backed scenario YAML file.

Usage (from repo root):

    python -m scenarios.generators.generate
    # or
    python scenarios/generators/generate.py

Writes ``scenarios/sN_<family>_<seed>.yaml`` deterministically. Re-running
produces byte-identical files — this is the corpus replay contract. Hand-authored
research variants may coexist in ``scenarios/``; this driver only owns files
returned by ``FAMILIES``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow running as a bare script: add repo root so `scenarios` imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scenarios.generators import FAMILIES
    from scenarios.generators.common import write_scenario
else:
    from . import FAMILIES
    from .common import write_scenario

OUT_DIR = Path(__file__).resolve().parents[1]  # -> scenarios/


def main() -> int:
    written: list[str] = []
    for fam in FAMILIES:
        for filename, spec in fam.generate():
            path = write_scenario(OUT_DIR, filename, spec)
            written.append(path.name)
    written.sort()
    print(f"wrote {len(written)} scenario files to {OUT_DIR}:")
    for name in written:
        print(f"  {name}")
    expected = sum(len(family.generate()) for family in FAMILIES)
    if len(written) != expected:
        print(f"ERROR: expected {expected} generated files, got {len(written)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
