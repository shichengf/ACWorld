"""Episode-owned artifact lifecycle.

One shared list + one prepare step used by BOTH launch paths (in-process
``EpisodeBatch`` and ``run_http_episode``). ``AuditLog`` opens its streams in
APPEND mode, so without removing prior generated files a re-run into the same
directory would duplicate audit envelopes or leak previous security/evidence
artifacts into the new run.

Only KNOWN, episode-owned generated files are removed — never the directory
itself, and never unknown user files. Preparation MUST happen before any
``AuditLog`` opens its append-mode streams.
"""

from __future__ import annotations

from pathlib import Path

#: Generated artifacts an episode OWNS in its out_dir. Removed on prepare so a
#: re-run starts clean. Keep this the single source of truth for both paths.
OWNED_ARTIFACTS: tuple[str, ...] = (
    "audit.jsonl",
    "audit.trace.jsonl",
    "audit.security.jsonl",
    # Purge artifacts produced by pre-v2.1 runners; current Episode execution
    # never recreates them.
    "score.json",
    "reward.json",
    "world.initial.json",
    "world.final.json",
    "world.commits.jsonl",
    "platform.decisions.jsonl",
    "platform.response-dispositions.jsonl",
    "actor.contexts.json",
    "actor.evidence.jsonl",
    "txn_diffs.jsonl",
    "replay.json",
    "episode.evidence.json",
    "extensions.json",
    "termination.json",
)


def prepare_out_dir(out_dir: "str | Path") -> Path:
    """Create ``out_dir`` and remove only the episode-owned generated artifacts.

    Returns the resolved ``Path``. Call this BEFORE constructing the ``AuditLog``
    (which opens append-mode handles). Never deletes the directory or unknown
    files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in OWNED_ARTIFACTS:
        (out / name).unlink(missing_ok=True)
    return out
