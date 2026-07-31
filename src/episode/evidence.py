"""Finalize one real CommerceWorld episode into replay-verifiable evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from evals.serialize import to_canonical
from protocol.envelope import from_json, to_json
from runtime.actor_context_artifact import (
    ACTOR_CONTEXTS_ARTIFACT,
    load_actor_contexts,
)
from runtime.actor_evidence import ACTOR_EVIDENCE_ARTIFACT, load_actor_evidence
from runtime.actor_evidence_verifier import verify_actor_evidence_causality
from runtime.evidence import (
    PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT,
    load_platform_decisions,
    load_platform_response_dispositions,
)


EPISODE_EVIDENCE_SCHEMA = "cwe.episode-evidence.v1"
EPISODE_EVIDENCE_ARTIFACT = "episode.evidence.json"
WORLD_COMMITS_ARTIFACT = "world.commits.jsonl"
PLATFORM_DECISIONS_ARTIFACT = "platform.decisions.jsonl"
REPLAY_ARTIFACT = "replay.json"

_REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("audit", "audit.jsonl"),
    ("platform_decisions", PLATFORM_DECISIONS_ARTIFACT),
    ("platform_response_dispositions", PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT),
    ("actor_contexts", ACTOR_CONTEXTS_ARTIFACT),
    ("actor_evidence", ACTOR_EVIDENCE_ARTIFACT),
    ("world_commits", WORLD_COMMITS_ARTIFACT),
    ("world_initial", "world.initial.json"),
    ("world_final", "world.final.json"),
    ("replay", REPLAY_ARTIFACT),
)

_OPTIONAL_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("audit_trace", "audit.trace.jsonl"),
    ("audit_security", "audit.security.jsonl"),
    ("transaction_diffs_compatibility", "txn_diffs.jsonl"),
    ("extensions", "extensions.json"),
    ("termination", "termination.json"),
)


def export_world_commit_journal(
    directory: str | Path,
    world: Any,
    *,
    after_cursor: int,
) -> tuple[dict[str, Any], ...]:
    """Export the append-only World commits after an episode's initial snapshot."""

    commits_since = getattr(world, "commits_since", None)
    if not callable(commits_since):
        raise TypeError("episode World does not expose commits_since(cursor)")
    chained: list[dict[str, Any]] = []
    previous_digest: str | None = None
    for source in commits_since(after_cursor):
        record = cast(dict[str, Any], to_canonical(source))
        record["previous_commit_sha256"] = previous_digest
        record["commit_sha256"] = _digest_json(record)
        previous_digest = record["commit_sha256"]
        chained.append(record)
    records = tuple(chained)
    path = Path(directory) / WORLD_COMMITS_ARTIFACT
    body = "".join(_canonical_line(record) for record in records)
    path.write_text(body, encoding="utf-8")
    return records


def finalize_episode_evidence(
    directory: str | Path,
    *,
    scenario_id: str,
    world: Any,
    commit_cursor: int,
    transport: str,
) -> dict[str, Any]:
    """Export commits, run strict state replay, and write the final manifest.

    The manifest is written last.  A malformed decision journal, incomplete
    World journal, or replay mismatch raises and leaves no formal manifest.
    """

    from episode.replay import verify_episode_replay

    root = Path(directory)
    manifest_path = root / EPISODE_EVIDENCE_ARTIFACT
    manifest_path.unlink(missing_ok=True)
    # Low-level Episode callers may construct Runtime directly and never emit an
    # actor report.  Materialize the required empty stream here.  If an actor
    # report was audited without the Runtime acceptance service, the coverage
    # join below still fails closed rather than treating the empty file as valid.
    (root / ACTOR_EVIDENCE_ARTIFACT).touch(exist_ok=True)
    commits = export_world_commit_journal(root, world, after_cursor=commit_cursor)
    decisions = load_platform_decisions(root / PLATFORM_DECISIONS_ARTIFACT, strict=True)
    response_dispositions = load_platform_response_dispositions(
        root / PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT,
        strict=True,
        require_terminal=True,
    )
    actor_contexts = load_actor_contexts(
        root / ACTOR_CONTEXTS_ARTIFACT,
        strict=True,
    )
    actor_evidence = load_actor_evidence(
        root / ACTOR_EVIDENCE_ARTIFACT,
        strict=True,
    )
    verify_actor_evidence_causality(
        audit_path=root / "audit.jsonl",
        contexts_path=root / ACTOR_CONTEXTS_ARTIFACT,
        journal_path=root / ACTOR_EVIDENCE_ARTIFACT,
        expected_episode_context_id=scenario_id,
        strict=True,
        termination_path=(
            root / "termination.json"
            if (root / "termination.json").is_file()
            else None
        ),
    )
    from episode.termination import load_verified_scoreable_termination

    scoreable_termination = load_verified_scoreable_termination(root)
    response_summary = _verify_platform_decision_coverage(
        root / "audit.jsonl",
        decisions,
        response_dispositions,
        allow_not_audited=scoreable_termination is not None,
    )

    replay_result = verify_episode_replay(root, strict=True)
    replay_payload = replay_result.to_dict()
    # Relocation-safe evidence never embeds the machine-specific output path.
    replay_payload["target"] = "."
    (root / REPLAY_ARTIFACT).write_text(
        json.dumps(replay_payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    initial_tables = _snapshot_tables(root / "world.initial.json", phase="initial")
    final_tables = _snapshot_tables(root / "world.final.json", phase="final")
    descriptors: dict[str, Any] = {}
    for role, name in _REQUIRED_ARTIFACTS:
        descriptors[role] = _file_descriptor(root, name)
    for role, name in _OPTIONAL_ARTIFACTS:
        if (root / name).is_file():
            descriptors[role] = _file_descriptor(root, name)

    manifest = {
        "schema_version": EPISODE_EVIDENCE_SCHEMA,
        "scenario_id": scenario_id,
        "execution_backend": "commerceworld_episode",
        "transport": transport,
        "artifacts": descriptors,
        "state": {
            "initial_sha256": _digest_json(initial_tables),
            "final_sha256": _digest_json(final_tables),
        },
        "streams": {
            "audit_envelopes": replay_result.events_verified,
            "platform_decisions": len(decisions),
            "platform_response_events": len(response_dispositions),
            "platform_responses_enqueued": response_summary["enqueued"],
            "platform_responses_audited": response_summary["audited"],
            "platform_responses_not_audited": response_summary["not_audited"],
            "actor_contexts": len(actor_contexts.registrations),
            "actor_evidence": len(actor_evidence),
            "world_commits": len(commits),
            "world_transactions": replay_result.transactions_replayed,
            "world_commit_start_cursor": commit_cursor,
            "world_commit_end_cursor": commit_cursor + len(commits),
            "world_commit_head_sha256": (
                commits[-1]["commit_sha256"] if commits else None
            ),
        },
        "replay": {
            key: value
            for key, value in replay_payload.items()
            if key != "target"
        },
    }
    _atomic_write_json(manifest_path, manifest)
    try:
        return verify_episode_evidence(root)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise


def verify_episode_evidence(directory: str | Path) -> dict[str, Any]:
    """Verify manifest integrity and independently replay its World journal."""

    from episode.replay import verify_episode_replay

    root = Path(directory)
    manifest_path = root / EPISODE_EVIDENCE_ARTIFACT
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid episode evidence manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("episode evidence manifest must be an object")
    if manifest.get("schema_version") != EPISODE_EVIDENCE_SCHEMA:
        raise ValueError("unsupported episode evidence manifest schema")
    if manifest.get("execution_backend") != "commerceworld_episode":
        raise ValueError("episode evidence is not CommerceWorld-backed")
    scenario_id = manifest.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("episode evidence has no scenario_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("episode evidence manifest has no artifact descriptors")

    expected_names = dict(_REQUIRED_ARTIFACTS + _OPTIONAL_ARTIFACTS)
    required_roles = {role for role, _ in _REQUIRED_ARTIFACTS}
    if not required_roles.issubset(artifacts):
        missing = sorted(required_roles - set(artifacts))
        raise ValueError(f"episode evidence is missing required descriptors: {missing}")
    for role, descriptor in artifacts.items():
        if role not in expected_names or not isinstance(descriptor, dict):
            raise ValueError(f"unexpected episode artifact descriptor: {role!r}")
        expected_name = expected_names[role]
        if descriptor.get("path") != expected_name:
            raise ValueError(f"episode artifact path mismatch for {role!r}")
        current = _file_descriptor(root, expected_name)
        if current != descriptor:
            raise ValueError(f"episode artifact integrity mismatch for {role!r}")

    decisions = load_platform_decisions(root / PLATFORM_DECISIONS_ARTIFACT, strict=True)
    response_dispositions = load_platform_response_dispositions(
        root / PLATFORM_RESPONSE_DISPOSITIONS_ARTIFACT,
        strict=True,
        require_terminal=True,
    )
    from episode.termination import load_verified_scoreable_termination

    scoreable_termination = load_verified_scoreable_termination(root)
    response_summary = _verify_platform_decision_coverage(
        root / "audit.jsonl",
        decisions,
        response_dispositions,
        allow_not_audited=scoreable_termination is not None,
    )
    actor_contexts = load_actor_contexts(
        root / ACTOR_CONTEXTS_ARTIFACT,
        strict=True,
    )
    actor_evidence = load_actor_evidence(
        root / ACTOR_EVIDENCE_ARTIFACT,
        strict=True,
    )
    verify_actor_evidence_causality(
        audit_path=root / "audit.jsonl",
        contexts_path=root / ACTOR_CONTEXTS_ARTIFACT,
        journal_path=root / ACTOR_EVIDENCE_ARTIFACT,
        expected_episode_context_id=scenario_id,
        strict=True,
        # Only a manifest-covered termination may relax declared/executed root
        # equality during independent verification.  A stray file outside the
        # artifact inventory has no authority.
        termination_path=(
            root / "termination.json" if "termination" in artifacts else None
        ),
    )
    replay_result = verify_episode_replay(root, strict=True)
    replay_file = json.loads((root / REPLAY_ARTIFACT).read_text(encoding="utf-8"))
    expected_replay = replay_result.to_dict()
    expected_replay["target"] = "."
    if replay_file != expected_replay:
        raise ValueError("stored replay report differs from independent replay")

    initial = _snapshot_tables(root / "world.initial.json", phase="initial")
    final = _snapshot_tables(root / "world.final.json", phase="final")
    state = manifest.get("state")
    if state != {
        "initial_sha256": _digest_json(initial),
        "final_sha256": _digest_json(final),
    }:
        raise ValueError("episode state digests do not match snapshots")
    streams = manifest.get("streams")
    if not isinstance(streams, dict):
        raise ValueError("episode evidence manifest has no stream summary")
    commit_count = artifacts["world_commits"].get("record_count")
    if streams.get("audit_envelopes") != replay_result.events_verified:
        raise ValueError("episode audit count differs from replay")
    if streams.get("platform_decisions") != len(decisions):
        raise ValueError("episode Platform decision count mismatch")
    if streams.get("platform_response_events") != len(response_dispositions):
        raise ValueError("episode Platform response event count mismatch")
    if streams.get("platform_responses_enqueued") != response_summary["enqueued"]:
        raise ValueError("episode Platform response enqueue count mismatch")
    if streams.get("platform_responses_audited") != response_summary["audited"]:
        raise ValueError("episode Platform audited response count mismatch")
    if (
        streams.get("platform_responses_not_audited")
        != response_summary["not_audited"]
    ):
        raise ValueError("episode Platform non-audited response count mismatch")
    if streams.get("actor_contexts") != len(actor_contexts.registrations):
        raise ValueError("episode actor context count mismatch")
    if streams.get("actor_evidence") != len(actor_evidence):
        raise ValueError("episode actor evidence count mismatch")
    if streams.get("world_commits") != commit_count:
        raise ValueError("episode World commit count mismatch")
    commit_records = [
        json.loads(line)
        for line in (root / WORLD_COMMITS_ARTIFACT).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    expected_head = commit_records[-1]["commit_sha256"] if commit_records else None
    if streams.get("world_commit_head_sha256") != expected_head:
        raise ValueError("episode World commit head digest mismatch")
    if streams.get("world_transactions") != replay_result.transactions_replayed:
        raise ValueError("episode transaction count differs from replay")
    start = streams.get("world_commit_start_cursor")
    end = streams.get("world_commit_end_cursor")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end != start + commit_count
    ):
        raise ValueError("episode World commit cursor range is invalid")
    manifest_replay = manifest.get("replay")
    expected_manifest_replay = {
        key: value for key, value in expected_replay.items() if key != "target"
    }
    if manifest_replay != expected_manifest_replay:
        raise ValueError("episode manifest replay summary is stale")
    return manifest


def _snapshot_tables(path: Path, *, phase: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("phase") != phase or not isinstance(value.get("tables"), dict):
        raise ValueError(f"invalid {phase} snapshot: {path}")
    return cast(dict[str, Any], value["tables"])


def _verify_platform_decision_coverage(
    audit_path: Path,
    decisions: list[dict[str, Any]],
    response_dispositions: list[dict[str, Any]],
    *,
    allow_not_audited: bool,
) -> dict[str, int]:
    """Join produced, queued, audited, and non-audited Platform responses.

    ``platform.decisions.jsonl`` is the authoritative production fact.
    ``audit.jsonl`` is the delivery fact.  Runtime's disposition stream binds
    the two without rewriting either append-only source.  A non-audited
    terminal is legal only for a separately verified, scoreable Agent abort.
    """

    requests: list[tuple[int, Any]] = []
    audited_by_identity: dict[tuple[str, str], list[tuple[int, Any]]] = {}
    platform_authored_by_position: dict[int, Any] = {}
    for line_number, line in enumerate(
        audit_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        encoded = record.get("envelope") if isinstance(record, dict) else None
        if not isinstance(encoded, str):
            raise ValueError(f"invalid audit envelope at {audit_path}:{line_number}")
        envelope = from_json(encoded)
        position = line_number - 1
        identity = (
            envelope.msg_id,
            hashlib.sha256(to_json(envelope).encode("utf-8")).hexdigest(),
        )
        audited_by_identity.setdefault(identity, []).append((position, envelope))
        if envelope.from_ == "platform" or envelope.from_.startswith("platform:"):
            platform_authored_by_position[position] = envelope
        if envelope.to == "platform" or envelope.to.startswith("platform:"):
            requests.append((position, envelope))
    if len(requests) != len(decisions):
        raise ValueError(
            "Platform decision coverage mismatch: every audited request requires one decision"
        )

    enqueued_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    terminal_by_occurrence: dict[str, dict[str, Any]] = {}
    for event in response_dispositions:
        state = event["state"]
        occurrence_id = str(event["occurrence_id"])
        if state == "enqueued":
            key = (int(event["decision_sequence"]), int(event["response_index"]))
            if key in enqueued_by_key:
                raise ValueError("Platform response enqueue key is ambiguous")
            enqueued_by_key[key] = event
        else:
            terminal_by_occurrence[occurrence_id] = event

    claimed_audit_positions: set[int] = set()
    produced_response_ids: set[str] = set()
    consumed_enqueue_keys: set[tuple[int, int]] = set()
    audited_count = 0
    not_audited_count = 0
    for position, ((request_position, request), decision) in enumerate(
        zip(requests, decisions, strict=True)
    ):
        normalized_action = json.loads(
            json.dumps(
                request.action,
                allow_nan=False,
                default=str,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        expected = {
            "request_msg_id": request.msg_id,
            "request_envelope_sha256": hashlib.sha256(
                to_json(request).encode("utf-8")
            ).hexdigest(),
            "actor_id": request.from_,
            "platform_endpoint": request.to,
            "action_kind": str(normalized_action.get("kind", "")),
            "idempotency_key": request.idempotency_key,
            "normalized_action": normalized_action,
        }
        for field, value in expected.items():
            if decision.get(field) != value:
                raise ValueError(
                    f"Platform decision {position} does not match audited request field {field!r}"
                )
        response_ids = decision.get("response_msg_ids", [])
        response_hashes = decision.get("response_sha256s", [])
        response_kinds = decision.get("response_kinds", [])
        if (
            not isinstance(response_ids, list)
            or not isinstance(response_hashes, list)
            or len(response_ids) != len(response_hashes)
            or not isinstance(response_kinds, list)
            or len(set(response_ids)) != len(response_ids)
        ):
            raise ValueError(f"Platform decision {position} has invalid response linkage")
        produced_kinds: list[str] = []
        for response_index, (response_id, response_sha256) in enumerate(
            zip(response_ids, response_hashes, strict=True)
        ):
            produced_response_ids.add(str(response_id))
            key = (position, response_index)
            enqueued = enqueued_by_key.get(key)
            if enqueued is None:
                raise ValueError(
                    f"Platform decision {position} response was not enqueued by Runtime"
                )
            consumed_enqueue_keys.add(key)
            if (
                enqueued.get("run_id") != decision.get("run_id")
                or enqueued.get("request_msg_id") != request.msg_id
                or enqueued.get("response_msg_id") != response_id
                or enqueued.get("response_envelope_sha256") != response_sha256
            ):
                raise ValueError(
                    f"Platform decision {position} response enqueue does not match production"
                )
            produced_kinds.append(str(enqueued.get("response_kind", "")))
            terminal = terminal_by_occurrence.get(str(enqueued["occurrence_id"]))
            if terminal is None:
                raise ValueError(
                    f"Platform decision {position} response has no terminal disposition"
                )
            identity = (str(response_id), str(response_sha256))
            candidates = [
                (candidate_position, candidate)
                for candidate_position, candidate in audited_by_identity.get(identity, [])
                if candidate_position > request_position
                and candidate_position not in claimed_audit_positions
            ]
            if terminal.get("state") == "audited":
                if not candidates:
                    raise ValueError(
                        f"Platform decision {position} audited response is absent"
                    )
                candidate_position, _candidate = candidates[0]
                claimed_audit_positions.add(candidate_position)
                audited_count += 1
            elif terminal.get("state") == "not_audited_at_shutdown":
                if not allow_not_audited:
                    raise ValueError(
                        "Platform response was not audited without a scoreable Agent abort"
                    )
                if candidates:
                    raise ValueError(
                        f"Platform decision {position} marks an audited response as absent"
                    )
                not_audited_count += 1
            else:
                raise ValueError(
                    f"Platform decision {position} response terminal state is invalid"
                )
        if sorted(produced_kinds) != response_kinds:
            raise ValueError(
                f"Platform decision {position} response kinds do not match Runtime enqueue"
            )
    if consumed_enqueue_keys != set(enqueued_by_key):
        raise ValueError("Runtime enqueued a Platform response without a decision claim")
    request_msg_ids = {request.msg_id for _position, request in requests}
    for audit_position, envelope in platform_authored_by_position.items():
        if audit_position in claimed_audit_positions:
            continue
        # A Platform-authored scenario input is not a decision response.  It
        # remains an ordinary audited envelope.  A response-shaped occurrence,
        # however, may not escape the Runtime disposition exact join merely by
        # changing its bytes or omitting its lifecycle record.
        if (
            envelope.msg_id in produced_response_ids
            or envelope.in_reply_to in request_msg_ids
        ):
            raise ValueError(
                "audited Platform response occurrence has no Runtime disposition"
            )
    return {
        "enqueued": len(enqueued_by_key),
        "audited": audited_count,
        "not_audited": not_audited_count,
    }


def _file_descriptor(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required episode artifact: {name}") from exc
    descriptor: dict[str, Any] = {
        "path": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if name.endswith(".jsonl"):
        descriptor["record_count"] = sum(
            1 for line in raw.splitlines() if line.strip()
        )
    return descriptor


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_line(value: Any) -> str:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "ACTOR_CONTEXTS_ARTIFACT",
    "ACTOR_EVIDENCE_ARTIFACT",
    "EPISODE_EVIDENCE_ARTIFACT",
    "EPISODE_EVIDENCE_SCHEMA",
    "PLATFORM_DECISIONS_ARTIFACT",
    "REPLAY_ARTIFACT",
    "WORLD_COMMITS_ARTIFACT",
    "export_world_commit_journal",
    "finalize_episode_evidence",
    "verify_episode_evidence",
]
