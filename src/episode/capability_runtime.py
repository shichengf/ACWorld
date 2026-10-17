"""Runtime benchmark task contracts and evidence readers.

This module is the formal benchmark boundary.  A result is eligible only when
it was produced by a real :class:`episode.runner.Episode` and contains the
artifacts emitted by Runtime, Platform, and World.  Compact direct simulators
are intentionally outside this API.

Family modules build :class:`RuntimeTaskBundleV2` objects.  The experiment
executor supplies the evaluated model channel, while the bundle supplies only
deterministic counterpart policies, an ideal conformance policy, and a Python
scorer over stored CommerceWorld evidence.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from episode.capability_benchmark import TaskDefinitionV2
from episode.types import ScenarioSpec
from protocol.envelope import from_json, to_json

if TYPE_CHECKING:
    from agents.inference import InferenceChannel


COMMERCEWORLD_EPISODE_BACKEND_V2 = "commerceworld_episode"
RUNTIME_TASK_SCORE_SCHEMA_V3 = "cwe.runtime-task-score.v3"
RUNTIME_MODEL_SAFETY_SCORE_CAP_V3 = 0.25


_OPERATION_EVIDENCE_CALL_RECORDER: ContextVar[list[tuple[str, str, Any]] | None] = ContextVar(
    "operation_evidence_call_recorder", default=None
)


@contextmanager
def record_verified_operation_evidence_calls() -> Iterator[list[tuple[str, str, Any]]]:
    """Capture successful registered authority-contract calls made by a scorer.

    The recorder is context-local, so concurrent task scorers cannot mix
    evidence.  It observes only results returned after the core exact-join
    verifier succeeds.
    """

    calls: list[tuple[str, str, Any]] = []
    token = _OPERATION_EVIDENCE_CALL_RECORDER.set(calls)
    try:
        yield calls
    finally:
        _OPERATION_EVIDENCE_CALL_RECORDER.reset(token)


class RuntimeEvidenceError(RuntimeError):
    """A run is missing or contradicts required CommerceWorld evidence."""


class RuntimeBenchmarkIntegrityError(RuntimeEvidenceError):
    """A run is ineligible because CommerceWorld execution integrity failed.

    This is deliberately not a rubric outcome.  Backend, replay, authority,
    and exact Platform/World joins are properties of the environment.  A
    failure here invalidates the run instead of lowering the evaluated
    model's capability score.
    """


@dataclass(frozen=True)
class LinkedPlatformExchangeV2:
    """One Platform request joined to its exact decision and audited responses.

    Family scorers must use the accepted-only reader when claiming an
    authoritative operation.  Formal protocol coverage may inspect both
    accepted and rejected exchanges, but only after this byte-exact join proves
    that the request reached Platform and received one canonical decision.
    """

    request: dict[str, Any]
    decision: dict[str, Any]
    responses: tuple[dict[str, Any], ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the stable digest used by evidence and contract manifests."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeEvidenceError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeEvidenceError(f"{label} at {path} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, required: bool, label: str) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        if required:
            raise RuntimeEvidenceError(f"missing required {label}: {path}")
        return ()
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for ordinal, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeEvidenceError(f"{label} row {ordinal} at {path} must be a JSON object")
            rows.append(value)
    except RuntimeEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeEvidenceError(f"cannot read {label} at {path}: {exc}") from exc
    return tuple(rows)


def _decode_audit_envelope(row: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    raw = row.get("envelope")
    if not isinstance(raw, str):
        raise RuntimeEvidenceError(f"audit row {ordinal} has no canonical envelope")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeEvidenceError(
            f"audit row {ordinal} contains an invalid envelope: {exc}"
        ) from exc
    if not isinstance(envelope, dict):
        raise RuntimeEvidenceError(f"audit row {ordinal} envelope must be an object")
    action = envelope.get("action")
    if not isinstance(action, dict) or not isinstance(action.get("kind"), str):
        raise RuntimeEvidenceError(f"audit row {ordinal} envelope has no typed action")
    return envelope


@dataclass(frozen=True)
class RuntimeEvidenceBundleV2:
    """Stored evidence from one completed CommerceWorld episode.

    ``platform_decisions`` and ``world_events`` are additive during the tracker
    migration.  Formal eligibility is controlled by ``require_complete_journal``;
    tests can inspect legacy episodes without accidentally approving them.
    """

    episode_dir: Path
    initial_world: dict[str, Any]
    final_world: dict[str, Any]
    audit_rows: tuple[dict[str, Any], ...]
    envelopes: tuple[dict[str, Any], ...]
    trace_rows: tuple[dict[str, Any], ...]
    platform_decisions: tuple[dict[str, Any], ...]
    world_events: tuple[dict[str, Any], ...]
    evidence_manifest: dict[str, Any] | None
    initial_digest: str
    final_digest: str
    actor_evidence: tuple[dict[str, Any], ...] = ()
    platform_response_dispositions: tuple[dict[str, Any], ...] = ()
    runtime_response_source_sha256: str | None = None

    @classmethod
    def load(
        cls,
        episode_dir: str | Path,
        *,
        require_complete_journal: bool = True,
    ) -> "RuntimeEvidenceBundleV2":
        root = Path(episode_dir)
        initial = _read_object(root / "world.initial.json", label="initial World snapshot")
        final = _read_object(root / "world.final.json", label="final World snapshot")
        if initial.get("phase") != "initial" or final.get("phase") != "final":
            raise RuntimeEvidenceError("World snapshots have invalid phase markers")
        if not isinstance(initial.get("tables"), dict) or not isinstance(final.get("tables"), dict):
            raise RuntimeEvidenceError("World snapshots must contain table objects")

        audit = _read_jsonl(root / "audit.jsonl", required=True, label="Runtime audit")
        if not audit:
            raise RuntimeEvidenceError("Runtime audit is empty")
        envelopes = tuple(
            _decode_audit_envelope(row, ordinal=index) for index, row in enumerate(audit, start=1)
        )
        trace = _read_jsonl(root / "audit.trace.jsonl", required=True, label="agent decision trace")

        platform_path = root / "platform.decisions.jsonl"
        world_path = root / "world.commits.jsonl"
        platform = _read_jsonl(
            platform_path,
            required=require_complete_journal,
            label="Platform decision journal",
        )
        platform_response_dispositions = _read_jsonl(
            root / "platform.response-dispositions.jsonl",
            required=require_complete_journal,
            label="Platform response disposition journal",
        )
        world_events = _read_jsonl(
            world_path,
            required=require_complete_journal,
            label="World commit journal",
        )
        actor_evidence = _read_jsonl(
            root / "actor.evidence.jsonl",
            required=require_complete_journal,
            label="Runtime actor evidence journal",
        )
        # Read-only and actor-report-only tasks may legitimately have no
        # Platform decision or World mutation.  The required files and manifest
        # coverage still prove that the empty streams are intentional.

        manifest: dict[str, Any] | None = None
        if require_complete_journal:
            try:
                from episode.evidence import verify_episode_evidence

                manifest = verify_episode_evidence(root)
            except Exception as exc:
                raise RuntimeEvidenceError(
                    f"episode evidence manifest or replay is invalid: {type(exc).__name__}: {exc}"
                ) from exc
            if manifest.get("execution_backend") != COMMERCEWORLD_EPISODE_BACKEND_V2:
                raise RuntimeEvidenceError("episode evidence used a non-CommerceWorld backend")

        return cls(
            episode_dir=root,
            initial_world=initial,
            final_world=final,
            audit_rows=audit,
            envelopes=envelopes,
            trace_rows=trace,
            platform_decisions=platform,
            world_events=world_events,
            evidence_manifest=manifest,
            initial_digest=canonical_sha256(initial["tables"]),
            final_digest=canonical_sha256(final["tables"]),
            actor_evidence=actor_evidence,
            platform_response_dispositions=platform_response_dispositions,
            runtime_response_source_sha256=(
                canonical_sha256(
                    {
                        "envelopes": envelopes,
                        "platform_decisions": platform,
                    }
                )
                if require_complete_journal
                else None
            ),
        )

    def actions(
        self,
        *,
        kind: str | None = None,
        actor_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return canonical envelopes filtered by action kind and sender."""

        rows = self.envelopes
        if kind is not None:
            rows = tuple(row for row in rows if row["action"]["kind"] == kind)
        if actor_id is not None:
            rows = tuple(row for row in rows if row.get("from") == actor_id)
        return rows

    def accepted_platform_exchanges(
        self,
        *,
        kind: str | None = None,
        actor_id: str | None = None,
        endpoint: str | None = None,
        response_kind: str | None = None,
    ) -> tuple[LinkedPlatformExchangeV2, ...]:
        """Return byte-linked Platform exchanges whose decision was accepted.

        Filtering happens only after the complete request/decision/response
        join is verified.  A rejected request, an unlinked actor action, or a
        response with a mismatched hash can therefore never receive authority
        credit from a family scorer.
        """

        return self.platform_exchanges(
            kind=kind,
            actor_id=actor_id,
            endpoint=endpoint,
            response_kind=response_kind,
            decision="accepted",
        )

    def platform_exchanges(
        self,
        *,
        kind: str | None = None,
        actor_id: str | None = None,
        endpoint: str | None = None,
        response_kind: str | None = None,
        decision: str | None = None,
    ) -> tuple[LinkedPlatformExchangeV2, ...]:
        """Return all byte-linked Platform decisions, optionally filtered.

        ``decision=None`` includes both accepted and rejected requests.  This
        does not grant transaction authority to a rejected action.  It exists
        so experiment accounting can distinguish a real, Platform-observed
        protocol attempt from no output or a Runtime-rejected envelope.
        """

        if decision not in {None, "accepted", "rejected"}:
            raise ValueError("decision must be accepted, rejected, or None")
        try:
            from runtime.exact_join import build_exact_join_context

            context = build_exact_join_context(
                envelopes=self.envelopes,
                platform_decisions=self.platform_decisions,
                platform_response_dispositions=(self._platform_response_dispositions_for_join()),
                world_commits=self.world_events,
                initial_tables=self.initial_world.get("tables", {}),
                final_tables=self.final_world.get("tables", {}),
                allow_not_audited_platform_responses=(
                    self._allows_not_audited_platform_responses()
                ),
            )
        except Exception as exc:
            raise RuntimeEvidenceError(
                f"Platform evidence does not form unique exact exchanges: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        joined: list[LinkedPlatformExchangeV2] = []
        for exchange in context.exchanges:
            outcome = exchange.decision
            if decision is not None and outcome.get("decision") != decision:
                continue
            if kind is not None and outcome.get("action_kind") != kind:
                continue
            if actor_id is not None and outcome.get("actor_id") != actor_id:
                continue
            if endpoint is not None and outcome.get("platform_endpoint") != endpoint:
                continue
            if response_kind is not None and not any(
                (row.get("action") or {}).get("kind") == response_kind for row in exchange.responses
            ):
                continue
            joined.append(
                LinkedPlatformExchangeV2(
                    request=exchange.request,
                    decision=outcome,
                    responses=exchange.responses,
                )
            )
        return tuple(joined)

    def verified_operation_evidence(
        self,
        contract_id: str,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Verify one registered Runtime→Platform→World operation graph.

        Family scorers call this core entry rather than rebuilding causal joins
        from task-local filters.  The registry verifies all Platform exchanges
        before applying the selected World operation contract, so duplicate or
        multiply-claimed evidence fails closed even when it belongs to another
        task lane.
        """

        manifest = self.evidence_manifest
        replay = manifest.get("replay") if isinstance(manifest, Mapping) else None
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("execution_backend") != COMMERCEWORLD_EPISODE_BACKEND_V2
            or not isinstance(replay, Mapping)
            or replay.get("replay_ok") is not True
        ):
            raise RuntimeEvidenceError(
                "verified operation evidence requires a complete replay-verified "
                "CommerceWorld episode manifest"
            )
        try:
            from runtime.exact_join import verify_registered_operation_evidence

            result = verify_registered_operation_evidence(
                contract_id,
                envelopes=self.envelopes,
                platform_decisions=self.platform_decisions,
                platform_response_dispositions=(self._platform_response_dispositions_for_join()),
                world_commits=self.world_events,
                initial_tables=self.initial_world.get("tables", {}),
                final_tables=self.final_world.get("tables", {}),
                options=options,
                allow_not_audited_platform_responses=(
                    self._allows_not_audited_platform_responses()
                ),
            )
            recorder = _OPERATION_EVIDENCE_CALL_RECORDER.get()
            if recorder is not None:
                recorder.append((contract_id, canonical_sha256(dict(options or {})), result))
            return result
        except Exception as exc:
            raise RuntimeEvidenceError(
                f"operation evidence contract {contract_id!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _allows_not_audited_platform_responses(self) -> bool:
        if not self._uses_runtime_owned_response_lifecycle():
            return False
        manifest = self.evidence_manifest
        streams = manifest.get("streams") if isinstance(manifest, Mapping) else None
        count = (
            streams.get("platform_responses_not_audited") if isinstance(streams, Mapping) else None
        )
        if not (isinstance(count, int) and not isinstance(count, bool) and count > 0):
            return False
        from episode.termination import load_verified_scoreable_termination

        try:
            return load_verified_scoreable_termination(self.episode_dir) is not None
        except ValueError:
            return False

    def _uses_runtime_owned_response_lifecycle(self) -> bool:
        expected = self.runtime_response_source_sha256
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        current = canonical_sha256(
            {
                "envelopes": self.envelopes,
                "platform_decisions": self.platform_decisions,
            }
        )
        return current == expected

    def _platform_response_dispositions_for_join(
        self,
    ) -> tuple[dict[str, Any], ...]:
        if not self._uses_runtime_owned_response_lifecycle():
            # Manually constructed or historically derived evidence continues
            # through the legacy exact response join.  A freshly loaded formal
            # Episode always has the source fingerprint above and therefore
            # cannot omit its Runtime-owned lifecycle.
            return ()
        return self.platform_response_dispositions

    def accepted_actor_results(
        self,
        *,
        action_kind: str | None = None,
        actor_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return Runtime-accepted, manifest-joined actor result records.

        ``load(require_complete_journal=True)`` has already verified each row's
        exact request hash against ``audit.jsonl``.  Family scorers use this
        stream for actor-authored conclusions and join it separately to
        authoritative Platform decisions and World facts.
        """

        rows = self.actor_evidence
        if action_kind is not None:
            rows = tuple(row for row in rows if row.get("action_kind") == action_kind)
        if actor_id is not None:
            rows = tuple(row for row in rows if row.get("actor_id") == actor_id)
        if task_id is not None:
            rows = tuple(row for row in rows if row.get("task_id") == task_id)
        return rows


def attested_world_catalog_reads_v2(
    evidence: RuntimeEvidenceBundleV2,
    *,
    actor_id: str,
) -> dict[str, Mapping[str, Any]]:
    """Return successful catalog reads proven by Tracker and wire audit.

    A model observation is scoreable only when one usable, completed Tracker
    tool result names the exact audited ``world.response`` that answers the
    exact audited ``world.read_catalog`` request.  This deliberately excludes
    the old in-process ``WorldTools`` shortcut, pending/request-only reads, and
    failed World results.  Missing or contradictory wire evidence is an Agent
    or World integrity failure, never a model grounding mistake.
    """

    from agents.turn import ToolCall, world_response_to_tool_result
    from runtime.tracker_evidence import tracker_row_has_usable_completed_steps

    requests = tuple(
        row
        for row in evidence.actions(kind="world.read_catalog", actor_id=actor_id)
        if row.get("to") == "world"
        and isinstance((row.get("action") or {}).get("payload"), Mapping)
        and isinstance((row.get("action") or {}).get("payload", {}).get("sku_id"), str)
    )
    responses_by_request: dict[str, list[dict[str, Any]]] = {}
    responses_by_id: dict[str, dict[str, Any]] = {}
    for row in evidence.actions(kind="world.response"):
        if row.get("from") != "world" or row.get("to") != actor_id:
            continue
        response_id = row.get("msg_id")
        request_id = row.get("in_reply_to")
        if not isinstance(response_id, str) or not isinstance(request_id, str):
            raise RuntimeBenchmarkIntegrityError(
                "actor World catalog response has invalid correlation identity"
            )
        if response_id in responses_by_id:
            raise RuntimeBenchmarkIntegrityError(
                "actor World catalog response identity is duplicated"
            )
        responses_by_id[response_id] = row
        responses_by_request.setdefault(request_id, []).append(row)

    request_by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        request_id = request.get("msg_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeBenchmarkIntegrityError(
                "actor World catalog request has no message identity"
            )
        if request_id in request_by_id:
            raise RuntimeBenchmarkIntegrityError(
                "actor World catalog request identity is duplicated"
            )
        request_by_id[request_id] = request
        correlated = responses_by_request.get(request_id, [])
        if len(correlated) != 1:
            raise RuntimeBenchmarkIntegrityError(
                "actor World catalog request has no unique audited response"
            )

    result_by_source: dict[str, Mapping[str, Any]] = {}
    for trace in evidence.trace_rows:
        if trace.get("agent_id") != actor_id or not tracker_row_has_usable_completed_steps(trace):
            continue
        for step in trace.get("steps", ()):
            if not isinstance(step, Mapping) or step.get("kind") != "tool_call":
                continue
            data = step.get("data")
            results = data.get("results", ()) if isinstance(data, Mapping) else ()
            for result_slot in results:
                if (
                    not isinstance(result_slot, Mapping)
                    or result_slot.get("tool") != "world.get_listing"
                ):
                    continue
                source_id = result_slot.get("source_msg_id")
                if not isinstance(source_id, str) or not source_id:
                    raise RuntimeBenchmarkIntegrityError(
                        "completed catalog observation bypassed audited World transport"
                    )
                if source_id in result_by_source:
                    raise RuntimeBenchmarkIntegrityError(
                        "audited World catalog response is claimed by multiple Tracker results"
                    )
                result_by_source[source_id] = result_slot

    found: dict[str, Mapping[str, Any]] = {}
    for request_id, request in request_by_id.items():
        [response] = responses_by_request[request_id]
        response_id = str(response["msg_id"])
        result_slot = result_by_source.get(response_id)
        if result_slot is None:
            raise RuntimeBenchmarkIntegrityError(
                "audited World catalog response has no completed Tracker result"
            )
        args = result_slot.get("args")
        request_payload = (request.get("action") or {}).get("payload")
        if not isinstance(args, Mapping) or not isinstance(request_payload, Mapping):
            raise RuntimeBenchmarkIntegrityError(
                "catalog Tracker result has malformed request arguments"
            )
        requested_sku = request_payload.get("sku_id")
        if not isinstance(requested_sku, str) or args != {"sku_id": requested_sku}:
            raise RuntimeBenchmarkIntegrityError(
                "catalog Tracker arguments contradict the audited World request"
            )
        try:
            response_envelope = from_json(_canonical_json(response))
            expected_result = world_response_to_tool_result(
                ToolCall(tool="world.get_listing", args={"sku_id": requested_sku}),
                response_envelope,
            )
        except Exception as exc:
            raise RuntimeBenchmarkIntegrityError(
                "audited World catalog response cannot be projected to a tool result"
            ) from exc
        if dict(result_slot) != expected_result:
            raise RuntimeBenchmarkIntegrityError(
                "catalog Tracker result contradicts the exact audited World response"
            )
        response_payload = (response.get("action") or {}).get("payload")
        result = result_slot.get("result")
        if (
            isinstance(response_payload, Mapping)
            and response_payload.get("sku_id") == requested_sku
            and isinstance(result, Mapping)
            and result.get("sku_id") == requested_sku
        ):
            found[requested_sku] = result

    unclaimed_sources = set(result_by_source).difference(responses_by_id)
    if unclaimed_sources:
        raise RuntimeBenchmarkIntegrityError(
            "completed catalog Tracker result names a non-audited World response"
        )
    return found


def _wire_envelope_sha256(envelope: Mapping[str, Any]) -> str:
    """Hash one decoded audit envelope through the canonical wire codec."""

    try:
        parsed = from_json(
            json.dumps(
                dict(envelope),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except Exception as exc:
        raise RuntimeEvidenceError("cannot canonicalize audited envelope") from exc
    return hashlib.sha256(to_json(parsed).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeRubricCheckV2:
    """One independently verifiable deterministic scoring step."""

    name: str
    weight: float
    credit: float
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("rubric check needs a name")
        if not 0 < self.weight <= 1:
            raise ValueError(f"rubric weight must be in (0, 1], got {self.weight!r}")
        if not 0 <= self.credit <= 1:
            raise ValueError(f"rubric credit must be in [0, 1], got {self.credit!r}")

    @property
    def earned(self) -> float:
        return self.weight * self.credit

    def to_dict(self) -> dict[str, Any]:
        # Score payloads are persisted as JSON and then compared byte-for-byte
        # with a fresh deterministic rescore.  A shallow copy would preserve
        # nested tuples in memory even though JSON turns them into arrays,
        # creating a false rescore mismatch.  Normalize the entire evidence
        # tree at this boundary so ``to_dict`` actually returns its advertised
        # JSON-native representation and rejects non-finite numbers.
        evidence = json.loads(_canonical_json(self.evidence))
        if not isinstance(evidence, dict):  # ``evidence`` is typed as an object.
            raise ValueError("rubric evidence must encode as a JSON object")
        return {
            "name": self.name,
            "weight": self.weight,
            "credit": self.credit,
            "earned": round(self.earned, 6),
            "evidence": evidence,
        }


@dataclass(frozen=True)
class RuntimeTaskScoreV3:
    """Capability-only score produced after CommerceWorld validity succeeds.

    Environment, protocol transport, Agent compilation, authority closure, and
    replay do not appear in this schema.  They must pass before a V3 score is
    materialized.  A model-authored unsafe business choice remains a genuine
    capability outcome and may retain the benchmark's safety cap.
    """

    task_id: str
    family: str
    evaluated_role: str
    checks: tuple[RuntimeRubricCheckV2, ...]
    issues: tuple[str, ...] = ()
    model_safety_violation: bool = False
    model_privacy_violation: bool = False
    schema_version: str = RUNTIME_TASK_SCORE_SCHEMA_V3

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_TASK_SCORE_SCHEMA_V3:
            raise ValueError("unsupported runtime task score v3 schema")
        total_weight = sum(row.weight for row in self.checks)
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"rubric weights must sum to 1, got {total_weight!r}")
        if len({row.name for row in self.checks}) != len(self.checks):
            raise ValueError("rubric check names must be unique")

    @property
    def raw_score(self) -> float:
        return round(sum(row.earned for row in self.checks), 6)

    @property
    def capability_score(self) -> float:
        if self.model_safety_violation:
            return round(min(self.raw_score, RUNTIME_MODEL_SAFETY_SCORE_CAP_V3), 6)
        return self.raw_score

    @property
    def strict_success(self) -> bool:
        return (
            self.capability_score == 1.0
            and not self.issues
            and not self.model_safety_violation
            and not self.model_privacy_violation
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability_checks": [row.to_dict() for row in self.checks],
            "capability_score": self.capability_score,
            "strict_success": self.strict_success,
            "model_safety_violation": self.model_safety_violation,
            "model_privacy_violation": self.model_privacy_violation,
        }


def require_runtime_benchmark_integrity_v2(
    evidence: RuntimeEvidenceBundleV2,
) -> None:
    """Fail closed unless stored evidence is a real, replayed CommerceWorld run.

    The gate checks only environment-owned invariants.  It intentionally does
    not require the evaluated actor to act, obtain Platform acceptance, or
    complete the task.  Those are model outcomes and remain scoreable as zero
    or partial progress.
    """

    manifest = evidence.evidence_manifest
    replay = manifest.get("replay") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("execution_backend") != COMMERCEWORLD_EPISODE_BACKEND_V2
        or not isinstance(replay, Mapping)
        or replay.get("replay_ok") is not True
    ):
        raise RuntimeBenchmarkIntegrityError(
            "episode manifest does not attest a replayed CommerceWorld backend"
        )

    # Scorers may receive an in-memory bundle that has been copied or replaced
    # after loading.  Rejoin every score-bearing stream to the immutable files
    # covered by the episode manifest so a same-length splice cannot evade the
    # stream-count checks below.
    def persisted_object(name: str, label: str) -> dict[str, Any]:
        try:
            return _read_object(evidence.episode_dir / name, label=label)
        except RuntimeEvidenceError as exc:
            raise RuntimeBenchmarkIntegrityError(
                f"manifest-covered {label} cannot be reread"
            ) from exc

    def persisted_jsonl(name: str, label: str) -> tuple[dict[str, Any], ...]:
        try:
            return _read_jsonl(
                evidence.episode_dir / name,
                required=True,
                label=label,
            )
        except RuntimeEvidenceError as exc:
            raise RuntimeBenchmarkIntegrityError(
                f"manifest-covered {label} cannot be reread"
            ) from exc

    persisted_manifest = persisted_object(
        "episode.evidence.json",
        "episode evidence manifest",
    )
    persisted_initial = persisted_object(
        "world.initial.json",
        "initial World snapshot",
    )
    persisted_final = persisted_object(
        "world.final.json",
        "final World snapshot",
    )
    persisted_audit = persisted_jsonl("audit.jsonl", "Runtime audit")
    persisted_streams = {
        "trace": persisted_jsonl("audit.trace.jsonl", "agent decision trace"),
        "platform": persisted_jsonl(
            "platform.decisions.jsonl", "Platform decision journal"
        ),
        "dispositions": persisted_jsonl(
            "platform.response-dispositions.jsonl",
            "Platform response disposition journal",
        ),
        "actor": persisted_jsonl(
            "actor.evidence.jsonl", "Runtime actor evidence journal"
        ),
        "world": persisted_jsonl("world.commits.jsonl", "World commit journal"),
    }
    persisted_envelopes = tuple(
        _decode_audit_envelope(row, ordinal=index)
        for index, row in enumerate(persisted_audit, start=1)
    )
    if (
        persisted_manifest != manifest
        or persisted_initial != evidence.initial_world
        or persisted_final != evidence.final_world
        or persisted_audit != evidence.audit_rows
        or persisted_envelopes != evidence.envelopes
        or persisted_streams["trace"] != evidence.trace_rows
        or persisted_streams["platform"] != evidence.platform_decisions
        or persisted_streams["dispositions"] != evidence.platform_response_dispositions
        or persisted_streams["actor"] != evidence.actor_evidence
        or persisted_streams["world"] != evidence.world_events
    ):
        raise RuntimeBenchmarkIntegrityError(
            "in-memory evidence differs from the manifest-covered episode artifacts"
        )

    initial_tables = evidence.initial_world.get("tables")
    final_tables = evidence.final_world.get("tables")
    if not isinstance(initial_tables, Mapping) or not isinstance(final_tables, Mapping):
        raise RuntimeBenchmarkIntegrityError("World snapshots do not contain table mappings")
    current_initial_digest = canonical_sha256(initial_tables)
    current_final_digest = canonical_sha256(final_tables)
    state_manifest = manifest.get("state")
    if (
        current_initial_digest != evidence.initial_digest
        or current_final_digest != evidence.final_digest
        or not isinstance(state_manifest, Mapping)
        or state_manifest.get("initial_sha256") != current_initial_digest
        or state_manifest.get("final_sha256") != current_final_digest
    ):
        raise RuntimeBenchmarkIntegrityError(
            "World snapshots do not match their captured manifest digests"
        )

    streams = manifest.get("streams")
    if not isinstance(streams, Mapping):
        raise RuntimeBenchmarkIntegrityError("episode manifest has no stream accounting")
    expected_stream_counts = {
        "audit_envelopes": len(evidence.envelopes),
        "platform_decisions": len(evidence.platform_decisions),
        "platform_response_events": len(evidence.platform_response_dispositions),
        "actor_evidence": len(evidence.actor_evidence),
        "world_commits": len(evidence.world_events),
    }
    if any(streams.get(name) != count for name, count in expected_stream_counts.items()):
        raise RuntimeBenchmarkIntegrityError(
            "in-memory evidence streams do not match manifest accounting"
        )

    commit_ids = tuple(row.get("commit_id") for row in evidence.world_events)
    sequences = tuple(row.get("sequence") for row in evidence.world_events)
    if any(not isinstance(value, str) or not value for value in commit_ids):
        raise RuntimeBenchmarkIntegrityError("World commit journal has a missing commit id")
    if len(commit_ids) != len(set(commit_ids)):
        raise RuntimeBenchmarkIntegrityError("World commit journal has duplicate commit ids")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in sequences):
        raise RuntimeBenchmarkIntegrityError("World commit journal has an invalid sequence")
    if len(sequences) != len(set(sequences)) or sequences != tuple(sorted(sequences)):
        raise RuntimeBenchmarkIntegrityError("World commit journal is not in canonical order")
    previous_commit_sha256: str | None = None
    for row in evidence.world_events:
        observed_commit_sha256 = row.get("commit_sha256")
        digest_payload = dict(row)
        digest_payload.pop("commit_sha256", None)
        invariants = row.get("invariants_held")
        if (
            row.get("previous_commit_sha256") != previous_commit_sha256
            or not isinstance(observed_commit_sha256, str)
            or canonical_sha256(digest_payload) != observed_commit_sha256
            or not isinstance(invariants, list)
            or not invariants
            or any(not isinstance(value, str) or not value for value in invariants)
        ):
            raise RuntimeBenchmarkIntegrityError(
                "World commit journal has a broken hash chain or invariant record"
            )
        previous_commit_sha256 = observed_commit_sha256
    if streams.get("world_commit_head_sha256") != previous_commit_sha256:
        raise RuntimeBenchmarkIntegrityError(
            "World commit journal head does not match the episode manifest"
        )

    try:
        # Building the unfiltered join validates every observed request,
        # decision, response disposition, response, and World commit edge.  An
        # empty set is valid when the model made no Platform request.
        evidence.platform_exchanges()
    except RuntimeEvidenceError as exc:
        raise RuntimeBenchmarkIntegrityError(
            "Platform evidence does not form exact CommerceWorld exchanges"
        ) from exc

    try:
        from episode.capability_runtime_authority import (
            runtime_evidence_core_ownership_violations_v2,
        )

        ownership_violations = runtime_evidence_core_ownership_violations_v2(evidence)
    except Exception as exc:
        raise RuntimeBenchmarkIntegrityError(
            "CommerceWorld core ownership could not be verified"
        ) from exc
    if ownership_violations:
        raise RuntimeBenchmarkIntegrityError(
            "CommerceWorld core ownership violation: " + "; ".join(ownership_violations)
        )


def renormalize_capability_checks_v2(
    checks: Sequence[RuntimeRubricCheckV2],
) -> tuple[RuntimeRubricCheckV2, ...]:
    """Renormalize capability checks after environment invariants are removed.

    Relative capability weights are preserved.  The final weight is computed
    from the remainder so floating-point drift cannot make an otherwise valid
    rubric fail the exact sum check.
    """

    rows = tuple(checks)
    if not rows:
        raise ValueError("a benchmark rubric needs at least one capability check")
    total = sum(row.weight for row in rows)
    if total <= 0:
        raise ValueError("capability rubric weight must be positive")
    normalized: list[RuntimeRubricCheckV2] = []
    assigned = 0.0
    for index, row in enumerate(rows):
        weight = 1.0 - assigned if index == len(rows) - 1 else row.weight / total
        normalized.append(
            RuntimeRubricCheckV2(
                name=row.name,
                weight=weight,
                credit=row.credit,
                evidence=dict(row.evidence),
            )
        )
        assigned += weight
    return tuple(normalized)


ChannelFactoryV2 = Callable[[], "InferenceChannel"]
RuntimeTaskScorerV3 = Callable[[RuntimeEvidenceBundleV2], RuntimeTaskScoreV3]


@dataclass(frozen=True)
class RuntimeMutationV2:
    """One deterministic, capability-targeted policy mutation.

    Mutation policies execute through the same Episode path as an ideal policy.
    ``expected_changed_checks`` names the rubric lanes that may lose credit;
    the global preflight rejects full-score and all-zero mutations so failures
    remain discriminating rather than collapsing into a binary score.
    """

    mutation_id: str
    channel: ChannelFactoryV2
    expected_changed_checks: tuple[str, ...]
    mutation_kind: str = "partial_failure"

    def __post_init__(self) -> None:
        if not self.mutation_id:
            raise ValueError("runtime mutation needs an id")
        if not self.expected_changed_checks:
            raise ValueError("runtime mutation needs at least one target rubric check")
        if len(set(self.expected_changed_checks)) != len(self.expected_changed_checks):
            raise ValueError("runtime mutation target checks must be unique")
        if self.mutation_kind not in {
            "partial_failure",
            "security_violation",
        }:
            raise ValueError("runtime mutation kind is not part of the frozen contract")


@dataclass(frozen=True)
class RuntimeTaskBundleV2:
    """One fixed task bound to the real CommerceWorld execution stack."""

    task: TaskDefinitionV2
    scenario: ScenarioSpec
    evaluated_actor_id: str
    ideal_channel: ChannelFactoryV2
    counterpart_channels: Mapping[str, ChannelFactoryV2]
    scorer: RuntimeTaskScorerV3
    semantic_hash: str
    mutations: tuple[RuntimeMutationV2, ...] = ()

    def __post_init__(self) -> None:
        if self.evaluated_actor_id.split(":", 1)[0] != self.task.evaluated_role:
            raise ValueError("evaluated actor role disagrees with task definition")
        if self.evaluated_actor_id in self.counterpart_channels:
            raise ValueError("evaluated actor cannot also be a counterpart")
        if len(self.semantic_hash) != 64:
            raise ValueError("runtime task semantic hash must be SHA-256")


def score_checks(
    task: TaskDefinitionV2,
    checks: Sequence[RuntimeRubricCheckV2],
    *,
    issues: Sequence[str] = (),
    model_safety_violation: bool = False,
    model_privacy_violation: bool = False,
) -> RuntimeTaskScoreV3:
    return RuntimeTaskScoreV3(
        task_id=task.task_id,
        family=task.family.value,
        evaluated_role=task.evaluated_role,
        checks=tuple(checks),
        issues=tuple(issues),
        model_safety_violation=model_safety_violation,
        model_privacy_violation=model_privacy_violation,
    )


__all__ = [
    "COMMERCEWORLD_EPISODE_BACKEND_V2",
    "LinkedPlatformExchangeV2",
    "RUNTIME_TASK_SCORE_SCHEMA_V3",
    "RUNTIME_MODEL_SAFETY_SCORE_CAP_V3",
    "RuntimeEvidenceBundleV2",
    "RuntimeBenchmarkIntegrityError",
    "RuntimeEvidenceError",
    "RuntimeRubricCheckV2",
    "RuntimeMutationV2",
    "RuntimeTaskBundleV2",
    "RuntimeTaskScoreV3",
    "attested_world_catalog_reads_v2",
    "canonical_sha256",
    "record_verified_operation_evidence_calls",
    "renormalize_capability_checks_v2",
    "require_runtime_benchmark_integrity_v2",
    "score_checks",
]
