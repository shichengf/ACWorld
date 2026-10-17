"""Contracts, source manifests, and reports for zero-model environment studies.

A CommerceWorld *environment* study measures whether the environment executes
correctly -- authoritative commits, isolation, idempotency, replay -- rather
than how capable a model is.  It therefore has its own lightweight frozen
contract and its own report shape that deliberately carry no capability score,
no strict-success rate, no model ranking, and no environment reward.

Freezing is real, not nominal.  ``frozen=True`` alone leaves nested ``dict``
values mutable, so both the contract and the report deep-freeze every mapping at
construction (a caller cannot mutate an input after the fact, and nothing can
mutate the object's internals), and ``to_dict`` rebuilds fresh mutable copies.

Freezing the *source revision* is done with a scoped
:class:`SourceManifestV1` rather than a bare git commit/dirty pair: two different
dirty worktrees share the same git metadata, so the manifest records the exact
bytes and SHA-256 of every declared execution-path file.  The contract's
component digests are derived from -- and verifiable against -- that manifest, so
a change to the executing code changes the contract identity and is detected by
:func:`verify_environment_study_contract`.  Paper, ``output/``, and ``tmp/``
drift is out of scope by construction (those paths are never in the manifest);
an explicit ``allowed_drift`` escape hatch exists for the rare declared case.

A paid multi-agent LLM trace uses a different contract and report (it is not
``provider_calls == 0``) and must not be recorded through this schema.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from experiments.contract import GitMetadata

ENVIRONMENT_STUDY_CONTRACT_SCHEMA = "cwe.environment-study-contract.v1"
ENVIRONMENT_STUDY_REPORT_SCHEMA = "cwe.environment-study-report.v1"
SOURCE_MANIFEST_SCHEMA = "cwe.environment-study-source-manifest.v1"
NETWORK_GUARD_SCHEMA = "cwe.environment-study-network-guard.v1"
ARTIFACT_INDEX_SCHEMA = "cwe.environment-study-artifact-index.v1"

# Local hosts a network-disabled study may still reach (the in-memory ASGI
# HTTP/VCP path makes no real connect at all, so these are only a convenience for
# a hypothetical loopback-server variant).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"})

# The canonical persisted-artifact order: two pre-run artifacts, then the
# post-run artifacts, with the artifact index written last (indexing the five
# that precede it).
ENVIRONMENT_STUDY_ARTIFACT_ORDER = (
    "source-manifest.json",
    "contract.json",
    "hashes-only-decision-log.json",
    "network-guard.json",
    "report.json",
)
ARTIFACT_INDEX_NAME = "artifact-index.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_VALUES = frozenset({"buyer", "merchant"})
_REQUIRED_COMPONENT_DIGESTS = frozenset(
    {
        "route_registry",
        "agent_bridge",
        "platform_policy",
        "world_schema",
        "business_decision_contract",
    }
)

# Declared execution-path closure.  This is the FULL src import closure a 1x1
# smoke study loads (discovered empirically and enforced by the post-run
# loaded-source audit, so a forgotten module invalidates the study rather than
# passing silently), plus the non-Python execution inputs the run reads: the
# merchant/buyer skill cards and the business system prompt.  The five semantic
# components the contract requires (route_registry, agent_bridge, platform_policy,
# world_schema, business_decision_contract) are kept explicit; the ``*_closure``
# components freeze each package's loaded modules.  It deliberately excludes paper
# sources, ``output/``, and ``tmp/`` (never on this path).  ``evals`` modules are
# imported transitively by the runner and frozen for integrity; the study never
# invokes an oracle scorer.
DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "route_registry": (
            "src/protocol/actions.py",
            "src/runtime/router.py",
        ),
        "agent_bridge": (
            "src/agents/agent_phase.py",
            "src/agents/business_decision.py",
        ),
        "platform_policy": (
            "src/agents/platform.py",
        ),
        "world_schema": (
            "src/world/state.py",
            "src/world/tables.py",
            "src/world/service.py",
        ),
        "business_decision_contract": (
            "src/agents/business_decision.py",
        ),
        "agents_closure": (
            "src/agents/__init__.py",
            "src/agents/after_sales_turn_authority.py",
            "src/agents/agent_phase.py",
            "src/agents/agent_routes.py",
            "src/agents/api.py",
            "src/agents/base.py",
            "src/agents/business_decision.py",
            "src/agents/business_intent_schema_policy.py",
            "src/agents/business_phase_adapter.py",
            "src/agents/business_prompt.py",
            "src/agents/buyer.py",
            "src/agents/commerce_grounding.py",
            "src/agents/decision_errors.py",
            "src/agents/domain_phase_adapters.py",
            "src/agents/errors.py",
            "src/agents/evidence_policy.py",
            "src/agents/governance_turn_authority.py",
            "src/agents/inference.py",
            "src/agents/interfaces.py",
            "src/agents/memory.py",
            "src/agents/merchant.py",
            "src/agents/merchant_data_csv.py",
            "src/agents/merchant_private_state.py",
            "src/agents/merchant_tools.py",
            "src/agents/merchant_tools_state.py",
            "src/agents/merchant_tools_types.py",
            "src/agents/negotiation.py",
            "src/agents/negotiation_turn_authority.py",
            "src/agents/platform.py",
            "src/agents/platform_api.py",
            "src/agents/platform_errors.py",
            "src/agents/protocol_event_turn_authority.py",
            "src/agents/provider_boundary_policy.py",
            "src/agents/public_task_execution.py",
            "src/agents/ranked_offer_turn_authority.py",
            "src/agents/remote_world.py",
            "src/agents/skill_selector.py",
            "src/agents/skill_selector_buyer.py",
            "src/agents/skill_selector_merchant/__init__.py",
            "src/agents/skill_selector_merchant/_shared.py",
            "src/agents/skill_selector_merchant/after_sales_lifecycle.py",
            "src/agents/skill_selector_merchant/aging_markdown.py",
            "src/agents/skill_selector_merchant/cart_quote_handle.py",
            "src/agents/skill_selector_merchant/catalog_serve.py",
            "src/agents/skill_selector_merchant/claim_truthfulness.py",
            "src/agents/skill_selector_merchant/demand_driven_markup.py",
            "src/agents/skill_selector_merchant/dispute_defense.py",
            "src/agents/skill_selector_merchant/fulfillment.py",
            "src/agents/skill_selector_merchant/inbound_restock.py",
            "src/agents/skill_selector_merchant/inquiry_handle.py",
            "src/agents/skill_selector_merchant/listing_claim_manage.py",
            "src/agents/skill_selector_merchant/listing_publish.py",
            "src/agents/skill_selector_merchant/market_governance.py",
            "src/agents/skill_selector_merchant/order_cancel.py",
            "src/agents/skill_selector_merchant/order_intake.py",
            "src/agents/skill_selector_merchant/peer_pricing.py",
            "src/agents/skill_selector_merchant/price_discovery.py",
            "src/agents/skill_selector_merchant/pricing_negotiate.py",
            "src/agents/skill_selector_merchant/private_utility_guard.py",
            "src/agents/skill_selector_merchant/protocol_event_handle.py",
            "src/agents/skill_selector_merchant/reputation_aware_pricing.py",
            "src/agents/skill_selector_merchant/restock_signal.py",
            "src/agents/skill_selector_merchant/return_adjudicate.py",
            "src/agents/skill_selector_merchant/stockout_aware_pricing.py",
            "src/agents/skill_selector_merchant/supply_logistics.py",
            "src/agents/skills.py",
            "src/agents/turn.py",
            "src/agents/types.py",
            "src/agents/world_client.py",
        ),
        "episode_closure": (
            "src/episode/__init__.py",
            "src/episode/actor_evidence.py",
            "src/episode/artifacts.py",
            "src/episode/benchmark.py",
            "src/episode/capability_benchmark.py",
            "src/episode/capability_materializer.py",
            "src/episode/capability_runtime.py",
            "src/episode/capability_runtime_fixture.py",
            "src/episode/capability_runtime_t1.py",
            "src/episode/capability_runtime_t1_content.py",
            "src/episode/catalog_provenance.py",
            "src/episode/seed_catalog.py",
            "src/episode/errors.py",
            "src/episode/evidence.py",
            "src/episode/extension_runtime.py",
            "src/episode/extensions.py",
            "src/episode/http_launcher.py",
            "src/episode/interfaces.py",
            "src/episode/replay.py",
            "src/episode/runner.py",
            "src/episode/scale.py",
            "src/episode/scenario.py",
            "src/episode/termination.py",
            "src/episode/types.py",
        ),
        "evals_closure": (
            "src/evals/__init__.py",
            "src/evals/errors.py",
            "src/evals/interfaces.py",
            "src/evals/market_metrics.py",
            "src/evals/metrics.py",
            "src/evals/oracle.py",
            "src/evals/probes/__init__.py",
            "src/evals/probes/order.py",
            "src/evals/serialize.py",
            "src/evals/types.py",
        ),
        "experiments_closure": (
            "src/experiments/__init__.py",
            "src/experiments/catalog_feasibility.py",
            "src/experiments/contract.py",
            "src/experiments/data_environment_study.py",
            "src/experiments/e1_persistence_study.py",
            "src/experiments/environment_integrity_study.py",
            "src/experiments/environment_smoke.py",
            "src/experiments/environment_study.py",
            "src/experiments/multiagent_openrouter.py",
            "src/experiments/multiagent_preflight.py",
            "src/experiments/real_catalog_multiagent.py",
            "src/experiments/rq2_extensions.py",
            "src/experiments/scripted_channel.py",
        ),
        "cli_closure": (
            "src/cli/__init__.py",
            "src/cli/environment_study.py",
            "src/cli/scenario.py",
        ),
        "extension_examples": (
            "src/cwe_examples/__init__.py",
            "src/cwe_examples/flash_restock_event.py",
            "src/cwe_examples/pet_supplies_domain.py",
            "src/cwe_examples/round_robin_ranking.py",
        ),
        "protocol_closure": (
            "src/protocol/__init__.py",
            "src/protocol/actions.py",
            "src/protocol/actor_result.py",
            "src/protocol/actor_terminal.py",
            "src/protocol/after_sales.py",
            "src/protocol/cart.py",
            "src/protocol/cart_quote_request.py",
            "src/protocol/cart_quote_state.py",
            "src/protocol/envelope.py",
            "src/protocol/envelopes.py",
            "src/protocol/errors.py",
            "src/protocol/event_receipts.py",
            "src/protocol/evidence_records.py",
            "src/protocol/interfaces.py",
            "src/protocol/listing_claims.py",
            "src/protocol/market_governance.py",
            "src/protocol/matching.py",
            "src/protocol/negotiation_state.py",
            "src/protocol/negotiation_turn_projection.py",
            "src/protocol/pricing.py",
            "src/protocol/pricing_policy.py",
            "src/protocol/remediation_audit.py",
            "src/protocol/schemas.py",
            "src/protocol/supply_authority.py",
            "src/protocol/tool_errors.py",
        ),
        "runtime_closure": (
            "src/runtime/__init__.py",
            "src/runtime/actor_context.py",
            "src/runtime/actor_context_artifact.py",
            "src/runtime/actor_evidence.py",
            "src/runtime/actor_evidence_verifier.py",
            "src/runtime/after_sales_evidence.py",
            "src/runtime/audit.py",
            "src/runtime/authority_operation_evidence.py",
            "src/runtime/bus.py",
            "src/runtime/cart_evidence.py",
            "src/runtime/errors.py",
            "src/runtime/evidence.py",
            "src/runtime/exact_join.py",
            "src/runtime/interfaces.py",
            "src/runtime/market_governance_evidence.py",
            "src/runtime/match_certificate_evidence.py",
            "src/runtime/privacy.py",
            "src/runtime/router.py",
            "src/runtime/scheduler.py",
            "src/runtime/supply_fulfillment_evidence.py",
            "src/runtime/trace.py",
            "src/runtime/tracker_evidence.py",
            "src/runtime/transport.py",
            "src/runtime/turn_failure.py",
            "src/runtime/types.py",
            "src/runtime/world_state_replay.py",
        ),
        "world_closure": (
            "src/world/__init__.py",
            "src/world/after_sales_authority_projection.py",
            "src/world/after_sales_core.py",
            "src/world/after_sales_persistence.py",
            "src/world/after_sales_service.py",
            "src/world/api.py",
            "src/world/cart_pricing.py",
            "src/world/catalog_mutations.py",
            "src/world/client.py",
            "src/world/errors.py",
            "src/world/evidence_contracts.py",
            "src/world/interfaces.py",
            "src/world/market_governance_core.py",
            "src/world/market_governance_persistence.py",
            "src/world/market_governance_service.py",
            "src/world/market_governance_world.py",
            "src/world/match_authorizations.py",
            "src/world/negotiations.py",
            "src/world/payment_fulfillment.py",
            "src/world/reset.py",
            "src/world/service.py",
            "src/world/state.py",
            "src/world/store.py",
            "src/world/tables.py",
            "src/world/tools.py",
            "src/world/transactions.py",
            "src/world/types.py",
        ),
        "skill_cards": (
            "skills/buyer/after-sales-lifecycle/SKILL.md",
            "skills/buyer/authenticated-review/SKILL.md",
            "skills/buyer/cart-checkout/SKILL.md",
            "skills/buyer/discovery-search/SKILL.md",
            "skills/buyer/mandate-parsing/SKILL.md",
            "skills/buyer/marketplace-message-safety/SKILL.md",
            "skills/buyer/negotiation/SKILL.md",
            "skills/buyer/protocol-event-handling/SKILL.md",
            "skills/buyer/purchase-confirmation/SKILL.md",
            "skills/buyer/return-refund/SKILL.md",
            "skills/buyer/supply-fulfillment/SKILL.md",
            "skills/merchant/after-sales-lifecycle/SKILL.md",
            "skills/merchant/aging-markdown/SKILL.md",
            "skills/merchant/cart-quote-handle/SKILL.md",
            "skills/merchant/catalog-serve/SKILL.md",
            "skills/merchant/claim-truthfulness/SKILL.md",
            "skills/merchant/demand-driven-markup/SKILL.md",
            "skills/merchant/dispute-defense/SKILL.md",
            "skills/merchant/fulfillment/SKILL.md",
            "skills/merchant/inbound-restock/SKILL.md",
            "skills/merchant/inquiry-handle/SKILL.md",
            "skills/merchant/listing-claim-manage/SKILL.md",
            "skills/merchant/listing-publish/SKILL.md",
            "skills/merchant/market-governance/SKILL.md",
            "skills/merchant/order-cancel/SKILL.md",
            "skills/merchant/order-intake/SKILL.md",
            "skills/merchant/peer-pricing/SKILL.md",
            "skills/merchant/price-discovery/SKILL.md",
            "skills/merchant/pricing-negotiate/SKILL.md",
            "skills/merchant/private-utility-guard/SKILL.md",
            "skills/merchant/protocol-event-handle/SKILL.md",
            "skills/merchant/reputation-aware-pricing/SKILL.md",
            "skills/merchant/restock-signal/SKILL.md",
            "skills/merchant/return-adjudicate/SKILL.md",
            "skills/merchant/stockout-aware-pricing/SKILL.md",
            "skills/merchant/supply-logistics/SKILL.md",
        ),
        "agent_prompts": (
            "src/agents/prompts/business.system.md",
        ),
        "real_catalog_inputs": (
            "data/catalog_provenance.json",
            "data/raw_data/allbirds.csv",
            "data/raw_data/brooklinen.csv",
            "data/raw_data/death_wish_coffee.csv",
            "data/raw_data/kylie_cosmetics.csv",
            "data/raw_data/manduka.csv",
            "data/raw_data/meowant.csv",
        ),
    }
)
# Substring markers that identify a model-score/ranking/reward field.  Report
# summary keys are additionally required to be safe snake_case identifiers, so
# this is an identifier allowlist backed by a marker denylist rather than an
# ever-growing list of exact field names.
_SCORE_MARKERS = (
    "score",
    "reward",
    "rank",
    "capab",
    "leaderboard",
    "strict_success",
    "win_rate",
    "elo",
    "overall",
)
_SUMMARY_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class EnvironmentStudyError(ValueError):
    """An environment-study contract, manifest, or report is invalid or unsafe."""


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EnvironmentStudyError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EnvironmentStudyError(f"{label} must be an integer")
    return int(value)


def _require_str(value: Any, *, label: str) -> str:
    """Require an actual string; never coerce (a coerced value hides tampering)."""

    if not isinstance(value, str):
        raise EnvironmentStudyError(f"{label} must be a string")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    """Require an actual bool; ``"false"`` and ``0``/``1`` must not become True."""

    if not isinstance(value, bool):
        raise EnvironmentStudyError(f"{label} must be a boolean")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EnvironmentStudyError(f"{label} must be a list")
    return value


def _require_exact_keys(
    raw: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    label: str,
) -> None:
    """Reject missing and unknown keys, so injected fields cannot slip through."""

    keys = {str(k) for k in raw}
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - keys
    if missing:
        raise EnvironmentStudyError(f"{label} is missing required keys: {sorted(missing)}")
    unknown = keys - allowed
    if unknown:
        raise EnvironmentStudyError(f"{label} has unknown keys: {sorted(unknown)}")


def _resolve_contained(root: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` and require it to stay inside ``root``.

    Rejects absolute paths, ``.``/``..`` segments, and -- after following
    symlinks via ``resolve()`` -- any target outside the repository root.
    """

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise EnvironmentStudyError("source path must be a non-empty relative path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise EnvironmentStudyError(
            f"source path must be repo-relative without '.'/'..': {relative_path}"
        )
    root_resolved = root.resolve()
    target = (root_resolved / relative_path).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise EnvironmentStudyError(
            f"source path escapes the repository root: {relative_path}"
        ) from exc
    return target


def _deep_freeze(value: Any) -> Any:
    """Return an immutable deep copy: mappings become read-only, lists tuples."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Rebuild a fresh mutable structure from a deep-frozen value."""

    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def capture_git_metadata(repo_root: str | Path) -> GitMetadata:
    """Return the current commit and dirty flag as coarse provenance.

    This is provenance only.  Two different dirty worktrees produce the same
    value, so it never replaces the :class:`SourceManifestV1` byte-level freeze.
    """

    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EnvironmentStudyError(f"cannot inspect git repository at {root}") from exc
    return GitMetadata.from_dict({"commit": commit, "dirty": bool(status.strip())})


# --- Source manifest -------------------------------------------------


@dataclass(frozen=True)
class FrozenSourceFileV1:
    """The frozen byte length and SHA-256 of one repo-relative source file."""

    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise EnvironmentStudyError("source file path must be a non-empty relative path")
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise EnvironmentStudyError(
                f"source file path must be repo-relative without '.'/'..': {self.path}"
            )
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise EnvironmentStudyError("source file bytes must be a non-negative integer")
        _require_digest(self.sha256, label=f"source file sha256[{self.path}]")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FrozenSourceFileV1":
        if not isinstance(raw, Mapping):
            raise EnvironmentStudyError("source file entry must be an object")
        _require_exact_keys(raw, required=("path", "bytes", "sha256"), label="source file entry")
        return cls(
            path=_require_str(raw.get("path"), label="source file path"),
            bytes=_require_int(raw.get("bytes"), label="source file bytes"),
            sha256=_require_str(raw.get("sha256"), label="source file sha256"),
        )


@dataclass(frozen=True)
class SourceManifestV1:
    """The frozen bytes/SHA-256 of every declared execution-path file."""

    components: Mapping[str, tuple[FrozenSourceFileV1, ...]]
    schema_version: str = SOURCE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA:
            raise EnvironmentStudyError("unsupported source manifest schema")
        if not isinstance(self.components, Mapping) or not self.components:
            raise EnvironmentStudyError("source manifest requires at least one component")
        frozen: dict[str, tuple[FrozenSourceFileV1, ...]] = {}
        for name, files in self.components.items():
            if not isinstance(name, str) or not name.strip():
                raise EnvironmentStudyError("source manifest component name must be non-empty")
            if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
                raise EnvironmentStudyError(f"component {name!r} needs at least one source file")
            rows = tuple(files)
            if any(not isinstance(row, FrozenSourceFileV1) for row in rows):
                raise EnvironmentStudyError(f"component {name!r} files must be FrozenSourceFileV1")
            paths = [row.path for row in rows]
            if len(paths) != len(set(paths)):
                raise EnvironmentStudyError(f"component {name!r} has duplicate source paths")
            frozen[name] = rows
        object.__setattr__(self, "components", MappingProxyType(frozen))

    def component_digest(self, name: str) -> str:
        """Return a stable digest over one component's sorted (path, sha256)."""

        rows = self.components.get(name)
        if not rows:
            raise EnvironmentStudyError(f"source manifest has no component {name!r}")
        payload = sorted((row.path, row.sha256) for row in rows)
        return _canonical_sha256({"component": name, "files": payload})

    def component_digests(self) -> dict[str, str]:
        """Return every component's derived digest."""

        return {name: self.component_digest(name) for name in self.components}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "components": {
                name: [row.to_dict() for row in rows]
                for name, rows in sorted(self.components.items())
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceManifestV1":
        if not isinstance(raw, Mapping):
            raise EnvironmentStudyError("source manifest root must be an object")
        _require_exact_keys(
            raw, required=("components",), optional=("schema_version",), label="source manifest"
        )
        components = raw.get("components", {})
        if not isinstance(components, Mapping):
            raise EnvironmentStudyError("source manifest components must be an object")
        parsed: dict[str, tuple[FrozenSourceFileV1, ...]] = {}
        for name, files in components.items():
            parsed[_require_str(name, label="component name")] = tuple(
                FrozenSourceFileV1.from_dict(row) for row in _require_list(files, label="component")
            )
        return cls(
            components=parsed,
            schema_version=_require_str(
                raw.get("schema_version", SOURCE_MANIFEST_SCHEMA), label="schema_version"
            ),
        )


def _hash_file(root: Path, relative_path: str) -> FrozenSourceFileV1:
    target = _resolve_contained(root, relative_path)
    if not target.is_file():
        raise EnvironmentStudyError(f"manifest source file is missing: {relative_path}")
    payload = target.read_bytes()
    return FrozenSourceFileV1(
        path=relative_path,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_source_manifest(
    repo_root: str | Path,
    component_sources: Mapping[str, Sequence[str]] = DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES,
) -> SourceManifestV1:
    """Freeze bytes/SHA-256 of each declared component file, read from disk."""

    root = Path(repo_root).resolve(strict=True)
    components: dict[str, tuple[FrozenSourceFileV1, ...]] = {}
    for name, paths in component_sources.items():
        components[str(name)] = tuple(_hash_file(root, str(path)) for path in paths)
    return SourceManifestV1(components=components)


def verify_source_manifest(
    manifest: SourceManifestV1,
    repo_root: str | Path,
) -> tuple[str, ...]:
    """Recompute every manifest file from disk; return the list of issues.

    Every declared file is checked -- there is no drift exemption, because a
    file in the execution manifest that changed means the executing code
    changed.  A path that escapes the repository root (including via a symlink)
    is itself reported as an issue.  An empty result means the on-disk bytes
    still match the frozen manifest exactly.
    """

    root = Path(repo_root).resolve(strict=True)
    issues: list[str] = []
    for name, rows in sorted(manifest.components.items()):
        for row in rows:
            try:
                target = _resolve_contained(root, row.path)
            except EnvironmentStudyError as exc:
                issues.append(f"{name}: {exc}")
                continue
            if not target.is_file():
                issues.append(f"{name}: missing source file {row.path}")
                continue
            payload = target.read_bytes()
            if len(payload) != row.bytes or hashlib.sha256(payload).hexdigest() != row.sha256:
                issues.append(f"{name}: source file changed on disk: {row.path}")
    return tuple(issues)


# --- Loaded-source audit ---------------------------------------------
#
# A hand-authored manifest can silently omit an execution-path module.  The
# audit closes that gap: after a run, every ``src`` module actually imported must
# already be frozen in the manifest, so a forgotten file invalidates the study
# instead of a human being trusted to have listed the complete closure.  Only
# Python modules can be discovered through ``sys.modules``; non-Python execution
# inputs (fixtures, policy configs, skill cards, data descriptions) are frozen by
# explicit path and covered by :func:`verify_source_manifest`.


def iter_loaded_module_files() -> tuple[str, ...]:
    """Return the ``__file__`` of every currently-imported module.

    ``sys.modules`` is process-wide, so callers scope the result to the
    repository -- and typically to a single run's newly-imported delta -- before
    auditing it against the frozen manifest.
    """

    files: list[str] = []
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if isinstance(path, str) and path:
            files.append(path)
    return tuple(files)


def repo_source_paths(files: Iterable[str], repo_root: str | Path) -> frozenset[str]:
    """Map module ``__file__`` paths to repo-relative ``src/*.py`` POSIX paths.

    Anything outside ``<repo_root>/src`` (the standard library, site-packages,
    and the test tree) is dropped: the manifest freezes the project's own ``src``
    execution closure.  Paths are resolved (following symlinks) before being made
    repo-relative, so an aliased path cannot masquerade as a different file.
    """

    root = Path(repo_root).resolve()
    src_root = (root / "src").resolve()
    out: set[str] = set()
    for raw in files:
        try:
            resolved = Path(raw).resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        if resolved.suffix != ".py":
            continue
        try:
            resolved.relative_to(src_root)
        except ValueError:
            continue
        out.add(resolved.relative_to(root).as_posix())
    return frozenset(out)


def manifest_source_paths(manifest: SourceManifestV1) -> frozenset[str]:
    """Return every ``.py`` path the manifest freezes, across all components."""

    return frozenset(
        row.path
        for rows in manifest.components.values()
        for row in rows
        if row.path.endswith(".py")
    )


def manifest_non_python_paths(manifest: SourceManifestV1) -> frozenset[str]:
    """Return every non-``.py`` path the manifest freezes (skills, prompts, data).

    These execution inputs cannot be discovered through ``sys.modules``; they are
    frozen by explicit path and their completeness is proven by the read audit
    (:func:`record_repo_reads`) rather than the loaded-module audit.
    """

    return frozenset(
        row.path
        for rows in manifest.components.values()
        for row in rows
        if not row.path.endswith(".py")
    )


@contextmanager
def record_repo_reads(
    repo_root: str | Path, *, ignore_dirs: Sequence[str | Path] = ()
) -> Iterator[set[str]]:
    """Record repo-relative NON-Python files read while the block is active.

    An audit-time instrument (not for production runs): it wraps the file-read
    entry points to record which repository-tracked non-``.py`` inputs -- skill
    cards, prompts, configs, data -- a run actually reads, then restores them on
    exit.  Combined with the frozen manifest this proves the non-Python execution
    inputs are completely declared, the way the loaded-module audit does for
    Python modules.  Reads outside ``repo_root`` (e.g. an episode's temp output
    dir, passed via ``ignore_dirs``) and ``.py`` files are ignored.  Recording
    never raises, so it cannot break the underlying read.
    """

    root = Path(repo_root).resolve()
    ignores = tuple(Path(d).resolve() for d in ignore_dirs)
    seen: set[str] = set()

    def record(target: Any) -> None:
        try:
            candidate = Path(target)
            if not candidate.is_absolute():
                candidate = root / candidate
            resolved = candidate.resolve()
            if resolved.suffix == ".py":
                return
            rel = resolved.relative_to(root)
            for ignore in ignores:
                if resolved == ignore or ignore in resolved.parents:
                    return
            seen.add(rel.as_posix())
        except Exception:  # noqa: BLE001 - recording must never break a real read
            return

    real_open = builtins.open
    real_os_open = os.open
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes
    real_path_open = Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        record(file)
        return real_open(file, *args, **kwargs)

    def guarded_os_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        record(path)
        return real_os_open(path, *args, **kwargs)

    def guarded_read_text(self: Path, *args: Any, **kwargs: Any) -> Any:
        record(self)
        return real_read_text(self, *args, **kwargs)

    def guarded_read_bytes(self: Path, *args: Any, **kwargs: Any) -> Any:
        record(self)
        return real_read_bytes(self, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        record(self)
        return real_path_open(self, *args, **kwargs)

    builtins.open = guarded_open
    os.open = guarded_os_open
    Path.read_text = guarded_read_text  # type: ignore[method-assign]
    Path.read_bytes = guarded_read_bytes  # type: ignore[method-assign]
    Path.open = guarded_path_open  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        builtins.open = real_open
        os.open = real_os_open
        Path.read_text = real_read_text  # type: ignore[method-assign]
        Path.read_bytes = real_read_bytes  # type: ignore[method-assign]
        Path.open = real_path_open  # type: ignore[method-assign]


def unlisted_loaded_sources(
    manifest: SourceManifestV1, loaded_source_paths: Iterable[str]
) -> tuple[str, ...]:
    """Return loaded ``src`` modules the frozen manifest does not declare.

    An empty result proves the manifest covers every executed module in the
    given set; a non-empty result names each undeclared module (sorted), so a
    forgotten execution file makes the study invalid rather than passing
    silently.
    """

    declared = manifest_source_paths(manifest)
    return tuple(sorted(set(loaded_source_paths) - declared))


# --- Contract --------------------------------------------------------


@dataclass(frozen=True)
class EnvironmentStudyContractV1:
    """Frozen inputs for one zero-model environment study."""

    git: GitMetadata
    scenario_digest: str
    actor_policy_digests: tuple[tuple[str, str], ...]
    component_digests: Mapping[str, str]
    data_provenance_digest: str
    backend: str
    transport: str
    seed: int
    invariants: tuple[str, ...]
    contract_id: str
    provider_calls: int = 0
    schema_version: str = ENVIRONMENT_STUDY_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ENVIRONMENT_STUDY_CONTRACT_SCHEMA:
            raise EnvironmentStudyError("unsupported environment study contract schema")
        if isinstance(self.provider_calls, bool) or self.provider_calls != 0:
            raise EnvironmentStudyError("environment study contract requires provider_calls == 0")
        _require_digest(self.scenario_digest, label="scenario_digest")
        _require_digest(self.data_provenance_digest, label="data_provenance_digest")

        if not self.actor_policy_digests:
            raise EnvironmentStudyError("environment study needs at least one actor policy")
        actor_ids = [actor_id for actor_id, _ in self.actor_policy_digests]
        if len(actor_ids) != len(set(actor_ids)):
            raise EnvironmentStudyError("environment study actor ids must be unique")
        if list(actor_ids) != sorted(actor_ids):
            raise EnvironmentStudyError("environment study actor policies must be sorted by id")
        for actor_id, digest in self.actor_policy_digests:
            if not isinstance(actor_id, str) or not actor_id.strip():
                raise EnvironmentStudyError("environment study actor id must be non-empty")
            _require_digest(digest, label=f"actor_policy_digest[{actor_id}]")

        if not isinstance(self.component_digests, Mapping):
            raise EnvironmentStudyError("component_digests must be an object")
        missing = _REQUIRED_COMPONENT_DIGESTS - set(self.component_digests)
        if missing:
            raise EnvironmentStudyError(
                "environment study contract is missing component digests: "
                + ", ".join(sorted(missing))
            )
        for name, digest in self.component_digests.items():
            _require_digest(digest, label=f"component_digest[{name}]")

        for label, value in (("backend", self.backend), ("transport", self.transport)):
            if not isinstance(value, str) or not value.strip():
                raise EnvironmentStudyError(f"environment study {label} must be non-empty text")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EnvironmentStudyError("environment study seed must be an integer")
        if not self.invariants:
            raise EnvironmentStudyError("environment study must declare at least one invariant")
        if len(self.invariants) != len(set(self.invariants)):
            raise EnvironmentStudyError("environment study invariants must be unique")

        expected = compute_environment_study_contract_id(
            git=self.git,
            scenario_digest=self.scenario_digest,
            actor_policy_digests=self.actor_policy_digests,
            component_digests=self.component_digests,
            data_provenance_digest=self.data_provenance_digest,
            backend=self.backend,
            transport=self.transport,
            seed=self.seed,
            invariants=self.invariants,
        )
        if self.contract_id != expected:
            raise EnvironmentStudyError("contract_id does not match the stable contract payload")

        # Deep-freeze the only mutable input so neither the caller's original
        # dict nor the object's internals can be changed after construction.
        object.__setattr__(self, "component_digests", _deep_freeze(dict(self.component_digests)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "git": self.git.to_dict(),
            "scenario_digest": self.scenario_digest,
            "actor_policy_digests": [list(pair) for pair in self.actor_policy_digests],
            "component_digests": dict(sorted(_thaw(self.component_digests).items())),
            "data_provenance_digest": self.data_provenance_digest,
            "backend": self.backend,
            "transport": self.transport,
            "seed": self.seed,
            "invariants": list(self.invariants),
            "provider_calls": self.provider_calls,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EnvironmentStudyContractV1":
        if not isinstance(raw, Mapping):
            raise EnvironmentStudyError("environment study contract root must be an object")
        _require_exact_keys(
            raw,
            required=(
                "contract_id",
                "git",
                "scenario_digest",
                "actor_policy_digests",
                "component_digests",
                "data_provenance_digest",
                "backend",
                "transport",
                "seed",
                "invariants",
            ),
            optional=("schema_version", "provider_calls"),
            label="environment study contract",
        )
        git_raw = raw.get("git")
        if not isinstance(git_raw, Mapping):
            raise EnvironmentStudyError("contract git must be an object")
        actor_rows = _require_list(raw.get("actor_policy_digests"), label="actor_policy_digests")
        actor_policy_digests: list[tuple[str, str]] = []
        for row in actor_rows:
            if not isinstance(row, list) or len(row) != 2:
                raise EnvironmentStudyError("actor_policy_digests rows must be [actor_id, digest]")
            actor_policy_digests.append(
                (
                    _require_str(row[0], label="actor id"),
                    _require_str(row[1], label="actor policy digest"),
                )
            )
        component = raw.get("component_digests")
        if not isinstance(component, Mapping):
            raise EnvironmentStudyError("component_digests must be an object")
        invariants = _require_list(raw.get("invariants"), label="invariants")
        return cls(
            git=GitMetadata.from_dict(git_raw),
            scenario_digest=_require_str(raw.get("scenario_digest"), label="scenario_digest"),
            actor_policy_digests=tuple(actor_policy_digests),
            component_digests={
                _require_str(k, label="component name"): _require_str(v, label="component digest")
                for k, v in component.items()
            },
            data_provenance_digest=_require_str(
                raw.get("data_provenance_digest"), label="data_provenance_digest"
            ),
            backend=_require_str(raw.get("backend"), label="backend"),
            transport=_require_str(raw.get("transport"), label="transport"),
            seed=_require_int(raw.get("seed"), label="seed"),
            invariants=tuple(_require_str(name, label="invariant") for name in invariants),
            contract_id=_require_str(raw.get("contract_id"), label="contract_id"),
            provider_calls=_require_int(raw.get("provider_calls", 0), label="provider_calls"),
            schema_version=_require_str(
                raw.get("schema_version", ENVIRONMENT_STUDY_CONTRACT_SCHEMA),
                label="schema_version",
            ),
        )


def compute_environment_study_contract_id(
    *,
    git: GitMetadata,
    scenario_digest: str,
    actor_policy_digests: Sequence[tuple[str, str]],
    component_digests: Mapping[str, str],
    data_provenance_digest: str,
    backend: str,
    transport: str,
    seed: int,
    invariants: Sequence[str],
) -> str:
    """Return the stable contract ID (host- and timestamp-independent)."""

    payload = {
        "schema_version": ENVIRONMENT_STUDY_CONTRACT_SCHEMA,
        "git": git.to_dict(),
        "scenario_digest": scenario_digest,
        "actor_policy_digests": [list(pair) for pair in sorted(actor_policy_digests)],
        "component_digests": dict(sorted(component_digests.items())),
        "data_provenance_digest": data_provenance_digest,
        "backend": backend,
        "transport": transport,
        "seed": seed,
        "invariants": list(invariants),
        "provider_calls": 0,
    }
    return _canonical_sha256(payload)


def build_environment_study_contract(
    *,
    git: GitMetadata,
    scenario_digest: str,
    actor_policy_digests: Sequence[tuple[str, str]],
    component_digests: Mapping[str, str],
    data_provenance_digest: str,
    backend: str,
    transport: str,
    seed: int,
    invariants: Sequence[str],
) -> EnvironmentStudyContractV1:
    """Assemble and validate a contract, computing its stable ID."""

    ordered_actors = tuple(sorted((str(a), str(d)) for a, d in actor_policy_digests))
    contract_id = compute_environment_study_contract_id(
        git=git,
        scenario_digest=scenario_digest,
        actor_policy_digests=ordered_actors,
        component_digests=component_digests,
        data_provenance_digest=data_provenance_digest,
        backend=backend,
        transport=transport,
        seed=seed,
        invariants=invariants,
    )
    return EnvironmentStudyContractV1(
        git=git,
        scenario_digest=scenario_digest,
        actor_policy_digests=ordered_actors,
        component_digests=dict(component_digests),
        data_provenance_digest=data_provenance_digest,
        backend=backend,
        transport=transport,
        seed=seed,
        invariants=tuple(invariants),
        contract_id=contract_id,
    )


def verify_environment_study_contract(
    contract: EnvironmentStudyContractV1,
    manifest: SourceManifestV1,
    repo_root: str | Path,
) -> None:
    """Require the on-disk code, the manifest, and the contract to still agree.

    Chain of custody: the on-disk bytes must match the frozen manifest, and the
    manifest's derived component digests must equal the contract's.  Any break
    raises :class:`EnvironmentStudyError`.
    """

    issues = list(verify_source_manifest(manifest, repo_root))
    manifest_digests = manifest.component_digests()
    contract_digests = dict(contract.component_digests)
    if manifest_digests != contract_digests:
        issues.append("contract component digests do not match the source manifest")
    if issues:
        raise EnvironmentStudyError(
            "environment study contract verification failed: " + "; ".join(issues)
        )


# --- Report ----------------------------------------------------------


@dataclass(frozen=True)
class InvariantResultV1:
    """One environment invariant outcome with its evidence references."""

    name: str
    passed: bool
    evidence_refs: tuple[str, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EnvironmentStudyError("invariant name must be non-empty")
        if not isinstance(self.passed, bool):
            raise EnvironmentStudyError("invariant passed must be a boolean")
        if self.passed and self.failure_reason is not None:
            raise EnvironmentStudyError("a passing invariant must not carry a failure reason")
        if not self.passed and not (self.failure_reason and self.failure_reason.strip()):
            raise EnvironmentStudyError("a failing invariant must record a failure reason")
        object.__setattr__(self, "evidence_refs", tuple(str(ref) for ref in self.evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence_refs": list(self.evidence_refs),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "InvariantResultV1":
        if not isinstance(raw, Mapping):
            raise EnvironmentStudyError("invariant result must be an object")
        _require_exact_keys(
            raw,
            required=("name", "passed"),
            optional=("evidence_refs", "failure_reason"),
            label="invariant result",
        )
        refs = _require_list(raw.get("evidence_refs", []), label="invariant evidence_refs")
        reason = raw.get("failure_reason")
        return cls(
            name=_require_str(raw.get("name"), label="invariant name"),
            passed=_require_bool(raw.get("passed"), label="invariant passed"),
            evidence_refs=tuple(_require_str(ref, label="evidence ref") for ref in refs),
            failure_reason=None if reason is None else _require_str(reason, label="failure_reason"),
        )


_ACTOR_COVERAGE_FIELDS = (
    "actor_id",
    "role",
    # Distinct sources with exact meanings -- deliberately NOT interchangeable,
    # and framework continuations are never counted as business decisions:
    "scripted_business_decisions",  # scripted-channel records emitted by the actor
    "verified_business_decisions",  # model/scripted business choices exactly joined to an envelope
    "complete_tracker_records",  # Agent Tracker complete records (incl. framework continuations)
    "accepted_platform_operations",  # accepted Platform exchanges for this actor
    "causally_linked_world_commits",  # World commit ids bound to this actor by the exact join
)


@dataclass(frozen=True)
class ActorCoverageV1:
    """Per-actor coverage with exact, non-interchangeable provenance.

    ``scripted_business_decisions`` is the scripted channel's own record count;
    ``verified_business_decisions`` is the number of business choices exactly
    joined to an audited envelope (framework continuations excluded);
    ``complete_tracker_records`` is the Agent Tracker's complete-record count
    (which includes framework continuations, so it can exceed the business
    counts); ``accepted_platform_operations`` counts accepted Platform
    exchanges; ``causally_linked_world_commits`` is the number of World commit
    ids the exact settlement join attributes to this actor.
    """

    actor_id: str
    role: str
    scripted_business_decisions: int
    verified_business_decisions: int
    complete_tracker_records: int
    accepted_platform_operations: int
    causally_linked_world_commits: int

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not self.actor_id.strip():
            raise EnvironmentStudyError("actor coverage requires a non-empty actor id")
        if self.role not in _ROLE_VALUES:
            raise EnvironmentStudyError("actor coverage role must be 'buyer' or 'merchant'")
        for label, value in (
            ("scripted_business_decisions", self.scripted_business_decisions),
            ("verified_business_decisions", self.verified_business_decisions),
            ("complete_tracker_records", self.complete_tracker_records),
            ("accepted_platform_operations", self.accepted_platform_operations),
            ("causally_linked_world_commits", self.causally_linked_world_commits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EnvironmentStudyError(f"actor coverage {label} must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "role": self.role,
            "scripted_business_decisions": self.scripted_business_decisions,
            "verified_business_decisions": self.verified_business_decisions,
            "complete_tracker_records": self.complete_tracker_records,
            "accepted_platform_operations": self.accepted_platform_operations,
            "causally_linked_world_commits": self.causally_linked_world_commits,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActorCoverageV1":
        if not isinstance(raw, Mapping):
            raise EnvironmentStudyError("actor coverage must be an object")
        _require_exact_keys(raw, required=_ACTOR_COVERAGE_FIELDS, label="actor coverage")
        return cls(
            actor_id=_require_str(raw.get("actor_id"), label="actor_id"),
            role=_require_str(raw.get("role"), label="role"),
            scripted_business_decisions=_require_int(
                raw.get("scripted_business_decisions"), label="scripted_business_decisions"
            ),
            verified_business_decisions=_require_int(
                raw.get("verified_business_decisions"), label="verified_business_decisions"
            ),
            complete_tracker_records=_require_int(
                raw.get("complete_tracker_records"), label="complete_tracker_records"
            ),
            accepted_platform_operations=_require_int(
                raw.get("accepted_platform_operations"), label="accepted_platform_operations"
            ),
            causally_linked_world_commits=_require_int(
                raw.get("causally_linked_world_commits"), label="causally_linked_world_commits"
            ),
        )


_SUMMARY_FIELDS = (
    "transaction_summary",
    "inventory_summary",
    "ledger_summary",
    "fulfillment_summary",
    "after_sales_summary",
    "replay",
    "diagnostics",
    "data_scope",
)


@dataclass(frozen=True)
class EnvironmentStudyReportV1:
    """A capability-free report for one zero-model environment study."""

    study_id: str
    contract_id: str
    valid: bool
    invariants: tuple[InvariantResultV1, ...]
    actor_coverage: tuple[ActorCoverageV1, ...]
    transaction_summary: Mapping[str, Any]
    inventory_summary: Mapping[str, Any]
    ledger_summary: Mapping[str, Any]
    fulfillment_summary: Mapping[str, Any]
    after_sales_summary: Mapping[str, Any]
    replay: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    data_scope: Mapping[str, Any]
    provider_calls: int = 0
    schema_version: str = ENVIRONMENT_STUDY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ENVIRONMENT_STUDY_REPORT_SCHEMA:
            raise EnvironmentStudyError("unsupported environment study report schema")
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise EnvironmentStudyError("environment study report requires a study id")
        _require_digest(self.contract_id, label="contract_id")
        if not isinstance(self.valid, bool):
            raise EnvironmentStudyError("environment study report valid must be boolean")
        if isinstance(self.provider_calls, bool) or self.provider_calls != 0:
            raise EnvironmentStudyError("environment study report requires provider_calls == 0")
        if not self.invariants:
            raise EnvironmentStudyError("environment study report needs at least one invariant")
        if any(not isinstance(row, InvariantResultV1) for row in self.invariants):
            raise EnvironmentStudyError("report invariants must be InvariantResultV1")
        names = [row.name for row in self.invariants]
        if len(names) != len(set(names)):
            raise EnvironmentStudyError("environment study invariant names must be unique")
        # Validity must be consistent with the invariants it summarizes.
        all_passed = all(row.passed for row in self.invariants)
        if self.valid and not all_passed:
            raise EnvironmentStudyError("a valid study cannot contain a failing invariant")
        if any(not isinstance(row, ActorCoverageV1) for row in self.actor_coverage):
            raise EnvironmentStudyError("report actor coverage must be ActorCoverageV1")
        actor_ids = [row.actor_id for row in self.actor_coverage]
        if len(actor_ids) != len(set(actor_ids)):
            raise EnvironmentStudyError("environment study actor coverage ids must be unique")

        for label in _SUMMARY_FIELDS:
            mapping = getattr(self, label)
            if not isinstance(mapping, Mapping):
                raise EnvironmentStudyError(f"environment study {label} must be an object")
            _require_score_free(mapping, label=label)
            # Deep-freeze so no summary can be mutated -- e.g. a score field
            # injected -- after the report has been validated.
            object.__setattr__(self, label, _deep_freeze(dict(mapping)))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "contract_id": self.contract_id,
            "valid": self.valid,
            "invariants": [row.to_dict() for row in self.invariants],
            "actor_coverage": [row.to_dict() for row in self.actor_coverage],
            "provider_calls": self.provider_calls,
        }
        for label in _SUMMARY_FIELDS:
            payload[label] = _thaw(getattr(self, label))
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EnvironmentStudyReportV1":
        """Rebuild and fully re-validate a report (rejects injected fields)."""

        if not isinstance(raw, Mapping):
            raise EnvironmentStudyError("environment study report root must be an object")
        _require_exact_keys(
            raw,
            required=("study_id", "contract_id", "valid", "invariants", "actor_coverage")
            + _SUMMARY_FIELDS,
            optional=("schema_version", "provider_calls"),
            label="environment study report",
        )
        # Audit the ENTIRE raw report -- not only the eight summaries -- for any
        # model-score/ranking/reward key at any depth before rebuilding.
        _require_score_free(raw, label="report")
        invariants = _require_list(raw.get("invariants"), label="report invariants")
        actor_coverage = _require_list(raw.get("actor_coverage"), label="report actor_coverage")
        summaries: dict[str, Any] = {}
        for label in _SUMMARY_FIELDS:
            value = raw.get(label)
            if not isinstance(value, Mapping):
                raise EnvironmentStudyError(f"report {label} must be an object")
            summaries[label] = dict(value)
        return cls(
            study_id=_require_str(raw.get("study_id"), label="study_id"),
            contract_id=_require_str(raw.get("contract_id"), label="contract_id"),
            valid=_require_bool(raw.get("valid"), label="valid"),
            invariants=tuple(InvariantResultV1.from_dict(row) for row in invariants),
            actor_coverage=tuple(ActorCoverageV1.from_dict(row) for row in actor_coverage),
            provider_calls=_require_int(raw.get("provider_calls", 0), label="provider_calls"),
            schema_version=_require_str(
                raw.get("schema_version", ENVIRONMENT_STUDY_REPORT_SCHEMA), label="schema_version"
            ),
            **summaries,
        )


def _require_score_free(value: Any, *, label: str) -> None:
    """Reject any model-score/ranking/reward key anywhere in a report mapping.

    Mapping keys must be safe snake_case identifiers (an allowlist by shape) and
    must not contain a score marker (a denylist by token), so smuggled variants
    such as ``capability_scores`` or ``strict_success_rate`` fail closed.
    """

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if not _SUMMARY_KEY_RE.fullmatch(key):
                raise EnvironmentStudyError(
                    f"environment study {label} key {key!r} is not a safe snake_case identifier"
                )
            if any(marker in key for marker in _SCORE_MARKERS):
                raise EnvironmentStudyError(
                    f"environment study {label} must not contain a model score field: {key}"
                )
            _require_score_free(item, label=label)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _require_score_free(item, label=label)


def write_json_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a deterministic, human-readable JSON artifact.

    Writes a sibling temp file and ``os.replace``s it into place so a crashed
    process never leaves a half-written artifact at ``path``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    handle, tmp_name = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return destination


# --- Network guard ---------------------------------------------------
#
# A zero-model environment study must not reach the network: a scripted policy is
# trusted code, but the process-level guarantee is what makes "provider_calls ==
# 0" enforceable rather than merely asserted.  The guard blocks any non-loopback
# outbound ``connect`` and records the attempts, so the persisted study can
# attest it ran egress-free.  The in-memory ASGI HTTP/VCP path opens no real
# socket, so the guard does not break in-process<->HTTP parity.


class NetworkAccessBlocked(RuntimeError):
    """The study process attempted a forbidden outbound network connection."""


@dataclass
class NetworkGuardLog:
    """Mutable record of connect attempts observed while the guard was active."""

    allow_loopback: bool = True
    active: bool = False
    allowed_loopback_connects: int = 0
    blocked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NETWORK_GUARD_SCHEMA,
            "policy": "loopback_only" if self.allow_loopback else "deny_all",
            "active": self.active,
            "allowed_loopback_connects": self.allowed_loopback_connects,
            "blocked_connect_count": len(self.blocked),
            "blocked_targets": sorted(set(self.blocked)),
            "egress_free": not self.blocked,
        }


@contextmanager
def network_disabled(*, allow_loopback: bool = True) -> Iterator[NetworkGuardLog]:
    """Disable outbound network in this process for the duration of the block.

    Wraps ``socket.socket.connect`` so any non-loopback connect raises
    :class:`NetworkAccessBlocked`, and restores the original on exit (so the
    guard is safe to use inside a shared test process).  Every attempt is
    recorded in the yielded :class:`NetworkGuardLog`.  This stops a scripted
    policy, or any accidental provider call, from reaching the network; it does
    not break the ASGI HTTP/VCP path, which makes no real ``connect``.
    """

    log = NetworkGuardLog(allow_loopback=allow_loopback, active=True)
    real_connect = socket.socket.connect

    def guarded_connect(self: Any, address: Any) -> Any:
        if isinstance(address, (tuple, list)) and address:
            host = str(address[0])
        else:
            host = str(address)
        if allow_loopback and host in _LOOPBACK_HOSTS:
            log.allowed_loopback_connects += 1
            return real_connect(self, address)
        log.blocked.append(host)
        raise NetworkAccessBlocked(
            f"outbound network is disabled in the environment-study process: {host!r}"
        )

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    try:
        yield log
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]


# --- Artifact index --------------------------------------------------


def build_artifact_index(out_dir: str | Path, filenames: Sequence[str]) -> dict[str, Any]:
    """Return an index of ``filenames`` in ``out_dir`` with their bytes+SHA-256.

    Each named artifact must already exist on disk; the index records its exact
    byte length and SHA-256 in the given order, so a downstream reader can verify
    the persisted bundle was not altered.
    """

    root = Path(out_dir)
    entries: list[dict[str, Any]] = []
    for name in filenames:
        payload = (root / name).read_bytes()
        entries.append(
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {"schema_version": ARTIFACT_INDEX_SCHEMA, "artifacts": entries}


__all__ = [
    "ARTIFACT_INDEX_NAME",
    "ARTIFACT_INDEX_SCHEMA",
    "DEFAULT_ENVIRONMENT_STUDY_COMPONENT_SOURCES",
    "ENVIRONMENT_STUDY_ARTIFACT_ORDER",
    "ENVIRONMENT_STUDY_CONTRACT_SCHEMA",
    "ENVIRONMENT_STUDY_REPORT_SCHEMA",
    "NETWORK_GUARD_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "ActorCoverageV1",
    "EnvironmentStudyContractV1",
    "EnvironmentStudyError",
    "EnvironmentStudyReportV1",
    "FrozenSourceFileV1",
    "InvariantResultV1",
    "NetworkAccessBlocked",
    "NetworkGuardLog",
    "SourceManifestV1",
    "build_artifact_index",
    "build_environment_study_contract",
    "build_source_manifest",
    "capture_git_metadata",
    "compute_environment_study_contract_id",
    "iter_loaded_module_files",
    "manifest_non_python_paths",
    "manifest_source_paths",
    "network_disabled",
    "record_repo_reads",
    "repo_source_paths",
    "unlisted_loaded_sources",
    "verify_environment_study_contract",
    "verify_source_manifest",
    "write_json_artifact",
]
