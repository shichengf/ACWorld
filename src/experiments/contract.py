"""Offline experiment contracts for reproducible CommerceWorld runs.

The contract records every input that can change a benchmark result without
ever inspecting environment variables or secret files.  Its stable identity
is intentionally independent of creation time and host/runtime metadata so a
frozen experiment has the same ID on another machine.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_SCHEMA = "cwe.experiment-contract.v1"
REQUIRED_CATEGORIES = frozenset(
    {"scenario", "reactive_policy", "prompt", "skill", "scorer", "model_manifest"}
)
_LOCKFILE_NAMES = ("uv.lock", "poetry.lock", "Pipfile.lock", "pdm.lock", "pixi.lock")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
)
_SECRET_TOKEN_NAMES = frozenset(
    {"token", "access_token", "auth_token", "bearer_token", "refresh_token", "api_token"}
)


class ContractError(ValueError):
    """An experiment contract is unsafe, incomplete, or internally invalid."""


class ContractVerificationError(ContractError):
    """The repository no longer matches a frozen experiment contract."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("experiment contract verification failed: " + "; ".join(self.issues))


@dataclass(frozen=True)
class GitMetadata:
    """Version-control state at contract creation time."""

    commit: str
    dirty: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GitMetadata:
        commit = str(raw.get("commit", "")).strip()
        dirty = raw.get("dirty")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise ContractError(f"invalid git commit: {commit!r}")
        if not isinstance(dirty, bool):
            raise ContractError("git.dirty must be a boolean")
        return cls(commit=commit, dirty=dirty)

    def to_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty}


@dataclass(frozen=True)
class RuntimeMetadata:
    """Informational runtime data excluded from the stable contract ID."""

    python: str
    implementation: str
    platform: str

    @classmethod
    def current(cls) -> RuntimeMetadata:
        return cls(
            python=platform.python_version(),
            implementation=platform.python_implementation(),
            platform=platform.platform(),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RuntimeMetadata:
        values = {
            name: str(raw.get(name, "")).strip()
            for name in ("python", "implementation", "platform")
        }
        if any(not value for value in values.values()):
            raise ContractError("runtime metadata fields must be non-empty")
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        return {
            "python": self.python,
            "implementation": self.implementation,
            "platform": self.platform,
        }


@dataclass(frozen=True, order=True)
class FrozenInput:
    """One repository-relative input digest."""

    category: str
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FrozenInput:
        category = str(raw.get("category", "")).strip()
        path = str(raw.get("path", "")).strip()
        size = raw.get("bytes")
        sha256 = str(raw.get("sha256", "")).strip()
        if not category or category not in REQUIRED_CATEGORIES | {"lockfile"}:
            raise ContractError(f"invalid frozen-input category: {category!r}")
        _validate_relative_manifest_path(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(f"invalid byte count for {path!r}: {size!r}")
        if not _SHA256_RE.fullmatch(sha256):
            raise ContractError(f"invalid sha256 for {path!r}")
        return cls(category=category, path=path, bytes=size, sha256=sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExperimentContract:
    """A complete, deterministic description of benchmark inputs."""

    contract_id: str
    created_at: str
    git: GitMetadata
    runtime: RuntimeMetadata
    argv: dict[str, Any]
    request_config: dict[str, Any]
    plan_sha256: str
    inputs: tuple[FrozenInput, ...]
    schema_version: str = CONTRACT_SCHEMA

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExperimentContract:
        if str(raw.get("schema_version", "")) != CONTRACT_SCHEMA:
            raise ContractError(
                f"unsupported contract schema {raw.get('schema_version')!r}; "
                f"expected {CONTRACT_SCHEMA!r}"
            )
        contract_id = str(raw.get("contract_id", "")).strip()
        if not _SHA256_RE.fullmatch(contract_id):
            raise ContractError("contract_id must be a lowercase sha256 digest")
        created_at = str(raw.get("created_at", "")).strip()
        if not created_at:
            raise ContractError("created_at must be non-empty")
        git_raw = raw.get("git")
        runtime_raw = raw.get("runtime")
        argv_raw = raw.get("argv")
        request_config_raw = raw.get("request_config")
        plan_sha256 = str(raw.get("plan_sha256", "")).strip()
        inputs_raw = raw.get("inputs")
        if not isinstance(git_raw, Mapping) or not isinstance(runtime_raw, Mapping):
            raise ContractError("git and runtime must be objects")
        if not isinstance(argv_raw, Mapping):
            raise ContractError("argv must be an object")
        if not isinstance(request_config_raw, Mapping):
            raise ContractError("request_config must be an object")
        if not _SHA256_RE.fullmatch(plan_sha256):
            raise ContractError("plan_sha256 must be a lowercase sha256 digest")
        if not isinstance(inputs_raw, list):
            raise ContractError("inputs must be a list")
        argv = _normalise_argv(argv_raw)
        request_config = _normalise_argv(request_config_raw, label="request_config")
        inputs = tuple(FrozenInput.from_dict(item) for item in inputs_raw)
        _validate_input_set(inputs)
        contract = cls(
            schema_version=CONTRACT_SCHEMA,
            contract_id=contract_id,
            created_at=created_at,
            git=GitMetadata.from_dict(git_raw),
            runtime=RuntimeMetadata.from_dict(runtime_raw),
            argv=argv,
            request_config=request_config,
            plan_sha256=plan_sha256,
            inputs=inputs,
        )
        expected = compute_contract_id(contract.git, contract.request_config, contract.inputs)
        if contract.contract_id != expected:
            raise ContractError(
                f"contract_id mismatch: recorded {contract.contract_id}, computed {expected}"
            )
        return contract

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "created_at": self.created_at,
            "git": self.git.to_dict(),
            "runtime": self.runtime.to_dict(),
            "argv": self.argv,
            "request_config": self.request_config,
            "plan_sha256": self.plan_sha256,
            "inputs": [item.to_dict() for item in self.inputs],
        }


def build_experiment_contract(
    repo_root: str | Path,
    *,
    argv: Mapping[str, Any],
    request_config: Mapping[str, Any],
    plan_sha256: str,
    scenario_paths: Iterable[str | Path],
    reactive_policy_paths: Iterable[str | Path],
    prompt_paths: Iterable[str | Path] | None = None,
    skill_paths: Iterable[str | Path] | None = None,
    scorer_paths: Iterable[str | Path] = ("src/evals/scorer.py",),
    model_manifest_path: str | Path,
    lockfile_paths: Iterable[str | Path] | None = None,
    created_at: str | None = None,
) -> ExperimentContract:
    """Build a contract using only explicitly safe repository inputs.

    Prompt and skill paths default to their checked-in benchmark directories;
    dependency lockfiles are included when present.  Scenarios and reactive
    policies are explicit so a caller freezes precisely the planned run set.
    """

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise ContractError(f"repository root is not a directory: {root}")
    safe_argv = _normalise_argv(argv)
    safe_request_config = _normalise_argv(request_config, label="request_config")
    if not _SHA256_RE.fullmatch(plan_sha256):
        raise ContractError("plan_sha256 must be a lowercase sha256 digest")
    categories: dict[str, list[str | Path]] = {
        "scenario": list(scenario_paths),
        "reactive_policy": list(reactive_policy_paths),
        "prompt": list(prompt_paths) if prompt_paths is not None else _glob(root, "src/agents/prompts/*"),
        "skill": list(skill_paths) if skill_paths is not None else _glob(root, "skills/**/SKILL.md"),
        "scorer": list(scorer_paths),
        "model_manifest": [model_manifest_path],
    }
    if lockfile_paths is None:
        lockfile_paths = [name for name in _LOCKFILE_NAMES if (root / name).is_file()]
    categories["lockfile"] = list(lockfile_paths)

    frozen: list[FrozenInput] = []
    for category, paths in categories.items():
        if category in REQUIRED_CATEGORIES and not paths:
            raise ContractError(f"required input category {category!r} is empty")
        for path in paths:
            frozen.append(_freeze_input(root, category, path))
    frozen.sort()
    inputs = tuple(frozen)
    _validate_input_set(inputs)
    git = _read_git_metadata(root)
    contract_id = compute_contract_id(git, safe_request_config, inputs)
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="seconds")
    return ExperimentContract(
        contract_id=contract_id,
        created_at=timestamp,
        git=git,
        runtime=RuntimeMetadata.current(),
        argv=safe_argv,
        request_config=safe_request_config,
        plan_sha256=plan_sha256,
        inputs=inputs,
    )


def compute_contract_id(
    git: GitMetadata,
    request_config: Mapping[str, Any],
    inputs: Sequence[FrozenInput],
) -> str:
    """Return the stable ID, excluding timestamps, host, and batch selection.

    ``request_config`` contains only parameters that affect model/runtime
    semantics.  The complete CLI argv and plan digest remain recorded as batch
    provenance, but compact/full selection and row subsets cannot fork the
    underlying benchmark contract.
    """

    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "git": git.to_dict(),
        "request_config": _normalise_argv(request_config, label="request_config"),
        "inputs": [item.to_dict() for item in sorted(inputs)],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def write_contract(contract: ExperimentContract, path: str | Path) -> Path:
    """Write a validated contract as deterministic, human-readable JSON."""

    validated = ExperimentContract.from_dict(contract.to_dict())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_contract(path: str | Path) -> ExperimentContract:
    """Load a contract and reject schema, identity, or completeness tampering."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ContractError("experiment contract root must be an object")
    return ExperimentContract.from_dict(raw)


def verify_contract(
    contract: ExperimentContract,
    repo_root: str | Path,
    *,
    verify_git: bool = True,
    plan_sha256: str | None = None,
) -> ExperimentContract:
    """Require all frozen files (and optionally git state) to still match."""

    root = Path(repo_root).resolve(strict=True)
    issues: list[str] = []
    expected_id = compute_contract_id(contract.git, contract.request_config, contract.inputs)
    if contract.contract_id != expected_id:
        issues.append("contract_id does not match the stable contract payload")
    for item in contract.inputs:
        try:
            current = _freeze_input(root, item.category, item.path)
        except ContractError as exc:
            issues.append(str(exc))
            continue
        if current.bytes != item.bytes:
            issues.append(
                f"{item.path}: byte count changed from {item.bytes} to {current.bytes}"
            )
        if current.sha256 != item.sha256:
            issues.append(f"{item.path}: sha256 changed")
    if verify_git:
        try:
            current_git = _read_git_metadata(root)
        except ContractError as exc:
            issues.append(str(exc))
        else:
            if current_git != contract.git:
                issues.append(
                    "git state changed: "
                    f"expected {contract.git.commit} dirty={contract.git.dirty}, "
                    f"got {current_git.commit} dirty={current_git.dirty}"
                )
    if plan_sha256 is not None:
        if not _SHA256_RE.fullmatch(plan_sha256):
            issues.append("current plan_sha256 is not a lowercase sha256 digest")
        elif plan_sha256 != contract.plan_sha256:
            issues.append(
                f"plan_sha256 changed from {contract.plan_sha256} to {plan_sha256}"
            )
    if issues:
        raise ContractVerificationError(issues)
    return contract


def _freeze_input(root: Path, category: str, source: str | Path) -> FrozenInput:
    if category not in REQUIRED_CATEGORIES | {"lockfile"}:
        raise ContractError(f"invalid frozen-input category: {category!r}")
    candidate = Path(source)
    unresolved = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ContractError(f"required input does not exist: {source}") from exc
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"input path escapes repository root: {source}") from exc
    if not resolved.is_file():
        raise ContractError(f"input is not a regular file: {source}")
    relative_path = relative.as_posix()
    _reject_sensitive_path(relative_path)
    if _is_git_ignored(root, relative_path):
        raise ContractError(f"refusing to read ignored input: {relative_path}")
    content = resolved.read_bytes()
    return FrozenInput(
        category=category,
        path=relative_path,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _read_git_metadata(root: Path) -> GitMetadata:
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
        raise ContractError(f"cannot inspect git repository at {root}") from exc
    return GitMetadata.from_dict({"commit": commit, "dirty": bool(status.strip())})


def _is_git_ignored(root: Path, relative_path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--quiet", "--no-index", "--", relative_path],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ContractError(f"cannot check ignore rules for {relative_path}") from exc
    if result.returncode not in (0, 1):
        raise ContractError(f"cannot check ignore rules for {relative_path}")
    return result.returncode == 0


def _normalise_argv(raw: Mapping[str, Any], *, label: str = "argv") -> dict[str, Any]:
    normalised = _normalise_json_value(raw, path=label)
    if not isinstance(normalised, dict):  # pragma: no cover - guaranteed by the input type
        raise ContractError("argv must be an object")
    return normalised


def _normalise_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = str(key).strip()
            if not name:
                raise ContractError(f"{path} contains an empty key")
            if _looks_secret(name):
                raise ContractError(f"refusing secret-like argv field: {path}.{name}")
            result[name] = _normalise_json_value(item, path=f"{path}.{name}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalise_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ContractError(f"{path} contains unsupported value type {type(value).__name__}")


def _looks_secret(value: str) -> bool:
    compact = value.lower().replace("-", "_")
    return any(marker in compact for marker in _SECRET_KEY_MARKERS) or compact in _SECRET_TOKEN_NAMES


def _reject_sensitive_path(relative_path: str) -> None:
    path = Path(relative_path)
    lower_parts = tuple(part.lower() for part in path.parts)
    if lower_parts[:2] == ("data", "raw_data"):
        raise ContractError(f"refusing raw real-data input: {relative_path}")
    if any(_looks_secret(part) for part in path.parts) or path.name.lower().startswith(".env"):
        raise ContractError(f"refusing secret-like input path: {relative_path}")


def _validate_relative_manifest_path(path: str) -> None:
    candidate = Path(path)
    if (
        not path
        or path == "."
        or candidate.is_absolute()
        or ".." in candidate.parts
        or path != candidate.as_posix()
    ):
        raise ContractError(f"input path must be normalized and repository-relative: {path!r}")
    _reject_sensitive_path(path)


def _validate_input_set(inputs: Sequence[FrozenInput]) -> None:
    if tuple(inputs) != tuple(sorted(inputs)):
        raise ContractError("frozen inputs must use deterministic category/path ordering")
    keys = [(item.category, item.path) for item in inputs]
    if len(keys) != len(set(keys)):
        raise ContractError("frozen inputs contain duplicate category/path entries")
    paths = [item.path for item in inputs]
    if len(paths) != len(set(paths)):
        raise ContractError("one file cannot be frozen under multiple categories")
    missing = REQUIRED_CATEGORIES - {item.category for item in inputs}
    if missing:
        raise ContractError(f"contract is missing required input categories: {sorted(missing)}")


def _glob(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "CONTRACT_SCHEMA",
    "REQUIRED_CATEGORIES",
    "ContractError",
    "ContractVerificationError",
    "ExperimentContract",
    "FrozenInput",
    "GitMetadata",
    "RuntimeMetadata",
    "build_experiment_contract",
    "compute_contract_id",
    "load_contract",
    "verify_contract",
    "write_contract",
]
