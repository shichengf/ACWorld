"""Machine checks for the benchmark paper's Appendix A and Appendix B.

Appendix A is executable documentation: identity, display name, role, task
range, difficulty axis, and runtime scorer binding must match the current
registry.  Appendix B is intentionally different.  Its external sources and
mapping rationale require human review, so automation checks only that every
capability has one complete six-column row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from episode.capability_benchmark import CAPABILITY_GROUPS_V2, TASK_REGISTRY_V2
from episode.capability_runtime_registry import runtime_bundle_v2


_APPENDIX_A_HEADING = "## 附录 A："
_APPENDIX_B_HEADING = "## 附录 B："
_APPENDIX_C_HEADING = "## 附录 C："
_CAPABILITY_CELL_RE = re.compile(r"^`(?P<capability>t(?:10|[1-9])\.[a-z0-9_]+)`$")
_TASK_RANGE_RE = re.compile(
    r"^`(?P<start>CWV2-T(?P<family>\d{2})-(?P<start_ordinal>\d{2}))`"
    r"–`(?:(?P<end_full>CWV2-T\d{2}-\d{2})|(?P<end_ordinal>\d{2}))`$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkAppendixAuditError(ValueError):
    """Appendix structure or code binding does not match the frozen suite."""


@dataclass(frozen=True)
class AppendixAuditReportV1:
    appendix_a_capabilities: int
    appendix_a_tasks: int
    appendix_a_scorer_bindings: int
    appendix_b_capabilities: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": "cwe.benchmark-appendix-audit.v1",
            "appendix_a_capabilities": self.appendix_a_capabilities,
            "appendix_a_tasks": self.appendix_a_tasks,
            "appendix_a_scorer_bindings": self.appendix_a_scorer_bindings,
            "appendix_b_capabilities": self.appendix_b_capabilities,
        }


def audit_benchmark_appendices_v1(document_path: str | Path) -> AppendixAuditReportV1:
    """Validate Appendix A exactly and Appendix B structurally."""

    text = Path(document_path).read_text(encoding="utf-8")
    appendix_a = _section(text, _APPENDIX_A_HEADING, _APPENDIX_B_HEADING)
    appendix_b = _section(text, _APPENDIX_B_HEADING, _APPENDIX_C_HEADING)
    rows_a = _capability_rows(appendix_a, expected_columns=9)
    rows_b = _capability_rows(appendix_b, expected_columns=6)

    groups = {group.capability_id: group for group in CAPABILITY_GROUPS_V2}
    _require_exact_capability_rows(rows_a, groups, label="Appendix A")
    _require_exact_capability_rows(rows_b, groups, label="Appendix B")

    scorer_bindings = 0
    for capability_id, row in rows_a.items():
        group = groups[capability_id]
        tasks = tuple(
            task
            for task in TASK_REGISTRY_V2.values()
            if task.capability_id == capability_id
        )
        start, end = _parse_task_range(row[4])
        expected_values = ", ".join(
            str(dict(profile)[_difficulty_axis(group.difficulty_profiles)])
            for profile in group.difficulty_profiles
        )
        expected = (
            capability_id,
            group.name,
            group.family.value,
            group.evaluated_role.title(),
            tasks[0].task_id,
            tasks[-1].task_id,
            str(len(tasks)),
            _difficulty_axis(group.difficulty_profiles),
            expected_values,
        )
        observed = (
            _unquote_code(row[0]),
            row[1],
            row[2],
            row[3],
            start,
            end,
            row[5],
            _unquote_code(row[6]),
            row[7],
        )
        if observed != expected:
            raise BenchmarkAppendixAuditError(
                f"Appendix A row drifted for {capability_id}: "
                f"observed={observed!r}, expected={expected!r}"
            )
        if not row[8]:
            raise BenchmarkAppendixAuditError(
                f"Appendix A state-object claim is empty for {capability_id}"
            )
        expected_module = (
            f"episode.capability_runtime_t{int(group.family.value[1:])}"
        )
        for task in tasks:
            bundle = runtime_bundle_v2(task.task_id)
            if (
                bundle.task != task
                or not callable(bundle.scorer)
                or getattr(bundle.scorer, "__module__", None) != expected_module
                or not _SHA256_RE.fullmatch(bundle.semantic_hash)
            ):
                raise BenchmarkAppendixAuditError(
                    f"Appendix A task/scorer binding drifted for {task.task_id}"
                )
            scorer_bindings += 1

    for capability_id, row in rows_b.items():
        if _unquote_code(row[0]) != capability_id or any(not cell for cell in row):
            raise BenchmarkAppendixAuditError(
                f"Appendix B row is structurally incomplete for {capability_id}"
            )

    return AppendixAuditReportV1(
        appendix_a_capabilities=len(rows_a),
        appendix_a_tasks=sum(group.task_count for group in groups.values()),
        appendix_a_scorer_bindings=scorer_bindings,
        appendix_b_capabilities=len(rows_b),
    )


def _section(text: str, start_heading: str, end_heading: str) -> str:
    start = text.find(start_heading)
    end = text.find(end_heading)
    if start < 0 or end < 0 or end <= start:
        raise BenchmarkAppendixAuditError(
            f"cannot locate appendix boundary {start_heading!r} -> {end_heading!r}"
        )
    return text[start:end]


def _capability_rows(
    section: str,
    *,
    expected_columns: int,
) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for line in section.splitlines():
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line[1:-1].split("|"))
        if not cells or not (match := _CAPABILITY_CELL_RE.fullmatch(cells[0])):
            continue
        if len(cells) != expected_columns:
            raise BenchmarkAppendixAuditError(
                f"{match.group('capability')} has {len(cells)} columns; "
                f"expected {expected_columns}"
            )
        capability_id = match.group("capability")
        if capability_id in rows:
            raise BenchmarkAppendixAuditError(
                f"appendix repeats capability {capability_id}"
            )
        rows[capability_id] = cells
    return rows


def _require_exact_capability_rows(
    rows: dict[str, tuple[str, ...]],
    groups: dict[str, object],
    *,
    label: str,
) -> None:
    if set(rows) != set(groups) or len(rows) != len(groups):
        missing = sorted(set(groups) - set(rows))
        extra = sorted(set(rows) - set(groups))
        raise BenchmarkAppendixAuditError(
            f"{label} capability coverage drifted: missing={missing[:3]!r}, "
            f"extra={extra[:3]!r}"
        )


def _difficulty_axis(profiles: Sequence[Sequence[tuple[str, object]]]) -> str:
    axes = {
        name
        for profile in profiles
        for name, _value in profile
        if name != "difficulty_level"
    }
    if len(axes) != 1:
        raise BenchmarkAppendixAuditError("capability has no single difficulty axis")
    return next(iter(axes))


def _parse_task_range(value: str) -> tuple[str, str]:
    match = _TASK_RANGE_RE.fullmatch(value)
    if match is None:
        raise BenchmarkAppendixAuditError(f"invalid Appendix A task range: {value!r}")
    start = match.group("start")
    end = match.group("end_full") or (
        f"CWV2-T{match.group('family')}-{match.group('end_ordinal')}"
    )
    return start, end


def _unquote_code(value: str) -> str:
    return value[1:-1] if value.startswith("`") and value.endswith("`") else value


__all__ = [
    "AppendixAuditReportV1",
    "BenchmarkAppendixAuditError",
    "audit_benchmark_appendices_v1",
]
