"""Fail-closed verifier-local replay of World transaction table writes.

Exact evidence contracts use this helper to reconstruct the authoritative
state visible at a particular Platform request position from serialized World
commit evidence.  It intentionally preserves every before/after-chain check.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from runtime.exact_join import ExactJoinError


def apply_commit_writes(
    state: dict[str, Any],
    commit: Mapping[str, Any],
    *,
    allowed_tables: frozenset[str],
) -> None:
    """Apply one exact World commit to mutable verifier-local ``state``."""

    writes = commit.get("table_writes")
    if not isinstance(writes, list) or not writes:
        raise ExactJoinError("claimed World transaction has no table writes")
    observed_tables = {
        str(row.get("table")) for row in writes if isinstance(row, Mapping)
    }
    if not observed_tables or not observed_tables.issubset(allowed_tables):
        raise ExactJoinError("World transaction wrote outside its authority tables")
    for ordinal, value in enumerate(writes):
        if not isinstance(value, Mapping):
            raise ExactJoinError(f"World table write[{ordinal}] is not an object")
        table = _required_text(value.get("table"), "World write table")
        key = _required_text(value.get("key"), "World write key")
        operation = _required_text(value.get("op"), "World write operation")
        before = value.get("before")
        after = value.get("after")
        if table not in state:
            raise ExactJoinError(f"World write names unknown initial table {table!r}")
        current = state[table]
        if isinstance(current, list):
            matches = [index for index, row in enumerate(current) if row == before]
            if operation == "create":
                if before is not None:
                    raise ExactJoinError("World create write has a non-null before row")
                current.append(copy.deepcopy(after))
            elif operation == "update":
                if len(matches) != 1:
                    raise ExactJoinError(
                        f"World update before row is not unique in {table!r}"
                    )
                current[matches[0]] = copy.deepcopy(after)
            elif operation == "delete":
                if len(matches) != 1 or after is not None:
                    raise ExactJoinError("World delete write has an invalid row chain")
                current.pop(matches[0])
            else:
                raise ExactJoinError(f"unsupported World write operation {operation!r}")
            if table == "orders" and operation in {"create", "update"}:
                revisions = state.get("order_state_revisions")
                if not isinstance(revisions, dict):
                    raise ExactJoinError(
                        "World order_state_revisions table has an invalid shape"
                    )
                prior = revisions.get(key, 0)
                if isinstance(prior, bool) or not isinstance(prior, int) or prior < 0:
                    raise ExactJoinError("World order revision is invalid")
                revisions[key] = prior + 1
        elif isinstance(current, dict):
            observed = current.get(key)
            if observed != before:
                raise ExactJoinError(
                    f"World mapping write before row changed for {table!r}:{key!r}"
                )
            if operation == "create":
                if key in current or before is not None:
                    raise ExactJoinError("World mapping create is not fresh")
                current[key] = copy.deepcopy(after)
            elif operation == "update":
                if key not in current:
                    raise ExactJoinError("World mapping update has no prior row")
                current[key] = copy.deepcopy(after)
            elif operation == "delete":
                if key not in current or after is not None:
                    raise ExactJoinError("World mapping delete has an invalid row chain")
                del current[key]
            else:
                raise ExactJoinError(f"unsupported World write operation {operation!r}")
        else:
            if key != "world" or current != before or operation != "update":
                raise ExactJoinError("World scalar write has an invalid state chain")
            state[table] = copy.deepcopy(after)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExactJoinError(f"{label} must be non-empty text")
    return value


__all__ = ["apply_commit_writes"]
