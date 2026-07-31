"""Deterministic logical-time scheduling with bounded concurrent preparation.

Agent inference may run concurrently, but world-visible commits must not depend
on network completion order. The scheduler therefore separates ``prepare``
(safe to parallelize) from ``commit`` (always applied in stable event order).
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from typing import Generic, TypeVar


PayloadT = TypeVar("PayloadT")
PreparedT = TypeVar("PreparedT")
CommittedT = TypeVar("CommittedT")


class SchedulingError(ValueError):
    """An event id, logical time, or concurrency setting is invalid."""


@dataclass(order=True, frozen=True, slots=True)
class ScheduledEvent(Generic[PayloadT]):
    """One event ordered by logical time, priority, and stable event id.

    ``sequence`` remains observable for audit/debugging, but it is only a final
    tie-breaker. Unique event ids therefore produce the same commit order even
    when concurrently prepared events are inserted in a different order.
    """

    _sort_key: tuple[int, int, str, int] = field(init=False, repr=False)
    logical_time: int = field(compare=False)
    priority: int = field(compare=False)
    sequence: int = field(compare=False)
    event_id: str = field(compare=False)
    payload: PayloadT = field(compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_sort_key",
            (self.logical_time, self.priority, self.event_id, self.sequence),
        )


class DeterministicScheduler(Generic[PayloadT]):
    """Thread-safe min-heap keyed by logical time, priority, then insertion."""

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent[PayloadT]] = []
        self._ids: set[str] = set()
        self._sequence = 0
        self._lock = RLock()

    def schedule(
        self,
        event_id: str,
        payload: PayloadT,
        *,
        logical_time: int = 0,
        priority: int = 0,
    ) -> ScheduledEvent[PayloadT]:
        if not event_id.strip():
            raise SchedulingError("event_id must be non-empty")
        if logical_time < 0:
            raise SchedulingError("logical_time must be non-negative")
        with self._lock:
            if event_id in self._ids:
                raise SchedulingError(f"duplicate event_id {event_id!r}")
            event = ScheduledEvent(
                logical_time, priority, self._sequence, event_id, payload
            )
            self._sequence += 1
            self._ids.add(event_id)
            heapq.heappush(self._heap, event)
            return event

    def pop(self) -> ScheduledEvent[PayloadT]:
        with self._lock:
            if not self._heap:
                raise IndexError("scheduler is empty")
            return heapq.heappop(self._heap)

    def drain(self) -> tuple[ScheduledEvent[PayloadT], ...]:
        events: list[ScheduledEvent[PayloadT]] = []
        while self:
            events.append(self.pop())
        return tuple(events)

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


def prepare_concurrently_commit_deterministically(
    events: Iterable[ScheduledEvent[PayloadT]],
    *,
    prepare: Callable[[ScheduledEvent[PayloadT]], PreparedT],
    commit: Callable[[ScheduledEvent[PayloadT], PreparedT], CommittedT],
    max_workers: int,
) -> tuple[CommittedT, ...]:
    """Prepare in a bounded worker pool, then commit in stable event order.

    All preparation results are collected before the first commit. A preparation
    exception therefore produces zero world-visible commits for this batch.
    """
    if isinstance(max_workers, bool) or max_workers <= 0:
        raise SchedulingError("max_workers must be a positive integer")
    ordered = tuple(sorted(events))
    if len({event.event_id for event in ordered}) != len(ordered):
        raise SchedulingError("batch contains duplicate event ids")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        prepared = tuple(pool.map(prepare, ordered))
    return tuple(commit(event, value) for event, value in zip(ordered, prepared))


__all__ = [
    "DeterministicScheduler",
    "ScheduledEvent",
    "SchedulingError",
    "prepare_concurrently_commit_deterministically",
]
