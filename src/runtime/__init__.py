"""Runtime package — bus, router, and audit log.

The ``Bus`` Protocol is the single seam that lets a cross-process transport
drop in without lane changes.
"""

from __future__ import annotations

from runtime.audit import AuditLog
from runtime.bus import Runtime
from runtime.errors import (
    AgentNotRegistered,
    ActorEvidenceUnavailable,
    BudgetExceeded,
    BusError,
    DuplicateAgentId,
    PrivateUtilityLeak,
)
from runtime.interfaces import AuditWriter, Bus, Router as RouterProto
from runtime.privacy import LeakFinding, MoneySecret, SecretRegistry
from runtime.router import Router
from runtime.scheduler import (
    DeterministicScheduler,
    ScheduledEvent,
    SchedulingError,
    prepare_concurrently_commit_deterministically,
)
from runtime.types import AuditRecord, AuditRecordView, InferenceBudget, RouteEntry, SecurityEvent

__all__ = [
    "AgentNotRegistered",
    "ActorEvidenceUnavailable",
    "AuditLog",
    "AuditRecord",
    "AuditRecordView",
    "AuditWriter",
    "BudgetExceeded",
    "Bus",
    "BusError",
    "DuplicateAgentId",
    "DeterministicScheduler",
    "InferenceBudget",
    "LeakFinding",
    "MoneySecret",
    "PrivateUtilityLeak",
    "RouteEntry",
    "Router",
    "RouterProto",
    "Runtime",
    "SecretRegistry",
    "ScheduledEvent",
    "SchedulingError",
    "SecurityEvent",
    "prepare_concurrently_commit_deterministically",
]
