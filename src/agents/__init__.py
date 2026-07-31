"""Agents package — the AGENT_CLASSES.md primitives + per-side concrete agents.

Public surface (per AGENT_CLASSES.md §1, §10):
    Primitives:    Agent, SkillLoader, Memory, AgentInputs
    Skill bundles: SkillManifest, SkillDocument
    Enums:         SkillGroup, MemoryType
    Loaders:       FilesystemSkillLoader, EmptySkillLoader, InMemoryStore
    Per-side:      make_buyer_agent / make_merchant_agent / make_platform_agent
"""

from __future__ import annotations

from agents.base import Agent
from agents.buyer import make_buyer_agent
from agents.errors import (
    AgentError,
    BudgetExceeded,
    FrontmatterError,
    MandateLeak,
    MultiEmitForbidden,
    SkillNotFound,
)
from agents.inference import (
    BusinessDecisionResponseV1,
    ChannelTransportError,
    InferenceChannel,
    ModelDecisionParseError,
    OpenAIChannel,
    ProviderResponseError,
)
from agents.interfaces import Memory, SkillLoader
from agents.memory import InMemoryStore
from agents.merchant import make_merchant_agent
from agents.platform import (
    AtomicSettleSkill,
    ForwardSearchSkill,
    PlatformService,
    RejectDisputeSkill,
    make_platform_agent,
)
from agents.skills import EmptySkillLoader, FilesystemSkillLoader
from agents.types import (
    AgentContext,
    AgentInputs,
    MemoryType,
    SkillDocument,
    SkillGroup,
    SkillManifest,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentError",
    "AgentInputs",
    "AtomicSettleSkill",
    "BusinessDecisionResponseV1",
    "BudgetExceeded",
    "ChannelTransportError",
    "EmptySkillLoader",
    "FilesystemSkillLoader",
    "ForwardSearchSkill",
    "FrontmatterError",
    "InMemoryStore",
    "InferenceChannel",
    "MandateLeak",
    "Memory",
    "MemoryType",
    "ModelDecisionParseError",
    "MultiEmitForbidden",
    "OpenAIChannel",
    "PlatformService",
    "ProviderResponseError",
    "RejectDisputeSkill",
    "SkillDocument",
    "SkillGroup",
    "SkillLoader",
    "SkillManifest",
    "SkillNotFound",
    "make_buyer_agent",
    "make_merchant_agent",
    "make_platform_agent",
]
