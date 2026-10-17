"""Public agent primitives, side-specific factories, and platform interfaces.

Buyer and Merchant factories compose agents from skill bundles. Their shared
interfaces include skill loaders, memory stores, and inference channels.
``PlatformService`` is a separate runtime service. The legacy platform-agent
factory and unsupported skill interfaces remain exported for compatibility;
their docstrings identify the supported service replacements.
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
