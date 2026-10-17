"""Error taxonomy for the typed commerce Agent decision boundary.

Only invalid model-authored typed business decisions are scoreable model
protocol failures. Missing or contradictory authority is an environment
defect and must escape the model-repair loop as an unscoreable infrastructure
failure.
"""

from __future__ import annotations


class SemanticBoundaryError(Exception):
    """Base class for typed business-decision boundary failures."""


class ModelBusinessDecisionError(SemanticBoundaryError, ValueError):
    """The model selected an unknown intent or invalid business arguments."""


class FrameworkAuthorityError(SemanticBoundaryError, RuntimeError):
    """The Agent lacks complete authenticated authority for a promised route."""


class PlatformContractError(FrameworkAuthorityError):
    """An authenticated Platform response violates its public contract."""


class DeterministicCompilerError(FrameworkAuthorityError):
    """Validated model intent could not compile to the actor terminal contract."""


__all__ = [
    "DeterministicCompilerError",
    "FrameworkAuthorityError",
    "ModelBusinessDecisionError",
    "PlatformContractError",
    "SemanticBoundaryError",
]
