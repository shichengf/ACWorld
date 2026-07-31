"""Evals package — oracle, named metrics, ablation probes.

The probe interface is wire-stable: new probes can be added without changing
the contract.
"""

from __future__ import annotations

from evals.errors import EvalError, MetricUndefined, ProbeConfigError
from evals.interfaces import Metric, Probe, Reporter
from evals.metrics import (
    BudgetAdherence,
    ConsumerRegret,
    MerchantMargin,
    ProtocolCorrectness,
    StateCorrectness,
    SubRoleLocalization,
    TaskSuccess,
)
from evals.market_metrics import (
    AllocationOracle,
    BuyerValuation,
    Exposure,
    MarketMetricValue,
    MarketMetrics,
    MarketMetricsInputError,
    MarketTransaction,
    MerchantFloor,
    PrivacyEvent,
    ProtocolEvent,
    compute_market_metrics,
)
from evals.oracle import count_violations, find_settle, score
from evals.probes.order import OrderAblationProbe
from evals.types import MetricResult, ProbeReport

__all__ = [
    "AllocationOracle",
    "BudgetAdherence",
    "BuyerValuation",
    "ConsumerRegret",
    "EvalError",
    "Exposure",
    "MarketMetricValue",
    "MarketMetrics",
    "MarketMetricsInputError",
    "MarketTransaction",
    "MerchantMargin",
    "MerchantFloor",
    "Metric",
    "MetricResult",
    "MetricUndefined",
    "OrderAblationProbe",
    "Probe",
    "ProbeConfigError",
    "ProbeReport",
    "ProtocolCorrectness",
    "ProtocolEvent",
    "PrivacyEvent",
    "Reporter",
    "StateCorrectness",
    "SubRoleLocalization",
    "TaskSuccess",
    "count_violations",
    "compute_market_metrics",
    "find_settle",
    "score",
]
