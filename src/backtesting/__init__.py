"""Walk-forward backtesting harness built on vectorbt.

See `harness.WalkForwardBacktester` for the main entry point,
`windows.generate_walk_forward_windows` for the calendar-based fold
generator, `costs.TransactionCostModel` for the transaction cost model, and
`kill_criteria` for the Layer 14 research-phase kill criteria (distinct
from the Tier 1 live-trading gate).
"""

from .costs import CostParameters, TransactionCostModel
from .harness import (
    BacktestReport,
    FoldBacktestResult,
    Layer14KillCriteriaFailed,
    SegmentResult,
    SignalStrategy,
    WalkForwardBacktester,
)
from .kill_criteria import (
    CriterionResult,
    Layer14KillCriteriaReport,
    OverfitDiagnostic,
    evaluate_ensemble_weight_stability,
    evaluate_is_oos_overfit,
    evaluate_layer14_kill_criteria,
    evaluate_oos_max_drawdown,
    evaluate_oos_profit_factor,
    evaluate_oos_sharpe,
    evaluate_risk_on_regime_performance,
)
from .windows import FoldWindow, generate_walk_forward_windows

__all__ = [
    "BacktestReport",
    "CostParameters",
    "CriterionResult",
    "FoldBacktestResult",
    "FoldWindow",
    "Layer14KillCriteriaFailed",
    "Layer14KillCriteriaReport",
    "OverfitDiagnostic",
    "SegmentResult",
    "SignalStrategy",
    "TransactionCostModel",
    "WalkForwardBacktester",
    "evaluate_ensemble_weight_stability",
    "evaluate_is_oos_overfit",
    "evaluate_layer14_kill_criteria",
    "evaluate_oos_max_drawdown",
    "evaluate_oos_profit_factor",
    "evaluate_oos_sharpe",
    "evaluate_risk_on_regime_performance",
    "generate_walk_forward_windows",
]
