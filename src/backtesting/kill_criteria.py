"""
Layer 14 research-phase kill criteria (MASTER_CONTEXT). Used during Phase
1/2 backtesting to decide continue-vs-revise-the-hypothesis.

THESE ARE NOT THE TIER 1 LIVE-TRADING GATE. MASTER_CONTEXT is explicit that
there are two distinct threshold sets used at different points, and that
conflating them is a real failure mode:

  - Layer 14 (this module): research-phase kill criteria, evaluated on this
    harness's out-of-sample walk-forward folds, to decide whether to keep
    developing this hypothesis at all.
  - Tier 1: a separate, stricter six-metric checklist (Profit Factor > 1.5,
    Sharpe > 1.0, Sortino > 1.5, Calmar > 0.5, Max Drawdown < 20%, EV per
    trade > 0, target > 0.3%) that gates whether real capital is ever
    deployed, checked again at Phase 5. NOT implemented in this module.

Every criterion below reports its own explicit status -- this module
deliberately does NOT collapse the five checks into one aggregate pass/fail
number, since a single boolean would hide which specific criterion failed
and why.

A separate, related but distinct check -- flagging when out-of-sample
Sharpe falls below 60% of in-sample Sharpe -- is an overfitting diagnostic,
not one of the five Layer 14 kill criteria enumerated in MASTER_CONTEXT.
It is implemented at the bottom of this module and clearly labeled as such
so it is never mistaken for a sixth Layer 14 criterion.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CriterionResult:
    """
    `status` is one of:
      "pass"          -- criterion met, no action required.
      "fail"          -- criterion violated; MASTER_CONTEXT's specified
                          action (see `detail`) applies.
      "not_evaluated" -- required inputs weren't available (e.g. no macro
                          regime labels, no ensemble weights supplied
                          because the linear ensemble meta-model isn't
                          wired in yet). This is NOT the same as "pass" --
                          callers must not treat a not_evaluated criterion
                          as satisfied.
    """

    name: str
    status: str
    detail: str
    measured_value: Optional[float] = None
    threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in ("pass", "fail", "not_evaluated"):
            raise ValueError(f"invalid status {self.status!r}; must be pass/fail/not_evaluated.")


@dataclass(frozen=True)
class Layer14KillCriteriaReport:
    """One CriterionResult per Layer 14 criterion. `any_failed` and
    `fully_evaluated` are convenience flags -- the individual criteria in
    `criteria` are the actual output that should be read and acted on."""

    criteria: List[CriterionResult]

    @property
    def any_failed(self) -> bool:
        return any(c.status == "fail" for c in self.criteria)

    @property
    def fully_evaluated(self) -> bool:
        return all(c.status != "not_evaluated" for c in self.criteria)

    def as_report(self) -> List[dict]:
        return [
            {
                "name": c.name,
                "status": c.status,
                "detail": c.detail,
                "measured_value": c.measured_value,
                "threshold": c.threshold,
            }
            for c in self.criteria
        ]


# ------------------------------------------------------------------------ #
# The five Layer 14 criteria, exact MASTER_CONTEXT numbers.
# ------------------------------------------------------------------------ #


def evaluate_oos_sharpe(oos_sharpe: float, threshold: float = 0.5) -> CriterionResult:
    if oos_sharpe is None or np.isnan(oos_sharpe):
        return CriterionResult(
            name="layer14_oos_sharpe",
            status="not_evaluated",
            detail="NOT EVALUATED -- OOS Sharpe could not be computed (insufficient OOS return data).",
            threshold=threshold,
        )
    passed = oos_sharpe >= threshold
    return CriterionResult(
        name="layer14_oos_sharpe",
        status="pass" if passed else "fail",
        detail=(
            f"OOS Sharpe {oos_sharpe:.3f} {'>=' if passed else '<'} {threshold} -> "
            f"{'continue' if passed else 'STOP, revise hypothesis'}"
        ),
        measured_value=float(oos_sharpe),
        threshold=threshold,
    )


def evaluate_oos_profit_factor(oos_profit_factor: float, threshold: float = 1.2) -> CriterionResult:
    if oos_profit_factor is None or np.isnan(oos_profit_factor):
        return CriterionResult(
            name="layer14_oos_profit_factor",
            status="not_evaluated",
            detail="NOT EVALUATED -- OOS profit factor could not be computed (no OOS trades, or no losing trades).",
            threshold=threshold,
        )
    passed = oos_profit_factor >= threshold
    return CriterionResult(
        name="layer14_oos_profit_factor",
        status="pass" if passed else "fail",
        detail=(
            f"OOS profit factor {oos_profit_factor:.3f} {'>=' if passed else '<'} {threshold} -> "
            f"{'continue' if passed else 'STOP, revise hypothesis'}"
        ),
        measured_value=float(oos_profit_factor),
        threshold=threshold,
    )


def evaluate_oos_max_drawdown(oos_max_dd: float, threshold: float = 0.25) -> CriterionResult:
    if oos_max_dd is None or np.isnan(oos_max_dd):
        return CriterionResult(
            name="layer14_oos_max_drawdown",
            status="not_evaluated",
            detail="NOT EVALUATED -- OOS max drawdown could not be computed (insufficient OOS equity data).",
            threshold=threshold,
        )
    passed = oos_max_dd <= threshold
    return CriterionResult(
        name="layer14_oos_max_drawdown",
        status="pass" if passed else "fail",
        detail=(
            f"OOS max drawdown {oos_max_dd:.1%} {'<=' if passed else '>'} {threshold:.0%} -> "
            f"{'continue' if passed else 'STOP, revise risk engine'}"
        ),
        measured_value=float(oos_max_dd),
        threshold=threshold,
    )


def evaluate_risk_on_regime_performance(
    risk_on_sharpe: Optional[float], aggregate_sharpe: Optional[float]
) -> CriterionResult:
    if (
        risk_on_sharpe is None
        or aggregate_sharpe is None
        or np.isnan(risk_on_sharpe)
        or np.isnan(aggregate_sharpe)
    ):
        return CriterionResult(
            name="layer14_risk_on_regime_vs_aggregate",
            status="not_evaluated",
            detail=(
                "NOT EVALUATED -- no macro regime labels were supplied to the harness "
                "(macro_mtf_liquidity_vol Layer 3 regime classifier is not yet wired in), "
                "or too few Risk-On-regime trades to compute a Sharpe ratio."
            ),
        )
    passed = risk_on_sharpe >= aggregate_sharpe
    return CriterionResult(
        name="layer14_risk_on_regime_vs_aggregate",
        status="pass" if passed else "fail",
        detail=(
            f"Risk-On Sharpe {risk_on_sharpe:.3f} {'>=' if passed else '<'} aggregate Sharpe "
            f"{aggregate_sharpe:.3f} -> "
            f"{'continue' if passed else 'macro conditioning is not working; remove macro conditioning and retest simpler hypothesis'}"
        ),
        measured_value=float(risk_on_sharpe - aggregate_sharpe),
        threshold=0.0,
    )


def evaluate_ensemble_weight_stability(
    fold_weights: Sequence[Optional[Dict[str, float]]],
    weight_delta_threshold: float = 0.20,
    consecutive_fold_threshold: int = 2,
) -> CriterionResult:
    """
    MASTER_CONTEXT: "Ensemble weight instability across > 2 consecutive
    folds -> unreliable, fall back to equal weighting and investigate."

    The "> 2 consecutive folds" count is the exact MASTER_CONTEXT number
    (`consecutive_fold_threshold` default). The per-transition instability
    magnitude (`weight_delta_threshold`) is NOT specified in MASTER_CONTEXT
    -- it is a configurable convention (default: any single ensemble weight
    shifting by more than 20 percentage points fold-over-fold counts as an
    unstable transition), not a spec number, and must not be treated as one.
    """
    if len(fold_weights) < 2 or any(w is None for w in fold_weights):
        return CriterionResult(
            name="layer14_ensemble_weight_stability",
            status="not_evaluated",
            detail=(
                "NOT EVALUATED -- ensemble weights were not supplied for every fold "
                "(linear ensemble meta-model is not yet wired in), or fewer than 2 "
                "folds ran."
            ),
        )

    keys = set(fold_weights[0].keys())
    for w in fold_weights[1:]:
        if set(w.keys()) != keys:
            raise ValueError("all folds' ensemble weight dicts must share the same keys.")
    if not keys:
        raise ValueError("ensemble weight dicts are empty.")

    unstable_transitions = [
        max(abs(fold_weights[i][k] - fold_weights[i - 1][k]) for k in keys) > weight_delta_threshold
        for i in range(1, len(fold_weights))
    ]

    longest_run = 0
    current_run = 0
    for unstable in unstable_transitions:
        current_run = current_run + 1 if unstable else 0
        longest_run = max(longest_run, current_run)

    passed = longest_run <= consecutive_fold_threshold
    return CriterionResult(
        name="layer14_ensemble_weight_stability",
        status="pass" if passed else "fail",
        detail=(
            f"longest run of consecutive unstable fold-to-fold weight shifts "
            f"(>{weight_delta_threshold:.0%} change on any component) = {longest_run}, "
            f"{'<=' if passed else '>'} {consecutive_fold_threshold} -> "
            f"{'continue' if passed else 'fall back to equal weighting and investigate'}"
        ),
        measured_value=float(longest_run),
        threshold=float(consecutive_fold_threshold),
    )


def evaluate_layer14_kill_criteria(
    oos_sharpe: float,
    oos_profit_factor: float,
    oos_max_dd: float,
    risk_on_sharpe: Optional[float],
    aggregate_sharpe: Optional[float],
    fold_ensemble_weights: Sequence[Optional[Dict[str, float]]],
    oos_sharpe_threshold: float = 0.5,
    oos_profit_factor_threshold: float = 1.2,
    oos_max_dd_threshold: float = 0.25,
    ensemble_weight_delta_threshold: float = 0.20,
    ensemble_consecutive_fold_threshold: int = 2,
) -> Layer14KillCriteriaReport:
    """Evaluates all five Layer 14 criteria and returns them as a single
    report -- each criterion keeps its own explicit status; nothing here
    reduces them to one number."""
    return Layer14KillCriteriaReport(
        criteria=[
            evaluate_oos_sharpe(oos_sharpe, oos_sharpe_threshold),
            evaluate_oos_profit_factor(oos_profit_factor, oos_profit_factor_threshold),
            evaluate_oos_max_drawdown(oos_max_dd, oos_max_dd_threshold),
            evaluate_risk_on_regime_performance(risk_on_sharpe, aggregate_sharpe),
            evaluate_ensemble_weight_stability(
                fold_ensemble_weights, ensemble_weight_delta_threshold, ensemble_consecutive_fold_threshold
            ),
        ]
    )


# ------------------------------------------------------------------------ #
# Overfitting diagnostic -- requested alongside, but NOT one of, the five
# Layer 14 criteria above. Reported separately so it is never conflated
# with either gate system.
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class OverfitDiagnostic:
    """
    Flags when a fold's out-of-sample Sharpe degrades to below
    `ratio_threshold` (default 60%) of its in-sample Sharpe -- a classic
    overfitting signature (the model looks good on the data it was fit/tuned
    on and materially worse on held-out data).

    Only meaningful when in-sample Sharpe is positive: if there's no
    in-sample edge to begin with, "OOS below 60% of IS" is not a coherent
    overfitting signal (e.g. IS Sharpe -1.0, OOS Sharpe -0.5 would satisfy
    the raw ratio despite OOS being the BETTER outcome). `flagged` is
    forced False whenever in-sample Sharpe <= 0; the Layer 14 OOS Sharpe
    criterion above already covers the no-edge case directly.
    """

    fold: int
    is_sharpe: float
    oos_sharpe: float
    ratio: float
    flagged: bool
    detail: str


def evaluate_is_oos_overfit(fold: int, is_sharpe: float, oos_sharpe: float, ratio_threshold: float = 0.6) -> OverfitDiagnostic:
    if is_sharpe is None or oos_sharpe is None or np.isnan(is_sharpe) or np.isnan(oos_sharpe):
        return OverfitDiagnostic(
            fold=fold, is_sharpe=float("nan"), oos_sharpe=float("nan"), ratio=float("nan"),
            flagged=False, detail="NOT EVALUATED -- missing IS or OOS Sharpe.",
        )
    if is_sharpe <= 0:
        return OverfitDiagnostic(
            fold=fold, is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, ratio=float("nan"),
            flagged=False,
            detail=(
                f"IS Sharpe {is_sharpe:.3f} <= 0 -- 60%-degradation check is not meaningful "
                f"without a positive in-sample edge; not flagged (see Layer 14 OOS Sharpe criterion instead)."
            ),
        )
    ratio = oos_sharpe / is_sharpe
    flagged = ratio < ratio_threshold
    return OverfitDiagnostic(
        fold=fold, is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, ratio=ratio, flagged=flagged,
        detail=(
            f"fold {fold}: OOS/IS Sharpe ratio {ratio:.3f} "
            f"{'<' if flagged else '>='} {ratio_threshold:.0%} -> "
            f"{'FLAGGED: possible overfitting' if flagged else 'ok'}"
        ),
    )
