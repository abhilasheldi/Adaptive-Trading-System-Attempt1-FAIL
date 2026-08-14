"""
Walk-forward backtesting harness (MASTER_CONTEXT constraint 4: walk-forward
validation only -- train, then validate, then test, rolling forward, never
overlapping within a fold).

Wires together:
  - windows.py:       calendar-based fold generation
  - costs.py:          transaction cost model, fit per-fold on TRAIN only
  - vectorbt:           mechanical order simulation and trade bookkeeping
                        (MASTER_CONTEXT's tech stack: "vectorbt (backtesting
                        scaffold -- don't build walk-forward/Monte Carlo
                        from scratch)". This module uses vectorbt strictly
                        for order/position mechanics; every cost, Sharpe,
                        profit factor, drawdown, and EV number reported is
                        computed by this harness's own cost-adjusted
                        pipeline, not read off vectorbt's built-in
                        (gross-of-our-cost-model) statistics.
  - metrics.py:         Sharpe / profit factor / max drawdown / EV
  - kill_criteria.py:   Layer 14 research-phase kill criteria (NOT the
                        Tier 1 live-trading gate) + the IS/OOS overfit
                        diagnostic

Scope note: one instrument per `run()` call. For BTC and ETH, run the
harness twice. Feature columns the strategy needs (rolling indicators,
regime labels, etc.) must already be present in the input DataFrame,
computed once over the full historical series upstream (see
`features/feature_calculator.py`) -- this harness partitions time ranges,
it does not manage feature warm-up/lookback buffers.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import vectorbt as vbt

from . import metrics
from .costs import TransactionCostModel
from .kill_criteria import (
    Layer14KillCriteriaReport,
    OverfitDiagnostic,
    evaluate_is_oos_overfit,
    evaluate_layer14_kill_criteria,
)
from .windows import FoldWindow, generate_walk_forward_windows


class Layer14KillCriteriaFailed(RuntimeError):
    """
    Raised by `WalkForwardBacktester.run()` when at least one Layer 14
    research-phase kill criterion fails and `raise_on_kill_criteria_failure`
    is True (the default).

    MASTER_CONTEXT's language for these criteria is imperative -- "STOP,
    revise hypothesis", "STOP, revise risk engine" -- not "note it and
    continue". A report field nobody is forced to check is not a stop; this
    exception is the actual halt. The full `BacktestReport` is attached via
    `.report` so a caller that deliberately wants to inspect a failing run
    can still do so (from the exception, or by passing
    `raise_on_kill_criteria_failure=False` up front to get the report back
    without an exception at all, e.g. for exploratory/notebook use).
    """

    def __init__(self, report: "BacktestReport") -> None:
        self.report = report
        failed_names = [c["name"] for c in report.kill_criteria.as_report() if c["status"] == "fail"]
        super().__init__(
            "Layer 14 research-phase kill criteria FAILED: " + ", ".join(failed_names) + ". "
            "This hypothesis should not proceed to further research phases without revision. "
            "Full per-criterion detail is available on this exception's .report attribute."
        )


class SignalStrategy(Protocol):
    """
    Contract a caller-supplied strategy must satisfy. The harness calls
    `fit` with only the train and validation partitions of a fold, then
    calls `generate_positions` separately on the in-sample (train+val) and
    out-of-sample (test) ranges -- `fit` structurally never receives the
    test partition, mirroring the isolation pattern in
    `predictive_core.boosted_core`.
    """

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        ...

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """Returns a target position size (fraction of capital, signed for
        direction) indexed like `df`. Risk caps (2%/6%/10%, quarter-Kelly)
        are the responsibility of the risk_sizing_selection module -- this
        harness executes whatever position series it is given."""
        ...

    def ensemble_weights(self) -> Optional[Dict[str, float]]:
        """Returns this fold's fitted linear ensemble meta-model weights,
        or None if the strategy doesn't use/expose an ensemble (in which
        case the Layer 14 ensemble-stability criterion reports
        not_evaluated)."""
        ...

    def diagnostics(self) -> Optional[dict]:
        """Returns arbitrary strategy-specific diagnostics captured during
        this fold's fit() (e.g. classifier calibration/discrimination
        reports), surfaced per-fold on FoldBacktestResult.diagnostics for
        reporting -- the harness itself doesn't interpret these, only
        carries them through. Return None if the strategy has nothing to
        report."""
        ...


@dataclass
class SegmentResult:
    segment: str  # "in_sample" | "out_of_sample"
    fold: int
    trades: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    sharpe: float
    profit_factor: float
    max_drawdown: float
    ev_per_trade: float
    n_trades: int
    # True when the equity curve reached zero or below during this segment.
    # Past that point the account is gone: every subsequent return, and any
    # Sharpe derived from them, is arithmetically meaningless (the pct_change
    # denominator has crossed zero). Surfaced so callers can DISQUALIFY the
    # Sharpe rather than compare it against runs that stayed solvent.
    equity_destroyed: bool = False


@dataclass
class FoldBacktestResult:
    fold: int
    window: FoldWindow
    in_sample: SegmentResult
    out_of_sample: SegmentResult
    overfit: OverfitDiagnostic
    risk_on_sharpe: Optional[float]
    ensemble_weights: Optional[Dict[str, float]]
    diagnostics: Optional[dict]


@dataclass
class BacktestReport:
    folds: List[FoldBacktestResult]
    aggregate_oos_sharpe: float
    aggregate_oos_profit_factor: float
    aggregate_oos_max_drawdown: float
    aggregate_risk_on_sharpe: Optional[float]
    kill_criteria: Layer14KillCriteriaReport
    overfit_diagnostics: List[OverfitDiagnostic] = field(default_factory=list)
    # Folds whose OOS trade count fell below `min_oos_trades_per_fold` and
    # were therefore EXCLUDED from `aggregate_oos_sharpe`. See
    # `WalkForwardBacktester.run`'s under-powered-fold handling.
    under_powered_folds: List[int] = field(default_factory=list)
    n_folds_in_aggregate_sharpe: int = 0
    # True when the account was destroyed in at least one fold (equity
    # reached zero or below). Any Sharpe computed past that point is
    # meaningless -- see `equity_destroyed` on SegmentResult.
    any_equity_destroyed: bool = False

    def summary(self) -> str:
        sharpe_note = ""
        if self.under_powered_folds:
            sharpe_note = (
                f"  [EXCLUDED under-powered fold(s) {self.under_powered_folds}; "
                f"{self.n_folds_in_aggregate_sharpe} fold(s) in the mean]"
            )
        lines = [
            f"Walk-forward backtest: {len(self.folds)} fold(s).",
            f"Aggregate OOS Sharpe (mean of per-fold equity-curve Sharpe): "
            f"{self.aggregate_oos_sharpe:.3f}{sharpe_note}",
            f"Aggregate OOS profit factor (pooled OOS trades): {self.aggregate_oos_profit_factor:.3f}",
            f"Aggregate OOS max drawdown (worst single fold): {self.aggregate_oos_max_drawdown:.1%}",
        ]
        if self.any_equity_destroyed:
            lines.append(
                "  *** EQUITY DESTROYED in at least one fold (equity reached <= 0). "
                "Sharpe over a non-positive equity curve is meaningless and must not "
                "be compared against other runs. ***"
            )
        lines += [
            "",
            "--- Layer 14 research-phase kill criteria (NOT the Tier 1 live gate) ---",
        ]
        for c in self.kill_criteria.as_report():
            lines.append(f"  [{c['status'].upper():>13}] {c['name']}: {c['detail']}")
        lines.append("")
        lines.append("--- IS vs OOS overfit diagnostic (per fold, not a Layer 14 criterion) ---")
        for d in self.overfit_diagnostics:
            lines.append(f"  fold {d.fold}: {d.detail}")
        return "\n".join(lines)


class WalkForwardBacktester:
    def __init__(
        self,
        strategy_factory: Callable[[], SignalStrategy],
        timeframe: str,
        cost_model_kwargs: Optional[dict] = None,
        init_cash: float = 10_000.0,
        train_size: Optional[pd.DateOffset] = None,
        val_size: Optional[pd.DateOffset] = None,
        test_size: Optional[pd.DateOffset] = None,
        step: Optional[pd.DateOffset] = None,
        min_folds: int = 4,
        regime_col: Optional[str] = None,
        risk_on_label: str = "Risk-On",
        oos_sharpe_threshold: float = 0.5,
        oos_profit_factor_threshold: float = 1.2,
        oos_max_dd_threshold: float = 0.25,
        is_oos_ratio_threshold: float = 0.6,
        ensemble_weight_delta_threshold: float = 0.20,
        ensemble_consecutive_fold_threshold: int = 2,
        raise_on_kill_criteria_failure: bool = True,
        min_oos_trades_per_fold: int = 200,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.timeframe = timeframe
        self.bar_timedelta = pd.Timedelta(timeframe)
        if self.bar_timedelta <= pd.Timedelta(0):
            raise ValueError("timeframe must resolve to a positive duration.")
        self.bars_per_year = pd.Timedelta(days=365.25) / self.bar_timedelta
        self.cost_model_kwargs = cost_model_kwargs or {}
        self.init_cash = init_cash
        # MASTER_CONTEXT defaults (2y / 6mo / 6mo / 3mo) applied here rather
        # than as literal call-time defaults in the signature, so the same
        # pd.DateOffset default isn't constructed once at import time and
        # reused (harmless for an immutable DateOffset, but avoided anyway).
        self.train_size = train_size if train_size is not None else pd.DateOffset(years=2)
        self.val_size = val_size if val_size is not None else pd.DateOffset(months=6)
        self.test_size = test_size if test_size is not None else pd.DateOffset(months=6)
        self.step = step if step is not None else pd.DateOffset(months=3)
        self.min_folds = min_folds
        self.regime_col = regime_col
        self.risk_on_label = risk_on_label
        self.oos_sharpe_threshold = oos_sharpe_threshold
        self.oos_profit_factor_threshold = oos_profit_factor_threshold
        self.oos_max_dd_threshold = oos_max_dd_threshold
        self.is_oos_ratio_threshold = is_oos_ratio_threshold
        self.ensemble_weight_delta_threshold = ensemble_weight_delta_threshold
        self.ensemble_consecutive_fold_threshold = ensemble_consecutive_fold_threshold
        self.raise_on_kill_criteria_failure = raise_on_kill_criteria_failure
        # Exact MASTER_CONTEXT number: the hypothesis requires "a minimum
        # 200-trade out-of-sample sample". Folds below this are excluded from
        # the aggregate Sharpe mean (see `run`) and reported as under-powered.
        if min_oos_trades_per_fold < 0:
            raise ValueError("min_oos_trades_per_fold must be non-negative.")
        self.min_oos_trades_per_fold = min_oos_trades_per_fold

    def run(self, df: pd.DataFrame) -> BacktestReport:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df index must be a DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("df index must be sorted ascending.")
        if not df.index.is_unique:
            raise ValueError(
                "df index must be unique -- this harness backtests one instrument per "
                "run() call. For a pooled multi-asset frame, split by asset and call "
                "run() once per asset."
            )
        required = {"close", "high", "low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df missing required columns: {sorted(missing)}")
        if self.regime_col is not None and self.regime_col not in df.columns:
            raise ValueError(
                f"regime_col={self.regime_col!r} was specified but is not a column of df."
            )

        windows = generate_walk_forward_windows(
            df.index,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            step=self.step,
            min_folds=self.min_folds,
        )

        fold_results: List[FoldBacktestResult] = []
        for w in windows:
            fold_results.append(self._run_fold(df, w))

        # UNDER-POWERED FOLD EXCLUSION (2026-08-14). MASTER_CONTEXT's
        # hypothesis requires "a minimum 200-trade out-of-sample sample". A
        # fold that produces far fewer than that yields a Sharpe dominated by
        # noise, and averaging it into the aggregate silently corrupts the
        # headline number: the 2026-08-14 ablation had folds of 46 and 12
        # trades feeding the same mean as folds of 1,215. Those folds are
        # still reported individually -- they are excluded only from the
        # aggregate Sharpe MEAN, where a per-fold statistic with no sample
        # behind it gets equal weight to a well-powered one.
        #
        # Pooled profit factor deliberately still uses ALL trades: pooling is
        # sample-weighted by construction, so a 12-trade fold contributes 12
        # trades rather than an equal-weighted statistic. Max drawdown also
        # keeps every fold -- a drawdown that happened, happened.
        under_powered = [
            f.fold for f in fold_results
            if f.out_of_sample.n_trades < self.min_oos_trades_per_fold
        ]
        eligible = [
            f.out_of_sample.sharpe for f in fold_results
            if f.out_of_sample.n_trades >= self.min_oos_trades_per_fold
        ]
        finite_eligible = [s for s in eligible if not np.isnan(s)]
        aggregate_oos_sharpe = float(np.mean(finite_eligible)) if finite_eligible else float("nan")
        any_equity_destroyed = any(f.out_of_sample.equity_destroyed for f in fold_results)

        pooled_oos_trades = pd.concat(
            [f.out_of_sample.trades for f in fold_results if not f.out_of_sample.trades.empty],
            ignore_index=True,
        ) if any(not f.out_of_sample.trades.empty for f in fold_results) else pd.DataFrame(columns=["net_pnl"])
        aggregate_oos_pf = metrics.profit_factor(pooled_oos_trades["net_pnl"]) if "net_pnl" in pooled_oos_trades else float("nan")

        oos_max_dds = [f.out_of_sample.max_drawdown for f in fold_results]
        finite_dds = [d for d in oos_max_dds if not np.isnan(d)]
        aggregate_oos_max_dd = float(np.max(finite_dds)) if finite_dds else float("nan")

        aggregate_risk_on_sharpe = (
            self._pooled_risk_on_sharpe(fold_results) if self.regime_col is not None else None
        )

        fold_ensemble_weights = [f.ensemble_weights for f in fold_results]

        kill_report = evaluate_layer14_kill_criteria(
            oos_sharpe=aggregate_oos_sharpe,
            oos_profit_factor=aggregate_oos_pf,
            oos_max_dd=aggregate_oos_max_dd,
            risk_on_sharpe=aggregate_risk_on_sharpe,
            aggregate_sharpe=self._pooled_aggregate_trade_sharpe(fold_results) if self.regime_col is not None else None,
            fold_ensemble_weights=fold_ensemble_weights,
            oos_sharpe_threshold=self.oos_sharpe_threshold,
            oos_profit_factor_threshold=self.oos_profit_factor_threshold,
            oos_max_dd_threshold=self.oos_max_dd_threshold,
            ensemble_weight_delta_threshold=self.ensemble_weight_delta_threshold,
            ensemble_consecutive_fold_threshold=self.ensemble_consecutive_fold_threshold,
        )

        report = BacktestReport(
            folds=fold_results,
            aggregate_oos_sharpe=aggregate_oos_sharpe,
            aggregate_oos_profit_factor=aggregate_oos_pf,
            aggregate_oos_max_drawdown=aggregate_oos_max_dd,
            aggregate_risk_on_sharpe=aggregate_risk_on_sharpe,
            kill_criteria=kill_report,
            overfit_diagnostics=[f.overfit for f in fold_results],
            under_powered_folds=under_powered,
            n_folds_in_aggregate_sharpe=len(finite_eligible),
            any_equity_destroyed=any_equity_destroyed,
        )

        if self.raise_on_kill_criteria_failure and kill_report.any_failed:
            raise Layer14KillCriteriaFailed(report)

        return report

    # ------------------------------------------------------------------ #
    # Per-fold execution
    # ------------------------------------------------------------------ #

    def _run_fold(self, df: pd.DataFrame, w: FoldWindow) -> FoldBacktestResult:
        train_df = df.loc[w.train_mask(df.index)]
        val_df = df.loc[w.val_mask(df.index)]
        test_df = df.loc[w.test_mask(df.index)]
        self._assert_isolated(train_df, val_df, test_df, w.fold)

        strategy = self.strategy_factory()
        strategy.fit(train_df, val_df)  # test_df is never passed to fit()

        cost_model = TransactionCostModel(**self.cost_model_kwargs).fit(train_df)

        is_df = pd.concat([train_df, val_df])
        is_positions = strategy.generate_positions(is_df)
        oos_positions = strategy.generate_positions(test_df)

        is_result = self._run_segment(is_df, is_positions, cost_model, w.fold, "in_sample")
        oos_result = self._run_segment(test_df, oos_positions, cost_model, w.fold, "out_of_sample")

        overfit = evaluate_is_oos_overfit(
            w.fold, is_result.sharpe, oos_result.sharpe, self.is_oos_ratio_threshold
        )

        risk_on_sharpe = None
        if self.regime_col is not None and not oos_result.trades.empty:
            # Annotate each OOS trade with its entry-time regime label, both
            # to compute this fold's own Risk-On Sharpe and so the label
            # survives into `_pooled_risk_on_sharpe` when trades from every
            # fold are pooled after the loop.
            oos_result.trades["regime_at_entry"] = (
                test_df[self.regime_col].reindex(oos_result.trades["Entry Timestamp"]).to_numpy()
            )
            risk_on_returns = oos_result.trades.loc[
                oos_result.trades["regime_at_entry"] == self.risk_on_label, "net_return"
            ]
            if len(risk_on_returns) >= 2:
                trades_per_year = self._trades_per_year_estimate(oos_result.trades)
                risk_on_sharpe = metrics.sharpe_ratio_from_trade_returns(risk_on_returns, trades_per_year)

        return FoldBacktestResult(
            fold=w.fold,
            window=w,
            in_sample=is_result,
            out_of_sample=oos_result,
            overfit=overfit,
            risk_on_sharpe=risk_on_sharpe,
            ensemble_weights=strategy.ensemble_weights(),
            diagnostics=strategy.diagnostics(),
        )

    def _run_segment(
        self, segment_df: pd.DataFrame, positions: pd.Series, cost_model: TransactionCostModel,
        fold: int, segment_name: str,
    ) -> SegmentResult:
        positions = positions.reindex(segment_df.index).fillna(0.0)
        fees, slippage = cost_model.per_bar_fees_and_slippage(segment_df, positions)

        changed = positions.ne(positions.shift(1).fillna(0.0))
        order_size = positions.where(changed, other=np.nan)

        empty_trades_cols = [
            "Entry Timestamp", "Exit Timestamp", "Size", "Avg Entry Price", "PnL",
            "net_pnl", "net_return", "gap_cost",
        ]

        if not changed.any():
            return SegmentResult(
                segment=segment_name, fold=fold,
                trades=pd.DataFrame(columns=empty_trades_cols),
                equity_curve=pd.Series(self.init_cash, index=segment_df.index),
                returns=pd.Series(dtype=float), sharpe=float("nan"), profit_factor=float("nan"),
                max_drawdown=float("nan"), ev_per_trade=float("nan"), n_trades=0,
            )

        pf = vbt.Portfolio.from_orders(
            close=segment_df["close"],
            size=order_size,
            size_type="targetpercent",
            direction="both",
            fees=fees,
            slippage=slippage,
            freq=self.timeframe,
            init_cash=self.init_cash,
        )
        trades = pf.trades.records_readable
        if trades.empty:
            return SegmentResult(
                segment=segment_name, fold=fold,
                trades=pd.DataFrame(columns=empty_trades_cols),
                equity_curve=pd.Series(self.init_cash, index=segment_df.index),
                returns=pd.Series(dtype=float), sharpe=float("nan"), profit_factor=float("nan"),
                max_drawdown=float("nan"), ev_per_trade=float("nan"), n_trades=0,
            )

        trades = cost_model.apply_gap_cost_to_trades(trades, segment_df.index)

        equity_curve = metrics.equity_curve_from_trade_pnls(
            trades["net_pnl"], trades["Exit Timestamp"], segment_df.index, self.init_cash
        )
        # RUIN DETECTION (2026-08-14). The equity curve is a cumulative sum
        # of realized trade PnL with no bankruptcy floor, so it can and does
        # cross zero. Once it does, `pct_change()` is dividing by a
        # zero-or-negative denominator and every subsequent "return" -- and
        # any Sharpe built from them -- is arithmetically meaningless. The
        # 2026-08-14 ablation reported maxDD of 1.023 and 1.235 (equity below
        # zero) alongside a Sharpe that looked BETTER than the solvent
        # baseline, which is exactly the trap this flag exists to expose.
        equity_destroyed = bool((equity_curve <= 0).any())

        returns = equity_curve.pct_change().dropna()
        sharpe = metrics.sharpe_ratio_from_returns(returns, self.bars_per_year)
        if equity_destroyed:
            # Refuse to publish a number that cannot be compared. maxDD and
            # profit factor remain valid and are what should be read instead.
            sharpe = float("nan")
        pf_ratio = metrics.profit_factor(trades["net_pnl"])
        max_dd = metrics.max_drawdown(equity_curve)
        notionals = trades["Size"].abs() * trades["Avg Entry Price"]
        ev = metrics.expected_value_per_trade(trades["net_pnl"], notionals)

        return SegmentResult(
            segment=segment_name, fold=fold, trades=trades, equity_curve=equity_curve,
            returns=returns, sharpe=sharpe, profit_factor=pf_ratio, max_drawdown=max_dd,
            ev_per_trade=ev, n_trades=len(trades), equity_destroyed=equity_destroyed,
        )

    # ------------------------------------------------------------------ #
    # Regime-conditional (Risk-On vs aggregate) helpers
    # ------------------------------------------------------------------ #

    def _pooled_all_oos_trades(self, fold_results: List["FoldBacktestResult"]) -> pd.DataFrame:
        frames = [f.out_of_sample.trades for f in fold_results if not f.out_of_sample.trades.empty]
        if not frames:
            return pd.DataFrame(columns=["net_return", "Entry Timestamp", "Exit Timestamp"])
        return pd.concat(frames, ignore_index=True)

    def _pooled_risk_on_sharpe(self, fold_results: List["FoldBacktestResult"]) -> Optional[float]:
        """Pools every fold's OOS trades tagged Risk-On at entry (see
        `_run_fold`, which stamps `regime_at_entry` onto each fold's trades
        frame) into one trade-return Sharpe. Pooling discrete trades across
        folds is safe even though folds' calendar test windows overlap
        (see windows.py docstring) -- each fold's trades come from an
        independently-fit model, they are not duplicates of each other.

        Annualized using the trade frequency of the FULL pooled OOS trade
        set (both regimes combined), not bar frequency -- annualizing a
        trade-return Sharpe by bars/year rather than trades/year silently
        inflates it by sqrt(bars_per_year / trades_per_year), which is
        enormous for anything but a strategy that trades every single bar.
        The same denominator is used for the aggregate side in
        `_pooled_aggregate_trade_sharpe` so the two stay directly comparable.
        """
        all_trades = self._pooled_all_oos_trades(fold_results)
        if "regime_at_entry" not in all_trades.columns:
            return None
        risk_on_returns = all_trades.loc[all_trades["regime_at_entry"] == self.risk_on_label, "net_return"]
        if len(risk_on_returns) < 2:
            return None
        trades_per_year = self._trades_per_year_estimate(all_trades)
        return metrics.sharpe_ratio_from_trade_returns(risk_on_returns, trades_per_year)

    def _pooled_aggregate_trade_sharpe(self, fold_results: List["FoldBacktestResult"]) -> Optional[float]:
        """Same pooling logic as `_pooled_risk_on_sharpe` but over ALL OOS
        trades regardless of regime -- the directly comparable 'aggregate'
        side of the Risk-On-vs-aggregate Layer 14 criterion (both computed
        via the same trade-return-Sharpe methodology and the same
        trades-per-year denominator, so the comparison is apples-to-apples)."""
        all_trades = self._pooled_all_oos_trades(fold_results)
        if len(all_trades) < 2:
            return None
        trades_per_year = self._trades_per_year_estimate(all_trades)
        return metrics.sharpe_ratio_from_trade_returns(all_trades["net_return"], trades_per_year)

    @staticmethod
    def _trades_per_year_estimate(trades: pd.DataFrame) -> float:
        """Estimates trading frequency from the span between the earliest
        entry and latest exit in `trades`, used to annualize a trade-return
        Sharpe correctly (see `_pooled_risk_on_sharpe`)."""
        if len(trades) < 2:
            return float("nan")
        span_days = (trades["Exit Timestamp"].max() - trades["Entry Timestamp"].min()).total_seconds() / 86400.0
        span_days = max(span_days, 1.0)
        return len(trades) * 365.25 / span_days

    @staticmethod
    def _assert_isolated(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> None:
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Fold {fold}: empty train/val/test window -- data does not cover this fold's range.")
        if train_df.index.max() >= val_df.index.min():
            raise AssertionError(f"Fold {fold}: train/val temporal overlap detected.")
        if val_df.index.max() >= test_df.index.min():
            raise AssertionError(f"Fold {fold}: val/test temporal overlap detected.")
