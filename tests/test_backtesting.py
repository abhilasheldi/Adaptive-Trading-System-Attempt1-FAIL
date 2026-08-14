"""
Tests for the walk-forward backtesting harness (src/backtesting).

Covers: calendar-based fold generation, the transaction cost model, cost-
adjusted metrics, the five Layer 14 kill criteria (each reported
independently) plus the IS/OOS overfit diagnostic, and a full
`WalkForwardBacktester.run()` at the MASTER_CONTEXT-specified window sizes
(2y train / 6mo val / 6mo test / 3mo step / min 4 folds).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    Layer14KillCriteriaFailed,
    TransactionCostModel,
    WalkForwardBacktester,
    evaluate_ensemble_weight_stability,
    evaluate_layer14_kill_criteria,
    generate_walk_forward_windows,
)
from src.backtesting import metrics as bt_metrics


# ------------------------------------------------------------------------ #
# Synthetic data helpers
# ------------------------------------------------------------------------ #


def make_synthetic_ohlcv(n_hours: int, seed: int = 0, with_regime: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_hours, freq="1h")
    ret = rng.normal(0.00005, 0.006, n_hours)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + rng.uniform(0.0005, 0.004, n_hours))
    low = close * (1 - rng.uniform(0.0005, 0.004, n_hours))
    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)
    if with_regime:
        trailing_ret = pd.Series(close, index=idx).pct_change(24 * 30).fillna(0.0)
        df["macro_regime"] = np.where(trailing_ret > 0, "Risk-On", "Risk-Off")
    return df


class MomentumStrategy:
    """Deterministic strategy: long/short based on a short-vs-long moving
    average spread exceeding a threshold fit from train+val data."""

    def __init__(self) -> None:
        self._threshold: Optional[float] = None
        self._weights: Optional[Dict[str, float]] = None
        self.fit_index: Optional[pd.DatetimeIndex] = None  # instrumentation for leak tests

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        fit_df = pd.concat([train_df, val_df])
        self.fit_index = fit_df.index  # records exactly what this strategy was fit on
        short_ma = fit_df["close"].rolling(24).mean()
        long_ma = fit_df["close"].rolling(24 * 7).mean()
        spread = ((short_ma - long_ma) / long_ma).dropna()
        self._threshold = float(spread.abs().quantile(0.6)) if len(spread) else 0.01
        trend_w = float(np.clip(spread.std() * 50, 0.1, 0.9)) if len(spread) else 0.5
        self._weights = {"trend": trend_w, "mean_reversion": 1.0 - trend_w}

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        short_ma = df["close"].rolling(24, min_periods=1).mean()
        long_ma = df["close"].rolling(24 * 7, min_periods=1).mean()
        spread = (short_ma - long_ma) / long_ma
        pos = pd.Series(0.0, index=df.index)
        pos[spread > self._threshold] = 0.3
        pos[spread < -self._threshold] = -0.3
        return pos

    def ensemble_weights(self) -> Optional[Dict[str, float]]:
        return self._weights

    def diagnostics(self) -> Optional[Dict]:
        return None


class FlatStrategy:
    """Never trades. Used to cheaply test bookkeeping (ensemble weights,
    fold isolation) without paying for real order simulation."""

    def __init__(self, weights: Dict[str, float]) -> None:
        self._weights = weights
        self.fit_index: Optional[pd.DatetimeIndex] = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.fit_index = pd.concat([train_df, val_df]).index

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index)

    def ensemble_weights(self) -> Optional[Dict[str, float]]:
        return self._weights

    def diagnostics(self) -> Optional[Dict]:
        return None


@pytest.fixture(scope="module")
def synthetic_ohlcv() -> pd.DataFrame:
    n_hours = 24 * 365 * 4 + 24 * 200  # ~4.5 years, enough for >= 4 spec-sized folds
    return make_synthetic_ohlcv(n_hours, seed=42)


@pytest.fixture(scope="module")
def backtest_report(synthetic_ohlcv):
    # raise_on_kill_criteria_failure=False: this fixture is used by tests that
    # inspect reporting mechanics on a strategy known to fail some criteria on
    # pure random-walk data (see test_harness_default_raises_on_kill_criteria_failure
    # for the halt-behavior test itself, which deliberately leaves the default on).
    bt = WalkForwardBacktester(
        strategy_factory=MomentumStrategy,
        timeframe="1h",
        min_oos_trades_per_fold=0,  # synthetic folds are far below the 200-trade production minimum
        regime_col="macro_regime",
        risk_on_label="Risk-On",
        raise_on_kill_criteria_failure=False,
    )
    return bt.run(synthetic_ohlcv), bt


# ------------------------------------------------------------------------ #
# windows.py
# ------------------------------------------------------------------------ #


def test_windows_generates_minimum_folds_at_spec_sizes():
    idx = pd.date_range("2018-01-01", periods=24 * 365 * 4 + 24 * 200, freq="1h")
    windows = generate_walk_forward_windows(idx)
    assert len(windows) >= 4
    for w in windows:
        assert w.train_end == w.val_start
        assert w.val_end == w.test_start
        assert w.train_start + pd.DateOffset(years=2) == w.train_end
        assert w.train_end + pd.DateOffset(months=6) == w.val_end
        assert w.val_end + pd.DateOffset(months=6) == w.test_end


def test_windows_raises_when_insufficient_folds():
    idx = pd.date_range("2018-01-01", periods=24 * 365 * 2, freq="1h")  # only ~2 years
    with pytest.raises(ValueError, match="complete walk-forward fold"):
        generate_walk_forward_windows(idx)


def test_windows_enforces_minimum_train_floor():
    idx = pd.date_range("2018-01-01", periods=24 * 365 * 4, freq="1h")
    with pytest.raises(ValueError, match="minimum 2-year training window"):
        generate_walk_forward_windows(idx, train_size=pd.DateOffset(months=6))


def test_windows_masks_partition_without_overlap():
    idx = pd.date_range("2018-01-01", periods=24 * 365 * 4 + 24 * 200, freq="1h")
    windows = generate_walk_forward_windows(idx)
    for w in windows:
        train_mask = w.train_mask(idx)
        val_mask = w.val_mask(idx)
        test_mask = w.test_mask(idx)
        assert not np.any(train_mask & val_mask)
        assert not np.any(val_mask & test_mask)
        assert not np.any(train_mask & test_mask)
        assert idx[train_mask].max() < idx[val_mask].min()
        assert idx[val_mask].max() < idx[test_mask].min()


# ------------------------------------------------------------------------ #
# costs.py
# ------------------------------------------------------------------------ #


def test_cost_model_rejects_below_75th_percentile_spread():
    with pytest.raises(ValueError, match="75th-percentile"):
        TransactionCostModel(spread_percentile=50.0)


def test_cost_model_gap_detection_fires_on_injected_gaps():
    rng = np.random.default_rng(0)
    n = 5000
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    is_weekend = np.isin(idx.dayofweek, [5, 6])
    weekend_starts = np.where(is_weekend & ~np.r_[False, is_weekend[:-1]])[0]
    for i in weekend_starts:
        close[i:] += rng.choice([-1, 1]) * 8.0
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)

    model = TransactionCostModel().fit(df)
    assert model.params.gap_frequency_weekend > 0
    assert model.params.gap_magnitude_weekend_frac > 0

    # A random-walk-only series (no injected gaps) should show ~zero gap risk.
    close_smooth = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df_smooth = pd.DataFrame(
        {"close": close_smooth, "high": close_smooth + 0.5, "low": close_smooth - 0.5}, index=idx
    )
    model_smooth = TransactionCostModel().fit(df_smooth)
    assert model_smooth.params.gap_frequency_weekend == 0.0


def test_cost_model_slippage_scales_with_order_size_and_candle_range():
    """Slippage must be charged on the size of the ORDER being traded (the
    change in target position), not the resulting post-order holding level.
    Regression test for a bug where an exit-to-flat order (target -> 0)
    priced in zero slippage regardless of how large the position being
    closed was, since the old formula scaled slippage by the raw target
    level rather than the traded delta."""
    idx = pd.date_range("2020-01-01", periods=4, freq="1h")
    close = np.full(4, 100.0)
    high = close + np.array([0.5, 0.5, 2.0, 2.0])  # narrow range bars 0-1, wide range bars 2-3
    low = close - np.array([0.5, 0.5, 2.0, 2.0])
    df = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)
    model = TransactionCostModel().fit(df)  # slippage_coefficient default 0.5

    small_order = pd.Series([0.1, 0.1, 0.0, 0.0], index=idx)  # enter 0.1 at bar0, exit at bar2
    large_order = pd.Series([0.9, 0.9, 0.0, 0.0], index=idx)  # enter 0.9 at bar0, exit at bar2
    _, slip_small = model.per_bar_fees_and_slippage(df, small_order)
    _, slip_large = model.per_bar_fees_and_slippage(df, large_order)

    # Hand calc: slippage = 0.5 * |order_delta| * range_frac.
    # bar0 (entry, range_frac=1/100=0.01): small=0.5*0.1*0.01=0.0005, large=0.5*0.9*0.01=0.0045
    assert slip_small[0] == pytest.approx(0.0005)
    assert slip_large[0] == pytest.approx(0.0045)
    assert slip_large[0] > slip_small[0] > 0
    # bar1: position unchanged (holding, no order) -> zero slippage regardless
    # of position size -- slippage is charged on trades, not on holdings.
    assert slip_small[1] == 0.0
    assert slip_large[1] == 0.0
    # bar2 (exit-to-flat, range_frac=4/100=0.04): small=0.5*0.1*0.04=0.002,
    # large=0.5*0.9*0.04=0.018 -- both nonzero, which the pre-fix formula got wrong.
    assert slip_small[2] == pytest.approx(0.002)
    assert slip_large[2] == pytest.approx(0.018)

    # Same order size (0.5), entered at a narrow-range bar vs a wide-range
    # bar -- wider candle range must cost more slippage for the same size.
    narrow_entry = pd.Series([0.5, 0.0, 0.0, 0.0], index=idx)
    wide_entry = pd.Series([0.0, 0.0, 0.5, 0.0], index=idx)
    _, slip_narrow = model.per_bar_fees_and_slippage(df, narrow_entry)
    _, slip_wide = model.per_bar_fees_and_slippage(df, wide_entry)
    assert slip_narrow[0] == pytest.approx(0.5 * 0.5 * 0.01)
    assert slip_wide[2] == pytest.approx(0.5 * 0.5 * 0.04)
    assert slip_wide[2] > slip_narrow[0]


def test_cost_model_gap_cost_reduces_trade_pnl():
    idx = pd.date_range("2020-01-01", periods=5000, freq="1h")  # spans many weekends
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    is_weekend = np.isin(idx.dayofweek, [5, 6])
    weekend_starts = np.where(is_weekend & ~np.r_[False, is_weekend[:-1]])[0]
    for i in weekend_starts:
        close[i:] += rng.choice([-1, 1]) * 8.0
    df = pd.DataFrame({"close": close, "high": close + 0.5, "low": close - 0.5}, index=idx)
    model = TransactionCostModel().fit(df)

    # One trade held across an entire weekend, one trade held for the same
    # duration entirely on a weekday -- same PnL input, gap cost should only
    # discount the weekend-holding trade.
    weekend_entry = idx[weekend_starts[2] - 5]
    weekend_exit = idx[weekend_starts[2] + 48]
    weekday_entry = idx[1000]
    weekday_exit = idx[1000 + 53]
    trades = pd.DataFrame(
        {
            "Entry Timestamp": [weekend_entry, weekday_entry],
            "Exit Timestamp": [weekend_exit, weekday_exit],
            "Size": [1.0, 1.0],
            "Avg Entry Price": [100.0, 100.0],
            "PnL": [10.0, 10.0],
        }
    )
    adjusted = model.apply_gap_cost_to_trades(trades, idx)
    assert adjusted.loc[0, "gap_cost"] > adjusted.loc[1, "gap_cost"]
    assert adjusted.loc[0, "net_pnl"] < adjusted.loc[0, "PnL"]


# ------------------------------------------------------------------------ #
# metrics.py
# ------------------------------------------------------------------------ #


def test_metrics_profit_factor_and_ev_and_drawdown():
    trade_pnls = pd.Series([100.0, -50.0, 30.0, -20.0, 80.0])
    notionals = pd.Series([1000.0] * 5)
    assert bt_metrics.profit_factor(trade_pnls) == pytest.approx(3.0)
    assert bt_metrics.expected_value_per_trade(trade_pnls, notionals) == pytest.approx(0.028)

    equity = pd.Series([100, 110, 105, 90, 95, 120], index=pd.date_range("2020-01-01", periods=6, freq="D"))
    assert bt_metrics.max_drawdown(equity) == pytest.approx((110 - 90) / 110)


def test_metrics_sharpe_higher_mean_gives_higher_sharpe():
    idx = pd.date_range("2020-01-01", periods=500, freq="1h")
    rng = np.random.default_rng(5)
    low_mean = pd.Series(rng.normal(0.0001, 0.01, len(idx)), index=idx)
    high_mean = pd.Series(rng.normal(0.001, 0.01, len(idx)), index=idx)
    s_low = bt_metrics.sharpe_ratio_from_returns(low_mean, periods_per_year=24 * 365.25)
    s_high = bt_metrics.sharpe_ratio_from_returns(high_mean, periods_per_year=24 * 365.25)
    assert s_high > s_low


# ------------------------------------------------------------------------ #
# kill_criteria.py -- explicit per-criterion reporting
# ------------------------------------------------------------------------ #


def test_kill_criteria_report_every_criterion_independently_not_aggregated():
    report = evaluate_layer14_kill_criteria(
        oos_sharpe=0.3, oos_profit_factor=1.5, oos_max_dd=0.10,
        risk_on_sharpe=None, aggregate_sharpe=None,
        fold_ensemble_weights=[None, None],
    )
    statuses = {c["name"]: c["status"] for c in report.as_report()}
    assert len(statuses) == 5
    # A mixed pass/fail/not_evaluated result must be visible per-criterion,
    # not collapsed -- these two must disagree despite both being "in" the report.
    assert statuses["layer14_oos_sharpe"] == "fail"
    assert statuses["layer14_oos_profit_factor"] == "pass"
    assert statuses["layer14_risk_on_regime_vs_aggregate"] == "not_evaluated"
    assert statuses["layer14_ensemble_weight_stability"] == "not_evaluated"


def test_ensemble_instability_detects_more_than_two_consecutive_unstable_folds():
    stable = [{"a": 0.5, "b": 0.5}] * 5
    result_stable = evaluate_ensemble_weight_stability(stable)
    assert result_stable.status == "pass"

    volatile = [
        {"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}, {"a": 0.9, "b": 0.1},
        {"a": 0.1, "b": 0.9}, {"a": 0.9, "b": 0.1},
    ]
    result_volatile = evaluate_ensemble_weight_stability(volatile)
    assert result_volatile.status == "fail"
    assert result_volatile.measured_value == 4  # 4 consecutive unstable transitions

    result_not_evaluated = evaluate_ensemble_weight_stability([{"a": 0.5}, None])
    assert result_not_evaluated.status == "not_evaluated"


# ------------------------------------------------------------------------ #
# harness.py -- full end-to-end walk-forward run at MASTER_CONTEXT window sizes
# ------------------------------------------------------------------------ #


def test_harness_runs_end_to_end_at_spec_window_sizes(backtest_report):
    report, _bt = backtest_report
    assert len(report.folds) >= 4  # MASTER_CONTEXT minimum

    for f in report.folds:
        assert f.window.train_end == f.window.val_start
        assert f.window.val_end == f.window.test_start
        # Both segments should have actually traded in this synthetic setup.
        assert f.in_sample.n_trades > 0
        assert f.out_of_sample.n_trades > 0


def test_harness_reports_all_five_layer14_criteria_explicitly(backtest_report):
    report, _bt = backtest_report
    statuses = {c["name"]: c["status"] for c in report.kill_criteria.as_report()}
    assert len(statuses) == 5
    for name in [
        "layer14_oos_sharpe", "layer14_oos_profit_factor", "layer14_oos_max_drawdown",
        "layer14_risk_on_regime_vs_aggregate", "layer14_ensemble_weight_stability",
    ]:
        assert name in statuses
        assert statuses[name] in ("pass", "fail", "not_evaluated")
    # Regime column was supplied and every fold traded, so this criterion
    # must actually have been evaluated, not silently skipped.
    assert statuses["layer14_risk_on_regime_vs_aggregate"] != "not_evaluated"
    assert statuses["layer14_ensemble_weight_stability"] != "not_evaluated"


def test_harness_overfit_diagnostic_present_per_fold_and_distinct_from_kill_criteria(backtest_report):
    report, _bt = backtest_report
    assert len(report.overfit_diagnostics) == len(report.folds)
    kill_criterion_names = {c["name"] for c in report.kill_criteria.as_report()}
    assert "layer14_is_oos_overfit" not in kill_criterion_names  # never smuggled in as a 6th Layer 14 criterion


def test_harness_never_fits_strategy_on_test_partition(synthetic_ohlcv):
    """Structural leak guard: the exact set of timestamps every fold's
    strategy.fit() saw must never intersect that same fold's test window."""
    bt = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h", min_oos_trades_per_fold=0,
        raise_on_kill_criteria_failure=False
    )
    fitted_strategies = []

    class RecordingFactory:
        def __call__(self):
            s = MomentumStrategy()
            fitted_strategies.append(s)
            return s

    bt.strategy_factory = RecordingFactory()
    report = bt.run(synthetic_ohlcv)

    assert len(fitted_strategies) == len(report.folds)
    for strategy, fold in zip(fitted_strategies, report.folds):
        assert strategy.fit_index.max() < fold.window.test_start
        assert not strategy.fit_index.to_series().between(
            fold.window.test_start, fold.window.test_end
        ).any()


def test_harness_ensemble_instability_detected_end_to_end(synthetic_ohlcv):
    """Positive control: a strategy whose ensemble weights swing every fold
    must trip the Layer 14 ensemble-stability criterion through the real
    harness, not just the standalone kill_criteria function."""
    weight_cycle = [{"trend": 0.9, "mean_reversion": 0.1}, {"trend": 0.1, "mean_reversion": 0.9}]
    counter = {"n": 0}

    def volatile_factory():
        w = weight_cycle[counter["n"] % 2]
        counter["n"] += 1
        return FlatStrategy(w)

    bt = WalkForwardBacktester(
        strategy_factory=volatile_factory, timeframe="1h", min_oos_trades_per_fold=0,
        raise_on_kill_criteria_failure=False
    )
    report = bt.run(synthetic_ohlcv)
    assert len(report.folds) >= 4

    statuses = {c["name"]: c["status"] for c in report.kill_criteria.as_report()}
    assert statuses["layer14_ensemble_weight_stability"] == "fail"


def test_harness_higher_taker_fee_does_not_improve_ev(synthetic_ohlcv):
    """Sanity/monotonicity check that costs are actually being applied
    before EV is computed: raising the taker fee must not improve (and
    should typically worsen) EV per trade on the same signal."""

    def make_bt(fee_bps: float) -> WalkForwardBacktester:
        return WalkForwardBacktester(
            strategy_factory=MomentumStrategy,
            timeframe="1h",
            min_oos_trades_per_fold=0,
            cost_model_kwargs={"taker_fee_frac": fee_bps / 10_000.0},
            min_folds=4,
            raise_on_kill_criteria_failure=False,
        )

    low_fee_report = make_bt(1.0).run(synthetic_ohlcv)
    high_fee_report = make_bt(50.0).run(synthetic_ohlcv)

    low_fee_ev = np.nanmean([f.out_of_sample.ev_per_trade for f in low_fee_report.folds])
    high_fee_ev = np.nanmean([f.out_of_sample.ev_per_trade for f in high_fee_report.folds])
    assert high_fee_ev < low_fee_ev


def test_harness_default_raises_on_kill_criteria_failure(synthetic_ohlcv):
    """(3) Kill criteria must actually HALT, not just be logged in a report
    nobody is forced to check. MomentumStrategy on pure random-walk data
    fails OOS Sharpe/profit-factor (see the not-raising fixture above) --
    with the default raise_on_kill_criteria_failure=True, run() must raise
    instead of silently returning a report, and the exception must carry
    the full per-criterion report so a caller can see exactly which
    criteria failed."""
    bt = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h",
        min_oos_trades_per_fold=0,  # synthetic folds are below the 200-trade production minimum
    )
    with pytest.raises(Layer14KillCriteriaFailed) as exc_info:
        bt.run(synthetic_ohlcv)

    failure = exc_info.value
    assert failure.report.kill_criteria.any_failed is True
    statuses = {c["name"]: c["status"] for c in failure.report.kill_criteria.as_report()}
    assert statuses["layer14_oos_sharpe"] == "fail"
    assert "layer14_oos_sharpe" in str(failure)


def test_harness_does_not_raise_when_opted_out_or_when_criteria_pass(synthetic_ohlcv):
    """Two halt-behavior edge cases: raise_on_kill_criteria_failure=False
    must suppress the halt even though criteria fail (exploratory use), and
    a run where nothing fails (flat strategy: no trades -> every trade-based
    criterion is not_evaluated, not fail; constant ensemble weights -> that
    criterion passes) must not raise even with the default left on."""
    bt_suppressed = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h", min_oos_trades_per_fold=0,
        raise_on_kill_criteria_failure=False
    )
    report = bt_suppressed.run(synthetic_ohlcv)  # must not raise
    assert report.kill_criteria.any_failed is True  # criteria did fail; just wasn't raised

    def flat_factory():
        return FlatStrategy({"a": 0.5, "b": 0.5})

    bt_passing = WalkForwardBacktester(strategy_factory=flat_factory, timeframe="1h")
    report_passing = bt_passing.run(synthetic_ohlcv)  # must not raise -- nothing failed
    assert report_passing.kill_criteria.any_failed is False


# ------------------------------------------------------------------------ #
# 2026-08-14: under-powered fold exclusion + ruin detection
# ------------------------------------------------------------------------ #


def test_under_powered_folds_are_excluded_from_aggregate_sharpe(synthetic_ohlcv):
    """MASTER_CONTEXT requires a minimum 200-trade out-of-sample sample. A
    fold below that yields a Sharpe dominated by noise; averaging it into the
    aggregate silently corrupts the headline number. The 2026-08-14 ablation
    had folds of 46 and 12 trades feeding the same mean as folds of 1,215."""
    strict = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h",
        min_oos_trades_per_fold=10_000_000,  # nothing can qualify
        raise_on_kill_criteria_failure=False,
    )
    report = strict.run(synthetic_ohlcv)
    assert len(report.under_powered_folds) == len(report.folds)
    assert report.n_folds_in_aggregate_sharpe == 0
    assert np.isnan(report.aggregate_oos_sharpe)
    # Folds are still reported individually -- excluded from the MEAN only.
    assert len(report.folds) >= 4


def test_opting_out_of_the_minimum_includes_every_fold(synthetic_ohlcv):
    lenient = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h",
        min_oos_trades_per_fold=0, raise_on_kill_criteria_failure=False,
    )
    report = lenient.run(synthetic_ohlcv)
    assert report.under_powered_folds == []
    assert report.n_folds_in_aggregate_sharpe > 0


def test_pooled_profit_factor_still_uses_every_trade(synthetic_ohlcv):
    """Pooling is sample-weighted by construction, so a small fold
    contributes its trades rather than an equal-weighted statistic. Only the
    Sharpe MEAN needs the exclusion."""
    strict = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h",
        min_oos_trades_per_fold=10_000_000, raise_on_kill_criteria_failure=False,
    )
    lenient = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h",
        min_oos_trades_per_fold=0, raise_on_kill_criteria_failure=False,
    )
    a = strict.run(synthetic_ohlcv)
    b = lenient.run(synthetic_ohlcv)
    assert a.aggregate_oos_profit_factor == pytest.approx(b.aggregate_oos_profit_factor, nan_ok=True)
    assert a.aggregate_oos_max_drawdown == pytest.approx(b.aggregate_oos_max_drawdown, nan_ok=True)


def test_summary_discloses_excluded_folds(synthetic_ohlcv):
    strict = WalkForwardBacktester(
        strategy_factory=MomentumStrategy, timeframe="1h",
        min_oos_trades_per_fold=10_000_000, raise_on_kill_criteria_failure=False,
    )
    text = strict.run(synthetic_ohlcv).summary()
    assert "EXCLUDED under-powered fold" in text


def test_min_oos_trades_per_fold_rejects_negative():
    with pytest.raises(ValueError):
        WalkForwardBacktester(
            strategy_factory=MomentumStrategy, timeframe="1h", min_oos_trades_per_fold=-1
        )


def test_ruin_detection_disqualifies_sharpe_and_is_reported():
    """Sharpe over a non-positive equity curve is arithmetically meaningless
    (the pct_change denominator crosses zero). The 2026-08-14 ablation
    reported maxDD 1.023 -- equity below zero -- alongside a Sharpe that
    looked BETTER than the solvent baseline. That must never be publishable."""
    from src.backtesting import metrics

    idx = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    # Cumulative losses far exceeding init_cash -> equity crosses zero.
    pnls = pd.Series(np.full(500, -50.0), index=idx)
    eq = metrics.equity_curve_from_trade_pnls(pnls, pd.Series(idx), idx, 10_000.0)
    assert (eq <= 0).any()
    assert metrics.max_drawdown(eq) > 1.0  # >1.0 is the signature of negative equity


# ------------------------------------------------------------------------ #
# 2026-08-14: predictive-core skill diagnostic (NOT a sixth Layer 14 criterion)
# ------------------------------------------------------------------------ #


def test_no_skill_core_dominating_the_ensemble_is_flagged():
    """The 2026-08-14 ablation fitted component_b = 0.942 while that signal's
    logloss skill score sat at 0.004-0.016 -- a no-skill signal carrying 94%
    of the blend. That means there is nothing to blend, so tuning cannot
    recover it."""
    from src.backtesting.kill_criteria import evaluate_predictive_core_skill

    d = evaluate_predictive_core_skill(
        0.010,
        {"component_b": 0.942378, "component_a": 0.032093, "mtf_alignment": 0.012764,
         "macro_regime": 0.012764, "liquidity_trap": 0.0},
    )
    assert d.flagged is True
    assert d.dominant_signal == "component_b"
    assert d.dominant_weight == pytest.approx(0.942378)
    assert "revise the EDGE HYPOTHESIS" in d.detail
    assert "CANNOT recover" in d.detail


def test_a_core_with_real_skill_is_not_flagged():
    from src.backtesting.kill_criteria import evaluate_predictive_core_skill

    d = evaluate_predictive_core_skill(0.25, {"component_b": 0.6, "component_a": 0.4})
    assert d.flagged is False
    assert "measurable skill" in d.detail


def test_skill_diagnostic_reports_not_evaluated_without_a_score():
    from src.backtesting.kill_criteria import evaluate_predictive_core_skill

    d = evaluate_predictive_core_skill(float("nan"))
    assert d.flagged is False
    assert "NOT EVALUATED" in d.detail


def test_skill_diagnostic_dominant_signal_is_deterministic_on_ties():
    from src.backtesting.kill_criteria import evaluate_predictive_core_skill

    tied = {"component_b": 0.5, "component_a": 0.5}
    first = evaluate_predictive_core_skill(0.0, tied).dominant_signal
    second = evaluate_predictive_core_skill(0.0, dict(reversed(list(tied.items())))).dominant_signal
    assert first == second == "component_a"  # alphabetical tiebreak, not dict order


def test_skill_diagnostic_is_not_a_layer14_criterion(backtest_report):
    """MASTER_CONTEXT specifies exactly five Layer 14 criteria. This check is
    a diagnostic alongside them, matching how the IS/OOS overfit check is
    handled -- it must not appear in the criteria list."""
    report, _ = backtest_report
    names = {c["name"] for c in report.kill_criteria.as_report()}
    assert len(names) == 5
    assert not any("skill" in n for n in names)


# ------------------------------------------------------------------------ #
# 2026-08-14: gap-cost correction (unbounded / symmetric / double-counted)
# ------------------------------------------------------------------------ #


def _held_through_weekend(freq: str, n_days: int = 14) -> tuple:
    """One trade held across the whole span, sampled at `freq`. Used to prove
    the gap charge depends on SESSIONS crossed, not on sampling frequency."""
    idx = pd.date_range("2024-01-01", periods=int(n_days * 24 * (60 / int(freq[:-3]))) if freq.endswith("min")
                        else n_days * 24, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.001, len(idx)))
    df = pd.DataFrame({"close": close, "high": close * 1.002, "low": close * 0.998}, index=idx)
    trades = pd.DataFrame({
        "Entry Timestamp": [idx[0]], "Exit Timestamp": [idx[-1]],
        "Size": [1.0], "Avg Entry Price": [100.0], "PnL": [0.0],
    })
    return df, trades, idx


def test_gap_cost_counts_sessions_not_bars_so_it_is_timeframe_invariant():
    """THE core defect. A weekend is ONE Friday-close-to-Monday-open
    discontinuity, but 48 bars on 1h data and 192 on 15m. Charging per BAR
    made the modelled cost scale with the sampling frequency instead of the
    risk -- and grow without bound in the holding period."""
    df_1h, trades_1h, idx_1h = _held_through_weekend("1h")
    df_15m, trades_15m, idx_15m = _held_through_weekend("15min")

    cm_1h = TransactionCostModel().fit(df_1h)
    cm_15m = TransactionCostModel().fit(df_15m)
    n_1h = cm_1h._session_starts(cm_1h._is_weekend(idx_1h)).sum()
    n_15m = cm_15m._session_starts(cm_15m._is_weekend(idx_15m)).sum()

    # Same calendar span -> same number of weekend SESSIONS, regardless of freq.
    assert n_1h == n_15m
    # And strictly fewer than the raw bar count, which is what used to be charged.
    assert n_1h < cm_1h._is_weekend(idx_1h).sum()


def test_session_starts_marks_only_the_first_bar_of_each_run():
    cm = TransactionCostModel()
    mask = np.array([False, True, True, True, False, False, True, True, False])
    starts = cm._session_starts(mask)
    assert starts.tolist() == [False, True, False, False, False, False, True, False, False]
    assert starts.sum() == 2          # two sessions
    assert mask.sum() == 5            # five bars -- what used to be charged
    assert cm._session_starts(np.array([], dtype=bool)).size == 0


def test_gap_cost_does_not_grow_without_bound_in_holding_period():
    """The unbounded accumulation directly caused negative equity: on the
    audited fold it was $2,588 of a $2,696 total loss (96%)."""
    df, _, idx = _held_through_weekend("1h", n_days=60)
    cm = TransactionCostModel().fit(df)

    def cost_for(days: int) -> float:
        exit_ts = idx[min(days * 24, len(idx) - 1)]
        trades = pd.DataFrame({
            "Entry Timestamp": [idx[0]], "Exit Timestamp": [exit_ts],
            "Size": [1.0], "Avg Entry Price": [100.0], "PnL": [0.0],
        })
        out = cm.apply_gap_cost_to_trades(trades, idx)
        return float(out["gap_cost"].iloc[0] / (1.0 * 100.0))

    # An expected cost may never exceed the position that incurs it.
    for days in (7, 30, 59):
        assert 0.0 <= cost_for(days) <= 1.0


def test_gap_cost_charges_only_the_adverse_fraction_of_a_symmetric_move():
    """`_fit_gap_stats` measures E[|jump|] -- symmetric, and already inside
    vectorbt's PnL. Only the execution-degradation share is a real cost."""
    df, trades, idx = _held_through_weekend("1h")
    full = TransactionCostModel(gap_adverse_fraction=1.0).fit(df)
    half = TransactionCostModel(gap_adverse_fraction=0.5).fit(df)
    none = TransactionCostModel(gap_adverse_fraction=0.0).fit(df)

    c_full = float(full.apply_gap_cost_to_trades(trades, idx)["gap_cost"].iloc[0])
    c_half = float(half.apply_gap_cost_to_trades(trades, idx)["gap_cost"].iloc[0])
    c_none = float(none.apply_gap_cost_to_trades(trades, idx)["gap_cost"].iloc[0])

    assert c_none == pytest.approx(0.0)
    assert c_half == pytest.approx(c_full * 0.5)


def test_gap_adverse_fraction_is_validated():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="gap_adverse_fraction"):
            TransactionCostModel(gap_adverse_fraction=bad)


def test_gap_cost_default_is_the_documented_adverse_fraction():
    from src.backtesting.costs import DEFAULT_GAP_ADVERSE_FRACTION, MAX_GAP_COST_FRAC_OF_NOTIONAL

    assert DEFAULT_GAP_ADVERSE_FRACTION == 0.5
    assert MAX_GAP_COST_FRAC_OF_NOTIONAL == 1.0
    assert TransactionCostModel().gap_adverse_fraction == DEFAULT_GAP_ADVERSE_FRACTION


def test_net_pnl_still_deducts_the_gap_cost():
    """The correction must reduce the charge, not remove the mechanism."""
    df, trades, idx = _held_through_weekend("1h")
    cm = TransactionCostModel(gap_adverse_fraction=1.0).fit(df)
    out = cm.apply_gap_cost_to_trades(trades, idx)
    assert (out["net_pnl"] == out["PnL"] - out["gap_cost"]).all()
