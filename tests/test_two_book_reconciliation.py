"""
Gate A criterion 5 -- the two-book reconciliation regression test (S12 / I7).

BUILD_PLAN_v2 amendment 2.1 retired the original criterion ("reproduce fold 2
of `all_layers` and confirm sizing-book and reported maxDD now agree"). That
artifact is not regenerable: `Phase2Strategy`, `run_layer_ablation.py`,
`HistoricalAnalogEngine` and `EnsembleMetaModel` were all deleted at Stage 0.
Its PURPOSE was verifying the S12 fix, and that does not need that specific
artifact. The substitute specified by the amendment is implemented here: a
synthetic strategy holding positions across bars with non-zero gap costs, run
through the harness, asserting the sizing book and the reported equity curve
agree -- as a permanent regression test rather than a one-off script.

WHY THIS IS DISCRIMINATING, RATHER THAN TRUE BY CONSTRUCTION
------------------------------------------------------------
Post-fix, `SegmentResult.equity_curve` simply *is* `pf.value()`. Asserting
"the two curves are equal" is therefore a tautology -- it would pass against a
harness that had reverted to two books and merely renamed one of them. The
property that actually separates one book from two is visible in the ORDER
SIZES:

  `size_type='targetpercent'` sizes every order as a fraction of the
  portfolio's own equity. When gap cost lives inside that portfolio (I7),
  charging it CHANGES THE SIZES. When gap cost is subtracted afterwards into
  a separate reported curve (S12), sizing never sees the charge and the sizes
  come out identical whether gap cost is charged or not.

So these tests run identical positions twice -- once with the gap channel
live, once with it switched off via `gap_adverse_fraction=0.0` -- and assert
that both the equity and the trade sizes respond. A harness that reverts to
two books still passes the equity assertion and fails on the sizes.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting import TransactionCostModel, WalkForwardBacktester

INIT_CASH = 10_000.0
TIMEFRAME = "1h"


# ------------------------------------------------------------------------ #
# Synthetic fixtures
# ------------------------------------------------------------------------ #


def make_gap_prone_ohlcv(n_hours: int = 5000, seed: int = 11) -> pd.DataFrame:
    """1h bars with a price discontinuity injected at the start of every
    weekend, so `TransactionCostModel._fit_gap_stats` measures a non-zero
    weekend gap frequency and magnitude. Without injected gaps the fitted gap
    cost is exactly zero and this whole test would be vacuous -- the fixture
    asserts its own gap-proneness in `fitted_cost_model` below."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n_hours))

    is_weekend = np.isin(idx.dayofweek, [5, 6])
    weekend_starts = np.where(is_weekend & ~np.r_[False, is_weekend[:-1]])[0]
    for i in weekend_starts:
        # Step the whole remaining series, so the discontinuity is a genuine
        # jump between adjacent closes rather than a spike that reverts.
        close[i:] += rng.choice([-1.0, 1.0]) * 8.0

    high = close + rng.uniform(0.1, 1.0, n_hours)
    low = close - rng.uniform(0.1, 1.0, n_hours)
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)


def weekend_holding_positions(df: pd.DataFrame, size: float = 0.5) -> pd.Series:
    """Deterministic positions that enter before each weekend and exit after
    it, so every trade's holding period spans at least one gap-prone session.

    Ends FLAT: the last 72 bars are forced to zero. `pf.value()` on the final
    bar then contains no marked-to-market open position, which is what makes
    the closed-form reconciliation in
    `test_no_cost_is_applied_outside_the_book_that_sizing_reads` exact."""
    pos = pd.Series(0.0, index=df.index)
    is_weekend = np.isin(df.index.dayofweek, [5, 6])
    weekend_starts = np.where(is_weekend & ~np.r_[False, is_weekend[:-1]])[0]

    for i in weekend_starts:
        entry = max(i - 12, 0)          # Friday midday
        exit_ = min(i + 60, len(df))    # Monday evening
        pos.iloc[entry:exit_] = size

    pos.iloc[-72:] = 0.0
    return pos


@pytest.fixture(scope="module")
def gap_prone_ohlcv() -> pd.DataFrame:
    return make_gap_prone_ohlcv()


def _segment(df: pd.DataFrame, positions: pd.Series, cost_model: TransactionCostModel):
    """Drive the harness's segment engine directly. `_run_segment` is where
    both books lived, so it is the unit the regression is about; going through
    `run()` would add fold generation and kill-criteria machinery that this
    test is not making a claim about."""
    bt = WalkForwardBacktester(
        strategy_factory=lambda: None,  # unused -- positions are supplied directly
        timeframe=TIMEFRAME,
        init_cash=INIT_CASH,
    )
    return bt._run_segment(df, positions, cost_model, fold=0, segment_name="out_of_sample")


@pytest.fixture(scope="module")
def books(gap_prone_ohlcv):
    """The same positions priced twice: gap channel live, and gap channel off.

    Split so the fit sees the injected weekend gaps (train) and the segment
    being priced is a later, disjoint slice (test) -- the cost model is never
    fitted on the bars it prices."""
    df = gap_prone_ohlcv
    train_df, test_df = df.iloc[:3000], df.iloc[3000:]

    live = TransactionCostModel().fit(train_df)
    # gap_adverse_fraction is the share of a symmetric gap magnitude charged
    # as a one-sided cost. Zero switches the gap channel off while leaving
    # fees and slippage identical, which isolates the gap term exactly.
    off = TransactionCostModel(gap_adverse_fraction=0.0).fit(train_df)

    positions = weekend_holding_positions(test_df)
    return {
        "test_df": test_df,
        "positions": positions,
        "live_params": live.params,
        "with_gap": _segment(test_df, positions, live),
        "without_gap": _segment(test_df, positions, off),
    }


# ------------------------------------------------------------------------ #
# The fixture must be genuinely gap-prone, or every assertion below is vacuous
# ------------------------------------------------------------------------ #


def test_fixture_actually_carries_non_zero_gap_cost(books):
    """Amendment 2.1 requires "non-zero gap costs". If the injected gaps ever
    stop being detected, the reconciliation tests would still pass while
    testing nothing -- so the precondition is asserted, not assumed."""
    p = books["live_params"]
    assert p.gap_frequency_weekend > 0.0
    assert p.gap_magnitude_weekend_frac > 0.0

    with_gap, without_gap = books["with_gap"], books["without_gap"]
    assert with_gap.n_trades > 0
    assert with_gap.n_trades == without_gap.n_trades  # same positions, same trades

    # At least one trade must actually span a weekend, or the gap parameters
    # above would never be applied to anything.
    entries = pd.DatetimeIndex(with_gap.trades["Entry Timestamp"])
    exits = pd.DatetimeIndex(with_gap.trades["Exit Timestamp"])
    spans_weekend = [
        np.isin(pd.date_range(a, b, freq=TIMEFRAME).dayofweek, [5, 6]).any()
        for a, b in zip(entries, exits)
    ]
    assert any(spans_weekend)


# ------------------------------------------------------------------------ #
# The regression itself
# ------------------------------------------------------------------------ #


def test_gap_cost_reaches_the_book_that_sizes_the_orders(books):
    """S12 / I7, stated as a falsifiable property.

    Under the two-book harness, gap cost was deducted from a reported curve
    that `size_type='targetpercent'` never read, so order sizes were identical
    with and without the charge. Under one book, the charge lowers equity,
    which lowers every subsequent targetpercent order.

    This is the assertion that a reverted harness fails."""
    with_gap, without_gap = books["with_gap"], books["without_gap"]

    # 1. The charge is real -- it moves the reported equity at all.
    assert with_gap.equity_curve.iloc[-1] < without_gap.equity_curve.iloc[-1]

    # 2. ...and it was charged INSIDE the book sizing reads. Sizes must differ.
    sizes_with = with_gap.trades["Size"].to_numpy(dtype=float)
    sizes_without = without_gap.trades["Size"].to_numpy(dtype=float)
    assert not np.allclose(sizes_with, sizes_without), (
        "Order sizes are unchanged by gap cost. That means sizing read an "
        "equity the gap charge never touched -- the S12 two-book defect."
    )

    # Direction: a cost cannot make the account richer, so the sized orders
    # after the first charge must be no larger than the uncharged ones.
    assert sizes_with[-1] < sizes_without[-1]


def test_no_cost_is_applied_outside_the_book_that_sizing_reads(books):
    """I7 in closed form. The segment ends flat, so the whole equity change
    is realised trade PnL. If any cost were applied outside the portfolio --
    the old book-B construction, where gap cost was subtracted from trade PnL
    after the fact -- these two quantities would separate by exactly that
    out-of-book charge."""
    for name in ("with_gap", "without_gap"):
        res = books[name]
        realised = float(res.trades["net_pnl"].sum())
        booked = float(res.equity_curve.iloc[-1]) - INIT_CASH
        assert booked == pytest.approx(realised, rel=1e-6, abs=1e-6), (
            f"[{name}] reported equity moved by {booked:.6f} while trades "
            f"account for {realised:.6f} -- a cost is being applied outside "
            f"the book."
        )


def test_equity_curve_is_read_from_the_portfolio_not_rebuilt_from_trade_pnl(books):
    """The fix's mechanism, not just its arithmetic.

    `equity_curve_from_trade_pnls` produces a step function that only moves at
    exit timestamps. `pf.value()` marks to market every bar. Asserting the
    curve varies *while a single trade is open* pins the harness to reading
    the portfolio directly, which is what makes it the same object sizing
    used."""
    res = books["with_gap"]
    entry = pd.Timestamp(res.trades["Entry Timestamp"].iloc[0])
    exit_ = pd.Timestamp(res.trades["Exit Timestamp"].iloc[0])

    open_span = res.equity_curve.loc[entry:exit_].iloc[1:-1]
    assert len(open_span) > 2, "first trade too short to inspect mark-to-market"
    assert open_span.nunique() > 2, (
        "Equity is piecewise-constant while a position is open -- the curve "
        "was rebuilt from trade PnL rather than read from the portfolio that "
        "sized the orders."
    )


def test_bankruptcy_floor_holds_on_both_books(books):
    """S11. maxDD >= 1.0 is impossible by construction once equity is floored,
    so seeing it is a bug rather than a result."""
    for name in ("with_gap", "without_gap"):
        res = books[name]
        assert (res.equity_curve >= 0.0).all()
        assert np.isnan(res.max_drawdown) or res.max_drawdown <= 1.0
