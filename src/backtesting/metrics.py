"""
Performance metrics for the walk-forward backtesting harness.

Every function here must be fed cost-adjusted inputs (`net_pnl`/`net_return`
from `costs.TransactionCostModel.apply_gap_cost_to_trades`, or an equity
curve already reduced by gap cost -- see `harness.build_cost_adjusted_equity_curve`).
None of these functions apply, remove, or otherwise touch transaction costs
themselves; that separation is deliberate so a caller can't accidentally
compute a metric from a gross-of-cost series just by skipping a step.

Two Sharpe conventions are provided because this harness uses both:
  - `sharpe_ratio_from_returns`: periodic (bar-level) equity-curve returns,
    used for the primary in-sample vs out-of-sample fold comparison.
  - `sharpe_ratio_from_trade_returns`: per-trade returns annualized by trade
    frequency, used for regime-conditional comparisons (e.g. Risk-On vs
    aggregate) where the relevant population is a set of discrete trades
    that may not span a contiguous time range.
"""

import numpy as np
import pandas as pd


def sharpe_ratio_from_returns(
    returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0
) -> float:
    """Annualized Sharpe from a periodic (bar-level) return series."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    excess = returns - (risk_free_rate / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sharpe_ratio_from_trade_returns(trade_returns: pd.Series, trades_per_year: float) -> float:
    """Annualized Sharpe from per-trade returns, scaled by trade frequency
    instead of bar frequency. Appropriate when the population being
    compared is a subset of trades (e.g. only those entered during a
    specific macro regime) rather than a contiguous equity curve."""
    trade_returns = pd.Series(trade_returns).dropna()
    if len(trade_returns) < 2:
        return float("nan")
    std = trade_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(trades_per_year) * trade_returns.mean() / std)


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss across a set of trades. `inf` when there
    are no losing trades and at least one winner; `nan` when there are no
    trades at all or no winners and no losers."""
    trade_pnls = pd.Series(trade_pnls).dropna()
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = -trade_pnls[trade_pnls < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Peak-to-trough capital loss as a positive fraction (0.18 = 18%
    drawdown). `nan` if the curve has fewer than 2 points."""
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(-drawdown.min())


def expected_value_per_trade(trade_pnls: pd.Series, trade_notionals: pd.Series) -> float:
    """Mean PnL per trade normalized by that trade's notional -- EV as a
    fraction of capital risked, matching MASTER_CONTEXT's 'EV per trade'
    framing (raw PnL isn't comparable across differently-sized trades)."""
    trade_pnls = pd.Series(trade_pnls)
    trade_notionals = pd.Series(trade_notionals).replace(0, np.nan)
    per_trade_return = (trade_pnls / trade_notionals).dropna()
    if per_trade_return.empty:
        return float("nan")
    return float(per_trade_return.mean())


def equity_curve_from_trade_pnls(
    trade_pnls: pd.Series,
    exit_timestamps: pd.Series,
    bar_index: pd.DatetimeIndex,
    init_cash: float,
    floor_at_zero: bool = True,
) -> pd.Series:
    """Step equity curve over `bar_index`, crediting each trade's (already
    cost-adjusted) PnL at its exit bar. Flat between exits.

    BANKRUPTCY FLOOR (Stage A.4, finding S11). Previously this returned
    `init_cash + cumsum()` with nothing bounding it below, so the curve
    crossed zero and kept going. Two consequences, both observed in the
    2026-08-14 ablation:

      - `pct_change()` past the zero crossing divides by a zero-or-negative
        denominator, so every subsequent "return" and any Sharpe built from
        them is arithmetically meaningless. That run reported maxDD of 1.023
        and 1.235 alongside a Sharpe that looked BETTER than the solvent
        baseline.
      - maxDD exceeded 1.0, which is impossible for a real account. A
        drawdown of 123% describes losing more than everything and then
        continuing to trade.

    RUIN IS ABSORBING. Once equity touches zero the account is gone; it does
    not recover on the next winning trade, because there is no capital left
    to place it with. The curve is therefore pinned at 0 from the first
    crossing onward, not merely clipped pointwise. Clipping alone would let
    a later cumsum recovery lift the curve back off the floor and understate
    the damage.

    Post-fix, maxDD <= 1.0 by construction (canonical definition, section 7:
    "Values >= 1.0 are impossible by construction post-fix; if seen, it is a
    bug").

    `floor_at_zero=False` exists only to reproduce the pre-fix behaviour in
    tests that assert the fix changed something.
    """
    pnl_by_exit = pd.Series(trade_pnls.to_numpy(), index=pd.DatetimeIndex(exit_timestamps))
    pnl_by_exit = pnl_by_exit.groupby(level=0).sum()
    realized = pnl_by_exit.reindex(bar_index, fill_value=0.0)
    curve = init_cash + realized.cumsum()

    if floor_at_zero:
        destroyed = curve <= 0.0
        if bool(destroyed.any()):
            first = int(np.argmax(destroyed.to_numpy()))
            curve.iloc[first:] = 0.0
    return curve


def infer_bars_per_year(bar_index: pd.DatetimeIndex) -> float:
    """Derives an annualization factor from the median bar spacing. Callers
    that already know the configured timeframe (e.g. '1h' -> 24*365.25)
    should pass that explicitly instead -- this is a fallback for ad hoc
    series where the timeframe isn't otherwise known."""
    if len(bar_index) < 2:
        raise ValueError("Need at least 2 timestamps to infer bar frequency.")
    diffs = bar_index.to_series().diff().dropna()
    median_seconds = diffs.dt.total_seconds().median()
    if median_seconds <= 0:
        raise ValueError("Non-positive median bar spacing; cannot infer frequency.")
    seconds_per_year = 365.25 * 24 * 3600
    return float(seconds_per_year / median_seconds)
