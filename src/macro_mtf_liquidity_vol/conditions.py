"""
H1 conditions 3 and 4, built to the pre-registered specifications.

    prereg/condition_3_macro.md   commit a5a8fbd8
    prereg/condition_4_mtf.md     commit a5a8fbd8

Both registrations were committed BEFORE any four-condition count was
computed. If this module disagrees with either document, THIS MODULE is
wrong -- the registration does not move.

=========================================================================
THIS MODULE EXISTS TO BE A CALLER (I11)
=========================================================================
`classify_macro_regime` and `score_mtf_alignment` were built, unit-tested,
and left with no caller outside their own test files. They passed their
tests for two phases while being unreachable, and B.5 discovered at the
kill test that two of four hypothesis conditions had never been applied --
S13 recurring.

This module is the production path that reaches them. It does not
reimplement their logic: it prepares their inputs point-in-time (I9) and
consumes their outputs.

=========================================================================
I9 IS THE WHOLE DIFFICULTY HERE
=========================================================================
Every input is lower-frequency than the 1h bars it conditions:

    1d structure/trend   knowable only at the daily close    (D + 1 day)
    1w structure/trend   knowable only at the weekly close   (W + 7 days)
    QQQ daily close      exchange close, treated as D + 1 business day
    FRED macro           publication lag, D + 1 business day (B.0.4)

Using a daily reading on an intraday bar inside its own session is the
funding-alignment defect in another shape, and no splitter can catch it.
Every join goes through `point_in_time.align_point_in_time`, which has no
policy-free code path.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data_engine.point_in_time import (
    BAR_CLOSE_1D,
    BAR_CLOSE_1W,
    FRED_DAILY,
    MARKET_DAILY_CLOSE,
    align_point_in_time,
)

from .macro_regime import MacroSignalSnapshot, classify_macro_regime
from .mtf_alignment import score_mtf_alignment

# Registered. Fixed. Not swept.
N_PERIODS = 20
MTF_MIN_ALIGNMENT = 3
MACRO_ADMITTED_STATES = ("Risk-On", "Transitional")

# The four registered reading labels. `score_mtf_alignment` requires exactly
# four readings; these are two timeframes x two reading types, because only
# 1d and 1w are ingested as independent series (4h would be a resample of
# 1h, not an independent structural read).
READING_LABELS = ("1D_trend", "1D_structure", "1W_trend", "1W_structure")


# ---------------------------------------------------------------------- #
# Condition 4 -- MTF alignment
# ---------------------------------------------------------------------- #


def signed_trend(close: pd.Series, n: int = N_PERIODS) -> pd.Series:
    """`sign(close - SMA(close, n))`, in {-1, 0, +1}. NaN during warm-up."""
    sma = close.rolling(n).mean()
    return pd.Series(np.sign(close - sma), index=close.index).where(sma.notna())


def signed_structure(high: pd.Series, low: pd.Series, n: int = N_PERIODS) -> pd.Series:
    """HH/HL vs LH/LL over two adjacent n-bar windows, in {-1, 0, +1}.

    Both legs are required, which is what makes it a *sequence* rather than
    a single extreme. Mixed structure reads 0 and agrees with nothing.
    """
    hi_now, lo_now = high.rolling(n).max(), low.rolling(n).min()
    hi_prev, lo_prev = hi_now.shift(n), lo_now.shift(n)

    hh, hl = hi_now > hi_prev, lo_now > lo_prev
    lh, ll = hi_now < hi_prev, lo_now < lo_prev

    out = pd.Series(0.0, index=high.index)
    out[hh & hl] = 1.0
    out[lh & ll] = -1.0
    return out.where(hi_prev.notna() & lo_prev.notna())


def mtf_readings_at_bars(
    daily: pd.DataFrame, weekly: pd.DataFrame, bar_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """The four registered readings, aligned point-in-time onto `bar_index`.

    A 1d reading is observable at D+1, a 1w reading at W+7d (I9). Bars
    before the first observable reading are NaN, and a NaN reading fails
    the condition rather than counting as neutral agreement.
    """
    d = pd.DataFrame({
        "1D_trend": signed_trend(daily["close"]),
        "1D_structure": signed_structure(daily["high"], daily["low"]),
    })
    w = pd.DataFrame({
        "1W_trend": signed_trend(weekly["close"]),
        "1W_structure": signed_structure(weekly["high"], weekly["low"]),
    })
    out = pd.concat(
        [
            align_point_in_time(d, bar_index, BAR_CLOSE_1D),
            align_point_in_time(w, bar_index, BAR_CLOSE_1W),
        ],
        axis=1,
    )
    return out[list(READING_LABELS)]


def alignment_against(readings: Dict[str, float], implied_direction: int) -> Optional[int]:
    """Count of readings agreeing with the sweep's implied reversal direction.

    Delegates classification to `score_mtf_alignment` -- this is the
    production caller that module lacked (I11). Its own `score` counts
    agreement with the MAJORITY direction, which is not the registered
    quantity, so the count is taken from `per_timeframe_direction` against
    the externally supplied direction.

    Returns None when any reading is unobservable (NaN).
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in readings.values()):
        return None

    result = score_mtf_alignment({k: float(v) for k, v in readings.items()})
    want = "bullish" if implied_direction > 0 else "bearish"
    return sum(1 for d in result.per_timeframe_direction.values() if d == want)


# ---------------------------------------------------------------------- #
# Condition 3 -- macro regime, restated
# ---------------------------------------------------------------------- #


def macro_inputs_daily(macro: pd.DataFrame, qqq: pd.DataFrame, btc_daily: pd.DataFrame) -> pd.DataFrame:
    """The registered macro legs, on a daily index, before alignment.

    Correlation is computed on daily RETURNS (not levels) over the trading
    days both BTC and QQQ have, which is QQQ's calendar -- BTC trades 24/7
    and QQQ does not, so the intersection is the binding constraint.
    """
    dxy = macro["dxy_proxy"].dropna()
    vix = macro["vix"].dropna()
    us10y = macro["us10y"].dropna()

    btc_ret = btc_daily["close"].pct_change()
    qqq_ret = qqq["close"].pct_change()
    btc_ret.index = btc_ret.index.normalize()
    qqq_ret.index = qqq_ret.index.normalize()
    joined = pd.concat([btc_ret.rename("btc"), qqq_ret.rename("qqq")], axis=1).dropna()
    corr = joined["btc"].rolling(N_PERIODS).corr(joined["qqq"])

    frame = pd.DataFrame({
        "dxy_20d_momentum_pct": dxy.pct_change(N_PERIODS) * 100.0,
        "vix_level": vix,
        "vix_5d_roc_pct": vix.pct_change(5) * 100.0,
        "us10y_level_pct": us10y,
        "us10y_20d_direction_pct": us10y.pct_change(N_PERIODS) * 100.0,
    })
    return frame, corr


def macro_state_at_bars(
    macro: pd.DataFrame,
    qqq: pd.DataFrame,
    btc_daily: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
) -> pd.Series:
    """Macro regime state per 1h bar, per the registered restatement.

    FRED legs carry the B.0.4 publication floor; the correlation leg carries
    QQQ's exchange-close policy. Bars where any leg is unobservable are NaN
    and FAIL condition 3 -- absence is not admission.

    `next_high_impact_event` is None throughout: no economic-calendar source
    is wired in, so Macro-Event Window never fires and bars within 24h of a
    release are admitted where the full definition would exclude them. The
    condition as implemented is therefore LOOSER than H1's intent (I5).
    """
    fred_frame, corr = macro_inputs_daily(macro, qqq, btc_daily)

    fred_at_bars = align_point_in_time(fred_frame, bar_index, FRED_DAILY)
    corr_at_bars = align_point_in_time(corr.rename("btc_qqq_corr_20d").to_frame(),
                                       bar_index, MARKET_DAILY_CLOSE)["btc_qqq_corr_20d"]

    cols = fred_at_bars.join(corr_at_bars)
    usable = cols.notna().all(axis=1)

    states = pd.Series(pd.NA, index=bar_index, dtype="object")
    for ts in bar_index[usable]:
        row = cols.loc[ts]
        snap = MacroSignalSnapshot(
            timestamp=ts,
            dxy_20d_momentum_pct=float(row["dxy_20d_momentum_pct"]),
            us10y_level_pct=float(row["us10y_level_pct"]),
            us10y_20d_direction_pct=float(row["us10y_20d_direction_pct"]),
            vix_level=float(row["vix_level"]),
            vix_5d_roc_pct=float(row["vix_5d_roc_pct"]),
            # Registered as REMOVED. The flag below is what removes it; this
            # value is never read when use_dominance_leg=False.
            btc_dominance_20d_trend_pct=0.0,
            btc_qqq_corr_20d=float(row["btc_qqq_corr_20d"]),
            next_high_impact_event=None,
        )
        states.loc[ts] = classify_macro_regime(snap, use_dominance_leg=False).state
    return states


def condition_3_holds(states: pd.Series) -> pd.Series:
    """`macro_regime in {Risk-On, Transitional}`. NaN state fails."""
    return states.isin(MACRO_ADMITTED_STATES).fillna(False)


__all__ = [
    "N_PERIODS",
    "MTF_MIN_ALIGNMENT",
    "MACRO_ADMITTED_STATES",
    "READING_LABELS",
    "signed_trend",
    "signed_structure",
    "mtf_readings_at_bars",
    "alignment_against",
    "macro_inputs_daily",
    "macro_state_at_bars",
    "condition_3_holds",
]
