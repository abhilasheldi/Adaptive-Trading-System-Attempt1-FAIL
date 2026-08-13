"""
Tests for the Liquidity Trap heuristic (Layer 5),
src/macro_mtf_liquidity_vol/liquidity_trap.py.

Uses fully deterministic, hand-crafted OHLCV fixtures rather than random
walks -- random-walk fixtures were used during manual verification of this
module and directly caught four real bugs (chained clustering with
unbounded span, "rejected" checked against a candle's own high instead of
the cluster's shared level, resistance/support clusters not filtered by
position relative to price, and min/max reversed when picking which level a
sweep actually reached) -- see the module's inline comments for each. For
the formal suite, deterministic fixtures with a wide, controlled per-step
price increment avoid ANY accidental incidental clustering, so every
assertion tests exactly the one behavior it names, not a coincidence of
random data (a second real trap this module's own development fell into:
an earlier version of this test file's manual checks showed a "control"
case unexpectedly sweeping because injected fixture candles had an
unintended side effect on the OTHER side's clustering).
"""

import numpy as np
import pandas as pd
import pytest

from src.macro_mtf_liquidity_vol.liquidity_trap import (
    EQUAL_LEVEL_LOOKBACK,
    estimate_liquidity_trap,
)


def _flat_baseline(n: int, base: float = 100.0, step: float = 5.0) -> pd.DataFrame:
    """n candles, strictly increasing by `step` (wide relative to ATR=1.0
    and the default 0.25 ATR tolerance), so no two highs/lows ever land
    within tolerance of each other by construction -- a clean canvas to
    engineer specific clusters onto without accidental interference."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = base + np.arange(n) * step
    df = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
        },
        index=idx,
    )
    df["vol_atr_14"] = 1.0
    return df


def _set_candle(df: pd.DataFrame, i: int, *, high=None, low=None, close=None, open_=None) -> None:
    if high is not None:
        df.iloc[i, df.columns.get_loc("high")] = high
    if low is not None:
        df.iloc[i, df.columns.get_loc("low")] = low
    if close is not None:
        df.iloc[i, df.columns.get_loc("close")] = close
    if open_ is not None:
        df.iloc[i, df.columns.get_loc("open")] = open_


# ------------------------------------------------------------------------ #
# Done-when bar: output is a probability (0-1), never a claimed exact price
# ------------------------------------------------------------------------ #


def test_trap_probability_is_always_a_clean_0_1_value():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    r = estimate_liquidity_trap(df)
    assert isinstance(r.trap_probability, float)
    assert 0.0 <= r.trap_probability <= 1.0
    assert 0.0 <= r.liquidity_attraction_score <= 1.0


def test_result_has_no_field_claiming_a_precise_price_as_the_core_output():
    """The required output is trap_probability. Cluster `level` fields (if
    any clusters exist) are documented as non-precision heuristic
    estimates, not the module's claimed deliverable -- this test locks in
    that trap_probability is present and dimensionless regardless of
    whether any cluster/level information is available at all."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    r = estimate_liquidity_trap(df)
    assert r.nearest_resistance_cluster is None  # strictly increasing baseline: no repeated levels
    assert r.nearest_support_cluster is None
    assert isinstance(r.trap_probability, float)  # the required output survives even with zero structure found


# ------------------------------------------------------------------------ #
# Cluster detection correctness
# ------------------------------------------------------------------------ #


def test_no_clusters_on_strictly_increasing_baseline():
    """No two highs or lows are ever close on this fixture by construction
    -- must not hallucinate structure that isn't there."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    r = estimate_liquidity_trap(df)
    assert r.nearest_resistance_cluster is None
    assert r.nearest_support_cluster is None
    assert r.trap_probability == 0.0
    assert r.sweep_just_confirmed is False


def test_equal_highs_cluster_detected_with_correct_touch_count():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    level = float(df.iloc[-1]["close"]) + 3.0  # above current price -- see reference_price note below
    for i in (5, 15, 25):
        _set_candle(df, i, high=level, close=level - 1.5, open_=level - 2.0)
    r = estimate_liquidity_trap(df)
    assert r.nearest_resistance_cluster is not None
    assert r.nearest_resistance_cluster.touch_count == 3
    assert r.nearest_resistance_cluster.level == pytest.approx(level, abs=0.01)


def test_close_through_the_level_does_not_count_as_a_touch():
    """A candle whose close is at/above the shared level BROKE OUT, it
    wasn't rejected -- must not count toward touch_count, and with only 1
    genuine rejection left, no cluster should form at all."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    level = float(df.iloc[-1]["close"]) + 3.0
    _set_candle(df, 5, high=level, close=level - 1.5, open_=level - 2.0)  # genuine rejection
    _set_candle(df, 15, high=level, close=level + 0.5, open_=level - 0.5)  # broke through -- not a rejection
    r = estimate_liquidity_trap(df)
    assert r.nearest_resistance_cluster is None  # only 1 genuine touch remains -- below the 2-touch minimum


def test_clusters_are_filtered_by_position_relative_to_current_price():
    """A 'resistance' cluster sitting BELOW current price isn't resistance
    at all (price has already moved past it) -- regression test for a real
    bug found during development where this filter was missing."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    stale_level = float(df.iloc[10]["close"])  # a level from early in the window, now far below current price
    for i in (5, 8):
        _set_candle(df, i, high=stale_level + 0.05, close=stale_level - 0.1, open_=stale_level - 0.2)
    r = estimate_liquidity_trap(df)
    # This level is now well below current price -- must not be reported as "resistance."
    if r.nearest_resistance_cluster is not None:
        assert r.nearest_resistance_cluster.level > float(df.iloc[-1]["close"])


# ------------------------------------------------------------------------ #
# Sweep detection: "stop-runs above equal highs or below equal lows,
# followed by a confirmed reversal" (spec's exact definition)
# ------------------------------------------------------------------------ #


def test_resistance_sweep_confirmed_when_current_candle_breaches_and_rejects():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1)
    reference_price = float(df.iloc[-2]["close"])
    level = reference_price + 3.0
    for i in (5, 15, 25):
        _set_candle(df, i, high=level, close=level - 1.5, open_=level - 2.0)
    # Current (last) candle sweeps above the level, then closes back below it.
    _set_candle(df, len(df) - 1, high=level + 0.3, close=level - 1.0, low=level - 1.2)

    r = estimate_liquidity_trap(df)
    assert r.sweep_just_confirmed is True
    assert r.sweep_direction == "bearish"


def test_support_sweep_confirmed_when_current_candle_breaches_and_rejects():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1)
    reference_price = float(df.iloc[-2]["close"])
    level = reference_price - 3.0
    for i in (5, 15, 25):
        _set_candle(df, i, low=level, close=level + 1.5, open_=level + 2.0)
    _set_candle(df, len(df) - 1, low=level - 0.3, close=level + 1.0, high=level + 1.2)

    r = estimate_liquidity_trap(df)
    assert r.sweep_just_confirmed is True
    assert r.sweep_direction == "bullish"


def test_no_sweep_when_current_candle_does_not_breach_the_cluster():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1)
    reference_price = float(df.iloc[-2]["close"])
    level = reference_price + 3.0
    for i in (5, 15, 25):
        _set_candle(df, i, high=level, close=level - 1.5, open_=level - 2.0)
    # Current candle stays well clear of the cluster -- baseline value, untouched.
    r = estimate_liquidity_trap(df)
    assert r.sweep_just_confirmed is False
    assert r.sweep_direction is None


def test_wick_that_breaches_but_does_not_close_back_is_not_a_confirmed_sweep():
    """A wick beyond the level without a rejecting close is a breakout
    attempt, not a 'confirmed reversal' -- must not count as a sweep."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1)
    reference_price = float(df.iloc[-2]["close"])
    level = reference_price + 3.0
    for i in (5, 15, 25):
        _set_candle(df, i, high=level, close=level - 1.5, open_=level - 2.0)
    # Wicks above the level but closes ABOVE it too -- no reversal, a breakout.
    _set_candle(df, len(df) - 1, high=level + 0.3, close=level + 0.1, low=level - 0.2)
    r = estimate_liquidity_trap(df)
    assert r.sweep_just_confirmed is False


# ------------------------------------------------------------------------ #
# No-lookahead: only strictly-prior candles inform the current candle's
# cluster detection.
# ------------------------------------------------------------------------ #


def test_future_candles_cannot_affect_the_current_evaluation():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 20)
    r_before = estimate_liquidity_trap(df.iloc[: EQUAL_LEVEL_LOOKBACK + 1])

    df_mutated = df.copy()
    # Mutate a candle strictly AFTER the evaluated "current" one.
    _set_candle(df_mutated, EQUAL_LEVEL_LOOKBACK + 5, high=9999.0, close=9998.0)
    r_after = estimate_liquidity_trap(df_mutated.iloc[: EQUAL_LEVEL_LOOKBACK + 1])

    assert r_before.trap_probability == r_after.trap_probability
    assert r_before.sweep_just_confirmed == r_after.sweep_just_confirmed


# ------------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------------ #


def test_rejects_missing_columns():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1).drop(columns=["low"])
    with pytest.raises(ValueError, match="missing required columns"):
        estimate_liquidity_trap(df)


def test_rejects_non_datetime_index():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1).reset_index(drop=True)
    with pytest.raises(TypeError):
        estimate_liquidity_trap(df)


def test_rejects_unsorted_index():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1).iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        estimate_liquidity_trap(df)


def test_rejects_too_few_rows():
    df = _flat_baseline(10)
    with pytest.raises(ValueError, match="lookback"):
        estimate_liquidity_trap(df)


def test_rejects_non_positive_atr():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1)
    df.iloc[-1, df.columns.get_loc("vol_atr_14")] = 0.0
    with pytest.raises(ValueError, match="positive finite"):
        estimate_liquidity_trap(df)


def test_exact_spec_lookback_is_50():
    """Locks in the one exact number this layer's source docs give:
    'Equal highs and equal lows within the trailing 50 candles.'"""
    assert EQUAL_LEVEL_LOOKBACK == 50
