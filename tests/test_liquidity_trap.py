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
    DEFAULT_LEVEL_TOLERANCE_ATR,
    DEFAULT_MIN_PENETRATION_ATR,
    DEFAULT_MIN_SWEEP_TOUCHES,
    EQUAL_LEVEL_LOOKBACK,
    LiquidityCluster,
    LiquidityTrapResult,
    _cluster_levels,
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

    r = estimate_liquidity_trap(df, min_sweep_touches=2)
    assert r.sweep_just_confirmed is True
    assert r.sweep_direction == "bearish"


def test_support_sweep_confirmed_when_current_candle_breaches_and_rejects():
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 1)
    reference_price = float(df.iloc[-2]["close"])
    level = reference_price - 3.0
    for i in (5, 15, 25):
        _set_candle(df, i, low=level, close=level + 1.5, open_=level + 2.0)
    _set_candle(df, len(df) - 1, low=level - 0.3, close=level + 1.0, high=level + 1.2)

    r = estimate_liquidity_trap(df, min_sweep_touches=2)
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


# ------------------------------------------------------------------------ #
# 2026-08-14 critical review findings
# ------------------------------------------------------------------------ #


def _two_sided_fixture() -> pd.DataFrame:
    """Flat 100.0 canvas with a resistance cluster at 105 and a support
    cluster at 95, so the current candle can sweep both sides."""
    n = EQUAL_LEVEL_LOOKBACK + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "vol_atr_14": 1.0},
        index=idx,
    )
    for i in (10, 20):
        _set_candle(df, i, high=105.0, close=100.0)
    for i in (12, 22):
        _set_candle(df, i, low=95.0, close=100.0)
    return df


def test_candle_sweeping_both_sides_reports_no_confirmed_sweep():
    """
    Regression test. The old code asserted both directions "can't be
    confirmed on the same candle in practice (would require closing both
    above AND below distinct levels)" -- inverted reasoning: it requires
    closing BELOW a resistance level and ABOVE a support level, which any
    ordinary wide-range candle closing mid-range does. Both branches fired
    and `sweep_direction` silently became "bullish" purely because that
    branch runs second, injecting a hard bullish lean into ensemble.py
    (which consumes sweep_direction scaled by trap_probability) from a
    perfectly symmetric, directionally meaningless bar.
    """
    df = _two_sided_fixture()
    _set_candle(df, len(df) - 1, high=106.0, low=94.0, close=100.0)

    r = estimate_liquidity_trap(df, min_sweep_touches=2)
    assert r.sweep_just_confirmed is False
    assert r.sweep_direction is None
    assert any("both" in reason.lower() for reason in r.reasons)


def test_breakout_closing_above_a_swept_level_is_not_a_confirmed_bearish_sweep():
    """
    Regression test: confirmation was tested against the FARTHEST swept
    level, the most permissive possible rule. With clusters at 102 and 110,
    a candle with high=111 closing at 108 was reported as a confirmed
    BEARISH sweep despite closing decisively ABOVE the 102 cluster -- a
    genuine bullish breakout through that level, labelled as its opposite.
    """
    n = EQUAL_LEVEL_LOOKBACK + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "vol_atr_14": 1.0},
        index=idx,
    )
    for i in (10, 20):
        _set_candle(df, i, high=102.0, close=100.0)
    for i in (12, 22):
        _set_candle(df, i, high=110.0, close=100.0)
    _set_candle(df, len(df) - 1, high=111.0, low=100.0, close=108.0)

    r = estimate_liquidity_trap(df)
    assert r.sweep_just_confirmed is False


def test_genuine_single_level_sweep_is_still_confirmed():
    """Guards against the confirmation fix over-correcting into never
    detecting a sweep at all."""
    n = EQUAL_LEVEL_LOOKBACK + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "vol_atr_14": 1.0},
        index=idx,
    )
    for i in (10, 20, 30):
        _set_candle(df, i, high=105.0, close=100.0)
    _set_candle(df, len(df) - 1, high=105.5, low=99.0, close=101.0)

    r = estimate_liquidity_trap(df, min_sweep_touches=2)
    assert r.sweep_just_confirmed is True
    assert r.sweep_direction == "bearish"


def test_just_swept_cluster_is_excluded_from_forward_trap_probability():
    """
    The trap_probability comment promises risk from an UN-SWEPT cluster,
    but `nearest_*` was computed over ALL clusters -- a pool that had just
    been consumed was still scored as pending forward risk (0.750 in this
    fixture).
    """
    n = EQUAL_LEVEL_LOOKBACK + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "vol_atr_14": 1.0},
        index=idx,
    )
    for i in (10, 20, 30):
        _set_candle(df, i, high=100.5, close=100.0)
    _set_candle(df, len(df) - 1, high=101.0, low=99.9, close=100.0)

    r = estimate_liquidity_trap(df, min_sweep_touches=2)
    assert r.sweep_direction == "bearish"
    assert r.nearest_resistance_cluster is None  # it was just swept
    assert r.trap_probability == 0.0


def test_cluster_level_excludes_wicks_disqualified_from_the_touch_count():
    """
    `level` was the mean of ALL proximity members, including ones the
    `respected` filter had just disqualified as non-touches, so the
    reported level was pulled by wicks explicitly judged not to be part of
    the cluster. That level also feeds distance_atr and the sweep-breach
    comparison, so the error propagated.
    """
    values = np.array([100.0, 100.05, 100.10])
    closes = np.array([99.0, 99.0, 100.5])  # third candle closed THROUGH the level
    clusters = _cluster_levels(values, closes, "resistance", atr=1.0, tolerance_atr=0.25)

    assert len(clusters) == 1
    assert clusters[0].touch_count == 2
    assert clusters[0].level == pytest.approx((100.0 + 100.05) / 2)  # not the 3-member mean


def test_result_rejects_confirmed_sweep_with_no_direction():
    """Both halves of the invariant are now enforced; previously only the
    converse (direction without confirmation) raised."""
    with pytest.raises(ValueError, match="sweep_direction"):
        LiquidityTrapResult(
            trap_probability=0.5, liquidity_attraction_score=0.5,
            sweep_just_confirmed=True, sweep_direction=None,
            nearest_resistance_cluster=None, nearest_support_cluster=None, reasons=[],
        )


def test_cluster_validates_its_own_invariants():
    """LiquidityCluster was the only result dataclass across all four
    macro_mtf_liquidity_vol modules with no validation at all."""
    with pytest.raises(ValueError, match="side"):
        LiquidityCluster(side="banana", level=100.0, touch_count=2, distance_atr=1.0)
    with pytest.raises(ValueError, match="touch_count"):
        LiquidityCluster(side="resistance", level=100.0, touch_count=1, distance_atr=1.0)
    with pytest.raises(ValueError, match="distance_atr"):
        LiquidityCluster(side="resistance", level=100.0, touch_count=2, distance_atr=-1.0)
    with pytest.raises(ValueError, match="level"):
        LiquidityCluster(side="resistance", level=float("nan"), touch_count=2, distance_atr=1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0},
        {"lookback": 1},
        {"level_tolerance_atr": 0.0},
        {"proximity_scale_atr": 0.0},
        {"density_saturation_touches": 0},
    ],
)
def test_invalid_tunables_are_rejected_with_a_clear_error(kwargs):
    """`lookback=0` previously produced an opaque IndexError; the two
    scale parameters are divisors and raised ZeroDivisionError deep inside
    the risk calculation."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    with pytest.raises(ValueError):
        estimate_liquidity_trap(df, **kwargs)


def test_non_finite_prices_are_rejected_rather_than_silently_finding_nothing():
    """NaN prices degrade every comparison to False, reporting "no
    structural risk" for what is really "no usable data"."""
    df = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    df.iloc[-1, df.columns.get_loc("high")] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        estimate_liquidity_trap(df)

    df2 = _flat_baseline(EQUAL_LEVEL_LOOKBACK + 5)
    df2.iloc[-10, df2.columns.get_loc("low")] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        estimate_liquidity_trap(df2)


# ------------------------------------------------------------------------ #
# 2026-08-14: sweep selectivity re-tune
# ------------------------------------------------------------------------ #


def _sweep_fixture(*, touches: int, penetration: float, atr: float = 1.0) -> pd.DataFrame:
    """Engineered resistance sweep with a controllable touch count and
    penetration depth, on a flat canvas so nothing else clusters."""
    n = EQUAL_LEVEL_LOOKBACK + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "vol_atr_14": atr},
        index=idx,
    )
    level = 105.0
    for k in range(touches):
        _set_candle(df, 5 + k * 5, high=level, close=100.0)
    _set_candle(df, n - 1, high=level + penetration, low=99.0, close=101.0)
    return df


def test_shallow_penetration_is_not_a_sweep():
    """A wick clearing a level by less than the level's own tolerance band is
    indistinguishable from touching it. Breach-and-close-back alone fired on
    44.1% of real BTC 1h bars -- near coin-flip for a supposedly specific,
    programmatically detectable event."""
    shallow = estimate_liquidity_trap(
        _sweep_fixture(touches=5, penetration=0.01), min_sweep_touches=2
    )
    assert shallow.sweep_just_confirmed is False

    deep = estimate_liquidity_trap(
        _sweep_fixture(touches=5, penetration=0.5), min_sweep_touches=2
    )
    assert deep.sweep_just_confirmed is True
    assert deep.sweep_direction == "bearish"


def test_penetration_threshold_is_measured_in_atr_not_price():
    """Scale-invariance: the same price penetration must fail at a larger
    ATR, since the tolerance band scales with ATR."""
    small_atr = estimate_liquidity_trap(
        _sweep_fixture(touches=5, penetration=0.5, atr=1.0), min_sweep_touches=2
    )
    large_atr = estimate_liquidity_trap(
        _sweep_fixture(touches=5, penetration=0.5, atr=10.0), min_sweep_touches=2
    )
    assert small_atr.sweep_just_confirmed is True
    assert large_atr.sweep_just_confirmed is False


def test_thinly_tested_pool_is_not_a_sweep_at_the_default():
    """A cluster is 'two or more wicks' per spec, but a SWEEP worth acting on
    requires a better-tested pool -- real BTC 1h carries a median of 7
    simultaneous resistance clusters, so any upward wick sweeps something."""
    thin = estimate_liquidity_trap(_sweep_fixture(touches=2, penetration=0.5))
    assert thin.sweep_just_confirmed is False

    thick = estimate_liquidity_trap(_sweep_fixture(touches=5, penetration=0.5))
    assert thick.sweep_just_confirmed is True


def test_cluster_minimum_of_two_is_still_reachable_explicitly():
    """The spec's own cluster definition must remain usable -- the stricter
    sweep bar is a separate, opt-out-able tuning decision."""
    r = estimate_liquidity_trap(
        _sweep_fixture(touches=2, penetration=0.5), min_sweep_touches=2
    )
    assert r.sweep_just_confirmed is True


def test_selectivity_defaults_are_the_b4_selected_values():
    """B.4 (2026-08-18). Tolerance retuned 0.25 -> 0.05 by sweeping
    {0.05, 0.10, 0.15} against a 1-5% confirmation-rate band fixed in
    advance, with the compression precondition applied. Exactly one
    configuration landed in the band: 4.03% on compressed bars, 815 events.

    Penetration stays at 0.25 and is now DECOUPLED from the tolerance -- the
    original rationale gives a lower bound (penetration >= tolerance), not an
    equality, and holding it at 0.25 is strictly more conservative."""
    assert DEFAULT_LEVEL_TOLERANCE_ATR == 0.05
    assert DEFAULT_MIN_PENETRATION_ATR == 0.25
    assert DEFAULT_MIN_PENETRATION_ATR >= DEFAULT_LEVEL_TOLERANCE_ATR
    assert DEFAULT_MIN_SWEEP_TOUCHES == 4


def test_penetration_below_tolerance_is_rejected():
    """The bound that replaced the old equality. Below the tolerance, a
    'penetration' is inside the level's own uncertainty band and cannot be
    distinguished from touching it."""
    with pytest.raises(ValueError, match="below level_tolerance_atr"):
        estimate_liquidity_trap(
            _sweep_fixture(touches=5, penetration=0.5),
            level_tolerance_atr=0.25,
            min_penetration_atr=0.05,
        )


def test_penetration_equal_to_tolerance_is_still_allowed():
    """The old coupled setting must remain reachable -- B.4 decoupled the
    default, it did not forbid the equality."""
    r = estimate_liquidity_trap(
        _sweep_fixture(touches=5, penetration=0.5),
        level_tolerance_atr=0.25,
        min_penetration_atr=0.25,
    )
    assert r.sweep_just_confirmed in (True, False)  # must not raise


def test_min_sweep_touches_below_the_cluster_minimum_is_rejected():
    """Below 2 it cannot select anything additional -- a cluster already
    requires two or more wicks."""
    with pytest.raises(ValueError, match="min_sweep_touches"):
        estimate_liquidity_trap(_sweep_fixture(touches=5, penetration=0.5), min_sweep_touches=1)


def test_negative_penetration_is_rejected():
    with pytest.raises(ValueError, match="min_penetration_atr"):
        estimate_liquidity_trap(
            _sweep_fixture(touches=5, penetration=0.5), min_penetration_atr=-0.1
        )
