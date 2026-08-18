"""
H1 conditions 3 and 4, against their pre-registered specifications.

    prereg/condition_3_macro.md   commit a5a8fbd8
    prereg/condition_4_mtf.md     commit a5a8fbd8

Includes the I11 wiring guards: a test that FAILS if either condition module
loses its production caller. Both passed their own tests for two phases
while being unreachable, and B.5 discovered it at the kill test.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.macro_mtf_liquidity_vol.conditions import (
    MTF_MIN_ALIGNMENT,
    N_PERIODS,
    READING_LABELS,
    alignment_against,
    condition_3_holds,
    mtf_readings_at_bars,
    signed_structure,
    signed_trend,
)
from src.macro_mtf_liquidity_vol.macro_regime import (
    MacroSignalSnapshot,
    classify_macro_regime,
)

SRC = Path("src")


# ---------------------------------------------------------------------- #
# I11 -- wiring guards
# ---------------------------------------------------------------------- #


def _production_callers(symbol: str) -> set[str]:
    """Modules under src/ (excluding the symbol's own module) that call it."""
    hits = set()
    for path in SRC.rglob("*.py"):
        # utf-8-sig: at least one source file carries a UTF-8 BOM, which
        # `ast.parse` rejects as a non-printable character.
        text = path.read_text(encoding="utf-8-sig")
        if f"def {symbol}" in text:
            continue  # the defining module
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == symbol:
                hits.add(str(path))
    return hits


@pytest.mark.parametrize("symbol", ["classify_macro_regime", "score_mtf_alignment"])
def test_condition_module_has_a_production_caller(symbol: str):
    """I11. A module with no caller outside its own test file is not a built
    condition. This fails the moment `conditions.py` stops calling it --
    which is exactly how conditions 3 and 4 sat unreachable for two phases
    while their own tests stayed green."""
    callers = _production_callers(symbol)
    assert callers, (
        f"{symbol} has no caller anywhere under src/. It is an orphan: its own "
        f"tests prove things about the module, not about the system (I11)."
    )


def test_the_wiring_guard_is_not_vacuous():
    """The guard must be able to fail. A symbol that genuinely has no
    production caller must come back empty."""
    assert _production_callers("no_such_function_anywhere") == set()


# ---------------------------------------------------------------------- #
# The dominance-leg equivalence, proved rather than asserted (I4)
# ---------------------------------------------------------------------- #


def _snap(dxy, vix, vix_roc, dom, corr) -> MacroSignalSnapshot:
    return MacroSignalSnapshot(
        timestamp=pd.Timestamp("2024-01-02T00:00Z"),
        dxy_20d_momentum_pct=dxy, us10y_level_pct=4.0, us10y_20d_direction_pct=0.0,
        vix_level=vix, vix_5d_roc_pct=vix_roc,
        btc_dominance_20d_trend_pct=dom, btc_qqq_corr_20d=corr,
    )


def test_dominance_leg_cannot_change_the_admitted_set():
    """Registered in prereg/condition_3_macro.md, proved here.

    `dominance_expanding` appears only in `risk_on`, never in `risk_off`.
    H1 admits {Risk-On, Transitional} = NOT Risk-Off. Removing the leg can
    only move bars Transitional -> Risk-On, and both pass.

    Swept over a grid spanning every branch, with dominance taking both
    signs -- if the admitted set ever differed, this finds it."""
    admitted = ("Risk-On", "Transitional")
    checked = 0
    for dxy in (-2.0, -0.1, 0.0, 0.1, 2.0):
        for vix in (12.0, 19.9, 22.0, 25.1, 40.0):
            for roc in (-25.0, -1.0, 0.0, 15.0, 35.0):
                for corr in (-0.9, -0.1, 0.0, 0.5, 0.7, 0.95):
                    for dom in (-5.0, 0.0, 5.0):
                        s = _snap(dxy, vix, roc, dom, corr)
                        with_leg = classify_macro_regime(s, use_dominance_leg=True).state
                        without = classify_macro_regime(s, use_dominance_leg=False).state
                        assert (with_leg in admitted) == (without in admitted), (
                            f"admitted set differs at dxy={dxy} vix={vix} roc={roc} "
                            f"corr={corr} dom={dom}: {with_leg} vs {without}"
                        )
                        checked += 1
    assert checked == 5 * 5 * 5 * 6 * 3


def test_removing_the_leg_does_move_states_so_the_test_above_is_not_vacuous():
    """The equivalence is about the ADMITTED SET, not about the states being
    identical. Dominance-negative bars really do move Transitional ->
    Risk-On; the point is that both are admitted."""
    s = _snap(dxy=-1.0, vix=15.0, vix_roc=-5.0, dom=-3.0, corr=0.5)
    assert classify_macro_regime(s, use_dominance_leg=True).state == "Transitional"
    assert classify_macro_regime(s, use_dominance_leg=False).state == "Risk-On"


def test_confidence_drops_the_dominance_term_rather_than_inventing_one():
    """A sentinel dominance value would feed `risk_on_score` and produce a
    confidence partly driven by data that does not exist."""
    hi = _snap(dxy=-3.0, vix=10.0, vix_roc=-30.0, dom=9.0, corr=0.9)
    lo = _snap(dxy=-3.0, vix=10.0, vix_roc=-30.0, dom=-9.0, corr=0.9)
    # With the leg off, dominance cannot affect confidence at all.
    a = classify_macro_regime(hi, use_dominance_leg=False).confidence
    b = classify_macro_regime(lo, use_dominance_leg=False).confidence
    assert a == pytest.approx(b)
    # With it on, it does.
    c = classify_macro_regime(hi, use_dominance_leg=True).confidence
    d = classify_macro_regime(lo, use_dominance_leg=True).confidence
    assert c != pytest.approx(d)


# ---------------------------------------------------------------------- #
# Condition 4 readings, against the registered definitions
# ---------------------------------------------------------------------- #


def test_signed_trend_is_the_registered_sign_test():
    idx = pd.date_range("2024-01-01", periods=25, freq="1D", tz="UTC")
    rising = pd.Series(np.arange(25.0), index=idx)
    out = signed_trend(rising)
    assert out.iloc[:N_PERIODS - 1].isna().all()      # warm-up is NaN, not 0
    assert out.iloc[-1] == 1.0
    falling = pd.Series(np.arange(25.0)[::-1], index=idx)
    assert signed_trend(falling).iloc[-1] == -1.0


def test_signed_structure_requires_both_legs_of_the_sequence():
    """HH alone is not an HH/HL sequence. Requiring both is what makes it a
    sequence rather than a single extreme."""
    n = N_PERIODS
    idx = pd.date_range("2024-01-01", periods=2 * n, freq="1D", tz="UTC")

    # Higher high AND higher low -> +1
    high = pd.Series([10.0] * n + [12.0] * n, index=idx)
    low = pd.Series([5.0] * n + [7.0] * n, index=idx)
    assert signed_structure(high, low).iloc[-1] == 1.0

    # Higher high but LOWER low -> mixed -> 0
    low_mixed = pd.Series([5.0] * n + [3.0] * n, index=idx)
    assert signed_structure(high, low_mixed).iloc[-1] == 0.0

    # Lower high AND lower low -> -1
    high_down = pd.Series([10.0] * n + [8.0] * n, index=idx)
    assert signed_structure(high_down, low_mixed).iloc[-1] == -1.0


def test_alignment_counts_against_the_sweeps_direction_not_the_majority():
    """The registered quantity is agreement with the sweep's IMPLIED
    reversal direction. `score_mtf_alignment`'s own `score` counts agreement
    with the majority, which is a different number whenever the two differ."""
    readings = {"1D_trend": 1.0, "1D_structure": 1.0, "1W_trend": 1.0, "1W_structure": -1.0}
    assert alignment_against(readings, implied_direction=+1) == 3
    assert alignment_against(readings, implied_direction=-1) == 1  # not 3


def test_zero_readings_agree_with_nothing():
    readings = {"1D_trend": 0.0, "1D_structure": 0.0, "1W_trend": 1.0, "1W_structure": 1.0}
    assert alignment_against(readings, +1) == 2
    assert alignment_against(readings, -1) == 0


def test_unobservable_reading_fails_rather_than_counting_as_neutral():
    """NaN is absence. Absence must not be admitted as agreement."""
    readings = {"1D_trend": np.nan, "1D_structure": 1.0, "1W_trend": 1.0, "1W_structure": 1.0}
    assert alignment_against(readings, +1) is None


def test_threshold_is_three_of_four_as_registered():
    assert MTF_MIN_ALIGNMENT == 3
    assert N_PERIODS == 20
    assert len(READING_LABELS) == 4


# ---------------------------------------------------------------------- #
# I9 on the 1d/1w readings
# ---------------------------------------------------------------------- #


def test_daily_reading_is_not_visible_inside_its_own_session():
    """The funding defect in another shape: a 1d bar for date D describes
    D 00:00-24:00 and is not knowable until D+1 00:00."""
    days = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    daily = pd.DataFrame({
        "close": np.arange(60.0), "high": np.arange(60.0) + 1, "low": np.arange(60.0) - 1,
    }, index=days)
    weeks = pd.date_range("2024-01-01", periods=60, freq="7D", tz="UTC")
    weekly = pd.DataFrame({
        "close": np.arange(60.0), "high": np.arange(60.0) + 1, "low": np.arange(60.0) - 1,
    }, index=weeks)

    bars = pd.date_range("2024-02-20T00:00Z", periods=48, freq="1h")
    out = mtf_readings_at_bars(daily, weekly, bars)

    # The reading carried at 2024-02-20 12:00 must come from 2024-02-19 or
    # earlier -- never from the 2024-02-20 daily bar, which has not closed.
    d_expected = signed_trend(daily["close"]).loc[pd.Timestamp("2024-02-19", tz="UTC")]
    assert out.loc[pd.Timestamp("2024-02-20T12:00Z"), "1D_trend"] == d_expected


def test_pre_observation_bars_are_nan_not_zero():
    days = pd.date_range("2024-06-01", periods=40, freq="1D", tz="UTC")
    daily = pd.DataFrame({
        "close": np.arange(40.0), "high": np.arange(40.0) + 1, "low": np.arange(40.0) - 1,
    }, index=days)
    weekly = daily.iloc[::7]
    bars = pd.date_range("2024-01-01T00:00Z", periods=72, freq="1h")  # all before the data
    out = mtf_readings_at_bars(daily, weekly, bars)
    assert out.isna().all().all()
    assert (out == 0).sum().sum() == 0


# ---------------------------------------------------------------------- #
# Condition 3 admission
# ---------------------------------------------------------------------- #


def test_condition_3_admits_risk_on_and_transitional_only():
    states = pd.Series(["Risk-On", "Transitional", "Risk-Off", "Macro-Event Window", None])
    out = condition_3_holds(states)
    assert out.tolist() == [True, True, False, False, False]
