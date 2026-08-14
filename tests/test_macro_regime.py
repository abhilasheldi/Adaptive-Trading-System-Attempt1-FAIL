"""
Tests for the Macro Regime Engine (Layer 3), src/macro_mtf_liquidity_vol/macro_regime.py.

Covers the "Done when" bar directly: all four states reachable from
synthetic input, and every result carries both a state label and the
confidence/macro_weight pair -- plus the specific behaviors the module
docstring commits to (Macro-Event Window precedence, macro_weight's
sign-independence and state-independence, tz-safety on the event-window
check, and input validation).
"""

import pandas as pd
import pytest

from src.macro_mtf_liquidity_vol.macro_regime import (
    CORRELATION_FULL_WEIGHT_THRESHOLD,
    CORRELATION_REDUCED_WEIGHT_THRESHOLD,
    DEFAULT_RISK_OFF_CORRELATION_THRESHOLD,
    DEFAULT_VIX_SPIKE_5D_ROC_PCT,
    VIX_RISK_OFF_MIN,
    VIX_RISK_ON_MAX,
    MacroSignalSnapshot,
    _clip01,
    classify_macro_regime,
)

T = pd.Timestamp("2024-01-01", tz="UTC")


def _snapshot(**overrides) -> MacroSignalSnapshot:
    defaults = dict(
        timestamp=T,
        dxy_20d_momentum_pct=0.0,
        us10y_level_pct=4.2,
        us10y_20d_direction_pct=0.1,
        vix_level=18.0,
        vix_5d_roc_pct=0.0,
        btc_dominance_20d_trend_pct=0.0,
        btc_qqq_corr_20d=0.3,
        next_high_impact_event=None,
    )
    defaults.update(overrides)
    return MacroSignalSnapshot(**defaults)


# ------------------------------------------------------------------------ #
# All four states reachable (the literal "Done when" bar)
# ------------------------------------------------------------------------ #


def test_risk_on_reachable():
    snap = _snapshot(
        dxy_20d_momentum_pct=-1.5, vix_level=14.0, vix_5d_roc_pct=-10.0,
        btc_dominance_20d_trend_pct=1.2, btc_qqq_corr_20d=0.5,
    )
    r = classify_macro_regime(snap)
    assert r.state == "Risk-On"
    assert r.trading_implication == "Full trading, standard EV thresholds"
    assert len(r.reasons) == 4  # all four Risk-On conditions reported


def test_risk_off_reachable():
    snap = _snapshot(dxy_20d_momentum_pct=2.0, vix_level=28.0, vix_5d_roc_pct=5.0, btc_qqq_corr_20d=0.85)
    r = classify_macro_regime(snap)
    assert r.state == "Risk-Off"
    assert r.trading_implication == "No new long entries in risk assets; existing positions reviewed for exit"


def test_risk_off_reachable_via_spiking_vix_not_level():
    """VIX under the level threshold but spiking on 5d ROC must still qualify
    -- MASTER_CONTEXT's condition is 'VIX > 25 OR spiking', not AND."""
    snap = _snapshot(dxy_20d_momentum_pct=1.0, vix_level=19.0, vix_5d_roc_pct=45.0, btc_qqq_corr_20d=0.9)
    r = classify_macro_regime(snap)
    assert r.state == "Risk-Off"


def test_transitional_reachable():
    snap = _snapshot(dxy_20d_momentum_pct=0.5, vix_level=18.0, vix_5d_roc_pct=5.0, btc_qqq_corr_20d=0.2)
    r = classify_macro_regime(snap)
    assert r.state == "Transitional"
    assert r.trading_implication == "Halve position sizes, apply more conservative EV threshold"


def test_macro_event_window_reachable():
    snap = _snapshot(next_high_impact_event=T + pd.Timedelta(hours=6))
    r = classify_macro_regime(snap)
    assert r.state == "Macro-Event Window"
    assert r.trading_implication == "No new entries; existing positions get tighter stops"
    assert r.confidence == 1.0


def test_every_result_carries_state_and_confidence_and_weight():
    for snap in [
        _snapshot(dxy_20d_momentum_pct=-1.5, vix_level=14.0, vix_5d_roc_pct=-10.0,
                  btc_dominance_20d_trend_pct=1.2, btc_qqq_corr_20d=0.5),
        _snapshot(dxy_20d_momentum_pct=2.0, vix_level=28.0, vix_5d_roc_pct=5.0, btc_qqq_corr_20d=0.85),
        _snapshot(dxy_20d_momentum_pct=0.5, vix_level=18.0, vix_5d_roc_pct=5.0, btc_qqq_corr_20d=0.2),
        _snapshot(next_high_impact_event=T + pd.Timedelta(hours=6)),
    ]:
        r = classify_macro_regime(snap)
        assert isinstance(r.state, str) and r.state
        assert 0.0 <= r.confidence <= 1.0
        assert 0.0 <= r.macro_weight <= 1.0
        assert r.reasons  # non-empty, always explainable


# ------------------------------------------------------------------------ #
# Macro-Event Window precedence
# ------------------------------------------------------------------------ #


def test_event_window_overrides_otherwise_risk_on_signals():
    """The exact precedence claim in the module docstring: an imminent event
    must win even when every other signal reads cleanly Risk-On."""
    snap = _snapshot(
        dxy_20d_momentum_pct=-1.5, vix_level=14.0, vix_5d_roc_pct=-10.0,
        btc_dominance_20d_trend_pct=1.2, btc_qqq_corr_20d=0.5,
        next_high_impact_event=T + pd.Timedelta(hours=1),
    )
    assert classify_macro_regime(snap).state == "Macro-Event Window"


def test_event_at_exact_24h_boundary_triggers():
    snap = _snapshot(next_high_impact_event=T + pd.Timedelta(hours=24))
    assert classify_macro_regime(snap).state == "Macro-Event Window"


def test_event_just_past_24h_boundary_does_not_trigger():
    snap = _snapshot(next_high_impact_event=T + pd.Timedelta(hours=24, minutes=1))
    assert classify_macro_regime(snap).state != "Macro-Event Window"


def test_event_in_the_past_does_not_trigger():
    snap = _snapshot(next_high_impact_event=T - pd.Timedelta(hours=1))
    assert classify_macro_regime(snap).state != "Macro-Event Window"


def test_event_window_check_is_tz_safe():
    """Regression-shaped test mirroring the tz-aware/tz-naive bug found and
    fixed in the since-deleted k-NN analog engine earlier this project: a
    tz-naive event timestamp against a tz-aware snapshot timestamp must not
    raise, and must be treated as the same wall-clock UTC instant."""
    naive_event = pd.Timestamp("2024-01-01 06:00:00")  # no tz, same instant as T + 6h in UTC
    snap = _snapshot(next_high_impact_event=naive_event)
    r = classify_macro_regime(snap)  # must not raise
    assert r.state == "Macro-Event Window"


# ------------------------------------------------------------------------ #
# macro_weight: sign-independent, state-independent, driven by |correlation|
# ------------------------------------------------------------------------ #


def test_macro_weight_is_sign_independent():
    r_pos = classify_macro_regime(_snapshot(btc_qqq_corr_20d=0.9))
    r_neg = classify_macro_regime(_snapshot(btc_qqq_corr_20d=-0.9))
    assert r_pos.macro_weight == pytest.approx(r_neg.macro_weight)


def test_macro_weight_low_when_correlation_near_zero():
    r_low = classify_macro_regime(_snapshot(btc_qqq_corr_20d=0.05))
    r_high = classify_macro_regime(_snapshot(btc_qqq_corr_20d=0.9))
    assert r_low.macro_weight < r_high.macro_weight
    assert r_low.macro_weight < 0.5


def test_macro_weight_computed_the_same_way_regardless_of_state():
    """A Risk-On read and a Risk-Off read with the same |correlation| must
    get the same macro_weight -- it's explicitly independent of state and
    of confidence (module docstring)."""
    risk_on = _snapshot(dxy_20d_momentum_pct=-1.5, vix_level=14.0, vix_5d_roc_pct=-10.0,
                         btc_dominance_20d_trend_pct=1.2, btc_qqq_corr_20d=0.8)
    risk_off = _snapshot(dxy_20d_momentum_pct=2.0, vix_level=28.0, vix_5d_roc_pct=5.0, btc_qqq_corr_20d=0.8)
    r_on = classify_macro_regime(risk_on)
    r_off = classify_macro_regime(risk_off)
    assert r_on.state == "Risk-On" and r_off.state == "Risk-Off"
    assert r_on.macro_weight == pytest.approx(r_off.macro_weight)


# ------------------------------------------------------------------------ #
# Confidence is graded, not trivially 1.0
# ------------------------------------------------------------------------ #


def test_deeper_risk_on_signals_produce_higher_confidence():
    marginal = _snapshot(  # just barely clears each Risk-On threshold
        dxy_20d_momentum_pct=-0.01, vix_level=19.9, vix_5d_roc_pct=-0.01,
        btc_dominance_20d_trend_pct=0.01, btc_qqq_corr_20d=0.01,
    )
    deep = _snapshot(  # deep in Risk-On territory
        dxy_20d_momentum_pct=-5.0, vix_level=10.0, vix_5d_roc_pct=-30.0,
        btc_dominance_20d_trend_pct=5.0, btc_qqq_corr_20d=0.9,
    )
    r_marginal = classify_macro_regime(marginal)
    r_deep = classify_macro_regime(deep)
    assert r_marginal.state == "Risk-On" and r_deep.state == "Risk-On"
    assert r_deep.confidence > r_marginal.confidence


def test_transitional_confidence_is_high_when_genuinely_mixed():
    """Both soft scores low (genuinely balanced/undecided) should read as a
    confidently Transitional call, not a low-confidence one."""
    genuinely_mixed = _snapshot(dxy_20d_momentum_pct=0.1, vix_level=18.0, vix_5d_roc_pct=1.0, btc_qqq_corr_20d=0.05)
    r = classify_macro_regime(genuinely_mixed)
    assert r.state == "Transitional"
    assert r.confidence > 0.7


# ------------------------------------------------------------------------ #
# US10Y is echoed, never used in classification (module docstring claim)
# ------------------------------------------------------------------------ #


def test_us10y_does_not_affect_classification_but_is_echoed():
    base = dict(dxy_20d_momentum_pct=-1.5, vix_level=14.0, vix_5d_roc_pct=-10.0,
                btc_dominance_20d_trend_pct=1.2, btc_qqq_corr_20d=0.5)
    low_yield = classify_macro_regime(_snapshot(us10y_level_pct=0.5, us10y_20d_direction_pct=-2.0, **base))
    high_yield = classify_macro_regime(_snapshot(us10y_level_pct=8.0, us10y_20d_direction_pct=3.0, **base))
    assert low_yield.state == high_yield.state == "Risk-On"
    assert low_yield.confidence == pytest.approx(high_yield.confidence)
    assert low_yield.macro_weight == pytest.approx(high_yield.macro_weight)
    assert low_yield.us10y_level_pct == 0.5
    assert high_yield.us10y_level_pct == 8.0


# ------------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------------ #


def test_rejects_non_timestamp_timestamp():
    with pytest.raises(TypeError):
        _snapshot(timestamp="2024-01-01")


def test_rejects_non_positive_vix():
    with pytest.raises(ValueError):
        _snapshot(vix_level=0.0)


def test_rejects_correlation_out_of_range():
    with pytest.raises(ValueError):
        _snapshot(btc_qqq_corr_20d=1.5)


def test_exact_spec_vix_thresholds_are_20_and_25():
    """Locks in the exact MASTER_CONTEXT numbers as module constants, not
    approximated/paraphrased elsewhere."""
    assert VIX_RISK_ON_MAX == 20.0
    assert VIX_RISK_OFF_MIN == 25.0


# ------------------------------------------------------------------------ #
# 2026-08-13 review fixes: exact correlation-weight thresholds, echoed
# correlation coefficient, and the explicit macro-event-window flag.
# ------------------------------------------------------------------------ #


def test_exact_correlation_weight_thresholds_are_0_6_and_0_3():
    """Locks in the exact numbers from the Adaptive doc's Layer 3 section
    ("correlation > 0.6 -> full weight", "< 0.3 -> reduced"), found during
    the 2026-08-13 review -- the first version of this module used an
    invented 0.5 threshold instead."""
    assert CORRELATION_FULL_WEIGHT_THRESHOLD == 0.6
    assert CORRELATION_REDUCED_WEIGHT_THRESHOLD == 0.3


def test_worked_example_correlation_0_64_gives_full_macro_weight():
    """Direct regression test against the Adaptive doc's own worked example:
    'BTC/QQQ corr: 0.64 -> macro weight: FULL'."""
    r = classify_macro_regime(_snapshot(btc_qqq_corr_20d=0.64))
    assert r.macro_weight == pytest.approx(1.0)


def test_macro_weight_exactly_1_at_the_full_weight_threshold():
    r = classify_macro_regime(_snapshot(btc_qqq_corr_20d=CORRELATION_FULL_WEIGHT_THRESHOLD))
    assert r.macro_weight == pytest.approx(1.0)


def test_macro_weight_is_clearly_reduced_at_the_reduced_weight_threshold():
    """At exactly the 0.3 'reduced' reference point, weight must be
    unambiguously below full (0.5 under the documented linear-ramp
    formula) -- not full, and not near-zero either."""
    r = classify_macro_regime(_snapshot(btc_qqq_corr_20d=CORRELATION_REDUCED_WEIGHT_THRESHOLD))
    assert r.macro_weight == pytest.approx(0.5)
    assert 0.0 < r.macro_weight < 1.0


def test_macro_weight_never_exceeds_1_above_the_full_weight_threshold():
    r = classify_macro_regime(_snapshot(btc_qqq_corr_20d=0.95))
    assert r.macro_weight == pytest.approx(1.0)


def test_correlation_coefficient_is_echoed_on_every_result():
    """Adaptive doc Layer 3 'Output:' line explicitly requires 'current
    BTC/QQQ correlation coefficient' as its own field, not just the
    derived macro_weight."""
    for corr in (-0.7, 0.0, 0.42):
        r = classify_macro_regime(_snapshot(btc_qqq_corr_20d=corr))
        assert r.btc_qqq_corr_20d == pytest.approx(corr)


def test_is_macro_event_window_flag_matches_state():
    """Adaptive doc Layer 3 'Output:' line explicitly requires a
    'macro-event window flag' distinct from the regime label."""
    event = classify_macro_regime(_snapshot(next_high_impact_event=T + pd.Timedelta(hours=2)))
    assert event.is_macro_event_window is True
    assert event.state == "Macro-Event Window"

    non_event = classify_macro_regime(_snapshot(
        dxy_20d_momentum_pct=-1.5, vix_level=14.0, vix_5d_roc_pct=-10.0,
        btc_dominance_20d_trend_pct=1.2, btc_qqq_corr_20d=0.5,
    ))
    assert non_event.is_macro_event_window is False
    assert non_event.state != "Macro-Event Window"


# ------------------------------------------------------------------------ #
# 2026-08-14 critical review findings
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "field",
    [
        "dxy_20d_momentum_pct",
        "us10y_level_pct",
        "us10y_20d_direction_pct",
        "vix_level",
        "vix_5d_roc_pct",
        "btc_dominance_20d_trend_pct",
        "btc_qqq_corr_20d",
    ],
)
def test_nan_in_any_numeric_field_is_rejected(field):
    """
    Regression test: only `btc_qqq_corr_20d` was implicitly NaN-safe (via
    its range check). A NaN `vix_level` passed `nan <= 0`, every downstream
    comparison against it silently evaluated False, all hard gates
    therefore failed, and the classifier returned "Transitional" -- i.e.
    "halve position sizes", meaning the system kept trading on entirely
    missing macro data. Not theoretical: docs/project_state.md records
    yfinance returning six entirely-NaN columns from this exact pipeline.
    """
    with pytest.raises(ValueError, match="finite"):
        _snapshot(**{field: float("nan")})


def test_clip01_rejects_nan_rather_than_returning_maximum():
    """
    The root cause of the above: `max(0.0, min(1.0, nan))` returns 1.0 --
    the MAXIMUM -- because every comparison against NaN is False. Missing
    inputs therefore produced maximum-confidence margin contributions.
    """
    with pytest.raises(ValueError):
        _clip01(float("nan"))
    assert _clip01(0.5) == 0.5
    assert _clip01(-3.0) == 0.0
    assert _clip01(7.0) == 1.0


def test_risk_off_confidence_is_near_zero_at_the_bare_minimum_trigger():
    """
    The VIX-ROC leg previously read 1.0 -- the MAXIMUM -- exactly AT the
    spike threshold, while its two siblings in the same expression (the
    VIX-level and correlation legs) correctly read 0.0 at their own
    thresholds. A snapshot sitting at the bare minimum of all three
    Risk-Off conditions therefore reported inflated confidence (0.168),
    driven by a leg that should contribute nothing.
    """
    r = classify_macro_regime(_snapshot(
        dxy_20d_momentum_pct=0.01,          # barely rising
        vix_level=15.0,                     # calm level; Risk-Off triggers via ROC only
        vix_5d_roc_pct=DEFAULT_VIX_SPIKE_5D_ROC_PCT,   # exactly at the spike threshold
        btc_qqq_corr_20d=DEFAULT_RISK_OFF_CORRELATION_THRESHOLD,  # exactly at threshold
    ))
    assert r.state == "Risk-Off"
    assert r.confidence < 0.02


def test_risk_off_vix_leg_uses_or_semantics_not_an_average():
    """
    `vix_elevated_or_spiking` is an OR, but its two legs were AVERAGED,
    halving the contribution of a decisively-triggered condition: VIX at 40
    with flat ROC scored (0.6 + 0)/2 = 0.3 despite the condition being
    unambiguously met. The satisfied disjunct must govern.
    """
    decisive_level = classify_macro_regime(_snapshot(
        dxy_20d_momentum_pct=3.0, vix_level=40.0, vix_5d_roc_pct=0.0,
        btc_qqq_corr_20d=0.95,
    ))
    bare_minimum = classify_macro_regime(_snapshot(
        dxy_20d_momentum_pct=0.01, vix_level=15.0,
        vix_5d_roc_pct=DEFAULT_VIX_SPIKE_5D_ROC_PCT,
        btc_qqq_corr_20d=DEFAULT_RISK_OFF_CORRELATION_THRESHOLD,
    ))
    assert decisive_level.state == bare_minimum.state == "Risk-Off"
    assert decisive_level.confidence > 0.7
    assert decisive_level.confidence > bare_minimum.confidence
