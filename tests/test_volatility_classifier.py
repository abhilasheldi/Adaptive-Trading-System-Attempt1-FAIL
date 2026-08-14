"""
Tests for the Volatility Regime Classifier (Layer 6),
src/macro_mtf_liquidity_vol/volatility_classifier.py.

Covers the literal "Done when" bar: all four states correctly triggered by
synthetic ATR sequences, with the Spike condition specifically verified to
be about doubling within 5 candles, not just a high percentile (i.e. a
LOW-percentile-but-doubling sequence must still classify as Spike).
"""

import pytest

from src.macro_mtf_liquidity_vol.volatility_classifier import (
    COMPRESSED_MAX_PERCENTILE,
    EXTENDED_MIN_PERCENTILE,
    SPIKE_LOOKBACK_CANDLES,
    SPIKE_RATIO_THRESHOLD,
    VolatilityStateResult,
    classify_volatility_state,
)

_FLAT_ATR = [1.0] * (SPIKE_LOOKBACK_CANDLES + 1)  # no doubling -- isolates percentile-band logic


# ------------------------------------------------------------------------ #
# Done-when bar: all four states triggered by synthetic ATR sequences
# ------------------------------------------------------------------------ #


def test_compressed_triggered_below_30th_percentile():
    r = classify_volatility_state(0.15, _FLAT_ATR)
    assert r.state == "Compressed"
    assert r.trading_implication == "Pre-breakout; reduce size until directional confirmation"


def test_expanding_triggered_in_30_to_70_band():
    r = classify_volatility_state(0.50, _FLAT_ATR)
    assert r.state == "Expanding"
    assert r.trading_implication == "Active trend initiation; standard position sizing"


def test_extended_triggered_above_70th_percentile():
    r = classify_volatility_state(0.85, _FLAT_ATR)
    assert r.state == "Extended"
    assert r.trading_implication == "Mature trend/blow-off; reduce size, tighten stops aggressively"


def test_spike_triggered_by_atr_doubling_within_5_candles():
    spiking = [1.0, 1.0, 1.0, 1.0, 1.0, 2.5]  # current=2.5, 5-candles-ago=1.0 -> ratio 2.5
    r = classify_volatility_state(0.50, spiking)
    assert r.state == "Spike"
    assert r.trading_implication == "Tail-risk event; no new entries across all assets"


def test_spike_overrides_a_low_percentile_that_would_otherwise_read_compressed():
    """The Done-when bar's explicit emphasis: Spike must be triggered by
    genuine doubling, NOT merely correlate with a high percentile -- a LOW
    percentile reading (which alone would classify as Compressed) must
    still classify as Spike if the raw ATR sequence actually doubled."""
    spiking = [1.0, 1.0, 1.0, 1.0, 1.0, 2.1]
    r = classify_volatility_state(0.05, spiking)  # deep in Compressed territory by percentile alone
    assert r.state == "Spike"


def test_spike_overrides_a_high_percentile_reading_too():
    spiking = [1.0, 1.0, 1.0, 1.0, 1.0, 2.5]
    r = classify_volatility_state(0.95, spiking)  # deep in Extended territory by percentile alone
    assert r.state == "Spike"


def test_high_percentile_without_doubling_is_extended_not_spike():
    """Sibling of the above: a high percentile alone, with NO recent
    doubling, must NOT be classified as Spike -- proves the two are
    genuinely independent signals, not proxies for each other."""
    r = classify_volatility_state(0.95, _FLAT_ATR)
    assert r.state == "Extended"


# ------------------------------------------------------------------------ #
# Exact threshold boundaries
# ------------------------------------------------------------------------ #


def test_percentile_boundary_at_exactly_30th_is_expanding_not_compressed():
    r = classify_volatility_state(COMPRESSED_MAX_PERCENTILE, _FLAT_ATR)
    assert r.state == "Expanding"


def test_percentile_just_below_30th_is_compressed():
    r = classify_volatility_state(COMPRESSED_MAX_PERCENTILE - 0.001, _FLAT_ATR)
    assert r.state == "Compressed"


def test_percentile_boundary_at_exactly_70th_is_expanding_not_extended():
    r = classify_volatility_state(EXTENDED_MIN_PERCENTILE, _FLAT_ATR)
    assert r.state == "Expanding"


def test_percentile_just_above_70th_is_extended():
    r = classify_volatility_state(EXTENDED_MIN_PERCENTILE + 0.001, _FLAT_ATR)
    assert r.state == "Extended"


def test_spike_ratio_exactly_at_threshold_triggers_spike():
    exact = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]  # ratio == 2.0 exactly
    r = classify_volatility_state(0.50, exact)
    assert r.state == "Spike"


def test_spike_ratio_just_below_threshold_does_not_trigger():
    almost = [1.0, 1.0, 1.0, 1.0, 1.0, 1.999]
    r = classify_volatility_state(0.50, almost)
    assert r.state != "Spike"
    assert r.state == "Expanding"


# ------------------------------------------------------------------------ #
# Result correctness / self-consistency
# ------------------------------------------------------------------------ #


def test_spike_ratio_always_reported_even_when_not_spiking():
    r = classify_volatility_state(0.50, _FLAT_ATR)
    assert r.spike_ratio == pytest.approx(1.0)
    assert r.state != "Spike"


def test_result_rejects_mismatched_trading_implication():
    with pytest.raises(ValueError):
        VolatilityStateResult(
            state="Compressed", trading_implication="wrong text",
            atr_percentile=0.1, spike_ratio=1.0, reasons=[],
        )


def test_result_rejects_invalid_state():
    with pytest.raises(ValueError):
        VolatilityStateResult(
            state="NotAState", trading_implication="x",
            atr_percentile=0.1, spike_ratio=1.0, reasons=[],
        )


# ------------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------------ #


def test_rejects_percentile_out_of_range():
    with pytest.raises(ValueError):
        classify_volatility_state(1.5, _FLAT_ATR)
    with pytest.raises(ValueError):
        classify_volatility_state(-0.1, _FLAT_ATR)


def test_rejects_too_short_atr_sequence():
    with pytest.raises(ValueError, match="at least"):
        classify_volatility_state(0.5, [1.0, 1.0, 1.0])


def test_rejects_non_positive_reference_atr():
    with pytest.raises(ValueError):
        classify_volatility_state(0.5, [0.0, 1.0, 1.0, 1.0, 1.0, 2.0])  # ATR-5-candles-ago == 0


def test_rejects_negative_current_atr():
    with pytest.raises(ValueError):
        classify_volatility_state(0.5, [1.0, 1.0, 1.0, 1.0, 1.0, -1.0])


# ------------------------------------------------------------------------ #
# Exact spec numbers
# ------------------------------------------------------------------------ #


def test_exact_spec_constants():
    assert COMPRESSED_MAX_PERCENTILE == 0.30
    assert EXTENDED_MIN_PERCENTILE == 0.70
    assert SPIKE_LOOKBACK_CANDLES == 5
    assert SPIKE_RATIO_THRESHOLD == 2.0


# ------------------------------------------------------------------------ #
# 2026-08-14 critical review findings
# ------------------------------------------------------------------------ #


def test_spike_detected_when_doubling_happens_inside_the_window_not_just_at_endpoints():
    """
    Regression test for the review's most severe finding. Spike detection
    previously compared ONLY `recent_atr[-1] / recent_atr[-6]`, so a
    doubling that occurred and then partly subsided inside the window was
    silently missed: this exact sequence (ATR QUADRUPLED mid-window)
    returned spike_ratio=1.10 and classified "Expanding" -- i.e. "standard
    position sizing" -- because 11/10 is close to 1.

    MASTER_CONTEXT says "ATR doubles WITHIN 5 candles", and Spike is
    simultaneously a Layer 17 kill-switch trigger, one of constraint 8's
    six entry gates, and the Risk Engine's 2.0x fat-tail multiplier -- a
    false negative here fails OPEN on all three at once.
    """
    r = classify_volatility_state(0.50, [10.0, 10.0, 40.0, 20.0, 12.0, 11.0])
    assert r.state == "Spike"
    assert r.spike_ratio == pytest.approx(4.0)  # 40 / 10, the real in-window run-up


def test_spike_ratio_is_order_aware_a_pure_decline_is_never_a_spike():
    """
    The fix must not be a naive max/min ratio: a window that starts high
    and falls has max/min == 4.0 but contains no doubling at all -- ATR
    only ever declined, the opposite of a tail-risk spike. Only genuine
    earlier-low-to-later-high moves may count.
    """
    r = classify_volatility_state(0.50, [40.0, 30.0, 20.0, 15.0, 12.0, 10.0])
    assert r.state != "Spike"
    assert r.spike_ratio <= 1.0


def test_spike_detected_when_the_run_up_ends_at_the_current_candle():
    """The endpoint case the old formula did handle must still work."""
    r = classify_volatility_state(0.50, [1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    assert r.state == "Spike"


def test_nan_atr_is_rejected_rather_than_silently_classified():
    """
    Regression test: a NaN ATR previously flowed straight through
    (`nan >= 2.0` is False, and VolatilityStateResult's `spike_ratio < 0`
    check is also False for NaN), so entirely missing volatility data
    classified as "Expanding" -- "standard position sizing" on absent data.
    """
    with pytest.raises(ValueError, match="finite"):
        classify_volatility_state(0.50, [float("nan")] * 6)

    with pytest.raises(ValueError, match="finite"):
        classify_volatility_state(0.50, [1.0, 1.0, float("nan"), 1.0, 1.0, 1.0])


def test_result_rejects_non_finite_spike_ratio():
    with pytest.raises(ValueError, match="finite"):
        VolatilityStateResult(
            state="Expanding", trading_implication="Active trend initiation; standard position sizing",
            atr_percentile=0.5, spike_ratio=float("nan"), reasons=[],
        )


def test_all_six_window_values_are_actually_examined():
    """
    The function has always REQUIRED 6 values but used to read only 2.
    Changing any interior value that creates a run-up must change the
    outcome -- otherwise the extra values are decorative.
    """
    flat = classify_volatility_state(0.50, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    interior_spike = classify_volatility_state(0.50, [1.0, 1.0, 1.0, 3.0, 1.0, 1.0])
    assert flat.state == "Expanding"
    assert interior_spike.state == "Spike"
