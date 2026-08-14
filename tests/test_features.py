"""
Tests for `features.feature_calculator`, currently focused on the `vol_spike`
ATR run-up definition (MASTER_CONTEXT Layer 6, "ATR doubles within 5 candles").

Context: before 2026-08-14 this module had no dedicated test file at all --
only the shared no-lookahead helpers in `test_no_lookahead.py` exercised it.
These tests were added alongside the `vol_spike` correction described below.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_calculator import (
    SPIKE_LOOKBACK_CANDLES,
    SPIKE_RATIO_THRESHOLD,
    _max_atr_runup_ratio,
    calculate_features,
)
from src.macro_mtf_liquidity_vol.volatility_classifier import _max_runup_ratio


def _atr(values) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV with enough history to clear the 252-candle
    percentile warm-up."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    # high/low must bound open AND close, or calculate_features rejects the
    # frame as malformed candles.
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(100, 1000, n)},
        index=idx,
    )


# ------------------------------------------------------------------------ #
# The 2026-08-14 correction: "doubles WITHIN 5 candles", not endpoint-only
# ------------------------------------------------------------------------ #


def test_in_window_doubling_is_detected_not_just_endpoint_doubling():
    """
    Regression test for the core bug. `vol_spike` was `atr / atr.shift(5)`,
    an ENDPOINT-ONLY comparison that silently missed any doubling which
    occurred and then partly subsided inside the window. Here ATR
    QUADRUPLES mid-window but the endpoint ratio is only 11/10 = 1.1.
    """
    atr = _atr([10, 10, 40, 20, 12, 11])
    ratio = _max_atr_runup_ratio(atr)

    endpoint_only = atr.iloc[-1] / atr.iloc[0]
    assert endpoint_only < SPIKE_RATIO_THRESHOLD  # what the old formula saw
    assert ratio.iloc[-1] == pytest.approx(4.0, rel=1e-6)  # the real in-window run-up
    assert ratio.iloc[-1] >= SPIKE_RATIO_THRESHOLD


def test_run_up_is_order_aware_a_pure_decline_never_flags():
    """
    The fix must NOT be a naive max/min ratio: this window has max/min == 4.0
    but contains no doubling at all -- ATR only ever declined, the opposite
    of a tail-risk spike.
    """
    atr = _atr([40, 30, 20, 15, 12, 10])
    ratio = _max_atr_runup_ratio(atr)
    assert ratio.iloc[-1] == pytest.approx(1.0)
    assert ratio.iloc[-1] < SPIKE_RATIO_THRESHOLD


def test_ratio_is_floored_at_one():
    """A flat window and a falling window both mean 'no run-up occurred'."""
    assert _max_atr_runup_ratio(_atr([5, 5, 5, 5, 5, 5])).iloc[-1] == pytest.approx(1.0)
    assert _max_atr_runup_ratio(_atr([9, 8, 7, 6, 5, 4])).iloc[-1] == pytest.approx(1.0)


def test_endpoint_doubling_still_detected():
    """The case the old formula DID handle must keep working."""
    atr = _atr([1, 1, 1, 1, 1, 2])
    assert _max_atr_runup_ratio(atr).iloc[-1] >= SPIKE_RATIO_THRESHOLD


def test_window_is_exactly_six_values_wide():
    """A doubling that has fully aged out of the trailing window must not
    keep flagging forever."""
    # Run-up at positions 0->1, then flat for well beyond the window.
    atr = _atr([1, 2] + [2] * 10)
    ratio = _max_atr_runup_ratio(atr)

    # A full 6-value window is required, so rows before that are NaN.
    assert ratio.iloc[:SPIKE_LOOKBACK_CANDLES].isna().all()
    # Window [0..5] still contains the 1->2 run-up.
    assert ratio.iloc[SPIKE_LOOKBACK_CANDLES] >= SPIKE_RATIO_THRESHOLD
    # Window [1..6] is all 2s -- the run-up has aged out, must stop flagging.
    assert ratio.iloc[SPIKE_LOOKBACK_CANDLES + 1] == pytest.approx(1.0)


# ------------------------------------------------------------------------ #
# Cross-module equivalence -- the actual bug was two implementations of one
# MASTER_CONTEXT rule silently drifting apart.
# ------------------------------------------------------------------------ #


def test_vol_spike_matches_the_layer6_classifier_runup_definition():
    """
    `features.feature_calculator._max_atr_runup_ratio` (vectorised over a
    column) and `macro_mtf_liquidity_vol.volatility_classifier
    ._max_runup_ratio` (scalar over one 6-value window) implement the SAME
    MASTER_CONTEXT rule in two packages. They diverging is exactly the bug
    being fixed, so equivalence is asserted rather than assumed.
    """
    rng = np.random.default_rng(42)
    # Positive, finite ATR-like series -- the classifier's valid domain.
    values = np.abs(rng.lognormal(0, 0.6, 500)) + 0.05
    atr = _atr(values)
    vectorised = _max_atr_runup_ratio(atr).to_numpy()

    for i in range(SPIKE_LOOKBACK_CANDLES, len(values)):
        window = list(values[i - SPIKE_LOOKBACK_CANDLES : i + 1])
        assert vectorised[i] == pytest.approx(_max_runup_ratio(window), rel=1e-9, abs=1e-9)


def test_spike_threshold_constants_agree_across_the_two_modules():
    from src.macro_mtf_liquidity_vol import volatility_classifier as vc

    assert SPIKE_LOOKBACK_CANDLES == vc.SPIKE_LOOKBACK_CANDLES == 5
    assert SPIKE_RATIO_THRESHOLD == vc.SPIKE_RATIO_THRESHOLD == 2.0


# ------------------------------------------------------------------------ #
# No-lookahead -- the single most important rule in this project
# ------------------------------------------------------------------------ #


def test_run_up_ratio_has_no_lookahead():
    """Mutating a FUTURE value must not change any value at or before the
    cut. Built only from .shift(k >= 0), but asserted rather than trusted."""
    values = list(np.linspace(1.0, 3.0, 60))
    baseline = _max_atr_runup_ratio(_atr(values)).to_numpy()

    cut = 30
    mutated_values = list(values)
    mutated_values[cut + 2] *= 50.0  # a violent future spike
    mutated = _max_atr_runup_ratio(_atr(mutated_values)).to_numpy()

    np.testing.assert_allclose(baseline[: cut + 1], mutated[: cut + 1], equal_nan=True)
    # Sanity: the mutation must actually have had an effect later on.
    assert not np.allclose(
        baseline[cut + 2 : cut + 6], mutated[cut + 2 : cut + 6], equal_nan=True
    )


# ------------------------------------------------------------------------ #
# End-to-end shape/contract through calculate_features
# ------------------------------------------------------------------------ #


def test_vol_spike_is_a_clean_binary_column_with_no_nans():
    """Warm-up rows must be filled with 0.0 (no spike), preserving the
    pre-existing contract."""
    feats = calculate_features(_synthetic_ohlcv())
    assert "vol_spike" in feats.columns
    assert feats["vol_spike"].isna().sum() == 0
    assert set(feats["vol_spike"].unique()) <= {0.0, 1.0}


def test_vol_spike_is_not_percentile_ranked():
    """It is registered via `add_excluded`: a binary flag must survive as a
    binary flag, not be re-ranked into a continuous [0, 1] percentile."""
    feats = calculate_features(_synthetic_ohlcv())
    assert set(feats["vol_spike"].unique()) <= {0.0, 1.0}
