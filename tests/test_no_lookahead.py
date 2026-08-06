"""
Reusable pytest helpers for enforcing strict temporal boundaries and preventing lookahead bias.
"""

from typing import Callable, Union

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features"))
from src.features.feature_calculator import (
    _calculate_structural_features,
    _calculate_volatility_features,
    _calculate_momentum_features,
    LOOKBACK_PERCENTILE,
)

def assert_no_lookahead_bias(
    df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]],
    timestamp_col: str = "timestamp",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Asserts that a feature calculation function does not use future data.

    This helper iterates through each row of the provided dataframe, computes the feature
    using only data up to and including that row, and verifies that the result matches
    the value computed on the full dataset. If any discrepancy is found, it raises an
    AssertionError indicating lookahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing historical data and a timestamp column.
    feature_func : Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]]
        A callable that accepts a DataFrame and returns the computed feature(s).
    timestamp_col : str, default "timestamp"
        The name of the column representing timestamps.
    rtol : float, default 1e-5
        Relative tolerance for numerical comparisons.
    atol : float, default 1e-8
        Absolute tolerance for numerical comparisons.

    Raises
    ------
    AssertionError
        If the feature value at any row differs when computed on a sliced dataset versus
        the full dataset, indicating potential lookahead bias.

    Notes
    -----
    This checks *temporal* consistency only: it verifies that a feature computed with
    data truncated at row i never differs from the value it takes on the full dataset.
    It does NOT know whether your `timestamp_col` represents a bar's open or close time,
    so it will happily "pass" a feature that legitimately uses the current row's data
    (e.g. a 5-period SMA including today's close). If your features must exclude the
    current row entirely (e.g. because the timestamp marks the bar's open, and the
    close isn't known yet), that's a semantic choice you need to enforce in
    `feature_func` itself (e.g. by shifting inputs by 1) -- this helper can't infer it.
    """
    # Ensure chronological order and reset index for consistent slicing
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

    # Compute feature on the complete dataset
    full_result = feature_func(df_sorted)

    # Normalize output to DataFrame for uniform handling
    if isinstance(full_result, pd.Series):
        full_df = full_result.to_frame()
    else:
        full_df = full_result.copy()

    if len(full_df) != len(df_sorted):
        raise AssertionError(
            f"feature_func returned {len(full_df)} rows for an input of {len(df_sorted)} "
            "rows. This helper assumes a 1:1, order-preserving mapping between input rows "
            "and output rows; row-dropping/reordering feature_funcs can't be safely checked "
            "by position and will invalidate the comparisons below."
        )

    for i in range(len(df_sorted)):
        # Slice dataset to include only data available up to and including the current row
        df_window = df_sorted.iloc[: i + 1].copy()
        window_result = feature_func(df_window)

        # Normalize window result
        if isinstance(window_result, pd.Series):
            window_df = window_result.to_frame()
        else:
            window_df = window_result.copy()

        full_row = full_df.iloc[i]
        window_row = window_df.iloc[-1]

        # Check column-by-column rather than skipping the whole row on any NaN.
        # A single slow-to-warm-up column (e.g. a 500-period SMA on a short frame)
        # must never suppress checks on the OTHER columns for this row.
        for col in full_df.columns:
            full_val = full_row[col]

            if pd.isna(full_val):
                # We have no known-good value to check against for this cell, so it's
                # unverifiable here -- but that's a per-cell judgment, not a per-row one.
                continue

            if col not in window_df.columns:
                raise AssertionError(
                    f"feature_func did not return column '{col}' when computed on the "
                    f"truncated window ending at index {i} "
                    f"({df_sorted.iloc[i][timestamp_col]}), but it did on the full dataset."
                )

            window_val = window_row[col]

            if pd.isna(window_val):
                # Full dataset has a real value here, but truncating to "data available
                # as of this row" produces NaN. That's the signature of a feature that
                # needs rows beyond i to resolve -- i.e. it's reaching into the future.
                raise AssertionError(
                    f"Lookahead bias detected in column '{col}' at index {i} "
                    f"({df_sorted.iloc[i][timestamp_col]}): value is NaN when computed "
                    f"using only data available up to this row, but is "
                    f"{full_val!r} when computed on the full dataset. This means the "
                    "feature depends on rows that don't exist yet at this point in time."
                )

            np.testing.assert_allclose(
                window_val,
                full_val,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"Lookahead bias detected in column '{col}' at index {i} "
                    f"({df_sorted.iloc[i][timestamp_col]}): value computed from data "
                    "truncated at this row does not match the value computed on the "
                    "full dataset."
                ),
            )


def make_ohlcv(n=400, seed=1, start_price=100.0) -> pd.DataFrame:
    """Generate a valid synthetic OHLCV series (high/low always bound open/close)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    close = start_price + np.cumsum(rng.normal(0, 1, n))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + rng.uniform(0, 1.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1.5, n)
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
 
 
def assert_no_lookahead(compute_fn, df: pd.DataFrame, feature_col: str,
                         cutoff: int, atol: float = 1e-9) -> None:
    """Assert `feature_col`, as produced by `compute_fn(df) -> DataFrame`,
    is unaffected by perturbing all candles strictly after `cutoff`.
 
    Args:
        compute_fn: callable taking an OHLCV DataFrame and returning a
            DataFrame of features (e.g. one of the module's private
            `_calculate_*_features` functions, called via a thin wrapper
            that returns `.frame`).
        df: a valid OHLCV DataFrame to use as the base series.
        feature_col: the column name to check.
        cutoff: integer row position; rows [0, cutoff] must be stable.
    """
    baseline = compute_fn(df)
 
    perturbed_df = df.copy()
    n = len(df)
    future_len = n - (cutoff + 1)
    if future_len <= 0:
        raise ValueError("cutoff leaves no future rows to perturb")
 
    rng = np.random.default_rng(999)
    # Replace the future segment with an entirely different, still-valid
    # random walk, so any dependency on future values is caught rather than
    # accidentally preserved by re-using similar magnitudes.
    seg_close = perturbed_df["close"].iloc[cutoff] + np.cumsum(
        rng.normal(0, 5, future_len)
    )
    seg_open = np.empty(future_len)
    seg_open[0] = perturbed_df["close"].iloc[cutoff]
    seg_open[1:] = seg_close[:-1]
    seg_high = np.maximum(seg_open, seg_close) + rng.uniform(0, 3, future_len)
    seg_low = np.minimum(seg_open, seg_close) - rng.uniform(0, 3, future_len)
    seg_volume = rng.uniform(100, 5000, future_len)
 
    future_idx = perturbed_df.index[cutoff + 1:]
    perturbed_df.loc[future_idx, "open"] = seg_open
    perturbed_df.loc[future_idx, "close"] = seg_close
    perturbed_df.loc[future_idx, "high"] = seg_high
    perturbed_df.loc[future_idx, "low"] = seg_low
    perturbed_df.loc[future_idx, "volume"] = seg_volume
 
    perturbed = compute_fn(perturbed_df)
 
    base_slice = baseline[feature_col].iloc[: cutoff + 1]
    pert_slice = perturbed[feature_col].iloc[: cutoff + 1]
 
    mismatches = ~np.isclose(
        base_slice.to_numpy(dtype=float),
        pert_slice.to_numpy(dtype=float),
        atol=atol,
        equal_nan=True,
    )
    if mismatches.any():
        bad_positions = np.flatnonzero(mismatches)
        raise AssertionError(
            f"Lookahead detected in '{feature_col}': values at row(s) "
            f"{bad_positions[:10].tolist()} (of {mismatches.sum()} total) "
            f"changed when only future data was perturbed.\n"
            f"baseline: {base_slice.to_numpy()[bad_positions[:5]]}\n"
            f"perturbed: {pert_slice.to_numpy()[bad_positions[:5]]}"
        )
 
 
@pytest.fixture
def ohlcv():
    return make_ohlcv(n=400, seed=42)
 
 
def test_no_lookahead_struct_bos_bullish(ohlcv):
    """struct_bos_bullish must only depend on prior candles' structure."""
    compute = lambda df: _calculate_structural_features(df).frame
    assert_no_lookahead(compute, ohlcv, "struct_bos_bullish", cutoff=300)
 
 
def test_no_lookahead_struct_dist_to_resist(ohlcv):
    """Regression guard for the bug where dist_to_resist used a
    current-candle-inclusive rolling high instead of prior structure —
    this specifically would NOT have caught that bug (both are backward
    looking), but it does guard against a future regression that
    accidentally introduces a centered or forward-looking window."""
    compute = lambda df: _calculate_structural_features(df).frame
    assert_no_lookahead(compute, ohlcv, "struct_dist_to_resist", cutoff=300)
 
 
def test_no_lookahead_vol_atr_pct_rank(ohlcv):
    """The trailing-252 percentile rank must not shift when future candles
    change — this is the highest-risk feature for accidental lookahead
    since it's a rolling rank over a large window."""
    compute = lambda df: _calculate_volatility_features(df).frame
    assert_no_lookahead(compute, ohlcv, "vol_atr_pct_rank", cutoff=300)
 
 
def test_no_lookahead_mom_rsi_div_bullish(ohlcv):
    """RSI divergence proxy must only look at trailing rolling windows."""
    compute = lambda df: _calculate_momentum_features(df).frame
    assert_no_lookahead(compute, ohlcv, "mom_rsi_div_bullish", cutoff=300)
 
 
def test_full_pipeline_produces_no_nans_and_scales_across_assets():
    """End-to-end smoke test: BTC-scale and ETH-scale price series both
    produce fully-populated, [0,1]-bounded continuous feature vectors."""

    from src.features.feature_calculator import calculate_features
 
    btc = make_ohlcv(n=400, seed=2, start_price=60000)
    eth = make_ohlcv(n=400, seed=3, start_price=3000)
 
    btc_feats = calculate_features(btc)
    eth_feats = calculate_features(eth)
 
    assert not btc_feats.isna().any().any()
    assert not eth_feats.isna().any().any()
    assert len(btc_feats) > 0 and len(eth_feats) > 0
    assert list(btc_feats.columns) == list(eth_feats.columns)