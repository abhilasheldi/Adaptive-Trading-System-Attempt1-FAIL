"""Synthetic OHLCV + feature data generator standing in for Layer 1/Layer 2
output, so the Predictive Core can be tested end-to-end without real market
data. Not part of the shipped package -- test scaffolding only."""

import numpy as np
import pandas as pd


def make_synthetic_asset(
    n: int, start: str, freq: str, seed: int, drift: float = 0.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)

    returns = rng.normal(drift, 1.0, n)
    close = 100 + np.cumsum(returns)
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    vol_atr_14 = pd.Series(np.abs(rng.normal(1.0, 0.3, n))).rolling(14, min_periods=1).mean().values

    # A handful of synthetic "Layer 2" style features, percentile-ranked
    # over a trailing window the way the real Feature Store would, to at
    # least structurally resemble the spec's normalization protocol.
    rsi_raw = pd.Series(close).diff().rolling(14, min_periods=1).mean().fillna(0).values
    momentum_5 = pd.Series(close).pct_change(5).fillna(0).values
    volume = rng.lognormal(mean=8, sigma=0.5, size=n)
    volume_z = (volume - pd.Series(volume).rolling(20, min_periods=1).mean().values) / (
        pd.Series(volume).rolling(20, min_periods=1).std(ddof=0).values + 1e-6
    )

    df = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "vol_atr_14": vol_atr_14,
            "feat_rsi": rsi_raw,
            "feat_momentum_5": momentum_5,
            "feat_volume_z": np.nan_to_num(volume_z),
        },
        index=idx,
    )
    return df


FEATURE_COLS = ["feat_rsi", "feat_momentum_5", "feat_volume_z"]