"""
Tests for `predictive_core.dataset`, the glue that bridges
`features.feature_calculator`'s output into `predictive_core.labeling`'s
input. See that module's docstring for why the bridge has to exist:
`calculate_features()` emits ONLY percentile-ranked feature columns (no raw
OHLCV, no raw ATR), but `apply_triple_barrier()` needs raw-scale
'close'/'high'/'low' and a raw-scale ATR to set real price-level barriers.
"""

import numpy as np
import pandas as pd

from src.predictive_core.dataset import build_asset_modeling_frame, feature_columns


def _make_raw_ohlcv(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    returns = rng.normal(0.0, 5.0, n)
    close = 20_000 + np.cumsum(returns)
    high = close + rng.uniform(1.0, 20.0, n)
    low = close - rng.uniform(1.0, 20.0, n)
    open_ = close - rng.normal(0.0, 5.0, n)
    # Keep OHLC internally consistent (high/low bound open/close).
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.lognormal(mean=5, sigma=0.5, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_build_asset_modeling_frame_has_raw_and_feature_columns():
    raw = _make_raw_ohlcv(600, seed=1)
    result = build_asset_modeling_frame(raw, max_candles=24)

    fcols = feature_columns(result)
    assert len(fcols) > 0
    # Raw OHLCV must survive into the output -- calculate_features() alone
    # drops all of it, which is exactly the gap this module exists to close.
    for col in ("open", "high", "low", "close", "volume"):
        assert col in result.columns
    # Triple-barrier labels must be present.
    for col in ("direction", "magnitude", "duration"):
        assert col in result.columns
    # The internal raw-ATR helper column must never leak into the output.
    assert not any("raw_atr" in c for c in result.columns)
    assert not result.isna().any().any()


def test_magnitude_is_price_scale_not_percentile_scale():
    """
    Regression test for the core bug this module fixes: if
    `calculate_features()`'s percentile-ranked 'vol_atr_14' (a value in
    [0, 1]) were used as the ATR for triple-barrier levels instead of a
    real, raw-scale ATR, barrier levels would sit a fraction of a dollar
    from entry regardless of actual volatility, and 'magnitude' (a
    max-favorable-excursion measured in raw price units) would come out on
    the order of 1e-3 to 1, not real BTC-scale dollar moves. With prices
    around 20,000 and realistic volatility, magnitude must be on a price
    scale consistent with tp_mult/sl_mult multiples of a real ATR (tens to
    low thousands), never sub-1.
    """
    raw = _make_raw_ohlcv(600, seed=2)
    result = build_asset_modeling_frame(raw, tp_mult=2.0, sl_mult=1.0, max_candles=24)

    assert result["magnitude"].min() > 1.0, (
        "magnitude looks percentile-scale (<=1), not raw price-scale -- "
        "the raw ATR bridge may be broken."
    )
    assert result["duration"].between(1, 24).all()
    assert result["direction"].isin([-1.0, 0.0, 1.0]).all()


def test_feature_columns_are_percentile_ranked_but_raw_ohlcv_is_not():
    raw = _make_raw_ohlcv(600, seed=3)
    result = build_asset_modeling_frame(raw, max_candles=24)
    fcols = feature_columns(result)

    continuous_like = [c for c in fcols if result[c].nunique() > 2]
    assert continuous_like, "expected at least one non-binary feature column to check ranking on"
    for col in continuous_like:
        assert result[col].min() >= 0.0 and result[col].max() <= 1.0, (
            f"{col} is a feature_columns() column and should be percentile-ranked into [0, 1]"
        )

    # Raw OHLCV must NOT be percentile-ranked -- it should retain real scale.
    assert result["close"].max() > 100  # nowhere near a [0,1] percentile scale
