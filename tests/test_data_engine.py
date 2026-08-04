"""Unit tests for data engine gap classification and outlier detection."""

import pandas as pd
import pytest

from src.data_engine.crypto_fetcher import classify_gaps, flag_outliers


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    data = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
    }
    df = pd.DataFrame(data, index=dates)
    return df


def test_classify_gaps_normal(sample_ohlcv: pd.DataFrame) -> None:
    """Test that regular intervals are classified as normal."""
    classifications = classify_gaps(sample_ohlcv, timeframe_minutes=60)
    assert (classifications == "normal").all()


def test_classify_gaps_outage(sample_ohlcv: pd.DataFrame) -> None:
    """Test classification of a large gap (>48h) as exchange outage."""
    # Insert a 3-day gap at index 50
    new_index = sample_ohlcv.index[:50].tolist() + \
                (sample_ohlcv.index[50] + pd.Timedelta(days=3)).to_series().index.tolist() + \
                sample_ohlcv.index[51:].tolist()
    df_gapped = sample_ohlcv.copy()
    df_gapped.index = new_index
    
    classifications = classify_gaps(df_gapped, timeframe_minutes=60)
    assert classifications.iloc[50] == "exchange_outage"


def test_classify_gaps_illiquid(sample_ohlcv: pd.DataFrame) -> None:
    """Test classification of a gap with zero volume as illiquid period."""
    # Insert a 12-hour gap at index 30
    new_index = sample_ohlcv.index[:30].tolist() + \
                (sample_ohlcv.index[30] + pd.Timedelta(hours=12)).to_series().index.tolist() + \
                sample_ohlcv.index[31:].tolist()
    df_gapped = sample_ohlcv.copy()
    df_gapped.index = new_index
    df_gapped.loc[df_gapped.index[30], "volume"] = 0.0
    
    classifications = classify_gaps(df_gapped, timeframe_minutes=60)
    assert classifications.iloc[30] == "illiquid_period"


def test_flag_outliers_normal(sample_ohlcv: pd.DataFrame) -> None:
    """Test that normal candles are not flagged as outliers."""
    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert not outliers.any()


def test_flag_outliers_extreme_wick(sample_ohlcv: pd.DataFrame) -> None:
    """Test detection of extreme wicks exceeding threshold."""
    # Create a candle with a 15% range
    sample_ohlcv.loc[sample_ohlcv.index[20], "high"] = 115.0
    sample_ohlcv.loc[sample_ohlcv.index[20], "low"] = 85.0
    
    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert outliers.iloc[20] is True


def test_flag_outliers_extreme_return(sample_ohlcv: pd.DataFrame) -> None:
    """Test detection of extreme price returns exceeding threshold."""
    # Create a 15% jump in close price
    sample_ohlcv.loc[sample_ohlcv.index[40], "close"] = 115.0
    
    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert outliers.iloc[40] is True


def test_temporal_boundary_no_lookahead(sample_ohlcv: pd.DataFrame) -> None:
    """Ensure gap/outlier logic only uses current and past data."""
    # The functions use .diff() and .pct_change(), which are strictly backward-looking.
    # We verify that modifying future candles does not affect past classifications.
    original_outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    
    # Mutate a future candle drastically
    sample_ohlcv.loc[sample_ohlcv.index[-1], "close"] = 500.0
    
    new_outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    
    # All indices except the last one must remain unchanged
    assert (original_outliers[:-1] == new_outliers[:-1]).all()
