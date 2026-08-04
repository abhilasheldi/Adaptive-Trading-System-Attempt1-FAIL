"""Unit tests for data engine gap classification and outlier detection."""

import pandas as pd
import pytest

from src.data_engine.crypto_fetcher import classify_gaps, flag_outliers


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    data = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
    }
    df = pd.DataFrame(data, index=dates)
    return df


def insert_gap(df: pd.DataFrame, position: int, gap: pd.Timedelta) -> pd.DataFrame:
    """Return a copy of df with the timestamp at `position` pushed forward by `gap`,
    creating a single gap immediately before that row. Everything from `position`
    onward shifts with it, so relative spacing after the gap is preserved.
    """
    out = df.copy()
    new_index = out.index[:position].tolist() + [ts + gap for ts in out.index[position:]]
    out.index = pd.DatetimeIndex(new_index)
    return out


def test_classify_gaps_normal(sample_ohlcv: pd.DataFrame) -> None:
    """Test that regular intervals are classified as normal."""
    classifications = classify_gaps(sample_ohlcv, timeframe_minutes=60)
    assert (classifications == "normal").all()


def test_classify_gaps_outage(sample_ohlcv: pd.DataFrame) -> None:
    """Test classification of a large gap (>48h) as exchange outage."""
    df_gapped = insert_gap(sample_ohlcv, position=50, gap=pd.Timedelta(days=3))
    classifications = classify_gaps(df_gapped, timeframe_minutes=60)
    assert classifications.iloc[50] == "exchange_outage"
    # Nothing before the gap should be affected.
    assert (classifications.iloc[:50] == "normal").all()


def test_classify_gaps_illiquid(sample_ohlcv: pd.DataFrame) -> None:
    """Test classification of a moderate, zero-volume-adjacent gap as illiquid."""
    df_gapped = insert_gap(sample_ohlcv, position=30, gap=pd.Timedelta(hours=12))
    df_gapped.loc[df_gapped.index[30], "volume"] = 0.0
    classifications = classify_gaps(df_gapped, timeframe_minutes=60)
    assert classifications.iloc[30] == "illiquid_period"


def test_classify_gaps_data_error(sample_ohlcv: pd.DataFrame) -> None:
    """A moderate gap with normal (non-zero) volume on both sides is neither an
    outage nor explained by illiquidity -- it should fall through to data_error.
    This path was previously untested."""
    df_gapped = insert_gap(sample_ohlcv, position=30, gap=pd.Timedelta(hours=12))
    # volume left at the fixture default (1000.0) on both sides of the gap
    classifications = classify_gaps(df_gapped, timeframe_minutes=60)
    assert classifications.iloc[30] == "data_error"


def test_classify_gaps_tolerance_scales_with_timeframe(sample_ohlcv: pd.DataFrame) -> None:
    """The same absolute gap should be judged differently depending on the
    configured candle interval: tolerance is `1.5 * timeframe_minutes`, not a
    fixed number of hours, so a 5h gap is unremarkable on a 4h timeframe
    (tolerance=6h) but is a real gap on a 1h timeframe (tolerance=1.5h)."""
    df_gapped = insert_gap(sample_ohlcv, position=30, gap=pd.Timedelta(hours=5))

    on_4h = classify_gaps(df_gapped, timeframe_minutes=240)
    assert on_4h.iloc[30] == "normal"

    on_1h = classify_gaps(df_gapped, timeframe_minutes=60)
    assert on_1h.iloc[30] != "normal"


def test_flag_outliers_normal(sample_ohlcv: pd.DataFrame) -> None:
    """Test that normal candles are not flagged as outliers."""
    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert not outliers.any()


def test_flag_outliers_extreme_wick(sample_ohlcv: pd.DataFrame) -> None:
    """Test detection of extreme wicks exceeding threshold."""
    sample_ohlcv.loc[sample_ohlcv.index[20], "high"] = 115.0
    sample_ohlcv.loc[sample_ohlcv.index[20], "low"] = 85.0

    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert bool(outliers.iloc[20]) is True


def test_flag_outliers_extreme_return(sample_ohlcv: pd.DataFrame) -> None:
    """Test detection of extreme price returns exceeding threshold."""
    sample_ohlcv.loc[sample_ohlcv.index[40], "close"] = 115.0

    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert bool(outliers.iloc[40]) is True


def test_flag_outliers_first_row_can_be_flagged(sample_ohlcv: pd.DataFrame) -> None:
    """The very first candle has no prior close, so its return-based check is
    inherently unverifiable -- but a genuine range-based outlier on that same
    candle must still be caught, not blanket-suppressed."""
    sample_ohlcv.loc[sample_ohlcv.index[0], "high"] = 130.0
    sample_ohlcv.loc[sample_ohlcv.index[0], "low"] = 70.0

    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert bool(outliers.iloc[0]) is True


def test_flag_outliers_ohlc_integrity_violation(sample_ohlcv: pd.DataFrame) -> None:
    """A close price outside the [low, high] range is an impossible candle
    regardless of magnitude, and must be flagged even if the raw % move is small."""
    sample_ohlcv.loc[sample_ohlcv.index[10], "close"] = 200.0  # far above 'high'

    outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)
    assert bool(outliers.iloc[10]) is True


def test_temporal_boundary_no_lookahead_outliers(sample_ohlcv: pd.DataFrame) -> None:
    """Ensure outlier logic only uses current and past data."""
    original_outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)

    sample_ohlcv.loc[sample_ohlcv.index[-1], "close"] = 500.0

    new_outliers = flag_outliers(sample_ohlcv, threshold_pct=10.0)

    assert (original_outliers[:-1] == new_outliers[:-1]).all()


def test_temporal_boundary_no_lookahead_gaps(sample_ohlcv: pd.DataFrame) -> None:
    """Ensure gap classification only uses current and past data -- mutating a
    late timestamp/volume must not retroactively change earlier classifications."""
    original = classify_gaps(sample_ohlcv, timeframe_minutes=60)

    mutated = sample_ohlcv.copy()
    mutated.loc[mutated.index[-1], "volume"] = 0.0

    new = classify_gaps(mutated, timeframe_minutes=60)

    assert (original.iloc[:-1] == new.iloc[:-1]).all()