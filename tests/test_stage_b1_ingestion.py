"""
Stage B.1 ingestion -- funding, FRED macro, basis, and weekly timeframe
support. No network: alignment and classification logic only.

The alignment tests matter more than they look. Funding and macro are
lower-frequency series joined onto 1h bars, and the only safe direction is
forward. A back-fill would place a value on bars that preceded its
observation, which is lookahead that no splitter can catch because it is
baked into the feature.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_engine.crypto_fetcher import (
    _pandas_resample_alias,
    _timeframe_minutes,
    remove_duplicates_and_skew,
)
from src.data_engine.derivatives_fetcher import (
    FUNDING_INTERVAL_HOURS,
    classify_funding_gaps,
    resample_funding_to_bars,
)
from src.data_engine.fred_fetcher import align_to_bars, classify_macro_gaps


# ---------------------------------------------------------------------- #
# Weekly timeframe support (condition 4 needs 1H/4H/1D/1W)
# ---------------------------------------------------------------------- #


def test_weekly_and_daily_timeframes_resolve():
    assert _timeframe_minutes("1d") == 1440
    assert _timeframe_minutes("1w") == 10080
    assert _timeframe_minutes("3d") == 4320
    assert _timeframe_minutes("2w") == 20160


def test_unknown_timeframe_still_raises():
    with pytest.raises(ValueError, match="Cannot parse timeframe"):
        _timeframe_minutes("1fortnight")


def test_weekly_alias_is_translated_for_pandas():
    """ccxt says '1w'; pandas deprecated lowercase 'w' and will drop it."""
    assert _pandas_resample_alias("1w") == "1W"
    assert _pandas_resample_alias("1h") == "1h"
    assert _pandas_resample_alias("1d") == "1d"


def test_weekly_resample_runs_without_deprecation_warning():
    """pandas still resamples on lowercase 'w' today, with a FutureWarning,
    and will stop later. Promoting the warning to an error means this fails
    now rather than silently breaking on a pandas upgrade."""
    import warnings

    idx = pd.date_range("2020-01-01", periods=200, freq="1d", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                       "volume": 10.0}, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = remove_duplicates_and_skew(df, timeframe="1w")
    assert len(out) > 0
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all()


# ---------------------------------------------------------------------- #
# Funding alignment -- forward only
# ---------------------------------------------------------------------- #


def _funding(hours, rates):
    idx = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
    return pd.DataFrame({"funding_rate": rates, "mark_price": np.arange(len(rates), dtype=float)},
                        index=pd.DatetimeIndex(idx))


def test_funding_is_forward_filled_never_back_filled():
    """A rate settled at 08:00 is unknown at 04:00. Back-filling it into the
    interval it was computed over is lookahead."""
    funding = _funding([8, 16], [0.001, 0.002])
    bars = pd.date_range("2024-01-01T00:00:00Z", periods=24, freq="1h")
    out = resample_funding_to_bars(funding, bars)

    assert out.loc[bars[0:8], "funding_rate"].isna().all(), "back-filled before first settlement"
    assert out.loc[bars[8], "funding_rate"] == 0.001
    assert out.loc[bars[15], "funding_rate"] == 0.001      # held, not interpolated
    assert out.loc[bars[16], "funding_rate"] == 0.002
    assert out.loc[bars[23], "funding_rate"] == 0.002


def test_bars_before_the_first_settlement_are_nan_not_zero():
    """Absent is not zero (I5). A zero funding rate is a real market state."""
    out = resample_funding_to_bars(_funding([10], [0.003]),
                                   pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="1h"))
    assert out["funding_rate"].iloc[:10].isna().all()
    assert not (out["funding_rate"].iloc[:10] == 0).any()


def test_empty_funding_yields_all_nan_not_an_exception():
    out = resample_funding_to_bars(pd.DataFrame(columns=["funding_rate", "mark_price"]),
                                   pd.date_range("2024-01-01T00:00:00Z", periods=5, freq="1h"))
    assert len(out) == 5 and out.isna().all().all()


def test_funding_gap_classification_uses_the_8h_cadence():
    idx = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z",
                          "2024-01-05T00:00:00Z"], utc=True)   # 88h hole
    df = pd.DataFrame({"funding_rate": [0.001, 0.002, 0.003]}, index=idx)
    g = classify_funding_gaps(df)
    assert g.iloc[1] == "normal"
    assert g.iloc[2] != "normal"
    assert FUNDING_INTERVAL_HOURS == 8


# ---------------------------------------------------------------------- #
# Macro alignment and calendar-aware gap classification
# ---------------------------------------------------------------------- #


def test_macro_is_forward_filled_onto_bars_at_observation_time_not_value_date():
    """UPDATED AT B.0.4. This previously asserted that a FRED value for date
    D was visible on D's own bars -- i.e. a zero publication lag. That was
    the defect, not the contract: FRED stamps an observation with the date
    it DESCRIBES, and macro now gates event formation, so an unlagged join
    contaminates the event set itself rather than one feature column.

    `align_to_bars` now shifts by FRED_DAILY's one-business-day floor.
    Monday 2024-01-01's value becomes observable Tuesday 2024-01-02."""
    macro = pd.DataFrame({"vix": [15.0, 16.0]},
                         index=pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"], utc=True))
    bars = pd.date_range("2024-01-01T00:00:00Z", periods=48, freq="1h")
    out = align_to_bars(macro, bars)

    # Monday's bars cannot see Monday's observation.
    assert out["vix"].iloc[0] != out["vix"].iloc[0]      # NaN
    assert out["vix"].iloc[23] != out["vix"].iloc[23]    # NaN
    # Tuesday 00:00 is when Monday's value becomes knowable.
    assert out["vix"].iloc[24] == 15.0
    assert out["vix"].iloc[47] == 15.0
    # Tuesday's own value is not observable until Wednesday, beyond this range.
    assert 16.0 not in set(out["vix"].dropna())


def test_weekend_is_market_closed_not_a_data_error():
    """The crypto gap classifier would call every Saturday a data error --
    ~2,900 false gaps over 30 years. Macro needs an exchange calendar."""
    idx = pd.to_datetime(["2024-01-04T00:00:00Z", "2024-01-05T00:00:00Z", "2024-01-08T00:00:00Z"], utc=True)  # Thu, Fri, Mon
    g = classify_macro_gaps(pd.DataFrame({"vix": [1.0, 2.0, 3.0]}, index=idx), "vix")
    assert g.iloc[1] == "normal"
    assert g.iloc[2] == "market_closed"
    assert "data_gap" not in set(g)


def test_a_real_hole_is_still_flagged_as_a_data_gap():
    idx = pd.to_datetime(["2024-01-04T00:00:00Z", "2024-02-04T00:00:00Z"], utc=True)
    g = classify_macro_gaps(pd.DataFrame({"vix": [1.0, 2.0]}, index=idx), "vix")
    assert g.iloc[1] == "data_gap"


def test_macro_frame_keeps_absences_visible_as_nan():
    """Outer join across series with different holiday calendars. An inner
    join would silently discard every date any one series is missing."""
    a = pd.Series([1.0], index=pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True))
    b = pd.Series([2.0], index=pd.to_datetime(["2024-01-02T00:00:00Z"], utc=True))
    frame = pd.DataFrame({"a": a, "b": b})
    assert len(frame) == 2
    assert frame.isna().sum().sum() == 2


