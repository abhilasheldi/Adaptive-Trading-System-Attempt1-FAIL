"""Manual/live smoke tests against real Binance data via ccxt.

These hit the real network and are not meant to run in CI on every commit --
run them ad hoc to sanity-check the pipeline against live data. Use:

    pytest tests/test_live_binance.py -v -s

The -s flag is needed to see the print() output.
"""

import ccxt
import pandas as pd

from src.data_engine.crypto_fetcher import (
    fetch_ohlcv,
    remove_duplicates_and_skew,
    classify_gaps,
    flag_outliers,
)


def test_fetch_real_binance_data():
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    df = fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        exchange=exchange,
    )

    print("\nDATA:")
    print(df.head())

    print("\nTAIL:")
    print(df.tail())

    print("\nINFO:")
    print(df.info())

    print("\nSHAPE:")
    print(df.shape)

    # since_ts defaults to the exchange epoch now (not None), specifically so
    # a first-time fetch pulls FULL history rather than just the last ~1000
    # candles -- assert that actually happened rather than just eyeballing it.
    span = df.index.max() - df.index.min()
    print("\nDATE SPAN:", span)
    assert span > pd.Timedelta(days=365), (
        f"Expected multiple years of 1h history, got only {span}. "
        "If this fails, since_ts is probably not reaching back far enough."
    )

    # No duplicate timestamps, strictly increasing index.
    assert df.index.is_unique
    assert df.index.is_monotonic_increasing

    # The most recent candle should not be "now" -- fetch_ohlcv trims a
    # still-forming trailing candle, so the last close should be at least
    # one full interval in the past.
    now = pd.Timestamp.now(tz="UTC")
    assert (now - df.index.max()) >= pd.Timedelta(hours=1)


def test_real_pipeline():
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    timeframe = "1h"

    df = fetch_ohlcv(
        "BTC/USDT",
        timeframe,
        exchange,
    )

    print("RAW:")
    print(df.head())
    print("RAW shape:", df.shape)

    # remove_duplicates_and_skew now requires the configured timeframe
    # explicitly -- it no longer tries to infer it from the data (that
    # inference was silently broken by a pandas frequency-alias case
    # mismatch and would otherwise resample everything as if it were 1h).
    df_clean = remove_duplicates_and_skew(df, timeframe)

    print("\nCLEAN:")
    print(df_clean.head())
    print("CLEAN shape:", df_clean.shape)

    gaps = classify_gaps(
        df_clean,
        timeframe_minutes=60,
    )

    print("\nGAPS:")
    print(gaps.value_counts())

    outliers = flag_outliers(
        df_clean,
        threshold_pct=10,
    )

    print("\nOUTLIERS:")
    print(outliers.value_counts())

    # A handful of sanity checks on real data, since "print and eyeball" alone
    # won't catch a regression on the next run.
    assert len(df_clean) > 0
    assert set(gaps.unique()).issubset(
        {"normal", "exchange_outage", "illiquid_period", "data_error"}
    )
    # On real BTC/USDT 1h data over a long history, the overwhelming majority
    # of candles should be "normal" -- if gaps dominate, something upstream
    # (cleaning, timeframe mismatch) is probably wrong rather than Binance
    # actually being down most of the time.
    normal_frac = (gaps == "normal").mean()
    print(f"\nnormal_frac: {normal_frac:.4f}")
    assert normal_frac > 0.95

    # Real BTC/USDT candles occasionally do have >10% wicks (flash crashes,
    # liquidation cascades), so don't assert zero outliers -- just that they
    # aren't the majority of the dataset, which would suggest the threshold
    # or the OHLC integrity checks are misfiring.
    outlier_frac = outliers.mean()
    print(f"outlier_frac: {outlier_frac:.4f}")
    assert outlier_frac < 0.05


if __name__ == "__main__":
    test_fetch_real_binance_data()
    test_real_pipeline()