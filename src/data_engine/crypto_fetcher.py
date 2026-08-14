"""Cryptocurrency OHLCV data ingestion, cleaning, and storage."""

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# Valid pandas resample offset aliases we support, keyed by ccxt timeframe string.
_TIMEFRAME_TO_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    # 'w' added at Stage B.1: hypothesis condition 4 requires 1H/4H/1D/1W,
    # and both the table and the fallback parser previously stopped at 'd',
    # so the weekly timeframe raised rather than resolving.
    "1d": 1440, "3d": 4320, "1w": 10080,
}


def _pandas_resample_alias(timeframe: str) -> str:
    """Translate a ccxt timeframe into a pandas offset alias.

    ccxt uses lowercase throughout ('1h', '1d', '1w'). pandas accepts
    lowercase for minutes/hours/days but DEPRECATED lowercase 'w' -- it
    still resamples correctly today while emitting a FutureWarning, and will
    stop working in a later pandas. Translating here keeps ccxt's vocabulary
    at the call sites and confines the pandas-version detail to one place.
    """
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe)
    if match and match.group(2) == "w":
        return f"{match.group(1)}W"
    return timeframe


def _timeframe_minutes(timeframe: str) -> int:
    """Resolve a ccxt-style timeframe string (e.g. '1h', '4h', '15m') to minutes."""
    if timeframe in _TIMEFRAME_TO_MINUTES:
        return _TIMEFRAME_TO_MINUTES[timeframe]
    # Fallback parser for anything not in the table, e.g. '2d', '2w'.
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe)
    if not match:
        raise ValueError(f"Cannot parse timeframe minutes from '{timeframe}'")
    qty, unit = int(match.group(1)), match.group(2)
    return qty * {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit]


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    exchange: ccxt.binance,
    since_ts: Optional[int] = None,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance with pagination, covering full available history.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT').
        timeframe: Candle timeframe (e.g., '1h', '4h').
        exchange: Initialised ccxt binance instance.
        since_ts: Unix timestamp in milliseconds to start fetching from. If None,
            fetching starts from the exchange epoch (0) so that the FULL available
            history is retrieved, not just the most recent `limit` candles -- ccxt's
            `since=None` returns only the latest candles, which silently truncates
            history on a first-time fetch.
        max_retries: Max consecutive network-error retries before giving up, with
            exponential backoff, to avoid hammering the API on persistent outages.

    Returns:
        DataFrame containing OHLCV data with a UTC datetime index. The final,
        possibly still-forming candle is dropped so we never persist an incomplete bar.
    """
    if not exchange.markets or symbol not in exchange.markets:
        exchange.load_markets()
    if not exchange.markets or symbol not in exchange.markets:
        raise ValueError(
            f"'{symbol}' is not a valid market on {exchange.id}. "
            f"Binance spot has no BTC/USD pair, for example -- check for BTC/USDT etc."
        )

    all_data: List[List] = []
    limit = 1000
    current_since = since_ts if since_ts is not None else 0
    retries = 0

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            retries = 0
        except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e:
            retries += 1
            if retries > max_retries:
                logger.error(
                    "Giving up on %s %s after %d consecutive retries: %s",
                    symbol, timeframe, max_retries, e,
                )
                break
            backoff = min(2 ** retries, 60)
            logger.warning(
                "Network error fetching %s %s (attempt %d/%d): %s. Backing off %ds.",
                symbol, timeframe, retries, max_retries, e, backoff,
            )
            time.sleep(backoff)
            continue
        except Exception as e:
            logger.error("Unexpected error fetching %s %s: %s", symbol, timeframe, e)
            break

        if not ohlcv:
            break
        all_data.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        current_since = last_ts + 1
        if len(ohlcv) < limit:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]  # last fetch of a given ts wins pre-cleaning

    # Drop the trailing candle if it may still be forming (its bucket hasn't closed
    # yet as of "now"), so we never write a value that could later change.
    if len(df) > 0:
        interval = pd.Timedelta(minutes=_timeframe_minutes(timeframe))
        now = datetime.now(timezone.utc)
        if df.index[-1] + interval > now:
            df = df.iloc[:-1]

    return df


def remove_duplicates_and_skew(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Remove duplicate timestamps and align clock skew.

    Args:
        df: Raw OHLCV DataFrame with datetime index.
        timeframe: The candle's INTENDED timeframe (e.g. '1h', '4h'), as configured
            by the caller. This must be passed explicitly rather than inferred from
            the data itself -- `pd.infer_freq` requires an already-regular index and
            returns None on the very duplicate/skewed data this function exists to
            clean, silently skipping the merge step it's supposed to perform.

    Returns:
        Cleaned DataFrame with unique timestamps, resampled onto the exact
        configured timeframe grid.
    """
    df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        return df

    # Merge any candles that fall within the same timeframe bucket (clock skew)
    # by taking the first open, max high, min low, last close, sum volume.
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df = df.resample(_pandas_resample_alias(timeframe)).agg(agg_dict).dropna()
    return df


def classify_gaps(
    df: pd.DataFrame,
    timeframe_minutes: int,
    tolerance_multiplier: float = 1.5,
    outage_multiplier: float = 48.0,
    illiquid_multiplier: float = 24.0,
) -> pd.Series:
    """Classify gaps in the time series.

    Args:
        df: Cleaned OHLCV DataFrame.
        timeframe_minutes: Expected candle duration in minutes.
        tolerance_multiplier: Multiplier of the expected interval defining "is a gap".
        outage_multiplier: Gaps beyond (expected_interval * outage_multiplier) are
            classified as exchange outages. Expressed relative to the timeframe so
            the same logic behaves sensibly whether applied to 1H or 4H candles,
            rather than a fixed absolute hour count.
        illiquid_multiplier: Threshold (relative to expected_interval) below which
            a zero-volume-adjacent gap is treated as an illiquid period rather than
            a data error.

    Returns:
        Series of gap classifications aligned with the DataFrame index.
    """
    expected_interval = pd.Timedelta(minutes=timeframe_minutes)
    tolerance = expected_interval * tolerance_multiplier
    outage_threshold = expected_interval * outage_multiplier
    illiquid_threshold = expected_interval * illiquid_multiplier

    gaps = df.index.to_series().diff()
    is_gap = gaps > tolerance

    classifications = pd.Series("normal", index=df.index, dtype="string")

    prev_vol = df["volume"].shift(1)
    curr_vol = df["volume"]
    zero_vol_adjacent = (prev_vol == 0) | (curr_vol == 0)

    is_outage = is_gap & (gaps > outage_threshold)
    is_illiquid = is_gap & ~is_outage & zero_vol_adjacent & (gaps < illiquid_threshold)
    is_data_error = is_gap & ~is_outage & ~is_illiquid

    classifications = classifications.mask(is_outage, "exchange_outage")
    classifications = classifications.mask(is_illiquid, "illiquid_period")
    classifications = classifications.mask(is_data_error, "data_error")

    return classifications


def flag_outliers(df: pd.DataFrame, threshold_pct: float = 10.0) -> pd.Series:
    """Flag candles with impossible price movements or internally inconsistent OHLC.

    Args:
        df: Cleaned OHLCV DataFrame.
        threshold_pct: Maximum allowed percentage move or wick relative to close.

    Returns:
        Boolean Series indicating outlier status per candle.
    """
    range_pct = (df["high"] - df["low"]) / df["close"] * 100
    ret_pct = df["close"].pct_change().abs() * 100

    # ret_pct is NaN on row 0 (no prior close), and NaN comparisons already
    # evaluate False, so no explicit override is needed there. A blanket
    # `outliers.iloc[0] = False` would also wrongly suppress a genuine
    # range-based outlier on the very first candle -- don't do that.
    magnitude_outliers = (range_pct > threshold_pct) | (ret_pct > threshold_pct)

    # Structural OHLC integrity violations are "impossible" regardless of magnitude.
    integrity_violations = (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )

    return magnitude_outliers | integrity_violations


def save_raw_data(df: pd.DataFrame, filepath: Path) -> None:
    """Append new data to raw parquet file without overwriting existing rows.

    Args:
        df: New OHLCV DataFrame to store.
        filepath: Target parquet file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        existing = pd.read_parquet(filepath)
        # Align indices and drop any overlap to prevent mutation of raw data.
        # We keep 'first' (existing) to guarantee immutability of previously written data.
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
    else:
        combined = df

    # Write atomically: to a temp file, then rename, so a crash mid-write can
    # never corrupt or truncate previously-written (supposedly immutable) data.
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    combined.to_parquet(tmp_path, engine="pyarrow")
    os.replace(tmp_path, filepath)


def ingest_crypto(config: Dict) -> None:
    """Main ingestion routine for cryptocurrency data.

    Args:
        config: System configuration dictionary.
    """
    logging.basicConfig(level=logging.INFO)
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()

    data_dir = Path(config["data_dir"])
    symbols = config["crypto"]["symbols"]
    timeframes = config["crypto"]["timeframes"]
    threshold = config["crypto"]["outlier_threshold_pct"]
    tolerance = config["crypto"]["gap_tolerance_multiplier"]

    for symbol in symbols:
        for tf in timeframes:
            try:
                _ingest_one(exchange, data_dir, symbol, tf, threshold, tolerance)
            except Exception:
                # One bad symbol/timeframe should never take down the whole run.
                logger.exception("Failed to ingest %s %s, continuing", symbol, tf)


def _ingest_one(
    exchange: ccxt.binance,
    data_dir: Path,
    symbol: str,
    tf: str,
    threshold: float,
    tolerance: float,
) -> None:
    logger.info("Fetching %s %s...", symbol, tf)
    filepath = data_dir / f"{symbol.replace('/', '_')}_{tf}.parquet"

    since_ts = None
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        if len(existing):
            since_ts = int(existing.index.max().timestamp() * 1000) + 1

    df = fetch_ohlcv(symbol, tf, exchange, since_ts=since_ts)
    if df.empty:
        logger.info("No new data for %s %s", symbol, tf)
        return

    df_clean = remove_duplicates_and_skew(df, tf)
    tf_minutes = _timeframe_minutes(tf)
    df_clean["gap_class"] = classify_gaps(df_clean, tf_minutes, tolerance)
    df_clean["is_outlier"] = flag_outliers(df_clean, threshold)

    save_raw_data(df_clean, filepath)
    logger.info("Saved %d candles to %s", len(df_clean), filepath)