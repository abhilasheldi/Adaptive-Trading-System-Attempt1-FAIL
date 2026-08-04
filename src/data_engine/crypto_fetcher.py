"""Cryptocurrency OHLCV data ingestion, cleaning, and storage."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    exchange: ccxt.binance,
    since_ts: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance with pagination.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT').
        timeframe: Candle timeframe (e.g., '1h', '4h').
        exchange: Initialised ccxt binance instance.
        since_ts: Unix timestamp in milliseconds to start fetching from.
        
    Returns:
        DataFrame containing OHLCV data with UTC datetime index.
    """
    all_data: List[List] = []
    limit = 1000
    current_since = since_ts
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            current_since = last_ts + 1
            if len(ohlcv) < limit:
                break
        except ccxt.NetworkError as e:
            print(f"Network error fetching {symbol} {timeframe}: {e}. Retrying...")
            continue
        except Exception as e:
            print(f"Unexpected error fetching {symbol} {timeframe}: {e}")
            break
            
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def remove_duplicates_and_skew(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps and align clock skew.
    
    Args:
        df: Raw OHLCV DataFrame with datetime index.
        
    Returns:
        Cleaned DataFrame with unique timestamps.
    """
    # Drop exact duplicates, keeping the first occurrence to preserve raw order
    df = df[~df.index.duplicated(keep="first")]
    
    # Handle clock skew: if multiple candles fall within the same timeframe bucket,
    # merge them by taking the first open, max high, min low, last close, sum volume
    timeframe = pd.infer_freq(df.index)
    if timeframe is None:
        return df
        
    freq_map = {"H": "1h", "4H": "4h"}
    resample_freq = freq_map.get(timeframe, "1h")
    
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df = df.resample(resample_freq).agg(agg_dict).dropna()
    return df


def classify_gaps(df: pd.DataFrame, timeframe_minutes: int, tolerance_multiplier: float = 1.5) -> pd.Series:
    """Classify gaps in the time series.
    
    Args:
        df: Cleaned OHLCV DataFrame.
        timeframe_minutes: Expected candle duration in minutes.
        tolerance_multiplier: Multiplier for acceptable gap size.
        
    Returns:
        Series of gap classifications aligned with the DataFrame index.
    """
    expected_interval = pd.Timedelta(minutes=timeframe_minutes)
    tolerance = expected_interval * tolerance_multiplier
    
    gaps = df.index.to_series().diff()
    is_gap = gaps > tolerance
    
    classifications = pd.Series("normal", index=df.index, dtype="string")
    
    for i in range(1, len(df)):
        if is_gap.iloc[i]:
            gap_duration = gaps.iloc[i]
            prev_vol = df["volume"].iloc[i - 1]
            curr_vol = df["volume"].iloc[i]
            
            # Heuristic classification based on duration and volume context
            if gap_duration > pd.Timedelta(hours=48):
                classifications.iloc[i] = "exchange_outage"
            elif (prev_vol == 0 or curr_vol == 0) and gap_duration < pd.Timedelta(hours=24):
                classifications.iloc[i] = "illiquid_period"
            else:
                classifications.iloc[i] = "data_error"
                
    return classifications


def flag_outliers(df: pd.DataFrame, threshold_pct: float = 10.0) -> pd.Series:
    """Flag candles with impossible price movements.
    
    Args:
        df: Cleaned OHLCV DataFrame.
        threshold_pct: Maximum allowed percentage move or wick relative to close.
        
    Returns:
        Boolean Series indicating outlier status per candle.
    """
    # Calculate intraday range percentage
    range_pct = (df["high"] - df["low"]) / df["close"] * 100
    
    # Calculate absolute return percentage
    ret_pct = df["close"].pct_change().abs() * 100
    
    # Flag if either exceeds threshold
    outliers = (range_pct > threshold_pct) | (ret_pct > threshold_pct)
    outliers.iloc[0] = False  # First candle has no previous close for return calc
    return outliers


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
        
    combined.to_parquet(filepath, engine="pyarrow")


def ingest_crypto(config: Dict) -> None:
    """Main ingestion routine for cryptocurrency data.
    
    Args:
        config: System configuration dictionary.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    data_dir = Path(config["data_dir"])
    symbols = config["crypto"]["symbols"]
    timeframes = config["crypto"]["timeframes"]
    threshold = config["crypto"]["outlier_threshold_pct"]
    tolerance = config["crypto"]["gap_tolerance_multiplier"]
    
    for symbol in symbols:
        for tf in timeframes:
            print(f"Fetching {symbol} {tf}...")
            filepath = data_dir / f"{symbol.replace('/', '_')}_{tf}.parquet"
            
            # Determine start timestamp if file exists to avoid re-fetching history
            since_ts = None
            if filepath.exists():
                existing = pd.read_parquet(filepath)
                since_ts = int(existing.index.max().timestamp() * 1000)
                
            df = fetch_ohlcv(symbol, tf, exchange, since_ts=since_ts)
            if df.empty:
                print(f"No new data for {symbol} {tf}")
                continue
                
            # Clean and classify
            df_clean = remove_duplicates_and_skew(df)
            tf_minutes = 60 if tf == "1h" else 240
            df_clean["gap_class"] = classify_gaps(df_clean, tf_minutes, tolerance)
            df_clean["is_outlier"] = flag_outliers(df_clean, threshold)
            
            save_raw_data(df_clean, filepath)
            print(f"Saved {len(df_clean)} candles to {filepath}")
