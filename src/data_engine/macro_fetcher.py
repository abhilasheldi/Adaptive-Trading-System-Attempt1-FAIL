"""Macro-economic and market regime data ingestion."""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf


def fetch_macro_tickers(tickers: List[str]) -> pd.DataFrame:
    """Fetch daily macro data via yfinance.
    
    Args:
        tickers: List of ticker symbols to download.
        
    Returns:
        DataFrame with daily OHLCV data for all tickers, columns multi-indexed by ticker.
    """
    # yfinance returns a DataFrame with MultiIndex columns if multiple tickers are passed
    data = yf.download(tickers, progress=False)
    
    # Flatten multi-index columns to 'Ticker_Metric' format for easier downstream handling
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [f"{ticker}_{metric}" for ticker, metric in data.columns]
        
    return data


def save_macro_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save macro data to parquet, appending only new dates.
    
    Args:
        df: Macro DataFrame with datetime index.
        filepath: Target parquet file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        # Combine and drop duplicate dates, keeping latest data to update macro feeds
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = df
        
    combined.to_parquet(filepath, engine="pyarrow")


def ingest_macro(config: Dict) -> None:
    """Main ingestion routine for macro data.
    
    Args:
        config: System configuration dictionary.
    """
    tickers = list(config["macro"]["tickers"].values())
    print(f"Fetching macro data for {tickers}...")
    
    df = fetch_macro_tickers(tickers)
    if df.empty:
        print("No macro data retrieved.")
        return
        
    filepath = Path(config["data_dir"]) / "macro_daily.parquet"
    save_macro_data(df, filepath)
    print(f"Saved {len(df)} daily records to {filepath}")
