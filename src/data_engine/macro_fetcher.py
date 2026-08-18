"""Macro-economic and market regime data ingestion."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# NOTE: ^TNX/^TYX/^FVX (CBOE Treasury yield indices) were once believed to
# need a "yield * 10" correction (e.g. 45.2 meaning 4.52%) and this module
# used to apply a 0.1 scale factor to "fix" it. That assumption was checked
# against live yfinance data and found FALSE for the data this module
# actually receives: ^TNX closed at 0.917 on 2021-01-04 (matches the real,
# well-known ~0.92% 10Y yield that day) and at 4.670 on 2026-08-06 (matches
# the real ~4.67% yield then) -- both already correctly-scaled percentages,
# with no "* 10" involved. Applying the old 0.1 factor silently divided real
# yield data by 10 (e.g. 4.670 -> 0.467), corrupting every downstream
# consumer of "us10y". No scale correction is applied for any ticker.

_COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

# ---------------------------------------------------------------------- #
# B.0.5 -- the dollar measure is DTWEXBGS, and only DTWEXBGS
# ---------------------------------------------------------------------- #
# Three dollar series were once reachable in this repo: ICE DXY (referenced
# in the source document), FRED DTWEXBGS (ingested at B.1), and UUP (an ETF
# tracking DXY futures, requested through this module via config.yaml).
# Deleting the config entry alone would leave the ROUTE open -- the next
# person who wants a dollar column adds a ticker and gets a fourth answer.
#
# So the route is closed here as well. This is a guard against reintroduction,
# not a validation of user input: the failure it prevents is silent
# disagreement between two dollar series, which surfaces as an inexplicable
# Layer 3 result months later rather than as an error.
_FORBIDDEN_DOLLAR_TICKERS = {
    "UUP", "UDN", "DX-Y.NYB", "DXY", "DX=F", "USDX", "^DXY",
}
_DOLLAR_KEY_HINTS = ("dxy", "dollar", "usd_index", "usdindex")


# B.5 STEP 3 -- QQQ is single-sourced, same ruling as DTWEXBGS. It is
# ingested by `scripts/ingest_qqq.py` into `data/raw/QQQ_daily.parquet` and
# feeds condition 3's BTC/QQQ correlation leg. A second route would give a
# second correlation series, and Layer 3 would then depend on whichever one
# a caller reached for -- the exact failure the dollar ruling closed.
_QQQ_CANONICAL_SOURCE = "scripts/ingest_qqq.py -> data/raw/QQQ_daily.parquet"


def _reject_duplicate_qqq_route(tickers: Dict[str, str]) -> None:
    """Hard-fail if QQQ is requested through this module (B.5 STEP 3)."""
    for name, symbol in tickers.items():
        if str(symbol).upper().strip() == "QQQ" or "qqq" in str(name).lower():
            raise ValueError(
                f"QQQ requested as {name!r}={symbol!r} through macro_fetcher. QQQ is "
                f"single-sourced at {_QQQ_CANONICAL_SOURCE} and feeds condition 3's "
                f"BTC/QQQ correlation leg (pre-registered at "
                f"prereg/condition_3_macro.md). Two routes would give two correlation "
                f"series and Layer 3 would depend on whichever a caller reached for -- "
                f"the same defect the DTWEXBGS ruling closed. If this needs to change, "
                f"change the registration first."
            )


def _reject_dollar_tickers(tickers: Dict[str, str]) -> None:
    """Hard-fail if a dollar index is requested through yfinance (B.0.5)."""
    for name, symbol in tickers.items():
        sym = str(symbol).upper().strip()
        key = str(name).lower()
        if sym in _FORBIDDEN_DOLLAR_TICKERS or any(h in key for h in _DOLLAR_KEY_HINTS):
            raise ValueError(
                f"Dollar-index ticker {symbol!r} requested as {name!r}. The dollar "
                f"measure is FRED DTWEXBGS via data_engine.fred_fetcher, decided at "
                f"B.0.5 and recorded as S19. UUP is an ETF tracking DXY futures -- a "
                f"third series with its own tracking error and US-hours calendar, and "
                f"having two reachable dollar columns is how Layer 3 ends up evaluated "
                f"against whichever one a caller happened to reach for. If this needs "
                f"to change, change the ruling first."
            )


def fetch_macro_tickers(
    tickers: Dict[str, str],
    start: Optional[str] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily macro data via yfinance.

    Args:
        tickers: Mapping of semantic name -> yfinance ticker symbol, e.g.
            {"dxy": "DX-Y.NYB", "us10y": "^TNX", "vix": "^VIX", "spx": "^GSPC",
            "qqq": "QQQ"}. Semantic names (not raw ticker symbols) are preserved
            in the output columns so downstream code doesn't need to memorize
            which ticker maps to which macro series.
        start: ISO date string to fetch from. If None, fetches the maximum
            history yfinance offers (period="max") -- yf.download's own default
            is only the trailing month when no start/period is given, which
            silently truncates history on every call unless this is set.
        max_retries: Retries with backoff on transient fetch failures.

    Returns:
        DataFrame with a UTC daily datetime index and columns named
        "{semantic_name}_{Metric}" (e.g. "vix_Close", "us10y_Close").
    """
    _reject_dollar_tickers(tickers)      # B.0.5 -- DTWEXBGS is the dollar measure
    _reject_duplicate_qqq_route(tickers)  # B.5 STEP 3 -- QQQ is single-sourced

    symbol_list = list(tickers.values())
    kwargs = {"start": start} if start else {"period": "max"}

    data = None
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(symbol_list, progress=False, group_by="column", **kwargs)
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error("Failed to fetch macro tickers after %d attempts: %s", max_retries, e)
                return pd.DataFrame()
            backoff = 2 ** attempt
            logger.warning("Macro fetch attempt %d/%d failed (%s), retrying in %ds", attempt, max_retries, e, backoff)
            time.sleep(backoff)

    if data is None or data.empty:
        return pd.DataFrame()

    ticker_to_name = {v: k for k, v in tickers.items()}

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance's default group_by="column" MultiIndex is (Metric, Ticker),
        # not (Ticker, Metric) -- flatten in the correct order and substitute
        # the semantic name for the raw ticker symbol.
        new_cols = []
        for metric, ticker in data.columns:
            name = ticker_to_name.get(ticker, ticker)
            new_cols.append(f"{name}_{metric}")
        data.columns = new_cols
    else:
        # Single ticker: columns are just the metric names.
        only_ticker = symbol_list[0]
        name = ticker_to_name.get(only_ticker, only_ticker)
        data.columns = [f"{name}_{metric}" for metric in data.columns]

    # Warn (don't fail) if a requested ticker came back entirely empty --
    # easy to miss otherwise since yfinance doesn't raise for a bad symbol
    # mixed in with valid ones.
    for name, ticker in tickers.items():
        cols = [c for c in data.columns if c.startswith(f"{name}_")]
        if cols and data[cols].isna().all().all():
            logger.warning("Ticker '%s' (%s) returned no data at all -- check the symbol.", name, ticker)

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    return data


def fetch_btc_dominance(max_retries: int = 3) -> pd.DataFrame:
    """Fetch current BTC market-cap dominance from CoinGecko's free public API.

    Neither yfinance nor FRED tracks crypto market dominance, so this uses a
    separate free, no-key-required source. NOTE: CoinGecko's free /global
    endpoint only exposes the CURRENT dominance snapshot, not a history --
    build a daily time series by calling and appending this once per day
    (e.g. as a scheduled job), rather than expecting a bulk historical pull.

    Returns:
        Single-row DataFrame indexed by the current UTC timestamp with a
        'btc_dominance_pct' column, or an empty DataFrame on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(_COINGECKO_GLOBAL_URL, timeout=10)
            resp.raise_for_status()
            pct = resp.json()["data"]["market_cap_percentage"]["btc"]
            now = pd.Timestamp.now(tz="UTC").normalize()
            return pd.DataFrame({"btc_dominance_pct": [pct]}, index=[now])
        except Exception as e:
            if attempt == max_retries:
                logger.error("Failed to fetch BTC dominance after %d attempts: %s", max_retries, e)
                return pd.DataFrame()
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def save_macro_data(df: pd.DataFrame, filepath: Path) -> None:
    """Save macro data to parquet, merging with existing data.

    Args:
        df: Macro DataFrame with datetime index.
        filepath: Target parquet file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        existing = pd.read_parquet(filepath)
        # Keep 'last' (newly fetched): unlike raw crypto candles, macro series
        # legitimately get revised (adjusted closes, corrected reference data),
        # so preferring the freshest fetch is correct here.
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    else:
        combined = df

    # Atomic write so a crash mid-write can't corrupt the file.
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    combined.to_parquet(tmp_path, engine="pyarrow")
    os.replace(tmp_path, filepath)


def ingest_macro(config: Dict) -> None:
    """Main ingestion routine for macro data.

    Args:
        config: System configuration dictionary. `config["macro"]["tickers"]`
            should map semantic names to yfinance symbols, e.g.
            {"dxy": "DX-Y.NYB", "us10y": "^TNX", "vix": "^VIX",
            "spx": "^GSPC", "qqq": "QQQ"}. A "btc_dominance" key, if present,
            is routed to CoinGecko instead of yfinance since neither
            yfinance nor FRED offers that series.
    """
    logging.basicConfig(level=logging.INFO)
    tickers = dict(config["macro"]["tickers"])
    btc_dom_requested = tickers.pop("btc_dominance", None) is not None

    filepath = Path(config["data_dir"]) / "macro_daily.parquet"

    start = None
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        if len(existing):
            start = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Fetching macro data for %s...", list(tickers.values()))
    df = fetch_macro_tickers(tickers, start=start) if tickers else pd.DataFrame()

    if btc_dom_requested:
        dom_df = fetch_btc_dominance()
        if not dom_df.empty:
            df = dom_df if df.empty else df.combine_first(dom_df)

    if df.empty:
        logger.info("No macro data retrieved.")
        return

    save_macro_data(df, filepath)
    logger.info("Saved %d daily records to %s", len(df), filepath)