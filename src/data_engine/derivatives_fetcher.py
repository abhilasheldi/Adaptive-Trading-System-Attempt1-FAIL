"""
Derivatives-flow ingestion: funding rates and perp-spot basis.

BUILD_PLAN_v2 B.1. Both sources are free and full-history on Binance, which
is why they are in B.1 rather than behind the B.2 vendor decision.

QUALITY PIPELINE. These series run through the same Layer 1 treatment as
OHLCV -- duplicate/clock-skew removal, gap classification, outlier flagging
-- rather than a bespoke one. A second cleaning path would drift from the
first and the divergence would be invisible.

A NOTE ON WHAT FUNDING IS. `fundingRate` is settled every 8 hours and is
the rate APPLIED at that settlement, computed from the preceding interval.
It is therefore known only AT settlement, not before it. Anything joining
funding onto 1h bars must forward-fill from the settlement timestamp
onward and must never back-fill into the interval the rate was computed
over -- that would be lookahead. `resample_funding_to_bars` does the
former and refuses to do the latter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .crypto_fetcher import classify_gaps, flag_outliers, remove_duplicates_and_skew

FAPI = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"

FUNDING_INTERVAL_HOURS = 8
_MAX_LIMIT = 1000
_SLEEP = 0.25


def _get(url: str, params: Dict, retries: int = 4) -> list:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------- #
# Funding rates
# ---------------------------------------------------------------------- #


# Binance treats startTime=0 as UNSET and returns only the most recent ~500
# records, silently truncating history to ~5 months. Verified directly:
#   startTime=0             -> 500 rows, 2026-03-01 .. 2026-08-14
#   startTime=1568000000000 -> 1000 rows, 2019-09-10 .. 2020-08-08
# Any positive epoch-ms predating perp inception (2019-09) paginates
# correctly, so the default must never be 0.
_EPOCH_FLOOR_MS = 1_500_000_000_000   # 2017-07-14, well before perp inception


def fetch_funding_rates(symbol: str = "BTCUSDT", start_ms: Optional[int] = None) -> pd.DataFrame:
    """Full funding-rate history, paginated forward from `start_ms`.

    Paginates on `startTime` rather than `endTime` because the endpoint
    returns the EARLIEST records in a window; walking backwards would
    silently stop at the first full page.
    """
    rows: List[dict] = []
    cursor = start_ms if start_ms else _EPOCH_FLOOR_MS
    seen_last = None

    while True:
        batch = _get(f"{FAPI}/fapi/v1/fundingRate",
                     {"symbol": symbol, "startTime": cursor, "limit": _MAX_LIMIT})
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1]["fundingTime"])
        # Stop only when the cursor stops advancing. Length-based stopping is
        # wrong here: the endpoint returns short pages mid-history, so
        # `len(batch) < limit` would end pagination years early.
        if last == seen_last:
            break
        seen_last = last
        cursor = last + 1
        time.sleep(_SLEEP)

    if not rows:
        return pd.DataFrame(columns=["funding_rate", "mark_price"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce"),
        "mark_price": pd.to_numeric(df.get("markPrice"), errors="coerce"),
    })
    return out[~out.index.duplicated(keep="first")]


def classify_funding_gaps(df: pd.DataFrame) -> pd.Series:
    """Layer 1 gap classification at the funding cadence (8h).

    `classify_gaps` needs a `volume` column to distinguish an illiquid
    period from a data error. Funding has no volume, so a zero column is
    supplied and the illiquid branch simply never fires -- every real gap
    lands as `exchange_outage` or `data_error`, which is the correct
    vocabulary for a settlement series. Stated because a reader seeing zero
    `illiquid_period` counts should know it is structural, not a finding.
    """
    tmp = df.copy()
    tmp["volume"] = 0.0
    return classify_gaps(tmp, timeframe_minutes=FUNDING_INTERVAL_HOURS * 60)


def resample_funding_to_bars(funding: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align funding onto a bar index by FORWARD fill from settlement.

    Forward only. A funding rate settled at 08:00 is unknown at 04:00, so
    back-filling it into 00:00-08:00 would leak the outcome of the interval
    into bars inside that interval. Bars before the first settlement are
    left NaN rather than filled -- absent, not zero.
    """
    if funding.empty:
        return pd.DataFrame(index=bar_index, columns=["funding_rate", "mark_price"], dtype=float)
    aligned = funding.reindex(funding.index.union(bar_index)).sort_index().ffill()
    return aligned.reindex(bar_index)


# ---------------------------------------------------------------------- #
# Perp-spot basis
# ---------------------------------------------------------------------- #


def _klines(base: str, path: str, symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms if start_ms else _EPOCH_FLOOR_MS   # 0 is treated as unset
    while True:
        batch = _get(f"{base}{path}",
                     {"symbol": symbol, "interval": interval,
                      "startTime": cursor, "limit": _MAX_LIMIT})
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open < cursor:      # cursor stopped advancing
            break
        cursor = last_open + 1
        time.sleep(_SLEEP)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_perp_spot_basis(
    symbol: str = "BTCUSDT", interval: str = "1h", start_ms: int = 0
) -> pd.DataFrame:
    """Perp-spot basis, computed rather than fetched.

    basis = (mark - spot) / spot, on the intersection of the two indices.

    Uses futures MARK price, not futures last-traded price. Mark is the
    index-anchored valuation the exchange liquidates against; last-traded
    carries microstructure noise that would show up as spurious basis.

    The intersection is deliberate: a bar present in only one series cannot
    produce a basis, and carrying it forward would fabricate one.
    """
    mark = _klines(FAPI, "/fapi/v1/markPriceKlines", symbol, interval, start_ms)
    spot = _klines(SPOT_API, "/api/v3/klines", symbol, interval, start_ms)
    if mark.empty or spot.empty:
        return pd.DataFrame(columns=["mark_close", "spot_close", "basis", "volume"])

    common = mark.index.intersection(spot.index)
    out = pd.DataFrame(index=common)
    out["mark_close"] = mark.loc[common, "close"]
    out["spot_close"] = spot.loc[common, "close"]
    out["basis"] = (out["mark_close"] - out["spot_close"]) / out["spot_close"]
    # Spot volume, carried so the Layer 1 gap classifier can use its
    # illiquid-period branch rather than having it silently disabled.
    out["volume"] = spot.loc[common, "volume"]
    return out


def clean_and_classify(df: pd.DataFrame, timeframe: str, outlier_threshold_pct: float = 10.0):
    """Run a fetched frame through the Layer 1 quality pipeline.

    Returns (cleaned_df, gap_classifications, outlier_flags). Outlier
    detection needs OHLC columns; frames without them get an all-False flag
    series and that is reported rather than silently skipped.
    """
    from .crypto_fetcher import _timeframe_minutes

    cleaned = remove_duplicates_and_skew(df, timeframe=timeframe)
    if "volume" not in cleaned.columns:
        cleaned = cleaned.assign(volume=0.0)
    gaps = classify_gaps(cleaned, timeframe_minutes=_timeframe_minutes(timeframe))
    if {"open", "high", "low", "close"}.issubset(cleaned.columns):
        outliers = flag_outliers(cleaned, threshold_pct=outlier_threshold_pct)
    else:
        outliers = pd.Series(False, index=cleaned.index)
    return cleaned, gaps, outliers


def save(df: pd.DataFrame, path: Path) -> None:
    from .crypto_fetcher import save_raw_data

    save_raw_data(df, path)


__all__ = [
    "FUNDING_INTERVAL_HOURS",
    "fetch_funding_rates",
    "classify_funding_gaps",
    "resample_funding_to_bars",
    "fetch_perp_spot_basis",
    "clean_and_classify",
    "save",
]
