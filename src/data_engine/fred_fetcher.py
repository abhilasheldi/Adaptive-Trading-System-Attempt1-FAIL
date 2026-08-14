"""
FRED macro ingestion -- DXY proxy, US10Y, VIX.

BUILD_PLAN_v2 B.1: "Macro: FRED (DXY, US10Y, VIX). Unblocks hypothesis
condition 3."

Uses FRED's public CSV endpoint, which needs no API key.

=========================================================================
DXY IS NOT ON FRED, AND THE SUBSTITUTE IS NOT THE SAME INDEX
=========================================================================
The ICE US Dollar Index (DXY) is a proprietary ICE product and is not
published by FRED. `DTWEXBGS` -- the Fed's Nominal Broad U.S. Dollar Index
-- is used instead, and it is a DIFFERENT INDEX, not a renamed one:

  DXY        6 currencies, ~58% EUR, fixed 1973 weights, published intraday
  DTWEXBGS   26 economies, trade-weighted, reweighted annually, DAILY only

They correlate strongly but are not interchangeable, and DTWEXBGS's heavier
emerging-market weighting behaves differently in exactly the risk-off
episodes Layer 3 is meant to detect. Every downstream use must treat this
as a proxy. Recorded here rather than in a commit message because Layer 3's
"DXY rising" rule will be evaluated against it.

`config.yaml` separately uses UUP (an ETF tracking DXY futures) via
yfinance. That is a THIRD series. Whichever is adopted, the choice must be
made once and stated, not left to whichever fetcher a caller happens to
reach for.

=========================================================================
DAILY DATA AGAINST HOURLY BARS
=========================================================================
All three series are DAILY. Joining them to 1h bars can only be a forward
fill, and a forward fill of a daily close carries a stale value for up to
23 hours. That is acceptable for a regime classifier (regimes persist) and
would not be acceptable for anything trading the macro release itself.

FRED publishes on the observation date with a release lag. This module does
NOT model that lag, so a strict point-in-time backtest would need to shift
these series by their real publication delay. Stated, not silently ignored.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# name -> (FRED series id, is_exact_match, note)
FRED_SERIES: Dict[str, tuple] = {
    "dxy_proxy": (
        "DTWEXBGS", False,
        "Nominal Broad USD Index. NOT DXY -- 26 economies vs DXY's 6. Proxy only.",
    ),
    "us10y": ("DGS10", True, "10-Year Treasury Constant Maturity Rate, percent."),
    "vix": ("VIXCLS", True, "CBOE Volatility Index, close."),
}


def fetch_fred_series(series_id: str, timeout: int = 45) -> pd.Series:
    """One FRED series as a tz-aware UTC daily Series.

    FRED encodes missing observations as '.', which `pd.to_numeric` would
    otherwise turn into a string column. Those become NaN and are DROPPED --
    a market holiday is an absent observation, not a zero.
    """
    r = requests.get(FRED_CSV, params={"id": series_id}, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    date_col = df.columns[0]
    value_col = [c for c in df.columns if c != date_col][0]
    idx = pd.to_datetime(df[date_col], utc=True)
    values = pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce")
    out = pd.Series(values.to_numpy(), index=idx, name=series_id).dropna()
    return out[~out.index.duplicated(keep="first")].sort_index()


def fetch_macro_frame(series: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """All configured FRED series on a single daily UTC index (outer join).

    Outer, not inner: the three series have different holiday calendars, and
    an inner join would silently discard every date where any one of them is
    missing. Absences stay visible as NaN.
    """
    series = series or FRED_SERIES
    cols = {}
    for name, (sid, _exact, _note) in series.items():
        cols[name] = fetch_fred_series(sid)
    return pd.DataFrame(cols).sort_index()


def classify_macro_gaps(df: pd.DataFrame, column: str) -> pd.Series:
    """Trading-day gap classification for a daily macro series.

    `crypto_fetcher.classify_gaps` is not reused here, and the reason is
    substantive rather than convenience: it assumes a CONTINUOUS market. A
    weekend in a 24/7 crypto series is a genuine anomaly; a weekend in
    Treasury yields is the market being closed. Passing daily macro through
    it would classify every Saturday as a data error and report ~2,900 false
    gaps over 30 years.

    Classification here is therefore relative to the EXCHANGE calendar:

      normal          <= 1 business day since the previous observation
      market_closed   a gap spanned only weekends/holidays (<= 4 calendar
                      days, e.g. a long weekend)
      data_gap        anything longer -- a real hole in the series
    """
    s = df[column].dropna()
    if s.empty:
        return pd.Series(dtype="string")
    deltas = s.index.to_series().diff()
    out = pd.Series("normal", index=s.index, dtype="string")
    out = out.mask((deltas > pd.Timedelta(days=1)) & (deltas <= pd.Timedelta(days=4)),
                   "market_closed")
    out = out.mask(deltas > pd.Timedelta(days=4), "data_gap")
    return out


def align_to_bars(macro: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill daily macro onto a bar index.

    Forward only. Back-filling would place a value on bars that preceded its
    observation. Bars before the first observation stay NaN -- absent, not
    zero (I5).
    """
    if macro.empty:
        return pd.DataFrame(index=bar_index, columns=macro.columns, dtype=float)
    aligned = macro.reindex(macro.index.union(bar_index)).sort_index().ffill()
    return aligned.reindex(bar_index)


def save(df: pd.DataFrame, path: Path) -> None:
    from .crypto_fetcher import save_raw_data

    save_raw_data(df, path)


__all__ = [
    "FRED_SERIES",
    "fetch_fred_series",
    "fetch_macro_frame",
    "classify_macro_gaps",
    "align_to_bars",
    "save",
]
