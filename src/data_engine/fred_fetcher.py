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

**The choice is settled (B.0.5, 2026-08-15): DTWEXBGS.** The UUP path -- an
ETF tracking DXY futures, formerly reachable via `macro_fetcher` and
`config.yaml` -- is DELETED, not deprecated, so there is no longer a third
dollar series a caller can reach for by accident. `dollar_substitution_note()`
returns the sentence that must appear wherever a Layer 3 result is reported:
DTWEXBGS is a different index, not a DXY proxy, and that belongs in the
report itself rather than only in this docstring.

=========================================================================
DAILY DATA AGAINST HOURLY BARS, AND PUBLICATION LAG
=========================================================================
Every series here is DAILY. Joining them to 1h bars can only be a forward
fill, and a forward fill of a daily close carries a stale value for up to
23 hours. That is acceptable for a regime classifier (regimes persist) and
would not be acceptable for anything trading the macro release itself.

**Publication lag is now modelled (B.0.4).** FRED stamps an observation
with the date it DESCRIBES, not the date it was released. Treating the two
as equal hands the backtest a value hours-to-days before anyone could have
seen it.

This stopped being a reporting caveat and became blocking when the macro
condition was restored to H1 (MASTER_CONTEXT section 4, four conditions).
Macro now gates EVENT FORMATION, not just a feature column: an event set
built on unlagged macro is contaminated at its definition, and every
measurement downstream inherits it. A contaminated feature stays in one
arm; a contaminated event set is in all of them.

`align_to_bars` therefore routes through `point_in_time` with the
`FRED_DAILY` policy -- minimum one BUSINESS day, per B.0.4. Business, not
calendar: a calendar shift would place Friday's observation on Saturday, a
day no statistical agency publishes on, which understates the delay exactly
when the weekend gap makes it matter most.

One business day is a floor, not the truth. Real release delays vary by
series and vintage. `fetch_alfred_vintage` is the hook for replacing the
floor with actual vintages; it needs a FRED API key, which this environment
does not have.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from .point_in_time import FRED_DAILY, ObservationPolicy, align_point_in_time

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# name -> (FRED series id, is_exact_match, note)
FRED_SERIES: Dict[str, tuple] = {
    "dxy_proxy": (
        "DTWEXBGS", False,
        "Nominal Broad USD Index. NOT DXY -- 26 economies vs DXY's 6. Proxy only.",
    ),
    "us10y": ("DGS10", True, "10-Year Treasury Constant Maturity Rate, percent."),
    "vix": ("VIXCLS", True, "CBOE Volatility Index, close."),
    # B.0.6 -- base currency. USD per ONE GBP, so USD -> GBP is a DIVISION.
    # Ingested here rather than in a separate FX module because it is a FRED
    # daily series with exactly the same publication-lag and weekend
    # semantics as the other three, and splitting it out would mean
    # maintaining that logic twice.
    "gbpusd": (
        "DEXUSUK",
        True,
        "U.S. Dollars to One British Pound, daily noon buying rate. "
        "Base-currency denominator (B.0.6): GBP = USD / gbpusd.",
    ),
}

# Series whose absence breaks accounting rather than a feature. Ingest
# failure here is fatal, not degraded -- see `fetch_macro_frame`.
REQUIRED_SERIES = ("gbpusd",)


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


def align_to_bars(
    macro: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    policy: ObservationPolicy = FRED_DAILY,
) -> pd.DataFrame:
    """Forward-fill daily macro onto a bar index, at OBSERVATION time (I9).

    Changed at B.0.4. This previously aligned on the value date, which
    treated a FRED observation as knowable on the day it describes. It now
    shifts by the policy's publication lag first -- one business day by
    default -- so a Friday observation reaches Monday's bars, not Friday's.

    `policy` is overridable but not optional-by-omission: the default is an
    explicit, named policy carrying its own justification, not a bare zero.
    """
    if macro.empty:
        return pd.DataFrame(index=bar_index, columns=macro.columns, dtype=float)
    return align_point_in_time(macro, bar_index, policy)


def fetch_alfred_vintage(series_id: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Real-time vintages from ALFRED -- the exact fix B.0.4 asks for.

    NOT IMPLEMENTED, and deliberately raising rather than silently falling
    back to the business-day floor. A silent fallback would make a
    vintage-accurate run indistinguishable from an approximate one in the
    output, which is the class of ambiguity this project spends its budget
    avoiding.

    ALFRED needs a FRED API key (`FRED_API_KEY`); this environment has none,
    which is why `FRED_DAILY`'s one-business-day floor is what actually runs.
    """
    raise NotImplementedError(
        f"ALFRED vintages for {series_id!r} need a FRED API key (FRED_API_KEY), "
        f"which is not configured. B.0.4 is satisfied in the meantime by the "
        f"one-business-day floor in point_in_time.FRED_DAILY -- conservative in "
        f"the safe direction. Wire this up before any result depends on the "
        f"exact release date rather than on there being a lag at all."
    )


def dollar_substitution_note() -> str:
    """The sentence that must appear wherever a Layer 3 result is reported.

    B.0.5 requires this phrasing in the REPORTS, not only in the data
    dictionary. Returned as a string so report writers cannot paraphrase it
    into something weaker -- "DXY proxy" is precisely the wording that would
    let a reader assume interchangeability.
    """
    return (
        "Dollar measure: FRED DTWEXBGS (Nominal Broad U.S. Dollar Index). "
        "This is a DIFFERENT INDEX from ICE DXY, not a proxy for it: 26 "
        "trade-weighted economies reweighted annually, versus DXY's 6 "
        "currencies on fixed 1973 weights (~58% EUR). DTWEXBGS's heavier "
        "emerging-market weighting behaves differently in exactly the "
        "risk-off episodes Layer 3 exists to detect. Any 'DXY rising' rule "
        "here is evaluated against a different instrument (S19)."
    )


def save(df: pd.DataFrame, path: Path) -> None:
    from .crypto_fetcher import save_raw_data

    save_raw_data(df, path)


__all__ = [
    "FRED_SERIES",
    "REQUIRED_SERIES",
    "fetch_fred_series",
    "fetch_macro_frame",
    "fetch_alfred_vintage",
    "classify_macro_gaps",
    "align_to_bars",
    "dollar_substitution_note",
    "save",
]
