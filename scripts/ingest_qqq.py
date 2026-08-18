"""
B.5 STEP 3 -- ingest QQQ daily for the BTC/QQQ correlation leg of condition 3.

Free via yfinance. Verified under I10, aligned under I9, and single-sourced:
`macro_fetcher._reject_duplicate_qqq_route` hard-fails if QQQ becomes
reachable from a second provider, per the DTWEXBGS ruling.

    python scripts/ingest_qqq.py
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.acquisition import (  # noqa: E402
    AcquisitionRequest,
    assert_acquisition,
)
from src.data_engine.crypto_fetcher import save_raw_data  # noqa: E402
from src.data_engine.point_in_time import (  # noqa: E402
    MARKET_DAILY_CLOSE,
    align_point_in_time,
    assert_no_lookahead,
)

OUT = Path("data/raw/QQQ_daily.parquet")
TICKER = "QQQ"

# QQQ launched 1999-03-10. Requesting from 2005 comfortably covers the BTC
# era with margin; the floor exists so an ignored start bound is visible.
REQUESTED_START = pd.Timestamp("2005-01-03", tz="UTC")
# ~252 trading days/yr over 20+ years. Set well below the true count -- a
# floor that catches an S21-class truncation, not an equality check.
EXPECTED_MIN_ROWS = 4_000


def main() -> int:
    print(f"fetching {TICKER} daily via yfinance (period=max)...")
    raw = yf.download(TICKER, period="max", progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise SystemExit("!! yfinance returned nothing for QQQ.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = pd.DataFrame({
        "open": pd.to_numeric(raw["Open"], errors="coerce"),
        "high": pd.to_numeric(raw["High"], errors="coerce"),
        "low": pd.to_numeric(raw["Low"], errors="coerce"),
        "close": pd.to_numeric(raw["Close"], errors="coerce"),
        "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
    })
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna(subset=["close"])

    # I10 -- verified, not trusted.
    assert_acquisition(df, AcquisitionRequest(
        source=f"yfinance:{TICKER}",
        requested_start=REQUESTED_START,
        expected_min_rows=EXPECTED_MIN_ROWS,
        # QQQ history starts 1999, well before the request, so the observed
        # start legitimately PRECEDES it -- that direction is never an error.
        # The slack guards the other direction: a start far AFTER the request.
        allowed_start_slack=pd.Timedelta(days=0),
        note="B.5 condition 3, BTC/QQQ correlation leg.",
    ))

    print(f"  {len(df):,} rows  {df.index.min().date()} -> {df.index.max().date()}")

    # I9 -- verify the point-in-time property on the cadence it is consumed
    # at, and verify the NAIVE join fails, so the guard is known to bind.
    bars = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    assert_no_lookahead(df[["close"]], align_point_in_time(df[["close"]], bars, MARKET_DAILY_CLOSE),
                        MARKET_DAILY_CLOSE)
    naive = df[["close"]].reindex(df.index.union(bars)).sort_index().ffill().reindex(bars)
    try:
        assert_no_lookahead(df[["close"]], naive, MARKET_DAILY_CLOSE)
    except AssertionError:
        print("  I9 guard binds: the naive value-timestamp join is rejected.")
    else:
        raise SystemExit("!! I9 guard did NOT reject the naive join -- it is not binding.")

    save_raw_data(df, OUT)
    print(f"\nsaved -> {OUT}")
    print(f"observation policy: {MARKET_DAILY_CLOSE.name} "
          f"(+{MARKET_DAILY_CLOSE.business_day_lag} business day)")
    print("\nRe-freeze before measuring: python scripts/freeze_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
