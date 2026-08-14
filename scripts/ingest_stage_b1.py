"""
Stage B.1 ingestion -- funding rates, FRED macro, 1D/1W bars, perp-spot basis.

Everything runs through the existing Layer 1 quality pipeline (duplicate /
clock-skew removal, gap classification, outlier detection). Reports coverage
window, gap counts by classification, and row count for each source, and
flags any source whose coverage is shorter than the BTC 1h reference window.

Run: python scripts/ingest_stage_b1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.crypto_fetcher import (  # noqa: E402
    _timeframe_minutes, classify_gaps, fetch_ohlcv, flag_outliers,
    remove_duplicates_and_skew, save_raw_data,
)
from src.data_engine.derivatives_fetcher import (  # noqa: E402
    classify_funding_gaps, fetch_funding_rates, fetch_perp_spot_basis,
)
from src.data_engine.fred_fetcher import (  # noqa: E402
    FRED_SERIES, classify_macro_gaps, fetch_macro_frame,
)

RAW = Path("data/raw")
REPORTS = []


def record(name, df, gaps=None, outliers=None, note=""):
    REPORTS.append({
        "source": name,
        "rows": len(df),
        "start": df.index.min() if len(df) else None,
        "end": df.index.max() if len(df) else None,
        "gaps": (gaps.value_counts().to_dict() if gaps is not None and len(gaps) else {}),
        "outliers": (int(outliers.sum()) if outliers is not None and len(outliers) else 0),
        "note": note,
    })


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    # ---- reference window ------------------------------------------------ #
    btc1h = pd.read_parquet(RAW / "BTC_USDT_1h.parquet")
    ref_start, ref_end = btc1h.index.min(), btc1h.index.max()
    print(f"REFERENCE  BTC/USDT 1h: {len(btc1h):,} rows  {ref_start} -> {ref_end}\n")
    record("BTC/USDT 1h (reference)", btc1h,
           classify_gaps(btc1h, _timeframe_minutes("1h")), flag_outliers(btc1h))

    # ---- 1D / 1W bars ---------------------------------------------------- #
    exchange = ccxt.binance({"enableRateLimit": True})
    for tf in ("1d", "1w"):
        print(f"fetching BTC/USDT {tf} ...", flush=True)
        try:
            # since_ts=None makes fetch_ohlcv start from the exchange epoch and
            # retrieve FULL history -- ccxt's own since=None returns only the
            # latest candles, which silently truncates.
            df = fetch_ohlcv("BTC/USDT", timeframe=tf, exchange=exchange, since_ts=None)
            df = remove_duplicates_and_skew(df, timeframe=tf)
            gaps = classify_gaps(df, _timeframe_minutes(tf))
            outl = flag_outliers(df)
            save_raw_data(df, RAW / f"BTC_USDT_{tf}.parquet")
            record(f"BTC/USDT {tf} OHLCV", df, gaps, outl,
                   "unblocks hypothesis condition 4 (MTF alignment)")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {tf} FAILED: {type(exc).__name__}: {exc}")
            REPORTS.append({"source": f"BTC/USDT {tf} OHLCV", "rows": 0, "start": None,
                            "end": None, "gaps": {}, "outliers": 0,
                            "note": f"FETCH FAILED: {exc}"})

    # ---- funding rates --------------------------------------------------- #
    print("fetching funding rates (paginated, full history) ...", flush=True)
    funding = fetch_funding_rates("BTCUSDT")
    if len(funding):
        fgaps = classify_funding_gaps(funding)
        save_raw_data(funding, RAW / "BTCUSDT_funding_8h.parquet")
        record("Funding rate 8h", funding, fgaps, None,
               "Binance /fapi/v1/fundingRate. Perp inception bounds this, not the 1h window.")

    # ---- perp-spot basis ------------------------------------------------- #
    print("fetching perp mark + spot for basis (paginated) ...", flush=True)
    basis = fetch_perp_spot_basis("BTCUSDT", "1h", start_ms=0)
    if len(basis):
        bgaps = classify_gaps(basis, _timeframe_minutes("1h"))
        save_raw_data(basis, RAW / "BTCUSDT_basis_1h.parquet")
        record("Perp-spot basis 1h", basis, bgaps, None,
               "computed (mark - spot)/spot on the index INTERSECTION")

    # ---- FRED macro ------------------------------------------------------ #
    print("fetching FRED macro ...", flush=True)
    macro = fetch_macro_frame()
    save_raw_data(macro, RAW / "macro_fred_daily.parquet")
    for name in FRED_SERIES:
        s = macro[name].dropna()
        record(f"FRED {name} ({FRED_SERIES[name][0]})",
               s.to_frame(), classify_macro_gaps(macro, name), None,
               FRED_SERIES[name][2])

    # ---- report ---------------------------------------------------------- #
    print("\n" + "=" * 112)
    print("STAGE B.1 INGESTION -- coverage, gaps, row counts")
    print("=" * 112)
    print(f"{'source':<34}{'rows':>10}{'start':>13}{'end':>13}{'outliers':>10}  gaps by class")
    print("-" * 112)
    for r in REPORTS:
        start = r["start"].date() if r["start"] is not None else "n/a"
        end = r["end"].date() if r["end"] is not None else "n/a"
        gapstr = ", ".join(f"{k}={v:,}" for k, v in sorted(r["gaps"].items())
                           if k != "normal") or "none"
        print(f"{r['source']:<34}{r['rows']:>10,}{str(start):>13}{str(end):>13}"
              f"{r['outliers']:>10,}  {gapstr}")

    print("\n" + "=" * 112)
    print("COVERAGE vs BTC 1h REFERENCE WINDOW")
    print("=" * 112)
    for r in REPORTS:
        if r["start"] is None or "reference" in r["source"]:
            continue
        short_start = r["start"] > ref_start + pd.Timedelta(days=2)
        short_end = r["end"] < ref_end - pd.Timedelta(days=7)
        if short_start or short_end:
            lost = (r["start"] - ref_start).days if short_start else 0
            flag = "SHORTER THAN BTC 1h"
            detail = []
            if short_start:
                detail.append(f"starts {lost:,} days later ({r['start'].date()} vs {ref_start.date()})")
            if short_end:
                detail.append(f"ends {(ref_end - r['end']).days:,} days early")
            print(f"  [{flag}] {r['source']}: " + "; ".join(detail))
        else:
            print(f"  [ok] {r['source']}: covers the reference window")

    print("\n" + "=" * 112)
    print("OUTLIER FLAG RATE -- the Layer 1 threshold is calibrated for 1h bars")
    print("=" * 112)
    for r in REPORTS:
        if not r["rows"] or r["outliers"] == 0:
            continue
        rate = r["outliers"] / r["rows"]
        warn = "  <- UNINFORMATIVE at this timeframe" if rate > 0.05 else ""
        print(f"  {r['source']:<34}{r['outliers']:>7,} of {r['rows']:>7,} = {rate:>6.1%}{warn}")
    print("\n  `flag_outliers(threshold_pct=10.0)` treats a 10% bar move as extreme, which")
    print("  it is on 1h data. On 1d and especially 1w it is routine for BTC, so the flag")
    print("  saturates and carries no information. NOT retuned here -- the brief specified")
    print("  the existing pipeline, and silently changing a threshold to make a number look")
    print("  reasonable is the failure mode this project guards against. It needs a")
    print("  timeframe-aware threshold before 1d/1w outlier flags are used for anything.")

    print("\nNOTES")
    for r in REPORTS:
        if r["note"]:
            print(f"  - {r['source']}: {r['note']}")


if __name__ == "__main__":
    main()
