"""
Stage A.1 -- backfill t1 for the existing labelled dataset and measure how
much label overlap there actually is.

There is no persisted labelled dataset in this repo (`data/raw/` holds raw
OHLCV only), so "backfill" here means: regenerate the labelled frame from
the same raw inputs, through the same `build_asset_modeling_frame` path
every prior measurement used, now carrying the Stage A.1 metadata.

This script MEASURES. It does not implement uniqueness weighting -- that is
Stage A.2. The n_eff figure printed at the end is a preview computed
directly from the concurrency counts, reported so the size of the problem
is visible before A.2 is built.

Run: python scripts/measure_label_overlap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictive_core.dataset import build_asset_modeling_frame  # noqa: E402

RAW = Path("data/raw")
MAX_CANDLES = 48  # config.yaml predictive_core.triple_barrier.max_candles


def _pct(series: pd.Series, q: float) -> float:
    return float(np.percentile(series.to_numpy(dtype=float), q))


def describe_durations(out: pd.DataFrame, label: str) -> None:
    d = out["duration"]
    print(f"\n--- {label}: label duration (t1 - t0), in bars ---")
    print(f"  n labels        : {len(d):,}")
    print(f"  min             : {int(d.min())}")
    print(f"  median          : {_pct(d, 50):.1f}")
    print(f"  mean            : {d.mean():.2f}")
    print(f"  p75             : {_pct(d, 75):.1f}")
    print(f"  p90             : {_pct(d, 90):.1f}")
    print(f"  p99             : {_pct(d, 99):.1f}")
    print(f"  max             : {int(d.max())}")

    print(f"\n--- {label}: resolution route ---")
    counts = out["barrier_touched"].value_counts()
    for route in ("tp", "sl", "vertical"):
        n = int(counts.get(route, 0))
        share = 100.0 * n / len(out) if len(out) else 0.0
        sub = out.loc[out["barrier_touched"] == route, "duration"]
        med = f"{_pct(sub, 50):.1f}" if len(sub) else "n/a"
        p90 = f"{_pct(sub, 90):.1f}" if len(sub) else "n/a"
        print(f"  {route:<9}: {n:>7,} ({share:5.2f}%)   median dur {med:>5}   p90 {p90:>5}")


def measure_overlap(out: pd.DataFrame, label: str) -> None:
    """Concurrency c_t = number of label windows [t0_i, t1_i] spanning bar t.

    This is the quantity that makes nominal n fictional (Invariant I6): if
    c_t averages k, then roughly k labels are all reading the same bar, and
    they cannot be k independent observations of it.
    """
    # Positions on the labelled frame's own axis. Every t1 is a real bar in
    # the source index, but a label's t1 can fall on a bar that was itself
    # dropped as unlabelled (warm-up / unobservable horizon), so map onto a
    # common bar axis built from the union of t0s and t1s.
    bars = pd.Index(sorted(set(out["t0"]) | set(out["t1"])))
    s = bars.get_indexer(out["t0"])
    e = bars.get_indexer(out["t1"])
    assert (s >= 0).all() and (e >= 0).all(), "t0/t1 not found on the bar axis"

    m = len(bars)
    diff = np.zeros(m + 1, dtype=np.int64)
    np.add.at(diff, s, 1)
    np.add.at(diff, e + 1, -1)
    c = np.cumsum(diff)[:m]           # concurrency per bar
    spanned = c > 0

    print(f"\n--- {label}: concurrency (how many labels read the same bar) ---")
    print(f"  bars spanned by >=1 label : {int(spanned.sum()):,} of {m:,}")
    print(f"  mean concurrency          : {c[spanned].mean():.2f}")
    print(f"  median concurrency        : {np.median(c[spanned]):.1f}")
    print(f"  p90 concurrency           : {np.percentile(c[spanned], 90):.1f}")
    print(f"  max concurrency           : {int(c.max())}")

    # Average uniqueness per label, then n_eff = sum. PREVIEW ONLY -- the
    # implementation that feeds training weights is Stage A.2.
    inv_c = np.zeros(m, dtype=float)
    inv_c[spanned] = 1.0 / c[spanned]
    prefix = np.concatenate([[0.0], np.cumsum(inv_c)])
    span_len = (e - s + 1).astype(float)
    u_bar = (prefix[e + 1] - prefix[s]) / span_len
    n_eff = float(u_bar.sum())

    print(f"\n--- {label}: PREVIEW of effective n (Stage A.2 implements this properly) ---")
    print(f"  nominal n       : {len(out):,}")
    print(f"  mean uniqueness : {u_bar.mean():.4f}")
    print(f"  effective n     : {n_eff:,.0f}   ({100.0 * n_eff / len(out):.2f}% of nominal)")


def main() -> None:
    for path, name in ((RAW / "BTC_USDT_1h.parquet", "BTC/USDT 1h"),
                       (RAW / "ETH_USDT_1h.parquet", "ETH/USDT 1h")):
        if not path.exists():
            print(f"\n!! {name}: {path} not found -- SKIPPED (stated, not silently omitted)")
            continue

        raw = pd.read_parquet(path)
        out = build_asset_modeling_frame(raw, max_candles=MAX_CANDLES)

        print("\n" + "=" * 74)
        print(f"{name}   raw rows {len(raw):,}   labelled rows {len(out):,}"
              f"   ({out.index.min().date()} -> {out.index.max().date()})")
        print("=" * 74)

        missing = [c for c in ("t0", "t1", "barrier_touched", "realised_return")
                   if c not in out.columns]
        if missing:
            print(f"!! Stage A.1 columns missing: {missing}")
            continue

        describe_durations(out, name)
        measure_overlap(out, name)

        rr = out["realised_return"]
        print(f"\n--- {name}: realised_return by resolution route ---")
        by_route = out.groupby("barrier_touched")["realised_return"].agg(["mean", "count"])
        for route, row in by_route.iterrows():
            print(f"  {route:<9}: mean {row['mean']:+.6f}   n {int(row['count']):,}")
        print(f"  overall  : mean {rr.mean():+.6f}   median {rr.median():+.6f}   "
              f"min {rr.min():+.4f}   max {rr.max():+.4f}")
        print("  NOTE: conditional on barrier_touched, realised_return is exactly")
        print("        mult*ATR/entry -- it varies across rows only with ATR/price,")
        print("        never with how the trade actually went. See labeling.py.")


if __name__ == "__main__":
    main()
