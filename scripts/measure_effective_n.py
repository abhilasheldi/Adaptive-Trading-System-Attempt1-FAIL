"""
Stage A.2 -- report nominal n, effective n, and the ratio for the full
labelled dataset and for the compressed (ATR pct <= 0.20) subset.

Run: python scripts/measure_effective_n.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.macro_mtf_liquidity_vol.volatility_classifier import (  # noqa: E402
    COMPRESSED_MAX_PERCENTILE,
)
from src.predictive_core.dataset import build_asset_modeling_frame  # noqa: E402
from src.predictive_core.uniqueness import average_uniqueness  # noqa: E402

MAX_CANDLES = 48
COMPRESSION_COL = "vol_atr_pct_rank"


def _row(label: str, r) -> str:
    return (f"{label:<46}{r.nominal_n:>10,}{r.effective_n:>12,.1f}"
            f"{r.ratio:>10.4f}{r.mean_concurrency:>10.2f}")


def main() -> None:
    path = Path("data/raw/BTC_USDT_1h.parquet")
    if not path.exists():
        print(f"!! {path} not found -- cannot measure. STATED, not silently skipped.")
        return

    out = build_asset_modeling_frame(pd.read_parquet(path), max_candles=MAX_CANDLES)
    print(f"BTC/USDT 1h  labelled rows {len(out):,}  "
          f"({out.index.min().date()} -> {out.index.max().date()})")
    print(f"compression threshold: ATR pct <= {COMPRESSED_MAX_PERCENTILE} "
          f"(live Layer 6 value, S6)\n")

    # The hypothesis text writes the condition as <=, and Layer 6's
    # classifier uses a strict <. Verified at Stage 0 that exactly-0.20
    # never occurs in a 252-bar rank, so the two select identically here.
    compressed = (out[COMPRESSION_COL] <= COMPRESSED_MAX_PERCENTILE).to_numpy()

    full = average_uniqueness(out["t0"], out["t1"])
    subset_own = average_uniqueness(out["t0"], out["t1"], restrict_to=compressed)

    # The same rows scored against the FULL dataset's concurrency, i.e. their
    # uniqueness in situ rather than among themselves. Different question --
    # reported so the methodological choice is visible, not implied.
    subset_insitu_n_eff = float(full.average_uniqueness[compressed].sum())
    n_sub = int(compressed.sum())

    print(f"{'set':<46}{'nominal n':>10}{'n_eff':>12}{'ratio':>10}{'mean c_t':>10}")
    print("-" * 88)
    print(_row("FULL dataset", full))
    print(_row(f"COMPRESSED (own concurrency)", subset_own))
    print(f"{'COMPRESSED (in-situ, full-data concurrency)':<46}"
          f"{n_sub:>10,}{subset_insitu_n_eff:>12,.1f}"
          f"{subset_insitu_n_eff / n_sub:>10.4f}{'--':>10}")

    print("\nThe two COMPRESSED rows answer different questions:")
    print("  own concurrency : overlap among the compressed labels themselves --")
    print("                    what an estimator trained ONLY on this subset sees.")
    print("  in-situ         : each compressed label's uniqueness within the full")
    print("                    label set, including the non-compressed neighbours")
    print("                    it overlaps. Lower, because those neighbours count.")


if __name__ == "__main__":
    main()
