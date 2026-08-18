"""
B.4 -- sweep detector re-specification.

Sweeps `level_tolerance_atr` over {0.05, 0.10, 0.15} (0.25 retained as the
current baseline for reference) and applies the compression precondition
`atr_percentile <= 0.20`, which the source definition specified and which
was never wired in.

=========================================================================
SELECTION RULE, FIXED IN ADVANCE
=========================================================================
Select the tolerance whose CONFIRMATION RATE lands in low single digits
(target band 1-5%, stated here before any number is read). Do NOT select on
downstream performance -- that tunes the event definition to the outcome
and voids the result. No AUC, no EV, no Sharpe is computed by this script,
so the temptation is structurally absent rather than merely resisted.

`atr_percentile <= 0.20` is A PRIORI, from the hypothesis text. S6 is
retired and the measured justification for 0.20 over 0.30 no longer exists.
It is legitimate as an untested condition and is PROHIBITED from re-tuning.
This script never varies it.

=========================================================================
THE PENETRATION COUPLING, WHICH IS A REAL CHOICE
=========================================================================
`DEFAULT_MIN_PENETRATION_ATR` is documented as deliberately equal to
`DEFAULT_LEVEL_TOLERANCE_ATR`: a cluster's level is only defined to within
+/- the tolerance, so a "penetration" smaller than that sits inside the
level's own uncertainty band and cannot be distinguished from touching it.

That coupling means tightening the tolerance LOOSENS the penetration
requirement, which pushes the confirmation rate the opposite way from what
"tighter tolerance" intuitively suggests. Both arms are therefore measured:

  COUPLED    penetration = tolerance      (the documented invariant)
  FIXED      penetration = 0.25           (tolerance varied alone)

Reporting only one would hide that the sweep has two moving parts.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.freeze import load_frozen, read_manifest  # noqa: E402
from src.macro_mtf_liquidity_vol.liquidity_trap import (  # noqa: E402
    EQUAL_LEVEL_LOOKBACK,
    estimate_liquidity_trap,
)
from src.predictive_core.dataset import RAW_ATR_COLUMN, build_asset_modeling_frame  # noqa: E402

TOLERANCES = (0.05, 0.10, 0.15, 0.25)   # 0.25 = current default, reference only
ATR_PERCENTILE_COL = "vol_atr_14"
COMPRESSION_MAX = 0.20                   # a priori, from H1. Never varied.
TARGET_BAND = (0.01, 0.05)               # low single digits, set in advance


def scan(df: pd.DataFrame, tolerance: float, penetration: float) -> pd.DataFrame:
    """Run the detector on every bar with a full lookback window.

    Returns one row per scanned bar: confirmed flag and direction. The
    detector is handed exactly `lookback + 1` rows so its own
    strictly-prior windowing holds -- no lookahead is introduced here.
    """
    lb = EQUAL_LEVEL_LOOKBACK
    idx, confirmed, direction = [], [], []
    skipped = 0

    for i in range(lb, len(df)):
        window = df.iloc[i - lb: i + 1]
        try:
            r = estimate_liquidity_trap(
                window,
                atr_col=RAW_ATR_COLUMN,
                level_tolerance_atr=tolerance,
                min_penetration_atr=penetration,
            )
        except ValueError:
            # Non-finite ATR/price in the window. Counted, not silently
            # treated as "no sweep" -- that would understate the rate.
            skipped += 1
            continue
        idx.append(df.index[i])
        confirmed.append(bool(r.sweep_just_confirmed))
        direction.append(r.sweep_direction)

    out = pd.DataFrame(
        {"confirmed": confirmed, "direction": direction},
        index=pd.DatetimeIndex(idx),
    )
    out.attrs["skipped"] = skipped
    return out


def main() -> int:
    manifest = read_manifest()
    print(f"frozen dataset v{manifest['version']}  {manifest['dataset_sha256'][:16]}...")

    raw = load_frozen("BTC_USDT_1h.parquet")
    df = build_asset_modeling_frame(raw, keep_raw_atr=True)
    print(f"BTC/USDT 1h: {len(raw):,} raw bars -> {len(df):,} modeling rows "
          f"({df.index.min().date()} -> {df.index.max().date()})")

    compressed = df[ATR_PERCENTILE_COL] <= COMPRESSION_MAX
    print(f"compression precondition atr_percentile <= {COMPRESSION_MAX}: "
          f"{int(compressed.sum()):,} of {len(df):,} bars ({compressed.mean():.2%})")
    print(f"\nselection target set IN ADVANCE: confirmation rate in "
          f"[{TARGET_BAND[0]:.0%}, {TARGET_BAND[1]:.0%}] under the precondition.")
    print("No downstream metric is computed by this script.\n")

    rows = []
    for mode, pen_of in (("coupled", lambda t: t), ("fixed", lambda t: 0.25)):
        for tol in TOLERANCES:
            pen = pen_of(tol)
            t0 = time.time()
            res = scan(df, tol, pen)
            comp = compressed.reindex(res.index).fillna(False)

            n_all, n_comp = len(res), int(comp.sum())
            fired_all = int(res["confirmed"].sum())
            fired_comp = int((res["confirmed"] & comp).sum())
            bull = int(((res["direction"] == "bullish") & comp).sum())
            bear = int(((res["direction"] == "bearish") & comp).sum())

            rows.append({
                "mode": mode, "tolerance": tol, "penetration": pen,
                "rate_all": fired_all / n_all if n_all else float("nan"),
                "events_all": fired_all,
                "rate_comp": fired_comp / n_comp if n_comp else float("nan"),
                "events_comp": fired_comp,
                "bullish": bull, "bearish": bear,
                "skipped": res.attrs["skipped"],
                "secs": time.time() - t0,
            })
            r = rows[-1]
            print(f"  [{mode:<7} tol={tol:.2f} pen={pen:.2f}]  "
                  f"all-bars {r['rate_all']:6.2%} ({fired_all:,})   "
                  f"compressed {r['rate_comp']:6.2%} ({fired_comp:,})   "
                  f"{r['secs']:.0f}s")

    table = pd.DataFrame(rows)

    print("\n" + "=" * 104)
    print("B.4 SWEEP RETUNE -- confirmation rate and event count per tolerance")
    print("=" * 104)
    print(f"{'mode':<9}{'tol':>6}{'pen':>6}{'rate (all bars)':>18}{'events':>10}"
          f"{'rate (compressed)':>20}{'events':>10}{'bull/bear':>14}")
    print("-" * 104)
    for _, r in table.iterrows():
        band = " *" if TARGET_BAND[0] <= r["rate_comp"] <= TARGET_BAND[1] else "  "
        print(f"{r['mode']:<9}{r['tolerance']:>6.2f}{r['penetration']:>6.2f}"
              f"{r['rate_all']:>17.2%}{int(r['events_all']):>10,}"
              f"{r['rate_comp']:>19.2%}{band}{int(r['events_comp']):>10,}"
              f"{str(int(r['bullish'])) + '/' + str(int(r['bearish'])):>14}")
    print("-" * 104)
    print(f"* = confirmation rate inside the pre-registered "
          f"[{TARGET_BAND[0]:.0%}, {TARGET_BAND[1]:.0%}] band")
    print(f"\nS8 reference: 44.1% on all bars, against 2.04% for the earlier "
          f"compression+SFP proxy.")

    out = Path("reports/b4_sweep_retune.csv")
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
