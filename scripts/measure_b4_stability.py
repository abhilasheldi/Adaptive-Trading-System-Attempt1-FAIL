"""
B.4 stability check -- does the selected config hold on each half of the
flow window independently, or only on the pooled figure?

NOT hyperparameter search. No new configuration is introduced and nothing is
selected here: the config was fixed at B.4 and this asks whether its
confirmation rate is stable across time or an artifact of pooling two
dissimilar regimes. A rate that only lands in-band when 2020 and 2025 are
averaged together is not a property of the detector.

Two configs are measured:

  0.05 / 0.25   the SELECTED config (tolerance 0.05, penetration held at 0.25)
  0.05 / 0.05   the COUPLED config -- included because it is easy to conflate
                with the above by its tolerance alone. It scored 9.84% pooled,
                well outside the band, and is reported here to keep the
                distinction visible rather than assumed.

The band is [1%, 5%] on compressed bars, unchanged from B.4 and still fixed
in advance. No downstream metric is computed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.freeze import load_frozen, read_manifest  # noqa: E402
from src.predictive_core.dataset import RAW_ATR_COLUMN, build_asset_modeling_frame  # noqa: E402

from measure_b4_sweep_retune import (  # noqa: E402
    ATR_PERCENTILE_COL,
    COMPRESSION_MAX,
    TARGET_BAND,
    scan,
)

FLOW_START = pd.Timestamp("2019-12-23", tz="UTC")
CONFIGS = [(0.05, 0.25, "SELECTED"), (0.05, 0.05, "coupled (not selected)")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used rather than the normal approximation
    because halving the sample is the whole point of this check and the
    normal approximation degrades exactly where it would matter."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def main() -> int:
    m = read_manifest()
    print(f"frozen dataset v{m['version']}  {m['dataset_sha256'][:16]}...")

    df = build_asset_modeling_frame(load_frozen("BTC_USDT_1h.parquet"), keep_raw_atr=True)
    compressed_all = df[ATR_PERCENTILE_COL] <= COMPRESSION_MAX

    flow_bars = df.index[df.index >= FLOW_START]
    midpoint = flow_bars.min() + (flow_bars.max() - flow_bars.min()) / 2
    print(f"flow window {flow_bars.min().date()} -> {flow_bars.max().date()}  ({len(flow_bars):,} bars)")
    print(f"time midpoint: {midpoint.date()}\n")

    halves = [
        ("H1", lambda i: (i >= FLOW_START) & (i < midpoint)),
        ("H2", lambda i: i >= midpoint),
        ("pooled", lambda i: i >= FLOW_START),
    ]

    rows = []
    for tol, pen, label in CONFIGS:
        res = scan(df, tol, pen)
        comp = compressed_all.reindex(res.index).fillna(False)
        for name, mask_fn in halves:
            sel = pd.Series(mask_fn(res.index), index=res.index)
            comp_sel = comp & sel
            n = int(comp_sel.sum())
            k = int((res["confirmed"] & comp_sel).sum())
            lo, hi = wilson(k, n)
            rows.append({
                "config": f"{tol:.2f}/{pen:.2f}", "label": label, "half": name,
                "compressed_bars": n, "events": k,
                "rate": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
                "in_band": TARGET_BAND[0] <= (k / n if n else -1) <= TARGET_BAND[1],
                "start": res.index[sel][0] if sel.any() else pd.NaT,
                "end": res.index[sel][-1] if sel.any() else pd.NaT,
            })

    table = pd.DataFrame(rows)

    print("=" * 100)
    print("B.4 STABILITY -- confirmation rate on compressed bars, flow window split by time")
    print("=" * 100)
    print(f"{'config':<12}{'half':<9}{'window':<26}{'comp bars':>11}{'events':>8}"
          f"{'rate':>9}{'95% CI':>18}  band")
    print("-" * 100)
    for _, r in table.iterrows():
        win = f"{str(r['start'])[:10]} -> {str(r['end'])[:10]}"
        ci = f"[{r['ci_lo']:.2%}, {r['ci_hi']:.2%}]"
        mark = "IN" if r["in_band"] else "OUT"
        print(f"{r['config']:<12}{r['half']:<9}{win:<26}{r['compressed_bars']:>11,}"
              f"{r['events']:>8,}{r['rate']:>9.2%}{ci:>18}  {mark}")
        if r["half"] == "pooled":
            print("-" * 100)

    print(f"\nband = [{TARGET_BAND[0]:.0%}, {TARGET_BAND[1]:.0%}] on compressed bars, fixed at B.4.")

    sel = table[(table["label"] == "SELECTED") & (table["half"].isin(["H1", "H2"]))]
    if sel["in_band"].all():
        print("\nSELECTED config lands in-band on BOTH halves independently.")
        print("The pooled 4.03% is not an averaging artifact.")
    else:
        out = sel[~sel["in_band"]]["half"].tolist()
        print(f"\nSELECTED config falls OUT of band on: {out}.")
        print("The pooled figure conceals a regime split -- the narrow-escape "
              "concern is CONFIRMED, not addressed.")

    out_path = Path("reports/b4_stability.csv")
    out_path.parent.mkdir(exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nwritten -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
