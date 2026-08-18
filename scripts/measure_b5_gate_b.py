"""
B.5 STEP 5 -- the four-condition event count and GATE B.

All four conditions are now reachable. Both blocked ones were pre-registered
and COMMITTED BEFORE this script produced any number:

    prereg/condition_3_macro.md   commit a5a8fbd8
    prereg/condition_4_mtf.md     commit a5a8fbd8

Nothing here is swept or tuned. The sweep tolerance is frozen from B.4, the
compression threshold is a priori from H1, and both new conditions come from
the registrations. If a count disappoints, no definition moves.

    python scripts/measure_b5_gate_b.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.freeze import load_frozen, read_manifest  # noqa: E402
from src.macro_mtf_liquidity_vol.conditions import (  # noqa: E402
    MTF_MIN_ALIGNMENT,
    N_PERIODS,
    READING_LABELS,
    alignment_against,
    condition_3_holds,
    macro_state_at_bars,
    mtf_readings_at_bars,
)
from src.macro_mtf_liquidity_vol.liquidity_trap import (  # noqa: E402
    DEFAULT_LEVEL_TOLERANCE_ATR,
    DEFAULT_MIN_PENETRATION_ATR,
    EQUAL_LEVEL_LOOKBACK,
    estimate_liquidity_trap,
)
from src.predictive_core.dataset import RAW_ATR_COLUMN, build_asset_modeling_frame  # noqa: E402
from src.predictive_core.trial_registry import log_trial  # noqa: E402
from src.predictive_core.uniqueness import average_uniqueness  # noqa: E402

ATR_PCT_COL = "vol_atr_14"
COMPRESSION_MAX = 0.20
GATE_MIN_EFFECTIVE = 200
MIDPOINT = pd.Timestamp("2023-04-18", tz="UTC")

WINDOWS = {
    "flow (2019-12-23 ->) [GATE]": pd.Timestamp("2019-12-23", tz="UTC"),
    "funding-only (2019-09-10 ->)": pd.Timestamp("2019-09-10", tz="UTC"),
    "full OHLCV (reference)": None,
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def compression_episode_id(compressed: pd.Series) -> pd.Series:
    c = compressed.astype(bool)
    starts = c & ~c.shift(1, fill_value=False)
    return starts.cumsum().where(c, -1).astype(int)


def collapse(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    return events[~events.duplicated(subset=["episode", "direction"], keep="first")]


def effective(events: pd.DataFrame) -> float:
    if events.empty:
        return 0.0
    return float(average_uniqueness(events["t0"], events["t1"]).effective_n)


def main() -> int:
    m = read_manifest()
    print(f"frozen dataset v{m['version']}  {m['dataset_sha256'][:16]}...")
    print(f"sweep tuning frozen at B.4: tol={DEFAULT_LEVEL_TOLERANCE_ATR}, "
          f"pen={DEFAULT_MIN_PENETRATION_ATR}")
    print(f"conditions 3 and 4 registered at commit a5a8fbd8, N={N_PERIODS}, "
          f"MTF >= {MTF_MIN_ALIGNMENT}/4\n")

    df = build_asset_modeling_frame(load_frozen("BTC_USDT_1h.parquet"), keep_raw_atr=True)
    daily = load_frozen("BTC_USDT_1d.parquet")
    weekly = load_frozen("BTC_USDT_1w.parquet")
    macro = load_frozen("macro_fred_daily.parquet")
    qqq = load_frozen("QQQ_daily.parquet")

    # --- condition 1 --------------------------------------------------- #
    compressed = df[ATR_PCT_COL] <= COMPRESSION_MAX
    episodes = compression_episode_id(compressed)

    # --- condition 2 --------------------------------------------------- #
    lb = EQUAL_LEVEL_LOOKBACK
    idx, conf, direc = [], [], []
    for i in range(lb, len(df)):
        try:
            r = estimate_liquidity_trap(
                df.iloc[i - lb: i + 1], atr_col=RAW_ATR_COLUMN,
                level_tolerance_atr=DEFAULT_LEVEL_TOLERANCE_ATR,
                min_penetration_atr=DEFAULT_MIN_PENETRATION_ATR,
            )
        except ValueError:
            continue
        idx.append(df.index[i]); conf.append(bool(r.sweep_just_confirmed))
        direc.append(r.sweep_direction)
    sweep = pd.DataFrame({"confirmed": conf, "direction": direc}, index=pd.DatetimeIndex(idx))

    bars = sweep.index
    comp_b = compressed.reindex(bars).fillna(False)

    # --- condition 3 --------------------------------------------------- #
    print("classifying macro regime per bar (condition 3)...")
    states = macro_state_at_bars(macro, qqq, daily, bars)
    cond3 = condition_3_holds(states)
    print(f"  states: {states.value_counts(dropna=False).to_dict()}")

    # --- condition 4 --------------------------------------------------- #
    print("scoring MTF alignment (condition 4)...")
    readings = mtf_readings_at_bars(daily, weekly, bars)
    implied = sweep["direction"].map({"bullish": 1, "bearish": -1})
    align = pd.Series(np.nan, index=bars, dtype=float)
    for ts in bars[sweep["confirmed"].to_numpy()]:
        d = implied.loc[ts]
        if pd.isna(d):
            continue
        a = alignment_against(readings.loc[ts].to_dict(), int(d))
        if a is not None:
            align.loc[ts] = a
    cond4 = align >= MTF_MIN_ALIGNMENT

    # --- cumulative ---------------------------------------------------- #
    c1 = comp_b
    c12 = c1 & sweep["confirmed"]
    c123 = c12 & cond3
    c1234 = c123 & cond4

    print("\n" + "=" * 96)
    print("CUMULATIVE CONDITION FILTER (BTC, full OHLCV window)")
    print("=" * 96)
    rows = [
        ("bars scanned", len(bars)),
        ("+ 1. atr_percentile <= 0.20", int(c1.sum())),
        ("+ 2. confirmed_sweep", int(c12.sum())),
        ("+ 3. macro_regime in {risk-on, transitional}", int(c123.sum())),
        (f"+ 4. mtf_alignment >= {MTF_MIN_ALIGNMENT}/4", int(c1234.sum())),
    ]
    prev = None
    for label, n in rows:
        drop = "" if prev is None else f"   (-{prev - n:,}, -{1 - n / prev:.1%})"
        print(f"{label:<48}{n:>10,}{drop}")
        prev = n

    print("\nEACH CONDITION INDIVIDUALLY")
    print("-" * 96)
    print(f"{'1. compression alone':<48}{int(c1.sum()):>10,}  ({c1.mean():.2%} of bars)")
    print(f"{'2. confirmed sweep alone':<48}{int(sweep['confirmed'].sum()):>10,}"
          f"  ({sweep['confirmed'].mean():.2%} of bars)")
    print(f"{'3. macro admitted alone':<48}{int(cond3.sum()):>10,}  ({cond3.mean():.2%} of bars)")
    n_scored = int(align.notna().sum())
    print(f"{'4. MTF >= 3/4 (only defined on sweep bars)':<48}{int(cond4.sum()):>10,}"
          f"  ({cond4.sum() / max(n_scored, 1):.2%} of {n_scored:,} scored)")

    print(f"\nMTF alignment distribution over scored sweep bars:")
    print(f"  {align.value_counts().sort_index().to_dict()}")

    # --- events + independence ----------------------------------------- #
    ev_idx = c1234[c1234].index
    events = pd.DataFrame({
        "direction": sweep.loc[ev_idx, "direction"],
        "episode": episodes.reindex(ev_idx),
        "t0": df.loc[ev_idx, "t0"], "t1": df.loc[ev_idx, "t1"],
    })

    print("\n" + "=" * 96)
    print("FOUR-CONDITION EVENT COUNT -- THE GATE NUMBER")
    print("=" * 96)
    print(f"{'window':<32}{'raw':>10}{'structural':>13}{'EFFECTIVE':>12}{'eff/raw':>10}")
    print("-" * 96)
    out_rows = []
    for name, start in WINDOWS.items():
        sel = events if start is None else events[events.index >= start]
        col = collapse(sel)
        eff = effective(col)
        out_rows.append({"window": name, "raw": len(sel), "structural": len(col), "effective": eff})
        ratio = eff / len(col) if len(col) else float("nan")
        print(f"{name:<32}{len(sel):>10,}{len(col):>13,}{eff:>12.1f}{ratio:>10.3f}")
    print("-" * 96)

    gate = out_rows[0]["effective"]

    # --- H1/H2 split ---------------------------------------------------- #
    flow = events[events.index >= WINDOWS["flow (2019-12-23 ->) [GATE]"]]
    print(f"\nH1/H2 SPLIT ON THE FLOW WINDOW (midpoint {MIDPOINT.date()})")
    print("-" * 96)
    print(f"{'half':<8}{'raw':>8}{'structural':>13}{'effective':>12}{'comp bars':>12}"
          f"{'events/comp bar':>18}")
    for half, mask in (("H1", flow.index < MIDPOINT), ("H2", flow.index >= MIDPOINT)):
        sub = collapse(flow[mask])
        cb = int(comp_b[(comp_b.index >= WINDOWS["flow (2019-12-23 ->) [GATE]"])
                        & ((comp_b.index < MIDPOINT) if half == "H1" else (comp_b.index >= MIDPOINT))].sum())
        lo, hi = wilson(len(sub), cb)
        print(f"{half:<8}{int(mask.sum()):>8,}{len(sub):>13,}{effective(sub):>12.1f}"
              f"{cb:>12,}{f'{len(sub)/cb:.3%}' if cb else 'n/a':>18}"
              f"  [{lo:.3%}, {hi:.3%}]")
    print("\nWilson intervals on structural events per compressed bar -- subgroup counts are")
    print("low, so a normal approximation would misstate them.")

    # --- GATE ------------------------------------------------------------ #
    print("\n" + "=" * 96)
    print(f"GATE B -- effective count >= {GATE_MIN_EFFECTIVE} on the flow window")
    print("=" * 96)
    print(f"  effective = {gate:.1f}")
    if gate < GATE_MIN_EFFECTIVE:
        print(f"\n  GATE B: FAILS. {gate:.1f} < {GATE_MIN_EFFECTIVE}.")
        print("  H1 fails on sample grounds. Decisive negative -> free revision or")
        print("  terminate. No condition, threshold or independence rule is loosened.")
    else:
        print(f"\n  GATE B: PASSES. {gate:.1f} >= {GATE_MIN_EFFECTIVE}.")
        print("  Sample is sufficient. Requires operator sign-off before Stage B.2.")
    print("=" * 96)

    log_trial(
        stage="B.5",
        params={
            "compression_max": COMPRESSION_MAX,
            "level_tolerance_atr": DEFAULT_LEVEL_TOLERANCE_ATR,
            "min_penetration_atr": DEFAULT_MIN_PENETRATION_ATR,
            "mtf_min_alignment": MTF_MIN_ALIGNMENT,
            "n_periods": N_PERIODS,
            "mtf_readings": list(READING_LABELS),
            "macro_dominance_leg": "removed",
            "prereg_commit": "a5a8fbd8723ec9ccdf09150ed8b709f272f588a3",
            "frozen_dataset": m["dataset_sha256"],
        },
        result_metrics={
            "gate_b_effective": gate,
            "gate_b_raw": out_rows[0]["raw"],
            "gate_b_structural": out_rows[0]["structural"],
            "gate_b_threshold": GATE_MIN_EFFECTIVE,
            "gate_b_passes": bool(gate >= GATE_MIN_EFFECTIVE),
        },
        n_eff=gate,
        nominal_n=out_rows[0]["raw"],
        source="manual",
        note="B.5 four-condition event count. Conditions 3 and 4 pre-registered before counting.",
    )
    print("\nlogged to trials.jsonl")

    out = Path("reports/b5_gate_b.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
