"""
B.5 -- fully-conditioned event set and the Gate B count.

=========================================================================
SCOPE LIMITATION, STATED FIRST (I5)
=========================================================================
The brief states "all four conditions use data in hand -- no external
dependency." That is NOT true of this codebase as it stands, and this
script does not pretend otherwise.

  1. atr_percentile <= 0.20        COMPUTABLE
  2. confirmed_sweep               COMPUTABLE (tolerance frozen at B.4)
  3. macro_regime in {risk-on,     BLOCKED -- data
     transitional}                 `MacroSignalSnapshot` requires
                                   `btc_dominance_20d_trend_pct` and
                                   `btc_qqq_corr_20d`. Neither has an
                                   ingested source. Both are legs of
                                   MASTER_CONTEXT's Risk-On definition.
  4. mtf_alignment >= 3/4          BLOCKED -- specification
                                   `score_mtf_alignment` scores agreement
                                   across four ALREADY-DERIVED signed
                                   directional readings and states that
                                   deriving them is out of scope. Nothing
                                   in the repo derives them.

Neither module has a single caller outside its own test file. This is S13
recurring: "Layers 3 and 4 require data that is not ingested."

=========================================================================
WHAT THIS SCRIPT THEREFORE COMPUTES, AND WHY IT IS STILL DECISIVE
=========================================================================
The TWO-condition event set (compression AND confirmed sweep), with the
full independence treatment B.5 specifies.

Adding conditions can only REMOVE events. The two-condition effective count
is therefore a strict UPPER BOUND on the four-condition effective count:

    if 2-condition effective < 200  ->  4-condition effective < 200
                                        Gate B FAILS on sample grounds,
                                        decisively, today.

    if 2-condition effective >= 200 ->  INDETERMINATE. The gate cannot be
                                        evaluated until conditions 3 and 4
                                        are unblocked.

A bound is reported as a bound. It is never reported as the gate number,
and the gate is not declared passed on it.

=========================================================================
INDEPENDENCE -- PRE-SPECIFIED, NO NEW CONSTANT
=========================================================================
STRUCTURAL   two events in the same compression episode (an unbroken run of
             bars with atr_percentile <= 0.20) in the same direction count
             as ONE. The first is kept.
TEMPORAL     concurrency over event label windows [t0, t1] via the A.2
             uniqueness code; effective count = sum of average uniqueness.

No temporal-separation constant is invented. Both rules come from
MASTER_CONTEXT section 7.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.freeze import load_frozen, read_manifest  # noqa: E402
from src.macro_mtf_liquidity_vol.liquidity_trap import (  # noqa: E402
    DEFAULT_LEVEL_TOLERANCE_ATR,
    DEFAULT_MIN_PENETRATION_ATR,
    EQUAL_LEVEL_LOOKBACK,
    estimate_liquidity_trap,
)
from src.predictive_core.dataset import RAW_ATR_COLUMN, build_asset_modeling_frame  # noqa: E402
from src.predictive_core.uniqueness import average_uniqueness  # noqa: E402

ATR_PCT_COL = "vol_atr_14"
COMPRESSION_MAX = 0.20                      # a priori, H1 text. Never varied.
GATE_MIN_EFFECTIVE = 200

WINDOWS = {
    "flow (2019-12-23 ->)  [GATE]": pd.Timestamp("2019-12-23", tz="UTC"),
    "funding-only (2019-09-10 ->)": pd.Timestamp("2019-09-10", tz="UTC"),
    "full OHLCV (reference)": None,
}


def sweep_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Confirmed-sweep flag and direction per bar, at the B.4-frozen tuning."""
    lb = EQUAL_LEVEL_LOOKBACK
    idx, conf, direc = [], [], []
    for i in range(lb, len(df)):
        try:
            r = estimate_liquidity_trap(
                df.iloc[i - lb: i + 1],
                atr_col=RAW_ATR_COLUMN,
                level_tolerance_atr=DEFAULT_LEVEL_TOLERANCE_ATR,
                min_penetration_atr=DEFAULT_MIN_PENETRATION_ATR,
            )
        except ValueError:
            continue
        idx.append(df.index[i])
        conf.append(bool(r.sweep_just_confirmed))
        direc.append(r.sweep_direction)
    return pd.DataFrame({"confirmed": conf, "direction": direc}, index=pd.DatetimeIndex(idx))


def compression_episode_id(compressed: pd.Series) -> pd.Series:
    """Label each unbroken run of compressed bars with an episode id.

    A run BREAKS the moment a bar exceeds the threshold, which is what
    "unbroken run of ATR pct <= 0.20" means. Non-compressed bars get -1.
    """
    c = compressed.astype(bool)
    starts = c & ~c.shift(1, fill_value=False)
    ids = starts.cumsum().where(c, -1)
    return ids.astype(int)


def collapse_structural(events: pd.DataFrame) -> pd.DataFrame:
    """One event per (compression episode, direction). Keeps the first."""
    if events.empty:
        return events
    return events[~events.duplicated(subset=["episode", "direction"], keep="first")]


def effective_count(events: pd.DataFrame) -> float:
    """Sum of average uniqueness over event label windows (A.2 code)."""
    if events.empty:
        return 0.0
    return float(average_uniqueness(events["t0"], events["t1"]).effective_n)


def report_window(name: str, start, events: pd.DataFrame, compressed: pd.Series) -> dict:
    sel = events if start is None else events[events.index >= start]
    collapsed = collapse_structural(sel)
    eff = effective_count(collapsed)
    n_comp = int(compressed[compressed.index >= start].sum()) if start is not None else int(compressed.sum())
    return {
        "window": name,
        "compressed_bars": n_comp,
        "raw_events": len(sel),
        "after_structural": len(collapsed),
        "effective": eff,
        "eff_over_raw": eff / len(collapsed) if len(collapsed) else float("nan"),
    }


def main() -> int:
    m = read_manifest()
    print(f"frozen dataset v{m['version']}  {m['dataset_sha256'][:16]}...")
    print(f"sweep tuning frozen at B.4: tolerance={DEFAULT_LEVEL_TOLERANCE_ATR}, "
          f"penetration={DEFAULT_MIN_PENETRATION_ATR}\n")

    print("=" * 92)
    print("CONDITION AVAILABILITY (I5 -- untestable is stated, never omitted)")
    print("=" * 92)
    print("  1. atr_percentile <= 0.20                COMPUTABLE")
    print("  2. confirmed_sweep                       COMPUTABLE")
    print("  3. macro_regime in {risk-on,transitional}  BLOCKED -- btc_dominance_20d_trend_pct")
    print("                                                      and btc_qqq_corr_20d not ingested")
    print("  4. mtf_alignment >= 3/4                  BLOCKED -- no derivation of per-timeframe")
    print("                                                      directional readings exists")
    print("\n  => the four-condition GATE NUMBER cannot be computed.")
    print("  => what follows is a strict UPPER BOUND on it.\n")

    df = build_asset_modeling_frame(load_frozen("BTC_USDT_1h.parquet"), keep_raw_atr=True)
    compressed = df[ATR_PCT_COL] <= COMPRESSION_MAX
    episodes = compression_episode_id(compressed)

    sw = sweep_flags(df)
    aligned = df.loc[sw.index]
    comp_a = compressed.loc[sw.index]

    print("CUMULATIVE CONDITION FILTER (BTC, full OHLCV window)")
    print("-" * 92)
    print(f"{'bars scanned':<46}{len(sw):>12,}")
    print(f"{'+ atr_percentile <= 0.20':<46}{int(comp_a.sum()):>12,}")
    both = comp_a & sw["confirmed"]
    print(f"{'+ confirmed_sweep':<46}{int(both.sum()):>12,}")
    print(f"{'+ macro_regime':<46}{'BLOCKED':>12}")
    print(f"{'+ mtf_alignment >= 3/4':<46}{'BLOCKED':>12}")
    print()
    print("EACH CONDITION INDIVIDUALLY")
    print("-" * 92)
    print(f"{'atr_percentile <= 0.20 alone':<46}{int(comp_a.sum()):>12,}"
          f"  ({comp_a.mean():.2%} of bars)")
    print(f"{'confirmed_sweep alone':<46}{int(sw['confirmed'].sum()):>12,}"
          f"  ({sw['confirmed'].mean():.2%} of bars)")

    events = pd.DataFrame({
        "direction": sw.loc[both[both].index, "direction"],
        "episode": episodes.reindex(both[both].index),
        "t0": aligned.loc[both[both].index, "t0"],
        "t1": aligned.loc[both[both].index, "t1"],
    })

    rows = [report_window(n, s, events, compressed) for n, s in WINDOWS.items()]
    table = pd.DataFrame(rows)

    print("\n" + "=" * 92)
    print("TWO-CONDITION EVENT COUNT -- UPPER BOUND ON THE GATE NUMBER")
    print("=" * 92)
    print(f"{'window':<32}{'comp bars':>11}{'raw':>8}{'structural':>12}"
          f"{'EFFECTIVE':>12}{'eff/raw':>9}")
    print("-" * 92)
    for _, r in table.iterrows():
        print(f"{r['window']:<32}{r['compressed_bars']:>11,}{r['raw_events']:>8,}"
              f"{r['after_structural']:>12,}{r['effective']:>12.1f}{r['eff_over_raw']:>9.3f}")
    print("-" * 92)

    gate_row = table.iloc[0]
    bound = gate_row["effective"]
    print(f"\nGate B threshold: effective >= {GATE_MIN_EFFECTIVE}")
    print(f"Two-condition upper bound on the flow window: {bound:.1f}")

    print("\n" + "=" * 92)
    if bound < GATE_MIN_EFFECTIVE:
        print("GATE B: FAILS.")
        print(f"The UPPER BOUND ({bound:.1f}) is already below {GATE_MIN_EFFECTIVE}. Conditions 3")
        print("and 4 can only remove events, so no value of the blocked conditions can")
        print("rescue the count. H1 fails on sample grounds. Decisive negative -> free")
        print("revision or terminate. STOP.")
    else:
        print("GATE B: INDETERMINATE -- NOT PASSED.")
        print(f"The upper bound ({bound:.1f}) clears {GATE_MIN_EFFECTIVE}, but two of four")
        print("conditions are unevaluated and both strictly reduce the count. The gate")
        print("number does not exist yet. Unblock conditions 3 and 4, then re-run.")
        print("This is NOT a pass and must not be recorded as one.")
    print("=" * 92)

    # ETH -- information only, explicitly outside the gate.
    print("\nETH/USDT -- OHLCV-only, INFORMATION ONLY, NOT part of the gate")
    print("-" * 92)
    eth = build_asset_modeling_frame(load_frozen("ETH_USDT_1h.parquet"), keep_raw_atr=True)
    eth_comp = eth[ATR_PCT_COL] <= COMPRESSION_MAX
    eth_ep = compression_episode_id(eth_comp)
    eth_sw = sweep_flags(eth)
    eth_both = eth_comp.loc[eth_sw.index] & eth_sw["confirmed"]
    eth_ev = pd.DataFrame({
        "direction": eth_sw.loc[eth_both[eth_both].index, "direction"],
        "episode": eth_ep.reindex(eth_both[eth_both].index),
        "t0": eth.loc[eth_both[eth_both].index, "t0"],
        "t1": eth.loc[eth_both[eth_both].index, "t1"],
    })
    eth_col = collapse_structural(eth_ev)
    print(f"  window {eth.index.min().date()} -> {eth.index.max().date()}  "
          f"({len(eth):,} bars, starts 2.6y INSIDE the flow window)")
    print(f"  raw {len(eth_ev):,}   structural {len(eth_col):,}   "
          f"effective {effective_count(eth_col):.1f}")
    print("  ETH has no funding or basis ingested: these events sit where Stage C and D")
    print("  cannot operate. Reported for information; does not enter the gate.")

    out = Path("reports/b5_event_count.csv")
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
