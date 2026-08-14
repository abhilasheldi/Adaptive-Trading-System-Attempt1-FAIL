"""
Stage A.4 verification -- does the sizing book and the reported book agree?

=========================================================================
WHAT THIS CAN AND CANNOT VERIFY (stated, not omitted -- I5)
=========================================================================
BUILD_PLAN_v2's Gate A asks to verify the harness fixes "by reproducing
fold 2 of `all_layers` and confirming sizing-book and reported maxDD now
agree."

THAT EXACT REPRODUCTION IS NOT POSSIBLE. `all_layers` was produced by
`Phase2Strategy` driven by `scripts/run_layer_ablation.py`, using
`HistoricalAnalogEngine` and `EnsembleMetaModel`. All four were deleted at
Stage 0 per MASTER_CONTEXT_v2 section 8. `reports/layer_ablation_v2.json`
retains that fold's Sharpe (-0.162) and trade count (468) but no positions,
no trades, and no equity curve -- nothing sufficient to rebuild it.

What IS verified here: the DEFECT, on real BTC data, at fold 2's geometry,
using a deterministic strategy so the run is reproducible. S12 is a property
of the harness's accounting, not of the strategy that fed it -- any strategy
holding positions across gap-prone sessions exhibits it. The numbers below
are therefore a real measurement of the mechanism, and are NOT a restatement
of the -0.162 fold.

Run: python scripts/verify_single_book.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting import metrics  # noqa: E402
from src.backtesting.costs import TransactionCostModel  # noqa: E402
from src.backtesting.harness import WalkForwardBacktester  # noqa: E402
from src.data_engine.crypto_fetcher import remove_duplicates_and_skew  # noqa: E402

INIT_CASH = 10_000.0
TIMEFRAME = "1h"


def deterministic_positions(df: pd.DataFrame) -> pd.Series:
    """A fixed, non-learned rule. Trades often and holds across weekends, so
    the gap-cost channel is genuinely exercised. No Optuna, no randomness --
    the run reproduces exactly."""
    fast = df["close"].rolling(12).mean()
    slow = df["close"].rolling(72).mean()
    raw = np.sign(fast - slow).fillna(0.0)
    return (raw * 0.02).rename(None)          # 2% target exposure


def run_books(segment_df, positions, cost_model):
    """Compute the OLD two books and the NEW single book on identical input."""
    positions = positions.reindex(segment_df.index).fillna(0.0)
    fees, slippage = cost_model.per_bar_fees_and_slippage(segment_df, positions)
    changed = positions.ne(positions.shift(1).fillna(0.0))
    order_size = positions.where(changed, other=np.nan)

    # ---- OLD path, pass 1: the book vectorbt sized off ------------------ #
    pf_a = vbt.Portfolio.from_orders(
        close=segment_df["close"], size=order_size, size_type="targetpercent",
        direction="both", fees=fees, slippage=slippage,
        freq=TIMEFRAME, init_cash=INIT_CASH,
    )
    sizing_book = pd.Series(np.asarray(pf_a.value()).ravel(), index=segment_df.index)
    trades = cost_model.apply_gap_cost_to_trades(pf_a.trades.records_readable, segment_df.index)

    # ---- OLD path: the SEPARATE reported book, gap cost deducted -------- #
    reported_book_old = metrics.equity_curve_from_trade_pnls(
        trades["net_pnl"], trades["Exit Timestamp"], segment_df.index, INIT_CASH,
    )

    # ---- NEW path: gap cost folded into the fees vectorbt consumes ------ #
    gap_fee = WalkForwardBacktester._gap_cost_as_exit_fee(trades, segment_df, pf_a)
    if gap_fee is None:
        pf_b, note = pf_a, "no gap cost in this segment"
    else:
        pf_b, note = vbt.Portfolio.from_orders(
            close=segment_df["close"], size=order_size, size_type="targetpercent",
            direction="both", fees=fees + gap_fee, slippage=slippage,
            freq=TIMEFRAME, init_cash=INIT_CASH,
        ), "gap cost inside the sizing book"
    single_book = pd.Series(np.asarray(pf_b.value()).ravel(), index=segment_df.index)

    return {
        "sizing_book_old": sizing_book,
        "reported_book_old": reported_book_old,
        "single_book_new": single_book,
        "gap_cost_total": float(trades["gap_cost"].sum()),
        "n_trades": len(trades),
        "note": note,
    }


def main() -> None:
    path = Path("data/raw/BTC_USDT_1h.parquet")
    if not path.exists():
        print(f"!! {path} not found -- cannot verify. STATED, not skipped silently.")
        return

    raw = remove_duplicates_and_skew(pd.read_parquet(path), timeframe=TIMEFRAME)

    # Fold 2 geometry from the ablation: 2y train / 6m val / 6m test, step 3m,
    # over the same 2022-08-01 start the ablation used. Fold 2 -> test window
    # begins 2 steps (6 months) after fold 0's.
    start = pd.Timestamp("2022-08-01", tz="UTC")
    train_end = start + pd.DateOffset(years=2) + pd.DateOffset(months=6)
    test_start = train_end + pd.DateOffset(months=6)
    test_end = test_start + pd.DateOffset(months=6)

    train_df = raw.loc[start:train_end]
    test_df = raw.loc[test_start:test_end]
    print(f"train {len(train_df):,} bars  test {len(test_df):,} bars "
          f"({test_df.index.min().date()} -> {test_df.index.max().date()})")

    cost_model = TransactionCostModel().fit(train_df)
    positions = deterministic_positions(test_df)
    r = run_books(test_df, positions, cost_model)

    dd = metrics.max_drawdown
    dd_sizing = dd(r["sizing_book_old"])
    dd_reported = dd(r["reported_book_old"])
    dd_single = dd(r["single_book_new"])

    print(f"\ntrades {r['n_trades']:,}   total gap cost ${r['gap_cost_total']:,.2f}   ({r['note']})")

    print("\n" + "=" * 74)
    print("BEFORE (S12) -- two books")
    print("=" * 74)
    print(f"  sizing book maxDD (what vectorbt sized off) : {dd_sizing:.6f}")
    print(f"  reported book maxDD (what was published)    : {dd_reported:.6f}")
    if dd_sizing > 0:
        print(f"  DIVERGENCE                                  : {dd_reported / dd_sizing:.2f}x")
    print(f"  final equity: sizing ${r['sizing_book_old'].iloc[-1]:,.2f}   "
          f"reported ${r['reported_book_old'].iloc[-1]:,.2f}   "
          f"gap ${r['sizing_book_old'].iloc[-1] - r['reported_book_old'].iloc[-1]:,.2f}")

    print("\n" + "=" * 74)
    print("AFTER (I7) -- one book")
    print("=" * 74)
    print(f"  single book maxDD (sized off AND reported)  : {dd_single:.6f}")
    print(f"  final equity                                : ${r['single_book_new'].iloc[-1]:,.2f}")
    print("\n  Sizing and reporting now read the SAME series, so the two numbers")
    print("  are one number by construction -- there is no second book left to")
    print("  diverge from. The residual vs the old reported book is the timing")
    print("  choice: gap cost is charged at exit rather than deducted afterwards.")
    print(f"  old reported maxDD {dd_reported:.6f} vs new {dd_single:.6f} "
          f"(diff {abs(dd_reported - dd_single):.6f})")

    print("\n  maxDD <= 1.0 on every book (S11 floor):",
          all(x <= 1.0 or np.isnan(x) for x in (dd_sizing, dd_reported, dd_single)))

    # ------------------------------------------------------------------ #
    # SENSITIVITY -- the divergence scales with gap cost, which is exactly
    # why S12's measured 20x is not reproducible today.
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("WHY THE DIVERGENCE IS SMALL NOW: it scales with gap cost, and the")
    print("gap MODEL was already fixed (session counting + adverse fraction +")
    print("notional cap, 2026-08-14). Sweeping the adverse fraction shows the")
    print("mechanism is intact and simply not loaded at current parameters.")
    print("=" * 74)
    print(f"{'adverse_frac':>13}{'gap cost $':>13}{'sizing maxDD':>15}"
          f"{'reported maxDD':>16}{'divergence':>12}")
    print("-" * 74)
    for frac in (0.5, 2.0, 10.0, 40.0):
        cm = TransactionCostModel(gap_adverse_fraction=min(frac, 1.0)).fit(train_df)
        if frac > 1.0:
            # Beyond the legal [0,1] range, scale the fitted magnitudes
            # instead -- same effect on cost, and it keeps the constructor
            # validation intact rather than bypassing it.
            p = cm.params
            object.__setattr__(p, "gap_magnitude_weekend_frac",
                               p.gap_magnitude_weekend_frac * frac)
            object.__setattr__(p, "gap_magnitude_overnight_frac",
                               p.gap_magnitude_overnight_frac * frac)
        rr = run_books(test_df, positions, cm)
        ds, dr = dd(rr["sizing_book_old"]), dd(rr["reported_book_old"])
        div = f"{dr / ds:.2f}x" if ds > 0 else "n/a"
        print(f"{frac:>13.1f}{rr['gap_cost_total']:>13,.2f}{ds:>15.6f}{dr:>16.6f}{div:>12}")
    print("\n  The two books separate as gap cost grows. At the ORIGINAL, unbounded")
    print("  per-bar gap model -- where gap cost was 96% of fold 2's loss -- that")
    print("  separation is what produced the 20x figure recorded as S12.")


if __name__ == "__main__":
    main()
