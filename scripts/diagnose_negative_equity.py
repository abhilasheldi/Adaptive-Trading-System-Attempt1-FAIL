"""
Diagnostic for the negative-equity finding (maxDD > 1.0) in the 2026-08-14
layer ablation. Reproduces ONE fold of the `all_layers` configuration and
reports exactly what the execution path did with position sizes, trades,
and the equity curve -- so the cause is measured rather than inferred.

Not a production tool; run manually.
    python -m scripts.diagnose_negative_equity
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting import metrics
from src.backtesting.costs import TransactionCostModel
from src.backtesting.phase2_strategy import Phase2Strategy
from src.backtesting.windows import generate_walk_forward_windows
from src.data_engine.crypto_fetcher import remove_duplicates_and_skew
from src.predictive_core.dataset import build_asset_modeling_frame, feature_columns

INIT_CASH = 10_000.0


def main() -> None:
    raw = pd.read_parquet("data/raw/BTC_USDT_1h.parquet")
    raw = remove_duplicates_and_skew(raw, "1h")
    raw = raw[raw.index >= pd.Timestamp("2022-08-01", tz="UTC")]
    df = build_asset_modeling_frame(raw, keep_raw_atr=True)
    fc = feature_columns(df)

    windows = generate_walk_forward_windows(
        df.index, train_size=pd.DateOffset(years=2), val_size=pd.DateOffset(months=6),
        test_size=pd.DateOffset(months=6), step=pd.DateOffset(months=3), min_folds=4,
    )
    # Fold 2 (index 1) is where the ablation's worst Sharpe (-30.7) landed.
    w = windows[1]
    train = df.loc[w.train_start : w.train_end]
    val = df.loc[w.val_start : w.val_end]
    test = df.loc[w.test_start : w.test_end]
    print(f"fold 2: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    strat = Phase2Strategy(
        fc, use_liquidity_trap=True, use_volatility_gate=True, use_ensemble=True,
        label_horizon=48, n_optuna_trials=2, early_stopping_rounds=10,
    )
    strat.fit(train, val)
    pos = strat.generate_positions(test)

    nz = pos[pos != 0.0]
    print("\n=== POSITION SERIES ===")
    print(f"  bars={len(pos)}  nonzero={len(nz)} ({100*len(nz)/len(pos):.1f}%)")
    print(f"  max |position| = {pos.abs().max():.6f}   (strategy cap = {strat.max_position})")
    print(f"  mean |nonzero| = {nz.abs().mean():.6f}")
    flips = int((np.sign(pos).diff().abs() == 2).sum())
    print(f"  direction flips (long<->short): {flips}")

    cm = TransactionCostModel()
    cm.fit(train)
    fees, slip = cm.per_bar_fees_and_slippage(test, pos)
    changed = pos.ne(pos.shift(1).fillna(0.0))
    order = pos.where(changed, other=np.nan)
    pf = vbt.Portfolio.from_orders(
        close=test["close"], size=order, size_type="targetpercent", direction="both",
        fees=fees, slippage=slip, freq="1h", init_cash=INIT_CASH,
    )
    tr = cm.apply_gap_cost_to_trades(pf.trades.records_readable, test.index)
    vbt_eq = pf.value()
    rep = metrics.equity_curve_from_trade_pnls(
        tr["net_pnl"], tr["Exit Timestamp"], test.index, INIT_CASH
    )
    notional = (tr["Size"].abs() * tr["Avg Entry Price"]).astype(float)

    print("\n=== TRADES ===")
    print(f"  n={len(tr)}")
    print(f"  notional: mean ${notional.mean():,.0f}  max ${notional.max():,.0f}")
    print(f"  net_pnl : mean ${tr['net_pnl'].mean():.2f}  sum ${tr['net_pnl'].sum():,.0f}")
    print(f"  worst single trade: ${tr['net_pnl'].min():,.2f}")
    print(f"  gap_cost total: ${tr['gap_cost'].sum():,.0f}")

    print("\n=== EQUITY ===")
    print(f"  vectorbt (sizing book): min ${vbt_eq.min():,.0f}  maxDD={metrics.max_drawdown(vbt_eq):.3f}")
    print(f"  reported (trade book) : min ${rep.min():,.0f}  maxDD={metrics.max_drawdown(rep):.3f}")
    print(f"  reported goes <= 0? {bool((rep <= 0).any())}")
    if (rep <= 0).any():
        first = int((rep <= 0).values.argmax())
        print(f"    first non-positive at bar {first}/{len(rep)} ({rep.index[first]})")
        print(f"    trades closed by then: {int((tr['Exit Timestamp'] <= rep.index[first]).sum())}")
    print(f"  peak equity: ${rep.cummax().max():,.0f}")


if __name__ == "__main__":
    main()
