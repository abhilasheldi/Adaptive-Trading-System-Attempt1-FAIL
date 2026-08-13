"""
Build Plan step 1D: runs the full walk-forward backtest on real BTC/ETH data
and reports the five Layer 14 research-phase kill criteria (MASTER_CONTEXT),
so the user can make the step 1E Go/No-Go call (an explicit human judgment
call per the Build Plan -- this script does not, and must not, decide it).

NOT the Tier 1 live-trading gate (a separate, stricter six-metric checklist
checked only at Phase 5, right before capital deployment).

All five Layer 14 criteria are always reported, structurally, even though
only three currently resolve to pass/fail on this codebase: Risk-On-vs-
aggregate performance and ensemble weight stability both require Phase 2
modules (Macro Regime Engine / Layer 3, linear Ensemble Meta-Model / Layer
7C) that don't exist yet, and report "not_evaluated" rather than being
silently omitted.

Usage:
    python scripts/run_walkforward_backtest.py --fast              # reduced Optuna budget, quick sanity pass
    python scripts/run_walkforward_backtest.py                     # full spec run (default Optuna budget)
    python scripts/run_walkforward_backtest.py --symbols BTC/USDT  # single asset
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.harness import WalkForwardBacktester
from src.backtesting.predictive_core_strategy import PredictiveCoreStrategy
from src.config import load_config
from src.data_engine.crypto_fetcher import fetch_ohlcv, remove_duplicates_and_skew, save_raw_data
from src.predictive_core.dataset import build_asset_modeling_frame, feature_columns


def _load_or_fetch_raw(symbol: str, timeframe: str, data_dir: Path) -> pd.DataFrame:
    """Same incremental-fetch-and-cache pattern as
    `crypto_fetcher._ingest_one`, scoped to exactly the (symbol, timeframe)
    this script needs -- avoids re-fetching full history from Binance on
    every run, and avoids fetching timeframes (e.g. 4h, if unused here) this
    script doesn't need."""
    filepath = data_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
    since_ts = None
    existing = pd.DataFrame()
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        if len(existing):
            since_ts = int(existing.index.max().timestamp() * 1000) + 1

    new_df = fetch_ohlcv(symbol, timeframe, _binance_exchange(), since_ts=since_ts)
    if not new_df.empty:
        new_df = remove_duplicates_and_skew(new_df, timeframe)
        save_raw_data(new_df, filepath)

    return pd.read_parquet(filepath) if filepath.exists() else existing


_EXCHANGE = None


def _binance_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        import ccxt

        _EXCHANGE = ccxt.binance({"enableRateLimit": True})
    return _EXCHANGE


def _build_cost_model_kwargs(costs_cfg: dict) -> dict:
    """Maps config.yaml's `backtesting.costs` keys onto
    `TransactionCostModel.__init__`'s actual parameter names -- these are
    NOT the same names (e.g. config's `taker_fee_bps` is basis points,
    the constructor's `taker_fee_frac` is a fraction; a naive `**costs_cfg`
    unpack would both use the wrong keyword names and be off by a factor of
    10,000 on the fee). This mapping was never written before now --
    config.yaml's `backtesting:` section was added when the harness was
    built but explicitly documented as "not yet wired to a config loader.\""""
    return {
        "taker_fee_frac": costs_cfg["taker_fee_bps"] / 10_000.0,
        "spread_percentile": costs_cfg["spread_percentile"],
        "spread_proxy_fraction_of_range": costs_cfg["spread_proxy_fraction_of_range"],
        "slippage_coefficient": costs_cfg["slippage_coefficient"],
        "overnight_start_utc_hour": costs_cfg["overnight_start_utc_hour"],
        "overnight_end_utc_hour": costs_cfg["overnight_end_utc_hour"],
        "gap_threshold_mult": costs_cfg["gap_threshold_mult"],
    }


def _build_kill_criteria_kwargs(kc_cfg: dict) -> dict:
    """Same mapping problem as costs: config.yaml's kill_criteria keys
    (oos_sharpe_min, ...) don't match WalkForwardBacktester's constructor
    parameter names (oos_sharpe_threshold, ...)."""
    return {
        "oos_sharpe_threshold": kc_cfg["oos_sharpe_min"],
        "oos_profit_factor_threshold": kc_cfg["oos_profit_factor_min"],
        "oos_max_dd_threshold": kc_cfg["oos_max_drawdown_max"],
        "is_oos_ratio_threshold": kc_cfg["is_oos_sharpe_ratio_min"],
        "ensemble_weight_delta_threshold": kc_cfg["ensemble_instability_weight_delta"],
        "ensemble_consecutive_fold_threshold": kc_cfg["ensemble_instability_consecutive_folds"],
    }


def _format_fold_table(report) -> str:
    """Per-fold IS/OOS breakdown underneath the aggregate five-criteria
    verdict -- `BacktestReport.summary()` gives the aggregate numbers and
    the terse overfit-diagnostic line, but not this level of per-fold
    detail, which is what actually explains *why* a criterion passed or
    failed rather than just stating that it did."""
    lines = ["", "--- Per-fold detail (in-sample vs. out-of-sample) ---"]
    header = (
        f"  {'fold':>4} {'segment':>13} {'train/val/test window':<58} "
        f"{'sharpe':>8} {'PF':>8} {'maxDD':>8} {'EV/trade':>10} {'n_trades':>9}"
    )
    lines.append(header)
    for f in report.folds:
        w = f.window
        window_str = (
            f"{w.train_start.date()}..{w.val_start.date()}.."
            f"{w.test_start.date()}..{w.test_end.date()}"
        )
        for seg in (f.in_sample, f.out_of_sample):
            lines.append(
                f"  {f.fold:>4} {seg.segment:>13} {window_str:<58} "
                f"{seg.sharpe:>8.3f} {seg.profit_factor:>8.3f} {seg.max_drawdown:>8.1%} "
                f"{seg.ev_per_trade:>10.4f} {seg.n_trades:>9}"
            )
    return "\n".join(lines)


def _format_direction_skill_table(report) -> str:
    """
    Per-fold Component B direction-classifier skill diagnostics -- NOT part
    of the five Layer 14 criteria, but essential context for interpreting
    them: with MASTER_CONTEXT's own triple-barrier geometry producing a
    heavily skewed real-data label split (observed ~65% down / 34% up / <1%
    neutral on BTC), a classifier that just parrots the majority class can
    still be well-calibrated (see skill_diagnostics.py) while contributing
    zero real signal to the trades the Layer 14 numbers above are computed
    from. Two things to actually look at:
      - "up recall": near 0 means the classifier essentially never calls
        "up" with any confidence -- a majority-parrot signature.
      - "logloss skill": ~0 means no better than guessing the train fold's
        base rates; negative is worse than that trivial baseline.

    IMPORTANT: these numbers are evaluated on `PredictiveCoreStrategy`'s
    val_purged partition, NOT genuine out-of-sample data -- see
    `PredictiveCoreStrategy.diagnostics()`'s docstring. They can only ever
    look as good as, or better than, real generalization, never worse. Use
    them to catch an obviously-broken/parroting classifier (which shows up
    just as badly here as it would OOS), not to judge how good a
    borderline-skillful classifier really is.
    """
    lines = ["", "--- Per-fold Component B direction-classifier skill diagnostics (val-partition, NOT OOS -- see script docstring) ---",
              "    (reliability [calibration] alone cannot catch a majority-class parrot -- see script docstring)"]
    header = (
        f"  {'fold':>4} {'logloss skill':>14} {'down P/R':>14} {'neutral P/R':>14} {'up P/R':>14}"
    )
    lines.append(header)
    for f in report.folds:
        diag = f.diagnostics
        if not diag:
            lines.append(f"  {f.fold:>4}  (no diagnostics reported by this strategy)")
            continue
        by_class = {d["class_label"]: d for d in diag["direction_val_partition_discrimination"]}
        skill = diag["direction_val_partition_logloss_skill_score"]
        pr = {
            k: f"{by_class[k]['precision']:.2f}/{by_class[k]['recall']:.2f}" if k in by_class else "n/a"
            for k in ("down", "neutral", "up")
        }
        lines.append(
            f"  {f.fold:>4} {skill:>14.3f} {pr['down']:>14} {pr['neutral']:>14} {pr['up']:>14}"
        )
    return "\n".join(lines)


def run_for_asset(
    symbol: str, timeframe: str, config: dict, n_optuna_trials: int, early_stopping_rounds: int, data_dir: Path,
) -> str:
    print(f"\n[{symbol}] fetching/updating raw {timeframe} data...", flush=True)
    raw = _load_or_fetch_raw(symbol, timeframe, data_dir)
    print(f"[{symbol}] raw: {len(raw)} candles, {raw.index.min()} .. {raw.index.max()}", flush=True)

    tb_cfg = config["predictive_core"]["triple_barrier"]
    modeling = build_asset_modeling_frame(
        raw,
        tp_mult=tb_cfg["tp_mult"],
        sl_mult=tb_cfg["sl_mult"],
        max_candles=tb_cfg["max_candles"],
    )
    fcols = feature_columns(modeling)
    print(f"[{symbol}] modeling frame: {len(modeling)} rows, {len(fcols)} feature columns", flush=True)

    label_horizon = tb_cfg["max_candles"]

    def strategy_factory():
        return PredictiveCoreStrategy(
            feature_cols=fcols,
            label_horizon=label_horizon,
            n_optuna_trials=n_optuna_trials,
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
        )

    win_cfg = config["backtesting"]["windows"]
    backtester = WalkForwardBacktester(
        strategy_factory=strategy_factory,
        timeframe=timeframe,
        train_size=pd.DateOffset(years=win_cfg["min_train_years"]),
        val_size=pd.DateOffset(months=win_cfg["val_months"]),
        test_size=pd.DateOffset(months=win_cfg["test_months"]),
        step=pd.DateOffset(months=win_cfg["step_months"]),
        min_folds=win_cfg["min_folds"],
        cost_model_kwargs=_build_cost_model_kwargs(config["backtesting"]["costs"]),
        # Exploratory run: report every fold regardless of outcome instead
        # of halting at the first failed Layer 14 criterion, so the user
        # sees the full picture to make the 1E judgment call from.
        raise_on_kill_criteria_failure=False,
        **_build_kill_criteria_kwargs(config["backtesting"]["kill_criteria"]),
    )

    print(f"[{symbol}] running walk-forward backtest (n_optuna_trials={n_optuna_trials})...", flush=True)
    t0 = time.perf_counter()
    report = backtester.run(modeling)
    elapsed = time.perf_counter() - t0
    print(f"[{symbol}] done in {elapsed / 60:.1f} min, {len(report.folds)} fold(s).", flush=True)

    header = f"{'=' * 78}\n{symbol} ({timeframe}) -- Build Plan step 1D walk-forward backtest\n{'=' * 78}"
    return "\n".join([
        header, report.summary(), _format_fold_table(report), _format_direction_skill_table(report),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=None, help="Overrides config.yaml crypto.symbols")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--fast", action="store_true", help="Reduced Optuna budget for a quick sanity pass")
    parser.add_argument("--n-optuna-trials", type=int, default=None, help="Overrides --fast/default budget")
    parser.add_argument("--output", default="reports/walkforward_1D_report.md")
    args = parser.parse_args()

    config = load_config("config.yaml")
    symbols: List[str] = args.symbols or config["crypto"]["symbols"]
    data_dir = Path(config["data_dir"])

    if args.n_optuna_trials is not None:
        n_optuna_trials, early_stopping_rounds = args.n_optuna_trials, 20
    elif args.fast:
        n_optuna_trials, early_stopping_rounds = 5, 10
    else:
        n_optuna_trials, early_stopping_rounds = 15, 50  # PredictiveCoreStrategy's own spec defaults

    sections = [
        run_for_asset(symbol, args.timeframe, config, n_optuna_trials, early_stopping_rounds, data_dir)
        for symbol in symbols
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
