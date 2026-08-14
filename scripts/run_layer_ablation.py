"""
Walk-forward LAYER ABLATION: run the 5-fold backtest once per configuration
and report each Phase 2 layer's marginal effect on out-of-sample Sharpe,
so a layer that makes things WORSE is flagged rather than silently kept.

    python -m scripts.run_layer_ablation --asset BTC/USDT --start 2022-08-01

=========================================================================
WHY THE MODELS ARE FIT ONCE AND SHARED ACROSS VARIANTS
=========================================================================
Every ablation variant differs ONLY in post-prediction logic: which
conditioning layers gate or blend Component A/B's output. None of them
changes how Component A/B are trained. If each variant re-trained the
models, two things would go wrong:

  1. ATTRIBUTION. LightGBM + Optuna are not perfectly reproducible across
     processes in every environment, so part of any Sharpe delta could be
     training noise rather than the layer under test. Sharing one fitted
     model per fold makes the comparison exact: identical Component A/B
     predictions, only the layer logic differs.
  2. RUNTIME. A full 5-fold fit at the spec Optuna budget takes ~45-50
     minutes. Re-fitting for each of 4+ variants would take hours.

`_SharedFitStrategy` below implements this with an explicit cache keyed on
the fold's train/val index range. The cache is only ever a HIT when the
harness hands over byte-identical partitions, which it does because every
variant runs over the same data with the same window parameters -- the
key includes the exact first/last timestamps and row counts of both
partitions, so a genuinely different fold can never collide.

=========================================================================
WHAT THIS DOES AND DOES NOT EVALUATE
=========================================================================
This is a **Layer 14 research-phase** check (continue-vs-revise the
hypothesis), NOT the Tier 1 live-trading gate. Those are separate,
non-interchangeable threshold sets and this script only ever reports the
former. Tier 1 (profit factor > 1.5, Sharpe > 1.0, Sortino > 1.5, Calmar
> 0.5, max DD < 20%, EV/trade > 0) is a Phase 5 check against real
out-of-sample results and is not evaluated anywhere in this codebase.

Trade counts are reported alongside Sharpe for every variant, because
Layers 5/6 act as FILTERS: a layer can raise Sharpe purely by cutting the
sample, and MASTER_CONTEXT's hypothesis requires a minimum 200-trade
out-of-sample sample. A Sharpe improvement bought with a collapsed sample
is not an improvement.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting.harness import Layer14KillCriteriaFailed, WalkForwardBacktester
from src.backtesting.phase2_strategy import Phase2Strategy
from src.data_engine.crypto_fetcher import remove_duplicates_and_skew
from src.predictive_core.dataset import build_asset_modeling_frame, feature_columns

# Each variant is (label, use_liquidity_trap, use_volatility_gate, use_ensemble).
# "baseline" is the Phase 1 reference: all Phase 2 layers off, behaviourally
# identical to PredictiveCoreStrategy (asserted in tests).
VARIANTS: List[Tuple[str, bool, bool, bool]] = [
    ("baseline_phase1", False, False, False),
    ("all_layers", True, True, True),
    ("no_liquidity_trap_L5", False, True, True),
    ("no_volatility_gate_L6", True, False, True),
    ("no_ensemble_L7C", True, True, False),
]


class _SharedFitStrategy(Phase2Strategy):
    """`Phase2Strategy` that reuses an already-fitted Component A/B from a
    shared cache when the same fold partitions come round again.

    The cache stores everything `fit()` produces that does not depend on the
    layer flags. The ensemble (Layer 7C) is deliberately NOT cached: its
    weights are fit per configuration on the validation partition and must
    reflect that configuration's own signal set.
    """

    def __init__(
        self, *args,
        shared_cache: Optional[dict] = None,
        shared_ensemble_holder: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._shared_cache = shared_cache if shared_cache is not None else {}
        # `WalkForwardBacktester` builds a NEW strategy per fold via its
        # `strategy_factory`, so an EnsembleMetaModel held on `self` would
        # start with an empty weight history every fold -- and the Adaptive
        # doc's cross-fold 0.15 stability rule (and its equal-weight
        # fallback) could never fire. This holder is shared by every
        # strategy instance a single variant's run creates, so the history
        # accumulates across folds as the spec requires. It is per-VARIANT,
        # not global: each variant fits its own weights on its own signal
        # set and must not inherit another configuration's history.
        self._shared_ensemble_holder = (
            shared_ensemble_holder if shared_ensemble_holder is not None else {}
        )
        # Identifies which fold's models are currently loaded, so the
        # prediction cache can never serve one fold's predictions to another.
        self._current_fold_key: Optional[tuple] = None

    @property
    def _ensemble(self):
        return self._shared_ensemble_holder.get("model")

    @_ensemble.setter
    def _ensemble(self, value) -> None:
        if value is not None:
            self._shared_ensemble_holder["model"] = value

    @staticmethod
    def _fold_key(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
        return (
            train_df.index[0], train_df.index[-1], len(train_df),
            val_df.index[0], val_df.index[-1], len(val_df),
        )

    @staticmethod
    def _partition_key(df: pd.DataFrame) -> tuple:
        return (df.index[0], df.index[-1], len(df))

    # --- Prediction caches -------------------------------------------- #
    # Component A/B predictions and Layer 5 readings are VARIANT-INVARIANT:
    # they depend only on the fitted models (shared per fold) and the data,
    # never on which conditioning layers are switched on. Recomputing them
    # per variant was the dominant cost -- Component A's `predict()` runs
    # per row (~11ms at realistic reference-set size), so a single 5-fold
    # variant spends most of its wall time re-deriving numbers identical to
    # the previous variant's. Caching them makes the ablation's marginal
    # cost per extra variant small, and is safe precisely BECAUSE they are
    # invariant -- the cache key carries the fold (hence the models) and the
    # exact partition, so a different fold or partition can never hit.

    def _component_predictions(self, df: pd.DataFrame):
        key = ("preds", self._current_fold_key, self._partition_key(df))
        cached = self._shared_cache.get(key)
        if cached is None:
            cached = super()._component_predictions(df)
            self._shared_cache[key] = cached
        return cached

    def _liquidity_trap_series(self, df: pd.DataFrame):
        if not self.use_liquidity_trap:
            return [None] * len(df)
        key = ("trap", self._partition_key(df))
        cached = self._shared_cache.get(key)
        if cached is None:
            cached = super()._liquidity_trap_series(df)
            self._shared_cache[key] = cached
        return cached

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        key = self._fold_key(train_df, val_df)
        self._current_fold_key = key
        cached = self._shared_cache.get(key)
        if cached is None:
            super().fit(train_df, val_df)
            self._shared_cache[key] = {
                "analog_engine": self._analog_engine,
                "distance_threshold": self._distance_threshold,
                "direction_model": self._direction_model,
                "magnitude_models": self._magnitude_models,
                "direction_result": self._direction_result,
            }
            return

        # Restore the shared Component A/B, then fit only this variant's own
        # ensemble on the validation partition.
        self._analog_engine = cached["analog_engine"]
        self._distance_threshold = cached["distance_threshold"]
        self._direction_model = cached["direction_model"]
        self._magnitude_models = cached["magnitude_models"]
        self._direction_result = cached["direction_result"]
        self._fitted = True

        if self.use_ensemble:
            val_purged = (
                val_df.iloc[: -self.label_horizon] if len(val_df) > self.label_horizon else val_df.iloc[:0]
            )
            from src.predictive_core.ensemble import EnsembleMetaModel

            if self._ensemble is None:
                self._ensemble = EnsembleMetaModel()
            fit_result = self._ensemble.fit(
                self._build_signal_sets(val_purged), val_purged["direction"].to_numpy()
            )
            self._ensemble_weights = dict(fit_result.weights)


def _load_frame(asset: str, start: Optional[str], timeframe: str) -> pd.DataFrame:
    cache = Path("data/raw") / f"{asset.replace('/', '_')}_{timeframe}.parquet"
    if not cache.exists():
        raise SystemExit(f"No cached data at {cache}. Fetch it first (see scripts/run_walkforward_backtest.py).")
    raw = pd.read_parquet(cache)
    raw = remove_duplicates_and_skew(raw, timeframe)
    if start:
        raw = raw[raw.index >= pd.Timestamp(start, tz="UTC")]
    return build_asset_modeling_frame(raw, keep_raw_atr=True)


def _run_variant(
    label: str, flags: Tuple[bool, bool, bool], df: pd.DataFrame,
    feature_cols: List[str], shared_cache: dict, args,
) -> Dict:
    use_lt, use_vol, use_ens = flags
    ensemble_holder: dict = {}  # per-variant, shared across that variant's folds

    def _factory():
        return _SharedFitStrategy(
            feature_cols,
            use_liquidity_trap=use_lt, use_volatility_gate=use_vol, use_ensemble=use_ens,
            label_horizon=args.label_horizon,
            n_optuna_trials=args.optuna_trials,
            early_stopping_rounds=args.early_stopping_rounds,
            shared_cache=shared_cache,
            shared_ensemble_holder=ensemble_holder,
        )

    backtester = WalkForwardBacktester(
        strategy_factory=_factory,
        timeframe=args.timeframe,
        min_folds=args.min_folds,
        # Never halt mid-ablation: a variant failing Layer 14 is a RESULT to
        # report and compare, not a reason to abort the remaining variants.
        # The criteria are still evaluated and reported in full below.
        raise_on_kill_criteria_failure=False,
    )
    t0 = time.time()
    try:
        report = backtester.run(df)
    except Layer14KillCriteriaFailed as exc:  # defensive: flag is off, so unreachable
        report = exc.report
    elapsed = time.time() - t0

    oos_trades = int(sum(f.out_of_sample.n_trades for f in report.folds))
    return {
        "variant": label,
        "layers": {"liquidity_trap_L5": use_lt, "volatility_gate_L6": use_vol, "ensemble_L7C": use_ens},
        "oos_sharpe": report.aggregate_oos_sharpe,
        "oos_profit_factor": report.aggregate_oos_profit_factor,
        "oos_max_drawdown": report.aggregate_oos_max_drawdown,
        "oos_trades": oos_trades,
        "per_fold_oos_sharpe": [f.out_of_sample.sharpe for f in report.folds],
        "per_fold_oos_trades": [f.out_of_sample.n_trades for f in report.folds],
        "equity_destroyed": bool(report.any_equity_destroyed),
        "under_powered_folds": list(report.under_powered_folds),
        "n_folds_in_aggregate_sharpe": report.n_folds_in_aggregate_sharpe,
        "layer14": report.kill_criteria.as_report(),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--label-horizon", type=int, default=48)
    ap.add_argument("--optuna-trials", type=int, default=15)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--min-folds", type=int, default=4,
                    help="Minimum complete walk-forward folds (MASTER_CONTEXT spec: 4).")
    ap.add_argument("--out", default="reports/layer_ablation.json")
    ap.add_argument("--only", nargs="*", help="Run only these variant labels.")
    args = ap.parse_args()

    df = _load_frame(args.asset, args.start, args.timeframe)
    feature_cols = feature_columns(df)
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features, "
          f"{df.index[0]} -> {df.index[-1]}", flush=True)

    variants = [v for v in VARIANTS if not args.only or v[0] in args.only]
    shared_cache: dict = {}
    results: List[Dict] = []
    for label, *flags in variants:
        print(f"\n=== {label} (L5={flags[0]} L6={flags[1]} L7C={flags[2]}) ===", flush=True)
        res = _run_variant(label, tuple(flags), df, feature_cols, shared_cache, args)
        results.append(res)
        print(f"  OOS Sharpe {res['oos_sharpe']:.3f} | PF {res['oos_profit_factor']:.3f} "
              f"| maxDD {res['oos_max_drawdown']:.3f} | trades {res['oos_trades']} "
              f"| {res['elapsed_s']}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"asset": args.asset, "start": args.start, "results": results}, indent=2, default=str))
    print(f"\nWrote {out}")
    print(_format_comparison(results))


# A variant whose equity curve reached zero is DISQUALIFIED: its Sharpe is
# not a worse number, it is a meaningless one (pct_change across a zero
# crossing). Ranking on it is what made the 2026-08-14 run flag Layer 6
# (which touches 0.174% of bars, so it cannot genuinely move Sharpe) while
# calling Layer 7C -- the layer that actually destroyed the account -- a
# help. Solvency is therefore checked BEFORE Sharpe is used for anything.
MAX_DD_RUIN = 1.0  # maxDD >= 1.0 means equity reached zero or below


def _is_disqualified(r: Dict) -> bool:
    dd = r.get("oos_max_drawdown")
    return bool(r.get("equity_destroyed")) or (dd is not None and not math.isnan(dd) and dd >= MAX_DD_RUIN)


def _fmt(value, spec: str = ">11.3f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{'n/a':>{spec.lstrip('>').split('.')[0]}}"
    return f"{value:{spec}}"


def _format_comparison(results: List[Dict]) -> str:
    by_label = {r["variant"]: r for r in results}
    lines = ["", "=" * 78, "LAYER ABLATION -- Layer 14 research criteria (NOT the Tier 1 live gate)", "=" * 78]
    lines.append(f"{'variant':<24} {'OOS Sharpe':>11} {'PF':>7} {'maxDD':>7} {'trades':>8}  status")
    for r in results:
        dq = _is_disqualified(r)
        status = "DISQUALIFIED (equity destroyed)" if dq else "ok"
        under = r.get("under_powered_folds") or []
        if under and not dq:
            status = f"under-powered fold(s) {under} excluded from Sharpe"
        sharpe_cell = "        n/a" if dq else _fmt(r["oos_sharpe"])
        lines.append(
            f"{r['variant']:<24} {sharpe_cell} {_fmt(r['oos_profit_factor'], '>7.3f')} "
            f"{_fmt(r['oos_max_drawdown'], '>7.3f')} {r['oos_trades']:>8d}  {status}"
        )

    disqualified = [r["variant"] for r in results if _is_disqualified(r)]
    if disqualified:
        lines += ["", (
            "*** DISQUALIFIED (equity reached zero or below): " + ", ".join(disqualified) + ". "
            "Their Sharpe is arithmetically meaningless and is NOT reported or ranked. "
            "Read max drawdown and profit factor instead. ***"
        )]

    baseline = by_label.get("baseline_phase1")
    full = by_label.get("all_layers")
    if baseline and full:
        if _is_disqualified(full) or _is_disqualified(baseline):
            lines += ["", (
                "Full stack vs Phase 1 baseline: NOT COMPARABLE on Sharpe (a variant was "
                f"disqualified). maxDD {baseline['oos_max_drawdown']:.3f} -> "
                f"{full['oos_max_drawdown']:.3f}, PF {baseline['oos_profit_factor']:.3f} -> "
                f"{full['oos_profit_factor']:.3f}."
            )]
        else:
            lines += ["", (
                f"Full stack vs Phase 1 baseline: "
                f"{full['oos_sharpe'] - baseline['oos_sharpe']:+.3f} Sharpe "
                f"({baseline['oos_sharpe']:.3f} -> {full['oos_sharpe']:.3f})"
            )]

    # Marginal contribution: full stack MINUS the leave-one-out variant.
    # Evaluated on maxDD and PF FIRST -- a layer that ruins the account is
    # degrading regardless of what Sharpe says -- and only then on Sharpe,
    # and only when both sides stayed solvent.
    lines += ["", "Marginal contribution of each layer (full stack minus leave-one-out):"]
    flagged: List[str] = []
    for layer, variant_label in (
        ("Layer 5 liquidity trap", "no_liquidity_trap_L5"),
        ("Layer 6 volatility gate", "no_volatility_gate_L6"),
        ("Layer 7C ensemble", "no_ensemble_L7C"),
    ):
        loo = by_label.get(variant_label)
        if not (full and loo):
            continue

        reasons: List[str] = []
        # 1. Solvency: does ADDING this layer destroy an otherwise-solvent run?
        if _is_disqualified(full) and not _is_disqualified(loo):
            reasons.append("adding it destroys the account")
        # 2. Profit factor.
        pf_delta = full["oos_profit_factor"] - loo["oos_profit_factor"]
        if not math.isnan(pf_delta) and pf_delta < 0:
            reasons.append(f"profit factor {loo['oos_profit_factor']:.3f} -> {full['oos_profit_factor']:.3f}")
        # 3. Max drawdown (higher is worse).
        dd_delta = full["oos_max_drawdown"] - loo["oos_max_drawdown"]
        if not math.isnan(dd_delta) and dd_delta > 0:
            reasons.append(f"max drawdown {loo['oos_max_drawdown']:.3f} -> {full['oos_max_drawdown']:.3f}")

        # 4. Sharpe -- ONLY when both sides are solvent.
        if _is_disqualified(full) or _is_disqualified(loo):
            sharpe_part = "Sharpe n/a (disqualified)"
        else:
            delta = full["oos_sharpe"] - loo["oos_sharpe"]
            sharpe_part = f"{delta:+.3f} Sharpe"
            if not math.isnan(delta) and delta < 0:
                reasons.append("lowers Sharpe")

        verdict = "DEGRADES -- FLAG" if reasons else "no degradation detected"
        lines.append(
            f"  {layer:<26} {sharpe_part:<26} [{verdict}]   "
            f"trades {loo['oos_trades']} -> {full['oos_trades']}"
        )
        if reasons:
            lines.append(f"      reasons: {'; '.join(reasons)}")
            flagged.append(layer)

    lines += ["", (
        "FLAGGED (degrade the run on solvency/PF/maxDD/Sharpe -- do not keep silently): "
        + (", ".join(flagged) if flagged else "none")
    )]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
