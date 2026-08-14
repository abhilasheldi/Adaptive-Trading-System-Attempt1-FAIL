"""
Stage A.5 -- backfill `trials.jsonl` from surviving evidence, and establish
a DOCUMENTED LOWER BOUND on how many configurations this project has
already evaluated.

=========================================================================
WHAT SURVIVES, AND WHAT DOES NOT
=========================================================================
Recoverable with per-configuration detail:
  - reports/layer_ablation.json      5 variants, full metrics, per-fold
  - reports/layer_ablation_v2.json   5 variants, full metrics, per-fold

Recoverable as a COUNT ONLY, from narrative records in
docs/project_state.md -- the configurations are described, their metrics
are recorded, but their individual hyperparameters are gone:
  - the Stage 1-3 walk-forward runs and the signal-ceiling measurements

NOT RECOVERABLE AT ALL:
  - Every individual Optuna trial. `mlruns/` contains only .gitkeep, so
    MLflow was configured and never written to. No Optuna storage was
    persisted (studies were built in-memory per call). The winning
    hyperparameters of each study are gone along with the 14 losing ones.
  - Any run whose output was not written to reports/ or narrated in
    project_state.md.

Every backfilled record is written with source='backfill' and a `note`
citing where the number came from, so a reader can tell reconstructed
entries from live ones. Optuna-trial counts that are known to have happened
but cannot be individually reconstructed are NOT written as fake records --
they are reported as a bound, below.

Run: python scripts/backfill_trials.py [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictive_core.trial_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    log_trial,
    registry_summary,
)

REPORTS = Path("reports")

# ---------------------------------------------------------------------- #
# Documented Optuna volume, from docs/project_state.md narrative records.
# Each entry cites the specific claim it rests on. Where a number is a
# range or unclear, the LOW end is used -- this is a lower bound.
# ---------------------------------------------------------------------- #
# STUDIES PER FOLD = 4, verified rather than assumed. `Phase2Strategy.fit()`
# (read before deletion, and `PredictiveCoreStrategy` was identical in this
# respect) called train_direction_classifier ONCE and train_quantile_regressor
# ONCE, for target_col='magnitude' only -- never duration. The quantile trainer
# creates one Optuna study PER QUANTILE, and QUANTILES = (0.1, 0.5, 0.9).
# So: 1 direction study + 3 magnitude studies = 4 studies per fold.
_STUDIES_PER_FOLD = 4

# DISTINCT FITS PER ABLATION RUN = 2, inferred from the wall-clock timings in
# reports/layer_ablation*.log rather than from the (deleted) runner source.
# v1: 2203.2s, 458.3s, then 8.9s / 13.7s / 4.0s.
# v2: 3302.3s, 654.9s, then 10.2s / 21.0s /  7.8s.
# The three sub-15-second variants cannot have retrained anything; the runner
# used a _SharedFitStrategy that reused an already-fitted model. Counting all
# 5 variants as full fits would inflate this bound ~2.5x.
_ABLATION_DISTINCT_FITS = 2

DOCUMENTED_OPTUNA_RUNS = [
    {
        "label": "Stage 1 walk-forward (cheap sanity check)",
        "citation": "project_state.md 2026-08-13: '2 Optuna trials', 5 folds",
        "optuna_trials_per_study": 2,
        "folds": 5,
        "studies_per_fold": _STUDIES_PER_FOLD,
    },
    {
        "label": "Stage 2 walk-forward (full spec budget)",
        "citation": "project_state.md 2026-08-13: '15 trials, 50 early-stopping rounds', 5 folds",
        "optuna_trials_per_study": 15,
        "folds": 5,
        "studies_per_fold": _STUDIES_PER_FOLD,
    },
    {
        "label": "Stage 3 walk-forward (class-weighted re-run)",
        "citation": "project_state.md 2026-08-13: 're-ran identically (again 15 trials...)', 5 folds",
        "optuna_trials_per_study": 15,
        "folds": 5,
        "studies_per_fold": _STUDIES_PER_FOLD,
    },
    {
        "label": "Layer ablation v1 (2 distinct fits, 3 variants reused)",
        "citation": "reports/layer_ablation.log timings; see _ABLATION_DISTINCT_FITS",
        "optuna_trials_per_study": 15,
        "folds": 5,
        "studies_per_fold": _STUDIES_PER_FOLD,
        "variants": _ABLATION_DISTINCT_FITS,
    },
    {
        "label": "Layer ablation v2 (2 distinct fits, 3 variants reused)",
        "citation": "reports/layer_ablation_v2.log timings; see _ABLATION_DISTINCT_FITS",
        "optuna_trials_per_study": 15,
        "folds": 5,
        "studies_per_fold": _STUDIES_PER_FOLD,
        "variants": _ABLATION_DISTINCT_FITS,
    },
]

# Non-Optuna configurations that were each an explicit, separate evaluation.
DOCUMENTED_MANUAL_CONFIGS = [
    ("Stage 4 multivariate ceiling, all bars", 1,
     "project_state.md 2026-08-13 Stage 4"),
    ("Stage 4 univariate AUC, 39 features", 39,
     "project_state.md 2026-08-13: 'for each of the 39 features ... compute its ROC-AUC'"),
    ("Stage 5 conditioned-subset ceiling (BTC filtered)", 1,
     "project_state.md 2026-08-13 Stage 5"),
    ("Stage 5 univariate AUC on filtered subset, 39 features", 39,
     "project_state.md 2026-08-13 Stage 5"),
    ("Stage 6 pooled BTC+ETH bounded-window ceiling", 1,
     "project_state.md 2026-08-13 Stage 6"),
    ("Stage 6 univariate AUC pooled, 39 features", 39,
     "project_state.md 2026-08-13 Stage 6"),
    ("Signal-ceiling re-measurement: 4 barrier geometries", 4,
     "project_state.md 2026-08-14: tp/sl 2:1, 1:1, 1:2, 3:1"),
    ("Signal-ceiling re-measurement: 5 volatility subsets", 5,
     "project_state.md 2026-08-14: ALL / <0.30 / <=0.20 / expanding / extended"),
    ("Signal-ceiling re-measurement: univariate 39 features", 39,
     "project_state.md 2026-08-14"),
    ("Compression-sweep proxy lookback probe (3/5/10/20 candles)", 4,
     "project_state.md 2026-08-13 Stage 5: 'checked 3/5/10/20-candle lookbacks'"),
    ("Sweep detector tolerance settings evaluated", 2,
     "project_state.md: Layer 5 'at two very different selectivity settings'"),
]


def backfill_ablations(write: bool, path) -> int:
    """Write one record per ablation variant. These have real metrics."""
    n = 0
    for fname, tag in (("layer_ablation.json", "ablation_v1"),
                       ("layer_ablation_v2.json", "ablation_v2")):
        p = REPORTS / fname
        if not p.exists():
            print(f"  !! {p} missing -- SKIPPED (stated, not silently omitted)")
            continue
        doc = json.loads(p.read_text())
        for res in doc.get("results", []):
            params = {
                "variant": res.get("variant"),
                "layers": res.get("layers"),
                "asset": doc.get("asset"),
                "start": doc.get("start"),
                "run": tag,
            }
            metrics = {
                k: res.get(k) for k in
                ("oos_sharpe", "oos_profit_factor", "oos_max_drawdown", "oos_trades",
                 "per_fold_oos_sharpe", "equity_destroyed", "under_powered_folds",
                 "n_folds_in_aggregate_sharpe")
                if k in res
            }
            if write:
                log_trial(
                    stage="phase2_layer_ablation",
                    params=params,
                    result_metrics=metrics,
                    n_eff=None,      # not computable retrospectively -- see note
                    nominal_n=35069,
                    source="backfill",
                    note=(f"Reconstructed from reports/{fname}. n_eff is null because "
                          f"uniqueness was not implemented when this ran (Stage A.2) "
                          f"and the labelled frame was not persisted."),
                    path=path,
                )
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually append to trials.jsonl (default: dry run)")
    ap.add_argument("--path", default=str(DEFAULT_REGISTRY_PATH))
    args = ap.parse_args()

    print("=" * 78)
    print("STAGE A.5 -- TRIAL REGISTRY BACKFILL")
    print("=" * 78)

    print("\n1. RECONSTRUCTED WITH FULL DETAIL (written as records)")
    n_records = backfill_ablations(args.write, args.path)
    print(f"   ablation variants recovered: {n_records}")

    print("\n2. DOCUMENTED OPTUNA VOLUME (counted, NOT reconstructable as records)")
    print(f"   {'run':<52}{'studies':>9}{'trials':>9}")
    print("   " + "-" * 70)
    optuna_total = 0
    for r in DOCUMENTED_OPTUNA_RUNS:
        variants = r.get("variants", 1)
        studies = r["folds"] * r["studies_per_fold"] * variants
        trials = studies * r["optuna_trials_per_study"]
        optuna_total += trials
        print(f"   {r['label']:<52}{studies:>9,}{trials:>9,}")
    print("   " + "-" * 70)
    print(f"   {'OPTUNA TRIALS (documented)':<52}{'':>9}{optuna_total:>9,}")

    print("\n3. DOCUMENTED NON-OPTUNA CONFIGURATIONS")
    manual_total = 0
    for label, count, _cite in DOCUMENTED_MANUAL_CONFIGS:
        manual_total += count
        print(f"   {label:<62}{count:>6,}")
    print("   " + "-" * 70)
    print(f"   {'NON-OPTUNA CONFIGURATIONS (documented)':<62}{manual_total:>6,}")

    lower_bound = optuna_total + manual_total + n_records
    print("\n" + "=" * 78)
    print(f"DOCUMENTED LOWER BOUND ON DISTINCT CONFIGURATIONS EVALUATED: {lower_bound:,}")
    print("=" * 78)
    print("  This is an UNDERESTIMATE, and is labelled as one. It omits:")
    print("   - every exploratory run not written to reports/ or narrated in")
    print("     project_state.md (unknown, unbounded)")
    print("   - the many-fold internal re-fits inside the backtest harness")
    print("   - hyperparameters of individual Optuna trials, which are gone:")
    print("     mlruns/ holds only .gitkeep, and studies were in-memory per call")
    print("  It may also OVERCOUNT in one direction: identical Optuna")
    print("  configurations re-proposed across folds are counted separately here,")
    print("  whereas the registry hashes them and would collapse duplicates.")
    print("  Both directions are stated rather than netted into a single figure.")

    if args.write:
        print("\nRegistry summary after backfill:")
        for k, v in registry_summary(args.path).items():
            print(f"  {k}: {v}")
    else:
        print("\n(dry run -- pass --write to append the reconstructable records)")


if __name__ == "__main__":
    main()
