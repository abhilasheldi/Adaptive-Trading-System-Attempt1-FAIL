"""
Stage A.6 -- re-measure the Stage 5 ceiling under the corrected apparatus.

Reproduces the Stage 5 subset table with three corrections that did not
exist when the original numbers were produced:

  PURGED CV            PurgedKFold + 1% embargo replaces the chronological
                       70/15/15 split. Training samples whose label window
                       reaches into the test block are dropped (A.3).
  UNIQUENESS WEIGHTING Each training sample weighted by its average
                       uniqueness u_bar_i (A.2).
  CORRECTED CI         The bootstrap resamples n_eff observations, not n.
                       The naive bootstrap resamples n i.i.d. draws, which
                       assumes every label is an independent observation --
                       measured at Stage A.1, ~8-9 labels span every bar, so
                       that assumption inflates precision by construction.

INVARIANT I1: AUC is a DIAGNOSTIC here. It is reported, and it is not
optimised, targeted, or used as a gate. No hyperparameter below was chosen
by looking at these numbers -- the model configuration is deliberately the
same generous, untuned LightGBM the original Stage 5 measurement used, so
the corrected numbers are comparable to the ones they replace.

Every configuration evaluated is logged to trials.jsonl (A.5).

Run: python scripts/measure_ceiling_corrected.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.macro_mtf_liquidity_vol.volatility_classifier import (  # noqa: E402
    COMPRESSED_MAX_PERCENTILE, EXTENDED_MIN_PERCENTILE,
)
from src.predictive_core.dataset import build_asset_modeling_frame, feature_columns  # noqa: E402
from src.predictive_core.purged_kfold import (  # noqa: E402
    PurgedKFold,
    assert_scored_rows_match_nominal,
)
from src.predictive_core.trial_registry import log_trial  # noqa: E402
from src.predictive_core.uniqueness import average_uniqueness  # noqa: E402

MAX_CANDLES = 48
N_SPLITS = 5
EMBARGO_PCT = 0.01
VAL_SIZE = 1500
N_BOOT = 2000
SEED = 42
STAGE = "A6_corrected_ceiling"

# Deliberately the same generous, untuned configuration Stage 5 used.
LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
    "num_leaves": 63, "max_depth": -1, "learning_rate": 0.05,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
    "min_child_samples": 20, "seed": SEED,
}
MAX_ROUNDS, EARLY_STOP = 2000, 50


def bootstrap_auc_ci(y, p, n_draw, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap CI, drawing `n_draw` observations per replicate.

    Passing n_draw = n_eff instead of n is the correction. Resampling n
    i.i.d. draws from overlapping labels treats shared information as
    independent evidence and produces a CI that is too narrow.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    n_draw = max(int(round(n_draw)), 10)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n_draw)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def evaluate(df, feats, label: str):
    """Out-of-fold AUC under purged CV with uniqueness-weighted training."""
    y_all = (df["direction"].to_numpy() > 0).astype(int)

    uniq = average_uniqueness(df["t0"], df["t1"])
    w_all = uniq.average_uniqueness
    w_all = w_all / w_all.mean()

    oof_pred = np.full(len(df), np.nan)
    rounds, purged, embargoed = [], 0, 0

    splitter = PurgedKFold(N_SPLITS, embargo_pct=EMBARGO_PCT, val_size=VAL_SIZE)
    skipped: list[str] = []
    for f in splitter.split_detailed(df):
        tr, va, te = f.train, f.val, f.test
        # Recorded rather than passed over in silence. The assertion after the
        # loop turns any skip into a hard failure; this only names the cause.
        if va is None or len(va) == 0:
            skipped.append(f"fold {f.fold}: empty validation set")
            continue
        if len(np.unique(y_all[tr])) < 2:
            skipped.append(f"fold {f.fold}: single-class training partition")
            continue
        purged += f.n_purged
        embargoed += f.n_embargoed

        model = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(df.iloc[tr][feats], label=y_all[tr], weight=w_all[tr]),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(df.iloc[va][feats], label=y_all[va])],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )
        rounds.append(int(model.best_iteration))
        oof_pred[te] = model.predict(df.iloc[te][feats], num_iteration=model.best_iteration)

    mask = ~np.isnan(oof_pred)

    # BUILD_PLAN_v2 amendment 2.1. A.6's first run scored exactly 4/5 of the
    # rows and nothing said so; the shortfall was found by hand. Counted here
    # instead, before any metric is computed from `mask`.
    assert_scored_rows_match_nominal(
        int(mask.sum()), len(df),
        context=f"[{label}] purged-CV out-of-fold scoring"
        + (f" (skipped: {'; '.join(skipped)})" if skipped else ""),
    )

    y, p = y_all[mask], oof_pred[mask]
    if len(np.unique(y)) < 2:
        return None

    auc = float(roc_auc_score(y, p))
    base_rate = float(y.mean())
    ll = float(log_loss(y, p, labels=[0, 1]))
    ll_base = float(log_loss(y, np.full(len(y), base_rate), labels=[0, 1]))

    # n_eff of the SCORED rows, which is what the CI must reflect.
    scored = df.iloc[np.flatnonzero(mask)]
    n_eff_scored = average_uniqueness(scored["t0"], scored["t1"]).effective_n

    lo_c, hi_c = bootstrap_auc_ci(y, p, n_eff_scored)
    lo_n, hi_n = bootstrap_auc_ci(y, p, len(y))

    return {
        "label": label,
        "mean_pred": float(p.mean()),
        "calib_gap": float(p.mean() - base_rate),
        "nominal_n": len(y),
        "n_eff": n_eff_scored,
        "ratio": n_eff_scored / len(y),
        "auc": auc,
        "ci_corrected": (lo_c, hi_c),
        "ci_naive": (lo_n, hi_n),
        "logloss": ll,
        "logloss_baseline": ll_base,
        "logloss_vs_baseline": ll_base - ll,
        "rounds": float(np.mean(rounds)) if rounds else float("nan"),
        "purged": purged,
        "embargoed": embargoed,
        "up_rate": base_rate,
    }


def main() -> None:
    path = Path("data/raw/BTC_USDT_1h.parquet")
    if not path.exists():
        print(f"!! {path} not found -- cannot measure. STATED, not silently skipped.")
        return

    full = build_asset_modeling_frame(pd.read_parquet(path), max_candles=MAX_CANDLES)
    feats = feature_columns(full)
    # Match Stage 5: binary up/down, neutral excluded.
    full = full[full["direction"] != 0].copy()
    pct = full["vol_atr_pct_rank"]

    subsets = [
        ("ALL bars (unconditioned)", full),
        ("Compressed  ATR pct < 0.30 (old Layer 6)", full[pct < 0.30]),
        (f"Compressed  ATR pct <= {COMPRESSED_MAX_PERCENTILE} (LIVE Layer 6)",
         full[pct <= COMPRESSED_MAX_PERCENTILE]),
        ("Expanding   0.30 <= ATR pct <= 0.70", full[(pct >= 0.30) & (pct <= 0.70)]),
        (f"Extended    ATR pct > {EXTENDED_MIN_PERCENTILE}", full[pct > EXTENDED_MIN_PERCENTILE]),
    ]

    print(f"BTC/USDT 1h  {len(full):,} labelled rows (neutral excluded)  "
          f"{len(feats)} features")
    print(f"Purged K-Fold K={N_SPLITS}, embargo={EMBARGO_PCT:.0%}, "
          f"uniqueness-weighted training, CI bootstrap on n_eff\n")

    rows = []
    for label, sub in subsets:
        if len(sub) < 2000:
            print(f"  {label}: only {len(sub):,} rows -- SKIPPED (stated)")
            continue
        print(f"  running {label} ({len(sub):,} rows) ...", flush=True)
        r = evaluate(sub.copy(), feats, label)
        if r is None:
            print(f"    -> not evaluable (single class after folding). STATED.")
            continue
        rows.append(r)
        log_trial(
            stage=STAGE,
            params={"subset": label, "model": "lgb_binary_ceiling",
                    "n_splits": N_SPLITS, "embargo_pct": EMBARGO_PCT,
                    "uniqueness_weighted": True, "purged_cv": True, **LGB_PARAMS},
            result_metrics={"auc": r["auc"], "ci_corrected": list(r["ci_corrected"]),
                            "logloss": r["logloss"], "logloss_baseline": r["logloss_baseline"],
                            "boost_rounds": r["rounds"]},
            n_eff=r["n_eff"], nominal_n=r["nominal_n"], source="manual",
            note="Stage A.6 corrected ceiling: purged CV + uniqueness weighting + n_eff CI",
        )

    print("\n" + "=" * 118)
    print("STAGE A.6 -- CORRECTED CEILING (purged CV, uniqueness-weighted, n_eff CI)")
    print("=" * 118)
    hdr = (f"{'subset':<42}{'nominal n':>10}{'n_eff':>9}{'ratio':>7}"
           f"{'AUC':>8}{'corrected 95% CI':>20}{'LL vs base':>12}{'rounds':>8}")
    print(hdr)
    print("-" * 118)
    for r in rows:
        lo, hi = r["ci_corrected"]
        straddles = "" if lo > 0.5 else "  <- CI includes 0.5"
        print(f"{r['label']:<42}{r['nominal_n']:>10,}{r['n_eff']:>9,.0f}{r['ratio']:>7.3f}"
              f"{r['auc']:>8.4f}   [{lo:.4f}, {hi:.4f}]{r['logloss_vs_baseline']:>12.5f}"
              f"{r['rounds']:>8.1f}{straddles}")

    print("\n" + "-" * 118)
    print("Same rows, NAIVE CI (bootstrap on nominal n) -- shown to make the widening visible:")
    print(f"{'subset':<42}{'naive 95% CI':>22}{'width':>9}{'corrected width':>18}{'factor':>9}")
    print("-" * 118)
    for r in rows:
        nlo, nhi = r["ci_naive"]
        clo, chi = r["ci_corrected"]
        nw, cw = nhi - nlo, chi - clo
        print(f"{r['label']:<42}   [{nlo:.4f}, {nhi:.4f}]{nw:>9.4f}{cw:>18.4f}"
              f"{(cw / nw if nw else float('nan')):>9.2f}x")

    print("\nPurge accounting (training rows removed per subset, summed over folds):")
    for r in rows:
        print(f"  {r['label']:<42} purged {r['purged']:>7,}  embargoed {r['embargoed']:>6,}")

    print("\n" + "-" * 118)
    print("CALIBRATION -- why 'LL vs base' is negative everywhere. Uniqueness weighting")
    print("shifts the CLASS PRIOR: sl labels resolve faster (median 4 bars) than tp")
    print("labels (7), so they overlap less, score higher uniqueness, and take more")
    print("weight. The model tracks the WEIGHTED prior while the test set carries the")
    print("UNWEIGHTED one. AUC is rank-based and unaffected; logloss is not.")
    print("-" * 118)
    print(f"{'subset':<42}{'test up-rate':>14}{'mean prediction':>18}{'calibration gap':>18}")
    print("-" * 118)
    for r in rows:
        print(f"{r['label']:<42}{r['up_rate']:>14.4f}{r['mean_pred']:>18.4f}"
              f"{r['calib_gap']:>18.4f}")


if __name__ == "__main__":
    main()
