"""
Measure the predictive ceiling of the current 39-feature set, unconditioned
and conditioned on volatility state (Layer 6).

Reproduces the Stage 4 methodology from docs/project_state.md so the numbers
are directly comparable, and extends it to answer a specific question: does
CONDITIONING on compression -- the first of MASTER_CONTEXT's four hypothesis
conditions, and the only one computable from the data this project ingests
-- recover any separable signal that the unconditioned test dilutes away?

Manual diagnostic; not production code.
    python -m scripts.measure_signal_ceiling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_engine.crypto_fetcher import remove_duplicates_and_skew
from src.predictive_core.dataset import build_asset_modeling_frame, feature_columns

SEED = 7


def _binary(df: pd.DataFrame) -> pd.DataFrame:
    """up-vs-down only; the neutral class is <1% and not the question."""
    return df[df["direction"] != 0.0]


def _univariate(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    y = (df["direction"] > 0).astype(int).to_numpy()
    rows = []
    for c in feats:
        x = df[c].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 100 or len(np.unique(y[ok])) < 2 or len(np.unique(x[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        rows.append({"feature": c, "auc": auc, "abs_edge": abs(auc - 0.5)})
    return pd.DataFrame(rows).sort_values("abs_edge", ascending=False)


def _multivariate_ceiling(df: pd.DataFrame, feats: list, label: str) -> dict:
    """Deliberately the MOST favourable conditions the same features can be
    given: high capacity, unbounded depth, real early stopping, no Optuna
    limits, no class weighting, no calibration, no gating."""
    import lightgbm as lgb

    n = len(df)
    if n < 400:
        return {"label": label, "n": n, "test_auc": float("nan"), "note": "too few rows"}
    i1, i2 = int(n * 0.70), int(n * 0.85)
    tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    ytr = (tr["direction"] > 0).astype(int)
    yva = (va["direction"] > 0).astype(int)
    yte = (te["direction"] > 0).astype(int)
    if min(ytr.nunique(), yva.nunique(), yte.nunique()) < 2:
        return {"label": label, "n": n, "test_auc": float("nan"), "note": "single-class split"}

    model = lgb.train(
        {"objective": "binary", "metric": "binary_logloss", "num_leaves": 63,
         "max_depth": -1, "learning_rate": 0.05, "verbose": -1, "seed": SEED,
         "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1},
        lgb.Dataset(tr[feats], ytr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(va[feats], yva)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p = model.predict(te[feats], num_iteration=model.best_iteration)
    auc = roc_auc_score(yte, p)
    base = float(ytr.mean())
    eps = 1e-12
    ll_model = float(-np.mean(yte * np.log(p + eps) + (1 - yte) * np.log(1 - p + eps)))
    ll_base = float(-np.mean(yte * np.log(base + eps) + (1 - yte) * np.log(1 - base + eps)))

    # Bootstrap CI on test AUC -- these subsets get small, and a point
    # estimate without an interval would overstate what they can support.
    rng = np.random.default_rng(SEED)
    yv, pv = yte.to_numpy(), np.asarray(p)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2:
            continue
        boot.append(roc_auc_score(yv[idx], pv[idx]))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
    return {
        "label": label, "n": n, "n_test": len(te), "rounds": model.best_iteration,
        "test_auc": auc, "ci_lo": lo, "ci_hi": hi,
        "logloss_model": ll_model, "logloss_baseline": ll_base,
        "up_rate": float((df["direction"] > 0).mean()),
    }


def main() -> None:
    raw = pd.read_parquet("data/raw/BTC_USDT_1h.parquet")
    raw = remove_duplicates_and_skew(raw, "1h")
    df = build_asset_modeling_frame(raw, keep_raw_atr=True)
    feats = feature_columns(df)
    print(f"Loaded {len(df)} rows, {len(feats)} features, {df.index[0].date()} -> {df.index[-1].date()}\n")

    b = _binary(df)
    print(f"up/down rows: {len(b)}  (up rate {float((b['direction']>0).mean()):.3f})\n")

    print("=" * 74)
    print("1. UNIVARIATE AUC -- every feature scored on its own")
    print("=" * 74)
    uni = _univariate(b, feats)
    print(uni.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  median |AUC-0.5| across all {len(uni)} features: {uni['abs_edge'].median():.4f}")
    print(f"  best   |AUC-0.5|                              : {uni['abs_edge'].max():.4f}")

    print("\n" + "=" * 74)
    print("2. MULTIVARIATE CEILING -- conditioned on volatility state (Layer 6)")
    print("=" * 74)
    pct = df["vol_atr_pct_rank"]
    # THRESHOLDS DELIBERATELY HARDCODED, NOT IMPORTED FROM LAYER 6.
    # This is a measurement tool, and its subset definitions must stay fixed
    # so runs remain comparable across sessions. Importing
    # COMPRESSED_MAX_PERCENTILE would silently redefine what "Compressed"
    # means here every time Layer 6 is retuned, making today's numbers
    # incomparable with the ones already recorded in project_state.md.
    #
    # Labels updated at Stage 0 task 0.4: 0.20 is now the LIVE Layer 6 value
    # (per S6), and 0.30 is the superseded one retained as a comparison arm.
    subsets = {
        "ALL bars (unconditioned)": b,
        "Compressed  (ATR pct < 0.30, SUPERSEDED value)": _binary(df[pct < 0.30]),
        "Compressed  (ATR pct <= 0.20, LIVE Layer 6)": _binary(df[pct <= 0.20]),
        "Expanding   (0.30 <= ATR pct <= 0.70, old band)": _binary(df[(pct >= 0.30) & (pct <= 0.70)]),
        "Expanding   (0.20 <= ATR pct <= 0.70, LIVE band)": _binary(df[(pct >= 0.20) & (pct <= 0.70)]),
        "Extended    (ATR pct > 0.70)": _binary(df[pct > 0.70]),
    }
    results = [_multivariate_ceiling(s, feats, name) for name, s in subsets.items()]
    print(f"{'subset':<48}{'n':>7}{'test AUC':>10}{'95% CI':>18}{'rounds':>8}")
    for r in results:
        if np.isnan(r.get("test_auc", np.nan)):
            print(f"{r['label']:<48}{r['n']:>7}{'n/a':>10}{r.get('note',''):>18}")
            continue
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"{r['label']:<48}{r['n']:>7}{r['test_auc']:>10.4f}{ci:>18}{r['rounds']:>8}")

    print("\n  logloss vs base-rate baseline (lower is better):")
    for r in results:
        if np.isnan(r.get("test_auc", np.nan)):
            continue
        d = r["logloss_baseline"] - r["logloss_model"]
        print(f"    {r['label']:<48} model {r['logloss_model']:.4f} vs base {r['logloss_baseline']:.4f}"
              f"  -> {d:+.5f}")

    print("\n  class balance per subset (up rate):")
    for r in results:
        print(f"    {r['label']:<48} {r['up_rate']:.3f}")


if __name__ == "__main__":
    main()
