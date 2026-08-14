"""
Gate A leakage test: the unpurged splitter leaks, the purged one does not.

=========================================================================
THE CONSTRUCTION
=========================================================================
A feature containing its own sample's label would inflate AUC under BOTH
splitters and prove nothing about purging. The leak must travel through the
OVERLAP mechanism specifically, so that removing overlapping samples is
what severs it.

    e_t             i.i.d. coin flip at every bar -- the resolution outcome
    h_i             per-sample label horizon, random in [1, H]
    t1_i = t0_i+h_i resolution bar (variable, as real triple barriers are)

    y_i = e_{t1_i}  the label is the outcome at the bar where it resolves
    x_i = t1_i      the feature identifies WHICH bar resolves it

Two samples resolving on the same bar therefore carry the SAME label. That
is overlapping-label leakage reduced to its minimum: shared resolution bars
mean shared outcomes.

There is no genuine signal anywhere. e is i.i.d., so the only achievable
AUC without leakage is 0.5000 -- and that holds for THIS realisation, not
merely in expectation. (An earlier version of this script used a random-walk
price level as the feature and its forward move as the label. That was
confounded: a single realised walk contains real level/return structure, so
one fold showed 0.72 under both splitters. The inflation there was the
artifact, not the leak. Recorded because the flawed version looked
convincing until the per-fold numbers were read.)

=========================================================================
WHY THE COMPARISON IS PURGE-ON vs PURGE-OFF ON IDENTICAL FOLDS
=========================================================================
Measured while building this: under the OLD walk-forward geometry
(train 2000 / val 400 / test 400, horizon <= 120), train-to-test overlap is
exactly ZERO -- the 400-bar validation block is wider than the horizon and
acts as an accidental buffer. The old splitter cannot be made to leak into
TEST in that layout, so comparing against it would understate the problem
rather than demonstrate it.

Its real defect in that layout is train-to-VALIDATION contamination, which
is measured separately at the bottom of this report. Validation drives
early stopping and Optuna selection, so it is not a harmless leak.

The headline comparison therefore holds fold geometry fixed and toggles ONLY
purging, so the delta is attributable to purging and nothing else.

Run: python scripts/leakage_test.py
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictive_core.purged_kfold import PurgedKFold  # noqa: E402
from src.predictive_core.splitter import WalkForwardSplitter  # noqa: E402

# Geometry chosen so the leak is MEASURABLE, not so it is flattering.
# The leak's size depends on horizon relative to block size: with H >> block,
# most test resolution bars are reachable from outside the block. At H=600 /
# block=200 the unpurged leak channel covers ~49% of each test fold, well
# clear of AUC noise. At the project's real max_candles=48 against
# thousand-bar folds the channel is far narrower -- that figure is measured
# and reported separately at the end, and it is the decision-relevant one.
N_BARS = 8000
MAX_H = 600
N_SPLITS = 40          # -> 200-bar test blocks
SEED = 11
FEATURES = ["resolution_bar", "noise"]

# Project's real configuration, for the closing comparison.
REAL_HORIZON = 48
REAL_N = 78378
REAL_SPLITS = 5


def build_leak_frame(n=N_BARS, max_h=MAX_H, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = rng.integers(0, 2, size=n + max_h + 1)
    h = rng.integers(1, max_h + 1, size=n)
    t0_pos = np.arange(n)
    t1_pos = t0_pos + h
    full_idx = pd.date_range("2020-01-01", periods=n + max_h + 1, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "resolution_bar": t1_pos.astype(float),
            "noise": rng.normal(size=n),
            "y": e[t1_pos].astype(int),
            "t0": full_idx[t0_pos],
            "t1": full_idx[t1_pos],
        },
        index=full_idx[t0_pos],
    )


def _auc(df, train_pos, eval_pos) -> float:
    tr, ev = df.iloc[train_pos], df.iloc[eval_pos]
    if tr["y"].nunique() < 2 or ev["y"].nunique() < 2:
        return float("nan")
    model = lgb.train(
        {"objective": "binary", "metric": "auc", "verbosity": -1,
         "learning_rate": 0.05, "num_leaves": 255, "min_child_samples": 2,
         "seed": SEED, "feature_fraction": 1.0, "bagging_fraction": 1.0},
        lgb.Dataset(tr[FEATURES], label=tr["y"]),
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(0)],
    )
    return float(roc_auc_score(ev["y"], model.predict(ev[FEATURES])))


def _shared(df, a_pos, b_pos) -> int:
    """Count of b-samples resolving on a bar some a-sample also resolves on
    -- the leak channel, counted directly rather than inferred from AUC."""
    a = set(df.iloc[a_pos]["resolution_bar"].to_numpy())
    b = df.iloc[b_pos]["resolution_bar"].to_numpy()
    return int(np.isin(b, list(a)).sum())


def run():
    df = build_leak_frame()
    unpurged = list(PurgedKFold(N_SPLITS, purge=False).split_detailed(df))
    purged = list(PurgedKFold(N_SPLITS, embargo_pct=0.01, purge=True).split_detailed(df))

    rows = []
    for u, p in zip(unpurged, purged):
        rows.append({
            "fold": u.fold,
            "n_test": len(u.test),
            "old_auc": _auc(df, u.train, u.test),
            "new_auc": _auc(df, p.train, p.test),
            "old_shared": _shared(df, u.train, u.test),
            "new_shared": _shared(df, p.train, p.test),
            "purge": p.summary(),
        })

    # The old splitter's actual defect in its own geometry: train -> val.
    wf = WalkForwardSplitter(2000, 400, 400)
    wf_rows = []
    for i, (tr, va, te) in enumerate(wf.split(df)):
        wf_rows.append({
            "fold": i,
            "train_val_shared": _shared(df, tr, va),
            "train_test_shared": _shared(df, tr, te),
            "n_val": len(va), "n_test": len(te),
            "val_auc": _auc(df, tr, va),
        })
    return rows, wf_rows


def main() -> None:
    rows, wf_rows = run()
    old = np.array([r["old_auc"] for r in rows], dtype=float)
    new = np.array([r["new_auc"] for r in rows], dtype=float)

    print("=" * 86)
    print("GATE A LEAKAGE TEST")
    print("Label = i.i.d. coin flip at the resolution bar. NO genuine signal exists.")
    print("Achievable AUC without leakage = 0.5000, exactly, for this realisation.")
    print(f"n={N_BARS}  max horizon={MAX_H} bars  K={N_SPLITS} folds  embargo=1%")
    print("=" * 86)

    print("\nPer-fold purge accounting:")
    for r in rows:
        print("  " + r["purge"])

    print("\nLEAK CHANNEL -- test samples sharing a resolution bar with a training sample:")
    print(f"{'fold':<8}{'UNPURGED':>12}{'PURGED':>10}{'of n_test':>12}")
    print("-" * 42)
    for r in rows:
        print(f"{r['fold']:<8}{r['old_shared']:>12}{r['new_shared']:>10}{r['n_test']:>12}")

    print(f"\nTEST AUC{'':<4}{'UNPURGED':>14}{'PURGED':>12}{'delta':>12}")
    print("-" * 46)
    for r in rows:
        print(f"{r['fold']:<12}{r['old_auc']:>14.4f}{r['new_auc']:>12.4f}"
              f"{r['old_auc'] - r['new_auc']:>12.4f}")
    print("-" * 46)
    print(f"{'MEAN':<12}{np.nanmean(old):>14.4f}{np.nanmean(new):>12.4f}"
          f"{np.nanmean(old) - np.nanmean(new):>12.4f}")

    print(f"\n  UNPURGED : mean AUC {np.nanmean(old):.4f}   "
          f"inflation over truth {np.nanmean(old) - 0.5:+.4f}")
    print(f"  PURGED   : mean AUC {np.nanmean(new):.4f}   "
          f"inflation over truth {np.nanmean(new) - 0.5:+.4f}")

    print("\n" + "=" * 86)
    print("OLD WalkForwardSplitter, in its own geometry (train 2000/val 400/test 400)")
    print("=" * 86)
    print(f"{'fold':<8}{'train->val shared':>20}{'train->test shared':>21}{'val AUC':>10}")
    print("-" * 60)
    for r in wf_rows:
        print(f"{r['fold']:<8}{r['train_val_shared']:>20}{r['train_test_shared']:>21}"
              f"{r['val_auc']:>10.4f}")
    tv = np.mean([r["train_val_shared"] for r in wf_rows])
    tt = np.mean([r["train_test_shared"] for r in wf_rows])
    print(f"\n  mean train->val shared {tv:.0f} of {wf_rows[0]['n_val']}   "
          f"mean train->test shared {tt:.1f} of {wf_rows[0]['n_test']}")
    print("  The old splitter purges NEITHER. train->test is small here only because")
    print("  the val block partly buffers it -- an accident of window sizing, not a")
    print("  property of the splitter. When the horizon exceeds the val block (as it")
    print("  does at this geometry) it stops holding, and train->test goes nonzero.")
    print("  train->val is large regardless, and val drives early stopping and Optuna")
    print("  selection -- so the old splitter contaminates the partition that CHOOSES")
    print("  the model, then reports that model's score on test as if it were clean.")

    # --- The project's real configuration, for scale ---------------------- #
    print("\n" + "=" * 86)
    print(f"SAME MEASUREMENT AT THE PROJECT'S REAL GEOMETRY "
          f"(horizon={REAL_HORIZON}, n={REAL_N:,}, K={REAL_SPLITS})")
    print("=" * 86)
    real = build_leak_frame(n=REAL_N, max_h=REAL_HORIZON, seed=SEED)
    ru = list(PurgedKFold(REAL_SPLITS, purge=False).split_detailed(real))
    rp = list(PurgedKFold(REAL_SPLITS, embargo_pct=0.01, purge=True).split_detailed(real))
    shares_u = [_shared(real, f.train, f.test) / len(f.test) for f in ru]
    shares_p = [_shared(real, f.train, f.test) / len(f.test) for f in rp]
    ret = [f.train_retention for f in rp]
    print(f"  unpurged leak channel : {np.mean(shares_u):.2%} of each test fold")
    print(f"  purged leak channel   : {np.mean(shares_p):.2%}")
    print(f"  training data retained after purge+embargo : {np.mean(ret):.1%}")
    print("\n  The channel is narrow here because a 48-bar horizon against ~15,700-bar")
    print("  folds touches only the fold edges. Narrow is not zero, and it is not")
    print("  bounded: it widens with horizon and with fold count.")


if __name__ == "__main__":
    main()
