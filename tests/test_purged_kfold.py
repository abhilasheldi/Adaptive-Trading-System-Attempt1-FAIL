"""
Stage A.3 -- purged K-Fold with embargo, and the Gate A leakage test.

The leakage test at the bottom is the one Gate A actually gates on: it
constructs a dataset with NO genuine signal, confirms an unpurged splitter
reports better-than-chance AUC anyway, and confirms purging removes it.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.predictive_core.purged_kfold import (
    PurgedKFold,
    purge_and_embargo,
    require_label_metadata,
)


def _frame(t0_pos, t1_pos, n_bars=None, extra=None):
    n_bars = n_bars or (int(max(t1_pos)) + 1)
    axis = pd.date_range("2020-01-01", periods=n_bars, freq="1h", tz="UTC")
    data = {"t0": axis[t0_pos], "t1": axis[t1_pos], "y": np.arange(len(t0_pos)) % 2}
    if extra:
        data.update(extra)
    return pd.DataFrame(data, index=axis[t0_pos])


# ---------------------------------------------------------------------- #
# Metadata contract
# ---------------------------------------------------------------------- #


def test_refuses_to_split_without_label_metadata():
    """A splitter that silently skipped purging would be indistinguishable
    from the leaking one it replaces."""
    df = _frame(np.arange(100), np.arange(100) + 5).drop(columns=["t1"])
    with pytest.raises(ValueError, match="t1"):
        list(PurgedKFold(3).split(df))


def test_rejects_t1_before_t0():
    df = _frame(np.arange(10), np.arange(10))
    df["t1"] = df["t0"] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="t1 >= t0"):
        require_label_metadata(df)


# ---------------------------------------------------------------------- #
# Purge and embargo semantics, hand-checkable
# ---------------------------------------------------------------------- #


def test_purge_drops_exactly_the_overlapping_windows():
    """Block = bars [10, 20]. Windows overlapping it must go; the rest stay."""
    axis = pd.date_range("2020-01-01", periods=40, freq="1h", tz="UTC")
    t0 = axis[np.array([0, 5, 8, 12, 25, 30])].to_numpy(dtype="datetime64[ns]")
    t1 = axis[np.array([3, 9, 14, 18, 28, 33])].to_numpy(dtype="datetime64[ns]")
    #                        keep  keep  DROP  DROP  keep  keep
    cand = np.arange(6)
    kept, n_pur, n_emb = purge_and_embargo(
        cand, t0, t1, [(axis[10].to_datetime64(), axis[20].to_datetime64())],
        np.unique(np.concatenate([t0, t1])), embargo_bars=0,
    )
    assert kept.tolist() == [0, 1, 4, 5]
    assert n_pur == 2 and n_emb == 0


def test_embargo_drops_samples_starting_just_after_the_block():
    """Sample 4 starts 5 bars after the block ends and does not overlap it,
    so purging alone keeps it. A 6-bar embargo must remove it."""
    axis = pd.date_range("2020-01-01", periods=60, freq="1h", tz="UTC")
    t0 = axis[np.array([0, 25, 40])].to_numpy(dtype="datetime64[ns]")
    t1 = axis[np.array([3, 28, 43])].to_numpy(dtype="datetime64[ns]")
    bars = np.unique(np.concatenate([t0, t1]))
    block = (axis[10].to_datetime64(), axis[20].to_datetime64())

    no_emb, _, e0 = purge_and_embargo(np.arange(3), t0, t1, [block], bars, embargo_bars=0)
    assert e0 == 0 and 1 in no_emb.tolist()

    with_emb, _, e1 = purge_and_embargo(np.arange(3), t0, t1, [block], bars, embargo_bars=20)
    assert e1 >= 1
    assert 1 not in with_emb.tolist()


def test_a_sample_caught_by_both_is_counted_once():
    axis = pd.date_range("2020-01-01", periods=60, freq="1h", tz="UTC")
    t0 = axis[np.array([12])].to_numpy(dtype="datetime64[ns]")
    t1 = axis[np.array([18])].to_numpy(dtype="datetime64[ns]")
    bars = np.unique(np.concatenate([t0, t1]))
    _, n_pur, n_emb = purge_and_embargo(
        np.arange(1), t0, t1,
        [(axis[10].to_datetime64(), axis[20].to_datetime64())], bars, embargo_bars=20,
    )
    assert (n_pur, n_emb) == (1, 0)


# ---------------------------------------------------------------------- #
# Fold structure
# ---------------------------------------------------------------------- #


def test_training_data_comes_from_both_sides_of_the_block():
    """The property that makes the embargo meaningful at all -- a pure
    walk-forward layout has no training data after the block, so the
    embargo clause could never fire."""
    n = 600
    df = _frame(np.arange(n), np.arange(n) + 5, n_bars=n + 10)
    folds = list(PurgedKFold(5, embargo_pct=0.0).split_detailed(df))
    middle = folds[2]
    assert (middle.train < middle.test.min()).any()
    assert (middle.train > middle.test.max()).any()


def test_embargo_actually_fires_in_kfold():
    n = 600
    df = _frame(np.arange(n), np.arange(n) + 5, n_bars=n + 10)
    folds = list(PurgedKFold(5, embargo_pct=0.02).split_detailed(df))
    assert sum(f.n_embargoed for f in folds) > 0


def test_purge_accounting_balances():
    n = 800
    rng = np.random.default_rng(3)
    h = rng.integers(1, 40, size=n)
    df = _frame(np.arange(n), np.arange(n) + h, n_bars=n + 50)
    for f in PurgedKFold(5, embargo_pct=0.01).split_detailed(df):
        assert len(f.train) == f.n_train_before - f.n_purged - f.n_embargoed
        assert 0 < f.train_retention <= 1.0


def test_no_surviving_train_window_touches_the_test_block():
    """The guard the old splitter lacked, asserted on real fold output."""
    n = 800
    rng = np.random.default_rng(5)
    h = rng.integers(1, 60, size=n)
    df = _frame(np.arange(n), np.arange(n) + h, n_bars=n + 70)
    t0, t1 = require_label_metadata(df)
    for f in PurgedKFold(5, embargo_pct=0.01).split_detailed(df):
        b0, b1 = t0[f.test].min(), t1[f.test].max()
        assert not ((t0[f.train] <= b1) & (t1[f.train] >= b0)).any()


def test_every_row_is_tested_exactly_once_across_folds():
    n = 500
    df = _frame(np.arange(n), np.arange(n) + 5, n_bars=n + 10)
    tested = np.concatenate([f.test for f in PurgedKFold(5).split_detailed(df)])
    assert sorted(tested.tolist()) == list(range(n))


def test_first_fold_still_gets_a_validation_set():
    """Regression, found at Stage A.6. Validation is carved from the rows
    BEFORE the test block, but fold 0's block starts at row 0 -- so `before`
    is empty and val came back empty, making callers skip fold 0 and lose
    1/K of the data from every out-of-fold estimate (scored rows came to
    exactly 4/5 of nominal). It must fall back to the rows AFTER the block,
    which is legitimate in K-fold where training data exists on both sides."""
    n = 3000
    df = _frame(np.arange(n), np.arange(n) + 30, n_bars=n + 40)
    folds = list(PurgedKFold(5, embargo_pct=0.01, val_size=200).split_detailed(df))
    assert len(folds) == 5
    for f in folds:
        assert f.val is not None and len(f.val) > 0, f"fold {f.fold} has no validation set"
        assert len(np.intersect1d(f.val, f.test)) == 0
        assert len(np.intersect1d(f.val, f.train)) == 0


def test_purge_false_is_available_but_changes_nothing_else():
    n = 500
    df = _frame(np.arange(n), np.arange(n) + 20, n_bars=n + 30)
    off = list(PurgedKFold(5, purge=False).split_detailed(df))
    on = list(PurgedKFold(5, purge=True, embargo_pct=0.01).split_detailed(df))
    for a, b in zip(off, on):
        assert a.test.tolist() == b.test.tolist()      # same geometry
        assert len(a.train) > len(b.train)             # only training differs
        assert a.n_purged == 0


# ---------------------------------------------------------------------- #
# GATE A: the leakage test
# ---------------------------------------------------------------------- #

_LEAK_N, _LEAK_H, _LEAK_K, _LEAK_SEED = 3000, 300, 15, 11
_FEATS = ["resolution_bar", "noise"]


def _leak_frame():
    """No genuine signal exists. The label is an i.i.d. coin flip at the
    bar where it resolves, so two samples resolving on the SAME bar share a
    label. Achievable AUC without leakage is 0.5 exactly."""
    rng = np.random.default_rng(_LEAK_SEED)
    e = rng.integers(0, 2, size=_LEAK_N + _LEAK_H + 1)
    h = rng.integers(1, _LEAK_H + 1, size=_LEAK_N)
    t0_pos = np.arange(_LEAK_N)
    t1_pos = t0_pos + h
    axis = pd.date_range("2020-01-01", periods=_LEAK_N + _LEAK_H + 1, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "resolution_bar": t1_pos.astype(float),
            "noise": rng.normal(size=_LEAK_N),
            "y": e[t1_pos].astype(int),
            "t0": axis[t0_pos],
            "t1": axis[t1_pos],
        },
        index=axis[t0_pos],
    )


def _auc(df, train_pos, test_pos):
    tr, te = df.iloc[train_pos], df.iloc[test_pos]
    if tr["y"].nunique() < 2 or te["y"].nunique() < 2:
        return float("nan")
    m = lgb.train(
        {"objective": "binary", "verbosity": -1, "learning_rate": 0.05,
         "num_leaves": 255, "min_child_samples": 2, "seed": _LEAK_SEED},
        lgb.Dataset(tr[_FEATS], label=tr["y"]),
        num_boost_round=120,
        callbacks=[lgb.log_evaluation(0)],
    )
    return float(roc_auc_score(te["y"], m.predict(te[_FEATS])))


def _shared(df, a_pos, b_pos):
    a = set(df.iloc[a_pos]["resolution_bar"].to_numpy())
    return int(np.isin(df.iloc[b_pos]["resolution_bar"].to_numpy(), list(a)).sum())


def test_gate_a_purging_closes_the_leak_channel_completely():
    """Deterministic half of the leak test: count test samples sharing a
    resolution bar with a training sample. Unpurged this is large; purged it
    must be exactly zero. No model, no noise, no thresholds."""
    df = _leak_frame()
    unpurged = list(PurgedKFold(_LEAK_K, purge=False).split_detailed(df))
    purged = list(PurgedKFold(_LEAK_K, embargo_pct=0.01, purge=True).split_detailed(df))

    leak_before = [_shared(df, f.train, f.test) for f in unpurged]
    leak_after = [_shared(df, f.train, f.test) for f in purged]

    assert sum(leak_before) > 0, "fixture failed to create a leak to close"
    assert sum(leak_after) == 0, f"purging left a leak channel open: {leak_after}"


def test_gate_a_unpurged_splitter_reports_skill_that_does_not_exist():
    """The statistical half. Truth is 0.5. The unpurged splitter must report
    a higher mean AUC than the purged one, and the purged one must sit
    closer to chance."""
    df = _leak_frame()
    unpurged = list(PurgedKFold(_LEAK_K, purge=False).split_detailed(df))
    purged = list(PurgedKFold(_LEAK_K, embargo_pct=0.01, purge=True).split_detailed(df))

    old = np.nanmean([_auc(df, f.train, f.test) for f in unpurged])
    new = np.nanmean([_auc(df, f.train, f.test) for f in purged])

    assert old > 0.5, f"fixture failed: unpurged AUC {old:.4f} shows no inflation"
    assert old - new > 0.005, f"purging did not reduce inflation: {old:.4f} -> {new:.4f}"
    assert abs(new - 0.5) < abs(old - 0.5), (
        f"purged AUC {new:.4f} is not closer to chance than unpurged {old:.4f}"
    )
