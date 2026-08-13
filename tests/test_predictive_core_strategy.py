"""
Tests for `PredictiveCoreStrategy.diagnostics()` -- the plumbing that
surfaces Component B's calibration/discrimination/skill-score numbers
(computed during fit() but previously discarded immediately afterward) so a
`WalkForwardBacktester` report can actually show them per fold -- and for
`PredictiveCoreStrategy.fit()`'s label_horizon cross-check against the
`max_candles` `apply_triple_barrier` recorded upstream in `.attrs`.
"""

import math

import pytest

from src.backtesting.predictive_core_strategy import PredictiveCoreStrategy
from src.predictive_core.labeling import apply_triple_barrier
from tests.synthetic_data import FEATURE_COLS, make_synthetic_asset


def _make_labeled_single_asset(n: int, seed: int, max_candles: int = 24):
    raw = make_synthetic_asset(n, "2022-01-01", "1h", seed=seed, drift=0.01)
    return apply_triple_barrier(raw, max_candles=max_candles)


def test_diagnostics_is_none_before_fit():
    strategy = PredictiveCoreStrategy(feature_cols=FEATURE_COLS, label_horizon=24, n_optuna_trials=2,
                                       early_stopping_rounds=5, random_state=42)
    assert strategy.diagnostics() is None


def test_diagnostics_populated_after_fit():
    labeled = _make_labeled_single_asset(1200, seed=5)
    train_df = labeled.iloc[:800]
    val_df = labeled.iloc[800:1000]

    strategy = PredictiveCoreStrategy(feature_cols=FEATURE_COLS, label_horizon=24, n_optuna_trials=2,
                                       early_stopping_rounds=5, random_state=42)
    strategy.fit(train_df, val_df)

    diag = strategy.diagnostics()
    assert diag is not None
    assert set(diag.keys()) == {
        "direction_val_logloss", "direction_val_partition_logloss",
        "direction_val_partition_logloss_baseline", "direction_val_partition_logloss_skill_score",
        "direction_val_partition_calibration", "direction_val_partition_discrimination",
    }
    # Field names must make the partition explicit: fit() passes val_purged
    # as BOTH val_df and the required test_df argument to
    # train_direction_classifier (the real test partition must never reach
    # fit()), so anything derived from that placeholder evaluation is NOT
    # out-of-sample and must not be named as if it were.
    assert not any(k.startswith("direction_test_") for k in diag)
    assert len(diag["direction_val_partition_discrimination"]) == 3
    assert len(diag["direction_val_partition_calibration"]) == 3
    for d in diag["direction_val_partition_discrimination"]:
        assert d["class_label"] in ("down", "neutral", "up")
        assert 0.0 <= d["precision"] <= 1.0
        assert 0.0 <= d["recall"] <= 1.0
    # Skill score must be a real, finite number -- not NaN/None -- so a
    # caller can actually compare it against 0 without a separate check.
    skill = diag["direction_val_partition_logloss_skill_score"]
    assert not math.isnan(skill)
    assert isinstance(skill, float)


def test_fit_raises_on_label_horizon_mismatch():
    """
    The exact failure mode this cross-check exists to catch: data labeled
    upstream with apply_triple_barrier(max_candles=24), but the strategy
    configured with a DIFFERENT label_horizon (48). Must raise loudly and
    name both numbers, rather than silently under-purging the fold
    boundary by 24 rows.
    """
    labeled = _make_labeled_single_asset(1200, seed=6, max_candles=24)
    train_df = labeled.iloc[:800]
    val_df = labeled.iloc[800:1000]
    assert train_df.attrs.get("label_horizon") == 24  # sanity: metadata survived slicing

    strategy = PredictiveCoreStrategy(feature_cols=FEATURE_COLS, label_horizon=48, n_optuna_trials=2,
                                       early_stopping_rounds=5, random_state=42)
    with pytest.raises(ValueError, match="label_horizon mismatch"):
        strategy.fit(train_df, val_df)


def test_fit_succeeds_when_label_horizon_matches():
    """Sibling case: identical setup but label_horizon correctly matches
    max_candles -- must NOT raise."""
    labeled = _make_labeled_single_asset(1200, seed=7, max_candles=24)
    train_df = labeled.iloc[:800]
    val_df = labeled.iloc[800:1000]

    strategy = PredictiveCoreStrategy(feature_cols=FEATURE_COLS, label_horizon=24, n_optuna_trials=2,
                                       early_stopping_rounds=5, random_state=42)
    strategy.fit(train_df, val_df)  # must not raise
    assert strategy.diagnostics() is not None


def test_fit_skips_check_when_upstream_metadata_absent():
    """
    If df.attrs doesn't carry 'label_horizon' -- e.g. data prepared by
    something other than apply_triple_barrier, or metadata lost to an
    operation pandas doesn't propagate .attrs through -- the check must be
    SKIPPED, not treated as a mismatch. A genuinely mismatched
    label_horizon={48} against data that was actually labeled with
    max_candles=24 (but whose metadata got stripped here to simulate the
    "can't check" case) must NOT raise, since absence of metadata is not
    evidence of a mismatch.
    """
    labeled = _make_labeled_single_asset(1200, seed=8, max_candles=24)
    labeled.attrs.clear()
    assert labeled.attrs == {}
    train_df = labeled.iloc[:800]
    val_df = labeled.iloc[800:1000]

    strategy = PredictiveCoreStrategy(feature_cols=FEATURE_COLS, label_horizon=48, n_optuna_trials=2,
                                       early_stopping_rounds=5, random_state=42)
    strategy.fit(train_df, val_df)  # must not raise -- nothing to cross-check against
