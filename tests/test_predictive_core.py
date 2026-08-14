"""
Regression tests for the surviving Predictive Core: triple-barrier
labelling, walk-forward splitting, and quantile regression.

STAGE 0 PRUNE: this file previously also covered the k-NN analog engine
(Component A), the Component B direction classifier's class-weighting
machinery, and an end-to-end trainer run asserting isotonic calibration
reports. All three components were deleted at Stage 0 per
MASTER_CONTEXT_v2 section 8, so their tests were deleted with them:

  - test_minimum_sample_gate_enforced                        (Component A)
  - test_deduplication_is_asset_aware_and_uses_real_time     (Component A)
  - test_predict_accepts_tz_aware_query_timestamp            (Component A)
  - test_deduplicate_matches_naive_reference_and_is_fast     (Component A)
  - test_compute_class_weights_matches_balanced_formula      (class weights)
  - test_compute_class_weights_handles_a_class_absent...     (class weights)
  - main() narrative script                                  (drove A and B)

The splitter and labelling tests below are NOT Component A/B tests and are
retained unchanged. `WalkForwardSplitter` itself is scheduled for
replacement at Stage A.3 (Purged K-Fold + embargo); its fold-isolation test
is kept until then so the replacement has a behavioural baseline to beat.
"""

import numpy as np

from src.predictive_core.labeling import apply_triple_barrier, pool_assets
from src.predictive_core.splitter import WalkForwardSplitter
from src.predictive_core.boosted_core import WalkForwardTrainer
from tests.synthetic_data import make_synthetic_asset, FEATURE_COLS


def build_pooled_dataset(n=2600, seed=1):
    btc_raw = make_synthetic_asset(n, "2022-01-01", "1h", seed=seed, drift=0.01)
    eth_raw = make_synthetic_asset(n, "2022-01-01", "1h", seed=seed + 1, drift=0.01)

    # Label EACH asset independently, before pooling (see labeling.py docstring).
    btc_labeled = apply_triple_barrier(btc_raw, max_candles=24)
    eth_labeled = apply_triple_barrier(eth_raw, max_candles=24)

    pooled = pool_assets({"BTC": btc_labeled, "ETH": eth_labeled})
    return pooled


def test_walk_forward_splitter_fold_isolation():
    pooled = build_pooled_dataset()
    splitter = WalkForwardSplitter(train_size=1600, val_size=400, test_size=400)
    folds = list(splitter.split(pooled))
    assert len(folds) >= 1

    for tr, va, te in folds:
        assert pooled.index[tr].max() <= pooled.index[va].min()
        assert pooled.index[va].max() <= pooled.index[te].min()
        assert set(tr.tolist()).isdisjoint(va.tolist())
        assert set(va.tolist()).isdisjoint(te.tolist())
        assert set(tr.tolist()).isdisjoint(te.tolist())


def test_apply_triple_barrier_records_label_horizon_in_attrs():
    """
    `apply_triple_barrier` must record the `max_candles` it actually used in
    `df.attrs['label_horizon']`, so downstream consumers can cross-check
    their own configured label_horizon against it instead of silently
    trusting a separately-configured number to agree.

    The original consumer (PredictiveCoreStrategy) was deleted at Stage 0.
    This contract is retained because Stage A.1 requires labelling to emit
    richer per-sample metadata (t0, t1, barrier_touched, realised_return),
    and `attrs['label_horizon']` is the existing precedent for that.
    """
    raw = make_synthetic_asset(500, "2022-01-01", "1h", seed=1, drift=0.01)
    labeled = apply_triple_barrier(raw, max_candles=24)
    assert labeled.attrs.get("label_horizon") == 24

    labeled_36 = apply_triple_barrier(raw, max_candles=36)
    assert labeled_36.attrs.get("label_horizon") == 36


def test_walk_forward_trainer_runs_quantile_regressors_end_to_end():
    """The walk-forward loop runs to completion across folds and produces
    real (non-NaN) pinball losses for both quantile targets.

    Previously this test also asserted the direction classifier's isotonic
    calibration reports; that half was deleted with the classifier. What
    remains asserts only the quantile heads, which section 8 did not name
    for deletion.
    """
    pooled = build_pooled_dataset(n=900, seed=7)
    splitter = WalkForwardSplitter(train_size=500, val_size=150, test_size=150)
    trainer = WalkForwardTrainer(
        splitter=splitter,
        feature_cols=FEATURE_COLS,
        categorical_cols=["asset"],
        n_optuna_trials=2,          # kept small so this test runs in seconds
        early_stopping_rounds=10,
        random_state=42,
    )
    results = trainer.run(pooled)
    assert len(results) >= 1

    for r in results:
        for q_result in (r.magnitude, r.duration):
            assert q_result.n_train > 0 and q_result.n_val > 0 and q_result.n_test > 0
            assert set(q_result.val_pinball_loss) == {0.1, 0.5, 0.9}
            for loss in q_result.val_pinball_loss.values():
                assert np.isfinite(loss)
            for loss in q_result.test_pinball_loss.values():
                assert np.isfinite(loss)

    assert results[0].magnitude.target_col == "magnitude"
    assert results[0].duration.target_col == "duration"
