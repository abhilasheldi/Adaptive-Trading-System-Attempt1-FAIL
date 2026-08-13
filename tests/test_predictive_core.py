"""
End-to-end verification of the Predictive Core (Components A and B).

Run with: python3 -m tests.test_predictive_core   (from repo root)

This is deliberately written as a script with assertions, not pytest, so it
can be read top-to-bottom as a verification report: each section prints what
it checked and what it found.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.predictive_core.labeling import apply_triple_barrier, pool_assets
from src.predictive_core.splitter import WalkForwardSplitter
from src.predictive_core.analog_engine import HistoricalAnalogEngine
from src.predictive_core.boosted_core import WalkForwardTrainer, _compute_class_weights
from tests.synthetic_data import make_synthetic_asset, FEATURE_COLS


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_pooled_dataset(n=2600, seed=1):
    btc_raw = make_synthetic_asset(n, "2022-01-01", "1h", seed=seed, drift=0.01)
    eth_raw = make_synthetic_asset(n, "2022-01-01", "1h", seed=seed + 1, drift=0.01)

    # Label EACH asset independently, before pooling (see labeling.py docstring).
    btc_labeled = apply_triple_barrier(btc_raw, max_candles=24)
    eth_labeled = apply_triple_barrier(eth_raw, max_candles=24)

    pooled = pool_assets({"BTC": btc_labeled, "ETH": eth_labeled})
    return pooled


# ------------------------------------------------------------------------ #
# Pytest-discovered regression tests. `main()` below remains as a
# human-readable narrative smoke test; these are the automated equivalents
# so `pytest` actually exercises this module (previously it collected zero
# tests from this file since none were named `test_*`).
# ------------------------------------------------------------------------ #


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


def test_minimum_sample_gate_enforced():
    pooled = build_pooled_dataset()
    splitter = WalkForwardSplitter(train_size=1600, val_size=400, test_size=400)
    tr0, va0, _te0 = next(splitter.split(pooled))
    train_df = pooled.iloc[tr0]
    query_df = pooled.iloc[va0]

    engine = HistoricalAnalogEngine(
        feature_cols=FEATURE_COLS, min_independent_instances=50, min_temporal_separation=24
    )
    engine.fit(train_df)
    query_vector = query_df[FEATURE_COLS].iloc[0].to_numpy()

    # Starved of candidates -> gate must refuse and every probability field
    # must be None (AnalogForecast.is_valid is the sole source of truth).
    gated = engine.predict(query_vector, distance_threshold=1e-9)
    assert gated.is_valid is False
    assert gated.n_independent < 50
    assert gated.probability_up is None
    assert gated.probability_down is None
    assert gated.probability_neutral is None

    # Whatever threshold is used, the gate outcome and the presence/absence
    # of probabilities must always agree with n_independent >= 50 -- this
    # must hold regardless of which branch is hit.
    threshold = engine.suggest_distance_threshold(percentile=30.0)
    forecast = engine.predict(query_vector, distance_threshold=threshold)
    assert forecast.is_valid == (forecast.n_independent >= 50)
    if forecast.is_valid:
        assert forecast.probability_up is not None
        assert forecast.probability_down is not None
        assert forecast.probability_neutral is not None
    else:
        assert forecast.probability_up is None
        assert forecast.probability_down is None
        assert forecast.probability_neutral is None

    # Reaching into internals to compute a probability directly, bypassing
    # predict() entirely, must not be possible via any public/documented
    # method -- predict() is the only method that returns an AnalogForecast.
    assert not hasattr(engine, "forecast_unchecked")
    assert not hasattr(engine, "predict_raw")


def test_deduplication_is_asset_aware_and_uses_real_time():
    """
    Regression test for a bug where independence enforcement used raw row
    position in the pooled (BTC+ETH interleaved) reference frame as a proxy
    for temporal separation. That silently did two wrong things at once:
    (1) folded candidates from DIFFERENT assets into the same "move" purely
    because they landed on nearby rows in the interleaved frame, and
    (2) treated genuinely-overlapping same-asset windows as independent,
    because N interleaved rows apart is only N / num_assets real candles
    apart. Both are direct violations of MASTER_CONTEXT constraint 9
    ("overlapping-window instances from the same directional move counted
    as one, not many"). Fixed by deduplicating on real elapsed time
    (`_ref_timestamps`) plus asset identity (`_ref_asset`) instead of row
    position.
    """
    pooled = build_pooled_dataset()
    train_df = pooled.iloc[:1600]
    engine = HistoricalAnalogEngine(
        feature_cols=FEATURE_COLS,
        min_independent_instances=1,
        min_temporal_separation=24,
        bar_interval=pd.Timedelta(hours=1),
    )
    engine.fit(train_df)
    query_vector = pooled.iloc[1600][FEATURE_COLS].to_numpy()
    threshold = engine.suggest_distance_threshold(percentile=30.0)

    distances = engine._weighted_distances(np.asarray(query_vector, dtype=float))
    candidate_idx = np.where(distances <= threshold)[0]
    order = candidate_idx[np.argsort(distances[candidate_idx])]
    accepted = engine._deduplicate(order)
    assert len(accepted) > 0, "test threshold too tight to exercise dedup -- widen percentile"

    assets = train_df["asset"].to_numpy()
    ts = train_df.index.to_numpy()
    required_gap = np.timedelta64(24, "h")

    same_asset_violations = [
        (accepted[i], accepted[j])
        for i in range(len(accepted))
        for j in range(i + 1, len(accepted))
        if assets[accepted[i]] == assets[accepted[j]]
        and abs(ts[accepted[i]] - ts[accepted[j]]) < required_gap
    ]
    assert same_asset_violations == [], (
        f"{len(same_asset_violations)} same-asset pairs both counted independent "
        f"despite being < 24 real hours apart"
    )

    # Cross-asset candidates that are extremely close in real time must both
    # survive dedup (never folded together) -- proves the independence check
    # never conflates two different instruments' instances.
    cross_asset_close_pairs = [
        (i, j)
        for i in accepted
        for j in accepted
        if i != j and assets[i] != assets[j] and abs(ts[i] - ts[j]) < np.timedelta64(1, "h")
    ]
    assert len(cross_asset_close_pairs) > 0, (
        "expected at least one same-timestamp cross-asset pair in this synthetic "
        "dataset; if none are present this test is not exercising the regression "
        "it exists to guard"
    )


def test_predict_accepts_tz_aware_query_timestamp():
    """
    Regression test: `HistoricalAnalogEngine.predict(..., query_timestamp=...)`
    used to raise `TypeError: can't compare offset-naive and offset-aware
    datetimes` whenever fed tz-aware data -- real crypto data from
    `data_engine.crypto_fetcher` is always UTC tz-aware (`pd.to_datetime(...,
    utc=True)`), while every other test in this file (and `synthetic_data.py`)
    uses tz-naive timestamps, so this was never exercised until real data was
    run through `PredictiveCoreStrategy.generate_positions()`, which passes
    `query_timestamp` on every single row -- i.e. it crashed on 100% of real
    (tz-aware) queries. Fixed by comparing via `_ref_timestamps_ns`/`_to_ns`
    (both normalized to UTC-based int64 ns) instead of raw
    `np.datetime64(query_timestamp)` against a tz-aware object array.
    """
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "direction": rng.choice([-1.0, 0.0, 1.0], size=n),
        },
        index=idx,
    )
    engine = HistoricalAnalogEngine(
        feature_cols=["f1", "f2"], min_independent_instances=1, min_temporal_separation=4
    )
    engine.fit(df.iloc[:150])
    query_vector = df[["f1", "f2"]].iloc[150].to_numpy()
    query_timestamp = df.index[150]

    # Must not raise. A tz-naive query_timestamp against tz-aware reference
    # data must also work (both normalize to the same UTC-based ns scale).
    forecast = engine.predict(query_vector, distance_threshold=10.0, query_timestamp=query_timestamp)
    assert forecast.n_raw_candidates >= 0  # ran to completion, didn't raise

    naive_ts = pd.Timestamp(query_timestamp).tz_convert("UTC").tz_localize(None)
    forecast_naive_query = engine.predict(query_vector, distance_threshold=10.0, query_timestamp=naive_ts)
    assert forecast.n_raw_candidates == forecast_naive_query.n_raw_candidates


def _naive_deduplicate_reference(engine, order):
    """
    Ground-truth reference: the original O(k^2) nested-Python-loop
    implementation of `_deduplicate`, kept here (not in production code)
    purely so this test can assert the optimized `bisect`-based
    implementation in analog_engine.py is exactly equivalent to it, not
    just "fast." See project_state.md: `_deduplicate` was ~100ms/call at
    realistic reference-set size (~5,000 rows) due to this nested loop
    (293k Python-level `abs()` calls across 20 queries), which made a real
    walk-forward run take tens of minutes to hours.
    """
    accepted = []
    accepted_ts = []
    accepted_asset = []
    gap = engine._min_temporal_gap
    for idx in order:
        ts = engine._ref_timestamps[idx]
        asset = engine._ref_asset[idx] if engine._ref_asset is not None else None
        collides = False
        for a_ts, a_asset in zip(accepted_ts, accepted_asset):
            if asset is not None and a_asset is not None and asset != a_asset:
                continue
            if abs(ts - a_ts) < gap:
                collides = True
                break
        if not collides:
            accepted.append(int(idx))
            accepted_ts.append(ts)
            accepted_asset.append(asset)
    return accepted


def test_deduplicate_matches_naive_reference_and_is_fast():
    """
    Regression test for the `_deduplicate` performance bug. Two things must
    both hold:
    (1) The optimized (per-asset sorted-bucket + bisect) implementation
        produces byte-identical accepted-index lists to the original O(k^2)
        nested-loop implementation, across randomized separation thresholds
        and acceptance orders -- this is a pure performance change, not a
        behavioral one.
    (2) It actually runs fast at a candidate-pool size representative of a
        real walk-forward fold (~2,000+ candidates), not just on toy inputs
        -- a generous but regression-catching bound: 200 calls in under 5
        seconds (the old implementation took ~100ms/call at this scale,
        i.e. ~20s for 200 calls; the fixed one is expected in the
        low-single-digit ms/call range).
    """
    pooled = build_pooled_dataset(n=3000, seed=11)
    engine = HistoricalAnalogEngine(
        feature_cols=FEATURE_COLS, min_independent_instances=1, min_temporal_separation=48
    )
    engine.fit(pooled.iloc[:5000] if len(pooled) > 5000 else pooled)

    rng = np.random.default_rng(3)
    n_ref = engine._ref_features.shape[0]

    # Equivalence check across randomized separation thresholds and orders.
    for trial in range(10):
        sep = int(rng.integers(1, 96))
        engine.min_temporal_separation = sep
        engine._min_temporal_gap = engine.bar_interval * sep
        order = rng.permutation(n_ref)[: rng.integers(50, min(500, n_ref))]
        assert engine._deduplicate(order) == _naive_deduplicate_reference(engine, order)

    # Performance check at realistic candidate-pool scale.
    engine.min_temporal_separation = 48
    engine._min_temporal_gap = engine.bar_interval * 48
    query_vector = pooled[FEATURE_COLS].iloc[-1].to_numpy()
    threshold = engine.suggest_distance_threshold(percentile=50.0)
    distances = engine._weighted_distances(np.asarray(query_vector, dtype=float))
    candidate_idx = np.where(distances <= threshold)[0]
    order = candidate_idx[np.argsort(distances[candidate_idx])]
    assert len(order) >= 500, "candidate pool too small to exercise the performance regression"

    import time

    t0 = time.perf_counter()
    for _ in range(200):
        engine._deduplicate(order)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"_deduplicate took {elapsed:.2f}s for 200 calls over {len(order)} candidates -- "
        f"this is the O(k^2) performance regression re-appearing."
    )


def test_apply_triple_barrier_records_label_horizon_in_attrs():
    """
    Regression/feature test: `apply_triple_barrier` must record the
    `max_candles` it actually used in `df.attrs['label_horizon']`, so
    downstream consumers (PredictiveCoreStrategy) can cross-check their own
    configured label_horizon against it instead of silently trusting a
    separately-configured number to agree.
    """
    raw = make_synthetic_asset(500, "2022-01-01", "1h", seed=1, drift=0.01)
    labeled = apply_triple_barrier(raw, max_candles=24)
    assert labeled.attrs.get("label_horizon") == 24

    labeled_36 = apply_triple_barrier(raw, max_candles=36)
    assert labeled_36.attrs.get("label_horizon") == 36


def test_compute_class_weights_matches_balanced_formula():
    """
    Locks in weight_i = n_samples / (n_classes * class_count[y_i]) -- the
    exact formula added to counteract the direction classifier's observed
    tendency to parrot the majority class under MASTER_CONTEXT's skewed
    real-data label split (see boosted_core._compute_class_weights
    docstring). Uses a 6/1/3 split (mirrors the shape of the real ~65/1/34
    imbalance) so the minority class's upweighting is unambiguous.
    """
    y_train = np.array([0] * 6 + [1] * 1 + [2] * 3)
    weights = _compute_class_weights(y_train, n_classes=3)

    n_samples = 10
    expected_w0 = n_samples / (3 * 6)
    expected_w1 = n_samples / (3 * 1)
    expected_w2 = n_samples / (3 * 3)

    assert np.allclose(weights[y_train == 0], expected_w0)
    assert np.allclose(weights[y_train == 1], expected_w1)
    assert np.allclose(weights[y_train == 2], expected_w2)
    # Rarest class (1 sample) must end up weighted above the mid-frequency
    # class (3 samples), which in turn must be weighted above the majority
    # class (6 samples) -- weight is strictly decreasing in class frequency.
    assert expected_w1 > expected_w2 > expected_w0


def test_compute_class_weights_handles_a_class_absent_from_this_fold():
    """A class with zero samples in a given fold's y_train must not raise
    (division-by-zero guard) -- plausible in real walk-forward folds where
    'neutral' (<1% of the real label distribution) can be entirely absent
    from a short fold."""
    y_train = np.array([0, 0, 0, 2, 2])  # class 1 (neutral) entirely absent
    weights = _compute_class_weights(y_train, n_classes=3)
    assert np.all(np.isfinite(weights))
    assert len(weights) == len(y_train)


def test_walk_forward_trainer_runs_end_to_end_with_calibration():
    """Covers both remaining 'done when' criteria: the walk-forward CV loop
    runs to completion across multiple folds, and calibration is checked
    and reports real (non-stub, non-NaN) numbers for every class."""
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
        assert r.direction.n_train > 0 and r.direction.n_val > 0 and r.direction.n_test > 0
        assert np.isfinite(r.direction.val_logloss)
        assert np.isfinite(r.direction.test_logloss)

        assert len(r.direction.calibration_reports) == 3
        for cr in r.direction.calibration_reports:
            assert isinstance(cr.expected_calibration_error, float)
            assert not np.isnan(cr.expected_calibration_error)
            assert 0.0 <= cr.expected_calibration_error <= 1.0
            assert isinstance(cr.brier_score, float)
            assert not np.isnan(cr.brier_score)
            assert cr.n_samples == r.direction.n_test


def main():
    # ------------------------------------------------------------------ #
    section("1. Building pooled BTC+ETH labeled dataset")
    pooled = build_pooled_dataset()
    print(f"pooled shape: {pooled.shape}")
    print(f"index unique: {pooled.index.is_unique}  (expected False -- BTC/ETH share timestamps)")
    print(f"asset dtype: {pooled['asset'].dtype}")
    print("direction class balance:")
    print(pooled["direction"].value_counts())
    assert set(pooled["direction"].unique()) <= {-1.0, 0.0, 1.0}
    assert pooled["asset"].dtype.name == "category"

    # ------------------------------------------------------------------ #
    section("2. Walk-forward splitter: fold isolation")
    splitter = WalkForwardSplitter(train_size=1600, val_size=400, test_size=400)
    folds = list(splitter.split(pooled))
    print(f"number of folds generated: {len(folds)}")
    assert len(folds) >= 1, "Splitter produced zero folds -- dataset too small for configured windows."

    for i, (tr, va, te) in enumerate(folds):
        train_max_ts = pooled.index[tr].max()
        val_min_ts = pooled.index[va].min()
        val_max_ts = pooled.index[va].max()
        test_min_ts = pooled.index[te].min()
        print(f"  fold {i}: train n={len(tr)} (<= {train_max_ts}) | "
              f"val n={len(va)} ({val_min_ts} .. {val_max_ts}) | "
              f"test n={len(te)} (>= {test_min_ts})")
        assert train_max_ts <= val_min_ts
        assert val_max_ts <= test_min_ts
        assert set(tr.tolist()).isdisjoint(set(va.tolist()))
        assert set(va.tolist()).isdisjoint(set(te.tolist()))
        assert set(tr.tolist()).isdisjoint(set(te.tolist()))
    print("PASS: every fold is strictly time-ordered and position-disjoint.")

    # Also prove the loc-vs-iloc duplicate-timestamp hazard is closed off:
    # positional slicing must be used, and label-based slicing on this index
    # would over-select rows (regression guard for the fix made to splitter.py).
    tr0, va0, te0 = folds[0]
    train_labels = pooled.index[tr0]
    loc_based = pooled.loc[train_labels]
    iloc_based = pooled.iloc[tr0]
    print(f"regression guard -- iloc-based train rows: {len(iloc_based)}, "
          f"loc-based (if it were used) would return: {len(loc_based)}")
    assert len(iloc_based) == len(tr0)

    # ------------------------------------------------------------------ #
    section("3. Component A -- k-NN Historical Analog Engine")
    train_df = pooled.iloc[tr0]
    query_df = pooled.iloc[va0]

    engine = HistoricalAnalogEngine(
        feature_cols=FEATURE_COLS,
        min_independent_instances=50,
        min_temporal_separation=24,
    )
    engine.fit(train_df)  # fit on TRAIN partition of fold 0 only
    threshold = engine.suggest_distance_threshold(percentile=10.0)
    print(f"data-driven distance threshold (derived from TRAIN fold only): {threshold:.4f}")

    query_vector = query_df[FEATURE_COLS].iloc[0].to_numpy()
    forecast = engine.predict(query_vector, distance_threshold=threshold)
    print("forecast for a validation-fold query point:")
    print(f"  {forecast.as_report()}")
    print(f"  -> N reported: {forecast.n_independent}, threshold reported: "
          f"{forecast.distance_threshold}, probability reported: "
          f"up={forecast.probability_up}")

    section("3b. Minimum-sample gate: verified impossible to bypass")
    tiny_threshold = 1e-9
    gated = engine.predict(query_vector, distance_threshold=tiny_threshold)
    assert gated.is_valid is False
    assert gated.probability_up is None and gated.probability_down is None and gated.probability_neutral is None
    print(f"gate correctly refused a forecast with only {gated.n_independent} independent "
          f"instances (< 50): is_valid={gated.is_valid}, probabilities=None")

    # Attempt bypass #1: directly read internal reference arrays to compute
    # a probability without going through predict().
    raw_dirs = engine._ref_direction  # noqa: SLF001 -- deliberately reaching for internals
    print(f"bypass attempt #1 (reach into _ref_direction directly): this returns raw "
          f"unfiltered label data ({len(raw_dirs)} rows), NOT a gated probability -- "
          f"there is no method that turns this into an AnalogForecast without "
          f"re-running distance filtering + dedup + the N>=50 check in predict().")

    # Attempt bypass #2: instantiate a second engine with min_independent_instances=1
    # to see if that's somehow the "real" gate living elsewhere. It isn't --
    # it's a constructor parameter, and dropping it to 1 is not a bypass, it's
    # a different (documented) configuration, and it's visible in the report.
    lenient_engine = HistoricalAnalogEngine(feature_cols=FEATURE_COLS, min_independent_instances=1,
                                             min_temporal_separation=24)
    lenient_engine.fit(train_df)
    lenient_forecast = lenient_engine.predict(query_vector, distance_threshold=threshold)
    print(f"bypass attempt #2 (construct engine with min_independent_instances=1): "
          f"succeeds ({lenient_forecast.is_valid}), but this is an explicit, visible "
          f"constructor argument -- not a silent code path around the gate. The "
          f"default and the spec-mandated engine both use 50.")

    # ------------------------------------------------------------------ #
    section("4. Component B -- walk-forward trainer (fast config for this smoke test)")
    trainer_splitter = WalkForwardSplitter(train_size=1600, val_size=400, test_size=400)
    trainer = WalkForwardTrainer(
        splitter=trainer_splitter,
        feature_cols=FEATURE_COLS,
        categorical_cols=["asset"],
        n_optuna_trials=5,          # kept small for smoke-test runtime; raise for production
        early_stopping_rounds=20,
        random_state=42,
    )
    results = trainer.run(pooled)
    print(f"folds trained: {len(results)}")

    for r in results:
        print(f"\n--- Fold {r.fold} ---")
        print(f"  n_train={r.direction.n_train} n_val={r.direction.n_val} n_test={r.direction.n_test}")
        print(f"  direction: val_logloss={r.direction.val_logloss:.4f} "
              f"test_logloss={r.direction.test_logloss:.4f}")
        for cr in r.direction.calibration_reports:
            print(f"    calibration[{cr.class_label}]: ECE={cr.expected_calibration_error:.4f} "
                  f"Brier={cr.brier_score:.4f} (n={cr.n_samples})")
        top_feats = sorted(r.direction.feature_importance.items(), key=lambda kv: -kv[1])[:3]
        print(f"  top feature importances (-> feed into Component A weights): {top_feats}")

        print(f"  magnitude quantile pinball loss (test): {r.magnitude.test_pinball_loss}")
        print(f"  magnitude quantile-crossing rate (test): {r.magnitude.quantile_crossing_rate:.4f}")
        print(f"  duration quantile pinball loss (test): {r.duration.test_pinball_loss}")
        print(f"  duration quantile-crossing rate (test): {r.duration.quantile_crossing_rate:.4f}")

        # Calibration must produce an actual number, not a stub.
        for cr in r.direction.calibration_reports:
            assert isinstance(cr.expected_calibration_error, float)
            assert not np.isnan(cr.expected_calibration_error)

    # ------------------------------------------------------------------ #
    section("5. Plugging Component B feature importances into Component A weights")
    best_fold = results[0]
    weights = [best_fold.direction.feature_importance[c] for c in FEATURE_COLS]
    engine.set_weights(weights)
    forecast_weighted = engine.predict(query_vector, distance_threshold=threshold)
    print(f"weights now set from Component B gain importance: {weights}")
    print(f"re-queried forecast with learned weights: {forecast_weighted.as_report()}")

    section("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()