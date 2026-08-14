"""
Tests for `scripts.run_layer_ablation`'s shared-fit machinery and its
verdict arithmetic.

The ablation's whole claim is that a Sharpe delta between two variants is
attributable to the LAYER that differs between them. Two things have to
hold for that:

  1. The shared Component A/B cache must return the SAME fitted models for
     the same fold, and must never serve one fold's models for a different
     fold (which would silently corrupt every downstream number).
  2. Each variant must fit its OWN ensemble weights -- sharing those across
     variants would leak one configuration's fit into another's result.
"""

import numpy as np
import pandas as pd

from scripts.run_layer_ablation import VARIANTS, _SharedFitStrategy, _format_comparison
from src.predictive_core.dataset import RAW_ATR_COLUMN

_FEATURES = ["f_a", "f_b", "f_c"]


def _frame(n=420, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    return pd.DataFrame(
        {
            "f_a": rng.uniform(0, 1, n), "f_b": rng.uniform(0, 1, n), "f_c": rng.uniform(0, 1, n),
            "close": close, "high": close + spread, "low": close - spread,
            RAW_ATR_COLUMN: np.abs(rng.normal(1.0, 0.15, n)) + 0.5,
            "vol_atr_pct_rank": rng.uniform(0, 1, n),
            "direction": rng.choice([-1.0, 0.0, 1.0], n, p=[0.45, 0.1, 0.45]),
            "magnitude": np.abs(rng.normal(1.5, 0.4, n)),
            "duration": rng.integers(1, 20, n).astype(float),
        },
        index=idx,
    )


def _kwargs(**overrides):
    base = dict(label_horizon=5, n_optuna_trials=1, early_stopping_rounds=5,
                random_state=7, min_direction_confidence=0.0)
    base.update(overrides)
    return base


def _split(df):
    n = len(df)
    return df.iloc[: int(n * 0.6)], df.iloc[int(n * 0.6) : int(n * 0.8)], df.iloc[int(n * 0.8) :]


# ------------------------------------------------------------------------ #
# Shared Component A/B cache
# ------------------------------------------------------------------------ #


def test_second_variant_reuses_the_cached_models_not_a_refit():
    """The point of the cache: identical Component A/B across variants, so a
    Sharpe delta is the layer, not LightGBM/Optuna training noise."""
    df = _frame()
    train, val, _ = _split(df)
    cache: dict = {}

    first = _SharedFitStrategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=False,
                               use_ensemble=False, shared_cache=cache, **_kwargs())
    first.fit(train, val)
    assert len(cache) == 1

    second = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=True,
                                use_ensemble=False, shared_cache=cache, **_kwargs())
    second.fit(train, val)
    assert len(cache) == 1  # a hit, not a second entry
    assert second._direction_model is first._direction_model
    assert second._analog_engine is first._analog_engine
    assert second._distance_threshold == first._distance_threshold


def test_a_different_fold_never_collides_with_a_cached_one():
    """The cache key must distinguish genuinely different partitions --
    serving fold A's models for fold B would corrupt everything downstream
    while still producing plausible-looking numbers."""
    df = _frame(n=600)
    cache: dict = {}
    fold_a = (df.iloc[:250], df.iloc[250:330])
    fold_b = (df.iloc[100:350], df.iloc[350:430])

    a = _SharedFitStrategy(_FEATURES, use_ensemble=False, shared_cache=cache, **_kwargs())
    a.fit(*fold_a)
    b = _SharedFitStrategy(_FEATURES, use_ensemble=False, shared_cache=cache, **_kwargs())
    b.fit(*fold_b)

    assert len(cache) == 2
    assert b._direction_model is not a._direction_model


def test_cache_key_distinguishes_same_endpoints_different_length():
    df = _frame(n=600)
    train, val, _ = _split(df)
    key_full = _SharedFitStrategy._fold_key(train, val)
    key_gappy = _SharedFitStrategy._fold_key(train.iloc[::2], val)
    assert key_full != key_gappy


# ------------------------------------------------------------------------ #
# Per-variant ensemble isolation
# ------------------------------------------------------------------------ #


def test_each_variant_fits_its_own_ensemble_weights():
    """Sharing ensemble weights across variants would leak one
    configuration's fit into another's measured result."""
    df = _frame(seed=2)
    train, val, _ = _split(df)
    cache: dict = {}

    with_trap = _SharedFitStrategy(
        _FEATURES, use_liquidity_trap=True, use_volatility_gate=False, use_ensemble=True,
        shared_cache=cache, shared_ensemble_holder={}, **_kwargs())
    with_trap.fit(train, val)

    without_trap = _SharedFitStrategy(
        _FEATURES, use_liquidity_trap=False, use_volatility_gate=False, use_ensemble=True,
        shared_cache=cache, shared_ensemble_holder={}, **_kwargs())
    without_trap.fit(train, val)

    assert with_trap._ensemble is not without_trap._ensemble
    assert with_trap.ensemble_weights() is not None
    assert without_trap.ensemble_weights() is not None


def test_ensemble_history_accumulates_across_folds_within_one_variant():
    """The harness builds a NEW strategy per fold via its factory, so the
    EnsembleMetaModel must live in the shared holder -- otherwise the
    Adaptive doc's cross-fold 0.15 stability rule could never fire."""
    df = _frame(seed=4, n=620)
    holder: dict = {}
    cache: dict = {}
    fold_a = (df.iloc[:250], df.iloc[250:340])
    fold_b = (df.iloc[90:340], df.iloc[340:430])

    s1 = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=False,
                            use_ensemble=True, shared_cache=cache,
                            shared_ensemble_holder=holder, **_kwargs())
    s1.fit(*fold_a)
    s2 = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=False,
                            use_ensemble=True, shared_cache=cache,
                            shared_ensemble_holder=holder, **_kwargs())
    s2.fit(*fold_b)

    assert s2._ensemble is s1._ensemble
    assert len(s2._ensemble._fold_weight_history) == 2


# ------------------------------------------------------------------------ #
# Verdict arithmetic
# ------------------------------------------------------------------------ #


def _result(variant, sharpe, trades=500, pf=1.0, dd=0.1, destroyed=False, under=None):
    return {"variant": variant, "oos_sharpe": sharpe, "oos_profit_factor": pf,
            "oos_max_drawdown": dd, "oos_trades": trades,
            "equity_destroyed": destroyed, "under_powered_folds": under or []}


def test_a_degrading_layer_is_flagged_not_silently_kept():
    """Marginal contribution = full stack MINUS leave-one-out. A layer whose
    removal HELPED is degrading and must be flagged."""
    results = [
        _result("baseline_phase1", -1.0),
        _result("all_layers", 0.30),
        _result("no_liquidity_trap_L5", 0.10),   # removing L5 hurt   -> L5 helps
        _result("no_volatility_gate_L6", 0.50),  # removing L6 helped -> L6 DEGRADES
        _result("no_ensemble_L7C", 0.30),        # removing L7C: no change
    ]
    text = _format_comparison(results)
    assert "Layer 6 volatility gate" in text
    assert "DEGRADES -- FLAG" in text
    flagged_line = [ln for ln in text.splitlines() if ln.startswith("FLAGGED")][0]
    assert "Layer 6 volatility gate" in flagged_line
    assert "Layer 5" not in flagged_line
    assert "Layer 7C" not in flagged_line


def test_no_degrading_layers_reports_none():
    results = [
        _result("baseline_phase1", -1.0), _result("all_layers", 0.5),
        _result("no_liquidity_trap_L5", 0.2), _result("no_volatility_gate_L6", 0.3),
        _result("no_ensemble_L7C", 0.1),
    ]
    flagged_line = [ln for ln in _format_comparison(results).splitlines() if ln.startswith("FLAGGED")][0]
    assert flagged_line.endswith("none")


# ------------------------------------------------------------------------ #
# 2026-08-14: solvency is checked BEFORE Sharpe is used for anything
# ------------------------------------------------------------------------ #


def test_destroyed_equity_variants_are_disqualified_not_ranked():
    """Sharpe over a non-positive equity curve is meaningless, not merely
    worse. It must not be printed or ranked."""
    results = [
        _result("baseline_phase1", -12.974, 3502, pf=0.236, dd=0.056),
        _result("all_layers", -8.284, 2817, pf=0.014, dd=1.023, destroyed=True),
    ]
    text = _format_comparison(results)
    assert "DISQUALIFIED (equity destroyed)" in text
    assert "-8.284" not in text          # the meaningless Sharpe is never shown
    assert "-12.974" in text             # the solvent one still is
    assert "NOT COMPARABLE on Sharpe" in text


def test_max_dd_at_or_above_one_disqualifies_even_without_the_flag():
    """maxDD >= 1.0 is itself the signature of equity reaching zero, so a
    result carrying it is disqualified even if the explicit flag is absent
    (e.g. reading an older report file)."""
    results = [_result("all_layers", 0.9, pf=0.5, dd=1.0, destroyed=False)]
    assert "DISQUALIFIED" in _format_comparison(results)


def test_the_real_2026_08_14_numbers_now_flag_layer7c_not_layer6():
    """Regression test against the actual failure. Sharpe-only ranking
    flagged Layer 6 (which touches 0.174% of bars, so it cannot genuinely
    move Sharpe) and called Layer 7C -- the layer that destroyed the
    account -- a help. Replaying the real numbers must now invert that."""
    results = [
        _result("baseline_phase1", -12.974, 3502, pf=0.236, dd=0.056),
        _result("all_layers", -8.284, 2817, pf=0.014, dd=1.023, destroyed=True),
        _result("no_liquidity_trap_L5", -8.284, 2817, pf=0.014, dd=1.023, destroyed=True),
        _result("no_volatility_gate_L6", -4.846, 2813, pf=0.008, dd=1.235, destroyed=True),
        _result("no_ensemble_L7C", -12.359, 3499, pf=0.238, dd=0.056),
    ]
    text = _format_comparison(results)
    flagged_line = [ln for ln in text.splitlines() if ln.startswith("FLAGGED")][0]
    assert "Layer 7C ensemble" in flagged_line
    assert "Layer 6" not in flagged_line          # no longer falsely flagged
    assert "adding it destroys the account" in text
    assert "profit factor 0.238 -> 0.014" in text
    assert "max drawdown 0.056 -> 1.023" in text


def test_under_powered_folds_are_disclosed_in_the_table():
    results = [_result("all_layers", 0.5, pf=1.2, dd=0.1, under=[3, 4])]
    assert "under-powered fold(s) [3, 4] excluded from Sharpe" in _format_comparison(results)


def test_comparison_reports_trade_counts_alongside_sharpe():
    """A layer can raise Sharpe purely by cutting the sample; MASTER_CONTEXT
    requires a minimum 200-trade OOS sample, so trade counts must be visible
    next to every Sharpe figure."""
    results = [
        _result("baseline_phase1", -1.0, trades=2000),
        _result("all_layers", 0.9, trades=12),
        _result("no_liquidity_trap_L5", 0.2, trades=900),
        _result("no_volatility_gate_L6", 0.3, trades=1500),
        _result("no_ensemble_L7C", 0.1, trades=800),
    ]
    text = _format_comparison(results)
    assert "trades" in text
    assert "12" in text
    assert "trades 900 -> 12" in text  # leave-one-out -> full stack, per layer


def test_comparison_states_layer14_not_tier1():
    text = _format_comparison([_result("baseline_phase1", 0.0), _result("all_layers", 0.0)])
    assert "Layer 14" in text
    assert "NOT the Tier 1 live gate" in text


def test_variant_matrix_is_a_proper_leave_one_out_design():
    labels = {v[0] for v in VARIANTS}
    assert {"baseline_phase1", "all_layers",
            "no_liquidity_trap_L5", "no_volatility_gate_L6", "no_ensemble_L7C"} <= labels
    by_label = {v[0]: v[1:] for v in VARIANTS}
    assert by_label["baseline_phase1"] == (False, False, False)
    assert by_label["all_layers"] == (True, True, True)
    # Each leave-one-out variant must differ from all_layers in exactly one flag.
    for label in ("no_liquidity_trap_L5", "no_volatility_gate_L6", "no_ensemble_L7C"):
        diffs = sum(1 for a, b in zip(by_label["all_layers"], by_label[label]) if a != b)
        assert diffs == 1, f"{label} differs from all_layers in {diffs} flags, not 1"


# ------------------------------------------------------------------------ #
# Prediction caching -- variant-invariant, and must not cross folds
# ------------------------------------------------------------------------ #


def test_component_predictions_are_cached_across_variants():
    """Component A/B predictions depend only on the fitted models and the
    data, never on which layers are enabled -- so the second variant must
    reuse them rather than re-running Component A's per-row predict()."""
    df = _frame(seed=6)
    train, val, test = _split(df)
    cache: dict = {}

    first = _SharedFitStrategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=False,
                               use_ensemble=False, shared_cache=cache, **_kwargs())
    first.fit(train, val)
    a1, c1, m1 = first._component_predictions(test)

    second = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=True,
                                use_ensemble=False, shared_cache=cache, **_kwargs())
    second.fit(train, val)
    a2, c2, m2 = second._component_predictions(test)

    assert a2 is a1 and c2 is c1 and m2 is m1  # same objects: a cache hit


def test_prediction_cache_never_serves_one_fold_to_another():
    """The cache key carries the fold, so different folds' models can never
    have their predictions confused -- that would corrupt every downstream
    number while still looking plausible."""
    df = _frame(n=620, seed=8)
    cache: dict = {}
    shared_test = df.iloc[430:520]

    a = _SharedFitStrategy(_FEATURES, use_ensemble=False, shared_cache=cache, **_kwargs())
    a.fit(df.iloc[:250], df.iloc[250:340])
    preds_a = a._component_predictions(shared_test)

    b = _SharedFitStrategy(_FEATURES, use_ensemble=False, shared_cache=cache, **_kwargs())
    b.fit(df.iloc[90:340], df.iloc[340:430])
    preds_b = b._component_predictions(shared_test)

    # Same partition, DIFFERENT fold -> must be a miss, not a hit.
    assert preds_b is not preds_a


def test_liquidity_trap_readings_are_cached_across_variants():
    df = _frame(seed=10)
    _, _, test = _split(df)
    cache: dict = {}

    s1 = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, shared_cache=cache, **_kwargs())
    r1 = s1._liquidity_trap_series(test)
    s2 = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, shared_cache=cache, **_kwargs())
    r2 = s2._liquidity_trap_series(test)
    assert r2 is r1


def test_disabled_liquidity_trap_still_returns_all_absent():
    df = _frame(seed=12)
    s = _SharedFitStrategy(_FEATURES, use_liquidity_trap=False, shared_cache={}, **_kwargs())
    out = s._liquidity_trap_series(df)
    assert len(out) == len(df) and all(x is None for x in out)


def test_caching_does_not_change_results():
    """The cache must be a pure speed optimisation: a cached run and an
    uncached run must produce identical positions."""
    df = _frame(seed=14)
    train, val, test = _split(df)

    warm_cache: dict = {}
    primer = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=True,
                                use_ensemble=False, shared_cache=warm_cache, **_kwargs())
    primer.fit(train, val)
    _ = primer.generate_positions(test)

    cached = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=True,
                                use_ensemble=False, shared_cache=warm_cache, **_kwargs())
    cached.fit(train, val)
    p_cached = cached.generate_positions(test)

    fresh = _SharedFitStrategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=True,
                               use_ensemble=False, shared_cache={}, **_kwargs())
    fresh.fit(train, val)
    p_fresh = fresh.generate_positions(test)

    pd.testing.assert_series_equal(p_cached, p_fresh)
