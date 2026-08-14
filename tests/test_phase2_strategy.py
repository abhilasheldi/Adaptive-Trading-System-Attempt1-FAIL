"""
Tests for `backtesting.phase2_strategy.Phase2Strategy` -- the Phase 2
partial integration (Layer 5 liquidity trap, Layer 6 volatility gate,
Layer 7C ensemble) used for the walk-forward layer ablation.

The single most important test here is
`test_all_layers_off_matches_phase1_baseline`: unless the all-flags-off
configuration is behaviourally identical to `PredictiveCoreStrategy`, an
ablation result of "layer X moved OOS Sharpe by N" is not attributable to
layer X -- it could be incidental drift between two separately-written
strategies. Everything else in the ablation rests on that equivalence.

Uses small synthetic data with a tiny Optuna budget throughout: these test
behaviour and wiring, not predictive performance.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.phase2_strategy import Phase2Strategy
from src.backtesting.predictive_core_strategy import PredictiveCoreStrategy
from src.macro_mtf_liquidity_vol.liquidity_trap import EQUAL_LEVEL_LOOKBACK
from src.predictive_core.dataset import RAW_ATR_COLUMN

_FEATURES = ["f_a", "f_b", "f_c"]
_LABEL_HORIZON = 5


def _frame(n=420, seed=0) -> pd.DataFrame:
    """Synthetic modeling frame with every column the strategy needs."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    df = pd.DataFrame(
        {
            "f_a": rng.uniform(0, 1, n),
            "f_b": rng.uniform(0, 1, n),
            "f_c": rng.uniform(0, 1, n),
            "close": close,
            "high": close + spread,
            "low": close - spread,
            RAW_ATR_COLUMN: np.abs(rng.normal(1.0, 0.15, n)) + 0.5,
            "vol_atr_pct_rank": rng.uniform(0, 1, n),
            "direction": rng.choice([-1.0, 0.0, 1.0], n, p=[0.45, 0.1, 0.45]),
            "magnitude": np.abs(rng.normal(1.5, 0.4, n)),
            "duration": rng.integers(1, 20, n).astype(float),
        },
        index=idx,
    )
    return df


def _split(df):
    n = len(df)
    return df.iloc[: int(n * 0.6)], df.iloc[int(n * 0.6) : int(n * 0.8)], df.iloc[int(n * 0.8) :]


def _kwargs(**overrides):
    base = dict(
        label_horizon=_LABEL_HORIZON,
        n_optuna_trials=1,
        early_stopping_rounds=5,
        random_state=7,
        min_direction_confidence=0.0,  # keep positions flowing on random data
    )
    base.update(overrides)
    return base


# ------------------------------------------------------------------------ #
# Baseline equivalence -- the foundation of the whole ablation
# ------------------------------------------------------------------------ #


def test_all_layers_off_matches_phase1_baseline():
    """With every Phase 2 flag off, Phase2Strategy must produce the exact
    same position series as PredictiveCoreStrategy on identical data.
    Without this, no ablation delta is attributable to a specific layer."""
    df = _frame()
    train, val, test = _split(df)

    baseline = PredictiveCoreStrategy(_FEATURES, **_kwargs())
    baseline.fit(train, val)
    baseline_positions = baseline.generate_positions(test)

    ablated = Phase2Strategy(
        _FEATURES,
        use_liquidity_trap=False, use_volatility_gate=False, use_ensemble=False,
        **_kwargs(),
    )
    ablated.fit(train, val)
    ablated_positions = ablated.generate_positions(test)

    pd.testing.assert_series_equal(baseline_positions, ablated_positions)


def test_baseline_configuration_reports_no_ensemble_weights():
    """All-off must leave the Layer 14 ensemble-stability criterion at
    not_evaluated, exactly as the Phase 1 adapter does."""
    df = _frame()
    train, val, _ = _split(df)
    s = Phase2Strategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=False,
                       use_ensemble=False, **_kwargs())
    s.fit(train, val)
    assert s.ensemble_weights() is None


# ------------------------------------------------------------------------ #
# Layer 6: volatility gate
# ------------------------------------------------------------------------ #


def test_volatility_gate_can_only_remove_positions_never_add():
    """A gate is subtractive by construction: every row the gated run trades
    must also be traded by the ungated run, at the same size."""
    df = _frame(seed=3)
    train, val, test = _split(df)

    ungated = Phase2Strategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=False,
                             use_ensemble=False, **_kwargs())
    ungated.fit(train, val)
    p_ungated = ungated.generate_positions(test)

    gated = Phase2Strategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=True,
                           use_ensemble=False, **_kwargs())
    gated.fit(train, val)
    p_gated = gated.generate_positions(test)

    traded = p_gated != 0.0
    pd.testing.assert_series_equal(p_gated[traded], p_ungated[traded])
    assert (p_gated != 0).sum() <= (p_ungated != 0).sum()


def test_spike_rows_are_blocked_by_the_gate():
    """Engineer a genuine ATR doubling and confirm the gate zeroes it."""
    df = _frame(seed=5)
    train, val, test = _split(df)
    test = test.copy()
    # Force a run-up in the raw ATR window ending at a specific row.
    target = 30
    test.iloc[target - 5 : target + 1, test.columns.get_loc(RAW_ATR_COLUMN)] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 3.0
    ]

    s = Phase2Strategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=True,
                       use_ensemble=False, **_kwargs())
    s.fit(train, val)
    assert bool(s._spike_mask(test)[target]) is True
    assert s.generate_positions(test).iloc[target] == 0.0


def test_gate_treats_missing_volatility_data_as_not_spike():
    """Documented fail-open choice for a research ablation: a missing
    reading is not evidence of a tail-risk event, and wrongly suppressing
    trades would corrupt the Sharpe comparison being measured."""
    df = _frame(seed=9)
    train, val, test = _split(df)
    test = test.copy()
    test.iloc[10, test.columns.get_loc(RAW_ATR_COLUMN)] = float("nan")

    s = Phase2Strategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=True,
                       use_ensemble=False, **_kwargs())
    s.fit(train, val)
    assert bool(s._spike_mask(test)[10]) is False  # no crash, not flagged


# ------------------------------------------------------------------------ #
# Layer 5: liquidity trap
# ------------------------------------------------------------------------ #


def test_liquidity_trap_series_is_absent_during_warmup_then_present():
    df = _frame(seed=11)
    s = Phase2Strategy(_FEATURES, use_liquidity_trap=True, **_kwargs())
    results = s._liquidity_trap_series(df)
    assert all(r is None for r in results[:EQUAL_LEVEL_LOOKBACK])
    assert any(r is not None for r in results[EQUAL_LEVEL_LOOKBACK:])


def test_liquidity_trap_is_point_in_time_no_lookahead():
    """Mutating a FUTURE bar must not change any earlier row's reading."""
    df = _frame(seed=13)
    s = Phase2Strategy(_FEATURES, use_liquidity_trap=True, **_kwargs())
    baseline = s._liquidity_trap_series(df)

    cut = 200
    mutated = df.copy()
    mutated.iloc[cut + 5, mutated.columns.get_loc("high")] *= 5.0
    after = s._liquidity_trap_series(mutated)

    for i in range(cut + 1):
        a, b = baseline[i], after[i]
        assert (a is None) == (b is None)
        if a is not None:
            assert a.trap_probability == pytest.approx(b.trap_probability)
            assert a.sweep_direction == b.sweep_direction


def test_liquidity_trap_disabled_yields_all_absent():
    df = _frame(seed=15)
    s = Phase2Strategy(_FEATURES, use_liquidity_trap=False, **_kwargs())
    assert all(r is None for r in s._liquidity_trap_series(df))


# ------------------------------------------------------------------------ #
# Layer 7C: ensemble
# ------------------------------------------------------------------------ #


def test_ensemble_fits_weights_and_exposes_them_per_fold():
    df = _frame(seed=17)
    train, val, _ = _split(df)
    s = Phase2Strategy(_FEATURES, use_ensemble=True, use_liquidity_trap=True,
                       use_volatility_gate=False, **_kwargs())
    s.fit(train, val)
    weights = s.ensemble_weights()
    assert weights is not None
    assert sum(weights.values()) == pytest.approx(1.0)


def test_ensemble_model_persists_across_folds_for_stability_tracking():
    """A fresh EnsembleMetaModel per fold would have an empty weight history
    every time, so the Adaptive doc's 0.15 cross-fold stability rule -- and
    its equal-weight fallback -- could never fire."""
    df = _frame(seed=19, n=520)
    train, val, _ = _split(df)
    s = Phase2Strategy(_FEATURES, use_ensemble=True, use_liquidity_trap=True,
                       use_volatility_gate=False, **_kwargs())
    s.fit(train, val)
    first_model = s._ensemble
    s.fit(train, val)
    assert s._ensemble is first_model  # same instance -> history accumulates
    assert len(s._ensemble._fold_weight_history) == 2


def test_unwired_layers_are_passed_as_absent_not_fabricated_neutral():
    """Layers 3 and 4 must reach the ensemble as None (excluded and
    renormalized), never as invented neutral result objects."""
    df = _frame(seed=21)
    train, val, _ = _split(df)
    s = Phase2Strategy(_FEATURES, use_ensemble=True, use_liquidity_trap=True, **_kwargs())
    s.fit(train, val)
    sets = s._build_signal_sets(val.iloc[-60:])
    assert sets, "expected at least one signal set"
    assert all(x.macro_regime is None for x in sets)
    assert all(x.mtf_alignment is None for x in sets)


def test_ensemble_path_produces_bounded_positions():
    df = _frame(seed=23)
    train, val, test = _split(df)
    s = Phase2Strategy(_FEATURES, use_ensemble=True, use_liquidity_trap=True,
                       use_volatility_gate=True, **_kwargs())
    s.fit(train, val)
    positions = s.generate_positions(test)
    assert positions.abs().max() <= s.max_position + 1e-12
    assert positions.index.equals(test.index)


# ------------------------------------------------------------------------ #
# Contract / validation
# ------------------------------------------------------------------------ #


def test_required_columns_depend_on_enabled_layers():
    off = Phase2Strategy(_FEATURES, use_liquidity_trap=False, use_volatility_gate=False,
                         use_ensemble=False, **_kwargs())
    on = Phase2Strategy(_FEATURES, use_liquidity_trap=True, use_volatility_gate=True, **_kwargs())
    assert RAW_ATR_COLUMN not in off.required_columns()
    assert RAW_ATR_COLUMN in on.required_columns()
    assert "vol_atr_pct_rank" in on.required_columns()


def test_missing_required_column_raises_clearly():
    df = _frame(seed=25).drop(columns=[RAW_ATR_COLUMN])
    train, val, _ = _split(df)
    s = Phase2Strategy(_FEATURES, use_liquidity_trap=True, **_kwargs())
    with pytest.raises(ValueError, match=RAW_ATR_COLUMN):
        s.fit(train, val)


def test_generate_positions_before_fit_raises():
    df = _frame(seed=27)
    s = Phase2Strategy(_FEATURES, **_kwargs())
    with pytest.raises(RuntimeError):
        s.generate_positions(df)


def test_label_horizon_mismatch_raises():
    df = _frame(seed=29)
    df.attrs["label_horizon"] = _LABEL_HORIZON + 11
    train, val, _ = _split(df)
    train.attrs["label_horizon"] = _LABEL_HORIZON + 11
    s = Phase2Strategy(_FEATURES, **_kwargs())
    with pytest.raises(ValueError, match="label_horizon mismatch"):
        s.fit(train, val)
