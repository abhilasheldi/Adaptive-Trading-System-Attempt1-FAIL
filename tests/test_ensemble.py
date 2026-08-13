"""
Tests for the Ensemble Meta-Model (Layer 7, Component C),
src/predictive_core/ensemble.py.

Covers: the Prompt Playbook's literal Done-when bar (raw Layer 2 feature
rejected by the signal-set constructor), the three Cursor review criteria
quoted verbatim in the build task -- (1) no raw feature can sneak in
alongside the five approved inputs, (2) weights are learned from a
validation-fold-only entry point, (3) the 0.15 instability threshold and
equal-weight fallback actually trigger -- plus the linear-blend arithmetic,
per-signal conversion formulas, and Component A exclusion/renormalization.
"""

import numpy as np
import pytest

from src.macro_mtf_liquidity_vol.liquidity_trap import LiquidityTrapResult
from src.macro_mtf_liquidity_vol.macro_regime import MacroRegimeResult
from src.macro_mtf_liquidity_vol.mtf_alignment import MTFAlignmentResult
from src.predictive_core.analog_engine import AnalogForecast
from src.predictive_core.ensemble import (
    ANALOG_MIN_INDEPENDENT_INSTANCES,
    DIRECTION_CLASSES,
    SIGNAL_NAMES,
    WEIGHT_INSTABILITY_THRESHOLD,
    ComponentBSignal,
    EnsembleForecast,
    EnsembleMetaModel,
    EnsembleSignalSet,
    blend_signals,
    check_weight_stability,
    equal_weights,
    fit_ensemble_weights,
)

# ------------------------------------------------------------------------ #
# Fixtures / helpers
# ------------------------------------------------------------------------ #


def _analog(p_up=0.5, p_down=0.5, p_neutral=0.0, n=80, is_valid=True, dist=1.0):
    return AnalogForecast(
        is_valid=is_valid, n_independent=n, n_raw_candidates=n + 5,
        distance_threshold=dist, probability_up=p_up, probability_down=p_down,
        probability_neutral=p_neutral, reason="ok" if is_valid else "insufficient",
    )


def _component_b(p_up=0.4, p_down=0.4, p_neutral=0.2):
    return ComponentBSignal(
        p_down=p_down, p_neutral=p_neutral, p_up=p_up,
        magnitude_p10=0.005, magnitude_p50=0.012, magnitude_p90=0.03,
        duration_p10=3.0, duration_p50=8.0, duration_p90=20.0,
    )


def _macro(state="Transitional", confidence=0.5, macro_weight=1.0):
    return MacroRegimeResult(
        state=state,
        trading_implication={
            "Risk-On": "Full trading, standard EV thresholds",
            "Risk-Off": "No new long entries in risk assets; existing positions reviewed for exit",
            "Transitional": "Halve position sizes, apply more conservative EV threshold",
            "Macro-Event Window": "No new entries; existing positions get tighter stops",
        }[state],
        confidence=confidence, macro_weight=macro_weight, reasons=["synthetic"],
        us10y_level_pct=4.5, us10y_20d_direction_pct=0.1,
        btc_qqq_corr_20d=0.5, is_macro_event_window=(state == "Macro-Event Window"),
    )


def _mtf(dominant_direction="neutral", confidence=0.0, score=0):
    action = {4: "Full position size permitted", 3: "Standard position size permitted",
              2: "No entry", 1: "No entry, existing positions reviewed",
              0: "No entry, existing positions reviewed"}[score]
    return MTFAlignmentResult(
        score=score, action=action, trade_permitted=score >= 3,
        dominant_direction=dominant_direction, confidence=confidence,
        per_timeframe_direction={}, reasons=["synthetic"],
    )


def _liquidity(trap_probability=0.1, sweep_just_confirmed=False, sweep_direction=None):
    return LiquidityTrapResult(
        trap_probability=trap_probability, liquidity_attraction_score=trap_probability,
        sweep_just_confirmed=sweep_just_confirmed, sweep_direction=sweep_direction,
        nearest_resistance_cluster=None, nearest_support_cluster=None, reasons=["synthetic"],
    )


def _signal_set(**overrides):
    defaults = {
        "component_a": _analog(), "component_b": _component_b(), "macro_regime": _macro(),
        "mtf_alignment": _mtf(), "liquidity_trap": _liquidity(),
    }
    defaults.update(overrides)
    return EnsembleSignalSet(**defaults)


# ------------------------------------------------------------------------ #
# Done-when bar: raw Layer 2 feature rejected by the constructor
# ------------------------------------------------------------------------ #


def test_raw_float_in_place_of_component_a_raises():
    with pytest.raises(TypeError):
        EnsembleSignalSet(
            component_a=0.73,  # a raw feature-like scalar, not an AnalogForecast
            component_b=_component_b(), macro_regime=_macro(),
            mtf_alignment=_mtf(), liquidity_trap=_liquidity(),
        )


def test_raw_dict_of_features_in_place_of_macro_regime_raises():
    with pytest.raises(TypeError):
        EnsembleSignalSet(
            component_a=_analog(), component_b=_component_b(),
            macro_regime={"rsi_14": 55.2, "atr_pct_rank": 0.6},  # raw Layer 2 features
            mtf_alignment=_mtf(), liquidity_trap=_liquidity(),
        )


def test_extra_unexpected_field_raises():
    """No **kwargs / catch-all field exists -- an extra raw feature smuggled
    in as an additional keyword argument fails at construction."""
    with pytest.raises(TypeError):
        EnsembleSignalSet(
            component_a=_analog(), component_b=_component_b(), macro_regime=_macro(),
            mtf_alignment=_mtf(), liquidity_trap=_liquidity(),
            struct_bos_bullish=1.0,  # not a real field
        )


def test_missing_field_raises():
    with pytest.raises(TypeError):
        EnsembleSignalSet(
            component_a=_analog(), component_b=_component_b(), macro_regime=_macro(),
            mtf_alignment=_mtf(),
            # liquidity_trap omitted
        )


def test_wrong_type_for_every_field_individually_raises():
    """Sweep across all five fields -- a raw pandas-Series-shaped value (the
    realistic vector of what a Layer 2 feature row actually looks like)
    passed for ANY single field must be rejected, not just component_a."""
    valid = {
        "component_a": _analog(), "component_b": _component_b(), "macro_regime": _macro(),
        "mtf_alignment": _mtf(), "liquidity_trap": _liquidity(),
    }
    for field_name in SIGNAL_NAMES:
        kwargs = dict(valid)
        kwargs[field_name] = {"not": "the right type"}
        with pytest.raises(TypeError):
            EnsembleSignalSet(**kwargs)


# ------------------------------------------------------------------------ #
# ComponentBSignal validation
# ------------------------------------------------------------------------ #


def test_component_b_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        ComponentBSignal(
            p_down=0.5, p_neutral=0.5, p_up=0.5,  # sums to 1.5
            magnitude_p10=0.01, magnitude_p50=0.02, magnitude_p90=0.03,
            duration_p10=1.0, duration_p50=2.0, duration_p90=3.0,
        )


def test_component_b_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        ComponentBSignal(
            p_down=1.5, p_neutral=-0.5, p_up=0.0,
            magnitude_p10=0.01, magnitude_p50=0.02, magnitude_p90=0.03,
            duration_p10=1.0, duration_p50=2.0, duration_p90=3.0,
        )


# ------------------------------------------------------------------------ #
# Linear blend arithmetic
# ------------------------------------------------------------------------ #


def test_blend_is_the_exact_weighted_average_of_the_five_distributions():
    """With all five signals present and confident, the blend must be
    exactly sum(w_i * p_i) -- no non-linear transform."""
    signals = _signal_set(
        component_a=_analog(p_up=0.9, p_down=0.1, p_neutral=0.0, n=200),  # far above gate -> full trust
        component_b=_component_b(p_up=0.8, p_down=0.1, p_neutral=0.1),
        macro_regime=_macro(state="Risk-On", confidence=1.0, macro_weight=1.0),
        mtf_alignment=_mtf(dominant_direction="bullish", confidence=1.0, score=4),
        liquidity_trap=_liquidity(sweep_just_confirmed=True, sweep_direction="bullish", trap_probability=1.0),
    )
    weights = equal_weights()
    blended = blend_signals(signals, weights)

    expected = (
        np.array([0.1, 0.0, 0.9])          # component_a
        + np.array([0.1, 0.1, 0.8])        # component_b
        + np.array([0.0, 0.0, 1.0])        # macro_regime: Risk-On, full confidence*weight
        + np.array([0.0, 0.0, 1.0])        # mtf_alignment: bullish, full confidence
        + np.array([0.0, 0.0, 1.0])        # liquidity_trap: confirmed bullish sweep, full trap_probability
    ) / 5.0
    np.testing.assert_allclose(blended, expected, atol=1e-9)


def test_blend_sums_to_one():
    signals = _signal_set()
    blended = blend_signals(signals, equal_weights())
    assert blended.sum() == pytest.approx(1.0)


def test_unequal_weights_change_the_blend():
    signals = _signal_set(
        component_b=_component_b(p_up=0.9, p_down=0.05, p_neutral=0.05),
        macro_regime=_macro(state="Risk-Off", confidence=0.9, macro_weight=1.0),
    )
    weights_favor_b = {"component_a": 0.0, "component_b": 1.0, "macro_regime": 0.0,
                        "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    weights_favor_macro = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 1.0,
                            "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blend_b = blend_signals(signals, weights_favor_b)
    blend_macro = blend_signals(signals, weights_favor_macro)
    # Component B leans bullish (up), Risk-Off leans bearish (down) -- must differ.
    assert blend_b[2] > blend_macro[2]  # "up" index
    assert blend_macro[0] > blend_b[0]  # "down" index


# ------------------------------------------------------------------------ #
# Component A: exclusion (invalid) + N-closeness confidence ramp
# ------------------------------------------------------------------------ #


def test_invalid_component_a_is_excluded_and_weight_renormalized():
    """is_valid=False -- component_a's weight must NOT vanish, it must be
    redistributed across the remaining four signals for this row."""
    signals = _signal_set(
        component_a=_analog(is_valid=False, n=10),
        component_b=_component_b(p_up=1.0, p_down=0.0, p_neutral=0.0),
    )
    weights = {"component_a": 0.5, "component_b": 0.5, "macro_regime": 0.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    # component_a excluded -> component_b's 0.5 becomes the ONLY remaining
    # weight among {component_b, macro_regime, mtf_alignment, liquidity_trap}
    # (all others are 0 anyway) -> renormalizes to 1.0 -> pure component_b.
    np.testing.assert_allclose(blended, [0.0, 0.0, 1.0], atol=1e-9)


def test_component_a_at_exactly_the_gate_is_fully_discounted_to_neutral():
    signals = _signal_set(
        component_a=_analog(p_up=1.0, p_down=0.0, n=ANALOG_MIN_INDEPENDENT_INSTANCES),
    )
    weights = {"component_a": 1.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.0, 1.0, 0.0], atol=1e-9)


def test_component_a_far_above_gate_is_fully_trusted():
    signals = _signal_set(
        component_a=_analog(p_up=1.0, p_down=0.0, n=ANALOG_MIN_INDEPENDENT_INSTANCES * 3),
    )
    weights = {"component_a": 1.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.0, 0.0, 1.0], atol=1e-9)


def test_component_a_confidence_ramps_monotonically_with_n():
    weights = {"component_a": 1.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    ns = [ANALOG_MIN_INDEPENDENT_INSTANCES, ANALOG_MIN_INDEPENDENT_INSTANCES + 25,
          ANALOG_MIN_INDEPENDENT_INSTANCES * 2]
    up_probs = []
    for n in ns:
        signals = _signal_set(component_a=_analog(p_up=1.0, p_down=0.0, n=n))
        up_probs.append(blend_signals(signals, weights)[2])
    assert up_probs[0] < up_probs[1] < up_probs[2]


# ------------------------------------------------------------------------ #
# Per-signal conversion formulas
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("state", ["Transitional", "Macro-Event Window"])
def test_macro_regime_non_directional_states_are_full_neutral(state):
    """Even Macro-Event Window's deterministic confidence=1.0 must NOT read
    as directional conviction -- it's certainty about the STATE, not a
    bullish/bearish call."""
    signals = _signal_set(macro_regime=_macro(state=state, confidence=1.0, macro_weight=1.0))
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 1.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.0, 1.0, 0.0], atol=1e-9)


def test_macro_regime_risk_on_leans_bullish_scaled_by_confidence_and_weight():
    signals = _signal_set(macro_regime=_macro(state="Risk-On", confidence=0.8, macro_weight=0.5))
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 1.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    effective = 0.8 * 0.5
    np.testing.assert_allclose(blended, [0.0, 1.0 - effective, effective], atol=1e-9)


def test_macro_regime_low_macro_weight_pulls_risk_off_toward_neutral():
    """macro_weight is the Adaptive doc's 'dynamic weight' -- low BTC/QQQ
    correlation should mute a confident Risk-Off read, not just its
    confidence."""
    confident_but_irrelevant = _macro(state="Risk-Off", confidence=1.0, macro_weight=0.0)
    signals = _signal_set(macro_regime=confident_but_irrelevant)
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 1.0,
               "mtf_alignment": 0.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.0, 1.0, 0.0], atol=1e-9)


def test_mtf_mixed_is_full_neutral_despite_nonzero_confidence():
    signals = _signal_set(mtf_alignment=_mtf(dominant_direction="mixed", confidence=0.7, score=2))
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 1.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.0, 1.0, 0.0], atol=1e-9)


def test_mtf_bearish_leans_down_scaled_by_confidence():
    signals = _signal_set(mtf_alignment=_mtf(dominant_direction="bearish", confidence=0.6, score=3))
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 1.0, "liquidity_trap": 0.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.6, 0.4, 0.0], atol=1e-9)


def test_liquidity_trap_without_confirmed_sweep_is_full_neutral_regardless_of_probability():
    """A high trap_probability with no confirmed sweep describes risk of a
    FUTURE sweep in either direction -- genuinely no directional info yet."""
    signals = _signal_set(liquidity_trap=_liquidity(trap_probability=0.95, sweep_just_confirmed=False))
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 0.0, "liquidity_trap": 1.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.0, 1.0, 0.0], atol=1e-9)


def test_liquidity_trap_confirmed_bearish_sweep_leans_down_scaled_by_probability():
    signals = _signal_set(
        liquidity_trap=_liquidity(trap_probability=0.7, sweep_just_confirmed=True, sweep_direction="bearish")
    )
    weights = {"component_a": 0.0, "component_b": 0.0, "macro_regime": 0.0,
               "mtf_alignment": 0.0, "liquidity_trap": 1.0}
    blended = blend_signals(signals, weights)
    np.testing.assert_allclose(blended, [0.7, 0.3, 0.0], atol=1e-9)


# ------------------------------------------------------------------------ #
# Weight learning: validation-fold-only entry point, log-loss minimization
# ------------------------------------------------------------------------ #


def test_fit_ensemble_weights_returns_a_simplex():
    signal_sets = [_signal_set() for _ in range(30)]
    directions = [1.0 if i % 2 == 0 else -1.0 for i in range(30)]
    result = fit_ensemble_weights(signal_sets, directions)
    assert set(result.weights) == set(SIGNAL_NAMES)
    assert all(0.0 <= w <= 1.0 for w in result.weights.values())
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_fit_ensemble_weights_favors_the_genuinely_informative_signal():
    """component_b perfectly predicts the outcome; the other four signals
    are pure noise (always neutral/uninformative). Log-loss minimization on
    the validation fold must learn a high weight for component_b."""
    rng = np.random.default_rng(7)
    signal_sets, directions = [], []
    for _ in range(200):
        is_up = rng.random() < 0.5
        direction = 1.0 if is_up else -1.0
        signal_sets.append(_signal_set(
            component_a=_analog(p_up=0.5, p_down=0.5, p_neutral=0.0, n=80),  # uninformative
            component_b=_component_b(
                p_up=0.95 if is_up else 0.025, p_down=0.025 if is_up else 0.95, p_neutral=0.025,
            ),
            macro_regime=_macro(state="Transitional"),  # uninformative (neutral)
            mtf_alignment=_mtf(dominant_direction="neutral", score=0),  # uninformative
            liquidity_trap=_liquidity(sweep_just_confirmed=False),  # uninformative
        ))
        directions.append(direction)

    result = fit_ensemble_weights(signal_sets, directions)
    assert result.weights["component_b"] > result.weights["component_a"]
    assert result.weights["component_b"] > result.weights["macro_regime"]
    assert result.weights["component_b"] > result.weights["mtf_alignment"]
    assert result.weights["component_b"] > result.weights["liquidity_trap"]
    assert result.weights["component_b"] > 0.5


def test_fit_ensemble_weights_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_ensemble_weights([_signal_set(), _signal_set()], [1.0])


def test_fit_ensemble_weights_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_ensemble_weights([], [])


def test_fit_ensemble_weights_rejects_non_signal_set_entries():
    with pytest.raises(TypeError):
        fit_ensemble_weights([{"rsi_14": 0.5}], [1.0])


# ------------------------------------------------------------------------ #
# Weight stability + equal-weight fallback (exact 0.15 threshold)
# ------------------------------------------------------------------------ #


def test_weight_instability_threshold_is_exactly_0_15():
    assert WEIGHT_INSTABILITY_THRESHOLD == 0.15


def test_stability_not_evaluated_with_fewer_than_two_folds():
    assert check_weight_stability([]).status == "NOT_EVALUATED"
    assert check_weight_stability([equal_weights()]).status == "NOT_EVALUATED"


def test_stability_stable_when_all_deltas_at_or_under_threshold():
    fold_0 = {"component_a": 0.30, "component_b": 0.30, "macro_regime": 0.20,
              "mtf_alignment": 0.10, "liquidity_trap": 0.10}
    fold_1 = {"component_a": 0.30 + WEIGHT_INSTABILITY_THRESHOLD, "component_b": 0.15,
              "macro_regime": 0.20, "mtf_alignment": 0.10, "liquidity_trap": 0.10}
    report = check_weight_stability([fold_0, fold_1])
    assert report.status == "STABLE"
    assert report.max_abs_delta == pytest.approx(WEIGHT_INSTABILITY_THRESHOLD)


def test_stability_unstable_when_one_weight_exceeds_threshold():
    fold_0 = {"component_a": 0.20, "component_b": 0.20, "macro_regime": 0.20,
              "mtf_alignment": 0.20, "liquidity_trap": 0.20}
    fold_1 = {"component_a": 0.50, "component_b": 0.15,
              "macro_regime": 0.15, "mtf_alignment": 0.15, "liquidity_trap": 0.05}
    report = check_weight_stability([fold_0, fold_1])
    assert report.status == "UNSTABLE"
    assert report.triggering_signal == "component_a"
    assert report.triggering_fold_pair == (0, 1)


def test_stability_checks_every_consecutive_pair_not_just_the_last():
    stable_pair = {"component_a": 0.20, "component_b": 0.20, "macro_regime": 0.20,
                    "mtf_alignment": 0.20, "liquidity_trap": 0.20}
    unstable_jump = {"component_a": 0.20, "component_b": 0.20, "macro_regime": 0.20,
                      "mtf_alignment": 0.55, "liquidity_trap": -0.15}
    # unstable_jump deliberately has a negative weight -- fine for this pure
    # stability-arithmetic test, which only checks deltas, not simplex validity.
    report = check_weight_stability([stable_pair, unstable_jump, stable_pair])
    assert report.status == "UNSTABLE"
    assert report.triggering_fold_pair == (0, 1)


# ------------------------------------------------------------------------ #
# EnsembleMetaModel: end-to-end fold sequencing, fallback, reset
# ------------------------------------------------------------------------ #


def _informative_fold(rng, n_rows=60):
    signal_sets, directions = [], []
    for _ in range(n_rows):
        is_up = rng.random() < 0.5
        direction = 1.0 if is_up else -1.0
        signal_sets.append(_signal_set(
            component_b=_component_b(
                p_up=0.9 if is_up else 0.05, p_down=0.05 if is_up else 0.9, p_neutral=0.05,
            ),
        ))
        directions.append(direction)
    return signal_sets, directions


def test_predict_before_fit_raises():
    model = EnsembleMetaModel()
    with pytest.raises(RuntimeError):
        model.predict(_signal_set())


def test_predict_rejects_non_signal_set():
    model = EnsembleMetaModel()
    model.fit(*_informative_fold(np.random.default_rng(1)))
    with pytest.raises(TypeError):
        model.predict({"rsi_14": 0.5})


def test_single_fold_uses_learned_weights_not_fallback():
    model = EnsembleMetaModel()
    model.fit(*_informative_fold(np.random.default_rng(2)))
    assert model.is_using_fallback is False
    assert model.stability_report.status == "NOT_EVALUATED"
    forecast = model.predict(_signal_set())
    assert forecast.used_fallback_equal_weighting is False
    assert forecast.weights_used == model.weights


def _swing_fold(rng, n_rows=60):
    """A fold where macro_regime is perfectly informative and component_b is
    pure noise -- the mirror image of `_informative_fold`. Fitting this
    right after `_informative_fold` (which makes component_b perfectly
    informative and everything else noise) forces a large weight swing
    between the two signals."""
    signal_sets, directions = [], []
    for _ in range(n_rows):
        is_up = rng.random() < 0.5
        direction = 1.0 if is_up else -1.0
        signal_sets.append(_signal_set(
            component_b=_component_b(p_up=0.34, p_down=0.33, p_neutral=0.33),  # noise
            macro_regime=_macro(
                state="Risk-On" if is_up else "Risk-Off", confidence=0.99, macro_weight=1.0,
            ),
        ))
        directions.append(direction)
    return signal_sets, directions


def test_instability_across_folds_triggers_equal_weight_fallback():
    """Deterministic (fixed seed, verified empirically to trigger UNSTABLE):
    fold 0 learns weight concentrated on component_b, fold 1's signal roles
    flip entirely -- component_b's weight must swing by more than 0.15,
    tripping the fallback."""
    model = EnsembleMetaModel()
    rng = np.random.default_rng(3)

    model.fit(*_informative_fold(rng))  # fold 0: component_b is the informative signal
    assert model.is_using_fallback is False

    model.fit(*_swing_fold(rng))  # fold 1: macro_regime becomes informative instead

    assert model.stability_report.status == "UNSTABLE"
    assert model.is_using_fallback is True
    assert model.weights == equal_weights()
    forecast = model.predict(_signal_set())
    assert forecast.used_fallback_equal_weighting is True
    assert forecast.weights_used == equal_weights()


def test_fallback_persists_across_a_later_individually_stable_fold_until_reset():
    """'Until diagnosed' means fallback does NOT auto-clear just because a
    subsequent fold's own newly-learned weights happen to be close to the
    previous (fallback) weights -- it clears ONLY on an explicit
    `reset_after_diagnosis()` call."""
    model = EnsembleMetaModel()
    rng = np.random.default_rng(3)
    model.fit(*_informative_fold(rng))  # fold 0
    model.fit(*_swing_fold(rng))        # fold 1 -- trips instability (see test above)
    assert model.is_using_fallback is True

    # Fold 2: repeat the SAME swing-fold setup as fold 1 -- the fold1->fold2
    # transition alone is a no-op (weights barely move), which would report
    # STABLE if `fit()` only looked at this newest pair in isolation. It
    # must NOT be read as permission to silently resume learned weights.
    model.fit(*_swing_fold(rng))
    assert model.stability_report.status == "STABLE"  # the fold1->fold2 transition itself
    assert model.is_using_fallback is True             # ...but fallback is still active
    assert model.weights == equal_weights()


def test_reset_after_diagnosis_survives_the_next_stable_fit_call():
    """Regression test for a real bug found during review: `fit()` used to
    re-derive the fallback decision by re-scanning the ENTIRE cumulative
    fold history on every call, so one old, already-diagnosed instability
    (e.g. the fold0->fold1 jump) would permanently re-trigger
    `is_using_fallback` on every subsequent `fit()` call regardless of how
    stable the actual new transition was -- making `reset_after_diagnosis()`
    silently undone by the very next fold. `fit()` must only ever act on the
    NEWEST fold-to-fold transition."""
    model = EnsembleMetaModel()
    rng = np.random.default_rng(3)
    model.fit(*_informative_fold(rng))  # fold 0
    model.fit(*_swing_fold(rng))        # fold 1 -- trips instability
    assert model.is_using_fallback is True

    model.reset_after_diagnosis()
    assert model.is_using_fallback is False

    # Fold 2: another swing-fold, matching fold 1's already-learned weights
    # almost exactly -- a genuinely stable transition. Before the fix, this
    # call would re-discover the stale fold0->fold1 jump (still present
    # earlier in `fold_weight_history`) and flip `is_using_fallback` back to
    # True even though fold1->fold2 itself was stable.
    model.fit(*_swing_fold(rng))
    assert model.stability_report.status == "STABLE"
    assert model.stability_report.max_abs_delta < WEIGHT_INSTABILITY_THRESHOLD
    assert model.is_using_fallback is False
    assert model.weights == model.learned_weights
    assert model.weights != equal_weights()

    # The old instability is still visible in the full-history diagnostic
    # view -- it isn't erased, just no longer drives automatic behavior.
    full_report = model.full_history_stability_report()
    assert full_report.status == "UNSTABLE"
    assert full_report.triggering_fold_pair == (0, 1)


def test_reset_after_diagnosis_resumes_learned_weights():
    model = EnsembleMetaModel()
    model.fit(*_informative_fold(np.random.default_rng(5)))
    learned = dict(model.learned_weights)
    model._is_using_fallback = True
    model._active_weights = equal_weights()
    assert model.weights == equal_weights()

    model.reset_after_diagnosis()
    assert model.is_using_fallback is False
    assert model.weights == learned


def test_equal_weights_sums_to_one_and_covers_all_signal_names():
    w = equal_weights()
    assert set(w) == set(SIGNAL_NAMES)
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0 / len(SIGNAL_NAMES)) for v in w.values())


# ------------------------------------------------------------------------ #
# EnsembleForecast self-consistency
# ------------------------------------------------------------------------ #


def test_forecast_direction_probabilities_sum_to_one_and_called_matches_argmax():
    model = EnsembleMetaModel()
    model.fit(*_informative_fold(np.random.default_rng(6)))
    forecast = model.predict(_signal_set(
        component_b=_component_b(p_up=0.8, p_down=0.1, p_neutral=0.1)
    ))
    assert sum(forecast.direction_probabilities.values()) == pytest.approx(1.0)
    assert forecast.called_direction == max(
        forecast.direction_probabilities, key=forecast.direction_probabilities.get
    )
    assert forecast.called_probability == forecast.direction_probabilities[forecast.called_direction]


def test_forecast_passes_through_component_b_magnitude_and_duration_unchanged():
    model = EnsembleMetaModel()
    model.fit(*_informative_fold(np.random.default_rng(8)))
    b = _component_b()
    forecast = model.predict(_signal_set(component_b=b))
    assert forecast.magnitude_p10 == b.magnitude_p10
    assert forecast.magnitude_p50 == b.magnitude_p50
    assert forecast.magnitude_p90 == b.magnitude_p90
    assert forecast.duration_p10 == b.duration_p10
    assert forecast.duration_p50 == b.duration_p50
    assert forecast.duration_p90 == b.duration_p90


def test_forecast_rejects_mismatched_called_probability():
    with pytest.raises(ValueError):
        EnsembleForecast(
            direction_probabilities={"down": 0.2, "neutral": 0.3, "up": 0.5},
            called_direction="up", called_probability=0.9,  # wrong -- should be 0.5
            magnitude_p10=0.01, magnitude_p50=0.02, magnitude_p90=0.03,
            duration_p10=1.0, duration_p50=2.0, duration_p90=3.0,
            analog_n_independent=80, analog_distance_threshold=1.0, analog_is_valid=True,
            macro_weight_applied=0.5, weights_used=equal_weights(),
            used_fallback_equal_weighting=False,
        )


def test_forecast_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        EnsembleForecast(
            direction_probabilities={"down": 0.5, "neutral": 0.5, "up": 0.5},
            called_direction="up", called_probability=0.5,
            magnitude_p10=0.01, magnitude_p50=0.02, magnitude_p90=0.03,
            duration_p10=1.0, duration_p50=2.0, duration_p90=3.0,
            analog_n_independent=80, analog_distance_threshold=1.0, analog_is_valid=True,
            macro_weight_applied=0.5, weights_used=equal_weights(),
            used_fallback_equal_weighting=False,
        )


def test_direction_classes_match_boosted_core_ordering():
    assert DIRECTION_CLASSES == ("down", "neutral", "up")
