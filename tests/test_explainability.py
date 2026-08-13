"""
Tests for the Explainability Engine (Layer 8),
src/predictive_core/explainability.py.

Covers: the Prompt Playbook's literal Done-when bar ("output format
matches the structure above and degrades gracefully if the API call
fails"), the provable sum identity behind `contributions_from_ensemble`,
the Wilson CI, the exact +0.15% EV gate, and -- critically -- that the
live prose path's numbers can never leak into the rendered report (only
the phrasing can come from the LLM, never the percentages).

No test in this file makes a real network call: the live-API path is
always exercised via dependency-injected fake generators, never the real
`AnthropicProseGenerator`.
"""

import dataclasses

import numpy as np
import pytest

from src.macro_mtf_liquidity_vol.liquidity_trap import LiquidityTrapResult
from src.macro_mtf_liquidity_vol.macro_regime import MacroRegimeResult
from src.macro_mtf_liquidity_vol.mtf_alignment import MTFAlignmentResult
from src.predictive_core import ensemble
from src.predictive_core.analog_engine import AnalogForecast
from src.predictive_core.ensemble import (
    ComponentBSignal,
    EnsembleSignalSet,
    WeightStabilityReport,
)
from src.predictive_core.explainability import (
    EV_LOWER_CI_GATE_PCT,
    ExplainabilityEngine,
    ProseGenerationError,
    SignalContribution,
    TradeRiskSummary,
    _wilson_score_interval,
    contributions_from_ensemble,
    direction_probability_ci,
)

# ------------------------------------------------------------------------ #
# Fixtures / helpers (same construction pattern as test_ensemble.py)
# ------------------------------------------------------------------------ #


def _analog(p_up=0.5, p_down=0.5, p_neutral=0.0, n=187, is_valid=True, dist=0.42):
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


def _macro(state="Risk-On", confidence=0.81, macro_weight=1.0):
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


def _mtf(dominant_direction="bullish", confidence=0.9, score=4):
    action = {4: "Full position size permitted", 3: "Standard position size permitted",
              2: "No entry", 1: "No entry, existing positions reviewed",
              0: "No entry, existing positions reviewed"}[score]
    return MTFAlignmentResult(
        score=score, action=action, trade_permitted=score >= 3,
        dominant_direction=dominant_direction, confidence=confidence,
        per_timeframe_direction={}, reasons=["synthetic"],
    )


def _liquidity(trap_probability=0.22, sweep_just_confirmed=False, sweep_direction=None):
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


def _bullish_forecast_and_signals(weights=None):
    """Builds an `EnsembleForecast` directly from `blend_signals_detailed`
    with an EXPLICIT, known weight vector (equal weights by default),
    rather than going through `EnsembleMetaModel.fit()` on a synthetic
    training fold. Deliberate: log-loss weight optimization has no reason
    to assign meaningful weight to a signal that happens to be constant
    (or otherwise uninformative) across a small synthetic fold -- it
    correctly drives that signal's weight toward ~0, which is CORRECT
    ensemble behavior but makes fixtures built that way unpredictable for
    tests that need every one of the five signals to carry a real,
    non-trivial contribution (matching the Adaptive doc's own
    multi-signal-agreement example). Using a known weight vector directly
    keeps these format/example tests deterministic and independent of
    optimizer behavior, which is already covered by `test_ensemble.py`."""
    signals = _signal_set(
        component_a=_analog(p_up=0.7, p_down=0.2, p_neutral=0.1, n=187),
        component_b=_component_b(p_up=0.8, p_down=0.1, p_neutral=0.1),
        macro_regime=_macro(state="Risk-On", confidence=0.81, macro_weight=1.0),
        mtf_alignment=_mtf(dominant_direction="bullish", confidence=0.9, score=4),
        liquidity_trap=_liquidity(trap_probability=0.22, sweep_just_confirmed=False),
    )
    weights = weights if weights is not None else ensemble.equal_weights()
    blended, _ = ensemble.blend_signals_detailed(signals, weights)
    direction_probabilities = dict(zip(ensemble.DIRECTION_CLASSES, (float(x) for x in blended)))
    called_direction = ensemble.DIRECTION_CLASSES[int(np.argmax(blended))]
    b = signals.component_b
    forecast = ensemble.EnsembleForecast(
        direction_probabilities=direction_probabilities,
        called_direction=called_direction,
        called_probability=direction_probabilities[called_direction],
        magnitude_p10=b.magnitude_p10, magnitude_p50=b.magnitude_p50, magnitude_p90=b.magnitude_p90,
        duration_p10=b.duration_p10, duration_p50=b.duration_p50, duration_p90=b.duration_p90,
        analog_n_independent=signals.component_a.n_independent,
        analog_distance_threshold=signals.component_a.distance_threshold,
        analog_is_valid=signals.component_a.is_valid,
        macro_weight_applied=signals.macro_regime.macro_weight,
        weights_used=weights,
        used_fallback_equal_weighting=False,
    )
    return forecast, signals


def _risk_summary(passes_gate=True):
    return TradeRiskSummary(
        ev_estimate_pct=0.41,
        ev_lower_ci_pct=0.18 if passes_gate else 0.10,
        position_size_pct=1.4,
    )


# ------------------------------------------------------------------------ #
# SignalContribution / TradeRiskSummary validation
# ------------------------------------------------------------------------ #


def test_signal_contribution_rejects_empty_label():
    with pytest.raises(ValueError):
        SignalContribution(label="", detail="x", impact_pct=1.0)


def test_signal_contribution_rejects_non_finite_impact():
    with pytest.raises(ValueError):
        SignalContribution(label="x", detail="y", impact_pct=float("nan"))


def test_trade_risk_summary_rejects_negative_position_size():
    with pytest.raises(ValueError):
        TradeRiskSummary(ev_estimate_pct=0.4, ev_lower_ci_pct=0.2, position_size_pct=-1.0)


def test_trade_risk_summary_rejects_non_finite_fields():
    with pytest.raises(ValueError):
        TradeRiskSummary(ev_estimate_pct=float("inf"), ev_lower_ci_pct=0.2, position_size_pct=1.0)


# ------------------------------------------------------------------------ #
# contributions_from_ensemble: provable sum identity
# ------------------------------------------------------------------------ #


def test_contributions_sum_to_the_called_vs_opposite_net_lean():
    """Exact identity claimed in the module docstring: summing every
    signal's impact_pct (BEFORE the display-filtering threshold drops
    small ones) equals 100 * (p_final[called] - p_final[opposite])."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)
    total = sum(c.impact_pct for c in contributions)

    down_idx = ensemble.DIRECTION_CLASSES.index("down")
    up_idx = ensemble.DIRECTION_CLASSES.index("up")
    called_idx = ensemble.DIRECTION_CLASSES.index(forecast.called_direction)
    opposite_idx = down_idx if forecast.called_direction == "up" else up_idx
    p_final = np.array([
        forecast.direction_probabilities["down"],
        forecast.direction_probabilities["neutral"],
        forecast.direction_probabilities["up"],
    ])
    expected = 100.0 * (p_final[called_idx] - p_final[opposite_idx])
    assert total == pytest.approx(expected, abs=1e-6)


def test_contributions_sorted_positive_first_then_negative_by_magnitude():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)
    signs = [c.impact_pct >= 0 for c in contributions]
    # All positives must precede all negatives.
    assert signs == sorted(signs, reverse=True)
    positives = [c.impact_pct for c in contributions if c.impact_pct >= 0]
    negatives = [c.impact_pct for c in contributions if c.impact_pct < 0]
    assert positives == sorted(positives, reverse=True)
    assert negatives == sorted(negatives)  # most negative last == largest magnitude first is same as ascending here


def test_contributions_below_threshold_are_dropped():
    forecast, signals = _bullish_forecast_and_signals()
    all_contributions = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)
    filtered = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=1000.0)
    assert len(filtered) == 0
    assert len(all_contributions) > 0


def test_contributions_disagreeing_signal_reads_negative():
    """A signal that actively disagrees with the called direction (despite
    being outweighted) must produce a genuinely negative impact_pct under
    the net-lean formula -- not merely a small positive one."""
    forecast, signals = _bullish_forecast_and_signals()
    assert forecast.called_direction == "up"
    # macro_regime reads Risk-Off (bearish) while the ensemble still calls "up" overall.
    disagreeing_signals = _signal_set(
        component_a=signals.component_a, component_b=signals.component_b,
        macro_regime=_macro(state="Risk-Off", confidence=0.9, macro_weight=1.0),
        mtf_alignment=signals.mtf_alignment, liquidity_trap=signals.liquidity_trap,
    )
    contributions = contributions_from_ensemble(forecast, disagreeing_signals, min_abs_pct_to_show=0.0)
    macro_line = next(c for c in contributions if c.label == "macro_regime")
    assert macro_line.impact_pct < 0


def test_contributions_detail_strings_match_known_format():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = {c.label: c for c in contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)}
    assert contributions["macro_regime"].detail == "Risk-On, confidence 0.81"
    assert contributions["mtf_alignment"].detail == "4/4 bullish"
    assert contributions["component_a"].detail == "N=187, distance threshold 0.42"
    assert "no confirmed sweep" in contributions["liquidity_trap"].detail


def test_contributions_component_a_invalid_reports_insufficient_detail():
    forecast, signals = _bullish_forecast_and_signals()
    invalid_signals = _signal_set(
        component_a=_analog(is_valid=False, n=10), component_b=signals.component_b,
        macro_regime=signals.macro_regime, mtf_alignment=signals.mtf_alignment,
        liquidity_trap=signals.liquidity_trap,
    )
    contributions = contributions_from_ensemble(forecast, invalid_signals, min_abs_pct_to_show=0.0)
    labels = {c.label for c in contributions}
    assert "component_a" not in labels  # excluded from the blend entirely, so no contribution line either


# ------------------------------------------------------------------------ #
# Wilson CI
# ------------------------------------------------------------------------ #


def test_wilson_ci_is_symmetric_ish_and_contains_point_estimate():
    lo, hi = _wilson_score_interval(0.5, 200)
    assert lo < 0.5 < hi


def test_wilson_ci_narrows_as_n_increases():
    lo_small, hi_small = _wilson_score_interval(0.7, 50)
    lo_large, hi_large = _wilson_score_interval(0.7, 5000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_direction_probability_ci_none_when_analog_invalid():
    forecast, _ = _bullish_forecast_and_signals()
    invalid_forecast = dataclasses.replace(forecast, analog_is_valid=False)
    assert direction_probability_ci(invalid_forecast) is None


def test_direction_probability_ci_matches_wilson_formula():
    forecast, _ = _bullish_forecast_and_signals()
    ci = direction_probability_ci(forecast)
    expected = _wilson_score_interval(forecast.called_probability, forecast.analog_n_independent)
    assert ci == pytest.approx(expected)


# ------------------------------------------------------------------------ #
# EV gate: exact +0.15% threshold, strict >
# ------------------------------------------------------------------------ #


def test_ev_gate_constant_is_exactly_0_15():
    assert EV_LOWER_CI_GATE_PCT == 0.15


def test_ev_gate_boundary_is_exclusive():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    engine = ExplainabilityEngine(enable_live_api=False)

    at_boundary = TradeRiskSummary(ev_estimate_pct=0.4, ev_lower_ci_pct=0.15, position_size_pct=1.0)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=at_boundary)
    assert result.ev_gate_passes is False

    just_above = TradeRiskSummary(ev_estimate_pct=0.4, ev_lower_ci_pct=0.150001, position_size_pct=1.0)
    result2 = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=just_above)
    assert result2.ev_gate_passes is True


# ------------------------------------------------------------------------ #
# Done-when bar: output format + graceful degradation
# ------------------------------------------------------------------------ #


def _stable_report():
    return WeightStabilityReport(
        status="STABLE", max_abs_delta=0.05, triggering_fold_pair=None,
        triggering_signal=None, per_pair_max_delta=[0.05],
    )


def test_report_text_matches_expected_structure_template_path():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    engine = ExplainabilityEngine(enable_live_api=False)
    result = engine.explain(
        asset="ETH/USD", forecast=forecast, contributions=contributions,
        risk=_risk_summary(), stability=_stable_report(),
    )
    text = result.report_text
    assert text.startswith("ETH/USD — LONG SETUP\n")
    assert "Bullish Probability:" in text
    assert "95% CI:" in text
    assert "Signal contributions:" in text
    assert "EV estimate: +0.41% per unit risk" in text
    assert "EV lower CI bound: +0.18% [GATE: passes +0.15% threshold]" in text
    assert "Position size: 1.4% of capital" in text
    assert "Ensemble weight stability: STABLE" in text
    assert result.used_live_api is False


def test_report_shows_setup_label_and_adjective_matching_direction():
    forecast, signals = _bullish_forecast_and_signals()
    assert forecast.called_direction == "up"
    contributions = contributions_from_ensemble(forecast, signals)
    engine = ExplainabilityEngine(enable_live_api=False)
    result = engine.explain(asset="BTC/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.setup_label == "LONG SETUP"
    assert "Bullish Probability" in result.report_text


def test_enable_live_api_false_never_calls_the_generator():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    def _explode(_contributions):
        raise AssertionError("live generator must not be called when enable_live_api=False")

    engine = ExplainabilityEngine(enable_live_api=False, prose_generator=_explode)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is False


def test_live_api_success_path_uses_injected_phrases():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    def _fake_generator(contribs):
        return {c.label: f"LIVE PHRASE for {c.label}" for c in contribs}

    engine = ExplainabilityEngine(enable_live_api=True, prose_generator=_fake_generator)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is True
    for line in result.contributions:
        assert line.prose == f"LIVE PHRASE for {line.label}"
        assert f"LIVE PHRASE for {line.label}" in result.report_text


def test_live_api_raising_falls_back_to_template_gracefully():
    """The Done-when bar, directly: the API call fails -> output still has
    the correct structure, using template prose instead."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    def _failing_generator(_contribs):
        raise ConnectionError("simulated network failure")

    engine = ExplainabilityEngine(enable_live_api=True, prose_generator=_failing_generator)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is False
    assert "Signal contributions:" in result.report_text
    assert "EV estimate: +0.41% per unit risk" in result.report_text
    # Template phrasing for at least one known label:
    macro_line = next(c for c in result.contributions if c.label == "macro_regime")
    assert macro_line.prose == "Macro regime (Risk-On, confidence 0.81)"


def test_live_api_incomplete_response_falls_back_to_template():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    def _incomplete_generator(contribs):
        # Missing an entry for one of the labels.
        return {c.label: "phrase" for c in list(contribs)[:-1]}

    engine = ExplainabilityEngine(enable_live_api=True, prose_generator=_incomplete_generator)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is False


def test_live_api_empty_string_phrase_is_treated_as_invalid():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    def _blank_generator(contribs):
        return {c.label: "   " for c in contribs}

    engine = ExplainabilityEngine(enable_live_api=True, prose_generator=_blank_generator)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is False


def test_live_api_cannot_smuggle_wrong_numbers_into_the_report():
    """Numeric-safety boundary: even if the injected generator returns a
    phrase containing a fabricated percentage, the REPORTED (trailing)
    percentage for that line -- the number actually appended after the
    prose, which is what any downstream consumer would parse as "the
    impact" -- must still be this module's own computed impact_pct, never
    anything the LLM wrote. The fabricated phrase text itself legitimately
    appears verbatim in the report (prose is rendered as-is); what must
    NEVER happen is the real trailing number being overridden by it."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    true_impacts = {c.label: c.impact_pct for c in contributions}

    def _lying_generator(contribs):
        # Unique fabricated text per label so each rendered line can be
        # located unambiguously below.
        return {c.label: f"Totally fabricated for {c.label} (+999%)" for c in contribs}

    engine = ExplainabilityEngine(enable_live_api=True, prose_generator=_lying_generator)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is True
    assert len(result.contributions) > 0
    for line in result.contributions:
        assert line.impact_pct == pytest.approx(true_impacts[line.label])
        assert "999" not in f"{line.impact_pct:+.0f}%"
        expected_trailing_number = f"{line.impact_pct:+.0f}%"
        rendered_line = next(
            row for row in result.report_text.splitlines() if f"Totally fabricated for {line.label}" in row
        )
        assert rendered_line.endswith(expected_trailing_number)
        assert not rendered_line.endswith("+999%")


def test_prose_generation_error_from_injected_generator_falls_back():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    def _raises_prose_error(_contribs):
        raise ProseGenerationError("simulated malformed response")

    engine = ExplainabilityEngine(enable_live_api=True, prose_generator=_raises_prose_error)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.used_live_api is False


def test_no_contributions_renders_placeholder_line():
    forecast, _ = _bullish_forecast_and_signals()
    engine = ExplainabilityEngine(enable_live_api=False)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=[], risk=_risk_summary())
    assert "(no individually significant contributions)" in result.report_text


def test_stability_none_renders_not_evaluated():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    engine = ExplainabilityEngine(enable_live_api=False)
    result = engine.explain(asset="ETH/USD", forecast=forecast, contributions=contributions, risk=_risk_summary())
    assert result.stability_status == "NOT_EVALUATED"
    assert "NOT_EVALUATED" in result.report_text


def test_unstable_fallback_forecast_shows_fallback_suffix():
    forecast, signals = _bullish_forecast_and_signals()
    # Simulate a forecast produced while the ensemble was using the fallback.
    fallback_forecast = dataclasses.replace(forecast, used_fallback_equal_weighting=True)
    unstable_report = WeightStabilityReport(
        status="UNSTABLE", max_abs_delta=0.3, triggering_fold_pair=(0, 1),
        triggering_signal="macro_regime", per_pair_max_delta=[0.3],
    )
    contributions = contributions_from_ensemble(forecast, signals)
    engine = ExplainabilityEngine(enable_live_api=False)
    result = engine.explain(
        asset="ETH/USD", forecast=fallback_forecast, contributions=contributions,
        risk=_risk_summary(), stability=unstable_report,
    )
    assert "UNSTABLE" in result.report_text
    assert "using equal-weight fallback" in result.report_text


def test_unrecognized_called_direction_raises():
    """`EnsembleForecast` itself already validates `called_direction` is one
    of `ensemble.DIRECTION_CLASSES`, so this defensive check in `explain()`
    is unreachable through a legitimately-constructed forecast -- it exists
    only as a guard against `explainability.py`'s own hardcoded
    `_SETUP_LABELS`/`_DIRECTIONAL_ADJECTIVE` dicts silently drifting out of
    sync with `ensemble.DIRECTION_CLASSES` in the future. Exercised here via
    a minimal duck-typed stand-in rather than a real `EnsembleForecast`,
    since the real dataclass cannot hold an invalid value to begin with."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    engine = ExplainabilityEngine(enable_live_api=False)

    class _FakeForecast:
        called_direction = "sideways"

    with pytest.raises(ValueError):
        engine.explain(asset="ETH/USD", forecast=_FakeForecast(), contributions=contributions, risk=_risk_summary())
