"""
Tests for the Explainability Engine (Layer 8),
src/predictive_core/explainability.py.

Covers the four Done-when bars for the pure template/rule-based rewrite:
(1) no LLM/API/network dependency anywhere in the module or its
dependencies, (2) output format matches the Adaptive doc's Layer 8 example
section-for-section, (3) rendering is byte-for-byte deterministic, and
(4) every edge case (near-zero attribution, ties, degraded data, weight
instability) produces readable output rather than a crash or a blank
field.

Also covers, unchanged from the pre-rewrite suite: the provable sum
identity behind `contributions_from_ensemble`, the Wilson CI, and the
exact +0.15% EV gate.

No test here makes a network call, because the module under test has no
code path that could.
"""

import dataclasses
import importlib
import inspect
import re
import sys

import numpy as np
import pytest

from src.macro_mtf_liquidity_vol.liquidity_trap import LiquidityTrapResult
from src.macro_mtf_liquidity_vol.macro_regime import MacroRegimeResult
from src.macro_mtf_liquidity_vol.mtf_alignment import MTFAlignmentResult
from src.predictive_core import ensemble
from src.predictive_core import explainability as explainability_module
from src.predictive_core.analog_engine import AnalogForecast
from src.predictive_core.ensemble import (
    ComponentBSignal,
    EnsembleSignalSet,
    WeightStabilityReport,
)
from src.predictive_core.explainability import (
    EV_LOWER_CI_GATE_PCT,
    NEGLIGIBLE_NET_LEAN_PCT,
    TIE_TOLERANCE_PCT,
    ExplainabilityEngine,
    SignalContribution,
    TradeRiskSummary,
    _wilson_score_interval,
    contributions_from_ensemble,
    direction_probability_ci,
    net_lean_pct,
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


def _forecast_from(signals, weights=None):
    """Builds an `EnsembleForecast` directly from `blend_signals_detailed`
    with an EXPLICIT, known weight vector (equal weights by default),
    rather than going through `EnsembleMetaModel.fit()` on a synthetic
    training fold. Deliberate: log-loss weight optimization has no reason
    to assign meaningful weight to a signal that happens to be constant
    across a small synthetic fold -- it correctly drives that weight toward
    ~0, which is CORRECT ensemble behavior but makes fixtures built that
    way unpredictable for tests that need every one of the five signals to
    carry a real contribution. Optimizer behavior is covered by
    `test_ensemble.py`."""
    weights = weights if weights is not None else ensemble.equal_weights()
    blended, _ = ensemble.blend_signals_detailed(signals, weights)
    direction_probabilities = dict(zip(ensemble.DIRECTION_CLASSES, (float(x) for x in blended)))
    called_direction = ensemble.DIRECTION_CLASSES[int(np.argmax(blended))]
    b = signals.component_b
    return ensemble.EnsembleForecast(
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


def _bullish_forecast_and_signals(weights=None):
    signals = _signal_set(
        component_a=_analog(p_up=0.7, p_down=0.2, p_neutral=0.1, n=187),
        component_b=_component_b(p_up=0.8, p_down=0.1, p_neutral=0.1),
        macro_regime=_macro(state="Risk-On", confidence=0.81, macro_weight=1.0),
        mtf_alignment=_mtf(dominant_direction="bullish", confidence=0.9, score=4),
        liquidity_trap=_liquidity(trap_probability=0.22, sweep_just_confirmed=False),
    )
    return _forecast_from(signals, weights), signals


def _risk_summary(passes_gate=True):
    return TradeRiskSummary(
        ev_estimate_pct=0.41,
        ev_lower_ci_pct=0.18 if passes_gate else 0.10,
        position_size_pct=1.4,
    )


def _stable_report():
    return WeightStabilityReport(
        status="STABLE", max_abs_delta=0.05, triggering_fold_pair=None,
        triggering_signal=None, per_pair_max_delta=[0.05],
    )


def _unstable_report():
    return WeightStabilityReport(
        status="UNSTABLE", max_abs_delta=0.3, triggering_fold_pair=(0, 1),
        triggering_signal="macro_regime", per_pair_max_delta=[0.3],
    )


def _explain(**overrides):
    """Standard happy-path render, with per-test overrides."""
    forecast, signals = _bullish_forecast_and_signals()
    kwargs = {
        "asset": "ETH/USD",
        "forecast": forecast,
        "contributions": contributions_from_ensemble(forecast, signals),
        "risk": _risk_summary(),
        "stability": _stable_report(),
    }
    kwargs.update(overrides)
    return ExplainabilityEngine().explain(**kwargs)


# ------------------------------------------------------------------------ #
# Done-when bar 1: no LLM, no API, no network -- anywhere
# ------------------------------------------------------------------------ #


def test_module_source_contains_no_llm_or_network_references():
    """The literal Done-when grep: 'api', 'anthropic', 'claude', and
    requests/httpx/urllib imports must not appear as anything that makes an
    external call.

    Prose mentions of these words in the docstring (explaining WHY the LLM
    path was removed) are legitimate and unavoidable, so this asserts on
    CODE rather than raw text: comments and docstrings are stripped first,
    then the remaining source is checked.
    """
    source = inspect.getsource(explainability_module)
    # Strip docstrings (triple-quoted) and # comments -- prose about the
    # removal is expected; executable references are not.
    code = re.sub(r'"""(?:.|\n)*?"""', "", source)
    code = re.sub(r"'''(?:.|\n)*?'''", "", code)
    code = re.sub(r"#.*", "", code)

    for banned in ("anthropic", "claude", "api_key", "apikey", "requests", "httpx", "urllib", "socket"):
        assert banned not in code.lower(), f"{banned!r} appears in executable code"
    # "api" as a standalone identifier fragment (enable_live_api, .api, etc.)
    assert not re.search(r"\bapi\b", code, flags=re.IGNORECASE)


def test_module_imports_nothing_that_makes_an_external_call():
    """Every top-level import must be stdlib, numpy, or this project's own
    ensemble module."""
    source = inspect.getsource(explainability_module)
    # Strip docstrings first: prose like "...never from summing the filtered
    # display lines" would otherwise be matched as an import statement.
    code = re.sub(r'"""(?:.|\n)*?"""', "", source)
    imported_roots = set()
    for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", code, flags=re.MULTILINE):
        imported_roots.add(match.group(1).split(".")[0])

    allowed = {"math", "dataclasses", "typing", "numpy", "src"}
    assert imported_roots <= allowed, f"unexpected imports: {imported_roots - allowed}"


def test_importing_the_module_does_not_pull_in_an_http_client():
    """Neither the module nor its dependency chain may drag in an HTTP
    stack. `anthropic` in particular must be entirely absent."""
    for name in [m for m in list(sys.modules) if m.split(".")[0] in {"anthropic", "httpx", "requests"}]:
        del sys.modules[name]
    importlib.reload(explainability_module)
    roots = {m.split(".")[0] for m in sys.modules}
    assert "anthropic" not in roots
    assert "requests" not in roots


def test_removed_llm_surface_is_gone_not_merely_disabled():
    for removed in (
        "AnthropicProseGenerator", "ProseGenerationError", "DEFAULT_PROSE_MODEL",
        "_build_prose_prompt", "_extract_json_object",
    ):
        assert not hasattr(explainability_module, removed), f"{removed} still present"

    params = inspect.signature(ExplainabilityEngine.__init__).parameters
    assert "enable_live_api" not in params
    assert "prose_generator" not in params
    assert not hasattr(_explain(), "used_live_api")


# ------------------------------------------------------------------------ #
# Done-when bar 3: determinism
# ------------------------------------------------------------------------ #


def test_output_is_byte_for_byte_deterministic_across_runs():
    """The core guarantee: identical attribution input must render
    identically every time, forever. An audit log that cannot be reproduced
    cannot be audited."""
    renders = []
    for _ in range(25):
        forecast, signals = _bullish_forecast_and_signals()
        result = ExplainabilityEngine().explain(
            asset="ETH/USD", forecast=forecast,
            contributions=contributions_from_ensemble(forecast, signals),
            risk=_risk_summary(), stability=_stable_report(),
            macro_regime_state="Risk-On",
        )
        renders.append(result.report_text)

    assert len(set(renders)) == 1
    assert renders[0].encode("utf-8") == renders[-1].encode("utf-8")


def test_determinism_holds_for_every_edge_case_variant():
    """Determinism must not be a property only of the happy path."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    tied = [
        SignalContribution(label="b_signal", detail="d", impact_pct=10.0),
        SignalContribution(label="a_signal", detail="d", impact_pct=10.0),
    ]
    variants = [
        {"contributions": []},
        {"contributions": tied},
        {"contributions": contributions, "stability": _unstable_report()},
        {"contributions": contributions, "stability": None},
        {"contributions": contributions, "risk": _risk_summary(passes_gate=False)},
        {"contributions": contributions, "macro_regime_state": "Macro-Event Window"},
        {"forecast": dataclasses.replace(forecast, analog_is_valid=False)},
    ]
    for overrides in variants:
        texts = {_explain(**overrides).report_text for _ in range(8)}
        assert len(texts) == 1, f"non-deterministic for {sorted(overrides)}"


def test_tied_contributions_render_in_a_stable_order_regardless_of_input_order():
    """Ties must be broken totally (by label), not left to input ordering --
    otherwise upstream dict ordering could silently change the report."""
    a = SignalContribution(label="alpha", detail="d", impact_pct=12.0)
    b = SignalContribution(label="beta", detail="d", impact_pct=12.0)
    forward = _explain(contributions=[a, b]).report_text
    reversed_ = _explain(contributions=[b, a]).report_text
    assert forward == reversed_


def test_engine_instances_do_not_accumulate_state():
    """A single engine reused across calls must render the same as fresh
    ones -- no memoisation or mutation between explanations."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    shared = ExplainabilityEngine()
    first = shared.explain(asset="ETH/USD", forecast=forecast, contributions=contributions,
                           risk=_risk_summary(), stability=_stable_report())
    _ = shared.explain(asset="BTC/USD", forecast=forecast, contributions=[],
                       risk=_risk_summary(passes_gate=False))
    third = shared.explain(asset="ETH/USD", forecast=forecast, contributions=contributions,
                           risk=_risk_summary(), stability=_stable_report())
    assert first.report_text == third.report_text


# ------------------------------------------------------------------------ #
# Done-when bar 2: output format matches the spec, section-for-section
# ------------------------------------------------------------------------ #


def test_spec_report_has_every_section_in_the_documented_order():
    """The Adaptive doc's Layer 8 example, section by section:
    header / probability+CI / contributions / EV estimate / EV lower CI
    with gate verdict / position size / stability."""
    result = _explain()
    lines = result.spec_report_text.splitlines()

    assert lines[0] == "ETH/USD — LONG SETUP"
    assert lines[1].startswith("Bullish Probability: ")
    assert "95% CI:" in lines[1] and "independent analogues" in lines[1]
    assert lines[2] == "Signal contributions:"

    tail = "\n".join(lines[3:])
    assert "EV estimate: +0.41% per unit risk" in tail
    assert "EV lower CI bound: +0.18% [GATE: passes +0.15% threshold]" in tail
    assert "Position size: 1.4% of capital" in tail
    assert tail.splitlines()[-1].startswith("Ensemble weight stability: STABLE")

    # Section ORDER, not just presence.
    positions = [
        result.spec_report_text.index(marker)
        for marker in ("Signal contributions:", "EV estimate:", "EV lower CI bound:",
                       "Position size:", "Ensemble weight stability:")
    ]
    assert positions == sorted(positions)


def test_contribution_lines_match_the_spec_shape():
    """Spec: `  + Macro regime (Risk-On, confidence 0.81):    +10%`"""
    result = _explain()
    body = result.spec_report_text.split("Signal contributions:\n")[1].split("\nEV estimate:")[0]
    for line in body.splitlines():
        assert re.match(r"^  [+-] .+\(.*\):\s+[+-]\d+%$", line), line


def test_contribution_percentages_are_right_aligned_into_a_column():
    result = _explain()
    body = result.spec_report_text.split("Signal contributions:\n")[1].split("\nEV estimate:")[0]
    assert len({len(line) for line in body.splitlines()}) == 1


def test_report_shows_setup_label_and_adjective_matching_direction():
    result = _explain(asset="BTC/USD")
    assert result.setup_label == "LONG SETUP"
    assert "Bullish Probability" in result.spec_report_text


def test_numbers_come_only_from_the_attribution_data():
    """Every percentage rendered on a contribution line must equal that
    contribution's own computed impact_pct -- never re-derived, rounded
    differently, or invented."""
    result = _explain()
    for line in result.contributions:
        rendered = next(
            row for row in result.spec_report_text.splitlines()
            if row.strip().startswith(("+ " + line.prose, "- " + line.prose))
        )
        assert rendered.endswith(f"{line.impact_pct:+.0f}%")


# ------------------------------------------------------------------------ #
# Done-when bar 4: edge cases produce readable output, never blanks/crashes
# ------------------------------------------------------------------------ #


def test_no_contributions_renders_a_placeholder_not_a_blank_block():
    result = _explain(contributions=[])
    assert "(no individually significant contributions)" in result.spec_report_text
    assert any("No individual signal cleared" in s for s in result.narrative)
    assert result.report_text.strip()


def test_near_zero_net_lean_is_described_as_undecided_not_dominated():
    """Attribution summing to near-zero: the narrative must NOT name a
    'dominant' driver of an essentially undecided call."""
    base, signals = _bullish_forecast_and_signals()
    # Bull and bear almost exactly offset: the call is technically "up" but
    # carries essentially no conviction. Constructed directly rather than
    # blended, so the fixture pins the precise condition under test.
    forecast = dataclasses.replace(
        base,
        direction_probabilities={"down": 0.485, "neutral": 0.020, "up": 0.495},
        called_direction="up",
        called_probability=0.495,
    )
    assert abs(net_lean_pct(forecast)) < NEGLIGIBLE_NET_LEAN_PCT

    result = _explain(
        forecast=forecast,
        contributions=contributions_from_ensemble(base, signals),
        stability=_stable_report(),
    )
    joined = " ".join(result.narrative)
    assert "negligible" in joined
    assert "dominant" not in joined


def test_tied_top_contributors_are_reported_as_a_tie():
    tied = [
        SignalContribution(label="mtf_alignment", detail="4/4 bullish", impact_pct=14.0),
        SignalContribution(label="macro_regime", detail="Risk-On, confidence 0.81", impact_pct=14.0),
        SignalContribution(label="liquidity_trap", detail="no confirmed sweep", impact_pct=2.0),
    ]
    result = _explain(contributions=tied)
    joined = " ".join(result.narrative)
    assert "equal weight" in joined
    assert "no single factor dominates" in joined
    # Both tied names must appear -- neither may be silently dropped.
    assert "Multi-timeframe alignment" in joined and "Macro regime" in joined


def test_near_tie_within_tolerance_is_still_a_tie_but_outside_it_is_not():
    base = SignalContribution(label="macro_regime", detail="d", impact_pct=14.0)
    inside = SignalContribution(
        label="mtf_alignment", detail="d", impact_pct=14.0 - TIE_TOLERANCE_PCT / 2
    )
    outside = SignalContribution(
        label="mtf_alignment", detail="d", impact_pct=14.0 - TIE_TOLERANCE_PCT * 4
    )
    assert "equal weight" in " ".join(_explain(contributions=[base, inside]).narrative)
    assert "dominant supporting factor" in " ".join(_explain(contributions=[base, outside]).narrative)


def test_negative_top_contributor_is_described_as_arguing_against():
    """Whether the top contributor is positive or negative selects a
    different pre-written sentence."""
    contributions = [
        SignalContribution(label="macro_regime", detail="Risk-Off, confidence 0.9", impact_pct=-18.0),
        SignalContribution(label="mtf_alignment", detail="4/4 bullish", impact_pct=6.0),
        SignalContribution(label="component_b", detail="called up at 70% confidence", impact_pct=5.0),
    ]
    joined = " ".join(_explain(contributions=contributions).narrative)
    assert "argues AGAINST" in joined
    assert "Macro regime" in joined


def test_degraded_data_is_called_out_in_a_data_quality_section():
    contributions = [
        SignalContribution(label="component_a", detail="no valid analog forecast", impact_pct=3.0, is_degraded=True),
        SignalContribution(label="mtf_alignment", detail="4/4 bullish", impact_pct=12.0),
    ]
    result = _explain(contributions=contributions)
    joined = " ".join(result.narrative)
    assert "Data quality" in joined
    assert "Historical analog match" in joined
    assert "incomplete upstream data" in joined


def test_missing_analog_sample_renders_unavailable_ci_not_a_fabricated_one():
    forecast, signals = _bullish_forecast_and_signals()
    degraded = dataclasses.replace(forecast, analog_is_valid=False)
    result = _explain(forecast=degraded, contributions=contributions_from_ensemble(forecast, signals))
    assert result.ci_lower_pct is None and result.ci_upper_pct is None
    assert "95% CI: unavailable (insufficient analog sample)" in result.spec_report_text
    assert "probability interval is unavailable" in " ".join(result.narrative)


def test_weight_instability_is_a_leading_caveat_not_a_footnote():
    """When weights are unstable the attribution percentages are themselves
    under suspicion, so the warning must come FIRST in the narrative."""
    result = _explain(stability=_unstable_report())
    assert result.narrative[0].startswith("WARNING")
    assert "UNSTABLE" in result.narrative[0]
    assert "should not be relied on" in result.narrative[0]
    assert "UNSTABLE" in result.spec_report_text


def test_equal_weight_fallback_is_disclosed():
    forecast, signals = _bullish_forecast_and_signals()
    fallback = dataclasses.replace(forecast, used_fallback_equal_weighting=True)
    result = _explain(
        forecast=fallback,
        contributions=contributions_from_ensemble(forecast, signals),
        stability=_stable_report(),
    )
    assert "using equal-weight fallback" in result.spec_report_text
    assert any("equal-weight fallback" in s for s in result.narrative)


def test_stability_not_evaluated_renders_readably():
    result = _explain(stability=None)
    assert result.stability_status == "NOT_EVALUATED"
    assert "NOT_EVALUATED" in result.spec_report_text
    assert any("has not been evaluated" in s for s in result.narrative)


@pytest.mark.parametrize(
    "state,expected",
    [
        ("Risk-On", "permits full standard EV thresholds"),
        ("Risk-Off", "blocks new long entries"),
        ("Transitional", "position sizes are halved"),
        ("Macro-Event Window", "blocks new entries outright"),
    ],
)
def test_each_macro_regime_selects_its_own_sentence(state, expected):
    result = _explain(macro_regime_state=state)
    assert any(expected in s for s in result.narrative)


def test_unmapped_macro_state_degrades_readably_rather_than_crashing():
    result = _explain(macro_regime_state="Some-Future-State")
    assert any("Some-Future-State" in s for s in result.narrative)
    assert result.report_text.strip()


def test_macro_sentence_is_skipped_when_state_not_supplied():
    result = _explain()  # no macro_regime_state
    assert not any("Macro backdrop" in s for s in result.narrative)


def test_ev_failure_selects_the_failure_sentence():
    passing = " ".join(_explain(risk=_risk_summary(passes_gate=True)).narrative)
    failing = " ".join(_explain(risk=_risk_summary(passes_gate=False)).narrative)
    assert "EV clears the gate" in passing
    assert "EV FAILS the gate" in failing
    assert "must not be taken" in failing


def test_no_rendered_field_is_ever_blank_across_edge_cases():
    """Done-when bar 4, directly: readable output, never a blank field."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)
    variants = [
        {"contributions": []},
        {"contributions": contributions, "stability": _unstable_report()},
        {"contributions": contributions, "stability": None},
        {"contributions": contributions, "risk": _risk_summary(passes_gate=False)},
        {"forecast": dataclasses.replace(forecast, analog_is_valid=False)},
        {"contributions": [SignalContribution(label="x", detail="d", impact_pct=0.0)]},
    ]
    for overrides in variants:
        result = _explain(**overrides)
        assert result.spec_report_text.strip()
        assert result.report_text.strip()
        assert result.narrative and all(s.strip() for s in result.narrative)
        for line in result.report_text.splitlines():
            assert line == "" or line.strip(), "whitespace-only line rendered"
        for c in result.contributions:
            assert c.prose.strip() and c.detail.strip()


def test_empty_asset_is_rejected():
    forecast, signals = _bullish_forecast_and_signals()
    with pytest.raises(ValueError):
        ExplainabilityEngine().explain(
            asset="   ", forecast=forecast,
            contributions=contributions_from_ensemble(forecast, signals), risk=_risk_summary(),
        )


def test_engine_rejects_invalid_thresholds():
    for kwargs in ({"min_abs_pct_to_show": -1.0}, {"tie_tolerance_pct": float("nan")},
                   {"negligible_net_lean_pct": -0.1}):
        with pytest.raises(ValueError):
            ExplainabilityEngine(**kwargs)


def test_unrecognized_called_direction_raises():
    """`EnsembleForecast` itself already validates `called_direction`, so
    this guard is unreachable through a legitimate forecast -- it exists
    only against `_SETUP_LABELS`/`_DIRECTIONAL_ADJECTIVE` drifting out of
    sync with `ensemble.DIRECTION_CLASSES`."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals)

    class _FakeForecast:
        called_direction = "sideways"

    with pytest.raises(ValueError):
        ExplainabilityEngine().explain(
            asset="ETH/USD", forecast=_FakeForecast(),
            contributions=contributions, risk=_risk_summary(),
        )


# ------------------------------------------------------------------------ #
# SignalContribution / TradeRiskSummary validation
# ------------------------------------------------------------------------ #


def test_signal_contribution_rejects_empty_label():
    with pytest.raises(ValueError):
        SignalContribution(label="", detail="x", impact_pct=1.0)


def test_signal_contribution_rejects_empty_detail():
    with pytest.raises(ValueError):
        SignalContribution(label="x", detail="  ", impact_pct=1.0)


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
    signal's impact_pct (BEFORE display filtering) equals
    100 * (p_final[called] - p_final[opposite])."""
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)
    total = sum(c.impact_pct for c in contributions)
    assert total == pytest.approx(net_lean_pct(forecast), abs=1e-6)


def test_net_lean_pct_uses_the_forecast_not_the_filtered_display_lines():
    """Guards the module docstring's NOTE: filtered-out contributions must
    not make the narrative understate conviction."""
    forecast, signals = _bullish_forecast_and_signals()
    unfiltered = sum(
        c.impact_pct for c in contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)
    )
    heavily_filtered = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=1000.0)
    assert heavily_filtered == []
    assert net_lean_pct(forecast) == pytest.approx(unfiltered, abs=1e-6)


def test_contributions_sorted_positive_first_then_negative_by_magnitude():
    forecast, signals = _bullish_forecast_and_signals()
    contributions = contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)
    signs = [c.impact_pct >= 0 for c in contributions]
    assert signs == sorted(signs, reverse=True)
    positives = [c.impact_pct for c in contributions if c.impact_pct >= 0]
    negatives = [c.impact_pct for c in contributions if c.impact_pct < 0]
    assert positives == sorted(positives, reverse=True)
    assert negatives == sorted(negatives)


def test_contributions_below_threshold_are_dropped():
    forecast, signals = _bullish_forecast_and_signals()
    assert len(contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)) > 0
    assert contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=1000.0) == []


def test_contributions_disagreeing_signal_reads_negative():
    """A signal that actively disagrees with the called direction (despite
    being outweighted) must produce a genuinely negative impact_pct."""
    forecast, signals = _bullish_forecast_and_signals()
    assert forecast.called_direction == "up"
    disagreeing = _signal_set(
        component_a=signals.component_a, component_b=signals.component_b,
        macro_regime=_macro(state="Risk-Off", confidence=0.9, macro_weight=1.0),
        mtf_alignment=signals.mtf_alignment, liquidity_trap=signals.liquidity_trap,
    )
    contributions = contributions_from_ensemble(forecast, disagreeing, min_abs_pct_to_show=0.0)
    assert next(c for c in contributions if c.label == "macro_regime").impact_pct < 0


def test_contributions_detail_strings_match_known_format():
    forecast, signals = _bullish_forecast_and_signals()
    by_label = {c.label: c for c in contributions_from_ensemble(forecast, signals, min_abs_pct_to_show=0.0)}
    assert by_label["macro_regime"].detail == "Risk-On, confidence 0.81"
    assert by_label["mtf_alignment"].detail == "4/4 bullish"
    assert by_label["component_a"].detail == "N=187, distance threshold 0.42"
    assert "no confirmed sweep" in by_label["liquidity_trap"].detail
    assert all(not c.is_degraded for c in by_label.values())


def test_component_a_invalid_is_excluded_from_the_blend_entirely():
    forecast, signals = _bullish_forecast_and_signals()
    invalid = _signal_set(
        component_a=_analog(is_valid=False, n=10), component_b=signals.component_b,
        macro_regime=signals.macro_regime, mtf_alignment=signals.mtf_alignment,
        liquidity_trap=signals.liquidity_trap,
    )
    contributions = contributions_from_ensemble(forecast, invalid, min_abs_pct_to_show=0.0)
    assert "component_a" not in {c.label for c in contributions}


# ------------------------------------------------------------------------ #
# Wilson CI
# ------------------------------------------------------------------------ #


def test_wilson_ci_contains_point_estimate():
    lo, hi = _wilson_score_interval(0.5, 200)
    assert lo < 0.5 < hi


def test_wilson_ci_narrows_as_n_increases():
    lo_small, hi_small = _wilson_score_interval(0.7, 50)
    lo_large, hi_large = _wilson_score_interval(0.7, 5000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_direction_probability_ci_none_when_analog_invalid():
    forecast, _ = _bullish_forecast_and_signals()
    assert direction_probability_ci(dataclasses.replace(forecast, analog_is_valid=False)) is None


def test_direction_probability_ci_matches_wilson_formula():
    forecast, _ = _bullish_forecast_and_signals()
    expected = _wilson_score_interval(forecast.called_probability, forecast.analog_n_independent)
    assert direction_probability_ci(forecast) == pytest.approx(expected)


# ------------------------------------------------------------------------ #
# EV gate: exact +0.15% threshold, strict >
# ------------------------------------------------------------------------ #


def test_ev_gate_constant_is_exactly_0_15():
    assert EV_LOWER_CI_GATE_PCT == 0.15


def test_ev_gate_boundary_is_exclusive():
    at_boundary = TradeRiskSummary(ev_estimate_pct=0.4, ev_lower_ci_pct=0.15, position_size_pct=1.0)
    just_above = TradeRiskSummary(ev_estimate_pct=0.4, ev_lower_ci_pct=0.150001, position_size_pct=1.0)
    assert _explain(risk=at_boundary).ev_gate_passes is False
    assert _explain(risk=just_above).ev_gate_passes is True
