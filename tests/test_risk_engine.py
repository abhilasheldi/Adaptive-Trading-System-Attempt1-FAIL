"""
Tests for the Risk Engine (Layer 9), src/risk_sizing_selection/risk_engine.py.

Covers the Prompt Playbook's literal Done-when bar -- "the fat-tail
multiplier visibly changes output only in Extended/Spike states, not in
Compressed/Expanding" -- as a direct, exact-value comparison across all
four volatility states with otherwise-identical inputs, plus the exact
+/-1%/2%/5%/10% output list, the exact 6% correlated-direction cap, the
monotonicity guard at the stop-loss boundary, gap-risk session
classification, and the composite risk score's per-component weighting.
"""

import pandas as pd
import pytest

from src.macro_mtf_liquidity_vol.liquidity_trap import LiquidityTrapResult
from src.macro_mtf_liquidity_vol.volatility_classifier import VolatilityStateResult
from src.risk_sizing_selection.risk_engine import (
    CORRELATED_DIRECTION_CAP_PCT,
    DEFAULT_OVERNIGHT_END_HOUR_UTC,
    DEFAULT_OVERNIGHT_START_HOUR_UTC,
    FAT_TAIL_MULTIPLIER_EXTENDED,
    FAT_TAIL_MULTIPLIER_NONE,
    FAT_TAIL_MULTIPLIER_SPIKE,
    LOSS_THRESHOLDS_PCT,
    RISK_SCORE_GATE_MAX,
    RiskAssessment,
    RiskEngineInputs,
    _classify_session,
    _compute_risk_score,
    _correlation_flag,
    _fat_tail_multiplier,
    _gap_risk_probability,
    _liquidity_trap_contribution,
    _loss_probability,
    assess_risk,
)

_TRADING_IMPLICATIONS = {
    "Compressed": "Pre-breakout; reduce size until directional confirmation",
    "Expanding": "Active trend initiation; standard position sizing",
    "Extended": "Mature trend/blow-off; reduce size, tighten stops aggressively",
    "Spike": "Tail-risk event; no new entries across all assets",
}


def _volatility(state="Expanding", atr_percentile=0.5, spike_ratio=1.0):
    return VolatilityStateResult(
        state=state, trading_implication=_TRADING_IMPLICATIONS[state],
        atr_percentile=atr_percentile, spike_ratio=spike_ratio, reasons=["synthetic"],
    )


def _liquidity(trap_probability=0.2, sweep_just_confirmed=False, sweep_direction=None):
    return LiquidityTrapResult(
        trap_probability=trap_probability, liquidity_attraction_score=trap_probability,
        sweep_just_confirmed=sweep_just_confirmed, sweep_direction=sweep_direction,
        nearest_resistance_cluster=None, nearest_support_cluster=None, reasons=["synthetic"],
    )


# A Tuesday (weekday), 12:00 UTC -- squarely "weekday_intraday", so tests
# using this timestamp isolate the volatility-regime effect from gap risk.
_WEEKDAY_INTRADAY_TS = pd.Timestamp("2026-08-11 12:00:00", tz="UTC")


def _inputs(
    volatility_state="Expanding",
    magnitude_p10_pct=3.0,
    stop_distance_pct=1.5,
    position_size_pct=1.0,
    existing_correlated_direction_exposure_pct=0.0,
    trap_probability=0.2,
    entry_timestamp=_WEEKDAY_INTRADAY_TS,
):
    return RiskEngineInputs(
        magnitude_p10_pct=magnitude_p10_pct,
        stop_distance_pct=stop_distance_pct,
        entry_timestamp=entry_timestamp,
        volatility_state=_volatility(volatility_state),
        liquidity_trap=_liquidity(trap_probability),
        position_size_pct=position_size_pct,
        existing_correlated_direction_exposure_pct=existing_correlated_direction_exposure_pct,
    )


# ------------------------------------------------------------------------ #
# RiskEngineInputs validation
# ------------------------------------------------------------------------ #


def test_rejects_negative_magnitude():
    with pytest.raises(ValueError):
        _inputs(magnitude_p10_pct=-1.0)


def test_rejects_non_positive_stop_distance():
    with pytest.raises(ValueError):
        _inputs(stop_distance_pct=0.0)
    with pytest.raises(ValueError):
        _inputs(stop_distance_pct=-2.0)


def test_rejects_wrong_type_for_volatility_state():
    with pytest.raises(TypeError):
        RiskEngineInputs(
            magnitude_p10_pct=3.0, stop_distance_pct=1.5, entry_timestamp=_WEEKDAY_INTRADAY_TS,
            volatility_state="Extended",  # raw string, not VolatilityStateResult
            liquidity_trap=_liquidity(), position_size_pct=1.0,
            existing_correlated_direction_exposure_pct=0.0,
        )


def test_rejects_wrong_type_for_liquidity_trap():
    with pytest.raises(TypeError):
        RiskEngineInputs(
            magnitude_p10_pct=3.0, stop_distance_pct=1.5, entry_timestamp=_WEEKDAY_INTRADAY_TS,
            volatility_state=_volatility(), liquidity_trap=0.2,  # raw float
            position_size_pct=1.0, existing_correlated_direction_exposure_pct=0.0,
        )


def test_rejects_negative_position_size():
    with pytest.raises(ValueError):
        _inputs(position_size_pct=-0.5)


def test_rejects_negative_existing_exposure():
    with pytest.raises(ValueError):
        _inputs(existing_correlated_direction_exposure_pct=-1.0)


# ------------------------------------------------------------------------ #
# Done-when bar: fat-tail multiplier ONLY changes output in Extended/Spike
# ------------------------------------------------------------------------ #


def test_compressed_and_expanding_produce_identical_probabilities():
    """Both states carry multiplier 1.0 -- output must be IDENTICAL between
    them, not merely 'similar'."""
    compressed = assess_risk(_inputs(volatility_state="Compressed"))
    expanding = assess_risk(_inputs(volatility_state="Expanding"))
    assert compressed.p_loss_gt_1pct == expanding.p_loss_gt_1pct
    assert compressed.p_loss_gt_2pct == expanding.p_loss_gt_2pct
    assert compressed.p_loss_gt_5pct == expanding.p_loss_gt_5pct
    assert compressed.p_loss_gt_10pct == expanding.p_loss_gt_10pct
    assert compressed.risk_score == expanding.risk_score
    assert compressed.fat_tail_multiplier_applied == expanding.fat_tail_multiplier_applied == FAT_TAIL_MULTIPLIER_NONE
    assert compressed.tail_risk_flag is False
    assert expanding.tail_risk_flag is False


def test_extended_multiplies_probabilities_by_exactly_1_5x():
    baseline = assess_risk(_inputs(volatility_state="Expanding"))
    extended = assess_risk(_inputs(volatility_state="Extended"))
    assert extended.fat_tail_multiplier_applied == pytest.approx(FAT_TAIL_MULTIPLIER_EXTENDED)
    assert extended.tail_risk_flag is True
    for base_p, ext_p in [
        (baseline.p_loss_gt_1pct, extended.p_loss_gt_1pct),
        (baseline.p_loss_gt_2pct, extended.p_loss_gt_2pct),
        (baseline.p_loss_gt_5pct, extended.p_loss_gt_5pct),
        (baseline.p_loss_gt_10pct, extended.p_loss_gt_10pct),
    ]:
        assert ext_p == pytest.approx(min(1.0, base_p * FAT_TAIL_MULTIPLIER_EXTENDED), abs=1e-9)


def test_spike_multiplies_probabilities_by_exactly_2_0x():
    baseline = assess_risk(_inputs(volatility_state="Expanding"))
    spike = assess_risk(_inputs(volatility_state="Spike"))
    assert spike.fat_tail_multiplier_applied == pytest.approx(FAT_TAIL_MULTIPLIER_SPIKE)
    assert spike.tail_risk_flag is True
    for base_p, spike_p in [
        (baseline.p_loss_gt_1pct, spike.p_loss_gt_1pct),
        (baseline.p_loss_gt_2pct, spike.p_loss_gt_2pct),
        (baseline.p_loss_gt_5pct, spike.p_loss_gt_5pct),
        (baseline.p_loss_gt_10pct, spike.p_loss_gt_10pct),
    ]:
        assert spike_p == pytest.approx(min(1.0, base_p * FAT_TAIL_MULTIPLIER_SPIKE), abs=1e-9)


def test_spike_probabilities_strictly_exceed_extended_which_strictly_exceed_baseline():
    """Direct, visible ordering -- the multiplier's effect must be
    observable in the actual numbers, not just in a metadata field."""
    baseline = assess_risk(_inputs(volatility_state="Compressed"))
    extended = assess_risk(_inputs(volatility_state="Extended"))
    spike = assess_risk(_inputs(volatility_state="Spike"))
    assert baseline.p_loss_gt_5pct < extended.p_loss_gt_5pct < spike.p_loss_gt_5pct
    assert baseline.risk_score < extended.risk_score < spike.risk_score


def test_fat_tail_multiplier_function_exact_mapping():
    assert _fat_tail_multiplier("Compressed") == FAT_TAIL_MULTIPLIER_NONE
    assert _fat_tail_multiplier("Expanding") == FAT_TAIL_MULTIPLIER_NONE
    assert _fat_tail_multiplier("Extended") == FAT_TAIL_MULTIPLIER_EXTENDED
    assert _fat_tail_multiplier("Spike") == FAT_TAIL_MULTIPLIER_SPIKE


def test_fat_tail_multiplier_constants_within_spec_range():
    assert 1.5 <= FAT_TAIL_MULTIPLIER_EXTENDED <= 2.0
    assert 1.5 <= FAT_TAIL_MULTIPLIER_SPIKE <= 2.0
    assert FAT_TAIL_MULTIPLIER_EXTENDED == 1.5
    assert FAT_TAIL_MULTIPLIER_SPIKE == 2.0


# ------------------------------------------------------------------------ #
# P(loss > T): monotonicity, stop-boundary behavior, gap-risk guard
# ------------------------------------------------------------------------ #


def test_loss_probability_at_zero_distance_is_one():
    assert _loss_probability(0.0, stop_distance_pct=2.0, scale_pct=3.0, gap_risk_probability=0.02) == 1.0


def test_loss_probability_monotonically_non_increasing():
    values = [
        _loss_probability(t, stop_distance_pct=2.0, scale_pct=3.0, gap_risk_probability=0.02)
        for t in [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0]
    ]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_loss_probability_drops_at_stop_boundary_when_gap_risk_is_small():
    """Crossing the stop-loss level should sharply reduce further-loss
    probability when gap risk is small relative to the at-stop probability
    -- this is the whole point of gating beyond-stop loss on gap risk."""
    just_below = _loss_probability(1.99, stop_distance_pct=2.0, scale_pct=3.0, gap_risk_probability=0.02)
    just_above = _loss_probability(2.01, stop_distance_pct=2.0, scale_pct=3.0, gap_risk_probability=0.02)
    assert just_above < just_below


def test_loss_probability_monotonicity_guard_when_gap_risk_exceeds_at_stop_probability():
    """Regression test for the exact failure mode the monotonicity guard
    exists to prevent: a stop distance large relative to scale makes the
    at-stop probability tiny -- without capping the beyond-stop ceiling at
    that value, P(loss > T) would jump UP just past the stop boundary."""
    scale_pct = 0.5  # small scale -> prob_at_stop decays fast
    stop_distance_pct = 10.0  # large stop relative to scale
    gap_risk_probability = 0.5  # deliberately larger than prob_at_stop would be
    values = [
        _loss_probability(t, stop_distance_pct, scale_pct, gap_risk_probability)
        for t in [1.0, 5.0, 9.9, 10.0, 10.1, 12.0, 20.0]
    ]
    assert all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1)), values


def test_loss_probability_zero_scale_is_zero_beyond_origin():
    assert _loss_probability(1.0, stop_distance_pct=2.0, scale_pct=0.0, gap_risk_probability=0.02) == 0.0


def test_loss_probability_larger_scale_increases_probability():
    """Bigger expected move size (looser conservative magnitude estimate)
    -> higher probability of a given adverse excursion -- sensible
    direction for the exponential-decay model."""
    small_scale = _loss_probability(2.0, stop_distance_pct=5.0, scale_pct=1.0, gap_risk_probability=0.02)
    large_scale = _loss_probability(2.0, stop_distance_pct=5.0, scale_pct=5.0, gap_risk_probability=0.02)
    assert large_scale > small_scale


# ------------------------------------------------------------------------ #
# Gap risk: session classification
# ------------------------------------------------------------------------ #


def test_weekday_intraday_session_classification():
    ts = pd.Timestamp("2026-08-11 12:00:00", tz="UTC")  # Tuesday
    assert ts.dayofweek == 1
    assert _classify_session(ts, DEFAULT_OVERNIGHT_START_HOUR_UTC, DEFAULT_OVERNIGHT_END_HOUR_UTC) == "weekday_intraday"


def test_weekend_session_classification():
    saturday = pd.Timestamp("2026-08-15 12:00:00", tz="UTC")
    sunday = pd.Timestamp("2026-08-16 12:00:00", tz="UTC")
    assert saturday.dayofweek == 5 and sunday.dayofweek == 6
    assert _classify_session(saturday, DEFAULT_OVERNIGHT_START_HOUR_UTC, DEFAULT_OVERNIGHT_END_HOUR_UTC) == "weekend"
    assert _classify_session(sunday, DEFAULT_OVERNIGHT_START_HOUR_UTC, DEFAULT_OVERNIGHT_END_HOUR_UTC) == "weekend"


def test_overnight_session_classification_boundaries():
    start_hour_ts = pd.Timestamp("2026-08-11 00:00:00", tz="UTC")  # Tuesday, hour == start
    end_hour_ts = pd.Timestamp("2026-08-11 06:00:00", tz="UTC")  # Tuesday, hour == end (exclusive)
    assert _classify_session(start_hour_ts, 0, 6) == "overnight"
    assert _classify_session(end_hour_ts, 0, 6) == "weekday_intraday"


def test_gap_risk_ordering_weekend_greater_than_overnight_greater_than_weekday():
    weekday = _gap_risk_probability(pd.Timestamp("2026-08-11 12:00:00", tz="UTC"))
    overnight = _gap_risk_probability(pd.Timestamp("2026-08-11 03:00:00", tz="UTC"))
    weekend = _gap_risk_probability(pd.Timestamp("2026-08-15 03:00:00", tz="UTC"))  # Saturday overnight-hour too
    assert weekday < overnight < weekend


def test_gap_risk_naive_timestamp_treated_as_utc():
    naive = pd.Timestamp("2026-08-11 12:00:00")  # tz-naive
    aware = pd.Timestamp("2026-08-11 12:00:00", tz="UTC")
    assert _gap_risk_probability(naive) == _gap_risk_probability(aware)


# ------------------------------------------------------------------------ #
# Liquidity trap risk: scaled by position size
# ------------------------------------------------------------------------ #


def test_liquidity_trap_contribution_zero_when_trap_probability_zero():
    assert _liquidity_trap_contribution(0.0, position_size_pct=10.0) == 0.0


def test_liquidity_trap_contribution_scales_with_position_size():
    small = _liquidity_trap_contribution(0.5, position_size_pct=0.5, full_weight_size_pct=5.0)
    large = _liquidity_trap_contribution(0.5, position_size_pct=5.0, full_weight_size_pct=5.0)
    assert small < large
    assert large == pytest.approx(0.5)  # full weight reached exactly at full_weight_size_pct


def test_liquidity_trap_contribution_capped_beyond_full_weight_size():
    at_full = _liquidity_trap_contribution(0.9, position_size_pct=5.0, full_weight_size_pct=5.0)
    beyond_full = _liquidity_trap_contribution(0.9, position_size_pct=50.0, full_weight_size_pct=5.0)
    assert at_full == beyond_full == pytest.approx(0.9)


# ------------------------------------------------------------------------ #
# Portfolio correlation risk: exact 6% cap
# ------------------------------------------------------------------------ #


def test_correlated_direction_cap_is_exactly_6_pct():
    assert CORRELATED_DIRECTION_CAP_PCT == 6.0


def test_correlation_flag_below_approach_threshold_is_false():
    assert _correlation_flag(exposure_pct=3.0) is False  # 3% of 6% cap = 50% < 80% approach fraction


def test_correlation_flag_at_approach_threshold_is_true():
    """Exactly 80% of the 6% cap (4.8) must flag -- including guarding
    against the float-precision trap where 6.0 * 0.8 != the literal 4.8
    (see `_correlation_flag`'s epsilon-tolerance comment)."""
    assert _correlation_flag(exposure_pct=4.8) is True
    assert _correlation_flag(exposure_pct=6.0 * 0.8) is True


def test_correlation_flag_end_to_end_via_assess_risk():
    under_cap = assess_risk(_inputs(position_size_pct=1.0, existing_correlated_direction_exposure_pct=1.0))
    approaching_cap = assess_risk(_inputs(position_size_pct=2.0, existing_correlated_direction_exposure_pct=3.0))
    assert under_cap.portfolio_correlation_flag is False
    assert approaching_cap.portfolio_correlation_exposure_pct == pytest.approx(5.0)
    assert approaching_cap.portfolio_correlation_flag is True


# ------------------------------------------------------------------------ #
# Composite risk score: per-component weighting
# ------------------------------------------------------------------------ #


def test_risk_score_bounds_are_0_to_100():
    minimal = _compute_risk_score(p_loss_2pct=0.0, gap_risk_probability=0.0,
                                   liquidity_trap_contribution=0.0, correlation_exposure_pct=0.0)
    maximal = _compute_risk_score(p_loss_2pct=1.0, gap_risk_probability=1.0,
                                   liquidity_trap_contribution=1.0, correlation_exposure_pct=100.0)
    assert minimal == pytest.approx(0.0)
    assert maximal == pytest.approx(100.0)


def test_risk_score_p_loss_2pct_contributes_up_to_60_points():
    score = _compute_risk_score(p_loss_2pct=1.0, gap_risk_probability=0.0,
                                 liquidity_trap_contribution=0.0, correlation_exposure_pct=0.0)
    assert score == pytest.approx(60.0)


def test_risk_score_liquidity_trap_contributes_up_to_15_points():
    score = _compute_risk_score(p_loss_2pct=0.0, gap_risk_probability=0.0,
                                 liquidity_trap_contribution=1.0, correlation_exposure_pct=0.0)
    assert score == pytest.approx(15.0)


def test_risk_score_gap_risk_contributes_up_to_15_points():
    score = _compute_risk_score(p_loss_2pct=0.0, gap_risk_probability=1.0,
                                 liquidity_trap_contribution=0.0, correlation_exposure_pct=0.0)
    assert score == pytest.approx(15.0)


def test_risk_score_correlation_contributes_up_to_10_points():
    score = _compute_risk_score(p_loss_2pct=0.0, gap_risk_probability=0.0,
                                 liquidity_trap_contribution=0.0, correlation_exposure_pct=6.0)
    assert score == pytest.approx(10.0)
    beyond_cap = _compute_risk_score(p_loss_2pct=0.0, gap_risk_probability=0.0,
                                      liquidity_trap_contribution=0.0, correlation_exposure_pct=60.0)
    assert beyond_cap == pytest.approx(10.0)  # clipped, not unbounded


# ------------------------------------------------------------------------ #
# assess_risk end-to-end / RiskAssessment validation
# ------------------------------------------------------------------------ #


def test_assess_risk_rejects_wrong_input_type():
    with pytest.raises(TypeError):
        assess_risk({"not": "a RiskEngineInputs"})


def test_assess_risk_output_types_and_ranges():
    result = assess_risk(_inputs())
    assert isinstance(result, RiskAssessment)
    assert 0.0 <= result.risk_score <= 100.0
    for field in (result.p_loss_gt_1pct, result.p_loss_gt_2pct, result.p_loss_gt_5pct, result.p_loss_gt_10pct):
        assert 0.0 <= field <= 1.0
    assert len(result.reasons) > 0


def test_assess_risk_passes_gate_boundary_exact_65():
    result = assess_risk(_inputs())
    assert result.passes_risk_gate == (result.risk_score <= RISK_SCORE_GATE_MAX)


def test_risk_score_gate_max_is_exactly_65():
    assert RISK_SCORE_GATE_MAX == 65.0


def test_loss_thresholds_are_exactly_1_2_5_10():
    assert LOSS_THRESHOLDS_PCT == (1.0, 2.0, 5.0, 10.0)


def test_risk_assessment_rejects_non_monotonic_probabilities():
    with pytest.raises(ValueError):
        RiskAssessment(
            risk_score=10.0, p_loss_gt_1pct=0.1, p_loss_gt_2pct=0.5,  # increases -- invalid
            p_loss_gt_5pct=0.05, p_loss_gt_10pct=0.01,
            gap_risk_probability=0.01, liquidity_trap_risk_contribution=0.1,
            portfolio_correlation_exposure_pct=1.0, portfolio_correlation_flag=False,
            tail_risk_flag=False, fat_tail_multiplier_applied=FAT_TAIL_MULTIPLIER_NONE,
            passes_risk_gate=True, reasons=["x"],
        )


def test_risk_assessment_rejects_tail_flag_multiplier_mismatch():
    with pytest.raises(ValueError):
        RiskAssessment(
            risk_score=10.0, p_loss_gt_1pct=0.5, p_loss_gt_2pct=0.3,
            p_loss_gt_5pct=0.1, p_loss_gt_10pct=0.01,
            gap_risk_probability=0.01, liquidity_trap_risk_contribution=0.1,
            portfolio_correlation_exposure_pct=1.0, portfolio_correlation_flag=False,
            tail_risk_flag=True,  # says tail risk...
            fat_tail_multiplier_applied=FAT_TAIL_MULTIPLIER_NONE,  # ...but multiplier says no adjustment
            passes_risk_gate=True, reasons=["x"],
        )


def test_risk_assessment_rejects_invalid_multiplier_value():
    with pytest.raises(ValueError):
        RiskAssessment(
            risk_score=10.0, p_loss_gt_1pct=0.5, p_loss_gt_2pct=0.3,
            p_loss_gt_5pct=0.1, p_loss_gt_10pct=0.01,
            gap_risk_probability=0.01, liquidity_trap_risk_contribution=0.1,
            portfolio_correlation_exposure_pct=1.0, portfolio_correlation_flag=False,
            tail_risk_flag=True, fat_tail_multiplier_applied=1.75,  # not one of the three valid values
            passes_risk_gate=True, reasons=["x"],
        )


def test_assess_risk_reasons_mention_every_component():
    result = assess_risk(_inputs(volatility_state="Spike"))
    joined = " ".join(result.reasons)
    assert "fat-tail multiplier" in joined
    assert "Gap risk" in joined
    assert "Liquidity trap" in joined
    assert "correlated-direction exposure" in joined
    assert "risk score" in joined
