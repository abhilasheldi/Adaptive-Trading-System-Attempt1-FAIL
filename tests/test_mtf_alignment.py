"""
Tests for the Multi-Timeframe Alignment Engine (Layer 4),
src/macro_mtf_liquidity_vol/mtf_alignment.py.

Covers the literal "Done when" bar (the four score bands map to the four
actions exactly as specified) plus the behaviors the module docstring
commits to: the 4-timeframe-only validation, tie handling, trade_permitted
gating, and confidence being a real graded number.
"""

import pytest

from src.macro_mtf_liquidity_vol.mtf_alignment import (
    DEFAULT_NEUTRAL_BAND,
    TRADE_PERMITTED_MIN_SCORE,
    MTFAlignmentResult,
    score_mtf_alignment,
)


# ------------------------------------------------------------------------ #
# Done-when bar: all four score bands map to the four actions exactly
# ------------------------------------------------------------------------ #


def test_4_of_4_maps_to_full_position_size():
    r = score_mtf_alignment({"1H": 0.8, "4H": 0.7, "1D": 0.6, "1W": 0.9})
    assert r.score == 4
    assert r.action == "Full position size permitted"
    assert r.trade_permitted is True


def test_4_of_4_all_bearish_also_maps_correctly():
    """Alignment is direction-agnostic -- 4/4 bearish is just as much '4/4
    aligned' as 4/4 bullish."""
    r = score_mtf_alignment({"1H": -0.9, "4H": -0.8, "1D": -0.7, "1W": -0.6})
    assert r.score == 4
    assert r.action == "Full position size permitted"
    assert r.dominant_direction == "bearish"


def test_3_of_4_maps_to_standard_position_size():
    r = score_mtf_alignment({"1H": 0.8, "4H": 0.7, "1D": 0.6, "1W": -0.3})
    assert r.score == 3
    assert r.action == "Standard position size permitted"
    assert r.trade_permitted is True


def test_2_of_4_maps_to_no_entry():
    r = score_mtf_alignment({"1H": 0.8, "4H": 0.7, "1D": -0.6, "1W": -0.5})
    assert r.score == 2
    assert r.action == "No entry"
    assert r.trade_permitted is False


def test_1_of_4_maps_to_no_entry_and_review():
    r = score_mtf_alignment({"1H": 0.8, "4H": -0.1, "1D": -0.1, "1W": -0.7})
    assert r.score == 1
    assert r.action == "No entry, existing positions reviewed"
    assert r.trade_permitted is False


def test_0_of_4_maps_to_no_entry_and_review():
    r = score_mtf_alignment({"1H": 0.05, "4H": -0.05, "1D": 0.1, "1W": -0.02})
    assert r.score == 0
    assert r.action == "No entry, existing positions reviewed"
    assert r.trade_permitted is False


@pytest.mark.parametrize("score,expected_action", [
    (4, "Full position size permitted"),
    (3, "Standard position size permitted"),
    (2, "No entry"),
    (1, "No entry, existing positions reviewed"),
    (0, "No entry, existing positions reviewed"),
])
def test_result_construction_enforces_exact_score_action_mapping(score, expected_action):
    """MTFAlignmentResult itself refuses to be constructed with a
    score/action pair that doesn't match the exact table -- this can't
    drift out of sync even if a future edit changes one but not the other."""
    MTFAlignmentResult(
        score=score, action=expected_action, trade_permitted=score >= TRADE_PERMITTED_MIN_SCORE,
        dominant_direction="bullish", confidence=0.5, per_timeframe_direction={}, reasons=[],
    )
    with pytest.raises(ValueError):
        MTFAlignmentResult(
            score=score, action="wrong text", trade_permitted=score >= TRADE_PERMITTED_MIN_SCORE,
            dominant_direction="bullish", confidence=0.5, per_timeframe_direction={}, reasons=[],
        )


# ------------------------------------------------------------------------ #
# trade_permitted: "a trade is only considered when alignment is 3/4 or 4/4"
# ------------------------------------------------------------------------ #


@pytest.mark.parametrize("score", [4, 3])
def test_trade_permitted_true_for_3_and_4(score):
    action = "Full position size permitted" if score == 4 else "Standard position size permitted"
    r = MTFAlignmentResult(score=score, action=action, trade_permitted=True,
                            dominant_direction="bullish", confidence=0.5, per_timeframe_direction={},
                            reasons=[])
    assert r.trade_permitted is True


@pytest.mark.parametrize("score", [2, 1, 0])
def test_trade_permitted_false_for_2_1_0(score):
    r = score_mtf_alignment(
        {"1H": 0.05, "4H": 0.05, "1D": 0.05, "1W": 0.05}
        if score == 0
        else {"1H": 0.8, "4H": -0.8, "1D": 0.05, "1W": 0.05}
    )
    assert r.trade_permitted is False


def test_trade_permitted_field_must_be_internally_consistent():
    with pytest.raises(ValueError):
        MTFAlignmentResult(score=4, action="Full position size permitted", trade_permitted=False,
                            dominant_direction="bullish", confidence=0.5, per_timeframe_direction={},
                            reasons=[])
    with pytest.raises(ValueError):
        MTFAlignmentResult(score=2, action="No entry", trade_permitted=True,
                            dominant_direction="mixed", confidence=0.5, per_timeframe_direction={},
                            reasons=[])


# ------------------------------------------------------------------------ #
# Exactly 4 timeframes required
# ------------------------------------------------------------------------ #


def test_rejects_fewer_than_4_timeframes():
    with pytest.raises(ValueError, match="exactly 4"):
        score_mtf_alignment({"1H": 0.5, "4H": 0.5})


def test_rejects_more_than_4_timeframes():
    with pytest.raises(ValueError, match="exactly 4"):
        score_mtf_alignment({"1H": 0.5, "4H": 0.5, "1D": 0.5, "1W": 0.5, "1M": 0.5})


def test_rejects_out_of_range_direction_value():
    with pytest.raises(ValueError):
        score_mtf_alignment({"1H": 1.5, "4H": 0.5, "1D": 0.5, "1W": 0.5})


# ------------------------------------------------------------------------ #
# Confidence is a real graded number, not trivially constant
# ------------------------------------------------------------------------ #


def test_confidence_reflects_conviction_strength():
    strong = score_mtf_alignment({"1H": 0.95, "4H": 0.9, "1D": 0.85, "1W": 0.9})
    weak = score_mtf_alignment({"1H": 0.2, "4H": 0.18, "1D": 0.2, "1W": 0.19})
    assert strong.score == weak.score == 4
    assert strong.confidence > weak.confidence


def test_confidence_is_zero_when_all_neutral():
    r = score_mtf_alignment({"1H": 0.05, "4H": 0.05, "1D": 0.05, "1W": 0.05})
    assert r.confidence == 0.0


# ------------------------------------------------------------------------ #
# Per-timeframe explainability output
# ------------------------------------------------------------------------ #


def test_per_timeframe_direction_reported_for_every_input():
    r = score_mtf_alignment({"1H": 0.8, "4H": -0.1, "1D": 0.6, "1W": -0.9})
    assert r.per_timeframe_direction == {
        "1H": "bullish", "4H": "neutral", "1D": "bullish", "1W": "bearish",
    }
    assert len(r.reasons) == 5  # one line per timeframe + one summary line


def test_neutral_band_is_configurable():
    """A value that reads as neutral under the default band should read as
    a real directional signal under a tighter band."""
    tfs = {"1H": 0.8, "4H": 0.12, "1D": 0.6, "1W": 0.7}
    assert 0.12 < DEFAULT_NEUTRAL_BAND, "test fixture assumes 0.12 falls inside the default neutral band"
    default = score_mtf_alignment(tfs)
    assert default.per_timeframe_direction["4H"] == "neutral"

    tight = score_mtf_alignment(tfs, neutral_band=0.05)
    assert tight.per_timeframe_direction["4H"] == "bullish"  # 0.12 > 0.05
    assert tight.score == 4  # now all four agree


# ------------------------------------------------------------------------ #
# 2026-08-13 critical re-audit findings
# ------------------------------------------------------------------------ #


def test_tie_confidence_is_symmetric_regardless_of_which_side_is_stronger():
    """
    Regression test for a real bug found during re-audit: confidence at a
    tie ("mixed") previously used only `bullish_tfs`, so two scenarios that
    are the SAME conflict from opposite sides -- a strong pair tied against
    a barely-non-neutral pair, with the labels swapped -- reported wildly
    different confidence (0.9 vs 0.16) purely because of which arbitrary
    side happened to be "bullish." Confidence at a tie must be computed
    over BOTH conflicting sides, making it symmetric.
    """
    strong_bullish = score_mtf_alignment({"1H": 0.9, "4H": 0.9, "1D": -0.16, "1W": -0.16})
    strong_bearish = score_mtf_alignment({"1H": 0.16, "4H": 0.16, "1D": -0.9, "1W": -0.9})
    assert strong_bullish.dominant_direction == "mixed"
    assert strong_bearish.dominant_direction == "mixed"
    assert strong_bullish.confidence == pytest.approx(strong_bearish.confidence)
    # Sanity: the shared value must reflect BOTH sides, not just one --
    # mean(0.9, 0.9, 0.16, 0.16), not mean(0.9, 0.9) or mean(0.16, 0.16).
    assert strong_bullish.confidence == pytest.approx((0.9 + 0.9 + 0.16 + 0.16) / 4)


def test_exact_spec_trade_permitted_threshold_is_3():
    """Locks in the exact MASTER_CONTEXT constraint 8 / Adaptive-doc number
    ('MTF alignment >= 3/4') as a named constant -- re-audit finding: this
    was previously only a bare literal `3` with no exported, testable name,
    inconsistent with every other exact spec number in this module's three
    sibling modules."""
    assert TRADE_PERMITTED_MIN_SCORE == 3


def test_reasons_field_has_no_default_matching_sibling_modules():
    """Re-audit finding: `reasons` previously had `= field(default_factory=
    list)`, an unexplained inconsistency with macro_regime.py's,
    liquidity_trap.py's, and volatility_classifier.py's analogous
    `reasons` fields, all of which are required. Confirms omitting it now
    raises, like the other three modules' result types."""
    with pytest.raises(TypeError):
        MTFAlignmentResult(
            score=4, action="Full position size permitted", trade_permitted=True,
            dominant_direction="bullish", confidence=0.5, per_timeframe_direction={},
        )  # no `reasons=` -- must raise now that there's no default
