"""
Stage A.4 fix 4 / Invariant I3 -- ablation flagging gates on solvency and
maxDD/PF before Sharpe.

The headline test replays the actual 2026-08-14 ablation numbers and
asserts the new rule reaches the opposite (correct) verdict to the old
Sharpe-only rule.
"""

import numpy as np
import pytest

from src.backtesting.ablation_flagging import (
    RUIN_MAX_DD,
    AblationVerdict,
    VariantOutcome,
    flag_degrading_layers,
    rank_variants,
)


def _v(name, sharpe, pf, dd, trades=1000, destroyed=False):
    return VariantOutcome(name, sharpe, pf, dd, trades, destroyed)


# ---------------------------------------------------------------------- #
# Solvency gate
# ---------------------------------------------------------------------- #


def test_maxdd_at_or_above_one_is_insolvent():
    assert not _v("x", 1.0, 2.0, 1.0).is_solvent
    assert not _v("x", 1.0, 2.0, 1.235).is_solvent
    assert _v("x", 1.0, 2.0, 0.999).is_solvent


def test_equity_destroyed_flag_overrides_a_healthy_looking_maxdd():
    assert not _v("x", 3.0, 5.0, 0.05, destroyed=True).is_solvent


def test_nan_maxdd_is_treated_as_insolvent_not_as_passing():
    """An unmeasurable drawdown is not a good drawdown."""
    assert not _v("x", 1.0, 2.0, float("nan")).is_solvent


def test_ranking_is_refused_when_any_variant_destroyed_equity():
    """Stronger than excluding the insolvent ones. An ablation compares
    variants against each other, and a comparison set containing a destroyed
    account has no common scale."""
    verdict = rank_variants([
        _v("healthy", 1.5, 2.0, 0.10),
        _v("blown_up", 2.9, 0.01, 1.235),
    ])
    assert verdict.rankable is False
    assert verdict.ranked == []
    assert [v.name for v in verdict.insolvent] == ["blown_up"]
    assert "not a measurement" in verdict.reason


def test_solvent_set_ranks_on_maxdd_before_sharpe():
    """The variant with the best Sharpe must NOT win if its drawdown is worse."""
    verdict = rank_variants([
        _v("high_sharpe_deep_dd", 3.0, 1.1, 0.40),
        _v("low_sharpe_shallow_dd", 0.4, 1.05, 0.05),
    ])
    assert verdict.rankable is True
    assert verdict.ranked[0].name == "low_sharpe_shallow_dd"


def test_profit_factor_breaks_ties_before_sharpe():
    verdict = rank_variants([
        _v("better_sharpe", 2.0, 1.1, 0.10),
        _v("better_pf", 0.5, 3.0, 0.10),
    ])
    assert verdict.ranked[0].name == "better_pf"


def test_empty_input_is_not_rankable():
    assert rank_variants([]).rankable is False


# ---------------------------------------------------------------------- #
# THE REGRESSION: the 2026-08-14 ablation's real numbers
# ---------------------------------------------------------------------- #


def test_replays_the_2026_08_14_ablation_and_reaches_the_opposite_verdict():
    """Real numbers from reports/layer_ablation.json.

    The old Sharpe-only rule flagged Layer 6 (removing it RAISED Sharpe from
    -8.284 to -4.846) and reported Layer 7C as HELPING (+4.075 Sharpe).

    Both runs had maxDD above 1.0 -- equity through zero -- so both Sharpes
    were arithmetic on a destroyed account. The correct response is to refuse
    the ranking entirely and read maxDD/PF instead, where 7C is visibly the
    destroyer: removing it takes maxDD from 1.023 to 0.056 and PF from 0.014
    to 0.238."""
    full_stack = _v("all_layers", -8.284, 0.014, 1.023, 2817)
    loo = {
        "layer5_liquidity_trap": _v("no_L5", -8.284, 0.014, 1.023, 2817),
        "layer6_volatility_gate": _v("no_L6", -4.846, 0.008, 1.235, 2813),
        "layer7c_ensemble": _v("no_L7C", -12.359, 0.238, 0.056, 3499),
    }

    # 1. No ranking is published at all -- these accounts are destroyed.
    verdict = rank_variants([full_stack, *loo.values()])
    assert verdict.rankable is False

    flagged = flag_degrading_layers(full_stack, loo)

    # 2. Layer 7C IS flagged. The old rule called it the biggest HELP.
    assert "layer7c_ensemble" in flagged
    reasons = " ".join(flagged["layer7c_ensemble"])
    assert "solvency" in reasons
    assert "drawdown" in reasons and "profit factor" in reasons

    # 3. Sharpe is never consulted for 7C: both runs were insolvent.
    assert "Sharpe" not in reasons

    # 4. Layer 6 is NOT flagged for its Sharpe "improvement" -- both runs
    #    insolvent, so that comparison is inadmissible. The old rule flagged
    #    it, and flagged ONLY it.
    l6 = " ".join(flagged.get("layer6_volatility_gate", []))
    assert "Sharpe" not in l6


def test_sharpe_is_consulted_when_both_runs_are_solvent():
    """The gate is an ordering, not a blanket ban on Sharpe."""
    full = _v("full", 0.5, 1.4, 0.10)
    loo = {"layerX": _v("no_X", 1.9, 1.4, 0.10)}
    reasons = " ".join(flag_degrading_layers(full, loo)["layerX"])
    assert "Sharpe" in reasons


def test_a_layer_that_helps_everything_is_not_flagged():
    full = _v("full", 2.0, 2.5, 0.05)
    loo = {"good_layer": _v("no_good", 0.2, 1.1, 0.30)}
    assert flag_degrading_layers(full, loo) == {}
