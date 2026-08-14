"""
Stage A.2 -- sample uniqueness and effective n (AFML ch. 4).

Includes closed-form cases where u_bar and n_eff can be derived by hand, so
the implementation is checked against arithmetic rather than against itself.
"""

import numpy as np
import pandas as pd
import pytest

from src.predictive_core.uniqueness import (
    average_uniqueness,
    uniqueness_sample_weights,
)


def _ts(hours):
    base = pd.Timestamp("2022-01-01", tz="UTC")
    return [base + pd.Timedelta(hours=int(h)) for h in hours]


# ---------------------------------------------------------------------- #
# Hand-derivable cases
# ---------------------------------------------------------------------- #


def test_non_overlapping_labels_are_perfectly_unique():
    """Disjoint windows: c_t == 1 on every spanned bar, so u_bar == 1 and
    n_eff == nominal n exactly."""
    r = average_uniqueness(_ts([0, 10, 20]), _ts([5, 15, 25]))
    assert np.allclose(r.average_uniqueness, 1.0)
    assert r.effective_n == pytest.approx(3.0)
    assert r.ratio == pytest.approx(1.0)
    assert r.max_concurrency == 1


def test_two_perfectly_coincident_labels_are_half_unique():
    """Identical windows: c_t == 2 everywhere, u_bar == 0.5 each,
    n_eff == 1.0 for 2 nominal samples."""
    r = average_uniqueness(_ts([0, 0]), _ts([10, 10]))
    assert np.allclose(r.average_uniqueness, 0.5)
    assert r.effective_n == pytest.approx(1.0)
    assert r.max_concurrency == 2


def test_k_coincident_labels_give_n_eff_of_one():
    for k in (2, 3, 5, 9):
        r = average_uniqueness(_ts([0] * k), _ts([10] * k))
        assert np.allclose(r.average_uniqueness, 1.0 / k)
        assert r.effective_n == pytest.approx(1.0)


def test_partial_overlap_matches_hand_computed_value():
    """A: bars 0..2, B: bars 1..3 (bar axis = {0,1,2,3}).

    c = [1, 2, 2, 1].
    A spans bars 0,1,2 -> u_bar_A = (1/1 + 1/2 + 1/2)/3 = 2/3
    B spans bars 1,2,3 -> u_bar_B = (1/2 + 1/2 + 1/1)/3 = 2/3
    n_eff = 4/3
    """
    r = average_uniqueness(_ts([0, 1]), _ts([2, 3]))
    assert np.allclose(r.average_uniqueness, [2 / 3, 2 / 3])
    assert r.effective_n == pytest.approx(4 / 3)


def test_single_bar_label_is_fully_unique():
    r = average_uniqueness(_ts([0]), _ts([0]))
    assert r.average_uniqueness[0] == pytest.approx(1.0)
    assert r.effective_n == pytest.approx(1.0)


# ---------------------------------------------------------------------- #
# Per-asset grouping -- the correctness property most easily got wrong
# ---------------------------------------------------------------------- #


def test_labels_on_different_assets_do_not_dilute_each_other():
    """A BTC label and an ETH label spanning the same hour read different
    information. Without grouping they would each be scored 0.5 unique."""
    t0, t1 = _ts([0, 0]), _ts([10, 10])
    grouped = average_uniqueness(t0, t1, groups=["BTC", "ETH"])
    assert np.allclose(grouped.average_uniqueness, 1.0)
    assert grouped.effective_n == pytest.approx(2.0)

    ungrouped = average_uniqueness(t0, t1)
    assert np.allclose(ungrouped.average_uniqueness, 0.5)
    assert ungrouped.effective_n == pytest.approx(1.0)


def test_grouping_still_penalises_overlap_within_one_asset():
    """Grouping must not become a blanket exemption: two coincident BTC
    labels are still half-unique even when an ETH label is present."""
    r = average_uniqueness(
        _ts([0, 0, 0]), _ts([10, 10, 10]), groups=["BTC", "BTC", "ETH"]
    )
    assert np.allclose(r.average_uniqueness, [0.5, 0.5, 1.0])
    assert r.effective_n == pytest.approx(2.0)


def test_group_results_are_returned_in_input_order():
    """Interleaved groups must map back to their original positions."""
    r = average_uniqueness(
        _ts([0, 0, 0, 0]), _ts([5, 5, 5, 5]), groups=["BTC", "ETH", "BTC", "ETH"]
    )
    assert np.allclose(r.average_uniqueness, [0.5, 0.5, 0.5, 0.5])
    assert r.effective_n == pytest.approx(2.0)


# ---------------------------------------------------------------------- #
# restrict_to -- uniqueness is a property of the sample SET
# ---------------------------------------------------------------------- #


def test_restricting_to_a_subset_raises_uniqueness_of_survivors():
    """Dropping the overlapping neighbours makes the survivors more unique.
    Computing on everything and then masking gives a DIFFERENT number, which
    is why the choice has to be explicit."""
    t0, t1 = _ts([0, 1, 2, 3]), _ts([2, 3, 4, 5])
    mask = np.array([True, False, False, True])   # keep the disjoint pair

    restricted = average_uniqueness(t0, t1, restrict_to=mask)
    assert len(restricted.average_uniqueness) == 2
    assert np.allclose(restricted.average_uniqueness, 1.0)   # now disjoint
    assert restricted.effective_n == pytest.approx(2.0)

    full = average_uniqueness(t0, t1)
    assert (full.average_uniqueness[mask] < 1.0).all()       # diluted in situ


def test_restrict_to_length_must_match():
    with pytest.raises(ValueError, match="restrict_to"):
        average_uniqueness(_ts([0, 1]), _ts([2, 3]), restrict_to=np.array([True]))


# ---------------------------------------------------------------------- #
# Invariants that must hold on any input
# ---------------------------------------------------------------------- #


def test_uniqueness_is_always_in_zero_one_and_n_eff_never_exceeds_nominal():
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(2, 60))
        starts = rng.integers(0, 200, size=n)
        spans = rng.integers(0, 30, size=n)
        r = average_uniqueness(_ts(starts), _ts(starts + spans))
        assert (r.average_uniqueness > 0).all()
        assert (r.average_uniqueness <= 1.0 + 1e-12).all()
        assert 0 < r.effective_n <= n + 1e-9


def test_rejects_t1_before_t0():
    with pytest.raises(ValueError, match="t1 >= t0"):
        average_uniqueness(_ts([10]), _ts([5]))


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        average_uniqueness(_ts([0, 1]), _ts([5]))


def test_empty_input_returns_zero_not_nan_n_eff():
    r = average_uniqueness([], [])
    assert r.nominal_n == 0
    assert r.effective_n == 0.0


# ---------------------------------------------------------------------- #
# Sample weights
# ---------------------------------------------------------------------- #


def test_weights_normalise_to_mean_one_without_changing_ratios():
    t0, t1 = _ts([0, 0, 50]), _ts([10, 10, 60])
    w, r = uniqueness_sample_weights(t0, t1)
    assert w.mean() == pytest.approx(1.0)
    # Relative ordering preserved: the coincident pair stays half as
    # influential as the isolated label.
    assert w[0] == pytest.approx(w[1])
    assert w[2] == pytest.approx(2.0 * w[0])
    # n_eff computed from UNNORMALISED u_bar, so normalisation cannot inflate it.
    assert r.effective_n == pytest.approx(2.0)


def test_unnormalised_weights_are_the_raw_uniqueness():
    t0, t1 = _ts([0, 0]), _ts([10, 10])
    w, r = uniqueness_sample_weights(t0, t1, normalise=False)
    assert np.allclose(w, r.average_uniqueness)
    assert np.allclose(w, 0.5)


# ---------------------------------------------------------------------- #
# I6 wiring: n_eff must travel with every result object
# ---------------------------------------------------------------------- #


def _pooled(n=900):
    from src.predictive_core.labeling import apply_triple_barrier, pool_assets
    from tests.synthetic_data import make_synthetic_asset

    btc = apply_triple_barrier(
        make_synthetic_asset(n, "2022-01-01", "1h", seed=1, drift=0.01), max_candles=24
    )
    eth = apply_triple_barrier(
        make_synthetic_asset(n, "2022-01-01", "1h", seed=2, drift=0.01), max_candles=24
    )
    return pool_assets({"BTC": btc, "ETH": eth})


def test_quantile_result_reports_n_eff_for_every_partition():
    """Invariant I6: nominal n never travels alone."""
    from src.predictive_core.boosted_core import train_quantile_regressor
    from tests.synthetic_data import FEATURE_COLS

    p = _pooled()
    tr, va, te = p.iloc[:1000], p.iloc[1000:1300], p.iloc[1300:]
    _, r = train_quantile_regressor(
        tr, va, te, FEATURE_COLS, ["asset"], "magnitude", 0,
        n_optuna_trials=1, early_stopping_rounds=5,
    )
    assert r.uniqueness_weighted is True
    for nominal, n_eff in (
        (r.n_train, r.n_eff_train), (r.n_val, r.n_eff_val), (r.n_test, r.n_eff_test)
    ):
        assert n_eff is not None
        assert 0 < n_eff < nominal          # overlap must reduce it, strictly
    assert 0 < r.n_eff_ratio_train() < 1


def test_missing_label_metadata_reports_none_not_a_fabricated_n_eff():
    """Without t0/t1 uniqueness is not computable. The result must say so,
    not silently report weighting it did not apply or an n_eff of nominal --
    an absent measurement is not a measurement of no overlap (I5)."""
    from src.predictive_core.boosted_core import train_quantile_regressor
    from tests.synthetic_data import FEATURE_COLS

    p = _pooled()
    tr, va, te = p.iloc[:1000], p.iloc[1000:1300], p.iloc[1300:]
    _, r = train_quantile_regressor(
        tr.drop(columns=["t0", "t1"]), va, te, FEATURE_COLS, ["asset"], "magnitude", 0,
        n_optuna_trials=1, early_stopping_rounds=5,
    )
    assert r.uniqueness_weighted is False
    assert r.n_eff_train is None
    assert r.n_eff_ratio_train() is None
