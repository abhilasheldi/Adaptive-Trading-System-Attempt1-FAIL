"""
Stage A.5 -- trial registry and Deflated Sharpe Ratio.

The registry's job is to be hard to quietly shrink, and DSR's job is to
make an honest trial count expensive to fake. Both are tested for the
failure direction that flatters a result.
"""

import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from src.predictive_core.dsr import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.predictive_core.trial_registry import (
    config_hash,
    distinct_configuration_count,
    log_trial,
    read_trials,
    registry_summary,
)


@pytest.fixture
def reg(tmp_path):
    return tmp_path / "trials.jsonl"


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #


def test_every_record_round_trips(reg):
    log_trial("stage_c", {"lr": 0.1}, {"auc": 0.53}, n_eff=1100.0, nominal_n=9000, path=reg)
    (rec,) = read_trials(reg)
    assert rec.stage == "stage_c"
    assert rec.params == {"lr": 0.1}
    assert rec.result_metrics["auc"] == 0.53
    assert rec.n_eff == 1100.0 and rec.nominal_n == 9000


def test_registry_is_append_only_jsonl(reg):
    for i in range(3):
        log_trial("s", {"i": i}, path=reg)
    lines = reg.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        json.loads(line)          # each line independently valid


def test_identical_configs_hash_the_same_regardless_of_key_order():
    a = config_hash({"lr": 0.1, "leaves": 31}, "stage_c")
    b = config_hash({"leaves": 31, "lr": 0.1}, "stage_c")
    assert a == b


def test_int_and_float_forms_of_the_same_value_hash_the_same():
    assert config_hash({"n": 5}, "s") == config_hash({"n": 5.0}, "s")


def test_same_params_at_different_stages_are_different_configurations():
    """The same hyperparameters evaluated at two stages are two searches of
    the hypothesis space, and DSR must count them as two."""
    assert config_hash({"lr": 0.1}, "stage_c") != config_hash({"lr": 0.1}, "stage_d")


def test_distinct_count_collapses_repeats_but_not_variations(reg):
    log_trial("s", {"lr": 0.1}, path=reg)
    log_trial("s", {"lr": 0.1}, path=reg)     # identical re-evaluation
    log_trial("s", {"lr": 0.2}, path=reg)
    assert len(read_trials(reg)) == 3
    assert distinct_configuration_count(reg) == 2


def test_n_eff_absent_is_written_as_null_not_as_nominal(reg):
    """I5/I6: an absent measurement is not a measurement of no overlap, and
    must never be silently backfilled with nominal n."""
    log_trial("s", {"a": 1}, nominal_n=5000, n_eff=None, path=reg)
    (rec,) = read_trials(reg)
    assert rec.n_eff is None
    assert json.loads(reg.read_text().strip())["n_eff"] is None


def test_corrupt_trailing_line_does_not_destroy_the_registry(reg):
    log_trial("s", {"a": 1}, path=reg)
    with open(reg, "a", encoding="utf-8") as fh:
        fh.write('{"partial": ')     # simulate a crash mid-append
    assert len(read_trials(reg)) == 1


def test_missing_registry_reads_as_empty_not_an_error(reg):
    assert read_trials(reg) == []
    assert distinct_configuration_count(reg) == 0


def test_summary_reports_how_many_records_carry_n_eff(reg):
    log_trial("s", {"a": 1}, n_eff=10.0, path=reg)
    log_trial("s", {"a": 2}, n_eff=None, path=reg)
    s = registry_summary(reg)
    assert s["total_records"] == 2 and s["with_n_eff"] == 1


def test_optuna_callback_logs_every_trial_not_only_the_best(reg):
    """A search that tried 6 configurations and kept 1 has searched 6."""
    import optuna

    from src.predictive_core.trial_registry import optuna_logging_callback

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(
        lambda t: (t.suggest_float("x", -1, 1)) ** 2,
        n_trials=6,
        callbacks=[optuna_logging_callback("stage_c", n_eff=500.0, path=reg)],
    )
    recs = read_trials(reg)
    assert len(recs) == 6
    assert all(r.source == "optuna" for r in recs)
    assert all(r.n_eff == 500.0 for r in recs)


# ---------------------------------------------------------------------- #
# DSR
# ---------------------------------------------------------------------- #


def test_expected_max_sharpe_is_zero_for_a_single_trial():
    """With one trial there is no selection to deflate away."""
    assert expected_max_sharpe(0.01, 1) == 0.0


def test_expected_max_sharpe_grows_with_trial_count():
    """The whole mechanism: more searching means a higher bar."""
    vals = [expected_max_sharpe(0.01, n) for n in (2, 10, 100, 1000, 10000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_expected_max_sharpe_matches_the_closed_form():
    v, n = 0.0025, 500
    z1 = norm.ppf(1 - 1 / n)
    z2 = norm.ppf(1 - 1 / (n * math.e))
    expected = math.sqrt(v) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    assert expected_max_sharpe(v, n) == pytest.approx(expected)


def test_psr_of_the_observed_sharpe_against_itself_is_one_half():
    """PSR(SR) = Phi(0) = 0.5, exactly, for any moments."""
    assert probabilistic_sharpe_ratio(0.05, 500, 0.0, 3.0, benchmark_sharpe=0.05) == pytest.approx(0.5)


def test_psr_rises_with_sample_length():
    a = probabilistic_sharpe_ratio(0.05, 100, 0.0, 3.0)
    b = probabilistic_sharpe_ratio(0.05, 5000, 0.0, 3.0)
    assert b > a


def test_psr_penalises_negative_skew_and_fat_tails():
    base = probabilistic_sharpe_ratio(0.10, 1000, 0.0, 3.0)
    skewed = probabilistic_sharpe_ratio(0.10, 1000, -1.5, 3.0)
    fat = probabilistic_sharpe_ratio(0.10, 1000, 0.0, 12.0)
    assert skewed < base
    assert fat < base


def test_dsr_falls_as_the_trial_count_rises():
    """The property the registry exists to feed. Same returns, more trials,
    lower DSR -- undercounting trials flatters the gate."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 2000)
    vals = [deflated_sharpe_ratio(r, n, var_trial_sharpes="null").dsr
            for n in (1, 10, 500, 10000)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))
    assert vals[0] > vals[-1]


def test_dsr_refuses_to_guess_the_trial_sharpe_variance():
    """SR* scales with sqrt(V), so a defaulted V sets the answer. Passing
    None must raise rather than quietly picking one."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 500)
    with pytest.raises(ValueError, match="var_trial_sharpes"):
        deflated_sharpe_ratio(r, 100)


def test_dsr_records_whether_the_variance_was_provided_or_assumed():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 500)
    assert deflated_sharpe_ratio(r, 50, var_trial_sharpes="null").var_source == "null-approximation"
    assert deflated_sharpe_ratio(r, 50, var_trial_sharpes=0.004).var_source == "provided"


def test_a_worthless_strategy_selected_from_many_trials_fails_the_gate():
    """The headline behaviour. Zero-mean returns, 2000 trials searched:
    DSR must not clear 0.5."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, 1500)
    res = deflated_sharpe_ratio(r, 2000, var_trial_sharpes="null")
    assert not res.passes_gate


def test_dsr_uses_non_excess_kurtosis():
    """Normal kurtosis is 3.0, not 0.0. Passing excess kurtosis by mistake
    must produce a different (more permissive) number, so the distinction
    cannot be silently wrong."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.002, 0.01, 1000)
    normal_k = deflated_sharpe_ratio(r, 100, skew=0.0, kurtosis=3.0, var_trial_sharpes="null")
    excess_k = deflated_sharpe_ratio(r, 100, skew=0.0, kurtosis=0.0, var_trial_sharpes="null")
    assert normal_k.dsr != excess_k.dsr


def test_dsr_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.01], 10, var_trial_sharpes="null")
    with pytest.raises(ValueError, match="zero variance"):
        deflated_sharpe_ratio([0.01] * 50, 10, var_trial_sharpes="null")
