"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

BUILD_PLAN_v2 A.5. DSR is a Stage E gate (MASTER_CONTEXT_v2 section 6:
DSR > 0.5), and it is the mechanism that makes the trial registry
load-bearing rather than bookkeeping.

WHAT IT ANSWERS
Given that N configurations were tried, what is the probability that the
best observed Sharpe reflects genuine skill rather than the maximum of N
draws from a zero-skill distribution? Selecting the best of many backtests
guarantees a positive Sharpe even when every strategy is worthless; DSR
prices that selection in, and also corrects for non-normal returns, which
the plain Sharpe ratio assumes away.

THE FORMULA

    Probabilistic Sharpe Ratio against a benchmark SR*:

        PSR(SR*) = Phi[ (SR - SR*) * sqrt(T - 1)
                        / sqrt(1 - g3*SR + ((g4 - 1)/4) * SR^2) ]

        SR = observed Sharpe, PER OBSERVATION (not annualised)
        T  = number of return observations
        g3 = skewness of returns
        g4 = kurtosis of returns, NON-EXCESS (normal = 3.0)

    Expected maximum Sharpe under the null across N independent trials:

        SR* = sqrt(V) * [ (1 - gamma) * Phi^-1(1 - 1/N)
                          + gamma * Phi^-1(1 - 1/(N*e)) ]

        V     = variance of the Sharpe ratios ACROSS trials
        gamma = Euler-Mascheroni constant, 0.5772...

    DSR = PSR(SR*)

THE PART THAT IS EASY TO FUDGE
`V`, the cross-trial variance of Sharpe ratios, is not optional and is not
derivable from the returns of a single strategy. It has to come from the
trial registry. Supplying it silently -- or defaulting it -- sets the answer,
because SR* scales directly with sqrt(V).

This implementation therefore REFUSES to guess: `var_trial_sharpes` must be
a float, or the literal string "null" to opt in explicitly to the
zero-skill null approximation V ~ 1/T. Passing None raises. A DSR computed
from an invented V looks rigorous and is not, which is the specific failure
mode this project exists to avoid.

ANNUALISATION: none, anywhere. SR here is per-observation. Passing an
annualised Sharpe with a per-observation T silently inflates DSR by roughly
sqrt(periods_per_year).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class DSRResult:
    """DSR plus every input that determined it, so a reported number can be
    audited without re-running the computation."""

    dsr: float
    observed_sharpe: float             # per observation
    expected_max_sharpe: float         # SR*
    n_observations: int
    n_trials: int
    skew: float
    kurtosis: float                    # non-excess
    var_trial_sharpes: float
    var_source: str                    # 'provided' | 'null-approximation'

    @property
    def passes_gate(self) -> bool:
        """MASTER_CONTEXT_v2 section 6 / BUILD_PLAN_v2 Gate E: DSR > 0.5."""
        return self.dsr > 0.5

    def summary(self) -> str:
        return (
            f"DSR={self.dsr:.4f} (gate >0.5: {'PASS' if self.passes_gate else 'FAIL'})  "
            f"SR={self.observed_sharpe:.4f}/obs  SR*={self.expected_max_sharpe:.4f}  "
            f"T={self.n_observations:,}  N_trials={self.n_trials:,}  "
            f"skew={self.skew:.3f}  kurt={self.kurtosis:.3f}  "
            f"V={self.var_trial_sharpes:.3e} ({self.var_source})"
        )


def expected_max_sharpe(var_trial_sharpes: float, n_trials: int) -> float:
    """SR* -- the Sharpe you expect from the BEST of N worthless strategies.

    Grows with N. This is why an honest trial count matters: undercount N
    and SR* is too low, so DSR is too high, so a worthless strategy clears
    the gate.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if var_trial_sharpes < 0:
        raise ValueError("var_trial_sharpes must be non-negative.")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(var_trial_sharpes) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    skew: float,
    kurtosis: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """PSR: probability the true Sharpe exceeds `benchmark_sharpe`.

    `kurtosis` is NON-EXCESS (3.0 for a normal distribution). Passing excess
    kurtosis understates the correction.
    """
    if n_observations < 2:
        raise ValueError("Need at least 2 observations.")
    sr = float(observed_sharpe)
    denom_sq = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        # Possible under extreme skew/kurtosis combined with a large SR.
        # Raising beats returning a NaN that gets averaged into a report.
        raise ValueError(
            f"PSR variance term is non-positive ({denom_sq:.6g}); the "
            f"skew/kurtosis/Sharpe combination is outside the range where this "
            f"approximation is valid (SR={sr:.4f}, skew={skew:.4f}, kurt={kurtosis:.4f})."
        )
    z = (sr - benchmark_sharpe) * math.sqrt(n_observations - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    var_trial_sharpes: Union[float, str, None] = None,
) -> DSRResult:
    """Deflated Sharpe Ratio.

    Parameters
    ----------
    returns : sequence of per-period returns (NOT annualised)
    n_trials : number of DISTINCT configurations evaluated. Take this from
        `trial_registry.distinct_configuration_count()`, never from memory.
    skew, kurtosis : moments of `returns`. Computed from the data when None.
        `kurtosis` is NON-EXCESS (normal = 3.0).
    var_trial_sharpes : variance of Sharpe ratios ACROSS trials.
        - float  : use it (compute from the registry's logged Sharpes)
        - "null" : opt in to the zero-skill approximation V ~ 1/T
        - None   : RAISES. See this module's docstring -- guessing V sets
                   the answer, and a fabricated V produces a number that
                   looks rigorous and is not.
    """
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 2:
        raise ValueError(f"Need at least 2 finite returns; got {t}.")

    sd = r.std(ddof=1)
    if sd <= 0:
        raise ValueError("Return series has zero variance; Sharpe is undefined.")
    sr = float(r.mean() / sd)

    if skew is None:
        from scipy.stats import skew as _skew
        skew = float(_skew(r, bias=False))
    if kurtosis is None:
        from scipy.stats import kurtosis as _kurt
        kurtosis = float(_kurt(r, fisher=False, bias=False))   # non-excess

    if var_trial_sharpes is None:
        raise ValueError(
            "var_trial_sharpes is required and will not be guessed. Supply the "
            "variance of Sharpe ratios across the logged trials, or pass "
            "var_trial_sharpes='null' to explicitly adopt the zero-skill "
            "approximation V ~ 1/T. SR* scales with sqrt(V), so an invented V "
            "sets the DSR."
        )
    if isinstance(var_trial_sharpes, str):
        if var_trial_sharpes != "null":
            raise ValueError("var_trial_sharpes string must be 'null'.")
        # Under H0 (all trials zero true Sharpe, returns iid), the estimated
        # per-observation Sharpe has variance ~ 1/T.
        v, source = 1.0 / t, "null-approximation"
    else:
        v, source = float(var_trial_sharpes), "provided"

    sr_star = expected_max_sharpe(v, n_trials)
    dsr = probabilistic_sharpe_ratio(sr, t, skew, kurtosis, benchmark_sharpe=sr_star)

    return DSRResult(
        dsr=dsr,
        observed_sharpe=sr,
        expected_max_sharpe=sr_star,
        n_observations=t,
        n_trials=n_trials,
        skew=float(skew),
        kurtosis=float(kurtosis),
        var_trial_sharpes=v,
        var_source=source,
    )


__all__ = [
    "DSRResult",
    "EULER_MASCHERONI",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
]
