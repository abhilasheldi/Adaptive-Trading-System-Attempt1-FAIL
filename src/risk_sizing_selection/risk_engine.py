"""
Risk Engine (Layer 9).

Per the Adaptive doc's Layer 9 section: "quantify the downside of each
potential trade to populate the loss side of the EV equation with
realistic numbers." Prereqs (Prompt Playbook 2G): 1C (Component B's
magnitude quantiles, `predictive_core.boosted_core`) and 2D (Volatility
Classifier, Layer 6, `macro_mtf_liquidity_vol.volatility_classifier`).

STAGE 0 CAVEAT: this module's `magnitude_p10_pct` input was previously
sourced from Component B via the deleted strategy adapters. Component A and
the direction classifier were deleted at Stage 0; the magnitude quantile
regressor survives, but nothing currently wires it to this engine. Layer 9
therefore has no live caller until Stage E.

Core outputs (exact spec list): P(loss > 1%), P(loss > 2%), P(loss > 5%),
P(loss > 10%), a 0-100 risk score, and "tail risk flag, portfolio
correlation risk flag, fat-tail multiplier applied."

Two inputs this module does NOT compute, and accepts as already-computed
(`RiskEngineInputs.stop_distance_pct` / `.position_size_pct` /
`.existing_correlated_direction_exposure_pct`): stop-loss placement is
Layer 11's job ("combined with the stop-loss placement from Layer 11",
verbatim), and position size / portfolio exposure aggregation is Layer
10's job. Neither layer exists in this codebase yet -- same "accept the
already-computed upstream input, don't invent the missing layer" pattern
used throughout this project (`mtf_alignment.py` accepting already-derived
directional reads).

P(loss > T) -- a documented judgment call, not a spec formula. The
Adaptive doc says these are "derived from the 10th-percentile magnitude
estimate from Component B... combined with the stop-loss placement from
Layer 11," but gives no closed-form equation. Component B's magnitude
regressor predicts MFE (Maximum Favorable Excursion) TOWARD the realized
direction (see `labeling.py`'s `apply_triple_barrier` -- `magnitude` is
always measured in the favorable direction, never a loss magnitude
directly); there is no adverse/loss-magnitude quantile anywhere in this
codebase to consult directly. This module uses the 10th-percentile
(conservative/small) magnitude estimate as a general SCALE for how far
price typically travels over the holding horizon, and applies a standard,
named exceedance-probability functional form (exponential decay, a common
choice for modeling "probability of exceeding a threshold decays with
distance relative to a typical scale" -- not invented here, but also not
literally fit to this asset's historical loss distribution) split into two
regimes at the stop-loss boundary:

    T <= stop_distance_pct:  P(loss > T) = exp(-T / scale)
    T >  stop_distance_pct:  P(loss > T) = min(gap_risk, exp(-stop/scale))
                                            * exp(-(T - stop_distance_pct) / scale)

The split exists because, under normal (non-gapping) execution, the
stop-loss order closes the position once price reaches `stop_distance_pct`
-- loss beyond that level can only happen if price GAPS THROUGH the stop
without executing there. This is exactly why gap risk is a separate named
"tail risk component" in the source doc rather than folded silently into
one continuous curve: crossing the stop boundary is gated by gap risk, not
by the same smooth decay that governs ordinary path risk below it. The
`min(gap_risk, exp(-stop/scale))` term is a monotonicity guard (see
`_loss_probability`'s docstring) -- without it, a large stop distance
relative to `scale` could make P(loss > T) INCREASE just past the stop
boundary, which is not a valid probability curve.

Gap risk -- session-classified, NOT literally historical gap frequency.
The Adaptive doc says gap risk should be "estimated from historical gap
frequency in the asset at the relevant session times" -- that requires a
dedicated historical-gap-analysis pipeline (scanning real OHLCV for
discontinuous session-open moves) this codebase does not have; building
one is a data-engineering task well beyond this module's scope. Instead,
`_gap_risk_probability` classifies a trade's entry timestamp into one of
three UTC-based sessions (weekend / overnight / weekday-intraday) and
applies a documented, tunable multiplier -- weekend > overnight > weekday,
matching the spec's own qualitative ordering ("higher for crypto weekends
and overnight liquidity") -- WITHOUT claiming these multipliers are
calibrated against this asset's real gap history. A caller with real
historical gap-frequency estimates can override the defaults directly
(`gap_risk_base`/`gap_risk_overnight_multiplier`/`gap_risk_weekend_multiplier`).
Crypto trades 24/7 (no literal market close), so "weekend"/"overnight"
here means thinned liquidity, not a closed exchange -- per the source
doc's own framing ("thin weekend and overnight liquidity").

Liquidity trap risk -- direct reuse of Layer 5's own output
(`liquidity_trap.LiquidityTrapResult.trap_probability`), scaled by
position size per the spec's explicit instruction ("more relevant at
larger position sizes; less relevant for BTC/ETH majors at retail
sizing") -- see `_liquidity_trap_contribution`.

Portfolio correlation risk -- flags when `position_size_pct +
existing_correlated_direction_exposure_pct` (the caller's own aggregate of
OTHER open positions directionally correlated with this one) reaches
`DEFAULT_CORRELATION_APPROACH_FRACTION` (a judgment call, not a spec
number) of the exact MASTER_CONTEXT 6% correlated-direction cap
(`CORRELATED_DIRECTION_CAP_PCT`) -- "flags when combined directional
exposure APPROACHES the 6% cap", not merely when it exceeds it, per the
spec's own wording.

Fat-tail multiplier -- exact spec range (1.5x-2.0x), state-to-endpoint
mapping is the judgment call: Spike (Layer 6's own "tail-risk event"
state, its most severe) gets the range's upper endpoint (2.0x); Extended
("mature trend/blow-off", less severe than Spike) gets the lower endpoint
(1.5x). Compressed/Expanding get 1.0x (no adjustment) -- this is the
literal Done-when bar: the multiplier must visibly change output ONLY in
Extended/Spike, never in Compressed/Expanding. Applied to the four P(loss
> T) probabilities BEFORE the risk score is derived from them, so the
score reflects the elevated regime automatically -- one source of truth,
not two separate multiplier applications.

Risk score (0-100) -- another documented judgment call (no spec formula
given for combining these components into one number): an explicit,
auditable weighted sum of four independently-bounded sub-scores (see
`_compute_risk_score`) -- P(loss > 2%) contributes up to 60 points (the
2% threshold chosen as the "headline" level since it matches
MASTER_CONTEXT's own single-trade risk cap), liquidity trap risk
contributes up to 15, gap risk up to 15, portfolio correlation exposure's
fraction of the 6% cap up to 10. `RISK_SCORE_GATE_MAX = 65.0` (exact
MASTER_CONTEXT constraint 8 number, "risk score <= 65/100") is echoed here
as `RiskAssessment.passes_risk_gate` for context -- this module does not
itself enforce the six-gate check (that is Layer 11's job), it only
computes the score the gate is later evaluated against.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from src.macro_mtf_liquidity_vol.liquidity_trap import LiquidityTrapResult
from src.macro_mtf_liquidity_vol.volatility_classifier import VolatilityStateResult

# ------------------------------------------------------------------------ #
# Exact MASTER_CONTEXT / Adaptive-doc numbers
# ------------------------------------------------------------------------ #

# Adaptive doc, Layer 9: the four exact drawdown levels the "Core outputs"
# list names verbatim.
LOSS_THRESHOLDS_PCT: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

# MASTER_CONTEXT constraint 6 / Layer 10: "Maximum correlated-direction
# exposure: 6% of capital."
#
# B.0.6 -- "of capital" now means OF GBP CAPITAL. The percentage itself is
# currency-invariant (2% of GBP equity buys the same position as 2% of the
# equivalent USD equity; the rate cancels), so this constant does not
# change. What changes is the BASE it multiplies, and that base is pinned
# by `backtesting.currency.RiskCapBasis` -- which E.1 must read when it
# builds the enforcement these constants still lack (S10). Scoring here is
# not enforcement.
CORRELATED_DIRECTION_CAP_PCT = 6.0

# MASTER_CONTEXT constraint 8's six-gate requirement: "risk score <= 65/100."
RISK_SCORE_GATE_MAX = 65.0

# Adaptive doc, Layer 9: "multiply all tail loss probabilities by
# 1.5x-2.0x" -- both endpoints are exact; the state-to-endpoint mapping is
# a judgment call (see module docstring).
FAT_TAIL_MULTIPLIER_EXTENDED = 1.5
FAT_TAIL_MULTIPLIER_SPIKE = 2.0
FAT_TAIL_MULTIPLIER_NONE = 1.0

# ------------------------------------------------------------------------ #
# Judgment-call defaults (NOT MASTER_CONTEXT/Adaptive-doc exact numbers --
# see module docstring for what each controls and why no exact number
# exists in the source docs).
# ------------------------------------------------------------------------ #

DEFAULT_GAP_RISK_BASE = 0.01  # weekday-intraday baseline P(stop-through gap)
DEFAULT_GAP_RISK_OVERNIGHT_MULTIPLIER = 2.0
DEFAULT_GAP_RISK_WEEKEND_MULTIPLIER = 3.0
DEFAULT_OVERNIGHT_START_HOUR_UTC = 0
DEFAULT_OVERNIGHT_END_HOUR_UTC = 6  # [start, end) -- thin-liquidity hours ahead of the European session

DEFAULT_LIQUIDITY_TRAP_FULL_WEIGHT_SIZE_PCT = 5.0  # position size at which liquidity trap risk reaches full weight
DEFAULT_CORRELATION_APPROACH_FRACTION = 0.8  # "approaching" the 6% cap == 80% of it
DEFAULT_GAP_RISK_SCORE_REFERENCE = 0.05  # gap risk level treated as "maximal" for risk-score scaling


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ------------------------------------------------------------------------ #
# Inputs
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class RiskEngineInputs:
    """
    Everything needed to assess one candidate (or open) trade's downside.

    Units, all consistently in PERCENTAGE POINTS of entry price (e.g. 1.2
    means 1.2%, not 0.012) -- `magnitude_p10_pct` and `stop_distance_pct`
    must be pre-converted by the caller to match this convention, the same
    way `predictive_core_strategy.py` converts Component B's absolute-price
    magnitude output to a fraction of `close` before use.
    """

    magnitude_p10_pct: float  # Component B's 10th-percentile magnitude estimate, as % of entry price
    stop_distance_pct: float  # Layer 11's stop-loss distance from entry, as % of entry price
    entry_timestamp: pd.Timestamp  # for gap-risk session classification
    volatility_state: VolatilityStateResult  # Layer 6
    liquidity_trap: LiquidityTrapResult  # Layer 5
    position_size_pct: float  # this trade's own size, % of capital
    existing_correlated_direction_exposure_pct: float  # sum of OTHER open positions' size, same directional bucket

    def __post_init__(self) -> None:
        if self.magnitude_p10_pct < 0:
            raise ValueError(f"magnitude_p10_pct must be non-negative, got {self.magnitude_p10_pct}.")
        if self.stop_distance_pct <= 0:
            raise ValueError(f"stop_distance_pct must be positive, got {self.stop_distance_pct}.")
        if not isinstance(self.entry_timestamp, pd.Timestamp):
            raise TypeError("entry_timestamp must be a pd.Timestamp.")
        if not isinstance(self.volatility_state, VolatilityStateResult):
            raise TypeError(
                f"volatility_state must be a VolatilityStateResult, got {type(self.volatility_state).__name__}."
            )
        if not isinstance(self.liquidity_trap, LiquidityTrapResult):
            raise TypeError(
                f"liquidity_trap must be a LiquidityTrapResult, got {type(self.liquidity_trap).__name__}."
            )
        if self.position_size_pct < 0:
            raise ValueError(f"position_size_pct must be non-negative, got {self.position_size_pct}.")
        if self.existing_correlated_direction_exposure_pct < 0:
            raise ValueError(
                "existing_correlated_direction_exposure_pct must be non-negative, got "
                f"{self.existing_correlated_direction_exposure_pct}."
            )


# ------------------------------------------------------------------------ #
# Output
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: float  # [0, 100]
    p_loss_gt_1pct: float
    p_loss_gt_2pct: float
    p_loss_gt_5pct: float
    p_loss_gt_10pct: float
    gap_risk_probability: float
    liquidity_trap_risk_contribution: float
    portfolio_correlation_exposure_pct: float
    portfolio_correlation_flag: bool
    tail_risk_flag: bool  # True iff volatility_state.state in {"Extended", "Spike"}
    fat_tail_multiplier_applied: float  # 1.0 | 1.5 | 2.0
    passes_risk_gate: bool  # risk_score <= RISK_SCORE_GATE_MAX (echoed for context; see module docstring)
    reasons: List[str]

    def __post_init__(self) -> None:
        if not (0.0 <= self.risk_score <= 100.0):
            raise ValueError(f"risk_score must be in [0, 100], got {self.risk_score}.")
        for name in ("p_loss_gt_1pct", "p_loss_gt_2pct", "p_loss_gt_5pct", "p_loss_gt_10pct",
                     "gap_risk_probability", "liquidity_trap_risk_contribution"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}.")
        if self.fat_tail_multiplier_applied not in (
            FAT_TAIL_MULTIPLIER_NONE, FAT_TAIL_MULTIPLIER_EXTENDED, FAT_TAIL_MULTIPLIER_SPIKE,
        ):
            raise ValueError(f"invalid fat_tail_multiplier_applied {self.fat_tail_multiplier_applied}.")
        expected_tail_flag = self.fat_tail_multiplier_applied != FAT_TAIL_MULTIPLIER_NONE
        if self.tail_risk_flag != expected_tail_flag:
            raise ValueError(
                f"tail_risk_flag={self.tail_risk_flag} inconsistent with "
                f"fat_tail_multiplier_applied={self.fat_tail_multiplier_applied}."
            )
        # P(loss > T) must be non-increasing in T -- a real probability
        # exceedance curve, not four independently-fabricated numbers.
        ordered = [self.p_loss_gt_1pct, self.p_loss_gt_2pct, self.p_loss_gt_5pct, self.p_loss_gt_10pct]
        if any(ordered[i] < ordered[i + 1] - 1e-9 for i in range(len(ordered) - 1)):
            raise ValueError(f"P(loss > T) must be non-increasing in T, got {ordered}.")


# ------------------------------------------------------------------------ #
# Fat-tail multiplier (the Done-when bar)
# ------------------------------------------------------------------------ #


def _fat_tail_multiplier(volatility_state: str) -> float:
    if volatility_state == "Spike":
        return FAT_TAIL_MULTIPLIER_SPIKE
    if volatility_state == "Extended":
        return FAT_TAIL_MULTIPLIER_EXTENDED
    return FAT_TAIL_MULTIPLIER_NONE  # "Compressed" | "Expanding"


# ------------------------------------------------------------------------ #
# P(loss > T)
# ------------------------------------------------------------------------ #


def _base_exceedance_prob(distance_pct: float, scale_pct: float) -> float:
    """P(adverse excursion >= distance_pct) under an exponential-tail model
    with scale `scale_pct` -- a standard exceedance-probability functional
    form (see module docstring), not literally fit to real loss data."""
    if distance_pct <= 0:
        return 1.0
    if scale_pct <= 0:
        return 0.0
    return math.exp(-distance_pct / scale_pct)


def _loss_probability(t_pct: float, stop_distance_pct: float, scale_pct: float, gap_risk_probability: float) -> float:
    """
    P(loss > t_pct), split at the stop-loss boundary (see module
    docstring). The `min(gap_risk_probability, prob_at_stop)` term is a
    monotonicity guard: without it, a stop distance large relative to
    `scale_pct` could make `prob_at_stop` smaller than `gap_risk_probability`
    itself, which would make P(loss > T) jump UP just past the stop
    boundary -- not a valid probability-of-exceedance curve. Capping the
    beyond-stop ceiling at `prob_at_stop` guarantees P(loss > T) is
    non-increasing in T everywhere.
    """
    prob_at_stop = _base_exceedance_prob(stop_distance_pct, scale_pct)
    if t_pct <= stop_distance_pct:
        return _base_exceedance_prob(t_pct, scale_pct)
    beyond = t_pct - stop_distance_pct
    ceiling = min(gap_risk_probability, prob_at_stop)
    return ceiling * _base_exceedance_prob(beyond, scale_pct)


# ------------------------------------------------------------------------ #
# Gap risk
# ------------------------------------------------------------------------ #


def _classify_session(
    ts: pd.Timestamp,
    overnight_start_hour_utc: int,
    overnight_end_hour_utc: int,
) -> str:
    """Crypto trades 24/7 -- 'weekend'/'overnight' here mean thinned
    liquidity, not a closed exchange (see module docstring)."""
    ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
    if ts_utc.dayofweek >= 5:  # 5=Saturday, 6=Sunday
        return "weekend"
    if overnight_start_hour_utc <= ts_utc.hour < overnight_end_hour_utc:
        return "overnight"
    return "weekday_intraday"


def _gap_risk_probability(
    ts: pd.Timestamp,
    *,
    base: float = DEFAULT_GAP_RISK_BASE,
    overnight_multiplier: float = DEFAULT_GAP_RISK_OVERNIGHT_MULTIPLIER,
    weekend_multiplier: float = DEFAULT_GAP_RISK_WEEKEND_MULTIPLIER,
    overnight_start_hour_utc: int = DEFAULT_OVERNIGHT_START_HOUR_UTC,
    overnight_end_hour_utc: int = DEFAULT_OVERNIGHT_END_HOUR_UTC,
) -> float:
    session = _classify_session(ts, overnight_start_hour_utc, overnight_end_hour_utc)
    if session == "weekend":
        return _clip01(base * weekend_multiplier)
    if session == "overnight":
        return _clip01(base * overnight_multiplier)
    return _clip01(base)


# ------------------------------------------------------------------------ #
# Liquidity trap risk
# ------------------------------------------------------------------------ #


def _liquidity_trap_contribution(
    trap_probability: float,
    position_size_pct: float,
    full_weight_size_pct: float = DEFAULT_LIQUIDITY_TRAP_FULL_WEIGHT_SIZE_PCT,
) -> float:
    """"More relevant at larger position sizes; less relevant for BTC/ETH
    majors at retail sizing" (Adaptive doc) -- linear ramp reaching full
    weight at `full_weight_size_pct` (a judgment call)."""
    size_scale = _clip01(position_size_pct / full_weight_size_pct) if full_weight_size_pct > 0 else 1.0
    return _clip01(trap_probability * size_scale)


# ------------------------------------------------------------------------ #
# Portfolio correlation risk
# ------------------------------------------------------------------------ #


def _correlation_flag(
    exposure_pct: float,
    cap_pct: float = CORRELATED_DIRECTION_CAP_PCT,
    approach_fraction: float = DEFAULT_CORRELATION_APPROACH_FRACTION,
) -> bool:
    # A small epsilon tolerance on the boundary comparison, not just style:
    # a caller summing position sizes in floating point (e.g. 3.0 + 1.8) who
    # lands on the conceptual "80% of 6% = 4.8" boundary could otherwise
    # fall on either side of it purely from float representation
    # (6.0 * 0.8 == 4.800000000000001 in IEEE 754, not the literal 4.8) --
    # the flag must trigger for a caller's every reasonable expectation of
    # "at the threshold", not silently miss it by 1e-15.
    return exposure_pct >= cap_pct * approach_fraction - 1e-9


# ------------------------------------------------------------------------ #
# Composite risk score
# ------------------------------------------------------------------------ #


def _compute_risk_score(
    p_loss_2pct: float,
    gap_risk_probability: float,
    liquidity_trap_contribution: float,
    correlation_exposure_pct: float,
    gap_risk_score_reference: float = DEFAULT_GAP_RISK_SCORE_REFERENCE,
    correlated_direction_cap_pct: float = CORRELATED_DIRECTION_CAP_PCT,
) -> float:
    """
    Documented, judgment-call composite (no spec formula given) -- an
    explicit, auditable weighted sum of four independently-bounded
    sub-scores summing to a 100-point maximum:
      - up to 60 points: P(loss > 2%) (the "headline" threshold, matching
        MASTER_CONTEXT's own 2% single-trade risk cap), scaled linearly.
      - up to 15 points: liquidity trap risk contribution.
      - up to 15 points: gap risk probability, scaled against
        `gap_risk_score_reference` (a gap risk level treated as "maximal"
        for scoring purposes; clipped beyond it).
      - up to 10 points: portfolio correlation exposure's fraction of the
        exact 6% cap, clipped at the cap (100% of the cap -> full 10
        points, not beyond).
    """
    headline = 60.0 * _clip01(p_loss_2pct)
    liquidity = 15.0 * _clip01(liquidity_trap_contribution)
    gap = 15.0 * (
        _clip01(gap_risk_probability / gap_risk_score_reference) if gap_risk_score_reference > 0 else 0.0
    )
    correlation = 10.0 * (
        _clip01(correlation_exposure_pct / correlated_direction_cap_pct) if correlated_direction_cap_pct > 0 else 0.0
    )
    return headline + liquidity + gap + correlation


# ------------------------------------------------------------------------ #
# Top-level entry point
# ------------------------------------------------------------------------ #


def assess_risk(inputs: RiskEngineInputs) -> RiskAssessment:
    """
    Assess one trade's downside risk at the exact four MASTER_CONTEXT
    drawdown levels (`LOSS_THRESHOLDS_PCT` -- 1%/2%/5%/10%, not
    configurable, matching how this project treats other exact spec
    numbers). See module docstring for the full design of every component
    computed here. The underlying per-threshold machinery
    (`_loss_probability` etc.) does accept an arbitrary threshold for
    direct unit testing; this entry point pins it to the spec's four.
    """
    if not isinstance(inputs, RiskEngineInputs):
        raise TypeError(f"inputs must be a RiskEngineInputs, got {type(inputs).__name__}.")

    multiplier = _fat_tail_multiplier(inputs.volatility_state.state)
    tail_risk_flag = multiplier != FAT_TAIL_MULTIPLIER_NONE

    gap_risk = _gap_risk_probability(inputs.entry_timestamp)

    def _multiplied_prob(t_pct: float) -> float:
        raw = _loss_probability(t_pct, inputs.stop_distance_pct, inputs.magnitude_p10_pct, gap_risk)
        # Fat-tail multiplier applied HERE, before anything downstream (the
        # risk score) is derived -- one source of truth, see module docstring.
        return _clip01(raw * multiplier)

    p_loss_1pct, p_loss_2pct, p_loss_5pct, p_loss_10pct = (_multiplied_prob(t) for t in LOSS_THRESHOLDS_PCT)

    liquidity_contribution = _liquidity_trap_contribution(
        inputs.liquidity_trap.trap_probability, inputs.position_size_pct
    )
    correlation_exposure = inputs.position_size_pct + inputs.existing_correlated_direction_exposure_pct
    correlation_flag = _correlation_flag(correlation_exposure)

    score = _compute_risk_score(p_loss_2pct, gap_risk, liquidity_contribution, correlation_exposure)

    reasons = [
        f"Volatility regime: {inputs.volatility_state.state} -> fat-tail multiplier {multiplier:.1f}x.",
        f"Gap risk probability: {gap_risk:.3f} (session: entry at {inputs.entry_timestamp}).",
        (
            f"Liquidity trap contribution: {liquidity_contribution:.3f} "
            f"(trap_probability={inputs.liquidity_trap.trap_probability:.2f}, "
            f"position_size={inputs.position_size_pct:.2f}%)."
        ),
        (
            f"Portfolio correlated-direction exposure: {correlation_exposure:.2f}% "
            f"(cap {CORRELATED_DIRECTION_CAP_PCT:.1f}%) -> "
            f"{'FLAGGED, approaching cap' if correlation_flag else 'within normal range'}."
        ),
        f"Composite risk score: {score:.1f}/100 (gate <= {RISK_SCORE_GATE_MAX:.0f}).",
    ]

    return RiskAssessment(
        risk_score=score,
        p_loss_gt_1pct=p_loss_1pct,
        p_loss_gt_2pct=p_loss_2pct,
        p_loss_gt_5pct=p_loss_5pct,
        p_loss_gt_10pct=p_loss_10pct,
        gap_risk_probability=gap_risk,
        liquidity_trap_risk_contribution=liquidity_contribution,
        portfolio_correlation_exposure_pct=correlation_exposure,
        portfolio_correlation_flag=correlation_flag,
        tail_risk_flag=tail_risk_flag,
        fat_tail_multiplier_applied=multiplier,
        passes_risk_gate=(score <= RISK_SCORE_GATE_MAX),
        reasons=reasons,
    )
