"""
Ensemble Meta-Model (Layer 7, Component C).

Architecture, taken directly from the Adaptive doc's Layer 7 Component C
section and Problem 4's solution text (not paraphrased): "strictly a linear
weighted average with learned weights. No non-linear stacking. No secondary
gradient-boosted model at this layer." This module has exactly ONE set of
learned parameters -- the five top-level signal weights -- and combines
already-independently-computed signal outputs with nothing more complex
than `sum(w_i * p_i)`. Every per-signal "how do I turn this result object
into a probability" step below is a FIXED, deterministic, documented
formula, never fit/optimized itself -- if it were tunable, this module
would quietly become a second model wearing an ensemble's clothes, exactly
the failure mode Problem 4 exists to rule out.

Inputs -- exactly five, non-overlapping, per the Adaptive doc's Layer 7C
"Inputs" list:
    1. Component A output: analog probability and sample size N
       (`analog_engine.AnalogForecast`).
    2. Component B output: model direction probability, magnitude
       quantiles, duration quantiles (`ComponentBSignal`, defined here --
       see its own docstring for why this bundle doesn't already exist
       upstream).
    3. Macro regime score AND dynamic weight, Layer 3
       (`macro_regime.MacroRegimeResult`).
    4. Multi-timeframe alignment score, Layer 4
       (`mtf_alignment.MTFAlignmentResult`).
    5. Liquidity trap probability, Layer 5
       (`liquidity_trap.LiquidityTrapResult`).

"What the ensemble explicitly does NOT take as input: raw structure
scores, raw momentum scores, or any feature-level input already present in
the Layer 2 vector." `EnsembleSignalSet` (below) is the single entry point
for supplying signals to `fit()`/`predict()`, and it is a frozen dataclass
with EXACTLY these five fields, each type-checked in `__post_init__`
against the specific upstream result class it must be an instance of. This
is what makes "feeding a raw Layer 2 feature into the ensemble constructor
raises an error rather than silently being accepted" (this module's
Done-when bar) true structurally, not just by convention: a bare float, a
pandas Series of feature columns, or an extra unexpected keyword argument
(e.g. `rsi_14=55.2`) cannot be accepted -- the first fails the isinstance
check, the second and third fail Python's own dataclass `__init__` (no
`**kwargs`, no untyped catch-all field exists to smuggle it through).

Why magnitude/duration are NOT part of the weighted average: the weighted
average only ever combines DIRECTIONAL information (a probability
distribution over down/neutral/up). Of the five inputs, only Component A
and Component B carry a directional read; Component B's magnitude and
duration quantiles have no second, independent source anywhere in this
five-input list to blend against -- macro regime, MTF alignment, and
liquidity trap are all directional-or-nothing signals, none of them
predict move size or holding period. "Ensembling" a single source against
nothing is a no-op, so magnitude/duration are passed straight through from
Component B into `EnsembleForecast`, unweighted, exactly matching the
Adaptive doc's own "Final output" list (which reports magnitude/duration
as direct quantile outputs, never described as blended).

How the three non-probabilistic directional signals become a distribution
(judgment calls, NOT MASTER_CONTEXT/Adaptive-doc formulas -- the source
docs establish that these three belong in the linear combination but give
no formula for turning a regime label, an alignment score, or a trap
probability into a down/neutral/up split):
  - Macro regime: Risk-On -> bullish lean, Risk-Off -> bearish lean, both
    scaled by `confidence * macro_weight` (both are genuine Layer 3
    outputs -- `macro_weight` is the "dynamic weight" the Adaptive doc's
    Input list names explicitly alongside "macro regime score", so both
    numbers are used, not just the state). Transitional and Macro-Event
    Window are BOTH treated as directionless (full neutral) -- neither
    state is a directional call: Transitional is literally "no clear
    directional consensus", and Macro-Event Window is a "no new entries"
    caution instruction, not a bullish/bearish read, even though its
    `confidence` field is always 1.0 (deterministic, calendar-driven) --
    that 1.0 describes certainty about the STATE, not about which way
    price will move, so it must not be read as directional conviction.
  - MTF alignment: bullish/bearish `dominant_direction` -> a lean scaled
    by `confidence`; "mixed" (a genuine tie, not an absence of a reading)
    and "neutral" (all four timeframes non-committal) both -> full
    neutral.
  - Liquidity trap: directional ONLY when `sweep_just_confirmed` is True,
    using `sweep_direction` scaled by `trap_probability` as conviction.
    When no sweep has just confirmed, `trap_probability` describes risk of
    a FUTURE sweep in either direction (Adaptive doc: liquidity trap
    "adjusts entry timing risk", never described as a directional
    confirmation the way MTF alignment is) -- genuinely no directional
    information exists yet, so it contributes full neutral regardless of
    how high `trap_probability` reads.
  - Component A: already a genuine empirical distribution
    (probability_down/neutral/up), used directly -- EXCEPT it is excluded
    entirely (not blended toward neutral) when `AnalogForecast.is_valid`
    is False, since "no forecast is issued from this component" in that
    case (analog_engine's own docstring) -- there is no estimate to blend,
    asserting a neutral one would misrepresent "insufficient data" as a
    positive belief. Row-level weight is renormalized across the
    remaining four signals for that row only (see `blend_signals`).
    When valid, "Forecasts with N close to the minimum threshold are
    weighted lower in the ensemble automatically" (Adaptive doc, no
    formula given) is implemented as a linear confidence ramp from 0 at
    `n_independent == min_independent_instances` to 1 at
    `n_independent == min_independent_instances * analog_confidence_ramp_multiple`
    (default 2x the gate) -- ALSO a fixed, non-learned formula.

Weight-learning process -- fully specified by the Adaptive doc, implemented
exactly: "Weights are optimised on the validation fold using a log-loss
minimisation objective. They are never optimised on the training fold...
and never evaluated on the test fold until after the weight-learning is
complete and frozen." `EnsembleMetaModel.fit()` / `fit_ensemble_weights()`
take a `validation_signals`/`validation_direction` argument pair -- named
that way deliberately, mirroring `HistoricalAnalogEngine.fit()`'s own
train-only discipline (see that module's docstring): there is no way for
this object to independently verify which walk-forward partition it was
handed, so, exactly like that precedent, the guarantee is enforced by
interface naming and caller discipline, not a runtime check. Weights are
constrained to a simplex (each in [0, 1], summing to 1) via
`scipy.optimize.minimize(method="SLSQP")` -- a genuine "weighted average"
requires non-negative weights summing to one; an unconstrained linear
regression over the five inputs would allow negative weights and would
functionally be closer to "stacking" than "averaging", which is exactly
what Problem 4 rules out.

Weight stability -- exact Adaptive-doc rule (Problem 4's solution text):
"If the optimal weight vector shifts materially across folds (defined as
any single weight changing by more than 0.15 between consecutive folds),
the ensemble is flagged as unstable and the default falls back to equal
weighting (each input weighted by 1/N) until the source of instability is
diagnosed." `WEIGHT_INSTABILITY_THRESHOLD = 0.15` is that exact number.
(Two illustrative example lines elsewhere in the Adaptive doc's sample
output templates -- "STABLE (all fold deltas < 0.10)", "max fold delta:
0.09" -- are flavor text for a hypothetical example output, not a second,
competing threshold; 0.15 is the only number Problem 4's actual rule
states, and is what this module enforces.) `EnsembleMetaModel.fit()` checks
the NEWEST fold-to-fold weight transition (this fold vs. the immediately
prior one -- NOT a re-scan of the entire cumulative fold history; see
`fit()`'s own docstring for a real bug this caused during review, where
re-scanning full history on every call made one old, already-flagged
instability permanently outrank any newer, genuinely stable transition)
and, once that check trips, keeps using equal weighting for every
subsequent `predict()` call regardless of how stable later folds' own
newly-learned weights look -- "until diagnosed" is an explicit, external,
human-driven decision (`reset_after_diagnosis()`), never an automatic
recovery just because the next fold happens to look fine in isolation. A
separate, purely diagnostic full-history view is available via
`EnsembleMetaModel.full_history_stability_report()` for a human doing that
diagnosis.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.macro_mtf_liquidity_vol.liquidity_trap import LiquidityTrapResult
from src.macro_mtf_liquidity_vol.macro_regime import MacroRegimeResult
from src.macro_mtf_liquidity_vol.mtf_alignment import MTFAlignmentResult
from src.predictive_core.analog_engine import AnalogForecast

# ------------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------------ #

# The five, and only five, approved ensemble inputs (Adaptive doc Layer 7C
# "Inputs" list). Fixed order used everywhere weights are represented as an
# array (optimizer) vs. a dict (everywhere else).
SIGNAL_NAMES: Tuple[str, ...] = (
    "component_a",
    "component_b",
    "macro_regime",
    "mtf_alignment",
    "liquidity_trap",
)

# Matches boosted_core.DIRECTION_CLASS_LABELS exactly (index 0/1/2) -- this
# module's local mapping to/from -1.0/0.0/1.0 direction values must stay in
# sync with that ordering; see `_direction_to_class_idx`.
DIRECTION_CLASSES: Tuple[str, ...] = ("down", "neutral", "up")

# Exact Adaptive-doc number (Problem 4 solution text) -- see module docstring.
WEIGHT_INSTABILITY_THRESHOLD = 0.15

# MASTER_CONTEXT constraint 9 / HistoricalAnalogEngine's own default -- exact
# spec number, matches PredictiveCoreStrategy's hardcoded 50 (see that
# module's fit()). NOT re-derived from AnalogForecast itself (it doesn't
# carry the gate value it was evaluated against) -- this MUST match whatever
# `min_independent_instances` the upstream HistoricalAnalogEngine was
# actually constructed with, same cross-module contract as
# PredictiveCoreStrategy.label_horizon.
ANALOG_MIN_INDEPENDENT_INSTANCES = 50

# Judgment call (NOT a spec number -- see module docstring): how many
# multiples of the minimum-N gate it takes for Component A's row-level
# confidence ramp to reach full trust.
DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE = 2.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ------------------------------------------------------------------------ #
# Component B per-row signal bundle
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class ComponentBSignal:
    """
    Component B's per-row output bundle: calibrated 3-class direction
    probability plus magnitude/duration quantiles.

    This does NOT already exist upstream as a single object --
    `boosted_core.py`'s `DirectionFoldResult`/`QuantileFoldResult` are
    fold-level TRAINING summaries (best_params, val/test logloss,
    calibration reports, ...), not per-row prediction bundles; the actual
    per-row values in `predictive_core_strategy.py` today are raw numpy
    array elements (`p_up[i]`, `mag_p50_frac[i]`) with no wrapping type.
    This is a new, minimal, ensemble-scoped structure bundling exactly the
    per-row fields the Adaptive doc's Component B output line calls for
    ("model direction probability, magnitude quantiles, duration
    quantiles") -- deliberately narrow (not a copy of the whole training
    result) so it can't smuggle in anything beyond that.
    """

    p_down: float
    p_neutral: float
    p_up: float
    magnitude_p10: float
    magnitude_p50: float
    magnitude_p90: float
    duration_p10: float
    duration_p50: float
    duration_p90: float

    def __post_init__(self) -> None:
        for name in ("p_down", "p_neutral", "p_up"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"ComponentBSignal.{name} must be in [0, 1], got {v}.")
        total = self.p_down + self.p_neutral + self.p_up
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"ComponentBSignal.p_down + p_neutral + p_up must sum to 1.0, got {total}."
            )
        for name in (
            "magnitude_p10", "magnitude_p50", "magnitude_p90",
            "duration_p10", "duration_p50", "duration_p90",
        ):
            v = getattr(self, name)
            if not np.isfinite(v):
                raise ValueError(f"ComponentBSignal.{name} must be finite, got {v}.")


# ------------------------------------------------------------------------ #
# The single, structurally-narrow entry point for supplying signals
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class EnsembleSignalSet:
    """
    Bundles exactly the five approved ensemble inputs for ONE point in
    time. This is the ONLY way to hand signals to `EnsembleMetaModel` --
    see module docstring for why its narrow, nominally-typed shape (rather
    than a dict/Series/**kwargs) is what makes rejecting a raw Layer 2
    feature structural rather than a matter of convention.
    """

    component_a: AnalogForecast
    component_b: ComponentBSignal
    macro_regime: MacroRegimeResult
    mtf_alignment: MTFAlignmentResult
    liquidity_trap: LiquidityTrapResult

    def __post_init__(self) -> None:
        expected_types = {
            "component_a": AnalogForecast,
            "component_b": ComponentBSignal,
            "macro_regime": MacroRegimeResult,
            "mtf_alignment": MTFAlignmentResult,
            "liquidity_trap": LiquidityTrapResult,
        }
        for field_name, expected_type in expected_types.items():
            value = getattr(self, field_name)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"EnsembleSignalSet.{field_name} must be a {expected_type.__name__} "
                    f"instance, got {type(value).__name__}. This guard exists specifically "
                    f"to keep raw Layer 2 features (or any other ad hoc value) from sneaking "
                    f"into the ensemble alongside the five approved, already-independent "
                    f"signals -- see module docstring."
                )


# ------------------------------------------------------------------------ #
# Per-signal: result object -> [p_down, p_neutral, p_up]
# ------------------------------------------------------------------------ #


def _lean_distribution(direction: Optional[str], strength: float) -> np.ndarray:
    """
    Fixed, deterministic formula (NOT learned/fit) turning a directional
    lean + conviction into a [p_down, p_neutral, p_up] distribution by
    blending against full neutral. `direction` is "bullish"/"bearish"/None
    (None == no directional information -> full neutral regardless of
    `strength`).
    """
    strength = _clip01(strength)
    neutral = np.array([0.0, 1.0, 0.0])
    if direction == "bullish":
        lean = np.array([0.0, 0.0, 1.0])
    elif direction == "bearish":
        lean = np.array([1.0, 0.0, 0.0])
    else:
        lean = neutral
    return strength * lean + (1.0 - strength) * neutral


def _component_a_distribution(
    forecast: AnalogForecast,
    min_independent_instances: int,
    ramp_multiple: float,
) -> Optional[np.ndarray]:
    """None means "excluded" (is_valid=False -- no forecast exists to
    blend, see module docstring). Otherwise the raw empirical distribution,
    scaled toward neutral by the N-closeness-to-gate confidence ramp."""
    if not forecast.is_valid:
        return None
    span = min_independent_instances * (ramp_multiple - 1.0)
    confidence_ramp = 1.0 if span <= 0 else _clip01(
        (forecast.n_independent - min_independent_instances) / span
    )
    base = np.array([forecast.probability_down, forecast.probability_neutral, forecast.probability_up])
    neutral = np.array([0.0, 1.0, 0.0])
    return confidence_ramp * base + (1.0 - confidence_ramp) * neutral


def _component_b_distribution(signal: ComponentBSignal) -> np.ndarray:
    """Already a genuine calibrated distribution -- used directly, no
    lean/strength conversion needed (unlike macro/MTF/liquidity trap)."""
    return np.array([signal.p_down, signal.p_neutral, signal.p_up])


def _macro_regime_distribution(result: MacroRegimeResult) -> np.ndarray:
    if result.state == "Risk-On":
        direction, state_confidence = "bullish", result.confidence
    elif result.state == "Risk-Off":
        direction, state_confidence = "bearish", result.confidence
    else:  # Transitional, Macro-Event Window -- neither is a directional call
        direction, state_confidence = None, 0.0
    # Both the regime score (state_confidence) AND the dynamic weight
    # (macro_weight) are named Adaptive-doc inputs -- both used, multiplied
    # together into one effective conviction (still a fixed formula, no
    # fitting).
    effective_confidence = state_confidence * result.macro_weight
    return _lean_distribution(direction, effective_confidence)


def _mtf_alignment_distribution(result: MTFAlignmentResult) -> np.ndarray:
    if result.dominant_direction == "bullish":
        return _lean_distribution("bullish", result.confidence)
    if result.dominant_direction == "bearish":
        return _lean_distribution("bearish", result.confidence)
    return _lean_distribution(None, 0.0)  # "mixed" or "neutral"


def _liquidity_trap_distribution(result: LiquidityTrapResult) -> np.ndarray:
    if not result.sweep_just_confirmed:
        return _lean_distribution(None, 0.0)
    return _lean_distribution(result.sweep_direction, result.trap_probability)


def _row_distributions(
    signals: EnsembleSignalSet,
    min_independent_instances: int,
    analog_confidence_ramp_multiple: float,
) -> Dict[str, Optional[np.ndarray]]:
    return {
        "component_a": _component_a_distribution(
            signals.component_a, min_independent_instances, analog_confidence_ramp_multiple
        ),
        "component_b": _component_b_distribution(signals.component_b),
        "macro_regime": _macro_regime_distribution(signals.macro_regime),
        "mtf_alignment": _mtf_alignment_distribution(signals.mtf_alignment),
        "liquidity_trap": _liquidity_trap_distribution(signals.liquidity_trap),
    }


# ------------------------------------------------------------------------ #
# The linear weighted average itself
# ------------------------------------------------------------------------ #


def blend_signals_detailed(
    signals: EnsembleSignalSet,
    weights: Dict[str, float],
    *,
    min_independent_instances: int = ANALOG_MIN_INDEPENDENT_INSTANCES,
    analog_confidence_ramp_multiple: float = DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE,
) -> Tuple[np.ndarray, Dict[str, Tuple[float, np.ndarray]]]:
    """
    The ensemble's one and only combination step: `sum(w_i * p_i)` over
    whichever signals are present for this row -- plus the per-signal
    terms of that sum, exposed as a second return value. Component A is
    the only signal that can be absent (`is_valid=False`); when absent,
    its weight is NOT silently dropped -- the remaining four signals'
    weights are renormalized to sum to 1 for this row only, so a row
    without a valid analog forecast is not systematically pulled toward
    neutral just because 20% of the nominal weight mass had nowhere to go.

    Returns
    -------
    (blended, per_signal) where `blended` is the final [down, neutral, up]
    distribution (identical to the old `blend_signals`'s only return value),
    and `per_signal` maps each PRESENT signal's name to
    `(normalized_row_weight, its_own_distribution)` -- the exact two
    numbers whose product is that signal's additive term in `blended`
    (before the final float-drift renormalization). Exists so a caller
    (e.g. `predictive_core.explainability`'s Layer 8 contribution
    breakdown) can attribute the final blend back to individual signals
    using the SAME arithmetic this function already computed, rather than
    re-deriving the per-signal distributions and row-level renormalization
    a second time in a different module -- two independent implementations
    of the same non-trivial logic (macro state->lean, MTF mixed->neutral,
    liquidity trap sweep-gating, Component A's N-ramp and exclusion) would
    be exactly the kind of duplication this project's cross-module audits
    have repeatedly caught drifting out of sync elsewhere.
    """
    if not isinstance(signals, EnsembleSignalSet):
        # `EnsembleMetaModel.predict()`/`fit_ensemble_weights()` both
        # isinstance-check before ever reaching here, so this is normally
        # unreachable -- but this function is also exported and callable
        # directly (used that way throughout this module's own tests), and
        # without this guard a raw dict/Series would fail on attribute
        # access with a confusing AttributeError deep in `_row_distributions`
        # instead of a clear, immediate TypeError naming the actual problem.
        raise TypeError(f"signals must be an EnsembleSignalSet, got {type(signals).__name__}.")
    distributions = _row_distributions(signals, min_independent_instances, analog_confidence_ramp_multiple)
    present = {name: dist for name, dist in distributions.items() if dist is not None}
    # Unreachable in practice: only component_a can be None, and the other
    # four always produce a (possibly fully-neutral) distribution.
    if not present:
        raise RuntimeError("blend_signals_detailed: no signals available to blend.")

    raw_weight_total = sum(weights[name] for name in present)
    if raw_weight_total <= 0:
        normalized = {name: 1.0 / len(present) for name in present}
    else:
        normalized = {name: weights[name] / raw_weight_total for name in present}

    blended = np.zeros(3)
    per_signal: Dict[str, Tuple[float, np.ndarray]] = {}
    for name, dist in present.items():
        blended += normalized[name] * dist
        per_signal[name] = (normalized[name], dist)
    total = blended.sum()
    if total > 0:
        blended = blended / total  # guard against float drift so it sums to exactly 1
    return blended, per_signal


def blend_signals(
    signals: EnsembleSignalSet,
    weights: Dict[str, float],
    *,
    min_independent_instances: int = ANALOG_MIN_INDEPENDENT_INSTANCES,
    analog_confidence_ramp_multiple: float = DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE,
) -> np.ndarray:
    """Thin wrapper around `blend_signals_detailed` for callers that only
    need the final blended distribution -- see that function for the
    per-signal breakdown and the full docstring of what this computes."""
    blended, _ = blend_signals_detailed(
        signals, weights,
        min_independent_instances=min_independent_instances,
        analog_confidence_ramp_multiple=analog_confidence_ramp_multiple,
    )
    return blended


def _direction_to_class_idx(direction: np.ndarray) -> np.ndarray:
    """-1.0 (down) -> 0, 0.0 (neutral) -> 1, 1.0 (up) -> 2. Same mapping as
    boosted_core.DIRECTION_CLASS_LABELS / boosted_core._direction_to_class_idx
    (kept as a local copy since that helper is private to boosted_core.py --
    the mapping is pinned by DIRECTION_CLASSES above, not reinvented
    independently)."""
    direction = np.asarray(direction, dtype=float)
    if not np.all(np.isin(direction, [-1.0, 0.0, 1.0])):
        raise ValueError("direction values must be only -1.0, 0.0, or 1.0.")
    return (direction + 1).astype(int)


def equal_weights() -> Dict[str, float]:
    """The stability-fallback weighting: each of the five inputs at 1/N."""
    n = len(SIGNAL_NAMES)
    return {name: 1.0 / n for name in SIGNAL_NAMES}


# ------------------------------------------------------------------------ #
# Weight learning (validation fold only, log-loss minimization)
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class WeightFitResult:
    weights: Dict[str, float]
    val_logloss: float
    n_validation_rows: int


def _mean_log_loss_for_weights(
    weights_array: np.ndarray,
    signal_sets: Sequence[EnsembleSignalSet],
    y_true_idx: np.ndarray,
    min_independent_instances: int,
    analog_confidence_ramp_multiple: float,
) -> float:
    weights = dict(zip(SIGNAL_NAMES, weights_array))
    preds = np.array([
        blend_signals(
            s, weights,
            min_independent_instances=min_independent_instances,
            analog_confidence_ramp_multiple=analog_confidence_ramp_multiple,
        )
        for s in signal_sets
    ])
    return float(log_loss(y_true_idx, preds, labels=[0, 1, 2]))


def fit_ensemble_weights(
    validation_signals: Sequence[EnsembleSignalSet],
    validation_direction: Sequence[float],
    *,
    min_independent_instances: int = ANALOG_MIN_INDEPENDENT_INSTANCES,
    analog_confidence_ramp_multiple: float = DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE,
) -> WeightFitResult:
    """
    Learn the five ensemble weights by minimizing mean multiclass log-loss
    over `validation_signals`/`validation_direction`. MUST be called with
    the validation-fold partition only -- see module docstring for why this
    is a naming/discipline contract, not a runtime-checkable one (the same
    situation `HistoricalAnalogEngine.fit()` documents for its own
    training-only requirement).

    Weights are constrained to the simplex (each in [0, 1], summing to 1)
    -- a genuine weighted average, not an unconstrained regression that
    could assign negative weights (see module docstring on why that would
    functionally be stacking, not averaging).
    """
    if len(validation_signals) == 0:
        raise ValueError("validation_signals must be non-empty.")
    if len(validation_signals) != len(validation_direction):
        raise ValueError(
            f"validation_signals ({len(validation_signals)}) and validation_direction "
            f"({len(validation_direction)}) must be the same length."
        )
    for i, s in enumerate(validation_signals):
        if not isinstance(s, EnsembleSignalSet):
            raise TypeError(
                f"validation_signals[{i}] must be an EnsembleSignalSet, got {type(s).__name__}."
            )

    y_idx = _direction_to_class_idx(np.asarray(validation_direction, dtype=float))
    n = len(SIGNAL_NAMES)
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        _mean_log_loss_for_weights,
        x0,
        args=(validation_signals, y_idx, min_independent_instances, analog_confidence_ramp_multiple),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Ensemble weight optimization failed to converge: {result.message}")

    w = np.clip(result.x, 0.0, None)
    w_sum = w.sum()
    w = w / w_sum if w_sum > 0 else np.full(n, 1.0 / n)
    weights = dict(zip(SIGNAL_NAMES, (float(x) for x in w)))
    return WeightFitResult(
        weights=weights,
        val_logloss=float(result.fun),
        n_validation_rows=len(validation_signals),
    )


# ------------------------------------------------------------------------ #
# Weight stability across walk-forward folds
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class WeightStabilityReport:
    status: str  # "STABLE" | "UNSTABLE" | "NOT_EVALUATED"
    max_abs_delta: Optional[float]  # None iff NOT_EVALUATED (fewer than 2 folds seen)
    triggering_fold_pair: Optional[Tuple[int, int]]  # (fold_i, fold_i+1) indices into the history
    triggering_signal: Optional[str]
    per_pair_max_delta: List[float]

    def __post_init__(self) -> None:
        if self.status not in ("STABLE", "UNSTABLE", "NOT_EVALUATED"):
            raise ValueError(f"invalid status {self.status!r}")
        if self.status == "NOT_EVALUATED" and self.max_abs_delta is not None:
            raise ValueError("NOT_EVALUATED status must have max_abs_delta=None.")
        if self.status != "UNSTABLE" and (
            self.triggering_fold_pair is not None or self.triggering_signal is not None
        ):
            raise ValueError("triggering_fold_pair/triggering_signal must be None unless status=UNSTABLE.")


def check_weight_stability(fold_weight_history: Sequence[Dict[str, float]]) -> WeightStabilityReport:
    """
    Exact Adaptive-doc rule: unstable iff any single weight changes by more
    than `WEIGHT_INSTABILITY_THRESHOLD` (0.15) between ANY two consecutive
    folds in the history. Needs at least 2 folds to evaluate anything.
    """
    if len(fold_weight_history) < 2:
        return WeightStabilityReport(
            status="NOT_EVALUATED", max_abs_delta=None,
            triggering_fold_pair=None, triggering_signal=None, per_pair_max_delta=[],
        )

    per_pair_max_delta: List[float] = []
    worst_delta = -1.0
    worst_pair: Optional[Tuple[int, int]] = None
    worst_signal: Optional[str] = None
    for i in range(1, len(fold_weight_history)):
        prev, curr = fold_weight_history[i - 1], fold_weight_history[i]
        pair_max = 0.0
        for name in SIGNAL_NAMES:
            delta = abs(curr[name] - prev[name])
            pair_max = max(pair_max, delta)
            if delta > worst_delta:
                worst_delta, worst_pair, worst_signal = delta, (i - 1, i), name
        per_pair_max_delta.append(pair_max)

    if worst_delta > WEIGHT_INSTABILITY_THRESHOLD:
        return WeightStabilityReport(
            status="UNSTABLE", max_abs_delta=worst_delta,
            triggering_fold_pair=worst_pair, triggering_signal=worst_signal,
            per_pair_max_delta=per_pair_max_delta,
        )
    return WeightStabilityReport(
        status="STABLE", max_abs_delta=worst_delta,
        triggering_fold_pair=None, triggering_signal=None,
        per_pair_max_delta=per_pair_max_delta,
    )


# ------------------------------------------------------------------------ #
# Final output
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class EnsembleForecast:
    """
    Layer 7's final combined output for one point in time. Deliberately
    does NOT include a 95% confidence interval on `direction_probabilities`
    despite the Adaptive doc's "Final output per asset" list mentioning one
    ("Direction probability... with 95% confidence interval") -- no source
    doc specifies a methodology for deriving a CI from a linear blend of
    five heterogeneous point estimates, and fabricating one here would be
    exactly the kind of unstated, invented rule this project's modules
    otherwise avoid (see e.g. macro_regime.py's US10Y non-use, or
    liquidity_trap.py's refusal to claim precise price levels). Rendering
    that CI is Layer 8 Explainability Engine's job (explicitly deferred
    past Phase 1 -- see the Adaptive doc's own Layer 8 section: "Phase 2
    addition... there is nothing to explain yet"); this module surfaces
    the raw material (`analog_n_independent`, `analog_distance_threshold`)
    Layer 8 would need for a sample-size-based interval when it is built.
    """

    direction_probabilities: Dict[str, float]  # "down"/"neutral"/"up" -> probability, sums to 1
    called_direction: str
    called_probability: float
    magnitude_p10: float
    magnitude_p50: float
    magnitude_p90: float
    duration_p10: float
    duration_p50: float
    duration_p90: float
    analog_n_independent: int
    analog_distance_threshold: float
    analog_is_valid: bool
    macro_weight_applied: float  # Layer 3's macro_weight, echoed -- "Macro conditioning weight applied"
    weights_used: Dict[str, float]
    used_fallback_equal_weighting: bool

    def __post_init__(self) -> None:
        if set(self.direction_probabilities) != set(DIRECTION_CLASSES):
            raise ValueError(
                f"direction_probabilities keys must be exactly {DIRECTION_CLASSES}, "
                f"got {sorted(self.direction_probabilities)}."
            )
        total = sum(self.direction_probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"direction_probabilities must sum to 1.0, got {total}.")
        if self.called_direction not in DIRECTION_CLASSES:
            raise ValueError(f"called_direction must be one of {DIRECTION_CLASSES}, got {self.called_direction!r}.")
        expected = self.direction_probabilities[self.called_direction]
        if not math.isclose(self.called_probability, expected, abs_tol=1e-9):
            raise ValueError(
                f"called_probability ({self.called_probability}) does not match "
                f"direction_probabilities[{self.called_direction!r}] ({expected})."
            )
        weight_total = sum(self.weights_used.values())
        if not math.isclose(weight_total, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights_used must sum to 1.0, got {weight_total}.")


# ------------------------------------------------------------------------ #
# Stateful wrapper: learns weights per fold, tracks stability, predicts
# ------------------------------------------------------------------------ #


class EnsembleMetaModel:
    """
    Fold-by-fold usage: call `fit()` once per walk-forward fold with that
    fold's VALIDATION partition (never train, never test), then `predict()`
    for each row to combine. `fit()` internally checks the NEWEST fold-to-
    fold weight transition (this fold's weights vs. the immediately prior
    fold's) and switches to equal weighting the moment that single
    transition moves any one weight by more than 0.15 -- and stays there
    (see `reset_after_diagnosis`) even if a later fold's freshly-learned
    weights would, on their own, look stable again. Deliberately does NOT
    re-derive this decision from the full cumulative fold history on every
    call (see `fit()`'s docstring for the bug that caused, and
    `full_history_stability_report()` for the separate, purely diagnostic,
    whole-history view).
    """

    def __init__(
        self,
        min_independent_instances: int = ANALOG_MIN_INDEPENDENT_INSTANCES,
        analog_confidence_ramp_multiple: float = DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE,
    ) -> None:
        self.min_independent_instances = min_independent_instances
        self.analog_confidence_ramp_multiple = analog_confidence_ramp_multiple

        self._learned_weights: Dict[str, float] = equal_weights()
        self._active_weights: Dict[str, float] = equal_weights()
        self._fold_weight_history: List[Dict[str, float]] = []
        self._is_using_fallback = False
        self._last_stability_report: Optional[WeightStabilityReport] = None
        self._fitted = False

    def fit(
        self,
        validation_signals: Sequence[EnsembleSignalSet],
        validation_direction: Sequence[float],
    ) -> WeightFitResult:
        """Learn this fold's weights (validation fold only -- see class and
        module docstrings), append to the cross-fold history, re-check
        stability, and update the weights `predict()` will use."""
        fit_result = fit_ensemble_weights(
            validation_signals, validation_direction,
            min_independent_instances=self.min_independent_instances,
            analog_confidence_ramp_multiple=self.analog_confidence_ramp_multiple,
        )
        self._learned_weights = fit_result.weights
        self._fold_weight_history.append(fit_result.weights)

        # Deliberately scores ONLY the newest fold-to-fold transition
        # (`[-2:]`), NOT the full cumulative history -- see
        # `full_history_stability_report` for why. Using the full history
        # here was a real bug found during review: `check_weight_stability`
        # reports UNSTABLE if ANY pair anywhere in the list it's given
        # exceeds the threshold, so re-scanning the entire history on every
        # `fit()` call meant one old, already-flagged instability would
        # permanently re-win every subsequent call -- making
        # `reset_after_diagnosis()` silently undone by the very next `fit()`
        # regardless of whether the new transition was itself stable
        # (confirmed by direct reproduction before this fix:
        # `triggering_fold_pair` stayed pinned at the original `(0, 1)` even
        # after a fold whose own delta was 0.0). Checking only the newest
        # pair makes `_is_using_fallback` a genuine STICKY flag driven
        # exclusively by NEW evidence (a newly-unstable transition) or an
        # explicit `reset_after_diagnosis()` call -- matching "flagged as
        # unstable... until the source of instability is diagnosed" (an
        # event that happens once, not a condition re-derived from scratch
        # on every fold).
        latest_transition = check_weight_stability(self._fold_weight_history[-2:])
        self._last_stability_report = latest_transition
        if latest_transition.status == "UNSTABLE":
            self._is_using_fallback = True

        self._active_weights = equal_weights() if self._is_using_fallback else self._learned_weights
        self._fitted = True
        return fit_result

    def full_history_stability_report(self) -> WeightStabilityReport:
        """
        Diagnostic-only view: scans every consecutive pair across the
        ENTIRE fold history seen so far, not just the newest transition
        `fit()` uses to decide whether to (keep) trigger(ing) the fallback
        (see `stability_report` / the `fit()` docstring). Exists so a human
        "diagnosing the source of instability" (the spec's own phrase for
        what must happen before `reset_after_diagnosis()` is called) can see
        the whole picture -- this method's result is informational and never
        itself changes `is_using_fallback` or `weights`.
        """
        return check_weight_stability(self._fold_weight_history)

    def reset_after_diagnosis(self) -> None:
        """Per spec: the equal-weight fallback persists 'until the source
        of instability is diagnosed' -- an explicit, external decision.
        Calling this resumes using the most recently learned weights."""
        self._is_using_fallback = False
        if self._fitted:
            self._active_weights = self._learned_weights

    def predict(self, signals: EnsembleSignalSet) -> EnsembleForecast:
        self._require_fitted()
        if not isinstance(signals, EnsembleSignalSet):
            raise TypeError(f"signals must be an EnsembleSignalSet, got {type(signals).__name__}.")

        blended = blend_signals(
            signals, self._active_weights,
            min_independent_instances=self.min_independent_instances,
            analog_confidence_ramp_multiple=self.analog_confidence_ramp_multiple,
        )
        direction_probabilities = dict(zip(DIRECTION_CLASSES, (float(x) for x in blended)))
        called_direction = DIRECTION_CLASSES[int(np.argmax(blended))]
        b = signals.component_b

        return EnsembleForecast(
            direction_probabilities=direction_probabilities,
            called_direction=called_direction,
            called_probability=direction_probabilities[called_direction],
            magnitude_p10=b.magnitude_p10, magnitude_p50=b.magnitude_p50, magnitude_p90=b.magnitude_p90,
            duration_p10=b.duration_p10, duration_p50=b.duration_p50, duration_p90=b.duration_p90,
            analog_n_independent=signals.component_a.n_independent,
            analog_distance_threshold=signals.component_a.distance_threshold,
            analog_is_valid=signals.component_a.is_valid,
            macro_weight_applied=signals.macro_regime.macro_weight,
            weights_used=dict(self._active_weights),
            used_fallback_equal_weighting=self._is_using_fallback,
        )

    @property
    def weights(self) -> Dict[str, float]:
        """The weights `predict()` is currently using (learned or fallback)."""
        return dict(self._active_weights)

    @property
    def learned_weights(self) -> Dict[str, float]:
        """The most recent fold's actually-learned weights, even while
        `is_using_fallback` is True (retained for diagnosis)."""
        return dict(self._learned_weights)

    @property
    def is_using_fallback(self) -> bool:
        return self._is_using_fallback

    @property
    def fold_weight_history(self) -> List[Dict[str, float]]:
        return list(self._fold_weight_history)

    @property
    def stability_report(self) -> Optional[WeightStabilityReport]:
        """The NEWEST fold-to-fold transition's stability report -- what
        `fit()` actually used to decide whether to (keep) trigger(ing) the
        fallback. For a full-history audit view, see
        `full_history_stability_report()`."""
        return self._last_stability_report

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("EnsembleMetaModel.fit() must be called before predict().")
