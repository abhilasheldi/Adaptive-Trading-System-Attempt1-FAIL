"""
Explainability Engine (Layer 8) -- PURE TEMPLATE / RULE-BASED.

Per the Adaptive doc's Layer 8 section: "generates human-readable
explanations mapping feature attribution weights to individual prediction
outputs... serves two functions: it builds justified trust in the system
by making the decision auditable, and it makes losses diagnosable -- when
a trade fails, the explanation identifies which inputs were misleading."
Prereq (Prompt Playbook 2F): the Ensemble Meta-Model (Layer 7C,
`predictive_core.ensemble`), already built.

=========================================================================
NO LLM, NO NETWORK, NO API KEY -- A HARD ARCHITECTURAL CONSTRAINT
=========================================================================
This module contains no LLM call of any kind, in any tier, on any path --
not in normal operation, not in a fallback, not behind a feature flag. It
performs no network I/O, requires no API key, and creates no billing
dependency. Its only imports are the standard library (`math`,
`dataclasses`, `typing`), `numpy`, and this project's own
`predictive_core.ensemble` -- none of which makes an external call.

An earlier version of this file called the Anthropic API to phrase each
contribution line, with a template path as a fallback. That entire path
(`AnthropicProseGenerator`, the prompt builder, the JSON response parser,
`ProseGenerationError`, the `enable_live_api` flag, and the `used_live_api`
output field) has been REMOVED, not merely disabled. The reasons this is
better here, not just cheaper:

  - DETERMINISM. A trading system's audit log must be reproducible: the
    same attribution input must render byte-for-byte identically forever,
    so that a post-mortem on a losing trade reads exactly what was shown
    at decision time. A sampled language model cannot guarantee that; a
    template can, and this module's test suite asserts it directly.
  - NO NUMERIC RISK SURFACE. The old design kept numbers away from the
    model as a defence-in-depth boundary. Removing the model removes the
    boundary's reason to exist -- there is now no path, however guarded,
    by which generated text can sit beside a number it did not come from.
  - NO RUNTIME DEPENDENCY. Explanation generation cannot fail, time out,
    rate-limit, or cost money at the moment a trade is being evaluated.

Prose is produced by mapping structured attribution facts onto
pre-written sentence structures (see `_PHRASE_BUILDERS` and
`_build_narrative`). Numbers are ALWAYS inserted directly from the
attribution data via format specifiers -- never generated, paraphrased,
rounded by a model, or re-derived.

=========================================================================
This module is a RENDERER, not a computer of new numbers
=========================================================================
Every quantity in its output (direction probability, EV, position size,
weight stability) is either read directly from an already-computed
upstream object (`ensemble.EnsembleForecast`,
`ensemble.WeightStabilityReport`) or derived by an exact, documented,
provable formula from one (the per-signal contribution breakdown, the 95%
CI) -- see `contributions_from_ensemble` and `direction_probability_ci`.
It never re-derives Component A/B, macro regime, MTF alignment, or
liquidity trap logic itself -- that would risk drifting out of sync with
`ensemble.py`'s own arithmetic, exactly the failure mode this project's
cross-module reviews have repeatedly caught elsewhere.

Three inputs this module explicitly does NOT compute, and accepts as
already-computed (`TradeRiskSummary`): EV estimate, EV lower CI bound, and
position size. These are Layer 9 (Risk Engine) and Layer 10 (Position
Sizing / quarter-Kelly) outputs per the Adaptive doc's own layer
boundaries. Rather than fabricate a placeholder Risk/Sizing computation
inside an Explainability module (whose job is to RENDER decisions, not
make them), the caller supplies these three numbers directly, exactly as
`mtf_alignment.py` accepts already-derived per-timeframe directional reads
instead of computing them from a raw Layer 2 feature vector itself.

Signal-contribution breakdown -- provable, not invented. Per signal i
(one of the ensemble's five inputs), define its contribution to the
called direction as:

    impact_pct_i = 100 * w_i * (p_i[called] - p_i[opposite])

where `w_i` is that signal's actual ROW-LEVEL weight (post row-level
renormalization -- see `ensemble.blend_signals_detailed`, which this
module reuses directly rather than re-deriving), `p_i` is signal i's own
[down, neutral, up] distribution, `called` is the ensemble's called class,
and `opposite` is the class most directly opposing it (down for up, up
for down; for a "neutral" call, the stronger of {down, up} stands in for
"opposite", since neutral has no single natural opposite). This is exact
and provable: summing `impact_pct_i` over ALL five signals equals exactly
`100 * (p_final[called] - p_final[opposite])`, the final blend's own net
lean -- verified directly in this module's test suite, not merely
asserted. A signal that actually disagrees with the ensemble's called
direction (despite being outweighted by the others) produces a genuinely
NEGATIVE `impact_pct` under this formula -- this is how the spec format's
"+/-" mix arises here, from a single uniform formula applied to all five
signals, not two different formulas for the positive and negative sides.

NOTE on the identity vs. the RENDERED lines: `contributions_from_ensemble`
drops contributions below `min_abs_pct_to_show` from the rendered
breakdown, so the DISPLAYED lines need not sum to the full net lean. Any
statement this module makes about overall conviction is therefore taken
from `EnsembleForecast`'s own probabilities (see `net_lean_pct`), never
from summing the filtered display lines.

The Adaptive doc's own example output additionally shows contribution
lines that are NOT among the ensemble's five formal inputs at all (e.g.
"Market structure (BOS + CHOCH confirmed, 4H)", "RSI bearish divergence
active (4H)" -- raw Layer 2 structural/momentum features the ensemble
deliberately never touches, per `ensemble.py`'s own scope boundary).
`contributions_from_ensemble` intentionally does NOT attempt to
reconstruct these: no source doc gives a formula for converting a raw
Layer 2 feature or flag into a percentage-point contribution (this
project's `feature_importance` is a fold-level TRAINING statistic, not a
per-prediction attribution, and treating it as one would be a real, if
subtle, methodological error). A caller that has such signals available
(e.g. from `volatility_classifier.VolatilityStateResult`, or a raw
struct/momentum flag) can construct additional `SignalContribution`
entries directly and pass them into `explain()` alongside the ensemble-
derived ones -- this module renders whatever structured contributions it
is given, from any source; it only computes the five formal ones itself
because those alone have an exact, provable formula behind them.

95% CI on the direction probability -- an approximation, clearly labeled
as one. The Adaptive doc's own example ties the CI directly to N ("95%
CI: 66%-81%, N=187 independent analogues"), so `direction_probability_ci`
uses a standard Wilson score interval (a well-established statistical
method -- not invented here) treating the ensemble's called probability
as if it were a proportion estimated from N independent Bernoulli trials,
with N = Component A's own `analog_n_independent`. This is a real
approximation, not a rigorous derivation: N is genuinely only Component
A's independent-analogue count, not a sample size behind the other four
blended signals. Returns None (rendered as "unavailable") when Component
A did not contribute a valid forecast -- fabricating a CI with no analog
sample behind it would be worse than omitting one.

Edge cases are handled EXPLICITLY rather than papered over -- this is the
main practical advantage of a rule-based renderer, since a generative
model would happily produce fluent text for any of them:

  - Near-zero attribution: when the ensemble's net lean is below
    `NEGLIGIBLE_NET_LEAN_PCT`, the narrative says so plainly instead of
    naming a "dominant" driver of an essentially undecided call.
  - Ties between top contributors: detected within `TIE_TOLERANCE_PCT`
    and reported AS a tie ("carry equal weight"), never resolved by
    arbitrarily naming whichever happened to sort first. Sorting is
    additionally made total (label as final tiebreak) so ordering is
    reproducible regardless of upstream dict order.
  - Missing/degraded data: `SignalContribution.is_degraded` marks a line
    built from incomplete inputs; degraded lines are called out in a
    dedicated "Data quality" section rather than silently rendered as if
    fully supported. An invalid Component A also degrades the CI, which
    renders as "unavailable" rather than as a fabricated interval.
  - Ensemble weight instability: surfaced in the spec's own stability
    line AND, when UNSTABLE, as an explicit leading caveat in the
    narrative -- because in that state the attribution percentages
    themselves are the thing under suspicion, so a reader must not take
    the breakdown at face value.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.predictive_core import ensemble
from src.predictive_core.ensemble import (
    EnsembleForecast,
    EnsembleSignalSet,
    WeightStabilityReport,
)

# ------------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------------ #

# Exact MASTER_CONTEXT numbers (constraint 7: "EV threshold uses the lower
# confidence-interval bound, not the point estimate. Minimum: EV lower CI
# bound > +0.15% per unit risk"; constraint 8's six-gate requirement
# repeats the same number). Strict ">", not ">=" -- "minimum" here means
# the boundary itself does not pass.
EV_LOWER_CI_GATE_PCT = 0.15

# Standard 95% two-sided normal critical value -- an exact statistical
# constant, not a judgment call.
CI_Z_SCORE_95 = 1.96

# Judgment calls (NOT MASTER_CONTEXT/Adaptive-doc numbers -- no source doc
# specifies display-filtering, tie, or negligibility thresholds for this
# layer). Named and importable so tests and callers can reference them
# rather than re-hardcoding, matching the convention used across
# `macro_mtf_liquidity_vol/`.
DEFAULT_MIN_ABS_PCT_TO_SHOW = 1.0  # contributions smaller than this are dropped from the breakdown
TIE_TOLERANCE_PCT = 0.5  # two contributions within this many points are reported as tied
NEGLIGIBLE_NET_LEAN_PCT = 2.0  # below this net lean, the call is described as essentially undecided

_DISPLAY_LABELS: Dict[str, str] = {
    "component_a": "Historical analog match",
    "component_b": "Direction classifier",
    "macro_regime": "Macro regime",
    "mtf_alignment": "Multi-timeframe alignment",
    "liquidity_trap": "Liquidity trap",
}

_SETUP_LABELS: Dict[str, str] = {"up": "LONG SETUP", "down": "SHORT SETUP", "neutral": "NO SETUP"}
_DIRECTIONAL_ADJECTIVE: Dict[str, str] = {"up": "Bullish", "down": "Bearish", "neutral": "Neutral"}

# Pre-written sentence structures, selected by macro regime state. Keys are
# `macro_regime.MacroRegimeResult.state`'s exact four values.
_MACRO_REGIME_SENTENCES: Dict[str, str] = {
    "Risk-On": "Macro backdrop is Risk-On, which permits full standard EV thresholds.",
    "Risk-Off": (
        "Macro backdrop is Risk-Off, which blocks new long entries in risk assets "
        "regardless of this setup's other merits."
    ),
    "Transitional": (
        "Macro backdrop is Transitional, so position sizes are halved and a more "
        "conservative EV threshold applies."
    ),
    "Macro-Event Window": (
        "A scheduled high-impact macro event is within 24 hours, which blocks new "
        "entries outright."
    ),
}


# ------------------------------------------------------------------------ #
# Structured inputs
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class SignalContribution:
    """One line item in the 'Signal contributions' breakdown: a label
    (matches an `ensemble.SIGNAL_NAMES` entry for ensemble-derived
    contributions, or any caller-chosen id for additional context lines),
    a plain deterministic detail string, and a signed percentage-point
    impact.

    `is_degraded` marks a contribution computed from incomplete or
    unavailable upstream data. It does not change the arithmetic -- it
    drives the "Data quality" section, so a reader can tell a well-founded
    line from one standing on a gap, instead of both rendering identically.
    """

    label: str
    detail: str
    impact_pct: float
    is_degraded: bool = False

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("SignalContribution.label must be non-empty.")
        if not self.detail.strip():
            raise ValueError("SignalContribution.detail must be non-empty.")
        if not np.isfinite(self.impact_pct):
            raise ValueError(
                f"SignalContribution.impact_pct must be finite, got {self.impact_pct}."
            )


@dataclass(frozen=True)
class TradeRiskSummary:
    """EV/position-size numbers this module does NOT compute -- Layer 9
    (Risk Engine) / Layer 10 (Position Sizing) outputs, supplied by the
    caller (see module docstring for why)."""

    ev_estimate_pct: float  # "EV estimate... per unit risk"
    ev_lower_ci_pct: float  # "EV lower CI bound"
    position_size_pct: float  # "% of capital"

    def __post_init__(self) -> None:
        for name in ("ev_estimate_pct", "ev_lower_ci_pct", "position_size_pct"):
            v = getattr(self, name)
            if not np.isfinite(v):
                raise ValueError(f"TradeRiskSummary.{name} must be finite, got {v}.")
        if self.position_size_pct < 0:
            raise ValueError(
                f"TradeRiskSummary.position_size_pct must be non-negative, "
                f"got {self.position_size_pct}."
            )


# ------------------------------------------------------------------------ #
# Signal-contribution breakdown (provable, see module docstring)
# ------------------------------------------------------------------------ #


def _detail_for_signal(name: str, signals: EnsembleSignalSet) -> Tuple[str, bool]:
    """Returns (detail_string, is_degraded) for one ensemble signal.

    Every branch returns a non-empty string for every reachable input
    state -- including the degraded ones -- so no rendered field can come
    out blank (Done-when bar 4).
    """
    if name == "component_a":
        a = signals.component_a
        if not a.is_valid:
            # Reachable and expected: Component A withholds a forecast below
            # the N>=50 independent-instances gate (MASTER_CONTEXT
            # constraint 9), which is correct conservative behaviour, not an
            # error -- but it IS degraded input for an explanation.
            return "no valid analog forecast (below minimum independent-sample gate)", True
        return f"N={a.n_independent}, distance threshold {a.distance_threshold:.3g}", False
    if name == "component_b":
        b = signals.component_b
        probs = {"down": b.p_down, "neutral": b.p_neutral, "up": b.p_up}
        called = max(probs, key=lambda k: (probs[k], k))  # total order -> deterministic on ties
        return f"called {called} at {probs[called]:.0%} confidence", False
    if name == "macro_regime":
        m = signals.macro_regime
        return f"{m.state}, confidence {m.confidence:.2f}", False
    if name == "mtf_alignment":
        t = signals.mtf_alignment
        return f"{t.score}/4 {t.dominant_direction}", False
    if name == "liquidity_trap":
        lq = signals.liquidity_trap
        if lq.sweep_just_confirmed:
            return (
                f"{lq.sweep_direction} sweep confirmed, probability "
                f"{lq.trap_probability:.0%}"
            ), False
        return f"no confirmed sweep, trap probability {lq.trap_probability:.0%}", False
    raise ValueError(f"no detail formatter for signal {name!r}.")  # unreachable for SIGNAL_NAMES


def _contribution_sort_key(c: "SignalContribution") -> Tuple[int, float, str]:
    """Positive contributions first (largest first), then negative (largest
    magnitude first) -- matching the Adaptive doc's example ordering.

    `label` is the final tiebreak so the ordering is TOTAL: two
    contributions with identical impact must not be able to swap places
    between runs based on upstream dict insertion order. That would break
    the byte-for-byte determinism guarantee this module makes.
    """
    return (0 if c.impact_pct >= 0 else 1, -abs(c.impact_pct), c.label)


def contributions_from_ensemble(
    forecast: EnsembleForecast,
    signals: EnsembleSignalSet,
    *,
    min_independent_instances: int = ensemble.ANALOG_MIN_INDEPENDENT_INSTANCES,
    analog_confidence_ramp_multiple: float = ensemble.DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE,
    min_abs_pct_to_show: float = DEFAULT_MIN_ABS_PCT_TO_SHOW,
) -> List[SignalContribution]:
    """
    The five ensemble-level contribution lines -- see module docstring for
    the exact formula and its provable sum-to-net-lean identity.

    Contributions with `|impact_pct| < min_abs_pct_to_show` are dropped
    (a judgment call, not a spec number -- it avoids a "+0%"/"-0%" line for
    a signal that barely moved the needle). Because of that filter the
    returned lines need not sum to the full net lean; see the module
    docstring's NOTE.
    """
    called_idx = ensemble.DIRECTION_CLASSES.index(forecast.called_direction)
    if forecast.called_direction == "up":
        opposite_idx: Optional[int] = ensemble.DIRECTION_CLASSES.index("down")
    elif forecast.called_direction == "down":
        opposite_idx = ensemble.DIRECTION_CLASSES.index("up")
    else:  # "neutral" has no single natural opposite -- see module docstring
        opposite_idx = None

    _, per_signal = ensemble.blend_signals_detailed(
        signals,
        forecast.weights_used,
        min_independent_instances=min_independent_instances,
        analog_confidence_ramp_multiple=analog_confidence_ramp_multiple,
    )

    down_idx = ensemble.DIRECTION_CLASSES.index("down")
    up_idx = ensemble.DIRECTION_CLASSES.index("up")
    contributions: List[SignalContribution] = []
    for name, (w, dist) in per_signal.items():
        if opposite_idx is not None:
            net_lean = dist[called_idx] - dist[opposite_idx]
        else:
            net_lean = dist[called_idx] - max(dist[down_idx], dist[up_idx])
        impact_pct = 100.0 * w * net_lean
        if abs(impact_pct) < min_abs_pct_to_show:
            continue
        detail, degraded = _detail_for_signal(name, signals)
        contributions.append(
            SignalContribution(
                label=name, detail=detail, impact_pct=impact_pct, is_degraded=degraded
            )
        )
    return sorted(contributions, key=_contribution_sort_key)


# ------------------------------------------------------------------------ #
# 95% CI on the direction probability (see module docstring)
# ------------------------------------------------------------------------ #


def _wilson_score_interval(p: float, n: int, z: float = CI_Z_SCORE_95) -> Tuple[float, float]:
    """Standard Wilson score interval for a binomial proportion -- a
    well-established statistical method, not invented for this module."""
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half_width), min(1.0, center + half_width)


def direction_probability_ci(forecast: EnsembleForecast) -> Optional[Tuple[float, float]]:
    """(lower, upper) in [0, 1], or None if Component A did not contribute
    a valid forecast (no N to anchor the approximation to -- see module
    docstring)."""
    if not forecast.analog_is_valid or forecast.analog_n_independent <= 0:
        return None
    return _wilson_score_interval(forecast.called_probability, forecast.analog_n_independent)


def net_lean_pct(forecast: EnsembleForecast) -> float:
    """The ensemble's own net conviction, in percentage points: the called
    class's probability minus its most direct opposite.

    Taken from `EnsembleForecast` rather than by summing the rendered
    contribution lines, because those are filtered for display (see the
    module docstring's NOTE) and would understate conviction whenever
    several small contributions were dropped.
    """
    probs = forecast.direction_probabilities
    called = forecast.called_direction
    if called == "up":
        return 100.0 * (probs["up"] - probs["down"])
    if called == "down":
        return 100.0 * (probs["down"] - probs["up"])
    return 100.0 * (probs["neutral"] - max(probs["down"], probs["up"]))


# ------------------------------------------------------------------------ #
# Rule-based prose: attribution pattern -> pre-written sentence structure
# ------------------------------------------------------------------------ #


def _display_label(label: str) -> str:
    return _DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def _phrase_for(contribution: SignalContribution) -> str:
    """The contribution line's phrase: `Display label (concrete detail)` --
    exactly the shape the Adaptive doc's example uses ("Macro regime
    (Risk-On, confidence 0.81)"). Pure string formatting; the detail is
    passed through verbatim from upstream, never paraphrased."""
    return f"{_display_label(contribution.label)} ({contribution.detail})"


def _top_contributors(
    contributions: Sequence[SignalContribution], tie_tolerance_pct: float
) -> List[SignalContribution]:
    """The joint-strongest contributions BY MAGNITUDE, as a list so a tie is
    representable rather than silently resolved. Returns [] for no input."""
    if not contributions:
        return []
    ranked = sorted(contributions, key=lambda c: (-abs(c.impact_pct), c.label))
    leader = abs(ranked[0].impact_pct)
    return [c for c in ranked if leader - abs(c.impact_pct) <= tie_tolerance_pct]


def _narrative_dominance_sentence(
    contributions: Sequence[SignalContribution],
    net_lean: float,
    negligible_net_lean_pct: float,
    tie_tolerance_pct: float,
) -> str:
    """Rule: attribution pattern -> pre-written sentence structure.

    Ordered most-specific-first, so the degenerate cases (no lines at all;
    an essentially undecided call) are never described in the language of a
    confident, dominated one.
    """
    if not contributions:
        return (
            "No individual signal cleared the reporting threshold: this call rests on "
            "the combination of several individually negligible contributions."
        )
    if abs(net_lean) < negligible_net_lean_pct:
        return (
            f"Overall conviction is negligible (net lean {net_lean:+.1f} points): the "
            f"signals substantially offset one another, so no factor should be read as "
            f"driving this call."
        )

    leaders = _top_contributors(contributions, tie_tolerance_pct)
    if len(leaders) > 1:
        names = ", ".join(_display_label(c.label) for c in leaders[:-1])
        names = f"{names} and {_display_label(leaders[-1].label)}"
        return (
            f"{names} carry effectively equal weight here "
            f"(each about {abs(leaders[0].impact_pct):.0f} points); no single factor "
            f"dominates."
        )

    top = leaders[0]
    if top.impact_pct >= 0:
        return (
            f"{_display_label(top.label)} is the dominant supporting factor at "
            f"{top.impact_pct:+.0f} points ({top.detail})."
        )
    return (
        f"The single strongest factor is {_display_label(top.label)} at "
        f"{top.impact_pct:+.0f} points, and it argues AGAINST this call "
        f"({top.detail}); the setup is carried by weaker, more numerous supporting "
        f"signals."
    )


def _narrative_ev_sentence(risk: TradeRiskSummary, gate_passes: bool) -> str:
    """Rule: EV pass/fail -> pre-written sentence structure."""
    if gate_passes:
        return (
            f"EV clears the gate: the lower CI bound of {risk.ev_lower_ci_pct:+.2f}% "
            f"exceeds the required +{EV_LOWER_CI_GATE_PCT:.2f}%, so the trade is "
            f"permitted on expected value."
        )
    return (
        f"EV FAILS the gate: the lower CI bound of {risk.ev_lower_ci_pct:+.2f}% does not "
        f"exceed the required +{EV_LOWER_CI_GATE_PCT:.2f}%, so this trade must not be "
        f"taken on expected value regardless of the signal picture above."
    )


def _narrative_stability_sentence(
    stability: Optional[WeightStabilityReport], used_fallback: bool
) -> Optional[str]:
    """Rule: ensemble weight instability -> a LEADING caveat.

    When weights are unstable the attribution percentages are themselves
    the quantity under suspicion, so this is stated before the breakdown is
    interpreted, not appended as a footnote.
    """
    if stability is not None and stability.status == "UNSTABLE":
        signal = stability.triggering_signal or "an unidentified signal"
        delta = (
            "unknown" if stability.max_abs_delta is None else f"{stability.max_abs_delta:.2f}"
        )
        return (
            f"WARNING -- ensemble weights are UNSTABLE (max fold delta {delta}, flagged on "
            f"{_display_label(signal)}). The contribution percentages below depend on those "
            f"weights and should not be relied on until the instability is diagnosed."
        )
    if used_fallback:
        return (
            "Note: the ensemble is using its equal-weight fallback, so contributions "
            "reflect equal weighting rather than fitted weights."
        )
    if stability is None or stability.status == "NOT_EVALUATED":
        return (
            "Note: ensemble weight stability has not been evaluated (fewer than two folds "
            "observed), so weight reliability is unknown."
        )
    return None


def _narrative_data_quality_sentence(
    contributions: Sequence[SignalContribution], ci_available: bool
) -> Optional[str]:
    degraded = sorted({c.label for c in contributions if c.is_degraded})
    parts: List[str] = []
    if degraded:
        names = ", ".join(_display_label(label) for label in degraded)
        parts.append(f"built on incomplete upstream data: {names}")
    if not ci_available:
        parts.append(
            "no analog sample was available, so the probability interval is unavailable "
            "rather than estimated"
        )
    if not parts:
        return None
    return "Data quality -- " + "; ".join(parts) + "."


def _build_narrative(
    contributions: Sequence[SignalContribution],
    forecast: EnsembleForecast,
    risk: TradeRiskSummary,
    gate_passes: bool,
    stability: Optional[WeightStabilityReport],
    macro_regime_state: Optional[str],
    ci_available: bool,
    *,
    negligible_net_lean_pct: float,
    tie_tolerance_pct: float,
) -> List[str]:
    """Assembles the rule-selected sentences, in a fixed order. Every
    sentence is chosen by an explicit branch above; none is generated."""
    sentences: List[str] = []

    caveat = _narrative_stability_sentence(stability, forecast.used_fallback_equal_weighting)
    if caveat is not None:
        sentences.append(caveat)

    sentences.append(
        _narrative_dominance_sentence(
            contributions, net_lean_pct(forecast), negligible_net_lean_pct, tie_tolerance_pct
        )
    )

    if macro_regime_state is not None:
        macro_sentence = _MACRO_REGIME_SENTENCES.get(
            macro_regime_state,
            f"Macro regime reported as {macro_regime_state}, which has no mapped "
            f"trading implication in this renderer.",
        )
        sentences.append(macro_sentence)

    sentences.append(_narrative_ev_sentence(risk, gate_passes))

    quality = _narrative_data_quality_sentence(contributions, ci_available)
    if quality is not None:
        sentences.append(quality)

    return sentences


# ------------------------------------------------------------------------ #
# Final rendered output
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class RenderedContributionLine:
    label: str
    detail: str
    impact_pct: float
    prose: str  # the template-built phrase actually rendered
    is_degraded: bool = False


@dataclass(frozen=True)
class ExplainabilityOutput:
    asset: str
    setup_label: str  # "LONG SETUP" | "SHORT SETUP" | "NO SETUP"
    called_direction: str
    called_probability_pct: float
    ci_lower_pct: Optional[float]
    ci_upper_pct: Optional[float]
    analog_n_independent: int
    contributions: List[RenderedContributionLine]
    ev_estimate_pct: float
    ev_lower_ci_pct: float
    ev_gate_passes: bool
    position_size_pct: float
    stability_status: str  # "STABLE" | "UNSTABLE" | "NOT_EVALUATED"
    used_fallback_equal_weighting: bool
    net_lean_pct: float
    narrative: List[str] = field(default_factory=list)
    spec_report_text: str = ""  # the Adaptive doc's exact format, section-for-section
    report_text: str = ""  # spec_report_text plus the rule-based Assessment block


def _format_ci(ci_lower_pct: Optional[float], ci_upper_pct: Optional[float], n: int) -> str:
    if ci_lower_pct is None or ci_upper_pct is None:
        return "95% CI: unavailable (insufficient analog sample)"
    return f"95% CI: {ci_lower_pct:.0f}%–{ci_upper_pct:.0f}%, N={n} independent analogues"


def _format_stability(
    stability: Optional[WeightStabilityReport], used_fallback: bool
) -> str:
    suffix = " (using equal-weight fallback)" if used_fallback else ""
    if stability is None or stability.status == "NOT_EVALUATED":
        return f"Ensemble weight stability: NOT_EVALUATED (fewer than 2 folds observed){suffix}"
    delta = "unknown" if stability.max_abs_delta is None else f"{stability.max_abs_delta:.2f}"
    if stability.status == "STABLE":
        return f"Ensemble weight stability: STABLE (max fold delta: {delta}){suffix}"
    return (
        f"Ensemble weight stability: UNSTABLE (max fold delta: {delta}, "
        f"flagged on {stability.triggering_signal}){suffix}"
    )


def _render_spec_report(
    asset: str,
    setup_label: str,
    directional_adjective: str,
    called_probability_pct: float,
    ci_line: str,
    lines: Sequence[RenderedContributionLine],
    risk: TradeRiskSummary,
    gate_passes: bool,
    stability_line: str,
) -> str:
    """The Adaptive doc's Layer 8 output format, section-for-section:
    header / probability+CI / signal contributions / EV estimate / EV lower
    CI bound with gate verdict / position size / stability."""
    body_lines: List[str] = []
    if lines:
        prefixes = [f"  {'+' if line.impact_pct >= 0 else '-'} {line.prose}:" for line in lines]
        width = max(len(p) for p in prefixes)
        for prefix, line in zip(prefixes, lines):
            body_lines.append(f"{prefix.ljust(width)} {line.impact_pct:+.0f}%")
    contributions_block = (
        "\n".join(body_lines) if body_lines else "  (no individually significant contributions)"
    )

    gate_word = "passes" if gate_passes else "FAILS"
    return (
        f"{asset} — {setup_label}\n"
        f"{directional_adjective} Probability: {called_probability_pct:.0f}% ({ci_line})\n"
        f"Signal contributions:\n"
        f"{contributions_block}\n"
        f"EV estimate: {risk.ev_estimate_pct:+.2f}% per unit risk\n"
        f"EV lower CI bound: {risk.ev_lower_ci_pct:+.2f}% "
        f"[GATE: {gate_word} +{EV_LOWER_CI_GATE_PCT:.2f}% threshold]\n"
        f"Position size: {risk.position_size_pct:.1f}% of capital\n"
        f"{stability_line}"
    )


class ExplainabilityEngine:
    """
    Renders one `ExplainabilityOutput` per call to `explain()`.

    Pure template/rule-based and fully deterministic: no LLM, no network,
    no API key, no randomness, no clock. The same inputs produce
    byte-for-byte identical output on every run, forever -- asserted
    directly by this module's test suite, because an audit log that cannot
    be reproduced cannot be audited.
    """

    def __init__(
        self,
        *,
        min_abs_pct_to_show: float = DEFAULT_MIN_ABS_PCT_TO_SHOW,
        tie_tolerance_pct: float = TIE_TOLERANCE_PCT,
        negligible_net_lean_pct: float = NEGLIGIBLE_NET_LEAN_PCT,
    ) -> None:
        for name, value in (
            ("min_abs_pct_to_show", min_abs_pct_to_show),
            ("tie_tolerance_pct", tie_tolerance_pct),
            ("negligible_net_lean_pct", negligible_net_lean_pct),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}.")
        self.min_abs_pct_to_show = min_abs_pct_to_show
        self.tie_tolerance_pct = tie_tolerance_pct
        self.negligible_net_lean_pct = negligible_net_lean_pct

    def explain(
        self,
        *,
        asset: str,
        forecast: EnsembleForecast,
        contributions: Sequence[SignalContribution],
        risk: TradeRiskSummary,
        stability: Optional[WeightStabilityReport] = None,
        macro_regime_state: Optional[str] = None,
    ) -> ExplainabilityOutput:
        """
        Parameters
        ----------
        macro_regime_state : str, optional
            Layer 3's `MacroRegimeResult.state`, used to select a
            pre-written macro sentence. Passed explicitly rather than
            parsed back out of the macro contribution's `detail` string --
            reconstructing structured state from rendered text would be
            exactly the kind of fragile coupling this project avoids.
            Omitted -> the macro sentence is skipped entirely rather than
            guessed.
        """
        if forecast.called_direction not in _SETUP_LABELS:
            raise ValueError(f"unrecognized called_direction {forecast.called_direction!r}.")
        if not asset.strip():
            raise ValueError("asset must be a non-empty string.")

        ordered = sorted(contributions, key=_contribution_sort_key)
        rendered = [
            RenderedContributionLine(
                label=c.label,
                detail=c.detail,
                impact_pct=c.impact_pct,
                prose=_phrase_for(c),
                is_degraded=c.is_degraded,
            )
            for c in ordered
        ]

        ci = direction_probability_ci(forecast)
        ci_lower_pct = None if ci is None else ci[0] * 100.0
        ci_upper_pct = None if ci is None else ci[1] * 100.0
        called_probability_pct = forecast.called_probability * 100.0
        gate_passes = risk.ev_lower_ci_pct > EV_LOWER_CI_GATE_PCT
        stability_status = "NOT_EVALUATED" if stability is None else stability.status
        setup_label = _SETUP_LABELS[forecast.called_direction]
        directional_adjective = _DIRECTIONAL_ADJECTIVE[forecast.called_direction]
        stability_line = _format_stability(stability, forecast.used_fallback_equal_weighting)
        ci_line = _format_ci(ci_lower_pct, ci_upper_pct, forecast.analog_n_independent)

        spec_report_text = _render_spec_report(
            asset,
            setup_label,
            directional_adjective,
            called_probability_pct,
            ci_line,
            rendered,
            risk,
            gate_passes,
            stability_line,
        )

        narrative = _build_narrative(
            ordered,
            forecast,
            risk,
            gate_passes,
            stability,
            macro_regime_state,
            ci_available=ci is not None,
            negligible_net_lean_pct=self.negligible_net_lean_pct,
            tie_tolerance_pct=self.tie_tolerance_pct,
        )
        narrative_block = "\n".join(f"  {s}" for s in narrative)
        report_text = f"{spec_report_text}\n\nAssessment:\n{narrative_block}"

        return ExplainabilityOutput(
            asset=asset,
            setup_label=setup_label,
            called_direction=forecast.called_direction,
            called_probability_pct=called_probability_pct,
            ci_lower_pct=ci_lower_pct,
            ci_upper_pct=ci_upper_pct,
            analog_n_independent=forecast.analog_n_independent,
            contributions=rendered,
            ev_estimate_pct=risk.ev_estimate_pct,
            ev_lower_ci_pct=risk.ev_lower_ci_pct,
            ev_gate_passes=gate_passes,
            position_size_pct=risk.position_size_pct,
            stability_status=stability_status,
            used_fallback_equal_weighting=forecast.used_fallback_equal_weighting,
            net_lean_pct=net_lean_pct(forecast),
            narrative=narrative,
            spec_report_text=spec_report_text,
            report_text=report_text,
        )
