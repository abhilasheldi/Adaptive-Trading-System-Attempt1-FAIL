"""
Explainability Engine (Layer 8).

Per the Adaptive doc's Layer 8 section: "generates human-readable
explanations mapping feature attribution weights to individual prediction
outputs... serves two functions: it builds justified trust in the system
by making the decision auditable, and it makes losses diagnosable -- when
a trade fails, the explanation identifies which inputs were misleading."
Prereq (Prompt Playbook 2F): the Ensemble Meta-Model (Layer 7C,
`predictive_core.ensemble`), already built.

This module is a RENDERER, not a computer of new numbers: every quantity
in its output (direction probability, EV, position size, weight
stability) is either read directly from an already-computed upstream
object (`ensemble.EnsembleForecast`, `ensemble.WeightStabilityReport`) or
derived by an exact, documented, provable formula from one (the per-signal
contribution breakdown, the 95% CI) -- see `contributions_from_ensemble`
and `direction_probability_ci` for exactly which. It never re-derives
Component A/B, macro regime, MTF alignment, or liquidity trap logic itself
-- that would risk drifting out of sync with `ensemble.py`'s own
arithmetic, exactly the failure mode this project's cross-module reviews
have repeatedly caught elsewhere.

Two inputs this module explicitly does NOT compute, and accepts as
already-computed (`TradeRiskSummary`): EV estimate, EV lower CI bound, and
position size. These are Layer 9 (Risk Engine) and Layer 10 (Position
Sizing / quarter-Kelly) outputs per the Adaptive doc's own layer
boundaries -- neither layer exists in this codebase yet (Prompt Playbook
2G, not yet built). Rather than fabricate a placeholder Risk/Sizing
computation inside an Explainability module (whose job is to RENDER
decisions, not make them), the caller supplies these three numbers
directly, exactly as `mtf_alignment.py` accepts already-derived
per-timeframe directional reads instead of computing them from a raw
Layer 2 feature vector itself.

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
and provable: summing `impact_pct_i` over all five signals equals exactly
`100 * (p_final[called] - p_final[opposite])`, the final blend's own net
lean toward the called direction versus its opposite -- verified directly
in this module's test suite, not merely asserted. A signal that actually
disagrees with the ensemble's called direction (despite being outweighted
by the others) produces a genuinely NEGATIVE `impact_pct` under this
formula -- this is how the format's "+/-" mix (e.g. "RSI bearish
divergence" reading negative in a bullish setup) arises here, from a
single uniform formula applied to all five signals, not two different
formulas for the positive and negative sides.

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
derived ones -- this module renders whatever structured contributions
it's given, from any source; it only computes the five formal ones itself
because those alone have an exact, provable formula behind them.

95% CI on the direction probability -- an approximation, clearly labeled
as one. The Adaptive doc's own example ties the CI directly to N ("95%
CI: 66%-81%, N=187 independent analogues"), so `direction_probability_ci`
uses a standard Wilson score interval (a well-established, standard
statistical method -- not invented here) treating the ensemble's called
probability as if it were a proportion estimated from N independent
Bernoulli trials, with N = Component A's own `analog_n_independent`. This
is a real approximation, not a rigorous derivation: N is genuinely only
Component A's independent-analogue count, not a sample size behind the
other four blended signals. Returns None (rendered as "unavailable") when
Component A did not contribute a valid forecast (`analog_is_valid=False`)
-- fabricating a CI with no analog sample behind it at all would be worse
than omitting one.

Live prose generation -- cheap/fast model tier, numeric-safety boundary.
Per the Prompt Playbook's 2F build instruction ("Haiku-tier API call for
live text"), `AnthropicProseGenerator` calls the Anthropic API with a
cheap, fast model to turn each contribution's already-computed structured
facts (label, detail string) into a short natural-language phrase. The
model is NEVER shown, and never asked to produce, any of this module's
actual numbers (percentages, N, EV, position size) -- it only phrases the
descriptive text for each contribution; every number in the final
rendered report always comes from this module's own arithmetic,
regardless of which prose path (live or template) produced the
descriptive text around it. This is a deliberate defense-in-depth
boundary: an LLM has no way to corrupt a number it is never given.

Graceful degradation (Done-when bar: "degrades gracefully if the API call
fails"). `ExplainabilityEngine.explain()` treats ANY failure of the live
prose path -- network error, auth error, timeout, malformed/incomplete
JSON response, or any other exception a caller-injected prose generator
raises -- identically: catch it, fall back to a deterministic,
template-based phrase for EVERY contribution (all-or-nothing, never a
partial mix of live and template phrasing within one report), and report
`used_live_api=False` on the result so the fallback is visible, not
silent. `enable_live_api=False` skips the live path entirely (useful for
tests, cost control, or deterministic/offline runs) -- verified directly
in this module's tests that the API path is never even attempted when the
flag is off, not merely that its result is ignored.
"""

import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
# specifies a model, token budget, timeout, or display-filtering threshold
# for this layer; "Haiku-tier" is the Prompt Playbook's own wording, not a
# specific model ID).
DEFAULT_PROSE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROSE_MAX_TOKENS = 600
DEFAULT_PROSE_TIMEOUT_SECONDS = 8.0
DEFAULT_MIN_ABS_PCT_TO_SHOW = 1.0  # contributions below this magnitude are dropped from the rendered breakdown

_DISPLAY_LABELS: Dict[str, str] = {
    "component_a": "Historical analog match",
    "component_b": "Direction classifier",
    "macro_regime": "Macro regime",
    "mtf_alignment": "Multi-timeframe alignment",
    "liquidity_trap": "Liquidity trap",
}

_SETUP_LABELS: Dict[str, str] = {"up": "LONG SETUP", "down": "SHORT SETUP", "neutral": "NO SETUP"}
_DIRECTIONAL_ADJECTIVE: Dict[str, str] = {"up": "Bullish", "down": "Bearish", "neutral": "Neutral"}


class ProseGenerationError(Exception):
    """Raised internally whenever the live prose path fails for any reason
    (network, auth, timeout, malformed/incomplete response). Caught by
    `ExplainabilityEngine.explain()` to trigger the template fallback --
    never expected to escape this module's own public methods."""


# ------------------------------------------------------------------------ #
# Structured inputs
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class SignalContribution:
    """One line item in the 'Signal contributions' breakdown: a label
    (matches an `ensemble.SIGNAL_NAMES` entry for ensemble-derived
    contributions, or any caller-chosen id for additional context lines),
    a plain deterministic detail string (used for the template fallback
    and as grounding context for the live prose path), and a signed
    percentage-point impact (always this module's own number -- never
    supplied or altered by the LLM, see module docstring)."""

    label: str
    detail: str
    impact_pct: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("SignalContribution.label must be non-empty.")
        if not self.detail.strip():
            raise ValueError("SignalContribution.detail must be non-empty.")
        if not np.isfinite(self.impact_pct):
            raise ValueError(f"SignalContribution.impact_pct must be finite, got {self.impact_pct}.")


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
            raise ValueError(f"TradeRiskSummary.position_size_pct must be non-negative, got {self.position_size_pct}.")


# ------------------------------------------------------------------------ #
# Signal-contribution breakdown (provable, see module docstring)
# ------------------------------------------------------------------------ #


def _default_detail_for_signal(name: str, signals: EnsembleSignalSet) -> str:
    if name == "component_a":
        a = signals.component_a
        if not a.is_valid:
            return "insufficient independent analogues"
        return f"N={a.n_independent}, distance threshold {a.distance_threshold:.3g}"
    if name == "component_b":
        b = signals.component_b
        probs = {"down": b.p_down, "neutral": b.p_neutral, "up": b.p_up}
        called = max(probs, key=probs.get)
        return f"called {called} at {probs[called]:.0%} confidence"
    if name == "macro_regime":
        m = signals.macro_regime
        return f"{m.state}, confidence {m.confidence:.2f}"
    if name == "mtf_alignment":
        t = signals.mtf_alignment
        return f"{t.score}/4 {t.dominant_direction}"
    if name == "liquidity_trap":
        lq = signals.liquidity_trap
        if lq.sweep_just_confirmed:
            return f"{lq.sweep_direction} sweep confirmed, probability {lq.trap_probability:.0%}"
        return f"no confirmed sweep, trap probability {lq.trap_probability:.0%}"
    raise ValueError(f"no default detail formatter for signal {name!r}.")  # unreachable for ensemble.SIGNAL_NAMES


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
    the exact formula and its provable sum-to-net-lean identity. Sorted
    positive-first (largest first), then negative (largest magnitude
    first) -- matching the Adaptive doc's example ordering. Contributions
    with `|impact_pct| < min_abs_pct_to_show` (a judgment call, not a spec
    number -- avoids a "+0%"/"-0%" line for a signal that barely moved the
    needle) are dropped.
    """
    called_idx = ensemble.DIRECTION_CLASSES.index(forecast.called_direction)
    if forecast.called_direction == "up":
        opposite_idx: Optional[int] = ensemble.DIRECTION_CLASSES.index("down")
    elif forecast.called_direction == "down":
        opposite_idx = ensemble.DIRECTION_CLASSES.index("up")
    else:  # "neutral" has no single natural opposite -- see module docstring
        opposite_idx = None

    _, per_signal = ensemble.blend_signals_detailed(
        signals, forecast.weights_used,
        min_independent_instances=min_independent_instances,
        analog_confidence_ramp_multiple=analog_confidence_ramp_multiple,
    )

    down_idx, up_idx = ensemble.DIRECTION_CLASSES.index("down"), ensemble.DIRECTION_CLASSES.index("up")
    contributions = []
    for name, (w, dist) in per_signal.items():
        if opposite_idx is not None:
            net_lean = dist[called_idx] - dist[opposite_idx]
        else:
            net_lean = dist[called_idx] - max(dist[down_idx], dist[up_idx])
        impact_pct = 100.0 * w * net_lean
        if abs(impact_pct) < min_abs_pct_to_show:
            continue
        contributions.append(SignalContribution(
            label=name, detail=_default_detail_for_signal(name, signals), impact_pct=impact_pct,
        ))
    return sorted(contributions, key=lambda c: (0 if c.impact_pct >= 0 else 1, -abs(c.impact_pct)))


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


# ------------------------------------------------------------------------ #
# Prose generation: template fallback + live API path
# ------------------------------------------------------------------------ #


def _template_prose(contribution: SignalContribution) -> str:
    display = _DISPLAY_LABELS.get(contribution.label, contribution.label.replace("_", " ").title())
    return f"{display} ({contribution.detail})"


def _validate_and_normalize_phrases(
    phrases: Dict[str, str], contributions: Sequence[SignalContribution],
) -> Dict[str, str]:
    """Applied to EVERY prose source (live or a caller-injected generator)
    -- a badly-behaved custom generator must trigger the same safe
    fallback a live-API failure does, not propagate a partially-broken
    result."""
    normalized: Dict[str, str] = {}
    missing = []
    for c in contributions:
        phrase = phrases.get(c.label)
        if not isinstance(phrase, str) or not phrase.strip():
            missing.append(c.label)
            continue
        normalized[c.label] = phrase.strip()
    if missing:
        raise ProseGenerationError(f"prose generator returned no usable phrase for: {missing}")
    return normalized


def _build_prose_prompt(contributions: Sequence[SignalContribution]) -> str:
    lines = [
        f'- id: "{c.label}", detail: "{c.detail}"'
        for c in contributions
    ]
    return (
        "You are formatting trade-signal explanations for a systematic trading "
        "system's audit log. For each signal below, write ONE short (under 12 "
        "words) natural-language phrase in the style "
        '"Market structure (BOS + CHOCH confirmed, 4H)" or '
        '"Multi-timeframe alignment (4/4 bullish)" -- a short label followed by '
        "a parenthetical with the concrete detail. Do NOT include any percentage "
        "numbers, and do not invent facts beyond the detail given.\n\n"
        "Signals:\n" + "\n".join(lines) + "\n\n"
        "Respond with ONLY a JSON object mapping each id to its phrase, no other "
        'text, e.g. {"component_a": "...", "macro_regime": "..."}'
    )


def _extract_json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ProseGenerationError(f"no JSON object found in prose response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class AnthropicProseGenerator:
    """
    Live prose generator -- Anthropic API, cheap/fast ("Haiku-tier") model
    per the Prompt Playbook's 2F build instruction. Constructs its client
    LAZILY, on first actual call, never at `__init__` -- so building an
    `ExplainabilityEngine(enable_live_api=True)` never requires an API key
    by itself; only actually generating prose does (and if that fails for
    any reason, `explain()` falls back to the template path -- see module
    docstring).
    """

    def __init__(
        self,
        model: str = DEFAULT_PROSE_MODEL,
        max_tokens: int = DEFAULT_PROSE_MAX_TOKENS,
        timeout: float = DEFAULT_PROSE_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily -- see class docstring
            self._client = anthropic.Anthropic(timeout=self.timeout)
        return self._client

    def __call__(self, contributions: Sequence[SignalContribution]) -> Dict[str, str]:
        return self.generate(contributions)

    def generate(self, contributions: Sequence[SignalContribution]) -> Dict[str, str]:
        import anthropic  # local import: only required when the live path actually runs

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": _build_prose_prompt(contributions)}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            parsed = _extract_json_object(text)
        except ProseGenerationError:
            raise
        except (anthropic.AnthropicError, json.JSONDecodeError) as exc:
            raise ProseGenerationError(f"live prose generation failed: {exc}") from exc
        except Exception as exc:
            raise ProseGenerationError(f"live prose generation failed unexpectedly: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ProseGenerationError(f"live prose response was not a JSON object: {type(parsed).__name__}")
        return _validate_and_normalize_phrases(parsed, contributions)


# ------------------------------------------------------------------------ #
# Final rendered output
# ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class RenderedContributionLine:
    label: str
    detail: str
    impact_pct: float
    prose: str  # the phrase actually used -- live-generated or template fallback


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
    used_live_api: bool
    report_text: str


def _format_ci(ci_lower_pct: Optional[float], ci_upper_pct: Optional[float], n: int) -> str:
    if ci_lower_pct is None or ci_upper_pct is None:
        return "95% CI: unavailable (insufficient analog sample)"
    return f"95% CI: {ci_lower_pct:.0f}%–{ci_upper_pct:.0f}%, N={n} independent analogues"


def _format_stability(stability: Optional[WeightStabilityReport], used_fallback: bool) -> str:
    suffix = " (using equal-weight fallback)" if used_fallback else ""
    if stability is None or stability.status == "NOT_EVALUATED":
        return f"Ensemble weight stability: NOT_EVALUATED (fewer than 2 folds observed){suffix}"
    if stability.status == "STABLE":
        return f"Ensemble weight stability: STABLE (max fold delta: {stability.max_abs_delta:.2f}){suffix}"
    return (
        f"Ensemble weight stability: UNSTABLE (max fold delta: {stability.max_abs_delta:.2f}, "
        f"flagged on {stability.triggering_signal}){suffix}"
    )


def _render_report_text(
    asset: str,
    setup_label: str,
    directional_adjective: str,
    called_probability_pct: float,
    ci_line: str,
    lines: List[RenderedContributionLine],
    risk: TradeRiskSummary,
    gate_passes: bool,
    stability_line: str,
) -> str:
    body_lines = []
    if lines:
        prefixes = [f"  {'+' if line.impact_pct >= 0 else '-'} {line.prose}:" for line in lines]
        width = max(len(p) for p in prefixes)
        for prefix, line in zip(prefixes, lines):
            body_lines.append(f"{prefix.ljust(width)} {line.impact_pct:+.0f}%")
    contributions_block = "\n".join(body_lines) if body_lines else "  (no individually significant contributions)"

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
    Renders one `ExplainabilityOutput` per call to `explain()`. See module
    docstring for the full design: this class computes nothing about the
    trade itself -- it formats already-computed inputs, using either a
    live Anthropic API call (cheap/fast tier) or a deterministic template
    to phrase each contribution line.
    """

    def __init__(
        self,
        enable_live_api: bool = True,
        prose_generator: Optional[Callable[[Sequence[SignalContribution]], Dict[str, str]]] = None,
        model: str = DEFAULT_PROSE_MODEL,
        max_tokens: int = DEFAULT_PROSE_MAX_TOKENS,
        timeout: float = DEFAULT_PROSE_TIMEOUT_SECONDS,
    ) -> None:
        """
        Parameters
        ----------
        enable_live_api : bool
            The Done-when bar's "config flag to swap in a template-based
            fallback". False forces the template path unconditionally --
            the live generator (real or injected) is never even called
            (verified directly in this module's tests, not just that its
            result would be ignored).
        prose_generator : callable, optional
            Dependency injection point for tests (and for swapping in a
            different provider without subclassing) -- any
            `Callable[[Sequence[SignalContribution]], Dict[str, str]]`.
            When omitted, a real `AnthropicProseGenerator` is constructed
            lazily on first use (never at `__init__` -- see that class's
            docstring).
        """
        self.enable_live_api = enable_live_api
        self._injected_generator = prose_generator
        self._model, self._max_tokens, self._timeout = model, max_tokens, timeout
        self._lazy_generator: Optional[AnthropicProseGenerator] = None

    def _get_generator(self) -> Callable[[Sequence[SignalContribution]], Dict[str, str]]:
        if self._injected_generator is not None:
            return self._injected_generator
        if self._lazy_generator is None:
            self._lazy_generator = AnthropicProseGenerator(
                model=self._model, max_tokens=self._max_tokens, timeout=self._timeout,
            )
        return self._lazy_generator

    def explain(
        self,
        *,
        asset: str,
        forecast: EnsembleForecast,
        contributions: Sequence[SignalContribution],
        risk: TradeRiskSummary,
        stability: Optional[WeightStabilityReport] = None,
    ) -> ExplainabilityOutput:
        if forecast.called_direction not in _SETUP_LABELS:
            raise ValueError(f"unrecognized called_direction {forecast.called_direction!r}.")

        used_live_api = False
        phrases: Optional[Dict[str, str]] = None
        if self.enable_live_api and contributions:
            try:
                generator = self._get_generator()
                phrases = _validate_and_normalize_phrases(generator(contributions), contributions)
                used_live_api = True
            except Exception:  # noqa: BLE001 -- ANY failure of the live path must degrade gracefully (see module docstring)
                phrases = None
                used_live_api = False

        rendered = [
            RenderedContributionLine(
                label=c.label, detail=c.detail, impact_pct=c.impact_pct,
                prose=(phrases[c.label] if phrases is not None else _template_prose(c)),
            )
            for c in contributions
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

        report_text = _render_report_text(
            asset, setup_label, directional_adjective, called_probability_pct, ci_line,
            rendered, risk, gate_passes, stability_line,
        )

        return ExplainabilityOutput(
            asset=asset, setup_label=setup_label, called_direction=forecast.called_direction,
            called_probability_pct=called_probability_pct, ci_lower_pct=ci_lower_pct, ci_upper_pct=ci_upper_pct,
            analog_n_independent=forecast.analog_n_independent, contributions=rendered,
            ev_estimate_pct=risk.ev_estimate_pct, ev_lower_ci_pct=risk.ev_lower_ci_pct, ev_gate_passes=gate_passes,
            position_size_pct=risk.position_size_pct, stability_status=stability_status,
            used_fallback_equal_weighting=forecast.used_fallback_equal_weighting, used_live_api=used_live_api,
            report_text=report_text,
        )
