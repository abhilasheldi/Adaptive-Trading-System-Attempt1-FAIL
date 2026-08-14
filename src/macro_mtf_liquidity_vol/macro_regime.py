"""
Macro Regime Engine (Layer 3) -- MASTER_CONTEXT's exact four-state macro
regime table:

    Risk-On:            DXY falling/flat, VIX < 20 and falling, BTC
                         dominance expanding, BTC/QQQ correlation positive
                         -> full trading, standard EV thresholds
    Risk-Off:            DXY rising, VIX > 25 or spiking, cross-asset
                         correlations compressing toward 1.0 -> no new long
                         entries in risk assets; existing positions
                         reviewed for exit
    Transitional:        Mixed signals, no clear directional consensus ->
                         halve position sizes, apply more conservative EV
                         threshold
    Macro-Event Window:  Scheduled high-impact event within 24 hours
                         (FOMC, NFP, CPI, BOJ meeting, GDP release) -> no
                         new entries; existing positions get tighter stops

This is a CLASSIFIER reading already-computed current signals, not a
forecaster -- it does not compute DXY momentum, VIX ROC, correlations, or
maintain an economic calendar itself. All of that is upstream data-
ingestion work (data_engine.macro_fetcher for the market-data signals; the
caller is responsible for supplying the next scheduled high-impact event,
since no economic-calendar source is wired into this project yet).

Precedence (a design decision, since MASTER_CONTEXT's table doesn't state
one explicitly): Macro-Event Window is checked FIRST and short-circuits
Risk-On/Risk-Off/Transitional evaluation entirely. Rationale: the table's
own "no new entries" language for an imminent high-impact event reads as
an unconditional caution instruction, not one that should be overridden by
an otherwise-calm signal reading -- a FOMC decision 6 hours out is a reason
to hold back on new entries regardless of how benign VIX/DXY currently
look, precisely because scheduled macro events are exactly the situation
where currently-calm readings are least trustworthy as a forward guide.

US10Y is accepted as an input (`us10y_level_pct` / `us10y_20d_direction_pct`)
because the Prompt Playbook's Layer 3 build instruction explicitly lists it
as one -- but checked directly against MASTER_CONTEXT's own regime table
(docs/MASTER_CONTEXT.md, and the source .docx it was extracted from) and
neither of them uses US10Y in ANY of the four regime conditions. Rather
than inventing an unspecified rule for it, this module accepts it, uses it
in NEITHER classification NOR confidence, and echoes it back on the result
for downstream/diagnostic visibility only. If a real justification for
using it surfaces later, add it explicitly and document why -- don't guess.

Every threshold below is labeled as either an EXACT MASTER_CONTEXT number
(hardcoded, not a function parameter -- consistent with how this project
already treats other exact spec numbers, e.g.
`HistoricalAnalogEngine`'s N>=50 gate) or an explicit, documented judgment
call (a tunable parameter, because MASTER_CONTEXT's table leaves it
genuinely unspecified): "VIX spiking" has no numeric definition anywhere
in the source docs (the *asset* volatility Spike definition -- "ATR
doubles within 5 candles" -- is a different variable, Layer 6's, not
VIX's), and "correlations compressing toward 1.0" has no stated numeric
threshold either. Both are configurable, clearly-labeled judgment calls,
not spec numbers -- don't mistake the defaults for MASTER_CONTEXT text.

MASTER_CONTEXT's own note ("BTC/QQQ rolling 20-day correlation determines
whether macro conditioning is even relevant to BTC at all -- in low-
correlation periods, reduce macro weight rather than applying it blindly")
is implemented as a SEPARATE output field (`macro_weight`), independent of
`confidence` and independent of which state was classified: `macro_weight`
is a continuous function of |BTC/QQQ correlation| alone, computed the same
way regardless of state, so a caller can (and per that note, should) scale
down how much this regime read matters even when the state itself reads
confidently Risk-On or Risk-Off.

CORRECTION (2026-08-13 review): the first version of this module scaled
`macro_weight` against an invented 0.5 correlation threshold, because
MASTER_CONTEXT.md's condensed note gives no exact number. A fuller re-read
of the Adaptive Probabilistic Trade Management System doc's own Layer 3
section -- which the first version's search missed -- gives an EXACT rule:
"When correlation is high (> 0.6), the macro signal is applied at full
weight. When correlation is low (< 0.3), the macro signal weight is
reduced proportionally... This is a dynamic weight, not a binary on/off."
A worked example in the same document's "Full System Output Specification"
corroborates it directly: "BTC/QQQ corr: 0.64 -> macro weight: FULL".
`CORRELATION_FULL_WEIGHT_THRESHOLD = 0.6` is now the exact, hardcoded
anchor (no longer a tunable parameter, matching this project's convention
for confirmed spec numbers). The source text gives only ONE precise
breakpoint (0.6 -> full weight); it does not give a second number
specifying exactly how "reduced" the weight should be at or below 0.3, so
rather than inventing an unstated second parameter to build a true
three-piece function, `_macro_weight` implements the simplest function
consistent with BOTH stated behaviors: a single linear ramp from
(corr=0, weight=0) to (corr=0.6, weight=1.0), clipped at 1.0 above 0.6.
This reproduces "> 0.6 -> full weight" exactly, and at corr=0.3 yields
weight=0.5 -- unambiguously "reduced" relative to full weight, consistent
with (if not uniquely pinned down by) the "< 0.3 -> reduced" description.
`CORRELATION_REDUCED_WEIGHT_THRESHOLD = 0.3` is kept as a named constant
purely for this documentation/consistency purpose, not as a second formula
breakpoint.

Two output fields the Adaptive doc's Layer 3 "Output:" line explicitly
requires were missing from the first version and are added in this
review: "current BTC/QQQ correlation coefficient" (now
`MacroRegimeResult.btc_qqq_corr_20d`, simply echoing the input) and
"macro-event window flag" (now `MacroRegimeResult.is_macro_event_window`,
a convenience boolean equal to `state == "Macro-Event Window"`). Both are
additive, non-breaking to the classification logic itself.

One wording discrepancy noted, not acted on: the Adaptive doc's own
Macro-Event Window action text is "No new entries; existing positions get
tighter stops OR STANDING EXIT INSTRUCTIONS" -- slightly fuller than
MASTER_CONTEXT.md's "No new entries; existing positions get tighter
stops", which is what `_TRADING_IMPLICATIONS` uses verbatim.
`docs/MASTER_CONTEXT.md` is this project's explicitly-labeled "exact
table" reference and has been treated as authoritative throughout this
project; the Adaptive doc's phrasing is not contradictory, just more
expansive, so the shorter MASTER_CONTEXT.md string is kept rather than
switched.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# ------------------------------------------------------------------------ #
# Exact MASTER_CONTEXT numbers -- hardcoded, not configurable (consistent
# with how this project treats other exact spec numbers elsewhere, e.g.
# HistoricalAnalogEngine's minimum-50-independent-instances gate).
# ------------------------------------------------------------------------ #
VIX_RISK_ON_MAX = 20.0
VIX_RISK_OFF_MIN = 25.0
EVENT_WINDOW_HOURS = 24.0
# Source: Adaptive doc Layer 3 ("When correlation is high (> 0.6), the
# macro signal is applied at full weight"), corroborated by that doc's own
# worked example ("BTC/QQQ corr: 0.64 -> macro weight: FULL"). See module
# docstring's 2026-08-13 correction note for why 0.3 is kept only as a
# documentation reference, not a second formula breakpoint.
CORRELATION_FULL_WEIGHT_THRESHOLD = 0.6
CORRELATION_REDUCED_WEIGHT_THRESHOLD = 0.3

# Judgment-call thresholds/scales -- NOT MASTER_CONTEXT/Adaptive-doc exact
# numbers (see classify_macro_regime's docstring for what each controls).
# Exposed as named module-level constants, not just inline function
# defaults (2026-08-13 cross-module re-audit finding): `mtf_alignment.py`
# and `liquidity_trap.py` both expose their analogous judgment-call
# defaults this way (DEFAULT_NEUTRAL_BAND, DEFAULT_LEVEL_TOLERANCE_ATR,
# etc.) so they're independently importable/referenceable/testable: this
# module's five judgment-call defaults previously existed only as bare
# inline literals in the function signature, the one inconsistent
# exception across all four macro_mtf_liquidity_vol modules.
DEFAULT_VIX_SPIKE_5D_ROC_PCT = 30.0
DEFAULT_RISK_OFF_CORRELATION_THRESHOLD = 0.7
DEFAULT_CONFIDENCE_SCALE_DXY_PCT = 3.0
DEFAULT_CONFIDENCE_SCALE_VIX_ROC_PCT = 20.0
DEFAULT_CONFIDENCE_SCALE_DOMINANCE_PCT = 3.0

_TRADING_IMPLICATIONS = {
    "Risk-On": "Full trading, standard EV thresholds",
    "Risk-Off": "No new long entries in risk assets; existing positions reviewed for exit",
    "Transitional": "Halve position sizes, apply more conservative EV threshold",
    "Macro-Event Window": "No new entries; existing positions get tighter stops",
}


@dataclass(frozen=True)
class MacroSignalSnapshot:
    """
    Point-in-time macro signal readings this classifier consumes --
    already computed upstream (data_engine.macro_fetcher for the market
    signals; caller-supplied for the event calendar). This class does not
    compute or validate momentum/ROC/correlation math itself, only the
    values a caller has already derived, matching the "classifier, not
    forecaster" framing of this module.

    Parameters
    ----------
    timestamp : pd.Timestamp
        When this snapshot was taken (must be tz-aware or tz-naive
        consistently with `next_high_impact_event`; both are normalized to
        UTC internally regardless -- see `_hours_until`).
    dxy_20d_momentum_pct : float
        % change in DXY (or proxy) over the trailing 20 days. Sign is the
        "direction" MASTER_CONTEXT's table refers to: <= 0 is "falling or
        flat", > 0 is "rising".
    us10y_level_pct : float
        Current US 10-Year yield level, in percentage points (e.g. 4.5 =
        4.5%). NOT used in classification -- see module docstring.
    us10y_20d_direction_pct : float
        % change in the US10Y level over the trailing 20 days. NOT used in
        classification -- see module docstring.
    vix_level : float
        Current VIX level.
    vix_5d_roc_pct : float
        % rate of change in VIX over the trailing 5 days. Negative = VIX
        falling.
    btc_dominance_20d_trend_pct : float
        % change in BTC market-cap dominance over the trailing 20 days.
        Positive = expanding.
    btc_qqq_corr_20d : float
        Rolling 20-day correlation between BTC and QQQ returns, in
        [-1, 1].
    next_high_impact_event : Optional[pd.Timestamp]
        Timestamp of the next scheduled high-impact macro event (FOMC,
        NFP, CPI, BOJ meeting, GDP release), if any is known. None if no
        such event is currently scheduled/known to the caller. This
        project has no economic-calendar data source wired in yet --
        supplying this is the caller's responsibility.
    """

    timestamp: pd.Timestamp
    dxy_20d_momentum_pct: float
    us10y_level_pct: float
    us10y_20d_direction_pct: float
    vix_level: float
    vix_5d_roc_pct: float
    btc_dominance_20d_trend_pct: float
    btc_qqq_corr_20d: float
    next_high_impact_event: Optional[pd.Timestamp] = None

    # Every float field feeds either a hard classification comparison or a
    # confidence margin; none of them tolerate NaN. See __post_init__.
    _NUMERIC_FIELDS = (
        "dxy_20d_momentum_pct",
        "us10y_level_pct",
        "us10y_20d_direction_pct",
        "vix_level",
        "vix_5d_roc_pct",
        "btc_dominance_20d_trend_pct",
        "btc_qqq_corr_20d",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, pd.Timestamp):
            raise TypeError("timestamp must be a pd.Timestamp.")

        # NaN/inf rejection (2026-08-14 review). Previously only
        # `btc_qqq_corr_20d` was implicitly NaN-safe (via its range check);
        # a NaN `vix_level` passed `nan <= 0` and every downstream
        # comparison against it silently evaluated False, so ALL hard gates
        # failed and the classifier returned "Transitional" -- i.e. "halve
        # position sizes", meaning the system kept trading on entirely
        # missing macro data instead of refusing to classify. Worse, the
        # margin scores are built on `_clip01`, and `_clip01(nan)` returned
        # 1.0 (because `min(1.0, nan)` is 1.0 -- every NaN comparison is
        # False), so missing inputs produced MAXIMUM margin contributions.
        # This is a demonstrated live failure mode, not a theoretical one:
        # docs/project_state.md records yfinance returning six entirely-NaN
        # columns from this exact upstream pipeline.
        for field_name in self._NUMERIC_FIELDS:
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be a finite number, got {value!r}. Macro "
                    f"signals are never imputed here -- a missing upstream reading "
                    f"must be resolved by the caller, not silently classified."
                )

        if self.vix_level <= 0:
            raise ValueError(f"vix_level must be positive, got {self.vix_level}.")
        if not (-1.0 <= self.btc_qqq_corr_20d <= 1.0):
            raise ValueError(
                f"btc_qqq_corr_20d must be in [-1, 1], got {self.btc_qqq_corr_20d}."
            )
        if self.next_high_impact_event is not None and not isinstance(
            self.next_high_impact_event, pd.Timestamp
        ):
            raise TypeError("next_high_impact_event must be a pd.Timestamp or None.")


@dataclass(frozen=True)
class MacroRegimeResult:
    """
    Classifier output. `state` and `trading_implication` are MASTER_CONTEXT's
    exact table text. `confidence` and `macro_weight` are the "confidence/
    weight to apply downstream" this layer is required to produce -- kept
    as two DISTINCT numbers, not one, because they answer different
    questions (see module docstring): `confidence` is "how strongly do the
    signals match the classified state's own definition" (a graded margin,
    not a binary pass/fail -- every classified state's defining conditions
    are, by construction, already satisfied to at least clear the
    threshold, so a trivial 1.0 confidence would carry no information);
    `macro_weight` is "how much should downstream logic even listen to a
    macro regime read at all right now", driven purely by |BTC/QQQ
    correlation| per MASTER_CONTEXT's explicit note, independent of state
    or confidence.

    `btc_qqq_corr_20d` and `is_macro_event_window` (added in the 2026-08-13
    review) exist because the Adaptive doc's Layer 3 "Output:" line
    explicitly lists "current BTC/QQQ correlation coefficient" and a
    "macro-event window flag" as required outputs, distinct from the
    regime label itself -- see module docstring.
    """

    state: str  # "Risk-On" | "Risk-Off" | "Transitional" | "Macro-Event Window"
    trading_implication: str
    confidence: float  # [0, 1]
    macro_weight: float  # [0, 1]
    reasons: List[str]
    us10y_level_pct: float  # echoed for visibility only -- not used above
    us10y_20d_direction_pct: float
    btc_qqq_corr_20d: float  # echoed raw correlation coefficient -- explicit Adaptive-doc output requirement
    is_macro_event_window: bool  # == (state == "Macro-Event Window"); explicit Adaptive-doc output requirement

    def __post_init__(self) -> None:
        if self.state not in _TRADING_IMPLICATIONS:
            raise ValueError(f"invalid state {self.state!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}.")
        if not (0.0 <= self.macro_weight <= 1.0):
            raise ValueError(f"macro_weight must be in [0, 1], got {self.macro_weight}.")
        if not (-1.0 <= self.btc_qqq_corr_20d <= 1.0):
            raise ValueError(f"btc_qqq_corr_20d must be in [-1, 1], got {self.btc_qqq_corr_20d}.")
        if self.is_macro_event_window != (self.state == "Macro-Event Window"):
            raise ValueError(
                f"is_macro_event_window={self.is_macro_event_window} inconsistent with "
                f"state={self.state!r}."
            )


def _clip01(x: float) -> float:
    """
    Clamp to [0, 1], rejecting NaN.

    Defence in depth (2026-08-14 review): the plain `max(0.0, min(1.0, x))`
    form silently maps NaN to 1.0 -- the MAXIMUM -- because every comparison
    against NaN is False, so `min(1.0, nan)` returns 1.0. That turned a
    missing upstream reading into a maximum-confidence margin contribution.
    `MacroSignalSnapshot.__post_init__` now rejects non-finite inputs at the
    boundary, which is the primary guard; this check exists so the same
    class of bug cannot reappear via some future code path that reaches
    `_clip01` without going through that validation.
    """
    if not math.isfinite(x):
        raise ValueError(f"_clip01 received a non-finite value ({x!r}); this is a bug.")
    return max(0.0, min(1.0, x))


def _hours_until(reference: pd.Timestamp, event: pd.Timestamp) -> float:
    """
    Hours from `reference` to `event` (negative if `event` is already in
    the past). Normalizes both to tz-aware UTC before subtracting, rather
    than assuming matching tz-awareness -- the exact class of bug found and
    fixed in `HistoricalAnalogEngine.predict()` earlier this project (see
    docs/project_state.md, 2026-08-13 entries): real market data is UTC
    tz-aware, but a caller-supplied event calendar could easily be
    constructed tz-naive, and comparing the two directly would raise.
    """

    def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    return (_to_utc(event) - _to_utc(reference)).total_seconds() / 3600.0


def classify_macro_regime(
    snapshot: MacroSignalSnapshot,
    *,
    vix_spike_5d_roc_pct: float = DEFAULT_VIX_SPIKE_5D_ROC_PCT,
    risk_off_correlation_threshold: float = DEFAULT_RISK_OFF_CORRELATION_THRESHOLD,
    confidence_scale_dxy_pct: float = DEFAULT_CONFIDENCE_SCALE_DXY_PCT,
    confidence_scale_vix_roc_pct: float = DEFAULT_CONFIDENCE_SCALE_VIX_ROC_PCT,
    confidence_scale_dominance_pct: float = DEFAULT_CONFIDENCE_SCALE_DOMINANCE_PCT,
) -> MacroRegimeResult:
    """
    Classify a single point-in-time macro regime snapshot.

    Parameters
    ----------
    snapshot : MacroSignalSnapshot
    vix_spike_5d_roc_pct, risk_off_correlation_threshold : float
        Judgment-call thresholds for "VIX spiking" and "correlations
        compressing toward 1.0" -- MASTER_CONTEXT's table names both
        conditions but gives no numeric definition for either anywhere in
        the source docs. NOT exact spec numbers; tune freely, but don't
        mistake these defaults for MASTER_CONTEXT text.
    confidence_scale_dxy_pct, confidence_scale_vix_roc_pct,
    confidence_scale_dominance_pct : float
        Scale parameters controlling how quickly `confidence` saturates
        toward 1.0 as the underlying signals move further past their
        classification thresholds. Purely for the graded confidence
        number -- never used in the state classification itself, which is
        governed entirely by the hard thresholds above (and MASTER_CONTEXT's
        exact VIX_RISK_ON_MAX/VIX_RISK_OFF_MIN/EVENT_WINDOW_HOURS constants).
        `macro_weight`'s scale is NOT here -- it's the exact, hardcoded
        `CORRELATION_FULL_WEIGHT_THRESHOLD`, not a tunable (see module
        docstring's 2026-08-13 correction note).

    Returns
    -------
    MacroRegimeResult
    """
    reasons: List[str] = []

    # --- Macro-Event Window: checked first, short-circuits everything else
    # (see module docstring for why). ---
    if snapshot.next_high_impact_event is not None:
        hours_until = _hours_until(snapshot.timestamp, snapshot.next_high_impact_event)
        if 0.0 <= hours_until <= EVENT_WINDOW_HOURS:
            reasons.append(
                f"Scheduled high-impact event in {hours_until:.1f}h "
                f"(<= {EVENT_WINDOW_HOURS:.0f}h window)."
            )
            return MacroRegimeResult(
                state="Macro-Event Window",
                trading_implication=_TRADING_IMPLICATIONS["Macro-Event Window"],
                confidence=1.0,  # deterministic (calendar-driven), not a graded read
                macro_weight=_macro_weight(snapshot),
                reasons=reasons,
                us10y_level_pct=snapshot.us10y_level_pct,
                us10y_20d_direction_pct=snapshot.us10y_20d_direction_pct,
                btc_qqq_corr_20d=snapshot.btc_qqq_corr_20d,
                is_macro_event_window=True,
            )

    # --- Soft (margin-based, unclipped-by-hard-gate) scores for Risk-On
    # and Risk-Off, computed regardless of which one hard-classifies --
    # used both for the winning state's confidence and, for Transitional,
    # to express "how mixed" the picture actually is (see module
    # docstring / MacroRegimeResult docstring for why this beats a trivial
    # binary read). ---
    dxy_not_rising = snapshot.dxy_20d_momentum_pct <= 0.0
    dxy_rising = snapshot.dxy_20d_momentum_pct > 0.0

    vix_calm = snapshot.vix_level < VIX_RISK_ON_MAX and snapshot.vix_5d_roc_pct < 0.0
    vix_elevated_or_spiking = (
        snapshot.vix_level > VIX_RISK_OFF_MIN or snapshot.vix_5d_roc_pct >= vix_spike_5d_roc_pct
    )

    dominance_expanding = snapshot.btc_dominance_20d_trend_pct > 0.0
    correlation_positive = snapshot.btc_qqq_corr_20d > 0.0
    correlation_compressing_to_one = snapshot.btc_qqq_corr_20d >= risk_off_correlation_threshold

    risk_on = dxy_not_rising and vix_calm and dominance_expanding and correlation_positive
    risk_off = dxy_rising and vix_elevated_or_spiking and correlation_compressing_to_one
    # dxy_not_rising/dxy_rising are mutually exclusive by construction, so
    # risk_on and risk_off can never both be true -- no tie-breaking needed.

    risk_on_score = (
        _clip01(-snapshot.dxy_20d_momentum_pct / confidence_scale_dxy_pct)
        + (
            _clip01((VIX_RISK_ON_MAX - snapshot.vix_level) / VIX_RISK_ON_MAX)
            + _clip01(-snapshot.vix_5d_roc_pct / confidence_scale_vix_roc_pct)
        )
        / 2.0
        + _clip01(snapshot.btc_dominance_20d_trend_pct / confidence_scale_dominance_pct)
        + _clip01(snapshot.btc_qqq_corr_20d)
    ) / 4.0

    # NOTE: Risk-Off has three defining conditions (DXY, VIX, correlation --
    # no "dominance" leg, unlike Risk-On's four), so this averages over 3,
    # not 4. Correlation margin is normalized relative to
    # risk_off_correlation_threshold (not 0, unlike Risk-On's), since that
    # threshold -- not zero -- is where Risk-Off's condition actually starts
    # being satisfied.
    #
    # TWO FIXES (2026-08-14 review), both to make this expression obey the
    # rules it already stated for itself:
    #
    # 1. The VIX-ROC leg was `_clip01(roc / vix_spike_5d_roc_pct)`, which
    #    reads 1.0 -- the MAXIMUM -- exactly AT the spike threshold, while
    #    its two siblings in the same expression (the VIX-level leg and the
    #    correlation leg) both correctly read 0.0 at their own thresholds
    #    per the note above. A snapshot sitting at the bare minimum of all
    #    three Risk-Off conditions therefore reported inflated confidence,
    #    driven by a leg that should have contributed nothing. Now
    #    normalized threshold-relative like the other two: 0 at the spike
    #    threshold, 1 at twice it (mirroring the VIX-level leg's own
    #    (vix - 25) / 25 shape, which is 0 at 25 and 1 at 50).
    #
    # 2. The two VIX legs were AVERAGED, but `vix_elevated_or_spiking` is an
    #    OR, not an AND (unlike Risk-On's `vix_calm`, which genuinely is an
    #    AND and is still averaged below). Averaging halved the VIX
    #    contribution of a decisively-triggered condition -- e.g. VIX at 40
    #    with flat ROC scored (0.6 + 0) / 2 = 0.3 despite the condition
    #    being unambiguously met. `max()` matches the OR's semantics: the
    #    satisfied disjunct governs.
    risk_off_vix_margin = max(
        _clip01((snapshot.vix_level - VIX_RISK_OFF_MIN) / VIX_RISK_OFF_MIN),
        _clip01(
            (snapshot.vix_5d_roc_pct - vix_spike_5d_roc_pct)
            / max(vix_spike_5d_roc_pct, 1e-9)
        ),
    )
    risk_off_score = (
        _clip01(snapshot.dxy_20d_momentum_pct / confidence_scale_dxy_pct)
        + risk_off_vix_margin
        + _clip01(
            (snapshot.btc_qqq_corr_20d - risk_off_correlation_threshold)
            / max(1.0 - risk_off_correlation_threshold, 1e-9)
        )
    ) / 3.0

    macro_weight = _macro_weight(snapshot)

    if risk_on:
        reasons = [
            f"DXY 20d momentum {snapshot.dxy_20d_momentum_pct:+.2f}% (falling/flat).",
            (
                f"VIX {snapshot.vix_level:.1f} < {VIX_RISK_ON_MAX:.0f} and 5d ROC "
                f"{snapshot.vix_5d_roc_pct:+.2f}% (falling)."
            ),
            f"BTC dominance 20d trend {snapshot.btc_dominance_20d_trend_pct:+.2f}% (expanding).",
            f"BTC/QQQ 20d correlation {snapshot.btc_qqq_corr_20d:+.2f} (positive).",
        ]
        return MacroRegimeResult(
            state="Risk-On",
            trading_implication=_TRADING_IMPLICATIONS["Risk-On"],
            confidence=_clip01(risk_on_score),
            macro_weight=macro_weight,
            reasons=reasons,
            us10y_level_pct=snapshot.us10y_level_pct,
            us10y_20d_direction_pct=snapshot.us10y_20d_direction_pct,
            btc_qqq_corr_20d=snapshot.btc_qqq_corr_20d,
            is_macro_event_window=False,
        )

    if risk_off:
        reasons = [
            f"DXY 20d momentum {snapshot.dxy_20d_momentum_pct:+.2f}% (rising).",
            (
                f"VIX {snapshot.vix_level:.1f} (threshold {VIX_RISK_OFF_MIN:.0f}) / 5d ROC "
                f"{snapshot.vix_5d_roc_pct:+.2f}% (elevated or spiking)."
            ),
            (
                f"BTC/QQQ 20d correlation {snapshot.btc_qqq_corr_20d:+.2f} "
                f"(>= {risk_off_correlation_threshold:.2f}, compressing toward 1.0)."
            ),
        ]
        return MacroRegimeResult(
            state="Risk-Off",
            trading_implication=_TRADING_IMPLICATIONS["Risk-Off"],
            confidence=_clip01(risk_off_score),
            macro_weight=macro_weight,
            reasons=reasons,
            us10y_level_pct=snapshot.us10y_level_pct,
            us10y_20d_direction_pct=snapshot.us10y_20d_direction_pct,
            btc_qqq_corr_20d=snapshot.btc_qqq_corr_20d,
            is_macro_event_window=False,
        )

    reasons = [
        "Signals do not cleanly satisfy either Risk-On or Risk-Off -- mixed/no consensus.",
        f"(soft Risk-On score {risk_on_score:.2f}, soft Risk-Off score {risk_off_score:.2f})",
    ]
    return MacroRegimeResult(
        state="Transitional",
        trading_implication=_TRADING_IMPLICATIONS["Transitional"],
        confidence=_clip01(1.0 - max(risk_on_score, risk_off_score)),
        macro_weight=macro_weight,
        reasons=reasons,
        us10y_level_pct=snapshot.us10y_level_pct,
        us10y_20d_direction_pct=snapshot.us10y_20d_direction_pct,
        btc_qqq_corr_20d=snapshot.btc_qqq_corr_20d,
        is_macro_event_window=False,
    )


def _macro_weight(snapshot: MacroSignalSnapshot) -> float:
    """
    Adaptive doc Layer 3, exact rule: "When correlation is high (> 0.6),
    the macro signal is applied at full weight. When correlation is low
    (< 0.3), the macro signal weight is reduced proportionally... This is
    a dynamic weight, not a binary on/off." Corroborated by that document's
    own worked example: "BTC/QQQ corr: 0.64 -> macro weight: FULL". See
    module docstring's 2026-08-13 correction note for why this is
    implemented as a single linear ramp anchored at
    `CORRELATION_FULL_WEIGHT_THRESHOLD` (0.6) rather than a three-piece
    function -- the source text gives only that one precise breakpoint.

    Uses |correlation| (not signed correlation): relevance is about
    whether BTC and QQQ move together AT ALL, in either direction -- a
    strong NEGATIVE correlation is just as informative about macro linkage
    as a strong positive one, whereas near-zero correlation means BTC is
    decoupled and macro regime signals shouldn't be trusted much
    regardless of sign. This is deliberately independent of `confidence`
    and of which state was classified -- it answers a different question
    (see module docstring).
    """
    return _clip01(abs(snapshot.btc_qqq_corr_20d) / CORRELATION_FULL_WEIGHT_THRESHOLD)
