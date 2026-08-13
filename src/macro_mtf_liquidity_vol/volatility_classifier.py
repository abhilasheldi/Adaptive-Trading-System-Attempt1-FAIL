"""
Volatility Regime Classifier (Layer 6) -- MASTER_CONTEXT's exact table:

    Compressed:  ATR < 30th percentile   -> Pre-breakout; reduce size until
                                             directional confirmation
    Expanding:   30th-70th percentile    -> Active trend initiation;
                                             standard position sizing
    Extended:    ATR > 70th percentile   -> Mature trend/blow-off; reduce
                                             size, tighten stops aggressively
    Spike:       ATR doubles within 5 candles -> Tail-risk event; no new
                                             entries across all assets

"Derived from the ATR percentile and Bollinger width features already
computed in Layer 2" (build instruction) -- this is a THIN classifier, not
a re-derivation. `feature_calculator.py`'s `_calculate_volatility_features`
already computes `vol_atr_pct_rank` (the trailing-252-candle rolling
percentile rank) and, in fact, already computes boolean flags for all four
of these exact states (`vol_state_compressed`/`vol_state_expanding`/
`vol_state_extended`/`vol_spike`) using these exact thresholds. This module
exists as a NAMED, testable classifier because the Adaptive doc's Layer 6
section explains why: it's "kept as a named output because it feeds the
Risk Engine (Layer 9) and Position Sizing Engine (Layer 10) directly with a
categorical label rather than a raw percentile" -- i.e. downstream layers
want one authoritative state string + trading implication, not four
independent booleans they'd each have to reconcile themselves.

Bollinger Band width is mentioned in the build instruction and is one of
the features Layer 2 computes, but MASTER_CONTEXT's own classification
table uses ONLY the ATR percentile as the discriminating variable for the
three percentile-based states -- Bollinger width isn't part of the actual
classification rule anywhere in the source docs. Not used here for the
same reason US10Y wasn't used in macro_regime.py's classification: it's
present as an input Layer 2 computes, but no source doc gives it a role in
THIS specific decision rule, so it isn't force-fit into one.

Percentile band vs. Spike input design, deliberately asymmetric: the
percentile band (Compressed/Expanding/Extended) is driven by
`atr_percentile`, accepted here as an ALREADY-COMPUTED scalar (Layer 2's
job -- recomputing a trailing-252-candle rolling rank in this module would
duplicate that logic and force this classifier to carry a long history it
doesn't otherwise need). Spike is different: it's computed HERE, directly,
from a short raw (non-percentile-ranked) ATR sequence, using the exact
same formula `feature_calculator.py`'s `vol_spike` flag already uses
(current ATR / ATR-5-candles-ago >= 2.0) -- not accepted as a pre-computed
boolean. This is deliberate: the Done-when bar for this module requires
"all four states are correctly triggered by synthetic ATR sequences,
including the Spike condition specifically (doubling within 5 candles,
not just a high percentile)" -- which calls for this module's own tests to
directly exercise the doubling arithmetic on synthetic sequences, not
trust an external flag it can't independently verify.

Precedence (a design decision, since MASTER_CONTEXT's table doesn't state
one explicitly -- the same kind of gap `macro_regime.py`'s Macro-Event
Window precedence and `mtf_alignment.py`'s tie-handling both had to fill):
Spike is checked FIRST and overrides whatever the percentile band would
otherwise say. Rationale, mirroring Macro-Event Window's precedence in
Layer 3: "tail-risk event; no new entries across all assets" is the single
most conservative response in this table, and a genuine 5-candle ATR
doubling can occur before it has meaningfully shifted a slow-moving
252-candle percentile rank (e.g. a low-percentile/Compressed regime that
has JUST begun spiking) -- in that exact scenario, MASTER_CONTEXT's intent
is clearly the tail-risk warning, not "still technically compressed."

One wording discrepancy noted, not acted on (same treatment as the
Layer 3/5 wording discrepancies found this session): the Adaptive doc's
Compressed action text adds "...then scale up" after "reduce size until
directional confirmation" -- MASTER_CONTEXT.md's shorter text is used
verbatim in `_TRADING_IMPLICATIONS`, since that file is this project's
explicitly-labeled "exact table" reference and has been authoritative
throughout this project.
"""

from dataclasses import dataclass
from typing import List, Sequence

# Exact MASTER_CONTEXT numbers -- hardcoded, not configurable (consistent
# with how this project treats other exact spec numbers elsewhere).
COMPRESSED_MAX_PERCENTILE = 0.30
EXTENDED_MIN_PERCENTILE = 0.70
SPIKE_LOOKBACK_CANDLES = 5
SPIKE_RATIO_THRESHOLD = 2.0  # "doubles" = current ATR / ATR-5-candles-ago >= 2.0

_TRADING_IMPLICATIONS = {
    "Compressed": "Pre-breakout; reduce size until directional confirmation",
    "Expanding": "Active trend initiation; standard position sizing",
    "Extended": "Mature trend/blow-off; reduce size, tighten stops aggressively",
    "Spike": "Tail-risk event; no new entries across all assets",
}


@dataclass(frozen=True)
class VolatilityStateResult:
    """`state` and `trading_implication` are MASTER_CONTEXT's exact table
    text (the required "categorical state label" output). `spike_ratio` is
    always reported (not just when Spike triggers) so a caller can see how
    close conditions are to the threshold even in a non-Spike read."""

    state: str  # "Compressed" | "Expanding" | "Extended" | "Spike"
    trading_implication: str
    atr_percentile: float  # echoed input, for downstream/diagnostic visibility
    spike_ratio: float  # current_atr / atr_5_candles_ago
    reasons: List[str]

    def __post_init__(self) -> None:
        if self.state not in _TRADING_IMPLICATIONS:
            raise ValueError(f"invalid state {self.state!r}")
        if self.trading_implication != _TRADING_IMPLICATIONS[self.state]:
            raise ValueError(
                f"trading_implication {self.trading_implication!r} does not match the exact "
                f"table text for state={self.state!r} ({_TRADING_IMPLICATIONS[self.state]!r})."
            )
        if not (0.0 <= self.atr_percentile <= 1.0):
            raise ValueError(f"atr_percentile must be in [0, 1], got {self.atr_percentile}.")
        if self.spike_ratio < 0:
            raise ValueError(f"spike_ratio must be non-negative, got {self.spike_ratio}.")


def classify_volatility_state(
    atr_percentile: float,
    recent_atr: Sequence[float],
) -> VolatilityStateResult:
    """
    Classify the current volatility regime.

    Parameters
    ----------
    atr_percentile : float
        Already-computed Layer 2 ATR percentile rank
        (`feature_calculator.vol_atr_pct_rank`), in [0, 1]. See module
        docstring for why this is accepted pre-computed rather than
        re-derived from a raw ATR history here.
    recent_atr : Sequence[float]
        Raw (NOT percentile-ranked) ATR values, chronologically ordered,
        ending at the current candle. Must contain at least
        `SPIKE_LOOKBACK_CANDLES + 1` (6) values -- the current candle's ATR
        plus the prior 5 -- so `recent_atr[-1]` is "now" and
        `recent_atr[-1 - SPIKE_LOOKBACK_CANDLES]` is "5 candles ago."

    Returns
    -------
    VolatilityStateResult
    """
    if not (0.0 <= atr_percentile <= 1.0):
        raise ValueError(f"atr_percentile must be in [0, 1], got {atr_percentile}.")
    min_len = SPIKE_LOOKBACK_CANDLES + 1
    if len(recent_atr) < min_len:
        raise ValueError(
            f"recent_atr must have at least {min_len} values (current + "
            f"{SPIKE_LOOKBACK_CANDLES} candles back), got {len(recent_atr)}."
        )

    current_atr = float(recent_atr[-1])
    atr_5_ago = float(recent_atr[-1 - SPIKE_LOOKBACK_CANDLES])
    if current_atr < 0 or atr_5_ago <= 0:
        raise ValueError(
            f"ATR values must be positive (atr_5_candles_ago used as a divisor); got "
            f"current_atr={current_atr}, atr_5_candles_ago={atr_5_ago}."
        )

    spike_ratio = current_atr / atr_5_ago
    is_spike = spike_ratio >= SPIKE_RATIO_THRESHOLD

    reasons: List[str] = [
        (
            f"ATR percentile rank: {atr_percentile:.3f} "
            f"(Compressed < {COMPRESSED_MAX_PERCENTILE:.2f}, Extended > {EXTENDED_MIN_PERCENTILE:.2f})."
        ),
        (
            f"Spike ratio (current ATR {current_atr:.4g} / ATR {SPIKE_LOOKBACK_CANDLES} candles ago "
            f"{atr_5_ago:.4g}): {spike_ratio:.3f} (threshold {SPIKE_RATIO_THRESHOLD:.1f})."
        ),
    ]

    if is_spike:
        reasons.append(
            f"Spike ratio {spike_ratio:.3f} >= {SPIKE_RATIO_THRESHOLD:.1f} -> Spike "
            f"(overrides percentile band regardless of atr_percentile={atr_percentile:.3f})."
        )
        state = "Spike"
    elif atr_percentile < COMPRESSED_MAX_PERCENTILE:
        state = "Compressed"
    elif atr_percentile > EXTENDED_MIN_PERCENTILE:
        state = "Extended"
    else:
        state = "Expanding"

    return VolatilityStateResult(
        state=state,
        trading_implication=_TRADING_IMPLICATIONS[state],
        atr_percentile=atr_percentile,
        spike_ratio=spike_ratio,
        reasons=reasons,
    )
