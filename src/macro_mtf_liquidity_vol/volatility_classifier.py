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
from a short raw (non-percentile-ranked) ATR sequence -- not accepted as a
pre-computed boolean. This is deliberate: the Done-when bar for this
module requires "all four states are correctly triggered by synthetic ATR
sequences, including the Spike condition specifically (doubling within 5
candles, not just a high percentile)" -- which calls for this module's own
tests to directly exercise the doubling arithmetic on synthetic sequences,
not trust an external flag it can't independently verify.

CORRECTION (2026-08-14 review): Spike detection previously compared ONLY
the two ENDPOINTS of the window (`recent_atr[-1] / recent_atr[-6]`),
i.e. it required 6 values but actually read 2. MASTER_CONTEXT's wording is
"ATR doubles WITHIN 5 candles" -- a doubling that happens and then partly
subsides inside the window satisfies that description, but was silently
missed. Demonstrated: the sequence [10, 10, 40, 20, 12, 11] (ATR
QUADRUPLED mid-window) returned spike_ratio=1.10 and classified
"Expanding" -- i.e. "standard position sizing" -- because 11/10 is close
to 1. That is a fail-OPEN error on a condition that is simultaneously a
Layer 17 kill-switch trigger, one of MASTER_CONTEXT constraint 8's six
entry gates, and the Risk Engine's 2.0x fat-tail multiplier, so a false
negative disables all three at once, contrary to this project's stated
capital-preservation-first priority order.

`spike_ratio` is now the maximum RUN-UP ratio anywhere within the window:
max over each position j of (ATR[j] / min(ATR[i] for i <= j)), i.e. the
largest factor by which ATR rose from any earlier point in the window to
any later one. This is a direct reading of "doubles within 5 candles" (it
asks whether such a doubling exists in the window at all) and is
order-aware -- a DECLINE from a high starting ATR is not a spike, so a
naive max/min ratio would be wrong. For an unchanged or monotonically
falling window it returns 1.0 (or lower), identical to the old formula's
behavior on flat input.

NOTE, deliberately not changed here: `feature_calculator.py`'s
`vol_spike` feature (line ~224) still uses the old endpoint-only formula.
That flag is a MODEL INPUT FEATURE, not the gating/kill-switch path this
module feeds (Risk Engine Layer 9 / Position Sizing Layer 10 consume this
classifier's categorical state), so the safety-critical path is the one
fixed here. Bringing `vol_spike` into line is a real follow-up, but it
belongs in its own scoped change -- it has its own test suite and its
output is percentile-ranked and consumed by the trained models, so
altering it silently changes model inputs mid-project.

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

import math
from dataclasses import dataclass
from typing import List, Sequence

# Exact MASTER_CONTEXT numbers -- hardcoded, not configurable (consistent
# with how this project treats other exact spec numbers elsewhere).
COMPRESSED_MAX_PERCENTILE = 0.30
EXTENDED_MIN_PERCENTILE = 0.70
SPIKE_LOOKBACK_CANDLES = 5
# "doubles" = ATR rose by >= this factor from any earlier point in the
# trailing window to any later one (see `_max_runup_ratio` and the module
# docstring's 2026-08-14 correction note -- this is NOT an endpoint-only
# comparison).
SPIKE_RATIO_THRESHOLD = 2.0

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
    spike_ratio: float  # max ATR run-up ratio within the trailing window (see _max_runup_ratio)
    reasons: List[str]

    def __post_init__(self) -> None:
        if self.state not in _TRADING_IMPLICATIONS:
            raise ValueError(f"invalid state {self.state!r}")
        if not math.isfinite(self.spike_ratio):
            # NaN previously slipped through: `nan < 0` is False, so the
            # non-negativity check below passed and a NaN spike_ratio was
            # accepted alongside a non-Spike state.
            raise ValueError(f"spike_ratio must be finite, got {self.spike_ratio}.")
        if self.trading_implication != _TRADING_IMPLICATIONS[self.state]:
            raise ValueError(
                f"trading_implication {self.trading_implication!r} does not match the exact "
                f"table text for state={self.state!r} ({_TRADING_IMPLICATIONS[self.state]!r})."
            )
        if not (0.0 <= self.atr_percentile <= 1.0):
            raise ValueError(f"atr_percentile must be in [0, 1], got {self.atr_percentile}.")
        if self.spike_ratio < 0:
            raise ValueError(f"spike_ratio must be non-negative, got {self.spike_ratio}.")


def _max_runup_ratio(window: Sequence[float]) -> float:
    """
    Largest factor by which ATR ROSE from any earlier point in `window` to
    any later one: max over j of (window[j] / min(window[i] for i <= j)).

    This is the "doubles WITHIN 5 candles" test (see the module docstring's
    2026-08-14 correction note). It is deliberately ORDER-AWARE rather than
    a plain max/min ratio: a window that starts high and falls (e.g.
    [40, 20, 10]) has max/min == 4.0 but contains no doubling at all -- ATR
    only ever declined, which is the opposite of a tail-risk spike. Tracking
    a RUNNING minimum ensures only genuine earlier-low-to-later-high moves
    count.

    FLOORED AT 1.0: the ratio only ever grows from its 1.0 starting point, so
    a flat window and a monotonically falling one BOTH return exactly 1.0 --
    "no run-up occurred." It never reports below 1.0; a decline is an absence
    of run-up, not a negative one. (An earlier draft of this docstring
    claimed "< 1.0 for a monotonically falling one", which the code never
    did -- corrected here.)

    Kept deliberately equivalent to `features.feature_calculator`'s
    vectorised `vol_spike` computation, which is this same formula expressed
    as shifted-series arithmetic over a whole column. That equivalence is
    enforced by a test rather than left to convention -- these two
    implementations silently diverging is precisely the bug this function was
    written to fix.
    """
    running_min = window[0]
    best = window[0] / window[0] if window[0] > 0 else 0.0
    for value in window[1:]:
        if running_min > 0:
            best = max(best, value / running_min)
        running_min = min(running_min, value)
    return best


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
        plus the prior 5. ALL 6 are examined (see `_max_runup_ratio`); the
        pre-2026-08-14 implementation required 6 but read only the two
        endpoints, which silently missed in-window spikes.

        Every value must be finite and positive. NaN is rejected rather
        than tolerated: a NaN ATR previously flowed straight through
        (`nan >= 2.0` is False, and the result dataclass's `spike_ratio < 0`
        check is False for NaN too), classifying missing volatility data as
        "Expanding" -- i.e. "standard position sizing" on absent data. This
        pipeline has already been observed producing all-NaN columns
        upstream (see docs/project_state.md's macro-fetcher entries), so
        this is a live failure mode, not a theoretical one.

    Returns
    -------
    VolatilityStateResult
    """
    if not (0.0 <= atr_percentile <= 1.0):
        # NB: this also rejects NaN, since every comparison against NaN is
        # False -- deliberate, and relied upon (there is no separate NaN
        # check for atr_percentile).
        raise ValueError(f"atr_percentile must be in [0, 1], got {atr_percentile}.")
    min_len = SPIKE_LOOKBACK_CANDLES + 1
    if len(recent_atr) < min_len:
        raise ValueError(
            f"recent_atr must have at least {min_len} values (current + "
            f"{SPIKE_LOOKBACK_CANDLES} candles back), got {len(recent_atr)}."
        )

    window = [float(v) for v in recent_atr[-min_len:]]
    for offset, value in enumerate(window):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"recent_atr values must be finite and positive (they are used as "
                f"divisors in the spike run-up ratio); got {value} at position "
                f"{offset - min_len} of the trailing {min_len}-value window."
            )

    current_atr = window[-1]
    spike_ratio = _max_runup_ratio(window)
    is_spike = spike_ratio >= SPIKE_RATIO_THRESHOLD

    reasons: List[str] = [
        (
            f"ATR percentile rank: {atr_percentile:.3f} "
            f"(Compressed < {COMPRESSED_MAX_PERCENTILE:.2f}, Extended > {EXTENDED_MIN_PERCENTILE:.2f})."
        ),
        (
            f"Max ATR run-up within the trailing {SPIKE_LOOKBACK_CANDLES}-candle window "
            f"(current ATR {current_atr:.4g}, window low {min(window):.4g}): "
            f"{spike_ratio:.3f}x (threshold {SPIKE_RATIO_THRESHOLD:.1f}x)."
        ),
    ]

    if is_spike:
        reasons.append(
            f"ATR run-up {spike_ratio:.3f}x >= {SPIKE_RATIO_THRESHOLD:.1f}x within "
            f"{SPIKE_LOOKBACK_CANDLES} candles -> Spike (overrides percentile band "
            f"regardless of atr_percentile={atr_percentile:.3f})."
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
