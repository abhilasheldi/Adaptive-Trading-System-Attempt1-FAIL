"""
Multi-Timeframe Alignment Engine (Layer 4) -- MASTER_CONTEXT's exact scoring
table:

    4/4 aligned            All timeframes pointing in the same direction
                            -> Full position size permitted
    3/4 aligned             One minor divergence, clear overall bias
                            -> Standard position size permitted
    2/4 aligned             Mixed structure, no clear directional case
                            -> No entry
    1/4 or 0/4              Severe structural conflict
                            -> No entry, existing positions reviewed

Per the Adaptive Probabilistic Trade Management System spec (Layer 4
section, checked directly rather than assumed): "The Layer 2 feature
vector is evaluated at each timeframe (1H, 4H, 1D, 1W) independently.
Directional alignment is scored" -- i.e. this module's job is to SCORE
alignment across four ALREADY-INDEPENDENTLY-DERIVED per-timeframe
directional reads, not to compute those reads from raw OHLCV/feature
vectors itself. Deriving a single directional call from a full Layer 2
feature vector at one timeframe is a separate, real design question this
module deliberately does not answer -- exactly the same "classifier
reading current signals, not a forecaster" split used in macro_regime.py
for Layer 3 (it reads DXY/VIX/etc. readings, it doesn't compute them).

IMPORTANT, checked directly against source docs (2026-08-13): the four
timeframes this table assumes -- 1H, 4H, 1D, 1W -- are NOT all currently
produced anywhere in this codebase. `config.yaml`'s `crypto.timeframes` is
`["1h", "4h"]` only, and the Adaptive doc itself states "MVP scope (Phase 1
only): BTC and ETH OHLCV at 1H and 4H. Everything else is added in Phase
2" -- 1D/1W data ingestion (Layer 1) and feature computation (Layer 2) for
those timeframes do not exist yet. This module is built correctly and
completely against the spec regardless (Phase 2 work, same as Layer 3),
but a real caller cannot supply genuine 1D/1W directional reads until that
ingestion exists. `score_mtf_alignment` validates that exactly 4 readings
are supplied and raises rather than silently accepting fewer -- it will
not pretend 2 timeframes is an acceptable substitute for the spec's 4.

"A trade is only considered when alignment is 3/4 or 4/4" (Adaptive doc,
Layer 4) -- exposed as the `trade_permitted` convenience field, matching
MASTER_CONTEXT constraint 8's six-gate requirement "MTF alignment >= 3/4".
"""

import math
from dataclasses import dataclass
from typing import Dict, List

# Exact MASTER_CONTEXT/Adaptive-doc table text -- not paraphrased.
_ACTIONS = {
    4: "Full position size permitted",
    3: "Standard position size permitted",
    2: "No entry",
    1: "No entry, existing positions reviewed",
    0: "No entry, existing positions reviewed",
}

# Exact MASTER_CONTEXT number (constraint 8's six-gate requirement, "MTF
# alignment >= 3/4"; also stated directly in the Adaptive doc's Layer 4
# section: "A trade is only considered when alignment is 3/4 or 4/4").
# 2026-08-13 re-audit finding: this was previously only a bare literal `3`
# inline in two places (the dataclass validation and the return statement)
# -- every other exact spec number in this module's three sibling modules
# (macro_regime.py's VIX_RISK_ON_MAX etc., volatility_classifier.py's
# COMPRESSED_MAX_PERCENTILE etc., liquidity_trap.py's EQUAL_LEVEL_LOOKBACK)
# is a named, top-level, directly-testable constant -- this was the one
# inconsistent exception, fixed here to match.
TRADE_PERMITTED_MIN_SCORE = 3

# The canonical four timeframes per the Adaptive doc's Layer 4 section.
# NOT enforced as required dict keys (a caller could reasonably supply a
# different four-timeframe set for a different asset/strategy variant) --
# documented here as the intended/spec default, with only the COUNT (must
# be exactly 4) actually validated. See module docstring for why 1D/1W
# aren't available from this codebase's current data pipeline.
CANONICAL_TIMEFRAMES = ("1H", "4H", "1D", "1W")

# Judgment call, NOT a MASTER_CONTEXT number: neither MASTER_CONTEXT nor
# the Adaptive doc quantifies what counts as "neutral" vs. a real
# directional read at a given timeframe. A per-timeframe directional score
# with |value| below this band is treated as non-committal (counts toward
# neither bullish nor bearish alignment) rather than as weak evidence for
# whichever side it barely leans toward.
DEFAULT_NEUTRAL_BAND = 0.15


@dataclass(frozen=True)
class MTFAlignmentResult:
    """
    `score` and `action` are the two required outputs ("Output both the
    numeric score and the resulting action label"). `trade_permitted`,
    `dominant_direction`, `confidence`, `per_timeframe_direction`, and
    `reasons` exist for the same reason macro_regime.py's result carries
    more than the bare minimum: this feeds Layer 7C's Ensemble Meta-Model
    and Layer 8's Explainability Engine downstream, both of which need more
    than a bare number to do their job.
    """

    score: int  # 0-4, count of timeframes agreeing with the dominant direction
    action: str  # exact table text for `score`
    trade_permitted: bool  # True iff score >= TRADE_PERMITTED_MIN_SCORE (3) -- "a trade is only considered when alignment is 3/4 or 4/4"
    dominant_direction: str  # "bullish" | "bearish" | "mixed" (tied, no majority) | "neutral" (all four non-committal)
    # [0, 1] -- NET conviction toward `dominant_direction`, averaged over
    # every non-neutral timeframe, so dissent reduces it (see
    # `score_mtf_alignment`'s 2026-08-14 note). For a tie ("mixed") it is
    # instead the symmetric mean |conviction| across both conflicting
    # sides -- there is no dominant direction to be net toward.
    confidence: float
    per_timeframe_direction: Dict[str, str]  # each input timeframe's classified "bullish"/"bearish"/"neutral"
    # No default (2026-08-13 re-audit finding): the three sibling modules'
    # analogous `reasons` fields are all required, no-default fields --
    # this one previously had `= field(default_factory=list)`, an
    # unexplained, unused-by-this-module's-own-logic inconsistency
    # (`score_mtf_alignment` always explicitly supplies `reasons` itself;
    # the default only mattered for tests directly constructing this
    # dataclass, which is more consistent to do explicitly anyway).
    reasons: List[str]

    def __post_init__(self) -> None:
        if self.score not in _ACTIONS:
            raise ValueError(f"score must be 0-4, got {self.score}.")
        if self.action != _ACTIONS[self.score]:
            raise ValueError(
                f"action {self.action!r} does not match the exact table text for "
                f"score={self.score} ({_ACTIONS[self.score]!r})."
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}.")
        expected_permitted = self.score >= TRADE_PERMITTED_MIN_SCORE
        if self.trade_permitted != expected_permitted:
            raise ValueError(
                f"trade_permitted={self.trade_permitted} inconsistent with score={self.score} "
                f"(must be True iff score >= {TRADE_PERMITTED_MIN_SCORE})."
            )


def _classify_timeframe(value: float, neutral_band: float) -> str:
    if value > neutral_band:
        return "bullish"
    if value < -neutral_band:
        return "bearish"
    return "neutral"


def score_mtf_alignment(
    timeframe_directions: Dict[str, float],
    *,
    neutral_band: float = DEFAULT_NEUTRAL_BAND,
) -> MTFAlignmentResult:
    """
    Score multi-timeframe structural alignment across exactly four
    timeframes' already-derived directional readings.

    Parameters
    ----------
    timeframe_directions : Dict[str, float]
        Mapping of timeframe label (e.g. "1H", "4H", "1D", "1W" -- see
        `CANONICAL_TIMEFRAMES`) to a signed directional reading in
        [-1, 1]: positive = bullish conviction, negative = bearish
        conviction, magnitude = strength. MUST contain exactly 4 entries --
        the table this function implements is inherently a 4-point scale
        (see module docstring on why fewer isn't accepted as a substitute).
        How a caller derives this signed value from a Layer 2 feature
        vector at one timeframe (e.g. from struct_bos_bullish/bearish,
        struct_trend_state, momentum features, etc.) is deliberately out of
        scope for this function -- it scores agreement across readings, it
        does not produce them (see module docstring).
    neutral_band : float
        Judgment-call threshold (NOT a MASTER_CONTEXT number) below which
        a timeframe's |directional reading| is treated as non-committal
        rather than weak evidence either way.

    Returns
    -------
    MTFAlignmentResult
    """
    # 2026-08-14 review: `neutral_band` was entirely unvalidated, and it
    # governs how every reading is interpreted. A NEGATIVE band inverted the
    # classifier's meaning -- with neutral_band=-0.5, four readings of
    # exactly 0.0 (no directional information whatsoever) classified as a
    # unanimous 4/4 BULLISH read with `trade_permitted=True`, i.e. "Full
    # position size permitted", because `0.0 > -0.5`. A band >= 1.0 fails
    # the other way, silently collapsing every possible reading to neutral
    # and returning score=0 forever.
    if not math.isfinite(neutral_band) or not (0.0 <= neutral_band < 1.0):
        raise ValueError(
            f"neutral_band must be a finite value in [0, 1), got {neutral_band}. "
            f"Negative values invert the classifier (a zero reading would count as "
            f"directional); values >= 1 make every reading neutral, since inputs "
            f"are constrained to [-1, 1]."
        )
    if len(timeframe_directions) != 4:
        raise ValueError(
            f"score_mtf_alignment requires exactly 4 timeframe readings "
            f"(MASTER_CONTEXT's table is a 4-point scale), got "
            f"{len(timeframe_directions)}: {sorted(timeframe_directions.keys())}. "
            f"Canonical set per the Adaptive doc's Layer 4 section is "
            f"{CANONICAL_TIMEFRAMES}; 1D/1W data ingestion does not exist "
            f"in this codebase yet -- see module docstring."
        )
    for tf, value in timeframe_directions.items():
        if not (-1.0 <= value <= 1.0):
            raise ValueError(
                f"timeframe_directions[{tf!r}] = {value} out of range; must be in [-1, 1]."
            )

    per_timeframe_direction = {
        tf: _classify_timeframe(v, neutral_band) for tf, v in timeframe_directions.items()
    }

    bullish_tfs = [tf for tf, d in per_timeframe_direction.items() if d == "bullish"]
    bearish_tfs = [tf for tf, d in per_timeframe_direction.items() if d == "bearish"]
    n_bullish, n_bearish = len(bullish_tfs), len(bearish_tfs)

    if n_bullish == 0 and n_bearish == 0:
        dominant_direction = "neutral"
        score = 0
        aligned_tfs: List[str] = []
    elif n_bullish == n_bearish:
        # Exact tie between the two non-neutral directions (e.g. 2 vs 2, or
        # 1 vs 1 with the rest neutral): no majority exists. This is
        # precisely "mixed structure, no clear directional case" when tied
        # at 2-2 (score lands at 2/4 below); a 1-1 tie with 2 neutral also
        # lands at score=1, deliberately treated with the SAME conservative
        # "no entry, review existing positions" action as a genuine 4-way
        # structural fight -- MASTER_CONTEXT's capital-preservation-first
        # philosophy favors caution when the picture is unclear, even
        # though "severe structural conflict" is a narrower description
        # than "mostly inconclusive." See module docstring.
        dominant_direction = "mixed"
        score = n_bullish
        # BUGFIX (2026-08-13 re-audit): using only `bullish_tfs` here made
        # `confidence` depend on which arbitrary side happened to be
        # labeled "bullish", not on how strongly mixed the picture actually
        # is. Concretely: {1H:+0.9, 4H:+0.9, 1D:-0.16, 1W:-0.16} (a strong
        # bullish pair tied against a barely-non-neutral bearish pair)
        # reported confidence=0.9, while the mirrored scenario
        # {1H:+0.16, 4H:+0.16, 1D:-0.9, 1W:-0.9} -- structurally identical,
        # just with the labels swapped -- reported confidence=0.16. Two
        # readings that are the same conflict from opposite sides must not
        # get different confidence scores. Using BOTH conflicting sides
        # makes this symmetric: both scenarios now report the same value.
        aligned_tfs = bullish_tfs + bearish_tfs
    elif n_bullish > n_bearish:
        dominant_direction = "bullish"
        score = n_bullish
        aligned_tfs = bullish_tfs + bearish_tfs
    else:
        dominant_direction = "bearish"
        score = n_bearish
        aligned_tfs = bullish_tfs + bearish_tfs

    # 2026-08-14 review -- completing the 2026-08-13 fix above. That fix
    # made the TIE branch symmetric but left the bullish/bearish branches
    # averaging over the winning side ALONE, which discarded the magnitude
    # of dissent entirely: {+0.2, +0.2, +0.2, -1.0} and
    # {+0.2, +0.2, +0.2, -0.16} both reported score=3 AND confidence=0.20,
    # so a violent opposing 1W read was indistinguishable from near-
    # consensus, and both cleared MASTER_CONTEXT constraint 8's MTF gate at
    # "Standard position size permitted."
    #
    # Confidence for a real dominant direction is now the NET conviction
    # toward it, averaged over every non-neutral timeframe: each dissenting
    # timeframe subtracts its own conviction. Clipped at 0, since a
    # direction that wins on COUNT can still be outweighed on conviction
    # (as in the -1.0 case above), and a negative confidence is not
    # meaningful -- the score already carries the count.
    #
    # Unanimous readings are unchanged by this (no dissent to subtract), so
    # the existing conviction-gradation behavior is preserved exactly.
    # Neutral timeframes are excluded from the denominator rather than
    # counted as zeros, matching the pre-existing convention that a
    # non-committal timeframe is an absence of evidence, not evidence of
    # weakness.
    if not aligned_tfs:
        confidence = 0.0
    elif dominant_direction == "mixed":
        # A tie has no dominant direction to be "net toward" -- keep the
        # symmetric mean-|conviction| reading of how strong the conflict is.
        confidence = sum(abs(timeframe_directions[tf]) for tf in aligned_tfs) / len(aligned_tfs)
    else:
        sign = 1.0 if dominant_direction == "bullish" else -1.0
        net = sum(timeframe_directions[tf] * sign for tf in aligned_tfs) / len(aligned_tfs)
        confidence = max(0.0, net)

    action = _ACTIONS[score]
    reasons = [
        f"{tf}: {timeframe_directions[tf]:+.2f} ({per_timeframe_direction[tf]})"
        for tf in timeframe_directions
    ]
    reasons.append(
        f"{score}/4 aligned on '{dominant_direction}' -> {action}"
        if dominant_direction not in ("mixed", "neutral")
        else f"{score}/4 -- no clear dominant direction ({dominant_direction}) -> {action}"
    )

    return MTFAlignmentResult(
        score=score,
        action=action,
        trade_permitted=score >= TRADE_PERMITTED_MIN_SCORE,
        dominant_direction=dominant_direction,
        confidence=confidence,
        per_timeframe_direction=per_timeframe_direction,
        reasons=reasons,
    )
