"""
Liquidity Trap heuristic (Layer 5) -- a heuristic probability estimator for
stop-hunt/liquidity-sweep setups, based on equal highs/lows and wick
behavior (the Prompt Playbook's exact scope for this build: "a heuristic
probability estimator for stop-hunt/liquidity-sweep setups based on equal
highs/lows and wick behavior").

CRITICAL, load-bearing limitation -- stated explicitly per the build
instruction, and worth restating in the strongest terms this module has:
this is a HEURISTIC estimator, NOT a measurement of real order-book depth.
Retail systems (this one included) do not have access to true historical
order-book/depth-of-market data at the granularity this kind of analysis
would actually require. Every "cluster," "level," and "probability" this
module produces is inferred from CANDLE WICKS in OHLCV data -- a proxy for
where stop orders might cluster, not a measurement of where they actually
are. The Adaptive Probabilistic Trade Management System spec (Layer 5
section) states this exact risk: "this layer operates as a heuristic
probability estimator, not a direct order book measurement... All outputs
are expressed as probability estimates, not precise physical levels.
Overstating the precision of this layer is a common failure mode in
systems using this approach." Every numeric output of this module is
built to honor that: `trap_probability` and `liquidity_attraction_score`
are dimensionless [0, 1] heuristic scores, and cluster "levels" are always
paired with a coarse, ATR-normalized DISTANCE estimate rather than being
presented as a precise, actionable price -- see `LiquidityCluster`.

Scope boundary (a deliberate choice, mirroring how `mtf_alignment.py`
scoped itself to its own Prompt Playbook instruction rather than the
fuller Adaptive-doc Layer 5 description): the Adaptive doc's Layer 5
section additionally describes Fair Value Gap (FVG) detection, "liquidity
void" detection, and "previous session high/low" tracking as things the
full Layer 5 "Detects." NONE of those are built here. The Prompt
Playbook's actual build instruction and "Done when" bar for this specific
task scope it to equal-highs/lows + wick behavior only -- this module
implements exactly that, rather than shipping a partial, unverified
implementation of everything the fuller design doc eventually envisions.

Sweep definition, taken directly from the spec's own hypothesis section
(not invented here): "Liquidity sweeps (stop-runs above equal highs or
below equal lows, followed by a confirmed reversal) are specific,
programmatically detectable events." A "stop cluster" is "two or more
wicks testing the same level without closing through it" (Adaptive doc,
Layer 5), within a trailing 50-candle window -- the one EXACT number this
layer's spec gives, used here as `EQUAL_LEVEL_LOOKBACK = 50`.

This module operates on raw OHLCV directly (unlike macro_regime.py and
mtf_alignment.py, which are classifiers reading already-derived signals):
detecting equal-high/low clusters and wick rejections IS the raw
structural analysis Layer 5 exists to do, there is no upstream module that
already computes it. `feature_calculator.py`'s `struct_sfp_bullish`/
`struct_sfp_bearish` are a related but genuinely different, simpler
concept (a single rolling N=20 extremum, not a multi-touch equal-level
cluster) -- this module is not a duplicate of those features.

No-lookahead discipline: cluster detection for a given candle is always
built from the window STRICTLY BEFORE that candle (mirroring
`feature_calculator.py`'s own prior-structure-then-current-candle-tests-it
pattern for BOS/SFP), so a candle can never contribute to "sweeping" a
cluster it helped form.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Exact spec number (Adaptive doc, Layer 5): "Equal highs and equal lows
# within the trailing 50 candles". Hardcoded default, not a magic number
# invented here -- overridable only because a caller might reasonably want
# to experiment, not because the spec leaves it open.
EQUAL_LEVEL_LOOKBACK = 50

# Everything below is a judgment call -- the spec calls this whole layer a
# "heuristic estimate" and gives no formula for combining proximity/density
# into a probability, and no numeric tolerance for what counts as "equal."
# NOT MASTER_CONTEXT/Adaptive-doc exact numbers; tune freely.
# B.4 RETUNE (2026-08-18). Was 0.25. Selected by sweeping {0.05, 0.10, 0.15}
# against a confirmation-rate band of 1-5% FIXED IN ADVANCE, with the
# compression precondition (atr_percentile <= 0.20) applied. Measured on the
# frozen v2 BTC 1h series, 78,378 modeling rows:
#
#   tol   pen    rate (all bars)   rate (compressed)   events (compressed)
#   0.05  0.25        2.94%              4.03%  <- selected      815
#   0.10  0.25        6.48%              8.65%                 1,747
#   0.15  0.25        8.61%             10.73%                 2,167
#   0.25  0.25       10.93%             13.04%                 2,634
#
# Exactly one configuration landed inside the pre-registered band. Selection
# was on confirmation rate alone -- no downstream metric was computed by
# `scripts/measure_b4_sweep_retune.py`, so the event definition cannot have
# been tuned to the outcome.
DEFAULT_LEVEL_TOLERANCE_ATR = 0.05  # wicks within this many ATRs of each other count as "equal"
# Sweep-selectivity thresholds (2026-08-14). Neither is a source-doc number;
# both exist because breach-and-close-back alone fired on 44.1% of real BTC
# 1h bars, which cannot be the "specific, programmatically detectable event"
# the spec describes. See the confirmation block in `estimate_liquidity_trap`
# for the full measurement and reasoning.
# DECOUPLED FROM THE TOLERANCE AT B.4 (2026-08-18). Previously set EQUAL to
# `DEFAULT_LEVEL_TOLERANCE_ATR` on the grounds that a cluster's level is only
# defined to within +/- that tolerance, so a "penetration" smaller than it
# sits inside the level's own uncertainty band and cannot be distinguished
# from touching the level.
#
# That rationale gives a LOWER BOUND -- penetration >= tolerance -- not an
# equality, and the two parameters mean different things: tolerance decides
# which wicks form one level, penetration decides how far price must clear it.
# Holding penetration at 0.25 while tightening tolerance to 0.05 satisfies the
# bound with room to spare and is strictly MORE conservative than the old
# equal setting.
#
# The distinction is not academic. Sweeping the two coupled together is
# NON-MONOTONIC, because the effects fight: a tighter tolerance forms fewer
# and smaller clusters (fewer sweeps) but simultaneously weakens the
# penetration requirement (more sweeps). Measured, compressed-bar rate:
#
#   coupled   0.25 -> 13.04%,  0.15 -> 17.27%,  0.10 -> 16.13%,  0.05 -> 9.84%
#
# Tightening the tolerance from 0.25 to 0.15 RAISES the rate. No coupled
# setting reaches the 1-5% band at all; only the decoupled sweep does.
# `_assert_penetration_at_least_tolerance` enforces the bound that survives.
DEFAULT_MIN_PENETRATION_ATR = 0.25
# Median cluster on real BTC 1h has 3 touches; requiring more than the median
# restricts sweeps to the better-tested pools. A CLUSTER is still "two or
# more wicks" per spec -- this is a separate, stricter bar for what counts as
# a sweep worth acting on.
DEFAULT_MIN_SWEEP_TOUCHES = 4
#
# NOT ADDED, deliberately -- a "rejection wick must be >= X% of the candle's
# range" test was implemented and then removed on 2026-08-14 because it is
# PROVABLY REDUNDANT here: confirmation already requires
# `close < swept_level < high`, which forces
# (high - close)/(high - low) >= (high - level)/(high - low) > 0. Measured on
# real BTC 1h it moved the sweep rate from 44.12% to 44.35% -- i.e. it never
# bound. Keeping a parameter that cannot bind would misrepresent how
# selective this layer is. Do not re-add it without a metric that is
# independent of the close-back-inside condition.
DEFAULT_PROXIMITY_SCALE_ATR = 2.0  # distance (in ATR) at which proximity contribution decays to ~0
DEFAULT_DENSITY_SATURATION_TOUCHES = 4  # touch count at which density contribution saturates at 1.0


@dataclass(frozen=True)
class LiquidityCluster:
    """
    One detected stop-cluster: 2+ prior candles' wicks testing a similar
    price level without closing through it.

    `level` is retained for internal distance calculations and debugging
    ONLY -- per the module's core limitation, no caller should treat it as
    an actionable, precise price. `distance_atr` (distance from the
    reference close, in ATR units) is the number meant for actual use:
    coarse, dimensionless, and explicit about being an estimate.
    """

    side: str  # "resistance" (equal highs, at/above reference price) | "support" (equal lows, at/below)
    level: float  # heuristic representative price (mean of touching wicks) -- NOT a precision claim, see module docstring
    touch_count: int
    distance_atr: float  # |level - reference_close| / atr -- the number to actually use downstream

    def __post_init__(self) -> None:
        # 2026-08-14 review: this was the only result dataclass across all
        # four macro_mtf_liquidity_vol modules with no validation at all --
        # `LiquidityCluster(side="banana", level=..., touch_count=-5,
        # distance_atr=-1.0)` constructed silently. Every sibling
        # (MacroRegimeResult, MTFAlignmentResult, VolatilityStateResult,
        # LiquidityTrapResult) validates its own invariants in __post_init__.
        if self.side not in ("resistance", "support"):
            raise ValueError(f"side must be 'resistance' or 'support', got {self.side!r}.")
        if not math.isfinite(self.level):
            raise ValueError(f"level must be finite, got {self.level}.")
        if self.touch_count < 2:
            raise ValueError(
                f"touch_count must be >= 2 -- a 'cluster' is by definition 'two or more "
                f"wicks testing the same level' (Adaptive doc, Layer 5); got {self.touch_count}."
            )
        if not math.isfinite(self.distance_atr) or self.distance_atr < 0:
            raise ValueError(
                f"distance_atr must be a finite, non-negative magnitude, got {self.distance_atr}."
            )


@dataclass(frozen=True)
class LiquidityTrapResult:
    """
    `trap_probability` is the required core output (Done-when bar: "output
    is a probability (0-1), not a claimed exact price level"). Everything
    else is supporting context for the same "heuristic probability
    estimator" the module docstring commits to -- none of it is a precise
    physical level either.
    """

    trap_probability: float  # [0, 1] -- REQUIRED core output
    liquidity_attraction_score: float  # [0, 1] -- proximity+density of pools on EITHER side of current price
    sweep_just_confirmed: bool
    sweep_direction: Optional[str]  # "bullish" | "bearish" | None (None unless sweep_just_confirmed)
    nearest_resistance_cluster: Optional[LiquidityCluster]
    nearest_support_cluster: Optional[LiquidityCluster]
    reasons: List[str]

    def __post_init__(self) -> None:
        if not (0.0 <= self.trap_probability <= 1.0):
            raise ValueError(f"trap_probability must be in [0, 1], got {self.trap_probability}.")
        if not (0.0 <= self.liquidity_attraction_score <= 1.0):
            raise ValueError(
                f"liquidity_attraction_score must be in [0, 1], got {self.liquidity_attraction_score}."
            )
        if self.sweep_direction is not None and not self.sweep_just_confirmed:
            raise ValueError("sweep_direction set without sweep_just_confirmed=True.")
        # 2026-08-14 review: the converse was NOT checked, so
        # `sweep_just_confirmed=True, sweep_direction=None` -- "a sweep
        # confirmed, but in no direction" -- constructed silently. Both
        # halves of the invariant are now enforced, matching how
        # `MacroRegimeResult` validates `is_macro_event_window` against
        # `state` in both directions.
        if self.sweep_just_confirmed and self.sweep_direction is None:
            raise ValueError(
                "sweep_just_confirmed=True requires a sweep_direction; a confirmed "
                "sweep with no direction is not a meaningful state."
            )
        if self.sweep_direction not in (None, "bullish", "bearish"):
            raise ValueError(f"invalid sweep_direction {self.sweep_direction!r}")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _cluster_levels(
    values: np.ndarray,
    closes: np.ndarray,
    side: str,
    atr: float,
    tolerance_atr: float,
) -> List[LiquidityCluster]:
    """
    Two-stage detection of "equal level" clusters (highs for
    side='resistance', lows for side='support'): (1) group wick extremes
    by proximity alone, (2) within each proximity group, count only the
    members that actually respect that group's shared level as genuine
    "touches" -- "two or more wicks testing the SAME LEVEL without closing
    through it" (Adaptive doc). A cluster with fewer than 2 such touches
    after filtering is discarded.

    `values`/`closes` cover only the reference window (already excludes
    the current candle by the caller -- see module docstring's
    no-lookahead note). Returns clusters sorted by touch_count descending
    (strongest/most-tested first).

    TWO BUGS fixed here, both found by actually running this against
    synthetic data before shipping, not by re-reading the code:

    1. Proximity grouping must anchor each cluster to the lowest unclaimed
    value and bound every member's distance to THAT anchor -- comparing
    each sorted value only to its immediate predecessor (a "chain") lets a
    cluster's total span drift without bound (e.g. 100.0, 100.2, 100.4,
    ... each within tolerance of its neighbor, but the first and last
    values arbitrarily far apart). The first draft did this and silently
    merged nearly an entire 50-candle random walk into one fake "cluster."

    2. "Didn't close through it" MUST be evaluated against the cluster's
    shared level, not against each candle's own high/low. The first draft
    checked `close < high` (resistance) per-candle in isolation -- which is
    trivially true for nearly every ordinary candle (a candle's close is
    almost never exactly equal to its own high), so it filtered out almost
    nothing and let essentially the whole window's noise into clustering.
    "Rejected at the level" only means something once a shared candidate
    level exists to be rejected AT -- hence the two-stage structure here.
    """
    if atr <= 0 or len(values) == 0:
        return []
    tol = tolerance_atr * atr

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_closes = closes[order]
    n = len(sorted_values)

    clusters: List[LiquidityCluster] = []
    i = 0
    while i < n:
        anchor = sorted_values[i]
        j = i + 1
        while j < n and sorted_values[j] - anchor <= tol:
            j += 1
        group_values = sorted_values[i:j]
        group_closes = sorted_closes[i:j]
        candidate_level = float(np.mean(group_values))

        if side == "resistance":
            respected = group_closes < candidate_level  # didn't close through (above) the shared level
        else:
            respected = group_closes > candidate_level  # didn't close through (below) the shared level
        touch_count = int(np.sum(respected))

        if touch_count >= 2:
            # 2026-08-14 review: `level` was the mean of ALL proximity
            # members, including the ones the `respected` filter had just
            # disqualified as non-touches. The reported level was therefore
            # pulled by wicks explicitly judged NOT to be part of the
            # cluster -- e.g. highs [100.00, 100.05, 100.10] where only the
            # first two were rejections reported level=100.050 rather than
            # the actual touching mean of 100.025. That level also feeds
            # `distance_atr` and the sweep-breach comparison, so the error
            # propagated. The candidate (all-member) mean is still what
            # DEFINES the shared level for the respect test -- it has to be,
            # since the test is what identifies the touches -- but the
            # REPORTED level is now the mean of the genuine touches only.
            level = float(np.mean(group_values[respected]))
            clusters.append(
                LiquidityCluster(
                    side=side,
                    level=level,
                    touch_count=touch_count,
                    distance_atr=abs(level - float(closes[-1])) / atr,
                )
            )
        i = j

    clusters.sort(key=lambda c: c.touch_count, reverse=True)
    return clusters


def estimate_liquidity_trap(
    df: pd.DataFrame,
    *,
    atr_col: str = "vol_atr_14",
    lookback: int = EQUAL_LEVEL_LOOKBACK,
    level_tolerance_atr: float = DEFAULT_LEVEL_TOLERANCE_ATR,
    proximity_scale_atr: float = DEFAULT_PROXIMITY_SCALE_ATR,
    density_saturation_touches: int = DEFAULT_DENSITY_SATURATION_TOUCHES,
    min_penetration_atr: float = DEFAULT_MIN_PENETRATION_ATR,
    min_sweep_touches: int = DEFAULT_MIN_SWEEP_TOUCHES,
) -> LiquidityTrapResult:
    """
    Estimate stop-hunt/liquidity-sweep trap risk for the LATEST candle in
    `df`, using equal-high/low clusters formed from the trailing `lookback`
    candles strictly before it.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV, DatetimeIndex, sorted ascending, must contain
        'high', 'low', 'close', and `atr_col` (a raw, price-scale ATR --
        NOT a percentile-ranked feature; see predictive_core.dataset's
        module docstring for why raw vs. normalized ATR matters). Must
        have at least `lookback` + 1 rows.
    atr_col : str
        Column name for the raw ATR value, used to make every distance/
        tolerance figure scale-invariant across assets and volatility
        regimes rather than a fixed dollar amount.
    lookback : int
        Trailing window (candles, EXCLUDING the current one) searched for
        equal-level clusters. Defaults to the one exact spec number this
        layer's source docs give (50).
    level_tolerance_atr, proximity_scale_atr, density_saturation_touches :
        Heuristic judgment-call parameters -- NOT spec numbers, see module
        docstring.

    Returns
    -------
    LiquidityTrapResult
    """
    # 2026-08-14 review: these four tunables were entirely unvalidated.
    # `lookback=0` produced an opaque IndexError from `window_closes[-1]`;
    # `proximity_scale_atr=0` / `density_saturation_touches=0` are used as
    # divisors and raised ZeroDivisionError deep inside `_side_risk`. Every
    # sibling module validates its own tunables' domain.
    if lookback < 2:
        raise ValueError(
            f"lookback must be >= 2 (a cluster requires at least two prior wicks), got {lookback}."
        )
    if not math.isfinite(level_tolerance_atr) or level_tolerance_atr <= 0:
        raise ValueError(
            f"level_tolerance_atr must be finite and positive, got {level_tolerance_atr}."
        )
    if not math.isfinite(proximity_scale_atr) or proximity_scale_atr <= 0:
        raise ValueError(
            f"proximity_scale_atr must be finite and positive (it is a divisor), "
            f"got {proximity_scale_atr}."
        )
    if density_saturation_touches < 1:
        raise ValueError(
            f"density_saturation_touches must be >= 1 (it is a divisor), "
            f"got {density_saturation_touches}."
        )
    if not math.isfinite(min_penetration_atr) or min_penetration_atr < 0:
        raise ValueError(
            f"min_penetration_atr must be finite and non-negative, got {min_penetration_atr}."
        )
    # B.4: the bound that survived decoupling. A cluster's level is only
    # defined to within +/- `level_tolerance_atr`, so a penetration smaller
    # than that is inside the level's own uncertainty band -- it cannot be
    # distinguished from merely touching the level, and calling it a sweep
    # would be reading noise as structure.
    if min_penetration_atr < level_tolerance_atr:
        raise ValueError(
            f"min_penetration_atr ({min_penetration_atr}) is below level_tolerance_atr "
            f"({level_tolerance_atr}). A cluster's level is only defined to within "
            f"+/- the tolerance, so a smaller penetration cannot be distinguished from "
            f"touching the level. Raise the penetration or tighten the tolerance."
        )
    if min_sweep_touches < 2:
        raise ValueError(
            f"min_sweep_touches must be >= 2 -- a cluster is by definition 'two or more wicks' "
            f"(Adaptive doc, Layer 5), so a lower bound cannot select anything additional; "
            f"got {min_sweep_touches}."
        )

    required = {"high", "low", "close", atr_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted ascending.")
    if len(df) < lookback + 1:
        raise ValueError(
            f"df has {len(df)} rows; need at least lookback+1={lookback + 1} "
            f"(lookback candles strictly before the current one)."
        )

    current = df.iloc[-1]
    window = df.iloc[-(lookback + 1):-1]  # strictly prior candles -- see no-lookahead note in module docstring
    atr = float(current[atr_col])
    if not np.isfinite(atr) or atr <= 0:
        raise ValueError(f"Current-candle {atr_col} must be a positive finite value, got {atr}.")

    # Non-finite price data silently degrades every comparison below to
    # False (no clusters form, no sweep is ever detected), which would
    # report "no structural risk" for what is really "no usable data" --
    # the same fail-open class of bug fixed across the sibling modules in
    # this 2026-08-14 review. Checked on the current candle and the window.
    for col in ("high", "low", "close"):
        if not math.isfinite(float(current[col])):
            raise ValueError(f"Current-candle {col!r} must be finite, got {current[col]!r}.")
    window_ohlc = window[["high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(window_ohlc).all():
        raise ValueError(
            f"The trailing {lookback}-candle window contains non-finite high/low/close "
            f"values; cluster detection would silently find nothing rather than fail."
        )

    window_closes = window["close"].to_numpy(dtype=float)
    reference_price = float(window_closes[-1])  # price immediately before the current candle

    # A cluster only makes sense as "resistance" if it sits ABOVE where
    # price actually was just before this candle (something price would
    # have to rise INTO), and only as "support" if it sits BELOW.
    # `_cluster_levels` clusters purely by wick proximity across the whole
    # window regardless of where price ended up -- without this filter, a
    # cluster from old price action that price has since moved away from
    # (e.g. a former "high" cluster now sitting below current price) would
    # get treated as active resistance, which is incoherent. This exact bug
    # was caught by testing an engineered sweep: `min(level for level in
    # swept)` was picking up an irrelevant cluster far below current price
    # (which a rising candle's high trivially exceeds, meaning nothing) instead
    # of the one genuine nearby resistance level actually being swept.
    resistance_clusters = [
        c for c in _cluster_levels(
            window["high"].to_numpy(dtype=float), window_closes, "resistance", atr, level_tolerance_atr
        )
        if c.level > reference_price
    ]
    support_clusters = [
        c for c in _cluster_levels(
            window["low"].to_numpy(dtype=float), window_closes, "support", atr, level_tolerance_atr
        )
        if c.level < reference_price
    ]
    # --- Sweep check: did the CURRENT candle's wick breach a prior cluster
    # level and then close back on the rejecting side (confirmed reversal)?
    # "Liquidity sweeps (stop-runs above equal highs or below equal lows,
    # followed by a confirmed reversal) are specific, programmatically
    # detectable events" (Adaptive doc, hypothesis section). ---
    sweep_just_confirmed = False
    sweep_direction: Optional[str] = None
    reasons: List[str] = []

    # CONFIRMATION RULE (rewritten 2026-08-14 review). Two distinct levels
    # matter here and they are NOT the same one:
    #
    #  - The EXTENT of the sweep (reported): the farthest level the wick
    #    reached -- highest for resistance, lowest for support. This is what
    #    describes how deep the stop-run went.
    #  - The level confirmation is TESTED against: the NEAREST swept level
    #    (lowest for resistance, highest for support). A "confirmed
    #    reversal" means price was rejected back inside the range -- i.e.
    #    back past ALL the liquidity it just ran, not merely back below the
    #    single farthest level.
    #
    # The previous implementation tested against the farthest level, which
    # is the most permissive possible rule and manufactured false
    # reversals: with clusters at 102 and 110, a candle with high=111 and
    # close=108 was reported as a "confirmed bearish sweep" even though it
    # closed decisively ABOVE the 102 cluster -- i.e. a genuine bullish
    # breakout through that level, labelled as its opposite. (Testing
    # against the farthest level was itself a fix for an earlier bug where
    # `min()` picked up an irrelevant cluster far below current price; that
    # original problem is now handled properly by the
    # `c.level > reference_price` / `c.level < reference_price` position
    # filters applied above, so the nearest-level test is safe here.)
    # SELECTIVITY REQUIREMENTS (2026-08-14). Breach-and-close-back alone is
    # not a stop-run: measured on real BTC 1h it fired on 44.1% of bars, with
    # a near-even bullish/bearish split (182/197) -- i.e. noise, not the
    # "specific, programmatically detectable event" the spec describes. For
    # scale, the earlier compression+SFP proxy selected 2.04% of candles.
    # Tightening `level_tolerance_atr` barely moved it (44.1% -> 35.4% going
    # from 0.25 to 0.05 ATR), which ruled out cluster tolerance as the cause.
    #
    # The actual gap: a high that exceeded a level by a cent and closed a
    # cent back counted identically to a genuine stop-run. Two additional
    # conditions, both squarely within this layer's "equal highs/lows and
    # wick behaviour" scope, now have to hold:
    #
    #   1. PENETRATION -- the wick must clear the level by at least
    #      `min_penetration_atr` ATR. A sweep that never meaningfully took
    #      the liquidity did not take it.
    #   2. REJECTION -- the rejecting wick must be at least
    #      `min_rejection_frac` of the candle's own range. This is the
    #      standard pin-bar/SFP shape test and is what distinguishes
    #      "rejected at the level" from "drifted past it and closed lower".
    #
    # Both are judgment-call numbers (no source doc quantifies either), which
    # is why they are named, documented, tunable constants rather than
    # literals -- see DEFAULT_MIN_PENETRATION_ATR / DEFAULT_MIN_REJECTION_FRAC.
    swept_resistance = [
        c for c in resistance_clusters
        if c.touch_count >= min_sweep_touches
        and current["high"] > c.level + min_penetration_atr * atr
    ]
    bearish_confirmed = False
    if swept_resistance:
        extent_cluster = max(swept_resistance, key=lambda c: c.level)
        nearest_swept_level = min(c.level for c in swept_resistance)
        if current["close"] < nearest_swept_level:
            bearish_confirmed = True
            reasons.append(
                f"Bearish sweep confirmed: high {current['high']:.4g} breached resistance to "
                f"{extent_cluster.level:.4g} ({extent_cluster.touch_count} touches) by "
                f">={min_penetration_atr:.2f} ATR, closed back below all "
                f"{len(swept_resistance)} swept level(s) at {current['close']:.4g}."
            )

    # Mirror of the resistance case above.
    swept_support = [
        c for c in support_clusters
        if c.touch_count >= min_sweep_touches
        and current["low"] < c.level - min_penetration_atr * atr
    ]
    bullish_confirmed = False
    if swept_support:
        extent_cluster = min(swept_support, key=lambda c: c.level)
        nearest_swept_level = max(c.level for c in swept_support)
        if current["close"] > nearest_swept_level:
            bullish_confirmed = True
            reasons.append(
                f"Bullish sweep confirmed: low {current['low']:.4g} breached support to "
                f"{extent_cluster.level:.4g} ({extent_cluster.touch_count} touches) by "
                f">={min_penetration_atr:.2f} ATR, closed back above all "
                f"{len(swept_support)} swept level(s) at {current['close']:.4g}."
            )

    # CONFLICT HANDLING (2026-08-14 review). The previous code asserted
    # that both directions "can't be confirmed on the same candle in
    # practice (would require closing both above AND below distinct
    # levels)" and simply let the bullish branch overwrite the bearish one.
    # That reasoning is inverted: it requires closing BELOW a resistance
    # level and ABOVE a support level, which is exactly what any ordinary
    # wide-range candle that closes near the middle of its range does --
    # e.g. resistance ~105, support ~95, candle high=106 low=94 close=100.
    # Both branches fired and `sweep_direction` silently became "bullish"
    # purely because that branch runs second. `ensemble.py` consumes
    # `sweep_direction` as a DIRECTIONAL signal scaled by
    # `trap_probability`, so a symmetric, directionally meaningless outside
    # bar was injecting a hard bullish lean into the meta-model.
    #
    # A candle that swept liquidity on BOTH sides and closed between them
    # has, by construction, given no directional information -- it is
    # reported as no confirmed sweep, which the ensemble treats as neutral.
    # This is the capital-preservation-consistent resolution, and the
    # conflict stays visible in `reasons` rather than being silently
    # collapsed.
    if bearish_confirmed and bullish_confirmed:
        reasons.append(
            "Both a bearish and a bullish sweep confirmed on the same candle "
            "(liquidity taken on both sides, close landed between the swept levels) -- "
            "reported as NO confirmed sweep: this candle carries no directional information."
        )
    elif bearish_confirmed:
        sweep_just_confirmed = True
        sweep_direction = "bearish"  # swept a resistance cluster (stop-run up), reversed down
    elif bullish_confirmed:
        sweep_just_confirmed = True
        sweep_direction = "bullish"  # swept a support cluster (stop-run down), reversed up

    # --- Trap probability: heuristic forward-looking risk that an
    # UN-SWEPT nearby cluster gets hunted next -- proximity (closer =
    # higher risk) combined with density (more touches = a more
    # "attractive"/loaded pool of resting stops = more likely target).
    # Deliberately NOT based on any actual future-price computation --
    # this is descriptive structure at the current moment, not a forecast
    # (matching the "heuristic estimator, not a forecaster" framing used
    # throughout this project's Layer 3/4 modules too). ---
    # 2026-08-14 review: the comment above promises risk from an UN-SWEPT
    # cluster, but `nearest_*` was previously computed over ALL clusters,
    # including any the current candle had just run. A pool that was just
    # consumed was still being scored as pending forward risk. The
    # already-swept clusters remain fully visible in `reasons` and in the
    # `sweep_*` fields; these two fields now mean "nearest cluster still
    # standing", consistent with what `trap_probability` claims to measure.
    swept_ids = {id(c) for c in swept_resistance} | {id(c) for c in swept_support}
    unswept_resistance = [c for c in resistance_clusters if id(c) not in swept_ids]
    unswept_support = [c for c in support_clusters if id(c) not in swept_ids]
    nearest_resistance = min(unswept_resistance, key=lambda c: c.distance_atr, default=None)
    nearest_support = min(unswept_support, key=lambda c: c.distance_atr, default=None)

    def _side_risk(cluster: Optional[LiquidityCluster]) -> float:
        if cluster is None:
            return 0.0
        proximity_score = _clip01(1.0 - cluster.distance_atr / proximity_scale_atr)
        density_score = _clip01(cluster.touch_count / density_saturation_touches)
        return _clip01((proximity_score + density_score) / 2.0)

    resistance_risk = _side_risk(nearest_resistance)
    support_risk = _side_risk(nearest_support)
    trap_probability = max(resistance_risk, support_risk)
    liquidity_attraction_score = _clip01((resistance_risk + support_risk) / 2.0)

    if nearest_resistance is not None:
        reasons.append(
            f"Nearest resistance cluster: {nearest_resistance.touch_count} touches, "
            f"{nearest_resistance.distance_atr:.2f} ATR away (risk contribution {resistance_risk:.2f})."
        )
    if nearest_support is not None:
        reasons.append(
            f"Nearest support cluster: {nearest_support.touch_count} touches, "
            f"{nearest_support.distance_atr:.2f} ATR away (risk contribution {support_risk:.2f})."
        )
    if nearest_resistance is None and nearest_support is None:
        reasons.append(
            "No equal-level clusters detected in the trailing window -- "
            "no structural basis for elevated trap risk from this heuristic."
        )

    return LiquidityTrapResult(
        trap_probability=trap_probability,
        liquidity_attraction_score=liquidity_attraction_score,
        sweep_just_confirmed=sweep_just_confirmed,
        sweep_direction=sweep_direction,
        nearest_resistance_cluster=nearest_resistance,
        nearest_support_cluster=nearest_support,
        reasons=reasons,
    )
