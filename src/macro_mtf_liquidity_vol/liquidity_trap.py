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
DEFAULT_LEVEL_TOLERANCE_ATR = 0.25  # wicks within this many ATRs of each other count as "equal"
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
        level = float(np.mean(group_values))

        if side == "resistance":
            respected = group_closes < level  # didn't close through (above) the shared level
        else:
            respected = group_closes > level  # didn't close through (below) the shared level
        touch_count = int(np.sum(respected))

        if touch_count >= 2:
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
    atr_col: str = "vol_atr_14",
    lookback: int = EQUAL_LEVEL_LOOKBACK,
    level_tolerance_atr: float = DEFAULT_LEVEL_TOLERANCE_ATR,
    proximity_scale_atr: float = DEFAULT_PROXIMITY_SCALE_ATR,
    density_saturation_touches: int = DEFAULT_DENSITY_SATURATION_TOUCHES,
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
    nearest_resistance = min(resistance_clusters, key=lambda c: c.distance_atr, default=None)
    nearest_support = min(support_clusters, key=lambda c: c.distance_atr, default=None)

    # --- Sweep check: did the CURRENT candle's wick breach a prior cluster
    # level and then close back on the rejecting side (confirmed reversal)?
    # "Liquidity sweeps (stop-runs above equal highs or below equal lows,
    # followed by a confirmed reversal) are specific, programmatically
    # detectable events" (Adaptive doc, hypothesis section). ---
    sweep_just_confirmed = False
    sweep_direction: Optional[str] = None
    reasons: List[str] = []

    # The meaningful "level swept" is the FARTHEST (highest, for resistance)
    # one the wick actually reached, not the nearest one it trivially also
    # exceeded along the way -- a high that clears a level 3 ATR away
    # necessarily also clears every resistance cluster closer than that,
    # but "confirmed reversal" has to be evaluated against the extent of
    # the sweep, not an unrelated closer level. Using `min()` here was a
    # real bug caught by testing an engineered sweep against real
    # (non-degenerate) synthetic data: it picked up an irrelevant nearby
    # cluster the price never meaningfully "swept" and incorrectly
    # required closing back below THAT instead of the level actually hunted.
    swept_resistance = [c for c in resistance_clusters if current["high"] > c.level]
    if swept_resistance:
        swept_extent = max(c.level for c in swept_resistance)
        if current["close"] < swept_extent:
            sweep_just_confirmed = True
            sweep_direction = "bearish"  # swept a resistance cluster (stop-run up), reversed down
            extent_cluster = max(swept_resistance, key=lambda c: c.level)
            reasons.append(
                f"Bearish sweep confirmed: high {current['high']:.4g} breached resistance cluster "
                f"(level~{swept_extent:.4g}, {extent_cluster.touch_count} touches), closed back "
                f"below at {current['close']:.4g}."
            )

    # Mirror of the resistance case above: the meaningful "level swept" is
    # the farthest (lowest) one reached, not the nearest one trivially also
    # undercut along the way.
    swept_support = [c for c in support_clusters if current["low"] < c.level]
    if swept_support:
        swept_extent = min(c.level for c in swept_support)
        if current["close"] > swept_extent:
            # Both directions can't be confirmed on the same candle in
            # practice (would require closing both above AND below
            # distinct levels), but don't assume -- report whichever is
            # detected; if genuinely both, bullish is reported second and
            # wins, an explicit, visible (not silent) tie-break choice.
            sweep_just_confirmed = True
            sweep_direction = "bullish"  # swept a support cluster (stop-run down), reversed up
            extent_cluster = min(swept_support, key=lambda c: c.level)
            reasons.append(
                f"Bullish sweep confirmed: low {current['low']:.4g} breached support cluster "
                f"(level~{swept_extent:.4g}, {extent_cluster.touch_count} touches), closed back "
                f"above at {current['close']:.4g}."
            )

    # --- Trap probability: heuristic forward-looking risk that an
    # UN-SWEPT nearby cluster gets hunted next -- proximity (closer =
    # higher risk) combined with density (more touches = a more
    # "attractive"/loaded pool of resting stops = more likely target).
    # Deliberately NOT based on any actual future-price computation --
    # this is descriptive structure at the current moment, not a forecast
    # (matching the "heuristic estimator, not a forecaster" framing used
    # throughout this project's Layer 3/4 modules too). ---
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
