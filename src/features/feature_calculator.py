"""Feature engineering module using pandas-ta with strict no-lookahead guarantees.

All calculations are explicitly backward-looking. Rolling windows never use
center=True, shifts are strictly positive (past data only), and every output
feature is percentile-ranked against a trailing 252-candle history before leaving
this module. Raw indicator values are never exposed to downstream consumers,
with one deliberate exception documented on `volu_on_breakout` below.

Feature "type" (continuous vs. already-bounded/binary/state) is tracked
explicitly via the `_FeatureSet` bookkeeping class below rather than inferred
from column-name substrings, so adding a feature can never silently change
how an unrelated, similarly-named feature is treated.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# Configuration constants aligned with MASTER_CONTEXT.md constraints
LOOKBACK_PERCENTILE = 252
STRUCTURAL_N_SHORT = 20
STRUCTURAL_N_LONG = 50
EPS = 1e-9

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class _FeatureSet:
    """Explicit bookkeeping of which output columns are continuous (and must be
    percentile-ranked) vs. binary/state/already-bounded (and must not be).

    This replaces substring pattern-matching on column names: every column is
    tagged at the moment it's created, so there is no way for a new feature to
    be silently mis-classified just because its name resembles an existing one.
    """

    def __init__(self, index: pd.Index):
        self._df = pd.DataFrame(index=index)
        self._continuous: List[str] = []
        self._excluded: List[str] = []

    def add_continuous(self, name: str, series: pd.Series) -> None:
        assert name not in self._df.columns, f"Duplicate feature column: {name}"
        self._df[name] = series
        self._continuous.append(name)

    def add_excluded(self, name: str, series: pd.Series) -> None:
        """For binary flags, state classifications, or values already scaled
        to a meaningful [0, 1] percentile that must not be re-ranked."""
        assert name not in self._df.columns, f"Duplicate feature column: {name}"
        self._df[name] = series
        self._excluded.append(name)

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    @property
    def continuous_cols(self) -> List[str]:
        return list(self._continuous)


def _validate_input(df: pd.DataFrame) -> None:
    """Ensure input DataFrame meets strict structural requirements."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError(
            "Input DataFrame contains duplicate timestamps. Please deduplicate "
            "before passing to feature calculator."
        )
    ohlcv = df[list(_REQUIRED_COLUMNS)]
    if not np.isfinite(ohlcv.to_numpy(dtype=float)).all():
        raise ValueError("Input OHLCV data contains NaN or infinite values.")
    if (ohlcv[["open", "high", "low", "close", "volume"]] < 0).any().any():
        raise ValueError("Input OHLCV data contains negative price/volume values.")
    if (df["high"] < df["low"]).any():
        raise ValueError("Malformed candles detected: high < low.")
    if ((df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))).any():
        raise ValueError(
            "Malformed candles detected: high/low do not bound open/close."
        )


def _get_col(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Fetch a pandas-ta output column by prefix instead of a hardcoded exact
    name. pandas-ta's column-name suffixes (e.g. the std-dev formatting on
    Bollinger Bands, or the k/d params on StochRSI) vary across versions and
    across float vs. int parameter types. Matching by unique prefix is what
    actually survives a `pip install -U pandas-ta`.
    """
    matches = [c for c in frame.columns if c.startswith(prefix)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one column starting with '{prefix}' from "
            f"pandas-ta output, found {matches}. Columns were: {list(frame.columns)}"
        )
    return frame[matches[0]]


def _calculate_structural_features(df: pd.DataFrame) -> _FeatureSet:
    """Structural market features (HH/HL/LH/LL, BOS, ChoCh, SFP, range/trend,
    distance to S/R, Wyckoff)."""
    feats = _FeatureSet(df.index)
    epsilon = 1e-6

    rolling_high_20 = df["high"].rolling(STRUCTURAL_N_SHORT).max()
    rolling_low_20 = df["low"].rolling(STRUCTURAL_N_SHORT).min()
    rolling_high_50 = df["high"].rolling(STRUCTURAL_N_LONG).max()
    rolling_low_50 = df["low"].rolling(STRUCTURAL_N_LONG).min()

    # Prior-structure references (current candle excluded) — used consistently
    # by BOS *and* the distance-to-S/R features below, so "structure" means the
    # same thing everywhere in this function.
    prior_high_20 = rolling_high_20.shift(1)
    prior_low_20 = rolling_low_20.shift(1)
    prior_high_50 = rolling_high_50.shift(1)
    prior_low_50 = rolling_low_50.shift(1)

    # HH/LL flags: is *this* candle the extreme of its trailing window
    # (window is inclusive of the current candle by construction/definition).
    feats.add_excluded(
        "struct_hh_20", np.isclose(df["high"], rolling_high_20, atol=epsilon).astype(float)
    )
    feats.add_excluded(
        "struct_ll_20", np.isclose(df["low"], rolling_low_20, atol=epsilon).astype(float)
    )
    feats.add_excluded(
        "struct_hh_50", np.isclose(df["high"], rolling_high_50, atol=epsilon).astype(float)
    )
    feats.add_excluded(
        "struct_ll_50", np.isclose(df["low"], rolling_low_50, atol=epsilon).astype(float)
    )

    bos_bullish = ((df["close"] > prior_high_20) | (df["close"] > prior_high_50)).astype(float)
    bos_bearish = ((df["close"] < prior_low_20) | (df["close"] < prior_low_50)).astype(float)
    feats.add_excluded("struct_bos_bullish", bos_bullish)
    feats.add_excluded("struct_bos_bearish", bos_bearish)

    feats.add_excluded(
        "struct_choch_bullish",
        ((bos_bearish.shift(1) == 1) & (bos_bullish == 1)).astype(float),
    )
    feats.add_excluded(
        "struct_choch_bearish",
        ((bos_bullish.shift(1) == 1) & (bos_bearish == 1)).astype(float),
    )

    feats.add_excluded(
        "struct_sfp_bullish",
        ((df["low"] < df["low"].shift(1).rolling(STRUCTURAL_N_SHORT).min())
         & (df["close"] > df["open"])).astype(float),
    )
    feats.add_excluded(
        "struct_sfp_bearish",
        ((df["high"] > df["high"].shift(1).rolling(STRUCTURAL_N_SHORT).max())
         & (df["close"] < df["open"])).astype(float),
    )

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx_val = _get_col(adx, "ADX_14")
    feats.add_excluded("struct_trend_state", (adx_val > 25).astype(float))

    atr = ta.atr(df["high"], df["low"], df["close"], length=14)  # Series, not DataFrame
    # Distance to S/R uses the SAME prior-structure references as BOS above —
    # i.e. it excludes the current candle, so it can't trivially read as ~0
    # just because the current candle happens to set a new high/low.
    feats.add_continuous("struct_dist_to_resist", (prior_high_20 - df["close"]) / (atr + EPS))
    feats.add_continuous("struct_dist_to_support", (df["close"] - prior_low_20) / (atr + EPS))

    price_range = df["high"] - df["low"]
    vol_sma = ta.sma(df["volume"], length=20)
    wyckoff = np.where(
        (df["volume"] > vol_sma * 1.5) & (price_range < price_range.rolling(STRUCTURAL_N_SHORT).mean()),
        1.0,
        0.0,
    )
    feats.add_excluded("struct_wyckoff_phase", pd.Series(wyckoff, index=df.index))

    return feats


def _calculate_volatility_features(df: pd.DataFrame) -> _FeatureSet:
    """Volatility features (ATR percentile, BB width, rolling vol, expansion/
    compression state)."""
    feats = _FeatureSet(df.index)

    atr = ta.atr(df["high"], df["low"], df["close"], length=14)  # Series, not DataFrame
    feats.add_continuous("vol_atr_14", atr)

    bb = ta.bbands(df["close"], length=20, std=2)
    bb_upper = _get_col(bb, "BBU_20_2.0")
    bb_lower = _get_col(bb, "BBL_20_2.0")
    bb_mid = _get_col(bb, "BBM_20_2.0")
    bb_width = (bb_upper - bb_lower) / (bb_mid + EPS)
    feats.add_continuous("vol_bb_width", bb_width)

    returns = df["close"].pct_change()
    feats.add_continuous("vol_std_20", returns.rolling(20).std())

    atr_pct_rank = atr.rolling(LOOKBACK_PERCENTILE).rank(pct=True)
    bb_width_pct_rank = bb_width.rolling(LOOKBACK_PERCENTILE).rank(pct=True)
    feats.add_excluded("vol_atr_pct_rank", atr_pct_rank)
    feats.add_excluded("vol_bb_width_pct_rank", bb_width_pct_rank)

    feats.add_excluded("vol_state_compressed", (atr_pct_rank < 0.3).astype(float))
    feats.add_excluded(
        "vol_state_expanding", ((atr_pct_rank >= 0.3) & (atr_pct_rank <= 0.7)).astype(float)
    )
    feats.add_excluded("vol_state_extended", (atr_pct_rank > 0.7).astype(float))

    atr_5_shift = atr.shift(5)
    spike_ratio = atr / (atr_5_shift + EPS)
    vol_spike = (spike_ratio >= 2.0).astype(float)
    vol_spike = vol_spike.fillna(0.0)  # no chained inplace assignment
    feats.add_excluded("vol_spike", vol_spike)

    return feats


def _calculate_momentum_features(df: pd.DataFrame) -> _FeatureSet:
    """Momentum features (RSI + ROC, divergence, MACD slope, StochRSI, ADX/DI,
    multi-horizon returns).

    Divergence detection uses a vectorised rolling-window proxy (comparison
    against trailing rolling min/max) rather than iterative swing/pivot
    detection, and will therefore fire more often than a textbook discretionary
    definition of divergence. This is a deliberate, documented approximation,
    not a bug.
    """
    feats = _FeatureSet(df.index)

    rsi = ta.rsi(df["close"], length=14)
    feats.add_continuous("mom_rsi_14", rsi)
    feats.add_continuous("mom_rsi_roc", rsi.pct_change())

    price_ll = df["low"].rolling(STRUCTURAL_N_SHORT).min()
    rsi_ll = rsi.rolling(STRUCTURAL_N_SHORT).min()
    feats.add_excluded(
        "mom_rsi_div_bullish",
        ((df["low"] < price_ll.shift(1)) & (rsi > rsi_ll.shift(1))).astype(float),
    )

    price_hh = df["high"].rolling(STRUCTURAL_N_SHORT).max()
    rsi_hh = rsi.rolling(STRUCTURAL_N_SHORT).max()
    feats.add_excluded(
        "mom_rsi_div_bearish",
        ((df["high"] > price_hh.shift(1)) & (rsi < rsi_hh.shift(1))).astype(float),
    )

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    macd_hist = _get_col(macd, "MACDh_12_26_9")
    feats.add_continuous("mom_macd_hist_slope", macd_hist.diff())

    stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14)
    feats.add_continuous("mom_stochrsi_k", _get_col(stochrsi, "STOCHRSIk_"))

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    feats.add_continuous("mom_adx_14", _get_col(adx, "ADX_14"))
    feats.add_continuous("mom_di_spread", _get_col(adx, "DMP_14") - _get_col(adx, "DMN_14"))

    for period in (5, 10, 20):
        feats.add_continuous(f"mom_ret_{period}", df["close"].pct_change(period))

    return feats


def _calculate_volume_features(df: pd.DataFrame) -> _FeatureSet:
    """Volume features (z-score, breakout ratio, body ratio, wick direction/
    magnitude)."""
    feats = _FeatureSet(df.index)

    vol_sma = ta.sma(df["volume"], length=20)
    vol_std = df["volume"].rolling(20).std()
    feats.add_continuous("volu_zscore", (df["volume"] - vol_sma) / (vol_std + EPS))

    prev_highs = df["high"].shift(1).rolling(STRUCTURAL_N_SHORT).max()
    prev_lows = df["low"].shift(1).rolling(STRUCTURAL_N_SHORT).min()
    is_breakout = (df["close"] > prev_highs) | (df["close"] < prev_lows)
    breakout_ratio_raw = df["volume"] / (vol_sma + EPS)
    # Rank the magnitude ONLY among candles where a breakout actually happened,
    # so "no breakout" (0) is never tied against "weak volume on a real
    # breakout". Non-breakout candles get an explicit sentinel of 0.0, strictly
    # below any real ranked value in (0, 1]. Ranking the raw magnitude
    # unconditionally (as before) let a huge mass of structural zeros pollute
    # the percentile distribution via tie-averaging.
    masked_for_rank = breakout_ratio_raw.where(is_breakout)
    breakout_rank = masked_for_rank.rolling(LOOKBACK_PERCENTILE, min_periods=1).rank(pct=True)
    breakout_feature = breakout_rank.where(is_breakout, 0.0)
    feats.add_excluded("volu_on_breakout", breakout_feature)  # already ranked/bounded

    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    feats.add_continuous("volu_body_ratio", body / (total_range + EPS))

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    feats.add_continuous("volu_upper_wick_mag", upper_wick / (total_range + EPS))
    feats.add_continuous("volu_lower_wick_mag", lower_wick / (total_range + EPS))

    return feats


def _apply_trailing_percentile_rank(
    combined: pd.DataFrame, continuous_cols: List[str], window: int = LOOKBACK_PERCENTILE
) -> pd.DataFrame:
    """Percentile-rank continuous columns against their trailing `window`
    history. Which columns are "continuous" is passed in explicitly by the
    caller (built up via `_FeatureSet.add_continuous`) rather than inferred
    from column-name substrings.
    """
    if not continuous_cols:
        return combined
    ranked = combined.copy()
    ranked[continuous_cols] = ranked[continuous_cols].rolling(window=window).rank(pct=True)
    return ranked


def _first_fully_warmed_index(df: pd.DataFrame, continuous_cols: List[str]) -> int:
    """Return the integer position of the first row at which every continuous
    (percentile-ranked) column has a non-NaN value, i.e. has cleared its
    trailing-252 warm-up. Rows before this are NOT a valid fixed-length
    feature vector — they're a jagged mix of ready and not-ready columns —
    and must not be shipped to a downstream consumer.
    """
    if not continuous_cols:
        return 0
    valid_mask = df[continuous_cols].notna().all(axis=1)
    valid_positions = np.flatnonzero(valid_mask.to_numpy())
    if len(valid_positions) == 0:
        return len(df)  # no fully-warmed row exists yet
    return int(valid_positions[0])


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and normalize technical features for OHLCV data.

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            and a DatetimeIndex. Must be sorted chronologically.

    Returns:
        DataFrame containing fixed-length, percentile-ranked feature vectors.
        Only rows where every continuous feature has cleared its trailing
        252-candle warm-up are returned — i.e. every row in the output is a
        genuinely complete, comparable feature vector. Callers that need the
        raw (partially-NaN) warm-up period should call the private
        `_calculate_*_features` functions directly.

    Raises:
        ValueError: If input DataFrame is empty, missing required columns, or
            contains malformed/invalid OHLCV values.
        TypeError: If index is not a DatetimeIndex.
    """
    _validate_input(df)

    data = df.copy()
    if not data.index.is_monotonic_increasing:
        data.sort_index(inplace=True)

    struct_feats = _calculate_structural_features(data)
    vol_feats = _calculate_volatility_features(data)
    mom_feats = _calculate_momentum_features(data)
    volu_feats = _calculate_volume_features(data)

    all_sets = [struct_feats, vol_feats, mom_feats, volu_feats]
    combined = pd.concat([fs.frame.reindex(data.index) for fs in all_sets], axis=1)
    if not combined.columns.is_unique:
        dupes = combined.columns[combined.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate feature column names detected: {dupes}")

    continuous_cols = [c for fs in all_sets for c in fs.continuous_cols]
    ranked_features = _apply_trailing_percentile_rank(combined, continuous_cols)
    ranked_features.index = data.index

    cutoff = _first_fully_warmed_index(ranked_features, continuous_cols)
    n_dropped = cutoff
    if n_dropped:
        logger.info(
            "Dropping %d warm-up rows (of %d) before all continuous features "
            "have a full %d-candle history.",
            n_dropped, len(ranked_features), LOOKBACK_PERCENTILE,
        )
    result = ranked_features.iloc[cutoff:]

    if result.isna().any().any():
        bad_cols = result.columns[result.isna().any()].tolist()
        raise RuntimeError(
            f"Unexpected NaNs remain in output after warm-up trim: {bad_cols}"
        )
    return result

