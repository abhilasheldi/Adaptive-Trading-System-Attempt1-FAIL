"""
Bridges raw OHLCV -> normalized feature vectors (`features.feature_calculator`)
+ triple-barrier labels (`predictive_core.labeling`) into ONE dataframe with
the exact column set every downstream consumer (`HistoricalAnalogEngine`,
`boosted_core`'s trainers, `PredictiveCoreStrategy`, `WalkForwardBacktester`)
actually requires: normalized feature columns + raw 'close'/'high'/'low'
(/'open'/'volume') + 'direction'/'magnitude'/'duration'.

Why this glue has to exist, not just "call calculate_features() then
apply_triple_barrier()": `calculate_features()` deliberately emits ONLY
percentile-ranked feature columns -- no raw OHLCV, no raw ATR (see that
module's "no raw values leave this module" rule, itself the fix for a prior
bug). But `apply_triple_barrier()` needs raw-scale 'close'/'high'/'low' and a
raw-scale ATR to set real price-level barriers (a percentile-ranked ATR in
[0, 1] used as a price offset is nonsensical -- it would place every barrier
a fraction of a dollar from entry regardless of actual BTC/ETH volatility),
and every downstream consumer (position sizing, cost model, order execution)
needs raw 'close'. Piping `calculate_features()`'s output directly into
`apply_triple_barrier()` either KeyErrors (no 'close' column at all) or
silently corrupts every barrier level. This module is the one place that
reconciles the two.
"""

from typing import List

import pandas as pd
import pandas_ta as ta

from src.features.feature_calculator import calculate_features
from src.predictive_core.labeling import apply_triple_barrier

# Every column calculate_features() produces uses one of these prefixes
# (struct_/vol_/mom_/volu_ -- see that module). Deriving feature_cols from
# the prefix rather than hardcoding an exact list means this stays correct
# if feature_calculator ever adds/removes a feature.
_FEATURE_COLUMN_PREFIXES = ("struct_", "vol_", "mom_", "volu_")

# Internal-only column name for the raw-scale ATR used to set triple-barrier
# levels. Dropped from the final output -- it exists only to satisfy
# apply_triple_barrier's atr_col requirement and must never be mistaken for
# a modeling feature (it is NOT percentile-ranked/point-in-time-normalized
# the way real feature_cols columns are).
_RAW_ATR_COL = "_raw_atr_for_labeling"

# Public name for that same raw-scale ATR when `keep_raw_atr=True`. Chosen to
# sit OUTSIDE `_FEATURE_COLUMN_PREFIXES` so `feature_columns()` can never pick
# it up as a model feature -- a raw price-scale column must never enter the
# percentile-ranked feature set.
RAW_ATR_COLUMN = "raw_atr"


def feature_columns(df: pd.DataFrame) -> List[str]:
    """The exact `feature_cols` list to pass to `HistoricalAnalogEngine`,
    `boosted_core`'s trainers, or `PredictiveCoreStrategy`: every column in
    `df` produced by `calculate_features()`, i.e. everything that isn't raw
    OHLCV or a triple-barrier label."""
    return [c for c in df.columns if c.startswith(_FEATURE_COLUMN_PREFIXES)]


def build_asset_modeling_frame(
    raw_ohlcv: pd.DataFrame,
    atr_length: int = 14,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_candles: int = 48,
    neutral_zone_atr: float = 0.25,
    keep_raw_atr: bool = False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    raw_ohlcv : pd.DataFrame
        Single-asset raw OHLCV, DatetimeIndex, sorted ascending, columns at
        least {'open','high','low','close','volume'} -- e.g. straight from
        `data_engine.crypto_fetcher`'s saved raw parquet (extra columns
        such as 'gap_class'/'is_outlier' are fine and simply ignored).
    atr_length, tp_mult, sl_mult, max_candles, neutral_zone_atr :
        Forwarded to `apply_triple_barrier` (`atr_length` controls the raw
        ATR computed here for barrier levels; `predictive_core.triple_barrier`
        in config.yaml is the canonical source for tp_mult/sl_mult/max_candles
        in a real run).
    keep_raw_atr : bool
        When True, the raw price-scale ATR used to set the triple-barrier
        levels is retained on the output as `RAW_ATR_COLUMN` ('raw_atr')
        instead of being dropped. Needed by Layer 5
        (`macro_mtf_liquidity_vol.liquidity_trap`), which measures cluster
        distances in ATR units and therefore requires a RAW-scale ATR --
        the percentile-ranked `vol_atr_14` feature is in [0, 1] and would
        silently corrupt every distance it is used for.

        Deliberately opt-in, defaulting False: this column is not a model
        feature (its name is outside `_FEATURE_COLUMN_PREFIXES`, so
        `feature_columns()` will never pick it up), and adding it
        unconditionally would change the output shape for every existing
        caller and test.

    Returns
    -------
    pd.DataFrame
        Indexed on the intersection of (a) calculate_features()'s post-warm-
        up index and (b) apply_triple_barrier()'s valid-label index. Columns:
        every `feature_columns()` column, raw 'open'/'high'/'low'/'close'/
        'volume', and 'direction'/'magnitude'/'duration'. Ready to hand to
        `HistoricalAnalogEngine.fit()`, `boosted_core`'s trainers,
        `PredictiveCoreStrategy`, or `WalkForwardBacktester.run()` (the last
        of these still wants one instrument per call -- see its docstring;
        this function must be called once per asset, never on a
        `pool_assets()`-pooled frame, for the same reason
        `apply_triple_barrier` itself must run before pooling).
    """
    features = calculate_features(raw_ohlcv)

    raw_atr = ta.atr(raw_ohlcv["high"], raw_ohlcv["low"], raw_ohlcv["close"], length=atr_length)

    ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in raw_ohlcv.columns]
    merged = features.join(raw_ohlcv[ohlcv_cols], how="inner")
    merged[_RAW_ATR_COL] = raw_atr.reindex(merged.index)
    # Defensive only: ATR(atr_length)'s own warm-up is much shorter than
    # calculate_features()'s 252-candle percentile warm-up, so this should
    # never actually drop rows -- but never silently hand apply_triple_barrier
    # a NaN "ATR" that would compare as False in every barrier check.
    merged = merged.dropna(subset=[_RAW_ATR_COL])

    labeled = apply_triple_barrier(
        merged,
        atr_col=_RAW_ATR_COL,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        max_candles=max_candles,
        neutral_zone_atr=neutral_zone_atr,
    )
    if keep_raw_atr:
        return labeled.rename(columns={_RAW_ATR_COL: RAW_ATR_COLUMN})
    return labeled.drop(columns=[_RAW_ATR_COL])
