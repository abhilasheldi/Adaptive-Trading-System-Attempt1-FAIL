"""Triple-barrier labeling engine and asset pooling utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List


def apply_triple_barrier(
    df: pd.DataFrame,
    atr_col: str = "vol_atr_14",
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_candles: int = 48,
    neutral_zone_atr: float = 0.25,
) -> pd.DataFrame:
    """
    Applies triple-barrier labeling to derive direction, magnitude (MFE), and duration.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'close', 'high', 'low' and the specified ATR column.
        Index must be a DatetimeIndex, sorted ascending, unique within this
        single-asset series (do not pass a pooled multi-asset frame here --
        label each asset separately, then pool with `pool_assets`).
    atr_col : str
        Column name for ATR(14) values.
    tp_mult : float
        Take-profit multiplier relative to ATR.
    sl_mult : float
        Stop-loss multiplier relative to ATR.
    max_candles : int
        Vertical barrier (maximum holding period in candles).
    neutral_zone_atr : float
        Only applies when neither the TP nor SL barrier is touched before the
        vertical barrier (max_candles). If the closing move at the vertical
        barrier is smaller than `neutral_zone_atr * ATR`, the instance is
        labeled neutral (0.0) rather than forced into up/down. This is what
        makes the direction label a genuine 3-class target (up / down /
        neutral) as required by the Component B calibrated classifier --
        without it, every non-barrier-hit row is forced into up or down even
        when price barely moved, which biases the classifier away from ever
        learning a neutral class. Set to 0.0 to reproduce the old
        (binary-only) behaviour.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'direction' (1.0 = up, -1.0 = down, 0.0 =
        neutral), 'magnitude' (MFE), 'duration' aligned with df.index. Rows
        where no barrier is hit within max_candles AND the vertical barrier
        itself falls beyond the end of the series are dropped, since their
        outcome cannot be observed without look-ahead. All other columns
        (features, regime tags) are preserved for the valid rows.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame index must be sorted ascending; triple-barrier logic "
            "assumes forward-looking windows are chronologically contiguous."
        )
    if not df.index.is_unique:
        raise ValueError(
            "DataFrame index contains duplicate timestamps. apply_triple_barrier "
            "must be called per-asset, before pooling, or forward-looking windows "
            "will span rows from different underlying instances."
        )

    n = len(df)
    directions = np.full(n, np.nan, dtype=float)
    magnitudes = np.full(n, np.nan, dtype=float)
    # BUGFIX (was dtype=int): np.full(n, np.nan, dtype=int) casts NaN into an
    # integer array, which is undefined behaviour in numpy (raises
    # "invalid value encountered in cast" and silently produces garbage
    # sentinel values instead of NaN). Keep as float until valid positions
    # are sliced out below, then cast to int only for the rows we keep.
    durations = np.full(n, np.nan, dtype=float)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df[atr_col].values

    valid_positions: List[int] = []

    for i in range(n):
        entry_price = closes[i]
        atr_val = atrs[i]

        if np.isnan(atr_val) or atr_val <= 0:
            continue

        tp_level = entry_price + tp_mult * atr_val
        sl_level = entry_price - sl_mult * atr_val

        end_idx = min(i + max_candles, n)
        fwd_highs = highs[i + 1:end_idx]
        fwd_lows = lows[i + 1:end_idx]

        hit_tp = np.any(fwd_highs >= tp_level)
        hit_sl = np.any(fwd_lows <= sl_level)

        if hit_tp or hit_sl:
            tp_indices = np.where(fwd_highs >= tp_level)[0]
            sl_indices = np.where(fwd_lows <= sl_level)[0]

            if len(tp_indices) > 0 and len(sl_indices) > 0:
                first_tp = tp_indices[0]
                first_sl = sl_indices[0]
                if first_tp <= first_sl:
                    touch_offset = first_tp + 1
                    direction = 1.0
                else:
                    touch_offset = first_sl + 1
                    direction = -1.0
            elif len(tp_indices) > 0:
                touch_offset = tp_indices[0] + 1
                direction = 1.0
            else:
                touch_offset = sl_indices[0] + 1
                direction = -1.0

            touch_idx = i + touch_offset

            window_highs = highs[i:touch_idx + 1]
            window_lows = lows[i:touch_idx + 1]

            if direction == 1.0:
                mfe = np.max(window_highs) - entry_price
            else:
                mfe = entry_price - np.min(window_lows)

            directions[i] = direction
            magnitudes[i] = max(mfe, 0.0)
            durations[i] = touch_offset
            valid_positions.append(i)
        else:
            touch_idx = i + max_candles
            if touch_idx >= n:
                # Vertical barrier falls beyond the observable series: outcome
                # unknown, must be dropped rather than guessed (horizon bias).
                continue

            exit_price = closes[touch_idx]
            move = exit_price - entry_price
            neutral_band = neutral_zone_atr * atr_val

            if abs(move) < neutral_band:
                direction = 0.0
            else:
                direction = 1.0 if move > 0 else -1.0

            duration = max_candles

            window_highs = highs[i:touch_idx + 1]
            window_lows = lows[i:touch_idx + 1]

            if direction >= 0.0:
                mfe = np.max(window_highs) - entry_price
            else:
                mfe = entry_price - np.min(window_lows)

            directions[i] = direction
            magnitudes[i] = max(mfe, 0.0)
            durations[i] = duration
            valid_positions.append(i)

    labels_df = pd.DataFrame(
        {
            "direction": directions[valid_positions],
            "magnitude": magnitudes[valid_positions],
            "duration": durations[valid_positions].astype(int),
        },
        index=df.index[valid_positions],
    )

    # BUGFIX (was `df.loc[valid_indices]`): valid_positions is a list of
    # positional row offsets (0..n-1), not index labels. On a DatetimeIndex,
    # `.loc[valid_positions]` raises a KeyError (or, worse, on an integer
    # index it would silently select the WRONG rows by label instead of by
    # position). `.iloc` is the only correct accessor here.
    result = df.iloc[valid_positions].copy()
    result["direction"] = labels_df["direction"].values
    result["magnitude"] = labels_df["magnitude"].values
    result["duration"] = labels_df["duration"].values

    return result


def pool_assets(asset_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pools multiple asset DataFrames into a single DataFrame, injecting asset identity
    as a categorical feature.

    Parameters
    ----------
    asset_dfs : Dict[str, pd.DataFrame]
        Mapping of asset name to its feature/OHLCV DataFrame. Each DataFrame
        should already be triple-barrier labeled independently (see
        `apply_triple_barrier`) -- pooling raw price series and labeling
        afterwards would let one asset's forward window read another asset's
        candles.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with an 'asset' column (categorical dtype) and
        sorted DatetimeIndex. The resulting index is NOT guaranteed unique
        (BTC and ETH can share a timestamp) -- downstream consumers must not
        do label-based (`.loc`) slicing against this index; use positional
        (`.iloc`) splitting instead (see `WalkForwardSplitter`).
    """
    if len(asset_dfs) == 0:
        raise ValueError("asset_dfs is empty; nothing to pool.")

    pooled = pd.concat(
        [df.assign(asset=asset_name) for asset_name, df in asset_dfs.items()],
        axis=0,
    )
    pooled.sort_index(inplace=True, kind="mergesort")  # stable sort: preserves
    # per-asset relative order for rows sharing a timestamp, so pooling is
    # deterministic and reproducible across runs.
    pooled["asset"] = pooled["asset"].astype("category")
    return pooled