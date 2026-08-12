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
) -> pd.DataFrame:
    """
    Applies triple-barrier labeling to derive direction, magnitude (MFE), and duration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'close', 'high', 'low' and the specified ATR column.
        Index must be a DatetimeIndex or monotonic integer index.
    atr_col : str
        Column name for ATR(14) values.
    tp_mult : float
        Take-profit multiplier relative to ATR.
    sl_mult : float
        Stop-loss multiplier relative to ATR.
    max_candles : int
        Vertical barrier (maximum holding period in candles).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'direction', 'magnitude', 'duration' aligned with df.index.
        Rows where no barrier is hit within max_candles are dropped to prevent 
        horizon-bias contamination. All other columns (features, regime tags) are 
        preserved for the valid rows.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
        
    n = len(df)
    directions = np.full(n, np.nan, dtype=float)
    magnitudes = np.full(n, np.nan, dtype=float)
    durations = np.full(n, np.nan, dtype=int)
    
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df[atr_col].values
    
    valid_indices: List[int] = []
    
    for i in range(n):
        entry_price = closes[i]
        atr_val = atrs[i]
        
        if np.isnan(atr_val) or atr_val <= 0:
            continue
            
        tp_level = entry_price + tp_mult * atr_val
        sl_level = entry_price - sl_mult * atr_val
        
        end_idx = min(i + max_candles, n)
        fwd_highs = highs[i+1:end_idx]
        fwd_lows = lows[i+1:end_idx]
        
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
            
            window_highs = highs[i:touch_idx+1]
            window_lows = lows[i:touch_idx+1]
            
            if direction == 1.0:
                mfe = np.max(window_highs) - entry_price
            else:
                mfe = entry_price - np.min(window_lows)
                
            directions[i] = direction
            magnitudes[i] = max(mfe, 0.0)
            durations[i] = touch_offset
            valid_indices.append(i)
        else:
            touch_idx = i + max_candles
            if touch_idx >= n:
                continue
                
            exit_price = closes[touch_idx]
            direction = 1.0 if exit_price > entry_price else -1.0
            duration = max_candles
            
            window_highs = highs[i:touch_idx+1]
            window_lows = lows[i:touch_idx+1]
            
            if direction == 1.0:
                mfe = np.max(window_highs) - entry_price
            else:
                mfe = entry_price - np.min(window_lows)
                
            directions[i] = direction
            magnitudes[i] = max(mfe, 0.0)
            durations[i] = duration
            valid_indices.append(i)
            
    labels_df = pd.DataFrame({
        "direction": directions[valid_indices],
        "magnitude": magnitudes[valid_indices],
        "duration": durations[valid_indices],
    }, index=df.index[valid_indices])
    
    # Preserve all original columns (features, regime tags, asset identity) for valid rows
    result = df.loc[valid_indices].copy()
    result["direction"] = labels_df["direction"]
    result["magnitude"] = labels_df["magnitude"]
    result["duration"] = labels_df["duration"]
    
    return result


def pool_assets(asset_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pools multiple asset DataFrames into a single DataFrame, injecting asset identity
    as a categorical feature.
    
    Parameters
    ----------
    asset_dfs : Dict[str, pd.DataFrame]
        Mapping of asset name to its feature/OHLCV DataFrame.
        
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with an 'asset' column and sorted DatetimeIndex.
        Macro/volatility regime tags present in the input DataFrames are preserved 
        for downstream per-regime calibration monitoring.
    """
    pooled = pd.concat(
        [df.assign(asset=asset_name) for asset_name, df in asset_dfs.items()],
        axis=0,
    )
    pooled.sort_index(inplace=True)
    return pooled
