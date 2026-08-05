"""Feature engineering module using pandas-ta with strict no-lookahead guarantees.

All calculations are explicitly backward-looking. Rolling windows never use 
center=True, shifts are strictly positive (past data only), and every output 
feature is percentile-ranked against a trailing 252-candle history before leaving 
this module. Raw indicator values are never exposed to downstream consumers.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# Configuration constants aligned with MASTER_CONTEXT.md constraints
LOOKBACK_PERCENTILE = 252
STRUCTURAL_N_SHORT = 20
STRUCTURAL_N_LONG = 50

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


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
        raise ValueError("Input DataFrame contains duplicate timestamps. Please deduplicate before passing to feature calculator.")


def _calculate_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate structural market features (HH/HL/LH/LL, BOS, ChoCh, SFP, range/trend, distance to S/R, Wyckoff)."""
    out = pd.DataFrame(index=df.index)
    
    # Higher High / Lower Low flags (N=20, N=50) with tolerance for floating-point precision
    epsilon = 1e-6
    rolling_high_20 = df["high"].rolling(STRUCTURAL_N_SHORT).max()
    rolling_low_20 = df["low"].rolling(STRUCTURAL_N_SHORT).min()
    out["struct_hh_20"] = np.isclose(df["high"], rolling_high_20, atol=epsilon).astype(float)
    out["struct_ll_20"] = np.isclose(df["low"], rolling_low_20, atol=epsilon).astype(float)
    
    rolling_high_50 = df["high"].rolling(STRUCTURAL_N_LONG).max()
    rolling_low_50 = df["low"].rolling(STRUCTURAL_N_LONG).min()
    out["struct_hh_50"] = np.isclose(df["high"], rolling_high_50, atol=epsilon).astype(float)
    out["struct_ll_50"] = np.isclose(df["low"], rolling_low_50, atol=epsilon).astype(float)
    
    # Break of Structure (BOS): Close exceeds recent structural pivot (N=20 or N=50)
    out["struct_bos_bullish"] = ((df["close"] > rolling_high_20.shift(1)) | 
                                 (df["close"] > rolling_high_50.shift(1))).astype(float)
    out["struct_bos_bearish"] = ((df["close"] < rolling_low_20.shift(1)) | 
                                 (df["close"] < rolling_low_50.shift(1))).astype(float)
    
    # Change of Character (ChoCh): Reversal signal after BOS
    out["struct_choch_bullish"] = ((out["struct_bos_bearish"].shift(1) == 1) & 
                                  (out["struct_bos_bullish"] == 1)).astype(float)
    out["struct_choch_bearish"] = ((out["struct_bos_bullish"].shift(1) == 1) & 
                                  (out["struct_bos_bearish"] == 1)).astype(float)
    
    # Swing Failure Pattern (SFP): Price breaks structure but closes back inside
    out["struct_sfp_bullish"] = ((df["low"] < df["low"].shift(1).rolling(STRUCTURAL_N_SHORT).min()) & 
                                (df["close"] > df["open"])).astype(float)
    out["struct_sfp_bearish"] = ((df["high"] > df["high"].shift(1).rolling(STRUCTURAL_N_SHORT).max()) & 
                                (df["close"] < df["open"])).astype(float)
    
    # Range vs Trend State: ADX > 25 implies trend, else range
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    out["struct_trend_state"] = (adx["ADX_14"] > 25).astype(float)
    
    # Distance to Support/Resistance (normalized by ATR)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)["ATR_14"]
    out["struct_dist_to_resist"] = (rolling_high_20 - df["close"]) / (atr + 1e-9)
    out["struct_dist_to_support"] = (df["close"] - rolling_low_20) / (atr + 1e-9)
    
    # Wyckoff Volume-Range Phase (heuristic: high volume + low range = accumulation/distribution)
    price_range = df["high"] - df["low"]
    vol_sma = ta.sma(df["volume"], length=20)
    out["struct_wyckoff_phase"] = np.where(
        (df["volume"] > vol_sma * 1.5) & (price_range < price_range.rolling(STRUCTURAL_N_SHORT).mean()),
        1.0,
        0.0
    )
    
    return out


def _calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility features (ATR percentile, BB width, rolling vol, expansion/compression state)."""
    out = pd.DataFrame(index=df.index)
    
    # ATR(14)
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)["ATR_14"]
    out["vol_atr_14"] = atr
    
    # Bollinger Band Width (normalized by middle band for statistical fidelity)
    bb = ta.bbands(df["close"], length=20, std=2)
    out["vol_bb_width"] = (bb["BBU_20_2.0"] - bb["BBD_20_2.0"]) / (bb["BBM_20_2.0"] + 1e-9)
    
    # 20-period rolling volatility (std of returns)
    returns = df["close"].pct_change()
    out["vol_std_20"] = returns.rolling(20).std()
    
    # ATR Percentile Rank over trailing 252 candles (strictly backward-looking)
    out["vol_atr_pct_rank"] = atr.rolling(LOOKBACK_PERCENTILE).rank(pct=True)
    
    # BB Width Percentile Rank over trailing 252 candles
    out["vol_bb_width_pct_rank"] = out["vol_bb_width"].rolling(LOOKBACK_PERCENTILE).rank(pct=True)
    
    # Volatility State Classification per MASTER_CONTEXT.md thresholds
    out["vol_state_compressed"] = (out["vol_atr_pct_rank"] < 0.3).astype(float)
    out["vol_state_expanding"] = ((out["vol_atr_pct_rank"] >= 0.3) & (out["vol_atr_pct_rank"] <= 0.7)).astype(float)
    out["vol_state_extended"] = (out["vol_atr_pct_rank"] > 0.7).astype(float)
    
    # Spike Detection: ATR doubles within 5 candles (tail-risk event)
    atr_5_shift = atr.shift(5)
    spike_ratio = atr / (atr_5_shift + 1e-9)
    out["vol_spike"] = (spike_ratio >= 2.0).astype(float)
    # Explicitly fill leading warm-up NaNs with 0.0 (no spike) to prevent downstream shape mismatches
    out["vol_spike"].fillna(0.0, inplace=True)
    
    return out


def _calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum features (RSI + ROC, divergence, MACD slope, StochRSI, ADX/DI, multi-horizon returns).
    
    All calculations are strictly backward-looking. Divergence detection uses a 
    vectorised rolling-window proxy to identify swing mismatches without iterative peak-finding.
    """
    out = pd.DataFrame(index=df.index)
    
    # RSI(14) and Rate of Change
    rsi = ta.rsi(df["close"], length=14)
    out["mom_rsi_14"] = rsi
    out["mom_rsi_roc"] = rsi.pct_change()
    
    # RSI Divergence Detection (Bullish & Bearish)
    # Vectorised proxy: compares current price swing against rolling min/max of previous N candles.
    # Bullish: Price makes lower low, but RSI makes higher low.
    price_ll = df["low"].rolling(STRUCTURAL_N_SHORT).min()
    rsi_ll = rsi.rolling(STRUCTURAL_N_SHORT).min()
    out["mom_rsi_div_bullish"] = ((df["low"] < price_ll.shift(1)) & 
                                 (rsi > rsi_ll.shift(1))).astype(float)
    
    # Bearish: Price makes higher high, but RSI makes lower high.
    price_hh = df["high"].rolling(STRUCTURAL_N_SHORT).max()
    rsi_hh = rsi.rolling(STRUCTURAL_N_SHORT).max()
    out["mom_rsi_div_bearish"] = ((df["high"] > price_hh.shift(1)) & 
                                 (rsi < rsi_hh.shift(1))).astype(float)
    
    # MACD Histogram Slope (momentum acceleration/deceleration)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    out["mom_macd_hist_slope"] = macd["MACDh_12_26_9"].diff()
    
    # StochRSI (K line only for responsiveness)
    stochrsi = ta.stochrsi(df["close"], length=14, stoch_length=14)
    out["mom_stochrsi_k"] = stochrsi["STOCHRSIk_14_14"]
    
    # ADX & Directional Indicator Spread (trend strength + directional bias)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    out["mom_adx_14"] = adx["ADX_14"]
    out["mom_di_spread"] = adx["DIP_14"] - adx["DIM_14"]
    
    # Multi-horizon price returns (5, 10, 20 periods)
    for period in [5, 10, 20]:
        out[f"mom_ret_{period}"] = df["close"].pct_change(period)
        
    return out


def _calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume features (z-score, breakout ratio, body ratio, wick direction/magnitude).
    
    All operations are vectorised and strictly backward-looking. Division by zero is 
    guarded against using a small epsilon to prevent silent NaN propagation.
    """
    out = pd.DataFrame(index=df.index)
    
    # Volume Z-Score: deviation from 20-period mean relative to rolling standard deviation
    vol_sma = ta.sma(df["volume"], length=20)
    vol_std = df["volume"].rolling(20).std()
    out["volu_zscore"] = (df["volume"] - vol_sma) / (vol_std + 1e-9)
    
    # Volume-on-Breakout Ratio: volume relative to SMA when price breaks recent structure
    # Breakout defined as closing above/below the rolling max/min of N=20 candles
    prev_highs = df["high"].shift(1).rolling(STRUCTURAL_N_SHORT).max()
    prev_lows = df["low"].shift(1).rolling(STRUCTURAL_N_SHORT).min()
    is_breakout = (df["close"] > prev_highs) | (df["close"] < prev_lows)
    out["volu_on_breakout"] = np.where(is_breakout, df["volume"] / (vol_sma + 1e-9), 0.0)
    
    # Candle Body Ratio: absolute body size relative to total candle range
    body = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"]
    out["volu_body_ratio"] = body / (total_range + 1e-9)
    
    # Wick Direction/Magnitude: upper and lower wick size relative to total range
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    out["volu_upper_wick_mag"] = upper_wick / (total_range + 1e-9)
    out["volu_lower_wick_mag"] = lower_wick / (total_range + 1e-9)
    
    return out


def _apply_trailing_percentile_rank(df: pd.DataFrame, window: int = LOOKBACK_PERCENTILE) -> pd.DataFrame:
    """Apply trailing percentile ranking to all continuous numeric columns.
    
    Strictly backward-looking: each value is ranked against the preceding `window` candles.
    Returns values scaled between 0 and 1. NaNs are preserved for warm-up periods.
    Binary flags, state classifications, and already-percentile-ranked columns are excluded 
    to preserve their mathematical properties and prevent distribution distortion.
    """
    # Deterministic exclusion list for binary/state/already-ranked features
    exclude_patterns = [
        "_pct_rank", "_state_", "_spike", 
        "struct_hh_", "struct_ll_", "struct_bos_", "struct_choch_", 
        "struct_sfp_", "struct_trend_state", "struct_wyckoff_phase", 
        "mom_rsi_div_"  # Covers both bullish and bearish divergence flags
    ]
    exclude_cols = [c for c in df.columns if any(pat in c for pat in exclude_patterns)]
    
    continuous_cols = [c for c in df.columns if c not in exclude_cols]
    if not continuous_cols:
        return df

    ranked = df.copy()
    ranked[continuous_cols] = ranked[continuous_cols].rolling(window=window).rank(pct=True)
    return ranked


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and normalize technical features for OHLCV data.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] 
            and a DatetimeIndex. Must be sorted chronologically.
            
    Returns:
        DataFrame containing fixed-length, percentile-ranked feature vectors.
        All continuous features are scaled between 0 and 1 based on trailing 252-candle history.
        NaNs are preserved where insufficient history exists for warm-up periods.
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns.
        TypeError: If index is not a DatetimeIndex.
    """
    _validate_input(df)
    
    # Work on a copy to guarantee raw data immutability per MASTER_CONTEXT.md
    data = df.copy()
    
    # Enforce chronological sorting on the working copy to guarantee strictly backward-looking operations
    if not data.index.is_monotonic_increasing:
        data.sort_index(inplace=True)
        
    # Calculate raw features across all requested categories
    struct_feats = _calculate_structural_features(data)
    vol_feats = _calculate_volatility_features(data)
    mom_feats = _calculate_momentum_features(data)
    volu_feats = _calculate_volume_features(data)
    
    # Explicitly reindex to guarantee strict 1:1 temporal mapping and prevent silent misalignment
    struct_feats = struct_feats.reindex(data.index)
    vol_feats = vol_feats.reindex(data.index)
    mom_feats = mom_feats.reindex(data.index)
    volu_feats = volu_feats.reindex(data.index)
    
    # Combine all features into a single DataFrame
    combined = pd.concat([struct_feats, vol_feats, mom_feats, volu_feats], axis=1)
    
    # Apply trailing percentile ranking to ensure scale consistency across assets/timeframes.
    # No raw values leave this module.
    ranked_features = _apply_trailing_percentile_rank(combined)
    
    # Align index explicitly and drop rows where warm-up periods produce all NaNs
    ranked_features.index = data.index
    return ranked_features.dropna(how="all")
