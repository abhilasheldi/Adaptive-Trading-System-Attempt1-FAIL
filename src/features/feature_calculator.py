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


def _calculate_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate structural market features (HH/HL/LH/LL, BOS, ChoCh, SFP, range/trend, distance to S/R, Wyckoff)."""
    out = pd.DataFrame(index=df.index)
    
    # Higher High / Lower Low flags (N=20, N=50)
    rolling_high_20 = df["high"].rolling(STRUCTURAL_N_SHORT).max()
    rolling_low_20 = df["low"].rolling(STRUCTURAL_N_SHORT).min()
    out["struct_hh_20"] = (df["high"] == rolling_high_20).astype(float)
    out["struct_ll_20"] = (df["low"] == rolling_low_20).astype(float)
    
    rolling_high_50 = df["high"].rolling(STRUCTURAL_N_LONG).max()
    rolling_low_50 = df["low"].rolling(STRUCTURAL_N_LONG).min()
    out["struct_hh_50"] = (df["high"] == rolling_high_50).astype(float)
    out["struct_ll_50"] = (df["low"] == rolling_low_50).astype(float)
    
    # Break of Structure (BOS): Close above/below recent swing extremes
    out["struct_bos_bullish"] = (df["close"] > df["high"].shift(1)).astype(float)
    out["struct_bos_bearish"] = (df["close"] < df["low"].shift(1)).astype(float)
    
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
    
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)["ATR_14"]
    out["vol_atr_14"] = atr
    
    bb = ta.bbands(df["close"], length=20, std=2)
    out["vol_bb_width"] = (bb["BBU_20_2.0"] - bb["BBD_20_2.0"]) / (df["close"] + 1e-9)
    
    # 20-period volatility (std of returns)
    returns = df["close"].pct_change()
    out["vol_std_20"] = returns.rolling(20).std()
    
    # Expansion/Compression state based on ATR percentile vs trailing history
    atr_pct = atr.rolling(LOOKBACK_PERCENTILE).rank(pct=True)
    out["vol_compressed"] = (atr_pct < 0.3).astype(float)
    out["vol_expanding"] = ((atr_pct >= 0.3) & (atr_pct <= 0.7)).astype(float)
    out["vol_extended"] = (atr_pct > 0.7).astype(float)
    
    return out


def _calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum features (RSI + ROC, divergence, MACD slope, StochRSI, ADX/DI, multi-horizon returns)."""
    out = pd.DataFrame(index=df.index)
    
    rsi = ta.rsi(df["close"], length=14)
    out["mom_rsi_14"] = rsi
    out["mom_rsi_roc"] = rsi.pct_change()
    
    # RSI Divergence (simplified: price makes lower low but RSI makes higher low)
    price_ll = df["low"].rolling(STRUCTURAL_N_SHORT).min()
    rsi_ll = rsi.rolling(STRUCTURAL_N_SHORT).min()
    out["mom_rsi_div_bullish"] = ((df["low"] < price_ll.shift(1)) & 
                                 (rsi > rsi_ll.shift(1))).astype(float)
    
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    out["mom_macd_hist_slope"] = macd["MACDh_12_26_9"].diff()
    
    stochrsi = ta.stochrsi(df["close"], length=14, stoch_length=14)
    out["mom_stochrsi_k"] = stochrsi["STOCHRSIk_14_14"]
    
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    out["mom_adx_14"] = adx["ADX_14"]
    out["mom_di_spread"] = adx["DIP_14"] - adx["DIM_14"]
    
    # Multi-horizon returns
    for period in [5, 10, 20]:
        out[f"mom_ret_{period}"] = df["close"].pct_change(period)
        
    return out


def _calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume features (z-score, breakout ratio, body ratio, wick direction/magnitude)."""
    out = pd.DataFrame(index=df.index)
    
    vol_sma = ta.sma(df["volume"], length=20)
    vol_std = df["volume"].rolling(20).std()
    out["volu_zscore"] = (df["volume"] - vol_sma) / (vol_std + 1e-9)
    
    # Volume on breakout ratio
    high_break = df["high"] > df["high"].shift(1).rolling(STRUCTURAL_N_SHORT).max()
    out["volu_on_breakout"] = np.where(high_break, df["volume"] / (vol_sma + 1e-9), 0.0)
    
    # Candle body ratio
    body = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"]
    out["volu_body_ratio"] = body / (total_range + 1e-9)
    
    # Wick direction/magnitude
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    out["volu_upper_wick_mag"] = upper_wick / (total_range + 1e-9)
    out["volu_lower_wick_mag"] = lower_wick / (total_range + 1e-9)
    
    return out


def _apply_trailing_percentile_rank(df: pd.DataFrame, window: int = LOOKBACK_PERCENTILE) -> pd.DataFrame:
    """Apply trailing percentile ranking to all numeric columns.
    
    Strictly backward-looking: each value is ranked against the preceding `window` candles.
    Returns values scaled between 0 and 1. NaNs are preserved for warm-up periods.
    """
    # rolling().rank(pct=True) is vectorised, strictly backward-looking, and avoids 
    # the performance penalty of apply() with custom lambdas.
    ranked = df.rolling(window=window).rank(pct=True)
    return ranked


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and normalize technical features for OHLCV data.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] 
            and a DatetimeIndex. Must be sorted chronologically.
            
    Returns:
        DataFrame containing fixed-length, percentile-ranked feature vectors.
        All features are scaled between 0 and 1 based on trailing 252-candle history.
        NaNs are preserved where insufficient history exists for warm-up periods.
        
    Raises:
        ValueError: If input DataFrame is empty or missing required columns.
        TypeError: If index is not a DatetimeIndex.
    """
    _validate_input(df)
    
    # Work on a copy to guarantee raw data immutability per MASTER_CONTEXT.md
    data = df.copy()
    
    # Calculate raw features across all requested categories
    struct_feats = _calculate_structural_features(data)
    vol_feats = _calculate_volatility_features(data)
    mom_feats = _calculate_momentum_features(data)
    volu_feats = _calculate_volume_features(data)
    
    # Combine all features into a single DataFrame
    combined = pd.concat([struct_feats, vol_feats, mom_feats, volu_feats], axis=1)
    
    # Apply trailing percentile ranking to ensure scale consistency across assets/timeframes.
    # No raw values leave this module.
    ranked_features = _apply_trailing_percentile_rank(combined)
    
    # Align index explicitly and drop rows where warm-up periods produce all NaNs
    ranked_features.index = data.index
    return ranked_features.dropna(how="all")
