"""
Minimal, diagnostic-only proxy for MASTER_CONTEXT's Layer 5 ("Liquidity
Trap heuristic") -- NOT a replacement for Layer 5 and NOT production code.
`macro_mtf_liquidity_vol/` (the real Layer 3/4/5 home) is still empty.

This exists to answer one narrow question cheaply, before committing to the
full Phase 2 build-out: does restricting the predictive core's training/
evaluation data to candles that look like MASTER_CONTEXT's hypothesized
"post-compression liquidity-sweep" setup change the feature-signal ceiling
found in the unconditioned Stage 4 check (docs/project_state.md, 2026-08-13
session)? See that entry for full context.

Built entirely from features `feature_calculator.py` already computes --
deliberately, so this requires no new indicator logic, only a filter:
  - "post-compression": `vol_state_compressed` was true at any point in the
    trailing `compression_lookback` candles (default 5) -- a real Layer 5
    would define this far more rigorously (compression duration, ATR
    contraction slope, etc.); this is a blunt proxy, chosen and empirically
    checked (see project_state.md) to be meaningfully selective (~2% of
    candles) without being so strict the resulting sample is unusably small.
  - "liquidity sweep": `struct_sfp_bullish` or `struct_sfp_bearish` fires on
    the current candle -- a Swing Failure Pattern (wick beyond a recent
    swing high/low, immediately rejected back inside range) is exactly a
    stop-hunt/liquidity-sweep pattern, already computed by
    `_calculate_structural_features`.

Both inputs are already point-in-time-safe (feature_calculator's own
no-lookahead guarantee); the rolling lookback here is trailing-only
(inclusive of the current row), so this filter introduces no lookahead.
"""

import pandas as pd


def compression_sweep_mask(df: pd.DataFrame, compression_lookback: int = 5) -> pd.Series:
    """
    Boolean mask, aligned to `df.index`, True on candles where a liquidity-
    sweep pattern (SFP) fires AND compression occurred within the trailing
    `compression_lookback` candles (inclusive of the current row).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'vol_state_compressed', 'struct_sfp_bullish',
        'struct_sfp_bearish' -- i.e. `feature_columns()` output from
        `predictive_core.dataset.build_asset_modeling_frame`.
    compression_lookback : int
        Trailing window (candles, inclusive of current) checked for recent
        compression. Not a MASTER_CONTEXT spec number -- a diagnostic
        choice, empirically checked against a few values (3/5/10/20) to
        land on one that is meaningfully selective; see project_state.md.
    """
    required = {"vol_state_compressed", "struct_sfp_bullish", "struct_sfp_bearish"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    recent_compression = df["vol_state_compressed"].rolling(compression_lookback, min_periods=1).max() > 0
    sweep_now = (df["struct_sfp_bullish"] > 0) | (df["struct_sfp_bearish"] > 0)
    return recent_compression & sweep_now
