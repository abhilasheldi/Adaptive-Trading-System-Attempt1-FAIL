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

        IMPORTANT, checked directly against source docs (2026-08-13): the
        2.0/1.0 default here is NOT a MASTER_CONTEXT-specified number.
        MASTER_CONTEXT.docx, the Adaptive Probabilistic Trade Management
        System spec, and the Prompt Playbook were all searched for
        "tp_mult"/"sl_mult"/"triple barrier"/"reward:risk" and none specify
        this ratio -- the Prompt Playbook's actual Component B build
        instruction only says "a calibrated direction classifier," with no
        mandated labeling method or barrier geometry at all. The one
        reward:risk figure that does appear in the source docs (an
        illustrative "Reward:Risk: 1.2:1" in a mocked-up live-trade output
        example) describes a DIFFERENT, later-pipeline-stage concept (a
        live trade's actual entry/stop/target after the full EV/sizing
        pipeline) and isn't even the same number. 2.0/1.0 is an
        implementation-time default with no cited empirical or spec-driven
        justification anywhere in this project's history.

        This matters mechanically, not just as a provenance footnote: for a
        directionless (no real edge) local random walk, the classic
        gambler's-ruin first-passage result gives P(hit TP first) =
        sl_mult / (tp_mult + sl_mult) = 1/3 for this default -- i.e. a
        ~33%/67% up/down split is exactly what pure noise would produce
        under THIS specific ratio, with zero real market signal required.
        Empirically observed on real BTC 2023-2026: 33.6% up / 65.8% down
        -- matching the noise-only prediction almost exactly. See
        docs/project_state.md (2026-08-13 entries) for the full
        investigation this derives from. Do not casually retune this value
        chasing better backtest numbers -- because it defines what the
        label itself means, tuning it against any data that later becomes
        part of a walk-forward test/val fold is a more fundamental leak
        than ordinary hyperparameter tuning (it changes the ground truth
        the split is built around, not just a model fit to that ground
        truth). Any change should be decided from first principles (e.g.
        empirical excursion distributions from a data slice that is never
        touched again) or via a pre-registered, held-out comparison across
        a small fixed candidate set -- not an ad hoc sweep.
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
        Rows where no barrier is hit within max_candles AND the vertical
        barrier itself falls beyond the end of the series are dropped, since
        their outcome cannot be observed without look-ahead. All other
        columns (features, regime tags) are preserved for the valid rows.

        Label columns:

        direction        1.0 = up, -1.0 = down, 0.0 = neutral
        magnitude        Maximum Favourable Excursion, in price units. NOT a
                         realised P&L -- it is the best the trade ever looked,
                         which no exit rule can actually capture.
        duration         bars from t0 to t1 (== t1 - t0 in bar counts)

        Stage A.1 metadata (BUILD_PLAN_v2 A.1):

        t0               event time. Also the index; emitted as a column too
                         because `pool_assets` yields a non-unique index.
        t1               barrier-touch or vertical-barrier time. THE critical
                         addition: purging is impossible without it, and its
                         absence is why the existing temporal partitioning
                         leaks -- it splits on t0 (a feature timestamp) while
                         labels stay unresolved until t1.
        barrier_touched  'tp' | 'sl' | 'vertical'
        realised_return  signed fractional return from t0 to t1, to a LONG
                         position. This is the return an exit rule could
                         actually have captured, unlike `magnitude`.

        `realised_return` pricing, stated because it is a modelling choice
        and not a fact: tp/sl resolve AT THE BARRIER LEVEL (a resting order
        fills where it sits), the vertical barrier resolves at the horizon
        bar's CLOSE (a market exit). Real fills are worse than this
        (slippage, gapping straight through a stop); that cost belongs to the
        execution model, not the label.

        CONSEQUENCE, VERIFIED ON REAL DATA AND IMPORTANT FOR STAGE D:
        conditional on `barrier_touched`, realised_return carries NO
        information about the price path. Every 'tp' row is exactly
        +tp_mult*ATR/entry and every 'sl' row exactly -sl_mult*ATR/entry --
        confirmed to 1e-16 across all 77,751 barrier-resolved BTC rows. The
        values differ between rows only because ATR/entry differs, never
        because the trade went better or worse.

        So realised_return here is the barrier geometry restated in return
        space, not a measurement of trade quality. A "top-decile realised EV"
        computed on these labels would be ranking bars by their ATR/price
        ratio, not by skill (BUILD_PLAN_v2 D.3 makes top-decile realised EV
        the PRIMARY Stage D metric -- it needs a realised return that
        responds to the path, e.g. one priced against actual fills, or the
        metric measures volatility exposure instead).

        LONG-ONLY CONVENTION: `direction` records which barrier a long
        position would have hit first, so `realised_return` is signed from a
        long's perspective throughout. A short book would need its own
        labelling pass; this function does not produce one.
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

    # --- Stage A.1 label metadata -------------------------------------- #
    # `t1_positions` holds the POSITIONAL row offset of each label's
    # resolution bar, resolved to a timestamp at the end. Positions, not
    # timestamps, because the vertical barrier is defined in bars and the
    # index may be irregular (exchange outages leave real gaps -- see
    # data_engine's gap classification). Converting at the end means t1 is
    # always an actual observed bar timestamp, never an extrapolated one.
    t1_positions = np.full(n, -1, dtype=np.int64)
    barrier_touched = np.empty(n, dtype=object)
    realised_returns = np.full(n, np.nan, dtype=float)

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

            # Realised return is priced AT THE BARRIER LEVEL, not at the
            # touching bar's close. A stop/target is a resting order: it
            # fills where it sits, not where the bar happens to close. Using
            # the close would credit a stopped-out long with whatever
            # recovery happened after the stop was already taken out --
            # optimistic by construction. Real fills are worse than this
            # (slippage, gaps through the level); that cost belongs to the
            # execution model, not the label.
            t1_positions[i] = touch_idx
            if direction == 1.0:
                barrier_touched[i] = "tp"
                realised_returns[i] = (tp_level - entry_price) / entry_price
            else:
                barrier_touched[i] = "sl"
                realised_returns[i] = (sl_level - entry_price) / entry_price
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

            # Vertical barrier: no resting order was hit, so the position is
            # closed at the market on the horizon bar. Its close IS the fill
            # here, unlike the tp/sl branch above.
            t1_positions[i] = touch_idx
            barrier_touched[i] = "vertical"
            realised_returns[i] = (exit_price - entry_price) / entry_price
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

    # --- Stage A.1 label metadata (BUILD_PLAN_v2 A.1) ------------------- #
    # t1 is the critical one: purging is impossible without it. The existing
    # temporal partitioning splits on t0 (the feature timestamp) alone, so a
    # training sample whose label is still unresolved at the fold boundary
    # keeps leaking across it. `duration` already encoded this offset, but
    # only as a bar COUNT -- unusable for purging without resolving it to an
    # actual timestamp, which is what this does.
    #
    # t0 is emitted as a COLUMN as well as being the index, deliberately:
    # `pool_assets` produces a non-unique index (BTC and ETH share
    # timestamps), so any code that needs a per-sample event time after
    # pooling cannot rely on the index alone.
    valid_pos_arr = np.asarray(valid_positions, dtype=np.int64)
    result["t0"] = df.index[valid_pos_arr]
    result["t1"] = df.index[t1_positions[valid_pos_arr]]
    result["barrier_touched"] = barrier_touched[valid_pos_arr]
    result["realised_return"] = realised_returns[valid_pos_arr]

    # Internal consistency: t1 must be the bar `duration` steps after t0.
    # Asserted rather than assumed, because the tp/sl and vertical branches
    # compute `touch_idx` by different routes and a divergence between them
    # would silently corrupt every purge boundary downstream.
    expected_t1_pos = valid_pos_arr + durations[valid_pos_arr].astype(np.int64)
    if not np.array_equal(expected_t1_pos, t1_positions[valid_pos_arr]):
        raise AssertionError(
            "t1 does not equal t0 + duration for every label -- the barrier "
            "branches disagree about the resolution bar. Purging built on "
            "this would be silently wrong."
        )

    # Record the max_candles this call actually used, so a downstream
    # consumer that ALSO needs to know this horizon (any splitter that
    # purges label-horizon overlap at fold boundaries must be configured
    # with a matching label_horizon) can cross-check itself against it
    # instead of trusting a separately-configured number to happen to
    # agree. This becomes load-bearing at Stage A.3, where Purged K-Fold
    # needs the real per-sample t1. Verified that pandas propagates .attrs through
    # .iloc/.copy/.assign/.drop/boolean .loc and through pd.concat when all
    # inputs agree -- and safely empties .attrs on concat when inputs
    # DISAGREE, rather than silently picking one, which is why a consumer
    # of this metadata must treat its absence as "couldn't check", not
    # "no mismatch".
    result.attrs["label_horizon"] = max_candles

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