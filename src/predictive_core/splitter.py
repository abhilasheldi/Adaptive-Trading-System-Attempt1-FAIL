"""Walk-forward temporal splitter for strict train/validation/test partitioning."""

import numpy as np
import pandas as pd
from typing import Generator, Optional, Tuple


class WalkForwardSplitter:
    """
    Generates non-overlapping train, validation, and test windows in a strict
    chronological walk-forward manner. Guarantees zero temporal leakage across folds.

    IMPORTANT: this splitter yields POSITIONAL (row-offset) index arrays, not
    label-based (`.loc`) index values. Use `.iloc[positions]` to materialize a
    fold, never `.loc[...]`.

    Why: `pool_assets` concatenates BTC and ETH into one DataFrame sorted by
    timestamp, and BTC/ETH rows can share the exact same timestamp. That makes
    the DataFrame's DatetimeIndex non-unique. If a caller took a slice of
    index *labels* (the previous behaviour of this class) and then did
    `pooled_df.loc[those_labels]`, pandas `.loc` matches ALL rows carrying any
    of those label values -- not just the intended slice. Concretely: if
    timestamp T's BTC row falls in the train window and T's ETH row falls in
    the validation window (a real possibility at any fold boundary), then
    `.loc[train_labels]` would silently pull the ETH validation row into the
    training set too, because it shares a label with an accepted BTC
    timestamp. That is a genuine train/validation/test leak, and it is silent
    -- no exception, no warning, just contaminated folds. Positional slicing
    is immune to this because row *position* is always unique regardless of
    duplicate timestamp values.
    """

    def __init__(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        step_size: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        train_size : int
            Number of samples in the training window.
        val_size : int
            Number of samples in the validation window.
        test_size : int
            Number of samples in the test window.
        step_size : int, optional
            Step size for advancing the window. Defaults to val_size + test_size
            to ensure strict separation between consecutive folds.
        """
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError("train_size, val_size, and test_size must all be positive.")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size or (val_size + test_size)

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yields tuples of (train_positions, val_positions, test_positions) for
        each fold, as arrays of integer row positions (for use with `.iloc`).

        Parameters
        ----------
        df : pd.DataFrame
            Chronologically sorted DataFrame with a DatetimeIndex. The index
            is NOT required to be unique (pooled multi-asset frames are
            supported), but it must be monotonically non-decreasing, or a
            positional split would no longer correspond to a temporal split.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Integer position arrays for train, validation, and test sets.
            Each window is strictly bounded to prevent any forward-looking
            data leakage. Consecutive folds' train windows may legitimately
            overlap with earlier folds' validation/test windows (standard
            expanding walk-forward reuse of now-past data); validation and
            test windows never overlap with each other, within or across
            folds.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "DataFrame index must be sorted ascending (non-decreasing) for "
                "a positional split to correspond to a temporal split."
            )

        n = len(df)
        start = 0
        fold = 0

        while start + self.train_size + self.val_size + self.test_size <= n:
            train_end = start + self.train_size
            val_end = train_end + self.val_size
            test_end = val_end + self.test_size

            train_pos = np.arange(start, train_end)
            val_pos = np.arange(train_end, val_end)
            test_pos = np.arange(val_end, test_end)

            self._assert_strict_ordering(df, train_pos, val_pos, test_pos, fold)

            yield (train_pos, val_pos, test_pos)

            start += self.step_size
            fold += 1

    @staticmethod
    def _assert_strict_ordering(
        df: pd.DataFrame,
        train_pos: np.ndarray,
        val_pos: np.ndarray,
        test_pos: np.ndarray,
        fold: int,
    ) -> None:
        """
        Defensive, non-negotiable guard: re-derives chronological ordering
        from the actual timestamps (not just position arithmetic) and raises
        if it does not hold. Cheap relative to model training, and it turns a
        silent leak into a loud failure if this class is ever refactored
        carelessly.
        """
        idx = df.index
        if len(train_pos) == 0 or len(val_pos) == 0 or len(test_pos) == 0:
            raise ValueError(f"Fold {fold}: empty split window generated.")

        train_max_ts = idx[train_pos].max()
        val_min_ts = idx[val_pos].min()
        val_max_ts = idx[val_pos].max()
        test_min_ts = idx[test_pos].min()

        if train_max_ts > val_min_ts:
            raise AssertionError(
                f"Fold {fold}: train window's latest timestamp ({train_max_ts}) "
                f"is after validation window's earliest timestamp ({val_min_ts}). "
                f"Temporal ordering violated -- refusing to yield this fold."
            )
        if val_max_ts > test_min_ts:
            raise AssertionError(
                f"Fold {fold}: validation window's latest timestamp ({val_max_ts}) "
                f"is after test window's earliest timestamp ({test_min_ts}). "
                f"Temporal ordering violated -- refusing to yield this fold."
            )

        # Positions are constructed from disjoint contiguous ranges, so
        # intersection is impossible by construction -- but assert it anyway
        # since this is the single most important invariant in the entire
        # pipeline and the cost of checking is negligible.
        if (
            set(train_pos.tolist()) & set(val_pos.tolist())
            or set(val_pos.tolist()) & set(test_pos.tolist())
            or set(train_pos.tolist()) & set(test_pos.tolist())
        ):
            raise AssertionError(f"Fold {fold}: overlapping row positions across splits.")