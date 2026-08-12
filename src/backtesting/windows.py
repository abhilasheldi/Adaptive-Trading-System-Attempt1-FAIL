"""
Calendar-based walk-forward fold generator for the backtesting harness.

Distinct from `predictive_core.splitter.WalkForwardSplitter`, which yields
fixed ROW-COUNT windows for ML cross-validation. This module yields
CALENDAR-TIME windows (years/months) for strategy backtesting, per
MASTER_CONTEXT constraint 4 (walk-forward validation only: train, then
validate, then test, rolling forward, never overlapping within a fold) and
the explicit window sizing this harness was built to: minimum 2-year
training window, 6-month validation window, 6-month test window never used
in training/tuning, rolling forward 3 months, minimum 4 complete folds.

Note on cross-fold test overlap: with a 3-month step and a 6-month test
window, consecutive folds' test windows overlap by 3 months. That is
expected and is not a leak -- the constraint that matters is per-fold
isolation (a given fold's test window is never used to fit or tune that
same fold's model), which this module enforces via non-overlapping
train/val/test *within* each fold. Two different folds legitimately use two
independently-fit models to evaluate overlapping calendar periods.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldWindow:
    """
    Half-open calendar windows: `*_start` is inclusive, `*_end` is exclusive.
    train_end == val_start and val_end == test_start by construction, so the
    three windows partition [train_start, test_end) with no gaps and no
    overlap.
    """

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.train_start) & (index < self.train_end))

    def val_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.val_start) & (index < self.val_end))

    def test_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.test_start) & (index < self.test_end))


def generate_walk_forward_windows(
    index: pd.DatetimeIndex,
    train_size: Optional[pd.DateOffset] = None,
    val_size: Optional[pd.DateOffset] = None,
    test_size: Optional[pd.DateOffset] = None,
    step: Optional[pd.DateOffset] = None,
    min_folds: int = 4,
    min_train_years: float = 2.0,
) -> List[FoldWindow]:
    """
    Generates calendar-time walk-forward folds covering `index`'s range.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Chronologically sorted timestamps of the data to be backtested.
    train_size, val_size, test_size, step : pd.DateOffset, optional
        Calendar-length of each segment and how far the window advances
        each fold. Default to MASTER_CONTEXT's exact numbers (2 years / 6
        months / 6 months / 3 months) when omitted. `pd.DateOffset` (not
        `pd.Timedelta`) is used deliberately so "1 year" / "6 months" mean
        calendar years/months, not a fixed day count that would drift
        across leap years.
    min_folds : int
        MASTER_CONTEXT requires a minimum of 4 complete folds. Silently
        returning fewer would let an under-powered backtest look validated
        when it isn't -- this raises instead.
    min_train_years : float
        Hard floor on `train_size`, enforcing "minimum 2-year training
        window" even if a caller passes a shorter `train_size` by mistake.

    Returns
    -------
    List[FoldWindow]
        At least `min_folds` folds, oldest first.
    """
    train_size = train_size if train_size is not None else pd.DateOffset(years=2)
    val_size = val_size if val_size is not None else pd.DateOffset(months=6)
    test_size = test_size if test_size is not None else pd.DateOffset(months=6)
    step = step if step is not None else pd.DateOffset(months=3)

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex.")
    if len(index) == 0:
        raise ValueError("index is empty.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending.")
    if min_folds < 1:
        raise ValueError("min_folds must be at least 1.")

    train_years_equiv = pd.Timestamp("2000-01-01") + train_size
    train_years_equiv = (train_years_equiv - pd.Timestamp("2000-01-01")).days / 365.25
    if train_years_equiv < min_train_years - 1e-9:
        raise ValueError(
            f"train_size ({train_size}) is shorter than the required minimum "
            f"of {min_train_years} years. MASTER_CONTEXT requires a minimum "
            f"2-year training window."
        )

    data_start = index.min()
    data_end = index.max()

    folds: List[FoldWindow] = []
    fold = 0
    train_start = data_start
    while True:
        train_end = train_start + train_size
        val_start = train_end
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + test_size

        if test_end > data_end:
            break

        folds.append(
            FoldWindow(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start = train_start + step
        fold += 1

    if len(folds) < min_folds:
        raise ValueError(
            f"Only {len(folds)} complete walk-forward fold(s) fit within the "
            f"available data range ({data_start} .. {data_end}) given "
            f"train_size={train_size}, val_size={val_size}, test_size={test_size}, "
            f"step={step}. MASTER_CONTEXT requires a minimum of {min_folds} "
            f"complete folds -- provide more historical data or shrink the "
            f"window sizes (test-driven configurations may reduce these, but "
            f"production runs must keep the spec defaults)."
        )
    return folds
