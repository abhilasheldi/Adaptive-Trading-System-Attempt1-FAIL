"""Walk-forward temporal splitter for strict train/validation/test partitioning."""

import pandas as pd
from typing import Generator, Optional, Tuple


class WalkForwardSplitter:
    """
    Generates non-overlapping train, validation, and test windows in a strict 
    chronological walk-forward manner. Guarantees zero temporal leakage across folds.
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
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size or (val_size + test_size)

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.Index, pd.Index, pd.Index], None, None]:
        """
        Yields tuples of (train_index, val_index, test_index) for each fold.
        
        Parameters
        ----------
        df : pd.DataFrame
            Chronologically sorted DataFrame with a DatetimeIndex.
            
        Yields
        ------
        Tuple[pd.Index, pd.Index, pd.Index]
            Index slices for train, validation, and test sets. Each window is 
            strictly bounded to prevent any forward-looking data leakage.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
            
        n = len(df)
        start = 0
        
        while start + self.train_size + self.val_size + self.test_size <= n:
            train_end = start + self.train_size
            val_end = train_end + self.val_size
            test_end = val_end + self.test_size
            
            yield (
                df.index[start:train_end],
                df.index[train_end:val_end],
                df.index[val_end:test_end],
            )
            
            start += self.step_size
