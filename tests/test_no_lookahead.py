"""
Reusable pytest helpers for enforcing strict temporal boundaries and preventing lookahead bias.
"""

from typing import Callable, Union

import numpy as np
import pandas as pd


def assert_no_lookahead_bias(
    df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]],
    timestamp_col: str = "timestamp",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Asserts that a feature calculation function does not use future data.

    This helper iterates through each row of the provided dataframe, computes the feature
    using only data up to and including that row, and verifies that the result matches
    the value computed on the full dataset. If any discrepancy is found, it raises an
    AssertionError indicating lookahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing historical data and a timestamp column.
    feature_func : Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]]
        A callable that accepts a DataFrame and returns the computed feature(s).
    timestamp_col : str, default "timestamp"
        The name of the column representing timestamps.
    rtol : float, default 1e-5
        Relative tolerance for numerical comparisons.
    atol : float, default 1e-8
        Absolute tolerance for numerical comparisons.

    Raises
    ------
    AssertionError
        If the feature value at any row differs when computed on a sliced dataset versus
        the full dataset, indicating potential lookahead bias.
    """
    # Ensure chronological order and reset index for consistent slicing
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Compute feature on the complete dataset
    full_result = feature_func(df_sorted)
    
    # Normalize output to DataFrame for uniform handling
    if isinstance(full_result, pd.Series):
        full_df = full_result.to_frame()
    else:
        full_df = full_result.copy()
        
    for i in range(1, len(df_sorted)):
        # Skip rows where the full result contains NaNs (e.g., rolling window warm-up)
        if full_df.iloc[i].isna().any():
            continue
            
        # Slice dataset to include only data available up to the current timestamp
        df_window = df_sorted.iloc[:i + 1].copy()
        window_result = feature_func(df_window)
        
        # Normalize window result
        if isinstance(window_result, pd.Series):
            window_df = window_result.to_frame()
        else:
            window_df = window_result.copy()
            
        # Compare the last computed value in the window with the corresponding full-dataset value
        np.testing.assert_allclose(
            window_df.iloc[-1].values,
            full_df.iloc[i].values,
            rtol=rtol,
            atol=atol,
            err_msg=f"Lookahead bias detected at index {i} ({df_sorted.iloc[i][timestamp_col]})"
        )
