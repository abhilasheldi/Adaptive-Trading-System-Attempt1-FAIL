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

    Notes
    -----
    This checks *temporal* consistency only: it verifies that a feature computed with
    data truncated at row i never differs from the value it takes on the full dataset.
    It does NOT know whether your `timestamp_col` represents a bar's open or close time,
    so it will happily "pass" a feature that legitimately uses the current row's data
    (e.g. a 5-period SMA including today's close). If your features must exclude the
    current row entirely (e.g. because the timestamp marks the bar's open, and the
    close isn't known yet), that's a semantic choice you need to enforce in
    `feature_func` itself (e.g. by shifting inputs by 1) -- this helper can't infer it.
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

    if len(full_df) != len(df_sorted):
        raise AssertionError(
            f"feature_func returned {len(full_df)} rows for an input of {len(df_sorted)} "
            "rows. This helper assumes a 1:1, order-preserving mapping between input rows "
            "and output rows; row-dropping/reordering feature_funcs can't be safely checked "
            "by position and will invalidate the comparisons below."
        )

    for i in range(len(df_sorted)):
        # Slice dataset to include only data available up to and including the current row
        df_window = df_sorted.iloc[: i + 1].copy()
        window_result = feature_func(df_window)

        # Normalize window result
        if isinstance(window_result, pd.Series):
            window_df = window_result.to_frame()
        else:
            window_df = window_result.copy()

        full_row = full_df.iloc[i]
        window_row = window_df.iloc[-1]

        # Check column-by-column rather than skipping the whole row on any NaN.
        # A single slow-to-warm-up column (e.g. a 500-period SMA on a short frame)
        # must never suppress checks on the OTHER columns for this row.
        for col in full_df.columns:
            full_val = full_row[col]

            if pd.isna(full_val):
                # We have no known-good value to check against for this cell, so it's
                # unverifiable here -- but that's a per-cell judgment, not a per-row one.
                continue

            if col not in window_df.columns:
                raise AssertionError(
                    f"feature_func did not return column '{col}' when computed on the "
                    f"truncated window ending at index {i} "
                    f"({df_sorted.iloc[i][timestamp_col]}), but it did on the full dataset."
                )

            window_val = window_row[col]

            if pd.isna(window_val):
                # Full dataset has a real value here, but truncating to "data available
                # as of this row" produces NaN. That's the signature of a feature that
                # needs rows beyond i to resolve -- i.e. it's reaching into the future.
                raise AssertionError(
                    f"Lookahead bias detected in column '{col}' at index {i} "
                    f"({df_sorted.iloc[i][timestamp_col]}): value is NaN when computed "
                    f"using only data available up to this row, but is "
                    f"{full_val!r} when computed on the full dataset. This means the "
                    "feature depends on rows that don't exist yet at this point in time."
                )

            np.testing.assert_allclose(
                window_val,
                full_val,
                rtol=rtol,
                atol=atol,
                err_msg=(
                    f"Lookahead bias detected in column '{col}' at index {i} "
                    f"({df_sorted.iloc[i][timestamp_col]}): value computed from data "
                    "truncated at this row does not match the value computed on the "
                    "full dataset."
                ),
            )