"""
Invariant I10 -- acquisition is verified, not trusted.

BUILD_PLAN_v2 B.0.3. Every ingest asserts what came back against what was
asked for, and hard-fails on mismatch.

=========================================================================
THE BUG THIS EXISTS FOR
=========================================================================
S21: Binance treats `startTime=0` as UNSET. The first funding run asked for
full history and received 500 rows beginning 2026-03-01 -- five months of a
seven-year series -- with HTTP 200, no warning, and no exception. The data
looked clean. It parsed, it had the right columns, the timestamps were
monotonic, and every downstream check passed. It was found only because
someone counted the rows against expectation.

Pagination had a second version of the same failure: the loop stopped on
`len(batch) < limit`, but the endpoint returns short pages mid-history, so
it terminated years early and again returned a well-formed prefix.

The common shape is that a partial result is indistinguishable from a
complete one by inspection. The only defence is to state the expectation
BEFORE the call and check it after.

Assume every new API has its own version of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


class AcquisitionError(RuntimeError):
    """Raised when returned data does not match what was requested.

    Deliberately not a warning. A short series propagates into every
    measurement downstream, and by the time it surfaces as an odd number it
    has been averaged into results that are already written down.
    """


@dataclass(frozen=True)
class AcquisitionRequest:
    """What was asked for, recorded before the call so it can be checked after.

    `expected_min_rows` and the date bounds are deliberately separate: a
    request can return the right span with holes in it, or the right count
    over the wrong window, and those are different defects.
    """

    source: str
    requested_start: Optional[pd.Timestamp] = None
    requested_end: Optional[pd.Timestamp] = None
    expected_min_rows: Optional[int] = None
    # Tolerance for the observed start postdating the request. Legitimate
    # when an instrument simply did not exist yet -- BTC perps begin
    # 2019-09 regardless of what you ask for -- so it must be stated per
    # call rather than assumed globally.
    allowed_start_slack: pd.Timedelta = pd.Timedelta(days=0)
    allowed_end_slack: pd.Timedelta = pd.Timedelta(days=0)
    note: str = ""


def assert_acquisition(df: pd.DataFrame, request: AcquisitionRequest) -> pd.DataFrame:
    """Verify a fetch against its request. Returns `df` unchanged on success.

    Returning the frame lets this wrap a call site directly:
        df = assert_acquisition(fetch_x(...), AcquisitionRequest(...))
    so that adding the check does not require restructuring the caller --
    a check that is awkward to add is a check that does not get added.
    """
    src = request.source

    if df is None or len(df) == 0:
        raise AcquisitionError(
            f"[{src}] returned no rows. Requested "
            f"{_fmt(request.requested_start)} -> {_fmt(request.requested_end)}. "
            f"An empty result is a failure, not an empty dataset."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise AcquisitionError(f"[{src}] index is {type(df.index).__name__}, not DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise AcquisitionError(f"[{src}] index is not sorted ascending.")

    n_dupes = int(df.index.duplicated().sum())
    if n_dupes:
        raise AcquisitionError(f"[{src}] {n_dupes:,} duplicate timestamps in the returned index.")

    observed_start, observed_end = df.index.min(), df.index.max()

    if request.expected_min_rows is not None and len(df) < request.expected_min_rows:
        raise AcquisitionError(
            f"[{src}] returned {len(df):,} rows, expected at least "
            f"{request.expected_min_rows:,}. This is the S21 signature -- a "
            f"well-formed prefix of the real series. Window returned: "
            f"{_fmt(observed_start)} -> {_fmt(observed_end)}."
        )

    if request.requested_start is not None:
        overshoot = observed_start - request.requested_start
        if overshoot > request.allowed_start_slack:
            raise AcquisitionError(
                f"[{src}] history starts {_fmt(observed_start)}, "
                f"{overshoot} after the requested {_fmt(request.requested_start)} "
                f"(slack {request.allowed_start_slack}). Either the endpoint silently "
                f"ignored the start bound (S21) or the instrument's history is "
                f"genuinely shorter -- state which, do not widen the slack to "
                f"make this pass."
            )

    if request.requested_end is not None:
        shortfall = request.requested_end - observed_end
        if shortfall > request.allowed_end_slack:
            raise AcquisitionError(
                f"[{src}] history ends {_fmt(observed_end)}, {shortfall} short of the "
                f"requested {_fmt(request.requested_end)} (slack "
                f"{request.allowed_end_slack}). Pagination probably terminated early -- "
                f"the S21 short-page defect."
            )

    return df


def acquisition_summary(df: pd.DataFrame, request: AcquisitionRequest) -> dict:
    """Machine-readable record of a verified fetch, for the ingest report."""
    return {
        "source": request.source,
        "rows": len(df),
        "start": df.index.min() if len(df) else pd.NaT,
        "end": df.index.max() if len(df) else pd.NaT,
        "requested_start": request.requested_start,
        "requested_end": request.requested_end,
        "note": request.note,
    }


def _fmt(ts: Optional[pd.Timestamp]) -> str:
    return "unbounded" if ts is None or pd.isna(ts) else str(ts)


__all__ = [
    "AcquisitionError",
    "AcquisitionRequest",
    "assert_acquisition",
    "acquisition_summary",
]
