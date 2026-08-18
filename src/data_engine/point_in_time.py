"""
Invariant I9 -- point-in-time discipline, enforced structurally.

BUILD_PLAN_v2 B.0.2. Every series declares its OBSERVATION timestamp
separately from its VALUE timestamp, and alignment onto a bar index is only
reachable through that declaration.

=========================================================================
WHY THIS IS A MODULE AND NOT A CONVENTION
=========================================================================
Back-filling a settled value into the interval it settles over is lookahead
that NO SPLITTER CAN CATCH, because it is baked into the feature rather
than the split. Purged K-Fold compares label windows against held-out
blocks; it has no way to know that a funding rate stamped 00:00 was not
knowable until 08:00. The corruption is inside the feature column, and it
survives every partition you can draw over it.

B.1 got this right by hand for funding and macro. Doing it by hand is the
failure mode -- the next series is aligned by whoever adds it, under
deadline, and the mistake is silent and permanent. So the aligning function
REQUIRES a policy argument. There is no default, and no way to call it
without stating when the data became knowable.

=========================================================================
THE TWO SHAPES THAT COVER EVERY SERIES IN THIS PROJECT
=========================================================================
`SETTLES_AT_PERIOD_END`
    The value describes an interval and is only knowable once the interval
    closes. Funding settles 08:00 for the window 00:00-08:00: its VALUE
    timestamp is the period, its OBSERVATION timestamp is 08:00.

`PUBLISHED_AFTER_VALUE_DATE`
    The value describes a date and is released some time later. FRED
    publishes an observation for date D on D+1 business day or later. The
    value timestamp is D; the observation timestamp is the release.

Both reduce to the same operation -- shift the index forward to the moment
of knowability, then forward-fill -- which is why they share one code path.
The distinction is kept because the two carry different failure modes and a
reader needs to see which one a series was declared under.

=========================================================================
WHAT "NaN, NEVER 0" BUYS
=========================================================================
Bars before a series' first observation are NaN. A zero funding rate is a
real market state; a zero VIX is not, but a model cannot tell the
difference between "absent" and "zero" once you have written the zero. NaN
propagates into any feature derived from it and shows up as a coverage
number rather than as a quietly wrong regime label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

Frame = Union[pd.Series, pd.DataFrame]

ObservationKind = Literal["settles_at_period_end", "published_after_value_date"]


@dataclass(frozen=True)
class ObservationPolicy:
    """When a value first became knowable, relative to what it describes.

    Attributes
    ----------
    name : str
        Series name, used in error messages and coverage reports.
    kind : ObservationKind
        Which of the two shapes above. Recorded for the reader; both are
        applied through the same shift.
    calendar_lag : pd.Timedelta, optional
        Fixed wall-clock delay from value timestamp to observability. Used
        by `settles_at_period_end` (e.g. 8h for funding).
    business_day_lag : int, optional
        Delay in BUSINESS days. Used by `published_after_value_date`, where
        a calendar shift would place a Friday observation on a Saturday --
        a day the publishing calendar does not have.
    note : str
        Why this lag, in one line. Appears in the policy repr so a wrong
        lag is arguable rather than mysterious.
    """

    name: str
    kind: ObservationKind
    calendar_lag: Optional[pd.Timedelta] = None
    business_day_lag: Optional[int] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.calendar_lag is None and self.business_day_lag is None:
            raise ValueError(
                f"[{self.name}] a policy must state a lag. Zero lag is a legitimate "
                f"choice (pass calendar_lag=pd.Timedelta(0)) but it must be stated, "
                f"not defaulted -- an unstated zero is exactly the assumption I9 exists "
                f"to prevent."
            )
        if self.calendar_lag is not None and self.business_day_lag is not None:
            raise ValueError(
                f"[{self.name}] state one lag, not both -- otherwise the effective "
                f"delay depends on which one the reader happens to look at."
            )
        if self.calendar_lag is not None and self.calendar_lag < pd.Timedelta(0):
            raise ValueError(f"[{self.name}] negative lag is lookahead by construction.")
        if self.business_day_lag is not None and self.business_day_lag < 0:
            raise ValueError(f"[{self.name}] negative lag is lookahead by construction.")

    def observation_timestamps(self, value_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Map value timestamps to the moment each became knowable."""
        if not isinstance(value_index, pd.DatetimeIndex):
            raise TypeError(f"[{self.name}] value index must be a DatetimeIndex.")
        if self.calendar_lag is not None:
            return value_index + self.calendar_lag
        # BusinessDay respects the Mon-Fri publishing calendar. A calendar
        # shift would put Friday's observation on Saturday, which no
        # statistical agency publishes on, understating the real delay.
        return value_index + pd.tseries.offsets.BusinessDay(self.business_day_lag)


# ---------------------------------------------------------------------- #
# Policies for the series actually in this project
# ---------------------------------------------------------------------- #

FUNDING_8H = ObservationPolicy(
    name="funding_8h",
    kind="settles_at_period_end",
    calendar_lag=pd.Timedelta(0),
    note=(
        "Binance stamps each funding row with its SETTLEMENT time, so the stored "
        "index is already the observation timestamp -- the value for 00:00-08:00 "
        "carries index 08:00. Zero additional lag, stated explicitly."
    ),
)

# FRED's real release delay varies by series and is not a constant. One
# business day is the MINIMUM defensible shift and is what B.0.4 specifies
# absent ALFRED vintages; it is deliberately conservative in the safe
# direction (too much lag costs signal, too little manufactures it).
FRED_DAILY = ObservationPolicy(
    name="fred_daily",
    kind="published_after_value_date",
    business_day_lag=1,
    note=(
        "FRED publishes an observation for date D no earlier than D+1 business day. "
        "The true delay varies by series and vintage; 1 business day is the minimum "
        "defensible shift per B.0.4. Replace with ALFRED vintages when an API key "
        "is available -- see fred_fetcher.fetch_alfred_vintage."
    ),
)

# B.5 -- an exchange-traded daily close, not a published statistic. The
# close on date D is knowable at D's market close, but this project runs on
# UTC 1h bars and QQQ closes at 20:00/21:00 UTC depending on DST. Rather
# than model a shifting intraday boundary, the whole bar is treated as
# observable one BUSINESS day later, matching DEXUSUK. Conservative in the
# safe direction: too much lag costs coverage, too little manufactures
# signal.
MARKET_DAILY_CLOSE = ObservationPolicy(
    name="market_daily_close",
    kind="published_after_value_date",
    business_day_lag=1,
    note=(
        "Exchange daily close (QQQ). Same one-business-day treatment as DEXUSUK, "
        "chosen over an intraday UTC boundary that would shift with US DST."
    ),
)

# B.5 condition 4 -- a 1d/1w bar is not knowable until it CLOSES. Using a
# daily bar on an intraday bar inside its own session is the funding defect
# in another shape, and no splitter can see it.
BAR_CLOSE_1D = ObservationPolicy(
    name="bar_close_1d",
    kind="settles_at_period_end",
    calendar_lag=pd.Timedelta(days=1),
    note="A 1d bar indexed at date D describes D 00:00-24:00; observable at D+1 00:00.",
)

BAR_CLOSE_1W = ObservationPolicy(
    name="bar_close_1w",
    kind="settles_at_period_end",
    calendar_lag=pd.Timedelta(days=7),
    note="A 1w bar indexed at week W describes W; observable at W+7d.",
)

BAR_CLOSE_1H = ObservationPolicy(
    name="bar_close_1h",
    kind="settles_at_period_end",
    calendar_lag=pd.Timedelta(hours=1),
    note=(
        "An OHLCV bar indexed at its OPEN time is not knowable until it closes. "
        "Declared for completeness so that any series joined against bar opens "
        "goes through the same check."
    ),
)


# ---------------------------------------------------------------------- #
# The aligning operation
# ---------------------------------------------------------------------- #


def align_point_in_time(
    values: Frame,
    bar_index: pd.DatetimeIndex,
    policy: ObservationPolicy,
) -> Frame:
    """Forward-fill `values` onto `bar_index`, keyed on OBSERVATION time.

    `policy` is required and has no default. That is the whole point of the
    module: you cannot align a series onto bars without saying when it
    became knowable.

    A bar at time `t` sees a value if and only if that value's observation
    timestamp is `<= t`. Bars before the first observation are NaN, never 0.

    Returns the same container type it was given (Series in, Series out).
    """
    if not isinstance(bar_index, pd.DatetimeIndex):
        raise TypeError("bar_index must be a DatetimeIndex.")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(f"[{policy.name}] values must be indexed by value timestamp.")
    if not values.index.is_monotonic_increasing:
        raise ValueError(f"[{policy.name}] value index must be sorted ascending.")

    was_series = isinstance(values, pd.Series)
    frame = values.to_frame() if was_series else values

    observed = policy.observation_timestamps(frame.index)
    shifted = frame.copy()
    shifted.index = observed
    # Two value dates can map to one observation instant (a Monday release
    # covering a holiday weekend). Keep the LAST -- it is the most recent
    # value knowable at that moment.
    shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()

    union = shifted.index.union(bar_index)
    aligned = shifted.reindex(union).ffill().reindex(bar_index)

    # Explicit rather than relying on ffill: everything strictly before the
    # first observation is unknowable, and must read as absent.
    if len(shifted):
        aligned.loc[aligned.index < shifted.index.min()] = np.nan

    return aligned.iloc[:, 0].rename(values.name) if was_series else aligned


def observation_coverage(
    values: Frame, bar_index: pd.DatetimeIndex, policy: ObservationPolicy
) -> dict:
    """Coverage of an aligned series, for the record rather than for a gate.

    Reports how many bars are NaN because the series had not started yet.
    That number is a fact about the flow window (S22) and belongs in every
    report that uses the series, per I5.
    """
    aligned_ = align_point_in_time(values, bar_index, policy)
    frame = aligned_.to_frame() if isinstance(aligned_, pd.Series) else aligned_
    observed = policy.observation_timestamps(values.index)
    first = observed.min() if len(observed) else pd.NaT

    n_bars = len(bar_index)
    pre = int((bar_index < first).sum()) if pd.notna(first) else n_bars
    return {
        "series": policy.name,
        "first_observable": first,
        "n_bars": n_bars,
        "n_pre_observation_bars": pre,
        "pre_observation_fraction": (pre / n_bars) if n_bars else float("nan"),
        "n_nan_after_start": int(frame.loc[frame.index >= first].isna().any(axis=1).sum())
        if pd.notna(first)
        else 0,
    }


def assert_no_lookahead(
    values: Frame,
    aligned: Frame,
    policy: ObservationPolicy,
) -> None:
    """Verify that an ALREADY-ALIGNED frame contains no lookahead.

    Takes the aligned frame as an argument rather than producing it. An
    earlier version aligned and then checked with the same policy, which is
    self-consistent by construction and therefore cannot fail -- it would
    have passed a back-filled series just as happily. Checking a function
    with itself proves nothing.

    So this verifies someone ELSE's alignment: hand it the source values,
    whatever frame was actually built from them, and the policy that frame
    is claimed to satisfy. Every non-NaN cell must trace to a value whose
    observation timestamp is at or before its bar.

    Use `align_point_in_time(...)` to produce a compliant frame; use this to
    audit one you did not produce.
    """
    frame = aligned.to_frame() if isinstance(aligned, pd.Series) else aligned
    src = values.to_frame() if isinstance(values, pd.Series) else values

    observed = pd.DatetimeIndex(policy.observation_timestamps(src.index))
    for col in frame.columns:
        filled = frame[col].notna().to_numpy()
        if not filled.any():
            continue
        bars = pd.DatetimeIndex(frame.index[filled])
        # Position of the most recent observation at or before each bar.
        pos = observed.searchsorted(bars, side="right") - 1
        if (pos < 0).any():
            bad = bars[pos < 0][0]
            raise AssertionError(
                f"[{policy.name}:{col}] bar {bad} carries a value with no observation "
                f"at or before it -- the series was back-filled."
            )
        if (observed[pos] > bars).any():
            i = int(np.argmax(observed[pos] > bars))
            raise AssertionError(
                f"[{policy.name}:{col}] bar {bars[i]} carries a value observed at "
                f"{observed[pos][i]} -- {observed[pos][i] - bars[i]} of lookahead."
            )


__all__ = [
    "ObservationPolicy",
    "FUNDING_8H",
    "FRED_DAILY",
    "MARKET_DAILY_CLOSE",
    "BAR_CLOSE_1D",
    "BAR_CLOSE_1W",
    "BAR_CLOSE_1H",
    "align_point_in_time",
    "observation_coverage",
    "assert_no_lookahead",
]
