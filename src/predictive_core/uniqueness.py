"""
Sample uniqueness and effective sample size (AFML ch. 4).

BUILD_PLAN_v2 A.2. Implements, exactly:

    indicator    1[t,i] = 1 if label i's window [t0_i, t1_i] spans bar t
    concurrency  c_t    = sum_i 1[t,i]
    uniqueness   u[t,i] = 1[t,i] / c_t
    avg uniq     u_bar_i = sum_t u[t,i] / sum_t 1[t,i]
    effective n  n_eff  = sum_i u_bar_i

WHY THIS EXISTS (Invariant I6): overlapping triple-barrier labels are not
independent observations. Measured at Stage A.1, roughly 8-9 BTC labels span
every bar, so 78,378 nominal samples carry far less than 78,378 samples'
worth of information. Every confidence interval computed on nominal n is
too narrow, and every result reporting nominal n alone is overstating its
own precision.

=========================================================================
TWO CORRECTNESS PROPERTIES THAT ARE EASY TO GET WRONG
=========================================================================

1. UNIQUENESS IS PER-ASSET. A BTC label and an ETH label spanning the same
   hour do NOT read the same information, so they must not dilute each
   other's uniqueness. `pool_assets` produces exactly such a frame, with a
   non-unique index. Pass `groups` (the asset column) and concurrency is
   computed independently within each asset. Omitting it on a pooled frame
   silently halves every uniqueness value.

   This is the same failure mode as the cross-asset bug found in the
   since-deleted k-NN engine's `_deduplicate` (project_state.md,
   2026-08-12), where row adjacency in a pooled frame was mistaken for
   temporal adjacency within one instrument.

2. UNIQUENESS IS A PROPERTY OF THE SAMPLE SET, NOT OF THE MARKET. u_bar_i
   answers "how much of label i's information is shared with the OTHER
   LABELS PRESENT". Subset the labels and the survivors become more unique,
   because the labels they overlapped with are gone. So `average_uniqueness`
   computed on a filtered event set is NOT the same number as the
   full-dataset value restricted to those rows. Neither is wrong; they
   answer different questions, and which one is correct depends on which
   set the estimator actually trains on. Callers must choose deliberately --
   `restrict_to` exists so the choice is explicit rather than implied by
   whichever frame happened to be passed.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniquenessResult:
    """Per-label average uniqueness plus the aggregates I6 requires."""

    average_uniqueness: np.ndarray   # u_bar_i, one per label, in input order
    nominal_n: int                   # count of labels
    effective_n: float               # n_eff = sum of u_bar_i
    max_concurrency: int             # max_t c_t, across all groups
    mean_concurrency: float          # mean of c_t over spanned bars

    @property
    def ratio(self) -> float:
        """n_eff / nominal n. Reported alongside both, never instead of."""
        return self.effective_n / self.nominal_n if self.nominal_n else float("nan")

    def summary(self) -> str:
        return (
            f"nominal n={self.nominal_n:,}  n_eff={self.effective_n:,.1f}  "
            f"ratio={self.ratio:.4f}  mean_concurrency={self.mean_concurrency:.2f}"
        )


def _one_group_uniqueness(
    t0: np.ndarray, t1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """u_bar for one group, plus that group's concurrency series.

    Uses a difference array rather than materialising the indicator matrix.
    The literal 1[t,i] matrix would be n_labels x n_bars -- at 78,378 labels
    it is ~6.1e9 cells, which is why AFML's pandas reference implementation
    is unusable at this scale. The difference-array form is exactly
    equivalent and O(n log n), dominated by the sort in `np.unique`.
    """
    if len(t0) == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=np.int64)

    # A bar axis spanning every timestamp any label touches. Built from the
    # union of endpoints: a label's window is contiguous in BAR SPACE, so
    # bars strictly inside a window but appearing as no label's endpoint
    # would still be spanned. Including all t0s covers them, because a label
    # begins on essentially every bar; where it does not, the axis simply
    # has no atom there and the ratio is unaffected (that bar contributes to
    # no label's numerator or denominator).
    bars = np.unique(np.concatenate([t0, t1]))
    s = np.searchsorted(bars, t0)
    e = np.searchsorted(bars, t1)

    m = len(bars)
    diff = np.zeros(m + 1, dtype=np.int64)
    np.add.at(diff, s, 1)
    np.add.at(diff, e + 1, -1)
    c = np.cumsum(diff)[:m]                      # c_t

    spanned = c > 0
    inv_c = np.zeros(m, dtype=float)
    inv_c[spanned] = 1.0 / c[spanned]            # u[t,i] numerator per bar

    # sum_t u[t,i] over [s_i, e_i], via prefix sums.
    prefix = np.concatenate([[0.0], np.cumsum(inv_c)])
    span_bars = (e - s + 1).astype(float)        # sum_t 1[t,i]
    u_bar = (prefix[e + 1] - prefix[s]) / span_bars
    return u_bar, c


def average_uniqueness(
    t0: Sequence,
    t1: Sequence,
    groups: Optional[Sequence] = None,
    restrict_to: Optional[np.ndarray] = None,
) -> UniquenessResult:
    """Average uniqueness u_bar_i per label, and n_eff.

    Parameters
    ----------
    t0, t1 : sequences of datetime-like, equal length
        Label event time and resolution time. `apply_triple_barrier` emits
        both as columns (Stage A.1).
    groups : sequence, optional
        Per-label group identity, normally the `asset` column. Concurrency is
        computed INDEPENDENTLY within each group. **Required for any pooled
        multi-asset frame** -- see this module's docstring, property 1.
    restrict_to : boolean mask, optional
        Compute uniqueness among ONLY the masked labels, then return u_bar
        for those labels. This is different from computing on everything and
        then masking the result -- see property 2. Returned arrays are
        length `restrict_to.sum()`, aligned to the masked labels in order.

    Returns
    -------
    UniquenessResult
    """
    t0 = pd.to_datetime(pd.Series(list(t0))).to_numpy(dtype="datetime64[ns]")
    t1 = pd.to_datetime(pd.Series(list(t1))).to_numpy(dtype="datetime64[ns]")
    if len(t0) != len(t1):
        raise ValueError(f"t0 and t1 must be the same length; got {len(t0)} and {len(t1)}.")
    if np.any(t1 < t0):
        raise ValueError(
            "Every label must satisfy t1 >= t0. A label resolving before its own "
            "event time would make the concurrency window empty or negative."
        )

    g = None if groups is None else np.asarray(list(groups), dtype=object)
    if g is not None and len(g) != len(t0):
        raise ValueError(f"groups must match t0/t1 length; got {len(g)} vs {len(t0)}.")

    if restrict_to is not None:
        mask = np.asarray(restrict_to, dtype=bool)
        if len(mask) != len(t0):
            raise ValueError(f"restrict_to must match t0 length; got {len(mask)} vs {len(t0)}.")
        t0, t1 = t0[mask], t1[mask]
        g = None if g is None else g[mask]

    n = len(t0)
    if n == 0:
        return UniquenessResult(np.zeros(0), 0, 0.0, 0, float("nan"))

    u_bar = np.empty(n, dtype=float)
    max_conc = 0
    conc_parts = []

    if g is None:
        u_bar, c = _one_group_uniqueness(t0, t1)
        max_conc = int(c.max()) if len(c) else 0
        conc_parts.append(c[c > 0])
    else:
        for key in pd.unique(g):
            idx = np.flatnonzero(g == key)
            u_part, c = _one_group_uniqueness(t0[idx], t1[idx])
            u_bar[idx] = u_part
            if len(c):
                max_conc = max(max_conc, int(c.max()))
                conc_parts.append(c[c > 0])

    all_conc = np.concatenate(conc_parts) if conc_parts else np.zeros(0)
    return UniquenessResult(
        average_uniqueness=u_bar,
        nominal_n=n,
        effective_n=float(u_bar.sum()),
        max_concurrency=max_conc,
        mean_concurrency=float(all_conc.mean()) if len(all_conc) else float("nan"),
    )


def uniqueness_sample_weights(
    t0: Sequence,
    t1: Sequence,
    groups: Optional[Sequence] = None,
    normalise: bool = True,
) -> tuple[np.ndarray, UniquenessResult]:
    """Training sample weights from average uniqueness.

    `normalise=True` rescales weights to mean 1.0. This does NOT change the
    relative weighting -- only the absolute scale -- but it keeps
    sample-count-sensitive hyperparameters (`min_child_samples`,
    `min_child_weight`) on the same footing as an unweighted run, so a
    weighted fit stays comparable to the baseline it replaces. n_eff is
    computed from the UNNORMALISED u_bar and is unaffected.
    """
    result = average_uniqueness(t0, t1, groups=groups)
    w = result.average_uniqueness.copy()
    if normalise and len(w) and w.mean() > 0:
        w = w / w.mean()
    return w, result


__all__ = [
    "UniquenessResult",
    "average_uniqueness",
    "uniqueness_sample_weights",
]
