"""
Purged K-Fold with embargo (AFML ch. 7).

BUILD_PLAN_v2 A.3. Replaces `WalkForwardSplitter`'s SPLITTING logic while
keeping per-fold reporting detail.

=========================================================================
WHAT WAS ACTUALLY WRONG WITH THE OLD SPLITTER
=========================================================================
`WalkForwardSplitter` partitions on t0 -- the FEATURE timestamp -- and
nothing else. Its `_assert_strict_ordering` guard confirms train timestamps
precede val timestamps precede test timestamps. That is genuinely true, and
genuinely insufficient: a triple-barrier label is not resolved at t0, it is
resolved at t1.

A training sample at t0 = B0 - 1 with a 48-bar horizon has t1 = B0 + 47.
Its LABEL is computed from price action inside the held-out block. The old
splitter accepts it because its t0 sits on the correct side of the
boundary. Ordering of t0 was never the property that needed guaranteeing.

Measured at Stage A.1: ~8-9 BTC labels span every bar, so every boundary
carries a band of such samples.

=========================================================================
WHY K-FOLD AND NOT WALK-FORWARD
=========================================================================
A.3's embargo clause drops training samples with `t0 in [T1, T1 + h]` --
i.e. AFTER the held-out block. In a pure walk-forward layout no training
sample is ever after the test block, so that clause can never fire and the
embargo is dead code. Training data exists on BOTH sides of a held-out
block only in K-fold, which is what AFML ch. 7 specifies and what the
embargo is designed for.

Measured directly while building this (see scripts/leakage_test.py): under
the old walk-forward geometry with a 400-bar validation block and a 120-bar
horizon, train-to-test overlap is exactly ZERO -- the validation block acts
as an accidental buffer. The old splitter's real leak in that layout is
train-to-VALIDATION, which is not harmless: validation drives early
stopping and Optuna selection, so contaminating it corrupts the test
estimate too. Both leaks are closed here.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

T0_COL, T1_COL = "t0", "t1"


def require_label_metadata(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """t0/t1 as datetime64 arrays, or a loud failure.

    Refusing is deliberate. A splitter that silently skipped purging when
    metadata was absent would be indistinguishable from the leaking splitter
    it replaces -- which is precisely how the leak survived two phases.
    """
    missing = [c for c in (T0_COL, T1_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Purged splitting requires Stage A.1 label metadata {missing}. "
            f"Without t1 the purge predicate cannot be evaluated."
        )
    t0 = pd.to_datetime(df[T0_COL]).to_numpy(dtype="datetime64[ns]")
    t1 = pd.to_datetime(df[T1_COL]).to_numpy(dtype="datetime64[ns]")
    if np.any(t1 < t0):
        raise ValueError("Every label must satisfy t1 >= t0.")
    return t0, t1


def purge_and_embargo(
    candidate: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    blocks: Sequence[Tuple[np.datetime64, np.datetime64]],
    bars: np.ndarray,
    embargo_bars: int,
) -> Tuple[np.ndarray, int, int]:
    """Drop candidates that overlap, or sit in the embargo after, any block.

    PURGE    t0_i <= B1 AND t1_i >= B0   (label window intersects the block)
    EMBARGO  B1 < t0_i <= B1 + embargo_bars

    The embargo bound is resolved against the actual bar axis rather than by
    adding a wall-clock duration: exchange outages leave real gaps, so
    "B1 + h hours" can land in a hole and embargo the wrong span.

    Returns (kept_positions, n_purged, n_embargoed).
    """
    c_t0, c_t1 = t0[candidate], t1[candidate]
    purged = np.zeros(len(candidate), dtype=bool)
    embargoed = np.zeros(len(candidate), dtype=bool)

    for b0, b1 in blocks:
        purged |= (c_t0 <= b1) & (c_t1 >= b0)
        if embargo_bars > 0:
            pos = int(np.searchsorted(bars, b1, side="right")) - 1
            e_end = bars[min(pos + embargo_bars, len(bars) - 1)]
            embargoed |= (c_t0 > b1) & (c_t0 <= e_end)

    embargoed &= ~purged            # counted once, as purged
    return candidate[~(purged | embargoed)], int(purged.sum()), int(embargoed.sum())


@dataclass(frozen=True)
class PurgedFold:
    """One fold plus what purging and embargo actually removed.

    The counts are the per-fold reporting detail A.3 asks to keep. A fold
    that purges nothing is suspicious under overlapping labels; a fold that
    purges most of its training data means the horizon is large relative to
    the block. Both are visible here rather than inferred.
    """

    fold: int
    train: np.ndarray
    test: np.ndarray
    val: Optional[np.ndarray] = None
    n_train_before: int = 0
    n_purged: int = 0
    n_embargoed: int = 0

    @property
    def train_retention(self) -> float:
        return len(self.train) / self.n_train_before if self.n_train_before else float("nan")

    def summary(self) -> str:
        v = f"  val {len(self.val):,}" if self.val is not None else ""
        return (
            f"fold {self.fold}: train {self.n_train_before:,} -> {len(self.train):,} "
            f"(purged {self.n_purged:,}, embargoed {self.n_embargoed:,}, "
            f"retained {self.train_retention:.1%})  test {len(self.test):,}{v}"
        )


class PurgedKFold:
    """K-Fold over contiguous time blocks, with purging and embargo.

    Yields POSITIONAL index arrays (same contract as `WalkForwardSplitter`:
    `.loc` is unsafe on pooled frames whose index is non-unique).

    Training data comes from BOTH sides of the held-out block, which is what
    makes the embargo meaningful. Set `purge=False` only to demonstrate what
    the leak looks like -- it is not a supported operating mode.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge: bool = True,
        val_size: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if embargo_pct < 0:
            raise ValueError("embargo_pct must be non-negative.")
        if val_size < 0:
            raise ValueError("val_size must be non-negative.")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge = purge
        self.val_size = val_size

    def split_detailed(self, df: pd.DataFrame) -> Generator[PurgedFold, None, None]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index must be sorted ascending.")

        t0, t1 = require_label_metadata(df)
        n = len(df)
        bars = np.unique(np.concatenate([t0, t1]))
        embargo_bars = int(np.ceil(self.embargo_pct * n)) if self.purge else 0

        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        # Every row must end up in exactly one held-out block. Accumulated
        # here and asserted once the generator is exhausted -- see
        # `_assert_test_blocks_partition_every_row`.
        covered: List[np.ndarray] = []

        for k in range(self.n_splits):
            lo, hi = bounds[k], bounds[k + 1]
            test_pos = np.arange(lo, hi)
            if len(test_pos) == 0:
                continue

            rest = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
            if len(rest) == 0:
                continue

            blocks = [(t0[test_pos].min(), t1[test_pos].max())]

            val_pos = None
            if self.val_size > 0:
                # Validation carved from the training side and itself treated
                # as a held-out block that training is purged against.
                #
                # Preferred source is the rows immediately BEFORE the test
                # block (validation should not postdate the data it selects a
                # model for). But the FIRST fold's block starts at row 0, so
                # there is nothing before it -- carving from `before` alone
                # silently produced an empty validation set and made callers
                # skip fold 0 entirely, losing 1/K of the data from every
                # out-of-fold estimate. Found at Stage A.6, where scored rows
                # came to exactly 4/5 of nominal.
                #
                # Fall back to the rows immediately AFTER the block. In K-fold
                # (unlike walk-forward) training data legitimately exists on
                # both sides, and purge+embargo apply to it identically.
                before = rest[rest < lo]
                if len(before) >= self.val_size:
                    val_pos = before[-self.val_size:]
                else:
                    after = rest[rest > hi]
                    val_pos = after[:self.val_size] if len(after) >= self.val_size else before
                if val_pos is not None and len(val_pos):
                    blocks.append((t0[val_pos].min(), t1[val_pos].max()))
                    rest = np.setdiff1d(rest, val_pos, assume_unique=False)
                else:
                    val_pos = None

            if self.purge:
                train_pos, n_pur, n_emb = purge_and_embargo(
                    rest, t0, t1, blocks, bars, embargo_bars
                )
            else:
                train_pos, n_pur, n_emb = rest, 0, 0

            if len(train_pos) == 0:
                raise ValueError(
                    f"Fold {k}: purging emptied the training partition. The label "
                    f"horizon is too large relative to the block size."
                )

            if self.purge:
                self._assert_no_overlap(t0, t1, train_pos, test_pos, k)

            # `rest` is already the candidate set handed to purge_and_embargo,
            # so it IS the pre-purge count. Adding the purge counts back would
            # double-count them and understate retention.
            assert len(train_pos) == len(rest) - n_pur - n_emb, (
                "purge accounting does not balance: "
                f"{len(rest)} - {n_pur} - {n_emb} != {len(train_pos)}"
            )
            covered.append(test_pos)
            yield PurgedFold(
                fold=k, train=train_pos, test=test_pos, val=val_pos,
                n_train_before=len(rest),
                n_purged=n_pur, n_embargoed=n_emb,
            )

        # Reached only on full exhaustion. A caller that breaks out early
        # never runs this, which is correct -- partial iteration is a caller
        # choice, whereas a splitter that drops a block is a defect.
        self._assert_test_blocks_partition_every_row(covered, n)

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for f in self.split_detailed(df):
            yield (f.train, f.test)

    @staticmethod
    def _assert_no_overlap(t0, t1, train, test, fold: int) -> None:
        """The guard the old splitter should have had: checks LABEL WINDOWS,
        not t0 ordering. This is the property that actually prevents the
        leak, asserted directly rather than inferred from construction."""
        b0, b1 = t0[test].min(), t1[test].max()
        bad = (t0[train] <= b1) & (t1[train] >= b0)
        if bad.any():
            raise AssertionError(
                f"Fold {fold}: {int(bad.sum())} training samples still have label "
                f"windows overlapping the held-out block after purging."
            )

    @staticmethod
    def _assert_test_blocks_partition_every_row(
        covered: Sequence[np.ndarray], n: int
    ) -> None:
        """BUILD_PLAN_v2 amendment 2.1 -- the fold-coverage guard.

        The held-out blocks must PARTITION the data: every row in exactly one
        block, none twice. The original splitter carved validation from the
        rows before the test block, which is empty for fold 0, so callers
        skipped that fold and every out-of-fold estimate silently lost 1/K of
        the data -- the earliest 20% of history was never scored. It was
        caught at A.6 only by manually reconciling scored rows against
        nominal. That reconciliation is now a guard rather than a habit."""
        if not covered:
            raise AssertionError(
                "Fold coverage: no held-out blocks were produced at all."
            )
        stacked = np.concatenate(covered)
        uniq = np.unique(stacked)

        if len(stacked) != len(uniq):
            raise AssertionError(
                f"Fold coverage: held-out blocks overlap -- {len(stacked)} positions "
                f"across {len(covered)} folds but only {len(uniq)} distinct. A row "
                f"scored in two folds is scored against training data that contains it."
            )
        if len(uniq) != n or uniq[0] != 0 or uniq[-1] != n - 1:
            missing = np.setdiff1d(np.arange(n), uniq)
            raise AssertionError(
                f"Fold coverage: {len(uniq)} of {n} rows fall in a held-out block; "
                f"{len(missing)} never scored (first missing positions: "
                f"{missing[:10].tolist()}). Any metric computed here describes a "
                f"different sample than the one reported."
            )


def assert_scored_rows_match_nominal(
    n_scored: int, n_nominal: int, context: str = "out-of-fold scoring"
) -> None:
    """Hard-fail when a CV loop scores fewer rows than it was given.

    Companion to `PurgedKFold._assert_test_blocks_partition_every_row`, for
    the other end of the pipeline: the splitter can partition correctly and a
    caller can still drop a fold (single-class training data, an empty
    validation set, a model that failed to fit). Both produce an estimate over
    a sample the report does not describe, and A.6 showed the drop is
    invisible unless counted.

    A fold that genuinely cannot be trained is an operator-visible condition,
    not something to absorb silently -- which is why this raises rather than
    warns."""
    if n_scored != n_nominal:
        shortfall = n_nominal - n_scored
        raise AssertionError(
            f"{context}: scored {n_scored:,} of {n_nominal:,} rows "
            f"({shortfall:,} unscored, {shortfall / max(n_nominal, 1):.1%}). "
            f"Folds were dropped. Report the reduced sample deliberately or fix "
            f"the cause -- do not average over whatever survived."
        )


__all__ = [
    "PurgedKFold",
    "PurgedFold",
    "purge_and_embargo",
    "require_label_metadata",
    "assert_scored_rows_match_nominal",
]
