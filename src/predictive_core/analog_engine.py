"""
Component A -- Historical Analog Engine (k-NN).

Retrieves the k nearest historical instances to a query feature vector using
a weighted Euclidean distance metric, and reports the empirical forward-return
distribution over those neighbours as a probability estimate.

Two safeguards are load-bearing and are enforced centrally, in one place,
so there is exactly one code path that can produce a probability:

1. Minimum sample gate: fewer than `min_independent_instances` independent
   historical analogues within the distance threshold -> no forecast.
2. Independence enforcement: historical instances of the SAME ASSET that are
   temporally close (in real elapsed time, not row position -- see
   `_deduplicate`) to an already-accepted instance are treated as belonging
   to the same continuous directional move and are folded into it rather
   than counted as a second independent data point. Instances from different
   assets are never folded into each other, no matter how close in time --
   a BTC window and an ETH window are not "the same directional move."
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalogForecast:
    """
    Result of a single k-NN analog query. `is_valid` is the single source of
    truth for whether the minimum-sample gate passed. Probability fields are
    None whenever `is_valid` is False -- there is no way to read a
    probability out of a gated (invalid) forecast, by construction.
    """

    is_valid: bool
    n_independent: int
    n_raw_candidates: int
    distance_threshold: float
    probability_up: Optional[float]
    probability_down: Optional[float]
    probability_neutral: Optional[float]
    reason: str = ""

    def as_report(self) -> dict:
        """Every analog engine output must report N, the distance threshold,
        and the resulting probability (spec requirement) -- this is the
        single formatting path so that requirement can't be dropped by an
        individual caller forgetting to log one of the three fields."""
        return {
            "is_valid": self.is_valid,
            "n_independent": self.n_independent,
            "n_raw_candidates": self.n_raw_candidates,
            "distance_threshold": self.distance_threshold,
            "probability_up": self.probability_up,
            "probability_down": self.probability_down,
            "probability_neutral": self.probability_neutral,
            "reason": self.reason,
        }


class HistoricalAnalogEngine:
    """
    k-NN historical analog engine (Component A).

    Weights default to uniform (explicit placeholder per spec: "use uniform
    weights as a placeholder for now"). `set_weights` exists so Component B's
    trained feature-importance scores can later be plugged in without
    changing the query interface.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        direction_col: str = "direction",
        min_independent_instances: int = 50,
        min_temporal_separation: int = 48,
        bar_interval: pd.Timedelta = pd.Timedelta(hours=1),
        asset_col: str = "asset",
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        min_temporal_separation : int
            Minimum gap, in candles, required between two accepted analogues
            before they are treated as independent. Converted to a real
            elapsed-time threshold via `bar_interval` -- see class docstring
            for why this must be real time, not row position.
        bar_interval : pd.Timedelta
            Wall-clock duration of one candle in the reference set (e.g.
            1H or 4H). `min_temporal_separation * bar_interval` is the actual
            time gap enforced between accepted analogues of the same asset.
        asset_col : str
            Column name (if present in the frame passed to `fit`) used to
            identify which instrument each reference row belongs to.
            Analogues from different assets are never folded into each
            other, regardless of how close in time they are -- a BTC window
            and an ETH window are never "the same directional move."
        """
        if min_independent_instances <= 0:
            raise ValueError("min_independent_instances must be positive.")
        if min_temporal_separation < 0:
            raise ValueError("min_temporal_separation must be non-negative.")
        if bar_interval <= pd.Timedelta(0):
            raise ValueError("bar_interval must be positive.")

        self.feature_cols = list(feature_cols)
        self.direction_col = direction_col
        self.min_independent_instances = min_independent_instances
        self.min_temporal_separation = min_temporal_separation
        self.bar_interval = pd.Timedelta(bar_interval)
        self.asset_col = asset_col
        self._min_temporal_gap = self.bar_interval * min_temporal_separation
        self._weights = None if weights is None else np.asarray(weights, dtype=float)

        self._fitted = False
        self._ref_df: Optional[pd.DataFrame] = None
        self._ref_features: Optional[np.ndarray] = None
        self._ref_direction: Optional[np.ndarray] = None
        self._ref_positions: Optional[np.ndarray] = None
        self._ref_timestamps: Optional[np.ndarray] = None
        self._ref_asset: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, historical_df: pd.DataFrame) -> "HistoricalAnalogEngine":
        """
        Registers the historical reference set. This must only ever be
        called with the TRAINING partition of a walk-forward fold -- fitting
        on validation or test data would let the analog engine "recognize"
        the exact instances it will later be scored against.
        """
        required = set(self.feature_cols) | {self.direction_col}
        missing = required - set(historical_df.columns)
        if missing:
            raise ValueError(f"historical_df missing required columns: {sorted(missing)}")
        if not isinstance(historical_df.index, pd.DatetimeIndex):
            raise TypeError("historical_df index must be a DatetimeIndex.")
        if not historical_df.index.is_monotonic_increasing:
            raise ValueError("historical_df must be sorted ascending by time before fit().")

        self._ref_df = historical_df.reset_index(drop=False)
        self._ref_features = self._ref_df[self.feature_cols].to_numpy(dtype=float)
        self._ref_direction = self._ref_df[self.direction_col].to_numpy(dtype=float)
        # Row position is kept only for diagnostics/reporting. It is NOT used
        # for independence enforcement: in a pooled multi-asset frame (see
        # pool_assets), consecutive row positions interleave BTC/ETH/etc, so
        # "N rows apart" does not correspond to "N candles apart in the same
        # asset's series" -- it silently corresponds to N / (number of assets
        # present) real candles, and can straddle two different instruments
        # entirely. Independence is enforced in `_deduplicate` using actual
        # elapsed time (`_ref_timestamps`) plus asset identity
        # (`_ref_asset`), which is correct regardless of how many assets or
        # timeframes are pooled into `historical_df`.
        self._ref_positions = np.arange(len(self._ref_df))
        ts_col = self._ref_df.columns[0]
        self._ref_timestamps = self._ref_df[ts_col].to_numpy()
        self._ref_asset = (
            historical_df[self.asset_col].to_numpy()
            if self.asset_col in historical_df.columns
            else None
        )

        n_features = self._ref_features.shape[1]
        if self._weights is None:
            self._weights = np.ones(n_features, dtype=float)
        elif self._weights.shape[0] != n_features:
            raise ValueError(
                f"weights length ({self._weights.shape[0]}) does not match "
                f"number of feature columns ({n_features})."
            )
        if np.any(self._weights < 0):
            raise ValueError("weights must be non-negative.")

        self._fitted = True
        return self

    def set_weights(self, weights: Sequence[float]) -> None:
        """Swap in Component B's feature-importance-derived weights."""
        w = np.asarray(weights, dtype=float)
        if self._fitted and w.shape[0] != self._ref_features.shape[1]:
            raise ValueError("weights length does not match fitted feature dimensionality.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        self._weights = w

    def suggest_distance_threshold(self, percentile: float = 5.0, sample_size: int = 2000,
                                    random_state: int = 42) -> float:
        """
        Data-driven distance threshold, derived EXCLUSIVELY from the fitted
        (training) reference set -- never from validation or test data, since
        choosing a threshold based on how well it performs out-of-sample
        would itself be a leak. Returns the `percentile`-th percentile of
        pairwise weighted distances among a random sample of the training
        reference points, giving a threshold that admits roughly that
        fraction of the training set as "similar" to a typical point.
        """
        self._require_fitted()
        rng = np.random.default_rng(random_state)
        n = self._ref_features.shape[0]
        if n < 2:
            raise ValueError("Not enough reference points to suggest a threshold.")
        sample_size = min(sample_size, n)
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        sample = self._ref_features[sample_idx]

        dists = []
        # Chunked pairwise distance to avoid an O(sample_size^2) memory blowup
        # for large samples.
        chunk = 200
        for start in range(0, sample_size, chunk):
            block = sample[start:start + chunk]
            diff = block[:, None, :] - sample[None, :, :]
            sq = (diff ** 2) * self._weights[None, None, :]
            d = np.sqrt(sq.sum(axis=2))
            dists.append(d[np.triu_indices_from(d, k=1)] if block.shape[0] == sample.shape[0] else d.ravel())
        all_d = np.concatenate(dists)
        all_d = all_d[all_d > 0]
        return float(np.percentile(all_d, percentile))

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def predict(
        self,
        query_vector: Sequence[float],
        distance_threshold: float,
        query_timestamp: Optional[pd.Timestamp] = None,
    ) -> AnalogForecast:
        """
        The ONLY method in this class that produces a probability. Every
        other method either prepares state (fit, set_weights) or reports
        diagnostics (suggest_distance_threshold). This is deliberate: a
        single choke point means the minimum-sample gate cannot be
        bypassed by calling some other, lower-level method that skips it.
        """
        self._require_fitted()
        qv = np.asarray(query_vector, dtype=float)
        if qv.shape[0] != len(self.feature_cols):
            raise ValueError(
                f"query_vector has {qv.shape[0]} elements, expected {len(self.feature_cols)}."
            )
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive.")

        distances = self._weighted_distances(qv)

        candidate_mask = distances <= distance_threshold
        if query_timestamp is not None:
            ts_col = self._ref_df.columns[0]
            candidate_mask &= (self._ref_df[ts_col].to_numpy() < np.datetime64(query_timestamp))

        candidate_idx = np.where(candidate_mask)[0]
        n_raw = int(len(candidate_idx))

        if n_raw == 0:
            return AnalogForecast(
                is_valid=False,
                n_independent=0,
                n_raw_candidates=0,
                distance_threshold=distance_threshold,
                probability_up=None,
                probability_down=None,
                probability_neutral=None,
                reason="No candidates within distance threshold.",
            )

        order = candidate_idx[np.argsort(distances[candidate_idx])]
        independent_idx = self._deduplicate(order)
        n_independent = len(independent_idx)

        if n_independent < self.min_independent_instances:
            return AnalogForecast(
                is_valid=False,
                n_independent=n_independent,
                n_raw_candidates=n_raw,
                distance_threshold=distance_threshold,
                probability_up=None,
                probability_down=None,
                probability_neutral=None,
                reason=(
                    f"Only {n_independent} independent instances found "
                    f"(after collapsing overlapping same-move windows out of "
                    f"{n_raw} raw candidates); requires "
                    f"{self.min_independent_instances}."
                ),
            )

        dirs = self._ref_direction[independent_idx]
        n = float(len(dirs))
        p_up = float(np.sum(dirs == 1.0) / n)
        p_down = float(np.sum(dirs == -1.0) / n)
        p_neutral = float(np.sum(dirs == 0.0) / n)

        return AnalogForecast(
            is_valid=True,
            n_independent=n_independent,
            n_raw_candidates=n_raw,
            distance_threshold=distance_threshold,
            probability_up=p_up,
            probability_down=p_down,
            probability_neutral=p_neutral,
            reason="ok",
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("HistoricalAnalogEngine.fit() must be called before use.")

    def _weighted_distances(self, query_vector: np.ndarray) -> np.ndarray:
        diff = self._ref_features - query_vector[None, :]
        sq = (diff ** 2) * self._weights[None, :]
        return np.sqrt(sq.sum(axis=1))

    def _deduplicate(self, order: np.ndarray) -> List[int]:
        """
        Independence enforcement.

        `order` is the candidate pool sorted by ascending distance (closest
        analog first). We greedily accept candidates in that order, but only
        if the candidate is at least `min_temporal_separation * bar_interval`
        of REAL elapsed time away from every already-accepted candidate OF
        THE SAME ASSET. A rejected candidate is not discarded from the
        forecast -- it is evidence for the same continuous move as the
        accepted candidate it was too close to, so its distance rank is what
        earned that move a seat at the table, but it does not add a second,
        spuriously "independent" vote.

        Two conditions must both hold before a candidate is folded into an
        already-accepted one:
          1. Same asset (via `_ref_asset`, when available) -- a BTC window
             and an ETH window are two different instruments' moves, never
             "the same directional move", no matter how close in time.
          2. Real elapsed time between them is less than
             `min_temporal_separation * bar_interval`. Elapsed time, not row
             position, is what "temporally adjacent" has to mean: in a
             pooled multi-asset frame, row position and real time are not
             proportional (see fit()), so using row gaps here would both
             wrongly fold together different assets' windows and wrongly
             treat genuinely-overlapping same-asset windows as independent.
        """
        accepted: List[int] = []
        accepted_ts: List[np.datetime64] = []
        accepted_asset: List[object] = []
        gap = self._min_temporal_gap
        for idx in order:
            ts = self._ref_timestamps[idx]
            asset = self._ref_asset[idx] if self._ref_asset is not None else None
            collides = False
            for a_ts, a_asset in zip(accepted_ts, accepted_asset):
                if asset is not None and a_asset is not None and asset != a_asset:
                    continue
                if abs(ts - a_ts) < gap:
                    collides = True
                    break
            if not collides:
                accepted.append(int(idx))
                accepted_ts.append(ts)
                accepted_asset.append(asset)
        return accepted
