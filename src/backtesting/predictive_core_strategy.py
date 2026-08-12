"""
PredictiveCoreStrategy -- adapter wiring the real Component A (k-NN analog
engine, `predictive_core.analog_engine`) and Component B (LightGBM
direction/magnitude/duration models, `predictive_core.boosted_core`) into
`WalkForwardBacktester`'s `SignalStrategy` protocol.

Three interface facts this adapter has to work around, none of them
guessable from the spec alone:

1. `train_direction_classifier` / `train_quantile_regressor` require a
   `test_df` argument and evaluate test-set metrics as a side effect of
   training -- there's no "just train" mode. Since `fit()` must never see
   this fold's real test partition, `val_df` is passed in that slot as a
   placeholder, and every test-specific field on the returned result
   (test_logloss, calibration_reports, test_pinball_loss,
   quantile_crossing_rate) is discarded -- it was computed against the
   wrong (placeholder) data and would be actively misleading to keep.
2. The direction model's raw `.predict()` output is UNCALIBRATED softmax.
   Calibrated probabilities require reapplying the per-class isotonic
   calibrators attached to the model as `.calibrators_` and renormalizing
   -- exactly mirroring what `train_direction_classifier` does internally
   when it evaluates its own (discarded, per point 1) test predictions.
   Skipping this and using `.predict()` directly would silently throw away
   the entire point of Component B's calibration step.
3. `WalkForwardTrainer` (boosted_core's own orchestrator) is NOT used here
   -- it does its own row-count-based fold splitting via
   `predictive_core.splitter.WalkForwardSplitter`, which is a different,
   incompatible splitter from this harness's calendar-based
   `backtesting.windows`. This adapter calls `train_direction_classifier` /
   `train_quantile_regressor` directly on the train/val partitions
   `WalkForwardBacktester` already handed it.

A fourth issue is not an interface quirk but a real correctness bug this
adapter has to guard against: `apply_triple_barrier` labels are
prospective (a row's direction/magnitude/duration depends on up to
`label_horizon` bars FORWARD). A row near the end of `train_df` can have a
label computed from price action that actually falls inside `val_df` --
training on it leaks val-partition information through the label. Same for
`val_df`'s tail relative to whatever follows. `fit()` purges the trailing
`label_horizon` rows of both train_df and val_df before use. `label_horizon`
must match whatever `max_candles` value was used when `apply_triple_barrier`
was originally run upstream -- this adapter has no way to verify that from
inside `fit()`; get it wrong and the purge silently protects against the
wrong window size.

Gating: MASTER_CONTEXT's six trade-entry gates (constraint 8) are EV lower
CI bound, MTF alignment, macro regime, volatility regime, analog sample
size >= 50, and risk score. Only the analog sample-size gate is enforced
here (it comes for free from `AnalogForecast.is_valid`) -- MTF, macro
regime, volatility regime, and the EV/risk-score gates all require modules
that don't exist yet (Phase 2+). This strategy is a Phase 1 research probe
for the Layer 14 walk-forward check, not final entry-gate logic.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.predictive_core.analog_engine import HistoricalAnalogEngine
from src.predictive_core.boosted_core import train_direction_classifier, train_quantile_regressor


class PredictiveCoreStrategy:
    def __init__(
        self,
        feature_cols: Sequence[str],
        label_horizon: int = 48,
        distance_threshold_percentile: float = 10.0,
        min_direction_confidence: float = 0.55,
        magnitude_position_scale: float = 5.0,
        max_position: float = 1.0,
        n_optuna_trials: int = 15,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        feature_cols : Sequence[str]
            Columns fed to BOTH the analog engine's distance metric and
            Component B's models -- must already be point-in-time-safe,
            normalized feature columns (see features/feature_calculator.py).
        label_horizon : int
            Must match the `max_candles` used when `apply_triple_barrier`
            was run on this data upstream. Used for two things: purging
            label-horizon overlap at fold boundaries (see module docstring),
            and as the analog engine's `min_temporal_separation` (two
            analogs closer together than one label horizon are treated as
            the same continuous move, matching how the labels themselves
            were constructed).
        min_direction_confidence : float
            Component B's own calibrated probability for whichever
            direction it calls must clear this before a position is taken.
            Not a MASTER_CONTEXT spec number -- an adapter-level modeling
            choice, tune freely.
        magnitude_position_scale, max_position : float
            Position size = min(predicted median magnitude as a fraction of
            price * magnitude_position_scale, max_position). A placeholder
            sizing rule, NOT quarter-Kelly sizing -- that's Layer 10
            (risk_sizing_selection), not built yet. This exists only so the
            Layer 14 backtest has *some* signed, bounded position series to
            evaluate.
        """
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty.")
        if label_horizon <= 0:
            raise ValueError("label_horizon must be positive.")

        self.feature_cols = list(feature_cols)
        self.label_horizon = label_horizon
        self.distance_threshold_percentile = distance_threshold_percentile
        self.min_direction_confidence = min_direction_confidence
        self.magnitude_position_scale = magnitude_position_scale
        self.max_position = max_position
        self.n_optuna_trials = n_optuna_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self._analog_engine: Optional[HistoricalAnalogEngine] = None
        self._distance_threshold: Optional[float] = None
        self._direction_model = None
        self._magnitude_models = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        required = set(self.feature_cols) | {"direction", "magnitude", "duration", "close"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing required columns: {sorted(missing)}")
        missing = required - set(val_df.columns)
        if missing:
            raise ValueError(f"val_df missing required columns: {sorted(missing)}")

        # Purge trailing label-horizon overlap at both internal boundaries
        # (train/val and val/[whatever follows]) -- see module docstring.
        train_purged = train_df.iloc[: -self.label_horizon] if len(train_df) > self.label_horizon else train_df.iloc[:0]
        val_purged = val_df.iloc[: -self.label_horizon] if len(val_df) > self.label_horizon else val_df.iloc[:0]
        if train_purged.empty or val_purged.empty:
            raise ValueError(
                f"train_df/val_df too short relative to label_horizon={self.label_horizon} "
                f"after boundary purging (train_purged={len(train_purged)}, val_purged={len(val_purged)})."
            )

        # --- Component A: analog reference set built from train+val purged data. ---
        self._analog_engine = HistoricalAnalogEngine(
            feature_cols=self.feature_cols,
            min_independent_instances=50,  # MASTER_CONTEXT constraint 9 -- exact spec number, not configurable
            min_temporal_separation=self.label_horizon,
        )
        self._analog_engine.fit(pd.concat([train_purged, val_purged]))
        self._distance_threshold = self._analog_engine.suggest_distance_threshold(
            percentile=self.distance_threshold_percentile
        )

        # --- Component B: direct calls, NOT WalkForwardTrainer (see module docstring point 3). ---
        # val_df/val_purged is passed as the required `test_df` positional
        # arg below -- never this fold's real test data. Every test-specific
        # field on the returned *FoldResult objects is discarded afterward.
        model, direction_result = train_direction_classifier(
            train_purged, val_purged, val_purged,
            feature_cols=self.feature_cols, categorical_cols=[],  # no 'asset' col: harness enforces single-instrument df
            fold=0, n_optuna_trials=self.n_optuna_trials,
            early_stopping_rounds=self.early_stopping_rounds, random_state=self.random_state,
        )
        self._direction_model = model  # carries .calibrators_, fit on val_purged -- the correct partition

        magnitude_models, _ = train_quantile_regressor(
            train_purged, val_purged, val_purged,
            feature_cols=self.feature_cols, categorical_cols=[], target_col="magnitude", fold=0,
            n_optuna_trials=self.n_optuna_trials, early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
        )
        self._magnitude_models = magnitude_models

        # Feed Component B's trained feature importance back into Component
        # A's distance metric -- the exact hook analog_engine.set_weights()
        # exists for (see its docstring).
        weights = [direction_result.feature_importance[c] for c in self.feature_cols]
        self._analog_engine.set_weights(weights)

        self._fitted = True

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"df missing required feature columns: {sorted(missing)}")
        if "close" not in df.columns:
            raise ValueError("df missing 'close' column (needed to convert magnitude to a price fraction).")

        positions = pd.Series(0.0, index=df.index)
        if len(df) == 0:
            return positions

        X = df[self.feature_cols]

        # --- Component B: batched -- LightGBM predicts over the whole matrix at once. ---
        raw_probs = self._direction_model.predict(X, num_iteration=self._direction_model.best_iteration)
        calibrated = np.column_stack(
            [self._direction_model.calibrators_[k].predict(raw_probs[:, k]) for k in range(3)]
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        calibrated = calibrated / row_sums  # mirrors train_direction_classifier's own calibration step exactly
        p_down, p_up = calibrated[:, 0], calibrated[:, 2]  # DIRECTION_CLASS_LABELS = ["down", "neutral", "up"]

        mag_model = self._magnitude_models[0.5]
        mag_p50 = mag_model.predict(X, num_iteration=mag_model.best_iteration)
        mag_p50_frac = mag_p50 / df["close"].to_numpy(dtype=float)  # magnitude is an absolute price MFE, not a fraction

        # --- Component A: per-row -- predict() takes one query vector at a time. ---
        # This is the likely performance bottleneck for large date ranges;
        # not fixed here per "correctness first, speed second" -- if this is
        # too slow in practice, the fix is adding a batched query method to
        # HistoricalAnalogEngine, not vectorizing around its current API.
        feature_matrix = X.to_numpy(dtype=float)
        for i, ts in enumerate(df.index):
            forecast = self._analog_engine.predict(
                feature_matrix[i], distance_threshold=self._distance_threshold, query_timestamp=ts,
            )
            # query_timestamp=ts excludes reference rows at/after this exact
            # timestamp. Required because fit() was called on train+val: an
            # in-sample query here would otherwise be able to match against
            # itself (or a near-duplicate a few bars away) in the reference
            # set, producing a trivially perfect, meaningless "analog".
            if not forecast.is_valid:
                continue  # MASTER_CONTEXT constraint 9: N < 50 independent analogs -> stay flat

            b_up, b_down = p_up[i], p_down[i]
            if b_up == b_down:
                continue
            b_direction = 1.0 if b_up > b_down else -1.0
            a_direction = 1.0 if forecast.probability_up > forecast.probability_down else -1.0
            if b_direction != a_direction:
                continue  # Component A and B disagree on direction -> no edge, stay flat

            agreement = max(b_up, b_down)  # Component B's own calibrated confidence in its called direction
            if agreement < self.min_direction_confidence:
                continue

            size = min(abs(mag_p50_frac[i]) * self.magnitude_position_scale, self.max_position)
            positions.iloc[i] = b_direction * size

        return positions

    def ensemble_weights(self) -> Optional[Dict[str, float]]:
        """No formal linear ensemble meta-model exists yet (Layer 7C,
        Phase 2) to blend Component A and B -- generate_positions() combines
        them with a hard agreement rule, not a learned weight. Returning
        None here is deliberate and correct: it makes
        layer14_ensemble_weight_stability report not_evaluated rather than
        a fabricated pass."""
        return None

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PredictiveCoreStrategy.fit() must be called before generate_positions().")
