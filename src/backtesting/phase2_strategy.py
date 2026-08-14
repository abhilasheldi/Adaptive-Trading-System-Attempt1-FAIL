"""
Phase2Strategy -- `PredictiveCoreStrategy` extended with the Phase 2
conditioning layers that are actually wireable today, each behind its own
on/off flag so a walk-forward ABLATION can measure each layer's marginal
contribution instead of assuming it helps.

=========================================================================
WHAT IS AND IS NOT INTEGRATED, AND WHY
=========================================================================
MASTER_CONTEXT's hypothesis is conditional on four things holding at once:
post-compression, a liquidity sweep, a risk-on macro regime, and
multi-timeframe structural alignment. Of the Phase 2 layers built to test
those conditions, only some can be wired up against the data this project
currently ingests:

  Layer 5 (Liquidity Trap)      WIRED -- operates on raw OHLCV, which the
                                modeling frame carries. Enters the ensemble
                                as its 5th approved signal.
  Layer 6 (Volatility Regime)   WIRED, as a GATE not an ensemble input.
                                Layer 6 is not one of Layer 7C's five
                                approved inputs; its spec role is entry
                                gate #4 of MASTER_CONTEXT constraint 8
                                ("volatility regime is not Spike") and the
                                Risk Engine's fat-tail multiplier.
  Layer 7C (Ensemble Meta-Model) WIRED -- replaces the Phase 1 hard
                                agreement rule with the real linear
                                `sum(w_i * p_i)`, weights fit on the
                                VALIDATION partition only.

  Layer 3 (Macro Regime)        NOT WIRED -- needs DXY/VIX/QQQ/BTC-dominance
                                series aligned per bar, which this project
                                does not ingest (`data/raw/` holds only
                                BTC/ETH OHLCV), plus an economic calendar
                                for the Macro-Event Window state.
  Layer 4 (MTF Alignment)       NOT WIRED -- requires exactly four
                                timeframes (1H/4H/1D/1W); `config.yaml` has
                                `["1h", "4h"]`. The module raises rather
                                than accepting fewer, deliberately.
  Layer 9 (Risk Engine)         NOT WIRED -- needs `stop_distance_pct` and
                                `position_size_pct` from Layers 10/11,
                                which do not exist.

The two unwired ensemble inputs are passed to `EnsembleSignalSet` as None,
which excludes them and renormalizes the remaining weights for that row --
the same treatment Component A already gets when its N>=50 gate withholds a
forecast. They are NOT passed as fabricated "neutral" results: a neutral
distribution is not an absence of evidence, it actively pulls the blend
toward neutral in proportion to its weight, so an unavailable layer would
silently damp every prediction instead of stepping out of the way.

=========================================================================
BASELINE EQUIVALENCE -- what makes the ablation trustworthy
=========================================================================
With all three flags False, this class is behaviourally identical to
`PredictiveCoreStrategy`: same Component A/B training, same hard
agreement rule, same magnitude-scaled sizing. That is asserted directly by
`tests/test_phase2_strategy.py::test_all_layers_off_matches_phase1_baseline`
rather than assumed, and it is what makes "layer X changed OOS Sharpe by
N" a statement about layer X rather than about incidental drift between
two separately-written strategies.

Everything Phase 1's adapter documented about its own workarounds still
applies verbatim and is not repeated here: the `test_df` placeholder
argument, the uncalibrated-softmax reapplication of `.calibrators_`,
avoiding `WalkForwardTrainer`'s incompatible row-count splitter, and the
label-horizon purge at fold boundaries. See
`backtesting.predictive_core_strategy`'s module docstring.

=========================================================================
GATING HONESTY
=========================================================================
MASTER_CONTEXT constraint 8 lists six entry gates. This strategy enforces
TWO of them: the analog sample-size gate (N>=50, free from
`AnalogForecast.is_valid`) and the volatility-regime gate (not Spike, from
Layer 6) when `use_volatility_gate=True`. The EV-lower-CI-bound gate, MTF
alignment, macro regime, and risk score still require layers that are
unbuilt or unwired. Position sizing remains the Phase 1 placeholder --
magnitude-scaled and capped at `max_position`, NOT quarter-Kelly (Layer 10).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.macro_mtf_liquidity_vol.liquidity_trap import (
    EQUAL_LEVEL_LOOKBACK,
    LiquidityTrapResult,
    estimate_liquidity_trap,
)
from src.macro_mtf_liquidity_vol.volatility_classifier import (
    SPIKE_LOOKBACK_CANDLES,
    classify_volatility_state,
)
from src.predictive_core.analog_engine import HistoricalAnalogEngine
from src.predictive_core.boosted_core import train_direction_classifier, train_quantile_regressor
from src.predictive_core.dataset import RAW_ATR_COLUMN
from src.predictive_core.ensemble import (
    ComponentBSignal,
    EnsembleMetaModel,
    EnsembleSignalSet,
)

# Layer 6 needs a percentile-ranked ATR (Layer 2 already computes it) and a
# short raw ATR sequence for the spike run-up check.
_ATR_PCT_RANK_COL = "vol_atr_pct_rank"


class Phase2Strategy:
    def __init__(
        self,
        feature_cols: Sequence[str],
        *,
        use_liquidity_trap: bool = True,
        use_volatility_gate: bool = True,
        use_ensemble: bool = True,
        label_horizon: int = 48,
        distance_threshold_percentile: float = 10.0,
        min_direction_confidence: float = 0.55,
        magnitude_position_scale: float = 5.0,
        max_position: float = 0.02,
        n_optuna_trials: int = 15,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        use_liquidity_trap : bool
            Layer 5 supplies the ensemble's `liquidity_trap` signal. Has no
            effect unless `use_ensemble=True` -- Layer 5 is an ensemble
            input, not a standalone gate.
        use_volatility_gate : bool
            Layer 6 blocks new entries when the volatility state reads
            Spike (MASTER_CONTEXT constraint 8's gate #4, and Layer 17's
            kill-switch condition). Independent of the ensemble.
        use_ensemble : bool
            Layer 7C replaces the Phase 1 hard agreement rule with the real
            linear blend. Weights are fit on the validation partition only.

        All remaining parameters carry the same meaning, and the same
        caveats, as `PredictiveCoreStrategy`'s -- see that module.
        """
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty.")
        if label_horizon <= 0:
            raise ValueError("label_horizon must be positive.")

        self.feature_cols = list(feature_cols)
        self.use_liquidity_trap = use_liquidity_trap
        self.use_volatility_gate = use_volatility_gate
        self.use_ensemble = use_ensemble
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
        self._direction_result = None
        self._ensemble: Optional[EnsembleMetaModel] = None
        self._ensemble_weights: Optional[Dict[str, float]] = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Layer 5 / Layer 6 per-row computation
    # ------------------------------------------------------------------ #

    def required_columns(self) -> set:
        """Columns this configuration needs beyond the feature columns."""
        needed = {"close"}
        if self.use_liquidity_trap:
            needed |= {"high", "low", RAW_ATR_COLUMN}
        if self.use_volatility_gate:
            needed |= {_ATR_PCT_RANK_COL, RAW_ATR_COLUMN}
        return needed

    def _liquidity_trap_series(self, df: pd.DataFrame) -> List[Optional[LiquidityTrapResult]]:
        """Layer 5 evaluated per row, using only that row's trailing window.

        `estimate_liquidity_trap` evaluates the LAST row of whatever frame it
        is handed and builds its clusters from the strictly-prior
        `EQUAL_LEVEL_LOOKBACK` candles, so slicing `df.iloc[i - lookback : i + 1]`
        gives a genuinely point-in-time read with no lookahead. Rows without
        a full trailing window return None (the layer is unavailable there),
        which the ensemble excludes and renormalizes around -- it does not
        fabricate a reading.
        """
        results: List[Optional[LiquidityTrapResult]] = [None] * len(df)
        if not self.use_liquidity_trap:
            return results
        window = EQUAL_LEVEL_LOOKBACK
        for i in range(window, len(df)):
            slice_ = df.iloc[i - window : i + 1]
            try:
                results[i] = estimate_liquidity_trap(slice_, atr_col=RAW_ATR_COLUMN)
            except ValueError:
                # Non-finite prices / non-positive ATR in this window: the
                # layer legitimately refuses to read. Left as None (absent)
                # rather than silently substituted with a neutral result.
                results[i] = None
        return results

    def _spike_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Layer 6: True where the volatility state reads Spike (entry blocked).

        Rows without a full trailing ATR window, or whose inputs the
        classifier rejects, are treated as NOT-Spike: this gate can only
        ever remove trades, and a missing reading is not evidence of a
        tail-risk event. That is a deliberate fail-open choice for a
        RESEARCH ablation, where wrongly suppressing trades would corrupt
        the very Sharpe comparison being measured. It is NOT appropriate for
        live trading, where MASTER_CONTEXT's capital-preservation-first
        ordering requires the opposite default.
        """
        mask = np.zeros(len(df), dtype=bool)
        if not self.use_volatility_gate:
            return mask
        atr = df[RAW_ATR_COLUMN].to_numpy(dtype=float)
        pct_rank = df[_ATR_PCT_RANK_COL].to_numpy(dtype=float)
        for i in range(SPIKE_LOOKBACK_CANDLES, len(df)):
            window = atr[i - SPIKE_LOOKBACK_CANDLES : i + 1]
            rank = pct_rank[i]
            if not np.isfinite(rank) or not np.all(np.isfinite(window)) or np.any(window <= 0):
                continue
            state = classify_volatility_state(float(np.clip(rank, 0.0, 1.0)), list(window))
            mask[i] = state.state == "Spike"
        return mask

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        required = set(self.feature_cols) | {"direction", "magnitude", "duration"} | self.required_columns()
        for name, frame in (("train_df", train_df), ("val_df", val_df)):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{name} missing required columns: {sorted(missing)}")

        self._check_label_horizon_matches_upstream(train_df, "train_df")
        self._check_label_horizon_matches_upstream(val_df, "val_df")

        train_purged = train_df.iloc[: -self.label_horizon] if len(train_df) > self.label_horizon else train_df.iloc[:0]
        val_purged = val_df.iloc[: -self.label_horizon] if len(val_df) > self.label_horizon else val_df.iloc[:0]
        if train_purged.empty or val_purged.empty:
            raise ValueError(
                f"train_df/val_df too short relative to label_horizon={self.label_horizon} "
                f"after boundary purging (train_purged={len(train_purged)}, val_purged={len(val_purged)})."
            )

        self._analog_engine = HistoricalAnalogEngine(
            feature_cols=self.feature_cols,
            min_independent_instances=50,  # MASTER_CONTEXT constraint 9 -- exact spec number
            min_temporal_separation=self.label_horizon,
        )
        self._analog_engine.fit(pd.concat([train_purged, val_purged]))
        self._distance_threshold = self._analog_engine.suggest_distance_threshold(
            percentile=self.distance_threshold_percentile
        )

        model, direction_result = train_direction_classifier(
            train_purged, val_purged, val_purged,
            feature_cols=self.feature_cols, categorical_cols=[],
            fold=0, n_optuna_trials=self.n_optuna_trials,
            early_stopping_rounds=self.early_stopping_rounds, random_state=self.random_state,
        )
        self._direction_model = model
        self._direction_result = direction_result

        magnitude_models, _ = train_quantile_regressor(
            train_purged, val_purged, val_purged,
            feature_cols=self.feature_cols, categorical_cols=[], target_col="magnitude", fold=0,
            n_optuna_trials=self.n_optuna_trials, early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
        )
        self._magnitude_models = magnitude_models

        weights = [direction_result.feature_importance[c] for c in self.feature_cols]
        self._analog_engine.set_weights(weights)
        self._fitted = True

        # --- Layer 7C: weights fit on the VALIDATION partition only. ---
        # Deliberately after `self._fitted = True`, because building the
        # per-row signal sets requires Component A/B inference, which
        # requires the models above. val_purged is used (not raw val_df) so
        # the ensemble never sees rows whose labels reach past the fold edge.
        if self.use_ensemble:
            signal_sets = self._build_signal_sets(val_purged)
            y = val_purged["direction"].to_numpy()
            # The EnsembleMetaModel is created ONCE and reused across folds,
            # never recreated per fold. This matters: its weight-stability
            # check compares each fold's learned weights against the previous
            # fold's, and switches to the equal-weight fallback when any
            # single weight moves by more than 0.15 (the Adaptive doc's exact
            # number). A fresh model per fold would have an empty history
            # every time, so that rule -- and its fallback -- could never
            # fire, silently discarding the spec behaviour. The harness
            # reuses one strategy instance across folds, which is what makes
            # this work.
            if self._ensemble is None:
                self._ensemble = EnsembleMetaModel()
            fit_result = self._ensemble.fit(signal_sets, y)
            self._ensemble_weights = dict(fit_result.weights)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def _component_predictions(self, df: pd.DataFrame):
        """Component A forecasts + Component B calibrated probabilities and
        median magnitude, for every row of `df`."""
        X = df[self.feature_cols]
        raw_probs = self._direction_model.predict(X, num_iteration=self._direction_model.best_iteration)
        calibrated = np.column_stack(
            [self._direction_model.calibrators_[k].predict(raw_probs[:, k]) for k in range(3)]
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        calibrated = calibrated / row_sums

        mag_model = self._magnitude_models[0.5]
        mag_p50 = mag_model.predict(X, num_iteration=mag_model.best_iteration)
        mag_p50_frac = mag_p50 / df["close"].to_numpy(dtype=float)

        feature_matrix = X.to_numpy(dtype=float)
        analogs = [
            self._analog_engine.predict(
                feature_matrix[i], distance_threshold=self._distance_threshold, query_timestamp=ts,
            )
            for i, ts in enumerate(df.index)
        ]
        return analogs, calibrated, mag_p50_frac

    def _build_signal_sets(self, df: pd.DataFrame) -> List[EnsembleSignalSet]:
        analogs, calibrated, _ = self._component_predictions(df)
        traps = self._liquidity_trap_series(df)
        sets: List[EnsembleSignalSet] = []
        for i in range(len(df)):
            sets.append(EnsembleSignalSet(
                component_a=analogs[i],
                component_b=ComponentBSignal(
                    p_down=float(calibrated[i, 0]),
                    p_neutral=float(calibrated[i, 1]),
                    p_up=float(calibrated[i, 2]),
                    magnitude_p10=0.0, magnitude_p50=0.0, magnitude_p90=0.0,
                    duration_p10=0.0, duration_p50=0.0, duration_p90=0.0,
                ),
                macro_regime=None,     # Layer 3 not wired -- see module docstring
                mtf_alignment=None,    # Layer 4 not wired -- see module docstring
                liquidity_trap=traps[i],
            ))
        return sets

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        missing = (set(self.feature_cols) | self.required_columns()) - set(df.columns)
        if missing:
            raise ValueError(f"df missing required columns: {sorted(missing)}")

        positions = pd.Series(0.0, index=df.index)
        if len(df) == 0:
            return positions

        analogs, calibrated, mag_p50_frac = self._component_predictions(df)
        p_down, p_up = calibrated[:, 0], calibrated[:, 2]
        spike = self._spike_mask(df)
        traps = self._liquidity_trap_series(df)

        for i in range(len(df)):
            if spike[i]:
                continue  # Layer 6 gate: MASTER_CONTEXT constraint 8, gate #4

            forecast = analogs[i]
            if not forecast.is_valid:
                continue  # MASTER_CONTEXT constraint 9: N < 50 independent analogs

            if self.use_ensemble:
                blended = self._ensemble.predict(EnsembleSignalSet(
                    component_a=forecast,
                    component_b=ComponentBSignal(
                        p_down=float(calibrated[i, 0]),
                        p_neutral=float(calibrated[i, 1]),
                        p_up=float(calibrated[i, 2]),
                        magnitude_p10=0.0, magnitude_p50=0.0, magnitude_p90=0.0,
                        duration_p10=0.0, duration_p50=0.0, duration_p90=0.0,
                    ),
                    macro_regime=None,
                    mtf_alignment=None,
                    liquidity_trap=traps[i],
                ))
                if blended.called_direction == "neutral":
                    continue
                direction = 1.0 if blended.called_direction == "up" else -1.0
                confidence = blended.called_probability
            else:
                # Phase 1 hard agreement rule, preserved byte-for-byte so
                # the all-flags-off configuration reproduces the baseline.
                b_up, b_down = p_up[i], p_down[i]
                if b_up == b_down:
                    continue
                direction = 1.0 if b_up > b_down else -1.0
                a_direction = 1.0 if forecast.probability_up > forecast.probability_down else -1.0
                if direction != a_direction:
                    continue
                confidence = max(b_up, b_down)

            if confidence < self.min_direction_confidence:
                continue

            size = min(abs(mag_p50_frac[i]) * self.magnitude_position_scale, self.max_position)
            positions.iloc[i] = direction * size

        return positions

    # ------------------------------------------------------------------ #
    # harness protocol
    # ------------------------------------------------------------------ #

    def ensemble_weights(self) -> Optional[Dict[str, float]]:
        """This fold's fitted Layer 7C weights, or None when the ensemble is
        disabled (which correctly makes the Layer 14 ensemble-stability
        criterion report `not_evaluated` rather than a fabricated pass)."""
        return self._ensemble_weights

    def diagnostics(self) -> Optional[Dict]:
        if self._direction_result is None:
            return None
        r = self._direction_result
        out = {
            "direction_val_logloss": r.val_logloss,
            "direction_val_partition_logloss": r.test_logloss,
            "direction_val_partition_logloss_baseline": r.test_logloss_baseline,
            "direction_val_partition_logloss_skill_score": r.test_logloss_skill_score,
            "layers_enabled": {
                "liquidity_trap_layer5": self.use_liquidity_trap,
                "volatility_gate_layer6": self.use_volatility_gate,
                "ensemble_layer7c": self.use_ensemble,
            },
            "direction_val_partition_discrimination": [
                {"class_label": d.class_label, "precision": d.precision,
                 "recall": d.recall, "f1": d.f1, "support": d.support}
                for d in r.discrimination_reports
            ],
        }
        if self._ensemble_weights is not None:
            out["ensemble_weights"] = dict(self._ensemble_weights)
        return out

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Phase2Strategy.fit() must be called before generate_positions().")

    def _check_label_horizon_matches_upstream(self, df: pd.DataFrame, name: str) -> None:
        """Identical contract to `PredictiveCoreStrategy`'s -- see that
        module's docstring for why absence of `.attrs` metadata is a skip,
        not a failure."""
        upstream_horizon = df.attrs.get("label_horizon")
        if upstream_horizon is None:
            return
        if upstream_horizon != self.label_horizon:
            raise ValueError(
                f"label_horizon mismatch on {name}: configured {self.label_horizon}, but "
                f"apply_triple_barrier recorded max_candles={upstream_horizon} in "
                f".attrs['label_horizon'] at labeling time."
            )


__all__ = ["Phase2Strategy"]
