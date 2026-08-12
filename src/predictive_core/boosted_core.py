"""
Component B -- Gradient-Boosted Classification and Quantile Regression.

Three independently trained LightGBM models sharing the same pooled,
walk-forward-split feature set:

  1. A calibrated 3-class (up / down / neutral) direction classifier.
  2. A magnitude quantile regressor (10th / 50th / 90th percentile).
  3. A duration quantile regressor (10th / 50th / 90th percentile).

Hard isolation rule enforced throughout this module: hyperparameter tuning
(Optuna) and early stopping are computed ONLY against the validation fold.
The test fold is passed into functions separately and only AFTER the model
and its hyperparameters are already fixed -- objective closures used by
Optuna never receive a reference to the test fold, so leakage isn't just
"avoided by discipline", it is structurally impossible to write by
accident within this module without deliberately threading test data into
a closure that doesn't currently accept it.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from .calibration import multiclass_calibration_report, CalibrationReport
from .splitter import WalkForwardSplitter

optuna.logging.set_verbosity(optuna.logging.WARNING)

DIRECTION_CLASS_LABELS = ["down", "neutral", "up"]  # index 0, 1, 2
QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


def _direction_to_class_idx(direction: np.ndarray) -> np.ndarray:
    """-1.0 (down) -> 0, 0.0 (neutral) -> 1, 1.0 (up) -> 2."""
    direction = np.asarray(direction, dtype=float)
    if not np.all(np.isin(direction, [-1.0, 0.0, 1.0])):
        raise ValueError("direction column must contain only -1.0, 0.0, or 1.0.")
    return (direction + 1).astype(int)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


# ---------------------------------------------------------------------- #
# Result containers
# ---------------------------------------------------------------------- #


@dataclass
class DirectionFoldResult:
    fold: int
    best_params: dict
    n_train: int
    n_val: int
    n_test: int
    val_logloss: float
    test_logloss: float
    calibration_reports: List[CalibrationReport]  # computed on TEST fold, post-calibration
    feature_importance: Dict[str, float]  # normalized gain importance -> feeds Component A weights


@dataclass
class QuantileFoldResult:
    fold: int
    target_col: str
    best_params_by_quantile: Dict[float, dict]
    n_train: int
    n_val: int
    n_test: int
    val_pinball_loss: Dict[float, float]
    test_pinball_loss: Dict[float, float]
    # Sanity check: 10th percentile prediction should be <= 90th percentile
    # prediction for (almost) every test row. Reported as a fraction so
    # quantile-crossing pathology is visible as a number, not silently ignored.
    quantile_crossing_rate: float


@dataclass
class FoldResult:
    fold: int
    direction: DirectionFoldResult
    magnitude: QuantileFoldResult
    duration: QuantileFoldResult


# ---------------------------------------------------------------------- #
# Direction classifier
# ---------------------------------------------------------------------- #


def _direction_objective(trial, X_train, y_train, X_val, y_val, categorical_cols, random_state,
                          early_stopping_rounds):
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "random_state": random_state,
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    # Closure captures ONLY train/val -- test is not an argument to this
    # function and therefore cannot leak into the objective, by construction.
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols,
                             free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols,
                           reference=train_set, free_raw_data=False)
    model = lgb.train(
        params, train_set, num_boost_round=1000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return log_loss(y_val, val_pred, labels=[0, 1, 2])


def train_direction_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    fold: int,
    n_optuna_trials: int = 15,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> Tuple[lgb.Booster, DirectionFoldResult]:
    model_cols = list(feature_cols) + [c for c in categorical_cols if c not in feature_cols]
    X_train, y_train = train_df[model_cols], _direction_to_class_idx(train_df["direction"].to_numpy())
    X_val, y_val = val_df[model_cols], _direction_to_class_idx(val_df["direction"].to_numpy())
    X_test, y_test = test_df[model_cols], _direction_to_class_idx(test_df["direction"].to_numpy())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(
        lambda trial: _direction_objective(
            trial, X_train, y_train, X_val, y_val, list(categorical_cols), random_state, early_stopping_rounds
        ),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
    )

    best_params = dict(study.best_params)
    best_params.update({
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "verbosity": -1, "random_state": random_state,
    })

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=list(categorical_cols),
                             free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=list(categorical_cols),
                           reference=train_set, free_raw_data=False)
    final_model = lgb.train(
        best_params, train_set, num_boost_round=1000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred_raw = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    val_logloss = log_loss(y_val, val_pred_raw, labels=[0, 1, 2])

    # --- Calibration: fit isotonic (one-vs-rest, per class) on the
    # VALIDATION fold's raw predictions, never on test. Only after the
    # calibrator is frozen do we touch the test fold, purely to evaluate it. ---
    calibrators: List[IsotonicRegression] = []
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_pred_raw[:, k], (y_val == k).astype(float))
        calibrators.append(iso)

    test_pred_raw = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    test_pred_calibrated = np.column_stack([calibrators[k].predict(test_pred_raw[:, k]) for k in range(3)])
    row_sums = test_pred_calibrated.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard against degenerate all-zero rows
    test_pred_calibrated = test_pred_calibrated / row_sums

    test_logloss = log_loss(y_test, test_pred_calibrated, labels=[0, 1, 2])
    calibration_reports = multiclass_calibration_report(
        y_test, test_pred_calibrated, DIRECTION_CLASS_LABELS, n_bins=10
    )

    gain = final_model.feature_importance(importance_type="gain")
    gain_sum = gain.sum()
    normalized = gain / gain_sum if gain_sum > 0 else np.ones_like(gain) / len(gain)
    # Only report importances for the k-NN's own feature space (Component A's
    # feature_cols) -- the categorical 'asset' importance is real but isn't a
    # distance-metric weight the analog engine can use, so it's excluded here
    # rather than silently smuggled into set_weights() by a caller.
    full_importance = dict(zip(model_cols, normalized.tolist()))
    feature_importance = {c: full_importance[c] for c in feature_cols}

    result = DirectionFoldResult(
        fold=fold,
        best_params=best_params,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        val_logloss=float(val_logloss),
        test_logloss=float(test_logloss),
        calibration_reports=calibration_reports,
        feature_importance=feature_importance,
    )
    # Attach calibrators + model for downstream inference use.
    final_model.calibrators_ = calibrators  # type: ignore[attr-defined]
    return final_model, result


# ---------------------------------------------------------------------- #
# Quantile regressors (shared logic for magnitude and duration)
# ---------------------------------------------------------------------- #


def _quantile_objective(trial, X_train, y_train, X_val, y_val, categorical_cols, quantile,
                         random_state, early_stopping_rounds):
    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "verbosity": -1,
        "random_state": random_state,
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols,
                             free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols,
                           reference=train_set, free_raw_data=False)
    model = lgb.train(
        params, train_set, num_boost_round=1000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return _pinball_loss(y_val, val_pred, quantile)


def train_quantile_regressor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    fold: int,
    quantiles: Sequence[float] = QUANTILES,
    n_optuna_trials: int = 15,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> Tuple[Dict[float, lgb.Booster], QuantileFoldResult]:
    model_cols = list(feature_cols) + [c for c in categorical_cols if c not in feature_cols]
    X_train, y_train = train_df[model_cols], train_df[target_col].to_numpy(dtype=float)
    X_val, y_val = val_df[model_cols], val_df[target_col].to_numpy(dtype=float)
    X_test, y_test = test_df[model_cols], test_df[target_col].to_numpy(dtype=float)

    models: Dict[float, lgb.Booster] = {}
    best_params_by_q: Dict[float, dict] = {}
    val_loss_by_q: Dict[float, float] = {}
    test_loss_by_q: Dict[float, float] = {}
    test_preds: Dict[float, np.ndarray] = {}

    for q in quantiles:
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        study.optimize(
            lambda trial: _quantile_objective(
                trial, X_train, y_train, X_val, y_val, list(categorical_cols), q,
                random_state, early_stopping_rounds
            ),
            n_trials=n_optuna_trials,
            show_progress_bar=False,
        )
        best_params = dict(study.best_params)
        best_params.update({
            "objective": "quantile", "alpha": q, "metric": "quantile",
            "verbosity": -1, "random_state": random_state,
        })

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=list(categorical_cols),
                                 free_raw_data=False)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=list(categorical_cols),
                               reference=train_set, free_raw_data=False)
        model = lgb.train(
            best_params, train_set, num_boost_round=1000, valid_sets=[val_set],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)

        models[q] = model
        best_params_by_q[q] = best_params
        val_loss_by_q[q] = _pinball_loss(y_val, val_pred, q)
        test_loss_by_q[q] = _pinball_loss(y_test, test_pred, q)
        test_preds[q] = test_pred

    crossing_rate = float("nan")
    if 0.1 in test_preds and 0.9 in test_preds:
        crossing_rate = float(np.mean(test_preds[0.1] > test_preds[0.9]))

    result = QuantileFoldResult(
        fold=fold,
        target_col=target_col,
        best_params_by_quantile=best_params_by_q,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        val_pinball_loss=val_loss_by_q,
        test_pinball_loss=test_loss_by_q,
        quantile_crossing_rate=crossing_rate,
    )
    return models, result


# ---------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------- #


class WalkForwardTrainer:
    """
    Runs Component B end-to-end across every walk-forward fold produced by a
    `WalkForwardSplitter`, training the direction classifier and both
    quantile regressors per fold with strict train/val/test isolation.
    """

    def __init__(
        self,
        splitter: WalkForwardSplitter,
        feature_cols: Sequence[str],
        categorical_cols: Sequence[str] = ("asset",),
        n_optuna_trials: int = 15,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        self.splitter = splitter
        self.feature_cols = list(feature_cols)
        self.categorical_cols = list(categorical_cols)
        self.n_optuna_trials = n_optuna_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    def run(self, pooled_df: pd.DataFrame) -> List[FoldResult]:
        if not isinstance(pooled_df.index, pd.DatetimeIndex):
            raise TypeError("pooled_df index must be a DatetimeIndex.")

        required_cols = set(self.feature_cols) | set(self.categorical_cols) | {
            "direction", "magnitude", "duration"
        }
        missing = required_cols - set(pooled_df.columns)
        if missing:
            raise ValueError(f"pooled_df missing required columns: {sorted(missing)}")

        results: List[FoldResult] = []

        for fold, (train_pos, val_pos, test_pos) in enumerate(self.splitter.split(pooled_df)):
            # Positional slicing only -- see splitter.py docstring for why
            # `.loc` is unsafe here (duplicate timestamps across pooled assets).
            train_df = pooled_df.iloc[train_pos]
            val_df = pooled_df.iloc[val_pos]
            test_df = pooled_df.iloc[test_pos]

            self._assert_isolated(train_df, val_df, test_df, fold)

            for col in self.categorical_cols:
                for frame in (train_df, val_df, test_df):
                    if not isinstance(frame[col].dtype, pd.CategoricalDtype):
                        raise TypeError(
                            f"Column '{col}' must be categorical dtype before training "
                            f"(call pool_assets, which sets this)."
                        )

            _, direction_result = train_direction_classifier(
                train_df, val_df, test_df, self.feature_cols, self.categorical_cols, fold,
                n_optuna_trials=self.n_optuna_trials,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=self.random_state,
            )
            _, magnitude_result = train_quantile_regressor(
                train_df, val_df, test_df, self.feature_cols, self.categorical_cols,
                "magnitude", fold,
                n_optuna_trials=self.n_optuna_trials,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=self.random_state,
            )
            _, duration_result = train_quantile_regressor(
                train_df, val_df, test_df, self.feature_cols, self.categorical_cols,
                "duration", fold,
                n_optuna_trials=self.n_optuna_trials,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=self.random_state,
            )

            results.append(FoldResult(fold, direction_result, magnitude_result, duration_result))

        if not results:
            raise ValueError(
                "WalkForwardTrainer.run produced zero folds -- pooled_df is too short "
                "for the configured train/val/test window sizes."
            )
        return results

    @staticmethod
    def _assert_isolated(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                          fold: int) -> None:
        """Belt-and-suspenders re-check at the trainer boundary, independent
        of the splitter's own internal assertion, so a future change to
        either module still can't silently ship a leaking fold."""
        if train_df.index.max() > val_df.index.min():
            raise AssertionError(f"Fold {fold}: train/val temporal overlap detected at trainer boundary.")
        if val_df.index.max() > test_df.index.min():
            raise AssertionError(f"Fold {fold}: val/test temporal overlap detected at trainer boundary.")