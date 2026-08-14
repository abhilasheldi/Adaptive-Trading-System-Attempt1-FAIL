"""
Component B -- Gradient-Boosted Quantile Regression.

Two independently trained LightGBM models sharing the same pooled,
walk-forward-split feature set:

  1. A magnitude quantile regressor (10th / 50th / 90th percentile).
  2. A duration quantile regressor (10th / 50th / 90th percentile).

The 3-class (up / down / neutral) direction classifier that used to live
here was DELETED at Stage 0 per MASTER_CONTEXT_v2 section 8. It carried 0.942
ensemble weight while measuring 0.004-0.016 logloss skill (S7), and the
class-weighting machinery that propped it up was compensating for barrier
arithmetic, not market structure (S1). Its isotonic calibrators went with
it. Do not reintroduce a direction classifier here: Stage D's secondary
model is a BINARY meta-label ("did this event reach target before stop?")
on a conditioned event set, which is a different problem and belongs in a
different module.

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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from .splitter import WalkForwardSplitter
from .trial_registry import DEFAULT_REGISTRY_PATH, optuna_logging_callback
from .uniqueness import average_uniqueness, uniqueness_sample_weights

# Stage A.1 label-metadata columns required to compute uniqueness.
_T0_COL, _T1_COL = "t0", "t1"


def _uniqueness_inputs(df: pd.DataFrame, categorical_cols: Sequence[str]):
    """(t0, t1, groups) if the frame carries Stage A.1 metadata, else None.

    `groups` is the asset column when present: concurrency must be computed
    per-asset on a pooled frame, or BTC and ETH labels sharing an hour would
    wrongly dilute each other. See uniqueness.py, property 1.
    """
    if _T0_COL not in df.columns or _T1_COL not in df.columns:
        return None
    groups = df["asset"] if "asset" in df.columns else None
    if groups is None and "asset" in categorical_cols:
        raise ValueError(
            "'asset' is declared categorical but absent from the frame; refusing to "
            "compute uniqueness that would silently pool concurrency across assets."
        )
    return df[_T0_COL], df[_T1_COL], groups


def _n_eff_or_none(df: pd.DataFrame, categorical_cols: Sequence[str]) -> Optional[float]:
    inputs = _uniqueness_inputs(df, categorical_cols)
    if inputs is None:
        return None
    return average_uniqueness(inputs[0], inputs[1], groups=inputs[2]).effective_n

optuna.logging.set_verbosity(optuna.logging.WARNING)

QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


# ---------------------------------------------------------------------- #
# Result containers
# ---------------------------------------------------------------------- #


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
    # --- Invariant I6: effective n travels with every nominal n ---------- #
    # None only when the frame lacked the Stage A.1 t0/t1 metadata needed to
    # compute it. None means "not computable here", never "no overlap" --
    # a consumer must not treat it as 1.0.
    n_eff_train: Optional[float] = None
    n_eff_val: Optional[float] = None
    n_eff_test: Optional[float] = None
    uniqueness_weighted: bool = False

    def n_eff_ratio_train(self) -> Optional[float]:
        if self.n_eff_train is None or not self.n_train:
            return None
        return self.n_eff_train / self.n_train


@dataclass
class FoldResult:
    fold: int
    magnitude: QuantileFoldResult
    duration: QuantileFoldResult


# ---------------------------------------------------------------------- #
# Quantile regressors (shared logic for magnitude and duration)
# ---------------------------------------------------------------------- #


def _quantile_objective(trial, X_train, y_train, X_val, y_val, categorical_cols, quantile,
                         random_state, early_stopping_rounds, train_sample_weight=None):
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
    # weight= applies ONLY to the training set. The validation objective
    # stays unweighted so early stopping and the Optuna score measure the
    # model on the raw val distribution -- weighting changes what the loss
    # optimises against, not how the result is judged.
    train_set = lgb.Dataset(X_train, label=y_train, weight=train_sample_weight,
                             categorical_feature=categorical_cols, free_raw_data=False)
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
    use_uniqueness_weights: bool = True,
    trial_registry_stage: Optional[str] = None,
    trial_registry_path=DEFAULT_REGISTRY_PATH,
) -> Tuple[Dict[float, lgb.Booster], QuantileFoldResult]:
    """Train magnitude/duration quantile regressors for one fold.

    use_uniqueness_weights : bool
        Weight each TRAINING sample by its average uniqueness u_bar_i (AFML
        ch. 4, BUILD_PLAN_v2 A.2), computed from the Stage A.1 `t0`/`t1`
        columns and grouped per asset. Overlapping triple-barrier labels are
        not independent observations; without this, a bar covered by nine
        overlapping labels contributes nine times to the loss.

        Silently becomes a no-op when the frame lacks `t0`/`t1` -- and
        `QuantileFoldResult.uniqueness_weighted` records which happened, so
        an unweighted fit is never mistaken for a weighted one.
    trial_registry_stage : str, optional
        When given, EVERY Optuna trial is appended to `trials.jsonl` under
        this stage label (MASTER_CONTEXT_v2 section 6). Not just the winning
        trial: DSR needs the size of the hypothesis space actually searched,
        and a search that tried 15 configurations and kept 1 has searched 15.

        Default None means no logging, so tests and exploratory calls do not
        pollute the registry with configurations that were never treated as
        results. Any run whose numbers are reported MUST set it.
    """
    model_cols = list(feature_cols) + [c for c in categorical_cols if c not in feature_cols]
    X_train, y_train = train_df[model_cols], train_df[target_col].to_numpy(dtype=float)
    X_val, y_val = val_df[model_cols], val_df[target_col].to_numpy(dtype=float)
    X_test, y_test = test_df[model_cols], test_df[target_col].to_numpy(dtype=float)

    # --- A.2: uniqueness weights, computed from TRAIN only --------------- #
    # Same isolation rule the rest of this module follows: the weights are a
    # property of the training partition's own label overlap and never see
    # val or test.
    train_sample_weight = None
    weighted = False
    if use_uniqueness_weights:
        inputs = _uniqueness_inputs(train_df, categorical_cols)
        if inputs is not None:
            train_sample_weight, _ = uniqueness_sample_weights(
                inputs[0], inputs[1], groups=inputs[2]
            )
            weighted = True

    # I6: effective n reported for every partition, alongside nominal n.
    n_eff_train = _n_eff_or_none(train_df, categorical_cols)
    n_eff_val = _n_eff_or_none(val_df, categorical_cols)
    n_eff_test = _n_eff_or_none(test_df, categorical_cols)

    models: Dict[float, lgb.Booster] = {}
    best_params_by_q: Dict[float, dict] = {}
    val_loss_by_q: Dict[float, float] = {}
    test_loss_by_q: Dict[float, float] = {}
    test_preds: Dict[float, np.ndarray] = {}

    for q in quantiles:
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        callbacks = []
        if trial_registry_stage is not None:
            callbacks.append(optuna_logging_callback(
                stage=trial_registry_stage,
                extra_params={
                    "model": "lgb_quantile", "target_col": target_col,
                    "quantile": q, "fold": fold,
                    "uniqueness_weighted": weighted,
                    "early_stopping_rounds": early_stopping_rounds,
                },
                n_eff=n_eff_train,
                nominal_n=len(train_df),
                path=trial_registry_path,
            ))

        study.optimize(
            lambda trial: _quantile_objective(
                trial, X_train, y_train, X_val, y_val, list(categorical_cols), q,
                random_state, early_stopping_rounds, train_sample_weight
            ),
            n_trials=n_optuna_trials,
            show_progress_bar=False,
            callbacks=callbacks,
        )
        best_params = dict(study.best_params)
        best_params.update({
            "objective": "quantile", "alpha": q, "metric": "quantile",
            "verbosity": -1, "random_state": random_state,
        })

        train_set = lgb.Dataset(X_train, label=y_train, weight=train_sample_weight,
                                 categorical_feature=list(categorical_cols),
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
        n_eff_train=n_eff_train,
        n_eff_val=n_eff_val,
        n_eff_test=n_eff_test,
        uniqueness_weighted=weighted,
    )
    return models, result


# ---------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------- #


class WalkForwardTrainer:
    """
    Runs Component B end-to-end across every walk-forward fold produced by a
    `WalkForwardSplitter`, training both quantile regressors per fold with
    strict train/val/test isolation.

    NOTE (Stage 0): this orchestrator drives `WalkForwardSplitter`, a
    row-count splitter with NO purging and NO embargo. Stage A.3 replaces
    that splitting logic with Purged K-Fold, because overlapping
    triple-barrier labels leak across a naive row-count boundary. Results
    produced through this path before Stage A lands are not trustworthy.
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

            results.append(FoldResult(fold, magnitude_result, duration_result))

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
