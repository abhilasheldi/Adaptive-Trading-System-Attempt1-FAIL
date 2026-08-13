"""
Discrimination/skill diagnostics for the Component B direction classifier --
complementary to, and NOT a substitute for, `calibration.py`.

Why this module has to exist separately from calibration: calibration (ECE,
Brier) answers "when the model says X%, is it right X% of the time" --
reliability. It does NOT answer "does the model's confidence actually change
based on what it's looking at" -- resolution/discrimination. A classifier
that always outputs the TRAINING set's base rates, regardless of input, is
PERFECTLY calibrated (a constant predictor outputting the true marginal
probability is calibrated by definition) while having learned nothing real.
With this codebase's triple-barrier geometry default (tp_mult=2.0,
sl_mult=1.0 -- an implementation choice in labeling.py/config.yaml, NOT a
number MASTER_CONTEXT itself specifies; see labeling.py's docstring)
producing a heavily skewed real-data label split (~65% down / 34% up / <1%
neutral on BTC 2023-2026, empirically), "predict the majority class" is a
very easy local optimum for a classifier to fall into without this module,
calibration would not catch it.

Two independent checks, both computed on the TEST fold only, both scored
against a baseline derived from TRAIN only (never test) so the baseline
itself doesn't peek at test labels -- the same discipline `boosted_core`
already applies to model fitting:

1. Per-class precision/recall/F1 (`multiclass_discrimination_report`) --
   MASTER_CONTEXT's own Tier 2 diagnostic table names this "Signal Quality
   (Precision/Recall)" as the explicit metric for evaluating the Layer 7B
   direction classifier. A majority-class parrot shows near-zero recall on
   the minority class(es) -- the direct, intuitive signature of this
   failure mode.
2. Logloss skill score (`baseline_logloss` + `logloss_skill_score`) -- how
   much better is the real model's test logloss than a trivial baseline
   that always predicts the TRAIN fold's class base rates? A skill score
   near 0 means "no better than guessing the base rate"; negative means
   worse than that trivial baseline (a red flag in itself).
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.metrics import log_loss, precision_recall_fscore_support


@dataclass(frozen=True)
class DiscriminationReport:
    """Precision/recall/F1 for ONE class, computed on the test fold from the
    classifier's ARGMAX prediction -- deliberately the same decision rule
    `PredictiveCoreStrategy` actually uses to call a direction (argmax of
    the calibrated probability matrix), not some other threshold, so this
    diagnostic reflects what the strategy would actually do."""

    class_label: str
    precision: float
    recall: float
    f1: float
    support: int


def multiclass_discrimination_report(
    y_true: np.ndarray, predicted_class: np.ndarray, class_labels: Sequence[str]
) -> List[DiscriminationReport]:
    """One DiscriminationReport per class. `predicted_class` must already be
    the argmax class index (0/1/2), not a probability matrix -- precision/
    recall are inherently about discrete decisions, unlike calibration."""
    y_true = np.asarray(y_true)
    predicted_class = np.asarray(predicted_class)
    n_classes = len(class_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted_class, labels=list(range(n_classes)), zero_division=0
    )
    return [
        DiscriminationReport(
            class_label=class_labels[k],
            precision=float(precision[k]),
            recall=float(recall[k]),
            f1=float(f1[k]),
            support=int(support[k]),
        )
        for k in range(n_classes)
    ]


def baseline_logloss(y_train_class_idx: np.ndarray, y_test_class_idx: np.ndarray, n_classes: int) -> float:
    """
    Logloss a trivial "always predict the TRAIN fold's class base rates"
    classifier would achieve on the TEST fold. Deliberately uses TRAIN
    (never test) to derive the base rates being predicted -- the baseline
    must be constructible under the exact same information constraint the
    real model is trained under, or the comparison isn't apples-to-apples.
    """
    y_train_class_idx = np.asarray(y_train_class_idx)
    y_test_class_idx = np.asarray(y_test_class_idx)
    train_base_rates = np.array(
        [np.mean(y_train_class_idx == k) for k in range(n_classes)], dtype=float
    )
    baseline_pred = np.tile(train_base_rates, (len(y_test_class_idx), 1))
    return float(log_loss(y_test_class_idx, baseline_pred, labels=list(range(n_classes))))


def logloss_skill_score(model_logloss: float, reference_logloss: float) -> float:
    """
    1 - (model_logloss / reference_logloss). 0 means the model is no better
    than the trivial base-rate baseline (no real skill); 1 would mean
    perfect (zero-logloss) prediction; negative means the model is WORSE
    than just guessing the base rate for every row -- itself a red flag
    (typically overfitting or a genuine bug, not merely "no edge").

    Standard "skill score" construction (same idea used for Brier skill
    scores in weather forecasting): compare a proper scoring rule's loss
    against a naive reference forecast's loss on the same evaluation set.
    """
    if reference_logloss <= 0 or not np.isfinite(reference_logloss):
        return float("nan")
    return 1.0 - (model_logloss / reference_logloss)
