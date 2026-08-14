"""
Calibration measurement: Expected Calibration Error and Brier score.

Spec requirement: "Calibration is checked explicitly: if the model says 70%,
it should be right 70% of the time, not 58%." A classifier that merely
outputs softmax scores is NOT calibrated by default -- gradient boosted
trees are known to produce overconfident probabilities. This module provides
the numbers that prove (or disprove) calibration, rather than asserting it.

STAGE 0 STATUS: this module is RETAINED but currently has no caller in
`src/`. Its previous caller, the Component B direction classifier, was
deleted at Stage 0 (MASTER_CONTEXT_v2 section 8).

It was retained deliberately rather than deleted alongside that classifier.
The Register's "calibration work premised on that imbalance" refers to the
isotonic calibrators that were FIT to the 65/34 three-class problem -- those
were deleted with the classifier. ECE and Brier are geometry-independent
measurements of any probabilistic classifier and are not premised on any
class balance. MASTER_CONTEXT_v2 section 1 ranks the measurement apparatus
above the models, and BUILD_PLAN_v2 D.3 names "calibration on the event set"
as a required secondary measurement for the Stage D meta-label model. This
module is that measurement, waiting for its Stage D caller.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    """Reports for ONE class (e.g. 'up') within a possibly-multiclass problem."""

    class_label: str
    n_samples: int
    expected_calibration_error: float
    brier_score: float
    bin_edges: List[float]
    bin_confidence: List[float]  # mean predicted probability per bin
    bin_accuracy: List[float]    # empirical frequency of positive class per bin
    bin_counts: List[int]


def expected_calibration_error(
    y_true_binary: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    """
    Computes the Expected Calibration Error (ECE) for one class in a
    one-vs-rest sense: bins predictions by confidence, and for each bin
    compares mean predicted probability against the empirical frequency of
    the positive class actually occurring in that bin. ECE is the
    sample-weighted mean absolute gap between the two across bins -- a
    single number, 0 = perfectly calibrated, higher = worse.

    Parameters
    ----------
    y_true_binary : np.ndarray of {0, 1}
        Ground truth for this class, one-vs-rest.
    y_prob : np.ndarray of float in [0, 1]
        Predicted probability of this class.
    n_bins : int
        Number of equal-width confidence bins.
    """
    y_true_binary = np.asarray(y_true_binary, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true_binary.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true_binary and y_prob must be the same length.")
    if y_true_binary.shape[0] == 0:
        raise ValueError("Cannot compute calibration on an empty array.")

    n = y_true_binary.shape[0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    bin_conf, bin_acc, bin_cnt = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
            bin_cnt.append(0)
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true_binary[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
        bin_conf.append(conf)
        bin_acc.append(acc)
        bin_cnt.append(cnt)

    brier = float(np.mean((y_prob - y_true_binary) ** 2))

    return CalibrationReport(
        class_label="",
        n_samples=n,
        expected_calibration_error=float(ece),
        brier_score=brier,
        bin_edges=bin_edges.tolist(),
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
        bin_counts=bin_cnt,
    )


def multiclass_calibration_report(
    y_true: np.ndarray, y_prob_matrix: np.ndarray, class_labels: List[str], n_bins: int = 10
) -> List[CalibrationReport]:
    """
    One-vs-rest calibration report per class, for a multiclass problem
    (e.g. up / down / neutral). Returns one CalibrationReport per class,
    each carrying its own ECE and Brier score -- these are the numbers that
    "calibration is checked explicitly" cashes out to.
    """
    y_true = np.asarray(y_true)
    y_prob_matrix = np.asarray(y_prob_matrix, dtype=float)
    if y_prob_matrix.shape[1] != len(class_labels):
        raise ValueError("y_prob_matrix column count must match len(class_labels).")

    reports = []
    for k, label in enumerate(class_labels):
        y_true_binary = (y_true == k).astype(float)
        rep = expected_calibration_error(y_true_binary, y_prob_matrix[:, k], n_bins=n_bins)
        reports.append(
            CalibrationReport(
                class_label=label,
                n_samples=rep.n_samples,
                expected_calibration_error=rep.expected_calibration_error,
                brier_score=rep.brier_score,
                bin_edges=rep.bin_edges,
                bin_confidence=rep.bin_confidence,
                bin_accuracy=rep.bin_accuracy,
                bin_counts=rep.bin_counts,
            )
        )
    return reports