"""
Tests for `predictive_core.skill_diagnostics` -- and specifically, a direct
proof that these diagnostics catch a classifier that has just learned to
parrot the training set's majority class, which calibration checks alone
cannot (see that module's docstring): a constant predictor outputting the
true base rate is perfectly calibrated by definition.
"""

import numpy as np

from src.predictive_core.skill_diagnostics import (
    baseline_logloss,
    logloss_skill_score,
    multiclass_discrimination_report,
)

# Mirrors the real, observed BTC 2023-2026 triple-barrier label split
# (~65% down / 34% up / <1% neutral) that motivated this module.
_DOWN, _NEUTRAL, _UP = 0, 1, 2
_CLASS_LABELS = ["down", "neutral", "up"]


def _make_skewed_labels(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([_DOWN, _NEUTRAL, _UP], size=n, p=[0.65, 0.01, 0.34])


def test_majority_class_parrot_has_near_zero_skill_score_and_zero_up_recall():
    """
    The exact scenario this module exists to catch: a "classifier" that
    always predicts the training set's majority class regardless of input.
    It must show a logloss skill score near 0 (no better than the trivial
    base-rate baseline) and ~0 recall on the minority "up" class -- even
    though, if you only looked at raw accuracy, it would appear to be right
    about 65% of the time.
    """
    y_train = _make_skewed_labels(5000, seed=1)
    y_test = _make_skewed_labels(2000, seed=2)

    # A constant predictor: always calls "down" (the majority class),
    # regardless of any input features -- literally has no per-input signal.
    predicted_class = np.full(len(y_test), _DOWN)

    accuracy = float(np.mean(predicted_class == y_test))
    assert accuracy > 0.6, "sanity check: this constant predictor should look deceptively accurate"

    reports = multiclass_discrimination_report(y_test, predicted_class, _CLASS_LABELS)
    by_label = {r.class_label: r for r in reports}
    assert by_label["up"].recall == 0.0, "a majority-class parrot must show zero recall on the minority class"
    assert by_label["down"].recall == 1.0  # predicts "down" for literally every row

    # Skill score: use the constant predictor's OWN implied probabilities
    # (100% down) as the "model" logloss input for this test -- a discrete
    # argmax parrot maps directly onto a one-hot "model" that is at least
    # as bad as (or worse than) the soft base-rate baseline in expectation.
    one_hot_pred = np.zeros((len(y_test), 3))
    one_hot_pred[:, _DOWN] = 1.0
    from sklearn.metrics import log_loss

    model_logloss = log_loss(y_test, one_hot_pred, labels=[0, 1, 2])
    baseline = baseline_logloss(y_train, y_test, n_classes=3)
    skill = logloss_skill_score(model_logloss, baseline)
    assert skill < 0.05, (
        f"a majority-class parrot's logloss skill score should be near 0 (or negative), got {skill:.3f}"
    )


def test_genuinely_discriminative_model_has_positive_skill_score_and_real_up_recall():
    """Sibling case: a "classifier" that actually uses per-row information
    (here, a synthetic latent signal correlated with the true label) must
    show a clearly positive skill score and non-trivial recall on the
    minority class -- proving these diagnostics don't just always read
    negative/zero regardless of input."""
    rng = np.random.default_rng(3)
    n_train, n_test = 5000, 2000

    def _make_correlated(n, seed):
        rng_local = np.random.default_rng(seed)
        y = rng_local.choice([_DOWN, _NEUTRAL, _UP], size=n, p=[0.65, 0.01, 0.34])
        # A "prediction" that is correct most of the time but not always --
        # simulates genuine (imperfect) discriminative skill.
        noise = rng_local.random(n) < 0.25
        pred = np.where(noise, rng_local.choice([_DOWN, _NEUTRAL, _UP], size=n), y)
        return y, pred

    y_train, _ = _make_correlated(n_train, seed=10)
    y_test, predicted_class = _make_correlated(n_test, seed=11)

    reports = multiclass_discrimination_report(y_test, predicted_class, _CLASS_LABELS)
    by_label = {r.class_label: r for r in reports}
    assert by_label["up"].recall > 0.5, "a genuinely discriminative model should recall most 'up' instances"

    # Soft "probabilities" concentrated on the predicted class, for the skill-score check.
    soft_pred = np.full((n_test, 3), 0.05)
    soft_pred[np.arange(n_test), predicted_class] = 0.9
    soft_pred = soft_pred / soft_pred.sum(axis=1, keepdims=True)
    from sklearn.metrics import log_loss

    model_logloss = log_loss(y_test, soft_pred, labels=[0, 1, 2])
    baseline = baseline_logloss(y_train, y_test, n_classes=3)
    skill = logloss_skill_score(model_logloss, baseline)
    assert skill > 0.1, f"a genuinely discriminative model should show a clearly positive skill score, got {skill:.3f}"


def test_baseline_logloss_uses_train_not_test_base_rates():
    """The baseline must be constructed from TRAIN only -- using test's own
    base rates would let the 'trivial' reference model peek at test labels,
    which the real model structurally cannot do either (apples-to-apples
    comparison)."""
    y_train_balanced = np.array([_DOWN, _NEUTRAL, _UP] * 100)
    y_test_skewed = np.array([_DOWN] * 190 + [_UP] * 10)

    baseline_from_train = baseline_logloss(y_train_balanced, y_test_skewed, n_classes=3)
    # A baseline built from test's own (skewed) rates would score much
    # better on that same skewed test set than one built from train's
    # (balanced) rates -- confirming train, not test, actually drives this.
    baseline_from_test_rates = baseline_logloss(y_test_skewed, y_test_skewed, n_classes=3)
    assert baseline_from_train > baseline_from_test_rates
