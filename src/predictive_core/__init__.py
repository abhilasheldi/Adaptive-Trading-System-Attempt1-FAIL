"""Predictive core: triple-barrier labelling, walk-forward splitting, and
gradient-boosted quantile regression.

Stage 0 (MASTER_CONTEXT_v2 section 8) removed the k-NN analog engine
(Component A), the Component B direction classifier, and the ensemble
meta-model from this package. What remains is labelling, splitting, dataset
assembly, and measurement apparatus. The learned component for Stage D is a
binary meta-label model on a conditioned event set and does not exist yet.
"""

from src.predictive_core.labeling import apply_triple_barrier, pool_assets
from src.predictive_core.splitter import WalkForwardSplitter
from src.predictive_core.purged_kfold import (
    PurgedKFold,
    PurgedFold,
    purge_and_embargo,
    assert_scored_rows_match_nominal,
)
from src.predictive_core.calibration import (
    expected_calibration_error,
    multiclass_calibration_report,
    CalibrationReport,
)
from src.predictive_core.uniqueness import (
    UniquenessResult,
    average_uniqueness,
    uniqueness_sample_weights,
)
from src.predictive_core.trial_registry import (
    TrialRecord,
    config_hash,
    distinct_configuration_count,
    log_trial,
    read_trials,
    registry_summary,
)
from src.predictive_core.dsr import (
    DSRResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.predictive_core.boosted_core import (
    WalkForwardTrainer,
    FoldResult,
    QuantileFoldResult,
    train_quantile_regressor,
    QUANTILES,
)

__all__ = [
    "apply_triple_barrier",
    "pool_assets",
    "WalkForwardSplitter",
    "PurgedKFold",
    "PurgedFold",
    "purge_and_embargo",
    "assert_scored_rows_match_nominal",
    "UniquenessResult",
    "average_uniqueness",
    "uniqueness_sample_weights",
    "TrialRecord",
    "config_hash",
    "distinct_configuration_count",
    "log_trial",
    "read_trials",
    "registry_summary",
    "DSRResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "expected_calibration_error",
    "multiclass_calibration_report",
    "CalibrationReport",
    "WalkForwardTrainer",
    "FoldResult",
    "QuantileFoldResult",
    "train_quantile_regressor",
    "QUANTILES",
]
