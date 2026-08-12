"""Predictive core: k-NN analog engine, gradient boosting models, and linear ensemble."""

from src.predictive_core.labeling import apply_triple_barrier, pool_assets
from src.predictive_core.splitter import WalkForwardSplitter
from src.predictive_core.analog_engine import HistoricalAnalogEngine, AnalogForecast
from src.predictive_core.calibration import (
    expected_calibration_error,
    multiclass_calibration_report,
    CalibrationReport,
)
from src.predictive_core.boosted_core import (
    WalkForwardTrainer,
    FoldResult,
    DirectionFoldResult,
    QuantileFoldResult,
    train_direction_classifier,
    train_quantile_regressor,
    DIRECTION_CLASS_LABELS,
    QUANTILES,
)

__all__ = [
    "apply_triple_barrier",
    "pool_assets",
    "WalkForwardSplitter",
    "HistoricalAnalogEngine",
    "AnalogForecast",
    "expected_calibration_error",
    "multiclass_calibration_report",
    "CalibrationReport",
    "WalkForwardTrainer",
    "FoldResult",
    "DirectionFoldResult",
    "QuantileFoldResult",
    "train_direction_classifier",
    "train_quantile_regressor",
    "DIRECTION_CLASS_LABELS",
    "QUANTILES",
]