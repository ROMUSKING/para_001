"""Method-blind evaluation of world-model predictions (NumPy/pandas only)."""

from .prediction import (
    OUTCOMES,
    RidgeForecaster,
    classify_relative_difference,
    derangement,
    effective_rank,
    horizon_mean_rmse,
    interval_coverage,
    paired_relative_difference,
    per_window_cosine,
    per_window_mse,
    per_window_native_mse,
    persistence_forecast,
    prediction_frame,
    rmse_by_horizon,
    stack_windows,
    summarize_frame,
)

__all__ = [
    "OUTCOMES",
    "RidgeForecaster",
    "classify_relative_difference",
    "derangement",
    "effective_rank",
    "horizon_mean_rmse",
    "interval_coverage",
    "paired_relative_difference",
    "per_window_cosine",
    "per_window_mse",
    "per_window_native_mse",
    "persistence_forecast",
    "prediction_frame",
    "rmse_by_horizon",
    "stack_windows",
    "summarize_frame",
]
