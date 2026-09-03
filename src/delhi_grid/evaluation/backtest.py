"""Orchestrate reproducible development-period baseline evaluation."""

from pathlib import Path

import pandas as pd

from delhi_grid.datasets import build_forecast_frame
from delhi_grid.evaluation.metrics import evaluate_prediction_support
from delhi_grid.evaluation.splits import (
    BacktestConfig,
    make_development_folds,
    select_fold_frames,
)
from delhi_grid.models import predict_baselines

DEFAULT_PREDICTIONS_PATH = Path("data/processed/baseline_predictions.parquet")
DEFAULT_METRICS_PATH = Path("data/processed/baseline_metrics.csv")


def run_development_baselines(
    hourly_demand: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baselines on configured development months only."""

    forecast_frame = build_forecast_frame(
        hourly_demand, horizon_hours=config.horizon_hours
    )
    fold_predictions: list[pd.DataFrame] = []
    for fold in make_development_folds(config):
        _, test_frame = select_fold_frames(forecast_frame, fold)
        predictions = predict_baselines(test_frame, hourly_demand)
        predictions.insert(0, "fold_id", fold.fold_id)
        fold_predictions.append(predictions)

    result = pd.concat(fold_predictions, ignore_index=True)
    holdout_start = pd.Timestamp(config.holdout_valid_start).tz_localize(
        config.timezone
    )
    if result["valid_time"].ge(holdout_start).any():
        raise AssertionError("development evaluation includes holdout valid times")


    #final v4 #agent goofup
    metrics = evaluate_prediction_support(result)
    return result, metrics


def write_baseline_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> tuple[Path, Path]:
    """Write standardized predictions and support-aware metrics."""

    prediction_output = Path(predictions_path)
    metrics_output = Path(metrics_path)
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        prediction_output,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    metrics.to_csv(metrics_output, index=False, lineterminator="\n")
    return prediction_output, metrics_output
