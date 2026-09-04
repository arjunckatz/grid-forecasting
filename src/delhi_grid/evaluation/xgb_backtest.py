"""Monthly development evaluation for the load-only XGBoost benchmark."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from delhi_grid.datasets import (
    MODEL_METADATA_COLUMNS,
    build_forecast_frame,
    build_model_frame,
    model_feature_columns,
)
from delhi_grid.evaluation.backtest import run_development_baselines
from delhi_grid.evaluation.metrics import evaluate_prediction_support
from delhi_grid.evaluation.splits import (
    BacktestConfig,
    make_development_folds,
    select_fold_frames,
)
from delhi_grid.features import LoadFeatureConfig
from delhi_grid.models import XGBoostConfig, train_predict_xgboost

DEFAULT_XGBOOST_PREDICTIONS_PATH = Path(
    "data/processed/xgboost_load_only_predictions.parquet"
)
DEFAULT_XGBOOST_METRICS_PATH = Path(
    "data/processed/xgboost_load_only_metrics.csv"
)


@dataclass(frozen=True)
class XGBoostDevelopmentResult:
    """Predictions, comparison metrics, and compact model-input diagnostics."""

    predictions: pd.DataFrame
    comparison_metrics: pd.DataFrame
    fold_training: pd.DataFrame
    feature_missingness: pd.DataFrame


def _missingness_rows(
    frame: pd.DataFrame,
    *,
    fold_id: str,
    split: str,
    feature_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    return [
        {
            "fold_id": fold_id,
            "split": split,
            "feature_name": feature,
            "rows": len(frame),
            "missing_count": int(frame[feature].isna().sum()),
            "missing_fraction": float(frame[feature].isna().mean()),
        }
        for feature in feature_columns
    ]


def run_development_xgboost(
    hourly_demand: pd.DataFrame,
    backtest_config: BacktestConfig,
    feature_config: LoadFeatureConfig,
    model_config: XGBoostConfig,
) -> XGBoostDevelopmentResult:
    """Fit one frozen model per 2024 development fold and compare baselines."""

    folds = make_development_folds(backtest_config)
    forecast_frame = build_forecast_frame(
        hourly_demand, horizon_hours=backtest_config.horizon_hours
    )
    forecast_frame = forecast_frame.loc[
        forecast_frame["valid_time"].le(folds[-1].test_valid_end)
    ].reset_index(drop=True)
    model_frame = build_model_frame(forecast_frame, hourly_demand, feature_config)
    feature_columns = model_feature_columns(feature_config)

    fold_predictions: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    for fold in folds:
        candidate_train, test = select_fold_frames(model_frame, fold)
        train = candidate_train.loc[candidate_train["target_load_mw"].notna()].copy()
        prediction = train_predict_xgboost(
            train,
            test,
            feature_columns=feature_columns,
            config=model_config,
        )

        result = test[list(MODEL_METADATA_COLUMNS)].copy()
        result.insert(0, "model_name", model_config.model_name)
        result.insert(0, "fold_id", fold.fold_id)
        result["prediction_mw"] = prediction
        result["max_source_time"] = test[
            "latest_load_feature_source_time"
        ].to_numpy()
        result["max_source_available_time"] = test[
            "latest_load_feature_available_time"
        ].to_numpy()
        fold_predictions.append(result)

        training_rows.append(
            {
                "fold_id": fold.fold_id,
                "model_fit_time": fold.model_fit_time,
                "train_valid_time_cutoff": fold.train_valid_time_cutoff,
                "candidate_training_rows": len(candidate_train),
                "target_rows_excluded": int(
                    candidate_train["target_load_mw"].isna().sum()
                ),
                "training_rows": len(train),
                "test_rows": len(test),
            }
        )
        missingness_rows.extend(
            _missingness_rows(
                train,
                fold_id=fold.fold_id,
                split="train",
                feature_columns=feature_columns,
            )
        )
        missingness_rows.extend(
            _missingness_rows(
                test,
                fold_id=fold.fold_id,
                split="test",
                feature_columns=feature_columns,
            )
        )

    predictions = pd.concat(fold_predictions, ignore_index=True)
    available = predictions["prediction_mw"].notna()
    if predictions.loc[available, "max_source_available_time"].gt(
        predictions.loc[available, "issue_time"]
    ).any():
        raise AssertionError("XGBoost feature availability exceeds issue time")
    holdout_start = pd.Timestamp(backtest_config.holdout_valid_start).tz_localize(
        backtest_config.timezone
    )
    if predictions["valid_time"].ge(holdout_start).any():
        raise AssertionError("XGBoost development predictions include holdout rows")

    baseline_predictions, _ = run_development_baselines(
        hourly_demand, backtest_config
    )
    comparison = pd.concat([baseline_predictions, predictions], ignore_index=True)
    return XGBoostDevelopmentResult(
        predictions=predictions,
        comparison_metrics=evaluate_prediction_support(comparison),
        fold_training=pd.DataFrame(training_rows),
        feature_missingness=pd.DataFrame(missingness_rows),
    )


def write_xgboost_results(
    result: XGBoostDevelopmentResult,
    *,
    predictions_path: str | Path = DEFAULT_XGBOOST_PREDICTIONS_PATH,
    metrics_path: str | Path = DEFAULT_XGBOOST_METRICS_PATH,
) -> tuple[Path, Path]:
    """Write ignored learned-model predictions and comparative metrics."""

    prediction_output = Path(predictions_path)
    metrics_output = Path(metrics_path)
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(
        prediction_output,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    result.comparison_metrics.to_csv(metrics_output, index=False, lineterminator="\n")
    return prediction_output, metrics_output
