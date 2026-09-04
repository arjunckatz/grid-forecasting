from dataclasses import replace
from pathlib import Path

import pandas as pd

from delhi_grid.datasets import (
    build_forecast_frame,
    build_model_frame,
    model_feature_columns,
)
from delhi_grid.evaluation import (
    load_backtest_config,
    run_development_xgboost,
)
from delhi_grid.models import load_xgboost_config, train_predict_xgboost

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _hourly_history() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2023-04-01 00:00:00",
        "2024-12-31 23:00:00",
        freq="h",
        tz="Asia/Kolkata",
    )
    load = 4000 + timestamps.hour * 20 + timestamps.dayofweek * 5
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": pd.Series(load, dtype="Float64"),
            "quality_flag": pd.Categorical(["complete"] * len(timestamps)),
        }
    )


def test_native_xgboost_repeat_is_deterministic() -> None:
    hourly = _hourly_history().iloc[:1000].copy()
    feature_config, model_config = load_xgboost_config(CONFIG_PATH)
    model_config = replace(model_config, boosting_rounds=5)
    forecast = build_forecast_frame(hourly, horizon_hours=24)
    model_frame = build_model_frame(forecast, hourly, feature_config)
    features = model_feature_columns(feature_config)
    train = model_frame.iloc[:800]
    test = model_frame.iloc[800:850]

    first = train_predict_xgboost(
        train, test, feature_columns=features, config=model_config
    )
    second = train_predict_xgboost(
        train, test, feature_columns=features, config=model_config
    )

    pd.testing.assert_series_equal(first, second)


def test_xgboost_backtest_reuses_folds_and_excludes_holdout() -> None:
    feature_config, model_config = load_xgboost_config(CONFIG_PATH)
    model_config = replace(model_config, boosting_rounds=2)
    result = run_development_xgboost(
        _hourly_history(),
        load_backtest_config(CONFIG_PATH),
        feature_config,
        model_config,
    )
    predictions = result.predictions

    assert len(predictions) == 6600
    assert predictions["fold_id"].nunique() == 9
    assert list(predictions.columns) == [
        "fold_id",
        "model_name",
        "issue_time",
        "valid_time",
        "target_load_mw",
        "target_quality_flag",
        "prediction_mw",
        "max_source_time",
        "max_source_available_time",
    ]
    assert predictions["valid_time"].max() == pd.Timestamp(
        "2024-12-31 23:00:00", tz="Asia/Kolkata"
    )
    assert predictions["prediction_mw"].notna().all()
    assert predictions["max_source_time"].eq(
        predictions["issue_time"] - pd.Timedelta(hours=1)
    ).all()
    assert predictions["max_source_available_time"].le(
        predictions["issue_time"]
    ).all()
    assert result.fold_training.loc[0, "train_valid_time_cutoff"] == pd.Timestamp(
        "2024-03-30 23:00:00", tz="Asia/Kolkata"
    )
