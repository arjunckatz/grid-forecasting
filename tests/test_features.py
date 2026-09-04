from math import sqrt
from pathlib import Path

import pandas as pd
import pytest

from delhi_grid.datasets import (
    build_forecast_frame,
    build_model_frame,
    model_feature_columns,
)
from delhi_grid.features import (
    CALENDAR_FEATURE_COLUMNS,
    build_calendar_features,
    build_load_features,
    load_feature_columns,
    rolling_minimum_periods,
)
from delhi_grid.models import load_xgboost_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _toy_hourly() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-01-01", periods=250, freq="h", tz="Asia/Kolkata"
    )
    loads = pd.Series(range(1, 251), dtype="Float64")
    loads.iloc[166] = 9999.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": loads,
            "quality_flag": pd.Categorical(["complete"] * len(timestamps)),
        }
    )


def test_calendar_features_have_exact_known_values() -> None:
    valid_time = pd.Series(
        [pd.Timestamp("2024-01-01 06:00:00", tz="Asia/Kolkata")]
    )

    features = build_calendar_features(valid_time).iloc[0]

    assert features["valid_hour"] == 6
    assert features["valid_day_of_week"] == 0
    assert features["valid_is_weekend"] == 0
    assert features["valid_month"] == 1
    assert features["valid_day_of_year"] == 1
    assert features["valid_hour_sin"] == pytest.approx(1.0)
    assert features["valid_hour_cos"] == pytest.approx(0.0, abs=1e-12)
    assert features["valid_year_sin"] == pytest.approx(0.0)
    assert features["valid_year_cos"] == pytest.approx(1.0)


def test_load_features_use_completed_buckets_and_explicit_contract() -> None:
    hourly = _toy_hourly()
    forecast = build_forecast_frame(hourly, horizon_hours=24)
    row = forecast.iloc[[190]].reset_index(drop=True)
    feature_config, _ = load_xgboost_config(CONFIG_PATH)

    load_features = build_load_features(row, hourly, feature_config).iloc[0]
    model_frame = build_model_frame(row, hourly, feature_config)
    feature_columns = model_feature_columns(feature_config)

    assert load_features["load_issue_lag_1h"] == 166.0
    assert load_features["load_issue_lag_2h"] == 165.0
    assert load_features["load_valid_lag_168h"] == 23.0
    assert load_features["load_roll_mean_6h"] == pytest.approx(163.5)
    assert load_features["load_roll_std_6h"] == pytest.approx(sqrt(3.5))
    assert load_features["load_issue_lag_1h"] != 9999.0
    assert load_features["latest_load_feature_source_time"] == (
        row.loc[0, "issue_time"] - pd.Timedelta(hours=1)
    )
    assert load_features["latest_load_feature_available_time"] == row.loc[
        0, "issue_time"
    ]
    assert feature_columns == CALENDAR_FEATURE_COLUMNS + load_feature_columns(
        feature_config
    )
    assert set(feature_columns).issubset(model_frame.columns)
    assert not {
        "issue_time",
        "valid_time",
        "target_load_mw",
        "target_quality_flag",
        "latest_load_feature_source_time",
        "latest_load_feature_available_time",
    }.intersection(feature_columns)


def test_rolling_features_require_75_percent_without_imputation() -> None:
    hourly = _toy_hourly()
    hourly.loc[[164, 165], "load_mw"] = pd.NA
    forecast = build_forecast_frame(hourly, horizon_hours=24)
    row = forecast.iloc[[190]].reset_index(drop=True)
    feature_config, _ = load_xgboost_config(CONFIG_PATH)

    features = build_load_features(row, hourly, feature_config).iloc[0]

    assert rolling_minimum_periods(6, 0.75) == 5
    assert rolling_minimum_periods(24, 0.75) == 18
    assert rolling_minimum_periods(168, 0.75) == 126
    assert pd.isna(features["load_issue_lag_1h"])
    assert pd.isna(features["load_roll_mean_6h"])
    assert pd.isna(features["load_roll_std_6h"])
