import pandas as pd

from delhi_grid.datasets import build_forecast_frame, hourly_load_available_time
from delhi_grid.models.baselines import (
    PREVIOUS_DAY_LAST_COMPLETED_HOUR,
    PREVIOUS_WEEK,
    TRAILING_4WEEK_MEDIAN,
    predict_baselines,
)


def _baseline_case(*, missing_recent_sources: bool = False):
    timestamps = pd.date_range(
        "2024-01-01 00:00:00", periods=800, freq="h", tz="Asia/Kolkata"
    )
    valid_time = timestamps[-1]
    loads = pd.Series(pd.NA, index=range(len(timestamps)), dtype="Float64")
    values = {
        valid_time: 500.0,
        valid_time - pd.Timedelta(hours=25): 99.0,
        valid_time - pd.Timedelta(hours=168): 10.0,
        valid_time - pd.Timedelta(hours=336): 20.0,
        valid_time - pd.Timedelta(hours=504): 30.0,
        valid_time - pd.Timedelta(hours=672): 40.0,
    }
    if missing_recent_sources:
        values.pop(valid_time - pd.Timedelta(hours=25))
        values.pop(valid_time - pd.Timedelta(hours=168))
    for timestamp, value in values.items():
        loads.loc[timestamps.get_loc(timestamp)] = value

    hourly = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": loads,
            "quality_flag": pd.Categorical(
                ["complete" if pd.notna(value) else "missing" for value in loads]
            ),
        }
    )
    frame = build_forecast_frame(hourly, horizon_hours=24)
    target_frame = frame.loc[frame["valid_time"].eq(valid_time)]
    return valid_time, predict_baselines(target_frame, hourly)


def test_baselines_use_exact_legal_source_times_and_values() -> None:
    valid_time, predictions = _baseline_case()
    by_model = predictions.set_index("model_name")

    daily = by_model.loc[PREVIOUS_DAY_LAST_COMPLETED_HOUR]
    assert daily["prediction_mw"] == 99.0
    assert daily["max_source_time"] == valid_time - pd.Timedelta(hours=25)
    assert daily["max_source_available_time"] == valid_time - pd.Timedelta(hours=24)
    assert daily["max_source_available_time"] == daily["issue_time"]
    assert by_model.loc[PREVIOUS_WEEK, "prediction_mw"] == 10.0
    assert by_model.loc[PREVIOUS_WEEK, "max_source_time"] == (
        valid_time - pd.Timedelta(hours=168)
    )
    assert by_model.loc[PREVIOUS_WEEK, "max_source_available_time"] == (
        valid_time - pd.Timedelta(hours=167)
    )
    assert by_model.loc[TRAILING_4WEEK_MEDIAN, "prediction_mw"] == 25.0
    assert by_model.loc[TRAILING_4WEEK_MEDIAN, "max_source_time"] == (
        valid_time - pd.Timedelta(hours=168)
    )
    assert by_model.loc[TRAILING_4WEEK_MEDIAN, "max_source_available_time"] == (
        valid_time - pd.Timedelta(hours=167)
    )
    candidate_times = pd.Series(
        [valid_time - pd.Timedelta(hours=lag) for lag in (168, 336, 504, 672)]
    )
    expected_available_times = candidate_times + pd.Timedelta(hours=1)
    pd.testing.assert_series_equal(
        hourly_load_available_time(candidate_times), expected_available_times
    )
    available = predictions["prediction_mw"].notna()
    assert predictions.loc[available, "max_source_available_time"].le(
        predictions.loc[available, "issue_time"]
    ).all()


def test_baselines_leave_missing_sources_unavailable_and_use_available_median() -> None:
    valid_time, predictions = _baseline_case(missing_recent_sources=True)
    by_model = predictions.set_index("model_name")

    daily = by_model.loc[PREVIOUS_DAY_LAST_COMPLETED_HOUR]
    assert pd.isna(daily["prediction_mw"])
    assert pd.isna(daily["max_source_time"])
    assert pd.isna(daily["max_source_available_time"])
    assert pd.isna(by_model.loc[PREVIOUS_WEEK, "prediction_mw"])
    assert pd.isna(by_model.loc[PREVIOUS_WEEK, "max_source_time"])
    assert pd.isna(by_model.loc[PREVIOUS_WEEK, "max_source_available_time"])
    assert by_model.loc[TRAILING_4WEEK_MEDIAN, "prediction_mw"] == 30.0
    assert by_model.loc[TRAILING_4WEEK_MEDIAN, "max_source_time"] == (
        valid_time - pd.Timedelta(hours=336)
    )
    assert by_model.loc[TRAILING_4WEEK_MEDIAN, "max_source_available_time"] == (
        valid_time - pd.Timedelta(hours=335)
    )
