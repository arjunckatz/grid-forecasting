"""Leakage-safe seasonal baselines for fixed-horizon demand forecasts."""

from collections.abc import Iterable

import pandas as pd

from delhi_grid.datasets import (
    HOURLY_AVAILABILITY_DELAY,
    hourly_load_available_time,
)

PREVIOUS_DAY_LAST_COMPLETED_HOUR = "previous_day_last_completed_hour"
PREVIOUS_WEEK = "previous_week"
TRAILING_4WEEK_MEDIAN = "trailing_4week_same_hour_median"
BASELINE_NAMES = [
    PREVIOUS_DAY_LAST_COMPLETED_HOUR,
    PREVIOUS_WEEK,
    TRAILING_4WEEK_MEDIAN,
]
WEEKLY_LAGS_HOURS = [168, 336, 504, 672]
REQUIRED_FRAME_COLUMNS = {
    "issue_time",
    "valid_time",
    "horizon_hours",
    "target_load_mw",
    "target_quality_flag",
}
REQUIRED_HOURLY_COLUMNS = {"timestamp", "load_mw"}


def _validate_inputs(forecast_frame: pd.DataFrame, hourly_demand: pd.DataFrame) -> None:
    missing_frame = sorted(REQUIRED_FRAME_COLUMNS.difference(forecast_frame.columns))
    if missing_frame:
        raise ValueError(f"forecast frame missing columns: {missing_frame}")
    missing_hourly = sorted(REQUIRED_HOURLY_COLUMNS.difference(hourly_demand.columns))
    if missing_hourly:
        raise ValueError(f"hourly demand missing columns: {missing_hourly}")
    if hourly_demand["timestamp"].duplicated().any():
        raise ValueError("hourly timestamps must be unique")


def _prediction_table(
    forecast_frame: pd.DataFrame,
    *,
    model_name: str,
    prediction: pd.Series,
    max_source_time: pd.Series,
) -> pd.DataFrame:
    result = forecast_frame[
        ["issue_time", "valid_time", "target_load_mw", "target_quality_flag"]
    ].copy()
    result.insert(0, "model_name", model_name)
    result["prediction_mw"] = prediction.astype("Float64")
    result["max_source_time"] = max_source_time
    result["max_source_available_time"] = hourly_load_available_time(
        max_source_time
    )
    return result


def _lookup_loads(
    load_by_time: pd.Series, source_times: Iterable[pd.Timestamp]
) -> pd.Series:
    return load_by_time.reindex(pd.DatetimeIndex(source_times)).reset_index(drop=True)


def predict_baselines(
    forecast_frame: pd.DataFrame,
    hourly_demand: pd.DataFrame,
) -> pd.DataFrame:
    """Produce all V1 baselines with latest-used-source provenance."""

    _validate_inputs(forecast_frame, hourly_demand)
    frame = forecast_frame.reset_index(drop=True)
    load_by_time = hourly_demand.set_index("timestamp")["load_mw"]

    previous_day_times = frame["issue_time"] - HOURLY_AVAILABILITY_DELAY
    previous_day_prediction = _lookup_loads(load_by_time, previous_day_times)
    previous_day_source = previous_day_times.where(previous_day_prediction.notna())

    previous_week_times = frame["valid_time"] - pd.Timedelta(hours=168)
    previous_week_prediction = _lookup_loads(load_by_time, previous_week_times)
    previous_week_source = previous_week_times.where(previous_week_prediction.notna())


    #debu issue mismatch?
    weekly_values: dict[int, pd.Series] = {}
    weekly_times: dict[int, pd.Series] = {}
    for lag in WEEKLY_LAGS_HOURS:
        source_time = frame["valid_time"] - pd.Timedelta(hours=lag)
        weekly_times[lag] = source_time
        weekly_values[lag] = _lookup_loads(load_by_time, source_time)


    candidates = pd.DataFrame(weekly_values)
    trailing_prediction = candidates.median(axis=1, skipna=True).astype("Float64")

    trailing_source = pd.Series(
        pd.NaT, index=frame.index, dtype=frame["valid_time"].dtype
    )
    for lag in WEEKLY_LAGS_HOURS:
        available_without_later_source = (
            trailing_source.isna() & candidates[lag].notna()
        )
        trailing_source.loc[available_without_later_source] = weekly_times[lag].loc[
            available_without_later_source
        ]

    predictions = pd.concat(
        [
            _prediction_table(
                frame,
                model_name=PREVIOUS_DAY_LAST_COMPLETED_HOUR,
                prediction=previous_day_prediction,
                max_source_time=previous_day_source,
            ),
            _prediction_table(
                frame,
                model_name=PREVIOUS_WEEK,
                prediction=previous_week_prediction,
                max_source_time=previous_week_source,
            ),
            _prediction_table(
                frame,
                model_name=TRAILING_4WEEK_MEDIAN,
                prediction=trailing_prediction,
                max_source_time=trailing_source,
            ),
        ],
        ignore_index=True,
    )
    available = predictions["prediction_mw"].notna()
    violations = predictions.loc[available, "max_source_available_time"].gt(
        predictions.loc[available, "issue_time"]
    )
    if violations.any():
        raise AssertionError("baseline source availability exceeds forecast issue time")
    return predictions
