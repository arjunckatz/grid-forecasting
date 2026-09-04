"""Causally available lagged and rolling demand features."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

import pandas as pd

from delhi_grid.datasets.availability import (
    HOURLY_AVAILABILITY_DELAY,
    hourly_load_available_time,
)

REQUIRED_FRAME_COLUMNS = {"issue_time", "valid_time"}
REQUIRED_HOURLY_COLUMNS = {"timestamp", "load_mw"}


@dataclass(frozen=True)
class LoadFeatureConfig:
    """Established historical load inputs and rolling completeness policy."""

    issue_lags_hours: tuple[int, ...]
    valid_lags_hours: tuple[int, ...]
    rolling_windows_hours: tuple[int, ...]
    rolling_minimum_fraction: float

    def __post_init__(self) -> None:
        groups = (
            self.issue_lags_hours,
            self.valid_lags_hours,
            self.rolling_windows_hours,
        )
        if any(not group or any(value <= 0 for value in group) for group in groups):
            raise ValueError("feature lags and rolling windows must be positive")
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("feature lags and rolling windows must be unique")
        if not 0 < self.rolling_minimum_fraction <= 1:
            raise ValueError("rolling minimum fraction must be in (0, 1]")


def rolling_minimum_periods(window_hours: int, minimum_fraction: float) -> int:
    """Convert a proportional completeness rule to an observed-value count."""

    return ceil(window_hours * minimum_fraction)


def load_feature_columns(config: LoadFeatureConfig) -> tuple[str, ...]:
    """Return every historical load feature in stable model-input order."""

    issue_lags = tuple(f"load_issue_lag_{lag}h" for lag in config.issue_lags_hours)
    valid_lags = tuple(f"load_valid_lag_{lag}h" for lag in config.valid_lags_hours)
    rolling = tuple(
        name
        for window in config.rolling_windows_hours
        for name in (f"load_roll_mean_{window}h", f"load_roll_std_{window}h")
    )
    return issue_lags + valid_lags + rolling


def _lookup_loads(
    load_by_time: pd.Series, source_times: Iterable[pd.Timestamp]
) -> pd.Series:
    values = load_by_time.reindex(pd.DatetimeIndex(source_times)).reset_index(drop=True)
    return values.astype("Float64")


def _validate_inputs(frame: pd.DataFrame, hourly_demand: pd.DataFrame) -> None:
    missing_frame = sorted(REQUIRED_FRAME_COLUMNS.difference(frame.columns))
    missing_hourly = sorted(REQUIRED_HOURLY_COLUMNS.difference(hourly_demand.columns))
    if missing_frame:
        raise ValueError(f"forecast frame missing columns: {missing_frame}")
    if missing_hourly:
        raise ValueError(f"hourly demand missing columns: {missing_hourly}")
    if hourly_demand["timestamp"].duplicated().any():
        raise ValueError("hourly timestamps must be unique")


def build_load_features(
    forecast_frame: pd.DataFrame,
    hourly_demand: pd.DataFrame,
    config: LoadFeatureConfig,
) -> pd.DataFrame:
    """Build load features whose source buckets are complete by issue time."""

    _validate_inputs(forecast_frame, hourly_demand)
    frame = forecast_frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("cannot build load features for an empty forecast frame")
    load_by_time = hourly_demand.set_index("timestamp")["load_mw"].sort_index()
    latest_legal_bucket = frame["issue_time"].max() - HOURLY_AVAILABILITY_DELAY
    load_by_time = load_by_time.loc[load_by_time.index <= latest_legal_bucket]
    features: dict[str, pd.Series] = {}
    source_time_candidates: list[pd.Series] = []

    for lag in config.issue_lags_hours:
        source_time = frame["issue_time"] - pd.Timedelta(hours=lag)
        features[f"load_issue_lag_{lag}h"] = _lookup_loads(
            load_by_time, source_time
        )
        source_time_candidates.append(source_time)

    for lag in config.valid_lags_hours:
        source_time = frame["valid_time"] - pd.Timedelta(hours=lag)
        features[f"load_valid_lag_{lag}h"] = _lookup_loads(
            load_by_time, source_time
        )
        source_time_candidates.append(source_time)

    rolling_end = frame["issue_time"] - HOURLY_AVAILABILITY_DELAY
    for window in config.rolling_windows_hours:
        minimum = rolling_minimum_periods(window, config.rolling_minimum_fraction)
        rolling = load_by_time.rolling(window=window, min_periods=minimum)
        features[f"load_roll_mean_{window}h"] = _lookup_loads(
            rolling.mean(), rolling_end
        )
        features[f"load_roll_std_{window}h"] = _lookup_loads(
            rolling.std(), rolling_end
        )
        source_time_candidates.append(rolling_end)

    latest_source_time = pd.concat(source_time_candidates, axis=1).max(axis=1)
    latest_available_time = hourly_load_available_time(latest_source_time)
    if latest_available_time.gt(frame["issue_time"]).any():
        raise AssertionError("load feature availability exceeds forecast issue time")

    result = pd.DataFrame(features)
    result["latest_load_feature_source_time"] = latest_source_time
    result["latest_load_feature_available_time"] = latest_available_time
    return result
