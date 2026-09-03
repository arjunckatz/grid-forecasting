"""Build the quality-aware canonical hourly demand dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

DEFAULT_HOURLY_PATH = Path("data/processed/hourly_demand.parquet")
QUALITY_LEVELS = ["complete", "usable_partial", "insufficient", "missing"]
REQUIRED_INPUT_COLUMNS = {"timestamp", "load_mw"}
EXPECTED_AGGREGATION_METHOD = "mean"
EXPECTED_MODELING_FREQUENCY = "1h"
EXPECTED_RAW_DURATION = pd.Timedelta("5min")
EXPECTED_TIMEZONE = "Asia/Kolkata"


class HourlyConfigError(ValueError):
    """Raised when the V1 hourly configuration is absent or inconsistent."""


@dataclass(frozen=True)
class HourlyDemandConfig:
    """Configuration fields that directly govern canonical hourly demand."""

    timezone: str
    raw_frequency: str
    modeling_frequency: str
    aggregation_method: str
    expected_readings_per_hour: int
    minimum_valid_readings_per_hour: int
    minimum_coverage_fraction: float

    def __post_init__(self) -> None:
        if self.timezone != EXPECTED_TIMEZONE:
            raise HourlyConfigError("timezone must be Asia/Kolkata")
        if self.modeling_frequency != EXPECTED_MODELING_FREQUENCY:
            raise HourlyConfigError("modeling_frequency must be 1h for this dataset")
        if self.aggregation_method != EXPECTED_AGGREGATION_METHOD:
            raise HourlyConfigError("aggregation method must be mean")
        if self.expected_readings_per_hour <= 0:
            raise HourlyConfigError("expected readings per hour must be positive")
        if not 1 <= self.minimum_valid_readings_per_hour <= (
            self.expected_readings_per_hour
        ):
            raise HourlyConfigError(
                "minimum valid readings must be between 1 and expected readings"
            )

        try:
            raw_duration = pd.Timedelta(self.raw_frequency)
            modeling_duration = pd.Timedelta(self.modeling_frequency)
        except (TypeError, ValueError) as error:
            raise HourlyConfigError("raw/modeling frequency is invalid") from error
        if raw_duration != EXPECTED_RAW_DURATION:
            raise HourlyConfigError("raw_frequency must represent five minutes")
        derived_expected = int(modeling_duration / raw_duration)
        if derived_expected != self.expected_readings_per_hour:
            raise HourlyConfigError(
                "expected readings must match modeling/raw frequency ratio"
            )

        derived_coverage = (
            self.minimum_valid_readings_per_hour / self.expected_readings_per_hour
        )
        if abs(derived_coverage - self.minimum_coverage_fraction) > 1e-12:
            raise HourlyConfigError(
                "minimum coverage must equal minimum/expected readings"
            )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HourlyConfigError(f"{name} must be a mapping")
    return value


def load_hourly_demand_config(path: str | Path) -> HourlyDemandConfig:
    """Load only the V1 fields used to construct canonical hourly demand."""

    with Path(path).open(encoding="utf-8") as config_file:
        document = _require_mapping(yaml.safe_load(config_file), "config")

    time = _require_mapping(document.get("time"), "time")
    aggregation = _require_mapping(document.get("aggregation"), "aggregation")
    try:
        return HourlyDemandConfig(
            timezone=str(time["timezone"]),
            raw_frequency=str(time["raw_frequency"]),
            modeling_frequency=str(time["modeling_frequency"]),
            aggregation_method=str(aggregation["method"]),
            expected_readings_per_hour=int(
                aggregation["expected_readings_per_hour"]
            ),
            minimum_valid_readings_per_hour=int(
                aggregation["minimum_valid_readings_per_hour"]
            ),
            minimum_coverage_fraction=float(
                aggregation["minimum_coverage_fraction"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HourlyConfigError(f"invalid hourly config field: {error}") from error


def _validate_observations(
    observations: pd.DataFrame, config: HourlyDemandConfig
) -> None:
    missing_columns = sorted(REQUIRED_INPUT_COLUMNS.difference(observations.columns))
    if missing_columns:
        raise ValueError(f"normalized observations missing columns: {missing_columns}")
    if observations.empty:
        raise ValueError("cannot build hourly demand from an empty table")

    timestamps = observations["timestamp"]
    if timestamps.isna().any():
        raise ValueError("all timestamps must parse before hourly aggregation")
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamps must be timezone-aware")
    if str(timestamps.dt.tz) != config.timezone:
        raise ValueError(f"timestamps must use {config.timezone}")
    if timestamps.duplicated().any():
        raise ValueError("duplicate timestamps require an explicit resolution policy")

    on_grid = (
        (timestamps.dt.minute % 5 == 0)
        & timestamps.dt.second.eq(0)
        & timestamps.dt.microsecond.eq(0)
    )
    if not on_grid.all():
        raise ValueError("timestamps must align to five-minute boundaries")
    if not is_numeric_dtype(observations["load_mw"].dtype):
        raise ValueError("load_mw must be numeric")


def build_hourly_demand(
    observations: pd.DataFrame,
    config: HourlyDemandConfig,
) -> pd.DataFrame:
    """Aggregate normalized telemetry onto a complete hourly grid.

    Only valid numeric measurements count toward coverage or the arithmetic
    mean. Hours below the configured threshold remain present with a missing
    target.
    """

    _validate_observations(observations, config)
    working = observations.loc[:, ["timestamp", "load_mw"]].copy()
    working["hour"] = working["timestamp"].dt.floor("h")

    grouped = working.groupby("hour", sort=True, observed=True).agg(
        n_timestamp_rows=("timestamp", "size"),
        n_observations=("load_mw", "count"),
        observed_mean_mw=("load_mw", "mean"),
    )
    hourly_grid = pd.date_range(
        working["hour"].min(),
        working["hour"].max(),
        freq=config.modeling_frequency,
        name="timestamp",
    )
    hourly = grouped.reindex(hourly_grid)

    count_columns = ["n_timestamp_rows", "n_observations"]
    hourly[count_columns] = hourly[count_columns].fillna(0).astype("int16")
    hourly["n_expected"] = pd.Series(
        config.expected_readings_per_hour,
        index=hourly.index,
        dtype="int16",
    )
    hourly["n_blank_loads"] = (
        hourly["n_timestamp_rows"] - hourly["n_observations"]
    ).astype("int16")
    hourly["coverage_fraction"] = (
        hourly["n_observations"] / hourly["n_expected"]
    ).astype("Float64")

    quality = pd.Series("missing", index=hourly.index, dtype="string")
    insufficient = hourly["n_observations"].between(
        1, config.minimum_valid_readings_per_hour - 1
    )
    quality.loc[insufficient] = "insufficient"
    quality.loc[
        hourly["n_observations"].between(
            config.minimum_valid_readings_per_hour,
            config.expected_readings_per_hour - 1,
        )
    ] = "usable_partial"
    quality.loc[
        hourly["n_observations"].eq(config.expected_readings_per_hour)
    ] = "complete"
    hourly["quality_flag"] = pd.Categorical(
        quality,
        categories=QUALITY_LEVELS,
        ordered=True,
    )

    usable = hourly["n_observations"].ge(
        config.minimum_valid_readings_per_hour
    )
    hourly["load_mw"] = hourly["observed_mean_mw"].where(usable).astype("Float64")

    return hourly.reset_index()[
        [
            "timestamp",
            "load_mw",
            "n_expected",
            "n_timestamp_rows",
            "n_observations",
            "n_blank_loads",
            "coverage_fraction",
            "quality_flag",
        ]
    ]


def write_hourly_demand(
    hourly_demand: pd.DataFrame,
    path: str | Path = DEFAULT_HOURLY_PATH,
) -> Path:
    """Write canonical hourly demand as a deterministic, index-free Parquet."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_demand.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    return output_path
