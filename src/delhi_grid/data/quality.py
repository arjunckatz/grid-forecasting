"""Deterministic diagnostics for normalized raw demand observations."""

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

EXPECTED_INTERVAL_SECONDS = 5 * 60
EXPECTED_READINGS_PER_DAY = 24 * 60 // 5
REQUIRED_COLUMNS = {
    "timestamp_raw",
    "load_mw_raw",
    "timestamp",
    "load_mw",
}


@dataclass(frozen=True)
class DemandAudit:
    """Serializable summary of raw-ingestion quality; no repairs are applied."""

    total_row_count: int
    parsed_timestamp_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    null_timestamp_count: int
    malformed_timestamp_count: int
    null_load_count: int
    non_numeric_load_count: int
    unique_timestamp_count: int
    duplicate_timestamp_count: int
    duplicate_timestamp_group_count: int
    conflicting_duplicate_timestamp_count: int
    timestamps_monotonic_in_source_order: bool
    backwards_timestamp_movement_count: int
    timestamps_monotonic_after_sorting: bool
    observed_interval_seconds: dict[str, int]
    five_minute_interval_count: int
    sub_5_minute_interval_count: int
    greater_than_5_minute_interval_count: int
    non_5_minute_interval_count: int
    missing_expected_slot_count: int
    off_5_minute_grid_timestamp_count: int
    complete_timestamp_day_count: int
    partial_timestamp_day_count: int
    missing_timestamp_day_count: int
    complete_numeric_load_day_count: int
    partial_numeric_load_day_count: int
    empty_numeric_load_day_count: int
    numeric_load_count: int
    minimum_load_mw: float | None
    maximum_load_mw: float | None
    mean_load_mw: float | None
    median_load_mw: float | None
    zero_load_count: int
    negative_load_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable audit fields."""

        return asdict(self)


def _optional_float(value: float, *, has_values: bool) -> float | None:
    return float(value) if has_values else None


def audit_demand(observations: pd.DataFrame) -> DemandAudit:
    """Audit normalized observations while preserving every input row."""

    missing_columns = sorted(REQUIRED_COLUMNS.difference(observations.columns))
    if missing_columns:
        raise ValueError(f"normalized observations missing columns: {missing_columns}")

    timestamp_raw = observations["timestamp_raw"].astype("string")
    load_raw = observations["load_mw_raw"].astype("string")
    timestamps = observations["timestamp"]
    loads = pd.to_numeric(observations["load_mw"], errors="coerce")

    null_timestamps = timestamp_raw.isna() | timestamp_raw.str.strip().eq("")
    null_loads = load_raw.isna() | load_raw.str.strip().eq("")
    malformed_timestamps = ~null_timestamps & timestamps.isna()
    non_numeric_loads = ~null_loads & loads.isna()

    valid_timestamps = timestamps.dropna()
    source_intervals = valid_timestamps.diff().dropna()
    sorted_unique = pd.DatetimeIndex(valid_timestamps.drop_duplicates().sort_values())
    duplicate_rows = valid_timestamps.duplicated(keep="first")
    duplicate_groups = valid_timestamps[valid_timestamps.duplicated(keep=False)]

    valid_timestamp_rows = observations.loc[
        timestamps.notna(), ["timestamp", "load_mw"]
    ]
    conflicting_groups = (
        valid_timestamp_rows.groupby("timestamp", observed=True)["load_mw"]
        .nunique(dropna=False)
        .gt(1)
    )

    intervals = sorted_unique.to_series(index=range(len(sorted_unique))).diff().dropna()
    interval_seconds = intervals.dt.total_seconds().astype("int64")
    interval_distribution = {
        str(int(seconds)): int(count)
        for seconds, count in interval_seconds.value_counts().sort_index().items()
    }

    if len(sorted_unique):
        on_grid = (
            (sorted_unique.minute % 5 == 0)
            & (sorted_unique.second == 0)
            & (sorted_unique.microsecond == 0)
        )
        grid_timestamps = sorted_unique[on_grid]
        expected_grid = pd.date_range(
            sorted_unique.min().floor("5min"),
            sorted_unique.max().floor("5min"),
            freq="5min",
        )
        missing_slots = len(expected_grid.difference(grid_timestamps))

        first_day = sorted_unique.min().normalize()
        last_day = sorted_unique.max().normalize()
        all_days = pd.date_range(first_day, last_day, freq="1D")
        counts_by_day = pd.Series(1, index=grid_timestamps).groupby(
            grid_timestamps.normalize()
        ).sum()
        daily_counts = counts_by_day.reindex(all_days, fill_value=0)
        complete_days = int(daily_counts.eq(EXPECTED_READINGS_PER_DAY).sum())
        partial_days = int(
            daily_counts.between(1, EXPECTED_READINGS_PER_DAY - 1).sum()
        )
        missing_days = int(daily_counts.eq(0).sum())

        numeric_timestamp_rows = observations.loc[
            timestamps.notna() & loads.notna(), ["timestamp"]
        ].drop_duplicates()
        numeric_timestamps = pd.DatetimeIndex(numeric_timestamp_rows["timestamp"])
        numeric_on_grid = (
            (numeric_timestamps.minute % 5 == 0)
            & (numeric_timestamps.second == 0)
            & (numeric_timestamps.microsecond == 0)
        )
        numeric_grid_timestamps = numeric_timestamps[numeric_on_grid]
        numeric_counts_by_day = pd.Series(1, index=numeric_grid_timestamps).groupby(
            numeric_grid_timestamps.normalize()
        ).sum()
        numeric_daily_counts = numeric_counts_by_day.reindex(all_days, fill_value=0)
        complete_numeric_days = int(
            numeric_daily_counts.eq(EXPECTED_READINGS_PER_DAY).sum()
        )
        partial_numeric_days = int(
            numeric_daily_counts.between(1, EXPECTED_READINGS_PER_DAY - 1).sum()
        )
        empty_numeric_days = int(numeric_daily_counts.eq(0).sum())
        first_timestamp = sorted_unique.min().isoformat()
        last_timestamp = sorted_unique.max().isoformat()
        off_grid_count = int((~on_grid).sum())
    else:
        missing_slots = 0
        complete_days = 0
        partial_days = 0
        missing_days = 0
        complete_numeric_days = 0
        partial_numeric_days = 0
        empty_numeric_days = 0
        first_timestamp = None
        last_timestamp = None
        off_grid_count = 0

    numeric_loads = loads.dropna().astype(float)
    has_numeric_loads = not numeric_loads.empty

    return DemandAudit(
        total_row_count=len(observations),
        parsed_timestamp_count=int(timestamps.notna().sum()),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        null_timestamp_count=int(null_timestamps.sum()),
        malformed_timestamp_count=int(malformed_timestamps.sum()),
        null_load_count=int(null_loads.sum()),
        non_numeric_load_count=int(non_numeric_loads.sum()),
        unique_timestamp_count=len(sorted_unique),
        duplicate_timestamp_count=int(duplicate_rows.sum()),
        duplicate_timestamp_group_count=int(duplicate_groups.nunique()),
        conflicting_duplicate_timestamp_count=int(conflicting_groups.sum()),
        timestamps_monotonic_in_source_order=bool(
            valid_timestamps.is_monotonic_increasing
        ),
        backwards_timestamp_movement_count=int(
            source_intervals.dt.total_seconds().lt(0).sum()
        ),
        timestamps_monotonic_after_sorting=bool(sorted_unique.is_monotonic_increasing),
        observed_interval_seconds=interval_distribution,
        five_minute_interval_count=int(
            interval_seconds.eq(EXPECTED_INTERVAL_SECONDS).sum()
        ),
        sub_5_minute_interval_count=int(
            interval_seconds.lt(EXPECTED_INTERVAL_SECONDS).sum()
        ),
        greater_than_5_minute_interval_count=int(
            interval_seconds.gt(EXPECTED_INTERVAL_SECONDS).sum()
        ),
        non_5_minute_interval_count=int(
            interval_seconds.ne(EXPECTED_INTERVAL_SECONDS).sum()
        ),
        missing_expected_slot_count=missing_slots,
        off_5_minute_grid_timestamp_count=off_grid_count,
        complete_timestamp_day_count=complete_days,
        partial_timestamp_day_count=partial_days,
        missing_timestamp_day_count=missing_days,
        complete_numeric_load_day_count=complete_numeric_days,
        partial_numeric_load_day_count=partial_numeric_days,
        empty_numeric_load_day_count=empty_numeric_days,
        numeric_load_count=len(numeric_loads),
        minimum_load_mw=_optional_float(
            numeric_loads.min(), has_values=has_numeric_loads
        ),
        maximum_load_mw=_optional_float(
            numeric_loads.max(), has_values=has_numeric_loads
        ),
        mean_load_mw=_optional_float(
            numeric_loads.mean(), has_values=has_numeric_loads
        ),
        median_load_mw=_optional_float(
            numeric_loads.median(), has_values=has_numeric_loads
        ),
        zero_load_count=int(loads.eq(0).sum()),
        negative_load_count=int(loads.lt(0).sum()),
    )
