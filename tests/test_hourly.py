from dataclasses import replace
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from delhi_grid.data.hourly import (
    QUALITY_LEVELS,
    build_hourly_demand,
    load_hourly_demand_config,
    write_hourly_demand,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _config():
    return load_hourly_demand_config(CONFIG_PATH)


def _quality_fixture() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2024-06-01 08:00:00",
        "2024-06-01 13:55:00",
        freq="5min",
        tz="Asia/Kolkata",
    )
    observations = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": pd.Series(4000.0, index=range(len(timestamps)), dtype="Float64"),
        }
    )
    observations.loc[11, "load_mw"] = 10000.0
    observations.loc[12 + 9 : 12 + 11, "load_mw"] = pd.NA
    observations.loc[24 + 8 : 24 + 11, "load_mw"] = pd.NA
    observations.loc[36:47, "load_mw"] = pd.NA
    observations = observations.loc[
        ~observations["timestamp"].between(
            pd.Timestamp("2024-06-01 12:00:00", tz="Asia/Kolkata"),
            pd.Timestamp("2024-06-01 12:55:00", tz="Asia/Kolkata"),
        )
    ].reset_index(drop=True)
    return observations


def test_config_loads_v1_hourly_invariants() -> None:
    config = _config()

    assert config.timezone == "Asia/Kolkata"
    assert config.expected_readings_per_hour == 12
    assert config.minimum_valid_readings_per_hour == 9
    assert config.minimum_coverage_fraction == 0.75


def test_hourly_quality_classes_and_threshold_boundaries() -> None:
    hourly = build_hourly_demand(_quality_fixture(), _config()).set_index("timestamp")

    complete = hourly.loc["2024-06-01 08:00:00+05:30"]
    usable_partial = hourly.loc["2024-06-01 09:00:00+05:30"]
    insufficient = hourly.loc["2024-06-01 10:00:00+05:30"]
    missing_loads = hourly.loc["2024-06-01 11:00:00+05:30"]
    missing_rows = hourly.loc["2024-06-01 12:00:00+05:30"]

    assert complete["quality_flag"] == "complete"
    assert complete["coverage_fraction"] == 1.0
    assert complete["load_mw"] == 4500.0
    assert usable_partial["quality_flag"] == "usable_partial"
    assert usable_partial["n_timestamp_rows"] == 12
    assert usable_partial["n_observations"] == 9
    assert usable_partial["n_blank_loads"] == 3
    assert usable_partial["coverage_fraction"] == 0.75
    assert usable_partial["load_mw"] == 4000.0
    assert insufficient["quality_flag"] == "insufficient"
    assert insufficient["n_observations"] == 8
    assert pd.isna(insufficient["load_mw"])
    assert missing_loads["quality_flag"] == "missing"
    assert missing_loads["n_timestamp_rows"] == 12
    assert missing_loads["n_blank_loads"] == 12
    assert pd.isna(missing_loads["load_mw"])
    assert missing_rows["quality_flag"] == "missing"
    assert missing_rows["n_timestamp_rows"] == 0
    assert missing_rows["n_observations"] == 0
    assert pd.isna(missing_rows["load_mw"])

#fx nam
def test_configured_threshold_change_affects_target_usability() -> None:
    config = replace(
        _config(),
        minimum_valid_readings_per_hour=10,
        minimum_coverage_fraction=10 / 12,
    )
    hourly = build_hourly_demand(_quality_fixture(), config).set_index("timestamp")

    nine_observations = hourly.loc["2024-06-01 09:00:00+05:30"]
    assert nine_observations["quality_flag"] == "insufficient"
    assert pd.isna(nine_observations["load_mw"])

#uhhhhh
def test_hour_labels_timezone_schema_and_dtypes() -> None:
    hourly = build_hourly_demand(_quality_fixture(), _config())

    assert hourly["timestamp"].iloc[0] == pd.Timestamp(
        "2024-06-01 08:00:00", tz="Asia/Kolkata"
    )
    assert str(hourly["timestamp"].dt.tz) == "Asia/Kolkata"
    assert list(hourly.columns) == [
        "timestamp",
        "load_mw",
        "n_expected",
        "n_timestamp_rows",
        "n_observations",
        "n_blank_loads",
        "coverage_fraction",
        "quality_flag",
    ]
    assert str(hourly["load_mw"].dtype) == "Float64"
    assert str(hourly["n_observations"].dtype) == "int16"
    assert str(hourly["coverage_fraction"].dtype) == "Float64"

    assert list(hourly["quality_flag"].cat.categories) == QUALITY_LEVELS


def test_missing_timestamp_row_reduces_row_and_observation_counts() -> None:
    timestamps = pd.date_range(
        "2024-01-01 08:00:00",
        "2024-01-01 09:00:00",
        freq="5min",
        tz="Asia/Kolkata",
    ).delete(5)
    observations = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": pd.Series(3000.0, index=range(len(timestamps)), dtype="Float64"),
        }
    )

    first_hour = build_hourly_demand(observations, _config()).iloc[0]

    assert first_hour["timestamp"] == pd.Timestamp(
        "2024-01-01 08:00:00", tz="Asia/Kolkata"
    )
    assert first_hour["n_expected"] == 12

    assert first_hour["n_timestamp_rows"] == 11
    assert first_hour["n_observations"] == 11
    assert first_hour["n_blank_loads"] == 0
    assert first_hour["coverage_fraction"] == 11 / 12


    assert first_hour["quality_flag"] == "usable_partial"


def test_write_hourly_demand_round_trips_parquet(tmp_path: Path) -> None:

    hourly = build_hourly_demand(_quality_fixture(), _config())
    output = tmp_path / "nested" / "hourly.parquet"

    result = write_hourly_demand(hourly, output)

    assert result == output
    assert output.is_file()
    assert_frame_equal(pd.read_parquet(output), hourly)
