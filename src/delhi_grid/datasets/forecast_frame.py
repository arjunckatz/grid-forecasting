"""Construct explicit issue-time/valid-time forecasting opportunities."""

import pandas as pd

REQUIRED_HOURLY_COLUMNS = {"timestamp", "load_mw", "quality_flag"}


def build_forecast_frame(
    hourly_demand: pd.DataFrame,
    *,
    horizon_hours: int,
) -> pd.DataFrame:
    """Attach fixed-horizon forecast semantics to every canonical hourly row."""

    missing_columns = sorted(
        REQUIRED_HOURLY_COLUMNS.difference(hourly_demand.columns)
    )
    if missing_columns:
        raise ValueError(f"hourly demand missing columns: {missing_columns}")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")

    valid_time = hourly_demand["timestamp"]
    if not isinstance(valid_time.dtype, pd.DatetimeTZDtype):
        raise ValueError("hourly timestamps must be timezone-aware")
    if valid_time.isna().any() or valid_time.duplicated().any():
        raise ValueError("hourly timestamps must be non-null and unique")


    #uhhh
    frame = pd.DataFrame(
        {
            "issue_time": valid_time - pd.Timedelta(hours=horizon_hours),
            "valid_time": valid_time,
            "horizon_hours": pd.Series(
                horizon_hours,
                index=hourly_demand.index,
                dtype="int16",
            ),
            "target_load_mw": hourly_demand["load_mw"].astype("Float64"),
            "target_quality_flag": hourly_demand["quality_flag"],
        }
    )


    return frame.reset_index(drop=True)
