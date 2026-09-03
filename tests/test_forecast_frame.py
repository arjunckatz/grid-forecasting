import pandas as pd

from delhi_grid.datasets import build_forecast_frame


def test_forecast_frame_retains_all_targets_and_explicit_horizon() -> None:
    timestamps = pd.date_range(
        "2024-01-01 00:00:00", periods=3, freq="h", tz="Asia/Kolkata"
    )
    hourly = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": pd.Series([100.0, pd.NA, 120.0], dtype="Float64"),
            "quality_flag": pd.Categorical(
                ["complete", "missing", "usable_partial"]
            ),
        }
    )

    frame = build_forecast_frame(hourly, horizon_hours=24)

    assert list(frame.columns) == [
        "issue_time",
        "valid_time",
        "horizon_hours",
        "target_load_mw",
        "target_quality_flag",
    ]
    assert len(frame) == 3
    assert (frame["valid_time"] - frame["issue_time"]).eq(
        pd.Timedelta(hours=24)
    ).all()
    assert frame.loc[0, "issue_time"] == pd.Timestamp(
        "2023-12-31 00:00:00", tz="Asia/Kolkata"
    )
    assert pd.isna(frame.loc[1, "target_load_mw"])
    assert frame.loc[1, "target_quality_flag"] == "missing"
    assert str(frame["issue_time"].dt.tz) == "Asia/Kolkata"
    assert str(frame["valid_time"].dt.tz) == "Asia/Kolkata"
