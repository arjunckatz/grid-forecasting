from pathlib import Path

import pandas as pd
import pytest

from delhi_grid.datasets import model_feature_columns
from delhi_grid.evaluation.failure_analysis import (
    REFERENCE_MODEL,
    XGBOOST_MODEL,
    build_development_ramps,
    build_error_analysis_frame,
    summarize_comparison,
)
from delhi_grid.models import load_xgboost_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    times = pd.DatetimeIndex(
        list(pd.date_range("2024-04-01", periods=4, freq="h", tz="Asia/Kolkata"))
        + [pd.Timestamp("2025-01-01", tz="Asia/Kolkata")]
    )
    common = {
        "fold_id": ["2024-04"] * 4 + ["2025-01"],
        "issue_time": times - pd.Timedelta(hours=24),
        "valid_time": times,
        "target_load_mw": pd.Series(
            [10.0, 20.0, 30.0, 40.0, 50.0], dtype="Float64"
        ),
        "target_quality_flag": pd.Categorical(["complete"] * 5),
    }
    xgb = pd.DataFrame(
        {
            **common,
            "model_name": [XGBOOST_MODEL] * 5,
            "prediction_mw": pd.Series(
                [12.0, 19.0, 29.0, 41.0, 100.0], dtype="Float64"
            ),
        }
    )
    reference = pd.DataFrame(
        {
            **common,
            "model_name": [REFERENCE_MODEL] * 5,
            "prediction_mw": pd.Series(
                [11.0, pd.NA, 33.0, 42.0, 0.0], dtype="Float64"
            ),
        }
    )
    other_baselines = []
    for model_name in ("previous_week", "trailing_4week_same_hour_median"):
        other = reference.copy()
        other["model_name"] = model_name
        other["prediction_mw"] = pd.Series(
            [9.0, 21.0, pd.NA, 39.0, 51.0],
            dtype="Float64",
        )
        other_baselines.append(other)
    baselines = pd.concat([reference, *other_baselines], ignore_index=True)
    return xgb, baselines, times


def test_analysis_frame_uses_identical_support_and_signed_error_direction() -> None:
    xgb, baselines, times = _predictions()
    unrelated_at_third_hour = baselines.loc[
        baselines["valid_time"].eq(times[2])
        & ~baselines["model_name"].eq(REFERENCE_MODEL)
    ]
    assert unrelated_at_third_hour["prediction_mw"].isna().all()

    frame = build_error_analysis_frame(
        xgb,
        baselines,
        development_start=times[0],
        development_end=times[3],
    )

    assert frame["valid_time"].tolist() == [times[0], times[2], times[3]]
    assert frame["xgb_signed_error_mw"].tolist() == [2.0, -1.0, 1.0]
    assert frame["reference_signed_error_mw"].tolist() == [1.0, 3.0, 2.0]
    assert frame["xgb_error_delta_mw"].tolist() == [1.0, -2.0, -1.0]
    comparison = summarize_comparison(frame, slice_type="toy").iloc[0]
    assert comparison["xgb_mae_mw"] == pytest.approx(4 / 3)
    assert comparison["reference_mae_mw"] == pytest.approx(2.0)
    assert comparison["xgb_skill_vs_reference"] == pytest.approx(1 / 3)


def test_ramps_do_not_bridge_missing_hours_or_include_post_development() -> None:
    timestamps = pd.DatetimeIndex(
        list(pd.date_range("2024-04-01", periods=4, freq="h", tz="Asia/Kolkata"))
        + list(pd.date_range("2025-01-01", periods=2, freq="h", tz="Asia/Kolkata"))
    )
    hourly = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": pd.Series(
                [10.0, pd.NA, 30.0, 40.0, 100.0, 1000.0], dtype="Float64"
            ),
        }
    )

    ramps = build_development_ramps(
        hourly,
        development_start=timestamps[0],
        development_end=timestamps[3],
    )

    assert ramps["valid_time"].max() == timestamps[3]
    assert ramps["ramp_mw"].isna().tolist() == [True, True, True, False]
    assert ramps.loc[3, "ramp_mw"] == 10.0
    assert ramps["abs_ramp_mw"].dropna().quantile(0.9) == 10.0


def test_diagnostic_labels_are_not_model_features() -> None:
    feature_config, _ = load_xgboost_config(CONFIG_PATH)
    features = model_feature_columns(feature_config)

    assert "target_load_mw" not in features
    assert "ramp_mw" not in features
    assert "abs_ramp_mw" not in features
