from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from delhi_grid.evaluation import (
    load_backtest_config,
    run_development_baselines,
    write_baseline_results,
)
from delhi_grid.models import BASELINE_NAMES

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _hourly_history() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2023-04-01 00:00:00",
        "2024-12-31 23:00:00",
        freq="h",
        tz="Asia/Kolkata",
    )
    loads = pd.Series(
        3000.0 + (timestamps.hour * 10),
        index=range(len(timestamps)),
        dtype="Float64",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": loads,
            "quality_flag": pd.Categorical(["complete"] * len(timestamps)),
        }
    )


def test_development_backtest_schema_and_holdout_boundary() -> None:
    predictions, metrics = run_development_baselines(
        _hourly_history(), load_backtest_config(CONFIG_PATH)
    )

    assert predictions["fold_id"].nunique() == 9
    assert set(predictions["model_name"]) == set(BASELINE_NAMES)
    assert list(predictions.columns) == [
        "fold_id",
        "model_name",
        "issue_time",
        "valid_time",
        "target_load_mw",
        "target_quality_flag",
        "prediction_mw",
        "max_source_time",
        "max_source_available_time",
    ]
    assert predictions["valid_time"].max() == pd.Timestamp(
        "2024-12-31 23:00:00", tz="Asia/Kolkata"
    )
    available = predictions["prediction_mw"].notna()
    assert predictions.loc[available, "max_source_available_time"].le(
        predictions.loc[available, "issue_time"]
    ).all()
    assert set(metrics["support"]) == {"own", "common"}
    assert set(metrics["scope"]) == {"fold", "development"}


def test_result_artifacts_round_trip(tmp_path: Path) -> None:
    predictions, metrics = run_development_baselines(
        _hourly_history(), load_backtest_config(CONFIG_PATH)
    )
    prediction_path = tmp_path / "predictions.parquet"
    metrics_path = tmp_path / "metrics.csv"

    write_baseline_results(
        predictions,
        metrics,
        predictions_path=prediction_path,
        metrics_path=metrics_path,
    )

    assert_frame_equal(pd.read_parquet(prediction_path), predictions)
    restored_metrics = pd.read_csv(metrics_path)
    assert len(restored_metrics) == len(metrics)
    assert list(restored_metrics.columns) == list(metrics.columns)
