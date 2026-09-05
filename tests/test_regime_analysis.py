from pathlib import Path

import pandas as pd

from delhi_grid.datasets import model_feature_columns
from delhi_grid.evaluation.failure_analysis import (
    REFERENCE_MODEL,
    XGBOOST_MODEL,
    build_error_analysis_frame,
)
from delhi_grid.evaluation.regime_analysis import (
    attach_training_regime,
    build_fold_regime_summary,
)
from delhi_grid.evaluation.splits import MonthlyFold
from delhi_grid.models import load_xgboost_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _fold(fold_id: str = "2024-04") -> MonthlyFold:
    timezone = "Asia/Kolkata"
    return MonthlyFold(
        fold_id=fold_id,
        test_valid_start=pd.Timestamp("2024-04-01", tz=timezone),
        test_valid_end=pd.Timestamp("2024-04-30 23:00", tz=timezone),
        earliest_issue_time=pd.Timestamp("2024-03-31", tz=timezone),
        model_fit_time=pd.Timestamp("2024-03-31", tz=timezone),
        train_valid_time_cutoff=pd.Timestamp("2024-03-30 23:00", tz=timezone),
    )


def test_training_statistics_use_only_legal_targets() -> None:
    timezone = "Asia/Kolkata"
    frame = pd.DataFrame(
        {
            "valid_time": pd.DatetimeIndex(
                [
                    pd.Timestamp("2024-03-30 22:00", tz=timezone),
                    pd.Timestamp("2024-03-30 23:00", tz=timezone),
                    pd.Timestamp("2024-04-01", tz=timezone),
                    pd.Timestamp("2024-04-02", tz=timezone),
                    pd.Timestamp("2025-01-01", tz=timezone),
                ]
            ),
            "target_load_mw": pd.Series(
                [10.0, 20.0, 100.0, 1000.0, 9999.0], dtype="Float64"
            ),
        }
    )

    summary = build_fold_regime_summary(frame, [_fold()]).iloc[0]

    assert summary["training_rows"] == 2
    assert summary["latest_training_valid_time"] == pd.Timestamp(
        "2024-03-30 23:00", tz=timezone
    )
    assert summary["train_p99_mw"] == 19.9
    assert summary["train_max_mw"] == 20.0
    assert summary["test_max_mw"] == 1000.0


def test_exceedance_flags_use_own_fold_and_strict_boundaries() -> None:
    pairwise = pd.DataFrame(
        {
            "fold_id": ["a", "a", "a", "b"],
            "target_load_mw": pd.Series([100.0, 120.0, 130.0, 130.0]),
        }
    )
    fold_summary = pd.DataFrame(
        {
            "fold_id": ["a", "b"],
            "train_p90_mw": [100.0, 90.0],
            "train_p95_mw": [110.0, 100.0],
            "train_p99_mw": [120.0, 110.0],
            "train_max_mw": [130.0, 120.0],
        }
    )

    result = attach_training_regime(pairwise, fold_summary)

    assert result["target_above_train_p90"].tolist() == [False, True, True, True]
    assert result["target_above_train_p99"].tolist() == [False, False, True, True]
    assert result["target_above_train_max"].tolist() == [False, False, False, True]
    assert result["target_minus_train_p99_mw"].tolist() == [-20.0, 0.0, 10.0, 20.0]


def test_regime_rows_keep_pairwise_support_and_labels_out_of_features() -> None:
    times = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-04-01", tz="Asia/Kolkata"),
            pd.Timestamp("2025-01-01", tz="Asia/Kolkata"),
        ]
    )
    common = {
        "fold_id": ["2024-04", "2025-01"],
        "issue_time": times - pd.Timedelta(hours=24),
        "valid_time": times,
        "target_load_mw": pd.Series([100.0, 1000.0], dtype="Float64"),
        "target_quality_flag": pd.Categorical(["complete", "complete"]),
    }
    xgb = pd.DataFrame(
        {
            **common,
            "model_name": XGBOOST_MODEL,
            "prediction_mw": pd.Series([90.0, 1.0], dtype="Float64"),
        }
    )
    reference = pd.DataFrame(
        {
            **common,
            "model_name": REFERENCE_MODEL,
            "prediction_mw": pd.Series([95.0, 1.0], dtype="Float64"),
        }
    )
    unrelated = reference.copy()
    unrelated["model_name"] = "previous_week"
    unrelated["prediction_mw"] = pd.Series([pd.NA, pd.NA], dtype="Float64")
    baselines = pd.concat([reference, unrelated], ignore_index=True)
    pairwise = build_error_analysis_frame(
        xgb,
        baselines,
        development_start=times[0],
        development_end=times[0],
    )
    thresholds = pd.DataFrame(
        {
            "fold_id": ["2024-04"],
            "train_p90_mw": [80.0],
            "train_p95_mw": [85.0],
            "train_p99_mw": [90.0],
            "train_max_mw": [95.0],
        }
    )

    diagnostics = attach_training_regime(pairwise, thresholds)
    feature_config, _ = load_xgboost_config(CONFIG_PATH)
    features = model_feature_columns(feature_config)

    assert diagnostics["valid_time"].tolist() == [times[0]]
    assert diagnostics["target_above_train_max"].tolist() == [True]
    assert not set(diagnostics.columns).intersection(features)
