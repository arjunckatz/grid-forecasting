from pathlib import Path

import pandas as pd

from delhi_grid.datasets import build_forecast_frame
from delhi_grid.evaluation.splits import (
    load_backtest_config,
    make_development_folds,
    select_fold_frames,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "v1_24h.yaml"


def _forecast_history() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2023-04-01 00:00:00",
        "2024-12-31 23:00:00",
        freq="h",
        tz="Asia/Kolkata",
    )
    hourly = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": pd.Series(4000.0, index=range(len(timestamps)), dtype="Float64"),
            "quality_flag": pd.Categorical(["complete"] * len(timestamps)),
        }
    )
    return build_forecast_frame(hourly, horizon_hours=24)


def test_development_folds_use_valid_months_and_legal_cutoffs() -> None:
    config = load_backtest_config(CONFIG_PATH)
    folds = make_development_folds(config)

    assert [fold.fold_id for fold in folds] == [
        "2024-04",
        "2024-05",
        "2024-06",
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
    ]
    april = folds[0]
    assert april.test_valid_start == pd.Timestamp(
        "2024-04-01 00:00:00", tz="Asia/Kolkata"
    )
    assert april.test_valid_end == pd.Timestamp(
        "2024-04-30 23:00:00", tz="Asia/Kolkata"
    )
    assert april.earliest_issue_time == pd.Timestamp(
        "2024-03-31 00:00:00", tz="Asia/Kolkata"
    )
    assert april.model_fit_time == april.earliest_issue_time
    assert april.train_valid_time_cutoff == pd.Timestamp(
        "2024-03-30 23:00:00", tz="Asia/Kolkata"
    )
    assert all(
        left.test_valid_end < right.test_valid_start
        for left, right in zip(folds, folds[1:])
    )


def test_fold_selection_expands_training_without_crossing_cutoff() -> None:
    frame = _forecast_history()
    folds = make_development_folds(load_backtest_config(CONFIG_PATH))
    training_sizes = []

    for fold in folds:
        train, test = select_fold_frames(frame, fold)
        training_sizes.append(len(train))
        assert train["valid_time"].max() <= fold.train_valid_time_cutoff
        assert (
            train["valid_time"] + pd.Timedelta(hours=1) <= fold.model_fit_time
        ).all()
        assert test["valid_time"].min() == fold.test_valid_start
        assert test["valid_time"].max() == fold.test_valid_end
        assert test["valid_time"].between(
            fold.test_valid_start, fold.test_valid_end
        ).all()

    assert training_sizes == sorted(training_sizes)
    assert len(set(training_sizes)) == len(training_sizes)
