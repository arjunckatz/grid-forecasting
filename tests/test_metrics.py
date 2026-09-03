from math import sqrt

import pandas as pd
import pytest

from delhi_grid.evaluation.metrics import (
    compute_metrics,
    evaluate_prediction_support,
)


def test_metrics_match_hand_computed_fractional_definitions() -> None:
    metrics = compute_metrics(
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([2.0, 2.0, 1.0]),
    )

    assert metrics.n_scored == 3
    assert metrics.mae_mw == pytest.approx(1.0)
    assert metrics.rmse_mw == pytest.approx(sqrt(5 / 3))
    assert metrics.wape == pytest.approx(0.5)
    assert metrics.mape == pytest.approx(5 / 9)


def test_metrics_filter_missing_pairs_but_report_scored_count() -> None:
    metrics = compute_metrics(
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([2.0, pd.NA, 1.0], dtype="Float64"),
    )

    assert metrics.n_scored == 2
    assert metrics.mae_mw == pytest.approx(1.5)


def test_metrics_reject_zero_targets_explicitly() -> None:
    with pytest.raises(ValueError, match="MAPE is undefined"):
        compute_metrics(pd.Series([0.0, 2.0]), pd.Series([1.0, 2.0]))


def test_evaluation_distinguishes_own_and_common_support() -> None:
    times = pd.date_range("2024-04-01", periods=3, freq="h", tz="Asia/Kolkata")
    predictions = pd.DataFrame(
        {
            "fold_id": ["2024-04"] * 6,
            "model_name": ["a"] * 3 + ["b"] * 3,
            "valid_time": list(times) * 2,
            "target_load_mw": [10.0, 20.0, pd.NA] * 2,
            "prediction_mw": pd.Series(
                [11.0, 18.0, 30.0, 12.0, pd.NA, 30.0], dtype="Float64"
            ),
        }
    )

    metrics = evaluate_prediction_support(predictions)
    aggregate = metrics.loc[metrics["scope"].eq("development")]
    own_a = aggregate.loc[
        aggregate["support"].eq("own") & aggregate["model_name"].eq("a")
    ].iloc[0]
    own_b = aggregate.loc[
        aggregate["support"].eq("own") & aggregate["model_name"].eq("b")
    ].iloc[0]
    common = aggregate.loc[aggregate["support"].eq("common")]

    assert own_a["evaluable_targets"] == 2
    assert own_a["available_predictions"] == 2
    assert own_a["mae_mw"] == pytest.approx(1.5)
    assert own_b["available_predictions"] == 1
    assert own_b["prediction_coverage"] == pytest.approx(0.5)
    assert common["n_scored"].eq(1).all()
    assert common["available_predictions"].eq(1).all()
