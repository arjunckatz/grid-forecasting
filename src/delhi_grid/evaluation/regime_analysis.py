"""Training-relative demand regime diagnostics for development forecasts."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from delhi_grid.datasets import build_forecast_frame, build_model_frame
from delhi_grid.evaluation.failure_analysis import (
    build_error_analysis_frame,
    summarize_comparison,
)
from delhi_grid.evaluation.splits import (
    BacktestConfig,
    MonthlyFold,
    make_development_folds,
    select_fold_frames,
)
from delhi_grid.features import LoadFeatureConfig

DEFAULT_REGIME_METRICS_PATH = Path("data/processed/regime_shift_metrics.csv")
REGIME_ORDER = (
    "at_or_below_train_p90",
    "train_p90_to_p99",
    "above_train_p99",
)
EXCEEDANCE_BIN_ORDER = (
    "0_to_250_mw",
    "250_to_500_mw",
    "500_to_1000_mw",
    "above_1000_mw",
)


@dataclass(frozen=True)
class RegimeAnalysisResult:
    """Fold distributions and pairwise training-relative error diagnostics."""

    fold_summary: pd.DataFrame
    row_diagnostics: pd.DataFrame
    regime_metrics: pd.DataFrame
    fold_exceedance_metrics: pd.DataFrame
    exceedance_bins: pd.DataFrame
    exceedance_error_correlation: float


def _distribution_statistics(series: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min_mw": float(series.min()),
        f"{prefix}_mean_mw": float(series.mean()),
        f"{prefix}_median_mw": float(series.median()),
        f"{prefix}_std_mw": float(series.std()),
        f"{prefix}_p90_mw": float(series.quantile(0.90)),
        f"{prefix}_p95_mw": float(series.quantile(0.95)),
        f"{prefix}_p99_mw": float(series.quantile(0.99)),
        f"{prefix}_max_mw": float(series.max()),
    }


def build_fold_regime_summary(
    model_frame: pd.DataFrame,
    folds: list[MonthlyFold],
) -> pd.DataFrame:
    """Summarize each fold's legal training targets and evaluable test targets."""

    rows: list[dict[str, object]] = []
    for fold in folds:
        candidate_train, candidate_test = select_fold_frames(model_frame, fold)
        train = candidate_train.loc[candidate_train["target_load_mw"].notna()]
        test = candidate_test.loc[candidate_test["target_load_mw"].notna()]
        if train.empty or test.empty:
            raise ValueError(f"fold {fold.fold_id} has an empty target distribution")
        if train["valid_time"].gt(fold.train_valid_time_cutoff).any():
            raise AssertionError(f"fold {fold.fold_id} training exceeds legal cutoff")

        train_stats = _distribution_statistics(train["target_load_mw"], "train")
        test_stats = _distribution_statistics(test["target_load_mw"], "test")
        above_p99 = test["target_load_mw"].gt(train_stats["train_p99_mw"])
        above_max = test["target_load_mw"].gt(train_stats["train_max_mw"])
        rows.append(
            {
                "fold_id": fold.fold_id,
                "model_fit_time": fold.model_fit_time,
                "train_valid_time_cutoff": fold.train_valid_time_cutoff,
                "latest_training_valid_time": train["valid_time"].max(),
                "training_rows": len(train),
                "test_evaluable_targets": len(test),
                **train_stats,
                **test_stats,
                "test_mean_minus_train_mean_mw": (
                    test_stats["test_mean_mw"] - train_stats["train_mean_mw"]
                ),
                "test_p90_minus_train_p90_mw": (
                    test_stats["test_p90_mw"] - train_stats["train_p90_mw"]
                ),
                "test_p99_minus_train_p99_mw": (
                    test_stats["test_p99_mw"] - train_stats["train_p99_mw"]
                ),
                "test_max_minus_train_max_mw": (
                    test_stats["test_max_mw"] - train_stats["train_max_mw"]
                ),
                "test_above_train_p99_count": int(above_p99.sum()),
                "test_above_train_p99_fraction": float(above_p99.mean()),
                "test_above_train_max_count": int(above_max.sum()),
                "test_above_train_max_fraction": float(above_max.mean()),
            }
        )
    return pd.DataFrame(rows)


def attach_training_regime(
    pairwise_frame: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach each row's own fold thresholds and retrospective exceedance labels."""

    threshold_columns = [
        "fold_id",
        "train_p90_mw",
        "train_p95_mw",
        "train_p99_mw",
        "train_max_mw",
    ]
    frame = pairwise_frame.merge(
        fold_summary[threshold_columns],
        on="fold_id",
        how="left",
        validate="many_to_one",
    )
    if frame[threshold_columns[1:]].isna().any().any():
        raise ValueError("pairwise rows are missing fold training thresholds")

    target = frame["target_load_mw"]
    frame["target_above_train_p90"] = target.gt(frame["train_p90_mw"])
    frame["target_above_train_p95"] = target.gt(frame["train_p95_mw"])
    frame["target_above_train_p99"] = target.gt(frame["train_p99_mw"])
    frame["target_above_train_max"] = target.gt(frame["train_max_mw"])
    frame["target_minus_train_p99_mw"] = target - frame["train_p99_mw"]
    frame["target_minus_train_max_mw"] = target - frame["train_max_mw"]
    frame["training_relative_regime"] = "above_train_p99"
    frame.loc[
        target.le(frame["train_p90_mw"]), "training_relative_regime"
    ] = "at_or_below_train_p90"
    frame.loc[
        target.gt(frame["train_p90_mw"])
        & target.le(frame["train_p99_mw"]),
        "training_relative_regime",
    ] = "train_p90_to_p99"
    frame["training_relative_regime"] = pd.Categorical(
        frame["training_relative_regime"], categories=REGIME_ORDER, ordered=True
    )
    return frame


def _error_summary(frame: pd.DataFrame, slice_type: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "slice_type": slice_type,
                    "slice_value": "all",
                    "n_rows": 0,
                    "xgb_mae_mw": float("nan"),
                    "reference_mae_mw": float("nan"),
                    "xgb_rmse_mw": float("nan"),
                    "reference_rmse_mw": float("nan"),
                    "xgb_mean_signed_error_mw": float("nan"),
                    "reference_mean_signed_error_mw": float("nan"),
                    "xgb_skill_vs_reference": float("nan"),
                }
            ]
        )
    return summarize_comparison(frame, slice_type=slice_type)


def build_regime_error_metrics(row_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compare models over disjoint upper-tail regimes and above training maximum."""

    partition = summarize_comparison(
        row_diagnostics,
        slice_type="training_relative_regime",
        group_column="training_relative_regime",
    )
    above_max = _error_summary(
        row_diagnostics.loc[row_diagnostics["target_above_train_max"]],
        "above_train_max",
    )
    return pd.concat([partition, above_max], ignore_index=True)


def build_fold_exceedance_metrics(row_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Report fold-level pairwise errors above training p99 and maximum."""

    rows = []
    for fold_id, fold_rows in row_diagnostics.groupby("fold_id", sort=True):
        for threshold_name, flag in (
            ("above_train_p99", "target_above_train_p99"),
            ("above_train_max", "target_above_train_max"),
        ):
            summary = _error_summary(fold_rows.loc[fold_rows[flag]], threshold_name)
            summary.insert(0, "fold_id", fold_id)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def build_exceedance_bins(row_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise errors by distance above the fold's training p99."""

    exceedance = row_diagnostics.loc[
        row_diagnostics["target_above_train_p99"]
    ].copy()
    exceedance["p99_exceedance_bin"] = pd.cut(
        exceedance["target_minus_train_p99_mw"],
        bins=[0.0, 250.0, 500.0, 1000.0, float("inf")],
        labels=EXCEEDANCE_BIN_ORDER,
    )
    return summarize_comparison(
        exceedance,
        slice_type="p99_exceedance_bin",
        group_column="p99_exceedance_bin",
    )


def _add_prediction_summary(
    fold_summary: pd.DataFrame,
    row_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    prediction_summary = (
        row_diagnostics.groupby("fold_id", sort=True)
        .agg(
            pairwise_rows=("valid_time", "size"),
            pairwise_target_p99_mw=("target_load_mw", lambda x: x.quantile(0.99)),
            pairwise_target_max_mw=("target_load_mw", "max"),
            xgb_prediction_p99_mw=(
                "xgb_prediction_mw",
                lambda x: x.quantile(0.99),
            ),
            reference_prediction_p99_mw=(
                "reference_prediction_mw",
                lambda x: x.quantile(0.99),
            ),
            xgb_prediction_max_mw=("xgb_prediction_mw", "max"),
            reference_prediction_max_mw=("reference_prediction_mw", "max"),
        )
        .reset_index()
    )
    return fold_summary.merge(
        prediction_summary, on="fold_id", how="left", validate="one_to_one"
    )


def run_regime_analysis(
    hourly_demand: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    xgboost_predictions: pd.DataFrame,
    backtest_config: BacktestConfig,
    feature_config: LoadFeatureConfig,
) -> RegimeAnalysisResult:
    """Run cutoff-safe training-regime diagnostics without regenerating forecasts."""

    folds = make_development_folds(backtest_config)
    development_end = folds[-1].test_valid_end
    development_hourly = hourly_demand.loc[
        hourly_demand["timestamp"].le(development_end)
    ].copy()
    forecast = build_forecast_frame(
        development_hourly, horizon_hours=backtest_config.horizon_hours
    )
    model_frame = build_model_frame(forecast, development_hourly, feature_config)
    fold_summary = build_fold_regime_summary(model_frame, folds)

    pairwise = build_error_analysis_frame(
        xgboost_predictions,
        baseline_predictions,
        development_start=folds[0].test_valid_start,
        development_end=development_end,
    )
    row_diagnostics = attach_training_regime(pairwise, fold_summary)
    if row_diagnostics["valid_time"].gt(development_end).any():
        raise AssertionError("regime analysis includes post-development rows")
    fold_summary = _add_prediction_summary(fold_summary, row_diagnostics)
    regime_metrics = build_regime_error_metrics(row_diagnostics)
    fold_exceedance_metrics = build_fold_exceedance_metrics(row_diagnostics)
    exceedance_bins = build_exceedance_bins(row_diagnostics)
    above_p99 = row_diagnostics.loc[row_diagnostics["target_above_train_p99"]]
    correlation = float(
        above_p99["target_minus_train_p99_mw"].corr(
            above_p99["xgb_signed_error_mw"]
        )
    )
    return RegimeAnalysisResult(
        fold_summary=fold_summary,
        row_diagnostics=row_diagnostics,
        regime_metrics=regime_metrics,
        fold_exceedance_metrics=fold_exceedance_metrics,
        exceedance_bins=exceedance_bins,
        exceedance_error_correlation=correlation,
    )


def write_regime_analysis(
    result: RegimeAnalysisResult,
    path: str | Path = DEFAULT_REGIME_METRICS_PATH,
) -> Path:
    """Write the compact per-fold regime-shift artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.fold_summary.to_csv(output, index=False, lineterminator="\n")
    return output
