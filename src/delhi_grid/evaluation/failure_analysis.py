"""Development-only failure slices for XGBoost and its fixed reference baseline."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from delhi_grid.datasets import build_forecast_frame, build_model_frame
from delhi_grid.evaluation.metrics import compute_metrics
from delhi_grid.evaluation.splits import BacktestConfig
from delhi_grid.features import LoadFeatureConfig, load_feature_columns

XGBOOST_MODEL = "xgboost_load_only"
REFERENCE_MODEL = "previous_day_last_completed_hour"
DEMAND_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
DEMAND_LABELS = ("bottom_25", "25_to_50", "50_to_75", "75_to_90", "top_10")
DEFAULT_SLICE_METRICS_PATH = Path("data/processed/failure_slice_metrics.csv")
DEFAULT_WORST_ERRORS_PATH = Path("data/processed/worst_xgboost_errors.csv")


@dataclass(frozen=True)
class FailureAnalysisResult:
    """Reusable analysis tables and development-only thresholds."""

    analysis_frame: pd.DataFrame
    slice_metrics: pd.DataFrame
    demand_boundaries_mw: tuple[float, ...]
    high_ramp_threshold_mw: float
    daily_peaks: pd.DataFrame
    regime_summary: pd.DataFrame
    exact_nine_hour_counts: pd.DataFrame
    exact_nine_sensitivity: pd.DataFrame
    worst_errors: pd.DataFrame


def _development_bounds(config: BacktestConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(config.development_valid_start).tz_localize(config.timezone)
    end = (
        pd.Timestamp(config.development_valid_end).tz_localize(config.timezone)
        + pd.Timedelta(days=1)
        - pd.Timedelta(hours=1)
    )
    return start, end


def build_error_analysis_frame(
    xgboost_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    development_start: pd.Timestamp,
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build one row per identical-support XGBoost/reference forecast."""

    xgb = xgboost_predictions.loc[
        xgboost_predictions["model_name"].eq(XGBOOST_MODEL)
        & xgboost_predictions["valid_time"].between(
            development_start, development_end
        )
    ].copy()
    reference = baseline_predictions.loc[
        baseline_predictions["model_name"].eq(REFERENCE_MODEL)
        & baseline_predictions["valid_time"].between(
            development_start, development_end
        )
    ].copy()
    keys = ["fold_id", "issue_time", "valid_time"]
    metadata = keys + ["target_load_mw", "target_quality_flag"]
    xgb = xgb[metadata + ["prediction_mw"]].rename(
        columns={"prediction_mw": "xgb_prediction_mw"}
    )
    reference = reference[metadata + ["prediction_mw"]].rename(
        columns={
            "target_load_mw": "reference_target_load_mw",
            "target_quality_flag": "reference_target_quality_flag",
            "prediction_mw": "reference_prediction_mw",
        }
    )
    frame = xgb.merge(reference, on=keys, how="inner", validate="one_to_one")
    if not frame["target_load_mw"].equals(frame["reference_target_load_mw"]):
        raise ValueError("XGBoost and reference targets differ")
    if not frame["target_quality_flag"].astype("string").equals(
        frame["reference_target_quality_flag"].astype("string")
    ):
        raise ValueError("XGBoost and reference target quality differs")
    frame = frame.drop(
        columns=["reference_target_load_mw", "reference_target_quality_flag"]
    )
    frame = frame.loc[
        frame["valid_time"].between(development_start, development_end)
        & frame["target_load_mw"].notna()
        & frame["xgb_prediction_mw"].notna()
        & frame["reference_prediction_mw"].notna()
    ].copy()
    if frame.empty:
        raise ValueError("development common support is empty")
    if frame["valid_time"].max() > development_end:
        raise AssertionError("failure analysis includes post-development rows")

    frame["xgb_signed_error_mw"] = (
        frame["xgb_prediction_mw"] - frame["target_load_mw"]
    )
    frame["reference_signed_error_mw"] = (
        frame["reference_prediction_mw"] - frame["target_load_mw"]
    )
    frame["xgb_abs_error_mw"] = frame["xgb_signed_error_mw"].abs()
    frame["reference_abs_error_mw"] = frame["reference_signed_error_mw"].abs()
    frame["xgb_error_delta_mw"] = (
        frame["xgb_abs_error_mw"] - frame["reference_abs_error_mw"]
    )
    return frame.sort_values("valid_time").reset_index(drop=True)


def build_development_ramps(
    hourly_demand: pd.DataFrame,
    *,
    development_start: pd.Timestamp,
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    """Calculate one-hour ramps without crossing missing or nonconsecutive rows."""

    hourly = (
        hourly_demand.loc[hourly_demand["timestamp"].le(development_end)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    previous_time = hourly["timestamp"].shift(1)
    previous_load = hourly["load_mw"].shift(1)
    consecutive = hourly["timestamp"].sub(previous_time).eq(pd.Timedelta(hours=1))
    usable = hourly["load_mw"].notna() & previous_load.notna() & consecutive
    ramp = (hourly["load_mw"] - previous_load).where(usable).astype("Float64")
    result = pd.DataFrame(
        {
            "valid_time": hourly["timestamp"],
            "ramp_mw": ramp,
            "abs_ramp_mw": ramp.abs(),
        }
    )
    return result.loc[
        result["valid_time"].between(development_start, development_end)
    ].reset_index(drop=True)


def summarize_comparison(
    frame: pd.DataFrame,
    *,
    slice_type: str,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Compare both models on identical rows for each requested slice."""

    groups = [("all", frame)] if group_column is None else frame.groupby(
        group_column, observed=True, sort=True
    )
    rows: list[dict[str, object]] = []
    for value, group in groups:
        xgb = compute_metrics(group["target_load_mw"], group["xgb_prediction_mw"])
        reference = compute_metrics(
            group["target_load_mw"], group["reference_prediction_mw"]
        )
        skill = (
            float("nan")
            if reference.mae_mw == 0
            else 1 - xgb.mae_mw / reference.mae_mw
        )
        rows.append(
            {
                "slice_type": slice_type,
                "slice_value": str(value),
                "n_rows": len(group),
                "xgb_mae_mw": xgb.mae_mw,
                "reference_mae_mw": reference.mae_mw,
                "xgb_rmse_mw": xgb.rmse_mw,
                "reference_rmse_mw": reference.rmse_mw,
                "xgb_mean_signed_error_mw": float(
                    group["xgb_signed_error_mw"].mean()
                ),
                "reference_mean_signed_error_mw": float(
                    group["reference_signed_error_mw"].mean()
                ),
                "xgb_skill_vs_reference": skill,
            }
        )
    return pd.DataFrame(rows)


def _assign_demand_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, ...]]:
    boundaries = tuple(
        float(value)
        for value in frame["target_load_mw"].quantile(DEMAND_QUANTILES).tolist()
    )
    if len(set(boundaries)) != len(boundaries):
        raise ValueError(
            f"development demand quantile edges are not unique: {boundaries}"
        )
    result = frame.copy()
    result["demand_bin"] = pd.cut(
        result["target_load_mw"],
        bins=boundaries,
        labels=DEMAND_LABELS,
        include_lowest=True,
    )
    return result, boundaries


def _daily_peak_frame(
    hourly_demand: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    *,
    development_start: pd.Timestamp,
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    hourly = hourly_demand.loc[
        hourly_demand["timestamp"].between(development_start, development_end)
        & hourly_demand["load_mw"].notna(),
        ["timestamp", "load_mw"],
    ].copy()
    hourly["calendar_day"] = hourly["timestamp"].dt.date
    peak_indices = hourly.groupby("calendar_day", sort=True)["load_mw"].idxmax()
    peaks = hourly.loc[peak_indices].rename(
        columns={"timestamp": "valid_time", "load_mw": "observed_peak_mw"}
    )
    columns = [
        "valid_time",
        "xgb_prediction_mw",
        "reference_prediction_mw",
        "xgb_signed_error_mw",
        "reference_signed_error_mw",
        "xgb_abs_error_mw",
        "reference_abs_error_mw",
    ]
    return peaks.merge(
        analysis_frame[columns], on="valid_time", how="left", validate="one_to_one"
    ).sort_values("valid_time").reset_index(drop=True)


def _regime_summary(
    hourly_demand: pd.DataFrame,
    ramps: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    *,
    high_ramp_threshold: float,
    development_start: pd.Timestamp,
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    hourly = hourly_demand.loc[
        hourly_demand["timestamp"].between(development_start, development_end)
        & hourly_demand["load_mw"].notna()
    ].copy()
    hourly["month"] = hourly["timestamp"].dt.month
    ramp_by_time = ramps.set_index("valid_time")
    hourly["abs_ramp_mw"] = hourly["timestamp"].map(ramp_by_time["abs_ramp_mw"])
    comparative = analysis_frame.assign(month=analysis_frame["valid_time"].dt.month)
    feature_missingness = comparative.groupby("month")[
        "n_missing_load_features"
    ].agg(["size", lambda values: values.gt(0).mean()])
    feature_missingness.columns = [
        "pairwise_support_rows",
        "load_feature_missing_fraction",
    ]
    rows = []
    for month, group in hourly.groupby("month", sort=True):
        ramps_present = group["abs_ramp_mw"].dropna()
        rows.append(
            {
                "month": int(month),
                "n_targets": len(group),
                "mean_target_mw": float(group["load_mw"].mean()),
                "target_std_mw": float(group["load_mw"].std()),
                "target_p90_mw": float(group["load_mw"].quantile(0.9)),
                "maximum_target_mw": float(group["load_mw"].max()),
                "mean_abs_ramp_mw": float(ramps_present.mean()),
                "high_ramp_fraction": float(
                    ramps_present.ge(high_ramp_threshold).mean()
                ),
                "usable_partial_fraction": float(
                    group["quality_flag"].eq("usable_partial").mean()
                ),
                "pairwise_support_rows": int(
                    feature_missingness.loc[month, "pairwise_support_rows"]
                ),
                "load_feature_missing_fraction": float(
                    feature_missingness.loc[month, "load_feature_missing_fraction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_failure_analysis(
    hourly_demand: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    xgboost_predictions: pd.DataFrame,
    backtest_config: BacktestConfig,
    feature_config: LoadFeatureConfig,
) -> FailureAnalysisResult:
    """Produce development-only error slices without regenerating predictions."""

    development_start, development_end = _development_bounds(backtest_config)
    development_hourly = hourly_demand.loc[
        hourly_demand["timestamp"].le(development_end)
    ].copy()
    frame = build_error_analysis_frame(
        xgboost_predictions,
        baseline_predictions,
        development_start=development_start,
        development_end=development_end,
    )
    canonical = development_hourly[
        ["timestamp", "n_observations", "coverage_fraction"]
    ].rename(columns={"timestamp": "valid_time"})
    frame = frame.merge(canonical, on="valid_time", how="left", validate="one_to_one")

    ramps = build_development_ramps(
        development_hourly,
        development_start=development_start,
        development_end=development_end,
    )
    frame = frame.merge(ramps, on="valid_time", how="left", validate="one_to_one")
    ramp_threshold = float(ramps["abs_ramp_mw"].dropna().quantile(0.9))

    forecast = build_forecast_frame(
        development_hourly, horizon_hours=backtest_config.horizon_hours
    )
    forecast = forecast.loc[
        forecast["valid_time"].between(development_start, development_end)
    ].reset_index(drop=True)
    model_frame = build_model_frame(forecast, development_hourly, feature_config)
    historical_features = load_feature_columns(feature_config)
    feature_missingness = pd.DataFrame(
        {
            "valid_time": model_frame["valid_time"],
            "n_missing_load_features": model_frame[list(historical_features)]
            .isna()
            .sum(axis=1),
        }
    )
    frame = frame.merge(
        feature_missingness, on="valid_time", how="left", validate="one_to_one"
    )

    frame["month"] = frame["valid_time"].dt.strftime("%Y-%m")
    frame["valid_hour"] = frame["valid_time"].dt.hour
    frame["day_type"] = frame["valid_time"].dt.dayofweek.ge(5).map(
        {False: "weekday", True: "weekend"}
    )
    frame["feature_missingness"] = frame["n_missing_load_features"].gt(0).map(
        {False: "none_missing", True: "at_least_one_missing"}
    )
    frame, demand_boundaries = _assign_demand_bins(frame)
    top_threshold = demand_boundaries[-2]
    frame["high_demand"] = frame["target_load_mw"].ge(top_threshold)
    frame["high_ramp"] = frame["abs_ramp_mw"].ge(ramp_threshold)
    frame["ramp_direction"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[frame["high_ramp"] & frame["ramp_mw"].gt(0), "ramp_direction"] = (
        "positive"
    )
    frame.loc[frame["high_ramp"] & frame["ramp_mw"].lt(0), "ramp_direction"] = (
        "negative"
    )

    slice_metrics = pd.concat(
        [
            summarize_comparison(frame, slice_type="overall"),
            summarize_comparison(frame, slice_type="month", group_column="month"),
            summarize_comparison(
                frame, slice_type="hour_of_day", group_column="valid_hour"
            ),
            summarize_comparison(
                frame, slice_type="day_type", group_column="day_type"
            ),
            summarize_comparison(
                frame, slice_type="demand_bin", group_column="demand_bin"
            ),
            summarize_comparison(
                frame.loc[frame["high_demand"]], slice_type="high_demand"
            ),
            summarize_comparison(
                frame.loc[frame["high_ramp"]], slice_type="high_ramp"
            ),
            summarize_comparison(
                frame.loc[frame["ramp_direction"].notna()],
                slice_type="ramp_direction",
                group_column="ramp_direction",
            ),
            summarize_comparison(
                frame, slice_type="target_quality", group_column="target_quality_flag"
            ),
            summarize_comparison(
                frame,
                slice_type="feature_missingness",
                group_column="feature_missingness",
            ),
        ],
        ignore_index=True,
    )

    exact_nine = frame.loc[frame["n_observations"].eq(9)]
    exact_nine_hours = (
        exact_nine.groupby("valid_hour", sort=True)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    exact_nine_sensitivity = pd.concat(
        [
            summarize_comparison(exact_nine, slice_type="exactly_9_of_12"),
            summarize_comparison(frame, slice_type="headline_with_exactly_9"),
            summarize_comparison(
                frame.loc[~frame["n_observations"].eq(9)],
                slice_type="headline_without_exactly_9",
            ),
        ],
        ignore_index=True,
    )
    peaks = _daily_peak_frame(
        development_hourly,
        frame,
        development_start=development_start,
        development_end=development_end,
    )
    comparable_peaks = peaks.dropna(
        subset=["xgb_prediction_mw", "reference_prediction_mw"]
    ).copy()
    comparable_peaks["target_load_mw"] = comparable_peaks["observed_peak_mw"]
    slice_metrics = pd.concat(
        [
            slice_metrics,
            summarize_comparison(comparable_peaks, slice_type="daily_peak"),
        ],
        ignore_index=True,
    )
    regime = _regime_summary(
        development_hourly,
        ramps,
        frame,
        high_ramp_threshold=ramp_threshold,
        development_start=development_start,
        development_end=development_end,
    )
    worst_columns = [
        "valid_time",
        "target_load_mw",
        "xgb_prediction_mw",
        "reference_prediction_mw",
        "xgb_abs_error_mw",
        "reference_abs_error_mw",
        "xgb_error_delta_mw",
        "target_quality_flag",
        "ramp_mw",
    ]
    worst = frame.sort_values(
        ["xgb_abs_error_mw", "valid_time"], ascending=[False, True]
    ).head(20)[worst_columns]
    return FailureAnalysisResult(
        analysis_frame=frame,
        slice_metrics=slice_metrics,
        demand_boundaries_mw=demand_boundaries,
        high_ramp_threshold_mw=ramp_threshold,
        daily_peaks=peaks,
        regime_summary=regime,
        exact_nine_hour_counts=exact_nine_hours,
        exact_nine_sensitivity=exact_nine_sensitivity,
        worst_errors=worst.reset_index(drop=True),
    )


def write_failure_analysis(
    result: FailureAnalysisResult,
    *,
    slice_metrics_path: str | Path = DEFAULT_SLICE_METRICS_PATH,
    worst_errors_path: str | Path = DEFAULT_WORST_ERRORS_PATH,
) -> tuple[Path, Path]:
    """Write the two compact, ignored failure-analysis artifacts."""

    slices_output = Path(slice_metrics_path)
    worst_output = Path(worst_errors_path)
    slices_output.parent.mkdir(parents=True, exist_ok=True)
    worst_output.parent.mkdir(parents=True, exist_ok=True)
    result.slice_metrics.to_csv(slices_output, index=False, lineterminator="\n")
    result.worst_errors.to_csv(worst_output, index=False, lineterminator="\n")
    return slices_output, worst_output
