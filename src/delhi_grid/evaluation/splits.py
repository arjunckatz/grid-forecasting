"""Leakage-safe monthly expanding-window split definitions."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from delhi_grid.datasets import HOURLY_AVAILABILITY_DELAY

EXPECTED_BACKTEST_FREQUENCY = "monthly"
EXPECTED_HORIZON_HOURS = 24
EXPECTED_TIMEZONE = "Asia/Kolkata"


class BacktestConfigError(ValueError):
    """Raised when V1 backtest configuration is absent or inconsistent."""


@dataclass(frozen=True)
class BacktestConfig:
    """Established V1 horizon and evaluation boundaries."""

    timezone: str
    horizon_hours: int
    development_valid_start: date
    development_valid_end: date
    holdout_valid_start: date
    holdout_valid_end: date
    backtest_frequency: str

    def __post_init__(self) -> None:
        if self.timezone != EXPECTED_TIMEZONE:
            raise BacktestConfigError("timezone must be Asia/Kolkata")
        if self.horizon_hours != EXPECTED_HORIZON_HOURS:
            raise BacktestConfigError("V1 horizon must be exactly 24 hours")
        if self.backtest_frequency != EXPECTED_BACKTEST_FREQUENCY:
            raise BacktestConfigError("backtest_frequency must be monthly")
        if self.development_valid_start > self.development_valid_end:
            raise BacktestConfigError("development date range is reversed")
        if self.holdout_valid_start > self.holdout_valid_end:
            raise BacktestConfigError("holdout date range is reversed")
        if self.development_valid_end >= self.holdout_valid_start:
            raise BacktestConfigError("development and holdout periods overlap")


@dataclass(frozen=True)
class MonthlyFold:
    """One valid-time test month, model-fit time, and legal label cutoff."""

    fold_id: str
    test_valid_start: pd.Timestamp
    test_valid_end: pd.Timestamp
    earliest_issue_time: pd.Timestamp
    model_fit_time: pd.Timestamp
    train_valid_time_cutoff: pd.Timestamp


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BacktestConfigError(f"{name} must be a mapping")
    return value


def _parse_date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise BacktestConfigError(f"{name} must be an ISO date") from error


def load_backtest_config(path: str | Path) -> BacktestConfig:
    """Load the V1 horizon and chronology needed by monthly backtests."""

    with Path(path).open(encoding="utf-8") as config_file:
        document = _require_mapping(yaml.safe_load(config_file), "config")
    time = _require_mapping(document.get("time"), "time")
    forecast = _require_mapping(document.get("forecast"), "forecast")
    evaluation = _require_mapping(document.get("evaluation"), "evaluation")
    try:
        return BacktestConfig(
            timezone=str(time["timezone"]),
            horizon_hours=int(forecast["horizon_hours"]),
            development_valid_start=_parse_date(
                evaluation["development_valid_start"], "development_valid_start"
            ),
            development_valid_end=_parse_date(
                evaluation["development_valid_end"], "development_valid_end"
            ),
            holdout_valid_start=_parse_date(
                evaluation["holdout_valid_start"], "holdout_valid_start"
            ),
            holdout_valid_end=_parse_date(
                evaluation["holdout_valid_end"], "holdout_valid_end"
            ),
            backtest_frequency=str(evaluation["backtest_frequency"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BacktestConfigError(f"invalid backtest config field: {error}") from error


def _local_midnight(day: date, timezone: str) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(timezone)


def make_development_folds(config: BacktestConfig) -> list[MonthlyFold]:
    """Create valid-month folds whose training labels are observable at fit."""

    development_start = _local_midnight(
        config.development_valid_start, config.timezone
    )
    development_end = _local_midnight(config.development_valid_end, config.timezone)
    month_starts = pd.date_range(
        development_start,
        development_end,
        freq="MS",
    )
    horizon = pd.Timedelta(hours=config.horizon_hours)
    folds: list[MonthlyFold] = []
    for month_start in month_starts:
        next_month = month_start + pd.offsets.MonthBegin(1)
        test_end = next_month - pd.Timedelta(hours=1)
        earliest_issue = month_start - horizon
        train_valid_time_cutoff = earliest_issue - HOURLY_AVAILABILITY_DELAY
        folds.append(
            MonthlyFold(
                fold_id=month_start.strftime("%Y-%m"),
                test_valid_start=month_start,
                test_valid_end=test_end,
                earliest_issue_time=earliest_issue,
                model_fit_time=earliest_issue,
                train_valid_time_cutoff=train_valid_time_cutoff,
            )
        )
    return folds


def select_fold_frames(
    forecast_frame: pd.DataFrame,
    fold: MonthlyFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return expanding training history and one valid-time test month."""

    train = forecast_frame.loc[
        forecast_frame["valid_time"].le(fold.train_valid_time_cutoff)
    ].copy()
    test = forecast_frame.loc[
        forecast_frame["valid_time"].between(
            fold.test_valid_start, fold.test_valid_end
        )
    ].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)
