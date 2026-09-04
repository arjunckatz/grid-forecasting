"""Explicit, auditable input frame for the load-only learned model."""

import pandas as pd

from delhi_grid.features import (
    CALENDAR_FEATURE_COLUMNS,
    LoadFeatureConfig,
    build_calendar_features,
    build_load_features,
    load_feature_columns,
)

MODEL_METADATA_COLUMNS = (
    "issue_time",
    "valid_time",
    "target_load_mw",
    "target_quality_flag",
)
MODEL_PROVENANCE_COLUMNS = (
    "latest_load_feature_source_time",
    "latest_load_feature_available_time",
)


def model_feature_columns(config: LoadFeatureConfig) -> tuple[str, ...]:
    """Return the only columns permitted to enter the learned model."""

    return CALENDAR_FEATURE_COLUMNS + load_feature_columns(config)


def build_model_frame(
    forecast_frame: pd.DataFrame,
    hourly_demand: pd.DataFrame,
    config: LoadFeatureConfig,
) -> pd.DataFrame:
    """Combine forecast metadata with an explicit load-only feature contract."""

    missing = sorted(set(MODEL_METADATA_COLUMNS).difference(forecast_frame.columns))
    if missing:
        raise ValueError(f"forecast frame missing columns: {missing}")
    frame = forecast_frame.reset_index(drop=True)
    calendar = build_calendar_features(frame["valid_time"])
    load = build_load_features(frame, hourly_demand, config)
    return pd.concat([frame[list(MODEL_METADATA_COLUMNS)], calendar, load], axis=1)
