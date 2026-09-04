"""Fixed native-XGBoost configuration and training for the load-only model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
import yaml

from delhi_grid.features import LoadFeatureConfig

EXPECTED_MODEL_NAME = "xgboost_load_only"


@dataclass(frozen=True)
class XGBoostConfig:
    """One deliberately fixed CPU benchmark configuration."""

    model_name: str
    boosting_rounds: int
    objective: str
    eval_metric: str
    tree_method: str
    max_depth: int
    eta: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    seed: int
    nthread: int

    def __post_init__(self) -> None:
        if self.model_name != EXPECTED_MODEL_NAME:
            raise ValueError(f"model_name must be {EXPECTED_MODEL_NAME}")
        if self.boosting_rounds <= 0:
            raise ValueError("boosting_rounds must be positive")
        if self.objective != "reg:squarederror":
            raise ValueError("objective must be reg:squarederror")
        if self.tree_method != "hist":
            raise ValueError("tree_method must be hist")
        if self.nthread != 1:
            raise ValueError("nthread must be 1 for reproducibility")

    def training_params(self) -> dict[str, str | int | float]:
        """Return parameters accepted by the native XGBoost training API."""

        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "max_depth": self.max_depth,
            "eta": self.eta,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
            "nthread": self.nthread,
        }


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = tuple(int(item) for item in value)
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def load_xgboost_config(
    path: str | Path,
) -> tuple[LoadFeatureConfig, XGBoostConfig]:
    """Load the established load-only feature and XGBoost configuration."""

    with Path(path).open(encoding="utf-8") as config_file:
        document = _require_mapping(yaml.safe_load(config_file), "config")
    features = _require_mapping(document.get("features"), "features")
    model = _require_mapping(document.get("xgboost"), "xgboost")
    try:
        feature_config = LoadFeatureConfig(
            issue_lags_hours=_positive_int_tuple(
                features["issue_lags_hours"], "issue_lags_hours"
            ),
            valid_lags_hours=_positive_int_tuple(
                features["valid_lags_hours"], "valid_lags_hours"
            ),
            rolling_windows_hours=_positive_int_tuple(
                features["rolling_windows_hours"], "rolling_windows_hours"
            ),
            rolling_minimum_fraction=float(features["rolling_minimum_fraction"]),
        )
        model_config = XGBoostConfig(
            model_name=str(model["model_name"]),
            boosting_rounds=int(model["boosting_rounds"]),
            objective=str(model["objective"]),
            eval_metric=str(model["eval_metric"]),
            tree_method=str(model["tree_method"]),
            max_depth=int(model["max_depth"]),
            eta=float(model["eta"]),
            min_child_weight=float(model["min_child_weight"]),
            subsample=float(model["subsample"]),
            colsample_bytree=float(model["colsample_bytree"]),
            seed=int(model["seed"]),
            nthread=int(model["nthread"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid load-only XGBoost config: {error}") from error
    return feature_config, model_config


def train_predict_xgboost(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    config: XGBoostConfig,
) -> pd.Series:
    """Fit one native CPU booster and predict without dropping missing features."""

    if training_frame["target_load_mw"].isna().any():
        raise ValueError("XGBoost training targets must be non-missing")
    excluded = {
        "target_load_mw",
        "target_quality_flag",
        "issue_time",
        "valid_time",
        "fold_id",
        "max_source_time",
        "max_source_available_time",
        "latest_load_feature_source_time",
        "latest_load_feature_available_time",
    }
    overlap = sorted(excluded.intersection(feature_columns))
    if overlap:
        raise ValueError(f"non-feature columns cannot enter XGBoost: {overlap}")

    train_matrix = xgb.DMatrix(
        training_frame.loc[:, feature_columns].to_numpy(
            dtype="float32", na_value=float("nan")
        ),
        label=training_frame["target_load_mw"].to_numpy(dtype="float32"),
        feature_names=list(feature_columns),
    )
    prediction_matrix = xgb.DMatrix(
        prediction_frame.loc[:, feature_columns].to_numpy(
            dtype="float32", na_value=float("nan")
        ),
        feature_names=list(feature_columns),
    )
    booster = xgb.train(
        config.training_params(),
        train_matrix,
        num_boost_round=config.boosting_rounds,
    )
    return pd.Series(booster.predict(prediction_matrix), dtype="Float64")
