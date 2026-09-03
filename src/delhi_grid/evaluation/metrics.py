"""Transparent point-forecast metrics and support-aware comparisons."""

from dataclasses import asdict, dataclass
from math import sqrt

import pandas as pd


@dataclass(frozen=True)
class RegressionMetrics:
    """Point metrics reported as MW or fractional error, never percentages."""

    n_scored: int
    mae_mw: float
    rmse_mw: float
    wape: float
    mape: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def compute_metrics(
    target: pd.Series,
    prediction: pd.Series,
) -> RegressionMetrics:
    """Compute metrics on non-missing pairs and expose the scored row count."""

    pairs = pd.DataFrame({"target": target, "prediction": prediction}).dropna()
    if pairs.empty:
        raise ValueError("at least one non-missing target/prediction pair is required")
    if pairs["target"].eq(0).any():
        raise ValueError("MAPE is undefined when any scored target is zero")

    errors = pairs["target"] - pairs["prediction"]
    absolute_errors = errors.abs()
    absolute_target_sum = pairs["target"].abs().sum()
    if absolute_target_sum == 0:
        raise ValueError("WAPE is undefined when absolute target sum is zero")

    return RegressionMetrics(
        n_scored=len(pairs),
        mae_mw=float(absolute_errors.mean()),
        rmse_mw=float(sqrt((errors.pow(2)).mean())),
        wape=float(absolute_errors.sum() / absolute_target_sum),
        mape=float((absolute_errors / pairs["target"].abs()).mean()),
    )


def evaluate_prediction_support(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report own- and common-support metrics per fold and over all rows."""

    required = {
        "fold_id",
        "model_name",
        "valid_time",
        "target_load_mw",
        "prediction_mw",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")

    model_names = list(predictions["model_name"].drop_duplicates())
    contexts: list[tuple[str, str, pd.DataFrame]] = [
        ("development", "all", predictions)
    ]
    contexts.extend(
        ("fold", str(fold_id), fold_rows)
        for fold_id, fold_rows in predictions.groupby("fold_id", sort=True)
    )

    rows: list[dict[str, object]] = []
    for scope, fold_id, context in contexts:
        target_by_time = context.drop_duplicates("valid_time").set_index("valid_time")[
            "target_load_mw"
        ]
        evaluable_times = target_by_time[target_by_time.notna()].index
        availability = context.pivot(
            index="valid_time", columns="model_name", values="prediction_mw"
        ).reindex(columns=model_names)
        common_times = availability.reindex(evaluable_times).dropna().index

        for model_name in model_names:
            model_rows = context.loc[context["model_name"].eq(model_name)].set_index(
                "valid_time"
            )
            own_rows = model_rows.reindex(evaluable_times)
            own_available = own_rows["prediction_mw"].notna()
            own_metrics = compute_metrics(
                own_rows.loc[own_available, "target_load_mw"],
                own_rows.loc[own_available, "prediction_mw"],
            )
            common_rows = model_rows.reindex(common_times)
            common_metrics = compute_metrics(
                common_rows["target_load_mw"], common_rows["prediction_mw"]
            )

            for support, available_count, metrics in [
                ("own", int(own_available.sum()), own_metrics),
                ("common", len(common_times), common_metrics),
            ]:
                rows.append(
                    {
                        "scope": scope,
                        "fold_id": fold_id,
                        "support": support,
                        "model_name": model_name,
                        "evaluable_targets": len(evaluable_times),
                        "available_predictions": available_count,
                        "prediction_coverage": available_count
                        / len(evaluable_times),
                        **metrics.to_dict(),
                    }
                )
    return pd.DataFrame(rows)
