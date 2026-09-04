"""Calendar features known in advance for each forecast valid time."""

from math import cos, pi, sin

import pandas as pd

CALENDAR_FEATURE_COLUMNS = (
    "valid_hour",
    "valid_day_of_week",
    "valid_is_weekend",
    "valid_month",
    "valid_day_of_year",
    "valid_hour_sin",
    "valid_hour_cos",
    "valid_year_sin",
    "valid_year_cos",
)
MEAN_GREGORIAN_YEAR_DAYS = 365.2425


def build_calendar_features(valid_time: pd.Series) -> pd.DataFrame:
    """Build deterministic calendar inputs from timezone-aware valid times."""

    if not isinstance(valid_time.dtype, pd.DatetimeTZDtype):
        raise ValueError("valid_time must be timezone-aware")

    hour = valid_time.dt.hour
    day_of_week = valid_time.dt.dayofweek
    day_of_year = valid_time.dt.dayofyear
    hour_angle = hour * (2 * pi / 24)
    year_angle = (day_of_year - 1) * (2 * pi / MEAN_GREGORIAN_YEAR_DAYS)
    return pd.DataFrame(
        {
            "valid_hour": hour.astype("int16"),
            "valid_day_of_week": day_of_week.astype("int16"),
            "valid_is_weekend": day_of_week.ge(5).astype("int8"),
            "valid_month": valid_time.dt.month.astype("int16"),
            "valid_day_of_year": day_of_year.astype("int16"),
            "valid_hour_sin": hour_angle.map(sin),
            "valid_hour_cos": hour_angle.map(cos),
            "valid_year_sin": year_angle.map(sin),
            "valid_year_cos": year_angle.map(cos),
        },
        index=valid_time.index,
    )
