"""Information-availability semantics for canonical hourly demand."""

import pandas as pd

HOURLY_AVAILABILITY_DELAY = pd.Timedelta(hours=1)


def hourly_load_available_time(bucket_start: pd.Series) -> pd.Series:
    """Return when hour-start-labelled demand buckets are fully observable."""

    return bucket_start + HOURLY_AVAILABILITY_DELAY
