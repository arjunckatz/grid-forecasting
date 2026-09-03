"""Read the Delhi SLDC-derived Kaggle CSV without applying cleaning policy."""

from pathlib import Path

import pandas as pd

DEFAULT_TIMEZONE = "Asia/Kolkata"
SOURCE_TIMESTAMP_COLUMN = "timestamp"
SOURCE_LOAD_COLUMN = "load_MW"
SOURCE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class SldcSchemaError(ValueError):
    """Raised when a raw CSV does not match the verified source schema."""


def read_sldc_csv(
    path: str | Path,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Read the verified source schema into a raw-evidence-preserving table.

    Timestamps are parsed exactly as ``YYYY-mm-dd HH:MM:SS`` and interpreted in
    ``timezone``. Failed timestamp and demand parses remain missing and are
    marked rather than discarded.
    """

    source_path = Path(path)
    header = pd.read_csv(source_path, nrows=0, encoding="utf-8", sep=",")
    expected_columns = [SOURCE_TIMESTAMP_COLUMN, SOURCE_LOAD_COLUMN]
    if list(header.columns) != expected_columns:
        raise SldcSchemaError(
            f"expected columns {expected_columns}, found {list(header.columns)}"
        )

    source = pd.read_csv(
        source_path,
        usecols=expected_columns,
        dtype="string",
        encoding="utf-8",
        keep_default_na=False,
        na_filter=False,
        sep=",",
        skip_blank_lines=False,
    )
    timestamp_raw = source[SOURCE_TIMESTAMP_COLUMN]
    load_raw = source[SOURCE_LOAD_COLUMN]

    timestamp_text = timestamp_raw.str.strip().mask(lambda values: values.eq(""))
    parsed_timestamp = pd.to_datetime(
        timestamp_text,
        format=SOURCE_TIMESTAMP_FORMAT,
        errors="coerce",
        exact=True,
    )
    parsed_timestamp = parsed_timestamp.dt.tz_localize(
        timezone,
        ambiguous="raise",
        nonexistent="raise",
    )

    load_text = load_raw.str.strip().mask(lambda values: values.eq(""))
    parsed_load = pd.to_numeric(load_text, errors="coerce").astype("Float64")

    return pd.DataFrame(
        {
            "source_line_number": pd.RangeIndex(2, len(source) + 2),
            "timestamp_raw": timestamp_raw,
            "load_mw_raw": load_raw,
            "timestamp": parsed_timestamp,
            "load_mw": parsed_load,
            "timestamp_parse_error": parsed_timestamp.isna(),
            "load_parse_error": parsed_load.isna(),
        }
    )
