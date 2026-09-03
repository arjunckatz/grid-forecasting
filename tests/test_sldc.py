from pathlib import Path

import pandas as pd
import pytest

from delhi_grid.data.sldc import SldcSchemaError, read_sldc_csv

FIXTURE = Path(__file__).parent / "fixtures" / "sldc_edge_cases.csv"


def test_read_sldc_csv_parses_timezone_and_preserves_evidence() -> None:
    observations = read_sldc_csv(FIXTURE)

    assert list(observations.columns) == [
        "source_line_number",
        "timestamp_raw",
        "load_mw_raw",
        "timestamp",
        "load_mw",
        "timestamp_parse_error",
        "load_parse_error",
    ]
    assert str(observations["timestamp"].dt.tz) == "Asia/Kolkata"
    assert observations.loc[0, "timestamp"] == pd.Timestamp(
        "2024-01-01 00:10:00", tz="Asia/Kolkata"
    )
    assert observations.loc[4, "load_mw_raw"] == "not available"
    assert pd.isna(observations.loc[4, "load_mw"])
    assert observations.loc[4, "load_parse_error"]
    assert observations.loc[7, "load_mw"] == 99999
    assert observations.loc[7, "load_mw_raw"] == "99999"
    assert observations["source_line_number"].tolist() == list(range(2, 14))


def test_read_sldc_csv_keeps_duplicate_and_unsorted_rows() -> None:
    observations = read_sldc_csv(FIXTURE)

    assert len(observations) == 12
    assert observations["timestamp"].dropna().duplicated().sum() == 1
    assert not observations["timestamp"].dropna().is_monotonic_increasing


def test_read_sldc_csv_rejects_an_unexpected_source_schema(tmp_path: Path) -> None:
    wrong_source = tmp_path / "wrong.csv"
    wrong_source.write_text("date,demand\n2024-01-01,4000\n", encoding="utf-8")

    with pytest.raises(SldcSchemaError, match="expected columns"):
        read_sldc_csv(wrong_source)
