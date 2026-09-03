import json
from pathlib import Path

from delhi_grid.data.quality import audit_demand
from delhi_grid.data.sldc import read_sldc_csv

FIXTURE = Path(__file__).parent / "fixtures" / "sldc_edge_cases.csv"


def test_audit_reports_parse_and_duplicate_failures() -> None:
    audit = audit_demand(read_sldc_csv(FIXTURE))

    assert audit.total_row_count == 12
    assert audit.parsed_timestamp_count == 10
    assert audit.null_timestamp_count == 1
    assert audit.malformed_timestamp_count == 1
    assert audit.null_load_count == 1
    assert audit.non_numeric_load_count == 1
    assert audit.unique_timestamp_count == 9
    assert audit.duplicate_timestamp_count == 1
    assert audit.duplicate_timestamp_group_count == 1
    assert audit.conflicting_duplicate_timestamp_count == 1
    assert not audit.timestamps_monotonic_in_source_order
    assert audit.backwards_timestamp_movement_count == 1
    assert audit.timestamps_monotonic_after_sorting


def test_audit_reports_intervals_coverage_and_values_without_cleaning() -> None:
    audit = audit_demand(read_sldc_csv(FIXTURE))

    assert audit.first_timestamp == "2024-01-01T00:00:00+05:30"
    assert audit.last_timestamp == "2024-01-01T00:47:00+05:30"
    assert audit.observed_interval_seconds == {"120": 1, "300": 6, "900": 1}
    assert audit.five_minute_interval_count == 6
    assert audit.sub_5_minute_interval_count == 1
    assert audit.greater_than_5_minute_interval_count == 1
    assert audit.non_5_minute_interval_count == 2
    assert audit.missing_expected_slot_count == 2
    assert audit.off_5_minute_grid_timestamp_count == 1
    assert audit.complete_timestamp_day_count == 0
    assert audit.partial_timestamp_day_count == 1
    assert audit.missing_timestamp_day_count == 0
    assert audit.complete_numeric_load_day_count == 0
    assert audit.partial_numeric_load_day_count == 1
    assert audit.empty_numeric_load_day_count == 0
    assert audit.numeric_load_count == 10
    assert audit.minimum_load_mw == -1
    assert audit.maximum_load_mw == 99999
    assert audit.zero_load_count == 1
    assert audit.negative_load_count == 1
    json.dumps(audit.to_dict())
