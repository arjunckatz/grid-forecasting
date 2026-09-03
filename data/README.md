# Data sources and local layout

Datasets are not stored in Git. This directory records provenance and the
intended local layout; retrieval and validation commands will be added with the
milestones that implement those pipelines.

## Electricity demand

The authoritative upstream source is the **Delhi State Load Despatch Centre
(Delhi SLDC)** [Load Curve](https://www.delhisldc.org/Loadcurve.aspx). It
publishes Delhi demand telemetry at approximately five-minute resolution.

The Kaggle dataset
[Delhi SLDC Load Data (5-min resolution)](https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution)
is an optional historical mirror for bootstrapping reproducible development. It
is not the authoritative source. Its published coverage is approximately April
2023 through January 2026 and must be verified after retrieval rather than
assumed.

Expected local location:

```text
data/raw/sldc/
```

The downloaded `archive (4).zip` contains `load_data.csv`; the extracted file is
byte-identical to that archive member. The CSV is UTF-8 without a byte-order
mark, comma-delimited, and has exactly two columns with a header:

```text
timestamp,load_MW
```

`timestamp` uses `%Y-%m-%d %H:%M:%S` local wall time and `load_MW` represents
Delhi total demand in MW. The 293,184 data rows span `2023-04-01 00:00:00`
through `2026-01-12 23:55:00`, inclusive. Every record has two fields; there
are no blank records or repeated headers.

SHA-256 provenance:

```text
archive (4).zip  9b58db1733c96eb0c453e8aaf7be925840aed14d9bbb6b3aab0f0831981aca45
load_data.csv    4a3b3f2da4fd3d6eb8e87d49905477ac07b5a1d8eba5b3be3038044bdf0b91d6
```

Ingestion preserves each CSV row, the original timestamp/load text, and its
source line number. Parsed timestamps are timezone-aware in `Asia/Kolkata` and
failed timestamp or demand parses are flagged without dropping the evidence.
The audit reports row and parse counts, timestamp range, duplicate/conflicting
timestamps, interval distribution, gaps, missing five-minute slots, grid
alignment, timestamp-grid and numeric-load daily completeness, and basic demand
statistics. In this copy every timestamp parses, is unique, lies on the
five-minute grid, and follows the preceding timestamp by five minutes. However,
7,932 rows have a blank load field, so timestamp completeness must not be
mistaken for usable-load completeness. Across 1,018 calendar days, all 1,018
have 288 timestamps; by numeric load availability, 129 days are complete, 886
are partial, and 3 have no numeric loads. The 285,252 numeric values range from
1,450.4 MW to 8,631.53 MW; there are no zero or negative values. Values are
reported, not removed; the SLDC site itself cautions that unavailable real-time
data can produce misleading steep graph lines. Commit 2 diagnoses these
conditions but makes no cleaning, deduplication, interpolation, or hourly
aggregation decisions.

## Observed weather (planned)

Observed weather will come from **Meteostat**, using the New Delhi / Safdarjung
station:

- Meteostat/WMO station ID: `42182`
- ICAO: `VIDD`
- Approximate coordinates: 28.5833, 77.2
- Local timezone: `Asia/Kolkata`

These observations are intended first for realized-weather diagnostics and
evaluation slices. Weather observed at a future target time is not a legal
operational forecasting feature.

Planned local location:

```text
data/raw/weather_observed/
```

## Archived weather forecasts (planned)

Historical forecast inputs are planned from **Open-Meteo Previous Runs** (or an
equivalently auditable archive). A forecast value may be joined to a demand
target only when its forecast run was available at or before the demand
`issue_time`. This as-of rule is essential: joining realized future temperature
would leak information.

Planned local location:

```text
data/raw/weather_forecasts/
```

## Canonical hourly demand

The generated V1 modeling target is:

```text
data/processed/hourly_demand.parquet
```

Each timezone-aware timestamp is the start of an hour. Its aggregated value is
conservatively treated as fully available one hour after that timestamp.
`n_expected` records 12 expected slots, `n_timestamp_rows` records present
source rows, `n_observations` records valid numeric loads, and `n_blank_loads`
records present rows without a numeric load. Coverage and one of `complete`,
`usable_partial`, `insufficient`, or `missing` complete the row.
Hours with fewer than 9 numeric observations remain present but have no numeric
target. The pipeline does not interpolate or impute them.

For the audited source, this produces 24,432 hourly rows: 23,112 `complete`, 640
`usable_partial`, 225 `insufficient`, and 455 `missing`. Thus 23,752 hours
(97.22%) have usable targets. Missingness is not random: every one of the 127
hours exactly at the 9/12 threshold lacks minutes `:45`, `:50`, and `:55`, and
125 of those hours begin at 22:00 or 23:00. The partial-hour mean follows the V1
contract but may not represent the unobserved end of an evening ramp; no
correction is invented here.

Generated outputs, raw downloads, and model artifacts are ignored by Git. Only
small, purpose-built test fixtures may be committed under `tests/`.

## Development baseline artifacts

Commit 4 produces two local, ignored outputs:

```text
data/processed/baseline_predictions.parquet
data/processed/baseline_metrics.csv
```

The prediction table records the monthly fold, baseline name, `issue_time`,
`valid_time`, retained target and quality flag, prediction, latest source bucket
start (`max_source_time`), and when that bucket was fully observable
(`max_source_available_time`). Every available prediction must satisfy
`max_source_available_time <= issue_time`.
The metrics table reports fold and concatenated-development results on both
each model's own support and the common support shared by all baselines, with
prediction availability kept separate from error.

Development folds cover valid times from April through December 2024. A fold's
model-fit time equals its earliest issue time, and its latest legal training
`valid_time` is one hour earlier so that the label is fully observable. The 2025
holdout is deliberately not evaluated or written to these artifacts.
