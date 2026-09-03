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

Planned local location:

```text
data/raw/sldc/
```

The ingestion milestone will document exact filenames, retrieval date,
checksums where useful, source schema, and discovered quality limitations. In
particular, duplicate timestamps, gaps, conflicting duplicate values, invalid
measurements, and irregular intervals must be audited before modeling.

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

## Generated data

Later pipelines are expected to create canonical and modeling artifacts beneath
`data/processed/`, including quality-aware 5-minute and hourly demand datasets.
Those outputs, raw downloads, checksums generated from private local copies, and
model artifacts are ignored by Git. Only small, purpose-built test fixtures may
be committed under `tests/`.
