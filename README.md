# Delhi Grid Forecasting

A reproducible, leakage-safe system for forecasting Delhi's total electricity
demand from messy operational time-series data. The project is designed to
demonstrate careful forecasting semantics, data-quality work, chronological
evaluation, strong baselines, and honest failure analysis—not a model
leaderboard or premature serving stack.

## Problem and motivation

Electricity-demand forecasts support grid operations only when they reflect the
information that was actually available when each forecast was issued. This
project will turn public Delhi State Load Despatch Centre (Delhi SLDC) telemetry
into an auditable forecasting benchmark, then test whether a load-only machine
learning model improves on strong seasonal baselines. Weather will be added only
after that benchmark is working and only with explicit availability semantics.

The intended progression is deliberately narrow: establish trustworthy demand
data, define forecasting examples without future leakage, evaluate
chronologically, benchmark simple methods, and investigate operationally
important failures such as peaks and ramps.

## V1 forecast contract

V1 answers one question:

> Given everything available at an `issue_time`, predict Delhi total
> electricity demand exactly 24 hours later.

| Property | V1 definition |
| --- | --- |
| Target | Delhi total electricity demand, in MW |
| Raw demand resolution | 5 minutes |
| Modeling resolution | Hourly |
| Hourly aggregation | Mean of valid 5-minute readings |
| Minimum hourly coverage | 9 of 12 readings (75%) |
| Forecast horizon | Fixed +24 hours |
| Timezone | `Asia/Kolkata`, timezone-aware |
| Primary metric | MAE in MW |
| Primary comparison | Fixed `previous_day_last_completed_hour` reference baseline |
| First ML model | XGBoost, after baselines and evaluation exist |

The machine-readable invariants are recorded in
[`configs/v1_24h.yaml`](configs/v1_24h.yaml).

## Forecast-time semantics and leakage

Every example must distinguish three fields:

- `issue_time`: when the forecast is made
- `valid_time`: when the prediction applies
- `horizon_hours`: the interval between them

For V1, `valid_time = issue_time + 24 hours`. Load observations and rolling
statistics may use only values fully available by `issue_time`. Calendar
properties of `valid_time` are legal because they are known in advance. Weather
at `valid_time` is legal only when it comes from a forecast issued at or before
`issue_time`; realized future weather is not an operational feature.

Canonical hourly timestamps label bucket starts. A value labeled `14:00` is the
mean of readings through `14:55` and is conservatively treated as fully
available at `15:00`. Historical load is legal only when its bucket availability
time, not merely its timestamp, is at or before `issue_time`.

The forecast frame retains every target hour, including hours whose target is
missing, so availability is measured rather than hidden by an early drop. Each
baseline prediction carries both `max_source_time` and
`max_source_available_time`; evaluation rejects availability later than
`issue_time`. Realized weather may be used for diagnostic slices or an
explicitly labelled oracle upper bound, never silently as a production input.

## Expected architecture

The planned data flow is:

```text
Delhi SLDC demand telemetry
  -> ingestion and validation
  -> canonical 5-minute demand
  -> quality-aware hourly demand
  -> explicit +24h forecast frame
  -> legal baselines and expanding-window backtests
  -> load-only XGBoost benchmark
  -> aggregate metrics and failure analysis
```

Observed weather and archived weather forecasts will remain separate source
pipelines until they are joined with explicit as-of semantics. Raw and generated
datasets remain local and are not committed to Git. See [`data/README.md`](data/README.md)
for source provenance and the planned local layout.

## Project status

**Current milestone: load-only forecast failure analysis.** The repository
builds explicit +24-hour forecast examples and evaluates three seasonal
baselines plus one fixed, compact learned model over monthly expanding-window
folds. It now diagnoses the learned model against the fixed
`previous_day_last_completed_hour` reference on their direct pairwise support.
Hourly resolution is the V1 modeling contract; the original
five-minute evidence remains available for quality, ramp, and event analysis.

The full historical mirror and generated evaluation artifacts are not stored in
Git. The 2025 holdout remains locked. Weather integration has intentionally not
been implemented yet.

## Audit a local demand file

Place `load_data.csv` beneath `data/raw/sldc/`. The reader validates the verified
`timestamp,load_MW` header and parses `%Y-%m-%d %H:%M:%S` timestamps:

```python
import json

from delhi_grid.data import audit_demand, read_sldc_csv

observations = read_sldc_csv("data/raw/sldc/load_data.csv")
report = audit_demand(observations)
print(json.dumps(report.to_dict(), indent=2))
```

The normalized table contains `source_line_number`, raw timestamp/load text,
parsed timezone-aware `timestamp`, numeric `load_mw`, and parse-error flags.
Every source row remains present and in source order.

## Build canonical hourly demand

An hourly timestamp labels the start of its bucket: `08:00` represents the 12
expected readings from `08:00` through `08:55`. Blank telemetry is missing—not
zero—and only numeric measurements contribute to coverage or the mean.

| Valid readings | Quality | Hourly target |
| ---: | --- | --- |
| 12 | `complete` | Mean of 12 readings |
| 9–11 | `usable_partial` | Mean of available readings |
| 1–8 | `insufficient` | Missing |
| 0 | `missing` | Missing |

Every expected hour remains in the output, including hours without source rows
or a usable target. The threshold comes from `configs/v1_24h.yaml`; changing it
changes classification rather than requiring a code edit.

Coverage metadata distinguishes the 12 expected slots (`n_expected`), source
rows actually present (`n_timestamp_rows`), valid numeric loads
(`n_observations`), and present rows with no numeric load (`n_blank_loads`). An
absent timestamp therefore reduces row and observation counts without being
mislabelled as a blank load.

```python
from delhi_grid.data import (
    build_hourly_demand,
    load_hourly_demand_config,
    read_sldc_csv,
    write_hourly_demand,
)

config = load_hourly_demand_config("configs/v1_24h.yaml")
raw = read_sldc_csv("data/raw/sldc/load_data.csv")
hourly = build_hourly_demand(raw, config)
write_hourly_demand(hourly)
```

The default generated artifact is `data/processed/hourly_demand.parquet`, which
is ignored by Git. No interpolation, imputation, or below-threshold target is
produced.

## Baselines and development evaluation

The legal V1 baselines are the latest completed prior-day-ish hour at lag 25h,
previous-week same-hour demand at lag 168h, and the median of available
same-hour observations at lags 168h, 336h, 504h, and 672h. The 25h baseline is
named `previous_day_last_completed_hour`: its source bucket becomes available
exactly at issue time. A missing historical source stays missing; the pipeline
does not impute it. For the four-week median, provenance records the latest
bucket start and its corresponding availability time.

Development evaluation uses nine monthly folds by `valid_time`, from April
through December 2024. Each fold is fit at its earliest test `issue_time` and
permits only labels fully observable then. For April, model-fit time is
`2024-03-31 00:00 Asia/Kolkata`, so the latest legal training `valid_time` is
`2024-03-30 23:00`. This is an expanding-window contract for later learned
models. Random train/test splitting is invalid because it would mix later grid
states into earlier forecast decisions.

MAE and RMSE are reported in MW. WAPE and MAPE are fractions, not percentages;
MAPE explicitly rejects zero-valued targets rather than silently choosing a
zero-denominator convention. Missing targets and predictions are excluded with
the scored count reported. Results are calculated both on each baseline's own
available support and on common support where every baseline predicts, at fold
level and once over the concatenated development predictions. Prediction
coverage is reported separately so a lower error cannot conceal poorer
availability.

The generated development-only artifacts are:

```text
data/processed/baseline_predictions.parquet
data/processed/baseline_metrics.csv
```

Both are ignored by Git. Their contents are restricted to April--December 2024;
the configured 2025 holdout is not scored during this milestone.

```python
import pandas as pd

from delhi_grid.evaluation import (
    load_backtest_config,
    run_development_baselines,
    write_baseline_results,
)

hourly = pd.read_parquet("data/processed/hourly_demand.parquet")
config = load_backtest_config("configs/v1_24h.yaml")
predictions, metrics = run_development_baselines(hourly, config)
write_baseline_results(predictions, metrics)
```

## Load-only XGBoost benchmark

`xgboost_load_only` uses nine calendar features known for `valid_time`, issue-time
load lags of 1, 2, 3, 6, 24, 48, and 168 hours, the target-seasonal load at
`valid_time - 168h`, and rolling means and sample standard deviations over 6,
24, and 168 completed hourly buckets. Raw timestamps, target load, target
quality, fold identifiers, predictions, and provenance timestamps are excluded
from the explicit model feature list.

Every issue-relative lag names its source bucket relative to `issue_time`.
Rolling windows end at `issue_time - 1h`; they require respectively 5, 18, and
126 observed loads, following a fixed 75% minimum-completeness policy. Missing
lag and rolling values remain missing for XGBoost's native handling. Calendar
features remain complete. Annual sine and cosine use day-of-year over the mean
Gregorian year length of 365.2425 days.

Each monthly model is fit once using legal expanding history, while test-row
features refresh using information available at that row's issue time. The
untuned V1 CPU configuration uses `reg:squarederror`, histogram trees, depth 6,
learning rate 0.05, full row and column sampling, seed 42, one thread, and 200
boosting rounds. It is a fixed benchmark rather than a development-tuned search.

Generated learned-model artifacts are:

```text
data/processed/xgboost_load_only_predictions.parquet
data/processed/xgboost_load_only_metrics.csv
```

The metrics artifact compares XGBoost with the existing legal baselines on own
and common support. Both files are ignored by Git and contain development
evaluation only; no 2025 features, predictions, or metrics are produced.

## Load-only failure analysis

On the 6,225-row XGBoost/reference pairwise support, the models remain nearly
tied: MAE is 323.09 MW versus 321.69 MW, for -0.44% XGBoost skill.
The aggregate result hides substantial April--December 2024 variation:

- XGBoost is weakest in May and June, with MAE of 639.57 and 607.58 MW and
  mean signed errors of -572.46 and -472.50 MW. It is strongest relative to
  the reference in November and December, with +41.72% and +64.68% skill.
- Error is concentrated at high demand. Above the development top-decile
  threshold of 6,501.33 MW, XGBoost MAE is 959.68 MW versus 451.53 MW for the
  reference, with -912.56 MW mean signed error.
- The development high-ramp threshold is 398.27 MW. XGBoost improves on the
  reference over all high-ramp rows (+15.46% skill), but not on strong negative
  ramps (-10.64% skill). This distinction cautions against treating volatility
  alone as the failure mechanism.
- At the observed daily peak hour, 261 days have pairwise predictions; XGBoost
  MAE is 456.13 MW versus 362.04 MW, with -378.99 MW mean signed error.
- Partial target quality and missing historical load features are sensitivity
  flags, not exclusions from official metrics. Removing the 32 pairwise rows
  with exactly 9 of 12 observations changes skill only from -0.437% to -0.306%.

May and June also have higher observed demand and larger average hourly ramps
than November, while December is more volatile than November. These are
associations, not causal explanations. Their alignment with the hottest-season
failures motivates testing legally available weather in a future experiment;
weather has not been joined here. The locked 2025 holdout remains uninspected.

The generated diagnostic tables are ignored by Git:

```text
data/processed/failure_slice_metrics.csv
data/processed/worst_xgboost_errors.csv
```

## Development

Python 3.11 or newer is required. Create a virtual environment and install the
package with its small development toolset:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Runtime dependencies are pandas, PyYAML, PyArrow, and XGBoost.
