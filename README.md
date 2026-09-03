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
| Primary comparison | Strongest leakage-safe naive baseline |
| First ML model | XGBoost, after baselines and evaluation exist |

The machine-readable invariants are recorded in
[`configs/v1_24h.yaml`](configs/v1_24h.yaml).

## Forecast-time semantics and leakage

Every example must distinguish three fields:

- `issue_time`: when the forecast is made
- `valid_time`: when the prediction applies
- `horizon_hours`: the interval between them

For V1, `valid_time = issue_time + 24 hours`. Load observations and rolling
statistics may use information only through `issue_time`. Calendar properties
of `valid_time` are legal because they are known in advance. Weather at
`valid_time` is legal only when it comes from a forecast issued at or before
`issue_time`; realized future weather is not an operational feature.

These rules will be enforced in feature construction and tests in later,
focused milestones. Realized weather may be used for diagnostic slices or an
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

**Milestone 1: scaffolding and V1 contract.** The repository currently contains
only packaging, configuration, documentation, and an import smoke test. Demand
ingestion, aggregation, forecast-frame construction, evaluation, models, and
weather integration have intentionally not been implemented yet.

## Development

Python 3.11 or newer is required. Create a virtual environment and install the
package with its small development toolset:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

No runtime dependencies are included until implementation requires them.
