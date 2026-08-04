# Dental Appointment No-show Prediction
> **Synthetic-data and use disclaimer**
>
> All records in this repository are fully synthetic. They do not represent
> real patients, clinical records, or healthcare operations. This educational
> project is not validated for clinical or operational use.
## Overview
This repository implements a reproducible, leakage-controlled workflow for
studying dental appointment no-show prediction. It includes immutable synthetic
raw inputs, prediction-time and target contracts, leakage-aware feature
eligibility, a chronological temporal split, canonical dataset construction,
exploratory data analysis, deterministic preprocessing, baseline estimators,
a deterministic Random Forest comparator, and temporal-validation model
comparison.
The repository contains reproducible baseline and tree-comparison validation
results. It does not persist a trained production model or report final
test-set performance.
## Current project status
Phases 01 through 08 are complete.
The implemented workflow now:
- verifies immutable raw-file integrity;
- constructs and validates the canonical analytical dataset;
- enforces prediction-time, temporal-split, and label-maturity safeguards;
- performs leakage-safe exploratory analysis;
- exposes ten approved modeling features;
- fits preprocessing only on the mature development population;
- fits the approved baseline estimators and a fixed Random Forest comparator;
- selects models by temporal-validation average precision;
- evaluates three deterministic baselines and one Random Forest comparator on
  temporal validation;
- reports probability metrics and a fixed `0.5` classification audit.
The development population contains 3,670 appointments and 432 no-shows. The
validation population contains 1,541 appointments and 192 no-shows.
The primary validation metric is average precision:
| Baseline | Average precision | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| `dummy_prior` | 0.124594 | 0.500000 | 0.109118 | 0.376205 |
| `logistic_unweighted` | 0.120959 | 0.476028 | 0.112111 | 0.396686 |
| `logistic_balanced` | 0.120720 | 0.475866 | 0.183289 | 0.555112 |
Under the declared average-precision rule, `dummy_prior` is the selected
baseline. This does not establish model usefulness or operational readiness.
The logistic baselines did not outperform the prevalence-only reference on the
observed future validation period.
Calibration, operational threshold and cost analysis, final pre-test fitting,
and untouched test-set evaluation remain outside the current implementation.
## Repository structure
```text
.
|-- data/
|   |-- raw/                         # Immutable synthetic source files
|   |-- interim/                     # Reserved intermediate products
|   `-- processed/                   # Generated dataset outputs
|-- docs/
|   |-- README.md                    # Methodology index
|   |-- phase_06_exploratory_data_analysis.md
|   |-- phase_07_baseline_modeling.md
|   `-- phase_08_tree_based_comparison.md
|-- reports/
|   `-- eda/                         # Generated and ignored EDA artifacts
|-- src/
|   |-- analysis/                    # Leakage-safe EDA implementation
|   |-- data/                        # Canonical dataset construction
|   `-- modeling/                    # Baseline and tree-comparison modeling
|-- tests/                           # Automated contracts and safeguards
|-- requirements.txt
|-- requirements-dev.txt
`-- requirements.lock.txt
```
Generated files under `data/processed/` and `reports/eda/` are ignored by Git
and should be rebuilt from immutable raw inputs.
## Quick start
Use Python 3.12 and run these commands from PowerShell:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.data.build_dataset
.\.venv\Scripts\python.exe -m src.analysis.run_eda
```
Another Python 3.12 interpreter may be used when the Windows `py` launcher is
not available.
## Generated outputs
Dataset construction generates:
- `data/processed/analytical_dataset.parquet`
- `data/processed/analytical_dataset.manifest.json`
Exploratory analysis generates deterministic CSV and PNG artifacts under:
- `reports/eda/`
Model comparison evaluates in memory and intentionally does not serialize
estimators, probabilities, metric tables, or test-set results.
## Methodology
The [documentation index](docs/README.md) links the approved contracts and
implemented phases in methodological order.
The modeling pipeline uses only the ten approved prediction-time features.
Identifiers, target and split fields, maturity flags, final outcomes,
post-event fields, evaluation-only fields, and test rows are excluded from
model fitting.
Preprocessing and estimators are fitted only on mature development rows.
Validation labels are used only for evaluation after fitting. Test features and
targets remain untouched.
## Scope boundary
The repository currently implements dataset construction, exploratory analysis,
baseline preprocessing and fitting, a fixed Random Forest comparator, and
temporal-validation model comparison.
It does not yet implement probability calibration, calibration-method
selection, operational threshold optimization, cost analysis, final pre-test
fitting, model serialization, deployment, or final test-set evaluation.
## Disclaimer
All source records and derived outputs are synthetic. No real patient
information is included, and repository results cannot establish clinical
effectiveness. This project must not be used to support clinical or operational
decisions.
