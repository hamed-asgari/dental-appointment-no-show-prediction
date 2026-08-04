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
a deterministic Random Forest comparator, chronological probability
calibration, and temporal-validation candidate selection.
The repository contains reproducible baseline, tree-comparison, and
calibration-validation and operational-sensitivity results. It does not
persist a trained production model or report final test-set performance.
## Current project status
Phases 01 through 10 are complete.
The implemented workflow now:
- verifies immutable raw-file integrity;
- constructs and validates the canonical analytical dataset;
- enforces prediction-time, temporal-split, and label-maturity safeguards;
- performs leakage-safe exploratory analysis;
- exposes ten approved modeling features;
- fits deterministic baselines and a fixed Random Forest comparator;
- selects the Phase 08 comparison model by validation average precision;
- divides mature development data into chronological base-fit and calibration
  populations;
- fits sigmoid and isotonic calibrators around a frozen Random Forest;
- compares calibrated probabilities with a leakage-safe recent-prior reference;
- selects Phase 09 candidates by Brier score, then log loss;
- enumerates the two threshold policies induced by the selected prior;
- derives ex-ante and validation-replay break-even sensitivity boundaries;
- keeps validation labels outside fitting and ex-ante policy construction;
- leaves the test population untouched.
The Phase 09 populations contain 2,520 base-fit appointments, 1,150 calibration
appointments, and 1,541 temporal-validation appointments.
The declared Phase 09 validation results are:
| Candidate | AP | ROC-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| `calibration_prior` | 0.124594 | 0.500000 | 0.109071 | 0.375982 |
| `random_forest_uncalibrated` | 0.123084 | 0.507137 | 0.118920 | 0.404801 |
| `random_forest_sigmoid` | 0.118851 | 0.492863 | 0.109574 | 0.378405 |
| `random_forest_isotonic` | 0.123552 | 0.495622 | 0.109101 | 0.376116 |
Under the declared Brier-score and log-loss rule, `calibration_prior` is the
selected Phase 09 candidate. It assigns the same recent-prevalence probability
to every appointment and therefore provides no appointment-level ranking.
The evaluated Random Forest calibration methods do not demonstrate validated
probability value beyond that recent-prior reference.
Phase 10 confirms that the constant-probability reference yields only
`intervene_all` and `intervene_none`, and records the break-even boundary
without choosing costs, effectiveness, a threshold, or an operational policy.
Final pre-test fitting, persistence, deployment, and untouched test-set
evaluation remain outside the current implementation.
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
|   |-- phase_08_tree_based_comparison.md
|   `-- phase_09_probability_calibration.md
|-- reports/
|   `-- eda/                         # Generated and ignored EDA artifacts
|-- src/
|   |-- analysis/                    # Leakage-safe EDA implementation
|   |-- data/                        # Canonical dataset construction
|   `-- modeling/                    # Baseline, comparison, and calibration
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
Model comparison, probability calibration, and operational sensitivity
analysis evaluate in memory and intentionally do not serialize estimators,
probabilities, metric tables, policy tables, or test-set results.
## Methodology
The [documentation index](docs/README.md) links the approved contracts and
implemented phases in methodological order.
The modeling pipeline uses only the ten approved prediction-time features.
Identifiers, target and split fields, maturity flags, final outcomes,
post-event fields, evaluation-only fields, and test rows are excluded from
model fitting.
Phase 07 and Phase 08 preprocessing and estimators are fitted only on
mature development rows. In Phase 09, the base estimator is fitted only on the
earlier base-fit population. Calibration labels fit only the frozen calibrators
and recent-prior reference. Phase 10 uses calibration prevalence for the
ex-ante boundary and validation prevalence only for replay audit.
Test features and targets remain untouched.
## Scope boundary
The repository currently implements dataset construction, exploratory analysis,
baseline preprocessing and fitting, a fixed Random Forest comparator,
chronological probability calibration, temporal-validation calibration
candidate selection, deterministic threshold-state enumeration, and
break-even sensitivity analysis.
It does not select operational cost values, intervention effectiveness, a final
threshold, or an operational policy. It also does not yet implement final
pre-test fitting, model serialization, deployment, or final test-set evaluation.
## Disclaimer
All source records and derived outputs are synthetic. No real patient
information is included, and repository results cannot establish clinical
effectiveness. This project must not be used to support clinical or operational
decisions.
