# Dental Appointment No-show Prediction

> **Synthetic-data and use disclaimer**
>
> All records in this repository are fully synthetic. They do not represent real
> patients, clinical records, or healthcare operations. This educational and
> portfolio-oriented project is not validated for clinical or operational use.

## Overview

This repository implements a reproducible, leakage-controlled foundation for
studying dental appointment no-show prediction. It includes immutable synthetic
raw inputs, approved prediction-time and target contracts, leakage-aware feature
eligibility, a chronological temporal split, canonical analytical-dataset
construction, automated contract tests, and deterministic Parquet and manifest
generation.

The repository does not yet contain a trained model or performance result.

## Current project status

Reproducible dataset construction is complete. The implemented pipeline verifies
raw-file integrity, reconstructs the eligible cohort, derives the approved
features, assigns chronological partitions and label-maturity flags, validates
the exact canonical schema, and writes generated outputs safely.

Exploratory data analysis and modeling have not started. Preprocessing,
calibration, threshold selection, and final test evaluation also remain outside
the current implementation.

## Repository structure

```text
.
|-- data/
|   |-- raw/                         # Immutable synthetic source files
|   |-- interim/                     # Reserved for later intermediate products
|   `-- processed/                   # Generated Parquet and JSON outputs
|-- docs/
|   |-- README.md                    # Methodology index
|   `-- dataset_construction.md      # Reproducible build contract
|-- src/
|   `-- data/
|       `-- build_dataset.py         # Canonical dataset pipeline and CLI
|-- tests/
|   `-- test_dataset_construction.py # Automated contract and safety tests
|-- requirements.txt
|-- requirements-dev.txt
`-- requirements.lock.txt
```

Generated files under `data/processed/` are ignored by Git and should be rebuilt
from the immutable raw inputs.

## Quick start

Use Python 3.12 and run these commands from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.data.build_dataset
```

The Windows `py -3.12` launcher command is not available everywhere. Another
Python 3.12 interpreter may be used to create `.venv` when necessary; subsequent
commands can still use the repository-local interpreter directly.

## Outputs

The default construction command generates:

- `data/processed/analytical_dataset.parquet`
- `data/processed/analytical_dataset.manifest.json`

Both files are generated artifacts and ignored by Git. The manifest records the
validated source hashes, schema, dtypes, counts, boundaries, package versions,
and generated Parquet hash.

## Methodology

The [documentation index](docs/README.md) links the approved contracts in order:
data intake, prediction time, target definition, feature eligibility, temporal
split, and reproducible dataset construction.

The construction pipeline uses only the ten approved prediction-time features.
Identifiers, target and split fields, maturity flags, final outcomes, and
post-event fields are excluded from predictors.

## Scope boundary

The repository currently constructs and validates the analytical dataset. It
does not yet implement exploratory analysis, preprocessing, model training,
probability calibration, operational threshold selection, or final test-set
evaluation.

## Disclaimer

All source records and derived outputs are synthetic. No real patient
information is included, and repository results cannot establish clinical
effectiveness. This project must not be used to support clinical or operational
decisions.
