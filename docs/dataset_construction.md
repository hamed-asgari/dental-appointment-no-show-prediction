# Reproducible Dataset Construction

## Purpose

The dataset-construction pipeline converts three immutable synthetic raw CSV
files into a canonical analytical dataset suitable for later model development.
It implements the repository's approved prediction-time, target, feature,
temporal-split, and label-maturity contracts as deterministic, validated code.

This repository is a portfolio and learning project. It has not been validated
for clinical or operational use.

## Source inputs and integrity

Construction uses exactly three source datasets:

- `data/raw/appointments.csv`
- `data/raw/patients.csv`
- `data/raw/dentists.csv`

The approved SHA-256 hashes are:

| File | SHA-256 |
|---|---|
| `appointments.csv` | `4F3736F78CDA615D1401D3F639B5E29E47781A1AE1C820C1E6F248EAE57A00DF` |
| `patients.csv` | `E416843A80568A91455E5CFF872BBCA5B49BE16F109022D56C687CDF2683CC69` |
| `dentists.csv` | `BF83D1848236E8F5FC8EE5EF3BB21FEC2690F85C3C2F259840C16C271A00AB47` |

The pipeline stops when a required file is missing, a hash differs, an approved
schema or row count does not match, a primary key is null or duplicated, a
required value is missing, a join key is unmatched, or a timestamp does not
match its required format. Raw files are read but never modified.

## Environment

The validated environment uses Python 3.12 in a repository-local `.venv`.
Direct runtime dependencies are recorded in `requirements.txt`, and development
dependencies, including pytest, are recorded in `requirements-dev.txt`.
`requirements.lock.txt` records the complete environment resolved and validated
on Windows with Python 3.12. It may therefore contain Windows-specific packages
and should not be assumed to describe every operating system.

PowerShell setup commands, without requiring environment activation, are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
```

The Windows `py -3.12` launcher command is not available on every system. When
necessary, use another Python 3.12 interpreter to create `.venv`, then continue
with the repository-local interpreter commands.

## Governing methodology

The implementation follows four approved methodological contracts:

- [Prediction time](prediction_time.md) defines when a prediction would be made
  and which appointments are active then.
- [Target definition](target_definition.md) defines the eligible population and
  binary outcome.
- [Feature eligibility](feature_eligibility.md) defines the leakage-controlled
  predictor boundary.
- [Temporal split](temporal_split.md) defines chronological partitions, label
  maturity, and test-set protection.

For each appointment:

`prediction_time = scheduled_start_at - 24 hours`

An appointment is eligible only when:

- `booked_at <= prediction_time`; and
- it is not already cancelled or rescheduled with
  `status_updated_at <= prediction_time`.

Within the eligible cohort:

- `target = 1` for final status `no_show`; and
- `target = 0` for final status `completed`, `cancelled`, or `rescheduled`.

Appointments cancelled or rescheduled after prediction time remain in the
cohort because they were active when the prediction would have been generated.

## Canonical output schema

The output contains exactly 18 columns in the following order. Only rows passing
all construction invariants are serialized.

| Column | Role | Dtype | Predictor allowed | Definition |
|---|---|---|---|---|
| `appointment_id` | Audit identifier | `int64` | No | Unique appointment key retained for traceability. |
| `patient_id` | Audit/join identifier | `int64` | No | Patient key retained for audit and controlled joins. |
| `dentist_id` | Audit/join identifier | `int64` | No | Dentist key retained for audit and controlled joins. |
| `prediction_time` | Decision timestamp | `datetime64[ns]` | No | Scheduled start minus 24 hours. |
| `target` | Binary outcome | `int8` | No | One for `no_show`; zero for other approved final statuses. |
| `split` | Temporal partition | `string` | No | `train`, `validation`, or `test`. |
| `planned_duration_min` | Predictor | `int16` | Yes | Planned appointment duration in minutes. |
| `visit_type` | Predictor | `string` | Yes | Planned visit category. |
| `booking_channel` | Predictor | `string` | Yes | Channel used to create the booking. |
| `booking_lead_time_hours` | Predictor | `float64` | Yes | Unrounded elapsed hours from booking to scheduled start. |
| `scheduled_weekday` | Predictor | `int8` | Yes | Scheduled weekday, Monday 0 through Sunday 6. |
| `scheduled_hour` | Predictor | `int8` | Yes | Scheduled integer hour from 0 through 23. |
| `scheduled_month` | Predictor | `int8` | Yes | Scheduled calendar month from 1 through 12. |
| `approximate_age_at_prediction` | Predictor | `int16` | Yes | Prediction year minus birth year. |
| `patient_registration_tenure_days` | Predictor | `int32` | Yes | Completed elapsed days from registration to prediction time. |
| `dentist_tenure_days` | Predictor | `int32` | Yes | Completed elapsed days from dentist start date to prediction time. |
| `development_fit_eligible` | Label-maturity control | `bool` | No | Label was available for the development fit under the approved split rule. |
| `pretest_fit_eligible` | Label-maturity control | `bool` | No | Label was available for the final pre-test fit under the approved split rule. |

The exact predictor allowlist is:

1. `planned_duration_min`
2. `visit_type`
3. `booking_channel`
4. `booking_lead_time_hours`
5. `scheduled_weekday`
6. `scheduled_hour`
7. `scheduled_month`
8. `approximate_age_at_prediction`
9. `patient_registration_tenure_days`
10. `dentist_tenure_days`

Identifiers are audit-only. The target, split, and maturity flags are not
predictors. Raw construction fields are removed before output. Scheduled
weekday, hour, and month are stored numerically but remain categorical concepts
for later preprocessing.

## Derived-feature definitions

- `booking_lead_time_hours` is the unrounded elapsed seconds from `booked_at` to
  `scheduled_start_at`, divided by 3,600.
- `scheduled_weekday` uses Monday 0 through Sunday 6.
- `scheduled_hour` is the integer scheduled hour from 0 through 23. Minutes are
  intentionally omitted from the initial baseline.
- `scheduled_month` is the integer scheduled month from 1 through 12.
- `approximate_age_at_prediction` is the prediction year minus `birth_year`.
  It is an approximation because the source does not contain a complete date of
  birth.
- `patient_registration_tenure_days` is the floor of elapsed seconds from
  `registered_at` to `prediction_time`, divided by 86,400.
- `dentist_tenure_days` is the floor of elapsed seconds from `start_date` to
  `prediction_time`, divided by 86,400.

Negative derived values are rejected. Booking lead time must also be finite and
at least 24 hours for every eligible appointment.

## Temporal partitions and label maturity

Partition assignment uses exact half-open intervals:

- Train: `prediction_time < 2025-03-01 00:00:00`
- Validation:
  `2025-03-01 00:00:00 <= prediction_time < 2025-08-01 00:00:00`
- Test: `prediction_time >= 2025-08-01 00:00:00`

Partition membership and fitting eligibility are separate. A label is mature
only when:

`status_updated_at < model_fit_time`

The comparison is strict. `development_fit_eligible` is true only for training
rows whose labels were available before `2025-03-01 00:00:00`.
`pretest_fit_eligible` is true only for training or validation rows whose labels
were available before `2025-08-01 00:00:00`.

## Reconciliation

| Population | Rows | Positives | Negatives |
|---|---:|---:|---:|
| Eligible cohort | 6,786 | 820 | 5,966 |
| Train | 3,682 | 434 | 3,248 |
| Validation | 1,541 | 192 | 1,349 |
| Test | 1,563 | 194 | 1,369 |
| Development fit-eligible | 3,670 | 432 | 3,238 |
| Pre-test fit-eligible | 5,223 | 626 | 4,597 |

## Leakage and test-set protection

`FEATURE_COLUMNS` is the only approved predictor allowlist. The feature,
development-row, and test-row selectors return defensive copies. The
development selector unconditionally excludes test rows, and the test selector
raises unless the caller explicitly passes `allow_test=True`.

Exact identifiers are never predictors. Outcome fields, outcome timestamps,
post-event fields, mutable snapshot attributes without historical evidence, and
their structural missingness do not enter canonical predictors. Test rows exist
in the canonical artifact for eventual evaluation but are not exposed by the
default development accessor. Later preprocessing and modeling must fit only on
the rows allowed by the split and label-maturity contracts.

This phase does not implement preprocessing or modeling.

## Path and write safety

Before writing, the pipeline resolves `raw_dir`, output, and manifest paths with
`Path.resolve(strict=False)`. Output and manifest may not resolve to the same
destination, and neither may equal or fall inside the verified raw directory.

The supplied raw directory must contain all three approved files with hashes
matching the normalized approved provenance. Missing, extra, malformed, or false
provenance mappings are rejected before output directories or temporary files
are created.

Parquet and JSON content are first written to temporary files in their
destination directories and then replaced atomically per file. The pair is not
a two-file transaction: a failure between the two replacements can leave a new
Parquet beside a previous or absent manifest. Re-running construction with valid
inputs and destinations recreates the pair.

## Construction command

Run construction with repository defaults:

```powershell
.\.venv\Scripts\python.exe -m src.data.build_dataset
```

Defaults are:

- Input: `data/raw`
- Parquet: `data/processed/analytical_dataset.parquet`
- Manifest: `data/processed/analytical_dataset.manifest.json`

Custom destinations can be supplied explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.data.build_dataset `
  --raw-dir data/raw `
  --output C:\temp\dental-no-show\analytical_dataset.parquet `
  --manifest C:\temp\dental-no-show\analytical_dataset.manifest.json
```

Generated processed Parquet and JSON files are ignored by Git. They should be
rebuilt from the immutable inputs rather than committed.

## Generated manifest

The deterministic JSON manifest records:

- schema version;
- validated raw input hashes;
- canonical and feature-column order;
- prohibited columns;
- prediction horizon;
- timestamp formats and timezone-naive policy;
- validation, test, development-fit, and pre-test-fit boundaries;
- total, target, split, and maturity counts;
- canonical dtypes;
- relevant Python, pandas, NumPy, and PyArrow versions; and
- the SHA-256 hash of the generated Parquet file.

## Validation and tests

Run the automated contract suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The current validated result is **66 tests passed**. The suite covers:

- raw hashes, required schemas, row counts, and key integrity;
- cohort and target construction;
- prediction-time and label-maturity boundary semantics;
- patient and dentist joins;
- feature formulas, schema order, and dtypes;
- temporal split and reconciliation counts;
- deterministic ordering and input non-mutation;
- feature, development, and test-access protection;
- normalized path and raw-directory safety;
- provenance validation and rejection;
- Parquet and manifest serialization; and
- temporary-file cleanup.

No coverage percentage is claimed.

## Reproducibility boundaries

Source timestamps are timezone-naive and interpreted exactly as stored. System
timezone, locale, current date, and randomness do not affect construction.
Output rows are stably sorted by `prediction_time` and then `appointment_id`.

Exact package pins improve reproducibility. Parquet byte identity is expected
only under the pinned, validated environment; semantic dataset invariants are
the cross-run contract outside that environment.

This phase constructs the canonical dataset. It does not perform exploratory
data analysis, preprocessing, model training, calibration, threshold selection,
or final test evaluation.

## Synthetic-data disclaimer

All records are synthetic. No real patient information, real clinical records,
or real healthcare operational data are included. Results from this repository
do not establish clinical effectiveness. The work is educational and
portfolio-oriented and must not be used for clinical or operational decisions.
