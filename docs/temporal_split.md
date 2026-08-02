# Temporal Train, Validation, and Test Split

## Purpose

This document defines the chronological train, validation, and test split for dental appointment no-show prediction. It establishes the canonical time variable, approved calendar boundaries, label-maturity rule, test-set protection policy, interpretation of patient overlap, and embargo decision.

The split is a methodological boundary for later development. It does not initiate model training or establish that any model or feature is effective, calibrated, fair, or suitable for real-world use.

## Governing Time Variable

The canonical split variable is the appointment-level prediction timestamp:

`prediction_time = scheduled_start_at - 24 hours`

All timestamps in the current synthetic source data are timezone-naive and are interpreted exactly as stored for this split. A real operational implementation must define the clinic timezone and daylight-saving-time policy before applying calendar boundaries.

The eligible cohort contains appointments that were booked by `prediction_time` and had not already been cancelled or rescheduled by `prediction_time`. Within that cohort:

- `target = 1` when final `status == "no_show"`; and
- `target = 0` when final `status` is `completed`, `cancelled`, or `rescheduled`.

The verified cohort contains 6,786 appointments: 820 positives and 5,966 negatives, for 12.08% prevalence. Partition assignment is based only on `prediction_time`. Whether a row's label is available for fitting at a particular time is governed separately by `status_updated_at`.

## Approved Chronological Boundaries

The approved split uses half-open chronological intervals:

| Partition | `prediction_time` interval | Calendar period | Rows | Positives | Prevalence |
|---|---|---|---:|---:|---:|
| Training | `prediction_time < 2025-03-01 00:00:00` | March 2024 through February 2025 | 3,682 | 434 | 11.79% |
| Validation | `2025-03-01 00:00:00 <= prediction_time < 2025-08-01 00:00:00` | March 2025 through July 2025 | 1,541 | 192 | 12.46% |
| Test | `prediction_time >= 2025-08-01 00:00:00` | August 2025 through December 2025 | 1,563 | 194 | 12.41% |

The nominal counts reconcile exactly:

- rows: `3,682 + 1,541 + 1,563 = 6,786`;
- positives: `434 + 192 + 194 = 820`; and
- observed allocation: approximately 54.3% training, 22.7% validation, and 23.0% test.

Every eligible row belongs to exactly one partition. The intervals do not overlap, and the test partition contains the most recent observations.

## Selection Rationale

This split was selected because it provides:

- twelve months of training history;
- balanced five-month validation and test windows;
- substantially more than 100 positive cases in both validation and test;
- stable outcome prevalence across all three periods;
- complete training support for the approved baseline categorical and calendar levels;
- no observed validation or test category with fewer than 20 appointments under the inspection definition;
- a genuinely future test period containing the most recent observations; and
- broader future-period representation than candidates with three-month test windows.

Exact target proportions were not prioritized over clean calendar boundaries, category support, positive counts, and sufficiently broad future evaluation periods.

Random splitting is not the primary evaluation design. A random split could place later appointments into training and would not represent evaluation on future appointments. Primary development and reporting must preserve chronological ordering.

## Label-Maturity Rule

Split assignment and fitting eligibility are separate decisions:

- `prediction_time` assigns a row to training, validation, or test; and
- `status_updated_at` establishes whether its final label was available when a model could have been fitted.

A row may be used for model fitting only when:

`status_updated_at < model_fit_time`

| Fit purpose | `model_fit_time` | Nominal fitting pool | Nominal rows / positives | Excluded immature labels | Excluded final outcomes | Label-mature rows / positives / negatives | Mature prevalence |
|---|---|---|---:|---:|---|---:|---:|
| Development fit evaluated on validation | `2025-03-01 00:00:00` | Training partition | 3,682 / 434 | 12 | 10 `completed`; 2 `no_show` | 3,670 / 432 / 3,238 | 11.77% |
| Final pre-test fit | `2025-08-01 00:00:00` | Combined training and validation partitions | 5,223 / 626 | 0 | None | 5,223 / 626 / 4,597 | 11.99% |

For the validation model-fit boundary, the 12 excluded rows retain their nominal training assignment; they are omitted only from that fitting event because their final statuses were first recorded from `2025-03-01 10:13:00` through `2025-03-01 18:59:00`. They must not be moved into validation.

For the final pre-test fit, every row in the combined training and validation pool satisfies the strict as-of rule at `2025-08-01 00:00:00`.

The completed synthetic dataset contains final outcomes for all eligible appointments. A real pipeline must nevertheless apply the same as-of label cutoff before fitting or evaluation and must verify outcome completeness explicitly. The observed timing in this dataset is not a universal healthcare label-maturation standard.

## Test-Set Protection

Once this split contract is finalized, the test period is frozen.

- Test features or outcomes must not guide feature eligibility, preprocessing choices, encoding, imputation, hyperparameter selection, model selection, calibration-method selection, or decision-threshold selection.
- Model-development decisions must use training and validation only.
- Test evaluation occurs once, after the modeling and calibration workflow is finalized.
- Preprocessing, feature selection, model fitting, and calibration fitting must not include test rows.
- Any later redesign prompted by test performance must be disclosed and must use a newly defined untouched evaluation period.

## Patient Overlap and Cold-Start Evaluation

The approved split has the following observed appointment-level composition:

| Measure | Validation | Test |
|---|---:|---:|
| Appointments from patients unseen in earlier partitions | 257 of 1,541 (16.68%) | 112 of 1,563 (7.17%) |
| First-observed patient appointments | 181 of 1,541 (11.75%) | 74 of 1,563 (4.73%) |
| Repeat-patient appointments | 1,360 of 1,541 (88.25%) | 1,489 of 1,563 (95.27%) |

For validation, "unseen" means absent from training. For test, it means absent from both training and validation. A first-observed appointment has no eligible appointment for the same patient at an earlier `prediction_time`; a repeat appointment has at least one earlier eligible appointment.

Patient overlap across chronological periods is not automatically leakage. Repeat patients are expected in the intended future-period setting, while the unseen-patient appointments preserve a measurable cold-start component. The primary evaluation does not use a patient group split, and exact `patient_id` remains excluded from baseline predictors.

Overall performance must later be supplemented by secondary diagnostics for first-observed and repeat-patient appointments. Any future patient-history feature will require strict as-of and fold-safe computation and a renewed split review.

## Dentist Coverage

All dentists appearing in validation or test also appear in training. Observed appointment counts are:

| `dentist_id` | Training | Validation | Test |
|---:|---:|---:|---:|
| 1 | 1,193 | 498 | 559 |
| 2 | 1,007 | 450 | 420 |
| 3 | 327 | 156 | 113 |
| 4 | 264 | 85 | 118 |
| 5 | 317 | 134 | 150 |
| 6 | 457 | 218 | 203 |
| 7 | 117 | 0 | 0 |

Dentist 7 appears only in training, so this split does not evaluate a completely unseen future dentist. Exact `dentist_id` remains excluded from baseline predictors. Provider-specific results, if later reported, are diagnostic and may be imprecise because of limited subgroup counts.

## Baseline Category Coverage

Training contains every category level observed in validation and test for the approved initial baseline fields and calendar derivations. No observed validation or test level has fewer than 20 appointments.

| Field or derived category | Validation levels | Minimum validation level count | Test levels | Minimum test level count | All later levels represented in training? |
|---|---:|---:|---:|---:|---|
| `visit_type` | 6 | 123 | 6 | 123 | Yes |
| `booking_channel` | 5 | 60 | 5 | 83 | Yes |
| `planned_duration_min` | 3 | 353 | 3 | 346 | Yes |
| Scheduled weekday | 6 | 238 | 6 | 245 | Yes |
| Scheduled hour | 9 | 133 | 9 | 110 | Yes |
| Scheduled month | 5 | 283 | 5 | 266 | Yes |

These checks establish category support only. They do not prescribe or perform encoding.

## Embargo Decision

No broad independence embargo is required for the initial baseline. Approved baseline features describe the current appointment or dated source events; exact identifiers and historical aggregates are excluded.

Label maturity is handled through the strict `status_updated_at < model_fit_time` rule rather than by inserting a broad temporal gap between partitions. This decision must be reviewed if later work introduces patient- or dentist-history aggregates, target encoding, likelihood encoding, provider effects, or other temporally dependent features.

## Implementation Requirements

Later implementation must:

- reconstruct the eligible cohort exactly from `booked_at`, `status`, and `status_updated_at` using the documented prediction-time rules;
- compute `prediction_time` before assigning partitions;
- assign partitions with the half-open intervals in this contract;
- assert that every eligible row belongs to exactly one partition;
- assert that partitions have no chronological overlap;
- apply `status_updated_at < model_fit_time` at each fitting event;
- preserve `appointment_id` only for audit and row traceability, never as a predictor;
- prevent test rows from entering preprocessing fit, feature selection, model fitting, or calibration fitting;
- store split assignment reproducibly rather than reselecting boundaries; and
- report nominal partition counts and label-mature fitting counts separately.

The resulting implementation must reproduce 6,786 assigned rows and 820 positives before any model work begins.

## Deferred Split Reviews

The split design must be reconsidered if later work introduces:

- patient-history features;
- dentist-history features;
- target or likelihood encoding;
- exact or grouped identity effects;
- additional datasets with delayed outcomes;
- a changed prediction horizon;
- rolling-origin or repeated temporal validation; or
- external real-world data.

Any revised design must preserve chronological evaluation and define a new untouched test policy where necessary.

## Data and Use Disclaimer

The source data are fully synthetic. The observed counts, timing relationships, patient overlap, and provider coverage describe this dataset only. This split has not been validated for real-world clinical or operational use and must not be described as clinically validated or used to support clinical decisions.
