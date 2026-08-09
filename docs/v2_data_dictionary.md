# Version 2 Data Dictionary

## Status

**Phase R1 frozen data dictionary**

This document describes the public Version 2 synthetic raw tables and the
canonical target-free processed feature artifact used by the recovery branch.
It is descriptive documentation for the already frozen schemas and feature
contract; it does not add new predictors or alter the protected evaluation
policy.

## Frozen identities

```text
raw dataset fingerprint:
d9fdfa1a93091fd15bc34a62d655aef313966e2603d901350a6bd969b4e3c1bf

processed dataset SHA-256:
08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53

processed manifest SHA-256:
2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073

processed dataset fingerprint:
0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787
```

The raw benchmark contains 4,000 patients, 7 dentists, and 24,000
appointments. The canonical processed artifact contains 21,755 eligible
scoring rows and 38 columns, including 32 approved model features.

## Prediction and target conventions

For every scored appointment:

```text
prediction_time = scheduled_start_at - 24 hours
```

A scored appointment must have been booked by prediction time and must not have
already been cancelled or rescheduled by prediction time.

Historical features use only events satisfying:

```text
historical_status_updated_at < current_prediction_time
```

The comparison is strict. Equal-time events are unavailable.

The binary outcome, when accessed through a controlled target path, is:

```text
1 = no_show
0 = completed, cancelled, or rescheduled
```

The committed processed artifact does **not** contain the target.

## Raw table: `patients`

Source:

```text
data/raw/v2/patients.csv
```

| Column | Logical type | Description |
|---|---|---|
| `patient_id` | integer | Deterministic synthetic patient identifier and primary key. Audit/grouping key only; prohibited as a direct model predictor. |
| `birth_year` | integer year | Synthetic year of birth. The model uses only `approximate_age_at_prediction`, not the raw year directly. |
| `sex` | categorical string | Synthetic demographic category. Not in the Version 2 model-feature allowlist. |
| `city_area` | categorical string | Synthetic clinic-area category. Not in the Version 2 model-feature allowlist. |
| `registered_at` | timestamp | Time the patient entered the clinic population. Used to derive registration tenure at prediction time. |
| `insurance_type` | categorical string | Synthetic payment/insurance category. Not in the Version 2 model-feature allowlist. |
| `referral_source` | categorical string | Synthetic source of patient referral. Not in the Version 2 model-feature allowlist. |
| `preferred_contact_channel` | categorical string | Synthetic contact preference. Not in the Version 2 model-feature allowlist. |
| `patient_status` | categorical string | Synthetic patient activity status used by generation/validation. Not in the Version 2 model-feature allowlist. |

Hidden patient attendance-risk and visit-frequency effects used by the
generator are never exported.

## Raw table: `dentists`

Source:

```text
data/raw/v2/dentists.csv
```

| Column | Logical type | Description |
|---|---|---|
| `dentist_id` | integer | Deterministic synthetic dentist identifier and primary key. May be used internally for as-of grouping, but is prohibited as a direct model predictor. |
| `dentist_role` | categorical string | Synthetic clinical role used for visit compatibility. Not in the Version 2 model-feature allowlist. |
| `engagement_type` | categorical string | Synthetic employment/engagement category. Not in the Version 2 model-feature allowlist. |
| `start_date` | date/timestamp | Dentist start date. Used to derive `dentist_tenure_days`. |
| `end_date` | optional date/timestamp | End of synthetic dentist availability when present. |
| `scheduled_hours_weekly` | integer | Synthetic scheduled weekly hours. Not in the Version 2 model-feature allowlist. |
| `active` | boolean | Synthetic dentist active-status flag. Not in the Version 2 model-feature allowlist. |

Hidden dentist risk effects are never exported.

## Raw table: `appointments`

Source:

```text
data/raw/v2/appointments.csv
```

| Column | Logical type | Description |
|---|---|---|
| `appointment_id` | integer | Deterministic synthetic appointment identifier and primary key. Used for audit/alignment only, never as a model predictor. |
| `patient_id` | integer | Foreign key to `patients.patient_id`; used for historical grouping only. |
| `dentist_id` | integer | Foreign key to `dentists.dentist_id`; used for historical grouping only. |
| `booked_at` | timestamp | Booking time. Must be at or before prediction time for a row to be score-eligible. |
| `scheduled_start_at` | timestamp | Planned appointment start. Defines prediction time and calendar features. |
| `planned_duration_min` | integer minutes | Planned visit duration; approved current-appointment feature. |
| `visit_type` | categorical string | Planned visit category; approved current-appointment feature and aggregate-history grouping key. |
| `booking_channel` | categorical string | Channel used to make the booking; approved current-appointment feature. |
| `status` | categorical string | Final synthetic outcome: `completed`, `no_show`, `cancelled`, or `rescheduled`. Never a direct predictor. |
| `status_updated_at` | timestamp | Time the final status became known. Governs strict history availability and label maturity. Never a direct predictor. |
| `reminder_sent` | boolean | Final reminder indicator. Raw value is prohibited as a predictor because it does not itself prove prediction-time availability. |
| `reminder_sent_at` | optional timestamp | Reminder timestamp. The derived reminder feature is true only when this time is at or before prediction time. |
| `check_in_at` | optional timestamp | Post-arrival workflow timestamp for completed appointments; prohibited as a predictor. |
| `chair_start_at` | optional timestamp | Post-arrival workflow timestamp; prohibited as a predictor. |
| `chair_end_at` | optional timestamp | Post-treatment workflow timestamp; prohibited as a predictor. |
| `checkout_at` | optional timestamp | Post-treatment workflow timestamp; prohibited as a predictor. |
| `status_change_reason` | optional string | Synthetic reason for cancellation/rescheduling. Prohibited as a predictor. |
| `rescheduled_from_appointment_id` | optional integer | Link to an earlier rescheduled appointment for the same patient. Not in the Version 2 model-feature allowlist. |

The public raw appointment table never exports generated no-show probability or
hidden latent risk effects.

## Canonical processed artifact

Source:

```text
data/processed/v2/v2_feature_dataset.csv
```

The artifact is target-free. Its 38 columns are divided into four groups:

- 4 audit/alignment columns;
- 32 approved model features;
- 1 evaluation-partition column; and
- 1 label-maturity metadata column.

### Audit and evaluation metadata

| Column | dtype | Model input? | Description |
|---|---:|---:|---|
| `appointment_id` | `int64` | No | Audit/alignment identifier. |
| `patient_id` | `int64` | No | Audit/alignment identifier. |
| `dentist_id` | `int64` | No | Audit/alignment identifier. |
| `prediction_time` | `datetime64[ns]` | No | Timestamp exactly 24 hours before scheduled start. |
| `evaluation_partition` | `string` | No | Frozen chronological partition. |
| `label_available_at` | `datetime64[ns]` | No | Time the appointment outcome became mature; training access requires it to be strictly before model-fit time. |

### Current-appointment model features

| Feature | dtype | Definition |
|---|---:|---|
| `planned_duration_min` | `int16` | Planned appointment duration in minutes. |
| `visit_type` | `string` | Planned visit category. |
| `booking_channel` | `string` | Booking channel known at booking time. |
| `booking_lead_time_hours` | `float64` | Hours from booking to scheduled start. |
| `scheduled_weekday` | `int8` | Scheduled weekday, Monday `0` through Sunday `6`. |
| `scheduled_hour` | `int8` | Scheduled start hour. |
| `scheduled_month` | `int8` | Scheduled calendar month `1` through `12`. |
| `approximate_age_at_prediction` | `int16` | Prediction year minus synthetic birth year. |
| `patient_registration_tenure_days` | `int32` | Whole days from patient registration to prediction time. |
| `dentist_tenure_days` | `int32` | Whole days from dentist start date to prediction time. |
| `reminder_sent_by_prediction_time` | `bool` | True only when `reminder_sent_at <= prediction_time`. |

### Patient-history model features

Every history event must satisfy
`status_updated_at < current_prediction_time`.

| Feature | dtype | Definition / cold start |
|---|---:|---|
| `patient_history_available` | `bool` | True when at least one prior final-status event is strictly available; otherwise `False`. |
| `patient_completed_history_available` | `bool` | True when at least one prior completed appointment is strictly available; otherwise `False`. |
| `patient_prior_known_appointment_count` | `int32` | Count of all strictly available prior final-status appointments; cold start `0`. |
| `patient_prior_attendance_count` | `int32` | Prior completed plus no-show count; cold start `0`. |
| `patient_prior_completed_count` | `int32` | Strictly available prior completed count; cold start `0`. |
| `patient_prior_no_show_count` | `int32` | Strictly available prior no-show count; cold start `0`. |
| `patient_prior_cancelled_count` | `int32` | Strictly available prior cancellation count; cold start `0`. |
| `patient_prior_rescheduled_count` | `int32` | Strictly available prior reschedule count; cold start `0`. |
| `patient_prior_no_show_rate_smoothed` | `float64` | `(prior_no_show + 1) / (prior_attendance + 10)`; cold-start value `0.10`. |
| `patient_days_since_last_known_status_update` | `float64` | Days since latest strictly available status update; cold start `0.0` with availability flag false. |
| `patient_days_since_last_completed_appointment` | `float64` | Days since scheduled start of latest strictly available completed appointment; cold start `0.0` with completed-history flag false. |
| `patient_mean_prior_booking_lead_days` | `float64` | Mean booking lead in days across all strictly available prior appointments; cold start `0.0`. |

Cancelled and rescheduled appointments count as known history but never enter
the no-show-rate attendance denominator.

### Aggregate-history model features

All aggregate no-show rates use the frozen Beta prior `alpha=1`, `beta=9`
(prior mean `0.10`) and require at least 10 prior attendance opportunities.
Below support 10, the support flag is false and the emitted rate remains
`0.10`.

| Feature | dtype | Definition |
|---|---:|---|
| `dentist_prior_attendance_count` | `int32` | Strictly available prior completed plus no-show appointments for the dentist. |
| `dentist_no_show_rate_supported` | `bool` | True when dentist prior attendance count is at least 10. |
| `dentist_prior_no_show_rate_smoothed` | `float64` | Supported Beta-smoothed prior dentist no-show rate, else `0.10`. |
| `visit_type_prior_attendance_count` | `int32` | Strictly available prior completed plus no-show appointments for the visit type. |
| `visit_type_no_show_rate_supported` | `bool` | True when visit-type prior attendance count is at least 10. |
| `visit_type_prior_no_show_rate_smoothed` | `float64` | Supported Beta-smoothed prior visit-type no-show rate, else `0.10`. |
| `weekday_hour_prior_attendance_count` | `int32` | Strictly available prior completed plus no-show appointments for the `(scheduled_weekday, scheduled_hour)` group. |
| `weekday_hour_no_show_rate_supported` | `bool` | True when weekday-hour prior attendance count is at least 10. |
| `weekday_hour_prior_no_show_rate_smoothed` | `float64` | Supported Beta-smoothed prior weekday-hour no-show rate, else `0.10`. |

## Frozen evaluation partitions

Partition assignment is based on `prediction_time`.

| Partition | Window / role | Rows |
|---|---|---:|
| `context_only` | Eligible prediction times before `2023-01-01`; history only, excluded from reported evaluation | 10 |
| `warmup` | `2023-01-01 <= prediction_time < 2024-01-01`; history only | 4,324 |
| `development_fit` | `2024-01-01 <= prediction_time < 2025-01-01` | 4,467 |
| `fold_1_validation` | `2025-01-01 <= prediction_time < 2025-07-01` | 2,150 |
| `fold_2_validation` | `2025-07-01 <= prediction_time < 2026-01-01` | 2,231 |
| `fold_3_validation` | `2026-01-01 <= prediction_time < 2026-07-01` | 2,086 |
| `calibration` | `2026-07-01 <= prediction_time < 2026-10-01` | 1,081 |
| `policy_selection` | `2026-10-01 <= prediction_time < 2027-01-01` | 1,063 |
| `final_test` | `2027-01-01 <= prediction_time < 2028-01-01`; protected | 4,343 |

## Target access boundary

Development labels may be requested only for explicitly allowed non-test
partitions and only when:

```text
label_available_at < model_fit_time
```

The public maturity path rejects any request containing `final_test`.

A successful final-test target access requires explicit `allow_test=True` plus
a complete, ordered, finite probability vector in `[0,1]` that is written and
SHA-256 sealed before raw final-test statuses are loaded or joined.

At the Phase R1 closeout checkpoint:

```text
target_included = false
final_test_target_accessed = false
```

No protected 2027 test metric has been inspected.

## Prohibited direct model inputs

The Version 2 selector defensively excludes exact identifiers, final outcome
columns, post-outcome timestamps, raw reminder fields, hidden synthetic effects,
generated target probability, target itself, and any target-derived field.
The authoritative allowlist is `V2_MODEL_FEATURE_COLUMNS` in
`src/features/schema.py`.

## Interpretation boundary

All data are synthetic. The dictionary describes the technical benchmark and
does not imply clinical validity, real-patient prevalence, or deployment
fitness.
