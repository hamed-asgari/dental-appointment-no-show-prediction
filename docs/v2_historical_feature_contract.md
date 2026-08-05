# Version 2 Historical Feature Contract

## Status

**Frozen before historical-feature implementation and recovered model training**

This contract defines the exact prediction-time-safe feature boundary for the
Version 2 recovery of the Dental Appointment No-show Prediction project.

It supplements:

- `docs/v2_data_generation_and_evaluation_policy.md`; and
- `docs/v2_synthetic_generator_contract.md`.

The Version 1 analytical pipeline remains an auditable checkpoint and must not
be rewritten to serve Version 2.

## Frozen benchmark identity

The feature implementation must accept only the approved Version 2 raw
benchmark:

```text
configuration SHA-256:
aafb631abc615f61f5f4efda9650ab1efac5664c50d62ea7e399a69c97fbaf50

manifest SHA-256:
7702fa5fa0638c52dd0598e28f35f678fb5d61a886faadf9b38a6e292fdcd561

dataset fingerprint:
d9fdfa1a93091fd15bc34a62d655aef313966e2603d901350a6bd969b4e3c1bf
```

Approved raw-file hashes:

```text
appointments.csv:
00d759e69fa51eb5250fafb07e844a7c7ba0cb16dec2b80de47ce78092a162ba

patients.csv:
37df61e47d4060f1e92af49de224228e02451c9eb72e2820f037285ffa9a8ad6

dentists.csv:
22426232d2fa4051ebe1484b5b495ed5afc2d51e2f5c3f19c654aa1f61cad5e8
```

A hash mismatch must stop construction. It must not be treated as a warning.

## Version 1 isolation

The existing Version 1 builder under `src/data/build_dataset.py` and its
protected outputs remain unchanged.

Version 2 work will use separate modules and outputs. Initial implementation
locations are:

```text
src/features/asof_history.py
src/data/build_v2_dataset.py
data/processed/v2/
```

No Version 2 change may alter Version 1 row counts, hashes, splits, or declared
results.

## Prediction time

For every appointment:

```text
prediction_time = scheduled_start_at - 24 hours
```

All timestamp comparisons use timezone-naive `datetime64[ns]` values because
that is the frozen synthetic-data convention.

An appointment is eligible for scoring only when:

```text
booked_at <= prediction_time
```

and it was not already cancelled or rescheduled by prediction time:

```text
not (
    status in {"cancelled", "rescheduled"}
    and status_updated_at <= prediction_time
)
```

Appointments cancelled or rescheduled after prediction time remain eligible
because they were still active when the score would have been produced.

## Evaluation-window boundary note

The raw appointment range begins on `2023-01-01`, but the 24-hour prediction
horizon creates ten eligible appointments whose `prediction_time` is on
`2022-12-31`.

Those ten rows are classified as:

```text
context_only
```

They are excluded from reported warm-up, development, calibration,
policy-selection, and final-test populations. Their outcomes may contribute to
later historical features only after those outcomes become strictly available.

This context-only rule does not change the frozen evaluation windows.

## Historical source population

Historical features are constructed from the complete approved appointment
event table, not only from appointments that were eligible at their own
prediction time.

A historical appointment may contribute when:

- its final status is one of `completed`, `no_show`, `cancelled`, or
  `rescheduled`;
- its required source fields are valid; and
- its outcome satisfies the strict availability rule below.

This reflects the information the clinic would possess from all known prior
appointment outcomes.

## Strict availability rule

A historical appointment is available to a current row only when:

```text
historical_status_updated_at < current_prediction_time
```

The comparison is intentionally strict.

An event with:

```text
historical_status_updated_at == current_prediction_time
```

must not contribute.

A deterministic appointment-ID tie breaker may be used only to stabilize
sorting. It must never make an equal-time or future event available.

## Sequential evaluation rule

Historical features are operationally sequential.

For a later row in the protected final-test period, an earlier final-test
appointment may contribute only when its status had already become known before
the later row's prediction time.

This is allowed because it reproduces prospective clinic operation. The
current row's target must never contribute to its own features.

The protected target accessor remains separate from feature construction.
Final-test probabilities must be written before final-test targets are loaded
or joined for evaluation.

## Attendance-opportunity denominator

No-show rates use only appointments for which attendance was possible:

```text
attendance statuses = {"completed", "no_show"}
```

Therefore:

```text
attendance_count = completed_count + no_show_count
```

Cancelled and rescheduled appointments do not enter a no-show-rate denominator.
Their counts remain separate features.

## Frozen smoothing rule

All no-show rates use the predeclared Beta prior:

```text
alpha = 1
beta = 9
prior mean = 0.10
prior strength = 10
```

When a rate is supported, it is calculated as:

```text
(no_show_count + 1) / (attendance_count + 10)
```

These constants are governance choices frozen before recovered-model
evaluation. They must not be tuned in response to validation or test metrics.

### Patient rate

The patient rate always uses the formula above. With no prior attendance
opportunities, it equals the prior mean `0.10`.

### Aggregate rates

Dentist, visit-type, and weekday-hour rates require:

```text
minimum attendance support = 10
```

Below ten prior attendance opportunities:

- the rate equals `0.10`; and
- the corresponding support flag is `False`.

At ten or more prior attendance opportunities:

- the Beta-smoothed group rate is emitted; and
- the support flag is `True`.

The prior attendance count is also emitted so model preprocessing can preserve
support information.

## Approved current-appointment features

These values must be known at prediction time:

| Feature | Type | Rule |
|---|---:|---|
| `planned_duration_min` | `int16` | Planned duration from the active booking |
| `visit_type` | `string` | Planned visit category |
| `booking_channel` | `string` | Booking channel known at booking |
| `booking_lead_time_hours` | `float64` | `(scheduled_start_at - booked_at)` in hours |
| `scheduled_weekday` | `int8` | Monday `0` through Sunday `6` |
| `scheduled_hour` | `int8` | Scheduled start hour |
| `scheduled_month` | `int8` | Calendar month `1` through `12` |
| `approximate_age_at_prediction` | `int16` | Prediction year minus birth year |
| `patient_registration_tenure_days` | `int32` | Whole days registered by prediction time |
| `dentist_tenure_days` | `int32` | Whole days since dentist start date |
| `reminder_sent_by_prediction_time` | `bool` | `reminder_sent_at <= prediction_time` |

`reminder_sent` without timestamp validation is not an approved feature.
A reminder sent after prediction time must produce
`reminder_sent_by_prediction_time = False`.

## Approved patient-history features

| Feature | Type | Cold-start value |
|---|---:|---:|
| `patient_history_available` | `bool` | `False` |
| `patient_completed_history_available` | `bool` | `False` |
| `patient_prior_known_appointment_count` | `int32` | `0` |
| `patient_prior_attendance_count` | `int32` | `0` |
| `patient_prior_completed_count` | `int32` | `0` |
| `patient_prior_no_show_count` | `int32` | `0` |
| `patient_prior_cancelled_count` | `int32` | `0` |
| `patient_prior_rescheduled_count` | `int32` | `0` |
| `patient_prior_no_show_rate_smoothed` | `float64` | `0.10` |
| `patient_days_since_last_known_status_update` | `float64` | `0.0` |
| `patient_days_since_last_completed_appointment` | `float64` | `0.0` |
| `patient_mean_prior_booking_lead_days` | `float64` | `0.0` |

Definitions:

```text
patient_history_available =
    patient_prior_known_appointment_count > 0

patient_completed_history_available =
    patient_prior_completed_count > 0
```

`patient_days_since_last_known_status_update` is measured from the latest
strictly available `status_updated_at`.

`patient_days_since_last_completed_appointment` is measured from the scheduled
start of the latest completed appointment whose status is strictly available.

`patient_mean_prior_booking_lead_days` uses all strictly available prior
appointments, including cancellations and reschedules.

Zero-valued recency and mean features are valid only together with their
availability indicators. Cold-start rows must not be dropped.

## Approved aggregate-history features

| Feature | Type | Unsupported value |
|---|---:|---:|
| `dentist_prior_attendance_count` | `int32` | `0` |
| `dentist_no_show_rate_supported` | `bool` | `False` |
| `dentist_prior_no_show_rate_smoothed` | `float64` | `0.10` |
| `visit_type_prior_attendance_count` | `int32` | `0` |
| `visit_type_no_show_rate_supported` | `bool` | `False` |
| `visit_type_prior_no_show_rate_smoothed` | `float64` | `0.10` |
| `weekday_hour_prior_attendance_count` | `int32` | `0` |
| `weekday_hour_no_show_rate_supported` | `bool` | `False` |
| `weekday_hour_prior_no_show_rate_smoothed` | `float64` | `0.10` |

The weekday-hour grouping key is:

```text
(scheduled_weekday, scheduled_hour)
```

Exact dentist identifiers remain audit-only. A dentist ID may be used
internally as a grouping key but must not appear in the model-feature
allowlist.

## Deterministic as-of algorithm

The reference implementation must behave as follows:

1. Validate frozen raw hashes and schemas.
2. Build eligible scoring rows.
3. Sort scoring rows by:
   ```text
   prediction_time, appointment_id
   ```
4. Sort historical events by:
   ```text
   status_updated_at, appointment_id
   ```
5. For each distinct prediction time, add all events satisfying:
   ```text
   status_updated_at < prediction_time
   ```
6. Compute every row at that prediction time from the same historical state.
7. Add no event with an equal timestamp until later prediction times.
8. Return rows sorted by:
   ```text
   prediction_time, appointment_id
   ```

Input row order must not affect outputs.

## Prohibited predictors

The following remain prohibited as direct model inputs:

- `appointment_id`;
- `patient_id`;
- `dentist_id`;
- current or future `status`;
- current or future `status_updated_at`;
- `check_in_at`;
- `chair_start_at`;
- `chair_end_at`;
- `checkout_at`;
- `status_change_reason`;
- raw `reminder_sent`;
- raw `reminder_sent_at`;
- hidden synthetic risk effects;
- generated no-show probability;
- any post-outcome field; and
- any exact target or target-derived field.

Identifiers may remain in an audit table or keyed feature artifact only when
the model-feature selector removes them defensively.

## Deferred features

The initial Version 2 implementation does not include:

- exact patient or dentist identifiers;
- categorical previous-outcome labels;
- reschedule-lineage indicators;
- demographic fields beyond approximate age and registration tenure;
- free-text reason fields;
- unsmoothed group rates; or
- features selected after inspecting protected final-test results.

Adding a deferred feature requires a documented contract amendment before
model evaluation with that feature.

## Feature and target separation

The Version 2 builder must support a feature artifact that does not contain the
target.

At minimum, construction must separate:

```text
audit keys
model features
evaluation partition
label-maturity metadata
target
```

The model-feature selector must use an explicit allowlist and return a defensive
copy.

The target must be joined only by a controlled evaluation path.

## Label maturity

At every model-fitting event, a row's target is eligible only when:

```text
status_updated_at < model_fit_time
```

Feature availability and label maturity are separate checks:

- features use history available before each row's prediction time;
- training labels must also be mature before the relevant model-fit time.

## Required leakage and integrity tests

Implementation is incomplete until tests prove all of the following:

1. An event one nanosecond before prediction time is available.
2. An event exactly at prediction time is unavailable.
3. An event after prediction time is unavailable.
4. A current row cannot contribute to its own features.
5. Rows sharing a prediction time use the same pre-time historical state.
6. Shuffling raw input rows does not change output.
7. Mutating a future outcome does not alter any earlier feature row.
8. Mutating an equal-time outcome does not alter that prediction-time batch.
9. Patient status counts reconcile to prior-known appointment count.
10. Attendance count equals completed plus no-show count.
11. Cold-start defaults and availability flags are exact.
12. Beta-smoothed rates reproduce the frozen formula.
13. Aggregate support switches only at ten attendance opportunities.
14. Cancelled and rescheduled rows do not enter attendance denominators.
15. Reminder timing is enforced exactly.
16. Context-only rows are excluded from evaluation windows.
17. IDs and prohibited source columns are absent from model features.
18. Final-test access requires explicit opt-in.
19. A final-test probability vector is written before target access.
20. Version 1 protected outputs and tests remain unchanged.

## Phase R1 implementation sequence

Implementation will proceed in separate reviewed batches:

1. immutable feature constants and schemas;
2. deterministic as-of patient-history engine;
3. deterministic aggregate-history engine;
4. Version 2 cohort and feature-only dataset builder;
5. evaluation partitions and label-maturity metadata;
6. protected target access and processed-data manifest;
7. full leakage, mutation, determinism, and cross-platform tests.

No recovered model may be selected before these batches and their tests are
complete.

## Acceptance criteria

The historical-feature phase is accepted only when:

- this contract was committed before feature implementation;
- the frozen raw benchmark identity is verified;
- Version 1 behavior remains unchanged;
- the feature builder is deterministic;
- every history comparison is strict;
- cold-start behavior is explicit;
- future-mutation tests pass;
- final-test access is gated;
- processed outputs have hashes and a deterministic manifest;
- the complete local test suite passes; and
- GitHub CI passes.
