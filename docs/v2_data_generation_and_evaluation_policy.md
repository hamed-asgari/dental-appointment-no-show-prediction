# Version 2.0 Synthetic Data and Evaluation Policy

## Status

**Frozen before Version 2.0 model development**

This policy defines the synthetic-data redesign, prediction-time contract,
historical-feature boundary, and protected evaluation schedule for the Version
2.0 recovery of the Dental Appointment No-show Prediction project.

The original Project 1 repository remains frozen at `v1.0.0`. Version 2 work
must not modify that repository.

## Why a new Version 2 benchmark is required

The audited Project 1 generator is reproducible, but it cannot be used directly
as the Version 2 predictive benchmark.

The audited source generator has these fixed properties:

- a module-level NumPy random generator with seed `42`;
- patient registration dates restricted to calendar year 2023;
- appointment dates hard-coded from `2024-03-01` through `2025-12-31`;
- appointment identifiers regenerated from `1` after chronological sorting;
- no CLI parameters for appointment start date, appointment end date, or seed;
- a no-show probability based only on:
  - whether a reminder was sent; and
  - whether booking lead time exceeded 30 days.

The source data do not contain a reminder timestamp, so the final
`reminder_sent` value cannot be proven to exist at the 24-hour prediction time.
It must not be used as a predictor under the current leakage contract.

Patient identity, dentist identity, and historical attendance are absent from
the source no-show formula. Therefore, merely extending the original generator
into later calendar dates would not create a principled longitudinal benchmark
for historical no-show features.

## Version 2 decision

Version 2 will use a **Project 2-owned longitudinal synthetic benchmark** that
is derived from the documented Project 1 schema but implemented, versioned, and
validated entirely inside the Project 2 repository.

This is not presented as external validation or real-world evidence. It is a
fully synthetic technical benchmark designed to demonstrate:

- prediction-time-safe longitudinal feature engineering;
- chronological model selection;
- probability calibration;
- operational threshold analysis;
- model interpretation;
- reproducible inference; and
- an evidence-appropriate Streamlit application.

The original Version 1 raw files and results remain preserved as the audited
negative-result checkpoint.

## Generator governance

### Generator ownership

The new generator will live under:

```text
src/synthetic/
```

The original Project 1 generator will not be edited.

### Version and configuration

The first frozen benchmark configuration will use:

```text
generator_version: 2.0.0
root_seed: 20260805
patient_count: 4000
appointment_count: 24000
appointment_start: 2023-01-01
appointment_end: 2027-12-31
prediction_horizon_hours: 24
```

The configuration must be stored in a machine-readable file and included in the
generated-data manifest.

### Randomness

The generator must not use one mutable module-level random stream for every
table and process. It must derive named deterministic random streams from the
root seed, including separate streams for at least:

- patient attributes and registration;
- dentist attributes and availability;
- appointment scheduling;
- reminder timing;
- latent risk effects;
- appointment outcomes; and
- post-outcome timestamps.

Changing one generator component should not silently alter all unrelated
components.

### Identifier policy

- identifiers must be unique, deterministic, and monotonic within their table;
- appointment identifiers must not restart within a combined dataset;
- reschedule links must reference valid appointment identifiers;
- generated foreign keys must pass explicit integrity tests.

### Data versions and hashes

Generated raw files must receive:

- schema version;
- generator version;
- configuration hash;
- SHA-256 file hashes;
- row counts;
- date ranges;
- status counts; and
- generation-environment metadata.

After hashes are frozen, changing generator logic, coefficients, configuration,
or raw files creates a new data version and invalidates the previous protected
test policy.

## Synthetic longitudinal design

### Patient flow

Patients must enter the clinic over time rather than all being registered in a
single historical year.

At every appointment:

- the patient must already be registered by booking time;
- new and repeat patients must both be represented;
- cold-start patients must remain possible in every scored period;
- inactive or unavailable patients must not be selected where prohibited by
  the generator contract.

### Appointment flow

Appointments must be distributed across the full benchmark period and preserve:

- six-day clinic operation;
- realistic booking lead times;
- planned visit categories;
- planned duration;
- booking channel;
- dentist availability;
- cancellation and rescheduling timestamps;
- completed, no-show, cancelled, and rescheduled outcomes.

### Reminder timing

Version 2 must add:

```text
reminder_sent_at
```

A reminder-derived predictor is eligible only when:

```text
reminder_sent_at <= prediction_time
```

A reminder recorded after prediction time must not enter the feature set for
that appointment.

### Latent longitudinal risk

The generator may use hidden synthetic effects to create stable but imperfect
longitudinal signal, including:

- patient-specific attendance propensity;
- weak dentist or clinic-session effects;
- booking lead-time effects;
- visit-type effects;
- time-of-day effects;
- seasonal or calendar drift;
- reminder timing effects; and
- random appointment-level variation.

Hidden effects must never be exported as model predictors.

The generator must not directly encode the final target into an input feature,
and no post-outcome field may influence prediction-time predictors.

### Anti-gaming rule

The generator must be frozen before recovered model results are inspected.

Generator coefficients must not be repeatedly adjusted to improve Average
Precision, ROC-AUC, calibration, or any preferred model. There is no minimum
performance target for accepting the dataset.

If the final benchmark still does not support useful appointment-level ranking,
the correct outcome is a transparent model-evaluation dashboard, not further
generator tuning.

## Prediction-time and target contract

For each appointment:

```text
prediction_time = scheduled_start_at - 24 hours
```

An appointment is eligible only when:

- it was booked by prediction time; and
- it had not already been cancelled or rescheduled by prediction time.

Within the eligible cohort:

- `target = 1` for final status `no_show`;
- `target = 0` for final status `completed`, `cancelled`, or `rescheduled`.

Appointments cancelled or rescheduled after prediction time remain eligible
because they were active when the prediction would have been produced.

A historical outcome is usable only when:

```text
historical_status_updated_at < current_prediction_time
```

The comparison is strict.

## Historical-feature contract

The initial approved historical feature candidates are:

### Patient history

- prior known appointment count;
- prior known completed count;
- prior known no-show count;
- smoothed prior no-show rate;
- prior known cancellation count;
- prior known reschedule count;
- days since previous known appointment;
- days since previous completed appointment;
- mean prior booking lead time;
- patient-history-available flag;
- first-observed-appointment flag.

### Dentist and clinic history

- prior known dentist appointment count;
- smoothed prior dentist no-show rate;
- prior known visit-type appointment count;
- smoothed prior visit-type no-show rate;
- prior known weekday-hour appointment count;
- smoothed prior weekday-hour no-show rate.

Provider and group rates must use explicit smoothing and minimum-support rules.
Exact patient and dentist identifiers remain audit-only and are not direct
predictors.

### Cold-start behavior

Rows with no available history must receive deterministic neutral defaults and
explicit history-availability indicators. They must not be dropped solely for
being cold start.

### Construction rule

Historical features must be generated in chronological order or by an
equivalent validated as-of algorithm. Tests must prove that changing a future
outcome cannot alter an earlier row's features.

## Protected chronological evaluation

Calendar windows are defined by `prediction_time`.

### Warm-up history

```text
2023-01-01 <= prediction_time < 2024-01-01
```

Warm-up rows provide prior history. They are not used for reported model
selection metrics.

### Rolling-origin model-selection folds

#### Fold 1

```text
fit:        2024-01-01 <= prediction_time < 2025-01-01
validation: 2025-01-01 <= prediction_time < 2025-07-01
```

#### Fold 2

```text
fit:        2024-01-01 <= prediction_time < 2025-07-01
validation: 2025-07-01 <= prediction_time < 2026-01-01
```

#### Fold 3

```text
fit:        2024-01-01 <= prediction_time < 2026-01-01
validation: 2026-01-01 <= prediction_time < 2026-07-01
```

At every fitting event, labels must satisfy:

```text
status_updated_at < model_fit_time
```

### Calibration fit

```text
2026-07-01 <= prediction_time < 2026-10-01
```

Calibration candidates may be fitted only after the base estimator and
preprocessing design are frozen from rolling-origin development.

### Operational policy selection

```text
2026-10-01 <= prediction_time < 2027-01-01
```

This period is used for threshold, capacity, and cost-policy selection. It must
not be used to refit the base model or choose new features.

### Final protected test

```text
2027-01-01 <= prediction_time < 2028-01-01
```

Final test labels must not be used for:

- generator revision;
- feature selection;
- preprocessing selection;
- estimator selection;
- hyperparameter selection;
- calibration selection;
- threshold selection; or
- application-behavior selection.

The test accessor must require an explicit opt-in such as `allow_test=True`.
The final test command must create the probability vector before loading or
joining test labels.

## Model-selection summaries

Rolling-origin results must report both:

- each fold separately; and
- an aggregate summary across folds.

Required metrics include:

- Average Precision;
- ROC-AUC;
- Brier score;
- log loss;
- calibration intercept and slope where supported;
- precision and recall at predeclared policy points; and
- sample size and positive count.

No single metric may be used as the sole justification for selection.

## Application decision gate

### Appointment-level risk demonstration

Allowed only when the protected evaluation supports useful ranking and
probability behavior beyond the population-prior baseline.

### Transparent model-evaluation dashboard

Required when individualized prediction remains unsupported.

The application must never conceal a negative result or present synthetic model
outputs as clinically validated risk.

## Phase R1 acceptance criteria

Phase R1 is complete only when:

- this policy is committed before Version 2 model development;
- the Project 1 repository remains unchanged;
- the Version 2 generator configuration is machine-readable;
- the longitudinal generator is deterministic;
- generated raw files and manifests have frozen hashes;
- all key and timestamp integrity tests pass;
- reminder timing is explicitly represented;
- as-of historical features are implemented;
- future-outcome mutation tests pass;
- cold-start behavior is tested;
- split and label-maturity tests pass;
- the protected 2027 test accessor is gated; and
- no protected test metric has been inspected.
