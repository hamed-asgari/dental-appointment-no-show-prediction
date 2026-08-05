# Version 2 Synthetic Generator Contract

## Status

The Version 2 generator core is deterministic and configuration-driven. The
generator produces tables in memory only at this checkpoint. Raw CSV files and
their final SHA-256 hashes are not frozen until the later manifest-and-export
checkpoint.

## Public tables

### `patients`

Columns:

- `patient_id`
- `birth_year`
- `sex`
- `city_area`
- `registered_at`
- `insurance_type`
- `referral_source`
- `preferred_contact_channel`
- `patient_status`

Patients enter over multiple calendar years. Only patients marked `active` are
eligible for synthetic appointment assignment. Hidden patient attendance risk
and visit-frequency effects are never exported.

### `dentists`

Columns:

- `dentist_id`
- `dentist_role`
- `engagement_type`
- `start_date`
- `end_date`
- `scheduled_hours_weekly`
- `active`

Dentist assignment respects role compatibility and start/end availability.

### `appointments`

Columns:

- `appointment_id`
- `patient_id`
- `dentist_id`
- `booked_at`
- `scheduled_start_at`
- `planned_duration_min`
- `visit_type`
- `booking_channel`
- `status`
- `status_updated_at`
- `reminder_sent`
- `reminder_sent_at`
- `check_in_at`
- `chair_start_at`
- `chair_end_at`
- `checkout_at`
- `status_change_reason`
- `rescheduled_from_appointment_id`

The clinic is closed on Fridays. Patients must be registered by booking time,
and dentists must be active for the appointment date.

## Reminder timing

`reminder_sent` is true exactly when `reminder_sent_at` is populated. A reminder
may occur either before or after the 24-hour prediction time. Downstream feature
engineering may use reminder information only when:

```text
reminder_sent_at <= prediction_time
```

## Outcomes and timestamps

Supported final statuses are:

- `completed`
- `no_show`
- `cancelled`
- `rescheduled`

Completed appointments contain workflow timestamps. No-show status is recorded
15 minutes after the scheduled start. Cancellation and rescheduling updates
occur after booking and before the appointment.

Non-null reschedule links reference valid earlier rescheduled appointments for
the same patient, with the replacement booking occurring after the original
status update.

## Hidden synthetic effects

The generator uses hidden synthetic patient and dentist effects to create
stable but imperfect longitudinal variation. Hidden effects and internal
probabilities are prohibited from all public tables.

The generator is not tuned to a target model score. A negative predictive result
must remain reportable.

## Reproducibility

All randomness comes from named streams derived from the frozen root seed in:

```text
configs/v2_synthetic_benchmark.json
```

Generating the same configuration twice must return identical tables. Patient
generation is isolated from appointment-count changes because it uses separate
named random streams.

## Current checkpoint

This checkpoint includes:

- frozen public schemas;
- deterministic in-memory generation;
- strict table validation;
- reminder-time availability;
- longitudinal patient entry;
- cold-start and repeat-patient cases;
- valid protected-period coverage;
- generator unit tests.

It does not yet include:

- raw-file export;
- generated-data manifest;
- final raw-file hashes;
- Version 2 analytical-dataset construction;
- historical feature engineering.
