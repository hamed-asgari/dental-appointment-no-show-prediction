# Feature Eligibility and Temporal Leakage Contract

## Purpose

This document defines the temporal feature-eligibility boundary for predicting dental appointment no-shows. It distinguishes information that was historically available at prediction time from information that is suitable for direct modeling. A field can be known at prediction time and still be restricted because it is an identifier, requires a controlled derivation, or presents a generalization concern.

The decisions below apply to the current raw appointment, patient, and dentist datasets. They authorize only the stated uses. They do not establish that an eligible feature will be useful, stable, fair, or retained in a final model.

## Governing Prediction-Time Rule

The prediction timestamp for each appointment is:

`prediction_time = scheduled_start_at - 24 hours`

A predictor must represent information genuinely available at or before this timestamp. The eligible prediction cohort contains only appointments that:

1. were already booked by the prediction timestamp; and
2. had not already been cancelled or rescheduled by the prediction timestamp.

Final outcomes may be used retrospectively to construct the cohort and target, but they must not be used as predictors. Historical availability must be demonstrated using event timing or effective-date evidence; a value in a current snapshot must not automatically be treated as historically available.

## Eligibility Status Definitions

- `eligible_direct_candidate`: A raw field that may be considered directly as an initial model input, subject to later preprocessing and evaluation.
- `eligible_derived_only`: A raw source field that must not be modeled as an unrestricted raw value but may support explicitly defined prediction-time derivations.
- `record_or_join_key_only`: An identifier used only for record integrity, joins, or carefully designed historical aggregation. It must not be directly encoded as a baseline predictor.
- `cohort_or_target_only`: A field used retrospectively to construct the eligible cohort or outcome label. It must not be used as a predictor.
- `prohibited_leakage`: A field containing post-prediction, final-outcome, or structurally outcome-revealing information.
- `deferred_temporal_ambiguity`: A field whose historical value or event timing cannot be established safely from the current data.
- `evaluation_only`: A field reserved initially for subgroup, fairness, or diagnostic evaluation rather than baseline model training.

## Appointment Field Decisions

| Field | Status | Permitted use | Temporal and leakage rationale |
|---|---|---|---|
| `appointment_id` | `record_or_join_key_only` | Record integrity and joins only. | This is a unique appointment identifier. Direct encoding would permit observation-level memorization. |
| `patient_id` | `record_or_join_key_only` | Join to the patient table; possible future as-of history key after separate review. | Direct encoding risks patient memorization and poor generalization. |
| `dentist_id` | `record_or_join_key_only` | Join to the dentist table; possible future grouped-history key after separate review. | Direct provider identity may learn clinician- or clinic-specific effects that do not generalize. |
| `booked_at` | `eligible_derived_only` | Derive booking lead time and confirm that booking occurred by prediction time. | The booking event must occur by the prediction timestamp for cohort eligibility. The unrestricted raw timestamp is not approved. |
| `scheduled_start_at` | `eligible_derived_only` | Construct `prediction_time`; derive scheduled weekday, hour, month, or season. | The scheduled value defines the prediction horizon. The unrestricted raw timestamp is not approved. |
| `planned_duration_min` | `eligible_direct_candidate` | Consider as a raw baseline input, subject to later preprocessing and evaluation. | Planned duration appears to describe the appointment before it occurs. This status does not guarantee usefulness. |
| `visit_type` | `eligible_direct_candidate` | Consider as a categorical baseline input, subject to later encoding and evaluation. | The field appears to describe the planned visit rather than its outcome. |
| `booking_channel` | `eligible_direct_candidate` | Consider as a categorical baseline input, subject to later encoding and evaluation. | The channel is associated with the booking event and appears available when the appointment is created. |
| `status` | `cohort_or_target_only` | Construct the final outcome label and support retrospective cohort accounting. | This is the final appointment outcome and the target source. It must not be a predictor. |
| `status_updated_at` | `cohort_or_target_only` | Reconstruct whether cancellation or rescheduling occurred before or after prediction time; document the synthetic no-show timing convention. | This timestamp is outcome-related. It must not be a predictor. |
| `reminder_sent` | `deferred_temporal_ambiguity` | No predictor use unless event timing is established in future data. | There is no `reminder_sent_at` timestamp, so the flag cannot be placed safely before or after prediction time. |
| `check_in_at` | `prohibited_leakage` | None. | Check-in occurs after prediction and reveals whether an appointment progressed to attendance. |
| `chair_start_at` | `prohibited_leakage` | None. | Chair activity is generated after prediction and reveals appointment progress. |
| `chair_end_at` | `prohibited_leakage` | None. | Chair completion is generated after prediction and reveals appointment progress. |
| `checkout_at` | `prohibited_leakage` | None. | Checkout is generated after prediction and reveals a completed attendance pathway. |
| `status_change_reason` | `prohibited_leakage` | None. | The field is outcome-related, and its value or presence structurally reveals cancellation or rescheduling status. |
| `rescheduled_from_appointment_id` | `record_or_join_key_only` | Prior-record linkage only. | The exact prior appointment identifier must not be modeled. A boolean lineage feature is not approved here and may be considered only after its population timing is formally documented. |

## Patient Field Decisions

| Field | Status | Permitted use | Temporal and leakage rationale |
|---|---|---|---|
| `patient_id` | `record_or_join_key_only` | Join to appointments; possible future as-of history key after separate review. | Direct encoding risks memorizing individual patients. |
| `birth_year` | `eligible_derived_only` | Derive approximate age at prediction time. | Only birth year is available. Exact age cannot be calculated because date of birth is unavailable. The unrestricted raw year is not approved. |
| `sex` | `evaluation_only` | Subgroup, fairness, and diagnostic evaluation only. | It is not approved as a baseline predictor. The current data also lack value-effective history. |
| `city_area` | `deferred_temporal_ambiguity` | None pending historical-value evidence. | The patient table contains one current snapshot per patient and no update timestamp or effective-date history for this value. |
| `registered_at` | `eligible_derived_only` | Derive patient registration tenure at prediction time. | This is a dated historical event and may be used only through the approved as-of tenure derivation. |
| `insurance_type` | `deferred_temporal_ambiguity` | None pending historical-value evidence. | The current snapshot does not establish which insurance value was visible at each historical prediction timestamp. |
| `referral_source` | `deferred_temporal_ambiguity` | None pending documented event semantics and historical availability. | The data do not establish that the stored value is the original registration-time value or that it remained unchanged. |
| `preferred_contact_channel` | `deferred_temporal_ambiguity` | None pending historical-value evidence. | Contact preference may change, and no update timestamp or effective-date history is available. |
| `patient_status` | `deferred_temporal_ambiguity` | None pending historical-value evidence. | This is a current-state field. Its present value must not be assumed to describe status at a historical prediction timestamp. |

The patient table is a current snapshot with one row per patient. Except for explicitly dated or intrinsically historical source information, it does not establish the value visible at each appointment's prediction timestamp.

## Dentist Field Decisions

| Field | Status | Permitted use | Temporal and leakage rationale |
|---|---|---|---|
| `dentist_id` | `record_or_join_key_only` | Join to appointments; possible future grouped-history key after separate review. | Exact provider identity is not approved as a baseline predictor because it may not generalize across clinicians or clinics. |
| `dentist_role` | `deferred_temporal_ambiguity` | None pending effective-date evidence. | The current snapshot does not show whether role changed or which role was visible at each prediction timestamp. |
| `engagement_type` | `deferred_temporal_ambiguity` | None pending effective-date evidence. | Engagement arrangements may change, and no historical versioning is available. |
| `start_date` | `eligible_derived_only` | Derive dentist tenure at prediction time. | This is a dated historical event and may be used only through the approved as-of tenure derivation. |
| `end_date` | `deferred_temporal_ambiguity` | None pending evidence of when the end date became known. | A recorded end date may describe a future event relative to prediction time; the dataset does not record when that information became available. |
| `scheduled_hours_weekly` | `deferred_temporal_ambiguity` | None pending effective-date evidence. | Scheduled hours can change, and the table contains no historical schedule versions. |
| `active` | `deferred_temporal_ambiguity` | None pending effective-date evidence. | This is a current-state indicator and must not be assumed to represent historical active status. |

The dentist table is a current snapshot without complete effective-date history. Present-day role, engagement, schedule, end-date, or active values must not be assumed to describe the value available at prediction time.

## Approved Prediction-Time Derivations

The following are eligible derived candidates. They are approved conceptually within this boundary but are not calculated in this documentation step.

1. `booking_lead_time_hours`
   - Sources: appointments.`booked_at`, appointments.`scheduled_start_at`
   - Definition: elapsed hours from booking to scheduled start.
   - Guard: the appointment must already be booked by `prediction_time`.

2. Scheduled calendar features
   - Source: appointments.`scheduled_start_at`
   - Permitted representations: scheduled weekday, scheduled hour, and scheduled month or season.

3. `approximate_age_at_prediction`
   - Sources: patients.`birth_year`, prediction year
   - Limitation: only year-level age can be derived. Exact age is unavailable because date of birth is not recorded.

4. `patient_registration_tenure`
   - Sources: patients.`registered_at`, `prediction_time`
   - Definition: elapsed time from patient registration to the appointment's prediction timestamp.

5. `dentist_tenure`
   - Sources: dentists.`start_date`, `prediction_time`
   - Definition: elapsed time from the dentist's start date to the appointment's prediction timestamp.

These derivations must use the source values available for the appointment at prediction time. Approval of a derivation does not approve unrestricted use of its raw timestamp or year source.

## Historical Aggregate Requirements

Patient- and dentist-history features are not approved for the initial baseline under this contract. Possible future examples include prior appointment count, prior no-show count or rate, prior cancellation count, days since the previous appointment, and dentist historical no-show rate.

Any future historical aggregate must:

- use only events whose outcomes were known on or before the current appointment's prediction timestamp;
- be calculated after the temporal split is defined;
- be implemented in a fold-safe or train-only manner where relevant;
- exclude future outcomes and full-dataset aggregation;
- avoid using patient or provider identity as a shortcut for memorization; and
- receive a separate temporal-leakage and generalization review before use.

## Structural Missingness and Outcome Leakage

Missingness in `check_in_at`, `chair_start_at`, `chair_end_at`, `checkout_at`, and `status_change_reason` is structurally related to final appointment outcomes. These missingness patterns must not be converted into predictor flags, imputation indicators, counts, interactions, or other model inputs.

The fields themselves and any representation of whether they are present are `prohibited_leakage`.

## Baseline Feature Boundary

The initial baseline may consider only the following eligible inputs.

Direct candidates:

- `planned_duration_min`
- `visit_type`
- `booking_channel`

Approved derived candidates:

- `booking_lead_time_hours`
- scheduled weekday, hour, and month or season
- `approximate_age_at_prediction`
- `patient_registration_tenure`
- `dentist_tenure`

No exact identifier, reminder flag, outcome field, post-prediction field, evaluation-only field, deferred snapshot field, or historical aggregate is inside this initial boundary. This boundary defines eligibility; it does not claim that every eligible feature will improve performance or remain in a final model.

Preprocessing, encoding, missing-value handling, split design, model selection, calibration, threshold selection, and fairness evaluation remain separate later decisions. Model development is not initiated by this contract.

## Deferred Decisions

The following require additional evidence or a separate review:

- event timing for `reminder_sent`;
- historical or effective-dated versions of mutable patient and dentist attributes;
- population timing for a possible boolean rescheduling-lineage feature;
- design and validation of patient- or dentist-history aggregates;
- treatment of `sex` after subgroup and fairness assessment; and
- any additional source table or feature not explicitly covered by this contract.

A deferred field remains unavailable for predictor use unless a later documented decision changes its status.

## Data and Use Disclaimer

The source data are fully synthetic. The timing relationships and eligibility decisions in this contract describe this dataset and its stated prediction design only. They have not been validated for real-world clinical or operational use and must not be used to support clinical decisions.
