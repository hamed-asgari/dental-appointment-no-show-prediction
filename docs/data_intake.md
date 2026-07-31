# Data Intake

## Purpose

This document records the initial source-data selection for the v0.1 foundation phase of the Dental Appointment No-show Prediction project.

All source data are fully synthetic and originate from the separate **Dental Clinic Operations Analytics** project, frozen at **Version 1.0.0**.

Project 1 remains separate and must not be modified.

## Source Inventory

Project 1 contains nine raw synthetic source datasets:

- `appointments.csv`
- `appointment_procedures.csv`
- `dentists.csv`
- `patients.csv`
- `payments.csv`
- `payment_allocations.csv`
- `procedure_catalog.csv`
- `treatment_plans.csv`
- `treatment_plan_items.csv`

The initial Project 2 intake intentionally uses only a subset of these files.

## Initial Core Datasets

### `appointments.csv`

Included because it contains the appointment-level scheduling records required to define the prediction population and, in a later phase, construct the outcome label.

The file contains 8,000 appointment records.

Some fields represent final or post-appointment information and must not be used as predictors. Inclusion of the raw file therefore does not imply predictor eligibility.

### `patients.csv`

Included because it contains patient-level attributes that may provide information available before an appointment.

The file contains 2,000 patient records.

Individual fields will still require prediction-time and leakage review. For example, a current-state field such as `patient_status` must not automatically be assumed to represent the patient's status at a historical prediction timestamp.

### `dentists.csv`

Included because provider characteristics may be known before the scheduled appointment and may later be considered during feature review.

The file contains seven dentist records.

Fields describing current state, such as `active`, will require temporal interpretation before any modeling use.

## Deferred Datasets

The following Project 1 datasets are intentionally not included in the initial raw intake.

### `appointment_procedures.csv`

Deferred because it contains fields such as `completion_status` and procedure-level information that may reflect events occurring during or after an appointment.

Any future use would require explicit proof that the relevant information was known before the prediction timestamp.

### `payments.csv`

Deferred because financial history could potentially be useful, but records are timestamped by `received_at` and would require strict historical filtering relative to each appointment's prediction timestamp.

### `payment_allocations.csv`

Deferred because the table does not contain an independent event timestamp and depends on relationships to payment and procedure records, making temporal availability more complex.

### `procedure_catalog.csv`

Deferred because it is primarily static reference data and currently has no independently justified leakage-safe role without an appropriate pre-appointment procedure context.

### `treatment_plans.csv`

Deferred because potential use would require temporal review of fields such as `proposed_at` and careful assessment of the relationship between treatment plans and scheduled appointments.

### `treatment_plan_items.csv`

Deferred because fields such as `decision_at` and treatment-plan decisions require temporal filtering and may not have been available at the prediction timestamp.

## Selection Principle

A source dataset is not included merely because it exists.

Data should enter the analytical workflow only when there is a justified role for the prediction problem and its information can be aligned with the chosen prediction timestamp without temporal or target leakage.

Deferred datasets may be reconsidered in later phases if their use can be justified and implemented with appropriate temporal safeguards.

## Raw Data Integrity

The three approved source datasets were copied exactly from Project 1 into Project 2:

- `data/raw/appointments.csv`
- `data/raw/patients.csv`
- `data/raw/dentists.csv`

SHA-256 verification confirmed that each Project 2 copy is identical to its Project 1 source file.

No transformations were applied before ingestion.

These files should be treated as immutable raw inputs.

## Scope Boundary

This data-intake decision does not define:

- the final modeling features,
- the binary target,
- the final handling of cancellations or rescheduling,
- exploratory analysis,
- feature engineering,
- or model development.

Those decisions belong to later project phases.