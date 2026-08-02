# Prediction Time

## Decision

The prediction timestamp for each appointment is defined as:

`prediction_time = scheduled_start_at - 24 hours`

At this timestamp, the model would estimate no-show risk only for appointments that are active at prediction time. Information created, updated, or first known after the prediction timestamp must not be used as a predictor.

## Operational Rationale

The intended use case is to identify appointments at elevated risk of no-show early enough for a dental clinic to consider a proportionate operational intervention, such as reminder outreach or scheduling follow-up.

The 24-hour horizon provides time for an operational response while limiting uncertainty from cancellations and rescheduling that occur between prediction and the scheduled start. It also aligns with an operationally familiar appointment-reminder workflow.

## Eligible Prediction Cohort

An appointment is included in the eligible prediction cohort only if:

1. it has already been booked by the prediction timestamp; and
2. it has not already been cancelled or rescheduled by that timestamp.

Appointments booked less than 24 hours before their scheduled start are not yet booked at the standard prediction timestamp and are therefore excluded. Appointments already cancelled or rescheduled are also excluded because they are not active at prediction time.

The 24-hour cohort reconciles as follows:

| Cohort step | Appointments |
|---|---:|
| Total raw appointments | 8,000 |
| Not yet booked at the prediction timestamp | 64 |
| Already cancelled or rescheduled by the prediction timestamp, among appointments already booked | 1,150 |
| **Active and eligible at prediction time** | **6,786** |

The 64 appointments not yet booked represent approximately 0.80% of all 8,000 raw appointments. The accounting reconciles exactly:

`8,000 - 64 - 1,150 = 6,786`

## Horizon Comparison

The following comparison uses the same prediction-time eligibility rules at each candidate horizon:

| Prediction horizon | Active cohort | No-shows | No-show rate | Cancelled later | Rescheduled later |
|---|---:|---:|---:|---:|---:|
| 24 hours | 6,786 | 820 | 12.08% | 52 | 24 |
| 48 hours | 6,766 | 810 | 11.97% | 104 | 58 |
| 72 hours | 6,730 | 796 | 11.83% | 159 | 91 |

"Cancelled later" and "Rescheduled later" refer to appointments that were active at the relevant prediction timestamp but received the final appointment status `cancelled` or `rescheduled` afterward.

The no-show rate is broadly stable across the three candidate horizons. Earlier prediction provides more intervention time, but it also increases the number of appointments subsequently cancelled or rescheduled. The 24-hour horizon balances operational lead time with lower uncertainty about these later outcomes.

## Leakage Principle

Predictors must represent information genuinely available at or before the prediction timestamp. Final appointment outcomes and information generated afterward, including final `status`, outcome-related `status_updated_at` values, check-in and chair timestamps, completed procedures, and post-appointment payments, must not be used as predictors. This document does not approve any variable for predictor use; feature eligibility requires a separate temporal leakage review.

The binary outcome and the treatment of appointments cancelled or rescheduled after prediction time are defined in [`docs/target_definition.md`](target_definition.md).
