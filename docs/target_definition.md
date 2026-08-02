# Target Definition

## Prediction Question

The primary prediction question is:

**Among appointments that are active 24 hours before the scheduled start, what is the probability that the appointment will ultimately become a no-show?**

The prediction timestamp for each appointment is:

`prediction_time = scheduled_start_at - 24 hours`

## Eligible Prediction Cohort

An appointment is included in the eligible prediction cohort only if:

1. it has already been booked by the prediction timestamp; and
2. it has not already been cancelled or rescheduled by that timestamp.

Appointments booked less than 24 hours before their scheduled start are excluded because they are not yet booked at the prediction timestamp. Appointments cancelled or rescheduled before the prediction timestamp are excluded because they are not active at prediction time and would not require a no-show prediction.

Of 8,000 raw appointments, 64 were not yet booked at the prediction timestamp. Among the appointments already booked, 1,150 had already been cancelled or rescheduled. This leaves 6,786 appointments active and eligible at prediction time:

`8,000 - 64 - 1,150 = 6,786`

## No-Show Definition and Observed Timing Convention

For this project, a no-show is an appointment in the eligible prediction cohort whose final appointment status in the synthetic dataset is `no_show` (`status == "no_show"`).

The full raw dataset contains 826 records with final `status == "no_show"`. For all 826 records, the data show the same relationship:

`status_updated_at = scheduled_start_at + 15 minutes`

Of those 826 raw no-show records, 820 are in the eligible prediction cohort.

This 15-minute interval is an observed convention in the fully synthetic dataset, not a validated clinical or industry standard. For this project, the final `no_show` status is interpreted as the synthetic dataset's operational outcome label after that interval. The interval does not establish a general real-world definition of a no-show.

## Binary Target

For appointments active at the prediction timestamp:

- `target = 1` when final `status == "no_show"`
- `target = 0` when final `status` is `completed`, `cancelled`, or `rescheduled`

The eligible prediction cohort has the following verified outcomes:

| Final appointment status | Count | Target |
|---|---:|---:|
| `completed` | 5,890 | 0 |
| `no_show` | 820 | 1 |
| `cancelled` after prediction time | 52 | 0 |
| `rescheduled` after prediction time | 24 | 0 |
| **Total eligible prediction cohort** | **6,786** | |

The counts reconcile exactly:

`5,890 + 820 + 52 + 24 = 6,786`

The positive-class prevalence is approximately 12.08% (`820 / 6,786`).

## Cancellation and Rescheduling Rule

Cancellation and rescheduling are handled according to what is known at the prediction timestamp:

- Appointments cancelled or rescheduled before prediction time are excluded because they are not active when a prediction would be made.
- Appointments active at prediction time but cancelled or rescheduled later remain in the cohort and receive `target = 0`.

Removing appointments that are cancelled or rescheduled after prediction time would use post-prediction information to select the development population and would introduce post-prediction selection. Retaining these appointments keeps the eligible prediction cohort aligned with the appointments for which predictions would have been generated operationally.

## Roles of Outcome Fields

The two appointment outcome fields have separate retrospective roles:

- `status` supplies the final appointment status used to construct the binary outcome label.
- `status_updated_at` is used to reconstruct whether a cancellation or rescheduling occurred before or after the prediction timestamp. It also documents the observed synthetic no-show timing convention described above.

Both fields contain outcome information. Neither `status` nor `status_updated_at` may be used as a model predictor.

## Secondary Sensitivity Analysis

A secondary sensitivity analysis is planned to compare only appointments with final `status` values of `completed` and `no_show`. This would answer a narrower completed-versus-no-show question and will not replace the primary target or determine the primary reported result.

## Leakage and Scope

Information created, updated, or first known after the prediction timestamp must not be used as a predictor. Feature eligibility and the broader leakage review have not yet been completed; modeling and validation are also outside the scope of this target-definition step.

No other appointment, patient, procedure, or payment variable is approved as a predictor by this document. Each candidate feature must later be assessed according to whether it was genuinely available at the prediction timestamp.

## Data Disclaimer

All observations in this project are fully synthetic. The cohort counts and timing pattern describe this dataset only.

This target definition is intended for a portfolio and learning project. It has not been validated for real-world clinical or operational use and must not be used to support clinical decisions.
