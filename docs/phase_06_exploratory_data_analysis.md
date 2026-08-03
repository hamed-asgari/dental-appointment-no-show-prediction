# Phase 06 — Exploratory Data Analysis

## 1. Purpose and leakage controls

This report records the approved Phase 06 exploratory findings from the eleven
deterministic CSV artifacts. The five generated PNGs were used only to cross-check the
tabular interpretation and are not embedded because `reports/eda` is generated and ignored.

The analysis enforces these population boundaries:

- Supervised target analysis uses only rows with `split == train` and
  `development_fit_eligible == True`.
- Train-to-validation drift uses predictor columns only. The generated artifacts contain
  3,670 mature-train feature rows and 1,541 nominal validation feature rows; neither drift
  population exposes a target.
- The 3,682-row nominal-train cohort is used only for the target-free maturity and temporal
  audit. Its 12 maturity-excluded rows do not enter supervised summaries.
- Validation target values are not analyzed.
- Test data are not used anywhere in Phase 06 EDA.
- The combined pretest fit population is reserved for later modeling work.
- Post-event fields and prohibited status information are not used as predictors or
  analytical features.

## 2. Analytical populations

| Population or audit | Rows | Role in Phase 06 |
|---|---:|---|
| Nominal train | 3,682 | Target-free maturity and temporal audit |
| Mature supervised train | 3,670 | Target summaries and feature EDA |
| Maturity exclusions | 12 | Audited only; excluded from supervised EDA |
| Validation feature population | 1,541 | Feature-only drift comparison |
| Test | — | Untouched |

The mature supervised population contains 432 target positives and 3,238 negatives. Its
no-show prevalence is 11.77%, with a 95% Wilson interval of 10.77%–12.85%.

Nominal-train prediction times run from `2024-03-01T09:00:00.000000` through
`2025-02-28T18:00:00.000000`. Mature-train prediction times run from
`2024-03-01T09:00:00.000000` through `2025-02-26T18:30:00.000000`. Coverage therefore spans
12 calendar months, from `2024-03` through `2025-02`.

Maturity is governed by the strict contract
`status_updated_at < model_fit_time`. The 12 exclusions are rows that do not satisfy that
contract. The artifacts do not establish any more specific cause, so none is inferred.

## 3. Target balance

| Measure | Mature-train result |
|---|---:|
| Appointments | 3,670 |
| No-shows (positive target) | 432 |
| Attended (negative target) | 3,238 |
| No-show prevalence | 11.77% |
| 95% Wilson interval | 10.77%–12.85% |
| Duplicate appointment IDs | 0 |

No-shows are the minority class. Model development will therefore need class-sensitive
evaluation, including attention to minority-class performance and probability quality. This
finding does not yet select a resampling method, decision threshold, or model.

## 4. Data quality and missingness

All ten approved features are complete in all 3,670 mature-train rows. Every feature has
zero missing values and a 0.00% missing rate, and no approved feature is constant.

Distinct non-null counts add useful context to feature granularity:

| Feature group | Distinct non-null values |
|---|---|
| `planned_duration_min` | 3 |
| `visit_type`; `booking_channel` | 6; 5 |
| `scheduled_weekday`; `scheduled_hour`; `scheduled_month` | 6; 9; 12 |
| `booking_lead_time_hours` | 3,566 |
| `approximate_age_at_prediction` | 76 |
| `patient_registration_tenure_days`; `dentist_tenure_days` | 657; 558 |

The low cardinality of planned duration is a property of the observed data, not evidence
that the feature is uninformative. The numerically stored weekday, hour, and month fields
are categorical scheduling fields.

## 5. Numerical feature distributions

The outlier audit uses the standard 1.5-IQR fences. Target-group summaries are descriptive
associations within mature train; they do not establish causality.

| Feature | Overall median (Q1–Q3) | Min–max | Fence outliers below / above | Attended median (Q1–Q3) | No-show median (Q1–Q3) |
|---|---:|---:|---:|---:|---:|
| `planned_duration_min` | 45 (30–60) | 30–60 | 0 / 0 | 45 (30–60) | 45 (30–60) |
| `booking_lead_time_hours` | 759.29 (386.74–1,103.51) | 24.50–8,297.00 | 0 / 50 | 742.89 (384.23–1,099.27) | 821.88 (432.37–1,148.07) |
| `approximate_age_at_prediction` | 43 (24–61) | 5–80 | 0 / 0 | 43 (24–61) | 42 (24.75–60) |
| `patient_registration_tenure_days` | 419 (309–530) | 71–784 | 0 / 0 | 421 (312–531) | 410.50 (295.25–522.25) |
| `dentist_tenure_days` | 519 (430–617) | 182–787 | 0 / 0 | 519 (430.25–619) | 523.50 (428–607) |

Booking lead time is the only numerical feature with fence outliers: 50 observations exceed
the 2,178.66-hour upper fence, and none falls below the lower fence. Its maximum of 8,297
hours is well beyond its 1,103.51-hour third quartile. No-show appointments have a higher
lead-time median than attended appointments, while the other target-group median differences
are small relative to their within-group IQRs. Univariate overlap alone is not a reason to
drop any feature.

## 6. Categorical feature distributions

The artifact marks a level as rare when its count is below 30 or its share is below 1%, and
as high-uncertainty when its count is below 30 or it has fewer than five positives. Rates
below are mature-train no-show rates with 95% Wilson intervals.

- **Visit type:** `treatment` dominates with 1,567 appointments (42.70%) and a 12.06% rate
  (10.54%–13.77%). `recall_examination` and `new_patient_examination` follow at 16.62% and
  14.55% of rows, with rates of 10.66% (8.45%–13.36%) and 10.30%
  (8.00%–13.17%). `emergency` has the highest observed rate, 15.17%
  (11.50%–19.75%), across 290 appointments. No visit-type level is marked rare or
  high-uncertainty.
- **Booking channel:** `phone` accounts for 1,675 rows (45.64%) and has a 12.18% rate
  (10.70%–13.83%); `online` accounts for 727 rows (19.81%) and has a 10.59% rate
  (8.56%–13.04%). The least frequent observed channel, `other`, has 143 rows (3.90%) and a
  9.79% rate with a wider interval (5.92%–15.76%). No booking-channel level is marked rare
  or high-uncertainty.
- **Scheduled weekday:** the six observed categories each contribute about 16%–17% of
  rows. Level `6` has the highest rate, 13.07% (10.60%–16.01%), and level `2` the lowest,
  9.12% (7.07%–11.69%). Level `4` has zero observations, an undefined rate, and is marked
  both rare and high-uncertainty.
- **Scheduled hour:** the most frequent observed level is `16`, with 526 rows (14.33%) and
  a 12.93% rate (10.33%–16.07%). The highest observed rate is at level `15`, 13.97%
  (10.94%–17.67%); the lowest is at level `9`, 10.00% (7.13%–13.84%). Levels `0`–`8`,
  `13`, and `19`–`23` have zero observations and are marked rare and high-uncertainty.
- **Scheduled month:** observed shares range from 7.22% for level `11` to 9.18% for level
  `7`; neither is rare. The highest observed rate is at level `3`, 15.63%
  (12.06%–20.01%), and the lowest at level `6`, 8.85% (6.16%–12.57%). No month level is
  marked rare or high-uncertainty.

The Wilson intervals, especially for less frequent levels, counsel against treating the
rank ordering of these rates as stable evidence. The numeric weekday, hour, and month codes
remain categorical scheduling fields throughout interpretation.

## 7. Temporal coverage and maturity exclusions

| Month | Nominal train | Mature train | Excluded | No-shows / attended | No-show rate (95% Wilson interval) |
|---|---:|---:|---:|---:|---:|
| `2024-03` | 333 | 333 | 0 | 51 / 282 | 15.32% (11.84%–19.58%) |
| `2024-04` | 319 | 319 | 0 | 41 / 278 | 12.85% (9.62%–16.97%) |
| `2024-05` | 306 | 306 | 0 | 37 / 269 | 12.09% (8.90%–16.22%) |
| `2024-06` | 306 | 306 | 0 | 27 / 279 | 8.82% (6.13%–12.53%) |
| `2024-07` | 335 | 335 | 0 | 34 / 301 | 10.15% (7.35%–13.85%) |
| `2024-08` | 298 | 298 | 0 | 32 / 266 | 10.74% (7.71%–14.77%) |
| `2024-09` | 321 | 321 | 0 | 34 / 287 | 10.59% (7.68%–14.44%) |
| `2024-10` | 302 | 302 | 0 | 32 / 270 | 10.60% (7.61%–14.58%) |
| `2024-11` | 277 | 277 | 0 | 30 / 247 | 10.83% (7.69%–15.04%) |
| `2024-12` | 285 | 285 | 0 | 40 / 245 | 14.04% (10.48%–18.55%) |
| `2025-01` | 321 | 321 | 0 | 38 / 283 | 11.84% (8.75%–15.83%) |
| `2025-02` | 279 | 267 | 12 | 36 / 231 | 13.48% (9.90%–18.10%) |

All 12 maturity exclusions occur in `2025-02`; every earlier month has identical nominal
and mature counts. Monthly mature counts range from 267 to 335, and rates range from 8.82%
to 15.32%. The intervals overlap substantially, the sequence does not repeat, and only 12
months are observed, so these movements do not establish seasonality.

## 8. Train-to-validation numerical drift

The signed standardized mean difference (SMD) uses validation minus train, divided by the
pooled scale; its sign is preserved. Quantile shifts also use validation minus train. Each
feature has 3,670 non-missing train values and 1,541 non-missing validation values, so every
missing-rate difference is 0.000 percentage points.

| Feature | Signed SMD | Q10 shift | Median shift | Q90 shift | Missing-rate difference |
|---|---:|---:|---:|---:|---:|
| `planned_duration_min` | -0.039 | 0.000 min | 0.000 min | 0.000 min | 0.000 pp |
| `booking_lead_time_hours` | +0.181 | +4.650 h | +9.242 h | +66.713 h | 0.000 pp |
| `approximate_age_at_prediction` | +0.114 | +2.000 years | +2.000 years | +2.000 years | 0.000 pp |
| `patient_registration_tenure_days` | +1.945 | +309.100 d | +260.000 d | +205.000 d | 0.000 pp |
| `dentist_tenure_days` | +2.350 | +307.000 d | +270.000 d | +202.000 d | 0.000 pp |

Validation tenure distributions are shifted upward across the reported quantiles. Booking
lead time has a modest median shift but a larger upper-quantile shift; its mean also rises
from 796.14 to 976.55 hours alongside greater validation dispersion. Planned duration has a
slightly negative SMD despite unchanged reported quantiles. These are measurements, not
acceptability labels; no arbitrary drift threshold is applied.

## 9. Train-to-validation categorical drift

All comparisons are feature-only. Missing-rate differences are validation minus train.

| Feature | Total variation | Maximum level-share difference | Unseen-in-train validation levels | Train levels absent from validation | Missing-rate difference |
|---|---:|---:|---:|---:|---:|
| `visit_type` | 0.024 | 0.019 | 0 | 0 | 0.000 pp |
| `booking_channel` | 0.016 | 0.011 | 0 | 0 | 0.000 pp |
| `scheduled_weekday` | 0.023 | 0.013 | 0 | 0 | 0.000 pp |
| `scheduled_hour` | 0.023 | 0.013 | 0 | 0 | 0.000 pp |
| `scheduled_month` | 0.568 | 0.136 | 0 | 7 | 0.000 pp |

The largest level contributions for `visit_type` are `treatment` (-1.945 percentage points)
and `recall_examination` (+1.873 pp). For `booking_channel`, they are `phone` (-1.059 pp)
and `in_person` (+0.931 pp); for weekday, level `0` (+1.270 pp) and level `3` (-0.905 pp);
and for hour, level `15` (+1.277 pp) and level `16` (-1.159 pp).

`scheduled_month` is structurally different across the temporal split. Validation contains
five observed levels, while train contains 12. Levels `1`, `2`, and `8`–`12` are absent
from validation; together they account for 2,083 train rows (56.76%). The largest positive
validation-minus-train shares are level `6` (+13.623 pp), level `5` (+12.315 pp), and level
`4` (+11.408 pp). No validation level is unseen in train for any categorical feature. These
findings use no validation outcomes and are not assigned pass/fail severity thresholds.

## 10. Numerical feature relationships

The approved relationship artifact uses the same 3,670-row mature-train feature-only
projection as the training side of drift. It does not include the 12 maturity-excluded
nominal-train rows. Every one of the ten feature pairs has 3,670 complete pairs, no missing
pairs, and defined Pearson and Spearman correlations.

The materially notable association is between `patient_registration_tenure_days` and
`dentist_tenure_days`: Pearson is +0.575 and Spearman is +0.579, the strongest positive
result by both measures. The next-largest positive results are much smaller:
`booking_lead_time_hours` with dentist tenure has Pearson +0.090 and Spearman +0.043, while
its association with patient tenure has Pearson +0.074 and Spearman +0.029.

The strongest negative Pearson result is only -0.011, between planned duration and age; the
strongest negative Spearman result is -0.021, between planned duration and booking lead
time. Across all pairs other than the two tenure features, absolute Pearson correlations are
at most 0.090 and absolute Spearman correlations are at most 0.043. Pairwise correlations
alone do not demonstrate a multicollinearity problem and do not replace model-based
diagnostics. This section contains no validation or test target analysis.

## 11. Modeling implications

- Class-sensitive evaluation metrics will be necessary because no-shows are the minority
  class; the specific metric set remains a later modeling decision.
- No approved feature is missing in the analyzed train or validation artifacts. A defined
  missing-value policy is still appropriate for pipeline robustness, but EDA does not
  identify a currently affected feature.
- Categorical encoding must handle rare, absent, and previously unseen levels without
  learning from validation or test data.
- All learned preprocessing must be fitted on the population appropriate to the modeling
  stage and then applied unchanged to later temporal partitions.
- Temporal validation should remain the primary development evaluation.
- The tenure and calendar-distribution shifts should inform robustness checks and future
  drift monitoring; the observed metrics are not standalone acceptance rules.
- Pairwise relationships do not replace model-based diagnostics or interaction assessment.
- No feature should be removed at this stage solely because of univariate overlap,
  frequency, drift, or pairwise correlation.

EDA does not select a final model, resampling strategy, probability threshold, feature
subset, hyperparameter set, or categorical encoder.

## 12. Limitations

- The data are fully synthetic and project-specific; behavior may not transfer to real
  dental practices, patients, or operational systems.
- Observed associations are descriptive and not causal.
- Validation labels were intentionally excluded, so Phase 06 does not describe validation
  target prevalence or target relationships.
- Test data remain untouched.
- The 12-month training span and single observed cycle restrict seasonality conclusions.
- Rare or unobserved categorical levels have wide or undefined target-rate uncertainty.
- Calendar-category drift partly reflects the temporal partition and should not be confused
  with an outcome-based finding.
- EDA does not estimate final generalization performance.

## 13. Reproduction

From the repository root, run exactly:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.run_eda
```

The command creates eleven deterministic CSV files and five deterministic PNG files under
`reports/eda`. That directory is generated and intentionally ignored by Git. The command
leaves `data/processed` unchanged.
