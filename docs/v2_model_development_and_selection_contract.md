# Version 2 Model Development and Selection Contract

## Status

**Frozen before any Version 2 recovered-model metric is computed.**

This contract governs recovery Phase R2. It is intentionally committed before
fitting or scoring the Version 2 population-prior, Logistic Regression, or
tree-based candidates. It does not record model performance.

The immutable Phase R1 input is:

```text
processed feature artifact:
data/processed/v2/v2_feature_dataset.csv

dataset SHA-256:
08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53

manifest SHA-256:
2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073

processed dataset fingerprint:
0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787
```

At contract freeze, `target_included = false` and
`final_test_target_accessed = false`.

The machine-readable companion is `configs/v2_model_development.json`.

## Purpose

Phase R2 asks whether the frozen 32-feature Version 2 representation supports
reproducible appointment-level ranking and probability estimation beyond a
population-prior baseline. Model ranking, calibration, and policy sensitivity
are separate decisions.

No feature, benchmark, generator, evaluation window, or candidate
hyperparameter may be revised in response to recovered model performance.

## Approved predictors and preprocessing

Only `V2_MODEL_FEATURE_COLUMNS` may enter an estimator.

Feature roles are frozen as:

- categorical: `visit_type`, `booking_channel`, `scheduled_weekday`,
  `scheduled_hour`, and `scheduled_month`;
- boolean: reminder availability/history-support indicators; and
- numeric: the remaining approved duration, tenure, count, recency, and
  smoothed-rate features.

Categorical values use a training-fitted constant missing-value fill and
one-hot encoding with unknown categories ignored. Numeric values use
training-only median imputation. Logistic Regression additionally uses a
training-fitted standard scaler for numeric values. The Random Forest does not
scale numeric values.

There is no feature selection, interaction search, hyperparameter search,
target encoding, oversampling, class reweighting, or data-driven feature
addition in R2.

## Frozen candidate set

### Population-prior baseline

The baseline probability is the no-show prevalence among the strictly mature
training labels available at that fitting event. It is constant for every
validation row.

### Logistic Regression

```text
C = 1.0
solver = lbfgs
max_iter = 2000
class_weight = None
random_state = 20260807
```

### Random Forest

```text
n_estimators = 300
max_depth = 10
min_samples_leaf = 10
min_samples_split = 20
max_features = sqrt
bootstrap = true
class_weight = None
random_state = 20260807
n_jobs = 1
```

No additional estimator may be added after metrics are inspected.

## Rolling-origin development

Training targets always satisfy:

```text
label_available_at < fit_time
```

| Fold | Fit time | Training partitions | Validation partition | Validation-label cutoff |
|---|---|---|---|---|
| 1 | 2025-01-01 | `development_fit` | `fold_1_validation` | 2025-07-01 |
| 2 | 2025-07-01 | `development_fit`, `fold_1_validation` | `fold_2_validation` | 2026-01-01 |
| 3 | 2026-01-01 | `development_fit`, `fold_1_validation`, `fold_2_validation` | `fold_3_validation` | 2026-07-01 |

Validation metrics use only labels strictly mature before the declared
validation-label cutoff. This means late labels at an exact boundary are
excluded rather than backfilled from the future.

Reports must include each fold, an unweighted macro mean across folds, and a
pooled validation summary.

## Ranking selection

Required fold metrics are Average Precision, ROC-AUC, Brier score, log loss,
sample size, and positive count.

A non-constant candidate qualifies as showing minimally useful development
ranking only when all three conditions hold:

1. macro mean Average Precision exceeds the population prior by at least
   `0.005` absolute;
2. macro mean ROC-AUC is at least `0.52`; and
3. Average Precision uplift over the prior is positive in at least two of
   three folds.

These are model-selection guardrails, not minimum performance requirements for
accepting or modifying the frozen synthetic benchmark.

Among qualifying non-constant candidates, ordering is fixed as higher macro
mean Average Precision, higher macro mean ROC-AUC, lower macro mean Brier
score, lower macro mean log loss, then Logistic Regression over Random Forest
if still exactly tied.

If neither non-constant candidate passes the usefulness gate, the R2 final
candidate falls back to the population prior and the project must report that
appointment-level ranking remains unsupported.

## Calibration chronology

After rolling-origin ranking selection, the selected non-constant research
candidate is refit once at `2026-07-01 00:00:00` using only strictly mature
labels from the four pre-calibration development partitions.

```text
calibration fit prediction_time:
2026-07-01 <= prediction_time < 2026-09-01

calibrator fit time:
2026-09-01 00:00:00

calibration evaluation prediction_time:
2026-09-01 <= prediction_time < 2026-10-01

calibration evaluation labels:
label_available_at < 2026-10-01 00:00:00
```

The frozen methods are uncalibrated, sigmoid, and isotonic.

Calibration selection uses the lowest Brier score subject to a log-loss
guardrail: a calibrated candidate may not worsen log loss by more than `0.005`
absolute versus the uncalibrated candidate. Brier differences smaller than
`0.001` are practically indifferent, with simplicity preference:

```text
uncalibrated -> sigmoid -> isotonic
```

Required reporting includes Brier score, log loss, calibration intercept and
slope when estimable, Average Precision, ROC-AUC, sample size, positive count,
and a 10-bin quantile reliability curve.

Calibration cannot change the ranking-based base-estimator choice. The base
estimator is not refit again after calibration.

## Policy-selection window

Threshold and capacity sensitivity use only `policy_selection` rows and only
labels satisfying:

```text
label_available_at < 2027-01-01 00:00:00
```

No threshold defaults to `0.50`.

Fixed capacity scenarios are `5%`, `10%`, and `20%`. Fixed
false-negative:false-positive classification-cost ratios are `1:1`, `2:1`,
`5:1`, and `10:1`.

The cost threshold is:

```text
1 / (1 + false_negative_to_false_positive_cost_ratio)
```

Report threshold, selected count/fraction, precision, recall, confusion
counts, and scenario cost for every scenario.

These are sensitivity scenarios, not validated clinic economics. R2 does not
advertise a single operational threshold without independent clinic
assumptions.

## Protected final-test prohibition during R2

```text
2027-01-01 <= prediction_time < 2028-01-01
```

During R2:

- protected final-test targets must not be accessed;
- no final-test metric may be computed;
- no final-test probability vector may be generated;
- final-test rows may not influence preprocessing, estimator choice,
  calibration, threshold analysis, or application behavior.

## Reproducibility and claims

R2 implementation must produce development, calibration, and policy artifacts
from one deterministic command. Exact candidate configuration, dataset
identity, result tables, and artifact hashes must be recorded.

All results are synthetic-data evidence. If ranking or calibration does not
beat the population-prior reference under this contract, the negative result
is the result; the benchmark, features, thresholds, and candidate menu must
not be retuned to manufacture a stronger portfolio claim.
