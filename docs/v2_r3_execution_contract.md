# Version 2 Phase R3 Execution Contract

## Status

**Frozen before any R3 protected-final-test probability vector or target access.**

This contract governs interpretation, error analysis, persistence,
reproducibility, the one-time protected 2027 final-test evaluation, and the app
decision gate for Version 2 recovery Phase R3.

Machine-readable contract:

```text
configs/v2_r3_execution.json
```

SHA-256:

```text
c0b259a4bb81790a30fd6e2c2fd2495e10869d700ae783196c5eb055db46f7a5
```

## Frozen upstream state

R3 inherits the formally closed R2 state at commit `7fdcf40`.

The selected ranking model remains `logistic_regression`; the selected
calibration method remains `uncalibrated`; no single operational threshold was
selected in R2.

The frozen policy manifest remains:

```text
33391bc9295c6a9d93bb8797e8e3836fb9af45062f1b431a4debc9bc7822e4a4
```

No R3 activity may change the 32-feature allowlist, preprocessing design,
estimator hyperparameters, calibration choice, or R2 policy grid.

## Frozen model fit

The persisted R3 model is the same frozen base-estimator design used by the R2
policy engine:

- fit time: `2026-07-01T00:00:00`
- training partitions:
  `development_fit`, `fold_1_validation`, `fold_2_validation`,
  `fold_3_validation`
- estimator: Logistic Regression
- calibration transform: none (`uncalibrated`)
- no refit on calibration or policy-selection data
- no feature selection or hyperparameter search

This keeps the final protected evaluation from introducing a post-R2 refit.

## R3 execution order

R3 must proceed in this order:

1. persist the frozen preprocessing/model pipeline and machine-readable
   metadata;
2. produce interpretation, error analysis, and subgroup diagnostics using only
   permitted non-test data;
3. generate the complete target-free protected-test probability vector;
4. validate its exact appointment order and SHA-256 seal it;
5. commit that probability vector and require CI green before any protected
   target access;
6. perform one explicit protected-target access and final evaluation;
7. document the app decision gate without post-test tuning.

The protected target must never be loaded merely to help create, repair, or
reorder the probability vector.

## Pre-test interpretation and diagnostics

The primary diagnostic population is the already permitted
`policy_selection` partition. It is the most recent pre-test population scored
by the frozen selected model and avoids mixing predictions from fold-specific
estimators.

Permutation importance is predeclared as:

- raw-feature permutation before preprocessing;
- Average Precision scoring;
- 20 repeats;
- random state `20260807`;
- mean and standard deviation reported;
- interpretation only, never feature selection.

First-time versus repeat-patient diagnostics are defined by
`patient_history_available = false` versus `true`.

Predeclared subgroup features are:

- `patient_history_available`
- `reminder_sent_by_prediction_time`
- `visit_type`
- `booking_channel`
- `scheduled_weekday`

A subgroup is quantitatively reported only when it has at least 100 rows and at
least 10 positive no-shows. Insufficient groups are reported as unsupported;
they do not trigger pooling, model changes, or threshold changes.

Continuous error analysis reports absolute probability error, per-row Brier
contribution, and per-row log-loss contribution. Capacity-based error summaries
use all three already registered capacity fractions: 5%, 10%, and 20%. No
single operational threshold is selected.

## Persistence and reproducibility

The frozen model pipeline will be persisted under `models/v2/` with:

- a loadable pipeline artifact;
- machine-readable metadata;
- an artifact manifest with SHA-256 identity;
- a load smoke test; and
- a prediction replay test.

Persistence is packaging of the frozen estimator, not a new modeling decision.

## Protected final-test gate

The protected test remains:

```text
2027-01-01 <= prediction_time < 2028-01-01
```

Expected target-free feature rows: `4343`.

Before target access, R3 must create exactly:

```text
appointment_id,no_show_probability
```

for every protected-test appointment in exact frozen appointment order.
Probabilities must be finite and in `[0, 1]`.

The vector is written to:

```text
reports/modeling/v2/final_test/final_test_probabilities.csv
```

It must be validated, SHA-256 sealed, committed, pushed, and CI-green before
`allow_test=True` may be used.

After the probability seal:

- no estimator refit;
- no feature changes;
- no calibration changes;
- no probability-vector replacement because of observed labels;
- no hyperparameter tuning;
- no final-test threshold selection.

Protected-target access is a one-time evaluation operation.

## Final-test reporting

The final test reports:

- Average Precision;
- ROC-AUC;
- Brier score;
- log loss;
- calibration intercept;
- calibration slope;
- sample size and positive count.

The comparison baseline is a constant population prior estimated only from the
frozen base-training labels.

The already registered capacity fractions (5%, 10%, 20%) and cost ratios
(1:1, 2:1, 5:1, 10:1) may be replayed descriptively on the final test. They
must not be converted into a newly selected operational threshold.

## App decision gate

The appointment-level risk demonstration is allowed only when **all** of the
following hold on the one-time protected test:

- Average Precision absolute uplift versus the frozen population-prior baseline
  is at least `0.005`;
- ROC-AUC is at least `0.52`;
- Brier score is no worse than the population-prior baseline;
- log loss worsens by no more than `0.005` versus the population-prior
  baseline.

The ranking thresholds reuse the frozen R2 usefulness gate; the log-loss
tolerance reuses the frozen R2 probability-quality guardrail. Adding the Brier
condition ensures that appointment-level risk presentation also requires
probability quality beyond the constant baseline.

If any condition fails, R4 must implement the transparent model-evaluation
dashboard instead of an individualized appointment-risk demonstration.

The app gate may choose the **type of app** only. It may not select a new model,
feature set, calibration method, or threshold from final-test outcomes.

## Protected state at contract freeze

```text
final_test_probabilities_generated = false
final_test_target_accessed = false
```

This contract is frozen before either event.
