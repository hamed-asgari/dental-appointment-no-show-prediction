# Version 2 Phase R3 Final Chronological Test Results

## Protected evaluation status

The one-time protected 2027 final-test evaluation has been performed after the
target-free probability vector was SHA-256 sealed, committed, pushed, and
verified green in GitHub CI run `31190546678`.

The sealed probability vector was not refit, recalibrated, replaced, or tuned
after target access.

## Final-test population

- rows: `4343`
- positives: `358`
- prevalence: `0.082431`
- frozen population-prior probability: `0.089552`

## Frozen model performance

- Average Precision: `0.147158`
- ROC-AUC: `0.630030`
- Brier score: `0.076205`
- log loss: `0.282623`
- calibration intercept: `-0.962052`
- calibration slope: `0.689034`
- mean predicted probability: `0.113190`

Population-prior baseline:

- Average Precision: `0.082431`
- ROC-AUC: `0.500000`
- Brier score: `0.075687`
- log loss: `0.284987`

Model-minus-baseline comparisons:

- AP uplift: `0.064726`
- Brier delta: `0.000518`
- log-loss delta: `-0.002364`

## Pre-frozen app decision gate

- AP uplift requirement passes:
  `true`
- ROC-AUC requirement passes:
  `true`
- Brier requirement passes:
  `false`
- log-loss requirement passes:
  `true`
- all appointment-level risk-demo requirements pass:
  `false`
- selected R4 app type:
  `transparent_model_evaluation_dashboard`

This gate chooses only the application type. It does not permit a new model,
feature set, calibration method, or final-test threshold.

## Registered policy replay

`final_test_policy_scenarios.csv` replays the already registered 5%, 10%, and
20% capacity fractions and 1:1, 2:1, 5:1, and 10:1 false-negative-to-false-
positive cost ratios. These scenarios are descriptive and do not select a
single operational threshold.

## Frozen post-test boundary

```text
final_test_probabilities_generated = true
final_test_target_accessed = true
single_operational_threshold_selected = false
post_test_model_tuning_permitted = false
```

All performance claims remain scoped to the synthetic longitudinal benchmark.
