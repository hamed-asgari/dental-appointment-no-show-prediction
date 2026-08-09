# Version 2 Phase R2 Rolling-origin Ranking Results

## Status

These are the first recovered Version 2 model metrics computed after the
pre-metric model-development contract was frozen and committed. The protected
2027 final-test target remained closed and no final-test probabilities were
generated.

## Frozen inputs

- model config SHA-256: `0b39dbe9b15c64579a81e7b9dbeaad5e5a6694fc5066698fbcfd4623a1bd1dd6`
- model contract SHA-256: `735953523db15e36b82bacfb022915c3eff0c4f4329f16c72877dc32f8ff597f`
- processed dataset SHA-256: `08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53`
- processed dataset fingerprint: `0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787`

## Strictly mature fold populations

| Fold | Training rows | Training positives | Validation rows | Validation positives |
|---|---:|---:|---:|---:|
| fold_1 | 4,452 | 383 | 2,133 | 217 |
| fold_2 | 6,600 | 601 | 2,222 | 182 |
| fold_3 | 8,839 | 785 | 2,073 | 191 |

## Fold-level threshold-free metrics

| Fold | Model | AP | ROC-AUC | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| fold_1 | `population_prior` | 0.1017 | 0.5000 | 0.0916 | 0.3304 |
| fold_1 | `logistic_regression` | 0.1378 | 0.5766 | 0.0974 | 0.3443 |
| fold_1 | `random_forest` | 0.1269 | 0.5335 | 0.0913 | 0.3289 |
| fold_2 | `population_prior` | 0.0819 | 0.5000 | 0.0753 | 0.2839 |
| fold_2 | `logistic_regression` | 0.1305 | 0.6085 | 0.0748 | 0.2802 |
| fold_2 | `random_forest` | 0.1334 | 0.5709 | 0.0827 | 0.3124 |
| fold_3 | `population_prior` | 0.0921 | 0.5000 | 0.0837 | 0.3075 |
| fold_3 | `logistic_regression` | 0.1652 | 0.6446 | 0.0821 | 0.2979 |
| fold_3 | `random_forest` | 0.1519 | 0.6240 | 0.0822 | 0.2998 |

## Macro fold summary

| Model | Mean AP | Mean ROC-AUC | Mean Brier | Mean log loss | AP uplift vs prior | Gate |
|---|---:|---:|---:|---:|---:|---|
| `population_prior` | 0.0919 | 0.5000 | 0.0835 | 0.3073 | 0.0000 | reference |
| `logistic_regression` | 0.1445 | 0.6099 | 0.0848 | 0.3075 | 0.0526 | pass |
| `random_forest` | 0.1374 | 0.5761 | 0.0854 | 0.3137 | 0.0455 | pass |

## Frozen ranking decision

The frozen ranking-selection rule selected **`logistic_regression`**.

At least one non-constant candidate passed the pre-registered
usefulness gate, so the project now has development evidence of
appointment-level ranking signal on the synthetic benchmark. This is
not final-test evidence and does not establish operational or clinical
usefulness.

Ranking and probability calibration are intentionally separate. Brier
score and log loss shown here describe the uncalibrated candidates and do
not alter the ranking choice under the frozen contract.

## Protected-test state

- `final_test_target_accessed = false`
- `final_test_probabilities_generated = false`

## Next R2 step

The next batch applies the already frozen calibration chronology to the
selected ranking candidate using only the calibration fit/evaluation
windows. No final-test target or probability vector is permitted during
that step.
