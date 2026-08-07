# Version 2 Recovery Phase R3 Formal Closeout

## Status

**R3 is complete and formally closed.**

The final reporting package at commit `08c274a` was exact-commit
CI-sealed before this closeout. The closeout does not modify the frozen model,
calibration, protected-test probability vector, final-test evaluation, policy
sensitivity, or application decision.

## R3 checkpoint chain

- R3 execution contract: `acd39ae`
- persisted frozen model pipeline: `330ae87`
- pre-test diagnostics: `722ebf9`
- target-free protected-test probability seal: `7fc3194`
- one-time protected final-test evaluation: `9d5c3e1`
- final reporting package: `08c274a`

## Exact final-reporting CI seal

- workflow: `CI`
- event: `pull_request`
- run ID: `31197218122`
- conclusion: `success`
- exact sealed commit: `08c274a`

## Frozen artifact identities

- final reporting manifest SHA-256:
  `15f47f11e0378376baf7a2f5c520beb389ed8952f3ac161321548aeb35ca64b3`
- final reporting summary SHA-256:
  `76ed771871b696b4e0cd0c262b7d13f3bb03d03a187edd84c40934f1d2bfbbf7`
- final evaluation manifest SHA-256:
  `c8a2158bf98c4230bc66180d66dc4e4e88f8e3fff8b2ce0538fd58f4cf29a2af`
- sealed protected-test probability vector SHA-256:
  `7a4af37da40c1515a6ee567dd12861b57cf08e2e65b516e3c7e3d2aa65237126`

## Protected final-test result

- rows: `4,343`
- positives: `358`
- prevalence: `0.082431498964`
- Average Precision: `0.147157699598`
- ROC-AUC: `0.630029510104`
- Brier score: `0.076205245350`
- log loss: `0.282623134261`
- calibration intercept: `-0.962052389503`
- calibration slope: `0.689033702318`

The frozen population-prior Brier score was `0.075687251878`. Because the
model Brier score was slightly worse than that baseline, the pre-frozen
all-requirements gate did not authorize an appointment-level risk
demonstration.

## Frozen application decision

```text
selected_app_type = transparent_model_evaluation_dashboard
```

No final-test operational threshold was selected.

## Post-test immutability

```text
target_access_count = 1
target_reaccess_performed = false
model_refit_performed = false
calibration_change_performed = false
final_test_threshold_selected = false
post_test_model_tuning_permitted = false
```

Any subsequent work must treat the R3 model, probability vector, protected-test
evaluation, app decision, and reporting evidence as frozen. R4 may build the
transparent evaluation dashboard and portfolio-facing application artifacts,
but it may not tune the model or reinterpret the protected test as a model
selection set.

All performance claims remain scoped to the synthetic longitudinal benchmark.
