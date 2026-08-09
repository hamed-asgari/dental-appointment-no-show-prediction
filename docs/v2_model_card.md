# Version 2 Model Card

## Model summary

- model: `logistic_regression`
- calibration: `uncalibrated`
- approved feature count: `32`
- frozen base fit time: `2026-07-01T00:00:00`
- base training population: `10,921` rows / `978` positives
- persisted artifact:
  `models/v2/frozen_logistic_pipeline.joblib`

The model and preprocessing pipeline were frozen before protected-test scoring.
No estimator refit, feature selection, hyperparameter change, or calibration
change was performed after the probability-vector seal.

## Intended use

This model is an educational artifact for a fully synthetic longitudinal dental
appointment benchmark. It supports methodological evaluation of leakage-safe
appointment no-show prediction.

It is **not validated for clinical or operational use** and must not be used to
make decisions about real patients, staff, scheduling, or interventions.

## Data and evaluation design

Version 2 uses prediction-time-safe features with strict historical as-of
eligibility. The protected chronological final-test window is 2027-01-01
through 2027-12-31.

The complete target-free 4,343-row final-test probability vector was generated,
SHA-256 sealed, committed, pushed, and CI-green before the protected target was
opened exactly once.

## Protected final-test performance

- rows: `4343`
- positives: `358`
- prevalence: `0.082431`
- Average Precision: `0.147158`
- ROC-AUC: `0.630030`
- Brier score: `0.076205`
- log loss: `0.282623`
- calibration intercept: `-0.962052`
- calibration slope: `0.689034`
- mean predicted probability: `0.113190`

Frozen population-prior baseline:

- probability: `0.089552`
- Average Precision: `0.082431`
- ROC-AUC: `0.500000`
- Brier score: `0.075687`
- log loss: `0.284987`

## App decision

The pre-frozen appointment-level risk-demo gate required all four conditions.
Observed results were:

- AP uplift requirement:
  `true`
- ROC-AUC requirement:
  `true`
- Brier requirement:
  `false`
- log-loss requirement:
  `true`

The Brier score (`0.076205`) is slightly worse than the frozen
population-prior baseline (`0.075687`). Because every
gate condition had to pass, the evidence-based R4 application type is:

```text
transparent_model_evaluation_dashboard
```

This decision does not authorize model, calibration, feature, or threshold
changes.

## Interpretation

Pre-test permutation importance on `policy_selection` ranked
`patient_prior_no_show_rate_smoothed` highest, followed by
`reminder_sent_by_prediction_time` and `visit_type`. Permutation importance is
descriptive only and was prohibited from driving feature selection.

The final-test calibration intercept is negative and the calibration slope is
below one. Together with mean predicted probability above observed prevalence,
this supports presenting calibration limitations prominently rather than
presenting probabilities as individualized operational risk estimates.

## Registered policy sensitivity

The 5%, 10%, and 20% capacity fractions and 1:1, 2:1, 5:1, and 10:1
false-negative-to-false-positive cost ratios are descriptive replays of the
pre-registered policy grid. No single operational threshold was selected.

## Reproducibility

From the repository root, the final analytical summary and three key figures
are regenerated from committed post-access evaluation artifacts with one
command:

```powershell
.\.venv\Scripts\python.exe -m src.modeling.v2_final_reporting --overwrite
```

This command does not call the protected-target accessor, refit the model,
change calibration, replace the sealed probability vector, or select a
final-test threshold.

Generated analytical figures:

- `reports/figures/v2_final_precision_recall_curve.png`
- `reports/figures/v2_final_calibration_curve.png`
- `reports/figures/v2_final_capacity_tradeoff.png`

Machine-readable reporting artifacts:

- `reports/modeling/v2/final_reporting/final_reporting_summary.json`
- `reports/modeling/v2/final_reporting/final_reporting_manifest.json`

## Limitations

- all records are synthetic;
- performance is established only on the frozen synthetic benchmark;
- no clinical effectiveness or intervention benefit has been established;
- subgroup diagnostics are support-limited and do not establish fairness;
- probability quality did not beat the frozen prior on Brier score;
- no operational threshold, clinic cost ratio, or intervention policy has been
  validated;
- external validation on independent real-world data would be required before
  any clinical or operational consideration.
