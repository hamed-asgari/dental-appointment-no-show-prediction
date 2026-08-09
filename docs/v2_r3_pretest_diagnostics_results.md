# Version 2 Phase R3 Pre-test Diagnostic Results

## Scope

These results implement the pre-test diagnostic portion of the frozen
`docs/v2_r3_execution_contract.md`.

The analysis uses only the already permitted `policy_selection` population and
the persisted frozen `logistic_regression` + `uncalibrated` pipeline. It does
not score the protected 2027 `final_test` partition and does not access its
targets.

## Overall policy-selection behavior

- rows: `1063`
- positives: `92`
- prevalence: `0.086548`
- Average Precision: `0.111898`
- ROC-AUC: `0.569460`
- Brier score: `0.080009`
- log loss: `0.297917`
- mean predicted probability: `0.086090`

These are pre-test diagnostic results, not final model performance claims.

## Permutation importance

Permutation importance uses Average Precision, 20 repeats, random state
`20260807`, and raw-feature permutation before preprocessing. It is descriptive
only and may not drive feature selection or refitting.

- 1. `patient_prior_no_show_rate_smoothed`: mean AP decrease `0.014009`, SD `0.004679`
- 2. `reminder_sent_by_prediction_time`: mean AP decrease `0.012289`, SD `0.004489`
- 3. `visit_type`: mean AP decrease `0.009597`, SD `0.012781`
- 4. `visit_type_prior_no_show_rate_smoothed`: mean AP decrease `0.005512`, SD `0.005470`
- 5. `patient_prior_no_show_count`: mean AP decrease `0.003142`, SD `0.002329`
- 6. `patient_prior_completed_count`: mean AP decrease `0.001830`, SD `0.000976`
- 7. `patient_mean_prior_booking_lead_days`: mean AP decrease `0.001644`, SD `0.000785`
- 8. `patient_history_available`: mean AP decrease `0.001543`, SD `0.001534`
- 9. `patient_days_since_last_known_status_update`: mean AP decrease `0.001410`, SD `0.000808`
- 10. `dentist_prior_attendance_count`: mean AP decrease `0.001097`, SD `0.004476`

A positive value means that permuting the feature reduced Average Precision on
this population. Negative values are retained rather than converted into
importance claims.

## First-time versus repeat patients

- `first_time`: n=`84`, positives=`11`, prevalence=`0.130952`, AP=`nan`, ROC-AUC=`nan`, Brier=`nan`, log loss=`nan`
- `repeat`: n=`979`, positives=`81`, prevalence=`0.082737`, AP=`0.111156`, ROC-AUC=`0.565619`, Brier=`0.076880`, log loss=`0.289290`

`patient_history_available=false` defines the first-time/no-history cohort and
`true` defines the repeat/history-available cohort.

## Subgroup support

The frozen quantitative-support rule requires at least 100 rows and at least 10
positive no-shows.

- subgroup rows evaluated: `21`
- quantitatively supported rows: `18`
- insufficient-support rows: `3`

Unsupported groups remain in the artifact with metrics withheld. They do not
trigger pooling, feature changes, calibration changes, or threshold changes.

## Error and capacity diagnostics

Per-row absolute probability error, Brier contribution, and log-loss
contribution are committed in `row_error_analysis.csv`.

Registered capacity summaries:

- 5%: selected=`53`, threshold=`0.173083`, precision=`0.094340`, recall=`0.054348`, mean absolute error=`0.275369`
- 10%: selected=`106`, threshold=`0.149200`, precision=`0.132075`, recall=`0.152174`, mean absolute error=`0.275981`
- 20%: selected=`212`, threshold=`0.118593`, precision=`0.108491`, recall=`0.250000`, mean absolute error=`0.234944`

These are descriptive replays of the already registered 5%, 10%, and 20%
capacity fractions. They do not select a single operational threshold.

## Reproducibility and boundary

The diagnostic manifest records SHA-256 identities for every generated artifact
and binds them to the frozen R3 config, R2 policy manifest, persisted model
manifest, and persisted pipeline.

```text
single_operational_threshold_selected = false
final_test_probabilities_generated = false
final_test_target_accessed = false
```

The next protected-test step remains unauthorized until these diagnostics are
committed and CI-green.
