# Phase 09 - Probability Calibration
## 1. Purpose and scope
Phase 09 evaluates whether the fixed Random Forest comparison model from
Phase 08 can produce useful probability estimates after chronological
calibration.
The phase establishes:
- a fixed chronological split within mature development data;
- one Random Forest fitted only on the earlier base-fit population;
- sigmoid and isotonic calibration fitted only on the later calibration
  population;
- a leakage-safe constant-prior reference fitted from the calibration target;
- deterministic validation evaluation;
- selection using probability-quality metrics;
- explicit diagnostics for calibration direction and probability resolution.
This phase does not implement threshold optimization, cost analysis, final
pre-test fitting, model persistence, deployment, or test-set evaluation.
## 2. Population boundaries
The mature development population is divided at the fixed boundary:
```text
2024-11-01 00:00:00
```
The boundary was fixed before calibration-validation metrics were inspected.
| Population | Rows | Positives | Negatives | Phase 09 role |
|---|---:|---:|---:|---|
| Base fit | 2,520 | 288 | 2,232 | Fit Random Forest |
| Calibration | 1,150 | 144 | 1,006 | Fit calibrators and recent prior |
| Temporal validation | 1,541 | 192 | 1,349 | Candidate selection |
| Test | Not exposed | Not exposed | Not exposed | Untouched |
The base-fit population precedes the calibration population, and both precede
the temporal-validation population.
Validation labels are used only for evaluation. They are never supplied to the
base estimator, sigmoid calibrator, isotonic calibrator, or calibration-prior
estimation.
Test features and targets remain outside the Phase 09 data contract.
## 3. Feature and preprocessing contract
Phase 09 preserves the exact ten prediction-time features and preprocessing
pipeline approved in Phases 07 and 08.
Numerical features:
- `planned_duration_min`
- `booking_lead_time_hours`
- `approximate_age_at_prediction`
- `patient_registration_tenure_days`
- `dentist_tenure_days`
Categorical features:
- `visit_type`
- `booking_channel`
- `scheduled_weekday`
- `scheduled_hour`
- `scheduled_month`
The preprocessing contract is unchanged. The Phase 09 Random Forest receives
a fresh preprocessor fitted only on the base-fit population.
The Phase 09 uncalibrated model is not numerically comparable to the Phase 08
Random Forest fit because Phase 08 used all 3,670 mature development rows,
whereas Phase 09 fits the base estimator on only 2,520 earlier rows.
## 4. Base estimator
The fixed base estimator remains:
```text
RandomForestClassifier(
    n_estimators=500,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=42,
    n_jobs=1,
)
```
No hyperparameter search, class-weight experiment, resampling, feature
redesign, or validation-driven estimator modification is performed.
## 5. Calibration candidates
The base Random Forest is fitted exactly once on base-fit features and targets.
Two deep-copied fitted pipelines are wrapped in `FrozenEstimator` before
calibration. This prevents the calibration population from refitting the base
pipeline.
The Random Forest calibration candidates are:
1. `random_forest_uncalibrated`
2. `random_forest_sigmoid`
3. `random_forest_isotonic`
The sigmoid and isotonic candidates use:
```text
CalibratedClassifierCV(
    estimator=FrozenEstimator(fitted_base_pipeline),
    method=<sigmoid or isotonic>,
    cv=None,
    n_jobs=None,
    ensemble=False,
)
```
Only calibration features and targets are supplied when the calibrators are
fitted.
## 6. Calibration-prior reference
Initial comparison among only the three Random Forest variants selected
`random_forest_isotonic`.
A subsequent reliability audit showed that isotonic calibration produced
almost constant probabilities. This revealed that the comparison lacked a
necessary leakage-safe reference representing the recent event rate.
The added reference is:
```text
calibration_prior
```
It predicts the calibration-population prevalence for every validation row:
```text
144 / 1150
= 0.125217391304348
```
This reference uses only `calibration_target`. It does not use validation
labels, validation features, or test data during fitting.
The complete declared result order is:
1. `calibration_prior`
2. `random_forest_uncalibrated`
3. `random_forest_sigmoid`
4. `random_forest_isotonic`
## 7. Evaluation and selection rules
The primary selection metric is:
```text
brier_score
```
Lower values are better.
The secondary metric is:
```text
log_loss
```
Lower values are better.
Exact ties are resolved by declared result order.
Average precision and ROC-AUC are reported only as discrimination audits. They
do not select the Phase 09 winner.
No classification threshold is searched, optimized, selected, or evaluated in
this phase.
## 8. Temporal-validation results
Validation prevalence is:
```text
192 / 1541
= 0.124594419208306
```
| Candidate | AP | ROC-AUC | Brier | Log loss | Mean probability |
|---|---:|---:|---:|---:|---:|
| `calibration_prior` | 0.124594 | 0.500000 | 0.109071 | 0.375982 | 0.125217 |
| `random_forest_uncalibrated` | 0.123084 | 0.507137 | 0.118920 | 0.404801 | 0.205155 |
| `random_forest_sigmoid` | 0.118851 | 0.492863 | 0.109574 | 0.378405 | 0.112533 |
| `random_forest_isotonic` | 0.123552 | 0.495622 | 0.109101 | 0.376116 | 0.125799 |
The official Phase 09 selection is:
```text
calibration_prior
```
It has the lowest validation Brier score and the lowest validation log loss.
## 9. Isotonic comparison with the selected reference
Relative to `calibration_prior`, isotonic produces:
```text
Brier difference:
0.109101376880843 - 0.109071038004684
= 0.000030338876159
Log-loss difference:
0.376115519200288 - 0.375981958827441
= 0.000133560372847
Average-precision difference:
0.123552124857081 - 0.124594419208306
= -0.001042294351225
ROC-AUC difference:
0.495621756856931 - 0.500000000000000
= -0.004378243143069
```
Positive Brier and log-loss differences mean isotonic is worse. Negative
average-precision and ROC-AUC differences mean isotonic also fails to improve
the discrimination audits.
## 10. Reliability and mapping diagnostics
The uncalibrated Random Forest produces 163 unique validation probabilities.
Its mean predicted probability is `0.205155094094744`, which exceeds validation
prevalence by `0.080560674886437`.
Sigmoid calibration preserves 163 unique validation probabilities but learns
a strictly decreasing mapping:
```text
Spearman(raw probability, sigmoid probability) = -1.0
```
The base score has calibration-period ROC-AUC `0.453518472498343`. The
decreasing sigmoid mapping converts that value to its complement:
```text
1 - 0.453518472498343
= 0.546481527501657
```
That reversal does not generalize. On validation, sigmoid changes the raw
ROC-AUC from `0.507136845193971` to:
```text
0.492863154806029
```
Isotonic calibration preserves an increasing mapping, but it collapses the
score distribution:
- 2 unique probabilities on the calibration population;
- 3 unique probabilities on temporal validation;
- validation range from `0.124887690925427` to `0.135135135135135`;
- only one effective bin in the requested ten-quantile reliability audit.
Its fitted thresholds are:
```text
X thresholds:
[0.018, 0.290, 0.294, 0.396]
Y thresholds:
[
    0.124887690925427,
    0.124887690925427,
    0.135135135135135,
    0.135135135135135,
]
```
The favorable isotonic Brier score therefore comes primarily from estimating
the overall event rate rather than producing useful appointment-level risk
resolution.
## 11. Interpretation
Phase 09 does not establish that the Random Forest produces useful calibrated
appointment-level probabilities.
The selected `calibration_prior` is a recent prevalence estimate. It produces
the best declared probability-quality metrics, but assigns the same
probability to every validation appointment.
The selection therefore establishes:
- recent prevalence is more reliable than the evaluated Random Forest
  probabilities;
- sigmoid calibration is unstable because it learned a reversed relationship;
- isotonic calibration removes nearly all probability resolution;
- no evaluated Random Forest calibration method adds validated probability
  value beyond the recent event rate.
The Phase 09 winner is a probability-quality reference, not a production
risk-stratification model.
## 12. Leakage and reproducibility controls
Phase 09 verifies that:
- the calibration boundary is fixed and chronological;
- base-fit, calibration, and validation indexes are disjoint;
- the base pipeline is fitted only on base-fit rows;
- fitted base pipelines are frozen before calibration;
- calibration labels cannot refit the base model;
- the recent-prior reference uses only calibration targets;
- validation labels are never supplied to fitting functions;
- test features and targets are never exposed;
- candidate and result order are deterministic;
- Brier, log-loss, and tie-breaking rules are fixed;
- repeated fits reproduce probabilities exactly;
- input DataFrames and Series remain unchanged;
- predicted classes, shapes, bounds, and probability sums are validated;
- production modules perform no filesystem writes or serialization.
All estimator fitting uses the pinned single-process environment.
## 13. Limitations
- All data are synthetic and project-specific.
- Calibration and validation each cover one chronological window.
- The approved features contain no patient-history or dentist-history
  aggregates.
- The base Random Forest is fitted on fewer rows than in Phase 08.
- Only sigmoid and isotonic calibration are evaluated.
- No cross-validated or rolling temporal calibration design is evaluated.
- Sigmoid calibration learns a decreasing relationship in the calibration
  period.
- Isotonic calibration produces almost constant probabilities.
- The selected reference provides no appointment-level ranking.
- No operational threshold or cost analysis is performed.
- The test period remains untouched.
- No clinical or operational utility is established.
## 14. Reproduction
From the repository root, run:
```powershell
.\.venv\Scripts\python.exe -m pytest `
    -p no:cacheprovider `
    --basetemp .venv\pytest-phase09-reproduction `
    tests/test_modeling_evaluation.py `
    tests/test_modeling_calibration_data.py `
    tests/test_modeling_calibration.py `
    tests/test_modeling_calibration_validation.py `
    -q
```
At this implementation checkpoint, the focused Phase 09 suite reports:
```text
72 passed
```
The full repository suite must be rerun before Phase 09 is finalized.
## 15. Next-phase boundary
Operational threshold and cost analysis must remain separate from probability
calibration.
Because `calibration_prior` assigns one probability to every appointment, it
cannot prioritize or rank appointments. A later threshold analysis must not
reinterpret its favorable Brier score as evidence of actionable
risk stratification.
The next phase may evaluate operational decision rules and explicitly conclude
that no useful threshold exists under the selected probability contract.
Final pre-test fitting, persistence, deployment, and test evaluation remain
out of scope. Test evaluation may occur only after the modeling and operational
decision workflow is frozen.
