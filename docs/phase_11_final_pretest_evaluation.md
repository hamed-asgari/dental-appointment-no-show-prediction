# Phase 11: Final Pre-test Fit and Test Evaluation
## 1. Purpose
Phase 11 performs the final fit of the selected probability contract and
evaluates it once on the previously untouched chronological test population.
The selected model remains:
```text
calibration_prior
```
This phase does not reopen model comparison, calibration-method selection,
feature design, threshold selection, cost selection, or operational-policy
selection.
## 2. Pre-registered sequence
The execution order was fixed before test outcomes were accessed:
1. validate the canonical dataset and raw-data hashes;
2. isolate the final pre-test target population;
3. expose test rows only through target-free metadata;
4. fit the selected prior from the pre-test target;
5. generate and seal the complete test probability vector;
6. access the authentic test target;
7. evaluate the frozen probability vector using the declared metrics.
The working tree was clean and `HEAD` was fixed at commit `71117bc` before the
one-time evaluation began.
## 3. Final fitting population
The final pre-test fitting population contains all eligible train and
validation rows.
| Quantity | Value |
|---|---:|
| Rows | 5,223 |
| Positives | 626 |
| Negatives | 4,597 |
| Included splits | `train`, `validation` |
| Test rows included | 0 |
The fitted probability is the mean of the approved pre-test target:
```text
626 / 5223 = 0.11985448975684472
```
No feature matrix, preprocessing pipeline, classifier, or calibrator is
required by the selected prior model.
## 4. Frozen test probability contract
Every test appointment receives the same probability:
```text
0.11985448975684472
```
The frozen vector contains:
| Quantity | Value |
|---|---:|
| Test rows | 1,563 |
| Unique probability values | 1 |
| Probability dtype | `float64` |
| Probability name | `no_show_probability` |
The probability vector was created before authentic test outcomes were
accessed.
## 5. Probability-vector seal
The execution produced the following SHA-256 audit fingerprint:
```text
6dec44195ebe3d7e94ffb27300c09a5cde7b1fcab24cebd51ccb06b26462abfe
```
The seal combines:
- the selected probability-model name;
- the exact fitted probability representation;
- the test row count;
- the indexed Pandas hash of the frozen probability Series.
This fingerprint records the evaluated vector for this execution environment.
It is not a serialized model artifact or a cross-version persistence format.
## 6. Chronological test population
The one-time test evaluation contains:
| Quantity | Value |
|---|---:|
| Rows | 1,563 |
| Positives | 194 |
| Negatives | 1,369 |
| Prevalence | 0.12412028150991683 |
The test prevalence exceeds the fitted prior by:
```text
0.12412028150991683 - 0.11985448975684472
= 0.0042657917530721096
```
This is approximately 0.43 percentage points.
## 7. Final test metrics
The pre-registered probability metrics are:
| Metric | Test result |
|---|---:|
| Average precision | 0.12412028150991683 |
| ROC AUC | 0.5 |
| Brier score | 0.1087326342070964 |
| Log loss | 0.375140145229552 |
No additional metric was introduced after test outcomes were observed.
## 8. Ranking interpretation
The final probability vector has one unique value.
Therefore:
- no appointment is ranked above another appointment;
- Average Precision equals the observed test prevalence;
- ROC AUC is 0.5;
- no appointment-level discrimination is available;
- no intermediate threshold policy can be formed.
The test evaluation confirms the same structural limitation observed during
validation and operational-threshold analysis.
## 9. Probability-quality interpretation
The final prior is close to the aggregate test prevalence.
The Brier score and log loss therefore describe the quality of a constant
population-level probability estimate. They do not demonstrate useful
appointment-level prediction.
The result supports only the narrow conclusion that the recent aggregate
no-show rate remained reasonably similar across the final pre-test and test
periods in this synthetic dataset.
## 10. Operational interpretation
Phase 11 selects none of the following:
- an intervention threshold;
- an alert volume;
- an operational policy;
- intervention costs;
- no-show costs;
- intervention effectiveness;
- deployment capacity.
The constant probability still produces only two threshold states under the
fixed `probability >= threshold` rule:
1. intervene on every appointment;
2. intervene on no appointments.
The test result does not make either policy operationally preferable.
## 11. Leakage controls
The implemented controls ensure that:
- pre-test fitting uses only eligible train and validation targets;
- test targets cannot affect the fitted prior;
- test metadata contains no target column;
- the test probability vector is generated before test-target access;
- test target and probability indexes must align exactly;
- fitting and evaluation inputs are copied and not mutated;
- model, threshold, and policy selection remain closed.
## 12. Test-set status after evaluation
The chronological test set was untouched through the end of Phase 10 and
through final probability generation in Phase 11.
Its outcomes were then accessed for the declared one-time evaluation.
The existing test period must therefore no longer be treated as untouched for
future model development, model selection, calibration selection, feature
engineering, or threshold selection.
Any future redesign requires a newly declared untouched period, external data,
or another prospectively frozen evaluation policy.
## 13. Persistence and deployment boundary
Phase 11 does not serialize:
- an estimator;
- a preprocessor;
- a probability vector;
- a threshold;
- an operational policy;
- a cost model;
- a deployment package.
The probability seal is an audit record only.
The repository remains an analytical and methodological project rather than a
production deployment system.
## 14. Final conclusion
The selected final model is a deterministic population prior:
```text
calibration_prior = 0.11985448975684472
```
Its one-time chronological test evaluation produced:
```text
test prevalence = 0.12412028150991683
average precision = 0.12412028150991683
ROC AUC = 0.5
Brier score = 0.1087326342070964
log loss = 0.375140145229552
```
The model provides an aggregate probability estimate but no appointment-level
ranking or targeting value.
Phase 11 does not justify deployment, threshold selection, or an operational
intervention policy.
