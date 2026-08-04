# Phase 07 - Baseline Modeling
## 1. Purpose and scope
Phase 07 implements deterministic, leakage-controlled baseline modeling on the
approved chronological development and validation populations.
The phase establishes:
- an explicit modeling-data contract;
- deterministic numerical and categorical preprocessing;
- one prevalence-only baseline and two logistic-regression baselines;
- threshold-free probability evaluation;
- a fixed `0.5` classification audit;
- deterministic model selection by average precision.
This phase does not implement calibration, threshold optimization, cost
analysis, final pre-test fitting, model persistence, deployment, or test-set
evaluation.
## 2. Population boundaries
| Population | Rows | Positives | Negatives | Phase 07 role |
|---|---:|---:|---:|---|
| Mature development | 3,670 | 432 | 3,238 | Preprocessing and fitting |
| Temporal validation | 1,541 | 192 | 1,349 | Evaluation after fitting |
| Maturity-excluded train | 12 | Not used | Not used | Excluded |
| Test | Not exposed | Not exposed | Not exposed | Untouched |
Development rows satisfy:
```text
split == "train"
development_fit_eligible == True
```
Validation rows satisfy:
```text
split == "validation"
```
Validation labels are never supplied to preprocessing or estimator fitting.
Test features and targets are not exposed by the Phase 07 modeling-data
contract.
## 3. Approved feature contract
The baseline uses exactly ten prediction-time features.
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
Identifiers, outcomes, split fields, maturity flags, post-event fields, and
evaluation-only fields are excluded.
## 4. Preprocessing
Each estimator receives a fresh preprocessing pipeline.
Numerical processing:
1. median imputation;
2. standard scaling.
Categorical processing:
1. most-frequent imputation with explicit `pd.NA` support;
2. one-hot encoding;
3. unknown categories ignored during later transforms;
4. sparse `float64` output.
The preprocessor is fitted only on mature development features. The authentic
fitted output contains 43 columns:
- 5 scaled numerical columns;
- 38 one-hot categorical columns.
The same fitted transformations are applied unchanged to validation features.
## 5. Baseline estimators
The declared model order is:
1. `dummy_prior`
2. `logistic_unweighted`
3. `logistic_balanced`
The dummy baseline uses:
```text
DummyClassifier(strategy="prior")
```
Both logistic baselines use:
```text
solver="liblinear"
max_iter=1000
random_state=42
```
The unweighted model uses `class_weight=None`. The balanced model uses
`class_weight="balanced"`.
No hyperparameter search, resampling, nonlinear estimator, calibration, or
threshold search occurs in this phase.
## 6. Evaluation metrics
The primary model-selection metric is:
```text
average_precision
```
The complete threshold-free metric set is:
- average precision;
- ROC-AUC;
- Brier score;
- log loss.
Average precision is primary because the positive no-show class is a minority
and the metric summarizes precision-recall ranking behavior. ROC-AUC provides
a complementary discrimination measure. Brier score and log loss evaluate
probability quality.
Higher values are better for average precision and ROC-AUC. Lower values are
better for Brier score and log loss.
## 7. Fixed-threshold audit
A fixed threshold of `0.5` is used only for descriptive audit metrics:
- precision;
- recall;
- F1;
- confusion-matrix counts;
- accuracy.
The threshold is not tuned, searched, optimized, or selected from validation
results. Accuracy is an audit metric rather than a selection criterion.
## 8. Temporal-validation results
| Model | Average precision | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| `dummy_prior` | 0.124594 | 0.500000 | 0.109118 | 0.376205 |
| `logistic_unweighted` | 0.120959 | 0.476028 | 0.112111 | 0.396686 |
| `logistic_balanced` | 0.120720 | 0.475866 | 0.183289 | 0.555112 |
Validation prevalence is `192 / 1541`, approximately `12.46%`. The constant
dummy baseline therefore has average precision equal to validation prevalence
and ROC-AUC equal to `0.5`.
Fixed `0.5` rate audit:
| Model | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| `dummy_prior` | 0.000000 | 0.000000 | 0.000000 | 0.875406 |
| `logistic_unweighted` | 0.000000 | 0.000000 | 0.000000 | 0.875406 |
| `logistic_balanced` | 0.106195 | 0.062500 | 0.078689 | 0.817651 |
Fixed `0.5` confusion-matrix audit:
| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| `dummy_prior` | 1,349 | 0 | 192 | 0 |
| `logistic_unweighted` | 1,349 | 0 | 192 | 0 |
| `logistic_balanced` | 1,248 | 101 | 180 | 12 |
## 9. Baseline selection
The selected model is:
```text
dummy_prior
```
Selection uses maximum validation average precision. Exact ties are resolved
by declared estimator order.
The selected dummy model is a reference baseline, not an operational model.
The result indicates that neither logistic baseline produced more useful
validation ranking than the prevalence-only reference in this temporal period.
No score inversion, post-hoc metric replacement, feature redesign, or
threshold adaptation is performed after observing validation results.
## 10. Leakage and reproducibility controls
Phase 07 verifies that:
- development fitting excludes validation and test rows;
- validation labels cannot affect fitted preprocessing or estimators;
- changes limited to test targets cannot affect fitted models;
- changes to maturity-excluded train targets cannot affect models;
- every estimator receives a fresh unfitted preprocessor;
- repeated fits reproduce learned states and probabilities;
- inputs remain unchanged;
- production modules perform no filesystem writes or serialization;
- imports and builder calls have no process-global side effects.
All tests run in the pinned single-process environment.
## 11. Interpretation
The logistic baselines do not outperform the dummy reference on average
precision, ROC-AUC, Brier score, or log loss.
The balanced logistic model shifts probabilities upward substantially and
performs especially poorly on probability-quality metrics. Class weighting
does not provide calibrated no-show probabilities in this observed setting.
These findings are limited to the synthetic dataset, approved feature set,
model specifications, and one future temporal-validation period.
## 12. Limitations
- All data are synthetic and project-specific.
- Validation covers one chronological window.
- The baseline has no patient- or dentist-history aggregates.
- Only linear logistic models and a dummy reference are evaluated.
- No calibration method is fitted or compared.
- The fixed `0.5` audit is not an operational threshold analysis.
- The test period remains untouched.
- No clinical or operational utility is established.
## 13. Reproduction
From the repository root, run:
```powershell
.\.venv\Scripts\python.exe -m pytest `
    tests/test_modeling_data.py `
    tests/test_modeling_preprocessing.py `
    tests/test_modeling_estimators.py `
    tests/test_modeling_evaluation.py `
    tests/test_modeling_validation.py
```
The authentic metrics, population boundaries, deterministic model selection,
input non-mutation, leakage guards, and side-effect contracts are asserted by
these test suites.
## 14. Next-phase boundary
Later work may address calibration, operational threshold and cost analysis,
or additional model development. Such work must continue to use only training
and validation for development decisions.
Test evaluation must occur only after the modeling and calibration workflow is
frozen. Any redesign prompted by test results would require disclosure and a
new untouched evaluation period.
