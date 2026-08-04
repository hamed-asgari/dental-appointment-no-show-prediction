# Phase 08 - Tree-Based Model Comparison
## 1. Purpose and scope
Phase 08 adds one deterministic tree-based comparison model to the approved
Phase 07 baselines and evaluates all four models on the same chronological
validation population.
The phase establishes:
- a fixed Random Forest comparison specification;
- reuse of the approved leakage-safe preprocessing contract;
- deterministic four-model validation orchestration;
- model selection using validation average precision;
- a descriptive fixed `0.5` classification audit;
- explicit interpretation of discrimination and probability quality.
This phase does not implement hyperparameter search, feature redesign,
resampling, calibration, threshold optimization, cost analysis, final pre-test
fitting, model persistence, deployment, or test-set evaluation.
## 2. Population boundaries
| Population | Rows | Positives | Negatives | Phase 08 role |
|---|---:|---:|---:|---|
| Mature development | 3,670 | 432 | 3,238 | Preprocessing and fitting |
| Temporal validation | 1,541 | 192 | 1,349 | Model comparison |
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
Validation labels are used only for evaluation. They are never supplied to
preprocessing or estimator fitting. Test features and targets remain outside
the modeling-data contract.
## 3. Approved feature and preprocessing contract
Phase 08 preserves the exact ten prediction-time features and preprocessing
pipeline approved in Phase 07.
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
Each estimator receives a fresh preprocessor fitted only on mature development
features. Authentic fitted preprocessing produces 43 columns:
- 5 scaled numerical columns;
- 38 one-hot categorical columns.
Reusing the same preprocessing contract isolates the comparison to estimator
behavior rather than introducing a simultaneous feature-processing change.
## 4. Tree-based comparator
The declared comparison model is:
```text
random_forest_unweighted
```
Its estimator specification is:
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
The configuration was fixed before the validation metrics were inspected.
Five hundred trees reduce simulation noise from the ensemble while
`random_state=42` and `n_jobs=1` preserve exact reproducibility in the pinned
environment. No validation-driven hyperparameter search or class-weight
variant is performed.
The Phase 07 baseline builder remains unchanged. The tree comparator and its
validation orchestration are implemented in separate modules so the historical
baseline contract is preserved.
## 5. Compared models
The complete declared model order is:
1. `dummy_prior`
2. `logistic_unweighted`
3. `logistic_balanced`
4. `random_forest_unweighted`
All models are fitted only on mature development rows and evaluated on the
same temporal-validation rows.
## 6. Evaluation and selection rules
The primary model-selection metric remains:
```text
average_precision
```
The threshold-free metric set is:
- average precision;
- ROC-AUC;
- Brier score;
- log loss.
Higher values are better for average precision and ROC-AUC. Lower values are
better for Brier score and log loss.
The selected model is the model with maximum validation average precision.
Exact ties are resolved by declared model order.
A fixed threshold of `0.5` is used only for descriptive audit metrics:
- precision;
- recall;
- F1;
- confusion-matrix counts;
- accuracy.
The threshold is not tuned, searched, optimized, or treated as an operational
decision rule.
## 7. Temporal-validation results
Threshold-free metrics:
| Model | Average precision | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| `dummy_prior` | 0.124594 | 0.500000 | 0.109118 | 0.376205 |
| `logistic_unweighted` | 0.120959 | 0.476028 | 0.112111 | 0.396686 |
| `logistic_balanced` | 0.120720 | 0.475866 | 0.183289 | 0.555112 |
| `random_forest_unweighted` | 0.133099 | 0.509009 | 0.114376 | 0.393564 |
The Random Forest improves average precision over the dummy baseline by:
```text
0.1330994923896915 - 0.1245944192083063
= 0.0085050731813852
```
Fixed `0.5` rate audit:
| Model | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| `dummy_prior` | 0.000000 | 0.000000 | 0.000000 | 0.875406 |
| `logistic_unweighted` | 0.000000 | 0.000000 | 0.000000 | 0.875406 |
| `logistic_balanced` | 0.106195 | 0.062500 | 0.078689 | 0.817651 |
| `random_forest_unweighted` | 0.000000 | 0.000000 | 0.000000 | 0.875406 |
Fixed `0.5` confusion-matrix audit:
| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| `dummy_prior` | 1,349 | 0 | 192 | 0 |
| `logistic_unweighted` | 1,349 | 0 | 192 | 0 |
| `logistic_balanced` | 1,248 | 101 | 180 | 12 |
| `random_forest_unweighted` | 1,349 | 0 | 192 | 0 |
## 8. Comparison selection
The selected comparison model is:
```text
random_forest_unweighted
```
Selection is based solely on maximum validation average precision.
This designation means that the fixed Random Forest specification ranks the
validation observations better than the three declared baselines according to
the primary metric. It does not establish operational readiness, clinical
utility, calibrated probability estimates, or an acceptable intervention
threshold.
No post-hoc score inversion, feature redesign, alternative hyperparameter
search, class-weight experiment, metric replacement, or threshold adaptation
is performed after observing the validation result.
## 9. Interpretation
The Random Forest produces the highest validation average precision, but the
absolute improvement over the prevalence-only baseline is small.
Its ROC-AUC of approximately `0.509` is only slightly above random ranking.
Its Brier score and log loss are worse than the dummy baseline, indicating
that the improved precision-recall ranking does not correspond to better
overall probability quality.
At the descriptive `0.5` threshold, the Random Forest predicts no positive
cases. This does not invalidate its threshold-free ranking result, but it
demonstrates that `0.5` is not an operationally useful decision rule for this
model in the observed validation period.
The Random Forest is therefore the Phase 08 comparison winner, not a frozen
production model.
## 10. Leakage and reproducibility controls
Phase 08 verifies that:
- the Phase 07 baseline contract remains unchanged;
- the tree estimator receives a fresh unfitted preprocessor;
- fitting uses only mature development features and targets;
- validation labels do not affect preprocessing or estimator fitting;
- test features and targets are never referenced;
- repeated fits reproduce probabilities, feature importances, and tree state;
- repeated four-model evaluations reproduce the metric table exactly;
- input DataFrames and Series remain unchanged;
- estimator order and tie-breaking are deterministic;
- predicted class order and probability bounds are validated;
- production modules perform no filesystem writes or serialization;
- imports, builders, evaluation, and failure paths have no process-global side
  effects.
All tests run in the pinned single-process environment.
## 11. Limitations
- All data are synthetic and project-specific.
- Validation covers one chronological window.
- The approved feature set contains no patient- or dentist-history aggregates.
- Only one fixed nonlinear estimator is added.
- No Random Forest hyperparameter search is performed.
- No resampling or class-weight alternative is compared.
- The average-precision improvement over dummy is small.
- ROC-AUC remains close to `0.5`.
- Probability quality is worse than the dummy baseline.
- No calibration method is fitted or compared.
- The fixed `0.5` audit is not an operational threshold analysis.
- The test period remains untouched.
- No clinical or operational utility is established.
## 12. Reproduction
From the repository root, run:
```powershell
.\.venv\Scripts\python.exe -m pytest `
    tests/test_modeling_data.py `
    tests/test_modeling_preprocessing.py `
    tests/test_modeling_estimators.py `
    tests/test_modeling_evaluation.py `
    tests/test_modeling_validation.py `
    tests/test_modeling_comparison.py `
    tests/test_modeling_comparison_validation.py
```
The authentic metrics, deterministic estimator state, model order, selection
rule, input non-mutation, leakage boundaries, and side-effect contracts are
asserted by these test suites.
The complete repository test suite at the end of this implementation slice
reports:
```text
1190 passed
```
## 13. Next-phase boundary
The next modeling decision should address probability calibration and
calibration-method comparison using only the approved development and
validation populations.
Operational threshold and cost analysis must remain separate from calibration
and must not reinterpret the fixed `0.5` descriptive audit as a selected
decision rule.
Test evaluation must occur only after model specification, calibration method,
and operational decision workflow are frozen. Any redesign prompted by test
results would require disclosure and a new untouched evaluation period.
