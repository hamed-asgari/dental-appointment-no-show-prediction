# Documentation
The documentation is organized in methodological order from source intake
through implemented tree-based model comparison:
1. [Data intake](data_intake.md) - Records the selected immutable synthetic
   source tables, provenance, and intake boundaries.
2. [Prediction time](prediction_time.md) - Defines the 24-hour prediction
   horizon and active appointment cohort.
3. [Target definition](target_definition.md) - Defines the binary no-show
   outcome and handling of later cancellations and rescheduling.
4. [Feature eligibility](feature_eligibility.md) - Establishes the
   prediction-time predictor allowlist and leakage restrictions.
5. [Temporal split](temporal_split.md) - Defines chronological partitions,
   label maturity, and test-set protection.
6. [Dataset construction](dataset_construction.md) - Documents the
   reproducible build, canonical schema, safeguards, outputs, and validation.
7. [Phase 06 exploratory data analysis](
   phase_06_exploratory_data_analysis.md) - Records leakage-safe development
   EDA and target-free train-to-validation drift findings.
8. [Phase 07 baseline modeling](phase_07_baseline_modeling.md) - Documents
   modeling populations, preprocessing, estimators, temporal validation,
   metrics, model selection, and limitations.
9. [Phase 08 tree-based model comparison](phase_08_tree_based_comparison.md) -
   Documents the fixed Random Forest comparator, four-model temporal
   validation, model selection, interpretation, and limitations.
The repository currently implements canonical dataset construction,
exploratory analysis, deterministic baseline preprocessing and fitting, a
fixed Random Forest comparator, and temporal-validation model comparison.
Probability calibration, operational threshold or cost analysis, final
pre-test fitting, and untouched test-set evaluation remain future phases.
