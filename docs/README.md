# Documentation
The documentation is organized in methodological order from source intake
through implemented operational threshold analysis:
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
9. [Phase 08 tree-based model comparison](
   phase_08_tree_based_comparison.md) - Documents the fixed Random Forest
   comparator, four-model temporal validation, model selection,
   interpretation, and limitations.
10. [Phase 09 probability calibration](
    phase_09_probability_calibration.md) - Documents the chronological
    base-fit and calibration populations, frozen-estimator calibration,
    recent-prior reference, probability-quality selection, reliability
    diagnostics, interpretation, and limitations.
11. [Phase 10 operational threshold analysis](
    phase_10_operational_threshold_analysis.md) - Documents the two distinct
    threshold states induced by the selected recent-prior reference, ex-ante
    break-even sensitivity, validation replay, decision boundaries, and
    limitations.
12. [Phase 11 final pre-test fit and test evaluation](
    phase_11_final_pretest_evaluation.md) - Documents the leakage-safe final
    prior fit, sealed test probability vector, one-time chronological test
    audit, ranking interpretation, test-set status, and deployment boundary.
13. [Version 2 data generation and evaluation policy](
    v2_data_generation_and_evaluation_policy.md) - Freezes the longitudinal
    synthetic benchmark, named random streams, prediction-time contract,
    historical-feature boundary, rolling-origin development schedule, and
    protected 2027 final test.
14. [Version 2 synthetic generator contract](
    v2_synthetic_generator_contract.md) - Documents the deterministic
    longitudinal generator, public schemas, reminder timing, hidden-effect
    boundary, frozen raw export, integrity manifest, and hashes.
15. [Version 2 historical feature contract](
    v2_historical_feature_contract.md) - Freezes the strict as-of event
    boundary, approved patient and aggregate history features, smoothing,
    cold-start behavior, feature-target separation, and leakage tests.
16. [Version 2 recovery plan](v2.0.0_recovery_plan.md) - Tracks the audited
    recovery phases and release-completion checklist.
The repository currently implements canonical dataset construction,
exploratory analysis, deterministic baseline preprocessing and fitting, a
fixed Random Forest comparator, chronological probability calibration,
temporal-validation calibration-candidate selection, deterministic
threshold-state enumeration, break-even sensitivity analysis, final pre-test
prior fitting, and one-time chronological test probability evaluation.
The selected Phase 09 candidate is `calibration_prior`. It provides the best
declared validation Brier score and log loss, but assigns the same probability
to every appointment and therefore provides no appointment-level ranking.
Phase 10 selects no cost values, effectiveness, threshold, or operational
policy. Phase 11 fits the selected prior on the eligible pre-test population
and performs the declared one-time chronological test probability audit. The
final probability remains constant and provides no ranking. The evaluated test
period is no longer untouched. Model persistence, deployment, and operational
policy selection remain outside the repository scope.
