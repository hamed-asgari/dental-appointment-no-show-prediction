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
16. [Version 2 processed data and protected target access](
    v2_processed_data_and_target_access.md) - Documents the deterministic
    target-free processed artifact, strict label-maturity access, final-test
    probability seal, hashes, and protected target gate.
17. [Version 2 data dictionary](v2_data_dictionary.md) - Defines the frozen
    raw schemas, all 32 approved model features, audit metadata, partitions,
    target-access boundary, and prohibited direct inputs.
18. [Phase R1 completion evidence](v2_r1_completion_evidence.md) - Records the
    frozen identities, implementation checkpoints, test/CI evidence, protected
    target state, and the gate into recovered modeling.
19. [Version 2 model development and selection contract](
    v2_model_development_and_selection_contract.md) - Freezes the exact
    candidate menu, preprocessing, rolling-origin ranking gate, calibration
    chronology, threshold-sensitivity grid, and protected-test prohibition
    before any recovered Version 2 model metric is computed.
20. [Phase R2 rolling-origin ranking results](
    v2_r2_rolling_origin_results.md) - Records the first recovered Version 2
    model metrics, strict mature fold populations, frozen ranking decision,
    artifact identities, and unchanged protected-test state.
21. [Phase R2 chronological calibration results](
    v2_r2_calibration_results.md) - Records the frozen chronology, calibration
    metrics, selection rule outcome, reliability-curve artifact, and unchanged
    protected-test state.
22. [Phase R2 policy execution specification](
    v2_r2_policy_execution_spec.md) - Freezes deterministic capacity,
    tie-breaking, cost-threshold, and scenario-accounting mechanics before
    policy outcomes are inspected.
23. [Phase R2 policy-sensitivity results](
    v2_r2_policy_results.md) - Records the registered capacity and cost
    sensitivity grid, deterministic replay, artifact identity, and explicit
    no-operational-threshold boundary.
24. [Phase R2 completion evidence](v2_r2_completion_evidence.md) - Records the
    frozen R2 checkpoints, test and CI evidence, artifact identities, and the
    protected-test gate into Phase R3.
25. [Phase R3 execution contract](v2_r3_execution_contract.md) - Freezes
    persistence, pre-test interpretation and subgroup diagnostics, the
    protected-test probability seal and one-time target gate, final reporting,
    and the evidence-based app decision rule.
26. [Phase R3 pre-test diagnostic results](
    v2_r3_pretest_diagnostics_results.md) - Records policy-selection
    permutation importance, error analysis, first-time versus repeat-patient
    behavior, subgroup support, and registered-capacity diagnostics before the
    protected final-test seal.
27. [Phase R3 protected final-test probability seal](
    v2_r3_final_test_probability_seal.md) - Records the target-free 4,343-row
    probability vector, appointment-order identity, SHA-256 seal, and the
    mandatory commit-plus-CI gate before protected-target access.
28. [Phase R3 final chronological test results](
    v2_r3_final_test_results.md) - Records the one-time protected evaluation,
    frozen population-prior comparison, registered policy replay, and the
    pre-frozen evidence-based R4 app decision.
29. [Version 2 model card](v2_model_card.md) - Summarizes the frozen
    Logistic Regression pipeline, protected-test evidence, calibration limits,
    app decision, reproducibility command, and synthetic-data scope.
30. [Version 2 recovery plan](v2.0.0_recovery_plan.md) - Tracks the audited
    recovery phases and release-completion checklist.
The Version 1 methodology through its one-time chronological test audit remains
an immutable audit checkpoint. Its selected `calibration_prior` assigns the
same probability to every appointment and therefore provides no
appointment-level ranking; that previously examined Version 1 test period is
not reused for recovery model selection.

Version 2 recovery Phases R1 and R2 are complete on the recovery branch.
Phase R2 completed its frozen rolling-origin ranking comparison, chronological
calibration evaluation, and policy-sensitivity analysis under pre-outcome
contracts. Phase R3 now has a frozen execution contract, a persisted frozen
model pipeline, completed pre-test interpretation/error/subgroup diagnostics,
and a one-time protected 2027 final-test evaluation performed only after the
4,343-row probability vector was exact-commit CI sealed. The pre-frozen app
decision gate selects `transparent_model_evaluation_dashboard`; no post-test model,
calibration, feature, or threshold tuning is permitted.
