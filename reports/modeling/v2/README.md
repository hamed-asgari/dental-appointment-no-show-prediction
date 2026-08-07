# Version 2 Recovery Modeling Artifacts

This directory contains Version 2 recovery Phase R2 modeling outputs generated
under the frozen pre-metric contract in
`configs/v2_model_development.json` and
`docs/v2_model_development_and_selection_contract.md`.

The rolling-origin files cover only the three development-validation folds.
They may contain mature development and validation targets used for R2 model
selection. They must not contain protected `final_test` rows, protected final
test targets, or final-test probabilities.

`rolling_origin_manifest.json` records the frozen input identities, runtime
versions, selected ranking candidate, protected-test state, and SHA-256
identities of the generated result artifacts.

Chronological calibration artifacts are committed under `calibration/` and
follow the already frozen calibration chronology. Policy-sensitivity artifacts
are committed under `policy/` and follow the frozen execution mechanics in
`docs/v2_r2_policy_execution_spec.md`. The policy batch contains no protected
`final_test` rows or probabilities and does not select an operational
threshold.

Phase R3 pre-test interpretation, error-analysis, and subgroup artifacts are
committed under `diagnostics/`. They use only `policy_selection` plus the
persisted frozen pipeline, preserve the registered capacity fractions, and do
not score or open the protected 2027 final test.

Phase R3 contains the immutable sealed
`final_test/final_test_probabilities.csv` and the one-time opened evaluation
artifacts under `final_test/`. The protected target was accessed only after the
probability-vector commit was CI-green. The pre-frozen app gate selects
`transparent_model_evaluation_dashboard`; no final-test threshold or post-test model
change is permitted.
