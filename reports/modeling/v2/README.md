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
are added in the next R2 batch.
