# Version 2 Chronological Calibration Artifacts

This directory contains the Phase R2 calibration evaluation generated under the
frozen pre-metric contract in `configs/v2_model_development.json`.

The selected rolling-origin ranking model is refit once at the frozen base-fit
time. Sigmoid and isotonic calibrators are fit only on the July-August 2026
calibration-fit window, and all three probability methods are evaluated only on
the September 2026 calibration-evaluation window with strictly mature labels.

The committed artifacts contain no protected `final_test` rows, no protected
final-test targets, and no final-test probabilities.

`calibration_manifest.json` records frozen input identities, runtime versions,
row counts, selected calibration method, protected-test state, and SHA-256
identities for the generated result artifacts.
