# Version 2 protected final-test probability seal

This directory contains the pre-outcome Phase R3 probability vector for the
protected 2027 `final_test` partition.

The vector was generated from the committed frozen Logistic Regression pipeline
using only the target-free frozen processed feature dataset. Protected
final-test targets were not loaded, joined, inferred, or used to compute any
performance metric.

Artifacts:

- `final_test_probabilities.csv` — exactly `appointment_id` and
  `no_show_probability`, preserving the frozen processed-feature appointment
  order for all 4,343 protected-test rows.
- `final_test_probability_manifest.json` — input identities, appointment-order
  SHA-256, probability-vector SHA-256 and byte size, and the protected-target
  gate state.

This checkpoint is not permission to inspect protected targets. The probability
vector must first be committed, pushed, and CI-green. Target access remains a
separate explicit one-time R3 operation.
