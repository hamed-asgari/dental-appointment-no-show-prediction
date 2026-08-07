# Configuration

`v2_synthetic_benchmark.json` is the frozen, machine-readable configuration
for the Version 2 longitudinal synthetic benchmark.

The file is governed by
[`docs/v2_data_generation_and_evaluation_policy.md`](../docs/v2_data_generation_and_evaluation_policy.md).
Changing its generator version, seed, population sizes, date boundaries,
random-stream names, or protected evaluation windows creates a new benchmark
version and invalidates previously frozen hashes and protected-test claims.

`v2_model_development.json` is the frozen machine-readable contract for
recovery Phase R2. It fixes the 32-feature roles, preprocessing, candidate
estimators and hyperparameters, rolling-origin selection rules, calibration
chronology, policy-sensitivity scenarios, and the prohibition on protected
final-test target access before any recovered Version 2 model metric is run.

Changing this file after metrics are inspected is a contract amendment and
must be documented explicitly rather than treated as ordinary tuning.
