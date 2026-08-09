# Data

This repository uses **fully synthetic** dental appointment data. No real
patient records are included.

## Version 2 raw benchmark

`data/raw/v2/` contains the frozen longitudinal synthetic benchmark and its
integrity manifest. Key files are `patients.csv`, `dentists.csv`,
`appointments.csv`, and `v2_synthetic_benchmark.manifest.json`.

The frozen raw dataset fingerprint is
`d9fdfa1a93091fd15bc34a62d655aef313966e2603d901350a6bd969b4e3c1bf`.

## Version 2 processed data

`data/processed/v2/` contains the target-free modeling feature artifact and
manifest. Historical features follow the strict as-of rule: status information
may contribute only when it was available before the current prediction time.

The frozen processed dataset contains 21,755 rows and 32 predictors. Its CSV
SHA-256 is
`08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53`.

## Protected target boundary

The processed Version 2 feature export is target-free. Protected 2027 final-test
outcomes were accessed exactly once under the frozen R3 protocol after a
prewritten probability vector existed. They must not be reused for feature,
model, calibration, or threshold changes.

See [`../docs/v2.0.0_recovery_plan.md`](../docs/v2.0.0_recovery_plan.md) and
[`../docs/v2_r5_clean_reproduction_evidence.md`](../docs/v2_r5_clean_reproduction_evidence.md).
