# Documentation

The documentation is organized in methodological order from source intake to
the implemented canonical dataset:

1. [Data intake](data_intake.md) — Records the selected immutable synthetic
   source tables, provenance, and intake boundaries.
2. [Prediction time](prediction_time.md) — Defines the 24-hour prediction
   horizon and active appointment cohort.
3. [Target definition](target_definition.md) — Defines the binary no-show
   outcome and handling of later cancellations and rescheduling.
4. [Feature eligibility](feature_eligibility.md) — Establishes the
   prediction-time predictor allowlist and leakage restrictions.
5. [Temporal split](temporal_split.md) — Defines chronological partitions,
   label maturity, and test-set protection.
6. [Dataset construction](dataset_construction.md) — Documents the implemented
   reproducible build, canonical schema, safeguards, outputs, and validation.

The repository currently implements dataset construction and its automated
contract tests. Exploratory analysis, preprocessing, modeling, calibration,
threshold selection, and final test evaluation have not started.
