# Changelog
This file records notable repository changes by release version.
## [1.0.0]
### Added
- Immutable synthetic source-data integrity contracts.
- Prediction-time, target-definition, feature-eligibility, and temporal-split
  safeguards.
- Reproducible canonical analytical dataset construction.
- Leakage-safe exploratory analysis and deterministic artifact generation.
- Deterministic baseline models and a fixed Random Forest comparator.
- Chronological probability calibration and candidate selection.
- Operational threshold-state enumeration and break-even sensitivity analysis.
- Leakage-safe final pre-test prior fitting.
- Pre-registered one-time chronological test probability evaluation.
- Python 3.12 package metadata, locked dependencies, and Windows CI.
- Citation metadata and an explicit line-ending contract.
### Final declared result
The selected final model is `calibration_prior`. It fits a single pre-test
probability of `0.11985448975684472` and assigns that value to every test
appointment.
The one-time chronological test audit reports:
- 1,563 test appointments;
- prevalence and average precision of `0.12412028150991683`;
- ROC AUC of `0.5`;
- Brier score of `0.1087326342070964`;
- log loss of `0.375140145229552`.
The constant probability provides no appointment-level ranking.
### Scope boundaries
Version 1.0.0 does not select an operational cost, intervention effectiveness,
decision threshold, or operational policy. It does not serialize a trained
production model or implement deployment. The evaluated test period is no
longer an untouched benchmark for future development.
No software license, DOI, or release archive identifier is declared.
