# Version 2 frozen model persistence

This directory contains the persisted Version 2 pipeline authorized by the
frozen Phase R3 execution contract.

The persisted estimator is the exact R2-selected `logistic_regression`
pipeline fitted at `2026-07-01T00:00:00` on the frozen base-training
partitions only. The selected calibration method is `uncalibrated`, so no
calibration transform is applied.

Artifacts:

- `frozen_logistic_pipeline.joblib` — loadable sklearn preprocessing/model pipeline.
- `frozen_logistic_pipeline.metadata.json` — fit identity, feature allowlist,
  upstream frozen hashes, runtime versions, and policy replay evidence.
- `frozen_logistic_pipeline.manifest.json` — SHA-256 and byte-size identities
  for the loadable pipeline and metadata.

The artifact is verified by reloading it and reproducing the already permitted
R2 policy-selection probabilities within the frozen CSV precision tolerance.
This persistence step does not score the protected 2027 final-test rows and
does not access protected final-test targets.
