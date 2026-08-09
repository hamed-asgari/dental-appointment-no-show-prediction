# Version 2 processed feature dataset

This directory contains the deterministic, target-free Version 2 feature
artifact derived from the frozen synthetic raw benchmark.

Files:

- `v2_feature_dataset.csv` — 21,755 eligible appointment rows, audit keys,
  32 approved model features, evaluation partition, and label-availability
  timestamp. It contains no target or source outcome columns.
- `v2_feature_dataset.manifest.json` — exact artifact hash, schema, dtypes,
  partition counts, source identities, code hashes, and processed fingerprint.

The protected 2027 final-test rows are present only as features and metadata.
The manifest declares `target_included=false` and
`final_test_target_accessed=false`.

Final-test targets may be accessed only through the gated accessor after an
exact final-test probability CSV has already been written and validated.
