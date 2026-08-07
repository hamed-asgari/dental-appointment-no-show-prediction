# Version 2 Phase R3 Protected Final-test Probability Seal

## Status

**Target-free protected-test probability vector generated and committed for CI
sealing; protected targets remain closed.**

The vector was generated from the frozen persisted pipeline after the R3
execution contract, persistence checkpoint, and pre-test diagnostic checkpoint
were all committed.

## Frozen identities

- R3 execution config SHA-256:
  `c0b259a4bb81790a30fd6e2c2fd2495e10869d700ae783196c5eb055db46f7a5`
- persisted pipeline SHA-256:
  `301029bd5bee1ffe346fbf09dcc6ed4570b231458ba8a081f8a0f6bb544d9df0`
- persistence manifest SHA-256:
  `ca19d477e0590f40d1abbad869119b182d05e923b2d582df19c42473d2795856`
- pre-test diagnostics manifest SHA-256:
  `5a207b8a4984a203f64d1015c7a99b254db1108440dde71738abd3c936f9f8f2`

## Probability vector

- partition: `final_test`
- rows: `4343`
- schema: `appointment_id,no_show_probability`
- appointment order: `frozen_processed_feature_dataset_row_order`
- appointment-order SHA-256:
  `addb9ab672383b10976aec4eaa94f359bbbe4c0bf5b634b313265c659c9c3cd6`
- vector SHA-256:
  `7a4af37da40c1515a6ee567dd12861b57cf08e2e65b516e3c7e3d2aa65237126`
- vector size bytes: `114781`

The vector contains no target column and no target-derived metric.

## Protected gate

```text
probability_metrics_computed = false
final_test_probabilities_generated = true
final_test_target_accessed = false
single_operational_threshold_selected = false
```

Protected-target access remains blocked until this exact probability-vector
commit is pushed and CI-green. After that checkpoint, target access is a
separate explicit one-time R3 operation; the model and probability vector may
not be changed in response to observed labels.
