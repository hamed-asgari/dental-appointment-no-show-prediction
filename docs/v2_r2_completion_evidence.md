# Version 2 Phase R2 Completion Evidence

## Status

**Phase R2 is complete on the recovery branch.**

This closeout records the frozen modeling, calibration, and policy-sensitivity
evidence before Phase R3 begins. It does not open or score the protected 2027
final test.

## Implementation checkpoints

- model-development contract: `8eadbaa`
- rolling-origin ranking implementation/results: `d06929e`
- chronological calibration implementation/results: `891221d`
- policy execution mechanics freeze: `1891ade`
- pre-outcome policy engine freeze: `e5192c6`
- policy-sensitivity result freeze: `db0e02b`

## Frozen identities

- model configuration SHA-256:
  `0b39dbe9b15c64579a81e7b9dbeaad5e5a6694fc5066698fbcfd4623a1bd1dd6`
- model-development contract SHA-256:
  `735953523db15e36b82bacfb022915c3eff0c4f4329f16c72877dc32f8ff597f`
- rolling-origin manifest SHA-256:
  `e575b10835645d3a643c396803cfff21f5c1c1cdad9b988ee07037ef045beb45`
- calibration manifest SHA-256:
  `5b2e701753a2d0e0d2f9a7efaddf46b2f316643e66dd8c11727373927c8a5d7a`
- policy execution specification SHA-256:
  `4e12c2db3a95ed096040e558b567106a7569a07f3fdec8fb2d28570dedc90863`
- policy manifest SHA-256:
  `33391bc9295c6a9d93bb8797e8e3836fb9af45062f1b431a4debc9bc7822e4a4`

## Frozen R2 decisions

The rolling-origin ranking gate selected `logistic_regression`; the population
prior was not used as a fallback.

The chronological calibration comparison retained `uncalibrated` under the
pre-registered Brier-score, log-loss guardrail, and simplicity rule.

The policy-sensitivity batch evaluated the complete registered 16-scenario
capacity/cost grid over 1,063 mature `policy_selection` rows containing 92
positive no-shows. It did not select a single operational threshold.

## Reproducibility evidence

At the policy-result implementation checkpoint `db0e02b`:

- full local test suite: 1640 passed
- GitHub CI run: 31176713663
- GitHub CI job: 92860198254
- GitHub CI status: successful

The first policy batch was replayed independently and all four generated policy
outputs were byte-identical.

## Protected final-test state

Throughout R2:

```text
final_test_target_accessed = false
final_test_probabilities_generated = false
```

No protected 2027 final-test metric has been inspected.

The protected final-test window remains:

```text
2027-01-01 <= prediction_time < 2028-01-01
```

The final-test target accessor remains gated. Phase R3 must preserve the
prewritten-and-sealed probability-vector requirement before any protected
target access.

## Gate into Phase R3

Phase R3 may now begin interpretation, error analysis, persistence, and
reproducibility work using only already permitted development, calibration,
and policy-selection evidence.

Protected final-test evaluation remains a later explicit R3 gate. No R2 result
may be used to retune the frozen model, calibration choice, feature set, or
policy-sensitivity grid.
