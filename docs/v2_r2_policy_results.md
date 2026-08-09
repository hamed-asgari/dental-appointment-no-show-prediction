# Version 2 Phase R2 Policy-Sensitivity Results

## Status

The first registered Version 2 R2 policy-sensitivity batch was executed only
after the policy mechanics and implementation had been frozen.

This report records sensitivity results. It does **not** select an operational
clinic threshold or assert a validated monetary intervention policy.

## Frozen upstream choices

- ranking model: `logistic_regression`
- calibration method: `uncalibrated`
- policy decision time: `2027-01-01T00:00:00`
- policy execution specification SHA-256:
  `4e12c2db3a95ed096040e558b567106a7569a07f3fdec8fb2d28570dedc90863`
- calibration manifest SHA-256:
  `5b2e701753a2d0e0d2f9a7efaddf46b2f316643e66dd8c11727373927c8a5d7a`

## Policy-selection population

- rows: `1063`
- positive no-shows: `92`
- positive rate: `0.086547507056`
- registered scenarios: `16`

## Registered capacity sensitivity

| Capacity | Selected | Threshold | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 5% | 53 | 0.173082792343 | 0.0943396226415 | 0.054347826087 |
| 10% | 106 | 0.149200302397 | 0.132075471698 | 0.152173913043 |
| 20% | 212 | 0.118593479612 | 0.108490566038 | 0.25 |

Capacity membership is rank-based. The confusion counts are identical across
cost ratios within each fixed capacity, while the registered relative scenario
cost changes with the false-negative:false-positive cost ratio.

## Registered cost-threshold sensitivity

| FN:FP ratio | Threshold | Selected | Precision | Recall | Scenario cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1:1 | 0.5 | 1 | 0 | 0 | 93 |
| 2:1 | 0.333333333333 | 3 | 0.333333333333 | 0.0108695652174 | 184 |
| 5:1 | 0.166666666667 | 63 | 0.111111111111 | 0.0760869565217 | 481 |
| 10:1 | 0.0909090909091 | 388 | 0.105670103093 | 0.445652173913 | 857 |

The `scenario_cost` values are unitless relative classification costs under the
pre-registered formula. They are not clinic-economic estimates.

## Interpretation boundary

The 10% capacity scenario has the highest precision among the three registered
capacity fractions, while the 20% capacity scenario captures a larger share of
positive no-shows. This is a descriptive sensitivity result, not an operational
threshold recommendation.

The registered cost-threshold scenarios illustrate the expected trade-off:
lower thresholds select more appointments and increase recall while also
increasing intervention volume.

No single operational threshold is selected in R2.

## Reproducibility and protection

The first batch was replayed independently and all four policy outputs were
byte-identical.

- policy manifest SHA-256:
  `33391bc9295c6a9d93bb8797e8e3836fb9af45062f1b431a4debc9bc7822e4a4`
- final-test target accessed: `false`
- final-test probabilities generated: `false`

The protected 2027 final test remains unopened and reserved for Phase R3.
