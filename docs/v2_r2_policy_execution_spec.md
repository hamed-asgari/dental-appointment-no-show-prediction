# Version 2 Phase R2 Policy-Sensitivity Execution Specification
## Status
**Frozen before any Version 2 `policy_selection` metric is computed.**
This document supplements, but does not modify, the frozen Version 2
model-development contract and machine-readable configuration. It resolves
deterministic implementation details needed to execute the already registered
policy-sensitivity scenarios reproducibly.
The pre-policy checkpoint is commit `891221d`.
## Frozen upstream decisions
The policy analysis must use:
- ranking model: `logistic_regression`
- calibration method: `uncalibrated`
- base-estimator refit time: `2026-07-01T00:00:00`
- no estimator refit after calibration
- policy decision time: `2027-01-01T00:00:00`
- policy partition: `policy_selection`
No feature, estimator, calibration method, benchmark window, capacity fraction,
cost ratio, or policy scenario may be changed in response to policy results.
## Policy evaluation population
Policy sensitivity uses only rows belonging to `policy_selection` whose labels
satisfy:
```text
label_available_at < 2027-01-01T00:00:00
```
The protected `final_test` partition is excluded from feature scoring, target
access, threshold analysis, and metric computation throughout R2.
## Frozen probability source
The policy-selection probabilities are produced by the already selected
`logistic_regression` base estimator using the same frozen preprocessing and
base-fit population used by chronological calibration.
Because `uncalibrated` was selected, no sigmoid or isotonic transformation is
applied.
The base estimator is not refit using calibration-evaluation or
policy-selection outcomes.
## Deterministic capacity scenarios
The registered capacity fractions are:
```text
0.05
0.10
0.20
```
For an evaluation population of size `N`, capacity count is:
```text
k = floor(N * capacity_fraction)
```
`k` must be at least one for every registered scenario.
Rows are ordered by:
1. descending `no_show_probability`;
2. ascending `appointment_id` as the deterministic tie-break.
Exactly the first `k` rows are selected.
The reported capacity threshold is the minimum predicted probability among the
selected `k` rows. Selection remains rank-based; when probabilities tie at the
boundary, the `appointment_id` tie-break determines membership.
## Deterministic cost-ratio scenarios
The registered false-negative:false-positive cost ratios are:
```text
1:1
2:1
5:1
10:1
```
False-positive cost is the unit cost `1`. For ratio `r`, false-negative cost is
`r`.
The registered probability threshold is:
```text
threshold = 1 / (1 + r)
```
A row is selected when:
```text
no_show_probability >= threshold
```
Scenario cost is the unitless relative classification cost:
```text
scenario_cost = false_positive + r * false_negative
```
It is not a monetary or validated clinic-economic estimate.
## Capacity-cost sensitivity grid
Each registered capacity scenario is evaluated under each registered cost
ratio. This produces `3 x 4 = 12` capacity-cost rows.
The selected set and confusion counts are determined by the capacity rule and
therefore repeat across cost ratios for a fixed capacity. `scenario_cost`
changes with the registered cost ratio.
The four registered cost-derived probability thresholds are also evaluated as
a separate cost-threshold family.
The complete deterministic policy table therefore contains:
```text
12 capacity-cost scenarios
4 cost-threshold scenarios
16 total scenarios
```
## Required metrics
Every scenario records at least:
- scenario family
- capacity fraction when applicable
- false-negative:false-positive cost ratio
- threshold
- selected count
- selected fraction
- precision
- recall
- true positive
- false positive
- true negative
- false negative
- scenario cost
Precision is reported as `0.0` when no positive predictions are selected.
## Interpretation boundary
These outputs are sensitivity analysis only.
No scenario is declared the operational policy, no threshold defaults to
`0.50`, and no clinic-economic claim is permitted without independently
validated operational assumptions.
Policy results must not be used to modify the frozen model, calibration choice,
feature set, benchmark, or scenario grid.
## Protected final-test boundary
During this R2 policy batch:
```text
final_test_target_accessed = false
final_test_probabilities_generated = false
```
The protected 2027 final test remains reserved for Phase R3.
