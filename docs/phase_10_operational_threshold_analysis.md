# Phase 10 - Operational Threshold Analysis
## 1. Purpose and scope
Phase 10 evaluates the operational implications of the probability model
selected in Phase 09.
The selected `calibration_prior` assigns the same probability to every
appointment. It therefore provides no appointment-level ranking and cannot
produce a conventional threshold trade-off curve.
This phase:
- enumerates every distinct threshold-induced policy state;
- fixes the threshold rule as `probability >= threshold`;
- records the exact temporal-validation confusion counts;
- derives an ex-ante break-even cost boundary;
- reports validation prevalence only as a replay audit;
- selects no cost, effectiveness, threshold, or operational policy;
- leaves the test population untouched.
This is an operational feasibility analysis. It is not threshold optimization,
a deployment decision, or evidence that the selected model is suitable for
appointment-level targeting.
## 2. Upstream probability contract
[Phase 09](phase_09_probability_calibration.md) selected
`calibration_prior` under the declared Brier-score and log-loss rule.
The selected probability is the calibration-population prevalence:
```text
0.12521739130434784
```
Every validation appointment receives exactly that value. The probability
vector therefore contains one unique value and no appointment-level ordering.
The Phase 10 production modules are:
- `src/modeling/operational_threshold.py`;
- `src/modeling/operational_cost.py`.
Their focused tests are:
- `tests/test_modeling_operational_threshold.py`;
- `tests/test_modeling_operational_cost.py`.
## 3. Population and information boundaries
Phase 10 reuses the Phase 09 chronological data contract.
The relevant populations are:
| Population | Rows | Positives | Negatives | Role |
|---|---:|---:|---:|---|
| Calibration | 1,150 | 144 | 1,006 | Ex-ante prevalence |
| Validation | 1,541 | 192 | 1,349 | Replay audit |
The calibration target determines:
- the selected constant probability;
- the ex-ante prevalence;
- the ex-ante break-even boundary.
The validation target determines only:
- validation confusion counts;
- validation replay prevalence;
- the replay break-even boundary.
Validation labels do not alter the ex-ante boundary. Test features and targets
are not accessed.
## 4. Threshold rule
The fixed classification rule is:
```text
predict intervention when probability >= threshold
```
Because every probability equals `0.12521739130434784`, all thresholds at or
below that value produce the same prediction vector: every appointment is
flagged.
The smallest representable floating-point value above the selected probability
is:
```text
0.12521739130434786
```
That threshold and every larger threshold produce the second prediction vector:
no appointment is flagged.
The equality boundary is intentional. A threshold exactly equal to the
probability belongs to `intervene_all` because the rule uses `>=`.
## 5. Distinct operational policy states
Only two distinct threshold-induced policy states exist.
| Policy | Threshold | Alerted | Rate | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `intervene_all` | 0.12521739130434784 | 1,541 | 1.0 | 0 | 1,349 | 0 | 192 |
| `intervene_none` | 0.12521739130434786 | 0 | 0.0 | 1,349 | 0 | 192 | 0 |
A dense threshold sweep would duplicate one of these two states and would not
add decision information.
No intermediate alert volume, ranking cutoff, top-k policy, or selective
appointment-level intervention can be obtained from the selected probability
contract.
## 6. Cost and effectiveness contract
Let:
- `intervention_cost` be the cost of intervening on one appointment;
- `no_show_cost` be the loss associated with one no-show;
- `effectiveness` be the fraction of true no-shows prevented by intervention;
- `N` be the number of appointments;
- `positives` be the number of true no-shows.
The two policy costs are:
```text
intervene_all =
    N * intervention_cost
    + positives * (1 - effectiveness) * no_show_cost
intervene_none =
    positives * no_show_cost
```
`intervene_all` is economically preferred when:
```text
intervention_cost / no_show_cost
<
prevalence * effectiveness
```
At equality, the two policies have equal modeled cost. Above the boundary,
`intervene_none` is preferred under this simplified contract.
Absolute currency values are unnecessary for the break-even comparison. Only
the intervention-to-no-show cost ratio is required.
## 7. Break-even sensitivity
The declared effectiveness grid is fixed at:
```text
0.00, 0.25, 0.50, 0.75, 1.00
```
The ex-ante boundary uses calibration prevalence only. The validation boundary
is reported solely as a retrospective replay audit.
| Effectiveness | Ex-ante boundary | Validation replay |
|---:|---:|---:|
| 0.00 | 0.000000000000000 | 0.000000000000000 |
| 0.25 | 0.031304347826087 | 0.031148604802077 |
| 0.50 | 0.062608695652174 | 0.062297209604153 |
| 0.75 | 0.093913043478261 | 0.093445814406230 |
| 1.00 | 0.125217391304348 | 0.124594419208306 |
For example, even under assumed `100%` effectiveness, `intervene_all` is
preferred ex ante only when:
```text
intervention_cost / no_show_cost
<
0.125217391304348
```
At `50%` effectiveness, the ex-ante ratio must be below:
```text
0.062608695652174
```
These are break-even boundaries, not selected business assumptions.
## 8. Ex-ante decision boundary
The production decision boundary is based on calibration information:
```text
ex_ante_boundary =
    calibration_prevalence * effectiveness
```
The validation replay is:
```text
validation_replay_boundary =
    validation_prevalence * effectiveness
```
The validation replay is retained to show how the boundary would have appeared
under the later validation prevalence. It is not fed back into the ex-ante
boundary and is not used to select a policy.
The calibration and validation prevalences are close:
| Source | Prevalence |
|---|---:|
| Calibration | 0.125217391304348 |
| Validation replay | 0.124594419208306 |
This proximity does not create appointment-level discrimination. It only shows
similar aggregate prevalence across the two periods.
## 9. Interpretation
The selected probability reference is useful as an aggregate probability
baseline but not as a targeting model.
Its operational consequences are strict:
- it cannot rank appointments;
- it cannot isolate a high-risk subgroup;
- it cannot support an intermediate alert capacity;
- it cannot produce a meaningful precision-recall operating curve;
- it reduces thresholding to `intervene_all` versus `intervene_none`.
The economic analysis therefore asks whether a universal intervention could be
justified under specified cost and effectiveness assumptions.
Phase 10 does not claim that universal intervention is appropriate. It records
the exact assumptions under which that policy would break even.
## 10. Leakage and reproducibility controls
The implementation enforces the following controls:
- calibration and validation targets must be exact pandas Series objects;
- targets must be non-empty and contain both binary classes;
- target indexes must be unique and disjoint;
- inputs are copied and are not mutated;
- the selected probability uses calibration target only;
- validation prevalence affects only replay outputs;
- the effectiveness grid and result ordering are deterministic;
- repeated evaluation returns exact equivalent results;
- no test rows are exposed;
- no estimator, probability, policy, or cost artifact is serialized.
The two policy states are ordered as:
1. `intervene_all`;
2. `intervene_none`.
The result dictionaries and table columns also have fixed declared order.
## 11. What Phase 10 does not select
Phase 10 intentionally selects none of the following:
- a monetary intervention cost;
- a monetary no-show cost;
- an intervention effectiveness estimate;
- a final threshold;
- an alert capacity;
- an operational policy;
- an appointment-level treatment assignment.
Those inputs require clinical, operational, financial, and preferably causal
evidence that is not contained in the current synthetic observational dataset.
## 12. Limitations
The break-even model is deliberately simple.
It assumes:
- one constant intervention cost per alerted appointment;
- one constant no-show cost;
- one effectiveness value shared by all appointments;
- no intervention harm or patient burden;
- no capacity constraints;
- no heterogeneous treatment effect;
- no interaction between repeated interventions and future behavior.
The analysis does not estimate causal intervention effectiveness. Historical
appointment outcomes alone cannot identify how many no-shows a new intervention
would prevent.
The synthetic dataset also cannot establish real clinical or financial cost
values.
## 13. Reproduction
The threshold-state tests can be run with:
```powershell
python -m pytest tests/test_modeling_operational_threshold.py -q
```
The break-even tests can be run with:
```powershell
python -m pytest tests/test_modeling_operational_cost.py -q
```
The broader focused Phase 10 contract can be reproduced with:
```powershell
python -m pytest `
    tests/test_modeling_calibration_data.py `
    tests/test_modeling_calibration_validation.py `
    tests/test_modeling_operational_threshold.py `
    tests/test_modeling_operational_cost.py `
    -q
```
The repository test environment uses an isolated pytest base directory on
Windows to avoid unrelated system temporary-directory permission failures.
## 14. Phase conclusion
Phase 10 confirms that the selected Phase 09 probability contract has exactly
two operational threshold states.
It also establishes the deterministic ex-ante rule:
```text
intervene_all preferred when
intervention_cost / no_show_cost
<
calibration_prevalence * effectiveness
```
No threshold or operational policy is selected. The result is a documented
feasibility boundary rather than a deployable decision system.
## 15. Next-phase boundary
A later phase may define an approved operational decision only after receiving
externally justified cost, capacity, intervention-effectiveness, and governance
inputs.
Final pre-test fitting, model persistence, deployment, and untouched test-set
evaluation remain outside Phase 10.
