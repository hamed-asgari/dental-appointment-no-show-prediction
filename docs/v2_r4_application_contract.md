# Version 2 Recovery Phase R4 Application Contract

## Status

**Frozen before any R4 application implementation.**

Phase R3 is formally closed and CI-sealed. R4 is therefore a presentation and
portfolio-delivery phase only. It may not change the frozen model, calibration,
feature set, protected-test probabilities, final-test evaluation, registered
policy grid, or application decision.

## Frozen application decision

The only authorized Version 2 application type is:

```text
transparent_model_evaluation_dashboard
```

An appointment-level risk demonstration is not authorized because the
pre-frozen all-requirements gate failed its Brier-score condition on the
protected final test.

R4 must not reinterpret the protected test as a new model-selection,
calibration, threshold-selection, or product-gating dataset.

## Framework and entry point

The application framework is Streamlit.

The primary entry point must be:

```text
app/streamlit_app.py
```

The documented launch command must be:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The Streamlit dependency must be pinned reproducibly in the repository's
runtime dependency files before the application is considered complete.

## Authorized data sources

The dashboard may read only committed, already-opened R3 reporting artifacts
and static documentation needed for display.

Primary machine-readable sources:

```text
reports/modeling/v2/final_reporting/final_reporting_summary.json
reports/modeling/v2/final_reporting/final_reporting_manifest.json
```

Approved analytical figures:

```text
reports/figures/v2_final_precision_recall_curve.png
reports/figures/v2_final_calibration_curve.png
reports/figures/v2_final_capacity_tradeoff.png
```

Frozen SHA-256 identities:

```text
final_reporting_summary.json
76ed771871b696b4e0cd0c262b7d13f3bb03d03a187edd84c40934f1d2bfbbf7

final_reporting_manifest.json
15f47f11e0378376baf7a2f5c520beb389ed8952f3ac161321548aeb35ca64b3

v2_final_precision_recall_curve.png
e706d92fa13e09c5fda7da2fcd0f6a8c55332d524ae0e221d370ef792ebd7b95

v2_final_calibration_curve.png
eb98404d809b0ab691decb2cb10a3d0c3b2a495cbef8637d75a468e86f25834f

v2_final_capacity_tradeoff.png
965ee9e77e35b05ae380d017b88a7fa952fd6b82bf608cb8fc71b66cf01827a1
```

The application must not read raw protected targets, regenerate the sealed
probability vector, call a protected-target accessor, or load the persisted
model for appointment scoring.

## Prohibited behavior

R4 must not provide any of the following:

- patient or appointment data-entry fields for individualized prediction;
- appointment-level no-show probabilities generated at runtime;
- patient ranking or "high risk" labels;
- a user-adjustable operating threshold;
- a threshold recommendation derived from the protected final test;
- model refitting, recalibration, feature changes, or hyperparameter changes;
- claims of clinical validation, operational effectiveness, or real-patient
  applicability.

No call path in the Streamlit application may invoke
`load_verified_v2_final_test_targets`, any `allow_test=True` path, or the
protected final-test evaluator.

## Required dashboard sections

The application must contain all of the following sections.

### 1. Scope and disclaimer

Display prominently that:

- all records are fully synthetic;
- the project is educational/research portfolio work;
- the model is not validated for clinical or operational use;
- the dashboard does not provide individualized patient risk estimates.

### 2. Evidence-based application decision

Show the four pre-frozen gate checks:

- Average Precision absolute uplift versus population prior;
- ROC-AUC;
- Brier score versus population prior;
- log loss versus population prior.

The failed Brier condition must be visually explicit. The dashboard must state
that this failure is why the appointment-level risk demonstration was rejected.

### 3. Protected final-test performance

Display the frozen protected-test population and metrics:

- 4,343 rows;
- 358 positives;
- prevalence;
- Average Precision;
- ROC-AUC;
- Brier score;
- log loss;
- calibration intercept;
- calibration slope.

Also display the frozen population-prior baseline where relevant.

### 4. Precision-recall evidence

Display the committed precision-recall figure and explain that the model shows
useful ranking/discrimination on the synthetic benchmark without implying that
its probabilities are suitable for individualized operational decisions.

### 5. Calibration evidence

Display the committed calibration figure and explain the observed calibration
limitations, including the protected-test Brier comparison and the calibration
intercept/slope.

### 6. Registered capacity sensitivity

Display the committed capacity-sensitivity figure for the pre-registered
5%, 10%, and 20% capacity scenarios.

This section is descriptive only. It must state that no operational threshold
or clinic intervention policy was selected or validated.

### 7. Pre-test interpretation

Display the committed pre-test permutation-importance ranking from the final
reporting summary. This is descriptive and must not be presented as post-test
feature selection.

### 8. Limitations and external validation

Summarize at minimum:

- synthetic-data-only evidence;
- no real-world external validation;
- no validated clinical intervention effect;
- no validated operational threshold or cost ratio;
- subgroup analysis does not establish fairness;
- further real-world validation would be required before any clinical or
  operational consideration.

## Input validation and integrity checks

Because this application is an evaluation dashboard rather than an
individualized risk calculator, "input validation" means validation of the
committed reporting inputs and application state.

Before rendering evidence, the application must validate:

- required files exist;
- reporting summary and manifest parse successfully;
- expected schema fields are present;
- `selected_app_type` equals
  `transparent_model_evaluation_dashboard`;
- `target_reaccess_performed` is false;
- `model_refit_performed` is false;
- `calibration_change_performed` is false;
- `final_test_threshold_selected` is false;
- `post_test_model_tuning_permitted` is false;
- frozen artifact SHA-256 identities match the values in this contract.

A validation failure must stop evidence rendering and show a clear integrity
error rather than silently falling back or recomputing results.

## Implementation boundaries

The app should separate data loading/validation from Streamlit rendering so the
integrity logic can be unit tested without starting a web server.

Recommended structure:

```text
app/
  README.md
  dashboard_data.py
  streamlit_app.py
```

`dashboard_data.py` should contain pure loading, hash verification, schema
validation, and display-data preparation functions.

`streamlit_app.py` should contain the Streamlit presentation layer and must not
contain model-fitting or protected-target-access logic.

## Testing requirements

R4 application tests must cover at least:

- valid committed artifacts load successfully;
- missing or altered artifacts are rejected;
- wrong `selected_app_type` is rejected;
- prohibited post-test state flags are rejected;
- the application source contains no protected-target accessor call;
- no appointment-level scoring form or operational-threshold control is
  introduced;
- the documented launch command remains valid.

Tests may use temporary copied/modified reporting artifacts. They must not call
the protected-target accessor.

## Portfolio artifacts

Before R4 closes, the repository must include:

- the implemented Streamlit application;
- application tests;
- updated root README;
- updated `app/README.md`;
- committed screenshots under `reports/screenshots/`;
- committed final analytical figures under `reports/figures/`;
- a portfolio architecture diagram;
- final limitations and external-validation plan.

Screenshots must depict the actual committed application state, not a mockup.

## R4 acceptance boundary

R4 can close only when:

- the app launches from a clean Python 3.12 environment;
- the dashboard behavior matches this contract;
- repository tests and CI are green;
- screenshots are committed;
- launch instructions are reproducible;
- stale Version 1 statements about the app being absent are removed or clearly
  contextualized as historical;
- no protected-target re-access, model refit, recalibration, feature change,
  or final-test threshold selection occurs.

All Version 2 performance claims remain scoped to the synthetic benchmark.
