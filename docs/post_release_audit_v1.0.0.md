# Audit Report — Dental Appointment No-show Prediction v1.0.0

## Executive verdict

The release is **not a broken or empty repository**. It contains a substantial methodological and software core: immutable synthetic inputs, target and prediction-time contracts, leakage controls, chronological splitting, reproducible dataset construction, EDA code, baseline and Random Forest evaluation, probability calibration, operational break-even analysis, final chronological test evaluation, extensive tests, and CI.

However, it is **not complete relative to the approved Project Charter and original portfolio target**. The release was finalized after silently narrowing its scope. The missing items include the Streamlit demonstration, model persistence or a reusable inference pipeline, portfolio screenshots/committed analytical outputs, and a meaningful model-interpretation/error-analysis layer.

The final selected model is a constant population prior, not an appointment-level risk model. It assigns the same probability to every appointment and has no ranking ability.

## What was verified

### Repository contents

The tagged archive contains 97 files and substantial implementation under:

- `src/data/`
- `src/analysis/`
- `src/modeling/`
- `tests/`
- `docs/`
- `.github/workflows/`

### Reproduced data contract

The analytical dataset builds successfully in memory from the tagged raw files:

- 6,786 eligible appointments
- 820 no-shows
- 5,966 negatives
- 3,682 nominal training rows
- 1,541 validation rows
- 1,563 test rows
- 18 canonical columns
- 10 approved prediction-time predictors

### Reproduced validation results

The tagged code reproduces the declared model-comparison results:

| Model | AP | ROC-AUC | Brier |
|---|---:|---:|---:|
| Dummy prior | 0.124594 | 0.500000 | 0.109118 |
| Logistic unweighted | 0.120959 | 0.476028 | 0.112111 |
| Logistic balanced | 0.120720 | 0.475866 | 0.183289 |
| Random Forest | 0.133099 | 0.509009 | 0.114376 |

The calibration comparison also reproduces:

| Candidate | AP | ROC-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Calibration prior | 0.124594 | 0.500000 | 0.109071 | 0.375982 |
| RF uncalibrated | 0.123084 | 0.507137 | 0.118920 | 0.404801 |
| RF sigmoid | 0.118851 | 0.492863 | 0.109574 | 0.378405 |
| RF isotonic | 0.123552 | 0.495622 | 0.109101 | 0.376116 |

The selected final model is therefore the constant `calibration_prior`.

## Confirmed strengths

1. **Prediction-time reasoning is strong.** The prediction horizon is explicitly set to 24 hours before the appointment.
2. **Target construction is defensible.** Appointments already cancelled or rescheduled at prediction time are excluded; later cancellation/rescheduling remains in the operational cohort as negative outcomes.
3. **Post-event leakage is carefully excluded.**
4. **Chronological train/validation/test partitions are clearly documented.**
5. **Label maturity is handled explicitly with `status_updated_at < model_fit_time`.**
6. **Calibration uses a frozen base estimator and separate chronological calibration population.**
7. **The final test probability vector is created before test labels are accessed.**
8. **Claims are conservative.** The documentation repeatedly states that the model has no appointment-level ranking value and is not clinically validated.
9. **Tests and CI are extensive.** The archive contains 1,389 collected tests, and the supplied GitHub screenshot shows a green CI check.

## Critical gaps

### 1. Scope drift between charter and release

The approved charter lists these final deliverables:

- model interpretation
- Streamlit demonstration application
- Version 1.0 portfolio release

But the released README later declares that serialization and deployment are outside scope. The release changed the finish line instead of completing the original finish line.

### 2. Streamlit application is absent

`app/` contains only `README.md`. That README still says no app is implemented in `v0.1`, even though the repository is tagged `v1.0.0`.

There is no Streamlit dependency, application entry point, user interface, inference workflow, or screenshot.

### 3. No reusable model or inference pipeline

`models/` contains only `README.md`, which also still describes the `v0.1` placeholder state.

The repository intentionally serializes no estimator, preprocessor, probability model, threshold, or policy. There is also no reproducible inference command for a new appointment.

### 4. Final model has no individualized prediction value

The selected model assigns one probability to every appointment:

`0.11985448975684472`

Consequences:

- ROC-AUC = 0.5
- Average Precision = test prevalence
- no appointment can rank above another
- only “intervene all” or “intervene none” are possible
- a patient-level risk calculator would be misleading

This is a valid negative methodological result, but it does not satisfy the original practical portfolio concept of appointment-level no-show risk prediction.

### 5. Historical features were deferred and never implemented

The project correctly identified possible as-of historical features such as:

- prior appointment count
- prior no-show count/rate
- prior cancellation count
- days since previous appointment
- dentist historical no-show rate

These were never implemented. Given the weak signal in the ten static features, this is the most important unfinished modeling avenue.

### 6. Model interpretation and error analysis are incomplete

There is no documented feature-importance table, SHAP/permutation analysis, error analysis, first-observed versus repeat-patient evaluation, or subgroup diagnostic implementation.

The temporal-split document says those diagnostics should later be added, but they are absent from the release.

### 7. No user-facing modeling pipeline command

The repository has CLI entry points for:

- dataset construction
- EDA generation

It does **not** have a modeling/calibration/final-evaluation runner that regenerates the declared model tables and outputs. Those results are evaluated in memory through Python functions and tests.

### 8. Portfolio artifacts are absent from the tag

The archive contains:

- `notebooks/.gitkeep`
- `reports/figures/.gitkeep`
- `reports/screenshots/.gitkeep`
- `data/processed/.gitkeep`

Generated EDA files are ignored. Therefore GitHub does not show notebooks, figures, screenshots, processed manifests, model cards, or application images.

Notebooks are not mandatory if the scripted workflow is strong, but the absence of any committed visual portfolio artifact is a major presentation weakness.

### 9. Validation was reused for multiple sequential decisions

The same temporal validation period supports baseline comparison, tree-model selection, calibration-candidate selection, and later audits. A final test exists, so this is not fatal, but it increases selection dependence on one validation window.

A stronger next version should use rolling-origin validation or a more explicit nested chronological design.

### 10. No software license

The release explicitly declares no license. For a public portfolio repository, this creates ambiguity about reuse and is usually undesirable.

## What is not a problem by itself

- An empty `notebooks/` folder is not automatically a bug; the project uses scripts and documentation instead.
- Empty `data/processed/` and EDA-output folders are expected when generated files are ignored.
- Local `build/` and `*.egg-info/` folders are packaging artifacts and are correctly excluded from Git.
- A negative model result is scientifically acceptable when reported honestly.

## Release classification

The current tag is best described as:

> A well-tested, leakage-aware methodological study with a negative appointment-level prediction result.

It is **not** yet:

> A complete end-to-end no-show prediction portfolio product with a usable Streamlit risk application.

## Recommended recovery strategy

### Recommended path: build a Version 2.0.0

Do not delete the existing tag immediately. Keep it as an auditable methodological checkpoint.

Version 2 should:

1. Open a post-release audit issue documenting the missing charter deliverables.
2. Add leakage-safe, as-of historical patient and provider features.
3. Freeze a new evaluation policy because the existing test outcomes have already been used.
4. Prefer rolling-origin validation and define a new untouched future period or a prospectively frozen synthetic extension.
5. Re-run baseline, tree comparison, calibration, and threshold analysis.
6. Decide the app type based on validated signal:
   - if useful ranking exists: build an appointment-level risk demonstration;
   - if ranking remains absent: build a transparent model-evaluation dashboard rather than a misleading risk calculator.
7. Add a reproducible modeling runner and persisted result artifacts.
8. Add model interpretation, error analysis, and cold-start/repeat-patient diagnostics.
9. Commit key figures and final screenshots.
10. Add a license and update all stale placeholder READMEs.
11. Publish a reviewed `v2.0.0` release only after the original charter checklist is satisfied.

### Minimal alternative: Version 1.1.0

A smaller release could add an educational Streamlit dashboard that displays the population prior and explains why individualized prediction is unavailable. This is honest, but it is weaker as a predictive-ML portfolio project and does not address the missing signal.

## Immediate safety recommendation

Until recovery is planned:

- do not delete the current release or tag;
- do not use the old test period for new model selection;
- do not build a patient-level risk calculator around the constant prior;
- do not call the repository a complete predictive application;
- do not make further commits directly to `main`.
