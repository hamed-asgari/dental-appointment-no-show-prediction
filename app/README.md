# Application

## Version 2 application

Version 2 implements a **transparent model-evaluation dashboard** in Streamlit.
The application type was frozen before R4 implementation after the protected
final-test gate rejected an appointment-level risk demonstration because the
model Brier score was slightly worse than the frozen population-prior baseline.

The dashboard is portfolio/research presentation only. All records are fully synthetic,
and the model is **not validated for clinical or operational use**.

The app does not:

- accept patient or appointment inputs for individualized prediction;
- score new appointments;
- display patient-level "high risk" labels;
- expose an operating-threshold control;
- access protected final-test targets;
- load the persisted model for runtime inference;
- refit or recalibrate the model.

## Launch

From the repository root, using the locked Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The application validates the SHA-256 identities and state flags of its frozen
R3 reporting inputs before rendering evidence. An integrity failure stops
evidence rendering rather than recomputing or silently replacing results.

## Evidence displayed

The dashboard displays:

- the four pre-frozen application-decision gates;
- frozen protected-test discrimination and calibration metrics;
- the committed precision-recall and calibration figures;
- the pre-registered capacity-sensitivity figure;
- pre-test permutation importance;
- limitations and an external-validation plan.

Its primary machine-readable inputs are the committed R3 final-reporting
summary and manifest under `reports/modeling/v2/final_reporting/`.

## Portfolio screenshots

Actual screenshots from the committed application state are stored in
`reports/screenshots/`:

- `v2_streamlit_overview.png`
- `v2_streamlit_performance.png`
- `v2_streamlit_calibration_capacity.png`
- `v2_streamlit_interpretation_limitations.png`

Their dimensions and SHA-256 identities are documented in
[`../reports/screenshots/README.md`](../reports/screenshots/README.md).

## Architecture

The presentation/data-flow boundary is documented in
[`../docs/v2_r4_portfolio_architecture.md`](../docs/v2_r4_portfolio_architecture.md).

The application consumes frozen reporting artifacts only. There is no runtime
path from user input to model inference, protected-target access, recalibration,
or threshold selection.
