# Dental Appointment No-show Prediction

## Project Charter

**Repository:** `dental-appointment-no-show-prediction`

**Project type:** Public portfolio and learning project in Clinical Data Science, Dental AI, and Digital Health.

### Purpose

Develop an end-to-end, reproducible machine-learning workflow for estimating the probability that a scheduled dental appointment will become a no-show, using fully synthetic data and only information that would realistically be available before the appointment.

### Primary objective

Demonstrate rigorous prediction-time reasoning, leakage prevention, probability-based model evaluation, calibration, and operationally meaningful decision-making rather than simply maximizing classification accuracy.

### Core principles

- All data used in the project will be explicitly disclosed as synthetic.
- Project 1 remains frozen at Version 1.0.0.
- Reused Project 1 data will be copied or reconstructed for this repository rather than modifying the original repository.
- The prediction target will not be finalized until the appointment data and relevant status definitions have been inspected.
- Every candidate feature must be evaluated according to whether it genuinely exists at the selected prediction time.
- Model performance will be reported conservatively and without implying real-world clinical validity.
- Real-world use would require external validation on representative operational data.

### Planned final deliverables

- reproducible data-preparation workflow
- documented target and prediction-time definition
- leakage assessment
- exploratory analysis
- interpretable baseline model
- tree-based comparison model
- discrimination and calibration evaluation
- operational threshold/cost analysis
- model interpretation
- Streamlit demonstration application
- technical documentation and README
- limitations and external-validation plan
- Version 1.0 portfolio release

## v0.1 Scope

Version v0.1 is the project-foundation phase.

Its purpose is to establish a clean, reproducible workspace before any modeling begins.

### Included in v0.1

#### Project setup

- create the separate repository structure
- establish a stable-main / feature-branch workflow concept
- add basic repository metadata
- establish dependency-management foundations

#### Data intake

- identify which synthetic Project 1 source files may later be relevant
- copy appropriate source data into Project 2 without changing Project 1
- preserve original synthetic source files as immutable raw inputs
- document data provenance and the fact that all observations are synthetic

#### Problem-definition preparation

- inspect the appointment-related source structure in a later step
- document candidate appointment-status values in a later step
- identify fields requiring prediction-time/leakage review in a later step
- prepare for a later explicit decision about:
  - the positive class
  - the negative class
  - cancellation handling
  - prediction timing

#### Reproducibility foundation

- define the Python environment
- establish configuration/path conventions later as needed
- create locations for data processing, notebooks, source code, models, reports, and application files
- establish a structure suitable for later testing and reproducible execution

### v0.1 completion criterion

v0.1 is complete when the repository can be cloned cleanly, its structure and data provenance are understandable, the relevant synthetic source data are available or reproducibly obtainable, and the project is ready to begin formal target-definition analysis.

No predictive model is required for v0.1.

## Explicitly Out of Scope for v0.1

The following are intentionally deferred:

- final binary target definition
- final treatment of cancellations
- final prediction-time decision
- feature engineering
- feature selection
- model training
- Logistic Regression
- Random Forest, Gradient Boosting, or other comparison models
- hyperparameter tuning
- train/validation/test splitting
- cross-validation
- ROC-AUC or PR-AUC analysis
- precision, recall, specificity, or confusion-matrix analysis
- calibration curves
- Brier score analysis
- Platt/sigmoid calibration
- isotonic calibration
- threshold optimization
- false-positive/false-negative cost assumptions
- operational sensitivity analysis
- SHAP or other model-interpretation methods
- Streamlit implementation
- deployment
- real-patient or real-clinic data
- claims of clinical effectiveness
- production integration with scheduling or EHR/practice-management systems
- automated patient interventions
- causal claims about why patients miss appointments
- modification or extension of the frozen Project 1 repository unless a genuine defect is discovered
