# Dental Appointment No-show Prediction

> **Synthetic-data and use disclaimer**
>
> All data used in this project are fully synthetic and do not represent real patients, real clinical records, or real healthcare operations.
>
> This project is a portfolio and learning exercise and is not validated for clinical or operational use.

## Overview

This repository is the foundation for an end-to-end Clinical Data Science and Dental AI portfolio project. The eventual project will examine how to estimate dental appointment no-show risk before an appointment using only information that would realistically be available at the selected prediction time.

The emphasis will be on reproducible work, careful leakage prevention, probability-focused evaluation, calibration, and transparent operational reasoning. Any future results will be presented as evidence from synthetic data only, not as real-world clinical validation.

## Current status

**Version v0.1: repository foundation.**

This version establishes the repository structure, documentation, data-provenance conventions, and a minimal Python environment. Modeling and analytical work have **not** started. In particular, no prediction target, cancellation policy, prediction time, features, data split, model, performance result, threshold, or application has been defined or implemented.

## Future objectives

Later project phases are intended to:

- establish a defensible target and prediction-time definition after inspecting the synthetic appointment data;
- build a reproducible data-preparation workflow with explicit leakage review;
- compare interpretable and tree-based predictive approaches;
- evaluate both discrimination and probability calibration conservatively;
- explore operationally meaningful decision thresholds and model interpretation; and
- present the completed workflow in a portfolio demonstration application.

These are planned objectives, not current capabilities.

## Repository structure

```text
.
├── app/                   # Reserved for a future portfolio demonstration
├── data/
│   ├── raw/               # Immutable synthetic source copies and provenance
│   ├── interim/           # Future intermediate data products
│   └── processed/         # Future analysis-ready data products
├── docs/                  # Project charter and supporting documentation
├── models/                # Future local serialized model artifacts
├── notebooks/             # Future analytical notebooks
├── reports/
│   ├── figures/           # Future generated figures
│   └── screenshots/       # Future portfolio screenshots
├── src/
│   ├── data/              # Future data-processing code
│   ├── evaluation/        # Future evaluation code
│   ├── features/          # Future feature code
│   └── models/            # Future modeling code
├── tests/                 # Future automated tests
├── .gitignore
├── README.md
└── requirements.txt
```

Directories reserved for later phases are intentionally empty in v0.1; they do not imply completed implementation.

## Environment foundation

Create an isolated Python environment and install the small v0.1 dependency set:

```bash
python -m venv .venv
```

Activate the environment using the command appropriate for your shell, then run:

```bash
python -m pip install -r requirements.txt
```

The initial requirements are intentionally unpinned and limited to general data and notebook tooling. Exact environment locking and additional dependencies should be introduced only when later analytical phases require them.

## Reproducibility and data provenance

Future work should treat files in `data/raw/` as immutable inputs, record provenance when a dataset is added, and write transformations only to `data/interim/` or `data/processed/`. Analytical decisions, configuration, dependencies, and execution instructions will be documented as the workflow develops.

No source dataset is included in v0.1. All datasets later added to this repository must be fully synthetic; real patient or clinical records do not belong here.

## Relationship to Project 1

Project 1, **Dental Clinic Operations Analytics**, is a separate repository and remains frozen at Version 1.0.0. This project will not modify Project 1. Appropriate fully synthetic source data may later be copied from or reconstructed from Project 1, but any copy used here must be stored and documented independently in this repository.

See the [project charter](docs/project_charter.md) for the approved scope and planned deliverables.
