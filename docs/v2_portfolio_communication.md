# Version 2 Portfolio Communication

## Positioning

Dental Appointment No-show Prediction is a reproducible Clinical AI portfolio
study built entirely on synthetic dental appointment data. The strongest
portfolio claim is not that it is a deployable patient-risk product; it is that
the project demonstrates leakage-controlled longitudinal feature engineering,
chronological model selection, disciplined one-time protected evaluation,
calibration-aware interpretation, reproducible artifacts, and an evidence-based
Streamlit evaluation dashboard.

## CV / resume bullets

- Built a deterministic longitudinal synthetic dental-appointment benchmark and
  32-feature prediction-time-safe dataset using strict historical as-of rules.
- Evaluated Logistic Regression and Random Forest with rolling-origin validation;
  selected Logistic Regression before protected-test access using pre-frozen
  ranking guardrails.
- Executed a one-time protected 2027 evaluation on 4,343 appointments, achieving
  average precision `0.1472` versus prevalence/prior `0.0824` and ROC AUC
  `0.6300`.
- Interpreted mixed calibration evidence rather than hiding it: Brier score was
  `0.07621` versus prior `0.07569`, while log loss improved to `0.28262` versus
  `0.28499`.
- Shipped a read-only Streamlit model-evaluation dashboard backed only by frozen,
  hash-verified reporting artifacts; no runtime patient scoring or post-test
  tuning is performed.
- Reproduced raw, target-free processed, and final-reporting artifacts
  byte-identically in a disposable Python 3.12 environment with full CI.

## LinkedIn project post draft

I completed Version 2 of my Dental Appointment No-show Prediction portfolio
project.

The project uses fully synthetic longitudinal dental appointment data and was
designed around a strict question: can I build and evaluate a ranking model
without leaking future attendance history into current predictions?

I implemented prediction-time as-of features, rolling-origin validation,
pre-frozen model-selection rules, and a one-time protected final test. Logistic
Regression improved average precision from a prior/prevalence baseline of
0.0824 to 0.1472 and reached ROC AUC 0.6300. Calibration evidence was mixed:
log loss improved, but Brier score was slightly worse than the prior baseline.

That result changed the product decision. Instead of presenting an
individualized patient-risk calculator, I built a read-only Streamlit
model-evaluation dashboard that surfaces ranking performance, calibration,
capacity sensitivity, interpretation, and limitations.

I also added clean-environment reproduction that regenerated the synthetic raw
benchmark, target-free processed features, and final reporting package
byte-identically.

Important limitation: this is a synthetic-data portfolio study with no external
clinical validation, so it is not a deployable clinical decision-support tool.

## 30-second explanation

I built an end-to-end no-show prediction study on fully synthetic longitudinal
dental data. The main technical focus was preventing temporal leakage with
strict as-of history features and rolling-origin validation. The selected
Logistic Regression improved ranking on a one-time protected test, but
calibration was mixed, so the final Streamlit product is a transparent
evaluation dashboard rather than a patient-risk calculator.

## 90-second explanation

The project starts with a deterministic synthetic longitudinal benchmark rather
than real patient data. For every appointment, historical attendance features
only use status information that was available before that appointment's
prediction time. I then used rolling-origin validation to compare a constant
prior, Logistic Regression, and Random Forest under pre-frozen selection rules.

Logistic Regression was selected before final-test access. On the protected
2027 test it reached average precision `0.1472` versus a prior/prevalence
baseline of `0.0824`, with ROC AUC `0.6300`. Log loss improved slightly, but
Brier score was slightly worse than the prior baseline.

Because the project had a pre-frozen application gate, I did not reinterpret
that mixed result as a deployable patient-level risk product. The final
Streamlit app is a read-only evaluation dashboard backed by frozen artifacts.
R5 clean reproduction also regenerated the raw benchmark, processed features,
and final reporting package byte-for-byte in a fresh Python 3.12 environment.

## 5-minute explanation

The project is organized around temporal integrity, model-selection discipline,
and reproducibility.

First, I generated a deterministic synthetic longitudinal dental benchmark.
The Version 2 feature contract uses strict as-of logic: a historical appointment
can contribute to the current appointment only when its status-update timestamp
is earlier than the current prediction time. That boundary is applied to
patient, dentist, visit-type, and weekday-hour attendance histories, with frozen
smoothing and minimum-support behavior. The resulting target-free modeling
dataset has 32 predictors.

Second, I treated model development as a chronological ranking problem rather
than random cross-validation. A constant prior, Logistic Regression, and Random
Forest were evaluated with rolling-origin folds. The pre-frozen gate required
meaningful average-precision uplift, ROC AUC above chance, and positive uplift
across folds. Logistic Regression passed most clearly and was selected without
using the protected 2027 outcomes.

Third, the final-test protocol was intentionally one-time. The selected pipeline
produced its frozen probability vector before protected labels were exposed. On
4,343 final-test appointments with 358 no-shows, average precision was
`0.1471577`, ROC AUC `0.6300295`, Brier `0.0762052`, and log loss `0.2826231`.
The prior comparator had average precision `0.0824315`, Brier `0.0756873`, and
log loss `0.2849868`. So ranking improved materially and log loss improved
slightly, but Brier score worsened slightly.

That calibration trade-off matters. The application decision had been frozen
before test exposure, so the project did not tune after seeing the result.
Instead, it selected a `transparent_model_evaluation_dashboard`. The Streamlit
app reads only frozen reporting artifacts and shows performance, calibration,
capacity sensitivity, interpretation, and limitations. It does not score a
new patient.

Finally, R5 clean-environment reproduction created a detached disposable
worktree and fresh Python 3.12 environment, installed locked dependencies,
passed the full test suite, regenerated the synthetic benchmark and target-free
processed features byte-identically, reproduced the reporting figures and
summary byte-identically, checked documentation links and tracked-tree hygiene,
and launched Streamlit successfully.

The core limitation is external validity. The data are synthetic and there is
no external clinical validation, prospective study, intervention-effect study,
or real-world fairness/safety assessment. The correct portfolio framing is a
reproducible modeling and evaluation system, not a clinically deployable
patient-risk tool.

## Synthetic-data and external-validation limitations

- All patient, dentist, and appointment records are synthetic.
- No real clinical population was used for training, validation, or testing.
- No external dataset has been used to test transportability.
- No prospective workflow study establishes clinical utility or intervention
  effectiveness.
- The protected final test is an internal synthetic benchmark, not independent
  external validation.
- The dashboard must not be presented as individualized medical advice or a
  production clinical decision-support system.
