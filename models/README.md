# Model artifacts

## Version 1 historical checkpoint

Version `v1.0.0` did not persist a reusable preprocessing pipeline, estimator,
calibrator, probability policy, or inference artifact. Its selected final
candidate was a constant population prior and provided no appointment-level
ranking.

Those statements describe the archived Version 1 checkpoint only.

## Version 2 frozen model artifacts

Version 2 persists the selected Logistic Regression preprocessing/model
pipeline and its audit metadata under `models/v2/`:

```text
models/v2/frozen_logistic_pipeline.joblib
models/v2/frozen_logistic_pipeline.metadata.json
models/v2/frozen_logistic_pipeline.manifest.json
```

The model artifact is tied to the frozen Version 2 feature and execution
contracts and was persisted before protected final-test target access.

The R4 Streamlit application deliberately does not load this model for appointment scoring.
The evidence-based application gate selected a
`transparent_model_evaluation_dashboard`, so the app reads committed final
reporting artifacts instead.

The persisted model remains useful for reproducibility and audit, but it is:

- trained only on fully synthetic data;
- not clinically or operationally validated;
- not authorized for individualized risk display;
- not associated with a selected operational threshold; and
- frozen against post-test refitting, recalibration, feature change, or
  hyperparameter tuning.

See `models/v2/` for the machine-readable artifact manifest and
`docs/v2_model_card.md` for the evaluated model boundary.
