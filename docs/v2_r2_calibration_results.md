# Version 2 Phase R2 Chronological Calibration Results

## Status

Calibration was evaluated only after the rolling-origin ranking choice was
frozen. The protected 2027 final-test target remained closed and no
final-test probabilities were generated.

## Chronology

- selected ranking model: `logistic_regression`
- base refit time: `2026-07-01T00:00:00`
- base fit rows / positives: `10,921` / `978`
- calibrator fit time: `2026-09-01T00:00:00`
- calibration fit rows / positives: `738` / `50`
- calibration evaluation label cutoff: `2026-10-01T00:00:00`
- calibration evaluation rows / positives: `328` / `25`

## Calibration evaluation metrics

| Method | AP | ROC-AUC | Brier | Log loss | Intercept | Slope | Mean p | Guardrail | Within Brier margin | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `uncalibrated` | 0.1394 | 0.6228 | 0.0697 | 0.2640 | -0.6877 | 0.7498 | 0.0847 | pass | yes | yes |
| `sigmoid` | 0.1394 | 0.6228 | 0.0698 | 0.2654 | 1.2183 | 1.3560 | 0.0600 | pass | yes | no |
| `isotonic` | 0.1094 | 0.6034 | 0.0708 | 0.5657 | -2.5359 | -0.0076 | 0.0567 | fail | no | no |

## Frozen calibration decision

The frozen rule selected **`uncalibrated`**.

Calibration selection is separate from ranking selection. The base
ranking model remains **`logistic_regression`**. The pre-registered
log-loss guardrail, Brier indifference margin, and simplicity order are
applied exactly as frozen.

## Reliability curve

`reports/modeling/v2/calibration/calibration_reliability_curve.csv`
contains 10 deterministic equal-frequency quantile-rank bins for each
method.

## Protected-test state

- `final_test_target_accessed = false`
- `final_test_probabilities_generated = false`

## Next R2 step

The next R2 batch applies the frozen policy-sensitivity scenarios to
the `policy_selection` window using this frozen ranking/calibration
choice. It does not open or score the protected final test.
