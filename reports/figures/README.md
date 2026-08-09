# Version 2 analytical figures

These figures are deterministic Phase R3 reporting outputs generated from the
committed one-time protected final-test evaluation artifacts.

They are analytical evidence for the transparent model-evaluation dashboard;
they are not screenshots and do not represent a deployed application.

Reproduce them from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m src.modeling.v2_final_reporting --overwrite
```

Outputs:

- `v2_final_precision_recall_curve.png`
- `v2_final_calibration_curve.png`
- `v2_final_capacity_tradeoff.png`

The reporting runner does not invoke the protected-target accessor, refit the
model, change calibration, or select an operational threshold.
