# Version 2 Streamlit portfolio screenshots

These PNG files are screenshots of the actual Version 2 Streamlit application
running from the CI-sealed R4 application code. They are presentation artifacts,
not analytical figures and not mockups.

The application type is:

```text
transparent_model_evaluation_dashboard
```

The screenshots do not depict an appointment-level risk calculator. The app
does not score patients, select an operating threshold, or access protected
final-test targets.

## Frozen screenshot set

| File | Dimensions | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| `v2_streamlit_overview.png` | 1837x926 | 150978 | `809f42bcfacd919248663f515595bbb1810dcdc35a92e0f186e74446320d8e79` |
| `v2_streamlit_performance.png` | 1861x519 | 69232 | `0e7f45905b1cf2940215ebf56b53351507f8ea84722ab5df13323d9acbc48c64` |
| `v2_streamlit_calibration_capacity.png` | 1222x878 | 124022 | `e6533a9432744a458327e1bfc7ee4a02b425040501bd42e2c6488a98b85a588a` |
| `v2_streamlit_interpretation_limitations.png` | 1920x1080 | 206902 | `572772460187355a2cd0508284c2b58c87494bf069b52fb5256dfb63fd3913be` |

## What each screenshot shows

- **Overview** — synthetic/non-clinical disclaimer, the pre-frozen application
  decision gate, and the explicit Brier-gate failure.
- **Performance** — protected-test population and frozen discrimination,
  probability-quality, and calibration metrics.
- **Capacity sensitivity** — the pre-registered 5%, 10%, and 20% descriptive
  capacity scenarios plus the no-operational-threshold boundary.
- **Interpretation and limitations** — pre-test permutation importance,
  synthetic-data limitations, and the external-validation plan.

The underlying analytical figures are stored separately under
`reports/figures/`.
