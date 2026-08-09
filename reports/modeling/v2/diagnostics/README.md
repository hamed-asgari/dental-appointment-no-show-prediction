# Version 2 R3 pre-test diagnostics

These artifacts are the frozen Phase R3 interpretation, error-analysis, and
subgroup diagnostics computed only on the already permitted `policy_selection`
population.

Inputs are the committed frozen Logistic Regression pipeline and the committed
R2 policy-selection predictions/targets. The protected 2027 `final_test`
partition is not scored here and its targets are not accessed.

Outputs:

- `permutation_importance.csv` — raw-feature permutation importance using
  Average Precision, 20 repeats, and random state `20260807`;
- `subgroup_metrics.csv` — predeclared subgroup metrics with the frozen minimum
  support rules;
- `first_time_vs_repeat.csv` — explicit cold-start/history cohort diagnostic;
- `row_error_analysis.csv` — per-row absolute error, Brier contribution, and
  log-loss contribution;
- `capacity_error_summary.csv` — error summaries at the already registered
  5%, 10%, and 20% capacity fractions;
- `diagnostics_summary.json` — machine-readable overall summary;
- `pretest_diagnostics_manifest.json` — input identities plus SHA-256/byte-size
  identities for every generated diagnostic artifact.

Permutation importance is descriptive only and cannot trigger feature
selection or model changes. No single operational threshold is selected.
