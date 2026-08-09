# Reports

`reports/` contains committed modeling, evaluation, reporting, application, and
release-evidence artifacts for the portfolio study.

## Modeling and evaluation

`reports/modeling/v2/` contains Version 2 development, calibration, policy, and
frozen final-test artifacts. The protected final test was accessed exactly once
under the R3 contract. No post-test model, feature, calibration, or threshold
tuning is permitted.

## Final reporting

The frozen R3 final-reporting summary SHA-256 is
`76ed771871b696b4e0cd0c262b7d13f3bb03d03a187edd84c40934f1d2bfbbf7`
and the reporting manifest SHA-256 is
`15f47f11e0378376baf7a2f5c520beb389ed8952f3ac161321548aeb35ca64b3`.

## Figures and screenshots

`reports/figures/` contains frozen analytical figures.
`reports/screenshots/` contains frozen Streamlit portfolio screenshots; see
[`screenshots/README.md`](screenshots/README.md) for their identities.

## Release evidence

`reports/release/v2/r5_clean_reproduction_evidence.json` is the machine-readable
R5.1 clean-environment evidence. It records dependency integrity, full-suite
success, byte-identical reproduction, link/hygiene checks, Streamlit health,
and false flags for protected target re-access and model refit.

See [`../docs/v2.0.0_release_notes.md`](../docs/v2.0.0_release_notes.md).
GitHub Release `Version 2.0.0` is published from annotated tag `v2.0.0` at https://github.com/hamed-asgari/dental-appointment-no-show-prediction/releases/tag/v2.0.0. Post-release provenance is recorded in [`../docs/v2_r5_post_release_evidence.md`](../docs/v2_r5_post_release_evidence.md).
