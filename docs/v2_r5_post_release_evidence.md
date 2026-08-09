# R5.5 Post-release Evidence — Version 2.0.0

## Purpose

This document records post-release provenance for published Version `2.0.0`
without modifying the immutable `v2.0.0` tag, published release assets, or the
frozen post-test scientific result. It is added on `post-release/v2.0.0-evidence` from exact
released `main`.

## Published release identity

- GitHub Release: `Version 2.0.0`
- Release URL: https://github.com/hamed-asgari/dental-appointment-no-show-prediction/releases/tag/v2.0.0
- Published at: `2026-08-09T19:30:29Z`
- Annotated tag: `v2.0.0`
- Tag-object SHA: `be6dedbb2cc80ef855c0d90c0de20d8993d46a2b`
- Tag peeled commit: `cf7ecca52bc54aeeccdad032b80be83f9d172fc9`
- Exact release source commit: `cf7ecca52bc54aeeccdad032b80be83f9d172fc9`
- Sealed recovery head: `1ecb5c184b21b87b9ba89f9041434af5c38a4db7`
- Exact-main CI run: `31328381538` (`success`)
- Exact-main CI URL: https://github.com/hamed-asgari/dental-appointment-no-show-prediction/actions/runs/31328381538

The release is neither draft nor prerelease.

## Published assets

| Asset | Size | SHA-256 |
| --- | ---: | --- |
| `dental_appointment_no_show_prediction-2.0.0-py3-none-any.whl` | 156042 bytes | `0c3ac6fe0e4fa0a911ab6027694cff9729c4bb95800b560c223faeddc41ec230` |
| `SHA256SUMS.txt` | 127 bytes | `f1b4b2b0fc5ec4900919a6b96f0e10b30122ae16b04485c324957a693bc14da3` |

R5.4 downloaded both published assets after release creation and verified
byte identity against the final locally verified release artifacts.

## R5.1 clean-environment provenance

- evidence commit: `fa6f8477d0bebe13ee6f6828713b9caefea41ba8`
- exact-head evidence CI: `31324234415` (`success`)
- evidence path: `reports/release/v2/r5_clean_reproduction_evidence.json`
- evidence size: `3597` bytes
- evidence SHA-256: `a28fc45030b9d83dc9f2f36debc2d99ed2f34d5323650c1381a88613cec6ec95`

The protected final target was not reopened during clean reproduction.

## Version 1.0.0 immutability

Version `1.0.0` remains the historical methodological/audit checkpoint. Its tag
still peels to `ec5aa5d84c8de6aaf7daca4c299073ba12d4e700`, its GitHub Release remains `Version 1.0.0`, and
its historical release assets retain their original names and sizes.

## Scientific and product boundary

The frozen application decision remains
`transparent_model_evaluation_dashboard`. The published application is a
read-only evidence dashboard, not an individualized patient risk calculator.

R5.5 performs no protected-target re-access, model refit, recalibration, feature
change, final-test threshold selection, or post-test tuning. The protected final
test remains a one-time evaluation and must not be reused for development.

## Integration boundary

This evidence is intentionally outside the `v2.0.0` tagged tree. The
`post-release/v2.0.0-evidence` branch passed exact-head pull-request CI run
`31335107018` on commit `4277cefcf8e3ac0b77cfe1afeec9be0b7d5b142a`. PR #15 merged to `main` at
`2026-08-09T20:57:30Z` via merge commit `33a46e813b9ca3150af41c83e7e5fd292734a496` with the released-main commit as
first parent and the R5.5 evidence commit as second parent. Exact-main push CI run
`31335579708` then completed successfully on that merge commit. Recovery Phase R5 is formally closed. This integration did not modify the published `v2.0.0`
tag or release assets and did not reopen the protected final target.

Machine-readable evidence:
`reports/release/v2/r5_post_release_evidence.json`.
