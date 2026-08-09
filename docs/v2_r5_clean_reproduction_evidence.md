# R5.1 Clean-environment Reproduction Evidence

## Status

**R5.1 clean-environment reproduction passed.**

The reproduction was executed from the exact CI-sealed runner/source commit:

```text
source_commit = 5e527e6a45f0b65e9889268d7b93585f931f3fdb
runner_ci_run = 31321490041
python_base = 3.12.13
python_clean_environment = 3.12.13
```

The exact machine-readable execution evidence is committed at:

```text
reports/release/v2/r5_clean_reproduction_evidence.json
```

Evidence identity:

```text
size_bytes = 3597
sha256 = a28fc45030b9d83dc9f2f36debc2d99ed2f34d5323650c1381a88613cec6ec95
```

## Validation results

The disposable clean environment successfully:

- installed the locked dependency set and passed `python -m pip check`;
- passed the complete repository test suite;
- reproduced the frozen Version 2 synthetic raw benchmark byte-identically;
- reproduced the target-free Version 2 processed feature artifacts
  byte-identically;
- reproduced the frozen R3 final-reporting summary, manifest, and analytical
  figures byte-identically;
- verified all four frozen Streamlit screenshot identities;
- checked 64 repository-relative Markdown links with zero broken links;
- passed tracked-tree hygiene with zero forbidden tracked local-only paths; and
- launched the Streamlit dashboard successfully with health result `ok`.

The frozen artifact identities include:

```text
raw_manifest = 7702fa5fa0638c52dd0598e28f35f678fb5d61a886faadf9b38a6e292fdcd561
processed_dataset = 08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53
processed_manifest = 2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073
reporting_summary = 76ed771871b696b4e0cd0c262b7d13f3bb03d03a187edd84c40934f1d2bfbbf7
reporting_manifest = 15f47f11e0378376baf7a2f5c520beb389ed8952f3ac161321548aeb35ca64b3
```

## Scientific immutability

The execution evidence records all of the following as false:

```text
forbidden_project_modules_invoked = false
protected_target_reaccess_performed = false
model_refit_performed = false
calibration_change_performed = false
threshold_selection_performed = false
post_test_tuning_performed = false
```

Only these project modules were used for reproduction:

```text
src.synthetic.export
src.data.export_v2_processed
src.modeling.v2_final_reporting
```

The protected final-test accessor and final-test evaluation/probability
generation modules were not invoked.

## R5 gate

R5.1 evidence is now recorded on the recovery branch. The next gate is
exact-head CI sealing of this evidence commit. Version `2.0.0` release metadata
must remain unchanged until that CI seal succeeds.
