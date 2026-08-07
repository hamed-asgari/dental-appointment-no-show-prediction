# Phase R1 Completion Evidence

## Status

**Phase R1 complete on `recovery/v2.0.0-completion`**

Closeout checkpoint:

```text
implementation HEAD: 44765dd
closeout date: 2026-08-07
full local test suite: 1534 passed
GitHub CI run: 31153155060
GitHub CI job: 92786874768
review snapshot SHA-256:
a4184151695d6e95a6ffb2a971d3b363099d6654193898c69607d76bfc84244b
```

No recovered model selection has started at this checkpoint.

## Frozen benchmark and evaluation policy

The Version 2 longitudinal benchmark and evaluation windows were frozen before
recovered modeling.

```text
configuration SHA-256:
aafb631abc615f61f5f4efda9650ab1efac5664c50d62ea7e399a69c97fbaf50

raw manifest SHA-256:
7702fa5fa0638c52dd0598e28f35f678fb5d61a886faadf9b38a6e292fdcd561

raw dataset fingerprint:
d9fdfa1a93091fd15bc34a62d655aef313966e2603d901350a6bd969b4e3c1bf
```

The protected final test remains:

```text
2027-01-01 <= prediction_time < 2028-01-01
```

and contains 4,343 eligible feature rows.

## Processed feature artifact

The canonical target-free Version 2 feature artifact is:

```text
data/processed/v2/v2_feature_dataset.csv
```

Frozen identity:

```text
rows: 21,755
columns: 38
approved model features: 32

dataset SHA-256:
08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53

manifest SHA-256:
2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073

processed dataset fingerprint:
0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787
```

The manifest records:

```text
target_included = false
final_test_target_accessed = false
```

## Implemented Phase R1 controls

The recovery branch now contains:

- a Project 2-owned longitudinal synthetic benchmark with reminder timing and
  hidden but unexported longitudinal risk effects;
- deterministic raw export with frozen configuration, file hashes, manifest,
  and dataset fingerprint;
- a historical-feature contract frozen before implementation;
- strict-as-of patient history;
- strict-as-of dentist, visit-type, and weekday-hour history;
- fixed Beta `(1, 9)` no-show smoothing and aggregate minimum attendance
  support of 10;
- prediction-time-safe current appointment features;
- a defensive 32-feature model allowlist;
- frozen chronological evaluation partitions;
- label-maturity metadata separated from feature construction;
- a target-free committed processed feature artifact;
- strict non-test target maturity access;
- a gated protected final-test accessor requiring a prewritten and sealed
  probability vector; and
- cross-platform artifact and protected-tree regression checks.

## Phase R1 implementation history

| Checkpoint | Commit |
|---|---|
| Freeze Version 2 data-generation and evaluation policy | `d462439` |
| Add frozen benchmark configuration | `9918650` |
| Add deterministic longitudinal generator core | `37eec55` |
| Freeze Version 2 raw benchmark | `0e21b6d` |
| Canonicalize Markdown snapshot hashing across EOL | `4dfed04` |
| Freeze historical feature contract | `70db535` |
| Add strict-as-of patient-history features | `b55c9c7` |
| Add strict-as-of aggregate-history features | `48c6ceb` |
| Add Version 2 feature-only dataset builder | `40f4247` |
| Add protected Version 2 targets and processed export | `44765dd` |

## Acceptance-criteria evidence

| Phase R1 criterion | Evidence |
|---|---|
| Historical specification exists before implementation | `docs/v2_historical_feature_contract.md` at `70db535` |
| Renewed temporal policy frozen before modeling | `docs/v2_data_generation_and_evaluation_policy.md` at `d462439` |
| Raw benchmark identity frozen | Raw manifest and three exact raw-file SHA-256 hashes |
| No row may use future outcomes | Strict `status_updated_at < prediction_time` sweep plus boundary/mutation tests |
| Deterministic feature calculations | Input-shuffle, same-time-batch, and frozen-benchmark tests |
| Cold-start behavior explicit | Frozen defaults and availability flags with tests |
| Reminder timing enforced | Derived feature requires `reminder_sent_at <= prediction_time` |
| Leakage-safe patient and aggregate histories | Patient, dentist, visit-type, and weekday-hour engines with strict as-of tests |
| Updated data dictionary | `docs/v2_data_dictionary.md` |
| Processed-data manifest | `data/processed/v2/v2_feature_dataset.manifest.json` |
| Label maturity separated from features | `label_available_at` plus strict development target accessor |
| Protected final-test access gated | Explicit opt-in plus probability-vector validation and sealing |
| Protected test remains uninspected | Manifest records `final_test_target_accessed = false`; no final-test metric is declared |
| Complete local suite passes | `1534 passed` at implementation checkpoint |
| GitHub CI passes | Run `31153155060`, job `92786874768` |
| Version 1 remains an audit checkpoint | Version 1 modules/results remain separate from Version 2 paths |

## Protected-test state

Phase R1 closes **before** any successful access to the real 2027 final-test
targets. The successful accessor path is tested only on synthetic fixtures.

The final-test feature rows may be generated because their construction is
target-free and operationally sequential. Their labels remain protected. No
protected 2027 test metric has been inspected.

## R2 entry gate

Phase R2 may begin only after this closeout documentation is committed and its
CI run is green.

R2 must not:

- revise the frozen generator in response to model performance;
- add or choose features after inspecting protected final-test outcomes;
- use `final_test` for preprocessing, estimator, hyperparameter, calibration,
  threshold, or application-behavior selection; or
- access protected final-test targets before the final probability vector is
  written and sealed.

The next phase is recovered model comparison, calibration, and threshold
analysis using only the predeclared chronological development, calibration,
and policy-selection windows.
