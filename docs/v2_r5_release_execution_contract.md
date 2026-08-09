# Version 2 Recovery Phase R5 Release-Execution Contract

## Status

**Frozen before R5 release execution.**

This contract governs clean-environment reproduction, release preparation,
merge, tagging, GitHub release publication, and final release evidence for
Version `2.0.0`.

The input checkpoint is the formally closed and CI-sealed R4 commit:

```text
r4_closeout_commit = 5efaa685e5b82f60e1ada165daf9bf7130ee8721
r4_closeout_ci_run = 31219799513
r4_closeout_ci_conclusion = success
```

The protected Version 2 final test is permanently exposed and is not an R5
execution resource.

## Immutable scientific boundary

R5 is release engineering and reproducibility work only.

The following remain frozen:

- selected model and preprocessing;
- selected calibration state;
- protected-test probability vector;
- protected-test evaluation and metrics;
- application decision
  `transparent_model_evaluation_dashboard`;
- policy-sensitivity evidence;
- feature definitions and hyperparameters; and
- the absence of a selected final-test operational threshold.

R5 must not:

- invoke the protected-target accessor or any `allow_test=True` path;
- execute `src.modeling.v2_final_test_evaluation`;
- regenerate `src.modeling.v2_final_test_probabilities`;
- refit or overwrite the persisted frozen Version 2 model;
- recalibrate or tune the model;
- change model features or hyperparameters;
- choose a threshold from protected-test outcomes; or
- reinterpret the protected final test as a development set.

## R5 release state at contract freeze

At this checkpoint:

- `origin/main` remains the immutable Version `1.0.0` release checkpoint;
- tag `v1.0.0` remains unchanged;
- GitHub release `v1.0.0` remains unchanged;
- PR `#14` is open, draft, mergeable, and targets `main`;
- `pyproject.toml` still declares package version `1.0.0`;
- `CITATION.cff` still declares version `1.0.0`;
- no `v2.0.0` tag or GitHub release exists; and
- the recovery branch is clean and exact-head CI-sealed through R4.

The Version 1 metadata remains valid until the later R5 release-preparation
checkpoint changes package and citation metadata together.

## Required R5 execution sequence

The following order is mandatory.

### R5.1 — clean-environment reproduction evidence

Reproduction must occur in a disposable checkout/worktree, never by rewriting
the canonical recovery working tree.

The reproduction runner must:

1. start from an exact committed R5 checkpoint;
2. use Python `3.12`;
3. install `requirements.lock.txt` into a newly created environment;
4. run `python -m pip check`;
5. run the complete repository test suite with pytest cache disabled and an
   isolated temporary base directory;
6. verify target-free Version 2 data-generation and processed-feature
   reproducibility in disposable output locations;
7. verify the frozen reporting/application evidence without protected-target
   re-access or model refit;
8. launch `app/streamlit_app.py` and verify the Streamlit health endpoint;
9. verify committed screenshot identities and documentation links; and
10. leave the canonical repository working tree byte-clean.

Any required reproduction runner must be implemented and tested before its
evidence is accepted. It may use only target-free inputs and already committed
frozen R3 reporting artifacts.

Successful evidence must record the exact source commit, Python version,
dependency-integrity result, test result, safe reproduction commands,
artifact comparisons, Streamlit smoke result, and post-run Git cleanliness.

### R5.2 — release-preparation commit

Only after R5.1 evidence is committed and exact-head CI-sealed may release
metadata change.

The release-preparation batch must:

- change `pyproject.toml` version from `1.0.0` to `2.0.0`;
- change `CITATION.cff` version from `1.0.0` to `2.0.0`;
- preserve the Version 1 release history verbatim;
- move completed Version 2 work out of the changelog `Unreleased` recovery
  section into a Version `2.0.0` release section while retaining a new
  `Unreleased` heading;
- add release notes that explicitly explain the relationship between
  `v1.0.0` and `v2.0.0`;
- synchronize README/docs release language without claiming that GitHub
  publication has already occurred;
- update release-metadata tests to the Version `2.0.0` contract; and
- keep the final publication checklist item open until publication is verified.

The release-preparation commit must pass the complete local test suite and
exact-head PR CI before merge is authorized.

### R5.3 — PR #14 merge and main CI

PR `#14` must remain draft until the R5.2 release-preparation commit is
exact-head CI-sealed.

After that seal:

1. mark PR `#14` ready for review;
2. verify its head is the exact sealed release-preparation commit;
3. verify required checks are green and merge state is clean;
4. merge using a **merge commit**;
5. do not squash or rebase the recovery history;
6. verify the sealed recovery head is an ancestor of the resulting `main`; and
7. wait for the `push` CI workflow on the exact new `main` commit to succeed.

A Version `2.0.0` tag must not be created before the exact merged-main CI
succeeds.

### R5.4 — tag, wheel, checksums, and GitHub release

After exact-main CI succeeds:

1. create annotated tag `v2.0.0` on the exact release source commit chosen by
   the release-preparation/merge protocol;
2. verify the tagged tree contains package version `2.0.0` and citation version
   `2.0.0`;
3. build the release wheel from a clean checkout of the exact tag using Python
   `3.12`;
4. verify the wheel metadata reports Version `2.0.0`;
5. run an installation/import smoke check against the built wheel where
   applicable;
6. compute SHA-256 for the wheel;
7. write `SHA256SUMS.txt`;
8. create GitHub release `v2.0.0` named `Version 2.0.0`;
9. attach the wheel and `SHA256SUMS.txt`; and
10. publish release notes that distinguish the Version 1 methodological
    checkpoint from the completed Version 2 portfolio application.

The Version 1 tag, release, and assets must not be modified.

### R5.5 — post-release evidence and final closure

Publication is not the end of the audit trail.

After the GitHub release exists, a separate post-release evidence update must
record:

- exact `v2.0.0` tag commit;
- GitHub release URL and publication state;
- exact wheel filename, size, and SHA-256;
- checksum-file identity;
- successful merged-main CI identity;
- clean-environment reproduction evidence;
- final documentation/release consistency checks; and
- the completed Version `2.0.0` checklist.

If repository mutation is required for that evidence, it must use a new
post-release branch/PR rather than rewriting the already merged recovery
history. Its exact commit must also pass CI before R5 is declared complete.

## Release artifact shape

Version `v1.0.0` established a release shape consisting of:

- one universal Python wheel; and
- `SHA256SUMS.txt`.

Version `v2.0.0` will preserve that minimal audited release-asset shape unless a
separately frozen amendment is committed and CI-sealed before release.

No PyPI publication is part of the current R5 contract.

## Merge-history rule

The recovery branch contains the auditable R0-R4 chain and exact commit
identities referenced throughout the documentation.

Therefore PR `#14` must be merged with a merge commit. Squash merge and rebase
merge are prohibited for this release because they would rewrite or discard
the recovery commit identities used as audit evidence.

## Pre-finalization documentation and portfolio audit requirements

The pre-finalization audit performed before this contract was committed found
that the core application, screenshots, analytical figures, links, license,
dependency lock, and R4 evidence are intact, but several release-completion
items are intentionally still pending.

Before Version `2.0.0` can be published, R5 must also:

- keep the root `README.md`, `app/README.md`, `models/README.md`,
  `docs/README.md`, release metadata, and recovery-plan status synchronized
  with the actual release state;
- add a top-level `reports/README.md` navigation document covering analytical
  figures, screenshots, and modeling evidence;
- add a top-level `data/README.md` navigation document covering the frozen raw
  Version 2 benchmark and target-free processed artifacts;
- create `docs/v2_portfolio_communication.md` containing final CV/resume
  bullets, a LinkedIn project post draft, 30-second, 90-second, and 5-minute
  project explanations, plus the synthetic-data and external-validation
  limitations;
- re-run a repository-local Markdown relative-link audit and require zero
  broken links;
- verify all four frozen Streamlit screenshots and all three analytical figures
  are present and retain their committed identities;
- verify local-only IDE, build, cache, egg-info, and virtual-environment
  directories are not introduced into the tracked release tree;
- preserve historical Version 1 statements in `CHANGELOG.md` and
  `docs/post_release_audit_v1.0.0.md` when they accurately describe the
  archived Version 1 checkpoint; and
- update current-state pre-release wording only at the appropriate R5
  transition: release-preparation wording before merge/publication, then
  published-release wording only after the GitHub release exists.

The current pre-release `1.0.0` values in `pyproject.toml` and `CITATION.cff`
are therefore not defects at contract-freeze time. They must change together
during R5.2, after clean-environment evidence is committed and CI-sealed.

## R5 acceptance boundary

R5 can close only when:

- clean-environment reproduction evidence is committed and CI-sealed;
- package and citation metadata consistently declare Version `2.0.0`;
- release notes accurately explain `v1.0.0` versus `v2.0.0`;
- the release-preparation commit is exact-head CI-sealed;
- PR `#14` is merged without rewriting recovery history;
- exact merged-main CI is successful;
- tag and GitHub release `v2.0.0` are verified;
- wheel and checksum identities are recorded;
- the final repository documentation reflects the published release;
- the Version `2.0.0` completion checklist is complete; and
- the scientific/post-test immutability boundary remains unchanged.

All performance claims remain scoped to the synthetic longitudinal benchmark.
