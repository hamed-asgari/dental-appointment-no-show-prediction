# R5.1 Clean-environment Reproduction Runner

## Purpose

Phase R5.1 verifies that the committed Version 2 repository can be reproduced
from an exact commit in a disposable Windows/Python 3.12 environment without
reopening the protected final-test target or refitting the selected model.

Repository-local runner:

```text
scripts/r5_clean_reproduction.py
```

The runner is implemented and CI-sealed before its reproduction evidence is
accepted.

## Scientific safety boundary

The runner invokes only these project modules:

```text
src.synthetic.export
src.data.export_v2_processed
src.modeling.v2_final_reporting
```

These actions regenerate the deterministic synthetic raw benchmark into a
disposable directory, rebuild the target-free processed feature artifact into
a disposable directory, and regenerate final reporting from already committed
post-access evaluation artifacts.

The runner does not invoke:

```text
src.modeling.v2_final_test_evaluation
src.modeling.v2_final_test_probabilities
src.modeling.v2_persistence
src.modeling.v2_development
src.modeling.v2_calibration
```

It therefore does not reopen the protected target, refit the frozen model,
recalibrate it, select a final-test threshold, or perform post-test tuning.

## Reproduction sequence

The runner requires an exact 40-character source commit. It verifies that the
canonical recovery checkout is clean and points to that commit, creates a
detached disposable Git worktree, and creates a new Python 3.12 virtual
environment outside the disposable checkout.

It then:

1. upgrades pip and installs `requirements.lock.txt`;
2. runs `python -m pip check`;
3. runs the complete test suite with pytest cache disabled and an isolated
   `--basetemp`;
4. regenerates the Version 2 synthetic raw benchmark into a disposable output
   directory and byte-compares its files with the committed frozen benchmark;
5. rebuilds the target-free Version 2 processed feature artifact into a
   disposable output directory and byte-compares it with the committed
   artifact;
6. regenerates the R3 final reporting package into disposable paths and
   byte-compares the summary, manifest, and figures with committed evidence;
7. verifies all four frozen Streamlit screenshot SHA-256 identities and all
   three frozen analytical figure identities;
8. audits repository-relative Markdown links and requires zero broken links;
9. audits tracked-tree hygiene for IDE, virtual-environment, pytest-cache,
   build, egg-info, bytecode, and `__pycache__` paths;
10. launches the Streamlit dashboard and requires
    `/_stcore/health` to return `ok`; and
11. verifies both the disposable checkout and canonical recovery checkout are
    clean after execution.

## Evidence output

Execution writes a JSON evidence file outside the disposable checkout. The
evidence records the exact source commit, Python versions, dependency-integrity
result, reproduction identities, Markdown-link result, tracked-tree hygiene,
Streamlit health result, and explicit false flags for protected-target
re-access, model refit, recalibration, threshold selection, and post-test
tuning.

The actual R5.1 reproduction run is authorized only after the runner commit
itself is exact-head CI-sealed.
