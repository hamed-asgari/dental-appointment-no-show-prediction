# Version 2 Recovery Phase R4 Formal Closeout

## Status

**R4 is complete and formally closed.**

The R4 portfolio-integration commit `568be27` was exact-commit CI-sealed before
this closeout. This closeout records the already-implemented application and
portfolio evidence; it does not modify the frozen model, calibration,
protected-test probability vector, protected-test evaluation, application
decision, or operational-threshold boundary.

## R4 checkpoint chain

- frozen R4 application contract: `c180a20`
- Streamlit evaluation dashboard implementation: `7cb6a83`
- current Streamlit width-API cleanup: `942cccf`
- portfolio integration and committed screenshots: `568be27`

## Exact portfolio CI seal

- workflow: `CI`
- event: `pull_request`
- run ID: `31217805371`
- conclusion: `success`
- exact sealed commit:
  `568be27e410a82f3a1acb94d4e40a337431113f7`

## Frozen application decision

```text
selected_app_type = transparent_model_evaluation_dashboard
```

The dashboard is a read-only model-evaluation presentation layer. It does not
accept patient or appointment inputs for individualized prediction, perform
runtime appointment scoring, expose an operational-threshold control, or invoke
the protected-target accessor.

## R4 acceptance evidence

The R4 application contract acceptance boundary is satisfied:

- the Streamlit application was runtime-smoke-tested under Python 3.12;
- dashboard behavior matches the pre-frozen evidence-based app decision;
- repository tests were green before portfolio integration was committed;
- the exact portfolio-integration commit was sealed by successful PR CI;
- four actual application screenshots are committed;
- launch instructions are reproducible from the repository root;
- stale current-state Version 1 app-absence statements were removed or
  explicitly contextualized as historical; and
- no protected-target re-access, model refit, recalibration, feature change,
  post-test tuning, or final-test threshold selection occurred.

## Frozen R4 artifact identities

- R4 application contract SHA-256:
  `1069cd5f66c6638fc858fb0767dc064dc84c4578a3247eec9daecb8289b25ef2`
- portfolio architecture SHA-256:
  `cd91aebc93dc616b351305e7eb81ca743a893a877c4a821550b4aeb88b0ff584`
- dashboard data layer SHA-256:
  `166255030a0c27e860d5374d3a8241af545fbbb152e6d580992a3d46080e0724`
- Streamlit presentation layer SHA-256:
  `e155c3ebd5b29acb446196625a262b145c79e3298915d95aa85a0cb577d1f6fb`
- screenshot manifest README SHA-256:
  `4e5320b013d84d903116968966d8de0b32f86ac94374f5c02083296567a5066c`

## Frozen screenshot identities

- overview:
  `809f42bcfacd919248663f515595bbb1810dcdc35a92e0f186e74446320d8e79`
- protected-test performance:
  `0e7f45905b1cf2940215ebf56b53351507f8ea84722ab5df13323d9acbc48c64`
- calibration/capacity view:
  `e6533a9432744a458327e1bfc7ee4a02b425040501bd42e2c6488a98b85a588a`
- interpretation/limitations view:
  `572772460187355a2cd0508284c2b58c87494bf069b52fb5256dfb63fd3913be`

## Post-test immutability

```text
target_access_count = 1
target_reaccess_performed = false
model_refit_performed = false
calibration_change_performed = false
final_test_threshold_selected = false
post_test_model_tuning_permitted = false
```

R4 presentation work did not reopen the protected final test as a model
selection resource. All Version 2 performance claims remain scoped to the
synthetic longitudinal benchmark.

## Gate into R5

R5 may now perform clean-environment reproduction, final documentation and
metadata consistency checks, release review, and Version `2.0.0` packaging.
R5 must preserve the frozen R3 model/evaluation boundary and the R4 application
decision.
