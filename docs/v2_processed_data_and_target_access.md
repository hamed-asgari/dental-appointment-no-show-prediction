# Version 2 processed data and protected target access

## Status

Implemented under the frozen Version 2 historical-feature and evaluation
contracts.

## Target-free processed artifact

The canonical Version 2 processed artifact is:

```text
data/processed/v2/v2_feature_dataset.csv
```

It contains:

- audit identifiers used only for alignment;
- `prediction_time`;
- the exact 32-column model-feature allowlist;
- the frozen evaluation partition; and
- `label_available_at` maturity metadata.

It does not contain:

- `target`;
- current or future `status`;
- post-outcome timestamps;
- hidden synthetic effects; or
- final-test outcomes.

The companion manifest records exact hashes, source identities, builder source
hashes, dtypes, partition counts, and the processed dataset fingerprint. The
manifest must declare:

```text
target_included = false
final_test_target_accessed = false
```

## Development target access

Non-test labels are built only for explicitly requested partitions and only
when:

```text
label_available_at < model_fit_time
```

The comparison is strict. Equal-time labels are unavailable. The public
maturity accessor rejects `final_test` even when it appears beside an allowed
development partition.

The binary target is:

```text
1 for no_show
0 for completed, cancelled, or rescheduled
```

Cancelled and rescheduled appointments remain target zero only when the row was
eligible at its own prediction time.

## Protected final-test access

Final-test target access is a separate operation. It requires all of the
following:

1. explicit `allow_test=True`;
2. a previously written CSV containing exactly:
   ```text
   appointment_id,no_show_probability
   ```
3. one unique probability for every final-test appointment;
4. exact final-test appointment order;
5. finite probabilities in `[0, 1]`; and
6. validation and SHA-256 sealing of that probability artifact before the raw
   status column is joined.

The production loader validates the target-free processed feature artifact and
the probability vector before loading the raw appointment statuses.

Repository tests do not perform a successful access against the real protected
2027 targets. They test the successful path only with small synthetic fixtures
and test denial paths against the frozen benchmark.

## Artifact integrity

The processed CSV uses deterministic UTF-8 serialization and LF line endings.
It is marked binary in `.gitattributes` so Git cannot rewrite its bytes.

The committed frozen-hash module records:

- processed CSV SHA-256;
- processed manifest SHA-256; and
- processed dataset fingerprint.

A mismatch stops loading.

## Version 1 isolation

The Version 1 builder, results, and protected outputs are not modified. Version
2 uses separate modules and `data/processed/v2/`.
