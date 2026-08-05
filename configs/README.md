# Configuration

`v2_synthetic_benchmark.json` is the frozen, machine-readable configuration
for the Version 2 longitudinal synthetic benchmark.

The file is governed by
[`docs/v2_data_generation_and_evaluation_policy.md`](../docs/v2_data_generation_and_evaluation_policy.md).
Changing its generator version, seed, population sizes, date boundaries,
random-stream names, or protected evaluation windows creates a new benchmark
version and invalidates previously frozen hashes and protected-test claims.
