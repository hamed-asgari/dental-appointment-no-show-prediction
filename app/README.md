# Application

## Current status

A Streamlit application was **not** implemented in Version `1.0.0`. This
directory is part of the Version `2.0.0` recovery scope.

The application type must be selected only after the recovered model is
evaluated under the renewed chronological evaluation policy:

1. **Appointment-level risk demonstration** - allowed only if the final model
   demonstrates useful appointment-level ranking and probability behavior
   beyond the population-prior baseline.
2. **Transparent model-evaluation dashboard** - required if individualized
   prediction remains unsupported.

The final application must include:

- a clear fully synthetic-data disclaimer;
- a statement that the project is not validated for clinical or operational use;
- documented launch instructions;
- input validation;
- behavior consistent with the evidence-based app decision gate;
- screenshots committed under `reports/screenshots/`.

No patient-level risk calculator should be built around the current constant
population-prior result.
