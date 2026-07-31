# Raw synthetic data

This directory is reserved for immutable source copies used by Project 2, **Dental Appointment No-show Prediction**.

Source files may later include appropriate fully synthetic datasets created for Project 1, **Dental Clinic Operations Analytics**. Any such files must be copied or reproducibly reconstructed here so that Project 2 remains separate. Project 1 is frozen at Version 1.0.0 and must never be modified by this project.

## Handling rules

- All data stored here must be fully synthetic. Real patient data, real clinical records, and real healthcare operational data do not belong in this directory.
- Preserve raw source files exactly as received or generated. Do not edit them manually after ingestion.
- Write transformed or cleaned outputs to `data/interim/` or `data/processed/`, never back to `data/raw/`.
- Record provenance whenever a dataset is added so its origin and handling are clear.
- Keep Project 2 copies independent of Project 1; no Project 2 process should write to the Project 1 repository.

## Provenance template

Create a provenance entry for each future source dataset using this template. Do not complete an entry until the corresponding file is present.

```text
Source dataset name:
Originating project/version:
Original filename:
Date copied/generated:
Transformation before raw ingestion, if any:
Notes:
```

No raw datasets or provenance entries are included in v0.1.
