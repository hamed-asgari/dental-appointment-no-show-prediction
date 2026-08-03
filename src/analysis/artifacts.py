"""Deterministically serialize the approved in-memory EDA table bundle."""

from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import tempfile

import pandas as pd


_TABLE_ARTIFACTS = (
    ("cohort_target", "01_cohort_target.csv"),
    ("missingness", "02_missingness.csv"),
    ("numeric_features", "03_numeric_features.csv"),
    ("numeric_by_target", "04_numeric_by_target.csv"),
    ("categorical_features", "05_categorical_features.csv"),
    ("temporal_coverage", "06_temporal_coverage.csv"),
    ("temporal_monthly", "07_temporal_monthly.csv"),
    ("numeric_drift", "08_numeric_drift.csv"),
    ("categorical_drift_levels", "09_categorical_drift_levels.csv"),
    ("categorical_drift_features", "10_categorical_drift_features.csv"),
    ("numeric_relationships", "11_numeric_relationships.csv"),
)
_TABLE_KEYS = tuple(key for key, _ in _TABLE_ARTIFACTS)


def _validate_tables(tables: object) -> tuple[pd.DataFrame, ...]:
    if type(tables) is not dict:
        raise TypeError("tables must be exactly a built-in dict")

    keys = tuple(tables)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("tables keys must all be strings")
    if keys != _TABLE_KEYS:
        raise ValueError(
            "tables keys and insertion order must be exactly "
            f"{_TABLE_KEYS}; got {keys}"
        )

    frames = tuple(tables[key] for key in _TABLE_KEYS)
    for key, frame in zip(_TABLE_KEYS, frames):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"tables[{key!r}] must be a pandas DataFrame")
    if len({id(frame) for frame in frames}) != len(frames):
        raise ValueError("each table key must reference a distinct DataFrame object")
    return frames


def _render_csv(frame: pd.DataFrame) -> bytes:
    buffer = StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    return buffer.getvalue().encode("utf-8")


def _cleanup_temporary_file(
    temporary_path: Path,
    operation_error: OSError,
) -> None:
    try:
        temporary_path.unlink(missing_ok=True)
    except OSError as cleanup_error:
        add_note = getattr(operation_error, "add_note", None)
        if add_note is not None:
            add_note(
                "Temporary-file cleanup failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def _write_atomic(payload: bytes, destination: Path) -> None:
    temporary_file = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    try:
        with temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except OSError as operation_error:
        _cleanup_temporary_file(temporary_path, operation_error)
        raise


def write_eda_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write one validated EDA table bundle as eleven deterministic CSV files."""

    frames = _validate_tables(tables)
    payloads = tuple(_render_csv(frame) for frame in frames)

    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(
            f"output_dir exists and is not a directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)

    paths = {
        key: directory / filename
        for key, filename in _TABLE_ARTIFACTS
    }
    for key, payload in zip(_TABLE_KEYS, payloads):
        _write_atomic(payload, paths[key])
    return paths
