"""Deterministically export and verify the frozen Version 2 raw benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.synthetic.config import (
    DEFAULT_CONFIG_PATH,
    BenchmarkConfig,
    calculate_config_sha256,
    derive_rng_stream_seeds,
    load_benchmark_config,
)
from src.synthetic.generator import generate_synthetic_tables
from src.synthetic.schema import (
    APPOINTMENT_COLUMNS,
    DENTIST_COLUMNS,
    FORBIDDEN_EXPORTED_COLUMNS,
    PATIENT_COLUMNS,
)
from src.synthetic.tables import SyntheticTables
from src.synthetic.validation import validate_synthetic_tables


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILENAME = "v2_synthetic_benchmark.manifest.json"
TABLE_FILENAMES = {
    "patients": "patients.csv",
    "dentists": "dentists.csv",
    "appointments": "appointments.csv",
}
_SOURCE_PATHS = (
    "src/synthetic/config.py",
    "src/synthetic/generator.py",
    "src/synthetic/schema.py",
    "src/synthetic/tables.py",
    "src/synthetic/validation.py",
    "src/synthetic/export.py",
)
_DATE_COLUMNS = {
    "dentists": frozenset({"start_date", "end_date"}),
}
_DATETIME_COLUMNS = {
    "patients": frozenset({"registered_at"}),
    "appointments": frozenset(
        {
            "booked_at",
            "scheduled_start_at",
            "status_updated_at",
            "reminder_sent_at",
            "check_in_at",
            "chair_start_at",
            "chair_end_at",
            "checkout_at",
        }
    ),
}
_TABLE_COLUMNS = {
    "patients": PATIENT_COLUMNS,
    "dentists": DENTIST_COLUMNS,
    "appointments": APPOINTMENT_COLUMNS,
}


def calculate_sha256(path: Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"File is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _serialize_value(
    value: Any,
    *,
    table_name: str,
    column: str,
) -> str:
    if pd.isna(value):
        return ""
    if column in _DATE_COLUMNS.get(table_name, frozenset()):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if column in _DATETIME_COLUMNS.get(table_name, frozenset()):
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def _write_csv(
    frame: pd.DataFrame,
    *,
    table_name: str,
    destination: Path,
) -> None:
    expected_columns = _TABLE_COLUMNS[table_name]
    if tuple(frame.columns) != expected_columns:
        raise ValueError(
            f"{table_name} columns do not match the frozen export schema"
        )
    temporary = _temporary_path(destination)
    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.writer(
                stream,
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow(expected_columns)
            for row in frame.itertuples(index=False, name=None):
                writer.writerow(
                    [
                        _serialize_value(
                            value,
                            table_name=table_name,
                            column=column,
                        )
                        for column, value in zip(expected_columns, row)
                    ]
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _locked_dependency_version(
    repository_root: Path,
    package_name: str,
) -> str:
    lock_path = repository_root / "requirements.lock.txt"
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{package_name}=="):
            return line.split("==", maxsplit=1)[1]
    raise ValueError(
        f"{package_name} is missing from {lock_path.relative_to(repository_root)}"
    )


def _environment_metadata(
    repository_root: Path,
) -> dict[str, str]:
    with (repository_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    return {
        "python_requirement": str(project["requires-python"]),
        "dependency_lock": "requirements.lock.txt",
        "numpy_version": _locked_dependency_version(
            repository_root,
            "numpy",
        ),
        "pandas_version": _locked_dependency_version(
            repository_root,
            "pandas",
        ),
    }


def _source_hashes(
    repository_root: Path,
) -> dict[str, str]:
    return {
        relative: calculate_sha256(repository_root / relative)
        for relative in _SOURCE_PATHS
    }


def _evaluation_payload(
    config: BenchmarkConfig,
) -> dict[str, Any]:
    return {
        "warmup": {
            "start": config.evaluation.warmup.start.isoformat(),
            "end": config.evaluation.warmup.end.isoformat(),
        },
        "rolling_folds": [
            {
                "name": fold.name,
                "fit_start": fold.fit.start.isoformat(),
                "fit_end": fold.fit.end.isoformat(),
                "validation_start": fold.validation.start.isoformat(),
                "validation_end": fold.validation.end.isoformat(),
            }
            for fold in config.evaluation.rolling_folds
        ],
        "calibration": {
            "start": config.evaluation.calibration.start.isoformat(),
            "end": config.evaluation.calibration.end.isoformat(),
        },
        "policy_selection": {
            "start": config.evaluation.policy_selection.start.isoformat(),
            "end": config.evaluation.policy_selection.end.isoformat(),
        },
        "final_test": {
            "start": config.evaluation.final_test.start.isoformat(),
            "end": config.evaluation.final_test.end.isoformat(),
        },
    }


def _timestamp_range(
    series: pd.Series,
) -> dict[str, str | None]:
    non_missing = series.dropna()
    if non_missing.empty:
        return {"minimum": None, "maximum": None}
    return {
        "minimum": pd.Timestamp(non_missing.min()).isoformat(sep=" "),
        "maximum": pd.Timestamp(non_missing.max()).isoformat(sep=" "),
    }


def _table_manifest_entry(
    table_name: str,
    frame: pd.DataFrame,
    *,
    filename: str,
    sha256: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "filename": filename,
        "sha256": sha256,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
    }
    if table_name == "patients":
        entry["registered_at_range"] = _timestamp_range(
            frame["registered_at"]
        )
        entry["patient_status_counts"] = {
            str(key): int(value)
            for key, value in frame["patient_status"].value_counts(
                sort=False
            ).sort_index().items()
        }
    elif table_name == "dentists":
        entry["start_date_range"] = _timestamp_range(frame["start_date"])
        entry["active_counts"] = {
            str(bool(key)).lower(): int(value)
            for key, value in frame["active"].value_counts(
                sort=False
            ).sort_index().items()
        }
    elif table_name == "appointments":
        entry["scheduled_start_at_range"] = _timestamp_range(
            frame["scheduled_start_at"]
        )
        entry["prediction_time_range"] = _timestamp_range(
            frame["scheduled_start_at"]
            - pd.Timedelta(hours=24)
        )
        entry["status_counts"] = {
            str(key): int(value)
            for key, value in frame["status"].value_counts(
                sort=False
            ).sort_index().items()
        }
        entry["reminder_sent_counts"] = {
            str(bool(key)).lower(): int(value)
            for key, value in frame["reminder_sent"].value_counts(
                sort=False
            ).sort_index().items()
        }
    return entry


def _dataset_fingerprint(
    *,
    config_sha256: str,
    source_hashes: Mapping[str, str],
    table_hashes: Mapping[str, str],
) -> str:
    payload = {
        "config_sha256": config_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "table_hashes": dict(sorted(table_hashes.items())),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_manifest(
    tables: SyntheticTables,
    *,
    config: BenchmarkConfig,
    config_path: Path,
    repository_root: Path,
    table_hashes: Mapping[str, str],
) -> dict[str, Any]:
    config_sha256 = calculate_config_sha256(config_path)
    source_hashes = _source_hashes(repository_root)
    fingerprint = _dataset_fingerprint(
        config_sha256=config_sha256,
        source_hashes=source_hashes,
        table_hashes=table_hashes,
    )
    frames = {
        "patients": tables.patients,
        "dentists": tables.dentists,
        "appointments": tables.appointments,
    }
    return {
        "manifest_version": "1.0.0",
        "dataset_name": "v2_synthetic_benchmark",
        "synthetic_data": True,
        "validated_for_clinical_use": False,
        "schema_version": config.schema_version,
        "generator_version": config.generator_version,
        "root_seed": config.root_seed,
        "configuration": {
            "path": config_path.relative_to(repository_root).as_posix(),
            "sha256": config_sha256,
        },
        "generator_source_sha256": source_hashes,
        "rng_stream_seeds": {
            key: int(value)
            for key, value in derive_rng_stream_seeds(config).items()
        },
        "generation_environment": _environment_metadata(repository_root),
        "prediction_contract": {
            "horizon_hours": config.prediction_horizon_hours,
            "prediction_time_formula": (
                "scheduled_start_at - prediction_horizon_hours"
            ),
            "historical_outcome_rule": (
                "historical_status_updated_at < current_prediction_time"
            ),
        },
        "evaluation": _evaluation_payload(config),
        "tables": {
            table_name: _table_manifest_entry(
                table_name,
                frames[table_name],
                filename=TABLE_FILENAMES[table_name],
                sha256=table_hashes[table_name],
            )
            for table_name in TABLE_FILENAMES
        },
        "dataset_fingerprint": fingerprint,
    }


def _write_manifest(
    manifest: Mapping[str, Any],
    destination: Path,
) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_exported_benchmark(
    output_dir: Path,
) -> dict[str, Any]:
    """Verify an exported benchmark against its own deterministic manifest."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read manifest: {manifest_path}"
        ) from exc

    if manifest.get("dataset_name") != "v2_synthetic_benchmark":
        raise ValueError("Manifest dataset name is invalid")
    table_entries = manifest.get("tables")
    if not isinstance(table_entries, dict):
        raise ValueError("Manifest tables section is invalid")

    for table_name, filename in TABLE_FILENAMES.items():
        entry = table_entries.get(table_name)
        if not isinstance(entry, dict):
            raise ValueError(
                f"Manifest entry is missing for {table_name}"
            )
        if entry.get("filename") != filename:
            raise ValueError(
                f"Manifest filename is invalid for {table_name}"
            )
        path = output_dir / filename
        actual_hash = calculate_sha256(path)
        if actual_hash != entry.get("sha256"):
            raise ValueError(
                f"SHA-256 mismatch for {filename}: "
                f"expected {entry.get('sha256')}, got {actual_hash}"
            )

    expected_fingerprint = _dataset_fingerprint(
        config_sha256=manifest["configuration"]["sha256"],
        source_hashes=manifest["generator_source_sha256"],
        table_hashes={
            table_name: table_entries[table_name]["sha256"]
            for table_name in TABLE_FILENAMES
        },
    )
    if manifest.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("Dataset fingerprint does not match manifest inputs")
    return manifest


def export_synthetic_benchmark(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path | None = None,
    overwrite: bool = False,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Generate, atomically export, and verify the Version 2 benchmark."""

    repository_root = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_benchmark_config(config_path)
    destination = (
        repository_root / config.output_directory
        if output_dir is None
        else Path(output_dir)
    ).resolve()
    required_paths = {
        destination / filename
        for filename in TABLE_FILENAMES.values()
    }
    required_paths.add(destination / MANIFEST_FILENAME)
    existing = {path for path in required_paths if path.exists()}

    if existing and not overwrite:
        if existing == required_paths:
            return verify_exported_benchmark(destination)
        names = sorted(path.name for path in existing)
        raise ValueError(
            "Output directory contains an incomplete existing export: "
            f"{names}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    tables = generate_synthetic_tables(config)
    validate_synthetic_tables(tables, config)

    frames = {
        "patients": tables.patients,
        "dentists": tables.dentists,
        "appointments": tables.appointments,
    }
    exported_columns = set().union(
        *(set(frame.columns) for frame in frames.values())
    )
    forbidden = sorted(exported_columns & FORBIDDEN_EXPORTED_COLUMNS)
    if forbidden:
        raise ValueError(
            f"Forbidden latent columns reached export: {forbidden}"
        )

    for table_name, filename in TABLE_FILENAMES.items():
        _write_csv(
            frames[table_name],
            table_name=table_name,
            destination=destination / filename,
        )

    table_hashes = {
        table_name: calculate_sha256(destination / filename)
        for table_name, filename in TABLE_FILENAMES.items()
    }
    manifest = _build_manifest(
        tables,
        config=config,
        config_path=config_path,
        repository_root=repository_root,
        table_hashes=table_hashes,
    )
    _write_manifest(
        manifest,
        destination / MANIFEST_FILENAME,
    )
    return verify_exported_benchmark(destination)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or verify the frozen Version 2 synthetic raw benchmark."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Frozen benchmark configuration path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional noncanonical output directory. "
            "The default comes from the frozen configuration."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing complete or partial export.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing export without regenerating data.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run benchmark export or verification from the command line."""

    args = _build_parser().parse_args(argv)
    config = load_benchmark_config(args.config)
    output_dir = (
        _REPOSITORY_ROOT / config.output_directory
        if args.output_dir is None
        else args.output_dir
    )
    if args.verify_only:
        manifest = verify_exported_benchmark(output_dir)
        print(
            "Verified Version 2 synthetic benchmark: "
            f"{manifest['dataset_fingerprint']}"
        )
        return 0

    manifest = export_synthetic_benchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Wrote Version 2 synthetic benchmark to {Path(output_dir)}")
    for table_name in TABLE_FILENAMES:
        entry = manifest["tables"][table_name]
        print(
            f"{entry['filename']}: "
            f"{entry['row_count']:,} rows, {entry['sha256']}"
        )
    print(
        "Dataset fingerprint: "
        f"{manifest['dataset_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "MANIFEST_FILENAME",
    "TABLE_FILENAMES",
    "calculate_sha256",
    "export_synthetic_benchmark",
    "main",
    "verify_exported_benchmark",
)
