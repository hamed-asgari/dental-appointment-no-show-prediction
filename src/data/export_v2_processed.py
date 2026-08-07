"""Deterministic target-free Version 2 processed-data export."""

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

from src.data.build_v2_dataset import (
    DEFAULT_V2_MANIFEST_PATH,
    build_verified_v2_feature_dataset,
    calculate_sha256,
    select_v2_model_features,
    validate_frozen_v2_inputs,
)
from src.features.schema import (
    AGGREGATE_HISTORY_DTYPES,
    CURRENT_APPOINTMENT_DTYPES,
    PATIENT_HISTORY_DTYPES,
    V2_FEATURE_DATASET_COLUMNS,
    V2_MODEL_FEATURE_COLUMNS,
    V2_PROHIBITED_MODEL_COLUMNS,
)
from src.synthetic.config import (
    DEFAULT_CONFIG_PATH,
    calculate_config_sha256,
)
from src.synthetic.frozen_hashes import (
    FROZEN_V2_DATASET_FINGERPRINT,
    FROZEN_V2_MANIFEST_SHA256,
    FROZEN_V2_RAW_HASHES,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_PROCESSED_DIR = _REPOSITORY_ROOT / "data" / "processed" / "v2"
PROCESSED_DATASET_FILENAME = "v2_feature_dataset.csv"
PROCESSED_MANIFEST_FILENAME = "v2_feature_dataset.manifest.json"
PROCESSED_DATASET_NAME = "v2_feature_dataset"
PROCESSED_SCHEMA_VERSION = "2.0.0"
_HISTORICAL_FEATURE_CONTRACT = "docs/v2_historical_feature_contract.md"
_SOURCE_PATHS = (
    "src/features/schema.py",
    "src/features/current_appointment.py",
    "src/features/asof_history.py",
    "src/features/aggregate_history.py",
    "src/data/build_v2_dataset.py",
    "src/data/export_v2_processed.py",
)
_DATETIME_COLUMNS = frozenset({"prediction_time", "label_available_at"})
_BOOLEAN_COLUMNS = frozenset(
    {
        column
        for column, dtype in {
            **dict(CURRENT_APPOINTMENT_DTYPES),
            **dict(PATIENT_HISTORY_DTYPES),
            **dict(AGGREGATE_HISTORY_DTYPES),
        }.items()
        if dtype == "bool"
    }
)
_INTEGER_COLUMNS = frozenset(
    {
        "appointment_id",
        "patient_id",
        "dentist_id",
        *(
            column
            for column, dtype in {
                **dict(CURRENT_APPOINTMENT_DTYPES),
                **dict(PATIENT_HISTORY_DTYPES),
                **dict(AGGREGATE_HISTORY_DTYPES),
            }.items()
            if dtype.startswith("int")
        ),
    }
)
_FLOAT_COLUMNS = frozenset(
    {
        column
        for column, dtype in {
            **dict(CURRENT_APPOINTMENT_DTYPES),
            **dict(PATIENT_HISTORY_DTYPES),
            **dict(AGGREGATE_HISTORY_DTYPES),
        }.items()
        if dtype == "float64"
    }
)
_STRING_COLUMNS = frozenset(
    set(V2_FEATURE_DATASET_COLUMNS)
    - _DATETIME_COLUMNS
    - _BOOLEAN_COLUMNS
    - _INTEGER_COLUMNS
    - _FLOAT_COLUMNS
)


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _serialize_value(value: Any, *, column: str) -> str:
    if pd.isna(value):
        raise ValueError(
            f"Processed feature dataset contains a missing value in {column}"
        )
    if column in _DATETIME_COLUMNS:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if column in _BOOLEAN_COLUMNS:
        return "true" if bool(value) else "false"
    if column in _INTEGER_COLUMNS:
        return str(int(value))
    if column in _FLOAT_COLUMNS:
        return format(float(value), ".17g")
    return str(value)


def _write_feature_csv(frame: pd.DataFrame, destination: Path) -> None:
    if tuple(frame.columns) != V2_FEATURE_DATASET_COLUMNS:
        raise ValueError(
            "Processed feature columns do not match the frozen Version 2 schema"
        )
    if frame.empty:
        raise ValueError("Processed feature dataset must not be empty")
    if "target" in frame.columns:
        raise ValueError("Processed feature export must not contain target")

    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(
                stream,
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow(V2_FEATURE_DATASET_COLUMNS)
            for row in frame.itertuples(index=False, name=None):
                writer.writerow(
                    [
                        _serialize_value(value, column=column)
                        for column, value in zip(
                            V2_FEATURE_DATASET_COLUMNS,
                            row,
                        )
                    ]
                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _source_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: calculate_sha256(repository_root / relative)
        for relative in _SOURCE_PATHS
    }


def _locked_dependency_version(
    repository_root: Path,
    package_name: str,
) -> str:
    lock_path = repository_root / "requirements.lock.txt"
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{package_name}=="):
            return line.split("==", maxsplit=1)[1]
    raise ValueError(f"{package_name} is missing from {lock_path}")


def _dtype_payload(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(frame[column].dtype) for column in frame.columns}


def _timestamp_range(series: pd.Series) -> dict[str, str]:
    return {
        "minimum": pd.Timestamp(series.min()).isoformat(sep=" "),
        "maximum": pd.Timestamp(series.max()).isoformat(sep=" "),
    }


def _processed_fingerprint(
    *,
    artifact_sha256: str,
    source_identity: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> str:
    payload = {
        "artifact_sha256": artifact_sha256,
        "source_identity": source_identity,
        "source_hashes": dict(sorted(source_hashes.items())),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_manifest(
    frame: pd.DataFrame,
    *,
    artifact_sha256: str,
    repository_root: Path,
    config_path: Path,
    raw_manifest_path: Path,
) -> dict[str, Any]:
    identities = validate_frozen_v2_inputs(
        raw_dir=raw_manifest_path.parent,
        manifest_path=raw_manifest_path,
        config_path=config_path,
    )
    contract_path = repository_root / _HISTORICAL_FEATURE_CONTRACT
    source_hashes = _source_hashes(repository_root)
    source_identity: dict[str, Any] = {
        "configuration_sha256": calculate_config_sha256(config_path),
        "raw_manifest_sha256": identities["manifest"],
        "raw_dataset_fingerprint": FROZEN_V2_DATASET_FINGERPRINT,
        "raw_file_sha256": dict(FROZEN_V2_RAW_HASHES),
        "historical_feature_contract_sha256": calculate_sha256(contract_path),
    }
    fingerprint = _processed_fingerprint(
        artifact_sha256=artifact_sha256,
        source_identity=source_identity,
        source_hashes=source_hashes,
    )
    return {
        "dataset_name": PROCESSED_DATASET_NAME,
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "target_included": False,
        "final_test_target_accessed": False,
        "artifact": {
            "filename": PROCESSED_DATASET_FILENAME,
            "sha256": artifact_sha256,
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "columns": list(frame.columns),
            "dtypes": _dtype_payload(frame),
            "model_feature_count": len(V2_MODEL_FEATURE_COLUMNS),
            "model_feature_columns": list(V2_MODEL_FEATURE_COLUMNS),
            "partition_counts": {
                str(key): int(value)
                for key, value in frame["evaluation_partition"]
                .value_counts(sort=False)
                .sort_index()
                .items()
            },
            "prediction_time_range": _timestamp_range(
                frame["prediction_time"]
            ),
            "label_available_at_range": _timestamp_range(
                frame["label_available_at"]
            ),
        },
        "source_identity": source_identity,
        "builder_source_sha256": source_hashes,
        "environment": {
            "python_requirement": ">=3.12,<3.13",
            "dependency_lock": "requirements.lock.txt",
            "numpy_version": _locked_dependency_version(
                repository_root,
                "numpy",
            ),
            "pandas_version": _locked_dependency_version(
                repository_root,
                "pandas",
            ),
        },
        "processed_dataset_fingerprint": fingerprint,
    }


def _write_manifest(manifest: Mapping[str, Any], destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_exported_v2_processed_dataset(
    output_dir: Path = DEFAULT_V2_PROCESSED_DIR,
) -> dict[str, Any]:
    """Verify a target-free processed export against its own manifest."""

    output_dir = Path(output_dir)
    dataset_path = output_dir / PROCESSED_DATASET_FILENAME
    manifest_path = output_dir / PROCESSED_MANIFEST_FILENAME
    if not dataset_path.is_file():
        raise ValueError(f"Processed dataset is missing: {dataset_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Processed manifest is missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read processed manifest: {manifest_path}"
        ) from exc

    if manifest.get("dataset_name") != PROCESSED_DATASET_NAME:
        raise ValueError("Processed manifest dataset name is invalid")
    if manifest.get("schema_version") != PROCESSED_SCHEMA_VERSION:
        raise ValueError("Processed manifest schema version is invalid")
    if manifest.get("target_included") is not False:
        raise ValueError("Processed manifest must declare target_included=false")
    if manifest.get("final_test_target_accessed") is not False:
        raise ValueError(
            "Processed manifest must declare final_test_target_accessed=false"
        )

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Processed manifest artifact block is invalid")
    if artifact.get("filename") != PROCESSED_DATASET_FILENAME:
        raise ValueError("Processed manifest artifact filename is invalid")
    actual_hash = calculate_sha256(dataset_path)
    if artifact.get("sha256") != actual_hash:
        raise ValueError(
            "Processed feature dataset SHA-256 mismatch: "
            f"expected={artifact.get('sha256')}, actual={actual_hash}"
        )
    if artifact.get("columns") != list(V2_FEATURE_DATASET_COLUMNS):
        raise ValueError("Processed manifest feature columns are invalid")
    if "target" in artifact.get("columns", []):
        raise ValueError("Processed manifest must not expose target")
    if artifact.get("model_feature_columns") != list(V2_MODEL_FEATURE_COLUMNS):
        raise ValueError("Processed manifest model-feature allowlist is invalid")
    if set(V2_MODEL_FEATURE_COLUMNS) & V2_PROHIBITED_MODEL_COLUMNS:
        raise RuntimeError("Frozen model feature allowlist contains prohibited columns")

    source_identity = manifest.get("source_identity")
    source_hashes = manifest.get("builder_source_sha256")
    if not isinstance(source_identity, dict) or not isinstance(source_hashes, dict):
        raise ValueError("Processed manifest source identity is invalid")
    expected_source_identity = {
        "configuration_sha256": calculate_config_sha256(DEFAULT_CONFIG_PATH),
        "raw_manifest_sha256": FROZEN_V2_MANIFEST_SHA256,
        "raw_dataset_fingerprint": FROZEN_V2_DATASET_FINGERPRINT,
        "raw_file_sha256": dict(FROZEN_V2_RAW_HASHES),
        "historical_feature_contract_sha256": calculate_sha256(
            _REPOSITORY_ROOT / _HISTORICAL_FEATURE_CONTRACT
        ),
    }
    if source_identity != expected_source_identity:
        raise ValueError("Processed manifest source identity is not frozen")
    expected_fingerprint = _processed_fingerprint(
        artifact_sha256=actual_hash,
        source_identity=source_identity,
        source_hashes=source_hashes,
    )
    if manifest.get("processed_dataset_fingerprint") != expected_fingerprint:
        raise ValueError("Processed dataset fingerprint is invalid")
    return manifest


def _cast_exported_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != V2_FEATURE_DATASET_COLUMNS:
        raise ValueError("Processed CSV columns do not match the frozen schema")
    if frame.empty:
        raise ValueError("Processed CSV must not be empty")
    if frame.isna().any().any():
        raise ValueError("Processed CSV must not contain missing values")

    result = frame.copy(deep=True)
    for column in _DATETIME_COLUMNS:
        result[column] = pd.to_datetime(
            result[column],
            errors="raise",
            format="mixed",
        ).astype("datetime64[ns]")
    for column in _BOOLEAN_COLUMNS:
        values = result[column].astype("string").str.casefold()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Processed boolean column is invalid: {column}")
        result[column] = values.eq("true").astype("bool")
    for column in _INTEGER_COLUMNS:
        target_dtype = "int64"
        for mapping in (
            CURRENT_APPOINTMENT_DTYPES,
            PATIENT_HISTORY_DTYPES,
            AGGREGATE_HISTORY_DTYPES,
        ):
            if column in mapping:
                target_dtype = mapping[column]
                break
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        ).astype(target_dtype)
    for column in _FLOAT_COLUMNS:
        try:
            result[column] = result[column].map(float).astype("float64")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Processed float column is invalid: {column}"
            ) from exc
    for column in _STRING_COLUMNS:
        result[column] = result[column].astype("string")
    return result.loc[:, list(V2_FEATURE_DATASET_COLUMNS)].copy(deep=True)


def load_frozen_v2_processed_feature_dataset(
    output_dir: Path = DEFAULT_V2_PROCESSED_DIR,
) -> pd.DataFrame:
    """Load the committed target-free export after frozen-hash verification."""

    from src.data.v2_processed_hashes import (
        FROZEN_V2_PROCESSED_DATASET_FINGERPRINT,
        FROZEN_V2_PROCESSED_DATASET_SHA256,
        FROZEN_V2_PROCESSED_MANIFEST_SHA256,
    )

    output_dir = Path(output_dir)
    manifest = verify_exported_v2_processed_dataset(output_dir)
    dataset_path = output_dir / PROCESSED_DATASET_FILENAME
    manifest_path = output_dir / PROCESSED_MANIFEST_FILENAME
    if calculate_sha256(dataset_path) != FROZEN_V2_PROCESSED_DATASET_SHA256:
        raise ValueError("Frozen processed feature dataset SHA-256 mismatch")
    if calculate_sha256(manifest_path) != FROZEN_V2_PROCESSED_MANIFEST_SHA256:
        raise ValueError("Frozen processed manifest SHA-256 mismatch")
    if (
        manifest.get("processed_dataset_fingerprint")
        != FROZEN_V2_PROCESSED_DATASET_FINGERPRINT
    ):
        raise ValueError("Frozen processed dataset fingerprint mismatch")

    try:
        raw = pd.read_csv(dataset_path, dtype="string")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read frozen processed feature dataset: {dataset_path}"
        ) from exc
    result = _cast_exported_frame(raw)
    if len(result) != manifest["artifact"]["row_count"]:
        raise ValueError("Processed CSV row count does not match the manifest")
    if "target" in result.columns:
        raise RuntimeError("Frozen processed feature dataset exposed target")
    return result


def export_v2_processed_feature_dataset(
    *,
    output_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    overwrite: bool = False,
    repository_root: Path = _REPOSITORY_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    raw_manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
) -> dict[str, Any]:
    """Build, atomically export, and verify the target-free Version 2 dataset."""

    destination = Path(output_dir).resolve()
    repository_root = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    raw_manifest_path = Path(raw_manifest_path).resolve()
    dataset_path = destination / PROCESSED_DATASET_FILENAME
    manifest_path = destination / PROCESSED_MANIFEST_FILENAME
    required = {dataset_path, manifest_path}
    existing = {path for path in required if path.exists()}

    if existing and not overwrite:
        if existing == required:
            return verify_exported_v2_processed_dataset(destination)
        raise ValueError(
            "Processed output directory contains an incomplete export: "
            + ", ".join(sorted(path.name for path in existing))
        )

    destination.mkdir(parents=True, exist_ok=True)
    frame = build_verified_v2_feature_dataset(
        raw_dir=raw_manifest_path.parent,
        manifest_path=raw_manifest_path,
        config_path=config_path,
    )
    selected = select_v2_model_features(frame)
    if selected.shape[1] != len(V2_MODEL_FEATURE_COLUMNS):
        raise RuntimeError("Processed model-feature count is invalid")
    if "target" in frame.columns:
        raise RuntimeError("Target reached processed feature export")

    _write_feature_csv(frame, dataset_path)
    artifact_hash = calculate_sha256(dataset_path)
    manifest = _build_manifest(
        frame,
        artifact_sha256=artifact_hash,
        repository_root=repository_root,
        config_path=config_path,
        raw_manifest_path=raw_manifest_path,
    )
    _write_manifest(manifest, manifest_path)
    return verify_exported_v2_processed_dataset(destination)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the target-free Version 2 processed feature dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_V2_PROCESSED_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.verify_only:
        manifest = verify_exported_v2_processed_dataset(args.output_dir)
        print(
            "Verified Version 2 processed feature dataset: "
            f"{manifest['processed_dataset_fingerprint']}"
        )
        return 0

    manifest = export_v2_processed_feature_dataset(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    artifact = manifest["artifact"]
    print(f"Wrote target-free Version 2 features to {Path(args.output_dir)}")
    print(
        f"{artifact['filename']}: {artifact['row_count']:,} rows, "
        f"{artifact['sha256']}"
    )
    print(
        "Processed dataset fingerprint: "
        f"{manifest['processed_dataset_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_V2_PROCESSED_DIR",
    "PROCESSED_DATASET_FILENAME",
    "PROCESSED_MANIFEST_FILENAME",
    "export_v2_processed_feature_dataset",
    "load_frozen_v2_processed_feature_dataset",
    "main",
    "verify_exported_v2_processed_dataset",
)
