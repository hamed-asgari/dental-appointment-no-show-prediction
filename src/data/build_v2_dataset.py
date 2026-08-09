"""Verified feature-only Version 2 analytical dataset construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import pandas as pd

from src.features.aggregate_history import (
    build_aggregate_history_features,
)
from src.features.asof_history import build_patient_history_features
from src.features.current_appointment import (
    build_current_appointment_features,
)
from src.features.schema import (
    V2_EVALUATION_PARTITIONS,
    V2_FEATURE_DATASET_COLUMNS,
    V2_MODEL_FEATURE_COLUMNS,
    V2_PROHIBITED_MODEL_COLUMNS,
)
from src.synthetic.config import (
    BenchmarkConfig,
    DEFAULT_CONFIG_PATH,
    calculate_config_sha256,
    load_benchmark_config,
)
from src.synthetic.frozen_hashes import (
    FROZEN_V2_DATASET_FINGERPRINT,
    FROZEN_V2_MANIFEST_SHA256,
    FROZEN_V2_RAW_HASHES,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw" / "v2"
DEFAULT_V2_MANIFEST_PATH = (
    DEFAULT_V2_RAW_DIR / "v2_synthetic_benchmark.manifest.json"
)


@dataclass(frozen=True, slots=True)
class V2RawTables:
    """Verified Version 2 raw source tables."""

    appointments: pd.DataFrame
    patients: pd.DataFrame
    dentists: pd.DataFrame


def calculate_sha256(path: Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_v2_inputs(
    *,
    raw_dir: Path = DEFAULT_V2_RAW_DIR,
    manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Mapping[str, str]:
    """Validate the exact frozen Version 2 benchmark identity."""

    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)

    if not raw_dir.is_dir():
        raise ValueError(f"Version 2 raw directory is missing: {raw_dir}")

    actual_manifest_hash = calculate_sha256(manifest_path)
    if actual_manifest_hash != FROZEN_V2_MANIFEST_SHA256:
        raise ValueError(
            "Version 2 manifest SHA-256 mismatch: "
            f"expected={FROZEN_V2_MANIFEST_SHA256}, "
            f"actual={actual_manifest_hash}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read Version 2 manifest: {manifest_path}"
        ) from exc

    if (
        manifest.get("dataset_fingerprint")
        != FROZEN_V2_DATASET_FINGERPRINT
    ):
        raise ValueError("Version 2 dataset fingerprint mismatch")

    actual_config_hash = calculate_config_sha256(config_path)
    manifest_config = manifest.get("configuration")
    if not isinstance(manifest_config, dict):
        raise ValueError("Version 2 manifest configuration block is invalid")
    if manifest_config.get("sha256") != actual_config_hash:
        raise ValueError(
            "Version 2 configuration hash does not match the manifest"
        )

    validated: dict[str, str] = {
        "manifest": actual_manifest_hash,
        "configuration": actual_config_hash,
    }
    manifest_tables = manifest.get("tables")
    if not isinstance(manifest_tables, dict):
        raise ValueError("Version 2 manifest tables block is invalid")

    for filename, expected_hash in FROZEN_V2_RAW_HASHES.items():
        path = raw_dir / filename
        actual_hash = calculate_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Version 2 raw SHA-256 mismatch for {filename}: "
                f"expected={expected_hash}, actual={actual_hash}"
            )

        table_name = Path(filename).stem
        table_manifest = manifest_tables.get(table_name)
        if not isinstance(table_manifest, dict):
            raise ValueError(
                f"Version 2 manifest is missing table {table_name!r}"
            )
        if table_manifest.get("sha256") != actual_hash:
            raise ValueError(
                f"Version 2 manifest table hash mismatch for {filename}"
            )
        validated[filename] = actual_hash

    return MappingProxyType(validated)


def load_verified_v2_raw_tables(
    *,
    raw_dir: Path = DEFAULT_V2_RAW_DIR,
    manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> V2RawTables:
    """Load raw tables only after frozen identity verification."""

    raw_dir = Path(raw_dir)
    validate_frozen_v2_inputs(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        config_path=config_path,
    )

    try:
        appointments = pd.read_csv(raw_dir / "appointments.csv")
        patients = pd.read_csv(raw_dir / "patients.csv")
        dentists = pd.read_csv(raw_dir / "dentists.csv")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read verified Version 2 raw tables from {raw_dir}"
        ) from exc

    return V2RawTables(
        appointments=appointments,
        patients=patients,
        dentists=dentists,
    )


def _as_naive_datetime_series(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    try:
        converted = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain valid timestamps") from exc
    if converted.isna().any():
        raise ValueError(f"{context} must not contain missing values")
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{context} must be timezone-naive")
    return converted.astype("datetime64[ns]")


def _window_mask(
    values: pd.Series,
    *,
    start: object,
    end: object,
) -> pd.Series:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    return values.ge(start_timestamp) & values.lt(end_timestamp)


def assign_v2_evaluation_partitions(
    prediction_time: pd.Series,
    config: BenchmarkConfig,
) -> pd.Series:
    """Assign disjoint frozen Version 2 evaluation partitions."""

    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")

    values = _as_naive_datetime_series(
        prediction_time,
        context="prediction_time",
    )
    result = pd.Series(
        pd.NA,
        index=prediction_time.index,
        dtype="string",
        name="evaluation_partition",
    )

    schedule = config.evaluation
    result.loc[values.lt(pd.Timestamp(schedule.warmup.start))] = (
        "context_only"
    )
    result.loc[
        _window_mask(
            values,
            start=schedule.warmup.start,
            end=schedule.warmup.end,
        )
    ] = "warmup"

    first_fold = schedule.rolling_folds[0]
    result.loc[
        _window_mask(
            values,
            start=first_fold.fit.start,
            end=first_fold.fit.end,
        )
    ] = "development_fit"

    for fold in schedule.rolling_folds:
        result.loc[
            _window_mask(
                values,
                start=fold.validation.start,
                end=fold.validation.end,
            )
        ] = f"{fold.name}_validation"

    result.loc[
        _window_mask(
            values,
            start=schedule.calibration.start,
            end=schedule.calibration.end,
        )
    ] = "calibration"
    result.loc[
        _window_mask(
            values,
            start=schedule.policy_selection.start,
            end=schedule.policy_selection.end,
        )
    ] = "policy_selection"
    result.loc[
        _window_mask(
            values,
            start=schedule.final_test.start,
            end=schedule.final_test.end,
        )
    ] = "final_test"

    if result.isna().any():
        unmapped = values.loc[result.isna()]
        raise ValueError(
            "prediction_time contains values outside the frozen evaluation "
            f"schedule: minimum={unmapped.min()}, maximum={unmapped.max()}"
        )
    if not set(result.unique()).issubset(V2_EVALUATION_PARTITIONS):
        raise ValueError("Unknown Version 2 evaluation partition produced")
    return result.astype("string")


def _parse_label_available_at(
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    required = {"appointment_id", "status_updated_at"}
    missing = sorted(required - set(appointments.columns))
    if missing:
        raise ValueError(
            "appointments is missing label-maturity columns: "
            + ", ".join(missing)
        )

    timing = appointments.loc[
        :,
        ["appointment_id", "status_updated_at"],
    ].copy(deep=True)
    if timing["appointment_id"].isna().any():
        raise ValueError(
            "appointments.appointment_id must not contain missing values"
        )
    timing["appointment_id"] = pd.to_numeric(
        timing["appointment_id"],
        errors="raise",
    ).astype("int64")
    if not timing["appointment_id"].is_unique:
        raise ValueError("appointments.appointment_id must be unique")
    timing["label_available_at"] = _as_naive_datetime_series(
        timing.pop("status_updated_at"),
        context="appointments.status_updated_at",
    )
    return timing


def _validate_feature_dataset(dataset: pd.DataFrame) -> None:
    expected = list(V2_FEATURE_DATASET_COLUMNS)
    if list(dataset.columns) != expected:
        raise ValueError(
            "Version 2 feature dataset columns must match the frozen order"
        )
    if dataset.empty:
        raise ValueError("Version 2 feature dataset must not be empty")
    if dataset["appointment_id"].duplicated().any():
        raise ValueError("Version 2 appointment_id must be unique")
    if dataset.loc[:, expected].isna().any().any():
        raise ValueError("Version 2 feature dataset must not contain missing values")
    if not dataset.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
    ).index.equals(dataset.index):
        raise ValueError(
            "Version 2 feature dataset must be sorted chronologically"
        )
    if not set(dataset["evaluation_partition"].unique()).issubset(
        V2_EVALUATION_PARTITIONS
    ):
        raise ValueError("Version 2 feature dataset has invalid partitions")

    leaked = set(V2_MODEL_FEATURE_COLUMNS) & V2_PROHIBITED_MODEL_COLUMNS
    if leaked:
        raise ValueError(
            "Frozen model-feature allowlist contains prohibited columns: "
            + ", ".join(sorted(leaked))
        )


def build_v2_feature_dataset(
    tables: V2RawTables,
    *,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Build the target-free Version 2 analytical feature dataset."""

    if not isinstance(tables, V2RawTables):
        raise TypeError("tables must be a V2RawTables instance")
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")

    current = build_current_appointment_features(
        tables.appointments,
        tables.patients,
        tables.dentists,
    )
    patient_history = build_patient_history_features(tables.appointments)
    aggregate_history = build_aggregate_history_features(
        tables.appointments
    )

    keys = ["appointment_id", "patient_id", "prediction_time"]
    combined = current.merge(
        patient_history,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    combined = combined.merge(
        aggregate_history,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    history_columns = [
        column
        for column in V2_MODEL_FEATURE_COLUMNS
        if column not in current.columns
    ]
    if combined.loc[:, history_columns].isna().any().any():
        raise ValueError(
            "Patient and aggregate history keys did not align with current rows"
        )

    combined["evaluation_partition"] = assign_v2_evaluation_partitions(
        combined["prediction_time"],
        config,
    )

    timing = _parse_label_available_at(tables.appointments)
    combined = combined.merge(
        timing,
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if combined["label_available_at"].isna().any():
        raise ValueError(
            "Eligible appointments are missing label-availability timestamps"
        )

    result = combined.loc[
        :,
        list(V2_FEATURE_DATASET_COLUMNS),
    ].copy()
    result["appointment_id"] = result["appointment_id"].astype("int64")
    result["patient_id"] = result["patient_id"].astype("int64")
    result["dentist_id"] = result["dentist_id"].astype("int64")
    result["prediction_time"] = result["prediction_time"].astype(
        "datetime64[ns]"
    )
    result["evaluation_partition"] = result[
        "evaluation_partition"
    ].astype("string")
    result["label_available_at"] = result[
        "label_available_at"
    ].astype("datetime64[ns]")

    result = result.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )
    _validate_feature_dataset(result)
    return result


def build_verified_v2_feature_dataset(
    *,
    raw_dir: Path = DEFAULT_V2_RAW_DIR,
    manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Load verified inputs and build the target-free Version 2 dataset."""

    tables = load_verified_v2_raw_tables(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    config = load_benchmark_config(config_path)
    return build_v2_feature_dataset(tables, config=config)


def select_v2_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy of the exact 32-column predictor allowlist."""

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    missing = sorted(set(V2_MODEL_FEATURE_COLUMNS) - set(dataset.columns))
    if missing:
        raise ValueError(
            "dataset is missing approved Version 2 model features: "
            + ", ".join(missing)
        )
    selected = dataset.loc[:, list(V2_MODEL_FEATURE_COLUMNS)].copy(deep=True)
    if set(selected.columns) & V2_PROHIBITED_MODEL_COLUMNS:
        raise ValueError("Prohibited columns entered Version 2 model features")
    return selected


def label_maturity_mask(
    dataset: pd.DataFrame,
    *,
    model_fit_time: pd.Timestamp,
    allowed_partitions: Iterable[str],
) -> pd.Series:
    """Return rows with strictly mature labels at a model-fit boundary."""

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    required = {"evaluation_partition", "label_available_at"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(
            "dataset is missing label-maturity columns: "
            + ", ".join(missing)
        )

    partitions = tuple(allowed_partitions)
    unknown = sorted(set(partitions) - set(V2_EVALUATION_PARTITIONS))
    if unknown:
        raise ValueError(
            "Unknown allowed Version 2 partitions: "
            + ", ".join(unknown)
        )

    fit_time = pd.Timestamp(model_fit_time)
    if fit_time.tz is not None:
        raise ValueError("model_fit_time must be timezone-naive")

    label_available_at = _as_naive_datetime_series(
        dataset["label_available_at"],
        context="dataset.label_available_at",
    )
    return (
        dataset["evaluation_partition"].isin(partitions)
        & label_available_at.lt(fit_time)
    ).astype(bool)


__all__ = (
    "DEFAULT_V2_MANIFEST_PATH",
    "DEFAULT_V2_RAW_DIR",
    "V2RawTables",
    "assign_v2_evaluation_partitions",
    "build_v2_feature_dataset",
    "build_verified_v2_feature_dataset",
    "calculate_sha256",
    "label_maturity_mask",
    "load_verified_v2_raw_tables",
    "select_v2_model_features",
    "validate_frozen_v2_inputs",
)
