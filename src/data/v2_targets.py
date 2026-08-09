"""Protected Version 2 target access and label-maturity contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.build_v2_dataset import (
    DEFAULT_V2_MANIFEST_PATH,
    DEFAULT_V2_RAW_DIR,
    V2RawTables,
    build_v2_feature_dataset,
    calculate_sha256,
    label_maturity_mask,
    load_verified_v2_raw_tables,
)
from src.features.schema import (
    HISTORY_STATUSES,
    V2_EVALUATION_PARTITIONS,
)
from src.synthetic.config import DEFAULT_CONFIG_PATH, load_benchmark_config


FINAL_TEST_PARTITION = "final_test"
TARGET_NAME = "target"
NO_SHOW_STATUS = "no_show"
V2_TARGET_TABLE_COLUMNS = (
    "appointment_id",
    "evaluation_partition",
    "label_available_at",
    TARGET_NAME,
)
FINAL_TEST_PROBABILITY_COLUMNS = (
    "appointment_id",
    "no_show_probability",
)


@dataclass(frozen=True, slots=True)
class FinalTestProbabilitySeal:
    """Identity of a validated target-free final-test probability artifact."""

    path: Path
    sha256: str
    row_count: int


@dataclass(frozen=True, slots=True)
class FinalTestTargetAccess:
    """Protected final-test target result and the prior probability seal."""

    target_table: pd.DataFrame
    probability_seal: FinalTestProbabilitySeal


def _validate_feature_dataset_for_targets(
    feature_dataset: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(feature_dataset, pd.DataFrame):
        raise TypeError("feature_dataset must be a pandas DataFrame")
    required = {
        "appointment_id",
        "prediction_time",
        "evaluation_partition",
        "label_available_at",
    }
    missing = sorted(required - set(feature_dataset.columns))
    if missing:
        raise ValueError(
            "feature_dataset is missing protected target columns: "
            + ", ".join(missing)
        )
    if feature_dataset.empty:
        raise ValueError("feature_dataset must not be empty")

    keys = feature_dataset.loc[:, list(required)].copy(deep=True)
    if keys.isna().any().any():
        raise ValueError(
            "feature_dataset protected target columns must not contain missing values"
        )
    keys["appointment_id"] = pd.to_numeric(
        keys["appointment_id"],
        errors="raise",
    ).astype("int64")
    if not keys["appointment_id"].is_unique:
        raise ValueError("feature_dataset appointment_id must be unique")
    keys["prediction_time"] = pd.to_datetime(
        keys["prediction_time"],
        errors="raise",
        format="mixed",
    ).astype("datetime64[ns]")
    keys["label_available_at"] = pd.to_datetime(
        keys["label_available_at"],
        errors="raise",
        format="mixed",
    ).astype("datetime64[ns]")
    keys["evaluation_partition"] = keys[
        "evaluation_partition"
    ].astype("string")
    observed = set(keys["evaluation_partition"].unique())
    unknown = sorted(observed - set(V2_EVALUATION_PARTITIONS))
    if unknown:
        raise ValueError(
            "feature_dataset contains unknown evaluation partitions: "
            + ", ".join(unknown)
        )
    return keys


def _prepare_status_source(appointments: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(appointments, pd.DataFrame):
        raise TypeError("appointments must be a pandas DataFrame")
    required = {"appointment_id", "status"}
    missing = sorted(required - set(appointments.columns))
    if missing:
        raise ValueError(
            "appointments is missing target source columns: "
            + ", ".join(missing)
        )
    source = appointments.loc[:, ["appointment_id", "status"]].copy(deep=True)
    if source.isna().any().any():
        raise ValueError(
            "appointments target source columns must not contain missing values"
        )
    source["appointment_id"] = pd.to_numeric(
        source["appointment_id"],
        errors="raise",
    ).astype("int64")
    if not source["appointment_id"].is_unique:
        raise ValueError("appointments.appointment_id must be unique")
    source["status"] = source["status"].astype("string")
    unknown = sorted(set(source["status"].unique()) - set(HISTORY_STATUSES))
    if unknown:
        raise ValueError(
            "appointments contains invalid final statuses: "
            + ", ".join(unknown)
        )
    return source


def _normalise_partitions(
    allowed_partitions: Iterable[str],
) -> tuple[str, ...]:
    if isinstance(allowed_partitions, (str, bytes)):
        raise TypeError("allowed_partitions must be an iterable of partition names")
    partitions = tuple(allowed_partitions)
    if not partitions:
        raise ValueError("allowed_partitions must not be empty")
    if len(set(partitions)) != len(partitions):
        raise ValueError("allowed_partitions must not contain duplicates")
    unknown = sorted(set(partitions) - set(V2_EVALUATION_PARTITIONS))
    if unknown:
        raise ValueError(
            "Unknown allowed Version 2 partitions: " + ", ".join(unknown)
        )
    return partitions


def _attach_target(
    rows: pd.DataFrame,
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    source = _prepare_status_source(appointments)
    joined = rows.merge(
        source,
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if joined["status"].isna().any():
        raise ValueError("Selected appointments are missing target source statuses")
    joined[TARGET_NAME] = joined["status"].eq(NO_SHOW_STATUS).astype("int8")
    result = joined.loc[:, list(V2_TARGET_TABLE_COLUMNS)].copy(deep=True)
    result["appointment_id"] = result["appointment_id"].astype("int64")
    result["evaluation_partition"] = result[
        "evaluation_partition"
    ].astype("string")
    result["label_available_at"] = result[
        "label_available_at"
    ].astype("datetime64[ns]")
    result[TARGET_NAME] = result[TARGET_NAME].astype("int8")
    if result[TARGET_NAME].isna().any():
        raise RuntimeError("Target construction produced missing values")
    return result


def build_mature_v2_target_table(
    feature_dataset: pd.DataFrame,
    appointments: pd.DataFrame,
    *,
    model_fit_time: pd.Timestamp,
    allowed_partitions: Iterable[str],
) -> pd.DataFrame:
    """Return only strictly mature non-test targets for model development."""

    keys = _validate_feature_dataset_for_targets(feature_dataset)
    partitions = _normalise_partitions(allowed_partitions)
    if FINAL_TEST_PARTITION in partitions:
        raise PermissionError(
            "final_test targets require the protected final-test accessor"
        )

    maturity = label_maturity_mask(
        feature_dataset,
        model_fit_time=model_fit_time,
        allowed_partitions=partitions,
    )
    selected = keys.loc[
        maturity,
        ["appointment_id", "evaluation_partition", "label_available_at"],
    ].copy(deep=True)
    if selected.empty:
        raise ValueError("No strictly mature target rows were selected")
    if selected["evaluation_partition"].eq(FINAL_TEST_PARTITION).any():
        raise RuntimeError("Mature training target selection exposed final_test")
    return _attach_target(selected, appointments)


def load_verified_v2_mature_targets(
    *,
    model_fit_time: pd.Timestamp,
    allowed_partitions: Iterable[str],
    raw_dir: Path = DEFAULT_V2_RAW_DIR,
    manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Build strictly mature targets from the verified frozen benchmark."""

    tables = load_verified_v2_raw_tables(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    feature_dataset = build_v2_feature_dataset(
        tables,
        config=load_benchmark_config(config_path),
    )
    return build_mature_v2_target_table(
        feature_dataset,
        tables.appointments,
        model_fit_time=model_fit_time,
        allowed_partitions=allowed_partitions,
    )


def validate_final_test_probability_artifact(
    probability_path: Path,
    feature_dataset: pd.DataFrame,
) -> FinalTestProbabilitySeal:
    """Validate a written final-test probability vector without target access."""

    path = Path(probability_path)
    if not path.is_file():
        raise ValueError(f"Final-test probability artifact is missing: {path}")

    keys = _validate_feature_dataset_for_targets(feature_dataset)
    final_rows = keys.loc[
        keys["evaluation_partition"].eq(FINAL_TEST_PARTITION),
        ["appointment_id"],
    ].copy(deep=True)
    if final_rows.empty:
        raise ValueError("feature_dataset has no final_test rows")

    try:
        probability = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            f"Could not read final-test probability artifact: {path}"
        ) from exc
    if tuple(probability.columns) != FINAL_TEST_PROBABILITY_COLUMNS:
        raise ValueError(
            "Final-test probability artifact columns must be exactly "
            "appointment_id and no_show_probability"
        )
    if probability.empty:
        raise ValueError("Final-test probability artifact must not be empty")
    if probability.isna().any().any():
        raise ValueError(
            "Final-test probability artifact must not contain missing values"
        )

    probability["appointment_id"] = pd.to_numeric(
        probability["appointment_id"],
        errors="raise",
    ).astype("int64")
    if not probability["appointment_id"].is_unique:
        raise ValueError(
            "Final-test probability appointment_id must be unique"
        )
    values = pd.to_numeric(
        probability["no_show_probability"],
        errors="raise",
    ).astype("float64")
    array = values.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(
            "Final-test probabilities must contain only finite values"
        )
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("Final-test probabilities must be within [0, 1]")

    expected_ids = final_rows["appointment_id"].reset_index(drop=True)
    actual_ids = probability["appointment_id"].reset_index(drop=True)
    if not actual_ids.equals(expected_ids):
        raise ValueError(
            "Final-test probability appointment_id values and order must "
            "match the protected feature dataset"
        )

    return FinalTestProbabilitySeal(
        path=path.resolve(),
        sha256=calculate_sha256(path),
        row_count=int(len(probability)),
    )


def _build_final_test_access(
    feature_dataset: pd.DataFrame,
    appointments: pd.DataFrame,
    *,
    probability_seal: FinalTestProbabilitySeal,
) -> FinalTestTargetAccess:
    keys = _validate_feature_dataset_for_targets(feature_dataset)
    selected = keys.loc[
        keys["evaluation_partition"].eq(FINAL_TEST_PARTITION),
        ["appointment_id", "evaluation_partition", "label_available_at"],
    ].copy(deep=True)
    if selected.empty:
        raise ValueError("feature_dataset has no final_test rows")
    target_table = _attach_target(selected, appointments)
    if len(target_table) != probability_seal.row_count:
        raise RuntimeError(
            "Final-test target rows do not match the sealed probability vector"
        )
    return FinalTestTargetAccess(
        target_table=target_table,
        probability_seal=probability_seal,
    )


def access_v2_final_test_targets(
    feature_dataset: pd.DataFrame,
    appointments: pd.DataFrame,
    *,
    probability_path: Path,
    allow_test: bool = False,
) -> FinalTestTargetAccess:
    """Expose final-test targets only after an exact probability artifact exists."""

    if type(allow_test) is not bool:
        raise TypeError("allow_test must be an exact bool")
    if not allow_test:
        raise PermissionError(
            "Final-test target access requires explicit allow_test=True"
        )

    probability_seal = validate_final_test_probability_artifact(
        probability_path,
        feature_dataset,
    )
    return _build_final_test_access(
        feature_dataset,
        appointments,
        probability_seal=probability_seal,
    )


def load_verified_v2_final_test_targets(
    *,
    probability_path: Path,
    allow_test: bool = False,
    processed_dir: Path | None = None,
    raw_dir: Path = DEFAULT_V2_RAW_DIR,
    manifest_path: Path = DEFAULT_V2_MANIFEST_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> FinalTestTargetAccess:
    """Validate probabilities first, then load and expose frozen test targets."""

    if type(allow_test) is not bool:
        raise TypeError("allow_test must be an exact bool")
    if not allow_test:
        raise PermissionError(
            "Final-test target access requires explicit allow_test=True"
        )

    from src.data.export_v2_processed import (
        DEFAULT_V2_PROCESSED_DIR,
        load_frozen_v2_processed_feature_dataset,
    )

    destination = (
        DEFAULT_V2_PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    )
    feature_dataset = load_frozen_v2_processed_feature_dataset(destination)
    probability_seal = validate_final_test_probability_artifact(
        probability_path,
        feature_dataset,
    )

    tables: V2RawTables = load_verified_v2_raw_tables(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    return _build_final_test_access(
        feature_dataset,
        tables.appointments,
        probability_seal=probability_seal,
    )


__all__ = (
    "FINAL_TEST_PARTITION",
    "FINAL_TEST_PROBABILITY_COLUMNS",
    "FinalTestProbabilitySeal",
    "FinalTestTargetAccess",
    "TARGET_NAME",
    "V2_TARGET_TABLE_COLUMNS",
    "access_v2_final_test_targets",
    "build_mature_v2_target_table",
    "load_verified_v2_final_test_targets",
    "load_verified_v2_mature_targets",
    "validate_final_test_probability_artifact",
)
