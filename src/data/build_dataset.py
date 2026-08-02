"""Build the canonical, leakage-controlled analytical dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


AUDIT_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "prediction_time",
)
TARGET_PARTITION_COLUMNS = ("target", "split")
FEATURE_COLUMNS = (
    "planned_duration_min",
    "visit_type",
    "booking_channel",
    "booking_lead_time_hours",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
MATURITY_COLUMNS = (
    "development_fit_eligible",
    "pretest_fit_eligible",
)
CANONICAL_COLUMNS = (
    *AUDIT_COLUMNS,
    *TARGET_PARTITION_COLUMNS,
    *FEATURE_COLUMNS,
    *MATURITY_COLUMNS,
)

PROHIBITED_COLUMNS = frozenset(
    {
        "status",
        "status_updated_at",
        "booked_at",
        "scheduled_start_at",
        "birth_year",
        "registered_at",
        "start_date",
        "reminder_sent",
        "check_in_at",
        "chair_start_at",
        "chair_end_at",
        "checkout_at",
        "status_change_reason",
        "rescheduled_from_appointment_id",
        "sex",
        "city_area",
        "insurance_type",
        "referral_source",
        "preferred_contact_channel",
        "patient_status",
        "dentist_role",
        "engagement_type",
        "end_date",
        "scheduled_hours_weekly",
        "active",
    }
)

EXPECTED_RAW_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "appointments.csv": (
            "4f3736f78cda615d1401d3f639b5e29e47781a1ae1c820c1e6f248eae57a00df"
        ),
        "patients.csv": (
            "e416843a80568a91455e5cff872bbca5b49be16f109022d56c687cdf2683cc69"
        ),
        "dentists.csv": (
            "bf83d1848236e8f5fc8ee5ef3bb21fec2690f85c3c2f259840c16c271a00ab47"
        ),
    }
)

PREDICTION_HORIZON_HOURS = 24
VALIDATION_START = pd.Timestamp("2025-03-01 00:00:00")
TEST_START = pd.Timestamp("2025-08-01 00:00:00")
DEVELOPMENT_FIT_TIME = VALIDATION_START
PRETEST_FIT_TIME = TEST_START

EXPECTED_RAW_ROW_COUNTS: Mapping[str, int] = MappingProxyType(
    {"appointments": 8_000, "patients": 2_000, "dentists": 7}
)
EXPECTED_TOTAL_ROWS = 6_786
EXPECTED_POSITIVES = 820
EXPECTED_NEGATIVES = 5_966
EXPECTED_SPLIT_ROWS: Mapping[str, int] = MappingProxyType(
    {"train": 3_682, "validation": 1_541, "test": 1_563}
)
EXPECTED_SPLIT_POSITIVES: Mapping[str, int] = MappingProxyType(
    {"train": 434, "validation": 192, "test": 194}
)
EXPECTED_MATURITY_ROWS: Mapping[str, int] = MappingProxyType(
    {"development_fit_eligible": 3_670, "pretest_fit_eligible": 5_223}
)
EXPECTED_MATURITY_POSITIVES: Mapping[str, int] = MappingProxyType(
    {"development_fit_eligible": 432, "pretest_fit_eligible": 626}
)

APPOINTMENT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
PATIENT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DENTIST_DATE_FORMAT = "%Y-%m-%d"

APPOINTMENT_SOURCE_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "booked_at",
    "scheduled_start_at",
    "planned_duration_min",
    "visit_type",
    "booking_channel",
    "status",
    "status_updated_at",
)
PATIENT_SOURCE_COLUMNS = ("patient_id", "birth_year", "registered_at")
DENTIST_SOURCE_COLUMNS = ("dentist_id", "start_date")
ALLOWED_ELIGIBLE_STATUSES = frozenset(
    {"completed", "no_show", "cancelled", "rescheduled"}
)
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})

EXPECTED_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "appointment_id": "int64",
        "patient_id": "int64",
        "dentist_id": "int64",
        "prediction_time": "datetime64[ns]",
        "target": "int8",
        "split": "string",
        "planned_duration_min": "int16",
        "visit_type": "string",
        "booking_channel": "string",
        "booking_lead_time_hours": "float64",
        "scheduled_weekday": "int8",
        "scheduled_hour": "int8",
        "scheduled_month": "int8",
        "approximate_age_at_prediction": "int16",
        "patient_registration_tenure_days": "int32",
        "dentist_tenure_days": "int32",
        "development_fit_eligible": "bool",
        "pretest_fit_eligible": "bool",
    }
)


@dataclass(frozen=True)
class RawTables:
    """The three approved raw source tables."""

    appointments: pd.DataFrame
    patients: pd.DataFrame
    dentists: pd.DataFrame


def calculate_sha256(path: Path) -> str:
    """Return a lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_paths(
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[Path, Path, Path]:
    """Resolve output paths and reject destinations that threaten raw inputs."""

    resolved_raw_dir = Path(raw_dir).resolve(strict=False)
    resolved_output = Path(output_path).resolve(strict=False)
    resolved_manifest = Path(manifest_path).resolve(strict=False)
    if resolved_output == resolved_manifest:
        raise ValueError(
            "Parquet output and manifest paths resolve to the same destination: "
            f"{resolved_output}"
        )
    for label, destination in (
        ("Parquet output", resolved_output),
        ("Manifest", resolved_manifest),
    ):
        if destination == resolved_raw_dir or destination.is_relative_to(
            resolved_raw_dir
        ):
            raise ValueError(
                f"{label} destination must not be inside the raw directory "
                f"{resolved_raw_dir}: {destination}"
            )
    return resolved_raw_dir, resolved_output, resolved_manifest


def normalize_input_hashes(input_hashes: Mapping[str, str]) -> dict[str, str]:
    """Validate exact approved provenance hashes and return lowercase values."""

    approved_filenames = frozenset(EXPECTED_RAW_HASHES)
    supplied_filenames = frozenset(input_hashes)
    if supplied_filenames != approved_filenames:
        missing = sorted(approved_filenames - supplied_filenames)
        extra = sorted(supplied_filenames - approved_filenames)
        raise ValueError(
            "Raw input hashes must contain exactly the three approved filenames; "
            f"missing={missing}, extra={extra}"
        )

    normalized: dict[str, str] = {}
    hexadecimal_characters = frozenset("0123456789abcdefABCDEF")
    for filename in EXPECTED_RAW_HASHES:
        digest = input_hashes[filename]
        if not isinstance(digest, str):
            raise ValueError(f"SHA-256 digest for {filename} must be a string")
        if len(digest) != 64:
            raise ValueError(
                f"SHA-256 digest for {filename} must contain exactly 64 hexadecimal "
                f"characters; got {len(digest)}"
            )
        if any(character not in hexadecimal_characters for character in digest):
            raise ValueError(
                f"SHA-256 digest for {filename} must contain only hexadecimal "
                "characters"
            )
        normalized_digest = digest.lower()
        approved_digest = EXPECTED_RAW_HASHES[filename]
        if normalized_digest != approved_digest:
            raise ValueError(
                f"SHA-256 digest for {filename} does not match the approved raw hash: "
                f"expected {approved_digest}, got {normalized_digest}"
            )
        normalized[filename] = normalized_digest
    return normalized


def validate_raw_hashes(
    raw_dir: Path,
    *,
    expected_hashes: Mapping[str, str] = EXPECTED_RAW_HASHES,
) -> dict[str, str]:
    """Validate all approved raw inputs and return their actual hashes."""

    raw_dir = Path(raw_dir)
    approved_filenames = frozenset(EXPECTED_RAW_HASHES)
    supplied_filenames = frozenset(expected_hashes)
    if supplied_filenames != approved_filenames:
        missing = sorted(approved_filenames - supplied_filenames)
        extra = sorted(supplied_filenames - approved_filenames)
        raise ValueError(
            "Expected hashes must cover exactly the three approved raw files; "
            f"missing={missing}, extra={extra}"
        )
    actual_hashes: dict[str, str] = {}
    for filename, expected_hash in expected_hashes.items():
        path = raw_dir / filename
        if not path.is_file():
            raise ValueError(f"Required raw file is missing: {path}")
        actual_hash = calculate_sha256(path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected "
                f"{expected_hash.lower()}, got {actual_hash.lower()}"
            )
        actual_hashes[filename] = actual_hash.lower()
    return actual_hashes


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(missing)}"
        )


def _parse_datetime_column(
    frame: pd.DataFrame,
    column: str,
    date_format: str,
    *,
    table_name: str,
) -> None:
    try:
        parsed = pd.to_datetime(frame[column], format=date_format, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{table_name}.{column} does not match required format {date_format}"
        ) from exc
    if parsed.dt.tz is not None:
        raise ValueError(f"{table_name}.{column} must be timezone-naive")
    frame[column] = parsed.astype("datetime64[ns]")


def _read_raw_table(
    path: Path,
    *,
    required_columns: tuple[str, ...],
    dtypes: Mapping[str, str],
    datetime_formats: Mapping[str, str],
    table_name: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Required raw file is missing: {path}")
    try:
        headers = pd.read_csv(path, nrows=0).columns
    except Exception as exc:
        raise ValueError(f"Could not read header from {path}") from exc
    missing = [column for column in required_columns if column not in headers]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(missing)}"
        )
    try:
        frame = pd.read_csv(path, usecols=list(required_columns), dtype=dict(dtypes))
    except Exception as exc:
        raise ValueError(f"Could not load {table_name} from {path}") from exc
    frame = frame.loc[:, required_columns].copy()
    for column, date_format in datetime_formats.items():
        _parse_datetime_column(
            frame,
            column,
            date_format,
            table_name=table_name,
        )
    return frame


def load_raw_data(raw_dir: Path) -> RawTables:
    """Load the approved raw files with explicit types and timestamp formats."""

    raw_dir = Path(raw_dir)
    appointments = _read_raw_table(
        raw_dir / "appointments.csv",
        required_columns=APPOINTMENT_SOURCE_COLUMNS,
        dtypes={
            "appointment_id": "int64",
            "patient_id": "int64",
            "dentist_id": "int64",
            "planned_duration_min": "int64",
            "visit_type": "string",
            "booking_channel": "string",
            "status": "string",
        },
        datetime_formats={
            "booked_at": APPOINTMENT_DATETIME_FORMAT,
            "scheduled_start_at": APPOINTMENT_DATETIME_FORMAT,
            "status_updated_at": APPOINTMENT_DATETIME_FORMAT,
        },
        table_name="appointments",
    )
    patients = _read_raw_table(
        raw_dir / "patients.csv",
        required_columns=PATIENT_SOURCE_COLUMNS,
        dtypes={"patient_id": "int64", "birth_year": "int64"},
        datetime_formats={"registered_at": PATIENT_DATETIME_FORMAT},
        table_name="patients",
    )
    dentists = _read_raw_table(
        raw_dir / "dentists.csv",
        required_columns=DENTIST_SOURCE_COLUMNS,
        dtypes={"dentist_id": "int64"},
        datetime_formats={"start_date": DENTIST_DATE_FORMAT},
        table_name="dentists",
    )
    tables = RawTables(
        appointments=appointments,
        patients=patients,
        dentists=dentists,
    )
    validate_required_columns(tables)
    return tables


def _validate_primary_key(
    frame: pd.DataFrame,
    key: str,
    *,
    table_name: str,
) -> None:
    if frame[key].isna().any():
        raise ValueError(f"{table_name}.{key} must not contain missing values")
    if not frame[key].is_unique:
        raise ValueError(f"{table_name}.{key} must be unique")


def validate_required_columns(tables: RawTables) -> None:
    """Validate required schemas, row counts, keys, and source completeness."""

    table_specs = (
        (
            "appointments",
            tables.appointments,
            APPOINTMENT_SOURCE_COLUMNS,
            "appointment_id",
        ),
        ("patients", tables.patients, PATIENT_SOURCE_COLUMNS, "patient_id"),
        ("dentists", tables.dentists, DENTIST_SOURCE_COLUMNS, "dentist_id"),
    )
    for table_name, frame, required_columns, primary_key in table_specs:
        _require_columns(frame, required_columns, table_name=table_name)
        expected_rows = EXPECTED_RAW_ROW_COUNTS[table_name]
        if len(frame) != expected_rows:
            raise ValueError(
                f"{table_name} row count must be {expected_rows:,}; got {len(frame):,}"
            )
        _validate_primary_key(frame, primary_key, table_name=table_name)
        missing_counts = frame.loc[:, required_columns].isna().sum()
        missing_columns = missing_counts[missing_counts.gt(0)].to_dict()
        if missing_columns:
            raise ValueError(
                f"{table_name} required columns contain missing values: "
                f"{missing_columns}"
            )


def reconstruct_eligible_cohort(appointments: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct appointments active at the approved prediction time."""

    required = (
        "appointment_id",
        "booked_at",
        "scheduled_start_at",
        "status",
        "status_updated_at",
    )
    _require_columns(appointments, required, table_name="appointments")
    working = appointments.copy(deep=True)
    _validate_primary_key(working, "appointment_id", table_name="appointments")
    prediction_time = working["scheduled_start_at"] - pd.Timedelta(
        hours=PREDICTION_HORIZON_HOURS
    )
    already_booked = working["booked_at"].le(prediction_time)
    inactive_before_prediction = working["status"].isin(
        ("cancelled", "rescheduled")
    ) & working["status_updated_at"].le(prediction_time)
    eligible = already_booked & ~inactive_before_prediction
    cohort = working.loc[eligible].copy()
    cohort["prediction_time"] = prediction_time.loc[eligible].astype(
        "datetime64[ns]"
    )
    if not cohort["appointment_id"].is_unique:
        raise ValueError("appointment_id must remain unique after cohort construction")
    return cohort


def join_reference_data(
    cohort: pd.DataFrame,
    patients: pd.DataFrame,
    dentists: pd.DataFrame,
) -> pd.DataFrame:
    """Join only the reference columns needed for approved derivations."""

    _require_columns(cohort, ("patient_id", "dentist_id"), table_name="cohort")
    _require_columns(patients, PATIENT_SOURCE_COLUMNS, table_name="patients")
    _require_columns(dentists, DENTIST_SOURCE_COLUMNS, table_name="dentists")
    _validate_primary_key(patients, "patient_id", table_name="patients")
    _validate_primary_key(dentists, "dentist_id", table_name="dentists")

    patient_lookup = patients.loc[:, PATIENT_SOURCE_COLUMNS].copy()
    dentist_lookup = dentists.loc[:, DENTIST_SOURCE_COLUMNS].copy()
    if patient_lookup[["birth_year", "registered_at"]].isna().any().any():
        raise ValueError("Patient derivation source fields must not be missing")
    if dentist_lookup[["start_date"]].isna().any().any():
        raise ValueError("Dentist derivation source fields must not be missing")

    joined = cohort.merge(
        patient_lookup,
        on="patient_id",
        how="left",
        validate="many_to_one",
        indicator="_patient_join",
        sort=False,
    )
    if not joined["_patient_join"].eq("both").all():
        missing_ids = joined.loc[
            joined["_patient_join"].ne("both"), "patient_id"
        ].unique()
        raise ValueError(f"Unmatched patient_id values: {missing_ids.tolist()}")
    joined = joined.drop(columns="_patient_join")

    joined = joined.merge(
        dentist_lookup,
        on="dentist_id",
        how="left",
        validate="many_to_one",
        indicator="_dentist_join",
        sort=False,
    )
    if not joined["_dentist_join"].eq("both").all():
        missing_ids = joined.loc[
            joined["_dentist_join"].ne("both"), "dentist_id"
        ].unique()
        raise ValueError(f"Unmatched dentist_id values: {missing_ids.tolist()}")
    joined = joined.drop(columns="_dentist_join")

    required_joined = ("birth_year", "registered_at", "start_date")
    if joined.loc[:, required_joined].isna().any().any():
        raise ValueError("Required joined derivation values must not be missing")
    return joined


def construct_target(joined: pd.DataFrame) -> pd.Series:
    """Construct the approved binary target for eligible appointments."""

    _require_columns(joined, ("status",), table_name="eligible appointments")
    if joined["status"].isna().any():
        raise ValueError("Eligible appointment status must not be missing")
    observed = frozenset(joined["status"].astype("string").unique().tolist())
    unexpected = observed - ALLOWED_ELIGIBLE_STATUSES
    if unexpected:
        raise ValueError(
            f"Unexpected eligible status values: {sorted(unexpected)}"
        )
    target = joined["status"].eq("no_show").astype("int8")
    target.name = "target"
    return target


def _validate_int_range(series: pd.Series, dtype: str, *, column: str) -> None:
    limits = np.iinfo(np.dtype(dtype))
    if series.lt(limits.min).any() or series.gt(limits.max).any():
        raise ValueError(f"{column} cannot be represented safely as {dtype}")


def derive_approved_features(joined: pd.DataFrame) -> pd.DataFrame:
    """Derive and return only the approved model-feature columns."""

    required = (
        "planned_duration_min",
        "visit_type",
        "booking_channel",
        "booked_at",
        "scheduled_start_at",
        "prediction_time",
        "birth_year",
        "registered_at",
        "start_date",
    )
    _require_columns(joined, required, table_name="joined cohort")
    if joined.loc[:, required].isna().any().any():
        raise ValueError("Approved feature source fields must not be missing")

    lead_hours = (
        joined["scheduled_start_at"].sub(joined["booked_at"]).dt.total_seconds()
        / 3_600
    ).astype("float64")
    if not np.isfinite(lead_hours).all() or lead_hours.lt(0).any():
        raise ValueError("booking_lead_time_hours must be finite and non-negative")
    if lead_hours.lt(PREDICTION_HORIZON_HOURS).any():
        raise ValueError(
            "Eligible appointments must have at least 24 hours of booking lead time"
        )

    age = joined["prediction_time"].dt.year.sub(joined["birth_year"])
    if age.lt(0).any():
        raise ValueError("approximate_age_at_prediction must not be negative")

    patient_elapsed_seconds = (
        joined["prediction_time"].sub(joined["registered_at"]).dt.total_seconds()
    )
    if patient_elapsed_seconds.lt(0).any():
        raise ValueError("patient registration must not occur after prediction_time")
    patient_tenure_days = np.floor(patient_elapsed_seconds / 86_400)

    dentist_elapsed_seconds = (
        joined["prediction_time"].sub(joined["start_date"]).dt.total_seconds()
    )
    if dentist_elapsed_seconds.lt(0).any():
        raise ValueError("dentist start_date must not occur after prediction_time")
    dentist_tenure_days = np.floor(dentist_elapsed_seconds / 86_400)

    _validate_int_range(
        joined["planned_duration_min"], "int16", column="planned_duration_min"
    )
    _validate_int_range(age, "int16", column="approximate_age_at_prediction")
    _validate_int_range(
        patient_tenure_days,
        "int32",
        column="patient_registration_tenure_days",
    )
    _validate_int_range(
        dentist_tenure_days,
        "int32",
        column="dentist_tenure_days",
    )

    features = pd.DataFrame(index=joined.index)
    features["planned_duration_min"] = joined["planned_duration_min"].astype(
        "int16"
    )
    features["visit_type"] = joined["visit_type"].astype("string")
    features["booking_channel"] = joined["booking_channel"].astype("string")
    features["booking_lead_time_hours"] = lead_hours
    features["scheduled_weekday"] = joined[
        "scheduled_start_at"
    ].dt.dayofweek.astype("int8")
    features["scheduled_hour"] = joined["scheduled_start_at"].dt.hour.astype(
        "int8"
    )
    features["scheduled_month"] = joined["scheduled_start_at"].dt.month.astype(
        "int8"
    )
    features["approximate_age_at_prediction"] = age.astype("int16")
    features["patient_registration_tenure_days"] = patient_tenure_days.astype(
        "int32"
    )
    features["dentist_tenure_days"] = dentist_tenure_days.astype("int32")
    return features.loc[:, FEATURE_COLUMNS].copy()


def assign_temporal_splits(prediction_time: pd.Series) -> pd.Series:
    """Assign the approved half-open chronological partitions."""

    if prediction_time.isna().any():
        raise ValueError("prediction_time must not contain missing values")
    if not isinstance(prediction_time.dtype, pd.DatetimeTZDtype):
        try:
            timezone = prediction_time.dt.tz
        except AttributeError as exc:
            raise ValueError("prediction_time must be datetime typed") from exc
        if timezone is not None:
            raise ValueError("prediction_time must be timezone-naive")
    else:
        raise ValueError("prediction_time must be timezone-naive")

    split = pd.Series(pd.NA, index=prediction_time.index, dtype="string", name="split")
    split.loc[prediction_time.lt(VALIDATION_START)] = "train"
    split.loc[
        prediction_time.ge(VALIDATION_START) & prediction_time.lt(TEST_START)
    ] = "validation"
    split.loc[prediction_time.ge(TEST_START)] = "test"
    if split.isna().any() or not set(split.unique()).issubset(ALLOWED_SPLITS):
        raise ValueError("Every prediction_time must map to exactly one split")
    return split.astype("string")


def label_maturity_mask(
    dataset_with_status_time: pd.DataFrame,
    *,
    model_fit_time: pd.Timestamp,
    allowed_splits: tuple[str, ...],
) -> pd.Series:
    """Return rows whose labels were strictly available at a fit boundary."""

    _require_columns(
        dataset_with_status_time,
        ("split", "status_updated_at"),
        table_name="dataset with outcome timing",
    )
    unknown_splits = set(allowed_splits) - ALLOWED_SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown allowed split values: {sorted(unknown_splits)}")
    if dataset_with_status_time["status_updated_at"].isna().any():
        raise ValueError("status_updated_at must not contain missing values")
    fit_time = pd.Timestamp(model_fit_time)
    if fit_time.tz is not None:
        raise ValueError("model_fit_time must be timezone-naive")
    mask = dataset_with_status_time["split"].isin(allowed_splits) & (
        dataset_with_status_time["status_updated_at"] < fit_time
    )
    return mask.astype(bool)


def _validate_canonical_structure(dataset: pd.DataFrame) -> None:
    if tuple(dataset.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            "Canonical columns or order differ from the approved schema: "
            f"expected {CANONICAL_COLUMNS}, got {tuple(dataset.columns)}"
        )
    leaked = set(dataset.columns) & PROHIBITED_COLUMNS
    if leaked:
        raise ValueError(
            f"Canonical dataset contains prohibited columns: {sorted(leaked)}"
        )
    disallowed_predictors = set(FEATURE_COLUMNS) & (
        set(AUDIT_COLUMNS)
        | set(TARGET_PARTITION_COLUMNS)
        | set(MATURITY_COLUMNS)
        | PROHIBITED_COLUMNS
    )
    if disallowed_predictors:
        raise ValueError(
            "Feature allowlist contains disallowed columns: "
            f"{sorted(disallowed_predictors)}"
        )
    for column, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = dataset[column].dtype
        if expected_dtype == "string":
            matches = isinstance(actual_dtype, pd.StringDtype)
        else:
            matches = str(actual_dtype) == expected_dtype
        if not matches:
            raise ValueError(
                f"{column} dtype must be {expected_dtype}; got {actual_dtype}"
            )


def select_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy of the exact approved predictor allowlist."""

    _validate_canonical_structure(dataset)
    return dataset.loc[:, FEATURE_COLUMNS].copy(deep=True)


def select_development_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return train and validation rows while excluding test unconditionally."""

    _validate_canonical_structure(dataset)
    return dataset.loc[dataset["split"].isin(("train", "validation"))].copy(
        deep=True
    )


def select_test_rows(
    dataset: pd.DataFrame,
    *,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Return test rows only after explicit opt-in."""

    if not allow_test:
        raise PermissionError("Test access requires explicit allow_test=True")
    _validate_canonical_structure(dataset)
    return dataset.loc[dataset["split"].eq("test")].copy(deep=True)


def _require_exact_count(actual: int, expected: int, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected:,}; got {actual:,}")


def validate_output_invariants(dataset: pd.DataFrame) -> None:
    """Validate the complete canonical schema and all approved reconciliations."""

    _validate_canonical_structure(dataset)
    if dataset.isna().any().any():
        missing = dataset.isna().sum()
        raise ValueError(
            f"Canonical dataset contains missing values: "
            f"{missing[missing.gt(0)].to_dict()}"
        )
    if not dataset["appointment_id"].is_unique:
        raise ValueError("appointment_id must be unique in the canonical dataset")
    if set(dataset["target"].unique()) != {0, 1}:
        raise ValueError("target must contain exactly binary values 0 and 1")
    if set(dataset["split"].unique()) != ALLOWED_SPLITS:
        raise ValueError("Every canonical row must belong to one approved split")

    _require_exact_count(len(dataset), EXPECTED_TOTAL_ROWS, label="total rows")
    positives = int(dataset["target"].sum())
    _require_exact_count(positives, EXPECTED_POSITIVES, label="target positives")
    _require_exact_count(
        len(dataset) - positives,
        EXPECTED_NEGATIVES,
        label="target negatives",
    )

    for split_name in ("train", "validation", "test"):
        split_rows = dataset["split"].eq(split_name)
        _require_exact_count(
            int(split_rows.sum()),
            EXPECTED_SPLIT_ROWS[split_name],
            label=f"{split_name} rows",
        )
        _require_exact_count(
            int(dataset.loc[split_rows, "target"].sum()),
            EXPECTED_SPLIT_POSITIVES[split_name],
            label=f"{split_name} positives",
        )

    chronological_checks = {
        "train": dataset["prediction_time"].lt(VALIDATION_START),
        "validation": dataset["prediction_time"].ge(VALIDATION_START)
        & dataset["prediction_time"].lt(TEST_START),
        "test": dataset["prediction_time"].ge(TEST_START),
    }
    for split_name, valid_time in chronological_checks.items():
        rows = dataset["split"].eq(split_name)
        if not valid_time.loc[rows].all():
            raise ValueError(f"{split_name} contains rows outside its time boundary")

    for maturity_column in MATURITY_COLUMNS:
        mature = dataset[maturity_column]
        _require_exact_count(
            int(mature.sum()),
            EXPECTED_MATURITY_ROWS[maturity_column],
            label=f"{maturity_column} rows",
        )
        _require_exact_count(
            int(dataset.loc[mature, "target"].sum()),
            EXPECTED_MATURITY_POSITIVES[maturity_column],
            label=f"{maturity_column} positives",
        )

    if not np.isfinite(dataset["booking_lead_time_hours"]).all():
        raise ValueError("booking_lead_time_hours must be finite")
    range_checks = {
        "planned_duration_min": dataset["planned_duration_min"].gt(0),
        "booking_lead_time_hours": dataset["booking_lead_time_hours"].ge(
            PREDICTION_HORIZON_HOURS
        ),
        "scheduled_weekday": dataset["scheduled_weekday"].between(0, 6),
        "scheduled_hour": dataset["scheduled_hour"].between(0, 23),
        "scheduled_month": dataset["scheduled_month"].between(1, 12),
        "approximate_age_at_prediction": dataset[
            "approximate_age_at_prediction"
        ].ge(0),
        "patient_registration_tenure_days": dataset[
            "patient_registration_tenure_days"
        ].ge(0),
        "dentist_tenure_days": dataset["dentist_tenure_days"].ge(0),
    }
    for feature_name, valid in range_checks.items():
        if not valid.all():
            raise ValueError(f"{feature_name} contains values outside its valid range")

    expected_order = dataset.sort_values(
        ["prediction_time", "appointment_id"], kind="mergesort"
    ).reset_index(drop=True)
    if not dataset.index.equals(pd.RangeIndex(len(dataset))):
        raise ValueError("Canonical dataset index must be a reset RangeIndex")
    if not dataset[["prediction_time", "appointment_id"]].equals(
        expected_order[["prediction_time", "appointment_id"]]
    ):
        raise ValueError(
            "Canonical dataset must be sorted by prediction_time then appointment_id"
        )


def build_analytical_dataset(tables: RawTables) -> pd.DataFrame:
    """Build, sort, and validate the approved canonical analytical dataset."""

    validate_required_columns(tables)
    cohort = reconstruct_eligible_cohort(tables.appointments)
    joined = join_reference_data(cohort, tables.patients, tables.dentists)
    target = construct_target(joined)
    features = derive_approved_features(joined)
    split = assign_temporal_splits(joined["prediction_time"])

    construction = joined.copy(deep=True)
    construction["target"] = target
    construction["split"] = split
    construction["development_fit_eligible"] = label_maturity_mask(
        construction,
        model_fit_time=DEVELOPMENT_FIT_TIME,
        allowed_splits=("train",),
    )
    construction["pretest_fit_eligible"] = label_maturity_mask(
        construction,
        model_fit_time=PRETEST_FIT_TIME,
        allowed_splits=("train", "validation"),
    )

    canonical = pd.concat(
        (
            construction.loc[:, AUDIT_COLUMNS],
            construction.loc[:, TARGET_PARTITION_COLUMNS],
            features,
            construction.loc[:, MATURITY_COLUMNS],
        ),
        axis=1,
    ).loc[:, CANONICAL_COLUMNS]

    canonical["appointment_id"] = canonical["appointment_id"].astype("int64")
    canonical["patient_id"] = canonical["patient_id"].astype("int64")
    canonical["dentist_id"] = canonical["dentist_id"].astype("int64")
    canonical["prediction_time"] = canonical["prediction_time"].astype(
        "datetime64[ns]"
    )
    canonical["target"] = canonical["target"].astype("int8")
    canonical["split"] = canonical["split"].astype("string")
    canonical["development_fit_eligible"] = canonical[
        "development_fit_eligible"
    ].astype(bool)
    canonical["pretest_fit_eligible"] = canonical[
        "pretest_fit_eligible"
    ].astype(bool)

    canonical = canonical.sort_values(
        ["prediction_time", "appointment_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_output_invariants(canonical)
    return canonical


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _build_manifest(
    dataset: pd.DataFrame,
    *,
    input_hashes: Mapping[str, str],
    parquet_sha256: str,
) -> dict[str, object]:
    split_counts = {
        split_name: {
            "positives": int(
                dataset.loc[dataset["split"].eq(split_name), "target"].sum()
            ),
            "rows": int(dataset["split"].eq(split_name).sum()),
        }
        for split_name in ("train", "validation", "test")
    }
    maturity_counts = {
        column: {
            "positives": int(dataset.loc[dataset[column], "target"].sum()),
            "rows": int(dataset[column].sum()),
        }
        for column in MATURITY_COLUMNS
    }
    return {
        "canonical_columns": list(CANONICAL_COLUMNS),
        "dtypes": [
            {"column": column, "dtype": str(dataset[column].dtype)}
            for column in CANONICAL_COLUMNS
        ],
        "feature_columns": list(FEATURE_COLUMNS),
        "fit_time_boundaries": {
            "development": _format_timestamp(DEVELOPMENT_FIT_TIME),
            "pretest": _format_timestamp(PRETEST_FIT_TIME),
        },
        "maturity_counts": maturity_counts,
        "package_versions": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "python": platform.python_version(),
        },
        "parquet_sha256": parquet_sha256.lower(),
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
        "prohibited_columns": sorted(PROHIBITED_COLUMNS),
        "raw_input_hashes": {
            filename: digest.lower()
            for filename, digest in sorted(input_hashes.items())
        },
        "schema_version": "1.0.0",
        "split_boundaries": {
            "test_start": _format_timestamp(TEST_START),
            "validation_start": _format_timestamp(VALIDATION_START),
        },
        "split_counts": split_counts,
        "target_counts": {
            "negative": int(dataset["target"].eq(0).sum()),
            "positive": int(dataset["target"].eq(1).sum()),
        },
        "timestamp_policy": {
            "appointment_and_patient_format": APPOINTMENT_DATETIME_FORMAT,
            "dentist_start_date_format": DENTIST_DATE_FORMAT,
            "timezone": "naive_as_stored",
        },
        "total_rows": int(len(dataset)),
    }


def _temporary_path(destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def write_outputs(
    dataset: pd.DataFrame,
    *,
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
    input_hashes: Mapping[str, str],
) -> None:
    """Write outputs while protecting the verified raw inputs for this build.

    Destinations outside the verified raw directory remain the caller's
    responsibility.
    """

    raw_dir, output_path, manifest_path = validate_output_paths(
        raw_dir,
        output_path,
        manifest_path,
    )
    normalized_hashes = normalize_input_hashes(input_hashes)
    validated_hashes = validate_raw_hashes(
        raw_dir,
        expected_hashes=normalized_hashes,
    )
    validate_output_invariants(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_temporary: Path | None = None
    manifest_temporary: Path | None = None
    try:
        parquet_temporary = _temporary_path(output_path)
        manifest_temporary = _temporary_path(manifest_path)
        dataset.to_parquet(
            parquet_temporary,
            engine="pyarrow",
            index=False,
            compression="snappy",
        )
        parquet_hash = calculate_sha256(parquet_temporary)
        manifest = _build_manifest(
            dataset,
            input_hashes=validated_hashes,
            parquet_sha256=parquet_hash,
        )
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(parquet_temporary, output_path)
        os.replace(manifest_temporary, manifest_path)
    finally:
        if parquet_temporary is not None:
            parquet_temporary.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)


def _build_parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the approved canonical analytical dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=repository_root / "data" / "raw",
        help="Directory containing the three immutable raw CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "data" / "processed" / "analytical_dataset.parquet",
        help="Destination Parquet path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            repository_root
            / "data"
            / "processed"
            / "analytical_dataset.manifest.json"
        ),
        help="Destination JSON manifest path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run canonical dataset construction from the command line."""

    repository_root = Path(__file__).resolve().parents[2]
    args = _build_parser(repository_root).parse_args(argv)
    raw_dir, output_path, manifest_path = validate_output_paths(
        args.raw_dir,
        args.output,
        args.manifest,
    )
    input_hashes = validate_raw_hashes(raw_dir)
    tables = load_raw_data(raw_dir)
    dataset = build_analytical_dataset(tables)
    write_outputs(
        dataset,
        raw_dir=raw_dir,
        output_path=output_path,
        manifest_path=manifest_path,
        input_hashes=input_hashes,
    )
    print(f"Wrote {len(dataset):,} rows to {output_path}")
    print(f"Wrote deterministic manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
