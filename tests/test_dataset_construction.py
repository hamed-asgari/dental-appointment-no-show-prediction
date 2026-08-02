"""Contract tests for deterministic analytical-dataset construction."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data import build_dataset as bd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
EXPECTED_RAW_HASHES = {
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
EXPECTED_CANONICAL_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "prediction_time",
    "target",
    "split",
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
    "development_fit_eligible",
    "pretest_fit_eligible",
)
EXPECTED_FEATURE_COLUMNS = (
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
EXPECTED_DTYPES = {
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
EXPECTED_TOTAL_ROWS = 6_786
EXPECTED_POSITIVES = 820
EXPECTED_NEGATIVES = 5_966
EXPECTED_SPLIT_ROWS = {
    "train": 3_682,
    "validation": 1_541,
    "test": 1_563,
}
EXPECTED_SPLIT_POSITIVES = {
    "train": 434,
    "validation": 192,
    "test": 194,
}
EXPECTED_MATURITY_ROWS = {
    "development_fit_eligible": 3_670,
    "pretest_fit_eligible": 5_223,
}
EXPECTED_MATURITY_POSITIVES = {
    "development_fit_eligible": 432,
    "pretest_fit_eligible": 626,
}


def _copy_raw_files(tmp_path: Path) -> Path:
    copied_raw_dir = tmp_path / "raw"
    copied_raw_dir.mkdir()
    for filename in EXPECTED_RAW_HASHES:
        shutil.copy2(RAW_DIR / filename, copied_raw_dir / filename)
    return copied_raw_dir


def _raw_hashes(raw_dir: Path) -> dict[str, str]:
    return {
        filename: bd.calculate_sha256(raw_dir / filename)
        for filename in EXPECTED_RAW_HASHES
    }


def _temporary_artifacts(root: Path) -> list[Path]:
    return list(root.rglob("*.tmp"))


@pytest.fixture(scope="session")
def raw_hashes() -> dict[str, str]:
    return bd.validate_raw_hashes(RAW_DIR)


@pytest.fixture(scope="session")
def raw_tables() -> bd.RawTables:
    return bd.load_raw_data(RAW_DIR)


@pytest.fixture(scope="session")
def canonical_dataset(raw_tables: bd.RawTables) -> pd.DataFrame:
    return bd.build_analytical_dataset(raw_tables)


def _appointment_fixture(
    *,
    booked_at: str = "2025-01-01 00:00:00",
    scheduled_start_at: str = "2025-01-02 00:00:00",
    status: str = "completed",
    status_updated_at: str = "2025-01-02 01:00:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "appointment_id": pd.Series([1], dtype="int64"),
            "booked_at": pd.to_datetime(
                [booked_at], format=bd.APPOINTMENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
            "scheduled_start_at": pd.to_datetime(
                [scheduled_start_at], format=bd.APPOINTMENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
            "status": pd.Series([status], dtype="string"),
            "status_updated_at": pd.to_datetime(
                [status_updated_at], format=bd.APPOINTMENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
        }
    )


def _joined_feature_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "planned_duration_min": pd.Series([45], dtype="int64"),
            "visit_type": pd.Series(["checkup"], dtype="string"),
            "booking_channel": pd.Series(["phone"], dtype="string"),
            "booked_at": pd.to_datetime(
                ["2025-03-03 08:00:00"], format=bd.APPOINTMENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
            "scheduled_start_at": pd.to_datetime(
                ["2025-03-05 09:30:00"], format=bd.APPOINTMENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
            "prediction_time": pd.to_datetime(
                ["2025-03-04 09:30:00"], format=bd.APPOINTMENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
            "birth_year": pd.Series([2000], dtype="int64"),
            "registered_at": pd.to_datetime(
                ["2025-03-01 10:00:00"], format=bd.PATIENT_DATETIME_FORMAT
            ).astype("datetime64[ns]"),
            "start_date": pd.to_datetime(
                ["2025-02-01"], format=bd.DENTIST_DATE_FORMAT
            ).astype("datetime64[ns]"),
        }
    )


def test_approved_raw_hashes_match(raw_hashes: dict[str, str]) -> None:
    assert dict(bd.EXPECTED_RAW_HASHES) == EXPECTED_RAW_HASHES
    assert raw_hashes == EXPECTED_RAW_HASHES
    assert all(value == value.lower() for value in raw_hashes.values())


def test_raw_hash_mismatch_is_rejected() -> None:
    expected = dict(bd.EXPECTED_RAW_HASHES)
    expected["appointments.csv"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch for appointments.csv"):
        bd.validate_raw_hashes(RAW_DIR, expected_hashes=expected)


def test_raw_hash_mapping_requires_all_approved_files() -> None:
    with pytest.raises(ValueError, match="exactly the three approved raw files"):
        bd.validate_raw_hashes(
            RAW_DIR,
            expected_hashes={
                "appointments.csv": bd.EXPECTED_RAW_HASHES["appointments.csv"]
            },
        )


def test_missing_raw_file_is_rejected_descriptively(tmp_path: Path) -> None:
    copied_raw_dir = _copy_raw_files(tmp_path)
    missing_path = copied_raw_dir / "patients.csv"
    missing_path.unlink()

    with pytest.raises(ValueError, match="Required raw file is missing.*patients.csv"):
        bd.validate_raw_hashes(copied_raw_dir)


def test_malformed_required_timestamp_is_rejected(tmp_path: Path) -> None:
    copied_raw_dir = _copy_raw_files(tmp_path)
    appointments_path = copied_raw_dir / "appointments.csv"
    appointments = pd.read_csv(appointments_path)
    appointments.loc[0, "booked_at"] = "not-a-timestamp"
    appointments.to_csv(appointments_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"appointments\.booked_at does not match required format",
    ):
        bd.load_raw_data(copied_raw_dir)


def test_raw_row_counts_and_required_columns(raw_tables: bd.RawTables) -> None:
    assert len(raw_tables.appointments) == 8_000
    assert len(raw_tables.patients) == 2_000
    assert len(raw_tables.dentists) == 7
    assert tuple(raw_tables.appointments.columns) == bd.APPOINTMENT_SOURCE_COLUMNS
    assert tuple(raw_tables.patients.columns) == bd.PATIENT_SOURCE_COLUMNS
    assert tuple(raw_tables.dentists.columns) == bd.DENTIST_SOURCE_COLUMNS
    bd.validate_required_columns(raw_tables)


def test_missing_required_column_is_rejected(raw_tables: bd.RawTables) -> None:
    invalid = bd.RawTables(
        appointments=raw_tables.appointments.drop(columns="status"),
        patients=raw_tables.patients,
        dentists=raw_tables.dentists,
    )
    with pytest.raises(ValueError, match="missing required columns: status"):
        bd.validate_required_columns(invalid)


@pytest.mark.parametrize(
    ("table_name", "primary_key", "defect", "message"),
    [
        ("appointments", "appointment_id", "duplicate", "must be unique"),
        (
            "appointments",
            "appointment_id",
            "null",
            "must not contain missing values",
        ),
        ("patients", "patient_id", "duplicate", "must be unique"),
        ("patients", "patient_id", "null", "must not contain missing values"),
        ("dentists", "dentist_id", "duplicate", "must be unique"),
        ("dentists", "dentist_id", "null", "must not contain missing values"),
    ],
)
def test_duplicate_and_null_primary_keys_are_rejected(
    raw_tables: bd.RawTables,
    table_name: str,
    primary_key: str,
    defect: str,
    message: str,
) -> None:
    frames = {
        "appointments": raw_tables.appointments.copy(deep=True),
        "patients": raw_tables.patients.copy(deep=True),
        "dentists": raw_tables.dentists.copy(deep=True),
    }
    frame = frames[table_name]
    if defect == "duplicate":
        frame.loc[1, primary_key] = frame.loc[0, primary_key]
    else:
        frame[primary_key] = frame[primary_key].astype("Int64")
        frame.loc[0, primary_key] = pd.NA
    invalid = bd.RawTables(**frames)

    with pytest.raises(
        ValueError,
        match=rf"{table_name}\.{primary_key} {message}",
    ):
        bd.validate_required_columns(invalid)


def test_primary_keys_and_join_integrity(raw_tables: bd.RawTables) -> None:
    assert raw_tables.appointments["appointment_id"].is_unique
    assert raw_tables.patients["patient_id"].is_unique
    assert raw_tables.dentists["dentist_id"].is_unique
    cohort = bd.reconstruct_eligible_cohort(raw_tables.appointments)
    joined = bd.join_reference_data(
        cohort,
        raw_tables.patients,
        raw_tables.dentists,
    )
    assert len(joined) == len(cohort)
    assert joined[["birth_year", "registered_at", "start_date"]].notna().all().all()


def test_unmatched_reference_key_is_rejected(raw_tables: bd.RawTables) -> None:
    cohort = bd.reconstruct_eligible_cohort(raw_tables.appointments)
    patient_id = int(cohort.iloc[0]["patient_id"])
    patients = raw_tables.patients.loc[
        raw_tables.patients["patient_id"].ne(patient_id)
    ].copy()
    with pytest.raises(ValueError, match="Unmatched patient_id"):
        bd.join_reference_data(cohort, patients, raw_tables.dentists)


def test_unmatched_dentist_key_is_rejected(raw_tables: bd.RawTables) -> None:
    cohort = bd.reconstruct_eligible_cohort(raw_tables.appointments)
    dentist_id = int(cohort.iloc[0]["dentist_id"])
    dentists = raw_tables.dentists.loc[
        raw_tables.dentists["dentist_id"].ne(dentist_id)
    ].copy()
    with pytest.raises(ValueError, match="Unmatched dentist_id"):
        bd.join_reference_data(cohort, raw_tables.patients, dentists)


def test_eligible_cohort_and_target_reconcile(raw_tables: bd.RawTables) -> None:
    cohort = bd.reconstruct_eligible_cohort(raw_tables.appointments)
    target = bd.construct_target(cohort)
    assert bd.EXPECTED_TOTAL_ROWS == EXPECTED_TOTAL_ROWS
    assert bd.EXPECTED_POSITIVES == EXPECTED_POSITIVES
    assert bd.EXPECTED_NEGATIVES == EXPECTED_NEGATIVES
    assert len(cohort) == EXPECTED_TOTAL_ROWS
    assert int(target.sum()) == EXPECTED_POSITIVES
    assert int(target.eq(0).sum()) == EXPECTED_NEGATIVES
    assert cohort["booked_at"].le(cohort["prediction_time"]).all()


def test_booking_exactly_at_prediction_time_is_included() -> None:
    appointments = _appointment_fixture(booked_at="2025-01-01 00:00:00")
    cohort = bd.reconstruct_eligible_cohort(appointments)
    assert cohort["appointment_id"].tolist() == [1]


@pytest.mark.parametrize("inactive_status", ["cancelled", "rescheduled"])
def test_inactive_exactly_at_prediction_time_is_excluded(
    inactive_status: str,
) -> None:
    appointments = _appointment_fixture(
        status=inactive_status,
        status_updated_at="2025-01-01 00:00:00",
    )
    cohort = bd.reconstruct_eligible_cohort(appointments)
    assert cohort.empty


def test_unexpected_eligible_status_is_rejected() -> None:
    joined = pd.DataFrame({"status": pd.Series(["unexpected"], dtype="string")})
    with pytest.raises(ValueError, match="Unexpected eligible status"):
        bd.construct_target(joined)


def test_exact_split_counts_and_positive_counts(
    canonical_dataset: pd.DataFrame,
) -> None:
    assert dict(bd.EXPECTED_SPLIT_ROWS) == EXPECTED_SPLIT_ROWS
    assert dict(bd.EXPECTED_SPLIT_POSITIVES) == EXPECTED_SPLIT_POSITIVES
    for split_name in ("train", "validation", "test"):
        rows = canonical_dataset["split"].eq(split_name)
        assert int(rows.sum()) == EXPECTED_SPLIT_ROWS[split_name]
        assert int(canonical_dataset.loc[rows, "target"].sum()) == (
            EXPECTED_SPLIT_POSITIVES[split_name]
        )
    assert canonical_dataset["split"].notna().all()
    assert set(canonical_dataset["split"].unique()) == bd.ALLOWED_SPLITS


def test_split_boundaries_are_half_open() -> None:
    prediction_time = pd.Series(
        pd.to_datetime(
            [
                "2025-02-28 23:59:59",
                "2025-03-01 00:00:00",
                "2025-07-31 23:59:59",
                "2025-08-01 00:00:00",
            ],
            format=bd.APPOINTMENT_DATETIME_FORMAT,
        ).astype("datetime64[ns]")
    )
    splits = bd.assign_temporal_splits(prediction_time)
    assert splits.tolist() == ["train", "validation", "validation", "test"]
    assert isinstance(splits.dtype, pd.StringDtype)


def test_exact_label_maturity_counts(canonical_dataset: pd.DataFrame) -> None:
    assert dict(bd.EXPECTED_MATURITY_ROWS) == EXPECTED_MATURITY_ROWS
    assert dict(bd.EXPECTED_MATURITY_POSITIVES) == EXPECTED_MATURITY_POSITIVES
    for column in EXPECTED_MATURITY_ROWS:
        mask = canonical_dataset[column]
        assert int(mask.sum()) == EXPECTED_MATURITY_ROWS[column]
        assert int(canonical_dataset.loc[mask, "target"].sum()) == (
            EXPECTED_MATURITY_POSITIVES[column]
        )


def test_label_maturity_uses_strict_less_than() -> None:
    frame = pd.DataFrame(
        {
            "split": pd.Series(["train", "train", "validation"], dtype="string"),
            "status_updated_at": pd.to_datetime(
                [
                    "2025-02-28 23:59:59",
                    "2025-03-01 00:00:00",
                    "2025-02-28 23:59:59",
                ],
                format=bd.APPOINTMENT_DATETIME_FORMAT,
            ).astype("datetime64[ns]"),
        }
    )
    mask = bd.label_maturity_mask(
        frame,
        model_fit_time=bd.DEVELOPMENT_FIT_TIME,
        allowed_splits=("train",),
    )
    assert mask.tolist() == [True, False, False]
    assert mask.dtype == bool


def test_feature_allowlist_is_exact_and_ordered() -> None:
    assert bd.FEATURE_COLUMNS == EXPECTED_FEATURE_COLUMNS


def test_feature_allowlist_excludes_all_non_predictive_roles() -> None:
    disallowed = (
        set(bd.AUDIT_COLUMNS)
        | set(bd.TARGET_PARTITION_COLUMNS)
        | set(bd.MATURITY_COLUMNS)
        | bd.PROHIBITED_COLUMNS
    )
    assert set(bd.FEATURE_COLUMNS).isdisjoint(disallowed)


def test_canonical_schema_and_prohibited_fields(
    canonical_dataset: pd.DataFrame,
) -> None:
    assert bd.CANONICAL_COLUMNS == EXPECTED_CANONICAL_COLUMNS
    assert tuple(canonical_dataset.columns) == EXPECTED_CANONICAL_COLUMNS
    assert len(canonical_dataset.columns) == 18
    assert set(canonical_dataset.columns).isdisjoint(bd.PROHIBITED_COLUMNS)


def test_approved_feature_formulas_match_hand_calculation() -> None:
    features = bd.derive_approved_features(_joined_feature_fixture())
    row = features.iloc[0]
    assert row["planned_duration_min"] == 45
    assert row["visit_type"] == "checkup"
    assert row["booking_channel"] == "phone"
    assert row["booking_lead_time_hours"] == pytest.approx(49.5)
    assert row["scheduled_weekday"] == 2
    assert row["scheduled_hour"] == 9
    assert row["scheduled_month"] == 3
    assert row["approximate_age_at_prediction"] == 25
    assert row["patient_registration_tenure_days"] == 2
    assert row["dentist_tenure_days"] == 31


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("booked_at", "2025-03-06 08:00:00", "booking_lead_time_hours"),
        ("birth_year", 2026, "approximate_age_at_prediction"),
        ("registered_at", "2025-03-05 10:00:00", "patient registration"),
        ("start_date", "2025-03-05 00:00:00", "dentist start_date"),
    ],
)
def test_impossible_derived_feature_values_are_rejected(
    column: str,
    value: object,
    message: str,
) -> None:
    joined = _joined_feature_fixture()
    if column in {"booked_at", "registered_at", "start_date"}:
        joined[column] = pd.to_datetime(
            [value], format=bd.APPOINTMENT_DATETIME_FORMAT
        ).astype("datetime64[ns]")
    else:
        joined[column] = value
    with pytest.raises(ValueError, match=message):
        bd.derive_approved_features(joined)


def test_exact_canonical_dtypes(canonical_dataset: pd.DataFrame) -> None:
    actual = {
        column: str(canonical_dataset[column].dtype)
        for column in EXPECTED_CANONICAL_COLUMNS
    }
    assert dict(bd.EXPECTED_DTYPES) == EXPECTED_DTYPES
    assert actual == EXPECTED_DTYPES


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("wrong_order", "Canonical columns or order differ"),
        ("missing_column", "Canonical columns or order differ"),
        ("extra_column", "Canonical columns or order differ"),
        ("wrong_dtype", "appointment_id dtype must be int64"),
        ("missing_value", "Canonical dataset contains missing values"),
    ],
)
def test_invalid_canonical_schema_is_rejected(
    canonical_dataset: pd.DataFrame,
    defect: str,
    message: str,
) -> None:
    invalid = canonical_dataset.copy(deep=True)
    if defect == "wrong_order":
        columns = list(EXPECTED_CANONICAL_COLUMNS)
        columns[0], columns[1] = columns[1], columns[0]
        invalid = invalid.loc[:, columns]
    elif defect == "missing_column":
        invalid = invalid.drop(columns="visit_type")
    elif defect == "extra_column":
        invalid["unexpected_column"] = 1
    elif defect == "wrong_dtype":
        invalid["appointment_id"] = invalid["appointment_id"].astype("int32")
    else:
        invalid.loc[0, "visit_type"] = pd.NA

    with pytest.raises(ValueError, match=message):
        bd.validate_output_invariants(invalid)


def test_output_completeness_uniqueness_and_order(
    canonical_dataset: pd.DataFrame,
) -> None:
    assert not canonical_dataset.isna().any().any()
    assert canonical_dataset["appointment_id"].is_unique
    assert canonical_dataset.index.equals(pd.RangeIndex(len(canonical_dataset)))
    expected = canonical_dataset.sort_values(
        ["prediction_time", "appointment_id"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical_dataset, expected)


def test_shuffled_raw_inputs_produce_identical_dataset(
    raw_tables: bd.RawTables,
    canonical_dataset: pd.DataFrame,
) -> None:
    shuffled = bd.RawTables(
        appointments=raw_tables.appointments.sample(frac=1, random_state=11),
        patients=raw_tables.patients.sample(frac=1, random_state=12),
        dentists=raw_tables.dentists.sample(frac=1, random_state=13),
    )
    actual = bd.build_analytical_dataset(shuffled)
    pd.testing.assert_frame_equal(actual, canonical_dataset)


def test_build_does_not_mutate_raw_frames(raw_tables: bd.RawTables) -> None:
    appointment_before = raw_tables.appointments.copy(deep=True)
    patient_before = raw_tables.patients.copy(deep=True)
    dentist_before = raw_tables.dentists.copy(deep=True)
    bd.build_analytical_dataset(raw_tables)
    pd.testing.assert_frame_equal(raw_tables.appointments, appointment_before)
    pd.testing.assert_frame_equal(raw_tables.patients, patient_before)
    pd.testing.assert_frame_equal(raw_tables.dentists, dentist_before)


def test_select_model_features_is_exact_and_defensive(
    canonical_dataset: pd.DataFrame,
) -> None:
    selected = bd.select_model_features(canonical_dataset)
    assert tuple(selected.columns) == EXPECTED_FEATURE_COLUMNS
    original = canonical_dataset.iloc[0]["planned_duration_min"]
    selected.iloc[0, 0] = original + 1
    assert canonical_dataset.iloc[0]["planned_duration_min"] == original


def test_select_development_rows_never_exposes_test(
    canonical_dataset: pd.DataFrame,
) -> None:
    development = bd.select_development_rows(canonical_dataset)
    assert set(development["split"].unique()) == {"train", "validation"}
    assert len(development) == 5_223
    development.iloc[0, 0] = -1
    assert canonical_dataset.iloc[0]["appointment_id"] != -1


def test_select_test_rows_requires_explicit_opt_in(
    canonical_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(PermissionError, match="allow_test=True"):
        bd.select_test_rows(canonical_dataset)


def test_select_test_rows_returns_only_test_after_opt_in(
    canonical_dataset: pd.DataFrame,
) -> None:
    test_rows = bd.select_test_rows(canonical_dataset, allow_test=True)
    assert set(test_rows["split"].unique()) == {"test"}
    assert len(test_rows) == 1_563
    test_rows.iloc[0, 0] = -1
    assert canonical_dataset.loc[canonical_dataset["split"].eq("test")].iloc[0][
        "appointment_id"
    ] != -1


def test_resolved_output_and_manifest_alias_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("artifacts") / "dataset.parquet"
    manifest_path = tmp_path / "artifacts" / "." / "nested" / ".." / "dataset.parquet"
    resolved_destination = (tmp_path / output_path).resolve(strict=False)

    with pytest.raises(ValueError, match="resolve to the same destination"):
        bd.write_outputs(
            canonical_dataset,
            raw_dir=RAW_DIR,
            output_path=output_path,
            manifest_path=manifest_path,
            input_hashes=raw_hashes,
        )

    assert not resolved_destination.exists()
    assert _temporary_artifacts(tmp_path) == []


def test_output_inside_raw_directory_is_rejected_without_modification(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
) -> None:
    copied_raw_dir = _copy_raw_files(tmp_path)
    hashes_before = _raw_hashes(copied_raw_dir)
    appointments_path = copied_raw_dir / "appointments.csv"
    manifest_path = tmp_path / "safe" / "manifest.json"

    with pytest.raises(ValueError, match="inside the raw directory"):
        bd.write_outputs(
            canonical_dataset,
            raw_dir=copied_raw_dir,
            output_path=appointments_path,
            manifest_path=manifest_path,
            input_hashes=raw_hashes,
        )

    assert _raw_hashes(copied_raw_dir) == hashes_before
    prefix = appointments_path.read_bytes()[:32]
    assert prefix.startswith(b"appointment_id,")
    assert not prefix.startswith(b"PAR1")
    assert not manifest_path.exists()
    assert _temporary_artifacts(tmp_path) == []


def test_manifest_inside_raw_directory_is_rejected_without_modification(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
) -> None:
    copied_raw_dir = _copy_raw_files(tmp_path)
    hashes_before = _raw_hashes(copied_raw_dir)
    output_path = tmp_path / "safe" / "dataset.parquet"
    manifest_path = copied_raw_dir / "nested" / "manifest.json"

    with pytest.raises(ValueError, match="inside the raw directory"):
        bd.write_outputs(
            canonical_dataset,
            raw_dir=copied_raw_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            input_hashes=raw_hashes,
        )

    assert _raw_hashes(copied_raw_dir) == hashes_before
    assert not output_path.exists()
    assert not manifest_path.exists()
    assert _temporary_artifacts(tmp_path) == []


@pytest.mark.parametrize("protected_destination", ["output", "manifest"])
def test_unrelated_raw_dir_cannot_bypass_protected_raw_files(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
    protected_destination: str,
) -> None:
    protected_raw_dir = _copy_raw_files(tmp_path)
    unrelated_raw_dir = tmp_path / "unrelated-raw"
    unrelated_raw_dir.mkdir()
    hashes_before = _raw_hashes(protected_raw_dir)
    safe_output = tmp_path / "safe" / "dataset.parquet"
    safe_manifest = tmp_path / "safe" / "manifest.json"
    output_path = (
        protected_raw_dir / "appointments.csv"
        if protected_destination == "output"
        else safe_output
    )
    manifest_path = (
        protected_raw_dir / "patients.csv"
        if protected_destination == "manifest"
        else safe_manifest
    )

    with pytest.raises(
        ValueError,
        match="Required raw file is missing.*appointments.csv",
    ):
        bd.write_outputs(
            canonical_dataset,
            raw_dir=unrelated_raw_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            input_hashes=raw_hashes,
        )

    assert _raw_hashes(protected_raw_dir) == hashes_before
    assert (protected_raw_dir / "appointments.csv").read_bytes().startswith(
        b"appointment_id,"
    )
    assert (protected_raw_dir / "patients.csv").read_bytes().startswith(b"patient_id,")
    assert not (protected_raw_dir / "appointments.csv").read_bytes().startswith(b"PAR1")
    assert not (protected_raw_dir / "patients.csv").read_bytes().startswith(b"PAR1")
    if protected_destination == "output":
        assert not safe_manifest.exists()
    else:
        assert not safe_output.exists()
    assert _temporary_artifacts(tmp_path) == []


@pytest.mark.parametrize("protected_destination", ["output", "manifest"])
def test_cli_rejects_destinations_inside_raw_directory(
    tmp_path: Path,
    protected_destination: str,
) -> None:
    copied_raw_dir = _copy_raw_files(tmp_path)
    hashes_before = _raw_hashes(copied_raw_dir)
    safe_output = tmp_path / "safe" / "dataset.parquet"
    safe_manifest = tmp_path / "safe" / "manifest.json"
    output_path = (
        copied_raw_dir / "appointments.csv"
        if protected_destination == "output"
        else safe_output
    )
    manifest_path = (
        copied_raw_dir / "blocked-manifest.json"
        if protected_destination == "manifest"
        else safe_manifest
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.build_dataset",
            "--raw-dir",
            str(copied_raw_dir),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must not be inside the raw directory" in result.stderr
    assert _raw_hashes(copied_raw_dir) == hashes_before
    appointments_prefix = (copied_raw_dir / "appointments.csv").read_bytes()[:32]
    assert appointments_prefix.startswith(b"appointment_id,")
    assert not appointments_prefix.startswith(b"PAR1")
    if protected_destination == "output":
        assert not safe_manifest.exists()
    else:
        assert not safe_output.exists()
        assert not manifest_path.exists()
    assert _temporary_artifacts(tmp_path) == []


@pytest.mark.parametrize(
    "input_hashes",
    [
        pytest.param({}, id="empty"),
        pytest.param(
            {
                filename: digest
                for filename, digest in EXPECTED_RAW_HASHES.items()
                if filename != "patients.csv"
            },
            id="missing-key",
        ),
        pytest.param(
            {**EXPECTED_RAW_HASHES, "unexpected.csv": "0" * 64},
            id="extra-key",
        ),
        pytest.param(
            {**EXPECTED_RAW_HASHES, "appointments.csv": 123},
            id="non-string",
        ),
        pytest.param(
            {**EXPECTED_RAW_HASHES, "appointments.csv": "0" * 63},
            id="63-characters",
        ),
        pytest.param(
            {**EXPECTED_RAW_HASHES, "appointments.csv": "0" * 65},
            id="65-characters",
        ),
        pytest.param(
            {**EXPECTED_RAW_HASHES, "appointments.csv": "g" * 64},
            id="non-hexadecimal",
        ),
        pytest.param(
            {**EXPECTED_RAW_HASHES, "appointments.csv": "0" * 64},
            id="incorrect-digest",
        ),
    ],
)
def test_invalid_provenance_hashes_fail_before_writing(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
    input_hashes: dict[str, object],
) -> None:
    output_path = tmp_path / "artifacts" / "dataset.parquet"
    manifest_path = tmp_path / "artifacts" / "manifest.json"

    with pytest.raises(ValueError):
        bd.write_outputs(
            canonical_dataset,
            raw_dir=RAW_DIR,
            output_path=output_path,
            manifest_path=manifest_path,
            input_hashes=input_hashes,
        )

    assert not output_path.exists()
    assert not manifest_path.exists()
    assert _temporary_artifacts(tmp_path) == []


def test_uppercase_provenance_hashes_are_normalized_in_manifest(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
) -> None:
    copied_raw_dir = _copy_raw_files(tmp_path)
    validated_hashes = bd.validate_raw_hashes(copied_raw_dir)
    output_path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "manifest.json"
    uppercase_hashes = {
        filename: digest.upper() for filename, digest in EXPECTED_RAW_HASHES.items()
    }

    bd.write_outputs(
        canonical_dataset,
        raw_dir=copied_raw_dir,
        output_path=output_path,
        manifest_path=manifest_path,
        input_hashes=uppercase_hashes,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["raw_input_hashes"] == validated_hashes
    assert manifest["raw_input_hashes"] == EXPECTED_RAW_HASHES


def test_first_temporary_file_is_removed_when_second_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
) -> None:
    output_path = tmp_path / "artifacts" / "dataset.parquet"
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    created_paths: list[Path] = []

    def fail_second_temporary_path(destination: Path) -> Path:
        if created_paths:
            raise OSError("forced second temporary-path creation failure")
        temporary_path = destination.parent / ".forced-first.tmp"
        temporary_path.touch()
        created_paths.append(temporary_path)
        return temporary_path

    monkeypatch.setattr(bd, "_temporary_path", fail_second_temporary_path)
    with pytest.raises(OSError, match="forced second temporary-path creation failure"):
        bd.write_outputs(
            canonical_dataset,
            raw_dir=RAW_DIR,
            output_path=output_path,
            manifest_path=manifest_path,
            input_hashes=raw_hashes,
        )

    assert len(created_paths) == 1
    assert not created_paths[0].exists()
    assert not output_path.exists()
    assert not manifest_path.exists()
    assert _temporary_artifacts(tmp_path) == []


def test_parquet_and_manifest_round_trip(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
) -> None:
    output_path = tmp_path / "analytical_dataset.parquet"
    manifest_path = tmp_path / "analytical_dataset.manifest.json"
    bd.write_outputs(
        canonical_dataset,
        raw_dir=RAW_DIR,
        output_path=output_path,
        manifest_path=manifest_path,
        input_hashes=raw_hashes,
    )
    restored = pd.read_parquet(output_path, engine="pyarrow")
    pd.testing.assert_frame_equal(restored, canonical_dataset)

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest_text.endswith("\n")
    assert manifest_text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["canonical_columns"] == list(EXPECTED_CANONICAL_COLUMNS)
    assert manifest["feature_columns"] == list(EXPECTED_FEATURE_COLUMNS)
    assert manifest["prohibited_columns"] == sorted(bd.PROHIBITED_COLUMNS)
    assert manifest["raw_input_hashes"] == EXPECTED_RAW_HASHES
    assert manifest["total_rows"] == 6_786
    assert manifest["target_counts"] == {"negative": 5_966, "positive": 820}
    assert manifest["split_counts"] == {
        "train": {"positives": 434, "rows": 3_682},
        "validation": {"positives": 192, "rows": 1_541},
        "test": {"positives": 194, "rows": 1_563},
    }
    assert manifest["maturity_counts"] == {
        "development_fit_eligible": {"positives": 432, "rows": 3_670},
        "pretest_fit_eligible": {"positives": 626, "rows": 5_223},
    }
    assert manifest["parquet_sha256"] == bd.calculate_sha256(output_path)
    assert manifest["dtypes"] == [
        {"column": column, "dtype": EXPECTED_DTYPES[column]}
        for column in EXPECTED_CANONICAL_COLUMNS
    ]


def test_repeated_writes_are_semantically_identical(
    tmp_path: Path,
    canonical_dataset: pd.DataFrame,
    raw_hashes: dict[str, str],
) -> None:
    first_output = tmp_path / "first.parquet"
    first_manifest = tmp_path / "first.json"
    second_output = tmp_path / "second.parquet"
    second_manifest = tmp_path / "second.json"
    bd.write_outputs(
        canonical_dataset,
        raw_dir=RAW_DIR,
        output_path=first_output,
        manifest_path=first_manifest,
        input_hashes=raw_hashes,
    )
    bd.write_outputs(
        canonical_dataset,
        raw_dir=RAW_DIR,
        output_path=second_output,
        manifest_path=second_manifest,
        input_hashes=raw_hashes,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first_output, engine="pyarrow"),
        pd.read_parquet(second_output, engine="pyarrow"),
    )
    assert json.loads(first_manifest.read_text(encoding="utf-8")) == json.loads(
        second_manifest.read_text(encoding="utf-8")
    )
    assert bd.calculate_sha256(first_output) == bd.calculate_sha256(second_output)


def test_processed_directory_contains_only_gitkeep() -> None:
    files = {
        path.name
        for path in (REPOSITORY_ROOT / "data" / "processed").iterdir()
        if path.is_file()
    }
    assert files == {".gitkeep"}
