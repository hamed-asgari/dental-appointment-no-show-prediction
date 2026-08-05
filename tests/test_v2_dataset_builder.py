from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_v2_dataset import (
    V2RawTables,
    assign_v2_evaluation_partitions,
    build_v2_feature_dataset,
    build_verified_v2_feature_dataset,
    label_maturity_mask,
    load_verified_v2_raw_tables,
    select_v2_model_features,
    validate_frozen_v2_inputs,
)
from src.features.current_appointment import (
    build_current_appointment_features,
)
from src.features.schema import (
    CURRENT_APPOINTMENT_DTYPES,
    CURRENT_APPOINTMENT_FEATURE_COLUMNS,
    CURRENT_APPOINTMENT_OUTPUT_COLUMNS,
    V2_EVALUATION_PARTITIONS,
    V2_FEATURE_DATASET_COLUMNS,
    V2_MODEL_FEATURE_COLUMNS,
    V2_PROHIBITED_MODEL_COLUMNS,
)
from src.synthetic.config import load_benchmark_config


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw" / "v2"
_MANIFEST_PATH = _RAW_DIR / "v2_synthetic_benchmark.manifest.json"
_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_synthetic_benchmark.json"


def _appointment(
    *,
    appointment_id: int,
    patient_id: int = 1,
    dentist_id: int = 1,
    booked_at: str = "2024-01-01 09:00:00",
    scheduled_start_at: str = "2024-01-10 10:00:00",
    status: str = "completed",
    status_updated_at: str = "2024-01-10 11:00:00",
    reminder_sent_at: str | None = "2024-01-08 12:00:00",
) -> dict[str, object]:
    return {
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "dentist_id": dentist_id,
        "booked_at": booked_at,
        "scheduled_start_at": scheduled_start_at,
        "planned_duration_min": 45,
        "visit_type": "checkup",
        "booking_channel": "phone",
        "status": status,
        "status_updated_at": status_updated_at,
        "reminder_sent_at": reminder_sent_at,
    }


def _patients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": 1,
                "birth_year": 1990,
                "registered_at": "2020-01-01 00:00:00",
            },
            {
                "patient_id": 2,
                "birth_year": 2000,
                "registered_at": "2023-01-01 00:00:00",
            },
        ]
    )


def _dentists() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dentist_id": 1,
                "start_date": "2020-01-01",
            },
            {
                "dentist_id": 2,
                "start_date": "2024-01-10",
            },
        ]
    )


@pytest.fixture(scope="module")
def benchmark_tables() -> V2RawTables:
    return load_verified_v2_raw_tables(
        raw_dir=_RAW_DIR,
        manifest_path=_MANIFEST_PATH,
        config_path=_CONFIG_PATH,
    )


@pytest.fixture(scope="module")
def benchmark_dataset() -> pd.DataFrame:
    return build_verified_v2_feature_dataset(
        raw_dir=_RAW_DIR,
        manifest_path=_MANIFEST_PATH,
        config_path=_CONFIG_PATH,
    )


def test_current_feature_schema_is_frozen() -> None:
    assert len(CURRENT_APPOINTMENT_FEATURE_COLUMNS) == 11
    assert len(CURRENT_APPOINTMENT_OUTPUT_COLUMNS) == 15
    assert tuple(CURRENT_APPOINTMENT_DTYPES) == CURRENT_APPOINTMENT_OUTPUT_COLUMNS
    assert len(V2_MODEL_FEATURE_COLUMNS) == 32
    assert len(V2_FEATURE_DATASET_COLUMNS) == 38
    assert set(V2_MODEL_FEATURE_COLUMNS).isdisjoint(
        V2_PROHIBITED_MODEL_COLUMNS
    )


def test_current_features_use_prediction_time_safe_values() -> None:
    appointments = pd.DataFrame([_appointment(appointment_id=1)])
    result = build_current_appointment_features(
        appointments,
        _patients(),
        _dentists(),
    )
    row = result.iloc[0]

    assert list(result.columns) == list(CURRENT_APPOINTMENT_OUTPUT_COLUMNS)
    assert row["prediction_time"] == pd.Timestamp("2024-01-09 10:00:00")
    assert row["booking_lead_time_hours"] == pytest.approx(217.0)
    assert row["scheduled_weekday"] == 2
    assert row["scheduled_hour"] == 10
    assert row["scheduled_month"] == 1
    assert row["approximate_age_at_prediction"] == 34
    assert row["patient_registration_tenure_days"] == 1469
    assert row["dentist_tenure_days"] == 1469
    assert bool(row["reminder_sent_by_prediction_time"]) is True


def test_reminder_after_prediction_time_is_false() -> None:
    appointments = pd.DataFrame(
        [
            _appointment(
                appointment_id=1,
                reminder_sent_at="2024-01-09 12:00:00",
            )
        ]
    )
    result = build_current_appointment_features(
        appointments,
        _patients(),
        _dentists(),
    )
    assert not bool(result.loc[0, "reminder_sent_by_prediction_time"])


def test_missing_reminder_timestamp_is_false() -> None:
    appointments = pd.DataFrame(
        [_appointment(appointment_id=1, reminder_sent_at=None)]
    )
    result = build_current_appointment_features(
        appointments,
        _patients(),
        _dentists(),
    )
    assert not bool(result.loc[0, "reminder_sent_by_prediction_time"])


def test_current_features_allow_pre_start_dentist_assignment() -> None:
    appointments = pd.DataFrame(
        [
            _appointment(
                appointment_id=1,
                dentist_id=2,
                scheduled_start_at="2024-01-10 10:00:00",
            )
        ]
    )
    result = build_current_appointment_features(
        appointments,
        _patients(),
        _dentists(),
    )
    assert result.loc[0, "dentist_tenure_days"] == -1


def test_current_features_reject_unknown_patient() -> None:
    appointments = pd.DataFrame(
        [_appointment(appointment_id=1, patient_id=99)]
    )
    with pytest.raises(ValueError, match="unknown patients"):
        build_current_appointment_features(
            appointments,
            _patients(),
            _dentists(),
        )


def test_current_features_reject_unknown_dentist() -> None:
    appointments = pd.DataFrame(
        [_appointment(appointment_id=1, dentist_id=99)]
    )
    with pytest.raises(ValueError, match="unknown dentists"):
        build_current_appointment_features(
            appointments,
            _patients(),
            _dentists(),
        )


def test_current_features_reject_duplicate_reference_keys() -> None:
    patients = pd.concat([_patients(), _patients().iloc[[0]]])
    with pytest.raises(ValueError, match="patients.patient_id must be unique"):
        build_current_appointment_features(
            pd.DataFrame([_appointment(appointment_id=1)]),
            patients,
            _dentists(),
        )


def test_current_feature_output_is_input_order_invariant() -> None:
    appointments = pd.DataFrame(
        [
            _appointment(appointment_id=1),
            _appointment(
                appointment_id=2,
                patient_id=2,
                scheduled_start_at="2024-02-10 11:00:00",
                status_updated_at="2024-02-10 12:00:00",
            ),
        ]
    )
    expected = build_current_appointment_features(
        appointments,
        _patients(),
        _dentists(),
    )
    actual = build_current_appointment_features(
        appointments.sample(frac=1.0, random_state=7),
        _patients().sample(frac=1.0, random_state=8),
        _dentists().sample(frac=1.0, random_state=9),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_evaluation_partition_boundaries_are_half_open() -> None:
    config = load_benchmark_config(_CONFIG_PATH)
    values = pd.Series(
        pd.to_datetime(
            [
                "2022-12-31 23:59:59",
                "2023-01-01",
                "2024-01-01",
                "2025-01-01",
                "2025-07-01",
                "2026-01-01",
                "2026-07-01",
                "2026-10-01",
                "2027-01-01",
                "2027-12-31 23:59:59",
            ],
            format="mixed",
        )
    )
    actual = assign_v2_evaluation_partitions(values, config)
    assert actual.tolist() == [
        "context_only",
        "warmup",
        "development_fit",
        "fold_1_validation",
        "fold_2_validation",
        "fold_3_validation",
        "calibration",
        "policy_selection",
        "final_test",
        "final_test",
    ]


def test_evaluation_partition_rejects_out_of_schedule_future() -> None:
    config = load_benchmark_config(_CONFIG_PATH)
    with pytest.raises(ValueError, match="outside the frozen evaluation"):
        assign_v2_evaluation_partitions(
            pd.Series(pd.to_datetime(["2028-01-01"])),
            config,
        )


def test_label_maturity_is_strict() -> None:
    dataset = pd.DataFrame(
        {
            "evaluation_partition": pd.Series(
                ["development_fit"] * 3,
                dtype="string",
            ),
            "label_available_at": pd.to_datetime(
                [
                    "2024-12-31 23:59:59.999999999",
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:00.000000001",
                ],
                format="mixed",
            ),
        }
    )
    mask = label_maturity_mask(
        dataset,
        model_fit_time=pd.Timestamp("2025-01-01"),
        allowed_partitions=("development_fit",),
    )
    assert mask.tolist() == [True, False, False]


def test_label_maturity_rejects_unknown_partition() -> None:
    dataset = pd.DataFrame(
        {
            "evaluation_partition": pd.Series(
                ["development_fit"],
                dtype="string",
            ),
            "label_available_at": pd.to_datetime(["2024-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="Unknown allowed"):
        label_maturity_mask(
            dataset,
            model_fit_time=pd.Timestamp("2025-01-01"),
            allowed_partitions=("not_a_partition",),
        )


def test_frozen_input_validation_returns_exact_identities() -> None:
    identities = validate_frozen_v2_inputs(
        raw_dir=_RAW_DIR,
        manifest_path=_MANIFEST_PATH,
        config_path=_CONFIG_PATH,
    )
    assert set(identities) == {
        "manifest",
        "configuration",
        "appointments.csv",
        "patients.csv",
        "dentists.csv",
    }
    assert all(len(value) == 64 for value in identities.values())


def test_tampered_raw_file_is_rejected(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "v2"
    raw_dir.mkdir()
    for path in _RAW_DIR.iterdir():
        if path.is_file():
            (raw_dir / path.name).write_bytes(path.read_bytes())
    appointments = raw_dir / "appointments.csv"
    appointments.write_bytes(appointments.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="raw SHA-256 mismatch"):
        validate_frozen_v2_inputs(
            raw_dir=raw_dir,
            manifest_path=raw_dir
            / "v2_synthetic_benchmark.manifest.json",
            config_path=_CONFIG_PATH,
        )


def test_benchmark_feature_dataset_has_frozen_shape_and_columns(
    benchmark_dataset: pd.DataFrame,
) -> None:
    assert benchmark_dataset.shape == (21_755, 38)
    assert list(benchmark_dataset.columns) == list(V2_FEATURE_DATASET_COLUMNS)
    assert benchmark_dataset["appointment_id"].is_unique
    assert not benchmark_dataset.isna().any().any()


def test_benchmark_partition_counts_are_frozen(
    benchmark_dataset: pd.DataFrame,
) -> None:
    assert benchmark_dataset["evaluation_partition"].value_counts().to_dict() == {
        "development_fit": 4_467,
        "final_test": 4_343,
        "warmup": 4_324,
        "fold_2_validation": 2_231,
        "fold_1_validation": 2_150,
        "fold_3_validation": 2_086,
        "calibration": 1_081,
        "policy_selection": 1_063,
        "context_only": 10,
    }
    assert set(benchmark_dataset["evaluation_partition"]) == set(
        V2_EVALUATION_PARTITIONS
    )


def test_context_only_rows_are_the_ten_pre_2023_predictions(
    benchmark_dataset: pd.DataFrame,
) -> None:
    context = benchmark_dataset.loc[
        benchmark_dataset["evaluation_partition"].eq("context_only")
    ]
    assert len(context) == 10
    assert context["prediction_time"].max() < pd.Timestamp("2023-01-01")
    assert context["appointment_id"].tolist() == list(range(1, 11))


def test_benchmark_current_feature_summaries_are_frozen(
    benchmark_dataset: pd.DataFrame,
) -> None:
    assert benchmark_dataset["reminder_sent_by_prediction_time"].sum() == 12_724
    assert benchmark_dataset["dentist_tenure_days"].min() == -1
    assert benchmark_dataset["booking_lead_time_hours"].min() >= 24.0


def test_benchmark_history_summaries_remain_aligned(
    benchmark_dataset: pd.DataFrame,
) -> None:
    assert benchmark_dataset["patient_history_available"].sum() == 18_781
    assert benchmark_dataset["dentist_no_show_rate_supported"].sum() == 21_670
    assert benchmark_dataset["visit_type_no_show_rate_supported"].sum() == 21_673
    assert (
        benchmark_dataset["weekday_hour_no_show_rate_supported"].sum()
        == 21_158
    )


def test_feature_dataset_excludes_target_and_source_outcomes(
    benchmark_dataset: pd.DataFrame,
) -> None:
    prohibited = {
        "target",
        "status",
        "status_updated_at",
        "reminder_sent",
        "reminder_sent_at",
        "check_in_at",
        "chair_start_at",
        "chair_end_at",
        "checkout_at",
    }
    assert set(benchmark_dataset.columns).isdisjoint(prohibited)
    final_test = benchmark_dataset.loc[
        benchmark_dataset["evaluation_partition"].eq("final_test")
    ]
    assert len(final_test) == 4_343
    assert "target" not in final_test.columns


def test_label_availability_matches_verified_source(
    benchmark_dataset: pd.DataFrame,
    benchmark_tables: V2RawTables,
) -> None:
    source = benchmark_tables.appointments.loc[
        :,
        ["appointment_id", "status_updated_at"],
    ].copy()
    source["status_updated_at"] = pd.to_datetime(
        source["status_updated_at"],
        format="mixed",
    ).astype("datetime64[ns]")
    joined = benchmark_dataset.loc[
        :,
        ["appointment_id", "label_available_at"],
    ].merge(
        source,
        on="appointment_id",
        validate="one_to_one",
    )
    assert str(joined["label_available_at"].dtype) == "datetime64[ns]"
    assert str(joined["status_updated_at"].dtype) == "datetime64[ns]"
    assert joined["label_available_at"].equals(joined["status_updated_at"])


def test_model_feature_selector_is_exact_and_defensive(
    benchmark_dataset: pd.DataFrame,
) -> None:
    selected = select_v2_model_features(benchmark_dataset)
    assert list(selected.columns) == list(V2_MODEL_FEATURE_COLUMNS)
    assert selected.shape == (21_755, 32)
    assert set(selected.columns).isdisjoint(V2_PROHIBITED_MODEL_COLUMNS)

    original = benchmark_dataset.loc[0, "planned_duration_min"]
    selected.loc[0, "planned_duration_min"] = 999
    assert benchmark_dataset.loc[0, "planned_duration_min"] == original


def test_model_feature_selector_rejects_missing_feature(
    benchmark_dataset: pd.DataFrame,
) -> None:
    broken = benchmark_dataset.drop(columns="scheduled_month")
    with pytest.raises(ValueError, match="missing approved"):
        select_v2_model_features(broken)


def test_full_builder_is_input_order_invariant(
    benchmark_tables: V2RawTables,
) -> None:
    config = load_benchmark_config(_CONFIG_PATH)
    expected = build_v2_feature_dataset(
        benchmark_tables,
        config=config,
    )
    shuffled = V2RawTables(
        appointments=benchmark_tables.appointments.sample(
            frac=1.0,
            random_state=101,
        ),
        patients=benchmark_tables.patients.sample(
            frac=1.0,
            random_state=102,
        ),
        dentists=benchmark_tables.dentists.sample(
            frac=1.0,
            random_state=103,
        ),
    )
    actual = build_v2_feature_dataset(shuffled, config=config)
    pd.testing.assert_frame_equal(actual, expected)
