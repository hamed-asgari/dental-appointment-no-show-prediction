"""Contract tests for strict-as-of Version 2 patient history features."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from src.features import (
    AGGREGATE_MIN_ATTENDANCE_SUPPORT,
    ATTENDANCE_STATUSES,
    AUDIT_KEY_COLUMNS,
    CURRENT_APPOINTMENT_FEATURE_COLUMNS,
    HISTORY_STATUSES,
    NO_SHOW_PRIOR_ALPHA,
    NO_SHOW_PRIOR_BETA,
    NO_SHOW_PRIOR_MEAN,
    NO_SHOW_PRIOR_STRENGTH,
    PATIENT_HISTORY_DTYPES,
    PATIENT_HISTORY_FEATURE_COLUMNS,
    PATIENT_HISTORY_OUTPUT_COLUMNS,
    PREDICTION_HORIZON_HOURS,
    V2_MODEL_FEATURE_COLUMNS,
    V2_PROHIBITED_MODEL_COLUMNS,
    build_eligible_scoring_rows,
    build_patient_history_features,
    normalize_history_appointments,
)


def _appointments(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["appointment_id"] = frame["appointment_id"].astype("int64")
    frame["patient_id"] = frame["patient_id"].astype("int64")
    frame["status"] = frame["status"].astype("string")
    for column in ("booked_at", "scheduled_start_at", "status_updated_at"):
        frame[column] = pd.to_datetime(
            frame[column],
            format="mixed",
        ).astype("datetime64[ns]")
    return frame


def _row(
    appointment_id: int,
    patient_id: int,
    *,
    booked_at: str,
    scheduled_start_at: str,
    status: str = "completed",
    status_updated_at: str,
) -> dict[str, object]:
    return {
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "booked_at": booked_at,
        "scheduled_start_at": scheduled_start_at,
        "status": status,
        "status_updated_at": status_updated_at,
    }


def _feature_row(features: pd.DataFrame, appointment_id: int) -> pd.Series:
    rows = features.loc[features["appointment_id"].eq(appointment_id)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_frozen_feature_schema_and_smoothing_constants() -> None:
    assert PREDICTION_HORIZON_HOURS == 24
    assert HISTORY_STATUSES == {
        "completed",
        "no_show",
        "cancelled",
        "rescheduled",
    }
    assert ATTENDANCE_STATUSES == {"completed", "no_show"}
    assert NO_SHOW_PRIOR_ALPHA == 1.0
    assert NO_SHOW_PRIOR_BETA == 9.0
    assert NO_SHOW_PRIOR_STRENGTH == 10.0
    assert NO_SHOW_PRIOR_MEAN == 0.1
    assert AGGREGATE_MIN_ATTENDANCE_SUPPORT == 10
    assert isinstance(PATIENT_HISTORY_DTYPES, MappingProxyType)
    assert len(CURRENT_APPOINTMENT_FEATURE_COLUMNS) == 11
    assert len(PATIENT_HISTORY_FEATURE_COLUMNS) == 12
    assert len(V2_MODEL_FEATURE_COLUMNS) == 32
    assert set(AUDIT_KEY_COLUMNS).isdisjoint(V2_MODEL_FEATURE_COLUMNS)
    assert set(V2_MODEL_FEATURE_COLUMNS).isdisjoint(V2_PROHIBITED_MODEL_COLUMNS)


def test_feature_dtype_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        PATIENT_HISTORY_DTYPES["appointment_id"] = "int32"  # type: ignore[index]


def test_normalization_requires_the_exact_minimum_source_fields() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            )
        ]
    ).drop(columns="status_updated_at")

    with pytest.raises(
        ValueError,
        match="missing required columns: status_updated_at",
    ):
        normalize_history_appointments(appointments)


def test_normalization_rejects_duplicate_ids_and_bad_statuses() -> None:
    duplicate = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                1,
                11,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 10:00:00",
                status_updated_at="2025-01-03 11:00:00",
            ),
        ]
    )
    with pytest.raises(ValueError, match="appointment_id must be unique"):
        normalize_history_appointments(duplicate)

    bad_status = duplicate.iloc[[0]].copy()
    bad_status["status"] = pd.Series(["unknown"], dtype="string")
    with pytest.raises(ValueError, match="unsupported values: unknown"):
        normalize_history_appointments(bad_status)


def test_eligibility_uses_booked_at_and_pre_prediction_inactive_status() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="completed",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-02 10:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="completed",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                3,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="cancelled",
                status_updated_at="2025-01-02 09:00:00",
            ),
            _row(
                4,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="cancelled",
                status_updated_at="2025-01-02 09:00:00.000000001",
            ),
        ]
    )

    scoring = build_eligible_scoring_rows(appointments)
    assert scoring["appointment_id"].tolist() == [1, 4]
    assert scoring["prediction_time"].eq(
        pd.Timestamp("2025-01-02 09:00:00")
    ).all()


def test_cold_start_defaults_and_output_dtypes_are_exact() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            )
        ]
    )

    features = build_patient_history_features(appointments)
    assert tuple(features.columns) == PATIENT_HISTORY_OUTPUT_COLUMNS
    assert {column: str(dtype) for column, dtype in features.dtypes.items()} == dict(
        PATIENT_HISTORY_DTYPES
    )

    row = features.iloc[0]
    assert not row["patient_history_available"]
    assert not row["patient_completed_history_available"]
    assert row["patient_prior_known_appointment_count"] == 0
    assert row["patient_prior_attendance_count"] == 0
    assert row["patient_prior_no_show_rate_smoothed"] == pytest.approx(0.1)
    assert row["patient_days_since_last_known_status_update"] == 0.0
    assert row["patient_days_since_last_completed_appointment"] == 0.0
    assert row["patient_mean_prior_booking_lead_days"] == 0.0


def test_event_one_nanosecond_before_prediction_is_available() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-25 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="no_show",
                status_updated_at="2025-01-02 08:59:59.999999999",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )

    row = _feature_row(build_patient_history_features(appointments), 2)
    assert row["patient_prior_known_appointment_count"] == 1
    assert row["patient_prior_no_show_count"] == 1
    assert row["patient_prior_attendance_count"] == 1


def test_event_exactly_at_prediction_is_unavailable() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-25 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="no_show",
                status_updated_at="2025-01-02 09:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )

    row = _feature_row(build_patient_history_features(appointments), 2)
    assert row["patient_prior_known_appointment_count"] == 0


def test_event_after_prediction_is_unavailable() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-25 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="no_show",
                status_updated_at="2025-01-02 09:00:00.000000001",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )

    row = _feature_row(build_patient_history_features(appointments), 2)
    assert row["patient_prior_known_appointment_count"] == 0


def test_current_row_cannot_contribute_to_its_own_features() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="no_show",
                status_updated_at="2025-01-03 09:30:00",
            )
        ]
    )

    row = _feature_row(build_patient_history_features(appointments), 1)
    assert row["patient_prior_known_appointment_count"] == 0
    assert row["patient_prior_no_show_count"] == 0


def test_rows_sharing_prediction_time_use_the_same_pre_time_state() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-20 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="completed",
                status_updated_at="2025-01-02 08:59:59",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                3,
                10,
                booked_at="2025-01-01 08:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="cancelled",
                status_updated_at="2025-01-02 09:00:00",
            ),
            _row(
                4,
                10,
                booked_at="2025-01-01 07:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="completed",
                status_updated_at="2025-01-03 11:00:00",
            ),
        ]
    )

    features = build_patient_history_features(appointments).set_index(
        "appointment_id"
    )
    assert features.loc[2, "patient_prior_known_appointment_count"] == 1
    assert features.loc[4, "patient_prior_known_appointment_count"] == 1
    assert features.loc[2, "patient_prior_completed_count"] == 1
    assert features.loc[4, "patient_prior_completed_count"] == 1


def test_input_row_order_does_not_change_output() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-20 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="completed",
                status_updated_at="2025-01-01 10:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="no_show",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                3,
                11,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-04 09:00:00",
                status="completed",
                status_updated_at="2025-01-04 10:00:00",
            ),
        ]
    )

    expected = build_patient_history_features(appointments)
    shuffled = appointments.sample(frac=1.0, random_state=42).reset_index(drop=True)
    actual = build_patient_history_features(shuffled)
    pd.testing.assert_frame_equal(actual, expected)


def test_mutating_future_outcome_does_not_change_earlier_features() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="completed",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-10 09:00:00",
                status="no_show",
                status_updated_at="2025-01-10 10:00:00",
            ),
        ]
    )
    before = _feature_row(build_patient_history_features(appointments), 1)

    mutated = appointments.copy(deep=True)
    mutated.loc[mutated["appointment_id"].eq(2), "status"] = "completed"
    after = _feature_row(build_patient_history_features(mutated), 1)

    pd.testing.assert_series_equal(after, before)


def test_mutating_equal_time_event_does_not_change_prediction_batch() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-20 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="no_show",
                status_updated_at="2025-01-02 09:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status="completed",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )
    before = _feature_row(build_patient_history_features(appointments), 2)

    mutated = appointments.copy(deep=True)
    mutated.loc[mutated["appointment_id"].eq(1), "status"] = "completed"
    after = _feature_row(build_patient_history_features(mutated), 2)

    pd.testing.assert_series_equal(after, before)


def test_status_counts_reconcile_and_cancelled_rescheduled_are_not_attendance() -> None:
    rows = [
        _row(
            1,
            10,
            booked_at="2024-12-01 09:00:00",
            scheduled_start_at="2024-12-20 09:00:00",
            status="completed",
            status_updated_at="2024-12-20 10:00:00",
        ),
        _row(
            2,
            10,
            booked_at="2024-12-01 09:00:00",
            scheduled_start_at="2024-12-21 09:00:00",
            status="no_show",
            status_updated_at="2024-12-21 10:00:00",
        ),
        _row(
            3,
            10,
            booked_at="2024-12-01 09:00:00",
            scheduled_start_at="2024-12-22 09:00:00",
            status="cancelled",
            status_updated_at="2024-12-22 10:00:00",
        ),
        _row(
            4,
            10,
            booked_at="2024-12-01 09:00:00",
            scheduled_start_at="2024-12-23 09:00:00",
            status="rescheduled",
            status_updated_at="2024-12-23 10:00:00",
        ),
        _row(
            5,
            10,
            booked_at="2024-12-20 09:00:00",
            scheduled_start_at="2025-01-03 09:00:00",
            status="completed",
            status_updated_at="2025-01-03 10:00:00",
        ),
    ]
    row = _feature_row(build_patient_history_features(_appointments(rows)), 5)

    status_sum = (
        row["patient_prior_completed_count"]
        + row["patient_prior_no_show_count"]
        + row["patient_prior_cancelled_count"]
        + row["patient_prior_rescheduled_count"]
    )
    assert status_sum == row["patient_prior_known_appointment_count"] == 4
    assert row["patient_prior_attendance_count"] == 2
    assert row["patient_prior_attendance_count"] == (
        row["patient_prior_completed_count"]
        + row["patient_prior_no_show_count"]
    )
    assert row["patient_prior_no_show_rate_smoothed"] == pytest.approx(2 / 12)


def test_recency_and_mean_booking_lead_use_strictly_available_events() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-20 09:00:00",
                scheduled_start_at="2024-12-30 09:00:00",
                status="completed",
                status_updated_at="2024-12-30 10:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2024-12-27 09:00:00",
                scheduled_start_at="2024-12-31 09:00:00",
                status="cancelled",
                status_updated_at="2024-12-31 10:00:00",
            ),
            _row(
                3,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 10:00:00",
                status="completed",
                status_updated_at="2025-01-03 11:00:00",
            ),
        ]
    )

    row = _feature_row(build_patient_history_features(appointments), 3)
    prediction = pd.Timestamp("2025-01-02 10:00:00")
    assert row["patient_days_since_last_known_status_update"] == pytest.approx(
        (prediction - pd.Timestamp("2024-12-31 10:00:00")).total_seconds()
        / 86_400
    )
    assert row["patient_days_since_last_completed_appointment"] == pytest.approx(
        (prediction - pd.Timestamp("2024-12-30 09:00:00")).total_seconds()
        / 86_400
    )
    assert row["patient_mean_prior_booking_lead_days"] == pytest.approx(7.0)


def test_latest_completed_recency_uses_latest_scheduled_appointment() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-20 09:00:00",
                scheduled_start_at="2025-01-01 09:00:00",
                status="completed",
                status_updated_at="2025-01-05 12:00:00",
            ),
            _row(
                2,
                10,
                booked_at="2024-12-21 09:00:00",
                scheduled_start_at="2025-01-02 09:00:00",
                status="completed",
                status_updated_at="2025-01-04 12:00:00",
            ),
            _row(
                3,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-07 12:00:00",
                status="completed",
                status_updated_at="2025-01-07 13:00:00",
            ),
        ]
    )

    row = _feature_row(build_patient_history_features(appointments), 3)
    prediction = pd.Timestamp("2025-01-06 12:00:00")
    assert row["patient_prior_completed_count"] == 2
    assert row["patient_days_since_last_completed_appointment"] == pytest.approx(
        (prediction - pd.Timestamp("2025-01-02 09:00:00")).total_seconds()
        / 86_400
    )


def test_builder_does_not_mutate_input() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            )
        ]
    )
    before = appointments.copy(deep=True)
    build_patient_history_features(appointments)
    pd.testing.assert_frame_equal(appointments, before)


def test_empty_eligible_cohort_returns_typed_empty_frame() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-02 10:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            )
        ]
    )
    features = build_patient_history_features(appointments)
    assert features.empty
    assert tuple(features.columns) == PATIENT_HISTORY_OUTPUT_COLUMNS
    assert {column: str(dtype) for column, dtype in features.dtypes.items()} == dict(
        PATIENT_HISTORY_DTYPES
    )


def test_frozen_benchmark_patient_history_summary() -> None:
    raw_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "v2"
        / "appointments.csv"
    )
    appointments = pd.read_csv(
        raw_path,
        dtype={
            "appointment_id": "int64",
            "patient_id": "int64",
            "status": "string",
        },
    )

    features = build_patient_history_features(appointments)

    assert len(features) == 21_755
    assert (
        features["prediction_time"] < pd.Timestamp("2023-01-01")
    ).sum() == 10
    assert features["patient_history_available"].sum() == 18_781
    assert features["patient_completed_history_available"].sum() == 18_055
    assert features["patient_prior_known_appointment_count"].max() == 57
    assert (
        features["patient_prior_attendance_count"]
        == features["patient_prior_completed_count"]
        + features["patient_prior_no_show_count"]
    ).all()
    assert (
        features["patient_prior_known_appointment_count"]
        == features["patient_prior_completed_count"]
        + features["patient_prior_no_show_count"]
        + features["patient_prior_cancelled_count"]
        + features["patient_prior_rescheduled_count"]
    ).all()
