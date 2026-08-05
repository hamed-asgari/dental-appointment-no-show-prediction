"""Contract tests for strict-as-of Version 2 aggregate history features."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from src.features import (
    AGGREGATE_HISTORY_DTYPES,
    AGGREGATE_HISTORY_OUTPUT_COLUMNS,
    AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    AGGREGATE_MIN_ATTENDANCE_SUPPORT,
    NO_SHOW_PRIOR_MEAN,
    build_aggregate_history_features,
    normalize_aggregate_history_appointments,
)


def _appointments(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("appointment_id", "patient_id", "dentist_id"):
        frame[column] = frame[column].astype("int64")
    for column in ("status", "visit_type"):
        frame[column] = frame[column].astype("string")
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
    dentist_id: int = 1,
    visit_type: str = "treatment",
    booked_at: str,
    scheduled_start_at: str,
    status: str = "completed",
    status_updated_at: str,
) -> dict[str, object]:
    return {
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "dentist_id": dentist_id,
        "visit_type": visit_type,
        "booked_at": booked_at,
        "scheduled_start_at": scheduled_start_at,
        "status": status,
        "status_updated_at": status_updated_at,
    }


def _feature_row(features: pd.DataFrame, appointment_id: int) -> pd.Series:
    rows = features.loc[features["appointment_id"].eq(appointment_id)]
    assert len(rows) == 1
    return rows.iloc[0]


def _ten_attendance_rows(
    *,
    no_show_count: int = 2,
    dentist_id: int = 1,
    visit_type: str = "treatment",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    first_start = pd.Timestamp("2025-01-06 09:00:00")
    for offset in range(10):
        scheduled = first_start + pd.Timedelta(days=7 * offset)
        status = "no_show" if offset < no_show_count else "completed"
        rows.append(
            _row(
                offset + 1,
                100 + offset,
                dentist_id=dentist_id,
                visit_type=visit_type,
                booked_at=str(scheduled - pd.Timedelta(days=2)),
                scheduled_start_at=str(scheduled),
                status=status,
                status_updated_at=str(scheduled + pd.Timedelta(hours=1)),
            )
        )
    return rows


def test_aggregate_schema_and_dtype_mapping_are_frozen() -> None:
    assert AGGREGATE_MIN_ATTENDANCE_SUPPORT == 10
    assert NO_SHOW_PRIOR_MEAN == 0.1
    assert isinstance(AGGREGATE_HISTORY_DTYPES, MappingProxyType)
    assert len(AGGREGATE_HISTORY_OUTPUT_COLUMNS) == 12
    assert AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS[-2:] == (
        "dentist_id",
        "visit_type",
    )
    with pytest.raises(TypeError):
        AGGREGATE_HISTORY_DTYPES["appointment_id"] = "int32"  # type: ignore[index]


def test_aggregate_normalization_requires_group_fields() -> None:
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
    ).drop(columns="dentist_id")

    with pytest.raises(
        ValueError,
        match="required aggregate-history columns: dentist_id",
    ):
        normalize_aggregate_history_appointments(appointments)


def test_aggregate_normalization_rejects_invalid_group_values() -> None:
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
    appointments["dentist_id"] = 1.5
    with pytest.raises(ValueError, match="integer identifiers"):
        normalize_aggregate_history_appointments(appointments)

    appointments = _appointments(
        [
            _row(
                1,
                10,
                visit_type=" ",
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            )
        ]
    )
    with pytest.raises(ValueError, match="empty values"):
        normalize_aggregate_history_appointments(appointments)


def test_aggregate_cold_start_defaults_and_dtypes_are_exact() -> None:
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

    features = build_aggregate_history_features(appointments)
    row = _feature_row(features, 1)

    for prefix in ("dentist", "visit_type", "weekday_hour"):
        assert row[f"{prefix}_prior_attendance_count"] == 0
        assert not row[f"{prefix}_no_show_rate_supported"]
        assert row[f"{prefix}_prior_no_show_rate_smoothed"] == 0.1

    assert tuple(features.columns) == AGGREGATE_HISTORY_OUTPUT_COLUMNS
    assert {column: str(dtype) for column, dtype in features.dtypes.items()} == dict(
        AGGREGATE_HISTORY_DTYPES
    )


def test_event_one_nanosecond_before_prediction_is_available() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-01 09:00:00",
                scheduled_start_at="2024-12-27 09:00:00",
                status_updated_at="2025-01-02 08:59:59.999999999",
            ),
            _row(
                2,
                11,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )

    row = _feature_row(build_aggregate_history_features(appointments), 2)
    assert row["dentist_prior_attendance_count"] == 1
    assert row["visit_type_prior_attendance_count"] == 1
    assert row["weekday_hour_prior_attendance_count"] == 1


def test_event_exactly_at_prediction_is_unavailable() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-01 09:00:00",
                scheduled_start_at="2024-12-27 09:00:00",
                status_updated_at="2025-01-02 09:00:00",
            ),
            _row(
                2,
                11,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )

    row = _feature_row(build_aggregate_history_features(appointments), 2)
    assert row["dentist_prior_attendance_count"] == 0
    assert row["visit_type_prior_attendance_count"] == 0
    assert row["weekday_hour_prior_attendance_count"] == 0


def test_event_after_prediction_is_unavailable() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-01 09:00:00",
                scheduled_start_at="2024-12-27 09:00:00",
                status_updated_at="2025-01-02 09:00:00.000000001",
            ),
            _row(
                2,
                11,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )

    row = _feature_row(build_aggregate_history_features(appointments), 2)
    assert row["dentist_prior_attendance_count"] == 0


def test_support_switches_only_at_ten_attendance_opportunities() -> None:
    rows = _ten_attendance_rows(no_show_count=0)
    rows[-1]["status_updated_at"] = "2025-03-16 09:00:00"
    rows.extend(
        [
            _row(
                11,
                211,
                booked_at="2025-03-10 09:00:00",
                scheduled_start_at="2025-03-17 09:00:00",
                status="cancelled",
                status_updated_at="2025-03-17 10:00:00",
            ),
            _row(
                12,
                212,
                booked_at="2025-03-17 09:00:00",
                scheduled_start_at="2025-03-24 09:00:00",
                status_updated_at="2025-03-24 10:00:00",
            ),
        ]
    )

    features = build_aggregate_history_features(_appointments(rows))
    unsupported = _feature_row(features, 11)
    supported = _feature_row(features, 12)

    for prefix in ("dentist", "visit_type", "weekday_hour"):
        assert unsupported[f"{prefix}_prior_attendance_count"] == 9
        assert not unsupported[f"{prefix}_no_show_rate_supported"]
        assert unsupported[f"{prefix}_prior_no_show_rate_smoothed"] == 0.1

        assert supported[f"{prefix}_prior_attendance_count"] == 10
        assert supported[f"{prefix}_no_show_rate_supported"]
        assert supported[f"{prefix}_prior_no_show_rate_smoothed"] == pytest.approx(
            1 / 20
        )


def test_supported_rates_use_frozen_beta_smoothing() -> None:
    rows = _ten_attendance_rows(no_show_count=2)
    rows.append(
        _row(
            11,
            211,
            booked_at="2025-03-10 09:00:00",
            scheduled_start_at="2025-03-17 09:00:00",
            status_updated_at="2025-03-17 10:00:00",
        )
    )

    row = _feature_row(
        build_aggregate_history_features(_appointments(rows)),
        11,
    )
    for prefix in ("dentist", "visit_type", "weekday_hour"):
        assert row[f"{prefix}_prior_attendance_count"] == 10
        assert row[f"{prefix}_no_show_rate_supported"]
        assert row[f"{prefix}_prior_no_show_rate_smoothed"] == pytest.approx(
            3 / 20
        )


def test_cancelled_and_rescheduled_events_do_not_enter_denominators() -> None:
    rows = [
        _row(
            index + 1,
            100 + index,
            booked_at=f"2025-01-{index + 1:02d} 07:00:00",
            scheduled_start_at=f"2025-01-{index + 1:02d} 09:00:00",
            status="cancelled" if index % 2 == 0 else "rescheduled",
            status_updated_at=f"2025-01-{index + 1:02d} 08:00:00",
        )
        for index in range(10)
    ]
    rows.append(
        _row(
            11,
            211,
            booked_at="2025-03-10 09:00:00",
            scheduled_start_at="2025-03-17 09:00:00",
            status_updated_at="2025-03-17 10:00:00",
        )
    )

    row = _feature_row(
        build_aggregate_history_features(_appointments(rows)),
        11,
    )
    for prefix in ("dentist", "visit_type", "weekday_hour"):
        assert row[f"{prefix}_prior_attendance_count"] == 0
        assert not row[f"{prefix}_no_show_rate_supported"]


def test_group_states_are_isolated_by_dentist_visit_type_and_weekday_hour() -> None:
    history = _ten_attendance_rows(
        no_show_count=2,
        dentist_id=1,
        visit_type="treatment",
    )
    history.extend(
        [
            _row(
                11,
                211,
                dentist_id=2,
                visit_type="treatment",
                booked_at="2025-03-10 09:00:00",
                scheduled_start_at="2025-03-17 09:00:00",
                status_updated_at="2025-03-17 10:00:00",
            ),
            _row(
                12,
                212,
                dentist_id=1,
                visit_type="consultation",
                booked_at="2025-03-10 09:00:00",
                scheduled_start_at="2025-03-17 09:00:00",
                status_updated_at="2025-03-17 10:00:00",
            ),
            _row(
                13,
                213,
                dentist_id=1,
                visit_type="treatment",
                booked_at="2025-03-10 09:00:00",
                scheduled_start_at="2025-03-17 10:00:00",
                status_updated_at="2025-03-17 11:00:00",
            ),
        ]
    )
    features = build_aggregate_history_features(_appointments(history))

    dentist_isolated = _feature_row(features, 11)
    assert dentist_isolated["dentist_prior_attendance_count"] == 0
    assert dentist_isolated["visit_type_prior_attendance_count"] == 10
    assert dentist_isolated["weekday_hour_prior_attendance_count"] == 10

    visit_isolated = _feature_row(features, 12)
    assert visit_isolated["dentist_prior_attendance_count"] == 10
    assert visit_isolated["visit_type_prior_attendance_count"] == 0
    assert visit_isolated["weekday_hour_prior_attendance_count"] == 10

    hour_isolated = _feature_row(features, 13)
    assert hour_isolated["dentist_prior_attendance_count"] == 10
    assert hour_isolated["visit_type_prior_attendance_count"] == 10
    assert hour_isolated["weekday_hour_prior_attendance_count"] == 0


def test_rows_sharing_prediction_time_use_same_pre_time_state() -> None:
    rows = _ten_attendance_rows(no_show_count=2)
    rows.extend(
        [
            _row(
                11,
                211,
                booked_at="2025-03-10 09:00:00",
                scheduled_start_at="2025-03-17 09:00:00",
                status="no_show",
                status_updated_at="2025-03-17 10:00:00",
            ),
            _row(
                12,
                212,
                booked_at="2025-03-10 09:00:00",
                scheduled_start_at="2025-03-17 09:00:00",
                status="completed",
                status_updated_at="2025-03-17 11:00:00",
            ),
        ]
    )

    features = build_aggregate_history_features(_appointments(rows))
    left = _feature_row(features, 11)
    right = _feature_row(features, 12)
    for column in AGGREGATE_HISTORY_OUTPUT_COLUMNS[3:]:
        assert left[column] == right[column]


def test_input_row_order_does_not_change_aggregate_output() -> None:
    rows = _ten_attendance_rows(no_show_count=2)
    rows.append(
        _row(
            11,
            211,
            booked_at="2025-01-10 09:00:00",
            scheduled_start_at="2025-01-20 09:00:00",
            status_updated_at="2025-01-20 10:00:00",
        )
    )
    appointments = _appointments(rows)

    expected = build_aggregate_history_features(appointments)
    shuffled = appointments.sample(frac=1.0, random_state=91).reset_index(drop=True)
    actual = build_aggregate_history_features(shuffled)

    pd.testing.assert_frame_equal(actual, expected)


def test_mutating_future_outcome_does_not_change_earlier_aggregate_features() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
            _row(
                2,
                11,
                booked_at="2025-01-02 09:00:00",
                scheduled_start_at="2025-01-04 09:00:00",
                status="completed",
                status_updated_at="2025-01-04 10:00:00",
            ),
        ]
    )
    expected = _feature_row(build_aggregate_history_features(appointments), 1)

    mutated = appointments.copy(deep=True)
    mutated.loc[mutated["appointment_id"].eq(2), "status"] = "no_show"
    actual = _feature_row(build_aggregate_history_features(mutated), 1)

    pd.testing.assert_series_equal(actual, expected)


def test_mutating_equal_time_event_does_not_change_aggregate_batch() -> None:
    appointments = _appointments(
        [
            _row(
                1,
                10,
                booked_at="2024-12-01 09:00:00",
                scheduled_start_at="2024-12-27 09:00:00",
                status="completed",
                status_updated_at="2025-01-02 09:00:00",
            ),
            _row(
                2,
                11,
                booked_at="2025-01-01 09:00:00",
                scheduled_start_at="2025-01-03 09:00:00",
                status_updated_at="2025-01-03 10:00:00",
            ),
        ]
    )
    expected = _feature_row(build_aggregate_history_features(appointments), 2)

    mutated = appointments.copy(deep=True)
    mutated.loc[mutated["appointment_id"].eq(1), "status"] = "no_show"
    actual = _feature_row(build_aggregate_history_features(mutated), 2)

    pd.testing.assert_series_equal(actual, expected)


def test_aggregate_builder_does_not_mutate_input() -> None:
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
    build_aggregate_history_features(appointments)
    pd.testing.assert_frame_equal(appointments, before)


def test_empty_eligible_cohort_returns_typed_aggregate_frame() -> None:
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
    features = build_aggregate_history_features(appointments)
    assert features.empty
    assert tuple(features.columns) == AGGREGATE_HISTORY_OUTPUT_COLUMNS
    assert {column: str(dtype) for column, dtype in features.dtypes.items()} == dict(
        AGGREGATE_HISTORY_DTYPES
    )


def test_frozen_benchmark_aggregate_history_summary() -> None:
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
            "dentist_id": "int64",
            "status": "string",
            "visit_type": "string",
        },
    )

    features = build_aggregate_history_features(appointments)

    assert len(features) == 21_755
    assert (
        features["prediction_time"] < pd.Timestamp("2023-01-01")
    ).sum() == 10
    assert features["dentist_no_show_rate_supported"].sum() == 21_670
    assert features["visit_type_no_show_rate_supported"].sum() == 21_673
    assert features["weekday_hour_no_show_rate_supported"].sum() == 21_158
    assert features["dentist_prior_attendance_count"].max() == 7_398
    assert features["visit_type_prior_attendance_count"].max() == 7_890
    assert features["weekday_hour_prior_attendance_count"].max() == 508

    for prefix in ("dentist", "visit_type", "weekday_hour"):
        unsupported = ~features[f"{prefix}_no_show_rate_supported"]
        assert (
            features.loc[
                unsupported,
                f"{prefix}_prior_no_show_rate_smoothed",
            ]
            == 0.1
        ).all()
