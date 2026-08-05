"""Tests for the Version 2 longitudinal synthetic generator core."""

from __future__ import annotations

import heapq
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.synthetic.config import load_benchmark_config
from src.synthetic.generator import (
    generate_patients,
    generate_synthetic_tables,
)
from src.synthetic.schema import (
    APPOINTMENT_COLUMNS,
    DENTIST_COLUMNS,
    FORBIDDEN_EXPORTED_COLUMNS,
    PATIENT_COLUMNS,
    STATUSES,
)
from src.synthetic.tables import SyntheticTables
from src.synthetic.validation import validate_synthetic_tables


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_synthetic_benchmark.json"


@pytest.fixture(scope="module")
def benchmark_config():
    return load_benchmark_config(_CONFIG_PATH)


@pytest.fixture(scope="module")
def generated_tables(benchmark_config):
    return generate_synthetic_tables(benchmark_config)


def _eligible_at_prediction(
    appointments: pd.DataFrame,
    horizon_hours: int,
) -> pd.Series:
    prediction_time = (
        appointments["scheduled_start_at"]
        - pd.Timedelta(hours=horizon_hours)
    )
    inactive_before_prediction = (
        appointments["status"].isin(
            ["cancelled", "rescheduled"]
        )
        & appointments["status_updated_at"].le(
            prediction_time
        )
    )
    return (
        appointments["booked_at"].le(prediction_time)
        & ~inactive_before_prediction
    )


def _known_patient_history_counts(
    appointments: pd.DataFrame,
    horizon_hours: int,
) -> pd.Series:
    working = appointments.copy()
    working["prediction_time"] = (
        working["scheduled_start_at"]
        - pd.Timedelta(hours=horizon_hours)
    )
    working = working.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
    )

    events = sorted(
        zip(
            appointments["status_updated_at"],
            appointments["patient_id"],
            strict=True,
        ),
        key=lambda item: item[0],
    )
    counts: dict[int, int] = {}
    event_index = 0
    observed: dict[int, int] = {}

    for row in working.itertuples(index=False):
        while (
            event_index < len(events)
            and events[event_index][0] < row.prediction_time
        ):
            _, patient_id = events[event_index]
            counts[int(patient_id)] = (
                counts.get(int(patient_id), 0) + 1
            )
            event_index += 1
        observed[int(row.appointment_id)] = counts.get(
            int(row.patient_id),
            0,
        )

    return appointments["appointment_id"].map(observed)


def test_full_generator_is_deterministic(
    benchmark_config,
    generated_tables,
) -> None:
    repeated = generate_synthetic_tables(benchmark_config)

    pd.testing.assert_frame_equal(
        generated_tables.patients,
        repeated.patients,
    )
    pd.testing.assert_frame_equal(
        generated_tables.dentists,
        repeated.dentists,
    )
    pd.testing.assert_frame_equal(
        generated_tables.appointments,
        repeated.appointments,
    )


def test_public_table_shapes_and_columns_match_frozen_contract(
    benchmark_config,
    generated_tables,
) -> None:
    assert generated_tables.patients.shape == (
        benchmark_config.patient_count,
        len(PATIENT_COLUMNS),
    )
    assert generated_tables.dentists.shape == (
        benchmark_config.dentist_count,
        len(DENTIST_COLUMNS),
    )
    assert generated_tables.appointments.shape == (
        benchmark_config.appointment_count,
        len(APPOINTMENT_COLUMNS),
    )
    assert tuple(generated_tables.patients) == PATIENT_COLUMNS
    assert tuple(generated_tables.dentists) == DENTIST_COLUMNS
    assert tuple(generated_tables.appointments) == APPOINTMENT_COLUMNS


def test_hidden_latent_state_is_never_exported(
    generated_tables,
) -> None:
    for frame in (
        generated_tables.patients,
        generated_tables.dentists,
        generated_tables.appointments,
    ):
        assert FORBIDDEN_EXPORTED_COLUMNS.isdisjoint(
            frame.columns
        )


def test_patient_stream_is_isolated_from_appointment_count(
    benchmark_config,
) -> None:
    changed = replace(
        benchmark_config,
        appointment_count=100,
    )

    pd.testing.assert_frame_equal(
        generate_patients(benchmark_config),
        generate_patients(changed),
    )


def test_patient_registration_spans_multiple_calendar_years(
    generated_tables,
) -> None:
    years = set(
        generated_tables.patients[
            "registered_at"
        ].dt.year
    )

    assert {2022, 2023, 2024, 2025, 2026, 2027} <= years
    assert (
        generated_tables.patients[
            "patient_status"
        ].eq("active").any()
    )


def test_keys_foreign_keys_and_booking_eligibility_are_valid(
    generated_tables,
) -> None:
    patients = generated_tables.patients
    dentists = generated_tables.dentists
    appointments = generated_tables.appointments

    assert patients["patient_id"].is_unique
    assert dentists["dentist_id"].is_unique
    assert appointments["appointment_id"].is_unique
    assert set(appointments["patient_id"]) <= set(
        patients["patient_id"]
    )
    assert set(appointments["dentist_id"]) <= set(
        dentists["dentist_id"]
    )

    registered_at = patients.set_index(
        "patient_id"
    )["registered_at"]
    assert appointments["patient_id"].map(
        registered_at
    ).le(appointments["booked_at"]).all()


def test_schedule_and_dentist_availability_contracts_hold(
    benchmark_config,
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    dentists = generated_tables.dentists.set_index(
        "dentist_id"
    )
    scheduled = appointments["scheduled_start_at"]

    assert scheduled.is_monotonic_increasing
    assert scheduled.min().date() >= benchmark_config.appointment_start
    assert scheduled.max().date() <= benchmark_config.appointment_end
    assert not scheduled.dt.dayofweek.eq(4).any()

    starts = appointments["dentist_id"].map(
        dentists["start_date"]
    )
    ends = appointments["dentist_id"].map(
        dentists["end_date"]
    )
    assert starts.le(scheduled).all()
    assert not (
        ends.notna()
        & ends.lt(scheduled)
    ).any()


def test_reminder_timing_is_explicit_and_prediction_time_safe(
    benchmark_config,
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    reminder_time = appointments["reminder_sent_at"]
    prediction_time = (
        appointments["scheduled_start_at"]
        - pd.Timedelta(
            hours=benchmark_config.prediction_horizon_hours
        )
    )

    assert reminder_time.notna().eq(
        appointments["reminder_sent"]
    ).all()
    assert (
        reminder_time.dropna()
        >= appointments.loc[
            reminder_time.notna(),
            "booked_at",
        ]
    ).all()
    assert (
        reminder_time.dropna()
        < appointments.loc[
            reminder_time.notna(),
            "scheduled_start_at",
        ]
    ).all()

    known = reminder_time.notna() & reminder_time.le(
        prediction_time
    )
    after_prediction = (
        reminder_time.notna()
        & reminder_time.gt(prediction_time)
    )
    assert known.any()
    assert after_prediction.any()


def test_status_specific_timestamp_semantics_hold(
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    completed = appointments["status"].eq("completed")
    no_show = appointments["status"].eq("no_show")
    changed = appointments["status"].isin(
        ["cancelled", "rescheduled"]
    )
    workflow = [
        "check_in_at",
        "chair_start_at",
        "chair_end_at",
        "checkout_at",
    ]

    assert appointments.loc[
        completed,
        workflow,
    ].notna().all().all()
    assert appointments.loc[
        ~completed,
        workflow,
    ].isna().all().all()
    assert appointments.loc[
        completed,
        "status_updated_at",
    ].eq(
        appointments.loc[
            completed,
            "checkout_at",
        ]
    ).all()
    assert appointments.loc[
        no_show,
        "status_updated_at",
    ].eq(
        appointments.loc[
            no_show,
            "scheduled_start_at",
        ]
        + pd.Timedelta(minutes=15)
    ).all()
    assert appointments.loc[
        changed,
        "status_updated_at",
    ].lt(
        appointments.loc[
            changed,
            "scheduled_start_at",
        ]
    ).all()


def test_all_statuses_are_present_without_performance_targeting(
    generated_tables,
) -> None:
    counts = generated_tables.appointments[
        "status"
    ].value_counts()

    assert set(counts.index) == set(STATUSES)
    assert counts.min() >= 500


def test_prediction_cohort_contains_required_operational_cases(
    benchmark_config,
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    prediction_time = (
        appointments["scheduled_start_at"]
        - pd.Timedelta(
            hours=benchmark_config.prediction_horizon_hours
        )
    )
    eligible = _eligible_at_prediction(
        appointments,
        benchmark_config.prediction_horizon_hours,
    )
    changed = appointments["status"].isin(
        ["cancelled", "rescheduled"]
    )

    assert eligible.sum() >= 20_000
    assert appointments.loc[
        eligible,
        "status",
    ].eq("no_show").any()
    assert appointments.loc[
        eligible,
        "status",
    ].eq("completed").any()
    assert (
        changed
        & appointments["status_updated_at"].le(
            prediction_time
        )
    ).any()
    assert (
        changed
        & appointments["status_updated_at"].gt(
            prediction_time
        )
    ).any()


def test_cold_start_and_known_repeat_history_both_exist(
    benchmark_config,
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    eligible = _eligible_at_prediction(
        appointments,
        benchmark_config.prediction_horizon_hours,
    )
    known_counts = _known_patient_history_counts(
        appointments,
        benchmark_config.prediction_horizon_hours,
    )

    assert known_counts.loc[eligible].eq(0).any()
    assert known_counts.loc[eligible].gt(0).any()


def test_each_protected_evaluation_period_has_substantial_rows(
    benchmark_config,
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    prediction_time = (
        appointments["scheduled_start_at"]
        - pd.Timedelta(
            hours=benchmark_config.prediction_horizon_hours
        )
    )
    schedule = benchmark_config.evaluation
    windows = [
        schedule.warmup,
        *(
            fold.validation
            for fold in schedule.rolling_folds
        ),
        schedule.calibration,
        schedule.policy_selection,
        schedule.final_test,
    ]

    for window in windows:
        mask = (
            prediction_time.ge(pd.Timestamp(window.start))
            & prediction_time.lt(pd.Timestamp(window.end))
        )
        assert mask.sum() >= 1_000


def test_reschedule_links_reference_valid_later_same_patient_rows(
    generated_tables,
) -> None:
    appointments = generated_tables.appointments
    linked = appointments[
        "rescheduled_from_appointment_id"
    ].notna()

    assert linked.sum() >= 500

    parents = appointments.set_index(
        "appointment_id"
    ).loc[
        appointments.loc[
            linked,
            "rescheduled_from_appointment_id",
        ].astype("int64")
    ].reset_index(drop=True)
    children = appointments.loc[
        linked
    ].reset_index(drop=True)

    assert parents["status"].eq("rescheduled").all()
    assert parents["patient_id"].eq(
        children["patient_id"]
    ).all()
    assert parents["scheduled_start_at"].lt(
        children["scheduled_start_at"]
    ).all()
    assert parents["status_updated_at"].le(
        children["booked_at"]
    ).all()


def test_validator_rejects_unknown_foreign_key(
    benchmark_config,
    generated_tables,
) -> None:
    tampered = generated_tables.appointments.copy()
    tampered.loc[0, "patient_id"] = (
        benchmark_config.patient_count + 1
    )

    with pytest.raises(
        ValueError,
        match="unknown values",
    ):
        validate_synthetic_tables(
            SyntheticTables(
                patients=generated_tables.patients,
                dentists=generated_tables.dentists,
                appointments=tampered,
            ),
            benchmark_config,
        )


def test_validator_rejects_reminder_boolean_timestamp_mismatch(
    benchmark_config,
    generated_tables,
) -> None:
    tampered = generated_tables.appointments.copy()
    reminder_index = tampered.index[
        tampered["reminder_sent"]
    ][0]
    tampered.loc[
        reminder_index,
        "reminder_sent_at",
    ] = pd.NaT

    with pytest.raises(
        ValueError,
        match="reminder_sent",
    ):
        validate_synthetic_tables(
            SyntheticTables(
                patients=generated_tables.patients,
                dentists=generated_tables.dentists,
                appointments=tampered,
            ),
            benchmark_config,
        )
