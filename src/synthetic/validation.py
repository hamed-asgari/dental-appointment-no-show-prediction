"""Validate Version 2 longitudinal synthetic raw-table contracts."""

from __future__ import annotations

import pandas as pd

from src.synthetic.config import BenchmarkConfig
from src.synthetic.schema import (
    APPOINTMENT_COLUMNS,
    BOOKING_CHANNELS,
    DENTIST_COLUMNS,
    FORBIDDEN_EXPORTED_COLUMNS,
    PATIENT_COLUMNS,
    STATUSES,
    VISIT_TYPES,
)
from src.synthetic.tables import SyntheticTables


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    observed = tuple(frame.columns)
    if observed != expected:
        raise ValueError(
            f"{table_name} columns must match the frozen contract; "
            f"expected={expected}, observed={observed}"
        )


def _require_primary_key(
    frame: pd.DataFrame,
    key: str,
    *,
    table_name: str,
) -> None:
    if frame[key].isna().any():
        raise ValueError(
            f"{table_name}.{key} must not be missing"
        )
    if not frame[key].is_unique:
        raise ValueError(
            f"{table_name}.{key} must be unique"
        )


def validate_synthetic_tables(
    tables: SyntheticTables,
    config: BenchmarkConfig,
) -> None:
    """Validate generated table schemas, keys, timestamps, and relationships."""

    patients = tables.patients
    dentists = tables.dentists
    appointments = tables.appointments

    _require_exact_columns(
        patients,
        PATIENT_COLUMNS,
        table_name="patients",
    )
    _require_exact_columns(
        dentists,
        DENTIST_COLUMNS,
        table_name="dentists",
    )
    _require_exact_columns(
        appointments,
        APPOINTMENT_COLUMNS,
        table_name="appointments",
    )

    expected_rows = {
        "patients": config.patient_count,
        "dentists": config.dentist_count,
        "appointments": config.appointment_count,
    }
    for name, frame in (
        ("patients", patients),
        ("dentists", dentists),
        ("appointments", appointments),
    ):
        if len(frame) != expected_rows[name]:
            raise ValueError(
                f"{name} row count must be "
                f"{expected_rows[name]:,}; got {len(frame):,}"
            )

    _require_primary_key(
        patients,
        "patient_id",
        table_name="patients",
    )
    _require_primary_key(
        dentists,
        "dentist_id",
        table_name="dentists",
    )
    _require_primary_key(
        appointments,
        "appointment_id",
        table_name="appointments",
    )

    if not set(
        appointments["patient_id"]
    ).issubset(set(patients["patient_id"])):
        raise ValueError(
            "appointments.patient_id contains unknown values"
        )
    if not set(
        appointments["dentist_id"]
    ).issubset(set(dentists["dentist_id"])):
        raise ValueError(
            "appointments.dentist_id contains unknown values"
        )

    if not appointments[
        "scheduled_start_at"
    ].is_monotonic_increasing:
        raise ValueError(
            "appointments must be sorted by scheduled_start_at"
        )

    scheduled = appointments["scheduled_start_at"]
    if scheduled.min().date() < config.appointment_start:
        raise ValueError(
            "appointments begin before appointment_start"
        )
    if scheduled.max().date() > config.appointment_end:
        raise ValueError(
            "appointments end after appointment_end"
        )
    if scheduled.dt.dayofweek.eq(4).any():
        raise ValueError(
            "appointments must not be scheduled on Friday"
        )
    if not appointments["booked_at"].lt(scheduled).all():
        raise ValueError(
            "every appointment must be booked before it starts"
        )

    patient_registration = patients.set_index(
        "patient_id"
    )["registered_at"]
    joined_registration = appointments[
        "patient_id"
    ].map(patient_registration)
    if not joined_registration.le(
        appointments["booked_at"]
    ).all():
        raise ValueError(
            "patients must be registered by booking time"
        )

    dentist_lookup = dentists.set_index("dentist_id")
    dentist_start = appointments["dentist_id"].map(
        dentist_lookup["start_date"]
    )
    dentist_end = appointments["dentist_id"].map(
        dentist_lookup["end_date"]
    )
    if not dentist_start.le(scheduled).all():
        raise ValueError(
            "dentists must have started by appointment time"
        )
    invalid_end = dentist_end.notna() & dentist_end.lt(
        scheduled
    )
    if invalid_end.any():
        raise ValueError(
            "appointments cannot use dentists after end_date"
        )

    if not set(appointments["visit_type"]).issubset(
        set(VISIT_TYPES)
    ):
        raise ValueError(
            "appointments contain an unknown visit_type"
        )
    if not set(appointments["booking_channel"]).issubset(
        set(BOOKING_CHANNELS)
    ):
        raise ValueError(
            "appointments contain an unknown booking_channel"
        )
    if not set(appointments["status"]).issubset(
        set(STATUSES)
    ):
        raise ValueError(
            "appointments contain an unknown status"
        )

    reminder_sent = appointments["reminder_sent"]
    reminder_time = appointments["reminder_sent_at"]
    if not reminder_time.notna().eq(reminder_sent).all():
        raise ValueError(
            "reminder_sent must match reminder_sent_at availability"
        )
    if (
        reminder_time.notna()
        & reminder_time.lt(appointments["booked_at"])
    ).any():
        raise ValueError(
            "reminders cannot be sent before booking"
        )
    if (
        reminder_time.notna()
        & reminder_time.ge(scheduled)
    ).any():
        raise ValueError(
            "reminders must be sent before appointment time"
        )

    if appointments["status_updated_at"].isna().any():
        raise ValueError(
            "status_updated_at must be complete"
        )

    completed = appointments["status"].eq("completed")
    completed_columns = (
        "check_in_at",
        "chair_start_at",
        "chair_end_at",
        "checkout_at",
    )
    if appointments.loc[
        completed,
        completed_columns,
    ].isna().any().any():
        raise ValueError(
            "completed appointments require all workflow timestamps"
        )
    non_completed = ~completed
    if appointments.loc[
        non_completed,
        completed_columns,
    ].notna().any().any():
        raise ValueError(
            "non-completed appointments must not have workflow timestamps"
        )
    if not (
        appointments.loc[
            completed,
            "check_in_at",
        ]
        <= appointments.loc[
            completed,
            "chair_start_at",
        ]
    ).all():
        raise ValueError(
            "check_in_at must not follow chair_start_at"
        )
    if not (
        appointments.loc[
            completed,
            "chair_start_at",
        ]
        < appointments.loc[
            completed,
            "chair_end_at",
        ]
    ).all():
        raise ValueError(
            "chair_start_at must precede chair_end_at"
        )
    if not (
        appointments.loc[
            completed,
            "chair_end_at",
        ]
        < appointments.loc[
            completed,
            "checkout_at",
        ]
    ).all():
        raise ValueError(
            "chair_end_at must precede checkout_at"
        )
    if not appointments.loc[
        completed,
        "status_updated_at",
    ].eq(
        appointments.loc[
            completed,
            "checkout_at",
        ]
    ).all():
        raise ValueError(
            "completed status_updated_at must equal checkout_at"
        )

    no_show = appointments["status"].eq("no_show")
    expected_no_show_update = (
        appointments.loc[
            no_show,
            "scheduled_start_at",
        ]
        + pd.Timedelta(minutes=15)
    )
    if not appointments.loc[
        no_show,
        "status_updated_at",
    ].eq(expected_no_show_update).all():
        raise ValueError(
            "no-show status must update 15 minutes after start"
        )

    changed = appointments["status"].isin(
        ["cancelled", "rescheduled"]
    )
    if appointments.loc[
        changed,
        "status_change_reason",
    ].isna().any():
        raise ValueError(
            "cancelled and rescheduled rows need a reason"
        )
    if appointments.loc[
        ~changed,
        "status_change_reason",
    ].notna().any():
        raise ValueError(
            "unchanged attendance rows must not have a reason"
        )
    if not appointments.loc[
        changed,
        "status_updated_at",
    ].lt(
        appointments.loc[
            changed,
            "scheduled_start_at",
        ]
    ).all():
        raise ValueError(
            "cancellation and reschedule updates must precede start"
        )
    if not appointments.loc[
        changed,
        "status_updated_at",
    ].gt(
        appointments.loc[
            changed,
            "booked_at",
        ]
    ).all():
        raise ValueError(
            "cancellation and reschedule updates must follow booking"
        )

    links = appointments[
        "rescheduled_from_appointment_id"
    ]
    linked = links.notna()
    if linked.any():
        parent_ids = links.loc[linked].astype("int64")
        if not parent_ids.isin(
            appointments["appointment_id"]
        ).all():
            raise ValueError(
                "reschedule links must reference valid appointments"
            )
        parent_lookup = appointments.set_index(
            "appointment_id"
        )
        parents = parent_lookup.loc[
            parent_ids.to_numpy()
        ].reset_index(drop=True)
        children = appointments.loc[
            linked
        ].reset_index(drop=True)
        if not parents["status"].eq(
            "rescheduled"
        ).all():
            raise ValueError(
                "reschedule links must point to rescheduled rows"
            )
        if not parents["patient_id"].eq(
            children["patient_id"]
        ).all():
            raise ValueError(
                "reschedule parent and child must share patient_id"
            )
        if not parents["scheduled_start_at"].lt(
            children["scheduled_start_at"]
        ).all():
            raise ValueError(
                "reschedule child must occur after its parent"
            )
        if not parents["status_updated_at"].le(
            children["booked_at"]
        ).all():
            raise ValueError(
                "reschedule child booking must follow parent update"
            )

    for name, frame in (
        ("patients", patients),
        ("dentists", dentists),
        ("appointments", appointments),
    ):
        overlap = FORBIDDEN_EXPORTED_COLUMNS.intersection(
            frame.columns
        )
        if overlap:
            raise ValueError(
                f"{name} exposes hidden latent columns: "
                f"{sorted(overlap)}"
            )


__all__ = ("validate_synthetic_tables",)
