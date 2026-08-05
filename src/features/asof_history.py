"""Deterministic strict-as-of patient history for the Version 2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.schema import (
    ATTENDANCE_STATUSES,
    HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    HISTORY_STATUSES,
    NO_SHOW_PRIOR_ALPHA,
    NO_SHOW_PRIOR_STRENGTH,
    PATIENT_HISTORY_DTYPES,
    PATIENT_HISTORY_OUTPUT_COLUMNS,
    PREDICTION_HORIZON_HOURS,
    PRE_PREDICTION_INACTIVE_STATUSES,
)


_NANOSECONDS_PER_DAY = 86_400_000_000_000


@dataclass(slots=True)
class _PatientState:
    known_count: int = 0
    attendance_count: int = 0
    completed_count: int = 0
    no_show_count: int = 0
    cancelled_count: int = 0
    rescheduled_count: int = 0
    booking_lead_days_sum: float = 0.0
    last_status_updated_at: pd.Timestamp | None = None
    last_completed_scheduled_start_at: pd.Timestamp | None = None

    def add_event(
        self,
        *,
        status: str,
        status_updated_at: pd.Timestamp,
        scheduled_start_at: pd.Timestamp,
        booking_lead_days: float,
    ) -> None:
        self.known_count += 1
        self.booking_lead_days_sum += booking_lead_days
        self.last_status_updated_at = status_updated_at

        if status == "completed":
            self.attendance_count += 1
            self.completed_count += 1
            if (
                self.last_completed_scheduled_start_at is None
                or scheduled_start_at > self.last_completed_scheduled_start_at
            ):
                self.last_completed_scheduled_start_at = scheduled_start_at
        elif status == "no_show":
            self.attendance_count += 1
            self.no_show_count += 1
        elif status == "cancelled":
            self.cancelled_count += 1
        elif status == "rescheduled":
            self.rescheduled_count += 1
        else:  # pragma: no cover - guarded by input validation
            raise ValueError(f"Unsupported historical status: {status!r}")


def _missing_columns(
    supplied: Iterable[str],
    required: Iterable[str],
) -> list[str]:
    return sorted(set(required) - set(supplied))


def _coerce_datetime(
    values: pd.Series,
    *,
    column: str,
) -> pd.Series:
    try:
        converted = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"appointments.{column} must contain valid timestamps"
        ) from exc

    if converted.isna().any():
        raise ValueError(
            f"appointments.{column} must not contain missing values"
        )
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"appointments.{column} must be timezone-naive"
        )
    return converted.astype("datetime64[ns]")


def normalize_history_appointments(
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize the minimum appointment history source table."""

    if not isinstance(appointments, pd.DataFrame):
        raise TypeError("appointments must be a pandas DataFrame")

    missing = _missing_columns(
        appointments.columns,
        HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    )
    if missing:
        raise ValueError(
            "appointments is missing required columns: "
            + ", ".join(missing)
        )

    normalized = appointments.loc[
        :,
        list(HISTORY_REQUIRED_APPOINTMENT_COLUMNS),
    ].copy(deep=True)

    for column in ("appointment_id", "patient_id"):
        if normalized[column].isna().any():
            raise ValueError(
                f"appointments.{column} must not contain missing values"
            )
        if not pd.api.types.is_integer_dtype(normalized[column].dtype):
            numeric = pd.to_numeric(normalized[column], errors="coerce")
            if numeric.isna().any() or not np.equal(
                numeric.to_numpy(dtype="float64"),
                np.floor(numeric.to_numpy(dtype="float64")),
            ).all():
                raise ValueError(
                    f"appointments.{column} must contain integer identifiers"
                )
            normalized[column] = numeric.astype("int64")
        else:
            normalized[column] = normalized[column].astype("int64")

    if not normalized["appointment_id"].is_unique:
        raise ValueError("appointments.appointment_id must be unique")

    for column in (
        "booked_at",
        "scheduled_start_at",
        "status_updated_at",
    ):
        normalized[column] = _coerce_datetime(
            normalized[column],
            column=column,
        )

    status = normalized["status"].astype("string")
    if status.isna().any():
        raise ValueError("appointments.status must not contain missing values")
    invalid_statuses = sorted(set(status.unique()) - HISTORY_STATUSES)
    if invalid_statuses:
        raise ValueError(
            "appointments.status contains unsupported values: "
            + ", ".join(invalid_statuses)
        )
    normalized["status"] = status

    invalid_booking_order = (
        normalized["booked_at"] > normalized["scheduled_start_at"]
    )
    if invalid_booking_order.any():
        raise ValueError(
            "appointments.booked_at must not be after scheduled_start_at"
        )

    if (normalized["status_updated_at"] < normalized["booked_at"]).any():
        raise ValueError(
            "appointments.status_updated_at must not be before booked_at"
        )

    attendance = normalized["status"].isin(ATTENDANCE_STATUSES)
    if (
        attendance
        & normalized["status_updated_at"].lt(
            normalized["scheduled_start_at"]
        )
    ).any():
        raise ValueError(
            "completed and no_show status updates must not be before "
            "scheduled_start_at"
        )

    return normalized


def _build_eligible_from_normalized(
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    prediction_time = normalized["scheduled_start_at"] - pd.Timedelta(
        hours=PREDICTION_HORIZON_HOURS
    )
    inactive_before_or_at_prediction = (
        normalized["status"].isin(PRE_PREDICTION_INACTIVE_STATUSES)
        & normalized["status_updated_at"].le(prediction_time)
    )
    eligible = (
        normalized["booked_at"].le(prediction_time)
        & ~inactive_before_or_at_prediction
    )

    scoring = normalized.loc[
        eligible,
        ["appointment_id", "patient_id"],
    ].copy()
    scoring["prediction_time"] = prediction_time.loc[eligible]
    scoring = scoring.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )
    scoring["appointment_id"] = scoring["appointment_id"].astype("int64")
    scoring["patient_id"] = scoring["patient_id"].astype("int64")
    scoring["prediction_time"] = scoring["prediction_time"].astype(
        "datetime64[ns]"
    )
    return scoring


def build_eligible_scoring_rows(
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    """Return deterministic Version 2 scoring keys under the 24-hour rule."""

    normalized = normalize_history_appointments(appointments)
    return _build_eligible_from_normalized(normalized)


def _days_between(
    later: pd.Timestamp,
    earlier: pd.Timestamp | None,
) -> float:
    if earlier is None:
        return 0.0
    delta = later - earlier
    return float(delta.value) / float(_NANOSECONDS_PER_DAY)


def _feature_record(
    *,
    appointment_id: int,
    patient_id: int,
    prediction_time: pd.Timestamp,
    state: _PatientState | None,
) -> dict[str, object]:
    if state is None:
        state = _PatientState()

    smoothed_rate = (
        state.no_show_count + NO_SHOW_PRIOR_ALPHA
    ) / (state.attendance_count + NO_SHOW_PRIOR_STRENGTH)
    mean_lead_days = (
        state.booking_lead_days_sum / state.known_count
        if state.known_count
        else 0.0
    )

    return {
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "prediction_time": prediction_time,
        "patient_history_available": state.known_count > 0,
        "patient_completed_history_available": state.completed_count > 0,
        "patient_prior_known_appointment_count": state.known_count,
        "patient_prior_attendance_count": state.attendance_count,
        "patient_prior_completed_count": state.completed_count,
        "patient_prior_no_show_count": state.no_show_count,
        "patient_prior_cancelled_count": state.cancelled_count,
        "patient_prior_rescheduled_count": state.rescheduled_count,
        "patient_prior_no_show_rate_smoothed": smoothed_rate,
        "patient_days_since_last_known_status_update": _days_between(
            prediction_time,
            state.last_status_updated_at,
        ),
        "patient_days_since_last_completed_appointment": _days_between(
            prediction_time,
            state.last_completed_scheduled_start_at,
        ),
        "patient_mean_prior_booking_lead_days": mean_lead_days,
    }


def _cast_patient_history_output(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, list(PATIENT_HISTORY_OUTPUT_COLUMNS)].copy()
    for column, dtype in PATIENT_HISTORY_DTYPES.items():
        result[column] = result[column].astype(dtype)
    return result


def build_patient_history_features(
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    """Build strict-as-of patient history for every eligible scoring row.

    Historical events become available only when ``status_updated_at`` is
    strictly earlier than the current row's prediction time. Rows sharing a
    prediction time are evaluated against the same pre-time state.
    """

    normalized = normalize_history_appointments(appointments)
    scoring = _build_eligible_from_normalized(normalized)

    if scoring.empty:
        empty = pd.DataFrame(columns=PATIENT_HISTORY_OUTPUT_COLUMNS)
        return _cast_patient_history_output(empty)

    events = normalized.assign(
        booking_lead_days=(
            normalized["scheduled_start_at"] - normalized["booked_at"]
        ).dt.total_seconds()
        / 86_400.0
    ).sort_values(
        ["status_updated_at", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )

    states: dict[int, _PatientState] = {}
    records: list[dict[str, object]] = []
    event_index = 0
    event_count = len(events)

    for prediction_time, batch in scoring.groupby(
        "prediction_time",
        sort=True,
        observed=True,
    ):
        current_time = pd.Timestamp(prediction_time)

        while (
            event_index < event_count
            and events.at[event_index, "status_updated_at"] < current_time
        ):
            event = events.iloc[event_index]
            patient_id = int(event["patient_id"])
            state = states.setdefault(patient_id, _PatientState())
            state.add_event(
                status=str(event["status"]),
                status_updated_at=pd.Timestamp(event["status_updated_at"]),
                scheduled_start_at=pd.Timestamp(event["scheduled_start_at"]),
                booking_lead_days=float(event["booking_lead_days"]),
            )
            event_index += 1

        for row in batch.itertuples(index=False):
            patient_id = int(row.patient_id)
            records.append(
                _feature_record(
                    appointment_id=int(row.appointment_id),
                    patient_id=patient_id,
                    prediction_time=current_time,
                    state=states.get(patient_id),
                )
            )

    result = pd.DataFrame.from_records(
        records,
        columns=PATIENT_HISTORY_OUTPUT_COLUMNS,
    )
    result = result.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )
    return _cast_patient_history_output(result)


__all__ = (
    "build_eligible_scoring_rows",
    "build_patient_history_features",
    "normalize_history_appointments",
)
