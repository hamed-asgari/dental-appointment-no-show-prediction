"""Deterministic strict-as-of aggregate history for the Version 2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.asof_history import (
    build_eligible_scoring_rows,
    normalize_history_appointments,
)
from src.features.schema import (
    AGGREGATE_HISTORY_DTYPES,
    AGGREGATE_HISTORY_OUTPUT_COLUMNS,
    AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    AGGREGATE_MIN_ATTENDANCE_SUPPORT,
    ATTENDANCE_STATUSES,
    NO_SHOW_PRIOR_ALPHA,
    NO_SHOW_PRIOR_MEAN,
    NO_SHOW_PRIOR_STRENGTH,
)


@dataclass(slots=True)
class _AggregateState:
    attendance_count: int = 0
    no_show_count: int = 0

    def add_attendance(self, *, status: str) -> None:
        self.attendance_count += 1
        if status == "no_show":
            self.no_show_count += 1


def _missing_columns(
    supplied: Iterable[str],
    required: Iterable[str],
) -> list[str]:
    return sorted(set(required) - set(supplied))


def _coerce_integer_identifiers(
    values: pd.Series,
    *,
    column: str,
) -> pd.Series:
    if values.isna().any():
        raise ValueError(
            f"appointments.{column} must not contain missing values"
        )

    if pd.api.types.is_integer_dtype(values.dtype):
        return values.astype("int64")

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_array = numeric.to_numpy(dtype="float64")
    if numeric.isna().any() or not np.equal(
        numeric_array,
        np.floor(numeric_array),
    ).all():
        raise ValueError(
            f"appointments.{column} must contain integer identifiers"
        )
    return numeric.astype("int64")


def normalize_aggregate_history_appointments(
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize appointment fields required for group history."""

    if not isinstance(appointments, pd.DataFrame):
        raise TypeError("appointments must be a pandas DataFrame")

    missing = _missing_columns(
        appointments.columns,
        AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    )
    if missing:
        raise ValueError(
            "appointments is missing required aggregate-history columns: "
            + ", ".join(missing)
        )

    core = normalize_history_appointments(appointments)

    extras = appointments.loc[
        :,
        ["appointment_id", "dentist_id", "visit_type"],
    ].copy(deep=True)
    extras["appointment_id"] = _coerce_integer_identifiers(
        extras["appointment_id"],
        column="appointment_id",
    )
    extras["dentist_id"] = _coerce_integer_identifiers(
        extras["dentist_id"],
        column="dentist_id",
    )

    visit_type = extras["visit_type"].astype("string")
    if visit_type.isna().any():
        raise ValueError(
            "appointments.visit_type must not contain missing values"
        )
    if visit_type.str.strip().eq("").any():
        raise ValueError(
            "appointments.visit_type must not contain empty values"
        )
    extras["visit_type"] = visit_type

    normalized = core.merge(
        extras,
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    normalized = normalized.loc[
        :,
        list(AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS),
    ].copy()
    normalized["dentist_id"] = normalized["dentist_id"].astype("int64")
    normalized["visit_type"] = normalized["visit_type"].astype("string")
    return normalized


def _rate_record(
    state: _AggregateState | None,
) -> tuple[int, bool, float]:
    if state is None:
        return 0, False, NO_SHOW_PRIOR_MEAN

    supported = (
        state.attendance_count >= AGGREGATE_MIN_ATTENDANCE_SUPPORT
    )
    if not supported:
        return state.attendance_count, False, NO_SHOW_PRIOR_MEAN

    smoothed_rate = (
        state.no_show_count + NO_SHOW_PRIOR_ALPHA
    ) / (state.attendance_count + NO_SHOW_PRIOR_STRENGTH)
    return state.attendance_count, True, smoothed_rate


def _cast_aggregate_history_output(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.loc[
        :,
        list(AGGREGATE_HISTORY_OUTPUT_COLUMNS),
    ].copy()
    for column, dtype in AGGREGATE_HISTORY_DTYPES.items():
        result[column] = result[column].astype(dtype)
    return result


def build_aggregate_history_features(
    appointments: pd.DataFrame,
) -> pd.DataFrame:
    """Build strict-as-of dentist, visit-type, and weekday-hour rates."""

    normalized = normalize_aggregate_history_appointments(appointments)
    scoring = build_eligible_scoring_rows(normalized)

    if scoring.empty:
        empty = pd.DataFrame(columns=AGGREGATE_HISTORY_OUTPUT_COLUMNS)
        return _cast_aggregate_history_output(empty)

    current_groups = normalized.loc[
        :,
        [
            "appointment_id",
            "dentist_id",
            "visit_type",
            "scheduled_start_at",
        ],
    ].copy()
    current_groups["scheduled_weekday"] = (
        current_groups["scheduled_start_at"].dt.weekday.astype("int8")
    )
    current_groups["scheduled_hour"] = (
        current_groups["scheduled_start_at"].dt.hour.astype("int8")
    )
    scoring = scoring.merge(
        current_groups.drop(columns="scheduled_start_at"),
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )

    events = normalized.loc[
        normalized["status"].isin(ATTENDANCE_STATUSES),
        [
            "appointment_id",
            "dentist_id",
            "visit_type",
            "scheduled_start_at",
            "status",
            "status_updated_at",
        ],
    ].copy()
    events["scheduled_weekday"] = (
        events["scheduled_start_at"].dt.weekday.astype("int8")
    )
    events["scheduled_hour"] = (
        events["scheduled_start_at"].dt.hour.astype("int8")
    )
    events = events.sort_values(
        ["status_updated_at", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )

    dentist_states: dict[int, _AggregateState] = {}
    visit_type_states: dict[str, _AggregateState] = {}
    weekday_hour_states: dict[tuple[int, int], _AggregateState] = {}

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
            status = str(event["status"])
            dentist_id = int(event["dentist_id"])
            visit_type = str(event["visit_type"])
            weekday_hour = (
                int(event["scheduled_weekday"]),
                int(event["scheduled_hour"]),
            )

            dentist_states.setdefault(
                dentist_id,
                _AggregateState(),
            ).add_attendance(status=status)
            visit_type_states.setdefault(
                visit_type,
                _AggregateState(),
            ).add_attendance(status=status)
            weekday_hour_states.setdefault(
                weekday_hour,
                _AggregateState(),
            ).add_attendance(status=status)
            event_index += 1

        for row in batch.itertuples(index=False):
            dentist_count, dentist_supported, dentist_rate = _rate_record(
                dentist_states.get(int(row.dentist_id))
            )
            visit_count, visit_supported, visit_rate = _rate_record(
                visit_type_states.get(str(row.visit_type))
            )
            weekday_hour_key = (
                int(row.scheduled_weekday),
                int(row.scheduled_hour),
            )
            (
                weekday_hour_count,
                weekday_hour_supported,
                weekday_hour_rate,
            ) = _rate_record(weekday_hour_states.get(weekday_hour_key))

            records.append(
                {
                    "appointment_id": int(row.appointment_id),
                    "patient_id": int(row.patient_id),
                    "prediction_time": current_time,
                    "dentist_prior_attendance_count": dentist_count,
                    "dentist_no_show_rate_supported": dentist_supported,
                    "dentist_prior_no_show_rate_smoothed": dentist_rate,
                    "visit_type_prior_attendance_count": visit_count,
                    "visit_type_no_show_rate_supported": visit_supported,
                    "visit_type_prior_no_show_rate_smoothed": visit_rate,
                    "weekday_hour_prior_attendance_count": weekday_hour_count,
                    "weekday_hour_no_show_rate_supported": (
                        weekday_hour_supported
                    ),
                    "weekday_hour_prior_no_show_rate_smoothed": (
                        weekday_hour_rate
                    ),
                }
            )

    result = pd.DataFrame.from_records(
        records,
        columns=AGGREGATE_HISTORY_OUTPUT_COLUMNS,
    )
    result = result.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )
    return _cast_aggregate_history_output(result)


__all__ = (
    "build_aggregate_history_features",
    "normalize_aggregate_history_appointments",
)
