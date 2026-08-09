"""Prediction-time-safe current appointment features for Version 2."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.features.aggregate_history import (
    normalize_aggregate_history_appointments,
)
from src.features.asof_history import build_eligible_scoring_rows
from src.features.schema import (
    CURRENT_APPOINTMENT_DTYPES,
    CURRENT_APPOINTMENT_OUTPUT_COLUMNS,
    CURRENT_FEATURE_REQUIRED_APPOINTMENT_COLUMNS,
    CURRENT_FEATURE_REQUIRED_DENTIST_COLUMNS,
    CURRENT_FEATURE_REQUIRED_PATIENT_COLUMNS,
)


_SECONDS_PER_DAY = 86_400.0


def _missing_columns(
    supplied: Iterable[str],
    required: Iterable[str],
) -> list[str]:
    return sorted(set(required) - set(supplied))


def _coerce_integer(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{context} must not contain missing values")
    if pd.api.types.is_integer_dtype(values.dtype):
        return values.astype("int64")

    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype="float64")
    if numeric.isna().any() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{context} must contain integers")
    return numeric.astype("int64")


def _coerce_required_datetime(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    try:
        converted = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain valid timestamps") from exc
    if converted.isna().any():
        raise ValueError(f"{context} must not contain missing values")
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{context} must be timezone-naive")
    return converted.astype("datetime64[ns]")


def _coerce_optional_datetime(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    try:
        converted = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must contain valid timestamps or missing values"
        ) from exc
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{context} must be timezone-naive")
    return converted.astype("datetime64[ns]")


def _validate_primary_key(
    frame: pd.DataFrame,
    column: str,
    *,
    table_name: str,
) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{table_name}.{column} must not contain missing values")
    if not frame[column].is_unique:
        raise ValueError(f"{table_name}.{column} must be unique")


def _validate_int_range(
    values: pd.Series | np.ndarray,
    dtype: str,
    *,
    column: str,
) -> None:
    array = np.asarray(values)
    if not np.isfinite(array.astype("float64")).all():
        raise ValueError(f"{column} must contain finite values")
    info = np.iinfo(dtype)
    if (array < info.min).any() or (array > info.max).any():
        raise ValueError(f"{column} exceeds {dtype} range")


def normalize_current_feature_sources(
    appointments: pd.DataFrame,
    patients: pd.DataFrame,
    dentists: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and normalize source fields required by current features."""

    if not isinstance(appointments, pd.DataFrame):
        raise TypeError("appointments must be a pandas DataFrame")
    if not isinstance(patients, pd.DataFrame):
        raise TypeError("patients must be a pandas DataFrame")
    if not isinstance(dentists, pd.DataFrame):
        raise TypeError("dentists must be a pandas DataFrame")

    missing = _missing_columns(
        appointments.columns,
        CURRENT_FEATURE_REQUIRED_APPOINTMENT_COLUMNS,
    )
    if missing:
        raise ValueError(
            "appointments is missing required current-feature columns: "
            + ", ".join(missing)
        )

    normalized_appointments = normalize_aggregate_history_appointments(
        appointments
    )
    extras = appointments.loc[
        :,
        [
            "appointment_id",
            "planned_duration_min",
            "booking_channel",
            "reminder_sent_at",
        ],
    ].copy(deep=True)
    extras["appointment_id"] = _coerce_integer(
        extras["appointment_id"],
        context="appointments.appointment_id",
    )

    planned_duration = _coerce_integer(
        extras["planned_duration_min"],
        context="appointments.planned_duration_min",
    )
    if planned_duration.le(0).any():
        raise ValueError(
            "appointments.planned_duration_min must be positive"
        )
    extras["planned_duration_min"] = planned_duration

    booking_channel = extras["booking_channel"].astype("string")
    if booking_channel.isna().any() or booking_channel.str.strip().eq("").any():
        raise ValueError(
            "appointments.booking_channel must contain non-empty values"
        )
    extras["booking_channel"] = booking_channel

    extras["reminder_sent_at"] = _coerce_optional_datetime(
        extras["reminder_sent_at"],
        context="appointments.reminder_sent_at",
    )

    normalized_appointments = normalized_appointments.merge(
        extras,
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    reminder_present = normalized_appointments["reminder_sent_at"].notna()
    if (
        reminder_present
        & normalized_appointments["reminder_sent_at"].lt(
            normalized_appointments["booked_at"]
        )
    ).any():
        raise ValueError(
            "appointments.reminder_sent_at must not be before booked_at"
        )
    if (
        reminder_present
        & normalized_appointments["reminder_sent_at"].gt(
            normalized_appointments["scheduled_start_at"]
        )
    ).any():
        raise ValueError(
            "appointments.reminder_sent_at must not be after scheduled_start_at"
        )

    missing_patients = _missing_columns(
        patients.columns,
        CURRENT_FEATURE_REQUIRED_PATIENT_COLUMNS,
    )
    if missing_patients:
        raise ValueError(
            "patients is missing required current-feature columns: "
            + ", ".join(missing_patients)
        )
    normalized_patients = patients.loc[
        :,
        list(CURRENT_FEATURE_REQUIRED_PATIENT_COLUMNS),
    ].copy(deep=True)
    normalized_patients["patient_id"] = _coerce_integer(
        normalized_patients["patient_id"],
        context="patients.patient_id",
    )
    _validate_primary_key(
        normalized_patients,
        "patient_id",
        table_name="patients",
    )
    normalized_patients["birth_year"] = _coerce_integer(
        normalized_patients["birth_year"],
        context="patients.birth_year",
    )
    normalized_patients["registered_at"] = _coerce_required_datetime(
        normalized_patients["registered_at"],
        context="patients.registered_at",
    )

    missing_dentists = _missing_columns(
        dentists.columns,
        CURRENT_FEATURE_REQUIRED_DENTIST_COLUMNS,
    )
    if missing_dentists:
        raise ValueError(
            "dentists is missing required current-feature columns: "
            + ", ".join(missing_dentists)
        )
    normalized_dentists = dentists.loc[
        :,
        list(CURRENT_FEATURE_REQUIRED_DENTIST_COLUMNS),
    ].copy(deep=True)
    normalized_dentists["dentist_id"] = _coerce_integer(
        normalized_dentists["dentist_id"],
        context="dentists.dentist_id",
    )
    _validate_primary_key(
        normalized_dentists,
        "dentist_id",
        table_name="dentists",
    )
    normalized_dentists["start_date"] = _coerce_required_datetime(
        normalized_dentists["start_date"],
        context="dentists.start_date",
    )

    return (
        normalized_appointments,
        normalized_patients,
        normalized_dentists,
    )


def _cast_output(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, list(CURRENT_APPOINTMENT_OUTPUT_COLUMNS)].copy()
    for column, dtype in CURRENT_APPOINTMENT_DTYPES.items():
        result[column] = result[column].astype(dtype)
    return result


def build_current_appointment_features(
    appointments: pd.DataFrame,
    patients: pd.DataFrame,
    dentists: pd.DataFrame,
) -> pd.DataFrame:
    """Build current appointment features known at the 24-hour horizon."""

    (
        normalized_appointments,
        normalized_patients,
        normalized_dentists,
    ) = normalize_current_feature_sources(
        appointments,
        patients,
        dentists,
    )

    scoring = build_eligible_scoring_rows(normalized_appointments)
    if scoring.empty:
        empty = pd.DataFrame(columns=CURRENT_APPOINTMENT_OUTPUT_COLUMNS)
        return _cast_output(empty)

    source = normalized_appointments.loc[
        :,
        [
            "appointment_id",
            "dentist_id",
            "booked_at",
            "scheduled_start_at",
            "planned_duration_min",
            "visit_type",
            "booking_channel",
            "reminder_sent_at",
        ],
    ].copy()

    joined = scoring.merge(
        source,
        on="appointment_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    joined = joined.merge(
        normalized_patients,
        on="patient_id",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator="_patient_join",
    )
    if joined["_patient_join"].ne("both").any():
        missing_ids = sorted(
            joined.loc[
                joined["_patient_join"].ne("both"),
                "patient_id",
            ].unique()
        )
        raise ValueError(
            "Eligible appointments reference unknown patients: "
            + ", ".join(str(value) for value in missing_ids)
        )
    joined = joined.drop(columns="_patient_join")

    joined = joined.merge(
        normalized_dentists,
        on="dentist_id",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator="_dentist_join",
    )
    if joined["_dentist_join"].ne("both").any():
        missing_ids = sorted(
            joined.loc[
                joined["_dentist_join"].ne("both"),
                "dentist_id",
            ].unique()
        )
        raise ValueError(
            "Eligible appointments reference unknown dentists: "
            + ", ".join(str(value) for value in missing_ids)
        )
    joined = joined.drop(columns="_dentist_join")

    lead_hours = (
        joined["scheduled_start_at"] - joined["booked_at"]
    ).dt.total_seconds() / 3_600.0
    if not np.isfinite(lead_hours).all() or lead_hours.lt(24.0).any():
        raise ValueError(
            "Eligible appointments must have at least 24 hours of lead time"
        )

    age = joined["prediction_time"].dt.year - joined["birth_year"]
    if age.lt(0).any():
        raise ValueError(
            "approximate_age_at_prediction must not be negative"
        )

    patient_tenure_seconds = (
        joined["prediction_time"] - joined["registered_at"]
    ).dt.total_seconds()
    if patient_tenure_seconds.lt(0).any():
        raise ValueError(
            "patient registration must not occur after prediction_time"
        )
    patient_tenure_days = np.floor(
        patient_tenure_seconds / _SECONDS_PER_DAY
    )

    dentist_tenure_seconds = (
        joined["prediction_time"] - joined["start_date"]
    ).dt.total_seconds()
    dentist_tenure_days = np.floor(
        dentist_tenure_seconds / _SECONDS_PER_DAY
    )

    _validate_int_range(
        joined["planned_duration_min"],
        "int16",
        column="planned_duration_min",
    )
    _validate_int_range(
        age,
        "int16",
        column="approximate_age_at_prediction",
    )
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

    result = pd.DataFrame(index=joined.index)
    result["appointment_id"] = joined["appointment_id"]
    result["patient_id"] = joined["patient_id"]
    result["dentist_id"] = joined["dentist_id"]
    result["prediction_time"] = joined["prediction_time"]
    result["planned_duration_min"] = joined["planned_duration_min"]
    result["visit_type"] = joined["visit_type"]
    result["booking_channel"] = joined["booking_channel"]
    result["booking_lead_time_hours"] = lead_hours
    result["scheduled_weekday"] = (
        joined["scheduled_start_at"].dt.weekday
    )
    result["scheduled_hour"] = joined["scheduled_start_at"].dt.hour
    result["scheduled_month"] = joined["scheduled_start_at"].dt.month
    result["approximate_age_at_prediction"] = age
    result["patient_registration_tenure_days"] = patient_tenure_days
    result["dentist_tenure_days"] = dentist_tenure_days
    result["reminder_sent_by_prediction_time"] = (
        joined["reminder_sent_at"].notna()
        & joined["reminder_sent_at"].le(joined["prediction_time"])
    )

    result = result.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
        ignore_index=True,
    )
    return _cast_output(result)


__all__ = (
    "build_current_appointment_features",
    "normalize_current_feature_sources",
)
