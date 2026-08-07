"""Immutable Version 2 feature names, dtypes, and smoothing constants."""

from __future__ import annotations

from types import MappingProxyType


PREDICTION_HORIZON_HOURS = 24
HISTORY_STATUSES = frozenset(
    {
        "completed",
        "no_show",
        "cancelled",
        "rescheduled",
    }
)
ATTENDANCE_STATUSES = frozenset({"completed", "no_show"})
PRE_PREDICTION_INACTIVE_STATUSES = frozenset({"cancelled", "rescheduled"})

NO_SHOW_PRIOR_ALPHA = 1.0
NO_SHOW_PRIOR_BETA = 9.0
NO_SHOW_PRIOR_STRENGTH = NO_SHOW_PRIOR_ALPHA + NO_SHOW_PRIOR_BETA
NO_SHOW_PRIOR_MEAN = NO_SHOW_PRIOR_ALPHA / NO_SHOW_PRIOR_STRENGTH
AGGREGATE_MIN_ATTENDANCE_SUPPORT = 10

HISTORY_REQUIRED_APPOINTMENT_COLUMNS = (
    "appointment_id",
    "patient_id",
    "booked_at",
    "scheduled_start_at",
    "status",
    "status_updated_at",
)

AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS = (
    *HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    "dentist_id",
    "visit_type",
)

AUDIT_KEY_COLUMNS = (
    "appointment_id",
    "patient_id",
    "prediction_time",
)

CURRENT_APPOINTMENT_FEATURE_COLUMNS = (
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
    "reminder_sent_by_prediction_time",
)

PATIENT_HISTORY_FEATURE_COLUMNS = (
    "patient_history_available",
    "patient_completed_history_available",
    "patient_prior_known_appointment_count",
    "patient_prior_attendance_count",
    "patient_prior_completed_count",
    "patient_prior_no_show_count",
    "patient_prior_cancelled_count",
    "patient_prior_rescheduled_count",
    "patient_prior_no_show_rate_smoothed",
    "patient_days_since_last_known_status_update",
    "patient_days_since_last_completed_appointment",
    "patient_mean_prior_booking_lead_days",
)

AGGREGATE_HISTORY_FEATURE_COLUMNS = (
    "dentist_prior_attendance_count",
    "dentist_no_show_rate_supported",
    "dentist_prior_no_show_rate_smoothed",
    "visit_type_prior_attendance_count",
    "visit_type_no_show_rate_supported",
    "visit_type_prior_no_show_rate_smoothed",
    "weekday_hour_prior_attendance_count",
    "weekday_hour_no_show_rate_supported",
    "weekday_hour_prior_no_show_rate_smoothed",
)

V2_MODEL_FEATURE_COLUMNS = (
    *CURRENT_APPOINTMENT_FEATURE_COLUMNS,
    *PATIENT_HISTORY_FEATURE_COLUMNS,
    *AGGREGATE_HISTORY_FEATURE_COLUMNS,
)


CURRENT_FEATURE_REQUIRED_APPOINTMENT_COLUMNS = (
    *AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS,
    "planned_duration_min",
    "booking_channel",
    "reminder_sent_at",
)

CURRENT_FEATURE_REQUIRED_PATIENT_COLUMNS = (
    "patient_id",
    "birth_year",
    "registered_at",
)

CURRENT_FEATURE_REQUIRED_DENTIST_COLUMNS = (
    "dentist_id",
    "start_date",
)

CURRENT_APPOINTMENT_OUTPUT_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "prediction_time",
    *CURRENT_APPOINTMENT_FEATURE_COLUMNS,
)

CURRENT_APPOINTMENT_DTYPES = MappingProxyType(
    {
        "appointment_id": "int64",
        "patient_id": "int64",
        "dentist_id": "int64",
        "prediction_time": "datetime64[ns]",
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
        "reminder_sent_by_prediction_time": "bool",
    }
)

V2_EVALUATION_PARTITIONS = (
    "context_only",
    "warmup",
    "development_fit",
    "fold_1_validation",
    "fold_2_validation",
    "fold_3_validation",
    "calibration",
    "policy_selection",
    "final_test",
)

V2_FEATURE_DATASET_AUDIT_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "prediction_time",
)

V2_FEATURE_DATASET_PARTITION_COLUMNS = (
    "evaluation_partition",
)

V2_FEATURE_DATASET_LABEL_METADATA_COLUMNS = (
    "label_available_at",
)

V2_FEATURE_DATASET_COLUMNS = (
    *V2_FEATURE_DATASET_AUDIT_COLUMNS,
    *V2_MODEL_FEATURE_COLUMNS,
    *V2_FEATURE_DATASET_PARTITION_COLUMNS,
    *V2_FEATURE_DATASET_LABEL_METADATA_COLUMNS,
)

PATIENT_HISTORY_OUTPUT_COLUMNS = (
    *AUDIT_KEY_COLUMNS,
    *PATIENT_HISTORY_FEATURE_COLUMNS,
)

PATIENT_HISTORY_DTYPES = MappingProxyType(
    {
        "appointment_id": "int64",
        "patient_id": "int64",
        "prediction_time": "datetime64[ns]",
        "patient_history_available": "bool",
        "patient_completed_history_available": "bool",
        "patient_prior_known_appointment_count": "int32",
        "patient_prior_attendance_count": "int32",
        "patient_prior_completed_count": "int32",
        "patient_prior_no_show_count": "int32",
        "patient_prior_cancelled_count": "int32",
        "patient_prior_rescheduled_count": "int32",
        "patient_prior_no_show_rate_smoothed": "float64",
        "patient_days_since_last_known_status_update": "float64",
        "patient_days_since_last_completed_appointment": "float64",
        "patient_mean_prior_booking_lead_days": "float64",
    }
)

AGGREGATE_HISTORY_OUTPUT_COLUMNS = (
    *AUDIT_KEY_COLUMNS,
    *AGGREGATE_HISTORY_FEATURE_COLUMNS,
)

AGGREGATE_HISTORY_DTYPES = MappingProxyType(
    {
        "appointment_id": "int64",
        "patient_id": "int64",
        "prediction_time": "datetime64[ns]",
        "dentist_prior_attendance_count": "int32",
        "dentist_no_show_rate_supported": "bool",
        "dentist_prior_no_show_rate_smoothed": "float64",
        "visit_type_prior_attendance_count": "int32",
        "visit_type_no_show_rate_supported": "bool",
        "visit_type_prior_no_show_rate_smoothed": "float64",
        "weekday_hour_prior_attendance_count": "int32",
        "weekday_hour_no_show_rate_supported": "bool",
        "weekday_hour_prior_no_show_rate_smoothed": "float64",
    }
)

V2_PROHIBITED_MODEL_COLUMNS = frozenset(
    {
        "appointment_id",
        "patient_id",
        "dentist_id",
        "status",
        "status_updated_at",
        "check_in_at",
        "chair_start_at",
        "chair_end_at",
        "checkout_at",
        "status_change_reason",
        "reminder_sent",
        "reminder_sent_at",
        "patient_risk_effect",
        "patient_visit_weight",
        "dentist_risk_effect",
        "no_show_probability",
        "target",
    }
)


__all__ = (
    "AGGREGATE_HISTORY_DTYPES",
    "AGGREGATE_HISTORY_FEATURE_COLUMNS",
    "AGGREGATE_HISTORY_OUTPUT_COLUMNS",
    "AGGREGATE_HISTORY_REQUIRED_APPOINTMENT_COLUMNS",
    "AGGREGATE_MIN_ATTENDANCE_SUPPORT",
    "ATTENDANCE_STATUSES",
    "AUDIT_KEY_COLUMNS",
    "CURRENT_APPOINTMENT_FEATURE_COLUMNS",
    "CURRENT_APPOINTMENT_DTYPES",
    "CURRENT_APPOINTMENT_OUTPUT_COLUMNS",
    "CURRENT_FEATURE_REQUIRED_APPOINTMENT_COLUMNS",
    "CURRENT_FEATURE_REQUIRED_DENTIST_COLUMNS",
    "CURRENT_FEATURE_REQUIRED_PATIENT_COLUMNS",
    "HISTORY_REQUIRED_APPOINTMENT_COLUMNS",
    "HISTORY_STATUSES",
    "NO_SHOW_PRIOR_ALPHA",
    "NO_SHOW_PRIOR_BETA",
    "NO_SHOW_PRIOR_MEAN",
    "NO_SHOW_PRIOR_STRENGTH",
    "PATIENT_HISTORY_DTYPES",
    "PATIENT_HISTORY_FEATURE_COLUMNS",
    "PATIENT_HISTORY_OUTPUT_COLUMNS",
    "PREDICTION_HORIZON_HOURS",
    "PRE_PREDICTION_INACTIVE_STATUSES",
    "V2_EVALUATION_PARTITIONS",
    "V2_FEATURE_DATASET_AUDIT_COLUMNS",
    "V2_FEATURE_DATASET_COLUMNS",
    "V2_FEATURE_DATASET_LABEL_METADATA_COLUMNS",
    "V2_FEATURE_DATASET_PARTITION_COLUMNS",
    "V2_MODEL_FEATURE_COLUMNS",
    "V2_PROHIBITED_MODEL_COLUMNS",
)
