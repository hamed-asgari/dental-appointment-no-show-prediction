"""Frozen public schema for the Version 2 longitudinal synthetic benchmark."""

from __future__ import annotations

from types import MappingProxyType


PATIENT_COLUMNS = (
    "patient_id",
    "birth_year",
    "sex",
    "city_area",
    "registered_at",
    "insurance_type",
    "referral_source",
    "preferred_contact_channel",
    "patient_status",
)

DENTIST_COLUMNS = (
    "dentist_id",
    "dentist_role",
    "engagement_type",
    "start_date",
    "end_date",
    "scheduled_hours_weekly",
    "active",
)

APPOINTMENT_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "booked_at",
    "scheduled_start_at",
    "planned_duration_min",
    "visit_type",
    "booking_channel",
    "status",
    "status_updated_at",
    "reminder_sent",
    "reminder_sent_at",
    "check_in_at",
    "chair_start_at",
    "chair_end_at",
    "checkout_at",
    "status_change_reason",
    "rescheduled_from_appointment_id",
)

VISIT_TYPES = (
    "new_patient_examination",
    "recall_examination",
    "consultation",
    "treatment",
    "emergency",
    "follow_up",
)

BOOKING_CHANNELS = (
    "phone",
    "in_person",
    "online",
    "referral",
    "other",
)

STATUSES = (
    "completed",
    "no_show",
    "cancelled",
    "rescheduled",
)

DURATION_BY_VISIT = MappingProxyType(
    {
        "new_patient_examination": 45,
        "recall_examination": 30,
        "consultation": 30,
        "treatment": 60,
        "emergency": 45,
        "follow_up": 30,
    }
)

ROLE_COMPATIBILITY = MappingProxyType(
    {
        "new_patient_examination": (
            "general_dentist",
            "prosthodontist",
            "periodontist",
        ),
        "recall_examination": (
            "general_dentist",
            "periodontist",
        ),
        "consultation": (
            "general_dentist",
            "endodontist",
            "oral_surgeon",
            "orthodontist",
            "prosthodontist",
            "periodontist",
        ),
        "treatment": (
            "general_dentist",
            "endodontist",
            "oral_surgeon",
            "orthodontist",
            "prosthodontist",
            "periodontist",
        ),
        "emergency": (
            "general_dentist",
            "endodontist",
            "oral_surgeon",
        ),
        "follow_up": (
            "general_dentist",
            "endodontist",
            "oral_surgeon",
            "orthodontist",
            "prosthodontist",
            "periodontist",
        ),
    }
)

FORBIDDEN_EXPORTED_COLUMNS = frozenset(
    {
        "patient_risk_effect",
        "patient_visit_weight",
        "dentist_risk_effect",
        "no_show_probability",
    }
)


__all__ = (
    "APPOINTMENT_COLUMNS",
    "BOOKING_CHANNELS",
    "DENTIST_COLUMNS",
    "DURATION_BY_VISIT",
    "FORBIDDEN_EXPORTED_COLUMNS",
    "PATIENT_COLUMNS",
    "ROLE_COMPATIBILITY",
    "STATUSES",
    "VISIT_TYPES",
)
