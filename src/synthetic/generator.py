"""Generate deterministic longitudinal synthetic dental appointment tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.random import Generator

from src.synthetic.config import (
    BenchmarkConfig,
    derive_rng_stream_seeds,
    load_benchmark_config,
)
from src.synthetic.schema import (
    APPOINTMENT_COLUMNS,
    BOOKING_CHANNELS,
    DENTIST_COLUMNS,
    DURATION_BY_VISIT,
    PATIENT_COLUMNS,
    ROLE_COMPATIBILITY,
    VISIT_TYPES,
)
from src.synthetic.tables import SyntheticTables


@dataclass(frozen=True, slots=True)
class _PatientLatentState:
    risk_effect: np.ndarray
    visit_weight: np.ndarray


@dataclass(frozen=True, slots=True)
class _DentistLatentState:
    risk_effect: Mapping[int, float]


def create_rng_streams(
    config: BenchmarkConfig,
) -> Mapping[str, Generator]:
    """Create fresh deterministic NumPy generators for every named stream."""

    seeds = derive_rng_stream_seeds(config)
    streams = {
        name: np.random.default_rng(seed)
        for name, seed in seeds.items()
    }
    return MappingProxyType(streams)


def _registration_timestamps(
    config: BenchmarkConfig,
    rng: Generator,
) -> pd.DatetimeIndex:
    pre_count = max(1, int(round(config.patient_count * 0.25)))
    post_count = config.patient_count - pre_count

    appointment_start = pd.Timestamp(config.appointment_start)
    appointment_end = pd.Timestamp(config.appointment_end)

    pre_days = rng.integers(
        1,
        366,
        size=pre_count,
    )
    pre_dates = appointment_start - pd.to_timedelta(
        pre_days,
        unit="D",
    )

    post_span_days = max(
        1,
        (appointment_end - appointment_start).days - 30,
    )
    post_fraction = rng.beta(
        1.15,
        1.65,
        size=post_count,
    )
    post_days = np.floor(
        post_fraction * post_span_days
    ).astype(int)
    post_dates = appointment_start + pd.to_timedelta(
        post_days,
        unit="D",
    )

    dates = pd.DatetimeIndex(
        np.concatenate(
            (
                pre_dates.to_numpy(),
                post_dates.to_numpy(),
            )
        )
    )
    minutes = rng.integers(
        8 * 60,
        20 * 60,
        size=config.patient_count,
    )
    return dates + pd.to_timedelta(minutes, unit="m")


def _generate_patients_and_latent(
    config: BenchmarkConfig,
    *,
    patients_rng: Generator,
    latent_rng: Generator,
) -> tuple[pd.DataFrame, _PatientLatentState]:
    registered_at = _registration_timestamps(
        config,
        patients_rng,
    )
    registration_year = registered_at.year.to_numpy()
    age_at_registration = np.clip(
        np.rint(
            patients_rng.normal(
                39.0,
                18.0,
                size=config.patient_count,
            )
        ).astype(int),
        5,
        85,
    )
    birth_year = registration_year - age_at_registration

    patient_status = patients_rng.choice(
        np.array(
            ["active", "inactive", "archived"],
            dtype=object,
        ),
        size=config.patient_count,
        p=[0.93, 0.06, 0.01],
    )

    patients = pd.DataFrame(
        {
            "patient_id": np.arange(
                1,
                config.patient_count + 1,
                dtype=np.int64,
            ),
            "birth_year": birth_year.astype(np.int16),
            "sex": patients_rng.choice(
                np.array(
                    ["female", "male", "other_unknown"],
                    dtype=object,
                ),
                size=config.patient_count,
                p=[0.53, 0.46, 0.01],
            ),
            "city_area": patients_rng.choice(
                np.array(
                    [
                        "fardis",
                        "karaj",
                        "west_tehran",
                        "central_tehran",
                        "other",
                    ],
                    dtype=object,
                ),
                size=config.patient_count,
                p=[0.35, 0.30, 0.15, 0.10, 0.10],
            ),
            "registered_at": registered_at,
            "insurance_type": patients_rng.choice(
                np.array(
                    [
                        "self_pay",
                        "basic_insurance",
                        "supplementary_insurance",
                        "mixed",
                    ],
                    dtype=object,
                ),
                size=config.patient_count,
                p=[0.38, 0.30, 0.12, 0.20],
            ),
            "referral_source": patients_rng.choice(
                np.array(
                    [
                        "existing_patient_referral",
                        "dentist_referral",
                        "online_search",
                        "social_media",
                        "walk_in",
                        "advertising",
                        "other",
                    ],
                    dtype=object,
                ),
                size=config.patient_count,
                p=[0.32, 0.12, 0.18, 0.16, 0.10, 0.07, 0.05],
            ),
            "preferred_contact_channel": patients_rng.choice(
                np.array(
                    [
                        "phone_call",
                        "sms",
                        "messaging_application",
                        "email",
                        "other",
                    ],
                    dtype=object,
                ),
                size=config.patient_count,
                p=[0.28, 0.20, 0.44, 0.05, 0.03],
            ),
            "patient_status": patient_status,
        }
    ).loc[:, PATIENT_COLUMNS]

    risk_effect = latent_rng.normal(
        0.0,
        0.85,
        size=config.patient_count,
    )
    visit_weight = latent_rng.gamma(
        shape=1.8,
        scale=1.0,
        size=config.patient_count,
    )
    visit_weight = np.where(
        patient_status == "active",
        visit_weight,
        0.0,
    )

    return (
        patients,
        _PatientLatentState(
            risk_effect=risk_effect,
            visit_weight=visit_weight,
        ),
    )


def generate_patients(
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Generate the public patient table without hidden latent state."""

    streams = create_rng_streams(config)
    patients, _ = _generate_patients_and_latent(
        config,
        patients_rng=streams["patients"],
        latent_rng=streams["latent_risk"],
    )
    return patients


def _dentist_records() -> list[dict[str, object]]:
    return [
        {
            "dentist_id": 1,
            "dentist_role": "general_dentist",
            "engagement_type": "employee",
            "start_date": date(2022, 7, 1),
            "end_date": pd.NaT,
            "scheduled_hours_weekly": 30,
            "active": True,
        },
        {
            "dentist_id": 2,
            "dentist_role": "general_dentist",
            "engagement_type": "contractor",
            "start_date": date(2022, 10, 15),
            "end_date": pd.NaT,
            "scheduled_hours_weekly": 24,
            "active": True,
        },
        {
            "dentist_id": 3,
            "dentist_role": "endodontist",
            "engagement_type": "visiting_specialist",
            "start_date": date(2023, 1, 1),
            "end_date": pd.NaT,
            "scheduled_hours_weekly": 8,
            "active": True,
        },
        {
            "dentist_id": 4,
            "dentist_role": "oral_surgeon",
            "engagement_type": "visiting_specialist",
            "start_date": date(2023, 1, 1),
            "end_date": pd.NaT,
            "scheduled_hours_weekly": 6,
            "active": True,
        },
        {
            "dentist_id": 5,
            "dentist_role": "orthodontist",
            "engagement_type": "visiting_specialist",
            "start_date": date(2023, 3, 1),
            "end_date": pd.NaT,
            "scheduled_hours_weekly": 8,
            "active": True,
        },
        {
            "dentist_id": 6,
            "dentist_role": "prosthodontist",
            "engagement_type": "contractor",
            "start_date": date(2022, 11, 1),
            "end_date": pd.NaT,
            "scheduled_hours_weekly": 12,
            "active": True,
        },
        {
            "dentist_id": 7,
            "dentist_role": "periodontist",
            "engagement_type": "visiting_specialist",
            "start_date": date(2023, 2, 1),
            "end_date": date(2025, 8, 31),
            "scheduled_hours_weekly": 6,
            "active": False,
        },
    ]


def _generate_dentists_and_latent(
    config: BenchmarkConfig,
    *,
    dentists_rng: Generator,
    latent_rng: Generator,
) -> tuple[pd.DataFrame, _DentistLatentState]:
    records = _dentist_records()
    if config.dentist_count != len(records):
        raise ValueError(
            "The frozen Version 2 dentist_count must equal "
            f"{len(records)}"
        )

    dentists = pd.DataFrame.from_records(records).loc[
        :,
        DENTIST_COLUMNS,
    ]
    dentists["start_date"] = pd.to_datetime(
        dentists["start_date"]
    )
    dentists["end_date"] = pd.to_datetime(
        dentists["end_date"]
    )

    dentist_ids = dentists["dentist_id"].to_numpy(dtype=int)
    shuffled_effects = latent_rng.normal(
        0.0,
        0.18,
        size=len(dentist_ids),
    )
    dentists_rng.shuffle(shuffled_effects)
    risk_effect = MappingProxyType(
        {
            int(dentist_id): float(effect)
            for dentist_id, effect in zip(
                dentist_ids,
                shuffled_effects,
                strict=True,
            )
        }
    )
    return (
        dentists,
        _DentistLatentState(
            risk_effect=risk_effect,
        ),
    )


def generate_dentists(
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Generate the public dentist table without hidden latent state."""

    streams = create_rng_streams(config)
    dentists, _ = _generate_dentists_and_latent(
        config,
        dentists_rng=streams["dentists"],
        latent_rng=streams["latent_risk"],
    )
    return dentists


def _sample_scheduled_start(
    config: BenchmarkConfig,
    rng: Generator,
) -> pd.DatetimeIndex:
    clinic_dates = pd.date_range(
        config.appointment_start,
        config.appointment_end,
        freq="D",
    )
    clinic_dates = clinic_dates[clinic_dates.dayofweek != 4]

    sampled_dates = pd.to_datetime(
        rng.choice(
            clinic_dates.to_numpy(),
            size=config.appointment_count,
            replace=True,
        )
    )
    hours = rng.choice(
        np.array(
            [9, 10, 11, 12, 14, 15, 16, 17, 18],
            dtype=int,
        ),
        size=config.appointment_count,
        p=[
            0.08,
            0.12,
            0.13,
            0.10,
            0.11,
            0.13,
            0.13,
            0.12,
            0.08,
        ],
    )
    minutes = rng.choice(
        np.array([0, 30], dtype=int),
        size=config.appointment_count,
    )
    scheduled = (
        sampled_dates
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    return pd.DatetimeIndex(scheduled).sort_values()


def _sample_lead_days(
    size: int,
    rng: Generator,
) -> np.ndarray:
    bands = rng.choice(
        np.array([0, 1, 2, 3], dtype=np.int8),
        size=size,
        p=[0.30, 0.35, 0.25, 0.10],
    )
    lead_days = np.empty(size, dtype=np.int16)
    bounds = (
        (1, 8),
        (8, 22),
        (22, 46),
        (46, 91),
    )
    for band, (low, high) in enumerate(bounds):
        mask = bands == band
        lead_days[mask] = rng.integers(
            low,
            high,
            size=int(mask.sum()),
        )
    return lead_days


def _sample_booked_at(
    scheduled_start_at: pd.DatetimeIndex,
    rng: Generator,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    lead_days = _sample_lead_days(
        len(scheduled_start_at),
        rng,
    )
    booking_minutes = rng.integers(
        8 * 60,
        20 * 60,
        size=len(scheduled_start_at),
    )
    booked_at = (
        (
            scheduled_start_at
            - pd.to_timedelta(lead_days, unit="D")
        ).normalize()
        + pd.to_timedelta(booking_minutes, unit="m")
    )
    return pd.DatetimeIndex(booked_at), lead_days


def _assign_patients(
    booked_at: pd.DatetimeIndex,
    patients: pd.DataFrame,
    latent: _PatientLatentState,
    rng: Generator,
) -> np.ndarray:
    active_mask = patients["patient_status"].eq("active").to_numpy()
    active = patients.loc[
        active_mask,
        ["patient_id", "registered_at"],
    ].copy()
    active["_weight"] = latent.visit_weight[active_mask]
    active = active.sort_values(
        ["registered_at", "patient_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    registration = active["registered_at"].to_numpy(
        dtype="datetime64[ns]"
    )
    patient_ids = active["patient_id"].to_numpy(dtype=np.int64)
    cumulative_weight = np.cumsum(
        active["_weight"].to_numpy(dtype=float)
    )

    selected = np.empty(len(booked_at), dtype=np.int64)
    for index, booking_time in enumerate(
        booked_at.to_numpy(dtype="datetime64[ns]")
    ):
        eligible_count = int(
            np.searchsorted(
                registration,
                booking_time,
                side="right",
            )
        )
        if eligible_count == 0:
            raise ValueError(
                "No active patient was registered before "
                f"booking time {booking_time}"
            )
        total_weight = cumulative_weight[eligible_count - 1]
        draw = rng.random() * total_weight
        selected_index = int(
            np.searchsorted(
                cumulative_weight[:eligible_count],
                draw,
                side="right",
            )
        )
        selected[index] = patient_ids[
            min(selected_index, eligible_count - 1)
        ]
    return selected


def _sample_visit_types(
    patient_ids: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    visit_types = np.empty(
        len(patient_ids),
        dtype=object,
    )
    prior_count: dict[int, int] = {}
    first_options = np.array(
        [
            "new_patient_examination",
            "consultation",
            "emergency",
            "treatment",
        ],
        dtype=object,
    )
    first_probabilities = [0.55, 0.20, 0.15, 0.10]
    repeat_options = np.array(VISIT_TYPES, dtype=object)
    repeat_probabilities = [0.05, 0.20, 0.08, 0.42, 0.08, 0.17]

    for index, patient_id in enumerate(patient_ids):
        count = prior_count.get(int(patient_id), 0)
        if count == 0:
            visit_types[index] = rng.choice(
                first_options,
                p=first_probabilities,
            )
        else:
            visit_types[index] = rng.choice(
                repeat_options,
                p=repeat_probabilities,
            )
        prior_count[int(patient_id)] = count + 1

    return visit_types


def _assign_dentists(
    scheduled_start_at: pd.DatetimeIndex,
    visit_types: np.ndarray,
    dentists: pd.DataFrame,
    rng: Generator,
) -> np.ndarray:
    starts = dentists["start_date"].to_numpy(
        dtype="datetime64[ns]"
    )
    ends = dentists["end_date"].to_numpy(
        dtype="datetime64[ns]"
    )
    ids = dentists["dentist_id"].to_numpy(dtype=np.int64)
    roles = dentists["dentist_role"].to_numpy(dtype=object)
    hours = dentists["scheduled_hours_weekly"].to_numpy(
        dtype=float
    )

    assigned = np.empty(len(visit_types), dtype=np.int64)
    used_at_time: dict[np.datetime64, set[int]] = {}

    for index, (scheduled, visit_type) in enumerate(
        zip(
            scheduled_start_at.to_numpy(dtype="datetime64[ns]"),
            visit_types,
            strict=True,
        )
    ):
        compatible_roles = ROLE_COMPATIBILITY[
            str(visit_type)
        ]
        active = (
            (starts <= scheduled)
            & (
                np.isnat(ends)
                | (ends >= scheduled)
            )
            & np.isin(roles, compatible_roles)
        )
        candidate_ids = ids[active]
        candidate_hours = hours[active]
        if len(candidate_ids) == 0:
            raise ValueError(
                "No compatible dentist was active at "
                f"{scheduled} for {visit_type}"
            )

        busy = used_at_time.setdefault(
            scheduled,
            set(),
        )
        available = np.array(
            [
                dentist_id not in busy
                for dentist_id in candidate_ids
            ],
            dtype=bool,
        )
        if available.any():
            candidate_ids = candidate_ids[available]
            candidate_hours = candidate_hours[available]

        weights = candidate_hours / candidate_hours.sum()
        selected = int(
            rng.choice(
                candidate_ids,
                p=weights,
            )
        )
        assigned[index] = selected
        busy.add(selected)

    return assigned


def _sample_reminders(
    booked_at: pd.DatetimeIndex,
    scheduled_start_at: pd.DatetimeIndex,
    rng: Generator,
) -> tuple[np.ndarray, pd.Series]:
    size = len(scheduled_start_at)
    sent = rng.random(size) < 0.86
    base_hours = rng.choice(
        np.array([72, 48, 24, 12, 6], dtype=int),
        size=size,
        p=[0.20, 0.35, 0.25, 0.15, 0.05],
    )
    jitter = rng.integers(
        -120,
        121,
        size=size,
    )
    candidate = (
        scheduled_start_at
        - pd.to_timedelta(base_hours, unit="h")
        + pd.to_timedelta(jitter, unit="m")
    )
    earliest = booked_at + pd.Timedelta(minutes=30)
    adjusted = pd.DatetimeIndex(
        np.maximum(
            candidate.to_numpy(dtype="datetime64[ns]"),
            earliest.to_numpy(dtype="datetime64[ns]"),
        )
    )
    still_before_appointment = adjusted < scheduled_start_at
    sent &= np.asarray(still_before_appointment, dtype=bool)

    reminder_sent_at = pd.Series(
        pd.NaT,
        index=np.arange(size),
        dtype="datetime64[ns]",
    )
    reminder_sent_at.loc[sent] = adjusted.to_numpy()[sent]
    return sent, reminder_sent_at


def _logistic(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sample_changed_at(
    booked_at: pd.DatetimeIndex,
    scheduled_start_at: pd.DatetimeIndex,
    changed_mask: np.ndarray,
    rng: Generator,
) -> pd.Series:
    result = pd.Series(
        pd.NaT,
        index=np.arange(len(changed_mask)),
        dtype="datetime64[ns]",
    )
    count = int(changed_mask.sum())
    if count == 0:
        return result

    early = rng.random(count) < 0.60
    notice_hours = np.empty(count, dtype=np.int16)
    notice_hours[early] = rng.integers(
        24,
        14 * 24 + 1,
        size=int(early.sum()),
    )
    notice_hours[~early] = rng.integers(
        2,
        24,
        size=int((~early).sum()),
    )

    changed_schedule = pd.DatetimeIndex(
        scheduled_start_at.to_numpy()[changed_mask]
    )
    changed_booked = pd.DatetimeIndex(
        booked_at.to_numpy()[changed_mask]
    )
    changed_at = (
        changed_schedule
        - pd.to_timedelta(notice_hours, unit="h")
    )
    earliest = changed_booked + pd.Timedelta(minutes=15)
    changed_at = pd.DatetimeIndex(
        np.maximum(
            changed_at.to_numpy(dtype="datetime64[ns]"),
            earliest.to_numpy(dtype="datetime64[ns]"),
        )
    )
    latest = changed_schedule - pd.Timedelta(minutes=30)
    changed_at = pd.DatetimeIndex(
        np.minimum(
            changed_at.to_numpy(dtype="datetime64[ns]"),
            latest.to_numpy(dtype="datetime64[ns]"),
        )
    )

    result.loc[changed_mask] = changed_at.to_numpy()
    return result


def _generate_outcomes(
    *,
    config: BenchmarkConfig,
    patient_latent: _PatientLatentState,
    dentist_latent: _DentistLatentState,
    patient_ids: np.ndarray,
    dentist_ids: np.ndarray,
    booked_at: pd.DatetimeIndex,
    scheduled_start_at: pd.DatetimeIndex,
    lead_days: np.ndarray,
    visit_types: np.ndarray,
    booking_channels: np.ndarray,
    reminder_sent_at: pd.Series,
    outcomes_rng: Generator,
    status_rng: Generator,
) -> pd.DataFrame:
    size = len(patient_ids)
    patient_effect = patient_latent.risk_effect[
        patient_ids - 1
    ]
    dentist_effect = np.array(
        [
            dentist_latent.risk_effect[int(value)]
            for value in dentist_ids
        ],
        dtype=float,
    )

    prediction_time = (
        scheduled_start_at
        - pd.Timedelta(
            hours=config.prediction_horizon_hours
        )
    )
    reminder_known = (
        reminder_sent_at.notna().to_numpy()
        & (
            reminder_sent_at.to_numpy(
                dtype="datetime64[ns]"
            )
            <= prediction_time.to_numpy(
                dtype="datetime64[ns]"
            )
        )
    )

    cancellation_logit = (
        -2.55
        + 0.20 * (lead_days > 30)
        + 0.15 * (lead_days > 60)
        + 0.10 * patient_effect
    )
    cancellation_probability = _logistic(
        cancellation_logit
    )
    cancelled = (
        outcomes_rng.random(size)
        < cancellation_probability
    )

    reschedule_logit = (
        -3.05
        + 0.22 * (lead_days > 21)
        + 0.08 * patient_effect
    )
    reschedule_probability = _logistic(
        reschedule_logit
    )
    rescheduled = (
        (~cancelled)
        & (
            outcomes_rng.random(size)
            < reschedule_probability
        )
    )

    active_for_attendance = ~(cancelled | rescheduled)

    scheduled_hour = scheduled_start_at.hour.to_numpy()
    scheduled_weekday = (
        scheduled_start_at.dayofweek.to_numpy()
    )
    month = scheduled_start_at.month.to_numpy()
    seasonal_effect = (
        0.12
        * np.sin(
            2.0
            * np.pi
            * (month - 1)
            / 12.0
        )
    )

    visit_effect_map = {
        "new_patient_examination": 0.18,
        "recall_examination": -0.10,
        "consultation": 0.05,
        "treatment": -0.04,
        "emergency": -0.22,
        "follow_up": -0.16,
    }
    channel_effect_map = {
        "phone": 0.02,
        "in_person": -0.12,
        "online": 0.04,
        "referral": -0.10,
        "other": 0.18,
    }

    visit_effect = np.array(
        [
            visit_effect_map[str(value)]
            for value in visit_types
        ],
        dtype=float,
    )
    channel_effect = np.array(
        [
            channel_effect_map[str(value)]
            for value in booking_channels
        ],
        dtype=float,
    )

    no_show_logit = (
        -2.35
        + 0.92 * patient_effect
        + dentist_effect
        + 0.34 * (lead_days > 30)
        + 0.20 * (lead_days > 60)
        + 0.18 * (scheduled_hour >= 17)
        + 0.08 * np.isin(
            scheduled_weekday,
            [0, 5],
        )
        + seasonal_effect
        + visit_effect
        + channel_effect
        - 0.58 * reminder_known
    )
    no_show_probability = _logistic(no_show_logit)
    no_show = (
        active_for_attendance
        & (
            outcomes_rng.random(size)
            < no_show_probability
        )
    )
    completed = active_for_attendance & ~no_show

    status = np.full(
        size,
        "completed",
        dtype=object,
    )
    status[no_show] = "no_show"
    status[cancelled] = "cancelled"
    status[rescheduled] = "rescheduled"

    status_updated_at = pd.Series(
        pd.NaT,
        index=np.arange(size),
        dtype="datetime64[ns]",
    )
    check_in_at = status_updated_at.copy()
    chair_start_at = status_updated_at.copy()
    chair_end_at = status_updated_at.copy()
    checkout_at = status_updated_at.copy()
    status_change_reason = pd.Series(
        pd.NA,
        index=np.arange(size),
        dtype="string",
    )

    completed_count = int(completed.sum())
    if completed_count:
        completed_schedule = pd.DatetimeIndex(
            scheduled_start_at.to_numpy()[completed]
        )
        arrival_offset = np.clip(
            np.rint(
                status_rng.normal(
                    -5.0,
                    10.0,
                    size=completed_count,
                )
            ).astype(int),
            -30,
            30,
        )
        completed_check_in = (
            completed_schedule
            + pd.to_timedelta(
                arrival_offset,
                unit="m",
            )
        )
        waiting = np.clip(
            np.rint(
                status_rng.gamma(
                    2.0,
                    6.0,
                    size=completed_count,
                )
            ).astype(int),
            0,
            60,
        )
        start_candidate = (
            completed_check_in
            + pd.to_timedelta(
                waiting,
                unit="m",
            )
        )
        earliest_start = (
            completed_schedule
            - pd.Timedelta(minutes=10)
        )
        completed_chair_start = pd.DatetimeIndex(
            np.maximum(
                start_candidate.to_numpy(
                    dtype="datetime64[ns]"
                ),
                earliest_start.to_numpy(
                    dtype="datetime64[ns]"
                ),
            )
        )

        planned_duration = np.array(
            [
                DURATION_BY_VISIT[str(value)]
                for value in visit_types[completed]
            ],
            dtype=int,
        )
        duration_multiplier = status_rng.lognormal(
            mean=0.0,
            sigma=0.18,
            size=completed_count,
        )
        actual_duration = np.clip(
            np.rint(
                planned_duration * duration_multiplier
            ).astype(int),
            15,
            180,
        )
        completed_chair_end = (
            completed_chair_start
            + pd.to_timedelta(
                actual_duration,
                unit="m",
            )
        )
        completed_checkout = (
            completed_chair_end
            + pd.to_timedelta(
                status_rng.integers(
                    5,
                    21,
                    size=completed_count,
                ),
                unit="m",
            )
        )

        completed_indices = np.flatnonzero(completed)
        check_in_at.iloc[
            completed_indices
        ] = completed_check_in.to_numpy()
        chair_start_at.iloc[
            completed_indices
        ] = completed_chair_start.to_numpy()
        chair_end_at.iloc[
            completed_indices
        ] = completed_chair_end.to_numpy()
        checkout_at.iloc[
            completed_indices
        ] = completed_checkout.to_numpy()
        status_updated_at.iloc[
            completed_indices
        ] = completed_checkout.to_numpy()

    no_show_indices = np.flatnonzero(no_show)
    status_updated_at.iloc[no_show_indices] = (
        scheduled_start_at[no_show]
        + pd.Timedelta(minutes=15)
    ).to_numpy()

    changed = cancelled | rescheduled
    changed_at = _sample_changed_at(
        booked_at,
        scheduled_start_at,
        changed,
        status_rng,
    )
    changed_indices = np.flatnonzero(changed)
    status_updated_at.iloc[
        changed_indices
    ] = changed_at.iloc[
        changed_indices
    ].to_numpy()
    status_change_reason.iloc[
        changed_indices
    ] = status_rng.choice(
        np.array(
            [
                "patient_related",
                "clinic_related",
                "financial",
                "illness",
                "scheduling_conflict",
                "other",
            ],
            dtype=object,
        ),
        size=len(changed_indices),
        p=[0.30, 0.08, 0.12, 0.16, 0.28, 0.06],
    )

    return pd.DataFrame(
        {
            "status": status,
            "status_updated_at": status_updated_at,
            "check_in_at": check_in_at,
            "chair_start_at": chair_start_at,
            "chair_end_at": chair_end_at,
            "checkout_at": checkout_at,
            "status_change_reason": status_change_reason,
        }
    )


def _assign_reschedule_links(
    appointments: pd.DataFrame,
) -> pd.Series:
    links = pd.Series(
        pd.NA,
        index=appointments.index,
        dtype="Int64",
    )
    by_patient: dict[int, list[int]] = {}
    for index, patient_id in enumerate(
        appointments["patient_id"].to_numpy(dtype=int)
    ):
        by_patient.setdefault(
            int(patient_id),
            [],
        ).append(index)

    used_replacements: set[int] = set()
    for original_index in appointments.index[
        appointments["status"].eq("rescheduled")
    ]:
        patient_id = int(
            appointments.at[
                original_index,
                "patient_id",
            ]
        )
        original_schedule = appointments.at[
            original_index,
            "scheduled_start_at",
        ]
        original_changed_at = appointments.at[
            original_index,
            "status_updated_at",
        ]
        for candidate_index in by_patient[patient_id]:
            if candidate_index <= original_index:
                continue
            if candidate_index in used_replacements:
                continue
            if (
                appointments.at[
                    candidate_index,
                    "scheduled_start_at",
                ]
                <= original_schedule + pd.Timedelta(days=1)
            ):
                continue
            if (
                appointments.at[
                    candidate_index,
                    "booked_at",
                ]
                < original_changed_at
            ):
                continue
            links.iloc[candidate_index] = int(
                appointments.at[
                    original_index,
                    "appointment_id",
                ]
            )
            used_replacements.add(candidate_index)
            break
    return links


def _generate_appointments(
    config: BenchmarkConfig,
    *,
    patients: pd.DataFrame,
    dentists: pd.DataFrame,
    patient_latent: _PatientLatentState,
    dentist_latent: _DentistLatentState,
    appointments_rng: Generator,
    reminders_rng: Generator,
    outcomes_rng: Generator,
    status_rng: Generator,
) -> pd.DataFrame:
    scheduled_start_at = _sample_scheduled_start(
        config,
        appointments_rng,
    )
    booked_at, lead_days = _sample_booked_at(
        scheduled_start_at,
        appointments_rng,
    )
    patient_ids = _assign_patients(
        booked_at,
        patients,
        patient_latent,
        appointments_rng,
    )
    visit_types = _sample_visit_types(
        patient_ids,
        appointments_rng,
    )
    dentist_ids = _assign_dentists(
        scheduled_start_at,
        visit_types,
        dentists,
        appointments_rng,
    )
    booking_channels = appointments_rng.choice(
        np.array(BOOKING_CHANNELS, dtype=object),
        size=config.appointment_count,
        p=[0.42, 0.16, 0.25, 0.12, 0.05],
    )
    planned_duration = np.array(
        [
            DURATION_BY_VISIT[str(value)]
            for value in visit_types
        ],
        dtype=np.int16,
    )

    reminder_sent, reminder_sent_at = _sample_reminders(
        booked_at,
        scheduled_start_at,
        reminders_rng,
    )

    outcomes = _generate_outcomes(
        config=config,
        patient_latent=patient_latent,
        dentist_latent=dentist_latent,
        patient_ids=patient_ids,
        dentist_ids=dentist_ids,
        booked_at=booked_at,
        scheduled_start_at=scheduled_start_at,
        lead_days=lead_days,
        visit_types=visit_types,
        booking_channels=booking_channels,
        reminder_sent_at=reminder_sent_at,
        outcomes_rng=outcomes_rng,
        status_rng=status_rng,
    )

    appointments = pd.DataFrame(
        {
            "appointment_id": np.arange(
                1,
                config.appointment_count + 1,
                dtype=np.int64,
            ),
            "patient_id": patient_ids,
            "dentist_id": dentist_ids,
            "booked_at": booked_at,
            "scheduled_start_at": scheduled_start_at,
            "planned_duration_min": planned_duration,
            "visit_type": visit_types,
            "booking_channel": booking_channels,
            "status": outcomes["status"],
            "status_updated_at": outcomes[
                "status_updated_at"
            ],
            "reminder_sent": reminder_sent,
            "reminder_sent_at": reminder_sent_at,
            "check_in_at": outcomes["check_in_at"],
            "chair_start_at": outcomes[
                "chair_start_at"
            ],
            "chair_end_at": outcomes["chair_end_at"],
            "checkout_at": outcomes["checkout_at"],
            "status_change_reason": outcomes[
                "status_change_reason"
            ],
        }
    )
    appointments[
        "rescheduled_from_appointment_id"
    ] = _assign_reschedule_links(appointments)
    return appointments.loc[:, APPOINTMENT_COLUMNS]


def generate_synthetic_tables(
    config: BenchmarkConfig | None = None,
) -> SyntheticTables:
    """Generate all Version 2 raw tables and validate the public contract."""

    if config is None:
        config = load_benchmark_config()

    streams = create_rng_streams(config)
    patients, patient_latent = _generate_patients_and_latent(
        config,
        patients_rng=streams["patients"],
        latent_rng=streams["latent_risk"],
    )
    dentists, dentist_latent = _generate_dentists_and_latent(
        config,
        dentists_rng=streams["dentists"],
        latent_rng=streams["latent_risk"],
    )
    appointments = _generate_appointments(
        config,
        patients=patients,
        dentists=dentists,
        patient_latent=patient_latent,
        dentist_latent=dentist_latent,
        appointments_rng=streams["appointments"],
        reminders_rng=streams["reminders"],
        outcomes_rng=streams["outcomes"],
        status_rng=streams["status_timestamps"],
    )
    tables = SyntheticTables(
        patients=patients,
        dentists=dentists,
        appointments=appointments,
    )
    from src.synthetic.validation import validate_synthetic_tables

    validate_synthetic_tables(tables, config)
    return tables



__all__ = (
    "create_rng_streams",
    "generate_dentists",
    "generate_patients",
    "generate_synthetic_tables",
)
