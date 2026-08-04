"""Build leakage-safe development and validation modeling data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build_dataset import (
    PREDICTION_HORIZON_HOURS as _PREDICTION_HORIZON_HOURS,
    TEST_START as _TEST_START,
    VALIDATION_START as _VALIDATION_START,
    _validate_canonical_structure as _validate_canonical_structure,
)


__all__ = (
    "NUMERIC_FEATURE_COLUMNS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "build_development_modeling_data",
)

NUMERIC_FEATURE_COLUMNS = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)

CATEGORICAL_FEATURE_COLUMNS = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)

_FEATURE_COLUMNS = (
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
)

_NUMERIC_STORAGE_FEATURE_COLUMNS = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)

_ORDER_COLUMNS = ("prediction_time", "appointment_id")
_ALLOWED_SPLITS = frozenset(("train", "validation", "test"))


def _validate_predictor_domains(canonical: pd.DataFrame) -> None:
    numeric_values = canonical.loc[
        :, _NUMERIC_STORAGE_FEATURE_COLUMNS
    ].to_numpy()
    if not np.isfinite(numeric_values).all():
        raise ValueError("Numerical predictors must contain only finite values")

    valid_domains = {
        "planned_duration_min": canonical["planned_duration_min"].gt(0),
        "booking_lead_time_hours": canonical[
            "booking_lead_time_hours"
        ].ge(_PREDICTION_HORIZON_HOURS),
        "scheduled_weekday": canonical["scheduled_weekday"].between(0, 6),
        "scheduled_hour": canonical["scheduled_hour"].between(0, 23),
        "scheduled_month": canonical["scheduled_month"].between(1, 12),
        "approximate_age_at_prediction": canonical[
            "approximate_age_at_prediction"
        ].ge(0),
        "patient_registration_tenure_days": canonical[
            "patient_registration_tenure_days"
        ].ge(0),
        "dentist_tenure_days": canonical["dentist_tenure_days"].ge(0),
    }
    for feature_name, valid in valid_domains.items():
        if not valid.all():
            raise ValueError(
                f"{feature_name} contains values outside its valid range"
            )


def _validate_temporal_splits(canonical: pd.DataFrame) -> None:
    prediction_time = canonical["prediction_time"]
    expected_split = pd.Series(
        pd.NA,
        index=canonical.index,
        dtype="string",
        name="split",
    )
    expected_split.loc[prediction_time.lt(_VALIDATION_START)] = "train"
    expected_split.loc[
        prediction_time.ge(_VALIDATION_START)
        & prediction_time.lt(_TEST_START)
    ] = "validation"
    expected_split.loc[prediction_time.ge(_TEST_START)] = "test"
    if not canonical["split"].equals(expected_split):
        raise ValueError(
            "split labels must match the authoritative prediction-time boundaries"
        )


def _validate_canonical(
    canonical: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    if type(canonical) is not pd.DataFrame:
        raise TypeError("canonical must be an exact pandas DataFrame")

    _validate_canonical_structure(canonical)

    if canonical.isna().any().any():
        missing = canonical.isna().sum()
        missing_columns = missing[missing.gt(0)].index.tolist()
        raise ValueError(
            "Canonical dataset contains null required values in "
            f"{missing_columns}"
        )
    index = canonical.index
    if (
        type(index) is not pd.RangeIndex
        or index.start != 0
        or index.step != 1
        or index.stop != len(canonical)
    ):
        raise ValueError("Canonical dataset index must be a zero-based RangeIndex")
    if not canonical["appointment_id"].is_unique:
        raise ValueError("appointment_id must be unique in the canonical dataset")

    target_values = set(canonical["target"].unique())
    if target_values != {0, 1}:
        raise ValueError("target values must be exactly 0 and 1")

    split_values = set(canonical["split"].unique())
    if not split_values.issubset(_ALLOWED_SPLITS):
        raise ValueError("split values must be limited to train, validation, and test")

    _validate_predictor_domains(canonical)

    expected_order = canonical.loc[:, _ORDER_COLUMNS].sort_values(
        list(_ORDER_COLUMNS), kind="mergesort"
    )
    if not canonical.loc[:, _ORDER_COLUMNS].equals(expected_order):
        raise ValueError(
            "Canonical dataset must be ordered by prediction_time then "
            "appointment_id"
        )

    _validate_temporal_splits(canonical)

    split = canonical["split"]
    development_eligible = canonical["development_fit_eligible"]

    if (development_eligible & ~split.eq("train")).any():
        raise ValueError("Development-fit eligibility is limited to train rows")
    if (
        development_eligible & ~canonical["pretest_fit_eligible"]
    ).any():
        raise ValueError(
            "Development-fit eligibility requires pretest-fit eligibility"
        )
    if (
        canonical["pretest_fit_eligible"]
        & ~split.isin(("train", "validation"))
    ).any():
        raise ValueError(
            "Pretest-fit eligibility is limited to train or validation rows"
        )

    development_mask = split.eq("train") & development_eligible
    validation_mask = split.eq("validation")
    if not development_mask.any():
        raise ValueError("Development population must not be empty")
    if set(canonical.loc[development_mask, "target"].unique()) != {0, 1}:
        raise ValueError("Development target must contain both classes")
    if not validation_mask.any():
        raise ValueError("Validation population must not be empty")

    return development_mask, validation_mask


def build_development_modeling_data(
    canonical: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Return fresh, aligned development and validation modeling objects."""

    development_mask, validation_mask = _validate_canonical(canonical)

    development_features = canonical.loc[
        development_mask, _FEATURE_COLUMNS
    ].copy(deep=True)
    development_target = canonical.loc[development_mask, "target"].copy(
        deep=True
    )
    validation_features = canonical.loc[
        validation_mask, _FEATURE_COLUMNS
    ].copy(deep=True)
    validation_target = canonical.loc[validation_mask, "target"].copy(
        deep=True
    )

    return {
        "development_features": development_features,
        "development_target": development_target,
        "validation_features": validation_features,
        "validation_target": validation_target,
    }
