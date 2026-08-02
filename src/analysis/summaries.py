"""Deterministic summaries for the mature supervised-training population."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np
import pandas as pd

from src.data import build_dataset as bd


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
_STRING_CATEGORICAL_COLUMNS = ("visit_type", "booking_channel")
_CALENDAR_CATEGORICAL_COLUMNS = (
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
_WEEKDAY_LEVELS = tuple(range(7))
_HOUR_LEVELS = tuple(range(24))
_MONTH_LEVELS = tuple(range(1, 13))
_SUPERVISED_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "target",
    *bd.FEATURE_COLUMNS,
)
_COHORT_TARGET_COLUMNS = (
    "rows",
    "positives",
    "negatives",
    "prevalence",
    "wilson_lower",
    "wilson_upper",
    "duplicate_appointment_ids",
)
_MISSINGNESS_COLUMNS = (
    "feature",
    "rows",
    "missing_count",
    "missing_rate",
    "non_missing_count",
    "unique_non_null",
    "is_constant",
)
_NUMERIC_COLUMNS = (
    "feature",
    "n",
    "missing_count",
    "missing_rate",
    "zero_count",
    "mean",
    "std",
    "min",
    "p01",
    "p05",
    "q1",
    "median",
    "q3",
    "p95",
    "p99",
    "max",
    "iqr",
    "lower_fence",
    "upper_fence",
    "below_fence_count",
    "above_fence_count",
)
_NUMERIC_BY_TARGET_COLUMNS = (
    "feature",
    "target",
    "n",
    "missing_count",
    "mean",
    "std",
    "min",
    "q1",
    "median",
    "q3",
    "max",
)
_CATEGORICAL_COLUMNS = (
    "feature",
    "level",
    "is_missing",
    "count",
    "share",
    "positives",
    "negatives",
    "no_show_rate",
    "wilson_lower",
    "wilson_upper",
    "is_rare",
    "has_high_uncertainty",
)
_WILSON_Z_95 = 1.959963984540054


def _validate_supervised_train(supervised_train: pd.DataFrame) -> None:
    """Require the exact target-bearing projection produced by EDA selection."""

    if not isinstance(supervised_train, pd.DataFrame):
        raise TypeError("supervised_train must be a pandas DataFrame")
    actual_columns = tuple(supervised_train.columns)
    if actual_columns != _SUPERVISED_COLUMNS:
        missing = [
            column for column in _SUPERVISED_COLUMNS if column not in actual_columns
        ]
        extra = [
            column for column in actual_columns if column not in _SUPERVISED_COLUMNS
        ]
        raise ValueError(
            "supervised_train columns and order must match the approved projection; "
            f"missing={missing}, extra={extra}"
        )

    required_non_null = ("appointment_id", "prediction_time", "target")
    missing_counts = supervised_train.loc[:, required_non_null].isna().sum()
    missing_required = missing_counts[missing_counts.gt(0)].to_dict()
    if missing_required:
        raise ValueError(
            "supervised_train required audit and target columns contain nulls: "
            f"{missing_required}"
        )
    if not supervised_train["target"].isin((0, 1)).all():
        raise ValueError("supervised_train target values must be binary 0 or 1")

    for column in NUMERIC_FEATURE_COLUMNS:
        series = supervised_train[column]
        dtype = series.dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(
            dtype
        ):
            raise ValueError(
                f"supervised_train.{column} must have a numeric non-Boolean dtype"
            )
        non_null_values = series.dropna().to_numpy(copy=True)
        if not np.isfinite(non_null_values).all():
            raise ValueError(
                f"supervised_train.{column} must contain only finite non-null values"
            )


def _wilson_interval(positives: int, rows: int) -> tuple[float, float]:
    if rows == 0:
        return np.nan, np.nan
    prevalence = positives / rows
    z_squared = _WILSON_Z_95**2
    denominator = 1.0 + z_squared / rows
    center = (prevalence + z_squared / (2.0 * rows)) / denominator
    radius = (
        _WILSON_Z_95
        * np.sqrt(
            prevalence * (1.0 - prevalence) / rows
            + z_squared / (4.0 * rows**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def _validate_categorical_roles(supervised_train: pd.DataFrame) -> None:
    for feature in _STRING_CATEGORICAL_COLUMNS:
        values = supervised_train[feature].dropna()
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(
                f"supervised_train.{feature} non-null values must be Python strings"
            )

    for feature in _CALENDAR_CATEGORICAL_COLUMNS:
        domain = _calendar_domain(feature)
        for value in supervised_train[feature].dropna():
            is_boolean = isinstance(value, (bool, np.bool_))
            is_numeric = isinstance(value, Real)
            is_finite = is_numeric and bool(np.isfinite(value))
            is_integer = is_finite and value == int(value)
            if is_boolean or not is_integer or int(value) not in domain:
                raise ValueError(
                    f"supervised_train.{feature} non-null values must be "
                    f"numeric non-Boolean integers in {domain[0]} through "
                    f"{domain[-1]}"
                )


def _calendar_domain(feature: str) -> tuple[int, ...]:
    if feature == "scheduled_weekday":
        return _WEEKDAY_LEVELS
    if feature == "scheduled_hour":
        return _HOUR_LEVELS
    if feature == "scheduled_month":
        return _MONTH_LEVELS
    raise ValueError(f"unsupported calendar feature: {feature}")


def _categorical_levels(
    feature: str,
    series: pd.Series,
) -> list[tuple[str, object, bool]]:
    if feature in _STRING_CATEGORICAL_COLUMNS:
        observed = sorted(series.dropna().unique().tolist())
    else:
        observed = list(_calendar_domain(feature))
    levels = [(str(value), value, False) for value in observed]
    if series.isna().any():
        levels.append(("<MISSING>", None, True))
    return levels


def _sorted_non_null(series: pd.Series) -> pd.Series:
    return series.dropna().sort_values(kind="mergesort")


def _quantiles(
    values: pd.Series,
    probabilities: Sequence[float],
) -> pd.Series:
    return values.quantile(list(probabilities), interpolation="linear")


def summarize_cohort_target(
    supervised_train: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mature-training cohort size, target balance, and duplicates."""

    _validate_supervised_train(supervised_train)
    rows = len(supervised_train)
    positives = int(supervised_train["target"].eq(1).sum())
    negatives = int(supervised_train["target"].eq(0).sum())
    prevalence = positives / rows if rows else np.nan
    wilson_lower, wilson_upper = _wilson_interval(positives, rows)
    duplicate_rows = int(
        supervised_train.duplicated("appointment_id", keep=False).sum()
    )
    return pd.DataFrame(
        [
            {
                "rows": rows,
                "positives": positives,
                "negatives": negatives,
                "prevalence": prevalence,
                "wilson_lower": wilson_lower,
                "wilson_upper": wilson_upper,
                "duplicate_appointment_ids": duplicate_rows,
            }
        ],
        columns=_COHORT_TARGET_COLUMNS,
    )


def summarize_missingness(
    supervised_train: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize missingness for the exact approved predictor allowlist."""

    _validate_supervised_train(supervised_train)
    rows = len(supervised_train)
    summaries: list[dict[str, object]] = []
    for feature in bd.FEATURE_COLUMNS:
        series = supervised_train[feature]
        missing_count = int(series.isna().sum())
        unique_non_null = int(series.nunique(dropna=True))
        summaries.append(
            {
                "feature": feature,
                "rows": rows,
                "missing_count": missing_count,
                "missing_rate": missing_count / rows if rows else np.nan,
                "non_missing_count": rows - missing_count,
                "unique_non_null": unique_non_null,
                "is_constant": unique_non_null == 1,
            }
        )
    return pd.DataFrame(summaries, columns=_MISSINGNESS_COLUMNS)


def summarize_numeric_features(
    supervised_train: pd.DataFrame,
) -> pd.DataFrame:
    """Return deterministic robust summaries for mature-training numerics."""

    _validate_supervised_train(supervised_train)
    rows = len(supervised_train)
    summaries: list[dict[str, object]] = []
    probabilities = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    for feature in NUMERIC_FEATURE_COLUMNS:
        values = _sorted_non_null(supervised_train[feature])
        n = len(values)
        missing_count = rows - n
        if n:
            quantiles = _quantiles(values, probabilities)
            q1 = float(quantiles.loc[0.25])
            q3 = float(quantiles.loc[0.75])
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            minimum = float(values.min())
            maximum = float(values.max())
            below_fence_count = int(values.lt(lower_fence).sum())
            above_fence_count = int(values.gt(upper_fence).sum())
        else:
            quantiles = pd.Series(np.nan, index=probabilities, dtype="float64")
            q1 = q3 = iqr = lower_fence = upper_fence = np.nan
            mean = std = minimum = maximum = np.nan
            below_fence_count = above_fence_count = 0
        summaries.append(
            {
                "feature": feature,
                "n": n,
                "missing_count": missing_count,
                "missing_rate": missing_count / rows if rows else np.nan,
                "zero_count": int(values.eq(0).sum()),
                "mean": mean,
                "std": std,
                "min": minimum,
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "q1": q1,
                "median": float(quantiles.loc[0.50]),
                "q3": q3,
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "max": maximum,
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "below_fence_count": below_fence_count,
                "above_fence_count": above_fence_count,
            }
        )
    return pd.DataFrame(summaries, columns=_NUMERIC_COLUMNS)


def summarize_numeric_by_target(
    supervised_train: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mature-training numerics within target 0 and target 1."""

    _validate_supervised_train(supervised_train)
    summaries: list[dict[str, object]] = []
    probabilities = (0.25, 0.50, 0.75)
    for feature in NUMERIC_FEATURE_COLUMNS:
        for target in (0, 1):
            group = supervised_train.loc[
                supervised_train["target"].eq(target), feature
            ]
            missing_count = int(group.isna().sum())
            values = _sorted_non_null(group)
            n = len(values)
            if n:
                quantiles = _quantiles(values, probabilities)
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if n > 1 else np.nan
                minimum = float(values.min())
                maximum = float(values.max())
            else:
                quantiles = pd.Series(
                    np.nan,
                    index=probabilities,
                    dtype="float64",
                )
                mean = std = minimum = maximum = np.nan
            summaries.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": n,
                    "missing_count": missing_count,
                    "mean": mean,
                    "std": std,
                    "min": minimum,
                    "q1": float(quantiles.loc[0.25]),
                    "median": float(quantiles.loc[0.50]),
                    "q3": float(quantiles.loc[0.75]),
                    "max": maximum,
                }
            )
    return pd.DataFrame(summaries, columns=_NUMERIC_BY_TARGET_COLUMNS)


def summarize_categorical_features(
    supervised_train: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize category frequencies and mature-training no-show rates."""

    _validate_supervised_train(supervised_train)
    _validate_categorical_roles(supervised_train)
    rows = len(supervised_train)
    target = supervised_train["target"]
    summaries: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURE_COLUMNS:
        series = supervised_train[feature]
        for level, value, is_missing in _categorical_levels(feature, series):
            if is_missing:
                mask = series.isna()
            else:
                mask = series.notna() & series.eq(value)
            count = int(mask.sum())
            positives = int(target.loc[mask].eq(1).sum())
            negatives = int(target.loc[mask].eq(0).sum())
            share = count / rows if rows else np.nan
            no_show_rate = positives / count if count else np.nan
            wilson_lower, wilson_upper = _wilson_interval(positives, count)
            summaries.append(
                {
                    "feature": feature,
                    "level": level,
                    "is_missing": is_missing,
                    "count": count,
                    "share": share,
                    "positives": positives,
                    "negatives": negatives,
                    "no_show_rate": no_show_rate,
                    "wilson_lower": wilson_lower,
                    "wilson_upper": wilson_upper,
                    "is_rare": count < 30 or share < 0.01,
                    "has_high_uncertainty": count < 30 or positives < 5,
                }
            )
    return pd.DataFrame(summaries, columns=_CATEGORICAL_COLUMNS)
