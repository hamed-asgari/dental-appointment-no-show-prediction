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
_MATURITY_AUDIT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
    "development_fit_eligible",
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
_TEMPORAL_COVERAGE_COLUMNS = (
    "nominal_train_rows",
    "mature_train_rows",
    "maturity_exclusion_rows",
    "nominal_prediction_time_min",
    "nominal_prediction_time_max",
    "mature_prediction_time_min",
    "mature_prediction_time_max",
    "first_prediction_month",
    "last_prediction_month",
    "calendar_months_spanned",
)
_TEMPORAL_MONTHLY_COLUMNS = (
    "prediction_month",
    "nominal_train_count",
    "mature_train_count",
    "maturity_exclusion_count",
    "positives",
    "negatives",
    "no_show_rate",
    "wilson_lower",
    "wilson_upper",
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


def _validate_maturity_audit(maturity_audit: pd.DataFrame) -> None:
    if not isinstance(maturity_audit, pd.DataFrame):
        raise TypeError("maturity_audit must be a pandas DataFrame")
    actual_columns = tuple(maturity_audit.columns)
    if actual_columns != _MATURITY_AUDIT_COLUMNS:
        missing = [
            column for column in _MATURITY_AUDIT_COLUMNS if column not in actual_columns
        ]
        extra = [
            column for column in actual_columns if column not in _MATURITY_AUDIT_COLUMNS
        ]
        raise ValueError(
            "maturity_audit columns and order must match the approved projection; "
            f"missing={missing}, extra={extra}"
        )

    for column in _MATURITY_AUDIT_COLUMNS:
        if maturity_audit[column].isna().any():
            raise ValueError(f"maturity_audit.{column} must not contain nulls")
    if not maturity_audit["appointment_id"].is_unique:
        raise ValueError("maturity_audit.appointment_id must be unique")
    if maturity_audit["prediction_time"].dtype != np.dtype("datetime64[ns]"):
        raise ValueError(
            "maturity_audit.prediction_time must have timezone-naive "
            "datetime64[ns] dtype"
        )
    if not maturity_audit["split"].eq("train").all():
        raise ValueError("maturity_audit.split values must all be exactly train")
    if not pd.api.types.is_bool_dtype(
        maturity_audit["development_fit_eligible"].dtype
    ):
        raise ValueError(
            "maturity_audit.development_fit_eligible must have Boolean dtype"
        )


def _validate_temporal_inputs(
    supervised_train: pd.DataFrame,
    maturity_audit: pd.DataFrame,
) -> None:
    _validate_supervised_train(supervised_train)
    _validate_maturity_audit(maturity_audit)
    if not supervised_train["appointment_id"].is_unique:
        raise ValueError("supervised_train.appointment_id must be unique")
    if supervised_train["prediction_time"].dtype != np.dtype("datetime64[ns]"):
        raise ValueError(
            "supervised_train.prediction_time must have timezone-naive "
            "datetime64[ns] dtype"
        )

    eligible_audit = maturity_audit.loc[
        maturity_audit["development_fit_eligible"],
        ["appointment_id", "prediction_time"],
    ]
    supervised_ids = supervised_train["appointment_id"]
    eligible_ids = eligible_audit["appointment_id"]
    absent_from_audit = supervised_ids.loc[~supervised_ids.isin(eligible_ids)].tolist()
    absent_from_supervised = eligible_ids.loc[
        ~eligible_ids.isin(supervised_ids)
    ].tolist()
    if absent_from_audit or absent_from_supervised:
        raise ValueError(
            "supervised and eligible maturity-audit appointment IDs must match "
            f"exactly; absent_from_audit={absent_from_audit}, "
            f"absent_from_supervised={absent_from_supervised}"
        )
    if len(supervised_train) != len(eligible_audit):
        raise ValueError(
            "supervised row count must equal eligible maturity-audit row count"
        )

    supervised_times = supervised_train.set_index("appointment_id")[
        "prediction_time"
    ]
    eligible_times = eligible_audit.set_index("appointment_id")["prediction_time"]
    aligned_eligible_times = eligible_times.reindex(supervised_times.index)
    mismatched_times = supervised_times.ne(aligned_eligible_times)
    if mismatched_times.any():
        mismatched_ids = supervised_times.index[mismatched_times].tolist()
        raise ValueError(
            "prediction_time must match for every eligible appointment; "
            f"mismatched appointment IDs={mismatched_ids}"
        )


def _prediction_month_range(prediction_time: pd.Series) -> pd.PeriodIndex:
    if prediction_time.empty:
        return pd.PeriodIndex([], freq="M")
    first_month = prediction_time.min().to_period("M")
    last_month = prediction_time.max().to_period("M")
    return pd.period_range(first_month, last_month, freq="M")


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


def summarize_temporal_coverage(
    supervised_train: pd.DataFrame,
    maturity_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize nominal and mature train coverage over prediction time."""

    _validate_temporal_inputs(supervised_train, maturity_audit)
    nominal_rows = len(maturity_audit)
    mature_rows = len(supervised_train)
    months = _prediction_month_range(maturity_audit["prediction_time"])
    if len(months):
        first_prediction_month: object = months[0].strftime("%Y-%m")
        last_prediction_month: object = months[-1].strftime("%Y-%m")
    else:
        first_prediction_month = pd.NA
        last_prediction_month = pd.NA
    result = pd.DataFrame(
        [
            {
                "nominal_train_rows": nominal_rows,
                "mature_train_rows": mature_rows,
                "maturity_exclusion_rows": nominal_rows - mature_rows,
                "nominal_prediction_time_min": maturity_audit[
                    "prediction_time"
                ].min(),
                "nominal_prediction_time_max": maturity_audit[
                    "prediction_time"
                ].max(),
                "mature_prediction_time_min": supervised_train[
                    "prediction_time"
                ].min(),
                "mature_prediction_time_max": supervised_train[
                    "prediction_time"
                ].max(),
                "first_prediction_month": first_prediction_month,
                "last_prediction_month": last_prediction_month,
                "calendar_months_spanned": len(months),
            }
        ],
        columns=_TEMPORAL_COVERAGE_COLUMNS,
    )
    for column in (
        "nominal_prediction_time_min",
        "nominal_prediction_time_max",
        "mature_prediction_time_min",
        "mature_prediction_time_max",
    ):
        result[column] = result[column].astype("datetime64[ns]")
    for column in ("first_prediction_month", "last_prediction_month"):
        result[column] = result[column].astype("string")
    return result


def summarize_temporal_monthly(
    supervised_train: pd.DataFrame,
    maturity_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize train maturity and mature outcomes by prediction month."""

    _validate_temporal_inputs(supervised_train, maturity_audit)
    months = _prediction_month_range(maturity_audit["prediction_time"])
    if not len(months):
        return pd.DataFrame(
            {
                "prediction_month": pd.Series(dtype="string"),
                "nominal_train_count": pd.Series(dtype="int64"),
                "mature_train_count": pd.Series(dtype="int64"),
                "maturity_exclusion_count": pd.Series(dtype="int64"),
                "positives": pd.Series(dtype="int64"),
                "negatives": pd.Series(dtype="int64"),
                "no_show_rate": pd.Series(dtype="float64"),
                "wilson_lower": pd.Series(dtype="float64"),
                "wilson_upper": pd.Series(dtype="float64"),
            },
            columns=_TEMPORAL_MONTHLY_COLUMNS,
        )

    audit_months = maturity_audit["prediction_time"].dt.to_period("M")
    supervised_months = supervised_train["prediction_time"].dt.to_period("M")
    summaries: list[dict[str, object]] = []
    for month in months:
        nominal_count = int(audit_months.eq(month).sum())
        mature_mask = supervised_months.eq(month)
        mature_count = int(mature_mask.sum())
        positives = int(
            supervised_train.loc[mature_mask, "target"].eq(1).sum()
        )
        negatives = int(
            supervised_train.loc[mature_mask, "target"].eq(0).sum()
        )
        exclusion_count = nominal_count - mature_count
        if mature_count != positives + negatives or exclusion_count < 0:
            raise ValueError("monthly temporal counts failed reconciliation")
        no_show_rate = positives / mature_count if mature_count else np.nan
        wilson_lower, wilson_upper = _wilson_interval(positives, mature_count)
        summaries.append(
            {
                "prediction_month": month.strftime("%Y-%m"),
                "nominal_train_count": nominal_count,
                "mature_train_count": mature_count,
                "maturity_exclusion_count": exclusion_count,
                "positives": positives,
                "negatives": negatives,
                "no_show_rate": no_show_rate,
                "wilson_lower": wilson_lower,
                "wilson_upper": wilson_upper,
            }
        )
    result = pd.DataFrame(summaries, columns=_TEMPORAL_MONTHLY_COLUMNS)
    result["prediction_month"] = result["prediction_month"].astype("string")
    return result
