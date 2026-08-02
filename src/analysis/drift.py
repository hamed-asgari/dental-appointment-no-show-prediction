"""Deterministic feature-only numerical and categorical drift summaries."""

from __future__ import annotations

import math
from numbers import Real as _Real

import numpy as np
import pandas as pd

from src.analysis.summaries import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from src.data import build_dataset as bd


_DRIFT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    *bd.FEATURE_COLUMNS,
)
_NUMERIC_DRIFT_COLUMNS = (
    "feature",
    "train_rows",
    "validation_rows",
    "train_n",
    "validation_n",
    "train_missing_count",
    "validation_missing_count",
    "train_missing_rate",
    "validation_missing_rate",
    "missing_rate_difference",
    "train_mean",
    "validation_mean",
    "train_std",
    "validation_std",
    "signed_smd",
    "train_q10",
    "validation_q10",
    "q10_shift",
    "train_median",
    "validation_median",
    "median_shift",
    "train_q90",
    "validation_q90",
    "q90_shift",
)
_FLOAT_OUTPUT_COLUMNS = _NUMERIC_DRIFT_COLUMNS[7:]
_STRING_CATEGORICAL_COLUMNS = ("visit_type", "booking_channel")
_WEEKDAY_LEVELS = tuple(range(7))
_HOUR_LEVELS = tuple(range(24))
_MONTH_LEVELS = tuple(range(1, 13))
_MISSING_LEVEL = "<MISSING>"
_CATEGORICAL_DRIFT_LEVEL_COLUMNS = (
    "feature",
    "level",
    "is_missing",
    "train_count",
    "validation_count",
    "train_share",
    "validation_share",
    "share_difference",
    "absolute_share_difference",
    "contribution_to_total_variation",
    "is_unseen_in_train",
    "is_absent_in_validation",
)
_CATEGORICAL_DRIFT_FEATURE_COLUMNS = (
    "feature",
    "train_rows",
    "validation_rows",
    "train_missing_count",
    "validation_missing_count",
    "train_missing_rate",
    "validation_missing_rate",
    "missing_rate_difference",
    "train_distinct_nonmissing_levels",
    "validation_distinct_nonmissing_levels",
    "unseen_in_train_level_count",
    "unseen_in_train_validation_count",
    "unseen_in_train_validation_share",
    "absent_in_validation_level_count",
    "absent_in_validation_train_count",
    "absent_in_validation_train_share",
    "total_variation_distance",
    "max_absolute_share_difference",
)
_CATEGORICAL_LEVEL_FLOAT_COLUMNS = _CATEGORICAL_DRIFT_LEVEL_COLUMNS[5:10]
_CATEGORICAL_FEATURE_FLOAT_COLUMNS = (
    "train_missing_rate",
    "validation_missing_rate",
    "missing_rate_difference",
    "unseen_in_train_validation_share",
    "absent_in_validation_train_share",
    "total_variation_distance",
    "max_absolute_share_difference",
)
_FLOAT64_MAX = float(np.finfo("float64").max)


def _validate_drift_frame(frame: pd.DataFrame, *, frame_name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame")
    actual_columns = tuple(frame.columns)
    if actual_columns != _DRIFT_COLUMNS:
        missing = [column for column in _DRIFT_COLUMNS if column not in actual_columns]
        extra = [column for column in actual_columns if column not in _DRIFT_COLUMNS]
        raise ValueError(
            f"{frame_name} columns and order must match the approved projection; "
            f"missing={missing}, extra={extra}"
        )

    if frame["appointment_id"].isna().any():
        raise ValueError(f"{frame_name}.appointment_id must not contain nulls")
    if not frame["appointment_id"].is_unique:
        raise ValueError(f"{frame_name}.appointment_id must be unique")
    if frame["prediction_time"].isna().any():
        raise ValueError(f"{frame_name}.prediction_time must not contain nulls")
    if frame["prediction_time"].dtype != np.dtype("datetime64[ns]"):
        raise ValueError(
            f"{frame_name}.prediction_time must have timezone-naive "
            "datetime64[ns] dtype"
        )

    for feature in NUMERIC_FEATURE_COLUMNS:
        series = frame[feature]
        dtype = series.dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(
            dtype
        ):
            raise ValueError(
                f"{frame_name}.{feature} must have a numeric non-Boolean dtype"
            )
        if pd.api.types.is_complex_dtype(dtype):
            raise ValueError(
                f"{frame_name}.{feature} must have a real-valued, non-complex "
                "numeric dtype"
            )
        non_null_values = series.dropna().to_numpy(copy=True)
        if not np.isfinite(non_null_values).all():
            raise ValueError(
                f"{frame_name}.{feature} must contain only finite non-null values"
            )

    prediction_time = frame["prediction_time"]
    if frame_name == "train_drift":
        if prediction_time.ge(bd.VALIDATION_START).any():
            raise ValueError(
                "train_drift prediction_time values must precede VALIDATION_START"
            )
    elif (
        prediction_time.lt(bd.VALIDATION_START)
        | prediction_time.ge(bd.TEST_START)
    ).any():
        raise ValueError(
            "validation_drift prediction_time values must be at least "
            "VALIDATION_START and precede TEST_START"
        )


def _validate_drift_inputs(
    train_drift: pd.DataFrame,
    validation_drift: pd.DataFrame,
) -> None:
    _validate_drift_frame(train_drift, frame_name="train_drift")
    _validate_drift_frame(validation_drift, frame_name="validation_drift")
    overlap = train_drift["appointment_id"].isin(
        validation_drift["appointment_id"]
    )
    if overlap.any():
        overlapping_ids = train_drift.loc[overlap, "appointment_id"].tolist()
        raise ValueError(
            "train_drift and validation_drift appointment IDs must be disjoint; "
            f"overlap={overlapping_ids}"
        )


def _categorical_domain(feature: str) -> tuple[int, ...]:
    if feature == "scheduled_weekday":
        return _WEEKDAY_LEVELS
    if feature == "scheduled_hour":
        return _HOUR_LEVELS
    if feature == "scheduled_month":
        return _MONTH_LEVELS
    raise ValueError(f"unsupported fixed-domain categorical feature: {feature}")


def _validate_categorical_roles(frame: pd.DataFrame, *, frame_name: str) -> None:
    for feature in _STRING_CATEGORICAL_COLUMNS:
        values = frame[feature].dropna()
        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(
                f"{frame_name}.{feature} non-null values must be Python strings"
            )

    for feature in CATEGORICAL_FEATURE_COLUMNS:
        if feature in _STRING_CATEGORICAL_COLUMNS:
            continue
        domain = _categorical_domain(feature)
        for value in frame[feature].dropna():
            is_boolean = isinstance(value, (bool, np.bool_))
            is_numeric = isinstance(value, _Real)
            is_finite = is_numeric and bool(np.isfinite(value))
            is_integer = is_finite and value == int(value)
            if is_boolean or not is_integer or int(value) not in domain:
                raise ValueError(
                    f"{frame_name}.{feature} non-null values must be numeric "
                    f"non-Boolean integers in {domain[0]} through {domain[-1]}"
                )


def _categorical_level_universe(
    feature: str,
    train: pd.Series,
    validation: pd.Series,
) -> list[tuple[str, object, bool]]:
    if feature in _STRING_CATEGORICAL_COLUMNS:
        observed = sorted(
            set(train.dropna().tolist()) | set(validation.dropna().tolist())
        )
    else:
        observed = list(_categorical_domain(feature))
    levels = [(str(value), value, False) for value in observed]
    levels.append((_MISSING_LEVEL, None, True))
    return levels


def _require_no_infinite_output(
    result: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    context: str,
) -> None:
    floating_values = result.loc[:, columns].to_numpy(
        dtype="float64",
        copy=True,
    )
    if np.isinf(floating_values).any():
        raise ValueError(f"{context} output must not contain infinity")


def _build_categorical_drift_levels(
    train_drift: pd.DataFrame,
    validation_drift: pd.DataFrame,
) -> pd.DataFrame:
    _validate_drift_inputs(train_drift, validation_drift)
    _validate_categorical_roles(train_drift, frame_name="train_drift")
    _validate_categorical_roles(
        validation_drift,
        frame_name="validation_drift",
    )
    train_rows = len(train_drift)
    validation_rows = len(validation_drift)
    summaries: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURE_COLUMNS:
        train = train_drift[feature]
        validation = validation_drift[feature]
        universe = _categorical_level_universe(feature, train, validation)
        for level, value, is_missing in universe:
            if is_missing:
                train_mask = train.isna()
                validation_mask = validation.isna()
            else:
                train_mask = train.notna() & train.eq(value)
                validation_mask = validation.notna() & validation.eq(value)
            train_count = int(train_mask.sum())
            validation_count = int(validation_mask.sum())
            train_share = train_count / train_rows if train_rows else np.nan
            validation_share = (
                validation_count / validation_rows
                if validation_rows
                else np.nan
            )
            share_difference = _safe_difference(
                validation_share,
                train_share,
            )
            absolute_share_difference = (
                abs(share_difference)
                if math.isfinite(share_difference)
                else np.nan
            )
            contribution = (
                0.5 * absolute_share_difference
                if math.isfinite(absolute_share_difference)
                else np.nan
            )
            summaries.append(
                {
                    "feature": feature,
                    "level": level,
                    "is_missing": is_missing,
                    "train_count": train_count,
                    "validation_count": validation_count,
                    "train_share": train_share,
                    "validation_share": validation_share,
                    "share_difference": share_difference,
                    "absolute_share_difference": absolute_share_difference,
                    "contribution_to_total_variation": contribution,
                    "is_unseen_in_train": (
                        not is_missing
                        and train_count == 0
                        and validation_count > 0
                    ),
                    "is_absent_in_validation": (
                        not is_missing
                        and train_count > 0
                        and validation_count == 0
                    ),
                }
            )
    result = pd.DataFrame(
        summaries,
        columns=_CATEGORICAL_DRIFT_LEVEL_COLUMNS,
    )
    _require_no_infinite_output(
        result,
        _CATEGORICAL_LEVEL_FLOAT_COLUMNS,
        context="categorical drift level",
    )
    return result


def _sorted_real_values(series: pd.Series) -> np.ndarray:
    values = series.dropna().to_numpy(dtype="float64", copy=True)
    values.sort(kind="mergesort")
    return values


def _maximum_absolute_value(values: np.ndarray) -> float:
    return max((abs(float(value)) for value in values), default=0.0)


def _stable_mean(values: np.ndarray) -> float:
    n = len(values)
    if not n:
        return np.nan
    minimum = float(values[0])
    maximum = float(values[-1])
    anchor = math.fsum((minimum / 2.0, maximum / 2.0))
    residual_mean = math.fsum(
        (float(value) - anchor) / n for value in values
    )
    mean = math.fsum((anchor, residual_mean))
    if not math.isfinite(mean):
        raise ValueError("stable arithmetic mean must be finite")
    return float(mean)


def _stable_sample_std(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return np.nan
    scale = _maximum_absolute_value(values)
    if scale == 0.0:
        return 0.0
    normalized = tuple(float(value) / scale for value in values)
    normalized_mean = math.fsum(normalized) / n
    sum_squared_deviations = math.fsum(
        (value - normalized_mean) ** 2 for value in normalized
    )
    normalized_std = math.sqrt(sum_squared_deviations / (n - 1))
    if normalized_std > _FLOAT64_MAX / scale:
        return np.nan
    standard_deviation = normalized_std * scale
    return (
        float(standard_deviation)
        if math.isfinite(standard_deviation)
        else np.nan
    )


def _stable_linear_quantile(values: np.ndarray, probability: float) -> float:
    n = len(values)
    if not n:
        return np.nan
    position = (n - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = float(values[lower_index])
    if lower_index == upper_index:
        return lower
    upper = float(values[upper_index])
    fraction = position - lower_index
    try:
        quantile = math.fsum(
            ((1.0 - fraction) * lower, fraction * upper)
        )
    except OverflowError:
        return np.nan
    return float(quantile) if math.isfinite(quantile) else np.nan


def _safe_difference(minuend: float, subtrahend: float) -> float:
    if not math.isfinite(minuend) or not math.isfinite(subtrahend):
        return np.nan
    opposite_signs = (minuend > 0.0 > subtrahend) or (
        minuend < 0.0 < subtrahend
    )
    if opposite_signs and abs(minuend) > _FLOAT64_MAX - abs(subtrahend):
        return np.nan
    difference = minuend - subtrahend
    return float(difference) if math.isfinite(difference) else np.nan


def _describe_numeric(series: pd.Series, rows: int) -> dict[str, float | int]:
    values = _sorted_real_values(series)
    n = len(values)
    missing_count = rows - n
    return {
        "n": n,
        "missing_count": missing_count,
        "missing_rate": missing_count / rows if rows else np.nan,
        "mean": _stable_mean(values),
        "std": _stable_sample_std(values),
        "q10": _stable_linear_quantile(values, 0.10),
        "median": _stable_linear_quantile(values, 0.50),
        "q90": _stable_linear_quantile(values, 0.90),
    }


def _signed_standardized_mean_difference(
    train_mean: float,
    validation_mean: float,
    train_std: float,
    validation_std: float,
) -> float:
    if not all(
        np.isfinite(value)
        for value in (train_mean, validation_mean, train_std, validation_std)
    ):
        return np.nan
    standard_deviation_scale = max(abs(train_std), abs(validation_std))
    if standard_deviation_scale == 0.0:
        difference = _safe_difference(validation_mean, train_mean)
        return 0.0 if difference == 0.0 else np.nan
    normalized_train_std = train_std / standard_deviation_scale
    normalized_validation_std = validation_std / standard_deviation_scale
    normalized_pooled_scale = math.sqrt(
        (
            normalized_train_std**2
            + normalized_validation_std**2
        )
        / 2.0
    )
    pooled_scale = normalized_pooled_scale * standard_deviation_scale
    difference = _safe_difference(validation_mean, train_mean)
    if math.isfinite(difference):
        signed_smd = difference / pooled_scale
        return float(signed_smd) if math.isfinite(signed_smd) else np.nan

    mean_scale = max(abs(train_mean), abs(validation_mean))
    normalized_difference = math.fsum(
        (validation_mean / mean_scale, -train_mean / mean_scale)
    )
    difference_mantissa, difference_exponent = math.frexp(
        normalized_difference
    )
    mean_mantissa, mean_exponent = math.frexp(mean_scale)
    denominator_mantissa, denominator_exponent = math.frexp(pooled_scale)
    try:
        signed_smd = math.ldexp(
            difference_mantissa * mean_mantissa / denominator_mantissa,
            difference_exponent + mean_exponent - denominator_exponent,
        )
    except OverflowError:
        return np.nan
    return float(signed_smd) if np.isfinite(signed_smd) else np.nan


def summarize_numeric_drift(
    train_drift: pd.DataFrame,
    validation_drift: pd.DataFrame,
) -> pd.DataFrame:
    """Compare train and validation numerical predictor distributions."""

    _validate_drift_inputs(train_drift, validation_drift)
    train_rows = len(train_drift)
    validation_rows = len(validation_drift)
    summaries: list[dict[str, object]] = []
    for feature in NUMERIC_FEATURE_COLUMNS:
        train = _describe_numeric(train_drift[feature], train_rows)
        validation = _describe_numeric(
            validation_drift[feature],
            validation_rows,
        )
        summaries.append(
            {
                "feature": feature,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "train_n": train["n"],
                "validation_n": validation["n"],
                "train_missing_count": train["missing_count"],
                "validation_missing_count": validation["missing_count"],
                "train_missing_rate": train["missing_rate"],
                "validation_missing_rate": validation["missing_rate"],
                "missing_rate_difference": _safe_difference(
                    validation["missing_rate"],
                    train["missing_rate"],
                ),
                "train_mean": train["mean"],
                "validation_mean": validation["mean"],
                "train_std": train["std"],
                "validation_std": validation["std"],
                "signed_smd": _signed_standardized_mean_difference(
                    train["mean"],
                    validation["mean"],
                    train["std"],
                    validation["std"],
                ),
                "train_q10": train["q10"],
                "validation_q10": validation["q10"],
                "q10_shift": _safe_difference(
                    validation["q10"],
                    train["q10"],
                ),
                "train_median": train["median"],
                "validation_median": validation["median"],
                "median_shift": _safe_difference(
                    validation["median"],
                    train["median"],
                ),
                "train_q90": train["q90"],
                "validation_q90": validation["q90"],
                "q90_shift": _safe_difference(
                    validation["q90"],
                    train["q90"],
                ),
            }
        )
    result = pd.DataFrame(summaries, columns=_NUMERIC_DRIFT_COLUMNS)
    floating_values = result.loc[:, _FLOAT_OUTPUT_COLUMNS].to_numpy(
        dtype="float64",
        copy=True,
    )
    if np.isinf(floating_values).any():
        raise ValueError("numeric drift output must not contain infinity")
    return result


def summarize_categorical_drift_levels(
    train_drift: pd.DataFrame,
    validation_drift: pd.DataFrame,
) -> pd.DataFrame:
    """Compare train and validation categorical predictor levels."""

    return _build_categorical_drift_levels(train_drift, validation_drift)


def summarize_categorical_drift_features(
    train_drift: pd.DataFrame,
    validation_drift: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate categorical drift by feature using total variation."""

    levels = _build_categorical_drift_levels(train_drift, validation_drift)
    train_rows = len(train_drift)
    validation_rows = len(validation_drift)
    summaries: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURE_COLUMNS:
        feature_levels = levels.loc[levels["feature"].eq(feature)]
        nonmissing = feature_levels.loc[~feature_levels["is_missing"]]
        missing = feature_levels.loc[feature_levels["is_missing"]].iloc[0]
        unseen = nonmissing.loc[nonmissing["is_unseen_in_train"]]
        absent = nonmissing.loc[nonmissing["is_absent_in_validation"]]
        train_missing_count = int(missing["train_count"])
        validation_missing_count = int(missing["validation_count"])
        train_missing_rate = (
            train_missing_count / train_rows if train_rows else np.nan
        )
        validation_missing_rate = (
            validation_missing_count / validation_rows
            if validation_rows
            else np.nan
        )
        unseen_validation_count = int(unseen["validation_count"].sum())
        absent_train_count = int(absent["train_count"].sum())
        if train_rows and validation_rows:
            total_variation_distance = math.fsum(
                feature_levels["contribution_to_total_variation"]
            )
            max_absolute_share_difference = float(
                feature_levels["absolute_share_difference"].max()
            )
        else:
            total_variation_distance = np.nan
            max_absolute_share_difference = np.nan
        summaries.append(
            {
                "feature": feature,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "train_missing_count": train_missing_count,
                "validation_missing_count": validation_missing_count,
                "train_missing_rate": train_missing_rate,
                "validation_missing_rate": validation_missing_rate,
                "missing_rate_difference": _safe_difference(
                    validation_missing_rate,
                    train_missing_rate,
                ),
                "train_distinct_nonmissing_levels": int(
                    nonmissing["train_count"].gt(0).sum()
                ),
                "validation_distinct_nonmissing_levels": int(
                    nonmissing["validation_count"].gt(0).sum()
                ),
                "unseen_in_train_level_count": len(unseen),
                "unseen_in_train_validation_count": unseen_validation_count,
                "unseen_in_train_validation_share": (
                    unseen_validation_count / validation_rows
                    if validation_rows
                    else np.nan
                ),
                "absent_in_validation_level_count": len(absent),
                "absent_in_validation_train_count": absent_train_count,
                "absent_in_validation_train_share": (
                    absent_train_count / train_rows if train_rows else np.nan
                ),
                "total_variation_distance": total_variation_distance,
                "max_absolute_share_difference": (
                    max_absolute_share_difference
                ),
            }
        )
    result = pd.DataFrame(
        summaries,
        columns=_CATEGORICAL_DRIFT_FEATURE_COLUMNS,
    )
    _require_no_infinite_output(
        result,
        _CATEGORICAL_FEATURE_FLOAT_COLUMNS,
        context="categorical drift feature",
    )
    return result
