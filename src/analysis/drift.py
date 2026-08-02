"""Deterministic feature-only numerical drift summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analysis.summaries import NUMERIC_FEATURE_COLUMNS
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
