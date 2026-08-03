"""Build deterministic in-memory figures from approved EDA tables."""

from __future__ import annotations

from itertools import combinations as _combinations
from numbers import Real as _Real
import re as _re

from matplotlib import colormaps as _colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter as _PercentFormatter
import numpy as np
import pandas as pd

from src.analysis.artifacts import _validate_tables
from src.analysis.summaries import (
    CATEGORICAL_FEATURE_COLUMNS as _CATEGORICAL_FEATURE_COLUMNS,
)
from src.analysis.summaries import (
    NUMERIC_FEATURE_COLUMNS as _NUMERIC_FEATURE_COLUMNS,
)


__all__ = ("build_eda_figures",)


_FIGURE_KEYS = (
    "class_balance",
    "temporal_monthly",
    "numeric_drift",
    "categorical_drift",
    "numeric_relationships",
)
_SOURCE_SCHEMAS = (
    ("cohort_target", (
        "rows",
        "positives",
        "negatives",
        "prevalence",
        "wilson_lower",
        "wilson_upper",
        "duplicate_appointment_ids",
    )),
    ("temporal_monthly", (
        "prediction_month",
        "nominal_train_count",
        "mature_train_count",
        "maturity_exclusion_count",
        "positives",
        "negatives",
        "no_show_rate",
        "wilson_lower",
        "wilson_upper",
    )),
    ("numeric_drift", (
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
    )),
    ("categorical_drift_features", (
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
    )),
    ("numeric_relationships", (
        "feature_a",
        "feature_b",
        "train_rows",
        "paired_n",
        "missing_pair_count",
        "paired_rate",
        "feature_a_unique_n",
        "feature_b_unique_n",
        "pearson_correlation",
        "absolute_pearson_correlation",
        "spearman_correlation",
        "absolute_spearman_correlation",
    )),
)
_EXPECTED_RELATIONSHIP_PAIRS = tuple(
    _combinations(_NUMERIC_FEATURE_COLUMNS, 2)
)
_TEMPORAL_COUNT_COLUMNS = (
    "nominal_train_count",
    "mature_train_count",
    "maturity_exclusion_count",
    "positives",
    "negatives",
)
_MONTH_LABEL = _re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")


def _validate_schema(
    table_name: str,
    frame: pd.DataFrame,
    expected: tuple[str, ...],
) -> None:
    actual = tuple(frame.columns)
    if actual != expected:
        raise ValueError(
            f"tables[{table_name!r}] columns and order must be exactly "
            f"{expected}; got {actual}"
        )


def _validate_exact_values(
    values: pd.Series,
    expected: tuple[str, ...],
    label: str,
) -> None:
    actual = values.tolist()
    if len(actual) != len(expected) or any(
        not isinstance(value, str) or value != wanted
        for value, wanted in zip(actual, expected)
    ):
        raise ValueError(f"{label} order must be exactly {expected}; got {actual}")


def _validate_real(value: object, label: str, *, allow_nan: bool) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, _Real):
        raise TypeError(f"{label} must be a real number")
    if np.isnan(value):
        if allow_nan:
            return
        raise ValueError(f"{label} must be finite")
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _validate_count(value: object, label: str) -> None:
    _validate_real(value, label, allow_nan=False)
    if value < 0 or value != np.floor(value):
        raise ValueError(f"{label} must be a non-negative integral value")


def _validate_class_balance(frame: pd.DataFrame) -> None:
    if len(frame) != 1:
        raise ValueError("tables['cohort_target'] must contain exactly one row")
    row = frame.iloc[0]
    for column in ("rows", "positives", "negatives"):
        _validate_count(row[column], f"cohort_target.{column}")
    _validate_count(
        row["duplicate_appointment_ids"],
        "cohort_target.duplicate_appointment_ids",
    )
    if row["rows"] != row["positives"] + row["negatives"]:
        raise ValueError("cohort_target rows must equal positives plus negatives")
    if row["duplicate_appointment_ids"] != 0:
        raise ValueError("cohort_target duplicate_appointment_ids must equal zero")
    prevalence = row["prevalence"]
    _validate_real(prevalence, "cohort_target.prevalence", allow_nan=False)
    if prevalence < 0 or prevalence > 1:
        raise ValueError("cohort_target prevalence must be inside [0, 1]")


def _validate_prediction_months(months: pd.Series) -> None:
    if months.isna().any():
        raise ValueError("temporal_monthly prediction_month must not contain nulls")

    dtype = months.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError("temporal_monthly prediction_month must be timezone-naive")
    if isinstance(dtype, pd.PeriodDtype):
        frequency = dtype.freq
        valid_dtype = frequency.n == 1 and frequency.rule_code in {"M", "ME"}
        month_keys = pd.PeriodIndex(months.array, copy=False)
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        valid_dtype = True
        month_keys = pd.PeriodIndex(months.dt.to_period("M"), copy=False)
    elif isinstance(dtype, pd.StringDtype):
        string_values = months.tolist()
        valid_dtype = all(
            isinstance(value, str) and _MONTH_LABEL.fullmatch(value)
            for value in string_values
        )
        month_keys = (
            pd.PeriodIndex(string_values, freq="M")
            if valid_dtype
            else None
        )
    else:
        valid_dtype = False
        month_keys = None
    if not valid_dtype:
        raise TypeError(
            "temporal_monthly prediction_month must be timezone-naive "
            "datetime-like or monthly-period-like"
        )
    if not month_keys.is_unique:
        raise ValueError(
            "temporal_monthly prediction_month calendar months must be unique"
        )
    if not month_keys.is_monotonic_increasing:
        raise ValueError(
            "temporal_monthly prediction_month calendar months must be "
            "strictly increasing"
        )


def _validate_temporal(frame: pd.DataFrame) -> None:
    _validate_prediction_months(frame["prediction_month"])
    for column in _TEMPORAL_COUNT_COLUMNS:
        for position, value in enumerate(frame[column]):
            _validate_count(
                value,
                f"temporal_monthly.{column}[{position}]",
            )
    for position, value in enumerate(frame["no_show_rate"]):
        label = f"temporal_monthly.no_show_rate[{position}]"
        _validate_real(value, label, allow_nan=True)
        if not np.isnan(value) and (value < 0 or value > 1):
            raise ValueError(f"{label} must be inside [0, 1] or NaN")


def _validate_numeric_drift(frame: pd.DataFrame) -> None:
    _validate_exact_values(
        frame["feature"],
        _NUMERIC_FEATURE_COLUMNS,
        "numeric_drift feature",
    )
    for position, value in enumerate(frame["signed_smd"]):
        _validate_real(
            value,
            f"numeric_drift.signed_smd[{position}]",
            allow_nan=True,
        )


def _validate_categorical_drift(frame: pd.DataFrame) -> None:
    _validate_exact_values(
        frame["feature"],
        _CATEGORICAL_FEATURE_COLUMNS,
        "categorical_drift_features feature",
    )
    for position, value in enumerate(frame["total_variation_distance"]):
        label = f"categorical_drift_features.total_variation_distance[{position}]"
        _validate_real(value, label, allow_nan=False)
        if value < 0 or value > 1:
            raise ValueError(f"{label} must be inside [0, 1]")


def _validate_relationships(frame: pd.DataFrame) -> None:
    actual_pairs = list(zip(frame["feature_a"], frame["feature_b"]))
    if len(actual_pairs) != len(_EXPECTED_RELATIONSHIP_PAIRS) or any(
        not isinstance(feature_a, str)
        or not isinstance(feature_b, str)
        or (feature_a, feature_b) != expected
        for (feature_a, feature_b), expected in zip(
            actual_pairs,
            _EXPECTED_RELATIONSHIP_PAIRS,
        )
    ):
        raise ValueError(
            "numeric_relationships pairs and order must be exactly "
            f"{_EXPECTED_RELATIONSHIP_PAIRS}; got {actual_pairs}"
        )
    for position, value in enumerate(frame["spearman_correlation"]):
        label = f"numeric_relationships.spearman_correlation[{position}]"
        _validate_real(value, label, allow_nan=True)
        if not np.isnan(value) and (value < -1 or value > 1):
            raise ValueError(f"{label} must be inside [-1, 1] or NaN")


def _validate_consumed_tables(
    cohort_target: pd.DataFrame,
    temporal_monthly: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift_features: pd.DataFrame,
    numeric_relationships: pd.DataFrame,
) -> None:
    frames = (
        cohort_target,
        temporal_monthly,
        numeric_drift,
        categorical_drift_features,
        numeric_relationships,
    )
    for (table_name, expected), frame in zip(
        _SOURCE_SCHEMAS,
        frames,
        strict=True,
    ):
        _validate_schema(table_name, frame, expected)
    _validate_class_balance(cohort_target)
    _validate_temporal(temporal_monthly)
    _validate_numeric_drift(numeric_drift)
    _validate_categorical_drift(categorical_drift_features)
    _validate_relationships(numeric_relationships)


def _new_figure(figsize: tuple[float, float]) -> Figure:
    figure = Figure(figsize=figsize, dpi=120, layout="constrained")
    _FigureCanvasAgg(figure)
    return figure


def _build_class_balance(frame: pd.DataFrame) -> Figure:
    row = frame.iloc[0]
    values = (row["negatives"], row["positives"])
    figure = _new_figure((8.0, 4.5))
    axis = figure.subplots()
    bars = axis.bar(
        ("Attended", "No-show"),
        values,
        color=("#4C78A8", "#E45756"),
    )
    axis.set_xlabel("Outcome")
    axis.set_ylabel("Appointments")
    axis.set_title("Mature Train Class Balance")
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{int(value)}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    axis.text(
        0.98,
        0.95,
        f"No-show prevalence: {row['prevalence']:.2%}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    return figure


def _prediction_month_labels(months: pd.Series) -> tuple[str, ...]:
    if pd.api.types.is_datetime64_any_dtype(months.dtype):
        return tuple(months.dt.strftime("%Y-%m").tolist())
    return tuple(str(value) for value in months.tolist())


def _build_temporal(frame: pd.DataFrame) -> Figure:
    positions = np.arange(len(frame))
    counts = frame["mature_train_count"].to_numpy(copy=False)
    rates = frame["no_show_rate"].to_numpy(copy=False)
    labels = _prediction_month_labels(frame["prediction_month"])

    figure = _new_figure((10.0, 5.5))
    count_axis = figure.subplots()
    count_axis.bar(positions, counts, color="#4C78A8", alpha=0.8)
    count_axis.set_xlabel("Prediction Month")
    count_axis.set_ylabel("Mature Appointments")
    count_axis.set_title("Mature Train Volume and No-show Rate by Month")
    count_axis.set_xticks(positions, labels, rotation=45, ha="right")

    rate_axis = count_axis.twinx()
    rate_axis.plot(
        positions,
        rates,
        color="#E45756",
        marker="o",
        linewidth=2.0,
    )
    rate_axis.set_ylabel("No-show Rate")
    rate_axis.set_ylim(0.0, 1.0)
    rate_axis.yaxis.set_major_formatter(_PercentFormatter(xmax=1.0))
    return figure


def _build_numeric_drift(frame: pd.DataFrame) -> Figure:
    positions = np.arange(len(frame))
    values = frame["signed_smd"].to_numpy(copy=False)
    figure = _new_figure((9.0, 4.8))
    axis = figure.subplots()
    axis.barh(positions, values, color="#4C78A8")
    axis.axvline(0.0, color="#333333", linewidth=1.0)
    axis.set_xlabel("Signed Standardized Mean Difference")
    axis.set_ylabel("Feature")
    axis.set_title("Numerical Feature Drift: Train vs Validation")
    axis.set_yticks(positions, _NUMERIC_FEATURE_COLUMNS)
    axis.invert_yaxis()
    for position, value in zip(positions, values):
        if np.isnan(value):
            axis.plot(
                0.0,
                position,
                marker="x",
                markersize=8.0,
                markeredgewidth=2.0,
                color="#E45756",
                linestyle="None",
            )
            axis.annotate(
                "NA",
                xy=(0.0, position),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
            )
        else:
            offset = 5 if value >= 0 else -5
            alignment = "left" if value >= 0 else "right"
            axis.annotate(
                f"{value:.3f}",
                xy=(value, position),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=alignment,
                va="center",
            )
    return figure


def _build_categorical_drift(frame: pd.DataFrame) -> Figure:
    positions = np.arange(len(frame))
    values = frame["total_variation_distance"].to_numpy(copy=False)
    figure = _new_figure((9.0, 4.8))
    axis = figure.subplots()
    axis.barh(positions, values, color="#72B7B2")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Total Variation Distance")
    axis.set_ylabel("Feature")
    axis.set_title("Categorical Feature Drift: Train vs Validation")
    axis.set_yticks(positions, _CATEGORICAL_FEATURE_COLUMNS)
    axis.invert_yaxis()
    for position, value in zip(positions, values):
        offset = -5 if value > 0.9 else 5
        alignment = "right" if value > 0.9 else "left"
        axis.annotate(
            f"{value:.3f}",
            xy=(value, position),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=alignment,
            va="center",
        )
    return figure


def _relationship_matrix(frame: pd.DataFrame) -> np.ndarray:
    size = len(_NUMERIC_FEATURE_COLUMNS)
    matrix = np.full((size, size), np.nan, dtype="float64")
    np.fill_diagonal(matrix, 1.0)
    feature_positions = {
        feature: position
        for position, feature in enumerate(_NUMERIC_FEATURE_COLUMNS)
    }
    for row in frame.itertuples(index=False):
        first = feature_positions[row.feature_a]
        second = feature_positions[row.feature_b]
        matrix[first, second] = row.spearman_correlation
        matrix[second, first] = row.spearman_correlation
    return matrix


def _build_relationships(frame: pd.DataFrame) -> Figure:
    matrix = _relationship_matrix(frame)
    figure = _new_figure((8.0, 7.0))
    axis = figure.subplots()
    colormap = _colormaps["coolwarm"].with_extremes(bad="#D9D9D9")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    positions = np.arange(len(_NUMERIC_FEATURE_COLUMNS))
    axis.set_xticks(
        positions,
        _NUMERIC_FEATURE_COLUMNS,
        rotation=45,
        ha="right",
    )
    axis.set_yticks(positions, _NUMERIC_FEATURE_COLUMNS)
    axis.set_title("Mature Train Numerical Spearman Relationships")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Spearman Correlation")
    for row_position in positions:
        for column_position in positions:
            value = matrix[row_position, column_position]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            axis.text(
                column_position,
                row_position,
                label,
                ha="center",
                va="center",
                color="#111111",
            )
    return figure


def build_eda_figures(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Figure]:
    """Return five fresh figures built only from one approved table bundle."""

    _validate_tables(tables)
    cohort_target = tables["cohort_target"]
    temporal_monthly = tables["temporal_monthly"]
    numeric_drift = tables["numeric_drift"]
    categorical_drift_features = tables["categorical_drift_features"]
    numeric_relationships = tables["numeric_relationships"]
    _validate_consumed_tables(
        cohort_target,
        temporal_monthly,
        numeric_drift,
        categorical_drift_features,
        numeric_relationships,
    )
    figures = {
        "class_balance": _build_class_balance(cohort_target),
        "temporal_monthly": _build_temporal(temporal_monthly),
        "numeric_drift": _build_numeric_drift(numeric_drift),
        "categorical_drift": _build_categorical_drift(
            categorical_drift_features
        ),
        "numeric_relationships": _build_relationships(numeric_relationships),
    }
    if tuple(figures) != _FIGURE_KEYS:
        raise RuntimeError("internal EDA figure order invariant failed")
    return figures
