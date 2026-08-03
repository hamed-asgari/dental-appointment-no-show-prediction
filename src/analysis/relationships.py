"""Deterministic feature-only numerical predictor relationships."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from src.analysis.drift import _validate_drift_frame
from src.analysis.summaries import NUMERIC_FEATURE_COLUMNS


_NUMERIC_RELATIONSHIP_COLUMNS = (
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
)
_FLOAT_OUTPUT_COLUMNS = (
    "paired_rate",
    "pearson_correlation",
    "absolute_pearson_correlation",
    "spearman_correlation",
    "absolute_spearman_correlation",
)
_CORRELATION_BOUND_TOLERANCE = 64.0 * float(np.finfo("float64").eps)


def _stable_centered_mean(values: tuple[float, ...]) -> float:
    minimum = min(values)
    maximum = max(values)
    anchor = math.fsum((minimum / 2.0, maximum / 2.0))
    count = len(values)
    try:
        residual_mean = math.fsum(
            (value - anchor) / count for value in values
        )
        mean = math.fsum((anchor, residual_mean))
    except (OverflowError, ValueError):
        return np.nan
    return float(mean) if math.isfinite(mean) else np.nan


def _scaled_centered_deviations(
    values: tuple[float, ...],
) -> tuple[float, ...] | None:
    raw_scale = max((abs(value) for value in values), default=0.0)
    if raw_scale == 0.0:
        return None
    scaled = tuple(value / raw_scale for value in values)
    if not all(math.isfinite(value) for value in scaled):
        return None
    scaled_mean = _stable_centered_mean(scaled)
    if not math.isfinite(scaled_mean):
        return None
    centered = tuple(value - scaled_mean for value in scaled)
    if not all(math.isfinite(value) for value in centered):
        return None
    deviation_scale = max((abs(value) for value in centered), default=0.0)
    if deviation_scale == 0.0:
        return None
    return tuple(value / deviation_scale for value in centered)


def _finalize_correlation(value: float) -> float:
    if math.isnan(value):
        return np.nan
    if not math.isfinite(value):
        raise ValueError("correlation must be finite or undefined")
    if value == 0.0:
        return 0.0
    if -1.0 <= value <= 1.0:
        return float(value)
    if value <= 1.0 + _CORRELATION_BOUND_TOLERANCE and value > 1.0:
        return 1.0
    if value >= -1.0 - _CORRELATION_BOUND_TOLERANCE and value < -1.0:
        return -1.0
    raise ValueError("correlation materially exceeds the interval [-1, 1]")


def _robust_pearson(pairs: tuple[tuple[float, float], ...]) -> float:
    if len(pairs) < 2:
        return np.nan
    ordered_pairs = tuple(sorted(pairs, key=lambda pair: (pair[0], pair[1])))
    x_values = tuple(pair[0] for pair in ordered_pairs)
    y_values = tuple(pair[1] for pair in ordered_pairs)
    x_deviations = _scaled_centered_deviations(x_values)
    y_deviations = _scaled_centered_deviations(y_values)
    if x_deviations is None or y_deviations is None:
        return np.nan
    if x_deviations == y_deviations:
        return 1.0
    if all(
        x_value == -y_value
        for x_value, y_value in zip(
            x_deviations,
            y_deviations,
            strict=True,
        )
    ):
        return -1.0
    cross_sum = math.fsum(
        x_value * y_value
        for x_value, y_value in zip(
            x_deviations,
            y_deviations,
            strict=True,
        )
    )
    x_squared_sum = math.fsum(value * value for value in x_deviations)
    y_squared_sum = math.fsum(value * value for value in y_deviations)
    denominator = math.sqrt(x_squared_sum) * math.sqrt(y_squared_sum)
    if denominator == 0.0 or not math.isfinite(denominator):
        return np.nan
    correlation = cross_sum / denominator
    return _finalize_correlation(correlation)


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return tuple(ranks)


def _spearman_correlation(
    pairs: tuple[tuple[float, float], ...],
) -> float:
    if len(pairs) < 2:
        return np.nan
    x_values = tuple(pair[0] for pair in pairs)
    y_values = tuple(pair[1] for pair in pairs)
    x_ranks = _average_ranks(x_values)
    y_ranks = _average_ranks(y_values)
    rank_pairs = tuple(zip(x_ranks, y_ranks, strict=True))
    return _robust_pearson(rank_pairs)


def _require_no_infinite_output(result: pd.DataFrame) -> None:
    floating_values = result.loc[:, _FLOAT_OUTPUT_COLUMNS].to_numpy(
        dtype="float64",
        copy=True,
    )
    if np.isinf(floating_values).any():
        raise ValueError("numeric relationship output must not contain infinity")


def summarize_numeric_relationships(
    train_drift: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize pairwise-complete mature-training numerical relationships."""

    _validate_drift_frame(train_drift, frame_name="train_drift")
    train_rows = len(train_drift)
    summaries: list[dict[str, object]] = []
    for feature_a, feature_b in combinations(NUMERIC_FEATURE_COLUMNS, 2):
        pair_mask = train_drift[feature_a].notna() & train_drift[feature_b].notna()
        complete = train_drift.loc[pair_mask, [feature_a, feature_b]]
        paired_n = len(complete)
        pairs = tuple(
            (float(feature_a_value), float(feature_b_value))
            for feature_a_value, feature_b_value in complete.itertuples(
                index=False,
                name=None,
            )
        )
        pearson = _robust_pearson(pairs)
        spearman = _spearman_correlation(pairs)
        summaries.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "train_rows": train_rows,
                "paired_n": paired_n,
                "missing_pair_count": train_rows - paired_n,
                "paired_rate": paired_n / train_rows if train_rows else np.nan,
                "feature_a_unique_n": int(complete[feature_a].nunique()),
                "feature_b_unique_n": int(complete[feature_b].nunique()),
                "pearson_correlation": pearson,
                "absolute_pearson_correlation": (
                    abs(pearson) if math.isfinite(pearson) else np.nan
                ),
                "spearman_correlation": spearman,
                "absolute_spearman_correlation": (
                    abs(spearman) if math.isfinite(spearman) else np.nan
                ),
            }
        )
    result = pd.DataFrame(
        summaries,
        columns=_NUMERIC_RELATIONSHIP_COLUMNS,
    )
    _require_no_infinite_output(result)
    return result
