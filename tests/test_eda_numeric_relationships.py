"""Tests for deterministic feature-only numerical relationships."""

from __future__ import annotations

import inspect
import itertools
import math
import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import relationships
from src.analysis.relationships import summarize_numeric_relationships
from src.analysis.run_eda import select_eda_populations
from src.analysis.summaries import NUMERIC_FEATURE_COLUMNS
from src.data import build_dataset as bd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
EXPECTED_NUMERIC_FEATURES = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
EXPECTED_DRIFT_COLUMNS = (
    "appointment_id",
    "prediction_time",
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
EXPECTED_VALIDATION_START = pd.Timestamp("2025-03-01 00:00:00")
EXPECTED_OUTPUT_COLUMNS = (
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
EXPECTED_OUTPUT_DTYPES = (
    "str",
    "str",
    "int64",
    "int64",
    "int64",
    "float64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "float64",
)
EXPECTED_PAIRS = tuple(itertools.combinations(EXPECTED_NUMERIC_FEATURES, 2))
FLOAT_OUTPUT_COLUMNS = (
    "paired_rate",
    "pearson_correlation",
    "absolute_pearson_correlation",
    "spearman_correlation",
    "absolute_spearman_correlation",
)


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(tables)


@pytest.fixture(scope="session")
def real_train_drift(canonical_dataset: pd.DataFrame) -> pd.DataFrame:
    return select_eda_populations(canonical_dataset)["train_drift"]


@pytest.fixture(scope="session")
def eda_populations(
    canonical_dataset: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return select_eda_populations(canonical_dataset)


def _train_fixture(rows: int = 6) -> pd.DataFrame:
    values = np.arange(rows, dtype="float64")
    frame = pd.DataFrame(
        {
            "appointment_id": pd.Series(
                np.arange(1, rows + 1),
                dtype="int64",
            ),
            "prediction_time": pd.Series(
                pd.date_range("2025-02-01", periods=rows, freq="h"),
                dtype="datetime64[ns]",
            ),
            "planned_duration_min": pd.Series(
                1.0 + values,
                dtype="float64",
            ),
            "visit_type": pd.Series(["exam"] * rows, dtype="object"),
            "booking_channel": pd.Series(["phone"] * rows, dtype="object"),
            "booking_lead_time_hours": pd.Series(
                10.0 + 2.0 * values,
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(values % 7, dtype="float64"),
            "scheduled_hour": pd.Series(values % 24, dtype="float64"),
            "scheduled_month": pd.Series(1.0 + values % 12, dtype="float64"),
            "approximate_age_at_prediction": pd.Series(
                30.0 + values**2,
                dtype="float64",
            ),
            "patient_registration_tenure_days": pd.Series(
                100.0 + 3.0 * values,
                dtype="float64",
            ),
            "dentist_tenure_days": pd.Series(
                500.0 - 5.0 * values,
                dtype="float64",
            ),
        }
    )
    return frame.loc[:, EXPECTED_DRIFT_COLUMNS]


def _set_numeric(
    frame: pd.DataFrame,
    feature: str,
    values: list[float] | tuple[float, ...],
) -> None:
    frame[feature] = pd.Series(values, index=frame.index, dtype="float64")


def _result_row(
    result: pd.DataFrame,
    feature_a: str,
    feature_b: str,
) -> pd.Series:
    matches = result.loc[
        result["feature_a"].eq(feature_a)
        & result["feature_b"].eq(feature_b)
    ]
    assert len(matches) == 1
    return matches.iloc[0]


def _ordinary_pearson(
    x_values: list[float] | np.ndarray | pd.Series,
    y_values: list[float] | np.ndarray | pd.Series,
) -> float:
    x = np.asarray(x_values, dtype="float64")
    y = np.asarray(y_values, dtype="float64")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = float(np.sum(x_centered * y_centered))
    denominator = math.sqrt(float(np.sum(x_centered**2))) * math.sqrt(
        float(np.sum(y_centered**2))
    )
    return numerator / denominator


def _decimal_mean(values: tuple[float, ...]) -> float:
    with localcontext() as context:
        context.prec = 800
        total = sum(
            (Decimal.from_float(value) for value in values),
            start=Decimal(0),
        )
        return float(total / Decimal(len(values)))


def test_pair_order_schema_signature_and_dtypes() -> None:
    result = summarize_numeric_relationships(_train_fixture())
    signature = inspect.signature(summarize_numeric_relationships)

    assert tuple(signature.parameters) == ("train_drift",)
    assert signature.return_annotation == "pd.DataFrame"
    assert NUMERIC_FEATURE_COLUMNS == EXPECTED_NUMERIC_FEATURES
    assert tuple(result.columns) == EXPECTED_OUTPUT_COLUMNS
    assert tuple(map(str, result.dtypes)) == EXPECTED_OUTPUT_DTYPES
    assert len(result) == 10
    assert tuple(result[["feature_a", "feature_b"]].itertuples(
        index=False,
        name=None,
    )) == EXPECTED_PAIRS
    assert not result["feature_a"].eq(result["feature_b"]).any()
    reversed_pairs = {(feature_b, feature_a) for feature_a, feature_b in EXPECTED_PAIRS}
    assert not set(EXPECTED_PAIRS) & reversed_pairs


def test_package_internal_train_validator_is_reused_in_exact_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _train_fixture()
    calls: list[tuple[pd.DataFrame, str]] = []

    def record_validation(candidate: pd.DataFrame, *, frame_name: str) -> None:
        calls.append((candidate, frame_name))

    monkeypatch.setattr(
        relationships,
        "_validate_drift_frame",
        record_validation,
    )
    summarize_numeric_relationships(frame)
    assert calls == [(frame, "train_drift")]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.75, 0.75),
        (-0.75, -0.75),
        (0.0, 0.0),
        (-0.0, 0.0),
    ),
)
def test_correlation_boundary_preserves_valid_values(
    value: float,
    expected: float,
) -> None:
    result = relationships._finalize_correlation(value)
    assert result == expected
    if value == 0.0:
        assert math.copysign(1.0, result) == 1.0


def test_correlation_boundary_clamps_only_tolerated_overshoot() -> None:
    epsilon = np.finfo(np.float64).eps
    assert relationships._finalize_correlation(1.0 + 32.0 * epsilon) == 1.0
    assert relationships._finalize_correlation(-1.0 - 32.0 * epsilon) == -1.0


@pytest.mark.parametrize("value", (1.01, -1.01, math.inf, -math.inf))
def test_correlation_boundary_rejects_material_or_infinite_values(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        relationships._finalize_correlation(value)


@pytest.mark.parametrize(
    "values",
    (
        (1e308, 1.0, -1e308),
        (1e308, 1e-308, -1e308),
    ),
)
def test_relationship_stable_mean_preserves_cancellation_residuals(
    values: tuple[float, ...],
) -> None:
    expected = _decimal_mean(values)
    result = relationships._stable_centered_mean(values)
    assert result == expected
    assert result != 0.0


def test_ordinary_pearson_matches_independent_calculation() -> None:
    frame = _train_fixture(7)
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 17.0, 23.0]
    y = [3.0, 1.0, 7.0, 11.0, 19.0, 13.0, 29.0]
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", y)
    result = summarize_numeric_relationships(frame)
    row = _result_row(
        result,
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    expected = _ordinary_pearson(x, y)
    independently_sorted = _ordinary_pearson(sorted(x), sorted(y))
    assert row["pearson_correlation"] == pytest.approx(
        expected,
        rel=0.0,
        abs=2e-15,
    )
    assert not math.isclose(
        expected,
        independently_sorted,
        rel_tol=0.0,
        abs_tol=0.01,
    )
    assert row["absolute_pearson_correlation"] == abs(
        row["pearson_correlation"]
    )


@pytest.mark.parametrize(
    ("y", "expected"),
    (
        ([2.0, 4.0, 6.0, 8.0], 1.0),
        ([8.0, 6.0, 4.0, 2.0], -1.0),
    ),
)
def test_perfect_signed_relationships(y: list[float], expected: float) -> None:
    frame = _train_fixture(4)
    _set_numeric(frame, "planned_duration_min", [1.0, 2.0, 3.0, 4.0])
    _set_numeric(frame, "booking_lead_time_hours", y)
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert row["pearson_correlation"] == expected
    assert row["spearman_correlation"] == expected
    assert row["absolute_pearson_correlation"] == 1.0
    assert row["absolute_spearman_correlation"] == 1.0


def test_strictly_nonlinear_monotonic_relationship() -> None:
    frame = _train_fixture(5)
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 4.0, 9.0, 16.0, 25.0]
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", y)
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    expected_pearson = _ordinary_pearson(x, y)
    assert row["pearson_correlation"] == pytest.approx(
        expected_pearson,
        rel=0.0,
        abs=2e-15,
    )
    assert 0.9 < row["pearson_correlation"] < 0.99
    assert row["spearman_correlation"] == 1.0


def test_spearman_uses_average_ranks_for_ties() -> None:
    frame = _train_fixture(7)
    x = pd.Series([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0])
    y = pd.Series([4.0, 1.0, 2.0, 2.0, 5.0, 3.0, 3.0])
    _set_numeric(frame, "planned_duration_min", x.tolist())
    _set_numeric(frame, "booking_lead_time_hours", y.tolist())
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )

    average_expected = _ordinary_pearson(
        x.rank(method="average"),
        y.rank(method="average"),
    )
    alternative_results = {
        method: _ordinary_pearson(
            x.rank(method=method),
            y.rank(method=method),
        )
        for method in ("first", "dense", "min", "max")
    }
    assert row["spearman_correlation"] == pytest.approx(
        average_expected,
        rel=0.0,
        abs=2e-15,
    )
    assert all(
        not math.isclose(
            row["spearman_correlation"],
            alternative,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for alternative in alternative_results.values()
    )


def test_pairwise_missingness_counts_each_excluded_row_once() -> None:
    frame = _train_fixture(6)
    x = [1.0, np.nan, 3.0, np.nan, 5.0, np.nan]
    y = [2.0, 4.0, np.nan, np.nan, 10.0, 12.0]
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", y)
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert row["train_rows"] == 6
    assert row["paired_n"] == 2
    assert row["missing_pair_count"] == 4
    assert row["paired_rate"] == 2 / 6
    assert row["feature_a_unique_n"] == 2
    assert row["feature_b_unique_n"] == 2
    assert row["pearson_correlation"] == 1.0
    assert row["spearman_correlation"] == 1.0


def test_complete_rows_are_selected_separately_for_each_pair() -> None:
    frame = _train_fixture(5)
    _set_numeric(
        frame,
        "planned_duration_min",
        [1.0, 2.0, 3.0, 4.0, np.nan],
    )
    _set_numeric(
        frame,
        "booking_lead_time_hours",
        [1.0, 2.0, np.nan, np.nan, 5.0],
    )
    _set_numeric(
        frame,
        "approximate_age_at_prediction",
        [1.0, np.nan, 3.0, 4.0, 5.0],
    )
    result = summarize_numeric_relationships(frame)
    pair_counts = {
        (feature_a, feature_b): paired_n
        for feature_a, feature_b, paired_n in result[
            ["feature_a", "feature_b", "paired_n"]
        ].itertuples(index=False, name=None)
    }
    assert pair_counts[
        ("planned_duration_min", "booking_lead_time_hours")
    ] == 2
    assert pair_counts[
        ("planned_duration_min", "approximate_age_at_prediction")
    ] == 3
    assert pair_counts[
        ("booking_lead_time_hours", "approximate_age_at_prediction")
    ] == 2


def test_typed_empty_input_returns_ten_undefined_rows() -> None:
    result = summarize_numeric_relationships(_train_fixture(0))
    assert len(result) == 10
    assert result["train_rows"].eq(0).all()
    assert result["paired_n"].eq(0).all()
    assert result["missing_pair_count"].eq(0).all()
    assert result["paired_rate"].isna().all()
    assert result["feature_a_unique_n"].eq(0).all()
    assert result["feature_b_unique_n"].eq(0).all()
    assert result[
        [
            "pearson_correlation",
            "absolute_pearson_correlation",
            "spearman_correlation",
            "absolute_spearman_correlation",
        ]
    ].isna().all().all()


def test_all_null_pair_has_zero_pair_rate_and_undefined_correlations() -> None:
    frame = _train_fixture(4)
    _set_numeric(frame, "planned_duration_min", [np.nan] * 4)
    _set_numeric(frame, "booking_lead_time_hours", [np.nan] * 4)
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert row["paired_n"] == 0
    assert row["missing_pair_count"] == 4
    assert row["paired_rate"] == 0.0
    assert row["feature_a_unique_n"] == 0
    assert row["feature_b_unique_n"] == 0
    assert math.isnan(row["pearson_correlation"])
    assert math.isnan(row["spearman_correlation"])


def test_single_complete_pair_has_unique_counts_but_no_correlation() -> None:
    frame = _train_fixture(4)
    _set_numeric(frame, "planned_duration_min", [1.0, np.nan, np.nan, np.nan])
    _set_numeric(
        frame,
        "booking_lead_time_hours",
        [2.0, np.nan, np.nan, np.nan],
    )
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert row["paired_n"] == 1
    assert row["missing_pair_count"] == 3
    assert row["feature_a_unique_n"] == 1
    assert row["feature_b_unique_n"] == 1
    assert math.isnan(row["pearson_correlation"])
    assert math.isnan(row["spearman_correlation"])


@pytest.mark.parametrize(
    ("x", "y"),
    (
        ([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
        ([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]),
    ),
)
def test_constant_features_return_nan_not_zero(
    x: list[float],
    y: list[float],
) -> None:
    frame = _train_fixture(4)
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", y)
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert math.isnan(row["pearson_correlation"])
    assert math.isnan(row["absolute_pearson_correlation"])
    assert math.isnan(row["spearman_correlation"])
    assert math.isnan(row["absolute_spearman_correlation"])


def test_extreme_finite_positive_and_negative_relationships_are_robust() -> None:
    frame = _train_fixture(4)
    x = [-1e308, -5e307, 5e307, 1e308]
    positive = [-1e308, -5e307, 5e307, 1e308]
    negative = [1e308, 5e307, -5e307, -1e308]
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", positive)
    _set_numeric(frame, "approximate_age_at_prediction", negative)

    result = summarize_numeric_relationships(frame)
    positive_row = _result_row(
        result,
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    negative_row = _result_row(
        result,
        "planned_duration_min",
        "approximate_age_at_prediction",
    )
    assert positive_row["pearson_correlation"] == 1.0
    assert positive_row["spearman_correlation"] == 1.0
    assert negative_row["pearson_correlation"] == -1.0
    assert negative_row["spearman_correlation"] == -1.0
    assert not np.isinf(result.loc[:, FLOAT_OUTPUT_COLUMNS].to_numpy()).any()
    pd.testing.assert_frame_equal(
        result,
        summarize_numeric_relationships(frame.sample(frac=1.0, random_state=7)),
    )


@pytest.mark.parametrize("relationship_sign", (1.0, -1.0))
def test_skewed_extreme_relationships_normalize_before_centering(
    relationship_sign: float,
) -> None:
    frame = _train_fixture(10)
    x = [-1e308] + [1e308] * 9
    y = [relationship_sign * value for value in x]
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", y)
    before = frame.copy(deep=True)

    result = summarize_numeric_relationships(frame)
    row = _result_row(
        result,
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert row["pearson_correlation"] == relationship_sign
    assert row["spearman_correlation"] == relationship_sign
    assert row["absolute_pearson_correlation"] == 1.0
    assert row["absolute_spearman_correlation"] == 1.0
    assert not np.isinf(result.loc[:, FLOAT_OUTPUT_COLUMNS].to_numpy()).any()
    pd.testing.assert_frame_equal(frame, before)

    reversed_result = summarize_numeric_relationships(frame.iloc[::-1])
    shuffled_result = summarize_numeric_relationships(
        frame.sample(frac=1.0, random_state=31)
    )
    changed_index = frame.copy(deep=True)
    changed_index.index = np.arange(2_000, 2_000 + len(changed_index))
    reindexed_result = summarize_numeric_relationships(changed_index)
    pd.testing.assert_frame_equal(result, reversed_result)
    pd.testing.assert_frame_equal(result, shuffled_result)
    pd.testing.assert_frame_equal(result, reindexed_result)


def test_extreme_unequal_vector_scales_preserve_perfect_relationship() -> None:
    frame = _train_fixture(10)
    large = [-1e308] + [1e308] * 9
    small = [-1.0] + [1.0] * 9
    _set_numeric(frame, "planned_duration_min", large)
    _set_numeric(frame, "booking_lead_time_hours", small)
    result = summarize_numeric_relationships(frame)
    row = _result_row(
        result,
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    assert row["pearson_correlation"] == 1.0
    assert row["spearman_correlation"] == 1.0
    assert not np.isinf(result.loc[:, FLOAT_OUTPUT_COLUMNS].to_numpy()).any()


def test_correlations_are_not_rounded_to_six_decimals() -> None:
    frame = _train_fixture(8)
    x = [1.0, 2.0, 4.0, 8.0, 16.0, 17.0, 23.0, 31.0]
    y = [3.0, 1.0, 7.0, 11.0, 19.0, 13.0, 29.0, 5.0]
    _set_numeric(frame, "planned_duration_min", x)
    _set_numeric(frame, "booking_lead_time_hours", y)
    row = _result_row(
        summarize_numeric_relationships(frame),
        "planned_duration_min",
        "booking_lead_time_hours",
    )
    expected_pearson = _ordinary_pearson(x, y)
    expected_spearman = _ordinary_pearson(
        pd.Series(x).rank(method="average"),
        pd.Series(y).rank(method="average"),
    )
    assert row["pearson_correlation"] == pytest.approx(
        expected_pearson,
        rel=0.0,
        abs=2e-15,
    )
    assert row["spearman_correlation"] == pytest.approx(
        expected_spearman,
        rel=0.0,
        abs=2e-15,
    )
    assert row["pearson_correlation"] != round(expected_pearson, 6)
    assert row["spearman_correlation"] != round(expected_spearman, 6)


@pytest.mark.parametrize("unexpected", (math.inf, -math.inf))
def test_output_infinity_from_private_helper_raises_without_mutation(
    unexpected: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _train_fixture()
    before = frame.copy(deep=True)
    monkeypatch.setattr(
        relationships,
        "_robust_pearson",
        lambda pairs: unexpected,
    )
    with pytest.raises(ValueError, match="must not contain infinity"):
        summarize_numeric_relationships(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_non_dataframe_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="train_drift"):
        summarize_numeric_relationships([1, 2, 3])  # type: ignore[arg-type]


def test_nonapproved_population_projections_are_rejected(
    canonical_dataset: pd.DataFrame,
    eda_populations: dict[str, pd.DataFrame],
) -> None:
    train = eda_populations["train_drift"]
    invalid_frames = (
        canonical_dataset,
        eda_populations["supervised_train"],
        eda_populations["validation_drift"],
        eda_populations["maturity_audit"],
        train.assign(target=0),
    )
    for invalid in invalid_frames:
        with pytest.raises(ValueError):
            summarize_numeric_relationships(invalid)


def _invalid_train_frame(case: str) -> pd.DataFrame:
    frame = _train_fixture()
    if case == "missing_column":
        frame = frame.drop(columns=["visit_type"])
    elif case == "extra_column":
        frame = frame.assign(extra=0)
    elif case == "reordered_columns":
        columns = list(frame.columns)
        columns[2], columns[3] = columns[3], columns[2]
        frame = frame.loc[:, columns]
    elif case == "null_id":
        frame.loc[frame.index[0], "appointment_id"] = np.nan
    elif case == "duplicate_id":
        frame.loc[frame.index[1], "appointment_id"] = frame.iloc[0][
            "appointment_id"
        ]
    elif case == "null_timestamp":
        frame.loc[frame.index[0], "prediction_time"] = pd.NaT
    elif case == "timestamp_object":
        frame["prediction_time"] = frame["prediction_time"].astype("str")
    elif case == "timezone_timestamp":
        frame["prediction_time"] = frame["prediction_time"].dt.tz_localize(
            "UTC"
        )
    elif case == "boundary_timestamp":
        frame.loc[frame.index[0], "prediction_time"] = EXPECTED_VALIDATION_START
    elif case == "late_timestamp":
        frame.loc[frame.index[0], "prediction_time"] = (
            EXPECTED_VALIDATION_START + pd.Timedelta(seconds=1)
        )
    elif case == "boolean_numeric":
        frame["planned_duration_min"] = True
    elif case == "nonnumeric":
        frame["planned_duration_min"] = "invalid"
    elif case == "complex_numeric":
        frame["planned_duration_min"] = 1.0 + 2.0j
    elif case == "positive_infinity":
        frame.loc[frame.index[0], "planned_duration_min"] = np.inf
    elif case == "negative_infinity":
        frame.loc[frame.index[0], "planned_duration_min"] = -np.inf
    else:
        raise AssertionError(f"unknown invalid case: {case}")
    return frame


@pytest.mark.parametrize(
    "case",
    (
        "missing_column",
        "extra_column",
        "reordered_columns",
        "null_id",
        "duplicate_id",
        "null_timestamp",
        "timestamp_object",
        "timezone_timestamp",
        "boundary_timestamp",
        "late_timestamp",
        "boolean_numeric",
        "nonnumeric",
        "complex_numeric",
        "positive_infinity",
        "negative_infinity",
    ),
)
def test_shared_train_validation_regression(case: str) -> None:
    with pytest.raises(ValueError):
        summarize_numeric_relationships(_invalid_train_frame(case))


def test_literal_validation_boundary_accepts_immediately_before() -> None:
    frame = _train_fixture()
    frame.loc[frame.index[0], "prediction_time"] = (
        EXPECTED_VALIDATION_START - pd.Timedelta(nanoseconds=1)
    )
    result = summarize_numeric_relationships(frame)
    assert len(result) == 10


def test_validation_failure_does_not_mutate_input() -> None:
    frame = _invalid_train_frame("boundary_timestamp")
    before = frame.copy(deep=True)
    with pytest.raises(ValueError):
        summarize_numeric_relationships(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_input_nonmutation_and_output_independence() -> None:
    frame = _train_fixture(7)
    frame.index = [11, 3, 17, 5, 13, 7, 19]
    before = frame.copy(deep=True)
    expected = summarize_numeric_relationships(frame)
    repeated = summarize_numeric_relationships(frame)

    assert expected is not repeated
    pd.testing.assert_frame_equal(frame, before)
    repeated.iloc[0, 0] = "changed"
    repeated["pearson_correlation"] = 0.0
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(summarize_numeric_relationships(frame), expected)


def test_output_is_exactly_row_order_and_index_independent() -> None:
    frame = _train_fixture(9)
    _set_numeric(
        frame,
        "planned_duration_min",
        [9.0, 1.0, np.nan, 7.0, 2.0, 8.0, 3.0, 6.0, 4.0],
    )
    _set_numeric(
        frame,
        "booking_lead_time_hours",
        [2.0, 8.0, 3.0, np.nan, 7.0, 1.0, 5.0, 9.0, 4.0],
    )
    expected = summarize_numeric_relationships(frame)
    repeated = summarize_numeric_relationships(frame)
    reversed_result = summarize_numeric_relationships(frame.iloc[::-1])
    shuffled = summarize_numeric_relationships(
        frame.sample(frac=1.0, random_state=29)
    )
    changed_index = frame.copy(deep=True)
    changed_index.index = np.arange(1_000, 1_000 + len(frame))
    reindexed = summarize_numeric_relationships(changed_index)

    pd.testing.assert_frame_equal(expected, repeated)
    pd.testing.assert_frame_equal(expected, reversed_result)
    pd.testing.assert_frame_equal(expected, shuffled)
    pd.testing.assert_frame_equal(expected, reindexed)


def test_authentic_train_only_reconciliation(
    real_train_drift: pd.DataFrame,
) -> None:
    train = real_train_drift
    result = summarize_numeric_relationships(train)
    assert tuple(train.columns) == EXPECTED_DRIFT_COLUMNS
    assert "target" not in train.columns
    assert len(train) == 3_670
    assert len(result) == 10
    assert result["train_rows"].eq(3_670).all()
    assert result["paired_n"].eq(3_670).all()
    assert result["missing_pair_count"].eq(0).all()
    assert result["paired_rate"].eq(1.0).all()

    for feature_a, feature_b in EXPECTED_PAIRS:
        row = _result_row(result, feature_a, feature_b)
        complete = train.loc[:, [feature_a, feature_b]].dropna()
        x = complete[feature_a]
        y = complete[feature_b]
        expected_pearson = _ordinary_pearson(x, y)
        expected_spearman = _ordinary_pearson(
            x.rank(method="average"),
            y.rank(method="average"),
        )
        assert row["feature_a_unique_n"] == x.nunique()
        assert row["feature_b_unique_n"] == y.nunique()
        assert row["pearson_correlation"] == pytest.approx(
            expected_pearson,
            rel=0.0,
            abs=2e-14,
        )
        assert row["spearman_correlation"] == pytest.approx(
            expected_spearman,
            rel=0.0,
            abs=2e-14,
        )
        assert row["absolute_pearson_correlation"] == abs(
            row["pearson_correlation"]
        )
        assert row["absolute_spearman_correlation"] == abs(
            row["spearman_correlation"]
        )
    correlations = result.loc[
        :,
        ["pearson_correlation", "spearman_correlation"],
    ].to_numpy()
    assert np.isfinite(correlations).all()
    assert (np.abs(correlations) <= 1.0).all()


def test_production_has_no_leakage_selectors_or_forbidden_correlation_calls() -> None:
    source = inspect.getsource(relationships)
    forbidden_patterns = (
        r"\btarget\b",
        r"\bvalidation_drift\b",
        r"\bselect_eda_populations\b",
        r"\bselect_development_rows\b",
        r"\bselect_test_rows\b",
        r"\bselect_model_features\b",
        r"\bbuild_canonical_dataset\b",
        r"validation[_ -]?label",
        r"test[_ -]?population",
        r"\.corr\s*\(",
        r"corrcoef\s*\(",
        r"scipy",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, source, flags=re.IGNORECASE) is None
