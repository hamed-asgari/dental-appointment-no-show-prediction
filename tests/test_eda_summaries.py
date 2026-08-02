"""Tests for deterministic mature-training exploratory summaries."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import summaries
from src.analysis.run_eda import select_eda_populations
from src.analysis.summaries import (
    summarize_cohort_target,
    summarize_missingness,
    summarize_numeric_by_target,
    summarize_numeric_features,
)
from src.data import build_dataset as bd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
EXPECTED_FEATURE_COLUMNS = (
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
EXPECTED_NUMERIC_FEATURES = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
EXPECTED_SUPERVISED_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "target",
    *EXPECTED_FEATURE_COLUMNS,
)
EXPECTED_COHORT_COLUMNS = (
    "rows",
    "positives",
    "negatives",
    "prevalence",
    "wilson_lower",
    "wilson_upper",
    "duplicate_appointment_ids",
)
EXPECTED_MISSINGNESS_COLUMNS = (
    "feature",
    "rows",
    "missing_count",
    "missing_rate",
    "non_missing_count",
    "unique_non_null",
    "is_constant",
)
EXPECTED_NUMERIC_COLUMNS = (
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
EXPECTED_NUMERIC_BY_TARGET_COLUMNS = (
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
SUMMARY_FUNCTIONS = (
    summarize_cohort_target,
    summarize_missingness,
    summarize_numeric_features,
    summarize_numeric_by_target,
)


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(tables)


@pytest.fixture(scope="session")
def populations(canonical_dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return select_eda_populations(canonical_dataset)


@pytest.fixture(scope="session")
def real_supervised(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return populations["supervised_train"]


def _supervised_fixture(targets: tuple[int, ...] = (0, 0, 1, 1, 1)) -> pd.DataFrame:
    rows = len(targets)
    values = np.arange(rows, dtype="float64")
    frame = pd.DataFrame(
        {
            "appointment_id": pd.Series(
                np.arange(1, rows + 1),
                dtype="int64",
            ),
            "prediction_time": pd.Series(
                pd.date_range("2024-01-01", periods=rows, freq="h"),
                dtype="datetime64[ns]",
            ),
            "target": pd.Series(targets, dtype="int8"),
            "planned_duration_min": pd.Series(values, dtype="float64"),
            "visit_type": pd.Series(
                [f"visit-{index % 2}" for index in range(rows)],
                dtype="string",
            ),
            "booking_channel": pd.Series(
                ["phone"] * rows,
                dtype="string",
            ),
            "booking_lead_time_hours": pd.Series(
                24.0 + values,
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(values % 7, dtype="float64"),
            "scheduled_hour": pd.Series(9.0 + values, dtype="float64"),
            "scheduled_month": pd.Series(1.0 + values, dtype="float64"),
            "approximate_age_at_prediction": pd.Series(
                20.0 + values,
                dtype="float64",
            ),
            "patient_registration_tenure_days": pd.Series(
                100.0 + values,
                dtype="float64",
            ),
            "dentist_tenure_days": pd.Series(
                200.0 + values,
                dtype="float64",
            ),
        }
    )
    return frame.loc[:, EXPECTED_SUPERVISED_COLUMNS]


def _set_float_values(
    frame: pd.DataFrame,
    column: str,
    values: list[float],
) -> None:
    frame[column] = pd.Series(values, index=frame.index, dtype="float64")


def test_real_data_integration_for_all_summaries(
    real_supervised: pd.DataFrame,
) -> None:
    assert tuple(real_supervised.columns) == EXPECTED_SUPERVISED_COLUMNS
    assert tuple(bd.FEATURE_COLUMNS) == EXPECTED_FEATURE_COLUMNS
    assert summaries.NUMERIC_FEATURE_COLUMNS == EXPECTED_NUMERIC_FEATURES

    cohort = summarize_cohort_target(real_supervised)
    assert tuple(cohort.columns) == EXPECTED_COHORT_COLUMNS
    assert cohort.loc[0, "rows"] == 3_670
    assert cohort.loc[0, "positives"] == 432
    assert cohort.loc[0, "negatives"] == 3_238
    assert cohort.loc[0, "prevalence"] == pytest.approx(432 / 3_670)
    assert cohort.loc[0, "duplicate_appointment_ids"] == 0

    missingness = summarize_missingness(real_supervised)
    assert tuple(missingness.columns) == EXPECTED_MISSINGNESS_COLUMNS
    assert tuple(missingness["feature"]) == EXPECTED_FEATURE_COLUMNS
    assert missingness["rows"].eq(3_670).all()
    assert missingness["missing_count"].eq(0).all()
    assert missingness["missing_rate"].eq(0.0).all()
    assert missingness["non_missing_count"].eq(3_670).all()

    numeric = summarize_numeric_features(real_supervised)
    assert tuple(numeric.columns) == EXPECTED_NUMERIC_COLUMNS
    assert tuple(numeric["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert numeric["n"].eq(3_670).all()
    assert numeric["missing_count"].eq(0).all()

    by_target = summarize_numeric_by_target(real_supervised)
    assert tuple(by_target.columns) == EXPECTED_NUMERIC_BY_TARGET_COLUMNS
    assert list(zip(by_target["feature"], by_target["target"], strict=True)) == [
        (feature, target)
        for feature in EXPECTED_NUMERIC_FEATURES
        for target in (0, 1)
    ]
    for feature in EXPECTED_NUMERIC_FEATURES:
        feature_rows = by_target.loc[by_target["feature"].eq(feature)]
        assert feature_rows.loc[feature_rows["target"].eq(0), "n"].item() == 3_238
        assert feature_rows.loc[feature_rows["target"].eq(1), "n"].item() == 432
        assert feature_rows["missing_count"].eq(0).all()


@pytest.mark.parametrize("summary_function", SUMMARY_FUNCTIONS)
def test_exact_input_contract_rejects_wrong_frames(
    summary_function: Callable[[pd.DataFrame], pd.DataFrame],
    canonical_dataset: pd.DataFrame,
    populations: dict[str, pd.DataFrame],
    real_supervised: pd.DataFrame,
) -> None:
    missing_column = real_supervised.drop(columns="visit_type")
    extra_column = real_supervised.assign(split="train")
    reordered_columns = list(EXPECTED_SUPERVISED_COLUMNS)
    reordered_columns[0], reordered_columns[1] = (
        reordered_columns[1],
        reordered_columns[0],
    )
    reordered = real_supervised.loc[:, reordered_columns]
    nonbinary_target = real_supervised.copy(deep=True)
    nonbinary_target.loc[nonbinary_target.index[0], "target"] = 2
    null_target = real_supervised.copy(deep=True)
    null_target["target"] = null_target["target"].astype("Int8")
    null_target.loc[null_target.index[0], "target"] = pd.NA
    null_appointment = real_supervised.copy(deep=True)
    null_appointment["appointment_id"] = null_appointment[
        "appointment_id"
    ].astype("Int64")
    null_appointment.loc[null_appointment.index[0], "appointment_id"] = pd.NA
    null_prediction_time = real_supervised.copy(deep=True)
    null_prediction_time.loc[
        null_prediction_time.index[0], "prediction_time"
    ] = pd.NaT
    nonnumeric_feature = real_supervised.copy(deep=True)
    nonnumeric_feature["planned_duration_min"] = nonnumeric_feature[
        "planned_duration_min"
    ].astype("string")

    invalid_frames = (
        canonical_dataset,
        populations["train_drift"],
        populations["validation_drift"],
        populations["maturity_audit"],
        missing_column,
        extra_column,
        reordered,
        nonbinary_target,
        null_target,
        null_appointment,
        null_prediction_time,
        nonnumeric_feature,
    )
    for invalid in invalid_frames:
        with pytest.raises(ValueError):
            summary_function(invalid)


@pytest.mark.parametrize("summary_function", SUMMARY_FUNCTIONS)
def test_exact_input_contract_rejects_non_dataframe(
    summary_function: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        summary_function([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize("column", EXPECTED_NUMERIC_FEATURES)
@pytest.mark.parametrize(
    "non_finite_value",
    (np.inf, -np.inf),
    ids=("positive-infinity", "negative-infinity"),
)
def test_numeric_contract_rejects_each_non_finite_feature_without_warning_or_mutation(
    column: str,
    non_finite_value: float,
) -> None:
    frame = _supervised_fixture()
    frame.loc[frame.index[0], column] = non_finite_value
    frame_before = frame.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=rf"{column}.*finite non-null"):
            summarize_numeric_features(frame)

    pd.testing.assert_frame_equal(frame, frame_before)


@pytest.mark.parametrize("summary_function", SUMMARY_FUNCTIONS)
def test_all_summary_entry_points_reject_non_finite_numeric_values_before_arithmetic(
    summary_function: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    frame = _supervised_fixture()
    frame.loc[frame.index[0], "booking_lead_time_hours"] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            ValueError,
            match=r"booking_lead_time_hours.*finite non-null",
        ):
            summary_function(frame)


@pytest.mark.parametrize("summary_function", SUMMARY_FUNCTIONS)
def test_all_summary_entry_points_reject_boolean_numeric_roles(
    summary_function: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    frame = _supervised_fixture()
    frame["planned_duration_min"] = pd.Series(
        [True, False, True, False, True],
        index=frame.index,
        dtype="bool",
    )

    with pytest.raises(
        ValueError,
        match=r"planned_duration_min.*numeric non-Boolean dtype",
    ):
        summary_function(frame)


def test_finite_negative_numeric_values_remain_accepted() -> None:
    frame = _supervised_fixture()
    for column in EXPECTED_NUMERIC_FEATURES:
        frame.loc[frame.index[0], column] = -1.0

    for summary_function in SUMMARY_FUNCTIONS:
        assert isinstance(summary_function(frame), pd.DataFrame)


def test_cohort_target_hand_calculation_wilson_and_duplicates() -> None:
    frame = _supervised_fixture((0, 1, 1, 0))
    frame["appointment_id"] = pd.Series([1, 2, 2, 4], dtype="int64")
    result = summarize_cohort_target(frame)

    z = 1.959963984540054
    rows = 4
    positives = 2
    prevalence = positives / rows
    denominator = 1.0 + z**2 / rows
    center = (prevalence + z**2 / (2.0 * rows)) / denominator
    radius = (
        z
        * math.sqrt(
            prevalence * (1.0 - prevalence) / rows
            + z**2 / (4.0 * rows**2)
        )
        / denominator
    )
    assert tuple(result.columns) == EXPECTED_COHORT_COLUMNS
    assert result.loc[0, "rows"] == 4
    assert result.loc[0, "positives"] == 2
    assert result.loc[0, "negatives"] == 2
    assert result.loc[0, "prevalence"] == 0.5
    assert result.loc[0, "wilson_lower"] == pytest.approx(center - radius, abs=1e-15)
    assert result.loc[0, "wilson_upper"] == pytest.approx(center + radius, abs=1e-15)
    assert result.loc[0, "duplicate_appointment_ids"] == 2


def test_cohort_target_empty_and_full_precision() -> None:
    empty = _supervised_fixture().iloc[:0].copy(deep=True)
    empty_result = summarize_cohort_target(empty)
    assert empty_result.loc[0, "rows"] == 0
    assert empty_result.loc[0, "positives"] == 0
    assert empty_result.loc[0, "negatives"] == 0
    assert empty_result.loc[0, "duplicate_appointment_ids"] == 0
    assert math.isnan(empty_result.loc[0, "prevalence"])
    assert math.isnan(empty_result.loc[0, "wilson_lower"])
    assert math.isnan(empty_result.loc[0, "wilson_upper"])

    thirds = _supervised_fixture((1, 0, 0))
    prevalence = summarize_cohort_target(thirds).loc[0, "prevalence"]
    assert prevalence == 1 / 3
    assert prevalence != round(prevalence, 6)


def test_missingness_hand_worked_fixture() -> None:
    frame = _supervised_fixture((0, 0, 1, 1))
    _set_float_values(frame, "planned_duration_min", [np.nan, 1.0, 1.0, 2.0])
    frame["visit_type"] = pd.Series(
        ["visit", pd.NA, "visit", "visit"],
        dtype="string",
    )
    frame["booking_channel"] = pd.Series(
        [pd.NA, pd.NA, pd.NA, pd.NA],
        dtype="string",
    )
    _set_float_values(frame, "booking_lead_time_hours", [24.0] * 4)
    _set_float_values(frame, "scheduled_weekday", [0.0, 1.0, 2.0, 3.0])
    _set_float_values(frame, "scheduled_hour", [9.0, 9.0, np.nan, 9.0])
    _set_float_values(frame, "scheduled_month", [1.0] * 4)
    _set_float_values(
        frame,
        "approximate_age_at_prediction",
        [np.nan] * 4,
    )
    _set_float_values(
        frame,
        "patient_registration_tenure_days",
        [0.0, 1.0, 2.0, 3.0],
    )
    _set_float_values(frame, "dentist_tenure_days", [5.0] * 4)

    result = summarize_missingness(frame)
    assert tuple(result.columns) == EXPECTED_MISSINGNESS_COLUMNS
    assert tuple(result["feature"]) == EXPECTED_FEATURE_COLUMNS
    expected = {
        "planned_duration_min": (1, 0.25, 3, 2, False),
        "visit_type": (1, 0.25, 3, 1, True),
        "booking_channel": (4, 1.0, 0, 0, False),
        "booking_lead_time_hours": (0, 0.0, 4, 1, True),
        "scheduled_weekday": (0, 0.0, 4, 4, False),
        "scheduled_hour": (1, 0.25, 3, 1, True),
        "scheduled_month": (0, 0.0, 4, 1, True),
        "approximate_age_at_prediction": (4, 1.0, 0, 0, False),
        "patient_registration_tenure_days": (0, 0.0, 4, 4, False),
        "dentist_tenure_days": (0, 0.0, 4, 1, True),
    }
    for feature, values in expected.items():
        row = result.loc[result["feature"].eq(feature)].iloc[0]
        assert tuple(
            row[
                [
                    "missing_count",
                    "missing_rate",
                    "non_missing_count",
                    "unique_non_null",
                    "is_constant",
                ]
            ]
        ) == values


def test_missingness_empty_frame() -> None:
    empty = _supervised_fixture().iloc[:0].copy(deep=True)
    result = summarize_missingness(empty)
    assert tuple(result["feature"]) == EXPECTED_FEATURE_COLUMNS
    assert result["rows"].eq(0).all()
    assert result["missing_count"].eq(0).all()
    assert result["missing_rate"].isna().all()
    assert result["non_missing_count"].eq(0).all()
    assert result["unique_non_null"].eq(0).all()
    assert ~result["is_constant"].any()


def test_numeric_summary_hand_calculation_and_strict_fences() -> None:
    frame = _supervised_fixture()
    _set_float_values(frame, "planned_duration_min", [0.0, 1.0, 2.0, 3.0, 100.0])
    result = summarize_numeric_features(frame)
    row = result.loc[result["feature"].eq("planned_duration_min")].iloc[0]
    mean = 21.2
    expected_std = math.sqrt(
        sum((value - mean) ** 2 for value in (0.0, 1.0, 2.0, 3.0, 100.0))
        / 4
    )
    assert tuple(result.columns) == EXPECTED_NUMERIC_COLUMNS
    assert tuple(result["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert row["n"] == 5
    assert row["missing_count"] == 0
    assert row["missing_rate"] == 0.0
    assert row["zero_count"] == 1
    assert row["mean"] == pytest.approx(mean)
    assert row["std"] == pytest.approx(expected_std)
    assert row["min"] == 0.0
    assert row["p01"] == pytest.approx(0.04)
    assert row["p05"] == pytest.approx(0.20)
    assert row["q1"] == 1.0
    assert row["median"] == 2.0
    assert row["q3"] == 3.0
    assert row["p95"] == pytest.approx(80.6)
    assert row["p99"] == pytest.approx(96.12)
    assert row["max"] == 100.0
    assert row["iqr"] == 2.0
    assert row["lower_fence"] == -2.0
    assert row["upper_fence"] == 6.0
    assert row["below_fence_count"] == 0
    assert row["above_fence_count"] == 1

    equality = _supervised_fixture()
    _set_float_values(equality, "planned_duration_min", [0.0, 1.0, 2.0, 3.0, 6.0])
    equality_row = summarize_numeric_features(equality).iloc[0]
    assert equality_row["upper_fence"] == 6.0
    assert equality_row["above_fence_count"] == 0

    outside = _supervised_fixture()
    _set_float_values(outside, "planned_duration_min", [0.0, 1.0, 2.0, 3.0, 7.0])
    outside_row = summarize_numeric_features(outside).iloc[0]
    assert outside_row["upper_fence"] == 6.0
    assert outside_row["above_fence_count"] == 1


def test_numeric_summary_preserves_unrounded_sample_standard_deviation() -> None:
    frame = _supervised_fixture((0, 0, 1, 1))
    _set_float_values(frame, "planned_duration_min", [0.0, 1.0, 2.0, 4.0])

    result = summarize_numeric_features(frame)
    actual_std = result.loc[
        result["feature"].eq("planned_duration_min"), "std"
    ].item()
    expected_std = math.sqrt(35.0 / 12.0)
    rounded_to_six_decimals = round(expected_std, 6)

    assert math.isclose(actual_std, expected_std, rel_tol=0.0, abs_tol=1e-15)
    assert not math.isclose(
        actual_std,
        rounded_to_six_decimals,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_numeric_summary_singleton_all_null_and_empty() -> None:
    singleton = _supervised_fixture((1,)).copy(deep=True)
    _set_float_values(singleton, "planned_duration_min", [5.0])
    singleton_row = summarize_numeric_features(singleton).iloc[0]
    assert singleton_row["n"] == 1
    assert math.isnan(singleton_row["std"])
    for column in ("min", "p01", "p05", "q1", "median", "q3", "p95", "p99", "max"):
        assert singleton_row[column] == 5.0
    assert singleton_row["iqr"] == 0.0
    assert singleton_row["lower_fence"] == 5.0
    assert singleton_row["upper_fence"] == 5.0
    assert singleton_row["below_fence_count"] == 0
    assert singleton_row["above_fence_count"] == 0

    all_null = _supervised_fixture((0, 1, 1))
    _set_float_values(all_null, "planned_duration_min", [np.nan] * 3)
    all_null_row = summarize_numeric_features(all_null).iloc[0]
    assert all_null_row["n"] == 0
    assert all_null_row["missing_count"] == 3
    assert all_null_row["missing_rate"] == 1.0
    assert all_null_row["zero_count"] == 0
    assert all_null_row[
        [
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
        ]
    ].isna().all()
    assert all_null_row["below_fence_count"] == 0
    assert all_null_row["above_fence_count"] == 0

    empty = _supervised_fixture().iloc[:0].copy(deep=True)
    empty_result = summarize_numeric_features(empty)
    assert tuple(empty_result["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert empty_result["n"].eq(0).all()
    assert empty_result["missing_count"].eq(0).all()
    assert empty_result["missing_rate"].isna().all()
    assert empty_result["zero_count"].eq(0).all()
    assert empty_result["mean"].isna().all()
    assert empty_result["below_fence_count"].eq(0).all()
    assert empty_result["above_fence_count"].eq(0).all()


def test_numeric_by_target_hand_worked_missing_and_singleton_groups() -> None:
    frame = _supervised_fixture((0, 0, 1, 1, 1))
    _set_float_values(
        frame,
        "planned_duration_min",
        [0.0, np.nan, 10.0, 14.0, np.nan],
    )
    _set_float_values(
        frame,
        "booking_lead_time_hours",
        [24.0, 26.0, 30.0, 34.0, np.nan],
    )
    result = summarize_numeric_by_target(frame)
    assert tuple(result.columns) == EXPECTED_NUMERIC_BY_TARGET_COLUMNS
    assert list(zip(result["feature"], result["target"], strict=True)) == [
        (feature, target)
        for feature in EXPECTED_NUMERIC_FEATURES
        for target in (0, 1)
    ]

    planned_zero = result.loc[
        result["feature"].eq("planned_duration_min") & result["target"].eq(0)
    ].iloc[0]
    assert planned_zero["n"] == 1
    assert planned_zero["missing_count"] == 1
    assert planned_zero["mean"] == 0.0
    assert math.isnan(planned_zero["std"])
    for column in ("min", "q1", "median", "q3", "max"):
        assert planned_zero[column] == 0.0

    planned_one = result.loc[
        result["feature"].eq("planned_duration_min") & result["target"].eq(1)
    ].iloc[0]
    assert planned_one["n"] == 2
    assert planned_one["missing_count"] == 1
    assert planned_one["mean"] == 12.0
    assert planned_one["std"] == pytest.approx(math.sqrt(8.0))
    assert planned_one["min"] == 10.0
    assert planned_one["q1"] == 11.0
    assert planned_one["median"] == 12.0
    assert planned_one["q3"] == 13.0
    assert planned_one["max"] == 14.0

    lead_zero = result.loc[
        result["feature"].eq("booking_lead_time_hours")
        & result["target"].eq(0)
    ].iloc[0]
    assert lead_zero["n"] == 2
    assert lead_zero["missing_count"] == 0
    assert lead_zero["mean"] == 25.0
    assert lead_zero["std"] == pytest.approx(math.sqrt(2.0))


def test_numeric_by_target_emits_absent_target_rows() -> None:
    frame = _supervised_fixture((0, 0))
    result = summarize_numeric_by_target(frame)
    target_one = result.loc[result["target"].eq(1)]
    assert tuple(target_one["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert target_one["n"].eq(0).all()
    assert target_one["missing_count"].eq(0).all()
    assert target_one[
        ["mean", "std", "min", "q1", "median", "q3", "max"]
    ].isna().all().all()


def test_summary_functions_do_not_call_dataset_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _supervised_fixture()

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("summary function attempted dataset selection")

    monkeypatch.setattr(bd, "select_development_rows", fail_if_called)
    monkeypatch.setattr(bd, "select_test_rows", fail_if_called)
    monkeypatch.setattr(bd, "select_model_features", fail_if_called)
    for summary_function in SUMMARY_FUNCTIONS:
        result = summary_function(frame)
        assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize("summary_function", SUMMARY_FUNCTIONS)
def test_summary_functions_are_non_mutating_and_deterministic(
    summary_function: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    frame = _supervised_fixture()
    _set_float_values(
        frame,
        "booking_lead_time_hours",
        [24.123456789, 40.0, np.nan, 35.0, 29.987654321],
    )
    frame_before = frame.copy(deep=True)

    first = summary_function(frame)
    second = summary_function(frame)
    reversed_result = summary_function(frame.iloc[::-1].copy(deep=True))

    assert first is not frame
    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, reversed_result)
