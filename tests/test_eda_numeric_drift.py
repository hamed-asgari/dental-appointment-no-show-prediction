"""Tests for deterministic feature-only numerical drift summaries."""

from __future__ import annotations

import inspect
import math
import re
import warnings
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import drift, run_eda
from src.analysis.drift import summarize_numeric_drift
from src.analysis.run_eda import select_eda_populations
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
EXPECTED_DRIFT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    *EXPECTED_FEATURE_COLUMNS,
)
EXPECTED_NUMERIC_FEATURES = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
EXPECTED_OUTPUT_COLUMNS = (
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
EXPECTED_OUTPUT_DTYPES = (
    "str",
    "int64",
    "int64",
    "int64",
    "int64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
)
EXPECTED_VALIDATION_START = pd.Timestamp("2025-03-01 00:00:00")
EXPECTED_TEST_START = pd.Timestamp("2025-08-01 00:00:00")
EXPECTED_TRAIN_ROWS = 3_670
EXPECTED_VALIDATION_ROWS = 1_541
EXPECTED_QUANTILES = (0.10, 0.50, 0.90)
EXPECTED_APPROXIMATE_SMDS = {
    "planned_duration_min": -0.039,
    "booking_lead_time_hours": 0.181,
    "approximate_age_at_prediction": 0.114,
    "patient_registration_tenure_days": 1.945,
    "dentist_tenure_days": 2.350,
}
EXPECTED_APPROXIMATE_MEDIAN_SHIFTS = {
    "booking_lead_time_hours": 9.24,
    "approximate_age_at_prediction": 2.0,
    "patient_registration_tenure_days": 260.0,
    "dentist_tenure_days": 270.0,
}


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(tables)


@pytest.fixture(scope="session")
def eda_populations(
    canonical_dataset: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return select_eda_populations(canonical_dataset)


@pytest.fixture(scope="session")
def real_drift_frames(
    eda_populations: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return eda_populations["train_drift"], eda_populations["validation_drift"]


def _drift_fixture(
    appointment_ids: tuple[int, ...],
    prediction_times: tuple[pd.Timestamp | str, ...],
) -> pd.DataFrame:
    rows = len(appointment_ids)
    values = np.arange(rows, dtype="float64")
    frame = pd.DataFrame(
        {
            "appointment_id": pd.Series(appointment_ids, dtype="int64"),
            "prediction_time": pd.Series(
                pd.to_datetime(list(prediction_times)),
                dtype="datetime64[ns]",
            ),
            "planned_duration_min": pd.Series(values, dtype="float64"),
            "visit_type": pd.Series(["exam"] * rows, dtype="object"),
            "booking_channel": pd.Series(["phone"] * rows, dtype="object"),
            "booking_lead_time_hours": pd.Series(
                24.0 + values,
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(values % 7, dtype="float64"),
            "scheduled_hour": pd.Series(9.0 + values % 8, dtype="float64"),
            "scheduled_month": pd.Series(1.0 + values % 12, dtype="float64"),
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
    return frame.loc[:, EXPECTED_DRIFT_COLUMNS]


def _train_fixture(rows: int = 4) -> pd.DataFrame:
    return _drift_fixture(
        tuple(range(1, rows + 1)),
        tuple(
            pd.Timestamp("2025-02-01") + pd.Timedelta(days=index)
            for index in range(rows)
        ),
    )


def _validation_fixture(rows: int = 4) -> pd.DataFrame:
    return _drift_fixture(
        tuple(range(101, 101 + rows)),
        tuple(
            pd.Timestamp("2025-03-01") + pd.Timedelta(days=index)
            for index in range(rows)
        ),
    )


def _set_numeric_values(
    frame: pd.DataFrame,
    feature: str,
    values: Sequence[float],
) -> None:
    frame[feature] = pd.Series(values, index=frame.index, dtype="float64")


def _independent_description(series: pd.Series) -> dict[str, float | int]:
    values = np.asarray(sorted(series.dropna().tolist()), dtype="float64")
    rows = len(series)
    n = len(values)
    missing_count = rows - n
    if n:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else math.nan
        q10, median, q90 = (
            float(value)
            for value in np.quantile(
                values,
                EXPECTED_QUANTILES,
                method="linear",
            )
        )
    else:
        mean = std = q10 = median = q90 = math.nan
    return {
        "n": n,
        "missing_count": missing_count,
        "missing_rate": missing_count / rows if rows else math.nan,
        "mean": mean,
        "std": std,
        "q10": q10,
        "median": median,
        "q90": q90,
    }


def _independent_smd(
    train_mean: float,
    validation_mean: float,
    train_std: float,
    validation_std: float,
) -> float:
    if any(
        math.isnan(value)
        for value in (train_mean, validation_mean, train_std, validation_std)
    ):
        return math.nan
    pooled_scale = math.sqrt((train_std**2 + validation_std**2) / 2.0)
    difference = validation_mean - train_mean
    if pooled_scale == 0.0:
        return 0.0 if difference == 0.0 else math.nan
    return difference / pooled_scale


def _result_row(result: pd.DataFrame, feature: str) -> pd.Series:
    matching = result.loc[result["feature"].eq(feature)]
    assert len(matching) == 1
    return matching.iloc[0]


def _assert_no_infinite_float_outputs(result: pd.DataFrame) -> None:
    floating = result.select_dtypes(include="floating").to_numpy(copy=True)
    assert not np.isinf(floating).any()


def _decimal_mean_of_exact_floats(values: Sequence[float]) -> float:
    with localcontext() as context:
        context.prec = 2_000
        total = sum(
            (Decimal.from_float(value) for value in values),
            start=Decimal(0),
        )
        return float(total / Decimal(len(values)))


def test_real_data_integration_recomputes_every_formula(
    real_drift_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, validation = real_drift_frames
    result = summarize_numeric_drift(train, validation)

    assert tuple(train.columns) == EXPECTED_DRIFT_COLUMNS
    assert tuple(validation.columns) == EXPECTED_DRIFT_COLUMNS
    assert tuple(result.columns) == EXPECTED_OUTPUT_COLUMNS
    assert tuple(result["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert len(train) == EXPECTED_TRAIN_ROWS
    assert len(validation) == EXPECTED_VALIDATION_ROWS
    assert result["train_rows"].eq(EXPECTED_TRAIN_ROWS).all()
    assert result["validation_rows"].eq(EXPECTED_VALIDATION_ROWS).all()
    assert result["train_missing_count"].eq(0).all()
    assert result["validation_missing_count"].eq(0).all()
    assert result["train_missing_rate"].eq(0.0).all()
    assert result["validation_missing_rate"].eq(0.0).all()

    for feature in EXPECTED_NUMERIC_FEATURES:
        row = _result_row(result, feature)
        train_expected = _independent_description(train[feature])
        validation_expected = _independent_description(validation[feature])
        expected_smd = _independent_smd(
            train_expected["mean"],
            validation_expected["mean"],
            train_expected["std"],
            validation_expected["std"],
        )
        for prefix, expected in (
            ("train", train_expected),
            ("validation", validation_expected),
        ):
            assert row[f"{prefix}_n"] == expected["n"]
            assert row[f"{prefix}_missing_count"] == expected["missing_count"]
            assert row[f"{prefix}_missing_rate"] == expected["missing_rate"]
            for statistic in ("mean", "std", "q10", "median", "q90"):
                assert row[f"{prefix}_{statistic}"] == pytest.approx(
                    expected[statistic],
                    rel=1e-14,
                    abs=1e-12,
                )
        assert row["missing_rate_difference"] == (
            validation_expected["missing_rate"] - train_expected["missing_rate"]
        )
        assert row["q10_shift"] == pytest.approx(
            validation_expected["q10"] - train_expected["q10"],
            rel=1e-14,
            abs=1e-12,
        )
        assert row["median_shift"] == pytest.approx(
            validation_expected["median"] - train_expected["median"],
            rel=1e-14,
            abs=1e-12,
        )
        assert row["q90_shift"] == pytest.approx(
            validation_expected["q90"] - train_expected["q90"],
            rel=1e-14,
            abs=1e-12,
        )
        assert row["signed_smd"] == pytest.approx(
            expected_smd,
            rel=1e-14,
            abs=1e-12,
        )

    for feature, approximate_smd in EXPECTED_APPROXIMATE_SMDS.items():
        assert _result_row(result, feature)["signed_smd"] == pytest.approx(
            approximate_smd,
            abs=0.002,
        )
    for feature, approximate_shift in EXPECTED_APPROXIMATE_MEDIAN_SHIFTS.items():
        assert _result_row(result, feature)["median_shift"] == pytest.approx(
            approximate_shift,
            abs=0.6,
        )
    assert set(result.columns).isdisjoint(
        {
            "target",
            "split",
            "patient_id",
            "dentist_id",
            "development_fit_eligible",
            "pretest_fit_eligible",
        }
    )


def test_public_signature_remains_exactly_two_ordered_parameters() -> None:
    signature = inspect.signature(summarize_numeric_drift)
    assert tuple(signature.parameters) == ("train_drift", "validation_drift")


@pytest.mark.parametrize("argument", ("train", "validation"))
def test_non_dataframe_inputs_are_rejected(argument: str) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        if argument == "train":
            summarize_numeric_drift([1, 2], validation)  # type: ignore[arg-type]
        else:
            summarize_numeric_drift(train, [1, 2])  # type: ignore[arg-type]


def test_wrong_projection_shapes_are_rejected(
    canonical_dataset: pd.DataFrame,
    eda_populations: dict[str, pd.DataFrame],
) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    reordered_columns = list(EXPECTED_DRIFT_COLUMNS)
    reordered_columns[0], reordered_columns[1] = (
        reordered_columns[1],
        reordered_columns[0],
    )
    invalid_frames = (
        canonical_dataset,
        eda_populations["supervised_train"],
        eda_populations["maturity_audit"],
        train.drop(columns="visit_type"),
        train.assign(extra="forbidden"),
        train.loc[:, reordered_columns],
        train.assign(target=0),
    )
    for invalid in invalid_frames:
        with pytest.raises(ValueError, match="approved projection"):
            summarize_numeric_drift(invalid, validation)


@pytest.mark.parametrize("argument", ("train", "validation"))
@pytest.mark.parametrize(
    "case",
    (
        "duplicate-appointment",
        "null-appointment",
        "null-prediction-time",
        "object-prediction-time",
        "timezone-aware-prediction-time",
        "Boolean-numeric",
        "nonnumeric",
        "positive-infinity",
        "negative-infinity",
    ),
)
def test_frame_contract_rejects_invalid_values(argument: str, case: str) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    frame = train if argument == "train" else validation
    if case == "duplicate-appointment":
        frame.loc[frame.index[1], "appointment_id"] = frame.iloc[0][
            "appointment_id"
        ]
    elif case == "null-appointment":
        frame["appointment_id"] = frame["appointment_id"].astype("Int64")
        frame.loc[frame.index[0], "appointment_id"] = pd.NA
    elif case == "null-prediction-time":
        frame.loc[frame.index[0], "prediction_time"] = pd.NaT
    elif case == "object-prediction-time":
        frame["prediction_time"] = frame["prediction_time"].astype(str)
    elif case == "timezone-aware-prediction-time":
        frame["prediction_time"] = frame["prediction_time"].dt.tz_localize("UTC")
    elif case == "Boolean-numeric":
        frame["planned_duration_min"] = pd.Series(
            [True, False, True, False],
            dtype="bool",
        )
    elif case == "nonnumeric":
        frame["planned_duration_min"] = frame["planned_duration_min"].astype(str)
    elif case == "positive-infinity":
        frame.loc[frame.index[0], "planned_duration_min"] = np.inf
    elif case == "negative-infinity":
        frame.loc[frame.index[0], "planned_duration_min"] = -np.inf

    with pytest.raises(ValueError):
        summarize_numeric_drift(train, validation)


@pytest.mark.parametrize("argument", ("train", "validation"))
def test_complex_numeric_dtype_is_rejected_without_mutation(argument: str) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    frame = train if argument == "train" else validation
    frame["planned_duration_min"] = pd.Series(
        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j],
        dtype="complex128",
    )
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)

    with pytest.raises(ValueError, match=r"planned_duration_min.*complex"):
        summarize_numeric_drift(train, validation)

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_independent_temporal_boundaries_are_accepted() -> None:
    train = _drift_fixture(
        (1,),
        (EXPECTED_VALIDATION_START - pd.Timedelta(1, unit="ns"),),
    )
    validation = _drift_fixture(
        (2, 3),
        (
            EXPECTED_VALIDATION_START,
            EXPECTED_TEST_START - pd.Timedelta(1, unit="ns"),
        ),
    )

    result = summarize_numeric_drift(train, validation)
    assert result["train_rows"].eq(1).all()
    assert result["validation_rows"].eq(2).all()


@pytest.mark.parametrize(
    "case",
    (
        "train-at-validation-start",
        "validation-before-start",
        "validation-at-test-start",
        "validation-after-test-start",
        "swapped-populations",
    ),
)
def test_temporal_population_guards_reject_wrong_periods(case: str) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    if case == "train-at-validation-start":
        train.loc[train.index[0], "prediction_time"] = EXPECTED_VALIDATION_START
    elif case == "validation-before-start":
        validation.loc[validation.index[0], "prediction_time"] = (
            EXPECTED_VALIDATION_START - pd.Timedelta(1, unit="ns")
        )
    elif case == "validation-at-test-start":
        validation.loc[validation.index[0], "prediction_time"] = EXPECTED_TEST_START
    elif case == "validation-after-test-start":
        validation.loc[validation.index[0], "prediction_time"] = (
            EXPECTED_TEST_START + pd.Timedelta(days=1)
        )
    elif case == "swapped-populations":
        train, validation = validation, train

    with pytest.raises(ValueError, match="prediction_time"):
        summarize_numeric_drift(train, validation)


def test_cross_population_appointment_overlap_is_rejected() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    validation.loc[validation.index[0], "appointment_id"] = train.iloc[0][
        "appointment_id"
    ]

    with pytest.raises(ValueError, match=r"appointment IDs.*disjoint"):
        summarize_numeric_drift(train, validation)


def test_hand_worked_descriptions_shifts_and_signed_smd() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", [0.0, 1.0, 2.0, 4.0])
    _set_numeric_values(validation, "planned_duration_min", [1.0, 3.0, 5.0, 7.0])

    row = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )
    train_std = math.sqrt(35.0 / 12.0)
    validation_std = math.sqrt(20.0 / 3.0)
    expected_smd = (4.0 - 1.75) / math.sqrt(
        (train_std**2 + validation_std**2) / 2.0
    )
    assert row["train_rows"] == 4
    assert row["validation_rows"] == 4
    assert row["train_n"] == 4
    assert row["validation_n"] == 4
    assert row["train_missing_count"] == 0
    assert row["validation_missing_count"] == 0
    assert row["train_mean"] == 1.75
    assert row["validation_mean"] == 4.0
    assert row["train_std"] == pytest.approx(train_std, abs=1e-15)
    assert row["validation_std"] == pytest.approx(validation_std, abs=1e-15)
    assert row["train_q10"] == pytest.approx(0.3, abs=1e-15)
    assert row["validation_q10"] == pytest.approx(1.6, abs=1e-15)
    assert row["q10_shift"] == pytest.approx(1.3, abs=1e-15)
    assert row["train_median"] == 1.5
    assert row["validation_median"] == 4.0
    assert row["median_shift"] == 2.5
    assert row["train_q90"] == pytest.approx(3.4, abs=1e-15)
    assert row["validation_q90"] == pytest.approx(6.4, abs=1e-15)
    assert row["q90_shift"] == pytest.approx(3.0, abs=1e-15)
    assert row["signed_smd"] == pytest.approx(expected_smd, abs=1e-15)


@pytest.mark.parametrize(
    ("train_values", "validation_values", "expected_sign"),
    (
        ([0.0, 1.0, 2.0], [2.0, 3.0, 4.0], 1),
        ([0.0, 1.0, 2.0], [-2.0, -1.0, 0.0], -1),
    ),
)
def test_signed_smd_preserves_validation_relative_to_train_sign(
    train_values: list[float],
    validation_values: list[float],
    expected_sign: int,
) -> None:
    train = _train_fixture(3)
    validation = _validation_fixture(3)
    _set_numeric_values(train, "planned_duration_min", train_values)
    _set_numeric_values(validation, "planned_duration_min", validation_values)

    smd = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )["signed_smd"]
    assert math.copysign(1.0, smd) == expected_sign


@pytest.mark.parametrize(
    ("train_values", "validation_values", "expected_difference"),
    (
        ([1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 3.0, 4.0], 0.25),
        ([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], -0.25),
        ([1.0, np.nan, 3.0, 4.0], [1.0, np.nan, 3.0, 4.0], 0.0),
    ),
)
def test_missing_rate_difference_direction(
    train_values: list[float],
    validation_values: list[float],
    expected_difference: float,
) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", train_values)
    _set_numeric_values(validation, "planned_duration_min", validation_values)
    row = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )

    assert row["missing_rate_difference"] == expected_difference


def test_empty_and_all_null_missingness_policies() -> None:
    empty_train = _train_fixture().iloc[:0].copy(deep=True)
    empty_validation = _validation_fixture().iloc[:0].copy(deep=True)
    both_empty = summarize_numeric_drift(empty_train, empty_validation)
    assert tuple(both_empty.columns) == EXPECTED_OUTPUT_COLUMNS
    assert tuple(both_empty["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert both_empty["train_rows"].eq(0).all()
    assert both_empty["validation_rows"].eq(0).all()
    assert both_empty["train_n"].eq(0).all()
    assert both_empty["validation_n"].eq(0).all()
    assert both_empty["train_missing_rate"].isna().all()
    assert both_empty["validation_missing_rate"].isna().all()
    assert both_empty["missing_rate_difference"].isna().all()
    assert tuple(both_empty.dtypes.astype(str)) == EXPECTED_OUTPUT_DTYPES

    validation = _validation_fixture()
    one_empty = _result_row(
        summarize_numeric_drift(empty_train, validation),
        "planned_duration_min",
    )
    assert math.isnan(one_empty["train_missing_rate"])
    assert one_empty["validation_missing_rate"] == 0.0
    assert math.isnan(one_empty["missing_rate_difference"])
    assert math.isnan(one_empty["signed_smd"])

    train = _train_fixture()
    _set_numeric_values(train, "planned_duration_min", [np.nan] * 4)
    all_null = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )
    assert all_null["train_n"] == 0
    assert all_null["train_missing_count"] == 4
    assert all_null["train_missing_rate"] == 1.0
    assert all_null[
        ["train_mean", "train_std", "train_q10", "train_median", "train_q90"]
    ].isna().all()
    assert math.isnan(all_null["signed_smd"])


@pytest.mark.parametrize(
    ("train_values", "validation_values", "expected_smd"),
    (
        ([5.0, 5.0], [5.0, 5.0], 0.0),
        ([5.0, 5.0], [6.0, 6.0], math.nan),
        ([5.0, 5.0], [6.0, 8.0], 2.0),
        ([5.0], [6.0, 8.0], math.nan),
    ),
    ids=("equal-constants", "unequal-constants", "constant-variable", "singleton"),
)
def test_signed_smd_edge_cases(
    train_values: list[float],
    validation_values: list[float],
    expected_smd: float,
) -> None:
    train = _train_fixture(len(train_values))
    validation = _validation_fixture(len(validation_values))
    _set_numeric_values(train, "planned_duration_min", train_values)
    _set_numeric_values(validation, "planned_duration_min", validation_values)
    row = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )

    if math.isnan(expected_smd):
        assert math.isnan(row["signed_smd"])
    else:
        assert row["signed_smd"] == pytest.approx(expected_smd, abs=1e-15)


def test_linear_quantiles_and_full_precision_are_preserved() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", [0.0, 10.0, 20.0, 30.0])
    _set_numeric_values(validation, "planned_duration_min", [1.0, 11.0, 21.0, 31.0])
    row = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )

    assert row["train_q10"] == pytest.approx(3.0, abs=1e-14)
    assert row["train_median"] == pytest.approx(15.0, abs=1e-14)
    assert row["train_q90"] == pytest.approx(27.0, abs=1e-14)
    assert row["validation_q10"] == pytest.approx(4.0, abs=1e-14)
    assert row["validation_median"] == pytest.approx(16.0, abs=1e-14)
    assert row["validation_q90"] == pytest.approx(28.0, abs=1e-14)
    assert row["q10_shift"] == pytest.approx(1.0, abs=1e-14)
    assert row["median_shift"] == pytest.approx(1.0, abs=1e-14)
    assert row["q90_shift"] == pytest.approx(1.0, abs=1e-14)

    precise_train = _train_fixture()
    _set_numeric_values(precise_train, "planned_duration_min", [0.0, 1.0, 2.0, 4.0])
    precise = _result_row(
        summarize_numeric_drift(precise_train, validation),
        "planned_duration_min",
    )
    expected_std = math.sqrt(35.0 / 12.0)
    assert math.isclose(
        precise["train_std"], expected_std, rel_tol=0.0, abs_tol=1e-15
    )
    assert not math.isclose(
        precise["train_std"],
        round(expected_std, 6),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_centered_mean_preserves_ordinary_cancellation_residual() -> None:
    values = [1e308, 1.0, -1e308]
    expected_mean = _decimal_mean_of_exact_floats(values)
    train = _train_fixture(3)
    validation = _validation_fixture(3)
    _set_numeric_values(train, "planned_duration_min", values)
    _set_numeric_values(validation, "planned_duration_min", values)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)

    row = _result_row(result, "planned_duration_min")
    assert expected_mean == 1.0 / 3.0
    assert row["train_mean"] == expected_mean
    assert row["validation_mean"] == expected_mean
    assert row["train_mean"] != 0.33333333333333315
    _assert_no_infinite_float_outputs(result)


def test_centered_mean_preserves_subnormal_residual_and_permutations() -> None:
    values = [1e308, 1e-308, -1e308]
    expected_mean = _decimal_mean_of_exact_floats(values)
    train = _train_fixture(3)
    validation = _validation_fixture(3)
    _set_numeric_values(train, "planned_duration_min", values)
    _set_numeric_values(validation, "planned_duration_min", values)
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)
        reversed_result = summarize_numeric_drift(
            train.iloc[::-1],
            validation.iloc[::-1],
        )
        shuffled_result = summarize_numeric_drift(
            train.sample(frac=1.0, random_state=31_001),
            validation.sample(frac=1.0, random_state=31_002),
        )

    row = _result_row(result, "planned_duration_min")
    assert expected_mean != 0.0
    assert math.isfinite(expected_mean)
    assert 0.0 < abs(expected_mean) < np.finfo("float64").tiny
    assert row["train_mean"] == expected_mean
    assert row["validation_mean"] == expected_mean
    assert row["train_mean"] != 0.0
    assert not math.isnan(row["train_mean"])
    _assert_no_infinite_float_outputs(result)
    pd.testing.assert_frame_equal(result, reversed_result)
    pd.testing.assert_frame_equal(result, shuffled_result)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_extreme_constant_values_are_stable_warning_free_and_deterministic() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", [1e308] * 4)
    _set_numeric_values(validation, "planned_duration_min", [1e308] * 4)
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)
        reversed_result = summarize_numeric_drift(
            train.iloc[::-1],
            validation.iloc[::-1],
        )
        shuffled_result = summarize_numeric_drift(
            train.sample(frac=1.0, random_state=30_001),
            validation.sample(frac=1.0, random_state=30_002),
        )

    row = _result_row(result, "planned_duration_min")
    for column in ("train_mean", "validation_mean"):
        assert math.isfinite(row[column])
        assert row[column] == pytest.approx(1e308, rel=1e-15)
    assert row["train_std"] == 0.0
    assert row["validation_std"] == 0.0
    for column in (
        "train_q10",
        "train_median",
        "train_q90",
        "validation_q10",
        "validation_median",
        "validation_q90",
    ):
        assert math.isfinite(row[column])
        assert row[column] == pytest.approx(1e308, rel=1e-15)
    assert row["q10_shift"] == 0.0
    assert row["median_shift"] == 0.0
    assert row["q90_shift"] == 0.0
    assert row["signed_smd"] == 0.0
    _assert_no_infinite_float_outputs(result)
    pd.testing.assert_frame_equal(result, reversed_result)
    pd.testing.assert_frame_equal(result, shuffled_result)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_negative_extreme_constants_remain_finite_and_warning_free() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", [-1e308] * 4)
    _set_numeric_values(validation, "planned_duration_min", [-1e308] * 4)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)

    row = _result_row(result, "planned_duration_min")
    assert math.isfinite(row["train_mean"])
    assert math.isfinite(row["validation_mean"])
    assert row["train_mean"] == pytest.approx(-1e308, rel=1e-15)
    assert row["validation_mean"] == pytest.approx(-1e308, rel=1e-15)
    assert row["train_std"] == 0.0
    assert row["validation_std"] == 0.0
    for column in (
        "train_q10",
        "train_median",
        "train_q90",
        "validation_q10",
        "validation_median",
        "validation_q90",
    ):
        assert math.isfinite(row[column])
        assert row[column] == pytest.approx(-1e308, rel=1e-15)
    _assert_no_infinite_float_outputs(result)


def test_extreme_opposite_signs_use_stable_quantiles_and_nan_std() -> None:
    train = _train_fixture(2)
    validation = _validation_fixture(2)
    extreme = 1.3e308
    _set_numeric_values(train, "planned_duration_min", [-extreme, extreme])
    _set_numeric_values(validation, "planned_duration_min", [-extreme, extreme])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)

    row = _result_row(result, "planned_duration_min")
    expected_q10 = math.fsum((-0.9 * extreme, 0.1 * extreme))
    expected_q90 = math.fsum((-0.1 * extreme, 0.9 * extreme))
    assert math.isfinite(row["train_mean"])
    assert row["train_mean"] == pytest.approx(0.0, abs=0.0)
    assert row["train_median"] == pytest.approx(0.0, abs=0.0)
    assert row["train_q10"] == pytest.approx(expected_q10, rel=1e-15)
    assert row["train_q90"] == pytest.approx(expected_q90, rel=1e-15)
    assert row["train_q10"] < 0.0 < row["train_q90"]
    assert math.isnan(row["train_std"])
    assert math.isnan(row["validation_std"])
    assert math.isnan(row["signed_smd"])
    _assert_no_infinite_float_outputs(result)


def test_extreme_representable_pooled_scale_and_smd_are_finite() -> None:
    train = _train_fixture(2)
    validation = _validation_fixture(2)
    _set_numeric_values(train, "planned_duration_min", [0.0, 1e308])
    _set_numeric_values(validation, "planned_duration_min", [1e307, 1.1e308])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)

    row = _result_row(result, "planned_duration_min")
    expected_std = 1e308 / math.sqrt(2.0)
    expected_smd = 1e307 / expected_std
    assert math.isfinite(row["train_std"])
    assert math.isfinite(row["validation_std"])
    assert row["train_std"] == pytest.approx(expected_std, rel=1e-15)
    assert row["validation_std"] == pytest.approx(expected_std, rel=1e-15)
    assert math.isfinite(row["signed_smd"])
    assert row["signed_smd"] == pytest.approx(expected_smd, rel=1e-15)
    _assert_no_infinite_float_outputs(result)


def test_unrepresentable_finite_shifts_become_nan_without_warning() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", [-1e308] * 4)
    _set_numeric_values(validation, "planned_duration_min", [1e308] * 4)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_numeric_drift(train, validation)

    row = _result_row(result, "planned_duration_min")
    assert math.isfinite(row["train_mean"])
    assert math.isfinite(row["validation_mean"])
    for column in (
        "train_q10",
        "train_median",
        "train_q90",
        "validation_q10",
        "validation_median",
        "validation_q90",
    ):
        assert math.isfinite(row[column])
    assert math.isnan(row["q10_shift"])
    assert math.isnan(row["median_shift"])
    assert math.isnan(row["q90_shift"])
    assert math.isnan(row["signed_smd"])
    _assert_no_infinite_float_outputs(result)


def test_negative_quantile_shifts_remain_signed() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(train, "planned_duration_min", [1.0, 11.0, 21.0, 31.0])
    _set_numeric_values(validation, "planned_duration_min", [0.0, 10.0, 20.0, 30.0])

    row = _result_row(
        summarize_numeric_drift(train, validation),
        "planned_duration_min",
    )
    assert row["q10_shift"] == pytest.approx(-1.0, rel=0.0, abs=1e-15)
    assert row["median_shift"] == -1.0
    assert row["q90_shift"] == pytest.approx(-1.0, rel=0.0, abs=1e-15)


@pytest.mark.parametrize("forced_infinity", (math.inf, -math.inf))
def test_final_postcondition_rejects_forced_infinity_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    forced_infinity: float,
) -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)
    monkeypatch.setattr(drift, "_stable_mean", lambda values: forced_infinity)

    with pytest.raises(ValueError, match=r"output.*infinity"):
        summarize_numeric_drift(train, validation)

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_non_mutation_and_row_order_determinism() -> None:
    train = _train_fixture()
    validation = _validation_fixture()
    _set_numeric_values(
        train,
        "planned_duration_min",
        [3.123456789, np.nan, -2.0, 7.0],
    )
    _set_numeric_values(
        validation,
        "planned_duration_min",
        [8.0, -1.987654321, np.nan, 2.0],
    )
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)

    first = summarize_numeric_drift(train, validation)
    repeated = summarize_numeric_drift(train, validation)
    reversed_result = summarize_numeric_drift(
        train.iloc[::-1],
        validation.iloc[::-1],
    )
    shuffled_result = summarize_numeric_drift(
        train.sample(frac=1.0, random_state=2_027),
        validation.sample(frac=1.0, random_state=2_028),
    )

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)
    pd.testing.assert_frame_equal(first, repeated)
    pd.testing.assert_frame_equal(first, reversed_result)
    pd.testing.assert_frame_equal(first, shuffled_result)
    assert tuple(first["feature"]) == EXPECTED_NUMERIC_FEATURES
    assert first is not train
    assert first is not validation

    first.iloc[0, -1] = -999.0
    first[first.columns[-1]] = -888.0
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_selector_outcome_and_population_non_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_source = inspect.getsource(drift)
    for forbidden_name in (
        "select_eda_populations",
        "select_development_rows",
        "select_test_rows",
        "select_model_features",
        "build_canonical_dataset",
        "build_analytical_dataset",
        "test_drift",
    ):
        assert forbidden_name not in module_source
    assert re.search(r"\btarget\b", module_source, flags=re.IGNORECASE) is None

    train = _train_fixture()
    validation = _validation_fixture()

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("numeric drift attempted data selection")

    monkeypatch.setattr(run_eda, "select_eda_populations", fail_if_called)
    for owner, name in (
        (bd, "select_development_rows"),
        (bd, "select_test_rows"),
        (bd, "select_model_features"),
        (bd, "build_analytical_dataset"),
    ):
        monkeypatch.setattr(owner, name, fail_if_called)

    result = summarize_numeric_drift(train, validation)
    assert isinstance(result, pd.DataFrame)
