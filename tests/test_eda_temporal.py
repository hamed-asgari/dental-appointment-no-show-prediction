"""Tests for deterministic train-only temporal EDA summaries."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import run_eda, summaries
from src.analysis.run_eda import select_eda_populations
from src.analysis.summaries import (
    summarize_temporal_coverage,
    summarize_temporal_monthly,
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
EXPECTED_SUPERVISED_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "target",
    *EXPECTED_FEATURE_COLUMNS,
)
EXPECTED_AUDIT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
    "development_fit_eligible",
)
EXPECTED_COVERAGE_COLUMNS = (
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
EXPECTED_MONTHLY_COLUMNS = (
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
EXPECTED_WILSON_Z = 1.959963984540054
EXPECTED_NOMINAL_ROWS = 3_682
EXPECTED_MATURE_ROWS = 3_670
EXPECTED_EXCLUSIONS = 12
EXPECTED_POSITIVES = 432
EXPECTED_NEGATIVES = 3_238
EXPECTED_NOMINAL_MIN = pd.Timestamp("2024-03-01 09:00:00")
EXPECTED_NOMINAL_MAX = pd.Timestamp("2025-02-28 18:00:00")
EXPECTED_MATURE_PREDICTION_TIME_MIN = pd.Timestamp("2024-03-01 09:00:00")
EXPECTED_MATURE_MAX = pd.Timestamp("2025-02-26 18:30:00")
EXPECTED_FIRST_MONTH = "2024-03"
EXPECTED_LAST_MONTH = "2025-02"
EXPECTED_MONTH_SPAN = 12
EXPECTED_COVERAGE_DTYPES = (
    "int64",
    "int64",
    "int64",
    "datetime64[ns]",
    "datetime64[ns]",
    "datetime64[ns]",
    "datetime64[ns]",
    "string",
    "string",
    "int64",
)
EXPECTED_MONTHLY_DTYPES = (
    "string",
    "int64",
    "int64",
    "int64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
)
TEMPORAL_FUNCTIONS = (
    summarize_temporal_coverage,
    summarize_temporal_monthly,
)


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(tables)


@pytest.fixture(scope="session")
def train_populations(
    canonical_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    populations = select_eda_populations(canonical_dataset)
    return populations["supervised_train"], populations["maturity_audit"]


def _supervised_fixture(
    appointment_ids: tuple[int, ...] = (1, 2, 3),
    prediction_times: tuple[pd.Timestamp | str, ...] = (
        "2025-01-05 09:00:00",
        "2025-01-20 10:00:00",
        "2025-03-01 11:00:00",
    ),
    targets: tuple[int, ...] = (0, 1, 1),
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
            "target": pd.Series(targets, dtype="int8"),
            "planned_duration_min": pd.Series(values, dtype="float64"),
            "visit_type": pd.Series(["exam"] * rows, dtype="object"),
            "booking_channel": pd.Series(["phone"] * rows, dtype="object"),
            "booking_lead_time_hours": pd.Series(
                24.0 + values,
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(values % 7, dtype="float64"),
            "scheduled_hour": pd.Series(9.0 + values % 8, dtype="float64"),
            "scheduled_month": pd.Series(12.0 - values % 12, dtype="float64"),
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


def _audit_fixture(
    appointment_ids: tuple[int, ...] = (1, 2, 3),
    prediction_times: tuple[pd.Timestamp | str, ...] = (
        "2025-01-05 09:00:00",
        "2025-01-20 10:00:00",
        "2025-03-01 11:00:00",
    ),
    eligible: tuple[bool, ...] = (True, True, True),
) -> pd.DataFrame:
    rows = len(appointment_ids)
    frame = pd.DataFrame(
        {
            "appointment_id": pd.Series(appointment_ids, dtype="int64"),
            "prediction_time": pd.Series(
                pd.to_datetime(list(prediction_times)),
                dtype="datetime64[ns]",
            ),
            "split": pd.Series(["train"] * rows, dtype="string"),
            "development_fit_eligible": pd.Series(eligible, dtype="bool"),
        }
    )
    return frame.loc[:, EXPECTED_AUDIT_COLUMNS]


def _hand_worked_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = _audit_fixture(
        appointment_ids=(1, 2, 3, 4, 5),
        prediction_times=(
            "2025-01-05 09:00:00",
            "2025-01-20 10:00:00",
            "2025-01-31 12:00:00",
            "2025-03-01 11:00:00",
            "2025-03-15 13:00:00",
        ),
        eligible=(True, False, True, True, False),
    )
    supervised = _supervised_fixture(
        appointment_ids=(1, 3, 4),
        prediction_times=(
            "2025-01-05 09:00:00",
            "2025-01-31 12:00:00",
            "2025-03-01 11:00:00",
        ),
        targets=(0, 1, 1),
    )
    return supervised, audit


def _distinct_temporal_extrema_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = _audit_fixture(
        appointment_ids=(10, 11, 12),
        prediction_times=("2025-01-01", "2025-01-05", "2025-01-10"),
        eligible=(False, True, True),
    )
    supervised = _supervised_fixture(
        appointment_ids=(11, 12),
        prediction_times=("2025-01-05", "2025-01-10"),
        targets=(0, 1),
    )
    return supervised, audit


def _expected_wilson(positives: int, count: int) -> tuple[float, float]:
    if count == 0:
        return math.nan, math.nan
    rate = positives / count
    z_squared = EXPECTED_WILSON_Z**2
    denominator = 1.0 + z_squared / count
    center = (rate + z_squared / (2.0 * count)) / denominator
    radius = (
        EXPECTED_WILSON_Z
        * math.sqrt(
            rate * (1.0 - rate) / count
            + z_squared / (4.0 * count**2)
        )
        / denominator
    )
    return center - radius, center + radius


def _monthly_row(result: pd.DataFrame, month: str) -> pd.Series:
    matching = result.loc[result["prediction_month"].eq(month)]
    assert len(matching) == 1
    return matching.iloc[0]


def test_real_train_only_temporal_integration(
    train_populations: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    supervised, audit = train_populations
    coverage = summarize_temporal_coverage(supervised, audit)
    monthly = summarize_temporal_monthly(supervised, audit)
    coverage_row = coverage.iloc[0]

    assert tuple(supervised.columns) == EXPECTED_SUPERVISED_COLUMNS
    assert tuple(audit.columns) == EXPECTED_AUDIT_COLUMNS
    assert tuple(coverage.columns) == EXPECTED_COVERAGE_COLUMNS
    assert tuple(monthly.columns) == EXPECTED_MONTHLY_COLUMNS
    assert coverage_row["nominal_train_rows"] == EXPECTED_NOMINAL_ROWS
    assert coverage_row["mature_train_rows"] == EXPECTED_MATURE_ROWS
    assert coverage_row["maturity_exclusion_rows"] == EXPECTED_EXCLUSIONS
    assert coverage_row["nominal_prediction_time_min"] == EXPECTED_NOMINAL_MIN
    assert coverage_row["nominal_prediction_time_max"] == EXPECTED_NOMINAL_MAX
    assert (
        coverage_row["mature_prediction_time_min"]
        == EXPECTED_MATURE_PREDICTION_TIME_MIN
    )
    assert coverage_row["mature_prediction_time_max"] == EXPECTED_MATURE_MAX
    assert coverage_row["first_prediction_month"] == EXPECTED_FIRST_MONTH
    assert coverage_row["last_prediction_month"] == EXPECTED_LAST_MONTH
    assert coverage_row["calendar_months_spanned"] == EXPECTED_MONTH_SPAN

    expected_months = [
        period.strftime("%Y-%m")
        for period in pd.period_range("2024-03", "2025-02", freq="M")
    ]
    assert monthly["prediction_month"].tolist() == expected_months
    assert len(monthly) == EXPECTED_MONTH_SPAN
    assert monthly["nominal_train_count"].sum() == EXPECTED_NOMINAL_ROWS
    assert monthly["mature_train_count"].sum() == EXPECTED_MATURE_ROWS
    assert monthly["maturity_exclusion_count"].sum() == EXPECTED_EXCLUSIONS
    assert monthly["positives"].sum() == EXPECTED_POSITIVES
    assert monthly["negatives"].sum() == EXPECTED_NEGATIVES
    assert monthly.loc[
        monthly["prediction_month"].ne("2025-02"),
        "maturity_exclusion_count",
    ].eq(0).all()
    assert _monthly_row(monthly, "2025-02")["maturity_exclusion_count"] == 12

    for row in monthly.itertuples(index=False):
        assert row.mature_train_count == row.positives + row.negatives
        assert row.maturity_exclusion_count >= 0
        if row.mature_train_count:
            assert row.no_show_rate == row.positives / row.mature_train_count
            assert math.isfinite(row.wilson_lower)
            assert math.isfinite(row.wilson_upper)
            assert row.wilson_lower <= row.no_show_rate <= row.wilson_upper


def test_exact_public_signatures_and_output_contracts() -> None:
    coverage_signature = inspect.signature(summarize_temporal_coverage)
    monthly_signature = inspect.signature(summarize_temporal_monthly)
    assert tuple(coverage_signature.parameters) == (
        "supervised_train",
        "maturity_audit",
    )
    assert tuple(monthly_signature.parameters) == (
        "supervised_train",
        "maturity_audit",
    )


@pytest.mark.parametrize(
    "case",
    (
        "missing-column",
        "extra-column",
        "target-bearing",
        "reordered-columns",
        "null-appointment",
        "duplicate-appointment",
        "null-prediction-time",
        "string-prediction-time",
        "timezone-aware-prediction-time",
        "null-split",
        "non-train-split",
        "null-eligibility",
        "non-Boolean-eligibility",
    ),
)
def test_maturity_audit_contract_rejects_invalid_frames(case: str) -> None:
    supervised = _supervised_fixture()
    audit = _audit_fixture()
    if case == "missing-column":
        audit = audit.drop(columns="split")
    elif case == "extra-column":
        audit = audit.assign(extra="forbidden")
    elif case == "target-bearing":
        audit = audit.assign(target=0)
    elif case == "reordered-columns":
        audit = audit.loc[:, list(reversed(EXPECTED_AUDIT_COLUMNS))]
    elif case == "null-appointment":
        audit["appointment_id"] = audit["appointment_id"].astype("Int64")
        audit.loc[audit.index[0], "appointment_id"] = pd.NA
    elif case == "duplicate-appointment":
        audit.loc[audit.index[1], "appointment_id"] = audit.iloc[0][
            "appointment_id"
        ]
    elif case == "null-prediction-time":
        audit.loc[audit.index[0], "prediction_time"] = pd.NaT
    elif case == "string-prediction-time":
        audit["prediction_time"] = audit["prediction_time"].astype(str)
    elif case == "timezone-aware-prediction-time":
        audit["prediction_time"] = audit["prediction_time"].dt.tz_localize("UTC")
    elif case == "null-split":
        audit.loc[audit.index[0], "split"] = pd.NA
    elif case == "non-train-split":
        audit.loc[audit.index[0], "split"] = "validation"
    elif case == "null-eligibility":
        audit["development_fit_eligible"] = audit[
            "development_fit_eligible"
        ].astype("boolean")
        audit.loc[audit.index[0], "development_fit_eligible"] = pd.NA
    elif case == "non-Boolean-eligibility":
        audit["development_fit_eligible"] = audit[
            "development_fit_eligible"
        ].astype("int8")

    with pytest.raises(ValueError):
        summarize_temporal_coverage(supervised, audit)


@pytest.mark.parametrize("temporal_function", TEMPORAL_FUNCTIONS)
def test_maturity_audit_contract_runs_at_both_public_entry_points(
    temporal_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    audit = _audit_fixture()
    audit.loc[audit.index[0], "split"] = "validation"
    with pytest.raises(ValueError, match=r"split.*train"):
        temporal_function(_supervised_fixture(), audit)


@pytest.mark.parametrize("temporal_function", TEMPORAL_FUNCTIONS)
def test_maturity_audit_rejects_non_dataframe(
    temporal_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    with pytest.raises(TypeError, match=r"maturity_audit.*pandas DataFrame"):
        temporal_function(_supervised_fixture(), [1, 2])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    (
        "duplicate-appointment",
        "string-prediction-time",
        "timezone-aware-prediction-time",
    ),
)
def test_supervised_temporal_contract_rejects_invalid_frames(case: str) -> None:
    supervised = _supervised_fixture()
    audit = _audit_fixture()
    if case == "duplicate-appointment":
        supervised.loc[supervised.index[1], "appointment_id"] = supervised.iloc[0][
            "appointment_id"
        ]
    elif case == "string-prediction-time":
        supervised["prediction_time"] = supervised["prediction_time"].astype(str)
    elif case == "timezone-aware-prediction-time":
        supervised["prediction_time"] = supervised[
            "prediction_time"
        ].dt.tz_localize("UTC")

    with pytest.raises(ValueError):
        summarize_temporal_monthly(supervised, audit)


def test_existing_supervised_contract_still_runs() -> None:
    supervised = _supervised_fixture().drop(columns="visit_type")
    with pytest.raises(ValueError, match="approved projection"):
        summarize_temporal_coverage(supervised, _audit_fixture())


@pytest.mark.parametrize(
    "case",
    (
        "missing-supervised-appointment",
        "extra-supervised-appointment",
        "eligible-audit-row-absent",
        "supervised-row-is-ineligible",
        "prediction-time-mismatch",
        "empty-supervised-with-eligible-audit",
    ),
)
def test_cross_population_mismatches_are_rejected(case: str) -> None:
    supervised = _supervised_fixture()
    audit = _audit_fixture()
    if case == "missing-supervised-appointment":
        supervised = supervised.iloc[:2].copy(deep=True)
    elif case == "extra-supervised-appointment":
        audit = audit.iloc[:2].copy(deep=True)
    elif case == "eligible-audit-row-absent":
        supervised = supervised.iloc[:2].copy(deep=True)
        audit.loc[audit.index[2], "development_fit_eligible"] = True
    elif case == "supervised-row-is-ineligible":
        audit.loc[audit.index[2], "development_fit_eligible"] = False
    elif case == "prediction-time-mismatch":
        audit.loc[audit.index[0], "prediction_time"] += pd.Timedelta(seconds=1)
    elif case == "empty-supervised-with-eligible-audit":
        supervised = supervised.iloc[:0].copy(deep=True)

    with pytest.raises(ValueError, match=r"appointment IDs|prediction_time"):
        summarize_temporal_monthly(supervised, audit)


def test_hand_worked_coverage_and_monthly_gap() -> None:
    supervised, audit = _hand_worked_inputs()
    coverage = summarize_temporal_coverage(supervised, audit)
    monthly = summarize_temporal_monthly(supervised, audit)
    row = coverage.iloc[0]

    assert row["nominal_train_rows"] == 5
    assert row["mature_train_rows"] == 3
    assert row["maturity_exclusion_rows"] == 2
    assert row["nominal_prediction_time_min"] == pd.Timestamp("2025-01-05 09:00")
    assert row["nominal_prediction_time_max"] == pd.Timestamp("2025-03-15 13:00")
    assert row["mature_prediction_time_min"] == pd.Timestamp("2025-01-05 09:00")
    assert row["mature_prediction_time_max"] == pd.Timestamp("2025-03-01 11:00")
    assert row["first_prediction_month"] == "2025-01"
    assert row["last_prediction_month"] == "2025-03"
    assert row["calendar_months_spanned"] == 3
    assert monthly["prediction_month"].tolist() == [
        "2025-01",
        "2025-02",
        "2025-03",
    ]

    january = _monthly_row(monthly, "2025-01")
    expected_lower, expected_upper = _expected_wilson(1, 2)
    assert january["nominal_train_count"] == 3
    assert january["mature_train_count"] == 2
    assert january["maturity_exclusion_count"] == 1
    assert january["positives"] == 1
    assert january["negatives"] == 1
    assert january["no_show_rate"] == 0.5
    assert january["wilson_lower"] == pytest.approx(
        expected_lower,
        rel=0.0,
        abs=1e-15,
    )
    assert january["wilson_upper"] == pytest.approx(
        expected_upper,
        rel=0.0,
        abs=1e-15,
    )
    assert not math.isclose(
        january["wilson_lower"],
        round(expected_lower, 6),
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    february = _monthly_row(monthly, "2025-02")
    assert february[
        [
            "nominal_train_count",
            "mature_train_count",
            "maturity_exclusion_count",
            "positives",
            "negatives",
        ]
    ].eq(0).all()
    assert february[["no_show_rate", "wilson_lower", "wilson_upper"]].isna().all()

    march = _monthly_row(monthly, "2025-03")
    assert march["nominal_train_count"] == 2
    assert march["mature_train_count"] == 1
    assert march["maturity_exclusion_count"] == 1
    assert march["positives"] == 1
    assert march["negatives"] == 0
    assert march["no_show_rate"] == 1.0


def test_mature_extrema_come_from_supervised_population() -> None:
    supervised, audit = _distinct_temporal_extrema_inputs()

    coverage = summarize_temporal_coverage(supervised, audit).iloc[0]

    assert coverage["nominal_train_rows"] == 3
    assert coverage["mature_train_rows"] == 2
    assert coverage["maturity_exclusion_rows"] == 1
    assert coverage["nominal_prediction_time_min"] == pd.Timestamp("2025-01-01")
    assert coverage["mature_prediction_time_min"] == pd.Timestamp("2025-01-05")
    assert coverage["nominal_prediction_time_max"] == pd.Timestamp("2025-01-10")
    assert coverage["mature_prediction_time_max"] == pd.Timestamp("2025-01-10")


def test_prediction_time_nanosecond_month_boundary_not_scheduled_month() -> None:
    january_end = pd.Timestamp("2025-01-31 23:59:59.999999999")
    february_start = pd.Timestamp("2025-02-01 00:00:00")
    supervised = _supervised_fixture(
        appointment_ids=(1, 2),
        prediction_times=(january_end, february_start),
        targets=(0, 1),
    )
    supervised["scheduled_month"] = pd.Series([12.0, 12.0], dtype="float64")
    audit = _audit_fixture(
        appointment_ids=(1, 2),
        prediction_times=(january_end, february_start),
        eligible=(True, True),
    )

    result = summarize_temporal_monthly(supervised, audit)
    assert result["prediction_month"].tolist() == ["2025-01", "2025-02"]
    assert result["mature_train_count"].tolist() == [1, 1]
    assert result["positives"].tolist() == [0, 1]


def test_empty_inputs_have_exact_schemas_values_and_dtypes() -> None:
    supervised = _supervised_fixture().iloc[:0].copy(deep=True)
    audit = _audit_fixture().iloc[:0].copy(deep=True)
    coverage = summarize_temporal_coverage(supervised, audit)
    monthly = summarize_temporal_monthly(supervised, audit)
    row = coverage.iloc[0]

    assert tuple(coverage.columns) == EXPECTED_COVERAGE_COLUMNS
    assert len(coverage) == 1
    assert row["nominal_train_rows"] == 0
    assert row["mature_train_rows"] == 0
    assert row["maturity_exclusion_rows"] == 0
    for column in (
        "nominal_prediction_time_min",
        "nominal_prediction_time_max",
        "mature_prediction_time_min",
        "mature_prediction_time_max",
    ):
        assert pd.isna(row[column])
    assert pd.isna(row["first_prediction_month"])
    assert pd.isna(row["last_prediction_month"])
    assert row["calendar_months_spanned"] == 0
    assert tuple(coverage.dtypes.astype(str)) == EXPECTED_COVERAGE_DTYPES

    assert tuple(monthly.columns) == EXPECTED_MONTHLY_COLUMNS
    assert monthly.empty
    assert tuple(monthly.dtypes.astype(str)) == EXPECTED_MONTHLY_DTYPES


def test_audit_only_immature_population_retains_nominal_months() -> None:
    supervised = _supervised_fixture().iloc[:0].copy(deep=True)
    audit = _audit_fixture(
        appointment_ids=(10, 11),
        prediction_times=("2025-01-15", "2025-03-20"),
        eligible=(False, False),
    )
    coverage = summarize_temporal_coverage(supervised, audit).iloc[0]
    monthly = summarize_temporal_monthly(supervised, audit)

    assert coverage["nominal_train_rows"] == 2
    assert coverage["mature_train_rows"] == 0
    assert coverage["maturity_exclusion_rows"] == 2
    assert coverage["nominal_prediction_time_min"] == pd.Timestamp("2025-01-15")
    assert coverage["nominal_prediction_time_max"] == pd.Timestamp("2025-03-20")
    assert pd.isna(coverage["mature_prediction_time_min"])
    assert pd.isna(coverage["mature_prediction_time_max"])
    assert monthly["prediction_month"].tolist() == [
        "2025-01",
        "2025-02",
        "2025-03",
    ]
    assert monthly["nominal_train_count"].tolist() == [1, 0, 1]
    assert monthly["mature_train_count"].eq(0).all()
    assert monthly["maturity_exclusion_count"].tolist() == [1, 0, 1]
    assert monthly["positives"].eq(0).all()
    assert monthly["negatives"].eq(0).all()
    assert monthly["no_show_rate"].isna().all()
    assert monthly["wilson_lower"].isna().all()
    assert monthly["wilson_upper"].isna().all()


def test_monthly_wilson_edge_cases() -> None:
    supervised = _supervised_fixture(
        appointment_ids=(1, 2, 3, 4, 5, 6, 7),
        prediction_times=(
            "2025-01-05",
            "2025-01-06",
            "2025-02-05",
            "2025-02-06",
            "2025-03-05",
            "2025-03-06",
            "2025-04-05",
        ),
        targets=(0, 0, 1, 1, 0, 1, 1),
    )
    audit = _audit_fixture(
        appointment_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        prediction_times=(
            "2025-01-05",
            "2025-01-06",
            "2025-02-05",
            "2025-02-06",
            "2025-03-05",
            "2025-03-06",
            "2025-04-05",
            "2025-05-05",
        ),
        eligible=(True, True, True, True, True, True, True, False),
    )
    result = summarize_temporal_monthly(supervised, audit)

    for month, positives, count in (
        ("2025-01", 0, 2),
        ("2025-02", 2, 2),
        ("2025-03", 1, 2),
        ("2025-04", 1, 1),
    ):
        row = _monthly_row(result, month)
        expected_lower, expected_upper = _expected_wilson(positives, count)
        assert row["wilson_lower"] == pytest.approx(
            expected_lower,
            rel=0.0,
            abs=1e-15,
        )
        assert row["wilson_upper"] == pytest.approx(
            expected_upper,
            rel=0.0,
            abs=1e-15,
        )
        assert 0.0 <= row["wilson_lower"] <= 1.0
        assert 0.0 <= row["wilson_upper"] <= 1.0

    zero_mature = _monthly_row(result, "2025-05")
    assert zero_mature["nominal_train_count"] == 1
    assert zero_mature["mature_train_count"] == 0
    assert math.isnan(zero_mature["no_show_rate"])
    assert math.isnan(zero_mature["wilson_lower"])
    assert math.isnan(zero_mature["wilson_upper"])


@pytest.mark.parametrize("temporal_function", TEMPORAL_FUNCTIONS)
def test_non_mutation_new_outputs_and_determinism(
    temporal_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    supervised, audit = _hand_worked_inputs()
    supervised.index = pd.Index([100, 300, 500])
    audit.index = pd.Index([90, 80, 70, 60, 50])
    supervised_before = supervised.copy(deep=True)
    audit_before = audit.copy(deep=True)

    first = temporal_function(supervised, audit)
    repeated = temporal_function(supervised, audit)
    reversed_result = temporal_function(
        supervised.iloc[::-1],
        audit.iloc[::-1],
    )
    shuffled_result = temporal_function(
        supervised.sample(frac=1.0, random_state=2_025),
        audit.sample(frac=1.0, random_state=2_026),
    )

    pd.testing.assert_frame_equal(supervised, supervised_before)
    pd.testing.assert_frame_equal(audit, audit_before)
    pd.testing.assert_frame_equal(first, repeated)
    pd.testing.assert_frame_equal(first, reversed_result)
    pd.testing.assert_frame_equal(first, shuffled_result)
    assert first is not supervised
    assert first is not audit

    first.iloc[0, -1] = -999
    first[first.columns[-1]] = -888
    pd.testing.assert_frame_equal(supervised, supervised_before)
    pd.testing.assert_frame_equal(audit, audit_before)


def test_selector_constructor_validation_and_test_non_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_source = inspect.getsource(summaries)
    for forbidden_name in (
        "select_eda_populations",
        "select_development_rows",
        "select_test_rows",
        "select_model_features",
        "build_canonical_dataset",
        "build_analytical_dataset",
        "validation_drift",
        "test_drift",
    ):
        assert forbidden_name not in module_source

    supervised, audit = _hand_worked_inputs()

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("temporal summary attempted data selection")

    monkeypatch.setattr(run_eda, "select_eda_populations", fail_if_called)
    for owner, name in (
        (bd, "select_development_rows"),
        (bd, "select_test_rows"),
        (bd, "select_model_features"),
        (bd, "build_analytical_dataset"),
    ):
        monkeypatch.setattr(owner, name, fail_if_called)

    assert isinstance(
        summarize_temporal_coverage(supervised, audit),
        pd.DataFrame,
    )
    assert isinstance(
        summarize_temporal_monthly(supervised, audit),
        pd.DataFrame,
    )
