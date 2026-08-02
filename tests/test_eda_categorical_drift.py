"""Tests for deterministic feature-only categorical drift summaries."""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import drift
from src.analysis.drift import (
    summarize_categorical_drift_features,
    summarize_categorical_drift_levels,
)
from src.analysis.run_eda import select_eda_populations
from src.analysis.summaries import CATEGORICAL_FEATURE_COLUMNS
from src.data import build_dataset as bd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
EXPECTED_FEATURES = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
EXPECTED_DRIFT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    *bd.FEATURE_COLUMNS,
)
EXPECTED_LEVEL_COLUMNS = (
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
EXPECTED_FEATURE_COLUMNS = (
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
EXPECTED_LEVEL_DTYPES = (
    "str",
    "str",
    "bool",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "bool",
    "bool",
)
EXPECTED_FEATURE_DTYPES = (
    "str",
    "int64",
    "int64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "int64",
    "int64",
    "int64",
    "int64",
    "float64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
)
EXPECTED_DOMAINS = {
    "scheduled_weekday": tuple(range(7)),
    "scheduled_hour": tuple(range(24)),
    "scheduled_month": tuple(range(1, 13)),
}
LEVEL_FLOAT_COLUMNS = EXPECTED_LEVEL_COLUMNS[5:10]
FEATURE_FLOAT_COLUMNS = (
    "train_missing_rate",
    "validation_missing_rate",
    "missing_rate_difference",
    "unseen_in_train_validation_share",
    "absent_in_validation_train_share",
    "total_variation_distance",
    "max_absolute_share_difference",
)
SUMMARIZERS = (
    summarize_categorical_drift_levels,
    summarize_categorical_drift_features,
)


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
    rows: int,
    *,
    population: str,
    id_start: int,
) -> pd.DataFrame:
    values = np.arange(rows, dtype="float64")
    start = "2025-02-01" if population == "train" else "2025-03-01"
    frame = pd.DataFrame(
        {
            "appointment_id": pd.Series(
                np.arange(id_start, id_start + rows),
                dtype="int64",
            ),
            "prediction_time": pd.Series(
                pd.date_range(start, periods=rows, freq="h"),
                dtype="datetime64[ns]",
            ),
            "planned_duration_min": pd.Series(
                30.0 + values,
                dtype="float64",
            ),
            "visit_type": pd.Series(
                ["alpha" if index % 2 == 0 else "beta" for index in range(rows)],
                dtype="object",
            ),
            "booking_channel": pd.Series(
                ["phone" if index % 2 == 0 else "web" for index in range(rows)],
                dtype="object",
            ),
            "booking_lead_time_hours": pd.Series(
                24.0 + values,
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(values % 7, dtype="float64"),
            "scheduled_hour": pd.Series(values % 24, dtype="float64"),
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


def _drift_pair(
    train_rows: int = 4,
    validation_rows: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _drift_fixture(train_rows, population="train", id_start=1)
    validation = _drift_fixture(
        validation_rows,
        population="validation",
        id_start=10_001,
    )
    return train, validation


def _assign_values(
    frame: pd.DataFrame,
    feature: str,
    values: list[object],
) -> pd.DataFrame:
    changed = frame.copy(deep=True)
    changed[feature] = pd.Series(values, index=changed.index, dtype="object")
    return changed


def _level_row(
    result: pd.DataFrame,
    feature: str,
    level: str,
    *,
    is_missing: bool = False,
) -> pd.Series:
    matches = result.loc[
        result["feature"].eq(feature)
        & result["level"].eq(level)
        & result["is_missing"].eq(is_missing)
    ]
    assert len(matches) == 1
    return matches.iloc[0]


def _feature_row(result: pd.DataFrame, feature: str) -> pd.Series:
    matches = result.loc[result["feature"].eq(feature)]
    assert len(matches) == 1
    return matches.iloc[0]


def _direct_count(series: pd.Series, value: object, is_missing: bool) -> int:
    mask = series.isna() if is_missing else series.notna() & series.eq(value)
    return int(mask.sum())


def test_public_signatures_output_contracts_and_dtypes() -> None:
    train, validation = _drift_pair()
    for summarizer in SUMMARIZERS:
        signature = inspect.signature(summarizer)
        assert tuple(signature.parameters) == (
            "train_drift",
            "validation_drift",
        )
        assert signature.return_annotation == "pd.DataFrame"

    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)
    assert CATEGORICAL_FEATURE_COLUMNS == EXPECTED_FEATURES
    assert tuple(levels.columns) == EXPECTED_LEVEL_COLUMNS
    assert tuple(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert tuple(map(str, levels.dtypes)) == EXPECTED_LEVEL_DTYPES
    assert tuple(map(str, features.dtypes)) == EXPECTED_FEATURE_DTYPES
    assert tuple(levels["feature"].drop_duplicates()) == EXPECTED_FEATURES
    assert tuple(features["feature"]) == EXPECTED_FEATURES


def test_real_data_reconciliation_and_independent_drift_calculations(
    real_drift_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, validation = real_drift_frames
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)

    assert len(train) == 3_670
    assert len(validation) == 1_541
    assert tuple(train.columns) == EXPECTED_DRIFT_COLUMNS
    assert tuple(validation.columns) == EXPECTED_DRIFT_COLUMNS
    assert "target" not in train.columns
    assert "target" not in validation.columns

    for feature in EXPECTED_FEATURES:
        feature_levels = levels.loc[levels["feature"].eq(feature)]
        feature_summary = _feature_row(features, feature)
        assert feature_levels["train_count"].sum() == len(train)
        assert feature_levels["validation_count"].sum() == len(validation)
        assert feature_levels["train_share"].sum() == pytest.approx(1.0)
        assert feature_levels["validation_share"].sum() == pytest.approx(1.0)

        train_values = set(train[feature].dropna().tolist())
        validation_values = set(validation[feature].dropna().tolist())
        if feature in EXPECTED_DOMAINS:
            universe = list(EXPECTED_DOMAINS[feature])
        else:
            universe = sorted(train_values | validation_values)

        direct_differences: list[float] = []
        for value in universe:
            level = _level_row(feature_levels, feature, str(value))
            train_count = _direct_count(train[feature], value, False)
            validation_count = _direct_count(validation[feature], value, False)
            assert level["train_count"] == train_count
            assert level["validation_count"] == validation_count
            direct_differences.append(
                validation_count / len(validation) - train_count / len(train)
            )
        missing = _level_row(
            feature_levels,
            feature,
            "<MISSING>",
            is_missing=True,
        )
        train_missing = int(train[feature].isna().sum())
        validation_missing = int(validation[feature].isna().sum())
        assert missing["train_count"] == train_missing
        assert missing["validation_count"] == validation_missing
        assert feature_summary["train_missing_count"] == train_missing
        assert feature_summary["validation_missing_count"] == validation_missing
        direct_differences.append(
            validation_missing / len(validation) - train_missing / len(train)
        )

        unseen = validation_values - train_values
        absent = train_values - validation_values
        direct_unseen_count = int(validation[feature].isin(unseen).sum())
        direct_absent_count = int(train[feature].isin(absent).sum())
        assert feature_summary["unseen_in_train_level_count"] == len(unseen)
        assert (
            feature_summary["unseen_in_train_validation_count"]
            == direct_unseen_count
        )
        assert feature_summary["absent_in_validation_level_count"] == len(absent)
        assert (
            feature_summary["absent_in_validation_train_count"]
            == direct_absent_count
        )
        direct_absolute = [abs(value) for value in direct_differences]
        assert feature_summary["total_variation_distance"] == pytest.approx(
            0.5 * math.fsum(direct_absolute),
            rel=0.0,
            abs=1e-15,
        )
        assert feature_summary["max_absolute_share_difference"] == pytest.approx(
            max(direct_absolute),
            rel=0.0,
            abs=1e-15,
        )

    weekday = levels.loc[levels["feature"].eq("scheduled_weekday")]
    assert weekday["level"].tolist() == [
        *[str(value) for value in range(7)],
        "<MISSING>",
    ]
    assert weekday["is_missing"].tolist() == [False] * 7 + [True]
    assert _level_row(weekday, "scheduled_weekday", "4")[
        ["train_count", "validation_count"]
    ].tolist() == [0, 0]


def test_identical_distributions_have_zero_total_variation() -> None:
    train, validation = _drift_pair(6, 6)
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)

    assert levels["share_difference"].eq(0.0).all()
    assert levels["absolute_share_difference"].eq(0.0).all()
    assert levels["contribution_to_total_variation"].eq(0.0).all()
    assert not levels["is_unseen_in_train"].any()
    assert not levels["is_absent_in_validation"].any()
    assert features["total_variation_distance"].eq(0.0).all()
    assert features["max_absolute_share_difference"].eq(0.0).all()


def test_fully_disjoint_string_distributions_have_unit_total_variation() -> None:
    train, validation = _drift_pair()
    train = _assign_values(train, "visit_type", ["train-only"] * 4)
    validation = _assign_values(
        validation,
        "visit_type",
        ["validation-only"] * 4,
    )
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)

    train_only = _level_row(levels, "visit_type", "train-only")
    validation_only = _level_row(levels, "visit_type", "validation-only")
    summary = _feature_row(features, "visit_type")
    assert train_only["share_difference"] == -1.0
    assert bool(train_only["is_absent_in_validation"])
    assert validation_only["share_difference"] == 1.0
    assert bool(validation_only["is_unseen_in_train"])
    assert summary["total_variation_distance"] == 1.0
    assert summary["max_absolute_share_difference"] == 1.0


def test_hand_worked_partial_drift() -> None:
    train, validation = _drift_pair()
    train = _assign_values(train, "visit_type", ["A", "A", "A", "B"])
    validation = _assign_values(
        validation,
        "visit_type",
        ["A", "B", "B", "C"],
    )
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)
    expected = {
        "A": (0.75, 0.25, -0.50),
        "B": (0.25, 0.50, 0.25),
        "C": (0.00, 0.25, 0.25),
    }
    for level, (train_share, validation_share, difference) in expected.items():
        row = _level_row(levels, "visit_type", level)
        assert row["train_share"] == train_share
        assert row["validation_share"] == validation_share
        assert row["share_difference"] == difference
        assert row["absolute_share_difference"] == abs(difference)
        assert row["contribution_to_total_variation"] == 0.5 * abs(difference)

    category_c = _level_row(levels, "visit_type", "C")
    summary = _feature_row(features, "visit_type")
    assert bool(category_c["is_unseen_in_train"])
    assert not levels.loc[
        levels["feature"].eq("visit_type"),
        "is_absent_in_validation",
    ].any()
    assert summary["total_variation_distance"] == 0.5
    assert summary["max_absolute_share_difference"] == 0.5


def test_missingness_is_distribution_mass_and_uses_all_rows() -> None:
    train, validation = _drift_pair()
    train = _assign_values(train, "visit_type", ["A", "A", None, None])
    validation = _assign_values(validation, "visit_type", ["A", "A", "A", None])
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)
    missing = _level_row(
        levels,
        "visit_type",
        "<MISSING>",
        is_missing=True,
    )
    summary = _feature_row(features, "visit_type")

    assert missing["train_count"] == 2
    assert missing["validation_count"] == 1
    assert missing["train_share"] == 0.5
    assert missing["validation_share"] == 0.25
    assert missing["share_difference"] == -0.25
    assert missing["contribution_to_total_variation"] == 0.125
    assert not bool(missing["is_unseen_in_train"])
    assert not bool(missing["is_absent_in_validation"])
    assert summary["train_missing_rate"] == 0.5
    assert summary["validation_missing_rate"] == 0.25
    assert summary["missing_rate_difference"] == -0.25
    assert summary["total_variation_distance"] == 0.25


def test_literal_missing_label_and_null_have_distinct_rows() -> None:
    train, validation = _drift_pair()
    train = _assign_values(
        train,
        "visit_type",
        ["<MISSING>", None, "A", "A"],
    )
    validation = _assign_values(
        validation,
        "visit_type",
        [None, "<MISSING>", "<MISSING>", "A"],
    )
    result = summarize_categorical_drift_levels(train, validation)
    literal = _level_row(result, "visit_type", "<MISSING>")
    missing = _level_row(
        result,
        "visit_type",
        "<MISSING>",
        is_missing=True,
    )
    assert literal[["train_count", "validation_count"]].tolist() == [1, 2]
    assert missing[["train_count", "validation_count"]].tolist() == [1, 1]
    assert not bool(literal["is_missing"])
    assert bool(missing["is_missing"])


def test_complete_fixed_domains_include_zero_levels_and_missing_last() -> None:
    train, validation = _drift_pair(2, 2)
    result = summarize_categorical_drift_levels(train, validation)
    for feature, domain in EXPECTED_DOMAINS.items():
        rows = result.loc[result["feature"].eq(feature)]
        assert rows["level"].tolist() == [
            *[str(value) for value in domain],
            "<MISSING>",
        ]
        assert rows["is_missing"].tolist() == [False] * len(domain) + [True]
        nonmissing = rows.loc[~rows["is_missing"]]
        assert not nonmissing.loc[
            nonmissing["train_count"].eq(0)
            & nonmissing["validation_count"].eq(0)
        ].empty


def test_open_category_union_is_lexical_and_missing_is_last() -> None:
    train, validation = _drift_pair()
    train = _assign_values(train, "visit_type", ["zeta", "beta", "zeta", None])
    validation = _assign_values(
        validation,
        "visit_type",
        ["gamma", "alpha", None, "beta"],
    )
    result = summarize_categorical_drift_levels(train, validation)
    rows = result.loc[result["feature"].eq("visit_type")]
    assert rows["level"].tolist() == [
        "alpha",
        "beta",
        "gamma",
        "zeta",
        "<MISSING>",
    ]
    assert rows["is_missing"].tolist() == [False] * 4 + [True]


def test_multiple_unseen_and_absent_categories_aggregate_by_rows() -> None:
    train, validation = _drift_pair(6, 6)
    train = _assign_values(
        train,
        "visit_type",
        ["shared", "train-a", "train-a", "train-b", None, None],
    )
    validation = _assign_values(
        validation,
        "visit_type",
        ["shared", "validation-a", "validation-b", "validation-b", None, None],
    )
    levels = summarize_categorical_drift_levels(train, validation)
    summary = _feature_row(
        summarize_categorical_drift_features(train, validation),
        "visit_type",
    )
    rows = levels.loc[levels["feature"].eq("visit_type")]
    assert set(rows.loc[rows["is_unseen_in_train"], "level"]) == {
        "validation-a",
        "validation-b",
    }
    assert set(rows.loc[rows["is_absent_in_validation"], "level"]) == {
        "train-a",
        "train-b",
    }
    assert summary["unseen_in_train_level_count"] == 2
    assert summary["unseen_in_train_validation_count"] == 3
    assert summary["unseen_in_train_validation_share"] == 0.5
    assert summary["absent_in_validation_level_count"] == 2
    assert summary["absent_in_validation_train_count"] == 3
    assert summary["absent_in_validation_train_share"] == 0.5
    missing = rows.loc[rows["is_missing"]].iloc[0]
    assert not bool(missing["is_unseen_in_train"])
    assert not bool(missing["is_absent_in_validation"])


@pytest.mark.parametrize(
    ("train_rows", "validation_rows"),
    ((0, 0), (0, 4), (4, 0)),
)
def test_empty_population_policy(train_rows: int, validation_rows: int) -> None:
    train, validation = _drift_pair(train_rows, validation_rows)
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)

    if not train_rows:
        assert levels["train_count"].eq(0).all()
        assert levels["train_share"].isna().all()
        assert features["train_missing_rate"].isna().all()
    if not validation_rows:
        assert levels["validation_count"].eq(0).all()
        assert levels["validation_share"].isna().all()
        assert features["validation_missing_rate"].isna().all()
    assert levels["share_difference"].isna().all()
    assert levels["absolute_share_difference"].isna().all()
    assert levels["contribution_to_total_variation"].isna().all()
    assert features["missing_rate_difference"].isna().all()
    assert features["total_variation_distance"].isna().all()
    assert features["max_absolute_share_difference"].isna().all()

    if not train_rows and validation_rows:
        positive_nonmissing = levels.loc[
            ~levels["is_missing"] & levels["validation_count"].gt(0)
        ]
        assert positive_nonmissing["is_unseen_in_train"].all()
        assert not levels["is_absent_in_validation"].any()
        for feature in EXPECTED_FEATURES:
            expected_count = int(validation[feature].notna().sum())
            summary = _feature_row(features, feature)
            assert summary["unseen_in_train_validation_count"] == expected_count
            assert summary["unseen_in_train_validation_share"] == (
                expected_count / validation_rows
            )
    if train_rows and not validation_rows:
        positive_nonmissing = levels.loc[
            ~levels["is_missing"] & levels["train_count"].gt(0)
        ]
        assert positive_nonmissing["is_absent_in_validation"].all()
        assert not levels["is_unseen_in_train"].any()

    if not train_rows and not validation_rows:
        for feature in ("visit_type", "booking_channel"):
            rows = levels.loc[levels["feature"].eq(feature)]
            assert rows[["level", "is_missing"]].values.tolist() == [
                ["<MISSING>", True]
            ]


def test_numeric_weekday_boundaries_are_accepted_without_mapping() -> None:
    train, validation = _drift_pair()
    train = _assign_values(train, "scheduled_weekday", [0, 6, 0.0, 6.0])
    validation = _assign_values(
        validation,
        "scheduled_weekday",
        [6, 0, np.int64(6), np.float64(0.0)],
    )
    result = summarize_categorical_drift_levels(train, validation)
    rows = result.loc[result["feature"].eq("scheduled_weekday")]
    assert rows["level"].tolist() == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "<MISSING>",
    ]
    assert _level_row(rows, "scheduled_weekday", "0")[
        ["train_count", "validation_count"]
    ].tolist() == [2, 2]
    assert _level_row(rows, "scheduled_weekday", "6")[
        ["train_count", "validation_count"]
    ].tolist() == [2, 2]


@pytest.mark.parametrize(
    "value",
    (-1, 7, 1.5, True, False, "Monday", "0", "Mon", "شنبه"),
)
@pytest.mark.parametrize("population", ("train", "validation"))
@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_invalid_numeric_weekday_values_are_rejected(
    value: object,
    population: str,
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    train, validation = _drift_pair()
    if population == "train":
        train = _assign_values(train, "scheduled_weekday", [value] * 4)
    else:
        validation = _assign_values(
            validation,
            "scheduled_weekday",
            [value] * 4,
        )
    with pytest.raises(ValueError, match=rf"{population}_drift.scheduled_weekday"):
        summarizer(train, validation)


@pytest.mark.parametrize(
    ("feature", "value"),
    (
        ("visit_type", 1),
        ("visit_type", True),
        ("visit_type", b"exam"),
        ("visit_type", ["exam"]),
        ("visit_type", {"name": "exam"}),
        ("visit_type", object()),
        ("booking_channel", 1),
        ("booking_channel", False),
        ("scheduled_hour", -1),
        ("scheduled_hour", 24),
        ("scheduled_hour", 1.5),
        ("scheduled_hour", True),
        ("scheduled_month", 0),
        ("scheduled_month", 13),
        ("scheduled_month", 1.5),
        ("scheduled_month", False),
    ),
)
@pytest.mark.parametrize("population", ("train", "validation"))
@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_remaining_invalid_categorical_roles_are_rejected(
    feature: str,
    value: object,
    population: str,
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    train, validation = _drift_pair()
    if population == "train":
        train = _assign_values(train, feature, [value] * 4)
    else:
        validation = _assign_values(validation, feature, [value] * 4)
    with pytest.raises(ValueError, match=rf"{population}_drift.{feature}"):
        summarizer(train, validation)


@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_non_dataframe_inputs_are_rejected_for_each_population(
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    train, validation = _drift_pair()
    with pytest.raises(TypeError, match="train_drift"):
        summarizer([1, 2], validation)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validation_drift"):
        summarizer(train, [1, 2])  # type: ignore[arg-type]


@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_nonapproved_population_projections_are_rejected(
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    canonical_dataset: pd.DataFrame,
    eda_populations: dict[str, pd.DataFrame],
) -> None:
    train = eda_populations["train_drift"]
    validation = eda_populations["validation_drift"]
    invalid_frames = (
        canonical_dataset,
        eda_populations["supervised_train"],
        eda_populations["maturity_audit"],
        train.assign(target=0),
    )
    for invalid in invalid_frames:
        with pytest.raises(ValueError, match="approved projection"):
            summarizer(invalid, validation)


def _invalid_shared_contract_pair(
    case: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, validation = _drift_pair()
    if case == "missing_column":
        train = train.drop(columns=["visit_type"])
    elif case == "extra_column":
        validation = validation.assign(extra=0)
    elif case == "reordered_columns":
        columns = list(train.columns)
        columns[2], columns[3] = columns[3], columns[2]
        train = train.loc[:, columns]
    elif case == "duplicate_id":
        train.loc[train.index[1], "appointment_id"] = train.iloc[0][
            "appointment_id"
        ]
    elif case == "null_id":
        validation.loc[validation.index[0], "appointment_id"] = np.nan
    elif case == "timestamp_dtype":
        train["prediction_time"] = train["prediction_time"].astype("str")
    elif case == "timezone_timestamp":
        validation["prediction_time"] = validation["prediction_time"].dt.tz_localize(
            "UTC"
        )
    elif case == "late_train":
        train.loc[train.index[0], "prediction_time"] = bd.VALIDATION_START
    elif case == "early_validation":
        validation.loc[validation.index[0], "prediction_time"] = (
            bd.VALIDATION_START - pd.Timedelta(seconds=1)
        )
    elif case == "late_validation":
        validation.loc[validation.index[0], "prediction_time"] = bd.TEST_START
    elif case == "overlap":
        validation.loc[validation.index[0], "appointment_id"] = train.iloc[0][
            "appointment_id"
        ]
    elif case == "numeric_object":
        train["planned_duration_min"] = train["planned_duration_min"].astype(
            "object"
        )
    elif case == "numeric_boolean":
        validation["planned_duration_min"] = True
    elif case == "numeric_complex":
        train["planned_duration_min"] = 1.0 + 2.0j
    elif case == "numeric_infinite":
        validation.loc[validation.index[0], "planned_duration_min"] = np.inf
    else:
        raise AssertionError(f"unknown shared-contract case: {case}")
    return train, validation


@pytest.mark.parametrize(
    "case",
    (
        "missing_column",
        "extra_column",
        "reordered_columns",
        "duplicate_id",
        "null_id",
        "timestamp_dtype",
        "timezone_timestamp",
        "late_train",
        "early_validation",
        "late_validation",
        "overlap",
        "numeric_object",
        "numeric_boolean",
        "numeric_complex",
        "numeric_infinite",
    ),
)
@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_shared_drift_contract_regression(
    case: str,
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    train, validation = _invalid_shared_contract_pair(case)
    with pytest.raises(ValueError):
        summarizer(train, validation)


@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_inputs_are_not_mutated_and_results_are_deterministic(
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    train, validation = _drift_pair(8, 8)
    train = _assign_values(
        train,
        "visit_type",
        ["zeta", "alpha", "beta", None, "zeta", "alpha", "beta", None],
    )
    validation = _assign_values(
        validation,
        "visit_type",
        ["gamma", "beta", None, "alpha", "gamma", "beta", None, "alpha"],
    )
    train.index = [7, 3, 9, 1, 8, 2, 6, 4]
    validation.index = [7, 3, 9, 1, 8, 2, 6, 4]
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)

    first = summarizer(train, validation)
    repeated = summarizer(train, validation)
    reversed_train = summarizer(train.iloc[::-1], validation)
    reversed_validation = summarizer(train, validation.iloc[::-1])
    shuffled = summarizer(
        train.sample(frac=1.0, random_state=17),
        validation.sample(frac=1.0, random_state=23),
    )

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)
    pd.testing.assert_frame_equal(first, repeated)
    pd.testing.assert_frame_equal(first, reversed_train)
    pd.testing.assert_frame_equal(first, reversed_validation)
    pd.testing.assert_frame_equal(first, shuffled)
    assert first is not repeated


@pytest.mark.parametrize("summarizer", SUMMARIZERS)
def test_output_mutation_cannot_affect_inputs_or_repeated_results(
    summarizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> None:
    train, validation = _drift_pair()
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)
    expected = summarizer(train, validation)
    changed = summarizer(train, validation)

    changed.iloc[0, 0] = "changed"
    changed[changed.columns[-1]] = False

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)
    pd.testing.assert_frame_equal(summarizer(train, validation), expected)
    assert not changed.equals(expected)


def test_all_defined_floating_outputs_are_finite_and_tvd_is_bounded() -> None:
    train, validation = _drift_pair(11, 7)
    levels = summarize_categorical_drift_levels(train, validation)
    features = summarize_categorical_drift_features(train, validation)
    assert np.isfinite(levels.loc[:, LEVEL_FLOAT_COLUMNS].to_numpy()).all()
    assert np.isfinite(features.loc[:, FEATURE_FLOAT_COLUMNS].to_numpy()).all()
    assert features["total_variation_distance"].between(0.0, 1.0).all()


def test_production_module_has_no_leakage_or_selector_access() -> None:
    source = inspect.getsource(drift)
    forbidden_patterns = (
        r"\btarget\b",
        r"\bselect_eda_populations\b",
        r"\bselect_development_rows\b",
        r"\bselect_test_rows\b",
        r"\bselect_model_features\b",
        r"\bbuild_canonical_dataset\b",
        r"validation[_ -]?label",
        r"test[_ -]?population",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, source, flags=re.IGNORECASE) is None


def test_level_representation_matches_existing_categorical_eda_contract() -> None:
    train, validation = _drift_pair()
    levels = summarize_categorical_drift_levels(train, validation)
    for feature, domain in EXPECTED_DOMAINS.items():
        rows = levels.loc[levels["feature"].eq(feature) & ~levels["is_missing"]]
        assert rows["level"].tolist() == [str(value) for value in domain]
    assert CATEGORICAL_FEATURE_COLUMNS == EXPECTED_FEATURES
