"""Tests for deterministic mature-training categorical summaries."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import run_eda, summaries
from src.analysis.run_eda import select_eda_populations
from src.analysis.summaries import summarize_categorical_features
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
EXPECTED_CATEGORICAL_FEATURES = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
EXPECTED_OUTPUT_COLUMNS = (
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
EXPECTED_WEEKDAYS = tuple(range(7))
EXPECTED_HOURS = tuple(range(24))
EXPECTED_MONTHS = tuple(range(1, 13))
EXPECTED_WILSON_Z = 1.959963984540054
EXPECTED_RARE_COUNT_THRESHOLD = 30
EXPECTED_RARE_SHARE_THRESHOLD = 0.01
EXPECTED_UNCERTAINTY_COUNT_THRESHOLD = 30
EXPECTED_UNCERTAINTY_POSITIVES_THRESHOLD = 5
EXPECTED_REAL_ROWS = 3_670
EXPECTED_REAL_POSITIVES = 432
EXPECTED_REAL_NEGATIVES = 3_238
EXPECTED_OUTPUT_DTYPES = (
    "str",
    "str",
    "bool",
    "int64",
    "float64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "bool",
    "bool",
)
EXPECTED_REAL_ANCHORS = (
    ("visit_type", "new_patient_examination", 534, 55, 479),
    ("visit_type", "emergency", 290, 44, 246),
    ("booking_channel", "other", 143, 14, 129),
    ("booking_channel", "referral", 459, 58, 401),
    ("scheduled_weekday", "2", 603, 55, 548),
    ("scheduled_weekday", "6", 597, 78, 519),
    ("scheduled_hour", "9", 310, 31, 279),
    ("scheduled_hour", "15", 408, 57, 351),
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


def _supervised_fixture(
    targets: tuple[int, ...] = (0, 1, 1, 0),
) -> pd.DataFrame:
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
                dtype="object",
            ),
            "booking_channel": pd.Series(
                [f"channel-{index % 2}" for index in range(rows)],
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
    return frame.loc[:, EXPECTED_SUPERVISED_COLUMNS]


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


def _result_row(
    result: pd.DataFrame,
    feature: str,
    level: str,
    *,
    is_missing: bool = False,
) -> pd.Series:
    matching = result.loc[
        result["feature"].eq(feature)
        & result["level"].eq(level)
        & result["is_missing"].eq(is_missing)
    ]
    assert len(matching) == 1
    return matching.iloc[0]


def _threshold_fixture(
    total: int,
    level_count: int,
    level_positives: int,
) -> pd.DataFrame:
    targets = (
        [1] * level_positives
        + [0] * (level_count - level_positives)
        + [0] * (total - level_count)
    )
    frame = _supervised_fixture(tuple(targets))
    frame["visit_type"] = pd.Series(
        ["candidate"] * level_count + ["other"] * (total - level_count),
        dtype="object",
    )
    return frame


def test_real_data_integration_contract_totals_and_literal_anchors(
    real_supervised: pd.DataFrame,
) -> None:
    result = summarize_categorical_features(real_supervised)

    assert tuple(real_supervised.columns) == EXPECTED_SUPERVISED_COLUMNS
    assert tuple(bd.FEATURE_COLUMNS) == EXPECTED_FEATURE_COLUMNS
    assert summaries.CATEGORICAL_FEATURE_COLUMNS == EXPECTED_CATEGORICAL_FEATURES
    assert tuple(result.columns) == EXPECTED_OUTPUT_COLUMNS
    assert tuple(result["feature"].drop_duplicates()) == EXPECTED_CATEGORICAL_FEATURES
    assert not result["is_missing"].any()

    for feature in ("visit_type", "booking_channel"):
        levels = result.loc[result["feature"].eq(feature), "level"].tolist()
        assert levels == sorted(levels)
    expected_domains = {
        "scheduled_weekday": [str(value) for value in EXPECTED_WEEKDAYS],
        "scheduled_hour": [str(value) for value in EXPECTED_HOURS],
        "scheduled_month": [str(value) for value in EXPECTED_MONTHS],
    }
    for feature, expected_levels in expected_domains.items():
        levels = result.loc[result["feature"].eq(feature), "level"].tolist()
        assert levels == expected_levels

    for feature in EXPECTED_CATEGORICAL_FEATURES:
        feature_rows = result.loc[result["feature"].eq(feature)]
        assert feature_rows["count"].sum() == EXPECTED_REAL_ROWS
        assert feature_rows["positives"].sum() == EXPECTED_REAL_POSITIVES
        assert feature_rows["negatives"].sum() == EXPECTED_REAL_NEGATIVES

    for feature, level, count, positives, negatives in EXPECTED_REAL_ANCHORS:
        row = _result_row(result, feature, level)
        assert row["count"] == count
        assert row["positives"] == positives
        assert row["negatives"] == negatives
        assert row["no_show_rate"] == positives / count

    populated = result.loc[result["count"].gt(0)]
    for row in populated.itertuples(index=False):
        assert row.count == row.positives + row.negatives
        assert row.no_show_rate == row.positives / row.count
        assert math.isfinite(row.wilson_lower)
        assert math.isfinite(row.wilson_upper)
        assert row.wilson_lower <= row.no_show_rate <= row.wilson_upper


def test_hand_worked_counts_rates_wilson_and_ordering() -> None:
    frame = _supervised_fixture((0, 1, 1, 0))
    frame["visit_type"] = pd.Series(
        ["zeta", "alpha", "alpha", "zeta"],
        dtype="object",
    )
    frame["booking_channel"] = pd.Series(
        ["web", "phone", "web", "phone"],
        dtype="object",
    )
    frame["scheduled_weekday"] = pd.Series(
        [2.0, 0.0, 2.0, 6.0],
        dtype="float64",
    )

    result = summarize_categorical_features(frame)

    assert tuple(result["feature"].drop_duplicates()) == EXPECTED_CATEGORICAL_FEATURES
    assert result.loc[result["feature"].eq("visit_type"), "level"].tolist() == [
        "alpha",
        "zeta",
    ]
    assert result.loc[
        result["feature"].eq("booking_channel"), "level"
    ].tolist() == ["phone", "web"]

    row = _result_row(result, "scheduled_weekday", "2")
    expected_lower, expected_upper = _expected_wilson(1, 2)
    assert row["count"] == 2
    assert row["share"] == 2 / 4
    assert row["positives"] == 1
    assert row["negatives"] == 1
    assert row["no_show_rate"] == 1 / 2
    assert math.isclose(
        row["wilson_lower"], expected_lower, rel_tol=0.0, abs_tol=1e-15
    )
    assert math.isclose(
        row["wilson_upper"], expected_upper, rel_tol=0.0, abs_tol=1e-15
    )
    assert not math.isclose(
        row["wilson_lower"],
        round(expected_lower, 6),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_missing_categories_are_counted_distinguished_and_ordered_last() -> None:
    frame = _supervised_fixture((0, 1, 1, 0))
    frame["visit_type"] = pd.Series(
        ["<MISSING>", None, "alpha", "alpha"],
        dtype="object",
    )
    frame["booking_channel"] = pd.Series(
        ["<MISSING>", None, "web", "phone"],
        dtype="object",
    )
    frame["scheduled_weekday"] = pd.Series(
        [0.0, np.nan, 2.0, 2.0],
        dtype="float64",
    )
    frame["scheduled_hour"] = pd.Series(
        [9.0, np.nan, 15.0, 15.0],
        dtype="float64",
    )
    frame["scheduled_month"] = pd.Series(
        [1.0, np.nan, 2.0, 2.0],
        dtype="float64",
    )

    result = summarize_categorical_features(frame)
    expected_lower, expected_upper = _expected_wilson(1, 1)
    for feature in EXPECTED_CATEGORICAL_FEATURES:
        feature_rows = result.loc[result["feature"].eq(feature)]
        missing_rows = feature_rows.loc[feature_rows["is_missing"]]
        assert len(missing_rows) == 1
        assert bool(feature_rows.iloc[-1]["is_missing"])
        missing = missing_rows.iloc[0]
        assert missing["level"] == "<MISSING>"
        assert missing["count"] == 1
        assert missing["positives"] == 1
        assert missing["negatives"] == 0
        assert missing["no_show_rate"] == 1.0
        assert missing["wilson_lower"] == pytest.approx(
            expected_lower,
            rel=0.0,
            abs=1e-15,
        )
        assert missing["wilson_upper"] == pytest.approx(
            expected_upper,
            rel=0.0,
            abs=1e-15,
        )
        assert feature_rows["share"].sum() == 1.0

    literal = _result_row(result, "visit_type", "<MISSING>")
    missing = _result_row(
        result,
        "visit_type",
        "<MISSING>",
        is_missing=True,
    )
    assert not bool(literal["is_missing"])
    assert bool(missing["is_missing"])


def test_complete_calendar_domains_and_zero_count_policy() -> None:
    frame = _supervised_fixture((0, 1, 1))
    result = summarize_categorical_features(frame)
    expected_domains = {
        "scheduled_weekday": EXPECTED_WEEKDAYS,
        "scheduled_hour": EXPECTED_HOURS,
        "scheduled_month": EXPECTED_MONTHS,
    }

    for feature, domain in expected_domains.items():
        feature_rows = result.loc[result["feature"].eq(feature)]
        assert feature_rows["level"].tolist() == [str(value) for value in domain]
        zero_rows = feature_rows.loc[feature_rows["count"].eq(0)]
        assert not zero_rows.empty
        assert zero_rows["share"].eq(0.0).all()
        assert zero_rows["positives"].eq(0).all()
        assert zero_rows["negatives"].eq(0).all()
        assert zero_rows["no_show_rate"].isna().all()
        assert zero_rows["wilson_lower"].isna().all()
        assert zero_rows["wilson_upper"].isna().all()
        assert zero_rows["is_rare"].all()
        assert zero_rows["has_high_uncertainty"].all()


def test_empty_input_contract_and_stable_dtypes() -> None:
    empty = _supervised_fixture().iloc[:0].copy(deep=True)
    result = summarize_categorical_features(empty)

    assert tuple(result.columns) == EXPECTED_OUTPUT_COLUMNS
    assert result.loc[result["feature"].eq("visit_type")].empty
    assert result.loc[result["feature"].eq("booking_channel")].empty
    assert result["feature"].value_counts().to_dict() == {
        "scheduled_hour": 24,
        "scheduled_month": 12,
        "scheduled_weekday": 7,
    }
    assert len(result) == 43
    assert result["count"].eq(0).all()
    assert result["positives"].eq(0).all()
    assert result["negatives"].eq(0).all()
    assert result["share"].isna().all()
    assert result["no_show_rate"].isna().all()
    assert result["wilson_lower"].isna().all()
    assert result["wilson_upper"].isna().all()
    assert result["is_rare"].all()
    assert result["has_high_uncertainty"].all()
    assert tuple(result.dtypes.astype(str)) == EXPECTED_OUTPUT_DTYPES


@pytest.mark.parametrize(
    ("count", "total", "expected_rare"),
    (
        (29, 100, True),
        (30, 100, False),
        (30, 3_001, True),
        (30, 3_000, False),
        (31, 100, False),
    ),
    ids=(
        "count-29",
        "count-30",
        "share-below-one-percent",
        "share-exactly-one-percent",
        "neither-rule",
    ),
)
def test_rare_level_strict_thresholds(
    count: int,
    total: int,
    expected_rare: bool,
) -> None:
    assert EXPECTED_RARE_COUNT_THRESHOLD == 30
    assert EXPECTED_RARE_SHARE_THRESHOLD == 0.01
    frame = _threshold_fixture(total, count, min(count, 5))
    row = _result_row(
        summarize_categorical_features(frame),
        "visit_type",
        "candidate",
    )

    assert row["count"] == count
    assert row["share"] == count / total
    assert bool(row["is_rare"]) is expected_rare


@pytest.mark.parametrize(
    ("count", "positives", "expected_uncertainty"),
    (
        (29, 5, True),
        (30, 5, False),
        (30, 4, True),
        (31, 5, False),
    ),
    ids=(
        "count-29",
        "count-30-and-five-positive",
        "four-positive",
        "neither-rule",
    ),
)
def test_high_uncertainty_strict_thresholds(
    count: int,
    positives: int,
    expected_uncertainty: bool,
) -> None:
    assert EXPECTED_UNCERTAINTY_COUNT_THRESHOLD == 30
    assert EXPECTED_UNCERTAINTY_POSITIVES_THRESHOLD == 5
    frame = _threshold_fixture(100, count, positives)
    row = _result_row(
        summarize_categorical_features(frame),
        "visit_type",
        "candidate",
    )

    assert row["count"] == count
    assert row["positives"] == positives
    assert bool(row["has_high_uncertainty"]) is expected_uncertainty


def test_wilson_zero_all_mixed_singleton_and_zero_count_edges() -> None:
    frame = _supervised_fixture((0, 0, 1, 1, 0, 1, 1))
    frame["visit_type"] = pd.Series(
        ["zero", "zero", "all", "all", "mixed", "mixed", "single"],
        dtype="object",
    )
    result = summarize_categorical_features(frame)

    for level, positives, count in (
        ("zero", 0, 2),
        ("all", 2, 2),
        ("mixed", 1, 2),
        ("single", 1, 1),
    ):
        row = _result_row(result, "visit_type", level)
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

    zero_count = _result_row(result, "scheduled_hour", "23")
    assert zero_count["count"] == 0
    assert math.isnan(zero_count["no_show_rate"])
    assert math.isnan(zero_count["wilson_lower"])
    assert math.isnan(zero_count["wilson_upper"])


def test_shared_input_contract_rejects_wrong_frames(
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
    )
    for invalid in invalid_frames:
        with pytest.raises(ValueError):
            summarize_categorical_features(invalid)


def test_shared_input_contract_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        summarize_categorical_features([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("column", "bad_value"),
    (
        ("visit_type", 3),
        ("visit_type", ["list-like"]),
        ("booking_channel", 7),
        ("scheduled_weekday", -1),
        ("scheduled_weekday", 7),
        ("scheduled_hour", -1),
        ("scheduled_hour", 24),
        ("scheduled_month", 0),
        ("scheduled_month", 13),
        ("scheduled_weekday", 1.5),
        ("scheduled_hour", False),
    ),
    ids=(
        "numeric-visit-type",
        "list-visit-type",
        "mixed-booking-channel",
        "weekday-below-domain",
        "weekday-above-domain",
        "hour-below-domain",
        "hour-above-domain",
        "month-below-domain",
        "month-above-domain",
        "fractional-calendar",
        "Boolean-calendar-value",
    ),
)
def test_categorical_role_rejects_invalid_values(
    column: str,
    bad_value: object,
) -> None:
    frame = _supervised_fixture()
    frame[column] = frame[column].astype("object")
    frame.at[frame.index[0], column] = bad_value

    with pytest.raises(ValueError, match=column):
        summarize_categorical_features(frame)


def test_categorical_role_rejects_boolean_calendar_dtype() -> None:
    frame = _supervised_fixture()
    frame["scheduled_month"] = pd.Series(
        [True, False, True, False],
        dtype="bool",
    )

    with pytest.raises(
        ValueError,
        match=r"scheduled_month.*numeric non-Boolean integers",
    ):
        summarize_categorical_features(frame)


def test_non_mutation_new_output_and_row_order_determinism() -> None:
    frame = _supervised_fixture((0, 1, 1, 0, 1, 0))
    frame["visit_type"] = pd.Series(
        ["zeta", "alpha", "middle", "alpha", "zeta", "middle"],
        dtype="object",
    )
    frame_before = frame.copy(deep=True)

    first = summarize_categorical_features(frame)
    repeated = summarize_categorical_features(frame)
    reversed_result = summarize_categorical_features(frame.iloc[::-1])
    shuffled_result = summarize_categorical_features(
        frame.sample(frac=1.0, random_state=8_521)
    )

    pd.testing.assert_frame_equal(frame, frame_before)
    pd.testing.assert_frame_equal(first, repeated)
    pd.testing.assert_frame_equal(first, reversed_result)
    pd.testing.assert_frame_equal(first, shuffled_result)
    assert first is not frame
    assert first.loc[first["feature"].eq("visit_type"), "level"].tolist() == [
        "alpha",
        "middle",
        "zeta",
    ]

    first.loc[first.index[0], "level"] = "changed-scalar"
    pd.testing.assert_frame_equal(frame, frame_before)
    first["level"] = "changed-column"
    pd.testing.assert_frame_equal(frame, frame_before)


def test_selector_dataset_constructor_and_leakage_non_use(
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
    ):
        assert forbidden_name not in module_source

    frame = _supervised_fixture()

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("categorical summary attempted data selection")

    monkeypatch.setattr(run_eda, "select_eda_populations", fail_if_called)
    selector_bindings: tuple[
        tuple[object, str],
        ...,
    ] = (
        (bd, "select_development_rows"),
        (bd, "select_test_rows"),
        (bd, "select_model_features"),
        (bd, "build_analytical_dataset"),
    )
    for owner, name in selector_bindings:
        monkeypatch.setattr(owner, name, fail_if_called)

    result = summarize_categorical_features(frame)
    assert isinstance(result, pd.DataFrame)
