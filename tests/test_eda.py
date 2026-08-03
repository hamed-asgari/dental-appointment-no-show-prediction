"""Contract tests for leakage-safe exploratory-analysis populations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import run_eda as eda
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
EXPECTED_KEYS = (
    "supervised_train",
    "train_drift",
    "validation_drift",
    "maturity_audit",
)
EXPECTED_SUPERVISED_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "target",
    *EXPECTED_FEATURE_COLUMNS,
)
EXPECTED_DRIFT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    *EXPECTED_FEATURE_COLUMNS,
)
EXPECTED_MATURITY_AUDIT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
    "development_fit_eligible",
)
EXPECTED_SPLITS = {"train", "validation", "test"}
EXPECTED_VALIDATION_START = pd.Timestamp("2025-03-01 00:00:00")
EXPECTED_TEST_START = pd.Timestamp("2025-08-01 00:00:00")
EXPECTED_OUTPUT_DTYPES = {
    "appointment_id": "int64",
    "prediction_time": "datetime64[ns]",
    "target": "int8",
    "planned_duration_min": "int16",
    "visit_type": "string",
    "booking_channel": "string",
    "booking_lead_time_hours": "float64",
    "scheduled_weekday": "int8",
    "scheduled_hour": "int8",
    "scheduled_month": "int8",
    "approximate_age_at_prediction": "int16",
    "patient_registration_tenure_days": "int32",
    "dentist_tenure_days": "int32",
    "split": "string",
    "development_fit_eligible": "bool",
}


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(tables)


def _assert_populations_equal(
    left: dict[str, pd.DataFrame],
    right: dict[str, pd.DataFrame],
) -> None:
    assert tuple(left) == EXPECTED_KEYS
    assert tuple(right) == EXPECTED_KEYS
    for population_name in EXPECTED_KEYS:
        pd.testing.assert_frame_equal(left[population_name], right[population_name])


def _limited_development_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    development = bd.select_development_rows(canonical)
    mature_train = development["split"].eq("train") & development[
        "development_fit_eligible"
    ]
    train_row = development.loc[mature_train].iloc[[0]]
    validation_row = development.loc[development["split"].eq("validation")].iloc[
        [0]
    ]
    return pd.concat((train_row, validation_row), axis=0).copy(deep=True)


def test_exact_population_selection_and_integration_counts(
    canonical_dataset: pd.DataFrame,
) -> None:
    populations = select_eda_populations(canonical_dataset)

    assert tuple(bd.FEATURE_COLUMNS) == EXPECTED_FEATURE_COLUMNS
    assert bd.ALLOWED_SPLITS == EXPECTED_SPLITS
    assert bd.VALIDATION_START == EXPECTED_VALIDATION_START
    assert bd.TEST_START == EXPECTED_TEST_START
    assert tuple(populations) == EXPECTED_KEYS
    assert tuple(populations["supervised_train"].columns) == (
        EXPECTED_SUPERVISED_COLUMNS
    )
    assert tuple(populations["train_drift"].columns) == EXPECTED_DRIFT_COLUMNS
    assert tuple(populations["validation_drift"].columns) == EXPECTED_DRIFT_COLUMNS
    assert tuple(populations["maturity_audit"].columns) == (
        EXPECTED_MATURITY_AUDIT_COLUMNS
    )

    supervised = populations["supervised_train"]
    assert len(supervised) == 3_670
    assert int(supervised["target"].sum()) == 432
    assert int(supervised["target"].eq(0).sum()) == 3_238
    assert len(populations["train_drift"]) == len(supervised)

    maturity_audit = populations["maturity_audit"]
    assert len(maturity_audit) == 3_682
    assert int(maturity_audit["development_fit_eligible"].sum()) == 3_670
    assert int((~maturity_audit["development_fit_eligible"]).sum()) == 12
    assert maturity_audit["split"].eq("train").all()

    validation_drift = populations["validation_drift"]
    assert len(validation_drift) == 1_541
    expected_validation_ids = canonical_dataset.loc[
        canonical_dataset["split"].eq("validation"), "appointment_id"
    ].tolist()
    assert validation_drift["appointment_id"].tolist() == expected_validation_ids

    test_ids = set(
        canonical_dataset.loc[
            canonical_dataset["split"].eq("test"), "appointment_id"
        ]
    )
    for frame in populations.values():
        assert test_ids.isdisjoint(frame["appointment_id"])


def test_immature_train_target_poisoning_is_invariant(
    canonical_dataset: pd.DataFrame,
) -> None:
    poisoned = canonical_dataset.copy(deep=True)
    immature_train = poisoned["split"].eq("train") & ~poisoned[
        "development_fit_eligible"
    ]
    poisoned.loc[immature_train, "target"] = (
        1 - poisoned.loc[immature_train, "target"]
    ).astype("int8")

    _assert_populations_equal(
        select_eda_populations(canonical_dataset),
        select_eda_populations(poisoned),
    )


def test_validation_target_poisoning_is_invariant(
    canonical_dataset: pd.DataFrame,
) -> None:
    poisoned = canonical_dataset.copy(deep=True)
    validation = poisoned["split"].eq("validation")
    poisoned.loc[validation, "target"] = (
        1 - poisoned.loc[validation, "target"]
    ).astype("int8")

    _assert_populations_equal(
        select_eda_populations(canonical_dataset),
        select_eda_populations(poisoned),
    )


def test_test_content_poisoning_is_invariant(
    canonical_dataset: pd.DataFrame,
) -> None:
    poisoned = canonical_dataset.copy(deep=True)
    test_rows = poisoned["split"].eq("test")
    poisoned.loc[test_rows, "target"] = (
        1 - poisoned.loc[test_rows, "target"]
    ).astype("int8")
    poisoned.loc[test_rows, "booking_lead_time_hours"] = (
        poisoned.loc[test_rows, "booking_lead_time_hours"] + 1.0
    )

    _assert_populations_equal(
        select_eda_populations(canonical_dataset),
        select_eda_populations(poisoned),
    )


def test_test_selector_is_never_called(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("select_test_rows must not be called by EDA")

    monkeypatch.setattr(eda.bd, "select_test_rows", fail_if_called)
    populations = select_eda_populations(canonical_dataset)
    assert tuple(populations) == EXPECTED_KEYS


def test_full_dataset_feature_selector_is_never_called(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("select_model_features must not be called by EDA")

    monkeypatch.setattr(eda.bd, "select_model_features", fail_if_called)
    populations = select_eda_populations(canonical_dataset)
    assert tuple(populations) == EXPECTED_KEYS


def test_development_selector_is_called_exactly_once(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_selector = eda.bd.select_development_rows
    call_count = 0

    def counting_selector(canonical: pd.DataFrame) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return original_selector(canonical)

    monkeypatch.setattr(eda.bd, "select_development_rows", counting_selector)
    populations = select_eda_populations(canonical_dataset)

    assert tuple(populations) == EXPECTED_KEYS
    assert call_count == 1


def test_all_filters_use_the_returned_development_frame(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_selector = eda.bd.select_development_rows
    limited_development = original_selector(canonical_dataset).iloc[::2].copy(
        deep=True
    )

    def return_limited_development(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return limited_development.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_limited_development,
    )
    populations = select_eda_populations(canonical_dataset)

    mature_train = limited_development["split"].eq("train") & limited_development[
        "development_fit_eligible"
    ]
    validation = limited_development["split"].eq("validation")
    nominal_train = limited_development["split"].eq("train")
    assert populations["supervised_train"]["appointment_id"].tolist() == (
        limited_development.loc[mature_train, "appointment_id"].tolist()
    )
    assert populations["train_drift"]["appointment_id"].tolist() == (
        limited_development.loc[mature_train, "appointment_id"].tolist()
    )
    assert populations["validation_drift"]["appointment_id"].tolist() == (
        limited_development.loc[validation, "appointment_id"].tolist()
    )
    assert populations["maturity_audit"]["appointment_id"].tolist() == (
        limited_development.loc[nominal_train, "appointment_id"].tolist()
    )


def test_masks_use_maturity_value_from_returned_development_frame(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = eda.bd.select_development_rows(canonical_dataset)
    selected_index = development.index[
        development["split"].eq("train")
        & development["development_fit_eligible"]
    ][0]
    selected_appointment_id = int(development.loc[selected_index, "appointment_id"])
    development.loc[selected_index, "development_fit_eligible"] = False

    def return_modified_development(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return development.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_modified_development,
    )
    populations = select_eda_populations(canonical_dataset)

    assert selected_appointment_id in set(
        populations["maturity_audit"]["appointment_id"]
    )
    assert selected_appointment_id not in set(
        populations["supervised_train"]["appointment_id"]
    )
    assert selected_appointment_id not in set(
        populations["train_drift"]["appointment_id"]
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("test_period_as_validation", "test-period timestamp"),
        ("literal_test_split", "contains test rows"),
        ("duplicate_appointment", "appointment_id values must be unique"),
        ("null_required_feature", "contains null values"),
        ("missing_required_column", "missing required columns"),
        ("nonbinary_supervised_target", "target values must be binary"),
        ("validation_development_eligible", "may be true only for train"),
        ("train_relabelled_validation", "temporal split inconsistency"),
        ("nonboolean_maturity", "development_fit_eligible dtype must be bool"),
    ],
)
def test_malicious_development_returns_are_rejected(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
) -> None:
    development = eda.bd.select_development_rows(canonical_dataset)
    if defect in {"test_period_as_validation", "literal_test_split"}:
        test_index = canonical_dataset.index[canonical_dataset["split"].eq("test")][
            0
        ]
        injected = canonical_dataset.loc[[test_index]].copy(deep=True)
        if defect == "test_period_as_validation":
            injected.loc[test_index, "split"] = "validation"
        development = pd.concat((development, injected), axis=0)
    elif defect == "duplicate_appointment":
        duplicate = development.iloc[[0]].copy(deep=True)
        development = pd.concat((development, duplicate), axis=0)
    elif defect == "null_required_feature":
        development.loc[development.index[0], "visit_type"] = pd.NA
    elif defect == "missing_required_column":
        development = development.drop(columns="visit_type")
    elif defect == "nonbinary_supervised_target":
        mature = development["split"].eq("train") & development[
            "development_fit_eligible"
        ]
        development.loc[development.index[mature][0], "target"] = 2
    elif defect == "validation_development_eligible":
        index = development.index[development["split"].eq("validation")][0]
        development.loc[index, "development_fit_eligible"] = True
    elif defect == "train_relabelled_validation":
        index = development.index[development["split"].eq("train")][0]
        development.loc[index, "split"] = "validation"
    else:
        development["development_fit_eligible"] = development[
            "development_fit_eligible"
        ].astype("int8")

    def return_malicious_development(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return development.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_malicious_development,
    )
    with pytest.raises(ValueError, match=message):
        select_eda_populations(canonical_dataset)


def test_extra_prohibited_field_from_development_selector_is_not_projected(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = eda.bd.select_development_rows(canonical_dataset)
    development["status"] = "malicious-extra-field"

    def return_extra_column(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return development.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_extra_column,
    )
    populations = select_eda_populations(canonical_dataset)

    assert tuple(populations) == EXPECTED_KEYS
    assert all("status" not in frame.columns for frame in populations.values())


def test_invalid_predictor_in_limited_development_return_is_rejected(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited = _limited_development_frame(canonical_dataset)
    limited.loc[limited.index[0], "scheduled_weekday"] = 7
    assert str(limited["scheduled_weekday"].dtype) == "int8"

    def return_invalid_limited_frame(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return limited.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_invalid_limited_frame,
    )
    with pytest.raises(
        ValueError,
        match="scheduled_weekday contains values outside",
    ):
        select_eda_populations(canonical_dataset)


def test_empty_typed_development_return_produces_four_empty_populations(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_development = canonical_dataset.loc[canonical_dataset.index[:0]].copy(
        deep=True
    )

    def return_empty_development(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return empty_development.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_empty_development,
    )
    populations = select_eda_populations(canonical_dataset)
    expected_columns = {
        "supervised_train": EXPECTED_SUPERVISED_COLUMNS,
        "train_drift": EXPECTED_DRIFT_COLUMNS,
        "validation_drift": EXPECTED_DRIFT_COLUMNS,
        "maturity_audit": EXPECTED_MATURITY_AUDIT_COLUMNS,
    }

    assert tuple(populations) == EXPECTED_KEYS
    for population_name, columns in expected_columns.items():
        frame = populations[population_name]
        assert frame.empty
        assert tuple(frame.columns) == columns
        assert {column: str(frame[column].dtype) for column in columns} == {
            column: EXPECTED_OUTPUT_DTYPES[column] for column in columns
        }


def test_nonbinary_validation_target_in_development_return_is_not_observed(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited = _limited_development_frame(canonical_dataset)
    validation = limited["split"].eq("validation")
    limited.loc[validation, "target"] = 2
    assert str(limited["target"].dtype) == "int8"
    expected_validation = limited.loc[
        validation,
        list(EXPECTED_DRIFT_COLUMNS),
    ].copy(deep=True)
    mature_train = limited["split"].eq("train") & limited[
        "development_fit_eligible"
    ]
    expected_supervised = limited.loc[
        mature_train,
        list(EXPECTED_SUPERVISED_COLUMNS),
    ].copy(deep=True)

    def return_poisoned_development(canonical: pd.DataFrame) -> pd.DataFrame:
        assert canonical is canonical_dataset
        return limited.copy(deep=True)

    monkeypatch.setattr(
        eda.bd,
        "select_development_rows",
        return_poisoned_development,
    )
    populations = select_eda_populations(canonical_dataset)

    assert "target" not in populations["validation_drift"].columns
    pd.testing.assert_frame_equal(
        populations["validation_drift"],
        expected_validation,
    )
    pd.testing.assert_frame_equal(
        populations["supervised_train"],
        expected_supervised,
    )


def test_returned_columns_enforce_role_boundaries(
    canonical_dataset: pd.DataFrame,
) -> None:
    populations = select_eda_populations(canonical_dataset)
    identifiers = {"appointment_id", "patient_id", "dentist_id"}

    for population_name, frame in populations.items():
        columns = set(frame.columns)
        assert columns & identifiers == {"appointment_id"}
        assert "pretest_fit_eligible" not in columns
        assert ("target" in columns) == (population_name == "supervised_train")
        assert ("split" in columns) == (population_name == "maturity_audit")
        assert ("development_fit_eligible" in columns) == (
            population_name == "maturity_audit"
        )
        if population_name == "maturity_audit":
            assert columns.isdisjoint(EXPECTED_FEATURE_COLUMNS)
        else:
            assert tuple(
                column for column in frame.columns if column in EXPECTED_FEATURE_COLUMNS
            ) == EXPECTED_FEATURE_COLUMNS


def test_returns_defensive_independent_copies_without_mutating_input(
    canonical_dataset: pd.DataFrame,
) -> None:
    canonical_before = canonical_dataset.copy(deep=True)
    for population_name in EXPECTED_KEYS:
        populations = select_eda_populations(canonical_dataset)
        assert len({id(frame) for frame in populations.values()}) == 4
        other_before = {
            name: frame.copy(deep=True)
            for name, frame in populations.items()
            if name != population_name
        }

        selected = populations[population_name]
        selected.iloc[0, 0] = -1
        pd.testing.assert_frame_equal(canonical_dataset, canonical_before)
        for other_name, expected in other_before.items():
            pd.testing.assert_frame_equal(populations[other_name], expected)

        selected[selected.columns[0]] = -2
        pd.testing.assert_frame_equal(canonical_dataset, canonical_before)
        for other_name, expected in other_before.items():
            pd.testing.assert_frame_equal(populations[other_name], expected)


def test_preserves_validated_input_order(canonical_dataset: pd.DataFrame) -> None:
    reordered = canonical_dataset.iloc[::-1].copy(deep=True)
    populations = select_eda_populations(reordered)

    mature_train = reordered["split"].eq("train") & reordered[
        "development_fit_eligible"
    ]
    validation = reordered["split"].eq("validation")
    nominal_train = reordered["split"].eq("train")
    assert populations["supervised_train"]["appointment_id"].tolist() == (
        reordered.loc[mature_train, "appointment_id"].tolist()
    )
    assert populations["supervised_train"].index.tolist() == (
        reordered.index[mature_train].tolist()
    )
    assert populations["train_drift"]["appointment_id"].tolist() == (
        reordered.loc[mature_train, "appointment_id"].tolist()
    )
    assert populations["train_drift"].index.tolist() == (
        reordered.index[mature_train].tolist()
    )
    assert populations["validation_drift"]["appointment_id"].tolist() == (
        reordered.loc[validation, "appointment_id"].tolist()
    )
    assert populations["validation_drift"].index.tolist() == (
        reordered.index[validation].tolist()
    )
    assert populations["maturity_audit"]["appointment_id"].tolist() == (
        reordered.loc[nominal_train, "appointment_id"].tolist()
    )
    assert populations["maturity_audit"].index.tolist() == (
        reordered.index[nominal_train].tolist()
    )


def test_semantic_validation_precedes_development_selection(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrupted = canonical_dataset.copy(deep=True)
    corrupted.loc[corrupted.index[1], "appointment_id"] = corrupted.loc[
        corrupted.index[0], "appointment_id"
    ]
    selector_called = False

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal selector_called
        selector_called = True
        raise AssertionError("development selector reached before semantic validation")

    monkeypatch.setattr(eda.bd, "select_development_rows", fail_if_called)
    with pytest.raises(ValueError, match="appointment_id values must be unique"):
        select_eda_populations(corrupted)

    assert not selector_called


@pytest.mark.parametrize(
    ("split_name", "timestamp", "accepted"),
    [
        pytest.param(
            "train",
            pd.Timestamp("2025-03-01 00:00:00"),
            False,
            id="train-at-validation-start-rejected",
        ),
        pytest.param(
            "validation",
            pd.Timestamp("2025-03-01 00:00:00"),
            True,
            id="validation-at-validation-start-accepted",
        ),
        pytest.param(
            "validation",
            pd.Timestamp("2025-08-01 00:00:00"),
            False,
            id="validation-at-test-start-rejected",
        ),
        pytest.param(
            "test",
            pd.Timestamp("2025-08-01 00:00:00"),
            True,
            id="test-at-test-start-accepted",
        ),
        pytest.param(
            "test",
            pd.Timestamp("2025-07-31 23:59:59.999999999"),
            False,
            id="test-immediately-before-test-start-rejected",
        ),
    ],
)
def test_exact_temporal_boundaries(
    canonical_dataset: pd.DataFrame,
    split_name: str,
    timestamp: pd.Timestamp,
    accepted: bool,
) -> None:
    assert EXPECTED_VALIDATION_START == pd.Timestamp("2025-03-01 00:00:00")
    assert EXPECTED_TEST_START == pd.Timestamp("2025-08-01 00:00:00")
    changed = canonical_dataset.copy(deep=True)
    index = changed.index[changed["split"].eq(split_name)][0]
    changed.loc[index, "prediction_time"] = timestamp
    assert str(changed["prediction_time"].dtype) == "datetime64[ns]"

    if accepted:
        populations = select_eda_populations(changed)
        assert tuple(populations) == EXPECTED_KEYS
    else:
        with pytest.raises(ValueError, match="temporal split inconsistency"):
            select_eda_populations(changed)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("test_relabelled_validation", "temporal split inconsistency"),
        ("immature_train_made_eligible", "development_fit_eligible row count"),
        ("duplicate_appointment", "appointment_id values must be unique"),
        ("null_predictor", "contains null values"),
        ("nonbinary_supervised_target", "target values must be binary"),
        ("unknown_split", "split values must be exactly"),
        ("validation_relabelled_train", "temporal split inconsistency"),
        ("validation_made_development_eligible", "may be true only for train"),
        ("test_made_pretest_eligible", "may be true only for train or validation"),
        ("development_without_pretest", "requires pretest_fit_eligible=True"),
        ("invalid_calendar_feature", "scheduled_hour contains values outside"),
        ("missing_row", "Canonical row count"),
    ],
)
def test_same_dtype_canonical_corruptions_are_rejected(
    canonical_dataset: pd.DataFrame,
    defect: str,
    message: str,
) -> None:
    corrupted = canonical_dataset.copy(deep=True)
    if defect == "test_relabelled_validation":
        index = corrupted.index[corrupted["split"].eq("test")][0]
        corrupted.loc[index, "split"] = "validation"
    elif defect == "immature_train_made_eligible":
        immature = corrupted["split"].eq("train") & ~corrupted[
            "development_fit_eligible"
        ]
        index = corrupted.index[immature][0]
        corrupted.loc[index, "development_fit_eligible"] = True
    elif defect == "duplicate_appointment":
        mature = corrupted["split"].eq("train") & corrupted[
            "development_fit_eligible"
        ]
        duplicate = corrupted.loc[[corrupted.index[mature][0]]].copy(deep=True)
        corrupted = pd.concat((corrupted, duplicate), axis=0)
    elif defect == "null_predictor":
        corrupted.loc[corrupted.index[0], "visit_type"] = pd.NA
    elif defect == "nonbinary_supervised_target":
        mature = corrupted["split"].eq("train") & corrupted[
            "development_fit_eligible"
        ]
        corrupted.loc[corrupted.index[mature][0], "target"] = 2
    elif defect == "unknown_split":
        corrupted.loc[corrupted.index[0], "split"] = "unknown"
    elif defect == "validation_relabelled_train":
        index = corrupted.index[corrupted["split"].eq("validation")][0]
        corrupted.loc[index, "split"] = "train"
    elif defect == "validation_made_development_eligible":
        index = corrupted.index[corrupted["split"].eq("validation")][0]
        corrupted.loc[index, "development_fit_eligible"] = True
    elif defect == "test_made_pretest_eligible":
        index = corrupted.index[corrupted["split"].eq("test")][0]
        corrupted.loc[index, "pretest_fit_eligible"] = True
    elif defect == "development_without_pretest":
        mature = corrupted["split"].eq("train") & corrupted[
            "development_fit_eligible"
        ]
        corrupted.loc[corrupted.index[mature][0], "pretest_fit_eligible"] = False
    elif defect == "invalid_calendar_feature":
        corrupted.loc[corrupted.index[0], "scheduled_hour"] = 24
    else:
        corrupted = corrupted.iloc[:-1].copy(deep=True)

    with pytest.raises(ValueError, match=message):
        select_eda_populations(corrupted)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("planned_duration_min", 1, id="duration-one"),
        pytest.param("booking_lead_time_hours", 24.0, id="lead-time-24"),
        pytest.param("scheduled_weekday", 0, id="weekday-zero"),
        pytest.param("scheduled_weekday", 6, id="weekday-six"),
        pytest.param("scheduled_hour", 0, id="hour-zero"),
        pytest.param("scheduled_hour", 23, id="hour-23"),
        pytest.param("scheduled_month", 1, id="month-one"),
        pytest.param("scheduled_month", 12, id="month-12"),
        pytest.param("approximate_age_at_prediction", 0, id="age-zero"),
        pytest.param(
            "patient_registration_tenure_days",
            0,
            id="patient-tenure-zero",
        ),
        pytest.param("dentist_tenure_days", 0, id="dentist-tenure-zero"),
    ],
)
def test_valid_predictor_domain_boundaries_are_accepted(
    canonical_dataset: pd.DataFrame,
    column: str,
    value: int | float,
) -> None:
    changed = canonical_dataset.copy(deep=True)
    original_dtype = changed[column].dtype
    changed.loc[changed.index[0], column] = value

    assert changed[column].dtype == original_dtype
    populations = select_eda_populations(changed)
    assert tuple(populations) == EXPECTED_KEYS


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("planned_duration_min", 0, id="duration-zero"),
        pytest.param("planned_duration_min", -1, id="duration-negative"),
        pytest.param(
            "booking_lead_time_hours",
            np.nextafter(24.0, -np.inf),
            id="lead-time-immediately-below-24",
        ),
        pytest.param("booking_lead_time_hours", np.inf, id="lead-time-positive-inf"),
        pytest.param("booking_lead_time_hours", -np.inf, id="lead-time-negative-inf"),
        pytest.param("scheduled_weekday", -1, id="weekday-negative"),
        pytest.param("scheduled_weekday", 7, id="weekday-seven"),
        pytest.param("scheduled_hour", -1, id="hour-negative"),
        pytest.param("scheduled_hour", 24, id="hour-24"),
        pytest.param("scheduled_month", 0, id="month-zero"),
        pytest.param("scheduled_month", 13, id="month-13"),
        pytest.param("approximate_age_at_prediction", -1, id="age-negative"),
        pytest.param(
            "patient_registration_tenure_days",
            -1,
            id="patient-tenure-negative",
        ),
        pytest.param("dentist_tenure_days", -1, id="dentist-tenure-negative"),
    ],
)
def test_invalid_predictor_domain_boundaries_are_rejected(
    canonical_dataset: pd.DataFrame,
    column: str,
    value: int | float,
) -> None:
    changed = canonical_dataset.copy(deep=True)
    original_dtype = changed[column].dtype
    changed.loc[changed.index[0], column] = value

    assert changed[column].dtype == original_dtype
    with pytest.raises(ValueError, match=rf"{column} contains values outside"):
        select_eda_populations(changed)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing", "Canonical columns or order differ"),
        ("extra", "Canonical columns or order differ"),
        ("wrong_order", "Canonical columns or order differ"),
        ("wrong_dtype", "appointment_id dtype must be int64"),
        ("prohibited", "Canonical columns or order differ"),
    ],
)
def test_invalid_canonical_inputs_are_rejected(
    canonical_dataset: pd.DataFrame,
    defect: str,
    message: str,
) -> None:
    invalid = canonical_dataset.copy(deep=True)
    if defect == "missing":
        invalid = invalid.drop(columns="visit_type")
    elif defect == "extra":
        invalid["unexpected"] = 1
    elif defect == "wrong_order":
        columns = list(invalid.columns)
        columns[0], columns[1] = columns[1], columns[0]
        invalid = invalid.loc[:, columns]
    elif defect == "wrong_dtype":
        invalid["appointment_id"] = invalid["appointment_id"].astype("int32")
    else:
        invalid["status"] = "completed"

    with pytest.raises(ValueError, match=message):
        select_eda_populations(invalid)


def test_invalid_input_is_rejected_before_development_selection(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = canonical_dataset.drop(columns="visit_type")

    def fail_if_called(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("development selection ran before validation")

    monkeypatch.setattr(bd, "select_development_rows", fail_if_called)
    with pytest.raises(ValueError, match="Canonical columns or order differ"):
        select_eda_populations(invalid)
