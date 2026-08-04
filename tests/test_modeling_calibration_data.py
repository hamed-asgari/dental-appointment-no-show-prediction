"""Tests for chronological calibration modeling populations."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from src.data import build_dataset as bd
from src.modeling import calibration_data
from src.modeling.calibration_data import (
    CALIBRATION_START,
    build_calibration_modeling_data,
)
@pytest.fixture
def canonical_dataset() -> pd.DataFrame:
    """Return a fresh authentic canonical dataset."""
    raw_dir = Path("data/raw")
    bd.validate_raw_hashes(raw_dir)

    return bd.build_analytical_dataset(
        bd.load_raw_data(raw_dir)
    )


_EXPECTED_KEYS = (
    "base_fit_features",
    "base_fit_target",
    "calibration_features",
    "calibration_target",
    "validation_features",
    "validation_target",
)
_EXPECTED_COUNTS = {
    "base_fit_target": {
        "rows": 2_520,
        "positives": 288,
        "negatives": 2_232,
    },
    "calibration_target": {
        "rows": 1_150,
        "positives": 144,
        "negatives": 1_006,
    },
    "validation_target": {
        "rows": 1_541,
        "positives": 192,
        "negatives": 1_349,
    },
}
def test_public_contract_and_authentic_counts(
    canonical_dataset: pd.DataFrame,
) -> None:
    result = build_calibration_modeling_data(
        canonical_dataset
    )
    assert CALIBRATION_START == pd.Timestamp(
        "2024-11-01 00:00:00"
    )
    assert tuple(result) == _EXPECTED_KEYS
    for target_name, expected in (
        _EXPECTED_COUNTS.items()
    ):
        target = result[target_name]
        assert type(target) is pd.Series
        assert len(target) == expected["rows"]
        assert int(target.sum()) == (
            expected["positives"]
        )
        assert int(target.eq(0).sum()) == (
            expected["negatives"]
        )
        assert set(target.unique()) == {0, 1}
    for feature_name in (
        "base_fit_features",
        "calibration_features",
        "validation_features",
    ):
        features = result[feature_name]
        target_name = feature_name.replace(
            "_features",
            "_target",
        )
        assert type(features) is pd.DataFrame
        assert features.index.equals(
            result[target_name].index
        )
    assert tuple(
        result["base_fit_features"].columns
    ) == tuple(
        result[
            "calibration_features"
        ].columns
    )
    assert tuple(
        result["base_fit_features"].columns
    ) == tuple(
        result[
            "validation_features"
        ].columns
    )
def test_populations_are_disjoint_and_chronological(
    canonical_dataset: pd.DataFrame,
) -> None:
    result = build_calibration_modeling_data(
        canonical_dataset
    )
    base_index = result[
        "base_fit_features"
    ].index
    calibration_index = result[
        "calibration_features"
    ].index
    validation_index = result[
        "validation_features"
    ].index
    assert base_index.is_unique
    assert calibration_index.is_unique
    assert validation_index.is_unique
    assert base_index.intersection(
        calibration_index
    ).empty
    assert base_index.intersection(
        validation_index
    ).empty
    assert calibration_index.intersection(
        validation_index
    ).empty
    base_time = canonical_dataset.loc[
        base_index,
        "prediction_time",
    ]
    calibration_time = canonical_dataset.loc[
        calibration_index,
        "prediction_time",
    ]
    validation_time = canonical_dataset.loc[
        validation_index,
        "prediction_time",
    ]
    assert base_time.lt(
        CALIBRATION_START
    ).all()
    assert calibration_time.ge(
        CALIBRATION_START
    ).all()
    assert calibration_time.lt(
        bd.VALIDATION_START
    ).all()
    assert validation_time.ge(
        bd.VALIDATION_START
    ).all()
    assert validation_time.lt(
        bd.TEST_START
    ).all()
    assert (
        base_time.max()
        < calibration_time.min()
    )
    assert (
        calibration_time.max()
        < validation_time.min()
    )
def test_repeated_calls_are_fresh_and_nonmutating(
    canonical_dataset: pd.DataFrame,
) -> None:
    canonical_before = canonical_dataset.copy(
        deep=True
    )
    first = build_calibration_modeling_data(
        canonical_dataset
    )
    second = build_calibration_modeling_data(
        canonical_dataset
    )
    assert first is not second
    for key in _EXPECTED_KEYS:
        assert first[key] is not second[key]
        if type(first[key]) is pd.DataFrame:
            pd.testing.assert_frame_equal(
                first[key],
                second[key],
            )
        else:
            pd.testing.assert_series_equal(
                first[key],
                second[key],
            )
    pd.testing.assert_frame_equal(
        canonical_dataset,
        canonical_before,
    )
def test_return_mutation_cannot_affect_future_calls(
    canonical_dataset: pd.DataFrame,
) -> None:
    expected = build_calibration_modeling_data(
        canonical_dataset
    )
    changed = build_calibration_modeling_data(
        canonical_dataset
    )
    changed[
        "base_fit_features"
    ].iloc[0, 0] = -999
    changed[
        "base_fit_target"
    ].iloc[0] = (
        1
        - changed[
            "base_fit_target"
        ].iloc[0]
    )
    repeated = build_calibration_modeling_data(
        canonical_dataset
    )
    for key in _EXPECTED_KEYS:
        if type(expected[key]) is pd.DataFrame:
            pd.testing.assert_frame_equal(
                repeated[key],
                expected[key],
            )
        else:
            pd.testing.assert_series_equal(
                repeated[key],
                expected[key],
            )
def test_boundary_is_half_open(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mature_train = (
        canonical_dataset["split"].eq(
            "train"
        )
        & canonical_dataset[
            "development_fit_eligible"
        ].eq(True)
    )
    development_times = canonical_dataset.loc[
        mature_train,
        "prediction_time",
    ]
    boundary = development_times.iloc[
        len(development_times) // 2
    ]
    monkeypatch.setattr(
        calibration_data,
        "CALIBRATION_START",
        boundary,
    )
    result = (
        calibration_data
        .build_calibration_modeling_data(
            canonical_dataset
        )
    )
    base_time = canonical_dataset.loc[
        result[
            "base_fit_features"
        ].index,
        "prediction_time",
    ]
    calibration_time = canonical_dataset.loc[
        result[
            "calibration_features"
        ].index,
        "prediction_time",
    ]
    assert base_time.lt(boundary).all()
    assert calibration_time.ge(
        boundary
    ).all()
    assert calibration_time.eq(
        boundary
    ).any()
def test_empty_base_fit_population_is_rejected(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration_data,
        "CALIBRATION_START",
        pd.Timestamp(
            "2024-03-01 00:00:00"
        ),
    )
    with pytest.raises(
        ValueError,
        match=(
            "base-fit population must "
            "not be empty"
        ),
    ):
        (
            calibration_data
            .build_calibration_modeling_data(
                canonical_dataset
            )
        )
def test_empty_calibration_population_is_rejected(
    canonical_dataset: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration_data,
        "CALIBRATION_START",
        bd.VALIDATION_START,
    )
    with pytest.raises(
        ValueError,
        match=(
            "calibration population must "
            "not be empty"
        ),
    ):
        (
            calibration_data
            .build_calibration_modeling_data(
                canonical_dataset
            )
        )
def test_invalid_canonical_is_rejected(
    canonical_dataset: pd.DataFrame,
) -> None:
    corrupted = canonical_dataset.copy(
        deep=True
    )
    corrupted.loc[
        corrupted.index[1],
        "appointment_id",
    ] = corrupted.loc[
        corrupted.index[0],
        "appointment_id",
    ]
    with pytest.raises(
        ValueError,
        match="appointment_id must be unique",
    ):
        build_calibration_modeling_data(
            corrupted
        )
def test_test_target_poisoning_is_invariant(
    canonical_dataset: pd.DataFrame,
) -> None:
    expected = build_calibration_modeling_data(
        canonical_dataset
    )
    poisoned = canonical_dataset.copy(
        deep=True
    )
    test_mask = poisoned[
        "split"
    ].eq("test")
    poisoned.loc[
        test_mask,
        "target",
    ] = (
        1
        - poisoned.loc[
            test_mask,
            "target",
        ]
    ).astype("int8")
    observed = build_calibration_modeling_data(
        poisoned
    )
    for key in _EXPECTED_KEYS:
        if type(expected[key]) is pd.DataFrame:
            pd.testing.assert_frame_equal(
                observed[key],
                expected[key],
            )
        else:
            pd.testing.assert_series_equal(
                observed[key],
                expected[key],
            )
