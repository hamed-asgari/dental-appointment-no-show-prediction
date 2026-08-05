from __future__ import annotations
from inspect import signature
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import build_dataset as bd
from src.modeling.final_pretest_data import (
    build_final_pretest_prior_data,
)
_EXPECTED_PARAMETERS = (
    "canonical",
)
_EXPECTED_RESULT_KEYS = (
    "pretest_target",
    "test_metadata",
)
_EXPECTED_METADATA_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
)
@pytest.fixture(scope="module")
def authentic_canonical() -> pd.DataFrame:
    raw_dir = Path("data/raw")
    bd.validate_raw_hashes(
        raw_dir
    )
    canonical = (
        bd.build_analytical_dataset(
            bd.load_raw_data(
                raw_dir
            )
        )
    )
    return canonical
def test_public_signature() -> None:
    observed = signature(
        build_final_pretest_prior_data
    )
    assert tuple(
        observed.parameters
    ) == _EXPECTED_PARAMETERS
def test_authentic_result_contract(
    authentic_canonical: pd.DataFrame,
) -> None:
    result = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    assert tuple(
        result
    ) == _EXPECTED_RESULT_KEYS
    assert type(
        result["pretest_target"]
    ) is pd.Series
    assert type(
        result["test_metadata"]
    ) is pd.DataFrame
def test_authentic_pretest_population_is_exact(
    authentic_canonical: pd.DataFrame,
) -> None:
    result = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    target = result[
        "pretest_target"
    ]
    assert type(target) is pd.Series
    assert len(target) == 5223
    assert int(
        target.sum()
    ) == 626
    assert int(
        target.eq(0).sum()
    ) == 4597
    assert target.dtype == np.dtype(
        "int64"
    )
def test_authentic_test_metadata_is_exact(
    authentic_canonical: pd.DataFrame,
) -> None:
    result = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    metadata = result[
        "test_metadata"
    ]
    assert type(metadata) is pd.DataFrame
    assert tuple(
        metadata.columns
    ) == _EXPECTED_METADATA_COLUMNS
    assert len(metadata) == 1563
    assert metadata[
        "appointment_id"
    ].is_unique
    assert tuple(
        metadata[
            "split"
        ].drop_duplicates()
    ) == (
        "test",
    )
    assert metadata[
        "prediction_time"
    ].min() == pd.Timestamp(
        "2025-08-01 09:00:00"
    )
    assert metadata[
        "prediction_time"
    ].max() == pd.Timestamp(
        "2025-12-30 18:30:00"
    )
    assert "target" not in metadata.columns
def test_authentic_indexes_are_disjoint(
    authentic_canonical: pd.DataFrame,
) -> None:
    result = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    target = result[
        "pretest_target"
    ]
    metadata = result[
        "test_metadata"
    ]
    assert type(target) is pd.Series
    assert type(metadata) is pd.DataFrame
    assert target.index.intersection(
        metadata.index
    ).empty
def test_test_target_changes_do_not_affect_contract(
    authentic_canonical: pd.DataFrame,
) -> None:
    first = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    changed = authentic_canonical.copy(
        deep=True
    )
    test_mask = changed[
        "split"
    ].eq(
        "test"
    )
    changed.loc[
        test_mask,
        "target",
    ] = 2
    second = (
        build_final_pretest_prior_data(
            changed
        )
    )
    pd.testing.assert_series_equal(
        first["pretest_target"],
        second["pretest_target"],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        first["test_metadata"],
        second["test_metadata"],
        check_exact=True,
    )
def test_missing_test_targets_do_not_affect_contract(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    test_mask = changed[
        "split"
    ].eq(
        "test"
    )
    changed["target"] = changed[
        "target"
    ].astype(
        "Int64"
    )
    changed.loc[
        test_mask,
        "target",
    ] = pd.NA
    result = (
        build_final_pretest_prior_data(
            changed
        )
    )
    target = result[
        "pretest_target"
    ]
    metadata = result[
        "test_metadata"
    ]
    assert type(target) is pd.Series
    assert type(metadata) is pd.DataFrame
    assert len(target) == 5223
    assert int(
        target.sum()
    ) == 626
    assert "target" not in metadata.columns
def test_repeated_build_is_exact(
    authentic_canonical: pd.DataFrame,
) -> None:
    first = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    second = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    pd.testing.assert_series_equal(
        first["pretest_target"],
        second["pretest_target"],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        first["test_metadata"],
        second["test_metadata"],
        check_exact=True,
    )
def test_result_objects_are_independent(
    authentic_canonical: pd.DataFrame,
) -> None:
    first = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    second = (
        build_final_pretest_prior_data(
            authentic_canonical
        )
    )
    first_target = first[
        "pretest_target"
    ]
    first_metadata = first[
        "test_metadata"
    ]
    second_target = second[
        "pretest_target"
    ]
    second_metadata = second[
        "test_metadata"
    ]
    assert type(first_target) is pd.Series
    assert type(first_metadata) is pd.DataFrame
    assert type(second_target) is pd.Series
    assert type(second_metadata) is pd.DataFrame
    first_target.iloc[0] = 99
    first_metadata.iloc[
        0,
        first_metadata.columns.get_loc(
            "split"
        ),
    ] = "changed"
    assert second_target.iloc[0] in (
        0,
        1,
    )
    assert second_metadata.iloc[
        0
    ][
        "split"
    ] == "test"
def test_input_is_not_mutated(
    authentic_canonical: pd.DataFrame,
) -> None:
    candidate = authentic_canonical.copy(
        deep=True
    )
    snapshot = candidate.copy(
        deep=True
    )
    build_final_pretest_prior_data(
        candidate
    )
    pd.testing.assert_frame_equal(
        candidate,
        snapshot,
        check_exact=True,
    )
@pytest.mark.parametrize(
    "invalid",
    (
        None,
        [],
        pd.Series(
            [0, 1]
        ),
        np.array(
            [0, 1]
        ),
    ),
)
def test_requires_exact_dataframe(
    invalid: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "canonical must be a "
            "pandas DataFrame"
        ),
    ):
        build_final_pretest_prior_data(
            invalid
        )
def test_empty_dataframe_is_rejected() -> None:
    empty = pd.DataFrame()
    with pytest.raises(
        ValueError,
        match=(
            "canonical must not be empty"
        ),
    ):
        build_final_pretest_prior_data(
            empty
        )
@pytest.mark.parametrize(
    "column",
    (
        "appointment_id",
        "prediction_time",
        "target",
        "split",
        "pretest_fit_eligible",
    ),
)
def test_missing_required_columns_are_rejected(
    authentic_canonical: pd.DataFrame,
    column: str,
) -> None:
    changed = authentic_canonical.drop(
        columns=[
            column
        ]
    )
    with pytest.raises(
        ValueError,
        match=(
            "missing required columns"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_duplicate_dataframe_index_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    index = changed.index.to_numpy(
        copy=True
    )
    index[1] = index[0]
    changed.index = index
    with pytest.raises(
        ValueError,
        match=(
            "canonical index must be unique"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_duplicate_appointment_id_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    changed.loc[
        changed.index[1],
        "appointment_id",
    ] = changed.loc[
        changed.index[0],
        "appointment_id",
    ]
    with pytest.raises(
        ValueError,
        match=(
            "appointment_id must be unique"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_prediction_time_requires_datetime_dtype(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    changed[
        "prediction_time"
    ] = changed[
        "prediction_time"
    ].astype(
        "string"
    )
    with pytest.raises(
        TypeError,
        match=(
            "prediction_time must have "
            "a datetime dtype"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_pretest_eligibility_requires_bool_dtype(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    changed[
        "pretest_fit_eligible"
    ] = changed[
        "pretest_fit_eligible"
    ].astype(
        "int64"
    )
    with pytest.raises(
        TypeError,
        match=(
            "pretest_fit_eligible must "
            "have a boolean dtype"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_unknown_split_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    changed.loc[
        changed.index[0],
        "split",
    ] = "unknown"
    with pytest.raises(
        ValueError,
        match=(
            "split must contain exactly"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_incorrect_pretest_eligibility_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    train_index = changed.loc[
        changed[
            "split"
        ].eq(
            "train"
        )
    ].index[0]
    changed.loc[
        train_index,
        "pretest_fit_eligible",
    ] = False
    with pytest.raises(
        ValueError,
        match=(
            "all and only train and "
            "validation rows"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_temporal_overlap_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    pretest_last = changed.loc[
        changed[
            "pretest_fit_eligible"
        ].eq(
            True
        ),
        "prediction_time",
    ].max()
    test_index = changed.loc[
        changed[
            "split"
        ].eq(
            "test"
        )
    ].index[0]
    changed.loc[
        test_index,
        "prediction_time",
    ] = pretest_last
    with pytest.raises(
        ValueError,
        match=(
            "must occur strictly before "
            "test rows"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_missing_pretest_target_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    changed["target"] = changed[
        "target"
    ].astype(
        "Int64"
    )
    pretest_index = changed.loc[
        changed[
            "pretest_fit_eligible"
        ].eq(
            True
        )
    ].index[0]
    changed.loc[
        pretest_index,
        "target",
    ] = pd.NA
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target must not "
            "contain missing values"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
def test_nonbinary_pretest_target_is_rejected(
    authentic_canonical: pd.DataFrame,
) -> None:
    changed = authentic_canonical.copy(
        deep=True
    )
    pretest_index = changed.loc[
        changed[
            "pretest_fit_eligible"
        ].eq(
            True
        )
    ].index[0]
    changed.loc[
        pretest_index,
        "target",
    ] = 2
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target must contain "
            "only 0 and 1"
        ),
    ):
        build_final_pretest_prior_data(
            changed
        )
