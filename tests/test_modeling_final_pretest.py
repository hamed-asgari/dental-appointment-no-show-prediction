from __future__ import annotations
from inspect import signature
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import build_dataset as bd
from src.modeling.final_pretest import (
    fit_final_pretest_prior,
)
from src.modeling.final_pretest_data import (
    build_final_pretest_prior_data,
)
_EXPECTED_PARAMETERS = (
    "pretest_target",
    "test_metadata",
)
_EXPECTED_RESULT_KEYS = (
    "selected_probability_model",
    "probability_value",
    "fit_population_rows",
    "fit_population_positives",
    "fit_population_negatives",
    "test_rows",
    "probability_source",
    "test_probability",
    "model_comparison_reopened",
    "threshold_selected",
    "operational_policy_selected",
    "test_target_used",
)
_EXPECTED_PROBABILITY = float(
    626 / 5223
)
@pytest.fixture(scope="module")
def authentic_inputs() -> tuple[
    pd.Series,
    pd.DataFrame,
]:
    raw_dir = Path(
        "data/raw"
    )
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
    contract = (
        build_final_pretest_prior_data(
            canonical
        )
    )
    target = contract[
        "pretest_target"
    ]
    metadata = contract[
        "test_metadata"
    ]
    assert type(target) is pd.Series
    assert type(metadata) is pd.DataFrame
    return (
        target,
        metadata,
    )
def _tiny_inputs() -> tuple[
    pd.Series,
    pd.DataFrame,
]:
    target = pd.Series(
        [0, 1, 0, 1],
        index=[
            10,
            11,
            12,
            13,
        ],
        name="target",
        dtype=np.int64,
    )
    metadata = pd.DataFrame(
        {
            "appointment_id": [
                "A-20",
                "A-21",
            ],
            "prediction_time": pd.to_datetime(
                [
                    "2025-08-01 09:00:00",
                    "2025-08-01 09:30:00",
                ]
            ),
            "split": [
                "test",
                "test",
            ],
        },
        index=[
            20,
            21,
        ],
    )
    return (
        target,
        metadata,
    )
def test_public_signature() -> None:
    observed = signature(
        fit_final_pretest_prior
    )
    assert tuple(
        observed.parameters
    ) == _EXPECTED_PARAMETERS
def test_authentic_result_contract(
    authentic_inputs: tuple[
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    result = fit_final_pretest_prior(
        *authentic_inputs
    )
    assert tuple(
        result
    ) == _EXPECTED_RESULT_KEYS
    assert result[
        "selected_probability_model"
    ] == "calibration_prior"
    assert result[
        "probability_value"
    ] == _EXPECTED_PROBABILITY
    assert result[
        "fit_population_rows"
    ] == 5223
    assert result[
        "fit_population_positives"
    ] == 626
    assert result[
        "fit_population_negatives"
    ] == 4597
    assert result[
        "test_rows"
    ] == 1563
    assert result[
        "probability_source"
    ] == "mean of pretest_target"
    assert result[
        "model_comparison_reopened"
    ] is False
    assert result[
        "threshold_selected"
    ] is False
    assert result[
        "operational_policy_selected"
    ] is False
    assert result[
        "test_target_used"
    ] is False
def test_authentic_test_probability_is_exact(
    authentic_inputs: tuple[
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    _, metadata = authentic_inputs
    result = fit_final_pretest_prior(
        *authentic_inputs
    )
    probability = result[
        "test_probability"
    ]
    assert type(probability) is pd.Series
    assert probability.name == (
        "no_show_probability"
    )
    assert probability.dtype == np.dtype(
        "float64"
    )
    assert len(probability) == 1563
    pd.testing.assert_index_equal(
        probability.index,
        metadata.index,
        exact=True,
    )
    np.testing.assert_array_equal(
        probability.to_numpy(
            dtype=np.float64,
            copy=True,
        ),
        np.full(
            1563,
            _EXPECTED_PROBABILITY,
            dtype=np.float64,
        ),
    )
def test_probability_matches_existing_prior_definition(
    authentic_inputs: tuple[
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    target, _ = authentic_inputs
    result = fit_final_pretest_prior(
        *authentic_inputs
    )
    assert result[
        "probability_value"
    ] == float(
        target.mean()
    )
def test_changed_pretest_target_updates_prior_exactly() -> None:
    target, metadata = _tiny_inputs()
    first = fit_final_pretest_prior(
        target,
        metadata,
    )
    changed = target.copy(
        deep=True
    )
    changed.loc[
        changed.eq(0).index[0]
    ] = 1
    second = fit_final_pretest_prior(
        changed,
        metadata,
    )
    assert first[
        "probability_value"
    ] == 0.5
    assert second[
        "probability_value"
    ] == 0.75
def test_test_metadata_contents_do_not_change_prior() -> None:
    target, metadata = _tiny_inputs()
    first = fit_final_pretest_prior(
        target,
        metadata,
    )
    changed = metadata.copy(
        deep=True
    )
    changed[
        "appointment_id"
    ] = [
        "changed-1",
        "changed-2",
    ]
    changed[
        "prediction_time"
    ] = pd.to_datetime(
        [
            "2026-01-01 09:00:00",
            "2026-01-01 09:30:00",
        ]
    )
    second = fit_final_pretest_prior(
        target,
        changed,
    )
    assert first[
        "probability_value"
    ] == second[
        "probability_value"
    ]
    pd.testing.assert_series_equal(
        first["test_probability"],
        second["test_probability"],
        check_exact=True,
    )
def test_repeated_fit_is_exact(
    authentic_inputs: tuple[
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    first = fit_final_pretest_prior(
        *authentic_inputs
    )
    second = fit_final_pretest_prior(
        *authentic_inputs
    )
    scalar_keys = tuple(
        key
        for key in _EXPECTED_RESULT_KEYS
        if key != "test_probability"
    )
    for key in scalar_keys:
        assert first[key] == second[key]
    pd.testing.assert_series_equal(
        first["test_probability"],
        second["test_probability"],
        check_exact=True,
    )
def test_result_probability_is_independent() -> None:
    target, metadata = _tiny_inputs()
    first = fit_final_pretest_prior(
        target,
        metadata,
    )
    second = fit_final_pretest_prior(
        target,
        metadata,
    )
    first_probability = first[
        "test_probability"
    ]
    second_probability = second[
        "test_probability"
    ]
    assert type(
        first_probability
    ) is pd.Series
    assert type(
        second_probability
    ) is pd.Series
    first_probability.iloc[0] = 0.0
    assert second_probability.iloc[0] == 0.5
def test_inputs_are_not_mutated() -> None:
    target, metadata = _tiny_inputs()
    target_snapshot = target.copy(
        deep=True
    )
    metadata_snapshot = metadata.copy(
        deep=True
    )
    fit_final_pretest_prior(
        target,
        metadata,
    )
    pd.testing.assert_series_equal(
        target,
        target_snapshot,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        metadata,
        metadata_snapshot,
        check_exact=True,
    )
@pytest.mark.parametrize(
    "invalid",
    (
        None,
        [],
        pd.DataFrame(
            {
                "target": [
                    0,
                    1,
                ]
            }
        ),
        np.array(
            [0, 1]
        ),
    ),
)
def test_pretest_target_requires_exact_series(
    invalid: object,
) -> None:
    _, metadata = _tiny_inputs()
    with pytest.raises(
        TypeError,
        match=(
            "pretest_target must be an "
            "exact pandas Series"
        ),
    ):
        fit_final_pretest_prior(
            invalid,
            metadata,
        )
@pytest.mark.parametrize(
    "invalid",
    (
        None,
        [],
        pd.Series(
            [1, 2]
        ),
        np.array(
            [[1, 2]]
        ),
    ),
)
def test_test_metadata_requires_exact_dataframe(
    invalid: object,
) -> None:
    target, _ = _tiny_inputs()
    with pytest.raises(
        TypeError,
        match=(
            "test_metadata must be an "
            "exact pandas DataFrame"
        ),
    ):
        fit_final_pretest_prior(
            target,
            invalid,
        )
def test_empty_pretest_target_is_rejected() -> None:
    _, metadata = _tiny_inputs()
    target = pd.Series(
        [],
        dtype=np.int64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target must not be empty"
        ),
    ):
        fit_final_pretest_prior(
            target,
            metadata,
        )
def test_duplicate_pretest_index_is_rejected() -> None:
    _, metadata = _tiny_inputs()
    target = pd.Series(
        [0, 1],
        index=[
            10,
            10,
        ],
        dtype=np.int64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target index must be unique"
        ),
    ):
        fit_final_pretest_prior(
            target,
            metadata,
        )
def test_missing_pretest_target_is_rejected() -> None:
    _, metadata = _tiny_inputs()
    target = pd.Series(
        [0, pd.NA, 1],
        dtype="Int64",
    )
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target must not contain "
            "missing values"
        ),
    ):
        fit_final_pretest_prior(
            target,
            metadata,
        )
@pytest.mark.parametrize(
    "target",
    (
        pd.Series(
            [False, True],
            dtype=bool,
        ),
        pd.Series(
            [0.0, 1.0],
            dtype=np.float64,
        ),
    ),
)
def test_pretest_target_requires_integer_dtype(
    target: pd.Series,
) -> None:
    _, metadata = _tiny_inputs()
    with pytest.raises(
        TypeError,
        match=(
            "pretest_target must have an "
            "integer dtype"
        ),
    ):
        fit_final_pretest_prior(
            target,
            metadata,
        )
def test_nonbinary_pretest_target_is_rejected() -> None:
    _, metadata = _tiny_inputs()
    target = pd.Series(
        [0, 1, 2],
        dtype=np.int64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target must contain "
            "only 0 and 1"
        ),
    ):
        fit_final_pretest_prior(
            target,
            metadata,
        )
@pytest.mark.parametrize(
    "target",
    (
        pd.Series(
            [0, 0],
            dtype=np.int64,
        ),
        pd.Series(
            [1, 1],
            dtype=np.int64,
        ),
    ),
)
def test_pretest_target_requires_both_classes(
    target: pd.Series,
) -> None:
    _, metadata = _tiny_inputs()
    with pytest.raises(
        ValueError,
        match=(
            "pretest_target must contain "
            "both classes"
        ),
    ):
        fit_final_pretest_prior(
            target,
            metadata,
        )
def test_empty_test_metadata_is_rejected() -> None:
    target, metadata = _tiny_inputs()
    empty = metadata.iloc[
        0:0
    ].copy(
        deep=True
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_metadata must not be empty"
        ),
    ):
        fit_final_pretest_prior(
            target,
            empty,
        )
def test_extra_test_target_column_is_rejected() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.copy(
        deep=True
    )
    changed[
        "target"
    ] = [
        0,
        1,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "test_metadata columns must be "
            "exactly"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
def test_reordered_test_columns_are_rejected() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.loc[
        :,
        [
            "split",
            "appointment_id",
            "prediction_time",
        ],
    ].copy(
        deep=True
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_metadata columns must be "
            "exactly"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
def test_duplicate_test_index_is_rejected() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.copy(
        deep=True
    )
    changed.index = [
        20,
        20,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "test_metadata index must be unique"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
def test_duplicate_test_appointment_id_is_rejected() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.copy(
        deep=True
    )
    changed[
        "appointment_id"
    ] = [
        "A-20",
        "A-20",
    ]
    with pytest.raises(
        ValueError,
        match=(
            "test appointment_id must be unique"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
def test_test_prediction_time_requires_datetime() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.copy(
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
            "test prediction_time must have "
            "a datetime dtype"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
def test_non_test_split_is_rejected() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.copy(
        deep=True
    )
    changed.loc[
        changed.index[0],
        "split",
    ] = "validation"
    with pytest.raises(
        ValueError,
        match=(
            "test_metadata must contain only "
            "test rows"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
def test_overlapping_indexes_are_rejected() -> None:
    target, metadata = _tiny_inputs()
    changed = metadata.copy(
        deep=True
    )
    changed.index = [
        10,
        20,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "indexes must be disjoint"
        ),
    ):
        fit_final_pretest_prior(
            target,
            changed,
        )
