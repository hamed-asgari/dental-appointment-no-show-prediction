from __future__ import annotations
from inspect import signature
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import build_dataset as bd
from src.modeling.operational_threshold import (
    evaluate_operational_threshold_states,
)
_EXPECTED_PARAMETERS = (
    "calibration_target",
    "validation_target",
)
_EXPECTED_RESULT_KEYS = (
    "policy_table",
    "selected_probability_model",
    "probability_value",
    "threshold_rule",
    "distinct_policy_states",
)
_EXPECTED_COLUMNS = (
    "policy",
    "threshold",
    "alerted_count",
    "alerted_rate",
    "tn",
    "fp",
    "fn",
    "tp",
)
@pytest.fixture(scope="module")
def authentic_targets() -> tuple[
    pd.Series,
    pd.Series,
]:
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
    development = canonical.loc[
        (
            canonical["split"]
            == "train"
        )
        & canonical[
            "development_fit_eligible"
        ].eq(True)
    ]
    calibration_start = pd.Timestamp(
        "2024-11-01 00:00:00"
    )
    calibration_target = (
        development.loc[
            development[
                "prediction_time"
            ].ge(
                calibration_start
            ),
            "target",
        ]
        .copy(deep=True)
    )
    validation_target = (
        canonical.loc[
            canonical["split"].eq(
                "validation"
            ),
            "target",
        ]
        .copy(deep=True)
    )
    return (
        calibration_target,
        validation_target,
    )
def _tiny_targets() -> tuple[
    pd.Series,
    pd.Series,
]:
    calibration = pd.Series(
        [0, 1, 0, 0],
        index=[10, 11, 12, 13],
        dtype="int64",
        name="no_show",
    )
    validation = pd.Series(
        [0, 1, 0, 1, 0],
        index=[20, 21, 22, 23, 24],
        dtype="int64",
        name="no_show",
    )
    return (
        calibration,
        validation,
    )
def test_public_signature() -> None:
    observed = signature(
        evaluate_operational_threshold_states
    )
    assert tuple(
        observed.parameters
    ) == _EXPECTED_PARAMETERS
def test_authentic_result_contract(
    authentic_targets: tuple[
        pd.Series,
        pd.Series,
    ],
) -> None:
    result = (
        evaluate_operational_threshold_states(
            *authentic_targets
        )
    )
    assert tuple(
        result
    ) == _EXPECTED_RESULT_KEYS
    assert result[
        "selected_probability_model"
    ] == "calibration_prior"
    assert result[
        "threshold_rule"
    ] == "probability >= threshold"
    assert result[
        "distinct_policy_states"
    ] == 2
    assert result[
        "probability_value"
    ] == 0.12521739130434784
def test_authentic_policy_table_is_exact(
    authentic_targets: tuple[
        pd.Series,
        pd.Series,
    ],
) -> None:
    result = (
        evaluate_operational_threshold_states(
            *authentic_targets
        )
    )
    observed = result[
        "policy_table"
    ]
    assert type(observed) is pd.DataFrame
    assert tuple(
        observed.columns
    ) == _EXPECTED_COLUMNS
    probability_value = (
        0.12521739130434784
    )
    expected = pd.DataFrame(
        [
            {
                "policy": "intervene_all",
                "threshold": (
                    probability_value
                ),
                "alerted_count": 1541,
                "alerted_rate": 1.0,
                "tn": 0,
                "fp": 1349,
                "fn": 0,
                "tp": 192,
            },
            {
                "policy": "intervene_none",
                "threshold": float(
                    np.nextafter(
                        probability_value,
                        np.inf,
                    )
                ),
                "alerted_count": 0,
                "alerted_rate": 0.0,
                "tn": 1349,
                "fp": 0,
                "fn": 192,
                "tp": 0,
            },
        ],
        columns=_EXPECTED_COLUMNS,
    )
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_exact=True,
    )
def test_threshold_equal_to_probability_alerts_all(
    authentic_targets: tuple[
        pd.Series,
        pd.Series,
    ],
) -> None:
    result = (
        evaluate_operational_threshold_states(
            *authentic_targets
        )
    )
    table = result[
        "policy_table"
    ]
    assert type(table) is pd.DataFrame
    all_policy = table.loc[
        table["policy"].eq(
            "intervene_all"
        )
    ].iloc[0]
    assert all_policy[
        "threshold"
    ] == result[
        "probability_value"
    ]
    assert all_policy[
        "alerted_count"
    ] == 1541
def test_next_float_above_probability_alerts_none(
    authentic_targets: tuple[
        pd.Series,
        pd.Series,
    ],
) -> None:
    result = (
        evaluate_operational_threshold_states(
            *authentic_targets
        )
    )
    table = result[
        "policy_table"
    ]
    assert type(table) is pd.DataFrame
    none_policy = table.loc[
        table["policy"].eq(
            "intervene_none"
        )
    ].iloc[0]
    assert none_policy[
        "threshold"
    ] == np.nextafter(
        result["probability_value"],
        np.inf,
    )
    assert none_policy[
        "alerted_count"
    ] == 0
def test_probability_uses_only_calibration_target() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    first = (
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
    )
    changed_validation = pd.Series(
        [1, 0, 1, 0, 1],
        index=validation.index,
        dtype="int64",
        name="no_show",
    )
    second = (
        evaluate_operational_threshold_states(
            calibration,
            changed_validation,
        )
    )
    assert first[
        "probability_value"
    ] == 0.25
    assert second[
        "probability_value"
    ] == 0.25
def test_repeated_evaluation_is_exact() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    first = (
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
    )
    second = (
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
    )
    assert first[
        "probability_value"
    ] == second[
        "probability_value"
    ]
    assert first[
        "threshold_rule"
    ] == second[
        "threshold_rule"
    ]
    assert first[
        "distinct_policy_states"
    ] == second[
        "distinct_policy_states"
    ]
    pd.testing.assert_frame_equal(
        first["policy_table"],
        second["policy_table"],
        check_exact=True,
    )
def test_result_tables_are_independent() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    first = (
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
    )
    second = (
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
    )
    first_table = first[
        "policy_table"
    ]
    second_table = second[
        "policy_table"
    ]
    assert type(first_table) is pd.DataFrame
    assert type(second_table) is pd.DataFrame
    first_table.loc[
        0,
        "alerted_count",
    ] = -1
    assert second_table.loc[
        0,
        "alerted_count",
    ] == 5
def test_inputs_are_not_mutated() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    calibration_snapshot = (
        calibration.copy(
            deep=True
        )
    )
    validation_snapshot = (
        validation.copy(
            deep=True
        )
    )
    evaluate_operational_threshold_states(
        calibration,
        validation,
    )
    pd.testing.assert_series_equal(
        calibration,
        calibration_snapshot,
    )
    pd.testing.assert_series_equal(
        validation,
        validation_snapshot,
    )
@pytest.mark.parametrize(
    "argument_name",
    (
        "calibration_target",
        "validation_target",
    ),
)
@pytest.mark.parametrize(
    "invalid",
    (
        [0, 1],
        np.array([0, 1]),
        pd.DataFrame(
            {"target": [0, 1]}
        ),
        None,
    ),
)
def test_inputs_require_exact_series(
    argument_name: str,
    invalid: object,
) -> None:
    calibration, validation = (
        _tiny_targets()
    )
    arguments: dict[str, object] = {
        "calibration_target": (
            calibration
        ),
        "validation_target": (
            validation
        ),
    }
    arguments[argument_name] = invalid
    with pytest.raises(
        TypeError,
        match=(
            f"{argument_name} must be "
            "a pandas Series"
        ),
    ):
        evaluate_operational_threshold_states(
            arguments[
                "calibration_target"
            ],
            arguments[
                "validation_target"
            ],
        )
@pytest.mark.parametrize(
    "argument_name",
    (
        "calibration_target",
        "validation_target",
    ),
)
def test_empty_targets_are_rejected(
    argument_name: str,
) -> None:
    calibration, validation = (
        _tiny_targets()
    )
    empty = pd.Series(
        [],
        dtype="int64",
    )
    arguments = {
        "calibration_target": (
            calibration
        ),
        "validation_target": (
            validation
        ),
    }
    arguments[argument_name] = empty
    with pytest.raises(
        ValueError,
        match=(
            f"{argument_name} must "
            "not be empty"
        ),
    ):
        evaluate_operational_threshold_states(
            arguments[
                "calibration_target"
            ],
            arguments[
                "validation_target"
            ],
        )
@pytest.mark.parametrize(
    "argument_name",
    (
        "calibration_target",
        "validation_target",
    ),
)
@pytest.mark.parametrize(
    "values",
    (
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 2],
    ),
)
def test_invalid_class_contract_is_rejected(
    argument_name: str,
    values: list[int],
) -> None:
    calibration, validation = (
        _tiny_targets()
    )
    invalid = pd.Series(
        values,
        index=range(
            100,
            100 + len(values),
        ),
        dtype="int64",
    )
    arguments = {
        "calibration_target": (
            calibration
        ),
        "validation_target": (
            validation
        ),
    }
    arguments[argument_name] = invalid
    with pytest.raises(
        ValueError,
    ):
        evaluate_operational_threshold_states(
            arguments[
                "calibration_target"
            ],
            arguments[
                "validation_target"
            ],
        )
@pytest.mark.parametrize(
    "argument_name",
    (
        "calibration_target",
        "validation_target",
    ),
)
def test_missing_values_are_rejected(
    argument_name: str,
) -> None:
    calibration, validation = (
        _tiny_targets()
    )
    invalid = pd.Series(
        [0, 1, pd.NA],
        index=[100, 101, 102],
        dtype="Int64",
    )
    arguments = {
        "calibration_target": (
            calibration
        ),
        "validation_target": (
            validation
        ),
    }
    arguments[argument_name] = invalid
    with pytest.raises(
        ValueError,
        match=(
            f"{argument_name} must not "
            "contain missing values"
        ),
    ):
        evaluate_operational_threshold_states(
            arguments[
                "calibration_target"
            ],
            arguments[
                "validation_target"
            ],
        )
@pytest.mark.parametrize(
    "dtype",
    (
        "bool",
        "float64",
        "object",
    ),
)
def test_non_integer_dtypes_are_rejected(
    dtype: str,
) -> None:
    calibration = pd.Series(
        [0, 1, 0, 1],
        index=[10, 11, 12, 13],
        dtype=dtype,
    )
    _, validation = _tiny_targets()
    with pytest.raises(
        TypeError,
        match=(
            "calibration_target must "
            "have an integer dtype"
        ),
    ):
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
def test_duplicate_indexes_are_rejected() -> None:
    calibration = pd.Series(
        [0, 1, 0],
        index=[10, 10, 11],
        dtype="int64",
    )
    _, validation = _tiny_targets()
    with pytest.raises(
        ValueError,
        match=(
            "calibration_target index "
            "must be unique"
        ),
    ):
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
def test_overlapping_indexes_are_rejected() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    validation = validation.copy(
        deep=True
    )
    validation.index = [
        10,
        21,
        22,
        23,
        24,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "indexes must be disjoint"
        ),
    ):
        evaluate_operational_threshold_states(
            calibration,
            validation,
        )
