from __future__ import annotations
from inspect import signature
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import build_dataset as bd
from src.modeling.calibration_data import (
    build_calibration_modeling_data,
)
from src.modeling.operational_cost import (
    evaluate_operational_break_even_sensitivity,
)
_EXPECTED_PARAMETERS = (
    "calibration_target",
    "validation_target",
)
_EXPECTED_RESULT_KEYS = (
    "sensitivity_table",
    "selected_probability_model",
    "threshold_rule",
    "distinct_policy_states",
    "decision_rule",
    "ex_ante_prevalence",
    "validation_replay_prevalence",
    "ex_ante_source",
    "validation_role",
    "specific_cost_values_selected",
    "effectiveness_selected",
    "operational_policy_selected",
)
_EXPECTED_COLUMNS = (
    "effectiveness",
    (
        "ex_ante_break_even_"
        "intervention_to_no_show_cost"
    ),
    (
        "validation_replay_break_even_"
        "intervention_to_no_show_cost"
    ),
)
_EXPECTED_EFFECTIVENESS = np.array(
    [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    ],
    dtype=np.float64,
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
    data = (
        build_calibration_modeling_data(
            canonical
        )
    )
    calibration_target = data[
        "calibration_target"
    ].copy(
        deep=True
    )
    validation_target = data[
        "validation_target"
    ].copy(
        deep=True
    )
    assert type(
        calibration_target
    ) is pd.Series
    assert type(
        validation_target
    ) is pd.Series
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
        name="target",
    )
    validation = pd.Series(
        [0, 1, 0, 1, 0],
        index=[20, 21, 22, 23, 24],
        dtype="int64",
        name="target",
    )
    return (
        calibration,
        validation,
    )
def test_public_signature() -> None:
    observed = signature(
        evaluate_operational_break_even_sensitivity
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
        evaluate_operational_break_even_sensitivity(
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
        "ex_ante_prevalence"
    ] == 0.12521739130434784
    assert result[
        "validation_replay_prevalence"
    ] == 0.1245944192083063
    assert result[
        "ex_ante_source"
    ] == "calibration_target only"
    assert result[
        "validation_role"
    ] == "replay audit only"
def test_authentic_sensitivity_table_is_exact(
    authentic_targets: tuple[
        pd.Series,
        pd.Series,
    ],
) -> None:
    result = (
        evaluate_operational_break_even_sensitivity(
            *authentic_targets
        )
    )
    observed = result[
        "sensitivity_table"
    ]
    assert type(observed) is pd.DataFrame
    expected = pd.DataFrame(
        {
            "effectiveness": (
                _EXPECTED_EFFECTIVENESS
            ),
            (
                "ex_ante_break_even_"
                "intervention_to_no_show_cost"
            ): (
                0.12521739130434784
                * _EXPECTED_EFFECTIVENESS
            ),
            (
                "validation_replay_break_even_"
                "intervention_to_no_show_cost"
            ): (
                0.1245944192083063
                * _EXPECTED_EFFECTIVENESS
            ),
        },
        columns=_EXPECTED_COLUMNS,
    )
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_exact=True,
    )
def test_tiny_sensitivity_table_is_exact() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    result = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    observed = result[
        "sensitivity_table"
    ]
    assert type(observed) is pd.DataFrame
    expected = pd.DataFrame(
        {
            "effectiveness": np.array(
                [
                    0.00,
                    0.25,
                    0.50,
                    0.75,
                    1.00,
                ],
                dtype=np.float64,
            ),
            (
                "ex_ante_break_even_"
                "intervention_to_no_show_cost"
            ): np.array(
                [
                    0.0000,
                    0.0625,
                    0.1250,
                    0.1875,
                    0.2500,
                ],
                dtype=np.float64,
            ),
            (
                "validation_replay_break_even_"
                "intervention_to_no_show_cost"
            ): (
                0.4
                * _EXPECTED_EFFECTIVENESS
            ),
        },
        columns=_EXPECTED_COLUMNS,
    )
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_exact=True,
    )
def test_validation_changes_only_replay_boundary() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    first = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    changed_validation = pd.Series(
        [1, 1, 1, 0, 1],
        index=validation.index,
        dtype="int64",
        name="target",
    )
    second = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            changed_validation,
        )
    )
    assert first[
        "ex_ante_prevalence"
    ] == 0.25
    assert second[
        "ex_ante_prevalence"
    ] == 0.25
    assert first[
        "validation_replay_prevalence"
    ] == 0.4
    assert second[
        "validation_replay_prevalence"
    ] == 0.8
    first_table = first[
        "sensitivity_table"
    ]
    second_table = second[
        "sensitivity_table"
    ]
    assert type(first_table) is pd.DataFrame
    assert type(second_table) is pd.DataFrame
    ex_ante_column = (
        "ex_ante_break_even_"
        "intervention_to_no_show_cost"
    )
    np.testing.assert_array_equal(
        first_table[
            ex_ante_column
        ].to_numpy(),
        second_table[
            ex_ante_column
        ].to_numpy(),
    )
def test_no_operational_inputs_are_selected() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    result = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    assert result[
        "specific_cost_values_selected"
    ] is False
    assert result[
        "effectiveness_selected"
    ] is False
    assert result[
        "operational_policy_selected"
    ] is False
    assert result[
        "decision_rule"
    ] == (
        "intervene_all preferred when "
        "intervention_cost / no_show_cost "
        "< prevalence * effectiveness"
    )
def test_repeated_evaluation_is_exact() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    first = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    second = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    assert first[
        "decision_rule"
    ] == second[
        "decision_rule"
    ]
    assert first[
        "ex_ante_prevalence"
    ] == second[
        "ex_ante_prevalence"
    ]
    assert first[
        "validation_replay_prevalence"
    ] == second[
        "validation_replay_prevalence"
    ]
    pd.testing.assert_frame_equal(
        first["sensitivity_table"],
        second["sensitivity_table"],
        check_exact=True,
    )
def test_result_tables_are_independent() -> None:
    calibration, validation = (
        _tiny_targets()
    )
    first = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    second = (
        evaluate_operational_break_even_sensitivity(
            calibration,
            validation,
        )
    )
    first_table = first[
        "sensitivity_table"
    ]
    second_table = second[
        "sensitivity_table"
    ]
    assert type(first_table) is pd.DataFrame
    assert type(second_table) is pd.DataFrame
    first_table.loc[
        0,
        "effectiveness",
    ] = -1.0
    assert second_table.loc[
        0,
        "effectiveness",
    ] == 0.0
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
    evaluate_operational_break_even_sensitivity(
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
def test_invalid_inputs_are_rejected(
    argument_name: str,
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
    arguments[argument_name] = [0, 1]
    with pytest.raises(
        TypeError,
        match=(
            f"{argument_name} must be "
            "a pandas Series"
        ),
    ):
        evaluate_operational_break_even_sensitivity(
            arguments[
                "calibration_target"
            ],
            arguments[
                "validation_target"
            ],
        )
