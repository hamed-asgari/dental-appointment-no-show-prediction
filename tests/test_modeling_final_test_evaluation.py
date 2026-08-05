from __future__ import annotations
from inspect import signature
import numpy as np
import pandas as pd
import pytest
from src.modeling.final_test_evaluation import (
    evaluate_final_test_prior,
)
_EXPECTED_PARAMETERS = (
    "test_target",
    "test_probability",
)
_EXPECTED_RESULT_KEYS = (
    "selected_probability_model",
    "evaluation_role",
    "test_rows",
    "test_positives",
    "test_negatives",
    "test_prevalence",
    "probability_value",
    "unique_probability_values",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
    "ranking_available",
    "model_comparison_reopened",
    "threshold_selected",
    "operational_policy_selected",
    "test_target_used",
)
def _valid_inputs() -> tuple[
    pd.Series,
    pd.Series,
]:
    target = pd.Series(
        [
            0,
            1,
            0,
            1,
        ],
        index=[
            20,
            21,
            22,
            23,
        ],
        name="target",
        dtype=np.int64,
    )
    probability = pd.Series(
        np.full(
            4,
            0.5,
            dtype=np.float64,
        ),
        index=target.index.copy(
            deep=True
        ),
        name="no_show_probability",
        dtype=np.float64,
    )
    return (
        target,
        probability,
    )
def test_public_signature() -> None:
    observed = signature(
        evaluate_final_test_prior
    )
    assert tuple(
        observed.parameters
    ) == _EXPECTED_PARAMETERS
def test_result_contract_is_exact() -> None:
    target, probability = (
        _valid_inputs()
    )
    result = evaluate_final_test_prior(
        target,
        probability,
    )
    assert tuple(
        result
    ) == _EXPECTED_RESULT_KEYS
    assert result[
        "selected_probability_model"
    ] == "calibration_prior"
    assert result[
        "evaluation_role"
    ] == (
        "one-time untouched test "
        "probability audit"
    )
    assert result[
        "test_rows"
    ] == 4
    assert result[
        "test_positives"
    ] == 2
    assert result[
        "test_negatives"
    ] == 2
    assert result[
        "test_prevalence"
    ] == 0.5
    assert result[
        "probability_value"
    ] == 0.5
    assert result[
        "unique_probability_values"
    ] == 1
    assert result[
        "ranking_available"
    ] is False
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
    ] is True
def test_declared_metrics_are_exact() -> None:
    target, probability = (
        _valid_inputs()
    )
    result = evaluate_final_test_prior(
        target,
        probability,
    )
    assert result[
        "average_precision"
    ] == pytest.approx(
        0.5,
        abs=1e-15,
    )
    assert result[
        "roc_auc"
    ] == pytest.approx(
        0.5,
        abs=1e-15,
    )
    assert result[
        "brier_score"
    ] == pytest.approx(
        0.25,
        abs=1e-15,
    )
    assert result[
        "log_loss"
    ] == pytest.approx(
        np.log(
            2.0
        ),
        abs=1e-15,
    )
def test_changed_target_changes_probability_metrics() -> None:
    target, probability = (
        _valid_inputs()
    )
    first = evaluate_final_test_prior(
        target,
        probability,
    )
    changed = target.copy(
        deep=True
    )
    changed.iloc[0] = 1
    second = evaluate_final_test_prior(
        changed,
        probability,
    )
    assert first[
        "test_prevalence"
    ] == 0.5
    assert second[
        "test_prevalence"
    ] == 0.75
    assert first[
        "average_precision"
    ] != second[
        "average_precision"
    ]
    assert first[
        "brier_score"
    ] == second[
        "brier_score"
    ]
def test_repeated_evaluation_is_exact() -> None:
    target, probability = (
        _valid_inputs()
    )
    first = evaluate_final_test_prior(
        target,
        probability,
    )
    second = evaluate_final_test_prior(
        target,
        probability,
    )
    assert first == second
def test_inputs_are_not_mutated() -> None:
    target, probability = (
        _valid_inputs()
    )
    target_snapshot = target.copy(
        deep=True
    )
    probability_snapshot = (
        probability.copy(
            deep=True
        )
    )
    evaluate_final_test_prior(
        target,
        probability,
    )
    pd.testing.assert_series_equal(
        target,
        target_snapshot,
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        probability,
        probability_snapshot,
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
def test_target_requires_exact_series(
    invalid: object,
) -> None:
    _, probability = (
        _valid_inputs()
    )
    with pytest.raises(
        TypeError,
        match=(
            "test_target must be an exact "
            "pandas Series"
        ),
    ):
        evaluate_final_test_prior(
            invalid,
            probability,
        )
@pytest.mark.parametrize(
    "invalid",
    (
        None,
        [],
        pd.DataFrame(
            {
                "probability": [
                    0.5,
                    0.5,
                ]
            }
        ),
        np.array(
            [0.5, 0.5]
        ),
    ),
)
def test_probability_requires_exact_series(
    invalid: object,
) -> None:
    target, _ = _valid_inputs()
    with pytest.raises(
        TypeError,
        match=(
            "test_probability must be an exact "
            "pandas Series"
        ),
    ):
        evaluate_final_test_prior(
            target,
            invalid,
        )
def test_empty_target_is_rejected() -> None:
    _, probability = (
        _valid_inputs()
    )
    target = pd.Series(
        [],
        dtype=np.int64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_target must not be empty"
        ),
    ):
        evaluate_final_test_prior(
            target,
            probability,
        )
def test_empty_probability_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    empty = probability.iloc[
        0:0
    ].copy(
        deep=True
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_probability must not be empty"
        ),
    ):
        evaluate_final_test_prior(
            target,
            empty,
        )
def test_duplicate_target_index_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = target.copy(
        deep=True
    )
    changed.index = [
        20,
        20,
        22,
        23,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "test_target index must be unique"
        ),
    ):
        evaluate_final_test_prior(
            changed,
            probability,
        )
def test_duplicate_probability_index_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.index = [
        20,
        20,
        22,
        23,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "test_probability index must be unique"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_missing_target_is_rejected() -> None:
    _, probability = (
        _valid_inputs()
    )
    target = pd.Series(
        [
            0,
            pd.NA,
            1,
        ],
        dtype="Int64",
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_target must not contain "
            "missing values"
        ),
    ):
        evaluate_final_test_prior(
            target,
            probability,
        )
def test_missing_probability_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.iloc[0] = np.nan
    with pytest.raises(
        ValueError,
        match=(
            "test_probability must not contain "
            "missing values"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
@pytest.mark.parametrize(
    "target",
    (
        pd.Series(
            [
                False,
                True,
            ],
            dtype=bool,
        ),
        pd.Series(
            [
                0.0,
                1.0,
            ],
            dtype=np.float64,
        ),
    ),
)
def test_target_requires_integer_dtype(
    target: pd.Series,
) -> None:
    probability = pd.Series(
        [
            0.5,
            0.5,
        ],
        index=target.index,
        name="no_show_probability",
        dtype=np.float64,
    )
    with pytest.raises(
        TypeError,
        match=(
            "test_target must have an "
            "integer dtype"
        ),
    ):
        evaluate_final_test_prior(
            target,
            probability,
        )
def test_nonbinary_target_is_rejected() -> None:
    target = pd.Series(
        [
            0,
            1,
            2,
        ],
        dtype=np.int64,
    )
    probability = pd.Series(
        [
            0.5,
            0.5,
            0.5,
        ],
        index=target.index,
        name="no_show_probability",
        dtype=np.float64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_target must contain "
            "only 0 and 1"
        ),
    ):
        evaluate_final_test_prior(
            target,
            probability,
        )
@pytest.mark.parametrize(
    "target",
    (
        pd.Series(
            [
                0,
                0,
            ],
            dtype=np.int64,
        ),
        pd.Series(
            [
                1,
                1,
            ],
            dtype=np.int64,
        ),
    ),
)
def test_target_requires_both_classes(
    target: pd.Series,
) -> None:
    probability = pd.Series(
        [
            0.5,
            0.5,
        ],
        index=target.index,
        name="no_show_probability",
        dtype=np.float64,
    )
    with pytest.raises(
        ValueError,
        match=(
            "test_target must contain "
            "both classes"
        ),
    ):
        evaluate_final_test_prior(
            target,
            probability,
        )
@pytest.mark.parametrize(
    "dtype",
    (
        np.int64,
        np.float32,
    ),
)
def test_probability_requires_float64_dtype(
    dtype: object,
) -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.astype(
        dtype
    )
    with pytest.raises(
        TypeError,
        match=(
            "test_probability must have "
            "float64 dtype"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
@pytest.mark.parametrize(
    "value",
    (
        -0.1,
        1.1,
    ),
)
def test_probability_range_is_enforced(
    value: float,
) -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.iloc[0] = value
    with pytest.raises(
        ValueError,
        match=(
            r"within \[0, 1\]"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_nonfinite_probability_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.iloc[0] = np.inf
    with pytest.raises(
        ValueError,
        match=(
            "only finite values"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_nonconstant_probability_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.iloc[0] = 0.4
    with pytest.raises(
        ValueError,
        match=(
            "exactly one unique value"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_probability_endpoints_are_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.iloc[:] = 0.0
    with pytest.raises(
        ValueError,
        match=(
            "strictly between 0 and 1"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_probability_name_is_enforced() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.rename(
        "probability"
    )
    with pytest.raises(
        ValueError,
        match=(
            "name must be "
            "no_show_probability"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_length_mismatch_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.iloc[
        :-1
    ].copy(
        deep=True
    )
    with pytest.raises(
        ValueError,
        match=(
            "must have equal length"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
def test_index_mismatch_is_rejected() -> None:
    target, probability = (
        _valid_inputs()
    )
    changed = probability.copy(
        deep=True
    )
    changed.index = [
        30,
        31,
        32,
        33,
    ]
    with pytest.raises(
        ValueError,
        match=(
            "indexes must align"
        ),
    ):
        evaluate_final_test_prior(
            target,
            changed,
        )
