"""Pre-registered evaluation contract for the final prior model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
)
from src.modeling.evaluation import (
    evaluate_binary_probabilities,
)
_SELECTED_PROBABILITY_MODEL = (
    "calibration_prior"
)
_PROBABILITY_NAME = (
    "no_show_probability"
)
_EVALUATION_ROLE = (
    "one-time untouched test probability audit"
)
_EXPECTED_METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
)
_RESULT_KEYS = (
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
def _validated_test_target(
    test_target: pd.Series,
) -> pd.Series:
    if type(test_target) is not pd.Series:
        raise TypeError(
            "test_target must be an exact "
            "pandas Series"
        )
    if test_target.empty:
        raise ValueError(
            "test_target must not be empty"
        )
    if not test_target.index.is_unique:
        raise ValueError(
            "test_target index must be unique"
        )
    if test_target.isna().any():
        raise ValueError(
            "test_target must not contain "
            "missing values"
        )
    if (
        is_bool_dtype(
            test_target.dtype
        )
        or not is_integer_dtype(
            test_target.dtype
        )
    ):
        raise TypeError(
            "test_target must have an "
            "integer dtype"
        )
    values = test_target.to_numpy(
        dtype=np.int64,
        copy=True,
    )
    if not np.isin(
        values,
        np.array(
            [0, 1],
            dtype=np.int64,
        ),
    ).all():
        raise ValueError(
            "test_target must contain "
            "only 0 and 1"
        )
    if not np.array_equal(
        np.unique(values),
        np.array(
            [0, 1],
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "test_target must contain "
            "both classes"
        )
    validated = test_target.astype(
        np.int64
    )
    return validated.copy(
        deep=True
    )
def _validated_test_probability(
    test_probability: pd.Series,
) -> pd.Series:
    if type(test_probability) is not pd.Series:
        raise TypeError(
            "test_probability must be an exact "
            "pandas Series"
        )
    if test_probability.empty:
        raise ValueError(
            "test_probability must not be empty"
        )
    if not test_probability.index.is_unique:
        raise ValueError(
            "test_probability index must be unique"
        )
    if test_probability.name != _PROBABILITY_NAME:
        raise ValueError(
            "test_probability name must be "
            "no_show_probability"
        )
    if test_probability.dtype != np.dtype(
        "float64"
    ):
        raise TypeError(
            "test_probability must have "
            "float64 dtype"
        )
    if test_probability.isna().any():
        raise ValueError(
            "test_probability must not contain "
            "missing values"
        )
    values = test_probability.to_numpy(
        dtype=np.float64,
        copy=True,
    )
    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "test_probability must contain "
            "only finite values"
        )
    if (
        np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            "test_probability values must be "
            "within [0, 1]"
        )
    unique = np.unique(
        values
    )
    if len(unique) != 1:
        raise ValueError(
            "test_probability must contain "
            "exactly one unique value"
        )
    probability_value = float(
        unique[0]
    )
    if not (
        0.0
        < probability_value
        < 1.0
    ):
        raise ValueError(
            "the unique probability value must "
            "be strictly between 0 and 1"
        )
    return test_probability.copy(
        deep=True
    )
def evaluate_final_test_prior(
    test_target: pd.Series,
    test_probability: pd.Series,
) -> dict[
    str,
    str | float | int | bool,
]:
    """Evaluate the frozen final prior on the untouched test outcome.
    This contract is intended for one invocation after the final pre-test
    probability vector has been frozen. It evaluates probability quality
    only and does not reopen model, calibration, threshold, or policy
    selection.
    """
    target = _validated_test_target(
        test_target
    )
    probability = (
        _validated_test_probability(
            test_probability
        )
    )
    if len(target) != len(
        probability
    ):
        raise ValueError(
            "test_target and test_probability "
            "must have equal length"
        )
    if not target.index.equals(
        probability.index
    ):
        raise ValueError(
            "test_target and test_probability "
            "indexes must align"
        )
    probability_values = (
        probability.to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )
    metrics = evaluate_binary_probabilities(
        target,
        probability_values,
    )
    if tuple(metrics) != _EXPECTED_METRIC_KEYS:
        raise RuntimeError(
            "final test metric contract "
            "is invalid"
        )
    metric_values = np.array(
        [
            metrics[key]
            for key in _EXPECTED_METRIC_KEYS
        ],
        dtype=np.float64,
    )
    if not np.isfinite(
        metric_values
    ).all():
        raise RuntimeError(
            "final test metrics must be finite"
        )
    test_rows = int(
        len(target)
    )
    test_positives = int(
        target.sum()
    )
    test_negatives = int(
        test_rows - test_positives
    )
    test_prevalence = float(
        target.mean()
    )
    probability_value = float(
        probability_values[0]
    )
    result: dict[
        str,
        str | float | int | bool,
    ] = {
        "selected_probability_model": (
            _SELECTED_PROBABILITY_MODEL
        ),
        "evaluation_role": (
            _EVALUATION_ROLE
        ),
        "test_rows": test_rows,
        "test_positives": (
            test_positives
        ),
        "test_negatives": (
            test_negatives
        ),
        "test_prevalence": (
            test_prevalence
        ),
        "probability_value": (
            probability_value
        ),
        "unique_probability_values": 1,
        "average_precision": float(
            metrics["average_precision"]
        ),
        "roc_auc": float(
            metrics["roc_auc"]
        ),
        "brier_score": float(
            metrics["brier_score"]
        ),
        "log_loss": float(
            metrics["log_loss"]
        ),
        "ranking_available": False,
        "model_comparison_reopened": False,
        "threshold_selected": False,
        "operational_policy_selected": False,
        "test_target_used": True,
    }
    if tuple(result) != _RESULT_KEYS:
        raise RuntimeError(
            "final test result contract "
            "is invalid"
        )
    return result
