"""Deterministic final pre-test fit for the selected prior model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
)
_SELECTED_PROBABILITY_MODEL = (
    "calibration_prior"
)
_TEST_METADATA_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
)
_PROBABILITY_NAME = (
    "no_show_probability"
)
_RESULT_KEYS = (
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
def _validated_pretest_target(
    pretest_target: pd.Series,
) -> pd.Series:
    if type(pretest_target) is not pd.Series:
        raise TypeError(
            "pretest_target must be an exact "
            "pandas Series"
        )
    if pretest_target.empty:
        raise ValueError(
            "pretest_target must not be empty"
        )
    if not pretest_target.index.is_unique:
        raise ValueError(
            "pretest_target index must be unique"
        )
    if pretest_target.isna().any():
        raise ValueError(
            "pretest_target must not contain "
            "missing values"
        )
    if (
        is_bool_dtype(
            pretest_target.dtype
        )
        or not is_integer_dtype(
            pretest_target.dtype
        )
    ):
        raise TypeError(
            "pretest_target must have an "
            "integer dtype"
        )
    values = pretest_target.to_numpy(
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
            "pretest_target must contain "
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
            "pretest_target must contain "
            "both classes"
        )
    validated = pretest_target.astype(
        np.int64
    )
    return validated.copy(
        deep=True
    )
def _validated_test_metadata(
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    if type(test_metadata) is not pd.DataFrame:
        raise TypeError(
            "test_metadata must be an exact "
            "pandas DataFrame"
        )
    if test_metadata.empty:
        raise ValueError(
            "test_metadata must not be empty"
        )
    if tuple(
        test_metadata.columns
    ) != _TEST_METADATA_COLUMNS:
        raise ValueError(
            "test_metadata columns must be "
            "exactly appointment_id, "
            "prediction_time, and split"
        )
    if not test_metadata.index.is_unique:
        raise ValueError(
            "test_metadata index must be unique"
        )
    appointment_id = test_metadata[
        "appointment_id"
    ]
    if appointment_id.isna().any():
        raise ValueError(
            "test appointment_id must not "
            "contain missing values"
        )
    if not appointment_id.is_unique:
        raise ValueError(
            "test appointment_id must be unique"
        )
    prediction_time = test_metadata[
        "prediction_time"
    ]
    if not is_datetime64_any_dtype(
        prediction_time.dtype
    ):
        raise TypeError(
            "test prediction_time must have "
            "a datetime dtype"
        )
    if prediction_time.isna().any():
        raise ValueError(
            "test prediction_time must not "
            "contain missing values"
        )
    split = test_metadata[
        "split"
    ]
    if split.isna().any():
        raise ValueError(
            "test split must not contain "
            "missing values"
        )
    if not split.eq(
        "test"
    ).all():
        raise ValueError(
            "test_metadata must contain only "
            "test rows"
        )
    return test_metadata.copy(
        deep=True
    )
def fit_final_pretest_prior(
    pretest_target: pd.Series,
    test_metadata: pd.DataFrame,
) -> dict[
    str,
    str | float | int | bool | pd.Series,
]:
    """Fit the selected prior and seal target-free test probabilities.
    The selected probability model remains ``calibration_prior``.
    Its fitted value is the mean of the approved final pre-test target.
    Test metadata determines only the output index and row count. No test
    target, feature, threshold, policy, or model-comparison input is used.
    """
    target = _validated_pretest_target(
        pretest_target
    )
    metadata = _validated_test_metadata(
        test_metadata
    )
    if not target.index.intersection(
        metadata.index
    ).empty:
        raise ValueError(
            "pretest_target and test_metadata "
            "indexes must be disjoint"
        )
    fit_rows = int(
        len(target)
    )
    fit_positives = int(
        target.sum()
    )
    fit_negatives = int(
        fit_rows - fit_positives
    )
    probability_value = float(
        target.mean()
    )
    if not np.isfinite(
        probability_value
    ):
        raise RuntimeError(
            "fitted prior probability must "
            "be finite"
        )
    if not (
        0.0
        < probability_value
        < 1.0
    ):
        raise RuntimeError(
            "fitted prior probability must "
            "be strictly between 0 and 1"
        )
    probability_values = np.full(
        len(metadata),
        probability_value,
        dtype=np.float64,
    )
    test_probability = pd.Series(
        probability_values,
        index=metadata.index.copy(
            deep=True
        ),
        name=_PROBABILITY_NAME,
        dtype=np.float64,
    )
    if not test_probability.index.equals(
        metadata.index
    ):
        raise RuntimeError(
            "test probability index contract "
            "is invalid"
        )
    if not np.array_equal(
        test_probability.to_numpy(
            dtype=np.float64,
            copy=True,
        ),
        probability_values,
    ):
        raise RuntimeError(
            "test probability values are invalid"
        )
    result: dict[
        str,
        str | float | int | bool | pd.Series,
    ] = {
        "selected_probability_model": (
            _SELECTED_PROBABILITY_MODEL
        ),
        "probability_value": (
            probability_value
        ),
        "fit_population_rows": fit_rows,
        "fit_population_positives": (
            fit_positives
        ),
        "fit_population_negatives": (
            fit_negatives
        ),
        "test_rows": int(
            len(metadata)
        ),
        "probability_source": (
            "mean of pretest_target"
        ),
        "test_probability": (
            test_probability
        ),
        "model_comparison_reopened": False,
        "threshold_selected": False,
        "operational_policy_selected": False,
        "test_target_used": False,
    }
    if tuple(result) != _RESULT_KEYS:
        raise RuntimeError(
            "final pre-test fit result "
            "contract is invalid"
        )
    return result
