"""Leakage-safe data contract for final pre-test prior fitting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
)
_REQUIRED_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "target",
    "split",
    "pretest_fit_eligible",
)
_ALLOWED_SPLITS = (
    "train",
    "validation",
    "test",
)
_PRETEST_SPLITS = (
    "train",
    "validation",
)
_TEST_METADATA_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
)
_RESULT_KEYS = (
    "pretest_target",
    "test_metadata",
)
def _validated_pretest_target(
    target: pd.Series,
) -> pd.Series:
    if type(target) is not pd.Series:
        raise TypeError(
            "pretest_target must be a pandas Series"
        )
    if target.empty:
        raise ValueError(
            "pretest_target must not be empty"
        )
    if not target.index.is_unique:
        raise ValueError(
            "pretest_target index must be unique"
        )
    if target.isna().any():
        raise ValueError(
            "pretest_target must not contain "
            "missing values"
        )
    if (
        is_bool_dtype(target.dtype)
        or not is_integer_dtype(target.dtype)
    ):
        raise TypeError(
            "pretest_target must have an "
            "integer dtype"
        )
    values = target.to_numpy(
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
    validated = target.astype(
        np.int64
    )
    return validated.copy(
        deep=True
    )
def build_final_pretest_prior_data(
    canonical: pd.DataFrame,
) -> dict[
    str,
    pd.Series | pd.DataFrame,
]:
    """Build the sealed final-fit contract without exposing test targets.
    The selected Phase 09 probability model is ``calibration_prior``.
    Its final pre-test fit therefore requires only the target values from
    rows approved by ``pretest_fit_eligible``.
    Test rows are exposed only through target-free metadata. Test target
    values are neither read nor validated by this function.
    """
    if type(canonical) is not pd.DataFrame:
        raise TypeError(
            "canonical must be a pandas DataFrame"
        )
    if canonical.empty:
        raise ValueError(
            "canonical must not be empty"
        )
    missing = [
        column
        for column in _REQUIRED_COLUMNS
        if column not in canonical.columns
    ]
    if missing:
        raise ValueError(
            "canonical is missing required "
            f"columns: {missing}"
        )
    if not canonical.index.is_unique:
        raise ValueError(
            "canonical index must be unique"
        )
    appointment_id = canonical[
        "appointment_id"
    ]
    if appointment_id.isna().any():
        raise ValueError(
            "appointment_id must not contain "
            "missing values"
        )
    if not appointment_id.is_unique:
        raise ValueError(
            "appointment_id must be unique"
        )
    prediction_time = canonical[
        "prediction_time"
    ]
    if not is_datetime64_any_dtype(
        prediction_time.dtype
    ):
        raise TypeError(
            "prediction_time must have a "
            "datetime dtype"
        )
    if prediction_time.isna().any():
        raise ValueError(
            "prediction_time must not contain "
            "missing values"
        )
    split = canonical[
        "split"
    ]
    if split.isna().any():
        raise ValueError(
            "split must not contain "
            "missing values"
        )
    observed_splits = set(
        split.unique()
    )
    if observed_splits != set(
        _ALLOWED_SPLITS
    ):
        raise ValueError(
            "split must contain exactly train, "
            "validation, and test"
        )
    eligibility = canonical[
        "pretest_fit_eligible"
    ]
    if not is_bool_dtype(
        eligibility.dtype
    ):
        raise TypeError(
            "pretest_fit_eligible must have "
            "a boolean dtype"
        )
    if eligibility.isna().any():
        raise ValueError(
            "pretest_fit_eligible must not "
            "contain missing values"
        )
    expected_pretest = split.isin(
        _PRETEST_SPLITS
    )
    observed_pretest = eligibility.eq(
        True
    )
    if not observed_pretest.equals(
        expected_pretest
    ):
        raise ValueError(
            "pretest_fit_eligible must identify "
            "all and only train and validation rows"
        )
    test_mask = split.eq(
        "test"
    )
    if not test_mask.any():
        raise ValueError(
            "test population must not be empty"
        )
    if not observed_pretest.any():
        raise ValueError(
            "pre-test fitting population "
            "must not be empty"
        )
    pretest_last = prediction_time.loc[
        observed_pretest
    ].max()
    test_first = prediction_time.loc[
        test_mask
    ].min()
    if pretest_last >= test_first:
        raise ValueError(
            "pre-test rows must occur strictly "
            "before test rows"
        )
    pretest_target = (
        canonical.loc[
            observed_pretest,
            "target",
        ]
        .copy(
            deep=True
        )
    )
    pretest_target = (
        _validated_pretest_target(
            pretest_target
        )
    )
    test_metadata = (
        canonical.loc[
            test_mask,
            list(
                _TEST_METADATA_COLUMNS
            ),
        ]
        .copy(
            deep=True
        )
    )
    if tuple(
        test_metadata.columns
    ) != _TEST_METADATA_COLUMNS:
        raise RuntimeError(
            "test metadata column contract "
            "is invalid"
        )
    if "target" in test_metadata.columns:
        raise RuntimeError(
            "test metadata must not expose target"
        )
    if not pretest_target.index.intersection(
        test_metadata.index
    ).empty:
        raise RuntimeError(
            "pre-test and test indexes overlap"
        )
    result: dict[
        str,
        pd.Series | pd.DataFrame,
    ] = {
        "pretest_target": pretest_target,
        "test_metadata": test_metadata,
    }
    if tuple(result) != _RESULT_KEYS:
        raise RuntimeError(
            "final pre-test result contract "
            "is invalid"
        )
    return result
