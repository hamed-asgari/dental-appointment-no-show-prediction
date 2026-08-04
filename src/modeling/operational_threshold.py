"""Deterministic operational threshold-state analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
)
_SELECTED_PROBABILITY_MODEL = (
    "calibration_prior"
)
_THRESHOLD_RULE = (
    "probability >= threshold"
)
_POLICY_COLUMNS = (
    "policy",
    "threshold",
    "alerted_count",
    "alerted_rate",
    "tn",
    "fp",
    "fn",
    "tp",
)
_RESULT_KEYS = (
    "policy_table",
    "selected_probability_model",
    "probability_value",
    "threshold_rule",
    "distinct_policy_states",
)
def _validated_binary_target(
    target: pd.Series,
    *,
    name: str,
) -> pd.Series:
    if type(target) is not pd.Series:
        raise TypeError(
            f"{name} must be a pandas Series"
        )
    if target.empty:
        raise ValueError(
            f"{name} must not be empty"
        )
    if not target.index.is_unique:
        raise ValueError(
            f"{name} index must be unique"
        )
    if target.isna().any():
        raise ValueError(
            f"{name} must not contain missing values"
        )
    if (
        is_bool_dtype(target.dtype)
        or not is_integer_dtype(target.dtype)
    ):
        raise TypeError(
            f"{name} must have an integer dtype"
        )
    values = target.to_numpy(
        dtype=np.int64,
        copy=True,
    )
    if not np.isin(
        values,
        np.array([0, 1], dtype=np.int64),
    ).all():
        raise ValueError(
            f"{name} must contain only 0 and 1"
        )
    if not np.array_equal(
        np.unique(values),
        np.array([0, 1], dtype=np.int64),
    ):
        raise ValueError(
            f"{name} must contain both classes"
        )
    validated = target.astype(
        np.int64
    )
    return validated.copy(
        deep=True
    )
def evaluate_operational_threshold_states(
    calibration_target: pd.Series,
    validation_target: pd.Series,
) -> dict[
    str,
    pd.DataFrame | str | float | int,
]:
    """Enumerate the distinct threshold policies for the selected prior.
    The selected Phase 09 probability contract assigns the calibration
    prevalence to every validation observation. Under the fixed
    ``probability >= threshold`` rule, only two distinct prediction
    states can therefore exist: intervene on every observation or
    intervene on none.
    This function introduces no cost assumptions and selects neither a
    threshold nor an operational policy.
    """
    calibration = _validated_binary_target(
        calibration_target,
        name="calibration_target",
    )
    validation = _validated_binary_target(
        validation_target,
        name="validation_target",
    )
    if not calibration.index.intersection(
        validation.index
    ).empty:
        raise ValueError(
            "calibration_target and validation_target "
            "indexes must be disjoint"
        )
    probability_value = float(
        calibration.mean()
    )
    if not np.isfinite(
        probability_value
    ):
        raise RuntimeError(
            "calibration probability is not finite"
        )
    probability = np.full(
        len(validation),
        probability_value,
        dtype=np.float64,
    )
    policy_thresholds = (
        (
            "intervene_all",
            probability_value,
        ),
        (
            "intervene_none",
            float(
                np.nextafter(
                    probability_value,
                    np.inf,
                )
            ),
        ),
    )
    target = validation.to_numpy(
        dtype=np.int64,
        copy=True,
    )
    rows: list[
        dict[str, str | float | int]
    ] = []
    prediction_states: set[bytes] = set()
    for policy, threshold in policy_thresholds:
        prediction = (
            probability >= threshold
        ).astype(
            np.int64,
            copy=False,
        )
        prediction_states.add(
            prediction.tobytes()
        )
        negative = target == 0
        positive = target == 1
        predicted_negative = prediction == 0
        predicted_positive = prediction == 1
        tn = int(
            np.sum(
                negative
                & predicted_negative
            )
        )
        fp = int(
            np.sum(
                negative
                & predicted_positive
            )
        )
        fn = int(
            np.sum(
                positive
                & predicted_negative
            )
        )
        tp = int(
            np.sum(
                positive
                & predicted_positive
            )
        )
        alerted_count = int(
            prediction.sum()
        )
        rows.append(
            {
                "policy": policy,
                "threshold": threshold,
                "alerted_count": (
                    alerted_count
                ),
                "alerted_rate": float(
                    alerted_count
                    / len(prediction)
                ),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    policy_table = pd.DataFrame(
        rows,
        columns=_POLICY_COLUMNS,
    )
    if len(
        prediction_states
    ) != 2:
        raise RuntimeError(
            "selected probability contract did not "
            "produce exactly two policy states"
        )
    if tuple(
        policy_table["policy"]
    ) != (
        "intervene_all",
        "intervene_none",
    ):
        raise RuntimeError(
            "operational policy order is invalid"
        )
    result: dict[
        str,
        pd.DataFrame | str | float | int,
    ] = {
        "policy_table": policy_table,
        "selected_probability_model": (
            _SELECTED_PROBABILITY_MODEL
        ),
        "probability_value": (
            probability_value
        ),
        "threshold_rule": (
            _THRESHOLD_RULE
        ),
        "distinct_policy_states": int(
            len(prediction_states)
        ),
    }
    if tuple(result) != _RESULT_KEYS:
        raise RuntimeError(
            "operational threshold result "
            "contract is invalid"
        )
    return result
