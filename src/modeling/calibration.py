"""Fit deterministic chronological probability-calibration candidates."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from src.modeling.comparison import (
    build_tree_comparison_estimators,
)
__all__ = (
    "fit_probability_calibration_candidates",
)
_EXPECTED_BASE_MODELS = (
    "random_forest_unweighted",
)
_EXPECTED_CANDIDATE_MODELS = (
    "random_forest_uncalibrated",
    "random_forest_sigmoid",
    "random_forest_isotonic",
)
_CALIBRATION_METHODS = (
    (
        "random_forest_sigmoid",
        "sigmoid",
    ),
    (
        "random_forest_isotonic",
        "isotonic",
    ),
)
def _validate_inputs(
    base_fit_features: pd.DataFrame,
    base_fit_target: pd.Series,
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
) -> None:
    for name, frame in (
        (
            "base_fit_features",
            base_fit_features,
        ),
        (
            "calibration_features",
            calibration_features,
        ),
    ):
        if type(frame) is not pd.DataFrame:
            raise TypeError(
                f"{name} must be an exact pandas DataFrame"
            )
        if frame.empty:
            raise ValueError(
                f"{name} must not be empty"
            )
        if frame.isna().any().any():
            raise ValueError(
                f"{name} must not contain missing values"
            )
    for name, target in (
        (
            "base_fit_target",
            base_fit_target,
        ),
        (
            "calibration_target",
            calibration_target,
        ),
    ):
        if type(target) is not pd.Series:
            raise TypeError(
                f"{name} must be an exact pandas Series"
            )
        if target.empty:
            raise ValueError(
                f"{name} must not be empty"
            )
        if target.isna().any():
            raise ValueError(
                f"{name} must not contain missing values"
            )
        if set(target.unique()) != {0, 1}:
            raise ValueError(
                f"{name} values must be exactly 0 and 1"
            )
    if len(
        base_fit_features
    ) != len(
        base_fit_target
    ):
        raise ValueError(
            "base-fit features and target must "
            "have equal length"
        )
    if len(
        calibration_features
    ) != len(
        calibration_target
    ):
        raise ValueError(
            "calibration features and target must "
            "have equal length"
        )
    if not base_fit_features.index.equals(
        base_fit_target.index
    ):
        raise ValueError(
            "base-fit feature and target indexes "
            "must align"
        )
    if not calibration_features.index.equals(
        calibration_target.index
    ):
        raise ValueError(
            "calibration feature and target indexes "
            "must align"
        )
    if tuple(
        base_fit_features.columns
    ) != tuple(
        calibration_features.columns
    ):
        raise ValueError(
            "base-fit and calibration feature "
            "columns must match"
        )
    if not base_fit_features.index.intersection(
        calibration_features.index
    ).empty:
        raise ValueError(
            "base-fit and calibration indexes "
            "must be disjoint"
        )
def _validate_fitted_classes(
    *,
    model_name: str,
    estimator: (
        Pipeline
        | CalibratedClassifierCV
    ),
) -> None:
    if not hasattr(
        estimator,
        "classes_",
    ):
        raise RuntimeError(
            f"{model_name} is not fitted"
        )
    if not np.array_equal(
        estimator.classes_,
        np.array([0, 1]),
    ):
        raise ValueError(
            f"{model_name} classes must be "
            "exactly 0 and 1"
        )
def fit_probability_calibration_candidates(
    base_fit_features: pd.DataFrame,
    base_fit_target: pd.Series,
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
) -> dict[
    str,
    Pipeline | CalibratedClassifierCV,
]:
    """Fit one raw forest and two frozen-estimator calibrations."""
    _validate_inputs(
        base_fit_features,
        base_fit_target,
        calibration_features,
        calibration_target,
    )
    base_fit_features_copy = (
        base_fit_features.copy(
            deep=True
        )
    )
    base_fit_target_copy = (
        base_fit_target.copy(
            deep=True
        )
    )
    calibration_features_copy = (
        calibration_features.copy(
            deep=True
        )
    )
    calibration_target_copy = (
        calibration_target.copy(
            deep=True
        )
    )
    suite = (
        build_tree_comparison_estimators()
    )
    if tuple(suite) != _EXPECTED_BASE_MODELS:
        raise ValueError(
            "base estimator order is invalid"
        )
    base_estimator = suite[
        "random_forest_unweighted"
    ]
    base_estimator.fit(
        base_fit_features_copy,
        base_fit_target_copy,
    )
    _validate_fitted_classes(
        model_name=(
            "random_forest_uncalibrated"
        ),
        estimator=base_estimator,
    )
    candidates: dict[
        str,
        Pipeline | CalibratedClassifierCV,
    ] = {
        "random_forest_uncalibrated": (
            deepcopy(base_estimator)
        ),
    }
    for (
        model_name,
        method,
    ) in _CALIBRATION_METHODS:
        calibrated = (
            CalibratedClassifierCV(
                estimator=FrozenEstimator(
                    deepcopy(
                        base_estimator
                    )
                ),
                method=method,
                cv=None,
                n_jobs=None,
                ensemble=False,
            )
        )
        calibrated.fit(
            calibration_features_copy,
            calibration_target_copy,
        )
        _validate_fitted_classes(
            model_name=model_name,
            estimator=calibrated,
        )
        candidates[
            model_name
        ] = calibrated
    if tuple(
        candidates
    ) != _EXPECTED_CANDIDATE_MODELS:
        raise RuntimeError(
            "calibration candidate order "
            "is invalid"
        )
    return candidates
