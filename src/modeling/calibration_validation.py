"""Evaluate chronological probability-calibration candidates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.calibration import (
    CalibratedClassifierCV,
)
from sklearn.pipeline import Pipeline
from src.modeling.calibration import (
    fit_probability_calibration_candidates,
)
from src.modeling.evaluation import (
    evaluate_binary_probabilities,
)
__all__ = (
    "evaluate_probability_calibration_validation",
)
_EXPECTED_MODELS = (
    "random_forest_uncalibrated",
    "random_forest_sigmoid",
    "random_forest_isotonic",
)
_RESULT_COLUMNS = (
    "model",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
    "mean_predicted_probability",
)
_PRIMARY_METRIC = "brier_score"
_SECONDARY_METRIC = "log_loss"
def _validate_validation_inputs(
    base_fit_features: pd.DataFrame,
    calibration_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
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
        (
            "validation_features",
            validation_features,
        ),
    ):
        if type(frame) is not pd.DataFrame:
            raise TypeError(
                f"{name} must be an exact "
                "pandas DataFrame"
            )
    if type(
        validation_target
    ) is not pd.Series:
        raise TypeError(
            "validation_target must be an "
            "exact pandas Series"
        )
    if validation_features.empty:
        raise ValueError(
            "validation_features must not be empty"
        )
    if validation_target.empty:
        raise ValueError(
            "validation_target must not be empty"
        )
    if validation_features.isna().any().any():
        raise ValueError(
            "validation_features must not "
            "contain missing values"
        )
    if validation_target.isna().any():
        raise ValueError(
            "validation_target must not "
            "contain missing values"
        )
    if set(
        validation_target.unique()
    ) != {0, 1}:
        raise ValueError(
            "validation_target values must be "
            "exactly 0 and 1"
        )
    if len(
        validation_features
    ) != len(
        validation_target
    ):
        raise ValueError(
            "validation features and target "
            "must have equal length"
        )
    if not validation_features.index.equals(
        validation_target.index
    ):
        raise ValueError(
            "validation feature and target "
            "indexes must align"
        )
    if tuple(
        validation_features.columns
    ) != tuple(
        base_fit_features.columns
    ):
        raise ValueError(
            "base-fit and validation feature "
            "columns must match"
        )
    if tuple(
        validation_features.columns
    ) != tuple(
        calibration_features.columns
    ):
        raise ValueError(
            "calibration and validation feature "
            "columns must match"
        )
    for population_name, index in (
        (
            "base-fit",
            base_fit_features.index,
        ),
        (
            "calibration",
            calibration_features.index,
        ),
    ):
        if not index.intersection(
            validation_features.index
        ).empty:
            raise ValueError(
                f"{population_name} and validation "
                "indexes must be disjoint"
            )
def _positive_probability(
    *,
    model_name: str,
    estimator: (
        Pipeline
        | CalibratedClassifierCV
    ),
    validation_features: pd.DataFrame,
) -> np.ndarray:
    if not np.array_equal(
        estimator.classes_,
        np.array([0, 1]),
    ):
        raise ValueError(
            f"{model_name} classes must be "
            "exactly 0 and 1"
        )
    class_probabilities = (
        estimator.predict_proba(
            validation_features
        )
    )
    if type(
        class_probabilities
    ) is not np.ndarray:
        raise TypeError(
            "predict_proba must return an "
            "exact NumPy ndarray"
        )
    if class_probabilities.shape != (
        len(validation_features),
        2,
    ):
        raise ValueError(
            "predict_proba must return one "
            "probability per class"
        )
    if not np.isfinite(
        class_probabilities
    ).all():
        raise ValueError(
            "predicted probabilities must be finite"
        )
    if (
        np.any(class_probabilities < 0.0)
        or np.any(class_probabilities > 1.0)
    ):
        raise ValueError(
            "predicted probabilities must be "
            "within [0, 1]"
        )
    if not np.allclose(
        class_probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "class probabilities must sum to one"
        )
    positive_index = int(
        np.flatnonzero(
            estimator.classes_ == 1
        )[0]
    )
    return class_probabilities[
        :,
        positive_index,
    ].astype(
        np.float64,
        copy=True,
    )
def evaluate_probability_calibration_validation(
    base_fit_features: pd.DataFrame,
    base_fit_target: pd.Series,
    calibration_features: pd.DataFrame,
    calibration_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> dict[
    str,
    pd.DataFrame | str | float,
]:
    """Fit candidates and evaluate untouched temporal validation."""
    _validate_validation_inputs(
        base_fit_features,
        calibration_features,
        validation_features,
        validation_target,
    )
    validation_features_copy = (
        validation_features.copy(
            deep=True
        )
    )
    validation_target_copy = (
        validation_target.copy(
            deep=True
        )
    )
    candidates = (
        fit_probability_calibration_candidates(
            base_fit_features,
            base_fit_target,
            calibration_features,
            calibration_target,
        )
    )
    if tuple(candidates) != _EXPECTED_MODELS:
        raise ValueError(
            "calibration candidate order is invalid"
        )
    rows: list[
        dict[str, float | str]
    ] = []
    for model_name, estimator in (
        candidates.items()
    ):
        positive_probability = (
            _positive_probability(
                model_name=model_name,
                estimator=estimator,
                validation_features=(
                    validation_features_copy
                ),
            )
        )
        probability_metrics = (
            evaluate_binary_probabilities(
                validation_target_copy,
                positive_probability,
            )
        )
        rows.append(
            {
                "model": model_name,
                "average_precision": (
                    probability_metrics[
                        "average_precision"
                    ]
                ),
                "roc_auc": (
                    probability_metrics[
                        "roc_auc"
                    ]
                ),
                "brier_score": (
                    probability_metrics[
                        "brier_score"
                    ]
                ),
                "log_loss": (
                    probability_metrics[
                        "log_loss"
                    ]
                ),
                "mean_predicted_probability": (
                    float(
                        positive_probability.mean()
                    )
                ),
            }
        )
    metrics = pd.DataFrame(
        rows,
        columns=_RESULT_COLUMNS,
    )
    ordered = metrics.assign(
        _declared_order=np.arange(
            len(metrics),
            dtype=np.int64,
        )
    ).sort_values(
        [
            _PRIMARY_METRIC,
            _SECONDARY_METRIC,
            "_declared_order",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    )
    selected_model = str(
        ordered.iloc[0]["model"]
    )
    return {
        "metrics": metrics,
        "primary_metric": _PRIMARY_METRIC,
        "secondary_metric": (
            _SECONDARY_METRIC
        ),
        "selected_model": selected_model,
        "validation_prevalence": float(
            validation_target_copy.mean()
        ),
    }
