"""Threshold-free evaluation of binary probability predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
__all__ = ("evaluate_binary_probabilities",)
def evaluate_binary_probabilities(
    target: pd.Series,
    positive_probability: np.ndarray,
) -> dict[str, float]:
    """Return ordered threshold-free metrics for binary probabilities."""
    if type(target) is not pd.Series:
        raise TypeError("target must be an exact pandas Series")
    if type(positive_probability) is not np.ndarray:
        raise TypeError(
            "positive_probability must be an exact NumPy ndarray"
        )
    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if positive_probability.ndim != 1:
        raise ValueError(
            "positive_probability must be one-dimensional"
        )
    if len(target) == 0:
        raise ValueError("evaluation inputs must not be empty")
    if len(target) != len(positive_probability):
        raise ValueError(
            "target and positive_probability must have equal length"
        )
    if target.isna().any():
        raise ValueError("target must not contain missing values")
    if set(target.unique()) != {0, 1}:
        raise ValueError("target values must be exactly 0 and 1")
    if not np.issubdtype(positive_probability.dtype, np.number):
        raise TypeError("positive_probability must be numerical")
    probability = positive_probability.astype(
        np.float64,
        copy=True,
    )
    target_values = target.to_numpy(
        dtype=np.int8,
        copy=True,
    )
    if not np.isfinite(probability).all():
        raise ValueError(
            "positive_probability must contain only finite values"
        )
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(
            "positive_probability values must be within [0, 1]"
        )
    class_probabilities = np.column_stack(
        (
            1.0 - probability,
            probability,
        )
    )
    return {
        "roc_auc": float(
            roc_auc_score(
                target_values,
                probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                target_values,
                probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                target_values,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                target_values,
                class_probabilities,
                labels=[0, 1],
            )
        ),
    }
