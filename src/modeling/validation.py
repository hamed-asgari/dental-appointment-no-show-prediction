"""Deterministic temporal-validation evaluation for baseline models."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.modeling.estimators import build_baseline_estimators
from src.modeling.evaluation import evaluate_binary_probabilities
__all__ = ("evaluate_baseline_validation",)
_FIXED_AUDIT_THRESHOLD = 0.5
_PRIMARY_METRIC = "average_precision"
_RESULT_COLUMNS = (
    "model",
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
    "precision_at_0_5",
    "recall_at_0_5",
    "f1_at_0_5",
    "true_negatives_at_0_5",
    "false_positives_at_0_5",
    "false_negatives_at_0_5",
    "true_positives_at_0_5",
    "accuracy_at_0_5_audit",
)
def _validate_inputs(
    development_features: pd.DataFrame,
    development_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> None:
    for name, frame in (
        ("development_features", development_features),
        ("validation_features", validation_features),
    ):
        if type(frame) is not pd.DataFrame:
            raise TypeError(
                f"{name} must be an exact pandas DataFrame"
            )
        if len(frame) == 0:
            raise ValueError(f"{name} must not be empty")
        if not frame.columns.is_unique:
            raise ValueError(
                f"{name} columns must be unique"
            )
    for name, target in (
        ("development_target", development_target),
        ("validation_target", validation_target),
    ):
        if type(target) is not pd.Series:
            raise TypeError(
                f"{name} must be an exact pandas Series"
            )
        if len(target) == 0:
            raise ValueError(f"{name} must not be empty")
        if target.isna().any():
            raise ValueError(
                f"{name} must not contain missing values"
            )
        if set(target.unique()) != {0, 1}:
            raise ValueError(
                f"{name} values must be exactly 0 and 1"
            )
    if len(development_features) != len(development_target):
        raise ValueError(
            "development features and target must have equal length"
        )
    if len(validation_features) != len(validation_target):
        raise ValueError(
            "validation features and target must have equal length"
        )
    if not development_features.index.equals(
        development_target.index
    ):
        raise ValueError(
            "development features and target indexes must align"
        )
    if not validation_features.index.equals(
        validation_target.index
    ):
        raise ValueError(
            "validation features and target indexes must align"
        )
    if tuple(development_features.columns) != tuple(
        validation_features.columns
    ):
        raise ValueError(
            "development and validation feature columns must match"
        )
def evaluate_baseline_validation(
    development_features: pd.DataFrame,
    development_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> dict[str, pd.DataFrame | str]:
    """Fit development baselines and evaluate temporal validation outcomes."""
    _validate_inputs(
        development_features,
        development_target,
        validation_features,
        validation_target,
    )
    development_features_copy = development_features.copy(
        deep=True
    )
    development_target_copy = development_target.copy(
        deep=True
    )
    validation_features_copy = validation_features.copy(
        deep=True
    )
    validation_target_copy = validation_target.copy(
        deep=True
    )
    rows: list[dict[str, float | int | str]] = []
    suite = build_baseline_estimators()
    for model_name, pipeline in suite.items():
        pipeline.fit(
            development_features_copy,
            development_target_copy,
        )
        classifier = pipeline.named_steps["classifier"]
        class_probabilities = pipeline.predict_proba(
            validation_features_copy
        )
        if not np.array_equal(
            classifier.classes_,
            np.array([0, 1]),
        ):
            raise ValueError(
                "baseline classifier classes must be exactly 0 and 1"
            )
        if class_probabilities.shape != (
            len(validation_target_copy),
            2,
        ):
            raise ValueError(
                "predict_proba must return one probability per class"
            )
        if not np.isfinite(class_probabilities).all():
            raise ValueError(
                "predicted probabilities must be finite"
            )
        if (
            np.any(class_probabilities < 0.0)
            or np.any(class_probabilities > 1.0)
        ):
            raise ValueError(
                "predicted probabilities must be within [0, 1]"
            )
        positive_index = int(
            np.flatnonzero(classifier.classes_ == 1)[0]
        )
        positive_probability = class_probabilities[
            :,
            positive_index,
        ]
        probability_metrics = evaluate_binary_probabilities(
            validation_target_copy,
            positive_probability,
        )
        fixed_prediction = (
            positive_probability >= _FIXED_AUDIT_THRESHOLD
        ).astype(np.int8)
        confusion = confusion_matrix(
            validation_target_copy,
            fixed_prediction,
            labels=[0, 1],
        )
        (
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
        ) = confusion.ravel()
        rows.append(
            {
                "model": model_name,
                "average_precision": probability_metrics[
                    "average_precision"
                ],
                "roc_auc": probability_metrics["roc_auc"],
                "log_loss": probability_metrics["log_loss"],
                "brier_score": probability_metrics[
                    "brier_score"
                ],
                "precision_at_0_5": float(
                    precision_score(
                        validation_target_copy,
                        fixed_prediction,
                        zero_division=0,
                    )
                ),
                "recall_at_0_5": float(
                    recall_score(
                        validation_target_copy,
                        fixed_prediction,
                        zero_division=0,
                    )
                ),
                "f1_at_0_5": float(
                    f1_score(
                        validation_target_copy,
                        fixed_prediction,
                        zero_division=0,
                    )
                ),
                "true_negatives_at_0_5": int(
                    true_negatives
                ),
                "false_positives_at_0_5": int(
                    false_positives
                ),
                "false_negatives_at_0_5": int(
                    false_negatives
                ),
                "true_positives_at_0_5": int(
                    true_positives
                ),
                "accuracy_at_0_5_audit": float(
                    accuracy_score(
                        validation_target_copy,
                        fixed_prediction,
                    )
                ),
            }
        )
    metrics = pd.DataFrame(
        rows,
        columns=_RESULT_COLUMNS,
    )
    selected_index = metrics[_PRIMARY_METRIC].idxmax()
    selected_model = str(
        metrics.loc[selected_index, "model"]
    )
    return {
        "metrics": metrics,
        "primary_metric": _PRIMARY_METRIC,
        "selected_model": selected_model,
    }
