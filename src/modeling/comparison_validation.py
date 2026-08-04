"""Deterministic validation of baseline and tree comparison models."""
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
from src.modeling.comparison import (
    build_tree_comparison_estimators,
)
from src.modeling.evaluation import (
    evaluate_binary_probabilities,
)
from src.modeling.validation import (
    evaluate_baseline_validation,
)
__all__ = ("evaluate_tree_comparison_validation",)
_FIXED_AUDIT_THRESHOLD = 0.5
_PRIMARY_METRIC = "average_precision"
_EXPECTED_TREE_MODELS = (
    "random_forest_unweighted",
)
def evaluate_tree_comparison_validation(
    development_features: pd.DataFrame,
    development_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> dict[str, pd.DataFrame | str]:
    """Evaluate baselines and the fixed tree comparator on validation."""
    baseline_result = evaluate_baseline_validation(
        development_features,
        development_target,
        validation_features,
        validation_target,
    )
    baseline_metrics = baseline_result[
        "metrics"
    ].copy(deep=True)
    development_features_copy = (
        development_features.copy(deep=True)
    )
    development_target_copy = (
        development_target.copy(deep=True)
    )
    validation_features_copy = (
        validation_features.copy(deep=True)
    )
    validation_target_copy = (
        validation_target.copy(deep=True)
    )
    suite = build_tree_comparison_estimators()
    if tuple(suite) != _EXPECTED_TREE_MODELS:
        raise ValueError(
            "tree comparison estimator order is invalid"
        )
    rows: list[dict[str, float | int | str]] = []
    for model_name, pipeline in suite.items():
        pipeline.fit(
            development_features_copy,
            development_target_copy,
        )
        classifier = pipeline.named_steps[
            "classifier"
        ]
        class_probabilities = pipeline.predict_proba(
            validation_features_copy
        )
        if not np.array_equal(
            classifier.classes_,
            np.array([0, 1]),
        ):
            raise ValueError(
                "comparison classifier classes must be "
                "exactly 0 and 1"
            )
        if class_probabilities.shape != (
            len(validation_target_copy),
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
        positive_index = int(
            np.flatnonzero(
                classifier.classes_ == 1
            )[0]
        )
        positive_probability = (
            class_probabilities[
                :,
                positive_index,
            ]
        )
        probability_metrics = (
            evaluate_binary_probabilities(
                validation_target_copy,
                positive_probability,
            )
        )
        fixed_prediction = (
            positive_probability
            >= _FIXED_AUDIT_THRESHOLD
        ).astype(np.int8)
        (
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
        ) = confusion_matrix(
            validation_target_copy,
            fixed_prediction,
            labels=[0, 1],
        ).ravel()
        rows.append(
            {
                "model": model_name,
                "average_precision": (
                    probability_metrics[
                        "average_precision"
                    ]
                ),
                "roc_auc": (
                    probability_metrics["roc_auc"]
                ),
                "log_loss": (
                    probability_metrics["log_loss"]
                ),
                "brier_score": (
                    probability_metrics[
                        "brier_score"
                    ]
                ),
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
    comparison_metrics = pd.DataFrame(
        rows,
        columns=baseline_metrics.columns,
    )
    metrics = pd.concat(
        (
            baseline_metrics,
            comparison_metrics,
        ),
        ignore_index=True,
    )
    selected_index = metrics[
        _PRIMARY_METRIC
    ].idxmax()
    selected_model = str(
        metrics.loc[
            selected_index,
            "model",
        ]
    )
    return {
        "metrics": metrics,
        "primary_metric": _PRIMARY_METRIC,
        "selected_model": selected_model,
    }
