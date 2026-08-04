"""Deterministic baseline estimator configurations."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.modeling.preprocessing import build_preprocessor
__all__ = ("build_baseline_estimators",)
def build_baseline_estimators() -> dict[str, Pipeline]:
    """Return fresh, ordered, unfitted baseline estimator pipelines."""
    return {
        "dummy_prior": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    DummyClassifier(strategy="prior"),
                ),
            ]
        ),
        "logistic_unweighted": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1_000,
                        random_state=42,
                        class_weight=None,
                    ),
                ),
            ]
        ),
        "logistic_balanced": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1_000,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }
