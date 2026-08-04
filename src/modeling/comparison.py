"""Deterministic tree-based comparison estimator configuration."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.modeling.preprocessing import build_preprocessor
__all__ = ("build_tree_comparison_estimators",)
def build_tree_comparison_estimators() -> dict[str, Pipeline]:
    """Return fresh, ordered, unfitted tree-comparison pipelines."""
    return {
        "random_forest_unweighted": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        criterion="gini",
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        bootstrap=True,
                        class_weight=None,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
