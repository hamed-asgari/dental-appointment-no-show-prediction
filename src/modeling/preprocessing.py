"""Deterministic preprocessing configuration for baseline modeling."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.modeling.data import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
__all__ = ("build_preprocessor",)
def build_preprocessor() -> ColumnTransformer:
    """Return a fresh, unfitted preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    missing_values=pd.NA,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(NUMERIC_FEATURE_COLUMNS),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        n_jobs=None,
        transformer_weights=None,
        verbose_feature_names_out=True,
    )
