"""Tests for the deterministic baseline preprocessing builder."""
from __future__ import annotations
import ast
import importlib
import inspect
import os
from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import build_dataset as bd
from src.modeling.data import build_development_modeling_data
import src.modeling.preprocessing as preprocessing
from src.modeling.preprocessing import build_preprocessor
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PREPROCESSING_PATH = (
    _REPOSITORY_ROOT / "src" / "modeling" / "preprocessing.py"
)
_NUMERIC_COLUMNS = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
_CATEGORICAL_COLUMNS = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
_FEATURE_COLUMNS = (
    "planned_duration_min",
    "visit_type",
    "booking_channel",
    "booking_lead_time_hours",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
_EXPECTED_CATEGORIES = (
    (
        "consultation",
        "emergency",
        "follow_up",
        "new_patient_examination",
        "recall_examination",
        "treatment",
    ),
    (
        "in_person",
        "online",
        "other",
        "phone",
        "referral",
    ),
    (0, 1, 2, 3, 5, 6),
    (9, 10, 11, 12, 14, 15, 16, 17, 18),
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
)
_EXPECTED_FEATURE_NAMES = (
    "numeric__planned_duration_min",
    "numeric__booking_lead_time_hours",
    "numeric__approximate_age_at_prediction",
    "numeric__patient_registration_tenure_days",
    "numeric__dentist_tenure_days",
    "categorical__visit_type_consultation",
    "categorical__visit_type_emergency",
    "categorical__visit_type_follow_up",
    "categorical__visit_type_new_patient_examination",
    "categorical__visit_type_recall_examination",
    "categorical__visit_type_treatment",
    "categorical__booking_channel_in_person",
    "categorical__booking_channel_online",
    "categorical__booking_channel_other",
    "categorical__booking_channel_phone",
    "categorical__booking_channel_referral",
    "categorical__scheduled_weekday_0",
    "categorical__scheduled_weekday_1",
    "categorical__scheduled_weekday_2",
    "categorical__scheduled_weekday_3",
    "categorical__scheduled_weekday_5",
    "categorical__scheduled_weekday_6",
    "categorical__scheduled_hour_9",
    "categorical__scheduled_hour_10",
    "categorical__scheduled_hour_11",
    "categorical__scheduled_hour_12",
    "categorical__scheduled_hour_14",
    "categorical__scheduled_hour_15",
    "categorical__scheduled_hour_16",
    "categorical__scheduled_hour_17",
    "categorical__scheduled_hour_18",
    "categorical__scheduled_month_1",
    "categorical__scheduled_month_2",
    "categorical__scheduled_month_3",
    "categorical__scheduled_month_4",
    "categorical__scheduled_month_5",
    "categorical__scheduled_month_6",
    "categorical__scheduled_month_7",
    "categorical__scheduled_month_8",
    "categorical__scheduled_month_9",
    "categorical__scheduled_month_10",
    "categorical__scheduled_month_11",
    "categorical__scheduled_month_12",
)
@pytest.fixture(scope="module")
def authentic_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = _REPOSITORY_ROOT / "data" / "raw"
    bd.validate_raw_hashes(raw_dir)
    canonical = bd.build_analytical_dataset(bd.load_raw_data(raw_dir))
    modeling_data = build_development_modeling_data(canonical)
    return (
        modeling_data["development_features"],
        modeling_data["validation_features"],
    )
def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "planned_duration_min": pd.Series(
                [30, 45, 30, 60],
                dtype="int16",
            ),
            "visit_type": pd.Series(
                [
                    "consultation",
                    "consultation",
                    "treatment",
                    "consultation",
                ],
                dtype="string",
            ),
            "booking_channel": pd.Series(
                ["online", "online", "phone", "online"],
                dtype="string",
            ),
            "booking_lead_time_hours": pd.Series(
                [24.0, 48.0, 72.0, 96.0],
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(
                [0, 1, 2, 3],
                dtype="int8",
            ),
            "scheduled_hour": pd.Series(
                [9, 10, 11, 12],
                dtype="int8",
            ),
            "scheduled_month": pd.Series(
                [1, 2, 3, 4],
                dtype="int8",
            ),
            "approximate_age_at_prediction": pd.Series(
                [20, 30, 40, 50],
                dtype="int16",
            ),
            "patient_registration_tenure_days": pd.Series(
                [10, 20, 30, 40],
                dtype="int32",
            ),
            "dentist_tenure_days": pd.Series(
                [100, 200, 300, 400],
                dtype="int32",
            ),
        },
        columns=list(_FEATURE_COLUMNS),
    )
def _assert_csr_equal(left: object, right: object) -> None:
    assert sparse.isspmatrix_csr(left)
    assert sparse.isspmatrix_csr(right)
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    np.testing.assert_array_equal(left.indptr, right.indptr)
    np.testing.assert_array_equal(left.indices, right.indices)
    np.testing.assert_array_equal(left.data, right.data)
def _numpy_random_states_equal(
    left: tuple[object, ...],
    right: tuple[object, ...],
) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )
def test_public_api_is_exact() -> None:
    signature = inspect.signature(build_preprocessor)
    assert tuple(signature.parameters) == ()
    assert signature.return_annotation is ColumnTransformer
    assert preprocessing.__all__ == ("build_preprocessor",)
    namespace: dict[str, object] = {}
    exec(
        "from src.modeling.preprocessing import *",
        namespace,
        namespace,
    )
    exported = {
        name
        for name in namespace
        if name != "__builtins__"
    }
    assert exported == {"build_preprocessor"}
def test_builder_returns_exact_fresh_unfitted_configuration() -> None:
    first = build_preprocessor()
    second = build_preprocessor()
    assert type(first) is ColumnTransformer
    assert first is not second
    assert not hasattr(first, "transformers_")
    assert not hasattr(first, "output_indices_")
    assert not hasattr(first, "n_features_in_")
    assert not hasattr(first, "feature_names_in_")
    assert first.remainder == "drop"
    assert first.sparse_threshold == 1.0
    assert first.n_jobs is None
    assert first.transformer_weights is None
    assert first.verbose_feature_names_out is True
    assert [item[0] for item in first.transformers] == [
        "numeric",
        "categorical",
    ]
    assert tuple(first.transformers[0][2]) == _NUMERIC_COLUMNS
    assert tuple(first.transformers[1][2]) == _CATEGORICAL_COLUMNS
    numeric = first.transformers[0][1]
    categorical = first.transformers[1][1]
    assert type(numeric) is Pipeline
    assert type(categorical) is Pipeline
    assert tuple(numeric.named_steps) == ("imputer", "scaler")
    assert tuple(categorical.named_steps) == ("imputer", "encoder")
    numeric_imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    categorical_imputer = categorical.named_steps["imputer"]
    encoder = categorical.named_steps["encoder"]
    assert type(numeric_imputer) is SimpleImputer
    assert numeric_imputer.strategy == "median"
    assert np.isnan(numeric_imputer.missing_values)
    assert type(scaler) is StandardScaler
    assert scaler.with_mean is True
    assert scaler.with_std is True
    assert type(categorical_imputer) is SimpleImputer
    assert categorical_imputer.strategy == "most_frequent"
    assert categorical_imputer.missing_values is pd.NA
    assert type(encoder) is OneHotEncoder
    assert encoder.handle_unknown == "ignore"
    assert encoder.sparse_output is True
    assert encoder.dtype is np.float64
    for index in (0, 1):
        assert first.transformers[index][1] is not second.transformers[index][1]
        assert first.transformers[index][2] is not second.transformers[index][2]
    for pipeline_index in (0, 1):
        first_pipeline = first.transformers[pipeline_index][1]
        second_pipeline = second.transformers[pipeline_index][1]
        for step_name in first_pipeline.named_steps:
            assert (
                first_pipeline.named_steps[step_name]
                is not second_pipeline.named_steps[step_name]
            )
def test_authentic_fit_is_sparse_auditable_and_development_only(
    authentic_features: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    development, validation = authentic_features
    development_snapshot = development.copy(deep=True)
    validation_snapshot = validation.copy(deep=True)
    preprocessor = build_preprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        transformed_development = preprocessor.fit_transform(development)
        transformed_validation = preprocessor.transform(validation)
    assert sparse.isspmatrix_csr(transformed_development)
    assert sparse.isspmatrix_csr(transformed_validation)
    assert transformed_development.shape == (3670, 43)
    assert transformed_validation.shape == (1541, 43)
    assert transformed_development.dtype == np.dtype("float64")
    assert transformed_validation.dtype == np.dtype("float64")
    assert np.isfinite(transformed_development.data).all()
    assert np.isfinite(transformed_validation.data).all()
    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    actual_categories = tuple(
        tuple(values.tolist())
        for values in encoder.categories_
    )
    assert actual_categories == _EXPECTED_CATEGORIES
    assert tuple(preprocessor.get_feature_names_out()) == (
        _EXPECTED_FEATURE_NAMES
    )
    pd.testing.assert_frame_equal(development, development_snapshot)
    pd.testing.assert_frame_equal(validation, validation_snapshot)
def test_unknown_category_is_ignored_without_changing_width() -> None:
    development = _synthetic_features()
    validation = development.iloc[[0]].copy(deep=True)
    validation.loc[:, "visit_type"] = "unseen_type"
    preprocessor = build_preprocessor()
    preprocessor.fit(development)
    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    categories_before = tuple(
        values.copy()
        for values in encoder.categories_
    )
    feature_names = preprocessor.get_feature_names_out()
    visit_type_indices = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("categorical__visit_type_")
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = preprocessor.transform(validation)
    assert caught == []
    assert sparse.isspmatrix_csr(transformed)
    assert transformed.dtype == np.dtype("float64")
    assert transformed.shape[1] == len(feature_names)
    dense = transformed.toarray()
    assert np.all(dense[0, visit_type_indices] == 0.0)
    for before, after in zip(categories_before, encoder.categories_):
        np.testing.assert_array_equal(before, after)
def test_missing_values_use_development_fitted_statistics() -> None:
    development = _synthetic_features().astype(
        {
            "planned_duration_min": "float64",
            "visit_type": "string",
        }
    )
    development.loc[1, "planned_duration_min"] = np.nan
    development.loc[2, "visit_type"] = pd.NA
    validation = development.iloc[[0]].copy(deep=True)
    validation.loc[:, "planned_duration_min"] = np.nan
    validation.loc[:, "visit_type"] = pd.NA
    preprocessor = build_preprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        development_output = preprocessor.fit_transform(development)
        validation_output = preprocessor.transform(validation)
    numeric_imputer = (
        preprocessor
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    categorical_imputer = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["imputer"]
    )
    assert numeric_imputer.statistics_[0] == 30.0
    assert categorical_imputer.statistics_[0] == "consultation"
    assert categorical_imputer.missing_values is pd.NA
    assert sparse.isspmatrix_csr(development_output)
    assert sparse.isspmatrix_csr(validation_output)
    assert np.isfinite(development_output.data).all()
    assert np.isfinite(validation_output.data).all()
def test_nullable_integer_categorical_missing_values_are_imputed() -> None:
    development = _synthetic_features().copy(deep=True)
    development["scheduled_weekday"] = pd.Series(
        [0, 1, pd.NA, 3],
        dtype="Int8",
    )
    validation = development.iloc[[0]].copy(deep=True)
    validation["scheduled_weekday"] = pd.Series(
        [pd.NA],
        index=validation.index,
        dtype="Int8",
    )
    development_snapshot = development.copy(deep=True)
    validation_snapshot = validation.copy(deep=True)
    preprocessor = build_preprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        development_output = preprocessor.fit_transform(development)
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["encoder"]
    statistics_before = categorical_imputer.statistics_.copy()
    categories_before = tuple(
        values.copy()
        for values in encoder.categories_
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validation_output = preprocessor.transform(validation)
    assert categorical_imputer.statistics_[2] == 0
    np.testing.assert_array_equal(
        encoder.categories_[2],
        np.array([0, 1, 3]),
    )
    np.testing.assert_array_equal(
        categorical_imputer.statistics_,
        statistics_before,
    )
    for before, after in zip(categories_before, encoder.categories_):
        np.testing.assert_array_equal(before, after)
    assert sparse.isspmatrix_csr(development_output)
    assert sparse.isspmatrix_csr(validation_output)
    assert development_output.dtype == np.dtype("float64")
    assert validation_output.dtype == np.dtype("float64")
    assert np.isfinite(development_output.data).all()
    assert np.isfinite(validation_output.data).all()
    pd.testing.assert_frame_equal(
        development,
        development_snapshot,
    )
    pd.testing.assert_frame_equal(
        validation,
        validation_snapshot,
    )

def test_repeated_authentic_fits_are_byte_deterministic(
    authentic_features: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    development, validation = authentic_features
    first = build_preprocessor()
    second = build_preprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first_development = first.fit_transform(development)
        first_validation = first.transform(validation)
        second_development = second.fit_transform(development)
        second_validation = second.transform(validation)
    _assert_csr_equal(first_development, second_development)
    _assert_csr_equal(first_validation, second_validation)
    assert tuple(first.get_feature_names_out()) == tuple(
        second.get_feature_names_out()
    )
    first_numeric = first.named_transformers_["numeric"]
    second_numeric = second.named_transformers_["numeric"]
    np.testing.assert_array_equal(
        first_numeric.named_steps["imputer"].statistics_,
        second_numeric.named_steps["imputer"].statistics_,
    )
    np.testing.assert_array_equal(
        first_numeric.named_steps["scaler"].mean_,
        second_numeric.named_steps["scaler"].mean_,
    )
    np.testing.assert_array_equal(
        first_numeric.named_steps["scaler"].scale_,
        second_numeric.named_steps["scaler"].scale_,
    )
    first_encoder = (
        first.named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    second_encoder = (
        second.named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    for first_values, second_values in zip(
        first_encoder.categories_,
        second_encoder.categories_,
    ):
        np.testing.assert_array_equal(first_values, second_values)
def test_import_reload_and_builder_have_no_process_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_cwd = Path.cwd()
    original_environment = dict(os.environ)
    original_filters = list(warnings.filters)
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    before_files = tuple(tmp_path.iterdir())
    try:
        os.chdir(tmp_path)
        reloaded = importlib.reload(preprocessing)
        built = reloaded.build_preprocessor()
        assert type(built) is ColumnTransformer
    finally:
        os.chdir(original_cwd)
    assert tuple(tmp_path.iterdir()) == before_files
    assert dict(os.environ) == original_environment
    assert warnings.filters == original_filters
    assert random.getstate() == original_python_state
    assert _numpy_random_states_equal(
        np.random.get_state(),
        original_numpy_state,
    )
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""
def test_production_scope_is_configuration_only() -> None:
    source = _PREPROCESSING_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    called_attributes: set[str] = set()
    referenced_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                referenced_names.add(node.func.id)
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
    assert imported_modules == {
        "numpy",
        "pandas",
        "sklearn.compose",
        "sklearn.impute",
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "src.modeling.data",
    }
    assert called_attributes.isdisjoint(
        {
            "fit",
            "fit_transform",
            "transform",
            "predict",
            "predict_proba",
            "decision_function",
            "score",
            "dump",
            "save",
            "to_csv",
            "to_parquet",
            "read_csv",
            "read_parquet",
            "open",
        }
    )
    assert referenced_names.isdisjoint(
        {
            "DummyClassifier",
            "LogisticRegression",
            "build_development_modeling_data",
            "development_features",
            "validation_features",
            "development_target",
            "validation_target",
            "target",
            "split",
            "pretest_fit_eligible",
            "development_fit_eligible",
            "appointment_id",
            "patient_id",
            "dentist_id",
            "prediction_time",
        }
    )
def test_dependency_pins_are_exact() -> None:
    runtime = (
        _REPOSITORY_ROOT / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()
    development = (
        _REPOSITORY_ROOT / "requirements-dev.txt"
    ).read_text(encoding="utf-8").splitlines()
    lock = (
        _REPOSITORY_ROOT / "requirements.lock.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert runtime.count("scikit-learn==1.9.0") == 1
    assert "-r requirements.txt" in development
    for requirement in (
        "scikit-learn==1.9.0",
        "scipy==1.18.0",
        "joblib==1.5.3",
        "narwhals==2.24.0",
        "threadpoolctl==3.6.0",
    ):
        assert lock.count(requirement) == 1
