"""Tests for deterministic baseline estimator configurations."""
from __future__ import annotations
import ast
import importlib
import inspect
import os
from pathlib import Path
import random
from typing import get_type_hints
import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data import build_dataset as bd
from src.modeling.data import build_development_modeling_data
import src.modeling.estimators as estimators
from src.modeling.estimators import build_baseline_estimators
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ESTIMATORS_PATH = _REPOSITORY_ROOT / "src" / "modeling" / "estimators.py"
_MODELING_INIT_PATH = _REPOSITORY_ROOT / "src" / "modeling" / "__init__.py"
_EXPECTED_MODEL_NAMES = (
    "dummy_prior",
    "logistic_unweighted",
    "logistic_balanced",
)
_EXPECTED_PIPELINE_STEPS = (
    "preprocessor",
    "classifier",
)
@pytest.fixture(scope="module")
def authentic_canonical() -> pd.DataFrame:
    raw_dir = _REPOSITORY_ROOT / "data" / "raw"
    bd.validate_raw_hashes(raw_dir)
    return bd.build_analytical_dataset(bd.load_raw_data(raw_dir))
@pytest.fixture(scope="module")
def authentic_modeling_data(
    authentic_canonical: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    return build_development_modeling_data(authentic_canonical)
def _synthetic_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "planned_duration_min": pd.Series(
                [30, 45, 30, 60, 45, 30, 60, 45],
                dtype="int16",
            ),
            "visit_type": pd.Series(
                [
                    "consultation",
                    "treatment",
                    "consultation",
                    "treatment",
                    "consultation",
                    "treatment",
                    "consultation",
                    "treatment",
                ],
                dtype="string",
            ),
            "booking_channel": pd.Series(
                [
                    "online",
                    "phone",
                    "online",
                    "phone",
                    "online",
                    "phone",
                    "online",
                    "phone",
                ],
                dtype="string",
            ),
            "booking_lead_time_hours": pd.Series(
                [24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 192.0],
                dtype="float64",
            ),
            "scheduled_weekday": pd.Series(
                [0, 1, 2, 3, 5, 6, 0, 1],
                dtype="int8",
            ),
            "scheduled_hour": pd.Series(
                [9, 10, 11, 12, 14, 15, 16, 17],
                dtype="int8",
            ),
            "scheduled_month": pd.Series(
                [1, 2, 3, 4, 5, 6, 7, 8],
                dtype="int8",
            ),
            "approximate_age_at_prediction": pd.Series(
                [20, 30, 40, 50, 60, 35, 45, 55],
                dtype="int16",
            ),
            "patient_registration_tenure_days": pd.Series(
                [10, 20, 30, 40, 50, 60, 70, 80],
                dtype="int32",
            ),
            "dentist_tenure_days": pd.Series(
                [100, 200, 300, 400, 500, 600, 700, 800],
                dtype="int32",
            ),
        }
    )
def _synthetic_target() -> pd.Series:
    return pd.Series(
        [0, 1, 0, 0, 1, 0, 1, 0],
        dtype="int8",
        name="target",
    )
def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()
def _module_bindings(statements: list[ast.stmt]) -> set[str]:
    bindings: set[str] = set()
    for node in statements:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            bindings.add(node.name)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bindings.update(_target_names(target))
            continue
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            bindings.update(_target_names(node.target))
            continue
        type_alias = getattr(ast, "TypeAlias", None)
        if type_alias is not None and isinstance(node, type_alias):
            bindings.update(_target_names(node.name))
            continue
        if isinstance(
            node,
            (ast.If, ast.For, ast.AsyncFor, ast.While),
        ):
            bindings.update(_module_bindings(node.body))
            bindings.update(_module_bindings(node.orelse))
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            bindings.update(_module_bindings(node.body))
            continue
        try_types = (ast.Try,)
        try_star = getattr(ast, "TryStar", None)
        if try_star is not None:
            try_types = (*try_types, try_star)
        if isinstance(node, try_types):
            bindings.update(_module_bindings(node.body))
            bindings.update(_module_bindings(node.orelse))
            bindings.update(_module_bindings(node.finalbody))
            for handler in node.handlers:
                bindings.update(_module_bindings(handler.body))
            continue
        if isinstance(node, ast.Match):
            for case in node.cases:
                bindings.update(_module_bindings(case.body))
    return bindings
def _fit_suite(
    modeling_data: dict[str, pd.DataFrame | pd.Series],
) -> tuple[dict[str, Pipeline], dict[str, np.ndarray]]:
    suite = build_baseline_estimators()
    probabilities: dict[str, np.ndarray] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in _EXPECTED_MODEL_NAMES:
            pipeline = suite[name]
            pipeline.fit(
                modeling_data["development_features"],
                modeling_data["development_target"],
            )
            probabilities[name] = pipeline.predict_proba(
                modeling_data["validation_features"]
            )
    assert caught == []
    return suite, probabilities
def _assert_fitted_suites_equal(
    first: dict[str, Pipeline],
    second: dict[str, Pipeline],
    first_probabilities: dict[str, np.ndarray],
    second_probabilities: dict[str, np.ndarray],
) -> None:
    assert tuple(first) == _EXPECTED_MODEL_NAMES
    assert tuple(second) == _EXPECTED_MODEL_NAMES
    for name in _EXPECTED_MODEL_NAMES:
        first_classifier = first[name].named_steps["classifier"]
        second_classifier = second[name].named_steps["classifier"]
        np.testing.assert_array_equal(
            first_probabilities[name],
            second_probabilities[name],
        )
        np.testing.assert_array_equal(
            first_classifier.classes_,
            second_classifier.classes_,
        )
        first_names = (
            first[name]
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )
        second_names = (
            second[name]
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )
        np.testing.assert_array_equal(first_names, second_names)
        if name == "dummy_prior":
            np.testing.assert_array_equal(
                first_classifier.class_prior_,
                second_classifier.class_prior_,
            )
        else:
            np.testing.assert_array_equal(
                first_classifier.coef_,
                second_classifier.coef_,
            )
            np.testing.assert_array_equal(
                first_classifier.intercept_,
                second_classifier.intercept_,
            )
def _numpy_states_equal(
    first: tuple[object, ...],
    second: tuple[object, ...],
) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )
def test_public_api_and_module_surface_are_exact() -> None:
    signature = inspect.signature(build_baseline_estimators)
    hints = get_type_hints(build_baseline_estimators)
    assert tuple(signature.parameters) == ()
    assert hints["return"] == dict[str, Pipeline]
    assert estimators.__all__ == ("build_baseline_estimators",)
    namespace: dict[str, object] = {}
    exec(
        "from src.modeling.estimators import *",
        namespace,
        namespace,
    )
    exported = {
        name
        for name in namespace
        if name != "__builtins__"
    }
    assert exported == {"build_baseline_estimators"}
    tree = ast.parse(_ESTIMATORS_PATH.read_text(encoding="utf-8"))
    public_bindings = {
        name
        for name in _module_bindings(tree.body)
        if not name.startswith("_")
    }
    assert public_bindings == {"build_baseline_estimators"}
    init_tree = ast.parse(
        _MODELING_INIT_PATH.read_text(encoding="utf-8")
    )
    assert len(init_tree.body) == 1
    assert isinstance(init_tree.body[0], ast.Expr)
    assert isinstance(init_tree.body[0].value, ast.Constant)
    assert isinstance(init_tree.body[0].value.value, str)
def test_builder_returns_ordered_fresh_unfitted_estimators() -> None:
    first = build_baseline_estimators()
    second = build_baseline_estimators()
    assert type(first) is dict
    assert type(second) is dict
    assert first is not second
    assert tuple(first) == _EXPECTED_MODEL_NAMES
    assert tuple(second) == _EXPECTED_MODEL_NAMES
    for name in _EXPECTED_MODEL_NAMES:
        first_pipeline = first[name]
        second_pipeline = second[name]
        assert type(first_pipeline) is Pipeline
        assert type(second_pipeline) is Pipeline
        assert tuple(first_pipeline.named_steps) == _EXPECTED_PIPELINE_STEPS
        assert tuple(second_pipeline.named_steps) == _EXPECTED_PIPELINE_STEPS
        assert first_pipeline is not second_pipeline
        first_preprocessor = first_pipeline.named_steps["preprocessor"]
        second_preprocessor = second_pipeline.named_steps["preprocessor"]
        first_classifier = first_pipeline.named_steps["classifier"]
        second_classifier = second_pipeline.named_steps["classifier"]
        assert first_preprocessor is not second_preprocessor
        assert first_classifier is not second_classifier
        assert not hasattr(first_preprocessor, "transformers_")
        assert not hasattr(first_classifier, "classes_")
    preprocessors = [
        first[name].named_steps["preprocessor"]
        for name in _EXPECTED_MODEL_NAMES
    ]
    classifiers = [
        first[name].named_steps["classifier"]
        for name in _EXPECTED_MODEL_NAMES
    ]
    assert len({id(item) for item in preprocessors}) == 3
    assert len({id(item) for item in classifiers}) == 3
    dummy = first["dummy_prior"].named_steps["classifier"]
    assert type(dummy) is DummyClassifier
    assert dummy.strategy == "prior"
    assert dummy.random_state is None
    assert dummy.constant is None
    expected_weights = {
        "logistic_unweighted": None,
        "logistic_balanced": "balanced",
    }
    for name, expected_weight in expected_weights.items():
        classifier = first[name].named_steps["classifier"]
        assert type(classifier) is LogisticRegression
        assert classifier.solver == "liblinear"
        assert classifier.max_iter == 1_000
        assert classifier.random_state == 42
        assert classifier.class_weight == expected_weight
        assert classifier.penalty == "deprecated"
        assert classifier.C == 1.0
        assert classifier.fit_intercept is True
        assert classifier.tol == 0.0001
        assert classifier.dual is False
        assert classifier.intercept_scaling == 1
        assert classifier.l1_ratio == 0.0
        assert classifier.verbose == 0
        assert classifier.warm_start is False
        assert classifier.n_jobs is None
def test_authentic_fit_is_compatible_and_does_not_mutate_inputs(
    authentic_modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    development_features = authentic_modeling_data[
        "development_features"
    ]
    development_target = authentic_modeling_data[
        "development_target"
    ]
    validation_features = authentic_modeling_data[
        "validation_features"
    ]
    development_snapshot = development_features.copy(deep=True)
    target_snapshot = development_target.copy(deep=True)
    validation_snapshot = validation_features.copy(deep=True)
    assert development_features.shape == (3670, 10)
    assert int(development_target.sum()) == 432
    assert validation_features.shape == (1541, 10)
    suite, probabilities = _fit_suite(authentic_modeling_data)
    expected_prevalence = 432 / 3670
    for name in _EXPECTED_MODEL_NAMES:
        classifier = suite[name].named_steps["classifier"]
        probability = probabilities[name]
        np.testing.assert_array_equal(
            classifier.classes_,
            np.array([0, 1]),
        )
        assert probability.shape == (1541, 2)
        assert np.isfinite(probability).all()
        assert np.all(probability >= 0.0)
        assert np.all(probability <= 1.0)
        positive_index = int(
            np.flatnonzero(classifier.classes_ == 1)[0]
        )
        positive_probability = probability[:, positive_index]
        if name == "dummy_prior":
            np.testing.assert_allclose(
                positive_probability,
                expected_prevalence,
                rtol=0.0,
                atol=1e-15,
            )
            assert np.ptp(positive_probability) == 0.0
        else:
            assert classifier.coef_.shape == (1, 43)
            assert classifier.intercept_.shape == (1,)
            assert np.isfinite(classifier.coef_).all()
            assert np.isfinite(classifier.intercept_).all()
            assert np.ptp(positive_probability) > 0.0
    pd.testing.assert_frame_equal(
        development_features,
        development_snapshot,
    )
    pd.testing.assert_series_equal(
        development_target,
        target_snapshot,
    )
    pd.testing.assert_frame_equal(
        validation_features,
        validation_snapshot,
    )
def test_repeated_authentic_fits_are_exactly_deterministic(
    authentic_modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    first_suite, first_probabilities = _fit_suite(
        authentic_modeling_data
    )
    second_suite, second_probabilities = _fit_suite(
        authentic_modeling_data
    )
    _assert_fitted_suites_equal(
        first_suite,
        second_suite,
        first_probabilities,
        second_probabilities,
    )
def test_validation_target_poisoning_cannot_change_fitted_models(
    authentic_canonical: pd.DataFrame,
) -> None:
    original = build_development_modeling_data(
        authentic_canonical.copy(deep=True)
    )
    poisoned_canonical = authentic_canonical.copy(deep=True)
    validation_mask = poisoned_canonical["split"].eq("validation")
    poisoned_canonical.loc[validation_mask, "target"] = (
        1 - poisoned_canonical.loc[validation_mask, "target"]
    ).astype("int8")
    poisoned = build_development_modeling_data(poisoned_canonical)
    assert not original["validation_target"].equals(
        poisoned["validation_target"]
    )
    pd.testing.assert_frame_equal(
        original["development_features"],
        poisoned["development_features"],
    )
    pd.testing.assert_series_equal(
        original["development_target"],
        poisoned["development_target"],
    )
    pd.testing.assert_frame_equal(
        original["validation_features"],
        poisoned["validation_features"],
    )
    first_suite, first_probabilities = _fit_suite(original)
    second_suite, second_probabilities = _fit_suite(poisoned)
    _assert_fitted_suites_equal(
        first_suite,
        second_suite,
        first_probabilities,
        second_probabilities,
    )
@pytest.mark.parametrize(
    "protected_population",
    (
        "test",
        "immature_train",
    ),
)
def test_protected_target_poisoning_cannot_change_models(
    authentic_canonical: pd.DataFrame,
    protected_population: str,
) -> None:
    original = build_development_modeling_data(
        authentic_canonical.copy(deep=True)
    )
    poisoned_canonical = authentic_canonical.copy(deep=True)
    if protected_population == "test":
        mask = poisoned_canonical["split"].eq("test")
    else:
        mask = (
            poisoned_canonical["split"].eq("train")
            & ~poisoned_canonical["development_fit_eligible"]
        )
    assert mask.any()
    poisoned_canonical.loc[mask, "target"] = (
        1 - poisoned_canonical.loc[mask, "target"]
    ).astype("int8")
    poisoned = build_development_modeling_data(poisoned_canonical)
    pd.testing.assert_frame_equal(
        original["development_features"],
        poisoned["development_features"],
    )
    pd.testing.assert_series_equal(
        original["development_target"],
        poisoned["development_target"],
    )
    pd.testing.assert_frame_equal(
        original["validation_features"],
        poisoned["validation_features"],
    )
    pd.testing.assert_series_equal(
        original["validation_target"],
        poisoned["validation_target"],
    )
    first_suite, first_probabilities = _fit_suite(original)
    second_suite, second_probabilities = _fit_suite(poisoned)
    _assert_fitted_suites_equal(
        first_suite,
        second_suite,
        first_probabilities,
        second_probabilities,
    )
def test_production_scope_is_configuration_only() -> None:
    tree = ast.parse(_ESTIMATORS_PATH.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    called_attributes: set[str] = set()
    referenced_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_attributes.add(node.func.attr)
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
    assert imported_modules == {
        "sklearn.dummy",
        "sklearn.linear_model",
        "sklearn.pipeline",
        "src.modeling.preprocessing",
    }
    assert called_names.issubset(
        {
            "Pipeline",
            "DummyClassifier",
            "LogisticRegression",
            "build_preprocessor",
        }
    )
    assert called_attributes.isdisjoint(
        {
            "fit",
            "fit_transform",
            "transform",
            "predict",
            "predict_proba",
            "decision_function",
            "score",
            "open",
            "dump",
            "save",
            "to_csv",
            "to_parquet",
            "read_csv",
            "read_parquet",
            "print",
        }
    )
    assert referenced_names.isdisjoint(
        {
            "build_development_modeling_data",
            "canonical",
            "development_features",
            "development_target",
            "validation_features",
            "validation_target",
            "target",
            "split",
            "development_fit_eligible",
            "pretest_fit_eligible",
            "appointment_id",
            "patient_id",
            "dentist_id",
            "prediction_time",
            "DummyClassifierCV",
            "LogisticRegressionCV",
            "RandomForestClassifier",
            "GridSearchCV",
            "RandomizedSearchCV",
        }
    )
def test_import_builder_fit_and_prediction_have_no_process_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_cwd = Path.cwd()
    original_environment = dict(os.environ)
    original_filters = list(warnings.filters)
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    before_files = tuple(tmp_path.iterdir())
    features = _synthetic_features()
    target = _synthetic_target()
    try:
        os.chdir(tmp_path)
        reloaded = importlib.reload(estimators)
        suite = reloaded.build_baseline_estimators()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for pipeline in suite.values():
                pipeline.fit(features, target)
                probabilities = pipeline.predict_proba(features)
                assert probabilities.shape == (8, 2)
                assert np.isfinite(probabilities).all()
        assert caught == []
    finally:
        os.chdir(original_cwd)
    captured = capsys.readouterr()
    assert tuple(tmp_path.iterdir()) == before_files
    assert dict(os.environ) == original_environment
    assert warnings.filters == original_filters
    assert random.getstate() == original_python_state
    assert _numpy_states_equal(
        np.random.get_state(),
        original_numpy_state,
    )
    assert captured.out == ""
    assert captured.err == ""
