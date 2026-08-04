"""Tests for deterministic tree-based comparison estimators."""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.data import build_dataset as bd
from src.modeling.comparison import (
    build_tree_comparison_estimators,
)
from src.modeling.data import build_development_modeling_data
from src.modeling.estimators import build_baseline_estimators
import src.modeling.comparison as comparison
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_COMPARISON_PATH = (
    _REPOSITORY_ROOT / "src" / "modeling" / "comparison.py"
)
_MODELING_INIT_PATH = (
    _REPOSITORY_ROOT / "src" / "modeling" / "__init__.py"
)
_EXPECTED_MODEL_NAMES = (
    "random_forest_unweighted",
)
_EXPECTED_PIPELINE_STEPS = (
    "preprocessor",
    "classifier",
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
def _module_bindings(
    statements: list[ast.stmt],
) -> set[str]:
    bindings: set[str] = set()
    for node in statements:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
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
        if (
            type_alias is not None
            and isinstance(node, type_alias)
        ):
            bindings.update(_target_names(node.name))
            continue
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
            ),
        ):
            bindings.update(
                _module_bindings(node.body)
            )
            bindings.update(
                _module_bindings(node.orelse)
            )
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            bindings.update(
                _module_bindings(node.body)
            )
            continue
        try_types = (ast.Try,)
        try_star = getattr(ast, "TryStar", None)
        if try_star is not None:
            try_types = (*try_types, try_star)
        if isinstance(node, try_types):
            bindings.update(
                _module_bindings(node.body)
            )
            bindings.update(
                _module_bindings(node.orelse)
            )
            bindings.update(
                _module_bindings(node.finalbody)
            )
            for handler in node.handlers:
                bindings.update(
                    _module_bindings(handler.body)
                )
            continue
        if isinstance(node, ast.Match):
            for case in node.cases:
                bindings.update(
                    _module_bindings(case.body)
                )
    return bindings
def _numpy_states_equal(
    first: tuple[object, ...],
    second: tuple[object, ...],
) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )
@pytest.fixture(scope="module")
def authentic_modeling_data(
) -> dict[str, pd.DataFrame | pd.Series]:
    raw_dir = _REPOSITORY_ROOT / "data" / "raw"
    bd.validate_raw_hashes(raw_dir)
    canonical = bd.build_analytical_dataset(
        bd.load_raw_data(raw_dir)
    )
    return build_development_modeling_data(
        canonical
    )
def test_public_api_and_module_surface_are_exact() -> None:
    signature = inspect.signature(
        build_tree_comparison_estimators
    )
    hints = get_type_hints(
        build_tree_comparison_estimators
    )
    assert tuple(signature.parameters) == ()
    assert hints["return"] == dict[str, Pipeline]
    assert comparison.__all__ == (
        "build_tree_comparison_estimators",
    )
    namespace: dict[str, object] = {}
    exec(
        "from src.modeling.comparison import *",
        namespace,
        namespace,
    )
    exported = {
        name
        for name in namespace
        if name != "__builtins__"
    }
    assert exported == {
        "build_tree_comparison_estimators",
    }
    tree = ast.parse(
        _COMPARISON_PATH.read_text(
            encoding="utf-8"
        )
    )
    public_bindings = {
        name
        for name in _module_bindings(tree.body)
        if not name.startswith("_")
    }
    assert public_bindings == {
        "build_tree_comparison_estimators",
    }
    init_tree = ast.parse(
        _MODELING_INIT_PATH.read_text(
            encoding="utf-8"
        )
    )
    assert len(init_tree.body) == 1
    assert isinstance(init_tree.body[0], ast.Expr)
    assert isinstance(
        init_tree.body[0].value,
        ast.Constant,
    )
    assert isinstance(
        init_tree.body[0].value.value,
        str,
    )
def test_builder_returns_exact_fresh_ordered_suite() -> None:
    first = build_tree_comparison_estimators()
    second = build_tree_comparison_estimators()
    assert type(first) is dict
    assert type(second) is dict
    assert tuple(first) == _EXPECTED_MODEL_NAMES
    assert tuple(second) == _EXPECTED_MODEL_NAMES
    assert first is not second
    first_pipeline = first[
        "random_forest_unweighted"
    ]
    second_pipeline = second[
        "random_forest_unweighted"
    ]
    assert type(first_pipeline) is Pipeline
    assert type(second_pipeline) is Pipeline
    assert first_pipeline is not second_pipeline
    assert tuple(
        first_pipeline.named_steps
    ) == _EXPECTED_PIPELINE_STEPS
    assert tuple(
        second_pipeline.named_steps
    ) == _EXPECTED_PIPELINE_STEPS
    assert (
        first_pipeline.named_steps["preprocessor"]
        is not second_pipeline.named_steps[
            "preprocessor"
        ]
    )
    assert (
        first_pipeline.named_steps["classifier"]
        is not second_pipeline.named_steps[
            "classifier"
        ]
    )
def test_random_forest_configuration_is_exact() -> None:
    pipeline = build_tree_comparison_estimators()[
        "random_forest_unweighted"
    ]
    classifier = pipeline.named_steps[
        "classifier"
    ]
    assert type(classifier) is RandomForestClassifier
    assert classifier.n_estimators == 500
    assert classifier.criterion == "gini"
    assert classifier.max_depth is None
    assert classifier.min_samples_split == 2
    assert classifier.min_samples_leaf == 1
    assert classifier.min_weight_fraction_leaf == 0.0
    assert classifier.max_features == "sqrt"
    assert classifier.max_leaf_nodes is None
    assert classifier.min_impurity_decrease == 0.0
    assert classifier.bootstrap is True
    assert classifier.oob_score is False
    assert classifier.n_jobs == 1
    assert classifier.random_state == 42
    assert classifier.verbose == 0
    assert classifier.warm_start is False
    assert classifier.class_weight is None
    assert classifier.ccp_alpha == 0.0
    assert classifier.max_samples is None
    assert classifier.monotonic_cst is None
    assert not hasattr(classifier, "estimators_")
def test_baseline_builder_contract_remains_unchanged() -> None:
    baseline_suite = build_baseline_estimators()
    assert tuple(baseline_suite) == (
        "dummy_prior",
        "logistic_unweighted",
        "logistic_balanced",
    )
    assert "random_forest_unweighted" not in (
        baseline_suite
    )
def test_authentic_fit_is_valid_and_does_not_mutate_inputs(
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
    validation_target = authentic_modeling_data[
        "validation_target"
    ]
    development_features_snapshot = (
        development_features.copy(deep=True)
    )
    development_target_snapshot = (
        development_target.copy(deep=True)
    )
    validation_features_snapshot = (
        validation_features.copy(deep=True)
    )
    validation_target_snapshot = (
        validation_target.copy(deep=True)
    )
    pipeline = build_tree_comparison_estimators()[
        "random_forest_unweighted"
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(
            development_features,
            development_target,
        )
        probabilities = pipeline.predict_proba(
            validation_features
        )
    assert caught == []
    classifier = pipeline.named_steps[
        "classifier"
    ]
    assert probabilities.shape == (1541, 2)
    assert probabilities.dtype == np.float64
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(1541),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        classifier.classes_,
        np.array([0, 1]),
    )
    assert len(classifier.estimators_) == 500
    assert classifier.n_features_in_ == 43
    transformed_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )
    assert transformed_names.shape == (43,)
    assert np.unique(transformed_names).size == 43
    pd.testing.assert_frame_equal(
        development_features,
        development_features_snapshot,
    )
    pd.testing.assert_series_equal(
        development_target,
        development_target_snapshot,
    )
    pd.testing.assert_frame_equal(
        validation_features,
        validation_features_snapshot,
    )
    pd.testing.assert_series_equal(
        validation_target,
        validation_target_snapshot,
    )
def test_repeated_authentic_fits_are_exactly_deterministic(
    authentic_modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    first = build_tree_comparison_estimators()[
        "random_forest_unweighted"
    ]
    second = build_tree_comparison_estimators()[
        "random_forest_unweighted"
    ]
    development_features = authentic_modeling_data[
        "development_features"
    ]
    development_target = authentic_modeling_data[
        "development_target"
    ]
    validation_features = authentic_modeling_data[
        "validation_features"
    ]
    first.fit(
        development_features,
        development_target,
    )
    second.fit(
        development_features,
        development_target,
    )
    first_probabilities = first.predict_proba(
        validation_features
    )
    second_probabilities = second.predict_proba(
        validation_features
    )
    np.testing.assert_array_equal(
        first_probabilities,
        second_probabilities,
    )
    first_classifier = first.named_steps[
        "classifier"
    ]
    second_classifier = second.named_steps[
        "classifier"
    ]
    np.testing.assert_array_equal(
        first_classifier.classes_,
        second_classifier.classes_,
    )
    np.testing.assert_array_equal(
        first_classifier.feature_importances_,
        second_classifier.feature_importances_,
    )
    assert len(first_classifier.estimators_) == 500
    assert len(second_classifier.estimators_) == 500
    for first_tree, second_tree in zip(
        first_classifier.estimators_,
        second_classifier.estimators_,
        strict=True,
    ):
        np.testing.assert_array_equal(
            first_tree.tree_.children_left,
            second_tree.tree_.children_left,
        )
        np.testing.assert_array_equal(
            first_tree.tree_.children_right,
            second_tree.tree_.children_right,
        )
        np.testing.assert_array_equal(
            first_tree.tree_.feature,
            second_tree.tree_.feature,
        )
        np.testing.assert_array_equal(
            first_tree.tree_.threshold,
            second_tree.tree_.threshold,
        )
        np.testing.assert_array_equal(
            first_tree.tree_.value,
            second_tree.tree_.value,
        )
def test_production_scope_excludes_search_and_side_effects() -> None:
    tree = ast.parse(
        _COMPARISON_PATH.read_text(
            encoding="utf-8"
        )
    )
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    called_attributes: set[str] = set()
    referenced_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_attributes.add(
                    node.func.attr
                )
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
    assert imported_modules == {
        "sklearn.ensemble",
        "sklearn.pipeline",
        "src.modeling.preprocessing",
    }
    assert called_names == {
        "Pipeline",
        "RandomForestClassifier",
        "build_preprocessor",
    }
    assert called_attributes.isdisjoint(
        {
            "fit",
            "predict",
            "predict_proba",
            "open",
            "dump",
            "save",
            "write_text",
            "write_bytes",
            "to_csv",
            "to_parquet",
        }
    )
    assert referenced_names.isdisjoint(
        {
            "GridSearchCV",
            "RandomizedSearchCV",
            "HalvingGridSearchCV",
            "HalvingRandomSearchCV",
            "cross_validate",
            "cross_val_score",
            "validation_target",
            "test_features",
            "test_target",
            "calibration_curve",
            "CalibratedClassifierCV",
            "joblib",
            "pickle",
        }
    )
def test_import_and_builder_have_no_process_side_effects(
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
        reloaded = importlib.reload(comparison)
        suite = (
            reloaded
            .build_tree_comparison_estimators()
        )
        assert tuple(suite) == _EXPECTED_MODEL_NAMES
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
