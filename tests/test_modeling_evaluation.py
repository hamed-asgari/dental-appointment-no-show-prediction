"""Tests for threshold-free binary probability evaluation."""
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
from sklearn.exceptions import ConvergenceWarning
from src.data import build_dataset as bd
from src.modeling.data import build_development_modeling_data
import src.modeling.evaluation as evaluation
from src.modeling.evaluation import evaluate_binary_probabilities
from src.modeling.estimators import build_baseline_estimators
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EVALUATION_PATH = _REPOSITORY_ROOT / "src" / "modeling" / "evaluation.py"
_MODELING_INIT_PATH = _REPOSITORY_ROOT / "src" / "modeling" / "__init__.py"
_EXPECTED_METRIC_NAMES = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
)
_EXPECTED_AUTHENTIC_METRICS = {
    "dummy_prior": {
        "roc_auc": 0.5,
        "average_precision": 0.1245944192083063,
        "brier_score": 0.109118029007233,
        "log_loss": 0.376204544533778,
    },
    "logistic_unweighted": {
        "roc_auc": 0.476027767482085,
        "average_precision": 0.120958549480471,
        "brier_score": 0.112110738382492,
        "log_loss": 0.396686219078739,
    },
    "logistic_balanced": {
        "roc_auc": 0.475865610328638,
        "average_precision": 0.120720303836034,
        "brier_score": 0.183289122207713,
        "log_loss": 0.555111620858532,
    },
}
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
    signature = inspect.signature(evaluate_binary_probabilities)
    hints = get_type_hints(evaluate_binary_probabilities)
    assert tuple(signature.parameters) == (
        "target",
        "positive_probability",
    )
    assert hints["target"] is pd.Series
    assert hints["positive_probability"] is np.ndarray
    assert hints["return"] == dict[str, float]
    assert evaluation.__all__ == (
        "evaluate_binary_probabilities",
    )
    namespace: dict[str, object] = {}
    exec(
        "from src.modeling.evaluation import *",
        namespace,
        namespace,
    )
    exported = {
        name
        for name in namespace
        if name != "__builtins__"
    }
    assert exported == {"evaluate_binary_probabilities"}
    tree = ast.parse(
        _EVALUATION_PATH.read_text(encoding="utf-8")
    )
    public_bindings = {
        name
        for name in _module_bindings(tree.body)
        if not name.startswith("_")
    }
    assert public_bindings == {
        "evaluate_binary_probabilities",
    }
    init_tree = ast.parse(
        _MODELING_INIT_PATH.read_text(encoding="utf-8")
    )
    assert len(init_tree.body) == 1
    assert isinstance(init_tree.body[0], ast.Expr)
    assert isinstance(init_tree.body[0].value, ast.Constant)
    assert isinstance(init_tree.body[0].value.value, str)
def test_perfect_predictions_have_exact_ordered_metrics() -> None:
    target = pd.Series(
        [0, 1, 0, 1],
        dtype="int8",
        name="target",
    )
    probability = np.array(
        [0.1, 0.9, 0.2, 0.8],
        dtype=np.float64,
    )
    target_snapshot = target.copy(deep=True)
    probability_snapshot = probability.copy()
    metrics = evaluate_binary_probabilities(
        target,
        probability,
    )
    assert type(metrics) is dict
    assert tuple(metrics) == _EXPECTED_METRIC_NAMES
    assert all(type(value) is float for value in metrics.values())
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    np.testing.assert_allclose(
        metrics["brier_score"],
        0.025,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        metrics["log_loss"],
        0.164252033486018,
        rtol=0.0,
        atol=1e-15,
    )
    pd.testing.assert_series_equal(target, target_snapshot)
    np.testing.assert_array_equal(
        probability,
        probability_snapshot,
    )
def test_constant_predictions_match_prevalence_identities() -> None:
    target = pd.Series(
        [0, 1, 0, 1],
        dtype="int8",
        name="target",
    )
    probability = np.full(
        shape=4,
        fill_value=0.25,
        dtype=np.float64,
    )
    metrics = evaluate_binary_probabilities(
        target,
        probability,
    )
    assert metrics["roc_auc"] == 0.5
    assert metrics["average_precision"] == 0.5
    np.testing.assert_allclose(
        metrics["brier_score"],
        0.3125,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        metrics["log_loss"],
        0.836988216785836,
        rtol=0.0,
        atol=1e-15,
    )
def test_authentic_baseline_validation_metrics_are_deterministic() -> None:
    raw_dir = _REPOSITORY_ROOT / "data" / "raw"
    bd.validate_raw_hashes(raw_dir)
    canonical = bd.build_analytical_dataset(
        bd.load_raw_data(raw_dir)
    )
    modeling_data = build_development_modeling_data(canonical)
    development_features = modeling_data["development_features"]
    development_target = modeling_data["development_target"]
    validation_features = modeling_data["validation_features"]
    validation_target = modeling_data["validation_target"]
    validation_snapshot = validation_target.copy(deep=True)
    assert development_features.shape == (3670, 10)
    assert int(development_target.sum()) == 432
    assert validation_features.shape == (1541, 10)
    assert int(validation_target.sum()) == 192
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", ConvergenceWarning)
        for name, pipeline in build_baseline_estimators().items():
            pipeline.fit(
                development_features,
                development_target,
            )
            classifier = pipeline.named_steps["classifier"]
            probabilities = pipeline.predict_proba(
                validation_features
            )
            np.testing.assert_array_equal(
                classifier.classes_,
                np.array([0, 1]),
            )
            positive_index = int(
                np.flatnonzero(classifier.classes_ == 1)[0]
            )
            positive_probability = probabilities[
                :,
                positive_index,
            ]
            probability_snapshot = positive_probability.copy()
            metrics = evaluate_binary_probabilities(
                validation_target,
                positive_probability,
            )
            assert tuple(metrics) == _EXPECTED_METRIC_NAMES
            for metric_name, expected_value in (
                _EXPECTED_AUTHENTIC_METRICS[name].items()
            ):
                np.testing.assert_allclose(
                    metrics[metric_name],
                    expected_value,
                    rtol=0.0,
                    atol=1e-12,
                )
            np.testing.assert_array_equal(
                positive_probability,
                probability_snapshot,
            )
    assert caught == []
    pd.testing.assert_series_equal(
        validation_target,
        validation_snapshot,
    )
@pytest.mark.parametrize(
    ("target", "probability", "message"),
    (
        (
            [0, 1],
            np.array([0.1, 0.9], dtype=np.float64),
            "target must be an exact pandas Series",
        ),
        (
            pd.Series([0, 1], dtype="int8"),
            [0.1, 0.9],
            "positive_probability must be an exact NumPy ndarray",
        ),
    ),
)
def test_nonexact_input_types_are_rejected(
    target: object,
    probability: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        evaluate_binary_probabilities(
            target,
            probability,
        )
@pytest.mark.parametrize(
    ("target", "probability", "message"),
    (
        (
            pd.Series([], dtype="int8"),
            np.array([], dtype=np.float64),
            "evaluation inputs must not be empty",
        ),
        (
            pd.Series([0, 1], dtype="int8"),
            np.array([0.5], dtype=np.float64),
            "must have equal length",
        ),
        (
            pd.Series([0, pd.NA], dtype="Int8"),
            np.array([0.1, 0.9], dtype=np.float64),
            "target must not contain missing values",
        ),
        (
            pd.Series([0, 0], dtype="int8"),
            np.array([0.1, 0.2], dtype=np.float64),
            "target values must be exactly 0 and 1",
        ),
        (
            pd.Series([0, 2], dtype="int8"),
            np.array([0.1, 0.9], dtype=np.float64),
            "target values must be exactly 0 and 1",
        ),
    ),
)
def test_invalid_targets_are_rejected(
    target: pd.Series,
    probability: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_binary_probabilities(
            target,
            probability,
        )
@pytest.mark.parametrize(
    ("probability", "exception", "message"),
    (
        (
            np.array([[0.1], [0.9]], dtype=np.float64),
            ValueError,
            "must be one-dimensional",
        ),
        (
            np.array(["low", "high"]),
            TypeError,
            "must be numerical",
        ),
        (
            np.array([np.nan, 0.9], dtype=np.float64),
            ValueError,
            "must contain only finite values",
        ),
        (
            np.array([np.inf, 0.9], dtype=np.float64),
            ValueError,
            "must contain only finite values",
        ),
        (
            np.array([-0.1, 0.9], dtype=np.float64),
            ValueError,
            r"must be within \[0, 1\]",
        ),
        (
            np.array([0.1, 1.1], dtype=np.float64),
            ValueError,
            r"must be within \[0, 1\]",
        ),
    ),
)
def test_invalid_probabilities_are_rejected(
    probability: np.ndarray,
    exception: type[Exception],
    message: str,
) -> None:
    target = pd.Series(
        [0, 1],
        dtype="int8",
        name="target",
    )
    with pytest.raises(exception, match=message):
        evaluate_binary_probabilities(
            target,
            probability,
        )
def test_each_call_returns_a_fresh_metric_dictionary() -> None:
    target = pd.Series(
        [0, 1, 0, 1],
        dtype="int8",
    )
    probability = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=np.float64,
    )
    first = evaluate_binary_probabilities(
        target,
        probability,
    )
    second = evaluate_binary_probabilities(
        target,
        probability,
    )
    assert first is not second
    assert first == second
    assert tuple(first) == _EXPECTED_METRIC_NAMES
    assert tuple(second) == _EXPECTED_METRIC_NAMES
def test_production_scope_is_metrics_only() -> None:
    tree = ast.parse(
        _EVALUATION_PATH.read_text(encoding="utf-8")
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
                called_attributes.add(node.func.attr)
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
    assert imported_modules == {
        "numpy",
        "pandas",
        "sklearn.metrics",
    }
    assert called_names.issubset(
        {
            "type",
            "TypeError",
            "ValueError",
            "len",
            "set",
            "float",
            "roc_auc_score",
            "average_precision_score",
            "brier_score_loss",
            "log_loss",
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
            "write_text",
            "write_bytes",
            "to_csv",
            "to_parquet",
            "read_csv",
            "read_parquet",
        }
    )
    assert referenced_names.isdisjoint(
        {
            "build_baseline_estimators",
            "build_development_modeling_data",
            "canonical",
            "development_features",
            "development_target",
            "validation_features",
            "validation_target",
            "test_features",
            "test_target",
            "split",
            "threshold",
            "confusion_matrix",
            "precision_score",
            "recall_score",
            "calibration_curve",
            "GridSearchCV",
            "RandomizedSearchCV",
            "joblib",
            "pickle",
            "print",
        }
    )
def test_import_evaluation_and_failures_have_no_process_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_cwd = Path.cwd()
    original_environment = dict(os.environ)
    original_filters = list(warnings.filters)
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    before_files = tuple(tmp_path.iterdir())
    target = pd.Series(
        [0, 1, 0, 1],
        dtype="int8",
    )
    probability = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=np.float64,
    )
    try:
        os.chdir(tmp_path)
        reloaded = importlib.reload(evaluation)
        metrics = reloaded.evaluate_binary_probabilities(
            target,
            probability,
        )
        assert tuple(metrics) == _EXPECTED_METRIC_NAMES
        with pytest.raises(ValueError):
            reloaded.evaluate_binary_probabilities(
                target,
                np.array(
                    [0.2, 0.8, np.nan, 0.7],
                    dtype=np.float64,
                ),
            )
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
