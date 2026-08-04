"""Tests for baseline and tree-comparison validation."""
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
import src.modeling.comparison_validation as comparison_validation
from src.modeling.comparison_validation import (
    evaluate_tree_comparison_validation,
)
from src.modeling.data import build_development_modeling_data
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _REPOSITORY_ROOT
    / "src"
    / "modeling"
    / "comparison_validation.py"
)
_MODELING_INIT_PATH = (
    _REPOSITORY_ROOT
    / "src"
    / "modeling"
    / "__init__.py"
)
_EXPECTED_RESULT_KEYS = (
    "metrics",
    "primary_metric",
    "selected_model",
)
_EXPECTED_MODEL_NAMES = (
    "dummy_prior",
    "logistic_unweighted",
    "logistic_balanced",
    "random_forest_unweighted",
)
_EXPECTED_COLUMNS = (
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
_EXPECTED_AUTHENTIC_ROWS = {
    "dummy_prior": {
        "average_precision": 0.1245944192083063,
        "roc_auc": 0.5,
        "log_loss": 0.376204544533778,
        "brier_score": 0.109118029007233,
        "precision_at_0_5": 0.0,
        "recall_at_0_5": 0.0,
        "f1_at_0_5": 0.0,
        "true_negatives_at_0_5": 1349,
        "false_positives_at_0_5": 0,
        "false_negatives_at_0_5": 192,
        "true_positives_at_0_5": 0,
        "accuracy_at_0_5_audit": 0.875405580791694,
    },
    "logistic_unweighted": {
        "average_precision": 0.120958549480471,
        "roc_auc": 0.476027767482085,
        "log_loss": 0.396686219078739,
        "brier_score": 0.112110738382492,
        "precision_at_0_5": 0.0,
        "recall_at_0_5": 0.0,
        "f1_at_0_5": 0.0,
        "true_negatives_at_0_5": 1349,
        "false_positives_at_0_5": 0,
        "false_negatives_at_0_5": 192,
        "true_positives_at_0_5": 0,
        "accuracy_at_0_5_audit": 0.875405580791694,
    },
    "logistic_balanced": {
        "average_precision": 0.120720303836034,
        "roc_auc": 0.475865610328638,
        "log_loss": 0.555111620858532,
        "brier_score": 0.183289122207713,
        "precision_at_0_5": 0.106194690265487,
        "recall_at_0_5": 0.0625,
        "f1_at_0_5": 0.078688524590164,
        "true_negatives_at_0_5": 1248,
        "false_positives_at_0_5": 101,
        "false_negatives_at_0_5": 180,
        "true_positives_at_0_5": 12,
        "accuracy_at_0_5_audit": 0.817650876054510,
    },
    "random_forest_unweighted": {
        "average_precision": 0.1330994923896915,
        "roc_auc": 0.509009374227823,
        "log_loss": 0.393564105145861,
        "brier_score": 0.114375987021415,
        "precision_at_0_5": 0.0,
        "recall_at_0_5": 0.0,
        "f1_at_0_5": 0.0,
        "true_negatives_at_0_5": 1349,
        "false_positives_at_0_5": 0,
        "false_negatives_at_0_5": 192,
        "true_positives_at_0_5": 0,
        "accuracy_at_0_5_audit": 0.875405580791694,
    },
}
class _StubClassifier:
    def __init__(
        self,
        classes: np.ndarray,
    ) -> None:
        self.classes_ = classes.copy()
class _StubPipeline:
    def __init__(
        self,
        probabilities: np.ndarray,
        *,
        classes: np.ndarray | None = None,
    ) -> None:
        if classes is None:
            classes = np.array([0, 1])
        self.named_steps = {
            "classifier": _StubClassifier(
                classes
            ),
        }
        self._probabilities = (
            probabilities.copy()
        )
        self.fit_features: (
            pd.DataFrame | None
        ) = None
        self.fit_target: pd.Series | None = None
        self.prediction_features: (
            pd.DataFrame | None
        ) = None
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> _StubPipeline:
        self.fit_features = features.copy(
            deep=True
        )
        self.fit_target = target.copy(
            deep=True
        )
        return self
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        self.prediction_features = (
            features.copy(deep=True)
        )
        return self._probabilities.copy()
def _target_names(
    target: ast.expr,
) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(
        target,
        (ast.Tuple, ast.List),
    ):
        names: set[str] = set()
        for item in target.elts:
            names.update(
                _target_names(item)
            )
        return names
    if isinstance(target, ast.Starred):
        return _target_names(
            target.value
        )
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
                bindings.update(
                    _target_names(target)
                )
            continue
        if isinstance(
            node,
            (ast.AnnAssign, ast.AugAssign),
        ):
            bindings.update(
                _target_names(node.target)
            )
            continue
        type_alias = getattr(
            ast,
            "TypeAlias",
            None,
        )
        if (
            type_alias is not None
            and isinstance(node, type_alias)
        ):
            bindings.update(
                _target_names(node.name)
            )
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
        if isinstance(
            node,
            (ast.With, ast.AsyncWith),
        ):
            bindings.update(
                _module_bindings(node.body)
            )
            continue
        try_types = (ast.Try,)
        try_star = getattr(
            ast,
            "TryStar",
            None,
        )
        if try_star is not None:
            try_types = (
                *try_types,
                try_star,
            )
        if isinstance(node, try_types):
            bindings.update(
                _module_bindings(node.body)
            )
            bindings.update(
                _module_bindings(node.orelse)
            )
            bindings.update(
                _module_bindings(
                    node.finalbody
                )
            )
            for handler in node.handlers:
                bindings.update(
                    _module_bindings(
                        handler.body
                    )
                )
            continue
        if isinstance(node, ast.Match):
            for case in node.cases:
                bindings.update(
                    _module_bindings(
                        case.body
                    )
                )
    return bindings
def _numpy_states_equal(
    first: tuple[object, ...],
    second: tuple[object, ...],
) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(
            first[1],
            second[1],
        )
        and first[2:] == second[2:]
    )
def _simple_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    development_features = pd.DataFrame(
        {
            "feature": [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        },
        index=pd.Index(
            [10, 11, 12, 13]
        ),
    )
    development_target = pd.Series(
        [0, 1, 0, 1],
        index=development_features.index,
        dtype="int8",
        name="target",
    )
    validation_features = pd.DataFrame(
        {
            "feature": [
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        },
        index=pd.Index(
            [20, 21, 22, 23]
        ),
    )
    validation_target = pd.Series(
        [0, 1, 0, 1],
        index=validation_features.index,
        dtype="int8",
        name="target",
    )
    return (
        development_features,
        development_target,
        validation_features,
        validation_target,
    )
def _baseline_metrics(
    *,
    dummy_average_precision: float = 0.4,
) -> pd.DataFrame:
    rows = []
    for position, model_name in enumerate(
        _EXPECTED_MODEL_NAMES[:3]
    ):
        average_precision = (
            dummy_average_precision
            - (position * 0.1)
        )
        rows.append(
            {
                "model": model_name,
                "average_precision": (
                    average_precision
                ),
                "roc_auc": 0.5,
                "log_loss": 0.5,
                "brier_score": 0.2,
                "precision_at_0_5": 0.0,
                "recall_at_0_5": 0.0,
                "f1_at_0_5": 0.0,
                "true_negatives_at_0_5": 2,
                "false_positives_at_0_5": 0,
                "false_negatives_at_0_5": 2,
                "true_positives_at_0_5": 0,
                "accuracy_at_0_5_audit": 0.5,
            }
        )
    return pd.DataFrame(
        rows,
        columns=_EXPECTED_COLUMNS,
    )
def _baseline_result(
    *,
    dummy_average_precision: float = 0.4,
) -> dict[str, pd.DataFrame | str]:
    return {
        "metrics": _baseline_metrics(
            dummy_average_precision=(
                dummy_average_precision
            )
        ),
        "primary_metric": (
            "average_precision"
        ),
        "selected_model": "dummy_prior",
    }
def _synthetic_modeling_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    features = pd.DataFrame(
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
            "booking_lead_time_hours": (
                pd.Series(
                    [
                        24.0,
                        48.0,
                        72.0,
                        96.0,
                        120.0,
                        144.0,
                        168.0,
                        192.0,
                    ],
                    dtype="float64",
                )
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
            "approximate_age_at_prediction": (
                pd.Series(
                    [
                        20,
                        30,
                        40,
                        50,
                        60,
                        35,
                        45,
                        55,
                    ],
                    dtype="int16",
                )
            ),
            "patient_registration_tenure_days": (
                pd.Series(
                    [
                        10,
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                    ],
                    dtype="int32",
                )
            ),
            "dentist_tenure_days": pd.Series(
                [
                    100,
                    200,
                    300,
                    400,
                    500,
                    600,
                    700,
                    800,
                ],
                dtype="int32",
            ),
        }
    )
    target = pd.Series(
        [0, 1, 0, 0, 1, 0, 1, 0],
        dtype="int8",
        name="target",
    )
    validation_features = features.iloc[
        [0, 1, 2, 3]
    ].copy(deep=True)
    validation_target = target.iloc[
        [0, 1, 2, 3]
    ].copy(deep=True)
    return (
        features,
        target,
        validation_features,
        validation_target,
    )
@pytest.fixture(scope="module")
def authentic_inputs() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    raw_dir = (
        _REPOSITORY_ROOT
        / "data"
        / "raw"
    )
    bd.validate_raw_hashes(raw_dir)
    canonical = bd.build_analytical_dataset(
        bd.load_raw_data(raw_dir)
    )
    modeling_data = (
        build_development_modeling_data(
            canonical
        )
    )
    return (
        modeling_data[
            "development_features"
        ],
        modeling_data[
            "development_target"
        ],
        modeling_data[
            "validation_features"
        ],
        modeling_data[
            "validation_target"
        ],
    )
def test_public_api_and_module_surface_are_exact(
) -> None:
    signature = inspect.signature(
        evaluate_tree_comparison_validation
    )
    hints = get_type_hints(
        evaluate_tree_comparison_validation
    )
    assert tuple(signature.parameters) == (
        "development_features",
        "development_target",
        "validation_features",
        "validation_target",
    )
    assert (
        hints["development_features"]
        is pd.DataFrame
    )
    assert (
        hints["development_target"]
        is pd.Series
    )
    assert (
        hints["validation_features"]
        is pd.DataFrame
    )
    assert (
        hints["validation_target"]
        is pd.Series
    )
    assert hints["return"] == dict[
        str,
        pd.DataFrame | str,
    ]
    assert (
        comparison_validation.__all__
        == (
            "evaluate_tree_comparison_validation",
        )
    )
    namespace: dict[str, object] = {}
    exec(
        (
            "from "
            "src.modeling.comparison_validation "
            "import *"
        ),
        namespace,
        namespace,
    )
    exported = {
        name
        for name in namespace
        if name != "__builtins__"
    }
    assert exported == {
        "evaluate_tree_comparison_validation",
    }
    tree = ast.parse(
        _MODULE_PATH.read_text(
            encoding="utf-8"
        )
    )
    public_bindings = {
        name
        for name in _module_bindings(
            tree.body
        )
        if not name.startswith("_")
    }
    assert public_bindings == {
        "evaluate_tree_comparison_validation",
    }
    init_tree = ast.parse(
        _MODELING_INIT_PATH.read_text(
            encoding="utf-8"
        )
    )
    assert len(init_tree.body) == 1
    assert isinstance(
        init_tree.body[0],
        ast.Expr,
    )
    assert isinstance(
        init_tree.body[0].value,
        ast.Constant,
    )
    assert isinstance(
        init_tree.body[0].value.value,
        str,
    )
def test_authentic_validation_result_is_exact(
    authentic_inputs: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    snapshots = (
        authentic_inputs[0].copy(
            deep=True
        ),
        authentic_inputs[1].copy(
            deep=True
        ),
        authentic_inputs[2].copy(
            deep=True
        ),
        authentic_inputs[3].copy(
            deep=True
        ),
    )
    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "error",
            RuntimeWarning,
        )
        warnings.simplefilter(
            "error",
            UserWarning,
        )
        warnings.simplefilter(
            "error",
            ConvergenceWarning,
        )
        result = (
            evaluate_tree_comparison_validation(
                *authentic_inputs
            )
        )
    assert caught == []
    assert type(result) is dict
    assert tuple(result) == (
        _EXPECTED_RESULT_KEYS
    )
    assert result["primary_metric"] == (
        "average_precision"
    )
    assert result["selected_model"] == (
        "random_forest_unweighted"
    )
    metrics = result["metrics"]
    assert type(metrics) is pd.DataFrame
    assert metrics.shape == (4, 13)
    assert tuple(metrics.columns) == (
        _EXPECTED_COLUMNS
    )
    assert tuple(metrics["model"]) == (
        _EXPECTED_MODEL_NAMES
    )
    for model_name in (
        _EXPECTED_MODEL_NAMES
    ):
        row = metrics.loc[
            metrics["model"].eq(
                model_name
            )
        ].iloc[0]
        expected = (
            _EXPECTED_AUTHENTIC_ROWS[
                model_name
            ]
        )
        for metric_name, expected_value in (
            expected.items()
        ):
            actual_value = row[
                metric_name
            ]
            if isinstance(
                expected_value,
                int,
            ):
                assert int(
                    actual_value
                ) == expected_value
            else:
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=0.0,
                    atol=1e-12,
                )
    pd.testing.assert_frame_equal(
        authentic_inputs[0],
        snapshots[0],
    )
    pd.testing.assert_series_equal(
        authentic_inputs[1],
        snapshots[1],
    )
    pd.testing.assert_frame_equal(
        authentic_inputs[2],
        snapshots[2],
    )
    pd.testing.assert_series_equal(
        authentic_inputs[3],
        snapshots[3],
    )
def test_repeated_authentic_evaluation_is_exact(
    authentic_inputs: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    first = (
        evaluate_tree_comparison_validation(
            *authentic_inputs
        )
    )
    second = (
        evaluate_tree_comparison_validation(
            *authentic_inputs
        )
    )
    assert first is not second
    assert (
        first["metrics"]
        is not second["metrics"]
    )
    assert (
        first["primary_metric"]
        == second["primary_metric"]
    )
    assert (
        first["selected_model"]
        == second["selected_model"]
    )
    pd.testing.assert_frame_equal(
        first["metrics"],
        second["metrics"],
        check_exact=True,
    )
def test_baseline_result_is_copied_and_inputs_are_delegated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _simple_inputs()
    baseline_result = _baseline_result()
    baseline_snapshot = baseline_result[
        "metrics"
    ].copy(deep=True)
    received: tuple[object, ...] | None = None
    def fake_baseline(
        development_features: pd.DataFrame,
        development_target: pd.Series,
        validation_features: pd.DataFrame,
        validation_target: pd.Series,
    ) -> dict[str, pd.DataFrame | str]:
        nonlocal received
        received = (
            development_features,
            development_target,
            validation_features,
            validation_target,
        )
        return baseline_result
    positive_probability = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=np.float64,
    )
    probabilities = np.column_stack(
        (
            1.0 - positive_probability,
            positive_probability,
        )
    )
    monkeypatch.setattr(
        comparison_validation,
        "evaluate_baseline_validation",
        fake_baseline,
    )
    monkeypatch.setattr(
        comparison_validation,
        "build_tree_comparison_estimators",
        lambda: {
            "random_forest_unweighted": (
                _StubPipeline(
                    probabilities
                )
            )
        },
    )
    result = (
        comparison_validation
        .evaluate_tree_comparison_validation(
            *inputs
        )
    )
    assert received is not None
    for actual, expected in zip(
        received,
        inputs,
        strict=True,
    ):
        assert actual is expected
    pd.testing.assert_frame_equal(
        baseline_result["metrics"],
        baseline_snapshot,
    )
    assert (
        result["metrics"]
        is not baseline_result["metrics"]
    )
    assert result["metrics"].shape == (
        4,
        13,
    )
def test_fixed_threshold_uses_greater_than_or_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _simple_inputs()
    positive_probability = np.array(
        [0.49, 0.50, 0.60, 0.10],
        dtype=np.float64,
    )
    probabilities = np.column_stack(
        (
            1.0 - positive_probability,
            positive_probability,
        )
    )
    monkeypatch.setattr(
        comparison_validation,
        "evaluate_baseline_validation",
        lambda *args: _baseline_result(),
    )
    monkeypatch.setattr(
        comparison_validation,
        "build_tree_comparison_estimators",
        lambda: {
            "random_forest_unweighted": (
                _StubPipeline(
                    probabilities
                )
            )
        },
    )
    result = (
        comparison_validation
        .evaluate_tree_comparison_validation(
            *inputs
        )
    )
    row = result["metrics"].loc[
        result["metrics"]["model"].eq(
            "random_forest_unweighted"
        )
    ].iloc[0]
    assert row[
        "true_negatives_at_0_5"
    ] == 1
    assert row[
        "false_positives_at_0_5"
    ] == 1
    assert row[
        "false_negatives_at_0_5"
    ] == 1
    assert row[
        "true_positives_at_0_5"
    ] == 1
    assert row[
        "precision_at_0_5"
    ] == 0.5
    assert row[
        "recall_at_0_5"
    ] == 0.5
    assert row[
        "f1_at_0_5"
    ] == 0.5
    assert row[
        "accuracy_at_0_5_audit"
    ] == 0.5
def test_primary_metric_tie_uses_declared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _simple_inputs()
    positive_probability = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=np.float64,
    )
    probabilities = np.column_stack(
        (
            1.0 - positive_probability,
            positive_probability,
        )
    )
    monkeypatch.setattr(
        comparison_validation,
        "evaluate_baseline_validation",
        lambda *args: _baseline_result(
            dummy_average_precision=1.0
        ),
    )
    monkeypatch.setattr(
        comparison_validation,
        "build_tree_comparison_estimators",
        lambda: {
            "random_forest_unweighted": (
                _StubPipeline(
                    probabilities
                )
            )
        },
    )
    result = (
        comparison_validation
        .evaluate_tree_comparison_validation(
            *inputs
        )
    )
    assert result["selected_model"] == (
        "dummy_prior"
    )
def test_tree_fit_and_prediction_use_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _simple_inputs()
    positive_probability = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=np.float64,
    )
    probabilities = np.column_stack(
        (
            1.0 - positive_probability,
            positive_probability,
        )
    )
    stub = _StubPipeline(
        probabilities
    )
    monkeypatch.setattr(
        comparison_validation,
        "evaluate_baseline_validation",
        lambda *args: _baseline_result(),
    )
    monkeypatch.setattr(
        comparison_validation,
        "build_tree_comparison_estimators",
        lambda: {
            "random_forest_unweighted": stub
        },
    )
    (
        comparison_validation
        .evaluate_tree_comparison_validation(
            *inputs
        )
    )
    assert stub.fit_features is not (
        inputs[0]
    )
    assert stub.fit_target is not (
        inputs[1]
    )
    assert (
        stub.prediction_features
        is not inputs[2]
    )
    pd.testing.assert_frame_equal(
        stub.fit_features,
        inputs[0],
    )
    pd.testing.assert_series_equal(
        stub.fit_target,
        inputs[1],
    )
    pd.testing.assert_frame_equal(
        stub.prediction_features,
        inputs[2],
    )
@pytest.mark.parametrize(
    ("position", "replacement", "message"),
    (
        (
            0,
            {"feature": [1, 2, 3, 4]},
            (
                "development_features must "
                "be an exact pandas DataFrame"
            ),
        ),
        (
            1,
            [0, 1, 0, 1],
            (
                "development_target must "
                "be an exact pandas Series"
            ),
        ),
        (
            2,
            {"feature": [5, 6, 7, 8]},
            (
                "validation_features must "
                "be an exact pandas DataFrame"
            ),
        ),
        (
            3,
            [0, 1, 0, 1],
            (
                "validation_target must "
                "be an exact pandas Series"
            ),
        ),
    ),
)
def test_invalid_input_types_are_rejected(
    position: int,
    replacement: object,
    message: str,
) -> None:
    inputs = list(_simple_inputs())
    inputs[position] = replacement
    with pytest.raises(
        TypeError,
        match=message,
    ):
        evaluate_tree_comparison_validation(
            *inputs
        )
def test_invalid_tree_suite_order_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _simple_inputs()
    positive_probability = np.array(
        [0.2, 0.8, 0.3, 0.7],
        dtype=np.float64,
    )
    probabilities = np.column_stack(
        (
            1.0 - positive_probability,
            positive_probability,
        )
    )
    monkeypatch.setattr(
        comparison_validation,
        "evaluate_baseline_validation",
        lambda *args: _baseline_result(),
    )
    monkeypatch.setattr(
        comparison_validation,
        "build_tree_comparison_estimators",
        lambda: {
            "unexpected_tree_model": (
                _StubPipeline(
                    probabilities
                )
            )
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            "tree comparison estimator "
            "order is invalid"
        ),
    ):
        (
            comparison_validation
            .evaluate_tree_comparison_validation(
                *inputs
            )
        )
@pytest.mark.parametrize(
    ("classes", "probabilities", "message"),
    (
        (
            np.array([1, 0]),
            np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.3, 0.7],
                ]
            ),
            (
                "comparison classifier classes "
                "must be exactly 0 and 1"
            ),
        ),
        (
            np.array([0, 1]),
            np.array(
                [
                    [0.8],
                    [0.2],
                    [0.7],
                    [0.3],
                ]
            ),
            (
                "predict_proba must return "
                "one probability per class"
            ),
        ),
        (
            np.array([0, 1]),
            np.array(
                [
                    [0.8, 0.2],
                    [np.nan, np.nan],
                    [0.7, 0.3],
                    [0.3, 0.7],
                ]
            ),
            (
                "predicted probabilities "
                "must be finite"
            ),
        ),
        (
            np.array([0, 1]),
            np.array(
                [
                    [0.8, 0.2],
                    [1.1, -0.1],
                    [0.7, 0.3],
                    [0.3, 0.7],
                ]
            ),
            (
                r"predicted probabilities "
                r"must be within \[0, 1\]"
            ),
        ),
    ),
)
def test_invalid_tree_outputs_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    classes: np.ndarray,
    probabilities: np.ndarray,
    message: str,
) -> None:
    inputs = _simple_inputs()
    monkeypatch.setattr(
        comparison_validation,
        "evaluate_baseline_validation",
        lambda *args: _baseline_result(),
    )
    monkeypatch.setattr(
        comparison_validation,
        "build_tree_comparison_estimators",
        lambda: {
            "random_forest_unweighted": (
                _StubPipeline(
                    probabilities,
                    classes=classes,
                )
            )
        },
    )
    with pytest.raises(
        ValueError,
        match=message,
    ):
        (
            comparison_validation
            .evaluate_tree_comparison_validation(
                *inputs
            )
        )
def test_production_scope_preserves_boundaries(
) -> None:
    tree = ast.parse(
        _MODULE_PATH.read_text(
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
        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module is not None:
                imported_modules.add(
                    node.module
                )
        elif isinstance(node, ast.Call):
            if isinstance(
                node.func,
                ast.Name,
            ):
                called_names.add(
                    node.func.id
                )
            elif isinstance(
                node.func,
                ast.Attribute,
            ):
                called_attributes.add(
                    node.func.attr
                )
        elif isinstance(node, ast.Name):
            referenced_names.add(
                node.id
            )
    assert imported_modules == {
        "__future__",
        "numpy",
        "pandas",
        "sklearn.metrics",
        "src.modeling.comparison",
        "src.modeling.evaluation",
        "src.modeling.validation",
    }
    assert {
        "fit",
        "predict_proba",
    }.issubset(
        called_attributes
    )
    assert called_attributes.isdisjoint(
        {
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
    assert called_names.isdisjoint(
        {
            "open",
            "print",
            "GridSearchCV",
            "RandomizedSearchCV",
            "cross_val_score",
            "cross_validate",
            "calibration_curve",
        }
    )
    assert referenced_names.isdisjoint(
        {
            "canonical",
            "test_features",
            "test_target",
            "pretest_features",
            "pretest_target",
            "prediction_time",
            "appointment_id",
            "patient_id",
            "dentist_id",
            "GridSearchCV",
            "RandomizedSearchCV",
            "CalibratedClassifierCV",
            "calibration_curve",
            "joblib",
            "pickle",
        }
    )
    threshold_assignments = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.Assign,
        )
        and any(
            isinstance(
                target,
                ast.Name,
            )
            and target.id
            == "_FIXED_AUDIT_THRESHOLD"
            for target in node.targets
        )
    ]
    assert len(
        threshold_assignments
    ) == 1
    threshold_value = (
        threshold_assignments[0].value
    )
    assert isinstance(
        threshold_value,
        ast.Constant,
    )
    assert threshold_value.value == 0.5
def test_import_evaluation_and_failure_have_no_side_effects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_cwd = Path.cwd()
    original_environment = dict(
        os.environ
    )
    original_filters = list(
        warnings.filters
    )
    original_python_state = (
        random.getstate()
    )
    original_numpy_state = (
        np.random.get_state()
    )
    before_files = tuple(
        tmp_path.iterdir()
    )
    inputs = (
        _synthetic_modeling_inputs()
    )
    try:
        os.chdir(tmp_path)
        reloaded = importlib.reload(
            comparison_validation
        )
        with warnings.catch_warnings(
            record=True
        ) as caught:
            warnings.simplefilter(
                "error",
                RuntimeWarning,
            )
            warnings.simplefilter(
                "error",
                UserWarning,
            )
            warnings.simplefilter(
                "error",
                ConvergenceWarning,
            )
            result = (
                reloaded
                .evaluate_tree_comparison_validation(
                    *inputs
                )
            )
        assert caught == []
        assert tuple(result) == (
            _EXPECTED_RESULT_KEYS
        )
        invalid_inputs = list(inputs)
        invalid_inputs[3] = (
            invalid_inputs[3].iloc[:-1]
        )
        with pytest.raises(
            ValueError
        ):
            (
                reloaded
                .evaluate_tree_comparison_validation(
                    *invalid_inputs
                )
            )
    finally:
        os.chdir(original_cwd)
    captured = capsys.readouterr()
    assert tuple(
        tmp_path.iterdir()
    ) == before_files
    assert dict(
        os.environ
    ) == original_environment
    assert warnings.filters == (
        original_filters
    )
    assert random.getstate() == (
        original_python_state
    )
    assert _numpy_states_equal(
        np.random.get_state(),
        original_numpy_state,
    )
    assert captured.out == ""
    assert captured.err == ""
