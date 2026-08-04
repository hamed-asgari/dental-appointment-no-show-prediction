"""Tests for chronological calibration validation."""
from __future__ import annotations
from inspect import signature
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import pytest
from src.data import build_dataset as bd
from src.modeling import calibration_validation
from src.modeling.calibration_data import (
    build_calibration_modeling_data,
)
from src.modeling.calibration_validation import (
    evaluate_probability_calibration_validation,
)
_EXPECTED_MODELS = (
    "calibration_prior",
    "random_forest_uncalibrated",
    "random_forest_sigmoid",
    "random_forest_isotonic",
)
_EXPECTED_COLUMNS = (
    "model",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
    "mean_predicted_probability",
)
_EXPECTED_RESULT_KEYS = (
    "metrics",
    "primary_metric",
    "secondary_metric",
    "selected_model",
    "validation_prevalence",
)
_EXPECTED_PARAMETERS = (
    "base_fit_features",
    "base_fit_target",
    "calibration_features",
    "calibration_target",
    "validation_features",
    "validation_target",
)
_EXPECTED_METRICS = {
    "calibration_prior": {
        "average_precision": (
            0.124594419208306
        ),
        "roc_auc": (
            0.500000000000000
        ),
        "brier_score": (
            0.109071038004684
        ),
        "log_loss": (
            0.375981958827441
        ),
        "mean_predicted_probability": (
            0.125217391304348
        ),
    },
    "random_forest_uncalibrated": {
        "average_precision": (
            0.123083748814041
        ),
        "roc_auc": (
            0.507136845193971
        ),
        "brier_score": (
            0.118919854639844
        ),
        "log_loss": (
            0.404800545985675
        ),
        "mean_predicted_probability": (
            0.205155094094744
        ),
    },
    "random_forest_sigmoid": {
        "average_precision": (
            0.118851128074777
        ),
        "roc_auc": (
            0.492863154806029
        ),
        "brier_score": (
            0.109573699567836
        ),
        "log_loss": (
            0.378404516262467
        ),
        "mean_predicted_probability": (
            0.112533249819596
        ),
    },
    "random_forest_isotonic": {
        "average_precision": (
            0.123552124857081
        ),
        "roc_auc": (
            0.495621756856931
        ),
        "brier_score": (
            0.109101376880843
        ),
        "log_loss": (
            0.376115519200288
        ),
        "mean_predicted_probability": (
            0.125798722629989
        ),
    },
}
class _FakeEstimator:
    def __init__(
        self,
        positive_probability: float,
    ) -> None:
        self.classes_ = np.array(
            [0, 1]
        )
        self._positive_probability = (
            positive_probability
        )
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        positive = np.full(
            len(features),
            self._positive_probability,
            dtype=np.float64,
        )
        return np.column_stack(
            (
                1.0 - positive,
                positive,
            )
        )
@pytest.fixture(scope="module")
def canonical_dataset() -> pd.DataFrame:
    raw_dir = Path("data/raw")
    bd.validate_raw_hashes(raw_dir)
    return bd.build_analytical_dataset(
        bd.load_raw_data(raw_dir)
    )
@pytest.fixture(scope="module")
def modeling_data(
    canonical_dataset: pd.DataFrame,
) -> dict[
    str,
    pd.DataFrame | pd.Series,
]:
    return build_calibration_modeling_data(
        canonical_dataset
    )
@pytest.fixture(scope="module")
def authentic_result(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> dict[
    str,
    pd.DataFrame | str | float,
]:
    return _evaluate(modeling_data)
def _evaluate(
    data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> dict[
    str,
    pd.DataFrame | str | float,
]:
    base_fit_features = data[
        "base_fit_features"
    ]
    base_fit_target = data[
        "base_fit_target"
    ]
    calibration_features = data[
        "calibration_features"
    ]
    calibration_target = data[
        "calibration_target"
    ]
    validation_features = data[
        "validation_features"
    ]
    validation_target = data[
        "validation_target"
    ]
    assert type(
        base_fit_features
    ) is pd.DataFrame
    assert type(
        base_fit_target
    ) is pd.Series
    assert type(
        calibration_features
    ) is pd.DataFrame
    assert type(
        calibration_target
    ) is pd.Series
    assert type(
        validation_features
    ) is pd.DataFrame
    assert type(
        validation_target
    ) is pd.Series
    return (
        evaluate_probability_calibration_validation(
            base_fit_features,
            base_fit_target,
            calibration_features,
            calibration_target,
            validation_features,
            validation_target,
        )
    )
def _tiny_data() -> dict[
    str,
    pd.DataFrame | pd.Series,
]:
    columns = ["feature"]
    return {
        "base_fit_features": pd.DataFrame(
            {"feature": [0.0, 1.0]},
            index=[0, 1],
            columns=columns,
        ),
        "base_fit_target": pd.Series(
            [0, 1],
            index=[0, 1],
            dtype="int8",
        ),
        "calibration_features": pd.DataFrame(
            {"feature": [2.0, 3.0]},
            index=[2, 3],
            columns=columns,
        ),
        "calibration_target": pd.Series(
            [0, 1],
            index=[2, 3],
            dtype="int8",
        ),
        "validation_features": pd.DataFrame(
            {"feature": [4.0, 5.0]},
            index=[4, 5],
            columns=columns,
        ),
        "validation_target": pd.Series(
            [0, 1],
            index=[4, 5],
            dtype="int8",
        ),
    }
def _fake_candidates() -> dict[
    str,
    _FakeEstimator,
]:
    return {
        "random_forest_uncalibrated": (
            _FakeEstimator(0.10)
        ),
        "random_forest_sigmoid": (
            _FakeEstimator(0.20)
        ),
        "random_forest_isotonic": (
            _FakeEstimator(0.30)
        ),
    }
def test_public_signature_and_result_contract(
    authentic_result: dict[
        str,
        pd.DataFrame | str | float,
    ],
) -> None:
    observed = signature(
        evaluate_probability_calibration_validation
    )
    assert tuple(
        observed.parameters
    ) == _EXPECTED_PARAMETERS
    assert tuple(
        authentic_result
    ) == _EXPECTED_RESULT_KEYS
    assert authentic_result[
        "primary_metric"
    ] == "brier_score"
    assert authentic_result[
        "secondary_metric"
    ] == "log_loss"
    assert authentic_result[
        "selected_model"
    ] == "calibration_prior"
    assert authentic_result[
        "validation_prevalence"
    ] == pytest.approx(
        192 / 1541,
        abs=1e-15,
    )
def test_authentic_metrics_are_exactly_ordered(
    authentic_result: dict[
        str,
        pd.DataFrame | str | float,
    ],
) -> None:
    metrics = authentic_result[
        "metrics"
    ]
    assert type(metrics) is pd.DataFrame
    assert tuple(
        metrics.columns
    ) == _EXPECTED_COLUMNS
    assert tuple(
        metrics["model"]
    ) == _EXPECTED_MODELS
    indexed = metrics.set_index(
        "model"
    )
    for model_name, expected in (
        _EXPECTED_METRICS.items()
    ):
        for metric_name, expected_value in (
            expected.items()
        ):
            assert indexed.loc[
                model_name,
                metric_name,
            ] == pytest.approx(
                expected_value,
                abs=1e-15,
            )
def test_authentic_result_values_are_finite(
    authentic_result: dict[
        str,
        pd.DataFrame | str | float,
    ],
) -> None:
    metrics = authentic_result[
        "metrics"
    ]
    assert type(metrics) is pd.DataFrame
    numeric = metrics.drop(
        columns="model"
    ).to_numpy(
        dtype=np.float64,
        copy=True,
    )
    assert np.isfinite(
        numeric
    ).all()
@pytest.mark.parametrize(
    (
        "rows",
        "expected_model",
    ),
    (
        (
            (
                {
                    "brier_score": 0.09,
                    "log_loss": 0.90,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.10,
                },
                {
                    "brier_score": 0.11,
                    "log_loss": 0.05,
                },
                {
                    "brier_score": 0.12,
                    "log_loss": 0.01,
                },
            ),
            "calibration_prior",
        ),
        (
            (
                {
                    "brier_score": 0.12,
                    "log_loss": 0.10,
                },
                {
                    "brier_score": 0.09,
                    "log_loss": 0.90,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.20,
                },
                {
                    "brier_score": 0.11,
                    "log_loss": 0.05,
                },
            ),
            "random_forest_uncalibrated",
        ),
        (
            (
                {
                    "brier_score": 0.10,
                    "log_loss": 0.40,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.30,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.20,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.10,
                },
            ),
            "random_forest_isotonic",
        ),
        (
            (
                {
                    "brier_score": 0.10,
                    "log_loss": 0.20,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.20,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.20,
                },
                {
                    "brier_score": 0.10,
                    "log_loss": 0.20,
                },
            ),
            "calibration_prior",
        ),
    ),
)
def test_selection_rule_is_fixed(
    monkeypatch: pytest.MonkeyPatch,
    rows: tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ],
    expected_model: str,
) -> None:
    monkeypatch.setattr(
        calibration_validation,
        "fit_probability_calibration_candidates",
        lambda *args: _fake_candidates(),
    )
    metric_rows = iter(rows)
    def fake_evaluation(
        target: pd.Series,
        probability: np.ndarray,
    ) -> dict[str, float]:
        del target
        del probability
        selected = next(
            metric_rows
        )
        return {
            "average_precision": 0.5,
            "roc_auc": 0.5,
            "brier_score": selected[
                "brier_score"
            ],
            "log_loss": selected[
                "log_loss"
            ],
        }
    monkeypatch.setattr(
        calibration_validation,
        "evaluate_binary_probabilities",
        fake_evaluation,
    )
    result = _evaluate(
        _tiny_data()
    )
    assert result[
        "selected_model"
    ] == expected_model


def test_validation_target_is_not_passed_to_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _tiny_data()
    captured: tuple[object, ...] = ()
    def capture_fit(
        *args: object,
    ) -> dict[
        str,
        _FakeEstimator,
    ]:
        nonlocal captured
        captured = args
        return _fake_candidates()
    monkeypatch.setattr(
        calibration_validation,
        "fit_probability_calibration_candidates",
        capture_fit,
    )
    _evaluate(data)
    assert len(captured) == 4
    assert captured[0] is data[
        "base_fit_features"
    ]
    assert captured[1] is data[
        "base_fit_target"
    ]
    assert captured[2] is data[
        "calibration_features"
    ]
    assert captured[3] is data[
        "calibration_target"
    ]
    validation_target = data[
        "validation_target"
    ]
    assert all(
        item is not validation_target
        for item in captured
    )
@pytest.mark.parametrize(
    "argument_name",
    (
        "base_fit_features",
        "calibration_features",
        "validation_features",
    ),
)
def test_feature_types_are_validated(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    argument_name: str,
) -> None:
    data = {
        key: value.copy(deep=True)
        for key, value in modeling_data.items()
    }
    data[argument_name] = []  # type: ignore[assignment]
    with pytest.raises(
        TypeError,
        match="exact pandas DataFrame",
    ):
        evaluate_probability_calibration_validation(
            data["base_fit_features"],  # type: ignore[arg-type]
            data["base_fit_target"],  # type: ignore[arg-type]
            data["calibration_features"],  # type: ignore[arg-type]
            data["calibration_target"],  # type: ignore[arg-type]
            data["validation_features"],  # type: ignore[arg-type]
            data["validation_target"],  # type: ignore[arg-type]
        )
def test_validation_target_type_is_validated(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = {
        key: value.copy(deep=True)
        for key, value in modeling_data.items()
    }
    with pytest.raises(
        TypeError,
        match="exact pandas Series",
    ):
        evaluate_probability_calibration_validation(
            data["base_fit_features"],  # type: ignore[arg-type]
            data["base_fit_target"],  # type: ignore[arg-type]
            data["calibration_features"],  # type: ignore[arg-type]
            data["calibration_target"],  # type: ignore[arg-type]
            data["validation_features"],  # type: ignore[arg-type]
            np.array([0, 1]),
        )
def test_validation_index_mismatch_is_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = {
        key: value.copy(deep=True)
        for key, value in modeling_data.items()
    }
    target = data[
        "validation_target"
    ]
    assert type(target) is pd.Series
    target.index = target.index + 100_000
    with pytest.raises(
        ValueError,
        match=(
            "validation feature and target "
            "indexes must align"
        ),
    ):
        _evaluate(data)
def test_validation_column_mismatch_is_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = {
        key: value.copy(deep=True)
        for key, value in modeling_data.items()
    }
    features = data[
        "validation_features"
    ]
    assert type(features) is pd.DataFrame
    data[
        "validation_features"
    ] = features.loc[
        :,
        list(features.columns)[::-1],
    ]
    with pytest.raises(
        ValueError,
        match=(
            "base-fit and validation "
            "feature columns must match"
        ),
    ):
        _evaluate(data)
def test_validation_overlap_is_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = {
        key: value.copy(deep=True)
        for key, value in modeling_data.items()
    }
    base_features = data[
        "base_fit_features"
    ]
    validation_features = data[
        "validation_features"
    ]
    validation_target = data[
        "validation_target"
    ]
    assert type(
        base_features
    ) is pd.DataFrame
    assert type(
        validation_features
    ) is pd.DataFrame
    assert type(
        validation_target
    ) is pd.Series
    overlap = base_features.index[
        :len(validation_features)
    ]
    validation_features.index = overlap
    validation_target.index = overlap
    with pytest.raises(
        ValueError,
        match=(
            "base-fit and validation indexes "
            "must be disjoint"
        ),
    ):
        _evaluate(data)
def test_candidate_order_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _fake_candidates()
    reordered = {
        "random_forest_sigmoid": candidates[
            "random_forest_sigmoid"
        ],
        "random_forest_uncalibrated": candidates[
            "random_forest_uncalibrated"
        ],
        "random_forest_isotonic": candidates[
            "random_forest_isotonic"
        ],
    }
    monkeypatch.setattr(
        calibration_validation,
        "fit_probability_calibration_candidates",
        lambda *args: reordered,
    )
    with pytest.raises(
        ValueError,
        match=(
            "calibration candidate order "
            "is invalid"
        ),
    ):
        _evaluate(
            _tiny_data()
        )
@pytest.mark.parametrize(
    (
        "estimator_factory",
        "expected_exception",
        "message",
    ),
    (
        (
            lambda: _InvalidClasses(),
            ValueError,
            "classes must be exactly 0 and 1",
        ),
        (
            lambda: _NonArrayProbability(),
            TypeError,
            "exact NumPy ndarray",
        ),
        (
            lambda: _WrongShapeProbability(),
            ValueError,
            "one probability per class",
        ),
        (
            lambda: _NonfiniteProbability(),
            ValueError,
            "must be finite",
        ),
        (
            lambda: _OutOfRangeProbability(),
            ValueError,
            r"within \[0, 1\]",
        ),
        (
            lambda: _InvalidSumProbability(),
            ValueError,
            "must sum to one",
        ),
    ),
)
def test_probability_contract_is_enforced(
    estimator_factory: Callable[
        [],
        object,
    ],
    expected_exception: type[Exception],
    message: str,
) -> None:
    features = pd.DataFrame(
        {"feature": [1.0, 2.0]}
    )
    with pytest.raises(
        expected_exception,
        match=message,
    ):
        calibration_validation._positive_probability(
            model_name="candidate",
            estimator=estimator_factory(),  # type: ignore[arg-type]
            validation_features=features,
        )
class _InvalidClasses(_FakeEstimator):
    def __init__(self) -> None:
        super().__init__(0.2)
        self.classes_ = np.array(
            [1, 0]
        )
class _NonArrayProbability(_FakeEstimator):
    def __init__(self) -> None:
        super().__init__(0.2)
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> list[list[float]]:
        return [
            [0.8, 0.2]
            for _ in range(len(features))
        ]
class _WrongShapeProbability(_FakeEstimator):
    def __init__(self) -> None:
        super().__init__(0.2)
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.full(
            (
                len(features),
                1,
            ),
            0.2,
        )
class _NonfiniteProbability(_FakeEstimator):
    def __init__(self) -> None:
        super().__init__(0.2)
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        result = super().predict_proba(
            features
        )
        result[0, 1] = np.nan
        return result
class _OutOfRangeProbability(_FakeEstimator):
    def __init__(self) -> None:
        super().__init__(0.2)
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.tile(
            np.array(
                [-0.1, 1.1]
            ),
            (
                len(features),
                1,
            ),
        )
class _InvalidSumProbability(_FakeEstimator):
    def __init__(self) -> None:
        super().__init__(0.2)
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.tile(
            np.array(
                [0.7, 0.2]
            ),
            (
                len(features),
                1,
            ),
        )
