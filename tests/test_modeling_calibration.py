"""Tests for deterministic probability-calibration candidates."""
from __future__ import annotations
from inspect import signature
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from src.data import build_dataset as bd
from src.modeling.calibration import (
    fit_probability_calibration_candidates,
)
from src.modeling.calibration_data import (
    build_calibration_modeling_data,
)
_EXPECTED_MODELS = (
    "random_forest_uncalibrated",
    "random_forest_sigmoid",
    "random_forest_isotonic",
)
_EXPECTED_PARAMETERS = (
    "base_fit_features",
    "base_fit_target",
    "calibration_features",
    "calibration_target",
)
@pytest.fixture(scope="module")
def canonical_dataset() -> pd.DataFrame:
    """Return one authentic canonical dataset."""
    raw_dir = Path("data/raw")
    bd.validate_raw_hashes(raw_dir)
    return bd.build_analytical_dataset(
        bd.load_raw_data(raw_dir)
    )
@pytest.fixture(scope="module")
def modeling_data(
    canonical_dataset: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Return authentic chronological calibration data."""
    return build_calibration_modeling_data(
        canonical_dataset
    )
@pytest.fixture(scope="module")
def candidate_suite(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> dict[
    str,
    Pipeline | CalibratedClassifierCV,
]:
    """Return one fitted authentic candidate suite."""
    return _fit_candidates(modeling_data)
def _copy_data(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> dict[
    str,
    pd.DataFrame | pd.Series,
]:
    return {
        key: value.copy(deep=True)
        for key, value in modeling_data.items()
    }
def _fit_candidates(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> dict[
    str,
    Pipeline | CalibratedClassifierCV,
]:
    base_fit_features = modeling_data[
        "base_fit_features"
    ]
    base_fit_target = modeling_data[
        "base_fit_target"
    ]
    calibration_features = modeling_data[
        "calibration_features"
    ]
    calibration_target = modeling_data[
        "calibration_target"
    ]
    assert type(base_fit_features) is pd.DataFrame
    assert type(base_fit_target) is pd.Series
    assert type(
        calibration_features
    ) is pd.DataFrame
    assert type(
        calibration_target
    ) is pd.Series
    return (
        fit_probability_calibration_candidates(
            base_fit_features,
            base_fit_target,
            calibration_features,
            calibration_target,
        )
    )
def test_public_signature_and_candidate_order(
    candidate_suite: dict[
        str,
        Pipeline | CalibratedClassifierCV,
    ],
) -> None:
    observed_signature = signature(
        fit_probability_calibration_candidates
    )
    assert tuple(
        observed_signature.parameters
    ) == _EXPECTED_PARAMETERS
    assert tuple(
        candidate_suite
    ) == _EXPECTED_MODELS
    assert type(
        candidate_suite[
            "random_forest_uncalibrated"
        ]
    ) is Pipeline
    assert type(
        candidate_suite[
            "random_forest_sigmoid"
        ]
    ) is CalibratedClassifierCV
    assert type(
        candidate_suite[
            "random_forest_isotonic"
        ]
    ) is CalibratedClassifierCV
def test_calibrator_configuration_is_explicit(
    candidate_suite: dict[
        str,
        Pipeline | CalibratedClassifierCV,
    ],
) -> None:
    sigmoid = candidate_suite[
        "random_forest_sigmoid"
    ]
    isotonic = candidate_suite[
        "random_forest_isotonic"
    ]
    assert type(sigmoid) is CalibratedClassifierCV
    assert type(isotonic) is CalibratedClassifierCV
    assert sigmoid.method == "sigmoid"
    assert isotonic.method == "isotonic"
    assert sigmoid.cv is None
    assert isotonic.cv is None
    assert sigmoid.ensemble is False
    assert isotonic.ensemble is False
    assert sigmoid.n_jobs is None
    assert isotonic.n_jobs is None
    assert type(
        sigmoid.estimator
    ) is FrozenEstimator
    assert type(
        isotonic.estimator
    ) is FrozenEstimator
def test_all_candidates_have_binary_classes(
    candidate_suite: dict[
        str,
        Pipeline | CalibratedClassifierCV,
    ],
) -> None:
    for model_name in _EXPECTED_MODELS:
        estimator = candidate_suite[
            model_name
        ]
        assert np.array_equal(
            estimator.classes_,
            np.array([0, 1]),
        )
def test_probabilities_are_valid(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    candidate_suite: dict[
        str,
        Pipeline | CalibratedClassifierCV,
    ],
) -> None:
    validation_features = modeling_data[
        "validation_features"
    ]
    assert type(
        validation_features
    ) is pd.DataFrame
    for model_name in _EXPECTED_MODELS:
        probability = candidate_suite[
            model_name
        ].predict_proba(
            validation_features
        )
        assert probability.shape == (
            1_541,
            2,
        )
        assert np.isfinite(
            probability
        ).all()
        assert (
            probability >= 0.0
        ).all()
        assert (
            probability <= 1.0
        ).all()
        assert np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-15,
        )
def test_frozen_estimators_match_raw_base_model(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    candidate_suite: dict[
        str,
        Pipeline | CalibratedClassifierCV,
    ],
) -> None:
    validation_features = modeling_data[
        "validation_features"
    ]
    assert type(
        validation_features
    ) is pd.DataFrame
    raw = candidate_suite[
        "random_forest_uncalibrated"
    ]
    sigmoid = candidate_suite[
        "random_forest_sigmoid"
    ]
    isotonic = candidate_suite[
        "random_forest_isotonic"
    ]
    assert type(raw) is Pipeline
    assert type(sigmoid) is CalibratedClassifierCV
    assert type(isotonic) is CalibratedClassifierCV
    sigmoid_frozen = (
        sigmoid.estimator.estimator
    )
    isotonic_frozen = (
        isotonic.estimator.estimator
    )
    assert type(
        sigmoid_frozen
    ) is Pipeline
    assert type(
        isotonic_frozen
    ) is Pipeline
    assert raw is not sigmoid_frozen
    assert raw is not isotonic_frozen
    assert (
        sigmoid_frozen
        is not isotonic_frozen
    )
    raw_probability = raw.predict_proba(
        validation_features
    )
    assert np.array_equal(
        raw_probability,
        sigmoid_frozen.predict_proba(
            validation_features
        ),
    )
    assert np.array_equal(
        raw_probability,
        isotonic_frozen.predict_proba(
            validation_features
        ),
    )
def test_repeated_fits_are_exactly_deterministic(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    first = _fit_candidates(
        modeling_data
    )
    second = _fit_candidates(
        modeling_data
    )
    validation_features = modeling_data[
        "validation_features"
    ]
    assert type(
        validation_features
    ) is pd.DataFrame
    assert first is not second
    for model_name in _EXPECTED_MODELS:
        assert (
            first[model_name]
            is not second[model_name]
        )
        assert np.array_equal(
            first[
                model_name
            ].predict_proba(
                validation_features
            ),
            second[
                model_name
            ].predict_proba(
                validation_features
            ),
        )
def test_inputs_are_not_mutated(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = _copy_data(
        modeling_data
    )
    snapshots = _copy_data(
        data
    )
    _fit_candidates(data)
    for key, expected in (
        snapshots.items()
    ):
        observed = data[key]
        if type(expected) is pd.DataFrame:
            assert type(
                observed
            ) is pd.DataFrame
            pd.testing.assert_frame_equal(
                observed,
                expected,
            )
        else:
            assert type(
                observed
            ) is pd.Series
            pd.testing.assert_series_equal(
                observed,
                expected,
            )
def test_calibration_target_cannot_refit_base_model(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    original_data = _copy_data(
        modeling_data
    )
    poisoned_data = _copy_data(
        modeling_data
    )
    poisoned_target = poisoned_data[
        "calibration_target"
    ]
    assert type(
        poisoned_target
    ) is pd.Series
    poisoned_data[
        "calibration_target"
    ] = (
        1 - poisoned_target
    ).astype("int8")
    original = _fit_candidates(
        original_data
    )
    poisoned = _fit_candidates(
        poisoned_data
    )
    validation_features = modeling_data[
        "validation_features"
    ]
    assert type(
        validation_features
    ) is pd.DataFrame
    raw_original = original[
        "random_forest_uncalibrated"
    ].predict_proba(
        validation_features
    )
    raw_poisoned = poisoned[
        "random_forest_uncalibrated"
    ].predict_proba(
        validation_features
    )
    assert np.array_equal(
        raw_original,
        raw_poisoned,
    )
    for model_name in (
        "random_forest_sigmoid",
        "random_forest_isotonic",
    ):
        calibrated_original = original[
            model_name
        ].predict_proba(
            validation_features
        )
        calibrated_poisoned = poisoned[
            model_name
        ].predict_proba(
            validation_features
        )
        assert not np.array_equal(
            calibrated_original,
            calibrated_poisoned,
        )
@pytest.mark.parametrize(
    ("argument_name", "replacement"),
    (
        (
            "base_fit_features",
            [],
        ),
        (
            "calibration_features",
            [],
        ),
        (
            "base_fit_target",
            np.array([0, 1]),
        ),
        (
            "calibration_target",
            np.array([0, 1]),
        ),
    ),
)
def test_exact_input_types_are_required(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    argument_name: str,
    replacement: object,
) -> None:
    data = _copy_data(
        modeling_data
    )
    data[argument_name] = replacement  # type: ignore[assignment]
    with pytest.raises(TypeError):
        fit_probability_calibration_candidates(
            data["base_fit_features"],
            data["base_fit_target"],
            data["calibration_features"],
            data["calibration_target"],
        )
@pytest.mark.parametrize(
    "target_name",
    (
        "base_fit_target",
        "calibration_target",
    ),
)
def test_nonbinary_targets_are_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    target_name: str,
) -> None:
    data = _copy_data(
        modeling_data
    )
    target = data[target_name]
    assert type(target) is pd.Series
    target.iloc[0] = 2
    with pytest.raises(
        ValueError,
        match="values must be exactly 0 and 1",
    ):
        _fit_candidates(data)
@pytest.mark.parametrize(
    "feature_name",
    (
        "base_fit_features",
        "calibration_features",
    ),
)
def test_missing_feature_values_are_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    feature_name: str,
) -> None:
    data = _copy_data(
        modeling_data
    )
    features = data[feature_name]
    assert type(features) is pd.DataFrame
    features.iloc[0, 0] = np.nan
    with pytest.raises(
        ValueError,
        match="must not contain missing values",
    ):
        _fit_candidates(data)
@pytest.mark.parametrize(
    "target_name",
    (
        "base_fit_target",
        "calibration_target",
    ),
)
def test_missing_target_values_are_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
    target_name: str,
) -> None:
    data = _copy_data(
        modeling_data
    )
    target = data[target_name]
    assert type(target) is pd.Series
    target.iloc[0] = pd.NA
    with pytest.raises(
        ValueError,
        match="must not contain missing values",
    ):
        _fit_candidates(data)
def test_feature_target_index_mismatch_is_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = _copy_data(
        modeling_data
    )
    target = data[
        "calibration_target"
    ]
    assert type(target) is pd.Series
    target.index = target.index + 100_000
    with pytest.raises(
        ValueError,
        match=(
            "calibration feature and target "
            "indexes must align"
        ),
    ):
        _fit_candidates(data)
def test_feature_column_mismatch_is_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = _copy_data(
        modeling_data
    )
    features = data[
        "calibration_features"
    ]
    assert type(features) is pd.DataFrame
    data[
        "calibration_features"
    ] = features.loc[
        :,
        list(features.columns)[::-1],
    ]
    with pytest.raises(
        ValueError,
        match=(
            "base-fit and calibration "
            "feature columns must match"
        ),
    ):
        _fit_candidates(data)
def test_overlapping_population_indexes_are_rejected(
    modeling_data: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    data = _copy_data(
        modeling_data
    )
    base_features = data[
        "base_fit_features"
    ]
    calibration_features = data[
        "calibration_features"
    ]
    calibration_target = data[
        "calibration_target"
    ]
    assert type(
        base_features
    ) is pd.DataFrame
    assert type(
        calibration_features
    ) is pd.DataFrame
    assert type(
        calibration_target
    ) is pd.Series
    overlapping_index = base_features.index[
        :len(calibration_features)
    ]
    calibration_features.index = (
        overlapping_index
    )
    calibration_target.index = (
        overlapping_index
    )
    with pytest.raises(
        ValueError,
        match=(
            "base-fit and calibration "
            "indexes must be disjoint"
        ),
    ):
        _fit_candidates(data)
