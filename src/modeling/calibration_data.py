"""Build leakage-safe chronological calibration modeling populations."""
from __future__ import annotations
import pandas as pd
from src.data.build_dataset import (
    VALIDATION_START as _VALIDATION_START,
)
from src.modeling.data import (
    build_development_modeling_data,
)
__all__ = (
    "CALIBRATION_START",
    "build_calibration_modeling_data",
)
CALIBRATION_START = pd.Timestamp(
    "2024-11-01 00:00:00"
)
_EXPECTED_KEYS = (
    "base_fit_features",
    "base_fit_target",
    "calibration_features",
    "calibration_target",
    "validation_features",
    "validation_target",
)
def build_calibration_modeling_data(
    canonical: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Return chronological base-fit, calibration, and validation data."""
    modeling_data = (
        build_development_modeling_data(
            canonical
        )
    )
    development_features = modeling_data[
        "development_features"
    ]
    development_target = modeling_data[
        "development_target"
    ]
    validation_features = modeling_data[
        "validation_features"
    ]
    validation_target = modeling_data[
        "validation_target"
    ]
    development_prediction_time = canonical.loc[
        development_features.index,
        "prediction_time",
    ].copy(deep=True)
    if development_prediction_time.ge(
        _VALIDATION_START
    ).any():
        raise ValueError(
            "development prediction times must "
            "precede validation"
        )
    base_fit_mask = development_prediction_time.lt(
        CALIBRATION_START
    )
    calibration_mask = (
        development_prediction_time.ge(
            CALIBRATION_START
        )
    )
    if not base_fit_mask.any():
        raise ValueError(
            "base-fit population must not be empty"
        )
    if not calibration_mask.any():
        raise ValueError(
            "calibration population must not be empty"
        )
    if (
        base_fit_mask
        & calibration_mask
    ).any():
        raise ValueError(
            "base-fit and calibration populations "
            "must be disjoint"
        )
    if not (
        base_fit_mask
        | calibration_mask
    ).all():
        raise ValueError(
            "development rows must belong to exactly "
            "one calibration population"
        )
    base_fit_features = development_features.loc[
        base_fit_mask
    ].copy(deep=True)
    base_fit_target = development_target.loc[
        base_fit_mask
    ].copy(deep=True)
    calibration_features = development_features.loc[
        calibration_mask
    ].copy(deep=True)
    calibration_target = development_target.loc[
        calibration_mask
    ].copy(deep=True)
    if set(
        base_fit_target.unique()
    ) != {0, 1}:
        raise ValueError(
            "base-fit target must contain both classes"
        )
    if set(
        calibration_target.unique()
    ) != {0, 1}:
        raise ValueError(
            "calibration target must contain both classes"
        )
    base_fit_time = development_prediction_time.loc[
        base_fit_mask
    ]
    calibration_time = (
        development_prediction_time.loc[
            calibration_mask
        ]
    )
    if (
        base_fit_time.max()
        >= calibration_time.min()
    ):
        raise ValueError(
            "base-fit rows must strictly precede "
            "calibration rows"
        )
    result = {
        "base_fit_features": base_fit_features,
        "base_fit_target": base_fit_target,
        "calibration_features": (
            calibration_features
        ),
        "calibration_target": (
            calibration_target
        ),
        "validation_features": (
            validation_features.copy(
                deep=True
            )
        ),
        "validation_target": (
            validation_target.copy(
                deep=True
            )
        ),
    }
    if tuple(result) != _EXPECTED_KEYS:
        raise RuntimeError(
            "calibration modeling-data order "
            "is invalid"
        )
    return result
