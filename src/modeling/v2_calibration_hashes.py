"""Frozen identities for committed Version 2 calibration results."""

from types import MappingProxyType


FROZEN_V2_CALIBRATION_MANIFEST_SHA256 = "5b2e701753a2d0e0d2f9a7efaddf46b2f316643e66dd8c11727373927c8a5d7a"
FROZEN_V2_SELECTED_CALIBRATION_METHOD = "uncalibrated"
FROZEN_V2_CALIBRATION_ARTIFACT_SHA256 = MappingProxyType(
    {'calibration_metrics.csv': '9c6ded314852564755cf83a7da8ad172b205f211dcf80b7f2471504b1b284f74', 'calibration_predictions.csv': 'aca987a269976affc69d698677edebf4d673073f788b4988f54884a499fe6783', 'calibration_reliability_curve.csv': '660c97889964114c24e5c6113f9e4a1d63d5cf2cb5edb9e954b0bc9dedf00044', 'calibration_selection.json': '5e2023517c9918387497a17c09592a31e8dd43cd574db8efc7d91ae004acd417'}
)

__all__ = (
    "FROZEN_V2_CALIBRATION_ARTIFACT_SHA256",
    "FROZEN_V2_CALIBRATION_MANIFEST_SHA256",
    "FROZEN_V2_SELECTED_CALIBRATION_METHOD",
)
