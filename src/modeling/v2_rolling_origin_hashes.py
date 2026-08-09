"""Frozen identities for committed Version 2 rolling-origin results."""

from types import MappingProxyType


FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256 = (
    "e575b10835645d3a643c396803cfff21f5c1c1cdad9b988ee07037ef045beb45"
)
FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL = "logistic_regression"
FROZEN_V2_ROLLING_ORIGIN_ARTIFACT_SHA256 = MappingProxyType(
    {'ranking_selection.json': '23796b642ab9ea89e09bc01c6ba713a9b3835fb858d151e0bb212f7a8265a28c', 'rolling_origin_fold_metrics.csv': '4a2b6d85e164673a146068dee25d3596b654ef3926b48265021ea8cc8f7d73ed', 'rolling_origin_macro_summary.csv': 'aba1c540fc2f6d664ee162d3bdf77211bef16412d76a3c2ea4262d3ef041349a', 'rolling_origin_pooled_summary.csv': 'db97378c152cf6dd4cf81cd6468aa45a70b41afda4c336adb5fe572a06a61352', 'rolling_origin_predictions.csv': '4aadf518af90cfe20d2de3a38281ea7d33f3e764cbd97fc8c4fb78759c33f636'}
)


__all__ = (
    "FROZEN_V2_ROLLING_ORIGIN_ARTIFACT_SHA256",
    "FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256",
    "FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL",
)
