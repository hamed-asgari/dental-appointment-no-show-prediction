from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.export_v2_processed import load_frozen_v2_processed_feature_dataset
from src.modeling.v2_development import _positive_class_probability, _select_features_by_ids
from src.modeling.v2_persistence import (
    DEFAULT_V2_PERSISTENCE_DIR,
    FROZEN_V2_POLICY_MANIFEST_SHA256,
    FROZEN_V2_R3_CONFIG_SHA256,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    PIPELINE_FILENAME,
    POLICY_REPLAY_ATOL,
    POLICY_REPLAY_RTOL,
    export_v2_frozen_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "reports" / "modeling" / "v2" / "policy"
POLICY_MANIFEST = POLICY_DIR / "policy_manifest.json"
POLICY_PREDICTIONS = POLICY_DIR / "policy_predictions.csv"
PIPELINE = DEFAULT_V2_PERSISTENCE_DIR / PIPELINE_FILENAME
METADATA = DEFAULT_V2_PERSISTENCE_DIR / METADATA_FILENAME
MANIFEST = DEFAULT_V2_PERSISTENCE_DIR / MANIFEST_FILENAME


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _metadata() -> dict[str, object]:
    return json.loads(METADATA.read_text(encoding="utf-8"))


def test_persistence_manifest_preserves_frozen_pretest_state() -> None:
    manifest = _manifest()
    assert manifest["stage"] == "frozen_model_persistence"
    assert manifest["r3_execution_config_sha256"] == FROZEN_V2_R3_CONFIG_SHA256
    assert manifest["policy_manifest_sha256"] == FROZEN_V2_POLICY_MANIFEST_SHA256
    assert manifest["selected_ranking_model"] == "logistic_regression"
    assert manifest["selected_calibration_method"] == "uncalibrated"
    assert manifest["single_operational_threshold_selected"] is False
    assert manifest["final_test_target_accessed"] is False
    assert manifest["final_test_probabilities_generated"] is False


def test_persistence_artifact_hashes_and_sizes_match_manifest() -> None:
    manifest = _manifest()
    artifacts = manifest["artifacts"]
    for path in (PIPELINE, METADATA):
        entry = artifacts[path.name]
        assert _digest(path) == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]


def test_persistence_metadata_matches_frozen_training_identity() -> None:
    metadata = _metadata()
    assert metadata["base_fit_time"] == "2026-07-01T00:00:00"
    assert metadata["base_training_rows"] == 10921
    assert metadata["base_training_positive_count"] == 978
    assert metadata["model_feature_count"] == 32
    assert len(metadata["model_feature_columns"]) == 32
    assert metadata["policy_prediction_replay_rows"] == 1063
    assert metadata["final_test_feature_rows_observed_without_scoring"] == 4343
    assert metadata["final_test_target_accessed"] is False
    assert metadata["final_test_probabilities_generated"] is False


def test_persisted_pipeline_loads_and_has_binary_classifier() -> None:
    estimator = joblib.load(PIPELINE)
    assert isinstance(estimator, Pipeline)
    assert tuple(estimator.named_steps) == ("preprocessor", "classifier")
    classes = np.asarray(estimator.named_steps["classifier"].classes_)
    assert np.array_equal(classes, np.array([0, 1]))


def test_persisted_pipeline_replays_frozen_policy_probabilities() -> None:
    policy_manifest = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
    assert _digest(POLICY_MANIFEST) == FROZEN_V2_POLICY_MANIFEST_SHA256
    frozen_entry = policy_manifest["artifacts"]["policy_predictions.csv"]
    assert _digest(POLICY_PREDICTIONS) == frozen_entry["sha256"]

    frozen = pd.read_csv(
        POLICY_PREDICTIONS,
        usecols=["appointment_id", "no_show_probability"],
        dtype={"appointment_id": "int64", "no_show_probability": "float64"},
    )
    feature_dataset = load_frozen_v2_processed_feature_dataset()
    features = _select_features_by_ids(feature_dataset, frozen["appointment_id"])

    estimator = joblib.load(PIPELINE)
    probability = _positive_class_probability(estimator, features)
    expected = frozen["no_show_probability"].to_numpy(dtype=np.float64, copy=True)

    assert len(probability) == 1063
    assert np.allclose(
        probability,
        expected,
        atol=POLICY_REPLAY_ATOL,
        rtol=POLICY_REPLAY_RTOL,
    )


def test_export_refuses_existing_persistence_outputs_before_refit(tmp_path: Path) -> None:
    occupied = tmp_path / MANIFEST_FILENAME
    occupied.write_text("{}\n", encoding="utf-8", newline="\n")

    class DummyBuild:
        estimator = None
        metadata = {}
        policy_features = pd.DataFrame()
        policy_probabilities = np.array([], dtype=np.float64)

    with pytest.raises(ValueError, match="already exist"):
        export_v2_frozen_pipeline(
            DummyBuild(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            overwrite=False,
        )
