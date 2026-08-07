from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.export_v2_processed import load_frozen_v2_processed_feature_dataset
from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.v2_development import _positive_class_probability
from src.modeling.v2_final_test_probabilities import (
    EXPECTED_FINAL_TEST_ROWS,
    EXPECTED_PROBABILITY_COLUMNS,
    FROZEN_V2_DIAGNOSTICS_MANIFEST_SHA256,
    FROZEN_V2_PERSISTENCE_MANIFEST_SHA256,
    FROZEN_V2_PIPELINE_SHA256,
    FROZEN_V2_R3_CONFIG_SHA256,
    MANIFEST_FILENAME,
    PROBABILITY_FILENAME,
    _appointment_order_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_DIR = ROOT / "reports" / "modeling" / "v2" / "final_test"
PROBABILITY_PATH = FINAL_TEST_DIR / PROBABILITY_FILENAME
MANIFEST_PATH = FINAL_TEST_DIR / MANIFEST_FILENAME
PIPELINE_PATH = ROOT / "models" / "v2" / "frozen_logistic_pipeline.joblib"
ENGINE_SOURCE = ROOT / "src" / "modeling" / "v2_final_test_probabilities.py"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_probability_seal_manifest_preserves_pre_target_boundary() -> None:
    manifest = _manifest()
    assert manifest["stage"] == "protected_final_test_probability_seal"
    assert manifest["r3_execution_config_sha256"] == FROZEN_V2_R3_CONFIG_SHA256
    assert manifest["pipeline_sha256"] == FROZEN_V2_PIPELINE_SHA256
    assert manifest["persistence_manifest_sha256"] == (
        FROZEN_V2_PERSISTENCE_MANIFEST_SHA256
    )
    assert manifest["diagnostics_manifest_sha256"] == (
        FROZEN_V2_DIAGNOSTICS_MANIFEST_SHA256
    )
    assert manifest["partition"] == "final_test"
    assert manifest["probability_metrics_computed"] is False
    assert manifest["single_operational_threshold_selected"] is False
    assert manifest["final_test_probabilities_generated"] is True
    assert manifest["final_test_target_accessed"] is False
    assert manifest["probability_commit_and_ci_green_required_before_target_access"] is True
    assert manifest["target_access_requires_explicit_allow_test_true"] is True


def test_probability_vector_has_exact_frozen_schema_and_row_count() -> None:
    vector = pd.read_csv(
        PROBABILITY_PATH,
        dtype={
            "appointment_id": "int64",
            "no_show_probability": "float64",
        },
    )
    assert list(vector.columns) == list(EXPECTED_PROBABILITY_COLUMNS)
    assert len(vector) == EXPECTED_FINAL_TEST_ROWS
    assert vector["appointment_id"].is_unique
    probability = vector["no_show_probability"].to_numpy(dtype=np.float64, copy=True)
    assert np.isfinite(probability).all()
    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)


def test_probability_vector_hash_and_size_match_manifest() -> None:
    manifest = _manifest()
    artifact = manifest["artifacts"][PROBABILITY_FILENAME]
    assert _digest(PROBABILITY_PATH) == artifact["sha256"]
    assert PROBABILITY_PATH.stat().st_size == artifact["size_bytes"]


def test_probability_vector_order_matches_frozen_target_free_feature_dataset() -> None:
    feature_dataset = load_frozen_v2_processed_feature_dataset()
    assert "target" not in feature_dataset.columns
    final_rows = feature_dataset.loc[
        feature_dataset["evaluation_partition"].astype("string").eq("final_test")
    ]
    expected_ids = final_rows["appointment_id"].to_numpy(dtype=np.int64, copy=True)

    vector = pd.read_csv(
        PROBABILITY_PATH,
        usecols=["appointment_id"],
        dtype={"appointment_id": "int64"},
    )
    actual_ids = vector["appointment_id"].to_numpy(dtype=np.int64, copy=True)

    assert len(expected_ids) == EXPECTED_FINAL_TEST_ROWS
    assert np.array_equal(actual_ids, expected_ids)
    assert _manifest()["appointment_order_sha256"] == _appointment_order_sha256(
        expected_ids
    )


def test_probability_vector_replays_frozen_persisted_pipeline_without_target() -> None:
    feature_dataset = load_frozen_v2_processed_feature_dataset()
    assert "target" not in feature_dataset.columns
    final_rows = feature_dataset.loc[
        feature_dataset["evaluation_partition"].astype("string").eq("final_test")
    ]
    features = final_rows.loc[:, list(V2_MODEL_FEATURE_COLUMNS)].reset_index(drop=True)

    estimator = joblib.load(PIPELINE_PATH)
    assert isinstance(estimator, Pipeline)
    replay = _positive_class_probability(estimator, features)

    vector = pd.read_csv(
        PROBABILITY_PATH,
        usecols=["no_show_probability"],
        dtype={"no_show_probability": "float64"},
        float_precision="round_trip",
    )
    expected = vector["no_show_probability"].to_numpy(dtype=np.float64, copy=True)

    assert np.array_equal(replay, expected)


def test_probability_generation_engine_does_not_import_target_builder() -> None:
    text = ENGINE_SOURCE.read_text(encoding="utf-8")
    assert "build_mature_v2_target_table" not in text
    assert "allow_test=True" not in text
    assert "average_precision_score" not in text
    assert "roc_auc_score" not in text
    assert "brier_score_loss" not in text
    assert "log_loss(" not in text


def test_probability_manifest_has_no_target_derived_fields() -> None:
    manifest = _manifest()
    forbidden = {
        "positive_count",
        "prevalence",
        "average_precision",
        "roc_auc",
        "brier_score",
        "log_loss",
        "calibration_intercept",
        "calibration_slope",
    }
    assert forbidden.isdisjoint(manifest)


def test_probability_vector_is_exactly_4343_unique_appointments() -> None:
    vector = pd.read_csv(PROBABILITY_PATH)
    assert len(vector) == 4343
    assert vector["appointment_id"].nunique() == 4343
