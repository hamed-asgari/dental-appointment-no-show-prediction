from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.modeling.v2_policy_hashes import (
    FROZEN_V2_POLICY_ARTIFACT_SHA256,
    FROZEN_V2_POLICY_ARTIFACT_SIZE_BYTES,
    FROZEN_V2_POLICY_FINAL_TEST_PROBABILITIES_GENERATED,
    FROZEN_V2_POLICY_FINAL_TEST_TARGET_ACCESSED,
    FROZEN_V2_POLICY_MANIFEST_SHA256,
    FROZEN_V2_POLICY_SCENARIO_COUNT,
    FROZEN_V2_POLICY_SELECTED_CALIBRATION_METHOD,
    FROZEN_V2_POLICY_SELECTED_RANKING_MODEL,
    FROZEN_V2_POLICY_SELECTION_POSITIVE_COUNT,
    FROZEN_V2_POLICY_SELECTION_ROWS,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "reports" / "modeling" / "v2" / "policy"
MANIFEST_PATH = POLICY_DIR / "policy_manifest.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_frozen_policy_manifest_identity() -> None:
    assert MANIFEST_PATH.is_file()
    assert digest(MANIFEST_PATH) == FROZEN_V2_POLICY_MANIFEST_SHA256


def test_frozen_policy_manifest_semantics() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["phase"] == "R2"
    assert manifest["stage"] == "policy_sensitivity"
    assert (
        manifest["selected_ranking_model"]
        == FROZEN_V2_POLICY_SELECTED_RANKING_MODEL
    )
    assert (
        manifest["selected_calibration_method"]
        == FROZEN_V2_POLICY_SELECTED_CALIBRATION_METHOD
    )
    assert manifest["policy_selection_rows"] == FROZEN_V2_POLICY_SELECTION_ROWS
    assert (
        manifest["policy_selection_positive_count"]
        == FROZEN_V2_POLICY_SELECTION_POSITIVE_COUNT
    )
    assert manifest["scenario_count"] == FROZEN_V2_POLICY_SCENARIO_COUNT
    assert manifest["single_operational_threshold_selected"] is False
    assert (
        manifest["final_test_target_accessed"]
        is FROZEN_V2_POLICY_FINAL_TEST_TARGET_ACCESSED
    )
    assert (
        manifest["final_test_probabilities_generated"]
        is FROZEN_V2_POLICY_FINAL_TEST_PROBABILITIES_GENERATED
    )


def test_frozen_policy_artifact_hashes_and_sizes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(FROZEN_V2_POLICY_ARTIFACT_SHA256)
    for name, expected_hash in FROZEN_V2_POLICY_ARTIFACT_SHA256.items():
        path = POLICY_DIR / name
        assert path.is_file()
        assert digest(path) == expected_hash
        assert path.stat().st_size == FROZEN_V2_POLICY_ARTIFACT_SIZE_BYTES[name]
        assert manifest["artifacts"][name]["sha256"] == expected_hash
        assert (
            manifest["artifacts"][name]["size_bytes"]
            == FROZEN_V2_POLICY_ARTIFACT_SIZE_BYTES[name]
        )


def test_frozen_policy_predictions_use_only_policy_selection() -> None:
    frame = pd.read_csv(
        POLICY_DIR / "policy_predictions.csv",
        parse_dates=["prediction_time", "label_available_at"],
    )
    assert len(frame) == FROZEN_V2_POLICY_SELECTION_ROWS
    assert int(frame["target"].sum()) == FROZEN_V2_POLICY_SELECTION_POSITIVE_COUNT
    assert frame["appointment_id"].is_unique
    assert set(frame["target"].unique()) == {0, 1}
    assert set(frame["evaluation_partition"].unique()) == {"policy_selection"}
    assert not frame["evaluation_partition"].eq("final_test").any()


def test_frozen_policy_scenario_grid() -> None:
    frame = pd.read_csv(POLICY_DIR / "policy_scenarios.csv")
    assert len(frame) == FROZEN_V2_POLICY_SCENARIO_COUNT
    assert frame["scenario_id"].is_unique
    assert (
        frame["scenario_family"].value_counts().to_dict()
        == {"capacity_cost": 12, "cost_threshold": 4}
    )
    capacity = frame.loc[frame["scenario_family"].eq("capacity_cost")]
    assert sorted(capacity["capacity_fraction"].dropna().unique().tolist()) == [
        0.05,
        0.10,
        0.20,
    ]
    threshold = frame.loc[frame["scenario_family"].eq("cost_threshold")]
    assert sorted(
        threshold["false_negative_to_false_positive_cost_ratio"].tolist()
    ) == [1.0, 2.0, 5.0, 10.0]


def test_frozen_policy_summary_does_not_select_operational_threshold() -> None:
    summary = json.loads(
        (POLICY_DIR / "policy_summary.json").read_text(encoding="utf-8")
    )
    assert summary["selected_ranking_model"] == "logistic_regression"
    assert summary["selected_calibration_method"] == "uncalibrated"
    assert summary["policy_selection_rows"] == 1063
    assert summary["policy_selection_positive_count"] == 92
    assert summary["scenario_count"] == 16
    assert summary["single_operational_threshold_selected"] is False
    assert summary["final_test_target_accessed"] is False
    assert summary["final_test_probabilities_generated"] is False
