from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.modeling.v2_calibration import (
    CALIBRATION_METHODS,
    DEFAULT_V2_CALIBRATION_OUTPUT_DIR,
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    RELIABILITY_COLUMNS,
    export_v2_calibration_results,
    run_v2_calibration_evaluation,
)
from src.modeling.v2_calibration_hashes import (
    FROZEN_V2_CALIBRATION_ARTIFACT_SHA256,
    FROZEN_V2_CALIBRATION_MANIFEST_SHA256,
    FROZEN_V2_SELECTED_CALIBRATION_METHOD,
)


MANIFEST_PATH = DEFAULT_V2_CALIBRATION_OUTPUT_DIR / "calibration_manifest.json"


@pytest.fixture(scope="module")
def result():
    return run_v2_calibration_evaluation()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_calibration_manifest_matches_frozen_identity() -> None:
    assert MANIFEST_PATH.is_file()
    assert _sha256(MANIFEST_PATH) == FROZEN_V2_CALIBRATION_MANIFEST_SHA256


def test_committed_calibration_artifact_set_is_exact() -> None:
    expected = {
        "README.md",
        "calibration_metrics.csv",
        "calibration_predictions.csv",
        "calibration_reliability_curve.csv",
        "calibration_selection.json",
        "calibration_manifest.json",
    }
    assert {
        path.name
        for path in DEFAULT_V2_CALIBRATION_OUTPUT_DIR.iterdir()
        if path.is_file()
    } == expected


def test_manifest_preserves_frozen_inputs_and_protected_test_state() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["phase"] == "R2"
    assert manifest["stage"] == "chronological_calibration"
    assert manifest["selected_ranking_model"] == "logistic_regression"
    assert manifest["final_test_target_accessed"] is False
    assert manifest["final_test_probabilities_generated"] is False
    assert manifest["processed_dataset_sha256"] == (
        "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53"
    )
    assert manifest["rolling_origin_manifest_sha256"] == (
        "e575b10835645d3a643c396803cfff21f5c1c1cdad9b988ee07037ef045beb45"
    )


def test_manifest_hashes_match_every_committed_calibration_artifact() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(FROZEN_V2_CALIBRATION_ARTIFACT_SHA256)
    for filename, expected_hash in FROZEN_V2_CALIBRATION_ARTIFACT_SHA256.items():
        path = DEFAULT_V2_CALIBRATION_OUTPUT_DIR / filename
        assert path.is_file()
        assert _sha256(path) == expected_hash
        assert manifest["artifacts"][filename]["sha256"] == expected_hash
        assert manifest["artifacts"][filename]["size_bytes"] == path.stat().st_size


def test_committed_calibration_metrics_match_frozen_selection() -> None:
    metrics = pd.read_csv(DEFAULT_V2_CALIBRATION_OUTPUT_DIR / "calibration_metrics.csv")
    assert tuple(metrics.columns) == METRIC_COLUMNS
    assert tuple(metrics["method"]) == CALIBRATION_METHODS
    selected = metrics.loc[metrics["selected"].astype(bool), "method"]
    assert len(selected) == 1
    assert selected.item() == FROZEN_V2_SELECTED_CALIBRATION_METHOD


def test_committed_calibration_predictions_have_no_final_test_rows() -> None:
    frame = pd.read_csv(
        DEFAULT_V2_CALIBRATION_OUTPUT_DIR / "calibration_predictions.csv"
    )
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert len(frame) == 984
    assert frame["evaluation_partition"].eq("calibration").all()
    assert not frame["evaluation_partition"].eq("final_test").any()


def test_committed_reliability_curve_is_ten_bins_per_method() -> None:
    curve = pd.read_csv(
        DEFAULT_V2_CALIBRATION_OUTPUT_DIR / "calibration_reliability_curve.csv"
    )
    assert tuple(curve.columns) == RELIABILITY_COLUMNS
    assert len(curve) == 30
    assert curve.groupby("method")["bin"].nunique().eq(10).all()
    assert curve.groupby("method")["bin_count"].sum().eq(328).all()


def test_export_refuses_overwrite_without_explicit_opt_in(
    result,
    tmp_path: Path,
) -> None:
    export_v2_calibration_results(result, output_dir=tmp_path)
    with pytest.raises(ValueError, match="already exist"):
        export_v2_calibration_results(result, output_dir=tmp_path)


def test_export_is_byte_deterministic_with_explicit_overwrite(
    result,
    tmp_path: Path,
) -> None:
    export_v2_calibration_results(result, output_dir=tmp_path)
    before = {
        path.name: _sha256(path)
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    export_v2_calibration_results(result, output_dir=tmp_path, overwrite=True)
    after = {
        path.name: _sha256(path)
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after == before


def test_recomputed_result_matches_committed_calibration_selection(result) -> None:
    assert result.selection["selected_calibration_method"] == (
        FROZEN_V2_SELECTED_CALIBRATION_METHOD
    )
    assert result.selection["final_test_target_accessed"] is False
    assert result.selection["final_test_probabilities_generated"] is False
