from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.modeling.v2_development import (
    DEFAULT_V2_MODELING_OUTPUT_DIR,
    PREDICTION_COLUMNS,
    export_v2_rolling_origin_results,
    run_v2_rolling_origin_development,
)
from src.modeling.v2_rolling_origin_hashes import (
    FROZEN_V2_ROLLING_ORIGIN_ARTIFACT_SHA256,
    FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
    FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
)


MANIFEST_PATH = (
    DEFAULT_V2_MODELING_OUTPUT_DIR / "rolling_origin_manifest.json"
)


@pytest.fixture(scope="module")
def result():
    return run_v2_rolling_origin_development()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_manifest_matches_frozen_identity() -> None:
    assert MANIFEST_PATH.is_file()
    assert _sha256(MANIFEST_PATH) == (
        FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256
    )


def test_committed_artifact_set_is_exact() -> None:
    expected = {
        "README.md",
        "ranking_selection.json",
        "rolling_origin_fold_metrics.csv",
        "rolling_origin_macro_summary.csv",
        "rolling_origin_pooled_summary.csv",
        "rolling_origin_predictions.csv",
        "rolling_origin_manifest.json",
    }
    assert {
        path.name
        for path in DEFAULT_V2_MODELING_OUTPUT_DIR.iterdir()
        if path.is_file()
    } == expected


def test_manifest_preserves_protected_test_state_and_input_identity() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["phase"] == "R2"
    assert manifest["stage"] == "rolling_origin_ranking"
    assert manifest["final_test_target_accessed"] is False
    assert manifest["final_test_probabilities_generated"] is False
    assert manifest["processed_dataset_sha256"] == (
        "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53"
    )
    assert manifest["processed_manifest_sha256"] == (
        "2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073"
    )
    assert manifest["processed_dataset_fingerprint"] == (
        "0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787"
    )


def test_manifest_hashes_match_every_committed_result_artifact() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(
        FROZEN_V2_ROLLING_ORIGIN_ARTIFACT_SHA256
    )
    for filename, expected_hash in (
        FROZEN_V2_ROLLING_ORIGIN_ARTIFACT_SHA256.items()
    ):
        path = DEFAULT_V2_MODELING_OUTPUT_DIR / filename
        assert path.is_file()
        assert _sha256(path) == expected_hash
        assert manifest["artifacts"][filename]["sha256"] == expected_hash
        assert manifest["artifacts"][filename]["size_bytes"] == path.stat().st_size


def test_committed_predictions_have_no_final_test_rows() -> None:
    frame = pd.read_csv(
        DEFAULT_V2_MODELING_OUTPUT_DIR / "rolling_origin_predictions.csv"
    )
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert len(frame) == 19284
    assert not frame["evaluation_partition"].eq("final_test").any()


def test_frozen_selected_model_matches_selection_artifact() -> None:
    selection = json.loads(
        (
            DEFAULT_V2_MODELING_OUTPUT_DIR / "ranking_selection.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert selection["selected_ranking_model"] == (
        FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL
    )
    assert manifest["selected_ranking_model"] == (
        FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL
    )
    assert selection["final_test_target_accessed"] is False
    assert selection["final_test_probabilities_generated"] is False


def test_selected_nonconstant_model_passes_frozen_gate_when_not_fallback() -> None:
    selection = json.loads(
        (
            DEFAULT_V2_MODELING_OUTPUT_DIR / "ranking_selection.json"
        ).read_text(encoding="utf-8")
    )
    macro = pd.read_csv(
        DEFAULT_V2_MODELING_OUTPUT_DIR / "rolling_origin_macro_summary.csv"
    )
    if selection["fallback_to_population_prior"]:
        assert selection["selected_ranking_model"] == "population_prior"
        assert not macro["passes_minimum_usefulness_gate"].astype(bool).any()
    else:
        selected = macro.loc[
            macro["model"].eq(selection["selected_ranking_model"])
        ]
        assert selected["passes_minimum_usefulness_gate"].astype(bool).item()


def test_export_refuses_overwrite_without_explicit_opt_in(
    result,
    tmp_path: Path,
) -> None:
    export_v2_rolling_origin_results(result, output_dir=tmp_path)
    with pytest.raises(ValueError, match="already exist"):
        export_v2_rolling_origin_results(result, output_dir=tmp_path)


def test_export_is_byte_deterministic_with_explicit_overwrite(
    result,
    tmp_path: Path,
) -> None:
    export_v2_rolling_origin_results(result, output_dir=tmp_path)
    before = {
        path.name: _sha256(path)
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    export_v2_rolling_origin_results(
        result,
        output_dir=tmp_path,
        overwrite=True,
    )
    after = {
        path.name: _sha256(path)
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after == before


def test_recomputed_result_matches_committed_selection(result) -> None:
    assert result.selection["selected_ranking_model"] == (
        FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL
    )
    assert result.selection["final_test_target_accessed"] is False
    assert result.selection["final_test_probabilities_generated"] is False
