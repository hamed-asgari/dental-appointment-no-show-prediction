from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.v2_diagnostics import (
    CAPACITY_FILENAME,
    FIRST_REPEAT_FILENAME,
    FROZEN_V2_PERSISTENCE_MANIFEST_SHA256,
    FROZEN_V2_PIPELINE_SHA256,
    FROZEN_V2_POLICY_MANIFEST_SHA256,
    FROZEN_V2_R3_CONFIG_SHA256,
    MANIFEST_FILENAME,
    PERMUTATION_FILENAME,
    ROW_ERROR_FILENAME,
    SUBGROUP_FILENAME,
    SUMMARY_FILENAME,
    _capacity_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "reports" / "modeling" / "v2" / "diagnostics"
POLICY_PREDICTIONS = ROOT / "reports" / "modeling" / "v2" / "policy" / "policy_predictions.csv"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(filename: str) -> dict[str, object]:
    return json.loads((DIAGNOSTICS / filename).read_text(encoding="utf-8"))


def test_pretest_diagnostic_manifest_preserves_frozen_boundaries() -> None:
    manifest = _json(MANIFEST_FILENAME)
    assert manifest["stage"] == "pretest_interpretation_error_subgroup_diagnostics"
    assert manifest["r3_execution_config_sha256"] == FROZEN_V2_R3_CONFIG_SHA256
    assert manifest["policy_manifest_sha256"] == FROZEN_V2_POLICY_MANIFEST_SHA256
    assert manifest["persistence_manifest_sha256"] == FROZEN_V2_PERSISTENCE_MANIFEST_SHA256
    assert manifest["pipeline_sha256"] == FROZEN_V2_PIPELINE_SHA256
    assert manifest["diagnostic_partition"] == "policy_selection"
    assert manifest["single_operational_threshold_selected"] is False
    assert manifest["permutation_importance_may_drive_feature_selection"] is False
    assert manifest["final_test_target_accessed"] is False
    assert manifest["final_test_probabilities_generated"] is False


def test_pretest_diagnostic_artifact_hashes_and_sizes_match_manifest() -> None:
    manifest = _json(MANIFEST_FILENAME)
    artifacts = manifest["artifacts"]
    for filename, entry in artifacts.items():
        path = DIAGNOSTICS / filename
        assert _digest(path) == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]


def test_pretest_diagnostic_summary_matches_frozen_policy_population() -> None:
    summary = _json(SUMMARY_FILENAME)
    assert summary["diagnostic_partition"] == "policy_selection"
    assert summary["sample_size"] == 1063
    assert summary["positive_count"] == 92
    assert math.isclose(summary["prevalence"], 92 / 1063, rel_tol=0.0, abs_tol=1e-15)
    for metric in ("average_precision", "roc_auc", "brier_score", "log_loss"):
        assert math.isfinite(float(summary[metric]))
    assert summary["permutation_n_repeats"] == 20
    assert summary["permutation_random_state"] == 20260807
    assert summary["single_operational_threshold_selected"] is False
    assert summary["final_test_target_accessed"] is False
    assert summary["final_test_probabilities_generated"] is False


def test_permutation_importance_covers_exact_frozen_raw_features() -> None:
    table = pd.read_csv(DIAGNOSTICS / PERMUTATION_FILENAME)
    assert len(table) == 32
    assert set(table["feature"]) == set(V2_MODEL_FEATURE_COLUMNS)
    assert table["rank"].tolist() == list(range(1, 33))
    assert np.isfinite(table["importance_mean"]).all()
    assert np.isfinite(table["importance_std"]).all()


def test_subgroups_follow_frozen_feature_and_support_rules() -> None:
    table = pd.read_csv(DIAGNOSTICS / SUBGROUP_FILENAME)
    expected_features = {
        "patient_history_available",
        "reminder_sent_by_prediction_time",
        "visit_type",
        "booking_channel",
        "scheduled_weekday",
    }
    assert set(table["subgroup_feature"]) == expected_features

    supported = table["supported_for_quantitative_reporting"].astype(bool)
    expected_supported = (table["sample_size"] >= 100) & (table["positive_count"] >= 10)
    assert np.array_equal(supported.to_numpy(), expected_supported.to_numpy())

    metric_columns = ["average_precision", "brier_score", "log_loss"]
    assert table.loc[supported, metric_columns].notna().all().all()
    assert table.loc[~supported, metric_columns].isna().all().all()


def test_first_time_repeat_diagnostic_is_explicit_and_complete() -> None:
    table = pd.read_csv(DIAGNOSTICS / FIRST_REPEAT_FILENAME)
    assert set(table["cohort"]) == {"first_time", "repeat"}
    assert set(table["subgroup_feature"]) == {"patient_history_available"}
    assert int(table["sample_size"].sum()) == 1063
    assert int(table["positive_count"].sum()) == 92


def test_row_error_analysis_matches_policy_ids_and_targets() -> None:
    errors = pd.read_csv(DIAGNOSTICS / ROW_ERROR_FILENAME)
    policy = pd.read_csv(
        POLICY_PREDICTIONS,
        usecols=["appointment_id", "target"],
        dtype={"appointment_id": "int64", "target": "int8"},
    )
    assert len(errors) == 1063
    assert errors["appointment_id"].tolist() == policy["appointment_id"].tolist()
    assert errors["target"].astype("int8").tolist() == policy["target"].tolist()
    assert int(errors["target"].sum()) == 92
    for column in (
        "no_show_probability",
        "absolute_probability_error",
        "brier_contribution",
        "log_loss_contribution",
    ):
        assert np.isfinite(errors[column]).all()


def test_capacity_error_summary_uses_registered_fractions_and_floor_counts() -> None:
    table = pd.read_csv(DIAGNOSTICS / CAPACITY_FILENAME)
    assert np.allclose(table["capacity_fraction"], [0.05, 0.10, 0.20])
    assert table["selected_count"].tolist() == [53, 106, 212]
    assert np.all((table["precision"] >= 0.0) & (table["precision"] <= 1.0))
    assert np.all((table["recall"] >= 0.0) & (table["recall"] <= 1.0))


def test_capacity_helper_uses_probability_descending_then_appointment_id() -> None:
    result = _capacity_rows(
        appointment_id=np.array([20, 10, 30, 40], dtype=np.int64),
        target=np.array([0, 1, 1, 0], dtype=np.int8),
        probability=np.array([0.8, 0.8, 0.2, 0.1], dtype=np.float64),
        fractions=(0.5,),
    )
    row = result.iloc[0]
    assert row["selected_count"] == 2
    assert row["selected_positive_count"] == 1
    assert math.isclose(float(row["reported_threshold"]), 0.8)


def test_diagnostic_export_is_not_a_final_test_artifact() -> None:
    for filename in (
        PERMUTATION_FILENAME,
        SUBGROUP_FILENAME,
        FIRST_REPEAT_FILENAME,
        ROW_ERROR_FILENAME,
        CAPACITY_FILENAME,
        SUMMARY_FILENAME,
        MANIFEST_FILENAME,
    ):
        text = (DIAGNOSTICS / filename).read_text(encoding="utf-8")
        assert "final_test_target_accessed,true" not in text
        assert '"final_test_target_accessed": true' not in text
        assert '"final_test_probabilities_generated": true' not in text
