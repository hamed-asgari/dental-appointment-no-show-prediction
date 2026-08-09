from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v2_r3_execution.json"
CONTRACT = ROOT / "docs" / "v2_r3_execution_contract.md"
POLICY_MANIFEST = (
    ROOT / "reports" / "modeling" / "v2" / "policy" / "policy_manifest.json"
)

EXPECTED_CONFIG_SHA256 = "c0b259a4bb81790a30fd6e2c2fd2495e10869d700ae783196c5eb055db46f7a5"
EXPECTED_POLICY_MANIFEST_SHA256 = (
    "33391bc9295c6a9d93bb8797e8e3836fb9af45062f1b431a4debc9bc7822e4a4"
)


def test_r3_config_identity() -> None:
    assert sha256(CONFIG.read_bytes()).hexdigest() == EXPECTED_CONFIG_SHA256


def test_r3_contract_records_config_identity() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert EXPECTED_CONFIG_SHA256 in text
    assert "Frozen before any R3 protected-final-test probability vector" in text


def test_r3_inherits_frozen_r2_choices() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["upstream"]["r2_closure_commit"] == "7fdcf40"
    assert config["upstream"]["selected_ranking_model"] == "logistic_regression"
    assert config["upstream"]["selected_calibration_method"] == "uncalibrated"
    assert config["upstream"]["single_operational_threshold_selected"] is False
    assert config["frozen_model"]["base_estimator_refit_after_calibration"] is False
    assert config["frozen_model"]["refit_on_calibration_or_policy_selection_data"] is False


def test_r3_pretest_diagnostics_exclude_final_test() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    diagnostics = config["pre_test_diagnostics"]
    assert diagnostics["primary_partition"] == "policy_selection"
    assert diagnostics["final_test_permitted"] is False
    assert diagnostics["permutation_importance"]["may_drive_feature_selection"] is False
    assert diagnostics["minimum_subgroup_rows"] == 100
    assert diagnostics["minimum_subgroup_positive_count"] == 10


def test_r3_probability_vector_must_be_sealed_before_target_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protected = config["protected_final_test"]
    assert protected["expected_feature_rows"] == 4343
    assert protected["probability_columns"] == [
        "appointment_id",
        "no_show_probability",
    ]
    assert protected["exact_appointment_order_required"] is True
    assert protected["probability_sha256_seal_required_before_target_access"] is True
    assert protected["probability_commit_and_ci_green_required_before_target_access"] is True
    assert protected["target_access_requires_explicit_allow_test_true"] is True
    assert protected["one_time_target_evaluation"] is True
    assert protected["probabilities_generated_at_contract_freeze"] is False
    assert protected["target_accessed_at_contract_freeze"] is False


def test_r3_app_gate_is_fixed_before_protected_test() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gate = config["app_decision_gate"]
    rules = gate["appointment_level_risk_demo_requires_all"]
    assert rules["average_precision_absolute_uplift_vs_population_prior_minimum"] == 0.005
    assert rules["roc_auc_minimum"] == 0.52
    assert rules["brier_score_no_worse_than_population_prior"] is True
    assert rules["log_loss_max_worsening_vs_population_prior"] == 0.005
    assert gate["otherwise"] == "transparent_model_evaluation_dashboard"
    assert gate["final_test_threshold_selection_permitted"] is False


def test_r3_contract_freeze_does_not_change_policy_identity() -> None:
    assert sha256(POLICY_MANIFEST.read_bytes()).hexdigest() == (
        EXPECTED_POLICY_MANIFEST_SHA256
    )
