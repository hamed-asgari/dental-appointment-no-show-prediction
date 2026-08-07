from __future__ import annotations

import json
from pathlib import Path

from src.features.schema import V2_MODEL_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_model_development.json"
CONTRACT_PATH = ROOT / "docs" / "v2_model_development_and_selection_contract.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
DOCS_INDEX = ROOT / "docs" / "README.md"
CONFIG_INDEX = ROOT / "configs" / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"

EXPECTED_DATASET_SHA256 = (
    "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53"
)
EXPECTED_MANIFEST_SHA256 = (
    "2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073"
)
EXPECTED_FINGERPRINT = (
    "0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_contract_files_exist_and_are_explicitly_pre_metric() -> None:
    assert CONFIG_PATH.is_file()
    assert CONTRACT_PATH.is_file()
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "Frozen before any Version 2 recovered-model metric is computed." in text
    assert "It does not record model performance." in text


def test_processed_artifact_identity_is_frozen() -> None:
    artifact = _config()["processed_artifact"]
    assert artifact == {
        "path": "data/processed/v2/v2_feature_dataset.csv",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "fingerprint": EXPECTED_FINGERPRINT,
        "model_feature_count": 32,
        "target_included": False,
    }


def test_feature_roles_are_disjoint_and_cover_exact_allowlist() -> None:
    roles = _config()["feature_roles"]
    categorical = tuple(roles["categorical"])
    boolean = tuple(roles["boolean"])
    numeric = tuple(roles["numeric"])
    assert not set(categorical) & set(boolean)
    assert not set(categorical) & set(numeric)
    assert not set(boolean) & set(numeric)
    combined = categorical + boolean + numeric
    assert len(combined) == 32
    assert set(combined) == set(V2_MODEL_FEATURE_COLUMNS)


def test_preprocessing_forbids_search_and_feature_selection() -> None:
    preprocessing = _config()["preprocessing"]
    assert preprocessing["feature_selection"] == "none"
    assert preprocessing["hyperparameter_search"] == "none"
    assert preprocessing["categorical"]["handle_unknown"] == "ignore"
    assert preprocessing["numeric"]["imputer"] == "median_fit_on_training_only"
    assert (
        preprocessing["logistic_numeric_scaler"]
        == "standard_scaler_fit_on_training_only"
    )


def test_candidate_menu_and_hyperparameters_are_exact() -> None:
    candidates = _config()["candidates"]
    assert tuple(candidates) == (
        "population_prior",
        "logistic_regression",
        "random_forest",
    )
    assert candidates["logistic_regression"]["C"] == 1.0
    assert candidates["logistic_regression"]["solver"] == "lbfgs"
    assert candidates["random_forest"]["n_estimators"] == 300
    assert candidates["random_forest"]["max_depth"] == 10
    assert candidates["random_forest"]["min_samples_leaf"] == 10
    assert candidates["random_forest"]["random_state"] == 20260807
    assert candidates["random_forest"]["n_jobs"] == 1


def test_rolling_origin_schedule_is_exact_and_strict() -> None:
    folds = _config()["rolling_origin"]
    assert [fold["name"] for fold in folds] == ["fold_1", "fold_2", "fold_3"]
    assert [fold["fit_time"] for fold in folds] == [
        "2025-01-01T00:00:00",
        "2025-07-01T00:00:00",
        "2026-01-01T00:00:00",
    ]
    assert [fold["validation_label_cutoff"] for fold in folds] == [
        "2025-07-01T00:00:00",
        "2026-01-01T00:00:00",
        "2026-07-01T00:00:00",
    ]
    text = " ".join(CONTRACT_PATH.read_text(encoding="utf-8").split())
    assert "label_available_at < fit_time" in text
    assert "late labels at an exact boundary are excluded" in text


def test_ranking_selection_gate_and_fallback_are_frozen() -> None:
    selection = _config()["ranking_selection"]
    gate = selection["minimum_usefulness_gate"]
    assert gate == {
        "mean_average_precision_absolute_uplift_vs_prior": 0.005,
        "mean_roc_auc_minimum": 0.52,
        "minimum_folds_with_positive_ap_uplift_vs_prior": 2,
    }
    assert (
        selection["fallback_if_no_nonconstant_candidate_passes_gate"]
        == "population_prior"
    )


def test_calibration_chronology_methods_and_selection_are_frozen() -> None:
    calibration = _config()["calibration"]
    assert calibration["base_refit_time"] == "2026-07-01T00:00:00"
    assert calibration["calibrator_fit_time"] == "2026-09-01T00:00:00"
    assert calibration["calibration_evaluation_label_cutoff"] == (
        "2026-10-01T00:00:00"
    )
    assert calibration["methods"] == ["uncalibrated", "sigmoid", "isotonic"]
    assert calibration["selection"]["primary"] == "lowest_brier_score"
    assert calibration["selection"]["brier_indifference_margin"] == 0.001
    assert calibration["base_estimator_refit_after_calibration"] is False


def test_policy_selection_is_sensitivity_only_and_not_default_point_five() -> None:
    policy = _config()["policy_selection"]
    assert policy["decision_time"] == "2027-01-01T00:00:00"
    assert policy["capacity_fractions"] == [0.05, 0.10, 0.20]
    assert policy["false_negative_to_false_positive_cost_ratios"] == [
        1.0,
        2.0,
        5.0,
        10.0,
    ]
    assert policy["single_operational_threshold_selected"] is False


def test_protected_final_test_is_forbidden_throughout_r2() -> None:
    protected = _config()["protected_final_test"]
    assert protected["target_access_permitted_during_r2"] is False
    assert protected["probability_vector_generation_permitted_during_r2"] is False
    assert protected["metric_computation_permitted_during_r2"] is False
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "no final-test probability vector may be generated" in text
    assert "no final-test metric may be computed" in text


def test_reporting_contract_requires_probability_and_calibration_metrics() -> None:
    selection_metrics = set(_config()["ranking_selection"]["required_metrics"])
    assert {
        "average_precision",
        "roc_auc",
        "brier_score",
        "log_loss",
        "sample_size",
        "positive_count",
    } <= selection_metrics
    calibration = _config()["calibration"]
    assert {"calibration_intercept", "calibration_slope"} <= set(
        calibration["required_metrics"]
    )
    assert _config()["reporting"]["calibration_bins"] == 10


def test_documentation_indexes_record_contract_without_model_results() -> None:
    docs = DOCS_INDEX.read_text(encoding="utf-8")
    configs = CONFIG_INDEX.read_text(encoding="utf-8")
    recovery = RECOVERY_PLAN.read_text(encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    assert "v2_model_development_and_selection_contract.md" in docs
    assert "v2_model_development.json" in configs
    r2 = recovery[recovery.index("## Phase R2"):recovery.index("## Phase R3")]
    assert (
        "**Status: rolling-origin ranking and chronological calibration complete; "
        "policy analysis pending.**"
    ) in r2
    assert "v2_r2_rolling_origin_results.md" in r2
    assert "v2_r2_calibration_results.md" in r2
    assert "protected 2027 final-test target remains closed" in r2
    assert "Frozen the Version 2 model-development and selection contract" in changelog
