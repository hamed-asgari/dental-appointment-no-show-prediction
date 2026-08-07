from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.evaluation import evaluate_binary_probabilities
from src.modeling.v2_calibration import _calibration_intercept_slope
from src.modeling.v2_final_test_evaluation import (
    APP_DECISION_FILENAME,
    EVALUATION_MANIFEST_FILENAME,
    EVALUATION_PREDICTIONS_FILENAME,
    FROZEN_V2_APPOINTMENT_ORDER_SHA256,
    FROZEN_V2_PROBABILITY_MANIFEST_SHA256,
    FROZEN_V2_PROBABILITY_VECTOR_SHA256,
    METRICS_FILENAME,
    SCENARIOS_FILENAME,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "reports" / "modeling" / "v2" / "final_test"
MANIFEST_PATH = FINAL_DIR / EVALUATION_MANIFEST_FILENAME
METRICS_PATH = FINAL_DIR / METRICS_FILENAME
PREDICTIONS_PATH = FINAL_DIR / EVALUATION_PREDICTIONS_FILENAME
SCENARIOS_PATH = FINAL_DIR / SCENARIOS_FILENAME
APP_DECISION_PATH = FINAL_DIR / APP_DECISION_FILENAME
PROBABILITY_MANIFEST = FINAL_DIR / "final_test_probability_manifest.json"
ENGINE_SOURCE = ROOT / "src" / "modeling" / "v2_final_test_evaluation.py"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_evaluation_manifest_records_exact_opened_gate() -> None:
    manifest = _json(MANIFEST_PATH)
    assert manifest["stage"] == "one_time_protected_final_test_evaluation"
    assert manifest["probability_vector_sha256"] == FROZEN_V2_PROBABILITY_VECTOR_SHA256
    assert manifest["probability_manifest_sha256"] == (
        FROZEN_V2_PROBABILITY_MANIFEST_SHA256
    )
    assert manifest["appointment_order_sha256"] == FROZEN_V2_APPOINTMENT_ORDER_SHA256
    assert manifest["sample_size"] == 4343
    assert manifest["target_access_method"] == "load_verified_v2_final_test_targets"
    assert manifest["target_access_explicit_allow_test_true"] is True
    assert manifest["target_access_count_this_evaluation_batch"] == 1
    assert manifest["final_test_probabilities_generated"] is True
    assert manifest["final_test_target_accessed"] is True
    assert manifest["single_operational_threshold_selected"] is False
    assert manifest["post_test_model_tuning_permitted"] is False


def test_evaluation_manifest_artifact_hashes_and_sizes() -> None:
    manifest = _json(MANIFEST_PATH)
    for filename, entry in manifest["artifacts"].items():
        path = FINAL_DIR / filename
        assert _digest(path) == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]


def test_committed_predictions_have_exact_schema_and_population() -> None:
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={
            "appointment_id": "int64",
            "target": "int8",
            "no_show_probability": "float64",
        },
        float_precision="round_trip",
    )
    assert list(predictions.columns) == [
        "appointment_id",
        "target",
        "no_show_probability",
    ]
    assert len(predictions) == 4343
    assert predictions["appointment_id"].is_unique
    assert set(predictions["target"].unique()) == {0, 1}
    assert np.isfinite(predictions["no_show_probability"]).all()


def test_committed_metrics_recompute_from_committed_predictions() -> None:
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={
            "appointment_id": "int64",
            "target": "int8",
            "no_show_probability": "float64",
        },
        float_precision="round_trip",
    )
    target = predictions["target"].astype("int8")
    probability = predictions["no_show_probability"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    observed = evaluate_binary_probabilities(target, probability)
    metrics = _json(METRICS_PATH)
    model = metrics["model"]
    for key in ("average_precision", "roc_auc", "brier_score", "log_loss"):
        assert math.isclose(
            float(model[key]),
            observed[key],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    intercept, slope = _calibration_intercept_slope(target, probability)
    assert math.isclose(
        float(model["calibration_intercept"]),
        intercept,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(model["calibration_slope"]),
        slope,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_population_prior_baseline_comes_from_frozen_base_fit() -> None:
    metrics = _json(METRICS_PATH)
    prior = 978 / 10921
    assert math.isclose(
        float(metrics["population_prior_probability"]),
        prior,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    predictions = pd.read_csv(PREDICTIONS_PATH)
    target = predictions["target"].astype("int8")
    baseline = np.full(len(target), prior, dtype=np.float64)
    recomputed = evaluate_binary_probabilities(target, baseline)
    recorded = metrics["population_prior_baseline"]
    for key in ("average_precision", "roc_auc", "brier_score", "log_loss"):
        assert math.isclose(
            float(recorded[key]),
            recomputed[key],
            rel_tol=0.0,
            abs_tol=1e-15,
        )


def test_final_policy_scenarios_reuse_frozen_grid_without_selecting_one() -> None:
    scenarios = pd.read_csv(SCENARIOS_PATH)
    assert len(scenarios) == 16
    assert set(scenarios["scenario_family"]) == {
        "capacity_cost",
        "cost_threshold",
    }
    capacities = sorted(
        scenarios.loc[
            scenarios["scenario_family"].eq("capacity_cost"),
            "capacity_fraction",
        ].unique()
    )
    assert np.allclose(capacities, [0.05, 0.10, 0.20])
    ratios = sorted(
        scenarios["false_negative_to_false_positive_cost_ratio"].unique()
    )
    assert np.allclose(ratios, [1.0, 2.0, 5.0, 10.0])
    assert _json(METRICS_PATH)["single_operational_threshold_selected"] is False


def test_app_decision_exactly_replays_prefrozen_gate() -> None:
    metrics = _json(METRICS_PATH)
    decision = _json(APP_DECISION_PATH)
    model = metrics["model"]
    baseline = metrics["population_prior_baseline"]

    checks = decision["checks"]
    assert math.isclose(
        float(
            checks[
                "average_precision_absolute_uplift_vs_population_prior"
            ]["observed"]
        ),
        float(model["average_precision"]) - float(baseline["average_precision"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    expected_pass = (
        float(model["average_precision"]) - float(baseline["average_precision"])
        >= 0.005
        and float(model["roc_auc"]) >= 0.52
        and float(model["brier_score"]) <= float(baseline["brier_score"])
        and float(model["log_loss"]) - float(baseline["log_loss"]) <= 0.005
    )
    assert (
        decision["passes_all_appointment_level_risk_demo_requirements"]
        is expected_pass
    )
    expected_app = (
        "appointment_level_risk_demonstration"
        if expected_pass
        else "transparent_model_evaluation_dashboard"
    )
    assert decision["selected_app_type"] == expected_app
    assert decision["final_test_threshold_selection_permitted"] is False
    assert decision["model_or_calibration_change_permitted"] is False


def test_probability_seal_remains_immutable_pre_target_evidence() -> None:
    probability_manifest = _json(PROBABILITY_MANIFEST)
    assert _digest(PROBABILITY_MANIFEST) == FROZEN_V2_PROBABILITY_MANIFEST_SHA256
    assert probability_manifest["final_test_probabilities_generated"] is True
    assert probability_manifest["final_test_target_accessed"] is False
    assert probability_manifest["probability_metrics_computed"] is False


def test_tests_do_not_reinvoke_real_protected_accessor() -> None:
    text = ENGINE_SOURCE.read_text(encoding="utf-8")
    engine_allow_test = "allow_" + "test=True"
    assert text.count(engine_allow_test) == 1
    test_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_accessor_call = "load_verified_v2_final_test_" + "targets("
    forbidden_allow_test = "allow_" + "test=True"
    assert forbidden_accessor_call not in test_text
    assert forbidden_allow_test not in test_text
