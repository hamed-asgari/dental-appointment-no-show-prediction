from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.v2_policy import (
    CAPACITY_FRACTIONS,
    FN_FP_COST_RATIOS,
    FROZEN_V2_POLICY_EXECUTION_SPEC_SHA256,
    POLICY_EXECUTION_SPEC_PATH,
    POLICY_SCENARIO_COLUMNS,
    V2PolicySensitivityResult,
    _build_policy_scenarios,
    _capacity_count,
    _capacity_selection,
    _cost_threshold,
    _scenario_metrics,
    _threshold_selection,
    _validate_policy_config,
    export_v2_policy_sensitivity_results,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_model_development.json"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _toy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "appointment_id": [30, 10, 20, 40, 50],
            "target": [1, 0, 1, 0, 1],
            "no_show_probability": [0.80, 0.80, 0.80, 0.40, 0.10],
        }
    )


def test_policy_execution_spec_hash_is_frozen() -> None:
    assert POLICY_EXECUTION_SPEC_PATH.is_file()
    assert sha256(POLICY_EXECUTION_SPEC_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_POLICY_EXECUTION_SPEC_SHA256
    )


def test_policy_grid_matches_frozen_config(config) -> None:
    _validate_policy_config(config)
    policy = config["policy_selection"]
    assert tuple(float(value) for value in policy["capacity_fractions"]) == (
        CAPACITY_FRACTIONS
    )
    assert tuple(
        float(value)
        for value in policy["false_negative_to_false_positive_cost_ratios"]
    ) == FN_FP_COST_RATIOS


def test_policy_validator_rejects_final_test_permission(config) -> None:
    mutated = json.loads(json.dumps(config))
    mutated["protected_final_test"]["target_access_permitted_during_r2"] = True
    with pytest.raises(RuntimeError, match="protects final_test"):
        _validate_policy_config(mutated)


def test_capacity_count_uses_floor() -> None:
    assert _capacity_count(101, 0.05) == 5
    assert _capacity_count(101, 0.10) == 10
    assert _capacity_count(101, 0.20) == 20


def test_capacity_count_rejects_zero_selected_rows() -> None:
    with pytest.raises(ValueError, match="fewer than one"):
        _capacity_count(3, 0.05)


def test_capacity_selection_uses_appointment_id_tie_break() -> None:
    predictions = _toy_predictions()
    selected, count, threshold = _capacity_selection(predictions, 0.40)
    selected_ids = set(
        predictions.loc[selected, "appointment_id"].astype(int).tolist()
    )
    assert count == 2
    assert threshold == pytest.approx(0.80)
    assert selected_ids == {10, 20}


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.0, 0.5),
        (2.0, 1.0 / 3.0),
        (5.0, 1.0 / 6.0),
        (10.0, 1.0 / 11.0),
    ],
)
def test_cost_threshold_matches_frozen_formula(ratio, expected) -> None:
    assert _cost_threshold(ratio) == pytest.approx(expected, abs=1e-15)


def test_threshold_selection_is_inclusive() -> None:
    predictions = pd.DataFrame(
        {
            "appointment_id": [1, 2, 3],
            "target": [0, 1, 1],
            "no_show_probability": [0.49, 0.50, 0.51],
        }
    )
    selected = _threshold_selection(predictions, 0.50)
    assert selected.tolist() == [False, True, True]


def test_scenario_metrics_use_frozen_relative_cost() -> None:
    predictions = pd.DataFrame(
        {
            "appointment_id": [1, 2, 3, 4],
            "target": [1, 1, 0, 0],
            "no_show_probability": [0.9, 0.2, 0.8, 0.1],
        }
    )
    selected = np.array([True, False, True, False])
    metrics = _scenario_metrics(predictions, selected, cost_ratio=5.0)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["scenario_cost"] == pytest.approx(6.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_policy_grid_contains_exactly_sixteen_scenarios(config) -> None:
    predictions = pd.DataFrame(
        {
            "appointment_id": list(range(1, 21)),
            "target": [0, 1] * 10,
            "no_show_probability": np.linspace(0.99, 0.01, 20),
        }
    )
    scenarios = _build_policy_scenarios(predictions, config)
    assert tuple(scenarios.columns) == POLICY_SCENARIO_COLUMNS
    assert len(scenarios) == 16
    assert tuple(scenarios["scenario_family"].iloc[:12]) == ("capacity_cost",) * 12
    assert tuple(scenarios["scenario_family"].iloc[12:]) == ("cost_threshold",) * 4


def test_capacity_membership_repeats_across_cost_ratios(config) -> None:
    predictions = pd.DataFrame(
        {
            "appointment_id": list(range(1, 101)),
            "target": ([0] * 90) + ([1] * 10),
            "no_show_probability": np.linspace(0.99, 0.01, 100),
        }
    )
    scenarios = _build_policy_scenarios(predictions, config)
    capacity = scenarios.loc[scenarios["scenario_family"].eq("capacity_cost")]
    for _fraction, rows in capacity.groupby("capacity_fraction", sort=False):
        assert rows["selected_count"].nunique() == 1
        assert rows["threshold"].nunique() == 1
        assert rows["true_positive"].nunique() == 1
        assert rows["false_positive"].nunique() == 1
        assert rows["true_negative"].nunique() == 1
        assert rows["false_negative"].nunique() == 1


def test_export_is_byte_deterministic_with_explicit_overwrite(
    config,
    tmp_path: Path,
) -> None:
    predictions = pd.DataFrame(
        {
            "appointment_id": list(range(1, 21)),
            "prediction_time": pd.date_range("2026-10-01", periods=20, freq="D"),
            "evaluation_partition": ["policy_selection"] * 20,
            "label_available_at": pd.date_range(
                "2026-10-02", periods=20, freq="D"
            ),
            "target": [0, 1] * 10,
            "no_show_probability": np.linspace(0.99, 0.01, 20),
        }
    )
    scenarios = _build_policy_scenarios(predictions, config)
    summary = {
        "selected_ranking_model": "logistic_regression",
        "selected_calibration_method": "uncalibrated",
        "base_training_rows": 10921,
        "base_training_positive_count": 978,
        "policy_decision_time": "2027-01-01T00:00:00",
        "policy_selection_rows": 20,
        "policy_selection_positive_count": 10,
        "policy_selection_positive_rate": 0.5,
        "scenario_count": 16,
    }
    result = V2PolicySensitivityResult(
        predictions=predictions,
        scenarios=scenarios,
        summary=summary,
    )

    export_v2_policy_sensitivity_results(result, output_dir=tmp_path)
    before = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    export_v2_policy_sensitivity_results(
        result,
        output_dir=tmp_path,
        overwrite=True,
    )
    after = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    assert after == before
    assert set(after) == {
        "policy_manifest.json",
        "policy_predictions.csv",
        "policy_scenarios.csv",
        "policy_summary.json",
    }


def test_export_refuses_overwrite_without_opt_in(config, tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "appointment_id": list(range(1, 21)),
            "prediction_time": pd.date_range("2026-10-01", periods=20, freq="D"),
            "evaluation_partition": ["policy_selection"] * 20,
            "label_available_at": pd.date_range(
                "2026-10-02", periods=20, freq="D"
            ),
            "target": [0, 1] * 10,
            "no_show_probability": np.linspace(0.99, 0.01, 20),
        }
    )
    scenarios = _build_policy_scenarios(predictions, config)
    summary = {
        "selected_ranking_model": "logistic_regression",
        "selected_calibration_method": "uncalibrated",
        "base_training_rows": 10921,
        "base_training_positive_count": 978,
        "policy_decision_time": "2027-01-01T00:00:00",
        "policy_selection_rows": 20,
        "policy_selection_positive_count": 10,
        "policy_selection_positive_rate": 0.5,
        "scenario_count": 16,
    }
    result = V2PolicySensitivityResult(
        predictions=predictions,
        scenarios=scenarios,
        summary=summary,
    )
    export_v2_policy_sensitivity_results(result, output_dir=tmp_path)
    with pytest.raises(ValueError, match="already exist"):
        export_v2_policy_sensitivity_results(result, output_dir=tmp_path)
