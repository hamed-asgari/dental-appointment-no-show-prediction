from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.modeling.v2_calibration import (
    CALIBRATION_METHODS,
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    RELIABILITY_COLUMNS,
    _choose_calibration_method,
    _load_ranking_selection,
    _window_feature_rows,
    run_v2_calibration_evaluation,
)
from src.modeling.v2_development import (
    FROZEN_V2_MODEL_CONFIG_SHA256,
    FROZEN_V2_MODEL_CONTRACT_SHA256,
)
from src.modeling.v2_rolling_origin_hashes import (
    FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_model_development.json"
CONTRACT_PATH = ROOT / "docs" / "v2_model_development_and_selection_contract.md"
RANKING_DIR = ROOT / "reports" / "modeling" / "v2"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result():
    return run_v2_calibration_evaluation()


def test_frozen_contract_hash_constants_remain_exact() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_MODEL_CONFIG_SHA256
    )
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_MODEL_CONTRACT_SHA256
    )


def test_calibration_methods_match_frozen_contract(config) -> None:
    assert CALIBRATION_METHODS == ("uncalibrated", "sigmoid", "isotonic")
    assert tuple(config["calibration"]["methods"]) == CALIBRATION_METHODS


def test_frozen_ranking_selection_is_the_calibrated_base_candidate() -> None:
    selection = _load_ranking_selection(RANKING_DIR)
    assert selection["selected_ranking_model"] == (
        FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL
    )
    assert selection["selected_ranking_model"] == "logistic_regression"
    assert selection["fallback_to_population_prior"] is False
    assert selection["final_test_target_accessed"] is False
    assert selection["final_test_probabilities_generated"] is False


def test_ranking_selection_loader_rejects_mutated_selection(tmp_path: Path) -> None:
    for filename in ("rolling_origin_manifest.json", "ranking_selection.json"):
        (tmp_path / filename).write_bytes((RANKING_DIR / filename).read_bytes())
    selection_path = tmp_path / "ranking_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selected_ranking_model"] = "random_forest"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="selected-model mismatch"):
        _load_ranking_selection(tmp_path)


def test_window_selector_refuses_final_test() -> None:
    features = load_frozen_v2_processed_feature_dataset(DEFAULT_V2_PROCESSED_DIR)
    with pytest.raises(PermissionError, match="final_test"):
        _window_feature_rows(
            features,
            partition="final_test",
            start=pd.Timestamp("2027-01-01"),
            end=pd.Timestamp("2027-02-01"),
        )


def _toy_metrics(
    *,
    uncalibrated_brier: float = 0.0700,
    uncalibrated_log_loss: float = 0.2600,
    sigmoid_brier: float = 0.0695,
    sigmoid_log_loss: float = 0.2610,
    isotonic_brier: float = 0.0680,
    isotonic_log_loss: float = 0.4000,
) -> pd.DataFrame:
    rows = []
    values = {
        "uncalibrated": (uncalibrated_brier, uncalibrated_log_loss),
        "sigmoid": (sigmoid_brier, sigmoid_log_loss),
        "isotonic": (isotonic_brier, isotonic_log_loss),
    }
    for method in CALIBRATION_METHODS:
        brier, loss = values[method]
        rows.append(
            {
                "method": method,
                "sample_size": 100,
                "positive_count": 10,
                "positive_rate": 0.10,
                "average_precision": 0.15,
                "roc_auc": 0.62,
                "brier_score": brier,
                "log_loss": loss,
                "calibration_intercept": 0.0,
                "calibration_slope": 1.0,
                "mean_predicted_probability": 0.10,
                "passes_log_loss_guardrail": False,
                "within_brier_indifference_margin": False,
                "selected": False,
            }
        )
    return pd.DataFrame(rows, columns=list(METRIC_COLUMNS))


def test_selection_prefers_uncalibrated_inside_brier_margin(config) -> None:
    metrics, selection = _choose_calibration_method(_toy_metrics(), config)
    assert selection["selected_calibration_method"] == "uncalibrated"
    assert metrics.loc[metrics["method"].eq("uncalibrated"), "selected"].item()


def test_selection_chooses_sigmoid_when_improvement_exceeds_margin(config) -> None:
    metrics, selection = _choose_calibration_method(
        _toy_metrics(sigmoid_brier=0.0680, isotonic_brier=0.0670),
        config,
    )
    assert selection["selected_calibration_method"] == "sigmoid"
    assert metrics.loc[metrics["method"].eq("sigmoid"), "selected"].item()


def test_log_loss_guardrail_excludes_bad_isotonic_candidate(config) -> None:
    metrics, selection = _choose_calibration_method(_toy_metrics(), config)
    isotonic = metrics.loc[metrics["method"].eq("isotonic")].iloc[0]
    assert not bool(isotonic["passes_log_loss_guardrail"])
    assert not bool(isotonic["selected"])
    assert selection["selected_calibration_method"] != "isotonic"


def test_real_result_uses_exact_strictly_mature_population_counts(result) -> None:
    selection = result.selection
    assert selection["base_training_rows"] == 10921
    assert selection["base_training_positive_count"] == 978
    assert selection["calibration_fit_rows"] == 738
    assert selection["calibration_fit_positive_count"] == 50
    assert selection["calibration_evaluation_rows"] == 328
    assert selection["calibration_evaluation_positive_count"] == 25


def test_real_metrics_have_exact_schema_and_method_order(result) -> None:
    assert tuple(result.metrics.columns) == METRIC_COLUMNS
    assert tuple(result.metrics["method"]) == CALIBRATION_METHODS
    assert result.metrics["sample_size"].eq(328).all()
    assert result.metrics["positive_count"].eq(25).all()


def test_real_selection_is_self_consistent(result) -> None:
    selected = str(result.selection["selected_calibration_method"])
    assert selected in CALIBRATION_METHODS
    assert int(result.metrics["selected"].sum()) == 1
    assert result.metrics.loc[result.metrics["selected"], "method"].item() == selected
    assert result.selection["selected_ranking_model"] == "logistic_regression"


def test_real_predictions_have_exact_schema_size_and_unique_keys(result) -> None:
    predictions = result.predictions
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 328 * len(CALIBRATION_METHODS)
    assert not predictions.duplicated(["method", "appointment_id"]).any()


def test_real_predictions_never_include_protected_final_test(result) -> None:
    assert result.predictions["evaluation_partition"].eq("calibration").all()
    assert not result.predictions["evaluation_partition"].eq("final_test").any()
    assert result.selection["final_test_target_accessed"] is False
    assert result.selection["final_test_probabilities_generated"] is False


def test_real_probabilities_and_metrics_are_finite_and_bounded(result) -> None:
    probability = result.predictions["no_show_probability"].to_numpy(dtype=float)
    assert np.isfinite(probability).all()
    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)
    metric_values = result.metrics.loc[
        :,
        [
            "average_precision",
            "roc_auc",
            "brier_score",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
        ],
    ].to_numpy(dtype=float)
    assert np.isfinite(metric_values).all()


def test_sigmoid_preserves_selected_base_ranking_on_evaluation(result) -> None:
    uncal = result.metrics.loc[result.metrics["method"].eq("uncalibrated")].iloc[0]
    sigmoid = result.metrics.loc[result.metrics["method"].eq("sigmoid")].iloc[0]
    assert sigmoid["average_precision"] == pytest.approx(
        uncal["average_precision"], abs=1e-12
    )
    assert sigmoid["roc_auc"] == pytest.approx(uncal["roc_auc"], abs=1e-12)


def test_real_reliability_curve_has_ten_equal_frequency_bins_per_method(result) -> None:
    curve = result.reliability_curve
    assert tuple(curve.columns) == RELIABILITY_COLUMNS
    assert len(curve) == 30
    assert set(curve["bin"]) == set(range(1, 11))
    counts = curve.groupby("method", sort=False)["bin_count"].sum()
    assert counts.to_dict() == {method: 328 for method in CALIBRATION_METHODS}


def test_real_calibration_fit_and_evaluation_chronology_is_frozen(result) -> None:
    selection = result.selection
    assert selection["base_fit_time"] == "2026-07-01T00:00:00"
    assert selection["calibrator_fit_time"] == "2026-09-01T00:00:00"
    assert selection["calibration_evaluation_label_cutoff"] == (
        "2026-10-01T00:00:00"
    )
    times = pd.to_datetime(result.predictions["prediction_time"])
    assert times.ge(pd.Timestamp("2026-09-01")).all()
    assert times.lt(pd.Timestamp("2026-10-01")).all()


def test_real_run_is_deterministic(result) -> None:
    repeated = run_v2_calibration_evaluation()
    pd.testing.assert_frame_equal(result.metrics, repeated.metrics)
    pd.testing.assert_frame_equal(result.predictions, repeated.predictions)
    pd.testing.assert_frame_equal(result.reliability_curve, repeated.reliability_curve)
    assert dict(result.selection) == dict(repeated.selection)


def test_processed_feature_artifact_remains_target_free_after_calibration(result) -> None:
    dataset = load_frozen_v2_processed_feature_dataset(DEFAULT_V2_PROCESSED_DIR)
    assert "target" not in dataset.columns
    assert len(dataset) == 21755


def test_real_selection_records_unchanged_frozen_input_identities(result) -> None:
    assert result.selection["model_config_sha256"] == FROZEN_V2_MODEL_CONFIG_SHA256
    assert result.selection["model_contract_sha256"] == (
        FROZEN_V2_MODEL_CONTRACT_SHA256
    )
