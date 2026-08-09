from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.v2_development import (
    CANDIDATE_NAMES,
    FOLD_METRIC_COLUMNS,
    FROZEN_V2_MODEL_CONFIG_SHA256,
    FROZEN_V2_MODEL_CONTRACT_SHA256,
    MACRO_SUMMARY_COLUMNS,
    POOLED_SUMMARY_COLUMNS,
    PREDICTION_COLUMNS,
    _build_macro_summary,
    _choose_ranking_model,
    _load_frozen_model_config,
    build_v2_candidate_estimator,
    run_v2_rolling_origin_development,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_model_development.json"
CONTRACT_PATH = ROOT / "docs" / "v2_model_development_and_selection_contract.md"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return _load_frozen_model_config()


@pytest.fixture(scope="module")
def result():
    return run_v2_rolling_origin_development()


def test_frozen_contract_hash_constants_match_files() -> None:
    import hashlib

    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_MODEL_CONFIG_SHA256
    )
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_MODEL_CONTRACT_SHA256
    )


def test_candidate_names_are_exactly_the_frozen_menu() -> None:
    assert CANDIDATE_NAMES == (
        "population_prior",
        "logistic_regression",
        "random_forest",
    )


def test_model_config_loader_rejects_mutation(tmp_path: Path) -> None:
    mutated = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["modeling_seed"] = 1
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="config SHA-256 mismatch"):
        _load_frozen_model_config(path, CONTRACT_PATH)


def test_feature_roles_cover_the_exact_v2_allowlist(config) -> None:
    roles = config["feature_roles"]
    combined = (
        tuple(roles["categorical"])
        + tuple(roles["boolean"])
        + tuple(roles["numeric"])
    )
    assert len(combined) == 32
    assert set(combined) == set(V2_MODEL_FEATURE_COLUMNS)


def test_logistic_pipeline_matches_frozen_hyperparameters(config) -> None:
    pipeline = build_v2_candidate_estimator("logistic_regression", config)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.C == 1.0
    assert classifier.penalty == "l2"
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 2000
    assert classifier.class_weight is None
    assert classifier.random_state == 20260807
    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = {
        name: transformer
        for name, transformer, _columns in preprocessor.transformers
    }
    assert "scaler" in transformers["numeric"].named_steps


def test_random_forest_pipeline_matches_frozen_hyperparameters(config) -> None:
    pipeline = build_v2_candidate_estimator("random_forest", config)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.n_estimators == 300
    assert classifier.max_depth == 10
    assert classifier.min_samples_leaf == 10
    assert classifier.min_samples_split == 20
    assert classifier.max_features == "sqrt"
    assert classifier.bootstrap is True
    assert classifier.class_weight is None
    assert classifier.random_state == 20260807
    assert classifier.n_jobs == 1
    preprocessor = pipeline.named_steps["preprocessor"]
    transformers = {
        name: transformer
        for name, transformer, _columns in preprocessor.transformers
    }
    assert "scaler" not in transformers["numeric"].named_steps


def test_unknown_candidate_is_rejected(config) -> None:
    with pytest.raises(ValueError, match="model_name must be"):
        build_v2_candidate_estimator("xgboost", config)


def _toy_fold_metrics(
    logistic_ap: tuple[float, float, float],
    logistic_roc: tuple[float, float, float],
    forest_ap: tuple[float, float, float],
    forest_roc: tuple[float, float, float],
) -> pd.DataFrame:
    rows = []
    prior_ap = (0.10, 0.10, 0.10)
    for index, fold in enumerate(("fold_1", "fold_2", "fold_3")):
        for model, aps, rocs in (
            ("population_prior", prior_ap, (0.50, 0.50, 0.50)),
            ("logistic_regression", logistic_ap, logistic_roc),
            ("random_forest", forest_ap, forest_roc),
        ):
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "fit_time": pd.Timestamp("2025-01-01"),
                    "validation_label_cutoff": pd.Timestamp("2025-07-01"),
                    "training_rows": 100,
                    "training_positive_count": 10,
                    "training_prevalence": 0.10,
                    "sample_size": 50,
                    "positive_count": 5,
                    "positive_rate": 0.10,
                    "average_precision": aps[index],
                    "roc_auc": rocs[index],
                    "brier_score": 0.09,
                    "log_loss": 0.32,
                }
            )
    return pd.DataFrame(rows, columns=list(FOLD_METRIC_COLUMNS))


def test_frozen_usefulness_gate_falls_back_to_population_prior(config) -> None:
    fold_metrics = _toy_fold_metrics(
        logistic_ap=(0.101, 0.100, 0.099),
        logistic_roc=(0.51, 0.51, 0.51),
        forest_ap=(0.102, 0.101, 0.100),
        forest_roc=(0.51, 0.51, 0.51),
    )
    macro = _build_macro_summary(fold_metrics, config)
    selection = _choose_ranking_model(macro)
    assert not macro["passes_minimum_usefulness_gate"].any()
    assert selection["selected_ranking_model"] == "population_prior"
    assert selection["fallback_to_population_prior"] is True


def test_frozen_usefulness_gate_and_order_can_select_logistic(config) -> None:
    fold_metrics = _toy_fold_metrics(
        logistic_ap=(0.14, 0.13, 0.12),
        logistic_roc=(0.61, 0.60, 0.59),
        forest_ap=(0.13, 0.12, 0.11),
        forest_roc=(0.60, 0.59, 0.58),
    )
    macro = _build_macro_summary(fold_metrics, config)
    selection = _choose_ranking_model(macro)
    assert macro.loc[
        macro["model"].eq("logistic_regression"),
        "passes_minimum_usefulness_gate",
    ].item()
    assert selection["selected_ranking_model"] == "logistic_regression"
    assert selection["fallback_to_population_prior"] is False


def test_real_result_has_exact_fold_metric_schema_and_grid(result) -> None:
    assert tuple(result.fold_metrics.columns) == FOLD_METRIC_COLUMNS
    assert len(result.fold_metrics) == 9
    observed = set(
        zip(
            result.fold_metrics["fold"],
            result.fold_metrics["model"],
            strict=True,
        )
    )
    expected = {
        (fold, model)
        for fold in ("fold_1", "fold_2", "fold_3")
        for model in CANDIDATE_NAMES
    }
    assert observed == expected


def test_real_result_uses_exact_strictly_mature_row_counts(result) -> None:
    one_per_fold = (
        result.fold_metrics.loc[
            result.fold_metrics["model"].eq("population_prior")
        ]
        .set_index("fold")
        .sort_index()
    )
    assert one_per_fold["training_rows"].to_dict() == {
        "fold_1": 4452,
        "fold_2": 6600,
        "fold_3": 8839,
    }
    assert one_per_fold["training_positive_count"].to_dict() == {
        "fold_1": 383,
        "fold_2": 601,
        "fold_3": 785,
    }
    assert one_per_fold["sample_size"].to_dict() == {
        "fold_1": 2133,
        "fold_2": 2222,
        "fold_3": 2073,
    }
    assert one_per_fold["positive_count"].to_dict() == {
        "fold_1": 217,
        "fold_2": 182,
        "fold_3": 191,
    }


def test_real_predictions_have_exact_schema_size_and_unique_keys(result) -> None:
    predictions = result.predictions
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == (2133 + 2222 + 2073) * 3
    assert not predictions.duplicated(["fold", "model", "appointment_id"]).any()


def test_real_predictions_never_include_protected_final_test(result) -> None:
    assert not result.predictions["evaluation_partition"].eq("final_test").any()
    assert result.selection["final_test_target_accessed"] is False
    assert result.selection["final_test_probabilities_generated"] is False


def test_population_prior_is_constant_within_each_fold(result) -> None:
    prior = result.predictions.loc[
        result.predictions["model"].eq("population_prior")
    ]
    assert prior.groupby("fold")["no_show_probability"].nunique().eq(1).all()


def test_nonconstant_candidates_really_rank_rows(result) -> None:
    predictions = result.predictions
    for model in ("logistic_regression", "random_forest"):
        subset = predictions.loc[predictions["model"].eq(model)]
        assert subset.groupby("fold")["no_show_probability"].nunique().gt(1).all()


def test_all_real_probabilities_are_finite_and_bounded(result) -> None:
    values = result.predictions["no_show_probability"].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


def test_all_real_threshold_free_metrics_are_finite(result) -> None:
    metrics = result.fold_metrics.loc[
        :,
        ["average_precision", "roc_auc", "brier_score", "log_loss"],
    ].to_numpy(dtype=np.float64)
    assert np.isfinite(metrics).all()


def test_real_macro_summary_schema_and_gate_are_self_consistent(result) -> None:
    macro = result.macro_summary
    assert tuple(macro.columns) == MACRO_SUMMARY_COLUMNS
    assert tuple(macro["model"]) == CANDIDATE_NAMES
    selected = result.selection["selected_ranking_model"]
    if result.selection["fallback_to_population_prior"]:
        assert selected == "population_prior"
        assert not macro["passes_minimum_usefulness_gate"].any()
    else:
        row = macro.loc[macro["model"].eq(selected)]
        assert row["passes_minimum_usefulness_gate"].item()


def test_real_pooled_summary_schema_and_counts(result) -> None:
    pooled = result.pooled_summary
    assert tuple(pooled.columns) == POOLED_SUMMARY_COLUMNS
    assert tuple(pooled["model"]) == CANDIDATE_NAMES
    assert pooled["sample_size"].eq(6428).all()
    assert pooled["positive_count"].eq(590).all()


def test_real_run_is_deterministic(result) -> None:
    repeated = run_v2_rolling_origin_development()
    pd.testing.assert_frame_equal(result.fold_metrics, repeated.fold_metrics)
    pd.testing.assert_frame_equal(result.macro_summary, repeated.macro_summary)
    pd.testing.assert_frame_equal(result.pooled_summary, repeated.pooled_summary)
    pd.testing.assert_frame_equal(result.predictions, repeated.predictions)
    assert dict(result.selection) == dict(repeated.selection)


def test_processed_feature_artifact_remains_target_free_after_modeling(result) -> None:
    dataset = load_frozen_v2_processed_feature_dataset(
        DEFAULT_V2_PROCESSED_DIR
    )
    assert "target" not in dataset.columns
    assert len(dataset) == 21755


def test_contract_identity_is_unchanged_after_real_modeling(result) -> None:
    import hashlib

    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_MODEL_CONFIG_SHA256
    )
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == (
        FROZEN_V2_MODEL_CONTRACT_SHA256
    )
