"""One-time protected Version 2 final-test evaluation from the sealed probability vector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.v2_targets import load_verified_v2_final_test_targets
from src.modeling.evaluation import evaluate_binary_probabilities
from src.modeling.v2_calibration import _calibration_intercept_slope
from src.modeling.v2_policy import (
    CAPACITY_FRACTIONS,
    FN_FP_COST_RATIOS,
    POLICY_SCENARIO_COLUMNS,
    _capacity_selection,
    _cost_threshold,
    _scenario_metrics,
    _threshold_selection,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V2_R3_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_r3_execution.json"
DEFAULT_V2_FINAL_TEST_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "final_test"
)
DEFAULT_V2_PERSISTENCE_DIR = _REPOSITORY_ROOT / "models" / "v2"

PROBABILITY_FILENAME = "final_test_probabilities.csv"
PROBABILITY_MANIFEST_FILENAME = "final_test_probability_manifest.json"

EVALUATION_PREDICTIONS_FILENAME = "final_test_evaluation_predictions.csv"
METRICS_FILENAME = "final_test_metrics.json"
SCENARIOS_FILENAME = "final_test_policy_scenarios.csv"
APP_DECISION_FILENAME = "final_test_app_decision.json"
EVALUATION_MANIFEST_FILENAME = "final_test_evaluation_manifest.json"

FROZEN_V2_R3_CONFIG_SHA256 = (
    "c0b259a4bb81790a30fd6e2c2fd2495e10869d700ae783196c5eb055db46f7a5"
)
FROZEN_V2_PROBABILITY_VECTOR_SHA256 = (
    "7a4af37da40c1515a6ee567dd12861b57cf08e2e65b516e3c7e3d2aa65237126"
)
FROZEN_V2_PROBABILITY_MANIFEST_SHA256 = (
    "412c75ff76882536eab0dc2ee5df4c3da8551c7b99fe83402ab5ac0f679b46e4"
)
FROZEN_V2_APPOINTMENT_ORDER_SHA256 = (
    "addb9ab672383b10976aec4eaa94f359bbbe4c0bf5b634b313265c659c9c3cd6"
)
FROZEN_V2_PERSISTENCE_METADATA_SHA256 = (
    "33eda2b123e592813008a004b4aa3f353ac1a2bda51ca5eaddb45954eeea6224"
)
FROZEN_V2_DIAGNOSTICS_MANIFEST_SHA256 = (
    "5a207b8a4984a203f64d1015c7a99b254db1108440dde71738abd3c936f9f8f2"
)

EXPECTED_FINAL_TEST_ROWS = 4343
PREDICTION_COLUMNS = (
    "appointment_id",
    "target",
    "no_show_probability",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load JSON artifact: {path}") from exc


def _load_pre_target_gate(
    *,
    final_test_dir: Path,
    persistence_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    config_path = Path(DEFAULT_V2_R3_CONFIG_PATH)
    if _sha256(config_path) != FROZEN_V2_R3_CONFIG_SHA256:
        raise ValueError("Frozen R3 execution config SHA-256 mismatch")
    config = _read_json(config_path)

    probability_path = Path(final_test_dir) / PROBABILITY_FILENAME
    probability_manifest_path = Path(final_test_dir) / PROBABILITY_MANIFEST_FILENAME
    if _sha256(probability_path) != FROZEN_V2_PROBABILITY_VECTOR_SHA256:
        raise ValueError("Sealed final-test probability vector SHA-256 mismatch")
    if _sha256(probability_manifest_path) != FROZEN_V2_PROBABILITY_MANIFEST_SHA256:
        raise ValueError("Sealed final-test probability manifest SHA-256 mismatch")

    probability_manifest = _read_json(probability_manifest_path)
    if probability_manifest["row_count"] != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError("Sealed final-test row count changed")
    if probability_manifest["appointment_order_sha256"] != (
        FROZEN_V2_APPOINTMENT_ORDER_SHA256
    ):
        raise RuntimeError("Sealed final-test appointment order changed")
    if probability_manifest["probability_metrics_computed"] is not False:
        raise RuntimeError("Target-derived metrics existed before final evaluation")
    if probability_manifest["final_test_probabilities_generated"] is not True:
        raise RuntimeError("Final-test probability generation was not sealed")
    if probability_manifest["final_test_target_accessed"] is not False:
        raise RuntimeError("Sealed probability checkpoint already records target access")

    protected = config["protected_final_test"]
    if protected["one_time_target_evaluation"] is not True:
        raise RuntimeError("Frozen R3 config no longer requires one-time evaluation")
    if protected["no_refit_after_probability_seal"] is not True:
        raise RuntimeError("Frozen R3 config no longer prevents post-seal refit")
    if protected["no_tuning_after_target_access"] is not True:
        raise RuntimeError("Frozen R3 config no longer prevents post-target tuning")
    if protected["target_access_requires_explicit_allow_test_true"] is not True:
        raise RuntimeError("Frozen R3 config target-access gate changed")

    metadata_path = (
        Path(persistence_dir) / "frozen_logistic_pipeline.metadata.json"
    )
    if _sha256(metadata_path) != FROZEN_V2_PERSISTENCE_METADATA_SHA256:
        raise ValueError("Frozen persistence metadata SHA-256 mismatch")
    persistence_metadata = _read_json(metadata_path)
    rows = int(persistence_metadata["base_training_rows"])
    positives = int(persistence_metadata["base_training_positive_count"])
    if rows != 10921 or positives != 978:
        raise RuntimeError("Frozen base-training prior identity changed")
    population_prior = float(positives / rows)
    return config, probability_manifest, population_prior


def _build_scenarios(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for capacity in CAPACITY_FRACTIONS:
        selected, expected_count, threshold = _capacity_selection(
            predictions,
            capacity,
        )
        if int(selected.sum()) != expected_count:
            raise RuntimeError("Final-test capacity selected-count mismatch")
        for ratio in FN_FP_COST_RATIOS:
            metrics = _scenario_metrics(
                predictions,
                selected,
                cost_ratio=ratio,
            )
            rows.append(
                {
                    "scenario_id": f"capacity_{capacity:.2f}_cost_{ratio:g}",
                    "scenario_family": "capacity_cost",
                    "capacity_fraction": float(capacity),
                    "false_negative_to_false_positive_cost_ratio": float(ratio),
                    "threshold": threshold,
                    **metrics,
                }
            )
    for ratio in FN_FP_COST_RATIOS:
        threshold = _cost_threshold(ratio)
        selected = _threshold_selection(predictions, threshold)
        metrics = _scenario_metrics(
            predictions,
            selected,
            cost_ratio=ratio,
        )
        rows.append(
            {
                "scenario_id": f"cost_threshold_{ratio:g}",
                "scenario_family": "cost_threshold",
                "capacity_fraction": float("nan"),
                "false_negative_to_false_positive_cost_ratio": float(ratio),
                "threshold": threshold,
                **metrics,
            }
        )
    scenarios = pd.DataFrame(rows, columns=list(POLICY_SCENARIO_COLUMNS))
    if len(scenarios) != 16:
        raise RuntimeError("Final-test descriptive policy grid must contain 16 rows")
    return scenarios


def _evaluate_app_gate(
    *,
    config: Mapping[str, Any],
    model_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> dict[str, Any]:
    gate = config["app_decision_gate"]
    rules = gate["appointment_level_risk_demo_requires_all"]

    ap_uplift = float(
        model_metrics["average_precision"]
        - baseline_metrics["average_precision"]
    )
    brier_delta = float(
        model_metrics["brier_score"]
        - baseline_metrics["brier_score"]
    )
    log_loss_delta = float(
        model_metrics["log_loss"]
        - baseline_metrics["log_loss"]
    )

    checks = {
        "average_precision_absolute_uplift_vs_population_prior": {
            "observed": ap_uplift,
            "required_minimum": float(
                rules[
                    "average_precision_absolute_uplift_vs_population_prior_minimum"
                ]
            ),
            "passes": bool(
                ap_uplift
                >= float(
                    rules[
                        "average_precision_absolute_uplift_vs_population_prior_minimum"
                    ]
                )
            ),
        },
        "roc_auc": {
            "observed": float(model_metrics["roc_auc"]),
            "required_minimum": float(rules["roc_auc_minimum"]),
            "passes": bool(
                float(model_metrics["roc_auc"])
                >= float(rules["roc_auc_minimum"])
            ),
        },
        "brier_score_vs_population_prior": {
            "model": float(model_metrics["brier_score"]),
            "baseline": float(baseline_metrics["brier_score"]),
            "delta_model_minus_baseline": brier_delta,
            "require_no_worse": bool(
                rules["brier_score_no_worse_than_population_prior"]
            ),
            "passes": bool(
                float(model_metrics["brier_score"])
                <= float(baseline_metrics["brier_score"])
            ),
        },
        "log_loss_vs_population_prior": {
            "model": float(model_metrics["log_loss"]),
            "baseline": float(baseline_metrics["log_loss"]),
            "delta_model_minus_baseline": log_loss_delta,
            "maximum_allowed_worsening": float(
                rules["log_loss_max_worsening_vs_population_prior"]
            ),
            "passes": bool(
                log_loss_delta
                <= float(rules["log_loss_max_worsening_vs_population_prior"])
            ),
        },
    }

    passes_all = bool(all(item["passes"] for item in checks.values()))
    if passes_all:
        app_type = "appointment_level_risk_demonstration"
    else:
        app_type = str(gate["otherwise"])

    return {
        "schema_version": "1.0.0",
        "phase": "R3",
        "decision_basis": "pre_frozen_final_test_app_gate",
        "checks": checks,
        "passes_all_appointment_level_risk_demo_requirements": passes_all,
        "selected_app_type": app_type,
        "final_test_threshold_selection_permitted": False,
        "model_or_calibration_change_permitted": False,
        "claims_scope": "synthetic_data_only",
    }


def run_one_time_v2_final_test_evaluation(
    *,
    final_test_dir: Path = DEFAULT_V2_FINAL_TEST_DIR,
    persistence_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
) -> dict[str, object]:
    """Perform the single explicit protected-target access and final evaluation."""

    final_test_dir = Path(final_test_dir)
    persistence_dir = Path(persistence_dir)
    config, probability_manifest, population_prior = _load_pre_target_gate(
        final_test_dir=final_test_dir,
        persistence_dir=persistence_dir,
    )

    probability_path = final_test_dir / PROBABILITY_FILENAME
    vector = pd.read_csv(
        probability_path,
        dtype={
            "appointment_id": "int64",
            "no_show_probability": "float64",
        },
        float_precision="round_trip",
    )

    access = load_verified_v2_final_test_targets(
        probability_path=probability_path,
        allow_test=True,
    )
    targets = access.target_table.reset_index(drop=True)

    if access.probability_seal.sha256 != FROZEN_V2_PROBABILITY_VECTOR_SHA256:
        raise RuntimeError("Protected accessor returned an unexpected vector identity")
    if access.probability_seal.row_count != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError("Protected accessor returned an unexpected row count")

    expected_ids = vector["appointment_id"].to_numpy(dtype=np.int64, copy=True)
    actual_ids = targets["appointment_id"].to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(expected_ids, actual_ids):
        raise RuntimeError("Protected target order differs from sealed vector order")
    if not targets["evaluation_partition"].astype("string").eq("final_test").all():
        raise RuntimeError("Protected target table contains a non-final-test row")

    target = targets["target"].astype("int8").reset_index(drop=True)
    probability = vector["no_show_probability"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    model_metrics = evaluate_binary_probabilities(target, probability)
    calibration_intercept, calibration_slope = _calibration_intercept_slope(
        target,
        probability,
    )

    baseline_probability = np.full(
        len(target),
        population_prior,
        dtype=np.float64,
    )
    baseline_metrics = evaluate_binary_probabilities(
        target,
        baseline_probability,
    )

    predictions = pd.DataFrame(
        {
            "appointment_id": expected_ids,
            "target": target.to_numpy(dtype=np.int8, copy=True),
            "no_show_probability": probability,
        },
        columns=list(PREDICTION_COLUMNS),
    )

    scenarios = _build_scenarios(predictions)

    metrics = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "one_time_protected_final_test_evaluation",
        "partition": "final_test",
        "sample_size": int(len(target)),
        "positive_count": int(target.sum()),
        "prevalence": float(target.mean()),
        "population_prior_probability": population_prior,
        "model": {
            **model_metrics,
            "calibration_intercept": float(calibration_intercept),
            "calibration_slope": float(calibration_slope),
            "mean_predicted_probability": float(probability.mean()),
        },
        "population_prior_baseline": {
            **baseline_metrics,
            "mean_predicted_probability": population_prior,
        },
        "average_precision_absolute_uplift_vs_population_prior": float(
            model_metrics["average_precision"]
            - baseline_metrics["average_precision"]
        ),
        "brier_delta_model_minus_population_prior": float(
            model_metrics["brier_score"]
            - baseline_metrics["brier_score"]
        ),
        "log_loss_delta_model_minus_population_prior": float(
            model_metrics["log_loss"]
            - baseline_metrics["log_loss"]
        ),
        "probability_vector_sha256": FROZEN_V2_PROBABILITY_VECTOR_SHA256,
        "probability_manifest_sha256": FROZEN_V2_PROBABILITY_MANIFEST_SHA256,
        "appointment_order_sha256": FROZEN_V2_APPOINTMENT_ORDER_SHA256,
        "selected_ranking_model": "logistic_regression",
        "selected_calibration_method": "uncalibrated",
        "single_operational_threshold_selected": False,
        "post_test_model_tuning_permitted": False,
        "final_test_probabilities_generated": True,
        "final_test_target_accessed": True,
        "target_access_count_this_evaluation_batch": 1,
    }

    app_decision = _evaluate_app_gate(
        config=config,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "scenarios": scenarios,
        "app_decision": app_decision,
        "pre_target_probability_manifest": probability_manifest,
    }


def export_one_time_v2_final_test_evaluation(
    result: Mapping[str, object],
    *,
    output_dir: Path = DEFAULT_V2_FINAL_TEST_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist the opened protected-test evaluation without altering the sealed vector."""

    output_dir = Path(output_dir)
    paths = {
        EVALUATION_PREDICTIONS_FILENAME: output_dir / EVALUATION_PREDICTIONS_FILENAME,
        METRICS_FILENAME: output_dir / METRICS_FILENAME,
        SCENARIOS_FILENAME: output_dir / SCENARIOS_FILENAME,
        APP_DECISION_FILENAME: output_dir / APP_DECISION_FILENAME,
        EVALUATION_MANIFEST_FILENAME: output_dir / EVALUATION_MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Protected final-test evaluation outputs already exist; "
            "the one-time evaluation may not be overwritten"
        )
    if overwrite:
        raise PermissionError(
            "Overwrite is prohibited after protected final-test target access"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = result["predictions"]
    scenarios = result["scenarios"]
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame")
    if not isinstance(scenarios, pd.DataFrame):
        raise TypeError("scenarios must be a DataFrame")

    predictions.to_csv(
        paths[EVALUATION_PREDICTIONS_FILENAME],
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    scenarios.to_csv(
        paths[SCENARIOS_FILENAME],
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )

    for filename, key in (
        (METRICS_FILENAME, "metrics"),
        (APP_DECISION_FILENAME, "app_decision"),
    ):
        paths[filename].write_text(
            json.dumps(dict(result[key]), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    artifacts: dict[str, dict[str, object]] = {}
    for filename in (
        EVALUATION_PREDICTIONS_FILENAME,
        METRICS_FILENAME,
        SCENARIOS_FILENAME,
        APP_DECISION_FILENAME,
    ):
        path = paths[filename]
        artifacts[filename] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    metrics = dict(result["metrics"])
    app_decision = dict(result["app_decision"])
    manifest = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "one_time_protected_final_test_evaluation",
        "probability_vector_sha256": FROZEN_V2_PROBABILITY_VECTOR_SHA256,
        "probability_manifest_sha256": FROZEN_V2_PROBABILITY_MANIFEST_SHA256,
        "appointment_order_sha256": FROZEN_V2_APPOINTMENT_ORDER_SHA256,
        "r3_execution_config_sha256": FROZEN_V2_R3_CONFIG_SHA256,
        "sample_size": int(metrics["sample_size"]),
        "positive_count": int(metrics["positive_count"]),
        "selected_app_type": app_decision["selected_app_type"],
        "target_access_method": "load_verified_v2_final_test_targets",
        "target_access_explicit_allow_test_true": True,
        "target_access_count_this_evaluation_batch": 1,
        "final_test_probabilities_generated": True,
        "final_test_target_accessed": True,
        "single_operational_threshold_selected": False,
        "post_test_model_tuning_permitted": False,
        "artifacts": artifacts,
    }
    paths[EVALUATION_MANIFEST_FILENAME].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Perform the explicit one-time Version 2 protected final-test "
            "target access and frozen evaluation."
        )
    )
    parser.add_argument(
        "--final-test-dir",
        type=Path,
        default=DEFAULT_V2_FINAL_TEST_DIR,
    )
    parser.add_argument(
        "--persistence-dir",
        type=Path,
        default=DEFAULT_V2_PERSISTENCE_DIR,
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required explicit acknowledgement for the one-time protected target access.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.allow_test is not True:
        raise PermissionError(
            "One-time protected final-test evaluation requires --allow-test"
        )

    result = run_one_time_v2_final_test_evaluation(
        final_test_dir=args.final_test_dir,
        persistence_dir=args.persistence_dir,
    )
    manifest = export_one_time_v2_final_test_evaluation(
        result,
        output_dir=args.final_test_dir,
        overwrite=False,
    )

    metrics = dict(result["metrics"])
    model = dict(metrics["model"])
    baseline = dict(metrics["population_prior_baseline"])
    decision = dict(result["app_decision"])

    print("One-time protected final-test evaluation completed.")
    print(
        "Rows / positives: "
        f"{metrics['sample_size']:,} / {metrics['positive_count']:,}"
    )
    print(f"Prevalence: {metrics['prevalence']:.12f}")
    print(f"Model Average Precision: {model['average_precision']:.12f}")
    print(f"Model ROC-AUC: {model['roc_auc']:.12f}")
    print(f"Model Brier score: {model['brier_score']:.12f}")
    print(f"Model log loss: {model['log_loss']:.12f}")
    print(
        "Calibration intercept / slope: "
        f"{model['calibration_intercept']:.12f} / "
        f"{model['calibration_slope']:.12f}"
    )
    print(
        "Population-prior AP / Brier / log loss: "
        f"{baseline['average_precision']:.12f} / "
        f"{baseline['brier_score']:.12f} / "
        f"{baseline['log_loss']:.12f}"
    )
    print(
        "AP uplift vs prior: "
        f"{metrics['average_precision_absolute_uplift_vs_population_prior']:.12f}"
    )
    print(f"Selected app type: {decision['selected_app_type']}")
    print(
        "App risk-demo gate passes all: "
        f"{str(decision['passes_all_appointment_level_risk_demo_requirements']).lower()}"
    )
    print(
        "Evaluation manifest SHA-256: "
        f"{_sha256(Path(args.final_test_dir) / EVALUATION_MANIFEST_FILENAME)}"
    )
    print("Final-test probabilities generated: true")
    print("Final-test target accessed: true")
    print("Single operational threshold selected: false")
    print("Post-test tuning permitted: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
