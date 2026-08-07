"""Version 2 chronological probability calibration under the frozen R2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.data.build_v2_dataset import load_verified_v2_raw_tables
from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.data.v2_targets import build_mature_v2_target_table
from src.modeling.evaluation import evaluate_binary_probabilities
from src.modeling.v2_development import (
    FROZEN_V2_MODEL_CONFIG_SHA256,
    FROZEN_V2_MODEL_CONTRACT_SHA256,
    _load_frozen_model_config,
    _select_features_by_ids,
    build_v2_candidate_estimator,
)
from src.modeling.v2_rolling_origin_hashes import (
    FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
    FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_RANKING_DIR = _REPOSITORY_ROOT / "reports" / "modeling" / "v2"
DEFAULT_V2_CALIBRATION_OUTPUT_DIR = DEFAULT_V2_RANKING_DIR / "calibration"

EXPECTED_PROCESSED_DATASET_SHA256 = (
    "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53"
)
EXPECTED_PROCESSED_MANIFEST_SHA256 = (
    "2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073"
)
EXPECTED_PROCESSED_FINGERPRINT = (
    "0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787"
)

CALIBRATION_METHODS = (
    "uncalibrated",
    "sigmoid",
    "isotonic",
)

METRIC_COLUMNS = (
    "method",
    "sample_size",
    "positive_count",
    "positive_rate",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "mean_predicted_probability",
    "passes_log_loss_guardrail",
    "within_brier_indifference_margin",
    "selected",
)

PREDICTION_COLUMNS = (
    "method",
    "appointment_id",
    "prediction_time",
    "evaluation_partition",
    "target",
    "no_show_probability",
)

RELIABILITY_COLUMNS = (
    "method",
    "bin",
    "bin_count",
    "positive_count",
    "mean_predicted_probability",
    "observed_no_show_rate",
    "min_predicted_probability",
    "max_predicted_probability",
)


@dataclass(frozen=True)
class V2CalibrationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    reliability_curve: pd.DataFrame
    selection: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ranking_selection(
    ranking_dir: Path = DEFAULT_V2_RANKING_DIR,
) -> dict[str, Any]:
    ranking_dir = Path(ranking_dir)
    manifest_path = ranking_dir / "rolling_origin_manifest.json"
    selection_path = ranking_dir / "ranking_selection.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise ValueError("Frozen rolling-origin ranking artifacts are missing")
    if _sha256(manifest_path) != FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256:
        raise ValueError("Frozen rolling-origin manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for source in (manifest, selection):
        if source.get("final_test_target_accessed") is not False:
            raise RuntimeError("Ranking artifact exposed protected final_test target")
        if source.get("final_test_probabilities_generated") is not False:
            raise RuntimeError("Ranking artifact generated final_test probabilities")
    selected_manifest = manifest.get("selected_ranking_model")
    selected_selection = selection.get("selected_ranking_model")
    if selected_manifest != selected_selection:
        raise ValueError("Ranking artifact selected-model mismatch")
    if selected_manifest != FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL:
        raise ValueError("Ranking selection no longer matches frozen identity")
    if bool(selection.get("fallback_to_population_prior")):
        raise RuntimeError(
            "Frozen ranking selected population_prior; non-constant calibration is unavailable"
        )
    return selection


def _window_feature_rows(
    feature_dataset: pd.DataFrame,
    *,
    partition: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if partition == "final_test":
        raise PermissionError("Calibration may not use final_test")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if not start < end:
        raise ValueError("Calibration window start must be before end")
    prediction_time = pd.to_datetime(
        feature_dataset["prediction_time"], errors="raise", format="mixed"
    ).astype("datetime64[ns]")
    mask = (
        feature_dataset["evaluation_partition"].astype("string").eq(partition)
        & prediction_time.ge(start)
        & prediction_time.lt(end)
    )
    selected = feature_dataset.loc[mask].copy(deep=True)
    if selected.empty:
        raise ValueError("Calibration feature window is empty")
    if selected["evaluation_partition"].astype("string").eq("final_test").any():
        raise RuntimeError("Calibration feature window exposed final_test")
    return selected


def _mature_window_targets(
    feature_dataset: pd.DataFrame,
    appointments: pd.DataFrame,
    *,
    partition: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    window = _window_feature_rows(
        feature_dataset,
        partition=partition,
        start=start,
        end=end,
    )
    targets = build_mature_v2_target_table(
        window,
        appointments,
        model_fit_time=pd.Timestamp(label_cutoff),
        allowed_partitions=(partition,),
    )
    if targets["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Calibration target access exposed final_test")
    if targets["label_available_at"].ge(pd.Timestamp(label_cutoff)).any():
        raise RuntimeError("Calibration target maturity boundary was not strict")
    return targets


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    probability = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise RuntimeError("Calibration estimator does not expose one positive class")
    result = probability[:, int(positive[0])].astype(np.float64, copy=True)
    if result.ndim != 1 or len(result) != len(features):
        raise RuntimeError("Calibration probability shape is invalid")
    if not np.isfinite(result).all():
        raise RuntimeError("Calibration probabilities must be finite")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("Calibration probabilities must lie within [0, 1]")
    return result


def _calibration_intercept_slope(
    target: pd.Series,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Fit descriptive logistic recalibration intercept/slope by IRLS."""

    target_values = target.to_numpy(dtype=np.float64, copy=True)
    if set(np.unique(target_values)) != {0.0, 1.0}:
        return (float("nan"), float("nan"))
    if np.unique(probability).size < 2:
        return (float("nan"), float("nan"))

    epsilon = 1e-12
    clipped = np.clip(
        probability.astype(np.float64, copy=True),
        epsilon,
        1.0 - epsilon,
    )
    predictor = np.log(clipped / (1.0 - clipped))
    design = np.column_stack(
        (np.ones(len(predictor), dtype=np.float64), predictor)
    )
    prevalence = float(np.clip(target_values.mean(), epsilon, 1.0 - epsilon))
    coefficients = np.array(
        [np.log(prevalence / (1.0 - prevalence)), 0.0],
        dtype=np.float64,
    )

    def log_likelihood(values: np.ndarray) -> float:
        linear = design @ values
        return float(
            np.sum(
                target_values * linear - np.logaddexp(0.0, linear)
            )
        )

    for _iteration in range(100):
        linear = design @ coefficients
        mean = np.empty_like(linear)
        nonnegative = linear >= 0.0
        mean[nonnegative] = 1.0 / (1.0 + np.exp(-linear[nonnegative]))
        exp_linear = np.exp(linear[~nonnegative])
        mean[~nonnegative] = exp_linear / (1.0 + exp_linear)
        weight = np.maximum(mean * (1.0 - mean), 1e-15)
        score = design.T @ (target_values - mean)
        information = design.T @ (weight[:, None] * design)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, score, rcond=None)[0]

        baseline = log_likelihood(coefficients)
        multiplier = 1.0
        while multiplier > 1e-8:
            candidate = coefficients + multiplier * step
            if log_likelihood(candidate) >= baseline:
                break
            multiplier *= 0.5
        coefficients = coefficients + multiplier * step
        if float(np.max(np.abs(multiplier * step))) < 1e-10:
            break
    else:
        raise RuntimeError("Calibration intercept/slope IRLS did not converge")

    if not np.isfinite(coefficients).all():
        raise RuntimeError("Calibration intercept/slope is non-finite")
    return float(coefficients[0]), float(coefficients[1])


def _build_reliability_curve(
    predictions: pd.DataFrame,
    *,
    n_bins: int,
) -> pd.DataFrame:
    if n_bins <= 1:
        raise ValueError("n_bins must be greater than one")
    rows: list[dict[str, object]] = []
    for method in CALIBRATION_METHODS:
        subset = predictions.loc[predictions["method"].eq(method)].copy(deep=True)
        if subset.empty:
            raise ValueError(f"No predictions for calibration method {method}")
        subset = subset.sort_values(
            ["no_show_probability", "appointment_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        rank = np.arange(len(subset), dtype=np.int64)
        bins = np.minimum((rank * n_bins) // len(subset), n_bins - 1) + 1
        subset["bin"] = bins.astype(np.int16)
        grouped = subset.groupby("bin", sort=True, observed=True)
        for bin_number, frame in grouped:
            rows.append(
                {
                    "method": method,
                    "bin": int(bin_number),
                    "bin_count": int(len(frame)),
                    "positive_count": int(frame["target"].sum()),
                    "mean_predicted_probability": float(
                        frame["no_show_probability"].mean()
                    ),
                    "observed_no_show_rate": float(frame["target"].mean()),
                    "min_predicted_probability": float(
                        frame["no_show_probability"].min()
                    ),
                    "max_predicted_probability": float(
                        frame["no_show_probability"].max()
                    ),
                }
            )
    result = pd.DataFrame(rows, columns=list(RELIABILITY_COLUMNS))
    if len(result) != len(CALIBRATION_METHODS) * n_bins:
        raise RuntimeError("Reliability curve did not produce the requested bin count")
    return result


def _choose_calibration_method(
    metric_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tuple(metric_frame["method"]) != CALIBRATION_METHODS:
        raise ValueError("Calibration metrics must follow the frozen method order")
    selection_config = config["calibration"]["selection"]
    uncalibrated_log_loss = float(
        metric_frame.loc[
            metric_frame["method"].eq("uncalibrated"), "log_loss"
        ].item()
    )
    max_worsening = float(
        selection_config["log_loss_guardrail_max_worsening_vs_uncalibrated"]
    )
    margin = float(selection_config["brier_indifference_margin"])

    annotated = metric_frame.copy(deep=True)
    annotated["passes_log_loss_guardrail"] = annotated["log_loss"].le(
        uncalibrated_log_loss + max_worsening
    )
    annotated.loc[
        annotated["method"].eq("uncalibrated"), "passes_log_loss_guardrail"
    ] = True

    eligible = annotated.loc[annotated["passes_log_loss_guardrail"]]
    if eligible.empty:
        raise RuntimeError("No calibration candidate passed the log-loss guardrail")
    best_brier = float(eligible["brier_score"].min())
    annotated["within_brier_indifference_margin"] = (
        annotated["passes_log_loss_guardrail"]
        & annotated["brier_score"].le(best_brier + margin)
    )

    simplicity = tuple(selection_config["simplicity_preference_within_margin"])
    if simplicity != CALIBRATION_METHODS:
        raise ValueError("Calibration simplicity order differs from the frozen contract")
    selected = next(
        method
        for method in simplicity
        if bool(
            annotated.loc[
                annotated["method"].eq(method),
                "within_brier_indifference_margin",
            ].item()
        )
    )
    annotated["selected"] = annotated["method"].eq(selected)
    annotated = annotated.loc[:, list(METRIC_COLUMNS)]

    selection = {
        "selected_ranking_model": FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
        "selected_calibration_method": selected,
        "uncalibrated_log_loss": uncalibrated_log_loss,
        "best_guardrail_eligible_brier_score": best_brier,
        "log_loss_guardrail_max_worsening_vs_uncalibrated": max_worsening,
        "brier_indifference_margin": margin,
        "simplicity_preference_within_margin": list(simplicity),
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
    }
    return annotated, selection


def run_v2_calibration_evaluation(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    ranking_dir: Path = DEFAULT_V2_RANKING_DIR,
) -> V2CalibrationResult:
    """Run the frozen chronological Version 2 calibration comparison."""

    config = _load_frozen_model_config()
    protected = config["protected_final_test"]
    if any(
        bool(protected[key])
        for key in (
            "target_access_permitted_during_r2",
            "probability_vector_generation_permitted_during_r2",
            "metric_computation_permitted_during_r2",
        )
    ):
        raise RuntimeError("Frozen R2 contract no longer protects final_test")

    ranking_selection = _load_ranking_selection(Path(ranking_dir))
    selected_model = str(ranking_selection["selected_ranking_model"])
    if selected_model not in {"logistic_regression", "random_forest"}:
        raise RuntimeError("Selected ranking model is not calibratable under R2")

    feature_dataset = load_frozen_v2_processed_feature_dataset(Path(processed_dir))
    if "target" in feature_dataset.columns:
        raise RuntimeError("Target reached the frozen feature artifact")
    if feature_dataset["evaluation_partition"].astype("string").eq("final_test").sum() != 4343:
        raise RuntimeError("Frozen final_test feature partition identity changed")

    tables = load_verified_v2_raw_tables()
    calibration = config["calibration"]

    base_fit_time = pd.Timestamp(calibration["base_refit_time"])
    base_targets = build_mature_v2_target_table(
        feature_dataset,
        tables.appointments,
        model_fit_time=base_fit_time,
        allowed_partitions=tuple(calibration["base_training_partitions"]),
    )

    fit_window = calibration["calibration_fit_prediction_time"]
    calibrator_fit_time = pd.Timestamp(calibration["calibrator_fit_time"])
    calibration_targets = _mature_window_targets(
        feature_dataset,
        tables.appointments,
        partition="calibration",
        start=pd.Timestamp(fit_window["start"]),
        end=pd.Timestamp(fit_window["end"]),
        label_cutoff=calibrator_fit_time,
    )

    evaluation_window = calibration["calibration_evaluation_prediction_time"]
    evaluation_cutoff = pd.Timestamp(
        calibration["calibration_evaluation_label_cutoff"]
    )
    evaluation_targets = _mature_window_targets(
        feature_dataset,
        tables.appointments,
        partition="calibration",
        start=pd.Timestamp(evaluation_window["start"]),
        end=pd.Timestamp(evaluation_window["end"]),
        label_cutoff=evaluation_cutoff,
    )

    if set(base_targets["target"].unique()) != {0, 1}:
        raise RuntimeError("Base-fit target must contain both classes")
    if set(calibration_targets["target"].unique()) != {0, 1}:
        raise RuntimeError("Calibration-fit target must contain both classes")
    if set(evaluation_targets["target"].unique()) != {0, 1}:
        raise RuntimeError("Calibration-evaluation target must contain both classes")

    overlap = set(calibration_targets["appointment_id"]) & set(
        evaluation_targets["appointment_id"]
    )
    if overlap:
        raise RuntimeError("Calibration fit and evaluation windows overlap")

    base_features = _select_features_by_ids(
        feature_dataset, base_targets["appointment_id"]
    )
    calibration_features = _select_features_by_ids(
        feature_dataset, calibration_targets["appointment_id"]
    )
    evaluation_features = _select_features_by_ids(
        feature_dataset, evaluation_targets["appointment_id"]
    )
    base_target = base_targets["target"].astype("int8").reset_index(drop=True)
    calibration_target = calibration_targets["target"].astype("int8").reset_index(
        drop=True
    )
    evaluation_target = evaluation_targets["target"].astype("int8").reset_index(
        drop=True
    )

    base_estimator = build_v2_candidate_estimator(selected_model, config)
    base_estimator.fit(base_features, base_target)

    candidates: dict[str, Any] = {"uncalibrated": deepcopy(base_estimator)}
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(deepcopy(base_estimator)),
            method=method,
            cv=None,
            n_jobs=None,
            ensemble=False,
        )
        calibrated.fit(calibration_features, calibration_target)
        candidates[method] = calibrated
    if tuple(candidates) != CALIBRATION_METHODS:
        raise RuntimeError("Calibration candidate order is invalid")

    evaluation_metadata = feature_dataset.set_index("appointment_id").loc[
        evaluation_targets["appointment_id"].to_numpy(dtype=np.int64),
        ["prediction_time", "evaluation_partition"],
    ].reset_index()
    evaluation_metadata["prediction_time"] = pd.to_datetime(
        evaluation_metadata["prediction_time"], errors="raise", format="mixed"
    ).astype("datetime64[ns]")
    evaluation_metadata["evaluation_partition"] = evaluation_metadata[
        "evaluation_partition"
    ].astype("string")
    if evaluation_metadata["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Calibration evaluation metadata exposed final_test")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for method in CALIBRATION_METHODS:
        probability = _positive_probability(candidates[method], evaluation_features)
        metrics = evaluate_binary_probabilities(evaluation_target, probability)
        intercept, slope = _calibration_intercept_slope(
            evaluation_target, probability
        )
        metric_rows.append(
            {
                "method": method,
                "sample_size": int(len(evaluation_target)),
                "positive_count": int(evaluation_target.sum()),
                "positive_rate": float(evaluation_target.mean()),
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "mean_predicted_probability": float(probability.mean()),
                "passes_log_loss_guardrail": False,
                "within_brier_indifference_margin": False,
                "selected": False,
            }
        )
        frame = evaluation_metadata.copy(deep=True)
        frame.insert(0, "method", method)
        frame["target"] = evaluation_target.to_numpy(dtype=np.int8)
        frame["no_show_probability"] = probability
        prediction_frames.append(frame.loc[:, list(PREDICTION_COLUMNS)])

    metrics = pd.DataFrame(metric_rows, columns=list(METRIC_COLUMNS))
    metrics, selection = _choose_calibration_method(metrics, config)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["method"] = predictions["method"].astype("string")
    predictions["appointment_id"] = predictions["appointment_id"].astype("int64")
    predictions["prediction_time"] = pd.to_datetime(
        predictions["prediction_time"], errors="raise"
    ).astype("datetime64[ns]")
    predictions["evaluation_partition"] = predictions[
        "evaluation_partition"
    ].astype("string")
    predictions["target"] = predictions["target"].astype("int8")
    predictions["no_show_probability"] = predictions[
        "no_show_probability"
    ].astype("float64")
    if predictions["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Calibration predictions exposed final_test")
    if predictions.duplicated(["method", "appointment_id"]).any():
        raise RuntimeError("Calibration prediction keys are not unique")

    reliability = _build_reliability_curve(
        predictions,
        n_bins=int(config["reporting"]["calibration_bins"]),
    )

    selection = {
        **selection,
        "base_fit_time": base_fit_time.isoformat(),
        "base_training_rows": int(len(base_target)),
        "base_training_positive_count": int(base_target.sum()),
        "calibrator_fit_time": calibrator_fit_time.isoformat(),
        "calibration_fit_rows": int(len(calibration_target)),
        "calibration_fit_positive_count": int(calibration_target.sum()),
        "calibration_evaluation_label_cutoff": evaluation_cutoff.isoformat(),
        "calibration_evaluation_rows": int(len(evaluation_target)),
        "calibration_evaluation_positive_count": int(evaluation_target.sum()),
        "model_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "model_contract_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "rolling_origin_manifest_sha256": FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
    }
    return V2CalibrationResult(
        metrics=metrics,
        predictions=predictions,
        reliability_curve=reliability,
        selection=selection,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S",
    )


def export_v2_calibration_results(
    result: V2CalibrationResult,
    *,
    output_dir: Path = DEFAULT_V2_CALIBRATION_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic R2 calibration results and a hash manifest."""

    destination = Path(output_dir)
    paths = {
        "metrics": destination / "calibration_metrics.csv",
        "predictions": destination / "calibration_predictions.csv",
        "reliability": destination / "calibration_reliability_curve.csv",
        "selection": destination / "calibration_selection.json",
        "manifest": destination / "calibration_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Calibration outputs already exist; use overwrite=True to replace them"
        )
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(result.metrics, paths["metrics"])
    _write_csv(result.predictions, paths["predictions"])
    _write_csv(result.reliability_curve, paths["reliability"])
    paths["selection"].write_text(
        json.dumps(dict(result.selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifact_entries: dict[str, dict[str, object]] = {}
    for key in ("metrics", "predictions", "reliability", "selection"):
        path = paths[key]
        artifact_entries[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    manifest = {
        "schema_version": "1.0.0",
        "phase": "R2",
        "stage": "chronological_calibration",
        "model_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "model_contract_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "rolling_origin_manifest_sha256": FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
        "processed_dataset_sha256": EXPECTED_PROCESSED_DATASET_SHA256,
        "processed_manifest_sha256": EXPECTED_PROCESSED_MANIFEST_SHA256,
        "processed_dataset_fingerprint": EXPECTED_PROCESSED_FINGERPRINT,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "selected_ranking_model": result.selection["selected_ranking_model"],
        "selected_calibration_method": result.selection[
            "selected_calibration_method"
        ],
        "base_training_rows": result.selection["base_training_rows"],
        "base_training_positive_count": result.selection[
            "base_training_positive_count"
        ],
        "calibration_fit_rows": result.selection["calibration_fit_rows"],
        "calibration_fit_positive_count": result.selection[
            "calibration_fit_positive_count"
        ],
        "calibration_evaluation_rows": result.selection[
            "calibration_evaluation_rows"
        ],
        "calibration_evaluation_positive_count": result.selection[
            "calibration_evaluation_positive_count"
        ],
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
        "artifacts": artifact_entries,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Version 2 chronological calibration evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_V2_CALIBRATION_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_v2_calibration_evaluation()
    manifest = export_v2_calibration_results(
        result,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print("Calibration evaluation metrics:")
    print(result.metrics.to_string(index=False))
    print()
    print(
        "Selected calibration method: "
        f"{result.selection['selected_calibration_method']}"
    )
    print(
        "Base training rows / positives: "
        f"{result.selection['base_training_rows']:,} / "
        f"{result.selection['base_training_positive_count']:,}"
    )
    print(
        "Calibration fit rows / positives: "
        f"{result.selection['calibration_fit_rows']:,} / "
        f"{result.selection['calibration_fit_positive_count']:,}"
    )
    print(
        "Calibration evaluation rows / positives: "
        f"{result.selection['calibration_evaluation_rows']:,} / "
        f"{result.selection['calibration_evaluation_positive_count']:,}"
    )
    print("Final-test target accessed: false")
    print("Final-test probabilities generated: false")
    print(
        "Calibration manifest: "
        f"{Path(args.output_dir) / 'calibration_manifest.json'}"
    )
    print(f"Manifest stage: {manifest['stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
