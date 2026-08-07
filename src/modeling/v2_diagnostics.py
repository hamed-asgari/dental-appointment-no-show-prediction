"""Pre-test interpretation, error, and subgroup diagnostics for frozen Version 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.v2_development import (
    _positive_class_probability,
    _select_features_by_ids,
)
from src.modeling.v2_persistence import (
    DEFAULT_V2_PERSISTENCE_DIR,
    FROZEN_V2_POLICY_MANIFEST_SHA256,
    FROZEN_V2_R3_CONFIG_SHA256,
    MANIFEST_FILENAME as PERSISTENCE_MANIFEST_FILENAME,
    METADATA_FILENAME as PERSISTENCE_METADATA_FILENAME,
    PIPELINE_FILENAME as PERSISTENCE_PIPELINE_FILENAME,
    POLICY_REPLAY_ATOL,
    POLICY_REPLAY_RTOL,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_R3_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_r3_execution.json"
DEFAULT_V2_POLICY_DIR = _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "policy"
DEFAULT_V2_DIAGNOSTICS_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "diagnostics"
)

FROZEN_V2_PERSISTENCE_MANIFEST_SHA256 = (
    "ca19d477e0590f40d1abbad869119b182d05e923b2d582df19c42473d2795856"
)
FROZEN_V2_PIPELINE_SHA256 = (
    "301029bd5bee1ffe346fbf09dcc6ed4570b231458ba8a081f8a0f6bb544d9df0"
)
FROZEN_V2_PERSISTENCE_METADATA_SHA256 = (
    "33eda2b123e592813008a004b4aa3f353ac1a2bda51ca5eaddb45954eeea6224"
)

POLICY_PREDICTIONS_FILENAME = "policy_predictions.csv"
PERMUTATION_FILENAME = "permutation_importance.csv"
SUBGROUP_FILENAME = "subgroup_metrics.csv"
FIRST_REPEAT_FILENAME = "first_time_vs_repeat.csv"
ROW_ERROR_FILENAME = "row_error_analysis.csv"
CAPACITY_FILENAME = "capacity_error_summary.csv"
SUMMARY_FILENAME = "diagnostics_summary.json"
MANIFEST_FILENAME = "pretest_diagnostics_manifest.json"

EXPECTED_POLICY_ROWS = 1063
EXPECTED_POLICY_POSITIVES = 92


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


def _stable_value(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def _safe_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    if len(target) != len(probability) or len(target) == 0:
        raise ValueError("Metric arrays must be non-empty and aligned")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Probabilities must lie within [0, 1]")

    ap = float(average_precision_score(target, probability))
    brier = float(brier_score_loss(target, probability))
    ll = float(log_loss(target, probability, labels=[0, 1]))
    auc: float | None
    if np.unique(target).size == 2:
        auc = float(roc_auc_score(target, probability))
    else:
        auc = None
    return {
        "average_precision": ap,
        "roc_auc": auc,
        "brier_score": brier,
        "log_loss": ll,
    }


def _capacity_rows(
    *,
    appointment_id: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    fractions: tuple[float, ...],
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "appointment_id": np.asarray(appointment_id, dtype=np.int64),
            "target": np.asarray(target, dtype=np.int8),
            "no_show_probability": np.asarray(probability, dtype=np.float64),
        }
    )
    base = base.sort_values(
        ["no_show_probability", "appointment_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    total_positives = int(base["target"].sum())
    n = int(len(base))
    for fraction in fractions:
        k = max(1, int(math.floor(n * float(fraction))))
        selected = base.iloc[:k].copy()
        selected_target = selected["target"].to_numpy(dtype=np.int8, copy=True)
        selected_probability = selected["no_show_probability"].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        selected_positive_count = int(selected_target.sum())
        false_positive_count = int(k - selected_positive_count)
        false_negative_count = int(total_positives - selected_positive_count)

        eps = np.finfo(np.float64).eps
        clipped = np.clip(selected_probability, eps, 1.0 - eps)
        row_log_loss = -(
            selected_target * np.log(clipped)
            + (1 - selected_target) * np.log1p(-clipped)
        )
        absolute_error = np.abs(selected_target - selected_probability)
        brier = np.square(selected_target - selected_probability)

        rows.append(
            {
                "capacity_fraction": float(fraction),
                "selected_count": k,
                "selected_fraction": float(k / n),
                "reported_threshold": float(selected_probability.min()),
                "selected_positive_count": selected_positive_count,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "precision": float(selected_positive_count / k),
                "recall": float(selected_positive_count / total_positives),
                "selected_mean_absolute_probability_error": float(
                    absolute_error.mean()
                ),
                "selected_mean_brier_contribution": float(brier.mean()),
                "selected_mean_log_loss_contribution": float(row_log_loss.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_v2_pretest_diagnostics(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    policy_dir: Path = DEFAULT_V2_POLICY_DIR,
    persistence_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
) -> dict[str, object]:
    """Compute frozen R3 diagnostics on policy_selection only."""

    r3_config_path = Path(DEFAULT_V2_R3_CONFIG_PATH)
    if _sha256(r3_config_path) != FROZEN_V2_R3_CONFIG_SHA256:
        raise ValueError("Frozen R3 execution config SHA-256 mismatch")
    config = _read_json(r3_config_path)
    diagnostics_config = config["pre_test_diagnostics"]
    if diagnostics_config["primary_partition"] != "policy_selection":
        raise RuntimeError("Frozen R3 diagnostic partition changed")
    if diagnostics_config["final_test_permitted"] is not False:
        raise RuntimeError("R3 diagnostics unexpectedly permit final_test")

    persistence_dir = Path(persistence_dir)
    pipeline_path = persistence_dir / PERSISTENCE_PIPELINE_FILENAME
    metadata_path = persistence_dir / PERSISTENCE_METADATA_FILENAME
    persistence_manifest_path = persistence_dir / PERSISTENCE_MANIFEST_FILENAME

    if _sha256(pipeline_path) != FROZEN_V2_PIPELINE_SHA256:
        raise ValueError("Frozen pipeline SHA-256 mismatch")
    if _sha256(metadata_path) != FROZEN_V2_PERSISTENCE_METADATA_SHA256:
        raise ValueError("Frozen persistence metadata SHA-256 mismatch")
    if _sha256(persistence_manifest_path) != FROZEN_V2_PERSISTENCE_MANIFEST_SHA256:
        raise ValueError("Frozen persistence manifest SHA-256 mismatch")

    persistence_manifest = _read_json(persistence_manifest_path)
    if persistence_manifest["final_test_target_accessed"] is not False:
        raise RuntimeError("Persistence manifest records final-test target access")
    if persistence_manifest["final_test_probabilities_generated"] is not False:
        raise RuntimeError("Persistence manifest records final-test probabilities")

    policy_dir = Path(policy_dir)
    policy_manifest_path = policy_dir / "policy_manifest.json"
    if _sha256(policy_manifest_path) != FROZEN_V2_POLICY_MANIFEST_SHA256:
        raise ValueError("Frozen policy manifest SHA-256 mismatch")
    policy_manifest = _read_json(policy_manifest_path)
    if policy_manifest["final_test_target_accessed"] is not False:
        raise RuntimeError("Policy manifest records final-test target access")
    if policy_manifest["final_test_probabilities_generated"] is not False:
        raise RuntimeError("Policy manifest records final-test probabilities")

    policy_prediction_path = policy_dir / POLICY_PREDICTIONS_FILENAME
    prediction_entry = policy_manifest["artifacts"][POLICY_PREDICTIONS_FILENAME]
    if _sha256(policy_prediction_path) != prediction_entry["sha256"]:
        raise ValueError("Frozen policy prediction SHA-256 mismatch")
    if policy_prediction_path.stat().st_size != int(prediction_entry["size_bytes"]):
        raise ValueError("Frozen policy prediction byte size mismatch")

    policy = pd.read_csv(
        policy_prediction_path,
        dtype={
            "appointment_id": "int64",
            "evaluation_partition": "string",
            "target": "int8",
            "no_show_probability": "float64",
        },
    )
    if len(policy) != EXPECTED_POLICY_ROWS:
        raise RuntimeError("Frozen policy row count changed")
    if int(policy["target"].sum()) != EXPECTED_POLICY_POSITIVES:
        raise RuntimeError("Frozen policy positive count changed")
    if not policy["appointment_id"].is_unique:
        raise RuntimeError("Frozen policy appointment IDs must be unique")
    if set(policy["evaluation_partition"].dropna().unique()) != {"policy_selection"}:
        raise PermissionError("Diagnostic target rows include a prohibited partition")

    feature_dataset = load_frozen_v2_processed_feature_dataset(Path(processed_dir))
    policy_features = _select_features_by_ids(
        feature_dataset,
        policy["appointment_id"],
    )
    selected_audit = feature_dataset.set_index("appointment_id").loc[
        policy["appointment_id"].tolist(),
        ["evaluation_partition"],
    ]
    if set(selected_audit["evaluation_partition"].astype("string").unique()) != {
        "policy_selection"
    }:
        raise PermissionError("Diagnostic feature rows include a prohibited partition")

    estimator = joblib.load(pipeline_path)
    if not isinstance(estimator, Pipeline):
        raise RuntimeError("Frozen persisted artifact is not an sklearn Pipeline")

    current_probability = _positive_class_probability(estimator, policy_features)
    frozen_probability = policy["no_show_probability"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    if not np.allclose(
        current_probability,
        frozen_probability,
        atol=POLICY_REPLAY_ATOL,
        rtol=POLICY_REPLAY_RTOL,
    ):
        raise RuntimeError("Frozen persisted model no longer replays policy probabilities")

    target = policy["target"].to_numpy(dtype=np.int8, copy=True)
    overall_metrics = _safe_metrics(target, current_probability)

    permutation_config = diagnostics_config["permutation_importance"]
    if permutation_config["metric"] != "average_precision":
        raise RuntimeError("Frozen permutation-importance metric changed")
    if permutation_config["permutation_unit"] != "raw_feature_column_before_preprocessing":
        raise RuntimeError("Frozen permutation unit changed")
    if permutation_config["may_drive_feature_selection"] is not False:
        raise RuntimeError("Permutation importance must remain interpretation-only")

    permutation = permutation_importance(
        estimator,
        policy_features,
        target,
        scoring="average_precision",
        n_repeats=int(permutation_config["n_repeats"]),
        random_state=int(permutation_config["random_state"]),
        n_jobs=1,
    )
    permutation_table = pd.DataFrame(
        {
            "feature": list(V2_MODEL_FEATURE_COLUMNS),
            "importance_mean": permutation.importances_mean.astype(np.float64),
            "importance_std": permutation.importances_std.astype(np.float64),
        }
    ).sort_values(
        ["importance_mean", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    permutation_table.insert(
        0,
        "rank",
        np.arange(1, len(permutation_table) + 1, dtype=np.int16),
    )

    diagnostic_frame = policy_features.copy()
    diagnostic_frame.insert(
        0,
        "appointment_id",
        policy["appointment_id"].to_numpy(dtype=np.int64, copy=True),
    )
    diagnostic_frame["target"] = target
    diagnostic_frame["no_show_probability"] = current_probability

    minimum_rows = int(diagnostics_config["minimum_subgroup_rows"])
    minimum_positives = int(diagnostics_config["minimum_subgroup_positive_count"])
    subgroup_rows: list[dict[str, object]] = []
    for feature in diagnostics_config["subgroup_features"]:
        if feature not in diagnostic_frame.columns:
            raise RuntimeError(f"Frozen subgroup feature is unavailable: {feature}")
        for value, group in diagnostic_frame.groupby(
            feature,
            dropna=False,
            sort=True,
        ):
            group_target = group["target"].to_numpy(dtype=np.int8, copy=True)
            group_probability = group["no_show_probability"].to_numpy(
                dtype=np.float64,
                copy=True,
            )
            n = int(len(group))
            positives = int(group_target.sum())
            negatives = int(n - positives)
            supported = n >= minimum_rows and positives >= minimum_positives
            metrics: dict[str, float | None]
            if supported:
                metrics = _safe_metrics(group_target, group_probability)
            else:
                metrics = {
                    "average_precision": None,
                    "roc_auc": None,
                    "brier_score": None,
                    "log_loss": None,
                }
            subgroup_rows.append(
                {
                    "subgroup_feature": str(feature),
                    "subgroup_value": _stable_value(value),
                    "sample_size": n,
                    "positive_count": positives,
                    "negative_count": negatives,
                    "prevalence": float(positives / n),
                    "supported_for_quantitative_reporting": bool(supported),
                    "average_precision": metrics["average_precision"],
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                    "log_loss": metrics["log_loss"],
                }
            )

    subgroup_table = pd.DataFrame(subgroup_rows)
    feature_order = {
        feature: index
        for index, feature in enumerate(diagnostics_config["subgroup_features"])
    }
    subgroup_table["_feature_order"] = subgroup_table["subgroup_feature"].map(
        feature_order
    )
    subgroup_table = subgroup_table.sort_values(
        ["_feature_order", "subgroup_value"],
        kind="mergesort",
    ).drop(columns="_feature_order").reset_index(drop=True)

    history_config = diagnostics_config["first_time_vs_repeat"]
    history_feature = str(history_config["feature"])
    cohort_map = {
        _stable_value(history_config["first_time_value"]): "first_time",
        _stable_value(history_config["repeat_value"]): "repeat",
    }
    first_repeat = subgroup_table.loc[
        subgroup_table["subgroup_feature"].eq(history_feature)
        & subgroup_table["subgroup_value"].isin(cohort_map),
    ].copy()
    first_repeat.insert(
        0,
        "cohort",
        first_repeat["subgroup_value"].map(cohort_map),
    )
    first_repeat = first_repeat.sort_values("cohort", kind="mergesort").reset_index(
        drop=True
    )
    if set(first_repeat["cohort"]) != {"first_time", "repeat"}:
        raise RuntimeError("First-time versus repeat diagnostic cohorts are incomplete")

    eps = np.finfo(np.float64).eps
    clipped = np.clip(current_probability, eps, 1.0 - eps)
    row_errors = pd.DataFrame(
        {
            "appointment_id": policy["appointment_id"].to_numpy(
                dtype=np.int64,
                copy=True,
            ),
            "target": target,
            "no_show_probability": current_probability,
            "absolute_probability_error": np.abs(target - current_probability),
            "brier_contribution": np.square(target - current_probability),
            "log_loss_contribution": -(
                target * np.log(clipped)
                + (1 - target) * np.log1p(-clipped)
            ),
        }
    )

    capacity_fractions = tuple(
        float(value)
        for value in diagnostics_config["error_analysis"][
            "registered_capacity_fractions"
        ]
    )
    capacity = _capacity_rows(
        appointment_id=policy["appointment_id"].to_numpy(
            dtype=np.int64,
            copy=True,
        ),
        target=target,
        probability=current_probability,
        fractions=capacity_fractions,
    )

    top_features = permutation_table.head(10).to_dict(orient="records")
    supported_subgroups = int(
        subgroup_table["supported_for_quantitative_reporting"].sum()
    )
    summary = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "pretest_interpretation_error_subgroup_diagnostics",
        "diagnostic_partition": "policy_selection",
        "sample_size": int(len(policy)),
        "positive_count": int(target.sum()),
        "prevalence": float(target.mean()),
        "average_precision": overall_metrics["average_precision"],
        "roc_auc": overall_metrics["roc_auc"],
        "brier_score": overall_metrics["brier_score"],
        "log_loss": overall_metrics["log_loss"],
        "mean_predicted_probability": float(current_probability.mean()),
        "permutation_metric": "average_precision",
        "permutation_n_repeats": int(permutation_config["n_repeats"]),
        "permutation_random_state": int(permutation_config["random_state"]),
        "top_10_permutation_features": top_features,
        "subgroup_feature_count": int(len(diagnostics_config["subgroup_features"])),
        "subgroup_row_count": int(len(subgroup_table)),
        "supported_subgroup_row_count": supported_subgroups,
        "minimum_subgroup_rows": minimum_rows,
        "minimum_subgroup_positive_count": minimum_positives,
        "capacity_fractions": list(capacity_fractions),
        "single_operational_threshold_selected": False,
        "permutation_importance_may_drive_feature_selection": False,
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
    }

    return {
        "permutation_importance": permutation_table,
        "subgroup_metrics": subgroup_table,
        "first_time_vs_repeat": first_repeat,
        "row_error_analysis": row_errors,
        "capacity_error_summary": capacity,
        "summary": summary,
    }


def export_v2_pretest_diagnostics(
    result: Mapping[str, object],
    *,
    output_dir: Path = DEFAULT_V2_DIAGNOSTICS_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic R3 pre-test diagnostic artifacts."""

    output_dir = Path(output_dir)
    output_paths = {
        PERMUTATION_FILENAME: output_dir / PERMUTATION_FILENAME,
        SUBGROUP_FILENAME: output_dir / SUBGROUP_FILENAME,
        FIRST_REPEAT_FILENAME: output_dir / FIRST_REPEAT_FILENAME,
        ROW_ERROR_FILENAME: output_dir / ROW_ERROR_FILENAME,
        CAPACITY_FILENAME: output_dir / CAPACITY_FILENAME,
        SUMMARY_FILENAME: output_dir / SUMMARY_FILENAME,
        MANIFEST_FILENAME: output_dir / MANIFEST_FILENAME,
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError("Pre-test diagnostic outputs already exist; use overwrite=True")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_keys = {
        PERMUTATION_FILENAME: "permutation_importance",
        SUBGROUP_FILENAME: "subgroup_metrics",
        FIRST_REPEAT_FILENAME: "first_time_vs_repeat",
        ROW_ERROR_FILENAME: "row_error_analysis",
        CAPACITY_FILENAME: "capacity_error_summary",
    }
    for filename, key in frame_keys.items():
        frame = result[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for {key}")
        frame.to_csv(
            output_paths[filename],
            index=False,
            lineterminator="\n",
            float_format="%.12g",
        )

    summary = dict(result["summary"])
    output_paths[SUMMARY_FILENAME].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifacts: dict[str, dict[str, object]] = {}
    for filename in (
        PERMUTATION_FILENAME,
        SUBGROUP_FILENAME,
        FIRST_REPEAT_FILENAME,
        ROW_ERROR_FILENAME,
        CAPACITY_FILENAME,
        SUMMARY_FILENAME,
    ):
        path = output_paths[filename]
        artifacts[filename] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    manifest = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "pretest_interpretation_error_subgroup_diagnostics",
        "r3_execution_config_sha256": FROZEN_V2_R3_CONFIG_SHA256,
        "policy_manifest_sha256": FROZEN_V2_POLICY_MANIFEST_SHA256,
        "persistence_manifest_sha256": FROZEN_V2_PERSISTENCE_MANIFEST_SHA256,
        "pipeline_sha256": FROZEN_V2_PIPELINE_SHA256,
        "diagnostic_partition": "policy_selection",
        "sample_size": int(summary["sample_size"]),
        "positive_count": int(summary["positive_count"]),
        "subgroup_feature_count": int(summary["subgroup_feature_count"]),
        "single_operational_threshold_selected": False,
        "permutation_importance_may_drive_feature_selection": False,
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
        "artifacts": artifacts,
    }
    output_paths[MANIFEST_FILENAME].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def run_and_export_v2_pretest_diagnostics(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    policy_dir: Path = DEFAULT_V2_POLICY_DIR,
    persistence_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
    output_dir: Path = DEFAULT_V2_DIAGNOSTICS_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    result = build_v2_pretest_diagnostics(
        processed_dir=processed_dir,
        policy_dir=policy_dir,
        persistence_dir=persistence_dir,
    )
    return export_v2_pretest_diagnostics(
        result,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen Version 2 R3 pre-test interpretation, error, "
            "and subgroup diagnostics."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_V2_PROCESSED_DIR)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_V2_POLICY_DIR)
    parser.add_argument(
        "--persistence-dir",
        type=Path,
        default=DEFAULT_V2_PERSISTENCE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V2_DIAGNOSTICS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest = run_and_export_v2_pretest_diagnostics(
        processed_dir=args.processed_dir,
        policy_dir=args.policy_dir,
        persistence_dir=args.persistence_dir,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    summary = _read_json(Path(args.output_dir) / SUMMARY_FILENAME)
    print("Frozen R3 pre-test diagnostics created.")
    print(
        "Policy-selection rows / positives: "
        f"{summary['sample_size']:,} / {summary['positive_count']:,}"
    )
    print(f"Average Precision: {summary['average_precision']:.12f}")
    print(f"ROC-AUC: {summary['roc_auc']:.12f}")
    print(f"Brier score: {summary['brier_score']:.12f}")
    print(f"Log loss: {summary['log_loss']:.12f}")
    print(
        "Top permutation feature: "
        f"{summary['top_10_permutation_features'][0]['feature']}"
    )
    print(
        "Supported subgroup rows: "
        f"{summary['supported_subgroup_row_count']:,}"
    )
    print(f"Scenario capacity rows: {len(summary['capacity_fractions'])}")
    print(
        "Manifest SHA-256: "
        f"{_sha256(Path(args.output_dir) / MANIFEST_FILENAME)}"
    )
    print("Single operational threshold selected: false")
    print("Final-test target accessed: false")
    print("Final-test probabilities generated: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
