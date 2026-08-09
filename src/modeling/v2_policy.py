"""Version 2 deterministic R2 policy-sensitivity analysis under frozen contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn

from src.data.build_v2_dataset import load_verified_v2_raw_tables
from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.data.v2_targets import build_mature_v2_target_table
from src.modeling.v2_calibration import (
    DEFAULT_V2_CALIBRATION_OUTPUT_DIR,
    EXPECTED_PROCESSED_DATASET_SHA256,
    EXPECTED_PROCESSED_FINGERPRINT,
    EXPECTED_PROCESSED_MANIFEST_SHA256,
    _positive_probability,
)
from src.modeling.v2_calibration_hashes import (
    FROZEN_V2_CALIBRATION_ARTIFACT_SHA256,
    FROZEN_V2_CALIBRATION_MANIFEST_SHA256,
    FROZEN_V2_SELECTED_CALIBRATION_METHOD,
)
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
DEFAULT_V2_POLICY_OUTPUT_DIR = _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "policy"
POLICY_EXECUTION_SPEC_PATH = _REPOSITORY_ROOT / "docs" / "v2_r2_policy_execution_spec.md"
FROZEN_V2_POLICY_EXECUTION_SPEC_SHA256 = (
    "4e12c2db3a95ed096040e558b567106a7569a07f3fdec8fb2d28570dedc90863"
)

CAPACITY_FRACTIONS = (0.05, 0.10, 0.20)
FN_FP_COST_RATIOS = (1.0, 2.0, 5.0, 10.0)

POLICY_PREDICTION_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "evaluation_partition",
    "label_available_at",
    "target",
    "no_show_probability",
)

POLICY_SCENARIO_COLUMNS = (
    "scenario_id",
    "scenario_family",
    "capacity_fraction",
    "false_negative_to_false_positive_cost_ratio",
    "threshold",
    "selected_count",
    "selected_fraction",
    "precision",
    "recall",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "scenario_cost",
)


@dataclass(frozen=True)
class V2PolicySensitivityResult:
    predictions: pd.DataFrame
    scenarios: pd.DataFrame
    summary: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_policy_config(config: Mapping[str, Any]) -> None:
    policy = config["policy_selection"]
    protected = config["protected_final_test"]

    if str(policy["partition"]) != "policy_selection":
        raise ValueError("Frozen policy partition changed")
    if str(policy["decision_time"]) != "2027-01-01T00:00:00":
        raise ValueError("Frozen policy decision time changed")

    capacities = tuple(float(value) for value in policy["capacity_fractions"])
    if capacities != CAPACITY_FRACTIONS:
        raise ValueError("Frozen policy capacity fractions changed")

    ratios = tuple(
        float(value)
        for value in policy["false_negative_to_false_positive_cost_ratios"]
    )
    if ratios != FN_FP_COST_RATIOS:
        raise ValueError("Frozen policy cost ratios changed")

    expected_formula = "1 / (1 + false_negative_to_false_positive_cost_ratio)"
    if str(policy["cost_threshold_formula"]) != expected_formula:
        raise ValueError("Frozen cost-threshold formula changed")
    if bool(policy["single_operational_threshold_selected"]):
        raise RuntimeError("R2 must not select one operational threshold")

    protected_keys = (
        "target_access_permitted_during_r2",
        "probability_vector_generation_permitted_during_r2",
        "metric_computation_permitted_during_r2",
    )
    if any(bool(protected[key]) for key in protected_keys):
        raise RuntimeError("Frozen R2 contract no longer protects final_test")


def _load_frozen_policy_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_frozen_model_config()
    _validate_policy_config(config)

    if not POLICY_EXECUTION_SPEC_PATH.is_file():
        raise ValueError("Frozen policy execution specification is missing")
    if _sha256(POLICY_EXECUTION_SPEC_PATH) != FROZEN_V2_POLICY_EXECUTION_SPEC_SHA256:
        raise ValueError("Frozen policy execution specification SHA-256 mismatch")

    manifest_path = DEFAULT_V2_CALIBRATION_OUTPUT_DIR / "calibration_manifest.json"
    selection_path = DEFAULT_V2_CALIBRATION_OUTPUT_DIR / "calibration_selection.json"

    if not manifest_path.is_file() or not selection_path.is_file():
        raise ValueError("Frozen calibration artifacts are missing")
    if _sha256(manifest_path) != FROZEN_V2_CALIBRATION_MANIFEST_SHA256:
        raise ValueError("Frozen calibration manifest SHA-256 mismatch")

    expected_selection_hash = FROZEN_V2_CALIBRATION_ARTIFACT_SHA256[
        "calibration_selection.json"
    ]
    if _sha256(selection_path) != expected_selection_hash:
        raise ValueError("Frozen calibration selection SHA-256 mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    for source in (manifest, selection):
        if source.get("final_test_target_accessed") is not False:
            raise RuntimeError("Calibration artifact exposed protected final_test target")
        if source.get("final_test_probabilities_generated") is not False:
            raise RuntimeError(
                "Calibration artifact generated protected final_test probabilities"
            )

    if manifest.get("selected_ranking_model") != FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL:
        raise ValueError("Calibration manifest ranking model changed")
    if selection.get("selected_ranking_model") != FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL:
        raise ValueError("Calibration selection ranking model changed")
    if manifest.get("selected_calibration_method") != FROZEN_V2_SELECTED_CALIBRATION_METHOD:
        raise ValueError("Calibration manifest method changed")
    if selection.get("selected_calibration_method") != FROZEN_V2_SELECTED_CALIBRATION_METHOD:
        raise ValueError("Calibration selection method changed")
    if FROZEN_V2_SELECTED_CALIBRATION_METHOD != "uncalibrated":
        raise RuntimeError("Current R2 policy implementation expects uncalibrated selection")

    return config, manifest, selection


def _capacity_count(sample_size: int, capacity_fraction: float) -> int:
    sample_size = int(sample_size)
    capacity_fraction = float(capacity_fraction)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must lie in (0, 1]")
    count = int(math.floor(sample_size * capacity_fraction))
    if count < 1:
        raise ValueError("Frozen capacity scenario selects fewer than one row")
    return count


def _cost_threshold(
    false_negative_to_false_positive_cost_ratio: float,
) -> float:
    ratio = float(false_negative_to_false_positive_cost_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("Cost ratio must be finite and positive")
    return float(1.0 / (1.0 + ratio))


def _validate_policy_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"appointment_id", "target", "no_show_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Policy predictions are missing columns: " + ", ".join(missing))
    if predictions.empty:
        raise ValueError("Policy predictions must not be empty")

    frame = predictions.copy(deep=True)
    frame["appointment_id"] = pd.to_numeric(
        frame["appointment_id"], errors="raise"
    ).astype("int64")
    if not frame["appointment_id"].is_unique:
        raise ValueError("Policy appointment_id values must be unique")

    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype("int8")
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("Policy target must be binary")
    if frame["target"].nunique() < 2:
        raise ValueError("Policy target must contain both classes")

    frame["no_show_probability"] = pd.to_numeric(
        frame["no_show_probability"], errors="raise"
    ).astype("float64")
    probability = frame["no_show_probability"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(probability).all():
        raise ValueError("Policy probabilities must be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Policy probabilities must lie within [0, 1]")
    return frame


def _capacity_selection(
    predictions: pd.DataFrame,
    capacity_fraction: float,
) -> tuple[np.ndarray, int, float]:
    frame = _validate_policy_prediction_frame(predictions)
    count = _capacity_count(len(frame), capacity_fraction)
    ranked = frame.sort_values(
        ["no_show_probability", "appointment_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_ids = set(
        ranked.iloc[:count]["appointment_id"].to_numpy(dtype=np.int64).tolist()
    )
    selected = frame["appointment_id"].isin(selected_ids).to_numpy(dtype=bool)
    if int(selected.sum()) != count:
        raise RuntimeError("Capacity selection count is inconsistent")
    threshold = float(ranked.iloc[count - 1]["no_show_probability"])
    return selected, count, threshold


def _threshold_selection(
    predictions: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    frame = _validate_policy_prediction_frame(predictions)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and lie within [0, 1]")
    return frame["no_show_probability"].ge(threshold).to_numpy(dtype=bool)


def _scenario_metrics(
    predictions: pd.DataFrame,
    selected: np.ndarray,
    *,
    cost_ratio: float,
) -> dict[str, float | int]:
    frame = _validate_policy_prediction_frame(predictions)
    selected = np.asarray(selected, dtype=bool)
    if selected.ndim != 1 or len(selected) != len(frame):
        raise ValueError("selected mask shape is invalid")

    target = frame["target"].to_numpy(dtype=np.int8, copy=True)
    positive = target == 1
    negative = ~positive

    true_positive = int(np.sum(selected & positive))
    false_positive = int(np.sum(selected & negative))
    true_negative = int(np.sum((~selected) & negative))
    false_negative = int(np.sum((~selected) & positive))

    if true_positive + false_positive:
        precision = float(true_positive / (true_positive + false_positive))
    else:
        precision = 0.0

    if true_positive + false_negative:
        recall = float(true_positive / (true_positive + false_negative))
    else:
        recall = 0.0

    ratio = float(cost_ratio)
    scenario_cost = float(false_positive + ratio * false_negative)

    return {
        "selected_count": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "scenario_cost": scenario_cost,
    }


def _build_policy_scenarios(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    _validate_policy_config(config)
    frame = _validate_policy_prediction_frame(predictions)
    rows: list[dict[str, object]] = []

    for capacity in CAPACITY_FRACTIONS:
        selected, expected_count, threshold = _capacity_selection(frame, capacity)
        if int(selected.sum()) != expected_count:
            raise RuntimeError("Capacity scenario selected-count mismatch")
        for ratio in FN_FP_COST_RATIOS:
            metrics = _scenario_metrics(frame, selected, cost_ratio=ratio)
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
        selected = _threshold_selection(frame, threshold)
        metrics = _scenario_metrics(frame, selected, cost_ratio=ratio)
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
        raise RuntimeError("Frozen policy grid must contain exactly 16 scenarios")
    if scenarios["scenario_id"].duplicated().any():
        raise RuntimeError("Policy scenario identifiers must be unique")
    return scenarios


def run_v2_policy_sensitivity(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
) -> V2PolicySensitivityResult:
    """Run the frozen R2 policy-sensitivity batch without touching final_test."""
    config, calibration_manifest, _calibration_selection = _load_frozen_policy_state()

    feature_dataset = load_frozen_v2_processed_feature_dataset(Path(processed_dir))
    if "target" in feature_dataset.columns:
        raise RuntimeError("Target reached the frozen feature artifact")
    final_test_rows = int(
        feature_dataset["evaluation_partition"].astype("string").eq("final_test").sum()
    )
    if final_test_rows != 4343:
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
    if set(base_targets["target"].unique()) != {0, 1}:
        raise RuntimeError("Policy base-fit target must contain both classes")

    policy = config["policy_selection"]
    policy_partition = str(policy["partition"])
    if policy_partition == "final_test":
        raise PermissionError("R2 policy sensitivity may not use final_test")
    decision_time = pd.Timestamp(policy["decision_time"])
    policy_targets = build_mature_v2_target_table(
        feature_dataset,
        tables.appointments,
        model_fit_time=decision_time,
        allowed_partitions=(policy_partition,),
    )
    if policy_targets["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Policy target access exposed final_test")
    if policy_targets["label_available_at"].ge(decision_time).any():
        raise RuntimeError("Policy target maturity boundary was not strict")
    if set(policy_targets["target"].unique()) != {0, 1}:
        raise RuntimeError("Policy-selection target must contain both classes")

    base_features = _select_features_by_ids(
        feature_dataset, base_targets["appointment_id"]
    )
    policy_features = _select_features_by_ids(
        feature_dataset, policy_targets["appointment_id"]
    )
    base_target = base_targets["target"].astype("int8").reset_index(drop=True)

    selected_model = str(calibration_manifest["selected_ranking_model"])
    selected_method = str(calibration_manifest["selected_calibration_method"])
    if selected_model != FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL:
        raise RuntimeError("Policy ranking model differs from frozen selection")
    if selected_method != "uncalibrated":
        raise RuntimeError("Policy implementation must not apply an unselected calibrator")

    estimator = build_v2_candidate_estimator(selected_model, config)
    estimator.fit(base_features, base_target)
    probability = _positive_probability(estimator, policy_features)

    indexed = feature_dataset.set_index("appointment_id", verify_integrity=True)
    policy_ids = policy_targets["appointment_id"].to_numpy(dtype=np.int64)
    metadata = indexed.loc[
        policy_ids,
        ["prediction_time", "evaluation_partition", "label_available_at"],
    ].reset_index()
    metadata["prediction_time"] = pd.to_datetime(
        metadata["prediction_time"], errors="raise", format="mixed"
    ).astype("datetime64[ns]")
    metadata["label_available_at"] = pd.to_datetime(
        metadata["label_available_at"], errors="raise", format="mixed"
    ).astype("datetime64[ns]")
    metadata["evaluation_partition"] = metadata["evaluation_partition"].astype("string")

    predictions = metadata.copy(deep=True)
    predictions["target"] = policy_targets["target"].to_numpy(dtype=np.int8)
    predictions["no_show_probability"] = probability
    predictions = predictions.loc[:, list(POLICY_PREDICTION_COLUMNS)]

    if predictions["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Policy probabilities exposed final_test")
    if not predictions["evaluation_partition"].eq("policy_selection").all():
        raise RuntimeError("Policy predictions contain an unexpected partition")
    if predictions["label_available_at"].ge(decision_time).any():
        raise RuntimeError("Policy predictions include an immature label")
    if predictions["appointment_id"].duplicated().any():
        raise RuntimeError("Policy prediction appointment IDs are not unique")

    scenarios = _build_policy_scenarios(predictions, config)
    summary = {
        "selected_ranking_model": selected_model,
        "selected_calibration_method": selected_method,
        "base_fit_time": base_fit_time.isoformat(),
        "base_training_rows": int(len(base_target)),
        "base_training_positive_count": int(base_target.sum()),
        "policy_decision_time": decision_time.isoformat(),
        "policy_selection_rows": int(len(predictions)),
        "policy_selection_positive_count": int(predictions["target"].sum()),
        "policy_selection_positive_rate": float(predictions["target"].mean()),
        "capacity_fractions": list(CAPACITY_FRACTIONS),
        "false_negative_to_false_positive_cost_ratios": list(FN_FP_COST_RATIOS),
        "scenario_count": int(len(scenarios)),
        "single_operational_threshold_selected": False,
        "model_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "model_contract_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "rolling_origin_manifest_sha256": FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
        "calibration_manifest_sha256": FROZEN_V2_CALIBRATION_MANIFEST_SHA256,
        "policy_execution_spec_sha256": FROZEN_V2_POLICY_EXECUTION_SPEC_SHA256,
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
    }
    return V2PolicySensitivityResult(
        predictions=predictions,
        scenarios=scenarios,
        summary=summary,
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


def export_v2_policy_sensitivity_results(
    result: V2PolicySensitivityResult,
    *,
    output_dir: Path = DEFAULT_V2_POLICY_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic R2 policy outputs and a hash manifest."""
    destination = Path(output_dir)
    paths = {
        "predictions": destination / "policy_predictions.csv",
        "scenarios": destination / "policy_scenarios.csv",
        "summary": destination / "policy_summary.json",
        "manifest": destination / "policy_manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError("Policy outputs already exist; use overwrite=True to replace them")

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(result.predictions, paths["predictions"])
    _write_csv(result.scenarios, paths["scenarios"])
    paths["summary"].write_text(
        json.dumps(dict(result.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifact_entries: dict[str, dict[str, object]] = {}
    for key in ("predictions", "scenarios", "summary"):
        path = paths[key]
        artifact_entries[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    manifest = {
        "schema_version": "1.0.0",
        "phase": "R2",
        "stage": "policy_sensitivity",
        "model_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "model_contract_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "rolling_origin_manifest_sha256": FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
        "calibration_manifest_sha256": FROZEN_V2_CALIBRATION_MANIFEST_SHA256,
        "policy_execution_spec_sha256": FROZEN_V2_POLICY_EXECUTION_SPEC_SHA256,
        "processed_dataset_sha256": EXPECTED_PROCESSED_DATASET_SHA256,
        "processed_manifest_sha256": EXPECTED_PROCESSED_MANIFEST_SHA256,
        "processed_dataset_fingerprint": EXPECTED_PROCESSED_FINGERPRINT,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "selected_ranking_model": result.summary["selected_ranking_model"],
        "selected_calibration_method": result.summary["selected_calibration_method"],
        "base_training_rows": result.summary["base_training_rows"],
        "base_training_positive_count": result.summary["base_training_positive_count"],
        "policy_decision_time": result.summary["policy_decision_time"],
        "policy_selection_rows": result.summary["policy_selection_rows"],
        "policy_selection_positive_count": result.summary[
            "policy_selection_positive_count"
        ],
        "policy_selection_positive_rate": result.summary[
            "policy_selection_positive_rate"
        ],
        "scenario_count": result.summary["scenario_count"],
        "single_operational_threshold_selected": False,
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
        description="Run frozen Version 2 R2 policy-sensitivity analysis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_V2_POLICY_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_v2_policy_sensitivity()
    manifest = export_v2_policy_sensitivity_results(
        result,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print("Policy sensitivity scenarios:")
    print(result.scenarios.to_string(index=False))
    print()
    print(
        "Policy rows / positives: "
        f"{result.summary['policy_selection_rows']:,} / "
        f"{result.summary['policy_selection_positive_count']:,}"
    )
    print(f"Scenario count: {result.summary['scenario_count']}")
    print("Selected operational threshold: none")
    print("Final-test target accessed: false")
    print("Final-test probabilities generated: false")
    print(
        "Policy manifest: "
        f"{Path(args.output_dir) / 'policy_manifest.json'}"
    )
    print(f"Manifest stage: {manifest['stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
