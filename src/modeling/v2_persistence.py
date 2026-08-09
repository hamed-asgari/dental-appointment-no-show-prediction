"""Persist and verify the frozen Version 2 R3 Logistic Regression pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.data.build_v2_dataset import load_verified_v2_raw_tables
from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.data.v2_targets import build_mature_v2_target_table
from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.v2_development import (
    FROZEN_V2_MODEL_CONFIG_SHA256,
    FROZEN_V2_MODEL_CONTRACT_SHA256,
    _load_frozen_model_config,
    _positive_class_probability,
    _select_features_by_ids,
    build_v2_candidate_estimator,
)
from src.modeling.v2_rolling_origin_hashes import (
    FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
    FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
)
from src.modeling.v2_calibration_hashes import (
    FROZEN_V2_CALIBRATION_MANIFEST_SHA256,
    FROZEN_V2_SELECTED_CALIBRATION_METHOD,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_R3_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_r3_execution.json"
FROZEN_V2_R3_CONFIG_SHA256 = (
    "c0b259a4bb81790a30fd6e2c2fd2495e10869d700ae783196c5eb055db46f7a5"
)
DEFAULT_V2_POLICY_DIR = _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "policy"
FROZEN_V2_POLICY_MANIFEST_SHA256 = (
    "33391bc9295c6a9d93bb8797e8e3836fb9af45062f1b431a4debc9bc7822e4a4"
)
DEFAULT_V2_PERSISTENCE_DIR = _REPOSITORY_ROOT / "models" / "v2"

PIPELINE_FILENAME = "frozen_logistic_pipeline.joblib"
METADATA_FILENAME = "frozen_logistic_pipeline.metadata.json"
MANIFEST_FILENAME = "frozen_logistic_pipeline.manifest.json"

EXPECTED_FINAL_TEST_FEATURE_ROWS = 4343
POLICY_REPLAY_ATOL = 5e-12
POLICY_REPLAY_RTOL = 5e-11


@dataclass(frozen=True, slots=True)
class V2FrozenPipelineBuild:
    """Frozen fitted pipeline plus evidence used for persistence."""

    estimator: Pipeline
    metadata: Mapping[str, Any]
    policy_features: pd.DataFrame
    policy_probabilities: np.ndarray


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


def _load_frozen_r3_config(
    path: Path = DEFAULT_V2_R3_CONFIG_PATH,
) -> dict[str, Any]:
    path = Path(path)
    if _sha256(path) != FROZEN_V2_R3_CONFIG_SHA256:
        raise ValueError("Frozen R3 execution config SHA-256 mismatch")
    config = _read_json(path)
    if config.get("status") != (
        "frozen_before_any_r3_final_test_probability_or_target_access"
    ):
        raise ValueError("R3 execution config is not in the frozen pre-test state")
    protected = config["protected_final_test"]
    if bool(protected["probabilities_generated_at_contract_freeze"]):
        raise RuntimeError("R3 contract was frozen after final-test probabilities")
    if bool(protected["target_accessed_at_contract_freeze"]):
        raise RuntimeError("R3 contract was frozen after final-test target access")
    return config


def _load_frozen_policy_manifest(
    policy_dir: Path = DEFAULT_V2_POLICY_DIR,
) -> dict[str, Any]:
    policy_dir = Path(policy_dir)
    path = policy_dir / "policy_manifest.json"
    if _sha256(path) != FROZEN_V2_POLICY_MANIFEST_SHA256:
        raise ValueError("Frozen policy manifest SHA-256 mismatch")
    manifest = _read_json(path)

    if manifest.get("selected_ranking_model") != FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL:
        raise ValueError("Policy manifest selected model changed")
    if manifest.get("selected_calibration_method") != FROZEN_V2_SELECTED_CALIBRATION_METHOD:
        raise ValueError("Policy manifest selected calibration method changed")
    if manifest.get("single_operational_threshold_selected") is not False:
        raise RuntimeError("Policy manifest unexpectedly selected an operational threshold")
    if manifest.get("final_test_target_accessed") is not False:
        raise RuntimeError("Policy manifest exposed protected final-test target")
    if manifest.get("final_test_probabilities_generated") is not False:
        raise RuntimeError("Policy manifest generated final-test probabilities")
    return manifest


def _load_frozen_policy_replay(
    *,
    policy_dir: Path,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    artifact = manifest["artifacts"]["policy_predictions.csv"]
    path = Path(policy_dir) / "policy_predictions.csv"
    if _sha256(path) != str(artifact["sha256"]):
        raise ValueError("Frozen policy prediction artifact SHA-256 mismatch")
    if path.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError("Frozen policy prediction artifact size mismatch")

    frame = pd.read_csv(
        path,
        usecols=["appointment_id", "no_show_probability"],
        dtype={"appointment_id": "int64", "no_show_probability": "float64"},
    )
    if frame.empty:
        raise ValueError("Frozen policy replay artifact must not be empty")
    if not frame["appointment_id"].is_unique:
        raise ValueError("Frozen policy replay appointment IDs must be unique")
    probability = frame["no_show_probability"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(probability).all():
        raise ValueError("Frozen policy replay probabilities must be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("Frozen policy replay probabilities must lie within [0, 1]")
    return frame


def build_v2_frozen_pipeline(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    policy_dir: Path = DEFAULT_V2_POLICY_DIR,
) -> V2FrozenPipelineBuild:
    """Fit the exact frozen R2-selected base estimator without touching final_test."""

    r3 = _load_frozen_r3_config()
    model_config = _load_frozen_model_config()
    policy_manifest = _load_frozen_policy_manifest(Path(policy_dir))

    upstream = r3["upstream"]
    frozen_model = r3["frozen_model"]
    if upstream["selected_ranking_model"] != FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL:
        raise RuntimeError("R3 selected ranking model differs from frozen R2 selection")
    if upstream["selected_calibration_method"] != FROZEN_V2_SELECTED_CALIBRATION_METHOD:
        raise RuntimeError("R3 selected calibration differs from frozen R2 selection")
    if FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL != "logistic_regression":
        raise RuntimeError("R3 persistence expects frozen Logistic Regression selection")
    if FROZEN_V2_SELECTED_CALIBRATION_METHOD != "uncalibrated":
        raise RuntimeError("R3 persistence must not apply an unselected calibrator")
    if bool(frozen_model["base_estimator_refit_after_calibration"]):
        raise RuntimeError("R3 contract unexpectedly permits post-calibration refit")
    if bool(frozen_model["refit_on_calibration_or_policy_selection_data"]):
        raise RuntimeError("R3 contract unexpectedly permits later-data refit")

    feature_dataset = load_frozen_v2_processed_feature_dataset(Path(processed_dir))
    if "target" in feature_dataset.columns:
        raise RuntimeError("Target reached the frozen feature artifact")

    final_rows = int(
        feature_dataset["evaluation_partition"].astype("string").eq("final_test").sum()
    )
    if final_rows != EXPECTED_FINAL_TEST_FEATURE_ROWS:
        raise RuntimeError("Frozen final-test feature partition identity changed")

    calibration = model_config["calibration"]
    base_fit_time = pd.Timestamp(frozen_model["base_refit_time"])
    if base_fit_time != pd.Timestamp(calibration["base_refit_time"]):
        raise RuntimeError("R3 base refit time differs from frozen R2 calibration contract")

    base_partitions = tuple(str(value) for value in frozen_model["base_training_partitions"])
    expected_partitions = tuple(str(value) for value in calibration["base_training_partitions"])
    if base_partitions != expected_partitions:
        raise RuntimeError("R3 base training partitions differ from frozen R2 contract")
    if "final_test" in base_partitions or "policy_selection" in base_partitions:
        raise PermissionError("Frozen R3 persistence attempted to train on a prohibited partition")

    tables = load_verified_v2_raw_tables()
    base_targets = build_mature_v2_target_table(
        feature_dataset,
        tables.appointments,
        model_fit_time=base_fit_time,
        allowed_partitions=base_partitions,
    )
    if base_targets["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Persistence training target access exposed final_test")
    if set(base_targets["target"].unique()) != {0, 1}:
        raise RuntimeError("Frozen persistence training target must contain both classes")

    base_rows = int(len(base_targets))
    base_positives = int(base_targets["target"].sum())
    if base_rows != int(policy_manifest["base_training_rows"]):
        raise RuntimeError("Persistence base-training row count differs from frozen policy run")
    if base_positives != int(policy_manifest["base_training_positive_count"]):
        raise RuntimeError(
            "Persistence base-training positive count differs from frozen policy run"
        )

    base_features = _select_features_by_ids(
        feature_dataset,
        base_targets["appointment_id"],
    )
    base_target = base_targets["target"].astype("int8").reset_index(drop=True)

    estimator = build_v2_candidate_estimator(
        FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
        model_config,
    )
    estimator.fit(base_features, base_target)

    replay = _load_frozen_policy_replay(
        policy_dir=Path(policy_dir),
        manifest=policy_manifest,
    )
    if len(replay) != int(policy_manifest["policy_selection_rows"]):
        raise RuntimeError("Frozen policy replay row count changed")

    policy_features = _select_features_by_ids(
        feature_dataset,
        replay["appointment_id"],
    )
    current_probability = _positive_class_probability(estimator, policy_features)
    frozen_probability = replay["no_show_probability"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    absolute_error = np.abs(current_probability - frozen_probability)
    max_abs_error = float(absolute_error.max(initial=0.0))
    if not np.allclose(
        current_probability,
        frozen_probability,
        atol=POLICY_REPLAY_ATOL,
        rtol=POLICY_REPLAY_RTOL,
    ):
        raise RuntimeError(
            "Fresh frozen-model fit does not replay the committed policy probabilities"
        )

    metadata = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "frozen_model_persistence",
        "r3_execution_config_sha256": FROZEN_V2_R3_CONFIG_SHA256,
        "model_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "model_contract_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "rolling_origin_manifest_sha256": FROZEN_V2_ROLLING_ORIGIN_MANIFEST_SHA256,
        "calibration_manifest_sha256": FROZEN_V2_CALIBRATION_MANIFEST_SHA256,
        "policy_manifest_sha256": FROZEN_V2_POLICY_MANIFEST_SHA256,
        "selected_ranking_model": FROZEN_V2_ROLLING_ORIGIN_SELECTED_MODEL,
        "selected_calibration_method": FROZEN_V2_SELECTED_CALIBRATION_METHOD,
        "base_fit_time": base_fit_time.isoformat(),
        "base_training_partitions": list(base_partitions),
        "base_training_rows": base_rows,
        "base_training_positive_count": base_positives,
        "base_training_prevalence": float(base_target.mean()),
        "model_feature_count": int(len(V2_MODEL_FEATURE_COLUMNS)),
        "model_feature_columns": list(V2_MODEL_FEATURE_COLUMNS),
        "policy_prediction_replay_rows": int(len(replay)),
        "policy_prediction_replay_max_absolute_error": max_abs_error,
        "policy_prediction_replay_atol": POLICY_REPLAY_ATOL,
        "policy_prediction_replay_rtol": POLICY_REPLAY_RTOL,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "final_test_feature_rows_observed_without_scoring": final_rows,
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
        "single_operational_threshold_selected": False,
    }
    return V2FrozenPipelineBuild(
        estimator=estimator,
        metadata=metadata,
        policy_features=policy_features,
        policy_probabilities=current_probability,
    )


def export_v2_frozen_pipeline(
    build: V2FrozenPipelineBuild,
    *,
    output_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist the frozen pipeline, verify reload replay, and write its manifest."""

    output_dir = Path(output_dir)
    paths = {
        "pipeline": output_dir / PIPELINE_FILENAME,
        "metadata": output_dir / METADATA_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError("Frozen persistence outputs already exist; use overwrite=True")

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(build.estimator, paths["pipeline"], compress=3)

    loaded = joblib.load(paths["pipeline"])
    if not isinstance(loaded, Pipeline):
        raise RuntimeError("Reloaded frozen artifact is not an sklearn Pipeline")
    replay_probability = _positive_class_probability(loaded, build.policy_features)
    if not np.allclose(
        replay_probability,
        build.policy_probabilities,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("Reloaded pipeline does not exactly replay in-memory probabilities")

    paths["metadata"].write_text(
        json.dumps(dict(build.metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifact_entries: dict[str, dict[str, object]] = {}
    for key in ("pipeline", "metadata"):
        path = paths[key]
        artifact_entries[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    manifest = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "frozen_model_persistence",
        "r3_execution_config_sha256": FROZEN_V2_R3_CONFIG_SHA256,
        "policy_manifest_sha256": FROZEN_V2_POLICY_MANIFEST_SHA256,
        "selected_ranking_model": build.metadata["selected_ranking_model"],
        "selected_calibration_method": build.metadata["selected_calibration_method"],
        "base_training_rows": build.metadata["base_training_rows"],
        "base_training_positive_count": build.metadata["base_training_positive_count"],
        "model_feature_count": build.metadata["model_feature_count"],
        "policy_prediction_replay_rows": build.metadata["policy_prediction_replay_rows"],
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


def run_and_export_v2_frozen_pipeline(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    policy_dir: Path = DEFAULT_V2_POLICY_DIR,
    output_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    build = build_v2_frozen_pipeline(
        processed_dir=processed_dir,
        policy_dir=policy_dir,
    )
    return export_v2_frozen_pipeline(
        build,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist the exact frozen Version 2 R3 Logistic Regression pipeline."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_V2_PROCESSED_DIR)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_V2_POLICY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V2_PERSISTENCE_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest = run_and_export_v2_frozen_pipeline(
        processed_dir=args.processed_dir,
        policy_dir=args.policy_dir,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    output_dir = Path(args.output_dir)
    print("Frozen R3 persistence artifact created.")
    print(f"Selected model: {manifest['selected_ranking_model']}")
    print(f"Selected calibration: {manifest['selected_calibration_method']}")
    print(
        "Base rows / positives: "
        f"{manifest['base_training_rows']:,} / "
        f"{manifest['base_training_positive_count']:,}"
    )
    print(f"Policy replay rows: {manifest['policy_prediction_replay_rows']:,}")
    print(f"Pipeline SHA-256: {manifest['artifacts'][PIPELINE_FILENAME]['sha256']}")
    print(f"Metadata SHA-256: {manifest['artifacts'][METADATA_FILENAME]['sha256']}")
    print(f"Manifest SHA-256: {_sha256(output_dir / MANIFEST_FILENAME)}")
    print("Final-test target accessed: false")
    print("Final-test probabilities generated: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
