"""Generate the sealed, target-free Version 2 protected-test probability vector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.v2_development import _positive_class_probability
from src.modeling.v2_persistence import (
    DEFAULT_V2_PERSISTENCE_DIR,
    FROZEN_V2_R3_CONFIG_SHA256,
    MANIFEST_FILENAME as PERSISTENCE_MANIFEST_FILENAME,
    METADATA_FILENAME as PERSISTENCE_METADATA_FILENAME,
    PIPELINE_FILENAME as PERSISTENCE_PIPELINE_FILENAME,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V2_R3_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_r3_execution.json"
DEFAULT_V2_DIAGNOSTICS_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "diagnostics"
)
DEFAULT_V2_FINAL_TEST_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "final_test"
)

FROZEN_V2_R3_CONFIG_SHA256 = (
    "c0b259a4bb81790a30fd6e2c2fd2495e10869d700ae783196c5eb055db46f7a5"
)
FROZEN_V2_PIPELINE_SHA256 = (
    "301029bd5bee1ffe346fbf09dcc6ed4570b231458ba8a081f8a0f6bb544d9df0"
)
FROZEN_V2_PERSISTENCE_METADATA_SHA256 = (
    "33eda2b123e592813008a004b4aa3f353ac1a2bda51ca5eaddb45954eeea6224"
)
FROZEN_V2_PERSISTENCE_MANIFEST_SHA256 = (
    "ca19d477e0590f40d1abbad869119b182d05e923b2d582df19c42473d2795856"
)
FROZEN_V2_DIAGNOSTICS_MANIFEST_SHA256 = (
    "5a207b8a4984a203f64d1015c7a99b254db1108440dde71738abd3c936f9f8f2"
)

PROBABILITY_FILENAME = "final_test_probabilities.csv"
MANIFEST_FILENAME = "final_test_probability_manifest.json"

EXPECTED_FINAL_TEST_ROWS = 4343
EXPECTED_PROBABILITY_COLUMNS = (
    "appointment_id",
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


def _appointment_order_sha256(ids: np.ndarray) -> str:
    payload = "".join(f"{int(value)}\n" for value in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_target_free_final_test_probabilities(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    persistence_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
    diagnostics_dir: Path = DEFAULT_V2_DIAGNOSTICS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score protected final-test feature rows without loading any target."""

    r3_config_path = Path(DEFAULT_V2_R3_CONFIG_PATH)
    if _sha256(r3_config_path) != FROZEN_V2_R3_CONFIG_SHA256:
        raise ValueError("Frozen R3 execution config SHA-256 mismatch")

    config = _read_json(r3_config_path)
    protected = config["protected_final_test"]

    if protected["partition"] != "final_test":
        raise RuntimeError("Frozen protected-test partition changed")
    if int(protected["expected_feature_rows"]) != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError("Frozen protected-test row count changed")
    if tuple(protected["probability_columns"]) != EXPECTED_PROBABILITY_COLUMNS:
        raise RuntimeError("Frozen protected-test probability schema changed")
    if protected["exact_appointment_order_required"] is not True:
        raise RuntimeError("Frozen protected-test order requirement changed")
    if protected["probability_sha256_seal_required_before_target_access"] is not True:
        raise RuntimeError("Frozen probability seal requirement changed")
    if protected["probability_commit_and_ci_green_required_before_target_access"] is not True:
        raise RuntimeError("Frozen CI gate requirement changed")
    if protected["target_access_requires_explicit_allow_test_true"] is not True:
        raise RuntimeError("Frozen one-time target gate requirement changed")
    if protected["probabilities_generated_at_contract_freeze"] is not False:
        raise RuntimeError("R3 contract was not frozen before probability generation")
    if protected["target_accessed_at_contract_freeze"] is not False:
        raise RuntimeError("R3 contract was not frozen before target access")

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
    if persistence_manifest["selected_ranking_model"] != "logistic_regression":
        raise RuntimeError("Persisted selected model changed")
    if persistence_manifest["selected_calibration_method"] != "uncalibrated":
        raise RuntimeError("Persisted calibration method changed")
    if persistence_manifest["final_test_target_accessed"] is not False:
        raise RuntimeError("Persistence checkpoint records target access")
    if persistence_manifest["final_test_probabilities_generated"] is not False:
        raise RuntimeError("Persistence checkpoint records premature test scoring")

    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_manifest_path = diagnostics_dir / "pretest_diagnostics_manifest.json"
    if _sha256(diagnostics_manifest_path) != FROZEN_V2_DIAGNOSTICS_MANIFEST_SHA256:
        raise ValueError("Frozen diagnostics manifest SHA-256 mismatch")
    diagnostics_manifest = _read_json(diagnostics_manifest_path)
    if diagnostics_manifest["final_test_target_accessed"] is not False:
        raise RuntimeError("Diagnostics checkpoint records target access")
    if diagnostics_manifest["final_test_probabilities_generated"] is not False:
        raise RuntimeError("Diagnostics checkpoint records premature test scoring")

    feature_dataset = load_frozen_v2_processed_feature_dataset(Path(processed_dir))
    if "target" in feature_dataset.columns:
        raise RuntimeError("Target reached the frozen processed feature dataset")

    partition = feature_dataset["evaluation_partition"].astype("string")
    final_rows = feature_dataset.loc[partition.eq("final_test")].copy()

    if len(final_rows) != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_TEST_ROWS} protected-test rows, "
            f"found {len(final_rows)}"
        )
    if not final_rows["appointment_id"].is_unique:
        raise RuntimeError("Protected-test appointment IDs must be unique")

    appointment_ids = final_rows["appointment_id"].to_numpy(
        dtype=np.int64,
        copy=True,
    )
    features = final_rows.loc[:, list(V2_MODEL_FEATURE_COLUMNS)].reset_index(drop=True)

    estimator = joblib.load(pipeline_path)
    if not isinstance(estimator, Pipeline):
        raise RuntimeError("Frozen persisted artifact is not an sklearn Pipeline")

    probabilities = _positive_class_probability(estimator, features)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.shape != (EXPECTED_FINAL_TEST_ROWS,):
        raise RuntimeError("Protected-test probability vector has unexpected shape")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Protected-test probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("Protected-test probabilities must lie within [0, 1]")

    vector = pd.DataFrame(
        {
            "appointment_id": appointment_ids,
            "no_show_probability": probabilities,
        },
        columns=list(EXPECTED_PROBABILITY_COLUMNS),
    )

    metadata = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "protected_final_test_probability_seal",
        "r3_execution_config_sha256": FROZEN_V2_R3_CONFIG_SHA256,
        "pipeline_sha256": FROZEN_V2_PIPELINE_SHA256,
        "persistence_metadata_sha256": FROZEN_V2_PERSISTENCE_METADATA_SHA256,
        "persistence_manifest_sha256": FROZEN_V2_PERSISTENCE_MANIFEST_SHA256,
        "diagnostics_manifest_sha256": FROZEN_V2_DIAGNOSTICS_MANIFEST_SHA256,
        "selected_ranking_model": "logistic_regression",
        "selected_calibration_method": "uncalibrated",
        "partition": "final_test",
        "prediction_time_start": protected["prediction_time"]["start"],
        "prediction_time_end": protected["prediction_time"]["end"],
        "row_count": int(len(vector)),
        "probability_columns": list(EXPECTED_PROBABILITY_COLUMNS),
        "appointment_order_rule": "frozen_processed_feature_dataset_row_order",
        "appointment_order_sha256": _appointment_order_sha256(appointment_ids),
        "probabilities_finite": True,
        "probabilities_within_unit_interval": True,
        "probability_metrics_computed": False,
        "target_access_requires_explicit_allow_test_true": True,
        "probability_commit_and_ci_green_required_before_target_access": True,
        "single_operational_threshold_selected": False,
        "final_test_probabilities_generated": True,
        "final_test_target_accessed": False,
    }
    return vector, metadata


def export_target_free_final_test_probabilities(
    vector: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_V2_FINAL_TEST_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the protected-test probability vector and its immutable seal manifest."""

    output_dir = Path(output_dir)
    probability_path = output_dir / PROBABILITY_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    existing = [path for path in (probability_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Protected-test probability seal outputs already exist; "
            "use overwrite=True only before the seal is committed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    vector.to_csv(
        probability_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )

    replay = pd.read_csv(
        probability_path,
        dtype={
            "appointment_id": "int64",
            "no_show_probability": "float64",
        },
        float_precision="round_trip",
    )
    if list(replay.columns) != list(EXPECTED_PROBABILITY_COLUMNS):
        raise RuntimeError("Written protected-test probability schema changed")
    if replay["appointment_id"].tolist() != vector["appointment_id"].tolist():
        raise RuntimeError("Written protected-test appointment order changed")
    if not np.array_equal(
        replay["no_show_probability"].to_numpy(dtype=np.float64, copy=True),
        vector["no_show_probability"].to_numpy(dtype=np.float64, copy=True),
    ):
        raise RuntimeError(
            "Written protected-test probabilities do not exactly round-trip"
        )

    manifest = dict(metadata)
    manifest["artifacts"] = {
        PROBABILITY_FILENAME: {
            "sha256": _sha256(probability_path),
            "size_bytes": int(probability_path.stat().st_size),
        }
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def run_and_export_target_free_final_test_probabilities(
    *,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
    persistence_dir: Path = DEFAULT_V2_PERSISTENCE_DIR,
    diagnostics_dir: Path = DEFAULT_V2_DIAGNOSTICS_DIR,
    output_dir: Path = DEFAULT_V2_FINAL_TEST_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    vector, metadata = build_target_free_final_test_probabilities(
        processed_dir=processed_dir,
        persistence_dir=persistence_dir,
        diagnostics_dir=diagnostics_dir,
    )
    return export_target_free_final_test_probabilities(
        vector,
        metadata,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and SHA-256 seal the target-free Version 2 protected-test "
            "probability vector without loading protected targets."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_V2_PROCESSED_DIR)
    parser.add_argument(
        "--persistence-dir",
        type=Path,
        default=DEFAULT_V2_PERSISTENCE_DIR,
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DEFAULT_V2_DIAGNOSTICS_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V2_FINAL_TEST_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest = run_and_export_target_free_final_test_probabilities(
        processed_dir=args.processed_dir,
        persistence_dir=args.persistence_dir,
        diagnostics_dir=args.diagnostics_dir,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    output_dir = Path(args.output_dir)
    artifact = manifest["artifacts"][PROBABILITY_FILENAME]

    print("Protected final-test probability vector generated and sealed.")
    print(f"Rows: {manifest['row_count']:,}")
    print(
        "Probability vector SHA-256: "
        f"{artifact['sha256']}"
    )
    print(
        "Appointment-order SHA-256: "
        f"{manifest['appointment_order_sha256']}"
    )
    print(
        "Manifest SHA-256: "
        f"{_sha256(output_dir / MANIFEST_FILENAME)}"
    )
    print("Probability metrics computed: false")
    print("Final-test probabilities generated: true")
    print("Final-test target accessed: false")
    print("Target access remains blocked pending commit + CI green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
