"""Reproducible post-access reporting for the frozen Version 2 R3 evaluation.

This module never invokes the protected-target accessor. It consumes only
committed, already-opened evaluation artifacts and regenerates the final
analytical summary and key figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FINAL_TEST_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "final_test"
)
DEFAULT_DIAGNOSTICS_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2" / "diagnostics"
)
DEFAULT_PERSISTENCE_DIR = _REPOSITORY_ROOT / "models" / "v2"
DEFAULT_FIGURE_DIR = _REPOSITORY_ROOT / "reports" / "figures"
DEFAULT_SUMMARY_PATH = (
    _REPOSITORY_ROOT
    / "reports"
    / "modeling"
    / "v2"
    / "final_reporting"
    / "final_reporting_summary.json"
)
DEFAULT_MANIFEST_PATH = (
    _REPOSITORY_ROOT
    / "reports"
    / "modeling"
    / "v2"
    / "final_reporting"
    / "final_reporting_manifest.json"
)

EVALUATION_MANIFEST_SHA256 = (
    "c8a2158bf98c4230bc66180d66dc4e4e88f8e3fff8b2ce0538fd58f4cf29a2af"
)
PROBABILITY_VECTOR_SHA256 = (
    "7a4af37da40c1515a6ee567dd12861b57cf08e2e65b516e3c7e3d2aa65237126"
)
PROBABILITY_MANIFEST_SHA256 = (
    "412c75ff76882536eab0dc2ee5df4c3da8551c7b99fe83402ab5ac0f679b46e4"
)
DIAGNOSTICS_MANIFEST_SHA256 = (
    "5a207b8a4984a203f64d1015c7a99b254db1108440dde71738abd3c936f9f8f2"
)
PERSISTENCE_METADATA_SHA256 = (
    "33eda2b123e592813008a004b4aa3f353ac1a2bda51ca5eaddb45954eeea6224"
)

FIGURE_FILENAMES = (
    "v2_final_precision_recall_curve.png",
    "v2_final_calibration_curve.png",
    "v2_final_capacity_tradeoff.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_source_artifacts(
    *,
    final_test_dir: Path,
    diagnostics_dir: Path,
    persistence_dir: Path,
) -> dict[str, Any]:
    final_test_dir = Path(final_test_dir)
    diagnostics_dir = Path(diagnostics_dir)
    persistence_dir = Path(persistence_dir)

    paths = {
        "evaluation_manifest": final_test_dir / "final_test_evaluation_manifest.json",
        "probability_vector": final_test_dir / "final_test_probabilities.csv",
        "probability_manifest": final_test_dir / "final_test_probability_manifest.json",
        "diagnostics_manifest": diagnostics_dir / "pretest_diagnostics_manifest.json",
        "persistence_metadata": persistence_dir / "frozen_logistic_pipeline.metadata.json",
    }
    expected = {
        "evaluation_manifest": EVALUATION_MANIFEST_SHA256,
        "probability_vector": PROBABILITY_VECTOR_SHA256,
        "probability_manifest": PROBABILITY_MANIFEST_SHA256,
        "diagnostics_manifest": DIAGNOSTICS_MANIFEST_SHA256,
        "persistence_metadata": PERSISTENCE_METADATA_SHA256,
    }
    for key, path in paths.items():
        actual = _sha256(path)
        if actual != expected[key]:
            raise ValueError(f"Frozen reporting source identity changed: {key}={actual}")

    evaluation_manifest = _read_json(paths["evaluation_manifest"])
    if evaluation_manifest["final_test_target_accessed"] is not True:
        raise RuntimeError("Committed evaluation no longer records opened target")
    if evaluation_manifest["target_access_count_this_evaluation_batch"] != 1:
        raise RuntimeError("Committed one-time target-access evidence changed")
    if evaluation_manifest["single_operational_threshold_selected"] is not False:
        raise RuntimeError("Committed evaluation selected an operational threshold")
    if evaluation_manifest["post_test_model_tuning_permitted"] is not False:
        raise RuntimeError("Committed evaluation permits post-test tuning")

    for filename, entry in evaluation_manifest["artifacts"].items():
        artifact_path = final_test_dir / filename
        if _sha256(artifact_path) != entry["sha256"]:
            raise ValueError(f"Committed evaluation artifact changed: {filename}")
        if artifact_path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Committed evaluation artifact size changed: {filename}")

    diagnostics_manifest = _read_json(paths["diagnostics_manifest"])
    for filename, entry in diagnostics_manifest["artifacts"].items():
        artifact_path = diagnostics_dir / filename
        if _sha256(artifact_path) != entry["sha256"]:
            raise ValueError(f"Committed diagnostic artifact changed: {filename}")
        if artifact_path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Committed diagnostic artifact size changed: {filename}")

    return {
        "evaluation_manifest": evaluation_manifest,
        "source_hashes": expected,
    }


def build_final_reporting_summary(
    *,
    final_test_dir: Path = DEFAULT_FINAL_TEST_DIR,
    diagnostics_dir: Path = DEFAULT_DIAGNOSTICS_DIR,
    persistence_dir: Path = DEFAULT_PERSISTENCE_DIR,
) -> dict[str, Any]:
    """Build the final R3 report summary from committed post-access artifacts."""

    verified = _verify_source_artifacts(
        final_test_dir=final_test_dir,
        diagnostics_dir=diagnostics_dir,
        persistence_dir=persistence_dir,
    )

    final_test_dir = Path(final_test_dir)
    diagnostics_dir = Path(diagnostics_dir)
    persistence_dir = Path(persistence_dir)

    metrics = _read_json(final_test_dir / "final_test_metrics.json")
    decision = _read_json(final_test_dir / "final_test_app_decision.json")
    diagnostics = _read_json(diagnostics_dir / "diagnostics_summary.json")
    persistence = _read_json(
        persistence_dir / "frozen_logistic_pipeline.metadata.json"
    )

    if metrics["sample_size"] != 4343 or metrics["positive_count"] != 358:
        raise RuntimeError("Final-test population identity changed")
    if decision["selected_app_type"] != "transparent_model_evaluation_dashboard":
        raise RuntimeError("Pre-frozen R4 app decision changed")
    if decision["passes_all_appointment_level_risk_demo_requirements"] is not False:
        raise RuntimeError("Pre-frozen R4 app gate result changed")
    if persistence["selected_ranking_model"] != "logistic_regression":
        raise RuntimeError("Frozen persisted model identity changed")
    if persistence["selected_calibration_method"] != "uncalibrated":
        raise RuntimeError("Frozen calibration choice changed")

    return {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "final_reporting_package",
        "source_hashes": verified["source_hashes"],
        "model": {
            "selected_ranking_model": persistence["selected_ranking_model"],
            "selected_calibration_method": persistence[
                "selected_calibration_method"
            ],
            "feature_count": int(persistence["model_feature_count"]),
            "base_fit_time": persistence["base_fit_time"],
            "base_training_rows": int(persistence["base_training_rows"]),
            "base_training_positive_count": int(
                persistence["base_training_positive_count"]
            ),
            "base_training_prevalence": float(
                persistence["base_training_prevalence"]
            ),
        },
        "final_test": {
            "sample_size": int(metrics["sample_size"]),
            "positive_count": int(metrics["positive_count"]),
            "prevalence": float(metrics["prevalence"]),
            "average_precision": float(metrics["model"]["average_precision"]),
            "roc_auc": float(metrics["model"]["roc_auc"]),
            "brier_score": float(metrics["model"]["brier_score"]),
            "log_loss": float(metrics["model"]["log_loss"]),
            "calibration_intercept": float(
                metrics["model"]["calibration_intercept"]
            ),
            "calibration_slope": float(metrics["model"]["calibration_slope"]),
            "mean_predicted_probability": float(
                metrics["model"]["mean_predicted_probability"]
            ),
        },
        "population_prior_baseline": {
            "probability": float(metrics["population_prior_probability"]),
            "average_precision": float(
                metrics["population_prior_baseline"]["average_precision"]
            ),
            "roc_auc": float(metrics["population_prior_baseline"]["roc_auc"]),
            "brier_score": float(
                metrics["population_prior_baseline"]["brier_score"]
            ),
            "log_loss": float(
                metrics["population_prior_baseline"]["log_loss"]
            ),
        },
        "model_minus_baseline": {
            "average_precision_absolute_uplift": float(
                metrics[
                    "average_precision_absolute_uplift_vs_population_prior"
                ]
            ),
            "brier_delta": float(
                metrics["brier_delta_model_minus_population_prior"]
            ),
            "log_loss_delta": float(
                metrics["log_loss_delta_model_minus_population_prior"]
            ),
        },
        "app_decision": {
            "selected_app_type": decision["selected_app_type"],
            "passes_all_appointment_level_risk_demo_requirements": bool(
                decision[
                    "passes_all_appointment_level_risk_demo_requirements"
                ]
            ),
            "checks": decision["checks"],
            "claims_scope": decision["claims_scope"],
        },
        "pretest_diagnostics": {
            "partition": diagnostics["diagnostic_partition"],
            "sample_size": int(diagnostics["sample_size"]),
            "positive_count": int(diagnostics["positive_count"]),
            "supported_subgroup_row_count": int(
                diagnostics["supported_subgroup_row_count"]
            ),
            "top_10_permutation_features": diagnostics[
                "top_10_permutation_features"
            ],
        },
        "reporting_boundary": {
            "reads_committed_opened_evaluation_only": True,
            "invokes_protected_target_accessor": False,
            "regenerates_or_refits_model": False,
            "changes_calibration": False,
            "selects_final_test_threshold": False,
            "post_test_model_tuning_permitted": False,
            "claims_scope": "synthetic_data_only",
        },
        "figure_filenames": list(FIGURE_FILENAMES),
    }


def _load_plot_inputs(final_test_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_test_dir = Path(final_test_dir)
    predictions = pd.read_csv(
        final_test_dir / "final_test_evaluation_predictions.csv",
        dtype={
            "appointment_id": "int64",
            "target": "int8",
            "no_show_probability": "float64",
        },
        float_precision="round_trip",
    )
    scenarios = pd.read_csv(
        final_test_dir / "final_test_policy_scenarios.csv"
    )
    if len(predictions) != 4343:
        raise RuntimeError("Final reporting predictions row count changed")
    if predictions["target"].sum() != 358:
        raise RuntimeError("Final reporting predictions positive count changed")
    return predictions, scenarios


def _save_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)


def render_final_figures(
    *,
    final_test_dir: Path = DEFAULT_FINAL_TEST_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> tuple[Path, Path, Path]:
    """Render deterministic final analytical figures from committed predictions."""

    final_test_dir = Path(final_test_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    predictions, scenarios = _load_plot_inputs(final_test_dir)
    target = predictions["target"].to_numpy(dtype=np.int8, copy=True)
    probability = predictions["no_show_probability"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    prevalence = float(target.mean())

    pr_path = figure_dir / FIGURE_FILENAMES[0]
    precision, recall, _ = precision_recall_curve(target, probability)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(recall, precision, label="Frozen logistic model")
    ax.axhline(
        prevalence,
        linestyle="--",
        label=f"Population prevalence ({prevalence:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Version 2 protected final test: precision-recall curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    _save_png(fig, pr_path)

    calibration_path = figure_dir / FIGURE_FILENAMES[1]
    fraction_positive, mean_predicted = calibration_curve(
        target,
        probability,
        n_bins=10,
        strategy="quantile",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Frozen logistic model",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Ideal calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed no-show fraction")
    ax.set_title("Version 2 protected final test: calibration")
    ax.set_xlim(0.0, max(0.35, float(mean_predicted.max()) * 1.08))
    ax.set_ylim(0.0, max(0.35, float(fraction_positive.max()) * 1.08))
    ax.legend()
    fig.tight_layout()
    _save_png(fig, calibration_path)

    capacity_path = figure_dir / FIGURE_FILENAMES[2]
    capacity = (
        scenarios.loc[
            scenarios["scenario_family"].eq("capacity_cost"),
            ["capacity_fraction", "precision", "recall"],
        ]
        .drop_duplicates(subset=["capacity_fraction"])
        .sort_values("capacity_fraction")
        .reset_index(drop=True)
    )
    if not np.allclose(
        capacity["capacity_fraction"].to_numpy(dtype=float),
        np.array([0.05, 0.10, 0.20], dtype=float),
    ):
        raise RuntimeError("Registered final-test capacity grid changed")

    x = capacity["capacity_fraction"].to_numpy(dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, capacity["precision"], marker="o", label="Precision")
    ax.plot(x, capacity["recall"], marker="o", label="Recall")
    ax.set_xlabel("Appointment capacity reviewed (%)")
    ax.set_ylabel("Metric")
    ax.set_title("Registered capacity sensitivity on protected final test")
    ax.set_xticks(x)
    ax.set_ylim(0.0, max(0.40, float(capacity["recall"].max()) * 1.12))
    ax.legend()
    fig.tight_layout()
    _save_png(fig, capacity_path)

    return pr_path, calibration_path, capacity_path


def export_final_reporting(
    *,
    final_test_dir: Path = DEFAULT_FINAL_TEST_DIR,
    diagnostics_dir: Path = DEFAULT_DIAGNOSTICS_DIR,
    persistence_dir: Path = DEFAULT_PERSISTENCE_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Regenerate final R3 analytical outputs without target re-access."""

    summary_path = Path(summary_path)
    manifest_path = Path(manifest_path)
    figure_dir = Path(figure_dir)
    output_paths = [
        summary_path,
        manifest_path,
        *(figure_dir / name for name in FIGURE_FILENAMES),
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Final reporting outputs already exist; use --overwrite to "
            "reproduce derived reporting artifacts."
        )

    summary = build_final_reporting_summary(
        final_test_dir=final_test_dir,
        diagnostics_dir=diagnostics_dir,
        persistence_dir=persistence_dir,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    figure_paths = render_final_figures(
        final_test_dir=final_test_dir,
        figure_dir=figure_dir,
    )

    artifacts: dict[str, dict[str, object]] = {
        DEFAULT_SUMMARY_PATH.name: {
            "sha256": _sha256(summary_path),
            "size_bytes": int(summary_path.stat().st_size),
        }
    }
    for path in figure_paths:
        artifacts[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    manifest = {
        "schema_version": "1.0.0",
        "phase": "R3",
        "stage": "final_reporting_package",
        "source_hashes": summary["source_hashes"],
        "selected_app_type": summary["app_decision"]["selected_app_type"],
        "target_reaccess_performed": False,
        "model_refit_performed": False,
        "calibration_change_performed": False,
        "final_test_threshold_selected": False,
        "post_test_model_tuning_permitted": False,
        "artifacts": artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the frozen Version 2 R3 final analytical summary and "
            "figures from committed evaluation artifacts only."
        )
    )
    parser.add_argument("--final-test-dir", type=Path, default=DEFAULT_FINAL_TEST_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--persistence-dir", type=Path, default=DEFAULT_PERSISTENCE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifest = export_final_reporting(
        final_test_dir=args.final_test_dir,
        diagnostics_dir=args.diagnostics_dir,
        persistence_dir=args.persistence_dir,
        figure_dir=args.figure_dir,
        summary_path=args.summary_path,
        manifest_path=args.manifest_path,
        overwrite=bool(args.overwrite),
    )

    print("Frozen Version 2 R3 final reporting package generated.")
    print(
        "Summary SHA-256: "
        f"{manifest['artifacts'][Path(args.summary_path).name]['sha256']}"
    )
    for filename in FIGURE_FILENAMES:
        print(
            f"{filename} SHA-256: "
            f"{manifest['artifacts'][filename]['sha256']}"
        )
    print(
        "Reporting manifest SHA-256: "
        f"{_sha256(Path(args.manifest_path))}"
    )
    print(
        "Selected app type: "
        f"{manifest['selected_app_type']}"
    )
    print("Target re-access performed: false")
    print("Model refit performed: false")
    print("Final-test threshold selected: false")
    print("Post-test model tuning permitted: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
