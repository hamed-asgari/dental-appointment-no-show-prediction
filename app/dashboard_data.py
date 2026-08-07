from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


APP_TYPE = "transparent_model_evaluation_dashboard"

SUMMARY_RELATIVE = Path(
    "reports/modeling/v2/final_reporting/final_reporting_summary.json"
)
MANIFEST_RELATIVE = Path(
    "reports/modeling/v2/final_reporting/final_reporting_manifest.json"
)
FIGURE_RELATIVES = (
    Path("reports/figures/v2_final_precision_recall_curve.png"),
    Path("reports/figures/v2_final_calibration_curve.png"),
    Path("reports/figures/v2_final_capacity_tradeoff.png"),
)

EXPECTED_SHA256 = {
    SUMMARY_RELATIVE:
        "76ed771871b696b4e0cd0c262b7d13f3bb03d03a187edd84c40934f1d2bfbbf7",
    MANIFEST_RELATIVE:
        "15f47f11e0378376baf7a2f5c520beb389ed8952f3ac161321548aeb35ca64b3",
    FIGURE_RELATIVES[0]:
        "e706d92fa13e09c5fda7da2fcd0f6a8c55332d524ae0e221d370ef792ebd7b95",
    FIGURE_RELATIVES[1]:
        "eb98404d809b0ab691decb2cb10a3d0c3b2a495cbef8637d75a468e86f25834f",
    FIGURE_RELATIVES[2]:
        "965ee9e77e35b05ae380d017b88a7fa952fd6b82bf608cb8fc71b66cf01827a1",
}

REQUIRED_SUMMARY_PATHS = (
    ("app_decision",),
    ("app_decision", "checks"),
    ("app_decision", "selected_app_type"),
    ("app_decision", "passes_all_appointment_level_risk_demo_requirements"),
    ("final_test",),
    ("model",),
    ("population_prior_baseline",),
    ("pretest_diagnostics",),
    ("reporting_boundary",),
)

REQUIRED_MANIFEST_KEYS = (
    "artifacts",
    "calibration_change_performed",
    "final_test_threshold_selected",
    "model_refit_performed",
    "post_test_model_tuning_permitted",
    "selected_app_type",
    "target_reaccess_performed",
)

PROHIBITED_TRUE_MANIFEST_FLAGS = (
    "calibration_change_performed",
    "final_test_threshold_selected",
    "model_refit_performed",
    "post_test_model_tuning_permitted",
    "target_reaccess_performed",
)

PROHIBITED_TRUE_BOUNDARY_FLAGS = (
    "changes_calibration",
    "invokes_protected_target_accessor",
    "post_test_model_tuning_permitted",
    "regenerates_or_refits_model",
    "selects_final_test_threshold",
)


class DashboardIntegrityError(RuntimeError):
    """Raised when frozen R4 dashboard inputs fail integrity validation."""


@dataclass(frozen=True)
class DashboardData:
    root: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]
    figure_paths: dict[str, Path]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DashboardIntegrityError(
            f"Required dashboard artifact is missing: {path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DashboardIntegrityError(
            f"Required dashboard JSON cannot be read safely: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise DashboardIntegrityError(
            f"Dashboard JSON must be an object: {path}"
        )
    return payload


def _require_nested(
    payload: dict[str, Any],
    path: tuple[str, ...],
) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise DashboardIntegrityError(
                f"Missing required dashboard field: {dotted}"
            )
        current = current[key]
    return current


def validate_dashboard_payload(
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    for path in REQUIRED_SUMMARY_PATHS:
        _require_nested(summary, path)

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            raise DashboardIntegrityError(
                f"Missing required manifest field: {key}"
            )

    if summary["app_decision"]["selected_app_type"] != APP_TYPE:
        raise DashboardIntegrityError(
            "Summary app decision does not match frozen R4 contract"
        )
    if manifest["selected_app_type"] != APP_TYPE:
        raise DashboardIntegrityError(
            "Manifest app decision does not match frozen R4 contract"
        )
    if summary["app_decision"][
        "passes_all_appointment_level_risk_demo_requirements"
    ]:
        raise DashboardIntegrityError(
            "Appointment-level risk demo unexpectedly passes all gates"
        )

    for key in PROHIBITED_TRUE_MANIFEST_FLAGS:
        if manifest[key] is not False:
            raise DashboardIntegrityError(
                f"Prohibited R4 manifest state is not false: {key}"
            )

    boundary = summary["reporting_boundary"]
    if not isinstance(boundary, dict):
        raise DashboardIntegrityError(
            "reporting_boundary must be an object"
        )
    for key in PROHIBITED_TRUE_BOUNDARY_FLAGS:
        if boundary.get(key) is not False:
            raise DashboardIntegrityError(
                f"Prohibited reporting boundary state: {key}"
            )

    if boundary.get("reads_committed_opened_evaluation_only") is not True:
        raise DashboardIntegrityError(
            "Dashboard reporting boundary must read committed "
            "opened evaluation only"
        )
    if boundary.get("claims_scope") != "synthetic_data_only":
        raise DashboardIntegrityError(
            "Dashboard claims scope must remain synthetic_data_only"
        )

    expected_figure_names = [
        path.name
        for path in FIGURE_RELATIVES
    ]
    if summary.get("figure_filenames") != expected_figure_names:
        raise DashboardIntegrityError(
            "Summary figure registry does not match frozen R4 contract"
        )

    manifest_artifacts = manifest["artifacts"]
    if not isinstance(manifest_artifacts, dict):
        raise DashboardIntegrityError(
            "Manifest artifacts field must be an object"
        )

    for relative in (SUMMARY_RELATIVE, *FIGURE_RELATIVES):
        record = manifest_artifacts.get(relative.name)
        if not isinstance(record, dict):
            raise DashboardIntegrityError(
                f"Manifest is missing artifact identity for {relative.name}"
            )
        if record.get("sha256") != EXPECTED_SHA256[relative]:
            raise DashboardIntegrityError(
                f"Manifest SHA-256 identity changed for {relative.name}"
            )


def load_dashboard_data(
    root: Path | None = None,
) -> DashboardData:
    root = Path(root) if root is not None else repository_root()

    for relative, expected_hash in EXPECTED_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise DashboardIntegrityError(
                f"Required dashboard artifact is missing: {relative}"
            )
        if sha256_file(path) != expected_hash:
            raise DashboardIntegrityError(
                f"Frozen dashboard artifact SHA-256 mismatch: {relative}"
            )

    summary = _load_json(root / SUMMARY_RELATIVE)
    manifest = _load_json(root / MANIFEST_RELATIVE)
    validate_dashboard_payload(summary, manifest)

    figures = {
        relative.name: root / relative
        for relative in FIGURE_RELATIVES
    }
    return DashboardData(
        root=root,
        summary=deepcopy(summary),
        manifest=deepcopy(manifest),
        figure_paths=figures,
    )


def gate_rows(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = summary["app_decision"]["checks"]
    ap = checks[
        "average_precision_absolute_uplift_vs_population_prior"
    ]
    brier = checks["brier_score_vs_population_prior"]
    log_loss = checks["log_loss_vs_population_prior"]
    roc = checks["roc_auc"]
    return [
        {
            "Gate": "Average Precision uplift",
            "Observed": ap["observed"],
            "Requirement": f">= {ap['required_minimum']:.3f}",
            "Pass": ap["passes"],
        },
        {
            "Gate": "ROC-AUC",
            "Observed": roc["observed"],
            "Requirement": f">= {roc['required_minimum']:.2f}",
            "Pass": roc["passes"],
        },
        {
            "Gate": "Brier vs population prior",
            "Observed": brier["model"],
            "Requirement": f"<= {brier['baseline']:.6f}",
            "Pass": brier["passes"],
        },
        {
            "Gate": "Log loss vs population prior",
            "Observed": log_loss["model"],
            "Requirement": (
                "worsening <= "
                f"{log_loss['maximum_allowed_worsening']:.3f}"
            ),
            "Pass": log_loss["passes"],
        },
    ]


def top_feature_rows(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = summary[
        "pretest_diagnostics"
    ]["top_10_permutation_features"]
    return [
        {
            "Rank": int(row["rank"]),
            "Feature": str(row["feature"]),
            "AP importance mean": float(row["importance_mean"]),
            "AP importance std": float(row["importance_std"]),
        }
        for row in rows
    ]
