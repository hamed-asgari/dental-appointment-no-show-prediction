from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from src.modeling.v2_final_reporting import (
    FIGURE_FILENAMES,
    export_final_reporting,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "modeling" / "v2"
REPORTING_DIR = REPORT_DIR / "final_reporting"
FIGURE_DIR = ROOT / "reports" / "figures"
SUMMARY = REPORTING_DIR / "final_reporting_summary.json"
MANIFEST = REPORTING_DIR / "final_reporting_manifest.json"
MODEL_CARD = ROOT / "docs" / "v2_model_card.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
SOURCE = ROOT / "src" / "modeling" / "v2_final_reporting.py"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_reporting_manifest_preserves_post_test_boundary() -> None:
    manifest = _json(MANIFEST)
    assert manifest["stage"] == "final_reporting_package"
    assert manifest["selected_app_type"] == "transparent_model_evaluation_dashboard"
    assert manifest["target_reaccess_performed"] is False
    assert manifest["model_refit_performed"] is False
    assert manifest["calibration_change_performed"] is False
    assert manifest["final_test_threshold_selected"] is False
    assert manifest["post_test_model_tuning_permitted"] is False


def test_final_reporting_summary_records_frozen_evidence() -> None:
    summary = _json(SUMMARY)
    assert summary["model"]["selected_ranking_model"] == "logistic_regression"
    assert summary["model"]["selected_calibration_method"] == "uncalibrated"
    assert summary["model"]["feature_count"] == 32
    assert summary["final_test"]["sample_size"] == 4343
    assert summary["final_test"]["positive_count"] == 358
    assert summary["app_decision"]["selected_app_type"] == (
        "transparent_model_evaluation_dashboard"
    )
    assert summary["app_decision"][
        "passes_all_appointment_level_risk_demo_requirements"
    ] is False
    assert summary["app_decision"]["checks"][
        "brier_score_vs_population_prior"
    ]["passes"] is False
    assert summary["reporting_boundary"]["invokes_protected_target_accessor"] is False
    assert summary["reporting_boundary"]["post_test_model_tuning_permitted"] is False


def test_final_reporting_artifact_hashes_and_png_signatures() -> None:
    manifest = _json(MANIFEST)
    for filename, entry in manifest["artifacts"].items():
        if filename == SUMMARY.name:
            path = SUMMARY
        else:
            path = FIGURE_DIR / filename
        assert _digest(path) == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]

    for filename in FIGURE_FILENAMES:
        path = FIGURE_DIR / filename
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000


def test_reporting_runner_reproduces_committed_outputs_byte_identically(
    tmp_path: Path,
) -> None:
    temp_figures = tmp_path / "figures"
    temp_summary = tmp_path / "summary.json"
    temp_manifest = tmp_path / "manifest.json"

    export_final_reporting(
        figure_dir=temp_figures,
        summary_path=temp_summary,
        manifest_path=temp_manifest,
        overwrite=False,
    )

    assert temp_summary.read_bytes() == SUMMARY.read_bytes()
    for filename in FIGURE_FILENAMES:
        assert (temp_figures / filename).read_bytes() == (
            FIGURE_DIR / filename
        ).read_bytes()

    committed = _json(MANIFEST)
    replay = _json(temp_manifest)
    assert replay == committed


def test_reporting_source_cannot_reaccess_protected_target() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    accessor_needle = "load_verified_v2_final_test_" + "targets"
    allow_needle = "allow_" + "test=True"
    assert accessor_needle not in text
    assert allow_needle not in text


def test_model_card_documents_decision_and_one_command_reproduction() -> None:
    text = MODEL_CARD.read_text(encoding="utf-8")
    assert "transparent_model_evaluation_dashboard" in text
    assert "Brier" in text
    assert "0.076205" in text
    assert "0.075687" in text
    assert "not validated for clinical or operational use" in text
    assert (
        ".\\.venv\\Scripts\\python.exe -m src.modeling.v2_final_reporting --overwrite"
        in text
    )


def test_recovery_plan_marks_r3_reporting_work_implemented_not_closed() -> None:
    text = RECOVERY_PLAN.read_text(encoding="utf-8")
    r3 = text[text.index("## Phase R3"):text.index("## Phase R4")]
    assert "final reporting package implemented" in r3
    assert "CI verification pending before R3 closeout" in r3
    assert "- [x] Modeling runner implemented" in text
    assert "- [ ] Figures and screenshots committed" in text
