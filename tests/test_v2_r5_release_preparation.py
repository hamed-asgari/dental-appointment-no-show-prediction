from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
import yaml


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
CHANGELOG = ROOT / "CHANGELOG.md"
README = ROOT / "README.md"
DOCS_INDEX = ROOT / "docs" / "README.md"
PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
RELEASE_NOTES = ROOT / "docs" / "v2.0.0_release_notes.md"
PORTFOLIO = ROOT / "docs" / "v2_portfolio_communication.md"
DATA_README = ROOT / "data" / "README.md"
REPORTS_README = ROOT / "reports" / "README.md"
EVIDENCE = ROOT / "reports" / "release" / "v2" / "r5_clean_reproduction_evidence.json"

EXPECTED_EVIDENCE_SHA256 = (
    "a28fc45030b9d83dc9f2f36debc2d99e"
    "d2f34d5323650c1381a88613cec6ec95"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def test_r5_2_release_metadata_is_prepared_together() -> None:
    with PYPROJECT.open("rb") as stream:
        project = tomllib.load(stream)["project"]
    citation = yaml.safe_load(_text(CITATION))
    assert project["version"] == "2.0.0"
    assert citation["version"] == "2.0.0"
    assert citation["license"] == "MIT"
    assert "doi" not in citation
    assert "date-released" not in citation


def test_r5_2_required_release_and_portfolio_docs_exist() -> None:
    for path in (RELEASE_NOTES, PORTFOLIO, DATA_README, REPORTS_README):
        assert path.is_file()


def test_release_notes_distinguish_v1_v2_without_claiming_publication() -> None:
    text = _normalized(RELEASE_NOTES)
    required = (
        "Version `1.0.0` remains an immutable methodological and audit checkpoint",
        "Version `2.0.0` completes the portfolio recovery",
        "`transparent_model_evaluation_dashboard`",
        "A GitHub Release has **not** yet been published.",
        "All appointment and patient data are synthetic.",
        "no external clinical validation",
    )
    for value in required:
        assert value in text


def test_portfolio_communication_has_required_formats_and_limits() -> None:
    text = _text(PORTFOLIO)
    required = (
        "## CV / resume bullets",
        "## LinkedIn project post draft",
        "## 30-second explanation",
        "## 90-second explanation",
        "## 5-minute explanation",
        "## Synthetic-data and external-validation limitations",
        "synthetic",
        "external clinical validation",
        "not a deployable clinical decision-support tool",
    )
    for value in required:
        assert value in text


def test_top_level_guides_surface_frozen_boundaries() -> None:
    data = _normalized(DATA_README)
    reports = _normalized(REPORTS_README)
    assert "fully synthetic" in data
    assert "strict as-of rule" in data
    assert "target-free" in data
    assert "accessed exactly once" in data
    assert "protected final test was accessed exactly once" in reports
    assert "No post-test model, feature, calibration, or threshold tuning is permitted." in reports
    assert "GitHub Release publication is still pending." in reports


def test_changelog_moves_completed_v2_work_into_2_0_0() -> None:
    text = _text(CHANGELOG)
    assert text.index("## [Unreleased]") < text.index("## [2.0.0]")
    assert text.index("## [2.0.0]") < text.index("## [1.0.0]")
    unreleased = text[text.index("## [Unreleased]"):text.index("## [2.0.0]")]
    v2 = text[text.index("## [2.0.0]"):text.index("## [1.0.0]")]
    assert "exact-main CI" in unreleased
    assert "GitHub Release publication" in unreleased
    assert "Recovery Phase R5.1 evidence" not in unreleased
    assert "Recovery Phase R5.1 evidence" in v2
    assert "Release preparation" in v2


def test_current_docs_report_release_prep_not_publication() -> None:
    readme = _normalized(README)
    plan = _normalized(PLAN)
    assert "Version `2.0.0` is still under recovery review" in readme
    assert "has **not** yet been released" in readme
    assert "R5.2 now prepares package/citation version `2.0.0`" in readme
    assert "R5.1 is complete and CI-sealed." in plan
    assert "R5.2 release preparation is implemented in this recovery commit" in plan
    assert "GitHub release publication has not occurred." in plan
    assert "- [ ] CI passed" in _text(PLAN)
    assert "- [ ] Version 2.0.0 release reviewed and published" in _text(PLAN)


def test_docs_index_surfaces_r5_2_materials() -> None:
    text = _text(DOCS_INDEX)
    for value in (
        "v2.0.0_release_notes.md",
        "v2_portfolio_communication.md",
        "../data/README.md",
        "../reports/README.md",
    ):
        assert value in text


def test_r5_1_evidence_bytes_remain_immutable() -> None:
    raw = EVIDENCE.read_bytes()
    assert len(raw) == 3597
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_EVIDENCE_SHA256
    evidence = json.loads(raw.decode("utf-8"))
    for key in (
        "forbidden_project_modules_invoked",
        "protected_target_reaccess_performed",
        "model_refit_performed",
        "calibration_change_performed",
        "threshold_selection_performed",
        "post_test_tuning_performed",
    ):
        assert evidence[key] is False
