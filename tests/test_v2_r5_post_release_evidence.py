from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "release" / "v2" / "r5_post_release_evidence.json"
R51 = ROOT / "reports" / "release" / "v2" / "r5_clean_reproduction_evidence.json"
README = ROOT / "README.md"
DOCS = ROOT / "docs" / "README.md"
PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
NOTES = ROOT / "docs" / "v2.0.0_release_notes.md"
POST = ROOT / "docs" / "v2_r5_post_release_evidence.md"
REPORTS = ROOT / "reports" / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _norm(path: Path) -> str:
    return " ".join(_text(path).split())


def test_release_identity_and_main_ci_are_recorded() -> None:
    e = json.loads(_text(EVIDENCE))
    assert e["phase"] == "R5.5"
    assert e["release"]["tag"] == "v2.0.0"
    assert e["release"]["tag_type"] == "annotated"
    assert e["release"]["tag_object_sha"] == "be6dedbb2cc80ef855c0d90c0de20d8993d46a2b"
    assert e["release"]["peeled_commit"] == "cf7ecca52bc54aeeccdad032b80be83f9d172fc9"
    assert e["release"]["published_at"] == "2026-08-09T19:30:29Z"
    assert e["release"]["draft"] is False
    assert e["release"]["prerelease"] is False
    assert e["release_source"]["commit"] == "cf7ecca52bc54aeeccdad032b80be83f9d172fc9"
    assert e["release_source"]["main_ci_run_id"] == "31328381538"
    assert e["release_source"]["main_ci_conclusion"] == "success"


def test_final_release_asset_identities_are_recorded() -> None:
    e = json.loads(_text(EVIDENCE))
    assert e["assets"]["wheel"]["name"] == "dental_appointment_no_show_prediction-2.0.0-py3-none-any.whl"
    assert e["assets"]["wheel"]["size_bytes"] == 156042
    assert e["assets"]["wheel"]["sha256"] == "0c3ac6fe0e4fa0a911ab6027694cff9729c4bb95800b560c223faeddc41ec230"
    assert e["assets"]["checksums"]["name"] == "SHA256SUMS.txt"
    assert e["assets"]["checksums"]["size_bytes"] == 127
    assert e["assets"]["checksums"]["sha256"] == "f1b4b2b0fc5ec4900919a6b96f0e10b30122ae16b04485c324957a693bc14da3"
    assert e["assets"]["remote_asset_byte_identity_verified_during_r5_4"] is True


def test_r5_1_evidence_remains_byte_identical() -> None:
    assert R51.stat().st_size == 3597
    assert hashlib.sha256(R51.read_bytes()).hexdigest() == "a28fc45030b9d83dc9f2f36debc2d99ed2f34d5323650c1381a88613cec6ec95"
    e = json.loads(_text(EVIDENCE))
    assert e["r5_1_clean_reproduction"]["evidence_commit"] == "fa6f8477d0bebe13ee6f6828713b9caefea41ba8"
    assert e["r5_1_clean_reproduction"]["evidence_ci_run_id"] == "31324234415"
    assert e["r5_1_clean_reproduction"]["evidence_sha256"] == "a28fc45030b9d83dc9f2f36debc2d99ed2f34d5323650c1381a88613cec6ec95"


def test_scientific_and_product_boundary_is_unchanged() -> None:
    e = json.loads(_text(EVIDENCE))
    assert e["application"]["decision"] == "transparent_model_evaluation_dashboard"
    assert e["application"]["individualized_patient_risk_calculator"] is False
    assert all(value is False for value in e["scientific_safety"].values())


def test_current_docs_report_published_release_and_pending_r5_5_ci() -> None:
    readme = _norm(README)
    docs = _norm(DOCS)
    plan = _text(PLAN)
    notes = _norm(NOTES)
    reports = _norm(REPORTS)

    assert "Version `2.0.0` was published" in readme
    assert "31328381538" in readme
    assert "0c3ac6fe0e4fa0a911ab6027694cff9729c4bb95800b560c223faeddc41ec230" in readme
    assert "still under recovery review" not in readme
    assert "has **not** yet been released" not in readme

    assert "[R5.5 post-release evidence](v2_r5_post_release_evidence.md)" in _text(DOCS)
    assert "Phase R5 release execution reached publication in R5.4" in docs

    assert "- [x] CI passed" in plan
    assert "- [x] Version 2.0.0 release reviewed and published" in plan
    assert "- [x] R5.5 post-release evidence recorded on a separate post-release branch" in plan
    assert "- [ ] R5.5 post-release evidence exact-head PR CI passed" in plan
    assert "- [ ] R5.5 post-release evidence merged to main and R5 formally closed" in plan

    assert "Published release status" in notes
    assert "A GitHub Release has **not** yet been published" not in notes
    assert "GitHub Release `Version 2.0.0` is published" in reports
    assert "GitHub Release publication is still pending." not in reports


def test_human_evidence_and_changelog_are_synchronized() -> None:
    doc = _text(POST)
    changelog = _text(CHANGELOG)
    unreleased = changelog[
        changelog.index("## [Unreleased]"):changelog.index("## [2.0.0]")
    ]
    for value in (
        "be6dedbb2cc80ef855c0d90c0de20d8993d46a2b",
        "cf7ecca52bc54aeeccdad032b80be83f9d172fc9",
        "31328381538",
        "0c3ac6fe0e4fa0a911ab6027694cff9729c4bb95800b560c223faeddc41ec230",
        "f1b4b2b0fc5ec4900919a6b96f0e10b30122ae16b04485c324957a693bc14da3",
        "a28fc45030b9d83dc9f2f36debc2d99ed2f34d5323650c1381a88613cec6ec95",
        "transparent_model_evaluation_dashboard",
        "no protected-target re-access",
        "exact-head pull-request CI",
    ):
        assert value in doc
    assert "### R5.5 post-release evidence" in unreleased
    assert "Planned for Version 2.0.0" not in unreleased


def test_version_1_is_recorded_as_unchanged() -> None:
    e = json.loads(_text(EVIDENCE))
    v1 = e["version_1_immutability"]
    assert v1["tag"] == "v1.0.0"
    assert v1["peeled_commit"] == "ec5aa5d84c8de6aaf7daca4c299073ba12d4e700"
    assert v1["release_name"] == "Version 1.0.0"
    assert v1["wheel_name"] == "dental_appointment_no_show_prediction-1.0.0-py3-none-any.whl"
    assert v1["wheel_size_bytes"] == 60712
    assert v1["unchanged"] is True
