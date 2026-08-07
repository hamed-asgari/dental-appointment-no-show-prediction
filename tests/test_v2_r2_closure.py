from __future__ import annotations

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "v2_r2_completion_evidence.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
DOCS_INDEX = ROOT / "docs" / "README.md"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
REPORTS_README = ROOT / "reports" / "modeling" / "v2" / "README.md"
POLICY_MANIFEST = (
    ROOT / "reports" / "modeling" / "v2" / "policy" / "policy_manifest.json"
)

EXPECTED_POLICY_MANIFEST_SHA256 = (
    "33391bc9295c6a9d93bb8797e8e3836fb9af45062f1b431a4debc9bc7822e4a4"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def test_r2_completion_evidence_records_checkpoints_and_ci() -> None:
    text = _text(EVIDENCE)
    required = (
        "**Phase R2 is complete on the recovery branch.**",
        "`d06929e`",
        "`891221d`",
        "`1891ade`",
        "`e5192c6`",
        "`db0e02b`",
        "full local test suite: 1640 passed",
        "GitHub CI run: 31176713663",
        "GitHub CI job: 92860198254",
        "GitHub CI status: successful",
    )
    for value in required:
        assert value in text


def test_r2_completion_evidence_records_frozen_identities() -> None:
    text = _text(EVIDENCE)
    required = (
        "e575b10835645d3a643c396803cfff21f5c1c1cdad9b988ee07037ef045beb45",
        "5b2e701753a2d0e0d2f9a7efaddf46b2f316643e66dd8c11727373927c8a5d7a",
        "4e12c2db3a95ed096040e558b567106a7569a07f3fdec8fb2d28570dedc90863",
        EXPECTED_POLICY_MANIFEST_SHA256,
        "`logistic_regression`",
        "`uncalibrated`",
        "16-scenario",
        "1,063",
        "92",
    )
    for value in required:
        assert value in text


def test_r2_completion_keeps_protected_test_closed() -> None:
    text = _normalized(EVIDENCE)
    required = (
        "final_test_target_accessed = false",
        "final_test_probabilities_generated = false",
        "No protected 2027 final-test metric has been inspected.",
        "prewritten-and-sealed probability-vector requirement",
    )
    for value in required:
        assert value in text


def test_policy_manifest_identity_is_unchanged_at_r2_closeout() -> None:
    assert sha256(POLICY_MANIFEST.read_bytes()).hexdigest() == (
        EXPECTED_POLICY_MANIFEST_SHA256
    )


def test_recovery_plan_marks_r2_complete_and_updates_checklist() -> None:
    text = _text(RECOVERY_PLAN)
    r2 = text[text.index("## Phase R2"):text.index("## Phase R3")]
    assert "**Status: complete on the recovery branch.**" in r2
    assert "docs/v2_r2_completion_evidence.md" in r2
    assert "- [x] model ranking and calibration are reported separately" in r2
    assert "- [x] deterministic policy outputs reproduce byte-identically" in r2
    assert "- [x] protected 2027 final-test targets remain unaccessed" in r2
    assert "- [x] Baselines and comparison models re-evaluated" in text
    assert "- [x] Calibration re-evaluated" in text
    assert "- [x] Threshold/cost analysis updated" in text


def test_documentation_surfaces_r2_evidence_after_r3_and_r4() -> None:
    docs = _normalized(DOCS_INDEX)
    readme = _normalized(README)
    reports = _normalized(REPORTS_README)

    assert "[Phase R2 completion evidence](v2_r2_completion_evidence.md)" in _text(
        DOCS_INDEX
    )
    assert "Version 2 recovery Phases R1 through R3 are complete" in docs
    assert "transparent_model_evaluation_dashboard" in docs
    assert "Phase R4 has implemented the read-only Streamlit evaluation dashboard" in docs
    assert "Policy sensitivity remains pending." not in docs

    assert "Recovery Phases R0 through R3 are complete." in readme
    assert "The protected 2027 final test has already been accessed exactly once" in readme
    assert "transparent_model_evaluation_dashboard" in readme
    assert "No protected final-test claim is made at this stage." not in readme
    assert "Policy sensitivity remains pending." not in readme

    assert "Policy-sensitivity artifacts are committed under `policy/`" in reports
    assert "does not select an operational threshold" in reports

def test_changelog_moves_completed_r2_modeling_out_of_planned_list() -> None:
    text = _text(CHANGELOG)
    planned = text[text.index("### Planned for Version 2.0.0"):text.index("## [1.0.0]")]
    assert "Recovered model comparison, calibration, and threshold analysis." not in planned
    assert "Closed recovery Phase R2" in text
