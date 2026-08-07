from __future__ import annotations

from pathlib import Path

from src.features.schema import (
    V2_FEATURE_DATASET_COLUMNS,
    V2_MODEL_FEATURE_COLUMNS,
)
from src.synthetic.schema import (
    APPOINTMENT_COLUMNS,
    DENTIST_COLUMNS,
    PATIENT_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DICTIONARY = ROOT / "docs" / "v2_data_dictionary.md"
R1_EVIDENCE = ROOT / "docs" / "v2_r1_completion_evidence.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
DOCS_INDEX = ROOT / "docs" / "README.md"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def test_v2_data_dictionary_documents_every_raw_column() -> None:
    text = _text(DATA_DICTIONARY)
    for column in (*PATIENT_COLUMNS, *DENTIST_COLUMNS, *APPOINTMENT_COLUMNS):
        assert f"`{column}`" in text


def test_v2_data_dictionary_documents_exact_processed_schema() -> None:
    text = _text(DATA_DICTIONARY)
    for column in V2_FEATURE_DATASET_COLUMNS:
        assert f"`{column}`" in text
    assert "38 columns" in text
    assert "32 approved model features" in text


def test_v2_data_dictionary_documents_exact_model_allowlist() -> None:
    text = _text(DATA_DICTIONARY)
    for feature in V2_MODEL_FEATURE_COLUMNS:
        assert f"`{feature}`" in text
    assert "The committed processed artifact does **not** contain the target." in text


def test_v2_data_dictionary_records_frozen_processed_identity() -> None:
    text = _text(DATA_DICTIONARY)
    required = (
        "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53",
        "2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073",
        "0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787",
        "target_included = false",
        "final_test_target_accessed = false",
    )
    for value in required:
        assert value in text


def test_r1_completion_evidence_records_test_ci_and_snapshot() -> None:
    text = _text(R1_EVIDENCE)
    required = (
        "implementation HEAD: 44765dd",
        "full local test suite: 1534 passed",
        "GitHub CI run: 31153155060",
        "GitHub CI job: 92786874768",
        "a4184151695d6e95a6ffb2a971d3b363099d6654193898c69607d76bfc84244b",
        "No recovered model selection has started",
    )
    for value in required:
        assert value in text


def test_r1_completion_evidence_keeps_protected_test_closed() -> None:
    text = _normalized(R1_EVIDENCE)
    required = (
        "target_included = false",
        "final_test_target_accessed = false",
        "No protected 2027 test metric has been inspected",
        "successful accessor path is tested only on synthetic fixtures",
    )
    for value in required:
        assert value in text


def test_recovery_plan_marks_r1_complete_and_r2_as_next_phase() -> None:
    text = _text(RECOVERY_PLAN)
    r1 = text[text.index("## Phase R1"):text.index("## Phase R2")]
    assert "**Status: complete on the recovery branch.**" in r1
    assert "docs/v2_data_dictionary.md" in r1
    assert "docs/v2_r1_completion_evidence.md" in r1
    assert "- [x] no row can use future appointment outcomes" in r1
    assert "- [x] no protected 2027 final-test metric has been inspected" in r1
    assert "- [x] Leakage-safe historical features implemented" in text
    assert "- [x] Feature leakage tests added" in text


def test_documentation_index_links_r1_closeout_documents() -> None:
    text = _text(DOCS_INDEX)
    assert "[Version 2 data dictionary](v2_data_dictionary.md)" in text
    assert "[Phase R1 completion evidence](v2_r1_completion_evidence.md)" in text
    assert "Version 2 recovery Phases R1 and R2 are complete" in text


def test_root_readme_reports_completed_r2_without_claiming_final_test_success() -> None:
    text = _normalized(README)
    required = (
        "Version 2 recovery Phases R1 and R2 are complete on the recovery branch.",
        "frozen three-fold rolling-origin comparison",
        "chronological calibration evaluation",
        "pre-registered policy-sensitivity analysis",
        "The protected 2027 final-test targets have not been accessed",
        "no final-test probability vector has been generated",
        "No protected final-test claim is made at this stage.",
    )
    for value in required:
        assert value in text
    assert "Policy sensitivity remains pending." not in text


def test_changelog_moves_completed_r1_items_out_of_planned_list() -> None:
    text = _text(CHANGELOG)
    planned = text[text.index("### Planned for Version 2.0.0"):text.index("## [1.0.0]")]
    assert "Renewed chronological evaluation policy." not in planned
    assert "Leakage-safe historical feature engineering." not in planned
    assert "Closed recovery Phase R1" in text
