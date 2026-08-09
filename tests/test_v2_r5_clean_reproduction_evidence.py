from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "reports"
    / "release"
    / "v2"
    / "r5_clean_reproduction_evidence.json"
)
DOC = ROOT / "docs" / "v2_r5_clean_reproduction_evidence.md"
README = ROOT / "README.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
CHANGELOG = ROOT / "CHANGELOG.md"

EXPECTED_EVIDENCE_SHA256 = (
    "a28fc45030b9d83dc9f2f36debc2d99e"
    "d2f34d5323650c1381a88613cec6ec95"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def test_r5_clean_reproduction_evidence_exact_identity() -> None:
    assert EVIDENCE.is_file()
    assert EVIDENCE.stat().st_size == 3597
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == (
        EXPECTED_EVIDENCE_SHA256
    )


def test_r5_clean_reproduction_evidence_records_success_and_source() -> None:
    evidence = json.loads(_text(EVIDENCE))

    assert evidence["schema_version"] == "1.0.0"
    assert evidence["success"] is True
    assert evidence["source_commit"] == (
        "5e527e6a45f0b65e9889268d7b93585f931f3fdb"
    )
    assert evidence["python"] == {
        "base_version": "3.12.13",
        "clean_version": "3.12.13",
    }
    assert evidence["dependency_integrity"]["pip_check_exit"] == 0
    assert evidence["full_test_suite_passed"] is True


def test_r5_clean_reproduction_evidence_records_byte_identity() -> None:
    evidence = json.loads(_text(EVIDENCE))

    assert evidence["raw_reproduction"]["files"][
        "v2_synthetic_benchmark.manifest.json"
    ] == "7702fa5fa0638c52dd0598e28f35f678fb5d61a886faadf9b38a6e292fdcd561"
    assert evidence["processed_reproduction"]["files"][
        "v2_feature_dataset.csv"
    ] == "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53"
    assert evidence["reporting_reproduction"]["summary_sha256"] == (
        "76ed771871b696b4e0cd0c262b7d13f3bb03d03a187edd84c40934f1d2bfbbf7"
    )
    assert evidence["reporting_reproduction"]["manifest_sha256"] == (
        "15f47f11e0378376baf7a2f5c520beb389ed8952f3ac161321548aeb35ca64b3"
    )


def test_r5_clean_reproduction_evidence_records_app_and_repo_health() -> None:
    evidence = json.loads(_text(EVIDENCE))

    assert evidence["markdown_links"] == {
        "broken": 0,
        "checked": 64,
    }
    assert evidence["tracked_tree_hygiene"] == {
        "forbidden_paths": [],
        "passed": True,
    }
    assert evidence["streamlit"]["health"] == "ok"
    assert evidence["canonical_checkout_clean_after_run"] is True
    assert evidence["disposable_checkout_clean_after_run"] is True


def test_r5_clean_reproduction_evidence_preserves_scientific_boundary() -> None:
    evidence = json.loads(_text(EVIDENCE))

    false_flags = (
        "forbidden_project_modules_invoked",
        "protected_target_reaccess_performed",
        "model_refit_performed",
        "calibration_change_performed",
        "threshold_selection_performed",
        "post_test_tuning_performed",
    )
    for key in false_flags:
        assert evidence[key] is False

    assert evidence["safe_project_module_commands"] == [
        "src.synthetic.export",
        "src.data.export_v2_processed",
        "src.modeling.v2_final_reporting",
    ]


def test_r5_clean_reproduction_evidence_document_records_gate() -> None:
    text = _normalized(DOC)

    required = (
        "R5.1 clean-environment reproduction passed.",
        "31321490041",
        EXPECTED_EVIDENCE_SHA256,
        "checked 64 repository-relative Markdown links with zero broken links",
        "health result `ok`",
        "Version `2.0.0` release metadata must remain unchanged",
    )
    for value in required:
        assert value in text


def test_current_docs_preserve_r5_1_evidence_after_release() -> None:
    readme = _normalized(README)
    plan = _text(RECOVERY_PLAN)
    changelog = _normalized(CHANGELOG)

    assert "R5.1 clean-environment reproduction and its exact-head evidence CI seal remain preserved" in readme
    assert "Version `2.0.0` was published" in readme
    assert "- [x] Clean-environment reproduction passed" in plan
    assert "- [x] CI passed" in plan
    assert "- [x] Version 2.0.0 release reviewed and published" in plan
    assert "Recovery Phase R5.1 evidence" in changelog
