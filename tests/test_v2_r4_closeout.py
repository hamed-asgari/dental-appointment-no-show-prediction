from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CLOSEOUT = ROOT / "docs" / "v2_r4_closeout.md"
DOCS_INDEX = ROOT / "docs" / "README.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def test_r4_closeout_records_exact_portfolio_ci_seal() -> None:
    text = _normalized(CLOSEOUT)

    required = (
        "R4 is complete and formally closed.",
        "568be27e410a82f3a1acb94d4e40a337431113f7",
        "31217805371",
        "transparent_model_evaluation_dashboard",
        "target_reaccess_performed = false",
        "model_refit_performed = false",
        "final_test_threshold_selected = false",
        "post_test_model_tuning_permitted = false",
    )
    for value in required:
        assert value in text


def test_r4_closeout_records_frozen_application_artifact_identities() -> None:
    text = _text(CLOSEOUT)

    digests = (
        "1069cd5f66c6638fc858fb0767dc064dc84c4578a3247eec9daecb8289b25ef2",
        "cd91aebc93dc616b351305e7eb81ca743a893a877c4a821550b4aeb88b0ff584",
        "166255030a0c27e860d5374d3a8241af545fbbb152e6d580992a3d46080e0724",
        "e155c3ebd5b29acb446196625a262b145c79e3298915d95aa85a0cb577d1f6fb",
        "809f42bcfacd919248663f515595bbb1810dcdc35a92e0f186e74446320d8e79",
        "0e7f45905b1cf2940215ebf56b53351507f8ea84722ab5df13323d9acbc48c64",
        "e6533a9432744a458327e1bfc7ee4a02b425040501bd42e2c6488a98b85a588a",
        "572772460187355a2cd0508284c2b58c87494bf069b52fb5256dfb63fd3913be",
    )
    for digest in digests:
        assert digest in text


def test_documentation_index_surfaces_formal_r4_closeout() -> None:
    text = _text(DOCS_INDEX)
    normalized = " ".join(text.split())

    assert "[Phase R4 formal closeout](v2_r4_closeout.md)" in text
    assert "Phase R4 has implemented the read-only Streamlit evaluation dashboard" in normalized
    assert "R4 is formally closed" in normalized
    assert "Phase R5 is the next recovery stage" in normalized


def test_recovery_plan_closes_r4_and_updates_completed_portfolio_items() -> None:
    text = _text(RECOVERY_PLAN)
    r4 = text[text.index("## Phase R4"):text.index("## Phase R5")]

    assert "**R4 is complete and formally closed.**" in r4
    assert "`568be27`" in r4
    assert "`31217805371`" in r4
    assert "remaining gate: exact-head CI seal and formal R4 closeout" not in r4

    required_checks = (
        "- [x] Streamlit app implemented",
        "- [x] Figures and screenshots committed",
        "- [x] README and all docs synchronized",
    )
    for value in required_checks:
        assert value in text

    assert "- [ ] Clean-environment reproduction passed" in text
    assert "- [ ] Version 2.0.0 release reviewed and published" in text


def test_root_readme_reports_r4_closed_and_r5_next() -> None:
    text = _normalized(README)

    assert "Recovery Phases R0 through R3 are complete." in text
    assert "Phase R4 is also complete and formally closed" in text
    assert "Phase R5 is the next recovery stage" in text
    assert "Version `2.0.0` is still under recovery review" in text
    assert "Formal R4 closeout still requires" not in text
    assert "release is gated by formal R4 closeout" not in text


def test_changelog_moves_completed_r4_items_out_of_planned_work() -> None:
    text = _text(CHANGELOG)
    planned = text[text.index("### Planned for Version 2.0.0"):text.index("## [1.0.0]")]

    assert "Closed recovery Phase R4" in text
    assert "Evidence-based Streamlit application." not in planned
    assert "Committed analytical figures and portfolio screenshots." not in planned
    assert "Interpretation, error analysis, and subgroup diagnostics." not in planned
    assert "Reproducible modeling runner and appropriate persisted artifacts." not in planned
    assert "Clean-environment reproduction" in planned
    assert "reviewed Version `2.0.0` release" in planned
