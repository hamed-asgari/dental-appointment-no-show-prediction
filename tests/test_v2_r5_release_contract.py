from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "v2_r5_release_execution_contract.md"
DOCS_INDEX = ROOT / "docs" / "README.md"
RECOVERY_PLAN = ROOT / "docs" / "v2.0.0_recovery_plan.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def test_r5_contract_freezes_exact_r4_input_checkpoint() -> None:
    text = _normalized(CONTRACT)
    required = (
        "5efaa685e5b82f60e1ada165daf9bf7130ee8721",
        "31219799513",
        "transparent_model_evaluation_dashboard",
        "Frozen before R5 release execution.",
    )
    for value in required:
        assert value in text


def test_r5_contract_prohibits_protected_test_reexecution() -> None:
    text = _text(CONTRACT)
    required = (
        "allow_test=True",
        "src.modeling.v2_final_test_evaluation",
        "src.modeling.v2_final_test_probabilities",
        "refit or overwrite the persisted frozen Version 2 model",
        "choose a threshold from protected-test outcomes",
    )
    for value in required:
        assert value in text


def test_r5_contract_requires_clean_disposable_reproduction() -> None:
    text = _normalized(CONTRACT)
    required = (
        "disposable checkout/worktree",
        "requirements.lock.txt",
        "python -m pip check",
        "complete repository test suite",
        "Streamlit health endpoint",
        "canonical repository working tree byte-clean",
    )
    for value in required:
        assert value in text


def test_r5_contract_freezes_release_metadata_transition() -> None:
    text = _normalized(CONTRACT)
    required = (
        "change `pyproject.toml` version from `1.0.0` to `2.0.0`",
        "change `CITATION.cff` version from `1.0.0` to `2.0.0`",
        "without claiming that GitHub publication has already occurred",
        "release-preparation commit must pass the complete local test suite",
    )
    for value in required:
        assert value in text


def test_r5_contract_preserves_recovery_history_and_v1_release() -> None:
    text = _normalized(CONTRACT)
    required = (
        "merge using a **merge commit**",
        "do not squash or rebase",
        "Version 1 tag, release, and assets must not be modified",
        "Squash merge and rebase merge are prohibited",
    )
    for value in required:
        assert value in text


def test_r5_contract_requires_main_ci_before_v2_tag() -> None:
    text = _normalized(CONTRACT)
    assert "A Version `2.0.0` tag must not be created before the exact merged-main CI succeeds." in text
    assert "create annotated tag `v2.0.0`" in text
    assert "SHA256SUMS.txt" in text
    assert "No PyPI publication is part of the current R5 contract." in text


def test_r5_contract_requires_post_release_evidence_before_completion() -> None:
    text = _normalized(CONTRACT)
    required = (
        "Publication is not the end of the audit trail.",
        "post-release evidence",
        "completed Version `2.0.0` checklist",
        "must also pass CI before R5 is declared complete",
    )
    for value in required:
        assert value in text


def test_docs_index_and_recovery_plan_surface_frozen_r5_contract() -> None:
    docs = _text(DOCS_INDEX)
    plan = _normalized(RECOVERY_PLAN)

    assert "[Phase R5 release-execution contract](v2_r5_release_execution_contract.md)" in docs
    assert (
        "**R5 execution contract is frozen and CI-sealed. "
        "R5.1 runner implementation is in progress; "
        "clean-environment evidence has not yet been accepted.**"
        in plan
    )
    assert "docs/v2_r5_release_execution_contract.md" in plan
    assert "- [ ] Clean-environment reproduction passed" in _text(RECOVERY_PLAN)
    assert "- [ ] Version 2.0.0 release reviewed and published" in _text(RECOVERY_PLAN)
def test_r5_contract_freezes_pre_finalization_documentation_scope() -> None:
    text = _normalized(CONTRACT)

    required = (
        "top-level `reports/README.md`",
        "top-level `data/README.md`",
        "`docs/v2_portfolio_communication.md`",
        "CV/resume bullets",
        "LinkedIn project post draft",
        "30-second, 90-second, and 5-minute project explanations",
        "zero broken links",
        "local-only IDE, build, cache, egg-info, and virtual-environment",
        "historical Version 1 statements",
    )
    for value in required:
        assert value in text


def test_r5_contract_keeps_pre_release_metadata_transition_ordered() -> None:
    text = _normalized(CONTRACT)

    assert (
        "current pre-release `1.0.0` values in `pyproject.toml` and "
        "`CITATION.cff` are therefore not defects at contract-freeze time"
        in text
    )
    assert (
        "They must change together during R5.2, after clean-environment "
        "evidence is committed and CI-sealed."
        in text
    )


def test_r5_recovery_plan_tracks_final_portfolio_documentation_outputs() -> None:
    text = _text(RECOVERY_PLAN)

    required = (
        "top-level `data/README.md` and `reports/README.md` navigation coverage",
        "final portfolio communication packet in `docs/v2_portfolio_communication.md`",
        "tracked-tree hygiene for IDE/build/cache/egg-info artifacts",
        "`docs/v2_portfolio_communication.md` containing:",
        "CV/resume bullets",
        "LinkedIn project post",
        "30-second explanation",
        "90-second explanation",
        "5-minute explanation",
        "synthetic-data and validation limitations",
    )
    for value in required:
        assert value in text
