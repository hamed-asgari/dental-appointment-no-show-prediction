from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "r5_clean_reproduction.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "r5_clean_reproduction",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r5_clean_reproduction_runner_exists_and_uses_safe_modules_only() -> None:
    runner = _load_runner()

    assert runner.SAFE_PROJECT_MODULE_COMMANDS == (
        "src.synthetic.export",
        "src.data.export_v2_processed",
        "src.modeling.v2_final_reporting",
    )
    assert "src.modeling.v2_final_test_evaluation" in (
        runner.FORBIDDEN_PROJECT_MODULES
    )
    assert "src.modeling.v2_final_test_probabilities" in (
        runner.FORBIDDEN_PROJECT_MODULES
    )
    assert "src.modeling.v2_persistence" in runner.FORBIDDEN_PROJECT_MODULES
    assert not (
        set(runner.SAFE_PROJECT_MODULE_COMMANDS)
        & set(runner.FORBIDDEN_PROJECT_MODULES)
    )


def test_r5_runner_freezes_frozen_screenshot_and_figure_hashes() -> None:
    runner = _load_runner()

    assert runner.SCREENSHOT_SHA256[
        "reports/screenshots/v2_streamlit_overview.png"
    ] == "809f42bcfacd919248663f515595bbb1810dcdc35a92e0f186e74446320d8e79"
    assert runner.FIGURE_SHA256[
        "reports/figures/v2_final_precision_recall_curve.png"
    ] == "e706d92fa13e09c5fda7da2fcd0f6a8c55332d524ae0e221d370ef792ebd7b95"


def test_markdown_link_audit_accepts_existing_relative_links() -> None:
    runner = _load_runner()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "target.md").write_text("# Target\n", encoding="utf-8")
        (root / "README.md").write_text(
            "[Target](target.md)\n",
            encoding="utf-8",
        )
        checked, broken = runner._relative_markdown_links(root)

    assert checked == 1
    assert broken == []


def test_markdown_link_audit_reports_missing_relative_links() -> None:
    runner = _load_runner()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text(
            "[Missing](does-not-exist.md)\n",
            encoding="utf-8",
        )
        checked, broken = runner._relative_markdown_links(root)

    assert checked == 1
    assert len(broken) == 1
    assert "does-not-exist.md" in broken[0]


def test_tracked_tree_hygiene_is_defined_for_release_local_artifacts() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    required = (
        '.idea/',
        '.venv/',
        '.pytest_cache/',
        'build/',
        '.egg-info',
        '__pycache__',
        '.pyc',
    )
    for value in required:
        assert value in source


def test_runner_requires_exact_full_source_commit_and_evidence_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "--source-commit" in source
    assert "--evidence-json" in source
    assert "40-character SHA" in source
    assert "Canonical checkout HEAD does not match --source-commit" in source


def test_runner_records_scientific_immutability_flags_false() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    required = (
        '"protected_target_reaccess_performed": False',
        '"model_refit_performed": False',
        '"calibration_change_performed": False',
        '"threshold_selection_performed": False',
        '"post_test_tuning_performed": False',
        '"forbidden_project_modules_invoked": False',
    )
    for value in required:
        assert value in source


def test_runner_uses_disposable_worktree_clean_venv_and_streamlit_health() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    required = (
        '"worktree",',
        '"add",',
        '"--detach",',
        '"venv",',
        '"requirements.lock.txt"',
        '"pip",',
        '"check",',
        '"pytest",',
        '"no:cacheprovider"',
        "/_stcore/health",
        '"worktree",',
        '"remove",',
    )
    for value in required:
        assert value in source
