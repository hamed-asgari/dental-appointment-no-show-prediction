from __future__ import annotations
from pathlib import Path
import tomllib
_ROOT = Path(
    __file__
).resolve().parents[1]
def test_pyproject_release_contract() -> None:
    path = _ROOT / "pyproject.toml"
    with path.open(
        "rb"
    ) as stream:
        metadata = tomllib.load(
            stream
        )
    project = metadata[
        "project"
    ]
    assert project[
        "name"
    ] == (
        "dental-appointment-no-show-prediction"
    )
    assert project[
        "version"
    ] == "1.0.0"
    assert project[
        "requires-python"
    ] == ">=3.12,<3.13"
    assert project[
        "description"
    ] == (
        "Reproducible leakage-controlled no-show "
        "prediction study using fully synthetic "
        "dental appointment data."
    )
    assert project[
        "urls"
    ] == {
        "Repository": (
            "https://github.com/hamed-asgari/"
            "dental-appointment-no-show-prediction"
        ),
        "Documentation": (
            "https://github.com/hamed-asgari/"
            "dental-appointment-no-show-prediction/"
            "tree/main/docs"
        ),
        "Issues": (
            "https://github.com/hamed-asgari/"
            "dental-appointment-no-show-prediction/"
            "issues"
        ),
    }
    assert project[
        "authors"
    ] == [
        {
            "name": "Hamed Asgari",
        }
    ]
    assert project[
        "dynamic"
    ] == [
        "dependencies",
    ]
    build = metadata[
        "build-system"
    ]
    assert build[
        "build-backend"
    ] == "setuptools.build_meta"
    dynamic = metadata[
        "tool"
    ][
        "setuptools"
    ][
        "dynamic"
    ]
    assert dynamic[
        "dependencies"
    ][
        "file"
    ] == [
        "requirements.txt",
    ]
def test_ci_uses_validated_runtime_and_lock() -> None:
    workflow = (
        _ROOT
        / ".github"
        / "workflows"
        / "ci.yml"
    ).read_text(
        encoding="utf-8"
    )
    required = (
        "runs-on: windows-latest",
        'python-version: "3.12"',
        "requirements.lock.txt",
        "python -m pip check",
        "python -m pytest",
        "-p no:cacheprovider",
        "--basetemp",
    )
    for value in required:
        assert value in workflow
def test_line_ending_contract() -> None:
    attributes = (
        _ROOT
        / ".gitattributes"
    ).read_text(
        encoding="utf-8"
    )
    required = (
        "*.py text eol=lf",
        "*.md text eol=lf",
        "*.toml text eol=lf",
        "*.yml text eol=lf",
        "*.txt text eol=lf",
        "*.parquet binary",
    )
    for value in required:
        assert value in attributes
def test_readme_reports_current_project_status() -> None:
    readme = (
        _ROOT
        / "README.md"
    ).read_text(
        encoding="utf-8"
    )
    normalized_readme = " ".join(
        readme.split()
    )
    assert (
        "Phases 01 through 11 are complete."
        in normalized_readme
    )
    assert (
        "final chronological test results"
        in normalized_readme
    )
    assert (
        "report final test-set performance"
        not in normalized_readme
    )
    assert (
        "Phases 01 through 10 are complete."
        not in normalized_readme
    )
    assert (
        "pyproject.toml"
        in readme
    )
    assert (
        ".github/"
        in readme
    )
