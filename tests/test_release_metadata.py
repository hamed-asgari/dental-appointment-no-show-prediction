from __future__ import annotations
from pathlib import Path
import tomllib
import yaml
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
        "license"
    ] == "MIT"
    assert project[
        "license-files"
    ] == [
        "LICENSE",
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
    assert "setuptools>=77" in build[
        "requires"
    ]
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
        "data/raw/README.md text eol=crlf",
        "*.toml text eol=lf",
        "*.yml text eol=lf",
        "*.txt text eol=lf",
        "*.csv text eol=lf",
        "data/raw/*.csv text eol=crlf",
        "*.parquet binary",
        "LICENSE text eol=lf",
    )
    for value in required:
        assert value in attributes
    assert attributes.index(
        "*.md text eol=lf"
    ) < attributes.index(
        "data/raw/README.md text eol=crlf"
    )
    assert attributes.index(
        "*.csv text eol=lf"
    ) < attributes.index(
        "data/raw/*.csv text eol=crlf"
    )
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

def test_citation_metadata_matches_release() -> None:
    citation_path = (
        _ROOT
        / "CITATION.cff"
    )
    citation = yaml.safe_load(
        citation_path.read_text(
            encoding="utf-8"
        )
    )
    with (
        _ROOT
        / "pyproject.toml"
    ).open(
        "rb"
    ) as stream:
        project = tomllib.load(
            stream
        )[
            "project"
        ]
    assert citation[
        "cff-version"
    ] == "1.2.0"
    assert citation[
        "title"
    ] == "Dental Appointment No-show Prediction"
    assert citation[
        "type"
    ] == "software"
    assert citation[
        "authors"
    ] == [
        {
            "family-names": "Asgari",
            "given-names": "Hamed",
        }
    ]
    assert citation[
        "version"
    ] == project[
        "version"
    ]
    assert citation[
        "repository-code"
    ] == project[
        "urls"
    ][
        "Repository"
    ]
    assert citation[
        "url"
    ] == project[
        "urls"
    ][
        "Repository"
    ]
    assert citation["license"] == "MIT"
    assert "doi" not in citation
    assert "date-released" not in citation
def test_changelog_records_release_boundaries() -> None:
    changelog = (
        _ROOT
        / "CHANGELOG.md"
    ).read_text(
        encoding="utf-8"
    )
    normalized = " ".join(
        changelog.split()
    )
    required = (
        "## [1.0.0]",
        "calibration_prior",
        "0.11985448975684472",
        "0.12412028150991683",
        "ROC AUC of `0.5`",
        "no appointment-level ranking",
        "no longer an untouched benchmark",
        "No software license",
    )
    for value in required:
        assert value in normalized
def test_readme_links_release_metadata() -> None:
    readme = (
        _ROOT
        / "README.md"
    ).read_text(
        encoding="utf-8"
    )
    required = (
        "[CHANGELOG.md](CHANGELOG.md)",
        "[CITATION.cff](CITATION.cff)",
        "[MIT License](LICENSE)",
    )
    for value in required:
        assert value in readme
    assert (
        _ROOT
        / "CHANGELOG.md"
    ).is_file()
    assert (
        _ROOT
        / "CITATION.cff"
    ).is_file()

def test_mit_license_is_declared_consistently() -> None:
    license_path = _ROOT / "LICENSE"
    assert license_path.is_file()
    license_text = license_path.read_text(
        encoding="utf-8"
    )
    assert license_text.startswith(
        "MIT License\n\n"
        "Copyright (c) 2026 Hamed Asgari\n"
    )
    assert license_text.endswith(
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )
