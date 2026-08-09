from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import unquote
from urllib.request import urlopen


SAFE_PROJECT_MODULE_COMMANDS = (
    "src.synthetic.export",
    "src.data.export_v2_processed",
    "src.modeling.v2_final_reporting",
)

FORBIDDEN_PROJECT_MODULES = (
    "src.modeling.v2_final_test_evaluation",
    "src.modeling.v2_final_test_probabilities",
    "src.modeling.v2_persistence",
    "src.modeling.v2_development",
    "src.modeling.v2_calibration",
)

SCREENSHOT_SHA256 = {
    "reports/screenshots/v2_streamlit_overview.png":
        "809f42bcfacd919248663f515595bbb1810dcdc35a92e0f186e74446320d8e79",
    "reports/screenshots/v2_streamlit_performance.png":
        "0e7f45905b1cf2940215ebf56b53351507f8ea84722ab5df13323d9acbc48c64",
    "reports/screenshots/v2_streamlit_calibration_capacity.png":
        "e6533a9432744a458327e1bfc7ee4a02b425040501bd42e2c6488a98b85a588a",
    "reports/screenshots/v2_streamlit_interpretation_limitations.png":
        "572772460187355a2cd0508284c2b58c87494bf069b52fb5256dfb63fd3913be",
}

FIGURE_SHA256 = {
    "reports/figures/v2_final_precision_recall_curve.png":
        "e706d92fa13e09c5fda7da2fcd0f6a8c55332d524ae0e221d370ef792ebd7b95",
    "reports/figures/v2_final_calibration_curve.png":
        "eb98404d809b0ab691decb2cb10a3d0c3b2a495cbef8637d75a468e86f25834f",
    "reports/figures/v2_final_capacity_tradeoff.png":
        "965ee9e77e35b05ae380d017b88a7fa952fd6b82bf608cb8fc71b66cf01827a1",
}

EVIDENCE_SCHEMA_VERSION = "1.0.0"


def _run(
    *args: str,
    cwd: Path,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def _output(*args: str, cwd: Path) -> str:
    return _run(*args, cwd=cwd, capture=True).stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _python_version(python: Path, cwd: Path) -> str:
    return _output(
        str(python),
        "-c",
        "import platform; print(platform.python_version())",
        cwd=cwd,
    )


def _base_python() -> Path:
    candidate = getattr(sys, "_base_executable", None)
    if candidate:
        return Path(candidate).resolve()
    return Path(sys.executable).resolve()


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _relative_markdown_links(repo: Path) -> tuple[int, list[str]]:
    link_re = re.compile(r"!?\[[^\]]*\]\((.*?)\)", re.S)
    checked = 0
    broken: list[str] = []

    for path in sorted(repo.rglob("*.md")):
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in link_re.finditer(text):
            destination = " ".join(match.group(1).split()).strip()
            if not destination or destination.startswith("#"):
                continue
            if destination.startswith(("http://", "https://", "mailto:")):
                continue
            if destination.startswith("<") and destination.endswith(">"):
                destination = destination[1:-1]
            destination = unquote(destination.split("#", 1)[0])
            if not destination:
                continue

            checked += 1
            target = (path.parent / destination).resolve()
            try:
                target.relative_to(repo.resolve())
            except ValueError:
                broken.append(
                    f"{path.relative_to(repo).as_posix()}: escapes repo -> "
                    f"{destination}"
                )
                continue
            if not target.exists():
                broken.append(
                    f"{path.relative_to(repo).as_posix()}: missing -> "
                    f"{destination}"
                )

    return checked, broken


def _tracked_tree_hygiene(repo: Path) -> list[str]:
    tracked = _output("git", "ls-files", cwd=repo).splitlines()
    forbidden: list[str] = []

    for raw in tracked:
        path = raw.replace("\\", "/")
        lowered = path.lower()
        parts = lowered.split("/")
        if (
            lowered.startswith(".idea/")
            or lowered.startswith(".venv/")
            or lowered.startswith(".pytest_cache/")
            or lowered.startswith("build/")
            or "__pycache__" in parts
            or any(part.endswith(".egg-info") for part in parts)
            or lowered.endswith(".pyc")
        ):
            forbidden.append(path)

    return forbidden


def _verify_hash_map(repo: Path, expected: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"Missing frozen artifact: {relative}")
        digest = _sha256(path)
        if digest != expected_hash:
            raise RuntimeError(
                f"Frozen artifact SHA-256 mismatch: {relative}: {digest}"
            )
        actual[relative] = digest
    return actual


def _module_constants(
    clean_python: Path,
    checkout: Path,
    module: str,
    names: tuple[str, ...],
) -> dict[str, Any]:
    code = (
        "import json; "
        f"import {module} as m; "
        "print(json.dumps({"
        + ",".join(
            repr(name) + ": str(getattr(m, " + repr(name) + "))"
            for name in names
        )
        + "}))"
    )
    return json.loads(
        _output(str(clean_python), "-c", code, cwd=checkout)
    )


def _compare_directory_files(
    generated: Path,
    canonical: Path,
) -> dict[str, str]:
    generated_files = sorted(
        path for path in generated.iterdir() if path.is_file()
    )
    if not generated_files:
        raise RuntimeError(f"No generated files found in {generated}")

    identities: dict[str, str] = {}
    for path in generated_files:
        counterpart = canonical / path.name
        if not counterpart.is_file():
            raise RuntimeError(
                f"Canonical counterpart missing for generated file: {path.name}"
            )
        generated_hash = _sha256(path)
        canonical_hash = _sha256(counterpart)
        if generated_hash != canonical_hash:
            raise RuntimeError(
                f"Reproduction mismatch for {path.name}: "
                f"{generated_hash} != {canonical_hash}"
            )
        identities[path.name] = generated_hash

    return identities


def _streamlit_smoke(
    clean_python: Path,
    checkout: Path,
    log_path: Path,
) -> dict[str, Any]:
    port = _find_free_local_port()
    command = [
        str(clean_python),
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=checkout,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.monotonic() + 90.0
        health = None
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Streamlit exited before health check succeeded"
                    )
                try:
                    with urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health",
                        timeout=2.0,
                    ) as response:
                        health = response.read().decode("utf-8").strip()
                    if health == "ok":
                        break
                except Exception:
                    time.sleep(1.0)

            if health != "ok":
                raise RuntimeError(
                    "Streamlit health endpoint did not return 'ok'"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    return {
        "port": port,
        "health": health,
        "log_sha256": _sha256(log_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run R5.1 clean-environment reproduction without protected-target "
            "re-access or model refit."
        )
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Exact committed source SHA to reproduce.",
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        required=True,
        help="Output JSON path outside the disposable checkout.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the disposable workspace for debugging.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    canonical = Path(__file__).resolve().parents[1]
    source_commit = args.source_commit.strip()

    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise SystemExit("--source-commit must be a full 40-character SHA")

    current_head = _output("git", "rev-parse", "HEAD", cwd=canonical)
    if current_head != source_commit:
        raise SystemExit(
            "Canonical checkout HEAD does not match --source-commit"
        )

    if _output("git", "status", "--porcelain", cwd=canonical):
        raise SystemExit(
            "Canonical repository must be clean before reproduction"
        )

    base_python = _base_python()
    base_version = _python_version(base_python, canonical)
    if not base_version.startswith("3.12."):
        raise SystemExit(
            f"R5.1 requires base Python 3.12, found {base_version}"
        )

    workspace = Path(tempfile.mkdtemp(prefix="dental-r5-clean-")).resolve()
    checkout = workspace / "checkout"
    clean_venv = workspace / "venv"
    clean_python = clean_venv / "Scripts" / "python.exe"
    pytest_root = workspace / "pytest"
    raw_out = workspace / "raw-reproduction"
    processed_out = workspace / "processed-reproduction"
    reporting_root = workspace / "reporting-reproduction"
    reporting_figures = reporting_root / "figures"
    reporting_summary = reporting_root / "final_reporting_summary.json"
    reporting_manifest = reporting_root / "final_reporting_manifest.json"
    streamlit_log = workspace / "streamlit.log"

    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_commit": source_commit,
        "canonical_repo": str(canonical),
        "workspace": str(workspace),
        "safe_project_module_commands": list(SAFE_PROJECT_MODULE_COMMANDS),
        "forbidden_project_modules_invoked": False,
        "protected_target_reaccess_performed": False,
        "model_refit_performed": False,
        "calibration_change_performed": False,
        "threshold_selection_performed": False,
        "post_test_tuning_performed": False,
    }

    worktree_added = False
    try:
        _run(
            "git",
            "worktree",
            "add",
            "--detach",
            str(checkout),
            source_commit,
            cwd=canonical,
        )
        worktree_added = True

        if _output("git", "rev-parse", "HEAD", cwd=checkout) != source_commit:
            raise RuntimeError("Disposable checkout SHA mismatch")
        if _output("git", "status", "--porcelain", cwd=checkout):
            raise RuntimeError("Disposable checkout is not initially clean")

        _run(
            str(base_python),
            "-m",
            "venv",
            str(clean_venv),
            cwd=checkout,
        )
        clean_version = _python_version(clean_python, checkout)
        if not clean_version.startswith("3.12."):
            raise RuntimeError(
                f"Clean environment is not Python 3.12: {clean_version}"
            )
        evidence["python"] = {
            "base_version": base_version,
            "clean_version": clean_version,
        }

        _run(
            str(clean_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            cwd=checkout,
        )
        _run(
            str(clean_python),
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.lock.txt",
            cwd=checkout,
        )

        pip_check = _run(
            str(clean_python),
            "-m",
            "pip",
            "check",
            cwd=checkout,
            capture=True,
        )
        evidence["dependency_integrity"] = {
            "pip_check_exit": pip_check.returncode,
            "pip_check_stdout": pip_check.stdout.strip(),
        }

        _run(
            str(clean_python),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(pytest_root),
            cwd=checkout,
        )
        evidence["full_test_suite_passed"] = True

        _run(
            str(clean_python),
            "-m",
            SAFE_PROJECT_MODULE_COMMANDS[0],
            "--output-dir",
            str(raw_out),
            cwd=checkout,
        )

        synthetic_constants = _module_constants(
            clean_python,
            checkout,
            "src.synthetic.export",
            ("DEFAULT_CONFIG_PATH", "MANIFEST_FILENAME"),
        )
        config_path = Path(synthetic_constants["DEFAULT_CONFIG_PATH"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        canonical_raw = checkout / config["output_directory"]
        raw_identities = _compare_directory_files(raw_out, canonical_raw)
        evidence["raw_reproduction"] = {
            "canonical_dir": str(canonical_raw.relative_to(checkout)),
            "files": raw_identities,
        }

        _run(
            str(clean_python),
            "-m",
            SAFE_PROJECT_MODULE_COMMANDS[1],
            "--output-dir",
            str(processed_out),
            cwd=checkout,
        )
        processed_constants = _module_constants(
            clean_python,
            checkout,
            "src.data.export_v2_processed",
            ("DEFAULT_V2_PROCESSED_DIR",),
        )
        canonical_processed = Path(
            processed_constants["DEFAULT_V2_PROCESSED_DIR"]
        )
        processed_identities = _compare_directory_files(
            processed_out,
            canonical_processed,
        )
        evidence["processed_reproduction"] = {
            "canonical_dir": str(canonical_processed.relative_to(checkout)),
            "files": processed_identities,
        }

        reporting_figures.mkdir(parents=True, exist_ok=True)
        _run(
            str(clean_python),
            "-m",
            SAFE_PROJECT_MODULE_COMMANDS[2],
            "--figure-dir",
            str(reporting_figures),
            "--summary-path",
            str(reporting_summary),
            "--manifest-path",
            str(reporting_manifest),
            cwd=checkout,
        )

        canonical_summary = (
            checkout
            / "reports"
            / "modeling"
            / "v2"
            / "final_reporting"
            / "final_reporting_summary.json"
        )
        canonical_manifest = (
            checkout
            / "reports"
            / "modeling"
            / "v2"
            / "final_reporting"
            / "final_reporting_manifest.json"
        )

        if _sha256(reporting_summary) != _sha256(canonical_summary):
            raise RuntimeError("Final reporting summary reproduction mismatch")
        if _sha256(reporting_manifest) != _sha256(canonical_manifest):
            raise RuntimeError("Final reporting manifest reproduction mismatch")

        reporting_figures_identity = _compare_directory_files(
            reporting_figures,
            checkout / "reports" / "figures",
        )
        evidence["reporting_reproduction"] = {
            "summary_sha256": _sha256(reporting_summary),
            "manifest_sha256": _sha256(reporting_manifest),
            "figures": reporting_figures_identity,
        }

        evidence["frozen_screenshots"] = _verify_hash_map(
            checkout,
            SCREENSHOT_SHA256,
        )
        evidence["frozen_analytical_figures"] = _verify_hash_map(
            checkout,
            FIGURE_SHA256,
        )

        checked_links, broken_links = _relative_markdown_links(checkout)
        if broken_links:
            raise RuntimeError(
                "Broken relative Markdown links: " + repr(broken_links)
            )
        evidence["markdown_links"] = {
            "checked": checked_links,
            "broken": 0,
        }

        hygiene = _tracked_tree_hygiene(checkout)
        if hygiene:
            raise RuntimeError(
                "Forbidden local-only artifacts are tracked: "
                + repr(hygiene)
            )
        evidence["tracked_tree_hygiene"] = {
            "forbidden_paths": [],
            "passed": True,
        }

        evidence["streamlit"] = _streamlit_smoke(
            clean_python,
            checkout,
            streamlit_log,
        )

        disposable_status = _output(
            "git",
            "status",
            "--porcelain",
            cwd=checkout,
        )
        if disposable_status:
            raise RuntimeError(
                "Disposable checkout became dirty: "
                + disposable_status
            )
        evidence["disposable_checkout_clean_after_run"] = True

        canonical_status = _output(
            "git",
            "status",
            "--porcelain",
            cwd=canonical,
        )
        if canonical_status:
            raise RuntimeError(
                "Canonical repository became dirty during reproduction"
            )
        evidence["canonical_checkout_clean_after_run"] = True
        evidence["success"] = True

        evidence_path = args.evidence_json.resolve()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        print("R5.1 clean-environment reproduction passed.")
        print(f"source_commit = {source_commit}")
        print(f"clean_python = {clean_version}")
        print("pip_check = pass")
        print("full_test_suite = pass")
        print("raw_reproduction = byte-identical")
        print("processed_reproduction = byte-identical")
        print("final_reporting_reproduction = byte-identical")
        print("markdown_broken_links = 0")
        print("tracked_tree_hygiene = pass")
        print("streamlit_health = ok")
        print("protected_target_reaccess_performed = false")
        print("model_refit_performed = false")
        print(f"evidence_json = {evidence_path}")
        return 0

    finally:
        if worktree_added:
            _run(
                "git",
                "worktree",
                "remove",
                "--force",
                str(checkout),
                cwd=canonical,
                check=False,
            )
            _run(
                "git",
                "worktree",
                "prune",
                cwd=canonical,
                check=False,
            )
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
