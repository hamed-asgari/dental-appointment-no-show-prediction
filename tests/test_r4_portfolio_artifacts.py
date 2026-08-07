from __future__ import annotations

import hashlib
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]

SCREENSHOTS = {
    "v2_streamlit_overview.png": (
        1837,
        926,
        "809f42bcfacd919248663f515595bbb1810dcdc35a92e0f186e74446320d8e79",
    ),
    "v2_streamlit_performance.png": (
        1861,
        519,
        "0e7f45905b1cf2940215ebf56b53351507f8ea84722ab5df13323d9acbc48c64",
    ),
    "v2_streamlit_calibration_capacity.png": (
        1222,
        878,
        "e6533a9432744a458327e1bfc7ee4a02b425040501bd42e2c6488a98b85a588a",
    ),
    "v2_streamlit_interpretation_limitations.png": (
        1920,
        1080,
        "572772460187355a2cd0508284c2b58c87494bf069b52fb5256dfb63fd3913be",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r4_portfolio_screenshot_identities_are_frozen() -> None:
    screenshot_dir = ROOT / "reports" / "screenshots"
    for name, (expected_width, expected_height, expected_hash) in SCREENSHOTS.items():
        path = screenshot_dir / name
        raw = path.read_bytes()
        assert raw[:8] == bytes.fromhex("89504e470d0a1a0a")
        width, height = struct.unpack(">II", raw[16:24])
        assert (width, height) == (expected_width, expected_height)
        assert _sha256(path) == expected_hash


def test_root_readme_describes_current_v2_application_state() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "transparent_model_evaluation_dashboard" in text
    assert "v2_streamlit_overview.png" in text
    assert "v2_streamlit_performance.png" in text
    assert "Version `2.0.0` is still under recovery review" in text
    assert "no implemented Streamlit application" not in text
    assert "protected 2027 final-test targets have not been accessed" not in text


def test_app_and_model_readmes_state_the_r4_boundary() -> None:
    app_text = (ROOT / "app" / "README.md").read_text(encoding="utf-8")
    model_text = (ROOT / "models" / "README.md").read_text(encoding="utf-8")

    assert "does not load this model for appointment scoring" in model_text
    assert "frozen_logistic_pipeline.joblib" in model_text
    assert "does not:" in app_text
    assert "score new appointments" in app_text
    assert "v2_r4_portfolio_architecture.md" in app_text


def test_portfolio_architecture_documents_no_runtime_scoring_path() -> None:
    text = (
        ROOT / "docs" / "v2_r4_portfolio_architecture.md"
    ).read_text(encoding="utf-8")

    assert "```mermaid" in text
    assert "One-time protected-target access" in text
    assert "not loaded by R4 app" in text
    assert "no path to app" in text
    assert "transparent_model_evaluation_dashboard" in text


def test_screenshot_readme_lists_every_frozen_identity() -> None:
    text = (
        ROOT / "reports" / "screenshots" / "README.md"
    ).read_text(encoding="utf-8")

    for name, (_, _, digest) in SCREENSHOTS.items():
        assert name in text
        assert digest in text
