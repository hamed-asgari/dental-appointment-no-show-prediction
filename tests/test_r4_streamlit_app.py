from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DATA_LAYER = ROOT / "app" / "dashboard_data.py"
APP_README = ROOT / "app" / "README.md"


def test_streamlit_app_runs_against_committed_frozen_evidence() -> None:
    app = AppTest.from_file(
        str(APP),
        default_timeout=20,
    ).run()
    assert not app.exception
    assert (
        app.title[0].value
        == "Dental Appointment No-show Prediction"
    )
    assert any(
        "Synthetic-data-only" in warning.value
        for warning in app.warning
    )


def test_app_source_has_no_protected_access_or_runtime_scoring() -> None:
    source = (
        APP.read_text(encoding="utf-8")
        + "\n"
        + DATA_LAYER.read_text(encoding="utf-8")
    )
    prohibited = (
        "load_verified_v2_final_test_targets",
        "allow_test=True",
        "predict_proba(",
        "joblib.load(",
        "st.slider(",
        "st.number_input(",
        "st.text_input(",
        "st.selectbox(",
        "st.multiselect(",
    )
    for needle in prohibited:
        assert needle not in source
    assert "use_container_width" not in source


def test_app_readme_has_frozen_launch_command_and_boundaries() -> None:
    text = APP_README.read_text(encoding="utf-8")
    assert (
        r".\.venv\Scripts\python.exe -m streamlit "
        r"run app/streamlit_app.py"
        in text
    )
    assert "fully synthetic" in text
    assert (
        "not validated for clinical or operational use"
        in text
    )
    assert "The app does not:" in text
