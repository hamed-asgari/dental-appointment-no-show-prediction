from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from app.dashboard_data import (
    APP_TYPE,
    DashboardIntegrityError,
    FIGURE_RELATIVES,
    MANIFEST_RELATIVE,
    SUMMARY_RELATIVE,
    load_dashboard_data,
    repository_root,
    validate_dashboard_payload,
)


def _copy_dashboard_inputs(destination: Path) -> None:
    root = repository_root()
    for relative in (
        SUMMARY_RELATIVE,
        MANIFEST_RELATIVE,
        *FIGURE_RELATIVES,
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)


def test_committed_dashboard_inputs_load() -> None:
    data = load_dashboard_data()
    assert (
        data.summary["app_decision"]["selected_app_type"]
        == APP_TYPE
    )
    assert data.manifest["selected_app_type"] == APP_TYPE
    assert set(data.figure_paths) == {
        path.name
        for path in FIGURE_RELATIVES
    }


def test_missing_dashboard_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    _copy_dashboard_inputs(tmp_path)
    (tmp_path / FIGURE_RELATIVES[0]).unlink()
    with pytest.raises(DashboardIntegrityError, match="missing"):
        load_dashboard_data(tmp_path)


def test_altered_dashboard_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    _copy_dashboard_inputs(tmp_path)
    summary = tmp_path / SUMMARY_RELATIVE
    summary.write_bytes(summary.read_bytes() + b"\n")
    with pytest.raises(
        DashboardIntegrityError,
        match="SHA-256 mismatch",
    ):
        load_dashboard_data(tmp_path)


def test_wrong_app_type_is_rejected() -> None:
    data = load_dashboard_data()
    summary = deepcopy(data.summary)
    summary["app_decision"][
        "selected_app_type"
    ] = "appointment_level_risk_demo"
    with pytest.raises(
        DashboardIntegrityError,
        match="app decision",
    ):
        validate_dashboard_payload(
            summary,
            data.manifest,
        )


@pytest.mark.parametrize(
    "flag",
    [
        "target_reaccess_performed",
        "model_refit_performed",
        "calibration_change_performed",
        "final_test_threshold_selected",
        "post_test_model_tuning_permitted",
    ],
)
def test_prohibited_manifest_state_is_rejected(
    flag: str,
) -> None:
    data = load_dashboard_data()
    manifest = deepcopy(data.manifest)
    manifest[flag] = True
    with pytest.raises(
        DashboardIntegrityError,
        match="Prohibited",
    ):
        validate_dashboard_payload(
            data.summary,
            manifest,
        )


def test_summary_must_remain_synthetic_scope() -> None:
    data = load_dashboard_data()
    summary = deepcopy(data.summary)
    summary["reporting_boundary"][
        "claims_scope"
    ] = "clinical"
    with pytest.raises(
        DashboardIntegrityError,
        match="synthetic_data_only",
    ):
        validate_dashboard_payload(
            summary,
            data.manifest,
        )
