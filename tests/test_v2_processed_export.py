from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.build_v2_dataset import (
    build_verified_v2_feature_dataset,
    calculate_sha256,
    select_v2_model_features,
)
import src.data.export_v2_processed as processed_export
from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    PROCESSED_DATASET_FILENAME,
    PROCESSED_MANIFEST_FILENAME,
    export_v2_processed_feature_dataset,
    load_frozen_v2_processed_feature_dataset,
    main,
    verify_exported_v2_processed_dataset,
)
from src.data.v2_processed_hashes import (
    FROZEN_V2_PROCESSED_DATASET_FINGERPRINT,
    FROZEN_V2_PROCESSED_DATASET_SHA256,
    FROZEN_V2_PROCESSED_MANIFEST_SHA256,
)
from src.features.schema import (
    V2_FEATURE_DATASET_COLUMNS,
    V2_MODEL_FEATURE_COLUMNS,
    V2_PROHIBITED_MODEL_COLUMNS,
)
from src.synthetic.frozen_hashes import (
    FROZEN_V2_DATASET_FINGERPRINT,
    FROZEN_V2_MANIFEST_SHA256,
    FROZEN_V2_RAW_HASHES,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw" / "v2"
_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_synthetic_benchmark.json"


@pytest.fixture(scope="module")
def frozen_manifest() -> dict[str, object]:
    return verify_exported_v2_processed_dataset(DEFAULT_V2_PROCESSED_DIR)


@pytest.fixture(scope="module")
def frozen_dataset() -> pd.DataFrame:
    return load_frozen_v2_processed_feature_dataset(DEFAULT_V2_PROCESSED_DIR)


def _copy_export(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        PROCESSED_DATASET_FILENAME,
        PROCESSED_MANIFEST_FILENAME,
    ):
        source = DEFAULT_V2_PROCESSED_DIR / filename
        (destination / filename).write_bytes(source.read_bytes())


def test_frozen_processed_hashes_match_committed_files() -> None:
    dataset_path = DEFAULT_V2_PROCESSED_DIR / PROCESSED_DATASET_FILENAME
    manifest_path = DEFAULT_V2_PROCESSED_DIR / PROCESSED_MANIFEST_FILENAME
    assert calculate_sha256(dataset_path) == FROZEN_V2_PROCESSED_DATASET_SHA256
    assert calculate_sha256(manifest_path) == FROZEN_V2_PROCESSED_MANIFEST_SHA256


def test_manifest_declares_target_free_protected_state(
    frozen_manifest: dict[str, object],
) -> None:
    assert frozen_manifest["target_included"] is False
    assert frozen_manifest["final_test_target_accessed"] is False
    assert (
        frozen_manifest["processed_dataset_fingerprint"]
        == FROZEN_V2_PROCESSED_DATASET_FINGERPRINT
    )


def test_manifest_records_frozen_raw_identity(
    frozen_manifest: dict[str, object],
) -> None:
    source = frozen_manifest["source_identity"]
    assert isinstance(source, dict)
    assert source["raw_manifest_sha256"] == FROZEN_V2_MANIFEST_SHA256
    assert source["raw_dataset_fingerprint"] == FROZEN_V2_DATASET_FINGERPRINT
    assert source["raw_file_sha256"] == dict(FROZEN_V2_RAW_HASHES)


def test_manifest_schema_and_partition_counts_are_exact(
    frozen_manifest: dict[str, object],
) -> None:
    artifact = frozen_manifest["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["row_count"] == 21_755
    assert artifact["column_count"] == 38
    assert artifact["columns"] == list(V2_FEATURE_DATASET_COLUMNS)
    assert artifact["model_feature_count"] == 32
    assert artifact["model_feature_columns"] == list(V2_MODEL_FEATURE_COLUMNS)
    assert artifact["partition_counts"] == {
        "calibration": 1_081,
        "context_only": 10,
        "development_fit": 4_467,
        "final_test": 4_343,
        "fold_1_validation": 2_150,
        "fold_2_validation": 2_231,
        "fold_3_validation": 2_086,
        "policy_selection": 1_063,
        "warmup": 4_324,
    }


def test_frozen_processed_dataset_shape_columns_and_target_absence(
    frozen_dataset: pd.DataFrame,
) -> None:
    assert frozen_dataset.shape == (21_755, 38)
    assert list(frozen_dataset.columns) == list(V2_FEATURE_DATASET_COLUMNS)
    assert "target" not in frozen_dataset.columns
    assert not frozen_dataset.isna().any().any()
    assert len(
        frozen_dataset.loc[
            frozen_dataset["evaluation_partition"].eq("final_test")
        ]
    ) == 4_343


def test_frozen_processed_dataset_matches_rebuilt_features(
    frozen_dataset: pd.DataFrame,
) -> None:
    rebuilt = build_verified_v2_feature_dataset()
    pd.testing.assert_frame_equal(
        frozen_dataset,
        rebuilt,
        check_exact=True,
    )


def test_frozen_model_feature_selector_remains_exact(
    frozen_dataset: pd.DataFrame,
) -> None:
    selected = select_v2_model_features(frozen_dataset)
    assert selected.shape == (21_755, 32)
    assert list(selected.columns) == list(V2_MODEL_FEATURE_COLUMNS)
    assert set(selected.columns).isdisjoint(V2_PROHIBITED_MODEL_COLUMNS)


def test_processed_csv_header_is_exact() -> None:
    path = DEFAULT_V2_PROCESSED_DIR / PROCESSED_DATASET_FILENAME
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    assert header == list(V2_FEATURE_DATASET_COLUMNS)
    assert "target" not in header


def test_export_is_deterministic_across_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frozen_dataset: pd.DataFrame,
) -> None:
    tiny = frozen_dataset.iloc[:3].copy(deep=True)
    monkeypatch.setattr(
        processed_export,
        "build_verified_v2_feature_dataset",
        lambda **_: tiny.copy(deep=True),
    )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = export_v2_processed_feature_dataset(
        output_dir=first_dir,
        overwrite=True,
    )
    second = export_v2_processed_feature_dataset(
        output_dir=second_dir,
        overwrite=True,
    )
    assert first == second
    for filename in (
        PROCESSED_DATASET_FILENAME,
        PROCESSED_MANIFEST_FILENAME,
    ):
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()


def test_existing_complete_export_is_verified_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "complete"
    _copy_export(destination)
    observed = export_v2_processed_feature_dataset(
        output_dir=destination,
        overwrite=False,
    )
    expected = verify_exported_v2_processed_dataset(destination)
    assert observed == expected


def test_incomplete_existing_export_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "incomplete"
    destination.mkdir()
    (destination / PROCESSED_DATASET_FILENAME).write_bytes(b"header\n")
    with pytest.raises(ValueError, match="incomplete export"):
        export_v2_processed_feature_dataset(
            output_dir=destination,
            overwrite=False,
        )


def test_tampered_processed_csv_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "tampered-csv"
    _copy_export(destination)
    dataset_path = destination / PROCESSED_DATASET_FILENAME
    dataset_path.write_bytes(dataset_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_exported_v2_processed_dataset(destination)


def test_tampered_processed_manifest_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "tampered-manifest"
    _copy_export(destination)
    manifest_path = destination / PROCESSED_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_included"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_included=false"):
        verify_exported_v2_processed_dataset(destination)


def test_frozen_loader_rejects_self_consistent_nonfrozen_copy(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nonfrozen"
    _copy_export(destination)
    dataset_path = destination / PROCESSED_DATASET_FILENAME
    dataset_path.write_bytes(dataset_path.read_bytes().replace(b"true", b"false", 1))
    manifest_path = destination / PROCESSED_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["sha256"] = calculate_sha256(dataset_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint|Frozen processed"):
        load_frozen_v2_processed_feature_dataset(destination)


def test_verify_only_cli_reports_fingerprint(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "--output-dir",
            str(DEFAULT_V2_PROCESSED_DIR),
            "--verify-only",
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert FROZEN_V2_PROCESSED_DATASET_FINGERPRINT in captured.out
