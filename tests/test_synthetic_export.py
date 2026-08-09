"""Contract tests for deterministic Version 2 benchmark export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.synthetic.config import (
    DEFAULT_CONFIG_PATH,
    calculate_config_sha256,
)
from src.synthetic.export import (
    MANIFEST_FILENAME,
    TABLE_FILENAMES,
    calculate_sha256,
    export_synthetic_benchmark,
    verify_exported_benchmark,
)
from src.synthetic.frozen_hashes import (
    FROZEN_V2_DATASET_FINGERPRINT,
    FROZEN_V2_MANIFEST_SHA256,
    FROZEN_V2_RAW_HASHES,
)
from src.synthetic.schema import (
    APPOINTMENT_COLUMNS,
    DENTIST_COLUMNS,
    FORBIDDEN_EXPORTED_COLUMNS,
    PATIENT_COLUMNS,
)


_ROOT = Path(__file__).resolve().parents[1]
_RAW_V2 = _ROOT / "data" / "raw" / "v2"


def test_frozen_export_files_exist_and_match_hashes() -> None:
    expected_names = {
        *TABLE_FILENAMES.values(),
        MANIFEST_FILENAME,
        "README.md",
    }
    actual_names = {
        path.name
        for path in _RAW_V2.iterdir()
        if path.is_file()
    }
    assert actual_names == expected_names

    for filename, expected_hash in FROZEN_V2_RAW_HASHES.items():
        assert calculate_sha256(_RAW_V2 / filename) == expected_hash

    assert (
        calculate_sha256(_RAW_V2 / MANIFEST_FILENAME)
        == FROZEN_V2_MANIFEST_SHA256
    )


def test_manifest_matches_frozen_contract() -> None:
    manifest = verify_exported_benchmark(_RAW_V2)

    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["dataset_name"] == "v2_synthetic_benchmark"
    assert manifest["synthetic_data"] is True
    assert manifest["validated_for_clinical_use"] is False
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["generator_version"] == "2.0.0"
    assert manifest["root_seed"] == 20260805
    assert (
        manifest["configuration"]["sha256"]
        == calculate_config_sha256(DEFAULT_CONFIG_PATH)
    )
    assert (
        manifest["dataset_fingerprint"]
        == FROZEN_V2_DATASET_FINGERPRINT
    )


def test_manifest_table_rows_columns_and_counts_are_frozen() -> None:
    manifest = verify_exported_benchmark(_RAW_V2)
    patients = manifest["tables"]["patients"]
    dentists = manifest["tables"]["dentists"]
    appointments = manifest["tables"]["appointments"]

    assert patients["row_count"] == 4000
    assert patients["columns"] == list(PATIENT_COLUMNS)
    assert dentists["row_count"] == 7
    assert dentists["columns"] == list(DENTIST_COLUMNS)
    assert appointments["row_count"] == 24000
    assert appointments["columns"] == list(APPOINTMENT_COLUMNS)
    assert appointments["status_counts"] == {
        "cancelled": 1818,
        "completed": 19185,
        "no_show": 1943,
        "rescheduled": 1054,
    }
    assert appointments["reminder_sent_counts"] == {
        "false": 3237,
        "true": 20763,
    }


def test_exported_csv_headers_match_public_schemas() -> None:
    expected = {
        "patients": PATIENT_COLUMNS,
        "dentists": DENTIST_COLUMNS,
        "appointments": APPOINTMENT_COLUMNS,
    }
    for table_name, filename in TABLE_FILENAMES.items():
        headers = tuple(
            pd.read_csv(
                _RAW_V2 / filename,
                nrows=0,
            ).columns
        )
        assert headers == expected[table_name]


def test_exported_csvs_contain_no_hidden_columns() -> None:
    all_headers: set[str] = set()
    for filename in TABLE_FILENAMES.values():
        all_headers.update(
            pd.read_csv(
                _RAW_V2 / filename,
                nrows=0,
            ).columns
        )
    assert all_headers.isdisjoint(FORBIDDEN_EXPORTED_COLUMNS)


def test_exported_appointment_rows_and_statuses_round_trip() -> None:
    appointments = pd.read_csv(
        _RAW_V2 / TABLE_FILENAMES["appointments"],
        dtype={
            "appointment_id": "int64",
            "patient_id": "int64",
            "dentist_id": "int64",
            "status": "string",
            "reminder_sent": "boolean",
        },
    )
    assert len(appointments) == 24000
    assert appointments["appointment_id"].is_unique
    assert appointments["status"].value_counts().sort_index().to_dict() == {
        "cancelled": 1818,
        "completed": 19185,
        "no_show": 1943,
        "rescheduled": 1054,
    }


def test_temporary_export_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = export_synthetic_benchmark(
        output_dir=first,
        overwrite=True,
    )
    second_manifest = export_synthetic_benchmark(
        output_dir=second,
        overwrite=True,
    )

    assert (
        first_manifest["dataset_fingerprint"]
        == second_manifest["dataset_fingerprint"]
        == FROZEN_V2_DATASET_FINGERPRINT
    )
    for filename in (
        *TABLE_FILENAMES.values(),
        MANIFEST_FILENAME,
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_complete_existing_export_is_verified_without_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmark"
    created = export_synthetic_benchmark(
        output_dir=output,
        overwrite=True,
    )
    verified = export_synthetic_benchmark(
        output_dir=output,
        overwrite=False,
    )
    assert verified == created


def test_incomplete_existing_export_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    output.mkdir()
    (output / TABLE_FILENAMES["patients"]).write_text(
        "patient_id\n1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="incomplete existing export",
    ):
        export_synthetic_benchmark(
            output_dir=output,
            overwrite=False,
        )


def test_tampered_csv_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    export_synthetic_benchmark(
        output_dir=output,
        overwrite=True,
    )
    appointment_path = output / TABLE_FILENAMES["appointments"]
    appointment_path.write_bytes(
        appointment_path.read_bytes() + b"\n"
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch for appointments.csv",
    ):
        verify_exported_benchmark(output)


def test_export_leaves_no_temporary_files(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    export_synthetic_benchmark(
        output_dir=output,
        overwrite=True,
    )
    assert not list(output.glob("*.tmp"))
    assert not list(output.glob(".*.tmp"))


def test_verify_only_cli_accepts_frozen_export() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.synthetic.export",
            "--verify-only",
        ],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Verified Version 2 synthetic benchmark" in completed.stdout


def test_manifest_json_is_canonical_and_newline_terminated() -> None:
    path = _RAW_V2 / MANIFEST_FILENAME
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    decoded = json.loads(text)
    assert (
        json.dumps(
            decoded,
            indent=2,
            sort_keys=True,
        )
        + "\n"
        == text
    )
