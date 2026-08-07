from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import src.data.v2_targets as vt
from src.data.build_v2_dataset import (
    V2RawTables,
    build_verified_v2_feature_dataset,
    label_maturity_mask,
    load_verified_v2_raw_tables,
)
from src.data.v2_targets import (
    FINAL_TEST_PROBABILITY_COLUMNS,
    V2_TARGET_TABLE_COLUMNS,
    access_v2_final_test_targets,
    build_mature_v2_target_table,
    load_verified_v2_final_test_targets,
    load_verified_v2_mature_targets,
    validate_final_test_probability_artifact,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw" / "v2"
_MANIFEST_PATH = _RAW_DIR / "v2_synthetic_benchmark.manifest.json"
_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_synthetic_benchmark.json"


def _feature_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "appointment_id": pd.Series([1, 2, 3, 4], dtype="int64"),
            "prediction_time": pd.to_datetime(
                [
                    "2024-01-01 09:00:00",
                    "2024-01-02 09:00:00",
                    "2025-01-01 09:00:00",
                    "2027-01-01 09:00:00",
                ]
            ).astype("datetime64[ns]"),
            "evaluation_partition": pd.Series(
                [
                    "development_fit",
                    "development_fit",
                    "fold_1_validation",
                    "final_test",
                ],
                dtype="string",
            ),
            "label_available_at": pd.to_datetime(
                [
                    "2024-01-02 10:00:00",
                    "2025-01-01 00:00:00",
                    "2025-01-02 10:00:00",
                    "2027-01-02 10:00:00",
                ]
            ).astype("datetime64[ns]"),
        }
    )


def _appointments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "appointment_id": pd.Series([1, 2, 3, 4], dtype="int64"),
            "status": pd.Series(
                ["completed", "no_show", "cancelled", "no_show"],
                dtype="string",
            ),
        }
    )


def _write_probability(
    path: Path,
    *,
    appointment_ids: list[int] | None = None,
    values: list[float] | None = None,
) -> Path:
    ids = [4] if appointment_ids is None else appointment_ids
    probability = [0.25] if values is None else values
    pd.DataFrame(
        {
            "appointment_id": ids,
            "no_show_probability": probability,
        }
    ).to_csv(path, index=False, lineterminator="\n")
    return path


@pytest.fixture(scope="module")
def benchmark_dataset() -> pd.DataFrame:
    return build_verified_v2_feature_dataset(
        raw_dir=_RAW_DIR,
        manifest_path=_MANIFEST_PATH,
        config_path=_CONFIG_PATH,
    )


@pytest.fixture(scope="module")
def benchmark_tables() -> V2RawTables:
    return load_verified_v2_raw_tables(
        raw_dir=_RAW_DIR,
        manifest_path=_MANIFEST_PATH,
        config_path=_CONFIG_PATH,
    )


def test_mature_target_table_is_exact_and_strict() -> None:
    result = build_mature_v2_target_table(
        _feature_dataset(),
        _appointments(),
        model_fit_time=pd.Timestamp("2025-01-01"),
        allowed_partitions=("development_fit",),
    )
    assert list(result.columns) == list(V2_TARGET_TABLE_COLUMNS)
    assert result["appointment_id"].tolist() == [1]
    assert result["target"].tolist() == [0]
    assert str(result["target"].dtype) == "int8"
    assert str(result["label_available_at"].dtype) == "datetime64[ns]"


def test_equal_time_label_is_not_mature() -> None:
    dataset = _feature_dataset()
    mask = label_maturity_mask(
        dataset,
        model_fit_time=pd.Timestamp("2025-01-01"),
        allowed_partitions=("development_fit",),
    )
    assert mask.tolist() == [True, False, False, False]


def test_mature_targets_support_multiple_non_test_partitions() -> None:
    result = build_mature_v2_target_table(
        _feature_dataset(),
        _appointments(),
        model_fit_time=pd.Timestamp("2026-01-01"),
        allowed_partitions=("development_fit", "fold_1_validation"),
    )
    assert result["appointment_id"].tolist() == [1, 2, 3]
    assert result["target"].tolist() == [0, 1, 0]


def test_mature_target_accessor_rejects_final_test() -> None:
    with pytest.raises(PermissionError, match="protected final-test"):
        build_mature_v2_target_table(
            _feature_dataset(),
            _appointments(),
            model_fit_time=pd.Timestamp("2028-01-01"),
            allowed_partitions=("development_fit", "final_test"),
        )


def test_mature_target_accessor_rejects_unknown_partition() -> None:
    with pytest.raises(ValueError, match="Unknown allowed"):
        build_mature_v2_target_table(
            _feature_dataset(),
            _appointments(),
            model_fit_time=pd.Timestamp("2028-01-01"),
            allowed_partitions=("unknown",),
        )


def test_mature_target_accessor_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="No strictly mature"):
        build_mature_v2_target_table(
            _feature_dataset(),
            _appointments(),
            model_fit_time=pd.Timestamp("2023-01-01"),
            allowed_partitions=("development_fit",),
        )


def test_mature_target_accessor_rejects_invalid_status() -> None:
    appointments = _appointments()
    appointments.loc[0, "status"] = "unknown"
    with pytest.raises(ValueError, match="invalid final statuses"):
        build_mature_v2_target_table(
            _feature_dataset(),
            appointments,
            model_fit_time=pd.Timestamp("2026-01-01"),
            allowed_partitions=("development_fit",),
        )


def test_mature_target_accessor_rejects_duplicate_source_id() -> None:
    appointments = pd.concat([_appointments(), _appointments().iloc[[0]]])
    with pytest.raises(ValueError, match="appointment_id must be unique"):
        build_mature_v2_target_table(
            _feature_dataset(),
            appointments,
            model_fit_time=pd.Timestamp("2026-01-01"),
            allowed_partitions=("development_fit",),
        )


def test_mature_target_inputs_are_not_mutated() -> None:
    dataset = _feature_dataset()
    appointments = _appointments()
    dataset_snapshot = dataset.copy(deep=True)
    appointments_snapshot = appointments.copy(deep=True)
    build_mature_v2_target_table(
        dataset,
        appointments,
        model_fit_time=pd.Timestamp("2026-01-01"),
        allowed_partitions=("development_fit",),
    )
    pd.testing.assert_frame_equal(dataset, dataset_snapshot)
    pd.testing.assert_frame_equal(appointments, appointments_snapshot)


def test_verified_benchmark_mature_target_count_matches_mask(
    benchmark_dataset: pd.DataFrame,
) -> None:
    fit_time = pd.Timestamp("2025-01-01")
    expected = label_maturity_mask(
        benchmark_dataset,
        model_fit_time=fit_time,
        allowed_partitions=("development_fit",),
    )
    result = load_verified_v2_mature_targets(
        model_fit_time=fit_time,
        allowed_partitions=("development_fit",),
        raw_dir=_RAW_DIR,
        manifest_path=_MANIFEST_PATH,
        config_path=_CONFIG_PATH,
    )
    assert len(result) == int(expected.sum())
    assert not result["evaluation_partition"].eq("final_test").any()
    assert set(result["target"].unique()).issubset({0, 1})


def test_final_test_requires_explicit_allow_true(tmp_path: Path) -> None:
    path = _write_probability(tmp_path / "probability.csv")
    with pytest.raises(PermissionError, match="allow_test=True"):
        access_v2_final_test_targets(
            _feature_dataset(),
            _appointments(),
            probability_path=path,
        )


def test_final_test_allow_test_requires_exact_bool(tmp_path: Path) -> None:
    path = _write_probability(tmp_path / "probability.csv")
    with pytest.raises(TypeError, match="exact bool"):
        access_v2_final_test_targets(
            _feature_dataset(),
            _appointments(),
            probability_path=path,
            allow_test=1,
        )


def test_probability_artifact_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact is missing"):
        validate_final_test_probability_artifact(
            tmp_path / "missing.csv",
            _feature_dataset(),
        )


def test_probability_artifact_columns_are_exact(tmp_path: Path) -> None:
    path = tmp_path / "probability.csv"
    pd.DataFrame(
        {
            "appointment_id": [4],
            "no_show_probability": [0.25],
            "target": [1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_final_test_probability_artifact(path, _feature_dataset())


def test_probability_artifact_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = _write_probability(
        tmp_path / "probability.csv",
        appointment_ids=[4, 4],
        values=[0.25, 0.25],
    )
    with pytest.raises(ValueError, match="must be unique"):
        validate_final_test_probability_artifact(path, _feature_dataset())


def test_probability_artifact_rejects_wrong_ids_or_order(tmp_path: Path) -> None:
    path = _write_probability(
        tmp_path / "probability.csv",
        appointment_ids=[99],
        values=[0.25],
    )
    with pytest.raises(ValueError, match="values and order"):
        validate_final_test_probability_artifact(path, _feature_dataset())


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf")])
def test_probability_artifact_rejects_invalid_values(
    tmp_path: Path,
    value: float,
) -> None:
    path = _write_probability(
        tmp_path / f"probability-{value}.csv",
        values=[value],
    )
    with pytest.raises(ValueError, match="finite|within"):
        validate_final_test_probability_artifact(path, _feature_dataset())


def test_probability_artifact_returns_exact_seal(tmp_path: Path) -> None:
    path = _write_probability(tmp_path / "probability.csv")
    seal = validate_final_test_probability_artifact(path, _feature_dataset())
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert seal.path == path.resolve()
    assert seal.sha256 == expected_hash
    assert seal.row_count == 1
    assert tuple(pd.read_csv(path).columns) == FINAL_TEST_PROBABILITY_COLUMNS


def test_fixture_final_test_access_requires_prior_probability(tmp_path: Path) -> None:
    path = _write_probability(tmp_path / "probability.csv")
    access = access_v2_final_test_targets(
        _feature_dataset(),
        _appointments(),
        probability_path=path,
        allow_test=True,
    )
    assert access.probability_seal.sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert access.target_table["appointment_id"].tolist() == [4]
    assert access.target_table["target"].tolist() == [1]
    assert access.target_table["evaluation_partition"].tolist() == [
        "final_test"
    ]


def test_verified_final_test_denial_does_not_need_probability_file() -> None:
    with pytest.raises(PermissionError, match="allow_test=True"):
        load_verified_v2_final_test_targets(
            probability_path=Path("does-not-exist.csv"),
            allow_test=False,
        )


def test_verified_loader_validates_probability_before_raw_status_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fake_feature_loader(_: Path) -> pd.DataFrame:
        return _feature_dataset()

    def fail_raw_loader(**_: object) -> V2RawTables:
        nonlocal called
        called = True
        raise AssertionError("raw target source was loaded too early")

    import src.data.export_v2_processed as processed

    monkeypatch.setattr(
        processed,
        "load_frozen_v2_processed_feature_dataset",
        fake_feature_loader,
    )
    monkeypatch.setattr(vt, "load_verified_v2_raw_tables", fail_raw_loader)

    with pytest.raises(ValueError, match="artifact is missing"):
        load_verified_v2_final_test_targets(
            probability_path=tmp_path / "missing.csv",
            allow_test=True,
            processed_dir=tmp_path,
        )
    assert called is False


def test_fixture_verified_loader_reads_raw_only_after_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_probability(tmp_path / "probability.csv")
    events: list[str] = []

    def fake_feature_loader(_: Path) -> pd.DataFrame:
        events.append("features")
        return _feature_dataset()

    original_validator = vt.validate_final_test_probability_artifact

    def recording_validator(
        probability_path: Path,
        feature_dataset: pd.DataFrame,
    ) -> vt.FinalTestProbabilitySeal:
        events.append("probability")
        return original_validator(probability_path, feature_dataset)

    def fake_raw_loader(**_: object) -> V2RawTables:
        events.append("raw")
        return V2RawTables(
            appointments=_appointments(),
            patients=pd.DataFrame(),
            dentists=pd.DataFrame(),
        )

    import src.data.export_v2_processed as processed

    monkeypatch.setattr(
        processed,
        "load_frozen_v2_processed_feature_dataset",
        fake_feature_loader,
    )
    monkeypatch.setattr(
        vt,
        "validate_final_test_probability_artifact",
        recording_validator,
    )
    monkeypatch.setattr(vt, "load_verified_v2_raw_tables", fake_raw_loader)

    access = load_verified_v2_final_test_targets(
        probability_path=path,
        allow_test=True,
        processed_dir=tmp_path,
    )
    assert events == ["features", "probability", "raw"]
    assert access.target_table["target"].tolist() == [1]
