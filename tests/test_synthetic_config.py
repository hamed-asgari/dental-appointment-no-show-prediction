"""Tests for the frozen Version 2 synthetic benchmark configuration."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from src.synthetic import config as benchmark_config


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "v2_synthetic_benchmark.json"


def _payload() -> dict[str, object]:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_config_file_loads_exact_declared_values() -> None:
    config = benchmark_config.load_benchmark_config(_CONFIG_PATH)

    assert config.schema_version == "2.0.0"
    assert config.generator_version == "2.0.0"
    assert config.root_seed == 20260805
    assert config.patient_count == 4_000
    assert config.dentist_count == 7
    assert config.appointment_count == 24_000
    assert config.appointment_start == date(2023, 1, 1)
    assert config.appointment_end == date(2027, 12, 31)
    assert config.prediction_horizon_hours == 24
    assert config.output_directory.as_posix() == "data/raw/v2"
    assert config.rng_streams == benchmark_config.EXPECTED_RNG_STREAMS


def test_evaluation_windows_match_frozen_policy() -> None:
    schedule = benchmark_config.load_benchmark_config(
        _CONFIG_PATH
    ).evaluation

    assert schedule.warmup.start == date(2023, 1, 1)
    assert schedule.warmup.end == date(2024, 1, 1)

    expected_folds = (
        (
            "fold_1",
            date(2024, 1, 1),
            date(2025, 1, 1),
            date(2025, 1, 1),
            date(2025, 7, 1),
        ),
        (
            "fold_2",
            date(2024, 1, 1),
            date(2025, 7, 1),
            date(2025, 7, 1),
            date(2026, 1, 1),
        ),
        (
            "fold_3",
            date(2024, 1, 1),
            date(2026, 1, 1),
            date(2026, 1, 1),
            date(2026, 7, 1),
        ),
    )
    observed = tuple(
        (
            fold.name,
            fold.fit.start,
            fold.fit.end,
            fold.validation.start,
            fold.validation.end,
        )
        for fold in schedule.rolling_folds
    )
    assert observed == expected_folds

    assert schedule.calibration.start == date(2026, 7, 1)
    assert schedule.calibration.end == date(2026, 10, 1)
    assert schedule.policy_selection.start == date(2026, 10, 1)
    assert schedule.policy_selection.end == date(2027, 1, 1)
    assert schedule.final_test.start == date(2027, 1, 1)
    assert schedule.final_test.end == date(2028, 1, 1)


def test_config_is_deeply_immutable() -> None:
    config = benchmark_config.load_benchmark_config(_CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.root_seed = 1  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        config.evaluation.warmup.start = date(2022, 1, 1)  # type: ignore[misc]


def test_config_hash_is_exact_and_stable() -> None:
    first = benchmark_config.calculate_config_sha256(_CONFIG_PATH)
    second = benchmark_config.calculate_config_sha256(_CONFIG_PATH)

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_named_rng_stream_seeds_are_stable_and_unique() -> None:
    config = benchmark_config.load_benchmark_config(_CONFIG_PATH)

    first = benchmark_config.derive_rng_stream_seeds(config)
    second = benchmark_config.derive_rng_stream_seeds(config)

    assert tuple(first) == benchmark_config.EXPECTED_RNG_STREAMS
    assert dict(first) == dict(second)
    assert len(set(first.values())) == len(first)
    assert all(type(value) is int and value >= 0 for value in first.values())

    with pytest.raises(TypeError):
        first["patients"] = 1  # type: ignore[index]


def test_unknown_top_level_key_is_rejected() -> None:
    payload = _payload()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="keys must match"):
        benchmark_config.parse_benchmark_config(payload)


def test_boolean_is_not_accepted_as_an_integer() -> None:
    payload = _payload()
    payload["root_seed"] = True

    with pytest.raises(ValueError, match="positive integer"):
        benchmark_config.parse_benchmark_config(payload)


def test_rng_stream_order_is_frozen() -> None:
    payload = _payload()
    streams = list(payload["rng_streams"])
    streams.reverse()
    payload["rng_streams"] = streams

    with pytest.raises(ValueError, match="ordered stream contract"):
        benchmark_config.parse_benchmark_config(payload)


def test_rolling_validation_gap_is_rejected() -> None:
    payload = _payload()
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    folds = evaluation["rolling_folds"]
    assert isinstance(folds, list)
    second = folds[1]
    assert isinstance(second, dict)
    second["validation_start"] = "2025-08-01"

    with pytest.raises(ValueError, match="fit end must equal validation start"):
        benchmark_config.parse_benchmark_config(payload)


def test_final_test_must_end_after_configured_appointment_end() -> None:
    payload = _payload()
    evaluation = payload["evaluation"]
    assert isinstance(evaluation, dict)
    final_test = evaluation["final_test"]
    assert isinstance(final_test, dict)
    final_test["end"] = "2027-12-31"

    with pytest.raises(
        ValueError,
        match="one day after appointment_end",
    ):
        benchmark_config.parse_benchmark_config(payload)


def test_unsafe_output_directory_is_rejected() -> None:
    payload = _payload()
    payload["output_directory"] = "../outside"

    with pytest.raises(ValueError, match="repository-relative safe path"):
        benchmark_config.parse_benchmark_config(payload)


def test_missing_and_invalid_json_files_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="missing"):
        benchmark_config.load_benchmark_config(
            tmp_path / "missing.json"
        )

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read"):
        benchmark_config.load_benchmark_config(invalid)
