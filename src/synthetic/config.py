"""Load and validate the frozen Version 2 synthetic benchmark configuration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    _REPOSITORY_ROOT / "configs" / "v2_synthetic_benchmark.json"
)
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
EXPECTED_RNG_STREAMS = (
    "patients",
    "dentists",
    "appointments",
    "reminders",
    "latent_risk",
    "outcomes",
    "status_timestamps",
)
EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "generator_version",
        "root_seed",
        "patient_count",
        "dentist_count",
        "appointment_count",
        "appointment_start",
        "appointment_end",
        "prediction_horizon_hours",
        "output_directory",
        "rng_streams",
        "evaluation",
    }
)
EXPECTED_EVALUATION_KEYS = frozenset(
    {
        "warmup",
        "rolling_folds",
        "calibration",
        "policy_selection",
        "final_test",
    }
)


@dataclass(frozen=True, slots=True)
class DateWindow:
    """A half-open calendar interval."""

    start: date
    end: date


@dataclass(frozen=True, slots=True)
class RollingFold:
    """A chronological fit and validation pair."""

    name: str
    fit: DateWindow
    validation: DateWindow


@dataclass(frozen=True, slots=True)
class EvaluationSchedule:
    """All protected Version 2 evaluation windows."""

    warmup: DateWindow
    rolling_folds: tuple[RollingFold, ...]
    calibration: DateWindow
    policy_selection: DateWindow
    final_test: DateWindow


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated frozen configuration for the Version 2 benchmark."""

    schema_version: str
    generator_version: str
    root_seed: int
    patient_count: int
    dentist_count: int
    appointment_count: int
    appointment_start: date
    appointment_end: date
    prediction_horizon_hours: int
    output_directory: Path
    rng_streams: tuple[str, ...]
    evaluation: EvaluationSchedule


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    supplied = frozenset(value)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"{context} keys must match the frozen contract; "
            f"missing={missing}, extra={extra}"
        )


def _parse_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use canonical YYYY-MM-DD format")
    return parsed


def _parse_positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _parse_window(value: Any, *, context: str) -> DateWindow:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(
        value,
        frozenset({"start", "end"}),
        context=context,
    )
    window = DateWindow(
        start=_parse_date(value["start"], field=f"{context}.start"),
        end=_parse_date(value["end"], field=f"{context}.end"),
    )
    if window.start >= window.end:
        raise ValueError(f"{context} must satisfy start < end")
    return window


def _parse_rolling_folds(value: Any) -> tuple[RollingFold, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("evaluation.rolling_folds must contain exactly 3 folds")

    folds: list[RollingFold] = []
    expected_names = ("fold_1", "fold_2", "fold_3")
    expected_keys = frozenset(
        {
            "name",
            "fit_start",
            "fit_end",
            "validation_start",
            "validation_end",
        }
    )
    for index, item in enumerate(value):
        context = f"evaluation.rolling_folds[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{context} must be an object")
        _require_exact_keys(item, expected_keys, context=context)
        name = item["name"]
        if name != expected_names[index]:
            raise ValueError(
                f"{context}.name must be {expected_names[index]!r}"
            )
        fit = DateWindow(
            start=_parse_date(
                item["fit_start"],
                field=f"{context}.fit_start",
            ),
            end=_parse_date(
                item["fit_end"],
                field=f"{context}.fit_end",
            ),
        )
        validation = DateWindow(
            start=_parse_date(
                item["validation_start"],
                field=f"{context}.validation_start",
            ),
            end=_parse_date(
                item["validation_end"],
                field=f"{context}.validation_end",
            ),
        )
        if fit.start >= fit.end:
            raise ValueError(f"{context}.fit must satisfy start < end")
        if validation.start >= validation.end:
            raise ValueError(
                f"{context}.validation must satisfy start < end"
            )
        if fit.end != validation.start:
            raise ValueError(
                f"{context} fit end must equal validation start"
            )
        folds.append(
            RollingFold(
                name=name,
                fit=fit,
                validation=validation,
            )
        )

    for previous, current in zip(folds, folds[1:]):
        if previous.validation.end != current.validation.start:
            raise ValueError(
                "rolling validation windows must be contiguous"
            )
        if current.fit.start != previous.fit.start:
            raise ValueError(
                "rolling folds must preserve the initial fit boundary"
            )
        if current.fit.end != previous.validation.end:
            raise ValueError(
                "each rolling fit must include the prior validation window"
            )
    return tuple(folds)


def _parse_output_directory(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("output_directory must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "output_directory must be a repository-relative safe path"
        )
    normalized = path.as_posix()
    if normalized != "data/raw/v2":
        raise ValueError(
            "output_directory must remain frozen as data/raw/v2"
        )
    return path


def _validate_schedule(
    config_start: date,
    config_end: date,
    schedule: EvaluationSchedule,
) -> None:
    folds = schedule.rolling_folds
    if schedule.warmup.start != config_start:
        raise ValueError(
            "warmup start must equal appointment_start"
        )
    if schedule.warmup.end != folds[0].fit.start:
        raise ValueError(
            "warmup end must equal the first fit start"
        )
    if folds[-1].validation.end != schedule.calibration.start:
        raise ValueError(
            "final rolling validation end must equal calibration start"
        )
    if schedule.calibration.end != schedule.policy_selection.start:
        raise ValueError(
            "calibration end must equal policy-selection start"
        )
    if schedule.policy_selection.end != schedule.final_test.start:
        raise ValueError(
            "policy-selection end must equal final-test start"
        )
    if schedule.final_test.end != config_end + timedelta(days=1):
        raise ValueError(
            "final-test end must be one day after appointment_end"
        )


def parse_benchmark_config(payload: Mapping[str, Any]) -> BenchmarkConfig:
    """Validate a decoded JSON object and return an immutable configuration."""

    if not isinstance(payload, dict):
        raise ValueError("benchmark configuration must be a JSON object")
    _require_exact_keys(
        payload,
        EXPECTED_TOP_LEVEL_KEYS,
        context="configuration",
    )

    schema_version = payload["schema_version"]
    generator_version = payload["generator_version"]
    for field, value in (
        ("schema_version", schema_version),
        ("generator_version", generator_version),
    ):
        if not isinstance(value, str) or VERSION_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field} must use semantic version syntax")

    root_seed = _parse_positive_int(
        payload["root_seed"],
        field="root_seed",
    )
    patient_count = _parse_positive_int(
        payload["patient_count"],
        field="patient_count",
    )
    dentist_count = _parse_positive_int(
        payload["dentist_count"],
        field="dentist_count",
    )
    appointment_count = _parse_positive_int(
        payload["appointment_count"],
        field="appointment_count",
    )
    appointment_start = _parse_date(
        payload["appointment_start"],
        field="appointment_start",
    )
    appointment_end = _parse_date(
        payload["appointment_end"],
        field="appointment_end",
    )
    if appointment_start >= appointment_end:
        raise ValueError(
            "appointment_start must be earlier than appointment_end"
        )
    prediction_horizon_hours = _parse_positive_int(
        payload["prediction_horizon_hours"],
        field="prediction_horizon_hours",
    )

    rng_streams_value = payload["rng_streams"]
    if not isinstance(rng_streams_value, list):
        raise ValueError("rng_streams must be a list")
    rng_streams = tuple(rng_streams_value)
    if rng_streams != EXPECTED_RNG_STREAMS:
        raise ValueError(
            "rng_streams must match the frozen ordered stream contract"
        )
    if len(set(rng_streams)) != len(rng_streams):
        raise ValueError("rng_streams must be unique")

    evaluation = payload["evaluation"]
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    _require_exact_keys(
        evaluation,
        EXPECTED_EVALUATION_KEYS,
        context="evaluation",
    )
    schedule = EvaluationSchedule(
        warmup=_parse_window(
            evaluation["warmup"],
            context="evaluation.warmup",
        ),
        rolling_folds=_parse_rolling_folds(
            evaluation["rolling_folds"]
        ),
        calibration=_parse_window(
            evaluation["calibration"],
            context="evaluation.calibration",
        ),
        policy_selection=_parse_window(
            evaluation["policy_selection"],
            context="evaluation.policy_selection",
        ),
        final_test=_parse_window(
            evaluation["final_test"],
            context="evaluation.final_test",
        ),
    )
    _validate_schedule(
        appointment_start,
        appointment_end,
        schedule,
    )

    return BenchmarkConfig(
        schema_version=schema_version,
        generator_version=generator_version,
        root_seed=root_seed,
        patient_count=patient_count,
        dentist_count=dentist_count,
        appointment_count=appointment_count,
        appointment_start=appointment_start,
        appointment_end=appointment_end,
        prediction_horizon_hours=prediction_horizon_hours,
        output_directory=_parse_output_directory(
            payload["output_directory"]
        ),
        rng_streams=rng_streams,
        evaluation=schedule,
    )


def load_benchmark_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> BenchmarkConfig:
    """Load and validate the frozen JSON configuration."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Benchmark configuration is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read benchmark configuration: {path}"
        ) from exc
    return parse_benchmark_config(payload)


def calculate_config_sha256(
    path: Path = DEFAULT_CONFIG_PATH,
) -> str:
    """Return the SHA-256 digest of the exact configuration file bytes."""

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Benchmark configuration is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_rng_stream_seeds(
    config: BenchmarkConfig,
) -> Mapping[str, int]:
    """Derive stable independent integer seeds for the named streams."""

    seeds: dict[str, int] = {}
    for stream_name in config.rng_streams:
        digest = hashlib.sha256(
            f"{config.root_seed}:{stream_name}".encode("utf-8")
        ).digest()
        seeds[stream_name] = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
    if len(set(seeds.values())) != len(seeds):
        raise RuntimeError("Derived RNG stream seeds must be unique")
    return MappingProxyType(seeds)


__all__ = (
    "BenchmarkConfig",
    "DateWindow",
    "EvaluationSchedule",
    "RollingFold",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_RNG_STREAMS",
    "calculate_config_sha256",
    "derive_rng_stream_seeds",
    "load_benchmark_config",
    "parse_benchmark_config",
)
