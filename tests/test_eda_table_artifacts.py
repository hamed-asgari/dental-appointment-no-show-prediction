"""Tests for deterministic serialization of the approved EDA table bundle."""

from __future__ import annotations

import ast
from collections import OrderedDict, UserDict
from collections.abc import Mapping
import csv
from importlib import util
import inspect
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Any, get_type_hints

import pandas as pd
import pytest

from src.analysis import artifacts
from src.analysis.artifacts import write_eda_tables
from src.analysis.run_eda import build_eda_tables
from src.data import build_dataset as bd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
ARTIFACTS_PATH = REPOSITORY_ROOT / "src" / "analysis" / "artifacts.py"
EXPECTED_ARTIFACTS = (
    ("cohort_target", "01_cohort_target.csv"),
    ("missingness", "02_missingness.csv"),
    ("numeric_features", "03_numeric_features.csv"),
    ("numeric_by_target", "04_numeric_by_target.csv"),
    ("categorical_features", "05_categorical_features.csv"),
    ("temporal_coverage", "06_temporal_coverage.csv"),
    ("temporal_monthly", "07_temporal_monthly.csv"),
    ("numeric_drift", "08_numeric_drift.csv"),
    ("categorical_drift_levels", "09_categorical_drift_levels.csv"),
    ("categorical_drift_features", "10_categorical_drift_features.csv"),
    ("numeric_relationships", "11_numeric_relationships.csv"),
)
EXPECTED_KEYS = tuple(key for key, _ in EXPECTED_ARTIFACTS)
EXPECTED_FILENAMES = tuple(filename for _, filename in EXPECTED_ARTIFACTS)
EXPECTED_HEADERS = {
    "cohort_target": (
        "rows",
        "positives",
        "negatives",
        "prevalence",
        "wilson_lower",
        "wilson_upper",
        "duplicate_appointment_ids",
    ),
    "missingness": (
        "feature",
        "rows",
        "missing_count",
        "missing_rate",
        "non_missing_count",
        "unique_non_null",
        "is_constant",
    ),
    "numeric_features": (
        "feature",
        "n",
        "missing_count",
        "missing_rate",
        "zero_count",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "q1",
        "median",
        "q3",
        "p95",
        "p99",
        "max",
        "iqr",
        "lower_fence",
        "upper_fence",
        "below_fence_count",
        "above_fence_count",
    ),
    "numeric_by_target": (
        "feature",
        "target",
        "n",
        "missing_count",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
    ),
    "categorical_features": (
        "feature",
        "level",
        "is_missing",
        "count",
        "share",
        "positives",
        "negatives",
        "no_show_rate",
        "wilson_lower",
        "wilson_upper",
        "is_rare",
        "has_high_uncertainty",
    ),
    "temporal_coverage": (
        "nominal_train_rows",
        "mature_train_rows",
        "maturity_exclusion_rows",
        "nominal_prediction_time_min",
        "nominal_prediction_time_max",
        "mature_prediction_time_min",
        "mature_prediction_time_max",
        "first_prediction_month",
        "last_prediction_month",
        "calendar_months_spanned",
    ),
    "temporal_monthly": (
        "prediction_month",
        "nominal_train_count",
        "mature_train_count",
        "maturity_exclusion_count",
        "positives",
        "negatives",
        "no_show_rate",
        "wilson_lower",
        "wilson_upper",
    ),
    "numeric_drift": (
        "feature",
        "train_rows",
        "validation_rows",
        "train_n",
        "validation_n",
        "train_missing_count",
        "validation_missing_count",
        "train_missing_rate",
        "validation_missing_rate",
        "missing_rate_difference",
        "train_mean",
        "validation_mean",
        "train_std",
        "validation_std",
        "signed_smd",
        "train_q10",
        "validation_q10",
        "q10_shift",
        "train_median",
        "validation_median",
        "median_shift",
        "train_q90",
        "validation_q90",
        "q90_shift",
    ),
    "categorical_drift_levels": (
        "feature",
        "level",
        "is_missing",
        "train_count",
        "validation_count",
        "train_share",
        "validation_share",
        "share_difference",
        "absolute_share_difference",
        "contribution_to_total_variation",
        "is_unseen_in_train",
        "is_absent_in_validation",
    ),
    "categorical_drift_features": (
        "feature",
        "train_rows",
        "validation_rows",
        "train_missing_count",
        "validation_missing_count",
        "train_missing_rate",
        "validation_missing_rate",
        "missing_rate_difference",
        "train_distinct_nonmissing_levels",
        "validation_distinct_nonmissing_levels",
        "unseen_in_train_level_count",
        "unseen_in_train_validation_count",
        "unseen_in_train_validation_share",
        "absent_in_validation_level_count",
        "absent_in_validation_train_count",
        "absent_in_validation_train_share",
        "total_variation_distance",
        "max_absolute_share_difference",
    ),
    "numeric_relationships": (
        "feature_a",
        "feature_b",
        "train_rows",
        "paired_n",
        "missing_pair_count",
        "paired_rate",
        "feature_a_unique_n",
        "feature_b_unique_n",
        "pearson_correlation",
        "absolute_pearson_correlation",
        "spearman_correlation",
        "absolute_spearman_correlation",
    ),
}
AUTHENTIC_ROW_COUNTS = {
    "cohort_target": 1,
    "missingness": 10,
    "numeric_features": 5,
    "numeric_by_target": 10,
    "categorical_features": 54,
    "temporal_coverage": 1,
    "temporal_monthly": 12,
    "numeric_drift": 5,
    "categorical_drift_levels": 59,
    "categorical_drift_features": 5,
    "numeric_relationships": 10,
}
_FORBIDDEN_PLOTTING_NAMES = {
    "plot",
    "plotting",
    "savefig",
    "show",
    "figure",
    "subplots",
    "imshow",
    "hist",
    "scatter",
    "bar",
    "boxplot",
    "violinplot",
    "heatmap",
}


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    raw_tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(raw_tables)


@pytest.fixture(scope="session")
def authentic_tables(
    canonical_dataset: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return build_eda_tables(canonical_dataset)


def _sentinel_tables() -> dict[str, pd.DataFrame]:
    return {
        key: pd.DataFrame({"sentinel": [position], "table": [key]})
        for position, key in enumerate(EXPECTED_KEYS, start=1)
    }


def _snapshot_mapping(
    mapping: Mapping[object, object],
) -> tuple[
    tuple[object, ...],
    tuple[int, ...],
    tuple[object, ...],
]:
    values = tuple(mapping.values())
    copies = tuple(
        value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
        for value in values
    )
    return tuple(mapping), tuple(id(value) for value in values), copies


def _assert_mapping_unchanged(
    mapping: Mapping[object, object],
    snapshot: tuple[
        tuple[object, ...],
        tuple[int, ...],
        tuple[object, ...],
    ],
) -> None:
    keys, identities, copies = snapshot
    values = tuple(mapping.values())
    assert tuple(mapping) == keys
    assert tuple(id(value) for value in values) == identities
    for actual, before in zip(values, copies):
        if isinstance(actual, pd.DataFrame):
            assert isinstance(before, pd.DataFrame)
            pd.testing.assert_frame_equal(actual, before)
        else:
            assert actual == before


def _csv_records(path: Path) -> tuple[list[str], ...]:
    text = path.read_bytes().decode("utf-8")
    return tuple(csv.reader(StringIO(text, newline="")))


def _csv_data_rows(path: Path) -> int:
    return len(_csv_records(path)) - 1


def _read_artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        filename: (output_dir / filename).read_bytes()
        for filename in EXPECTED_FILENAMES
    }


def _assert_only_completed_artifacts(
    output_dir: Path,
    completed_count: int,
) -> None:
    entries = tuple(output_dir.iterdir())
    assert {path.name for path in entries} == set(
        EXPECTED_FILENAMES[:completed_count]
    )
    assert all(path.is_file() for path in entries)
    assert not any(path.suffix == ".tmp" for path in output_dir.rglob("*"))
    assert not any(path.name.startswith(".") for path in output_dir.rglob("*"))


def _assignment_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(
            *(_assignment_target_names(element) for element in target.elts)
        )
    if isinstance(target, ast.Starred):
        return _assignment_target_names(target.value)
    return set()


def _module_owned_bindings(tree: ast.Module) -> dict[str, set[str]]:
    bindings = {
        "functions": set(),
        "classes": set(),
        "assignments": set(),
        "type_aliases": set(),
    }
    type_alias_node = getattr(ast, "TypeAlias", None)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings["functions"].add(node.name)
        elif isinstance(node, ast.ClassDef):
            bindings["classes"].add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                bindings["assignments"].update(
                    _assignment_target_names(target)
                )
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            bindings["assignments"].update(
                _assignment_target_names(node.target)
            )
        elif type_alias_node is not None and isinstance(node, type_alias_node):
            bindings["type_aliases"].update(
                _assignment_target_names(node.name)
            )
    return bindings


def _public_module_owned_names(tree: ast.Module) -> set[str]:
    bindings = _module_owned_bindings(tree)
    return {
        name
        for names in bindings.values()
        for name in names
        if not name.startswith("_")
    }


def _plotting_executable_names(tree: ast.AST) -> set[str]:
    detected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_PLOTTING_NAMES:
                detected.add(node.attr)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                terminal_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                terminal_name = node.func.attr
            else:
                continue
            if terminal_name in _FORBIDDEN_PLOTTING_NAMES:
                detected.add(terminal_name)
    return detected


def test_public_signature_is_exact() -> None:
    signature = inspect.signature(write_eda_tables)
    hints = get_type_hints(write_eda_tables)

    assert tuple(signature.parameters) == ("tables", "output_dir")
    assert len(signature.parameters) == 2
    assert hints["tables"] == dict[str, pd.DataFrame]
    assert hints["output_dir"] == str | Path
    assert hints["return"] == dict[str, Path]


def test_exact_filename_and_returned_key_order(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "tables"
    result = write_eda_tables(_sentinel_tables(), output_dir)

    assert tuple(result) == EXPECTED_KEYS
    assert tuple(path.name for path in result.values()) == EXPECTED_FILENAMES
    assert all(path.parent == output_dir for path in result.values())
    assert tuple(path.name for path in output_dir.iterdir()) == EXPECTED_FILENAMES


def test_exact_csv_bytes(tmp_path: Path) -> None:
    tables = _sentinel_tables()
    tables["cohort_target"] = pd.DataFrame(
        {
            "integer": pd.Series([-7, 42, 0, 1, 2, 3], dtype="int64"),
            "floating": [
                1.2345678901234567,
                1e-100,
                1e100,
                0.0,
                -0.0,
                float("nan"),
            ],
            "text": [
                "",
                "plain ASCII",
                "\u03bb",
                "with,comma",
                'with "quote"',
                "line one\nline two",
            ],
            "observed_at": [
                pd.Timestamp("2024-01-02 03:04:05.123456"),
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
        }
    )
    expected = (
        b"integer,floating,text,observed_at\n"
        b"-7,1.2345678901234567,,2024-01-02T03:04:05.123456\n"
        b"42,1e-100,plain ASCII,<NA>\n"
        b"0,1e+100,\xce\xbb,<NA>\n"
        b'1,0,"with,comma",<NA>\n'
        b'2,-0,"with ""quote""",<NA>\n'
        b'3,<NA>,"line one\nline two",<NA>\n'
    )

    path = write_eda_tables(tables, tmp_path / "output")["cohort_target"]
    actual = path.read_bytes()

    assert actual == expected
    assert not actual.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in actual
    assert b"\r" not in actual
    assert len(actual) - len(actual.rstrip(b"\n")) == 1
    assert b"1.234568" not in actual
    assert not actual.startswith(b",")
    assert b",1e-100," in actual
    assert b",1e+100," in actual
    assert b",0," in actual
    assert b",-0," in actual
    assert b"\xce\xbb" in actual
    assert b',"line one\nline two",' in actual
    records = tuple(csv.reader(StringIO(actual.decode("utf-8"), newline="")))
    assert len(records) == 7
    assert records[-1][2] == "line one\nline two"


def test_authentic_integration_reconciles_all_rows(
    tmp_path: Path,
    authentic_tables: dict[str, pd.DataFrame],
) -> None:
    output_dir = tmp_path / "authentic"
    result = write_eda_tables(authentic_tables, output_dir)

    assert tuple(EXPECTED_HEADERS) == EXPECTED_KEYS
    assert tuple(result) == EXPECTED_KEYS
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        EXPECTED_FILENAMES
    )
    assert all(path.is_file() for path in output_dir.iterdir())
    assert len(tuple(output_dir.iterdir())) == 11
    assert {
        key: _csv_data_rows(result[key])
        for key in EXPECTED_KEYS
    } == AUTHENTIC_ROW_COUNTS
    for key in EXPECTED_KEYS:
        in_memory_header = tuple(authentic_tables[key].columns)
        records = _csv_records(result[key])
        parsed_header = tuple(records[0])
        assert in_memory_header == EXPECTED_HEADERS[key]
        assert parsed_header == EXPECTED_HEADERS[key]
        assert parsed_header == in_memory_header
        assert len(parsed_header) == len(set(parsed_header))


def test_authentic_bytes_are_deterministic_across_directories(
    tmp_path: Path,
    authentic_tables: dict[str, pd.DataFrame],
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = write_eda_tables(authentic_tables, first_dir)
    second = write_eda_tables(authentic_tables, second_dir)

    assert tuple(first) == tuple(second) == EXPECTED_KEYS
    assert tuple(path.name for path in first.values()) == tuple(
        path.name for path in second.values()
    )
    assert _read_artifact_bytes(first_dir) == _read_artifact_bytes(second_dir)


def test_repeated_overwrite_is_deterministic_and_leaves_no_temporary_file(
    tmp_path: Path,
    authentic_tables: dict[str, pd.DataFrame],
) -> None:
    output_dir = tmp_path / "repeated"
    first = write_eda_tables(authentic_tables, output_dir)
    first_bytes = _read_artifact_bytes(output_dir)

    second = write_eda_tables(authentic_tables, output_dir)

    assert first == second
    assert tuple(first) == tuple(second) == EXPECTED_KEYS
    assert _read_artifact_bytes(output_dir) == first_bytes
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        EXPECTED_FILENAMES
    )


def test_existing_unrelated_entries_are_preserved(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    unrelated_file = output_dir / "notes.txt"
    unrelated_file.write_bytes(b"leave this unchanged\n")
    unrelated_dir = output_dir / "supporting-material"
    unrelated_dir.mkdir()
    nested_file = unrelated_dir / "source.txt"
    nested_file.write_bytes(b"also unchanged\n")

    write_eda_tables(_sentinel_tables(), output_dir)

    assert unrelated_file.read_bytes() == b"leave this unchanged\n"
    assert nested_file.read_bytes() == b"also unchanged\n"
    assert unrelated_dir.is_dir()


def test_existing_approved_outputs_are_replaced(tmp_path: Path) -> None:
    tables = _sentinel_tables()
    reference_dir = tmp_path / "reference"
    expected = write_eda_tables(tables, reference_dir)
    expected_bytes = {
        key: path.read_bytes()
        for key, path in expected.items()
    }
    output_dir = tmp_path / "replace"
    output_dir.mkdir()
    for filename in EXPECTED_FILENAMES:
        (output_dir / filename).write_bytes(b"incorrect\r\n")

    actual = write_eda_tables(tables, output_dir)

    assert {
        key: path.read_bytes()
        for key, path in actual.items()
    } == expected_bytes


def _invalid_bundle(case: str) -> Mapping[object, object]:
    tables: dict[object, object] = _sentinel_tables()
    if case == "dict_subclass":
        return OrderedDict(tables)
    if case == "non_dict_mapping":
        return UserDict(tables)
    if case == "missing_key":
        tables.pop("numeric_relationships")
    elif case == "extra_key":
        tables["extra"] = pd.DataFrame({"sentinel": [12]})
    elif case == "reordered_keys":
        tables = dict(reversed(tuple(tables.items())))
    elif case == "non_string_key":
        first_value = tables.pop("cohort_target")
        tables = {1: first_value, **tables}
    elif case == "non_dataframe_value":
        tables["numeric_features"] = "not a frame"
    elif case == "duplicate_dataframe":
        tables["numeric_features"] = tables["missingness"]
    else:
        raise AssertionError(f"unknown invalid case: {case}")
    return tables


@pytest.mark.parametrize(
    ("case", "exception_type"),
    (
        ("dict_subclass", TypeError),
        ("non_dict_mapping", TypeError),
        ("missing_key", ValueError),
        ("extra_key", ValueError),
        ("reordered_keys", ValueError),
        ("non_string_key", TypeError),
        ("non_dataframe_value", TypeError),
        ("duplicate_dataframe", ValueError),
    ),
)
def test_invalid_bundle_is_rejected_before_filesystem_mutation(
    tmp_path: Path,
    case: str,
    exception_type: type[Exception],
) -> None:
    tables = _invalid_bundle(case)
    snapshot = _snapshot_mapping(tables)
    output_dir = tmp_path / case

    with pytest.raises(exception_type, match="tables|table key"):
        write_eda_tables(tables, output_dir)  # type: ignore[arg-type]

    assert not output_dir.exists()
    _assert_mapping_unchanged(tables, snapshot)


def test_output_dir_that_exists_as_file_is_preserved(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    original = b"existing file\x00content"
    output_file.write_bytes(original)

    with pytest.raises(NotADirectoryError, match="not a directory"):
        write_eda_tables(_sentinel_tables(), output_file)

    assert output_file.is_file()
    assert output_file.read_bytes() == original


def test_invalid_output_path_type_is_rejected_without_filesystem_changes(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())

    with pytest.raises(TypeError):
        write_eda_tables(_sentinel_tables(), object())  # type: ignore[arg-type]

    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.parametrize("failure_position", (1, 6, 11))
def test_serialization_failure_precedes_filesystem_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    error = type("UniqueSerializationError", (Exception,), {})(
        f"serialization failure at {failure_position}"
    )
    tables = _sentinel_tables()
    frame_keys = {
        id(frame): key
        for key, frame in tables.items()
    }
    serialized_keys: list[str] = []
    original_to_csv = pd.DataFrame.to_csv

    def record_serialization(
        frame: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> str | None:
        key = frame_keys[id(frame)]
        serialized_keys.append(key)
        if len(serialized_keys) == failure_position:
            raise error
        return original_to_csv(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", record_serialization)
    snapshot = _snapshot_mapping(tables)
    output_dir = tmp_path / f"serialization-failure-{failure_position}"

    with pytest.raises(type(error)) as caught:
        write_eda_tables(tables, output_dir)

    assert caught.value is error
    assert serialized_keys == list(EXPECTED_KEYS[:failure_position])
    assert not output_dir.exists()
    assert tuple(tmp_path.iterdir()) == ()
    _assert_mapping_unchanged(tables, snapshot)


@pytest.mark.parametrize("failure_position", (1, 6, 11))
def test_replace_failure_propagates_and_removes_active_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    tables = _sentinel_tables()
    snapshot = _snapshot_mapping(tables)
    output_dir = tmp_path / f"replace-failure-{failure_position}"
    error = OSError(f"replace failure at {failure_position}")
    original_replace = artifacts.os.replace
    attempted_destinations: list[Path] = []

    def fail_selected_replace(source: Path, destination: Path) -> None:
        attempted_destinations.append(Path(destination))
        if len(attempted_destinations) == failure_position:
            raise error
        original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_selected_replace)

    with pytest.raises(OSError) as caught:
        write_eda_tables(tables, output_dir)

    assert caught.value is error
    assert tuple(path.name for path in attempted_destinations) == (
        EXPECTED_FILENAMES[:failure_position]
    )
    _assert_only_completed_artifacts(output_dir, failure_position - 1)
    failed_path = output_dir / EXPECTED_FILENAMES[failure_position - 1]
    assert not failed_path.exists()
    assert not getattr(error, "__notes__", ())
    _assert_mapping_unchanged(tables, snapshot)


@pytest.mark.parametrize("failure_position", (1, 6, 11))
def test_fsync_failure_propagates_and_removes_active_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    tables = _sentinel_tables()
    snapshot = _snapshot_mapping(tables)
    output_dir = tmp_path / f"fsync-failure-{failure_position}"
    error = OSError(f"fsync failure at {failure_position}")
    original_fsync = artifacts.os.fsync
    original_replace = artifacts.os.replace
    fsync_calls = 0
    replaced_destinations: list[Path] = []

    def fail_selected_fsync(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == failure_position:
            raise error
        original_fsync(file_descriptor)

    def record_replace(source: Path, destination: Path) -> None:
        replaced_destinations.append(Path(destination))
        original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "fsync", fail_selected_fsync)
    monkeypatch.setattr(artifacts.os, "replace", record_replace)

    with pytest.raises(OSError) as caught:
        write_eda_tables(tables, output_dir)

    assert caught.value is error
    assert fsync_calls == failure_position
    assert tuple(path.name for path in replaced_destinations) == (
        EXPECTED_FILENAMES[: failure_position - 1]
    )
    _assert_only_completed_artifacts(output_dir, failure_position - 1)
    failed_path = output_dir / EXPECTED_FILENAMES[failure_position - 1]
    assert not failed_path.exists()
    assert not getattr(error, "__notes__", ())
    _assert_mapping_unchanged(tables, snapshot)


def test_atomic_operation_order_and_temporary_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "atomic-order"
    output_dir.mkdir()
    destination = output_dir / EXPECTED_FILENAMES[0]
    events: list[str] = []
    temporary_paths: list[Path] = []
    wrapped_files: list[Any] = []
    original_named_temporary_file = artifacts.tempfile.NamedTemporaryFile
    original_fsync = artifacts.os.fsync
    original_replace = artifacts.os.replace

    class RecordingTemporaryFile:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self.name = wrapped.name

        def __enter__(self) -> RecordingTemporaryFile:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

        def write(self, payload: bytes) -> int:
            events.append("write")
            return self._wrapped.write(payload)

        def flush(self) -> None:
            events.append("flush")
            self._wrapped.flush()

        def fileno(self) -> int:
            return self._wrapped.fileno()

        def close(self) -> None:
            events.append("close")
            self._wrapped.close()

        @property
        def closed(self) -> bool:
            return self._wrapped.closed

    def create_temporary_file(*args: Any, **kwargs: Any) -> Any:
        events.append("create")
        assert Path(kwargs["dir"]) == output_dir
        wrapped = original_named_temporary_file(*args, **kwargs)
        recording_file = RecordingTemporaryFile(wrapped)
        temporary_paths.append(Path(recording_file.name))
        wrapped_files.append(recording_file)
        return recording_file

    def record_fsync(file_descriptor: int) -> None:
        events.append("fsync")
        original_fsync(file_descriptor)

    def record_replace(source: Path, final_path: Path) -> None:
        events.append("replace")
        assert temporary_paths == [Path(source)]
        assert Path(final_path) == destination
        assert wrapped_files[0].closed
        assert not destination.exists()
        original_replace(source, final_path)

    monkeypatch.setattr(
        artifacts.tempfile,
        "NamedTemporaryFile",
        create_temporary_file,
    )
    monkeypatch.setattr(artifacts.os, "fsync", record_fsync)
    monkeypatch.setattr(artifacts.os, "replace", record_replace)

    artifacts._write_atomic(b"complete payload\n", destination)

    assert events == ["create", "write", "flush", "fsync", "close", "replace"]
    assert temporary_paths[0].parent == output_dir
    assert destination.read_bytes() == b"complete payload\n"
    assert not temporary_paths[0].exists()


@pytest.mark.parametrize("failed_operation", ("fsync", "replace"))
def test_cleanup_failure_does_not_mask_original_operation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_operation: str,
) -> None:
    tables = _sentinel_tables()
    snapshot = _snapshot_mapping(tables)
    output_dir = tmp_path / f"cleanup-failure-{failed_operation}"
    original_error = OSError(f"original {failed_operation} failure")
    cleanup_error = PermissionError("controlled unlink denial")
    controlled_leftovers: list[Path] = []

    def fail_operation(*args: Any, **kwargs: Any) -> None:
        raise original_error

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        assert missing_ok
        controlled_leftovers.append(path)
        raise cleanup_error

    with monkeypatch.context() as failure_patches:
        failure_patches.setattr(
            artifacts.os,
            failed_operation,
            fail_operation,
        )
        failure_patches.setattr(artifacts.Path, "unlink", fail_cleanup)
        with pytest.raises(OSError) as caught:
            write_eda_tables(tables, output_dir)

    assert caught.value is original_error
    notes = "\n".join(getattr(original_error, "__notes__", ()))
    assert "cleanup failed" in notes.lower()
    assert "PermissionError" in notes
    assert "controlled unlink denial" in notes
    assert controlled_leftovers and len(controlled_leftovers) == 1
    leftover = controlled_leftovers[0]
    assert leftover.parent == output_dir
    assert leftover.exists()
    assert not (output_dir / EXPECTED_FILENAMES[0]).exists()
    _assert_mapping_unchanged(tables, snapshot)

    leftover.unlink()
    assert tuple(output_dir.iterdir()) == ()


def test_every_created_file_is_inside_nested_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "container" / "nested" / "artifacts"

    result = write_eda_tables(_sentinel_tables(), str(output_dir))

    resolved_output = output_dir.resolve()
    created_files = tuple(
        path.resolve()
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert set(created_files) == {path.resolve() for path in result.values()}
    assert all(path.is_relative_to(resolved_output) for path in created_files)


def test_success_and_repeated_writes_do_not_mutate_inputs(tmp_path: Path) -> None:
    tables = _sentinel_tables()
    for position, frame in enumerate(tables.values()):
        frame.index = pd.Index([position + 10], name="original_index")
    snapshot = _snapshot_mapping(tables)

    write_eda_tables(tables, tmp_path / "first")
    _assert_mapping_unchanged(tables, snapshot)
    write_eda_tables(tables, tmp_path / "second")
    _assert_mapping_unchanged(tables, snapshot)


def test_each_call_returns_a_fresh_mapping(tmp_path: Path) -> None:
    tables = _sentinel_tables()
    output_dir = tmp_path / "fresh"

    first = write_eda_tables(tables, output_dir)
    second = write_eda_tables(tables, output_dir)

    assert first is not second
    assert first == second
    first.clear()
    third = write_eda_tables(tables, output_dir)
    assert tuple(third) == EXPECTED_KEYS
    assert len(tuple(output_dir.iterdir())) == 11


def test_production_has_no_static_leakage_or_forbidden_dependencies() -> None:
    source = ARTIFACTS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    referenced_names: set[str] = set()
    string_constants: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Name):
            referenced_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced_names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_constants.append(node.value.lower().replace("\\", "/"))

    assert imported_modules <= {
        "__future__",
        "io",
        "os",
        "pathlib",
        "tempfile",
        "pandas",
    }
    assert not referenced_names & {
        "build_eda_tables",
        "select_eda_populations",
        "build_analytical_dataset",
        "build_canonical_dataset",
        "canonical",
        "target",
        "validation",
        "validation_labels",
        "test_population",
        "test_populations",
        "select_development_rows",
        "select_test_rows",
        "modeling",
        "ArgumentParser",
        "parse_args",
    }
    assert not any(name.startswith("summarize_") for name in referenced_names)
    forbidden_import_parts = {
        "matplotlib",
        "seaborn",
        "pyplot",
    }
    assert not any(
        set(module.split(".")) & forbidden_import_parts
        for module in imported_modules
    )
    assert not imported_modules & {
        "src.analysis.run_eda",
        "src.analysis.summaries",
        "src.data.build_dataset",
    }
    assert _plotting_executable_names(tree) == set()
    assert not any("data/processed" in value for value in string_constants)
    assert not any("reports/eda" in value for value in string_constants)


@pytest.mark.parametrize(
    ("source", "expected_name"),
    (
        ("frame.plot()", "plot"),
        ("frame.plotting()", "plotting"),
        ('plt.savefig("figure.png")', "savefig"),
        ('savefig("figure.png")', "savefig"),
        ("def _helper(frame):\n    return frame.plot()", "plot"),
        ("_private_lambda = lambda frame: frame.plot()", "plot"),
    ),
)
def test_plotting_detector_rejects_executable_ast_references(
    source: str,
    expected_name: str,
) -> None:
    assert _plotting_executable_names(ast.parse(source)) == {expected_name}


def test_plotting_detector_ignores_strings_and_comments() -> None:
    tree = ast.parse(
        'message = "frame.plot() and savefig()"\n'
        "# plt.show() is only a comment\n"
    )
    assert _plotting_executable_names(tree) == set()


def test_module_defines_only_the_approved_public_api_surface() -> None:
    tree = ast.parse(ARTIFACTS_PATH.read_text(encoding="utf-8"))
    bindings = _module_owned_bindings(tree)

    assert {
        name
        for name in bindings["functions"]
        if not name.startswith("_")
    } == {"write_eda_tables"}
    assert not {
        name
        for name in bindings["classes"]
        if not name.startswith("_")
    }
    assert not {
        name
        for name in bindings["assignments"]
        if not name.startswith("_")
    }
    assert not {
        name
        for name in bindings["type_aliases"]
        if not name.startswith("_")
    }
    assert _public_module_owned_names(tree) == {"write_eda_tables"}


@pytest.mark.parametrize(
    ("source", "expected_public_names"),
    (
        ("PUBLIC_VALUE = 1", {"PUBLIC_VALUE"}),
        ("PUBLIC_VALUE: int = 1", {"PUBLIC_VALUE"}),
        ("PUBLIC_VALUE += 1", {"PUBLIC_VALUE"}),
        (
            "PUBLIC_REGISTRY, _private = ({}, None)",
            {"PUBLIC_REGISTRY"},
        ),
        (
            "[PUBLIC_LIST, _private] = [[], None]",
            {"PUBLIC_LIST"},
        ),
        ("*PUBLIC_STARRED, _private = ()", {"PUBLIC_STARRED"}),
        (
            "PUBLIC_LEFT = PUBLIC_RIGHT = 1",
            {"PUBLIC_LEFT", "PUBLIC_RIGHT"},
        ),
        ("class PublicWriter:\n    pass", {"PublicWriter"}),
        ("def public_helper():\n    pass", {"public_helper"}),
    ),
)
def test_module_binding_detector_finds_public_source_probes(
    source: str,
    expected_public_names: set[str],
) -> None:
    tree = ast.parse(source)
    assert _public_module_owned_names(tree) == expected_public_names


def test_module_binding_detector_finds_public_type_alias() -> None:
    if not hasattr(ast, "TypeAlias"):
        pytest.skip("ast.TypeAlias requires Python 3.12 or later")

    tree = ast.parse("type PublicArtifactMap = dict[str, str]")
    bindings = _module_owned_bindings(tree)

    assert bindings["type_aliases"] == {"PublicArtifactMap"}
    assert _public_module_owned_names(tree) == {"PublicArtifactMap"}


def test_module_binding_detector_accepts_private_equivalents() -> None:
    private_sources = [
        "_private_value = 1",
        "_private_annotated: int = 1",
        "_private_augmented += 1",
        "_private_registry, _private = ({}, None)",
        "[_private_list, _private] = [[], None]",
        "*_private_starred, _private = ()",
        "_private_left = _private_right = 1",
        "class _PrivateWriter:\n    pass",
        "def _private_helper():\n    pass",
        "from pathlib import Path\nimport pandas as pd",
    ]
    if hasattr(ast, "TypeAlias"):
        private_sources.append("type _PrivateArtifactMap = dict[str, str]")

    for source in private_sources:
        assert _public_module_owned_names(ast.parse(source)) == set()


def test_import_has_no_filesystem_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    spec = util.spec_from_file_location("isolated_eda_artifacts", ARTIFACTS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    assert isinstance(module, ModuleType)

    spec.loader.exec_module(module)

    assert tuple(tmp_path.rglob("*")) == ()


def test_every_csv_has_stable_lf_line_endings(tmp_path: Path) -> None:
    result = write_eda_tables(_sentinel_tables(), tmp_path / "line-endings")

    for path in result.values():
        payload = path.read_bytes()
        assert b"\r\n" not in payload
        assert b"\r" not in payload
        assert payload.endswith(b"\n")
        assert len(payload) - len(payload.rstrip(b"\n")) == 1


def test_typed_empty_frames_write_headers_only(tmp_path: Path) -> None:
    tables = {
        key: pd.DataFrame(
            {f"{key}_column": pd.Series(dtype="string")}
        )
        for key in EXPECTED_KEYS
    }

    result = write_eda_tables(tables, tmp_path / "empty")

    assert tuple(result) == EXPECTED_KEYS
    for key, path in result.items():
        assert path.read_bytes() == f"{key}_column\n".encode("utf-8")
