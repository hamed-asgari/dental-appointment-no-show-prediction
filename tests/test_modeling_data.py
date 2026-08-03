"""Contract tests for leakage-safe Phase 07 modeling data."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import random
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from src.data import build_dataset as bd
from src.modeling import data as modeling_data


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw"
_DATA_MODULE_PATH = _REPOSITORY_ROOT / "src" / "modeling" / "data.py"
_PACKAGE_INIT_PATH = _REPOSITORY_ROOT / "src" / "modeling" / "__init__.py"

_EXPECTED_CANONICAL_COLUMNS = (
    "appointment_id",
    "patient_id",
    "dentist_id",
    "prediction_time",
    "target",
    "split",
    "planned_duration_min",
    "visit_type",
    "booking_channel",
    "booking_lead_time_hours",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
    "development_fit_eligible",
    "pretest_fit_eligible",
)
_EXPECTED_DTYPES = {
    "appointment_id": "int64",
    "patient_id": "int64",
    "dentist_id": "int64",
    "prediction_time": "datetime64[ns]",
    "target": "int8",
    "split": "string",
    "planned_duration_min": "int16",
    "visit_type": "string",
    "booking_channel": "string",
    "booking_lead_time_hours": "float64",
    "scheduled_weekday": "int8",
    "scheduled_hour": "int8",
    "scheduled_month": "int8",
    "approximate_age_at_prediction": "int16",
    "patient_registration_tenure_days": "int32",
    "dentist_tenure_days": "int32",
    "development_fit_eligible": "bool",
    "pretest_fit_eligible": "bool",
}
_EXPECTED_NUMERIC_FEATURES = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
_EXPECTED_CATEGORICAL_FEATURES = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
_EXPECTED_FEATURES = (
    "planned_duration_min",
    "visit_type",
    "booking_channel",
    "booking_lead_time_hours",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
_EXPECTED_RETURN_KEYS = (
    "development_features",
    "development_target",
    "validation_features",
    "validation_target",
)
_FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "appointment_id",
        "patient_id",
        "dentist_id",
        "prediction_time",
        "target",
        "split",
        "development_fit_eligible",
        "pretest_fit_eligible",
        "status",
        "status_updated_at",
    }
)
_EXPECTED_DEVELOPMENT_ROWS = 3_670
_EXPECTED_DEVELOPMENT_POSITIVES = 432
_EXPECTED_DEVELOPMENT_NEGATIVES = 3_238
_EXPECTED_VALIDATION_ROWS = 1_541
_EXPECTED_VALIDATION_POSITIVES = 192
_EXPECTED_VALIDATION_NEGATIVES = 1_349
_EXPECTED_VALIDATION_START = pd.Timestamp("2025-03-01 00:00:00")
_EXPECTED_TEST_START = pd.Timestamp("2025-08-01 00:00:00")
_EXPECTED_PREDICTION_HORIZON_HOURS = 24
_FORBIDDEN_PRODUCTION_COUNTS = frozenset(
    {3_682, 3_670, 432, 3_238, 1_541, 192, 1_349, 5_223, 626, 4_597}
)


def _small_canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "appointment_id": pd.Series(
                [101, 102, 103, 104, 105, 106], dtype="int64"
            ),
            "patient_id": pd.Series(
                [201, 202, 203, 204, 205, 206], dtype="int64"
            ),
            "dentist_id": pd.Series([1, 2, 1, 2, 1, 2], dtype="int64"),
            "prediction_time": pd.Series(
                pd.to_datetime(
                    [
                        "2025-01-01 08:00:00",
                        "2025-01-02 09:00:00",
                        "2025-02-28 10:00:00",
                        "2025-03-01 08:00:00",
                        "2025-03-02 09:00:00",
                        "2025-08-01 08:00:00",
                    ],
                    format="%Y-%m-%d %H:%M:%S",
                ),
                dtype="datetime64[ns]",
            ),
            "target": pd.Series([0, 1, 1, 0, 1, 1], dtype="int8"),
            "split": pd.Series(
                [
                    "train",
                    "train",
                    "train",
                    "validation",
                    "validation",
                    "test",
                ],
                dtype="string",
            ),
            "planned_duration_min": pd.Series(
                [30, 45, 60, 30, 45, 60], dtype="int16"
            ),
            "visit_type": pd.Series(
                [
                    "checkup",
                    "filling",
                    "cleaning",
                    "checkup",
                    "filling",
                    "cleaning",
                ],
                dtype="string",
            ),
            "booking_channel": pd.Series(
                ["phone", "web", "desk", "phone", "web", "desk"],
                dtype="string",
            ),
            "booking_lead_time_hours": pd.Series(
                [48.0, 72.0, 96.0, 48.0, 72.0, 96.0], dtype="float64"
            ),
            "scheduled_weekday": pd.Series(
                [0, 1, 2, 3, 4, 5], dtype="int8"
            ),
            "scheduled_hour": pd.Series(
                [8, 9, 10, 11, 12, 13], dtype="int8"
            ),
            "scheduled_month": pd.Series(
                [1, 1, 2, 3, 3, 8], dtype="int8"
            ),
            "approximate_age_at_prediction": pd.Series(
                [25, 30, 35, 40, 45, 50], dtype="int16"
            ),
            "patient_registration_tenure_days": pd.Series(
                [100, 200, 300, 400, 500, 600], dtype="int32"
            ),
            "dentist_tenure_days": pd.Series(
                [500, 600, 700, 800, 900, 1_000], dtype="int32"
            ),
            "development_fit_eligible": pd.Series(
                [True, True, False, False, False, False], dtype="bool"
            ),
            "pretest_fit_eligible": pd.Series(
                [True, True, False, True, True, False], dtype="bool"
            ),
        },
        columns=_EXPECTED_CANONICAL_COLUMNS,
    )


@pytest.fixture(scope="session")
def _authentic_canonical() -> pd.DataFrame:
    bd.validate_raw_hashes(_RAW_DIR)
    raw_tables = bd.load_raw_data(_RAW_DIR)
    return bd.build_analytical_dataset(raw_tables)


def _bound_target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _bound_target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_bound_target_names(element))
        return names
    return set()


def _match_pattern_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()
    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            names.add(pattern.name)
        if pattern.pattern is not None:
            names.update(_match_pattern_names(pattern.pattern))
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            names.add(pattern.name)
    elif isinstance(pattern, ast.MatchMapping):
        if pattern.rest is not None:
            names.add(pattern.rest)
        for child in pattern.patterns:
            names.update(_match_pattern_names(child))
    elif isinstance(pattern, ast.MatchClass):
        for child in (*pattern.patterns, *pattern.kwd_patterns):
            names.update(_match_pattern_names(child))
    elif isinstance(pattern, ast.MatchSequence):
        for child in pattern.patterns:
            names.update(_match_pattern_names(child))
    elif isinstance(pattern, ast.MatchOr):
        for child in pattern.patterns:
            names.update(_match_pattern_names(child))
    return names


def _module_scope_statements(
    statements: list[ast.stmt],
) -> Iterator[ast.stmt]:
    for statement in statements:
        yield statement
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            yield from _module_scope_statements(statement.body)
            yield from _module_scope_statements(statement.orelse)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            yield from _module_scope_statements(statement.body)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            yield from _module_scope_statements(statement.body)
            for handler in statement.handlers:
                yield from _module_scope_statements(handler.body)
            yield from _module_scope_statements(statement.orelse)
            yield from _module_scope_statements(statement.finalbody)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                yield from _module_scope_statements(case.body)


def _module_bindings(source: str) -> set[str]:
    bindings: set[str] = set()
    tree = ast.parse(source)
    for statement in _module_scope_statements(tree.body):
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            bindings.add(statement.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                bindings.update(_bound_target_names(target))
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            bindings.update(_bound_target_names(statement.target))
        elif isinstance(statement, ast.TypeAlias):
            bindings.update(_bound_target_names(statement.name))
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            bindings.update(_bound_target_names(statement.target))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                if item.optional_vars is not None:
                    bindings.update(_bound_target_names(item.optional_vars))
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            for handler in statement.handlers:
                if handler.name is not None:
                    bindings.add(handler.name)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                bindings.update(_match_pattern_names(case.pattern))
    return bindings


def _assert_outputs_equal(
    left: dict[str, pd.DataFrame | pd.Series],
    right: dict[str, pd.DataFrame | pd.Series],
) -> None:
    pd.testing.assert_frame_equal(
        left["development_features"], right["development_features"]
    )
    pd.testing.assert_series_equal(
        left["development_target"], right["development_target"]
    )
    pd.testing.assert_frame_equal(
        left["validation_features"], right["validation_features"]
    )
    pd.testing.assert_series_equal(
        left["validation_target"], right["validation_target"]
    )


def _rotate_values(
    canonical: pd.DataFrame,
    mask: pd.Series,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        values = canonical.loc[mask, column].to_numpy(copy=True)
        canonical.loc[mask, column] = np.roll(values, 1)


def _numpy_random_state() -> tuple[str, np.ndarray, int, int, float]:
    state = np.random.get_state()
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _process_state_snapshot(root: Path) -> dict[str, object]:
    return {
        "cwd": Path.cwd(),
        "environment": os.environ.copy(),
        "warning_filters": list(warnings.filters),
        "python_random": random.getstate(),
        "numpy_random": _numpy_random_state(),
        "paths": {path.relative_to(root) for path in root.rglob("*")},
    }


def _assert_process_state_equal(
    before: dict[str, object],
    after: dict[str, object],
) -> None:
    assert after["cwd"] == before["cwd"]
    assert after["environment"] == before["environment"]
    assert after["warning_filters"] == before["warning_filters"]
    assert after["python_random"] == before["python_random"]
    before_numpy = before["numpy_random"]
    after_numpy = after["numpy_random"]
    assert isinstance(before_numpy, tuple)
    assert isinstance(after_numpy, tuple)
    assert before_numpy[0] == after_numpy[0]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert after["paths"] == before["paths"]


def test_exact_public_function_signature_and_annotations() -> None:
    function = modeling_data.build_development_modeling_data
    signature = inspect.signature(function)
    assert tuple(signature.parameters) == ("canonical",)
    parameter = signature.parameters["canonical"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    assert signature.return_annotation == "dict[str, pd.DataFrame | pd.Series]"
    assert get_type_hints(function) == {
        "canonical": pd.DataFrame,
        "return": dict[str, pd.DataFrame | pd.Series],
    }


def test_exact_all_and_controlled_wildcard_export() -> None:
    assert modeling_data.__all__ == (
        "NUMERIC_FEATURE_COLUMNS",
        "CATEGORICAL_FEATURE_COLUMNS",
        "build_development_modeling_data",
    )
    namespace: dict[str, object] = {}
    exec("from src.modeling.data import *", namespace)
    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == set(modeling_data.__all__)


def test_module_binding_detector_covers_all_python_312_binding_forms() -> None:
    source = """
def public_function():
    pass
async def public_async_function():
    pass
class PublicClass:
    pass
plain = 1
annotated: int = 2
incremented += 1
left = right = 3
(first, *middle, last) = values
[list_first, list_second] = values
type PublicAlias = int
"""
    assert _module_bindings(source) == {
        "public_function",
        "public_async_function",
        "PublicClass",
        "plain",
        "annotated",
        "incremented",
        "left",
        "right",
        "first",
        "middle",
        "last",
        "list_first",
        "list_second",
        "PublicAlias",
    }


def test_module_binding_detector_recurses_through_control_flow() -> None:
    source = """
if condition:
    nested_if = 1
try:
    nested_try = 2
except Exception as public_error:
    nested_except = 3
else:
    nested_else = 4
finally:
    nested_finally = 5
for public_item in values:
    nested_for = 6
with manager() as public_context:
    nested_with = 7
match subject:
    case {"key": public_capture, **public_rest}:
        nested_match = 8
"""
    assert {
        "nested_if",
        "nested_try",
        "public_error",
        "nested_except",
        "nested_else",
        "nested_finally",
        "public_item",
        "nested_for",
        "public_context",
        "nested_with",
        "public_capture",
        "public_rest",
        "nested_match",
    }.issubset(_module_bindings(source))


def test_module_binding_detector_does_not_descend_owned_scopes() -> None:
    source = """
def public_function():
    hidden_function_local = 1
async def public_async_function():
    hidden_async_local = 2
class PublicClass:
    hidden_class_local = 3
module_comprehension = [hidden_comprehension for hidden_comprehension in values]
"""
    bindings = _module_bindings(source)
    assert {
        "public_function",
        "public_async_function",
        "PublicClass",
        "module_comprehension",
    }.issubset(bindings)
    assert {
        "hidden_function_local",
        "hidden_async_local",
        "hidden_class_local",
        "hidden_comprehension",
    }.isdisjoint(bindings)


def test_production_module_has_only_approved_public_bindings() -> None:
    source = _DATA_MODULE_PATH.read_text(encoding="utf-8")
    bindings = _module_bindings(source)
    public_bindings = {
        name for name in bindings if not name.startswith("_")
    }
    assert public_bindings == {
        "NUMERIC_FEATURE_COLUMNS",
        "CATEGORICAL_FEATURE_COLUMNS",
        "build_development_modeling_data",
    }


def test_modeling_init_contains_only_a_package_docstring() -> None:
    source = _PACKAGE_INIT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert len(tree.body) == 1
    statement = tree.body[0]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Constant)
    assert isinstance(statement.value.value, str)
    assert ast.get_docstring(tree) == "Leakage-safe modeling data contracts."


def test_feature_group_contracts_are_exact() -> None:
    assert modeling_data.NUMERIC_FEATURE_COLUMNS == _EXPECTED_NUMERIC_FEATURES
    assert (
        modeling_data.CATEGORICAL_FEATURE_COLUMNS
        == _EXPECTED_CATEGORICAL_FEATURES
    )
    assert set(_EXPECTED_FEATURES).isdisjoint(_FORBIDDEN_FEATURE_COLUMNS)


def test_small_valid_fixture_owns_exact_schema_and_dtypes() -> None:
    canonical = _small_canonical()
    assert tuple(canonical.columns) == _EXPECTED_CANONICAL_COLUMNS
    assert {
        column: str(canonical[column].dtype)
        for column in _EXPECTED_CANONICAL_COLUMNS
    } == _EXPECTED_DTYPES


def test_valid_small_fixture_is_not_rejected_by_authentic_counts() -> None:
    result = modeling_data.build_development_modeling_data(_small_canonical())
    assert len(result["development_features"]) == 2
    assert len(result["validation_features"]) == 2


def test_exact_return_contract_and_combined_feature_order() -> None:
    canonical = _small_canonical()
    result = modeling_data.build_development_modeling_data(canonical)
    assert type(result) is dict
    assert tuple(result) == _EXPECTED_RETURN_KEYS

    development_features = result["development_features"]
    development_target = result["development_target"]
    validation_features = result["validation_features"]
    validation_target = result["validation_target"]

    assert type(development_features) is pd.DataFrame
    assert type(validation_features) is pd.DataFrame
    assert type(development_target) is pd.Series
    assert type(validation_target) is pd.Series
    assert tuple(development_features.columns) == _EXPECTED_FEATURES
    assert tuple(validation_features.columns) == _EXPECTED_FEATURES
    assert development_target.name == "target"
    assert validation_target.name == "target"
    assert str(development_target.dtype) == "int8"
    assert str(validation_target.dtype) == "int8"
    assert development_features.index.equals(development_target.index)
    assert validation_features.index.equals(validation_target.index)


def test_population_masks_preserve_exact_source_order_and_indexes() -> None:
    canonical = _small_canonical()
    result = modeling_data.build_development_modeling_data(canonical)
    development_mask = canonical["split"].eq("train") & canonical[
        "development_fit_eligible"
    ]
    validation_mask = canonical["split"].eq("validation")

    expected_development = canonical.loc[
        development_mask, _EXPECTED_FEATURES
    ]
    expected_validation = canonical.loc[validation_mask, _EXPECTED_FEATURES]
    pd.testing.assert_frame_equal(
        result["development_features"], expected_development
    )
    pd.testing.assert_series_equal(
        result["development_target"], canonical.loc[development_mask, "target"]
    )
    pd.testing.assert_frame_equal(
        result["validation_features"], expected_validation
    )
    pd.testing.assert_series_equal(
        result["validation_target"], canonical.loc[validation_mask, "target"]
    )


def test_authentic_population_counts_and_alignment(
    _authentic_canonical: pd.DataFrame,
) -> None:
    result = modeling_data.build_development_modeling_data(
        _authentic_canonical
    )
    development_features = result["development_features"]
    development_target = result["development_target"]
    validation_features = result["validation_features"]
    validation_target = result["validation_target"]

    assert len(development_features) == _EXPECTED_DEVELOPMENT_ROWS
    assert int(development_target.sum()) == _EXPECTED_DEVELOPMENT_POSITIVES
    assert int(development_target.eq(0).sum()) == (
        _EXPECTED_DEVELOPMENT_NEGATIVES
    )
    assert len(validation_features) == _EXPECTED_VALIDATION_ROWS
    assert int(validation_target.sum()) == _EXPECTED_VALIDATION_POSITIVES
    assert int(validation_target.eq(0).sum()) == _EXPECTED_VALIDATION_NEGATIVES
    assert tuple(development_features.columns) == _EXPECTED_FEATURES
    assert tuple(validation_features.columns) == _EXPECTED_FEATURES
    assert development_features.index.equals(development_target.index)
    assert validation_features.index.equals(validation_target.index)


def test_outputs_never_contain_identifiers_controls_test_or_pretest() -> None:
    canonical = _small_canonical()
    result = modeling_data.build_development_modeling_data(canonical)
    assert tuple(result) == _EXPECTED_RETURN_KEYS
    for key in ("development_features", "validation_features"):
        assert set(result[key].columns).isdisjoint(_FORBIDDEN_FEATURE_COLUMNS)
    returned_indexes = result["development_features"].index.union(
        result["validation_features"].index
    )
    test_indexes = canonical.index[canonical["split"].eq("test")]
    immature_indexes = canonical.index[
        canonical["split"].eq("train")
        & ~canonical["development_fit_eligible"]
    ]
    assert returned_indexes.intersection(test_indexes).empty
    assert result["development_features"].index.intersection(
        immature_indexes
    ).empty


def test_authoritative_structure_validator_is_reused_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _small_canonical()
    original = modeling_data._validate_canonical_structure
    calls: list[pd.DataFrame] = []

    def validate(candidate: pd.DataFrame) -> None:
        calls.append(candidate)
        original(candidate)

    monkeypatch.setattr(
        modeling_data, "_validate_canonical_structure", validate
    )
    modeling_data.build_development_modeling_data(canonical)
    assert calls == [canonical]
    assert calls[0] is canonical


@pytest.mark.parametrize("invalid", [None, object(), [], {}])
def test_non_dataframe_input_is_rejected(invalid: object) -> None:
    with pytest.raises(TypeError, match="exact pandas DataFrame"):
        modeling_data.build_development_modeling_data(invalid)


def test_dataframe_subclass_is_rejected() -> None:
    class _DataFrameSubclass(pd.DataFrame):
        pass

    invalid = _DataFrameSubclass(_small_canonical())
    with pytest.raises(TypeError, match="exact pandas DataFrame"):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize("defect", ["missing", "extra", "reordered"])
def test_exact_canonical_columns_and_order_are_enforced(defect: str) -> None:
    invalid = _small_canonical()
    if defect == "missing":
        invalid = invalid.drop(columns="visit_type")
    elif defect == "extra":
        invalid["unexpected"] = 1
    else:
        columns = list(_EXPECTED_CANONICAL_COLUMNS)
        columns[0], columns[1] = columns[1], columns[0]
        invalid = invalid.loc[:, columns]
    with pytest.raises(ValueError, match="Canonical columns or order"):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize(
    ("column", "wrong_dtype"),
    [
        ("appointment_id", "int32"),
        ("patient_id", "int32"),
        ("dentist_id", "int32"),
        ("prediction_time", "string"),
        ("target", "int16"),
        ("split", "object"),
        ("planned_duration_min", "int32"),
        ("visit_type", "object"),
        ("booking_channel", "object"),
        ("booking_lead_time_hours", "float32"),
        ("scheduled_weekday", "int16"),
        ("scheduled_hour", "int16"),
        ("scheduled_month", "int16"),
        ("approximate_age_at_prediction", "int32"),
        ("patient_registration_tenure_days", "int64"),
        ("dentist_tenure_days", "int64"),
        ("development_fit_eligible", "int8"),
        ("pretest_fit_eligible", "int8"),
    ],
)
def test_dtype_drift_is_rejected_for_every_canonical_role(
    column: str,
    wrong_dtype: str,
) -> None:
    invalid = _small_canonical()
    invalid[column] = invalid[column].astype(wrong_dtype)
    with pytest.raises(ValueError, match=rf"{column} dtype must be"):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize("target_case", ["boolean", "nonnumeric", "null"])
def test_boolean_nonnumeric_and_null_targets_are_rejected(
    target_case: str,
) -> None:
    invalid = _small_canonical()
    if target_case == "boolean":
        invalid["target"] = invalid["target"].astype(bool)
    elif target_case == "nonnumeric":
        invalid["target"] = invalid["target"].astype("string")
    else:
        invalid["target"] = invalid["target"].astype("Int8")
        invalid.loc[0, "target"] = pd.NA
    with pytest.raises(ValueError, match="target dtype must be int8"):
        modeling_data.build_development_modeling_data(invalid)


def test_target_outside_binary_domain_is_rejected() -> None:
    invalid = _small_canonical()
    invalid.loc[0, "target"] = 2
    with pytest.raises(ValueError, match="target values must be exactly"):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize("split_case", ["invalid", "null"])
def test_invalid_or_null_split_is_rejected(split_case: str) -> None:
    invalid = _small_canonical()
    invalid.loc[5, "split"] = (
        "future" if split_case == "invalid" else pd.NA
    )
    message = "split values" if split_case == "invalid" else "null required"
    with pytest.raises(ValueError, match=message):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize("failure_case", ["dtype", "invalid_split"])
def test_dtype_and_invalid_split_failures_preserve_exact_input(
    failure_case: str,
) -> None:
    invalid = _small_canonical()
    if failure_case == "dtype":
        invalid["appointment_id"] = invalid["appointment_id"].astype("int32")
    else:
        invalid.loc[5, "split"] = "future"
    invalid.attrs["failure_case"] = failure_case
    before = invalid.copy(deep=True)
    with pytest.raises(ValueError):
        modeling_data.build_development_modeling_data(invalid)
    pd.testing.assert_frame_equal(invalid, before)
    assert invalid.attrs == before.attrs


def test_duplicate_appointment_identifier_is_rejected() -> None:
    invalid = _small_canonical()
    invalid.loc[1, "appointment_id"] = invalid.loc[0, "appointment_id"]
    with pytest.raises(ValueError, match="appointment_id must be unique"):
        modeling_data.build_development_modeling_data(invalid)


def test_equivalent_int64_index_is_rejected() -> None:
    invalid = _small_canonical()
    invalid.index = pd.Index(np.arange(len(invalid)), dtype="int64")
    assert type(invalid.index) is not pd.RangeIndex
    with pytest.raises(ValueError, match="zero-based RangeIndex"):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize(
    "invalid_index",
    [
        pd.RangeIndex(start=1, stop=7, step=1),
        pd.RangeIndex(start=0, stop=12, step=2),
        pd.RangeIndex(start=-1, stop=5, step=1),
    ],
    ids=("nonzero-start", "nonunit-step-and-stop", "negative-start"),
)
def test_invalid_range_index_metadata_is_rejected(
    invalid_index: pd.RangeIndex,
) -> None:
    invalid = _small_canonical()
    invalid.index = invalid_index
    with pytest.raises(ValueError, match="zero-based RangeIndex"):
        modeling_data.build_development_modeling_data(invalid)


def test_exact_range_index_metadata_is_accepted() -> None:
    canonical = _small_canonical()
    assert type(canonical.index) is pd.RangeIndex
    assert canonical.index.start == 0
    assert canonical.index.step == 1
    assert canonical.index.stop == len(canonical)
    modeling_data.build_development_modeling_data(canonical)


def test_reordered_canonical_rows_are_rejected() -> None:
    canonical = _small_canonical()
    invalid = canonical.iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordered by prediction_time"):
        modeling_data.build_development_modeling_data(invalid)


def test_phase05_temporal_constants_are_reused_exactly() -> None:
    assert bd.VALIDATION_START == _EXPECTED_VALIDATION_START
    assert bd.TEST_START == _EXPECTED_TEST_START
    assert modeling_data._VALIDATION_START is bd.VALIDATION_START
    assert modeling_data._TEST_START is bd.TEST_START


@pytest.mark.parametrize(
    "defect",
    [
        "train_at_validation_boundary",
        "validation_before_boundary",
        "validation_at_test_boundary",
        "test_before_boundary",
        "swap_train_validation",
        "swap_validation_test",
    ],
)
def test_temporal_split_inconsistency_is_rejected_without_mutation(
    defect: str,
) -> None:
    invalid = _small_canonical()
    if defect == "train_at_validation_boundary":
        invalid.loc[2, "prediction_time"] = _EXPECTED_VALIDATION_START
    elif defect == "validation_before_boundary":
        invalid.loc[3, "prediction_time"] = (
            _EXPECTED_VALIDATION_START - pd.Timedelta(1, unit="ns")
        )
    elif defect == "validation_at_test_boundary":
        invalid.loc[4, "prediction_time"] = _EXPECTED_TEST_START
    elif defect == "test_before_boundary":
        invalid.loc[5, "prediction_time"] = (
            _EXPECTED_TEST_START - pd.Timedelta(1, unit="ns")
        )
    elif defect == "swap_train_validation":
        invalid.loc[2, "split"] = "validation"
        invalid.loc[3, "split"] = "train"
    else:
        invalid.loc[4, "pretest_fit_eligible"] = False
        invalid.loc[4, "split"] = "test"
        invalid.loc[5, "split"] = "validation"
    before = invalid.copy(deep=True)
    with pytest.raises(ValueError, match="prediction-time boundaries"):
        modeling_data.build_development_modeling_data(invalid)
    pd.testing.assert_frame_equal(invalid, before)


def test_exact_half_open_temporal_boundaries_are_accepted() -> None:
    canonical = _small_canonical()
    canonical.loc[2, "prediction_time"] = (
        _EXPECTED_VALIDATION_START - pd.Timedelta(1, unit="ns")
    )
    canonical.loc[3, "prediction_time"] = _EXPECTED_VALIDATION_START
    canonical.loc[4, "prediction_time"] = (
        _EXPECTED_TEST_START - pd.Timedelta(1, unit="ns")
    )
    canonical.loc[5, "prediction_time"] = _EXPECTED_TEST_START
    modeling_data.build_development_modeling_data(canonical)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_numerical_predictor_is_rejected(
    invalid_value: float,
) -> None:
    invalid = _small_canonical()
    invalid.loc[0, "booking_lead_time_hours"] = invalid_value
    message = "null required" if np.isnan(invalid_value) else "finite values"
    with pytest.raises(ValueError, match=message):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("planned_duration_min", 0),
        ("booking_lead_time_hours", 23.0),
        ("scheduled_weekday", -1),
        ("scheduled_weekday", 7),
        ("scheduled_hour", -1),
        ("scheduled_hour", 24),
        ("scheduled_month", 0),
        ("scheduled_month", 13),
        ("approximate_age_at_prediction", -1),
        ("patient_registration_tenure_days", -1),
        ("dentist_tenure_days", -1),
    ],
)
def test_phase05_predictor_domain_failures_are_rejected_without_mutation(
    column: str,
    invalid_value: int | float,
) -> None:
    invalid = _small_canonical()
    invalid.loc[0, column] = invalid_value
    before = invalid.copy(deep=True)
    with pytest.raises(ValueError, match=rf"{column}.*valid range"):
        modeling_data.build_development_modeling_data(invalid)
    pd.testing.assert_frame_equal(invalid, before)


@pytest.mark.parametrize(
    ("column", "valid_value"),
    [
        ("planned_duration_min", 1),
        ("planned_duration_min", np.iinfo(np.int16).max),
        (
            "booking_lead_time_hours",
            float(_EXPECTED_PREDICTION_HORIZON_HOURS),
        ),
        ("booking_lead_time_hours", 1_000_000.0),
        ("scheduled_weekday", 0),
        ("scheduled_weekday", 6),
        ("scheduled_hour", 0),
        ("scheduled_hour", 23),
        ("scheduled_month", 1),
        ("scheduled_month", 12),
        ("approximate_age_at_prediction", 0),
        ("approximate_age_at_prediction", np.iinfo(np.int16).max),
        ("patient_registration_tenure_days", 0),
        (
            "patient_registration_tenure_days",
            np.iinfo(np.int32).max,
        ),
        ("dentist_tenure_days", 0),
        ("dentist_tenure_days", np.iinfo(np.int32).max),
    ],
)
def test_exact_and_storage_predictor_domain_boundaries_are_accepted(
    column: str,
    valid_value: int | float,
) -> None:
    canonical = _small_canonical()
    canonical.loc[0, column] = valid_value
    modeling_data.build_development_modeling_data(canonical)


def test_phase05_defines_no_closed_categorical_feature_domains() -> None:
    canonical = _small_canonical()
    canonical.loc[0, "visit_type"] = "new_valid_visit_level"
    canonical.loc[0, "booking_channel"] = "new_valid_booking_level"
    modeling_data.build_development_modeling_data(canonical)


def test_null_categorical_predictor_is_rejected() -> None:
    invalid = _small_canonical()
    invalid.loc[0, "visit_type"] = pd.NA
    with pytest.raises(ValueError, match="null required"):
        modeling_data.build_development_modeling_data(invalid)


@pytest.mark.parametrize("split_name", ["validation", "test"])
def test_development_eligibility_outside_train_is_rejected(
    split_name: str,
) -> None:
    invalid = _small_canonical()
    index = invalid.index[invalid["split"].eq(split_name)][0]
    invalid.loc[index, "development_fit_eligible"] = True
    invalid.loc[index, "pretest_fit_eligible"] = True
    with pytest.raises(ValueError, match="Development-fit eligibility"):
        modeling_data.build_development_modeling_data(invalid)


def test_test_row_marked_pretest_eligible_is_rejected() -> None:
    invalid = _small_canonical()
    index = invalid.index[invalid["split"].eq("test")][0]
    invalid.loc[index, "pretest_fit_eligible"] = True
    with pytest.raises(ValueError, match="Pretest-fit eligibility"):
        modeling_data.build_development_modeling_data(invalid)


def test_development_eligibility_requires_pretest_eligibility() -> None:
    invalid = _small_canonical()
    index = invalid.index[invalid["development_fit_eligible"]][0]
    invalid.loc[index, "pretest_fit_eligible"] = False
    with pytest.raises(ValueError, match="requires pretest-fit"):
        modeling_data.build_development_modeling_data(invalid)


def test_empty_development_population_is_rejected() -> None:
    invalid = _small_canonical()
    invalid["development_fit_eligible"] = False
    with pytest.raises(ValueError, match="Development population"):
        modeling_data.build_development_modeling_data(invalid)


def test_empty_validation_population_is_rejected() -> None:
    invalid = _small_canonical()
    validation = invalid["split"].eq("validation")
    invalid = invalid.loc[~validation].reset_index(drop=True)
    with pytest.raises(ValueError, match="Validation population"):
        modeling_data.build_development_modeling_data(invalid)


def test_single_class_development_target_is_rejected() -> None:
    invalid = _small_canonical()
    development = invalid["development_fit_eligible"]
    invalid.loc[development, "target"] = 0
    with pytest.raises(ValueError, match="Development target"):
        modeling_data.build_development_modeling_data(invalid)


def test_single_class_validation_is_deferred_to_metrics_contract() -> None:
    canonical = _small_canonical()
    validation = canonical["split"].eq("validation")
    canonical.loc[validation, "target"] = 0
    result = modeling_data.build_development_modeling_data(canonical)
    assert set(result["validation_target"].unique()) == {0}


def test_input_is_not_mutated_on_success() -> None:
    canonical = _small_canonical()
    canonical.attrs["contract_owner"] = "test"
    before = canonical.copy(deep=True)
    identity = id(canonical)
    modeling_data.build_development_modeling_data(canonical)
    assert id(canonical) == identity
    pd.testing.assert_frame_equal(canonical, before)
    assert canonical.attrs == before.attrs


@pytest.mark.parametrize(
    "defect",
    [
        "extra_column",
        "invalid_target",
        "duplicate_id",
        "non_range_index",
        "reordered",
        "infinite",
        "null_category",
        "validation_development",
        "test_pretest",
        "development_without_pretest",
        "empty_development",
        "empty_validation",
        "single_class_development",
    ],
)
def test_input_is_not_mutated_on_validation_failure(defect: str) -> None:
    invalid = _small_canonical()
    if defect == "extra_column":
        invalid["unexpected"] = 1
    elif defect == "invalid_target":
        invalid.loc[0, "target"] = 2
    elif defect == "duplicate_id":
        invalid.loc[1, "appointment_id"] = invalid.loc[0, "appointment_id"]
    elif defect == "non_range_index":
        invalid.index = pd.Index(range(10, 10 + len(invalid)), dtype="int64")
    elif defect == "reordered":
        invalid = invalid.iloc[[1, 0, 2, 3, 4, 5]].reset_index(drop=True)
    elif defect == "infinite":
        invalid.loc[0, "booking_lead_time_hours"] = np.inf
    elif defect == "null_category":
        invalid.loc[0, "visit_type"] = pd.NA
    elif defect == "validation_development":
        invalid.loc[3, "development_fit_eligible"] = True
    elif defect == "test_pretest":
        invalid.loc[5, "pretest_fit_eligible"] = True
    elif defect == "development_without_pretest":
        invalid.loc[0, "pretest_fit_eligible"] = False
    elif defect == "empty_development":
        invalid["development_fit_eligible"] = False
    elif defect == "empty_validation":
        invalid.loc[invalid["split"].eq("validation"), "split"] = "train"
    else:
        invalid.loc[invalid["development_fit_eligible"], "target"] = 0
    invalid.attrs["failure_case"] = defect
    before = invalid.copy(deep=True)
    with pytest.raises((TypeError, ValueError)):
        modeling_data.build_development_modeling_data(invalid)
    pd.testing.assert_frame_equal(invalid, before)
    assert invalid.attrs == before.attrs


def test_every_call_returns_fresh_mutation_isolated_objects() -> None:
    canonical = _small_canonical()
    canonical_before = canonical.copy(deep=True)
    first = modeling_data.build_development_modeling_data(canonical)
    second = modeling_data.build_development_modeling_data(canonical)
    second_before = {
        key: value.copy(deep=True) for key, value in second.items()
    }

    assert first is not second
    for key in _EXPECTED_RETURN_KEYS:
        assert first[key] is not second[key]

    first["development_features"].iloc[0, 0] += 1
    first["validation_features"].iloc[0, 0] += 1
    first["development_target"].iloc[0] = (
        1 - first["development_target"].iloc[0]
    )
    first["validation_target"].iloc[0] = (
        1 - first["validation_target"].iloc[0]
    )

    for key in ("development_features", "validation_features"):
        pd.testing.assert_frame_equal(second[key], second_before[key])
    for key in ("development_target", "validation_target"):
        pd.testing.assert_series_equal(second[key], second_before[key])
    pd.testing.assert_frame_equal(canonical, canonical_before)

    later = modeling_data.build_development_modeling_data(canonical)
    _assert_outputs_equal(second, later)


def test_numeric_arrays_have_practical_memory_isolation() -> None:
    canonical = _small_canonical()
    first = modeling_data.build_development_modeling_data(canonical)
    second = modeling_data.build_development_modeling_data(canonical)
    numeric_columns = (
        "planned_duration_min",
        "booking_lead_time_hours",
        "scheduled_weekday",
        "scheduled_hour",
        "scheduled_month",
        "approximate_age_at_prediction",
        "patient_registration_tenure_days",
        "dentist_tenure_days",
    )
    for population in ("development_features", "validation_features"):
        for column in numeric_columns:
            output_values = first[population][column].to_numpy(copy=False)
            source_values = canonical[column].to_numpy(copy=False)
            repeated_values = second[population][column].to_numpy(copy=False)
            assert not np.shares_memory(output_values, source_values)
            assert not np.shares_memory(output_values, repeated_values)

    for population in ("development_target", "validation_target"):
        output_values = first[population].to_numpy(copy=False)
        source_values = canonical["target"].to_numpy(copy=False)
        repeated_values = second[population].to_numpy(copy=False)
        assert not np.shares_memory(output_values, source_values)
        assert not np.shares_memory(output_values, repeated_values)

    for key in _EXPECTED_RETURN_KEYS:
        assert first[key] is not second[key]
    assert first["development_features"] is not first["validation_features"]
    assert first["development_target"] is not first["validation_target"]


def test_valid_test_content_poisoning_is_invariant(
    _authentic_canonical: pd.DataFrame,
) -> None:
    baseline = modeling_data.build_development_modeling_data(
        _authentic_canonical
    )
    poisoned = _authentic_canonical.copy(deep=True)
    test_rows = poisoned["split"].eq("test")
    _rotate_values(poisoned, test_rows, _EXPECTED_FEATURES + ("target",))
    assert {
        column: str(poisoned[column].dtype)
        for column in _EXPECTED_CANONICAL_COLUMNS
    } == _EXPECTED_DTYPES
    changed = modeling_data.build_development_modeling_data(poisoned)
    _assert_outputs_equal(baseline, changed)


def test_valid_immature_train_content_poisoning_is_invariant(
    _authentic_canonical: pd.DataFrame,
) -> None:
    baseline = modeling_data.build_development_modeling_data(
        _authentic_canonical
    )
    poisoned = _authentic_canonical.copy(deep=True)
    immature_train = poisoned["split"].eq("train") & ~poisoned[
        "development_fit_eligible"
    ]
    assert int(immature_train.sum()) > 1
    _rotate_values(
        poisoned,
        immature_train,
        _EXPECTED_FEATURES + ("target",),
    )
    assert {
        column: str(poisoned[column].dtype)
        for column in _EXPECTED_CANONICAL_COLUMNS
    } == _EXPECTED_DTYPES
    changed = modeling_data.build_development_modeling_data(poisoned)
    _assert_outputs_equal(baseline, changed)


def test_validation_labels_do_not_affect_development_output() -> None:
    canonical = _small_canonical()
    baseline = modeling_data.build_development_modeling_data(canonical)
    changed = canonical.copy(deep=True)
    validation = changed["split"].eq("validation")
    _rotate_values(changed, validation, ("target",))
    result = modeling_data.build_development_modeling_data(changed)
    pd.testing.assert_frame_equal(
        baseline["development_features"], result["development_features"]
    )
    pd.testing.assert_series_equal(
        baseline["development_target"], result["development_target"]
    )


def test_import_and_reload_are_free_of_process_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = _process_state_snapshot(tmp_path)

    package = importlib.import_module("src.modeling")
    importlib.reload(package)
    importlib.reload(modeling_data)

    after = _process_state_snapshot(tmp_path)
    _assert_process_state_equal(before, after)


def _side_effect_case(case: str) -> tuple[pd.DataFrame, bool]:
    canonical = _small_canonical()
    if case == "schema_failure":
        canonical = canonical.drop(columns="visit_type")
    elif case == "dtype_failure":
        canonical["appointment_id"] = canonical["appointment_id"].astype(
            "int32"
        )
    elif case == "invalid_split_failure":
        canonical.loc[5, "split"] = "future"
    elif case == "temporal_failure":
        canonical.loc[5, "prediction_time"] = (
            _EXPECTED_TEST_START - pd.Timedelta(1, unit="ns")
        )
    elif case == "domain_failure":
        canonical.loc[0, "planned_duration_min"] = 0
    elif case == "target_failure":
        canonical.loc[0, "target"] = 2
    elif case == "maturity_failure":
        canonical.loc[3, "development_fit_eligible"] = True
    elif case == "empty_population_failure":
        canonical["development_fit_eligible"] = False
    return canonical, case != "success"


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "schema_failure",
        "dtype_failure",
        "invalid_split_failure",
        "temporal_failure",
        "domain_failure",
        "target_failure",
        "maturity_failure",
        "empty_population_failure",
    ],
)
def test_calls_have_no_filesystem_or_process_state_side_effects(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, should_raise = _side_effect_case(case)
    canonical_before = canonical.copy(deep=True)
    monkeypatch.chdir(tmp_path)
    before = _process_state_snapshot(tmp_path)
    if should_raise:
        with pytest.raises((TypeError, ValueError)):
            modeling_data.build_development_modeling_data(canonical)
    else:
        modeling_data.build_development_modeling_data(canonical)
    after = _process_state_snapshot(tmp_path)
    _assert_process_state_equal(before, after)
    pd.testing.assert_frame_equal(canonical, canonical_before)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    docstrings: set[int] = set()
    scope_nodes = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, scope_nodes) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return docstrings


def _subscript_field(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _is_alias_expression(
    node: ast.AST,
    field: str,
    aliases: set[str],
) -> bool:
    return _subscript_field(node) == field or (
        isinstance(node, ast.Name) and node.id in aliases
    )


def _assignment_entries(
    tree: ast.AST,
) -> list[tuple[list[ast.expr], ast.expr]]:
    entries: list[tuple[list[ast.expr], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            entries.append((node.targets, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            entries.append(([node.target], node.value))
        elif isinstance(node, ast.NamedExpr):
            entries.append(([node.target], node.value))
    return entries


def _field_aliases(
    entries: list[tuple[list[ast.expr], ast.expr]],
    field: str,
) -> set[str]:
    aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for targets, value in entries:
            if not _is_alias_expression(value, field, aliases):
                continue
            for target in targets:
                for name in _bound_target_names(target):
                    if name not in aliases:
                        aliases.add(name)
                        changed = True
    return aliases


def _literal_strings(node: ast.AST) -> set[str]:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _contains_alias_or_field(
    node: ast.AST,
    field: str,
    aliases: set[str],
) -> bool:
    return any(
        _is_alias_expression(child, field, aliases)
        for child in ast.walk(node)
    )


def _is_test_mask(node: ast.AST, split_aliases: set[str]) -> bool:
    if isinstance(node, ast.Compare):
        operands = (node.left, *node.comparators)
        has_split = any(
            _is_alias_expression(item, "split", split_aliases)
            for item in operands
        )
        if has_split and "test" in _literal_strings(node):
            return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        method = node.func.attr
        receiver = node.func.value
        if method in {"eq", "isin"} and _is_alias_expression(
            receiver, "split", split_aliases
        ):
            if "test" in _literal_strings(node):
                return True
        if method == "query":
            query_text = " ".join(_literal_strings(node)).lower()
            if "split" in query_text and "test" in query_text:
                return True
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.Invert, ast.Not)
    ):
        operand = node.operand
        if isinstance(operand, ast.Call) and isinstance(
            operand.func, ast.Attribute
        ):
            values = _literal_strings(operand)
            if (
                operand.func.attr == "isin"
                and _is_alias_expression(
                    operand.func.value, "split", split_aliases
                )
                and values == {"train", "validation"}
            ):
                return True
    return any(
        _is_test_mask(child, split_aliases)
        for child in ast.iter_child_nodes(node)
    )


def _call_leaf_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _static_contract_violations(source: str) -> set[str]:
    tree = ast.parse(source)
    docstring_ids = _docstring_node_ids(tree)
    entries = _assignment_entries(tree)
    split_aliases = _field_aliases(entries, "split")
    pretest_aliases = _field_aliases(entries, "pretest_fit_eligible")
    violations: set[str] = set()

    forbidden_module_prefixes = (
        "sklearn",
        "scipy",
        "joblib",
        "src.analysis",
        "pickle",
        "pathlib",
    )
    forbidden_calls = {
        "fit",
        "fit_transform",
        "predict",
        "predict_proba",
        "decision_function",
        "score",
        "open",
        "Path",
        "read_csv",
        "read_parquet",
        "read_pickle",
        "to_csv",
        "to_parquet",
        "to_pickle",
        "save",
        "dump",
        "dumps",
        "mkdir",
        "makedirs",
        "write_text",
        "write_bytes",
        "write",
        "writelines",
        "touch",
        "unlink",
        "rmdir",
        "remove",
        "rename",
        "chdir",
        "putenv",
        "unsetenv",
        "filterwarnings",
        "simplefilter",
        "seed",
        "set_state",
        "setstate",
        "now",
        "utcnow",
        "today",
        "time",
    }
    forbidden_fields = {
        "status",
        "status_updated_at",
        "check_in_at",
        "chair_start_at",
        "chair_end_at",
        "checkout_at",
        "status_change_reason",
        "rescheduled_from_appointment_id",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(forbidden_module_prefixes):
                    violations.add("forbidden_dependency")
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith(forbidden_module_prefixes):
                violations.add("forbidden_dependency")
            for alias in node.names:
                if alias.name in forbidden_calls:
                    violations.add("forbidden_import")
        elif isinstance(node, (ast.Name, ast.Attribute, ast.arg)):
            name = (
                node.id
                if isinstance(node, ast.Name)
                else node.attr
                if isinstance(node, ast.Attribute)
                else node.arg
            )
            lowered = name.lower()
            if name in {
                "select_development_rows",
                "select_test_rows",
                "allow_test",
            } or ("select" in lowered and "test" in lowered):
                violations.add("protected_selector")
            if "calibrat" in lowered or "threshold" in lowered:
                violations.add("forbidden_modeling")
        elif isinstance(node, ast.Call):
            call_name = _call_leaf_name(node)
            if call_name is not None:
                lowered_call = call_name.lower()
                if call_name in forbidden_calls:
                    violations.add("forbidden_call")
                if "metric" in lowered_call or lowered_call.endswith("_score"):
                    violations.add("forbidden_modeling")
            if isinstance(node.func, ast.Attribute):
                receiver_attributes = {
                    child.attr
                    for child in ast.walk(node.func.value)
                    if isinstance(child, ast.Attribute)
                }
                receiver_names = {
                    child.id
                    for child in ast.walk(node.func.value)
                    if isinstance(child, ast.Name)
                }
                if (
                    call_name
                    in {"append", "clear", "extend", "insert", "pop", "update"}
                    and receiver_attributes.intersection({"environ", "filters"})
                ):
                    violations.add("process_global_mutation")
                if (
                    "random" in receiver_names
                    or "random" in receiver_attributes
                ) and call_name not in {"get_state", "getstate"}:
                    violations.add("process_global_mutation")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "query"
                and _is_test_mask(node, split_aliases)
            ):
                violations.add("protected_test_population")
        elif isinstance(node, ast.Constant) and id(node) not in docstring_ids:
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                if node.value in _FORBIDDEN_PRODUCTION_COUNTS:
                    violations.add("dataset_specific_count")
            if isinstance(node.value, str):
                normalized = node.value.lower().replace("\\", "/")
                if normalized in forbidden_fields:
                    violations.add("post_event_field")
                if any(
                    token in normalized
                    for token in ("src.analysis", "reports/eda", "data/processed")
                ):
                    violations.add("forbidden_artifact_reference")

    for targets, value in entries:
        if _is_test_mask(value, split_aliases):
            violations.add("protected_test_population")
        if _is_alias_expression(
            value, "pretest_fit_eligible", pretest_aliases
        ):
            violations.add("protected_pretest_population")
        for target in targets:
            target_nodes = ast.walk(target)
            if any(
                isinstance(child, ast.Attribute)
                and child.attr in {"environ", "filters"}
                for child in target_nodes
            ):
                violations.add("process_global_mutation")

    for node in ast.walk(tree):
        if isinstance(node, (ast.AugAssign, ast.Delete)):
            targets = [node.target] if isinstance(node, ast.AugAssign) else node.targets
            for target in targets:
                if any(
                    isinstance(child, ast.Attribute)
                    and child.attr in {"environ", "filters"}
                    for child in ast.walk(target)
                ):
                    violations.add("process_global_mutation")

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            selection = node.slice
            if _is_test_mask(selection, split_aliases):
                violations.add("protected_test_population")
            if _contains_alias_or_field(
                selection, "pretest_fit_eligible", pretest_aliases
            ):
                violations.add("protected_pretest_population")
        elif isinstance(node, ast.Return) and node.value is not None:
            if _is_test_mask(node.value, split_aliases):
                violations.add("protected_test_population")
            if _contains_alias_or_field(
                node.value, "pretest_fit_eligible", pretest_aliases
            ):
                violations.add("protected_pretest_population")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            for generator in node.generators:
                for condition in generator.ifs:
                    if _is_test_mask(condition, split_aliases):
                        violations.add("protected_test_population")
                    if _contains_alias_or_field(
                        condition,
                        "pretest_fit_eligible",
                        pretest_aliases,
                    ):
                        violations.add("protected_pretest_population")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {
            name
            for target in node.targets
            for name in _bound_target_names(target)
        }
        if not any("FEATURE" in name for name in target_names):
            continue
        if _literal_strings(node.value).intersection(
            _FORBIDDEN_FEATURE_COLUMNS
        ):
            violations.add("feature_leakage")

    return violations


def test_production_static_dependency_and_leakage_guards() -> None:
    source = _DATA_MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    assert imports == {
        "__future__",
        "numpy",
        "pandas",
        "src.data.build_dataset",
    }
    assert _static_contract_violations(source) == set()


@pytest.mark.parametrize(
    ("mutation", "expected_violation"),
    [
        ("result = estimator.fit(features)", "forbidden_call"),
        ("result = estimator.predict(features)", "forbidden_call"),
        ("result = estimator.predict_proba(features)", "forbidden_call"),
        (
            '_holdout = canonical["split"].eq("test")',
            "protected_test_population",
        ),
        (
            '_holdout = canonical["split"].isin(("test",))',
            "protected_test_population",
        ),
        (
            '_holdout = canonical.query("split == \'test\'")',
            "protected_test_population",
        ),
        (
            '_eligible = canonical["pretest_fit_eligible"]',
            "protected_pretest_population",
        ),
        ("expected_positives = 432", "dataset_specific_count"),
        ('frame.to_csv("output.csv")', "forbidden_call"),
        (
            'try:\n    pass\nexcept Exception:\n    open("failure.txt")',
            "forbidden_call",
        ),
        ("pickle.dump(model, handle)", "forbidden_call"),
        ("joblib.dump(model, handle)", "forbidden_call"),
        ("os.environ['MODEL'] = 'value'", "process_global_mutation"),
        ("warnings.filters.append(rule)", "process_global_mutation"),
        ("np.random.seed(7)", "process_global_mutation"),
        ("timestamp = time.time()", "forbidden_call"),
        ("from sklearn import metrics", "forbidden_dependency"),
        ("from pathlib import Path", "forbidden_dependency"),
    ],
)
def test_structural_static_guards_reject_executable_mutations(
    mutation: str,
    expected_violation: str,
) -> None:
    assert expected_violation in _static_contract_violations(mutation)


def test_static_guards_ignore_comments_and_docstrings() -> None:
    harmless = '''
"""fit predict src.analysis reports/eda and count 432 are documentation."""
# fit_transform, predict_proba, and data/processed are comments only.
def explain_contract():
    """Mention sklearn, pickle.dump, joblib.dump, and to_csv harmlessly."""
    return 64
DOMAIN_UPPER_BOUND = 23
MONTH_UPPER_BOUND = 12
BOUNDARY = "2025-03-01 00:00:00"
'''
    assert _static_contract_violations(harmless) == set()


@pytest.mark.parametrize("forbidden_field", sorted(_FORBIDDEN_FEATURE_COLUMNS))
def test_structural_guard_rejects_every_forbidden_feature_role(
    forbidden_field: str,
) -> None:
    mutation = (
        "_FEATURE_COLUMNS = "
        f"('planned_duration_min', '{forbidden_field}')"
    )
    assert "feature_leakage" in _static_contract_violations(mutation)


def test_feature_constants_cannot_gain_leaking_roles() -> None:
    tree = ast.parse(_DATA_MODULE_PATH.read_text(encoding="utf-8"))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for constant_name in (
        "NUMERIC_FEATURE_COLUMNS",
        "CATEGORICAL_FEATURE_COLUMNS",
        "_FEATURE_COLUMNS",
    ):
        literal = ast.literal_eval(assignments[constant_name])
        assert set(literal).isdisjoint(_FORBIDDEN_FEATURE_COLUMNS)
