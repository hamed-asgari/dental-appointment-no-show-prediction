"""Tests for the bounded deterministic in-memory EDA figure builder."""

from __future__ import annotations

import ast
from collections import UserDict
from copy import deepcopy
import gc
from importlib import reload, util
from itertools import combinations
import inspect
import math
from pathlib import Path
import sys
from typing import get_type_hints
import weakref

import matplotlib
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import pytest

from src.analysis import figures
from src.analysis.figures import build_eda_figures
from src.analysis.run_eda import build_eda_tables
from src.data import build_dataset as bd


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw"
_FIGURES_PATH = _REPOSITORY_ROOT / "src" / "analysis" / "figures.py"
_TABLE_KEYS = (
    "cohort_target",
    "missingness",
    "numeric_features",
    "numeric_by_target",
    "categorical_features",
    "temporal_coverage",
    "temporal_monthly",
    "numeric_drift",
    "categorical_drift_levels",
    "categorical_drift_features",
    "numeric_relationships",
)
_FIGURE_KEYS = (
    "class_balance",
    "temporal_monthly",
    "numeric_drift",
    "categorical_drift",
    "numeric_relationships",
)
_CONSUMED_TABLE_KEYS = (
    "cohort_target",
    "temporal_monthly",
    "numeric_drift",
    "categorical_drift_features",
    "numeric_relationships",
)
_UNUSED_TABLE_KEYS = (
    "missingness",
    "numeric_features",
    "numeric_by_target",
    "categorical_features",
    "temporal_coverage",
    "categorical_drift_levels",
)
_TEMPORAL_COUNT_COLUMNS = (
    "nominal_train_count",
    "mature_train_count",
    "maturity_exclusion_count",
    "positives",
    "negatives",
)
_NUMERIC_FEATURES = (
    "planned_duration_min",
    "booking_lead_time_hours",
    "approximate_age_at_prediction",
    "patient_registration_tenure_days",
    "dentist_tenure_days",
)
_CATEGORICAL_FEATURES = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
_PAIR_ORDER = tuple(combinations(_NUMERIC_FEATURES, 2))
_SCHEMAS = {
    "cohort_target": (
        "rows",
        "positives",
        "negatives",
        "prevalence",
        "wilson_lower",
        "wilson_upper",
        "duplicate_appointment_ids",
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
_EXPECTED_SIZES = {
    "class_balance": (8.0, 4.5),
    "temporal_monthly": (10.0, 5.5),
    "numeric_drift": (9.0, 4.8),
    "categorical_drift": (9.0, 4.8),
    "numeric_relationships": (8.0, 7.0),
}
_EXPECTED_AXES_COUNTS = {
    "class_balance": 1,
    "temporal_monthly": 2,
    "numeric_drift": 1,
    "categorical_drift": 1,
    "numeric_relationships": 2,
}


def _cohort_table() -> pd.DataFrame:
    return pd.DataFrame(
        [[100, 20, 80, 0.175, 0.1, 0.3, 0]],
        columns=_SCHEMAS["cohort_target"],
    )


def _temporal_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_month": pd.Series(
                ["2024-01", "2024-02", "2024-03"],
                dtype="string",
            ),
            "nominal_train_count": [11, 14, 1],
            "mature_train_count": [10, 12, 0],
            "maturity_exclusion_count": [1, 2, 1],
            "positives": [1, 0, 0],
            "negatives": [9, 12, 0],
            "no_show_rate": [0.1, np.nan, 0.25],
            "wilson_lower": [0.02, np.nan, np.nan],
            "wilson_upper": [0.4, np.nan, np.nan],
        },
        columns=_SCHEMAS["temporal_monthly"],
    )


def _numeric_drift_table() -> pd.DataFrame:
    row_count = len(_NUMERIC_FEATURES)
    values: dict[str, object] = {
        "feature": list(_NUMERIC_FEATURES),
        "train_rows": [100] * row_count,
        "validation_rows": [40] * row_count,
        "train_n": [100] * row_count,
        "validation_n": [40] * row_count,
        "train_missing_count": [0] * row_count,
        "validation_missing_count": [0] * row_count,
        "train_missing_rate": [0.0] * row_count,
        "validation_missing_rate": [0.0] * row_count,
        "missing_rate_difference": [0.0] * row_count,
        "train_mean": [1.0] * row_count,
        "validation_mean": [2.0] * row_count,
        "train_std": [1.0] * row_count,
        "validation_std": [1.0] * row_count,
        "signed_smd": [-0.25, 0.0, np.nan, 0.4, 0.1],
        "train_q10": [0.0] * row_count,
        "validation_q10": [0.0] * row_count,
        "q10_shift": [0.0] * row_count,
        "train_median": [1.0] * row_count,
        "validation_median": [1.0] * row_count,
        "median_shift": [0.0] * row_count,
        "train_q90": [2.0] * row_count,
        "validation_q90": [2.0] * row_count,
        "q90_shift": [0.0] * row_count,
    }
    return pd.DataFrame(values, columns=_SCHEMAS["numeric_drift"])


def _categorical_drift_table() -> pd.DataFrame:
    row_count = len(_CATEGORICAL_FEATURES)
    values: dict[str, object] = {
        "feature": list(_CATEGORICAL_FEATURES),
        "train_rows": [100] * row_count,
        "validation_rows": [40] * row_count,
        "train_missing_count": [0] * row_count,
        "validation_missing_count": [0] * row_count,
        "train_missing_rate": [0.0] * row_count,
        "validation_missing_rate": [0.0] * row_count,
        "missing_rate_difference": [0.0] * row_count,
        "train_distinct_nonmissing_levels": [3] * row_count,
        "validation_distinct_nonmissing_levels": [3] * row_count,
        "unseen_in_train_level_count": [0] * row_count,
        "unseen_in_train_validation_count": [0] * row_count,
        "unseen_in_train_validation_share": [0.0] * row_count,
        "absent_in_validation_level_count": [0] * row_count,
        "absent_in_validation_train_count": [0] * row_count,
        "absent_in_validation_train_share": [0.0] * row_count,
        "total_variation_distance": [0.1, 0.2, 0.3, 0.4, 0.95],
        "max_absolute_share_difference": [0.05] * row_count,
    }
    return pd.DataFrame(
        values,
        columns=_SCHEMAS["categorical_drift_features"],
    )


def _relationship_table() -> pd.DataFrame:
    row_count = len(_PAIR_ORDER)
    correlations = [0.1, -0.2, np.nan, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 1.0]
    values: dict[str, object] = {
        "feature_a": [pair[0] for pair in _PAIR_ORDER],
        "feature_b": [pair[1] for pair in _PAIR_ORDER],
        "train_rows": [100] * row_count,
        "paired_n": [100] * row_count,
        "missing_pair_count": [0] * row_count,
        "paired_rate": [1.0] * row_count,
        "feature_a_unique_n": [10] * row_count,
        "feature_b_unique_n": [10] * row_count,
        "pearson_correlation": [0.77] * row_count,
        "absolute_pearson_correlation": [0.77] * row_count,
        "spearman_correlation": correlations,
        "absolute_spearman_correlation": [
            np.nan if np.isnan(value) else abs(value)
            for value in correlations
        ],
    }
    return pd.DataFrame(values, columns=_SCHEMAS["numeric_relationships"])


def _valid_tables() -> dict[str, pd.DataFrame]:
    tables = {
        key: pd.DataFrame({"unused": [position]})
        for position, key in enumerate(_TABLE_KEYS)
    }
    tables["cohort_target"] = _cohort_table()
    tables["temporal_monthly"] = _temporal_table()
    tables["numeric_drift"] = _numeric_drift_table()
    tables["categorical_drift_features"] = _categorical_drift_table()
    tables["numeric_relationships"] = _relationship_table()
    return tables


def _render(figure: Figure) -> bytes:
    figure.canvas.draw()
    return bytes(figure.canvas.buffer_rgba())


def _tick_text(axis: object, direction: str) -> tuple[str, ...]:
    getter = getattr(axis, f"get_{direction}ticklabels")
    return tuple(label.get_text() for label in getter())


def _snapshot_bundle(
    tables: dict[str, pd.DataFrame],
) -> tuple[
    tuple[str, ...],
    dict[str, int],
    dict[str, pd.DataFrame],
]:
    return (
        tuple(tables),
        {key: id(frame) for key, frame in tables.items()},
        {key: frame.copy(deep=True) for key, frame in tables.items()},
    )


def _assert_bundle_unchanged(
    tables: dict[str, pd.DataFrame],
    snapshot: tuple[
        tuple[str, ...],
        dict[str, int],
        dict[str, pd.DataFrame],
    ],
) -> None:
    keys, identities, frames = snapshot
    assert tuple(tables) == keys
    assert {key: id(frame) for key, frame in tables.items()} == identities
    for key, expected in frames.items():
        pd.testing.assert_frame_equal(tables[key], expected)


def _assert_validation_failure_without_allocation(
    tables: dict[str, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot_bundle(tables)
    allocations: list[tuple[float, float]] = []

    def tracked_constructor(figsize: tuple[float, float]) -> Figure:
        allocations.append(figsize)
        raise AssertionError("validation allocated a Figure")

    monkeypatch.setattr(figures, "_new_figure", tracked_constructor)
    with pytest.raises((TypeError, ValueError)):
        build_eda_figures(tables)
    assert allocations == []
    _assert_bundle_unchanged(tables, snapshot)


def _target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name
            for element in target.elts
            for name in _target_names(element)
        )
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return ()


def _module_owned_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.extend(_target_names(node.target))
        elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
            names.extend(_target_names(node.name))
    return tuple(names)


def _matplotlib_state() -> tuple[
    object,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    tuple[int, ...],
]:
    backend = matplotlib.get_backend()
    return (
        backend,
        deepcopy(dict(matplotlib.rcParams)),
        deepcopy(dict(matplotlib.rcParamsDefault)),
        deepcopy(dict(matplotlib.rcParamsOrig)),
        tuple(id(manager) for manager in Gcf.get_all_fig_managers()),
    )


def _assert_matplotlib_state_unchanged(
    expected: tuple[
        object,
        dict[str, object],
        dict[str, object],
        dict[str, object],
        tuple[int, ...],
    ],
) -> None:
    actual = _matplotlib_state()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert actual[2] == expected[2]
    assert actual[3] == expected[3]
    assert actual[4] == expected[4]


def _reference_parts(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_reference_parts(node.value), node.attr)
    if isinstance(node, ast.Subscript):
        return _reference_parts(node.value)
    return ()


def _global_state_violations(tree: ast.Module) -> tuple[str, ...]:
    matplotlib_names = {"matplotlib"}
    rc_names = {"rcParams", "rcParamsDefault", "rcParamsOrig"}
    gcf_names = {"Gcf"}
    backend_calls = {"use", "switch_backend"}
    rc_context_calls = {"rc_context"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib":
                    matplotlib_names.add(alias.asname or "matplotlib")
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if node.module == "matplotlib" and alias.name in rc_names:
                    rc_names.add(bound_name)
                if alias.name in {"use", "switch_backend"}:
                    backend_calls.add(bound_name)
                if alias.name == "rc_context":
                    rc_context_calls.add(bound_name)
                if node.module == "matplotlib._pylab_helpers" and (
                    alias.name == "Gcf"
                ):
                    gcf_names.add(bound_name)

    def is_rc_reference(node: ast.AST) -> bool:
        parts = _reference_parts(node)
        return bool(parts) and (
            parts[0] in rc_names
            or (
                parts[0] in matplotlib_names
                and any(part in rc_names for part in parts[1:])
            )
        )

    def is_gcf_reference(node: ast.AST) -> bool:
        parts = _reference_parts(node)
        return bool(parts) and parts[0] in gcf_names

    def is_backend_registry_reference(node: ast.AST) -> bool:
        parts = _reference_parts(node)
        return bool(parts) and parts[0] in matplotlib_names and any(
            part in {"backends", "backend_registry"} for part in parts[1:]
        )

    def target_mutates(
        target: ast.AST,
        predicate: object,
    ) -> bool:
        check = predicate
        if callable(check) and check(target):
            return True
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(target_mutates(element, predicate) for element in target.elts)
        if isinstance(target, ast.Starred):
            return target_mutates(target.value, predicate)
        return False

    violations: list[str] = []
    mutating_methods = {
        "__delitem__",
        "__setitem__",
        "clear",
        "destroy",
        "destroy_all",
        "destroy_fig",
        "pop",
        "popitem",
        "register",
        "set_active",
        "setdefault",
        "update",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts = _reference_parts(node.func)
            called_name = parts[-1] if parts else ""
            if called_name in backend_calls or called_name in rc_context_calls:
                violations.append(f"global configuration call: {called_name}")
            if isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if called_name in mutating_methods and (
                    is_rc_reference(owner)
                    or is_gcf_reference(owner)
                    or is_backend_registry_reference(owner)
                ):
                    violations.append(f"global-state mutation call: {called_name}")
            if isinstance(node.func, ast.Name) and node.func.id == "setattr":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    attribute = node.args[1].value
                    if attribute in rc_names or attribute in backend_calls:
                        violations.append(f"global setattr mutation: {attribute}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if any(
                    target_mutates(target, predicate)
                    for predicate in (
                        is_rc_reference,
                        is_gcf_reference,
                        is_backend_registry_reference,
                    )
                ):
                    violations.append("global-state assignment")
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if any(
                target_mutates(node.target, predicate)
                for predicate in (
                    is_rc_reference,
                    is_gcf_reference,
                    is_backend_registry_reference,
                )
            ):
                violations.append("global-state annotated/augmented assignment")
    return tuple(violations)


@pytest.fixture(scope="module")
def authentic_tables() -> dict[str, pd.DataFrame]:
    canonical = bd.build_analytical_dataset(bd.load_raw_data(_RAW_DIR))
    return build_eda_tables(canonical)


def test_exact_public_signature() -> None:
    signature = inspect.signature(build_eda_figures)
    assert tuple(signature.parameters) == ("tables",)
    parameter = signature.parameters["tables"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    hints = get_type_hints(build_eda_figures)
    assert hints == {
        "tables": dict[str, pd.DataFrame],
        "return": dict[str, Figure],
    }


def test_exact_all_and_controlled_wildcard_export() -> None:
    assert figures.__all__ == ("build_eda_figures",)
    namespace: dict[str, object] = {}
    exec("from src.analysis.figures import *", namespace)
    exported = {
        name: value
        for name, value in namespace.items()
        if not name.startswith("__")
    }
    assert exported == {"build_eda_figures": build_eda_figures}


def test_exact_keys_figure_types_sizes_dpi_and_axes_counts() -> None:
    result = build_eda_figures(_valid_tables())
    assert type(result) is dict
    assert tuple(result) == _FIGURE_KEYS
    for key, figure in result.items():
        assert type(figure) is Figure
        assert type(figure.canvas) is FigureCanvasAgg
        assert tuple(figure.get_size_inches()) == _EXPECTED_SIZES[key]
        assert figure.dpi == 120
        assert len(figure.axes) == _EXPECTED_AXES_COUNTS[key]


def test_reuses_package_private_bundle_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    class ValidatorCalled(Exception):
        pass

    def validator(value: object) -> None:
        assert value is sentinel
        raise ValidatorCalled

    monkeypatch.setattr(figures, "_validate_tables", validator)
    with pytest.raises(ValidatorCalled):
        build_eda_figures(sentinel)  # type: ignore[arg-type]


def test_consumed_source_schemas_are_literal_and_exact() -> None:
    tables = _valid_tables()
    for key, expected in _SCHEMAS.items():
        assert tuple(tables[key].columns) == expected


@pytest.mark.parametrize("case", ["non_dict", "wrong_order", "missing", "extra"])
def test_rejects_malformed_bundle_keys_and_type(case: str) -> None:
    tables = _valid_tables()
    if case == "non_dict":
        malformed: object = UserDict(tables)
        error = TypeError
    elif case == "wrong_order":
        first = tables.pop("cohort_target")
        tables["cohort_target"] = first
        malformed = tables
        error = ValueError
    elif case == "missing":
        tables.pop("missingness")
        malformed = tables
        error = ValueError
    else:
        tables["extra"] = pd.DataFrame()
        malformed = tables
        error = ValueError
    with pytest.raises(error):
        build_eda_figures(malformed)  # type: ignore[arg-type]


def test_rejects_non_dataframe_and_duplicate_dataframe_identity() -> None:
    non_frame = _valid_tables()
    non_frame["missingness"] = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="pandas DataFrame"):
        build_eda_figures(non_frame)

    duplicate = _valid_tables()
    duplicate["missingness"] = duplicate["numeric_features"]
    with pytest.raises(ValueError, match="distinct DataFrame"):
        build_eda_figures(duplicate)


@pytest.mark.parametrize("table_name", tuple(_SCHEMAS))
@pytest.mark.parametrize("malformation", ["missing", "extra", "reordered"])
def test_rejects_malformed_consumed_schema(
    table_name: str,
    malformation: str,
) -> None:
    tables = _valid_tables()
    frame = tables[table_name].copy(deep=True)
    if malformation == "missing":
        frame = frame.drop(columns=frame.columns[-1])
    elif malformation == "extra":
        frame["unexpected"] = 0
    else:
        columns = list(frame.columns)
        columns[0], columns[1] = columns[1], columns[0]
        frame = frame.loc[:, columns]
    tables[table_name] = frame
    with pytest.raises(ValueError, match="columns and order"):
        build_eda_figures(tables)


def test_all_validation_occurs_before_figure_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    tables["numeric_relationships"].loc[0, "spearman_correlation"] = np.inf
    calls = 0

    def forbidden_constructor(figsize: tuple[float, float]) -> Figure:
        nonlocal calls
        calls += 1
        raise AssertionError(figsize)

    monkeypatch.setattr(figures, "_new_figure", forbidden_constructor)
    with pytest.raises(ValueError, match="must be finite"):
        build_eda_figures(tables)
    assert calls == 0


@pytest.mark.parametrize("table_name", ["numeric_drift", "categorical_drift"])
def test_rejects_reordered_feature_rows(table_name: str) -> None:
    tables = _valid_tables()
    source_key = (
        "numeric_drift"
        if table_name == "numeric_drift"
        else "categorical_drift_features"
    )
    tables[source_key] = tables[source_key].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="order must be exactly"):
        build_eda_figures(tables)


@pytest.mark.parametrize(
    "case",
    ["descending", "duplicate", "null", "timezone_aware", "not_month_like"],
)
def test_temporal_order_and_type_guards(case: str) -> None:
    tables = _valid_tables()
    frame = tables["temporal_monthly"].copy(deep=True)
    if case == "descending":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif case == "duplicate":
        frame.loc[1, "prediction_month"] = "2024-01"
    elif case == "null":
        frame.loc[1, "prediction_month"] = pd.NA
    elif case == "timezone_aware":
        frame["prediction_month"] = pd.date_range(
            "2024-01-01",
            periods=3,
            freq="MS",
            tz="UTC",
        )
    else:
        frame["prediction_month"] = ["January", "February", "March"]
    tables["temporal_monthly"] = frame
    with pytest.raises((TypeError, ValueError)):
        build_eda_figures(tables)


@pytest.mark.parametrize("dtype", ["datetime", "period"])
def test_accepts_naive_datetime_and_monthly_period_order(dtype: str) -> None:
    tables = _valid_tables()
    frame = tables["temporal_monthly"].copy(deep=True)
    if dtype == "datetime":
        frame["prediction_month"] = pd.date_range(
            "2024-01-01",
            periods=3,
            freq="MS",
        )
    else:
        frame["prediction_month"] = pd.period_range(
            "2024-01",
            periods=3,
            freq="M",
        )
    tables["temporal_monthly"] = frame
    result = build_eda_figures(tables)
    assert _tick_text(result["temporal_monthly"].axes[0], "x") == (
        "2024-01",
        "2024-02",
        "2024-03",
    )


@pytest.mark.parametrize(
    ("start", "frequency"),
    [
        ("2024-01-01", "D"),
        ("2024Q1", "Q"),
        ("2024-01", "2M"),
    ],
)
def test_rejects_unsupported_period_frequencies_without_side_effects(
    start: str,
    frequency: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    tables["temporal_monthly"]["prediction_month"] = pd.Series(
        pd.period_range(start, periods=3, freq=frequency)
    )
    bundle_snapshot = _snapshot_bundle(tables)
    allocations: list[tuple[float, float]] = []
    no_result = object()
    result: object = no_result

    def tracked_constructor(figsize: tuple[float, float]) -> Figure:
        allocations.append(figsize)
        raise AssertionError("unsupported Period allocated a Figure")

    monkeypatch.setattr(figures, "_new_figure", tracked_constructor)
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))
    state_before = _matplotlib_state()
    with pytest.raises(TypeError):
        result = build_eda_figures(tables)
    assert result is no_result
    assert allocations == []
    _assert_bundle_unchanged(tables, bundle_snapshot)
    _assert_matplotlib_state_unchanged(state_before)
    assert tuple(tmp_path.rglob("*")) == files_before


def test_accepts_unit_month_period_control_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    expected_months = pd.Series(
        pd.period_range("2024-01", periods=3, freq="M"),
        name="prediction_month",
    )
    tables["temporal_monthly"]["prediction_month"] = expected_months.copy()
    bundle_snapshot = _snapshot_bundle(tables)
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))
    state_before = _matplotlib_state()
    result = build_eda_figures(tables)
    assert tuple(result) == _FIGURE_KEYS
    pd.testing.assert_series_equal(
        tables["temporal_monthly"]["prediction_month"],
        expected_months,
    )
    _assert_bundle_unchanged(tables, bundle_snapshot)
    _assert_matplotlib_state_unchanged(state_before)
    assert tuple(tmp_path.rglob("*")) == files_before


@pytest.mark.parametrize(
    "values",
    [
        ("2024-01-01", "2024-01-31", "2024-02-01"),
        ("2024-01-31", "2024-01-01", "2024-02-01"),
        (
            "2024-01-10 08:00:00",
            "2024-01-10 17:30:00",
            "2024-02-01 00:00:00",
        ),
        ("2024-01-01", "2024-02-01", "2024-01-15"),
    ],
)
def test_rejects_duplicate_calendar_month_datetimes_before_allocation(
    values: tuple[str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    tables["temporal_monthly"]["prediction_month"] = pd.to_datetime(values)
    _assert_validation_failure_without_allocation(tables, monkeypatch)


def test_accepts_one_datetime_per_strictly_increasing_calendar_month() -> None:
    tables = _valid_tables()
    tables["temporal_monthly"]["prediction_month"] = pd.to_datetime(
        (
            "2024-01-31 23:59:59",
            "2024-02-01 00:00:01",
            "2024-03-15 12:30:00",
        )
    )
    result = build_eda_figures(tables)
    assert _tick_text(result["temporal_monthly"].axes[0], "x") == (
        "2024-01",
        "2024-02",
        "2024-03",
    )


@pytest.mark.parametrize(
    "labels",
    [
        ("2024-1", "2024-02", "2024-12"),
        ("2024-01-01", "2024-02", "2024-12"),
        (" 2024-01", "2024-02", "2024-12"),
        ("2024-01 ", "2024-02", "2024-12"),
        ("2024-00", "2024-02", "2024-12"),
        ("2024-01", "2024-13", "2024-12"),
        ("2024-01", "invalid", "2024-12"),
        ("2024-01", pd.NA, "2024-12"),
    ],
)
def test_rejects_non_strict_month_strings_before_allocation(
    labels: tuple[object, object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    tables["temporal_monthly"]["prediction_month"] = pd.Series(
        labels,
        dtype="string",
    )
    _assert_validation_failure_without_allocation(tables, monkeypatch)


def test_accepts_strict_increasing_month_strings() -> None:
    tables = _valid_tables()
    tables["temporal_monthly"]["prediction_month"] = pd.Series(
        ("2024-01", "2024-02", "2024-12"),
        dtype="string",
    )
    result = build_eda_figures(tables)
    assert _tick_text(result["temporal_monthly"].axes[0], "x") == (
        "2024-01",
        "2024-02",
        "2024-12",
    )


def test_rejects_object_dtype_month_strings_without_broadening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    tables["temporal_monthly"]["prediction_month"] = pd.Series(
        ("2024-01", "2024-02", "2024-03"),
        dtype="object",
    )
    _assert_validation_failure_without_allocation(tables, monkeypatch)


@pytest.mark.parametrize(
    "case",
    ["missing", "extra", "duplicate", "reversed", "self", "reordered"],
)
def test_relationship_pair_order_guards(case: str) -> None:
    tables = _valid_tables()
    frame = tables["numeric_relationships"].copy(deep=True)
    if case == "missing":
        frame = frame.iloc[:-1].copy()
    elif case == "extra":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif case == "duplicate":
        frame.loc[1, ["feature_a", "feature_b"]] = frame.loc[
            0,
            ["feature_a", "feature_b"],
        ].to_numpy()
    elif case == "reversed":
        frame.loc[0, ["feature_a", "feature_b"]] = frame.loc[
            0,
            ["feature_b", "feature_a"],
        ].to_numpy()
    elif case == "self":
        frame.loc[0, "feature_b"] = frame.loc[0, "feature_a"]
    else:
        frame = frame.iloc[[1, 0, *range(2, len(frame))]].reset_index(drop=True)
    tables["numeric_relationships"] = frame
    with pytest.raises(ValueError, match="pairs and order"):
        build_eda_figures(tables)


def test_class_balance_semantics_and_direct_prevalence() -> None:
    axis = build_eda_figures(_valid_tables())["class_balance"].axes[0]
    assert [bar.get_height() for bar in axis.patches] == [80, 20]
    assert _tick_text(axis, "x") == ("Attended", "No-show")
    assert axis.get_xlabel() == "Outcome"
    assert axis.get_ylabel() == "Appointments"
    assert axis.get_title() == "Mature Train Class Balance"
    text = tuple(annotation.get_text() for annotation in axis.texts)
    assert "80" in text
    assert "20" in text
    assert "No-show prevalence: 17.50%" in text


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("rows", 99),
        ("positives", -1),
        ("negatives", 80.5),
        ("prevalence", -0.1),
        ("prevalence", 1.1),
        ("prevalence", np.nan),
        ("prevalence", np.inf),
        ("duplicate_appointment_ids", 1),
    ],
)
def test_class_balance_rejects_invalid_values(column: str, value: float) -> None:
    tables = _valid_tables()
    if column in {"rows", "positives", "negatives"}:
        tables["cohort_target"][column] = tables["cohort_target"][
            column
        ].astype("float64")
    tables["cohort_target"].loc[0, column] = value
    with pytest.raises((TypeError, ValueError)):
        build_eda_figures(tables)


def test_class_balance_requires_one_row() -> None:
    tables = _valid_tables()
    tables["cohort_target"] = pd.concat(
        [tables["cohort_target"], tables["cohort_target"]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="exactly one row"):
        build_eda_figures(tables)


def test_temporal_semantics_percentage_axis_and_nan_gap() -> None:
    figure = build_eda_figures(_valid_tables())["temporal_monthly"]
    count_axis, rate_axis = figure.axes
    assert [bar.get_height() for bar in count_axis.patches] == [10, 12, 0]
    line = rate_axis.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
    np.testing.assert_allclose(
        line.get_ydata(),
        [0.1, np.nan, 0.25],
        equal_nan=True,
    )
    assert np.isnan(line.get_path().vertices[1, 1])
    assert _tick_text(count_axis, "x") == (
        "2024-01",
        "2024-02",
        "2024-03",
    )
    assert count_axis.get_xlabel() == "Prediction Month"
    assert count_axis.get_ylabel() == "Mature Appointments"
    assert rate_axis.get_ylabel() == "No-show Rate"
    assert count_axis.get_title() == (
        "Mature Train Volume and No-show Rate by Month"
    )
    assert rate_axis.get_ylim() == (0.0, 1.0)
    assert isinstance(rate_axis.yaxis.get_major_formatter(), PercentFormatter)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("mature_train_count", -1),
        ("mature_train_count", 1.5),
        ("no_show_rate", -0.01),
        ("no_show_rate", 1.01),
        ("no_show_rate", np.inf),
        ("no_show_rate", -np.inf),
    ],
)
def test_temporal_rejects_invalid_rates_and_counts(
    column: str,
    value: float,
) -> None:
    tables = _valid_tables()
    if column == "mature_train_count":
        tables["temporal_monthly"][column] = tables["temporal_monthly"][
            column
        ].astype("float64")
    tables["temporal_monthly"].loc[0, column] = value
    with pytest.raises((TypeError, ValueError)):
        build_eda_figures(tables)


@pytest.mark.parametrize("column", _TEMPORAL_COUNT_COLUMNS)
@pytest.mark.parametrize(
    "value",
    [-1, 1.5, True, False, math.inf, -math.inf, math.nan],
)
def test_every_temporal_count_rejects_invalid_values_before_allocation(
    column: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    frame = tables["temporal_monthly"]
    frame[column] = frame[column].astype("object")
    frame.at[0, column] = value
    _assert_validation_failure_without_allocation(tables, monkeypatch)


@pytest.mark.parametrize(
    "value",
    [
        np.int8(3),
        np.uint64(3),
        np.float32(3.0),
        pd.array([3], dtype="Int64")[0],
    ],
)
def test_temporal_counts_accept_supported_integral_numeric_scalars(
    value: object,
) -> None:
    tables = _valid_tables()
    frame = tables["temporal_monthly"]
    frame["nominal_train_count"] = frame["nominal_train_count"].astype(
        "object"
    )
    frame.at[0, "nominal_train_count"] = value
    result = build_eda_figures(tables)
    assert [
        bar.get_height()
        for bar in result["temporal_monthly"].axes[0].patches
    ] == [10, 12, 0]


def test_numeric_drift_signed_widths_zero_reference_and_na_marker() -> None:
    axis = build_eda_figures(_valid_tables())["numeric_drift"].axes[0]
    widths = np.array([bar.get_width() for bar in axis.patches])
    np.testing.assert_allclose(
        widths,
        [-0.25, 0.0, np.nan, 0.4, 0.1],
        equal_nan=True,
    )
    assert _tick_text(axis, "y") == _NUMERIC_FEATURES
    assert axis.get_xlabel() == "Signed Standardized Mean Difference"
    assert axis.get_ylabel() == "Feature"
    assert axis.get_title() == "Numerical Feature Drift: Train vs Validation"
    assert any(
        np.array_equal(line.get_xdata(), [0.0, 0.0])
        for line in axis.lines
    )
    labels = tuple(text.get_text() for text in axis.texts)
    assert "-0.250" in labels
    assert "0.000" in labels
    assert "NA" in labels
    na_lines = [line for line in axis.lines if line.get_marker() == "x"]
    assert len(na_lines) == 1
    assert np.array_equal(na_lines[0].get_xdata(), [0.0])


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_numeric_drift_rejects_infinity(value: float) -> None:
    tables = _valid_tables()
    tables["numeric_drift"].loc[2, "signed_smd"] = value
    with pytest.raises(ValueError, match="must be finite"):
        build_eda_figures(tables)


def test_categorical_drift_semantics_and_exact_values() -> None:
    axis = build_eda_figures(_valid_tables())["categorical_drift"].axes[0]
    widths = [bar.get_width() for bar in axis.patches]
    assert widths == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.95])
    assert _tick_text(axis, "y") == _CATEGORICAL_FEATURES
    assert axis.get_xlim() == (0.0, 1.0)
    assert axis.get_xlabel() == "Total Variation Distance"
    assert axis.get_ylabel() == "Feature"
    assert axis.get_title() == "Categorical Feature Drift: Train vs Validation"
    assert tuple(text.get_text() for text in axis.texts) == (
        "0.100",
        "0.200",
        "0.300",
        "0.400",
        "0.950",
    )


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_categorical_drift_rejects_invalid_tvd(value: float) -> None:
    tables = _valid_tables()
    tables["categorical_drift_features"].loc[
        0,
        "total_variation_distance",
    ] = value
    with pytest.raises((TypeError, ValueError)):
        build_eda_figures(tables)


def test_relationship_heatmap_exact_symmetric_spearman_matrix() -> None:
    figure = build_eda_figures(_valid_tables())["numeric_relationships"]
    axis, colorbar_axis = figure.axes
    image = axis.images[0]
    expected = np.array(
        [
            [1.0, 0.1, -0.2, np.nan, 0.4],
            [0.1, 1.0, 0.5, -0.6, 0.7],
            [-0.2, 0.5, 1.0, 0.8, -0.9],
            [np.nan, -0.6, 0.8, 1.0, 1.0],
            [0.4, 0.7, -0.9, 1.0, 1.0],
        ]
    )
    actual = np.ma.filled(image.get_array(), np.nan)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert image.get_clim() == (-1.0, 1.0)
    assert _tick_text(axis, "x") == _NUMERIC_FEATURES
    assert _tick_text(axis, "y") == _NUMERIC_FEATURES
    assert axis.get_title() == (
        "Mature Train Numerical Spearman Relationships"
    )
    assert colorbar_axis.get_ylabel() == "Spearman Correlation"
    labels = tuple(text.get_text() for text in axis.texts)
    assert len(labels) == 25
    assert labels.count("NA") == 2
    assert labels.count("1.00") == 7


@pytest.mark.parametrize("value", [-1.01, 1.01, np.inf, -np.inf])
def test_relationships_reject_invalid_correlation(value: float) -> None:
    tables = _valid_tables()
    tables["numeric_relationships"].loc[0, "spearman_correlation"] = value
    with pytest.raises(ValueError):
        build_eda_figures(tables)


def test_authentic_integration_reconciles_literal_source_values(
    authentic_tables: dict[str, pd.DataFrame],
) -> None:
    result = build_eda_figures(authentic_tables)
    assert tuple(result) == _FIGURE_KEYS

    class_axis = result["class_balance"].axes[0]
    assert [bar.get_height() for bar in class_axis.patches] == [3238, 432]
    assert "No-show prevalence: 11.77%" in {
        text.get_text() for text in class_axis.texts
    }

    temporal_axis, rate_axis = result["temporal_monthly"].axes
    assert [bar.get_height() for bar in temporal_axis.patches] == [
        333,
        319,
        306,
        306,
        335,
        298,
        321,
        302,
        277,
        285,
        321,
        267,
    ]
    np.testing.assert_allclose(
        rate_axis.lines[0].get_ydata()[[0, -1]],
        [0.15315315315315314, 0.1348314606741573],
    )

    numeric_widths = [
        bar.get_width() for bar in result["numeric_drift"].axes[0].patches
    ]
    np.testing.assert_allclose(
        numeric_widths,
        [-0.039155, 0.180708, 0.114324, 1.944817, 2.350378],
        rtol=2e-5,
    )
    categorical_widths = [
        bar.get_width()
        for bar in result["categorical_drift"].axes[0].patches
    ]
    np.testing.assert_allclose(
        categorical_widths,
        [0.023808, 0.016283, 0.023166, 0.022779, 0.567575],
        rtol=2e-5,
    )
    matrix = np.ma.filled(
        result["numeric_relationships"].axes[0].images[0].get_array(),
        np.nan,
    )
    assert matrix[0, 1] == -0.020593881200389668
    assert matrix[3, 4] == 0.5788107808424946


def test_input_mapping_and_all_dataframes_are_not_mutated() -> None:
    tables = _valid_tables()
    key_snapshot = tuple(tables)
    identity_snapshot = {key: id(frame) for key, frame in tables.items()}
    frame_snapshots = {
        key: frame.copy(deep=True) for key, frame in tables.items()
    }
    build_eda_figures(tables)
    assert tuple(tables) == key_snapshot
    assert {key: id(frame) for key, frame in tables.items()} == identity_snapshot
    for key, expected in frame_snapshots.items():
        pd.testing.assert_frame_equal(tables[key], expected)


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "temporal",
        "class_balance",
        "numeric_drift",
        "categorical_drift",
        "relationships",
    ],
)
def test_complete_bundle_is_unchanged_on_validation_failures(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    if case == "schema":
        tables["temporal_monthly"] = tables["temporal_monthly"].drop(
            columns="wilson_upper"
        )
    elif case == "temporal":
        frame = tables["temporal_monthly"]
        frame["nominal_train_count"] = frame["nominal_train_count"].astype(
            "object"
        )
        frame.at[0, "nominal_train_count"] = -1
    elif case == "class_balance":
        tables["cohort_target"].at[0, "rows"] = 99
    elif case == "numeric_drift":
        tables["numeric_drift"].at[0, "signed_smd"] = math.inf
    elif case == "categorical_drift":
        tables["categorical_drift_features"].at[
            0,
            "total_variation_distance",
        ] = -0.1
    else:
        tables["numeric_relationships"].at[
            0,
            "spearman_correlation",
        ] = math.inf
    _assert_validation_failure_without_allocation(tables, monkeypatch)


def test_complete_bundle_is_unchanged_on_middle_builder_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    bundle_snapshot = _snapshot_bundle(tables)
    sentinel = RuntimeError("injected numeric builder failure")
    original_constructor = figures._new_figure
    created_references: list[weakref.ReferenceType[Figure]] = []

    def tracking_constructor(figsize: tuple[float, float]) -> Figure:
        figure = original_constructor(figsize)
        created_references.append(weakref.ref(figure))
        return figure

    def fail_numeric_builder(frame: pd.DataFrame) -> Figure:
        assert frame is tables["numeric_drift"]
        raise sentinel

    monkeypatch.setattr(figures, "_new_figure", tracking_constructor)
    monkeypatch.setattr(figures, "_build_numeric_drift", fail_numeric_builder)
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))
    state_before = _matplotlib_state()
    no_result = object()
    result: object = no_result
    caught: RuntimeError | None = None
    try:
        result = build_eda_figures(tables)
    except RuntimeError as error:
        caught = error

    assert caught is sentinel
    assert result is no_result
    assert len(created_references) == 2
    _assert_matplotlib_state_unchanged(state_before)
    manager_figures = tuple(
        manager.canvas.figure for manager in Gcf.get_all_fig_managers()
    )
    created_figures = tuple(reference() for reference in created_references)
    assert all(figure not in manager_figures for figure in created_figures)
    _assert_bundle_unchanged(tables, bundle_snapshot)
    assert tuple(tmp_path.rglob("*")) == files_before

    del created_figures
    caught = None
    sentinel.__traceback__ = None
    sentinel.__context__ = None
    sentinel.__cause__ = None
    gc.collect()
    assert all(reference() is None for reference in created_references)


def test_fresh_figures_canvases_and_mutation_isolation() -> None:
    tables = _valid_tables()
    first = build_eda_figures(tables)
    second = build_eda_figures(tables)
    assert first is not second
    assert len({id(figure) for figure in first.values()}) == 5
    assert len({id(figure.canvas) for figure in first.values()}) == 5
    for key in _FIGURE_KEYS:
        assert first[key] is not second[key]
        assert first[key].canvas is not second[key].canvas

    first["class_balance"].axes[0].set_title("mutated")
    first["numeric_drift"].axes[0].patches[0].set_width(99)
    later = build_eda_figures(tables)
    assert later["class_balance"].axes[0].get_title() == (
        "Mature Train Class Balance"
    )
    assert later["numeric_drift"].axes[0].patches[0].get_width() == -0.25


def test_rgba_buffers_are_byte_identical_across_calls() -> None:
    first = build_eda_figures(_valid_tables())
    second = build_eda_figures(_valid_tables())
    assert {
        key: _render(figure) for key, figure in first.items()
    } == {
        key: _render(figure) for key, figure in second.items()
    }


def test_changed_valid_indexes_do_not_change_rendered_buffers() -> None:
    original = _valid_tables()
    changed = {
        key: frame.copy(deep=True) for key, frame in original.items()
    }
    for position, frame in enumerate(changed.values(), start=1):
        frame.index = pd.Index(
            range(position * 100, position * 100 + len(frame)),
            name=f"index_{position}",
        )
    first = build_eda_figures(original)
    second = build_eda_figures(changed)
    assert {
        key: _render(figure) for key, figure in first.items()
    } == {
        key: _render(figure) for key, figure in second.items()
    }


def test_current_working_directory_does_not_change_buffers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    baseline = {
        key: _render(figure)
        for key, figure in build_eda_figures(tables).items()
    }
    monkeypatch.chdir(tmp_path)
    changed = {
        key: _render(figure)
        for key, figure in build_eda_figures(tables).items()
    }
    assert changed == baseline


def test_no_pyplot_figure_manager_growth() -> None:
    from matplotlib._pylab_helpers import Gcf

    before = tuple(Gcf.get_all_fig_managers())
    result = build_eda_figures(_valid_tables())
    for figure in result.values():
        _render(figure)
    after = tuple(Gcf.get_all_fig_managers())
    assert after == before


def test_isolated_import_and_reload_preserve_matplotlib_global_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    files_before_import = tuple(tmp_path.rglob("*"))
    state_before_import = _matplotlib_state()
    specification = util.spec_from_file_location(
        "_isolated_eda_figures_global_state_probe",
        _FIGURES_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    isolated_module = util.module_from_spec(specification)
    specification.loader.exec_module(isolated_module)
    _assert_matplotlib_state_unchanged(state_before_import)
    assert tuple(tmp_path.rglob("*")) == files_before_import
    assert not any(
        isinstance(value, Figure)
        for value in vars(isolated_module).values()
    )

    files_before_reload = tuple(tmp_path.rglob("*"))
    state_before_reload = _matplotlib_state()
    reloaded = reload(figures)
    assert reloaded is figures
    _assert_matplotlib_state_unchanged(state_before_reload)
    assert tuple(tmp_path.rglob("*")) == files_before_reload


def test_successful_and_repeated_builds_preserve_matplotlib_global_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    bundle_snapshot = _snapshot_bundle(tables)
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))

    state_before_first = _matplotlib_state()
    first = build_eda_figures(tables)
    assert tuple(first) == _FIGURE_KEYS
    _assert_matplotlib_state_unchanged(state_before_first)

    state_before_repeated = _matplotlib_state()
    second = build_eda_figures(tables)
    third = build_eda_figures(tables)
    assert tuple(second) == _FIGURE_KEYS
    assert tuple(third) == _FIGURE_KEYS
    _assert_matplotlib_state_unchanged(state_before_repeated)
    _assert_bundle_unchanged(tables, bundle_snapshot)
    assert tuple(tmp_path.rglob("*")) == files_before


def test_rendering_every_figure_preserves_matplotlib_global_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    result = build_eda_figures(tables)
    bundle_snapshot = _snapshot_bundle(tables)
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))
    state_before = _matplotlib_state()
    rendered = {key: _render(figure) for key, figure in result.items()}
    assert tuple(rendered) == _FIGURE_KEYS
    assert all(rendered.values())
    _assert_matplotlib_state_unchanged(state_before)
    _assert_bundle_unchanged(tables, bundle_snapshot)
    assert tuple(tmp_path.rglob("*")) == files_before


@pytest.mark.parametrize("case", ["schema", "temporal_semantic"])
def test_validation_failures_preserve_matplotlib_global_state(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tables = _valid_tables()
    if case == "schema":
        tables["temporal_monthly"] = tables["temporal_monthly"].drop(
            columns="wilson_upper"
        )
    else:
        frame = tables["temporal_monthly"]
        frame["positives"] = frame["positives"].astype("object")
        frame.at[1, "positives"] = -1
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))
    state_before = _matplotlib_state()
    _assert_validation_failure_without_allocation(tables, monkeypatch)
    _assert_matplotlib_state_unchanged(state_before)
    assert tuple(tmp_path.rglob("*")) == files_before


def test_hostile_rc_context_is_preserved_while_fixed_data_contracts_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_state = _matplotlib_state()
    monkeypatch.chdir(tmp_path)
    files_before = tuple(tmp_path.rglob("*"))
    with matplotlib.rc_context(
        {
            "axes.facecolor": "#A1B2C3",
            "axes.grid": True,
            "figure.dpi": 77.0,
            "font.size": 17.0,
        }
    ):
        tables = _valid_tables()
        bundle_snapshot = _snapshot_bundle(tables)
        hostile_state = _matplotlib_state()
        result = build_eda_figures(tables)
        _assert_matplotlib_state_unchanged(hostile_state)
        assert tuple(result) == _FIGURE_KEYS
        for key, figure in result.items():
            assert tuple(figure.get_size_inches()) == _EXPECTED_SIZES[key]
            assert figure.dpi == 120
        assert [
            bar.get_height()
            for bar in result["class_balance"].axes[0].patches
        ] == [80, 20]
        _assert_bundle_unchanged(tables, bundle_snapshot)
    _assert_matplotlib_state_unchanged(outer_state)
    assert tuple(tmp_path.rglob("*")) == files_before


def test_no_filesystem_side_effect_in_isolated_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.rglob("*"))
    result = build_eda_figures(_valid_tables())
    for figure in result.values():
        _render(figure)
    assert tuple(tmp_path.rglob("*")) == before


def test_build_function_accesses_only_five_approved_tables() -> None:
    tree = ast.parse(_FIGURES_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_eda_figures"
    )
    parents = {
        id(child): parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }
    validator_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_tables"
    ]
    assert len(validator_calls) == 1
    assert len(validator_calls[0].args) == 1
    assert isinstance(validator_calls[0].args[0], ast.Name)
    assert validator_calls[0].args[0].id == "tables"

    table_loads: list[tuple[int, str]] = []
    for name in ast.walk(function):
        if not isinstance(name, ast.Name) or name.id != "tables":
            continue
        parent = parents[id(name)]
        if isinstance(parent, ast.Subscript) and parent.value is name:
            assert isinstance(parent.slice, ast.Constant)
            assert isinstance(parent.slice.value, str)
            table_loads.append((parent.lineno, parent.slice.value))
            continue
        if parent is validator_calls[0]:
            continue
        pytest.fail(
            "the complete tables bundle escaped its validator or direct load: "
            f"{ast.dump(parent)}"
        )

    direct_keys = tuple(key for _, key in sorted(table_loads))
    assert direct_keys == _CONSUMED_TABLE_KEYS
    assert not set(direct_keys).intersection(_UNUSED_TABLE_KEYS)


def test_unused_tables_are_not_inspected_at_runtime() -> None:
    blocked_attributes = {
        "__getitem__",
        "__iter__",
        "columns",
        "dtypes",
        "index",
        "items",
        "iterrows",
        "itertuples",
        "shape",
        "values",
    }

    class TripwireFrame(pd.DataFrame):
        def __getattribute__(self, name: str) -> object:
            if name in blocked_attributes:
                raise AssertionError(f"unused table inspected through {name}")
            return super().__getattribute__(name)

    tables = _valid_tables()
    for position, key in enumerate(_UNUSED_TABLE_KEYS):
        tables[key] = TripwireFrame({"unused": [position]})
    result = build_eda_figures(tables)
    assert tuple(result) == _FIGURE_KEYS


def test_complete_module_owned_binding_surface() -> None:
    tree = ast.parse(_FIGURES_PATH.read_text(encoding="utf-8"))
    names = _module_owned_names(tree)
    assert names.count("build_eda_figures") == 1
    assert names.count("__all__") == 1
    assert all(
        name == "build_eda_figures"
        or name == "__all__"
        or name.startswith("_")
        for name in names
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("PUBLIC_VALUE = 1", ("PUBLIC_VALUE",)),
        (
            "PUBLIC_FIRST = PUBLIC_SECOND = 1",
            ("PUBLIC_FIRST", "PUBLIC_SECOND"),
        ),
        ("PUBLIC_VALUE: int = 1", ("PUBLIC_VALUE",)),
        ("PUBLIC_VALUE += 1", ("PUBLIC_VALUE",)),
        (
            "PUBLIC_REGISTRY, _private = ({}, None)",
            ("PUBLIC_REGISTRY", "_private"),
        ),
        (
            "[PUBLIC_LIST, _private] = [[], None]",
            ("PUBLIC_LIST", "_private"),
        ),
        (
            "*PUBLIC_STARRED, _private = ()",
            ("PUBLIC_STARRED", "_private"),
        ),
        ("class PublicFigureFactory:\n    pass", ("PublicFigureFactory",)),
        ("def public_helper():\n    pass", ("public_helper",)),
    ],
)
def test_module_binding_detector_covers_public_definition_forms(
    source: str,
    expected: tuple[str, ...],
) -> None:
    assert _module_owned_names(ast.parse(source)) == expected


def test_module_binding_detector_covers_type_alias_when_available() -> None:
    if not hasattr(ast, "TypeAlias"):
        pytest.skip("ast.TypeAlias requires Python 3.12+")
    tree = ast.parse("type PublicFigureMap = dict[str, object]")
    assert _module_owned_names(tree) == ("PublicFigureMap",)


def test_module_binding_detector_accepts_private_equivalents() -> None:
    source = """
_PRIVATE_VALUE = 1
_PRIVATE_FIRST = _PRIVATE_SECOND = 1
_PRIVATE_ANNOTATED: int = 1
_PRIVATE_VALUE += 1
_PRIVATE_REGISTRY, _private = ({}, None)
[_PRIVATE_LIST, _private] = [[], None]
*_PRIVATE_STARRED, _private = ()
class _PrivateFigureFactory:
    pass
def _private_helper():
    pass
"""
    if hasattr(ast, "TypeAlias"):
        source += "type _PrivateFigureMap = dict[str, object]\n"
    assert all(
        name.startswith("_")
        for name in _module_owned_names(ast.parse(source))
    )


def test_filesystem_writer_ast_guard() -> None:
    tree = ast.parse(_FIGURES_PATH.read_text(encoding="utf-8"))
    forbidden_modules = {"os", "pathlib", "tempfile", "shutil"}
    forbidden_names = {
        "open",
        "Path",
        "mkdir",
        "makedirs",
        "touch",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "replace",
        "save",
        "savez",
        "savefig",
        "savetxt",
        "print_png",
        "to_csv",
        "to_json",
        "to_parquet",
        "to_pickle",
        "to_excel",
        "imsave",
    }
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    executable_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    executable_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not imports.intersection(forbidden_modules)
    assert not executable_names.intersection(forbidden_names)
    assert not executable_attributes.intersection(forbidden_names)
    assert "__file__" not in executable_names


def test_matplotlib_global_state_ast_guard() -> None:
    tree = ast.parse(_FIGURES_PATH.read_text(encoding="utf-8"))
    assert _global_state_violations(tree) == ()


@pytest.mark.parametrize(
    "source",
    [
        'import matplotlib\nmatplotlib.use("Agg")',
        (
            "import matplotlib\n"
            'matplotlib.rcParams["figure.dpi"] = 72'
        ),
        (
            "import matplotlib\n"
            'matplotlib.rcParams.update({"font.size": 20})'
        ),
        (
            "from matplotlib import rcParams\n"
            'rcParams["axes.grid"] = True'
        ),
        (
            "from matplotlib import rcParams\n"
            'rcParams |= {"figure.dpi": 72}'
        ),
        (
            "from matplotlib._pylab_helpers import Gcf\n"
            "Gcf.set_active(manager)"
        ),
        (
            "from matplotlib._pylab_helpers import Gcf\n"
            "Gcf.figs[1] = manager"
        ),
        (
            "import matplotlib\n"
            "def _helper():\n"
            '    matplotlib.rcParams["font.size"] = 20'
        ),
        (
            "import matplotlib.pyplot\n"
            'matplotlib.pyplot.switch_backend("Agg")'
        ),
        (
            "from matplotlib import rc_context\n"
            'with rc_context({"font.size": 20}):\n'
            "    pass"
        ),
        (
            "import matplotlib\n"
            "matplotlib.backends.backend_registry.register(backend)"
        ),
    ],
)
def test_global_state_detector_flags_mutation_probes(source: str) -> None:
    assert _global_state_violations(ast.parse(source))


def test_global_state_detector_allows_read_only_rcparams_probe() -> None:
    source = (
        "import matplotlib\n"
        'value = matplotlib.rcParams["figure.dpi"]'
    )
    assert _global_state_violations(ast.parse(source)) == ()


def test_direct_filesystem_writers_are_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_writer(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"filesystem writer called: {args!r} {kwargs!r}")

    writer_methods = (
        (Figure, "savefig"),
        (FigureCanvasAgg, "print_png"),
        (pd.DataFrame, "to_csv"),
        (pd.DataFrame, "to_json"),
        (pd.DataFrame, "to_parquet"),
        (pd.DataFrame, "to_pickle"),
        (pd.DataFrame, "to_excel"),
        (Path, "write_text"),
        (Path, "write_bytes"),
        (Path, "touch"),
    )
    for owner, name in writer_methods:
        monkeypatch.setattr(owner, name, forbidden_writer)
    result = build_eda_figures(_valid_tables())
    assert tuple(result) == _FIGURE_KEYS


def test_static_dependency_and_leakage_guards() -> None:
    source = _FIGURES_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_text = (
        "build_eda_tables",
        "select_eda_populations",
        "summarize_",
        "matplotlib.pyplot",
        "savefig",
        "print_png",
        "pathlib",
        "tempfile",
        "mkdir",
        "feature_selection",
        "sklearn",
    )
    assert all(token not in source for token in forbidden_text)

    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imported_modules.intersection(
        {"os", "pathlib", "tempfile", "src.analysis.run_eda", "src.data"}
    )
    assert not imported_from.intersection(
        {
            "os",
            "pathlib",
            "tempfile",
            "src.analysis.run_eda",
            "src.analysis.drift",
            "src.analysis.relationships",
            "src.data",
        }
    )


def test_no_extra_public_api_or_defined_classes() -> None:
    tree = ast.parse(_FIGURES_PATH.read_text(encoding="utf-8"))
    public_functions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert public_functions == ["build_eda_figures"]
    assert classes == []


def test_import_time_has_no_figure_or_manager_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import reload
    from matplotlib._pylab_helpers import Gcf

    before = tuple(Gcf.get_all_fig_managers())
    reloaded = reload(figures)
    after = tuple(Gcf.get_all_fig_managers())
    assert reloaded is figures
    assert after == before
    assert not any(
        isinstance(value, Figure)
        for value in vars(reloaded).values()
    )


def test_display_text_has_no_threshold_or_recommendation_language() -> None:
    result = build_eda_figures(_valid_tables())
    display_text = " ".join(
        text.get_text()
        for figure in result.values()
        for axis in figure.axes
        for text in axis.texts
    ).lower()
    assert "threshold" not in display_text
    assert "significant" not in display_text
    assert "remove" not in display_text
    assert "recommend" not in display_text


def test_module_import_does_not_register_pyplot() -> None:
    source = _FIGURES_PATH.read_text(encoding="utf-8")
    assert "pyplot" not in source
    assert "plt" not in {
        name for name in vars(figures) if not name.startswith("__")
    }


def test_authentic_source_tables_are_not_returned(
    authentic_tables: dict[str, pd.DataFrame],
) -> None:
    result = build_eda_figures(authentic_tables)
    source_ids = {id(frame) for frame in authentic_tables.values()}
    assert all(id(value) not in source_ids for value in result.values())
    assert all(isinstance(value, Figure) for value in result.values())


def test_source_indexes_do_not_appear_in_figure_text() -> None:
    tables = _valid_tables()
    for frame in tables.values():
        frame.index = pd.Index(
            [f"secret-index-{position}" for position in range(len(frame))]
        )
    result = build_eda_figures(tables)
    all_text = " ".join(
        text.get_text()
        for figure in result.values()
        for axis in figure.axes
        for text in axis.texts
    )
    assert "secret-index" not in all_text


def test_python_runtime_matches_required_major_minor() -> None:
    assert sys.version_info[:2] == (3, 12)
