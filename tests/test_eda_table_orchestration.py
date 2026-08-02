"""Tests for deterministic in-memory EDA table orchestration."""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from src.analysis import run_eda
from src.analysis.drift import (
    summarize_categorical_drift_features,
    summarize_categorical_drift_levels,
    summarize_numeric_drift,
)
from src.analysis.relationships import summarize_numeric_relationships
from src.analysis.run_eda import build_eda_tables, select_eda_populations
from src.analysis.summaries import (
    summarize_categorical_features,
    summarize_cohort_target,
    summarize_missingness,
    summarize_numeric_by_target,
    summarize_numeric_features,
    summarize_temporal_coverage,
    summarize_temporal_monthly,
)
from src.data import build_dataset as bd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
EXPECTED_KEYS = (
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
EXPECTED_CATEGORICAL_FEATURES = (
    "visit_type",
    "booking_channel",
    "scheduled_weekday",
    "scheduled_hour",
    "scheduled_month",
)
EXPECTED_CATEGORICAL_UNIVERSES = {
    "visit_type": (
        "consultation",
        "emergency",
        "follow_up",
        "new_patient_examination",
        "recall_examination",
        "treatment",
        "<MISSING>",
    ),
    "booking_channel": (
        "in_person",
        "online",
        "other",
        "phone",
        "referral",
        "<MISSING>",
    ),
    "scheduled_weekday": (
        *tuple(str(value) for value in range(7)),
        "<MISSING>",
    ),
    "scheduled_hour": (
        *tuple(str(value) for value in range(24)),
        "<MISSING>",
    ),
    "scheduled_month": (
        *tuple(str(value) for value in range(1, 13)),
        "<MISSING>",
    ),
}
EXPECTED_CALL_TARGETS = (
    "select_eda_populations",
    "summarize_cohort_target",
    "summarize_missingness",
    "summarize_numeric_features",
    "summarize_numeric_by_target",
    "summarize_categorical_features",
    "summarize_temporal_coverage",
    "summarize_temporal_monthly",
    "summarize_numeric_drift",
    "summarize_categorical_drift_levels",
    "summarize_categorical_drift_features",
    "summarize_numeric_relationships",
)
SUMMARY_ROUTING = (
    ("summarize_cohort_target", "cohort_target", ("supervised_train",)),
    ("summarize_missingness", "missingness", ("supervised_train",)),
    ("summarize_numeric_features", "numeric_features", ("supervised_train",)),
    (
        "summarize_numeric_by_target",
        "numeric_by_target",
        ("supervised_train",),
    ),
    (
        "summarize_categorical_features",
        "categorical_features",
        ("supervised_train",),
    ),
    (
        "summarize_temporal_coverage",
        "temporal_coverage",
        ("supervised_train", "maturity_audit"),
    ),
    (
        "summarize_temporal_monthly",
        "temporal_monthly",
        ("supervised_train", "maturity_audit"),
    ),
    (
        "summarize_numeric_drift",
        "numeric_drift",
        ("train_drift", "validation_drift"),
    ),
    (
        "summarize_categorical_drift_levels",
        "categorical_drift_levels",
        ("train_drift", "validation_drift"),
    ),
    (
        "summarize_categorical_drift_features",
        "categorical_drift_features",
        ("train_drift", "validation_drift"),
    ),
    (
        "summarize_numeric_relationships",
        "numeric_relationships",
        ("train_drift",),
    ),
)


@pytest.fixture(scope="session")
def canonical_dataset() -> pd.DataFrame:
    tables = bd.load_raw_data(RAW_DIR)
    return bd.build_analytical_dataset(tables)


def _sentinel_populations() -> dict[str, pd.DataFrame]:
    return {
        name: pd.DataFrame({"population": [name]})
        for name in (
            "supervised_train",
            "train_drift",
            "validation_drift",
            "maturity_audit",
        )
    }


def _install_routing_spies(
    monkeypatch: pytest.MonkeyPatch,
    canonical: pd.DataFrame,
    *,
    raise_at: str | None = None,
) -> tuple[
    list[str],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    Exception | None,
]:
    calls: list[str] = []
    populations = _sentinel_populations()
    outputs = {
        key: pd.DataFrame({"table": [key]})
        for key in EXPECTED_KEYS
    }

    def select(candidate: pd.DataFrame) -> dict[str, pd.DataFrame]:
        assert candidate is canonical
        calls.append("select_eda_populations")
        return populations

    monkeypatch.setattr(run_eda, "select_eda_populations", select)
    raised_exception: Exception | None = None
    if raise_at is not None:
        raised_exception = type(
            "UniqueDownstreamError",
            (Exception,),
            {},
        )("downstream sentinel")
    for function_name, key, population_names in SUMMARY_ROUTING:
        expected = tuple(populations[name] for name in population_names)

        def summary(
            *args: pd.DataFrame,
            expected: tuple[pd.DataFrame, ...] = expected,
            key: str = key,
            raised_exception: Exception | None = raised_exception,
        ) -> pd.DataFrame:
            assert len(args) == len(expected)
            assert all(actual is wanted for actual, wanted in zip(args, expected))
            calls.append(key)
            if key == raise_at:
                assert raised_exception is not None
                raise raised_exception
            return outputs[key]

        monkeypatch.setattr(run_eda, function_name, summary)
    return calls, populations, outputs, raised_exception


def _direct_tables(canonical: pd.DataFrame) -> dict[str, pd.DataFrame]:
    populations = select_eda_populations(canonical)
    supervised = populations["supervised_train"]
    train = populations["train_drift"]
    validation = populations["validation_drift"]
    audit = populations["maturity_audit"]
    return {
        "cohort_target": summarize_cohort_target(supervised),
        "missingness": summarize_missingness(supervised),
        "numeric_features": summarize_numeric_features(supervised),
        "numeric_by_target": summarize_numeric_by_target(supervised),
        "categorical_features": summarize_categorical_features(supervised),
        "temporal_coverage": summarize_temporal_coverage(supervised, audit),
        "temporal_monthly": summarize_temporal_monthly(supervised, audit),
        "numeric_drift": summarize_numeric_drift(train, validation),
        "categorical_drift_levels": summarize_categorical_drift_levels(
            train,
            validation,
        ),
        "categorical_drift_features": summarize_categorical_drift_features(
            train,
            validation,
        ),
        "numeric_relationships": summarize_numeric_relationships(train),
    }


def _build_eda_tables_ast() -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(build_eda_tables))
    module = ast.parse(source)
    function_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]
    assert len(function_nodes) == 1
    return function_nodes[0]


def test_public_signature() -> None:
    signature = inspect.signature(build_eda_tables)
    hints = get_type_hints(build_eda_tables)
    assert tuple(signature.parameters) == ("canonical",)
    assert hints["canonical"] is pd.DataFrame
    assert hints["return"] == dict[str, pd.DataFrame]


def test_canonical_has_exactly_one_approved_ast_load() -> None:
    function = _build_eda_tables_ast()
    parents = {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }
    canonical_loads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and node.id == "canonical"
        and isinstance(node.ctx, ast.Load)
    ]
    selector_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "select_eda_populations"
    ]
    assert len(canonical_loads) == 1
    assert len(selector_calls) == 1
    canonical_load = canonical_loads[0]
    selector_call = selector_calls[0]
    assert selector_call.args == [canonical_load]
    assert selector_call.keywords == []
    assert parents[canonical_load] is selector_call


def test_orchestration_direct_call_targets_match_exact_allowlist() -> None:
    function = _build_eda_tables_ast()
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert all(isinstance(call.func, ast.Name) for call in calls)
    call_targets = [call.func.id for call in calls if isinstance(call.func, ast.Name)]
    assert Counter(call_targets) == Counter(EXPECTED_CALL_TARGETS)
    assert len(call_targets) == len(EXPECTED_CALL_TARGETS)


def test_exact_selector_count_call_order_routing_and_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = pd.DataFrame({"canonical": [True]})
    calls, populations, outputs, raised_exception = _install_routing_spies(
        monkeypatch,
        canonical,
    )
    result = build_eda_tables(canonical)

    assert raised_exception is None
    assert calls == ["select_eda_populations", *EXPECTED_KEYS]
    assert calls.count("select_eda_populations") == 1
    assert tuple(result) == EXPECTED_KEYS
    assert all(result[key] is outputs[key] for key in EXPECTED_KEYS)
    assert len({id(frame) for frame in result.values()}) == len(EXPECTED_KEYS)
    assert set(populations) == {
        "supervised_train",
        "train_drift",
        "validation_drift",
        "maturity_audit",
    }


def test_controlled_execution_creates_no_files_or_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = pd.DataFrame({"canonical": [True]})
    calls, _, _, _ = _install_routing_spies(monkeypatch, canonical)

    def directory_contents() -> tuple[str, ...]:
        return tuple(
            sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
        )

    before = directory_contents()
    monkeypatch.chdir(tmp_path)
    result = build_eda_tables(canonical)
    after = directory_contents()

    assert before == ()
    assert after == before
    assert calls == ["select_eda_populations", *EXPECTED_KEYS]
    assert tuple(result) == EXPECTED_KEYS


@pytest.mark.parametrize(
    "failure_key",
    ("temporal_monthly", "numeric_relationships"),
)
def test_downstream_exception_identity_propagates_and_stops_later_calls(
    monkeypatch: pytest.MonkeyPatch,
    failure_key: str,
) -> None:
    canonical = pd.DataFrame({"canonical": [True]})
    canonical_before = canonical.copy(deep=True)
    calls, _, _, raised_exception = _install_routing_spies(
        monkeypatch,
        canonical,
        raise_at=failure_key,
    )
    assert raised_exception is not None
    with pytest.raises(type(raised_exception)) as caught:
        build_eda_tables(canonical)
    assert caught.value is raised_exception
    assert calls == [
        "select_eda_populations",
        *EXPECTED_KEYS[: EXPECTED_KEYS.index(failure_key) + 1],
    ]
    assert calls.count("select_eda_populations") == 1
    pd.testing.assert_frame_equal(canonical, canonical_before)


def test_selector_exception_propagates_before_any_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UniqueSelectorError(Exception):
        pass

    canonical = pd.DataFrame({"canonical": [True]})
    summary_calls: list[str] = []

    def fail_selector(candidate: pd.DataFrame) -> dict[str, pd.DataFrame]:
        assert candidate is canonical
        raise UniqueSelectorError("selector sentinel")

    def record_summary(*args: pd.DataFrame) -> pd.DataFrame:
        summary_calls.append("unexpected")
        return pd.DataFrame()

    monkeypatch.setattr(run_eda, "select_eda_populations", fail_selector)
    for function_name, _, _ in SUMMARY_ROUTING:
        monkeypatch.setattr(run_eda, function_name, record_summary)
    with pytest.raises(UniqueSelectorError, match="selector sentinel"):
        build_eda_tables(canonical)
    assert summary_calls == []


def test_authentic_tables_equal_direct_calls(
    canonical_dataset: pd.DataFrame,
) -> None:
    actual = build_eda_tables(canonical_dataset)
    expected = _direct_tables(canonical_dataset)
    assert tuple(actual) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        pd.testing.assert_frame_equal(actual[key], expected[key])


def test_authentic_core_reconciliation(canonical_dataset: pd.DataFrame) -> None:
    tables = build_eda_tables(canonical_dataset)
    assert len(tables["cohort_target"]) == 1
    assert len(tables["missingness"]) == 10
    assert len(tables["numeric_features"]) == 5
    assert len(tables["numeric_by_target"]) == 10
    categorical = tables["categorical_features"]
    assert tuple(categorical["feature"].drop_duplicates()) == (
        EXPECTED_CATEGORICAL_FEATURES
    )

    coverage = tables["temporal_coverage"].iloc[0]
    assert coverage["nominal_train_rows"] == 3_682
    assert coverage["mature_train_rows"] == 3_670
    assert coverage["maturity_exclusion_rows"] == 12
    monthly = tables["temporal_monthly"]
    assert len(monthly) == 12
    assert monthly["nominal_train_count"].sum() == 3_682
    assert monthly["mature_train_count"].sum() == 3_670
    assert monthly["positives"].sum() == 432
    assert monthly["negatives"].sum() == 3_238

    numeric_drift = tables["numeric_drift"]
    assert len(numeric_drift) == 5
    assert numeric_drift["train_rows"].eq(3_670).all()
    assert numeric_drift["validation_rows"].eq(1_541).all()
    categorical_drift = tables["categorical_drift_features"]
    assert len(categorical_drift) == 5
    assert tuple(categorical_drift["feature"]) == EXPECTED_CATEGORICAL_FEATURES
    assert categorical_drift["train_rows"].eq(3_670).all()
    assert categorical_drift["validation_rows"].eq(1_541).all()
    relationships_table = tables["numeric_relationships"]
    assert len(relationships_table) == 10
    assert relationships_table["train_rows"].eq(3_670).all()
    assert relationships_table["paired_n"].eq(3_670).all()

    levels = tables["categorical_drift_levels"]
    for feature in EXPECTED_CATEGORICAL_FEATURES:
        feature_levels = levels.loc[levels["feature"].eq(feature)]
        assert tuple(feature_levels["level"]) == EXPECTED_CATEGORICAL_UNIVERSES[
            feature
        ]
        assert feature_levels["is_missing"].tolist() == [
            *[False] * (len(EXPECTED_CATEGORICAL_UNIVERSES[feature]) - 1),
            True,
        ]
        assert feature_levels["train_count"].sum() == 3_670
        assert feature_levels["validation_count"].sum() == 1_541


def test_canonical_is_not_mutated_and_repeated_calls_are_fresh(
    canonical_dataset: pd.DataFrame,
) -> None:
    canonical = canonical_dataset.copy(deep=True)
    before = canonical.copy(deep=True)
    first = build_eda_tables(canonical)
    second = build_eda_tables(canonical)

    pd.testing.assert_frame_equal(canonical, before)
    assert first is not second
    assert len({id(frame) for frame in first.values()}) == len(first)
    for key in EXPECTED_KEYS:
        assert first[key] is not second[key]
        pd.testing.assert_frame_equal(first[key], second[key])


def test_output_mutation_is_isolated_from_input_other_tables_and_later_calls(
    canonical_dataset: pd.DataFrame,
) -> None:
    canonical = canonical_dataset.copy(deep=True)
    before = canonical.copy(deep=True)
    result = build_eda_tables(canonical)
    missingness_before = result["missingness"].copy(deep=True)
    expected_later = build_eda_tables(canonical)

    result["cohort_target"].iloc[0, 0] = -1
    result["cohort_target"]["new_column"] = "changed"

    pd.testing.assert_frame_equal(canonical, before)
    pd.testing.assert_frame_equal(result["missingness"], missingness_before)
    later = build_eda_tables(canonical)
    for key in EXPECTED_KEYS:
        pd.testing.assert_frame_equal(later[key], expected_later[key])


def test_validation_and_test_target_poisoning_does_not_change_tables(
    canonical_dataset: pd.DataFrame,
) -> None:
    baseline = build_eda_tables(canonical_dataset)
    poisoned = canonical_dataset.copy(deep=True)
    poison_mask = poisoned["split"].isin(("validation", "test"))
    poisoned.loc[poison_mask, "target"] = 1 - poisoned.loc[
        poison_mask,
        "target",
    ]
    changed = build_eda_tables(poisoned)
    for key in EXPECTED_KEYS:
        pd.testing.assert_frame_equal(changed[key], baseline[key])


def test_empty_canonical_propagates_existing_selector_error(
    canonical_dataset: pd.DataFrame,
) -> None:
    empty = canonical_dataset.iloc[0:0].copy(deep=True)
    with pytest.raises(ValueError):
        build_eda_tables(empty)


def _malformed_canonical(canonical: pd.DataFrame, case: str) -> pd.DataFrame:
    malformed = canonical.copy(deep=True)
    if case == "missing_column":
        return malformed.drop(columns=[malformed.columns[-1]])
    if case == "reordered_schema":
        columns = list(malformed.columns)
        columns[0], columns[1] = columns[1], columns[0]
        return malformed.loc[:, columns]
    if case == "duplicate_id":
        malformed.loc[malformed.index[1], "appointment_id"] = malformed.iloc[0][
            "appointment_id"
        ]
    elif case == "invalid_timestamp":
        malformed["prediction_time"] = malformed["prediction_time"].astype("str")
    elif case == "invalid_target":
        malformed.loc[malformed.index[0], "target"] = 2
    else:
        raise AssertionError(f"unknown malformed case: {case}")
    return malformed


@pytest.mark.parametrize(
    "case",
    (
        "missing_column",
        "reordered_schema",
        "duplicate_id",
        "invalid_timestamp",
        "invalid_target",
    ),
)
def test_malformed_canonical_errors_propagate_without_input_mutation(
    canonical_dataset: pd.DataFrame,
    case: str,
) -> None:
    malformed = _malformed_canonical(canonical_dataset, case)
    before = malformed.copy(deep=True)
    with pytest.raises((TypeError, ValueError)):
        build_eda_tables(malformed)
    pd.testing.assert_frame_equal(malformed, before)


def test_orchestration_ast_has_no_leakage_or_side_effect_operations() -> None:
    source = textwrap.dedent(inspect.getsource(build_eda_tables))
    tree = ast.parse(source)
    forbidden_names = {
        "select_test_rows",
        "select_model_features",
        "select_development_rows",
        "build_canonical_dataset",
        "open",
        "Path",
    }
    forbidden_attributes = {
        "target",
        "to_csv",
        "to_json",
        "to_parquet",
        "to_pickle",
        "savefig",
        "plot",
    }
    forbidden_subscripts = {
        "target",
        "test",
        "test_drift",
        "pretest_fit_eligible",
    }
    assert not {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } & forbidden_names
    assert not {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } & forbidden_attributes
    string_subscripts = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert not string_subscripts & forbidden_subscripts
    assert "try:" not in source
    assert "except" not in source
