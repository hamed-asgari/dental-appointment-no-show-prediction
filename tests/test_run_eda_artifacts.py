"""End-to-end contracts for the bounded EDA artifact runner."""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
from importlib import reload, util
import inspect
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from typing import get_type_hints
import warnings

import matplotlib
from matplotlib._pylab_helpers import Gcf
import numpy as np
import pandas as pd
import pytest

from src.analysis import artifacts, figure_artifacts, run_eda
from src.analysis.run_eda import build_eda_tables, generate_eda_artifacts


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPOSITORY_ROOT / "src" / "analysis" / "run_eda.py"
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw"
_PROCESSED_DIR = _REPOSITORY_ROOT / "data" / "processed"
_EXPECTED_RAW_TREE = (
    (
        'README.md',
        'file',
        3_523,
        'c688f0bd2dd2bd77e78916abeea89532bea2746be1a590cba0493a3ece698be6',
        None,
    ),
    (
        'appointments.csv',
        'file',
        1_410_520,
        '4f3736f78cda615d1401d3f639b5e29e47781a1ae1c820c1e6f248eae57a00df',
        None,
    ),
    (
        'dentists.csv',
        'file',
        468,
        'bf83d1848236e8f5fc8ee5ef3bb21fec2690f85c3c2f259840c16c271a00ab47',
        None,
    ),
    (
        'patients.csv',
        'file',
        191_039,
        'e416843a80568a91455e5cff872bbca5b49be16f109022d56c687cdf2683cc69',
        None,
    ),
    (
        'v2',
        'directory',
        None,
        None,
        None,
    ),
    (
        'v2/README.md',
        'file',
        1_031,
        '3e4cf20dc7fb9fb523a202146298d9f520291d1a4976497fc12de1fe014a4d42',
        None,
    ),
    (
        'v2/appointments.csv',
        'file',
        4_723_397,
        '00d759e69fa51eb5250fafb07e844a7c7ba0cb16dec2b80de47ce78092a162ba',
        None,
    ),
    (
        'v2/dentists.csv',
        'file',
        460,
        '22426232d2fa4051ebe1484b5b495ed5afc2d51e2f5c3f19c654aa1f61cad5e8',
        None,
    ),
    (
        'v2/patients.csv',
        'file',
        377_003,
        '37df61e47d4060f1e92af49de224228e02451c9eb72e2820f037285ffa9a8ad6',
        None,
    ),
    (
        'v2/v2_synthetic_benchmark.manifest.json',
        'file',
        5_169,
        '7702fa5fa0638c52dd0598e28f35f678fb5d61a886faadf9b38a6e292fdcd561',
        None,
    ),
)
_EXPECTED_PROCESSED_TREE = (
    (
        ".gitkeep",
        "file",
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        None,
    ),
)
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
_EXPECTED_SPLIT_COUNTS = {
    "train": 3_682,
    "validation": 1_541,
    "test": 1_563,
}
_EXPECTED_SPLIT_POSITIVES = {
    "train": 434,
    "validation": 192,
    "test": 194,
}
_EXPECTED_TOTAL_ROWS = 6_786
_EXPECTED_POSITIVES = 820
_EXPECTED_NEGATIVES = 5_966
_EXPECTED_MATURITY_COUNTS = {
    "development_fit_eligible": 3_670,
    "pretest_fit_eligible": 5_223,
}
_EXPECTED_TABLE_ARTIFACTS = (
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
_EXPECTED_FIGURE_ARTIFACTS = (
    ("class_balance", "01_class_balance.png"),
    ("temporal_monthly", "02_temporal_monthly.png"),
    ("numeric_drift", "03_numeric_drift.png"),
    ("categorical_drift", "04_categorical_drift.png"),
    ("numeric_relationships", "05_numeric_relationships.png"),
)
_EXPECTED_HEADERS = {
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
_EXPECTED_DIMENSIONS = {
    "01_class_balance.png": (960, 540),
    "02_temporal_monthly.png": (1200, 660),
    "03_numeric_drift.png": (1080, 576),
    "04_categorical_drift.png": (1080, 576),
    "05_numeric_relationships.png": (960, 840),
}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_snapshot(
    root: Path,
) -> tuple[tuple[str, str, int | None, str | None, str | None], ...]:
    entries: list[tuple[str, str, int | None, str | None, str | None]] = []

    def walk(directory: Path, relative: Path) -> None:
        with os.scandir(directory) as children:
            ordered = sorted(children, key=lambda child: child.name)
        for child in ordered:
            child_relative = relative / child.name
            relative_name = child_relative.as_posix()
            if child.is_symlink():
                entries.append(
                    (relative_name, "symlink", None, None, os.readlink(child.path))
                )
            elif child.is_dir(follow_symlinks=False):
                entries.append((relative_name, "directory", None, None, None))
                walk(Path(child.path), child_relative)
            else:
                stat = child.stat(follow_symlinks=False)
                entries.append(
                    (
                        relative_name,
                        "file",
                        stat.st_size,
                        _file_sha256(Path(child.path)),
                        None,
                    )
                )

    walk(root, Path())
    return tuple(entries)


def _protected_tree_snapshot(
) -> tuple[
    tuple[tuple[str, str, int | None, str | None, str | None], ...],
    tuple[tuple[str, str, int | None, str | None, str | None], ...],
]:
    return _tree_snapshot(_RAW_DIR), _tree_snapshot(_PROCESSED_DIR)


def _assert_protected_trees_unchanged(
    before: tuple[
        tuple[tuple[str, str, int | None, str | None, str | None], ...],
        tuple[tuple[str, str, int | None, str | None, str | None], ...],
    ],
) -> None:
    assert _protected_tree_snapshot() == before


def _stable_state_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return _stable_state_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return ("float", "nan")
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        return tuple(
            (
                _stable_state_value(key),
                _stable_state_value(item),
            )
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_stable_state_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        stable = (_stable_state_value(item) for item in value)
        return tuple(sorted(stable, key=repr))
    return repr(value)


def _stable_mapping(mapping: object) -> object:
    copied = copy.deepcopy(dict(mapping))  # type: ignore[arg-type]
    return _stable_state_value(copied)


def _process_state_snapshot() -> dict[str, object]:
    return {
        "cwd": Path.cwd(),
        "environment": tuple(sorted(os.environ.items())),
        "warnings": _stable_state_value(copy.deepcopy(warnings.filters)),
        "python_random": _stable_state_value(random.getstate()),
        "numpy_random": _stable_state_value(np.random.get_state()),
        "backend": matplotlib.get_backend(),
        "rcParams": _stable_mapping(matplotlib.rcParams),
        "rcParamsDefault": _stable_mapping(matplotlib.rcParamsDefault),
        "rcParamsOrig": _stable_mapping(matplotlib.rcParamsOrig),
        "gcf_managers": tuple(id(manager) for manager in Gcf.get_all_fig_managers()),
    }


def _assert_process_state_unchanged(before: dict[str, object]) -> None:
    after = _process_state_snapshot()
    for key, expected in before.items():
        assert after[key] == expected, f"process state changed: {key}"


def _instrument_pre_writer_mutations(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_writers: bool,
) -> list[str]:
    mutations: list[str] = []

    def blocked(name: str):
        def reject(*args: object, **kwargs: object) -> None:
            mutations.append(name)
            raise AssertionError(f"unexpected filesystem mutation: {name}")

        return reject

    monkeypatch.setattr(Path, "mkdir", blocked("Path.mkdir"))
    monkeypatch.setattr(
        artifacts.tempfile,
        "NamedTemporaryFile",
        blocked("temporary-file creation"),
    )
    monkeypatch.setattr(artifacts.os, "replace", blocked("os.replace"))
    monkeypatch.setattr(
        artifacts,
        "_write_atomic",
        blocked("table atomic write"),
    )
    monkeypatch.setattr(
        figure_artifacts,
        "_write_atomic",
        blocked("figure atomic write"),
    )
    if include_writers:
        monkeypatch.setattr(
            run_eda,
            "write_eda_tables",
            blocked("write_eda_tables"),
        )
        monkeypatch.setattr(
            run_eda,
            "write_eda_figures",
            blocked("write_eda_figures"),
        )
    return mutations


def _assert_exact_parser_surface(parser: object) -> None:
    actions = tuple(parser._actions)  # type: ignore[attr-defined]
    assert len(actions) == 2
    help_action, output_action = actions
    assert tuple(help_action.option_strings) == ("-h", "--help")
    assert help_action.dest == "help"
    assert tuple(output_action.option_strings) == ("--output-dir",)
    assert output_action.dest == "output_dir"
    assert output_action.type is Path
    assert output_action.default == Path("reports/eda")
    assert output_action.required is False
    assert output_action.choices is None
    assert output_action.const is None
    assert output_action.nargs is None
    assert all(action.option_strings for action in actions)


def _artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload.startswith(_PNG_SIGNATURE)
    assert payload[12:16] == b"IHDR"
    assert payload.endswith(b"IEND\xaeB`\x82")
    return struct.unpack(">II", payload[16:24])


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.asname is None:
                    root = imported.name.split(".", maxsplit=1)[0]
                    aliases[root] = root
                else:
                    aliases[imported.asname] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for imported in node.names:
                local = imported.asname or imported.name
                aliases[local] = f"{node.module}.{imported.name}"
    return aliases


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value, aliases)
        return f"{owner}.{node.attr}" if owner else node.attr
    if isinstance(node, ast.Subscript):
        return _qualified_name(node.value, aliases)
    return ""


def _assignment_targets(node: ast.AST) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.Assign):
        return tuple(node.targets)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        return (node.target,)
    if isinstance(node, ast.Delete):
        return tuple(node.targets)
    return ()


def _public_module_owned_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_target_names(node.target))
        elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
            names.update(_target_names(node.name))
    return {name for name in names if not name.startswith("_")}


def _install_success_spies(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
) -> tuple[list[str], dict[str, object]]:
    calls: list[str] = []
    objects: dict[str, object] = {
        "canonical": object(),
        "tables": {"table": object()},
        "figures": {"figure": object()},
        "table_paths": {"table": output_dir / "table.csv"},
        "figure_paths": {"figure": output_dir / "figure.png"},
    }

    def build_canonical() -> object:
        calls.append("canonical")
        return objects["canonical"]

    def build_tables(canonical: object) -> object:
        assert canonical is objects["canonical"]
        calls.append("tables")
        return objects["tables"]

    def build_figures(tables: object) -> object:
        assert tables is objects["tables"]
        calls.append("figures")
        return objects["figures"]

    def write_tables(tables: object, directory: Path) -> object:
        assert tables is objects["tables"]
        assert directory == output_dir
        objects["output_path"] = directory
        calls.append("write_tables")
        return objects["table_paths"]

    def write_figures(figures: object, directory: Path) -> object:
        assert figures is objects["figures"]
        assert directory is objects["output_path"]
        calls.append("write_figures")
        return objects["figure_paths"]

    monkeypatch.setattr(run_eda, "_build_canonical_dataset", build_canonical)
    monkeypatch.setattr(run_eda, "build_eda_tables", build_tables)
    monkeypatch.setattr(run_eda, "build_eda_figures", build_figures)
    monkeypatch.setattr(run_eda, "write_eda_tables", write_tables)
    monkeypatch.setattr(run_eda, "write_eda_figures", write_figures)
    return calls, objects


@pytest.fixture(scope="module")
def _authentic_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, dict[str, Path]]]:
    output_dir = tmp_path_factory.mktemp("authentic-runner") / "eda"
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    result = generate_eda_artifacts(output_dir)
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)
    return output_dir, result


def test_generate_eda_artifacts_has_exact_signature() -> None:
    signature = inspect.signature(generate_eda_artifacts)
    assert tuple(signature.parameters) == ("output_dir",)
    parameter = signature.parameters["output_dir"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    assert get_type_hints(generate_eda_artifacts) == {
        "output_dir": str | Path,
        "return": dict[str, dict[str, Path]],
    }


def test_build_eda_tables_signature_is_preserved() -> None:
    signature = inspect.signature(build_eda_tables)
    assert tuple(signature.parameters) == ("canonical",)
    parameter = signature.parameters["canonical"]
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameter.default is inspect.Parameter.empty
    assert get_type_hints(build_eda_tables) == {
        "canonical": pd.DataFrame,
        "return": dict[str, pd.DataFrame],
    }


def test_exact_public_exports_and_wildcard_control() -> None:
    assert run_eda.__all__ == (
        "build_eda_tables",
        "generate_eda_artifacts",
    )
    namespace: dict[str, object] = {}
    exec("from src.analysis.run_eda import *", {}, namespace)
    namespace.pop("__builtins__", None)
    assert namespace == {
        "build_eda_tables": build_eda_tables,
        "generate_eda_artifacts": generate_eda_artifacts,
    }


def test_no_new_public_module_owned_binding_or_public_main() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    assert _public_module_owned_names(tree) == {
        "select_eda_populations",
        "build_eda_tables",
        "generate_eda_artifacts",
    }
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
        for node in tree.body
    )


def test_protected_tree_snapshots_match_exact_test_owned_literals() -> None:
    assert _tree_snapshot(_RAW_DIR) == _EXPECTED_RAW_TREE
    assert _tree_snapshot(_PROCESSED_DIR) == _EXPECTED_PROCESSED_TREE


def test_phase_five_builder_is_reused_with_verified_raw_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    raw_tables = object()
    canonical = object()

    def validate(path: Path) -> dict[str, str]:
        calls.append(("validate", path))
        return {}

    def load(path: Path) -> object:
        calls.append(("load", path))
        return raw_tables

    def build(tables: object) -> object:
        calls.append(("build", tables))
        return canonical

    monkeypatch.setattr(run_eda.bd, "validate_raw_hashes", validate)
    monkeypatch.setattr(run_eda.bd, "load_raw_data", load)
    monkeypatch.setattr(run_eda.bd, "build_analytical_dataset", build)
    assert run_eda._build_canonical_dataset() is canonical
    assert calls == [
        ("validate", run_eda._RAW_DIR),
        ("load", run_eda._RAW_DIR),
        ("build", raw_tables),
    ]


@pytest.mark.parametrize(
    "failure_stage",
    ("validate_raw_hashes", "load_raw_data", "build_analytical_dataset"),
)
def test_real_canonical_stage_failures_stop_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    output_dir = tmp_path / failure_stage
    failure = RuntimeError(f"unique {failure_stage} failure")
    raw_tables = object()
    calls: list[str] = []
    downstream_calls: list[str] = []

    def validate(path: Path) -> dict[str, str]:
        assert path == run_eda._RAW_DIR
        calls.append("validate_raw_hashes")
        if failure_stage == "validate_raw_hashes":
            raise failure
        return {}

    def load(path: Path) -> object:
        assert path == run_eda._RAW_DIR
        calls.append("load_raw_data")
        if failure_stage == "load_raw_data":
            raise failure
        return raw_tables

    def build(tables: object) -> object:
        assert tables is raw_tables
        calls.append("build_analytical_dataset")
        if failure_stage == "build_analytical_dataset":
            raise failure
        return object()

    def downstream(name: str):
        def reject(*args: object, **kwargs: object) -> None:
            downstream_calls.append(name)
            raise AssertionError(f"unexpected downstream call: {name}")

        return reject

    monkeypatch.setattr(run_eda.bd, "validate_raw_hashes", validate)
    monkeypatch.setattr(run_eda.bd, "load_raw_data", load)
    monkeypatch.setattr(run_eda.bd, "build_analytical_dataset", build)
    monkeypatch.setattr(
        run_eda,
        "build_eda_tables",
        downstream("build_eda_tables"),
    )
    monkeypatch.setattr(
        run_eda,
        "build_eda_figures",
        downstream("build_eda_figures"),
    )
    mutations = _instrument_pre_writer_mutations(
        monkeypatch,
        include_writers=True,
    )
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    no_result = object()
    result: object = no_result
    caught: BaseException | None = None
    try:
        result = generate_eda_artifacts(output_dir)
    except BaseException as error:
        caught = error
    assert caught is failure
    assert result is no_result
    expected_calls = {
        "validate_raw_hashes": ("validate_raw_hashes",),
        "load_raw_data": ("validate_raw_hashes", "load_raw_data"),
        "build_analytical_dataset": (
            "validate_raw_hashes",
            "load_raw_data",
            "build_analytical_dataset",
        ),
    }
    assert tuple(calls) == expected_calls[failure_stage]
    assert downstream_calls == []
    assert mutations == []
    assert not output_dir.exists()
    _assert_protected_trees_unchanged(trees_before)
    _assert_process_state_unchanged(state_before)


def test_authentic_builder_pins_canonical_contract_and_order() -> None:
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    run_eda.bd.validate_raw_hashes(_RAW_DIR)
    raw_tables = run_eda.bd.load_raw_data(_RAW_DIR)
    canonical = run_eda.bd.build_analytical_dataset(raw_tables)
    run_eda.bd.validate_raw_hashes(_RAW_DIR)
    repeated_raw = run_eda.bd.load_raw_data(_RAW_DIR)
    repeated = run_eda.bd.build_analytical_dataset(repeated_raw)
    assert tuple(canonical.columns) == _EXPECTED_CANONICAL_COLUMNS
    assert len(canonical.columns) == 18
    assert canonical.shape == (6_786, 18)
    assert len(canonical) == _EXPECTED_TOTAL_ROWS
    assert int(canonical["target"].sum()) == _EXPECTED_POSITIVES
    assert int(canonical["target"].eq(0).sum()) == _EXPECTED_NEGATIVES
    assert set(canonical["split"].unique()) == set(_EXPECTED_SPLIT_COUNTS)
    for split_name, expected_rows in _EXPECTED_SPLIT_COUNTS.items():
        rows = canonical["split"].eq(split_name)
        assert int(rows.sum()) == expected_rows
        assert int(canonical.loc[rows, "target"].sum()) == (
            _EXPECTED_SPLIT_POSITIVES[split_name]
        )
    for column, expected_rows in _EXPECTED_MATURITY_COUNTS.items():
        assert int(canonical[column].sum()) == expected_rows
    assert canonical.index.equals(
        pd.RangeIndex(start=0, stop=_EXPECTED_TOTAL_ROWS, step=1)
    )
    assert canonical["appointment_id"].is_unique
    expected_order = canonical.sort_values(
        ["prediction_time", "appointment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(canonical, expected_order)
    pd.testing.assert_frame_equal(repeated, canonical)
    _assert_protected_trees_unchanged(trees_before)
    _assert_process_state_unchanged(state_before)


def test_exact_orchestration_order_identity_and_return_structure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts"
    calls, objects = _install_success_spies(monkeypatch, output_dir)
    result = generate_eda_artifacts(str(output_dir))
    assert calls == [
        "canonical",
        "tables",
        "figures",
        "write_tables",
        "write_figures",
    ]
    assert type(result) is dict
    assert tuple(result) == ("tables", "figures")
    assert result["tables"] is objects["table_paths"]
    assert result["figures"] is objects["figure_paths"]
    assert type(result["tables"]) is dict
    assert type(result["figures"]) is dict


def test_all_analytical_construction_precedes_first_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "not-created-yet"
    events: list[str] = []
    canonical = object()
    tables = object()
    figures = object()

    def build_canonical() -> object:
        assert not output_dir.exists()
        events.append("canonical")
        return canonical

    def build_tables(candidate: object) -> object:
        assert candidate is canonical
        assert not output_dir.exists()
        events.append("tables")
        return tables

    def build_figures(candidate: object) -> object:
        assert candidate is tables
        assert not output_dir.exists()
        events.append("figures")
        return figures

    def write_tables(candidate: object, directory: Path) -> dict[str, Path]:
        assert candidate is tables
        assert directory == output_dir
        assert events == ["canonical", "tables", "figures"]
        directory.mkdir()
        events.append("write_tables")
        return {}

    def write_figures(candidate: object, directory: Path) -> dict[str, Path]:
        assert candidate is figures
        assert directory == output_dir
        assert output_dir.is_dir()
        events.append("write_figures")
        return {}

    monkeypatch.setattr(run_eda, "_build_canonical_dataset", build_canonical)
    monkeypatch.setattr(run_eda, "build_eda_tables", build_tables)
    monkeypatch.setattr(run_eda, "build_eda_figures", build_figures)
    monkeypatch.setattr(run_eda, "write_eda_tables", write_tables)
    monkeypatch.setattr(run_eda, "write_eda_figures", write_figures)
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    generate_eda_artifacts(output_dir)
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)
    assert events == [
        "canonical",
        "tables",
        "figures",
        "write_tables",
        "write_figures",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_calls"),
    (
        ("canonical", ("canonical",)),
        ("tables", ("canonical", "tables")),
        ("figures", ("canonical", "tables", "figures")),
        (
            "write_tables",
            ("canonical", "tables", "figures", "write_tables"),
        ),
        (
            "write_figures",
            (
                "canonical",
                "tables",
                "figures",
                "write_tables",
                "write_figures",
            ),
        ),
    ),
)
def test_failure_matrix_preserves_exception_and_stops_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
    expected_calls: tuple[str, ...],
) -> None:
    output_dir = tmp_path / failure_stage
    calls: list[str] = []
    failure = RuntimeError(failure_stage)
    canonical = object()
    tables = object()
    figures = object()

    def stage(name: str, result: object) -> object:
        calls.append(name)
        if name == failure_stage:
            raise failure
        return result

    def build_canonical() -> object:
        return stage("canonical", canonical)

    def build_tables(candidate: object) -> object:
        assert candidate is canonical
        return stage("tables", tables)

    def build_figures(candidate: object) -> object:
        assert candidate is tables
        return stage("figures", figures)

    def write_tables(candidate: object, directory: Path) -> object:
        assert candidate is tables
        assert directory == output_dir
        result = stage("write_tables", {"table": directory / "table.csv"})
        if failure_stage == "write_figures":
            directory.mkdir()
            (directory / "completed.csv").write_bytes(b"complete")
        return result

    def write_figures(candidate: object, directory: Path) -> object:
        assert candidate is figures
        assert directory == output_dir
        return stage("write_figures", {"figure": directory / "figure.png"})

    monkeypatch.setattr(run_eda, "_build_canonical_dataset", build_canonical)
    monkeypatch.setattr(run_eda, "build_eda_tables", build_tables)
    monkeypatch.setattr(run_eda, "build_eda_figures", build_figures)
    monkeypatch.setattr(run_eda, "write_eda_tables", write_tables)
    monkeypatch.setattr(run_eda, "write_eda_figures", write_figures)
    mutations: list[str] = []
    if failure_stage in {"canonical", "tables", "figures"}:
        mutations = _instrument_pre_writer_mutations(
            monkeypatch,
            include_writers=False,
        )
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    no_result = object()
    result: object = no_result
    caught: BaseException | None = None
    try:
        result = generate_eda_artifacts(output_dir)
    except BaseException as error:
        caught = error
    assert caught is failure
    assert result is no_result
    assert tuple(calls) == expected_calls
    if failure_stage in {"canonical", "tables", "figures"}:
        assert mutations == []
        assert not output_dir.exists()
    if failure_stage == "write_tables":
        assert not output_dir.exists()
    if failure_stage == "write_figures":
        assert (output_dir / "completed.csv").read_bytes() == b"complete"
        assert tuple(path.name for path in output_dir.iterdir()) == (
            "completed.csv",
        )
    _assert_protected_trees_unchanged(trees_before)
    _assert_process_state_unchanged(state_before)


def test_authentic_output_has_exact_sixteen_flat_files_and_return_paths(
    _authentic_artifacts: tuple[Path, dict[str, dict[str, Path]]],
) -> None:
    output_dir, result = _authentic_artifacts
    expected_names = tuple(
        filename
        for _, filename in (
            *_EXPECTED_TABLE_ARTIFACTS,
            *_EXPECTED_FIGURE_ARTIFACTS,
        )
    )
    contents = tuple(sorted(output_dir.iterdir(), key=lambda path: path.name))
    assert len(contents) == 16
    assert all(path.is_file() for path in contents)
    assert not any(path.is_dir() for path in output_dir.rglob("*"))
    assert {path.name for path in contents} == set(expected_names)
    assert tuple(result) == ("tables", "figures")
    assert tuple(result["tables"]) == tuple(
        key for key, _ in _EXPECTED_TABLE_ARTIFACTS
    )
    assert tuple(result["figures"]) == tuple(
        key for key, _ in _EXPECTED_FIGURE_ARTIFACTS
    )
    assert tuple(result["tables"].values()) == tuple(
        output_dir / filename for _, filename in _EXPECTED_TABLE_ARTIFACTS
    )
    assert tuple(result["figures"].values()) == tuple(
        output_dir / filename for _, filename in _EXPECTED_FIGURE_ARTIFACTS
    )


def test_authentic_csv_headers_are_exact_test_owned_literals(
    _authentic_artifacts: tuple[Path, dict[str, dict[str, Path]]],
) -> None:
    _, result = _authentic_artifacts
    assert tuple(_EXPECTED_HEADERS) == tuple(
        key for key, _ in _EXPECTED_TABLE_ARTIFACTS
    )
    for key, path in result["tables"].items():
        with path.open("r", encoding="utf-8", newline="") as source:
            header = tuple(next(csv.reader(source)))
        assert header == _EXPECTED_HEADERS[key]


def test_authentic_png_structure_and_dimensions_are_exact(
    _authentic_artifacts: tuple[Path, dict[str, dict[str, Path]]],
) -> None:
    _, result = _authentic_artifacts
    assert {
        path.name: _png_dimensions(path)
        for path in result["figures"].values()
    } == _EXPECTED_DIMENSIONS


def test_byte_determinism_across_directories(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    trees_before = _protected_tree_snapshot()
    first_state = _process_state_snapshot()
    first = generate_eda_artifacts(first_dir)
    _assert_process_state_unchanged(first_state)
    _assert_protected_trees_unchanged(trees_before)
    second_state = _process_state_snapshot()
    second = generate_eda_artifacts(second_dir)
    _assert_process_state_unchanged(second_state)
    _assert_protected_trees_unchanged(trees_before)
    assert tuple(first) == tuple(second) == ("tables", "figures")
    assert _artifact_bytes(first_dir) == _artifact_bytes(second_dir)


def test_repeated_execution_is_deterministic_and_replaces_sixteen_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "repeated"
    trees_before = _protected_tree_snapshot()
    first_state = _process_state_snapshot()
    first = generate_eda_artifacts(output_dir)
    _assert_process_state_unchanged(first_state)
    _assert_protected_trees_unchanged(trees_before)
    expected = _artifact_bytes(output_dir)
    for path in (*first["tables"].values(), *first["figures"].values()):
        path.write_bytes(b"stale artifact")
    second_state = _process_state_snapshot()
    second = generate_eda_artifacts(output_dir)
    _assert_process_state_unchanged(second_state)
    _assert_protected_trees_unchanged(trees_before)
    assert _artifact_bytes(output_dir) == expected
    assert tuple(second["tables"].values()) == tuple(first["tables"].values())
    assert tuple(second["figures"].values()) == tuple(first["figures"].values())


def test_unrelated_existing_content_is_preserved(tmp_path: Path) -> None:
    output_dir = tmp_path / "preservation"
    nested = output_dir / "unrelated"
    nested.mkdir(parents=True)
    note = output_dir / "reviewer-note.txt"
    marker = nested / "marker.bin"
    note.write_text("keep me\n", encoding="utf-8", newline="\n")
    marker.write_bytes(b"preserve")
    generate_eda_artifacts(output_dir)
    assert note.read_text(encoding="utf-8") == "keep me\n"
    assert marker.read_bytes() == b"preserve"


def test_every_call_rebuilds_fresh_objects_and_mappings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonicals: list[object] = []
    tables_seen: list[dict[str, object]] = []
    figures_seen: list[dict[str, object]] = []
    writer_results: list[dict[str, Path]] = []

    def build_canonical() -> object:
        canonical = object()
        canonicals.append(canonical)
        return canonical

    def build_tables(canonical: object) -> dict[str, object]:
        assert canonical is canonicals[-1]
        tables = {"table": object()}
        tables_seen.append(tables)
        return tables

    def build_figures(tables: dict[str, object]) -> dict[str, object]:
        assert tables is tables_seen[-1]
        figures = {"figure": object()}
        figures_seen.append(figures)
        return figures

    def write_tables(
        tables: dict[str, object],
        output_dir: Path,
    ) -> dict[str, Path]:
        assert tables is tables_seen[-1]
        result = {"table": output_dir / "table.csv"}
        writer_results.append(result)
        return result

    def write_figures(
        figures: dict[str, object],
        output_dir: Path,
    ) -> dict[str, Path]:
        assert figures is figures_seen[-1]
        result = {"figure": output_dir / "figure.png"}
        writer_results.append(result)
        return result

    monkeypatch.setattr(run_eda, "_build_canonical_dataset", build_canonical)
    monkeypatch.setattr(run_eda, "build_eda_tables", build_tables)
    monkeypatch.setattr(run_eda, "build_eda_figures", build_figures)
    monkeypatch.setattr(run_eda, "write_eda_tables", write_tables)
    monkeypatch.setattr(run_eda, "write_eda_figures", write_figures)
    first_state = _process_state_snapshot()
    first = generate_eda_artifacts(tmp_path / "first")
    _assert_process_state_unchanged(first_state)
    second_state = _process_state_snapshot()
    second = generate_eda_artifacts(tmp_path / "second")
    _assert_process_state_unchanged(second_state)
    assert len(canonicals) == len(tables_seen) == len(figures_seen) == 2
    assert canonicals[0] is not canonicals[1]
    assert tables_seen[0] is not tables_seen[1]
    assert figures_seen[0] is not figures_seen[1]
    assert first is not second
    assert first["tables"] is not second["tables"]
    assert first["figures"] is not second["figures"]
    assert first["tables"] is writer_results[0]
    assert first["figures"] is writer_results[1]
    assert second["tables"] is writer_results[2]
    assert second["figures"] is writer_results[3]


def test_cwd_environment_and_matplotlib_global_state_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    output_dir = tmp_path / "state"
    generate_eda_artifacts(output_dir)
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)


def test_cwd_independence_produces_identical_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "repository-cwd"
    second_dir = tmp_path / "alternate-cwd"
    first_state = _process_state_snapshot()
    generate_eda_artifacts(first_dir)
    _assert_process_state_unchanged(first_state)
    alternate = tmp_path / "cwd"
    alternate.mkdir()
    monkeypatch.chdir(alternate)
    second_state = _process_state_snapshot()
    generate_eda_artifacts(second_dir)
    _assert_process_state_unchanged(second_state)
    assert _artifact_bytes(first_dir) == _artifact_bytes(second_dir)


def test_fresh_import_and_reload_preserve_complete_process_state() -> None:
    trees_before = _protected_tree_snapshot()
    specification = util.spec_from_file_location(
        "_run_eda_process_state_probe",
        _MODULE_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = util.module_from_spec(specification)
    import_state = _process_state_snapshot()
    specification.loader.exec_module(module)
    _assert_process_state_unchanged(import_state)
    assert not (_REPOSITORY_ROOT / "reports" / "eda").exists()
    reload_state = _process_state_snapshot()
    reload(run_eda)
    _assert_process_state_unchanged(reload_state)
    _assert_protected_trees_unchanged(trees_before)


@pytest.mark.parametrize("reload_module", (False, True))
def test_import_and_reload_have_no_side_effects(
    tmp_path: Path,
    reload_module: bool,
) -> None:
    expression = "import src.analysis.run_eda"
    if reload_module:
        expression = (
            "import importlib; import src.analysis.run_eda as module; "
            "importlib.reload(module)"
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPOSITORY_ROOT)
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    completed = subprocess.run(
        [sys.executable, "-c", expression],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert tuple(tmp_path.iterdir()) == ()


def test_cli_parser_default_and_explicit_output_dir(tmp_path: Path) -> None:
    state_before = _process_state_snapshot()
    parser = run_eda._build_parser()
    _assert_exact_parser_surface(parser)
    assert parser.parse_args([]).output_dir == Path("reports/eda")
    explicit = tmp_path / "explicit"
    assert parser.parse_args(["--output-dir", str(explicit)]).output_dir == (
        explicit
    )
    _assert_process_state_unchanged(state_before)


def test_parser_surface_probe_rejects_any_additional_action() -> None:
    parser = run_eda._build_parser()
    parser.add_argument("--unused")
    with pytest.raises(AssertionError):
        _assert_exact_parser_surface(parser)


def test_cli_success_output_order_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "cli"
    table_paths = {
        key: output_dir / filename for key, filename in _EXPECTED_TABLE_ARTIFACTS
    }
    figure_paths = {
        key: output_dir / filename for key, filename in _EXPECTED_FIGURE_ARTIFACTS
    }

    def generate(directory: Path) -> dict[str, dict[str, Path]]:
        assert directory == output_dir
        return {"tables": table_paths, "figures": figure_paths}

    monkeypatch.setattr(run_eda, "generate_eda_artifacts", generate)
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    assert run_eda._main(["--output-dir", str(output_dir)]) == 0
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        *(f"Wrote table: {path}" for path in table_paths.values()),
        *(f"Wrote figure: {path}" for path in figure_paths.values()),
    ]


def test_cli_invalid_arguments_use_argparse_failure() -> None:
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    with pytest.raises(SystemExit) as captured:
        run_eda._main(["--unknown-option"])
    assert captured.value.code != 0
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)


def test_cli_operational_exception_is_not_converted_to_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failure = RuntimeError("operational failure")

    def fail(output_dir: Path) -> dict[str, dict[str, Path]]:
        raise failure

    monkeypatch.setattr(run_eda, "generate_eda_artifacts", fail)
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    caught: BaseException | None = None
    try:
        run_eda._main(["--output-dir", str(tmp_path / "failure")])
    except BaseException as error:
        caught = error
    assert caught is failure
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)


def test_repeated_module_execution_is_byte_and_stdout_deterministic(
    tmp_path: Path,
) -> None:
    output_dirs = (tmp_path / "module-first", tmp_path / "module-second")
    expected_names = tuple(
        filename
        for _, filename in (
            *_EXPECTED_TABLE_ARTIFACTS,
            *_EXPECTED_FIGURE_ARTIFACTS,
        )
    )
    trees_before = _protected_tree_snapshot()
    completed_runs: list[subprocess.CompletedProcess[str]] = []
    normalized_stdout: list[tuple[str, ...]] = []
    for output_dir in output_dirs:
        state_before = _process_state_snapshot()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.analysis.run_eda",
                "--output-dir",
                str(output_dir),
            ],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        _assert_process_state_unchanged(state_before)
        _assert_protected_trees_unchanged(trees_before)
        assert completed.returncode == 0
        assert completed.stderr == ""
        lines = completed.stdout.splitlines()
        assert len(lines) == 16
        assert all(lines)
        expected_lines = [
            *(
                f"Wrote table: {output_dir / filename}"
                for _, filename in _EXPECTED_TABLE_ARTIFACTS
            ),
            *(
                f"Wrote figure: {output_dir / filename}"
                for _, filename in _EXPECTED_FIGURE_ARTIFACTS
            ),
        ]
        assert lines == expected_lines
        normalized_stdout.append(
            tuple(
                line.replace(str(output_dir), "<OUTPUT_DIR>", 1)
                for line in lines
            )
        )
        contents = tuple(
            sorted(output_dir.iterdir(), key=lambda path: path.name)
        )
        assert len(contents) == 16
        assert all(path.is_file() for path in contents)
        assert {path.name for path in contents} == set(expected_names)
        assert not any(path.is_dir() for path in output_dir.rglob("*"))
        assert not (_REPOSITORY_ROOT / "reports" / "eda").exists()
        completed_runs.append(completed)
    assert normalized_stdout[0] == normalized_stdout[1]
    assert _artifact_bytes(output_dirs[0]) == _artifact_bytes(output_dirs[1])
    assert len(completed_runs) == 2


def test_invalid_argument_subprocess_has_no_artifact_side_effect(
    tmp_path: Path,
) -> None:
    trees_before = _protected_tree_snapshot()
    state_before = _process_state_snapshot()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_eda",
            "--definitely-invalid-option",
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr.lower()
    assert "unrecognized arguments: --definitely-invalid-option" in (
        completed.stderr
    )
    assert "Wrote table:" not in completed.stderr
    assert "Wrote figure:" not in completed.stderr
    assert tuple(tmp_path.iterdir()) == ()
    assert not (_REPOSITORY_ROOT / "reports" / "eda").exists()


def test_static_orchestration_dependency_boundary() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_eda_artifacts"
    )
    call_names = []
    referenced_names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            referenced_names.add(node.id.lower())
        if isinstance(node, ast.Attribute):
            referenced_names.add(node.attr.lower())
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
    assert call_names == [
        "Path",
        "_build_canonical_dataset",
        "build_eda_tables",
        "build_eda_figures",
        "write_eda_tables",
        "write_eda_figures",
    ]
    forbidden = {
        "select_eda_populations",
        "target",
        "validation",
        "test",
        "pretest",
        "model",
        "preprocessing",
    }
    assert referenced_names.isdisjoint(forbidden)
    assert not any(name.startswith("summarize_") for name in referenced_names)
    assert not any("drift" in name for name in referenced_names)
    assert not any("relationship" in name for name in referenced_names)


def test_no_pyplot_or_process_global_state_mutation_code() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    aliases = _import_aliases(tree)
    assert "matplotlib.pyplot" not in source
    assert "pyplot" not in source
    assert "matplotlib.use" not in source
    assert "switch_backend" not in source
    forbidden_calls = {
        "os.chdir",
        "os.putenv",
        "os.unsetenv",
        "warnings.filterwarnings",
        "warnings.simplefilter",
        "warnings.resetwarnings",
        "random.seed",
        "random.setstate",
        "numpy.random.seed",
        "numpy.random.set_state",
        "matplotlib.use",
        "matplotlib.pyplot.switch_backend",
    }
    state_roots = {
        "os.environ",
        "warnings.filters",
        "matplotlib.rcparams",
        "matplotlib.rcparamsdefault",
        "matplotlib.rcparamsorig",
    }
    mutating_methods = {
        "__delitem__",
        "__setitem__",
        "append",
        "clear",
        "extend",
        "insert",
        "pop",
        "popitem",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "update",
    }
    gcf_mutations = {
        "destroy",
        "destroy_all",
        "set_active",
        "_set_new_active_manager",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            qualified = _qualified_name(node.func, aliases).lower()
            if qualified in forbidden_calls:
                violations.append(qualified)
            if isinstance(node.func, ast.Attribute):
                owner = _qualified_name(node.func.value, aliases).lower()
                if owner in state_roots and node.func.attr in mutating_methods:
                    violations.append(f"{owner}.{node.func.attr}")
                if owner.endswith(".gcf") and node.func.attr in gcf_mutations:
                    violations.append(f"{owner}.{node.func.attr}")
        for target in _assignment_targets(node):
            for candidate in ast.walk(target):
                qualified = _qualified_name(candidate, aliases).lower()
                if qualified in state_roots or any(
                    qualified.startswith(f"{root}.") for root in state_roots
                ):
                    violations.append(f"assignment:{qualified}")
                if qualified.endswith(".gcf") or ".gcf." in qualified:
                    violations.append(f"assignment:{qualified}")
    assert violations == []


def test_no_processed_writer_or_additional_artifact_format() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    owned = {
        node.name: ast.get_source_segment(
            _MODULE_PATH.read_text(encoding="utf-8"), node
        )
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_build_canonical_dataset",
            "generate_eda_artifacts",
            "_build_parser",
            "_main",
        }
    }
    runner_source = "\n".join(segment or "" for segment in owned.values()).lower()
    assert "data/processed" not in runner_source
    assert "data\\processed" not in runner_source
    for suffix in (".json", ".html", ".ipynb", ".pdf", ".svg", ".parquet"):
        assert suffix not in runner_source


def test_raw_and_processed_data_remain_protected_after_authentic_execution(
    tmp_path: Path,
) -> None:
    trees_before = _protected_tree_snapshot()
    assert trees_before == (_EXPECTED_RAW_TREE, _EXPECTED_PROCESSED_TREE)
    state_before = _process_state_snapshot()
    generate_eda_artifacts(tmp_path / "protected-data")
    _assert_process_state_unchanged(state_before)
    _assert_protected_trees_unchanged(trees_before)
