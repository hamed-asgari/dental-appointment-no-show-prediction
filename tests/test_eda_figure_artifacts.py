"""Tests for deterministic serialization of the approved EDA figure bundle."""

from __future__ import annotations

import ast
import builtins
from collections import OrderedDict, UserDict
from collections.abc import Mapping
from copy import deepcopy
from importlib import reload, util
import inspect
from pathlib import Path
import struct
from types import ModuleType
from typing import Any, get_type_hints
import zlib

import matplotlib
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from src.analysis import artifacts, figure_artifacts
from src.analysis.figure_artifacts import write_eda_figures
from src.analysis.figures import build_eda_figures
from src.analysis.run_eda import build_eda_tables
from src.data import build_dataset as bd


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw"
_MODULE_PATH = (
    _REPOSITORY_ROOT / "src" / "analysis" / "figure_artifacts.py"
)
_EXPECTED_ARTIFACTS = (
    ("class_balance", "01_class_balance.png"),
    ("temporal_monthly", "02_temporal_monthly.png"),
    ("numeric_drift", "03_numeric_drift.png"),
    ("categorical_drift", "04_categorical_drift.png"),
    ("numeric_relationships", "05_numeric_relationships.png"),
)
_EXPECTED_KEYS = tuple(key for key, _ in _EXPECTED_ARTIFACTS)
_EXPECTED_FILENAMES = tuple(name for _, name in _EXPECTED_ARTIFACTS)
_EXPECTED_DIMENSIONS = (
    (960, 540),
    (1200, 660),
    (1080, 576),
    (1080, 576),
    (960, 840),
)
_EXPECTED_SOFTWARE = "dental-appointment-no-show-prediction"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ATOMIC_STAGES = ("create", "write", "flush", "fsync", "close", "replace")
_FAILURE_POSITIONS = (1, 3, 5)


@pytest.fixture(scope="session")
def _authentic_tables() -> dict[str, pd.DataFrame]:
    raw_tables = bd.load_raw_data(_RAW_DIR)
    canonical = bd.build_analytical_dataset(raw_tables)
    return build_eda_tables(canonical)


@pytest.fixture
def _authentic_figures(
    _authentic_tables: dict[str, pd.DataFrame],
) -> dict[str, Figure]:
    return build_eda_figures(_authentic_tables)


def _simple_figures() -> dict[str, Figure]:
    figures: dict[str, Figure] = {}
    for position, ((key, _), dimensions) in enumerate(
        zip(_EXPECTED_ARTIFACTS, _EXPECTED_DIMENSIONS),
        start=1,
    ):
        width, height = dimensions
        figure = Figure(
            figsize=(width / 120, height / 120),
            dpi=120,
        )
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.bar([0, 1], [position, position + 1], label="bars")
        axis.plot([0, 1], [position + 1, position], label="line")
        axis.set_title(f"Figure {position}")
        axis.set_xlabel("x label")
        axis.set_ylabel("y label")
        axis.text(0.25, position + 0.5, f"annotation {position}")
        figures[key] = figure
    return figures


@pytest.fixture(scope="module")
def _expected_simple_pngs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, bytes]:
    output_dir = tmp_path_factory.mktemp("expected-simple-pngs")
    write_eda_figures(_simple_figures(), output_dir)
    return _read_artifacts(output_dir)


def _array_snapshot(value: object) -> tuple[str, tuple[int, ...], str, str]:
    array = np.ma.asarray(value)
    mask = np.ma.getmaskarray(array)
    return (
        str(array.dtype),
        tuple(array.shape),
        repr(array.tolist()),
        repr(mask.tolist()),
    )


def _figure_semantics(figure: Figure) -> tuple[object, ...]:
    axes = []
    for axis in figure.axes:
        patches = tuple(
            (
                type(patch).__name__,
                getattr(patch, "get_x", lambda: None)(),
                getattr(patch, "get_y", lambda: None)(),
                getattr(patch, "get_width", lambda: None)(),
                getattr(patch, "get_height", lambda: None)(),
            )
            for patch in axis.patches
        )
        lines = tuple(
            (
                _array_snapshot(line.get_xdata()),
                _array_snapshot(line.get_ydata()),
            )
            for line in axis.lines
        )
        images = tuple(
            (
                _array_snapshot(image.get_array()),
                image.get_cmap().name,
                image.norm.vmin,
                image.norm.vmax,
            )
            for image in axis.images
        )
        axes.append(
            (
                axis.get_title(),
                axis.get_xlabel(),
                axis.get_ylabel(),
                tuple(label.get_text() for label in axis.get_xticklabels()),
                tuple(label.get_text() for label in axis.get_yticklabels()),
                tuple(axis.get_xlim()),
                tuple(axis.get_ylim()),
                patches,
                lines,
                images,
                tuple(
                    (text.get_text(), tuple(text.get_position()))
                    for text in axis.texts
                ),
            )
        )
    return (
        id(figure),
        id(figure.canvas),
        tuple(figure.get_size_inches()),
        figure.dpi,
        len(figure.axes),
        tuple(axes),
    )


def _bundle_snapshot(
    figures: Mapping[object, object],
) -> tuple[tuple[object, ...], tuple[int, ...], tuple[object, ...]]:
    values = tuple(figures.values())
    return (
        tuple(figures),
        tuple(id(value) for value in values),
        tuple(
            _figure_semantics(value)
            if isinstance(value, Figure)
            else value
            for value in values
        ),
    )


def _assert_bundle_unchanged(
    figures: Mapping[object, object],
    expected: tuple[
        tuple[object, ...],
        tuple[int, ...],
        tuple[object, ...],
    ],
) -> None:
    assert _bundle_snapshot(figures) == expected


def _png_chunks(payload: bytes) -> tuple[tuple[bytes, bytes], ...]:
    assert payload.startswith(_PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(_PNG_SIGNATURE)
    while offset < len(payload):
        assert len(payload) - offset >= 12
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        assert crc_end <= len(payload)
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        assert zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF == expected_crc
        chunks.append((chunk_type, chunk_data))
        offset = crc_end
        if chunk_type == b"IEND":
            break
    assert offset == len(payload)
    return tuple(chunks)


def _decode_png_keyword(payload: bytes) -> str:
    if not 1 <= len(payload) <= 79:
        raise ValueError("PNG textual keyword length must be inside [1, 79]")
    if any(byte < 32 or 127 <= byte <= 160 for byte in payload):
        raise ValueError("PNG textual keyword contains a forbidden byte")
    keyword = payload.decode("latin-1")
    if keyword.startswith(" ") or keyword.endswith(" ") or "  " in keyword:
        raise ValueError("PNG textual keyword contains invalid spacing")
    return keyword


def _split_png_field(payload: bytes, label: str) -> tuple[bytes, bytes]:
    field, separator, remainder = payload.partition(b"\x00")
    if not separator:
        raise ValueError(f"malformed PNG textual chunk: missing {label}")
    return field, remainder


def _decompress_png_text(payload: bytes) -> bytes:
    try:
        return zlib.decompress(payload)
    except zlib.error as error:
        raise ValueError("malformed compressed PNG text") from error


def _decode_textual_metadata(
    chunks: tuple[tuple[bytes, bytes], ...],
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for chunk_type, chunk_data in chunks:
        if chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
            continue

        keyword_bytes, remainder = _split_png_field(
            chunk_data,
            "keyword separator",
        )
        keyword = _decode_png_keyword(keyword_bytes)
        if keyword in metadata:
            raise ValueError(f"duplicate PNG textual keyword: {keyword!r}")

        if chunk_type == b"tEXt":
            if b"\x00" in remainder:
                raise ValueError("malformed tEXt value")
            value = remainder.decode("latin-1")
        elif chunk_type == b"zTXt":
            if not remainder:
                raise ValueError("malformed zTXt compression fields")
            if remainder[0] != 0:
                raise ValueError("unsupported zTXt compression method")
            decoded = _decompress_png_text(remainder[1:])
            if b"\x00" in decoded:
                raise ValueError("malformed zTXt value")
            value = decoded.decode("latin-1")
        else:
            if len(remainder) < 2:
                raise ValueError("malformed iTXt compression fields")
            compression_flag, compression_method = remainder[:2]
            if compression_flag not in {0, 1}:
                raise ValueError("invalid iTXt compression flag")
            if compression_method != 0:
                raise ValueError("unsupported iTXt compression method")
            language, remainder = _split_png_field(
                remainder[2:],
                "iTXt language separator",
            )
            translated, text_payload = _split_png_field(
                remainder,
                "iTXt translated-keyword separator",
            )
            try:
                language.decode("ascii")
                translated.decode("utf-8")
                decoded = (
                    _decompress_png_text(text_payload)
                    if compression_flag == 1
                    else text_payload
                )
                if b"\x00" in decoded:
                    raise ValueError("malformed iTXt value")
                value = decoded.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("malformed iTXt text encoding") from error
        metadata[keyword] = value
    return metadata


def _assert_approved_textual_metadata(
    chunks: tuple[tuple[bytes, bytes], ...],
) -> None:
    assert _decode_textual_metadata(chunks) == {
        "Software": "dental-appointment-no-show-prediction"
    }


def _assert_valid_png(payload: bytes, dimensions: tuple[int, int]) -> None:
    chunks = _png_chunks(payload)
    chunk_types = tuple(chunk_type for chunk_type, _ in chunks)
    assert chunk_types[0] == b"IHDR"
    assert chunk_types[-1] == b"IEND"
    assert chunk_types.count(b"IHDR") == 1
    assert chunk_types.count(b"IEND") == 1
    assert chunk_types.count(b"IDAT") >= 1
    assert b"tIME" not in chunk_types
    ihdr = chunks[0][1]
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    assert (width, height) == dimensions
    assert (bit_depth, color_type) == (8, 6)
    assert (compression, filtering, interlace) == (0, 0, 0)
    compressed = b"".join(
        data for chunk_type, data in chunks if chunk_type == b"IDAT"
    )
    decoded = zlib.decompress(compressed)
    assert len(decoded) == height * (1 + width * 4)
    _assert_approved_textual_metadata(chunks)


def _read_artifacts(output_dir: Path) -> dict[str, bytes]:
    return {
        filename: (output_dir / filename).read_bytes()
        for filename in _EXPECTED_FILENAMES
    }


def _assert_only_completed(output_dir: Path, completed: int) -> None:
    entries = tuple(output_dir.iterdir())
    assert {path.name for path in entries} == set(
        _EXPECTED_FILENAMES[:completed]
    )
    assert all(path.is_file() for path in entries)
    assert not any(path.suffix == ".tmp" for path in output_dir.rglob("*"))


def _target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            name for child in target.elts for name in _target_names(child)
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


def _direct_writer_violations(tree: ast.Module) -> tuple[str, ...]:
    builtins_modules = {"builtins"}
    builtin_open_names = {"open"}
    bytesio_constructors = {"BytesIO"}
    io_modules = {"io"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    builtins_modules.add(alias.asname or alias.name)
                elif alias.name == "io":
                    io_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "builtins":
                builtin_open_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "open"
                )
            elif node.module == "io":
                bytesio_constructors.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "BytesIO"
                )

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            aliases_open = (
                isinstance(value, ast.Name)
                and value.id in builtin_open_names
            ) or (
                isinstance(value, ast.Attribute)
                and value.attr == "open"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_modules
            )
            if aliases_open:
                for target in node.targets:
                    for name in _target_names(target):
                        if name not in builtin_open_names:
                            builtin_open_names.add(name)
                            changed = True

    byte_buffers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        constructor = value.func
        is_bytesio = (
            isinstance(constructor, ast.Name)
            and constructor.id in bytesio_constructors
        ) or (
            isinstance(constructor, ast.Attribute)
            and constructor.attr == "BytesIO"
            and isinstance(constructor.value, ast.Name)
            and constructor.value.id in io_modules
        )
        if is_bytesio:
            for target in node.targets:
                byte_buffers.update(_target_names(target))

    forbidden_attributes = {
        "open",
        "savefig",
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_json",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "write_bytes",
        "write_text",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callable_node = node.func
        if (
            isinstance(callable_node, ast.Name)
            and callable_node.id in builtin_open_names
        ):
            violations.append("builtins.open")
            continue
        if not isinstance(callable_node, ast.Attribute):
            continue
        if callable_node.attr in forbidden_attributes:
            violations.append(callable_node.attr)
            continue
        if callable_node.attr != "print_png":
            continue
        destination = node.args[0] if node.args else next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"filename_or_obj", "filename"}
            ),
            None,
        )
        in_memory = (
            isinstance(destination, ast.Name)
            and destination.id in byte_buffers
        ) or (
            isinstance(destination, ast.Call)
            and isinstance(destination.func, ast.Name)
            and destination.func.id in bytesio_constructors
        )
        if not in_memory:
            violations.append("print_png-filesystem")
    return tuple(violations)


def _matplotlib_state() -> tuple[object, object, tuple[int, ...]]:
    return (
        matplotlib.get_backend(),
        deepcopy(dict(matplotlib.rcParams)),
        tuple(id(manager) for manager in Gcf.get_all_fig_managers()),
    )


def _invalid_bundle(case: str) -> Mapping[object, object]:
    figures: dict[object, object] = _simple_figures()
    if case == "dict_subclass":
        return OrderedDict(figures)
    if case == "non_dict_mapping":
        return UserDict(figures)
    if case == "missing_key":
        figures.pop("numeric_relationships")
    elif case == "extra_key":
        extra = Figure()
        FigureCanvasAgg(extra)
        figures["extra"] = extra
    elif case == "reordered_keys":
        figures = dict(reversed(tuple(figures.items())))
    elif case == "non_string_key":
        first = figures.pop("class_balance")
        figures = {1: first, **figures}
    elif case == "non_figure_value":
        figures["numeric_drift"] = "not a figure"
    elif case == "duplicate_figure":
        figures["numeric_drift"] = figures["temporal_monthly"]
    elif case == "duplicate_canvas":
        first = figures["class_balance"]
        second = figures["temporal_monthly"]
        assert isinstance(first, Figure)
        assert isinstance(second, Figure)
        second.set_canvas(first.canvas)
    elif case == "non_agg_canvas":
        figure = figures["categorical_drift"]
        assert isinstance(figure, Figure)
        figure.set_canvas(FigureCanvasBase(figure))
    else:
        raise AssertionError(f"unknown invalid case: {case}")
    return figures


def _install_atomic_failure(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    failure_position: int,
    operation_error: OSError,
    cleanup_error: OSError | None = None,
) -> dict[str, Any]:
    if stage not in _ATOMIC_STAGES:
        raise AssertionError(f"unknown atomic stage: {stage}")

    original_temporary_file = artifacts.tempfile.NamedTemporaryFile
    original_fsync = artifacts.os.fsync
    original_replace = artifacts.os.replace
    original_unlink = artifacts.Path.unlink
    state: dict[str, Any] = {
        "create_count": 0,
        "events": [],
        "create_dirs": [],
        "temporary_paths": [],
        "cleanup_paths": [],
        "replace_sources": [],
        "replace_destinations": [],
    }
    temporary_positions: dict[Path, int] = {}

    class _ControlledTemporaryFile:
        def __init__(self, wrapped: Any, position: int) -> None:
            self._wrapped = wrapped
            self._position = position
            self.name = wrapped.name

        def __enter__(self) -> _ControlledTemporaryFile:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def write(self, payload: bytes) -> int:
            state["events"].append((self._position, "write"))
            if stage == "write" and self._position == failure_position:
                raise operation_error
            return self._wrapped.write(payload)

        def flush(self) -> None:
            state["events"].append((self._position, "flush"))
            if stage == "flush" and self._position == failure_position:
                raise operation_error
            self._wrapped.flush()

        def fileno(self) -> int:
            return self._wrapped.fileno()

        def close(self) -> None:
            self._wrapped.close()
            state["events"].append((self._position, "close"))
            if stage == "close" and self._position == failure_position:
                raise operation_error

        @property
        def closed(self) -> bool:
            return self._wrapped.closed

    def create_temporary(*args: object, **kwargs: object) -> object:
        position = state["create_count"] + 1
        state["create_count"] = position
        state["events"].append((position, "create"))
        state["create_dirs"].append(Path(kwargs["dir"]))
        if stage == "create" and position == failure_position:
            raise operation_error
        wrapped = original_temporary_file(*args, **kwargs)
        controlled = _ControlledTemporaryFile(wrapped, position)
        temporary_path = Path(controlled.name)
        state["temporary_paths"].append(temporary_path)
        temporary_positions[temporary_path] = position
        return controlled

    def controlled_fsync(file_descriptor: int) -> None:
        position = state["create_count"]
        state["events"].append((position, "fsync"))
        if stage == "fsync" and position == failure_position:
            raise operation_error
        original_fsync(file_descriptor)

    def controlled_replace(source: Path, destination: Path) -> None:
        position = state["create_count"]
        state["events"].append((position, "replace"))
        state["replace_sources"].append(Path(source))
        state["replace_destinations"].append(Path(destination))
        if stage == "replace" and position == failure_position:
            raise operation_error
        original_replace(source, destination)

    def controlled_unlink(path: Path, *, missing_ok: bool = False) -> None:
        position = temporary_positions.get(Path(path), state["create_count"])
        state["events"].append((position, "cleanup"))
        state["cleanup_paths"].append(Path(path))
        if cleanup_error is not None:
            raise cleanup_error
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        artifacts.tempfile,
        "NamedTemporaryFile",
        create_temporary,
    )
    monkeypatch.setattr(artifacts.os, "fsync", controlled_fsync)
    monkeypatch.setattr(artifacts.os, "replace", controlled_replace)
    monkeypatch.setattr(artifacts.Path, "unlink", controlled_unlink)
    return state


def _expected_failure_events(
    stage: str,
    failure_position: int,
) -> list[tuple[int, str]]:
    events = [
        (position, event)
        for position in range(1, failure_position)
        for event in _ATOMIC_STAGES
    ]
    failed_sequences = {
        "create": ("create",),
        "write": ("create", "write", "close", "cleanup"),
        "flush": ("create", "write", "flush", "close", "cleanup"),
        "fsync": (
            "create",
            "write",
            "flush",
            "fsync",
            "close",
            "cleanup",
        ),
        "close": (
            "create",
            "write",
            "flush",
            "fsync",
            "close",
            "cleanup",
        ),
        "replace": (*_ATOMIC_STAGES, "cleanup"),
    }
    events.extend(
        (failure_position, event) for event in failed_sequences[stage]
    )
    return events


def test_exact_public_signature() -> None:
    signature = inspect.signature(write_eda_figures)
    assert tuple(signature.parameters) == ("figures", "output_dir")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert get_type_hints(write_eda_figures) == {
        "figures": dict[str, Figure],
        "output_dir": str | Path,
        "return": dict[str, Path],
    }


def test_exact_all_and_controlled_wildcard_export() -> None:
    assert figure_artifacts.__all__ == ("write_eda_figures",)
    namespace: dict[str, object] = {}
    exec("from src.analysis.figure_artifacts import *", namespace)
    exported = {
        name: value
        for name, value in namespace.items()
        if not name.startswith("__")
    }
    assert exported == {"write_eda_figures": write_eda_figures}


def test_complete_module_owned_binding_surface() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    names = _module_owned_names(tree)
    assert names.count("write_eda_figures") == 1
    assert names.count("__all__") == 1
    assert all(
        name == "write_eda_figures"
        or name == "__all__"
        or name.startswith("_")
        for name in names
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("PUBLIC = 1", ("PUBLIC",)),
        ("PUBLIC: int = 1", ("PUBLIC",)),
        ("PUBLIC += 1", ("PUBLIC",)),
        ("PUBLIC, _private = ({}, None)", ("PUBLIC", "_private")),
        ("[PUBLIC, _private] = ([], None)", ("PUBLIC", "_private")),
        ("*PUBLIC, _private = ()", ("PUBLIC", "_private")),
        ("class PublicWriter:\n    pass", ("PublicWriter",)),
        ("def public_helper():\n    pass", ("public_helper",)),
    ),
)
def test_binding_detector_covers_public_forms(
    source: str,
    expected: tuple[str, ...],
) -> None:
    assert _module_owned_names(ast.parse(source)) == expected


def test_binding_detector_covers_python312_type_alias() -> None:
    tree = ast.parse("type PublicFigureMap = dict[str, object]")
    assert _module_owned_names(tree) == ("PublicFigureMap",)


def test_exact_keys_filenames_final_paths_and_plain_dict(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "figures"
    result = write_eda_figures(_simple_figures(), output_dir)
    assert type(result) is dict
    assert tuple(result) == _EXPECTED_KEYS
    assert tuple(path.name for path in result.values()) == _EXPECTED_FILENAMES
    assert all(path.parent == output_dir for path in result.values())
    assert set(output_dir.iterdir()) == set(result.values())


@pytest.mark.parametrize(
    ("case", "error_type"),
    (
        ("dict_subclass", TypeError),
        ("non_dict_mapping", TypeError),
        ("missing_key", ValueError),
        ("extra_key", ValueError),
        ("reordered_keys", ValueError),
        ("non_string_key", TypeError),
        ("non_figure_value", TypeError),
        ("duplicate_figure", ValueError),
        ("duplicate_canvas", ValueError),
        ("non_agg_canvas", TypeError),
    ),
)
def test_malformed_bundle_is_rejected_without_filesystem_mutation(
    tmp_path: Path,
    case: str,
    error_type: type[Exception],
) -> None:
    figures = _invalid_bundle(case)
    snapshot = _bundle_snapshot(figures)
    output_dir = tmp_path / case
    with pytest.raises(error_type, match="figures|figure|canvas"):
        write_eda_figures(figures, output_dir)  # type: ignore[arg-type]
    assert not output_dir.exists()
    _assert_bundle_unchanged(figures, snapshot)


@pytest.mark.parametrize(
    "case",
    (
        "dict_subclass",
        "non_dict_mapping",
        "missing_key",
        "extra_key",
        "reordered_keys",
        "non_string_key",
        "non_figure_value",
        "duplicate_figure",
        "duplicate_canvas",
        "non_agg_canvas",
    ),
)
def test_complete_validation_precedes_any_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    render_calls = 0

    def forbidden_render(*args: object, **kwargs: object) -> None:
        nonlocal render_calls
        render_calls += 1
        raise AssertionError("invalid bundle reached rendering")

    monkeypatch.setattr(FigureCanvasAgg, "print_png", forbidden_render)
    with pytest.raises((TypeError, ValueError)):
        write_eda_figures(
            _invalid_bundle(case),  # type: ignore[arg-type]
            tmp_path / case,
        )
    assert render_calls == 0
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize("failure_position", (1, 3, 5))
def test_render_failure_matrix_precedes_all_filesystem_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    figures = _simple_figures()
    snapshot = _bundle_snapshot(figures)
    canvas_keys = {
        id(figure.canvas): key for key, figure in figures.items()
    }
    rendered: list[str] = []
    error = RuntimeError(f"render failure at {failure_position}")
    original = FigureCanvasAgg.print_png

    def tracked_render(
        canvas: FigureCanvasAgg,
        target: object,
        **kwargs: object,
    ) -> None:
        rendered.append(canvas_keys[id(canvas)])
        if len(rendered) == failure_position:
            raise error
        original(canvas, target, **kwargs)

    monkeypatch.setattr(FigureCanvasAgg, "print_png", tracked_render)
    output_dir = tmp_path / f"render-failure-{failure_position}"
    with pytest.raises(RuntimeError) as caught:
        write_eda_figures(figures, output_dir)
    assert caught.value is error
    assert rendered == list(_EXPECTED_KEYS[:failure_position])
    assert not output_dir.exists()
    assert tuple(tmp_path.iterdir()) == ()
    _assert_bundle_unchanged(figures, snapshot)


def test_exact_png_structure_metadata_and_authentic_dimensions(
    tmp_path: Path,
    _authentic_figures: dict[str, Figure],
) -> None:
    output_dir = tmp_path / "authentic"
    result = write_eda_figures(_authentic_figures, output_dir)
    assert tuple(result) == _EXPECTED_KEYS
    assert tuple(path.name for path in result.values()) == _EXPECTED_FILENAMES
    assert set(output_dir.iterdir()) == set(result.values())
    assert len(tuple(output_dir.iterdir())) == 5
    for path, dimensions in zip(result.values(), _EXPECTED_DIMENSIONS):
        _assert_valid_png(path.read_bytes(), dimensions)


def test_textual_metadata_parser_accepts_only_approved_text_entry() -> None:
    chunks = ((
        b"tEXt",
        b"Software\x00dental-appointment-no-show-prediction",
    ),)
    assert _decode_textual_metadata(chunks) == {
        "Software": "dental-appointment-no-show-prediction"
    }
    _assert_approved_textual_metadata(chunks)


def test_textual_metadata_parser_exposes_extra_author_entry() -> None:
    chunks = (
        (
            b"tEXt",
            b"Software\x00dental-appointment-no-show-prediction",
        ),
        (b"tEXt", b"Author\x00reviewer"),
    )
    assert _decode_textual_metadata(chunks) == {
        "Software": "dental-appointment-no-show-prediction",
        "Author": "reviewer",
    }
    with pytest.raises(AssertionError):
        _assert_approved_textual_metadata(chunks)


def test_textual_metadata_parser_rejects_duplicate_software() -> None:
    chunks = (
        (
            b"tEXt",
            b"Software\x00dental-appointment-no-show-prediction",
        ),
        (
            b"zTXt",
            b"Software\x00\x00" + zlib.compress(b"duplicate"),
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        _decode_textual_metadata(chunks)


def test_textual_metadata_parser_exposes_additional_ztxt_entry() -> None:
    chunks = (
        (
            b"tEXt",
            b"Software\x00dental-appointment-no-show-prediction",
        ),
        (
            b"zTXt",
            b"Comment\x00\x00" + zlib.compress(b"stable environment"),
        ),
    )
    metadata = _decode_textual_metadata(chunks)
    assert metadata["Comment"] == "stable environment"
    with pytest.raises(AssertionError):
        _assert_approved_textual_metadata(chunks)


def test_textual_metadata_parser_exposes_additional_itxt_entry() -> None:
    chunks = (
        (
            b"tEXt",
            b"Software\x00dental-appointment-no-show-prediction",
        ),
        (
            b"iTXt",
            b"Creation Time\x00\x01\x00en\x00Created\x00"
            + zlib.compress("2026-08-03T12:00:00Z".encode("utf-8")),
        ),
    )
    metadata = _decode_textual_metadata(chunks)
    assert metadata["Creation Time"] == "2026-08-03T12:00:00Z"
    with pytest.raises(AssertionError):
        _assert_approved_textual_metadata(chunks)


@pytest.mark.parametrize(
    "chunk",
    (
        (b"tEXt", b"Software-without-null"),
        (b"tEXt", b"\x00empty-keyword"),
        (b"tEXt", b" Software\x00leading-space"),
        (b"zTXt", b"Comment\x00"),
        (b"zTXt", b"Comment\x00\x01invalid-method"),
        (b"zTXt", b"Comment\x00\x00not-zlib"),
        (b"iTXt", b"Comment\x00"),
        (b"iTXt", b"Comment\x00\x02\x00en\x00\x00text"),
        (b"iTXt", b"Comment\x00\x00\x01en\x00\x00text"),
        (b"iTXt", b"Comment\x00\x00\x00missing-separators"),
        (b"iTXt", b"Comment\x00\x00\x00en\x00\x00\xff"),
    ),
)
def test_textual_metadata_parser_rejects_malformed_chunks(
    chunk: tuple[bytes, bytes],
) -> None:
    with pytest.raises(ValueError):
        _decode_textual_metadata((chunk,))


def test_bytes_are_deterministic_across_directories(tmp_path: Path) -> None:
    figures = _simple_figures()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = write_eda_figures(figures, first_dir)
    second = write_eda_figures(figures, second_dir)
    assert tuple(first) == tuple(second) == _EXPECTED_KEYS
    assert _read_artifacts(first_dir) == _read_artifacts(second_dir)


def test_repeated_overwrite_is_byte_deterministic(tmp_path: Path) -> None:
    figures = _simple_figures()
    output_dir = tmp_path / "repeated"
    first = write_eda_figures(figures, output_dir)
    first_bytes = _read_artifacts(output_dir)
    second = write_eda_figures(figures, output_dir)
    assert first == second
    assert _read_artifacts(output_dir) == first_bytes
    assert {path.name for path in output_dir.iterdir()} == set(
        _EXPECTED_FILENAMES
    )


def test_fresh_visually_identical_bundles_have_identical_bytes(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "fresh-first"
    second_dir = tmp_path / "fresh-second"
    write_eda_figures(_simple_figures(), first_dir)
    write_eda_figures(_simple_figures(), second_dir)
    assert _read_artifacts(first_dir) == _read_artifacts(second_dir)


def test_current_working_directory_does_not_affect_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    figures = _simple_figures()
    first_dir = tmp_path / "cwd-first"
    second_dir = tmp_path / "cwd-second"
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    write_eda_figures(figures, first_dir)
    monkeypatch.chdir(elsewhere)
    write_eda_figures(figures, second_dir)
    assert _read_artifacts(first_dir) == _read_artifacts(second_dir)


def test_existing_unrelated_content_is_preserved(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    note = output_dir / "notes.txt"
    note.write_bytes(b"preserve this\n")
    supporting = output_dir / "supporting"
    supporting.mkdir()
    nested = supporting / "source.bin"
    nested.write_bytes(b"preserve nested\x00")
    write_eda_figures(_simple_figures(), output_dir)
    assert note.read_bytes() == b"preserve this\n"
    assert nested.read_bytes() == b"preserve nested\x00"
    assert supporting.is_dir()


def test_existing_approved_pngs_are_replaced_without_semantic_mutation(
    tmp_path: Path,
) -> None:
    figures = _simple_figures()
    snapshot = _bundle_snapshot(figures)
    reference = tmp_path / "reference"
    expected = write_eda_figures(figures, reference)
    expected_bytes = {key: path.read_bytes() for key, path in expected.items()}
    output_dir = tmp_path / "replace"
    output_dir.mkdir()
    for filename in _EXPECTED_FILENAMES:
        (output_dir / filename).write_bytes(b"incorrect PNG")
    actual = write_eda_figures(figures, output_dir)
    assert {key: path.read_bytes() for key, path in actual.items()} == (
        expected_bytes
    )
    _assert_bundle_unchanged(figures, snapshot)


def test_output_dir_existing_as_file_is_preserved(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    original = b"existing file\x00content"
    output_file.write_bytes(original)
    with pytest.raises(NotADirectoryError, match="not a directory"):
        write_eda_figures(_simple_figures(), output_file)
    assert output_file.is_file()
    assert output_file.read_bytes() == original


def test_invalid_output_path_type_is_rejected_without_filesystem_change(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())
    with pytest.raises(TypeError):
        write_eda_figures(_simple_figures(), object())  # type: ignore[arg-type]
    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.parametrize("failure_position", (1, 3, 5))
def test_replace_failure_matrix_uses_atomic_helper_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    figures = _simple_figures()
    snapshot = _bundle_snapshot(figures)
    output_dir = tmp_path / f"replace-failure-{failure_position}"
    error = OSError(f"replace failure at {failure_position}")
    original_replace = artifacts.os.replace
    attempted: list[Path] = []

    def fail_selected(source: Path, destination: Path) -> None:
        attempted.append(Path(destination))
        if len(attempted) == failure_position:
            raise error
        original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_selected)
    with pytest.raises(OSError) as caught:
        write_eda_figures(figures, output_dir)
    assert caught.value is error
    assert tuple(path.name for path in attempted) == (
        _EXPECTED_FILENAMES[:failure_position]
    )
    _assert_only_completed(output_dir, failure_position - 1)
    assert not getattr(error, "__notes__", ())
    _assert_bundle_unchanged(figures, snapshot)


@pytest.mark.parametrize("failure_position", (1, 3, 5))
def test_fsync_failure_matrix_uses_atomic_helper_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
) -> None:
    figures = _simple_figures()
    snapshot = _bundle_snapshot(figures)
    output_dir = tmp_path / f"fsync-failure-{failure_position}"
    error = OSError(f"fsync failure at {failure_position}")
    original_fsync = artifacts.os.fsync
    original_replace = artifacts.os.replace
    fsync_calls = 0
    replaced: list[Path] = []

    def fail_selected(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == failure_position:
            raise error
        original_fsync(file_descriptor)

    def record_replace(source: Path, destination: Path) -> None:
        replaced.append(Path(destination))
        original_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "fsync", fail_selected)
    monkeypatch.setattr(artifacts.os, "replace", record_replace)
    with pytest.raises(OSError) as caught:
        write_eda_figures(figures, output_dir)
    assert caught.value is error
    assert fsync_calls == failure_position
    assert tuple(path.name for path in replaced) == (
        _EXPECTED_FILENAMES[: failure_position - 1]
    )
    _assert_only_completed(output_dir, failure_position - 1)
    assert not getattr(error, "__notes__", ())
    _assert_bundle_unchanged(figures, snapshot)


@pytest.mark.parametrize("stage", _ATOMIC_STAGES)
@pytest.mark.parametrize("failure_position", _FAILURE_POSITIONS)
def test_complete_atomic_failure_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _expected_simple_pngs: dict[str, bytes],
    stage: str,
    failure_position: int,
) -> None:
    output_dir = (
        tmp_path
        / "nested parent with spaces"
        / "خروجی شکل‌ها"
        / f"{stage}-{failure_position}"
    )
    output_dir.mkdir(parents=True)
    unrelated_file = output_dir / "reviewer notes.txt"
    unrelated_file.write_bytes(b"preserve unrelated file\n")
    unrelated_dir = output_dir / "پشتیبان"
    unrelated_dir.mkdir()
    nested_file = unrelated_dir / "source.bin"
    nested_file.write_bytes(b"preserve nested\x00")

    figures = _simple_figures()
    bundle_before = _bundle_snapshot(figures)
    matplotlib_before = _matplotlib_state()
    cwd_before = Path.cwd()
    error = OSError(f"unique {stage} failure at {failure_position}")
    state = _install_atomic_failure(
        monkeypatch,
        stage,
        failure_position,
        error,
    )
    result: dict[str, Path] | None = None

    with pytest.raises(OSError) as caught:
        result = write_eda_figures(figures, output_dir)

    assert caught.value is error
    assert result is None
    assert state["create_count"] == failure_position
    assert state["events"] == _expected_failure_events(
        stage,
        failure_position,
    )
    assert not getattr(error, "__notes__", ())

    completed = failure_position - 1
    for position, (filename, dimensions) in enumerate(
        zip(_EXPECTED_FILENAMES, _EXPECTED_DIMENSIONS),
        start=1,
    ):
        path = output_dir / filename
        if position <= completed:
            payload = path.read_bytes()
            assert payload == _expected_simple_pngs[filename]
            _assert_valid_png(payload, dimensions)
        else:
            assert not path.exists()
    assert unrelated_file.read_bytes() == b"preserve unrelated file\n"
    assert nested_file.read_bytes() == b"preserve nested\x00"
    assert unrelated_dir.is_dir()
    assert not any(path.suffix == ".tmp" for path in output_dir.rglob("*"))
    assert not any(path.exists() for path in state["temporary_paths"])

    if stage == "create":
        assert len(state["temporary_paths"]) == completed
        assert state["cleanup_paths"] == []
    else:
        assert len(state["temporary_paths"]) == failure_position
        assert len(state["cleanup_paths"]) == 1

    resolved_output = output_dir.resolve()
    observed_paths = (
        state["create_dirs"]
        + state["temporary_paths"]
        + state["cleanup_paths"]
        + state["replace_sources"]
        + state["replace_destinations"]
    )
    assert observed_paths
    assert all(
        Path(path).resolve().is_relative_to(resolved_output)
        for path in observed_paths
    )
    _assert_bundle_unchanged(figures, bundle_before)
    assert _matplotlib_state() == matplotlib_before
    assert Path.cwd() == cwd_before


@pytest.mark.parametrize("failed_operation", ("fsync", "replace"))
def test_cleanup_failure_note_preserves_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_operation: str,
) -> None:
    figures = _simple_figures()
    snapshot = _bundle_snapshot(figures)
    output_dir = tmp_path / f"cleanup-failure-{failed_operation}"
    original_error = OSError(f"original {failed_operation} failure")
    cleanup_error = PermissionError("controlled unlink denial")
    leftovers: list[Path] = []

    def fail_operation(*args: object, **kwargs: object) -> None:
        raise original_error

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        assert missing_ok
        leftovers.append(path)
        raise cleanup_error

    with monkeypatch.context() as failure_patches:
        failure_patches.setattr(
            artifacts.os,
            failed_operation,
            fail_operation,
        )
        failure_patches.setattr(artifacts.Path, "unlink", fail_cleanup)
        with pytest.raises(OSError) as caught:
            write_eda_figures(figures, output_dir)
    assert caught.value is original_error
    notes = "\n".join(getattr(original_error, "__notes__", ()))
    assert "cleanup failed" in notes.lower()
    assert "PermissionError" in notes
    assert "controlled unlink denial" in notes
    assert len(leftovers) == 1
    assert leftovers[0].parent == output_dir
    assert leftovers[0].exists()
    assert not (output_dir / _EXPECTED_FILENAMES[0]).exists()
    _assert_bundle_unchanged(figures, snapshot)
    leftovers[0].unlink()


@pytest.mark.parametrize("primary_stage", ("write", "fsync", "replace"))
def test_cleanup_failure_preserves_write_fsync_and_replace_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _expected_simple_pngs: dict[str, bytes],
    primary_stage: str,
) -> None:
    output_dir = (
        tmp_path / "cleanup paths with spaces" / "یونیکد" / primary_stage
    )
    figures = _simple_figures()
    bundle_before = _bundle_snapshot(figures)
    matplotlib_before = _matplotlib_state()
    cwd_before = Path.cwd()
    primary_error = OSError(f"unique primary {primary_stage} failure")
    cleanup_error = PermissionError(
        f"unique cleanup denial after {primary_stage}"
    )
    result: dict[str, Path] | None = None

    with monkeypatch.context() as failure_patches:
        state = _install_atomic_failure(
            failure_patches,
            primary_stage,
            3,
            primary_error,
            cleanup_error,
        )
        with pytest.raises(OSError) as caught:
            result = write_eda_figures(figures, output_dir)

    assert caught.value is primary_error
    assert caught.value is not cleanup_error
    assert caught.value.__cause__ is None
    assert result is None
    notes = "\n".join(getattr(primary_error, "__notes__", ()))
    assert "cleanup failed" in notes.lower()
    assert "PermissionError" in notes
    assert str(cleanup_error) in notes
    assert state["create_count"] == 3
    assert state["events"] == _expected_failure_events(primary_stage, 3)
    assert len(state["cleanup_paths"]) == 1
    leftover = state["cleanup_paths"][0]
    assert leftover.exists()
    assert leftover == state["temporary_paths"][-1]

    for position, filename in enumerate(_EXPECTED_FILENAMES, start=1):
        path = output_dir / filename
        if position < 3:
            assert path.read_bytes() == _expected_simple_pngs[filename]
        else:
            assert not path.exists()
    resolved_output = output_dir.resolve()
    assert all(
        Path(path).resolve().is_relative_to(resolved_output)
        for path in (
            state["temporary_paths"]
            + state["cleanup_paths"]
            + state["replace_sources"]
            + state["replace_destinations"]
        )
    )
    _assert_bundle_unchanged(figures, bundle_before)
    assert _matplotlib_state() == matplotlib_before
    assert Path.cwd() == cwd_before

    leftover.unlink()
    assert not any(path.suffix == ".tmp" for path in output_dir.rglob("*"))


def test_atomic_order_and_temporary_file_confinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "atomic-order"
    events: list[str] = []
    temporary_paths: list[Path] = []
    wrapped_files: list[Any] = []
    destinations: list[Path] = []
    original_temporary_file = artifacts.tempfile.NamedTemporaryFile
    original_fsync = artifacts.os.fsync
    original_replace = artifacts.os.replace

    class _RecordingTemporaryFile:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self.name = wrapped.name

        def __enter__(self) -> _RecordingTemporaryFile:
            return self

        def __exit__(self, *args: object) -> None:
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

    def create_temporary(*args: object, **kwargs: object) -> object:
        events.append("create")
        assert Path(kwargs["dir"]) == output_dir
        wrapped = original_temporary_file(*args, **kwargs)
        recording = _RecordingTemporaryFile(wrapped)
        temporary_paths.append(Path(recording.name))
        wrapped_files.append(recording)
        return recording

    def record_fsync(file_descriptor: int) -> None:
        events.append("fsync")
        original_fsync(file_descriptor)

    def record_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        index = len(destinations)
        assert Path(source) == temporary_paths[index]
        assert wrapped_files[index].closed
        destinations.append(Path(destination))
        original_replace(source, destination)

    monkeypatch.setattr(
        artifacts.tempfile,
        "NamedTemporaryFile",
        create_temporary,
    )
    monkeypatch.setattr(artifacts.os, "fsync", record_fsync)
    monkeypatch.setattr(artifacts.os, "replace", record_replace)
    write_eda_figures(_simple_figures(), output_dir)
    assert events == [
        event
        for _ in _EXPECTED_FILENAMES
        for event in ("create", "write", "flush", "fsync", "close", "replace")
    ]
    assert tuple(path.name for path in destinations) == _EXPECTED_FILENAMES
    assert all(path.parent == output_dir for path in temporary_paths)
    assert not any(path.exists() for path in temporary_paths)


def test_no_direct_final_file_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DirectFinalWriteCalled(Exception):
        pass

    figures = _simple_figures()
    calls: list[tuple[object, ...]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        calls.append(args)
        raise _DirectFinalWriteCalled(
            f"direct filesystem writer called: {args!r} {kwargs!r}"
        )

    with monkeypatch.context() as writer_guards:
        writer_guards.setattr(builtins, "open", forbidden)
        writer_guards.setattr(Figure, "savefig", forbidden)
        writer_guards.setattr(Path, "open", forbidden)
        writer_guards.setattr(Path, "write_bytes", forbidden)
        writer_guards.setattr(Path, "write_text", forbidden)
        result = write_eda_figures(figures, tmp_path / "indirect")

    assert calls == []
    assert tuple(result) == _EXPECTED_KEYS


def test_all_final_and_temporary_paths_stay_under_output_dir(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "container" / "nested" / "figures"
    result = write_eda_figures(_simple_figures(), str(output_dir))
    resolved_output = output_dir.resolve()
    created_files = tuple(
        path.resolve() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert set(created_files) == {path.resolve() for path in result.values()}
    assert all(path.is_relative_to(resolved_output) for path in created_files)


def test_visible_semantics_and_input_dict_survive_success_and_repeat(
    tmp_path: Path,
    _authentic_figures: dict[str, Figure],
) -> None:
    snapshot = _bundle_snapshot(_authentic_figures)
    write_eda_figures(_authentic_figures, tmp_path / "success")
    _assert_bundle_unchanged(_authentic_figures, snapshot)
    write_eda_figures(_authentic_figures, tmp_path / "success")
    _assert_bundle_unchanged(_authentic_figures, snapshot)


def test_each_success_returns_a_fresh_independent_dict(tmp_path: Path) -> None:
    figures = _simple_figures()
    output_dir = tmp_path / "fresh-return"
    first = write_eda_figures(figures, output_dir)
    second = write_eda_figures(figures, output_dir)
    assert first is not second
    assert first == second
    first.clear()
    third = write_eda_figures(figures, output_dir)
    assert tuple(third) == _EXPECTED_KEYS
    assert len(tuple(output_dir.iterdir())) == 5


def test_backend_rcparams_and_managers_unchanged_across_writer_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _matplotlib_state()
    figures = _simple_figures()
    write_eda_figures(figures, tmp_path / "global-success")
    write_eda_figures(figures, tmp_path / "global-success")
    assert _matplotlib_state() == before

    error = RuntimeError("controlled render failure")

    def fail_render(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(FigureCanvasAgg, "print_png", fail_render)
    with pytest.raises(RuntimeError) as caught:
        write_eda_figures(figures, tmp_path / "global-failure")
    assert caught.value is error
    assert _matplotlib_state() == before


def test_reload_has_no_global_or_filesystem_side_effect(tmp_path: Path) -> None:
    before = _matplotlib_state()
    before_entries = tuple(tmp_path.rglob("*"))
    reloaded = reload(figure_artifacts)
    assert reloaded is figure_artifacts
    assert _matplotlib_state() == before
    assert tuple(tmp_path.rglob("*")) == before_entries


def test_isolated_import_has_no_filesystem_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    spec = util.spec_from_file_location(
        "isolated_eda_figure_artifacts",
        _MODULE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)
    assert tuple(tmp_path.rglob("*")) == ()


@pytest.mark.parametrize(
    "source",
    (
        (
            "def _write(path, payload):\n"
            '    open(path, "wb").write(payload)\n'
        ),
        (
            "def _write(path, payload):\n"
            '    with open(path, mode="wb") as handle:\n'
            "        handle.write(payload)\n"
        ),
        (
            "def _write(paths, key, payload):\n"
            '    open(paths[key], "wb").write(payload)\n'
        ),
        (
            "from builtins import open as _builtin_open\n"
            "def _write(path, payload):\n"
            '    _builtin_open(path, "w").write(payload)\n'
        ),
        (
            "import builtins as _builtins\n"
            "def _write(path, payload):\n"
            '    _builtins.open(path, "wb").write(payload)\n'
        ),
    ),
)
def test_direct_writer_detector_rejects_builtins_open(source: str) -> None:
    violations = _direct_writer_violations(ast.parse(source))
    assert "builtins.open" in violations or "open" in violations


@pytest.mark.parametrize(
    ("source", "violation"),
    (
        ('Path(path).open("wb")', "open"),
        ('path.write_bytes(b"payload")', "write_bytes"),
        ('path.write_text("payload")', "write_text"),
        ('figure.savefig(path)', "savefig"),
        ('frame.to_csv(path)', "to_csv"),
        ('canvas.print_png(path)', "print_png-filesystem"),
    ),
)
def test_direct_writer_detector_rejects_attribute_writers(
    source: str,
    violation: str,
) -> None:
    assert violation in _direct_writer_violations(ast.parse(source))


def test_direct_writer_detector_allows_in_memory_png_and_harmless_text() -> None:
    source = (
        "from io import BytesIO\n"
        "def _render(canvas):\n"
        "    buffer = BytesIO()\n"
        "    canvas.print_png(buffer)\n"
        "    message = 'open(path, \\\"wb\\\") is only text'\n"
        "    return buffer.getvalue(), message\n"
    )
    assert _direct_writer_violations(ast.parse(source)) == ()


def test_static_leakage_dependency_and_writer_guards() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    referenced_names: set[str] = set()
    strings: list[str] = []
    called_attributes: set[str] = set()
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
            strings.append(node.value.lower().replace("\\", "/"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)

    assert imported_modules <= {
        "__future__",
        "io",
        "pathlib",
        "matplotlib.backends.backend_agg",
        "matplotlib.figure",
        "src.analysis.artifacts",
    }
    forbidden_names = {
        "build_eda_figures",
        "build_eda_tables",
        "select_eda_populations",
        "summarize",
        "summaries",
        "canonical",
        "target",
        "validation",
        "validation_labels",
        "test_data",
        "test_population",
        "modeling",
        "ArgumentParser",
        "parse_args",
        "pyplot",
        "plt",
        "use",
        "switch_backend",
        "rcParams",
        "Gcf",
    }
    assert not referenced_names & forbidden_names
    assert not any(name.startswith("summarize_") for name in referenced_names)
    assert "_write_atomic" in referenced_names
    assert "print_png" in called_attributes
    assert _direct_writer_violations(tree) == ()
    assert not called_attributes & {
        "savefig",
        "write_bytes",
        "write_text",
        "open",
    }
    assert not any("reports/eda" in value for value in strings)
    assert not any("data/processed" in value for value in strings)


def test_production_imports_and_reuses_existing_private_atomic_helper() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "src.analysis.artifacts"
    ]
    assert len(imports) == 1
    assert tuple(alias.name for alias in imports[0].names) == ("_write_atomic",)
    assert figure_artifacts._write_atomic is artifacts._write_atomic
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_atomic"
    ]
    assert len(calls) == 1
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_cleanup_temporary_file", "_write_atomic"}
        for node in tree.body
    )
