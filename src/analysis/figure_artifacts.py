"""Deterministically serialize the approved in-memory EDA figure bundle."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.analysis.artifacts import _write_atomic


__all__ = ("write_eda_figures",)


_FIGURE_ARTIFACTS = (
    ("class_balance", "01_class_balance.png"),
    ("temporal_monthly", "02_temporal_monthly.png"),
    ("numeric_drift", "03_numeric_drift.png"),
    ("categorical_drift", "04_categorical_drift.png"),
    ("numeric_relationships", "05_numeric_relationships.png"),
)
_FIGURE_KEYS = tuple(key for key, _ in _FIGURE_ARTIFACTS)
_SOFTWARE = "dental-appointment-no-show-prediction"


def _validate_figures(figures: object) -> tuple[Figure, ...]:
    if type(figures) is not dict:
        raise TypeError("figures must be exactly a built-in dict")

    keys = tuple(figures)
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("figures keys must all be strings")
    if keys != _FIGURE_KEYS:
        raise ValueError(
            "figures keys and insertion order must be exactly "
            f"{_FIGURE_KEYS}; got {keys}"
        )

    bundle = tuple(figures[key] for key in _FIGURE_KEYS)
    for key, figure in zip(_FIGURE_KEYS, bundle):
        if not isinstance(figure, Figure):
            raise TypeError(f"figures[{key!r}] must be a Matplotlib Figure")
    if len({id(figure) for figure in bundle}) != len(bundle):
        raise ValueError("each figure key must reference a distinct Figure object")

    canvases = tuple(figure.canvas for figure in bundle)
    for key, canvas in zip(_FIGURE_KEYS, canvases):
        if not isinstance(canvas, FigureCanvasAgg):
            raise TypeError(
                f"figures[{key!r}] must have a FigureCanvasAgg canvas"
            )
    if len({id(canvas) for canvas in canvases}) != len(canvases):
        raise ValueError("each figure must have a distinct canvas object")
    return bundle


def _render_png(figure: Figure) -> bytes:
    buffer = BytesIO()
    figure.canvas.print_png(buffer, metadata={"Software": _SOFTWARE})
    return buffer.getvalue()


def write_eda_figures(
    figures: dict[str, Figure],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write one validated EDA figure bundle as five deterministic PNG files."""

    bundle = _validate_figures(figures)
    payloads = tuple(_render_png(figure) for figure in bundle)

    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(
            f"output_dir exists and is not a directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)

    paths = {
        key: directory / filename
        for key, filename in _FIGURE_ARTIFACTS
    }
    for key, payload in zip(_FIGURE_KEYS, payloads):
        _write_atomic(payload, paths[key])
    return paths
