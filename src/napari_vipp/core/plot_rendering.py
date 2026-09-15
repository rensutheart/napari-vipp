"""Shared, Qt-free rendering and staged publication of measurement figures.

The plot recipe and exact prepared values remain authoritative. Interactive
point thinning is explicitly requested by the caller; export always uses every
prepared point. No source image or recorded source path is opened.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from napari_vipp.core.measurement_collection import _check, _ordinary_path
from napari_vipp.core.measurement_export import (
    _publish,
    _RecoveryFailed,
    measurement_export_destination_revisions,
)

PLOT_COLORS = ("#278AC7", "#D88027", "#289D8F", "#AA6BC4", "#CB536C", "#80733D")
PLOT_MARKERS = ("o", "D", "s", "^", "v", "P")
PLOT_JITTER_SEED = 1729


def build_plot_figure(
    result,
    *,
    size_inches=(6.8, 4.6),
    dpi=100,
    publication=False,
    colors=None,
    display_only=False,
    compact=False,
):
    """Build a Figure without pyplot, a GUI backend or global style mutation.

    ``colors`` accepts background/text/grid hex values from the UI palette.
    A light publication background overrides that palette. Pickable scatter
    artists carry ``vipp_source_rows`` for object identity inspection.
    """
    palette = {"background": "#FFFFFF", "text": "#273644", "grid": "#D3DBE0"}
    if colors and not publication:
        palette.update(colors)
    figure = Figure(figsize=size_inches, dpi=dpi, layout="constrained")
    figure.set_facecolor(palette["background"])
    axes = figure.add_subplot(111)
    axes.set_facecolor(palette["background"])
    recipe = result.recipe
    font_size = 8 if compact else 10
    axes.tick_params(colors=palette["text"], labelsize=font_size)
    for side, spine in axes.spines.items():
        spine.set_visible(side in {"left", "bottom"})
        spine.set_color(palette["grid"])
    axes.set_axisbelow(True)
    if recipe.show_grid:
        axes.grid(axis="y", color=palette["grid"], linewidth=0.6, alpha=0.6)

    for index, series in enumerate(result.series):
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        marker = PLOT_MARKERS[index % len(PLOT_MARKERS)]
        indices = (
            tuple(series.display_indices)
            if display_only
            else tuple(range(len(series.y)))
        )
        x = np.asarray(series.x, dtype=float)
        y = np.asarray(series.y, dtype=float)
        if recipe.plot_type in {"Compare groups", "Scatter"}:
            if recipe.plot_type == "Compare groups":
                # Seeded jitter avoids artificial diagonal patterns for ordered
                # measurements. It is a visual aid, never scientific data.
                jitter = np.random.default_rng(PLOT_JITTER_SEED + index).uniform(
                    -0.21, 0.21, len(y)
                )
                x = np.full(len(y), index, dtype=float) + jitter
            selected = np.asarray(indices, dtype=int)
            artist = axes.scatter(
                x[selected],
                y[selected],
                s=float(recipe.point_size) ** 2,
                c=color,
                marker=marker,
                alpha=0.7,
                edgecolors=palette["background"],
                linewidths=0.35,
                label=series.name,
                picker=5,
            )
            artist.vipp_source_rows = tuple(series.source_rows[i] for i in indices)
            artist.vipp_point_values = tuple((x[i], y[i]) for i in indices)
            if recipe.plot_type == "Compare groups" and recipe.summary != "None":
                summary = series.mean if recipe.summary == "Mean" else series.median
                if summary is not None and math.isfinite(summary):
                    axes.plot(
                        [index - 0.25, index + 0.25],
                        [summary, summary],
                        color=palette["text"],
                        linewidth=2.3,
                        solid_capstyle="round",
                    )
        elif recipe.distribution == "Histogram":
            # One shared step outline, not opaque outlines on every narrow bin.
            axes.stairs(
                series.histogram_values,
                series.bin_edges,
                color=color,
                linewidth=1.65,
                label=series.name,
            )
            axes.stairs(
                series.histogram_values,
                series.bin_edges,
                color=color,
                fill=True,
                alpha=0.12,
                linewidth=0,
            )
        else:
            axes.step(
                series.ecdf_x,
                series.ecdf_y,
                where="post",
                color=color,
                linewidth=1.8,
                label=series.name,
            )

    if recipe.plot_type == "Compare groups":
        axes.set_xticks(range(len(result.series)))
        axes.set_xticklabels([series.name for series in result.series])
        axes.set_xlim(-0.6, max(len(result.series) - 0.4, 0.6))
    elif not recipe.log_x:
        axes.xaxis.set_major_locator(MaxNLocator(nbins=4 if compact else 6))
    if recipe.log_x and recipe.plot_type != "Compare groups":
        axes.set_xscale("log")
    if recipe.log_y:
        axes.set_yscale("log")
    else:
        axes.yaxis.set_major_locator(MaxNLocator(nbins=4 if compact else 6))
    if recipe.plot_type == "Distribution" and not recipe.log_y:
        axes.set_ylim(bottom=0)
    title = (
        recipe.title
        or {
            "Compare groups": "Compare measurements",
            "Distribution": "Measurement distribution",
            "Scatter": "Compare two measurements",
        }[recipe.plot_type]
    )
    axes.set_title(title, color=palette["text"], fontsize=font_size + 2, pad=12)
    axes.set_xlabel(result.x_label, color=palette["text"], fontsize=font_size)
    axes.set_ylabel(result.y_label, color=palette["text"], fontsize=font_size)
    for text in (
        axes.title,
        axes.xaxis.label,
        axes.yaxis.label,
        *axes.get_xticklabels(),
    ):
        text.set_parse_math(False)
    if recipe.plot_type != "Compare groups" and len(result.series) > 1:
        legend = axes.legend(
            loc="best",
            frameon=False,
            fontsize=font_size,
            labelcolor=palette["text"],
            handlelength=1.6,
        )
        for text in legend.get_texts():
            text.set_parse_math(False)
    if not result.counts.plotted_points:
        axes.text(
            0.5,
            0.5,
            "No eligible measurements",
            transform=axes.transAxes,
            ha="center",
            va="center",
            color=palette["text"],
        )
    return figure


@dataclass(frozen=True)
class PlotExportResult:
    paths: tuple[Path, ...]
    notes: tuple[str, ...]


def plot_export_targets(path, *, include_data=False):
    """List figure and optional companion destinations before confirmation."""
    if not str(path).strip():
        raise ValueError("Choose a figure file name.")
    target = Path(path)
    if target.suffix.lower() not in {".png", ".tif", ".tiff", ".svg", ".pdf"}:
        raise ValueError("Choose PNG, TIFF, SVG or PDF for the figure.")
    if ".." in target.parts or str(target).startswith(("\\\\", "//")):
        raise ValueError("Choose a direct local export path.")
    target = target.absolute()
    if not include_data:
        return (target,)
    return (
        target,
        target.with_name(f"{target.stem}-plotted-data.csv"),
        target.with_name(f"{target.stem}-plot-settings.json"),
    )


def _destination_key(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def export_plot_result(
    result,
    path,
    *,
    width_mm=160,
    height_mm=110,
    dpi=300,
    include_data=False,
    protected_paths=(),
    overwrite=False,
    expected_destination_revisions=None,
    cancellation=None,
    is_current=None,
):
    """Export all prepared values at explicit dimensions on a light background.

    Each file is atomically published; companion publication rolls back ordinary
    errors as a group, but is not a crash-atomic multi-file transaction. CSV is
    plain text: import identifier/formula-like columns as text in spreadsheets.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    _check(cancellation)
    if is_current is not None and not is_current():
        raise ValueError("This plot is out of date. Calculate again before exporting.")
    sizes = (width_mm, height_mm, dpi)
    if any(isinstance(v, bool) or not math.isfinite(float(v)) for v in sizes):
        raise ValueError("Figure dimensions and resolution must be finite numbers.")
    if not (20 <= width_mm <= 500 and 20 <= height_mm <= 500 and 72 <= dpi <= 1200):
        raise ValueError("Choose 20–500 mm dimensions and 72–1200 DPI.")
    if width_mm / 25.4 * dpi * height_mm / 25.4 * dpi > 100_000_000:
        raise ValueError("The figure exceeds 100 million pixels. Reduce size or DPI.")
    if type(overwrite) is not bool or type(include_data) is not bool:
        raise ValueError("Overwrite and companion choices must be explicit.")
    targets = plot_export_targets(path, include_data=include_data)
    protected = {_destination_key(p) for p in protected_paths if str(p)}
    for target in targets:
        _ordinary_path(target, may_be_missing=True)
        if _destination_key(target) in protected:
            raise ValueError("A figure must not overwrite a source or workflow file.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Export already exists: {target}")
    current = measurement_export_destination_revisions(targets)
    expected = (
        current
        if expected_destination_revisions is None
        else expected_destination_revisions
    )
    if current != expected:
        raise ValueError("An export destination changed after review. Review it again.")
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".vipp-figure-", dir=targets[0].parent))
    keep_recovery = False
    figure = None
    try:
        staged = tuple(directory / target.name for target in targets)
        figure = build_plot_figure(
            result,
            size_inches=(width_mm / 25.4, height_mm / 25.4),
            dpi=dpi,
            publication=True,
            display_only=False,
        )
        FigureCanvasAgg(figure)
        _check(cancellation)
        figure.savefig(staged[0], dpi=dpi, facecolor="white")
        _check(cancellation)
        if include_data:
            table = result.plotted_table
            with staged[1].open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(table.columns)
                writer.writerows(table.rows)
                stream.flush()
                os.fsync(stream.fileno())
            settings = {
                "format": "VIPP plot settings",
                "version": 1,
                "recipe": result.recipe.to_params(),
                "counts": asdict(result.counts),
                "warnings": list(result.warnings),
                "column_units": dict(result.source_table.column_units),
                "width_mm": width_mm,
                "height_mm": height_mm,
                "dpi": dpi,
                "rendered_population": (
                    "All eligible prepared points (no display thinning)"
                ),
                "source_rows": (
                    "Zero-based row indices in the connected measurement table"
                ),
                "presentation": {
                    "jitter": "Uniform horizontal display jitter for Compare groups",
                    "jitter_seed": PLOT_JITTER_SEED,
                    "jitter_seed_policy": "Seed plus zero-based group index",
                    "jitter_range": [-0.21, 0.21],
                    "jitter_affects_measurements": False,
                },
            }
            with staged[2].open("w", encoding="utf-8") as stream:
                json.dump(
                    settings, stream, ensure_ascii=False, indent=2, allow_nan=False
                )
                stream.flush()
                os.fsync(stream.fileno())
        _check(cancellation)
        if is_current is not None and not is_current():
            raise ValueError(
                "This plot changed while exporting. Calculate and export again."
            )
        _publish(staged, targets, directory, expected)
    except _RecoveryFailed:
        keep_recovery = True
        raise
    finally:
        if figure is not None:
            figure.clear()
        if not keep_recovery:
            shutil.rmtree(directory)
    return PlotExportResult(
        targets,
        (
            "All eligible measurements are rendered; interactive display thinning "
            "is not exported.",
        ),
    )


def save_plot_output(
    path, result, *, file_format=None, metadata=None, cancellation=None
):
    """Batch/Python single-figure writer with explicit fixed physical size.

    Batch's staged destination ownership is authoritative. Metadata remains in
    the batch record, not silently injected into graphics-format private fields.
    """
    del metadata
    target = Path(path)
    chosen = str(file_format or target.suffix.lstrip(".") or "png").lower()
    if chosen == "auto":
        chosen = target.suffix.lstrip(".").lower() or "png"
    if chosen not in {"png", "tif", "tiff", "svg", "pdf"}:
        raise ValueError("Plot output requires PNG, TIFF, SVG or PDF.")
    if not target.suffix:
        target = target.with_suffix(f".{chosen}")
    suffix_format = target.suffix.lower().lstrip(".")
    if suffix_format != chosen and {suffix_format, chosen} != {"tif", "tiff"}:
        raise ValueError("The plot file extension must match its selected format.")
    exported = export_plot_result(
        result,
        target,
        width_mm=140,
        height_mm=100,
        dpi=300,
        overwrite=True,
        cancellation=cancellation,
    )
    return exported.paths[0]
