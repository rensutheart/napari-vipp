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
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.ticker import (
    Formatter,
    Locator,
    LogLocator,
    MaxNLocator,
    NullLocator,
    ScalarFormatter,
    StrMethodFormatter,
)

from napari_vipp.core.measurement_collection import _check, _ordinary_path
from napari_vipp.core.measurement_export import (
    _publish,
    _RecoveryFailed,
    measurement_export_destination_revisions,
)
from napari_vipp.core.result_plots import (
    interval_ticks,
    parse_tick_interval,
    plot_count_axes,
    validate_plot_intervals,
)

PLOT_COLORS = ("#278AC7", "#D88027", "#289D8F", "#AA6BC4", "#CB536C", "#80733D")
PLOT_MARKERS = ("o", "D", "s", "^", "v", "P")
PLOT_JITTER_SEED = 1729


class _CountLogLocator(LogLocator):
    """Logarithmic count ticks cannot describe fractions of an observation."""

    def tick_values(self, vmin, vmax):
        ticks = super().tick_values(vmin, vmax)
        low, high = sorted((vmin, vmax))
        # A narrow count range (for example 1–2) may contain only one decade
        # tick. Add useful integer ticks without changing the logarithmic scale.
        if np.count_nonzero((ticks >= max(low, 1)) & (ticks <= high)) <= 1:
            ticks = MaxNLocator(nbins=4, integer=True, min_n_ticks=1).tick_values(
                max(low, 1), max(high, 1)
            )
        return ticks[(ticks >= 1) & (ticks == np.floor(ticks))]


def _set_count_axis(axis, *, logarithmic, compact):
    """Presentation only: use exact count ticks, never round scientific data."""
    axis.set_major_locator(
        _CountLogLocator()
        if logarithmic
        else MaxNLocator(nbins=4 if compact else 6, integer=True, min_n_ticks=1)
    )
    axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    # Default log minor locators can otherwise introduce fractional count ticks.
    axis.set_minor_locator(NullLocator())


class _FixedIntervalLocator(Locator):
    """A bounded numeric major interval, shared by labels and grid lines."""

    def __init__(self, interval, axis_name):
        self.interval = interval
        self.axis_name = axis_name

    def tick_values(self, vmin, vmax):
        return np.asarray(
            interval_ticks(vmin, vmax, self.interval, axis=self.axis_name), dtype=float
        )

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())


def _group_tick_labels(result):
    """Short display labels only; never round group keys or plotted values."""
    names = [series.name for series in result.series]
    numeric = {}
    if result.recipe.group_column:
        column = result.source_table.columns.index(result.recipe.group_column)
        for index, series in enumerate(result.series):
            value = result.source_table.rows[series.source_rows[0][0]][column]
            if isinstance(value, Real) and not isinstance(
                value, (Integral, bool, np.bool_)
            ):
                numeric[index] = value
    # Increase precision when needed so nearby numeric groups remain distinct.
    labels = list(names)
    for precision in (4, 6, 8, 10, 12, 15, 17):
        labels = [
            format(numeric[index], f".{precision}g") if index in numeric else name
            for index, name in enumerate(names)
        ]
        if len(set(labels)) == len(set(names)):
            break
    duplicates = Counter(labels)
    labels = [
        name if duplicates[label] > 1 else label
        for name, label in zip(names, labels, strict=True)
    ]
    wrapped = [
        textwrap.fill(
            label, width=20, break_on_hyphens=False, max_lines=3, placeholder="…"
        )
        if index not in numeric
        else label
        for index, label in enumerate(labels)
    ]
    duplicates = Counter(wrapped)
    # Long common-prefix identifiers must still have distinct visible labels.
    wrapped = [
        f"{label}\n[group {index + 1}]" if duplicates[label] > 1 else label
        for index, label in enumerate(wrapped)
    ]
    return wrapped, any(a != b for a, b in zip(wrapped, names, strict=True))


class _GroupLabelFormatter(Formatter):
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, value, pos=None):
        index = round(value)
        return self.labels[index] if 0 <= index < len(self.labels) else ""


class _GroupLabelLocator(Locator):
    """Fit categorical text at draw time, including resize and vector export.

    Measure in typographic points so DPI alone never changes the label budget.
    Only tick labels are thinned. Data, summaries and category positions are
    untouched, and the figure states when some labels are not shown.
    """

    def __init__(self, labels, note, *, font_size, shortened, compact):
        self.labels = labels
        self.note = note
        self.font_size = font_size
        self.shortened = shortened
        self.compact = compact
        metrics = TextToPath()
        font = FontProperties(size=font_size)
        self.sizes = [
            (
                max(
                    metrics.get_text_width_height_descent(line, font, False)[0]
                    for line in (label.splitlines() or [""])
                ),
                max(1, len(label.splitlines())) * font_size * 1.3,
            )
            for label in labels
        ]

    def __call__(self):
        if not self.labels:
            self.note.set_text("")
            return np.array([])
        low, high = sorted(self.axis.get_view_interval())
        indices = np.arange(
            max(0, math.ceil(low)), min(len(self.labels), math.floor(high) + 1)
        )
        if not len(indices):
            return indices
        figure = self.axis.axes.figure
        width = self.axis.axes.bbox.width * 72 / figure.dpi
        spacing = width / max(high - low, 1)
        max_height = min(
            85 if not self.compact else 58, figure.get_figheight() * 72 * 0.3
        )
        choices = []
        for angle in (0, 45, 90):
            radians = math.radians(angle)
            w = max(
                self.sizes[i][0] * math.cos(radians)
                + self.sizes[i][1] * math.sin(radians)
                for i in indices
            )
            h = max(
                self.sizes[i][0] * math.sin(radians)
                + self.sizes[i][1] * math.cos(radians)
                for i in indices
            )
            if angle and h > max_height:
                continue
            # Extra space covers font hinting and minor constrained-layout shifts.
            stride = max(1, math.ceil((w + self.font_size * 0.9) / max(spacing, 1)))
            choices.append((stride, angle))
            if stride == 1:
                break
        stride, angle = min(choices)
        # Evenly space the labelled categories; retain both endpoints when possible.
        count = max(1, (len(indices) - 1) // stride + 1)
        chosen = (
            indices[np.linspace(0, len(indices) - 1, count, dtype=int)]
            if count > 1
            else indices[[len(indices) // 2]]
        )
        self.axis.set_tick_params(labelrotation=angle)
        for tick in self.axis.get_major_ticks(len(chosen)):
            tick.label1.set(
                horizontalalignment="center",
                verticalalignment="top",
                rotation_mode="default",
                parse_math=False,
            )
        details = []
        if len(chosen) < len(self.labels):
            details.append(f"{len(chosen)} group labels shown")
        if self.shortened:
            details.append("labels shortened")
        message = (
            f"All {len(self.labels)} groups plotted; {', '.join(details)}.\n"
            "Full group values are in plotted data."
            if details
            else ""
        )
        # The footnote participates in constrained layout, not an overlay.
        wrap_width = max(14, int(figure.get_figwidth() * 72 / (self.font_size * 0.6)))
        self.note.set_text(
            "\n".join(
                textwrap.fill(line, width=wrap_width) for line in message.splitlines()
            )
        )
        return chosen


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
    validate_plot_intervals(result)
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
        axes.grid(
            axis="y" if recipe.plot_type == "Compare groups" else "both",
            color=palette["grid"],
            linewidth=0.6,
            alpha=0.6,
        )

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
        labels, shortened = _group_tick_labels(result)
        note = figure.supxlabel("", fontsize=font_size - 1, color=palette["text"])
        note.set_parse_math(False)
        axes.xaxis.set_major_locator(
            _GroupLabelLocator(
                labels, note, font_size=font_size, shortened=shortened, compact=compact
            )
        )
        axes.xaxis.set_major_formatter(_GroupLabelFormatter(labels))
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
    for name, axis, is_count in zip(
        ("x", "y"), (axes.xaxis, axes.yaxis), plot_count_axes(result), strict=True
    ):
        if is_count:
            _set_count_axis(
                axis, logarithmic=getattr(recipe, f"log_{name}"), compact=compact
            )
        interval = parse_tick_interval(
            getattr(recipe, f"{name}_tick_interval"), axis=name
        )
        if interval is not None:
            locator = _FixedIntervalLocator(interval, name)
            # Validate the auto-padded visible range before handing off to the
            # Qt draw callback. This also keeps exporters bounded before save.
            locator.tick_values(*axis.get_view_interval())
            axis.set_major_locator(locator)
            axis.set_minor_locator(NullLocator())
            if not is_count:
                axis.set_major_formatter(ScalarFormatter(useOffset=False))
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
    # Keep authored labels literal. Numeric ticks retain Matplotlib's math
    # rendering for log/scientific notation; group ticks are handled above.
    for text in (axes.title, axes.xaxis.label, axes.yaxis.label):
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
                    "group_labels": (
                        "Labels may be shortened, wrapped, rotated or spaced to fit. "
                        "All categorical groups retain their exact identity and order; "
                        "full group values are in the plotted-data CSV."
                    ),
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
