"""Graphics and atomic export contracts for prepared measurement plots."""

import csv
import json

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from napari_vipp.core.plot_rendering import (
    build_plot_figure,
    export_plot_result,
    plot_export_targets,
    save_plot_output,
)
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData


def _result(**params):
    table = TableData(
        ("label_id", "image", "condition", "area", "intensity"),
        (
            (101, "one", "Control", 12.0, 110.0),
            (102, "one", "Control", 18.0, 140.0),
            (103, "two", "Control", 21.0, 120.0),
            (201, "three", "Treated", 33.0, 200.0),
            (202, "three", "Treated", 45.0, 290.0),
            (203, "four", "Treated", 38.0, 300.0),
        ),
        column_units=(("area", "µm²"), ("intensity", "a.u.")),
    )
    return build_plot_result(
        table,
        **{
            "y_column": "area",
            "x_column": "intensity",
            "group_column": "condition",
            **params,
        },
    )


@pytest.mark.parametrize(
    "mode,distribution",
    [
        ("Compare groups", "Histogram"),
        ("Scatter", "Histogram"),
        ("Distribution", "Histogram"),
        ("Distribution", "Cumulative"),
    ],
)
def test_all_plot_modes_render_without_gui(mode, distribution):
    result = _result(plot_type=mode, distribution=distribution)
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    assert figure.axes[0].get_xlabel() == result.x_label
    assert figure.axes[0].get_ylabel() == result.y_label
    assert figure.axes[0].get_title()
    figure.clear()


def test_scatter_artist_preserves_object_identity():
    figure = build_plot_figure(_result(plot_type="Scatter"))
    artist = figure.axes[0].collections[0]
    assert artist.vipp_source_rows == ((0,), (1,), (2,))
    assert len(artist.get_offsets()) == 3


def test_group_summary_uses_prepared_mean():
    figure = build_plot_figure(_result(summary="Mean"))
    assert np.allclose(figure.axes[0].lines[0].get_ydata(), [17.0, 17.0])


def test_histogram_uses_step_outline_not_per_bin_bars():
    figure = build_plot_figure(_result(plot_type="Distribution", bins=20))
    assert len(figure.axes[0].patches) == 4  # Two outlines and two faint fills.
    assert not figure.axes[0].containers


@pytest.mark.parametrize("suffix", ["png", "tif", "tiff", "svg", "pdf"])
def test_export_formats_have_fixed_dimensions(tmp_path, suffix):
    target = tmp_path / f"measurements.{suffix}"
    exported = export_plot_result(
        _result(),
        target,
        width_mm=101.6,
        height_mm=76.2,
        dpi=100,
    )
    assert exported.paths == (target,)
    if suffix in {"png", "tif", "tiff"}:
        with Image.open(target) as image:
            assert image.size == (400, 300)
    elif suffix == "svg":
        text = target.read_text(encoding="utf-8")
        assert 'width="288pt"' in text and 'height="216pt"' in text
    else:
        assert target.read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.glob(".vipp-figure-*"))


def test_export_sidecars_include_exact_recipe_and_all_prepared_rows(tmp_path):
    result = _result()
    exported = export_plot_result(result, tmp_path / "figure.svg", include_data=True)
    assert len(exported.paths) == 3
    with exported.paths[1].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert tuple(rows[0]) == result.plotted_table.columns
    assert len(rows) == 7
    settings = json.loads(exported.paths[2].read_text(encoding="utf-8"))
    assert settings["recipe"] == result.recipe.to_params()
    assert settings["counts"]["plotted_points"] == 6
    assert settings["column_units"]["area"] == "µm²"


def test_no_clobber_and_protected_files(tmp_path):
    target = tmp_path / "image.png"
    target.write_bytes(b"original source")
    with pytest.raises(FileExistsError):
        export_plot_result(_result(), target)
    with pytest.raises(ValueError, match="source"):
        export_plot_result(_result(), target, protected_paths=(target,), overwrite=True)
    assert target.read_bytes() == b"original source"


def test_failed_render_preserves_existing_files_and_leaves_no_stage(
    tmp_path, monkeypatch
):
    target = tmp_path / "figure.png"
    target.write_bytes(b"original")
    monkeypatch.setattr(
        "matplotlib.figure.Figure.savefig",
        lambda *a, **k: (_ for _ in ()).throw(OSError("failed")),
    )
    with pytest.raises(OSError, match="failed"):
        export_plot_result(_result(), target, overwrite=True)
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob(".vipp-figure-*"))


def test_publication_failure_rolls_back_companions(tmp_path, monkeypatch):
    import napari_vipp.core.measurement_export as publication

    targets = plot_export_targets(tmp_path / "figure.svg", include_data=True)
    for index, target in enumerate(targets):
        target.write_bytes(f"original {index}".encode())
    original_replace = publication.os.replace

    def fail_settings(source, target):
        if str(source).endswith("plot-settings.json"):
            raise OSError("publication failed")
        return original_replace(source, target)

    monkeypatch.setattr(publication.os, "replace", fail_settings)
    with pytest.raises(OSError, match="publication failed"):
        export_plot_result(_result(), targets[0], include_data=True, overwrite=True)
    assert [p.read_bytes() for p in targets] == [
        f"original {i}".encode() for i in range(3)
    ]


@pytest.mark.parametrize(
    "dimensions",
    [
        {"width_mm": 0},
        {"height_mm": float("nan")},
        {"dpi": 20},
        {"width_mm": 500, "height_mm": 500, "dpi": 1200},
    ],
)
def test_invalid_or_excessive_export_dimensions_are_rejected(tmp_path, dimensions):
    with pytest.raises(ValueError):
        export_plot_result(_result(), tmp_path / "figure.png", **dimensions)
    assert not list(tmp_path.iterdir())


def test_batch_writer_extension_and_cancel(tmp_path):
    from napari_vipp.core.progress import OperationCancelled

    with pytest.raises(OperationCancelled):
        save_plot_output(
            tmp_path / "cancelled.svg", _result(), cancellation=lambda: True
        )
    with pytest.raises(ValueError, match="extension"):
        save_plot_output(tmp_path / "mismatch.svg", _result(), file_format="png")
    assert (
        save_plot_output(tmp_path / "batch", _result(), file_format="svg").suffix
        == ".svg"
    )


def test_plot_that_becomes_stale_during_render_is_not_published(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    current = [True]
    original = Figure.savefig

    def render_then_stale(*args, **kwargs):
        original(*args, **kwargs)
        current[0] = False

    monkeypatch.setattr(Figure, "savefig", render_then_stale)
    with pytest.raises(ValueError, match="changed while exporting"):
        export_plot_result(
            _result(),
            tmp_path / "stale.svg",
            is_current=lambda: current[0],
        )
    assert not list(tmp_path.iterdir())


def test_batch_tiff_format_accepts_tif_extension(tmp_path):
    target = tmp_path / "batch.tif"
    assert save_plot_output(target, _result(), file_format="tiff") == target
    with Image.open(target) as image:
        assert image.format == "TIFF"


@pytest.mark.parametrize("rows", [(), ((1.25, "One group"),)])
def test_empty_and_single_category_labels_render(rows):
    result = build_plot_result(
        TableData(("measurement", "condition"), rows),
        y_column="measurement",
        group_column="condition",
    )
    figure = build_plot_figure(result, size_inches=(3, 2.8), compact=True)
    FigureCanvasAgg(figure).draw()
    assert len(figure.axes[0].get_xticklabels()) == len(rows)
    assert not figure.texts[0].get_text()
