from __future__ import annotations

import numpy as np
import pytest
import tifffile
from qtpy.QtGui import QColor, QFont, QFontMetrics, QImage, QPalette

from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID
from napari_vipp.core.tables import HistogramResultMetadata, TableData
from napari_vipp.ui.histogram_dialog import HistogramDialog
from napari_vipp.ui.plots import DetailedHistogramPlot

HISTOGRAM_COLUMNS = (
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "bin_width",
    "count",
    "fraction",
    "density",
    "cumulative_count",
    "cumulative_fraction",
)

Y_VALUE_LABELS = (
    "Count",
    "Fraction",
    "Probability density",
    "Cumulative count",
    "Cumulative fraction",
)


def _histogram_table() -> TableData:
    edges = np.asarray([1.0, 10.0, 100.0, 1000.0])
    counts = np.asarray([3, 2, 1], dtype=np.int64)
    widths = np.diff(edges)
    fractions = counts / counts.sum()
    density = fractions / widths
    cumulative_count = np.cumsum(counts)
    cumulative_fraction = np.cumsum(fractions)
    rows = tuple(
        zip(
            np.arange(1, 4),
            edges[:-1],
            edges[1:],
            edges[:-1] + widths / 2.0,
            widths,
            counts,
            fractions,
            density,
            cumulative_count,
            cumulative_fraction,
            strict=True,
        )
    )
    return TableData(
        columns=HISTOGRAM_COLUMNS,
        rows=rows,
        name="Intensity histogram",
        table_kind="Intensity histogram bins",
        source_name="sample.tif",
        histogram_metadata=HistogramResultMetadata(
            input_value_count=10,
            finite_value_count=8,
            nan_value_count=1,
            positive_infinite_value_count=1,
            negative_infinite_value_count=0,
            binned_value_count=6,
            underflow_count=1,
            overflow_count=1,
            nonpositive_excluded_count=0,
            effective_minimum=1.0,
            effective_maximum=1000.0,
            bin_count=3,
            bin_spacing="Logarithmic",
        ),
    )


def _combo_items(combo) -> tuple[str, ...]:
    return tuple(combo.itemText(index) for index in range(combo.count()))


def _displayed_values(plot: DetailedHistogramPlot) -> np.ndarray:
    return np.asarray(plot.y_values).squeeze(axis=0)


def _dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#151922"))
    palette.setColor(QPalette.Base, QColor("#151922"))
    palette.setColor(QPalette.Text, QColor("#f8fafc"))
    palette.setColor(QPalette.WindowText, QColor("#f8fafc"))
    palette.setColor(QPalette.Highlight, QColor("#60a5fa"))
    return palette


class _RecordingPainter:
    """Minimal painter spy for verifying text-band geometry."""

    def __init__(self, font: QFont):
        self._font = QFont(font)
        self.text_calls: list[tuple[int, int, str, QFont]] = []
        self.fill_rects = []

    def font(self) -> QFont:
        return QFont(self._font)

    def setFont(self, font: QFont) -> None:  # noqa: N802
        self._font = QFont(font)

    def fontMetrics(self) -> QFontMetrics:  # noqa: N802
        return QFontMetrics(self._font)

    def setPen(self, _pen) -> None:  # noqa: N802
        return None

    def drawText(self, x: int, y: int, text: str) -> None:  # noqa: N802
        self.text_calls.append((int(x), int(y), str(text), QFont(self._font)))

    def fillRect(self, rect, _color) -> None:  # noqa: N802
        self.fill_rects.append(rect)

    def save(self) -> None:
        return None

    def restore(self) -> None:
        return None

    def translate(self, _x: int, _y: int) -> None:
        return None

    def rotate(self, _angle: int) -> None:
        return None


def _text_vertical_bounds(call: tuple[int, int, str, QFont]) -> tuple[int, int]:
    _x, baseline, _text, font = call
    metrics = QFontMetrics(font)
    return baseline - metrics.ascent(), baseline + metrics.descent()


def test_detailed_histogram_plot_preserves_raw_values_and_true_axis_modes(qtbot):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    edges = np.asarray([1.0, 10.0, 100.0, 1000.0])
    values = np.asarray([3.0, 2.0, 1.0])

    plot.set_histogram(
        edges,
        values,
        x_scale="log10",
        y_scale="log10",
        title="Intensity histogram",
        x_axis_label="Intensity (a.u.)",
        y_axis_label="Count",
    )

    np.testing.assert_array_equal(plot.bin_edges, edges)
    np.testing.assert_array_equal(_displayed_values(plot), values)
    assert plot.x_logarithmic
    assert plot.y_logarithmic


def test_detailed_histogram_plot_rejects_nonpositive_log_x_edges(qtbot):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)

    with pytest.raises(ValueError, match="(?i)positive"):
        plot.set_histogram(
            np.asarray([0.0, 1.0, 10.0]),
            np.asarray([2.0, 1.0]),
            x_scale="log10",
        )


def test_detailed_histogram_plot_maps_decades_to_equal_screen_intervals(qtbot):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(700, 420)
    plot.set_histogram(
        np.asarray([1.0, 10.0, 100.0, 1000.0]),
        np.asarray([1.0, 10.0, 100.0]),
        x_scale="log10",
        y_scale="log10",
    )
    rect = plot._plot_rect()

    x_pixels = np.asarray(
        [plot._x_pixel(value, rect) for value in (1.0, 10.0, 100.0, 1000.0)]
    )
    y_pixels = np.asarray(
        [plot._y_pixel(value, rect) for value in (1.0, 10.0, 100.0)]
    )

    np.testing.assert_allclose(np.diff(x_pixels), np.diff(x_pixels)[0], atol=1)
    np.testing.assert_allclose(np.diff(y_pixels), np.diff(y_pixels)[0], atol=1)


@pytest.mark.parametrize("size", [(320, 165), (440, 280)])
def test_detailed_histogram_layout_separates_axis_titles_and_large_ticks(
    qtbot,
    size,
):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(*size)
    plot.set_histogram(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([8.02e6, 6.01e6, 2.00e6]),
        title="Intensity histogram",
        x_axis_label="Input value (a.u.)",
        y_axis_label="Count",
    )

    outer = plot.rect().adjusted(8, 8, -8, -8)
    plot_rect = plot._plot_rect()
    tick_font = plot._tick_label_font()
    axis_font = plot._axis_label_font()
    tick_metrics = QFontMetrics(tick_font)
    axis_metrics = QFontMetrics(axis_font)
    widest_y_tick = max(
        tick_metrics.horizontalAdvance(f"{value:.2e}")
        for value in (2.00e6, 4.01e6, 6.02e6, 8.02e6)
    )
    leftmost_tick = plot_rect.left() - widest_y_tick - 6
    y_title_right = (
        outer.left() + axis_metrics.ascent() + axis_metrics.descent() + 2
    )
    x_tick_bottom = (
        plot_rect.bottom()
        + tick_metrics.ascent()
        + 5
        + tick_metrics.descent()
    )
    x_title_top = (
        outer.bottom()
        - axis_metrics.descent()
        - 8
        - axis_metrics.ascent()
    )

    assert leftmost_tick >= y_title_right + 6
    assert x_tick_bottom + 4 <= x_title_top
    assert axis_font.pointSizeF() > tick_font.pointSizeF()
    assert axis_font.weight() == QFont.DemiBold


@pytest.mark.parametrize(
    "size",
    [(320, 165), (1100, 520)],
    ids=("inspector", "popout"),
)
def test_multiseries_histogram_reserves_nonoverlapping_title_and_legend_bands(
    qtbot,
    size,
):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(*size)
    plot.set_histogram(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray(
            [
                [8.0, 5.0, 2.0],
                [7.0, 4.0, 1.0],
            ]
        ),
        title="Intensity histogram",
        x_axis_label="Input value (a.u.)",
        y_axis_label="Count",
        series_labels=("Channel 1", "Channel 2"),
        colors=("#14b8a6", "#d946ef"),
    )

    outer = plot.rect().adjusted(8, 8, -8, -8)
    plot_rect = plot._plot_rect()
    painter = _RecordingPainter(plot.font())
    plot._draw_title_and_labels(painter, outer, plot_rect)
    plot._draw_legend(painter, outer, plot_rect)

    title_call = next(
        call for call in painter.text_calls if call[2] == "Intensity histogram"
    )
    legend_calls = [
        call
        for call in painter.text_calls
        if call[2] in {"Channel 1", "Channel 2"}
    ]
    assert len(legend_calls) == 2
    assert len(painter.fill_rects) == 2

    _title_top, title_bottom = _text_vertical_bounds(title_call)
    legend_text_bounds = [_text_vertical_bounds(call) for call in legend_calls]
    legend_top = min(
        *(top for top, _bottom in legend_text_bounds),
        *(rect.top() for rect in painter.fill_rects),
    )
    legend_bottom = max(
        *(bottom for _top, bottom in legend_text_bounds),
        *(rect.bottom() for rect in painter.fill_rects),
    )

    assert title_bottom + 4 <= legend_top
    assert legend_bottom + 4 <= plot_rect.top()


@pytest.mark.parametrize(
    "size",
    [(320, 165), (1100, 520)],
    ids=("inspector", "popout"),
)
def test_histogram_x_axis_title_keeps_bottom_frame_padding(qtbot, size):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(*size)
    plot.set_histogram(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([8.0, 5.0, 2.0]),
        title="Intensity histogram",
        x_axis_label="Input value (a.u.)",
        y_axis_label="Count",
    )

    outer = plot.rect().adjusted(8, 8, -8, -8)
    plot_rect = plot._plot_rect()
    painter = _RecordingPainter(plot.font())
    plot._draw_title_and_labels(painter, outer, plot_rect)
    x_axis_call = next(
        call for call in painter.text_calls if call[2] == "Input value (a.u.)"
    )
    _label_top, label_bottom = _text_vertical_bounds(x_axis_call)

    assert label_bottom + 6 <= outer.bottom()


@pytest.mark.parametrize(
    ("edges", "values", "message"),
    [
        (
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 2.0]),
            "one value per interval",
        ),
        (np.asarray([0.0, 1.0, 1.0]), np.asarray([1.0, 2.0]), "increasing"),
        (np.asarray([0.0, np.inf]), np.asarray([1.0]), "finite"),
        (np.asarray([0.0, 1.0]), np.asarray([-1.0]), "non-negative"),
    ],
)
def test_detailed_histogram_plot_validates_scientific_arrays(
    qtbot,
    edges,
    values,
    message,
):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)

    with pytest.raises(ValueError, match=rf"(?i){message}"):
        plot.set_histogram(edges, values)


def test_histogram_dialog_ingests_table_and_discloses_population_accounting(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)

    dialog.set_histogram(
        _histogram_table(),
        title="Sample intensity histogram",
        x_axis_label="Intensity (a.u.)",
    )

    assert not dialog.isModal()
    assert dialog.windowTitle().startswith("Sample intensity histogram")
    assert "Histogram" in dialog.windowTitle()
    assert _combo_items(dialog.y_values_combo) == Y_VALUE_LABELS
    assert dialog.y_values_combo.currentText() == "Count"
    np.testing.assert_array_equal(dialog.plot.bin_edges, [1.0, 10.0, 100.0, 1000.0])
    np.testing.assert_array_equal(_displayed_values(dialog.plot), [3.0, 2.0, 1.0])
    summary = dialog.summary_label.text().casefold()
    for expected in (
        "10",
        "8",
        "6",
        "nan",
        "1",
        "underflow",
        "overflow",
        "3 bins",
        "logarithmic",
    ):
        assert expected in summary


def test_histogram_dialog_calculation_controls_follow_operation_spec(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    specs = NODE_LIBRARY_BY_ID["intensity_histogram"].parameters
    expected = {spec.name: spec.default for spec in specs}

    assert tuple(dialog.calculation_controls) == tuple(expected)
    assert dialog.calculation_parameters == expected

    snapshot = dialog.calculation_parameters
    snapshot["bin_count"] = 2
    assert dialog.calculation_parameters == expected

    for spec in specs:
        control = dialog.calculation_controls[spec.name]
        assert control.spec == spec
        if spec.kind == "choice":
            assert _combo_items(control.combo) == spec.choices
            continue
        value_box = control.value_box
        assert value_box.minimum() == pytest.approx(spec.minimum)
        assert value_box.maximum() == pytest.approx(spec.maximum)
        assert value_box.singleStep() == pytest.approx(spec.step)
        if spec.kind == "float":
            assert value_box.decimals() == spec.decimals


def test_histogram_dialog_programmatic_calculation_sync_is_quiet(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    emitted = []
    dialog.calculationParametersChanged.connect(emitted.append)
    parameters = {
        "bin_count": 1_024,
        "range_mode": "Custom range",
        "custom_min": 1.25,
        "custom_max": 9_000.5,
        "bin_spacing": "Logarithmic",
    }

    dialog.set_calculation_parameters(parameters)

    assert emitted == []
    assert dialog.calculation_parameters == parameters
    assert {
        name: control.value()
        for name, control in dialog.calculation_controls.items()
    } == parameters


def test_histogram_dialog_custom_range_rows_retain_hidden_values(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    controls = dialog.calculation_controls
    custom_controls = (controls["custom_min"], controls["custom_max"])

    for control in custom_controls:
        assert not control.isVisible()
        assert not dialog.calculation_form.labelForField(control).isVisible()

    controls["range_mode"].combo.setCurrentText("Custom range")

    for control in custom_controls:
        assert control.isVisible()
        assert dialog.calculation_form.labelForField(control).isVisible()

    controls["custom_max"].value_box.setValue(90.25)
    controls["custom_min"].value_box.setValue(10.5)
    controls["range_mode"].combo.setCurrentText("Data range")

    assert dialog.calculation_parameters["custom_min"] == pytest.approx(10.5)
    assert dialog.calculation_parameters["custom_max"] == pytest.approx(90.25)
    for control in custom_controls:
        assert not control.isVisible()
        assert not dialog.calculation_form.labelForField(control).isVisible()

    controls["range_mode"].combo.setCurrentText("Custom range")

    assert controls["custom_min"].value() == pytest.approx(10.5)
    assert controls["custom_max"].value() == pytest.approx(90.25)
    for control in custom_controls:
        assert control.isVisible()
        assert dialog.calculation_form.labelForField(control).isVisible()


def test_histogram_dialog_user_edit_emits_one_complete_calculation_mapping(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    emitted = []
    dialog.calculationParametersChanged.connect(
        lambda parameters: emitted.append(dict(parameters))
    )

    dialog.calculation_controls["bin_count"].value_box.setValue(512)

    assert emitted == [
        {
            "bin_count": 512,
            "range_mode": "Data range",
            "custom_min": 0.0,
            "custom_max": 1.0,
            "bin_spacing": "Linear",
        }
    ]
    assert emitted[0] == dialog.calculation_parameters


@pytest.mark.parametrize(
    ("label", "column"),
    [
        ("Count", "count"),
        ("Fraction", "fraction"),
        ("Probability density", "density"),
        ("Cumulative count", "cumulative_count"),
        ("Cumulative fraction", "cumulative_fraction"),
    ],
)
def test_histogram_dialog_switches_y_measure_without_recalculating_table(
    qtbot,
    label,
    column,
):
    table = _histogram_table()
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(table)

    dialog.y_values_combo.setCurrentText(label)

    index = table.columns.index(column)
    expected = np.asarray([row[index] for row in table.rows], dtype=np.float64)
    np.testing.assert_allclose(_displayed_values(dialog.plot), expected)
    assert dialog.y_values_combo.currentText() == label


def test_histogram_dialog_axis_log_controls_update_the_same_plot(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(_histogram_table())
    plot_identity = id(dialog.plot)

    dialog.log_x_checkbox.setChecked(True)
    dialog.log_y_checkbox.setChecked(True)

    assert id(dialog.plot) == plot_identity
    assert dialog.plot.x_logarithmic
    assert dialog.plot.y_logarithmic
    np.testing.assert_array_equal(dialog.plot.bin_edges, [1.0, 10.0, 100.0, 1000.0])
    np.testing.assert_array_equal(_displayed_values(dialog.plot), [3.0, 2.0, 1.0])


def test_histogram_dialog_grid_divisions_are_exact_and_presentation_only(qtbot):
    table = _histogram_table()
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(table)
    emitted = []
    dialog.presentationChanged.connect(lambda *args: emitted.append(args))
    original_edges = dialog.plot.bin_edges.copy()
    original_values = dialog.plot.values.copy()

    assert dialog.x_grid_divisions_spinbox.minimum() == 2
    assert dialog.x_grid_divisions_spinbox.maximum() == 12
    assert dialog.x_grid_divisions_spinbox.value() == 4
    assert dialog.y_grid_divisions_spinbox.value() == 4
    assert "endpoints" in dialog.x_grid_divisions_spinbox.toolTip().casefold()

    dialog.x_grid_divisions_spinbox.setValue(6)
    dialog.y_grid_divisions_spinbox.setValue(3)

    assert dialog.plot.x_grid_divisions == 6
    assert dialog.plot.y_grid_divisions == 3
    assert len(dialog.plot._x_ticks()) == 7
    assert len(dialog.plot._y_ticks()) == 4
    assert dialog.table is table
    np.testing.assert_array_equal(dialog.plot.bin_edges, original_edges)
    np.testing.assert_array_equal(dialog.plot.values, original_values)
    assert emitted == []


@pytest.mark.parametrize("width", (520, 1_400))
def test_histogram_dialog_grid_division_controls_stay_grouped(qtbot, width):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.resize(width, 720)
    dialog.show()
    qtbot.waitExposed(dialog)

    ordered = (
        dialog.grid_divisions_label,
        dialog.x_grid_divisions_label,
        dialog.x_grid_divisions_spinbox,
        dialog.y_grid_divisions_label,
        dialog.y_grid_divisions_spinbox,
    )
    gaps = tuple(
        right.geometry().left() - left.geometry().right() - 1
        for left, right in zip(ordered, ordered[1:], strict=False)
    )

    assert all(0 <= gap <= 12 for gap in gaps)
    if width >= 1_000:
        assert dialog.y_grid_divisions_spinbox.geometry().right() < dialog.width() // 2
    else:
        assert (
            dialog.y_grid_divisions_spinbox.geometry().right()
            <= dialog.rect().right() - dialog.layout().contentsMargins().right()
        )


def test_detailed_histogram_grid_divisions_are_equal_in_log_space(qtbot):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.resize(700, 420)
    plot.set_histogram(
        np.asarray([2.0, 20.0, 200.0]),
        np.asarray([1.0, 100.0]),
        x_scale="log10",
        y_scale="log10",
    )
    plot.set_grid_divisions(x=2, y=2)
    plot_rect = plot._plot_rect()

    np.testing.assert_allclose(plot._x_ticks(), [2.0, 20.0, 200.0])
    np.testing.assert_allclose(plot._y_ticks(), [1.0, 10.0, 100.0])
    x_pixels = np.asarray(
        [plot._x_pixel(value, plot_rect) for value in plot._x_ticks()]
    )
    y_pixels = np.asarray(
        [plot._y_pixel(value, plot_rect) for value in plot._y_ticks()]
    )
    np.testing.assert_allclose(np.diff(x_pixels), np.diff(x_pixels)[0], atol=1)
    np.testing.assert_allclose(np.diff(y_pixels), np.diff(y_pixels)[0], atol=1)


def test_histogram_grid_divisions_persist_across_axis_and_result_refresh(qtbot):
    table = _histogram_table()
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(table)
    dialog.x_grid_divisions_spinbox.setValue(2)
    dialog.y_grid_divisions_spinbox.setValue(7)

    dialog.log_x_checkbox.setChecked(True)
    dialog.log_y_checkbox.setChecked(True)
    dialog.y_values_combo.setCurrentText("Fraction")
    dialog.set_histogram(table)

    assert dialog.plot.x_grid_divisions == 2
    assert dialog.plot.y_grid_divisions == 7
    assert len(dialog.plot._x_ticks()) == 3
    assert len(dialog.plot._y_ticks()) == 8


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("value", [1, 13, 3.5, True])
def test_detailed_histogram_plot_rejects_invalid_grid_divisions(
    qtbot,
    axis,
    value,
):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)

    with pytest.raises(ValueError, match=rf"(?i){axis} grid divisions"):
        plot.set_grid_divisions(**{axis: value})


def test_histogram_dialog_disables_log_x_for_nonpositive_edges(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(
        bin_edges=np.asarray([-1.0, 0.0, 1.0]),
        count=np.asarray([2.0, 3.0]),
        fraction=np.asarray([0.4, 0.6]),
        density=np.asarray([0.4, 0.6]),
        cumulative_count=np.asarray([2.0, 5.0]),
        cumulative_fraction=np.asarray([0.4, 1.0]),
    )

    assert not dialog.log_x_checkbox.isEnabled()
    assert not dialog.plot.x_logarithmic
    assert "greater than zero" in dialog.log_x_checkbox.toolTip().casefold()


def test_histogram_dialog_rejects_a_non_histogram_table(qtbot):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    malformed = TableData(("value",), ((1.0,),), name="Not a histogram")

    with pytest.raises(ValueError, match="(?i)histogram.*columns"):
        dialog.set_histogram(malformed)


@pytest.mark.parametrize("suffix", [".png", ".tif", ".tiff"])
def test_histogram_dialog_exports_current_plot_resolution(qtbot, tmp_path, suffix):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(_histogram_table())
    dialog.plot.resize(640, 420)
    expected_shape = (dialog.plot.height(), dialog.plot.width())
    target = tmp_path / f"histogram{suffix}"

    exported = dialog.export_image(target)

    assert exported == target
    assert target.exists()
    if suffix == ".png":
        saved = QImage(str(target))
        assert not saved.isNull()
        assert (saved.height(), saved.width()) == expected_shape
    else:
        saved = tifffile.imread(target)
        assert saved.shape == expected_shape + (3,)
        assert saved.dtype == np.uint8


def test_histogram_dialog_rejects_unknown_image_export_format(qtbot, tmp_path):
    dialog = HistogramDialog()
    qtbot.addWidget(dialog)
    dialog.set_histogram(_histogram_table())

    with pytest.raises(ValueError, match="PNG, TIF, and TIFF"):
        dialog.export_image(tmp_path / "histogram.jpg")


def test_detailed_histogram_plot_renders_under_a_dark_palette(qtbot):
    plot = DetailedHistogramPlot()
    qtbot.addWidget(plot)
    plot.setPalette(_dark_palette())
    plot.resize(500, 320)
    plot.set_histogram(
        np.asarray([1.0, 10.0, 100.0, 1000.0]),
        np.asarray([3.0, 2.0, 1.0]),
        x_scale="log10",
        y_scale="log10",
        title="Intensity histogram",
        x_axis_label="Intensity (a.u.)",
        y_axis_label="Count",
    )
    plot.show()
    qtbot.waitExposed(plot)

    pixmap = plot.grab()
    image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)

    assert not image.isNull()
    assert image.width() == round(plot.width() * pixmap.devicePixelRatio())
    assert image.height() == round(plot.height() * pixmap.devicePixelRatio())
    # The plot must not render as an unreadable single-colour rectangle under
    # the application's dark palette.
    colors = {
        image.pixelColor(x, y).name()
        for x, y in (
            (5, 5),
            (image.width() // 2, image.height() // 2),
            (image.width() // 4, 3 * image.height() // 4),
            (3 * image.width() // 4, 3 * image.height() // 4),
        )
    }
    assert len(colors) >= 2
    assert plot.palette().color(QPalette.Text).lightness() > (
        plot.palette().color(QPalette.Base).lightness()
    )
