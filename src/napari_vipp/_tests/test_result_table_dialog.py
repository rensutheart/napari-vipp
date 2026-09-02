from __future__ import annotations

import csv
import threading
from decimal import Decimal

import numpy as np
import pytest
from qtpy.QtCore import QItemSelectionModel, QRect, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPalette
from qtpy.QtWidgets import QFileDialog, QStyle, QStyleOption

import napari_vipp.ui.result_table_dialog as result_table_dialog_module
from napari_vipp.core.tables import TableData
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.result_table_dialog import (
    ResultTableDialog,
    ResultTableModel,
    choose_table_export_target,
)


def _column_values(model: ResultTableModel, column: int = 0) -> list[object]:
    return [model.raw_value(row, column) for row in range(model.rowCount())]


def _palette(*, dark: bool) -> QPalette:
    palette = QPalette()
    surface = QColor("#161b22" if dark else "#f8fafc")
    alternate = QColor("#202833" if dark else "#eef2f7")
    text = QColor("#f8fafc" if dark else "#17202a")
    highlight = QColor("#2563eb")
    palette.setColor(QPalette.Window, surface)
    palette.setColor(QPalette.Base, surface)
    palette.setColor(QPalette.AlternateBase, alternate)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    return palette


def test_result_table_model_sorts_numbers_and_keeps_missing_values_last():
    rows = (
        (10.0,),
        (2.0,),
        (float("nan"),),
        (None,),
        (-3.0,),
        (float("inf"),),
        (float("-inf"),),
    )
    table = TableData(("value",), rows)
    model = ResultTableModel(table)

    model.sort(0, Qt.AscendingOrder)
    ascending = _column_values(model)
    assert ascending[:5] == [float("-inf"), -3.0, 2.0, 10.0, float("inf")]
    assert np.isnan(ascending[5])
    assert ascending[6] is None

    model.sort(0, Qt.DescendingOrder)
    descending = _column_values(model)
    assert descending[:5] == [float("inf"), 10.0, 2.0, -3.0, float("-inf")]
    assert np.isnan(descending[5])
    assert descending[6] is None
    assert table.rows == rows


def test_result_table_model_sorts_text_naturally_and_booleans_logically():
    table = TableData(
        ("sample", "accepted"),
        (
            ("sample10", True),
            ("Sample1", False),
            ("sample2", True),
            ("", False),
        ),
    )
    model = ResultTableModel(table)

    model.sort(0, Qt.AscendingOrder)
    assert _column_values(model) == ["Sample1", "sample2", "sample10", ""]
    model.sort(0, Qt.DescendingOrder)
    assert _column_values(model) == ["sample10", "sample2", "Sample1", ""]

    model.sort(1, Qt.AscendingOrder)
    assert _column_values(model, 1) == [False, False, True, True]


def test_result_table_model_sorts_heterogeneous_numbers_and_decimal_nan_safely():
    table = TableData(
        ("mixed",),
        (
            (np.timedelta64(1, "D"),),
            (2,),
            (Decimal("NaN"),),
        ),
    )
    model = ResultTableModel(table)

    model.sort(0, Qt.AscendingOrder)

    values = _column_values(model)
    assert values[:2] == [np.timedelta64(1, "D"), 2]
    assert values[2].is_nan()

    model.sort(0, Qt.DescendingOrder)
    values = _column_values(model)
    assert values[:2] == [2, np.timedelta64(1, "D")]
    assert values[2].is_nan()


def test_result_table_dialog_exposes_complete_table_and_header_sorting(qtbot):
    table = TableData(
        ("object_id", "area"),
        tuple((index, 300 - index) for index in range(250)),
        name="Objects",
        table_kind="object measurements",
        column_units=(("area", "µm²"),),
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        table,
        title="Measure Objects",
        default_export_name="measure_objects.csv",
        context_key=("measure", 0),
    )
    dialog.show()

    assert not dialog.isModal()
    assert dialog.model.rowCount() == 250
    assert dialog.model.columnCount() == 2
    assert "250 rows × 2 fields" in dialog.summary_label.text()
    assert dialog.model.headerData(1, Qt.Horizontal, Qt.DisplayRole) == "area\n(µm²)"
    assert dialog.context_key == ("measure", 0)

    header = dialog.table_view.horizontalHeader()
    first_index = dialog.model.index(0, 0)
    dialog.table_view.selectionModel().select(
        first_index,
        QItemSelectionModel.Select,
    )
    assert dialog.table_view.selectionModel().selectedIndexes()
    header.sectionClicked.emit(1)
    assert dialog.model.raw_value(0, 1) == 51
    assert dialog.table_view.selectionModel().selectedIndexes() == []
    assert header.isSortIndicatorShown()
    assert header.sortIndicatorSection() == 1
    assert header.sortIndicatorOrder() == Qt.AscendingOrder
    header.sectionClicked.emit(1)
    assert dialog.model.raw_value(0, 1) == 300
    assert header.sortIndicatorSection() == 1
    assert header.sortIndicatorOrder() == Qt.DescendingOrder

    dialog.resize(420, 300)
    assert dialog.width() >= 420
    assert dialog.table_view.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff


def test_result_table_dialog_presents_stale_action_and_theme(qtbot):
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        TableData(("label",), ((1,),)),
        title="Measurements",
        default_export_name="measurements.csv",
    )
    requests = []
    dialog.recalculationRequested.connect(lambda: requests.append(True))
    palette = _palette(dark=True)
    dialog.refresh_theme(palette)

    dialog.set_result_status(
        "This is a stale cached result.",
        action_text="Recalculate",
        action_enabled=True,
        attention_required=True,
    )

    warning = theme_colors(palette).warning
    assert not dialog.result_status_panel.isHidden()
    assert dialog.result_status_label.text() == "This is a stale cached result."
    assert dialog.result_action_button.text() == "Recalculate"
    assert dialog.result_action_button.isEnabled()
    assert warning.foreground.name() in dialog.result_action_button.styleSheet()
    assert dialog.export_button.isEnabled()
    assert "not current" in dialog.export_button.toolTip()
    dialog.result_action_button.click()
    assert requests == [True]

    dialog.set_result_status()

    assert dialog.result_status_panel.isHidden()
    assert dialog.result_action_button.styleSheet() == ""
    assert "CSV or TSV" in dialog.export_button.toolTip()


def test_result_table_dialog_export_preserves_original_order_after_sort(
    qtbot,
    tmp_path,
):
    table = TableData(
        ("label", "mean"),
        ((10, 1.5), (2, 3.5), (7, 2.5)),
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        table,
        title="Measurements",
        default_export_name="measurements.csv",
    )
    dialog.table_view.horizontalHeader().sectionClicked.emit(0)
    assert _column_values(dialog.model) == [2, 7, 10]

    exported = dialog.export_table(tmp_path / "measurements.tsv", format="tsv")

    with exported.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows == [
        ["label", "mean"],
        ["10", "1.5"],
        ["2", "3.5"],
        ["7", "2.5"],
    ]


def test_choose_table_export_target_respects_suffix_with_all_files(
    qtbot,
    tmp_path,
    monkeypatch,
):
    parent = ResultTableDialog()
    qtbot.addWidget(parent)
    requested = tmp_path / "measurements.tsv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(requested), "All files (*.*)"),
    )

    target = choose_table_export_target(
        parent,
        default_name="measurements.csv",
        caption="Save result table",
    )

    assert target == (requested, "tsv")


@pytest.mark.parametrize(
    ("requested_name", "selected_filter", "expected_name", "expected_format"),
    [
        ("measurements.csv", "TSV table (*.tsv)", "measurements.tsv", "tsv"),
        ("measurements.tsv", "CSV table (*.csv)", "measurements.csv", "csv"),
    ],
)
def test_choose_table_export_target_canonicalizes_filter_and_extension(
    qtbot,
    tmp_path,
    monkeypatch,
    requested_name,
    selected_filter,
    expected_name,
    expected_format,
):
    parent = ResultTableDialog()
    qtbot.addWidget(parent)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            str(tmp_path / requested_name),
            selected_filter,
        ),
    )

    target = choose_table_export_target(
        parent,
        default_name="measurements.csv",
        caption="Save result table",
    )

    assert target == (tmp_path / expected_name, expected_format)


def test_large_table_sort_runs_in_background(qtbot, monkeypatch):
    monkeypatch.setattr(
        result_table_dialog_module,
        "_BACKGROUND_TASK_ROW_THRESHOLD",
        1,
    )
    table = TableData(("value",), ((3,), (1,), (2,)))
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        table,
        title="Large measurements",
        default_export_name="measurements.csv",
    )

    with qtbot.waitSignal(dialog.sortCompleted, timeout=5000):
        dialog.table_view.horizontalHeader().sectionClicked.emit(0)

    assert _column_values(dialog.model) == [1, 2, 3]
    assert dialog.table_view.horizontalHeader().sectionsClickable()
    assert dialog.sort_hint.text().startswith("Click a column heading")


def test_background_sort_shows_requested_chevron_before_rows_change(
    qtbot,
    monkeypatch,
):
    monkeypatch.setattr(
        result_table_dialog_module,
        "_BACKGROUND_TASK_ROW_THRESHOLD",
        1,
    )
    entered = threading.Event()
    release = threading.Event()
    original_sort = result_table_dialog_module._sorted_source_rows

    def blocking_sort(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return original_sort(*args, **kwargs)

    monkeypatch.setattr(
        result_table_dialog_module,
        "_sorted_source_rows",
        blocking_sort,
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        TableData(("label", "value"), ((1, 30), (2, 10), (3, 20))),
        title="Large measurements",
        default_export_name="measurements.csv",
    )
    header = dialog.table_view.horizontalHeader()

    try:
        header.sectionClicked.emit(1)
        assert entered.wait(timeout=2)
        assert header.isSortIndicatorShown()
        assert header.sortIndicatorSection() == 1
        assert header.sortIndicatorOrder() == Qt.AscendingOrder
        assert not header.sectionsClickable()
        assert _column_values(dialog.model, 1) == [30, 10, 20]
    finally:
        release.set()

    qtbot.waitUntil(lambda: dialog._active_sort_worker is None, timeout=5000)
    assert _column_values(dialog.model, 1) == [10, 20, 30]
    assert header.sortIndicatorSection() == 1
    assert header.sortIndicatorOrder() == Qt.AscendingOrder


def test_cancelled_background_sort_restores_committed_chevron(qtbot, monkeypatch):
    table = TableData(
        ("label", "value"),
        ((3, 30), (1, 10), (2, 20)),
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        table,
        title="Measurements",
        default_export_name="measurements.csv",
    )
    header = dialog.table_view.horizontalHeader()
    header.sectionClicked.emit(0)
    assert header.sortIndicatorSection() == 0
    assert header.sortIndicatorOrder() == Qt.AscendingOrder

    monkeypatch.setattr(
        result_table_dialog_module,
        "_BACKGROUND_TASK_ROW_THRESHOLD",
        1,
    )
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    original_sort = result_table_dialog_module._sorted_source_rows

    def blocking_sort(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        try:
            return original_sort(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(
        result_table_dialog_module,
        "_sorted_source_rows",
        blocking_sort,
    )

    try:
        header.sectionClicked.emit(1)
        assert entered.wait(timeout=2)
        assert header.sortIndicatorSection() == 1
        dialog._cancel_active_sort()
        assert header.isSortIndicatorShown()
        assert header.sortIndicatorSection() == 0
        assert header.sortIndicatorOrder() == Qt.AscendingOrder
    finally:
        release.set()

    qtbot.waitUntil(exited.is_set, timeout=5000)


def test_failed_background_sort_restores_committed_chevron(qtbot, monkeypatch):
    table = TableData(
        ("label", "value"),
        ((3, 30), (1, 10), (2, 20)),
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        table,
        title="Measurements",
        default_export_name="measurements.csv",
    )
    header = dialog.table_view.horizontalHeader()
    header.sectionClicked.emit(0)

    monkeypatch.setattr(
        result_table_dialog_module,
        "_BACKGROUND_TASK_ROW_THRESHOLD",
        1,
    )
    entered = threading.Event()
    release = threading.Event()

    def failing_sort(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("sort unavailable")

    monkeypatch.setattr(
        result_table_dialog_module,
        "_sorted_source_rows",
        failing_sort,
    )

    try:
        header.sectionClicked.emit(1)
        assert entered.wait(timeout=2)
        assert header.sortIndicatorSection() == 1
    finally:
        release.set()

    qtbot.waitUntil(lambda: dialog._active_sort_worker is None, timeout=5000)
    assert header.isSortIndicatorShown()
    assert header.sortIndicatorSection() == 0
    assert header.sortIndicatorOrder() == Qt.AscendingOrder
    assert dialog.sort_hint.text() == "Sort failed: sort unavailable"


def test_large_table_export_runs_in_background_and_preserves_order(
    qtbot,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        result_table_dialog_module,
        "_BACKGROUND_TASK_ROW_THRESHOLD",
        1,
    )
    target = tmp_path / "measurements.tsv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "TSV table (*.tsv)"),
    )
    table = TableData(("value",), ((3,), (1,), (2,)))
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        table,
        title="Large measurements",
        default_export_name="measurements.csv",
    )
    dialog.model.sort(0, Qt.AscendingOrder)

    with qtbot.waitSignal(dialog.exportCompleted, timeout=5000):
        dialog.request_export()

    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows == [["value"], ["3"], ["1"], ["2"]]
    assert dialog.export_button.isEnabled()
    assert dialog.export_button.text() == "Export CSV/TSV…"


def test_stale_background_sort_cannot_replace_a_new_table(qtbot, monkeypatch):
    monkeypatch.setattr(
        result_table_dialog_module,
        "_BACKGROUND_TASK_ROW_THRESHOLD",
        1,
    )
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    original_sort = result_table_dialog_module._sorted_source_rows

    def blocking_sort(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        try:
            kwargs.pop("cancel_event", None)
            return original_sort(*args, **kwargs)
        finally:
            exited.set()

    monkeypatch.setattr(
        result_table_dialog_module,
        "_sorted_source_rows",
        blocking_sort,
    )
    first = TableData(("value",), ((3,), (1,), (2,)))
    second = TableData(("value",), ((20,), (10,)))
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        first,
        title="First",
        default_export_name="first.csv",
        context_key=("first", 0),
    )
    completions = []
    dialog.sortCompleted.connect(lambda *args: completions.append(args))

    dialog.table_view.horizontalHeader().sectionClicked.emit(0)
    assert entered.wait(timeout=2)
    dialog.set_table(
        second,
        title="Second",
        default_export_name="second.csv",
        context_key=("second", 0),
    )
    release.set()
    qtbot.waitUntil(exited.is_set, timeout=5000)
    qtbot.wait(50)

    assert dialog.table is second
    assert _column_values(dialog.model) == [20, 10]
    assert not dialog.table_view.horizontalHeader().isSortIndicatorShown()
    assert completions == []


def test_background_export_keeps_its_original_table_snapshot(
    qtbot,
    tmp_path,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    original_save = result_table_dialog_module.save_table_output
    exported_tables = []

    def blocking_save(table, *args, **kwargs):
        exported_tables.append(table)
        entered.set()
        release.wait(timeout=5)
        return original_save(table, *args, **kwargs)

    monkeypatch.setattr(
        result_table_dialog_module,
        "save_table_output",
        blocking_save,
    )
    target = tmp_path / "first.tsv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "TSV table (*.tsv)"),
    )
    first = TableData(("value",), ((3,), (1,), (2,)))
    second = TableData(("value",), ((20,), (10,)))
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        first,
        title="First",
        default_export_name="first.csv",
    )

    with qtbot.waitSignal(dialog.exportCompleted, timeout=5000):
        dialog.request_export()
        assert entered.wait(timeout=2)
        dialog.set_table(
            second,
            title="Second",
            default_export_name="second.csv",
        )
        release.set()

    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows == [["value"], ["3"], ["1"], ["2"]]
    assert exported_tables == [first]
    assert dialog.table is second
    assert dialog.export_button.isEnabled()


def test_background_export_failure_restores_controls_on_gui_thread(
    qtbot,
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "broken.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(target), "CSV table (*.csv)"),
    )
    monkeypatch.setattr(
        result_table_dialog_module,
        "save_table_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    errors = []
    gui_thread = threading.current_thread()
    monkeypatch.setattr(
        result_table_dialog_module.QMessageBox,
        "critical",
        lambda *args, **kwargs: errors.append(
            (args[2], threading.current_thread())
        ),
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(
        TableData(("value",), ((1,),)),
        title="Measurements",
        default_export_name="measurements.csv",
    )

    dialog.request_export()
    qtbot.waitUntil(lambda: bool(errors), timeout=5000)

    assert errors == [("disk unavailable", gui_thread)]
    assert dialog.export_button.isEnabled()
    assert dialog.export_button.text() == "Export CSV/TSV…"
    assert "exact row order" in dialog.export_note.text()


def test_result_table_dialog_refreshes_dark_and_light_theme(qtbot):
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)

    for dark in (True, False):
        palette = _palette(dark=dark)
        dialog.refresh_theme(palette)
        colors = theme_colors(palette)
        style = dialog.styleSheet()
        assert colors.surface.name() in style
        assert colors.alternate_surface.name() in style
        assert colors.text.name() in style

        header = dialog.table_view.horizontalHeader()
        header.setSortIndicator(0, Qt.AscendingOrder)
        header.setSortIndicatorShown(True)
        image = QImage(18, 18, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        option = QStyleOption()
        option.rect = QRect(0, 0, image.width(), image.height())
        option.palette = palette
        painter = QPainter(image)
        dialog._sort_chevron_style.drawPrimitive(
            QStyle.PE_IndicatorHeaderArrow,
            option,
            painter,
            header,
        )
        painter.end()
        rendered = [
            image.pixelColor(x, y)
            for y in range(image.height())
            for x in range(image.width())
            if image.pixelColor(x, y).alpha() >= 200
        ]
        assert rendered
        assert any(
            color.red() == colors.text.red()
            and color.green() == colors.text.green()
            and color.blue() == colors.text.blue()
            for color in rendered
        )
