"""Original-input details summarize evidence without changing run authority."""

from dataclasses import replace
from html import escape
from pathlib import Path

import pytest
from qtpy.QtCore import QPoint, Qt, QTimer
from qtpy.QtGui import QFont, QPalette
from qtpy.QtWidgets import QDialog, QLabel, QTextBrowser, QWidget

from napari_vipp._tests.test_batch_reproduction_ui import _dialog as _batch_dialog
from napari_vipp._tests.test_batch_reproduction_ui import _request
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp.core.reproduction import ReproductionCheck, ReproductionRow
from napari_vipp.ui.batch_reproduction_details import ReproductionCheckDetailsDialog
from napari_vipp.ui.palette_roles import palette_is_dark, theme_colors


def _check(*, statuses=("matched",), can_run=True, problems=(), **kwargs):
    return ReproductionCheck(
        status="verified" if can_run else "mismatch",
        can_run=can_run,
        recorded_vipp_version="0.15.0a2",
        current_vipp_version="0.15.0a2",
        version_override_used=False,
        rows=tuple(
            ReproductionRow(
                "input", index, status, "Check evidence", path=f"field-{index}.npy"
            )
            for index, status in enumerate(statuses, 1)
        ),
        problems=problems,
        **kwargs,
    )


def _details(qtbot, check, *, parent=None):
    dialog = ReproductionCheckDetailsDialog(check, parent=parent)
    qtbot.addWidget(dialog)
    return dialog


def _text(dialog):
    return "\n".join(label.text() for label in dialog.findChildren(QLabel))


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [620, 440])
def test_success_is_compact_readable_and_theme_aware(qtbot, theme, width):
    from napari.qt import get_stylesheet

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setPalette(_palette(theme == "dark"))
    parent.setStyleSheet(get_stylesheet(theme))
    parent.setFont(QFont("Segoe UI", 10))
    dialog = _details(qtbot, _check(statuses=("matched",) * 14), parent=parent)
    dialog.resize(width, dialog.sizeHint().height())
    dialog.show()
    qtbot.wait(20)

    assert dialog.heading.text() == "Original inputs match"
    assert dialog.matched_value.text() == "14"
    assert dialog.attention_value.text() == "0"
    assert dialog.status_panel.property("status") == "success"
    assert palette_is_dark(dialog.palette()) is (theme == "dark")
    tone = theme_colors(dialog.palette()).success
    assert tone.accent.name() in dialog.styleSheet()
    assert dialog.heading.palette().color(QPalette.WindowText) == tone.foreground
    assert dialog.width() == width
    assert dialog.height() < 500
    assert dialog.problem_list.isHidden()
    assert not dialog.findChildren(QTextBrowser)
    assert dialog.review_target == "items"
    assert "Check batch again" not in dialog.guidance.text()
    assert "result" in dialog.disclaimer.text().lower()
    for label in dialog.findChildren(QLabel):
        assert label.textFormat() == Qt.PlainText
        if label.isVisible():
            top_left = label.mapTo(dialog, QPoint())
            assert top_left.x() >= 0
            assert top_left.y() >= 0
            assert top_left.x() + label.width() <= dialog.width()
            assert top_left.y() + label.height() <= dialog.height()
    for button in (dialog.review_button, dialog.close_button):
        assert button.isVisible()
        assert button.height() >= button.minimumSizeHint().height()


def test_mismatches_have_full_counts_breakdown_and_actionable_guidance(qtbot):
    states = (
        "matched",
        "matched",
        "changed",
        "changed",
        "missing",
        "extra",
        "ambiguous",
        "unreadable",
        "selector-mismatch",
    )
    problems = ("Original input is missing.", "The image selection has changed.")
    check = _check(statuses=states, can_run=False, problems=problems)
    before = check.to_dict()
    dialog = _details(qtbot, check)
    dialog.show()
    qtbot.wait(20)

    assert dialog.heading.text() == "Inputs need attention"
    assert dialog.matched_value.text() == "2"
    assert dialog.attention_value.text() == "7"
    assert dialog.status_panel.property("status") == "warning"
    assert theme_colors(dialog.palette()).warning.accent.name() in dialog.styleSheet()
    breakdown = dialog.breakdown.text().lower()
    for category in ("changed", "missing", "unexpected", "ambiguous", "unreadable"):
        assert category in breakdown
    assert "selection" in breakdown
    assert "2" in breakdown
    assert "Check batch again" in dialog.guidance.text()
    assert dialog.review_target == "attention"
    assert not dialog.problem_list.isHidden()
    assert dialog.problem_list.count() == len(problems)
    assert [dialog.problem_list.item(i).text() for i in range(len(problems))] == list(
        problems
    )
    assert dialog.problem_list.maximumHeight() <= 180
    assert check.to_dict() == before


@pytest.mark.parametrize("can_run", [True, False])
def test_status_colors_follow_live_palette_changes(qtbot, can_run):
    dialog = _details(qtbot, _check(can_run=can_run))
    dialog.show()
    foregrounds = []
    for dark in (False, True, False):
        dialog.setPalette(_palette(dark))
        qtbot.wait(10)
        colors = theme_colors(dialog.palette())
        tone = colors.success if can_run else colors.warning
        assert palette_is_dark(dialog.palette()) is dark
        assert tone.surface.name() in dialog.styleSheet()
        assert dialog.heading.palette().color(QPalette.WindowText) == tone.foreground
        foregrounds.append(tone.foreground.name())
    assert foregrounds[0] == foregrounds[2] != foregrounds[1]


@pytest.mark.parametrize("can_run", [True, False])
def test_status_colors_follow_the_styled_parent_theme(qtbot, can_run):
    from napari.qt import get_stylesheet

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setPalette(_palette(False))
    parent.setStyleSheet(get_stylesheet("light"))
    dialog = _details(qtbot, _check(can_run=can_run), parent=parent)
    dialog.show()
    for theme in ("light", "dark", "light"):
        parent.setPalette(_palette(theme == "dark"))
        parent.setStyleSheet(get_stylesheet(theme))
        qtbot.wait(20)
        colors = theme_colors(dialog.palette())
        tone = colors.success if can_run else colors.warning
        assert palette_is_dark(dialog.palette()) is (theme == "dark")
        assert tone.surface.name() in dialog.styleSheet()
        assert dialog.heading.palette().color(QPalette.WindowText) == tone.foreground


def test_version_only_block_does_not_look_like_success(qtbot):
    check = replace(
        _check(statuses=("matched",) * 3, can_run=False),
        status="version-mismatch",
        recorded_vipp_version="0.14.0a1",
        problems=("The recorded VIPP version differs from the current version.",),
    )
    dialog = _details(qtbot, check)

    assert dialog.heading.text() == "Checks need attention"
    assert dialog.matched_value.text() == "3"
    assert dialog.attention_value.text() == "0"
    assert dialog.status_panel.property("status") == "warning"
    assert dialog.review_target == "setup"
    assert "0.14.0a1" in dialog.version_label.text()
    assert "0.15.0a2" in dialog.version_label.text()
    assert "blocked" in _text(dialog).lower()


def test_accepted_version_difference_stays_visible_as_deviation(qtbot):
    check = replace(
        _check(), recorded_vipp_version="0.14.0a1", version_override_used=True
    )
    dialog = _details(qtbot, check)

    assert dialog.heading.text() == "Original inputs match"
    assert dialog.status_panel.property("status") == "warning"
    assert dialog.review_target == "items"
    assert "0.14.0a1" in dialog.version_label.text()
    assert "0.15.0a2" in dialog.version_label.text()
    assert "accepted" in dialog.version_label.text().lower()
    assert "recorded" in dialog.version_label.text().lower()
    assert "result" in dialog.disclaimer.text().lower()


def test_no_rows_does_not_claim_original_inputs_match(qtbot):
    dialog = _details(qtbot, _check(statuses=(), can_run=False))

    assert dialog.heading.text() == "Inputs not verified"
    assert dialog.matched_value.text() == "0"
    assert dialog.attention_value.text() == "0"
    assert dialog.status_panel.property("status") == "warning"
    assert dialog.review_target == "setup"
    assert "Original inputs match" not in _text(dialog)


@pytest.mark.parametrize(
    "states,generic",
    [
        (
            ("missing",),
            "Original inputs did not all match; review every input comparison below.",
        ),
        ((), "No original inputs have been verified."),
    ],
)
@pytest.mark.parametrize("has_detail", [False, True])
def test_repeated_generic_summary_is_omitted_but_specific_problems_remain(
    qtbot, states, generic, has_detail
):
    detail = "Reader could not open input 2. Check the selected collection."
    problems = (generic, detail) if has_detail else (generic,)
    check = _check(statuses=states, can_run=False, problems=problems)
    dialog = _details(qtbot, check)

    assert dialog.problem_list.count() == int(has_detail)
    assert dialog.problem_list.isHidden() is (not has_detail)
    if has_detail:
        assert dialog.problem_list.item(0).text() == detail
    assert generic not in (
        dialog.problem_list.item(index).text()
        for index in range(dialog.problem_list.count())
    )
    assert check.problems == problems


def test_matching_versions_remain_green_even_when_inputs_need_attention(qtbot):
    dialog = _details(qtbot, _check(statuses=("missing",), can_run=False))
    dialog.show()
    qtbot.wait(10)
    colors = theme_colors(dialog.palette())

    assert (
        dialog.heading.palette().color(QPalette.WindowText) == colors.warning.foreground
    )
    assert dialog.version_label.palette().color(QPalette.WindowText) == (
        colors.success.foreground
    )


def test_large_collection_counts_every_check_without_per_row_widgets(qtbot):
    small = _details(qtbot, _check())
    check = _check(statuses=("matched",) * 10001 + ("missing",) * 2, can_run=False)
    large = _details(qtbot, check)

    assert large.matched_value.text() == "10,001"
    assert large.attention_value.text() == "2"
    assert sum(
        int(label.text().replace(",", ""))
        for label in (large.matched_value, large.attention_value)
    ) == len(check.rows)
    assert large.problem_list.count() == 0
    assert len(large.findChildren(QWidget)) == len(small.findChildren(QWidget))
    assert "field-" not in _text(large)


def test_long_untrusted_diagnostics_are_plain_text_complete_and_bounded(qtbot):
    untrusted = '<img src="file:///not-read.png"><b>missing & changed</b> '
    problems = tuple(
        f"{index}: {untrusted}{'long path / ' * 80}" for index in range(120)
    )
    check = replace(
        _check(statuses=("unreadable",), can_run=False, problems=problems),
        recorded_vipp_version="<b>unknown</b>",
    )
    dialog = _details(qtbot, check)
    dialog.resize(440, dialog.sizeHint().height())
    dialog.show()
    qtbot.wait(20)

    assert not dialog.findChildren(QTextBrowser)
    assert dialog.problem_list.count() == len(problems)
    assert dialog.problem_list.maximumHeight() <= 180
    assert dialog.width() == 440
    assert dialog.height() <= 700
    for index, problem in enumerate(problems):
        item = dialog.problem_list.item(index)
        assert item.text() == problem
        assert not item.flags() & Qt.ItemIsEditable
        assert item.toolTip()
        assert escape(problem) in item.toolTip()
        assert "<img" not in item.toolTip()
    assert "<b>unknown</b>" in dialog.version_label.text()
    assert all(
        label.textFormat() == Qt.PlainText for label in dialog.findChildren(QLabel)
    )
    dialog.problem_list.scrollToBottom()
    qtbot.wait(10)
    assert dialog.problem_list.verticalScrollBar().maximum() > 0
    assert dialog.problem_list.visualItemRect(
        dialog.problem_list.item(len(problems) - 1)
    ).intersects(dialog.problem_list.viewport().rect())


def test_details_only_read_the_existing_snapshot(qtbot, monkeypatch):
    check = _check(statuses=("missing",), can_run=False)
    before = check.to_dict()

    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("Opening check details must not access source files")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", unexpected_io)
        for name in ("open", "read_bytes", "read_text", "stat"):
            patch.setattr(Path, name, unexpected_io)
        dialog = _details(qtbot, check)
        dialog.review_button.click()
    assert dialog.result() == QDialog.Accepted
    assert check.to_dict() == before


@pytest.mark.parametrize(
    "action,expected", [("review", QDialog.Accepted), ("close", QDialog.Rejected)]
)
def test_details_actions_only_accept_or_close(qtbot, action, expected):
    dialog = _details(qtbot, _check())
    QTimer.singleShot(0, getattr(dialog, f"{action}_button").click)
    assert dialog.exec() == expected


@pytest.mark.parametrize(
    "case,target,tab,filter_index",
    [
        ("success", "items", 1, 0),
        ("mismatch", "attention", 1, 1),
        ("version", "setup", 0, 2),
        ("empty", "setup", 0, 2),
    ],
)
def test_review_navigates_without_rechecking_running_or_changing_mode(
    qtbot, tmp_path, monkeypatch, case, target, tab, filter_index
):
    request = _request()
    host, result, previewed = _batch_dialog(qtbot, tmp_path, request=request)
    states = ("matched", "matched", "changed" if case == "mismatch" else "matched")
    check = _check(
        statuses=() if case == "empty" else states, can_run=case == "success"
    )
    if case == "version":
        check = replace(
            check, status="version-mismatch", recorded_vipp_version="0.14.0a1"
        )
    result = replace(result, reproduction=check)
    host.apply_preview_result(result, preview_representative=False)
    checked, runs, modes, opened = [], [], [], []
    host._actions = replace(host._actions, check_batch=lambda *_: checked.append(True))
    host.runRequested.connect(runs.append)
    host.reproductionChanged.connect(modes.append)
    host.tabs.setCurrentIndex(3)
    host.item_filter.setCurrentIndex(2)
    host.item_search.setText("stale search")
    host._item_page = 5
    before_values = host.values()
    before_block = host._reproduction_block_reason()

    def review(dialog):
        opened.append(dialog)
        assert dialog.review_target == target
        dialog.review_button.click()
        return dialog.result()

    monkeypatch.setattr(ReproductionCheckDetailsDialog, "exec", review)
    host._show_reproduction_details()

    assert len(opened) == 1
    assert opened[0].result() == QDialog.Accepted
    assert host.tabs.currentIndex() == tab
    assert host.item_filter.currentIndex() == filter_index
    if target != "setup":
        assert not host.item_search.text()
        assert host._item_page == 0
        assert host.preview_table.rowCount() == (1 if case == "mismatch" else 3)
    assert host.values() == before_values
    assert host._reproduction_check() is check
    assert host._preview_result is result
    assert host._reproduction_request is request
    assert host._reproduction_block_reason() == before_block
    assert checked == previewed == runs == modes == []
    if not check.can_run:
        host._request_run()
        assert not runs


def test_close_details_does_not_navigate_or_modify_review(qtbot, tmp_path, monkeypatch):
    request = _request()
    host, result, _ = _batch_dialog(qtbot, tmp_path, request=request)
    check = _check()
    result = replace(result, reproduction=check)
    host.apply_preview_result(result, preview_representative=False)
    host.tabs.setCurrentIndex(3)
    host.item_search.setText("retain this search")
    host.item_filter.setCurrentIndex(2)
    host._item_page = 4

    def close(dialog):
        dialog.close_button.click()
        return dialog.result()

    monkeypatch.setattr(ReproductionCheckDetailsDialog, "exec", close)
    host._show_reproduction_details()

    assert host.tabs.currentIndex() == 3
    assert host.item_search.text() == "retain this search"
    assert host.item_filter.currentIndex() == 2
    assert host._item_page == 4
    assert host._preview_result is result
    assert host._reproduction_check() is check


def test_details_are_not_created_without_an_existing_check(
    qtbot, tmp_path, monkeypatch
):
    host, _, _ = _batch_dialog(qtbot, tmp_path, request=_request())
    before = host.values()

    def unexpected_dialog(_dialog):
        raise AssertionError("No details dialog should be shown before a check exists")

    monkeypatch.setattr(ReproductionCheckDetailsDialog, "exec", unexpected_dialog)
    host._show_reproduction_details()

    assert host.values() == before
    assert host._reproduction_check() is None
