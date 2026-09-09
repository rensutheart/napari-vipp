"""Reproduction actions share the status row without reserving hidden rows."""

from dataclasses import replace

import pytest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

from napari_vipp._tests.test_batch_reproduction_ui import _check, _dialog, _request
from napari_vipp._tests.test_completed_reproduction_banner import _finish_verified_run


def _show_banner(qtbot, dialog, width):
    """Measure the real banner independently of the workspace minimum width."""
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(dialog.reproduction_banner)
    layout.addStretch()
    host.setFixedWidth(width)
    host.resize(width, 400)
    host.show()
    QApplication.processEvents()
    return dialog.reproduction_banner


def _assert_side_by_side(dialog, banner):
    layout = banner.layout()
    assert isinstance(layout, QHBoxLayout)
    assert layout.count() == 2
    assert layout.stretch(0) > layout.stretch(1)
    text_layout = layout.itemAt(0).layout()
    action_layout = layout.itemAt(1).layout()
    text_rect = text_layout.geometry()
    action_rect = action_layout.geometry()
    assert text_rect.right() < action_rect.left()
    assert abs(text_rect.center().y() - action_rect.center().y()) <= 1
    for label in (dialog.reproduction_mode_label, dialog.reproduction_status_label):
        assert label.wordWrap()
        assert label.width() > 0
        assert label.height() >= label.heightForWidth(label.width())
        assert banner.rect().contains(label.geometry())
    for button in (
        dialog.reproduction_details_button,
        dialog.reproduction_new_data_button,
        dialog.reproduction_version_button,
    ):
        if button.isVisible():
            assert button.width() >= button.minimumSizeHint().width()
            assert banner.rect().contains(button.geometry())
    return text_layout, action_layout


@pytest.mark.parametrize("width", [640, 1280])
def test_completed_banner_keeps_its_only_action_beside_the_status(
    qtbot, tmp_path, width
):
    dialog, _ = _finish_verified_run(qtbot, tmp_path)
    banner = _show_banner(qtbot, dialog, width)
    text_layout, action_layout = _assert_side_by_side(dialog, banner)
    button = dialog.reproduction_new_data_button
    assert button.isVisible()
    assert dialog.reproduction_details_button.isHidden()
    assert dialog.reproduction_version_button.isHidden()
    # The hidden details button and version-acknowledgement row take no space.
    assert action_layout.sizeHint() == button.sizeHint()
    assert action_layout.geometry().left() == button.geometry().left()
    assert action_layout.geometry().right() == button.geometry().right()
    text_center = text_layout.geometry().center().y()
    assert abs(text_center - button.geometry().center().y()) <= 1
    assert button.width() == button.sizeHint().width()
    margins = banner.layout().contentsMargins()
    assert banner.height() == (
        margins.top()
        + max(text_layout.geometry().height(), button.height())
        + margins.bottom()
    )


@pytest.mark.parametrize("width", [640, 1280])
@pytest.mark.parametrize("version_mismatch", [False, True])
def test_checked_banner_groups_actions_on_the_right(
    qtbot, tmp_path, width, version_mismatch
):
    request = _request(version="0.0.1" if version_mismatch else None)
    dialog, preview, _ = _dialog(qtbot, tmp_path, request=request)
    check = _check(
        request,
        can_run=not version_mismatch,
        status="version-mismatch" if version_mismatch else "verified",
    )
    dialog.apply_preview_result(
        replace(preview, reproduction=check), preview_representative=False
    )
    banner = _show_banner(qtbot, dialog, width)
    _, action_layout = _assert_side_by_side(dialog, banner)
    details = dialog.reproduction_details_button
    new_data = dialog.reproduction_new_data_button
    version = dialog.reproduction_version_button
    assert details.isVisible()
    assert new_data.isVisible()
    assert details.geometry().right() < new_data.geometry().left()
    assert details.geometry().top() == new_data.geometry().top()
    assert details.geometry().bottom() == new_data.geometry().bottom()
    assert version.isVisible() is version_mismatch
    if version_mismatch:
        assert version.geometry().top() > new_data.geometry().bottom()
        assert version.geometry().left() == details.geometry().left()
        assert version.geometry().right() == new_data.geometry().right()
        assert action_layout.sizeHint().height() == (
            details.sizeHint().height()
            + action_layout.spacing()
            + version.sizeHint().height()
        )
    else:
        assert action_layout.sizeHint().height() == new_data.sizeHint().height()


def test_new_data_banner_does_not_reserve_an_empty_action_column(qtbot, tmp_path):
    dialog, _, _ = _dialog(qtbot, tmp_path)
    dialog.set_reproduction_request(None, package_context=True)
    banner = _show_banner(qtbot, dialog, 640)
    layout = banner.layout()
    assert isinstance(layout, QHBoxLayout)
    assert layout.itemAt(1).isEmpty()
    for button in (
        dialog.reproduction_details_button,
        dialog.reproduction_new_data_button,
        dialog.reproduction_version_button,
    ):
        assert button.isHidden()
    margins = layout.contentsMargins()
    assert dialog.reproduction_status_label.width() == (
        banner.width() - margins.left() - margins.right()
    )
