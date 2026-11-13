"""Presentation policy does not own user geometry or semantic state."""

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from napari_vipp.ui.design import (
    apply_text_role,
    body_point_size,
    content_width,
    install_window_policy,
    preview_height,
    type_style,
)


def test_roles_use_host_and_do_not_ratchet(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Arial", 12))
    label = QLabel("Section", host)
    apply_text_role(label, "title")
    original = label.styleSheet()
    for _ in range(8):
        apply_text_role(label, "title")
        QApplication.processEvents()
    assert label.styleSheet() == original
    assert "16pt" in original
    assert body_point_size(label) == 12
    host.setFont(QFont("Arial", 18))
    qtbot.waitUntil(lambda: "22pt" in label.styleSheet())
    assert body_point_size(label) == 18


def test_requested_font_preserved_and_reparent_refreshes(qtbot):
    first, second = QWidget(), QWidget()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first._requested_font = QFont("Arial", 16)
    second.setFont(QFont("Arial", 11))
    label = QLabel("Title", first)
    apply_text_role(label, "section")
    assert "16.5pt" in label.styleSheet()
    label.setParent(second)
    qtbot.waitUntil(lambda: "11.5pt" in label.styleSheet())
    second.setFont(QFont("Arial", 14))
    qtbot.waitUntil(lambda: "14.5pt" in label.styleSheet())


def test_metrics_scale_without_changing_values(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    label = QLabel("-12345.6789", host)
    host.setFont(QFont("Arial", 9))
    small = (content_width(label, label.text()), preview_height(label))
    host.setFont(QFont("Arial", 18))
    large = (content_width(label, label.text()), preview_height(label))
    assert large[0] > small[0]
    assert large[1] > small[1]
    assert label.text() == "-12345.6789"
    assert "pt" in type_style(None)
    with pytest.raises(ValueError):
        type_style(label, "unexpected")
    with pytest.raises(ValueError):
        apply_text_role(label, "unexpected")


def test_standalone_window_explicit_font_is_the_body_base(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.setFont(QFont("Arial", 16))
    assert body_point_size(window) == 16
    assert "20pt" in type_style(window, "title")
    for _ in range(5):
        assert "20pt" in type_style(window, "title")


def test_pending_role_refresh_after_reparent_and_old_host_deletion(qtbot):
    first, second = QWidget(), QWidget()
    qtbot.addWidget(second)
    second.setFont(QFont("Arial", 14))
    label = QLabel("Still owned", first)
    apply_text_role(label, "title")
    label.setParent(second)
    first.deleteLater()
    QCoreApplication.sendPostedEvents(first, QEvent.DeferredDelete)
    qtbot.waitUntil(lambda: "18pt" in label.styleSheet())


def test_semantic_styles_composed_on_leaf_never_reuse_its_styled_font(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Arial", 14))
    label = QLabel("Semantic title", host)
    font = QFont(label.font())
    font.setBold(True)
    label.setFont(font)
    for _ in range(12):
        label.setStyleSheet(type_style(label, "title"))
        QApplication.processEvents()
        assert "18pt" in label.styleSheet()


def test_window_first_show_fits_screen_but_later_geometry_is_user_owned(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    install_window_policy(window, "workspace")
    layout = QVBoxLayout(window)
    layout.addWidget(QLabel("Constructor creates children after policy", window))
    window.resize(9000, 9000)
    window.show()
    qtbot.waitUntil(lambda: window._vipp_window_policy.fitted)
    available = window.screen().availableGeometry()
    assert window.width() < available.width()
    assert window.height() < available.height()
    window.resize(450, 350)
    expected = window.size()
    window.setFont(QFont("Arial", 14))
    QApplication.processEvents()
    window.hide()
    window.show()
    QApplication.processEvents()
    assert window.size() == expected
    install_window_policy(window, "chooser")
    assert window._vipp_window_policy.fitted
    with pytest.raises(ValueError):
        install_window_policy(window, "unexpected")
