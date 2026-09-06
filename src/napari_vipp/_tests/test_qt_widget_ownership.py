"""Test windows retain a Python owner while queued Qt callbacks can use them."""

import gc
import weakref

from qtpy.compat import isalive
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QLabel, QWidget


def test_registered_window_survives_local_reference_and_cyclic_gc(qtbot):
    window = QWidget()
    window.reference_cycle = window
    label = QLabel("Queued native paint", window)
    qtbot.addWidget(window)
    window.show()
    window_reference = weakref.ref(window)
    del window

    gc.collect()

    assert window_reference() is not None
    assert isalive(window_reference())
    assert isalive(label)
    QApplication.processEvents()


def test_registration_preserves_explicit_parent_and_child_destruction(qtbot):
    window = QWidget()
    label = QLabel("Child", window)
    qtbot.add_widget(window)
    window.deleteLater()

    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    assert not isalive(window)
    assert not isalive(label)
    QApplication.processEvents()
