from __future__ import annotations

import pytest
from qtpy.compat import isalive

from napari_vipp.core.compute_history import PIPELINE_TIMING_HISTORY_PATH_ENV
from napari_vipp.ui import (
    presentation_settings,
    reader_support,
    recent_paths,
    updates,
    workflow_save_settings,
)


@pytest.fixture(autouse=True)
def _isolate_reader_support_session(monkeypatch):
    """Share diagnostics within each test, never across QApplication test reuse."""
    sessions = []

    def cache():
        if not sessions:
            sessions.append(reader_support._ReaderSupportSession(
                reader_support.QApplication.instance()
            ))
        return sessions[0]

    monkeypatch.setattr(reader_support, "_session_cache", cache)
    yield
    for session in sessions:
        session.close()
        session.deleteLater()


@pytest.fixture
def qtbot(qtbot, monkeypatch):
    """Keep registered test windows owned until pytest-qt closes them."""
    registered_widgets = []
    add_widget = qtbot.addWidget

    def register_widget(widget, *, before_close_func=None):
        add_widget(widget, before_close_func=before_close_func)
        registered_widgets.append(widget)
        registrations = qtbot._request.node.qt_widgets
        reference, before_close = registrations[-1]

        def live_reference():
            registered = reference()
            if registered is not None and isalive(registered):
                return registered
            return None

        # pytest-qt's close pass checks only whether its reference returns None.
        # Treat explicit native deletion as gone, even while we retain the Python
        # wrapper. Release wrappers only after teardown, never from a C++
        # destroyed signal while the native destructor may still be active.
        registrations[-1] = (live_reference, before_close)

    # pytest-qt stores weakrefs and dispatches events after the test returns.
    # Without an owner, GC can collect a window cycle inside a child's native
    # paint/layout call. The real application retains these windows. Match that
    # lifetime until pytest-qt's normal close/delete pass, which runs before
    # fixture teardown. Explicit deleteLater/parent destruction still work.
    monkeypatch.setattr(qtbot, "addWidget", register_widget)
    monkeypatch.setattr(qtbot, "add_widget", register_widget)
    yield qtbot
    registered_widgets.clear()


class _MemorySettings:
    def __init__(self, values: dict[str, object]):
        self._values = values

    def value(self, key: str, default=None):
        return self._values.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self._values[key] = value

    def sync(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolate_ui_settings(monkeypatch):
    """Keep UI preference tests deterministic and out of user settings."""
    values: dict[str, object] = {}
    settings = _MemorySettings(values)
    monkeypatch.setattr(recent_paths, "_settings", lambda: settings)
    monkeypatch.setattr(presentation_settings, "_settings", lambda: settings)
    monkeypatch.setattr(workflow_save_settings, "_settings", lambda: settings)
    monkeypatch.setattr(updates, "_settings", lambda: settings)
    monkeypatch.setenv(updates.DISABLE_AUTO_ENV, "1")


@pytest.fixture(autouse=True)
def _isolate_pipeline_timing_history(monkeypatch, tmp_path):
    """Keep every test and inherited subprocess out of user timing history."""

    monkeypatch.setenv(
        PIPELINE_TIMING_HISTORY_PATH_ENV,
        str(tmp_path / ".napari-vipp-test-state" / "pipeline-timing-history-v2.json"),
    )
