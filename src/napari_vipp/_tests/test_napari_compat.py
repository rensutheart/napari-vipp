from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy.QtWidgets import QMainWindow, QWidget

from napari_vipp.ui.napari_compat import viewer_camera, viewer_qt_window


def test_viewer_camera_prefers_scene_camera() -> None:
    scene_camera = object()
    legacy_camera = object()
    viewer = SimpleNamespace(
        scene=SimpleNamespace(camera=scene_camera),
        camera=legacy_camera,
    )

    assert viewer_camera(viewer) is scene_camera


@pytest.mark.parametrize(
    "scene",
    [None, SimpleNamespace(camera=None)],
)
def test_viewer_camera_falls_back_to_legacy_camera(scene: object) -> None:
    legacy_camera = object()
    viewer = SimpleNamespace(scene=scene, camera=legacy_camera)

    assert viewer_camera(viewer) is legacy_camera


def test_viewer_camera_fails_clearly_when_unavailable() -> None:
    with pytest.raises(RuntimeError, match="does not expose a camera"):
        viewer_camera(SimpleNamespace())


def test_viewer_qt_window_prefers_anchor_parent_traversal(qtbot) -> None:
    main_window = QMainWindow()
    container = QWidget(main_window)
    anchor = QWidget(container)
    qtbot.addWidget(main_window)

    class WindowThatMustNotBeRead:
        @property
        def qt_viewer(self):
            raise AssertionError("anchor traversal should run first")

        @property
        def _qt_window(self):
            raise AssertionError("legacy fallback should not be inspected")

    viewer = SimpleNamespace(window=WindowThatMustNotBeRead())

    assert viewer_qt_window(viewer, anchor=anchor) is main_window


def test_viewer_qt_window_uses_exposed_qt_viewer_before_legacy(qtbot) -> None:
    main_window = QMainWindow()
    qt_viewer = QWidget(main_window)
    qtbot.addWidget(main_window)

    class PublicWindow:
        @property
        def qt_viewer(self):
            return qt_viewer

        @property
        def _qt_window(self):
            raise AssertionError("legacy fallback should not be inspected")

    viewer = SimpleNamespace(window=PublicWindow())

    assert viewer_qt_window(viewer) is main_window


def test_viewer_qt_window_has_bounded_legacy_fallback(qtbot) -> None:
    legacy_window = QMainWindow()
    qtbot.addWidget(legacy_window)
    viewer = SimpleNamespace(
        window=SimpleNamespace(qt_viewer=None, _qt_window=legacy_window)
    )

    assert viewer_qt_window(viewer) is legacy_window


def test_viewer_qt_window_fails_clearly_when_unavailable() -> None:
    viewer = SimpleNamespace(window=SimpleNamespace())

    with pytest.raises(RuntimeError, match="Could not resolve"):
        viewer_qt_window(viewer)
