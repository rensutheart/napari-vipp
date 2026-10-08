from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QRect,
    QSize,
    QSortFilterProxyModel,
    Qt,
)
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import QListView, QMainWindow, QStyleOptionViewItem, QWidget

from napari_vipp.ui.napari_compat import (
    install_layer_delegate_safety,
    viewer_camera,
    viewer_qt_window,
)


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


class _LayerRoleModel(QAbstractListModel):
    """Keep a valid Qt proxy index while its pending layer roles disappear."""

    def __init__(self, parent):
        super().__init__(parent)
        self.roles = {}

    def rowCount(self, parent=QModelIndex()):  # noqa: N802, B008
        return 0 if parent.isValid() else 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        return self.roles.get(role) if index.isValid() else None

    def all_loaded(self):
        return True


@pytest.fixture
def layer_paint_context(qtbot):
    from napari._qt.containers._layer_delegate import LayerDelegate
    from napari._qt.containers.qt_layer_model import LoadedRole, ThumbnailRole

    view = QListView()
    qtbot.addWidget(view)
    model = _LayerRoleModel(view)
    proxy = QSortFilterProxyModel(view)
    proxy.setSourceModel(model)
    view.setModel(proxy)
    delegate = LayerDelegate(view)
    view.setItemDelegate(delegate)
    thumbnail = QImage(32, 32, QImage.Format.Format_RGBA8888)
    thumbnail.fill(Qt.GlobalColor.red)
    model.roles.update(
        {
            LoadedRole: False,
            ThumbnailRole: thumbnail,
            Qt.ItemDataRole.SizeHintRole: QSize(200, 34),
        }
    )
    viewer = SimpleNamespace(
        window=SimpleNamespace(qt_viewer=SimpleNamespace(layers=view))
    )
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 200, 34)
    option.widget = view
    canvas = QImage(200, 40, QImage.Format.Format_RGBA8888)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    context = SimpleNamespace(
        viewer=viewer,
        view=view,
        model=model,
        delegate=delegate,
        index=proxy.index(0, 0),
        option=option,
        painter=painter,
        canvas=canvas,
        loaded_role=LoadedRole,
        thumbnail_role=ThumbnailRole,
    )
    yield context
    painter.end()
    delegate._load_movie.stop()


def test_layer_delegate_missing_roles_reproduce_and_fix_height_error(
    layer_paint_context,
) -> None:
    ctx = layer_paint_context
    ctx.model.roles.clear()
    assert ctx.index.isValid()  # The upstream validity check alone cannot help.
    with pytest.raises(AttributeError, match="height"):
        ctx.delegate._paint_loading(ctx.painter, ctx.option, ctx.index)
    ctx.delegate._load_movie.stop()

    assert install_layer_delegate_safety(ctx.viewer)
    # Exercise the full original delegate, not just a replacement test helper.
    ctx.delegate.paint(ctx.painter, ctx.option, ctx.index)
    assert ctx.delegate._load_movie.state() == ctx.delegate._load_movie.NotRunning


def test_layer_delegate_invalid_index_is_safe(layer_paint_context) -> None:
    ctx = layer_paint_context
    install_layer_delegate_safety(ctx.viewer)
    ctx.delegate._paint_loading(ctx.painter, ctx.option, QModelIndex())
    ctx.delegate._paint_thumbnail(ctx.painter, ctx.option, QModelIndex())
    assert ctx.delegate._load_movie.state() == ctx.delegate._load_movie.NotRunning


@pytest.mark.parametrize("loaded", [False, True])
def test_layer_delegate_normal_decorations_keep_original_painting(
    layer_paint_context, loaded
) -> None:
    ctx = layer_paint_context
    ctx.model.roles[ctx.loaded_role] = loaded
    method = "_paint_thumbnail" if loaded else "_paint_loading"
    getattr(ctx.delegate, method)(ctx.painter, ctx.option, ctx.index)
    original = ctx.canvas.copy()
    ctx.canvas.fill(Qt.GlobalColor.transparent)
    assert install_layer_delegate_safety(ctx.viewer)
    getattr(ctx.delegate, method)(ctx.painter, ctx.option, ctx.index)
    assert ctx.canvas == original


@pytest.mark.parametrize("loaded", [False, True])
def test_layer_delegate_valid_item_without_size_hint_uses_row_geometry(
    layer_paint_context, loaded
) -> None:
    ctx = layer_paint_context
    ctx.model.roles[ctx.loaded_role] = loaded
    del ctx.model.roles[Qt.ItemDataRole.SizeHintRole]
    baseline = ctx.canvas.copy()
    assert install_layer_delegate_safety(ctx.viewer)
    method = "_paint_thumbnail" if loaded else "_paint_loading"
    getattr(ctx.delegate, method)(ctx.painter, ctx.option, ctx.index)
    assert ctx.canvas != baseline
    if loaded:
        assert ctx.canvas.pixelColor(5, 5) == Qt.GlobalColor.red


def test_layer_delegate_missing_thumbnail_does_not_call_from_image(
    layer_paint_context,
) -> None:
    ctx = layer_paint_context
    ctx.model.roles[ctx.loaded_role] = True
    del ctx.model.roles[ctx.thumbnail_role]
    baseline = ctx.canvas.copy()
    assert install_layer_delegate_safety(ctx.viewer)
    ctx.delegate._paint_thumbnail(ctx.painter, ctx.option, ctx.index)
    assert ctx.canvas == baseline


def test_layer_delegate_safety_is_per_viewer_and_idempotent(
    layer_paint_context,
) -> None:
    from napari._qt.containers._layer_delegate import LayerDelegate

    ctx = layer_paint_context
    original_class_method = LayerDelegate._paint_loading
    original_delegate = ctx.view.itemDelegate()
    assert install_layer_delegate_safety(ctx.viewer)
    first_wrapper = ctx.delegate._paint_loading
    assert install_layer_delegate_safety(ctx.viewer)
    assert ctx.delegate._paint_loading is first_wrapper
    assert ctx.view.itemDelegate() is original_delegate
    assert LayerDelegate._paint_loading is original_class_method


def test_layer_delegate_safety_does_not_hide_other_paint_errors(
    layer_paint_context, monkeypatch
) -> None:
    from napari._qt.containers._layer_delegate import LayerDelegate

    ctx = layer_paint_context

    def broken_paint(*args):
        raise RuntimeError("another paint problem")

    monkeypatch.setattr(LayerDelegate, "_paint_loading", broken_paint)
    assert install_layer_delegate_safety(ctx.viewer)
    with pytest.raises(RuntimeError, match="another paint problem"):
        ctx.delegate._paint_loading(ctx.painter, ctx.option, ctx.index)


@pytest.mark.parametrize(
    "viewer",
    [SimpleNamespace(), SimpleNamespace(window=SimpleNamespace(qt_viewer=None))],
)
def test_layer_delegate_safety_ignores_model_only_viewers(viewer) -> None:
    assert not install_layer_delegate_safety(viewer)
