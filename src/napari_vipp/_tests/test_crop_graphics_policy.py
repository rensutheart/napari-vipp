"""Crop chooses a display method once; it never changes the viewer backend."""

from types import SimpleNamespace

import pytest

from napari_vipp import _widget
from napari_vipp.ui.viewer_graphics_status import (
    ViewerGraphicsStatus,
    windows_crop_needs_bounds_preview,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, True),
        (ViewerGraphicsStatus("", "NVIDIA Corporation"), True),
        (ViewerGraphicsStatus("Intel(R) Iris(R) Xe Graphics", "Intel"), True),
        (ViewerGraphicsStatus("Intel(R) Arc(TM) A770 Graphics", "Intel"), True),
        (ViewerGraphicsStatus("GeForce RTX 4050/PCIe/SSE2", "NVIDIA"), False),
        (ViewerGraphicsStatus("AMD Radeon RX 6600"), False),
        (ViewerGraphicsStatus("llvmpipe", "Intel"), False),
        (ViewerGraphicsStatus("Unidentified hardware renderer"), True),
    ],
)
def test_windows_crop_renderer_policy(status, expected):
    assert windows_crop_needs_bounds_preview(status) is expected


def test_process_opt_in_chooses_once_without_querying_or_switching_gl(
    qtbot, monkeypatch
):
    from napari_vipp._tests.test_widget import _Viewer

    monkeypatch.setenv("VIPP_SAFE_CROP_PREVIEW", "1")
    monkeypatch.setattr(
        _widget,
        "query_viewer_graphics",
        lambda viewer: pytest.fail("An explicit bounds preview needs no GL query."),
    )
    editor = _widget.VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(editor)
    monkeypatch.delenv("VIPP_SAFE_CROP_PREVIEW")
    assert editor._crop_uses_bounds_preview()
    editor._probe_crop_graphics()
    assert editor._crop_uses_bounds_preview()


@pytest.mark.parametrize("first_crop_before_probe", [False, True])
def test_crop_renderer_choice_is_stable_and_does_not_probe_on_click(
    monkeypatch, first_crop_before_probe
):
    calls = []

    def read(viewer):
        calls.append(viewer)
        return ViewerGraphicsStatus("NVIDIA GeForce RTX")

    monkeypatch.setattr(_widget, "query_viewer_graphics", read)
    obj = SimpleNamespace(
        viewer=object(),
        _closing=False,
        _crop_bounds_mode=None,
        _selected_viewer_refresh_in_progress=False,
        graph_view=SimpleNamespace(node_press_dispatch_active=lambda: False),
        _crop_draft_node_id=None,
    )
    if first_crop_before_probe:
        assert _widget.VippWidget._crop_uses_bounds_preview(obj)
    for _ in range(3):
        _widget.VippWidget._probe_crop_graphics(obj)
        assert (
            _widget.VippWidget._crop_uses_bounds_preview(obj) is first_crop_before_probe
        )
    assert calls == ([] if first_crop_before_probe else [obj.viewer])
