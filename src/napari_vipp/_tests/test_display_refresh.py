"""Exercise display controls against rendered pixels, not just combo labels."""

import numpy as np
import pytest

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.app import _configure_initial_workflow
from napari_vipp.core.preview import thumbnail_contrast_limits
from napari_vipp.core.workflow import serialize_workflow


def _pixels(widget, node_id="input"):
    return widget.graph_view._cards[node_id].preview.source_pixmap().toImage()


@pytest.mark.parametrize("auto_calculate", (False, True))
@pytest.mark.parametrize(
    ("field", "index"),
    (("Contrast method", 1), ("Contrast method", 2), ("Contrast based on", 0)),
)
def test_display_picker_repaints_cached_pixels(
    qtbot, monkeypatch, field, index, auto_calculate,
):
    # A strong outlier and a brighter second slice make both choices visible.
    plane = np.linspace(100, 1000, 64 * 64).reshape(64, 64)
    data = np.stack((plane, plane * 10)).astype(np.uint16)
    data[0, -1, -1] = 60000
    data.setflags(write=False)
    widget = VippWidget(_Viewer(data))
    qtbot.addWidget(widget)
    widget.auto_recalculate_checkbox.setChecked(auto_calculate)
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    widget.thumbnail_contrast_combo.setCurrentText("Percentile")
    widget._populate_preview_display_menu()
    outputs = dict(widget.pipeline.outputs)
    snapshots = {key: value.copy() for key, value in outputs.items()}
    workflow = serialize_workflow(widget.pipeline)

    def no_calculation(*_args, **_kwargs):
        pytest.fail("A display preference must not run the scientific workflow")

    monkeypatch.setattr(widget.pipeline, "run", no_calculation)
    before = _pixels(widget)
    picker = widget._preview_menu_combos[field]
    picker.setCurrentIndex(index)
    qtbot.waitUntil(lambda: _pixels(widget) != before, timeout=5000)
    assert serialize_workflow(widget.pipeline) == workflow
    assert widget.pipeline.outputs.keys() == outputs.keys()
    for key, output in outputs.items():
        assert widget.pipeline.outputs[key] is output
        np.testing.assert_array_equal(output, snapshots[key])
    assert not widget._debounce_timer.isActive()
    assert widget._active_pipeline_run_id is None


@pytest.mark.parametrize("field", ("Contrast method", "Contrast based on"))
def test_startup_display_picker_repaints(qtbot, field):
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    _configure_initial_workflow(widget)
    qtbot.waitUntil(lambda: widget.graph_view.node_has_thumbnail("input"), timeout=5000)
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    widget._populate_preview_display_menu()
    before = _pixels(widget)
    picker = widget._preview_menu_combos[field]
    picker.setCurrentIndex(2 if field == "Contrast method" else 0)
    qtbot.waitUntil(lambda: _pixels(widget) != before, timeout=5000)


def test_startup_middle_slice_can_legitimately_look_unchanged(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    _configure_initial_workflow(widget)
    qtbot.waitUntil(lambda: widget.graph_view.node_has_thumbnail("input"), timeout=5000)
    # Explicitly fix the display position independent of the fake host's shape.
    monkeypatch.setattr(widget, "_current_step", lambda: (5, 0, 0))
    monkeypatch.setattr(widget, "_current_step_nsteps", lambda: (12, 96, 128))
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget._update_thumbnails()
    widget._populate_preview_display_menu()
    data = widget.pipeline.outputs["input"]
    for population in (data, data[5]):
        for method in ("Percentile", "Min-max"):
            assert thumbnail_contrast_limits(
                population, contrast_mode=method,
            ) == (0.0, 255.0)
    before = _pixels(widget)
    for scope_index in (0, 1):
        widget._preview_menu_combos["Contrast based on"].setCurrentIndex(scope_index)
        for method_index in (0, 1, 2):
            widget._preview_menu_combos["Contrast method"].setCurrentIndex(method_index)
            qtbot.waitUntil(
                lambda: not widget._pending_thumbnail_contrast_limit_keys
                and not widget._queued_thumbnail_contrast_limit_requests,
                timeout=5000,
            )
            assert _pixels(widget) == before
