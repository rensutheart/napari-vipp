"""Read-only pane/model contracts; native rendering is separately qualified."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from qtpy.QtCore import QEvent, QPoint
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QFrame, QWidget

from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.review_images import default_review_settings, prepare_review_input
from napari_vipp.ui import image_review
from napari_vipp.ui.image_review import ImageReviewWindow, visibility_colormap
from napari_vipp.ui.image_review_controller import (
    REVIEW_STALE_MESSAGE,
    REVIEW_UPDATING_MESSAGE,
)
from napari_vipp.ui.napari_compat import viewer_camera


def _input(data, order="yx", *, kind=None, name="Image", unit=None, scale=1):
    axes = tuple(
        AxisMetadata(axis, "time" if axis == "t" else "space", unit=unit, scale=scale)
        for axis in order
    )
    if data.ndim == len(order) + 1:
        axes += (AxisMetadata("rgb" if data.shape[-1] == 3 else "rgba", "channel"),)
    state = image_state_from_array(data, axes=axes)
    if kind is not None:
        state = replace(state, kind=kind)
    return prepare_review_input(data, state, name)


class _ModelOnlyQtViewer(QWidget):
    def __init__(self, viewer, show_welcome_screen=False):
        super().__init__()
        self.viewer = viewer
        self.dims = QWidget(self)


@pytest.fixture
def model_window(qtbot, monkeypatch):
    monkeypatch.setattr(image_review, "QtViewer", _ModelOnlyQtViewer)
    windows = []

    def create(inputs, settings=None, callback=None):
        window = ImageReviewWindow(
            inputs=inputs, settings=settings, on_settings_changed=callback
        )
        windows.append(window)
        qtbot.addWidget(window)
        return window

    yield create
    for window in windows:
        window.shutdown()


def test_independent_wrappers_readonly_buffers_and_main_viewer_isolation(model_window):
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    original = data.copy()
    main = ViewerModel()
    layer = main.add_image(data)
    layer.opacity = 0.8
    window = model_window((_input(data), _input(data * 2)))
    left, right = window.panes
    assert left.viewer is not right.viewer and left.viewer is not main
    assert left.layers["a"][0] is not right.layers["a"][0]
    for pane in window.panes:
        assert np.shares_memory(pane.layers["a"][0].data, data)
        assert not pane.layers["a"][0].data.flags.writeable
    window._set_style("a", "opacity", 0.2)
    assert left.layers["a"][0].opacity == right.layers["a"][0].opacity == 0.2
    assert layer.opacity == 0.8
    np.testing.assert_array_equal(data, original)
    assert data.flags.writeable


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_live_same_grid_inputs_replace_data_without_rebuilding_or_resetting_review(
    model_window, monkeypatch, ndisplay
):
    data = np.arange(240, dtype=np.float32).reshape(4, 3, 4, 5)
    mask = data > 100
    changed_data, changed_mask = data + 10, data < 50
    original_data, original_mask = changed_data.copy(), changed_mask.copy()
    records = []
    window = model_window(
        (_input(data, "tzyx", name="Before"), _input(mask, "tzyx", name="Mask")),
        settings={"ndisplay": ndisplay, "link_navigation": False},
        callback=records.append,
    )
    window._set_style("a", "opacity", 0.3)
    window._set_style("a", "colormap", "magma")
    window._set_style("a", "lock_contrast", True)
    window._set_style("b", "mask_color", "#FF0000")
    window._navigate_axis("t", 3)
    poses = []
    for index, viewer in enumerate(window.viewers):
        camera = viewer_camera(viewer)
        camera.center = (1 + index, 2, 3)
        camera.zoom = 2 + index
        camera.angles = (10, 20, 30 + index)
        poses.append((camera.center, camera.zoom, camera.angles, viewer.dims.point))
    layers = [tuple(pane.viewer.layers) for pane in window.panes]
    recipe = window.settings
    records.clear()

    def unexpected(*args, **kwargs):
        raise AssertionError("Live replacement must not fit or rebuild the panes")

    monkeypatch.setattr(window, "_build_layers", unexpected)
    monkeypatch.setattr(window, "fit_view", unexpected)
    window.update_inputs(
        (
            _input(changed_data, "tzyx", name="After"),
            _input(changed_mask, "tzyx", name="New mask"),
        )
    )
    assert window.settings == recipe and records == []
    assert window._original_limits["a"] == (10, 249)
    assert window._input_sections["a"].title() == "A · After"
    assert window.left_combo.itemText(window.left_combo.findData("a")) == "A · After"
    for pane, old_layers, pose in zip(window.panes, layers, poses, strict=True):
        assert tuple(pane.viewer.layers) == old_layers
        camera = viewer_camera(pane.viewer)
        assert (
            camera.center,
            camera.zoom,
            camera.angles,
            pane.viewer.dims.point,
        ) == pose
        assert np.shares_memory(pane.layers["a"][0].data, changed_data)
        assert np.shares_memory(pane.layers["b"][0].data, changed_mask)
        assert not pane.layers["a"][0].data.flags.writeable
        assert not pane.layers["b"][0].data.flags.writeable
        assert pane.layers["a"][0].opacity == 0.3
        assert pane.layers["a"][0].name == "A · After"
        assert "After" in pane.title_label.text()
    np.testing.assert_array_equal(changed_data, original_data)
    np.testing.assert_array_equal(changed_mask, original_mask)
    assert changed_data.flags.writeable and changed_mask.flags.writeable


@pytest.mark.parametrize("ndisplay", [2, 3])
@pytest.mark.parametrize("kind", ["rgb", "labels"])
def test_live_rgb_and_labels_preserve_wrappers_and_native_presentation(
    model_window, kind, ndisplay
):
    if kind == "rgb":
        old = np.zeros((3, 4, 5, 3), np.uint8)
        new = old.copy()
        new[..., 1] = 255
        old_input, new_input = _input(old, "zyx"), _input(new, "zyx")
    else:
        old = np.zeros((3, 4, 5), np.uint32)
        new = old.copy()
        new[1, 2, 3] = 765432
        old_input = _input(old, "zyx", kind="label image")
        new_input = _input(new, "zyx", kind="label image")
    window = model_window((old_input,), settings={"ndisplay": ndisplay})
    wrappers = [tuple(pane.layers["a"]) for pane in window.panes]
    window.update_inputs((new_input,))
    for pane, prior in zip(window.panes, wrappers, strict=True):
        assert tuple(pane.layers["a"]) == prior
        for layer in pane.layers["a"]:
            assert np.shares_memory(layer.data, new) and not layer.data.flags.writeable
        if kind == "labels":
            assert not pane.layers["a"][0].editable
            assert pane.layers["a"][0].data[1, 2, 3] == 765432
        elif ndisplay == 3:
            assert len(pane.layers["a"]) == 3
            assert all(layer.rendering == "mip" for layer in pane.layers["a"])
            assert np.all(pane.layers["a"][1].data == 255)
        else:
            assert pane.layers["a"][0].rgb
    assert new.flags.writeable


@pytest.mark.parametrize("change", ["count", "kind", "shape", "scale", "origin"])
def test_live_incompatible_preflight_leaves_both_panes_and_descriptors_untouched(
    model_window, change
):
    old = _input(np.zeros((3, 4, 5)), "zyx")
    window = model_window((old, old))
    prior_inputs, prior_map = window.inputs, window._inputs
    prior_layers = [
        (layer, layer.data, layer.name)
        for pane in window.panes
        for layer in pane.viewer.layers
    ]
    new = _input(np.ones((3, 4, 5)), "zyx", name="Changed")
    if change == "count":
        candidates = (new,)
    else:
        if change == "kind":
            new = _input(np.ones((3, 4, 5), bool), "zyx")
        elif change == "shape":
            new = _input(np.ones((3, 4, 6)), "zyx")
        elif change == "scale":
            new = _input(np.ones((3, 4, 5)), "zyx", scale=2)
        elif change == "origin":
            axes = tuple(replace(axis, translation=10) for axis in new.state.axes)
            new = prepare_review_input(new.data, replace(new.state, axes=axes))
        candidates = (prior_inputs[0], new)
    with pytest.raises(ValueError, match="Reopen Review Images"):
        window.update_inputs(candidates)
    assert window.inputs is prior_inputs and window._inputs is prior_map
    for layer, values, name in prior_layers:
        assert layer.data is values and layer.name == name


def test_live_mask_resource_refusal_precedes_any_layer_replacement(
    model_window, monkeypatch
):
    old_data = np.zeros((3, 4, 5))
    old_mask = np.zeros((3, 4, 5), bool)
    window = model_window((_input(old_data, "zyx"), _input(old_mask, "zyx")))
    originals = [
        (layer, layer.data) for pane in window.panes for layer in pane.viewer.layers
    ]
    inputs = window.inputs

    def refused(image):
        raise ValueError("Mask display resource denied")

    monkeypatch.setattr(image_review, "prepare_mask_review_data", refused)
    with pytest.raises(ValueError, match="resource denied"):
        window.update_inputs((_input(old_data + 1, "zyx"), _input(~old_mask, "zyx")))
    assert window.inputs is inputs
    assert all(layer.data is values for layer, values in originals)


def test_live_layer_application_failure_rolls_back_complete_pair(
    model_window, monkeypatch
):
    from napari.layers import Image

    old = np.zeros((3, 4, 5))
    window = model_window((_input(old, "zyx"), _input(old > 0, "zyx")))
    prior_inputs, prior_recipe = window.inputs, window.settings
    original = [
        (layer, layer.data, layer.name)
        for pane in window.panes
        for layer in pane.viewer.layers
    ]
    failing_layer = window.panes[1].layers["b"][0]
    setter = Image.data.fset
    failed = False

    def data_setter(layer, values):
        nonlocal failed
        setter(layer, values)
        if layer is failing_layer and not failed:
            failed = True
            raise RuntimeError("Injected replacement failure after assignment")

    monkeypatch.setattr(Image, "data", property(Image.data.fget, data_setter))
    with pytest.raises(RuntimeError, match="Injected replacement"):
        window.update_inputs((_input(old + 1, "zyx"), _input(old == 0, "zyx")))
    assert failed and window.inputs is prior_inputs and window.settings == prior_recipe
    assert window.updatesEnabled() and not window._syncing
    for layer, values, name in original:
        assert layer.data is values and layer.name == name


def test_live_data_events_only_observe_complete_new_pair_and_reset_uses_new_limits(
    model_window,
):
    old = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    new, mask = old + 100, old < 20
    changes, observed = [], []
    window = model_window(
        (_input(old, "zyx"), _input(old > 20, "zyx")), callback=changes.append
    )
    for pane in window.panes:
        for layer in pane.viewer.layers:
            layer.events.data.connect(
                lambda _event: observed.append(
                    all(
                        np.shares_memory(p.layers["a"][0].data, new)
                        and np.shares_memory(p.layers["b"][0].data, mask)
                        for p in window.panes
                    )
                )
            )
    window.update_inputs((_input(new, "zyx"), _input(mask, "zyx")))
    assert observed == [True] * 4
    assert changes == []
    window.reset_display()
    assert window.settings["a"]["contrast_limits"] == [100, 159]


def test_live_native_refresh_failure_rolls_back_all_layers_and_names(
    model_window, monkeypatch
):
    old = np.zeros((3, 4, 5))
    window = model_window((_input(old, "zyx"), _input(old > 0, "zyx")))
    old_inputs, old_limits = window.inputs, window._original_limits
    originals = [
        (layer, layer.data, layer.name)
        for pane in window.panes
        for layer in pane.viewer.layers
    ]
    target = window.panes[1].layers["a"][0]
    refresh = target.refresh
    failed = False

    def once_after_commit(*args, **kwargs):
        nonlocal failed
        if window.inputs is not old_inputs and not failed:
            failed = True
            raise RuntimeError("Native refresh failed after complete assignment")
        return refresh(*args, **kwargs)

    monkeypatch.setattr(target, "refresh", once_after_commit)
    with pytest.raises(RuntimeError, match="Native refresh failed"):
        window.update_inputs(
            (
                _input(old + 1, "zyx", name="New intensity"),
                _input(old == 0, "zyx", name="New mask"),
            )
        )
    assert failed and window.inputs is old_inputs
    assert window._original_limits is old_limits
    for layer, values, name in originals:
        assert layer.data is values and layer.name == name
    assert window._input_sections["a"].title() == "A · Image"
    assert window.left_combo.itemText(window.left_combo.findData("a")) == "A · Image"
    assert window.updatesEnabled() and not window._syncing


def test_live_rollback_failure_hard_hides_pair_and_pending_cannot_reveal_it(
    model_window, monkeypatch
):
    from napari.layers import Image

    old = np.zeros((3, 4, 5))
    window = model_window((_input(old, "zyx"), _input(old > 0, "zyx")))
    prior_inputs = window.inputs
    target = window.panes[1].layers["a"][0]
    setter = Image.data.fset
    calls = 0

    def broken_setter(layer, values):
        nonlocal calls
        setter(layer, values)
        if layer is target:
            calls += 1
            raise RuntimeError("Replacement and rollback both failed")

    monkeypatch.setattr(Image, "data", property(Image.data.fget, broken_setter))
    new = (_input(old + 1, "zyx"), _input(old == 0, "zyx"))
    with pytest.raises(ValueError, match="Reopen Review Images"):
        window.update_inputs(new)
    assert calls == 2 and window.update_failed
    assert window.inputs is prior_inputs
    assert window.pane_splitter.isHidden()
    assert window.updatesEnabled() and not window._syncing
    window.set_pending("Waiting for calculated inputs")
    assert window.pane_splitter.isHidden()
    assert "Reopen Review Images" in window.status_label.text()
    window.set_current(True)
    assert window.pane_splitter.isHidden()
    with pytest.raises(ValueError, match="Reopen Review Images"):
        window.update_inputs(new)


def test_pending_keeps_last_complete_panes_visible_without_geometry_collapse(
    model_window, qtbot
):
    data = np.zeros((3, 4, 5))
    records = []
    window = model_window((_input(data, "zyx"),), callback=records.append)
    window.show()
    qtbot.waitUntil(lambda: window.pane_splitter.height() > 0)
    geometry = window.pane_splitter.geometry()
    layers = [tuple(pane.viewer.layers) for pane in window.panes]
    recipe = window.settings
    window.set_pending("Waiting for new results; last complete display is not current.")
    QApplication.processEvents()
    assert window.pane_splitter.isVisible()
    assert window.pane_splitter.geometry() == geometry
    assert not window.sidebar_scroll.isEnabled()
    assert not window.orientation_controls.isEnabled()
    assert "not current" in window.status_label.text()
    window._set_style("a", "opacity", 0.3)
    window._set_general("mode", "overlay")
    window._choose_orientation("oblique")
    window._choose_orientation("xy")
    window.reset_display()
    assert window.settings == recipe and records == []
    window.update_inputs((_input(data + 1, "zyx"),))
    window.set_current(True)
    QApplication.processEvents()
    assert window.pane_splitter.geometry() == geometry
    assert window.sidebar_scroll.isEnabled()
    assert "Native analysis resolution" in window.status_label.text()
    assert all(
        tuple(pane.viewer.layers) == prior
        for pane, prior in zip(window.panes, layers, strict=True)
    )


@pytest.mark.parametrize("width", [900, 1280])
def test_routine_upstream_status_never_resizes_review_panes(model_window, qtbot, width):
    from napari.qt import get_stylesheet

    window = model_window((_input(np.zeros((3, 4, 5)), "zyx"),))
    window.setStyleSheet(get_stylesheet("dark", extra_variables={"font_size": "12pt"}))
    window.resize(width, 820)
    window.show()
    qtbot.waitUntil(lambda: window.pane_splitter.height() > 0)
    QApplication.processEvents()
    geometry = window.pane_splitter.geometry()
    for message in (REVIEW_STALE_MESSAGE, REVIEW_UPDATING_MESSAGE):
        window.set_pending(message)
        QApplication.processEvents()
        assert window.pane_splitter.geometry() == geometry
        assert "previous results" in window.status_label.text()
    window.set_current(True)
    QApplication.processEvents()
    assert window.pane_splitter.geometry() == geometry


def test_default_raw_and_overlay_then_independent_selections(model_window):
    a = _input(np.ones((4, 5), np.float32), name="Raw")
    b = _input(np.ones((4, 5), bool), name="Mask")
    window = model_window((a, b))
    assert window.panes[0].layers["a"][0].visible
    assert not window.panes[0].layers["b"][0].visible
    assert window.panes[1].layers["a"][0].visible
    assert window.panes[1].layers["b"][0].visible
    window._set_general("right", "b")
    assert not window.panes[1].layers["a"][0].visible
    assert window.panes[1].title_label.text() == "B · Mask"
    window._set_general("mode", "overlay")
    assert window.panes[0].isHidden()
    assert not window.panes[1].isHidden()
    assert window.right_combo.currentData() == "overlay"
    assert not window.right_combo.isEnabled()
    assert window.panes[0].layers["a"][0].visible
    assert window.panes[0].layers["b"][0].visible


def test_independent_contrast_and_optional_link(model_window):
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    window = model_window((_input(data), _input(data * 10)))
    assert not window.settings["link_contrast"]
    assert window.settings["a"]["contrast_limits"] == [0, 19]
    assert window.settings["b"]["contrast_limits"] == [0, 190]
    window._controls["a"]["black"].setValue(2)
    window._controls["a"]["white"].setValue(12)
    window._contrast_edited("a")
    assert window.settings["a"]["contrast_limits"] == [2, 12]
    assert window.settings["b"]["contrast_limits"] == [0, 190]
    window._set_general("link_contrast", True)
    assert window.settings["b"]["contrast_limits"] == [2, 12]
    assert window.settings["b"]["colormap"] == "green"
    assert window.settings["b"]["opacity"] == 0.5


def test_scalar_rendering_controls_are_kind_aware_and_only_enabled_in_3d(
    model_window,
):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    window = model_window((_input(data, "zyx"), _input(data > 20, "zyx")))
    controls = window._controls["a"]
    assert "rendering" not in window._controls["b"]
    assert controls["rendering"].currentText() == "Maximum intensity (MIP)"
    assert not controls["rendering"].isEnabled()
    assert controls["attenuation"].isHidden()
    assert controls["iso_threshold"].isHidden()
    assert "available in 3D" in controls["rendering_note"].text()
    window._set_general("ndisplay", 3)
    assert controls["rendering"].isEnabled()
    assert controls["attenuation"].isHidden()
    assert controls["iso_threshold"].isHidden()
    controls["rendering"].setCurrentIndex(
        controls["rendering"].findData("attenuated_mip")
    )
    assert not controls["attenuation"].isHidden()
    assert controls["attenuation"].isEnabled()
    assert controls["iso_threshold"].isHidden()
    controls["rendering"].setCurrentIndex(controls["rendering"].findData("iso"))
    assert controls["attenuation"].isHidden()
    assert not controls["iso_threshold"].isHidden()
    assert controls["iso_auto"].isChecked()
    window._set_general("ndisplay", 2)
    assert not controls["rendering"].isEnabled()
    assert controls["iso_threshold"].isHidden()
    assert window.settings["a"]["rendering"] == "iso"
    for pane in window.panes:
        assert pane.layers["a"][0].rendering == "mip"
    window.viewers[1].dims.ndisplay = 3
    for pane in window.panes:
        assert pane.layers["a"][0].rendering == "iso"
        assert pane.layers["b"][0].rendering == "iso"


@pytest.mark.parametrize("rendering", ["mip", "attenuated_mip", "iso"])
def test_scalar_3d_rendering_is_persisted_and_same_input_matches_both_panes(
    model_window, rendering
):
    data = np.linspace(0, 1, 60, dtype=np.float32).reshape(3, 4, 5)
    second = data * 10
    original, original_second = data.copy(), second.copy()
    changes = []
    window = model_window(
        (_input(data, "zyx"), _input(second, "zyx")),
        settings={"ndisplay": 3},
        callback=changes.append,
    )
    controls = window._controls["a"]
    controls["rendering"].setCurrentIndex(controls["rendering"].findData(rendering))
    window._set_style("a", "attenuation", 0.2)
    window._set_style("a", "iso_threshold", -3)
    assert changes[-1]["a"]["rendering"] == rendering
    assert changes[-1]["a"]["attenuation"] == 0.2
    assert changes[-1]["a"]["iso_threshold"] == -3
    assert window.settings["b"]["rendering"] == "mip"
    assert window.settings["b"]["attenuation"] == 0.05
    assert window.settings["b"]["iso_threshold"] is None
    for pane in window.panes:
        layer = pane.layers["a"][0]
        assert layer.rendering == rendering
        assert layer.attenuation == 0.2
        assert layer.iso_threshold == -3
        assert pane.layers["b"][0].rendering == "mip"
        assert np.shares_memory(layer.data, data)
        assert not layer.data.flags.writeable
    window._set_general("mode", "overlay")
    external = window.settings
    external["b"]["rendering"] = "iso"
    external["b"]["iso_threshold"] = 4
    before = len(changes)
    window.set_settings(external)
    assert len(changes) == before
    assert window.settings == external
    for pane in window.panes:
        assert pane.layers["a"][0].rendering == rendering
        assert pane.layers["b"][0].rendering == "iso"
        assert pane.layers["b"][0].iso_threshold == 4
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(second, original_second)


def test_surface_midpoint_tracks_contrast_until_explicit_source_value_is_edited(
    model_window,
):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    window = model_window(
        (_input(data, "zyx"),),
        settings={"ndisplay": 3, "a": {"rendering": "iso"}},
    )
    controls = window._controls["a"]
    assert window.settings["a"]["iso_threshold"] is None
    assert controls["iso_threshold"].value() == 29.5
    controls["black"].setValue(10)
    controls["white"].setValue(20)
    window._contrast_edited("a")
    assert controls["iso_threshold"].value() == 15
    assert all(pane.layers["a"][0].iso_threshold == 15 for pane in window.panes)
    controls["iso_threshold"].setValue(-2.5)
    controls["iso_threshold"].editingFinished.emit()
    assert window.settings["a"]["iso_threshold"] == -2.5
    assert not controls["iso_auto"].isChecked()
    controls["white"].setValue(30)
    window._contrast_edited("a")
    assert all(pane.layers["a"][0].iso_threshold == -2.5 for pane in window.panes)
    controls["iso_auto"].setChecked(True)
    assert window.settings["a"]["iso_threshold"] is None
    assert controls["iso_threshold"].value() == 20
    assert all(pane.layers["a"][0].iso_threshold == 20 for pane in window.panes)


@pytest.mark.parametrize(
    "limits,expected", [((-1.7e308, 1.7e308), 0.0), ((1.6e308, 1.7e308), 1.65e308)]
)
def test_surface_auto_midpoint_avoids_valid_finite_range_overflow(limits, expected):
    midpoint = ImageReviewWindow._surface_level(
        {"iso_threshold": None, "contrast_limits": limits}
    )
    assert np.isfinite(midpoint)
    assert limits[0] <= midpoint <= limits[1]
    assert midpoint == pytest.approx(expected, rel=np.finfo(float).eps)


def test_depth_weight_entry_accepts_nonnegative_values_outside_default_window(
    model_window,
):
    window = model_window(
        (_input(np.ones((3, 4, 5)), "zyx"),),
        settings={"ndisplay": 3, "a": {"rendering": "attenuated_mip"}},
    )
    entry = window._controls["a"]["attenuation"]
    assert entry.minimum() == 0
    assert entry.maximum() == image_review._DOUBLE_LIMIT
    entry.setValue(150)
    entry.editingFinished.emit()
    assert window.settings["a"]["attenuation"] == 150
    assert all(pane.layers["a"][0].attenuation == 150 for pane in window.panes)
    entry.setValue(0)
    entry.editingFinished.emit()
    assert window.settings["a"]["attenuation"] == 0


@pytest.mark.parametrize("rendering", ["attenuated_mip", "iso"])
def test_shaded_volume_warns_and_lock_still_only_protects_base_scale(
    model_window, rendering
):
    data = np.linspace(0, 1, 60, dtype=np.float32).reshape(3, 4, 5)
    window = model_window(
        (_input(data, "zyx"), _input(data, "zyx")),
        settings={
            "ndisplay": 3,
            "a": {"contrast_limits": [0, 1], "lock_contrast": True},
        },
    )
    window._set_style("a", "rendering", rendering)
    controls = window._controls["a"]
    assert controls["scale_label"].text() == "Base colour scale (before shading)"
    assert "not quantitative intensity" in controls["rendering_note"].text()
    assert "no physical inter-layer occlusion" in controls["rendering_note"].text()
    assert not controls["black"].isEnabled()
    assert not controls["white"].isEnabled()
    assert window.settings["a"]["contrast_limits"] == [0, 1]
    assert controls["scale"]._limits == (0, 1)
    if rendering == "iso":
        assert "display-only cutoff hides internal details" in (
            controls["rendering_note"].text()
        )
        assert controls["iso_threshold"].isEnabled()
    window._set_general("ndisplay", 2)
    assert controls["scale_label"].text() == "Fixed colour scale"
    assert window.settings["a"]["rendering"] == rendering


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("font_size", [10, 12])
def test_rendering_explanation_wraps_to_full_height_in_themed_narrow_sidebar(
    model_window, qtbot, theme, font_size
):
    from napari.qt import get_stylesheet

    data = np.linspace(0, 1, 60, dtype=np.float32).reshape(3, 4, 5)
    window = model_window(
        (_input(data, "zyx"), _input(data > 0.5, "zyx")),
        settings={"ndisplay": 3},
    )
    window.setStyleSheet(
        get_stylesheet(theme, extra_variables={"font_size": f"{font_size}pt"})
    )
    window.resize(900, 760)
    window.show()
    note = window._controls["a"]["rendering_note"]
    qtbot.waitUntil(lambda: note.width() > 0)
    for rendering in ("attenuated_mip", "iso", "mip", "attenuated_mip"):
        window._set_style("a", "rendering", rendering)
        assert note.wordWrap()
        assert note.sizePolicy().hasHeightForWidth()
        assert note.hasHeightForWidth()
        qtbot.waitUntil(lambda: note.height() >= note.heightForWidth(note.width()))
        assert note.heightForWidth(note.width()) > note.fontMetrics().height()
        assert note.height() >= note.heightForWidth(note.width())
        content = window.sidebar_scroll.widget()
        assert content.width() == window.sidebar_scroll.viewport().width()
        assert window.sidebar_scroll.horizontalScrollBar().maximum() == 0
    window._set_style("a", "rendering", "iso")
    qtbot.waitUntil(lambda: note.height() >= note.heightForWidth(note.width()))
    narrow_width, narrow_height = note.width(), note.height()
    window.main_splitter.setSizes([350, 520])
    qtbot.waitUntil(lambda: note.width() > narrow_width)
    # QFormLayout can shrink the row before the queued height-fit callback
    # restores its exact bounds. Wait for that callback's completed contract,
    # not the intermediate geometry (and retain all exact assertions below).
    qtbot.waitUntil(
        lambda: (
            note.height() < narrow_height
            and note.minimumHeight() == note.maximumHeight() == note.height()
            and note.height() >= note.heightForWidth(note.width())
        )
    )
    assert note.height() >= note.heightForWidth(note.width())
    assert note.minimumHeight() == note.maximumHeight() == note.height()
    window._set_style("a", "rendering", "attenuated_mip")
    qtbot.waitUntil(
        lambda: (
            note.minimumHeight() == note.maximumHeight() == note.height()
            and note.height() >= note.heightForWidth(note.width())
        )
    )
    before = note.geometry()
    # Unrelated appearance edits retain the same explanatory text and height;
    # they must not briefly clear its constraints and move the inspector.
    for field, value in (("opacity", 0.5), ("attenuation", 0.1), ("attenuation", 0.2)):
        window._set_style("a", field, value)
        assert note.geometry() == before
        assert note.minimumHeight() == note.maximumHeight() == before.height()
        QApplication.processEvents()
        assert note.geometry() == before
        assert note.minimumHeight() == note.maximumHeight() == before.height()


def test_reset_display_restores_mip_rendering_without_mutating_source(model_window):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    original = data.copy()
    window = model_window(
        (_input(data, "zyx"),),
        settings={
            "ndisplay": 3,
            "a": {"rendering": "iso", "attenuation": 0.7, "iso_threshold": 5},
        },
    )
    window.reset_display()
    assert window.settings["a"]["rendering"] == "mip"
    assert window.settings["a"]["attenuation"] == 0.05
    assert window.settings["a"]["iso_threshold"] is None
    window.viewers[0].dims.ndisplay = 3
    for pane in window.panes:
        layer = pane.layers["a"][0]
        assert layer.rendering == "mip"
        assert np.shares_memory(layer.data, data)
        assert not layer.data.flags.writeable
    np.testing.assert_array_equal(data, original)


def test_single_input_has_unambiguous_pane_selector(model_window):
    window = model_window((_input(np.ones((4, 5), np.float32)),))
    assert window.left_combo.currentData() == "a"
    assert window.right_combo.currentData() == "overlay"
    assert "only connected input" in window.right_combo.currentText()
    item = window.right_combo.model().item(window.right_combo.findData("b"))
    assert not item.isEnabled()


def test_approved_inspector_sidebar_is_on_the_right(model_window):
    window = model_window((_input(np.zeros((4, 5), np.float32)),))
    assert window.main_splitter.widget(1) is window.sidebar_scroll
    assert window.main_splitter.widget(0).findChildren(QWidget)


@pytest.mark.parametrize("theme", ("dark", "light"))
@pytest.mark.parametrize("width", (900, 1280))
def test_readonly_status_shares_the_orientation_row(model_window, qtbot, theme, width):
    from napari.qt import get_stylesheet

    data = np.zeros((3, 4, 5), np.float32)
    window = model_window((_input(data, "zyx"),), settings={"ndisplay": 3})
    window.setStyleSheet(get_stylesheet(theme, extra_variables={"font_size": "12pt"}))
    window.resize(width, 760)
    window.show()
    qtbot.waitUntil(lambda: window.status_label.width() > 0)
    row = window.status_label.parentWidget()
    assert row is window.orientation_controls.parentWidget()
    assert window.status_label.text() == "Read-only · Native analysis resolution"
    assert row.rect().contains(window.status_label.geometry())
    assert row.rect().contains(window.orientation_controls.geometry())
    assert window.status_label.geometry().right() < (
        window.orientation_controls.geometry().left()
    )
    assert (
        abs(
            window.status_label.geometry().center().y()
            - window.orientation_controls.geometry().center().y()
        )
        <= 1
    )
    assert window.status_label.heightForWidth(window.status_label.width()) <= (
        window.status_label.height()
    )
    assert window.status_label.height() <= window.orientation_controls.height()
    assert window.navigation_bar.isHidden()
    for button in (*window.orientation_buttons.values(), window.fit_button):
        assert button.isVisible()
        assert window.orientation_controls.rect().contains(button.geometry())
    assert window.width() == width


@pytest.mark.parametrize("theme", (None, "dark", "light"))
def test_long_footer_warning_wraps_and_stale_view_buttons_stay_disabled(
    model_window,
    qtbot,
    theme,
):
    from napari.qt import get_stylesheet

    changes = []
    data = np.zeros((3, 4, 5), np.float32)
    original = data.copy()
    window = model_window((_input(data, "zyx"),), callback=changes.append)
    if theme is not None:
        window.setStyleSheet(
            get_stylesheet(theme, extra_variables={"font_size": "12pt"})
        )
    assert window.status_label.sizePolicy().hasHeightForWidth()
    window.resize(900, 760)
    message = (
        "Source changed. Recalculate the connected upstream images before opening "
        "Review Images again. This snapshot is not the current calculated output."
    )
    window.set_current(False, message)
    window.show()
    qtbot.waitUntil(lambda: window.status_label.width() > 0)
    cameras = [(viewer_camera(v).center, viewer_camera(v).zoom) for v in window.viewers]
    settings = window.settings
    for button in (*window.orientation_buttons.values(), window.fit_button):
        assert not button.isEnabled()
        button.click()
    assert window.status_label.isEnabled()
    assert window.status_label.text() == message
    assert window.status_label.heightForWidth(window.status_label.width()) <= (
        window.status_label.height()
    )
    assert not window.status_label.geometry().intersects(
        window.orientation_controls.geometry()
    )
    narrow_height = window.bottom_toolbar.height()
    window.resize(1280, 760)
    qtbot.waitUntil(lambda: window.width() == 1280)
    QApplication.processEvents()
    assert window.bottom_toolbar.height() < narrow_height
    assert window.status_label.heightForWidth(window.status_label.width()) <= (
        window.status_label.height()
    )
    assert window.settings == settings
    assert changes == []
    assert cameras == [
        (viewer_camera(v).center, viewer_camera(v).zoom) for v in window.viewers
    ]
    np.testing.assert_array_equal(data, original)
    window.set_current(True)
    assert window.orientation_controls.isEnabled()
    assert window.fit_button.isEnabled()


@pytest.mark.parametrize("time_series", (False, True))
def test_slider_bar_visibility_follows_hidden_dimensions(model_window, time_series):
    data = np.zeros((4, 3, 4, 5) if time_series else (3, 4, 5), np.float32)
    window = model_window((_input(data, "tzyx" if time_series else "zyx"),))
    assert not window.navigation_bar.isHidden()
    assert not window._navigation_controls["slice"]["group"].isHidden()
    window._set_general("ndisplay", 3)
    assert window._navigation_controls["slice"]["group"].isHidden()
    assert window.navigation_bar.isHidden() == (not time_series)
    if time_series:
        assert not window._navigation_controls["t"]["group"].isHidden()
    window._set_general("ndisplay", 2)
    assert not window.navigation_bar.isHidden()
    assert not window._navigation_controls["slice"]["group"].isHidden()


@pytest.mark.parametrize("width", [900, 1280])
def test_mockup_layout_owns_top_pane_bottom_and_right_controls(
    model_window, qtbot, width
):
    data = np.zeros((3, 4, 5, 6), np.float32)
    window = model_window((_input(data, "tzyx"), _input(data, "tzyx")))
    window.resize(width, 760)
    window.show()
    qtbot.waitUntil(lambda: window.navigation_bar.width() > 0)
    assert window.width() == width
    for widget in (
        *window.mode_buttons.values(),
        window.dim_combo,
        window.link_navigation_check,
        window.axes_check,
        window.scale_bar_check,
    ):
        assert window.top_toolbar.isAncestorOf(widget)
    assert window.link_navigation_check.text() == "Link viewpoint"
    assert window.orientation_controls.isAncestorOf(window.fit_button)
    assert not window.top_toolbar.isAncestorOf(window.fit_button)
    assert window.bottom_toolbar.isAncestorOf(window.status_label)
    assert window.bottom_toolbar.isAncestorOf(window.orientation_controls)
    for pane, combo in zip(
        window.panes, (window.left_combo, window.right_combo), strict=True
    ):
        assert isinstance(pane, QFrame)
        assert pane.objectName() == "ReviewPaneCard"
        assert pane.header.isAncestorOf(combo)
        assert pane.footer.isAncestorOf(pane.title_label)
        assert pane.footer.isAncestorOf(pane.position_label)
        assert (
            pane.header.mapTo(window, QPoint()).y()
            < pane.qt_viewer.mapTo(window, QPoint()).y()
        )
        assert pane.qt_viewer.geometry().bottom() < pane.footer.geometry().top()
        for part in (pane.header, pane.qt_viewer, pane.footer):
            assert pane.contentsRect().contains(part.geometry())
    sidebar = window.sidebar_scroll.widget()
    for controls in window._controls.values():
        assert sidebar.isAncestorOf(controls["black"])
        assert sidebar.isAncestorOf(controls["black_slider"])
    assert window.viewer_container.isAncestorOf(window.navigation_bar)
    assert window.navigation_bar.mapTo(window, QPoint()).y() >= (
        window.pane_splitter.mapTo(window, QPoint()).y() + window.pane_splitter.height()
    )
    assert window.top_toolbar.mapTo(window, QPoint()).y() < (
        window.viewer_container.mapTo(window, QPoint()).y()
    )
    orientation_rect = window.orientation_controls.geometry()
    assert window.bottom_toolbar.rect().contains(orientation_rect)
    assert window.status_label.geometry().right() < orientation_rect.left()
    for controls in window._navigation_controls.values():
        assert controls["total"].isVisible()
        assert controls["total"].geometry().right() < controls["group"].width()


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width", [900, 1280])
def test_pane_cards_contain_header_canvas_footer_under_native_theme(
    model_window, qtbot, theme, width
):
    from napari.qt import get_stylesheet

    data = np.zeros((4, 3, 4, 5), np.float32)
    window = model_window(
        (
            _input(data, "tzyx", name="A deliberately long primary source title"),
            _input(data, "tzyx", name="A deliberately long secondary source title"),
        )
    )
    window.setStyleSheet(get_stylesheet(theme, extra_variables={"font_size": "12pt"}))
    window.resize(width, 760)
    window.show()
    qtbot.waitUntil(lambda: window.panes[0].footer.width() > 0)
    assert window.width() == width
    assert window.scale_bar_check.geometry().right() < window.top_toolbar.width()
    for pane, combo in zip(
        window.panes, (window.left_combo, window.right_combo), strict=True
    ):
        assert "border: 1px solid" in pane.styleSheet()
        for part in (pane.header, pane.qt_viewer, pane.footer):
            assert pane.contentsRect().contains(part.geometry())
        assert pane.header.rect().contains(combo.geometry())
        assert pane.footer.rect().contains(pane.title_label.geometry())
        assert pane.footer.rect().contains(pane.position_label.geometry())
        assert (
            pane.position_label.geometry().left() > pane.title_label.geometry().right()
        )
        assert pane.title_label.toolTip()
    # A global native stylesheet change must also restyle the one-piece cards.
    prior = [pane.styleSheet() for pane in window.panes]
    other_theme = "light" if theme == "dark" else "dark"
    window.setStyleSheet(
        get_stylesheet(other_theme, extra_variables={"font_size": "12pt"})
    )
    QApplication.processEvents()
    assert all(
        pane.styleSheet() != old for pane, old in zip(window.panes, prior, strict=True)
    )
    assert all(
        np.shares_memory(pane.layers["a"][0].data, data) for pane in window.panes
    )


def test_pane_footer_positions_are_truthful_shared_and_read_only(model_window):
    data = np.zeros((4, 3, 4, 5), np.float32)
    original = data.copy()
    image = _input(data, "tzyx")
    window = model_window((image,))
    window._set_general("link_navigation", False)
    window._navigation_controls["t"]["entry"].setValue(4)
    window._navigation_controls["slice"]["entry"].setValue(2)
    for pane in window.panes:
        assert pane.position_label.text() == "Z 2 / 3 · T 4 / 4"
        assert "Z:" in pane.position_label.toolTip()
        assert "T:" in pane.position_label.toolTip()
    window._set_general("ndisplay", 3)
    assert all(pane.position_label.text() == "T 4 / 4" for pane in window.panes)
    static = model_window((_input(data[0], "zyx"),), settings={"ndisplay": 3})
    assert all(pane.position_label.isHidden() for pane in static.panes)
    assert all(not pane.position_label.text() for pane in static.panes)
    np.testing.assert_array_equal(data, original)
    assert data.flags.writeable and not image.data.flags.writeable


def test_only_one_shared_dimension_bar_survives_native_show(model_window, qtbot):
    window = model_window((_input(np.zeros((3, 4, 5)), "zyx"),))
    window.show()
    for pane in window.panes:
        pane.qt_viewer.dims.show()
        assert pane.qt_viewer.dims.isHidden()
    assert not window._navigation_controls["slice"]["group"].isHidden()
    assert window._navigation_controls["t"]["group"].isHidden()


def test_level_sliders_remain_sensible_while_numeric_entry_can_exceed_window(
    model_window,
):
    values = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    window = model_window((_input(values),))
    controls = window._controls["a"]
    slider = controls["white_slider"]
    assert (slider.minimum(), slider.maximum()) == (0, 1000)
    controls["white"].setValue(50)
    window._contrast_edited("a")
    assert window.settings["a"]["contrast_limits"] == [0, 50]
    assert slider.value() == 1000
    assert (slider.minimum(), slider.maximum()) == (0, 1000)
    slider.setValue(250)
    assert window.settings["a"]["contrast_limits"] == [0, 0.25]
    window._set_style("a", "lock_contrast", True)
    assert not controls["black_slider"].isEnabled()
    assert not controls["white_slider"].isEnabled()
    window._set_style("a", "lock_contrast", False)
    assert controls["black_slider"].isEnabled()
    assert controls["white_slider"].isEnabled()


@pytest.mark.parametrize(
    "ndisplay,rendering", [(2, "mip"), (3, "attenuated_mip"), (3, "iso")]
)
def test_finite_numeric_ranges_do_not_force_horizontal_sidebar_clipping(
    model_window, qtbot, ndisplay, rendering
):
    rgb = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    index = np.linspace(0, 1, 60, dtype=np.float32).reshape(3, 4, 5)
    window = model_window(
        (
            _input(rgb, "zyx", name="VIPP review 3D RGB composite"),
            _input(index, "zyx", name="VIPP review 3D synthetic index"),
        ),
        settings={"ndisplay": ndisplay, "b": {"rendering": rendering}},
    )
    window.show()
    qtbot.waitUntil(lambda: window.sidebar_scroll.viewport().width() > 0)
    content = window.sidebar_scroll.widget()
    assert content.width() == window.sidebar_scroll.viewport().width()
    assert window.sidebar_scroll.horizontalScrollBar().maximum() == 0
    for controls in window._controls.values():
        for widget in controls.values():
            if not widget.isVisible():
                continue
            left = widget.mapTo(content, QPoint()).x()
            assert 0 <= left and left + widget.width() <= content.width()


@pytest.mark.parametrize("limits", [(1.907e-18, 1.0), (-1e20, 1e20)])
def test_numeric_controls_and_legend_retain_authored_limits(model_window, limits):
    settings = default_review_settings()
    settings["a"]["contrast_limits"] = list(limits)
    window = model_window((_input(np.zeros((4, 5), np.float32)),), settings=settings)
    controls = window._controls["a"]
    assert controls["black"].value() == limits[0]
    assert controls["white"].value() == limits[1]
    assert controls["scale"]._limits == limits
    assert len(controls["black"].text()) < 25
    window._contrast_edited("a")
    assert window.settings["a"]["contrast_limits"] == list(limits)


def test_numeric_entry_retains_large_finite_double_without_render_claim(model_window):
    window = model_window((_input(np.zeros((4, 5), np.float32)),))
    spin = window._numeric(1e150)
    assert spin.value() == 1e150
    assert len(spin.text()) < 25


def test_locked_color_scale_prevents_edits_and_linking_but_not_hide_below(model_window):
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    window = model_window((_input(data), _input(data / 19)))
    window._set_style("b", "lock_contrast", True)
    assert not window._controls["b"]["black"].isEnabled()
    assert not window._controls["b"]["white"].isEnabled()
    window._set_general("link_contrast", True)
    assert window.settings["b"]["contrast_limits"] == [0, 1]
    window._controls["a"]["white"].setValue(10)
    window._contrast_edited("a")
    assert window.settings["b"]["contrast_limits"] == [0, 1]
    window._controls["b"]["white"].setValue(10)
    window._contrast_edited("b")
    assert window.settings["b"]["contrast_limits"] == [0, 1]
    window._set_style("b", "threshold", 0.2)
    assert window._controls["b"]["scale"]._limits == (0, 1)
    assert window.settings["b"]["threshold"] == 0.2
    window._set_style("b", "lock_contrast", False)
    assert window._controls["b"]["white"].isEnabled()
    window._controls["b"]["white"].setValue(2)
    window._contrast_edited("b")
    assert window.settings["a"]["contrast_limits"] == [0, 2]


def test_invalid_contrast_keeps_previous_view_without_publishing(model_window):
    records = []
    window = model_window(
        (_input(np.arange(20).reshape(4, 5)),), callback=records.append
    )
    original = window.settings
    window._controls["a"]["black"].setValue(100)
    window._contrast_edited("a")
    assert window.settings == original
    assert "previous display is retained" in window.status_label.text()
    assert records == []


@pytest.mark.parametrize("threshold", [-1.0, 0.2, 0.8, 1.5])
def test_visibility_cutoff_preserves_colors_and_hides_below(threshold):
    colormap, limits = visibility_colormap("viridis", (0, 1), threshold)
    values = np.array([threshold - 0.01, threshold + 0.01, 0.9, 1.6])
    displayed = colormap.map((values - limits[0]) / (limits[1] - limits[0]))
    expected = AVAILABLE_COLORMAPS["viridis"].map(values)
    np.testing.assert_allclose(displayed[:, :3], expected[:, :3], atol=1e-6)
    assert np.all(displayed[values < threshold, 3] == 0)
    assert np.all(displayed[values >= threshold, 3] == 1)


def test_threshold_preserves_data_and_fixed_scale(model_window):
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    original = values.copy()
    window = model_window((_input(values),))
    window._set_style("a", "threshold", 5)
    assert window.settings["a"]["contrast_limits"] == [0, 19]
    assert window._controls["a"]["scale"]._limits == (0, 19)
    assert np.shares_memory(window.panes[0].layers["a"][0].data, values)
    np.testing.assert_array_equal(values, original)


def test_mask_kind_aware_controls_and_transparent_background(model_window):
    window = model_window(
        (_input(np.ones((4, 5), np.float32)), _input(np.eye(4, 5, dtype=bool)))
    )
    assert "black" not in window._controls["b"]
    assert "mask_color" in window._controls["b"]
    assert not window.link_contrast_check.isEnabled()
    mask = window.panes[0].layers["b"][0]
    assert mask.data.dtype == bool
    assert mask.colormap.map([0])[0, 3] == 0
    assert mask.colormap.map([1])[0, 3] == 1
    window._set_style("b", "mask_color", "#FF0000")
    np.testing.assert_allclose(mask.colormap.map([1])[0], [1, 0, 0, 1])


def test_volume_mask_prepared_once_before_viewers_and_reused_across_panes_modes(
    model_window, monkeypatch
):
    data = np.zeros((3, 4, 5), dtype=bool)
    data[1, 1:3, 1:4] = True
    original = data.copy()
    events = []
    real_prepare = image_review.prepare_mask_review_data

    def prepare(image):
        events.append("prepare")
        return real_prepare(image)

    class RecordingViewer(_ModelOnlyQtViewer):
        def __init__(self, *args, **kwargs):
            events.append("viewer")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(image_review, "prepare_mask_review_data", prepare)
    monkeypatch.setattr(image_review, "QtViewer", RecordingViewer)
    window = model_window((_input(data, "zyx"),))
    assert events == ["prepare", "viewer", "viewer"]
    occupancy = window._mask_display_data["a"]
    assert occupancy.dtype == np.uint8 and not occupancy.flags.writeable
    assert np.shares_memory(occupancy, data)
    for _ in range(2):
        window._set_general("ndisplay", 3)
        for pane in window.panes:
            assert pane.layers["a"][0].data is occupancy
        window._set_general("ndisplay", 2)
        for pane in window.panes:
            assert pane.layers["a"][0].data.dtype == bool
            assert np.shares_memory(pane.layers["a"][0].data, data)
    assert events.count("prepare") == 1
    np.testing.assert_array_equal(data, original)
    assert data.flags.writeable
    window.shutdown()
    assert window._mask_display_data == {}


def test_mask_allocation_refusal_precedes_any_qt_viewer_construction(
    model_window, monkeypatch
):
    created_viewers = []

    def denied(image):
        raise ValueError("Not enough memory for this mask's display conversion.")

    def unexpected_viewer(*args, **kwargs):
        created_viewers.append(args)
        raise AssertionError("A viewer must not exist before mask preparation succeeds")

    monkeypatch.setattr(image_review, "prepare_mask_review_data", denied)
    monkeypatch.setattr(image_review, "QtViewer", unexpected_viewer)
    data = np.zeros((3, 4, 5), dtype=bool)
    with pytest.raises(ValueError, match="Not enough memory"):
        model_window((_input(data, "zyx"),))
    assert created_viewers == []
    assert data.flags.writeable and not np.any(data)


def test_inherently_2d_mask_does_not_prepare_volume_occupancy(
    model_window, monkeypatch
):
    def unwanted_prepare(image):
        raise AssertionError("An inherently 2D mask does not need a volume buffer")

    monkeypatch.setattr(image_review, "prepare_mask_review_data", unwanted_prepare)
    data = np.eye(4, 5, dtype=bool)
    window = model_window((_input(data),))
    assert window._mask_display_data == {}
    assert np.shares_memory(window.panes[0].layers["a"][0].data, data)


def test_labels_noneditable_identity_preserved(model_window):
    data = np.zeros((4, 5), np.uint32)
    data[1, 2] = 765432
    window = model_window((_input(data, kind="label image"),))
    for pane in window.panes:
        labels = pane.layers["a"][0]
        assert not labels.editable
        assert str(labels.mode) == "pan_zoom"
        assert np.shares_memory(labels.data, data)
        assert labels.data[1, 2] == 765432
    assert "black" not in window._controls["a"]


def test_camera_unlinks_but_time_and_spatial_positions_remain_shared(model_window):
    data = np.zeros((3, 4, 5, 6), np.float32)
    window = model_window((_input(data, "tzyx"), _input(data, "tzyx")))
    left, right = window.viewers
    left.dims.set_point(0, 2)
    left.dims.set_point(1, 3)
    assert right.dims.point[:2] == left.dims.point[:2] == (2, 3)
    left.dims.order = (0, 2, 1, 3)
    assert right.dims.order == left.dims.order
    camera = viewer_camera(left)
    camera.center = (2, 3, 4)
    camera.zoom = 2.5
    assert viewer_camera(right).center == camera.center
    assert viewer_camera(right).zoom == 2.5
    window._set_general("link_navigation", False)
    camera.zoom = 9
    left.dims.set_point(0, 0)
    assert viewer_camera(right).zoom == 2.5
    assert right.dims.point[0] == 0


@pytest.mark.parametrize(
    ("orientation", "order", "slice_axis"),
    [
        ("xy", (0, 1, 2, 3), "z"),
        ("xz", (0, 2, 1, 3), "y"),
        ("yz", (0, 3, 1, 2), "x"),
    ],
)
def test_orthogonal_2d_planes_use_semantic_axes_not_array_guessing(
    model_window, orientation, order, slice_axis
):
    data = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
    original = data.copy()
    window = model_window((_input(data, "tzyx"),))
    window._choose_orientation(orientation)
    assert window.settings["orientation"] == orientation
    for pane in window.panes:
        assert pane.viewer.dims.order == order
        assert np.shares_memory(pane.layers["a"][0].data, data)
    controls = window._navigation_controls["slice"]
    assert controls["label"].text() == slice_axis.upper()
    controls["slider"].setValue(1)
    axis = controls["axis"]
    assert all(viewer.dims.point[axis] == 1 for viewer in window.viewers)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    ("orientation", "direction"),
    [
        ("xy", (-1, 0, 0)),
        ("xz", (0, -1, 0)),
        ("yz", (0, 0, -1)),
        ("oblique", (-1, -1, 1)),
    ],
)
def test_volume_presets_use_public_camera_direction_and_keep_axis_semantics(
    model_window, orientation, direction
):
    settings = default_review_settings()
    settings.update(ndisplay=3, orientation=orientation)
    data = np.zeros((3, 4, 5, 6), np.float32)
    window = model_window((_input(data, "tzyx"),), settings=settings)
    expected = np.asarray(direction, dtype=float)
    expected /= np.linalg.norm(expected)
    for viewer in window.viewers:
        assert viewer.dims.order == (0, 1, 2, 3)
        assert viewer.dims.displayed == (1, 2, 3)
        np.testing.assert_allclose(
            viewer_camera(viewer).view_direction, expected, atol=1e-12
        )
    assert window._navigation_controls["slice"]["group"].isHidden()
    assert not window._navigation_controls["t"]["group"].isHidden()


@pytest.mark.parametrize("orientation", ["xy", "xz", "yz", "oblique"])
@pytest.mark.parametrize("route", ["toolbar", "native"])
def test_switching_volume_to_slice_starts_in_xy_with_shared_z(
    model_window, orientation, route
):
    data = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
    original = data.copy()
    records = []
    window = model_window(
        (_input(data, "tzyx"),),
        settings={"ndisplay": 3, "orientation": orientation},
        callback=records.append,
    )
    window._navigate_axis("t", 2)
    for viewer in window.viewers:
        viewer.dims.set_point(1, 1)
    before = window.viewers[0].dims.point
    records.clear()
    if route == "toolbar":
        window.dim_combo.setCurrentIndex(window.dim_combo.findData(2))
    else:
        window.viewers[1].dims.ndisplay = 2
    assert window.settings["ndisplay"] == 2
    assert window.settings["orientation"] == "xy"
    assert records[-1]["orientation"] == "xy"
    assert window.orientation_buttons["xy"].isChecked()
    for viewer in window.viewers:
        assert viewer.dims.order == (0, 1, 2, 3)
        assert viewer.dims.displayed == (2, 3)
        assert viewer.dims.point[0] == before[0]
        assert viewer.dims.point == window.viewers[0].dims.point
        if route == "toolbar":
            assert viewer.dims.point == before
        # Native napari centers a newly undisplayed spatial axis before its
        # ndisplay notification; both panes must share that resulting position.
    controls = window._navigation_controls["slice"]
    assert controls["axis"] == 1
    assert controls["label"].text() == "Z"
    controls["entry"].setValue(4)
    assert all(viewer.dims.point[1] == 3 for viewer in window.viewers)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("orientation,axis", [("xz", "Y"), ("yz", "X")])
def test_saved_deliberate_2d_plane_is_not_replaced_by_default_xy(
    model_window, orientation, axis
):
    window = model_window(
        (_input(np.zeros((4, 5, 6), np.float32), "zyx"),),
        settings={"ndisplay": 2, "orientation": orientation},
    )
    assert window.settings["orientation"] == orientation
    assert window._navigation_controls["slice"]["label"].text() == axis
    window._set_general("ndisplay", 2)
    assert window.settings["orientation"] == orientation
    recipe = window.settings
    recipe["orientation"] = "yz" if orientation == "xz" else "xz"
    window.set_settings(recipe)
    assert window.settings == recipe
    assert window._navigation_controls["slice"]["label"].text() == (
        "X" if orientation == "xz" else "Y"
    )


def test_navigation_total_is_outside_editable_position(model_window, qtbot):
    window = model_window((_input(np.zeros((3, 4, 5, 6)), "tzyx"),))
    window.show()
    qtbot.waitUntil(lambda: window.navigation_bar.width() > 0)
    for key, count in (("t", 3), ("slice", 4)):
        controls = window._navigation_controls[key]
        entry = controls["entry"]
        total = controls["total"]
        assert entry.suffix() == ""
        assert total.text() == f"/ {count}"
        assert total.parentWidget() is controls["group"]
        assert not entry.isAncestorOf(total)
        row = controls["group"].layout()
        assert row.indexOf(total) == row.indexOf(entry) + 1
        for index in (0, count - 1):
            controls["slider"].setValue(index)
            assert entry.lineEdit().text() == str(index + 1)
            assert total.text() == f"/ {count}"
        entry.lineEdit().selectAll()
        qtbot.keyClicks(entry.lineEdit(), "2")
        qtbot.keyPress(entry.lineEdit(), image_review.Qt.Key_Return)
        assert controls["slider"].value() == 1
        assert total.text() == f"/ {count}"
        assert all(
            viewer.dims.current_step[controls["axis"]] == 1 for viewer in window.viewers
        )


def test_calibrated_shared_positions_are_runtime_only_even_when_camera_unlinked(
    model_window,
):
    data = np.zeros((3, 4, 5, 6), np.float32)
    axes = (
        AxisMetadata("t", "time", unit="millisecond", scale=500, translation=1000),
        AxisMetadata("z", "space", unit="nanometer", scale=2000, translation=3000),
        AxisMetadata("y", "space", unit="micrometer", scale=0.5, translation=5),
        AxisMetadata("x", "space", unit="micrometer", scale=0.4, translation=7),
    )
    state = image_state_from_array(data, axes=axes)
    original_axes = state.axes
    descriptor = prepare_review_input(data, state, "Calibrated")
    records = []
    window = model_window((descriptor,), callback=records.append)
    window._set_general("link_navigation", False)
    previous = window.settings
    records.clear()
    window._navigation_controls["t"]["entry"].setValue(3)
    window._navigation_controls["slice"]["slider"].setValue(2)
    for viewer in window.viewers:
        np.testing.assert_allclose(viewer.dims.point[:2], (2, 7))
    assert window._navigation_controls["t"]["caption"].text() == "2 s"
    assert window._navigation_controls["slice"]["caption"].text() == "7 µm"
    assert window.settings == previous
    assert records == []
    assert state.axes == original_axes
    assert data.flags.writeable and not descriptor.data.flags.writeable


def test_2d_inputs_disable_unavailable_volume_presets(model_window):
    window = model_window((_input(np.zeros((4, 5))),))
    assert window.orientation_buttons["xy"].isEnabled()
    assert window.orientation_buttons["xy"].isChecked()
    assert not window.orientation_buttons["xz"].isEnabled()
    assert not window.orientation_buttons["yz"].isEnabled()
    assert not window.orientation_buttons["oblique"].isEnabled()
    assert all(
        controls["group"].isHidden()
        for controls in window._navigation_controls.values()
    )


def test_appearance_and_arrangement_changes_preserve_dragged_camera_and_slice(
    model_window,
):
    settings = default_review_settings()
    settings.update(ndisplay=3, orientation="xz")
    window = model_window((_input(np.zeros((3, 4, 5)), "zyx"),), settings=settings)
    camera = viewer_camera(window.viewers[0])
    camera.angles = (27, 31, 19)
    camera.center = (1, 2, 3)
    camera.zoom = 2
    pose = (camera.angles, camera.center, camera.zoom)
    window._set_style("a", "opacity", 0.2)
    window._set_style("a", "threshold", 0.1)
    window._set_general("mode", "overlay")
    assert (camera.angles, camera.center, camera.zoom) == pose
    assert window.settings["orientation"] == "xz"
    external = window.settings
    external["a"]["colormap"] = "viridis"
    window.set_settings(external)
    assert (camera.angles, camera.center, camera.zoom) == pose
    # A clicked preset deliberately restores orientation; a style edit does not.
    window._choose_orientation("xz")
    np.testing.assert_allclose(camera.view_direction, (0, -1, 0), atol=1e-12)


def test_owned_native_overlays_use_calibration_without_changing_metadata(model_window):
    image = _input(np.zeros((3, 4, 5)), "zyx", unit="micrometer", scale=0.5)
    state = image.state
    settings = default_review_settings()
    settings["ndisplay"] = 3
    window = model_window((image,), settings=settings)
    for viewer in window.viewers:
        overlays = getattr(getattr(viewer, "canvas", None), "overlays", viewer)
        assert overlays.axes.visible
        assert overlays.scale_bar.visible
        if hasattr(viewer.layers[0], "units"):
            assert all(str(unit) == "micrometer" for unit in viewer.layers[0].units)
    assert image.state == state
    window._set_general("ndisplay", 2)
    for viewer in window.viewers:
        overlays = getattr(getattr(viewer, "canvas", None), "overlays", viewer)
        assert not overlays.axes.visible
        assert overlays.scale_bar.visible


def test_global_axes_toggle_is_persisted_shared_and_not_a_scientific_edit(model_window):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    original = data.copy()
    image = _input(data, "zyx", unit="micrometer", scale=0.5)
    records = []
    window = model_window((image,), settings={"ndisplay": 3}, callback=records.append)

    def assert_axes(expected):
        for viewer in window.viewers:
            overlays = getattr(getattr(viewer, "canvas", None), "overlays", viewer)
            assert overlays.axes.visible == expected
            if hasattr(overlays.axes, "box"):
                assert not overlays.axes.box

    assert window.settings["show_axes"]
    assert window.axes_check.isChecked()
    assert_axes(True)
    camera = viewer_camera(window.viewers[0])
    pose = (camera.angles, camera.center, camera.zoom)
    point = window.viewers[0].dims.point
    window.axes_check.setChecked(False)
    assert_axes(False)
    assert not window.settings["show_axes"]
    assert records[-1]["show_axes"] is False
    assert (camera.angles, camera.center, camera.zoom) == pose
    assert window.viewers[0].dims.point == point
    window._set_general("ndisplay", 2)
    assert_axes(False)
    window._set_general("ndisplay", 3)
    assert_axes(False)
    # Applying saved presentation state updates both decorations, not pixels.
    recipe = window.settings
    recipe["show_axes"] = True
    window.set_settings(recipe)
    assert_axes(True)
    window.axes_check.setChecked(False)
    window.reset_display()
    assert window.settings["show_axes"] is True
    assert window.axes_check.isChecked()
    assert_axes(False)  # Reset returns to 2D, retaining the default axes intent.
    window._set_general("ndisplay", 3)
    assert_axes(True)
    np.testing.assert_array_equal(data, original)
    assert data.flags.writeable and not image.data.flags.writeable


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_global_scale_bar_toggle_is_persisted_shared_and_presentation_only(
    model_window, ndisplay
):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    original = data.copy()
    image = _input(data, "zyx", unit="micrometer", scale=0.5)
    records = []
    window = model_window(
        (image,), settings={"ndisplay": ndisplay}, callback=records.append
    )

    def assert_bars(expected):
        for viewer in window.viewers:
            overlays = getattr(getattr(viewer, "canvas", None), "overlays", viewer)
            assert overlays.scale_bar.visible == expected
            if hasattr(overlays.scale_bar, "box"):
                assert not overlays.scale_bar.box

    assert window.scale_bar_check.text() == "Scale bar"
    assert window.scale_bar_check.isChecked()
    assert window.settings["show_scale_bar"] is True
    assert_bars(True)
    camera = viewer_camera(window.viewers[0])
    pose = (camera.angles, camera.center, camera.zoom)
    point = window.viewers[0].dims.point
    window.scale_bar_check.setChecked(False)
    assert window.settings["show_scale_bar"] is False
    assert records[-1]["show_scale_bar"] is False
    assert_bars(False)
    assert (camera.angles, camera.center, camera.zoom) == pose
    assert window.viewers[0].dims.point == point
    window._set_general("mode", "overlay")
    assert_bars(False)
    window._set_general("ndisplay", 5 - ndisplay)
    assert_bars(False)
    # Restore the saved flag without emitting a user-edit callback.
    recipe = window.settings
    recipe["show_scale_bar"] = True
    before = len(records)
    window.set_settings(recipe)
    assert len(records) == before
    assert window.scale_bar_check.isChecked()
    assert_bars(True)
    window.scale_bar_check.setChecked(False)
    window.reset_display()
    assert window.settings["show_scale_bar"] is True
    assert window.scale_bar_check.isChecked()
    assert_bars(True)
    np.testing.assert_array_equal(data, original)
    assert data.flags.writeable and not image.data.flags.writeable


def test_scale_bar_respects_mixed_spatial_unit_eligibility_and_legacy_api(model_window):
    data = np.zeros((3, 4, 5))
    axes = (
        AxisMetadata("z", "space", unit="micrometer"),
        AxisMetadata("y", "space", unit="pixel"),
        AxisMetadata("x", "space", unit="pixel"),
    )
    image = prepare_review_input(data, image_state_from_array(data, axes=axes))
    window = model_window((image,))
    for viewer in window.viewers:
        overlays = getattr(getattr(viewer, "canvas", None), "overlays", viewer)
        assert not overlays.scale_bar.visible
    window.scale_bar_check.setChecked(False)
    window.scale_bar_check.setChecked(True)
    for viewer in window.viewers:
        overlays = getattr(getattr(viewer, "canvas", None), "overlays", viewer)
        assert not overlays.scale_bar.visible
    calibrated = model_window((_input(data, "zyx", unit="micrometer"),))
    axes_overlay = SimpleNamespace(visible=False, box=True)
    bar = SimpleNamespace(visible=False, box=True, unit=None)
    legacy = SimpleNamespace(
        axes=axes_overlay, scale_bar=bar, dims=SimpleNamespace(ndisplay=2), layers=()
    )
    calibrated._configure_scene_overlays(legacy)
    assert bar.visible and bar.unit == "micrometer" and not bar.box
    calibrated.scale_bar_check.setChecked(False)
    calibrated._configure_scene_overlays(legacy)
    assert not bar.visible and not bar.box
    assert not axes_overlay.visible and not axes_overlay.box


def test_legacy_scene_axes_without_box_keep_native_arrows_and_labels(model_window):
    window = model_window((_input(np.zeros((3, 4, 5)), "zyx", unit="micrometer"),))
    axes = SimpleNamespace(visible=False, arrows=True, labels=True)
    scale_bar = SimpleNamespace(visible=False, box=True, unit=None)
    legacy_viewer = SimpleNamespace(
        axes=axes, scale_bar=scale_bar, dims=SimpleNamespace(ndisplay=3), layers=()
    )
    window._configure_scene_overlays(legacy_viewer)
    assert axes.visible and axes.arrows and axes.labels
    assert not hasattr(axes, "box")
    assert scale_bar.visible and not scale_bar.box
    assert scale_bar.unit == "micrometer"
    legacy_viewer.dims.ndisplay = 2
    window._configure_scene_overlays(legacy_viewer)
    assert not axes.visible and axes.arrows and axes.labels


def test_mockup_vector_icons_follow_palette_changes_and_retain_text(model_window):
    window = model_window((_input(np.zeros((3, 4, 5)), "zyx"),))
    buttons = [
        *window.mode_buttons.values(),
        window.link_navigation_check,
        window.axes_check,
        window.scale_bar_check,
        window.fit_button,
        window.reset_button,
        *window.orientation_buttons.values(),
        window._controls["a"]["visible"],
    ]
    for button in buttons:
        assert not button.icon().isNull()
        assert button.text()
        assert button.accessibleName()
    assert all(
        not window.dim_combo.itemIcon(index).isNull()
        for index in range(window.dim_combo.count())
    )
    palette = QPalette(window.palette())
    palette.setColor(QPalette.ButtonText, QColor("#112233"))
    window.setPalette(palette)
    first = window.fit_button.icon().cacheKey()
    palette.setColor(QPalette.ButtonText, QColor("#aabbcc"))
    window.setPalette(palette)
    assert window.fit_button.icon().cacheKey() != first
    latest = window.fit_button.icon().cacheKey()
    QApplication.sendEvent(window, QEvent(QEvent.ApplicationPaletteChange))
    assert window.fit_button.icon().cacheKey() == latest


def test_compatible_units_share_normalized_world_coordinates(model_window):
    data = np.ones((4, 5), np.float32)
    window = model_window(
        (
            _input(data, unit="micrometer", scale=0.5),
            _input(data, unit="nanometer", scale=500),
        )
    )
    for pane in window.panes:
        np.testing.assert_allclose(pane.layers["a"][0].scale, (0.5, 0.5))
        np.testing.assert_allclose(pane.layers["b"][0].scale, (0.5, 0.5))


@pytest.mark.parametrize("kind", ["scalar", "rgb", "mask", "labels"])
def test_normalized_units_exist_before_public_layer_insertion(
    model_window, monkeypatch, kind
):
    data = np.ones((4, 5, 6), dtype=np.float32)
    declared_kind = None
    if kind == "rgb":
        data = np.ones((4, 5, 6, 3), dtype=np.uint8) * 120
    elif kind == "mask":
        data = np.ones((4, 5, 6), dtype=bool)
    elif kind == "labels":
        data = np.ones((4, 5, 6), dtype=np.uint32) * 7
        declared_kind = "label image"
    image = _input(data, "zyx", kind=declared_kind, unit="nanometer", scale=600)
    inserted_units = []
    original_add = ViewerModel.add_layer

    def capture(viewer, layer):
        inserted_units.append(
            tuple(map(str, layer.units)) if hasattr(layer, "units") else None
        )
        return original_add(viewer, layer)

    monkeypatch.setattr(ViewerModel, "add_layer", capture)
    window = model_window((image,))
    if any(units is not None for units in inserted_units):
        assert inserted_units == [("micrometer",) * 3, ("micrometer",) * 3]
    for pane in window.panes:
        np.testing.assert_allclose(pane.layers["a"][0].scale, (0.6, 0.6, 0.6))


def test_initial_calibrated_rgb_scalar_slices_are_ready_without_cursor_movement(
    model_window,
):
    scalar = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    rgb = np.stack((scalar, scalar + 1, scalar + 2), axis=-1).astype(np.uint8)
    axes = (
        AxisMetadata("z", "space", unit="nanometer", scale=600, translation=-2000),
        AxisMetadata("y", "space", unit="micrometer", scale=0.5, translation=4),
        AxisMetadata("x", "space", unit="micrometer", scale=0.4, translation=8),
    )
    scalar_state = image_state_from_array(scalar, axes=axes)
    rgb_state = image_state_from_array(
        rgb, axes=axes + (AxisMetadata("rgb", "channel"),)
    )
    window = model_window(
        (
            prepare_review_input(rgb, rgb_state, "RGB"),
            prepare_review_input(scalar, scalar_state, "Scalar"),
        )
    )
    point = window.viewers[0].dims.point
    expected_z = window.viewers[0].dims.current_step[0]
    # Inspect the native model's existing completed slice, not a refresh, a
    # cursor move, or a newly submitted request that could hide startup failure.
    for pane in window.panes:
        for key, source in (("a", rgb), ("b", scalar)):
            layer = pane.layers[key][0]
            if hasattr(layer, "_slicing_state"):
                assert layer._slicing_state._units == layer.units
            np.testing.assert_array_equal(layer._slice.image.view, source[expected_z])
    assert window.viewers[0].dims.point == point
    window._set_general("ndisplay", 3)
    window._set_general("ndisplay", 2)
    for pane in window.panes:
        for key, source in (("a", rgb), ("b", scalar)):
            layer = pane.layers[key][0]
            np.testing.assert_array_equal(layer._slice.image.view, source[expected_z])
    np.testing.assert_array_equal(scalar, np.arange(120).reshape(4, 5, 6))


def test_rgb_native_slices_and_explicit_readonly_component_mip(model_window):
    data = np.zeros((3, 4, 5, 3), np.uint8)
    data[..., 1] = 200
    window = model_window((_input(data, "zyx"),))
    assert window.settings["ndisplay"] == 2
    assert window.panes[0].layers["a"][0].rgb
    assert "black" not in window._controls["a"]
    window._set_general("ndisplay", 3)
    for pane in window.panes:
        components = pane.layers["a"]
        assert len(components) == 3
        for component, layer in enumerate(components):
            assert not layer.rgb and not layer.data.flags.writeable
            assert np.shares_memory(layer.data, data)
            np.testing.assert_array_equal(layer.data, data[..., component])
            assert str(layer.blending) == "additive"
            assert str(layer.rendering) == "mip"
            assert layer.contrast_limits == [0, 255]
    assert "per-component MIP" in window.status_label.text()
    window._set_general("ndisplay", 2)
    assert window.panes[0].layers["a"][0].rgb


def test_rgba_volume_never_silently_drops_alpha(model_window):
    data = np.zeros((3, 4, 5, 4), np.uint8)
    data[..., 3] = 120
    window = model_window((_input(data, "zyx"),))
    assert not window._three_d_available
    window._set_general("ndisplay", 3)
    assert window.settings["ndisplay"] == 2
    assert "voxel alpha is not supported" in window.status_label.text()
    assert window.panes[0].layers["a"][0].rgb
    assert window.panes[0].layers["a"][0].data.shape[-1] == 4


def test_native_canvas_dimension_button_uses_rgb_adapter_gate(model_window):
    rgb = np.zeros((3, 4, 5, 3), np.uint8)
    window = model_window((_input(rgb, "zyx"),))
    window.viewers[0].dims.ndisplay = 3
    assert window.settings["ndisplay"] == 3
    assert len(window.panes[0].layers["a"]) == 3
    rgba = np.zeros((3, 4, 5, 4), np.uint8)
    alpha_window = model_window((_input(rgba, "zyx"),))
    alpha_window.viewers[0].dims.ndisplay = 3
    assert alpha_window.viewers[0].dims.ndisplay == 2
    assert alpha_window.settings["ndisplay"] == 2
    assert "voxel alpha is not supported" in alpha_window.status_label.text()


def test_detached_json_and_external_settings_do_not_emit(model_window):
    changes = []
    window = model_window(
        (_input(np.arange(20).reshape(4, 5)),), callback=changes.append
    )
    window._set_style("a", "opacity", 0.3)
    assert len(changes) == 1
    changes[0]["a"]["opacity"] = 1
    assert window.settings["a"]["opacity"] == 0.3
    settings = window.settings
    settings["a"]["opacity"] = 0.6
    window.set_settings(settings)
    assert window.panes[0].layers["a"][0].opacity == 0.6
    assert len(changes) == 1


def test_stale_snapshot_unavailable_close_disconnects(model_window):
    window = model_window((_input(np.zeros((4, 5))),))
    window.set_current(False, "Source changed. Recalculate before Open Review.")
    assert not window.sidebar_scroll.isEnabled()
    assert not window.pane_splitter.isVisible()
    assert "Source changed" in window.status_label.text()
    left, right = window.viewers
    before = viewer_camera(right).zoom
    window.shutdown()
    assert window._connections == []
    viewer_camera(left).zoom = 99
    assert viewer_camera(right).zoom == before
    assert len(left.layers) == len(right.layers) == 0
    window.shutdown()


def test_volume_reset_preserves_buffers(model_window):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    settings = default_review_settings()
    settings["ndisplay"] = 3
    window = model_window((_input(data, "zyx"),), settings=settings)
    assert all(viewer.dims.ndisplay == 3 for viewer in window.viewers)
    window._set_style("a", "opacity", 0.2)
    window.reset_display()
    assert window.settings["a"]["opacity"] == 1
    assert all(viewer.dims.ndisplay == 2 for viewer in window.viewers)
    assert np.shares_memory(window.panes[0].layers["a"][0].data, data)
