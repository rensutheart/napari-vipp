from __future__ import annotations

import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QApplication, QWidget

from napari_vipp.installer.engine import _SHORTCUT_SCRIPT
from napari_vipp.ui.desktop_branding import apply_desktop_branding
from scripts.generate_desktop_icon import ICON_SIZES, render_icon


def _assert_same_icon_raster(actual, expected):
    assert actual.mode == expected.mode == "RGBA"
    assert actual.size == expected.size
    # Qt 6.10/6.11 can unpremultiply an antialiased edge to different straight
    # RGB values (e.g. red 17/18 at alpha 102) from the same 8-bit raster. ICO/PNG
    # encoding is lossless. Permit only the observed one-level edge rounding,
    # and only if the premultiplied raster remains exactly identical. Alpha,
    # opaque colors and transparent pixels still have to match byte-for-byte.
    actual_pixels, expected_pixels = np.asarray(actual), np.asarray(expected)
    alpha = actual_pixels[..., 3]
    np.testing.assert_array_equal(alpha, expected_pixels[..., 3])
    edge = (alpha > 0) & (alpha < 255)
    np.testing.assert_array_equal(actual_pixels[~edge], expected_pixels[~edge])
    edge_difference = np.abs(
        actual_pixels[edge, :3].astype(np.int16)
        - expected_pixels[edge, :3].astype(np.int16)
    )
    assert edge_difference.max(initial=0) <= 1
    difference = ImageChops.difference(actual.convert("RGBa"), expected.convert("RGBa"))
    assert difference.getextrema() == ((0, 0),) * 4


def test_windows_icon_contains_all_sizes_and_matches_reviewed_svg(tmp_path):
    branding = Path(str(files("napari_vipp").joinpath("assets", "branding")))
    generated = tmp_path / "mark.ico"
    render_icon(branding / "vipp-mark.svg", generated)
    with (
        Image.open(branding / "vipp-mark.ico") as actual,
        Image.open(generated) as expected,
    ):
        assert (
            actual.ico.sizes()
            == expected.ico.sizes()
            == {(size, size) for size in ICON_SIZES}
        )
        for size in actual.ico.sizes():
            image = actual.ico.getimage(size)
            _assert_same_icon_raster(image, expected.ico.getimage(size))
            assert image.getextrema()[-1] == (0, 255)


@pytest.mark.parametrize(
    "actual,expected",
    [
        ((17, 25, 40, 102), (18, 25, 40, 102)),  # Linux Qt 6.11 edge
        ((0, 42, 42, 6), (0, 43, 43, 6)),  # macOS Qt 6.11 edge
    ],
)
def test_icon_comparison_accepts_equivalent_unpremultiplication(actual, expected):
    assert actual != expected
    _assert_same_icon_raster(
        Image.new("RGBA", (1, 1), actual), Image.new("RGBA", (1, 1), expected)
    )


@pytest.mark.parametrize(
    "change",
    [
        "alpha",
        "opaque-color",
        "edge-color",
        "hidden-color",
        "edge-rounding",
        "geometry",
    ],
)
def test_icon_comparison_rejects_changed_artwork(change):
    expected = Image.new("RGBA", (3, 3), (0, 0, 0, 0))
    expected.putpixel((1, 1), (17, 24, 39, 255))
    expected.putpixel((0, 1), (17, 25, 40, 102))
    actual = expected.copy()
    if change == "alpha":
        actual.putpixel((0, 1), (17, 25, 40, 101))
    elif change == "opaque-color":
        actual.putpixel((1, 1), (18, 24, 39, 255))
    elif change == "edge-color":
        # A one-level RGB change is not allowed if premultiplication differs.
        expected.putpixel((0, 1), (18, 25, 40, 102))
        actual.putpixel((0, 1), (19, 25, 40, 102))
    elif change == "hidden-color":
        actual.putpixel((0, 0), (1, 0, 0, 0))
    elif change == "edge-rounding":
        # Equal premultiplied pixels alone must not admit arbitrary RGB drift.
        expected.putpixel((0, 1), (0, 42, 42, 6))
        actual.putpixel((0, 1), (0, 44, 42, 6))
    else:
        actual = ImageChops.offset(actual, 1, 0)
    with pytest.raises(AssertionError):
        _assert_same_icon_raster(actual, expected)


def test_desktop_branding_sets_window_icon_without_changing_host_metadata(qtbot):
    application = QApplication.instance()
    old_icon = application.windowIcon()
    original_identity = (application.applicationName(), application.organizationName())
    window = QWidget()
    qtbot.addWidget(window)
    try:
        apply_desktop_branding(application, window)
        assert not window.windowIcon().isNull()
        assert QSize(16, 16) in window.windowIcon().availableSizes()
        assert QSize(256, 256) in window.windowIcon().availableSizes()
        assert window.windowIcon().cacheKey() == application.windowIcon().cacheKey()
        application.setWindowIcon(old_icon)
        assert not window.windowIcon().isNull()  # napari can change its global icon
        assert (
            application.applicationName(),
            application.organizationName(),
        ) == original_identity
    finally:
        application.setWindowIcon(old_icon)


def test_real_napari_startup_gives_workflow_space_and_keeps_desktop_icon(
    qtbot, make_napari_viewer
):
    if QApplication.instance().platformName() in {"offscreen", "minimal"}:
        pytest.skip("Real napari canvas requires a native OpenGL-capable Qt platform")
    from napari.settings import get_settings

    from napari_vipp._widget import VippWidget
    from napari_vipp.app import _configure_initial_workflow, _construct_vipp_widget
    from napari_vipp.startup import LaunchProfile

    settings = get_settings()
    settings.application.first_time = False
    settings.application.save_window_geometry = True
    settings.application.window_size = (1500, 1000)

    viewer = make_napari_viewer(show=False)
    widget = _construct_vipp_widget(VippWidget, viewer, LaunchProfile.CPU)
    dock = viewer.window.add_dock_widget(widget, area="bottom", name="VIPP Workflow")
    dock.window().setAttribute(Qt.WA_DontShowOnScreen)
    dock.window().setAnimated(False)
    application = QApplication.instance()
    original_icon = application.windowIcon()
    try:
        apply_desktop_branding(application, dock.window())
        branded_icon = dock.window().windowIcon().cacheKey()
        _configure_initial_workflow(widget)
        viewer.show(block=False)
        QApplication.processEvents()
        widget._apply_initial_dock_size()
        qtbot.waitUntil(lambda: widget._initial_dock_size_applied)
        QApplication.processEvents()
        central = dock.window().centralWidget()
        usable = dock.height() + central.height()
        initial_height = dock.height()
        assert initial_height / usable <= 0.72
        assert central.height() >= 0.28 * usable
        if initial_height < usable * 0.60:
            # Napari's layer controls/list can require more than one third.
            # In that case VIPP must already occupy the full allowable height.
            dock.window().resizeDocks([dock], [10000], Qt.Vertical)
            QApplication.processEvents()
            assert dock.height() <= initial_height + 2
        assert dock.window().windowIcon().cacheKey() == branded_icon
        assert widget.pipeline.nodes["input"].params["source_mode"] == "sample"
    finally:
        application.setWindowIcon(original_icon)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcut COM integration")
def test_real_windows_shortcut_has_vipp_icon_and_desktop_argument(tmp_path):
    # Only create/read a temporary link, never touch the user's actual shortcuts.
    script = tmp_path / "create shortcut.ps1"
    script.write_text(_SHORTCUT_SCRIPT, encoding="utf-8")
    destination = tmp_path / "VIPP test.lnk"
    icon = Path(
        str(files("napari_vipp").joinpath("assets", "branding", "vipp-mark.ico"))
    )
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Destination",
            str(destination),
            "-Target",
            sys.executable,
            "-WorkingDirectory",
            str(tmp_path),
            "-Description",
            "Open VIPP",
            "-IconPath",
            str(icon),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Pass paths through argv, including spaces/apostrophes, not script interpolation.
    inspector = tmp_path / "read shortcut.ps1"
    inspector.write_text(
        "param([string]$Path)\n"
        "$link = (New-Object -ComObject WScript.Shell).CreateShortcut($Path)\n"
        "@{icon=$link.IconLocation; arguments=$link.Arguments; target=$link.TargetPath}"
        " | ConvertTo-Json -Compress\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(inspector),
            "-Path",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    properties = json.loads(result.stdout)
    assert properties["arguments"] == "--desktop"
    assert properties["icon"].rsplit(",", 1) == [str(icon), "0"]
    assert Path(properties["target"]) == Path(sys.executable)
