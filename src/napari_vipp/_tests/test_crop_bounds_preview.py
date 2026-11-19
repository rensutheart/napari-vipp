"""The compatibility Crop preview uses fixed-size Qt paint operations only."""

from qtpy.QtGui import QColor, QPalette

from napari_vipp.ui.crop_bounds_preview import CropBoundsPreview, _inner_interval


def test_schematic_retained_ranges_update_and_clear(qtbot):
    widget = CropBoundsPreview()
    qtbot.addWidget(widget)
    widget.resize(240, widget.sizeHint().height())
    widget.show()
    empty = widget.grab().toImage()
    widget.set_bounds(x=(80, 5, 10), y=(40, 2, 9), z=(9, 1, 3))
    assert widget.accessibleDescription() == (
        "Retained samples: X 6–70 of 80, Y 3–31 of 40, Z 2–6 of 9."
    )
    volume = widget.grab().toImage()
    assert volume != empty
    widget.set_bounds(x=(80, 5, 10), y=(40, 2, 9))
    assert "Z " not in widget.accessibleDescription()
    assert widget.grab().toImage() != volume
    for incomplete in [
        {"x": (80, 0, 0)},
        {"x": (80, 5, 75), "y": (40, 0, 0)},
        {"x": (80, 0, 0), "y": (40, 0, 0), "z": (9, -1, 0)},
    ]:
        widget.set_bounds(**incomplete)
        assert widget.accessibleDescription() == "Crop bounds unavailable."
        assert widget.grab().toImage() == empty


def test_schematic_retains_correct_side_and_single_sample_ranges(qtbot):
    widget = CropBoundsPreview()
    qtbot.addWidget(widget)
    palette = widget.palette()
    palette.setColor(QPalette.Highlight, QColor("#ff0000"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    widget.setPalette(palette)
    widget.resize(220, widget.sizeHint().height())
    widget.show()
    axis = (100, 0, 0)
    widget.set_bounds(x=(100, 50, 0), y=axis)
    im = widget.grab().toImage()
    ratio = im.devicePixelRatio()
    # At vertical centre the leftmost excluded quarter stays white; the
    # retained quarter to the right is tinted with the highlight colour.
    label_w = widget.fontMetrics().horizontalAdvance("XY") + 6
    centre = (widget.width() + label_w) // 2
    row = int(widget.height() * 0.65 * ratio)
    excluded = im.pixelColor(int((centre - 10) * ratio), row)
    retained = im.pixelColor(int((centre + 10) * ratio), row)
    assert excluded.red() == excluded.green() == excluded.blue() == 255
    assert retained.red() > retained.green()
    assert _inner_interval((5, 4, 0), 10, 20) == (26, 4)
    widget.set_bounds(x=(10**12, 0, 10**12 - 1), y=(1, 0, 0))
    assert "X 1–1 of 1000000000000" in widget.accessibleDescription()
    widget.resize(30, widget.sizeHint().height())
    assert not widget.grab().isNull()
