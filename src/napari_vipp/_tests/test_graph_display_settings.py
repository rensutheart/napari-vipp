from types import SimpleNamespace

import pytest

from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.ui.graph_display_settings import (
    graph_display_option_label,
    thumbnail_has_stack,
)


@pytest.mark.parametrize(
    ("control", "canonical", "expected"),
    (
        ("Thumbnail view", "Slice", "Current slice"),
        ("Thumbnail view", "MIP", "Maximum-intensity projection (MIP)"),
        ("Thumbnail view", "Off", "Hidden"),
        ("Contrast based on", "Stack", "Entire stack"),
        ("Contrast based on", "Slice", "Current slice"),
        ("Contrast method", "Percentile", "Percentile (0.5–99.9%)"),
        ("Contrast method", "Min-max", "Minimum–maximum"),
        ("Contrast method", "Raw", "Raw"),
        ("Thumbnail resolution", "Low (90 × 55)", "Low (90 × 55 px)"),
        ("Thumbnail resolution", "Standard (180 × 110)", "Standard (180 × 110 px)"),
        ("Thumbnail resolution", "High (360 × 220)", "High (360 × 220 px)"),
        ("Thumbnail resolution", "Very High (720 × 440)", "Very high (720 × 440 px)"),
        ("Input/output labels", "Ambiguous only", "When needed"),
        ("Input/output labels", "Show all", "Always"),
        ("Input/output labels", "Hide all", "Never"),
        ("Colour map", "Gray", "Gray"),
        ("Colour map", "Viridis", "Viridis"),
    ),
)
def test_display_labels_are_separate_from_canonical_values(
    control, canonical, expected,
):
    assert graph_display_option_label(
        control, canonical, has_stack=True, preview_mode="Slice"
    ) == expected


@pytest.mark.parametrize("control", ("Thumbnail view", "Contrast based on"))
def test_plain_images_use_current_image_even_with_a_saved_mip_preference(control):
    assert graph_display_option_label(
        control, "Slice", has_stack=False, preview_mode="MIP"
    ) == "Current image"


@pytest.mark.parametrize(
    ("shape", "axis_names", "expected"),
    (
        ((8, 12), "yx", False),
        ((8, 12, 3), "yxc", False),
        ((3, 8, 12), "cyx", False),
        ((1, 8, 12), "zyx", False),
        ((4, 8, 3), "zyx", True),
        ((2, 3, 8, 12), "tcyx", True),
        ((1, 3, 8, 12), "tcyx", False),
        ((4, 8, 12), "qyx", True),
        ((8, 12, 3), "", False),
        ((4, 8, 12), "", True),
    ),
)
def test_stack_detection_uses_shapes_and_axis_roles(shape, axis_names, expected):
    axes = tuple(
        AxisMetadata(name, "channel" if name == "c" else "space")
        for name in axis_names
    )
    state = SimpleNamespace(axes=axes) if axes else None
    assert thumbnail_has_stack(shape, state) is expected
