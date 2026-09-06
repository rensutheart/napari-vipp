from __future__ import annotations

import os
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy import API_NAME
from qtpy.QtWidgets import QApplication

from napari_vipp._widget import VippWidget
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array


def test_requested_qt_binding_is_the_only_active_binding() -> None:
    """Keep the compatibility job honest about its platform Qt choice."""

    expected = os.environ.get("VIPP_EXPECTED_QT_API")
    if expected:
        assert API_NAME.lower() == expected.lower()
        other_binding = "PySide6" if expected.lower() == "pyqt6" else "PyQt6"
        assert find_spec(other_binding) is None


def test_vipp_starts_and_closes_with_real_napari_viewer_model(qtbot) -> None:
    """Exercise the public model and VIPP widget lifecycle without a GL canvas."""

    viewer = ViewerModel()
    source = viewer.add_image(
        np.arange(4 * 8 * 10, dtype=np.uint16).reshape(4, 8, 10),
        name="compatibility source",
        metadata={"axes": "ZYX"},
    )
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)

    assert widget.viewer is viewer
    assert viewer.layers[source.name] is source
    assert widget.pipeline.nodes["input"].operation_id == "input"

    widget.close()
    QApplication.processEvents()

    assert not widget.isVisible()


def test_node_library_icons_and_compact_controls_are_binding_neutral(qtbot) -> None:
    """Keep the responsive node library on the PyQt/PySide runtime matrix."""

    widget = VippWidget(ViewerModel(), defer_initial_run=True)
    qtbot.addWidget(widget)

    first_category = widget.palette.topLevelItem(0)
    assert first_category.text(0) == "Image Data"
    assert not first_category.icon(0).isNull()
    widget.palette_panel.set_compact(True)
    category_buttons = tuple(widget.palette_panel.compact_rail.category_buttons)

    assert widget.palette_panel.is_compact
    assert category_buttons
    assert all(button.accessibleName() for button in category_buttons)
    assert all(not button.icon().isNull() for button in category_buttons)

    widget.close()
    QApplication.processEvents()


def test_vipp_generated_image_and_labels_use_real_napari_layers(qtbot) -> None:
    """Cover the generated-layer API shared by Inspect and pinned outputs."""

    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)

    image = np.arange(3 * 6 * 8, dtype=np.uint16).reshape(3, 6, 8)
    labels = np.zeros((3, 6, 8), dtype=bool)
    labels[:, 2:5, 3:7] = 1
    state = image_state_from_array(image, layer_metadata={"axes": "ZYX"})
    label_state = image_state_from_array(labels, layer_metadata={"axes": "ZYX"})
    assert state is not None
    assert label_state is not None
    common = {
        "napari_vipp_kind": "pinned",
        "node_id": "input",
        "output_port": 0,
        "vipp_image_state": state.to_dict(),
    }

    widget._set_or_add_generated_layer(  # noqa: SLF001 - compatibility seam
        "compatibility image",
        image,
        metadata=dict(common),
        role="pinned",
    )
    image_layer = viewer.layers["compatibility image"]
    assert image_layer.data.shape == image.shape
    assert image_layer.metadata["napari_vipp_kind"] == "pinned"
    assert tuple(image_layer.axis_labels) == ("Z", "Y", "X")

    renamed_state = image_state_from_array(
        image,
        axes=(
            AxisMetadata("t", "time"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert renamed_state is not None
    widget._set_or_add_generated_layer(  # noqa: SLF001 - compatibility seam
        "compatibility image",
        image,
        metadata={**common, "vipp_image_state": renamed_state.to_dict()},
        role="pinned",
    )
    assert viewer.layers["compatibility image"] is image_layer
    assert tuple(image_layer.axis_labels) == ("T", "Y", "X")

    widget._set_or_add_generated_layer(  # noqa: SLF001 - compatibility seam
        "compatibility labels",
        labels,
        metadata={**common, "vipp_image_state": label_state.to_dict()},
        role="pinned",
    )
    labels_layer = viewer.layers["compatibility labels"]
    assert labels_layer.data.shape == labels.shape
    assert labels_layer.metadata["display_kind"] == "labels"
    assert tuple(labels_layer.axis_labels) == ("Z", "Y", "X")

    widget.close()
    QApplication.processEvents()


@pytest.mark.parametrize(
    ("component_count", "component_name", "kind"),
    [
        (3, "c", "RGB image"),
        (4, "rgba", "RGBA image"),
    ],
)
def test_real_napari_rgb_component_axis_is_not_a_viewer_dimension(
    qtbot,
    component_count,
    component_name,
    kind,
) -> None:
    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    data = np.zeros((6, 8, component_count), dtype=np.uint8)
    state = replace(
        image_state_from_array(
            data,
            axes=(
                AxisMetadata("y", "space", scale=0.2),
                AxisMetadata("x", "space", scale=0.3),
                AxisMetadata(component_name, "channel"),
            ),
        ),
        kind=kind,
    )
    metadata = {
        "napari_vipp_kind": "inspect",
        "node_id": "input",
        "data_kind": "image",
        "display_kind": "image",
        "display_ndim": data.ndim,
        "display_shape": data.shape,
        "display_rgb": True,
        "vipp_image_state": state.to_dict(),
    }

    layer = widget._add_image_or_labels(  # noqa: SLF001 - compatibility seam
        "compatibility rgb",
        data,
        metadata,
    )

    assert layer.rgb is True
    assert tuple(layer.axis_labels) == ("Y", "X")
    assert tuple(layer.scale) == (0.2, 0.3)

    widget.close()
    QApplication.processEvents()


def test_real_napari_scalar_mask_with_trailing_three_axis_stays_three_dimensional(
    qtbot,
) -> None:
    """Do not let napari's shape heuristic override VIPP's scalar metadata."""

    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    # napari only guesses RGB for sufficiently large images, so use dimensions
    # above its heuristic cutoff to reproduce the reported presentation crash.
    data = np.zeros((50, 60, 3), dtype=bool)
    state = replace(
        image_state_from_array(
            data,
            axes=(
                AxisMetadata("y", "space", scale=0.2, translation=1.0),
                AxisMetadata("x", "space", scale=0.3, translation=2.0),
                AxisMetadata("rgb", "channel"),
            ),
        ),
        kind="binary mask",
    )
    metadata = {
        "napari_vipp_kind": "inspect",
        "node_id": "threshold",
        "data_kind": "mask",
        "display_kind": "image",
        "display_ndim": data.ndim,
        "display_shape": data.shape,
        "display_rgb": False,
        "vipp_image_state": state.to_dict(),
    }

    layer = widget._add_image_or_labels(  # noqa: SLF001 - compatibility seam
        "compatibility scalar mask",
        data,
        metadata,
    )

    assert layer.rgb is False
    assert layer.ndim == 3
    assert tuple(layer.axis_labels) == ("Y", "X", "rgb")
    assert tuple(layer.scale) == (0.2, 0.3, 1.0)
    assert tuple(layer.translate) == (1.0, 2.0, 0.0)

    widget.close()
    QApplication.processEvents()
