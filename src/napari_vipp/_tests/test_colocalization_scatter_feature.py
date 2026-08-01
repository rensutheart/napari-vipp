from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AxisMetadata,
    image_state_from_array,
    transform_multi_input_image_state,
)
from napari_vipp.core.operations import (
    _coloc_scatter_density_for_output,
    colocalization_populated_ranges,
    colocalization_scatter_plot,
)
from napari_vipp.core.pipeline import (
    COLOCALIZATION_THRESHOLD_OPERATIONS,
    NODE_LIBRARY_BY_ID,
    SAME_SHAPE_GRID_OPERATIONS,
    PrototypePipeline,
    SourcePayload,
)


def test_populated_ranges_are_native_independent_and_roi_restricted():
    channel_1 = np.asarray([0.0, 10.0, 20.0, 30.0, 10_000.0])
    channel_2 = np.asarray([500.0, 510.0, 520.0, 530.0, -10_000.0])
    roi = np.asarray([True, True, True, True, False])

    range_1, range_2 = colocalization_populated_ranges(
        channel_1,
        channel_2,
        roi_mask=roi,
    )

    assert range_1 == (0.0, 30.0)
    assert range_2 == (500.0, 530.0)


def test_populated_range_percentile_clips_sparse_outliers():
    channel_1 = np.asarray([0.0, 1.0, 2.0, 3.0, 10_000.0])
    channel_2 = np.asarray([-10_000.0, 100.0, 101.0, 102.0, 103.0])

    range_1, range_2 = colocalization_populated_ranges(
        channel_1,
        channel_2,
        percentile=80.0,
    )

    assert range_1[1] < 10_000.0
    assert range_2[0] > -10_000.0
    assert range_1 != range_2


def test_scatter_render_has_configurable_density_and_output_resolution():
    channel_1 = np.arange(256, dtype=np.float32).reshape(16, 16)
    channel_2 = (500.0 + channel_1 * 2.0).astype(np.float32)

    scatter = colocalization_scatter_plot(
        [channel_1, channel_2],
        channel_1_threshold=64.0,
        channel_2_threshold=700.0,
        bins=256,
        output_size=1_024,
        range_percentile=99.0,
    )

    assert scatter.shape == (1_024, 1_024, 3)
    assert scatter.dtype == np.float32
    assert 0.0 <= float(scatter.min()) <= float(scatter.max()) <= 1.0


def test_scatter_downsampling_aggregates_all_bins_and_preserves_total():
    density = np.zeros((512, 512), dtype=np.float64)
    density[1, 1] = 7.0
    density[510, 510] = 5.0

    reduced = _coloc_scatter_density_for_output(density, 64)

    assert reduced.shape == (64, 64)
    assert reduced[0, 0] == 7.0
    assert reduced[-1, -1] == 5.0
    assert float(reduced.sum()) == 12.0


def test_scatter_guides_survive_large_bin_downsampling_with_asymmetric_ranges():
    channel_1 = np.linspace(0.0, 10.0, 257, dtype=np.float32).reshape(1, -1)
    channel_2 = np.linspace(100.0, 200.0, 257, dtype=np.float32).reshape(1, -1)

    scatter = colocalization_scatter_plot(
        [channel_1, channel_2],
        channel_1_threshold=5.0,
        channel_2_threshold=150.0,
        bins=512,
        output_size=64,
    )

    x_position = 32
    y_position = 31
    np.testing.assert_allclose(scatter[0, x_position], (1.0, 0.2, 0.2))
    np.testing.assert_allclose(scatter[y_position, 0], (0.2, 1.0, 0.2))
    np.testing.assert_allclose(scatter[y_position, x_position], (1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 31}, "bins"),
        ({"bins": 4_097}, "bins"),
        ({"output_size": 63}, "output size"),
        ({"output_size": 4_097}, "output size"),
        ({"range_percentile": 0.0}, "percentile"),
    ],
)
def test_scatter_render_rejects_unsafe_resolution_or_range(kwargs, message):
    channel = np.arange(64, dtype=np.float32).reshape(8, 8)

    with pytest.raises(ValueError, match=message):
        colocalization_scatter_plot([channel, channel], **kwargs)


@pytest.mark.parametrize(
    ("operation_id", "input_count"),
    [
        ("colocalization_scatter_plot", 2),
        ("masked_colocalization_scatter_plot", 3),
    ],
)
def test_scatter_pipeline_nodes_are_registered(operation_id, input_count):
    spec = NODE_LIBRARY_BY_ID[operation_id]
    parameters = {parameter.name: parameter for parameter in spec.parameters}

    assert spec.function is colocalization_scatter_plot
    assert spec.output_type == "image"
    assert spec.execution_policy == "manual"
    assert len(spec.input_ports) == input_count
    assert parameters["bins"].maximum == 4_096
    assert parameters["output_size"].maximum == 4_096
    assert parameters["range_percentile"].default == 100.0
    assert operation_id in COLOCALIZATION_THRESHOLD_OPERATIONS
    assert operation_id in SAME_SHAPE_GRID_OPERATIONS


def test_scatter_pipeline_metadata_is_explicit_rgb():
    input_data = np.zeros((4, 5), dtype=np.uint16)
    input_state = image_state_from_array(
        input_data,
        axes=(
            AxisMetadata(
                "y",
                "space",
                unit="micrometer",
                scale=0.2,
                translation=7.0,
            ),
            AxisMetadata(
                "x",
                "space",
                unit="micrometer",
                scale=0.3,
                translation=11.0,
            ),
        ),
    )
    output = np.zeros((768, 768, 3), dtype=np.float32)

    state = transform_multi_input_image_state(
        output,
        [input_state, input_state],
        operation_id="colocalization_scatter_plot",
        operation_title="Colocalization Scatter Plot",
        params={
            "bins": 512,
            "output_size": 768,
            "range_percentile": 99.5,
            "threshold_mode": "Manual",
        },
    )

    assert state is not None
    assert tuple(axis.name for axis in state.axes) == ("y", "x", "rgb")
    assert tuple(axis.unit for axis in state.axes) == ("pixel", "pixel", None)
    assert tuple(axis.scale for axis in state.axes) == (1.0, 1.0, 1.0)
    assert tuple(axis.translation for axis in state.axes) == (0.0, 0.0, 0.0)
    assert all(axis.source_axis is None for axis in state.axes)
    assert state.kind == "RGB image"
    assert "512 bins, 768 x 768 RGB" in state.history[-1]


def test_scatter_pipeline_node_executes_with_configured_output_size():
    channel_1 = np.arange(64, dtype=np.float32).reshape(8, 8)
    channel_2 = 500.0 + channel_1
    state_1 = image_state_from_array(channel_1, layer_metadata={"axes": "YX"})
    state_2 = image_state_from_array(channel_2, layer_metadata={"axes": "YX"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_1 = pipeline.add_node("input")
    source_2 = pipeline.add_node("input")
    scatter = pipeline.add_node("colocalization_scatter_plot")
    pipeline.set_param(scatter.id, "bins", 64)
    pipeline.set_param(scatter.id, "output_size", 256)
    pipeline.connect(source_1.id, scatter.id, target_port=0)
    pipeline.connect(source_2.id, scatter.id, target_port=1)

    outputs = pipeline.run(
        channel_1,
        source_payloads={
            source_1.id: SourcePayload(channel_1, image_state=state_1),
            source_2.id: SourcePayload(channel_2, image_state=state_2),
        },
    )

    assert outputs[scatter.id].shape == (256, 256, 3)
    assert pipeline.output_states[scatter.id].kind == "RGB image"
