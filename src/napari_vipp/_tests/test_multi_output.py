from __future__ import annotations

import numpy as np

from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _rgb_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((8, 8, 3)) * 255).astype(np.uint8)


def _two_channel_image() -> np.ndarray:
    data = np.zeros((2, 8, 8), dtype=np.uint8)
    data[0] = 10
    data[1] = 20
    return data


def test_default_node_has_single_named_output():
    pipeline = PrototypePipeline()
    ports = pipeline.output_ports("gaussian")
    assert len(ports) == 1
    assert ports[0].name == "out"


def test_split_channels_defaults_to_three_output_ports():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("split_channels")
    ports = pipeline.output_ports(node.id)
    assert [port.name for port in ports] == [
        "channel_1",
        "channel_2",
        "channel_3",
    ]
    assert [port.label for port in ports] == ["Ch 1", "Ch 2", "Ch 3"]


def test_split_channels_preserves_mask_output_type_for_downstream_labels():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    split = pipeline.add_node("split_channels")
    labels = pipeline.add_node("label_connected_components")

    assert pipeline.connect("threshold", split.id).success
    assert {
        port.output_type for port in pipeline.output_ports(split.id)
    } == {"mask"}

    result = pipeline.connect(split.id, labels.id, source_port=0)

    assert result.success
    assert result.connection is not None
    assert result.connection.source_port == 0


def test_split_channels_run_produces_lossless_outputs():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)

    image = _rgb_image()
    pipeline.run(image)

    outputs = pipeline.node_outputs[node.id]
    assert len(outputs) == 3
    assert all(out.shape == (8, 8) for out in outputs)
    assert all(out.dtype == image.dtype for out in outputs)
    np.testing.assert_array_equal(outputs[0], image[..., 0])
    np.testing.assert_array_equal(outputs[1], image[..., 1])
    np.testing.assert_array_equal(outputs[2], image[..., 2])


def test_split_channels_adjusts_ports_to_true_channel_count():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)

    pipeline.run(_two_channel_image(), input_metadata={"axes": "CYX"})

    ports = pipeline.output_ports(node.id)
    assert [port.name for port in ports] == ["channel_1", "channel_2"]
    assert len(pipeline.node_outputs[node.id]) == 2


def test_split_axis_grayscale_yields_one_port_per_axis_index():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_axis")
    pipeline.connect("input", node.id)

    image = np.full((3, 8), 7, dtype=np.uint8)
    pipeline.run(image, input_metadata={"axes": "YX"})

    ports = pipeline.output_ports(node.id)
    assert [port.name for port in ports] == ["y_1", "y_2", "y_3"]
    assert len(pipeline.node_outputs[node.id]) == 3


def test_trim_invalid_output_connections_drops_stale_ports():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_axis")
    pipeline.connect("input", node.id)
    pipeline.connect(node.id, "gaussian", None, 2)

    # Two slices shrink the node to ports 0 and 1, invalidating port 2.
    pipeline.run(_two_channel_image())
    assert pipeline.outputs["gaussian"] is None
    removed = pipeline.trim_invalid_output_connections(node.id)

    assert [connection.source_port for connection in removed] == [2]
    assert all(
        connection.source_id != node.id or connection.source_port < 2
        for connection in pipeline.connections
    )


def test_split_output_metadata_drops_channel_axis():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)

    pipeline.run(_rgb_image())

    states = pipeline.node_output_states[node.id]
    assert [state.axis_order for state in states] == ["YX", "YX", "YX"]


def test_connect_routes_selected_source_port():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)
    result = pipeline.connect(node.id, "gaussian", None, 1)
    assert result.success
    assert result.connection.source_port == 1

    image = _rgb_image()
    pipeline.run(image)

    np.testing.assert_array_equal(
        pipeline.input_data_for_node("gaussian"), image[..., 1]
    )


def test_connect_rejects_out_of_range_source_port():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    result = pipeline.connect("input", node.id)
    assert result.success

    bad = pipeline.connect(node.id, "gaussian", None, 5)
    assert not bad.success
    assert "output" in bad.message.lower()


def test_workflow_roundtrip_preserves_source_port():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)
    pipeline.connect(node.id, "gaussian", None, 2)

    restored = deserialize_workflow(serialize_workflow(pipeline))
    matches = [
        connection
        for connection in restored["connections"]
        if connection.target_id == "gaussian" and connection.source_id == node.id
    ]
    assert matches
    assert matches[0].source_port == 2


def test_workflow_restore_preserves_referenced_dynamic_port_above_default():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_axis")
    pipeline.connect("input", node.id)
    image = np.stack(
        [np.full((6, 7), index, dtype=np.uint8) for index in range(5)]
    )
    pipeline.run(image, input_metadata={"axes": "ZYX"})
    assert pipeline.connect(node.id, "gaussian", source_port=4).success

    workflow = deserialize_workflow(serialize_workflow(pipeline))
    restored = PrototypePipeline()
    restored.restore_graph(workflow["nodes"], workflow["connections"])

    assert len(restored.output_ports(node.id)) == 5
    restored.run(image, input_metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(restored.outputs["gaussian"], image[4])


def test_export_indexes_multi_output_source():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)
    pipeline.connect(node.id, "gaussian", None, 1)

    code = export_pipeline_to_python(pipeline)

    assert f"v_{node.id} = split_channels(" in code
    assert f"gaussian_blur(v_{node.id}[1]" in code


def test_graph_view_renders_three_output_ports(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")
    pipeline.connect("input", node.id)

    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)

    view.set_node_output_ports(
        node.id,
        3,
        ["Ch 1", "Ch 2", "Ch 3"],
        ["#ef4444", "#22c55e", "#60a5fa"],
        ["mask", "mask", "mask"],
    )

    proxy = view._proxies[node.id]
    assert len(proxy.output_ports) == 3
    assert [port.port_index for port in proxy.output_ports] == [0, 1, 2]
    assert proxy.output_port_at(1).accent_color == "#22c55e"
    assert proxy.output_port_at(1).data_type == "mask"


def test_graph_view_renders_typed_input_ports(qtbot):
    pipeline = PrototypePipeline()
    node = pipeline.add_node("measure_objects_intensity")

    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    input_ports = pipeline.input_ports(node.id)
    view.set_node_input_ports(
        node.id,
        len(input_ports),
        [port.label for port in input_ports],
        [None for _port in input_ports],
        [port.input_type for port in input_ports],
    )

    proxy = view._proxies[node.id]
    assert [port.label for port in proxy.input_ports] == [
        "Labels",
        "Intensity image",
    ]
    assert [port.data_type for port in proxy.input_ports] == ["labels", "image"]


def test_graph_view_connects_specific_output_port(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    node = pipeline.add_node("split_channels")

    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    view.set_node_output_ports(node.id, 3, ["Ch 1", "Ch 2", "Ch 3"], [None, None, None])

    emitted: list[tuple] = []
    view.connection_requested.connect(
        lambda s, t, tp, sp: emitted.append((s, t, tp, sp))
    )

    view.begin_connection(
        view._proxies[node.id].output_port_at(2),
        view._proxies[node.id].port_scene_pos("output", 2),
    )
    view.complete_connection(view._proxies["gaussian"].input_port_at(0))

    assert emitted == [(node.id, "gaussian", 0, 2)]
