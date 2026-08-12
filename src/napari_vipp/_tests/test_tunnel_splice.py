from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from napari_vipp.core.pipeline import GraphConnection, PrototypePipeline


def _graph_state(pipeline: PrototypePipeline):
    return tuple(pipeline.connections), tuple(pipeline.output_tunnel_list())


def test_node_has_graph_bindings_covers_wires_tunnels_and_declarations():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    loose = pipeline.add_node("median_filter")

    assert not pipeline.node_has_graph_bindings("missing")
    assert not pipeline.node_has_graph_bindings(loose.id)

    assert pipeline.connect("input", loose.id).success
    assert pipeline.node_has_graph_bindings("input")
    assert pipeline.node_has_graph_bindings(loose.id)

    pipeline.disconnect("input", loose.id)
    pipeline.add_output_tunnel("Unsubscribed", loose.id)

    assert pipeline.node_has_graph_bindings(loose.id)


def test_node_has_graph_bindings_covers_tunnel_subscriber_connections():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    subscriber = pipeline.add_node("median_filter")
    pipeline.add_output_tunnel("Raw", "input")
    assert pipeline.connect_to_tunnel("Raw", subscriber.id).success

    assert pipeline.node_has_graph_bindings("input")
    assert pipeline.node_has_graph_bindings(subscriber.id)


def test_splice_existing_node_before_tunnel_preserves_name_and_all_subscribers():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    direct = pipeline.add_node("gaussian_blur")
    first_subscriber = pipeline.add_node("median_filter")
    second_subscriber = pipeline.add_node("normalize_image")
    inserted = pipeline.add_node("rescale_intensity")
    assert pipeline.connect("input", direct.id).success
    original_direct = GraphConnection("input", direct.id)
    original_tunnel = pipeline.add_output_tunnel("Raw reference", "input")
    assert pipeline.connect_to_tunnel("Raw reference", first_subscriber.id).success
    assert pipeline.connect_to_tunnel("Raw reference", second_subscriber.id).success

    result = pipeline.splice_node_before_output_tunnel(
        "raw reference",
        inserted.id,
    )

    assert result.previous_tunnel == original_tunnel
    assert result.tunnel.name == "Raw reference"
    assert result.tunnel.source_id == inserted.id
    assert result.upstream_connection == GraphConnection("input", inserted.id)
    assert original_direct in pipeline.connections
    assert {
        (connection.source_id, connection.target_id, connection.tunnel_name)
        for connection in result.subscriber_connections
    } == {
        (inserted.id, first_subscriber.id, "Raw reference"),
        (inserted.id, second_subscriber.id, "Raw reference"),
    }
    assert not any(
        connection.source_id == "input" and connection.tunnel_name
        for connection in pipeline.connections
    )
    with pytest.raises(FrozenInstanceError):
        result.tunnel = original_tunnel


def test_splice_before_tunnel_supports_zero_subscribers():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    inserted = pipeline.add_node("gamma_correction")
    pipeline.add_output_tunnel("Future branch", "input")

    result = pipeline.splice_node_before_output_tunnel(
        "Future branch",
        inserted.id,
    )

    assert result.subscriber_connections == ()
    assert result.upstream_connection == GraphConnection("input", inserted.id)
    assert pipeline.output_tunnel("Future branch") == result.tunnel


def test_splice_connects_upstream_before_resolving_type_preserving_outputs():
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    subscriber = pipeline.add_node("skeletonize")
    inserted = pipeline.add_node("split_axis")
    pipeline.add_output_tunnel("Mask", "threshold")
    assert pipeline.connect_to_tunnel("Mask", subscriber.id).success

    assert pipeline.output_ports(inserted.id)[0].output_type == "image"

    result = pipeline.splice_node_before_output_tunnel("Mask", inserted.id)

    assert result.upstream_connection == GraphConnection("threshold", inserted.id)
    assert pipeline.output_ports(inserted.id)[0].output_type == "mask"
    assert result.subscriber_connections == (
        GraphConnection(inserted.id, subscriber.id, 0, 0, "Mask"),
    )


def test_splice_supports_explicit_input_and_output_ports():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    inserted = pipeline.add_node("mask_image")
    subscriber = pipeline.add_node("normalize_image")
    pipeline.add_output_tunnel("Image", "input")
    assert pipeline.connect_to_tunnel("Image", subscriber.id).success

    result = pipeline.splice_node_before_output_tunnel(
        "Image",
        inserted.id,
        inserted_input_port=0,
        inserted_output_port=0,
    )

    assert result.upstream_connection.target_port == 0
    assert result.tunnel.source_port == 0


@pytest.mark.parametrize(
    ("input_port", "output_port", "message"),
    (
        (-1, 0, "input port cannot be negative"),
        (1, 0, "input 1 does not exist"),
        (0, -1, "output port cannot be negative"),
        (0, 99, "references output 99"),
    ),
)
def test_splice_invalid_ports_restore_graph_exactly(
    input_port: int,
    output_port: int,
    message: str,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    inserted = pipeline.add_node("rescale_intensity")
    subscriber = pipeline.add_node("median_filter")
    pipeline.add_output_tunnel("Raw", "input")
    assert pipeline.connect_to_tunnel("Raw", subscriber.id).success
    before = _graph_state(pipeline)

    with pytest.raises(ValueError, match=message):
        pipeline.splice_node_before_output_tunnel(
            "Raw",
            inserted.id,
            inserted_input_port=input_port,
            inserted_output_port=output_port,
        )

    assert _graph_state(pipeline) == before


def test_splice_rejects_incompatible_upstream_without_mutation():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    inserted = pipeline.add_node("add_metadata_columns")
    pipeline.add_output_tunnel("Raw", "input")
    before = _graph_state(pipeline)

    with pytest.raises(ValueError, match="image output to table input"):
        pipeline.splice_node_before_output_tunnel("Raw", inserted.id)

    assert _graph_state(pipeline) == before


def test_splice_reroute_failure_removes_temporary_upstream_connection():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    subscriber = pipeline.add_node("mask_image")
    inserted = pipeline.add_node("otsu_threshold")
    pipeline.add_output_tunnel("Raw", "input")
    assert pipeline.connect_to_tunnel("Raw", subscriber.id, target_port=0).success
    before = _graph_state(pipeline)

    with pytest.raises(ValueError, match="mask output to image input"):
        pipeline.splice_node_before_output_tunnel("Raw", inserted.id)

    assert _graph_state(pipeline) == before
    assert not pipeline.node_has_graph_bindings(inserted.id)


def test_splice_rejects_any_existing_wire_or_tunnel_binding():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    inserted = pipeline.add_node("rescale_intensity")
    pipeline.add_output_tunnel("Destination", "input")
    pipeline.add_output_tunnel("Owned by inserted node", inserted.id)
    before = _graph_state(pipeline)

    with pytest.raises(ValueError, match="disconnect the node"):
        pipeline.splice_node_before_output_tunnel("Destination", inserted.id)

    assert _graph_state(pipeline) == before


def test_splice_rejects_missing_tunnel_or_node_without_mutation():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.add_output_tunnel("Raw", "input")
    before = _graph_state(pipeline)

    with pytest.raises(ValueError, match="Unknown tunnel"):
        pipeline.splice_node_before_output_tunnel("Missing", "input")
    with pytest.raises(ValueError, match="missing node"):
        pipeline.splice_node_before_output_tunnel("Raw", "not-a-node")

    assert _graph_state(pipeline) == before
