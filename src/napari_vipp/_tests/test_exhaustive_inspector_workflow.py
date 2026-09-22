from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from napari_vipp._sample_data import _deconvolution_image_sample
from napari_vipp.core.pipeline import (
    PALETTE_HIDDEN_OPERATION_IDS,
    PALETTE_NODE_LIBRARY,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.workflow import (
    WORKFLOW_TYPE,
    WORKFLOW_VERSION,
    canonical_workflow_document,
    workflow_document_from_snapshot,
    workflow_snapshot_from_document,
)
from napari_vipp.ui.examples import EXHAUSTIVE_EXTERNAL_SOURCE_IDS

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / "examples" / "exhaustive-inspector-showcase.json"
MANUAL_WORKFLOW_PATH = (
    REPO_ROOT / "examples" / "manual" / "exhaustive-inspector-showcase.json"
)


def _showcase_document() -> dict[str, object]:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_bundled_exhaustive_inspector_showcase_matches_manual_qa_source():
    assert WORKFLOW_PATH.read_bytes() == MANUAL_WORKFLOW_PATH.read_bytes()
    packaged = REPO_ROOT / "src" / "napari_vipp" / "examples" / WORKFLOW_PATH.name
    assert packaged.read_bytes() == WORKFLOW_PATH.read_bytes()


def test_exhaustive_inspector_showcase_is_current_and_canonical():
    document = _showcase_document()

    assert document["type"] == WORKFLOW_TYPE
    assert document["version"] == WORKFLOW_VERSION

    snapshot = workflow_snapshot_from_document(document)
    assert workflow_document_from_snapshot(snapshot) == document

    canonical = canonical_workflow_document(document)
    assert canonical == document
    assert canonical_workflow_document(canonical) == canonical


def test_exhaustive_showcase_covers_palette_with_required_2d_preparation():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    operation_counts = Counter(node.operation_id for node in snapshot.graph.nodes)
    palette_ids = {
        operation.id for operation in PALETTE_NODE_LIBRARY
    } - EXHAUSTIVE_EXTERNAL_SOURCE_IDS

    assert set(operation_counts) == palette_ids
    assert not (set(operation_counts) & PALETTE_HIDDEN_OPERATION_IDS)
    assert operation_counts["input"] >= 1
    assert {
        operation_id: count
        for operation_id, count in operation_counts.items()
        if operation_id != "input" and count != 1
    } == {
        "binary_threshold": 2,
        "h_maxima_markers": 2,
        "convert_dtype": 2,
        "rescale_intensity": 2,
        "cellprofiler_propagation": 2,
    }


def test_showcase_external_data_exception_cannot_hide_processing_nodes():
    excluded = [
        spec
        for spec in PALETTE_NODE_LIBRARY
        if spec.id in EXHAUSTIVE_EXTERNAL_SOURCE_IDS
    ]
    assert {spec.id for spec in excluded} == {"table_source"}
    assert all(not spec.has_input and spec.output_type == "table" for spec in excluded)


def test_exhaustive_inspector_showcase_places_and_connects_every_node():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    pipeline = snapshot.graph.to_pipeline()
    node_ids = set(pipeline.nodes)
    positions = snapshot.positions_dict()

    assert set(positions) == node_ids
    assert all(
        math.isfinite(coordinate)
        for position in positions.values()
        for coordinate in position
    )

    declared_order = list(pipeline.nodes)
    topological_order = pipeline.topological_order()
    assert declared_order == topological_order
    topological_rank = {
        node_id: index for index, node_id in enumerate(topological_order)
    }

    for connection in pipeline.connections:
        assert (
            topological_rank[connection.source_id]
            < topological_rank[connection.target_id]
        )
        assert (
            0
            <= connection.source_port
            < len(pipeline.output_ports(connection.source_id))
        )
        assert (
            0
            <= connection.target_port
            < pipeline.input_port_count(connection.target_id)
        )

    for node_id, node in pipeline.nodes.items():
        required_connections = pipeline._required_input_connections(node_id)
        assert required_connections is not None, (
            f"{node_id!r} ({node.operation_id}) has an unconnected required input"
        )
        assert len(required_connections) == (
            0
            if not pipeline.operation_spec(node.operation_id).has_input
            else pipeline._required_inputs_for(node)
        )


def test_exhaustive_inspector_showcase_uses_modern_time_slice_parameters():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    selector = next(
        node
        for node in snapshot.graph.nodes
        if node.operation_id == "select_axis_slice"
    )

    params = selector.params
    assert set(params) == {
        "axis",
        "index",
        "axes",
        "indices",
        "ranges",
        "range_mode",
        "remove_axes",
        "remove_indices",
    }
    assert params["range_mode"] is True
    assert params["ranges"] == ""
    assert params["axis"] == 0
    assert params["axes"] == params["remove_axes"] == "0"

    selected_index = params["index"]
    assert isinstance(selected_index, int)
    assert selected_index >= 0
    assert params["indices"] == params["remove_indices"] == str(selected_index)


def test_exhaustive_inspector_showcase_includes_binary_colocalization_mask():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    pipeline = snapshot.graph.to_pipeline()
    mask = pipeline.nodes["colocalization_mask_1"]

    assert mask.params["threshold_mode"] == "Manual"
    assert mask.params["channel_1_threshold"] == 12000.0
    assert mask.params["channel_2_threshold"] == 12000.0
    inputs = sorted(
        (
            connection.target_port,
            connection.source_id,
            connection.source_port,
            connection.tunnel_name,
        )
        for connection in pipeline.connections
        if connection.target_id == mask.id
    )
    assert inputs == [
        (0, "split_channels_1", 0, "Red channel"),
        (1, "split_channels_1", 1, "Green channel"),
    ]
    # The overlap mask is deliberately separate from RGB visualization nodes.
    assert "display_mode" not in mask.params
    notes = _showcase_document()["notes"]
    explanation = next(
        note for note in notes if note["id"] == "colocalization_mask_cleanup"
    )
    assert "Connected Components" in explanation["text"]
    assert "not counts of the original organelles" in explanation["text"]


def test_exhaustive_inspector_showcase_uses_tunnels_selectively():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    pipeline = snapshot.graph.to_pipeline()

    expected_tunnels = {
        "Born-Wolf PSF": ("born_wolf_psf_1", 0, 1),
        "Expanded labels": ("expand_labels_1", 0, 1),
        "Green channel": ("split_channels_1", 1, 15),
        "Object labels": ("relabel_sequential_1", 0, 6),
        "ROI mask": ("binary_threshold_1", 0, 12),
        "Raw volume": ("input_2", 0, 4),
        "Red channel": ("split_channels_1", 0, 20),
        "Skeleton mask": ("skeletonize_1", 0, 5),
        "Watershed labels": ("auto_watershed_from_mask_1", 0, 2),
        "Registration moving image": ("input_10", 0, 1),
        "Registration reference image": ("input_11", 0, 1),
        "Registration aligned image": ("apply_transform_1", 0, 1),
        "Registration valid coverage": ("apply_transform_1", 1, 1),
    }
    actual_tunnels = {
        tunnel.name: (tunnel.source_id, tunnel.source_port)
        for tunnel in pipeline.output_tunnel_list()
    }
    assert actual_tunnels == {
        name: (source_id, source_port)
        for name, (source_id, source_port, _) in expected_tunnels.items()
    }

    tunnel_counts = Counter(
        connection.tunnel_name
        for connection in pipeline.connections
        if connection.tunnel_name
    )
    assert tunnel_counts == Counter(
        {
            name: subscriber_count
            for name, (*_, subscriber_count) in expected_tunnels.items()
        }
    )
    assert sum(tunnel_counts.values()) == 70
    assert sum(not connection.tunnel_name for connection in pipeline.connections) == 115

    for connection in pipeline.connections:
        if not connection.tunnel_name:
            continue
        tunnel = pipeline.output_tunnel(connection.tunnel_name)
        assert tunnel is not None
        assert (connection.source_id, connection.source_port) == (
            tunnel.source_id,
            tunnel.source_port,
        )

    # The busy fan-outs still retain direct local wires for the main lane paths.
    for tunnel_name in (
        "Raw volume",
        "Red channel",
        "Green channel",
        "ROI mask",
        "Object labels",
        "Skeleton mask",
    ):
        tunnel = pipeline.output_tunnel(tunnel_name)
        assert tunnel is not None
        assert any(
            connection.source_id == tunnel.source_id
            and connection.source_port == tunnel.source_port
            and not connection.tunnel_name
            for connection in pipeline.connections
        )


def test_exhaustive_inspector_showcase_cannot_auto_save_to_disk():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    save_nodes = [
        node for node in snapshot.graph.nodes if node.operation_id == "save_output"
    ]

    assert len(save_nodes) == 1
    assert save_nodes[0].params["enabled"] == "off"
    assert save_nodes[0].params["path"] == ""


def test_showcase_registration_lane_executes_known_motion_and_coverage():
    from napari_vipp._sample_data import make_registration_sample_data
    from napari_vipp.core.registration_samples import landmark_errors, translation_pair
    from napari_vipp.core.tables import TableData
    from napari_vipp.core.transforms import TransformData

    graph = workflow_snapshot_from_document(_showcase_document()).graph.to_pipeline()
    node_ids = {
        "input_10",
        "input_11",
        "estimate_registration_1",
        "apply_transform_1",
        "compare_images_1",
    }
    branch = PrototypePipeline()
    branch.restore_graph(
        [node for node in graph.nodes.values() if node.id in node_ids],
        [
            edge
            for edge in graph.connections
            if edge.source_id in node_ids and edge.target_id in node_ids
        ],
        [
            tunnel
            for tunnel in graph.output_tunnel_list()
            if tunnel.source_id in node_ids
        ],
    )
    catalog = {
        kwargs["name"]: (image, kwargs)
        for image, kwargs, _kind in make_registration_sample_data()
    }
    sources = {}
    copies = {}
    for node_id in ("input_10", "input_11"):
        image, kwargs = catalog[branch.nodes[node_id].params["sample_name"]]
        copies[node_id] = image.copy()
        sources[node_id] = SourcePayload(image, kwargs["metadata"], kwargs["name"])
    branch.preflight_axis_contract(sources)
    branch.run(None, source_payloads=sources)
    transform, diagnostics = branch.node_outputs["estimate_registration_1"]
    aligned, coverage = branch.node_outputs["apply_transform_1"]
    comparison = branch.node_outputs["compare_images_1"][0]
    assert isinstance(transform, TransformData)
    assert isinstance(diagnostics, TableData)
    assert max(landmark_errors(transform.matrices[0], translation_pair())) < 0.12
    assert coverage.dtype == bool and 0.8 < coverage.mean() < 1.0
    assert aligned.dtype == np.float64
    record = comparison.records()[0]
    assert record["valid_count"] == np.count_nonzero(coverage)
    assert record["pearson_r"] > 0.99
    assert 0 < record["ssim_window_count"] < record["valid_count"]
    for node_id, payload in sources.items():
        assert not payload.data.flags.writeable
        np.testing.assert_array_equal(payload.data, copies[node_id])


def test_showcase_propagation_lane_executes_on_real_yx_inputs():
    from centrosome.propagate import propagate

    snapshot = workflow_snapshot_from_document(_showcase_document())
    graph = snapshot.graph.to_pipeline()
    node_ids = {
        "input_8",
        "h_maxima_markers_2",
        "binary_threshold_2",
        "cellprofiler_propagation_1",
    }
    branch = PrototypePipeline()
    branch.restore_graph(
        [node for node in graph.nodes.values() if node.id in node_ids],
        [
            edge
            for edge in graph.connections
            if edge.source_id in node_ids and edge.target_id in node_ids
        ],
    )
    image, kwargs, _kind = _deconvolution_image_sample()
    before = image.copy()
    image.setflags(write=False)
    outputs = branch.run(
        image,
        source_payloads={
            "input_8": SourcePayload(image, kwargs["metadata"], kwargs["name"]),
        },
    )
    labels = outputs["cellprofiler_propagation_1"]
    seeds = outputs["h_maxima_markers_2"]
    mask = outputs["binary_threshold_2"]
    assert image.ndim == seeds.ndim == mask.ndim == 2
    assert seeds.dtype == np.int32 and mask.dtype == bool
    assert int(seeds.max()) == 4
    assert np.count_nonzero(labels) > np.count_nonzero(seeds)
    reference, _distances = propagate(image, seeds, mask, 3276.75)
    np.testing.assert_array_equal(labels, reference)
    np.testing.assert_array_equal(image, before)
    assert branch.output_states["cellprofiler_propagation_1"].axis_order == "YX"


def test_showcase_compartment_lane_explicitly_scales_and_connects_both_nucleus_ports():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    graph = snapshot.graph.to_pipeline()
    node_ids = {
        "input_9",
        "convert_dtype_2",
        "rescale_intensity_2",
        "cellprofiler_smooth_1",
        "cellprofiler_primary_objects_1",
        "cellprofiler_threshold_1",
        "cellprofiler_propagation_seeds_1",
        "cellprofiler_propagation_2",
        "cellprofiler_finish_cells_1",
        "cellprofiler_cytoplasm_1",
    }
    branch = PrototypePipeline()
    branch.restore_graph(
        [node for node in graph.nodes.values() if node.id in node_ids],
        [
            edge
            for edge in graph.connections
            if edge.source_id in node_ids and edge.target_id in node_ids
        ],
    )
    image, kwargs, _kind = _deconvolution_image_sample()
    before = image.copy()
    image.setflags(write=False)
    sources = {"input_9": SourcePayload(image, kwargs["metadata"], kwargs["name"])}
    branch.preflight_axis_contract(sources)
    outputs = branch.run(None, source_payloads=sources)
    normalized = outputs["rescale_intensity_2"]
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(
        normalized, image.astype(np.float32) / np.float32(65535)
    )
    nuclei, unedited = branch.node_outputs["cellprofiler_primary_objects_1"]
    cells = outputs["cellprofiler_finish_cells_1"]
    cytoplasm = outputs["cellprofiler_cytoplasm_1"]
    assert (
        nuclei.shape == unedited.shape == cells.shape == cytoplasm.shape == image.shape
    )
    assert nuclei.max() > 0
    assert np.count_nonzero(cytoplasm) > 0
    assert set(np.unique(cytoplasm)) <= set(np.unique(cells))
    assert branch.output_states["cellprofiler_cytoplasm_1"].axis_order == "YX"
    np.testing.assert_array_equal(image, before)
    assert {
        (edge.source_id, edge.source_port, edge.target_port)
        for edge in branch.connections
        if edge.target_id == "cellprofiler_propagation_seeds_1"
    } == {
        ("cellprofiler_primary_objects_1", 1, 0),
        ("cellprofiler_primary_objects_1", 0, 1),
    }
