"""Detached registry/metadata checks for assistant-authored graph additions.

Only normal graph restoration and scientific axis preflight projectors are
used here. No processing kernel, source loader, scheduler or writer is called.
"""

from __future__ import annotations

from napari_vipp.core.metadata import ImageState
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    GraphConnection,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.workflow_proposal_catalogue import (
    MAX_AUTHORING_PORTS,
    required_input_ports_for,
)


def validate_graph_contracts(nodes, edges, source_state):
    """Check a source-bound candidate, with no guessed dynamic output counts."""
    if not isinstance(source_state, ImageState):
        raise ValueError("Capture fresh source metadata before preparing this plan.")
    detached = PrototypePipeline()
    detached.reset_empty_graph()
    source = detached.nodes["input"]
    detached = GraphSnapshot(
        [NodeSnapshot("input", "input", source.params)]
        + [NodeSnapshot(node.key, node.operation_id, node.params) for node in nodes],
        [
            GraphConnection(
                edge.source, edge.target, edge.target_port, edge.source_port
            )
            for edge in edges
        ],
    ).to_pipeline()
    incoming = {}
    occupied = set()
    for edge in edges:
        incoming.setdefault(edge.target, []).append(edge)
        occupied.add((edge.target, edge.target_port))
    for node in nodes:
        for port in required_input_ports_for(
            node.operation_id, node.params, pipeline=detached, node_id=node.key
        ):
            if (node.key, port) not in occupied:
                raise ValueError(
                    f"Node {node.key!r} is missing required input port {port}."
                )
        if node.operation_id == "filter_labels_by_property":
            _check_label_table_identity(node.key, incoming, detached)

    proxy = detached._axis_contract_proxy(source_state.shape, source_state.dtype)
    payloads = {"input": SourcePayload(proxy, image_state=source_state)}
    execution = detached.prepare_execution(manual_node_ids=detached.manual_node_ids())
    while execution.remaining_node_ids:
        ready = [
            key
            for key in detached.topological_order()
            if key in execution.remaining_node_ids
            and all(
                parent in execution.completed_node_ids
                for parent in detached._input_sources(key)
            )
        ]
        if not ready:
            raise ValueError("The proposed graph has unresolved input dependencies.")
        for key in ready:
            node = detached.nodes[key]
            spec = NODE_LIBRARY_BY_ID[node.operation_id]
            if spec.output_factory is not None:
                count = detached._inferred_dynamic_output_count_for_node(key, spec)
                if count is None:
                    raise ValueError(
                        f"{node.title} needs known input axes/shape "
                        "to bind its outputs; "
                        "an unresolved default output count is not used."
                    )
                if not 1 <= count <= MAX_AUTHORING_PORTS:
                    raise ValueError(
                        f"{node.title} exceeds the {MAX_AUTHORING_PORTS}-output "
                        "authoring resource limit. Select a smaller axis range first."
                    )
            if not spec.has_input:
                results = detached.source_node_results(
                    key,
                    None,
                    None,
                    "",
                    payloads,
                    defer_statistics=True,
                    metadata_only=True,
                )
            else:
                call = detached.prepare_node_call(key, axis_contract_only=True)
                if call is not None:
                    _check_known_semantics(call)
                if node.operation_id == "build_tracks":
                    _check_tracking_series(key, incoming, detached)
                if node.operation_id == "filter_labels_by_property":
                    table_edge = next(
                        edge for edge in incoming[key] if edge.target_port == 1
                    )
                    measurement = detached.nodes[table_edge.source]
                    mode = measurement.params.get("spatial_mode")
                    wrong_count = "area_pixels" if mode == "3D ZYX" else "volume_voxels"
                    if node.params.get("property_column") == wrong_count:
                        raise ValueError(
                            "The selected measurement column does not match "
                            "Measure Objects' 2D/3D spatial mode."
                        )
                if node.operation_id == "save_output":
                    source_edge = next(
                        edge for edge in incoming[key] if edge.target_port == 0
                    )
                    kind = detached.output_ports(source_edge.source)[
                        source_edge.source_port
                    ].output_type
                    mesh_format = node.params["format"] in {"obj", "3mf"}
                    if mesh_format != (kind == "mesh"):
                        raise ValueError(
                            "Save format must match its input: OBJ/3MF for meshes, "
                            "image formats for images."
                        )
                results = detached._axis_contract_preflight_results(call)
            detached.commit_node_results(execution, key, results)

    # Restore permits provisional factory hints while no source is loaded.
    # Recheck every referenced port against the actual metadata-bound count.
    for edge in edges:
        ports = detached.output_ports(edge.source)
        if edge.source_port >= len(ports):
            raise ValueError("Connection uses an unavailable source-bound output port.")
    return detached


def _check_tracking_series(key, incoming, pipeline):
    """Carry only known observation-column identity, never invented table rows.

    Actual row completeness, coordinate values and revision evidence remain
    the ordinary Build Tracks kernel's responsibility on explicit calculation.
    """
    from napari_vipp.core.observation_series import object_observation_state
    from napari_vipp.core.operations import (
        _parse_metadata_columns,
        _parse_table_column_list,
    )

    def protected_columns(source, source_port, visited):
        if source in visited:
            return None
        visited = visited | {source}
        node = pipeline.nodes[source]
        operation = node.operation_id
        edges = incoming.get(source, ())
        primary = next((edge for edge in edges if edge.target_port == 0), None)
        if operation in {"detect_spots_per_frame", "measure_objects"}:
            if primary is None:
                return None
            state = pipeline._resolved_output_state(primary.source, primary.source_port)
            if not object_observation_state(state):
                return None
            spatial_names = [axis.name.lower() for axis in state.axes[1:]]
            if operation == "detect_spots_per_frame":
                return {
                    "t_index",
                    "detection_id",
                    *(f"{axis}_index" for axis in spatial_names),
                }
            if node.params.get("spatial_mode") != (
                "3D ZYX" if len(spatial_names) == 3 else "2D YX"
            ):
                return None
            return {
                "t_index",
                "label_id",
                *(f"centroid_{axis}" for axis in spatial_names),
            }
        if (
            operation not in {"add_metadata_columns", "select_table_columns"}
            or primary is None
            or source_port != 0
        ):
            return None
        protected = protected_columns(primary.source, primary.source_port, visited)
        if protected is None:
            return None
        if operation == "add_metadata_columns":
            additions = _parse_metadata_columns(node.params["metadata_columns"])
            if protected.intersection(name for name, _value in additions):
                return None
        if operation == "select_table_columns":
            text = node.params["columns"]
            if str(text).strip().lower() in {"none", "<none>"}:
                return None
            requested = set(_parse_table_column_list(text))
            if requested:
                if (
                    str(node.params.get("selection_mode", "Keep listed columns"))
                    .lower()
                    .startswith("drop")
                ):
                    if protected & requested:
                        return None
                elif (
                    not str(node.params.get("append_unlisted", "no"))
                    .lower()
                    .startswith("y")
                ):
                    if not protected <= requested:
                        return None
        return protected

    if not any(
        protected_columns(edge.source, edge.source_port, set()) is not None
        for edge in incoming.get(key, ())
    ):
        raise ValueError(
            "Build Tracks requires a complete explicitly timed TYX/TZYX "
            "detection or object-measurement series with its original frame, "
            "object ID and position columns; summaries or edited identities "
            "cannot supply that evidence."
        )


def _check_label_table_identity(key, incoming, pipeline):
    bindings = {edge.target_port: edge for edge in incoming.get(key, ())}
    if 0 not in bindings or 1 not in bindings:
        return  # Required-port checks report this first.
    label_edge, table_edge = bindings[0], bindings[1]
    producer = pipeline.nodes[table_edge.source]
    measured = next(
        (edge for edge in incoming.get(producer.id, ()) if edge.target_port == 0), None
    )
    if (
        producer.operation_id != "measure_objects"
        or measured is None
        or (measured.source, measured.source_port)
        != (label_edge.source, label_edge.source_port)
    ):
        raise ValueError(
            "Property filtering requires a Measure Objects table from the same "
            "labels input; a histogram or unrelated table has no matching "
            "object identity."
        )


def _check_known_semantics(call):
    if call.operation_id == "detect_spots_per_frame":
        # Reuse the real detector's metadata/dtype boundary on a constant-size
        # proxy. This does not inspect pixels or calculate any detections.
        from napari_vipp.core.time_detection import _series_array

        state = call.input_states[0]
        if not isinstance(state, ImageState):
            raise ValueError("Detect Spots per Frame needs known image metadata.")
        _series_array(
            PrototypePipeline._axis_contract_proxy(state.shape, state.dtype), state
        )
    if call.operation_id in {
        "add_images",
        "subtract_images",
        "ratio_image",
        "logical_and",
        "logical_or",
        "logical_xor",
    }:
        states = [state for state in call.input_states if isinstance(state, ImageState)]
        if states and any(state.shape != states[0].shape for state in states[1:]):
            raise ValueError(
                "Image arithmetic and logical inputs must have matching shapes "
                "and axis grids. Align or crop them explicitly first."
            )
    if call.operation_id in {
        "gaussian_blur",
        "gaussian_blur_3d",
        "rescale_intensity",
        "subtract_background",
        "non_local_means_filter",
    }:
        for state in call.input_states:
            if isinstance(state, ImageState) and state.kind == "label image":
                raise ValueError(
                    "Intensity processing must not smooth or rescale "
                    "integer label IDs. "
                    "Choose an explicit mask/intensity conversion first."
                )
    if (
        call.operation_id == "measure_objects"
        and call.kwargs.get("spatial_mode") == "3D ZYX"
    ):
        if call.kwargs.get("include_2d_boundary_descriptors") or call.kwargs.get(
            "include_2d_shape_moments"
        ):
            raise ValueError(
                "2D-only measurement descriptors are not supported for "
                "3D ZYX measurement."
            )
