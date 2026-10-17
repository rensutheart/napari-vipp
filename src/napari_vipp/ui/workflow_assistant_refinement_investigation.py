"""GUI-only default evidence selection for refinement; no worker/live access."""

from __future__ import annotations

import numpy as np

from napari_vipp.core.grid import ImageGrid, compare_aligned_grids


def default_completed_output_roles(pipeline, snapshot, preservation=None):
    """Select bounded structural outputs by exact wires, never biological names.

    A labels selection may match a mesh's binary input only when the positive
    foreground AND carried grid agree exactly and that input is unambiguous.
    No output is converted, edited, substituted or silently resampled.
    """
    snapshot.assert_current(pipeline)
    ids = set(snapshot.candidate_node_ids)
    graph = snapshot.baseline
    creators = [
        n.id
        for n in graph.nodes
        if n.id in ids and n.operation_id in {"mask_to_3d_mesh", "labels_to_3d_mesh"}
    ]
    inputs = list(
        dict.fromkeys(
            (e.source_id, e.source_port)
            for e in graph.connections
            if e.target_id in creators and e.target_port == 0
        )
    )
    refs = []
    protected_ref = None
    if preservation is not None:
        protected_ref = preservation.primary_ref
        if creators and protected_ref not in inputs:
            original, state = _port(pipeline, protected_ref)
            matches = []
            for reference in inputs:
                data, metadata = _port(pipeline, reference)
                if (
                    isinstance(data, np.ndarray)
                    and isinstance(original, np.ndarray)
                    and data.shape == original.shape
                    and compare_aligned_grids(
                        ImageGrid.from_image_state(state),
                        ImageGrid.from_image_state(metadata),
                    ).compatible
                    and np.array_equal(data > 0, original > 0)
                ):
                    matches.append(reference)
            if len(matches) != 1:
                raise ValueError(
                    "Select the exact delivered mesh input as the protected mask; "
                    "its relationship to this labels output is ambiguous."
                )
            protected_ref = matches[0]
        refs.append(("protected_mask", protected_ref))
    ordered = inputs + [(node, 0) for node in creators]
    ordered += [(output.node_id, output.port) for output in snapshot._outputs]
    for reference in ordered:
        if reference in [ref for _, ref in refs]:
            continue
        if len(refs) == 8:
            break
        refs.append((f"output{len(refs) + 1}", reference))
    return [dict(role=role, node=ref[0], port=ref[1]) for role, ref in refs], (
        "protected_mask" if protected_ref is not None else None
    )


def _port(pipeline, reference):
    node, port = reference
    values = pipeline.node_outputs.get(node, ())
    states = pipeline.node_output_states.get(node, ())
    return (
        values[port] if values else pipeline.outputs[node],
        states[port] if states else pipeline.output_states[node],
    )
