"""Presentation-only format choices for connected output nodes."""

from napari_vipp.core.meshes import is_mesh_data
from napari_vipp.core.tables import is_table_data


def writer_format_choices(pipeline, node_id, declared_choices):
    """Use graph types before calculation; never inspect or convert pixel data."""
    node = pipeline.nodes[node_id]
    default = "auto" if node.operation_id == "save_output" else "batch default"
    ports = pipeline.output_ports(node_id)
    kind = ports[0].output_type if ports else "any"
    if kind == "any":
        data = pipeline.input_data_for_node(node_id)
        if is_mesh_data(data):
            kind = "mesh"
        elif is_table_data(data):
            kind = "table"
        elif data is not None and hasattr(data, "shape") and hasattr(data, "dtype"):
            kind = "array"
    if kind == "mesh":
        allowed = {default, "obj", "3mf"}
    elif kind == "table":
        allowed = (
            {default, "csv", "tsv"} if node.operation_id == "batch_output" else set()
        )
    elif kind in {"array", "image", "mask", "labels"}:
        allowed = set(declared_choices) - {"obj", "3mf", "csv", "tsv"}
    else:
        allowed = {default}
    return tuple(value for value in declared_choices if value in allowed)
