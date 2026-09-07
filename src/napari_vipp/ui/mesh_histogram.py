"""Input mesh distributions from cached geometry measurements, never pixels."""

from dataclasses import dataclass
from math import ceil, isfinite, log10

import numpy as np

MESH_FILTER_LABELS = {
    "mesh_volume_physical": "Volume",
    "mesh_surface_area_physical": "Surface area",
    "sphericity": "Sphericity",
    "mesh_extent_z_physical": "Z extent",
    "mesh_extent_y_physical": "Y extent",
    "mesh_extent_x_physical": "X extent",
    "vertex_count": "Vertex count",
    "triangle_count": "Triangle count",
    "mesh_id": "Object ID",
}


@dataclass(frozen=True)
class MeshFilterHistogram:
    label: str
    summary: str
    counts: np.ndarray
    edges: np.ndarray
    x_range: tuple[float, float]
    omitted: int


def mesh_filter_decimals(table, property_name, limits=()):
    """Keep fractional physical bounds useful without rounding counts or IDs."""
    if property_name in {"mesh_id", "triangle_count", "vertex_count"}:
        return 0
    positive = [abs(v) for v in limits if isfinite(v) and v != 0]
    if table is not None and property_name in table.columns:
        column = table.columns.index(property_name)
        positive.extend(
            abs(row[column])
            for row in table.rows
            if row[column] is not None and isfinite(row[column]) and row[column] != 0
        )
    if not positive:
        return 6
    return max(6, min(15, ceil(-log10(min(positive))) + 3))


def mesh_filter_histogram(table, property_name, minimum, maximum, keep, *, log=False):
    """Match the filter's finite inclusive rule; bounds never stretch the plot."""
    if (
        isinstance(minimum, (bool, np.bool_))
        or isinstance(maximum, (bool, np.bool_))
        or not isfinite(minimum)
        or not isfinite(maximum)
        or minimum > maximum
    ):
        raise ValueError("Mesh filter limits must be finite with minimum ≤ maximum.")
    if keep not in {"In range", "Outside range"}:
        raise ValueError("Choose In range or Outside range.")
    title = MESH_FILTER_LABELS[property_name]
    column = table.columns.index(property_name)
    unit = (
        str(table.unit_for(property_name) or "").replace("^3", "³").replace("^2", "²")
    )
    label = f"{title} ({unit})" if unit else title
    integer = property_name in {"mesh_id", "triangle_count", "vertex_count"}
    values = []
    for row in table.rows:
        value = row[column]
        if value is not None and isfinite(value):
            values.append(int(value) if integer else float(value))
    omitted = table.row_count - len(values)
    # Python integer comparisons retain wide object IDs instead of silently
    # rounding them through float64 while deciding how many objects match.
    inside = [minimum <= value <= maximum for value in values]
    matching = sum(inside) if keep == "In range" else len(values) - sum(inside)
    summary = (
        f"{table.row_count} input objects · {matching} match · {omitted} unavailable"
    )
    empty = MeshFilterHistogram(
        label,
        summary,
        np.array([], dtype=np.int64),
        np.array([], dtype=float),
        (0.0, 1.0),
        omitted,
    )
    if not values:
        return empty
    if integer and max(abs(value) for value in values) > 2**53:
        return MeshFilterHistogram(
            label,
            summary + "\nValues exceed exact plot precision; use numeric limits.",
            empty.counts,
            empty.edges,
            empty.x_range,
            omitted,
        )
    array = np.asarray(values, dtype=np.float64)
    largest = float(array.max())
    summary += f"\n{label} · median {np.median(array):.5g} · largest {largest:.5g}"
    if unit.startswith("voxel"):
        summary += "\nMesh geometry in voxel coordinates; not a foreground voxel count."
    upper = largest if largest > 0 else 1.0
    bins = int(np.clip(np.ceil(np.sqrt(array.size)) * 2, 8, 64))
    if log:
        counts, edges = np.histogram(
            np.log1p(array), bins=bins, range=(0.0, np.log1p(upper))
        )
        edges = np.expm1(edges)
    else:
        counts, edges = np.histogram(array, bins=bins, range=(0.0, upper))
    # log1p/expm1 rounding must not advertise a final edge smaller than the
    # largest included measurement in the hover text.
    edges[0], edges[-1] = 0.0, upper
    return MeshFilterHistogram(label, summary, counts, edges, (0.0, upper), omitted)
