"""Detached, object-coloured buffers for a single napari Surface layer."""

import numpy as np


def mesh_surface_display(mesh):
    """Keep object boundaries sharp even when objects share a vertex index.

    Only presentation vertices are duplicated. The scientific geometry, IDs
    and measurements are unchanged, and one Surface handles the whole set.
    """
    pairs, inverse = np.unique(
        np.column_stack((np.repeat(mesh.face_object_ids, 3), mesh.faces.ravel())),
        axis=0,
        return_inverse=True,
    )
    objects = sorted(mesh.objects, key=lambda item: item.object_id)
    ids = np.array([item.object_id for item in objects], dtype=np.int64)
    colors = np.array([item.color for item in objects], dtype=np.float32)
    values = pairs[:, 0].copy()
    surface = (mesh.vertices[pairs[:, 1]].copy(), inverse.reshape((-1, 3)), values)
    return surface, colors[np.searchsorted(ids, values)].copy()
