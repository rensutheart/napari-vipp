"""Pixel-free shape and grid projection for full-placement template scores."""

from dataclasses import replace

import numpy as np

from napari_vipp.core.detection import template_match_grid_contract
from napari_vipp.core.metadata import ImageState


def project_detection_outputs(pipeline, call):
    """Publish both centre-grid descriptors without calculating correlation.

    Numerical validity and revision evidence exist only after execution; this
    planner deliberately does not manufacture TemplateMatchMetadata or scores.
    """
    if call is None or call.operation_id != "template_match":
        return None
    if len(call.inputs) != 2 or len(call.input_states) != 2:
        raise ValueError("Template Match needs Search image and Template image.")
    if any(state is None for state in call.input_states):
        return ((None, None), (None, None))
    for value, state in zip(call.inputs, call.input_states, strict=True):
        if (
            not isinstance(state, ImageState)
            or tuple(getattr(value, "shape", ())) != state.shape
            or getattr(value, "dtype", None) is None
            or np.dtype(value.dtype).name != state.dtype
        ):
            raise ValueError("Template Match data and carried shape/dtype disagree.")
    search, template = call.input_states
    shape, axes = template_match_grid_contract(
        search.shape, template.shape, search, template
    )
    results = []
    for dtype, kind in (
        (np.dtype(np.float64), "template match score image"),
        (np.dtype(bool), "binary mask"),
    ):
        proxy = pipeline._axis_contract_proxy(shape, dtype)
        state = pipeline._axis_contract_replaced_state(
            search,
            shape,
            axes,
            dtype,
            operation_id="template_match",
            history_item="Metadata-only template centre grid; no scores calculated",
            channels=(),
        )
        results.append((proxy, replace(state, kind=kind, template_match_metadata=None)))
    return tuple(results)
