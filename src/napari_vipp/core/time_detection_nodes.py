"""CPU-only graph adapter and declaration for explicit time-series detection."""

from dataclasses import replace
from types import MappingProxyType

TIME_DETECTION_RUNTIME_KEYWORDS = MappingProxyType(
    {"detect_spots_per_frame": frozenset({"input_states"})}
)


def detect_spots_per_frame_node(
    inputs, *, input_states, progress=None, progress_context=None, **params
):
    from napari_vipp.core.time_detection import detect_spots_per_frame

    if progress is not None and progress_context is not None:
        raise ValueError("Supply only one Detect Spots per Frame progress context.")
    mode = params.get("mode", "Local peaks")
    expected = 2 if mode == "Template match" else 1
    if len(inputs) != expected or len(input_states) != expected:
        raise ValueError(
            "Detect Spots per Frame requires one source series and, in Template match "
            "mode only, one fixed scalar template with carried states."
        )
    return detect_spots_per_frame(
        inputs[0],
        inputs[1] if expected == 2 else None,
        image_state=input_states[0],
        template_state=input_states[1] if expected == 2 else None,
        progress_context=progress if progress is not None else progress_context,
        **params,
    )


def time_detection_node_specs():
    from napari_vipp.core.detection_nodes import detection_node_specs
    from napari_vipp.core.pipeline import (
        IMAGE_DATA_CATEGORY,
        InputSpec,
        OperationSpec,
        ParameterSpec,
    )
    peak_parameters = tuple(
        replace(
            parameter,
            label="Maximum detections per frame",
        )
        if parameter.name == "maximum_detections"
        else parameter
        for parameter in detection_node_specs()[1].parameters
        if parameter.name != "use_mask"
    )
    return (
        OperationSpec(
            "detect_spots_per_frame",
            "Detect Spots per Frame",
            IMAGE_DATA_CATEGORY,
            "image",
            "table",
            (
                ParameterSpec(
                    "mode",
                    "Detection mode",
                    "choice",
                    "Local peaks",
                    0,
                    0,
                    1,
                    choices=("Local peaks", "Template match"),
                    tooltip=(
                        "Detect intensity peaks independently in each frame, or "
                        "match one fixed scalar template at its supplied size and "
                        "orientation before finding peaks."
                    ),
                ),
                *peak_parameters,
            ),
            detect_spots_per_frame_node,
            max_inputs=2,
            inputs=(
                InputSpec("image", "image", "Time series (TYX / TZYX)"),
                InputSpec("template", "image", "Fixed template (optional)"),
            ),
            subcategory="Detection",
            execution_policy="manual",
            stack_processing_note=(
                "Explicit scalar TYX or TZYX; select one channel upstream. "
                "CPU detection processes one frame at a time and returns a table, "
                "not a response stack. IDs are frame-local. Every frame, including "
                "empty and capped frames, retains exact population evidence. "
                "Use Build Tracks to estimate persistent identities."
            ),
        ),
    )
