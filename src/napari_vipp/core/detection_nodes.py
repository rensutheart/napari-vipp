"""Qt-free graph declarations and runtime-state adapters for detection."""

from types import MappingProxyType

DETECTION_RUNTIME_KEYWORDS = MappingProxyType(
    {
        operation: frozenset({"input_states"})
        for operation in ("template_match", "find_peaks")
    }
)


def template_match_node(inputs, *, input_states, progress=None):
    from napari_vipp.core.detection import template_match_arrays

    if len(inputs) != 2 or len(input_states) != 2:
        raise ValueError(
            "Template Match requires Search and Template inputs with states."
        )
    return template_match_arrays(
        inputs[0],
        inputs[1],
        search_state=input_states[0],
        template_state=input_states[1],
        progress_context=progress,
    )


def find_peaks_node(inputs, *, input_states, use_mask=False, progress=None, **params):
    from napari_vipp.core.detection import find_peaks

    if (
        not isinstance(use_mask, bool)
        or len(inputs) != (2 if use_mask else 1)
        or len(input_states) != len(inputs)
    ):
        raise ValueError(
            "Connect Image and, when Use valid mask is enabled, its Boolean Valid mask."
        )
    return find_peaks(
        inputs[0],
        inputs[1] if use_mask else None,
        image_state=input_states[0],
        mask_state=input_states[1] if use_mask else None,
        progress_context=progress,
        **params,
    )


def detection_node_specs():
    from napari_vipp.core.pipeline import (
        IMAGE_DATA_CATEGORY,
        InputSpec,
        OperationSpec,
        OutputSpec,
        ParameterSpec,
    )

    return (
        OperationSpec(
            "template_match",
            "Template Match",
            IMAGE_DATA_CATEGORY,
            "image",
            "image",
            (),
            template_match_node,
            max_inputs=2,
            inputs=(
                InputSpec("search", "image", "Search image"),
                InputSpec("template", "image", "Template image"),
            ),
            outputs=(
                OutputSpec("scores", "image", "Match scores"),
                OutputSpec("valid_mask", "mask", "Valid scores"),
            ),
            subcategory="Detection",
            execution_policy="manual",
            stack_processing_note=(
                "Explicit scalar YX or ZYX only; select T/C upstream. "
                "CPU signed normalized correlation at supplied size/orientation, "
                "complete placements only. Scores are resemblance, not probability. "
                "Connect both outputs to Find Peaks."
            ),
        ),
        OperationSpec(
            "find_peaks",
            "Find Peaks",
            IMAGE_DATA_CATEGORY,
            "image",
            "table",
            (
                ParameterSpec(
                    "use_mask",
                    "Use valid mask",
                    "bool",
                    False,
                    0,
                    1,
                    1,
                    tooltip=(
                        "A Boolean same-grid mask excludes samples from candidates "
                        "and their local neighborhoods. Required for Template Match "
                        "scores; connect the matching Valid scores output."
                    ),
                ),
                ParameterSpec(
                    "minimum_value",
                    "Minimum value",
                    "float",
                    0.5,
                    -1e12,
                    1e12,
                    0.05,
                    6,
                    slider_minimum=-1.0,
                    slider_maximum=1.0,
                    tooltip=(
                        "Inclusive threshold. Template scores are signed resemblance, "
                        "not probability; a positive value excludes "
                        "contrast-inverted matches."
                    ),
                ),
                ParameterSpec(
                    "minimum_separation",
                    "Minimum separation",
                    "float",
                    1.0,
                    0.0,
                    1e12,
                    1.0,
                    6,
                    slider_minimum=0.0,
                    slider_maximum=100.0,
                    tooltip=(
                        "Greedy Euclidean spacing: reject distance strictly below "
                        "this value. Local maxima use a full 3-by-3 (or 3-by-3-by-3) "
                        "neighborhood; flat plateaus keep their first coordinate."
                    ),
                ),
                ParameterSpec(
                    "separation_units",
                    "Separation units",
                    "choice",
                    "Pixels",
                    0,
                    0,
                    1,
                    choices=("Pixels", "Physical (micrometers)"),
                    tooltip=(
                        "Physical separation uses all spatial-axis scales, "
                        "including anisotropic Z. Requires known compatible "
                        "length units; converts them to micrometers."
                    ),
                ),
                ParameterSpec(
                    "maximum_detections",
                    "Maximum detections",
                    "int",
                    1000,
                    1,
                    1000000,
                    1,
                    slider_minimum=1,
                    slider_maximum=10000,
                    tooltip=(
                        "Apply the cap after all separation decisions. Exact "
                        "pre-cap count and truncation are retained, "
                        "including for empty tables."
                    ),
                ),
                ParameterSpec(
                    "border_exclusion",
                    "Border exclusion (score-grid pixels)",
                    "int",
                    0,
                    0,
                    1000000,
                    1,
                    slider_minimum=0,
                    slider_maximum=100,
                    tooltip=(
                        "Exclude this many samples at every score-image boundary. "
                        "Zero permits complete-template border placements; this "
                        "is not a physical-distance setting."
                    ),
                ),
            ),
            find_peaks_node,
            max_inputs=2,
            inputs=(
                InputSpec("image", "image", "Score / intensity image"),
                InputSpec("valid_mask", "mask", "Valid mask"),
            ),
            subcategory="Detection",
            execution_policy="manual",
            stack_processing_note=(
                "Explicit scalar YX or ZYX local maxima, deterministic "
                "score/coordinate ordering. Table-first source-centre coordinates "
                "with non-editing inspection; no segmentation labels or editable "
                "points output."
            ),
        ),
    )
